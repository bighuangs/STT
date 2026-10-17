"""不联网、不执行文档内容的本地多格式解析器。

Office Open XML 使用标准库 ``zipfile``/``ElementTree`` 读取，所有成员先经过路径、
数量、解压体积、压缩比、加密位与 XML 主动内容检查。宏、OLE 嵌入、外部关系只产生
稳定告警，从不加载或执行。
"""

from __future__ import annotations

import csv
import hashlib
import mimetypes
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

import chardet

from app.domain.parsing import (
    DocumentParseError,
    ParsedAsset,
    ParsedBlock,
    ParsedDocument,
    ParseWarning,
    SourceLocation,
)
from app.rag.import_.parsing.base import ParseRequest, ParserPort


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"

SAFE_RASTER_SIGNATURES = {
    "image/png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "image/gif": lambda data: data.startswith((b"GIF87a", b"GIF89a")),
    "image/bmp": lambda data: data.startswith(b"BM"),
    "image/tiff": lambda data: data.startswith((b"II*\x00", b"MM\x00*")),
    "image/webp": lambda data: data.startswith(b"RIFF") and data[8:12] == b"WEBP",
}


def _q(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _Builder:
    def __init__(self, request: ParseRequest) -> None:
        self.request = request
        self.metadata: dict[str, Any] = {"title": request.path.stem, "source": request.path.name}
        self.blocks: list[ParsedBlock] = []
        self.assets: list[ParsedAsset] = []
        self.warnings: list[ParseWarning] = []
        self._warning_keys: set[tuple[str, str]] = set()
        self._headings: list[str] = []
        self._asset_by_hash: dict[str, str] = {}

    def warn(
        self,
        code: str,
        message: str,
        *,
        location: SourceLocation | None = None,
        severity: str = "warning",
    ) -> None:
        location_key = location.model_dump_json() if location else ""
        key = (code, location_key)
        if key in self._warning_keys or len(self.warnings) >= 1_000:
            return
        self._warning_keys.add(key)
        self.warnings.append(
            ParseWarning(code=code, severity=severity, message=message, location=location)
        )

    def add(
        self,
        block_type: str,
        text: str = "",
        *,
        structured_data: dict[str, Any] | list[Any] | None = None,
        location: SourceLocation | None = None,
        asset_refs: list[str] | None = None,
        heading_level: int | None = None,
    ) -> None:
        text = text.strip()
        if not text and structured_data is None and not asset_refs:
            return
        if len(self.blocks) >= self.request.limits.max_blocks:
            raise DocumentParseError("PARSE_TOO_MANY_BLOCKS", "文档结构块数量超过解析上限。")
        parent_path = list(self._headings)
        self.blocks.append(
            ParsedBlock(
                block_id=f"block-{len(self.blocks):08d}",
                type=block_type,
                text=text,
                structured_data=structured_data,
                location=location or SourceLocation(),
                parent_heading_path=parent_path,
                order=len(self.blocks),
                asset_refs=asset_refs or [],
            )
        )
        if block_type == "heading":
            level = min(max(heading_level or 1, 1), 6)
            self._headings = self._headings[: level - 1]
            self._headings.append(text)

    def add_asset(self, data: bytes, source_name: str, media_type: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        existing = self._asset_by_hash.get(digest)
        if existing:
            return existing
        asset_id = f"asset-{digest[:24]}"
        suffix = Path(source_name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = mimetypes.guess_extension(media_type) or ".bin"
        relative = Path("assets") / f"{digest}{suffix}"
        destination = self.request.work_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        self.assets.append(
            ParsedAsset(
                asset_id=asset_id,
                object_key=relative.as_posix(),
                sha256=digest,
                media_type=media_type,
                size_bytes=len(data),
            )
        )
        self._asset_by_hash[digest] = asset_id
        return asset_id

    def finish(self) -> ParsedDocument:
        return ParsedDocument(
            parser_id=self.request.detection.parser_id,
            parser_version=self.request.detection.parser_version,
            source_sha256=self.request.source_sha256,
            source_format=self.request.detection.format,
            source_mime_type=self.request.detection.mime_type,
            metadata=self.metadata,
            blocks=self.blocks,
            assets=self.assets,
            warnings=self.warnings,
        )


def _safe_raster_mime(data: bytes) -> str | None:
    return next((mime for mime, validator in SAFE_RASTER_SIGNATURES.items() if validator(data)), None)


def _load_local_raster_asset(
    builder: _Builder,
    source: str,
    *,
    location: SourceLocation | None = None,
) -> str | None:
    """只读取与源文档同目录树内、签名可信的相对路径光栅图。"""

    parsed = urlsplit(source)
    if parsed.scheme or parsed.netloc:
        builder.warn(
            "PARSE_EXTERNAL_RESOURCE_SKIPPED",
            "图片引用指向外部资源，解析器不会联网或解码 data URI。",
            location=location,
        )
        return None
    raw_path = unquote(parsed.path).strip()
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute():
        builder.warn(
            "PARSE_LOCAL_ASSET_OUTSIDE_ROOT",
            "图片必须使用源文档目录内的相对路径，当前引用已跳过。",
            location=location,
        )
        return None
    source_root = builder.request.path.parent.resolve()
    try:
        candidate = (source_root / relative).resolve(strict=True)
        candidate.relative_to(source_root)
    except (OSError, ValueError):
        builder.warn(
            "PARSE_LOCAL_ASSET_UNAVAILABLE",
            "本地图片不存在、不可读取或越出源文档目录，已跳过。",
            location=location,
        )
        return None
    if not candidate.is_file():
        builder.warn(
            "PARSE_LOCAL_ASSET_UNAVAILABLE", "本地图片引用不是普通文件，已跳过。", location=location
        )
        return None
    try:
        if candidate.stat().st_size > builder.request.limits.max_zip_entry_bytes:
            builder.warn(
                "PARSE_LOCAL_ASSET_TOO_LARGE", "本地图片超过资产大小上限，已跳过。", location=location
            )
            return None
        data = candidate.read_bytes()
    except OSError:
        builder.warn(
            "PARSE_LOCAL_ASSET_UNAVAILABLE", "本地图片无法读取，已跳过。", location=location
        )
        return None
    media_type = _safe_raster_mime(data)
    if media_type is None:
        builder.warn(
            "PARSE_UNSAFE_IMAGE_SKIPPED",
            "本地图片不是签名可信的安全光栅格式，已跳过。",
            location=location,
        )
        return None
    return builder.add_asset(data, candidate.name, media_type)


def _read_text(request: ParseRequest, builder: _Builder) -> str:
    try:
        data = request.path.read_bytes()
    except OSError as exc:
        raise DocumentParseError("PARSE_FILE_UNREADABLE", "待解析文件无法读取。") from exc
    detected = chardet.detect(data[: min(len(data), 256 * 1024)])
    encoding = detected.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
    try:
        text = data.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError):
        encoding = "utf-8"
        text = data.decode(encoding, errors="replace")
        builder.warn("PARSE_TEXT_DECODE_REPLACED", "部分非法字节已用替代字符解码。")
    builder.metadata["encoding"] = encoding
    builder.metadata["language"] = "und"
    return text.lstrip("\ufeff")


def _split_markdown_row(line: str) -> list[str]:
    value = line.strip().strip("|")
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", value)]


def _markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


@dataclass(frozen=True, slots=True)
class MarkdownParser:
    parser_id: str = "local.markdown"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("markdown",)
    mime_types: tuple[str, ...] = ("text/markdown", "text/x-markdown")
    extensions: tuple[str, ...] = (".md", ".markdown", ".mdown")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        lines = _read_text(request, builder).splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            fence = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
            if fence:
                start = index + 1
                marker = fence.group(1)
                language = fence.group(2).strip()
                content: list[str] = []
                index += 1
                while index < len(lines) and not re.match(
                    rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$", lines[index]
                ):
                    content.append(lines[index])
                    index += 1
                if index >= len(lines):
                    builder.warn(
                        "PARSE_MARKDOWN_UNCLOSED_FENCE",
                        "代码围栏未闭合，已解析至文件结尾。",
                        location=SourceLocation(line_start=start),
                    )
                else:
                    index += 1
                builder.add(
                    "code",
                    "\n".join(content),
                    structured_data={"language": language or None},
                    location=SourceLocation(line_start=start, line_end=max(start, index)),
                )
                continue
            heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                builder.add(
                    "heading",
                    heading.group(2),
                    heading_level=len(heading.group(1)),
                    location=SourceLocation(line_start=index + 1, line_end=index + 1),
                )
                index += 1
                continue
            if index + 1 < len(lines) and "|" in line and _markdown_table_separator(lines[index + 1]):
                start = index + 1
                headers = _split_markdown_row(line)
                index += 2
                rows: list[list[str]] = []
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    rows.append(_split_markdown_row(lines[index]))
                    index += 1
                builder.add(
                    "table",
                    "\n".join("\t".join(row) for row in [headers, *rows]),
                    structured_data={"headers": headers, "rows": rows},
                    location=SourceLocation(line_start=start, line_end=index),
                )
                continue
            image = re.fullmatch(r"\s*!\[([^\]]*)\]\(([^\s\)]+)(?:\s+[^\)]*)?\)\s*", line)
            if image:
                source = image.group(2)
                location = SourceLocation(line_start=index + 1, line_end=index + 1)
                asset_id = _load_local_raster_asset(builder, source, location=location)
                builder.add(
                    "image",
                    image.group(1) or "图片",
                    structured_data={"source": source, "imported": asset_id is not None},
                    location=location,
                    asset_refs=[asset_id] if asset_id else None,
                )
                index += 1
                continue
            if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
                start = index + 1
                items: list[str] = []
                while index < len(lines):
                    match = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", lines[index])
                    if not match:
                        break
                    items.append(match.group(1).strip())
                    index += 1
                builder.add(
                    "list",
                    "\n".join(items),
                    structured_data={"items": items},
                    location=SourceLocation(line_start=start, line_end=index),
                )
                continue
            start = index + 1
            paragraph: list[str] = []
            while index < len(lines) and lines[index].strip():
                if paragraph and (
                    re.match(r"^\s{0,3}#{1,6}\s+", lines[index])
                    or re.match(r"^\s*(`{3,}|~{3,})", lines[index])
                ):
                    break
                paragraph.append(lines[index].strip())
                index += 1
            builder.add(
                "text",
                "\n".join(paragraph),
                location=SourceLocation(line_start=start, line_end=max(start, index)),
            )
        builder.metadata["line_count"] = len(lines)
        return builder.finish()


@dataclass(frozen=True, slots=True)
class TextParser:
    parser_id: str = "local.text"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("txt",)
    mime_types: tuple[str, ...] = ("text/plain",)
    extensions: tuple[str, ...] = (".txt", ".text", ".log")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        lines = _read_text(request, builder).splitlines()
        start: int | None = None
        paragraph: list[str] = []
        for number, line in enumerate([*lines, ""], start=1):
            if line.strip():
                if start is None:
                    start = number
                paragraph.append(line.rstrip())
                continue
            if paragraph:
                builder.add(
                    "text",
                    "\n".join(paragraph),
                    location=SourceLocation(line_start=start, line_end=number - 1),
                )
                start = None
                paragraph = []
        builder.metadata["line_count"] = len(lines)
        return builder.finish()


class _SafeHTMLParser(HTMLParser):
    BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "pre", "code", "li"}
    SKIP_TAGS = {"script", "style", "iframe", "object", "embed", "applet", "noscript"}

    def __init__(self, builder: _Builder) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = builder
        self.skip_depth = 0
        self.capture_tag: str | None = None
        self.capture_depth = 0
        self.buffer: list[str] = []
        self.capture_line = 1
        self.title_capture = False
        self.title_parts: list[str] = []
        self.table_depth = 0
        self.table_rows: list[list[str]] = []
        self.table_row: list[str] | None = None
        self.table_cell: list[str] | None = None
        self.table_start = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            self.builder.warn("PARSE_HTML_ACTIVE_CONTENT_REMOVED", f"已移除主动内容标签 <{tag}>。")
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.title_capture = True
            return
        if tag == "table":
            if not self.table_depth:
                self.table_rows = []
                self.table_start = self.getpos()[0]
            self.table_depth += 1
            return
        if self.table_depth:
            if tag == "tr":
                self.table_row = []
            elif tag in {"td", "th"}:
                self.table_cell = []
            elif tag == "br" and self.table_cell is not None:
                self.table_cell.append("\n")
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", *self.BLOCK_TAGS}:
            if self.capture_tag is None:
                self.capture_tag = tag
                self.capture_depth = 1
                self.capture_line = self.getpos()[0]
                self.buffer = []
            elif tag == self.capture_tag:
                self.capture_depth += 1
            return
        if tag == "br" and self.capture_tag:
            self.buffer.append("\n")
        if tag == "img":
            source = attributes.get("src", "")
            alt = attributes.get("alt", "") or attributes.get("title", "") or "图片"
            location = SourceLocation(line_start=self.getpos()[0])
            asset_id = _load_local_raster_asset(self.builder, source, location=location)
            self.builder.add(
                "image",
                alt,
                structured_data={"source": source, "imported": asset_id is not None},
                location=location,
                asset_refs=[asset_id] if asset_id else None,
            )
        if tag == "a" and attributes.get("href"):
            self.builder.warn(
                "PARSE_EXTERNAL_LINK_SKIPPED",
                "HTML 链接只保留显示文本，不访问目标地址。",
                location=SourceLocation(line_start=self.getpos()[0]),
            )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in {"img", "br", "input", "meta", "link", "hr"}:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.title_capture:
            self.title_parts.append(data)
        elif self.table_cell is not None:
            self.table_cell.append(data)
        elif self.capture_tag:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.title_capture = False
            title = " ".join("".join(self.title_parts).split())
            if title:
                self.builder.metadata["title"] = title
            return
        if self.table_depth:
            if tag in {"td", "th"} and self.table_cell is not None:
                value = " ".join("".join(self.table_cell).split())
                if self.table_row is not None:
                    self.table_row.append(value)
                self.table_cell = None
            elif tag == "tr" and self.table_row is not None:
                self.table_rows.append(self.table_row)
                self.table_row = None
            elif tag == "table":
                self.table_depth -= 1
                if not self.table_depth and self.table_rows:
                    headers = self.table_rows[0]
                    rows = self.table_rows[1:]
                    self.builder.add(
                        "table",
                        "\n".join("\t".join(row) for row in self.table_rows),
                        structured_data={"headers": headers, "rows": rows},
                        location=SourceLocation(
                            line_start=self.table_start, line_end=self.getpos()[0]
                        ),
                    )
            return
        if self.capture_tag and tag == self.capture_tag:
            self.capture_depth -= 1
            if self.capture_depth:
                return
            text = "\n".join(
                " ".join(part.split()) for part in "".join(self.buffer).splitlines() if part.strip()
            )
            if self.capture_tag.startswith("h") and self.capture_tag[1:].isdigit():
                block_type = "heading"
                level = int(self.capture_tag[1:])
            elif self.capture_tag in {"pre", "code"}:
                block_type = "code"
                level = None
            elif self.capture_tag == "li":
                block_type = "list"
                level = None
            else:
                block_type = "text"
                level = None
            self.builder.add(
                block_type,
                text,
                heading_level=level,
                location=SourceLocation(
                    line_start=self.capture_line, line_end=self.getpos()[0]
                ),
            )
            self.capture_tag = None
            self.capture_depth = 0
            self.buffer = []


@dataclass(frozen=True, slots=True)
class HTMLDocumentParser:
    parser_id: str = "local.html"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("html",)
    mime_types: tuple[str, ...] = ("text/html", "application/xhtml+xml")
    extensions: tuple[str, ...] = (".html", ".htm", ".xhtml")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        text = _read_text(request, builder)
        parser = _SafeHTMLParser(builder)
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:
            raise DocumentParseError("PARSE_MALFORMED_HTML", "HTML 内容无法安全解析。") from exc
        builder.metadata["line_count"] = text.count("\n") + 1
        return builder.finish()


@dataclass(frozen=True, slots=True)
class CSVDocumentParser:
    parser_id: str = "local.csv"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("csv",)
    mime_types: tuple[str, ...] = ("text/csv", "text/tab-separated-values")
    extensions: tuple[str, ...] = (".csv", ".tsv")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        text = _read_text(request, builder)
        sample = text[:65_536]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel_tab if request.path.suffix.lower() == ".tsv" else csv.excel
        old_limit = csv.field_size_limit()
        csv.field_size_limit(request.limits.max_cell_chars)
        rows: list[list[str]] = []
        formula_location: SourceLocation | None = None
        try:
            for row_number, row in enumerate(csv.reader(text.splitlines(), dialect), start=1):
                if row_number > request.limits.max_rows:
                    raise DocumentParseError("PARSE_TOO_MANY_ROWS", "CSV 行数超过解析上限。")
                if len(row) > request.limits.max_columns:
                    raise DocumentParseError("PARSE_TOO_MANY_COLUMNS", "CSV 列数超过解析上限。")
                if any(len(cell) > request.limits.max_cell_chars for cell in row):
                    raise DocumentParseError("PARSE_CELL_TOO_LARGE", "CSV 单元格超过解析上限。")
                if formula_location is None and any(
                    cell.lstrip().startswith(("=", "+", "-", "@")) for cell in row
                ):
                    formula_location = SourceLocation(line_start=row_number, line_end=row_number)
                rows.append(row)
        except csv.Error as exc:
            raise DocumentParseError("PARSE_MALFORMED_CSV", "CSV 内容格式错误。") from exc
        finally:
            csv.field_size_limit(old_limit)
        if formula_location:
            builder.warn(
                "PARSE_CSV_FORMULA_NOT_EXECUTED",
                "检测到可能被表格软件解释为公式的单元格；仅按纯文本保留。",
                location=formula_location,
            )
        if rows:
            builder.add(
                "table",
                "\n".join("\t".join(row) for row in rows),
                structured_data={"headers": rows[0], "rows": rows[1:], "delimiter": dialect.delimiter},
                location=SourceLocation(line_start=1, line_end=len(rows)),
            )
        builder.metadata.update({"row_count": len(rows), "column_count": max(map(len, rows), default=0)})
        return builder.finish()


class _SafeOfficePackage:
    ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}

    def __init__(self, request: ParseRequest, builder: _Builder) -> None:
        self.request = request
        self.builder = builder
        self.archive: zipfile.ZipFile | None = None
        self.entries: dict[str, zipfile.ZipInfo] = {}
        self.content_defaults: dict[str, str] = {}
        self.content_overrides: dict[str, str] = {}

    def __enter__(self) -> "_SafeOfficePackage":
        try:
            self.archive = zipfile.ZipFile(self.request.path)
            self._validate()
            self._load_content_types()
            return self
        except DocumentParseError:
            if self.archive:
                self.archive.close()
            raise
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            if self.archive:
                self.archive.close()
            raise DocumentParseError("PARSE_CORRUPT_ARCHIVE", "Office 文档压缩包已损坏。") from exc

    def __exit__(self, *_args: object) -> None:
        if self.archive:
            self.archive.close()

    @staticmethod
    def _normalise_name(name: str) -> str:
        value = name.replace("\\", "/")
        path = PurePosixPath(value)
        if value.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
            raise DocumentParseError(
                "PARSE_ZIP_PATH_TRAVERSAL", "Office 文档包含不安全的压缩包路径。"
            )
        if path.parts and ":" in path.parts[0]:
            raise DocumentParseError(
                "PARSE_ZIP_PATH_TRAVERSAL", "Office 文档包含不安全的压缩包路径。"
            )
        return path.as_posix()

    def _validate(self) -> None:
        assert self.archive is not None
        infos = self.archive.infolist()
        limits = self.request.limits
        if len(infos) > limits.max_zip_entries:
            raise DocumentParseError("PARSE_ZIP_TOO_MANY_ENTRIES", "Office 文档条目数超过安全上限。")
        total = 0
        seen: set[str] = set()
        for info in infos:
            name = self._normalise_name(info.filename)
            key = name.casefold()
            if key in seen:
                raise DocumentParseError(
                    "PARSE_ZIP_DUPLICATE_ENTRY", "Office 文档包含重复压缩条目。"
                )
            seen.add(key)
            if info.flag_bits & 0x1:
                raise DocumentParseError(
                    "PARSE_ENCRYPTED_DOCUMENT", "加密 Office 文档不支持解析。", retryable=False
                )
            if info.compress_type not in self.ALLOWED_COMPRESSION:
                raise DocumentParseError(
                    "PARSE_ZIP_UNSUPPORTED_COMPRESSION", "Office 文档使用了不受支持的压缩算法。"
                )
            if info.file_size > limits.max_zip_entry_bytes:
                raise DocumentParseError("PARSE_ZIP_ENTRY_TOO_LARGE", "Office 文档单个条目过大。")
            total += info.file_size
            if total > limits.max_zip_uncompressed_bytes:
                raise DocumentParseError("PARSE_ZIP_TOO_LARGE", "Office 文档解压总体积超过安全上限。")
            if info.file_size:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > limits.max_compression_ratio:
                    raise DocumentParseError("PARSE_ZIP_BOMB", "Office 文档压缩比超过安全上限。")
            self.entries[name] = info

        lowered = {name.lower() for name in self.entries}
        if any(name.endswith("vbaproject.bin") for name in lowered):
            self.builder.warn("PARSE_MACRO_SKIPPED", "检测到 Office 宏，解析器不会读取或执行宏。")
        if any("/embeddings/" in name or "oleobject" in name for name in lowered):
            self.builder.warn(
                "PARSE_EMBEDDED_OBJECT_SKIPPED", "检测到 OLE/嵌入对象，解析器已跳过。"
            )
        if any("/externallinks/" in name for name in lowered):
            self.builder.warn(
                "PARSE_EXTERNAL_LINK_SKIPPED", "检测到 Office 外部链接，解析器不会访问。"
            )

    def read(self, name: str, *, required: bool = True, max_bytes: int | None = None) -> bytes:
        normalised = self._normalise_name(name.lstrip("/"))
        info = self.entries.get(normalised)
        if info is None:
            if required:
                raise DocumentParseError(
                    "PARSE_CORRUPT_OFFICE", f"Office 文档缺少必要条目：{normalised}。"
                )
            return b""
        limit = max_bytes or self.request.limits.max_zip_entry_bytes
        if info.file_size > limit:
            raise DocumentParseError("PARSE_ZIP_ENTRY_TOO_LARGE", "Office 文档条目超过读取上限。")
        assert self.archive is not None
        with self.archive.open(info, "r") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise DocumentParseError("PARSE_ZIP_ENTRY_TOO_LARGE", "Office 文档条目超过读取上限。")
        return data

    def xml(self, name: str, *, required: bool = True) -> ET.Element | None:
        data = self.read(name, required=required, max_bytes=self.request.limits.max_xml_bytes)
        if not data and not required:
            return None
        lowered = data[: min(len(data), 1_000_000)].lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered:
            raise DocumentParseError(
                "PARSE_XML_ACTIVE_CONTENT", "Office XML 包含 DTD/实体声明，已拒绝解析。"
            )
        try:
            return ET.fromstring(data)
        except ET.ParseError as exc:
            raise DocumentParseError("PARSE_CORRUPT_XML", "Office XML 内容已损坏。") from exc

    def relationships(self, source_part: str) -> dict[str, str]:
        directory = posixpath.dirname(source_part)
        filename = posixpath.basename(source_part)
        rel_name = posixpath.join(directory, "_rels", f"{filename}.rels")
        root = self.xml(rel_name, required=False)
        result: dict[str, str] = {}
        if root is None:
            return result
        for relation in root.iter():
            if _local(relation.tag) != "Relationship":
                continue
            relation_id = relation.attrib.get("Id", "")
            target = relation.attrib.get("Target", "")
            if relation.attrib.get("TargetMode", "").lower() == "external":
                self.builder.warn(
                    "PARSE_EXTERNAL_LINK_SKIPPED", "Office 外部关系已跳过，不会访问目标地址。"
                )
                continue
            joined = posixpath.normpath(posixpath.join(directory, target.lstrip("/")))
            if joined.startswith("../") or joined == "..":
                raise DocumentParseError(
                    "PARSE_ZIP_PATH_TRAVERSAL", "Office 关系目标逃逸出文档包。"
                )
            if relation_id:
                result[relation_id] = joined
        return result

    def _load_content_types(self) -> None:
        root = self.xml("[Content_Types].xml", required=False)
        if root is None:
            return
        for node in root:
            local = _local(node.tag)
            if local == "Default":
                self.content_defaults[node.attrib.get("Extension", "").lower()] = node.attrib.get(
                    "ContentType", "application/octet-stream"
                )
            elif local == "Override":
                self.content_overrides[node.attrib.get("PartName", "").lstrip("/")] = node.attrib.get(
                    "ContentType", "application/octet-stream"
                )

    def media_type(self, name: str) -> str:
        return self.content_overrides.get(
            name,
            self.content_defaults.get(
                Path(name).suffix.lstrip(".").lower(),
                mimetypes.guess_type(name)[0] or "application/octet-stream",
            ),
        )

    def extract_image(self, name: str) -> str | None:
        declared_media_type = self.media_type(name)
        data = self.read(name)
        media_type = _safe_raster_mime(data)
        if media_type is None:
            self.builder.warn(
                "PARSE_UNSAFE_IMAGE_SKIPPED", "非受信任光栅格式的嵌入对象已跳过。"
            )
            return None
        if declared_media_type != media_type:
            self.builder.warn(
                "PARSE_IMAGE_SIGNATURE_MISMATCH",
                "图片声明类型与实际签名不匹配，已按实际安全光栅类型保存。",
            )
        return self.builder.add_asset(data, name, media_type)


def _paragraph_text(node: ET.Element, text_namespace: str = W) -> str:
    parts: list[str] = []
    for child in node.iter():
        local = _local(child.tag)
        if local == "t":
            parts.append(child.text or "")
        elif local == "tab":
            parts.append("\t")
        elif local in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _core_metadata(package: _SafeOfficePackage, builder: _Builder) -> None:
    root = package.xml("docProps/core.xml", required=False)
    if root is not None:
        mapping = {
            _q(DC, "title"): "title",
            _q(DC, "creator"): "author",
            _q(DC, "subject"): "subject",
            _q(DC, "language"): "language",
            _q(CP, "lastModifiedBy"): "last_modified_by",
            _q(DCTERMS, "created"): "created_at",
            _q(DCTERMS, "modified"): "modified_at",
        }
        for child in root:
            key = mapping.get(child.tag)
            if key and (child.text or "").strip():
                builder.metadata[key] = child.text.strip()


@dataclass(frozen=True, slots=True)
class DocxParser:
    parser_id: str = "local.docx"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("docx",)
    mime_types: tuple[str, ...] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    extensions: tuple[str, ...] = (".docx",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        with _SafeOfficePackage(request, builder) as package:
            root = package.xml("word/document.xml")
            assert root is not None
            _core_metadata(package, builder)
            relations = package.relationships("word/document.xml")
            body = root.find(_q(W, "body"))
            if body is None:
                raise DocumentParseError("PARSE_CORRUPT_OFFICE", "DOCX 缺少文档正文。")
            for child in body:
                local = _local(child.tag)
                if local == "p":
                    text = _paragraph_text(child)
                    style_node = child.find(f".//{_q(W, 'pStyle')}")
                    style = style_node.attrib.get(_q(W, "val"), "") if style_node is not None else ""
                    heading_match = re.search(r"(?:heading|标题)\s*([1-6])", style, re.I)
                    is_list = child.find(f".//{_q(W, 'numPr')}") is not None
                    if heading_match:
                        builder.add("heading", text, heading_level=int(heading_match.group(1)))
                    elif is_list:
                        builder.add("list", text, structured_data={"items": [text]})
                    else:
                        builder.add("text", text)
                elif local == "tbl":
                    rows: list[list[str]] = []
                    for row in child.findall(f"./{_q(W, 'tr')}"):
                        cells = [
                            "\n".join(
                                filter(None, (_paragraph_text(p) for p in cell.findall(f".//{_q(W, 'p')}")))
                            )
                            for cell in row.findall(f"./{_q(W, 'tc')}")
                        ]
                        rows.append(cells)
                    if rows:
                        builder.add(
                            "table",
                            "\n".join("\t".join(row) for row in rows),
                            structured_data={"headers": rows[0], "rows": rows[1:]},
                        )
                for blip in child.findall(f".//{_q(A, 'blip')}"):
                    relation_id = blip.attrib.get(_q(R, "embed"))
                    target = relations.get(relation_id or "")
                    if target and target in package.entries:
                        asset_id = package.extract_image(target)
                        if asset_id:
                            builder.add("image", "文档内图片", asset_refs=[asset_id])
            app_root = package.xml("docProps/app.xml", required=False)
            if app_root is not None:
                for node in app_root:
                    if _local(node.tag) == "Pages" and (node.text or "").isdigit():
                        builder.metadata["page_count"] = int(node.text or "0")
        return builder.finish()


def _drawing_paragraphs(node: ET.Element) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in node.findall(f".//{_q(A, 'p')}"):
        text = "".join(part.text or "" for part in paragraph.findall(f".//{_q(A, 't')}"))
        if text.strip():
            paragraphs.append(text.strip())
    return paragraphs


@dataclass(frozen=True, slots=True)
class PptxParser:
    parser_id: str = "local.pptx"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("pptx",)
    mime_types: tuple[str, ...] = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    extensions: tuple[str, ...] = (".pptx",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        with _SafeOfficePackage(request, builder) as package:
            presentation = package.xml("ppt/presentation.xml")
            assert presentation is not None
            _core_metadata(package, builder)
            presentation_relations = package.relationships("ppt/presentation.xml")
            slide_ids = presentation.findall(f".//{_q(P, 'sldId')}")
            for slide_no, slide_id in enumerate(slide_ids, start=1):
                target = presentation_relations.get(slide_id.attrib.get(_q(R, "id"), ""))
                if not target:
                    builder.warn(
                        "PARSE_MISSING_RELATIONSHIP",
                        "演示文稿中存在缺失的幻灯片关系。",
                        location=SourceLocation(slide_no=slide_no),
                    )
                    continue
                slide = package.xml(target)
                assert slide is not None
                relations = package.relationships(target)
                shape_tree = slide.find(f".//{_q(P, 'spTree')}")
                if shape_tree is None:
                    continue
                for shape in shape_tree:
                    local = _local(shape.tag)
                    location = SourceLocation(slide_no=slide_no)
                    if local == "sp":
                        paragraphs = _drawing_paragraphs(shape)
                        text = "\n".join(paragraphs)
                        placeholder = shape.find(f".//{_q(P, 'ph')}")
                        placeholder_type = (
                            placeholder.attrib.get("type", "") if placeholder is not None else ""
                        )
                        if placeholder_type in {"title", "ctrTitle"}:
                            builder.add("heading", text, heading_level=1, location=location)
                        else:
                            builder.add("text", text, location=location)
                    table = shape.find(f".//{_q(A, 'tbl')}")
                    if table is not None:
                        rows: list[list[str]] = []
                        for row in table.findall(f"./{_q(A, 'tr')}"):
                            rows.append(
                                [
                                    "\n".join(_drawing_paragraphs(cell))
                                    for cell in row.findall(f"./{_q(A, 'tc')}")
                                ]
                            )
                        if rows:
                            builder.add(
                                "table",
                                "\n".join("\t".join(row) for row in rows),
                                structured_data={"headers": rows[0], "rows": rows[1:]},
                                location=location,
                            )
                    for blip in shape.findall(f".//{_q(A, 'blip')}"):
                        relation_id = blip.attrib.get(_q(R, "embed"))
                        image_target = relations.get(relation_id or "")
                        if image_target and image_target in package.entries:
                            asset_id = package.extract_image(image_target)
                            if asset_id:
                                builder.add(
                                    "image", "幻灯片内图片", location=location, asset_refs=[asset_id]
                                )
            builder.metadata.update(
                {"slide_count": len(slide_ids), "page_count": len(slide_ids)}
            )
        return builder.finish()


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> tuple[str, str | None]:
    cell_type = cell.attrib.get("t", "")
    formula_node = cell.find(_q(S, "f"))
    formula = formula_node.text if formula_node is not None else None
    if cell_type == "inlineStr":
        value = "".join(node.text or "" for node in cell.findall(f".//{_q(S, 't')}"))
    else:
        value_node = cell.find(_q(S, "v"))
        value = value_node.text or "" if value_node is not None else ""
        if cell_type == "s" and value.isdigit():
            index = int(value)
            value = shared[index] if index < len(shared) else ""
        elif cell_type == "b":
            value = "TRUE" if value == "1" else "FALSE"
    if formula and not value:
        value = f"={formula}"
    return value, formula


def _column_number(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference)
    if not letters:
        return 0
    number = 0
    for char in letters.group(0).upper():
        number = number * 26 + ord(char) - 64
    return number


@dataclass(frozen=True, slots=True)
class XlsxParser:
    parser_id: str = "local.xlsx"
    parser_version: str = "1.0.0"
    formats: tuple[str, ...] = ("xlsx",)
    mime_types: tuple[str, ...] = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    extensions: tuple[str, ...] = (".xlsx",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        builder = _Builder(request)
        with _SafeOfficePackage(request, builder) as package:
            workbook = package.xml("xl/workbook.xml")
            assert workbook is not None
            _core_metadata(package, builder)
            relations = package.relationships("xl/workbook.xml")
            shared: list[str] = []
            shared_root = package.xml("xl/sharedStrings.xml", required=False)
            if shared_root is not None:
                for item in shared_root.findall(f".//{_q(S, 'si')}"):
                    value = "".join(node.text or "" for node in item.findall(f".//{_q(S, 't')}"))
                    if len(value) > request.limits.max_cell_chars:
                        raise DocumentParseError("PARSE_CELL_TOO_LARGE", "共享字符串超过解析上限。")
                    shared.append(value)

            sheets = workbook.findall(f".//{_q(S, 'sheet')}")
            for sheet in sheets:
                sheet_name = sheet.attrib.get("name", "未命名工作表")
                target = relations.get(sheet.attrib.get(_q(R, "id"), ""))
                if not target:
                    builder.warn("PARSE_MISSING_RELATIONSHIP", f"工作表 {sheet_name} 缺少关系。")
                    continue
                root = package.xml(target)
                assert root is not None
                sheet_relations = package.relationships(target)
                matrix: list[list[str]] = []
                formulas: dict[str, str] = {}
                min_ref: str | None = None
                max_ref: str | None = None
                for row_index, row in enumerate(root.findall(f".//{_q(S, 'row')}"), start=1):
                    if row_index > request.limits.max_rows:
                        raise DocumentParseError("PARSE_TOO_MANY_ROWS", "工作表行数超过解析上限。")
                    values: list[str] = []
                    for cell in row.findall(f"./{_q(S, 'c')}"):
                        reference = cell.attrib.get("r", "")
                        column = _column_number(reference) or len(values) + 1
                        if column > request.limits.max_columns:
                            raise DocumentParseError(
                                "PARSE_TOO_MANY_COLUMNS", "工作表列数超过解析上限。"
                            )
                        while len(values) < column - 1:
                            values.append("")
                        value, formula = _xlsx_cell_value(cell, shared)
                        if len(value) > request.limits.max_cell_chars:
                            raise DocumentParseError("PARSE_CELL_TOO_LARGE", "单元格超过解析上限。")
                        values.append(value)
                        if formula:
                            formulas[reference] = formula
                            if len(formulas) == 1:
                                builder.warn(
                                    "PARSE_FORMULA_NOT_EVALUATED",
                                    "工作表公式不会执行，仅保留公式与已有缓存值。",
                                    location=SourceLocation(sheet=sheet_name, cell_range=reference),
                                )
                        if reference:
                            min_ref = min_ref or reference
                            max_ref = reference
                    matrix.append(values)
                if matrix:
                    cell_range = f"{min_ref}:{max_ref}" if min_ref and max_ref else None
                    builder.add(
                        "table",
                        "\n".join("\t".join(row) for row in matrix),
                        structured_data={
                            "sheet": sheet_name,
                            "headers": matrix[0],
                            "rows": matrix[1:],
                            "formulas": formulas,
                        },
                        location=SourceLocation(sheet=sheet_name, cell_range=cell_range),
                    )
                for drawing in root.findall(f".//{_q(S, 'drawing')}"):
                    drawing_target = sheet_relations.get(drawing.attrib.get(_q(R, "id"), ""))
                    if not drawing_target:
                        continue
                    drawing_root = package.xml(drawing_target, required=False)
                    if drawing_root is None:
                        continue
                    drawing_relations = package.relationships(drawing_target)
                    for blip in drawing_root.findall(f".//{_q(A, 'blip')}"):
                        image_target = drawing_relations.get(blip.attrib.get(_q(R, "embed"), ""))
                        if image_target and image_target in package.entries:
                            asset_id = package.extract_image(image_target)
                            if asset_id:
                                builder.add(
                                    "image",
                                    f"工作表 {sheet_name} 内图片",
                                    location=SourceLocation(sheet=sheet_name),
                                    asset_refs=[asset_id],
                                )
            builder.metadata.update({"sheet_count": len(sheets), "sheets": [s.attrib.get("name", "") for s in sheets]})
        return builder.finish()


def build_default_local_parsers() -> tuple[ParserPort, ...]:
    return (
        MarkdownParser(),
        TextParser(),
        HTMLDocumentParser(),
        CSVDocumentParser(),
        DocxParser(),
        PptxParser(),
        XlsxParser(),
    )


__all__ = [
    "CSVDocumentParser",
    "DocxParser",
    "HTMLDocumentParser",
    "MarkdownParser",
    "PptxParser",
    "TextParser",
    "XlsxParser",
    "build_default_local_parsers",
]
