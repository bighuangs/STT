"""解析器注册表与基于签名、MIME、扩展名的多证据文件探测。"""

from __future__ import annotations

import csv
import hashlib
import mimetypes
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

import chardet

from app.domain.parsing import DocumentParseError, ParsedDocument, ParseWarning
from app.rag.import_.parsing.base import DetectionResult, ParseRequest, ParserLimits, ParserPort


FORMAT_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "markdown": "text/markdown",
    "html": "text/html",
    "csv": "text/csv",
    "txt": "text/plain",
}

EXTENSION_FORMAT: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "txt",
    ".text": "txt",
    ".log": "txt",
}

MIME_FORMAT: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/csv": "csv",
    "application/csv": "csv",
    "text/tab-separated-values": "csv",
    "text/plain": "txt",
}


def _warning(code: str, message: str) -> ParseWarning:
    return ParseWarning(code=code, severity="warning", message=message)


def _normalise_mime(value: str | None) -> str | None:
    if not value:
        return None
    result = value.split(";", 1)[0].strip().lower()
    return result or None


def _decode_sample(data: bytes) -> str:
    detection = chardet.detect(data)
    encoding = detection.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
    try:
        return data.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def _zip_format(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 10_000:
                raise DocumentParseError(
                    "PARSE_ZIP_TOO_MANY_ENTRIES",
                    "压缩文档条目数超过安全上限。",
                )
            names = {info.filename.replace("\\", "/").lstrip("/") for info in infos}
    except DocumentParseError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise DocumentParseError(
            "PARSE_CORRUPT_ARCHIVE",
            "文件具有 ZIP 签名但不是可读取的 Office 文档。",
        ) from exc
    if "word/document.xml" in names:
        return "docx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    return None


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if sample.startswith((b"\xff\xfe", b"\xfe\xff", b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return False
    if b"\x00" in sample:
        return True
    control = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return control / len(sample) > 0.05


def _text_format(text: str, extension_format: str | None, mime_format: str | None) -> str:
    prefix = text[:32_768]
    if re.search(r"<!doctype\s+html|<html(?:\s|>)|<(?:head|body)(?:\s|>)", prefix, re.I):
        return "html"
    if extension_format == "html" or mime_format == "html":
        return "html"

    nonblank = [line for line in prefix.splitlines() if line.strip()]
    markdown_score = sum(
        bool(
            re.match(r"^\s{0,3}(?:#{1,6}\s+|```|~~~|[-*+]\s+|\d+[.)]\s+|>\s+)", line)
            or re.search(r"!?\[[^\]]+\]\([^\)]+\)", line)
        )
        for line in nonblank[:200]
    )
    if extension_format == "markdown" or mime_format == "markdown" or markdown_score >= 2:
        return "markdown"

    if extension_format == "csv" or mime_format == "csv":
        return "csv"
    registered_hint = extension_format or mime_format
    if registered_hint and registered_hint not in {"txt", "markdown", "html", "csv"}:
        return registered_hint
    try:
        sample = "\n".join(nonblank[:30])
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        rows = list(csv.reader(sample.splitlines(), dialect))
        widths = [len(row) for row in rows if row]
        if len(widths) >= 2 and min(widths) >= 2 and len(set(widths)) <= 2:
            return "csv"
    except csv.Error:
        pass
    return "txt"


class ParserRegistry:
    """格式到解析器的运行时注册表；增加格式不需要修改导入图拓扑。"""

    def __init__(self) -> None:
        self._parsers: dict[str, ParserPort] = {}
        self._extension_formats = dict(EXTENSION_FORMAT)
        self._mime_formats = dict(MIME_FORMAT)
        self._format_mimes = dict(FORMAT_MIME)

    def register(self, parser: ParserPort, *, replace: bool = False) -> None:
        if not isinstance(parser, ParserPort):
            raise TypeError("parser does not implement ParserPort")
        format_keys = [format_name.strip().lower() for format_name in parser.formats]
        if not format_keys or any(not key for key in format_keys):
            raise ValueError("parser format must not be empty")
        for key in format_keys:
            if key in self._parsers and not replace:
                raise ValueError(f"parser already registered for format: {key}")
        if len(parser.formats) == 1:
            format_name = format_keys[0]
            for extension in parser.extensions:
                key = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
                current = self._extension_formats.get(key)
                if current not in {None, format_name} and not replace:
                    raise ValueError(f"extension already mapped to another format: {key}")
            for mime_type in parser.mime_types:
                key = _normalise_mime(mime_type)
                if not key:
                    continue
                current = self._mime_formats.get(key)
                if current not in {None, format_name} and not replace:
                    raise ValueError(f"MIME already mapped to another format: {key}")
        for key in format_keys:
            self._parsers[key] = parser
        if len(parser.formats) == 1:
            format_name = format_keys[0]
            for extension in parser.extensions:
                key = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
                self._extension_formats[key] = format_name
            for mime_type in parser.mime_types:
                key = _normalise_mime(mime_type)
                if not key:
                    continue
                self._mime_formats[key] = format_name
            if parser.mime_types:
                self._format_mimes[format_name] = _normalise_mime(parser.mime_types[0]) or (
                    "application/octet-stream"
                )

    def resolve(self, format_name: str) -> ParserPort:
        key = format_name.strip().lower()
        parser = self._parsers.get(key)
        if parser is None:
            raise DocumentParseError(
                "PARSE_UNSUPPORTED_FORMAT",
                f"暂不支持该文件格式：{key or 'unknown'}。",
                status_code=415,
            )
        return parser

    def detect(self, path: str | Path, declared_mime: str | None = None) -> DetectionResult:
        source = Path(path)
        if not source.exists() or not source.is_file():
            raise DocumentParseError("PARSE_FILE_NOT_FOUND", "待解析文件不存在。", status_code=404)
        try:
            size = source.stat().st_size
            with source.open("rb") as stream:
                sample = stream.read(128 * 1024)
        except OSError as exc:
            raise DocumentParseError("PARSE_FILE_UNREADABLE", "待解析文件无法读取。") from exc
        if size == 0:
            raise DocumentParseError("PARSE_EMPTY_FILE", "不能解析空文件。")
        if size > 256 * 1024 * 1024:
            raise DocumentParseError("PARSE_FILE_TOO_LARGE", "文件大小超过解析安全上限。")

        extension = source.suffix.lower()
        extension_format = self._extension_formats.get(extension)
        declared = _normalise_mime(declared_mime)
        mime_format = self._mime_formats.get(declared or "")
        warnings: list[ParseWarning] = []

        if sample.startswith(b"%PDF-"):
            detected_format = "pdf"
        elif sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            detected_format = _zip_format(source)
            if detected_format is None:
                raise DocumentParseError(
                    "PARSE_UNSUPPORTED_ARCHIVE",
                    "仅支持 DOCX、PPTX、XLSX，不解析普通压缩包。",
                    status_code=415,
                )
        else:
            detected_format = self._registered_signature_match(source, sample)
            if detected_format is None and _looks_binary(sample):
                raise DocumentParseError(
                    "PARSE_UNSUPPORTED_FORMAT",
                    "文件内容与受支持格式均不匹配。",
                    status_code=415,
                )
            if detected_format is None:
                detected_format = _text_format(
                    _decode_sample(sample), extension_format, mime_format
                )

        if extension_format and extension_format != detected_format:
            warnings.append(
                _warning(
                    "PARSE_EXTENSION_MISMATCH",
                    f"扩展名表示 {extension_format}，实际内容识别为 {detected_format}。",
                )
            )
        if declared and mime_format and mime_format != detected_format:
            warnings.append(
                _warning(
                    "PARSE_MIME_MISMATCH",
                    f"声明 MIME 表示 {mime_format}，实际内容识别为 {detected_format}。",
                )
            )
        elif declared and not mime_format:
            guessed, _ = mimetypes.guess_type(source.name)
            warnings.append(
                _warning(
                    "PARSE_UNKNOWN_DECLARED_MIME",
                    f"未识别的声明 MIME：{declared}；已按文件内容解析"
                    + (f"（文件名推测 {guessed}）" if guessed else "")
                    + "。",
                )
            )

        parser = self._parsers.get(detected_format)
        return DetectionResult(
            format=detected_format,
            mime_type=self._format_mimes.get(detected_format, "application/octet-stream"),
            warnings=warnings,
            parser_id=parser.parser_id if parser else f"unregistered.{detected_format}",
            parser_version=parser.parser_version if parser else "0",
        )

    def profile(self, path: str | Path, declared_mime: str | None = None) -> DetectionResult:
        """``detect`` 的语义化别名，供 API 预览接口使用。"""

        return self.detect(path, declared_mime)

    def parse(
        self,
        path: str | Path,
        *,
        declared_mime: str | None = None,
        work_dir: str | Path,
        config: Mapping[str, Any] | None = None,
        source_sha256: str | None = None,
    ) -> ParsedDocument:
        source = Path(path)
        detection = self.detect(source, declared_mime)
        parser = self.resolve(detection.format)
        parser_config = config or {}
        parsed_limits = ParserLimits.from_config(parser_config)
        if source.stat().st_size > parsed_limits.max_source_bytes:
            raise DocumentParseError("PARSE_FILE_TOO_LARGE", "文件大小超过任务解析上限。")
        digest = source_sha256 or self._sha256(source)
        target = Path(work_dir)
        target.mkdir(parents=True, exist_ok=True)
        request = ParseRequest(
            path=source,
            work_dir=target,
            detection=detection,
            source_sha256=digest,
            config=parser_config,
        )
        parsed = parser.parse(request)
        merged_warnings = [*detection.warnings, *parsed.warnings]
        return parsed.model_copy(
            update={
                "parser_id": parser.parser_id,
                "parser_version": parser.parser_version,
                "source_sha256": digest,
                "source_format": detection.format,
                "source_mime_type": detection.mime_type,
                "warnings": merged_warnings,
            }
        )

    def capabilities(self) -> list[dict[str, Any]]:
        seen: set[int] = set()
        result: list[dict[str, Any]] = []
        for parser in self._parsers.values():
            identity = id(parser)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(
                {
                    "parser_id": parser.parser_id,
                    "parser_version": parser.parser_version,
                    "formats": list(parser.formats),
                    "mime_types": list(parser.mime_types),
                    "extensions": list(parser.extensions),
                    "network_access": False,
                }
            )
        return sorted(result, key=lambda item: item["parser_id"])

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _registered_signature_match(self, path: Path, sample: bytes) -> str | None:
        """调用解析器可选的纯本地签名探测器，支持注册新二进制格式。"""

        seen: set[int] = set()
        for parser in self._parsers.values():
            if id(parser) in seen:
                continue
            seen.add(id(parser))
            matcher = getattr(parser, "matches_signature", None)
            if not callable(matcher):
                continue
            try:
                matched = bool(matcher(sample, path))
            except Exception as exc:
                raise DocumentParseError(
                    "PARSE_SIGNATURE_DETECTOR_FAILED",
                    f"解析器 {parser.parser_id} 的签名探测失败。",
                ) from exc
            if matched:
                return parser.formats[0]
        return None


def _build_default_registry() -> ParserRegistry:
    from app.rag.import_.parsing.local_parsers import build_default_local_parsers

    registry = ParserRegistry()
    for parser in build_default_local_parsers():
        registry.register(parser)
    return registry


default_parser_registry = _build_default_registry()


__all__ = [
    "DetectionResult",
    "ParserRegistry",
    "default_parser_registry",
]
