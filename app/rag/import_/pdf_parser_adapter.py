"""MinerU PDF 适配器：把旧 PDF→Markdown 服务纳入统一 ParserPort。"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import re
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import requests
from pypdf import PasswordType, PdfReader, PdfWriter
from pypdf.errors import PdfReadError, PyPdfError

from app.domain.parsing import (
    DocumentParseError,
    ParsedAsset,
    ParsedBlock,
    ParsedDocument,
    ParseWarning,
    SourceLocation,
)
from app.rag.import_.parsing.base import DetectionResult, ParseRequest
from app.rag.import_.parsing.local_parsers import MarkdownParser
from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown


_SKIPPED_MINERU_TYPES = {
    "header",
    "footer",
    "page_header",
    "page_footer",
    "page_number",
    "discarded",
}

_PRESERVED_TEXT_CONTROLS = {"\n", "\r", "\t"}


def _sanitize_extracted_text(value: str) -> str:
    """删除解析服务泄漏到正文中的不可显示 C0/C1 控制字符。

    PDF 字体映射或 MinerU 的 content-list 偶尔会把 ``U+0001`` 一类字形
    控制码夹在两个正常词元之间。这些字符没有可检索语义，却会继续污染切块、
    embedding 与最终回答。这里只删除 C0、DEL、C1，并保留文本结构所需的
    TAB/CR/LF；不做 Unicode 归一化，也不删除任何其他 Unicode 字符。
    """

    return "".join(
        character
        for character in value
        if character in _PRESERVED_TEXT_CONTROLS
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
    )


def _sanitize_extracted_value(value: Any) -> Any:
    """递归清理 MinerU 结构字段，同时保持其 list/dict 形状。"""

    if isinstance(value, str):
        return _sanitize_extracted_text(value)
    if isinstance(value, list):
        return [_sanitize_extracted_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_extracted_value(item) for key, item in value.items()}
    return value


def _sanitize_parsed_document_text(document: ParsedDocument) -> ParsedDocument:
    """把 MinerU Markdown 降级路径也收敛到相同的文本边界。"""

    return document.model_copy(
        update={
            "blocks": [
                block.model_copy(
                    update={
                        "text": _sanitize_extracted_text(block.text),
                        "structured_data": _sanitize_extracted_value(
                            block.structured_data
                        ),
                        "parent_heading_path": [
                            _sanitize_extracted_text(item)
                            for item in block.parent_heading_path
                        ],
                    }
                )
                for block in document.blocks
            ]
        }
    )


def _normalized_mineru_markdown_path(markdown_path: Path, work_dir: Path) -> Path:
    """在 Markdown 语法解析前移除控制码，避免 C1 被误判成换行。"""

    original = markdown_path.read_text(encoding="utf-8")
    normalized = _sanitize_extracted_text(original)
    if normalized == original:
        return markdown_path
    target = work_dir / "normalized" / "mineru-sanitized.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalized, encoding="utf-8")
    return target


@dataclass(frozen=True, slots=True)
class _PdfTextFragment:
    """PDF 文本层中一个带基线坐标的原始文本片段。"""

    text: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _PdfPageTextLayer:
    """未旋转 PDF 页的文本层坐标信息。"""

    left: float
    bottom: float
    width: float
    height: float
    fragments: tuple[_PdfTextFragment, ...]


def _load_pdf_text_layers(
    source_path: Path,
    page_numbers: set[int],
) -> dict[int, _PdfPageTextLayer]:
    """尽力读取 PDF 自带文本层，失败时不影响 MinerU 主结果。

    该文本层只用于修复 MinerU 在混合字体行中遗漏的内嵌片段，不能替代
    MinerU 的版面分析。旋转页的坐标变换容易产生错误匹配，因此先明确跳过。
    """

    if not page_numbers:
        return {}
    try:
        reader = PdfReader(source_path, strict=False)
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            return {}
        page_count = len(reader.pages)
    except Exception:
        # 这是对已成功生成的 MinerU 结果做可选增强，PDF 文本层的任何异常
        # 都只能关闭补偿，不能反向使主解析结果失败。
        return {}

    layers: dict[int, _PdfPageTextLayer] = {}
    for page_no in sorted(page_numbers):
        if page_no < 1 or page_no > page_count:
            continue
        try:
            page = reader.pages[page_no - 1]
            if int(page.rotation or 0) % 360:
                continue
            cropbox = page.cropbox
            left = float(cropbox.left)
            bottom = float(cropbox.bottom)
            width = float(cropbox.right) - left
            height = float(cropbox.top) - bottom
            if not (
                math.isfinite(width)
                and math.isfinite(height)
                and width > 0
                and height > 0
            ):
                continue
            fragments: list[_PdfTextFragment] = []

            def visit_text(
                text: str,
                current_matrix: list[float],
                text_matrix: list[float],
                _font: Any,
                _font_size: float,
            ) -> None:
                if not text or not text.strip() or len(current_matrix) < 6 or len(text_matrix) < 6:
                    return
                # PDF 点先经过文本矩阵，再经过 current transformation matrix。
                text_x = float(text_matrix[4])
                text_y = float(text_matrix[5])
                x = (
                    float(current_matrix[0]) * text_x
                    + float(current_matrix[2]) * text_y
                    + float(current_matrix[4])
                )
                y = (
                    float(current_matrix[1]) * text_x
                    + float(current_matrix[3]) * text_y
                    + float(current_matrix[5])
                )
                if math.isfinite(x) and math.isfinite(y):
                    fragments.append(_PdfTextFragment(text=text, x=x, y=y))

            page.extract_text(visitor_text=visit_text)
            if fragments:
                layers[page_no] = _PdfPageTextLayer(
                    left=left,
                    bottom=bottom,
                    width=width,
                    height=height,
                    fragments=tuple(fragments),
                )
        except Exception:
            # 文本层补偿是可选增强；单页异常不得让已有 MinerU 结果失败。
            continue
    return layers


def _join_pdf_text_fragments(fragments: list[_PdfTextFragment]) -> str:
    parts: list[str] = []
    for fragment in fragments:
        value = _sanitize_extracted_text(fragment.text)
        if not value:
            continue
        previous = parts[-1][-1] if parts and parts[-1] else ""
        current = value[0]
        # 某些 PDF 会把标准代号和编号拆成独立字体片段，却不在文本操作符中
        # 保留空格。仅在 ASCII 字母/数字类型切换时补空格，避免拆散普通单词。
        needs_token_space = (
            previous
            and not previous.isspace()
            and not current.isspace()
            and (
                (previous.isascii() and previous.isalpha() and current.isdigit())
                or (previous.isdigit() and current.isascii() and current.isalpha())
            )
        )
        if needs_token_space:
            parts.append(" ")
        parts.append(value)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _text_layer_candidate(
    layer: _PdfPageTextLayer,
    normalized_bbox: list[float],
) -> str:
    x0, y0, x1, y1 = normalized_bbox
    left = layer.left + x0 * layer.width
    right = layer.left + x1 * layer.width
    bottom = layer.bottom + (1.0 - y1) * layer.height
    top = layer.bottom + (1.0 - y0) * layer.height
    x_tolerance = max(2.0, layer.width * 0.002)
    y_tolerance = max(2.0, layer.height * 0.002)
    selected = [
        fragment
        for fragment in layer.fragments
        if left - x_tolerance <= fragment.x <= right + x_tolerance
        and bottom - y_tolerance <= fragment.y <= top + y_tolerance
    ]
    return _join_pdf_text_fragments(selected)


def _comparison_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace()
    )


def _is_subsequence(needle: str, haystack: str) -> bool:
    cursor = 0
    for character in needle:
        cursor = haystack.find(character, cursor)
        if cursor < 0:
            return False
        cursor += 1
    return True


def _recover_inline_text_gap(existing: str, candidate: str) -> str | None:
    """仅在候选是同一句话的严格超集时接受 PDF 文本层补偿。"""

    existing_compact = _comparison_text(existing)
    candidate_compact = _comparison_text(candidate)
    if len(existing_compact) < 4 or len(candidate_compact) <= len(existing_compact) + 1:
        return None
    if "\ufffd" in candidate or not candidate.isprintable():
        return None
    if len(candidate_compact) > len(existing_compact) * 4 + 64:
        return None
    if len(existing_compact) / len(candidate_compact) < 0.4:
        return None
    if not _is_subsequence(existing_compact, candidate_compact):
        return None
    anchor_size = min(4, max(2, len(existing_compact) // 4))
    if not (
        candidate_compact.startswith(existing_compact[:anchor_size])
        and candidate_compact.endswith(existing_compact[-anchor_size:])
    ):
        return None
    return candidate


def _prepare_pdf_for_mineru(request: ParseRequest) -> tuple[Path, ParseWarning | None]:
    """区分真正需要密码的 PDF 与空用户密码的 owner-only 权限限制。

    不能仅凭 trailer 中存在 ``/Encrypt`` 就认定文档不可读：很多厂商手册
    使用空 user password，只设置 owner password 来限制复制/修改。对于这类
    文档，先在任务工作目录生成不带加密字典的临时副本，再把副本交给 MinerU；
    真正需要用户密码的文档仍在外发前 fail closed。
    """

    try:
        reader = PdfReader(request.path, strict=False)
        if not reader.is_encrypted:
            return request.path, None
        password_type = reader.decrypt("")
        if password_type == PasswordType.NOT_DECRYPTED:
            raise DocumentParseError(
                "PARSE_ENCRYPTED_DOCUMENT",
                "PDF 需要密码才能读取，当前解析服务不接收文档密码",
            )

        # 强制读取页面树，确保不是“密码验证成功但正文对象仍不可解密”。
        if not reader.pages:
            raise PdfReadError("PDF does not contain any pages")
        request.work_dir.mkdir(parents=True, exist_ok=True)
        # 这里只需要任务内唯一的、MinerU 能识别的 PDF 文件名。不能把完整
        # source hash 和 UUID 都拼进名称：work_dir 本身已经是内容寻址长路径，
        # 再把长名称用作 MinerU 解压目录会在传统 Windows MAX_PATH 环境失败。
        destination = request.work_dir / f"normalized-{uuid4().hex}.pdf"
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        try:
            with destination.open("xb") as output:
                writer.write(output)
            if destination.stat().st_size > request.limits.max_source_bytes:
                raise DocumentParseError(
                    "PARSE_FILE_TOO_LARGE",
                    "PDF 解密副本超过解析安全上限",
                )

            verification = PdfReader(destination, strict=False)
            if verification.is_encrypted or not verification.pages:
                raise PdfReadError("normalized PDF is still encrypted or has no pages")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination, ParseWarning(
            code="PARSE_PDF_EMPTY_PASSWORD_NORMALIZED",
            severity="info",
            message="PDF 使用空用户密码和所有者权限限制；已在任务目录生成安全解密副本后解析。",
        )
    except DocumentParseError:
        raise
    except (PyPdfError, OSError, ValueError) as exc:
        raise DocumentParseError(
            "PARSE_CORRUPT_PDF",
            "PDF 加密结构损坏或无法安全读取",
        ) from exc


def _with_warning(
    parsed: ParsedDocument,
    warning: ParseWarning | None,
) -> ParsedDocument:
    if warning is None:
        return parsed
    return parsed.model_copy(update={"warnings": [*parsed.warnings, warning]})


def _joined_text(value: Any) -> str:
    if isinstance(value, str):
        return _sanitize_extracted_text(value).strip()
    if isinstance(value, list):
        parts = [
            _sanitize_extracted_text(str(item)).strip()
            for item in value
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _safe_image_mime(data: bytes) -> str | None:
    signatures = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/gif": data.startswith((b"GIF87a", b"GIF89a")),
        "image/bmp": data.startswith(b"BM"),
        "image/tiff": data.startswith((b"II*\x00", b"MM\x00*")),
        "image/webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    return next((media_type for media_type, valid in signatures.items() if valid), None)


def _parse_mineru_content_list(request: ParseRequest, content_list_path: Path) -> ParsedDocument:
    """按 MinerU content_list 的权威顺序生成带页码与标准化坐标的领域对象。"""

    if content_list_path.stat().st_size > request.limits.max_zip_entry_bytes:
        raise ValueError("MinerU content_list 超过解析大小上限")
    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MinerU content_list 顶层必须是数组")

    text_layer_pages = {
        raw_item["page_idx"] + 1
        for raw_item in payload
        if isinstance(raw_item, dict)
        and str(raw_item.get("type") or "text").strip().lower() == "text"
        and isinstance(raw_item.get("page_idx"), int)
        and not isinstance(raw_item.get("page_idx"), bool)
        and raw_item["page_idx"] >= 0
        and isinstance(raw_item.get("bbox"), list)
    }
    text_layers = _load_pdf_text_layers(request.path, text_layer_pages)

    blocks: list[ParsedBlock] = []
    assets: list[ParsedAsset] = []
    warnings: list[ParseWarning] = list(request.detection.warnings)
    warning_keys: set[tuple[str, int | None]] = set()
    headings: list[str] = []
    asset_by_hash: dict[str, str] = {}
    recovered_blocks_by_page: dict[int, int] = {}
    extraction_root = content_list_path.parent.resolve()

    def warn(
        code: str,
        message: str,
        *,
        page_no: int | None = None,
        severity: str = "warning",
    ) -> None:
        key = (code, page_no)
        if key in warning_keys:
            return
        warning_keys.add(key)
        warnings.append(
            ParseWarning(
                code=code,
                severity=severity,
                message=message,
                location=SourceLocation(page_no=page_no) if page_no else None,
            )
        )

    def location_for(item: dict[str, Any]) -> SourceLocation:
        raw_page = item.get("page_idx")
        page_no: int | None = None
        if isinstance(raw_page, int) and not isinstance(raw_page, bool) and raw_page >= 0:
            page_no = raw_page + 1
        else:
            warn(
                "PARSE_PDF_PAGE_INDEX_INVALID",
                "MinerU 块缺少合法的零基 page_idx；该块不会伪造页码。",
            )

        raw_bbox = item.get("bbox")
        if raw_bbox is None:
            warn(
                "PARSE_PDF_BBOX_UNAVAILABLE",
                "MinerU 块未提供 bbox；保留可验证页码但不伪造坐标。",
                page_no=page_no,
                severity="info",
            )
            return SourceLocation(page_no=page_no)
        valid_bbox = (
            isinstance(raw_bbox, list)
            and len(raw_bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_bbox
            )
        )
        if valid_bbox:
            values = [float(value) for value in raw_bbox]
            valid_bbox = (
                all(0.0 <= value <= 1000.0 for value in values)
                and values[2] > values[0]
                and values[3] > values[1]
            )
        if not valid_bbox or page_no is None:
            warn(
                "PARSE_PDF_BBOX_INVALID",
                "MinerU bbox 非法或缺少合法页码；保留合法页码但不伪造坐标。",
                page_no=page_no,
            )
            return SourceLocation(page_no=page_no)
        return SourceLocation(
            page_no=page_no,
            bbox=[value / 1000.0 for value in values],
            coordinate_space="normalized_page",
        )

    def add_asset(raw_path: Any, *, page_no: int | None) -> str | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        parsed = urlsplit(raw_path.strip())
        relative = Path(unquote(parsed.path))
        if parsed.scheme or parsed.netloc or relative.is_absolute():
            warn(
                "PARSE_PDF_ASSET_OUTSIDE_ROOT",
                "MinerU 图片必须是抽取根目录内的相对路径，当前资源已跳过。",
                page_no=page_no,
            )
            return None
        try:
            candidate = (extraction_root / relative).resolve(strict=True)
            candidate.relative_to(extraction_root)
        except (OSError, ValueError):
            warn(
                "PARSE_PDF_ASSET_OUTSIDE_ROOT",
                "MinerU 图片不存在或越出抽取根目录，当前资源已跳过。",
                page_no=page_no,
            )
            return None
        if not candidate.is_file() or candidate.stat().st_size > request.limits.max_zip_entry_bytes:
            warn(
                "PARSE_PDF_ASSET_UNAVAILABLE",
                "MinerU 图片不是可读取的普通文件或超过大小上限，已跳过。",
                page_no=page_no,
            )
            return None
        data = candidate.read_bytes()
        media_type = _safe_image_mime(data)
        if media_type is None:
            warn(
                "PARSE_PDF_UNSAFE_IMAGE_SKIPPED",
                "MinerU 图片签名不是受支持的安全光栅格式，已跳过。",
                page_no=page_no,
            )
            return None
        digest = hashlib.sha256(data).hexdigest()
        if digest in asset_by_hash:
            return asset_by_hash[digest]
        asset_id = f"asset-{digest[:24]}"
        suffix = candidate.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = mimetypes.guess_extension(media_type) or ".bin"
        relative_output = Path("assets") / f"{digest}{suffix}"
        destination = request.work_dir / relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        assets.append(
            ParsedAsset(
                asset_id=asset_id,
                object_key=relative_output.as_posix(),
                sha256=digest,
                media_type=media_type,
                size_bytes=len(data),
            )
        )
        asset_by_hash[digest] = asset_id
        return asset_id

    for source_order, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            warn("PARSE_PDF_CONTENT_ITEM_INVALID", "MinerU content_list 含非对象条目，已跳过。")
            continue
        raw_type = str(raw_item.get("type") or "text").strip().lower()
        if raw_type in _SKIPPED_MINERU_TYPES:
            continue
        location = location_for(raw_item)
        raw_level = raw_item.get("text_level")
        level = raw_level if isinstance(raw_level, int) and not isinstance(raw_level, bool) else None
        block_type = "text"
        if raw_type in {"title", "heading"} or (raw_type == "text" and level is not None):
            block_type = "heading"
            level = min(max(level or 1, 1), 6)
        elif raw_type == "table":
            block_type = "table"
        elif raw_type == "image":
            block_type = "image"
        elif raw_type in {"list", "list_item"}:
            block_type = "list"
        elif raw_type in {"code", "equation", "interline_equation"}:
            block_type = "code"
        elif raw_type != "text":
            warn(
                "PARSE_PDF_BLOCK_TYPE_FALLBACK",
                f"未知 MinerU 块类型 {raw_type!r} 已按文本保留。",
                page_no=location.page_no,
                severity="info",
            )

        text = _joined_text(raw_item.get("text"))
        if (
            raw_type == "text"
            and text
            and location.page_no is not None
            and location.bbox is not None
            and (text_layer := text_layers.get(location.page_no)) is not None
        ):
            candidate = _text_layer_candidate(text_layer, location.bbox)
            recovered = _recover_inline_text_gap(text, candidate)
            if recovered is not None:
                text = recovered
                recovered_blocks_by_page[location.page_no] = (
                    recovered_blocks_by_page.get(location.page_no, 0) + 1
                )
        structured_data: dict[str, Any] | None = None
        asset_refs: list[str] = []
        if block_type == "table":
            caption = _joined_text(raw_item.get("table_caption"))
            footnote = _joined_text(raw_item.get("table_footnote"))
            body = _joined_text(raw_item.get("table_body"))
            text = "\n".join(
                part for part in (caption, text, body, footnote) if part
            ).strip()
            structured_data = {
                key: _sanitize_extracted_value(raw_item[key])
                for key in ("table_body", "table_caption", "table_footnote")
                if key in raw_item
            }
        elif block_type == "image":
            caption = _joined_text(raw_item.get("image_caption"))
            footnote = _joined_text(raw_item.get("image_footnote"))
            text = "\n".join(part for part in (caption, text, footnote) if part).strip()
            asset_id = add_asset(raw_item.get("img_path"), page_no=location.page_no)
            if asset_id:
                asset_refs.append(asset_id)
            if not text and not asset_refs:
                text = "图片（资源不可用）"
        if not text and structured_data is None and not asset_refs:
            warn(
                "PARSE_PDF_EMPTY_BLOCK_SKIPPED",
                "MinerU 空内容块已跳过。",
                page_no=location.page_no,
                severity="info",
            )
            continue
        parent_heading_path = list(headings)
        blocks.append(
            ParsedBlock(
                block_id=f"block-{len(blocks):08d}",
                type=block_type,
                text=text,
                structured_data=structured_data,
                location=location,
                parent_heading_path=parent_heading_path,
                order=len(blocks),
                asset_refs=asset_refs,
            )
        )
        if block_type == "heading":
            heading_level = level or 1
            headings = headings[: heading_level - 1]
            headings.append(text)
        if len(blocks) >= request.limits.max_blocks and source_order < len(payload) - 1:
            raise DocumentParseError("PARSE_TOO_MANY_BLOCKS", "文档结构块数量超过解析上限。")

    if not blocks:
        warn(
            "PARSE_PDF_CONTENT_LIST_EMPTY",
            "MinerU content_list 没有可索引正文块。",
            severity="info",
        )
    for page_no, count in sorted(recovered_blocks_by_page.items()):
        warn(
            "PARSE_PDF_TEXT_LAYER_GAP_RECOVERED",
            f"已使用同页同坐标的 PDF 文本层补回 MinerU 遗漏的内嵌文本；共 {count} 块。",
            page_no=page_no,
            severity="info",
        )
    return ParsedDocument(
        parser_id=MinerUPdfParser.parser_id,
        parser_version=MinerUPdfParser.parser_version,
        source_sha256=request.source_sha256,
        source_format="pdf",
        source_mime_type="application/pdf",
        metadata={
            "title": request.path.stem,
            "source": request.path.name,
            "mineru_content_list": str(content_list_path),
            "location_coordinate_space": "normalized_page",
        },
        blocks=blocks,
        assets=assets,
        warnings=warnings,
    )


class MinerUPdfParser:
    parser_id = "mineru.pdf"
    parser_version = "1.1.3"
    formats = ("pdf",)
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        mineru_path, preprocessing_warning = _prepare_pdf_for_mineru(request)
        state = {
            "task_id": f"parse-{request.source_sha256[:16]}",
            "pdf_path": str(mineru_path),
            "local_file_path": str(mineru_path),
            "local_dir": str(request.work_dir),
            "parser_config": dict(request.config),
        }
        try:
            parsed_state = parse_pdf_to_markdown(state)
        except (requests.Timeout, TimeoutError) as exc:
            raise DocumentParseError(
                "PARSE_TIMEOUT",
                "MinerU PDF 解析超时，可在依赖恢复后重试",
                retryable=True,
                status_code=504,
            ) from exc
        except DocumentParseError:
            raise
        except (requests.ConnectionError, ConnectionError) as exc:
            raise DocumentParseError(
                "PARSE_DEPENDENCY_UNAVAILABLE",
                "MinerU PDF 解析服务当前不可用",
                retryable=True,
                status_code=503,
            ) from exc
        except Exception as exc:
            raise DocumentParseError(
                "PARSE_PDF_FAILED",
                "MinerU 未能生成可用的 PDF 解析结果",
                retryable=False,
            ) from exc
        finally:
            if mineru_path != request.path:
                with suppress(OSError):
                    mineru_path.unlink(missing_ok=True)

        content_list_value = parsed_state.get("mineru_content_list_path")
        content_list_warning: ParseWarning | None = None
        if content_list_value:
            content_list_path = Path(str(content_list_value))
            try:
                if not content_list_path.is_file():
                    raise FileNotFoundError(content_list_path)
                return _with_warning(
                    _parse_mineru_content_list(request, content_list_path.resolve()),
                    preprocessing_warning,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                content_list_warning = ParseWarning(
                    code="PARSE_PDF_CONTENT_LIST_INVALID",
                    severity="warning",
                    message=f"MinerU content_list 不可用，已安全降级到 Markdown：{type(exc).__name__}",
                )

        markdown_path = Path(parsed_state.get("md_path") or "")
        if not markdown_path.is_file():
            raise DocumentParseError(
                "PARSE_PDF_EMPTY_RESULT",
                "MinerU 返回结果中没有 Markdown 正文",
            )
        normalized_markdown_path = _normalized_mineru_markdown_path(
            markdown_path,
            request.work_dir,
        )
        markdown_request = ParseRequest(
            path=normalized_markdown_path,
            work_dir=request.work_dir,
            detection=DetectionResult(
                format="markdown",
                mime_type="text/markdown",
                parser_id="builtin.markdown",
                parser_version="1.0.0",
            ),
            source_sha256=request.source_sha256,
            config=request.config,
        )
        intermediate = _sanitize_parsed_document_text(
            MarkdownParser().parse(markdown_request)
        )
        warnings = list(intermediate.warnings)
        if preprocessing_warning is not None:
            warnings.append(preprocessing_warning)
        if content_list_warning is not None:
            warnings.append(content_list_warning)
        else:
            warnings.append(
                ParseWarning(
                    code="PARSE_PDF_CONTENT_LIST_UNAVAILABLE",
                    severity="info",
                    message="MinerU 未提供唯一的非 v2 content_list；已降级到 Markdown。",
                )
            )
        if not any(block.location.page_no for block in intermediate.blocks):
            warnings.append(
                ParseWarning(
                    code="PARSE_PDF_PAGE_LOCATION_UNAVAILABLE",
                    severity="info",
                    message="当前 MinerU Markdown 未提供可靠页码坐标；块仍可回溯到结构 block。",
                )
            )
        metadata = dict(intermediate.metadata)
        metadata.update(
            {
                "source": request.path.name,
                "intermediate_markdown": str(markdown_path),
            }
        )
        return intermediate.model_copy(
            update={
                "parser_id": self.parser_id,
                "parser_version": self.parser_version,
                "source_format": "pdf",
                "source_mime_type": "application/pdf",
                "metadata": metadata,
                "warnings": warnings,
            }
        )


def ensure_pdf_parser_registered(registry) -> None:
    """幂等注册，便于离线测试替换 registry 中的 PDF fake。"""

    try:
        registry.resolve("pdf")
    except DocumentParseError:
        registry.register(MinerUPdfParser())


__all__ = ["MinerUPdfParser", "ensure_pdf_parser_registered"]
