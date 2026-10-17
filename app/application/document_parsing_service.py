"""TODO-07 统一解析、制品缓存、模板切块与预览编排。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from app.application.parse_artifact_store import (
    LocalParseArtifactStore,
    MinIOParseArtifactMirror,
    build_parse_cache_key,
    canonical_json,
)
from app.domain.parsing import (
    DocumentParseError,
    ParseArtifactInfo,
    ParseWarning,
    ParsedBlock,
    ParsedDocument,
    ParserProfile,
)
from app.infra.object_storage.minio_gateway import minio_gateway
from app.rag.import_.chunking_templates import (
    DEFAULT_TEMPLATE_ID,
    chunk_parsed_document_with_profile,
    default_chunking_template_registry,
)
from app.rag.import_.parsing import default_parser_registry
from app.rag.import_.pdf_parser_adapter import ensure_pdf_parser_registered
from app.shared.runtime.logger import PROJECT_ROOT, logger


PARSE_ARTIFACT_ROOT = PROJECT_ROOT / "output" / "parsed-artifacts"


@dataclass(slots=True)
class DocumentParsingOutcome:
    document: ParsedDocument
    artifact: ParseArtifactInfo
    profile: ParserProfile
    chunks: list[dict[str, Any]]
    chunking_config: dict[str, Any]
    markdown_path: Path
    warnings: list[ParseWarning]
    chunk_artifact: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser_limits_config(config: dict[str, Any]) -> dict[str, Any]:
    """把 API/旧配置别名归一到 ParserLimits 字段。"""

    result = dict(config)
    aliases = {
        "zip_max_entries": "max_zip_entries",
        "zip_max_uncompressed_bytes": "max_zip_uncompressed_bytes",
        "zip_max_compression_ratio": "max_compression_ratio",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result[new] = result[old]
    return result


def _requested_template(config: dict[str, Any]) -> tuple[str, str | None]:
    template_id = str(
        config.get("chunking_template")
        or config.get("chunking_template_id")
        or DEFAULT_TEMPLATE_ID
    ).strip().lower()
    if template_id == "auto":
        raise DocumentParseError(
            "CHUNK_TEMPLATE_AUTO_NOT_SELECTABLE",
            "auto 仅用于模板建议；请显式选择 general/manual/paper/qa/table/presentation",
        )
    version = str(config.get("chunking_template_version") or "").strip() or None
    return template_id, version


def resolve_parser_profile(
    path: str | Path,
    *,
    declared_mime: str | None = None,
    parser_config: dict[str, Any] | None = None,
) -> ParserProfile:
    """上传落盘后解析内容签名，并冻结实际 parser/template 选择。"""

    ensure_pdf_parser_registered(default_parser_registry)
    config = dict(parser_config or {})
    detection = default_parser_registry.detect(path, declared_mime)
    requested_parser = str(config.get("parser_id") or "auto").strip().lower()
    if requested_parser not in {"", "auto", detection.parser_id.lower()}:
        raise DocumentParseError(
            "PARSE_PARSER_MISMATCH",
            f"请求 parser={requested_parser} 不支持实际格式 {detection.format}；"
            f"应使用 {detection.parser_id}",
        )
    requested_parser_version = str(config.get("parser_version") or "").strip()
    if requested_parser_version and requested_parser_version != detection.parser_version:
        raise DocumentParseError(
            "PARSE_PARSER_VERSION_UNAVAILABLE",
            f"请求 parser 版本 {requested_parser_version} 不可用；当前为 {detection.parser_version}",
        )
    template_id, template_version = _requested_template(config)
    template = default_chunking_template_registry.resolve(template_id, template_version)
    format_suggestion = {
        "pptx": "presentation",
        "xlsx": "table",
        "csv": "table",
    }.get(detection.format, "general")
    return ParserProfile(
        detected_format=detection.format,
        detected_mime_type=detection.mime_type,
        parser_id=detection.parser_id,
        parser_version=detection.parser_version,
        template_id=template.template_id,
        template_version=template.version,
        suggested_template_id=format_suggestion,
        warnings=detection.warnings,
    )


def resolved_parser_config_snapshot(
    path: str | Path,
    *,
    declared_mime: str | None,
    parser_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], ParserProfile]:
    """生成写入 DocumentVersion 的 requested + resolved 不可变快照。"""

    requested = dict(parser_config or {})
    profile = resolve_parser_profile(
        path,
        declared_mime=declared_mime,
        parser_config=requested,
    )
    snapshot = dict(requested)
    snapshot.update(
        {
            "parser_id": profile.parser_id,
            "parser_version": profile.parser_version,
            "detected_format": profile.detected_format,
            "detected_mime_type": profile.detected_mime_type,
            "chunking_template": profile.template_id,
            "chunking_template_version": profile.template_version,
            "suggested_chunking_template_id": profile.suggested_template_id,
            "parse_profile_schema_version": "parse-profile.v1",
            "parse_profile_warnings": [
                warning.model_dump(mode="json") for warning in profile.warnings
            ],
        }
    )
    return snapshot, profile


def _load_or_parse(
    path: Path,
    profile: ParserProfile,
    config: dict[str, Any],
    source_sha256: str,
    store: LocalParseArtifactStore,
) -> tuple[ParsedDocument, ParseArtifactInfo, Path, list[ParseWarning]]:
    cache_key = build_parse_cache_key(
        input_sha256=source_sha256,
        parser_id=profile.parser_id,
        parser_version=profile.parser_version,
        parser_config=config,
    )
    warnings: list[ParseWarning] = []
    cached = store.load(cache_key)
    mirror: MinIOParseArtifactMirror | None = None
    use_minio = bool(config.get("persist_parsed_artifact_to_minio", True))
    require_minio = bool(config.get("require_parsed_artifact_minio", False))
    if cached is None and use_minio:
        try:
            mirror = MinIOParseArtifactMirror(
                minio_gateway.client(), minio_gateway.bucket_name
            )
            cached = mirror.download_into(cache_key, store)
        except Exception as exc:
            if require_minio:
                raise DocumentParseError(
                    "PARSE_ARTIFACT_STORAGE_UNAVAILABLE",
                    "解析制品对象存储不可用",
                    retryable=True,
                    status_code=503,
                ) from exc
            logger.warning(f"解析制品 MinIO 读取失败，使用本地缓存/解析：{exc}")
            warnings.append(
                ParseWarning(
                    code="PARSE_ARTIFACT_MINIO_FALLBACK",
                    severity="warning",
                    message="MinIO 暂不可用，本次解析制品已使用受校验的本地持久缓存。",
                )
            )
    work_dir = store.artifact_path(cache_key).parent / f"{cache_key}.work"
    if cached is not None:
        document, artifact = cached
        if document.source_sha256 != source_sha256:
            raise DocumentParseError(
                "PARSE_ARTIFACT_INPUT_MISMATCH",
                "缓存制品与当前输入 SHA-256 不一致",
            )
        return document, artifact, work_dir, warnings

    document = default_parser_registry.parse(
        path,
        declared_mime=profile.detected_mime_type,
        work_dir=work_dir,
        config=_parser_limits_config(config),
        source_sha256=source_sha256,
    )
    artifact = store.save(cache_key, document)
    if use_minio:
        try:
            mirror = mirror or MinIOParseArtifactMirror(
                minio_gateway.client(), minio_gateway.bucket_name
            )
            artifact = mirror.upload(artifact, document=document, work_dir=work_dir)
        except Exception as exc:
            if require_minio:
                raise DocumentParseError(
                    "PARSE_ARTIFACT_STORAGE_UNAVAILABLE",
                    "解析制品无法写入 MinIO",
                    retryable=True,
                    status_code=503,
                ) from exc
            logger.warning(f"解析制品 MinIO 写入失败，保留本地持久副本：{exc}")
            warnings.append(
                ParseWarning(
                    code="PARSE_ARTIFACT_MINIO_FALLBACK",
                    severity="warning",
                    message="MinIO 写入失败；完整解析制品已原子写入本地持久目录。",
                )
            )
    return document, artifact, work_dir, warnings


def _table_markdown(block: ParsedBlock) -> str:
    data = block.structured_data if isinstance(block.structured_data, dict) else {}
    headers = data.get("headers") if isinstance(data, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return block.text
    if not headers and rows:
        headers, rows = rows[0], rows[1:]
    if not isinstance(headers, list):
        return block.text
    width = max(1, len(headers))

    def row(values) -> str:
        cells = list(values) if isinstance(values, list) else [values]
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in cells]
        cells.extend([""] * (width - len(cells)))
        return "| " + " | ".join(cells[:width]) + " |"

    return "\n".join([row(headers), row(["---"] * width), *(row(item) for item in rows)])


def _render_markdown(
    document: ParsedDocument,
    *,
    source_path: Path,
    work_dir: Path,
) -> tuple[Path, dict[str, str], list[ParseWarning]]:
    """产生现有图片/VLM 节点可消费的规范 Markdown 与 images 目录。"""

    normalized_dir = work_dir / "normalized"
    images_dir = normalized_dir / "images"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    asset_map: dict[str, str] = {}
    warnings: list[ParseWarning] = []
    for asset in document.assets:
        candidate = Path(asset.object_key)
        source = candidate if candidate.is_absolute() else work_dir / candidate
        source = source.resolve()
        if work_dir.resolve() not in source.parents or not source.is_file():
            warnings.append(
                ParseWarning(
                    code="PARSE_ASSET_MISSING",
                    severity="warning",
                    message=f"解析资产缺失或路径越界：{asset.asset_id}",
                )
            )
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        target = images_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        asset_map[asset.object_key] = f"images/{target.name}"

    asset_by_id = {asset.asset_id: asset for asset in document.assets}
    lines: list[str] = []
    for block in document.blocks:
        if block.type == "heading":
            level = min(6, max(1, len(block.parent_heading_path) + 1))
            lines.append(f"{'#' * level} {block.text}")
        elif block.type == "table":
            lines.append(_table_markdown(block))
        elif block.type == "code":
            language = ""
            if isinstance(block.structured_data, dict):
                language = str(block.structured_data.get("language") or "")
            lines.append(f"```{language}\n{block.text}\n```")
        elif block.type == "list":
            items = (
                block.structured_data.get("items")
                if isinstance(block.structured_data, dict)
                else None
            )
            lines.append(
                "\n".join(f"- {item}" for item in items)
                if isinstance(items, list)
                else block.text
            )
        elif block.type == "image":
            rendered = False
            for asset_id in block.asset_refs:
                asset = asset_by_id.get(asset_id)
                if asset and asset.object_key in asset_map:
                    lines.append(f"![{block.text or '图片'}]({asset_map[asset.object_key]})")
                    rendered = True
            if not rendered and block.text:
                lines.append(block.text)
        else:
            lines.append(block.text)
        lines.append("")
    markdown_path = normalized_dir / f"{source_path.stem}.md"
    markdown_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return markdown_path, asset_map, warnings


def _save_chunk_artifact(
    *,
    store: LocalParseArtifactStore,
    parse_cache_key: str,
    template_id: str,
    template_version: str,
    chunking_config: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    config_hash = hashlib.sha256(canonical_json(chunking_config)).hexdigest()
    # 保持目录和临时名短小，兼容仍启用传统 MAX_PATH 的 Windows 环境。
    directory = store.artifact_path(parse_cache_key).parent / f"{parse_cache_key}.chunks"
    target = directory / f"{template_id}-{template_version}-{config_hash[:16]}.json"
    payload = canonical_json(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".tmp-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "uri": str(target),
        "sha256": digest,
        "size_bytes": len(payload),
        "config_hash": config_hash,
    }


def parse_and_chunk_document(
    path: str | Path,
    *,
    declared_mime: str | None = None,
    parser_config: dict[str, Any] | None = None,
    artifact_store: LocalParseArtifactStore | None = None,
) -> DocumentParsingOutcome:
    source = Path(path).resolve()
    config = dict(parser_config or {})
    profile = resolve_parser_profile(
        source,
        declared_mime=declared_mime,
        parser_config=config,
    )
    source_sha256 = sha256_file(source)
    store = artifact_store or LocalParseArtifactStore(PARSE_ARTIFACT_ROOT)
    document, artifact, work_dir, storage_warnings = _load_or_parse(
        source, profile, config, source_sha256, store
    )
    template_id, template_version = _requested_template(config)
    chunks, chunking_profile = chunk_parsed_document_with_profile(
        document,
        template_id=template_id,
        template_version=template_version,
        file_title=source.stem,
        parser_config=config,
    )
    markdown_path, asset_map, render_warnings = _render_markdown(
        document, source_path=source, work_dir=work_dir
    )
    for chunk in chunks:
        images = chunk.get("images") or []
        mapped = [asset_map.get(str(item), str(item)) for item in images]
        chunk["images"] = mapped or None
        content = str(chunk.get("content") or "")
        for original, replacement in asset_map.items():
            content = content.replace(original, replacement)
        chunk["content"] = content
    warning_candidates = [
        *profile.warnings,
        *document.warnings,
        *storage_warnings,
        *render_warnings,
    ]
    all_warnings: list[ParseWarning] = []
    warning_keys: set[bytes] = set()
    for warning in warning_candidates:
        key = canonical_json(warning.model_dump(mode="json"))
        if key not in warning_keys:
            warning_keys.add(key)
            all_warnings.append(warning)
    resolved_profile = profile.model_copy(
        update={
            "template_id": chunking_profile["template_id"],
            "template_version": chunking_profile["template_version"],
            "suggested_template_id": chunking_profile["suggested_template_id"],
            "warnings": all_warnings,
        }
    )
    chunk_artifact = _save_chunk_artifact(
        store=store,
        parse_cache_key=artifact.cache_key,
        template_id=chunking_profile["template_id"],
        template_version=chunking_profile["template_version"],
        chunking_config=chunking_profile["config"],
        chunks=chunks,
    )
    return DocumentParsingOutcome(
        document=document,
        artifact=artifact,
        profile=resolved_profile,
        chunks=chunks,
        chunking_config=chunking_profile["config"],
        markdown_path=markdown_path,
        warnings=all_warnings,
        chunk_artifact=chunk_artifact,
    )


def apply_parsing_to_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    outcome = parse_and_chunk_document(
        state["local_file_path"],
        declared_mime=(state.get("parser_config") or {}).get("detected_mime_type"),
        parser_config=state.get("parser_config") or {},
    )
    profile = outcome.profile
    parse_artifact_id = ""
    chunking_run_id = ""
    if state.get("document_version_id"):
        from app.application.control_plane_service import (
            record_document_chunking_run,
            record_document_parse_artifact,
        )

        parse_record = record_document_parse_artifact(
            document_version_id=state["document_version_id"],
            input_sha256=outcome.document.source_sha256,
            parser_id=profile.parser_id,
            parser_version=profile.parser_version,
            schema_version=outcome.document.schema_version,
            status="succeeded",
            # cache URI 是稳定逻辑地址；物理本地/MinIO 地址记录到 metrics。
            artifact_uri=f"parse-cache://{outcome.artifact.cache_key}",
            artifact_sha256=outcome.artifact.sha256,
            artifact_size=outcome.artifact.size_bytes,
            warnings=[warning.model_dump(mode="json") for warning in outcome.warnings],
            metrics={
                "cache_hit": outcome.artifact.cache_hit,
                "local_uri": outcome.artifact.uri,
                "object_key": outcome.artifact.object_key,
                "block_count": len(outcome.document.blocks),
                "asset_count": len(outcome.document.assets),
            },
        )
        chunk_record = record_document_chunking_run(
            parse_artifact_id=parse_record["id"],
            template_id=profile.template_id,
            template_version=profile.template_version,
            config_hash=outcome.chunk_artifact["config_hash"],
            status="succeeded",
            artifact_uri=outcome.chunk_artifact["uri"],
            artifact_sha256=outcome.chunk_artifact["sha256"],
            artifact_size=outcome.chunk_artifact["size_bytes"],
            chunk_count=len(outcome.chunks),
        )
        parse_artifact_id = parse_record["id"]
        chunking_run_id = chunk_record["id"]
    state.update(
        {
            "is_document_read_enabled": True,
            "input_format": profile.detected_format,
            "input_mime_type": profile.detected_mime_type,
            "parser_id": profile.parser_id,
            "parser_version": profile.parser_version,
            "chunking_template_id": profile.template_id,
            "chunking_template_version": profile.template_version,
            "suggested_chunking_template_id": profile.suggested_template_id,
            "parsed_artifact_uri": outcome.artifact.uri,
            "parsed_artifact_object_key": outcome.artifact.object_key or "",
            "parsed_artifact_sha256": outcome.artifact.sha256,
            "parsed_artifact_cache_key": outcome.artifact.cache_key,
            "parsed_artifact_cache_hit": outcome.artifact.cache_hit,
            "parsed_document_summary": {
                "schema_version": outcome.document.schema_version,
                "block_count": len(outcome.document.blocks),
                "asset_count": len(outcome.document.assets),
                "warning_count": len(outcome.warnings),
                "source_sha256": outcome.document.source_sha256,
                "chunking_config": outcome.chunking_config,
                "parse_artifact_id": parse_artifact_id,
                "chunking_run_id": chunking_run_id,
                "chunk_artifact": outcome.chunk_artifact,
            },
            "parse_warnings": [
                warning.model_dump(mode="json") for warning in outcome.warnings
            ],
            "md_path": str(outcome.markdown_path),
            "md_content": outcome.markdown_path.read_text(encoding="utf-8"),
            "chunks": outcome.chunks,
            "tables": [],
            "hyperlinks": [],
            "image_refs": [],
            "doc_metadata": outcome.document.metadata,
            "structured_chunks_ready": True,
        }
    )
    return state


def parsing_capabilities() -> dict[str, Any]:
    ensure_pdf_parser_registered(default_parser_registry)
    return {
        "parsed_schema_version": "parsed-document.v1",
        "parsers": default_parser_registry.capabilities(),
        "chunking_templates": default_chunking_template_registry.capabilities(),
    }


def preview_document(
    path: str | Path,
    *,
    declared_mime: str | None,
    parser_config: dict[str, Any] | None,
    max_blocks: int = 20,
) -> dict[str, Any]:
    if not 1 <= max_blocks <= 100:
        raise DocumentParseError(
            "PARSE_PREVIEW_LIMIT_INVALID",
            "max_blocks 必须在 1..100",
        )
    config = dict(parser_config or {})
    # 预览默认不把临时用户文件产生的制品强制写入远端；本地 SHA 缓存仍可复用。
    config.setdefault("persist_parsed_artifact_to_minio", False)
    outcome = parse_and_chunk_document(
        path,
        declared_mime=declared_mime,
        parser_config=config,
    )
    return {
        "detected_format": outcome.profile.detected_format,
        "detected_mime_type": outcome.profile.detected_mime_type,
        "parser_id": outcome.profile.parser_id,
        "parser_version": outcome.profile.parser_version,
        "chunking_template_id": outcome.profile.template_id,
        "chunking_template_version": outcome.profile.template_version,
        "suggested_chunking_template_id": outcome.profile.suggested_template_id,
        "parsed_schema_version": outcome.document.schema_version,
        "artifact": outcome.artifact.model_dump(mode="json"),
        "metadata": outcome.document.metadata,
        "warnings": [warning.model_dump(mode="json") for warning in outcome.warnings],
        "blocks": [
            block.model_dump(mode="json")
            for block in outcome.document.blocks[:max_blocks]
        ],
        "chunks": outcome.chunks[:max_blocks],
        "total_blocks": len(outcome.document.blocks),
        "total_chunks": len(outcome.chunks),
    }


__all__ = [
    "DocumentParsingOutcome",
    "PARSE_ARTIFACT_ROOT",
    "apply_parsing_to_graph_state",
    "parse_and_chunk_document",
    "parsing_capabilities",
    "preview_document",
    "resolve_parser_profile",
    "resolved_parser_config_snapshot",
]
