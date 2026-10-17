"""Application services for the TODO-12 import workbench."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm.exc import StaleDataError

from app.api.schemas.knowledge_workbench import ChunkReplacement
from app.application.parse_artifact_store import (
    LocalParseArtifactStore,
    MinIOParseArtifactMirror,
    canonical_json,
)
from app.application.immutable_source_service import (
    ImmutableSourceIntegrityError,
    ImmutableSourceUnavailableError,
    resolve_immutable_local_source,
)
from app.domain.parsing import DocumentParseError, ParsedDocument
from app.infra.object_storage.minio_gateway import minio_gateway
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.index_lifecycle_repository import IndexLifecycleRepository
from app.infra.persistence.sql.knowledge_workbench_repository import (
    KnowledgeWorkbenchRepository,
    WorkbenchCursorError,
    WorkbenchPreconditionError,
    document_etag,
    draft_etag,
    sanitize_public_json,
)
from app.infra.persistence.sql.models import (
    ChunkEditDraftModel,
    ChunkEditOperationModel,
    DocumentChunkingRunModel,
    DocumentParseArtifactModel,
    DocumentVersionModel,
    IngestionJobModel,
)
from app.infra.persistence.sql.repositories import (
    ControlRepository,
    ResourceConflictError,
    ResourceNotFoundError,
    canonical_json_sha256,
)
from app.security.principal import Principal
from app.security.signed_cursor import (
    SignedCursorError,
    decode_signed_cursor,
    encode_signed_cursor,
)
from app.shared.runtime.logger import PROJECT_ROOT


PARSE_ARTIFACT_ROOT = PROJECT_ROOT / "output" / "parsed-artifacts"
ORIGINALS_ROOT = PROJECT_ROOT / "output" / "originals"
_ARTIFACT_CURSOR_PURPOSE = "knowledge-workbench-artifact-v1"


def list_knowledge_bases(principal: Principal, **filters) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).list_knowledge_bases(**filters)


def get_knowledge_base(principal: Principal, kb_id: str) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).knowledge_base_detail(kb_id)


def update_knowledge_base(
    principal: Principal, kb_id: str, changes: dict[str, Any], if_match: str | None
) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).update_knowledge_base(
            kb_id, changes, if_match
        )


def list_documents(principal: Principal, kb_id: str, **filters) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).list_documents(
            kb_id, **filters
        )


def get_document(principal: Principal, document_id: str) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).document_detail(document_id)


def list_versions(principal: Principal, document_id: str) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).list_versions(document_id)


def get_version(principal: Principal, version_id: str) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        version, _, _ = repository.get_version(version_id)
        return repository.version_projection(version)


def list_jobs(principal: Principal, **filters) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).list_jobs(**filters)


def get_job_detail(principal: Principal, job_id: str) -> dict:
    with session_scope() as session:
        return KnowledgeWorkbenchRepository(session, principal).job_detail(job_id)


def _cache_key(artifact: DocumentParseArtifactModel) -> str:
    prefix = "parse-cache://"
    uri = str(artifact.artifact_uri or "")
    if not uri.startswith(prefix):
        raise DocumentParseError(
            "PARSE_ARTIFACT_ADDRESS_UNSUPPORTED",
            "解析制品没有可验证的逻辑地址",
            status_code=410,
        )
    value = uri[len(prefix) :]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DocumentParseError(
            "INVALID_CACHE_KEY", "解析制品缓存身份非法", status_code=410
        )
    return value


def _load_parsed_artifact(artifact: DocumentParseArtifactModel) -> ParsedDocument:
    key = _cache_key(artifact)
    store = LocalParseArtifactStore(PARSE_ARTIFACT_ROOT)
    loaded = store.load(key)
    if loaded is None:
        try:
            loaded = MinIOParseArtifactMirror(
                minio_gateway.client(), minio_gateway.bucket_name
            ).download_into(key, store)
        except Exception as exc:
            raise DocumentParseError(
                "PARSE_ARTIFACT_UNAVAILABLE",
                "解析制品已迁移、清理或当前不可读取",
                retryable=True,
                status_code=503,
            ) from exc
    if loaded is None:
        raise DocumentParseError(
            "PARSE_ARTIFACT_GONE",
            "解析制品已被清理",
            status_code=410,
        )
    document, info = loaded
    if artifact.artifact_sha256 and info.sha256 != artifact.artifact_sha256:
        raise DocumentParseError(
            "PARSE_ARTIFACT_INTEGRITY_ERROR",
            "解析制品 SHA-256 与控制面事实不一致",
            status_code=409,
        )
    return document


def _safe_artifact_path(run: DocumentChunkingRunModel) -> Path:
    raw = Path(str(run.artifact_uri or "")).expanduser()
    target = raw.resolve()
    root = PARSE_ARTIFACT_ROOT.resolve()
    if not target.is_file() or root not in target.parents:
        raise DocumentParseError(
            "CHUNK_ARTIFACT_GONE",
            "切片制品已被清理或不在受信目录",
            status_code=410,
        )
    return target


def _load_chunk_artifact(run: DocumentChunkingRunModel) -> list[dict[str, Any]]:
    target = _safe_artifact_path(run)
    payload = target.read_bytes()
    actual = sha256(payload).hexdigest()
    if run.artifact_sha256 and run.artifact_sha256 != actual:
        raise DocumentParseError(
            "CHUNK_ARTIFACT_INTEGRITY_ERROR",
            "切片制品 SHA-256 与控制面事实不一致",
            status_code=409,
        )
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise DocumentParseError(
            "CHUNK_ARTIFACT_SCHEMA_ERROR", "切片制品不是合法 JSON", status_code=409
        ) from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DocumentParseError(
            "CHUNK_ARTIFACT_SCHEMA_ERROR", "切片制品不是对象数组", status_code=409
        )
    return [dict(item) for item in value]


def _latest_parse_artifact(session, version_id: str) -> DocumentParseArtifactModel:
    artifact = session.scalar(
        select(DocumentParseArtifactModel)
        .where(
            DocumentParseArtifactModel.document_version_id == version_id,
            DocumentParseArtifactModel.status == "succeeded",
        )
        .order_by(
            DocumentParseArtifactModel.created_at.desc(),
            DocumentParseArtifactModel.id.desc(),
        )
    )
    if artifact is None:
        raise ResourceConflictError("document version has no succeeded parse artifact")
    return artifact


def _latest_chunk_run(session, version_id: str) -> DocumentChunkingRunModel:
    run = session.scalar(
        select(DocumentChunkingRunModel)
        .join(
            DocumentParseArtifactModel,
            DocumentParseArtifactModel.id == DocumentChunkingRunModel.parse_artifact_id,
        )
        .where(
            DocumentParseArtifactModel.document_version_id == version_id,
            DocumentChunkingRunModel.status == "succeeded",
        )
        .order_by(
            DocumentChunkingRunModel.created_at.desc(),
            DocumentChunkingRunModel.id.desc(),
        )
    )
    if run is None:
        raise ResourceConflictError("document version has no succeeded chunking run")
    return run


def _page_cursor(
    cursor: str | None,
    *,
    kind: str,
    artifact_hash: str,
    filters: dict[str, Any],
) -> int:
    if cursor is None:
        return 0
    try:
        payload = decode_signed_cursor(cursor, purpose=_ARTIFACT_CURSOR_PURPOSE)
    except SignedCursorError as exc:
        raise WorkbenchCursorError("artifact cursor is invalid") from exc
    expected_filter = canonical_json_sha256(filters)[:24]
    if (
        type(payload.get("v")) is not int
        or payload["v"] != 1
        or payload.get("kind") != kind
        or payload.get("artifact") != artifact_hash
        or payload.get("filter") != expected_filter
        or type(payload.get("offset")) is not int
        or payload["offset"] < 0
    ):
        raise WorkbenchCursorError("artifact cursor does not match this immutable view")
    return int(payload["offset"])


def _next_page_cursor(
    offset: int,
    *,
    kind: str,
    artifact_hash: str,
    filters: dict[str, Any],
) -> str:
    payload = {
        "v": 1,
        "kind": kind,
        "artifact": artifact_hash,
        "filter": canonical_json_sha256(filters)[:24],
        "offset": offset,
    }
    return encode_signed_cursor(payload, purpose=_ARTIFACT_CURSOR_PURPOSE)


def _paged(
    values: list[dict[str, Any]],
    *,
    cursor: str | None,
    limit: int,
    kind: str,
    artifact_hash: str,
    filters: dict[str, Any],
) -> dict:
    offset = _page_cursor(
        cursor, kind=kind, artifact_hash=artifact_hash, filters=filters
    )
    bounded = max(1, min(int(limit), 200))
    selected = values[offset : offset + bounded]
    next_cursor = (
        _next_page_cursor(
            offset + bounded,
            kind=kind,
            artifact_hash=artifact_hash,
            filters=filters,
        )
        if offset + bounded < len(values)
        else None
    )
    return {
        "schema_version": "workbench-page.v1",
        "items": selected,
        "total": len(values),
        "limit": bounded,
        "next_cursor": next_cursor,
    }


def list_blocks(
    principal: Principal,
    version_id: str,
    *,
    cursor: str | None,
    limit: int,
    q: str | None,
    block_type: str | None,
    page_no: int | None,
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        repository.get_version(version_id)
        artifact = _latest_parse_artifact(session, version_id)
        parsed = _load_parsed_artifact(artifact)
    filters = {
        "q": (q or "").strip().casefold(),
        "block_type": block_type or "",
        "page_no": page_no,
    }
    output: list[dict[str, Any]] = []
    assets_by_id = {item.asset_id: item for item in parsed.assets}
    for block in parsed.blocks:
        if block_type and block.type != block_type:
            continue
        if page_no and block.location.page_no != page_no:
            continue
        if q and q.strip() and q.strip().casefold() not in block.text.casefold():
            continue
        output.append(
            {
                "block_id": block.block_id,
                "type": block.type,
                "text": block.text,
                "structured_data": sanitize_public_json(block.structured_data),
                "location": block.location.model_dump(mode="json", exclude_none=True),
                "parent_heading_path": list(block.parent_heading_path),
                "order": block.order,
                "assets": [
                    {
                        "asset_id": asset_id,
                        "media_type": assets_by_id[asset_id].media_type,
                        "width": assets_by_id[asset_id].width,
                        "height": assets_by_id[asset_id].height,
                        "size_bytes": assets_by_id[asset_id].size_bytes,
                    }
                    for asset_id in block.asset_refs
                    if asset_id in assets_by_id
                ],
            }
        )
    return _paged(
        output,
        cursor=cursor,
        limit=limit,
        kind="blocks",
        artifact_hash=artifact.artifact_sha256 or sha256(canonical_json(parsed.model_dump(mode="json"))).hexdigest(),
        filters=filters,
    )


def _chunk_id(value: dict[str, Any]) -> str:
    return str(value.get("chunk_id") or value.get("id") or "").strip()


def _chunk_provenance(value: dict[str, Any]) -> dict[str, Any]:
    provenance = value.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _chunk_source_block_ids(value: dict[str, Any]) -> list[str]:
    provenance = _chunk_provenance(value)
    raw = value.get("source_block_ids") or provenance.get("source_block_ids") or []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw]


def _safe_chunk(value: dict[str, Any]) -> dict[str, Any]:
    content = str(value.get("content") or value.get("text") or "")
    provenance = _chunk_provenance(value)
    source_block_ids = _chunk_source_block_ids(value)
    location = (
        value.get("location")
        or value.get("provenance_spans")
        or provenance.get("locations")
        or []
    )
    parent_heading_path = (
        value.get("parent_heading_path")
        or provenance.get("heading_path")
        or []
    )
    return {
        "chunk_id": _chunk_id(value),
        "content": content,
        "type": str(value.get("type") or "text"),
        "source_block_ids": source_block_ids,
        "parent_heading_path": list(parent_heading_path),
        "location": sanitize_public_json(location),
        "metadata": sanitize_public_json(dict(value.get("metadata") or {})),
        "has_images": bool(value.get("images") or value.get("img_id") or value.get("img_ids")),
    }


def list_chunks(
    principal: Principal,
    version_id: str,
    *,
    cursor: str | None,
    limit: int,
    q: str | None,
    block_type: str | None,
    page_no: int | None,
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        repository.get_version(version_id)
        run = _latest_chunk_run(session, version_id)
        raw = _load_chunk_artifact(run)
    filters = {
        "q": (q or "").strip().casefold(),
        "block_type": block_type or "",
        "page_no": page_no,
    }
    output = []
    for item in raw:
        safe = _safe_chunk(item)
        if block_type and safe["type"] != block_type:
            continue
        if q and q.strip() and q.strip().casefold() not in safe["content"].casefold():
            continue
        if page_no:
            locations = safe["location"] if isinstance(safe["location"], list) else [safe["location"]]
            if not any(
                isinstance(location, dict) and location.get("page_no") == page_no
                for location in locations
            ):
                continue
        output.append(safe)
    return _paged(
        output,
        cursor=cursor,
        limit=limit,
        kind="chunks",
        artifact_hash=run.artifact_sha256 or sha256(canonical_json(raw)).hexdigest(),
        filters=filters,
    )


def _chunk_alignment_key(value: dict[str, Any]) -> str:
    source_ids = sorted(_chunk_source_block_ids(value))
    if source_ids:
        return "blocks:" + "|".join(source_ids)
    return "chunk:" + _chunk_id(value)


def _aligned_chunks(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep repeated provenance groups distinct using their stable occurrence order."""

    occurrences: dict[str, int] = {}
    aligned: dict[str, dict[str, Any]] = {}
    for value in values:
        base = _chunk_alignment_key(value)
        occurrence = occurrences.get(base, 0) + 1
        occurrences[base] = occurrence
        aligned[f"{base}#{occurrence}"] = value
    return aligned


def version_diff(
    principal: Principal,
    left_version_id: str,
    right_version_id: str,
    *,
    cursor: str | None,
    limit: int,
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        left, left_document, _ = repository.get_version(left_version_id)
        right, right_document, _ = repository.get_version(right_version_id)
        if left_document.id != right_document.id:
            raise ResourceConflictError("version diff requires two versions of one document")
        left_run = _latest_chunk_run(session, left.id)
        right_run = _latest_chunk_run(session, right.id)
        left_chunks = _load_chunk_artifact(left_run)
        right_chunks = _load_chunk_artifact(right_run)
        left_projection = repository.version_projection(left)
        right_projection = repository.version_projection(right)
    before = _aligned_chunks(left_chunks)
    after = _aligned_chunks(right_chunks)
    hunks: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        old_content = str((old or {}).get("content") or "")
        new_content = str((new or {}).get("content") or "")
        if old is not None and new is not None and old_content == new_content:
            continue
        kind = "added" if old is None else "removed" if new is None else "modified"
        hunks.append(
            {
                "alignment_key": key,
                "kind": kind,
                "before": (
                    {
                        **_safe_chunk(old),
                        "content_hash": sha256(old_content.encode("utf-8")).hexdigest(),
                    }
                    if old
                    else None
                ),
                "after": (
                    {
                        **_safe_chunk(new),
                        "content_hash": sha256(new_content.encode("utf-8")).hexdigest(),
                    }
                    if new
                    else None
                ),
            }
        )
    artifact_hash = sha256(
        f"{left_run.artifact_sha256}|{right_run.artifact_sha256}".encode("utf-8")
    ).hexdigest()
    page = _paged(
        hunks,
        cursor=cursor,
        limit=limit,
        kind="version-diff",
        artifact_hash=artifact_hash,
        filters={"left": left.id, "right": right.id},
    )
    return {
        "schema_version": "document-version-diff.v1",
        "document_id": left_document.id,
        "left": left_projection,
        "right": right_projection,
        "summary": {
            "left_chunk_count": len(left_chunks),
            "right_chunk_count": len(right_chunks),
            "changed_hunk_count": len(hunks),
            "parser_config_changed": left_projection["parser_config"]
            != right_projection["parser_config"],
            "content_changed": left.content_sha256 != right.content_sha256,
        },
        "hunks": page,
    }


def source_file(principal: Principal, version_id: str) -> tuple[Path, dict[str, Any]]:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        version, document, _ = repository.get_version(version_id)
        try:
            source = resolve_immutable_local_source(
                object_key=version.object_key,
                expected_size=version.size_bytes,
                expected_sha256=version.content_sha256,
                originals_root=ORIGINALS_ROOT,
            )
        except ImmutableSourceUnavailableError as exc:
            raise ResourceNotFoundError(
                "source object is not locally available"
            ) from exc
        except ImmutableSourceIntegrityError as exc:
            raise DocumentParseError(
                "SOURCE_OBJECT_INTEGRITY_ERROR",
                "源文档内容与不可变版本事实不一致",
                status_code=409,
            ) from exc
        return source.path, {
            "filename": document.logical_name,
            "media_type": version.mime_type,
            "etag": f'"sha256:{version.content_sha256}"',
            "size": source.size,
        }


def batch_document_action(
    principal: Principal, action: str, items: Iterable[dict[str, str]]
) -> dict:
    results: list[dict[str, Any]] = []
    for item in items:
        document_id = item["document_id"]
        try:
            with session_scope() as session:
                repository = KnowledgeWorkbenchRepository(session, principal)
                document, kb = repository.get_document(document_id)
                repository.assert_document_etag(document, item.get("etag"))
                if action == "enable":
                    repository.assert_action(kb, "enable_document")
                    result = IndexLifecycleRepository(session).set_document_enabled(
                        document.id, True
                    )
                    value = {"document": {"id": result["id"], "status": result["status"]}}
                elif action == "disable":
                    repository.assert_action(kb, "disable_document")
                    result = IndexLifecycleRepository(session).set_document_enabled(
                        document.id, False
                    )
                    value = {"document": {"id": result["id"], "status": result["status"]}}
                elif action in {"soft_delete", "purge"}:
                    repository.assert_action(kb, "delete_document")
                    value = IndexLifecycleRepository(session).soft_delete_document(
                        document.id, "purge" if action == "purge" else "soft"
                    )
                    value = {
                        "document": {
                            "id": value["document"]["id"],
                            "status": value["document"]["status"],
                        },
                        "cleanup_job": {
                            "id": value["cleanup_job"]["id"],
                            "status": value["cleanup_job"]["status"],
                            "mode": value["cleanup_job"]["mode"],
                        },
                    }
                elif action == "retry":
                    if (
                        not repository._has_permission("job:manage")
                        or repository._grant_role(kb)
                        not in {"owner", "admin", "editor"}
                    ):
                        raise ResourceNotFoundError("workbench resource not found")
                    latest = session.scalar(
                        select(IngestionJobModel)
                        .join(
                            DocumentVersionModel,
                            DocumentVersionModel.id == IngestionJobModel.document_version_id,
                        )
                        .where(DocumentVersionModel.document_id == document.id)
                        .order_by(
                            DocumentVersionModel.version_no.desc(),
                            IngestionJobModel.attempt.desc(),
                        )
                    )
                    if latest is None or latest.status not in {"failed", "cancelled"}:
                        raise ResourceConflictError("document has no retryable ingestion job")
                    job = ControlRepository(session).create_job_for_version(
                        latest.document_version_id
                    )
                    job_row = session.get(IngestionJobModel, job["id"])
                    assert job_row is not None
                    chunk_draft = session.scalar(
                        select(ChunkEditDraftModel).where(
                            ChunkEditDraftModel.published_document_version_id
                            == latest.document_version_id
                        )
                    )
                    if chunk_draft is not None:
                        if chunk_draft.status not in {"publishing", "indexed"}:
                            raise ResourceConflictError(
                                "published chunk draft cannot be rebound to a retry job"
                            )
                        chunk_draft.published_job_id = job_row.id
                    value = {"job": repository._safe_job(job_row)}
                else:
                    raise ResourceConflictError("unsupported document batch action")
                session.flush()
                current_etag = (
                    document_etag(document) if document.status != "deleted" else None
                )
                results.append(
                    {
                        "document_id": document_id,
                        "ok": True,
                        "status": 200,
                        "etag": current_etag,
                        "result": value,
                    }
                )
        except WorkbenchPreconditionError as exc:
            results.append(
                {
                    "document_id": document_id,
                    "ok": False,
                    "status": 428 if exc.required else 412,
                    "error": {
                        "code": "PRECONDITION_REQUIRED"
                        if exc.required
                        else "ETAG_MISMATCH",
                        "message": str(exc),
                        "current_etag": exc.current_etag,
                    },
                }
            )
        except ResourceNotFoundError:
            results.append(
                {
                    "document_id": document_id,
                    "ok": False,
                    "status": 404,
                    "error": {"code": "NOT_FOUND", "message": "resource not found"},
                }
            )
        except ResourceConflictError as exc:
            results.append(
                {
                    "document_id": document_id,
                    "ok": False,
                    "status": 409,
                    "error": {"code": "STATE_CONFLICT", "message": str(exc)},
                }
            )
        except StaleDataError:
            results.append(
                {
                    "document_id": document_id,
                    "ok": False,
                    "status": 412,
                    "error": {
                        "code": "ETAG_MISMATCH",
                        "message": "document changed during the conditional update",
                    },
                }
            )
    return {
        "schema_version": "document-batch-result.v1",
        "action": action,
        "items": results,
        "succeeded": sum(1 for item in results if item["ok"]),
        "failed": sum(1 for item in results if not item["ok"]),
    }


def _operation_output(value: ChunkReplacement) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude_none=True)


def _generated_chunk_id(
    draft_id: str,
    sequence: int,
    ordinal: int,
    inputs: list[str],
    content: str,
) -> str:
    payload = json.dumps(
        [draft_id, sequence, ordinal, inputs, content],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "chk_" + sha256(payload.encode("utf-8")).hexdigest()[:40]


def _strip_runtime_fields(value: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "dense_vector",
        "sparse_vector",
        "content_ltks",
        "content_sm_ltks",
        "question_tks",
        "question_kwd",
        "key_tokens",
        "key_words",
        "artifact_uri",
        "collection_alias",
        "collection_name",
        "document_version_id",
        "index_version_id",
        "lease_token",
        "local_uri",
        "object_key",
        "object_keys",
        "source_uri",
        "create_time",
        "create_timestamp_flt",
    }
    return {key: deepcopy(item) for key, item in value.items() if key not in excluded}


def apply_chunk_operations(
    base_chunks: list[dict[str, Any]],
    operations: Iterable[ChunkEditOperationModel | dict[str, Any]],
    *,
    draft_id: str,
) -> list[dict[str, Any]]:
    chunks = [_strip_runtime_fields(item) for item in base_chunks]
    for raw in operations:
        sequence = int(raw.sequence if isinstance(raw, ChunkEditOperationModel) else raw["sequence"])
        operation = str(raw.operation if isinstance(raw, ChunkEditOperationModel) else raw["operation"])
        inputs = list(
            raw.input_chunk_ids_json
            if isinstance(raw, ChunkEditOperationModel)
            else raw["input_chunk_ids"]
        )
        outputs = list(
            raw.output_chunks_json
            if isinstance(raw, ChunkEditOperationModel)
            else raw.get("output_chunks", [])
        )
        positions = {identifier: index for index, identifier in enumerate(map(_chunk_id, chunks))}
        if any(identifier not in positions for identifier in inputs):
            raise ResourceConflictError(
                f"chunk edit operation {sequence} references a missing chunk"
            )
        indexes = [positions[identifier] for identifier in inputs]
        if operation in {"replace", "split", "delete"} and len(indexes) != 1:
            raise ResourceConflictError(f"{operation} requires exactly one input chunk")
        if operation == "merge" and indexes != list(range(min(indexes), max(indexes) + 1)):
            raise ResourceConflictError("merge inputs must be adjacent and ordered")
        if operation == "replace" and len(outputs) != 1:
            raise ResourceConflictError("replace requires exactly one output chunk")
        if operation == "split" and not 2 <= len(outputs) <= 20:
            raise ResourceConflictError("split requires 2..20 output chunks")
        if operation == "merge" and len(outputs) != 1:
            raise ResourceConflictError("merge requires exactly one output chunk")
        if operation == "delete" and outputs:
            raise ResourceConflictError("delete cannot contain output chunks")
        anchor = deepcopy(chunks[min(indexes)])
        source_ids: list[str] = []
        images: list[Any] = []
        for index in indexes:
            for source_id in chunks[index].get("source_block_ids") or []:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            for image in chunks[index].get("images") or []:
                if image not in images:
                    images.append(image)
        replacements: list[dict[str, Any]] = []
        for ordinal, output in enumerate(outputs, start=1):
            candidate = _strip_runtime_fields(anchor)
            candidate.update(
                {
                    "content": str(output["content"]),
                    "type": output.get("type") or candidate.get("type") or "text",
                    "metadata": dict(output.get("metadata") or {}),
                    "source_block_ids": list(output.get("source_block_ids") or source_ids),
                    "parent_heading_path": list(
                        output.get("parent_heading_path")
                        or candidate.get("parent_heading_path")
                        or []
                    ),
                    "images": images or candidate.get("images"),
                }
            )
            identifier = _generated_chunk_id(
                draft_id, sequence, ordinal, inputs, candidate["content"]
            )
            candidate["id"] = identifier
            candidate["chunk_id"] = identifier
            replacements.append(candidate)
        start = min(indexes)
        end = max(indexes) + 1
        chunks[start:end] = replacements
    if not chunks:
        raise ResourceConflictError("chunk draft cannot delete every chunk")
    identifiers = [_chunk_id(item) for item in chunks]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ResourceConflictError("materialized chunk ids must be present and unique")
    return chunks


def create_chunk_draft(
    principal: Principal, version_id: str, base_chunking_run_id: str | None
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        draft = repository.create_chunk_draft(version_id, base_chunking_run_id)
        return repository.draft_projection(draft)


def get_chunk_draft(principal: Principal, draft_id: str) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        draft, _ = repository.get_chunk_draft(draft_id)
        run = session.get(DocumentChunkingRunModel, draft.base_chunking_run_id)
        assert run is not None
        materialized = apply_chunk_operations(
            _load_chunk_artifact(run),
            repository.chunk_operations(draft.id),
            draft_id=draft.id,
        )
        result = repository.draft_projection(draft)
        result["materialized_chunk_count"] = len(materialized)
        return result


def list_chunk_draft_chunks(
    principal: Principal,
    draft_id: str,
    *,
    cursor: str | None,
    limit: int,
    q: str | None,
    block_type: str | None,
    page_no: int | None,
) -> dict:
    """Read the current immutable materialization of an append-only draft."""

    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        draft, kb = repository.get_chunk_draft(draft_id)
        repository.assert_action(kb, "read")
        run = session.get(DocumentChunkingRunModel, draft.base_chunking_run_id)
        assert run is not None
        materialized = apply_chunk_operations(
            _load_chunk_artifact(run),
            repository.chunk_operations(draft.id),
            draft_id=draft.id,
        )
        artifact_hash = sha256(
            f"{run.artifact_sha256 or ''}|{draft.operations_hash or ''}|r{draft.revision}".encode(
                "utf-8"
            )
        ).hexdigest()
    filters = {
        "draft_id": draft_id,
        "q": (q or "").strip().casefold(),
        "block_type": block_type or "",
        "page_no": page_no,
    }
    output: list[dict[str, Any]] = []
    for item in materialized:
        safe = _safe_chunk(item)
        if block_type and safe["type"] != block_type:
            continue
        if q and q.strip() and q.strip().casefold() not in safe["content"].casefold():
            continue
        if page_no:
            locations = (
                safe["location"]
                if isinstance(safe["location"], list)
                else [safe["location"]]
            )
            if not any(
                isinstance(location, dict) and location.get("page_no") == page_no
                for location in locations
            ):
                continue
        output.append(safe)
    return _paged(
        output,
        cursor=cursor,
        limit=limit,
        kind="chunk-draft-chunks",
        artifact_hash=artifact_hash,
        filters=filters,
    )


def _append_operation(
    principal: Principal,
    draft_id: str,
    *,
    if_match: str | None,
    operation: str,
    input_chunk_ids: list[str],
    output_chunks: list[ChunkReplacement],
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        draft, _ = repository.get_chunk_draft(draft_id)
        repository.assert_chunk_draft_etag(draft, if_match)
        run = session.get(DocumentChunkingRunModel, draft.base_chunking_run_id)
        assert run is not None
        current_operations = repository.chunk_operations(draft.id)
        current = apply_chunk_operations(
            _load_chunk_artifact(run), current_operations, draft_id=draft.id
        )
        preview = {
            "sequence": len(current_operations) + 1,
            "operation": operation,
            "input_chunk_ids": input_chunk_ids,
            "output_chunks": [_operation_output(item) for item in output_chunks],
        }
        # Validate adjacency, cardinality, existence and non-empty final state before
        # the append-only fact is persisted.
        apply_chunk_operations(current, [preview], draft_id=draft.id)
        updated = repository.append_chunk_operation(
            draft.id,
            if_match=if_match,
            operation=operation,
            input_chunk_ids=input_chunk_ids,
            output_chunks=preview["output_chunks"],
        )
        result = repository.draft_projection(updated)
        result["materialized_chunk_count"] = len(
            apply_chunk_operations(current, [preview], draft_id=draft.id)
        )
        result["operation_result"] = {
            "input_chunk_ids": list(input_chunk_ids),
            "output_chunk_ids": [
                _generated_chunk_id(
                    draft.id,
                    int(preview["sequence"]),
                    ordinal,
                    input_chunk_ids,
                    str(output["content"]),
                )
                for ordinal, output in enumerate(
                    preview["output_chunks"], start=1
                )
            ],
        }
        return result


def replace_chunk(
    principal: Principal,
    draft_id: str,
    chunk_id: str,
    chunk: ChunkReplacement,
    if_match: str | None,
) -> dict:
    return _append_operation(
        principal,
        draft_id,
        if_match=if_match,
        operation="replace",
        input_chunk_ids=[chunk_id],
        output_chunks=[chunk],
    )


def split_chunk(
    principal: Principal,
    draft_id: str,
    chunk_id: str,
    chunks: list[ChunkReplacement],
    if_match: str | None,
) -> dict:
    return _append_operation(
        principal,
        draft_id,
        if_match=if_match,
        operation="split",
        input_chunk_ids=[chunk_id],
        output_chunks=chunks,
    )


def merge_chunks(
    principal: Principal,
    draft_id: str,
    chunk_ids: list[str],
    chunk: ChunkReplacement,
    if_match: str | None,
) -> dict:
    return _append_operation(
        principal,
        draft_id,
        if_match=if_match,
        operation="merge",
        input_chunk_ids=chunk_ids,
        output_chunks=[chunk],
    )


def delete_chunk(
    principal: Principal,
    draft_id: str,
    chunk_id: str,
    if_match: str | None,
) -> dict:
    return _append_operation(
        principal,
        draft_id,
        if_match=if_match,
        operation="delete",
        input_chunk_ids=[chunk_id],
        output_chunks=[],
    )


def publish_chunk_draft(
    principal: Principal,
    draft_id: str,
    *,
    if_match: str | None,
    note: str,
) -> dict:
    with session_scope() as session:
        repository = KnowledgeWorkbenchRepository(session, principal)
        draft, kb = repository.get_chunk_draft(draft_id)
        repository.assert_action(kb, "create_chunk_draft")
        if draft.status != "draft":
            result = repository.draft_projection(draft)
            if draft.published_job_id:
                job = session.get(IngestionJobModel, draft.published_job_id)
                result["job"] = repository._safe_job(job) if job else None
            return result
        supplied = (if_match or "").strip()
        if not supplied:
            raise WorkbenchPreconditionError("If-Match is required", required=True)
        if supplied != draft_etag(draft):
            raise WorkbenchPreconditionError(
                "chunk draft changed since it was read",
                required=False,
                current_etag=draft_etag(draft),
            )
        operations = repository.chunk_operations(draft.id)
        if not operations:
            raise ResourceConflictError("empty chunk draft cannot be published")
        base_run = session.get(DocumentChunkingRunModel, draft.base_chunking_run_id)
        assert base_run is not None
        materialized = apply_chunk_operations(
            _load_chunk_artifact(base_run), operations, draft_id=draft.id
        )
        if not materialized:
            raise ResourceConflictError("chunk draft cannot publish an empty document")
        base_version = session.get(DocumentVersionModel, draft.document_version_id)
        assert base_version is not None
        document = ControlRepository(session).get_document_model(draft.document_id)
        config = dict(base_version.parser_config_snapshot or {})
        operation_snapshot = [
            {
                "sequence": item.sequence,
                "operation": item.operation,
                "inputs": item.input_chunk_ids_json,
                "outputs": item.output_chunks_json,
            }
            for item in operations
        ]
        operations_hash = canonical_json_sha256(operation_snapshot)
        config.update(
            {
                "chunk_override_draft_id": draft.id,
                "chunk_override_revision": draft.revision,
                "chunk_override_operations_hash": operations_hash,
                "chunk_override_note": note.strip(),
            }
        )
        fingerprint = sha256(
            json.dumps(
                {
                    "draft_id": draft.id,
                    "revision": draft.revision,
                    "operations_hash": operations_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        registered = ControlRepository(session).register_upload_and_job(
            kb_id=draft.kb_id,
            logical_name=document.logical_name,
            content_sha256=base_version.content_sha256,
            size_bytes=base_version.size_bytes,
            mime_type=base_version.mime_type,
            object_key=base_version.object_key,
            parser_config_snapshot=config,
            source_uri=document.source_uri,
            document_id=document.id,
            idempotency_key=f"chunk-draft:{draft.id}:r{draft.revision}",
            request_fingerprint=fingerprint,
        )
        version = registered["document_version"]
        job = registered["ingestion_job"]
        index = registered["index_version"]
        job_row = session.get(IngestionJobModel, job["id"])
        assert job_row is not None
        job_row.operation = "chunk_override"
        draft.status = "publishing"
        draft.operations_hash = operations_hash
        draft.frozen_revision = draft.revision
        draft.published_document_version_id = version["id"]
        draft.published_index_version_id = index["id"]
        draft.published_job_id = job["id"]
        try:
            session.flush()
        except StaleDataError as exc:
            raise WorkbenchPreconditionError(
                "chunk draft changed during publication", required=False
            ) from exc
        result = repository.draft_projection(draft)
        result.update(
            {
                "document_version": {
                    "id": version["id"],
                    "version_no": version["version_no"],
                    "status": version["status"],
                },
                "index_version": {
                    "id": index["id"],
                    "version_no": index["version_no"],
                    "status": index["status"],
                },
                "job": repository._safe_job(job_row),
            }
        )
        return result


def apply_chunk_override_for_worker(
    state: dict[str, Any], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fail-closed worker hook executed immediately before embedding."""

    config = dict(state.get("parser_config") or {})
    draft_id = str(config.get("chunk_override_draft_id") or "").strip()
    if not draft_id:
        return chunks
    version_id = str(state.get("document_version_id") or "").strip()
    with session_scope() as session:
        draft = session.get(ChunkEditDraftModel, draft_id)
        if (
            draft is None
            or draft.status not in {"publishing", "indexed", "published"}
            or draft.published_document_version_id != version_id
            or draft.operations_hash
            != str(config.get("chunk_override_operations_hash") or "")
            or draft.frozen_revision
            != int(config.get("chunk_override_revision") or 0)
        ):
            raise ResourceConflictError("chunk override publication identity is stale")
        operations = session.scalars(
            select(ChunkEditOperationModel)
            .where(ChunkEditOperationModel.draft_id == draft.id)
            .order_by(ChunkEditOperationModel.sequence.asc())
        ).all()
        materialized = apply_chunk_operations(chunks, operations, draft_id=draft.id)
        revision = draft.frozen_revision
        if revision is None:
            raise ResourceConflictError("chunk override publication revision is missing")
        operations_hash = draft.operations_hash or canonical_json_sha256([])
    payload = canonical_json(materialized)
    digest = sha256(payload).hexdigest()
    directory = (PARSE_ARTIFACT_ROOT / "workbench" / draft_id).resolve()
    root = PARSE_ARTIFACT_ROOT.resolve()
    if root not in directory.parents:
        raise ResourceConflictError("chunk override artifact path escaped trusted root")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"r{revision}-{operations_hash[:16]}.json"
    temporary = directory / f".tmp-{os.getpid()}-{uuid4().hex[:12]}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    summary = dict(state.get("parsed_document_summary") or {})
    parse_artifact_id = str(summary.get("parse_artifact_id") or "")
    if not parse_artifact_id:
        raise ResourceConflictError("chunk override requires a persisted parse artifact")
    with session_scope() as session:
        parse_artifact = session.get(DocumentParseArtifactModel, parse_artifact_id)
        if (
            parse_artifact is None
            or parse_artifact.document_version_id != version_id
            or parse_artifact.status != "succeeded"
        ):
            raise ResourceConflictError(
                "chunk override parse artifact does not belong to the published version"
            )
    from app.application.control_plane_service import record_document_chunking_run

    run = record_document_chunking_run(
        parse_artifact_id=parse_artifact_id,
        template_id="workbench-manual",
        template_version="1.0.0",
        config_hash=operations_hash,
        status="succeeded",
        artifact_uri=str(target),
        artifact_sha256=digest,
        artifact_size=len(payload),
        chunk_count=len(materialized),
    )
    summary.update(
        {
            "chunking_run_id": run["id"],
            "chunk_override_draft_id": draft_id,
            "chunk_override_revision": revision,
            "chunk_artifact": {
                "uri": str(target),
                "sha256": digest,
                "size_bytes": len(payload),
                "config_hash": operations_hash,
            },
        }
    )
    state["parsed_document_summary"] = summary
    return materialized


__all__ = [
    "WorkbenchCursorError",
    "WorkbenchPreconditionError",
    "apply_chunk_operations",
    "apply_chunk_override_for_worker",
    "batch_document_action",
    "create_chunk_draft",
    "delete_chunk",
    "get_chunk_draft",
    "get_document",
    "get_job_detail",
    "get_knowledge_base",
    "get_version",
    "list_blocks",
    "list_chunks",
    "list_chunk_draft_chunks",
    "list_documents",
    "list_jobs",
    "list_knowledge_bases",
    "list_versions",
    "merge_chunks",
    "publish_chunk_draft",
    "replace_chunk",
    "source_file",
    "split_chunk",
    "update_knowledge_base",
    "version_diff",
]
