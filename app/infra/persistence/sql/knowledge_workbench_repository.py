"""Authorization-first read/write repository for the TODO-12 import workbench."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any, Iterable

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from app.infra.persistence.sql.models import (
    ChunkEditDraftModel,
    ChunkEditOperationModel,
    DocumentChunkingRunModel,
    DocumentIndexBindingModel,
    DocumentModel,
    DocumentParseArtifactModel,
    DocumentVersionModel,
    IndexBuildManifestModel,
    IndexVersionModel,
    IngestionJobCheckpointModel,
    IngestionJobEventModel,
    IngestionJobModel,
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
    canonical_json_sha256,
    new_id,
)
from app.security.permissions import effective_permissions
from app.security.principal import Principal
from app.security.signed_cursor import (
    SignedCursorError,
    decode_signed_cursor,
    encode_signed_cursor,
)


READ_GRANT_ROLES = frozenset({"owner", "admin", "editor", "reader", "query"})
WRITE_GRANT_ROLES = frozenset({"owner", "admin", "editor"})
ADMIN_GRANT_ROLES = frozenset({"owner", "admin"})
_UNSET = object()
_INTERNAL_PUBLIC_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "artifact_uri",
        "bucket",
        "bucket_name",
        "collection_alias",
        "collection_name",
        "credential",
        "lease_token",
        "local_uri",
        "path",
        "file_path",
        "object_key",
        "object_keys",
        "password",
        "refresh_token",
        "request_fingerprint",
        "secret",
        "source_uri",
        "token",
        "uri",
    }
)


def sanitize_public_json(value: Any) -> Any:
    """Recursively remove infrastructure locators and credentials from public JSON."""

    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if (
                normalized in _INTERNAL_PUBLIC_KEYS
                or normalized.endswith(
                    ("_password", "_secret", "_token", "_uri", "_path", "_bucket")
                )
            ):
                continue
            output[str(key)] = sanitize_public_json(item)
        return output
    if isinstance(value, (list, tuple)):
        return [sanitize_public_json(item) for item in value]
    return value


class WorkbenchCursorError(ValueError):
    pass


class WorkbenchPreconditionError(RuntimeError):
    def __init__(self, message: str, *, required: bool, current_etag: str | None = None):
        super().__init__(message)
        self.required = required
        self.current_etag = current_etag


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _filter_hash(value: object) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()[:24]


_KEYSET_CURSOR_PURPOSE = "knowledge-workbench-keyset-v1"


def _encode_cursor(*, kind: str, sort: str, filters: object, values: list[object]) -> str:
    payload = {
        "v": 1,
        "k": kind,
        "s": sort,
        "f": _filter_hash(filters),
        "p": values,
    }
    return encode_signed_cursor(payload, purpose=_KEYSET_CURSOR_PURPOSE)


def _strict_iso_datetime(value: object) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise WorkbenchCursorError("cursor position is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkbenchCursorError("cursor position is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkbenchCursorError("cursor position is invalid")
    return parsed


def _decode_cursor(
    cursor: str | None, *, kind: str, sort: str, filters: object
) -> list[object] | None:
    if cursor is None:
        return None
    try:
        payload = decode_signed_cursor(cursor, purpose=_KEYSET_CURSOR_PURPOSE)
    except SignedCursorError as exc:
        raise WorkbenchCursorError("cursor is invalid") from exc
    if (
        type(payload.get("v")) is not int
        or payload["v"] != 1
        or payload.get("k") != kind
        or payload.get("s") != sort
        or payload.get("f") != _filter_hash(filters)
        or not isinstance(payload.get("p"), list)
        or len(payload["p"]) != 2
    ):
        raise WorkbenchCursorError("cursor does not belong to this filtered stable ordering")
    position = payload["p"]
    position_id = position[1]
    if not isinstance(position_id, str) or not position_id or len(position_id) > 255:
        raise WorkbenchCursorError("cursor position is invalid")
    if sort == "name_asc":
        if not isinstance(position[0], str) or not position[0] or len(position[0]) > 1000:
            raise WorkbenchCursorError("cursor position is invalid")
    else:
        _strict_iso_datetime(position[0])
    return list(position)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _normalized_etag(value: str | None) -> str:
    if not value:
        raise WorkbenchPreconditionError("If-Match is required", required=True)
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    return normalized


def kb_etag(row: KnowledgeBaseModel) -> str:
    return f'"kb:{row.id}:r{row.revision}"'


def document_etag(row: DocumentModel) -> str:
    return f'"document:{row.id}:r{row.revision}"'


def draft_etag(row: ChunkEditDraftModel) -> str:
    return f'"chunk-draft:{row.id}:r{row.revision}"'


class KnowledgeWorkbenchRepository:
    def __init__(self, session: Session, principal: Principal):
        self.session = session
        self.principal = principal
        self.permissions = effective_permissions(principal.roles, principal.scopes)

    def _has_permission(self, action: str) -> bool:
        return "*" in self.permissions or action in self.permissions

    def _service_scope_ids(self) -> tuple[bool, set[str]]:
        scopes = set(self.principal.scopes)
        wildcard = "*" in scopes or "kb:*:read" in scopes
        identifiers: set[str] = set()
        for value in scopes:
            parts = value.split(":")
            if len(parts) == 3 and parts[0] == "kb" and parts[2] in {"read", "query"}:
                identifiers.add(parts[1])
        return wildcard, identifiers

    def _visible_kb_predicate(self):
        base = [
            KnowledgeBaseModel.tenant_id == self.principal.tenant_id,
            KnowledgeBaseModel.status != "deleted",
        ]
        if "TenantAdmin" in self.principal.roles:
            return and_(*base)
        if self.principal.auth_type == "api_key":
            wildcard, identifiers = self._service_scope_ids()
            if wildcard:
                return and_(*base)
            if not identifiers:
                return and_(*base, KnowledgeBaseModel.id == "__denied__")
            return and_(*base, KnowledgeBaseModel.id.in_(identifiers))
        acl = (
            select(KnowledgeBaseAclModel.id)
            .where(
                KnowledgeBaseAclModel.kb_id == KnowledgeBaseModel.id,
                func.lower(KnowledgeBaseAclModel.role).in_(READ_GRANT_ROLES),
                or_(
                    and_(
                        KnowledgeBaseAclModel.principal_type == "user",
                        KnowledgeBaseAclModel.principal_id == self.principal.user_id,
                    ),
                    and_(
                        KnowledgeBaseAclModel.principal_type == "role",
                        KnowledgeBaseAclModel.principal_id.in_(self.principal.roles),
                    ),
                ),
            )
            .exists()
        )
        return and_(
            *base,
            or_(KnowledgeBaseModel.created_by == self.principal.user_id, acl),
        )

    def _grant_role(self, kb: KnowledgeBaseModel) -> str:
        if "TenantAdmin" in self.principal.roles:
            return "admin"
        if kb.created_by == self.principal.user_id:
            return "owner"
        if self.principal.auth_type == "api_key":
            return "query"
        rows = self.session.scalars(
            select(KnowledgeBaseAclModel)
            .where(
                KnowledgeBaseAclModel.kb_id == kb.id,
                func.lower(KnowledgeBaseAclModel.role).in_(READ_GRANT_ROLES),
                or_(
                    and_(
                        KnowledgeBaseAclModel.principal_type == "user",
                        KnowledgeBaseAclModel.principal_id == self.principal.user_id,
                    ),
                    and_(
                        KnowledgeBaseAclModel.principal_type == "role",
                        KnowledgeBaseAclModel.principal_id.in_(self.principal.roles),
                    ),
                ),
            )
            .order_by(KnowledgeBaseAclModel.created_at.asc())
        ).all()
        rank = {"owner": 0, "admin": 1, "editor": 2, "reader": 3, "query": 4}
        return min((str(row.role).lower() for row in rows), key=lambda item: rank[item])

    def resource_actions(self, kb: KnowledgeBaseModel) -> list[str]:
        role = self._grant_role(kb)
        scoped_service_read = False
        if self.principal.auth_type == "api_key":
            wildcard, identifiers = self._service_scope_ids()
            scoped_service_read = wildcard or kb.id in identifiers
        actions = (
            ["read"]
            if self._has_permission("kb:read") or scoped_service_read
            else []
        )
        if role in ADMIN_GRANT_ROLES and self._has_permission("kb:update"):
            actions.append("update")
        if role in WRITE_GRANT_ROLES and self._has_permission("document:upload"):
            actions.extend(["upload_document", "create_chunk_draft"])
        if role in ADMIN_GRANT_ROLES and self._has_permission("document:delete"):
            actions.extend(["enable_document", "disable_document", "delete_document"])
        if role in ADMIN_GRANT_ROLES and self._has_permission("index:publish"):
            actions.append("publish_index")
        if role in ADMIN_GRANT_ROLES and self._has_permission("kb:delete"):
            actions.append("delete")
        return sorted(set(actions))

    def assert_action(self, kb: KnowledgeBaseModel, action: str) -> None:
        if action not in self.resource_actions(kb):
            # Do not reveal whether the KB exists but is outside this principal's grant.
            raise ResourceNotFoundError("workbench resource not found")

    def get_kb(self, kb_id: str, *, include_deleted: bool = False) -> KnowledgeBaseModel:
        conditions = [KnowledgeBaseModel.id == kb_id, self._visible_kb_predicate()]
        if include_deleted:
            # Deleted resources are only needed for already-created cleanup facts.  The
            # workbench itself never re-opens deleted KBs.
            conditions = [
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == self.principal.tenant_id,
            ]
        row = self.session.scalar(select(KnowledgeBaseModel).where(*conditions))
        if row is None:
            raise ResourceNotFoundError("workbench resource not found")
        return row

    def _kb_aggregate_columns(self):
        document_count = (
            select(func.count(DocumentModel.id))
            .where(
                DocumentModel.kb_id == KnowledgeBaseModel.id,
                DocumentModel.status != "deleted",
            )
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        active_bytes = (
            select(func.coalesce(func.sum(DocumentVersionModel.size_bytes), 0))
            .join(DocumentModel, DocumentModel.active_version_id == DocumentVersionModel.id)
            .where(
                DocumentModel.kb_id == KnowledgeBaseModel.id,
                DocumentModel.status.in_({"ready", "processing", "disabled"}),
            )
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        active_chunks = (
            select(func.coalesce(func.sum(DocumentIndexBindingModel.chunk_count), 0))
            .join(
                DocumentModel,
                DocumentModel.active_version_id
                == DocumentIndexBindingModel.document_version_id,
            )
            .join(
                IndexVersionModel,
                IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
            )
            .where(
                DocumentModel.kb_id == KnowledgeBaseModel.id,
                DocumentIndexBindingModel.status == "published",
                IndexVersionModel.status == "active",
                IndexVersionModel.active_slot == "active",
            )
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        processing_count = (
            select(func.count(IngestionJobModel.id))
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .where(
                DocumentModel.kb_id == KnowledgeBaseModel.id,
                IngestionJobModel.status.in_(
                    {"queued", "leased", "running", "retry_wait", "processing"}
                ),
            )
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        latest_failure = (
            select(IngestionJobModel.error_code)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .where(
                DocumentModel.kb_id == KnowledgeBaseModel.id,
                IngestionJobModel.status == "failed",
            )
            .order_by(IngestionJobModel.finished_at.desc(), IngestionJobModel.id.desc())
            .limit(1)
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        active_index_version_id = (
            select(IndexVersionModel.id)
            .where(
                IndexVersionModel.kb_id == KnowledgeBaseModel.id,
                IndexVersionModel.status == "active",
                IndexVersionModel.active_slot == "active",
            )
            .limit(1)
            .correlate(KnowledgeBaseModel)
            .scalar_subquery()
        )
        return (
            document_count,
            active_bytes,
            active_chunks,
            processing_count,
            latest_failure,
            active_index_version_id,
        )

    def _kb_projection(self, kb: KnowledgeBaseModel, aggregates: Iterable[Any]) -> dict:
        (
            document_count,
            active_bytes,
            active_chunks,
            processing_count,
            latest_failure,
            active_index_version_id,
        ) = aggregates
        actions = self.resource_actions(kb)
        readiness = (
            "processing"
            if int(processing_count or 0) and not int(active_chunks or 0)
            else "ready"
            if int(active_chunks or 0)
            else "empty"
            if not int(document_count or 0)
            else "blocked"
        )
        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "status": kb.status,
            "data_classification": kb.data_classification,
            "external_llm_allowed": bool(kb.external_llm_allowed),
            "web_search_allowed": bool(kb.web_search_allowed),
            "retrieval_profile_id": kb.retrieval_config_id,
            "active_index_version_id": active_index_version_id,
            "readiness": readiness,
            "aggregates": {
                "document_count": int(document_count or 0),
                "active_chunk_count": int(active_chunks or 0),
                "active_source_bytes": int(active_bytes or 0),
                "processing_job_count": int(processing_count or 0),
                "latest_failure_code": latest_failure,
            },
            "access": {"grant_role": self._grant_role(kb)},
            "actions": actions,
            "etag": kb_etag(kb),
            "created_at": _iso(kb.created_at),
            "updated_at": _iso(kb.updated_at),
        }

    def list_knowledge_bases(
        self,
        *,
        cursor: str | None,
        limit: int,
        q: str | None,
        status: str | None,
        sort: str,
    ) -> dict:
        if sort not in {"updated_desc", "created_desc", "name_asc"}:
            raise WorkbenchCursorError("unsupported knowledge-base ordering")
        filters = {"q": (q or "").strip().casefold(), "status": status or ""}
        position = _decode_cursor(
            cursor, kind="knowledge-bases", sort=sort, filters=filters
        )
        where = [self._visible_kb_predicate()]
        if status:
            where.append(KnowledgeBaseModel.status == status)
        if q and q.strip():
            pattern = f"%{q.strip().casefold()}%"
            where.append(
                or_(
                    func.lower(KnowledgeBaseModel.name).like(pattern),
                    func.lower(KnowledgeBaseModel.description).like(pattern),
                )
            )
        aggregate_columns = self._kb_aggregate_columns()
        query = select(KnowledgeBaseModel, *aggregate_columns).where(*where)
        if sort == "name_asc":
            key = func.lower(KnowledgeBaseModel.name)
            if position:
                query = query.where(
                    or_(
                        key > str(position[0]),
                        and_(key == str(position[0]), KnowledgeBaseModel.id > str(position[1])),
                    )
                )
            query = query.order_by(key.asc(), KnowledgeBaseModel.id.asc())
        else:
            key = (
                KnowledgeBaseModel.updated_at
                if sort == "updated_desc"
                else KnowledgeBaseModel.created_at
            )
            if position:
                value = _strict_iso_datetime(position[0])
                query = query.where(
                    or_(
                        key < value,
                        and_(key == value, KnowledgeBaseModel.id < str(position[1])),
                    )
                )
            query = query.order_by(key.desc(), KnowledgeBaseModel.id.desc())
        bounded = max(1, min(int(limit), 200))
        rows = self.session.execute(query.limit(bounded + 1)).all()
        total = int(
            self.session.scalar(
                select(func.count(KnowledgeBaseModel.id)).where(*where)
            )
            or 0
        )
        selected = rows[:bounded]
        items = [self._kb_projection(row[0], row[1:]) for row in selected]
        next_cursor = None
        if len(rows) > bounded and selected:
            last = selected[-1][0]
            cursor_value = (
                last.name.casefold()
                if sort == "name_asc"
                else _iso(last.updated_at if sort == "updated_desc" else last.created_at)
            )
            next_cursor = _encode_cursor(
                kind="knowledge-bases",
                sort=sort,
                filters=filters,
                values=[cursor_value, last.id],
            )
        return {"items": items, "total": total, "limit": bounded, "next_cursor": next_cursor}

    def knowledge_base_detail(self, kb_id: str) -> dict:
        kb = self.get_kb(kb_id)
        aggregates = self.session.execute(
            select(*self._kb_aggregate_columns()).where(KnowledgeBaseModel.id == kb.id)
        ).one()
        return self._kb_projection(kb, aggregates)

    def update_knowledge_base(
        self, kb_id: str, changes: dict[str, Any], if_match: str | None
    ) -> dict:
        kb = self.get_kb(kb_id)
        self.assert_action(kb, "update")
        supplied = _normalized_etag(if_match)
        current = kb_etag(kb)
        if supplied != current:
            raise WorkbenchPreconditionError(
                "knowledge base changed since it was read",
                required=False,
                current_etag=current,
            )
        # Reuse the domain transition and uniqueness behavior already enforced by
        # ControlRepository.  The mapper revision automatically advances on flush.
        return self._kb_projection(
            kb=self._apply_kb_changes(kb, changes),
            aggregates=self.session.execute(
                select(*self._kb_aggregate_columns()).where(
                    KnowledgeBaseModel.id == kb.id
                )
            ).one(),
        )

    def _apply_kb_changes(
        self, kb: KnowledgeBaseModel, changes: dict[str, Any]
    ) -> KnowledgeBaseModel:
        from app.domain.knowledge_base import ensure_knowledge_base_transition

        requested_name = changes.get("name")
        if requested_name is not None:
            duplicate = self.session.scalar(
                select(KnowledgeBaseModel.id).where(
                    KnowledgeBaseModel.tenant_id == kb.tenant_id,
                    KnowledgeBaseModel.name == str(requested_name).strip(),
                    KnowledgeBaseModel.id != kb.id,
                    KnowledgeBaseModel.status != "deleted",
                )
            )
            if duplicate is not None:
                raise ResourceConflictError(
                    "knowledge base name already exists in tenant"
                )
        for key, value in changes.items():
            if key == "name":
                kb.name = str(value).strip()
            elif key == "description":
                kb.description = str(value).strip()
            elif key == "status":
                ensure_knowledge_base_transition(kb.status, str(value))
                kb.status = str(value)
            elif key in {
                "data_classification",
                "external_llm_allowed",
                "web_search_allowed",
            }:
                setattr(kb, key, value)
        try:
            self.session.flush()
        except StaleDataError as exc:
            raise WorkbenchPreconditionError(
                "knowledge base changed during the conditional update",
                required=False,
            ) from exc
        return kb

    def get_document(self, document_id: str) -> tuple[DocumentModel, KnowledgeBaseModel]:
        row = self.session.execute(
            select(DocumentModel, KnowledgeBaseModel)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .where(
                DocumentModel.id == document_id,
                DocumentModel.status != "deleted",
                self._visible_kb_predicate(),
            )
        ).first()
        if row is None:
            raise ResourceNotFoundError("workbench resource not found")
        return row[0], row[1]

    def _document_columns(self):
        version_count = (
            select(func.count(DocumentVersionModel.id))
            .where(DocumentVersionModel.document_id == DocumentModel.id)
            .correlate(DocumentModel)
            .scalar_subquery()
        )
        latest_version_no = (
            select(func.max(DocumentVersionModel.version_no))
            .where(DocumentVersionModel.document_id == DocumentModel.id)
            .correlate(DocumentModel)
            .scalar_subquery()
        )
        active_chunks = (
            select(DocumentIndexBindingModel.chunk_count)
            .join(
                IndexVersionModel,
                IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
            )
            .where(
                DocumentIndexBindingModel.document_version_id == DocumentModel.active_version_id,
                DocumentIndexBindingModel.status == "published",
                IndexVersionModel.status == "active",
                IndexVersionModel.active_slot == "active",
            )
            .limit(1)
            .correlate(DocumentModel)
            .scalar_subquery()
        )
        latest_job_id = (
            select(IngestionJobModel.id)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .where(DocumentVersionModel.document_id == DocumentModel.id)
            .order_by(
                DocumentVersionModel.version_no.desc(),
                IngestionJobModel.attempt.desc(),
                IngestionJobModel.created_at.desc(),
                IngestionJobModel.id.desc(),
            )
            .limit(1)
            .correlate(DocumentModel)
            .scalar_subquery()
        )
        return version_count, latest_version_no, active_chunks, latest_job_id

    def _document_projection(
        self,
        document: DocumentModel,
        kb: KnowledgeBaseModel,
        values: Iterable[Any],
        *,
        latest_job: IngestionJobModel | None | object = _UNSET,
    ) -> dict:
        version_count, latest_version_no, active_chunks, latest_job_id = values
        if latest_job is _UNSET:
            latest_job = (
                self.session.get(IngestionJobModel, latest_job_id)
                if latest_job_id
                else None
            )
        actions = self.resource_actions(kb)
        document_actions = ["read", "upload_version"] if "upload_document" in actions else ["read"]
        if "disable_document" in actions and document.status == "ready":
            document_actions.append("disable")
        if "enable_document" in actions and document.status == "disabled":
            document_actions.append("enable")
        if "delete_document" in actions:
            document_actions.extend(["soft_delete", "purge"])
        if latest_job and latest_job.status in {"failed", "cancelled"} and self._has_permission("job:manage"):
            document_actions.append("retry")
        return {
            "id": document.id,
            "kb_id": document.kb_id,
            "logical_name": document.logical_name,
            "source_type": document.source_type,
            "status": document.status,
            "active_version_id": document.active_version_id,
            "version_count": int(version_count or 0),
            "latest_version_no": int(latest_version_no or 0),
            "active_chunk_count": int(active_chunks or 0),
            "latest_job": self._safe_job(latest_job) if latest_job else None,
            "actions": sorted(set(document_actions)),
            "etag": document_etag(document),
            "created_at": _iso(document.created_at),
            "updated_at": _iso(document.updated_at),
        }

    @staticmethod
    def _safe_job(row: IngestionJobModel) -> dict:
        return {
            "id": row.id,
            "document_version_id": row.document_version_id,
            "index_version_id": row.index_version_id,
            "operation": row.operation,
            "status": row.status,
            "stage": row.stage,
            "progress": row.progress,
            "attempt": row.attempt,
            "run_attempt": row.run_attempt,
            "max_attempts": row.max_attempts,
            "next_retry_at": _iso(row.next_retry_at),
            "error_category": row.error_category,
            "error_code": row.error_code,
            "error_detail": (
                "任务执行未成功；详细诊断信息仅在服务端可见。"
                if row.error_detail
                else None
            ),
            "checkpoint_stage": row.checkpoint_stage,
            "checkpoint_sequence": row.checkpoint_sequence,
            "last_event_sequence": row.last_event_sequence,
            "started_at": _iso(row.started_at),
            "finished_at": _iso(row.finished_at),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    def list_documents(
        self,
        kb_id: str,
        *,
        cursor: str | None,
        limit: int,
        q: str | None,
        status: str | None,
        mime_type: str | None,
        sort: str,
    ) -> dict:
        kb = self.get_kb(kb_id)
        self.assert_action(kb, "read")
        if sort not in {"updated_desc", "created_desc", "name_asc"}:
            raise WorkbenchCursorError("unsupported document ordering")
        filters = {
            "kb_id": kb_id,
            "q": (q or "").strip().casefold(),
            "status": status or "",
            "mime_type": mime_type or "",
        }
        position = _decode_cursor(cursor, kind="documents", sort=sort, filters=filters)
        where = [DocumentModel.kb_id == kb_id, DocumentModel.status != "deleted"]
        if status:
            where.append(DocumentModel.status == status)
        if q and q.strip():
            where.append(func.lower(DocumentModel.logical_name).like(f"%{q.strip().casefold()}%"))
        if mime_type:
            where.append(
                select(DocumentVersionModel.id)
                .where(
                    DocumentVersionModel.document_id == DocumentModel.id,
                    DocumentVersionModel.mime_type == mime_type,
                )
                .exists()
            )
        columns = self._document_columns()
        query = select(DocumentModel, *columns).where(*where)
        if sort == "name_asc":
            key = func.lower(DocumentModel.logical_name)
            if position:
                query = query.where(
                    or_(
                        key > str(position[0]),
                        and_(key == str(position[0]), DocumentModel.id > str(position[1])),
                    )
                )
            query = query.order_by(key.asc(), DocumentModel.id.asc())
        else:
            key = DocumentModel.updated_at if sort == "updated_desc" else DocumentModel.created_at
            if position:
                value = _strict_iso_datetime(position[0])
                query = query.where(
                    or_(key < value, and_(key == value, DocumentModel.id < str(position[1])))
                )
            query = query.order_by(key.desc(), DocumentModel.id.desc())
        bounded = max(1, min(int(limit), 200))
        rows = self.session.execute(query.limit(bounded + 1)).all()
        total = int(self.session.scalar(select(func.count(DocumentModel.id)).where(*where)) or 0)
        selected = rows[:bounded]
        latest_job_ids = {str(row[4]) for row in selected if row[4]}
        latest_jobs = {
            item.id: item
            for item in self.session.scalars(
                select(IngestionJobModel).where(IngestionJobModel.id.in_(latest_job_ids))
            ).all()
        }
        items = [
            self._document_projection(
                row[0], kb, row[1:], latest_job=latest_jobs.get(str(row[4]))
            )
            for row in selected
        ]
        next_cursor = None
        if len(rows) > bounded and selected:
            last = selected[-1][0]
            value = (
                last.logical_name.casefold()
                if sort == "name_asc"
                else _iso(last.updated_at if sort == "updated_desc" else last.created_at)
            )
            next_cursor = _encode_cursor(
                kind="documents", sort=sort, filters=filters, values=[value, last.id]
            )
        return {"items": items, "total": total, "limit": bounded, "next_cursor": next_cursor}

    def document_detail(self, document_id: str) -> dict:
        document, kb = self.get_document(document_id)
        values = self.session.execute(
            select(*self._document_columns()).where(DocumentModel.id == document.id)
        ).one()
        result = self._document_projection(document, kb, values)
        result["versions"] = self.list_versions(document.id, limit=200)["items"]
        return result

    def list_versions(self, document_id: str, *, limit: int = 200) -> dict:
        document, kb = self.get_document(document_id)
        self.assert_action(kb, "read")
        rows = self.session.scalars(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document.id)
            .order_by(DocumentVersionModel.version_no.desc(), DocumentVersionModel.id.desc())
            .limit(max(1, min(limit, 200)))
        ).all()
        items = [self.version_projection(row) for row in rows]
        total = int(
            self.session.scalar(
                select(func.count(DocumentVersionModel.id)).where(
                    DocumentVersionModel.document_id == document.id
                )
            )
            or 0
        )
        return {"items": items, "total": total, "limit": min(limit, 200), "next_cursor": None}

    def get_version(self, version_id: str) -> tuple[DocumentVersionModel, DocumentModel, KnowledgeBaseModel]:
        row = self.session.execute(
            select(DocumentVersionModel, DocumentModel, KnowledgeBaseModel)
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .where(DocumentVersionModel.id == version_id, self._visible_kb_predicate())
        ).first()
        if row is None:
            raise ResourceNotFoundError("workbench resource not found")
        return row[0], row[1], row[2]

    def version_projection(self, version: DocumentVersionModel) -> dict:
        parse_artifacts = self.session.scalars(
            select(DocumentParseArtifactModel)
            .where(DocumentParseArtifactModel.document_version_id == version.id)
            .order_by(DocumentParseArtifactModel.created_at.asc())
        ).all()
        runs = self.session.scalars(
            select(DocumentChunkingRunModel)
            .join(
                DocumentParseArtifactModel,
                DocumentParseArtifactModel.id == DocumentChunkingRunModel.parse_artifact_id,
            )
            .where(DocumentParseArtifactModel.document_version_id == version.id)
            .order_by(DocumentChunkingRunModel.created_at.asc())
        ).all()
        jobs = self.session.scalars(
            select(IngestionJobModel)
            .where(IngestionJobModel.document_version_id == version.id)
            .order_by(IngestionJobModel.attempt.desc(), IngestionJobModel.id.desc())
        ).all()
        config = dict(version.parser_config_snapshot or {})
        # Retain reproducibility-relevant parser/template settings while keeping any
        # future secret-like extension out of this public DTO.
        safe_config = sanitize_public_json(config)
        return {
            "id": version.id,
            "document_id": version.document_id,
            "version_no": version.version_no,
            "content_sha256": version.content_sha256,
            "pipeline_fingerprint": version.pipeline_fingerprint,
            "size_bytes": version.size_bytes,
            "mime_type": version.mime_type,
            "parser_config": safe_config,
            "status": version.status,
            "published_at": _iso(version.published_at),
            "created_at": _iso(version.created_at),
            "parse_artifacts": [
                {
                    "id": item.id,
                    "parser_id": item.parser_id,
                    "parser_version": item.parser_version,
                    "schema_version": item.schema_version,
                    "status": item.status,
                    "artifact_sha256": item.artifact_sha256,
                    "artifact_size": item.artifact_size,
                    "warnings": item.warnings_json or [],
                    "metrics": {
                        key: value
                        for key, value in (item.metrics_json or {}).items()
                        if key in {"cache_hit", "block_count", "asset_count"}
                    },
                }
                for item in parse_artifacts
            ],
            "chunking_runs": [
                {
                    "id": item.id,
                    "parse_artifact_id": item.parse_artifact_id,
                    "template_id": item.template_id,
                    "template_version": item.template_version,
                    "config_hash": item.config_hash,
                    "status": item.status,
                    "artifact_sha256": item.artifact_sha256,
                    "artifact_size": item.artifact_size,
                    "chunk_count": item.chunk_count,
                }
                for item in runs
            ],
            "jobs": [self._safe_job(item) for item in jobs],
            "source_url": f"/api/v1/workbench/document-versions/{version.id}/source",
            "blocks_url": f"/api/v1/workbench/document-versions/{version.id}/blocks",
            "chunks_url": f"/api/v1/workbench/document-versions/{version.id}/chunks",
        }

    def assert_document_etag(self, document: DocumentModel, if_match: str | None) -> None:
        supplied = _normalized_etag(if_match)
        current = document_etag(document)
        if supplied != current:
            raise WorkbenchPreconditionError(
                "document changed since it was read", required=False, current_etag=current
            )

    def list_jobs(
        self,
        *,
        kb_id: str | None,
        cursor: str | None,
        limit: int,
        status: str | None,
    ) -> dict:
        if kb_id:
            kb = self.get_kb(kb_id)
            self.assert_action(kb, "read")
        filters = {"kb_id": kb_id or "", "status": status or ""}
        position = _decode_cursor(
            cursor, kind="ingestion-jobs", sort="created_desc", filters=filters
        )
        where = [self._visible_kb_predicate()]
        if kb_id:
            where.append(DocumentModel.kb_id == kb_id)
        if status:
            where.append(IngestionJobModel.status == status)
        query = (
            select(IngestionJobModel, DocumentModel, KnowledgeBaseModel)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .where(*where)
        )
        if position:
            value = _strict_iso_datetime(position[0])
            query = query.where(
                or_(
                    IngestionJobModel.created_at < value,
                    and_(
                        IngestionJobModel.created_at == value,
                        IngestionJobModel.id < str(position[1]),
                    ),
                )
            )
        query = query.order_by(IngestionJobModel.created_at.desc(), IngestionJobModel.id.desc())
        bounded = max(1, min(int(limit), 200))
        rows = self.session.execute(query.limit(bounded + 1)).all()
        count_query = (
            select(func.count(IngestionJobModel.id))
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .where(*where)
        )
        total = int(self.session.scalar(count_query) or 0)
        selected = rows[:bounded]
        items = []
        for job, document, kb in selected:
            item = self._safe_job(job)
            item.update(
                {
                    "document_id": document.id,
                    "logical_name": document.logical_name,
                    "kb_id": document.kb_id,
                    "actions": self._job_actions(job, kb),
                }
            )
            items.append(item)
        next_cursor = None
        if len(rows) > bounded and selected:
            last = selected[-1][0]
            next_cursor = _encode_cursor(
                kind="ingestion-jobs",
                sort="created_desc",
                filters=filters,
                values=[_iso(last.created_at), last.id],
            )
        return {"items": items, "total": total, "limit": bounded, "next_cursor": next_cursor}

    def _job_actions(self, job: IngestionJobModel, kb: KnowledgeBaseModel) -> list[str]:
        if "read" not in self.resource_actions(kb):
            return []
        actions = ["read"]
        if self._has_permission("job:manage") and self._grant_role(kb) in WRITE_GRANT_ROLES:
            if job.status in {"queued", "leased", "running", "retry_wait", "processing"}:
                actions.append("cancel")
            if job.status in {"failed", "cancelled"}:
                actions.append("retry")
        return actions

    def get_job(self, job_id: str) -> tuple[IngestionJobModel, DocumentVersionModel, DocumentModel, KnowledgeBaseModel]:
        row = self.session.execute(
            select(IngestionJobModel, DocumentVersionModel, DocumentModel, KnowledgeBaseModel)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .where(IngestionJobModel.id == job_id, self._visible_kb_predicate())
        ).first()
        if row is None:
            raise ResourceNotFoundError("workbench resource not found")
        return row[0], row[1], row[2], row[3]

    def job_detail(self, job_id: str) -> dict:
        job, version, document, kb = self.get_job(job_id)
        events = self.session.scalars(
            select(IngestionJobEventModel)
            .where(IngestionJobEventModel.job_id == job.id)
            .order_by(IngestionJobEventModel.sequence.asc())
        ).all()
        checkpoints = self.session.scalars(
            select(IngestionJobCheckpointModel)
            .where(IngestionJobCheckpointModel.job_id == job.id)
            .order_by(IngestionJobCheckpointModel.sequence.asc())
        ).all()
        manifest = self.session.scalar(
            select(IndexBuildManifestModel).where(IndexBuildManifestModel.job_id == job.id)
        )
        parse_artifacts = self.session.scalars(
            select(DocumentParseArtifactModel).where(
                DocumentParseArtifactModel.document_version_id == version.id
            )
        ).all()
        result = self._safe_job(job)
        result.update(
            {
                "document": {
                    "id": document.id,
                    "kb_id": document.kb_id,
                    "logical_name": document.logical_name,
                    "version_id": version.id,
                    "version_no": version.version_no,
                },
                "actions": self._job_actions(job, kb),
                "events": [
                    {
                        "sequence": item.sequence,
                        "event_type": item.event_type,
                        "status": item.status,
                        "stage": item.stage,
                        "progress": item.progress,
                        "payload": sanitize_public_json(item.payload or {}),
                        "created_at": _iso(item.created_at),
                    }
                    for item in events
                ],
                "checkpoints": [
                    {
                        "id": item.id,
                        "sequence": item.sequence,
                        "stage": item.stage,
                        "resume_node": item.resume_node,
                        "artifact_sha256": item.artifact_sha256,
                        "artifact_size": item.artifact_size,
                        "input_sha256": item.input_sha256,
                        "config_sha256": item.config_sha256,
                        "created_at": _iso(item.created_at),
                    }
                    for item in checkpoints
                ],
                "manifest": (
                    {
                        "id": manifest.id,
                        "document_version_id": manifest.document_version_id,
                        "index_version_id": manifest.index_version_id,
                        "expected_chunk_count": manifest.expected_chunk_count,
                        "written_chunk_count": manifest.written_chunk_count,
                        "searchable_chunk_count": manifest.searchable_chunk_count,
                        "chunk_ids_sha256": manifest.chunk_ids_sha256,
                        "vector_dim": manifest.vector_dim,
                        "validation_status": manifest.validation_status,
                        "validation_errors": sanitize_public_json(
                            manifest.validation_errors or []
                        ),
                        "validated_at": _iso(manifest.validated_at),
                    }
                    if manifest
                    else None
                ),
                "warnings": [
                    sanitize_public_json(warning)
                    for artifact in parse_artifacts
                    for warning in (artifact.warnings_json or [])
                ],
                "artifacts": [
                    {
                        "id": artifact.id,
                        "kind": "parsed_document",
                        "schema_version": artifact.schema_version,
                        "sha256": artifact.artifact_sha256,
                        "size": artifact.artifact_size,
                        "status": artifact.status,
                    }
                    for artifact in parse_artifacts
                ],
            }
        )
        return result

    def create_chunk_draft(
        self, document_version_id: str, base_chunking_run_id: str | None
    ) -> ChunkEditDraftModel:
        version, document, kb = self.get_version(document_version_id)
        self.assert_action(kb, "create_chunk_draft")
        query = (
            select(DocumentChunkingRunModel)
            .join(
                DocumentParseArtifactModel,
                DocumentParseArtifactModel.id == DocumentChunkingRunModel.parse_artifact_id,
            )
            .where(
                DocumentParseArtifactModel.document_version_id == version.id,
                DocumentChunkingRunModel.status == "succeeded",
            )
        )
        if base_chunking_run_id:
            query = query.where(DocumentChunkingRunModel.id == base_chunking_run_id)
        run = self.session.scalar(
            query.order_by(DocumentChunkingRunModel.created_at.desc(), DocumentChunkingRunModel.id.desc())
        )
        if run is None:
            raise ResourceConflictError("document version has no readable succeeded chunking run")
        draft = ChunkEditDraftModel(
            id=new_id("ced"),
            tenant_id=self.principal.tenant_id,
            kb_id=kb.id,
            document_id=document.id,
            document_version_id=version.id,
            base_chunking_run_id=run.id,
            status="draft",
            revision=1,
            created_by=self.principal.user_id,
        )
        self.session.add(draft)
        self.session.flush()
        return draft

    def get_chunk_draft(self, draft_id: str) -> tuple[ChunkEditDraftModel, KnowledgeBaseModel]:
        row = self.session.execute(
            select(ChunkEditDraftModel, KnowledgeBaseModel)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == ChunkEditDraftModel.kb_id)
            .where(
                ChunkEditDraftModel.id == draft_id,
                ChunkEditDraftModel.tenant_id == self.principal.tenant_id,
                self._visible_kb_predicate(),
            )
        ).first()
        if row is None:
            raise ResourceNotFoundError("workbench resource not found")
        return row[0], row[1]

    def chunk_operations(self, draft_id: str) -> list[ChunkEditOperationModel]:
        return list(
            self.session.scalars(
                select(ChunkEditOperationModel)
                .where(ChunkEditOperationModel.draft_id == draft_id)
                .order_by(ChunkEditOperationModel.sequence.asc())
            ).all()
        )

    @staticmethod
    def assert_chunk_draft_etag(
        draft: ChunkEditDraftModel, if_match: str | None
    ) -> None:
        supplied = _normalized_etag(if_match)
        current = draft_etag(draft)
        if supplied != current:
            raise WorkbenchPreconditionError(
                "chunk draft changed since it was read",
                required=False,
                current_etag=current,
            )

    def draft_projection(self, draft: ChunkEditDraftModel) -> dict:
        operations = self.chunk_operations(draft.id)
        return {
            "id": draft.id,
            "kb_id": draft.kb_id,
            "document_id": draft.document_id,
            "document_version_id": draft.document_version_id,
            "base_chunking_run_id": draft.base_chunking_run_id,
            "status": draft.status,
            "revision": draft.revision,
            "frozen_revision": draft.frozen_revision,
            "etag": draft_etag(draft),
            "operation_count": len(operations),
            "operations": [
                {
                    "id": item.id,
                    "sequence": item.sequence,
                    "operation": item.operation,
                    "input_chunk_ids": item.input_chunk_ids_json or [],
                    "output_chunks": sanitize_public_json(
                        item.output_chunks_json or []
                    ),
                    "created_at": _iso(item.created_at),
                }
                for item in operations
            ],
            "published_document_version_id": draft.published_document_version_id,
            "published_index_version_id": draft.published_index_version_id,
            "published_job_id": draft.published_job_id,
            "published_at": _iso(draft.published_at),
            "created_at": _iso(draft.created_at),
            "updated_at": _iso(draft.updated_at),
        }

    def append_chunk_operation(
        self,
        draft_id: str,
        *,
        if_match: str | None,
        operation: str,
        input_chunk_ids: list[str],
        output_chunks: list[dict[str, Any]],
    ) -> ChunkEditDraftModel:
        draft, kb = self.get_chunk_draft(draft_id)
        self.assert_action(kb, "create_chunk_draft")
        if draft.status != "draft":
            raise ResourceConflictError("published or publishing chunk draft is immutable")
        self.assert_chunk_draft_etag(draft, if_match)
        existing_operations = self.chunk_operations(draft.id)
        sequence = max((item.sequence for item in existing_operations), default=0) + 1
        row = ChunkEditOperationModel(
            id=new_id("ceo"),
            draft_id=draft.id,
            sequence=sequence,
            operation=operation,
            input_chunk_ids_json=list(input_chunk_ids),
            output_chunks_json=list(output_chunks),
            created_by=self.principal.user_id,
        )
        self.session.add(row)
        # Assignment forces an UPDATE even if no other scalar draft field changed.
        draft.operations_hash = canonical_json_sha256(
            [
                {
                    "sequence": item.sequence,
                    "operation": item.operation,
                    "inputs": item.input_chunk_ids_json,
                    "outputs": item.output_chunks_json,
                }
                for item in [*existing_operations, row]
            ]
        )
        try:
            self.session.flush()
        except StaleDataError as exc:
            raise WorkbenchPreconditionError(
                "chunk draft changed during the conditional update",
                required=False,
            ) from exc
        return draft


__all__ = [
    "KnowledgeWorkbenchRepository",
    "WorkbenchCursorError",
    "WorkbenchPreconditionError",
    "document_etag",
    "draft_etag",
    "kb_etag",
    "sanitize_public_json",
]
