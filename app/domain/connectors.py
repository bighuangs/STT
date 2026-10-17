"""TODO-17 数据源连接器与增量同步领域契约。

领域对象不持有凭据正文，也不依赖具体连接器、数据库或导入实现。所有对象都显式携带
tenant/KB/data-source scope，避免调用方通过 metadata 注入或扩大资源范围。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key"
    r"|authorization|credential|session[_-]?cookie)",
    re.IGNORECASE,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_identifier(value: str) -> str:
    value = value.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("identifier contains unsupported characters")
    return value


def _inline_credential_paths(value: Any, path: str = "config") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _SECRET_KEY_RE.search(str(key)):
                found.append(child_path)
            found.extend(_inline_credential_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_inline_credential_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.query:
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                if _SECRET_KEY_RE.search(key):
                    found.append(f"{path}.url_query.{key}")
    return tuple(found)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConnectorScope(StrictModel):
    tenant_id: str
    kb_id: str

    _tenant = field_validator("tenant_id")(_validate_identifier)
    _kb = field_validator("kb_id")(_validate_identifier)


class ConnectorCredentialRef(StrictModel):
    """只保存密钥管理器引用；本对象永远不接受凭据正文。"""

    provider: str = Field(min_length=1, max_length=64)
    reference: str = Field(min_length=1, max_length=512)
    version: str = Field(default="current", min_length=1, max_length=128)

    @field_validator("reference")
    @classmethod
    def reject_inline_secret(cls, value: str) -> str:
        lowered = value.lower()
        if lowered.startswith(("http://", "https://")) or "://" in lowered:
            raise ValueError("credential reference must be an opaque secret-manager reference")
        if "=" in value or "\n" in value or "\r" in value:
            raise ValueError("credential reference appears to contain an inline credential")
        return value.strip()


class DeletePropagationPolicy(StrEnum):
    DISABLE = "disable"
    SOFT_DELETE = "soft_delete"
    PURGE = "purge"


class SyncPolicy(StrictModel):
    delete_policy: DeletePropagationPolicy = DeletePropagationPolicy.DISABLE
    deletion_grace_seconds: int = Field(default=86_400, ge=0, le=31_536_000)
    max_changes_per_run: int = Field(default=500, ge=1, le=10_000)
    max_object_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    allowed_mime_types: tuple[str, ...] = ()
    allow_partial_cursor_advance: bool = False
    allow_empty_snapshot_deletes: bool = False
    map_remote_acl: bool = False

    @field_validator("allowed_mime_types")
    @classmethod
    def normalize_mime_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({item.strip().lower() for item in value if item.strip()}))


class ConnectorCapabilities(StrictModel):
    incremental_cursor: bool = True
    delete_events: bool = True
    folders: bool = False
    acl: bool = False
    webhook: bool = False
    external_network: bool = True
    max_object_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    supported_mime_types: tuple[str, ...] = ()
    rate_limit_per_minute: int | None = Field(default=None, ge=1)

    @field_validator("supported_mime_types")
    @classmethod
    def normalize_mime_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({item.strip().lower() for item in value if item.strip()}))


class ConnectorValidation(StrictModel):
    valid: bool
    capabilities: ConnectorCapabilities
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def errors_match_validity(self) -> "ConnectorValidation":
        if self.valid and self.errors:
            raise ValueError("valid connector validation cannot contain errors")
        if not self.valid and not self.errors:
            raise ValueError("invalid connector validation must explain the error")
        return self


class DataSourceStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DELETED = "deleted"


class DataSource(StrictModel):
    data_source_id: str
    scope: ConnectorScope
    name: str = Field(min_length=1, max_length=256)
    connector_key: str
    credential_ref: ConnectorCredentialRef | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    sync_policy: SyncPolicy = Field(default_factory=SyncPolicy)
    status: DataSourceStatus = DataSourceStatus.ACTIVE
    config_version: int = Field(default=1, ge=1)

    _source_id = field_validator("data_source_id")(_validate_identifier)
    _connector_key = field_validator("connector_key")(_validate_identifier)

    @field_validator("config")
    @classmethod
    def reject_inline_credentials(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = sorted(set(_inline_credential_paths(value)))
        if forbidden:
            raise ValueError(
                "connector config contains credential-like fields; use credential_ref: "
                + ", ".join(forbidden)
            )
        return value


class SyncCursor(StrictModel):
    scope: ConnectorScope
    data_source_id: str
    value: str = ""
    generation: int = Field(default=0, ge=0)
    committed_at: datetime | None = None

    _source_id = field_validator("data_source_id")(_validate_identifier)

    def advance(self, value: str, *, at: datetime | None = None) -> "SyncCursor":
        return self.model_copy(
            update={
                "value": value,
                "generation": self.generation + 1,
                "committed_at": at or utc_now(),
            }
        )


class SourceItem(StrictModel):
    source_item_id: str
    remote_version: str = Field(min_length=1, max_length=512)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_name: str = Field(min_length=1, max_length=1024)
    mime_type: str = Field(min_length=1, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    locator: str = Field(default="", max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _item_id = field_validator("source_item_id")(_validate_identifier)

    @field_validator("mime_type")
    @classmethod
    def normalize_mime_type(cls, value: str) -> str:
        return value.split(";", 1)[0].strip().lower()

    @field_validator("metadata")
    @classmethod
    def prevent_scope_injection(cls, value: dict[str, Any]) -> dict[str, Any]:
        protected = {
            "tenant_id",
            "kb_id",
            "data_source_id",
            "document_id",
            "document_version_id",
            "index_version_id",
        }
        collisions = protected.intersection(value)
        if collisions:
            raise ValueError(
                "source metadata cannot override scope fields: " + ", ".join(sorted(collisions))
            )
        return value


class SourceTombstone(StrictModel):
    source_item_id: str
    remote_version: str = Field(min_length=1, max_length=512)
    prior_locator: str = Field(default="", max_length=4096)
    reason: str = Field(default="remote_delete", max_length=512)

    _item_id = field_validator("source_item_id")(_validate_identifier)


class SourceChangeKind(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


class SourceChange(StrictModel):
    event_id: str
    sequence: int = Field(ge=0)
    kind: SourceChangeKind
    item: SourceItem | None = None
    tombstone: SourceTombstone | None = None
    occurred_at: datetime = Field(default_factory=utc_now)

    _event_id = field_validator("event_id")(_validate_identifier)

    @model_validator(mode="after")
    def validate_payload(self) -> "SourceChange":
        if self.kind == SourceChangeKind.DELETE:
            if self.tombstone is None or self.item is not None:
                raise ValueError("delete change requires tombstone and forbids item")
        elif self.item is None or self.tombstone is not None:
            raise ValueError("non-delete change requires item and forbids tombstone")
        return self

    @property
    def source_item_id(self) -> str:
        payload = self.item if self.item is not None else self.tombstone
        assert payload is not None
        return payload.source_item_id

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"occurred_at"}))


class ChangePage(StrictModel):
    changes: tuple[SourceChange, ...] = ()
    next_cursor: str
    has_more: bool = False
    retry_after_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_events(self) -> "ChangePage":
        fingerprints: dict[str, str] = {}
        for change in self.changes:
            existing = fingerprints.setdefault(change.event_id, change.fingerprint)
            if existing != change.fingerprint:
                raise ValueError(f"event_id {change.event_id!r} has conflicting payloads")
        return self


class SourceSnapshot(StrictModel):
    items: tuple[SourceItem, ...]
    cursor: str
    complete: bool
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_items(self) -> "SourceSnapshot":
        item_ids = [item.source_item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("snapshot contains duplicate source_item_id values")
        return self


class SourceBindingStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DISABLED = "disabled"
    SOFT_DELETED = "soft_deleted"
    PURGED = "purged"
    FAILED = "failed"


class SourceItemBinding(StrictModel):
    scope: ConnectorScope
    data_source_id: str
    source_item_id: str
    document_id: str
    active_document_version_id: str | None = None
    pending_document_version_id: str | None = None
    remote_version: str
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_name: str = ""
    locator: str = ""
    status: SourceBindingStatus = SourceBindingStatus.PENDING
    last_seen_at: datetime = Field(default_factory=utc_now)
    disabled_at: datetime | None = None
    purge_after: datetime | None = None

    _source_id = field_validator("data_source_id")(_validate_identifier)
    _item_id = field_validator("source_item_id")(_validate_identifier)
    _document_id = field_validator("document_id")(_validate_identifier)


class SyncAction(StrEnum):
    CREATE_VERSION = "create_version"
    UPDATE_VERSION = "update_version"
    UPDATE_METADATA = "update_metadata"
    DISABLE_DOCUMENT = "disable_document"
    SOFT_DELETE_DOCUMENT = "soft_delete_document"
    PURGE_DOCUMENT = "purge_document"
    NOOP = "noop"
    ALREADY_APPLIED = "already_applied"


class SyncOperation(StrictModel):
    ordinal: int = Field(ge=0)
    event_id: str
    event_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_item_id: str
    action: SyncAction
    reason: str
    idempotency_key: str = Field(pattern=r"^sync_[0-9a-f]{40}$")
    change: SourceChange
    prior_binding: SourceItemBinding | None = None


class SyncPlan(StrictModel):
    plan_version: str = "connector-sync-plan/1.0.0"
    scope: ConnectorScope
    data_source_id: str
    run_id: str
    from_cursor: str
    to_cursor: str
    has_more: bool
    reconciliation: bool = False
    dry_run: bool = False
    operations: tuple[SyncOperation, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class SyncEventStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class SyncEvent(StrictModel):
    scope: ConnectorScope
    data_source_id: str
    run_id: str
    event_id: str
    event_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_item_id: str
    action: SyncAction
    status: SyncEventStatus
    idempotency_key: str
    document_id: str | None = None
    document_version_id: str | None = None
    ingestion_job_id: str | None = None
    detail: str = ""
    applied_at: datetime = Field(default_factory=utc_now)


class SyncRunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    DRY_RUN = "dry_run"


class SyncRun(StrictModel):
    run_id: str
    scope: ConnectorScope
    data_source_id: str
    status: SyncRunStatus
    from_cursor: str
    to_cursor: str | None = None
    plan_hash: str | None = None
    applied_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    error_code: str | None = None
    error_detail: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class FetchedSourceObject(StrictModel):
    """连接器读取结果。

    首版核心使用 bytes 保证哈希和幂等边界清晰；生产适配器可在调用本服务前使用受限
    spool/流式哈希，但不得暴露不安全临时文件路径。
    """

    source_item_id: str
    remote_version: str
    content: bytes
    mime_type: str
    logical_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    _item_id = field_validator("source_item_id")(_validate_identifier)


class DocumentVersionReceipt(StrictModel):
    document_id: str
    document_version_id: str
    ingestion_job_id: str
    created: bool

    _document_id = field_validator("document_id")(_validate_identifier)
    _version_id = field_validator("document_version_id")(_validate_identifier)
    _job_id = field_validator("ingestion_job_id")(_validate_identifier)


class DeletePropagationReceipt(StrictModel):
    document_id: str
    state: SourceBindingStatus
    changed: bool

    _document_id = field_validator("document_id")(_validate_identifier)


def sync_idempotency_key(data_source_id: str, change: SourceChange) -> str:
    digest = canonical_sha256(
        {
            "data_source_id": data_source_id,
            "event_id": change.event_id,
            "fingerprint": change.fingerprint,
        }
    )
    return f"sync_{digest[:40]}"


__all__ = [
    "ChangePage",
    "ConnectorCapabilities",
    "ConnectorCredentialRef",
    "ConnectorScope",
    "ConnectorValidation",
    "DataSource",
    "DataSourceStatus",
    "DeletePropagationPolicy",
    "DeletePropagationReceipt",
    "DocumentVersionReceipt",
    "FetchedSourceObject",
    "SourceBindingStatus",
    "SourceChange",
    "SourceChangeKind",
    "SourceItem",
    "SourceItemBinding",
    "SourceSnapshot",
    "SourceTombstone",
    "SyncAction",
    "SyncCursor",
    "SyncEvent",
    "SyncEventStatus",
    "SyncOperation",
    "SyncPlan",
    "SyncPolicy",
    "SyncRun",
    "SyncRunStatus",
    "canonical_sha256",
    "sync_idempotency_key",
    "utc_now",
]
