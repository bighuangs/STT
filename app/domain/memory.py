"""TODO-22 长期记忆、来源、同意、作用域与遗忘领域契约。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")


def utc_now() -> datetime:
    return datetime.now(UTC)


def _identifier(value: str) -> str:
    value = value.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("identifier contains unsupported characters")
    return value


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryNamespace(StrictModel):
    tenant_id: str
    subject_user_id: str
    kb_id: str

    _tenant = field_validator("tenant_id")(_identifier)
    _subject = field_validator("subject_user_id")(_identifier)
    _kb = field_validator("kb_id")(_identifier)


class MemoryAccessScope(StrictModel):
    tenant_id: str
    subject_user_id: str
    kb_id: str
    actor_user_id: str

    _tenant = field_validator("tenant_id")(_identifier)
    _subject = field_validator("subject_user_id")(_identifier)
    _kb = field_validator("kb_id")(_identifier)
    _actor = field_validator("actor_user_id")(_identifier)

    @property
    def namespace(self) -> MemoryNamespace:
        return MemoryNamespace(
            tenant_id=self.tenant_id,
            subject_user_id=self.subject_user_id,
            kb_id=self.kb_id,
        )


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    PROFILE_FACT = "profile_fact"
    EPISODIC_SUMMARY = "episodic_summary"
    WORKING_MEMORY = "working_memory"


class MemorySensitivity(StrEnum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class MemoryConsent(StrEnum):
    NONE = "none"
    EXPLICIT_REQUEST = "explicit_request"
    CONFIRMED = "confirmed"
    WITHDRAWN = "withdrawn"


class MemoryStatus(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    EXPIRED = "expired"
    REJECTED = "rejected"


class MemorySourceKind(StrEnum):
    USER_MESSAGE = "user_message"
    MANUAL_ENTRY = "manual_entry"
    SESSION_SUMMARY = "session_summary"


class MemoryProvenance(StrictModel):
    source_kind: MemorySourceKind
    session_id: str | None = None
    source_message_ids: tuple[str, ...] = ()
    extractor_version: str = Field(default="memory-rules/1.0.0", max_length=128)
    source_excerpt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("session_id")
    @classmethod
    def validate_optional_session(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else None

    @field_validator("source_message_ids")
    @classmethod
    def validate_messages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(_identifier(item) for item in value))
        if not normalized:
            raise ValueError("memory provenance requires at least one source message")
        return normalized


class MemoryCandidate(StrictModel):
    candidate_id: str
    namespace: MemoryNamespace
    type: MemoryType
    content: str = Field(min_length=1, max_length=4096)
    structured_value: dict[str, Any] = Field(default_factory=dict)
    conflict_key: str = Field(min_length=1, max_length=256)
    confidence: float = Field(ge=0, le=1)
    sensitivity: MemorySensitivity
    consent: MemoryConsent
    provenance: MemoryProvenance
    expires_at: datetime
    proposed_at: datetime = Field(default_factory=utc_now)

    _candidate = field_validator("candidate_id")(_identifier)

    @model_validator(mode="after")
    def validate_candidate(self) -> "MemoryCandidate":
        if self.type == MemoryType.WORKING_MEMORY:
            raise ValueError("working memory cannot enter the long-term memory domain")
        if self.expires_at <= self.proposed_at:
            raise ValueError("memory candidate must expire after it is proposed")
        return self


class LongTermMemory(StrictModel):
    memory_id: str
    candidate_id: str
    namespace: MemoryNamespace
    type: MemoryType
    content: str = Field(max_length=4096)
    structured_value: dict[str, Any] = Field(default_factory=dict)
    conflict_key: str = Field(min_length=1, max_length=256)
    provenance: MemoryProvenance
    confidence: float = Field(ge=0, le=1)
    sensitivity: MemorySensitivity
    consent: MemoryConsent
    status: MemoryStatus
    valid_from: datetime
    valid_to: datetime | None = None
    expires_at: datetime
    created_by: str
    embedding_version: str | None = None
    supersedes_memory_id: str | None = None
    forgotten_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    _memory = field_validator("memory_id")(_identifier)
    _candidate = field_validator("candidate_id")(_identifier)
    _created_by = field_validator("created_by")(_identifier)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "LongTermMemory":
        if self.type == MemoryType.WORKING_MEMORY:
            raise ValueError("working memory cannot be persisted as long-term memory")
        if self.expires_at <= self.created_at:
            raise ValueError("memory must expire after creation")
        if self.status == MemoryStatus.ACTIVE and self.consent != MemoryConsent.CONFIRMED:
            raise ValueError("active memory requires confirmed consent")
        if self.status == MemoryStatus.FORGOTTEN:
            if self.content or self.structured_value:
                raise ValueError("forgotten memory must not retain user content")
            if self.forgotten_at is None:
                raise ValueError("forgotten memory requires forgotten_at")
        if self.status == MemoryStatus.SUPERSEDED and self.valid_to is None:
            raise ValueError("superseded memory requires valid_to")
        return self

    @property
    def recallable(self) -> bool:
        return self.status == MemoryStatus.ACTIVE


class MemoryRetrievalHit(StrictModel):
    memory_id: str
    type: MemoryType
    content: str
    structured_value: dict[str, Any]
    provenance: MemoryProvenance
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    expires_at: datetime


class MemoryContext(StrictModel):
    namespace: MemoryNamespace
    query: str
    rendered: str
    hits: tuple[MemoryRetrievalHit, ...]
    used_memory_ids: tuple[str, ...]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(ge=0)
    omitted_count: int = Field(ge=0)


class MemoryAuditAction(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    EXPIRED = "expired"
    REJECTED = "rejected"
    EXPORTED = "exported"


class MemoryAuditEvent(StrictModel):
    audit_id: str
    namespace: MemoryNamespace
    memory_id: str
    actor_user_id: str
    action: MemoryAuditAction
    reason_code: str
    occurred_at: datetime = Field(default_factory=utc_now)
    # 审计 metadata 只允许控制信息，不存 content/structured_value/source excerpt。
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ForgetReceipt(StrictModel):
    memory_id: str
    forgotten: bool
    derivative_deletions: tuple[str, ...]
    audit_id: str
    forgotten_at: datetime


def memory_candidate_id(
    *,
    namespace: MemoryNamespace,
    memory_type: MemoryType,
    content: str,
    source_message_ids: tuple[str, ...],
) -> str:
    digest = _stable_hash(
        {
            "namespace": namespace.model_dump(mode="json"),
            "type": memory_type.value,
            "content": " ".join(content.split()).casefold(),
            "source_message_ids": source_message_ids,
        }
    )
    return f"mcand_{digest[:40]}"


def memory_id_for_candidate(candidate_id: str) -> str:
    return f"mem_{_stable_hash(candidate_id)[:40]}"


def provenance_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "ForgetReceipt",
    "LongTermMemory",
    "MemoryAccessScope",
    "MemoryAuditAction",
    "MemoryAuditEvent",
    "MemoryCandidate",
    "MemoryConsent",
    "MemoryContext",
    "MemoryNamespace",
    "MemoryProvenance",
    "MemoryRetrievalHit",
    "MemorySensitivity",
    "MemorySourceKind",
    "MemoryStatus",
    "MemoryType",
    "memory_candidate_id",
    "memory_id_for_candidate",
    "provenance_hash",
    "utc_now",
]
