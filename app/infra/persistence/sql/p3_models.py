"""P3 缓存/用量与向量拓扑控制面 ORM。

查询运行的 Conversation/Turn/Message/QueryRun/Event 模型位于
``query_runtime_models.py``。本文件承载 TODO-14、TODO-32、TODO-37 的其余持久
事实，所有表仍注册到同一个 ``Base.metadata``。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.infra.persistence.sql.models import Base, TimestampMixin


class QueryIdempotencyKeyModel(Base):
    __tablename__ = "query_idempotency_keys"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "key_hash", name="uq_query_idempotency_tenant_key"
        ),
        Index("ix_query_idempotency_expiry", "expires_at"),
        CheckConstraint(
            "key_hash LIKE 'sha256:%' AND length(key_hash) = 71",
            name="ck_query_idempotency_key_hash",
        ),
        CheckConstraint(
            "payload_hash LIKE 'sha256:%' AND length(payload_hash) = 71",
            name="ck_query_idempotency_payload_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    response_status: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class QueryCacheEntryModel(Base, TimestampMixin):
    __tablename__ = "query_cache_entries"
    __table_args__ = (
        Index("ix_query_cache_tenant_layer_expiry", "tenant_id", "layer", "expires_at"),
        Index("ix_query_cache_scope", "tenant_id", "authorized_scope_hash"),
        CheckConstraint(
            "cache_key LIKE 'sha256:%' AND length(cache_key) = 71",
            name="ck_query_cache_key",
        ),
        CheckConstraint(
            "authorized_scope_hash LIKE 'sha256:%' "
            "AND length(authorized_scope_hash) = 71",
            name="ck_query_cache_scope_hash",
        ),
        CheckConstraint(
            "classification IN ('public', 'internal')",
            name="ck_query_cache_classification",
        ),
    )

    cache_key: Mapped[str] = mapped_column(String(71), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    authorized_scope_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    layer: Mapped[str] = mapped_column(String(32), nullable=False)
    value_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    versions_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    classification: Mapped[str] = mapped_column(String(16), nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class QuerySingleflightLeaseModel(Base, TimestampMixin):
    __tablename__ = "query_singleflight_leases"
    __table_args__ = (
        Index("ix_query_singleflight_expiry", "lease_expires_at"),
        CheckConstraint("fencing_token >= 1", name="ck_query_singleflight_fencing"),
    )

    fingerprint: Mapped[str] = mapped_column(String(71), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    lease_token_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    result_cache_key: Mapped[str | None] = mapped_column(String(71))


class QueryReplayManifestModel(Base, TimestampMixin):
    __tablename__ = "query_replay_manifests"
    __table_args__ = (
        UniqueConstraint("query_run_id", name="uq_query_replay_run"),
        Index("ix_query_replay_tenant_created", "tenant_id", "created_at"),
        CheckConstraint(
            "request_hash LIKE 'sha256:%' AND length(request_hash) = 71",
            name="ck_query_replay_request_hash",
        ),
        CheckConstraint(
            "event_stream_hash LIKE 'sha256:%' AND length(event_stream_hash) = 71",
            name="ck_query_replay_event_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    fingerprint: Mapped[str] = mapped_column(String(71), nullable=False, index=True)
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    manifest_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    event_stream_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    object_refs_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class ModelUsageRecordModel(Base):
    __tablename__ = "model_usage_records"
    __table_args__ = (
        Index("ix_model_usage_tenant_created", "tenant_id", "created_at"),
        Index("ix_model_usage_query_run", "query_run_id"),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND cache_read_tokens >= 0",
            name="ck_model_usage_tokens",
        ),
        CheckConstraint("calls >= 1", name="ck_model_usage_calls"),
        CheckConstraint("cost >= 0", name="ck_model_usage_cost"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[str | None] = mapped_column(String(64), index=True)
    query_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scene: Mapped[str] = mapped_column(String(64), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calls: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CNY")
    unit_price_version: Mapped[str] = mapped_column(String(64), nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class VectorCollectionFamilyModel(Base, TimestampMixin):
    __tablename__ = "vector_collection_families"
    __table_args__ = (
        UniqueConstraint(
            "physical_database",
            "physical_collection",
            name="uq_vector_family_physical_name",
        ),
        Index(
            "ix_vector_family_compatibility_status",
            "compatibility_fingerprint",
            "status",
        ),
        CheckConstraint("epoch >= 1", name="ck_vector_family_epoch"),
        CheckConstraint(
            "status IN ('creating', 'validating', 'active', 'draining', "
            "'retired', 'failed')",
            name="ck_vector_family_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    physical_database: Mapped[str] = mapped_column(String(255), nullable=False)
    physical_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    placement_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    compatibility_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    index_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    compatibility_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    partition_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    num_partitions: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VectorPlacementModel(Base, TimestampMixin):
    __tablename__ = "vector_placements"
    __table_args__ = (
        UniqueConstraint("index_version_id", "read_slot", name="uq_vector_active_read"),
        UniqueConstraint("index_version_id", "write_slot", name="uq_vector_active_write"),
        Index("ix_vector_placement_scope", "tenant_id", "kb_id", "status"),
        Index("ix_vector_placement_family", "family_id", "status"),
        CheckConstraint("epoch >= 1", name="ck_vector_placement_epoch"),
        CheckConstraint(
            "status IN ('planned', 'active', 'source', 'target', 'retired')",
            name="ck_vector_placement_status",
        ),
        CheckConstraint(
            "(readable = true AND read_slot = 'active') OR "
            "(readable = false AND read_slot IS NULL)",
            name="ck_vector_placement_read_slot",
        ),
        CheckConstraint(
            "(writable = true AND write_slot = 'active') OR "
            "(writable = false AND write_slot IS NULL)",
            name="ck_vector_placement_write_slot",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    family_id: Mapped[str] = mapped_column(
        ForeignKey("vector_collection_families.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    partition_bucket: Mapped[str | None] = mapped_column(String(255))
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    placement_reason_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    readable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    writable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    read_slot: Mapped[str | None] = mapped_column(String(16))
    write_slot: Mapped[str | None] = mapped_column(String(16))
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VectorGcRunModel(Base, TimestampMixin):
    __tablename__ = "vector_gc_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_vector_gc_idempotency"),
        Index(
            "ix_vector_gc_scope",
            "tenant_id",
            "kb_id",
            "document_id",
            "document_version_id",
            "index_version_id",
        ),
        Index("ix_vector_gc_status_retry", "status", "next_retry_at"),
        CheckConstraint("expected_count >= 0", name="ck_vector_gc_expected"),
        CheckConstraint("attempt >= 0", name="ck_vector_gc_attempt"),
        CheckConstraint("fencing_token >= 0", name="ck_vector_gc_fencing"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    index_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scope_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    expected_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    matched_before: Mapped[int | None] = mapped_column(BigInteger)
    delete_acknowledged: Mapped[bool | None] = mapped_column(Boolean)
    remaining_after: Mapped[int | None] = mapped_column(BigInteger)
    compaction_id: Mapped[str | None] = mapped_column(String(255))
    compaction_status: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="planned")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token_hash: Mapped[str | None] = mapped_column(String(71))
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VectorMigrationRunModel(Base, TimestampMixin):
    __tablename__ = "vector_migration_runs"
    __table_args__ = (
        UniqueConstraint(
            "source_family_id",
            "target_family_id",
            "index_version_id",
            name="uq_vector_migration_path",
        ),
        Index("ix_vector_migration_scope_status", "tenant_id", "kb_id", "status"),
        CheckConstraint("expected_count >= 0", name="ck_vector_migration_expected"),
        CheckConstraint("attempt >= 0", name="ck_vector_migration_attempt"),
        CheckConstraint("placement_epoch >= 0", name="ck_vector_migration_epoch"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    index_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_family_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    target_family_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    expected_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_ids_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    target_compatibility_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="planned")
    checkpoint_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    validation_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    rollback_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    placement_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token_hash: Mapped[str | None] = mapped_column(String(71))
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = [
    "ModelUsageRecordModel",
    "QueryCacheEntryModel",
    "QueryIdempotencyKeyModel",
    "QueryReplayManifestModel",
    "QuerySingleflightLeaseModel",
    "VectorCollectionFamilyModel",
    "VectorGcRunModel",
    "VectorMigrationRunModel",
    "VectorPlacementModel",
]
