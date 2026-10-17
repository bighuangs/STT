"""TODO-01 控制库 ORM 模型。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    JSON,
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantModel(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class WorkspaceModel(Base, TimestampMixin):
    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_workspace_tenant_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class KnowledgeBaseModel(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_kb_tenant_name"),
        Index("ix_kb_workspace_status", "workspace_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    retrieval_config_id: Mapped[str | None] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    data_classification: Mapped[str] = mapped_column(String(32), nullable=False, default="internal")
    external_llm_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    web_search_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retention_policy_id: Mapped[str | None] = mapped_column(String(64))
    # Workbench optimistic concurrency token.  SQLAlchemy's version mapper makes
    # every ORM mutation (including legacy services) advance the revision, so an
    # If-Match check cannot silently miss a write performed outside the BFF.
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

    __mapper_args__ = {"version_id_col": revision}


class DocumentModel(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("kb_id", "logical_name", name="uq_document_kb_logical_name"),
        Index("ix_document_kb_status", "kb_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    logical_name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")
    source_uri: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    active_version_id: Mapped[str | None] = mapped_column(String(64), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

    __mapper_args__ = {"version_id_col": revision}


class DocumentVersionModel(Base, TimestampMixin):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_no", name="uq_document_version_no"),
        UniqueConstraint(
            "document_id",
            "content_sha256",
            "pipeline_fingerprint",
            name="uq_document_content_pipeline",
        ),
        Index("ix_document_version_status", "document_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # 同一源文件允许使用不同解析器/切块模板形成独立、可追溯的版本。
    # 旧数据由 0007 迁移回填为 legacy；新上传使用 parser_config_snapshot 的规范 JSON SHA-256。
    pipeline_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="legacy", server_default="legacy"
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    parser_config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentParseArtifactModel(Base, TimestampMixin):
    """与切块模板无关的统一解析制品，可在多个切块运行之间复用。"""

    __tablename__ = "document_parse_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "parser_id",
            "parser_version",
            "schema_version",
            name="uq_parse_artifact_version_parser_schema",
        ),
        Index("ix_parse_artifact_version_status", "document_version_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_id: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="succeeded")
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    artifact_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warnings_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DocumentChunkingRunModel(Base, TimestampMixin):
    """基于既有解析制品执行一次版本化切块模板的持久事实。"""

    __tablename__ = "document_chunking_runs"
    __table_args__ = (
        UniqueConstraint(
            "parse_artifact_id",
            "template_id",
            "template_version",
            "config_hash",
            name="uq_chunking_run_artifact_template_config",
        ),
        Index("ix_chunking_run_artifact_status", "parse_artifact_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parse_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("document_parse_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    template_id: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="succeeded")
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    artifact_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ChunkEditDraftModel(Base, TimestampMixin):
    """A mutable, CAS-protected draft over one immutable chunking run.

    Publishing never changes the base artifact or an Evidence row.  It creates a
    new DocumentVersion and a normal durable ingestion job; the worker consumes
    the frozen operations before embedding.
    """

    __tablename__ = "chunk_edit_drafts"
    __table_args__ = (
        Index("ix_chunk_edit_draft_scope_status", "tenant_id", "kb_id", "status"),
        Index("ix_chunk_edit_draft_document_version", "document_version_id", "created_at"),
        UniqueConstraint(
            "published_document_version_id",
            name="uq_chunk_edit_draft_published_document_version",
        ),
        UniqueConstraint(
            "published_job_id", name="uq_chunk_edit_draft_published_job"
        ),
        CheckConstraint(
            "status IN ('draft', 'publishing', 'indexed', 'published', 'abandoned')",
            name="ck_chunk_edit_draft_status",
        ),
        CheckConstraint(
            "frozen_revision IS NULL OR frozen_revision >= 1",
            name="ck_chunk_edit_draft_frozen_revision",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    base_chunking_run_id: Mapped[str] = mapped_column(
        ForeignKey("document_chunking_runs.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    operations_hash: Mapped[str | None] = mapped_column(String(64))
    frozen_revision: Mapped[int | None] = mapped_column(Integer)
    published_document_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("document_versions.id", ondelete="RESTRICT"), index=True
    )
    published_index_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("index_versions.id", ondelete="RESTRICT"), index=True
    )
    published_job_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("ingestion_jobs.id", ondelete="RESTRICT"), index=True
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __mapper_args__ = {"version_id_col": revision}


class ChunkEditOperationModel(Base):
    """Append-only chunk operation; order is part of the published snapshot."""

    __tablename__ = "chunk_edit_operations"
    __table_args__ = (
        UniqueConstraint("draft_id", "sequence", name="uq_chunk_edit_operation_sequence"),
        CheckConstraint(
            "operation IN ('replace', 'split', 'merge', 'delete')",
            name="ck_chunk_edit_operation_kind",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_edit_drafts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    input_chunk_ids_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    output_chunks_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IndexVersionModel(Base, TimestampMixin):
    __tablename__ = "index_versions"
    __table_args__ = (
        UniqueConstraint("kb_id", "version_no", name="uq_index_kb_version_no"),
        UniqueConstraint("kb_id", "active_slot", name="uq_index_kb_active_slot"),
        Index("ix_index_version_kb_status", "kb_id", "status"),
        CheckConstraint(
            "vector_governance_mode IN ('legacy', 'managed')",
            name="ck_index_vector_governance_mode",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    vector_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="creating")
    active_slot: Mapped[str | None] = mapped_column(String(16))
    collection_alias: Mapped[str] = mapped_column(String(255), nullable=False)
    # Alembic 0012 将升级前的存量行显式回填为 legacy；升级后新建索引默认 managed，
    # 因而“漏建 placement”不能再静默退回共享 kb_chunks。
    vector_governance_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="managed", server_default="managed"
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    previous_version_id: Mapped[str | None] = mapped_column(String(64), index=True)
    build_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="initial")
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="validated")
    manifest_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class EvidenceRecordModel(Base, TimestampMixin):
    """已发布本地 chunk 的不可变证据事实。

    ``provenance_spans_json`` 保存跨页/跨块完整位置；``primary_*`` 仅用于快速过滤和
    预览，不能代替完整 spans。
    """

    __tablename__ = "evidence_records"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "index_version_id",
            "chunk_id",
            name="uq_evidence_version_index_chunk",
        ),
        UniqueConstraint(
            "tenant_id",
            "source_url",
            "content_hash",
            name="uq_evidence_web_tenant_url_hash",
        ),
        Index("ix_evidence_tenant_kb", "tenant_id", "kb_id"),
        Index("ix_evidence_document_version", "document_id", "document_version_id"),
        Index("ix_evidence_index_chunk", "index_version_id", "chunk_id"),
        Index("ix_evidence_primary_page", "document_version_id", "primary_page_no"),
        CheckConstraint(
            "content_hash LIKE 'sha256:%' AND length(content_hash) = 71",
            name="ck_evidence_content_hash",
        ),
        CheckConstraint(
            "(source_type = 'local' AND tenant_id IS NOT NULL AND kb_id IS NOT NULL "
            "AND document_id IS NOT NULL AND document_version_id IS NOT NULL "
            "AND index_version_id IS NOT NULL AND chunk_id IS NOT NULL "
            "AND source_url IS NULL AND source_fetched_at IS NULL) OR "
            "(source_type = 'web' AND source_url IS NOT NULL AND source_fetched_at IS NOT NULL "
            "AND tenant_id IS NOT NULL "
            "AND kb_id IS NULL AND document_id IS NULL AND document_version_id IS NULL "
            "AND index_version_id IS NULL AND chunk_id IS NULL)",
            name="ck_evidence_source_shape",
        ),
        CheckConstraint(
            "primary_availability IN ('exact', 'derived', 'unavailable')",
            name="ck_evidence_primary_availability",
        ),
        CheckConstraint(
            "primary_coordinate_space IN "
            "('none', 'pdf_points', 'normalized_page', 'image_pixels', "
            "'slide_points', 'spreadsheet_cells', 'text_lines')",
            name="ck_evidence_coordinate_space",
        ),
        CheckConstraint(
            "primary_availability != 'unavailable' OR "
            "(primary_page_no IS NULL AND primary_bbox_json IS NULL AND "
            "primary_slide_no IS NULL AND primary_sheet IS NULL AND "
            "primary_cell_range IS NULL AND primary_line_start IS NULL AND "
            "primary_line_end IS NULL AND primary_coordinate_space = 'none' AND "
            "primary_reason_code IS NOT NULL)",
            name="ck_evidence_unavailable_empty_location",
        ),
        CheckConstraint(
            "primary_availability = 'unavailable' OR "
            "(primary_page_no IS NOT NULL OR primary_bbox_json IS NOT NULL OR "
            "primary_slide_no IS NOT NULL OR primary_sheet IS NOT NULL OR "
            "primary_cell_range IS NOT NULL OR primary_line_start IS NOT NULL)",
            name="ck_evidence_available_has_location",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="evidence.v1", server_default="evidence.v1"
    )
    source_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="local", server_default="local"
    )
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    kb_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    index_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("index_versions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    chunk_id: Mapped[str | None] = mapped_column(String(128))
    source_url: Mapped[str | None] = mapped_column(Text)
    source_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    source_block_ids_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    provenance_spans_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    heading_path_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    asset_refs_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    source_label: Mapped[str] = mapped_column(String(1024), nullable=False, default="")

    primary_block_id: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_availability: Mapped[str] = mapped_column(String(16), nullable=False)
    primary_coordinate_space: Mapped[str] = mapped_column(
        String(32), nullable=False, default="none"
    )
    primary_page_no: Mapped[int | None] = mapped_column(Integer)
    primary_bbox_json: Mapped[list | None] = mapped_column(JSON)
    primary_slide_no: Mapped[int | None] = mapped_column(Integer)
    primary_sheet: Mapped[str | None] = mapped_column(String(255))
    primary_cell_range: Mapped[str | None] = mapped_column(String(64))
    primary_line_start: Mapped[int | None] = mapped_column(Integer)
    primary_line_end: Mapped[int | None] = mapped_column(Integer)
    primary_reason_code: Mapped[str | None] = mapped_column(String(128))


class CitationFactModel(Base, TimestampMixin):
    """答案与 Evidence 之间经服务端校验的幂等引用事实。"""

    __tablename__ = "citation_facts"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "evidence_id",
            "answer_span_start",
            "answer_span_end",
            "quote_hash",
            name="uq_citation_answer_evidence_span_quote",
        ),
        Index("ix_citation_answer_ordinal", "answer_id", "ordinal"),
        Index("ix_citation_evidence", "evidence_id"),
        CheckConstraint(
            "answer_span_start >= 0 AND answer_span_end > answer_span_start",
            name="ck_citation_answer_span",
        ),
        CheckConstraint(
            "quote_hash LIKE 'sha256:%' AND length(quote_hash) = 71",
            name="ck_citation_quote_hash",
        ),
        CheckConstraint(
            "validation_status IN ('valid', 'invalid', 'degraded')",
            name="ck_citation_validation_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="citation.v1", server_default="citation.v1"
    )
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_records.id", ondelete="RESTRICT"), nullable=False
    )
    answer_span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_quote: Mapped[str] = mapped_column(Text, nullable=False)
    quote_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    source_label: Mapped[str] = mapped_column(String(1024), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason_codes_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class IngestionJobModel(Base, TimestampMixin):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("document_version_id", "attempt", name="uq_job_version_attempt"),
        UniqueConstraint("idempotency_key", name="uq_ingestion_job_idempotency_key"),
        Index("ix_ingestion_job_status_created", "status", "created_at"),
        Index("ix_ingestion_job_ready", "status", "next_retry_at", "priority", "created_at"),
        Index("ix_ingestion_job_lease", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config_version: Mapped[str] = mapped_column(String(64), nullable=False, default="cfg_legacy")
    operation: Mapped[str] = mapped_column(String(32), nullable=False, default="import")
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(128), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    run_attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_category: Mapped[str | None] = mapped_column(String(32))
    progress_detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    checkpoint_stage: Mapped[str | None] = mapped_column(String(128))
    checkpoint_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_event_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recovered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestionJobEventModel(Base):
    __tablename__ = "ingestion_job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "sequence", name="uq_ingestion_event_job_sequence"),
        Index("ix_ingestion_event_job_created", "job_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IngestionIdempotencyKeyModel(Base):
    __tablename__ = "ingestion_idempotency_keys"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IngestionJobCheckpointModel(Base):
    __tablename__ = "ingestion_job_checkpoints"
    __table_args__ = (
        UniqueConstraint("job_id", "sequence", name="uq_ingestion_checkpoint_job_sequence"),
        Index("ix_ingestion_checkpoint_job_created", "job_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(128), nullable=False)
    resume_node: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_size: Mapped[int] = mapped_column(Integer, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IngestionWorkerModel(Base):
    __tablename__ = "ingestion_workers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    current_job_id: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DocumentIndexBindingModel(Base):
    __tablename__ = "document_index_bindings"

    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), primary_key=True
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="RESTRICT"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="published")
    job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    manifest_id: Mapped[str | None] = mapped_column(String(64), index=True)
    staged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class IndexBuildManifestModel(Base, TimestampMixin):
    __tablename__ = "index_build_manifests"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_index_manifest_job"),
        Index("ix_index_manifest_version_status", "index_version_id", "validation_status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    written_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    searchable_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_ids_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    object_keys: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    vector_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_errors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CleanupJobModel(Base, TimestampMixin):
    __tablename__ = "cleanup_jobs"
    __table_args__ = (
        Index("ix_cleanup_job_status_created", "status", "created_at"),
        Index("ix_cleanup_job_target", "target_kind", "target_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    plan_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_detail: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserModel(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class TenantMembershipModel(Base, TimestampMixin):
    __tablename__ = "tenant_memberships"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", name="uq_membership_tenant_user"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class ApiCredentialModel(Base, TimestampMixin):
    __tablename__ = "api_credentials"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RefreshCredentialModel(Base, TimestampMixin):
    __tablename__ = "refresh_credentials"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeBaseAclModel(Base, TimestampMixin):
    __tablename__ = "knowledge_base_acl"
    __table_args__ = (
        UniqueConstraint("kb_id", "principal_type", "principal_id", name="uq_kb_acl_principal"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    principal_type: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    principal_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)


class RetrievalProfileModel(Base, TimestampMixin):
    """租户/知识库内不可变、可发布的检索配置版本。

    ``id`` 是控制库行标识并被 ``knowledge_bases.retrieval_config_id`` 用作发布
    指针；``profile_id + version`` 是稳定的领域版本标识。发布或回滚只改变
    生命周期字段，不会原地改写 ``config_json`` 与 ``config_hash``。
    """

    __tablename__ = "retrieval_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "kb_id",
            "profile_id",
            "version",
            name="uq_retrieval_profile_version",
        ),
        Index(
            "ix_retrieval_profile_tenant_kb_status",
            "tenant_id",
            "kb_id",
            "status",
        ),
        Index(
            "ix_retrieval_profile_logical_version",
            "profile_id",
            "version",
        ),
        CheckConstraint(
            "status IN ('draft', 'published', 'retired')",
            name="ck_retrieval_profile_status",
        ),
        CheckConstraint("version > 0", name="ck_retrieval_profile_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    profile_id: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MetadataCatalogEntryModel(Base, TimestampMixin):
    """由控制库维护的块元数据目录，不依赖向量库全量扫描。"""

    __tablename__ = "metadata_catalog_entries"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "kb_id",
            "document_version_id",
            "index_version_id",
            "metadata_key",
            "value_hash",
            name="uq_metadata_catalog_scope_value",
        ),
        Index(
            "ix_metadata_catalog_scope_key",
            "tenant_id",
            "kb_id",
            "index_version_id",
            "metadata_key",
        ),
        Index(
            "ix_metadata_catalog_document",
            "document_id",
            "document_version_id",
        ),
        CheckConstraint(
            "value_type IN ('string', 'integer', 'number', 'boolean', 'object')",
            name="ck_metadata_catalog_value_type",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metadata_key: Mapped[str] = mapped_column(String(255), nullable=False)
    value_type: Mapped[str] = mapped_column(String(16), nullable=False)
    value_json: Mapped[object] = mapped_column(JSON, nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class CapabilityResourceModel(Base, TimestampMixin):
    """P2 高级能力的统一版本注册表。

    领域对象仍由各自严格 Pydantic schema 校验；本表负责 tenant/KB 作用域、
    不可变版本、发布状态和 payload hash，不把七套能力退化为自由 JSON。
    """

    __tablename__ = "capability_resources"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "resource_type",
            "logical_id",
            "version",
            "scope_key",
            name="uq_capability_resource_version",
        ),
        Index(
            "ix_capability_resource_scope_status",
            "tenant_id",
            "resource_type",
            "kb_id",
            "status",
        ),
        CheckConstraint(
            "status IN ('draft', 'published', 'retired')",
            name="ck_capability_resource_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    # SQL UNIQUE 对 NULL 的处理会允许多个 tenant-scoped 同版本记录；使用稳定的
    # 非空 scope key 同时支持 tenant 级资源和不同 KB 下同名同版本资源。
    scope_key: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    logical_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CapabilityRunModel(Base, TimestampMixin):
    """路由、同步、Agent、增强和评测的可审计运行事实。"""

    __tablename__ = "capability_runs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "run_type",
            "idempotency_key",
            name="uq_capability_run_idempotency",
        ),
        Index(
            "ix_capability_run_scope_status",
            "tenant_id",
            "run_type",
            "status",
            "created_at",
        ),
        CheckConstraint(
            "status IN ('planned', 'running', 'waiting_approval', 'succeeded', "
            "'partial', 'failed', 'cancelled')",
            name="ck_capability_run_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    resource_id: Mapped[str | None] = mapped_column(
        ForeignKey("capability_resources.id", ondelete="SET NULL"), index=True
    )
    run_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="planned")
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(256))
    input_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    input_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MetadataValueRecordModel(Base, TimestampMixin):
    """TODO-15 元数据事实源；Milvus catalog 只是 published 投影。"""

    __tablename__ = "metadata_value_records"
    __table_args__ = (
        Index(
            "ix_metadata_value_scope_status",
            "tenant_id",
            "kb_id",
            "scope_type",
            "scope_id",
            "status",
        ),
        Index(
            "ix_metadata_value_schema_field",
            "schema_resource_id",
            "field_key",
            "revision",
        ),
        CheckConstraint(
            "scope_type IN ('document', 'document_version', 'chunk')",
            name="ck_metadata_value_scope",
        ),
        CheckConstraint(
            "source IN ('auto', 'manual', 'import')",
            name="ck_metadata_value_source",
        ),
        CheckConstraint(
            "status IN ('proposed', 'published', 'rejected', 'superseded')",
            name="ck_metadata_value_status",
        ),
        CheckConstraint(
            "(scope_type = 'document' AND document_version_id IS NULL "
            "AND chunk_id IS NULL) OR "
            "(scope_type = 'document_version' AND document_version_id IS NOT NULL "
            "AND chunk_id IS NULL) OR "
            "(scope_type = 'chunk' AND document_version_id IS NOT NULL "
            "AND chunk_id IS NOT NULL)",
            name="ck_metadata_value_target_shape",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_metadata_value_confidence",
        ),
        CheckConstraint(
            "source != 'auto' OR confidence IS NOT NULL",
            name="ck_metadata_value_auto_confidence",
        ),
        CheckConstraint(
            "(status = 'proposed' AND reviewed_by IS NULL AND reviewed_at IS NULL) OR "
            "(status != 'proposed' AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)",
            name="ck_metadata_value_review",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    schema_resource_id: Mapped[str] = mapped_column(
        ForeignKey("capability_resources.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    schema_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    scope_type: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str | None] = mapped_column(String(128), index=True)
    target_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    field_key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_json: Mapped[object] = mapped_column(JSON, nullable=False)
    value_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    proposed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_reason: Mapped[str | None] = mapped_column(Text)
    enhancement_json: Mapped[dict | None] = mapped_column(JSON)
    provenance_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("metadata_value_records.id", ondelete="SET NULL")
    )


class ConnectorBindingModel(Base, TimestampMixin):
    """TODO-17 远端 source item 与统一 Document 的稳定映射。"""

    __tablename__ = "connector_bindings"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "kb_id",
            "data_source_id",
            "source_item_id",
            name="uq_connector_source_item",
        ),
        Index(
            "ix_connector_binding_scope_status",
            "tenant_id",
            "kb_id",
            "data_source_id",
            "status",
        ),
        CheckConstraint(
            "status IN ('pending', 'active', 'disabled', 'soft_deleted', 'purged', 'failed')",
            name="ck_connector_binding_status",
        ),
        CheckConstraint(
            "content_sha256 IS NULL OR "
            "(length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*')",
            name="ck_connector_binding_content_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data_source_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_item_id: Mapped[str] = mapped_column(String(512), nullable=False)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    remote_version: Mapped[str] = mapped_column(String(256), nullable=False)
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    logical_name: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    locator: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    active_document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL"), index=True
    )
    pending_document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL"), index=True
    )
    last_sync_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("capability_runs.id", ondelete="SET NULL"), index=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    purge_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConnectorCursorModel(Base, TimestampMixin):
    """游标只在事件事实落库后推进。"""

    __tablename__ = "connector_cursors"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "kb_id",
            "data_source_id",
            name="uq_connector_cursor_source",
        ),
        CheckConstraint("generation >= 0", name="ck_connector_cursor_generation"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data_source_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    cursor_value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cursor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("capability_runs.id", ondelete="SET NULL"), index=True
    )
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LongTermMemoryModel(Base, TimestampMixin):
    """TODO-22 用户长期记忆；与知识库和会话历史物理分离。"""

    __tablename__ = "long_term_memories"
    __table_args__ = (
        Index(
            "ix_memory_subject_status_expiry",
            "tenant_id",
            "subject_user_id",
            "kb_id",
            "status",
            "expires_at",
        ),
        Index("ix_memory_lineage", "tenant_id", "kb_id", "supersedes_id"),
        Index(
            "ix_memory_conflict",
            "tenant_id",
            "subject_user_id",
            "kb_id",
            "conflict_key",
            "status",
        ),
        UniqueConstraint(
            "tenant_id",
            "subject_user_id",
            "kb_id",
            "candidate_id",
            name="uq_memory_candidate_scope",
        ),
        CheckConstraint(
            "memory_type IN ('preference', 'profile_fact', 'episodic_summary')",
            name="ck_memory_type",
        ),
        CheckConstraint(
            "status IN "
            "('proposed', 'confirmed', 'active', 'superseded', 'forgotten', "
            "'expired', 'rejected')",
            name="ck_memory_status",
        ),
        CheckConstraint(
            "consent IN ('none', 'explicit_request', 'confirmed', 'withdrawn')",
            name="ck_memory_consent",
        ),
        CheckConstraint(
            "status != 'active' OR consent = 'confirmed'",
            name="ck_memory_active_consent",
        ),
        CheckConstraint(
            "status != 'forgotten' OR (content = '' AND forgotten_at IS NOT NULL)",
            name="ck_memory_forgotten_redacted",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    subject_user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    structured_value: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    conflict_key: Mapped[str] = mapped_column(String(256), nullable=False)
    provenance_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(32), nullable=False)
    consent: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_version: Mapped[str | None] = mapped_column(String(128))
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("long_term_memories.id", ondelete="SET NULL")
    )
    forgotten_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MediaAssetModel(Base, TimestampMixin):
    """TODO-23 原始媒体资产与派生链控制事实。"""

    __tablename__ = "media_assets"
    __table_args__ = (
        Index(
            "ix_media_asset_scope_status",
            "tenant_id",
            "kb_id",
            "document_version_id",
            "status",
        ),
        Index("ix_media_asset_owner", "tenant_id", "owner_user_id", "status"),
        CheckConstraint(
            "source_kind IN ('document_asset', 'query_attachment')",
            name="ck_media_asset_source_kind",
        ),
        CheckConstraint(
            "modality IN ('image', 'audio', 'video')",
            name="ck_media_asset_modality",
        ),
        CheckConstraint(
            "status IN ('active', 'quarantined', 'retired', 'deleted')",
            name="ck_media_asset_status",
        ),
        CheckConstraint(
            "(source_kind = 'document_asset' AND kb_id IS NOT NULL "
            "AND document_id IS NOT NULL AND document_version_id IS NOT NULL "
            "AND owner_user_id IS NULL) OR "
            "(source_kind = 'query_attachment' AND kb_id IS NULL "
            "AND document_id IS NULL AND document_version_id IS NULL "
            "AND owner_user_id IS NOT NULL)",
            name="ck_media_asset_scope_shape",
        ),
        CheckConstraint(
            "classification IN ('public', 'internal', 'confidential', 'restricted')",
            name="ck_media_asset_classification",
        ),
        CheckConstraint("size_bytes > 0", name="ck_media_asset_size"),
        CheckConstraint(
            "content_hash LIKE 'sha256:%' AND length(content_hash) = 71",
            name="ck_media_asset_content_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    kb_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    owner_user_id: Mapped[str | None] = mapped_column(String(128), index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    location_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    asset_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    derivations_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QuerySessionModel(Base, TimestampMixin):
    __tablename__ = "query_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    config_version: Mapped[str] = mapped_column(String(64), nullable=False, default="cfg_legacy")
    retrieval_profile_id: Mapped[str | None] = mapped_column(String(64), index=True)
    retrieval_profile_version: Mapped[int | None] = mapped_column(Integer)
    retrieval_profile_hash: Mapped[str | None] = mapped_column(String(64))


class AuditEventModel(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)
    actor_id: Mapped[str | None] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(128))
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvaluationDatasetModel(Base, TimestampMixin):
    """An immutable, tenant-scoped evaluation dataset version."""

    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("tenant_id", "dataset_key", "version", name="uq_eval_dataset_version"),
        Index("ix_eval_dataset_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_key: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    source_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    parent_dataset_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")


class EvaluationCaseModel(Base):
    __tablename__ = "evaluation_cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_id", name="uq_eval_case_dataset_case"),
        Index("ix_eval_case_dataset_order", "dataset_id", "ordinal"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    conversation: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    unanswerable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    relevant_document_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    relevant_document_names: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    relevant_chunk_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    relevant_evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    expected_terms: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    forbidden_terms: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    sensitivity: Mapped[str] = mapped_column(String(32), nullable=False, default="internal")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvaluationRunModel(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_eval_run_tenant_created", "tenant_id", "created_at"),
        Index("ix_eval_run_dataset_status", "dataset_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    baseline_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="live")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    comparison_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    gate_result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    report_paths: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        UniqueConstraint("run_id", "case_id", name="uq_eval_result_run_case"),
        Index("ix_eval_result_run_passed", "run_id", "passed"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    observation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_detail: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FeedbackModel(Base, TimestampMixin):
    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_tenant_status", "tenant_id", "status"),
        Index("ix_feedback_answer", "answer_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    answer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluation_result_id: Mapped[str | None] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    reason_codes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    selected_citation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    corrected_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    reviewed_by: Mapped[str | None] = mapped_column(String(64))
    review_note: Mapped[str] = mapped_column(Text, nullable=False, default="")


class AnnotationTaskModel(Base, TimestampMixin):
    __tablename__ = "annotation_tasks"
    __table_args__ = (Index("ix_annotation_tenant_status", "tenant_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feedback_id: Mapped[str] = mapped_column(
        ForeignKey("feedback.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    target_dataset_key: Mapped[str] = mapped_column(String(128), nullable=False)
    proposed_case: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    reviews: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    arbiter_id: Mapped[str | None] = mapped_column(String(64))
    resolution: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
