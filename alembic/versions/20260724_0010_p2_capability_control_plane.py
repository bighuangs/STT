"""P2 advanced-capability registry, runs, metadata, connectors, memory and media.

Revision ID: 20260724_0010
Revises: 20260724_0009
"""

from alembic import op
import sqlalchemy as sa


revision = "20260724_0010"
down_revision = "20260724_0009"
branch_labels = None
depends_on = None


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def upgrade() -> None:
    op.create_table(
        "capability_resources",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        ),
        sa.Column("scope_key", sa.String(80), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("logical_id", sa.String(128), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("owner_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(71), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "resource_type",
            "logical_id",
            "version",
            "scope_key",
            name="uq_capability_resource_version",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'published', 'retired')",
            name="ck_capability_resource_status",
        ),
    )
    for column in ("tenant_id", "kb_id"):
        op.create_index(
            f"ix_capability_resources_{column}",
            "capability_resources",
            [column],
        )
    op.create_index(
        "ix_capability_resource_scope_status",
        "capability_resources",
        ["tenant_id", "resource_type", "kb_id", "status"],
    )

    op.create_table(
        "capability_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "resource_id",
            sa.String(64),
            sa.ForeignKey("capability_resources.id", ondelete="SET NULL"),
        ),
        sa.Column("run_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="planned"),
        sa.Column("trace_id", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(256)),
        sa.Column("input_hash", sa.String(71), nullable=False),
        sa.Column("input_summary", sa.JSON(), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(128)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "run_type",
            "idempotency_key",
            name="uq_capability_run_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('planned', 'running', 'waiting_approval', 'succeeded', "
            "'partial', 'failed', 'cancelled')",
            name="ck_capability_run_status",
        ),
    )
    for column in ("tenant_id", "resource_id", "trace_id"):
        op.create_index(
            f"ix_capability_runs_{column}",
            "capability_runs",
            [column],
        )
    op.create_index(
        "ix_capability_run_scope_status",
        "capability_runs",
        ["tenant_id", "run_type", "status", "created_at"],
    )

    op.create_table(
        "metadata_value_records",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "schema_resource_id",
            sa.String(64),
            sa.ForeignKey("capability_resources.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("schema_id", sa.String(128), nullable=False),
        sa.Column("schema_revision", sa.Integer(), nullable=False),
        sa.Column("scope_type", sa.String(32), nullable=False),
        sa.Column("scope_id", sa.String(1024), nullable=False),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="CASCADE"),
        ),
        sa.Column("chunk_id", sa.String(128)),
        sa.Column("target_json", sa.JSON(), nullable=False),
        sa.Column("field_key", sa.String(128), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("value_hash", sa.String(71), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("status", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column("proposed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_by", sa.String(128)),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("review_reason", sa.Text()),
        sa.Column("enhancement_json", sa.JSON()),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column(
            "supersedes_id",
            sa.String(64),
            sa.ForeignKey("metadata_value_records.id", ondelete="SET NULL"),
        ),
        *_timestamps(),
        sa.CheckConstraint(
            "scope_type IN ('document', 'document_version', 'chunk')",
            name="ck_metadata_value_scope",
        ),
        sa.CheckConstraint(
            "source IN ('auto', 'manual', 'import')",
            name="ck_metadata_value_source",
        ),
        sa.CheckConstraint(
            "status IN ('proposed', 'published', 'rejected', 'superseded')",
            name="ck_metadata_value_status",
        ),
        sa.CheckConstraint(
            "(scope_type = 'document' AND document_version_id IS NULL "
            "AND chunk_id IS NULL) OR "
            "(scope_type = 'document_version' AND document_version_id IS NOT NULL "
            "AND chunk_id IS NULL) OR "
            "(scope_type = 'chunk' AND document_version_id IS NOT NULL "
            "AND chunk_id IS NOT NULL)",
            name="ck_metadata_value_target_shape",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_metadata_value_confidence",
        ),
        sa.CheckConstraint(
            "source != 'auto' OR confidence IS NOT NULL",
            name="ck_metadata_value_auto_confidence",
        ),
        sa.CheckConstraint(
            "(status = 'proposed' AND reviewed_by IS NULL AND reviewed_at IS NULL) OR "
            "(status != 'proposed' AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)",
            name="ck_metadata_value_review",
        ),
    )
    for column in (
        "tenant_id",
        "kb_id",
        "schema_resource_id",
        "document_id",
        "document_version_id",
        "chunk_id",
    ):
        op.create_index(
            f"ix_metadata_value_records_{column}",
            "metadata_value_records",
            [column],
        )
    op.create_index(
        "ix_metadata_value_scope_status",
        "metadata_value_records",
        ["tenant_id", "kb_id", "scope_type", "scope_id", "status"],
    )
    op.create_index(
        "ix_metadata_value_schema_field",
        "metadata_value_records",
        ["schema_resource_id", "field_key", "revision"],
    )

    op.create_table(
        "connector_bindings",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("data_source_id", sa.String(128), nullable=False),
        sa.Column("source_item_id", sa.String(512), nullable=False),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("remote_version", sa.String(256), nullable=False),
        sa.Column("content_sha256", sa.String(64)),
        sa.Column("logical_name", sa.String(1024), nullable=False, server_default=""),
        sa.Column("locator", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column(
            "active_document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "pending_document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "last_sync_run_id",
            sa.String(64),
            sa.ForeignKey("capability_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.Column("disabled_at", sa.DateTime(timezone=True)),
        sa.Column("purge_after", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "kb_id",
            "data_source_id",
            "source_item_id",
            name="uq_connector_source_item",
        ),
        sa.CheckConstraint(
            "status IN "
            "('pending', 'active', 'disabled', 'soft_deleted', 'purged', 'failed')",
            name="ck_connector_binding_status",
        ),
        sa.CheckConstraint(
            "content_sha256 IS NULL OR "
            "(length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*')",
            name="ck_connector_binding_content_hash",
        ),
    )
    for column in (
        "tenant_id",
        "kb_id",
        "data_source_id",
        "document_id",
        "active_document_version_id",
        "pending_document_version_id",
        "last_sync_run_id",
    ):
        op.create_index(
            f"ix_connector_bindings_{column}",
            "connector_bindings",
            [column],
        )
    op.create_index(
        "ix_connector_binding_scope_status",
        "connector_bindings",
        ["tenant_id", "kb_id", "data_source_id", "status"],
    )

    op.create_table(
        "connector_cursors",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("data_source_id", sa.String(128), nullable=False),
        sa.Column("cursor_value", sa.Text(), nullable=False, server_default=""),
        sa.Column("cursor_hash", sa.String(64), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "last_run_id",
            sa.String(64),
            sa.ForeignKey("capability_runs.id", ondelete="SET NULL"),
        ),
        sa.Column("committed_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "kb_id",
            "data_source_id",
            name="uq_connector_cursor_source",
        ),
        sa.CheckConstraint(
            "generation >= 0",
            name="ck_connector_cursor_generation",
        ),
    )
    for column in ("tenant_id", "kb_id", "data_source_id", "last_run_id"):
        op.create_index(
            f"ix_connector_cursors_{column}",
            "connector_cursors",
            [column],
        )

    op.create_table(
        "long_term_memories",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("subject_user_id", sa.String(128), nullable=False),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("memory_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(71), nullable=False),
        sa.Column("structured_value", sa.JSON(), nullable=False),
        sa.Column("conflict_key", sa.String(256), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sensitivity", sa.String(32), nullable=False),
        sa.Column("consent", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column("embedding_version", sa.String(128)),
        sa.Column(
            "supersedes_id",
            sa.String(64),
            sa.ForeignKey("long_term_memories.id", ondelete="SET NULL"),
        ),
        sa.Column("forgotten_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id",
            "subject_user_id",
            "kb_id",
            "candidate_id",
            name="uq_memory_candidate_scope",
        ),
        sa.CheckConstraint(
            "memory_type IN ('preference', 'profile_fact', 'episodic_summary')",
            name="ck_memory_type",
        ),
        sa.CheckConstraint(
            "status IN "
            "('proposed', 'confirmed', 'active', 'superseded', 'forgotten', "
            "'expired', 'rejected')",
            name="ck_memory_status",
        ),
        sa.CheckConstraint(
            "consent IN ('none', 'explicit_request', 'confirmed', 'withdrawn')",
            name="ck_memory_consent",
        ),
        sa.CheckConstraint(
            "status != 'active' OR consent = 'confirmed'",
            name="ck_memory_active_consent",
        ),
        sa.CheckConstraint(
            "status != 'forgotten' OR (content = '' AND forgotten_at IS NOT NULL)",
            name="ck_memory_forgotten_redacted",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_confidence",
        ),
    )
    for column in (
        "tenant_id",
        "subject_user_id",
        "kb_id",
        "candidate_id",
        "expires_at",
    ):
        op.create_index(
            f"ix_long_term_memories_{column}",
            "long_term_memories",
            [column],
        )
    op.create_index(
        "ix_memory_subject_status_expiry",
        "long_term_memories",
        ["tenant_id", "subject_user_id", "kb_id", "status", "expires_at"],
    )
    op.create_index(
        "ix_memory_lineage",
        "long_term_memories",
        ["tenant_id", "kb_id", "supersedes_id"],
    )
    op.create_index(
        "ix_memory_conflict",
        "long_term_memories",
        ["tenant_id", "subject_user_id", "kb_id", "conflict_key", "status"],
    )

    op.create_table(
        "media_assets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "kb_id",
            sa.String(64),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="CASCADE"),
        ),
        sa.Column("owner_user_id", sa.String(128)),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(71), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("location_json", sa.JSON(), nullable=False),
        sa.Column("asset_json", sa.JSON(), nullable=False),
        sa.Column("derivations_json", sa.JSON(), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        *_timestamps(),
        sa.CheckConstraint(
            "source_kind IN ('document_asset', 'query_attachment')",
            name="ck_media_asset_source_kind",
        ),
        sa.CheckConstraint(
            "modality IN ('image', 'audio', 'video')",
            name="ck_media_asset_modality",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'quarantined', 'retired', 'deleted')",
            name="ck_media_asset_status",
        ),
        sa.CheckConstraint(
            "(source_kind = 'document_asset' AND kb_id IS NOT NULL "
            "AND document_id IS NOT NULL AND document_version_id IS NOT NULL "
            "AND owner_user_id IS NULL) OR "
            "(source_kind = 'query_attachment' AND kb_id IS NULL "
            "AND document_id IS NULL AND document_version_id IS NULL "
            "AND owner_user_id IS NOT NULL)",
            name="ck_media_asset_scope_shape",
        ),
        sa.CheckConstraint(
            "classification IN ('public', 'internal', 'confidential', 'restricted')",
            name="ck_media_asset_classification",
        ),
        sa.CheckConstraint(
            "size_bytes > 0",
            name="ck_media_asset_size",
        ),
        sa.CheckConstraint(
            "content_hash LIKE 'sha256:%' AND length(content_hash) = 71",
            name="ck_media_asset_content_hash",
        ),
    )
    for column in (
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "owner_user_id",
    ):
        op.create_index(
            f"ix_media_assets_{column}",
            "media_assets",
            [column],
        )
    op.create_index(
        "ix_media_asset_scope_status",
        "media_assets",
        ["tenant_id", "kb_id", "document_version_id", "status"],
    )
    op.create_index(
        "ix_media_asset_owner",
        "media_assets",
        ["tenant_id", "owner_user_id", "status"],
    )


def downgrade() -> None:
    op.drop_table("media_assets")
    op.drop_table("long_term_memories")
    op.drop_table("connector_cursors")
    op.drop_table("connector_bindings")
    op.drop_table("metadata_value_records")
    op.drop_table("capability_runs")
    op.drop_table("capability_resources")
