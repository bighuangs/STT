"""TODO-01 control plane domain baseline.

Revision ID: 20260715_0001
Revises:
Create Date: 2026-07-15
"""

from alembic import op
import sqlalchemy as sa


revision = "20260715_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    def timestamps():
        return (
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("status", sa.String(32), nullable=False),
        *timestamps(),
    )
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("tenant_id", "name", name="uq_workspace_tenant_name"),
    )
    op.create_index("ix_workspaces_tenant_id", "workspaces", ["tenant_id"])
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("workspace_id", sa.String(64), sa.ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("retrieval_config_id", sa.String(64)),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.UniqueConstraint("tenant_id", "name", name="uq_kb_tenant_name"),
    )
    op.create_index("ix_knowledge_bases_tenant_id", "knowledge_bases", ["tenant_id"])
    op.create_index("ix_knowledge_bases_workspace_id", "knowledge_bases", ["workspace_id"])
    op.create_index("ix_kb_workspace_status", "knowledge_bases", ["workspace_id", "status"])
    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kb_id", sa.String(64), sa.ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("logical_name", sa.String(512), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_uri", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("active_version_id", sa.String(64)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.UniqueConstraint("kb_id", "logical_name", name="uq_document_kb_logical_name"),
    )
    op.create_index("ix_documents_kb_id", "documents", ["kb_id"])
    op.create_index("ix_documents_active_version_id", "documents", ["active_version_id"])
    op.create_index("ix_document_kb_status", "documents", ["kb_id", "status"])
    op.create_table(
        "document_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("parser_config_snapshot", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.UniqueConstraint("document_id", "version_no", name="uq_document_version_no"),
        sa.UniqueConstraint("document_id", "content_sha256", name="uq_document_content_hash"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_version_status", "document_versions", ["document_id", "status"])
    op.create_table(
        "index_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kb_id", sa.String(64), sa.ForeignKey("knowledge_bases.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("embedding_model_id", sa.String(255), nullable=False),
        sa.Column("vector_dim", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("active_slot", sa.String(16)),
        sa.Column("collection_alias", sa.String(255), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.UniqueConstraint("kb_id", "version_no", name="uq_index_kb_version_no"),
        sa.UniqueConstraint("kb_id", "active_slot", name="uq_index_kb_active_slot"),
    )
    op.create_index("ix_index_versions_kb_id", "index_versions", ["kb_id"])
    op.create_index("ix_index_version_kb_status", "index_versions", ["kb_id", "status"])
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("document_version_id", sa.String(64), sa.ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("index_version_id", sa.String(64), sa.ForeignKey("index_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(128), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(128)),
        sa.Column("error_detail", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.UniqueConstraint("document_version_id", "attempt", name="uq_job_version_attempt"),
    )
    op.create_index("ix_ingestion_jobs_document_version_id", "ingestion_jobs", ["document_version_id"])
    op.create_index("ix_ingestion_jobs_index_version_id", "ingestion_jobs", ["index_version_id"])
    op.create_index("ix_ingestion_job_status_created", "ingestion_jobs", ["status", "created_at"])
    op.create_table(
        "document_index_bindings",
        sa.Column("document_version_id", sa.String(64), sa.ForeignKey("document_versions.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("index_version_id", sa.String(64), sa.ForeignKey("index_versions.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("published_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("document_index_bindings")
    op.drop_index("ix_ingestion_job_status_created", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_index_version_id", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_document_version_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_index_version_kb_status", table_name="index_versions")
    op.drop_index("ix_index_versions_kb_id", table_name="index_versions")
    op.drop_table("index_versions")
    op.drop_index("ix_document_version_status", table_name="document_versions")
    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_index("ix_document_kb_status", table_name="documents")
    op.drop_index("ix_documents_active_version_id", table_name="documents")
    op.drop_index("ix_documents_kb_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_kb_workspace_status", table_name="knowledge_bases")
    op.drop_index("ix_knowledge_bases_workspace_id", table_name="knowledge_bases")
    op.drop_index("ix_knowledge_bases_tenant_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
    op.drop_index("ix_workspaces_tenant_id", table_name="workspaces")
    op.drop_table("workspaces")
    op.drop_table("tenants")
