"""TODO-12 workbench optimistic revisions and persistent chunk edit drafts.

Revision ID: 20260809_0014
Revises: 20260807_0013
Create Date: 2026-08-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20260809_0014"
down_revision = "20260807_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.add_column(
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1")
        )
    with op.batch_alter_table("documents") as batch:
        batch.add_column(
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1")
        )

    op.create_table(
        "chunk_edit_drafts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("kb_id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("document_version_id", sa.String(length=64), nullable=False),
        sa.Column("base_chunking_run_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("operations_hash", sa.String(length=64), nullable=True),
        sa.Column("frozen_revision", sa.Integer(), nullable=True),
        sa.Column("published_document_version_id", sa.String(length=64), nullable=True),
        sa.Column("published_index_version_id", sa.String(length=64), nullable=True),
        sa.Column("published_job_id", sa.String(length=64), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'publishing', 'indexed', 'published', 'abandoned')",
            name="ck_chunk_edit_draft_status",
        ),
        sa.CheckConstraint(
            "frozen_revision IS NULL OR frozen_revision >= 1",
            name="ck_chunk_edit_draft_frozen_revision",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["base_chunking_run_id"],
            ["document_chunking_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["published_document_version_id"],
            ["document_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["published_index_version_id"],
            ["index_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["published_job_id"], ["ingestion_jobs.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "published_document_version_id",
            name="uq_chunk_edit_draft_published_document_version",
        ),
        sa.UniqueConstraint(
            "published_job_id", name="uq_chunk_edit_draft_published_job"
        ),
    )
    op.create_index(
        "ix_chunk_edit_draft_scope_status",
        "chunk_edit_drafts",
        ["tenant_id", "kb_id", "status"],
    )
    op.create_index(
        "ix_chunk_edit_draft_document_version",
        "chunk_edit_drafts",
        ["document_version_id", "created_at"],
    )
    for column in ("tenant_id", "kb_id", "document_id", "document_version_id"):
        op.create_index(
            f"ix_chunk_edit_drafts_{column}", "chunk_edit_drafts", [column]
        )
    op.create_index(
        "ix_chunk_edit_drafts_published_document_version_id",
        "chunk_edit_drafts",
        ["published_document_version_id"],
    )
    op.create_index(
        "ix_chunk_edit_drafts_published_index_version_id",
        "chunk_edit_drafts",
        ["published_index_version_id"],
    )
    op.create_index(
        "ix_chunk_edit_drafts_published_job_id",
        "chunk_edit_drafts",
        ["published_job_id"],
    )

    op.create_table(
        "chunk_edit_operations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("draft_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=16), nullable=False),
        sa.Column("input_chunk_ids_json", sa.JSON(), nullable=False),
        sa.Column("output_chunks_json", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "operation IN ('replace', 'split', 'merge', 'delete')",
            name="ck_chunk_edit_operation_kind",
        ),
        sa.ForeignKeyConstraint(["draft_id"], ["chunk_edit_drafts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("draft_id", "sequence", name="uq_chunk_edit_operation_sequence"),
    )
    op.create_index(
        "ix_chunk_edit_operations_draft_id",
        "chunk_edit_operations",
        ["draft_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_edit_operations_draft_id", table_name="chunk_edit_operations")
    op.drop_table("chunk_edit_operations")
    op.drop_index(
        "ix_chunk_edit_drafts_published_job_id", table_name="chunk_edit_drafts"
    )
    op.drop_index(
        "ix_chunk_edit_drafts_published_index_version_id",
        table_name="chunk_edit_drafts",
    )
    op.drop_index(
        "ix_chunk_edit_drafts_published_document_version_id",
        table_name="chunk_edit_drafts",
    )
    op.drop_index(
        "ix_chunk_edit_draft_document_version", table_name="chunk_edit_drafts"
    )
    for column in ("document_version_id", "document_id", "kb_id", "tenant_id"):
        op.drop_index(
            f"ix_chunk_edit_drafts_{column}", table_name="chunk_edit_drafts"
        )
    op.drop_index("ix_chunk_edit_draft_scope_status", table_name="chunk_edit_drafts")
    op.drop_table("chunk_edit_drafts")
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("revision")
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_column("revision")
