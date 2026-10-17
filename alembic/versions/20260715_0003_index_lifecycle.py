"""TODO-03 index manifests, atomic publication, rebuild and cleanup.

Revision ID: 20260715_0003
Revises: 20260715_0002
"""

from alembic import op
import sqlalchemy as sa


revision = "20260715_0003"
down_revision = "20260715_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("index_versions", sa.Column("previous_version_id", sa.String(64)))
    op.add_column("index_versions", sa.Column("build_kind", sa.String(32), nullable=False, server_default="initial"))
    op.add_column("index_versions", sa.Column("validation_status", sa.String(32), nullable=False, server_default="validated"))
    op.add_column("index_versions", sa.Column("manifest_summary", sa.JSON(), nullable=False, server_default="{}"))
    op.create_index("ix_index_versions_previous_version_id", "index_versions", ["previous_version_id"])

    op.add_column("ingestion_jobs", sa.Column("operation", sa.String(32), nullable=False, server_default="import"))

    with op.batch_alter_table("document_index_bindings", recreate="always") as batch:
        batch.add_column(sa.Column("status", sa.String(32), nullable=False, server_default="published"))
        batch.add_column(sa.Column("job_id", sa.String(64)))
        batch.add_column(sa.Column("manifest_id", sa.String(64)))
        batch.add_column(sa.Column("staged_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("retired_at", sa.DateTime(timezone=True)))
        batch.alter_column("published_at", existing_type=sa.DateTime(timezone=True), nullable=True)
    op.create_index("ix_document_index_bindings_job_id", "document_index_bindings", ["job_id"])
    op.create_index("ix_document_index_bindings_manifest_id", "document_index_bindings", ["manifest_id"])

    op.create_table(
        "index_build_manifests",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_version_id", sa.String(64), sa.ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("index_version_id", sa.String(64), sa.ForeignKey("index_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("config_sha256", sa.String(64), nullable=False),
        sa.Column("expected_chunk_count", sa.Integer(), nullable=False),
        sa.Column("written_chunk_count", sa.Integer(), nullable=False),
        sa.Column("searchable_chunk_count", sa.Integer(), nullable=False),
        sa.Column("chunk_ids_sha256", sa.String(64), nullable=False),
        sa.Column("chunk_ids", sa.JSON(), nullable=False),
        sa.Column("object_keys", sa.JSON(), nullable=False),
        sa.Column("vector_dim", sa.Integer(), nullable=False),
        sa.Column("collection_name", sa.String(255), nullable=False),
        sa.Column("validation_status", sa.String(32), nullable=False),
        sa.Column("validation_errors", sa.JSON(), nullable=False),
        sa.Column("validated_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("job_id", name="uq_index_manifest_job"),
    )
    op.create_index("ix_index_build_manifests_job_id", "index_build_manifests", ["job_id"])
    op.create_index("ix_index_build_manifests_document_version_id", "index_build_manifests", ["document_version_id"])
    op.create_index("ix_index_build_manifests_index_version_id", "index_build_manifests", ["index_version_id"])
    op.create_index("ix_index_manifest_version_status", "index_build_manifests", ["index_version_id", "validation_status"])

    op.create_table(
        "cleanup_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("target_kind", sa.String(32), nullable=False),
        sa.Column("target_id", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("error_detail", sa.Text()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_cleanup_job_status_created", "cleanup_jobs", ["status", "created_at"])
    op.create_index("ix_cleanup_job_target", "cleanup_jobs", ["target_kind", "target_id"])


def downgrade() -> None:
    op.drop_index("ix_cleanup_job_target", table_name="cleanup_jobs")
    op.drop_index("ix_cleanup_job_status_created", table_name="cleanup_jobs")
    op.drop_table("cleanup_jobs")
    op.drop_index("ix_index_manifest_version_status", table_name="index_build_manifests")
    op.drop_index("ix_index_build_manifests_index_version_id", table_name="index_build_manifests")
    op.drop_index("ix_index_build_manifests_document_version_id", table_name="index_build_manifests")
    op.drop_index("ix_index_build_manifests_job_id", table_name="index_build_manifests")
    op.drop_table("index_build_manifests")
    op.drop_index("ix_document_index_bindings_manifest_id", table_name="document_index_bindings")
    op.drop_index("ix_document_index_bindings_job_id", table_name="document_index_bindings")
    with op.batch_alter_table("document_index_bindings", recreate="always") as batch:
        batch.alter_column("published_at", existing_type=sa.DateTime(timezone=True), nullable=False)
        batch.drop_column("retired_at")
        batch.drop_column("staged_at")
        batch.drop_column("manifest_id")
        batch.drop_column("job_id")
        batch.drop_column("status")
    op.drop_column("ingestion_jobs", "operation")
    op.drop_index("ix_index_versions_previous_version_id", table_name="index_versions")
    op.drop_column("index_versions", "manifest_summary")
    op.drop_column("index_versions", "validation_status")
    op.drop_column("index_versions", "build_kind")
    op.drop_column("index_versions", "previous_version_id")
