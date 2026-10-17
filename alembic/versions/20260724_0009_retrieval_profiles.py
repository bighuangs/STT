"""TODO-09 retrieval profiles, metadata catalog and query-session pinning.

Revision ID: 20260724_0009
Revises: 20260723_0008
"""

from alembic import op
import sqlalchemy as sa


revision = "20260724_0009"
down_revision = "20260723_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_profiles",
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
            sa.ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("profile_id", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
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
        sa.UniqueConstraint(
            "tenant_id",
            "kb_id",
            "profile_id",
            "version",
            name="uq_retrieval_profile_version",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'published', 'retired')",
            name="ck_retrieval_profile_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_retrieval_profile_version"),
    )
    op.create_index(
        "ix_retrieval_profiles_tenant_id",
        "retrieval_profiles",
        ["tenant_id"],
    )
    op.create_index(
        "ix_retrieval_profiles_kb_id",
        "retrieval_profiles",
        ["kb_id"],
    )
    op.create_index(
        "ix_retrieval_profile_tenant_kb_status",
        "retrieval_profiles",
        ["tenant_id", "kb_id", "status"],
    )
    op.create_index(
        "ix_retrieval_profile_logical_version",
        "retrieval_profiles",
        ["profile_id", "version"],
    )

    op.create_table(
        "metadata_catalog_entries",
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
            sa.ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
            nullable=False,
        ),
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
            nullable=False,
        ),
        sa.Column(
            "index_version_id",
            sa.String(64),
            sa.ForeignKey("index_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metadata_key", sa.String(255), nullable=False),
        sa.Column("value_type", sa.String(16), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("value_text", sa.Text(), nullable=False),
        sa.Column("value_hash", sa.String(64), nullable=False),
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
        sa.UniqueConstraint(
            "tenant_id",
            "kb_id",
            "document_version_id",
            "index_version_id",
            "metadata_key",
            "value_hash",
            name="uq_metadata_catalog_scope_value",
        ),
        sa.CheckConstraint(
            "value_type IN ('string', 'integer', 'number', 'boolean', 'object')",
            name="ck_metadata_catalog_value_type",
        ),
    )
    op.create_index(
        "ix_metadata_catalog_entries_tenant_id",
        "metadata_catalog_entries",
        ["tenant_id"],
    )
    op.create_index(
        "ix_metadata_catalog_entries_kb_id",
        "metadata_catalog_entries",
        ["kb_id"],
    )
    op.create_index(
        "ix_metadata_catalog_entries_document_id",
        "metadata_catalog_entries",
        ["document_id"],
    )
    op.create_index(
        "ix_metadata_catalog_entries_document_version_id",
        "metadata_catalog_entries",
        ["document_version_id"],
    )
    op.create_index(
        "ix_metadata_catalog_entries_index_version_id",
        "metadata_catalog_entries",
        ["index_version_id"],
    )
    op.create_index(
        "ix_metadata_catalog_scope_key",
        "metadata_catalog_entries",
        ["tenant_id", "kb_id", "index_version_id", "metadata_key"],
    )
    op.create_index(
        "ix_metadata_catalog_document",
        "metadata_catalog_entries",
        ["document_id", "document_version_id"],
    )

    with op.batch_alter_table("query_sessions") as batch:
        batch.add_column(sa.Column("retrieval_profile_id", sa.String(64)))
        batch.add_column(sa.Column("retrieval_profile_version", sa.Integer()))
        batch.add_column(sa.Column("retrieval_profile_hash", sa.String(64)))
        batch.create_index(
            "ix_query_sessions_retrieval_profile_id",
            ["retrieval_profile_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("query_sessions") as batch:
        batch.drop_index("ix_query_sessions_retrieval_profile_id")
        batch.drop_column("retrieval_profile_hash")
        batch.drop_column("retrieval_profile_version")
        batch.drop_column("retrieval_profile_id")
    op.drop_table("metadata_catalog_entries")
    op.drop_table("retrieval_profiles")
