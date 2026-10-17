"""TODO-05 config version trace for ingestion and query.

Revision ID: 20260716_0005
Revises: 20260715_0004
"""
from alembic import op
import sqlalchemy as sa

revision = "20260716_0005"
down_revision = "20260715_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("config_version", sa.String(64), nullable=False, server_default="cfg_legacy"),
    )
    op.add_column(
        "query_sessions",
        sa.Column("config_version", sa.String(64), nullable=False, server_default="cfg_legacy"),
    )


def downgrade() -> None:
    op.drop_column("query_sessions", "config_version")
    op.drop_column("ingestion_jobs", "config_version")
