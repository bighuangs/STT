"""TODO-04 authentication, tenant RBAC and resource isolation.

Revision ID: 20260715_0004
Revises: 20260715_0003
"""

from alembic import op
import sqlalchemy as sa

revision = "20260715_0004"
down_revision = "20260715_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_bases", sa.Column("data_classification", sa.String(32), nullable=False, server_default="internal"))
    op.add_column("knowledge_bases", sa.Column("external_llm_allowed", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("knowledge_bases", sa.Column("web_search_allowed", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("knowledge_bases", sa.Column("retention_policy_id", sa.String(64)))

    op.create_table("users",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=False), sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("token_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_users_email", "users", ["email"])
    op.create_table("tenant_memberships",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("role", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.UniqueConstraint("tenant_id", "user_id", name="uq_membership_tenant_user"))
    op.create_index("ix_tenant_memberships_tenant_id", "tenant_memberships", ["tenant_id"])
    op.create_index("ix_tenant_memberships_user_id", "tenant_memberships", ["user_id"])
    op.create_table("api_credentials",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(255), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False, unique=True), sa.Column("secret_hash", sa.String(64), nullable=False), sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("expires_at", sa.DateTime(timezone=True)), sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_api_credentials_prefix", "api_credentials", ["prefix"])
    op.create_index("ix_api_credentials_tenant_id", "api_credentials", ["tenant_id"])
    op.create_index("ix_api_credentials_user_id", "api_credentials", ["user_id"])
    op.create_table("refresh_credentials",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(64), nullable=False), sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True), sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False), sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_refresh_credentials_tenant_id", "refresh_credentials", ["tenant_id"])
    op.create_index("ix_refresh_credentials_user_id", "refresh_credentials", ["user_id"])
    op.create_table("knowledge_base_acl",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("kb_id", sa.String(64), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("principal_type", sa.String(16), nullable=False), sa.Column("principal_id", sa.String(64), nullable=False), sa.Column("role", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("kb_id", "principal_type", "principal_id", name="uq_kb_acl_principal"))
    op.create_index("ix_knowledge_base_acl_kb_id", "knowledge_base_acl", ["kb_id"])
    op.create_index("ix_knowledge_base_acl_principal_id", "knowledge_base_acl", ["principal_id"])
    op.create_table("query_sessions",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(64), nullable=False), sa.Column("user_id", sa.String(64), nullable=False), sa.Column("kb_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_query_sessions_tenant_id", "query_sessions", ["tenant_id"])
    op.create_index("ix_query_sessions_user_id", "query_sessions", ["user_id"])
    op.create_index("ix_query_sessions_kb_id", "query_sessions", ["kb_id"])
    op.create_table("audit_events",
        sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(64)), sa.Column("actor_id", sa.String(64)),
        sa.Column("action", sa.String(128), nullable=False), sa.Column("resource_type", sa.String(64)), sa.Column("resource_id", sa.String(128)),
        sa.Column("result", sa.String(16), nullable=False), sa.Column("request_id", sa.String(64)), sa.Column("ip_address", sa.String(64)), sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_audit_tenant_created", "audit_events", ["tenant_id", "created_at"])
    op.create_index("ix_audit_events_tenant_id", "audit_events", ["tenant_id"])
    op.create_index("ix_audit_events_actor_id", "audit_events", ["actor_id"])
    op.create_index("ix_audit_events_request_id", "audit_events", ["request_id"])


def downgrade() -> None:
    for table in ("audit_events", "query_sessions", "knowledge_base_acl", "refresh_credentials", "api_credentials", "tenant_memberships", "users"):
        op.drop_table(table)
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_column("retention_policy_id")
        batch.drop_column("web_search_allowed")
        batch.drop_column("external_llm_allowed")
        batch.drop_column("data_classification")
