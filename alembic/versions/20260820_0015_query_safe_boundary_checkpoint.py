"""Durable encrypted safe-boundary checkpoints for reliable Query runs.

Revision ID: 20260820_0015
Revises: 20260809_0014
Create Date: 2026-08-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision = "20260820_0015"
down_revision = "20260809_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    checkpoint_blob = sa.LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")
    op.create_table(
        "query_run_executions",
        sa.Column("query_run_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("auth_type", sa.String(length=32), nullable=False),
        sa.Column("credential_id", sa.String(length=128), nullable=False),
        sa.Column("identity_version", sa.Integer(), nullable=True),
        sa.Column("roles_digest", sa.String(length=71), nullable=False),
        sa.Column("scopes_digest", sa.String(length=71), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_namespace", sa.String(length=64), nullable=False),
        sa.Column("workflow_version", sa.String(length=64), nullable=False),
        sa.Column("state_schema_version", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_schema_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ready"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_token_hash", sa.String(length=71), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_node", sa.String(length=64), nullable=True),
        sa.Column("checkpoint_sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("recovery_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("safe_error_code", sa.String(length=128), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("revision >= 0", name="ck_query_execution_revision"),
        sa.CheckConstraint("fencing_token >= 0", name="ck_query_execution_fence"),
        sa.CheckConstraint(
            "checkpoint_sequence >= 0",
            name="ck_query_execution_checkpoint_seq",
        ),
        sa.CheckConstraint(
            "attempt_sequence >= 0",
            name="ck_query_execution_attempt_seq",
        ),
        sa.CheckConstraint(
            "recovery_count >= 0",
            name="ck_query_execution_recovery_count",
        ),
        sa.CheckConstraint(
            "status IN ('ready', 'running', 'completed', 'interrupted', "
            "'failed', 'cancelled', 'timed_out')",
            name="ck_query_execution_status",
        ),
        sa.ForeignKeyConstraint(
            ["query_run_id"],
            ["query_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("query_run_id"),
        sa.UniqueConstraint(
            "thread_id",
            "checkpoint_namespace",
            name="uq_query_execution_thread_namespace",
        ),
    )
    op.create_index(
        "ix_query_execution_recovery",
        "query_run_executions",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "ix_query_execution_owner",
        "query_run_executions",
        ["tenant_id", "user_id", "status"],
    )
    for column in ("tenant_id", "user_id", "lease_expires_at"):
        op.create_index(
            f"ix_query_run_executions_{column}",
            "query_run_executions",
            [column],
        )

    op.create_table(
        "query_run_checkpoints",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("query_run_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("checkpoint_namespace", sa.String(length=64), nullable=False),
        sa.Column("workflow_version", sa.String(length=64), nullable=False),
        sa.Column("state_schema_version", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_schema_version", sa.String(length=64), nullable=False),
        sa.Column("completed_node", sa.String(length=64), nullable=True),
        sa.Column("next_node", sa.String(length=64), nullable=True),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("ciphertext", checkpoint_blob, nullable=False),
        sa.Column("ciphertext_sha256", sa.String(length=71), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("sequence >= 0", name="ck_query_checkpoint_sequence"),
        sa.CheckConstraint("size_bytes >= 16", name="ck_query_checkpoint_size"),
        sa.ForeignKeyConstraint(
            ["query_run_id"],
            ["query_run_executions.query_run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "query_run_id",
            "sequence",
            name="uq_query_checkpoint_run_sequence",
        ),
    )
    op.create_index(
        "ix_query_checkpoint_latest",
        "query_run_checkpoints",
        ["query_run_id", "sequence"],
    )
    op.create_index(
        "ix_query_run_checkpoints_tenant_id",
        "query_run_checkpoints",
        ["tenant_id"],
    )
    op.create_index(
        "ix_query_run_checkpoints_query_run_id",
        "query_run_checkpoints",
        ["query_run_id"],
    )

    op.create_table(
        "query_node_attempts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("query_run_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("node_name", sa.String(length=64), nullable=False),
        sa.Column("effect_class", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="scheduled",
        ),
        sa.Column("checkpoint_sequence_before", sa.Integer(), nullable=False),
        sa.Column("checkpoint_sequence_after", sa.Integer(), nullable=True),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("lease_owner", sa.String(length=128), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("safe_error_code", sa.String(length=128), nullable=True),
        sa.Column(
            "scheduled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checkpointed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("sequence >= 1", name="ck_query_node_attempt_sequence"),
        sa.CheckConstraint("revision >= 0", name="ck_query_node_attempt_revision"),
        sa.CheckConstraint(
            "checkpoint_sequence_before >= 0",
            name="ck_query_node_attempt_checkpoint_before",
        ),
        sa.CheckConstraint(
            "checkpoint_sequence_after IS NULL OR "
            "checkpoint_sequence_after > checkpoint_sequence_before",
            name="ck_query_node_attempt_checkpoint_after",
        ),
        sa.CheckConstraint("fencing_token >= 1", name="ck_query_node_attempt_fence"),
        sa.CheckConstraint(
            "status IN ('scheduled', 'started', 'checkpointed', 'interrupted', 'failed')",
            name="ck_query_node_attempt_status",
        ),
        sa.ForeignKeyConstraint(
            ["query_run_id"],
            ["query_run_executions.query_run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "query_run_id",
            "sequence",
            name="uq_query_node_attempt_sequence",
        ),
    )
    op.create_index(
        "ix_query_node_attempt_active",
        "query_node_attempts",
        ["query_run_id", "status", "sequence"],
    )
    op.create_index(
        "ix_query_node_attempts_tenant_id",
        "query_node_attempts",
        ["tenant_id"],
    )
    op.create_index(
        "ix_query_node_attempts_query_run_id",
        "query_node_attempts",
        ["query_run_id"],
    )
    with op.batch_alter_table("query_events") as batch:
        batch.add_column(sa.Column("operation_id", sa.String(length=128), nullable=True))
        batch.create_unique_constraint(
            "uq_query_event_run_operation",
            ["query_run_id", "operation_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("query_events") as batch:
        batch.drop_constraint("uq_query_event_run_operation", type_="unique")
        batch.drop_column("operation_id")
    op.drop_index(
        "ix_query_node_attempts_query_run_id",
        table_name="query_node_attempts",
    )
    op.drop_index("ix_query_node_attempts_tenant_id", table_name="query_node_attempts")
    op.drop_index("ix_query_node_attempt_active", table_name="query_node_attempts")
    op.drop_table("query_node_attempts")
    op.drop_index(
        "ix_query_run_checkpoints_query_run_id",
        table_name="query_run_checkpoints",
    )
    op.drop_index(
        "ix_query_run_checkpoints_tenant_id",
        table_name="query_run_checkpoints",
    )
    op.drop_index("ix_query_checkpoint_latest", table_name="query_run_checkpoints")
    op.drop_table("query_run_checkpoints")
    for column in ("lease_expires_at", "user_id", "tenant_id"):
        op.drop_index(
            f"ix_query_run_executions_{column}",
            table_name="query_run_executions",
        )
    op.drop_index("ix_query_execution_owner", table_name="query_run_executions")
    op.drop_index("ix_query_execution_recovery", table_name="query_run_executions")
    op.drop_table("query_run_executions")
