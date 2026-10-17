"""TODO-02 durable ingestion queue, leases, events and checkpoints.

Revision ID: 20260715_0002
Revises: 20260715_0001
Create Date: 2026-07-15
"""

from alembic import op
import sqlalchemy as sa


revision = "20260715_0002"
down_revision = "20260715_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite 的 batch_alter_table 会创建 _alembic_tmp_ingestion_jobs。
    # 若进程在“创建/复制临时表”后、替换原表前被强制终止，原表和临时表会同时
    # 存在，后续启动将因临时表重名而永远无法迁移。仅当原表仍存在时删除残留
    # 临时表是安全的：事实数据仍以 ingestion_jobs 原表为准，随后完整重跑 batch。
    connection = op.get_bind()
    def table_names() -> set[str]:
        return set(sa.inspect(connection).get_table_names())

    def index_names(table_name: str) -> set[str]:
        return {
            row["name"] for row in sa.inspect(connection).get_indexes(table_name)
        }

    if connection.dialect.name == "sqlite":
        tables = table_names()
        temporary = "_alembic_tmp_ingestion_jobs"
        if temporary in tables and "ingestion_jobs" in tables:
            op.drop_table(temporary)
        elif temporary in tables and "ingestion_jobs" not in tables:
            # batch 已删除原表但尚未来得及 rename；临时表就是完整的新表。
            op.rename_table(temporary, "ingestion_jobs")

    job_columns = {
        row["name"] for row in sa.inspect(connection).get_columns("ingestion_jobs")
    }
    if "run_attempt" not in job_columns:
        with op.batch_alter_table("ingestion_jobs") as batch:
            batch.add_column(sa.Column("run_attempt", sa.Integer(), server_default="0", nullable=False))
            batch.add_column(sa.Column("max_attempts", sa.Integer(), server_default="5", nullable=False))
            batch.add_column(sa.Column("priority", sa.Integer(), server_default="0", nullable=False))
            batch.add_column(sa.Column("idempotency_key", sa.String(255)))
            batch.add_column(sa.Column("request_fingerprint", sa.String(64)))
            batch.add_column(sa.Column("lease_owner", sa.String(128)))
            batch.add_column(sa.Column("lease_token", sa.String(64)))
            batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
            batch.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True)))
            batch.add_column(sa.Column("next_retry_at", sa.DateTime(timezone=True)))
            batch.add_column(sa.Column("cancel_requested_at", sa.DateTime(timezone=True)))
            batch.add_column(sa.Column("error_category", sa.String(32)))
            batch.add_column(sa.Column("progress_detail", sa.JSON(), server_default="{}", nullable=False))
            batch.add_column(sa.Column("checkpoint_stage", sa.String(128)))
            batch.add_column(sa.Column("checkpoint_sequence", sa.Integer(), server_default="0", nullable=False))
            batch.add_column(sa.Column("last_event_sequence", sa.Integer(), server_default="0", nullable=False))
            batch.add_column(sa.Column("recovered_count", sa.Integer(), server_default="0", nullable=False))
            batch.create_unique_constraint("uq_ingestion_job_idempotency_key", ["idempotency_key"])
            batch.create_index("ix_ingestion_job_ready", ["status", "next_retry_at", "priority", "created_at"])
            batch.create_index("ix_ingestion_job_lease", ["lease_expires_at"])

    # 下面各段都检查现状，使 SQLite 非事务 DDL 在任意 create-table 边界中断后可续跑。
    if "ix_ingestion_job_ready" not in index_names("ingestion_jobs"):
        op.create_index("ix_ingestion_job_ready", "ingestion_jobs", ["status", "next_retry_at", "priority", "created_at"])
    if "ix_ingestion_job_lease" not in index_names("ingestion_jobs"):
        op.create_index("ix_ingestion_job_lease", "ingestion_jobs", ["lease_expires_at"])

    if "ingestion_job_events" not in table_names():
        op.create_table(
            "ingestion_job_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("job_id", sa.String(64), sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(64), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("stage", sa.String(128), nullable=False),
            sa.Column("progress", sa.Float(), nullable=False),
            sa.Column("payload", sa.JSON(), server_default="{}", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("job_id", "sequence", name="uq_ingestion_event_job_sequence"),
        )
    if "ix_ingestion_event_job_created" not in index_names("ingestion_job_events"):
        op.create_index("ix_ingestion_event_job_created", "ingestion_job_events", ["job_id", "created_at"])

    if "ingestion_idempotency_keys" not in table_names():
        op.create_table(
            "ingestion_idempotency_keys",
            sa.Column("key", sa.String(255), primary_key=True),
            sa.Column("request_fingerprint", sa.String(64), nullable=False),
            sa.Column("job_id", sa.String(64), sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    if "ix_ingestion_idempotency_keys_job_id" not in index_names("ingestion_idempotency_keys"):
        op.create_index("ix_ingestion_idempotency_keys_job_id", "ingestion_idempotency_keys", ["job_id"])

    if "ingestion_job_checkpoints" not in table_names():
        op.create_table(
            "ingestion_job_checkpoints",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("job_id", sa.String(64), sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("stage", sa.String(128), nullable=False),
            sa.Column("resume_node", sa.String(128), nullable=False),
            sa.Column("artifact_uri", sa.Text(), nullable=False),
            sa.Column("artifact_sha256", sa.String(64), nullable=False),
            sa.Column("artifact_size", sa.Integer(), nullable=False),
            sa.Column("input_sha256", sa.String(64), nullable=False),
            sa.Column("config_sha256", sa.String(64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("job_id", "sequence", name="uq_ingestion_checkpoint_job_sequence"),
        )
    if "ix_ingestion_checkpoint_job_created" not in index_names("ingestion_job_checkpoints"):
        op.create_index("ix_ingestion_checkpoint_job_created", "ingestion_job_checkpoints", ["job_id", "created_at"])

    if "ingestion_workers" not in table_names():
        op.create_table(
            "ingestion_workers",
            sa.Column("id", sa.String(128), primary_key=True),
            sa.Column("status", sa.String(32), server_default="idle", nullable=False),
            sa.Column("current_job_id", sa.String(64)),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("metadata_json", sa.JSON(), server_default="{}", nullable=False),
        )

    # 既有 TODO-01 任务补一条初始事件，事件流从数据库现状连续开始。
    jobs = connection.execute(sa.text(
        "SELECT id AS job_id, status, stage, progress "
        "FROM ingestion_jobs ORDER BY created_at"
    )).mappings()
    for job in jobs:
        latest_sequence = connection.execute(
            sa.text(
                "SELECT MAX(sequence) FROM ingestion_job_events WHERE job_id = :job_id"
            ),
            {"job_id": job["job_id"]},
        ).scalar()
        if latest_sequence is None:
            connection.execute(sa.text(
                "INSERT INTO ingestion_job_events "
                "(job_id, sequence, event_type, status, stage, progress, payload) "
                "VALUES (:job_id, 1, 'migrated', :status, :stage, :progress, '{}')"
            ), dict(job))
            latest_sequence = 1
        connection.execute(sa.text(
            "UPDATE ingestion_jobs SET last_event_sequence = :sequence "
            "WHERE id = :job_id AND last_event_sequence < :sequence"
        ), {"job_id": job["job_id"], "sequence": latest_sequence})


def downgrade() -> None:
    op.drop_table("ingestion_workers")
    op.drop_index("ix_ingestion_checkpoint_job_created", table_name="ingestion_job_checkpoints")
    op.drop_table("ingestion_job_checkpoints")
    op.drop_index("ix_ingestion_idempotency_keys_job_id", table_name="ingestion_idempotency_keys")
    op.drop_table("ingestion_idempotency_keys")
    op.drop_index("ix_ingestion_event_job_created", table_name="ingestion_job_events")
    op.drop_table("ingestion_job_events")
    with op.batch_alter_table("ingestion_jobs") as batch:
        batch.drop_index("ix_ingestion_job_lease")
        batch.drop_index("ix_ingestion_job_ready")
        batch.drop_constraint("uq_ingestion_job_idempotency_key", type_="unique")
        for name in (
            "recovered_count", "last_event_sequence", "checkpoint_sequence", "checkpoint_stage",
            "progress_detail", "error_category", "cancel_requested_at", "next_retry_at",
            "heartbeat_at", "lease_expires_at", "lease_token", "lease_owner",
            "request_fingerprint", "idempotency_key", "priority", "max_attempts", "run_attempt",
        ):
            batch.drop_column(name)
