"""TODO-07 parser pipeline fingerprints, parse artifacts and chunking runs.

Revision ID: 20260722_0007
Revises: 20260721_0006
"""

from alembic import op
import sqlalchemy as sa


revision = "20260722_0007"
down_revision = "20260721_0006"
branch_labels = None
depends_on = None


def _set_sqlite_foreign_keys(enabled: bool) -> bool:
    """SQLite batch 重建被引用表时临时关闭 FK；其他数据库无需处理。"""
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return False
    bind.exec_driver_sql(f"PRAGMA foreign_keys={'ON' if enabled else 'OFF'}")
    return True


def _check_sqlite_foreign_keys() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"foreign key violations after TODO-07 migration: {violations[:10]}")


def upgrade() -> None:
    # SQLite 无法原地删除唯一约束；Alembic batch 会安全重建 document_versions。
    # server_default 既完成旧行回填，也保护迁移期间的并发旧写入。
    sqlite_batch = _set_sqlite_foreign_keys(False)
    try:
        with op.batch_alter_table("document_versions") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "pipeline_fingerprint",
                    sa.String(64),
                    nullable=False,
                    server_default="legacy",
                )
            )
            batch_op.drop_constraint("uq_document_content_hash", type_="unique")
            batch_op.create_unique_constraint(
                "uq_document_content_pipeline",
                ["document_id", "content_sha256", "pipeline_fingerprint"],
            )
    finally:
        if sqlite_batch:
            _set_sqlite_foreign_keys(True)
            _check_sqlite_foreign_keys()

    op.create_table(
        "document_parse_artifacts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "document_version_id",
            sa.String(64),
            sa.ForeignKey("document_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("parser_id", sa.String(128), nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="succeeded"),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("artifact_sha256", sa.String(64)),
        sa.Column("artifact_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
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
            "document_version_id",
            "parser_id",
            "parser_version",
            "schema_version",
            name="uq_parse_artifact_version_parser_schema",
        ),
    )
    op.create_index(
        "ix_document_parse_artifacts_document_version_id",
        "document_parse_artifacts",
        ["document_version_id"],
    )
    op.create_index(
        "ix_parse_artifact_version_status",
        "document_parse_artifacts",
        ["document_version_id", "status"],
    )

    op.create_table(
        "document_chunking_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "parse_artifact_id",
            sa.String(64),
            sa.ForeignKey("document_parse_artifacts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("template_id", sa.String(128), nullable=False),
        sa.Column("template_version", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="succeeded"),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("artifact_sha256", sa.String(64)),
        sa.Column("artifact_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
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
            "parse_artifact_id",
            "template_id",
            "template_version",
            "config_hash",
            name="uq_chunking_run_artifact_template_config",
        ),
    )
    op.create_index(
        "ix_document_chunking_runs_parse_artifact_id",
        "document_chunking_runs",
        ["parse_artifact_id"],
    )
    op.create_index(
        "ix_chunking_run_artifact_status",
        "document_chunking_runs",
        ["parse_artifact_id", "status"],
    )


def downgrade() -> None:
    op.drop_table("document_chunking_runs")
    op.drop_table("document_parse_artifacts")

    # 若数据库已经存在“同文档、同内容、不同 pipeline”的版本，旧约束无法无损恢复，
    # 此处让数据库唯一约束创建失败并中止事务，而不是静默删除任一版本。
    sqlite_batch = _set_sqlite_foreign_keys(False)
    try:
        with op.batch_alter_table("document_versions") as batch_op:
            batch_op.drop_constraint("uq_document_content_pipeline", type_="unique")
            batch_op.create_unique_constraint(
                "uq_document_content_hash", ["document_id", "content_sha256"]
            )
            batch_op.drop_column("pipeline_fingerprint")
    finally:
        if sqlite_batch:
            _set_sqlite_foreign_keys(True)
            _check_sqlite_foreign_keys()
