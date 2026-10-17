"""Distinguish migration-era legacy indexes from managed vector indexes.

Revision ID: 20260725_0012
Revises: 20260725_0011
Create Date: 2026-07-25
"""

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Iterator

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_0012"
down_revision: str | None = "20260725_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_default(column: dict[str, object]) -> str:
    return str(column.get("default") or "").strip("()'\" ").lower()


@contextmanager
def _sqlite_batch_foreign_key_guard() -> Iterator[None]:
    """允许 SQLite 安全重建被其他表引用的 index_versions。

    SQLite 的 batch alter 通过“建临时表 → 复制 → DROP 原表 → rename”实现。
    ``index_versions`` 被 binding、placement 等表引用；若保持即时 FK 检查，
    DROP 会在有真实引用行时失败。这里使用可在事务内启用的
    ``defer_foreign_keys`` 把检查延后到迁移事务提交；不能在 migration 内
    ``commit``，否则 Alembic 随后的 revision 更新会被留在失效事务中。
    """

    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        yield
        return

    # Alembic 将 SQLite DDL 视为 non-transactional；downgrade 在进入这里前没有
    # DML，只有 SQLAlchemy 的逻辑 autobegin，SQLite 本身尚未开启事务。
    # 零行 UPDATE 显式开启实际事务，否则下一条 batch DDL 会让 deferred FK
    # 设置失效。upgrade 已有回填 UPDATE，但重复执行这条语句仍无副作用。
    bind.exec_driver_sql("UPDATE index_versions SET id = id WHERE 1 = 0")
    bind.exec_driver_sql("PRAGMA defer_foreign_keys=ON")
    if int(bind.exec_driver_sql("PRAGMA defer_foreign_keys").scalar_one()) != 1:
        raise RuntimeError("failed to defer SQLite foreign keys for batch migration")
    try:
        yield
    finally:
        violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        bind.exec_driver_sql("PRAGMA defer_foreign_keys=OFF")
        if violations:
            raise RuntimeError(
                f"SQLite foreign key violations after vector governance migration: {violations[:5]}"
            )


def upgrade() -> None:
    # 先以 legacy 默认值新增，保证升级前的每一行都有明确代际；随后把数据库默认值
    # 改为 managed，使升级后任何新索引若漏建 placement 都会失败关闭。列存在性
    # 检查允许从 SQLite 非事务 DDL 的“第一阶段已完成、version 尚未推进”状态恢复。
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"]: column for column in inspector.get_columns("index_versions")}
    if "vector_governance_mode" not in columns:
        with op.batch_alter_table("index_versions") as batch:
            batch.add_column(
                sa.Column(
                    "vector_governance_mode",
                    sa.String(length=16),
                    nullable=False,
                    server_default="legacy",
                )
            )
    op.execute(
        sa.text(
            "UPDATE index_versions "
            "SET vector_governance_mode = 'legacy' "
            "WHERE vector_governance_mode IS NULL OR vector_governance_mode = ''"
        )
    )
    inspector = sa.inspect(bind)
    columns = {column["name"]: column for column in inspector.get_columns("index_versions")}
    check_names = {
        constraint.get("name") for constraint in inspector.get_check_constraints("index_versions")
    }
    needs_managed_default = _column_default(columns["vector_governance_mode"]) != "managed"
    needs_check = "ck_index_vector_governance_mode" not in check_names
    if needs_managed_default or needs_check:
        with _sqlite_batch_foreign_key_guard():
            with op.batch_alter_table("index_versions") as batch:
                if needs_managed_default:
                    batch.alter_column(
                        "vector_governance_mode",
                        existing_type=sa.String(length=16),
                        nullable=False,
                        server_default="managed",
                    )
                if needs_check:
                    batch.create_check_constraint(
                        "ck_index_vector_governance_mode",
                        "vector_governance_mode IN ('legacy', 'managed')",
                    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("index_versions")}
    if "vector_governance_mode" not in columns:
        return
    check_names = {
        constraint.get("name") for constraint in inspector.get_check_constraints("index_versions")
    }
    with _sqlite_batch_foreign_key_guard():
        with op.batch_alter_table("index_versions") as batch:
            if "ck_index_vector_governance_mode" in check_names:
                batch.drop_constraint("ck_index_vector_governance_mode", type_="check")
            batch.drop_column("vector_governance_mode")
