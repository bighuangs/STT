"""Query safe-boundary schema 的 Alembic 与跨方言类型合同。"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable

from app.infra.persistence.sql.query_runtime_models import QueryRunCheckpointModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def test_query_checkpoint_migration_up_down_and_back_to_head(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'checkpoint-migration.db').as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "head")

    from sqlalchemy import create_engine

    engine = create_engine(database_url, future=True)
    inspector = inspect(engine)
    expected_tables = {
        "query_run_executions",
        "query_run_checkpoints",
        "query_node_attempts",
    }
    assert expected_tables.issubset(inspector.get_table_names())
    checkpoint_columns = {
        column["name"]: column for column in inspector.get_columns("query_run_checkpoints")
    }
    assert {"nonce", "ciphertext", "ciphertext_sha256"}.issubset(checkpoint_columns)
    assert "state_json" not in checkpoint_columns
    assert "payload_json" not in checkpoint_columns
    engine.dispose()

    command.downgrade(config, "20260809_0014")
    engine = create_engine(database_url, future=True)
    assert expected_tables.isdisjoint(inspect(engine).get_table_names())
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url, future=True)
    assert expected_tables.issubset(inspect(engine).get_table_names())
    engine.dispose()


def test_checkpoint_ciphertext_uses_binary_types_on_postgres_and_mysql() -> None:
    table = QueryRunCheckpointModel.__table__
    postgres_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    mysql_sql = str(CreateTable(table).compile(dialect=mysql.dialect()))
    assert "ciphertext BYTEA" in postgres_sql
    assert "ciphertext LONGBLOB" in mysql_sql
    assert "state_json" not in postgres_sql
    assert "state_json" not in mysql_sql
