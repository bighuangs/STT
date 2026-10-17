from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import pytest

from app.infra.persistence.sql import database as database_module


@pytest.fixture(autouse=True)
def isolate_process_database_binding():
    """这些测试会重绑全局引擎，结束后恢复原 URL 事实。"""

    original_url = database_module._database_url
    database_module.dispose_database()
    try:
        yield
    finally:
        database_module.dispose_database()
        database_module._database_url = original_url


def test_file_sqlite_enables_connect_wait_busy_timeout_and_wal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    real_create_engine = database_module.create_engine

    def create_engine_spy(database_url, **kwargs):
        captured.update(kwargs)
        return real_create_engine(database_url, **kwargs)

    monkeypatch.setattr(database_module, "create_engine", create_engine_spy)
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'concurrency.db').as_posix()}"
    engine = database_module.configure_database(database_url)

    assert captured["connect_args"] == {
        "check_same_thread": False,
        "timeout": database_module.SQLITE_CONNECT_TIMEOUT_SECONDS,
    }
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert (
            connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
            == database_module.SQLITE_BUSY_TIMEOUT_MS
        )
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one().lower() == "wal"


def test_sqlite_busy_timeout_waits_for_contending_writer(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'writer-contention.db').as_posix()}"
    engine = database_module.configure_database(database_url)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE concurrency_probe (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )

    # 两条物理连接先建立，确保测试测量的是写锁等待，而不是建连或 WAL 初始化。
    holder = engine.raw_connection()
    contender = engine.raw_connection()
    holder_cursor = holder.cursor()
    attempting_write = Event()
    try:
        holder_cursor.execute("BEGIN IMMEDIATE")
        holder_cursor.execute(
            "INSERT INTO concurrency_probe (id, value) VALUES (?, ?)",
            (1, "holder"),
        )

        def write_after_lock() -> float:
            cursor = contender.cursor()
            started = monotonic()
            attempting_write.set()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(
                    "INSERT INTO concurrency_probe (id, value) VALUES (?, ?)",
                    (2, "contender"),
                )
                contender.commit()
                return monotonic() - started
            finally:
                cursor.close()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(write_after_lock)
            assert attempting_write.wait(timeout=1)
            sleep(0.15)
            holder.commit()
            waited_seconds = future.result(timeout=3)

        assert waited_seconds >= 0.10
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql("SELECT COUNT(*) FROM concurrency_probe").scalar_one()
                == 2
            )
    finally:
        holder_cursor.close()
        holder.close()
        contender.close()


def test_non_sqlite_engine_does_not_receive_sqlite_connection_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_engine = Mock()
    create_engine_mock = Mock(return_value=fake_engine)
    listen_mock = Mock()
    monkeypatch.setattr(database_module, "create_engine", create_engine_mock)
    monkeypatch.setattr(database_module.event, "listen", listen_mock)

    configured = database_module.configure_database(
        "postgresql+psycopg://stt:secret@database.example/stt"
    )

    assert configured is fake_engine
    assert create_engine_mock.call_args.kwargs["connect_args"] == {}
    listen_mock.assert_not_called()

