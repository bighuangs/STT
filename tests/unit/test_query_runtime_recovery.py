from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import app.application.query_runtime_maintenance as maintenance
from app.application.query_runtime_maintenance import (
    recover_expired_query_runs,
    run_query_runtime_recovery_loop,
)
from app.application.query_runtime_service import QueryRuntimeService
from app.domain.query_runtime import (
    QueryEventType,
    QueryRunStatus,
    TERMINAL_EVENT_TYPES,
)
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.models import Base
from app.infra.persistence.sql.query_runtime_repository import (
    QueryRuntimeRepository,
)


def test_restart_recovery_closes_expired_runs_once(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'recovery.db').as_posix()}"
    engine = configure_database(database_url)
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)

    with session_scope() as session:
        repository = QueryRuntimeRepository(
            session,
            tenant_id="tenant_recovery",
            user_id="user_recovery",
        )
        service = QueryRuntimeService(repository)
        expired = service.accept(
            kb_id="kb_recovery",
            client_turn_id="expired",
            user_content="过期查询",
            deadline_at=now - timedelta(seconds=1),
        )
        expired_run = repository.transition_run(
            expired.run.query_run_id,
            expected_revision=expired.run.revision,
            target=QueryRunStatus.PLANNING,
        )
        expired_run = repository.transition_run(
            expired_run.query_run_id,
            expected_revision=expired_run.revision,
            target=QueryRunStatus.RETRIEVING,
        )
        cancelled = service.accept(
            kb_id="kb_recovery",
            client_turn_id="cancelled",
            user_content="等待取消",
            conversation_id=expired.conversation.conversation_id,
            deadline_at=now - timedelta(seconds=1),
        )
        repository.request_cancel(
            cancelled.run.query_run_id,
            expected_revision=cancelled.run.revision,
        )
        future = service.accept(
            kb_id="kb_recovery",
            client_turn_id="future",
            user_content="仍在有效期",
            conversation_id=expired.conversation.conversation_id,
            deadline_at=now + timedelta(minutes=5),
        )

    report = recover_expired_query_runs(now=now)
    assert report == {
        "scanned": 2,
        "timed_out": 1,
        "cancelled": 1,
        "raced": 0,
        "failed": 0,
    }

    with session_scope() as session:
        repository = QueryRuntimeRepository(
            session,
            tenant_id="tenant_recovery",
            user_id="user_recovery",
        )
        assert repository.get_run(expired.run.query_run_id).status == QueryRunStatus.TIMED_OUT
        assert repository.get_run(cancelled.run.query_run_id).status == QueryRunStatus.CANCELLED
        assert repository.get_run(future.run.query_run_id).status == QueryRunStatus.ACCEPTED
        for run_id, expected_type in (
            (expired.run.query_run_id, QueryEventType.RUN_TIMED_OUT),
            (cancelled.run.query_run_id, QueryEventType.RUN_CANCELLED),
        ):
            page = repository.read_event_page(run_id, limit=100)
            terminal = [event.type for event in page.events if event.type in TERMINAL_EVENT_TYPES]
            assert terminal == [expected_type]

    assert recover_expired_query_runs(now=now)["scanned"] == 0
    dispose_database()


def test_recovery_loop_keeps_scanning_until_shutdown(monkeypatch) -> None:
    calls = 0

    def fake_recover():
        nonlocal calls
        calls += 1
        return {
            "scanned": 0,
            "timed_out": 0,
            "cancelled": 0,
            "raced": 0,
            "failed": 0,
        }

    monkeypatch.setattr(maintenance, "recover_expired_query_runs", fake_recover)

    async def exercise() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run_query_runtime_recovery_loop(stop, interval_seconds=0.01))
        while calls < 2:
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise())
    assert calls >= 2
