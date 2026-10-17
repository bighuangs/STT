"""TODO-24/25/31 查询身份、会话提交、CAS、续传和背压合同测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session

from app.application.query_runtime_service import (
    QueryCancellationRequested,
    QueryRuntimeService,
)
from app.domain._state_machine import InvalidStateTransition
from app.domain.query_runtime import (
    EventPage,
    QueryEventType,
    QueryRunStatus,
    TurnStatus,
)
from app.infra.persistence.sql.models import Base
from app.infra.persistence.sql.query_runtime_models import (
    QueryConversationModel,
    QueryEventModel,
    QueryMessageModel,
    QueryRunModel,
    QueryTurnModel,
)
from app.infra.persistence.sql.query_runtime_repository import (
    EventCursorExpired,
    EventCursorInvalid,
    QueryRuntimeRepository,
    StaleQueryRevision,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.shared.utils.query_event_stream import (
    DurableStreamPolicy,
    durable_sse_generator,
)


RUNTIME_TABLES = [
    QueryConversationModel.__table__,
    QueryTurnModel.__table__,
    QueryRunModel.__table__,
    QueryMessageModel.__table__,
    QueryEventModel.__table__,
]


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    result = create_engine(f"sqlite:///{(tmp_path / 'runtime.db').as_posix()}", future=True)
    event.listen(
        result,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(result, tables=RUNTIME_TABLES)
    yield result
    result.dispose()


def _repository(session: Session, *, tenant: str = "tenant_a", user: str = "user_a"):
    return QueryRuntimeRepository(session, tenant_id=tenant, user_id=user)


def _advance_to_validating(
    repository: QueryRuntimeRepository,
    query_run_id: str,
) -> object:
    run = repository.get_run(query_run_id)
    for stage in (
        QueryRunStatus.PLANNING,
        QueryRunStatus.RETRIEVING,
        QueryRunStatus.PACKING,
        QueryRunStatus.GENERATING,
        QueryRunStatus.VALIDATING,
    ):
        run = repository.transition_run(
            query_run_id,
            expected_revision=run.revision,
            target=stage,
        )
    return run


def test_turn_identity_is_idempotent_but_payload_collision_is_rejected(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        first_turn, first_run, first_history, created = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="browser-tab-1-turn-1",
            user_content="在使用 HAK180 之前需要做什么？",
            idempotency_key="http-request-1",
            index_snapshot={"index_version_ids": ["idx_hak180_v1"]},
        )
        second_turn, second_run, second_history, created_again = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="browser-tab-1-turn-1",
            user_content="在使用 HAK180 之前需要做什么？",
            idempotency_key="http-request-1",
            index_snapshot={"index_version_ids": ["idx_hak180_v1"]},
        )
        assert created is True
        assert created_again is False
        assert second_turn.turn_id == first_turn.turn_id
        assert second_run.query_run_id == first_run.query_run_id
        assert second_history.content_hash == first_history.content_hash
        assert first_run.query_run_id != conversation.conversation_id
        with pytest.raises(ResourceConflictError, match="another request payload"):
            repository.begin_turn(
                conversation_id=conversation.conversation_id,
                client_turn_id="browser-tab-1-turn-1",
                user_content="换成另一条问题",
                idempotency_key="http-request-1",
                index_snapshot={"index_version_ids": ["idx_hak180_v1"]},
            )


def test_same_conversation_ten_out_of_order_runs_commit_in_turn_order(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        accepted = [
            repository.begin_turn(
                conversation_id=conversation.conversation_id,
                client_turn_id=f"client-turn-{number}",
                user_content=f"问题 {number}",
            )
            for number in range(1, 11)
        ]
        prepared = [_advance_to_validating(repository, item[1].query_run_id) for item in accepted]
        for number, run in reversed(list(enumerate(prepared, start=1))):
            repository.complete_run(
                run.query_run_id,
                expected_revision=run.revision,
                answer=f"答案 {number}",
            )
        committed = repository.get_conversation(conversation.conversation_id)
        assert committed.resolved_turn_seq == 10
        assert committed.history_version == 10
        messages = repository.list_messages(conversation.conversation_id)
        assert [message.content for message in messages] == [
            value for number in range(1, 11) for value in (f"问题 {number}", f"答案 {number}")
        ]
        assert len({message.message_id for message in messages}) == 20


def test_rewrite_and_answer_receive_the_same_frozen_history_hash(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        service = QueryRuntimeService(repository)
        first = service.accept(
            kb_id="kb_hak180",
            client_turn_id="turn-1",
            user_content="HAK180 是什么？",
        )
        run = _advance_to_validating(repository, first.run.query_run_id)
        service.complete(
            run.query_run_id,
            expected_revision=run.revision,
            answer="HAK180 是手持产品。",
        )
        second = service.accept(
            kb_id="kb_hak180",
            conversation_id=first.conversation.conversation_id,
            client_turn_id="turn-2",
            user_content="使用它之前呢？",
        )
        identity = second.graph_identity()
        assert identity["history_snapshot_loaded"] is True
        assert identity["history_snapshot_hash"] == second.run.history_snapshot_hash
        assert [item["text"] for item in identity["history"]] == [
            "HAK180 是什么？",
            "HAK180 是手持产品。",
        ]
        # 图节点只消费这一份 state/history，不需要再读 Mongo。
        assert second.history.content_hash == identity["history_snapshot_hash"]


def test_failed_retry_reuses_turn_and_commits_user_message_exactly_once(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        turn, first_run, original_snapshot, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="retry-turn",
            user_content="启动前有哪些检查？",
        )
        first_run = repository.transition_run(
            first_run.query_run_id,
            expected_revision=first_run.revision,
            target=QueryRunStatus.PLANNING,
        )
        first_run = repository.fail_run(
            first_run.query_run_id,
            expected_revision=first_run.revision,
            error_code="MODEL_TIMEOUT",
            safe_message="模型暂时不可用",
            retryable=True,
            error_detail_ref="audit://errors/internal-only",
        )
        assert repository.list_messages(conversation.conversation_id) == []
        retry, retry_snapshot, created = repository.retry_run(
            first_run.query_run_id,
            idempotency_key="retry-click-1",
        )
        replay, replay_snapshot, replay_created = repository.retry_run(
            first_run.query_run_id,
            idempotency_key="retry-click-1",
        )
        assert created is True
        assert replay_created is False
        assert retry.query_run_id == replay.query_run_id
        assert retry.turn_id == turn.turn_id
        assert retry.attempt == 2
        assert retry_snapshot.content_hash == original_snapshot.content_hash
        assert replay_snapshot.content_hash == retry_snapshot.content_hash
        retry = _advance_to_validating(repository, retry.query_run_id)
        repository.complete_run(
            retry.query_run_id,
            expected_revision=retry.revision,
            answer="启动前应检查电源和工作环境。",
        )
        messages = repository.list_messages(conversation.conversation_id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert [message.content for message in messages] == [
            "启动前有哪些检查？",
            "启动前应检查电源和工作环境。",
        ]


def test_cancel_wins_after_its_cas_and_never_commits_a_partial_answer(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        turn, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="cancel-turn",
            user_content="生成一个较长答案",
        )
        run = _advance_to_validating(repository, run.query_run_id)
        old_revision = run.revision
        requested, accepted = repository.request_cancel(
            run.query_run_id,
            expected_revision=old_revision,
        )
        assert accepted is True
        assert requested.status == QueryRunStatus.CANCEL_REQUESTED
        token = QueryRuntimeService(repository).cancellation_token(run.query_run_id)
        with pytest.raises(QueryCancellationRequested):
            token.raise_if_cancelled()
        with pytest.raises(InvalidStateTransition):
            repository.complete_run(
                run.query_run_id,
                expected_revision=old_revision,
                answer="这个迟到答案必须丢弃",
            )
        cancelled = repository.finalize_cancel(
            run.query_run_id,
            expected_revision=requested.revision,
        )
        assert cancelled.status == QueryRunStatus.CANCELLED
        assert repository.get_turn(turn.turn_id).status == TurnStatus.CANCELLED
        assert repository.list_messages(conversation.conversation_id) == []
        events = repository.read_event_page(run.query_run_id).events
        terminal = [
            item
            for item in events
            if item.type
            in {
                QueryEventType.RUN_COMPLETED,
                QueryEventType.RUN_FAILED,
                QueryEventType.RUN_CANCELLED,
                QueryEventType.RUN_TIMED_OUT,
            }
        ]
        assert [item.type for item in terminal] == [QueryEventType.RUN_CANCELLED]


def test_stale_worker_cannot_overwrite_a_newer_run_revision(engine: Engine) -> None:
    with Session(engine) as setup:
        repository = _repository(setup)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="cas-turn",
            user_content="并发请求",
        )
        setup.commit()
        run_id = run.query_run_id

    with Session(engine) as worker_a, Session(engine) as worker_b:
        repo_a = _repository(worker_a)
        repo_b = _repository(worker_b)
        snapshot_a = repo_a.get_run(run_id)
        snapshot_b = repo_b.get_run(run_id)
        updated = repo_a.transition_run(
            run_id,
            expected_revision=snapshot_a.revision,
            target=QueryRunStatus.PLANNING,
        )
        worker_a.commit()
        assert updated.revision == snapshot_a.revision + 1
        with pytest.raises(StaleQueryRevision):
            repo_b.transition_run(
                run_id,
                expected_revision=snapshot_b.revision,
                target=QueryRunStatus.PLANNING,
            )
        worker_b.rollback()


def test_event_sequence_resume_scope_and_retention_are_explicit(engine: Engine) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, first, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="event-turn-1",
            user_content="问题一",
        )
        first = repository.transition_run(
            first.query_run_id,
            expected_revision=first.revision,
            target=QueryRunStatus.PLANNING,
        )
        repository.append_event(
            first.query_run_id,
            event_type=QueryEventType.ANSWER_DELTA,
            data={"operation": "append", "text": "增量"},
            persistence_class="compactable",
        )
        _, second, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="event-turn-2",
            user_content="问题二",
        )
        page = repository.read_event_page(first.query_run_id, limit=2)
        assert [item.seq for item in page.events] == [1, 2]
        resumed = repository.read_event_page(
            first.query_run_id,
            last_event_id=page.events[-1].event_id,
        )
        assert [item.seq for item in resumed.events] == [3]
        with pytest.raises(EventCursorInvalid):
            repository.read_event_page(
                second.query_run_id,
                last_event_id=page.events[-1].event_id,
            )
        expired = repository.append_event(
            first.query_run_id,
            event_type=QueryEventType.USAGE_UPDATE,
            data={"status": "estimated", "total_tokens": 5},
            occurred_at=datetime.now(UTC) - timedelta(hours=2),
            retention=timedelta(seconds=1),
        )
        with pytest.raises(EventCursorExpired):
            repository.read_event_page(
                first.query_run_id,
                last_event_id=expired.event_id,
            )
        with pytest.raises(ValueError, match="append or replace"):
            repository.append_event(
                first.query_run_id,
                event_type=QueryEventType.ANSWER_DELTA,
                data={"text": "缺少 operation"},
            )
        sequences = session.scalars(
            select(QueryEventModel.seq)
            .where(QueryEventModel.query_run_id == first.query_run_id)
            .order_by(QueryEventModel.seq)
        ).all()
        assert sequences == list(range(1, len(sequences) + 1))


def test_event_store_is_tenant_and_user_scoped(engine: Engine) -> None:
    with Session(engine) as owner:
        repository = _repository(owner)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="private-turn",
            user_content="私有问题",
        )
        owner.commit()
        run_id = run.query_run_id
        conversation_id = conversation.conversation_id
    with Session(engine) as stranger:
        for tenant, user in (("tenant_b", "user_a"), ("tenant_a", "user_b")):
            repository = _repository(stranger, tenant=tenant, user=user)
            with pytest.raises(ResourceNotFoundError):
                repository.get_conversation(conversation_id)
            with pytest.raises(ResourceNotFoundError):
                repository.get_run(run_id)
            with pytest.raises(ResourceNotFoundError):
                repository.read_event_page(run_id)


def test_durable_sse_resumes_after_event_id_and_closes_on_unique_terminal(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="sse-turn",
            user_content="SSE 问题",
        )
        run = _advance_to_validating(repository, run.query_run_id)
        run = repository.complete_run(
            run.query_run_id,
            expected_revision=run.revision,
            answer="SSE 答案",
        )
        all_events = repository.read_event_page(run.query_run_id).events
        cursor = all_events[1].event_id

        def reader(query_run_id: str, after_seq: int, event_id: str | None, limit: int):
            return repository.read_event_page(
                query_run_id,
                after_seq=after_seq,
                last_event_id=event_id,
                limit=limit,
            )

        async def collect() -> list[str]:
            return [
                frame
                async for frame in durable_sse_generator(
                    query_run_id=run.query_run_id,
                    read_page=reader,
                    last_event_id=cursor,
                    policy=DurableStreamPolicy(
                        batch_size=100,
                        max_lag_events=1000,
                        max_batch_bytes=1_048_576,
                        poll_interval_seconds=0.001,
                        heartbeat_interval_seconds=1,
                    ),
                )
            ]

        frames = asyncio.run(collect())
        assert "event: stream.ready" in frames[0]
        delivered_ids = [
            event.event_id for event in all_events if f"id: {event.event_id}" in "".join(frames)
        ]
        assert delivered_ids == [event.event_id for event in all_events[2:]]
        assert sum("event: run.completed" in frame for frame in frames) == 1


def test_durable_sse_supports_manual_event_source_sequence_resume(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="manual-sse-turn",
            user_content="SSE 手工重连",
        )
        run = _advance_to_validating(repository, run.query_run_id)
        repository.complete_run(
            run.query_run_id,
            expected_revision=run.revision,
            answer="完成",
        )
        all_events = repository.read_event_page(run.query_run_id).events
        after_seq = all_events[1].seq

        def reader(query_run_id: str, cursor: int, event_id: str | None, limit: int):
            assert event_id is None
            return repository.read_event_page(
                query_run_id,
                after_seq=cursor,
                limit=limit,
            )

        async def collect() -> list[str]:
            return [
                frame
                async for frame in durable_sse_generator(
                    query_run_id=run.query_run_id,
                    read_page=reader,
                    initial_after_seq=after_seq,
                    policy=DurableStreamPolicy(
                        batch_size=100,
                        max_lag_events=1000,
                        max_batch_bytes=1_048_576,
                        poll_interval_seconds=0.001,
                        heartbeat_interval_seconds=1,
                    ),
                )
            ]

        frames = asyncio.run(collect())
        payload = "".join(frames)
        assert f'"resume_after_seq":{after_seq}' in frames[0]
        assert all(f"id: {event.event_id}" not in payload for event in all_events[:2])
        assert all(f"id: {event.event_id}" in payload for event in all_events[2:])


def test_slow_consumer_receives_reset_instead_of_unbounded_buffer() -> None:
    run_id = "qry_" + ("1" * 32)

    def reader(query_run_id: str, after_seq: int, event_id: str | None, limit: int):
        assert query_run_id == run_id
        assert event_id is None
        return EventPage(
            query_run_id=query_run_id,
            after_seq=after_seq,
            events=(),
            first_available_seq=1,
            last_available_seq=100,
            terminal=False,
        )

    async def collect() -> list[str]:
        return [
            frame
            async for frame in durable_sse_generator(
                query_run_id=run_id,
                read_page=reader,
                policy=DurableStreamPolicy(
                    batch_size=2,
                    max_lag_events=10,
                    max_batch_bytes=1024,
                    poll_interval_seconds=0.001,
                    heartbeat_interval_seconds=1,
                ),
            )
        ]

    frames = asyncio.run(collect())
    assert len(frames) == 2
    assert "event: stream.reset" in frames[1]
    assert '"code":"SLOW_CONSUMER"' in frames[1]


def test_database_constraints_keep_event_and_turn_sequences_unique(
    engine: Engine,
) -> None:
    with Session(engine) as session:
        repository = _repository(session)
        conversation = repository.create_conversation(kb_id="kb_hak180")
        _, run, _, _ = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="constraint-turn",
            user_content="约束问题",
        )
        assert (
            session.scalar(
                select(func.count(QueryEventModel.id)).where(
                    QueryEventModel.query_run_id == run.query_run_id
                )
            )
            == 1
        )
        assert repository.get_run(run.query_run_id).event_sequence == 1
