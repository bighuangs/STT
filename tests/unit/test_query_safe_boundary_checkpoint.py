"""可靠 Query 的密文节点边界、fence 与不重放合同。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event, select, update
from sqlalchemy.orm import Session

from app.agent_system.agents.query.durable_executor import (
    DurableQueryExecutor,
    QueryBoundaryCancellationRequested,
    QueryCheckpointCompatibilityError,
    QueryCheckpointLoadError,
    QueryExecutionCancelled,
    QueryExecutionDeadlineExceeded,
    QueryInterruptedDuringNode,
)
from app.agent_system.agents.query.workflow import QUERY_NODE_REGISTRY, QUERY_WORKFLOW_VERSION
from app.domain.query_checkpoint import (
    QUERY_CHECKPOINT_NAMESPACE,
    QUERY_CHECKPOINT_SCHEMA_VERSION,
    QUERY_STATE_SCHEMA_VERSION,
    QueryExecutionStatus,
    QueryNodeAttemptStatus,
)
from app.domain.query_runtime import QueryEventType, QueryRunStatus
from app.domain.answering.plan import (
    AnswerDecision,
    AnswerPlan,
    ClaimKind,
    ClaimPlan,
    RenderingRule,
)
from app.domain.answering.proof import (
    AnswerProof,
    ClaimProof,
    CompletenessProofStatus,
    OverallProofStatus,
    ProofDimensionStatus,
    ProofRelation,
    ProofVerdict,
    VerifierRef,
)
from app.domain.answering.question import QuestionContract
from app.domain.evidence import AnswerSpan, sha256_digest
from app.foundation.query_checkpoint_crypto import (
    QueryCheckpointCipher,
    QueryCheckpointCorrupt,
    QueryCheckpointEncryptionContext,
    QueryCheckpointKeyInvalid,
)
from app.infra.persistence.sql.models import Base
from app.infra.persistence.sql.query_runtime_models import (
    QueryEventModel,
    QueryRunCheckpointModel,
    QueryRunExecutionModel,
)
from app.infra.persistence.sql.query_runtime_repository import (
    QueryCheckpointRepository,
    QueryExecutionFenceLost,
    QueryRuntimeRepository,
)


@dataclass
class _Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    result = create_engine(
        f"sqlite:///{(tmp_path / 'query-checkpoint.db').as_posix()}",
        future=True,
    )
    event.listen(
        result,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(result)
    yield result
    result.dispose()


def _seed_run(engine: Engine) -> str:
    with Session(engine) as session:
        repository = QueryRuntimeRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        conversation = repository.create_conversation(kb_id="kb_checkpoint")
        _turn, run, _snapshot, _created = repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="checkpoint-turn",
            user_content="不得出现在 checkpoint 明文列的敏感问题",
        )
        session.commit()
        return run.query_run_id


def _state(query_run_id: str, *, deadline_at: str | None = None) -> dict:
    return {
        "tenant_id": "tenant_checkpoint",
        "user_id": "user_checkpoint",
        "query_run_id": query_run_id,
        "deadline_at": deadline_at,
        "original_query": "不得出现在 checkpoint 明文列的敏感问题",
        "query_plan": {},
        "answer": "",
    }


def _safe_answer_proof() -> dict:
    """为 durable executor 测试建立真实的非检索可提交 proof。"""

    answer = "安全答案"
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "0" * 32,
        source_query="测试安全直答",
        facets=(),
        builder_version="durable-checkpoint-test.v1",
    )
    claim = ClaimPlan.build(
        ordinal=0,
        obligation_id=None,
        witness_ids=(),
        claim_kind=ClaimKind.UNCERTAINTY,
        rendering_rule=RenderingRule.SYSTEM_NOTICE,
        template_id="notice.ui-help.v1",
    )
    plan = AnswerPlan.build(
        question_contract=question,
        catalog=None,
        decision=AnswerDecision.DIRECT_ANSWER,
        claims=(claim,),
    )
    verifier = VerifierRef(
        verifier_id="durable-checkpoint-test",
        verifier_version="v1",
    )
    claim_proof = ClaimProof.build(
        claim_id=claim.claim_id,
        obligation_id=None,
        status=ProofVerdict.PROVEN,
        method=RenderingRule.SYSTEM_NOTICE,
        witness_ids=(),
        answer_span=AnswerSpan(start=0, end=len(answer)),
        rendered_text_hash=sha256_digest(answer),
        relation=ProofRelation.NOT_APPLICABLE,
        verifier=verifier,
    )
    return AnswerProof.build(
        answer_plan=plan,
        question_contract=question,
        catalog=None,
        answer_id="ans_" + "0" * 32,
        answer_text=answer,
        overall_status=OverallProofStatus.NOT_APPLICABLE,
        grounding_status=ProofDimensionStatus.NOT_APPLICABLE,
        relevance_status=ProofDimensionStatus.NOT_APPLICABLE,
        completeness_status=CompletenessProofStatus.NOT_APPLICABLE,
        claim_proofs=(claim_proof,),
        obligation_proofs=(),
        verifier_versions=(verifier,),
    ).to_dict()


def _store_scope(engine: Engine):
    @contextmanager
    def scope():
        with Session(engine) as session:
            try:
                yield QueryCheckpointRepository(
                    session,
                    tenant_id="tenant_checkpoint",
                    user_id="user_checkpoint",
                )
                session.commit()
            except BaseException:
                session.rollback()
                raise

    return scope


def _handlers(calls: dict[str, int], *, crashing_node: str | None = None):
    class SimulatedProcessKill(BaseException):
        pass

    def handler(node_name: str):
        def invoke(_state: dict) -> dict:
            calls[node_name] = calls.get(node_name, 0) + 1
            if node_name == crashing_node:
                raise SimulatedProcessKill()
            if node_name == "node_query_rewrite":
                return {
                    "query_plan": {
                        "disposition": "direct_answer",
                        "routes": [],
                    }
                }
            if node_name == "node_answer_render":
                return {
                    "answer": "安全答案",
                    "answer_proof": _safe_answer_proof(),
                }
            return {}

        return invoke

    return {
        node_name: handler(node_name) for node_name in QUERY_NODE_REGISTRY
    }, SimulatedProcessKill


def _executor(
    engine: Engine,
    *,
    handlers,
    clock: _Clock,
    owner: str,
) -> DurableQueryExecutor:
    return DurableQueryExecutor(
        store_scope=_store_scope(engine),
        cipher=QueryCheckpointCipher(bytes(range(32))),
        node_handlers=handlers,
        lease_owner=owner,
        lease_duration=timedelta(seconds=5),
        now=clock,
    )


def test_cipher_requires_independent_256_bit_key_and_authenticates_context() -> None:
    with pytest.raises(QueryCheckpointKeyInvalid):
        QueryCheckpointCipher("not-a-32-byte-key")
    cipher = QueryCheckpointCipher(bytes(range(32)))
    context = QueryCheckpointEncryptionContext(
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        query_run_id="qry_test",
        sequence=0,
        completed_node=None,
        next_node="node_query_rewrite",
        checkpoint_namespace=QUERY_CHECKPOINT_NAMESPACE,
        workflow_version=QUERY_WORKFLOW_VERSION,
        state_schema_version=QUERY_STATE_SCHEMA_VERSION,
        checkpoint_schema_version=QUERY_CHECKPOINT_SCHEMA_VERSION,
    )
    state = {
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "query_run_id": context.query_run_id,
        "original_query": "敏感问题",
    }
    encrypted = cipher.encrypt_state(state, context=context)
    assert "敏感问题".encode() not in encrypted.ciphertext
    assert "ciphertext=" not in repr(encrypted)
    assert "nonce=" not in repr(encrypted)
    assert cipher.decrypt_state(encrypted, context=context) == state
    wrong_context = replace(context, sequence=1)
    with pytest.raises(QueryCheckpointCorrupt):
        cipher.decrypt_state(encrypted, context=wrong_context)


def test_initial_checkpoint_precedes_all_nodes_and_final_boundary_is_durable(
    engine: Engine,
) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    executor = _executor(engine, handlers=handlers, clock=clock, owner="worker-a")

    assert executor.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    assert calls == {}
    with Session(engine) as session:
        checkpoint = session.scalar(select(QueryRunCheckpointModel))
        assert checkpoint is not None
        assert checkpoint.sequence == 0
        assert checkpoint.next_node == "node_query_rewrite"
        assert "敏感问题".encode() not in bytes(checkpoint.ciphertext)
        assert not hasattr(checkpoint, "state_json")

    result = executor.execute(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
    )
    assert result.state["answer"] == "安全答案"
    assert calls == {
        "node_query_rewrite": 1,
        "node_question_contract": 1,
        "node_evidence_witness": 1,
        "node_answer_plan": 1,
        "node_answer_verify": 1,
        "node_answer_coverage": 1,
        "node_answer_render": 1,
        "node_answer_commit": 1,
    }
    assert result.checkpoint_sequence == 8
    execution = executor.finish_completed(result)
    assert execution.status == QueryExecutionStatus.COMPLETED
    assert execution.next_node is None


def test_killed_active_node_becomes_interrupted_and_is_never_replayed(
    engine: Engine,
) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    crashing_handlers, simulated_kill = _handlers(
        calls,
        crashing_node="node_query_rewrite",
    )
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    first = _executor(engine, handlers=crashing_handlers, clock=clock, owner="worker-old")
    first.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    with pytest.raises(simulated_kill):
        first.execute(
            query_run_id=query_run_id,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
    assert calls["node_query_rewrite"] == 1

    clock.advance(seconds=6)
    recovery_handlers, _unused = _handlers(calls)
    recovery = _executor(
        engine,
        handlers=recovery_handlers,
        clock=clock,
        owner="worker-recovery",
    )
    with pytest.raises(QueryInterruptedDuringNode):
        recovery.execute(
            query_run_id=query_run_id,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
    assert calls["node_query_rewrite"] == 1
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        execution = store.get_execution(query_run_id)
        attempt = store.latest_attempt(query_run_id)
        assert execution.status == QueryExecutionStatus.INTERRUPTED
        assert execution.safe_error_code == "QUERY_INTERRUPTED_DURING_NODE"
        assert attempt is not None
        assert attempt.status == QueryNodeAttemptStatus.INTERRUPTED


def test_expired_fence_rejects_a_stale_worker(engine: Engine) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    executor = _executor(engine, handlers=handlers, clock=clock, owner="initializer")
    executor.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        stale = store.claim_execution(
            query_run_id,
            lease_owner="worker-stale",
            lease_duration=timedelta(seconds=5),
            now=clock(),
        )
        session.commit()
    clock.advance(seconds=6)
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        current = store.claim_execution(
            query_run_id,
            lease_owner="worker-current",
            lease_duration=timedelta(seconds=5),
            now=clock(),
        )
        session.commit()
    assert current.fencing_token == stale.fencing_token + 1
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        with pytest.raises(QueryExecutionFenceLost):
            store.schedule_attempt(
                stale,
                node_name="node_query_rewrite",
                effect_class="provider_and_history",
                now=clock(),
            )


@pytest.mark.parametrize("failure", ["old_version", "corrupt_ciphertext"])
def test_old_or_corrupt_checkpoint_fails_before_any_node(
    engine: Engine,
    failure: str,
) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    executor = _executor(engine, handlers=handlers, clock=clock, owner="worker-a")
    executor.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    with Session(engine) as session:
        if failure == "old_version":
            session.execute(
                update(QueryRunExecutionModel)
                .where(QueryRunExecutionModel.query_run_id == query_run_id)
                .values(workflow_version="query-workflow.v0")
            )
        else:
            checkpoint = session.scalar(
                select(QueryRunCheckpointModel).where(
                    QueryRunCheckpointModel.query_run_id == query_run_id
                )
            )
            assert checkpoint is not None
            checkpoint.ciphertext = bytes(checkpoint.ciphertext[:-1]) + bytes(
                [checkpoint.ciphertext[-1] ^ 1]
            )
        session.commit()

    expected = (
        QueryCheckpointCompatibilityError if failure == "old_version" else QueryCheckpointLoadError
    )
    with pytest.raises(expected):
        executor.execute(
            query_run_id=query_run_id,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
    assert calls == {}


def test_deadline_and_cancel_stop_before_provider_and_preserve_fence_for_terminalization(
    engine: Engine,
) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    executor = _executor(engine, handlers=handlers, clock=clock, owner="worker-deadline")
    executor.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id, deadline_at="2026-08-19T00:00:00+00:00"),
    )
    with pytest.raises(QueryExecutionDeadlineExceeded) as captured:
        executor.execute(
            query_run_id=query_run_id,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
    executor.finish_stopped(
        captured.value.lease,
        status=QueryExecutionStatus.TIMED_OUT,
        safe_error_code="QUERY_DEADLINE_EXCEEDED",
    )
    assert calls == {}

    other_run_id = _seed_run(engine)
    cancel_executor = _executor(
        engine,
        handlers=handlers,
        clock=clock,
        owner="worker-cancel",
    )
    cancel_executor.initialize(
        query_run_id=other_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(other_run_id),
    )

    def cancelled(_state: object) -> None:
        raise QueryBoundaryCancellationRequested("cancel requested")

    with pytest.raises(QueryExecutionCancelled) as cancel_capture:
        cancel_executor.execute(
            query_run_id=other_run_id,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
            boundary_guard=cancelled,
        )
    cancel_executor.finish_stopped(
        cancel_capture.value.lease,
        status=QueryExecutionStatus.CANCELLED,
        safe_error_code="QUERY_CANCELLED",
    )
    assert calls == {}


def test_mark_started_cas_rejects_terminal_fence_interleaving(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2 在 T1 assert 后终结时，T1 不得取得 handler 执行授权。"""

    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    executor = _executor(engine, handlers=handlers, clock=clock, owner="worker-t1")
    executor.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        lease = store.claim_execution(
            query_run_id,
            lease_owner="worker-t1",
            lease_duration=timedelta(seconds=5),
            now=clock(),
        )
        attempt = store.schedule_attempt(
            lease,
            node_name="node_query_rewrite",
            effect_class="provider_and_history",
            now=clock(),
        )
        session.commit()

    with Session(engine) as first_session:
        first_store = QueryCheckpointRepository(
            first_session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        original_assert = first_store._assert_lease_row

        def terminalize_after_assert(row, asserted_lease, *, now=None) -> None:
            original_assert(row, asserted_lease, now=now)
            # 释放 T1 读事务后由真实第二 session 提交 QueryRun 终态和新 fence。
            first_session.commit()
            with Session(engine) as second_session:
                runtime = QueryRuntimeRepository(
                    second_session,
                    tenant_id="tenant_checkpoint",
                    user_id="user_checkpoint",
                )
                run = runtime.get_run(query_run_id)
                runtime.fail_run(
                    query_run_id,
                    expected_revision=run.revision,
                    error_code="QUERY_TERMINATED_BY_TEST",
                    safe_message="查询已终止。",
                    retryable=True,
                )
                QueryCheckpointRepository(
                    second_session,
                    tenant_id="tenant_checkpoint",
                    user_id="user_checkpoint",
                ).reconcile_terminal_run(query_run_id)
                second_session.commit()

        monkeypatch.setattr(first_store, "_assert_lease_row", terminalize_after_assert)
        with pytest.raises(QueryExecutionFenceLost):
            first_store.mark_attempt_started(
                lease,
                attempt.attempt_id,
                now=clock(),
            )
        first_session.rollback()

    assert calls == {}
    with Session(engine) as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        execution = store.get_execution(query_run_id)
        latest_attempt = store.latest_attempt(query_run_id)
        assert execution.status == QueryExecutionStatus.FAILED
        assert latest_attempt is not None
        assert latest_attempt.status == QueryNodeAttemptStatus.INTERRUPTED
        assert latest_attempt.safe_error_code == "QUERY_ATTEMPT_TERMINAL_RECONCILED"


def test_final_checkpoint_recovers_without_reinvoking_any_handler(
    engine: Engine,
) -> None:
    query_run_id = _seed_run(engine)
    calls: dict[str, int] = {}
    handlers, _kill = _handlers(calls)
    clock = _Clock(datetime(2026, 8, 20, tzinfo=UTC))
    first = _executor(engine, handlers=handlers, clock=clock, owner="worker-final-old")
    first.initialize(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
        state=_state(query_run_id),
    )
    first_result = first.execute(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
    )
    assert first_result.state["answer"] == "安全答案"
    calls_at_kill = dict(calls)

    # 模拟 final checkpoint 已提交、QueryRun complete 尚未提交时 kill。
    clock.advance(seconds=6)

    def forbidden_handler(_state: dict) -> dict:
        raise AssertionError("final checkpoint recovery must not call a handler")

    recovery = _executor(
        engine,
        handlers={name: forbidden_handler for name in QUERY_NODE_REGISTRY},
        clock=clock,
        owner="worker-final-recovery",
    )
    recovered = recovery.execute(
        query_run_id=query_run_id,
        tenant_id="tenant_checkpoint",
        user_id="user_checkpoint",
    )
    assert recovered.state["answer"] == "安全答案"
    assert calls == calls_at_kill

    with Session(engine) as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        run = runtime.get_run(query_run_id)
        for target in (
            QueryRunStatus.PLANNING,
            QueryRunStatus.RETRIEVING,
            QueryRunStatus.PACKING,
            QueryRunStatus.GENERATING,
            QueryRunStatus.VALIDATING,
        ):
            run = runtime.transition_run(
                query_run_id,
                expected_revision=run.revision,
                target=target,
            )
        runtime.complete_run(
            query_run_id,
            expected_revision=run.revision,
            answer="安全答案",
            result={"answer": "安全答案"},
        )
        QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        ).reconcile_terminal_run(query_run_id)
        session.commit()

    with Session(engine) as session:
        execution = QueryCheckpointRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        ).get_execution(query_run_id)
        assert execution.status == QueryExecutionStatus.COMPLETED


def test_final_answer_event_operations_are_idempotent_after_partial_commit(
    engine: Engine,
) -> None:
    query_run_id = _seed_run(engine)
    first_chunk = "甲" * 8_000
    second_chunk = "乙"
    with Session(engine) as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        runtime.append_event(
            query_run_id,
            event_type=QueryEventType.ANSWER_DELTA,
            stage=QueryRunStatus.VALIDATING.value,
            operation_id="final-answer:00000000",
            persistence_class="compactable",
            data={"operation": "replace", "text": first_chunk},
        )
        session.commit()

    # 模拟 kill 后从第一块重新 materialize；logical operation key 必须去重。
    with Session(engine) as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id="tenant_checkpoint",
            user_id="user_checkpoint",
        )
        for operation_id, operation, text in (
            ("final-answer:00000000", "replace", first_chunk),
            ("final-answer:00000001", "append", second_chunk),
        ):
            runtime.append_event(
                query_run_id,
                event_type=QueryEventType.ANSWER_DELTA,
                stage=QueryRunStatus.VALIDATING.value,
                operation_id=operation_id,
                persistence_class="compactable",
                data={"operation": operation, "text": text},
            )
        session.commit()

    with Session(engine) as session:
        rows = session.scalars(
            select(QueryEventModel)
            .where(
                QueryEventModel.query_run_id == query_run_id,
                QueryEventModel.event_type == QueryEventType.ANSWER_DELTA.value,
            )
            .order_by(QueryEventModel.seq)
        ).all()
        assert [row.operation_id for row in rows] == [
            "final-answer:00000000",
            "final-answer:00000001",
        ]
        assert "".join(str(row.data_json["text"]) for row in rows) == (first_chunk + second_chunk)
