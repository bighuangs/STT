"""Durable Query 跨进程身份、双向终态缝与有界关闭测试。"""

from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from time import monotonic

import pytest
from sqlalchemy import func, select, update

from app.agent_system.agents.query.workflow import QUERY_ENTRY_NODE, QUERY_NODE_REGISTRY
from app.api.http import query_runtime_routes
from app.application.query_cache_service import QueryFingerprint
from app.domain.query_checkpoint import (
    QueryExecutionStatus,
    QueryNodeAttemptStatus,
)
from app.domain.query_runtime import QueryRunStatus
from app.domain.query_security import (
    ActiveIndexBinding,
    AuthorizedKnowledgeBase,
    AuthorizedRetrievalScope,
    EgressPolicySnapshot,
    compute_scope_hash,
)
from app.evaluation.observation import QueryGraphEvalAdapter
from app.foundation.query_checkpoint_crypto import (
    QueryCheckpointCipher,
)
from app.infra.retrieval.control_plane_resolvers import (
    active_scope_digest,
    serialize_active_scopes,
)
from app.infra.retrieval.milvus_adapter import ActiveIndexScope
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.models import (
    ApiCredentialModel,
    Base,
    TenantMembershipModel,
    TenantModel,
    UserModel,
)
from app.infra.persistence.sql.p3_models import QueryIdempotencyKeyModel
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.infra.persistence.sql.query_runtime_models import (
    QueryNodeAttemptModel,
    QueryRunCheckpointModel,
    QueryRunExecutionModel,
)
from app.infra.persistence.sql.query_runtime_repository import (
    QueryCheckpointRepository,
    QueryRuntimeRepository,
)
from app.infrastructure.query_runtime_durability import (
    DurableQueryRecoveryController,
    QueryRecoveryStopped,
    bind_query_execution_identity,
    build_durable_query_executor,
    initialize_query_execution,
    recover_durable_query_executions,
    run_durable_query_recovery_loop,
)
from app.security.principal import (
    Principal,
    reset_current_principal,
    set_current_principal,
)


@pytest.fixture
def runtime_database(tmp_path: Path):
    engine = configure_database(f"sqlite:///{(tmp_path / 'durable-recovery.db').as_posix()}")
    Base.metadata.create_all(engine)
    yield engine
    dispose_database()


def _seed_identity(*, api_key: bool = False) -> Principal:
    with session_scope() as session:
        session.add(TenantModel(id="tenant_recovery", name="Recovery", status="active"))
        session.add(
            UserModel(
                id="user_recovery",
                email="recovery@example.test",
                display_name="Recovery user",
                password_hash="not-a-real-password-hash",
                status="active",
                token_version=7,
            )
        )
        session.flush()
        session.add(
            TenantMembershipModel(
                id="membership_recovery",
                tenant_id="tenant_recovery",
                user_id="user_recovery",
                role="Member",
                status="active",
            )
        )
        if api_key:
            session.add(
                ApiCredentialModel(
                    id="api_recovery",
                    tenant_id="tenant_recovery",
                    user_id="user_recovery",
                    name="Recovery API",
                    prefix="stt_recovery",
                    secret_hash="0" * 64,
                    scopes=["kb:kb_recovery:read"],
                    status="active",
                )
            )
    if api_key:
        return Principal(
            user_id="user_recovery",
            tenant_id="tenant_recovery",
            roles=("ServiceAccount",),
            scopes=("kb:kb_recovery:read",),
            auth_type="api_key",
            credential_id="api_recovery",
            request_id="req_recovery",
        )
    return Principal(
        user_id="user_recovery",
        tenant_id="tenant_recovery",
        roles=("Member",),
        scopes=(),
        auth_type="password",
        credential_id="password_recovery",
        request_id="req_recovery",
    )


def _seed_execution(principal: Principal) -> str:
    cipher = QueryCheckpointCipher(base64.urlsafe_b64encode(b"r" * 32).decode("ascii"))
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        conversation = runtime.create_conversation(kb_id="kb_recovery")
        _turn, run, _history, _created = runtime.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="recovery-turn",
            user_content="不得写入普通 checkpoint 列",
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        binding = bind_query_execution_identity(session, principal)
        state = {
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "query_run_id": run.query_run_id,
            "original_query": "不得写入普通 checkpoint 列",
            "deadline_at": run.deadline_at.isoformat() if run.deadline_at else None,
        }
        initialize_query_execution(
            session,
            principal=principal,
            run=run,
            state=state,
            cipher=cipher,
            auth_binding=binding,
        )
        return run.query_run_id


def _fingerprint() -> QueryFingerprint:
    return QueryFingerprint.from_request(
        tenant_id="tenant_recovery",
        principal_scope={"user": "user_recovery"},
        normalized_query="恢复测试",
        conversation_history=[],
        authorized_scope={"scope": "recovery"},
        filters={},
        index_versions=["idx_recovery"],
        query_profile_version="rpf.recovery",
        planner_version="planner.recovery",
        embedding_model_version="embedding.recovery",
        rerank_model_version="rerank.recovery",
        generator_model_version="generator.recovery",
        prompt_version="prompt.recovery",
        policy_version="policy.recovery",
        web_snapshot_policy="disabled",
    )


def _executor(principal: Principal, *, owner: str):
    return build_durable_query_executor(
        principal=principal,
        cipher=QueryCheckpointCipher(base64.urlsafe_b64encode(b"r" * 32).decode("ascii")),
        node_handlers={name: (lambda _state: {}) for name in QUERY_NODE_REGISTRY},
        lease_owner=owner,
    )


def _put_cache(principal: Principal) -> tuple[QueryFingerprint, dict]:
    fingerprint = _fingerprint()
    with session_scope() as session:
        repository = QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        )
        assert repository.put_cache(
            cache_key=fingerprint.key,
            authorized_scope_hash=fingerprint.authorized_scope_hash,
            layer="answer",
            value={"answer": "cached-safe-answer"},
            versions=fingerprint.model_dump(mode="json"),
            classification="internal",
            ttl=timedelta(minutes=5),
        )
        cached, cache_status = repository.get_cache(
            cache_key=fingerprint.key,
            authorized_scope_hash=fingerprint.authorized_scope_hash,
        )
    assert cache_status == "hit"
    assert cached is not None
    return fingerprint, cached


def _evaluation_config() -> dict:
    active_binding = ActiveIndexBinding(
        kb_id="kb_recovery",
        document_id="doc_recovery",
        document_version_id="dver_recovery",
        index_version_id="idx_recovery",
    )
    resources = (
        AuthorizedKnowledgeBase(
            kb_id="kb_recovery",
            document_ids=None,
            grant_source="tenant_admin",
            grant_role="admin",
        ),
    )
    scope_values = {
        "tenant_id": "tenant_recovery",
        "principal_id": "user_recovery",
        "workspace_id": "workspace_recovery",
        "resources": resources,
        "business_filter": None,
        "active_indexes": (active_binding,),
        "policy_version": "query-authz.recovery.v1",
        "entitlement_version": "entitlement.recovery.v1",
    }
    authorized_scope = AuthorizedRetrievalScope(
        **scope_values,
        scope_hash=compute_scope_hash(**scope_values),
    )
    vector_scope = ActiveIndexScope(
        tenant_id="tenant_recovery",
        kb_id="kb_recovery",
        document_id="doc_recovery",
        document_version_id="dver_recovery",
        index_version_id="idx_recovery",
        collection_name="query_recovery",
        partition_name=None,
    )
    return {
        "config_version": "cfg_recovery",
        "retrieval_profile": {},
        "vector_read_scopes": serialize_active_scopes([vector_scope]),
        "vector_read_set_digest": active_scope_digest([vector_scope]),
        "evaluation_security": {
            "scope_authorized": True,
            "authorized_scope": authorized_scope.model_dump(mode="json"),
            "data_classification": "internal",
            "external_llm_allowed": True,
            "egress_policies": {
                "llm_generation": EgressPolicySnapshot(
                    policy_version="egress.recovery.v1",
                    enabled=True,
                    allowed_providers=("configured-llm",),
                    allowed_purposes=("llm_generation",),
                    max_classification="internal",
                ).model_dump(mode="json"),
                "web_search": EgressPolicySnapshot().model_dump(mode="json"),
            },
            "content_security_policy": {
                "policy_version": "rag-security.recovery.v1",
                "web_enabled": False,
                "high_risk_sources_enabled": False,
                "allowed_fetch_domains": [],
                "allowed_output_link_domains": [],
            },
        },
    }


def test_graph_and_eval_adapter_remain_stateless_without_checkpoint_key(
    runtime_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.process.query.agent import main_graph
    from app.runtime import graphs as graph_registry

    principal = _seed_identity()
    monkeypatch.setattr(
        query_runtime_routes,
        "settings",
        query_runtime_routes.settings.model_copy(update={"query_checkpoint_encryption_key": None}),
    )
    with pytest.raises(query_runtime_routes.QueryRuntimeApiError) as key_error:
        query_runtime_routes._checkpoint_cipher()
    assert key_error.value.code == "QUERY_CHECKPOINT_KEY_UNAVAILABLE"

    handlers = {name: (lambda _state: {}) for name in main_graph.QUERY_NODE_REGISTRY}
    handlers["node_query_rewrite"] = lambda _state: {
        "query_plan": {"disposition": "direct_answer"},
        "rewritten_query": "stateless",
    }
    handlers["node_answer_render"] = lambda _state: {
        "answer": "stateless",
    }
    monkeypatch.setattr(main_graph, "_QUERY_NODE_HANDLERS", handlers)
    # 本用例只验证无 Checkpointer 时的执行形态；proof 提交门由回答图专项测试覆盖。
    monkeypatch.setattr(
        main_graph,
        "node_answer_render_after_router",
        lambda _state: "node_answer_commit",
    )
    graph = main_graph.compile_query_graph()
    assert graph.checkpointer is None
    assert graph.invoke({})["answer"] == "stateless"

    monkeypatch.setattr(graph_registry, "_query_graph", graph)
    token = set_current_principal(principal)
    try:
        observation = QueryGraphEvalAdapter().observe(
            {"case_id": "case-stateless", "query": "stateless eval"},
            kb_id="kb_recovery",
            run_id="run-stateless",
            config=_evaluation_config(),
        )
    finally:
        reset_current_principal(token)
    assert observation["answer"] == "stateless"

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(QueryRunExecutionModel)) == 0
        assert session.scalar(select(func.count()).select_from(QueryRunCheckpointModel)) == 0
        assert session.scalar(select(func.count()).select_from(QueryNodeAttemptModel)) == 0


def test_revoked_api_credential_stops_recovery_before_execute(
    runtime_database,
) -> None:
    principal = _seed_identity(api_key=True)
    query_run_id = _seed_execution(principal)
    with session_scope() as session:
        QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        ).claim_idempotency(
            idempotency_key="revoked-credential",
            payload_hash="sha256:" + ("d" * 64),
            query_run_id=query_run_id,
        )
        credential = session.get(ApiCredentialModel, "api_recovery")
        assert credential is not None
        credential.status = "revoked"

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["rejected"] == 1
    assert execute_calls == []
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        idempotency = session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.query_run_id == query_run_id
            )
        )
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == "QUERY_RESUME_CREDENTIAL_REVOKED"
        assert execution.status == QueryExecutionStatus.FAILED
        assert idempotency is not None
        assert idempotency.response_status == 403


def test_cancel_requested_wins_over_revoked_credential_idempotency(
    runtime_database,
) -> None:
    principal = _seed_identity(api_key=True)
    query_run_id = _seed_execution(principal)
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        ).claim_idempotency(
            idempotency_key="cancel-revoked-credential",
            payload_hash="sha256:" + ("b" * 64),
            query_run_id=query_run_id,
        )
        runtime.request_cancel(query_run_id, expected_revision=run.revision)
        credential = session.get(ApiCredentialModel, "api_recovery")
        assert credential is not None
        credential.status = "revoked"

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["rejected"] == 1
    assert execute_calls == []
    with session_scope() as session:
        run = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_run(query_run_id)
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        idempotency = session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.query_run_id == query_run_id
            )
        )
        assert run.status == QueryRunStatus.CANCELLED
        assert execution.status == QueryExecutionStatus.CANCELLED
        assert idempotency is not None
        assert idempotency.response_status == 409


def test_missing_initial_checkpoint_is_terminalized_without_execute(
    runtime_database,
) -> None:
    principal = _seed_identity()
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        conversation = runtime.create_conversation(kb_id="kb_recovery")
        _turn, run, _history, _created = runtime.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="missing-checkpoint",
            user_content="missing checkpoint",
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        query_run_id = run.query_run_id
        QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        ).claim_idempotency(
            idempotency_key="missing-checkpoint",
            payload_hash="sha256:" + ("e" * 64),
            query_run_id=query_run_id,
        )

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["rejected"] == 1
    assert execute_calls == []
    with session_scope() as session:
        run = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_run(query_run_id)
        idempotency = session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.query_run_id == query_run_id
            )
        )
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == "QUERY_CHECKPOINT_NOT_INITIALIZED"
        assert idempotency is not None
        assert idempotency.response_status == 500


def test_cancel_requested_wins_when_durable_execution_is_missing(
    runtime_database,
) -> None:
    principal = _seed_identity()
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        conversation = runtime.create_conversation(kb_id="kb_recovery")
        _turn, run, _history, _created = runtime.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id="cancel-missing-checkpoint",
            user_content="cancel missing checkpoint",
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        query_run_id = run.query_run_id
        QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        ).claim_idempotency(
            idempotency_key="cancel-missing-checkpoint",
            payload_hash="sha256:" + ("c" * 64),
            query_run_id=query_run_id,
        )
        runtime.request_cancel(query_run_id, expected_revision=run.revision)

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["rejected"] == 1
    assert execute_calls == []
    with session_scope() as session:
        run = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_run(query_run_id)
        idempotency = session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.query_run_id == query_run_id
            )
        )
        execution_count = session.scalar(
            select(func.count())
            .select_from(QueryRunExecutionModel)
            .where(QueryRunExecutionModel.query_run_id == query_run_id)
        )
        assert run.status == QueryRunStatus.CANCELLED
        assert idempotency is not None
        assert idempotency.response_status == 409
        assert execution_count == 0


def test_execution_terminal_before_query_run_is_reconciled_without_execute(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    with session_scope() as session:
        session.execute(
            update(QueryRunExecutionModel)
            .where(QueryRunExecutionModel.query_run_id == query_run_id)
            .values(
                status=QueryExecutionStatus.INTERRUPTED.value,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                safe_error_code="QUERY_INTERRUPTED_DURING_NODE",
            )
        )

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["reconciled"] == 1
    assert execute_calls == []
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == "QUERY_INTERRUPTED_DURING_NODE"
        assert execution.status == QueryExecutionStatus.INTERRUPTED
        assert execution.next_node is None


def test_query_run_terminal_before_execution_is_reconciled_without_execute(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
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
            answer="恢复后的终态答案",
            result={"answer": "恢复后的终态答案"},
        )

    execute_calls: list[str] = []
    report = recover_durable_query_executions(
        lambda run_id, _principal, _query: execute_calls.append(run_id)
    )
    assert report["reconciled"] == 1
    assert execute_calls == []
    with session_scope() as session:
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        assert execution.status == QueryExecutionStatus.COMPLETED
        assert execution.next_node is None
        assert execution.lease_owner is None


def test_cached_terminal_run_reconciles_execution_lease(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    executor = _executor(principal, owner="cache-terminal-worker")
    lease = executor.claim(query_run_id)
    preflight = executor.preflight(
        query_run_id=query_run_id,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        lease=lease,
    )
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        runtime.fail_run(
            query_run_id,
            expected_revision=run.revision,
            error_code="QUERY_TERMINAL_BEFORE_CACHE",
            safe_message="查询已终结。",
            retryable=True,
            error_detail_ref=f"trace:{run.trace_id}",
        )

    assert query_runtime_routes._cached_completion(
        principal=principal,
        query_run_id=query_run_id,
        cached={},
        fingerprint=_fingerprint(),
        cache_role="hit",
        execution_lease=lease,
        execution_preflight=preflight,
    )
    with session_scope() as session:
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        assert execution.status == QueryExecutionStatus.FAILED
        assert execution.lease_owner is None
        assert execution.next_node is None


def test_cached_cancel_completes_idempotency_and_execution(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    executor = _executor(principal, owner="cache-cancel-worker")
    lease = executor.claim(query_run_id)
    preflight = executor.preflight(
        query_run_id=query_run_id,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        lease=lease,
    )
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        QueryReliabilityRepository(
            session,
            tenant_id=principal.tenant_id,
        ).claim_idempotency(
            idempotency_key="cache-cancel",
            payload_hash="sha256:" + ("a" * 64),
            query_run_id=query_run_id,
        )
        runtime.request_cancel(
            query_run_id,
            expected_revision=run.revision,
        )

    assert query_runtime_routes._cached_completion(
        principal=principal,
        query_run_id=query_run_id,
        cached={},
        fingerprint=_fingerprint(),
        cache_role="hit",
        execution_lease=lease,
        execution_preflight=preflight,
    )
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        idempotency = session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.query_run_id == query_run_id
            )
        )
        assert run.status == QueryRunStatus.CANCELLED
        assert execution.status == QueryExecutionStatus.CANCELLED
        assert execution.lease_owner is None
        assert idempotency is not None
        assert idempotency.response_status == 409


def test_active_attempt_preflight_rejects_before_available_cache(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    dead_worker = _executor(principal, owner="dead-cache-worker")
    dead_lease = dead_worker.claim(query_run_id)
    with session_scope() as session:
        store = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        attempt = store.schedule_attempt(
            dead_lease,
            node_name=QUERY_ENTRY_NODE,
            effect_class="provider_and_history",
        )
        store.mark_attempt_started(dead_lease, attempt.attempt_id)
        session.execute(
            update(QueryRunExecutionModel)
            .where(QueryRunExecutionModel.query_run_id == query_run_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    _fingerprint_value, cached = _put_cache(principal)
    assert cached["answer"] == "cached-safe-answer"

    recovery = _executor(principal, owner="recovery-cache-worker")
    recovery_lease = recovery.claim(query_run_id)
    with pytest.raises(query_runtime_routes.QueryInterruptedDuringNode):
        recovery.preflight(
            query_run_id=query_run_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            lease=recovery_lease,
        )
    query_runtime_routes._finalize_worker_failure(
        principal=principal,
        query_run_id=query_run_id,
        error_code="QUERY_INTERRUPTED_DURING_NODE",
        safe_message="查询在 active node 中断。",
        retryable=True,
    )

    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        store = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        execution = store.get_execution(query_run_id)
        latest_attempt = store.latest_attempt(query_run_id)
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == "QUERY_INTERRUPTED_DURING_NODE"
        assert execution.status == QueryExecutionStatus.INTERRUPTED
        assert latest_attempt is not None
        assert latest_attempt.status == QueryNodeAttemptStatus.INTERRUPTED


@pytest.mark.parametrize(
    ("failure", "expected_exception", "error_code"),
    [
        (
            "corrupt_ciphertext",
            query_runtime_routes.QueryCheckpointLoadError,
            "QUERY_CHECKPOINT_CORRUPT",
        ),
        (
            "old_workflow",
            query_runtime_routes.QueryCheckpointCompatibilityError,
            "QUERY_CHECKPOINT_VERSION_OR_ROUTE_UNSUPPORTED",
        ),
    ],
)
def test_invalid_checkpoint_preflight_rejects_before_available_cache(
    runtime_database,
    failure: str,
    expected_exception: type[Exception],
    error_code: str,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    _fingerprint_value, cached = _put_cache(principal)
    assert cached["answer"] == "cached-safe-answer"
    with session_scope() as session:
        checkpoint = session.scalar(
            select(QueryRunCheckpointModel).where(
                QueryRunCheckpointModel.query_run_id == query_run_id,
                QueryRunCheckpointModel.sequence == 0,
            )
        )
        assert checkpoint is not None
        if failure == "corrupt_ciphertext":
            corrupted = bytearray(checkpoint.ciphertext)
            corrupted[0] ^= 1
            checkpoint.ciphertext = bytes(corrupted)
        else:
            checkpoint.workflow_version = "query-workflow.old"

    recovery = _executor(principal, owner="recovery-corrupt-worker")
    recovery_lease = recovery.claim(query_run_id)
    with pytest.raises(expected_exception):
        recovery.preflight(
            query_run_id=query_run_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            lease=recovery_lease,
        )
    query_runtime_routes._finalize_worker_failure(
        principal=principal,
        query_run_id=query_run_id,
        error_code=error_code,
        safe_message="查询 checkpoint 无法恢复。",
        retryable=True,
    )
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        execution = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).get_execution(query_run_id)
        run = runtime.get_run(query_run_id)
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == error_code
        assert execution.status == QueryExecutionStatus.FAILED


def test_shutdown_controller_fences_active_attempt_and_stops_new_claims(
    runtime_database,
) -> None:
    principal = _seed_identity()
    query_run_id = _seed_execution(principal)
    controller = DurableQueryRecoveryController(owner_id="query-recovery:test")
    with controller.claim_gate():
        with session_scope() as session:
            store = QueryCheckpointRepository(
                session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
            lease = store.claim_execution(
                query_run_id,
                lease_owner=controller.owner_id,
                lease_duration=timedelta(minutes=5),
            )
            attempt = store.schedule_attempt(
                lease,
                node_name=QUERY_ENTRY_NODE,
                effect_class="provider_and_history",
            )
            store.mark_attempt_started(lease, attempt.attempt_id)

    started = monotonic()
    report = controller.stop_and_interrupt_bounded(timeout_seconds=1)
    assert monotonic() - started < 1.5
    assert report["timed_out"] == 0
    assert report["interrupted"] == 1
    with pytest.raises(QueryRecoveryStopped, match="stopping"):
        with controller.claim_gate():
            pass
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        run = runtime.get_run(query_run_id)
        store = QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        execution = store.get_execution(query_run_id)
        latest_attempt = store.latest_attempt(query_run_id)
        assert run.status == QueryRunStatus.FAILED
        assert run.error_code == "QUERY_INTERRUPTED_DURING_SHUTDOWN"
        assert execution.status == QueryExecutionStatus.INTERRUPTED
        assert latest_attempt is not None
        assert latest_attempt.status == QueryNodeAttemptStatus.INTERRUPTED

    follow_up = recover_durable_query_executions(lambda *_args: None)
    assert follow_up["scanned"] == 0


def test_recovery_loop_yields_readiness_and_stops_while_execute_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = ThreadEvent()
    release = ThreadEvent()

    def blocked_recover(_execute, *, limit):
        assert limit == 1
        entered.set()
        release.wait(timeout=5)
        return {
            "scanned": 1,
            "resumed": 0,
            "reconciled": 0,
            "rejected": 0,
            "raced": 0,
            "failed": 0,
        }

    import app.infrastructure.query_runtime_durability as durability

    monkeypatch.setattr(durability, "recover_durable_query_executions", blocked_recover)

    async def exercise() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_durable_query_recovery_loop(
                stop,
                execute=lambda *_args: None,
                interval_seconds=60,
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0.005)
        # loop 已在 daemon worker 内阻塞，事件循环仍可推进（readiness 可 yield）。
        await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, timeout=0.5)

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_query_server_lifespan_is_ready_and_stops_waiting_for_blocked_recovery(
    runtime_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI

    import app.api.http.query_server as query_server
    import app.infrastructure.query_runtime_durability as durability

    entered = ThreadEvent()
    release = ThreadEvent()

    def blocked_recover(_execute, *, limit):
        assert limit == 1
        entered.set()
        release.wait(timeout=5)
        return {
            "scanned": 1,
            "resumed": 0,
            "reconciled": 0,
            "rejected": 0,
            "raced": 0,
            "failed": 0,
        }

    async def passive_loop(stop, **_kwargs):
        await stop.wait()

    monkeypatch.setattr(durability, "recover_durable_query_executions", blocked_recover)
    monkeypatch.setattr(query_server, "observability_lifespan", lambda _name: nullcontext())
    monkeypatch.setattr(query_server, "bootstrap_security", lambda: None)
    monkeypatch.setattr(query_server, "bootstrap_evaluation", lambda: None)
    monkeypatch.setattr(query_server, "validate_startup", lambda _service: None)
    monkeypatch.setattr(
        query_server,
        "recover_stale_evaluation_runs",
        lambda: {"scanned": 0},
    )
    monkeypatch.setattr(
        query_server,
        "recover_expired_query_runs",
        lambda: {"scanned": 0},
    )
    monkeypatch.setattr(query_server, "run_query_runtime_recovery_loop", passive_loop)
    monkeypatch.setattr(query_server, "run_evaluation_recovery_loop", passive_loop)
    monkeypatch.setattr(
        query_server,
        "interrupt_owned_evaluation_runs",
        lambda _owner: {"scanned": 0},
    )
    monkeypatch.setattr(query_server, "evaluation_execution_owner", lambda: "eval:test")
    monkeypatch.setattr(query_server, "close_runtime_resources", lambda: None)

    async def exercise() -> None:
        started = monotonic()
        async with query_server.lifespan(FastAPI()):
            while not entered.is_set():
                await asyncio.sleep(0.005)
            assert monotonic() - started < 1
        assert monotonic() - started < 1.5

    try:
        asyncio.run(asyncio.wait_for(exercise(), timeout=1.75))
    finally:
        release.set()
