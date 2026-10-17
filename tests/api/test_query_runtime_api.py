from __future__ import annotations

import base64
from datetime import timedelta
from hashlib import sha256
import inspect
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest
from sqlalchemy import func, select

import app.api.http.query_runtime_routes as runtime_routes
from app.api.http import p3_routes
from app.application.query_cache_service import (
    QueryFingerprint,
    QueryReplayManifest,
    canonical_hash,
)
from app.domain.content_security import ContentSecurityPolicy
from app.domain.query_runtime import QueryEventType, QueryRunStatus
from app.domain.query_security import (
    ActiveIndexBinding,
    AuthorizedKnowledgeBase,
    AuthorizedRetrievalScope,
    DataClassification,
    EgressPolicySnapshot,
    compute_scope_hash,
)
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.models import (
    ApiCredentialModel,
    Base,
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
    TenantMembershipModel,
    TenantModel,
    UserModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.infra.persistence.sql.p3_models import QueryIdempotencyKeyModel
from app.infra.persistence.sql.query_runtime_models import (
    QueryRunCheckpointModel,
    QueryRunExecutionModel,
    QueryRunModel,
)
from app.infra.persistence.sql.query_runtime_repository import (
    QueryCheckpointRepository,
    QueryRuntimeRepository,
)
from app.security.principal import Principal


def _scope() -> AuthorizedRetrievalScope:
    resources = (
        AuthorizedKnowledgeBase(
            kb_id="kb_runtime",
            document_ids=None,
            grant_source="tenant_admin",
            grant_role="admin",
        ),
    )
    indexes = (
        ActiveIndexBinding(
            kb_id="kb_runtime",
            document_id="doc_runtime",
            document_version_id="dver_runtime",
            index_version_id="idx_runtime",
        ),
    )
    values = {
        "tenant_id": "tenant_runtime",
        "principal_id": "user_runtime",
        "workspace_id": "workspace_runtime",
        "resources": resources,
        "business_filter": None,
        "active_indexes": indexes,
        "policy_version": "query-authz.test.v1",
        "entitlement_version": "entitlement.test.v1",
    }
    return AuthorizedRetrievalScope(
        **values,
        scope_hash=compute_scope_hash(**values),
    )


def _frozen(payload) -> runtime_routes._FrozenExecution:
    scope = _scope()
    config: dict = {}
    profile = {
        "profile_id": "rpf_test",
        "version": 1,
        "config_hash": sha256(json.dumps(config, separators=(",", ":")).encode()).hexdigest(),
        "config": config,
    }
    llm_policy = EgressPolicySnapshot(
        policy_version="egress.test.llm.v1",
        enabled=True,
        allowed_providers=("configured-llm",),
        allowed_purposes=("llm_generation",),
        max_classification=DataClassification.INTERNAL,
        retention_policy="test-no-retention",
    )
    web_policy = EgressPolicySnapshot()
    security_policy = ContentSecurityPolicy(
        policy_version="rag-security.test.v1",
    )
    vector_read_scopes = [
        {
            "tenant_id": "tenant_runtime",
            "kb_id": "kb_runtime",
            "document_id": "doc_runtime",
            "document_version_id": "dver_runtime",
            "index_version_id": "idx_runtime",
            "collection_name": "kb_runtime_chunks",
            "partition_name": None,
        }
    ]
    vector_read_set_digest = canonical_hash(
        {
            "schema_version": "vector-read-set.v1",
            "scopes": vector_read_scopes,
        }
    )
    snapshot = {
        "schema_version": "query-execution-config.v1",
        "config_version": runtime_routes.settings.config_version,
        "authorized_scope": scope.model_dump(mode="json"),
        "retrieval_profile": profile,
        "data_classification": "internal",
        "external_llm_allowed": True,
        "vector_read_set_digest": vector_read_set_digest,
        "egress_policies": {
            "llm_generation": llm_policy.model_dump(mode="json"),
            "web_search": web_policy.model_dump(mode="json"),
        },
        "content_security_policy": security_policy.model_dump(mode="json"),
        "egress": {
            "llm": {
                "allowed": True,
                "provider": "configured-llm",
                "purpose": "llm_generation",
                "data_classes": ["internal"],
                "redactions": [],
                "allowed_domains": [],
                "retention_policy": "test-no-retention",
                "reason_code": "EGRESS_ALLOWED",
                "decision_id": "egd_" + "1" * 64,
                "policy_version": "egress.test.llm.v1",
                "scope_hash": scope.scope_hash,
            },
            "web": {
                "allowed": False,
                "provider": "dashscope-web",
                "purpose": "web_search",
                "data_classes": ["internal"],
                "redactions": [],
                "allowed_domains": [],
                "retention_policy": "no-retention",
                "reason_code": "EGRESS_NOT_REQUESTED",
                "decision_id": "egd_" + "2" * 64,
                "policy_version": "egress.v1-deny",
                "scope_hash": scope.scope_hash,
            },
            "decisions": {},
            "llm_local": False,
            "web_search_allowed": False,
        },
    }
    return runtime_routes._FrozenExecution(
        scope=scope,
        profile=profile,
        index_snapshot={
            "schema_version": "query-index-snapshot.v1",
            "scope_hash": scope.scope_hash,
            "active_indexes": [item.model_dump(mode="json") for item in scope.active_indexes],
            "vector_read_scopes": vector_read_scopes,
            "vector_read_set_digest": vector_read_set_digest,
        },
        config_snapshot=snapshot,
        execution_query=payload.query,
    )


def _fingerprint() -> QueryFingerprint:
    return QueryFingerprint.from_request(
        tenant_id="tenant_runtime",
        principal_scope={"user": "user_runtime"},
        normalized_query="测试问题",
        conversation_history=[],
        authorized_scope={"scope": "test"},
        filters={},
        index_versions=["idx_runtime"],
        query_profile_version="rpf_test@1",
        planner_version="planner.test",
        embedding_model_version="embedding.test",
        rerank_model_version="rerank.test",
        generator_model_version="generator.test",
        prompt_version="prompt.test",
        policy_version="policy.test",
        web_snapshot_policy="disabled",
    )


def _member_frozen(payload) -> runtime_routes._FrozenExecution:
    frozen = _frozen(payload)
    resources = (
        AuthorizedKnowledgeBase(
            kb_id="kb_runtime",
            document_ids=None,
            grant_source="user_acl",
            grant_role="query",
        ),
    )
    values = {
        "tenant_id": "tenant_runtime",
        "principal_id": "user_runtime",
        "workspace_id": "workspace_runtime",
        "resources": resources,
        "business_filter": None,
        "active_indexes": frozen.scope.active_indexes,
        "policy_version": "query-authz.sql.v1",
        "entitlement_version": "entitlement.member.test.v1",
    }
    scope = AuthorizedRetrievalScope(
        **values,
        scope_hash=compute_scope_hash(**values),
    )
    config_snapshot = dict(frozen.config_snapshot)
    config_snapshot["authorized_scope"] = scope.model_dump(mode="json")
    index_snapshot = dict(frozen.index_snapshot)
    index_snapshot["scope_hash"] = scope.scope_hash
    return runtime_routes._FrozenExecution(
        scope=scope,
        profile=frozen.profile,
        index_snapshot=index_snapshot,
        config_snapshot=config_snapshot,
        execution_query=frozen.execution_query,
    )


def test_query_runtime_accept_dedupe_events_cancel_and_replay(
    tmp_path,
    monkeypatch,
) -> None:
    engine = configure_database(f"sqlite:///{(tmp_path / 'query-runtime-api.db').as_posix()}")
    Base.metadata.create_all(engine)
    with session_scope() as session:
        session.add(
            TenantModel(
                id="tenant_runtime",
                name="Runtime tenant",
                status="active",
            )
        )
        session.add(
            UserModel(
                id="user_runtime",
                email="runtime@example.test",
                display_name="Runtime user",
                password_hash="not-a-real-password-hash",
                status="active",
                token_version=1,
            )
        )
        session.flush()
        session.add(
            TenantMembershipModel(
                id="membership_runtime",
                tenant_id="tenant_runtime",
                user_id="user_runtime",
                role="TenantAdmin",
                status="active",
            )
        )
    principal = Principal(
        user_id="user_runtime",
        tenant_id="tenant_runtime",
        roles=("TenantAdmin",),
        scopes=(),
        auth_type="password",
        credential_id="credential_runtime",
        request_id="req_runtime",
    )
    app = FastAPI()

    @app.middleware("http")
    async def inject_principal(request: Request, call_next):
        request.state.request_id = "req_runtime"
        request.state.principal = principal
        return await call_next(request)

    runtime_routes.register_query_runtime_routes(app)
    p3_routes.register_p3_routes(app)
    monkeypatch.setattr(
        runtime_routes,
        "settings",
        runtime_routes.settings.model_copy(
            update={
                "query_checkpoint_encryption_key": SecretStr(
                    base64.urlsafe_b64encode(b"q" * 32).decode("ascii")
                )
            }
        ),
    )
    monkeypatch.setattr(runtime_routes, "_freeze_execution", lambda **kw: _frozen(kw["payload"]))
    executions: list[str] = []
    seed_checkpoints: list[int] = []

    def fake_execute(query_run_id: str, worker_principal: Principal, query: str) -> None:
        with session_scope() as session:
            execution = session.get(QueryRunExecutionModel, query_run_id)
            checkpoint = session.scalar(
                select(QueryRunCheckpointModel).where(
                    QueryRunCheckpointModel.query_run_id == query_run_id,
                    QueryRunCheckpointModel.sequence == 0,
                )
            )
            assert execution is not None
            assert checkpoint is not None
            seed_checkpoints.append(checkpoint.sequence)
        executions.append(query_run_id)
        if query == "hold":
            return
        with session_scope() as session:
            repository = QueryRuntimeRepository(
                session,
                tenant_id=worker_principal.tenant_id,
                user_id=worker_principal.user_id,
            )
            history = repository.get_history_snapshot(query_run_id)
            run = repository.get_run(query_run_id)
            for target in (
                QueryRunStatus.PLANNING,
                QueryRunStatus.RETRIEVING,
                QueryRunStatus.PACKING,
                QueryRunStatus.GENERATING,
                QueryRunStatus.VALIDATING,
            ):
                run = repository.transition_run(
                    query_run_id,
                    expected_revision=run.revision,
                    target=target,
                )
            repository.append_event(
                query_run_id,
                event_type=QueryEventType.ANSWER_DELTA,
                data={"operation": "replace", "text": "测试答案"},
            )
            run = repository.complete_run(
                query_run_id,
                expected_revision=run.revision,
                answer="测试答案",
                result={"answer": "测试答案"},
            )
            fingerprint = _fingerprint()
            QueryReliabilityRepository(
                session,
                tenant_id=worker_principal.tenant_id,
            ).save_replay_manifest(
                QueryReplayManifest(
                    schema_version=runtime_routes._REPLAY_MANIFEST_SCHEMA_VERSION,
                    query_run_id=query_run_id,
                    fingerprint=fingerprint,
                    request_hash=canonical_hash({"query": query}),
                    history_snapshot_ref=history.content_hash,
                    query_plan={"disposition": "test"},
                    route_result_refs=(
                        canonical_hash(
                            {
                                "query_plan": {"disposition": "test"},
                                "query_understanding": None,
                                "sources": None,
                            }
                        ),
                        runtime_routes._ReplayProofCheckpointBinding(
                            query_run_id=query_run_id,
                            sequence=1,
                            ciphertext_sha256=canonical_hash("checkpoint"),
                            proof_projection_sha256=canonical_hash("proof"),
                        ).to_ref(),
                    ),
                    answer_envelope={"answer": "测试答案"},
                    event_stream_hash=canonical_hash(["completed"]),
                ),
                ttl=timedelta(days=1),
            )

    monkeypatch.setattr(runtime_routes, "_execute_query_run", fake_execute)
    monkeypatch.setattr(
        runtime_routes,
        "_restore_replay_proof_state_or_error",
        lambda *, session, principal, manifest: {"answer_envelope": dict(manifest.answer_envelope)},
    )
    # 本用例只验证可靠运行协议，数据库未构造控制面 KB entitlement；历史 API
    # 的实时撤权语义由独立 API 测试覆盖。
    monkeypatch.setattr(
        runtime_routes,
        "_authorized_conversation_kb_ids",
        lambda *, principal, candidate_kb_ids: candidate_kb_ids,
    )

    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/query-runs",
            headers={"Idempotency-Key": "same-request"},
            json={"query": "测试问题", "kb_ids": ["kb_runtime"]},
        )
        assert accepted.status_code == 202
        accepted_body = accepted.json()
        assert accepted_body["created"] is True
        query_run_id = accepted_body["query_run_id"]
        assert seed_checkpoints == [0]

        duplicate = client.post(
            "/api/v1/query-runs",
            headers={"Idempotency-Key": "same-request"},
            json={"query": "测试问题", "kb_ids": ["kb_runtime"]},
        )
        assert duplicate.status_code == 202
        assert duplicate.json()["query_run_id"] == query_run_id
        assert duplicate.json()["created"] is False
        assert executions == [query_run_id]

        run = client.get(f"/api/v1/query-runs/{query_run_id}")
        assert run.status_code == 200
        assert run.json()["status"] == "completed"
        events = client.get(f"/api/v1/query-runs/{query_run_id}/events")
        assert events.status_code == 200
        event_rows = events.json()["page"]["events"]
        event_types = [item["type"] for item in event_rows]
        assert event_types[0] == "run.accepted"
        assert event_types[-1] == "run.completed"
        stream = client.get(
            f"/api/v1/query-runs/{query_run_id}/events/stream",
            params={"after_seq": event_rows[1]["seq"]},
        )
        assert stream.status_code == 200
        assert "event: stream.ready" in stream.text
        assert f"id: {event_rows[0]['event_id']}" not in stream.text
        assert "event: run.completed" in stream.text
        result = client.get(f"/api/v1/query-runs/{query_run_id}/result")
        assert result.json()["result"]["answer"] == "测试答案"
        observability = client.get(f"/api/v1/p3/query-runs/{query_run_id}/observability")
        assert observability.status_code == 200
        observation = observability.json()["data"]
        assert observation["query_run_id"] == query_run_id
        assert observation["status"] == "completed"
        assert observation["safety"]["query_returned"] is False
        assert observation["answer_facts"]["answer_body_returned"] is False

        manifest = client.get(f"/api/v1/query-runs/{query_run_id}/replay")
        assert manifest.status_code == 200
        assert manifest.json()["manifest"]["query_run_id"] == query_run_id
        replay = client.post(
            f"/api/v1/query-runs/{query_run_id}:replay",
            json={"mode": "fixed_context"},
        )
        assert replay.status_code == 200
        replay_result = replay.json()["result"]
        assert replay_result["replay_of"] == query_run_id
        assert replay_result["committed_to_conversation"] is False
        assert replay_result["external_side_effects"] is False

        held = client.post(
            "/api/v1/query-runs",
            headers={"Idempotency-Key": "held-request"},
            json={"query": "hold", "kb_ids": ["kb_runtime"]},
        )
        held_id = held.json()["query_run_id"]
        cancelled = client.post(
            f"/api/v1/query-runs/{held_id}:cancel",
            json={},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        held_events = client.get(f"/api/v1/query-runs/{held_id}/events")
        assert held_events.json()["page"]["events"][-1]["type"] == "run.cancelled"
        retried = client.post(
            f"/api/v1/query-runs/{held_id}:retry",
            headers={"Idempotency-Key": "held-retry"},
            json={},
        )
        assert retried.status_code == 202
        assert retried.json()["query_run_id"] != held_id
        assert retried.json()["attempt"] == 2

        conversations = client.get("/api/v1/conversations")
        assert conversations.status_code == 200
        assert len(conversations.json()["items"]) == 2

    dispose_database()


@pytest.mark.parametrize(
    "configured_key",
    [None, SecretStr(""), SecretStr("not-url-safe-base64!")],
    ids=["missing", "empty", "invalid"],
)
def test_checkpoint_key_fails_before_query_run_creation_or_provider(
    tmp_path,
    monkeypatch,
    configured_key: SecretStr | None,
) -> None:
    engine = configure_database(f"sqlite:///{(tmp_path / 'query-key-fail-closed.db').as_posix()}")
    Base.metadata.create_all(engine)
    principal = Principal(
        user_id="user_runtime",
        tenant_id="tenant_runtime",
        roles=("TenantAdmin",),
        scopes=(),
        auth_type="password",
        credential_id="credential_runtime",
        request_id="req_runtime",
    )
    app = FastAPI()

    @app.middleware("http")
    async def inject_principal(request: Request, call_next):
        request.state.request_id = "req_runtime"
        request.state.principal = principal
        return await call_next(request)

    runtime_routes.register_query_runtime_routes(app)
    monkeypatch.setattr(
        runtime_routes,
        "settings",
        runtime_routes.settings.model_copy(
            update={"query_checkpoint_encryption_key": configured_key}
        ),
    )
    provider_calls: list[str] = []
    monkeypatch.setattr(
        runtime_routes,
        "_execute_query_run",
        lambda query_run_id, _principal, _query: provider_calls.append(query_run_id),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query-runs",
            json={"query": "不得落库", "kb_ids": ["kb_runtime"]},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "QUERY_CHECKPOINT_KEY_UNAVAILABLE"
    assert provider_calls == []
    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(QueryRunModel)) == 0
        assert session.scalar(select(func.count()).select_from(QueryRunExecutionModel)) == 0
        assert session.scalar(select(func.count()).select_from(QueryRunCheckpointModel)) == 0
    dispose_database()


def test_revoked_kb_acl_stops_resume_before_executor_or_provider(
    tmp_path,
    monkeypatch,
) -> None:
    engine = configure_database(f"sqlite:///{(tmp_path / 'query-acl-revoked.db').as_posix()}")
    Base.metadata.create_all(engine)
    with session_scope() as session:
        session.add(TenantModel(id="tenant_runtime", name="Runtime", status="active"))
        session.add(
            UserModel(
                id="user_runtime",
                email="runtime-acl@example.test",
                display_name="Runtime ACL user",
                password_hash="not-a-real-password-hash",
                status="active",
                token_version=1,
            )
        )
        session.flush()
        session.add(
            WorkspaceModel(
                id="workspace_runtime",
                tenant_id="tenant_runtime",
                name="Runtime workspace",
                status="active",
            )
        )
        session.add(
            TenantMembershipModel(
                id="membership_runtime",
                tenant_id="tenant_runtime",
                user_id="user_runtime",
                role="Member",
                status="active",
            )
        )
        session.flush()
        session.add(
            KnowledgeBaseModel(
                id="kb_runtime",
                tenant_id="tenant_runtime",
                workspace_id="workspace_runtime",
                name="Runtime KB",
                description="",
                status="active",
                created_by="another_user",
            )
        )
        session.flush()
        session.add(
            KnowledgeBaseAclModel(
                id="acl_runtime",
                kb_id="kb_runtime",
                principal_type="user",
                principal_id="user_runtime",
                role="query",
            )
        )
    principal = Principal(
        user_id="user_runtime",
        tenant_id="tenant_runtime",
        roles=("Member",),
        scopes=(),
        auth_type="password",
        credential_id="credential_runtime",
        request_id="req_runtime",
    )
    app = FastAPI()

    @app.middleware("http")
    async def inject_principal(request: Request, call_next):
        request.state.request_id = "req_runtime"
        request.state.principal = principal
        return await call_next(request)

    runtime_routes.register_query_runtime_routes(app)
    monkeypatch.setattr(
        runtime_routes,
        "settings",
        runtime_routes.settings.model_copy(
            update={
                "query_checkpoint_encryption_key": SecretStr(
                    base64.urlsafe_b64encode(b"a" * 32).decode("ascii")
                )
            }
        ),
    )
    monkeypatch.setattr(
        runtime_routes,
        "_freeze_execution",
        lambda **kw: _member_frozen(kw["payload"]),
    )
    execute_query_run = runtime_routes._execute_query_run
    monkeypatch.setattr(runtime_routes, "_execute_query_run", lambda *_args: None)
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/query-runs",
            headers={"Idempotency-Key": "acl-revoke"},
            json={"query": "ACL 撤销测试", "kb_ids": ["kb_runtime"]},
        )
    assert accepted.status_code == 202
    query_run_id = accepted.json()["query_run_id"]
    with session_scope() as session:
        acl = session.get(KnowledgeBaseAclModel, "acl_runtime")
        assert acl is not None
        session.delete(acl)

    executor_builds: list[str] = []

    def forbidden_executor(**_kwargs):
        executor_builds.append("called")
        raise AssertionError("revoked ACL must stop before executor/provider")

    monkeypatch.setattr(
        runtime_routes,
        "build_durable_query_executor",
        forbidden_executor,
    )
    execute_query_run(query_run_id, principal, None)
    assert executor_builds == []
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
        assert run.error_code == "QUERY_RESUME_SCOPE_REVOKED"
        assert execution.status.value == "failed"
        assert idempotency is not None
        assert idempotency.response_status == 403
    dispose_database()


@pytest.mark.parametrize(
    ("revoked_authority", "expected_code"),
    [
        ("credential", "QUERY_RESUME_CREDENTIAL_REVOKED"),
        ("membership", "QUERY_RESUME_IDENTITY_REVOKED"),
    ],
)
def test_accept_then_identity_revoke_stops_background_worker_before_executor(
    tmp_path,
    monkeypatch,
    revoked_authority: str,
    expected_code: str,
) -> None:
    engine = configure_database(
        f"sqlite:///{(tmp_path / f'query-{revoked_authority}-revoked.db').as_posix()}"
    )
    Base.metadata.create_all(engine)
    with session_scope() as session:
        session.add(TenantModel(id="tenant_runtime", name="Runtime", status="active"))
        session.add(
            UserModel(
                id="user_runtime",
                email=f"runtime-{revoked_authority}@example.test",
                display_name="Runtime revoked user",
                password_hash="not-a-real-password-hash",
                status="active",
                token_version=1,
            )
        )
        session.flush()
        session.add(
            TenantMembershipModel(
                id="membership_runtime",
                tenant_id="tenant_runtime",
                user_id="user_runtime",
                role="Member",
                status="active",
            )
        )
        session.add(
            ApiCredentialModel(
                id="api_runtime",
                tenant_id="tenant_runtime",
                user_id="user_runtime",
                name="Runtime API",
                prefix="stt_runtime",
                secret_hash="0" * 64,
                scopes=["kb:kb_runtime:read"],
                status="active",
            )
        )
    principal = Principal(
        user_id="user_runtime",
        tenant_id="tenant_runtime",
        roles=("ServiceAccount",),
        scopes=("kb:kb_runtime:read",),
        auth_type="api_key",
        credential_id="api_runtime",
        request_id="req_runtime",
    )
    app = FastAPI()

    @app.middleware("http")
    async def inject_principal(request: Request, call_next):
        request.state.request_id = "req_runtime"
        request.state.principal = principal
        return await call_next(request)

    runtime_routes.register_query_runtime_routes(app)
    monkeypatch.setattr(
        runtime_routes,
        "settings",
        runtime_routes.settings.model_copy(
            update={
                "query_checkpoint_encryption_key": SecretStr(
                    base64.urlsafe_b64encode(b"i" * 32).decode("ascii")
                )
            }
        ),
    )
    monkeypatch.setattr(
        runtime_routes,
        "_freeze_execution",
        lambda **kw: _frozen(kw["payload"]),
    )
    execute_query_run = runtime_routes._execute_query_run
    monkeypatch.setattr(runtime_routes, "_execute_query_run", lambda *_args: None)
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/query-runs",
            headers={"Idempotency-Key": f"{revoked_authority}-revoke"},
            json={"query": "身份撤销测试", "kb_ids": ["kb_runtime"]},
        )
    assert accepted.status_code == 202
    query_run_id = accepted.json()["query_run_id"]
    with session_scope() as session:
        if revoked_authority == "credential":
            credential = session.get(ApiCredentialModel, "api_runtime")
            assert credential is not None
            credential.status = "revoked"
        else:
            membership = session.get(TenantMembershipModel, "membership_runtime")
            assert membership is not None
            membership.status = "revoked"

    executor_builds: list[str] = []

    def forbidden_executor(**_kwargs):
        executor_builds.append("called")
        raise AssertionError("revoked identity must stop before executor/provider")

    monkeypatch.setattr(
        runtime_routes,
        "build_durable_query_executor",
        forbidden_executor,
    )
    execute_query_run(query_run_id, principal, None)
    assert executor_builds == []
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
        assert run.error_code == expected_code
        assert execution.status.value == "failed"
        assert idempotency is not None
        assert idempotency.response_status == 403
    dispose_database()


def test_runtime_has_one_durable_executor_entry() -> None:
    source = inspect.getsource(runtime_routes._execute_query_run)
    assert source.count("executor.execute(") == 1
    assert source.index("executor.preflight(") < source.index("_cache_or_singleflight(")


def test_proof_v2_cache_bypass_happens_before_any_repository_lookup(monkeypatch) -> None:
    def forbidden_session_scope():
        raise AssertionError("proof v2 cache bypass must not open a repository session")

    monkeypatch.setattr(runtime_routes, "session_scope", forbidden_session_scope)

    cached, role, lease = runtime_routes._cache_or_singleflight(
        principal=object(),
        run=object(),
        fingerprint=object(),
    )

    assert cached is None
    assert role == "cache_bypass_answer_proof_identity"
    assert lease is None
