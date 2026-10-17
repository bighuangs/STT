from __future__ import annotations

import pytest

import app.api.http.query_runtime_routes as runtime_routes
from app.domain.retrieval import RetrievalEngineError, RetrievalScope
from app.domain.query_security import (
    ActiveIndexBinding,
    AuthorizedKnowledgeBase,
    AuthorizedRetrievalScope,
    compute_scope_hash,
)
from app.infra.retrieval.control_plane_resolvers import (
    FrozenActiveScopeResolver,
    active_scope_digest,
    freeze_vector_read_set,
    serialize_active_scopes,
)
from app.infra.retrieval.milvus_adapter import ActiveIndexScope


def _scopes() -> tuple[ActiveIndexScope, ...]:
    return (
        ActiveIndexScope(
            tenant_id="tenant_a",
            kb_id="kb_a",
            document_id="doc_a",
            document_version_id="dv_a",
            index_version_id="idx_a",
            collection_name="physical_source",
            partition_name="p_07",
        ),
    )


def _request_scope(*, tenant_id: str = "tenant_a", kb_id: str = "kb_a") -> RetrievalScope:
    return RetrievalScope(
        tenant_id=tenant_id,
        user_id="user_a",
        kb_ids=(kb_id,),
    )


def _authorized_scope() -> AuthorizedRetrievalScope:
    values = {
        "tenant_id": "tenant_a",
        "principal_id": "user_a",
        "workspace_id": "workspace_a",
        "resources": (
            AuthorizedKnowledgeBase(
                kb_id="kb_a",
                document_ids=None,
                grant_source="tenant_admin",
                grant_role="admin",
            ),
        ),
        "business_filter": None,
        "active_indexes": (
            ActiveIndexBinding(
                kb_id="kb_a",
                document_id="doc_a",
                document_version_id="dv_a",
                index_version_id="idx_a",
            ),
        ),
        "policy_version": "query-authz.test.v1",
        "entitlement_version": "entitlement.test.v1",
    }
    return AuthorizedRetrievalScope(
        **values,
        scope_hash=compute_scope_hash(**values),
    )


def test_frozen_scope_replays_exact_collection_and_partition() -> None:
    scopes = _scopes()
    resolver = FrozenActiveScopeResolver(
        serialize_active_scopes(scopes),
        expected_digest=active_scope_digest(list(scopes)),
    )

    resolved = resolver.resolve(_request_scope())

    assert resolved == scopes
    assert resolved[0].collection_name == "physical_source"
    assert resolved[0].partition_name == "p_07"


def test_frozen_scope_rejects_tampered_physical_target() -> None:
    scopes = _scopes()
    rows = serialize_active_scopes(scopes)
    rows[0]["collection_name"] = "attacker_collection"

    with pytest.raises(RetrievalEngineError, match="摘要不匹配") as exc_info:
        FrozenActiveScopeResolver(
            rows,
            expected_digest=active_scope_digest(list(scopes)),
        )

    assert exc_info.value.code == "RETRIEVAL_STALE_SCOPE"
    assert exc_info.value.retryable is False


def test_frozen_scope_rejects_runtime_scope_switch() -> None:
    scopes = _scopes()
    resolver = FrozenActiveScopeResolver(
        serialize_active_scopes(scopes),
        expected_digest=active_scope_digest(list(scopes)),
    )

    with pytest.raises(RetrievalEngineError, match="认证检索范围不一致"):
        resolver.resolve(_request_scope(kb_id="kb_other"))


def test_frozen_scope_rejects_duplicate_logical_binding() -> None:
    scopes = _scopes()
    rows = serialize_active_scopes(scopes)
    rows.append({**rows[0], "collection_name": "physical_other"})

    with pytest.raises(RetrievalEngineError, match="重复逻辑索引身份"):
        FrozenActiveScopeResolver(
            rows,
            expected_digest=active_scope_digest(
                [ActiveIndexScope.model_validate(item) for item in rows]
            ),
        )


def test_query_run_acceptance_freezes_resolved_physical_scope(monkeypatch) -> None:
    class FakeSqlResolver:
        def resolve(self, request_scope):
            assert request_scope == _request_scope()
            return _scopes()

    monkeypatch.setattr(
        "app.infra.retrieval.control_plane_resolvers.SqlActiveScopeResolver",
        FakeSqlResolver,
    )

    rows, digest = runtime_routes._freeze_vector_read_set(_authorized_scope())

    assert rows == serialize_active_scopes(_scopes())
    assert digest == active_scope_digest(list(_scopes()))
    runtime_routes._validate_frozen_vector_read_set(
        scope=_authorized_scope(),
        rows=rows,
        digest=digest,
    )


def test_query_run_acceptance_rejects_logically_different_resolver_result(
    monkeypatch,
) -> None:
    class FakeSqlResolver:
        def resolve(self, _request_scope):
            return (_scopes()[0].model_copy(update={"document_version_id": "dv_changed"}),)

    monkeypatch.setattr(
        "app.infra.retrieval.control_plane_resolvers.SqlActiveScopeResolver",
        FakeSqlResolver,
    )

    with pytest.raises(RetrievalEngineError, match="活动索引不一致"):
        runtime_routes._freeze_vector_read_set(_authorized_scope())


def test_shared_freeze_resolves_and_serializes_the_exact_physical_scope(
    monkeypatch,
) -> None:
    class FakeSqlResolver:
        def resolve(self, request_scope):
            assert request_scope == _request_scope()
            return _scopes()

    monkeypatch.setattr(
        "app.infra.retrieval.control_plane_resolvers.SqlActiveScopeResolver",
        FakeSqlResolver,
    )

    resolved, rows, digest = freeze_vector_read_set(_request_scope())

    assert resolved == _scopes()
    assert rows == serialize_active_scopes(_scopes())
    assert digest == active_scope_digest(list(_scopes()))


def test_shared_freeze_rejects_an_empty_active_scope(monkeypatch) -> None:
    class EmptySqlResolver:
        def resolve(self, _request_scope):
            return ()

    monkeypatch.setattr(
        "app.infra.retrieval.control_plane_resolvers.SqlActiveScopeResolver",
        EmptySqlResolver,
    )

    with pytest.raises(RetrievalEngineError, match="没有可冻结") as exc_info:
        freeze_vector_read_set(_request_scope())

    assert exc_info.value.code == "RETRIEVAL_STALE_SCOPE"
    assert exc_info.value.retryable is False
