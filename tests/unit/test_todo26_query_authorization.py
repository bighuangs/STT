"""TODO-26 授权范围、传播与外发策略的离线合同测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.query_authorization_service import (
    ScopePropagationGuard,
    compile_milvus_authorized_scope,
    redact_external_payload,
    scope_cache_fragment,
)
from app.domain.query_security import (
    ActiveIndexBinding,
    DataClassification,
    EgressPolicySnapshot,
    EntitledKnowledgeBase,
    EntitlementSnapshot,
    ExternalDataRequest,
    RequestedDocument,
    RequestedRetrievalScope,
    ScopeAuthorizationError,
    authorize_retrieval_scope,
    decide_external_data,
)
from app.domain.retrieval import FilterPredicate


def _entitlement() -> EntitlementSnapshot:
    return EntitlementSnapshot(
        tenant_id="tenant_a",
        principal_id="user_a",
        workspace_id="workspace_a",
        grants=(
            EntitledKnowledgeBase(
                kb_id="kb_a",
                document_ids=None,
                grant_source="owner",
                grant_role="owner",
            ),
            EntitledKnowledgeBase(
                kb_id="kb_limited",
                document_ids=("doc_1", "doc_2"),
                grant_source="user_acl",
                grant_role="reader",
            ),
        ),
        policy_version="authz.v7",
        entitlement_version="ent.v12",
    )


def _index(
    kb_id: str,
    document_id: str,
    suffix: str,
) -> ActiveIndexBinding:
    return ActiveIndexBinding(
        kb_id=kb_id,
        document_id=document_id,
        document_version_id=f"dv_{suffix}",
        index_version_id=f"idx_{suffix}",
    )


def test_authorized_scope_is_intersection_and_never_broadens_requested_kbs() -> None:
    requested = RequestedRetrievalScope(
        kb_ids=("kb_a", "kb_denied"),
        documents=(RequestedDocument(kb_id="kb_a", document_id="doc_a"),),
    )
    scope = authorize_retrieval_scope(
        requested,
        _entitlement(),
        active_indexes=(
            _index("kb_a", "doc_a", "a"),
            _index("kb_denied", "doc_secret", "secret"),
        ),
    )

    assert scope.allowed_kb_ids == ("kb_a",)
    assert scope.allowed_document_ids == ("doc_a",)
    assert len(scope.active_indexes) == 1
    assert scope.active_indexes[0].kb_id == "kb_a"
    assert scope.scope_hash.startswith("scope_")
    assert scope.audit_fields()["document_restricted"] is True
    assert "doc_a" not in scope.audit_fields().values()


def test_explicit_document_acl_is_intersected_and_empty_result_fails_closed() -> None:
    allowed = authorize_retrieval_scope(
        RequestedRetrievalScope(
            kb_ids=("kb_limited",),
            documents=(
                RequestedDocument(kb_id="kb_limited", document_id="doc_2"),
                RequestedDocument(kb_id="kb_limited", document_id="doc_denied"),
            ),
        ),
        _entitlement(),
    )
    assert allowed.allowed_document_ids == ("doc_2",)

    with pytest.raises(ScopeAuthorizationError):
        authorize_retrieval_scope(
            RequestedRetrievalScope(
                kb_ids=("kb_limited",),
                documents=(
                    RequestedDocument(
                        kb_id="kb_limited",
                        document_id="doc_denied",
                    ),
                ),
            ),
            _entitlement(),
        )


def test_missing_empty_duplicate_or_cross_kb_scope_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RequestedRetrievalScope(kb_ids=())
    with pytest.raises(ValidationError):
        RequestedRetrievalScope(kb_ids=("kb_a", "kb_a"))
    with pytest.raises(ValidationError):
        RequestedRetrievalScope(
            kb_ids=("kb_a",),
            documents=(RequestedDocument(kb_id="kb_other", document_id="doc_a"),),
        )
    with pytest.raises(ScopeAuthorizationError):
        authorize_retrieval_scope(
            RequestedRetrievalScope(kb_ids=("kb_unknown",)),
            _entitlement(),
        )


def test_scope_hash_is_stable_and_tampering_is_rejected() -> None:
    requested = RequestedRetrievalScope(kb_ids=("kb_a",))
    first = authorize_retrieval_scope(requested, _entitlement())
    second = authorize_retrieval_scope(requested, _entitlement())
    assert first.scope_hash == second.scope_hash
    assert scope_cache_fragment(first) == scope_cache_fragment(second)

    with pytest.raises(ValidationError, match="scope_hash"):
        first.model_copy(update={"scope_hash": "scope_" + ("0" * 64)}).__class__(
            **{
                **first.model_dump(mode="python"),
                "scope_hash": "scope_" + ("0" * 64),
            }
        )


def test_hit_recheck_requires_tenant_resource_and_frozen_versions() -> None:
    scope = authorize_retrieval_scope(
        RequestedRetrievalScope(kb_ids=("kb_a",)),
        _entitlement(),
        active_indexes=(_index("kb_a", "doc_a", "a"),),
    )
    assert scope.permits(
        tenant_id="tenant_a",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_a",
        index_version_id="idx_a",
    )
    assert not scope.permits(
        tenant_id="tenant_b",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_a",
        index_version_id="idx_a",
    )
    assert scope.mandatory_filter_ast["conditions"][0]["predicates"][-1] == {
        "field": "available",
        "operator": "eq",
        "value": True,
    }
    expression = compile_milvus_authorized_scope(scope)
    assert 'tenant_id == "tenant_a"' in expression
    assert 'index_version_id == "idx_a"' in expression
    assert "available_int == 1" in expression


def test_business_filter_must_be_resolved_and_can_only_narrow_compiled_scope() -> None:
    scope = authorize_retrieval_scope(
        RequestedRetrievalScope(
            kb_ids=("kb_a",),
            business_filter=FilterPredicate(
                field="metadata.category",
                operator="eq",
                value="manual",
            ),
        ),
        _entitlement(),
        active_indexes=(
            _index("kb_a", "doc_a", "a"),
            _index("kb_a", "doc_b", "b"),
        ),
    )
    with pytest.raises(ScopeAuthorizationError, match="not been resolved"):
        compile_milvus_authorized_scope(scope)
    expression = compile_milvus_authorized_scope(
        scope,
        resolved_business_document_ids=("doc_b", "doc_outside"),
    )
    assert '"doc_b"' in expression
    assert '"doc_a"' not in expression
    assert "doc_outside" not in expression
    assert not scope.permits(
        tenant_id="tenant_a",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_old",
        index_version_id="idx_a",
    )


def test_scope_propagation_guard_rejects_missing_or_replaced_scope() -> None:
    scope = authorize_retrieval_scope(
        RequestedRetrievalScope(kb_ids=("kb_a",)),
        _entitlement(),
    )
    guard = ScopePropagationGuard(run_id="run_a", authorized_scope=scope)
    guard.consume("dense", scope)
    guard.consume("citation", scope)
    guard.assert_required_stages(stages={"dense", "citation"})

    with pytest.raises(ScopeAuthorizationError, match="requires authorized scope"):
        guard.consume("parent", None)
    with pytest.raises(ScopeAuthorizationError, match="not consumed"):
        guard.assert_required_stages(stages={"dense", "sparse"})

    other_entitlement = _entitlement().model_copy(update={"entitlement_version": "ent.v13"})
    other = authorize_retrieval_scope(
        RequestedRetrievalScope(kb_ids=("kb_a",)),
        other_entitlement,
    )
    with pytest.raises(ScopeAuthorizationError, match="replace or broaden"):
        guard.consume("rerank", other)


def _egress_request(**updates) -> ExternalDataRequest:
    payload = {
        "requested": True,
        "provider": "approved-llm",
        "purpose": "llm_generation",
        "scope_hash": "scope_" + ("a" * 64),
        "data_classes": (DataClassification.INTERNAL,),
    }
    payload.update(updates)
    return ExternalDataRequest(**payload)


def _egress_policy(**updates) -> EgressPolicySnapshot:
    payload = {
        "policy_version": "egress.v9",
        "enabled": True,
        "allowed_providers": ("approved-llm",),
        "allowed_purposes": ("llm_generation",),
        "allowed_domains": ("api.example.com",),
        "max_classification": DataClassification.INTERNAL,
        "retention_policy": "zero-retention",
    }
    payload.update(updates)
    return EgressPolicySnapshot(**payload)


def test_external_data_is_default_deny_and_policy_is_all_conditions() -> None:
    assert decide_external_data(_egress_request(), None).reason_code == "EGRESS_DISABLED"
    assert (
        decide_external_data(
            _egress_request(provider="unknown"),
            _egress_policy(),
        ).reason_code
        == "EGRESS_PROVIDER_DENIED"
    )
    assert (
        decide_external_data(
            _egress_request(data_classes=(DataClassification.RESTRICTED,)),
            _egress_policy(),
        ).reason_code
        == "EGRESS_CLASSIFICATION_DENIED"
    )
    assert (
        decide_external_data(
            _egress_request(contains_secret=True),
            _egress_policy(),
        ).reason_code
        == "EGRESS_SECRET_BLOCKED"
    )
    assert (
        decide_external_data(
            _egress_request(target_domain="evil.example"),
            _egress_policy(),
        ).reason_code
        == "EGRESS_DOMAIN_DENIED"
    )


def test_external_data_allows_only_versioned_policy_and_marks_redaction() -> None:
    decision = decide_external_data(
        _egress_request(
            target_domain="api.example.com",
            contains_direct_identifier=True,
        ),
        _egress_policy(),
    )
    assert decision.allowed is True
    assert decision.reason_code == "EGRESS_ALLOWED"
    assert decision.redactions == ("direct_identifier",)
    assert decision.policy_version == "egress.v9"
    assert decision.decision_id.startswith("egd_")
    assert decision.scope_hash == "scope_" + ("a" * 64)
    sanitized = redact_external_payload(
        "联系 owner@example.com 或 13800138000，api_key=top-secret-value",
        decision,
    )
    assert "owner@example.com" not in sanitized
    assert "13800138000" not in sanitized
    assert "top-secret-value" not in sanitized

    denied = decide_external_data(_egress_request(), None)
    with pytest.raises(PermissionError, match="EGRESS_DISABLED"):
        redact_external_payload("must not leave process", denied)
