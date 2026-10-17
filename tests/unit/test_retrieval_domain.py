from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.domain.retrieval import (
    DenseSearchConfig,
    FilterGroup,
    FilterPredicate,
    MetadataFilterPolicy,
    RetrievalProfile,
    RetrievalRequest,
    RetrievalScope,
    default_retrieval_profile,
    iter_filter_predicates,
)


def _scope() -> RetrievalScope:
    return RetrievalScope(
        tenant_id="tenant-a",
        user_id="user-a",
        kb_ids=("kb-a",),
    )


def test_default_profile_preserves_the_pre_todo09_retrieval_behavior():
    profile = default_retrieval_profile("tenant-a")

    assert profile.status == "published"
    assert profile.routes == ("hybrid",)
    assert profile.candidate_k == 5
    assert profile.top_k == 5
    assert profile.dense.metric == "IP"
    assert profile.dense.weight == pytest.approx(0.9)
    assert profile.sparse.weight == pytest.approx(0.1)
    assert profile.timeout_ms == 90_000
    assert profile.rrf.k == 60
    assert profile.rrf.weight_for("base") == pytest.approx(1.0)
    assert profile.rrf.weight_for("hyde") == pytest.approx(1.0)
    assert profile.reranker.min_top_k == 1
    assert profile.reranker.max_top_k == 10
    assert profile.reranker.max_input_tokens == 512


def test_candidate_window_is_independent_from_final_top_k():
    profile = RetrievalProfile(
        tenant_id="tenant-a",
        candidate_k=40,
        top_k=8,
    )
    assert profile.candidate_k == 40
    assert profile.top_k == 8

    with pytest.raises(ValidationError, match="candidate_k"):
        RetrievalProfile(tenant_id="tenant-a", candidate_k=4, top_k=5)


def test_published_profile_and_nested_configuration_are_frozen():
    profile = default_retrieval_profile("tenant-a")

    with pytest.raises(ValidationError):
        profile.top_k = 20  # type: ignore[misc]
    with pytest.raises(ValidationError):
        profile.dense.weight = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "tenant_id",
        "TENANT_ID",
        "kb_id",
        "index_version_id",
        "document_version_id",
        "available_int",
        "searchable",
        "collection_name",
    ],
)
def test_filter_ast_cannot_override_server_controlled_scope_fields(field: str):
    with pytest.raises(ValidationError, match="controlled by the server"):
        FilterPredicate(field=field, operator="eq", value="attacker")


@pytest.mark.parametrize(
    "field",
    [
        "metadata.department\") or tenant_id == \"tenant-b",
        "metadata.__class__",
        "metadata.",
        "doc_id",
        "title",
    ],
)
def test_filter_ast_only_accepts_structurally_safe_whitelisted_fields(field: str):
    with pytest.raises(ValidationError, match="document_id or a safe metadata"):
        FilterPredicate(field=field, operator="eq", value="x")


def test_filter_ast_accepts_document_and_metadata_predicates_in_input_order():
    root = FilterGroup(
        operator="and",
        conditions=(
            FilterPredicate(field="document_id", operator="in", value=("doc-1", "doc-2")),
            FilterGroup(
                operator="or",
                conditions=(
                    FilterPredicate(
                        field="metadata.department",
                        operator="eq",
                        value="研发",
                    ),
                    FilterPredicate(
                        field="metadata.pressure.max",
                        operator="gte",
                        value=10.5,
                    ),
                ),
            ),
        ),
    )

    assert [item.field for item in iter_filter_predicates(root)] == [
        "document_id",
        "metadata.department",
        "metadata.pressure.max",
    ]


def test_filter_ast_rejects_invalid_operator_value_shapes():
    with pytest.raises(ValidationError, match="requires an array"):
        FilterPredicate(field="document_id", operator="in", value="doc-1")
    with pytest.raises(ValidationError, match="requires a scalar"):
        FilterPredicate(
            field="metadata.department",
            operator="eq",
            value=("研发",),
        )
    with pytest.raises(ValidationError, match="requires a string"):
        FilterPredicate(field="metadata.department", operator="contains", value=7)
    with pytest.raises(ValidationError, match="boolean"):
        FilterPredicate(field="metadata.department", operator="exists", value="yes")


def test_filter_ast_enforces_global_depth_and_predicate_limits():
    node: dict = {
        "kind": "predicate",
        "field": "document_id",
        "operator": "eq",
        "value": "doc-1",
    }
    for _ in range(5):
        node = {"kind": "group", "operator": "not", "conditions": [node]}
    with pytest.raises(ValidationError, match="depth cannot exceed 5"):
        FilterGroup.model_validate(node)

    left = FilterGroup(
        conditions=tuple(
            FilterPredicate(
                field=f"metadata.left{i}",
                operator="eq",
                value=i,
            )
            for i in range(20)
        )
    )
    right = FilterGroup(
        conditions=tuple(
            FilterPredicate(
                field=f"metadata.right{i}",
                operator="eq",
                value=i,
            )
            for i in range(13)
        )
    )
    with pytest.raises(ValidationError, match="more than 32 predicates"):
        FilterGroup(conditions=(left, right))


def test_request_enforces_profile_tenant_and_published_online_version():
    draft = RetrievalProfile(tenant_id="tenant-a", status="draft")

    with pytest.raises(ValidationError, match="published profile"):
        RetrievalRequest(
            query="如何校准？",
            scope=_scope(),
            profile=draft,
            trace_id="trace-online",
        )

    preview = RetrievalRequest(
        query="如何校准？",
        scope=_scope(),
        profile=draft,
        execution_mode="preview",
        trace_id="trace-preview",
    )
    assert preview.execution_mode == "preview"

    with pytest.raises(ValidationError, match="tenant must match"):
        RetrievalRequest(
            query="如何校准？",
            scope=_scope(),
            profile=default_retrieval_profile("tenant-b"),
            trace_id="trace-cross-tenant",
        )


def test_request_applies_profile_specific_filter_limits_and_aware_deadline():
    profile = RetrievalProfile(
        tenant_id="tenant-a",
        metadata_filter=MetadataFilterPolicy(max_predicates=1),
    )
    root = FilterGroup(
        conditions=(
            FilterPredicate(field="metadata.a", operator="eq", value=1),
            FilterPredicate(field="metadata.b", operator="eq", value=2),
        )
    )
    with pytest.raises(ValidationError, match="more than 1 predicates"):
        RetrievalRequest(
            query="q",
            scope=_scope(),
            profile=profile,
            filters=root,
            trace_id="trace-filter-limit",
        )

    with pytest.raises(ValidationError, match="timezone"):
        RetrievalRequest(
            query="q",
            scope=_scope(),
            profile=profile,
            trace_id="trace-naive-deadline",
            deadline_at=datetime.now(),
        )

    request = RetrievalRequest(
        query="q",
        scope=_scope(),
        profile=profile,
        trace_id="trace-aware-deadline",
        deadline_at=datetime.now(UTC) + timedelta(seconds=10),
    )
    assert request.deadline_at is not None


def test_profile_rejects_incompatible_dense_dimension_and_empty_route_weight():
    with pytest.raises(ValidationError, match="positive dense"):
        RetrievalProfile(
            tenant_id="tenant-a",
            routes=("dense",),
            dense=DenseSearchConfig(dimension=4, weight=0),
        )
