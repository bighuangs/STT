from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.application.advanced_retrieval_service import (
    AdvancedIndexLifecycleService,
    AdvancedRetrievalError,
    AdvancedRetrievalService,
    validate_advanced_snapshot,
)
from app.domain.advanced_retrieval import (
    AdvancedEvaluationGate,
    AdvancedFeatureFlags,
    AdvancedIndexSnapshot,
    AdvancedIndexVersion,
    AdvancedRetrievalBudget,
    AdvancedRetrievalRequest,
    GraphCommunity,
    GraphEntity,
    GraphRelation,
    GraphSnapshot,
    RaptorSnapshot,
    RaptorSummaryNode,
    StructureNode,
    StructureSnapshot,
)
from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.infra.advanced_retrieval import InMemoryAdvancedIndexRepository


NOW = datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc)
TENANT = "tenant_default"
KB = "kb_manual"
INDEX = "idx_manual_v1"
DOCUMENT = "doc_hak180"
DOCUMENT_VERSION = "dv_hak180_v1"


def _evidence(
    chunk: str,
    content: str,
    *,
    index_version: str = INDEX,
    tenant: str = TENANT,
    kb: str = KB,
    document: str = DOCUMENT,
    document_version: str = DOCUMENT_VERSION,
) -> Evidence:
    block_id = f"block-{chunk}"
    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id=tenant,
            kb_id=kb,
            document_id=document,
            document_version_id=document_version,
            index_version_id=index_version,
            chunk_id=chunk,
            source_block_ids=[block_id],
        ),
        content=content,
        provenance_spans=[
            EvidenceLocationSpan(
                block_id=block_id,
                availability="exact",
                coordinate_space="normalized_page",
                page_no=1,
                bbox=(0.1, 0.1, 0.9, 0.3),
            )
        ],
        heading_path=["安全", "使用前准备"],
        source_label="hak180产品安全手册.pdf",
    )


def _catalog() -> dict[str, Evidence]:
    items = (
        _evidence("chunk_intro", "HAK180 产品安全手册使用前准备。"),
        _evidence("chunk_ground", "使用 HAK180 前必须检查保护接地是否可靠。"),
        _evidence("chunk_emergency", "启动前检查急停按钮是否有效。"),
    )
    return {item.evidence_id: item for item in items if item.evidence_id is not None}


def _gate(*, passed: bool = True) -> AdvancedEvaluationGate:
    return AdvancedEvaluationGate(
        evaluation_id="eval_hak180",
        holdout_dataset_id="holdout_hak180",
        passed=passed,
        quality_delta=0.12,
        citation_coverage_delta=0.08,
        p95_latency_ms=80,
        estimated_cost_per_query=0.01,
        estimated_storage_bytes=2048,
        budget_approved=passed,
        rollback_ready=passed,
        report_uri="reports/hak180.json" if passed else None,
    )


def _index(
    index_type: str,
    *,
    advanced_id: str,
    status: str = "active",
    source_index: str = INDEX,
    version: int = 1,
    passed: bool = True,
) -> AdvancedIndexVersion:
    return AdvancedIndexVersion(
        advanced_index_id=advanced_id,
        index_type=index_type,
        tenant_id=TENANT,
        kb_id=KB,
        scope="document" if index_type in {"toc", "raptor"} else "kb",
        document_id=DOCUMENT if index_type in {"toc", "raptor"} else None,
        version=version,
        source_index_version_id=source_index,
        source_document_version_ids=(DOCUMENT_VERSION,),
        status=status,
        evaluation=_gate(passed=passed),
        created_by="admin",
        created_at=NOW,
        published_at=NOW if status == "active" else None,
    )


def _structure_snapshot(
    catalog: dict[str, Evidence],
    *,
    status: str = "active",
    advanced_id: str = "adv_toc_1",
    version: int = 1,
) -> AdvancedIndexSnapshot:
    intro, ground, emergency = tuple(catalog.values())
    root = StructureNode(
        node_id="toc_prepare",
        node_type="toc",
        tenant_id=TENANT,
        kb_id=KB,
        document_id=DOCUMENT,
        document_version_id=DOCUMENT_VERSION,
        source_index_version_id=INDEX,
        title="使用前准备",
        toc_path=("使用前准备",),
        order=0,
        source_evidence_ids=(intro.evidence_id,),
        child_node_ids=("heading_safety",),
        next_node_id="heading_safety",
    )
    child = StructureNode(
        node_id="heading_safety",
        node_type="heading",
        tenant_id=TENANT,
        kb_id=KB,
        document_id=DOCUMENT,
        document_version_id=DOCUMENT_VERSION,
        source_index_version_id=INDEX,
        title="安全检查",
        toc_path=("使用前准备", "安全检查"),
        order=1,
        source_evidence_ids=(ground.evidence_id, emergency.evidence_id),
        parent_node_id=root.node_id,
        previous_node_id=root.node_id,
    )
    structure = StructureSnapshot(
        advanced_index_id=advanced_id,
        source_index_version_id=INDEX,
        nodes=(root, child),
    )
    return AdvancedIndexSnapshot(
        index=_index(
            "toc",
            advanced_id=advanced_id,
            status=status,
            version=version,
        ),
        evidence_ids=tuple(catalog),
        structure=structure,
    )


def _raptor_snapshot(
    catalog: dict[str, Evidence],
    *,
    status: str = "active",
    advanced_id: str = "adv_raptor_1",
) -> AdvancedIndexSnapshot:
    intro, ground, emergency = tuple(catalog.values())
    leaf = RaptorSummaryNode(
        summary_id="summary_leaf",
        level=1,
        text="HAK180 使用前需要安全检查。",
        source_evidence_ids=(ground.evidence_id, emergency.evidence_id),
        source_chunk_ids=("chunk_ground", "chunk_emergency"),
        cluster_method="deterministic",
        prompt_id="raptor_summary",
        prompt_version="v1",
        model_id="qwen",
    )
    root = RaptorSummaryNode(
        summary_id="summary_root",
        level=2,
        text="HAK180 使用前准备包括阅读手册、保护接地和急停检查。",
        source_evidence_ids=(
            intro.evidence_id,
            ground.evidence_id,
            emergency.evidence_id,
        ),
        source_chunk_ids=("chunk_intro", "chunk_ground", "chunk_emergency"),
        child_summary_ids=(leaf.summary_id,),
        cluster_method="deterministic",
        prompt_id="raptor_summary",
        prompt_version="v1",
        model_id="qwen",
    )
    raptor = RaptorSnapshot(
        advanced_index_id=advanced_id,
        tree_version="tree_v1",
        source_index_version_id=INDEX,
        summaries=(leaf, root),
    )
    return AdvancedIndexSnapshot(
        index=_index("raptor", advanced_id=advanced_id, status=status),
        evidence_ids=tuple(catalog),
        raptor=raptor,
    )


def _graph_snapshot(
    catalog: dict[str, Evidence],
    *,
    status: str = "active",
    advanced_id: str = "adv_graph_1",
) -> AdvancedIndexSnapshot:
    intro, ground, emergency = tuple(catalog.values())
    device = GraphEntity(
        entity_id="entity_hak180",
        canonical_name="HAK180",
        entity_type="device",
        aliases=("HAK 180",),
        source_evidence_ids=(intro.evidence_id, ground.evidence_id),
    )
    grounding = GraphEntity(
        entity_id="entity_grounding",
        canonical_name="保护接地",
        entity_type="safety_control",
        source_evidence_ids=(ground.evidence_id,),
    )
    relation = GraphRelation(
        relation_id="relation_requires_ground",
        source_entity_id=device.entity_id,
        target_entity_id=grounding.entity_id,
        relation_type="requires_before_use",
        statement="HAK180 使用前要求检查保护接地。",
        inferred=True,
        confidence=0.93,
        source_evidence_ids=(ground.evidence_id,),
    )
    community = GraphCommunity(
        community_id="community_preuse",
        title="使用前安全",
        summary="使用 HAK180 前需要接地与急停检查。",
        entity_ids=(device.entity_id, grounding.entity_id),
        relation_ids=(relation.relation_id,),
        source_evidence_ids=(ground.evidence_id, emergency.evidence_id),
        prompt_id="graph_community",
        prompt_version="v1",
        model_id="qwen",
    )
    graph = GraphSnapshot(
        advanced_index_id=advanced_id,
        graph_version="graph_v1",
        source_index_version_id=INDEX,
        entities=(device, grounding),
        relations=(relation,),
        communities=(community,),
    )
    return AdvancedIndexSnapshot(
        index=_index("graphrag", advanced_id=advanced_id, status=status),
        evidence_ids=tuple(catalog),
        graph=graph,
    )


def _request(
    routes: tuple[str, ...],
    *,
    flags: AdvancedFeatureFlags,
    active_source_index: str = INDEX,
    budget: AdvancedRetrievalBudget | None = None,
) -> AdvancedRetrievalRequest:
    return AdvancedRetrievalRequest(
        query="HAK180 使用前需要做哪些安全检查？",
        tenant_id=TENANT,
        kb_id=KB,
        active_source_index_version_id=active_source_index,
        trace_id="trace_hak180",
        requested_routes=routes,
        planner_reasons={route: "复杂手册问题" for route in routes},
        flags=flags,
        budget=budget or AdvancedRetrievalBudget(),
    )


def test_all_advanced_feature_flags_are_off_by_default() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(("toc", "raptor", "graphrag"), flags=AdvancedFeatureFlags()),
        snapshots=(
            _structure_snapshot(catalog),
            _raptor_snapshot(catalog),
            _graph_snapshot(catalog),
        ),
        evidence_by_id=catalog,
    )
    assert result.candidates == ()
    assert result.diagnostics.degraded_to_baseline
    assert all(item.reason == "FEATURE_DISABLED" for item in result.diagnostics.outcomes)


def test_structure_snapshot_rejects_dangling_and_asymmetric_edges() -> None:
    catalog = _catalog()
    evidence_id = next(iter(catalog))
    with pytest.raises(ValidationError, match="dangling"):
        StructureSnapshot(
            advanced_index_id="adv_toc",
            source_index_version_id=INDEX,
            nodes=(
                StructureNode(
                    node_id="root",
                    node_type="toc",
                    tenant_id=TENANT,
                    kb_id=KB,
                    document_id=DOCUMENT,
                    document_version_id=DOCUMENT_VERSION,
                    source_index_version_id=INDEX,
                    title="根",
                    toc_path=("根",),
                    order=0,
                    source_evidence_ids=(evidence_id,),
                    child_node_ids=("missing",),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="symmetric"):
        StructureSnapshot(
            advanced_index_id="adv_toc",
            source_index_version_id=INDEX,
            nodes=(
                StructureNode(
                    node_id="root",
                    node_type="toc",
                    tenant_id=TENANT,
                    kb_id=KB,
                    document_id=DOCUMENT,
                    document_version_id=DOCUMENT_VERSION,
                    source_index_version_id=INDEX,
                    title="根",
                    toc_path=("根",),
                    order=0,
                    source_evidence_ids=(evidence_id,),
                    child_node_ids=("child",),
                ),
                StructureNode(
                    node_id="child",
                    node_type="heading",
                    tenant_id=TENANT,
                    kb_id=KB,
                    document_id=DOCUMENT,
                    document_version_id=DOCUMENT_VERSION,
                    source_index_version_id=INDEX,
                    title="子",
                    toc_path=("根", "子"),
                    order=1,
                    source_evidence_ids=(evidence_id,),
                ),
            ),
        )


def test_advanced_snapshot_rejects_undeclared_provenance() -> None:
    catalog = _catalog()
    snapshot = _structure_snapshot(catalog)
    with pytest.raises(ValidationError, match="outside"):
        AdvancedIndexSnapshot(
            index=snapshot.index,
            evidence_ids=(next(iter(catalog)),),
            structure=snapshot.structure,
        )


def test_integrity_validator_rejects_missing_and_cross_scope_evidence() -> None:
    catalog = _catalog()
    snapshot = _structure_snapshot(catalog)
    truncated = dict(catalog)
    truncated.pop(next(iter(truncated)))
    with pytest.raises(AdvancedRetrievalError) as error:
        validate_advanced_snapshot(snapshot, truncated)
    assert error.value.code == "MISSING_PROVENANCE"
    first_id = next(iter(catalog))
    cross_scope = dict(catalog)
    cross_scope[first_id] = _evidence(
        "cross",
        catalog[first_id].content,
        tenant="tenant_other",
    )
    # Keep the map key intentionally different from the replacement stable ID.
    with pytest.raises(AdvancedRetrievalError) as error:
        validate_advanced_snapshot(snapshot, cross_scope)
    assert error.value.code == "EVIDENCE_CATALOG_KEY_MISMATCH"


def test_lifecycle_requires_integrity_and_holdout_gate_before_publish() -> None:
    catalog = _catalog()
    repository = InMemoryAdvancedIndexRepository()
    lifecycle = AdvancedIndexLifecycleService(repository, clock=lambda: NOW)
    staging = _structure_snapshot(catalog, status="staging")
    staging = staging.model_copy(
        update={"index": staging.index.model_copy(update={"evaluation": _gate(passed=False)})}
    )
    lifecycle.register_staging(staging)
    validated = lifecycle.validate(
        staging.index.advanced_index_id,
        evidence_by_id=catalog,
    )
    assert validated.index.status == "validated"
    with pytest.raises(AdvancedRetrievalError) as error:
        lifecycle.publish(
            staging.index.advanced_index_id,
            active_source_index_version_id=INDEX,
        )
    assert error.value.code == "EVALUATION_GATE_FAILED"


def test_lifecycle_publish_retires_previous_version_and_rollback_restores_it() -> None:
    catalog = _catalog()
    repository = InMemoryAdvancedIndexRepository()
    lifecycle = AdvancedIndexLifecycleService(repository, clock=lambda: NOW)
    first = _structure_snapshot(
        catalog,
        status="staging",
        advanced_id="adv_toc_v1",
        version=1,
    )
    lifecycle.register_staging(first)
    lifecycle.validate(first.index.advanced_index_id, evidence_by_id=catalog)
    active_first = lifecycle.publish(
        first.index.advanced_index_id,
        active_source_index_version_id=INDEX,
    )
    second = _structure_snapshot(
        catalog,
        status="staging",
        advanced_id="adv_toc_v2",
        version=2,
    )
    lifecycle.register_staging(second)
    lifecycle.validate(second.index.advanced_index_id, evidence_by_id=catalog)
    active_second = lifecycle.publish(
        second.index.advanced_index_id,
        active_source_index_version_id=INDEX,
    )
    assert active_second.index.status == "active"
    retired_first = repository.get_snapshot(active_first.index.advanced_index_id)
    assert retired_first is not None
    assert retired_first.index.status == "retired"
    rolled_back = lifecycle.rollback(
        active_first.index.advanced_index_id,
        active_source_index_version_id=INDEX,
    )
    assert rolled_back.index.status == "active"
    retired_second = repository.get_snapshot(active_second.index.advanced_index_id)
    assert retired_second is not None
    assert retired_second.index.status == "retired"


def test_source_index_change_marks_old_advanced_index_stale() -> None:
    catalog = _catalog()
    repository = InMemoryAdvancedIndexRepository()
    repository.save_snapshot(_structure_snapshot(catalog))
    lifecycle = AdvancedIndexLifecycleService(repository, clock=lambda: NOW)
    stale = lifecycle.mark_stale_for_source_change(
        tenant_id=TENANT,
        kb_id=KB,
        active_source_index_version_id="idx_manual_v2",
    )
    assert len(stale) == 1
    assert stale[0].index.status == "stale"
    assert stale[0].index.stale_reason == "SOURCE_INDEX_REPLACED"


def test_toc_retrieval_expands_to_real_body_evidence_with_separate_scores() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(
            ("toc",),
            flags=AdvancedFeatureFlags(toc_enabled=True),
        ),
        snapshots=(_structure_snapshot(catalog),),
        evidence_by_id=catalog,
    )
    assert result.candidates
    assert not result.diagnostics.degraded_to_baseline
    assert {candidate.evidence.evidence_id for candidate in result.candidates}.issubset(catalog)
    observation = result.candidates[0].observations[0]
    assert observation.route == "toc"
    assert "parent_score=" in observation.explanation
    assert "content_score=" in observation.explanation
    assert not observation.generated_routing_text
    assert result.diagnostics.outcomes[0].status == "succeeded"


def test_toc_budget_is_hard_and_deterministic() -> None:
    catalog = _catalog()
    request = _request(
        ("toc",),
        flags=AdvancedFeatureFlags(toc_enabled=True),
        budget=AdvancedRetrievalBudget(
            max_total_candidates=1,
            max_candidates_per_route=1,
            toc_node_limit=1,
            toc_expansion_limit=1,
        ),
    )
    service = AdvancedRetrievalService()
    one = service.search(
        request,
        snapshots=(_structure_snapshot(catalog),),
        evidence_by_id=catalog,
    )
    two = service.search(
        request,
        snapshots=(_structure_snapshot(catalog),),
        evidence_by_id=catalog,
    )
    assert len(one.candidates) <= 1
    assert one.model_dump(mode="json") == two.model_dump(mode="json")


def test_raptor_tree_rejects_missing_or_incomplete_leaf_provenance() -> None:
    catalog = _catalog()
    evidence_ids = tuple(catalog)
    child = RaptorSummaryNode(
        summary_id="child",
        level=1,
        text="叶子摘要",
        source_evidence_ids=(evidence_ids[0], evidence_ids[1]),
        source_chunk_ids=("a", "b"),
        cluster_method="deterministic",
        prompt_id="summary",
        prompt_version="v1",
        model_id="qwen",
    )
    parent = RaptorSummaryNode(
        summary_id="parent",
        level=2,
        text="父摘要",
        source_evidence_ids=(evidence_ids[0],),
        source_chunk_ids=("a",),
        child_summary_ids=("child",),
        cluster_method="deterministic",
        prompt_id="summary",
        prompt_version="v1",
        model_id="qwen",
    )
    with pytest.raises(ValidationError, match="contain all child"):
        RaptorSnapshot(
            advanced_index_id="adv_raptor",
            tree_version="tree_v1",
            source_index_version_id=INDEX,
            summaries=(child, parent),
        )


def test_raptor_summary_only_routes_and_citations_remain_leaf_evidence() -> None:
    catalog = _catalog()
    snapshot = _raptor_snapshot(catalog)
    result = AdvancedRetrievalService().search(
        _request(
            ("raptor",),
            flags=AdvancedFeatureFlags(raptor_enabled=True),
        ),
        snapshots=(snapshot,),
        evidence_by_id=catalog,
    )
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.evidence.evidence_id in catalog
        assert candidate.evidence.content != snapshot.raptor.summaries[0].text
        assert candidate.observations[0].generated_routing_text
        assert "citation_target=leaf_evidence" in candidate.observations[0].explanation


def test_graph_contract_rejects_inferred_relation_without_confidence_or_entity() -> None:
    catalog = _catalog()
    evidence_id = next(iter(catalog))
    with pytest.raises(ValidationError, match="requires confidence"):
        GraphRelation(
            relation_id="rel",
            source_entity_id="a",
            target_entity_id="b",
            relation_type="requires",
            statement="A requires B",
            inferred=True,
            source_evidence_ids=(evidence_id,),
        )
    entity = GraphEntity(
        entity_id="a",
        canonical_name="A",
        entity_type="device",
        source_evidence_ids=(evidence_id,),
    )
    relation = GraphRelation(
        relation_id="rel",
        source_entity_id="a",
        target_entity_id="missing",
        relation_type="requires",
        statement="A requires missing",
        inferred=False,
        source_evidence_ids=(evidence_id,),
    )
    with pytest.raises(ValidationError, match="missing entity"):
        GraphSnapshot(
            advanced_index_id="adv_graph",
            graph_version="graph_v1",
            source_index_version_id=INDEX,
            entities=(entity,),
            relations=(relation,),
        )


def test_graph_relation_provenance_must_support_both_endpoint_entities() -> None:
    catalog = _catalog()
    first, second = tuple(catalog)[:2]
    left = GraphEntity(
        entity_id="left",
        canonical_name="Left",
        entity_type="device",
        source_evidence_ids=(first,),
    )
    right = GraphEntity(
        entity_id="right",
        canonical_name="Right",
        entity_type="control",
        source_evidence_ids=(second,),
    )
    relation = GraphRelation(
        relation_id="unsupported",
        source_entity_id="left",
        target_entity_id="right",
        relation_type="requires",
        statement="Unsupported inferred edge",
        inferred=True,
        confidence=0.8,
        source_evidence_ids=(first,),
    )
    with pytest.raises(ValidationError, match="both endpoint"):
        GraphSnapshot(
            advanced_index_id="adv_graph",
            graph_version="graph_v1",
            source_index_version_id=INDEX,
            entities=(left, right),
            relations=(relation,),
        )


def test_graph_retrieval_exposes_inference_and_returns_only_source_evidence() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(
            ("graphrag",),
            flags=AdvancedFeatureFlags(graphrag_enabled=True),
        ),
        snapshots=(_graph_snapshot(catalog),),
        evidence_by_id=catalog,
    )
    assert result.candidates
    assert all(candidate.evidence.evidence_id in catalog for candidate in result.candidates)
    observations = [
        observation for candidate in result.candidates for observation in candidate.observations
    ]
    provenance = [(item.provenance_kind, item.explanation) for item in observations] + [
        (support.provenance_kind, support.explanation)
        for item in observations
        for support in item.supporting_provenance
    ]
    assert any(
        kind == "graph_relation" and "inferred_confidence=0.9300" in explanation
        for kind, explanation in provenance
    )
    assert all(item.route == "graphrag" for item in observations)


def test_missing_provenance_fails_route_and_degrades_without_fake_candidate() -> None:
    catalog = _catalog()
    snapshot = _raptor_snapshot(catalog)
    incomplete = dict(catalog)
    incomplete.pop(next(iter(incomplete)))
    result = AdvancedRetrievalService().search(
        _request(
            ("raptor",),
            flags=AdvancedFeatureFlags(raptor_enabled=True),
        ),
        snapshots=(snapshot,),
        evidence_by_id=incomplete,
    )
    assert result.candidates == ()
    outcome = next(item for item in result.diagnostics.outcomes if item.route == "raptor")
    assert outcome.status == "failed"
    assert outcome.reason == "INTEGRITY_ERROR"
    assert outcome.omitted_missing_evidence == 1
    assert result.diagnostics.degraded_to_baseline


def test_source_index_mismatch_skips_route_before_reading_snapshot() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(
            ("toc",),
            flags=AdvancedFeatureFlags(toc_enabled=True),
            active_source_index="idx_manual_v2",
        ),
        snapshots=(_structure_snapshot(catalog),),
        evidence_by_id=catalog,
    )
    assert result.candidates == ()
    assert result.diagnostics.outcomes[0].reason == "SOURCE_INDEX_MISMATCH"


def test_three_routes_merge_by_real_evidence_with_explainable_route_observations() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(
            ("toc", "raptor", "graphrag"),
            flags=AdvancedFeatureFlags(
                toc_enabled=True,
                raptor_enabled=True,
                graphrag_enabled=True,
            ),
        ),
        snapshots=(
            _structure_snapshot(catalog),
            _raptor_snapshot(catalog),
            _graph_snapshot(catalog),
        ),
        evidence_by_id=catalog,
    )
    assert result.candidates
    assert any(len(candidate.observations) >= 2 for candidate in result.candidates)
    assert all(
        candidate.evidence.evidence_id in catalog and 0 <= candidate.final_score <= 1
        for candidate in result.candidates
    )
    assert [item.route for item in result.diagnostics.outcomes] == [
        "toc",
        "raptor",
        "graphrag",
    ]


def test_unrequested_routes_are_explicitly_traced_as_skipped() -> None:
    catalog = _catalog()
    result = AdvancedRetrievalService().search(
        _request(
            ("toc",),
            flags=AdvancedFeatureFlags(
                toc_enabled=True,
                raptor_enabled=True,
                graphrag_enabled=True,
            ),
        ),
        snapshots=(
            _structure_snapshot(catalog),
            _raptor_snapshot(catalog),
            _graph_snapshot(catalog),
        ),
        evidence_by_id=catalog,
    )
    by_route = {item.route: item for item in result.diagnostics.outcomes}
    assert by_route["raptor"].reason == "ROUTE_NOT_REQUESTED"
    assert by_route["graphrag"].reason == "ROUTE_NOT_REQUESTED"
