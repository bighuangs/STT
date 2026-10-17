from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.domain.query_planning import (
    FacetLedger,
    FacetNecessity,
    FacetSupportStatus,
    QueryFacet,
)
from app.rag.query.context_pack_service import (
    ContextBudget,
    ContextPack,
    ContextPackInputError,
    ContextPackPolicy,
    ExpansionCandidate,
    ExpansionRequest,
    build_context_pack,
)
from app.rag.query.context_assembly_service import _pack_summary


@dataclass(frozen=True)
class _CharacterCounter:
    """Deterministic one-character-per-token counter for contract tests."""

    @property
    def tokenizer_id(self) -> str:
        return "test-character-counter"

    @property
    def tokenizer_version(self) -> str:
        return "v1"

    def count(self, text: str) -> int:
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: max(0, max_tokens)]


def _evidence(
    *,
    chunk: str,
    content: str,
    document: str = "doc_manual",
    document_version: str = "dv_manual_v1",
    index_version: str = "idx_manual_v1",
    page: int = 1,
) -> Evidence:
    block_id = f"block-{chunk}"
    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id="tenant_default",
            kb_id="kb_manual",
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
                page_no=page,
                bbox=(0.1, 0.1, 0.9, 0.3),
            )
        ],
        source_label=f"{document}.pdf",
    )


def _document(
    evidence: Evidence,
    *,
    score: float,
    sub_query_ids: tuple[str, ...] = (),
) -> dict:
    return {
        "evidence_id": evidence.evidence_id,
        "evidence": evidence.model_dump(mode="json", exclude_none=True),
        "title": evidence.source_label,
        "text": evidence.content,
        "content": evidence.content,
        "type": "milvus",
        "rerank_score": score,
        "sub_query_ids": list(sub_query_ids),
    }


def _budget(
    evidence_tokens: int = 2_000,
    *,
    characters: int | None = None,
) -> ContextBudget:
    return ContextBudget.from_window(
        model_context_limit=4_000,
        reserved_system=200,
        reserved_history=300,
        reserved_output=500,
        safety_margin=100,
        evidence_budget=evidence_tokens,
        evidence_character_budget=characters,
    )


def test_pack_is_stable_budgeted_and_maps_labels_to_real_evidence() -> None:
    first = _evidence(chunk="chunk-a", content="启动设备前必须可靠接地。", document="doc-a")
    second = _evidence(chunk="chunk-b", content="检查急停按钮是否有效。", document="doc-b")
    documents = [
        _document(second, score=0.7),
        _document(first, score=0.9),
    ]

    one = build_context_pack(
        documents,
        query_run_id="run-one",
        budget=_budget(),
        counter=_CharacterCounter(),
    )
    two = build_context_pack(
        list(reversed(documents)),
        query_run_id="run-two",
        budget=_budget(),
        counter=_CharacterCounter(),
    )

    assert one.pack.query_run_id != two.pack.query_run_id
    assert one.pack.content_hash == two.pack.content_hash
    assert [block.evidence_id for block in one.pack.packed_blocks] == [
        first.evidence_id,
        second.evidence_id,
    ]
    assert [item.label for item in one.pack.label_mapping] == ["E1", "E2"]
    assert one.pack.used_tokens == len(one.pack.context_text)
    assert one.pack.used_tokens <= one.pack.budget.evidence_budget
    assert "[E1]（仅可用此标签引用）" in one.pack.context_text
    assert "第1页" in one.pack.context_text
    assert one.pack.packed_blocks[0].representation_id.startswith("rep_")
    assert one.pack.packed_blocks[0].original_content_hash == first.content_hash

    # TODO-08 citation validation receives the full original Evidence, while the
    # prompt uses the budgeted representation.
    assert one.citation_aliases[0]["label"] == "E1"
    assert one.citation_aliases[0]["evidence"]["evidence_id"] == first.evidence_id
    assert one.citation_aliases[0]["evidence"]["content"] == first.content


def test_required_sub_queries_are_covered_before_global_high_score_candidates() -> None:
    high = _evidence(
        chunk="chunk-high",
        content="一般背景信息。",
        document="doc-high",
    )
    q1 = _evidence(chunk="chunk-q1", content="问题一证据。", document="doc-q1")
    q2 = _evidence(chunk="chunk-q2", content="问题二证据。", document="doc-q2")

    result = build_context_pack(
        [
            _document(high, score=0.99),
            _document(q2, score=0.1, sub_query_ids=("sq-2",)),
            _document(q1, score=0.2, sub_query_ids=("sq-1",)),
        ],
        query_run_id="run-coverage",
        budget=_budget(),
        counter=_CharacterCounter(),
        required_sub_query_ids=("sq-1", "sq-2", "sq-missing"),
    )

    blocks = result.pack.packed_blocks
    assert [block.evidence_id for block in blocks[:2]] == [
        q1.evidence_id,
        q2.evidence_id,
    ]
    assert blocks[0].inclusion_reason == "sub_query_coverage"
    assert blocks[1].inclusion_reason == "sub_query_coverage"
    assert blocks[0].placement == "front"
    assert blocks[1].placement == "middle"
    assert blocks[-1].placement == "tail"
    assert result.pack.coverage.packed_sub_query_ids == ("sq-1", "sq-2")
    assert result.pack.coverage.uncovered_sub_query_ids == ("sq-missing",)
    assert result.pack.coverage.evidence_count_by_sub_query["sq-missing"] == 0


def test_context_pack_binds_real_evidence_to_facets_without_claiming_semantic_entailment() -> None:
    q1 = _evidence(chunk="chunk-facet-q1", content="安装步骤证据。", document="doc-q1")
    ledger = FacetLedger(
        facets=(
            QueryFacet(
                facet_id="qf_11111111111111111111111111111111",
                information_need="HAK180 如何安装",
                retrieval_expression="HAK180 如何安装",
                sub_query_ids=("sq-1",),
            ),
            QueryFacet(
                facet_id="qf_22222222222222222222222222222222",
                information_need="HAK180 如何维护",
                retrieval_expression="HAK180 如何维护",
                sub_query_ids=("sq-2",),
            ),
            QueryFacet(
                facet_id="qf_33333333333333333333333333333333",
                information_need="如果检查失败应如何处理",
                retrieval_expression="HAK180 如果检查失败应如何处理",
                necessity=FacetNecessity.CONDITIONAL,
            ),
        )
    )

    result = build_context_pack(
        [_document(q1, score=0.9, sub_query_ids=("sq-1",))],
        query_run_id="run-facet-ledger",
        budget=_budget(),
        counter=_CharacterCounter(),
        required_sub_query_ids=("sq-1", "sq-2"),
        facet_ledger=ledger,
    )

    facets = {facet.facet_id: facet for facet in result.pack.facet_ledger.facets}
    assert facets["qf_11111111111111111111111111111111"].support_status == (
        FacetSupportStatus.SUPPORTED
    )
    assert facets["qf_11111111111111111111111111111111"].evidence_ids == (q1.evidence_id,)
    assert facets["qf_22222222222222222222222222222222"].support_status == (
        FacetSupportStatus.UNSUPPORTED
    )
    assert facets["qf_33333333333333333333333333333333"].support_status == (
        FacetSupportStatus.UNASSESSED
    )
    summary = result.pack.coverage.facet_coverage
    assert summary.binding_semantics == "retrieval_branch_lineage_only"
    assert summary.branch_bound_facet_ids == ("qf_11111111111111111111111111111111",)
    assert summary.required_facet_ids == (
        "qf_11111111111111111111111111111111",
        "qf_22222222222222222222222222222222",
    )
    assert summary.conditional_facet_ids == ("qf_33333333333333333333333333333333",)
    assert summary.covered_facet_ids == ("qf_11111111111111111111111111111111",)
    assert summary.uncovered_required_facet_ids == ("qf_22222222222222222222222222222222",)
    assert result.pack.diagnostics.uncovered_required_facet_count == 1
    assert result.pack.diagnostics.branch_bound_facet_count == 1
    public_summary = _pack_summary(
        result.pack.model_dump(mode="json"),
        prompt_tokens=321,
    )
    assert public_summary["facet_coverage"] == summary.model_dump(mode="json")
    assert public_summary["uncovered_required_facet_count"] == 1


def test_context_pack_without_facet_fields_restores_as_legacy_checkpoint() -> None:
    evidence = _evidence(chunk="chunk-legacy-facet", content="旧 ContextPack 证据。")
    result = build_context_pack(
        [_document(evidence, score=1.0)],
        query_run_id="run-legacy-facet",
        budget=_budget(),
        counter=_CharacterCounter(),
    )
    payload = result.pack.model_dump(mode="python")
    payload["schema_version"] = "context-pack.v1"
    payload["strategy_version"] = "context-pack-builder.v1"
    payload.pop("facet_ledger")
    payload["coverage"].pop("facet_coverage")
    for field in (
        "facet_count",
        "required_facet_count",
        "supported_facet_count",
        "branch_bound_facet_count",
        "uncovered_required_facet_count",
    ):
        payload["diagnostics"].pop(field)

    restored = ContextPack.model_validate(payload)

    assert restored.facet_ledger == FacetLedger()
    assert restored.coverage.facet_coverage.covered_facet_ids == ()


def test_context_pack_v2_versions_hash_relevant_facet_contract() -> None:
    evidence = _evidence(chunk="chunk-v2-facet", content="启动前检查接地。")
    ledger = FacetLedger(
        facets=(
            QueryFacet(
                facet_id="qf_44444444444444444444444444444444",
                information_need="启动前检查哪些事项",
                retrieval_expression="启动前检查哪些事项",
                sub_query_ids=("sq-primary",),
            ),
        )
    )

    result = build_context_pack(
        [_document(evidence, score=1.0, sub_query_ids=("sq-primary",))],
        query_run_id="run-context-v2",
        budget=_budget(),
        counter=_CharacterCounter(),
        required_sub_query_ids=("sq-primary",),
        facet_ledger=ledger,
    )

    assert result.pack.schema_version == "context-pack.v2"
    assert result.pack.strategy_version == "context-pack-builder.v2"
    assert result.pack.facet_ledger.schema_version == "query-facet-ledger.v2"


def test_context_pack_rejects_policy_that_can_overflow_facet_evidence_binding():
    evidence = _evidence(chunk="chunk-binding-limit", content="启动前检查接地。")
    ledger = FacetLedger(
        facets=(
            QueryFacet(
                facet_id="qf_55555555555555555555555555555555",
                information_need="启动前检查事项",
                retrieval_expression="启动前检查事项",
                sub_query_ids=("sq-primary",),
            ),
        )
    )

    with pytest.raises(ContextPackInputError, match="max_blocks.*facet Evidence binding limit"):
        build_context_pack(
            [_document(evidence, score=1.0, sub_query_ids=("sq-primary",))],
            query_run_id="run-binding-limit",
            budget=_budget(),
            counter=_CharacterCounter(),
            policy=ContextPackPolicy(
                max_blocks=129,
                max_blocks_per_document=None,
                max_blocks_per_source=None,
            ),
            required_sub_query_ids=("sq-primary",),
            facet_ledger=ledger,
        )


def test_context_pack_accepts_legacy_subquery_ids_without_losing_coverage() -> None:
    evidence = _evidence(
        chunk="chunk-legacy-coverage",
        content="首次安装打印头时不要触摸芯片。",
    )
    document = _document(evidence, score=0.9)
    document.pop("sub_query_ids", None)
    document["subquery_ids"] = ["sq-install"]

    result = build_context_pack(
        [document],
        query_run_id="run-legacy-subquery-coverage",
        budget=_budget(),
        counter=_CharacterCounter(),
        required_sub_query_ids=("sq-install",),
    )

    assert result.pack.packed_blocks[0].sub_query_ids == ("sq-install",)
    assert result.pack.coverage.packed_sub_query_ids == ("sq-install",)
    assert result.pack.coverage.uncovered_sub_query_ids == ()


def test_exact_and_near_duplicates_are_omitted_and_sub_query_coverage_is_merged() -> None:
    retained = _evidence(
        chunk="chunk-best",
        content="设备启动前检查急停按钮。",
        document="doc-best",
    )
    same_evidence = _document(retained, score=0.8, sub_query_ids=("sq-duplicate",))
    same_content = _evidence(
        chunk="chunk-copy",
        content=retained.content,
        document="doc-copy",
    )
    near_content = _evidence(
        chunk="chunk-near",
        content="设备启动前，检查急停按钮！",
        document="doc-near",
    )

    result = build_context_pack(
        [
            _document(retained, score=0.9, sub_query_ids=("sq-main",)),
            same_evidence,
            _document(same_content, score=0.7, sub_query_ids=("sq-copy",)),
            _document(near_content, score=0.6, sub_query_ids=("sq-near",)),
        ],
        query_run_id="run-dedup",
        budget=_budget(),
        counter=_CharacterCounter(),
        required_sub_query_ids=("sq-main", "sq-duplicate", "sq-copy", "sq-near"),
    )

    assert len(result.pack.packed_blocks) == 1
    assert set(result.pack.packed_blocks[0].sub_query_ids) == {
        "sq-main",
        "sq-duplicate",
        "sq-copy",
        "sq-near",
    }
    assert {item.reason for item in result.pack.omitted} == {
        "duplicate_evidence",
        "duplicate_content",
        "near_duplicate_content",
    }
    assert all(
        item.duplicate_of_evidence_id == retained.evidence_id for item in result.pack.omitted
    )
    assert result.pack.diagnostics.deduplicated_count == 3
    assert result.pack.coverage.uncovered_sub_query_ids == ()


def test_token_and_character_budgets_use_extractive_truncation_with_explanation() -> None:
    long = _evidence(
        chunk="chunk-long",
        content="A" * 800,
        document="doc-long",
    )
    overflow = _evidence(
        chunk="chunk-overflow",
        content="B" * 300,
        document="doc-overflow",
    )
    result = build_context_pack(
        [
            _document(long, score=0.9),
            _document(overflow, score=0.8),
        ],
        query_run_id="run-budget",
        budget=_budget(evidence_tokens=220, characters=220),
        counter=_CharacterCounter(),
        policy=ContextPackPolicy(
            max_block_characters=500,
            min_truncated_tokens=8,
            max_blocks_per_document=None,
            max_blocks_per_source=None,
        ),
    )

    assert result.pack.used_tokens <= 220
    assert result.pack.used_characters <= 220
    assert len(result.pack.packed_blocks) == 1
    packed = result.pack.packed_blocks[0]
    assert packed.truncation is not None
    assert "block_character_limit" in packed.truncation.reasons
    assert "token_budget" in packed.truncation.reasons
    assert packed.text.endswith("…[证据正文因上下文预算被截断]")
    assert packed.original_content_hash == long.content_hash
    assert packed.text != long.content
    assert result.citation_aliases[0]["evidence"]["content"] == long.content
    assert result.pack.omitted[-1].reason in {"token_budget", "character_budget"}
    assert result.pack.diagnostics.truncated_count == 1


class _ExpansionResolver:
    def __init__(self, responses: list[ExpansionCandidate]) -> None:
        self.responses = responses
        self.requests: list[ExpansionRequest] = []

    def resolve(self, request: ExpansionRequest) -> list[ExpansionCandidate]:
        self.requests.append(request)
        return self.responses


def test_expansion_accepts_only_real_same_scope_evidence_and_records_relationship() -> None:
    anchor = _evidence(
        chunk="chunk-anchor",
        content="命中子块。",
        document="doc-manual",
    )
    parent = _evidence(
        chunk="chunk-parent",
        content="真实存在的父块。",
        document="doc-manual",
    )
    foreign_version = _evidence(
        chunk="chunk-neighbor",
        content="其他版本的邻块。",
        document="doc-manual",
        document_version="dv-manual-v2",
        index_version="idx-manual-v2",
    )
    resolver = _ExpansionResolver(
        [
            ExpansionCandidate(
                anchor_evidence_id=anchor.evidence_id,
                relation="parent",
                distance=0,
                evidence=parent,
            ),
            ExpansionCandidate(
                anchor_evidence_id=anchor.evidence_id,
                relation="next",
                distance=1,
                evidence=foreign_version,
            ),
        ]
    )

    result = build_context_pack(
        [_document(anchor, score=0.9, sub_query_ids=("sq-1",))],
        query_run_id="run-expansion",
        budget=_budget(),
        counter=_CharacterCounter(),
        policy=ContextPackPolicy(
            expansion_mode="parent_and_neighbors",
            neighbor_window=1,
            max_blocks_per_document=None,
            max_blocks_per_source=None,
        ),
        expansion_resolver=resolver,
    )

    assert resolver.requests[0].document_version_id == "dv_manual_v1"
    assert {block.evidence_id for block in result.pack.packed_blocks} == {
        anchor.evidence_id,
        parent.evidence_id,
    }
    parent_block = next(
        block for block in result.pack.packed_blocks if block.evidence_id == parent.evidence_id
    )
    assert parent_block.inclusion_reason == "expanded_parent"
    assert parent_block.related_refs[0].relation == "parent"
    assert parent_block.related_refs[0].anchor_evidence_id == anchor.evidence_id
    rejected = next(
        item for item in result.pack.omitted if item.evidence_id == foreign_version.evidence_id
    )
    assert rejected.reason == "expansion_scope_mismatch"
    assert result.pack.diagnostics.expansion_status == "completed_with_issues"


def test_missing_or_empty_expansion_resolver_never_fabricates_parent_or_neighbors() -> None:
    anchor = _evidence(chunk="chunk-anchor", content="唯一真实命中。")
    without_resolver = build_context_pack(
        [_document(anchor, score=1.0)],
        query_run_id="run-no-resolver",
        budget=_budget(),
        counter=_CharacterCounter(),
        policy=ContextPackPolicy(expansion_mode="parent_and_neighbors"),
    )
    assert len(without_resolver.pack.packed_blocks) == 1
    assert without_resolver.pack.packed_blocks[0].related_refs == ()
    assert without_resolver.pack.diagnostics.expansion_status == "resolver_unavailable"
    assert without_resolver.pack.diagnostics.expansion_issues == ("EXPANSION_RESOLVER_UNAVAILABLE",)

    empty_resolver = _ExpansionResolver([])
    empty = build_context_pack(
        [_document(anchor, score=1.0)],
        query_run_id="run-empty-resolver",
        budget=_budget(),
        counter=_CharacterCounter(),
        policy=ContextPackPolicy(expansion_mode="parent"),
        expansion_resolver=empty_resolver,
    )
    assert len(empty.pack.packed_blocks) == 1
    assert empty.pack.diagnostics.expansion_candidate_count == 0
    assert empty.pack.diagnostics.expansion_status == "completed"


def test_document_and_source_quotas_are_explainable() -> None:
    first = _evidence(chunk="chunk-1", content="第一块。", document="same-doc")
    second = _evidence(chunk="chunk-2", content="第二块。", document="same-doc")
    third = _evidence(chunk="chunk-3", content="第三块。", document="other-doc")
    result = build_context_pack(
        [
            _document(first, score=0.9),
            _document(second, score=0.8),
            _document(third, score=0.7),
        ],
        query_run_id="run-quotas",
        budget=_budget(),
        counter=_CharacterCounter(),
        policy=ContextPackPolicy(
            max_blocks_per_document=1,
            max_blocks_per_source=2,
        ),
    )
    assert [block.evidence_id for block in result.pack.packed_blocks] == [
        first.evidence_id,
        third.evidence_id,
    ]
    assert (
        next(item for item in result.pack.omitted if item.evidence_id == second.evidence_id).reason
        == "document_quota"
    )


def test_builder_fails_closed_for_unhydrated_candidates_and_invalid_budget() -> None:
    with pytest.raises(ContextPackInputError, match="prepare_reranked_evidence"):
        build_context_pack(
            [{"text": "arbitrary untrusted text", "score": 1.0}],
            query_run_id="run-invalid",
            budget=_budget(),
            counter=_CharacterCounter(),
        )

    with pytest.raises(ValueError, match="exceed model limit"):
        ContextBudget(
            model_context_limit=100,
            reserved_system=20,
            reserved_history=20,
            reserved_output=20,
            safety_margin=20,
            evidence_budget=30,
        )
