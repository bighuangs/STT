from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

from app.evaluation.metrics import evaluate_case


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)
sys.path.insert(0, str(CONTRACT_ROOT))

import concept_materialization  # noqa: E402
import activate_reviewed_suite as activate_suite_module  # noqa: E402
import review_contracts as review_contracts_module  # noqa: E402
from concept_materialization import meta_directive_reason  # noqa: E402

from merge_reviewed_contracts import (  # noqa: E402
    FROZEN_CASE_PASS_POLICY,
    _materialize_concept_groups,
    _merge_contract,
    _validate_contract_coverage,
    _variant_weak_reason,
)
from activate_reviewed_suite import (  # noqa: E402
    TARGET_COUNTS,
    ActivationError,
    _write_exact_or_refuse,
)
from review_contracts import (  # noqa: E402
    _POLARITY_REQUIRED_VALUES,
    _call_api,
    _canonicalize_saved_quote_fragment_inverse_composition,
    _canonicalize_saved_nonsemantic_composition,
    _canonicalize_same_response_coverage_statuses,
    _canonicalize_same_response_empty_required_branch_block,
    _canonicalize_same_response_inverse_edges,
    _canonicalize_same_response_unsupported_required_concept_block,
    _normalize_review_structure,
    _recover_prior_invalid_review,
    _sha256_json,
    _validate_against_payload,
    _validate_coverage_status_canonicalization_audit,
    _validate_empty_required_branch_block_audit,
    _validate_nonsemantic_canonicalization_composition_audit,
    _validate_quote_fragment_inverse_composition_audit,
    _validate_review,
    _validate_saved_strict_revalidation_audit,
    _validate_unsupported_required_concept_block_audit,
)
from validate_suite_v2 import (  # noqa: E402
    _checksum_coverage_errors,
    _contract_coverage_errors,
    _expected_concept_groups,
    _reviewed_parent_coverage_errors,
    _strict_success_response_errors,
    _variant_weak_reason as _validator_weak_reason,
)


def _answer_contract() -> dict:
    return {
        "answerability": {"required": "answer"},
        "required_concepts": {
            "all_of": [
                {
                    "concept_id": "c01",
                    "canonical_text": "条件甲",
                    "any_of": ["甲条件"],
                    "evidence_support": [{"evidence_id": "e1"}],
                    "query_branch_spans": ["分支一"],
                },
                {
                    "concept_id": "c02",
                    "canonical_text": "条件乙",
                    "any_of": ["乙条件"],
                    "evidence_support": [{"evidence_id": "e2"}],
                    "query_branch_spans": ["分支一"],
                },
            ]
        },
        "critical_constraints": {"numerics": [], "identifiers": [], "polarity": {}},
        "required_branches": [
            {
                "query_span": "分支一",
                "required_concept_ids": ["c01", "c02"],
            }
        ],
        "evidence_concept_coverage": [
            {
                "evidence_id": "e1",
                "required_concept_ids": ["c01"],
                "coverage_status": "covered",
            },
            {
                "evidence_id": "e2",
                "required_concept_ids": ["c02"],
                "coverage_status": "covered",
            },
        ],
    }


def test_weak_variant_gate_rejects_composed_meta_action_and_numeric_substring() -> None:
    for gate in (_variant_weak_reason, _validator_weak_reason):
        assert gate(
            "打开内容", critical_values=set(), evidence_tokens=set()
        ) == "generic_action_without_object"
        assert gate(
            "0", critical_values=set(), evidence_tokens={"2000"}
        ) == "short_identifier_or_numeric_without_explicit_support"
        assert gate(
            "0", critical_values=set(), evidence_tokens={"0"}
        ) == "short_identifier_or_numeric_without_explicit_support"
        assert gate("0", critical_values={"0"}, evidence_tokens={"0"}) is None
        for action in ("打印", "开启", "进入", "访问"):
            assert gate(
                action, critical_values=set(), evidence_tokens=set()
            ) == "generic_action_or_meta"


def test_short_compound_constraint_requires_atomic_critical_and_evidence_support() -> None:
    reason = concept_materialization.variant_weak_reason
    evidence = "AC220~240V(±10%),50HZ/60HZ(±2%),4A"
    critical = {"50", "60", "2"}
    for variant in (
        "50HZ/60HZ(±2%)",
        "50Hz/60Hz(±2%)",
        "50HZ/60HZ ±2%",
    ):
        assert reason(
            variant,
            critical_values=critical,
            evidence_tokens=set(),
            evidence_source_text=evidence,
        ) is None

    for variant, critical_values, source in (
        ("0", {"0"}, "2000"),
        ("3000", {"3000"}, "13000"),
        ("Android4.4", {"android44"}, "Android4.40"),
        ("192.168.0.1", {"19216801"}, "192.168.0.102"),
        ("TL-413H", {"tl413h"}, "TL-413H2"),
    ):
        assert reason(
            variant,
            critical_values=critical_values,
            evidence_tokens=set(),
            evidence_source_text=source,
        ) == "short_identifier_or_numeric_without_explicit_support"

    contract = {
        "required_concepts": {
            "all_of": [
                {
                    "concept_id": "c01",
                    "canonical_text": "50HZ/60HZ(±2%)",
                    "any_of": ["50Hz/60Hz(±2%)", "50HZ/60HZ ±2%"],
                }
            ]
        },
        "critical_constraints": {
            "numerics": [
                {"value": "50"},
                {"value": "60"},
                {"value": "2"},
            ],
            "identifiers": [],
        },
    }
    groups, _audit = concept_materialization.materialize_concept_groups(
        contract, [{"text": evidence}]
    )
    assert groups["all_of"][0]["any_of"] == [
        "50HZ/60HZ(±2%)",
        "50HZ/60HZ ±2%",
    ]


def test_query_meta_directive_cannot_be_materialized_as_answer_concept() -> None:
    for directive in (
        "按顺序说明操作步骤",
        "请说明",
        "列出全部步骤",
        "结合证据回答",
        "具体路径",
        "分别回答两个问题",
    ):
        assert meta_directive_reason(directive) == (
            "query_meta_directive_without_answer_fact"
        )
    assert meta_directive_reason("按顺序打开纸盒并取出卡纸") is None
    assert meta_directive_reason("具体路径为系统设置 > 网络设置") is None

    contract = _answer_contract()
    contract["required_concepts"]["all_of"][0].update(
        {
            "canonical_text": "按顺序说明操作步骤",
            "any_of": ["依次说明操作"],
        }
    )
    with pytest.raises(RuntimeError, match="query meta-directive"):
        _materialize_concept_groups(contract, case_id="meta-directive", evidence=[])


def test_cross_concept_production_match_collision_is_filtered_with_validator_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _answer_contract()
    contract["required_concepts"]["all_of"][0]["canonical_text"] = "条件甲完整表述"
    contract["required_concepts"]["all_of"][0]["any_of"] = ["甲条件"]
    contract["required_concepts"]["all_of"][1]["canonical_text"] = "条件乙完整表述"
    contract["required_concepts"]["all_of"][1]["any_of"] = ["乙条件", "条件"]

    def synthetic_production_match(answer: str, variant: str) -> bool:
        return variant in answer

    monkeypatch.setattr(
        concept_materialization,
        "concept_variant_covered",
        synthetic_production_match,
    )
    groups, audit = _materialize_concept_groups(
        contract, case_id="collision", evidence=[]
    )
    validator_groups, validator_audit = _expected_concept_groups(contract, [])
    assert groups == validator_groups
    assert audit == validator_audit
    assert groups["all_of"][1]["any_of"] == ["条件乙完整表述", "乙条件"]
    assert any(
        item["variant"] == "条件"
        and item["reason"] == "global_collision_free_subset_exclusion"
        for item in audit["removed_variants"]
    )


def test_evidence_enumerated_cjk_siblings_remain_production_discriminative() -> None:
    contract = _answer_contract()
    contract["required_concepts"]["all_of"][0].update(
        canonical_text="有线网络信息", any_of=["有线网络信息"]
    )
    contract["required_concepts"]["all_of"][1].update(
        canonical_text="无线网络信息", any_of=["无线网络信息"]
    )
    evidence = [{"text": "二级菜单：1.有线网络信息；2.无线网络信息。"}]

    groups, audit = _materialize_concept_groups(
        contract, case_id="cjk-siblings", evidence=evidence
    )

    assert groups == {
        "all_of": [
            {"concept_key": "c01", "any_of": ["有线网络信息"]},
            {"concept_key": "c02", "any_of": ["无线网络信息"]},
        ]
    }
    assert audit["production_match_collision_count"] == 0

    case = {
        "case_id": "cjk-sibling-score",
        "query": "网络信息菜单有哪些选项？",
        "reference_answer": "有线网络信息、无线网络信息。",
        "expected_terms": [],
        "relevant_chunk_ids": ["e1"],
        "unanswerable": False,
        "concept_groups": groups,
    }
    evidence_row = {"chunk_id": "e1", "text": evidence[0]["text"]}
    complete = evaluate_case(
        case,
        {
            "answer": "可查看有线网络信息和无线网络信息。",
            "retrieved": [evidence_row],
            "context": [evidence_row],
        },
    )
    missing_one = evaluate_case(
        case,
        {
            "answer": "可查看有线网络信息。",
            "retrieved": [evidence_row],
            "context": [evidence_row],
        },
    )
    assert complete["metrics"]["generation"]["concept_group_coverage"] == 1.0
    assert missing_one["metrics"]["generation"]["concept_group_coverage"] == 0.5


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("网络信息", "无线网络信息"),
        ("无线网络", "无线网络信息"),
        ("WIFI", "WIFI 热点信息"),
        ("3000", "3000"),
    ],
)
def test_true_containment_and_atomic_collisions_are_not_cjk_sibling_exemptions(
    left: str, right: str
) -> None:
    groups = [
        {"concept_key": "c01", "any_of": [left]},
        {"concept_key": "c02", "any_of": [right]},
    ]

    assert concept_materialization._collision_witnesses(groups)


def test_v6_exact_duplicate_is_selected_by_global_solver_not_deleted_from_both_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨组完全相同variant也必须作为冲突边参与全局可行解选择。"""

    contract = _answer_contract()
    contract["required_concepts"]["all_of"][0].update(
        canonical_text="共享表述", any_of=["甲独立表述"]
    )
    contract["required_concepts"]["all_of"][1].update(
        canonical_text="共享表述", any_of=[]
    )

    def synthetic_match(answer: str, variant: str) -> bool:
        return answer == variant

    monkeypatch.setattr(
        concept_materialization, "concept_variant_covered", synthetic_match
    )
    groups, audit = concept_materialization.materialize_concept_groups(
        contract, [], policy_version=concept_materialization.MATERIALIZATION_POLICY_VERSION
    )

    assert groups == {
        "all_of": [
            {"concept_key": "c01", "any_of": ["甲独立表述"]},
            {"concept_key": "c02", "any_of": ["共享表述"]},
        ]
    }
    assert audit["subset_selection"]["completed"] is True
    assert any(
        item["concept_key"] == "c01"
        and item["variant"] == "共享表述"
        and item["reason"] == "global_collision_free_subset_exclusion"
        for item in audit["removed_variants"]
    )


def test_v6_solver_rejects_more_than_160_variants_before_any_matcher_call() -> None:
    groups = [
        {
            "concept_key": "c01",
            "any_of": [f"variant-{index}" for index in range(161)],
        }
    ]
    matcher_calls = 0

    def matcher(_answer: str, _variant: str) -> bool:
        nonlocal matcher_calls
        matcher_calls += 1
        return False

    with pytest.raises(ValueError, match="total-vertex cap exceeded"):
        concept_materialization._maximum_collision_free_subsets(
            groups, matcher=matcher
        )
    assert matcher_calls == 0


def test_v6_solver_fails_closed_on_component_and_global_state_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = [
        {"concept_key": f"c{index:02d}", "any_of": [f"v{index:02d}"]}
        for index in range(23)
    ]

    with pytest.raises(ValueError, match="component exceeds"):
        concept_materialization._maximum_collision_free_subsets(
            connected, matcher=lambda _left, _right: True
        )

    monkeypatch.setattr(
        concept_materialization, "SUBSET_SOLVER_MAX_TOTAL_STATES", 0
    )
    with pytest.raises(ValueError, match="global state cap exceeded"):
        concept_materialization._maximum_collision_free_subsets(
            [
                {"concept_key": "c01", "any_of": ["alpha"]},
                {"concept_key": "c02", "any_of": ["beta"]},
            ],
            matcher=lambda _left, _right: False,
        )


def test_v6_solver_maximizes_variants_then_uses_authored_order_tie_break() -> None:
    groups = [
        {"concept_key": "c01", "any_of": ["a1", "a2"]},
        {"concept_key": "c02", "any_of": ["b1", "b2"]},
    ]

    def matcher(left: str, right: str) -> bool:
        return {left, right} in ({"a1", "b1"}, {"a2", "b2"})

    selected, audit = concept_materialization._maximum_collision_free_subsets(
        groups, matcher=matcher
    )
    # 两组必须非空，所以相同最大数量的可行方案选择更早的a1。
    assert selected == [
        {"concept_key": "c01", "any_of": ["a1"]},
        {"concept_key": "c02", "any_of": ["b2"]},
    ]
    assert audit["algorithm"] == concept_materialization.SUBSET_SOLVER_ALGORITHM
    assert audit["objective"] == concept_materialization.SUBSET_SOLVER_OBJECTIVE
    assert audit["tie_break"] == concept_materialization.SUBSET_SOLVER_TIE_BREAK
    assert audit["completed"] is True
    assert audit["visited_total_states"] > 0


def test_reviewed_parent_coverage_rejects_duplicate_parent_with_relabelled_case() -> None:
    expected_ids = {"easy": {"V2-1", "V2-2"}}
    expected_parents = {"V2-1": "P-1", "V2-2": "P-2"}
    reviewed = {
        "easy": [
            {"case_id": "V2-1", "metadata": {"parent_case_id": "P-1"}},
            {"case_id": "V2-2", "metadata": {"parent_case_id": "P-1"}},
        ]
    }
    reviewed_qa = [
        {"difficulty": "easy", "case_id": "V2-1", "parent_case_id": "P-1"},
        {"difficulty": "easy", "case_id": "V2-2", "parent_case_id": "P-1"},
    ]
    errors = _reviewed_parent_coverage_errors(
        expected_case_ids_by_difficulty=expected_ids,
        expected_parent_by_case_id=expected_parents,
        reviewed_cases_by_difficulty=reviewed,
        reviewed_qa=reviewed_qa,
        active_payload_parent_ids={"P-1"},
    )
    assert any("parent lineage mismatch" in error for error in errors)
    assert any("150 unique candidate parents" in error for error in errors)
    assert any("active blind payloads" in error for error in errors)


def test_checksum_coverage_rejects_duplicate_and_missing_final_artifacts() -> None:
    errors = _checksum_coverage_errors(
        ["build_suite_v2.py", "build_suite_v2.py"],
        {"build_suite_v2.py", "review_contracts.py", "reviewed/summary.json"},
    )
    assert any("duplicate" in error for error in errors)
    assert any("review_contracts.py" in error for error in errors)
    assert any("reviewed/summary.json" in error for error in errors)


def test_answer_contract_coverage_rejects_swapped_concept_evidence_edges() -> None:
    contract = _answer_contract()
    contract["evidence_concept_coverage"][0]["required_concept_ids"] = ["c02"]
    contract["evidence_concept_coverage"][1]["required_concept_ids"] = ["c01"]
    with pytest.raises(RuntimeError, match="edge coverage"):
        _validate_contract_coverage(
            contract,
            [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            case_id="edge-swap",
            expected_branch_spans=["分支一"],
        )
    assert _contract_coverage_errors(
        contract,
        [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
        expected_branch_spans=["分支一"],
        expected_unanswerable=False,
    ) == ["concept/evidence/branch edges are incomplete or asymmetric"]


def test_answer_contract_coverage_rejects_missing_query_branch() -> None:
    contract = _answer_contract()
    with pytest.raises(RuntimeError, match="grain"):
        _validate_contract_coverage(
            contract,
            [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            case_id="missing-branch",
            expected_branch_spans=["分支一", "分支二"],
        )


def _partial_evidence_review() -> tuple[dict, dict]:
    review = {
        "verdict": "revise",
        "answerability": "answer",
        "required_concepts": {
            "all_of": [
                {
                    "concept_key": "c01",
                    "canonical_text": "出纸托盘防止纸张滑落",
                    "any_of": ["防止打印出来的纸张滑落"],
                    "evidence_labels": ["E1"],
                    "branch_labels": ["B1"],
                },
                {
                    "concept_key": "c02",
                    "canonical_text": "多功能纸盒导纸板调整纸张两侧宽度",
                    "any_of": ["调整纸张两侧宽度"],
                    "evidence_labels": ["E1"],
                    "branch_labels": ["B2"],
                },
            ]
        },
        "critical_constraints": {
            "numerics": [],
            "identifiers": [],
            "polarity": {"required": "not_applicable", "cues": []},
        },
        "required_branches": [
            {
                "branch_label": "B1",
                "query_span": "出纸托盘的主要功能是什么",
                "required_concept_keys": ["c01"],
            },
            {
                "branch_label": "B2",
                "query_span": "与多功能纸盒导纸板相比有何区别",
                "required_concept_keys": ["c02"],
            },
        ],
        "evidence_concept_coverage": [
            {
                "evidence_label": "E1",
                "required_concept_keys": ["c01", "c02"],
                "status": "covered",
            },
            {
                "evidence_label": "E2",
                "required_concept_keys": [],
                "status": "uncovered",
            },
        ],
        "review_notes": ["E2仅为相关部件图像，不直接支持问句答案。"],
    }
    payload = {
        "query": "问题",
        "reference_answer": "答案",
        "unanswerable": False,
        "legacy_expected_terms": [],
        "forbidden_terms": [],
        "branches": [
            {"branch_label": "B1", "query_span": "出纸托盘的主要功能是什么"},
            {
                "branch_label": "B2",
                "query_span": "与多功能纸盒导纸板相比有何区别",
            },
        ],
        "evidence": [
            {"evidence_label": "E1", "text": "直接支持答案"},
            {"evidence_label": "E2", "text": "仅主题相关"},
        ],
    }
    return review, payload


def test_answer_review_allows_explicit_uncovered_distractor_evidence() -> None:
    review, payload = _partial_evidence_review()
    assert _validate_review(review) == []
    assert _validate_against_payload(review, payload) == []

    candidate = {
        "case_id": "SYNTH-PARTIAL-EVIDENCE-V2-001",
        "query": payload["query"],
        "reference_answer": payload["reference_answer"],
        "unanswerable": False,
        "expected_terms": [],
        "forbidden_terms": [],
        "relevant_evidence": [
            {
                "evidence_id": "e1",
                "chunk_id": "chunk-1",
                "page": 13,
                "text": "出纸托盘防止打印出来的纸张滑落；多功能纸盒导纸板调整纸张两侧宽度。",
            },
            {
                "evidence_id": "e2",
                "chunk_id": "chunk-2",
                "page": 15,
                "text": "激光碳粉盒图像。",
            },
        ],
        "answer_contract": {
            "required_branches": [
                {"branch_id": "b01", "query_span": payload["branches"][0]["query_span"]},
                {"branch_id": "b02", "query_span": payload["branches"][1]["query_span"]},
            ],
            "review": {"flags": []},
        },
        "metadata": {
            "parent_case_id": "SYNTH-PARTIAL-EVIDENCE-001",
            "difficulty": "hard",
        },
    }
    merged, _qa = _merge_contract(
        candidate_case=candidate,
        response={
            "custom_id": "partial-evidence",
            "model": "test-model",
            "prompt_sha256": "prompt",
            "review": review,
        },
    )
    assert [
        (item["coverage_status"], item["required_concept_ids"])
        for item in merged["evidence_concept_coverage"]
    ] == [("covered", ["c01", "c02"]), ("uncovered", [])]
    _validate_contract_coverage(
        merged,
        candidate["relevant_evidence"],
        case_id=candidate["case_id"],
        expected_branch_spans=[item["query_span"] for item in payload["branches"]],
    )


def _partial_evidence_candidate(payload: dict) -> dict:
    return {
        "case_id": "SYNTH-NOT-REQUIRED-V2-001",
        "query": payload["query"],
        "reference_answer": payload["reference_answer"],
        "unanswerable": False,
        "expected_terms": [],
        "forbidden_terms": [],
        "relevant_evidence": [
            {
                "evidence_id": "e1",
                "chunk_id": "chunk-1",
                "page": 13,
                "text": "出纸托盘防止纸张滑落；导纸板调整纸张两侧宽度。",
            },
            {
                "evidence_id": "e2",
                "chunk_id": "chunk-2",
                "page": 15,
                "text": "仅包含相关部件图像。",
            },
        ],
        "answer_contract": {
            "required_branches": [
                {"branch_id": "b01", "query_span": payload["branches"][0]["query_span"]},
                {"branch_id": "b02", "query_span": payload["branches"][1]["query_span"]},
            ],
            "review": {"flags": []},
        },
        "metadata": {
            "parent_case_id": "SYNTH-NOT-REQUIRED-001",
            "difficulty": "hard",
        },
    }


@pytest.mark.parametrize(
    "alias", ["not_required", "not_applicable", "not_needed", "not_covered"]
)
def test_zero_support_status_alias_canonicalizes_with_merge_validator_parity(
    alias: str,
) -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = alias
    review["critical_constraints"]["identifiers"].append(
        {"value": "SSID", "evidence_labels": ["E2"]}
    )
    payload["evidence"][1]["text"] = (
        "图注标出 SSID，但不完整支持任一回答概念。"
    )

    projected, actions, raw_errors = _canonicalize_same_response_coverage_statuses(
        review, payload
    )

    assert raw_errors == [
        "evidence_concept_coverage[1].status must be one of "
        "['covered', 'insufficient', 'uncovered']"
    ]
    assert actions == [
        {
            "coverage_index": 1,
            "evidence_label": "E2",
            "field": "status",
            "before": alias,
            "after": "uncovered",
            "required_concept_keys": [],
            "authority": "zero_direct_concept_support_edges",
        }
    ]
    assert projected["evidence_concept_coverage"][1]["status"] == "uncovered"
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []

    candidate = _partial_evidence_candidate(payload)
    candidate["relevant_evidence"][1]["text"] = payload["evidence"][1]["text"]
    merged, _qa = _merge_contract(
        candidate_case=candidate,
        response={
            "custom_id": "not-required-projected",
            "model": "test-model",
            "prompt_sha256": "prompt",
            "review": projected,
        },
    )
    expected_spans = [item["query_span"] for item in payload["branches"]]
    _validate_contract_coverage(
        merged,
        candidate["relevant_evidence"],
        case_id=candidate["case_id"],
        expected_branch_spans=expected_spans,
    )
    assert _contract_coverage_errors(
        merged,
        candidate["relevant_evidence"],
        expected_branch_spans=expected_spans,
        expected_unanswerable=False,
    ) == []


@pytest.mark.parametrize("status", ["uncovered", "insufficient"])
def test_constraint_only_evidence_is_independent_of_full_concept_coverage(
    status: str,
) -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = status
    review["critical_constraints"]["identifiers"].append(
        {"value": "SSID", "evidence_labels": ["E2"]}
    )
    payload["evidence"][1]["text"] = (
        "图注标出 SSID，但不完整支持任一回答概念。"
    )

    assert _validate_review(review) == []
    assert _validate_against_payload(review, payload) == []

    candidate = _partial_evidence_candidate(payload)
    candidate["relevant_evidence"][1]["text"] = payload["evidence"][1]["text"]
    merged, _qa = _merge_contract(
        candidate_case=candidate,
        response={
            "custom_id": "constraint-only-evidence",
            "model": "test-model",
            "prompt_sha256": "prompt",
            "review": review,
        },
    )
    expected_spans = [item["query_span"] for item in payload["branches"]]
    _validate_contract_coverage(
        merged,
        candidate["relevant_evidence"],
        case_id=candidate["case_id"],
        expected_branch_spans=expected_spans,
    )
    assert _contract_coverage_errors(
        merged,
        candidate["relevant_evidence"],
        expected_branch_spans=expected_spans,
        expected_unanswerable=False,
    ) == []
    assert merged["critical_constraints"]["identifiers"][-1]["evidence_ids"] == [
        "e2"
    ]
    assert merged["evidence_concept_coverage"][1] == {
        "evidence_id": "e2",
        "chunk_id": "chunk-2",
        "page": 15,
        "required_concept_ids": [],
        "coverage_status": status,
    }


def test_constraint_evidence_label_must_still_exist_in_payload_universe() -> None:
    review, payload = _partial_evidence_review()
    review["critical_constraints"]["identifiers"].append(
        {"value": "SSID", "evidence_labels": ["E99"]}
    )

    errors = _validate_against_payload(review, payload)

    assert errors == ["unknown evidence labels: ['E99']"]
    with pytest.raises(RuntimeError, match="unknown evidence label"):
        _merge_contract(
            candidate_case=_partial_evidence_candidate(payload),
            response={
                "custom_id": "unknown-constraint-evidence",
                "model": "test-model",
                "prompt_sha256": "prompt",
                "review": review,
            },
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["evidence_concept_coverage"][1].update(
            required_concept_keys=["c01"]
        ),
        lambda review: review["required_concepts"]["all_of"][0][
            "evidence_labels"
        ].append("E2"),
        lambda review: review["evidence_concept_coverage"][0].update(
            status="not_required", required_concept_keys=[]
        ),
        lambda review: review["evidence_concept_coverage"][1].update(
            status="irrelevant"
        ),
        lambda review: review["evidence_concept_coverage"][1].update(status="unused"),
        lambda review: review["evidence_concept_coverage"][1].update(status="none"),
        lambda review: review["evidence_concept_coverage"][1].update(
            status="not-applicable"
        ),
    ],
)
@pytest.mark.parametrize(
    "alias", ["not_required", "not_applicable", "not_needed", "not_covered"]
)
def test_zero_support_status_alias_rejects_any_support_or_other_defect(
    mutator, alias: str
) -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = alias
    mutator(review)

    projected, actions, _raw_errors = _canonicalize_same_response_coverage_statuses(
        review, payload
    )

    assert projected == review
    assert actions == []


def test_saved_not_required_status_recovery_requires_append_only_source_and_is_adoptable() -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = "not_required"
    raw_errors = [*_validate_review(review), *_validate_against_payload(review, payload)]
    request_row = {
        "custom_id": "saved-not-required",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = {
        "custom_id": "saved-not-required",
        "status": "failed",
        "prompt_sha256": "prompt",
        "attempt": 1,
        "model": "test-model",
        "runner_version": "2.14.0-allowlisted-provider-error-receipt",
        "completed_at": "2026-08-11T00:00:00+00:00",
        "invalid_attempts": [
            {
                "attempt": 1,
                "review": review,
                "review_sha256": _sha256_json(review),
                "validation_errors": raw_errors,
            }
        ],
    }

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    assert recovered["review"]["evidence_concept_coverage"][1]["status"] == "uncovered"
    audit = recovered["same_response_coverage_status_canonicalization"]
    assert audit["api_call_performed"] is False
    assert audit["raw_review_sha256"] == _sha256_json(review)
    assert _validate_coverage_status_canonicalization_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    ) == []
    legacy_pairs = [
        (
            "2.15.0-not-required-coverage-canonicalization",
            "same-response-zero-support-not-required-to-uncovered-v1",
        ),
        (
            "2.16.0-unsupported-required-concept-block",
            "same-response-zero-support-not-required-to-uncovered-v1",
        ),
        (
            "2.18.0-zero-support-coverage-status-aliases",
            "same-response-zero-support-status-alias-to-uncovered-v2",
        ),
        (
            "2.19.0-native-block-zero-support-status-alias",
            "same-response-zero-support-status-alias-to-uncovered-v3",
        ),
    ]
    for runner_version, policy in legacy_pairs:
        legacy = deepcopy(recovered)
        legacy["runner_version"] = runner_version
        legacy_audit = legacy[
            "same_response_coverage_status_canonicalization"
        ]
        legacy_audit["policy"] = policy
        for action in legacy_audit["actions"]:
            action["authority"] = "zero_direct_support_edges"
        assert _validate_coverage_status_canonicalization_audit(
            legacy,
            payload,
            source_response_rows=[failed_row],
        ) == []
    legacy["runner_version"] = "untrusted-legacy"
    assert any(
        "policy mismatch" in error
        for error in _validate_coverage_status_canonicalization_audit(
            legacy,
            payload,
            source_response_rows=[failed_row],
        )
    )
    assert any(
        "requires the append-only source response log" in error
        for error in _validate_coverage_status_canonicalization_audit(
            recovered, payload
        )
    )
    assert any(
        "not present exactly once" in error
        for error in _validate_coverage_status_canonicalization_audit(
            recovered,
            payload,
            source_response_rows=[failed_row, deepcopy(failed_row)],
        )
    )
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {"saved-not-required": request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic",
    )
    assert strict_errors == []
    assert succeeded == {"saved-not-required"}

    candidate = _partial_evidence_candidate(payload)
    merged, _qa = _merge_contract(candidate_case=candidate, response=recovered)
    expected_spans = [item["query_span"] for item in payload["branches"]]
    _validate_contract_coverage(
        merged,
        candidate["relevant_evidence"],
        case_id=candidate["case_id"],
        expected_branch_spans=expected_spans,
    )
    assert _contract_coverage_errors(
        merged,
        candidate["relevant_evidence"],
        expected_branch_spans=expected_spans,
        expected_unanswerable=False,
    ) == []

    tampered_source = deepcopy(failed_row)
    tampered_source["prompt_sha256"] = "different"
    assert _validate_coverage_status_canonicalization_audit(
        recovered,
        payload,
        source_response_rows=[tampered_source],
    )


def _unsupported_required_concept_review() -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    review["required_concepts"]["all_of"][1]["evidence_labels"] = []
    review["evidence_concept_coverage"][0]["required_concept_keys"] = ["c01"]
    review["review_notes"] = [
        "B2要求的区别性结论在E1/E2中均无直接支持，必须作为数据阻塞处理。"
    ]
    return review, payload


def test_legacy_coverage_audit_cannot_claim_new_constraint_only_semantics() -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = "not_required"
    review["critical_constraints"]["identifiers"].append(
        {"value": "SSID", "evidence_labels": ["E2"]}
    )
    request_row = {
        "custom_id": "constraint-only-legacy-forgery",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="constraint-only-legacy-forgery",
        prompt_sha256="prompt",
        reviews=[review],
        payload=payload,
    )
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    forged = deepcopy(recovered)
    forged["runner_version"] = "2.19.0-native-block-zero-support-status-alias"
    audit = forged["same_response_coverage_status_canonicalization"]
    audit["policy"] = "same-response-zero-support-status-alias-to-uncovered-v3"
    for action in audit["actions"]:
        action["authority"] = "zero_direct_support_edges"

    errors = _validate_coverage_status_canonicalization_audit(
        forged,
        payload,
        source_response_rows=[failed_row],
    )

    assert any("legacy" in error and "constraint" in error for error in errors)


def test_native_block_can_canonicalize_zero_support_not_covered_alias() -> None:
    review, payload = _unsupported_required_concept_review()
    review["verdict"] = "block"
    review["evidence_concept_coverage"][1]["status"] = "not_covered"
    review["critical_constraints"]["identifiers"].append(
        {"value": "BLOCK-ID", "evidence_labels": ["E2"]}
    )

    projected, actions, raw_errors = _canonicalize_same_response_coverage_statuses(
        review, payload
    )

    assert raw_errors == [
        "evidence_concept_coverage[1].status must be one of "
        "['ambiguous', 'covered', 'insufficient', 'uncovered']"
    ]
    assert actions == [
        {
            "coverage_index": 1,
            "evidence_label": "E2",
            "field": "status",
            "before": "not_covered",
            "after": "uncovered",
            "required_concept_keys": [],
            "authority": "zero_direct_concept_support_edges",
        }
    ]
    assert projected["verdict"] == "block"
    assert projected["required_concepts"] == review["required_concepts"]
    assert projected["evidence_concept_coverage"][1]["status"] == "uncovered"
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["evidence_concept_coverage"][1].update(
            required_concept_keys=["c01"]
        ),
        lambda review: review["required_concepts"]["all_of"][0][
            "evidence_labels"
        ].append("E2"),
        lambda review: review.update(verdict="revise"),
    ],
)
def test_native_block_status_alias_rejects_support_edges_or_nonblock_gap(
    mutator,
) -> None:
    review, payload = _unsupported_required_concept_review()
    review["verdict"] = "block"
    review["evidence_concept_coverage"][1]["status"] = "not_covered"
    mutator(review)

    projected, actions, _raw_errors = _canonicalize_same_response_coverage_statuses(
        review, payload
    )

    assert projected == review
    assert actions == []


def _semantic_quota_failed_row(
    *,
    custom_id: str,
    prompt_sha256: str,
    reviews: list[dict],
    payload: dict,
    runner_version: str = "2.22.0-saved-quote-fragment-inverse-composition",
) -> dict:
    invalid_attempts = []
    failure_attempts = []
    attempt_usage = []
    for attempt, review in enumerate(reviews, start=1):
        review_hash = _sha256_json(review)
        raw_errors = [
            *_validate_review(review),
            *_validate_against_payload(review, payload),
        ]
        invalid_attempts.append(
            {
                "attempt": attempt,
                "review": review,
                "review_sha256": review_hash,
                "validation_errors": raw_errors,
                "sse_integrity": {
                    "termination": "done_sentinel",
                    "done_seen": True,
                    "finish_reason": "stop",
                    "finish_reason_stop_seen": True,
                    "event_integrity": "passed",
                    "usage_integrity": {"status": "passed"},
                    "review_validation": "failed",
                    "content_sha256": _sha256_json(
                        {"attempt": attempt, "review": review}
                    ),
                },
            }
        )
        failure_attempts.append(
            {
                "attempt": attempt,
                "termination": "review_validation_error",
                "failure_category": "semantic",
                "response_sha256": review_hash,
                "response_hash_status": "complete_json_sha256",
            }
        )
        attempt_usage.append(
            {
                "attempt": attempt,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        )
    count = len(reviews)
    return {
        "custom_id": custom_id,
        "status": "failed",
        "prompt_sha256": prompt_sha256,
        "attempt": count,
        "model": "test-model",
        "runner_version": runner_version,
        "completed_at": "2026-08-11T00:00:00+00:00",
        "failure_termination": "review_validation_error",
        "retry_accounting": {
            "policy": review_contracts_module.RETRY_ACCOUNTING_POLICY,
            "semantic_failure_quota": count,
            "semantic_failure_count": count,
            "transport_failure_quota": count,
            "transport_failure_count": 0,
            "hard_total_attempt_quota": count * 2,
            "total_attempts": count,
            "quota_exhausted": "semantic",
        },
        "invalid_review": reviews[-1],
        "invalid_attempts": invalid_attempts,
        "failure_attempts": failure_attempts,
        "attempt_usage": attempt_usage,
        "usage": {
            "prompt_tokens": 10 * count,
            "completion_tokens": 20 * count,
            "total_tokens": 30 * count,
        },
    }


def _saved_redundant_field_review(
    *, any_of_suffix: str = "", include_identifier: bool = True
) -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    review["required_concepts"]["all_of"].append(
        {
            "concept_key": "c03",
            "canonical_text": "直接支持答案",
        }
    )
    review["required_branches"][1]["required_concept_keys"].append("c03")
    review["evidence_concept_coverage"][0]["required_concept_keys"].append("c03")
    review["evidence_concept_coverage"][1]["status"] = "not_required"
    if any_of_suffix:
        review["required_concepts"]["all_of"][0]["any_of"].append(any_of_suffix)
    if include_identifier:
        review["critical_constraints"]["identifiers"].append(
            {"value": "SYNTH-ID", "evidence_labels": ["E1"]}
        )
    return review, payload


def _raw_provenance_semantic_quota_failed_row(
    *, custom_id: str, prompt_sha256: str, reviews: list[dict], payload: dict
) -> dict:
    invalid_attempts = []
    failure_attempts = []
    attempt_usage = []
    for attempt, raw_review in enumerate(reviews, start=1):
        raw_review = deepcopy(raw_review)
        pre_normalization = deepcopy(raw_review)
        normalized, actions = _normalize_review_structure(
            pre_normalization, payload
        )
        validation_errors = [*_validate_review(normalized)]
        if actions:
            validation_errors.append(
                "strict runner rejects structural normalization: "
                f"{len(actions)} action(s)"
            )
        validation_errors.extend(_validate_against_payload(normalized, payload))
        raw_hash = _sha256_json(raw_review)
        pre_hash = _sha256_json(pre_normalization)
        normalized_hash = _sha256_json(normalized)
        invalid_attempts.append(
            {
                "attempt": attempt,
                "raw_review": raw_review,
                "raw_review_sha256": raw_hash,
                "pre_normalization_review_sha256": pre_hash,
                "review": normalized,
                "review_sha256": normalized_hash,
                "normalization_provenance": {
                    "policy": review_contracts_module.RAW_NORMALIZATION_PROVENANCE_POLICY,
                    "raw_review_sha256": raw_hash,
                    "pre_normalization_review_sha256": pre_hash,
                    "normalized_review_sha256": normalized_hash,
                    "actions": actions,
                },
                "validation_errors": validation_errors,
                "sse_integrity": {
                    "attempt": attempt,
                    "termination": "done_sentinel",
                    "done_seen": True,
                    "finish_reason": "stop",
                    "finish_reason_stop_seen": True,
                    "event_integrity": "passed",
                    "usage_integrity": {"status": "passed"},
                    "review_validation": "failed",
                    "content_sha256": _sha256_json(
                        {"attempt": attempt, "review": raw_review}
                    ),
                },
            }
        )
        failure_attempts.append(
            {
                "attempt": attempt,
                "termination": "review_validation_error",
                "failure_category": "semantic",
                "response_sha256": raw_hash,
                "response_hash_status": "complete_json_sha256",
            }
        )
        attempt_usage.append(
            {
                "attempt": attempt,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        )
    count = len(reviews)
    sse_integrity_attempts = [
        deepcopy(item["sse_integrity"]) for item in invalid_attempts
    ]
    return {
        "custom_id": custom_id,
        "status": "failed",
        "prompt_sha256": prompt_sha256,
        "attempt": count,
        "sequence": 122,
        "model": "test-model",
        "runner_version": review_contracts_module.RUNNER_VERSION,
        "completed_at": "2026-08-11T00:00:00+00:00",
        "failure_termination": "review_validation_error",
        "retry_accounting": {
            "policy": review_contracts_module.RETRY_ACCOUNTING_POLICY,
            "semantic_failure_quota": count,
            "semantic_failure_count": count,
            "transport_failure_quota": count,
            "transport_failure_count": 0,
            "hard_total_attempt_quota": count * 2,
            "total_attempts": count,
            "quota_exhausted": "semantic",
        },
        "invalid_review": invalid_attempts[-1]["review"],
        "invalid_attempts": invalid_attempts,
        "sse_integrity": deepcopy(sse_integrity_attempts[-1]),
        "sse_integrity_attempts": sse_integrity_attempts,
        "failure_attempts": failure_attempts,
        "attempt_usage": attempt_usage,
        "usage": {
            "prompt_tokens": 10 * count,
            "completion_tokens": 20 * count,
            "total_tokens": 30 * count,
        },
        "usage_is_lower_bound": False,
    }


def test_saved_redundant_field_format_recovery_selects_final_attempt_only() -> None:
    attempt1, payload = _saved_redundant_field_review(
        any_of_suffix="较早响应的同义变体"
    )
    attempt2, _ = _saved_redundant_field_review(
        any_of_suffix="中间响应的同义变体"
    )
    attempt3, _ = _saved_redundant_field_review(include_identifier=False)
    request_row = {
        "custom_id": "saved-redundant-format-recovery",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[attempt1, attempt2, attempt3],
        payload=payload,
    )

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    audit = recovered["saved_response_redundant_field_coverage_composition"]
    assert audit["selection"]["selected_attempt"] == 3
    assert audit["selection"]["semantic_consensus_claimed"] is False
    assert audit["selection"]["cross_attempt_merge"] is False
    assert audit["attempt_shape_receipts"][0]["projected_review_sha256"] != (
        audit["attempt_shape_receipts"][2]["projected_review_sha256"]
    )
    assert recovered["review"]["critical_constraints"]["identifiers"] == []
    assert "较早响应的同义变体" not in json.dumps(
        recovered["review"], ensure_ascii=False
    )
    validate_audit = getattr(
        review_contracts_module,
        "_validate_redundant_field_coverage_composition_audit",
    )
    assert validate_audit(
        recovered, payload, source_response_rows=[failed_row]
    ) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, _payload: review["required_concepts"]["all_of"][2].update(
            any_of=[]
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][2].update(
            evidence_labels=None
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][2].update(
            branch_labels="B2"
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][2].update(
            canonical_text=""
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][2].update(
            schema_extra="forbidden"
        ),
        lambda review, _payload: review["required_branches"][1].update(
            required_concept_keys=[]
        ),
        lambda review, _payload: review["required_branches"][1].update(
            query_span="截断"
        ),
        lambda review, _payload: review["evidence_concept_coverage"][0].update(
            status="uncovered", required_concept_keys=[]
        ),
        lambda review, _payload: review["evidence_concept_coverage"][1].update(
            status="irrelevant"
        ),
        lambda review, _payload: review.update(verdict="approve"),
        lambda review, _payload: review.update(answerability="abstain"),
    ],
)
def test_saved_redundant_field_projector_rejects_nonredundant_or_extra_defect(
    mutator,
) -> None:
    review, payload = _saved_redundant_field_review()
    mutator(review, payload)
    projector = getattr(
        review_contracts_module,
        "_canonicalize_saved_redundant_field_coverage_composition",
    )

    projected, stages, _raw_errors = projector(review, payload)

    assert projected == review
    assert stages == []


def test_saved_redundant_field_recovery_requires_persisted_consistent_shape() -> None:
    review, payload = _saved_redundant_field_review()
    different_shape = deepcopy(review)
    different_shape["required_concepts"]["all_of"][2]["any_of"] = [
        "直接支持答案"
    ]
    request_row = {
        "custom_id": "saved-redundant-shape-gate",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[review, different_shape],
        payload=payload,
    )
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    ) is None

    single = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[review],
        payload=payload,
    )
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[single]
    ) is None

    synthetic_live = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[review, review],
        payload=payload,
    )
    synthetic_live.pop("sequence")
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[synthetic_live]
    ) is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda row: row["retry_accounting"].update(policy="forged"),
        lambda row: row["retry_accounting"].update(
            semantic_failure_count=1
        ),
        lambda row: row["retry_accounting"].update(
            semantic_failure_quota=3
        ),
        lambda row: row["retry_accounting"].update(
            transport_failure_quota=3
        ),
        lambda row: row["retry_accounting"].update(
            hard_total_attempt_quota=5
        ),
        lambda row: row["sse_integrity_attempts"].pop(),
        lambda row: row["sse_integrity"].update(attempt=1),
        lambda row: row.update(invalid_review={}),
        lambda row: row["invalid_attempts"][-1]["sse_integrity"].update(
            content_sha256="0" * 64
        ),
        lambda row: row.update(usage_is_lower_bound=True),
    ],
)
def test_saved_redundant_field_recovery_rejects_ledger_tamper(tamper) -> None:
    review, payload = _saved_redundant_field_review()
    request_row = {
        "custom_id": "saved-redundant-ledger",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[review, review],
        payload=payload,
    )
    tamper(failed_row)

    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    ) is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["stages"].reverse(),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["stages"][0].update(output_review_sha256="0" * 64),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["stages"][0]["actions"][0].update(json_path="/verdict"),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["selection"].update(selected_attempt=1),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["selection"].update(semantic_consensus_claimed=True),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["selection"].update(cross_attempt_merge=True),
        lambda response: response["review"]["required_concepts"]["all_of"][0][
            "any_of"
        ].append("跨attempt拼入的变体"),
        lambda response: response[
            "saved_response_redundant_field_coverage_composition"
        ]["source"].update(response_row_sha256="0" * 64),
        lambda response: response.update(usage_is_lower_bound=True),
        lambda response: response.update(
            successful_response_usage={"total_tokens": 1}
        ),
    ],
)
def test_saved_redundant_field_audit_rejects_tamper(tamper) -> None:
    review, payload = _saved_redundant_field_review()
    request_row = {
        "custom_id": "saved-redundant-audit",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[review, review],
        payload=payload,
    )
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    tamper(recovered)
    validate_audit = getattr(
        review_contracts_module,
        "_validate_redundant_field_coverage_composition_audit",
    )

    assert validate_audit(
        recovered, payload, source_response_rows=[failed_row]
    )


def _exact_payload_quote_prefix_review(
    *, opener: str = "“", closer: str = "”"
) -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    prefix = "如果用户未勾选"
    payload["branches"][1]["query_span"] = (
        f"{prefix}{opener}显示密码{closer}，输入的密码将如何显示"
    )
    review["required_branches"][1]["query_span"] = prefix
    return review, payload


@pytest.mark.parametrize(
    ("opener", "closer"),
    [("\"", "\""), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")],
)
def test_exact_payload_quote_prefix_projector_is_finite_and_changes_only_span(
    opener: str, closer: str
) -> None:
    review, payload = _exact_payload_quote_prefix_review(
        opener=opener, closer=closer
    )
    projector = getattr(
        review_contracts_module,
        "_canonicalize_saved_exact_payload_branch_span_quote_prefix",
    )

    projected, stages, raw_errors = projector(review, payload)

    expected_span = payload["branches"][1]["query_span"]
    assert raw_errors == [
        "branch query_span differs from payload for B2",
        "required branch labels/query spans must exactly match payload",
    ]
    assert len(stages) == 1
    assert stages[0]["stage"] == "exact_payload_branch_span_quote_prefix_recovery"
    assert stages[0]["input_review_sha256"] == _sha256_json(review)
    assert stages[0]["output_review_sha256"] == _sha256_json(projected)
    assert stages[0]["actions"] == [
        {
            "operation": "replace",
            "json_path": "/required_branches/1/query_span",
            "before": "如果用户未勾选",
            "after": expected_span,
            "authority": "same_request_payload_exact_branch_span",
            "quote_pair": [opener, closer],
        }
    ]
    unchanged = deepcopy(projected)
    unchanged["required_branches"][1]["query_span"] = "如果用户未勾选"
    assert unchanged == review
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, _payload: review["required_branches"][1].update(
            query_span=""
        ),
        lambda review, _payload: review["required_branches"][1].update(
            query_span="如果用户"
        ),
        lambda review, _payload: review["required_branches"][1].update(
            query_span="不是payload前缀"
        ),
        lambda review, payload: payload["branches"][1].update(
            query_span="如果用户未勾选“显示密码，输入的密码将如何显示"
        ),
        lambda review, payload: payload["branches"][1].update(
            query_span="如果用户未勾选“显示”密码”，输入的密码将如何显示"
        ),
        lambda review, payload: payload["branches"][1].update(
            query_span="如果用户未勾选“”，输入的密码将如何显示"
        ),
        lambda review, payload: payload["branches"][1].update(
            query_span="如果用户未勾选“显示密码”"
        ),
        lambda review, _payload: review["required_branches"][0].update(
            query_span="出纸托盘"
        ),
        lambda review, _payload: review["required_branches"][1].update(
            required_concept_keys=[]
        ),
        lambda review, _payload: review["required_branches"].reverse(),
        lambda review, _payload: review["required_concepts"]["all_of"][0].update(
            canonical_text=""
        ),
        lambda review, _payload: review.update(verdict="approve"),
        lambda review, _payload: review.update(answerability="abstain"),
    ],
)
def test_exact_payload_quote_prefix_projector_rejects_every_broader_shape(
    mutator,
) -> None:
    review, payload = _exact_payload_quote_prefix_review()
    mutator(review, payload)
    projector = getattr(
        review_contracts_module,
        "_canonicalize_saved_exact_payload_branch_span_quote_prefix",
    )

    projected, stages, _raw_errors = projector(review, payload)

    assert projected == review
    assert stages == []


def _exact_payload_quote_prefix_recovery_fixture() -> tuple[dict, dict, dict]:
    final_review, payload = _exact_payload_quote_prefix_review()
    structural1 = deepcopy(final_review)
    structural2 = deepcopy(final_review)
    for structural in (structural1, structural2):
        concept = structural["required_concepts"]["all_of"][1]
        concept.pop("any_of")
        concept.pop("evidence_labels")
        concept.pop("branch_labels")
    request_row = {
        "custom_id": "saved-exact-payload-quote-prefix",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _raw_provenance_semantic_quota_failed_row(
        custom_id=request_row["custom_id"],
        prompt_sha256="prompt",
        reviews=[structural1, structural2, final_review],
        payload=payload,
    )
    failed_row["sequence"] = 123
    failed_row["runner_version"] = (
        "2.24.0-saved-redundant-field-format-recovery"
    )
    return request_row, payload, failed_row


def test_exact_payload_quote_prefix_recovery_selects_only_final_attempt() -> None:
    request_row, payload, failed_row = _exact_payload_quote_prefix_recovery_fixture()

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    audit = recovered[
        "saved_response_exact_payload_branch_span_quote_prefix_recovery"
    ]
    assert audit["selection"] == {
        "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
        "selected_attempt": 3,
        "semantic_consensus_claimed": False,
        "cross_attempt_merge": False,
    }
    assert audit["source"]["source_attempt"] == 3
    assert recovered["review"]["required_branches"][1]["query_span"] == (
        payload["branches"][1]["query_span"]
    )
    validator = getattr(
        review_contracts_module,
        "_validate_exact_payload_branch_span_quote_prefix_recovery_audit",
    )
    assert validator(
        recovered, payload, source_response_rows=[failed_row]
    ) == []
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {request_row["custom_id"]: request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic-exact-payload-quote-prefix",
    )
    assert strict_errors == []
    assert succeeded == {request_row["custom_id"]}


def test_exact_payload_quote_prefix_recovery_rejects_nonfinal_and_quota_bypass() -> None:
    request_row, payload, failed_row = _exact_payload_quote_prefix_recovery_fixture()
    final_valid = deepcopy(failed_row)
    final_valid["invalid_attempts"][2] = deepcopy(
        final_valid["invalid_attempts"][1]
    )
    final_valid["invalid_attempts"][2]["attempt"] = 3
    final_valid["invalid_attempts"][2]["sse_integrity"]["attempt"] = 3
    final_valid["invalid_attempts"][2]["sse_integrity"]["content_sha256"] = (
        _sha256_json({"attempt": 3, "structural": True})
    )
    final_valid["sse_integrity_attempts"][2] = deepcopy(
        final_valid["invalid_attempts"][2]["sse_integrity"]
    )
    final_valid["sse_integrity"] = deepcopy(final_valid["sse_integrity_attempts"][2])
    final_valid["invalid_review"] = deepcopy(final_valid["invalid_attempts"][2]["review"])
    final_valid["failure_attempts"][2]["response_sha256"] = final_valid[
        "invalid_attempts"
    ][2]["raw_review_sha256"]
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[final_valid]
    ) is None

    single = deepcopy(failed_row)
    for key in ("invalid_attempts", "sse_integrity_attempts", "failure_attempts", "attempt_usage"):
        single[key] = [deepcopy(single[key][-1])]
        single[key][0]["attempt"] = 1
    single["invalid_attempts"][0]["sse_integrity"]["attempt"] = 1
    single["sse_integrity_attempts"][0]["attempt"] = 1
    single["sse_integrity"] = deepcopy(single["sse_integrity_attempts"][0])
    single["invalid_review"] = deepcopy(single["invalid_attempts"][0]["review"])
    single["attempt"] = 1
    single["retry_accounting"].update(
        semantic_failure_quota=1,
        semantic_failure_count=1,
        transport_failure_quota=1,
        hard_total_attempt_quota=2,
        total_attempts=1,
    )
    single["usage"] = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[single]
    ) is None

    legacy = deepcopy(failed_row)
    legacy["runner_version"] = (
        "2.23.0-constraint-evidence-separation-raw-provenance"
    )
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[legacy]
    ) is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ].update(policy="forged"),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["stages"][0].update(input_review_sha256="0" * 64),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["stages"][0]["actions"][0].update(json_path="/verdict"),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["stages"][0]["actions"][0].update(after="tampered"),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["selection"].update(selected_attempt=2),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["selection"].update(cross_attempt_merge=True),
        lambda response: response[
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        ]["source"].update(response_row_sha256="0" * 64),
        lambda response: response["review"]["required_branches"][1].update(
            required_concept_keys=[]
        ),
        lambda response: response.update(usage_is_lower_bound=True),
        lambda response: response.update(attempt_usage=[{"total_tokens": 1}]),
    ],
)
def test_exact_payload_quote_prefix_audit_rejects_tamper(tamper) -> None:
    request_row, payload, failed_row = _exact_payload_quote_prefix_recovery_fixture()
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    tamper(recovered)
    validator = getattr(
        review_contracts_module,
        "_validate_exact_payload_branch_span_quote_prefix_recovery_audit",
    )

    assert validator(recovered, payload, source_response_rows=[failed_row])


def _nonsemantic_composition_review() -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = "not_required"
    review["required_branches"].append(
        {
            "branch_label": "B3",
            "query_span": "请结合证据中的注释内容回答",
            "required_concept_keys": ["c01", "c02"],
        }
    )
    payload["branches"].append(
        {
            "branch_label": "B3",
            "query_span": "请结合证据中的注释内容回答",
        }
    )
    return review, payload


def test_saved_nonsemantic_composition_changes_only_disjoint_redundant_fields() -> None:
    review, payload = _nonsemantic_composition_review()

    projected, stages, raw_errors = _canonicalize_saved_nonsemantic_composition(
        review, payload
    )

    assert raw_errors == [
        "evidence_concept_coverage[1].status must be one of "
        "['covered', 'insufficient', 'uncovered']",
        "concept.branch_labels and required_branches edges must be symmetric",
    ]
    assert [stage["stage"] for stage in stages] == [
        "coverage_status_alias",
        "inverse_branch_labels",
    ]
    assert projected["verdict"] == review["verdict"] == "revise"
    assert projected["answerability"] == review["answerability"] == "answer"
    assert projected["required_branches"] == review["required_branches"]
    assert projected["evidence_concept_coverage"][1]["status"] == "uncovered"
    assert projected["evidence_concept_coverage"][1]["required_concept_keys"] == []
    assert projected["required_concepts"]["all_of"][0]["branch_labels"] == [
        "B1",
        "B3",
    ]
    assert projected["required_concepts"]["all_of"][1]["branch_labels"] == [
        "B2",
        "B3",
    ]
    assert stages[0]["output_review_sha256"] == stages[1]["input_review_sha256"]
    assert stages[1]["output_review_sha256"] == _sha256_json(projected)
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


def test_saved_nonsemantic_composition_is_append_only_replayable() -> None:
    review, payload = _nonsemantic_composition_review()
    request_row = {
        "custom_id": "saved-nonsemantic-composition",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="saved-nonsemantic-composition",
        prompt_sha256="prompt",
        reviews=[deepcopy(review), deepcopy(review), deepcopy(review)],
        payload=payload,
    )

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    assert recovered["prior_response_recovery"]["mode"] == (
        "saved_nonsemantic_canonicalizer_composition"
    )
    audit = recovered[
        "same_response_nonsemantic_canonicalization_composition"
    ]
    assert audit["source"]["source_attempt"] == 3
    assert audit["intermediate_review_sha256"] == audit["stages"][0][
        "output_review_sha256"
    ]
    assert _validate_nonsemantic_canonicalization_composition_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    ) == []
    legacy = deepcopy(recovered)
    legacy["runner_version"] = "2.21.0-saved-nonsemantic-canonicalizer-composition"
    legacy_audit = legacy[
        "same_response_nonsemantic_canonicalization_composition"
    ]
    legacy_audit["policy"] = (
        "saved-response-coverage-status-then-inverse-branch-labels-v1"
    )
    legacy_audit["stages"][0]["policy"] = (
        "same-response-zero-support-status-alias-to-uncovered-v3"
    )
    assert _validate_nonsemantic_canonicalization_composition_audit(
        legacy,
        payload,
        source_response_rows=[failed_row],
    ) == []
    legacy["runner_version"] = "untrusted-legacy"
    assert _validate_nonsemantic_canonicalization_composition_audit(
        legacy,
        payload,
        source_response_rows=[failed_row],
    )
    assert _validate_nonsemantic_canonicalization_composition_audit(
        recovered, payload
    )
    assert _validate_nonsemantic_canonicalization_composition_audit(
        recovered,
        payload,
        source_response_rows=[failed_row, deepcopy(failed_row)],
    )
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {"saved-nonsemantic-composition": request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic-composition",
    )
    assert strict_errors == []
    assert succeeded == {"saved-nonsemantic-composition"}

    incomplete_source = deepcopy(failed_row)
    incomplete_source.pop("retry_accounting")
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[incomplete_source]
    ) is None

    for tampered in (
        lambda value: value[
            "same_response_nonsemantic_canonicalization_composition"
        ]["stages"].reverse(),
        lambda value: value[
            "same_response_nonsemantic_canonicalization_composition"
        ]["stages"][0].update(output_review_sha256="0" * 64),
        lambda value: value[
            "same_response_nonsemantic_canonicalization_composition"
        ]["stages"][0]["actions"][0].update(json_path="/verdict"),
        lambda value: value["review"].update(verdict="block"),
    ):
        changed = deepcopy(recovered)
        tampered(changed)
        assert _validate_nonsemantic_canonicalization_composition_audit(
            changed,
            payload,
            source_response_rows=[failed_row],
        )


def test_legacy_composition_cannot_claim_new_constraint_only_semantics() -> None:
    review, payload = _nonsemantic_composition_review()
    review["critical_constraints"]["identifiers"].append(
        {"value": "SSID", "evidence_labels": ["E2"]}
    )
    request_row = {
        "custom_id": "constraint-only-composition-forgery",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="constraint-only-composition-forgery",
        prompt_sha256="prompt",
        reviews=[review],
        payload=payload,
    )
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    forged = deepcopy(recovered)
    forged["runner_version"] = "2.21.0-saved-nonsemantic-canonicalizer-composition"
    audit = forged["same_response_nonsemantic_canonicalization_composition"]
    audit["policy"] = (
        "saved-response-coverage-status-then-inverse-branch-labels-v1"
    )
    audit["stages"][0]["policy"] = (
        "same-response-zero-support-status-alias-to-uncovered-v3"
    )

    errors = _validate_nonsemantic_canonicalization_composition_audit(
        forged,
        payload,
        source_response_rows=[failed_row],
    )

    assert any("legacy" in error and "constraint" in error for error in errors)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, _payload: review["evidence_concept_coverage"][1].update(
            status="irrelevant"
        ),
        lambda review, _payload: (
            review["evidence_concept_coverage"][1].update(
                required_concept_keys=["c01"]
            ),
            review["required_concepts"]["all_of"][0]["evidence_labels"].append(
                "E2"
            ),
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][0][
            "evidence_labels"
        ].append("E2"),
        lambda review, _payload: review["required_branches"][2].update(
            required_concept_keys=["c99"]
        ),
        lambda review, _payload: review["required_branches"][2].update(
            query_span="截断"
        ),
        lambda review, _payload: review["required_branches"][2].update(
            required_concept_keys=[]
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][0].update(
            canonical_text=""
        ),
        lambda review, _payload: review.update(verdict="block"),
        lambda review, _payload: review["required_concepts"]["all_of"][1].update(
            canonical_text="请说明", any_of=["列出全部"]
        ),
    ],
)
def test_saved_nonsemantic_composition_rejects_any_expansion_or_extra_error(
    mutator,
) -> None:
    review, payload = _nonsemantic_composition_review()
    mutator(review, payload)

    projected, stages, _raw_errors = _canonicalize_saved_nonsemantic_composition(
        review, payload
    )

    assert projected == review
    assert stages == []


def _quote_fragment_inverse_review(
    *, opener: str = "“", closer: str = "”", fragment_key: str = "安装完成"
) -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    prefix = "在USB安装过程中，系统自动安装驱动后，弹出的"
    suffix = "界面会在多久后自动关闭"
    review["required_branches"][0]["query_span"] = prefix
    review["required_branches"][0][fragment_key] = suffix
    payload["branches"][0]["query_span"] = (
        f"{prefix}{opener}{fragment_key}{closer}{suffix}"
    )
    review["required_branches"].append(
        {
            "branch_label": "B3",
            "query_span": "请结合两条证据中的描述作答",
            "required_concept_keys": ["c01", "c02"],
        }
    )
    payload["branches"].append(
        {
            "branch_label": "B3",
            "query_span": "请结合两条证据中的描述作答",
        }
    )
    return review, payload


@pytest.mark.parametrize(
    ("opener", "closer"),
    [("\"", "\""), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")],
)
def test_saved_quote_fragment_inverse_composition_is_finite_and_semantic_free(
    opener: str, closer: str
) -> None:
    review, payload = _quote_fragment_inverse_review(
        opener=opener,
        closer=closer,
        fragment_key="安装/完成~状态",
    )

    projected, stages, raw_errors = (
        _canonicalize_saved_quote_fragment_inverse_composition(review, payload)
    )

    assert raw_errors == [
        "review contains schema-external fields: "
        "['$.required_branches[0].安装/完成~状态']",
        "branch query_span differs from payload for B1",
        "required branch labels/query spans must exactly match payload",
        "concept.branch_labels and required_branches edges must be symmetric",
    ]
    assert [stage["stage"] for stage in stages] == [
        "quote_fragment_branch_reconstruction",
        "inverse_branch_labels",
    ]
    assert [action["json_path"] for action in stages[0]["actions"]] == [
        "/required_branches/0/query_span",
        "/required_branches/0/安装~1完成~0状态",
    ]
    assert stages[0]["actions"][1]["operation"] == "remove"
    assert stages[0]["actions"][1]["after"] is None
    assert projected["verdict"] == review["verdict"] == "revise"
    assert projected["answerability"] == review["answerability"] == "answer"
    assert set(projected["required_branches"][0]) == {
        "branch_label",
        "query_span",
        "required_concept_keys",
    }
    assert projected["required_branches"][0]["required_concept_keys"] == ["c01"]
    assert projected["required_concepts"]["all_of"][0]["branch_labels"] == [
        "B1",
        "B3",
    ]
    assert projected["required_concepts"]["all_of"][1]["branch_labels"] == [
        "B2",
        "B3",
    ]
    assert projected["evidence_concept_coverage"] == review[
        "evidence_concept_coverage"
    ]
    assert projected["critical_constraints"] == review["critical_constraints"]
    assert stages[0]["output_review_sha256"] == stages[1][
        "input_review_sha256"
    ]
    assert stages[1]["output_review_sha256"] == _sha256_json(projected)
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


def test_saved_quote_fragment_inverse_recovery_uses_only_reconstructable_attempts() -> None:
    attempt1, payload = _quote_fragment_inverse_review()
    attempt2 = deepcopy(attempt1)
    attempt2["required_branches"][0].pop("安装完成")
    attempt3 = deepcopy(attempt1)
    request_row = {
        "custom_id": "saved-quote-fragment-composition",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="saved-quote-fragment-composition",
        prompt_sha256="prompt",
        reviews=[attempt1, attempt2, attempt3],
        payload=payload,
    )

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    assert recovered["prior_response_recovery"]["mode"] == (
        "saved_quote_fragment_inverse_canonicalization_composition"
    )
    audit = recovered[
        "same_response_quote_fragment_inverse_canonicalization_composition"
    ]
    assert audit["source"]["source_attempt"] == 3
    assert _validate_quote_fragment_inverse_composition_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    ) == []
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {"saved-quote-fragment-composition": request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic-quote-fragment-composition",
    )
    assert strict_errors == []
    assert succeeded == {"saved-quote-fragment-composition"}

    attempt3_unrecoverable = deepcopy(failed_row)
    attempt3_unrecoverable["invalid_attempts"][2] = deepcopy(
        attempt3_unrecoverable["invalid_attempts"][1]
    )
    attempt3_unrecoverable["invalid_attempts"][2]["attempt"] = 3
    attempt3_unrecoverable["invalid_attempts"][2]["sse_integrity"]["content_sha256"] = (
        _sha256_json({"attempt": 3, "review": attempt2})
    )
    attempt3_unrecoverable["failure_attempts"][2]["response_sha256"] = (
        attempt3_unrecoverable["invalid_attempts"][2]["review_sha256"]
    )
    recovered_from_first = _recover_prior_invalid_review(
        row=request_row, response_rows=[attempt3_unrecoverable]
    )
    assert recovered_from_first is not None
    assert recovered_from_first[
        "same_response_quote_fragment_inverse_canonicalization_composition"
    ]["source"]["source_attempt"] == 1

    only_fragment = _semantic_quota_failed_row(
        custom_id="saved-quote-fragment-composition",
        prompt_sha256="prompt",
        reviews=[attempt2, attempt2, attempt2],
        payload=payload,
    )
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[only_fragment]
    ) is None


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, _payload: review["required_branches"][0].update(
            another="fragment"
        ),
        lambda review, _payload: review["required_branches"][0].update(
            {"": "fragment"}
        ),
        lambda review, _payload: review["required_branches"][0].update(
            安装完成=""
        ),
        lambda review, _payload: review["required_branches"][0].update(
            branch_label="UNKNOWN"
        ),
        lambda _review, payload: payload["branches"][0].update(
            query_span="不匹配任何有限引号模板"
        ),
        lambda review, _payload: review["required_concepts"]["all_of"][0].update(
            canonical_text=""
        ),
        lambda review, _payload: review.update(verdict="block"),
        lambda review, _payload: review["required_branches"][0].update(
            required_concept_keys=["c99"]
        ),
        lambda review, _payload: review["evidence_concept_coverage"][0].update(
            status="ambiguous"
        ),
        lambda review, _payload: review["critical_constraints"]["identifiers"].append(
            {"value": "EXTRA", "evidence_labels": ["E99"]}
        ),
    ],
)
def test_saved_quote_fragment_inverse_composition_rejects_any_extra_error(
    mutator,
) -> None:
    review, payload = _quote_fragment_inverse_review()
    mutator(review, payload)

    projected, stages, _raw_errors = (
        _canonicalize_saved_quote_fragment_inverse_composition(review, payload)
    )

    assert projected == review
    assert stages == []


def test_saved_quote_fragment_reconstruction_requires_unique_quote_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review, payload = _quote_fragment_inverse_review()
    monkeypatch.setattr(
        review_contracts_module,
        "QUOTE_FRAGMENT_PAIRS",
        (("“", "”"), ("“", "”")),
    )

    projected, stages, _raw_errors = (
        _canonicalize_saved_quote_fragment_inverse_composition(review, payload)
    )

    assert projected == review
    assert stages == []


@pytest.mark.parametrize(
    "tamper",
    [
        lambda response: response[
            "same_response_quote_fragment_inverse_canonicalization_composition"
        ]["stages"].reverse(),
        lambda response: response[
            "same_response_quote_fragment_inverse_canonicalization_composition"
        ]["stages"][0].update(output_review_sha256="0" * 64),
        lambda response: response[
            "same_response_quote_fragment_inverse_canonicalization_composition"
        ]["stages"][0]["actions"][0].update(json_path="/verdict"),
        lambda response: response[
            "same_response_quote_fragment_inverse_canonicalization_composition"
        ]["source"].update(source_attempt=2),
        lambda response: response[
            "same_response_quote_fragment_inverse_canonicalization_composition"
        ]["source"].update(response_row_sha256="0" * 64),
        lambda response: response["review"].update(verdict="block"),
        lambda response: response["review"].update(answerability="abstain"),
        lambda response: response["review"]["required_branches"][0].update(
            branch_label="B2"
        ),
        lambda response: response["review"]["required_branches"][0].update(
            required_concept_keys=["c02"]
        ),
        lambda response: response["review"]["required_concepts"]["all_of"][0].update(
            canonical_text="tampered"
        ),
        lambda response: response["review"]["evidence_concept_coverage"][0].update(
            status="uncovered"
        ),
        lambda response: response["review"]["critical_constraints"]["identifiers"].append(
            {"value": "tampered", "evidence_labels": ["E1"]}
        ),
    ],
)
def test_saved_quote_fragment_inverse_audit_rejects_every_tamper(tamper) -> None:
    review, payload = _quote_fragment_inverse_review()
    request_row = {
        "custom_id": "saved-quote-fragment-audit",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="saved-quote-fragment-audit",
        prompt_sha256="prompt",
        reviews=[review, review, review],
        payload=payload,
    )
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    tamper(recovered)

    assert _validate_quote_fragment_inverse_composition_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    )


def test_single_nonsemantic_projector_still_uses_legacy_single_audit_path() -> None:
    coverage_only, payload = _partial_evidence_review()
    coverage_only["evidence_concept_coverage"][1]["status"] = "not_required"
    projected, stages, _errors = _canonicalize_saved_nonsemantic_composition(
        coverage_only, payload
    )
    assert projected == coverage_only
    assert stages == []

    branch_only, branch_payload = _nonsemantic_composition_review()
    branch_only["evidence_concept_coverage"][1]["status"] = "uncovered"
    projected, stages, _errors = _canonicalize_saved_nonsemantic_composition(
        branch_only, branch_payload
    )
    assert projected == branch_only
    assert stages == []


def test_unsupported_required_concept_changes_only_revise_to_data_block() -> None:
    review, payload = _unsupported_required_concept_review()

    projected, actions, raw_errors = (
        _canonicalize_same_response_unsupported_required_concept_block(
            review, payload
        )
    )

    assert raw_errors == ["concept[1].evidence_labels must be non-empty"]
    assert actions == [
        {
            "field": "verdict",
            "before": "revise",
            "after": "block",
            "unsupported_concept_keys": ["c02"],
            "unsupported_branch_labels": ["B2"],
            "authority": "required_branch_has_no_directly_supported_concept",
        }
    ]
    expected = deepcopy(review)
    expected["verdict"] = "block"
    assert projected == expected
    assert projected["required_concepts"] == review["required_concepts"]
    assert projected["required_branches"] == review["required_branches"]
    assert projected["evidence_concept_coverage"] == review[
        "evidence_concept_coverage"
    ]
    assert projected["review_notes"] == review["review_notes"]
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


def test_all_of_unsupported_concept_blocks_with_supported_sibling_and_constraints() -> None:
    review, payload = _unsupported_required_concept_review()
    review["required_branches"][1]["required_concept_keys"] = ["c01", "c02"]
    review["required_concepts"]["all_of"][0]["branch_labels"].append("B2")
    review["critical_constraints"]["identifiers"].append(
        {"value": "出纸托盘", "evidence_labels": ["E1"]}
    )

    projected, actions, raw_errors = (
        _canonicalize_same_response_unsupported_required_concept_block(
            review, payload
        )
    )

    assert raw_errors == ["concept[1].evidence_labels must be non-empty"]
    assert projected["verdict"] == "block"
    assert actions[0]["unsupported_concept_keys"] == ["c02"]
    assert actions[0]["unsupported_branch_labels"] == ["B2"]


@pytest.mark.parametrize(
    ("constraint_kind", "value", "unit", "target", "expect_block"),
    [
        ("numerics", "60", "页", "60页", False),
        ("numerics", "60", "页", "约60页", False),
        ("numerics", "60", "页", "60页容量不足", True),
        ("identifiers", "TL-413H", None, "TL-413H", False),
        ("identifiers", "TL-413H", None, "TL-413H不兼容", True),
        ("numerics", "2000", None, "0", True),
        ("identifiers", "Android4.40", None, "Android4.4", True),
        ("identifiers", "192.168.0.102", None, "192.168.0.1", True),
    ],
)
def test_unsupported_target_requires_complete_directional_constraint_support(
    constraint_kind: str,
    value: str,
    unit: str | None,
    target: str,
    expect_block: bool,
) -> None:
    review, payload = _unsupported_required_concept_review()
    target_concept = review["required_concepts"]["all_of"][1]
    target_concept["canonical_text"] = target
    target_concept["any_of"] = [target]
    constraint = {"value": value, "evidence_labels": ["E1"]}
    if constraint_kind == "numerics":
        constraint["unit"] = unit
    review["critical_constraints"][constraint_kind].append(constraint)

    projected, actions, _raw_errors = (
        _canonicalize_same_response_unsupported_required_concept_block(
            review, payload
        )
    )

    assert bool(actions) is expect_block
    assert projected["verdict"] == ("block" if expect_block else "revise")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: (
            review["required_branches"][1].update(required_concept_keys=[]),
            review["required_concepts"]["all_of"][1].update(branch_labels=[]),
        ),
        lambda review: review["evidence_concept_coverage"][0][
            "required_concept_keys"
        ].append("c02"),
        lambda review: review["required_branches"][0].update(query_span="截断"),
        lambda review: review.update(verdict="block"),
        lambda review: review.update(verdict="approve"),
        lambda review: review.update(answerability="abstain"),
        lambda review: review.update(
            required_concepts={
                "any_of": review["required_concepts"]["all_of"]
            }
        ),
    ],
)
def test_unsupported_required_concept_block_rejects_partial_or_ambiguous_cases(
    mutator,
) -> None:
    review, payload = _unsupported_required_concept_review()
    mutator(review)

    projected, actions, _raw_errors = (
        _canonicalize_same_response_unsupported_required_concept_block(
            review, payload
        )
    )

    assert projected == review
    assert actions == []


def test_saved_unsupported_required_concept_block_is_replayable_but_not_runnable() -> None:
    review, payload = _unsupported_required_concept_review()
    request_row = {
        "custom_id": "saved-unsupported-required",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="saved-unsupported-required",
        prompt_sha256="prompt",
        reviews=[deepcopy(review), deepcopy(review), deepcopy(review)],
        payload=payload,
    )

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    expected = deepcopy(review)
    expected["verdict"] = "block"
    assert recovered["review"] == expected
    audit = recovered["same_response_unsupported_required_concept_block"]
    assert audit["api_call_performed"] is False
    assert audit["raw_review_sha256"] == _sha256_json(review)
    assert audit["consensus"]["response_count"] == 3
    assert [item["source_attempt"] for item in audit["consensus"]["sources"]] == [
        1,
        2,
        3,
    ]
    assert _validate_unsupported_required_concept_block_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    ) == []
    legacy = deepcopy(recovered)
    legacy["runner_version"] = "2.16.0-unsupported-required-concept-block"
    legacy["same_response_unsupported_required_concept_block"]["policy"] = (
        "same-response-unsupported-required-concept-revise-to-block-v1"
    )
    legacy["same_response_unsupported_required_concept_block"].pop("consensus")
    assert _validate_unsupported_required_concept_block_audit(
        legacy,
        payload,
        source_response_rows=[failed_row],
    ) == []
    legacy["runner_version"] = "untrusted-legacy"
    assert any(
        "consensus audit mismatch" in error
        for error in _validate_unsupported_required_concept_block_audit(
            legacy,
            payload,
            source_response_rows=[failed_row],
        )
    )
    assert any(
        "requires the append-only source response log" in error
        for error in _validate_unsupported_required_concept_block_audit(
            recovered, payload
        )
    )
    assert any(
        "not present exactly once" in error
        for error in _validate_unsupported_required_concept_block_audit(
            recovered,
            payload,
            source_response_rows=[failed_row, deepcopy(failed_row)],
        )
    )
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {"saved-unsupported-required": request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic",
    )
    assert strict_errors == []
    assert succeeded == {"saved-unsupported-required"}

    candidate = _partial_evidence_candidate(payload)
    contract, qa = _merge_contract(candidate_case=candidate, response=recovered)
    assert contract["contract_status"] == "llm_reviewed_blocked"
    assert qa["verdict"] == "block"
    with pytest.raises(RuntimeError):
        _validate_contract_coverage(
            contract,
            candidate["relevant_evidence"],
            case_id=candidate["case_id"],
            expected_branch_spans=[
                item["query_span"] for item in payload["branches"]
            ],
        )

    contract = _answer_contract()
    contract["required_concepts"]["all_of"][1]["evidence_support"] = [
        {"evidence_id": "e1"}
    ]
    contract["evidence_concept_coverage"] = [
        {
            "evidence_id": "e1",
            "required_concept_ids": ["c01", "c02"],
            "coverage_status": "covered",
        },
        {
            "evidence_id": "e2",
            "required_concept_ids": [],
            "coverage_status": "uncovered",
        },
    ]
    _validate_contract_coverage(
        contract,
        [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
        case_id="partial-evidence",
        expected_branch_spans=["分支一"],
    )
    assert (
        _contract_coverage_errors(
            contract,
            [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            expected_branch_spans=["分支一"],
            expected_unanswerable=False,
        )
        == []
    )


def test_unsupported_required_concept_never_projects_from_one_saved_response() -> None:
    review, payload = _unsupported_required_concept_review()
    request_row = {
        "custom_id": "single-unsupported-required",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="single-unsupported-required",
        prompt_sha256="prompt",
        reviews=[review],
        payload=payload,
    )

    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    ) is None


@pytest.mark.parametrize("drift", ["target_text", "branch_topology", "target_key"])
def test_unsupported_required_consensus_rejects_target_or_topology_drift(
    drift: str,
) -> None:
    review, payload = _unsupported_required_concept_review()
    reviews = [deepcopy(review), deepcopy(review), deepcopy(review)]
    changed = reviews[1]
    if drift == "target_text":
        changed["required_concepts"]["all_of"][1]["canonical_text"] = (
            "不同的无证据必需结论"
        )
    elif drift == "branch_topology":
        changed["required_branches"][1]["required_concept_keys"] = ["c01", "c02"]
        changed["required_concepts"]["all_of"][0]["branch_labels"].append("B2")
    else:
        changed["required_concepts"]["all_of"][1]["concept_key"] = "c03"
        changed["required_branches"][1]["required_concept_keys"] = ["c03"]

    request_row = {
        "custom_id": "drifted-unsupported-required",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="drifted-unsupported-required",
        prompt_sha256="prompt",
        reviews=reviews,
        payload=payload,
    )

    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    ) is None


def _recover_unsupported_with_numeric_consensus(
    middle_numerics: list[dict],
) -> dict | None:
    review, payload = _unsupported_required_concept_review()
    review["required_concepts"]["all_of"][0]["evidence_labels"].append("E2")
    review["evidence_concept_coverage"][1].update(
        status="covered", required_concept_keys=["c01"]
    )
    endpoint_numerics = [
        {"value": "60", "unit": "g/m²", "evidence_labels": ["E1", "E2"]},
        {"value": "105", "unit": "g/m²", "evidence_labels": ["E1", "E2"]},
    ]
    reviews = [deepcopy(review), deepcopy(review), deepcopy(review)]
    reviews[0]["critical_constraints"]["numerics"] = deepcopy(endpoint_numerics)
    reviews[1]["critical_constraints"]["numerics"] = deepcopy(middle_numerics)
    reviews[2]["critical_constraints"]["numerics"] = deepcopy(endpoint_numerics)
    request_row = {
        "custom_id": "numeric-range-consensus",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = _semantic_quota_failed_row(
        custom_id="numeric-range-consensus",
        prompt_sha256="prompt",
        reviews=reviews,
        payload=payload,
    )
    return _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )


@pytest.mark.parametrize("range_value", ["60~105", "105~60", "60至105"])
def test_unsupported_consensus_normalizes_only_closed_numeric_range_endpoints(
    range_value: str,
) -> None:
    recovered = _recover_unsupported_with_numeric_consensus(
        [
            {
                "value": range_value,
                "unit": "g/m²",
                "evidence_labels": ["E1", "E2"],
            }
        ]
    )

    assert recovered is not None
    assert recovered["review"]["verdict"] == "block"


@pytest.mark.parametrize(
    "middle_numerics",
    [
        [{"value": "60~105", "unit": "kg", "evidence_labels": ["E1", "E2"]}],
        [{"value": "60~105", "unit": "g/m²", "evidence_labels": ["E1"]}],
        [{"value": "60~106", "unit": "g/m²", "evidence_labels": ["E1", "E2"]}],
        [{"value": "60~", "unit": "g/m²", "evidence_labels": ["E1", "E2"]}],
        [
            {"value": "60~105", "unit": "g/m²", "evidence_labels": ["E1", "E2"]},
            {"value": "90", "unit": "g/m²", "evidence_labels": ["E1", "E2"]},
        ],
    ],
)
def test_unsupported_consensus_rejects_non_equivalent_numeric_constraints(
    middle_numerics: list[dict],
) -> None:
    assert _recover_unsupported_with_numeric_consensus(middle_numerics) is None


def _empty_context_branch_review() -> tuple[dict, dict]:
    review, payload = _partial_evidence_review()
    context_span = "卡纸故障的解决办法中提到应清洁进纸通道和给纸辊"
    review["required_concepts"]["all_of"] = [
        deepcopy(review["required_concepts"]["all_of"][1])
    ]
    review["required_branches"][0] = {
        "branch_label": "B1",
        "query_span": context_span,
        "required_concept_keys": [],
    }
    review["evidence_concept_coverage"][0]["required_concept_keys"] = ["c02"]
    review["review_notes"] = [
        "B1 is a contextual premise, not a question requiring an answer concept."
    ]
    payload["query"] = f"{context_span}。若打印多页进纸，应如何处理？"
    payload["branches"][0]["query_span"] = context_span
    return review, payload


def test_empty_context_branch_projector_changes_only_verdict() -> None:
    review, payload = _empty_context_branch_review()

    projected, actions, raw_errors = (
        _canonicalize_same_response_empty_required_branch_block(review, payload)
    )

    assert raw_errors == [
        "required_branches[0].required_concept_keys must be non-empty"
    ]
    assert actions == [
        {
            "field": "verdict",
            "before": "revise",
            "after": "block",
            "empty_branch_labels": ["B1"],
            "empty_branch_query_spans": [
                "卡纸故障的解决办法中提到应清洁进纸通道和给纸辊"
            ],
            "authority": "context_preamble_has_no_required_answer_concept",
        }
    ]
    expected = deepcopy(review)
    expected["verdict"] = "block"
    assert projected == expected
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, payload: (
            review["required_branches"][0].update(
                query_span="卡纸故障应如何处理"
            ),
            payload["branches"][0].update(query_span="卡纸故障应如何处理"),
        ),
        lambda review, payload: (
            review["required_branches"][0].update(
                query_span="请说明卡纸故障的解决办法"
            ),
            payload["branches"][0].update(
                query_span="请说明卡纸故障的解决办法"
            ),
        ),
        lambda review, _payload: review.update(review_notes=["B1无需新增内容。"]),
        lambda review, _payload: review["required_concepts"]["all_of"][0][
            "branch_labels"
        ].append("B1"),
        lambda review, _payload: review["critical_constraints"]["numerics"].append(
            {"value": "2", "unit": "次", "evidence_labels": ["E1"]}
        ),
        lambda review, _payload: review["required_branches"][1].update(
            required_concept_keys=[]
        ),
        lambda review, _payload: review.update(verdict="approve"),
        lambda review, _payload: review.update(verdict="block"),
        lambda review, _payload: review.update(answerability="abstain"),
    ],
)
def test_empty_context_branch_projector_rejects_questions_or_other_defects(
    mutator,
) -> None:
    review, payload = _empty_context_branch_review()
    mutator(review, payload)

    projected, actions, _raw_errors = (
        _canonicalize_same_response_empty_required_branch_block(review, payload)
    )

    assert projected == review
    assert actions == []


def test_saved_empty_context_branch_requires_quota_exhausted_consensus() -> None:
    review, payload = _empty_context_branch_review()
    request_row = {
        "custom_id": "saved-empty-context-branch",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    reviews = [deepcopy(review), deepcopy(review), deepcopy(review)]
    reviews[1]["review_notes"] = [
        "B1 is a contextual preamble and requires no answer concept."
    ]
    reviews[2]["review_notes"] = ["B1 is contextual background, not a question."]
    failed_row = _semantic_quota_failed_row(
        custom_id="saved-empty-context-branch",
        prompt_sha256="prompt",
        reviews=reviews,
        payload=payload,
    )

    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )

    assert recovered is not None
    assert recovered["review"]["verdict"] == "block"
    assert recovered["review"]["review_notes"] == reviews[-1]["review_notes"]
    audit = recovered["multi_response_empty_required_branch_block"]
    assert audit["consensus"]["response_count"] == 3
    assert audit["source"]["source_attempt"] == 3
    assert _validate_empty_required_branch_block_audit(
        recovered,
        payload,
        source_response_rows=[failed_row],
    ) == []
    strict_errors, succeeded = _strict_success_response_errors(
        [failed_row, recovered],
        {"saved-empty-context-branch": request_row},
        expected_prompt_hash="prompt",
        pack_name="synthetic-empty-branch",
    )
    assert strict_errors == []
    assert succeeded == {"saved-empty-context-branch"}
    candidate = _partial_evidence_candidate(payload)
    contract, qa = _merge_contract(candidate_case=candidate, response=recovered)
    assert contract["contract_status"] == "llm_reviewed_blocked"
    assert qa["verdict"] == "block"
    with pytest.raises(RuntimeError):
        _validate_contract_coverage(
            contract,
            candidate["relevant_evidence"],
            case_id=candidate["case_id"],
            expected_branch_spans=[
                branch["query_span"] for branch in payload["branches"]
            ],
        )

    inconsistent_reviews = [deepcopy(review), deepcopy(review), deepcopy(review)]
    inconsistent_reviews[1]["required_concepts"]["all_of"][0][
        "concept_key"
    ] = "c03"
    inconsistent_reviews[1]["required_branches"][1][
        "required_concept_keys"
    ] = ["c03"]
    inconsistent_reviews[1]["evidence_concept_coverage"][0][
        "required_concept_keys"
    ] = ["c03"]
    inconsistent = _semantic_quota_failed_row(
        custom_id="saved-empty-context-branch",
        prompt_sha256="prompt",
        reviews=inconsistent_reviews,
        payload=payload,
    )
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[inconsistent]
    ) is None
    incomplete_terminal = deepcopy(failed_row)
    incomplete_terminal["invalid_attempts"][1]["sse_integrity"][
        "usage_integrity"
    ] = {"status": "missing"}
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[incomplete_terminal]
    ) is None
    quota_not_exhausted = deepcopy(failed_row)
    quota_not_exhausted["retry_accounting"]["quota_exhausted"] = None
    assert _recover_prior_invalid_review(
        row=request_row, response_rows=[quota_not_exhausted]
    ) is None


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (
            lambda review: review["evidence_concept_coverage"][1].update(
                required_concept_keys=["c01"]
            ),
            "must be empty when status=uncovered",
        ),
        (
            lambda review: review["evidence_concept_coverage"][0].update(
                required_concept_keys=[]
            ),
            "required_concept_keys must be non-empty",
        ),
        (
            lambda review: review["evidence_concept_coverage"].pop(),
            "evidence coverage labels must exactly match payload evidence",
        ),
        (
            lambda review: review["evidence_concept_coverage"].append(
                {
                    "evidence_label": "E3",
                    "required_concept_keys": [],
                    "status": "uncovered",
                }
            ),
            "unknown evidence labels",
        ),
    ],
)
def test_partial_evidence_review_rejects_invalid_status_key_or_label_grain(
    mutator, expected_error: str
) -> None:
    review, payload = _partial_evidence_review()
    mutator(review)
    errors = [*_validate_review(review), *_validate_against_payload(review, payload)]
    assert any(expected_error in error for error in errors)


def test_partial_evidence_review_rejects_ambiguous_zero_covered_or_false_support() -> None:
    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1].update(
        status="insufficient", required_concept_keys=["c01"]
    )
    assert any(
        "must be empty when status=insufficient" in error
        for error in _validate_review(review)
    )

    review, payload = _partial_evidence_review()
    review["evidence_concept_coverage"][1]["status"] = "ambiguous"
    assert any("status must be one of" in error for error in _validate_review(review))

    review, payload = _partial_evidence_review()
    for item in review["evidence_concept_coverage"]:
        item.update(status="uncovered", required_concept_keys=[])
    errors = [*_validate_review(review), *_validate_against_payload(review, payload)]
    assert any("at least one covered evidence" in error for error in errors)

    review, payload = _partial_evidence_review()
    review["required_concepts"]["all_of"][0]["evidence_labels"] = ["E2"]
    errors = _validate_against_payload(review, payload)
    assert any("edges must be symmetric" in error for error in errors)

def _partial_answer_contract() -> dict:
    contract = _answer_contract()
    contract["required_concepts"]["all_of"][1]["evidence_support"] = [
        {"evidence_id": "e1"}
    ]
    contract["evidence_concept_coverage"] = [
        {
            "evidence_id": "e1",
            "required_concept_ids": ["c01", "c02"],
            "coverage_status": "covered",
        },
        {
            "evidence_id": "e2",
            "required_concept_ids": [],
            "coverage_status": "uncovered",
        },
    ]
    return contract


@pytest.mark.parametrize(
    "mutator",
    [
        lambda contract: contract["evidence_concept_coverage"][0].update(
            required_concept_ids=[]
        ),
        lambda contract: contract["evidence_concept_coverage"][1].update(
            required_concept_ids=["c01"]
        ),
        lambda contract: contract["evidence_concept_coverage"][1].update(
            coverage_status="insufficient", required_concept_ids=["c01"]
        ),
        lambda contract: contract["evidence_concept_coverage"][1].update(
            coverage_status="ambiguous"
        ),
        lambda contract: [
            item.update(coverage_status="uncovered", required_concept_ids=[])
            for item in contract["evidence_concept_coverage"]
        ],
    ],
)
def test_partial_evidence_contract_merge_and_independent_validator_stay_in_parity(
    mutator,
) -> None:
    contract = _partial_answer_contract()
    mutator(contract)
    with pytest.raises(RuntimeError, match="edge coverage"):
        _validate_contract_coverage(
            contract,
            [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            case_id="partial-invalid",
            expected_branch_spans=["分支一"],
        )
    assert _contract_coverage_errors(
        contract,
        [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
        expected_branch_spans=["分支一"],
        expected_unanswerable=False,
    ) == ["concept/evidence/branch edges are incomplete or asymmetric"]


def test_partial_evidence_inverse_edge_repair_uses_only_covered_rows() -> None:
    review, payload = _partial_evidence_review()
    review["required_concepts"]["all_of"][0]["evidence_labels"] = ["E1", "E2"]
    projected, actions, raw_errors = _canonicalize_same_response_inverse_edges(
        review, payload
    )
    assert raw_errors == [
        "concept.evidence_labels and evidence_concept_coverage edges must be symmetric"
    ]
    assert projected["required_concepts"]["all_of"][0]["evidence_labels"] == ["E1"]
    assert any(action["field"] == "evidence_labels" for action in actions)
    assert projected["evidence_concept_coverage"][1] == {
        "evidence_label": "E2",
        "required_concept_keys": [],
        "status": "uncovered",
    }

    missing_labels = deepcopy(review)
    del missing_labels["required_concepts"]["all_of"][0]["evidence_labels"]
    normalized, normalization_actions = _normalize_review_structure(
        missing_labels, payload
    )
    assert normalized["required_concepts"]["all_of"][0]["evidence_labels"] == ["E1"]
    assert "crossref_filled_evidence_labels:c1" in normalization_actions


def test_answer_contract_rejects_zero_supported_concept_and_uncovered_edges() -> None:
    contract = _answer_contract()
    contract["required_concepts"]["all_of"][1]["evidence_support"] = []
    contract["evidence_concept_coverage"][1] = {
        "evidence_id": "e2",
        "required_concept_ids": [],
        "coverage_status": "uncovered",
    }
    with pytest.raises(RuntimeError, match="unsupported"):
        _validate_contract_coverage(
            contract,
            [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            case_id="unsupported-concept",
            expected_branch_spans=["分支一"],
        )
    assert _contract_coverage_errors(
        contract,
        [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
        expected_branch_spans=["分支一"],
        expected_unanswerable=False,
    ) == ["concept/evidence/branch edges are incomplete or asymmetric"]


def test_saved_partial_evidence_response_is_revalidated_without_projection_or_api() -> None:
    review, payload = _partial_evidence_review()
    request_row = {
        "custom_id": "partial-evidence-replay",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = {
        "custom_id": "partial-evidence-replay",
        "status": "failed",
        "prompt_sha256": "prompt",
        "attempt": 3,
        "model": "test-model",
        "runner_version": "2.10.0-same-response-exact-dedupe",
        "completed_at": "2026-08-11T00:00:00+00:00",
        "invalid_attempts": [
            {
                "attempt": 1,
                "review": review,
                "validation_errors": [
                    "legacy validator required every evidence row to be covered"
                ],
            }
        ],
    }
    recovered = _recover_prior_invalid_review(
        row=request_row, response_rows=[failed_row]
    )
    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["transport"] == "saved-response-recovery-no-api"
    assert recovered["review"] == review
    assert recovered["usage"]["total_tokens"] == 0
    audit = recovered["saved_response_strict_revalidation"]
    assert audit["api_call_performed"] is False
    assert audit["raw_review_sha256"] == _sha256_json(review)
    assert audit["projected_review_sha256"] == _sha256_json(review)
    assert (
        _validate_saved_strict_revalidation_audit(
            recovered, payload, source_response_rows=[failed_row]
        )
        == []
    )
    assert any(
        "requires the append-only source response log" in error
        for error in _validate_saved_strict_revalidation_audit(recovered, payload)
    )
    assert any(
        "not present exactly once" in error
        for error in _validate_saved_strict_revalidation_audit(
            recovered, payload, source_response_rows=[]
        )
    )
    assert any(
        "not present exactly once" in error
        for error in _validate_saved_strict_revalidation_audit(
            recovered, payload, source_response_rows=[failed_row, deepcopy(failed_row)]
        )
    )

    tampered = deepcopy(recovered)
    tampered["review"]["evidence_concept_coverage"][1][
        "required_concept_keys"
    ] = ["c01"]
    assert _validate_saved_strict_revalidation_audit(
        tampered, payload, source_response_rows=[failed_row]
    )


def test_saved_strict_revalidation_never_recovers_other_schema_or_alignment_defects() -> None:
    review, payload = _partial_evidence_review()
    review["required_branches"][1]["query_span"] = "截断"
    request_row = {
        "custom_id": "invalid-replay",
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }
    failed_row = {
        "custom_id": "invalid-replay",
        "status": "failed",
        "prompt_sha256": "prompt",
        "attempt": 1,
        "model": "test-model",
        "invalid_attempts": [
            {
                "attempt": 1,
                "review": review,
                "validation_errors": ["branch query_span differs from payload for B2"],
            }
        ],
    }
    assert (
        _recover_prior_invalid_review(row=request_row, response_rows=[failed_row])
        is None
    )


def test_abstain_materialization_is_nonempty_and_never_claims_evidence_support() -> None:
    candidate = {
        "case_id": "SYNTH-ABSTAIN-V2-001",
        "query": "今天的实时价格是多少？",
        "reference_answer": "资料未提供实时价格，无法确定。",
        "unanswerable": True,
        "expected_terms": [],
        "forbidden_terms": [],
        "relevant_evidence": [
            {"evidence_id": "e1", "chunk_id": "chunk-1", "page": 1, "text": "产品手册"}
        ],
        "answer_contract": {
            "required_branches": [{"branch_id": "b01", "query_span": "今天的实时价格是多少"}],
            "review": {"flags": []},
        },
        "metadata": {"parent_case_id": "SYNTH-ABSTAIN-001", "difficulty": "hard"},
    }
    response = {
        "custom_id": "synthetic-abstain-review",
        "prompt_sha256": "prompt",
        "review": {
            "verdict": "approve",
            "answerability": "abstain",
            "required_concepts": {
                "all_of": [
                    {
                        "concept_key": "c01",
                        "canonical_text": "资料不足，无法依据已授权证据确定答案",
                        "any_of": ["资料不足", "未提供相关信息", "无法确定"],
                        "evidence_labels": [],
                        "branch_labels": ["B1"],
                    }
                ]
            },
            "critical_constraints": {
                "numerics": [],
                "identifiers": [],
                "polarity": {
                    "required": "abstain_without_factual_claims",
                    "cues": ["资料不足", "无法确定"],
                },
            },
            "required_branches": [
                {
                    "branch_label": "B1",
                    "query_span": "今天的实时价格是多少",
                    "required_concept_keys": ["c01"],
                }
            ],
            "evidence_concept_coverage": [
                {"evidence_label": "E1", "required_concept_keys": [], "status": "uncovered"}
            ],
            "review_notes": ["资料不足，因此只允许拒答。"],
        },
    }
    assert _validate_review(response["review"]) == []
    contract, _qa = _merge_contract(candidate_case=candidate, response=response)
    _validate_contract_coverage(
        contract,
        candidate["relevant_evidence"],
        case_id=candidate["case_id"],
        expected_branch_spans=["今天的实时价格是多少"],
    )
    groups, _audit = _materialize_concept_groups(
        contract, case_id=candidate["case_id"], evidence=candidate["relevant_evidence"]
    )
    assert len(groups["all_of"]) == 1
    assert contract["required_concepts"]["all_of"][0]["evidence_support"] == []
    assert all(
        item["coverage_status"] == "not_applicable_to_abstention"
        and item["required_concept_ids"] == []
        for item in contract["evidence_concept_coverage"]
    )
    assert contract["critical_constraints"]["numerics"] == []
    assert contract["critical_constraints"]["identifiers"] == []
    assert FROZEN_CASE_PASS_POLICY["abstention_factual_claim_maximum"] == 0
    assert (
        _contract_coverage_errors(
            contract,
            candidate["relevant_evidence"],
            expected_branch_spans=["今天的实时价格是多少"],
            expected_unanswerable=True,
        )
        == []
    )


def test_abstain_review_rejects_factual_constraints_and_mixed_variants() -> None:
    review = {
        "verdict": "revise",
        "answerability": "abstain",
        "required_concepts": {
            "all_of": [
                {
                    "concept_key": "c1",
                    "canonical_text": "资料未提供价格，无法确定",
                    "any_of": ["价格是2000元"],
                    "evidence_labels": [],
                    "branch_labels": ["B1"],
                }
            ]
        },
        "critical_constraints": {
            "numerics": [{"value": "2000", "unit": "元", "evidence_labels": []}],
            "identifiers": [],
            "polarity": {"required": "negative", "cues": ["无法"]},
        },
        "required_branches": [],
        "evidence_concept_coverage": [],
        "review_notes": [],
    }
    errors = _validate_review(deepcopy(review))
    assert any("variants must each express" in error for error in errors)
    assert any("must not require factual numerics" in error for error in errors)


def test_polarity_vocabulary_is_finite_and_preserves_mixed_conditional_semantics() -> None:
    expected = {
        "positive",
        "negative",
        "not_applicable",
        "mixed_preserve_each_claim",
        "affirmative",
        "positive_with_conditional",
        "affirmative_or_explicit_yes_no",
        "abstain_without_factual_claims",
    }
    assert _POLARITY_REQUIRED_VALUES == expected
    review = {
        "verdict": "block",
        "answerability": "answer",
        "required_concepts": {"all_of": []},
        "critical_constraints": {
            "numerics": [],
            "identifiers": [],
            "polarity": {"required": "not_applicable", "cues": []},
        },
        "required_branches": [],
        "evidence_concept_coverage": [],
        "review_notes": ["source contract requires review"],
    }
    for polarity in sorted(expected):
        candidate = deepcopy(review)
        candidate["critical_constraints"]["polarity"]["required"] = polarity
        assert _validate_review(candidate) == []
    review["critical_constraints"]["polarity"]["required"] = "free_form_unknown"
    assert any(
        "frozen polarity values" in error for error in _validate_review(review)
    )


def test_final_rereview_adoption_rejects_structural_normalization() -> None:
    review = {
        "verdict": "block",
        "answerability": "answer",
        "required_concepts": {"all_of": []},
        "critical_constraints": {
            "numerics": [],
            "identifiers": [],
            "polarity": {"required": "not_applicable", "cues": []},
        },
        "required_branches": [],
        "evidence_concept_coverage": [],
        "review_notes": ["source contract requires review"],
    }
    payload = {
        "query": "问题",
        "reference_answer": "答案",
        "unanswerable": False,
        "legacy_expected_terms": [],
        "legacy_forbidden_terms": [],
        "branches": [],
        "evidence": [],
    }
    requests = {
        "r1": {
            "request": {
                "messages": [
                    {"role": "system", "content": "blind"},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
            }
        }
    }
    errors, succeeded = _strict_success_response_errors(
        [
            {
                "custom_id": "r1",
                "status": "succeeded",
                "prompt_sha256": "prompt",
                "review": review,
                "structural_normalization": ["restored_missing_semantics"],
            }
        ],
        requests,
        expected_prompt_hash="prompt",
        pack_name="test-rereview",
    )
    assert succeeded == {"r1"}
    assert any("structural normalization" in error for error in errors)


def test_activation_targets_and_write_gate_refuse_nonidentical_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TARGET_COUNTS == {
        "easy": {"correct_count": 48, "case_count": 50},
        "medium": {"correct_count": 43, "case_count": 50},
        "hard": {"correct_count": None, "case_count": 50},
    }
    original_final_root = activate_suite_module.FINAL_ROOT
    with tempfile.TemporaryDirectory(
        prefix=".test-activation-write-",
        dir=original_final_root,
    ) as directory:
        trusted_final_root = Path(directory)
        with monkeypatch.context() as context:
            context.setattr(activate_suite_module, "FINAL_ROOT", trusted_final_root)
            target = trusted_final_root / "activation.json"
            _write_exact_or_refuse(target, b"first")
            _write_exact_or_refuse(target, b"first")
            with pytest.raises(ActivationError, match="refusing to overwrite"):
                _write_exact_or_refuse(target, b"different")


def test_sse_transport_reassembles_content_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    review = {
        "verdict": "block",
        "answerability": "answer",
        "required_concepts": {"all_of": []},
        "critical_constraints": {"numerics": [], "identifiers": [], "polarity": {}},
        "required_branches": [],
        "evidence_concept_coverage": [],
        "review_notes": ["blocked"],
    }
    serialized = json.dumps(review, ensure_ascii=False)
    events = [
        {"choices": [{"delta": {"content": serialized[:20]}}]},
        {"choices": [{"delta": {"content": serialized[20:]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    ]

    class FakeResponse:
        headers = {"Content-Type": "text/event-stream; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n".encode()
            yield b"data: [DONE]\n"

    captured_body: dict = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured_body.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    parsed, usage = _call_api(
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        request_body={"model": "test", "messages": []},
        timeout=10,
    )
    assert parsed == review
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert captured_body["stream"] is True
    assert captured_body["stream_options"] == {"include_usage": True}
