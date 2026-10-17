from __future__ import annotations

from copy import deepcopy

from app.application.evaluation_service import _case_decision, _gate
from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    EVALUATION_GATE_POLICY_VERSION,
    freeze_case_pass_policy,
    freeze_evaluation_gate_policy,
    validate_dataset_manifest,
)
from app.evaluation.metrics import aggregate_results, evaluate_case, paired_diff


POLICY = {"policy_version": AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION}


def _table_case() -> dict:
    source_text = "标签 ○ ○ ○ ○ ○ ○；包装材料 ○ ○ ○ ○ ○ ○"
    anchor = {"evidence_id": "ev-table", "text": source_text}
    return {
        "case_id": "blind-table-ellipsis",
        "query": "表格里哪些部件的六类有害物质全部标为 ○？",
        "reference_answer": "标签和包装材料的六类有害物质全部标为 ○。",
        "unanswerable": False,
        "expected_terms": ["标签", "包装材料"],
        "forbidden_terms": [],
        "relevant_evidence": [
            {
                "evidence_id": "ev-table",
                "text": source_text,
                "match_terms": ["标签", "包装材料", "○"],
            }
        ],
        "concept_groups": {
            "contract_version": "stt.eval.authored-concepts.v3",
            "all_of": [
                {
                    "concept_key": "label",
                    "any_of": ["标签"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["标签"],
                        "anchors": [anchor],
                    },
                },
                {
                    "concept_key": "packaging",
                    "any_of": ["包装材料"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["包装材料"],
                        "anchors": [anchor],
                    },
                },
                {
                    "concept_key": "all_six_clear",
                    "any_of": ["六类有害物质全部标为 ○"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["○"],
                        "anchors": [anchor],
                    },
                    "satisfaction": {
                        "mode": "query_predicate_ellipsis",
                        "query_any_of": ["六类有害物质全部标为 ○"],
                        "requires_concept_keys": ["label", "packaging"],
                    },
                },
            ],
        },
        "authored_claims": {
            "contract_version": "stt.eval.authored-claims.v1",
            "atoms": [
                {
                    "claim_key": "label_clear",
                    "requires_concept_keys": ["label", "all_six_clear"],
                    "citation_required": True,
                    "source_alignment": {
                        "anchors": [{"evidence_id": "ev-table", "text": "标签 ○ ○ ○ ○ ○ ○"}],
                        "source_terms": ["标签", "○"],
                    },
                },
                {
                    "claim_key": "packaging_clear",
                    "requires_concept_keys": ["packaging", "all_six_clear"],
                    "citation_required": True,
                    "source_alignment": {
                        "anchors": [
                            {
                                "evidence_id": "ev-table",
                                "text": "包装材料 ○ ○ ○ ○ ○ ○",
                            }
                        ],
                        "source_terms": ["包装材料", "○"],
                    },
                },
            ],
        },
    }


def _observation(answer: str, *, source_text: str | None = None) -> dict:
    candidate = {
        "evidence_id": "ev-table",
        "chunk_id": "chunk-table",
        "label": "E1",
        "text": source_text or "标签 ○ ○ ○ ○ ○ ○；包装材料 ○ ○ ○ ○ ○ ○",
    }
    return {
        "answer": answer,
        "retrieved": [candidate],
        "context": [candidate],
    }


def _evaluate_and_decide(case: dict, observation: dict) -> tuple[dict, dict]:
    evaluated = evaluate_case(case, observation, case_policy=POLICY)
    decision = _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy=POLICY,
        observation=observation,
    )
    evaluated["metrics"]["case_outcome"] = decision
    return evaluated, decision


def test_v5_source_backed_query_predicate_ellipsis_atomizes_table_rows() -> None:
    case = _table_case()
    validate_dataset_manifest(
        {
            "dataset_key": "blind-v5",
            "name": "blind v5",
            "version": "1.0.0",
            "schema_version": "stt.eval.v1",
            "source": {},
            "gate_policy": {"case_pass_policy": POLICY},
            "cases": [case],
        }
    )
    observation = _observation("以下部件 [E1]\n- 标签 [E1]\n- 包装材料 [E1]")

    evaluated, decision = _evaluate_and_decide(case, observation)
    generation = evaluated["metrics"]["generation"]

    assert generation["concept_group_coverage"] == 1.0
    assert generation["authored_claim_coverage"] == 1.0
    assert generation["faithfulness"] == 1.0
    assert decision["final_answer_correct"] is True
    assert decision["strict_supported_case_pass"] is True


def test_v5_query_is_never_factual_support_without_live_joint_source() -> None:
    case = _table_case()
    observation = _observation(
        "以下部件 [E1]\n- 标签 [E1]\n- 包装材料 [E1]",
        source_text="标签；包装材料",
    )

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["authored_claim_coverage"] == 0.0
    assert decision["final_answer_correct"] is False
    assert decision["strict_supported_case_pass"] is False
    assert "authored_claim_incomplete" in decision["reason_codes"]
    assert "faithfulness_below_minimum" in decision["reason_codes"]


def test_v5_missing_table_subject_still_fails_final_correctness() -> None:
    case = _table_case()
    observation = _observation("以下部件 [E1]\n- 标签 [E1]")

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["concept_group_coverage"] < 1.0
    assert decision["final_answer_correct"] is False
    assert "concept_group_incomplete" in decision["reason_codes"]


def test_v5_partial_authored_atom_citation_is_a_conjunctive_failure() -> None:
    case = _table_case()
    observation = _observation("以下部件 [E1]\n- 标签 [E1]\n- 包装材料")

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["concept_group_coverage"] == 1.0
    assert evaluated["metrics"]["generation"]["authored_claim_coverage"] == 0.5
    assert decision["final_answer_correct"] is False
    assert decision["strict_supported_case_pass"] is False
    assert "authored_claim_incomplete" in decision["reason_codes"]


def test_v5_answerable_refusal_fails_closed() -> None:
    case = _table_case()
    observation = _observation("现有信息不足，无法确定。")

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["abstention"]["contract_predicted"] is True
    assert decision["final_answer_correct"] is False
    assert decision["strict_supported_case_pass"] is False


def test_v5_correct_abstention_is_strict_and_grounding_not_applicable() -> None:
    case = {
        "case_id": "blind-unanswerable",
        "query": "手册未声明的规格是多少？",
        "reference_answer": "手册未给出该规格，因此无法确定。",
        "unanswerable": True,
        "expected_terms": ["未给出", "无法确定"],
        "forbidden_terms": [],
        "relevant_evidence": [],
    }
    observation = {
        "answer": "手册未给出该规格，因此无法确定。",
        "abstained": True,
        "retrieved": [],
        "context": [],
        "visible_answer_policy": {
            "strict_grounding_eligible": False,
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        },
    }

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["faithfulness"] is None
    assert evaluated["metrics"]["citation"]["coverage"] is None
    assert decision["final_answer_correct"] is True
    assert decision["strict_supported_case_pass"] is True
    assert decision["reason_codes"] == []


def test_v5_does_not_hide_extra_unsupported_claim() -> None:
    case = _table_case()
    observation = _observation("以下部件 [E1]\n- 标签 [E1]\n- 包装材料 [E1]\n设备支持卫星通信 [E1]")

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["unsupported_factual_claim_count"] == 1
    assert decision["final_answer_correct"] is False
    assert "unsupported_factual_claim" in decision["reason_codes"]


def test_v5_policy_and_authored_metric_are_frozen_in_aggregate_and_pairing() -> None:
    frozen = freeze_case_pass_policy(POLICY)
    case = _table_case()
    passing, _ = _evaluate_and_decide(
        case,
        _observation("以下部件 [E1]\n- 标签 [E1]\n- 包装材料 [E1]"),
    )
    failing, _ = _evaluate_and_decide(case, _observation("以下部件 [E1]\n- 标签 [E1]"))

    summary = aggregate_results([passing])
    comparison = paired_diff([failing], [passing])

    assert frozen["policy_version"] == AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION
    assert summary["metrics"]["generation"]["authored_claim_coverage"]["mean"] == 1.0
    assert comparison["metric_deltas"]["metrics.generation.authored_claim_coverage"]["delta"] > 0


def test_v5_paired_gate_records_explicit_policy_version() -> None:
    gate_policy = freeze_evaluation_gate_policy({"case_pass_policy": POLICY})
    run = {
        "mode": "live",
        "baseline_run_id": "baseline-blind",
        "config_snapshot": {"gate_policy": gate_policy},
    }
    cases = [{"case_id": "blind-table-ellipsis", "metadata": {"critical": False}}]
    results = [{"case_id": "blind-table-ellipsis", "passed": True, "metrics": {}}]
    comparison = {
        "matched_case_count": 1,
        "matched_case_ids": ["blind-table-ellipsis"],
        "baseline_only_case_ids": [],
        "candidate_only_case_ids": [],
        "metric_deltas": {},
    }

    gate = _gate(run, cases, results, {"runtime": {}}, comparison)

    assert gate["passed"] is True
    assert gate["policy_version"] == EVALUATION_GATE_POLICY_VERSION
    assert gate["case_pass_policy_version"] == AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION


def test_v5_invalid_or_unbound_ellipsis_contract_fails_manifest_validation() -> None:
    case = _table_case()
    invalid = deepcopy(case)
    invalid["authored_claims"]["atoms"] = invalid["authored_claims"]["atoms"][:1]
    manifest = {
        "dataset_key": "blind-v5-invalid",
        "name": "blind v5 invalid",
        "version": "1.0.0",
        "schema_version": "stt.eval.v1",
        "source": {},
        "gate_policy": {"case_pass_policy": POLICY},
        "cases": [invalid],
    }

    try:
        validate_dataset_manifest(manifest)
    except ValueError as exc:
        assert "for every required concept" in str(exc)
    else:
        raise AssertionError("unbound query predicate contract must fail closed")
