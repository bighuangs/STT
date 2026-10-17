from __future__ import annotations

from copy import deepcopy

import pytest

from app.application.evaluation_service import _case_decision
from app.domain.evaluation import (
    CASE_PASS_POLICY_VERSION,
    SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
    freeze_case_pass_policy,
    validate_dataset_manifest,
)
from app.evaluation.metrics import evaluate_case


def _source_aligned_case() -> dict:
    source_text = "请勿将本设备连接到直流电源或逆变器。存在触电风险。"
    return {
        "case_id": "blind-coordinated-prohibition",
        "query": "设备可否连接两类非标准电源？",
        "reference_answer": "禁止连接直流电源，也禁止连接逆变器，否则存在触电风险。",
        "unanswerable": False,
        "expected_terms": ["直流电源", "逆变器", "触电风险"],
        "forbidden_terms": ["可以连接直流电源"],
        "relevant_chunk_ids": ["chunk-blind-source"],
        "relevant_evidence": [
            {
                "evidence_id": "ev-blind-source",
                "text": source_text,
                "match_terms": ["直流电源", "逆变器", "触电"],
            }
        ],
        "concept_groups": {
            "contract_version": "stt.eval.authored-concepts.v2",
            "all_of": [
                {
                    "concept_key": "dc_prohibited",
                    "any_of": ["禁止连接直流电源"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["直流电源"],
                        "anchors": [
                            {
                                "evidence_id": "ev-blind-source",
                                "text": "请勿将本设备连接到直流电源或逆变器",
                            }
                        ],
                    },
                },
                {
                    "concept_key": "inverter_prohibited",
                    "any_of": ["禁止连接逆变器"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["逆变器"],
                        "anchors": [
                            {
                                "evidence_id": "ev-blind-source",
                                "text": "请勿将本设备连接到直流电源或逆变器",
                            }
                        ],
                    },
                },
                {
                    "concept_key": "shock_risk",
                    "any_of": ["触电风险"],
                    "source_alignment": {
                        "mode": "required",
                        "source_terms": ["触电风险"],
                        "anchors": [
                            {
                                "evidence_id": "ev-blind-source",
                                "text": "存在触电风险",
                            }
                        ],
                    },
                },
            ],
        },
    }


def _observation(answer: str, evidence: str) -> dict:
    candidate = {"chunk_id": "chunk-blind-source", "text": evidence}
    return {
        "answer": answer,
        "retrieved": [candidate],
        "context": [candidate],
    }


def _evaluate_and_decide(case: dict, answer: str, evidence: str) -> tuple[dict, dict]:
    policy = {"policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION}
    observation = _observation(answer, evidence)
    evaluated = evaluate_case(case, observation, case_policy=policy)
    decision = _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy=policy,
        observation=observation,
    )
    return evaluated, decision


def test_v4_source_anchor_is_provenance_only_and_never_an_answer_variant() -> None:
    case = _source_aligned_case()
    answer = case["relevant_evidence"][0]["text"]
    observation = _observation(answer, answer)

    v3 = evaluate_case(
        case,
        observation,
        case_policy={"policy_version": CASE_PASS_POLICY_VERSION},
    )
    v4, outcome = _evaluate_and_decide(case, answer, answer)

    assert v3["metrics"]["generation"]["concept_group_coverage"] == pytest.approx(1 / 3)
    assert v4["metrics"]["generation"]["concept_group_coverage"] == pytest.approx(1 / 3)
    components = v4["metrics"]["generation"]["correctness_components"]
    assert components["contract_coverage_source"] == "authored_concept_groups"
    assert components["contract_coverage_score"] == pytest.approx(1 / 3)
    assert outcome["policy_version"] == SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION
    assert outcome["final_answer_correct"] is False


def test_v4_uses_authored_groups_as_coverage_authority() -> None:
    case = _source_aligned_case()
    case["reference_answer"] = "设备需要先断电并锁定防护盖。"
    case["expected_terms"] = ["断电"]
    case["forbidden_terms"] = []
    case["relevant_evidence"][0]["text"] = "设备需要先断电并锁定防护盖。"
    case["concept_groups"] = {
        "contract_version": "stt.eval.authored-concepts.v2",
        "all_of": [
            {
                "concept_key": "power_off",
                "any_of": ["先断电"],
                "source_alignment": {
                    "mode": "required",
                    "source_terms": ["先断电"],
                    "anchors": [{"evidence_id": "ev-blind-source", "text": "先断电"}],
                },
            },
            {
                "concept_key": "lock_guard",
                "any_of": ["锁定防护盖"],
                "source_alignment": {
                    "mode": "required",
                    "source_terms": ["锁定防护盖"],
                    "anchors": [{"evidence_id": "ev-blind-source", "text": "锁定防护盖"}],
                },
            },
        ],
    }
    answer = "设备需要先断电。"
    observation = _observation(answer, case["relevant_evidence"][0]["text"])

    v3 = evaluate_case(
        case,
        observation,
        case_policy={"policy_version": CASE_PASS_POLICY_VERSION},
    )
    v4, outcome = _evaluate_and_decide(
        case,
        answer,
        case["relevant_evidence"][0]["text"],
    )

    assert v3["metrics"]["generation"]["correctness_components"]["contract_coverage_score"] == 1.0
    components = v4["metrics"]["generation"]["correctness_components"]
    assert components["concept_group_coverage"] == 0.5
    assert components["contract_coverage_score"] == 0.5
    assert outcome["final_answer_correct"] is False
    assert "concept_group_incomplete" in outcome["reason_codes"]


def test_v4_excludes_only_explicit_unaligned_reference_details() -> None:
    case = _source_aligned_case()
    case["reference_answer"] = "必须断电；还建议联系未出现在黄金证据中的服务方。"
    case["expected_terms"] = ["断电"]
    case["forbidden_terms"] = []
    case["relevant_evidence"][0]["text"] = "维护前必须断电。"
    case["concept_groups"] = {
        "contract_version": "stt.eval.authored-concepts.v2",
        "all_of": [
            {
                "concept_key": "power_off",
                "any_of": ["必须断电"],
                "source_alignment": {
                    "mode": "required",
                    "source_terms": ["必须断电"],
                    "anchors": [{"evidence_id": "ev-blind-source", "text": "必须断电"}],
                },
            },
            {
                "concept_key": "reference_only_contact",
                "any_of": ["联系服务方"],
                "source_alignment": {
                    "mode": "diagnostic",
                    "reason_code": "REFERENCE_DETAIL_NOT_QREL_SUPPORTED",
                },
            },
        ],
    }

    evaluated, outcome = _evaluate_and_decide(
        case,
        "维护前必须断电。",
        "维护前必须断电。",
    )
    components = evaluated["metrics"]["generation"]["correctness_components"]

    assert components["concept_group_count"] == 1
    assert components["concept_group_coverage"] == 1.0
    assert components["diagnostic_concept_group_count"] == 1
    assert components["diagnostic_concept_group_coverage"] == 0.0
    assert components["missing_diagnostic_concept_group_keys"] == ["reference_only_contact"]
    assert outcome["final_answer_correct"] is True


@pytest.mark.parametrize(
    ("reference", "answer", "reason_code"),
    [
        ("额定压力为 16 MPa。", "额定压力为 18 MPa。", "critical_number_mismatch"),
        ("不得打开防护盖。", "可以打开防护盖。", "polarity_mismatch"),
    ],
)
def test_v4_keeps_number_and_polarity_guards(
    reference: str,
    answer: str,
    reason_code: str,
) -> None:
    case = _source_aligned_case()
    case["reference_answer"] = reference
    case["expected_terms"] = []
    case["forbidden_terms"] = []
    case["relevant_evidence"][0]["text"] = reference
    case["concept_groups"] = {
        "contract_version": "stt.eval.authored-concepts.v2",
        "all_of": [
            {
                "concept_key": "subject",
                "any_of": ["额定压力" if "压力" in reference else "防护盖"],
                "source_alignment": {
                    "mode": "required",
                    "source_terms": ["额定压力" if "压力" in reference else "防护盖"],
                    "anchors": [
                        {
                            "evidence_id": "ev-blind-source",
                            "text": "额定压力" if "压力" in reference else "防护盖",
                        }
                    ],
                },
            }
        ],
    }

    _, outcome = _evaluate_and_decide(case, answer, reference)

    assert outcome["final_answer_correct"] is False
    assert reason_code in outcome["reason_codes"]


def test_v4_keeps_unsupported_claim_guard() -> None:
    case = _source_aligned_case()
    case["query"] = "该设备支持卫星通信，请说明如何使用。"
    evidence = case["relevant_evidence"][0]["text"]

    _, outcome = _evaluate_and_decide(
        case,
        evidence + " 此外，该设备支持卫星通信。",
        evidence,
    )

    assert outcome["final_answer_correct"] is False
    assert "unsupported_factual_claim" in outcome["reason_codes"]


def test_v4_policy_is_explicit_and_cannot_relax_strict_guards() -> None:
    frozen = freeze_case_pass_policy({"policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION})

    assert frozen["policy_version"] == SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION
    assert frozen["concept_group_coverage_minimum"] == 1.0
    assert frozen["unsupported_factual_claim_maximum"] == 0
    with pytest.raises(ValueError, match="evaluation case pass field"):
        freeze_case_pass_policy(
            {
                "policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
                "concept_group_coverage_minimum": 0.99,
            }
        )


def test_v4_direct_evaluation_fails_closed_without_explicit_concept_contract() -> None:
    case = _source_aligned_case()
    case["concept_groups"].pop("contract_version")
    evidence = case["relevant_evidence"][0]["text"]

    evaluated, outcome = _evaluate_and_decide(case, evidence, evidence)

    assert (
        evaluated["metrics"]["generation"]["correctness_components"]["concept_contract_valid"]
        is False
    )
    assert outcome["final_answer_correct"] is False
    assert "correctness_contract_missing" in outcome["reason_codes"]


def test_v4_manifest_rejects_unbound_or_non_source_anchor() -> None:
    case = _source_aligned_case()
    manifest = {
        "dataset_key": "blind-v4-contract",
        "name": "blind v4 contract",
        "version": "1.0.0",
        "schema_version": "stt.eval.dataset.v1",
        "source": {},
        "gate_policy": {
            "case_pass_policy": {"policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION}
        },
        "cases": [case],
    }
    validate_dataset_manifest(manifest)

    invalid = deepcopy(manifest)
    invalid["cases"][0]["concept_groups"]["all_of"][0]["source_alignment"]["anchors"][0]["text"] = (
        "黄金证据中不存在的锚点"
    )

    with pytest.raises(ValueError, match="source anchor must occur"):
        validate_dataset_manifest(invalid)

    wrongly_bound = deepcopy(manifest)
    wrongly_bound["cases"][0]["concept_groups"]["all_of"][0]["source_alignment"]["source_terms"] = [
        "卫星通信"
    ]
    with pytest.raises(ValueError, match="any_of semantics"):
        validate_dataset_manifest(wrongly_bound)
