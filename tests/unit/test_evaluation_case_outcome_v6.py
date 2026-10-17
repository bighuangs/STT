from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from app.application.evaluation_service import _case_decision
from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
)
from app.evaluation.metrics import aggregate_results, evaluate_case
from evals.datasets.registry import load_dataset_asset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_POLICY = {"policy_version": AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION}
V6_POLICY = {"policy_version": TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION}


def _digest(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _case(original_query: str, claim: str) -> dict[str, Any]:
    return {
        "case_id": "blind-temporal-contract",
        "query": original_query,
        "reference_answer": claim,
        "unanswerable": False,
    }


def _observation(
    *,
    original_query: str,
    rewritten_query: str,
    claim: str,
    evidence: str,
    suffix: str,
) -> dict[str, Any]:
    label = "E1"
    evidence_id = f"ev_blind_{suffix}"
    answer = f"{claim} [{label}]"
    answer_hash = _digest(answer)
    evidence_ids = [evidence_id]
    envelope_claim = {
        "claim_id": "clm_1",
        "text": claim,
        "answer_start": 0,
        "answer_end": len(claim),
        "claim_type": "factual",
        "citation_ids": ["cit_1"],
        "supported": True,
    }
    return {
        "query": original_query,
        "rewritten_query": rewritten_query,
        "query_understanding": {
            "original_query": rewritten_query,
            "display_query": rewritten_query,
            "standalone_query": rewritten_query,
        },
        "answer": answer,
        "answer_envelope": {
            "schema_version": "answer.v1",
            "answer": answer,
            "claims": [envelope_claim],
            "citations": [
                {
                    "citation_id": "cit_1",
                    "claim_id": "clm_1",
                    "evidence_id": evidence_id,
                    "answer_start": 0,
                    "answer_end": len(claim),
                    "validation": "valid",
                    "relation": "SUPPORTS",
                }
            ],
            "sources": [{"evidence_id": evidence_id}],
        },
        "claims": [deepcopy(envelope_claim)],
        "abstained": False,
        "retrieved": [
            {
                "context_label": label,
                "evidence_id": evidence_id,
                "evidence": {"evidence_id": evidence_id},
                "text": evidence,
                "content": evidence,
            }
        ],
        "reranked": [],
        "context": [
            {
                "context_label": label,
                "evidence_id": evidence_id,
                "evidence": {"evidence_id": evidence_id},
                "text": evidence,
                "content": evidence,
            }
        ],
        "context_pack": {
            "packed_blocks": [{"label": label, "evidence_id": evidence_id, "text": evidence}],
            "label_mapping": [{"label": label, "evidence_id": evidence_id}],
        },
        "citations": [
            {
                "label": label,
                "evidence_id": evidence_id,
                "claim_text": claim,
                "answer_span": {"start": 0, "end": len(claim)},
                "validation_status": "valid",
            }
        ],
        "citation_validation": {
            "status": "valid",
            "coverage": 1.0,
            "citation_count": 1,
            "trusted_count": 1,
            "issues": [],
        },
        "answer_decision": {"action": "ANSWER"},
        "completeness_self_review": {
            "schema_version": "completeness-self-review-audit.v6",
            "execution_contract": "deterministic-zero-call.v1",
            "provider_review_enabled": False,
            "status": "committed_deterministic_temporal_binary_verdict",
            "calls_attempted": 0,
            "provider_attempt_count": 0,
            "retry_count": 0,
            "output_token_cap": 0,
            "prompt_sha256": "",
            "provider_response_sha256": "",
            "usage": {
                "status": "not_applicable",
                "reason": "deterministic zero-call completeness contract",
            },
            "materialized_candidate_sha256": answer_hash,
            "security_approved_candidate_sha256": answer_hash,
            "committed_answer_sha256": answer_hash,
            "runtime_completeness_receipt": {
                "schema_version": "runtime-completeness-receipt.v1",
                "solver_schema_version": "runtime-completeness-solver.v1",
                "status": "INDETERMINATE",
                "reason_code": "CURRENT_COVERAGE_UNPROVEN_NO_GAP",
                "solution_cardinality": "indeterminate",
                "contract_hash": "",
                "projection_hash": "",
                "candidate_count": 1,
                "authorized_candidate_count": 1,
                "authorized_universe_block_count": 1,
                "universe_digest": "",
                "universe_complete": True,
                "facet_set_digest": "",
                "facet_contract_complete": True,
                "zero_bindable_facet_count": 0,
                "nonzero_bindable_facet_count": 1,
                "facet_count": 1,
                "query_obligation_count": 1,
                "constraint_count": 0,
                "obligation_count": 1,
                "option_count": 1,
                "search_state_count": 0,
                "solution_count": 0,
                "selection_count": 0,
                "selection_hash": "",
                "residual_selection_count": 0,
                "materializer_validation_count": 0,
                "materializer_rejection_count": 0,
                "materializer_rejection_reason_counts": {},
                "materializer_rejection_reason_hash": "",
                "current_mapping_complete": True,
                "current_projection_available": True,
                "current_candidate_count": 1,
                "current_trusted_claim_count": 1,
                "current_mapped_claim_count": 1,
                "current_ambiguous_claim_count": 0,
                "current_unmapped_claim_count": 0,
                "current_incomplete_facet_count": 0,
                "current_uncovered_constraint_count": 0,
                "current_uncovered_query_obligation_count": 1,
                "current_uncovered_query_obligation_type_counts": {"direct_binary_verdict": 1},
                "trusted_floor_trusted_claim_count": 0,
                "trusted_floor_mapped_claim_count": 0,
                "trusted_floor_ambiguous_claim_count": 0,
                "trusted_floor_unmapped_claim_count": 0,
                "trusted_floor_mapping_complete": False,
                "trusted_floor_projection_available": False,
                "trusted_floor_candidate_count": 0,
                "trusted_floor_coverage_credit_count": 0,
                "trusted_floor_source": "none",
                "initial_floor_candidate_count": 0,
                "model_calls_added": 0,
                "semantic_absence_proven": False,
                "abstention_authorized": False,
                "canonical_selection_count": 0,
                "canonical_selection_hash": "",
                "canonicalization_policy": "",
            },
            "temporal_binary_verdict": {
                "rule_version": "direct-binary-temporal-gating/1.0.0",
                "status": "candidate_ready",
                "reason_code": "UNIQUE_MANDATORY_TEMPORAL_GATE",
                "query_sha256": _digest(rewritten_query),
                "evidence_set_sha256": _digest(_canonical_json(evidence_ids)),
                "selected_evidence_sha256": _digest(evidence_id),
                "authorized_evidence_count": 1,
                "direct_binary_query": True,
                "immediate_semantics": True,
                "current_envelope_valid": True,
                "qualifying_gate_count": 1,
                "conflict_count": 0,
                "condition_already_satisfied": False,
                "condition_state_ambiguous": False,
                "unsatisfied_trigger_proven": True,
            },
        },
        "latency_ms": 1.0,
        "versions": {},
        "side_effects": {},
    }


_BLIND_CASES = (
    (
        "请根据维护规则判断。机器刚停机，能立即打开维护盖吗？",
        "机器刚停机，能立即打开维护盖吗？",
        "不能立即打开维护盖",
        "必须先等待指示灯熄灭，再打开维护盖。",
        "cn_open",
    ),
    (
        "请只回答当前是否可操作。装置刚运行结束，可不可以马上移除过滤盒？",
        "装置刚运行结束，可不可以马上移除过滤盒？",
        "不可马上移除过滤盒",
        "先等待风扇完全停止，再移除过滤盒。",
        "cn_remove",
    ),
    (
        "Use the safety sequence. The unit has just finished running. "
        "Can I immediately open the service panel?",
        "The unit has just finished running. Can I immediately open the service panel?",
        "Right now the machine cannot open the service panel",
        "You must wait for the indicator to turn off before opening the service panel.",
        "en_open",
    ),
)


def _strip_temporal_diagnostics(generation: dict[str, Any]) -> dict[str, Any]:
    copied = deepcopy(generation)
    for field in (
        "temporal_derived_claim_count",
        "temporal_derived_validated_count",
        "temporal_receipt_status",
    ):
        copied.pop(field, None)
    return copied


def _bind_rewrite_authority(observation: dict[str, Any]) -> None:
    understanding = observation["query_understanding"]
    understanding["rewrite"] = {
        "status": "accepted",
        "semantic_preservation_score": 1.0,
        "reasons": ["rewrite_equivalent"],
    }
    observation["query_plan"] = {"understanding": deepcopy(understanding)}


@pytest.mark.parametrize(
    ("original_query", "rewritten_query", "claim", "evidence", "suffix"),
    _BLIND_CASES,
)
def test_v6_validates_one_blind_temporal_claim_without_changing_v5(
    original_query: str,
    rewritten_query: str,
    claim: str,
    evidence: str,
    suffix: str,
) -> None:
    case = _case(original_query, claim)
    observation = _observation(
        original_query=original_query,
        rewritten_query=rewritten_query,
        claim=claim,
        evidence=evidence,
        suffix=suffix,
    )
    _bind_rewrite_authority(observation)

    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert v5["faithfulness"] == 0.0
    assert v5["claim_count"] == 1
    assert v5["supported_claim_count"] == 0
    assert "temporal_receipt_status" not in v5
    assert v6["faithfulness"] == 1.0
    assert v6["claim_count"] == 1
    assert v6["supported_claim_count"] == 1
    assert v6["supported_claim_count"] - v5["supported_claim_count"] == 1
    assert v6["temporal_derived_claim_count"] == 1
    assert v6["temporal_derived_validated_count"] == 1
    assert v6["temporal_receipt_status"] == {
        "contract_version": TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
        "status": "validated",
        "reason_code": "TEMPORAL_DERIVED_CLAIM_VALIDATED",
    }


@pytest.mark.parametrize("field", ["original_query", "display_query", "standalone_query"])
@pytest.mark.parametrize("replacement", ["raw_top", "unrelated"])
def test_v6_binds_every_authoritative_query_field_to_consumed_rewrite(
    field: str,
    replacement: str,
) -> None:
    original_query, rewritten_query, claim, evidence, suffix = _BLIND_CASES[0]
    case = _case(original_query, claim)
    observation = _observation(
        original_query=original_query,
        rewritten_query=rewritten_query,
        claim=claim,
        evidence=evidence,
        suffix=f"{suffix}_{field}_{replacement}",
    )
    _bind_rewrite_authority(observation)
    drift = original_query if replacement == "raw_top" else "unrelated-consumed-query"
    observation["query_understanding"][field] = drift
    observation["query_plan"]["understanding"][field] = drift

    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert _strip_temporal_diagnostics(v6) == v5
    assert v6["temporal_derived_validated_count"] == 0
    assert v6["temporal_receipt_status"]["reason_code"] == ("TEMPORAL_QUERY_BINDING_INVALID")


def _add_context_alias(
    observation: dict[str, Any],
    *,
    evidence_id: str,
    text: str,
) -> None:
    label = f"E{len(observation['context']) + 1}"
    observation["context"].append(
        {
            "context_label": label,
            "evidence_id": evidence_id,
            "evidence": {"evidence_id": evidence_id},
            "text": text,
            "content": text,
        }
    )
    observation["context_pack"]["packed_blocks"].append(
        {"label": label, "evidence_id": evidence_id, "text": text}
    )
    observation["context_pack"]["label_mapping"].append(
        {"label": label, "evidence_id": evidence_id}
    )
    evidence_ids = sorted(
        {item["evidence_id"] for item in observation["context_pack"]["packed_blocks"]}
    )
    inner = observation["completeness_self_review"]["temporal_binary_verdict"]
    inner["authorized_evidence_count"] = len(evidence_ids)
    inner["evidence_set_sha256"] = _digest(_canonical_json(evidence_ids))


def _set_rewritten_query(observation: dict[str, Any], query: str) -> None:
    observation["rewritten_query"] = query
    for field in ("original_query", "display_query", "standalone_query"):
        observation["query_understanding"][field] = query
        observation["query_plan"]["understanding"][field] = query
    observation["completeness_self_review"]["temporal_binary_verdict"]["query_sha256"] = _digest(
        query
    )


def _rehash_answer(observation: dict[str, Any]) -> None:
    answer_hash = _digest(observation["answer"])
    audit = observation["completeness_self_review"]
    audit["materialized_candidate_sha256"] = answer_hash
    audit["security_approved_candidate_sha256"] = answer_hash
    audit["committed_answer_sha256"] = answer_hash


def _mutate(observation: dict[str, Any], mutation: str) -> None:
    audit = observation["completeness_self_review"]
    inner = audit["temporal_binary_verdict"]
    receipt = audit["runtime_completeness_receipt"]
    citation = observation["citations"][0]
    if mutation == "top_query_drift":
        observation["query"] += "漂移"
    elif mutation == "rewritten_drift":
        observation["rewritten_query"] += "漂移"
    elif mutation == "understanding_drift":
        observation["query_understanding"]["standalone_query"] += "漂移"
    elif mutation == "query_hash_top_confusion":
        inner["query_sha256"] = _digest(observation["query"])
    elif mutation == "query_hash":
        inner["query_sha256"] = _digest("unrelated")
    elif mutation == "materialized_hash":
        audit["materialized_candidate_sha256"] = _digest("unrelated")
    elif mutation == "security_hash":
        audit["security_approved_candidate_sha256"] = _digest("unrelated")
    elif mutation == "committed_hash":
        audit["committed_answer_sha256"] = _digest("unrelated")
    elif mutation == "evidence_set_hash":
        inner["evidence_set_sha256"] = _digest("unrelated")
    elif mutation == "selected_evidence_hash":
        inner["selected_evidence_sha256"] = _digest("unrelated")
    elif mutation == "outer_status":
        audit["status"] = "current_complete"
    elif mutation == "inner_status":
        inner["status"] = "not_applicable"
    elif mutation == "rule":
        inner["rule_version"] = "direct-binary-temporal-gating/0.9.0"
    elif mutation == "reason":
        inner["reason_code"] = "OTHER"
    elif mutation == "model_calls":
        receipt["model_calls_added"] = 1
    elif mutation == "provider_calls":
        audit["provider_attempt_count"] = 1
    elif mutation == "provider_response_hash":
        audit["provider_response_sha256"] = _digest("unrelated")
    elif mutation == "zero_call_usage":
        audit["usage"]["status"] = "available"
    elif mutation == "typed_gap":
        receipt["current_uncovered_query_obligation_count"] = True
    elif mutation == "two_gaps":
        receipt["current_uncovered_query_obligation_count"] = 2
    elif mutation == "authorized_count":
        inner["authorized_evidence_count"] = 2
    elif mutation == "qualifying_count":
        inner["qualifying_gate_count"] = 2
    elif mutation == "direct_query_flag":
        inner["direct_binary_query"] = False
    elif mutation == "duplicate_gate":
        _add_context_alias(
            observation,
            evidence_id="ev_duplicate_gate",
            text="必须先等待指示灯熄灭，再打开维护盖。",
        )
        inner["qualifying_gate_count"] = 2
    elif mutation == "conflict":
        _add_context_alias(
            observation,
            evidence_id="ev_conflicting_permission",
            text="现在可以立即打开维护盖。",
        )
        inner["conflict_count"] = 1
    elif mutation == "satisfied":
        query = "机器刚停机，指示灯已经熄灭，能立即打开维护盖吗？"
        _set_rewritten_query(observation, query)
        inner["condition_already_satisfied"] = True
    elif mutation == "ambiguous":
        query = "机器刚停机，指示灯已经恢复，能立即打开维护盖吗？"
        _set_rewritten_query(observation, query)
        inner["condition_state_ambiguous"] = True
    elif mutation == "action":
        claim = "不能立即移除维护盖"
        observation["answer"] = f"{claim} [E1]"
        citation["claim_text"] = claim
        citation["answer_span"]["end"] = len(claim)
        _rehash_answer(observation)
    elif mutation == "citation_label":
        citation["label"] = "E2"
    elif mutation == "citation_evidence":
        citation["evidence_id"] = "ev_other"
    elif mutation == "citation_span":
        citation["answer_span"]["start"] = 1
    elif mutation == "citation_status":
        citation["validation_status"] = "degraded"
    elif mutation == "citation_count":
        observation["citation_validation"]["citation_count"] = 2
    elif mutation == "envelope_citation_id":
        observation["answer_envelope"]["claims"][0]["citation_ids"] = ["cit_2"]
        observation["claims"] = deepcopy(observation["answer_envelope"]["claims"])
    elif mutation == "envelope_citation_span":
        observation["answer_envelope"]["citations"][0]["answer_start"] = 1
    elif mutation == "envelope_answer":
        observation["answer_envelope"]["answer"] += " "
    elif mutation == "envelope_source_identity":
        observation["answer_envelope"]["sources"][0]["evidence_id"] = "ev_other"
    elif mutation == "runtime_duplicate_span":
        observation["citations"].append(deepcopy(citation))
        observation["citation_validation"]["citation_count"] = 2
        observation["citation_validation"]["trusted_count"] = 2
    elif mutation == "runtime_other_span":
        citation["answer_span"] = {
            "start": 1,
            "end": len(citation["claim_text"]) + 1,
        }
    elif mutation == "context_identity":
        observation["context"][0]["evidence_id"] = "ev_other"
    elif mutation == "context_text_drift":
        observation["context"][0]["content"] += "漂移"
    elif mutation == "block_whitespace_drift":
        observation["context_pack"]["packed_blocks"][0]["text"] += " "
    elif mutation == "extra_direct_claim":
        observation["answer"] += "\n不得马上打开维护盖。[E1]"
        _rehash_answer(observation)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("top_query_drift", "TEMPORAL_QUERY_BINDING_INVALID"),
        ("rewritten_drift", "TEMPORAL_QUERY_BINDING_INVALID"),
        ("understanding_drift", "TEMPORAL_QUERY_BINDING_INVALID"),
        ("query_hash_top_confusion", "TEMPORAL_QUERY_HASH_MISMATCH"),
        ("query_hash", "TEMPORAL_QUERY_HASH_MISMATCH"),
        ("materialized_hash", "TEMPORAL_ANSWER_HASH_MISMATCH"),
        ("security_hash", "TEMPORAL_ANSWER_HASH_MISMATCH"),
        ("committed_hash", "TEMPORAL_ANSWER_HASH_MISMATCH"),
        ("evidence_set_hash", "TEMPORAL_EVIDENCE_SET_HASH_MISMATCH"),
        ("selected_evidence_hash", "TEMPORAL_SELECTED_EVIDENCE_HASH_MISMATCH"),
        ("outer_status", "TEMPORAL_OUTER_STATUS_INVALID"),
        ("inner_status", "TEMPORAL_INNER_RECEIPT_INVALID"),
        ("rule", "TEMPORAL_INNER_RECEIPT_INVALID"),
        ("reason", "TEMPORAL_INNER_RECEIPT_INVALID"),
        ("model_calls", "TEMPORAL_INDETERMINATE_RECEIPT_INVALID"),
        ("provider_calls", "TEMPORAL_ZERO_CALL_BOUNDARY_INVALID"),
        ("provider_response_hash", "TEMPORAL_ZERO_CALL_BOUNDARY_INVALID"),
        ("zero_call_usage", "TEMPORAL_ZERO_CALL_BOUNDARY_INVALID"),
        ("typed_gap", "TEMPORAL_INDETERMINATE_RECEIPT_INVALID"),
        ("two_gaps", "TEMPORAL_INDETERMINATE_RECEIPT_INVALID"),
        ("authorized_count", "TEMPORAL_INNER_RECEIPT_INVALID"),
        ("qualifying_count", "TEMPORAL_GATE_NOT_UNIQUE"),
        ("direct_query_flag", "TEMPORAL_INNER_RECEIPT_INVALID"),
        ("duplicate_gate", "TEMPORAL_GATE_NOT_UNIQUE"),
        ("conflict", "TEMPORAL_CONFLICTING_PERMISSION"),
        ("satisfied", "TEMPORAL_CONDITION_STATE_INVALID"),
        ("ambiguous", "TEMPORAL_CONDITION_STATE_INVALID"),
        ("action", "TEMPORAL_QUERY_ACTION_INVALID"),
        ("citation_label", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("citation_evidence", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("citation_span", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("citation_status", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("citation_count", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("envelope_citation_id", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("envelope_citation_span", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("envelope_answer", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("envelope_source_identity", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("runtime_duplicate_span", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("runtime_other_span", "TEMPORAL_CITATION_IDENTITY_INVALID"),
        ("context_identity", "TEMPORAL_CONTEXT_IDENTITY_INVALID"),
        ("context_text_drift", "TEMPORAL_CONTEXT_IDENTITY_INVALID"),
        ("block_whitespace_drift", "TEMPORAL_CONTEXT_IDENTITY_INVALID"),
        ("extra_direct_claim", "TEMPORAL_DIRECT_CLAIM_COUNT_INVALID"),
    ],
)
def test_v6_temporal_receipt_adversarial_mutations_fall_back_to_v5(
    mutation: str,
    reason_code: str,
) -> None:
    original_query, rewritten_query, claim, evidence, suffix = _BLIND_CASES[0]
    case = _case(original_query, claim)
    observation = _observation(
        original_query=original_query,
        rewritten_query=rewritten_query,
        claim=claim,
        evidence=evidence,
        suffix=suffix,
    )
    _bind_rewrite_authority(observation)
    _mutate(observation, mutation)

    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert _strip_temporal_diagnostics(v6) == v5
    assert v6["temporal_derived_validated_count"] == 0
    assert v6["temporal_receipt_status"]["status"] == "rejected"
    assert v6["temporal_receipt_status"]["reason_code"] == reason_code
    assert set(v6["temporal_receipt_status"]) == {
        "contract_version",
        "status",
        "reason_code",
    }


def test_v6_without_receipt_replays_every_v1_3_case_like_v5() -> None:
    manifest = load_dataset_asset(
        "hak180-safety@1.3.0",
        project_root=PROJECT_ROOT,
    ).manifest
    observation = {
        "answer": "",
        "abstained": True,
        "retrieved": [],
        "reranked": [],
        "context": [],
        "latency_ms": 0.0,
        "versions": {},
        "side_effects": {},
    }

    assert len(manifest["cases"]) == 28
    for case in manifest["cases"]:
        v5 = evaluate_case(case, observation, case_policy=V5_POLICY)
        v6 = evaluate_case(case, observation, case_policy=V6_POLICY)
        v6_without_diagnostics = deepcopy(v6)
        generation = v6_without_diagnostics["metrics"]["generation"]
        for field in (
            "temporal_derived_claim_count",
            "temporal_derived_validated_count",
            "temporal_receipt_status",
        ):
            generation.pop(field)
        assert v6_without_diagnostics == v5
        v6_generation = v6["metrics"]["generation"]
        assert v6_generation["temporal_derived_claim_count"] == 0
        assert v6_generation["temporal_derived_validated_count"] == 0
        assert v6_generation["temporal_receipt_status"]["reason_code"] == "TEMPORAL_RECEIPT_ABSENT"

        v5_decision = _case_decision(
            case,
            v5,
            had_error=False,
            case_policy=V5_POLICY,
            observation=observation,
        )
        v6_decision = _case_decision(
            case,
            v6,
            had_error=False,
            case_policy=V6_POLICY,
            observation=observation,
        )
        assert v5_decision.pop("policy_version") == AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION
        assert v6_decision.pop("policy_version") == TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION
        assert v6_decision == v5_decision


def test_v6_aggregate_reports_only_the_three_validated_blind_claims() -> None:
    evaluated = []
    for original_query, rewritten_query, claim, evidence, suffix in _BLIND_CASES:
        observation = _observation(
            original_query=original_query,
            rewritten_query=rewritten_query,
            claim=claim,
            evidence=evidence,
            suffix=suffix,
        )
        _bind_rewrite_authority(observation)
        evaluated.append(
            evaluate_case(_case(original_query, claim), observation, case_policy=V6_POLICY)
        )

    aggregate = aggregate_results(evaluated)["metrics"]["generation"]

    assert aggregate["temporal_derived_claim_count"]["count"] == 3
    assert aggregate["temporal_derived_claim_count"]["mean"] == 1.0
    assert aggregate["temporal_derived_validated_count"]["count"] == 3
    assert aggregate["temporal_derived_validated_count"]["mean"] == 1.0
