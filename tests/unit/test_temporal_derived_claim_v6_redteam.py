from __future__ import annotations

from copy import deepcopy
from typing import Any
import unicodedata

import pytest

from app.evaluation.metrics import evaluate_case
from tests.unit.test_evaluation_case_outcome_v6 import (
    V5_POLICY,
    V6_POLICY,
    _BLIND_CASES,
    _add_context_alias,
    _bind_rewrite_authority,
    _case,
    _canonical_json,
    _digest,
    _observation,
    _set_rewritten_query,
    _strip_temporal_diagnostics,
)


def _valid_subject() -> tuple[dict[str, Any], dict[str, Any]]:
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
    return case, observation


def _assert_falls_back(case: dict[str, Any], observation: dict[str, Any]) -> None:
    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert _strip_temporal_diagnostics(v6) == v5
    assert v6["temporal_derived_validated_count"] == 0
    assert v6["temporal_receipt_status"]["status"] == "rejected"
    assert set(v6["temporal_receipt_status"]) == {
        "contract_version",
        "status",
        "reason_code",
    }


def _assert_validated(case: dict[str, Any], observation: dict[str, Any]) -> None:
    generation = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]
    assert generation["temporal_derived_claim_count"] == 1
    assert generation["temporal_derived_validated_count"] == 1
    assert generation["temporal_receipt_status"]["reason_code"] == (
        "TEMPORAL_DERIVED_CLAIM_VALIDATED"
    )


@pytest.mark.parametrize(
    "rewrite",
    [
        None,
        {"status": "rejected", "reasons": ["rewrite_not_equivalent"]},
        {"status": "accepted", "reasons": ["semantic_expansion"]},
    ],
)
def test_v6_fails_closed_when_rewrite_acceptance_is_not_proven(
    rewrite: dict[str, object] | None,
) -> None:
    case, observation = _valid_subject()
    understanding = observation["query_understanding"]
    assert isinstance(understanding, dict)
    plan = observation["query_plan"]
    assert isinstance(plan, dict)
    plan_understanding = plan["understanding"]
    assert isinstance(plan_understanding, dict)
    if rewrite is None:
        understanding.pop("rewrite")
        plan_understanding.pop("rewrite")
    else:
        understanding["rewrite"] = deepcopy(rewrite)
        plan_understanding["rewrite"] = deepcopy(rewrite)

    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert _strip_temporal_diagnostics(v6) == v5
    assert v6["temporal_derived_validated_count"] == 0
    assert v6["temporal_receipt_status"] == {
        "contract_version": "stt.eval.temporal-derived-claim.v1",
        "status": "rejected",
        "reason_code": "TEMPORAL_QUERY_BINDING_INVALID",
    }


def test_v6_fails_closed_when_query_plan_understanding_disagrees() -> None:
    case, observation = _valid_subject()
    understanding = observation["query_understanding"]
    assert isinstance(understanding, dict)
    plan = observation["query_plan"]
    assert isinstance(plan, dict)
    plan_understanding = plan["understanding"]
    assert isinstance(plan_understanding, dict)
    plan_understanding["standalone_query"] = "tampered-consumed-query"

    v5 = evaluate_case(case, observation, case_policy=V5_POLICY)["metrics"]["generation"]
    v6 = evaluate_case(case, observation, case_policy=V6_POLICY)["metrics"]["generation"]

    assert _strip_temporal_diagnostics(v6) == v5
    assert v6["temporal_derived_validated_count"] == 0
    assert v6["temporal_receipt_status"]["reason_code"] == "TEMPORAL_QUERY_BINDING_INVALID"


def _bind_real_consumed_query_shape(observation: dict[str, Any]) -> None:
    rewritten_query = observation["rewritten_query"]
    understanding = observation["query_understanding"]
    plan_understanding = observation["query_plan"]["understanding"]
    for target in (understanding, plan_understanding):
        target["original_query"] = rewritten_query
        target["display_query"] = rewritten_query
        target["standalone_query"] = rewritten_query


def test_v6_accepts_real_shape_when_raw_top_and_consumed_query_differ() -> None:
    case, observation = _valid_subject()
    assert observation["query"] != observation["rewritten_query"]
    _bind_real_consumed_query_shape(observation)

    _assert_validated(case, observation)


@pytest.mark.parametrize(
    ("projection", "field", "replacement"),
    [
        (projection, field, replacement)
        for projection in ("query_understanding", "query_plan")
        for field in ("original_query", "display_query", "standalone_query")
        for replacement in ("raw_top", "drift")
    ],
)
def test_v6_rejects_any_consumed_query_projection_drift(
    projection: str,
    field: str,
    replacement: str,
) -> None:
    case, observation = _valid_subject()
    _bind_real_consumed_query_shape(observation)
    target = (
        observation["query_understanding"]
        if projection == "query_understanding"
        else observation["query_plan"]["understanding"]
    )
    target[field] = observation["query"] if replacement == "raw_top" else "drifted-query"

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_schema",
        "known_boolean_count",
        "known_nan_count",
        "extra_receipt_field",
        "extra_inner_field",
    ],
)
def test_v6_fails_closed_on_versioned_receipt_shape_and_types(mutation: str) -> None:
    case, observation = _valid_subject()
    audit = observation["completeness_self_review"]
    receipt = audit["runtime_completeness_receipt"]
    inner = audit["temporal_binary_verdict"]
    if mutation == "outer_schema":
        audit["schema_version"] = "completeness-self-review-audit.unknown"
    elif mutation == "known_boolean_count":
        receipt["candidate_count"] = True
    elif mutation == "known_nan_count":
        receipt["search_state_count"] = float("nan")
    elif mutation == "extra_receipt_field":
        receipt["unexpected_contract_field"] = 0
    elif mutation == "extra_inner_field":
        inner["unexpected_contract_field"] = 0
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(mutation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    ("mode", "value", "expected_valid"),
    [
        ("missing", None, True),
        ("present", 0, True),
        ("present", 1, False),
        ("present", True, False),
        ("present", float("nan"), False),
    ],
)
def test_v6_floor_coverage_credit_is_missing_or_exact_integer_zero(
    mode: str,
    value: object,
    expected_valid: bool,
) -> None:
    case, observation = _valid_subject()
    receipt = observation["completeness_self_review"]["runtime_completeness_receipt"]
    if mode == "missing":
        receipt.pop("trusted_floor_coverage_credit_count")
    else:
        receipt["trusted_floor_coverage_credit_count"] = value

    if expected_valid:
        _assert_validated(case, observation)
    else:
        _assert_falls_back(case, observation)


def _historical_indeterminate_floor_subject() -> tuple[dict[str, Any], dict[str, Any]]:
    case, observation = _valid_subject()
    receipt = observation["completeness_self_review"]["runtime_completeness_receipt"]
    current_candidate_count = receipt["current_candidate_count"]
    receipt.pop("trusted_floor_coverage_credit_count", None)
    receipt.update(
        {
            "initial_floor_candidate_count": 0,
            "residual_selection_count": 0,
            "selection_count": 0,
            "solution_count": 0,
            "canonical_selection_count": 0,
            "materializer_validation_count": 0,
            "materializer_rejection_count": 0,
            "search_state_count": 0,
            "zero_bindable_facet_count": 0,
            "selection_hash": "",
            "canonical_selection_hash": "",
            "canonicalization_policy": "",
            "materializer_rejection_reason_hash": "",
            "materializer_rejection_reason_counts": {},
            "semantic_absence_proven": False,
            "abstention_authorized": False,
            "trusted_floor_ambiguous_claim_count": 0,
            "trusted_floor_unmapped_claim_count": 0,
            "trusted_floor_mapping_complete": True,
            "trusted_floor_projection_available": True,
            "trusted_floor_source": "current",
            "trusted_floor_trusted_claim_count": current_candidate_count,
            "trusted_floor_mapped_claim_count": current_candidate_count,
            "trusted_floor_candidate_count": current_candidate_count,
        }
    )
    return case, observation


def test_v6_accepts_historical_indeterminate_floor_receipt_contract() -> None:
    case, observation = _historical_indeterminate_floor_subject()
    receipt = observation["completeness_self_review"]["runtime_completeness_receipt"]

    assert "trusted_floor_coverage_credit_count" not in receipt
    _assert_validated(case, observation)


@pytest.mark.parametrize(
    "mutation",
    [
        "nonzero:trusted_floor_coverage_credit_count",
        "nonzero:initial_floor_candidate_count",
        "nonzero:residual_selection_count",
        "nonzero:selection_count",
        "nonzero:solution_count",
        "nonzero:canonical_selection_count",
        "nonzero:materializer_validation_count",
        "nonzero:materializer_rejection_projection",
        "nonzero:search_state_count",
        "nonzero:zero_bindable_facet_count",
        "nonempty:selection_hash",
        "nonempty:canonical_selection_hash",
        "nonempty:canonicalization_policy",
        "nonempty:materializer_rejection_reason_hash",
        "true:semantic_absence_proven",
        "true:abstention_authorized",
        "nonzero:trusted_floor_ambiguous_claim_count",
        "nonzero:trusted_floor_unmapped_claim_count",
        "false:trusted_floor_mapping_complete",
        "false:trusted_floor_projection_available",
        "drift:trusted_floor_source",
        "mismatch:trusted_floor_trusted_claim_count",
        "mismatch:trusted_floor_mapped_claim_count",
        "mismatch:trusted_floor_candidate_count",
    ],
)
def test_v6_rejects_historical_indeterminate_floor_receipt_drift(mutation: str) -> None:
    case, observation = _historical_indeterminate_floor_subject()
    receipt = observation["completeness_self_review"]["runtime_completeness_receipt"]
    operation, field = mutation.split(":", maxsplit=1)
    if field == "materializer_rejection_projection":
        reason_counts = {"SYNTHETIC_REJECTION": 1}
        receipt["materializer_rejection_count"] = 1
        receipt["materializer_rejection_reason_counts"] = reason_counts
        receipt["materializer_rejection_reason_hash"] = _digest(
            _canonical_json([{"reason_code": "SYNTHETIC_REJECTION", "count": 1}])
        )
    elif operation == "nonzero":
        receipt[field] = 1
    elif operation == "nonempty":
        receipt[field] = "synthetic-drift"
    elif operation == "true":
        receipt[field] = True
    elif operation == "false":
        receipt[field] = False
    elif operation == "drift":
        receipt[field] = "none"
    elif operation == "mismatch":
        receipt[field] = receipt["current_candidate_count"] + 1
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(mutation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_citation_start",
        "floating_citation_end",
        "numeric_claim_id",
        "numeric_citation_id",
    ],
)
def test_v6_fails_closed_on_envelope_identifier_and_span_types(mutation: str) -> None:
    case, observation = _valid_subject()
    envelope = observation["answer_envelope"]
    envelope_claim = envelope["claims"][0]
    projected_claim = observation["claims"][0]
    envelope_citation = envelope["citations"][0]
    if mutation == "boolean_citation_start":
        envelope_citation["answer_start"] = False
    elif mutation == "floating_citation_end":
        envelope_citation["answer_end"] = float(envelope_citation["answer_end"])
    elif mutation == "numeric_claim_id":
        envelope_claim["claim_id"] = 1
        projected_claim["claim_id"] = 1
        envelope_citation["claim_id"] = "1"
    elif mutation == "numeric_citation_id":
        envelope_claim["citation_ids"] = ["1"]
        projected_claim["citation_ids"] = ["1"]
        envelope_citation["citation_id"] = 1
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(mutation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    "mutation",
    [
        "unreferenced_duplicate_envelope_citation",
        "overlapping_runtime_citation",
        "overlapping_envelope_claim",
    ],
)
def test_v6_fails_closed_on_claim_citation_chain_ambiguity(mutation: str) -> None:
    case, observation = _valid_subject()
    envelope = observation["answer_envelope"]
    if mutation == "unreferenced_duplicate_envelope_citation":
        duplicate = deepcopy(envelope["citations"][0])
        duplicate["citation_id"] = "cit_2"
        envelope["citations"].append(duplicate)
    elif mutation == "overlapping_runtime_citation":
        duplicate = deepcopy(observation["citations"][0])
        duplicate["answer_span"]["end"] -= 1
        duplicate["claim_text"] = duplicate["claim_text"][:-1]
        observation["citations"].append(duplicate)
        observation["citation_validation"]["citation_count"] = 2
        observation["citation_validation"]["trusted_count"] = 2
    elif mutation == "overlapping_envelope_claim":
        claim = envelope["claims"][0]
        overlapping = {
            "claim_id": "clm_2",
            "text": claim["text"][1:],
            "answer_start": 1,
            "answer_end": claim["answer_end"],
            "claim_type": "factual",
            "citation_ids": ["cit_2"],
            "supported": True,
        }
        citation = deepcopy(envelope["citations"][0])
        citation.update(
            {
                "citation_id": "cit_2",
                "claim_id": "clm_2",
                "answer_start": 1,
            }
        )
        envelope["claims"].append(overlapping)
        envelope["citations"].append(citation)
        observation["claims"] = deepcopy(envelope["claims"])
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(mutation)

    _assert_falls_back(case, observation)


def test_v6_rejects_one_evidence_identity_mapped_to_multiple_labels() -> None:
    case, observation = _valid_subject()
    evidence_id = observation["context"][0]["evidence_id"]
    _add_context_alias(
        observation,
        evidence_id=evidence_id,
        text="附加说明只记录维护日期。",
    )

    _assert_falls_back(case, observation)


def test_v6_context_order_is_irrelevant_after_identity_binding() -> None:
    case, observation = _valid_subject()
    _add_context_alias(
        observation,
        evidence_id="ev_blind_neutral_second",
        text="附加说明只记录维护日期。",
    )
    observation["context"].reverse()
    observation["context_pack"]["packed_blocks"].reverse()
    observation["context_pack"]["label_mapping"].reverse()

    _assert_validated(case, observation)


def test_v6_hashes_the_exact_unicode_query_without_normalized_aliases() -> None:
    case, observation = _valid_subject()
    exact_query = observation["rewritten_query"].replace("能立即", "能\N{NO-BREAK SPACE}立即")
    _set_rewritten_query(observation, exact_query)

    _assert_validated(case, observation)

    normalized = unicodedata.normalize("NFKC", exact_query)
    assert normalized != exact_query
    inner = observation["completeness_self_review"]["temporal_binary_verdict"]
    inner["query_sha256"] = _digest(normalized)
    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    "scope",
    ["claim_reporting", "query_reporting", "evidence_reporting"],
)
def test_v6_rejects_reporting_about_an_action_as_the_action_itself(scope: str) -> None:
    original_query, rewritten_query, claim, evidence, suffix = _BLIND_CASES[0]
    if scope == "claim_reporting":
        claim = "现在不能声称立即打开维护盖"
    elif scope == "query_reporting":
        original_query = rewritten_query = "机器刚停机，能立即声称打开维护盖吗？"
    elif scope == "evidence_reporting":
        evidence = "必须先等待指示灯熄灭，再声称打开维护盖。"
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(scope)
    case = _case(original_query, claim)
    observation = _observation(
        original_query=original_query,
        rewritten_query=rewritten_query,
        claim=claim,
        evidence=evidence,
        suffix=f"{suffix}_{scope}",
    )
    _bind_rewrite_authority(observation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    ("scope", "reporting_verb"),
    [
        (scope, reporting_verb)
        for scope in ("claim", "query", "evidence")
        for reporting_verb in ("讲解", "介绍", "说明", "讨论", "提及")
    ],
)
def test_v6_rejects_natural_meta_actions_structurally(
    scope: str,
    reporting_verb: str,
) -> None:
    original_query, rewritten_query, claim, evidence, suffix = _BLIND_CASES[0]
    if scope == "claim":
        claim = f"现在不能立即{reporting_verb}打开维护盖的步骤"
    elif scope == "query":
        original_query = rewritten_query = f"机器刚停机，能立即{reporting_verb}打开维护盖的步骤吗？"
    elif scope == "evidence":
        evidence = f"必须先等待指示灯熄灭，再{reporting_verb}打开维护盖的步骤。"
    else:  # pragma: no cover - 参数表受本文件冻结
        raise AssertionError(scope)
    case = _case(original_query, claim)
    observation = _observation(
        original_query=original_query,
        rewritten_query=rewritten_query,
        claim=claim,
        evidence=evidence,
        suffix=f"{suffix}_{scope}_{reporting_verb}",
    )
    _bind_rewrite_authority(observation)

    _assert_falls_back(case, observation)


def test_v6_accepts_an_exact_cjk_action_core_without_fixture_vocabulary() -> None:
    query = "舱门刚操作结束，现在能立即拔出锁销吗？"
    claim = "现在不能拔出锁销"
    evidence = "必须先等待压力归零，再拔出锁销。"
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix="cn_exact_action_core",
    )
    _bind_rewrite_authority(observation)

    _assert_validated(case, observation)


@pytest.mark.parametrize(
    ("verb", "inflected", "object_name", "suffix"),
    [
        ("open", "opening", "the service panel", "open"),
        ("remove", "removing", "the safety cover", "remove"),
        ("close", "closing", "the service hatch", "close"),
        ("stop", "stopping", "the cooling fan", "stop"),
    ],
)
def test_v6_accepts_regular_ascii_action_inflections(
    verb: str,
    inflected: str,
    object_name: str,
    suffix: str,
) -> None:
    query = f"The unit has just finished running. Can the operator {verb} {object_name} right now?"
    claim = f"Right now the machine cannot {verb} {object_name}"
    evidence = f"You must wait for the indicator to turn off before {inflected} {object_name}."
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix=f"ascii_inflection_{suffix}",
    )
    _bind_rewrite_authority(observation)

    _assert_validated(case, observation)


@pytest.mark.parametrize(
    ("claim_verb", "evidence_verb", "object_name", "suffix"),
    [
        ("remove", "moving", "the safety cover", "remove_move"),
        ("close", "enclosing", "the service hatch", "close_enclose"),
        ("stop", "restarting", "the cooling fan", "stop_restart"),
        ("open", "reopening", "the service panel", "open_reopen"),
    ],
)
def test_v6_does_not_merge_different_ascii_action_cores(
    claim_verb: str,
    evidence_verb: str,
    object_name: str,
    suffix: str,
) -> None:
    query = (
        "The unit has just finished running. "
        f"Can the operator {claim_verb} {object_name} right now?"
    )
    claim = f"Right now the machine cannot {claim_verb} {object_name}"
    evidence = f"You must wait for the indicator to turn off before {evidence_verb} {object_name}."
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix=f"ascii_distinct_{suffix}",
    )
    _bind_rewrite_authority(observation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    ("query", "claim", "evidence", "suffix"),
    [
        (
            "模块刚运行结束，现在能立即打开USB维护盖吗？",
            "现在不能打开USB维护盖",
            "必须先等待指示灯熄灭，再关闭USB维护盖。",
            "different_verb",
        ),
        (
            "模块刚运行结束，现在能立即打开内部USB维护盖吗？",
            "现在不能打开内部USB维护盖",
            "必须先等待指示灯熄灭，再打开外部USB维护盖。",
            "different_object",
        ),
        (
            "模块刚运行结束，现在能立即打开USB维护盖吗？",
            "现在不能立即讲解打开USB维护盖",
            "必须先等待指示灯熄灭，再打开USB维护盖。",
            "claim_meta",
        ),
        (
            "模块刚运行结束，现在能立即讲解打开USB维护盖吗？",
            "现在不能打开USB维护盖",
            "必须先等待指示灯熄灭，再打开USB维护盖。",
            "query_meta",
        ),
        (
            "模块刚运行结束，现在能立即打开USB维护盖吗？",
            "现在不能打开USB维护盖",
            "必须先等待指示灯熄灭，再讲解打开USB维护盖。",
            "evidence_meta",
        ),
    ],
)
def test_v6_does_not_erase_cjk_action_core_when_ascii_is_present(
    query: str,
    claim: str,
    evidence: str,
    suffix: str,
) -> None:
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix=f"mixed_negative_{suffix}",
    )
    _bind_rewrite_authority(observation)

    _assert_falls_back(case, observation)


@pytest.mark.parametrize(
    ("query", "claim", "evidence", "suffix"),
    [
        (
            "模块刚运行结束，现在能立即打开USB维护盖吗？",
            "现在不能打开USB维护盖",
            "必须先等待指示灯熄灭，再打开USB维护盖。",
            "usb_exact",
        ),
        (
            "模块刚运行结束，现在能立即开始 remove the safety cover吗？",
            "现在不能开始 remove a safety cover",
            "必须先等待指示灯熄灭，再开始 removing the safety cover。",
            "cjk_ascii_inflection",
        ),
    ],
)
def test_v6_accepts_matching_cjk_core_with_compatible_ascii_tokens(
    query: str,
    claim: str,
    evidence: str,
    suffix: str,
) -> None:
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix=f"mixed_positive_{suffix}",
    )
    _bind_rewrite_authority(observation)

    _assert_validated(case, observation)


@pytest.mark.parametrize(
    ("query", "claim", "evidence", "suffix"),
    [
        (
            "传送带刚运行结束，现在能立即拔下安全钥匙吗？",
            "现在不能拔下安全钥匙",
            "需要锁销归位之后才可以拔下安全钥匙。",
            "cn_key",
        ),
        (
            "The chamber has just stopped. Can the operator open the pressure hatch right now?",
            "Right now the machine cannot open the pressure hatch",
            "Only after the gauge resets may the machine open the pressure hatch.",
            "en_hatch",
        ),
    ],
)
def test_v6_accepts_diverse_temporal_gate_wording(
    query: str,
    claim: str,
    evidence: str,
    suffix: str,
) -> None:
    case = _case(query, claim)
    observation = _observation(
        original_query=query,
        rewritten_query=query,
        claim=claim,
        evidence=evidence,
        suffix=suffix,
    )
    _bind_rewrite_authority(observation)

    _assert_validated(case, observation)
