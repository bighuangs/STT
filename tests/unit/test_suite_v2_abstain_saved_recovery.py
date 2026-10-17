from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


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

import review_contracts  # noqa: E402


def _review(*, suffix: str = "", with_numeric: bool = True) -> tuple[dict, dict]:
    concept_text = f"资料未提供实时状态，无法确定{suffix}"
    payload = {
        "query": "当前实时状态是什么？",
        "reference_answer": "资料未提供实时状态，无法确定。",
        "unanswerable": True,
        "legacy_expected_terms": [],
        "forbidden_terms": [],
        "branches": [{"branch_label": "B1", "query_span": "当前实时状态是什么"}],
        "evidence": [],
    }
    value = {
        "verdict": "approve",
        "answerability": "abstain",
        "required_concepts": {
            "all_of": [
                {
                    "concept_key": "c01",
                    "canonical_text": concept_text,
                    "any_of": [concept_text],
                    "evidence_labels": [],
                    "branch_labels": ["B1"],
                }
            ]
        },
        "critical_constraints": {
            "numerics": (
                [{"value": "2026", "unit": None, "evidence_labels": []}]
                if with_numeric
                else []
            ),
            "identifiers": [{"value": f"P3500{suffix}", "evidence_labels": []}],
            "polarity": {
                "required": "abstain_without_factual_claims",
                "cues": ["无法确定"],
            },
        },
        "required_branches": [
            {
                "branch_label": "B1",
                "query_span": "当前实时状态是什么",
                "required_concept_keys": ["c01"],
            }
        ],
        "evidence_concept_coverage": [],
        "review_notes": [f"资料不足，只允许拒答{suffix}"],
    }
    return value, payload


def _failed_row(*, custom_id: str, reviews: list[dict], payload: dict) -> dict:
    helper = __import__("test_suite_v2_concept_contract")
    row = helper._raw_provenance_semantic_quota_failed_row(
        custom_id=custom_id,
        prompt_sha256="prompt",
        reviews=reviews,
        payload=payload,
    )
    row["sequence"] = 146
    row["runner_version"] = "2.29.0-real-sse-calibration-ledger"
    row["transport"] = review_contracts.TRANSPORT
    row["sse_completion_policy"] = review_contracts.SSE_COMPLETION_POLICY
    row["error"] = "ValueError: " + "; ".join(
        row["invalid_attempts"][-1]["validation_errors"]
    )
    for item, failure, integrity in zip(
        row["invalid_attempts"],
        row["failure_attempts"],
        row["sse_integrity_attempts"],
        strict=True,
    ):
        failure["error_type"] = "ValueError"
        integrity.update(
            completion_policy=review_contracts.SSE_COMPLETION_POLICY,
            finish_reasons=["stop"],
            event_count=4,
            content_event_count=2,
            content_chars=len(
                json.dumps(
                    item["raw_review"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
            usage_integrity={
                "status": "passed",
                "usage_event_count": 1,
                "required_token_fields_valid": True,
                "total_matches_prompt_plus_completion": True,
                "at_or_after_finish_reason_stop": True,
            },
        )
        item["sse_integrity"] = deepcopy(integrity)
    row["sse_integrity"] = deepcopy(row["sse_integrity_attempts"][-1])
    return row


def _request_row(custom_id: str, payload: dict) -> dict:
    return {
        "custom_id": custom_id,
        "prompt_sha256": "prompt",
        "request": {
            "messages": [
                {"role": "system", "content": "blind"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        },
    }


def test_abstain_constraint_projector_replaces_text_with_sentinel_and_constraints() -> None:
    value, payload = _review()
    projector = getattr(
        review_contracts, "_canonicalize_saved_abstain_factual_constraints"
    )

    projected, stages, raw_errors = projector(value, payload)

    assert raw_errors == [
        "abstain concept[0] variants must each express only an explicit "
        "absence/uncertainty rationale",
        "abstain response must not require factual numerics or identifiers",
    ]
    concept = projected["required_concepts"]["all_of"][0]
    assert concept["canonical_text"] == "资料不足，无法依据已授权证据确定答案"
    assert concept["any_of"] == ["资料不足", "未提供相关信息", "无法确定"]
    assert projected["critical_constraints"]["polarity"] == {
        "required": "abstain_without_factual_claims",
        "cues": ["资料不足", "无法确定"],
    }
    assert projected["critical_constraints"]["numerics"] == []
    assert projected["critical_constraints"]["identifiers"] == []
    unchanged = deepcopy(projected)
    unchanged_concept = unchanged["required_concepts"]["all_of"][0]
    raw_concept = value["required_concepts"]["all_of"][0]
    unchanged_concept["canonical_text"] = raw_concept["canonical_text"]
    unchanged_concept["any_of"] = deepcopy(raw_concept["any_of"])
    unchanged["critical_constraints"]["polarity"] = deepcopy(
        value["critical_constraints"]["polarity"]
    )
    unchanged["critical_constraints"]["numerics"] = deepcopy(
        value["critical_constraints"]["numerics"]
    )
    unchanged["critical_constraints"]["identifiers"] = deepcopy(
        value["critical_constraints"]["identifiers"]
    )
    assert unchanged == value
    assert len(stages) == 1
    assert len(stages[0]["actions"]) == 1
    assert stages[0]["actions"][0]["json_paths"] == [
        "/required_concepts/all_of/0/canonical_text",
        "/required_concepts/all_of/0/any_of",
        "/critical_constraints/polarity",
        "/critical_constraints/numerics",
        "/critical_constraints/identifiers",
    ]
    assert review_contracts._validate_review(projected) == []
    assert review_contracts._validate_against_payload(projected, payload) == []


def test_abstain_constraint_projector_does_not_read_reference_or_candidate() -> None:
    value, payload = _review()
    altered = deepcopy(payload)
    altered.update(
        case_id="forbidden-case-identity",
        reference_answer="完全不同的参考答案",
        deterministic_candidate={"critical_constraints": "forbidden authority"},
    )
    projector = getattr(
        review_contracts, "_canonicalize_saved_abstain_factual_constraints"
    )

    projected, stages, errors = projector(value, altered)

    assert projected["critical_constraints"]["numerics"] == []
    assert projected["critical_constraints"]["identifiers"] == []
    assert len(stages) == 1
    assert errors[-1] == (
        "abstain response must not require factual numerics or identifiers"
    )


def test_v2_projector_accepts_only_sentinel_mismatch_with_empty_constraints() -> None:
    value, payload = _review()
    value["critical_constraints"]["numerics"] = []
    value["critical_constraints"]["identifiers"] = []

    projected, stages, errors = (
        review_contracts._canonicalize_saved_abstain_factual_constraints(
            value, payload
        )
    )

    assert errors == [
        "abstain concept[0] variants must each express only an explicit "
        "absence/uncertainty rationale"
    ]
    assert len(stages) == 1
    assert projected["required_concepts"]["all_of"][0]["canonical_text"] == (
        "资料不足，无法依据已授权证据确定答案"
    )
    assert review_contracts._validate_review(projected) == []
    assert review_contracts._validate_against_payload(projected, payload) == []


def test_abstain_v2_validator_accepts_only_complete_frozen_sentinel() -> None:
    value, _payload = _review()
    concept = value["required_concepts"]["all_of"][0]
    concept["canonical_text"] = "资料不足，无法依据已授权证据确定答案"
    concept["any_of"] = ["资料不足", "未提供相关信息", "无法确定"]
    value["critical_constraints"]["numerics"] = []
    value["critical_constraints"]["identifiers"] = []
    value["critical_constraints"]["polarity"] = {
        "required": "abstain_without_factual_claims",
        "cues": ["资料不足", "无法确定"],
    }

    assert review_contracts._validate_review(value) == []


@pytest.mark.parametrize(
    "text",
    [
        "手册给出实时状态为正常",
        "P3500实时状态",
        "当前实时状态是什么",
        "实时状态",
        "手册未提供实时价格，但价格是2999元",
        "没有说明无线能力，但设备支持Wi-Fi",
        "并非未提供价格，而是明确给出2999元",
        "无线信息显示设备支持Wi-Fi",
        "手册未提供实时价格且价格是2999元",
        "手册未提供实时价格并明确给出价格2999元",
        "The manual does not provide a price and the price is 2999.",
        "未提供实时价格：价格是2999元",
        "未提供实时价格（价格是2999元）",
        "未提供实时价格/价格是2999元",
        "未提供实时价格以及价格是2999元",
        "未提供实时价格却明确给出价格2999元",
        "未提供实时价格而价格是2999元",
        "未提供实时价格只是价格是2999元",
        "无法确定价格：价格是2999元",
        "未说明无线能力却明确支持Wi-Fi",
        "The manual does not provide a price: the price is 2999.",
        "The manual does not provide a price yet the price is 2999.",
        "The manual does not provide a price while the price is 2999.",
        "The manual does not provide a price also the price is 2999.",
        "手册未提供实时状态正常",
        "手册未提供无线能力具备无线网络",
        "手册未提供实时价格售价三千",
        "手册未提供实时价格等于三千",
        "未提供实时价格三千",
        "未提供实时价格零",
        "The manual does not provide a price although it is expensive.",
        "未提供设备具备无线网络功能",
        "未提供设备已经开机状态",
        "未提供设备可用状态",
        "未提供实时价格等于市场价格",
        "未提供打印速度超过市场规格",
        "未提供设备允许双面打印功能",
        "未提供机器保持在线状态",
        "未提供产品优于竞品的证据",
        "未提供无线能力—设备具备无线网络功能",
        "未提供无线能力–设备具备无线网络功能",
        "未提供无线能力…设备具备无线网络功能",
        "未提供无线能力｜设备具备无线网络功能",
        "未提供无线能力|设备具备无线网络功能",
        "未提供无线能力·设备具备无线网络功能",
        "未提供无线能力•设备具备无线网络功能",
        "未提供无线能力\n设备具备无线网络功能",
        "未提供无线能力\t设备具备无线网络功能",
        "未提供无线能力﹔设备具备无线网络功能",
        "未提供无线能力､设备具备无线网络功能",
        "未提供无线能力、设备具备无线网络功能",
    ],
)
def test_abstain_refusal_cue_vocabulary_rejects_facts_and_query_nouns(text) -> None:
    value, _payload = _review()
    concept = value["required_concepts"]["all_of"][0]
    concept["canonical_text"] = text
    concept["any_of"] = [text]
    value["critical_constraints"]["numerics"] = []
    value["critical_constraints"]["identifiers"] = []

    assert any(
        "variants must each express" in error
        for error in review_contracts._validate_review(value)
    )


@pytest.mark.parametrize(
    "text",
    [
        "手册未提供五年期故障统计、概率模型或维修价格信息，无法据此计算年度故障概率和平均维修费用",
        "手册没有给出五年期故障统计、概率模型或维修价格",
        "无法据此计算年度故障概率和平均维修费用",
        "手册中缺少相关数据，无法计算",
        "没有给出相关信息，无法计算",
        "未提供无线能力—设备具备无线网络功能",
    ],
)
def test_v2_projector_never_preserves_model_authored_abstention_text(text) -> None:
    value, payload = _review()
    concept = value["required_concepts"]["all_of"][0]
    concept["canonical_text"] = text
    concept["any_of"] = [text]

    projected, stages, _errors = (
        review_contracts._canonicalize_saved_abstain_factual_constraints(
            value, payload
        )
    )

    assert len(stages) == 1
    projected_concept = projected["required_concepts"]["all_of"][0]
    assert projected_concept["canonical_text"] == (
        "资料不足，无法依据已授权证据确定答案"
    )
    assert projected_concept["any_of"] == [
        "资料不足",
        "未提供相关信息",
        "无法确定",
    ]
    assert text not in json.dumps(projected_concept, ensure_ascii=False)


@pytest.mark.parametrize(
    "cues",
    [
        ["价格是2999元"],
        ["设备支持Wi-Fi"],
        ["无法确定—价格是2999元"],
        [""],
        ["无法确定", "价格是2999元"],
    ],
)
def test_v2_projector_freezes_untrusted_polarity_cues(cues) -> None:
    value, payload = _review()
    value["critical_constraints"]["polarity"]["cues"] = cues

    projected, stages, _errors = (
        review_contracts._canonicalize_saved_abstain_factual_constraints(
            value, payload
        )
    )

    assert len(stages) == 1
    assert projected["critical_constraints"]["polarity"] == {
        "required": "abstain_without_factual_claims",
        "cues": ["资料不足", "无法确定"],
    }


def test_abstain_constraint_recovery_selects_final_attempt_without_consensus() -> None:
    first, payload = _review()
    second, _ = _review(with_numeric=False)
    final, _ = _review()
    first["review_notes"] = ["资料不足，只允许拒答（第一版）"]
    second["review_notes"] = ["资料不足，只允许拒答（第二版）"]
    final["review_notes"] = ["资料不足，只允许拒答（最终版）"]
    custom_id = "saved-abstain-factual-constraints"
    failed = _failed_row(custom_id=custom_id, reviews=[first, second, final], payload=payload)

    recovered = review_contracts._recover_prior_invalid_review(
        row=_request_row(custom_id, payload), response_rows=[failed]
    )

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    audit = recovered["saved_response_abstain_factual_constraint_recovery"]
    assert audit["selection"] == {
        "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
        "selected_attempt": 3,
        "semantic_consensus_claimed": False,
        "cross_attempt_merge": False,
    }
    assert len(audit["attempt_shape_receipts"]) == 3
    assert len(
        {item["projected_review_sha256"] for item in audit["attempt_shape_receipts"]}
    ) == 3
    assert "最终版" in json.dumps(recovered["review"], ensure_ascii=False)
    assert "第一版" not in json.dumps(recovered["review"], ensure_ascii=False)
    validator = getattr(
        review_contracts, "_validate_abstain_factual_constraint_recovery_audit"
    )
    assert validator(recovered, payload, source_response_rows=[failed]) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(answerability="answer"),
        lambda value: value.update(verdict="reject"),
        lambda value: value["required_concepts"]["all_of"].append(
            deepcopy(value["required_concepts"]["all_of"][0])
        ),
        lambda value: value["required_concepts"]["all_of"][0].update(
            evidence_labels=["E1"]
        ),
        lambda value: value["evidence_concept_coverage"].append(
            {"evidence_label": "E1", "required_concept_keys": [], "status": "uncovered"}
        ),
        lambda value: value["critical_constraints"]["polarity"].update(
            required="negative"
        ),
        lambda value: value["required_branches"].clear(),
        lambda value: value["critical_constraints"]["numerics"][0].update(
            evidence_labels=["E1"]
        ),
        lambda value: value["critical_constraints"]["identifiers"][0].update(
            value=""
        ),
    ],
)
def test_abstain_constraint_projector_rejects_broader_or_semantic_shapes(mutator) -> None:
    value, payload = _review()
    mutator(value)
    projector = getattr(
        review_contracts, "_canonicalize_saved_abstain_factual_constraints"
    )

    projected, stages, _errors = projector(value, payload)

    assert projected == value
    assert stages == []


def test_v2_projector_requires_payload_unanswerable_boundary() -> None:
    value, payload = _review()
    payload["unanswerable"] = False

    projected, stages, _errors = (
        review_contracts._canonicalize_saved_abstain_factual_constraints(
            value, payload
        )
    )

    assert projected == value
    assert stages == []


def test_abstain_constraint_recovery_rejects_extra_error_nonfinal_and_old_row() -> None:
    value, payload = _review()
    custom_id = "saved-abstain-guards"
    request = _request_row(custom_id, payload)

    extra_error = _failed_row(
        custom_id=custom_id, reviews=[value, value], payload=payload
    )
    extra_error["invalid_attempts"][0]["validation_errors"].append("extra")
    assert review_contracts._recover_prior_invalid_review(
        row=request, response_rows=[extra_error]
    ) is None

    final_invalid = deepcopy(value)
    final_invalid["critical_constraints"]["polarity"]["required"] = "negative"
    nonfinal = _failed_row(
        custom_id=custom_id, reviews=[value, final_invalid], payload=payload
    )
    assert review_contracts._recover_prior_invalid_review(
        row=request, response_rows=[nonfinal]
    ) is None

    old_failed = _failed_row(custom_id=custom_id, reviews=[value, value], payload=payload)
    old_failed["invalid_attempts"][0]["validation_errors"].append("extra")
    latest = _failed_row(custom_id=custom_id, reviews=[value, value], payload=payload)
    recovered = review_contracts._recover_prior_invalid_review(
        row=request, response_rows=[old_failed, latest]
    )
    assert recovered is not None
    assert recovered["saved_response_abstain_factual_constraint_recovery"]["source"][
        "response_row_sha256"
    ] == review_contracts._sha256_json(latest)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda row: row["retry_accounting"].update(semantic_failure_count=1),
        lambda row: row["retry_accounting"].update(semantic_failure_quota=3),
        lambda row: row["retry_accounting"].update(quota_exhausted=None),
        lambda row: row["sse_integrity_attempts"].pop(),
        lambda row: row["invalid_attempts"][-1]["sse_integrity"].update(
            termination="normal_eof"
        ),
        lambda row: row["failure_attempts"][-1].update(
            response_sha256="0" * 64
        ),
        lambda row: row.update(invalid_review={}),
        lambda row: row["attempt_usage"][-1].update(total_tokens=31),
        lambda row: row.update(usage_is_lower_bound=True),
    ],
)
def test_abstain_constraint_recovery_rejects_ledger_tamper(tamper) -> None:
    value, payload = _review()
    custom_id = "saved-abstain-ledger"
    failed = _failed_row(custom_id=custom_id, reviews=[value, value], payload=payload)
    tamper(failed)

    assert review_contracts._recover_prior_invalid_review(
        row=_request_row(custom_id, payload), response_rows=[failed]
    ) is None


def _tamper_all_sse_mirrors(row: dict, **updates) -> None:
    for index, integrity in enumerate(row["sse_integrity_attempts"]):
        integrity.update(deepcopy(updates))
        row["invalid_attempts"][index]["sse_integrity"] = deepcopy(integrity)
    row["sse_integrity"] = deepcopy(row["sse_integrity_attempts"][-1])


@pytest.mark.parametrize(
    ("kind", "updates"),
    [
        ("sse", {"done_seen": False}),
        ("sse", {"event_count": 1, "content_event_count": 999}),
        (
            "usage_integrity",
            {
                "status": "passed",
                "usage_event_count": 0,
                "required_token_fields_valid": True,
                "total_matches_prompt_plus_completion": True,
                "at_or_after_finish_reason_stop": True,
            },
        ),
        (
            "usage_integrity",
            {
                "status": "passed",
                "usage_event_count": True,
                "required_token_fields_valid": True,
                "total_matches_prompt_plus_completion": True,
                "at_or_after_finish_reason_stop": True,
            },
        ),
        (
            "usage_integrity",
            {
                "status": "passed",
                "usage_event_count": 1,
                "required_token_fields_valid": 1,
                "total_matches_prompt_plus_completion": 1,
                "at_or_after_finish_reason_stop": 1,
            },
        ),
        ("sse", {"completion_policy": "other"}),
        ("sse", {"finish_reasons": ["length"]}),
        ("top", {"transport": "calibration-test"}),
        ("top", {"calibration_manifest_sha256": "0" * 64}),
        ("top", {"sequence": True}),
        ("token_bool", {}),
    ],
)
def test_v229_formal_source_envelope_rejects_coordinated_tamper(
    kind: str, updates: dict
) -> None:
    value, payload = _review()
    custom_id = "saved-v229-formal-envelope"
    failed = _failed_row(
        custom_id=custom_id, reviews=[value, value], payload=payload
    )
    if kind == "usage_integrity":
        _tamper_all_sse_mirrors(failed, usage_integrity=updates)
    elif kind == "sse":
        _tamper_all_sse_mirrors(failed, **updates)
    elif kind == "token_bool":
        for usage in failed["attempt_usage"]:
            usage["prompt_tokens"] = True
            usage["total_tokens"] = usage["completion_tokens"] + 1
        failed["usage"] = {
            key: sum(usage[key] for usage in failed["attempt_usage"])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    else:
        failed.update(updates)

    assert review_contracts._recover_prior_invalid_review(
        row=_request_row(custom_id, payload), response_rows=[failed]
    ) is None


def test_abstain_constraint_recovery_rejects_single_structural_and_legacy() -> None:
    value, payload = _review()
    custom_id = "saved-abstain-quota-and-provenance"
    request = _request_row(custom_id, payload)

    single = _failed_row(custom_id=custom_id, reviews=[value], payload=payload)
    assert review_contracts._recover_prior_invalid_review(
        row=request, response_rows=[single]
    ) is None

    structural = deepcopy(value)
    structural["required_concepts"]["all_of"][0].pop("any_of")
    structural_row = _failed_row(
        custom_id=custom_id, reviews=[structural, structural], payload=payload
    )
    assert review_contracts._recover_prior_invalid_review(
        row=request, response_rows=[structural_row]
    ) is None

    for runner_version in (
        "2.24.0-saved-redundant-field-format-recovery",
        "2.25.0-saved-exact-payload-branch-span-quote-prefix-recovery",
        "2.26.0-saved-abstain-factual-constraint-recovery",
        "2.27.0-saved-abstain-sentinel-recovery",
        "2.28.0-calibration-only-abstain-sentinel-projection",
        "2.30.0-saved-abstain-sentinel-mismatch-recovery",
        "2.31.0-v229-formal-failure-envelope-replay",
        "2.32.0-v229-formal-ledger-strict-scalars",
        "2.33.0-future-runner",
        "unknown-runner",
    ):
        legacy = _failed_row(
            custom_id=custom_id, reviews=[value, value], payload=payload
        )
        legacy["runner_version"] = runner_version
        assert review_contracts._recover_prior_invalid_review(
            row=request, response_rows=[legacy]
        ) is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["selection"].update(selected_attempt=1),
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["selection"].update(semantic_consensus_claimed=True),
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["selection"].update(cross_attempt_merge=True),
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["stages"][0]["actions"][0].update(json_paths=["/verdict"]),
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["stages"][0]["actions"][0]["after"]["polarity"]["cues"].append(
            "价格是2999元"
        ),
        lambda response: response[
            "saved_response_abstain_factual_constraint_recovery"
        ]["source"].update(response_row_sha256="0" * 64),
        lambda response: response["review"].update(verdict="block"),
        lambda response: response["review"]["required_concepts"]["all_of"][0].update(
            canonical_text="未提供无线能力—设备具备无线网络功能"
        ),
        lambda response: response.update(attempt_usage=[{"total_tokens": 1}]),
    ],
)
def test_abstain_constraint_audit_rejects_tamper(tamper) -> None:
    value, payload = _review()
    custom_id = "saved-abstain-audit"
    failed = _failed_row(custom_id=custom_id, reviews=[value, value], payload=payload)
    recovered = review_contracts._recover_prior_invalid_review(
        row=_request_row(custom_id, payload), response_rows=[failed]
    )
    assert recovered is not None
    tamper(recovered)
    validator = getattr(
        review_contracts, "_validate_abstain_factual_constraint_recovery_audit"
    )
    assert validator(recovered, payload, source_response_rows=[failed])


def _real_review_history() -> tuple[list[dict], dict[str, dict], Path]:
    review_root = CONTRACT_ROOT / "review"
    responses_path = review_root / "responses.jsonl"
    responses = review_contracts._read_jsonl(responses_path)
    requests = {
        str(row["custom_id"]): row
        for row in review_contracts._read_jsonl(review_root / "requests.jsonl")
    }
    return responses, requests, responses_path


def test_real_seq146_to_149_historical_recovery_audits_still_replay() -> None:
    responses, requests, _responses_path = _real_review_history()
    validator = getattr(
        review_contracts, "_validate_abstain_factual_constraint_recovery_audit"
    )

    for sequence in (146, 147, 148, 149):
        successful = [
            row
            for row in responses
            if row.get("sequence") == sequence and row.get("status") == "succeeded"
        ]
        assert len(successful) == 1
        response = successful[0]
        request = requests[str(response["custom_id"])]
        payload = json.loads(request["request"]["messages"][1]["content"])
        assert validator(
            response, payload, source_response_rows=responses
        ) == []


def test_current_runner_does_not_reopen_seq149_historical_source() -> None:
    responses, requests, responses_path = _real_review_history()
    before = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    failed = [
        row
        for row in responses
        if row.get("sequence") == 149 and row.get("status") == "failed"
    ]
    assert len(failed) == 1
    source = failed[-1]
    request = requests[str(source["custom_id"])]

    assert review_contracts._recover_prior_invalid_review(
        row=request, response_rows=responses
    ) is None
    assert hashlib.sha256(responses_path.read_bytes()).hexdigest() == before


def test_real_seq150_dry_recovery_is_latest_attempt3_zero_usage_and_read_only() -> None:
    responses, requests, responses_path = _real_review_history()
    before = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    failed = [
        row
        for row in responses
        if row.get("sequence") == 150 and row.get("status") == "failed"
    ]
    assert len(failed) == 1
    source = failed[-1]
    assert source["runner_version"] == "2.29.0-real-sse-calibration-ledger"
    sentinel_error = (
        "abstain concept[0] variants must each express only an explicit "
        "absence/uncertainty rationale"
    )
    assert all(
        item["validation_errors"] == [sentinel_error]
        and item["normalization_provenance"]["actions"] == []
        for item in source["invalid_attempts"]
    )
    request = requests[str(source["custom_id"])]
    payload = json.loads(request["request"]["messages"][1]["content"])

    recovered = review_contracts._recover_prior_invalid_review(
        row=request, response_rows=responses
    )

    assert recovered is not None
    audit = recovered["saved_response_abstain_factual_constraint_recovery"]
    assert audit["source"]["response_row_sha256"] == review_contracts._sha256_json(
        source
    )
    assert audit["selection"] == {
        "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
        "selected_attempt": 3,
        "semantic_consensus_claimed": False,
        "cross_attempt_merge": False,
    }
    assert recovered["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert review_contracts._validate_abstain_factual_constraint_recovery_audit(
        recovered, payload, source_response_rows=responses
    ) == []
    assert hashlib.sha256(responses_path.read_bytes()).hexdigest() == before
