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

import review_contracts as runner  # noqa: E402
import review_blocker_rereview as blocker_wrapper  # noqa: E402
import review_contract_rereview as contract_wrapper  # noqa: E402
import merge_reviewed_contracts as merge_reviewed  # noqa: E402
from concept_materialization import (  # noqa: E402
    MATERIALIZATION_POLICY_VERSION,
    materialize_concept_groups,
)
from review_release_integrity import (  # noqa: E402
    EXPECTED_REREVIEW_MODEL,
    formal_rereview_failed_result_errors,
    formal_rereview_success_ledger,
    validate_rereview_request_policy,
)


TARGET_ID = "concept-review-c753ea69e724ecaf38ce"
EXPECTED_SOURCE_ROW_SHA256 = (
    "b6cb8b1ba514c0ad3a16d09bc8c9de41f40cbeef93c2d31bd18a01f8b7c9bd12"
)
EXPECTED_EXECUTION_CONTRACT = (
    "60694067e286a704a7644e70c63786c0c062e44be27373b8b79927dfc9ba9414"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_response(
    rows: list[dict],
    *,
    custom_id: str,
    status: str | None = None,
    runner_version: str | None = None,
    transport: str | None = None,
    row_sha256: str | None = None,
) -> dict:
    matches = [
        row
        for row in rows
        if row.get("custom_id") == custom_id
        and (status is None or row.get("status") == status)
        and (runner_version is None or row.get("runner_version") == runner_version)
        and (transport is None or row.get("transport") == transport)
        and (row_sha256 is None or runner._sha256_json(row) == row_sha256)
    ]
    assert matches, f"missing response fixture for {custom_id}"
    return matches[-1]


def _source_response(rows: list[dict]) -> dict:
    return _latest_response(
        rows,
        custom_id=TARGET_ID,
        status="failed",
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        row_sha256=EXPECTED_SOURCE_ROW_SHA256,
    )


def _source_bound_prefix(rows: list[dict]) -> list[dict]:
    for position, row in enumerate(rows):
        if runner._sha256_json(row) == EXPECTED_SOURCE_ROW_SHA256:
            return rows[: position + 1]
    raise AssertionError("saved recovery source row is missing")


def _fixture() -> tuple[list[dict], list[dict], dict, dict, dict]:
    requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    responses = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    request = next(row for row in requests if row["custom_id"] == TARGET_ID)
    payload = json.loads(request["request"]["messages"][1]["content"])
    source = _source_response(responses)
    return requests, responses, request, payload, source


def _materialize(review: dict, payload: dict) -> tuple[dict, dict]:
    contract = {
        "answerability": {"required": review["answerability"]},
        "required_concepts": {
            "all_of": [
                {
                    "concept_id": concept["concept_key"],
                    "canonical_text": concept["canonical_text"],
                    "any_of": concept["any_of"],
                }
                for concept in review["required_concepts"]["all_of"]
            ]
        },
        "critical_constraints": review["critical_constraints"],
    }
    return materialize_concept_groups(contract, payload["evidence"])


def _validate_live_formal_result(result: dict, request: dict) -> list[str]:
    payload = json.loads(request["request"]["messages"][1]["content"])
    review = result.get("review")
    if not isinstance(review, dict):
        return ["live formal review must be an object"]
    errors = runner._validate_review(review)
    errors.extend(runner._validate_against_payload(review, payload))
    errors.extend(runner._validate_inverse_edge_canonicalization_audit(result, payload))
    errors.extend(runner._validate_exact_variant_deduplication_audit(result, payload))
    errors.extend(runner._validate_coverage_status_canonicalization_audit(result, payload))
    for item in result.get("invalid_attempts_before_success") or []:
        errors.extend(
            runner._validate_invalid_attempt_raw_normalization_provenance(
                item, result, payload
            )
        )
    return errors


def _failed_result_validator(requests: list[dict]):
    """按冻结请求顺序验证物理 failed 行，禁止测试绕过正式 schema。"""

    sequence_by_id = {
        request["custom_id"]: sequence
        for sequence, request in enumerate(requests, start=1)
    }

    def validate(result: dict, request: dict) -> list[str]:
        return formal_rereview_failed_result_errors(
            result,
            request,
            expected_sequence=sequence_by_id[request["custom_id"]],
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    return validate


def test_simple_delete_is_rejected_even_when_existing_validators_pass() -> None:
    _, _, _, payload, source = _fixture()
    raw = source["invalid_attempts"][-1]["raw_review"]
    simple_delete = deepcopy(raw)
    del simple_delete["required_concepts"]["all_of"][1]["省墨"]
    del simple_delete["required_concepts"]["all_of"][2]["较淡"]

    assert runner._validate_review(simple_delete) == []
    assert runner._validate_against_payload(simple_delete, payload) == []
    assert runner._review_materialization_errors(simple_delete, payload) == []
    _, materialization_audit = _materialize(simple_delete, payload)
    assert materialization_audit["production_match_collision_count"] == 4
    assert (
        runner._project_saved_same_concept_variant_subset(simple_delete, payload)
        is None
    )


def test_subset_replacement_recovers_only_raw_any_of_semantics() -> None:
    _, _, _, payload, source = _fixture()
    raw = source["invalid_attempts"][-1]["raw_review"]
    projection = runner._project_saved_same_concept_variant_subset(raw, payload)
    assert projection is not None
    projected, actions, raw_errors, materialization_audit = projection

    concepts = projected["required_concepts"]["all_of"]
    assert concepts[1]["canonical_text"] == "省墨"
    assert concepts[2]["canonical_text"] == "较淡"
    assert "省墨" not in set(concepts[1]) - runner._CONCEPT_ALLOWED_FIELDS
    assert "较淡" not in set(concepts[2]) - runner._CONCEPT_ALLOWED_FIELDS
    assert [action["operation"] for action in actions] == [
        "replace_canonical_with_existing_any_of",
        "remove_schema_extra",
        "replace_canonical_with_existing_any_of",
        "remove_schema_extra",
    ]
    assert raw_errors == source["invalid_attempts"][-1]["validation_errors"]
    assert materialization_audit["removed_variant_count"] == 0
    assert materialization_audit["weak_canonical_count"] == 0
    assert materialization_audit["production_match_collision_count"] == 0


@pytest.mark.parametrize("value", [" ", "。", "!", "abc", "1", "省墨", "\n", "\t"])
def test_subset_replacement_accepts_only_frozen_empty_or_ascii_comma_values(
    value: str,
) -> None:
    _, _, _, payload, source = _fixture()
    raw = deepcopy(source["invalid_attempts"][-1]["raw_review"])
    raw["required_concepts"]["all_of"][1]["省墨"] = value
    assert runner._project_saved_same_concept_variant_subset(raw, payload) is None


def test_subset_replacement_rejects_non_first_variant_other_path_and_mixed_errors() -> None:
    _, _, _, payload, source = _fixture()
    raw = source["invalid_attempts"][-1]["raw_review"]

    not_first = deepcopy(raw)
    concept = not_first["required_concepts"]["all_of"][1]
    concept["省墨模式"] = concept.pop("省墨")
    assert runner._project_saved_same_concept_variant_subset(not_first, payload) is None

    other_path = deepcopy(raw)
    other_path["省墨"] = other_path["required_concepts"]["all_of"][1].pop("省墨")
    assert runner._project_saved_same_concept_variant_subset(other_path, payload) is None

    mixed = deepcopy(raw)
    mixed["review_notes"] = []
    assert runner._project_saved_same_concept_variant_subset(mixed, payload) is None


def test_append_reconstruction_is_forbidden() -> None:
    _, _, _, payload, source = _fixture()
    raw = deepcopy(source["invalid_attempts"][-1]["raw_review"])
    for concept, key in zip(
        raw["required_concepts"]["all_of"][1:], ("省墨", "较淡"), strict=True
    ):
        concept["canonical_text"] += key
        del concept[key]
    _, audit = _materialize(raw, payload)
    assert audit["production_match_collision_count"] > 0
    assert runner._project_saved_same_concept_variant_subset(raw, payload) is None


def test_real_saved_recovery_selects_latest_final_terminal_attempt_without_writes() -> None:
    requests, responses, request, _, source = _fixture()
    source_rows = _source_bound_prefix(responses)
    responses_path = CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl"
    before_sha = _sha256_file(responses_path)
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        sse_completion_policy=runner.SSE_COMPLETION_POLICY,
    )

    recovered = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=source_rows,
        execution_contract_sha256=execution_contract,
    )
    assert recovered is not None
    assert recovered["runner_version"] == runner.POST150_SAVED_RECOVERY_RUNNER_VERSION
    assert recovered["transport"] == runner.SAVED_RESPONSE_TRANSPORT
    assert recovered["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    audit = recovered["saved_response_same_concept_variant_subset_recovery"]
    assert audit["source"]["response_row_sha256"] == EXPECTED_SOURCE_ROW_SHA256
    assert audit["source"]["source_attempt"] == source["attempt"] == 3
    assert audit["selection"]["semantic_consensus"] is False
    assert audit["selection"]["cross_attempt_merge"] is False
    assert audit["api_call_performed"] is False
    assert _sha256_file(responses_path) == before_sha

    recoverable = [
        row["custom_id"]
        for row in requests
        if runner._recover_latest_saved_same_concept_variant_subset(
            row=row,
            response_rows=source_rows,
            execution_contract_sha256=execution_contract,
        )
        is not None
    ]
    assert recoverable == [TARGET_ID]


def test_recovery_rejects_not_latest_normalized_single_nonterminal_and_nonquota() -> None:
    requests, responses, request, _, source = _fixture()
    source_rows = _source_bound_prefix(responses)
    execution_contract = EXPECTED_EXECUTION_CONTRACT

    assert (
        runner._recover_latest_saved_same_concept_variant_subset(
            row=request,
            response_rows=source_rows[:-1],
            execution_contract_sha256=execution_contract,
        )
        is None
    )
    for mutate in ("normalized", "single", "nonterminal", "not_quota"):
        rows = deepcopy(source_rows)
        latest = _source_response(rows)
        if mutate == "normalized":
            latest["invalid_attempts"][-1]["normalization_provenance"]["actions"] = [
                "self_filled_any_of:c2"
            ]
        elif mutate == "single":
            latest["retry_accounting"]["semantic_failure_quota"] = 1
        elif mutate == "nonterminal":
            latest["invalid_attempts"][-1]["sse_integrity"]["finish_reason"] = None
        else:
            latest["retry_accounting"]["quota_exhausted"] = None
        assert (
            runner._recover_latest_saved_same_concept_variant_subset(
                row=request,
                response_rows=rows,
                execution_contract_sha256=execution_contract,
            )
            is None
        )
    assert runner._sha256_json(source) == EXPECTED_SOURCE_ROW_SHA256


def test_recovery_never_unions_an_earlier_attempt() -> None:
    _, responses, request, _, _ = _fixture()
    rows = deepcopy(_source_bound_prefix(responses))
    latest = _source_response(rows)
    latest["invalid_attempts"][-1]["raw_review"] = deepcopy(
        latest["invalid_attempts"][0]["raw_review"]
    )
    assert (
        runner._recover_latest_saved_same_concept_variant_subset(
            row=request,
            response_rows=rows,
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        )
        is None
    )


def test_saved_receipt_replay_rejects_source_and_audit_tamper() -> None:
    _, responses, request, payload, _ = _fixture()
    source_rows = _source_bound_prefix(responses)
    recovered = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert recovered is not None
    assert runner._validate_saved_same_concept_variant_subset_recovery(
        recovered,
        request=request,
        semantic_payload=payload,
        source_response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    ) == []

    assert runner._validate_saved_same_concept_variant_subset_recovery(
        recovered,
        request=request,
        semantic_payload=payload,
        source_response_rows=source_rows[:-1],
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert runner._validate_saved_same_concept_variant_subset_recovery(
        recovered,
        request=request,
        semantic_payload=payload,
        source_response_rows=[*source_rows, deepcopy(_source_response(source_rows))],
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )

    tamper_paths = (
        ("saved_response_same_concept_variant_subset_recovery", "policy"),
        ("saved_response_same_concept_variant_subset_recovery", "actions"),
        ("saved_response_same_concept_variant_subset_recovery", "raw_review_sha256"),
        ("saved_response_same_concept_variant_subset_recovery", "selection"),
    )
    for audit_key, field in tamper_paths:
        tampered = deepcopy(recovered)
        audit = tampered[audit_key]
        audit[field] = "tampered" if field != "actions" else []
        assert runner._validate_saved_same_concept_variant_subset_recovery(
            tampered,
            request=request,
            semantic_payload=payload,
            source_response_rows=source_rows,
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        )


def test_formal_ledger_preserves_live_rows_and_accepts_exact_saved_profile() -> None:
    requests, responses, request, payload, _ = _fixture()
    request_by_id = {row["custom_id"]: row for row in requests}
    response_hashes = [runner._sha256_json(row) for row in responses]
    live = formal_rereview_success_ledger(
        responses,
        requests_by_id=request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_sha256=request["prompt_sha256"],
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        validate_live_result=_validate_live_formal_result,
        validate_saved_result=lambda *_args: [],
        validate_failed_result=_failed_result_validator(requests),
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        validate_materialization_result=lambda result, current_request: (
            runner._review_materialization_errors(
                result["review"],
                json.loads(current_request["request"]["messages"][1]["content"]),
            )
        ),
        label="seven-live-preservation",
    )
    assert [runner._sha256_json(row) for row in responses] == response_hashes
    assert list(live) == [
        row["custom_id"] for row in requests if row["custom_id"] in live
    ]

    saved = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=_source_bound_prefix(responses),
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert saved is not None
    source_rows = _source_bound_prefix(responses)
    ledger = formal_rereview_success_ledger(
        [*source_rows, saved],
        requests_by_id=request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_sha256=request["prompt_sha256"],
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        validate_live_result=_validate_live_formal_result,
        validate_saved_result=lambda result, current_request, all_rows: (
            runner._validate_saved_same_concept_variant_subset_recovery(
                result,
                request=current_request,
                semantic_payload=payload,
                source_response_rows=all_rows,
                execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
            )
        ),
        validate_failed_result=_failed_result_validator(requests),
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        validate_materialization_result=lambda result, current_request: (
            runner._review_materialization_errors(
                result["review"],
                json.loads(current_request["request"]["messages"][1]["content"]),
            )
        ),
        label="live-or-saved",
    )
    assert TARGET_ID in ledger
    assert "concept-review-8f4066c9d919364718f5" not in ledger


def _real_live_formal_ledger(rows: list[dict], requests: list[dict]) -> dict[str, dict]:
    request_by_id = {row["custom_id"]: row for row in requests}
    return formal_rereview_success_ledger(
        rows,
        requests_by_id=request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_sha256=requests[0]["prompt_sha256"],
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        validate_live_result=_validate_live_formal_result,
        validate_saved_result=lambda *_args: [],
        validate_failed_result=_failed_result_validator(requests),
        label="formal-latest-event",
    )


def test_formal_ledger_uses_physical_latest_event_for_each_current_id() -> None:
    requests, responses, _, _, _ = _fixture()
    request = requests[7]
    good = deepcopy(
        _latest_response(
            responses,
            custom_id=request["custom_id"],
            status="succeeded",
            runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
            transport=runner.SAVED_RESPONSE_TRANSPORT,
        )
    )
    failed = deepcopy(
        _latest_response(
            responses,
            custom_id=request["custom_id"],
            status="failed",
            runner_version=runner.RUNNER_VERSION,
            transport=runner.TRANSPORT,
        )
    )

    assert good["custom_id"] not in _real_live_formal_ledger(
        [good, failed], requests
    )
    assert good["custom_id"] in _real_live_formal_ledger(
        [failed, good], requests
    )


def test_formal_ledger_rejects_malformed_physical_failed_event() -> None:
    requests, responses, _, _, _ = _fixture()
    good = deepcopy(
        _latest_response(
            responses,
            custom_id=requests[0]["custom_id"],
            status="succeeded",
            runner_version=runner.RUNNER_VERSION,
            transport=runner.TRANSPORT,
        )
    )
    malformed = {
        "custom_id": good["custom_id"],
        "status": "failed",
        "completed_at": "2026-08-12T00:00:00+00:00",
    }

    with pytest.raises(RuntimeError, match="live failed response replay failed"):
        _real_live_formal_ledger([good, malformed], requests)


def test_formal_ledger_rejects_foreign_success_even_when_current_rows_are_valid() -> None:
    requests, responses, _, _, _ = _fixture()
    current = _latest_response(
        responses,
        custom_id=requests[0]["custom_id"],
        status="succeeded",
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
    )
    foreign = deepcopy(current)
    foreign["custom_id"] = "foreign-success-must-not-be-ignored"
    with pytest.raises(RuntimeError, match="outside current pack"):
        _real_live_formal_ledger([current, foreign], requests)


@pytest.mark.parametrize(
    "tamper",
    (
        "done_seen_false",
        "impossible_event_counts",
        "non_hex_content_hash",
        "missing_completed_at",
        "missing_completion_policy",
        "wrong_sequence",
        "semantically_invalid_review",
    ),
)
def test_formal_live_receipt_rejects_terminal_and_schema_tamper(
    tamper: str,
) -> None:
    requests, responses, _, _, _ = _fixture()
    row = deepcopy(
        _latest_response(
            responses,
            custom_id=requests[0]["custom_id"],
            status="succeeded",
            runner_version=runner.RUNNER_VERSION,
            transport=runner.TRANSPORT,
        )
    )
    integrity = row["sse_integrity"]
    attempt_integrity = row["sse_integrity_attempts"][-1]
    if tamper == "done_seen_false":
        integrity["done_seen"] = False
        attempt_integrity["done_seen"] = False
    elif tamper == "impossible_event_counts":
        integrity["event_count"] = 1
        integrity["content_event_count"] = 999
        attempt_integrity["event_count"] = 1
        attempt_integrity["content_event_count"] = 999
    elif tamper == "non_hex_content_hash":
        integrity["content_sha256"] = "z" * 64
        attempt_integrity["content_sha256"] = "z" * 64
    elif tamper == "missing_completed_at":
        del row["completed_at"]
    elif tamper == "missing_completion_policy":
        del integrity["completion_policy"]
        del attempt_integrity["completion_policy"]
    elif tamper == "wrong_sequence":
        row["sequence"] = 99
    else:
        del row["review"]["critical_constraints"]
        row["response_sha256"] = runner._sha256_json(row["review"])

    with pytest.raises(RuntimeError, match="live response replay failed"):
        _real_live_formal_ledger([row], requests)


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_invalid_attempt",
        "failure_response_hash",
        "retry_semantic_count",
        "missing_attempt_usage",
        "invalid_sse_mirror",
        "schema_external_field",
    ),
)
def test_formal_live_receipt_rejects_retry_and_invalid_attempt_tamper(
    tamper: str,
) -> None:
    requests, responses, _, _, _ = _fixture()
    sequence_five_id = next(
        row["custom_id"]
        for row in responses
        if row.get("sequence") == 5
        and row.get("status") == "succeeded"
        and row.get("runner_version") == runner.RUNNER_VERSION
    )
    row = deepcopy(
        _latest_response(
            responses,
            custom_id=sequence_five_id,
            status="succeeded",
            runner_version=runner.RUNNER_VERSION,
            transport=runner.TRANSPORT,
        )
    )
    assert row["attempt"] == 2
    if tamper == "missing_invalid_attempt":
        del row["invalid_attempts_before_success"]
    elif tamper == "failure_response_hash":
        row["failures_before_success"][0]["response_sha256"] = "a" * 64
    elif tamper == "retry_semantic_count":
        row["retry_accounting"]["semantic_failure_count"] = 0
    elif tamper == "missing_attempt_usage":
        row["attempt_usage"] = row["attempt_usage"][1:]
    elif tamper == "invalid_sse_mirror":
        row["invalid_attempts_before_success"][0]["sse_integrity"][
            "content_chars"
        ] += 1
    else:
        row["unfrozen_extra"] = True

    with pytest.raises(RuntimeError, match="live response replay failed"):
        _real_live_formal_ledger([row], requests)


def _synthetic_valid_transport_retry_success(responses: list[dict]) -> dict:
    requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    row = deepcopy(
        _latest_response(
            responses,
            custom_id=requests[0]["custom_id"],
            status="succeeded",
            runner_version=runner.RUNNER_VERSION,
            transport=runner.TRANSPORT,
        )
    )
    transport_failure = next(
        row
        for row in reversed(responses)
        if row.get("custom_id") == TARGET_ID
        and row.get("status") == "failed"
        and (row.get("retry_accounting") or {}).get("quota_exhausted") == "transport"
    )
    row["attempt"] = 2
    row["usage_is_lower_bound"] = True
    row["attempt_usage"][0]["attempt"] = 2
    row["sse_integrity"]["attempt"] = 2
    row["sse_integrity_attempts"] = [
        deepcopy(transport_failure["sse_integrity_attempts"][0]),
        row["sse_integrity"],
    ]
    row["retry_accounting"]["total_attempts"] = 2
    row["retry_accounting"]["transport_failure_count"] = 1
    row["failures_before_success"] = [
        deepcopy(transport_failure["failure_attempts"][0])
    ]
    return row


def test_formal_live_receipt_accepts_exact_transport_retry_lineage() -> None:
    requests, responses, _, _, _ = _fixture()
    row = _synthetic_valid_transport_retry_success(responses)
    ledger = _real_live_formal_ledger([row], requests)
    assert list(ledger) == [row["custom_id"]]


def test_formal_live_receipt_rejects_transport_provider_receipt_tamper() -> None:
    requests, responses, _, _, _ = _fixture()
    row = _synthetic_valid_transport_retry_success(responses)
    row["sse_integrity_attempts"][0]["provider_error_receipt"][
        "message_sha256"
    ] = "not-a-digest"
    with pytest.raises(RuntimeError, match="live response replay failed"):
        _real_live_formal_ledger([row], requests)


def test_merge_formal_ledger_replays_same_saved_profile() -> None:
    requests, responses, request, _, _ = _fixture()
    request_by_id = {row["custom_id"]: row for row in requests}
    saved = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=_source_bound_prefix(responses),
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert saved is not None
    source_rows = _source_bound_prefix(responses)
    baseline = merge_reviewed._formal_post150_success_ledger(
        source_rows,
        request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_hash=request["prompt_sha256"],
        label="merge-parity-baseline",
    )
    ledger = merge_reviewed._formal_post150_success_ledger(
        [*source_rows, saved],
        request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_hash=request["prompt_sha256"],
        label="merge-parity",
    )
    assert set(ledger) == {*baseline, TARGET_ID}
    assert TARGET_ID in ledger


def test_blocker_wrapper_saved_only_path_does_not_read_key_or_call_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, responses, request, _, _ = _fixture()
    source_rows = _source_bound_prefix(responses)
    monkeypatch.setattr(blocker_wrapper, "_validate_requests", lambda: (requests, 15))
    monkeypatch.setattr(blocker_wrapper, "_read_json", lambda _path: {
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT
    })
    monkeypatch.setattr(blocker_wrapper, "validate_rereview_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(blocker_wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(blocker_wrapper.runner, "_acquire_execution_lock", lambda **_kwargs: None)
    calibration_requests = [
        {"custom_id": f"calibration-test-{index}"} for index in range(4)
    ]
    monkeypatch.setattr(
        blocker_wrapper.runner,
        "_calibration_requests",
        lambda _row: deepcopy(calibration_requests),
    )
    monkeypatch.setattr(
        blocker_wrapper,
        "rereview_calibration_ledger",
        lambda _rows, *, requests, **_kwargs: {
            row["custom_id"]: {"status": "succeeded"} for row in requests
        },
    )
    monkeypatch.setattr(
        blocker_wrapper.runner,
        "_read_jsonl",
        lambda path: source_rows
        if path == blocker_wrapper.RESPONSES_PATH
        else []
        if path == blocker_wrapper.runner.CALIBRATION_RESPONSES_PATH
        else requests,
    )
    monkeypatch.setattr(
        blocker_wrapper.runner,
        "_execute_one",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )
    appended: list[dict] = []

    class StopAfterSavedAppend(RuntimeError):
        pass

    def append_once(_path: Path, row: dict) -> None:
        appended.append(row)
        raise StopAfterSavedAppend

    monkeypatch.setattr(blocker_wrapper.runner, "_append_jsonl", append_once)
    with pytest.raises(StopAfterSavedAppend):
        blocker_wrapper.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
                "--api-key-env",
                "POST150_SAVED_ONLY_TEST_KEY_MUST_NOT_EXIST",
            ]
        )
    assert len(appended) == 1
    assert appended[0]["custom_id"] == request["custom_id"] == TARGET_ID
    assert appended[0]["transport"] == runner.SAVED_RESPONSE_TRANSPORT
    assert appended[0]["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("wrapper", (blocker_wrapper, contract_wrapper))
def test_post150_wrapper_appends_saved_before_incomplete_calibration_or_key_read(
    monkeypatch: pytest.MonkeyPatch, wrapper
) -> None:
    requests, responses, request, _, _ = _fixture()
    source_rows = _source_bound_prefix(responses)
    monkeypatch.setattr(wrapper, "_validate_requests", lambda: (requests, len(requests)))
    monkeypatch.setattr(
        wrapper,
        "_read_json",
        lambda _path: {"execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT},
    )
    monkeypatch.setattr(wrapper, "validate_rereview_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(wrapper.runner, "_acquire_execution_lock", lambda **_kwargs: None)
    monkeypatch.setattr(
        wrapper.runner,
        "_calibration_requests",
        lambda _row: [
            {"custom_id": f"calibration-incomplete-{index}"} for index in range(4)
        ],
    )
    monkeypatch.setattr(wrapper, "rereview_calibration_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        wrapper.runner,
        "_read_jsonl",
        lambda path: source_rows
        if path == wrapper.RESPONSES_PATH
        else []
        if path == wrapper.runner.CALIBRATION_RESPONSES_PATH
        else requests,
    )
    monkeypatch.delenv("POST150_ORDER_TEST_KEY", raising=False)
    monkeypatch.setattr(
        wrapper.runner,
        "_execute_one",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )
    appended: list[dict] = []

    class StopAfterSavedAppend(RuntimeError):
        pass

    def append_once(_path: Path, row: dict) -> None:
        appended.append(row)
        raise StopAfterSavedAppend

    monkeypatch.setattr(wrapper.runner, "_append_jsonl", append_once)
    with pytest.raises(StopAfterSavedAppend):
        wrapper.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
                "--api-key-env",
                "POST150_ORDER_TEST_KEY",
            ]
        )
    assert len(appended) == 1
    assert appended[0]["custom_id"] == request["custom_id"] == TARGET_ID
    assert appended[0]["transport"] == runner.SAVED_RESPONSE_TRANSPORT


@pytest.mark.parametrize("wrapper", (blocker_wrapper, contract_wrapper))
def test_post150_wrapper_saved_append_survives_until_lazy_key_gate(
    monkeypatch: pytest.MonkeyPatch, wrapper
) -> None:
    requests, responses, request, _, _ = _fixture()
    source_rows = _source_bound_prefix(responses)
    monkeypatch.setattr(wrapper, "_validate_requests", lambda: (requests, len(requests)))
    monkeypatch.setattr(
        wrapper,
        "_read_json",
        lambda _path: {"execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT},
    )
    monkeypatch.setattr(wrapper, "validate_rereview_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(wrapper.runner, "_acquire_execution_lock", lambda **_kwargs: None)
    monkeypatch.setattr(
        wrapper.runner,
        "_calibration_requests",
        lambda _row: [
            {"custom_id": f"calibration-incomplete-{index}"} for index in range(4)
        ],
    )
    monkeypatch.setattr(wrapper, "rereview_calibration_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        wrapper.runner,
        "_read_jsonl",
        lambda path: source_rows
        if path == wrapper.RESPONSES_PATH
        else []
        if path == wrapper.runner.CALIBRATION_RESPONSES_PATH
        else requests,
    )
    monkeypatch.delenv("POST150_LAZY_KEY_TEST", raising=False)
    monkeypatch.setattr(
        wrapper.runner,
        "_execute_one",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )
    appended: list[dict] = []
    monkeypatch.setattr(
        wrapper.runner,
        "_append_jsonl",
        lambda _path, row: appended.append(deepcopy(row)),
    )

    with pytest.raises(SystemExit, match="POST150_LAZY_KEY_TEST is empty"):
        wrapper.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
                "--api-key-env",
                "POST150_LAZY_KEY_TEST",
            ]
        )
    assert [row["custom_id"] for row in appended] == [request["custom_id"]]
    assert appended[0]["transport"] == runner.SAVED_RESPONSE_TRANSPORT
