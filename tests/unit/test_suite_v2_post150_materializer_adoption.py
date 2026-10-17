from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
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
from concept_materialization import MATERIALIZATION_POLICY_VERSION  # noqa: E402
from review_release_integrity import (  # noqa: E402
    EXPECTED_REREVIEW_BASE_URL,
    FORMAL_STREAM_DIAGNOSTIC_POLICY,
    MATERIALIZATION_ADOPTION_POLICY,
    MATERIALIZATION_RETRY_QUOTA,
    build_materialization_invalidation,
    build_materialization_retry_outcome,
    build_materialization_retry_started,
    execute_materialization_retry_episodes,
    formal_rereview_completion_errors,
    formal_rereview_failed_result_errors,
    formal_rereview_state_reducer,
)
import review_blocker_rereview as blocker_wrapper  # noqa: E402
import review_contract_rereview as contract_wrapper  # noqa: E402
from validate_suite_v2 import (  # noqa: E402
    _materialization_adoption_reporting_errors,
)


TARGET_ID = "concept-review-8f4066c9d919364718f5"
EXECUTION_CONTRACT = "60694067e286a704a7644e70c63786c0c062e44be27373b8b79927dfc9ba9414"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _latest_response(
    rows: list[dict],
    *,
    custom_id: str,
    status: str,
    row_sha256: str | None = None,
    quota_exhausted: str | None = None,
) -> dict:
    matches = [
        row
        for row in rows
        if row.get("custom_id") == custom_id
        and row.get("status") == status
        and row.get("runner_version") == runner.RUNNER_VERSION
        and row.get("transport") == runner.TRANSPORT
        and (row_sha256 is None or runner._sha256_json(row) == row_sha256)
        and (
            quota_exhausted is None
            or (row.get("retry_accounting") or {}).get("quota_exhausted")
            == quota_exhausted
        )
    ]
    assert matches, f"missing response fixture for {custom_id}"
    return matches[-1]


def _materialization_source(rows: list[dict]) -> dict:
    invalidation = next(
        row
        for row in reversed(rows)
        if row.get("custom_id") == TARGET_ID
        and row.get("status") == "materialization_invalidated"
    )
    return _latest_response(
        rows,
        custom_id=TARGET_ID,
        status="succeeded",
        row_sha256=str(invalidation["source_success_sha256"]),
    )


def _retry_eligible_baseline(rows: list[dict]) -> list[dict]:
    source = _materialization_source(rows)
    source_sha256 = runner._sha256_json(source)
    return [
        row
        for row in rows
        if row.get("custom_id") != TARGET_ID
        or runner._sha256_json(row) == source_sha256
    ]


def _timestamp_after(row: dict, field: str) -> str:
    instant = datetime.fromisoformat(str(row[field]).replace("Z", "+00:00"))
    return (instant + timedelta(microseconds=1)).isoformat()


def _fixture() -> tuple[list[dict], dict, dict]:
    requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    responses = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    request = next(row for row in requests if row["custom_id"] == TARGET_ID)
    source = _materialization_source(responses)
    return requests, request, source


def _core(_result: dict, _request: dict) -> list[str]:
    return []


def _saved(_result: dict, _request: dict, _rows: list[dict]) -> list[str]:
    return []


def _reduce(rows: list[dict], requests: list[dict], materialize, validate_failed=None) -> dict:
    return formal_rereview_state_reducer(
        rows,
        requests_by_id={row["custom_id"]: row for row in requests},
        execution_contract_sha256=EXECUTION_CONTRACT,
        expected_prompt_sha256=requests[0]["prompt_sha256"],
        expected_model="qwen3.7-max",
        expected_endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        validate_live_result=_core,
        validate_saved_result=_saved,
        validate_materialization_result=materialize,
        validate_failed_result=validate_failed or (lambda *_args: []),
        label="materializer-adoption-test",
    )


def test_real_sequence_four_is_only_materialization_retry_eligible_live_success() -> None:
    requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    rows = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")

    def materialize(result: dict, request: dict) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        return runner._review_materialization_errors(result["review"], payload)

    baseline_rows = _retry_eligible_baseline(rows)
    state = _reduce(baseline_rows, requests, materialize)
    latest_by_id = {row["custom_id"]: row for row in baseline_rows}
    assert set(state["successful_by_id"]) == {
        custom_id
        for custom_id, row in latest_by_id.items()
        if row.get("status") == "succeeded" and custom_id != TARGET_ID
    }
    assert list(state["retry_eligible_by_id"]) == [TARGET_ID]
    eligible = state["retry_eligible_by_id"][TARGET_ID]
    assert eligible["sequence"] == 4
    assert eligible["materialization_errors"] == [
        "concept materialization failed: concept 'c05' has no production-discriminative variant"
    ]


def test_stream_content_receipt_is_diagnostic_but_review_hash_is_authoritative() -> None:
    requests, _, source = _fixture()
    diagnostic_tamper = deepcopy(source)
    for receipt in (
        diagnostic_tamper["sse_integrity"],
        diagnostic_tamper["sse_integrity_attempts"][-1],
    ):
        receipt["content_sha256"] = "e" * 64
        receipt["content_chars"] += 1
    state = _reduce(
        [diagnostic_tamper],
        requests,
        lambda result, request: runner._review_materialization_errors(
            result["review"],
            json.loads(request["request"]["messages"][1]["content"]),
        ),
    )
    assert list(state["retry_eligible_by_id"]) == [TARGET_ID]
    authority = blocker_wrapper.materialization_state_summary(state)["stream_receipt_authority"]
    assert authority == {
        "policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "content_sha256_authoritative": False,
        "content_chars_authoritative": False,
        "event_counts_authoritative": False,
        "semantic_authority": (
            "response_sha256_recomputed_from_review_plus_exact_core_payload_"
            "canonicalization_materializer_and_reducer_replay"
        ),
    }

    for field in ("review", "response_sha256"):
        semantic_tamper = deepcopy(source)
        if field == "review":
            semantic_tamper["review"]["review_notes"].append("tampered")
        else:
            semantic_tamper["response_sha256"] = "e" * 64
        with pytest.raises(RuntimeError, match="response binding"):
            _reduce(
                [semantic_tamper],
                requests,
                lambda *_args: ["materialization still fails"],
            )


def test_exact_adjacent_episode_adopts_one_fresh_live_success() -> None:
    requests, request, source = _fixture()
    source_errors = ["concept materialization failed: collision"]
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=source_errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    started = build_materialization_retry_started(
        invalidation=invalidated,
        request=request,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")

    state = _reduce(
        [source, invalidated, started, fresh],
        requests,
        lambda result, _request: (
            source_errors if result.get("completed_at") == source.get("completed_at") else []
        ),
    )
    assert state["successful_by_id"][TARGET_ID] == fresh
    assert state["adopted_by_id"][TARGET_ID]["episode_id"] == invalidated["episode_id"]
    assert state["retry_eligible_by_id"] == {}
    assert state["retry_started_by_id"] == {}


def test_second_success_without_complete_episode_is_rejected() -> None:
    requests, _, source = _fixture()
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(source, "completed_at")
    with pytest.raises(RuntimeError, match="duplicate successful response"):
        _reduce([source, fresh], requests, lambda *_args: [])


def test_started_latest_is_fail_stop_and_never_retry_eligible() -> None:
    requests, request, source = _fixture()
    errors = ["concept materialization failed: collision"]
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    started = build_materialization_retry_started(
        invalidation=invalidated,
        request=request,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    state = _reduce([source, invalidated, started], requests, lambda *_args: errors)
    assert list(state["retry_started_by_id"]) == [TARGET_ID]
    assert state["retry_eligible_by_id"] == {}
    assert TARGET_ID not in state["successful_by_id"]


@pytest.mark.parametrize(
    ("event", "field"),
    (
        ("invalidated", "policy"),
        ("invalidated", "fresh_retry_quota"),
        ("invalidated", "source_success_sha256"),
        ("started", "retry_ordinal"),
        ("started", "previous_same_id_sha256"),
        ("started", "side_effect_status"),
    ),
)
def test_episode_receipt_tamper_is_rejected(event: str, field: str) -> None:
    requests, request, source = _fixture()
    errors = ["concept materialization failed: collision"]
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    started = build_materialization_retry_started(
        invalidation=invalidated,
        request=request,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    target = invalidated if event == "invalidated" else started
    target[field] = "tampered"
    with pytest.raises(RuntimeError, match="materialization"):
        _reduce([source, invalidated, started], requests, lambda *_args: errors)


def test_materializer_pass_success_cannot_be_invalidated() -> None:
    requests, request, source = _fixture()
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=["stale error"],
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    assert invalidated["policy"] == MATERIALIZATION_ADOPTION_POLICY
    with pytest.raises(RuntimeError, match="materializer-pass"):
        _reduce([source, invalidated], requests, lambda *_args: [])


def test_invalidation_must_be_physically_adjacent_to_source() -> None:
    requests, request, source = _fixture()
    errors = ["concept materialization failed: collision"]
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    intervening = {"custom_id": TARGET_ID, "status": "failed"}
    with pytest.raises(RuntimeError, match="adjacent"):
        _reduce([source, intervening, invalidated], requests, lambda *_args: errors)


def _episode() -> tuple[list[dict], dict, dict, dict, dict, list[str]]:
    requests, request, source = _fixture()
    errors = ["concept materialization failed: collision"]
    invalidated = build_materialization_invalidation(
        source_success=source,
        request=request,
        materialization_errors=errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
    )
    started = build_materialization_retry_started(
        invalidation=invalidated,
        request=request,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    return requests, request, source, invalidated, started, errors


def test_materialization_rejected_wraps_full_bound_success_and_is_terminal() -> None:
    requests, request, source, invalidated, started, errors = _episode()
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")
    rejected = build_materialization_retry_outcome(
        started=started,
        request=request,
        bound_result=fresh,
        materialization_errors=errors,
    )
    state = _reduce([source, invalidated, started, rejected], requests, lambda *_args: errors)
    assert state["successful_by_id"] == {}
    assert state["terminal_by_id"][TARGET_ID] == rejected
    assert rejected["bound_result_sha256"] == runner._sha256_json(fresh)
    assert rejected["usage"] == fresh["usage"]


def test_materialization_call_failed_wraps_full_bound_failure_and_is_terminal() -> None:
    requests, request, source, invalidated, started, errors = _episode()
    all_requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    all_responses = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    failed_request = next(
        row for row in all_requests if row["custom_id"] == "concept-review-c753ea69e724ecaf38ce"
    )
    failed = deepcopy(
        _latest_response(
            all_responses,
            custom_id=failed_request["custom_id"],
            status="failed",
            quota_exhausted="semantic",
        )
    )
    failed["custom_id"] = TARGET_ID
    failed["prompt_sha256"] = request["prompt_sha256"]
    failed["sequence"] = 4
    outcome = build_materialization_retry_outcome(
        started=started, request=request, bound_result=failed
    )

    def validate_failed(result: dict, _request: dict) -> list[str]:
        compatibility = deepcopy(result)
        compatibility["custom_id"] = failed_request["custom_id"]
        compatibility["sequence"] = 8
        return formal_rereview_failed_result_errors(
            compatibility,
            failed_request,
            expected_sequence=8,
            execution_contract_sha256=EXECUTION_CONTRACT,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    state = _reduce(
        [source, invalidated, started, outcome],
        requests,
        lambda *_args: errors,
        validate_failed,
    )
    assert state["successful_by_id"] == {}
    assert state["terminal_by_id"][TARGET_ID] == outcome
    assert outcome["usage"] == {
        key: failed["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }

    tampered = deepcopy(outcome)
    tampered["bound_result"]["invalid_attempts"][-1]["validation_errors"].append("tampered")
    with pytest.raises(RuntimeError, match="failed-call"):
        _reduce(
            [source, invalidated, started, tampered],
            requests,
            lambda *_args: errors,
            validate_failed,
        )


def test_exact_failed_validator_accepts_transport_only_and_rejects_nested_tamper() -> None:
    all_requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    all_responses = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    request = next(
        row for row in all_requests if row["custom_id"] == "concept-review-c753ea69e724ecaf38ce"
    )
    transport = _latest_response(
        all_responses,
        custom_id=request["custom_id"],
        status="failed",
        quota_exhausted="transport",
    )
    errors = formal_rereview_failed_result_errors(
        transport,
        request,
        expected_sequence=8,
        execution_contract_sha256=EXECUTION_CONTRACT,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        validate_invalid_attempt=(runner._validate_invalid_attempt_raw_normalization_provenance),
    )
    assert errors == []

    tampered = deepcopy(transport)
    tampered["sse_integrity_attempts"][0]["provider_error_receipt"]["message_sha256"] = "0" * 63
    assert formal_rereview_failed_result_errors(
        tampered,
        request,
        expected_sequence=8,
        execution_contract_sha256=EXECUTION_CONTRACT,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        validate_invalid_attempt=(runner._validate_invalid_attempt_raw_normalization_provenance),
    )


def test_exact_failed_validator_accepts_interleaved_hard_total_profile() -> None:
    all_requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")
    all_responses = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    request = next(
        row for row in all_requests if row["custom_id"] == "concept-review-c753ea69e724ecaf38ce"
    )
    semantic = _latest_response(
        all_responses,
        custom_id=request["custom_id"],
        status="failed",
        quota_exhausted="semantic",
    )
    mixed = deepcopy(semantic)
    mixed["attempt"] = 6
    mixed["retry_accounting"] = {
        **mixed["retry_accounting"],
        "total_attempts": 6,
        "transport_failure_count": 3,
        "semantic_failure_count": 3,
        "quota_exhausted": "hard_total",
    }
    mixed["usage_is_lower_bound"] = True
    mixed["failure_attempts"] = []
    mixed["sse_integrity_attempts"] = []
    mixed["invalid_attempts"] = []
    mixed["attempt_usage"] = []
    semantic_by_attempt = {item["attempt"]: item for item in semantic["invalid_attempts"]}
    usage_by_attempt = {item["attempt"]: item for item in semantic["attempt_usage"]}
    profile = ["transport", "semantic", "transport", "semantic", "semantic", "transport"]
    semantic_ordinal = 0
    for attempt, category in enumerate(profile, start=1):
        if category == "transport":
            failure = {
                "attempt": attempt,
                "termination": "http_error",
                "error_type": "HTTPError",
                "failure_category": "transport",
                "response_sha256": None,
                "response_hash_status": ("not_available_incomplete_transport_or_parser_failure"),
                "http_status": 503,
                "http_reason": "Service Unavailable",
            }
        else:
            semantic_ordinal += 1
            source_attempt = semantic_ordinal
            failure = deepcopy(semantic["failure_attempts"][source_attempt - 1])
            integrity = deepcopy(semantic["sse_integrity_attempts"][source_attempt - 1])
            invalid = deepcopy(semantic_by_attempt[source_attempt])
            usage = deepcopy(usage_by_attempt[source_attempt])
            invalid["attempt"] = attempt
            invalid["sse_integrity"] = deepcopy(integrity)
            invalid["sse_integrity"]["attempt"] = attempt
            usage["attempt"] = attempt
            mixed["invalid_attempts"].append(invalid)
            mixed["attempt_usage"].append(usage)
            integrity["attempt"] = attempt
            mixed["sse_integrity_attempts"].append(integrity)
        failure["attempt"] = attempt
        mixed["failure_attempts"].append(failure)
    mixed["invalid_review"] = mixed["invalid_attempts"][-1]["review"]
    mixed["sse_integrity"] = mixed["sse_integrity_attempts"][-1]
    mixed["failure_termination"] = mixed["failure_attempts"][-1]["termination"]
    mixed["error"] = "HTTPError: status=503 reason=Service Unavailable"
    mixed["usage"] = {
        key: sum(item[key] for item in mixed["attempt_usage"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    assert (
        formal_rereview_failed_result_errors(
            mixed,
            request,
            expected_sequence=8,
            execution_contract_sha256=EXECUTION_CONTRACT,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )
        == []
    )


def test_exact_failed_validator_accepts_http_only_without_sse_or_usage() -> None:
    requests, request, _ = _fixture()
    failed = {
        "custom_id": TARGET_ID,
        "status": "failed",
        "attempt": 3,
        "model": "qwen3.7-max",
        "runner_version": runner.RUNNER_VERSION,
        "transport": runner.TRANSPORT,
        "sse_completion_policy": runner.SSE_COMPLETION_POLICY,
        "prompt_sha256": request["prompt_sha256"],
        "error": "HTTPError: status=503 reason=Service Unavailable",
        "completed_at": "2026-08-12T15:00:00+00:00",
        "usage_is_lower_bound": True,
        "retry_accounting": {
            "policy": runner.RETRY_ACCOUNTING_POLICY,
            "transport_failure_quota": 3,
            "semantic_failure_quota": 3,
            "hard_total_attempt_quota": 6,
            "total_attempts": 3,
            "transport_failure_count": 3,
            "semantic_failure_count": 0,
            "quota_exhausted": "transport",
        },
        "failure_termination": "http_error",
        "failure_attempts": [
            {
                "attempt": attempt,
                "termination": "http_error",
                "error_type": "HTTPError",
                "failure_category": "transport",
                "response_sha256": None,
                "response_hash_status": ("not_available_incomplete_transport_or_parser_failure"),
                "http_status": 503,
                "http_reason": "Service Unavailable",
            }
            for attempt in range(1, 4)
        ],
        "execution_contract_sha256": EXECUTION_CONTRACT,
        "sequence": 4,
    }
    assert (
        formal_rereview_failed_result_errors(
            failed,
            request,
            expected_sequence=4,
            execution_contract_sha256=EXECUTION_CONTRACT,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )
        == []
    )
    assert requests[3] == request


def test_rejected_actual_pass_and_nested_or_top_mirror_tamper_are_rejected() -> None:
    requests, request, source, invalidated, started, errors = _episode()
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")
    rejected = build_materialization_retry_outcome(
        started=started,
        request=request,
        bound_result=fresh,
        materialization_errors=errors,
    )

    def source_fails_fresh_passes(result: dict, _request: dict) -> list[str]:
        return errors if result.get("completed_at") == source.get("completed_at") else []

    with pytest.raises(RuntimeError, match="actually passes"):
        _reduce(
            [source, invalidated, started, rejected],
            requests,
            source_fails_fresh_passes,
        )

    for target in ("top", "nested"):
        tampered = deepcopy(rejected)
        if target == "top":
            tampered["usage"]["total_tokens"] += 1
        else:
            tampered["bound_result"]["usage"]["total_tokens"] += 1
        with pytest.raises(RuntimeError, match="materialization"):
            _reduce(
                [source, invalidated, started, tampered],
                requests,
                lambda *_args: errors,
            )


def test_shared_driver_appends_fences_then_only_executes_target() -> None:
    requests, request, source = _fixture()
    errors = ["concept materialization failed: collision"]
    appended: list[dict] = []
    executed: list[str] = []

    def materialize(result: dict, _request: dict) -> list[str]:
        return errors if result.get("completed_at") == source["completed_at"] else []

    def load(rows: list[dict]) -> dict:
        return _reduce(rows, requests, materialize)

    def execute(row: dict) -> dict:
        executed.append(row["custom_id"])
        fresh = deepcopy(source)
        retry_start = next(
            event
            for event in reversed(appended)
            if event.get("status") == "materialization_retry_started"
        )
        fresh["completed_at"] = _timestamp_after(retry_start, "started_at")
        del fresh["execution_contract_sha256"]
        del fresh["sequence"]
        return fresh

    rows, state, code = execute_materialization_retry_episodes(
        rows=[source],
        requests=requests,
        state_loader=load,
        append_event=lambda event: appended.append(deepcopy(event)),
        execute_one=execute,
        validate_materialization_result=materialize,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    assert code == 0
    assert executed == [TARGET_ID]
    assert [row["status"] for row in appended] == [
        "materialization_invalidated",
        "materialization_retry_started",
        "succeeded",
    ]
    assert state["successful_by_id"][TARGET_ID] == rows[-1]


def test_shared_driver_restart_after_started_calls_nothing() -> None:
    requests, _, source, invalidated, started, errors = _episode()
    calls = 0

    def execute(_row: dict) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("started restart must never call provider")

    rows, state, code = execute_materialization_retry_episodes(
        rows=[source, invalidated, started],
        requests=requests,
        state_loader=lambda current: _reduce(current, requests, lambda *_args: errors),
        append_event=lambda _event: raise_for_append(),
        execute_one=execute,
        validate_materialization_result=lambda *_args: errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    assert code == 2
    assert calls == 0
    assert rows[-1] == started
    assert TARGET_ID in state["retry_started_by_id"]


def test_shared_driver_scoped_target_ignores_non_target_terminal_gate() -> None:
    requests, request, source, invalidated, started, errors = _episode()
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")
    rejected = build_materialization_retry_outcome(
        started=started,
        request=request,
        bound_result=fresh,
        materialization_errors=errors,
    )
    all_rows = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    target_request = requests[0]
    target_success = _latest_response(
        all_rows,
        custom_id=target_request["custom_id"],
        status="succeeded",
    )
    rows = [target_success, source, invalidated, started, rejected]
    calls = 0

    def execute(_row: dict) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("已成功的scope目标不得调用provider")

    output, state, code = execute_materialization_retry_episodes(
        rows=rows,
        requests=requests,
        state_loader=lambda current: _reduce(
            current,
            requests,
            lambda result, _request: (
                errors if result.get("custom_id") == TARGET_ID else []
            ),
        ),
        append_event=lambda _event: raise_for_append(),
        execute_one=execute,
        validate_materialization_result=lambda *_args: errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
        target_custom_ids=[target_request["custom_id"]],
    )

    assert code == 0
    assert calls == 0
    assert output == rows
    assert set(state["terminal_by_id"]) == {TARGET_ID}
    assert len(state["episode_by_id"]) == 1


def test_shared_driver_scoped_target_does_not_reset_global_episode_quota() -> None:
    requests, request, source, invalidated, started, errors = _episode()
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")
    rejected = build_materialization_retry_outcome(
        started=started,
        request=request,
        bound_result=fresh,
        materialization_errors=errors,
    )
    all_rows = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    target_request = requests[0]
    target_success = _latest_response(
        all_rows,
        custom_id=target_request["custom_id"],
        status="succeeded",
    )
    rows = [target_success, source, invalidated, started, rejected]
    calls = 0

    def execute(_row: dict) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("全局fresh-retry配额耗尽后不得调用provider")

    output, state, code = execute_materialization_retry_episodes(
        rows=rows,
        requests=requests,
        state_loader=lambda current: _reduce(current, requests, lambda *_args: errors),
        append_event=lambda _event: raise_for_append(),
        execute_one=execute,
        validate_materialization_result=lambda *_args: errors,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
        target_custom_ids=[target_request["custom_id"]],
    )

    assert code == 2
    assert calls == 0
    assert output == rows
    assert set(state["retry_eligible_by_id"]) == {target_request["custom_id"]}
    assert len(state["episode_by_id"]) == MATERIALIZATION_RETRY_QUOTA


def test_materialization_episode_cannot_be_reset_by_later_bare_failure() -> None:
    requests, _, source, invalidated, started, errors = _episode()
    failed = {"custom_id": TARGET_ID, "status": "failed"}
    with pytest.raises(RuntimeError, match="invalidation.*retry start"):
        _reduce(
            [source, invalidated, failed],
            requests,
            lambda *_args: errors,
        )
    fresh = deepcopy(source)
    fresh["completed_at"] = _timestamp_after(started, "started_at")
    rejected = build_materialization_retry_outcome(
        started=started,
        request=next(row for row in requests if row["custom_id"] == TARGET_ID),
        bound_result=fresh,
        materialization_errors=errors,
    )
    with pytest.raises(RuntimeError, match="terminal.*later event"):
        _reduce(
            [source, invalidated, started, rejected, failed],
            requests,
            lambda *_args: errors,
        )


def test_global_fresh_retry_quota_calls_provider_once_for_two_eligible_ids() -> None:
    requests = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "requests.jsonl")[:2]
    sources = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")[:2]
    source_hashes = {runner._sha256_json(row) for row in sources}
    calls: list[str] = []

    def materialize(result: dict, _request: dict) -> list[str]:
        return (
            ["concept materialization failed: test-only collision"]
            if runner._sha256_json(result) in source_hashes
            else []
        )

    def load(rows: list[dict]) -> dict:
        return _reduce(rows, requests, materialize)

    def execute(row: dict) -> dict:
        calls.append(str(row["custom_id"]))
        source = next(item for item in sources if item["custom_id"] == row["custom_id"])
        fresh = deepcopy(source)
        fresh.pop("execution_contract_sha256")
        fresh.pop("sequence")
        fresh["completed_at"] = datetime.now(UTC).isoformat()
        return fresh

    rows, state, code = execute_materialization_retry_episodes(
        rows=deepcopy(sources),
        requests=requests,
        state_loader=load,
        append_event=lambda _event: None,
        execute_one=execute,
        validate_materialization_result=materialize,
        execution_contract_sha256=EXECUTION_CONTRACT,
        materializer_policy_version=MATERIALIZATION_POLICY_VERSION,
        endpoint=EXPECTED_REREVIEW_BASE_URL + "/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    assert code == 2
    assert calls == [requests[0]["custom_id"]]
    assert set(state["adopted_by_id"]) == {requests[0]["custom_id"]}
    assert set(state["retry_eligible_by_id"]) == {requests[1]["custom_id"]}
    assert len(state["episode_by_id"]) == 1
    assert len(rows) == len(sources) + 3
    summary = blocker_wrapper.materialization_state_summary(state)
    assert summary["global_fresh_retry_quota"] == 1
    assert summary["global_fresh_retry_consumed_count"] == 1
    assert formal_rereview_completion_errors(
        state,
        expected_custom_ids=[str(row["custom_id"]) for row in requests],
    )


def raise_for_append() -> None:
    raise AssertionError("fail-stop restart must not append")


def test_wrapper_missing_key_keeps_temp_responses_unchanged_and_seq4_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, _, _ = _fixture()
    real_rows = _read_jsonl(CONTRACT_ROOT / "blocker-rereview" / "responses.jsonl")
    baseline_rows = _retry_eligible_baseline(real_rows)
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    before = responses_path.read_bytes()
    calls = 0

    def forbid_execute(**_kwargs) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("missing key must stop before provider execution")

    monkeypatch.setattr(blocker_wrapper, "RESPONSES_PATH", responses_path)
    monkeypatch.setattr(blocker_wrapper, "SUMMARY_PATH", tmp_path / "summary.json")
    monkeypatch.setattr(blocker_wrapper, "_validate_requests", lambda: (requests, len(requests)))
    monkeypatch.setattr(blocker_wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(runner, "_acquire_execution_lock", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_one", forbid_execute)
    monkeypatch.delenv("POST150_TEST_MISSING_KEY", raising=False)

    with pytest.raises(SystemExit, match="POST150_TEST_MISSING_KEY is empty"):
        blocker_wrapper.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
                "--api-key-env",
                "POST150_TEST_MISSING_KEY",
            ]
        )
    assert calls == 0
    assert responses_path.read_bytes() == before
    state = blocker_wrapper._formal_state(
        _read_jsonl(responses_path),
        requests,
        execution_contract_sha256=EXECUTION_CONTRACT,
        label="missing-key-postcondition",
    )
    assert list(state["retry_eligible_by_id"]) == [TARGET_ID]
    assert TARGET_ID not in state["successful_by_id"]


def test_qa_and_summary_bind_complete_replayed_materialization_adoption() -> None:
    receipt = {
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "stream_content_sha256_authoritative": False,
        "episode_id": "post150-materialization-0123456789abcdef01234567",
        "source_success_sha256": "a" * 64,
        "invalidation_sha256": "b" * 64,
        "retry_started_sha256": "c" * 64,
        "fresh_success_sha256": "d" * 64,
    }
    qa = [
        {
            "case_id": "P3500-HARD-V2-004",
            "parent_case_id": "P3500-HARD-004",
            "post150_materialization_adoption": deepcopy(receipt),
        }
    ]
    summary = {
        "post150_materialization_adoption_policy": MATERIALIZATION_ADOPTION_POLICY,
        "post150_materialization_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "post150_materialization_global_fresh_retry_quota": (MATERIALIZATION_RETRY_QUOTA),
        "post150_materialization_global_fresh_retry_consumed_count": 1,
        "post150_formal_stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "post150_formal_stream_content_sha256_authoritative": False,
        "post150_formal_stream_content_chars_authoritative": False,
        "post150_formal_stream_event_counts_authoritative": False,
        "post150_materialization_adopted_case_count": 1,
        "post150_materialization_adopted_case_ids": ["P3500-HARD-V2-004"],
        "post150_materialization_adopted_episode_ids": [receipt["episode_id"]],
        "post150_materialization_adoption_by_case_id": {"P3500-HARD-V2-004": deepcopy(receipt)},
    }
    expected = {"P3500-HARD-004": receipt}
    assert (
        _materialization_adoption_reporting_errors(
            reviewed_qa=qa,
            reviewed_summary=summary,
            expected_by_parent_case_id=expected,
        )
        == []
    )

    tampered_qa = deepcopy(qa)
    tampered_qa[0]["post150_materialization_adoption"]["source_success_sha256"] = "e" * 64
    assert _materialization_adoption_reporting_errors(
        reviewed_qa=tampered_qa,
        reviewed_summary=summary,
        expected_by_parent_case_id=expected,
    )

    tampered_summary = deepcopy(summary)
    tampered_summary["post150_materialization_adoption_by_case_id"]["P3500-HARD-V2-004"][
        "retry_started_sha256"
    ] = "f" * 64
    assert _materialization_adoption_reporting_errors(
        reviewed_qa=qa,
        reviewed_summary=tampered_summary,
        expected_by_parent_case_id=expected,
    )


def _run_empty_wrapper_with_results(
    wrapper,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    results: list[dict],
    *,
    initial_rows: list[dict] | None = None,
) -> tuple[int, list[dict], dict]:
    _, request, _ = _fixture()
    responses_path = tmp_path / f"{wrapper.__name__}-responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (initial_rows or [])),
        encoding="utf-8",
    )
    summary_path = tmp_path / f"{wrapper.__name__}-summary.json"
    monkeypatch.setattr(wrapper, "RESPONSES_PATH", responses_path)
    monkeypatch.setattr(wrapper, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(wrapper, "_validate_requests", lambda: ([request], 1))
    monkeypatch.setattr(wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(
        wrapper,
        "_read_json",
        lambda _path: {"execution_contract_sha256": EXECUTION_CONTRACT},
    )
    monkeypatch.setattr(
        wrapper,
        "validate_rereview_request_policy",
        lambda *_args, **_kwargs: EXECUTION_CONTRACT,
    )
    monkeypatch.setattr(wrapper, "validate_rereview_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        wrapper,
        "rereview_calibration_ledger",
        lambda _rows, *, requests, **_kwargs: {str(row["custom_id"]): {} for row in requests},
    )
    monkeypatch.setattr(runner, "_acquire_execution_lock", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_recover_latest_saved_same_concept_variant_subset",
        lambda **_kwargs: None,
    )
    calls: list[str] = []

    def execute_one(*, row: dict, **_kwargs) -> dict:
        calls.append(str(row["custom_id"]))
        if not results:
            raise AssertionError("wrapper executed more provider calls than expected")
        result = deepcopy(results.pop(0))
        result["completed_at"] = datetime.now(UTC).isoformat()
        return result

    monkeypatch.setattr(runner, "_execute_one", execute_one)
    monkeypatch.setenv("POST150_WRAPPER_TEST_KEY", "test-only-key")
    code = wrapper.main(
        [
            "--execute",
            "--acknowledge-cost",
            "--confirm-full-review",
            "--confirm-no-concurrent-review",
            "--api-key-env",
            "POST150_WRAPPER_TEST_KEY",
        ]
    )
    rows = _read_jsonl(responses_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calls == [TARGET_ID] * len(calls)
    return code, rows, summary


def _unbound_live_result(source: dict, *, materializer_pass: bool) -> dict:
    result = deepcopy(source)
    result.pop("execution_contract_sha256")
    result.pop("sequence")
    result["completed_at"] = datetime.now(UTC).isoformat()
    if materializer_pass:
        c04 = next(
            concept
            for concept in result["review"]["required_concepts"]["all_of"]
            if concept["concept_key"] == "c04"
        )
        c04["canonical_text"] = "重新分离打印介质"
        result["response_sha256"] = runner._sha256_json(result["review"])
    return result


def _assert_authoritative_progress_summary(
    summary: dict, rows: list[dict], *, completed: int, expected: int
) -> None:
    assert summary["already_completed_case_calls"] == completed
    assert summary["remaining_case_calls"] == expected - completed
    assert summary["usage"] == runner._usage_totals(rows)
    assert summary["api_attempts_recorded"] == runner._attempt_count(rows)
    assert summary["usage_is_lower_bound"] is any(
        row.get("usage_is_lower_bound") is True for row in rows
    )


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_authoritative_summary_snapshot_replays_all_response_rows(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses_path = tmp_path / f"{wrapper.__name__}-responses.jsonl"
    rows = [
        {
            "custom_id": "case-a",
            "status": "succeeded",
            "attempt": 1,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "usage_is_lower_bound": False,
        },
        {
            "custom_id": "case-b",
            "status": "failed",
            "attempt": 3,
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
            },
            "usage_is_lower_bound": True,
        },
    ]
    responses_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    requests = [{"custom_id": "case-a"}, {"custom_id": "case-b"}]
    expected_state = {
        "successful_by_id": {"case-a": rows[0]},
    }
    reducer_calls: list[list[dict]] = []

    def replay_state(actual_rows: list[dict], *_args, **_kwargs) -> dict:
        reducer_calls.append(actual_rows)
        return expected_state

    monkeypatch.setattr(wrapper, "RESPONSES_PATH", responses_path)
    monkeypatch.setattr(wrapper, "_formal_state", replay_state)

    actual_rows, state, progress = wrapper._authoritative_summary_snapshot(
        requests,
        expected_cases=2,
        execution_contract_sha256=EXECUTION_CONTRACT,
        label="summary-snapshot-test",
    )

    assert actual_rows == rows
    assert reducer_calls == [rows]
    assert state is expected_state
    _assert_authoritative_progress_summary(progress, actual_rows, completed=1, expected=2)


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_wrapper_never_reports_success_for_new_materializer_failed_live_result(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source = _fixture()
    code, rows, summary = _run_empty_wrapper_with_results(
        wrapper,
        tmp_path,
        monkeypatch,
        [
            _unbound_live_result(source, materializer_pass=False),
            _unbound_live_result(source, materializer_pass=False),
        ],
    )
    assert code == 2
    assert summary["status"] != "succeeded"
    _assert_authoritative_progress_summary(summary, rows, completed=0, expected=1)
    assert [row["status"] for row in rows] == [
        "succeeded",
        "materialization_invalidated",
        "materialization_retry_started",
        "materialization_rejected",
    ]


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_wrapper_early_saved_success_uses_authoritative_summary_snapshot(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source = _fixture()
    saved_success = _unbound_live_result(source, materializer_pass=True)
    saved_success["execution_contract_sha256"] = EXECUTION_CONTRACT
    saved_success["sequence"] = 1

    code, rows, summary = _run_empty_wrapper_with_results(
        wrapper,
        tmp_path,
        monkeypatch,
        [],
        initial_rows=[saved_success],
    )

    assert code == 0
    assert summary["status"] == "succeeded"
    assert summary["case_success_count"] == 1
    _assert_authoritative_progress_summary(summary, rows, completed=1, expected=1)


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_wrapper_reports_success_only_after_legal_materialization_adoption(
    wrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source = _fixture()
    code, rows, summary = _run_empty_wrapper_with_results(
        wrapper,
        tmp_path,
        monkeypatch,
        [
            _unbound_live_result(source, materializer_pass=False),
            _unbound_live_result(source, materializer_pass=True),
        ],
    )
    assert code == 0
    assert summary["status"] == "succeeded"
    assert summary["case_success_count"] == 1
    _assert_authoritative_progress_summary(summary, rows, completed=1, expected=1)
    assert summary["materialization_adoption"]["adopted_episode_count"] == 1
    assert [row["status"] for row in rows] == [
        "succeeded",
        "materialization_invalidated",
        "materialization_retry_started",
        "succeeded",
    ]
