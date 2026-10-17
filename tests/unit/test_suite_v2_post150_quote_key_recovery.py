from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta
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
if str(CONTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRACT_ROOT))

import merge_reviewed_contracts as merge_reviewed  # noqa: E402
import review_contract_rereview as contract_wrapper  # noqa: E402
import review_contracts as runner  # noqa: E402
import validate_suite_v2 as suite_validator  # noqa: E402
from review_release_integrity import (  # noqa: E402
    EXPECTED_REREVIEW_MODEL,
    formal_rereview_failed_result_errors,
    formal_rereview_success_ledger,
)


TARGET_ID = "concept-contract-rereview-f31fc21248805442494a"
EXPECTED_SOURCE_ROW_SHA256 = (
    "cc254dbd9ea8d594d4310e8e09ae75a50b5b50f2f2c010e7aebb23d852aed939"
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


def _fixture() -> tuple[list[dict], list[dict], dict, dict, dict]:
    root = CONTRACT_ROOT / "contract-rereview"
    requests = _read_jsonl(root / "requests.jsonl")
    responses = _read_jsonl(root / "responses.jsonl")
    request = next(row for row in requests if row["custom_id"] == TARGET_ID)
    source = next(
        row
        for row in responses
        if runner._sha256_json(row) == EXPECTED_SOURCE_ROW_SHA256
    )
    payload = json.loads(request["request"]["messages"][1]["content"])
    return requests, responses, request, payload, source


def _source_bound_prefix(responses: list[dict]) -> list[dict]:
    source_index = next(
        index
        for index, row in enumerate(responses)
        if runner._sha256_json(row) == EXPECTED_SOURCE_ROW_SHA256
    )
    return responses[: source_index + 1]


def _recover() -> tuple[list[dict], dict, dict, dict]:
    _requests, responses, request, payload, _source = _fixture()
    source_rows = _source_bound_prefix(responses)
    result = runner._recover_latest_saved_quote_key_branch_span(
        row=request,
        response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert result is not None
    return source_rows, request, payload, result


def _failed_validator(requests: list[dict]):
    sequence_by_id = {
        str(request["custom_id"]): sequence
        for sequence, request in enumerate(requests, start=1)
    }

    def validate(result: dict, request: dict) -> list[str]:
        return formal_rereview_failed_result_errors(
            result,
            request,
            expected_sequence=sequence_by_id[str(request["custom_id"])],
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    return validate


def test_real_quote_key_projection_is_exact_and_read_only() -> None:
    _, _, _, payload, source = _fixture()
    responses_path = CONTRACT_ROOT / "contract-rereview" / "responses.jsonl"
    before = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    raw = source["invalid_attempts"][-1]["raw_review"]

    projection = runner._project_saved_quote_key_branch_span(raw, payload)
    assert projection is not None
    projected, actions, raw_errors, materialization_audit = projection
    branch = projected["required_branches"][0]
    assert branch == {
        "branch_label": "B1",
        "query_span": "控制面板上的“返回键”可以执行哪些操作",
        "required_concept_keys": ["c01", "c02", "c03"],
    }
    assert [action["operation"] for action in actions] == [
        "remove_quoted_schema_extra_key",
        "replace_branch_span_with_exact_payload_span",
    ]
    assert raw_errors == source["invalid_attempts"][-1]["validation_errors"]
    assert materialization_audit["production_match_collision_count"] == 0
    assert hashlib.sha256(responses_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("wrong_key", "返回"),
        ("wrong_key", "返回键可以"),
        ("wrong_value", ","),
        ("wrong_value", " "),
        ("wrong_value", None),
        ("other_extra", "x"),
        ("multiple_quote_pairs", None),
        ("non_prefix", None),
    ),
)
def test_quote_key_projector_rejects_every_non_exact_shape(
    mutation: str,
    value: object,
) -> None:
    _, _, _, payload, source = _fixture()
    raw = deepcopy(source["invalid_attempts"][-1]["raw_review"])
    branch = raw["required_branches"][0]
    if mutation == "wrong_key":
        branch[str(value)] = branch.pop("返回键")
    elif mutation == "wrong_value":
        branch["返回键"] = value
    elif mutation == "other_extra":
        branch[str(value)] = ""
    elif mutation == "multiple_quote_pairs":
        payload = deepcopy(payload)
        payload["branches"][0]["query_span"] = (
            "控制面板上的“返回键”“取消键”可以执行哪些操作"
        )
    else:
        branch["query_span"] = "控制面板"

    assert runner._project_saved_quote_key_branch_span(raw, payload) is None


def test_real_quote_key_saved_receipt_is_zero_cost_and_exactly_replayable() -> None:
    responses, request, payload, result = _recover()
    source = responses[-1]
    assert result["runner_version"] == (
        runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION
    )
    assert result["transport"] == runner.SAVED_RESPONSE_TRANSPORT
    assert result["attempt"] == 0
    assert result["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    audit = result["saved_response_quote_key_branch_span_recovery"]
    assert audit["api_call_performed"] is False
    assert len(audit["actions"]) == 2
    assert audit["source"]["response_row_sha256"] == EXPECTED_SOURCE_ROW_SHA256
    assert audit["source"]["source_attempt"] == 3
    assert audit["completed_at_policy"] == (
        runner.SAVED_QUOTE_KEY_RECOVERY_EVENT_TIME_POLICY
    )
    assert result["completed_at"] == (
        datetime.fromisoformat(source["completed_at"]) + timedelta(microseconds=1)
    ).isoformat()
    assert runner._validate_saved_post150_recovery(
        result,
        request=request,
        semantic_payload=payload,
        source_response_rows=responses,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    ) == []


@pytest.mark.parametrize(
    "tamper",
    (
        "policy",
        "actions",
        "source",
        "review",
        "usage",
        "api",
        "completed_at",
        "usage_bool",
        "top_bool",
        "audit_bool",
        "prior_bool",
        "selection_bool",
        "float_attempt",
        "float_sequence",
        "float_source_attempt",
    ),
)
def test_quote_key_saved_receipt_rejects_tamper(tamper: str) -> None:
    responses, request, payload, result = _recover()
    changed = deepcopy(result)
    audit = changed["saved_response_quote_key_branch_span_recovery"]
    if tamper == "policy":
        audit["policy"] = "foreign"
    elif tamper == "actions":
        audit["actions"] = list(reversed(audit["actions"]))
    elif tamper == "source":
        audit["source"]["source_attempt"] = 2
    elif tamper == "review":
        changed["review"]["required_branches"][0]["query_span"] = "foreign"
    elif tamper == "usage":
        changed["usage"]["total_tokens"] = 1
    elif tamper == "api":
        audit["api_call_performed"] = True
    elif tamper == "completed_at":
        changed["completed_at"] = "2099-01-01T00:00:00+00:00"
    elif tamper == "usage_bool":
        changed["usage"]["total_tokens"] = False
    elif tamper == "top_bool":
        changed["usage_is_lower_bound"] = 0
    elif tamper == "audit_bool":
        audit["api_call_performed"] = 0
    elif tamper == "prior_bool":
        changed["prior_response_recovery"]["api_call_performed"] = 0
    elif tamper == "selection_bool":
        audit["selection"]["final_complete_terminal_attempt"] = 1
    elif tamper == "float_attempt":
        changed["attempt"] = 0.0
    elif tamper == "float_sequence":
        changed["sequence"] = 1.0
    else:
        audit["source"]["source_attempt"] = 3.0
    assert runner._validate_saved_post150_recovery(
        changed,
        request=request,
        semantic_payload=payload,
        source_response_rows=responses,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )


def test_quote_key_saved_receipt_rejects_nonlatest_future_self_and_duplicate_source() -> None:
    responses, request, payload, result = _recover()
    source = responses[-1]
    assert runner._sha256_json(source) == EXPECTED_SOURCE_ROW_SHA256
    later = deepcopy(source)
    later["completed_at"] = "2099-01-01T00:00:00+00:00"
    cases = (
        [*responses, later],
        [result, *responses],
        [*responses, result, deepcopy(result)],
        [*responses, deepcopy(source)],
    )
    for rows in cases:
        assert runner._validate_saved_post150_recovery(
            result,
            request=request,
            semantic_payload=payload,
            source_response_rows=rows,
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        )


def test_formal_reducer_accepts_both_saved_profiles_and_latest_failed_wins() -> None:
    requests, physical_responses, request, _payload, _ = _fixture()
    responses = _source_bound_prefix(physical_responses)
    quote = runner._recover_latest_saved_quote_key_branch_span(
        row=request,
        response_rows=responses,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert quote is not None
    request_by_id = {str(row["custom_id"]): row for row in requests}
    kwargs = {
        "requests_by_id": request_by_id,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT,
        "expected_prompt_sha256": request["prompt_sha256"],
        "expected_model": EXPECTED_REREVIEW_MODEL,
        "live_runner_version": runner.RUNNER_VERSION,
        "live_transport": runner.TRANSPORT,
        "saved_runner_version": runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        "saved_transport": runner.SAVED_RESPONSE_TRANSPORT,
        "additional_saved_profiles": (
            (
                runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
                runner.SAVED_RESPONSE_TRANSPORT,
            ),
        ),
        "validate_live_result": lambda *_args: [],
        "validate_saved_result": lambda result, current, rows: (
            runner._validate_saved_post150_recovery(
                result,
                request=current,
                semantic_payload=json.loads(
                    current["request"]["messages"][1]["content"]
                ),
                source_response_rows=rows,
                execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
            )
        ),
        "validate_failed_result": _failed_validator(requests),
        "label": "quote-key-dual-profile",
    }
    ledger = formal_rereview_success_ledger([*responses, quote], **kwargs)
    assert ledger[TARGET_ID] == quote
    later_failure = deepcopy(responses[-1])
    later_failure["completed_at"] = "2099-01-01T00:00:00+00:00"
    ledger_after_failure = formal_rereview_success_ledger(
        [*responses, quote, later_failure],
        **kwargs,
    )
    assert TARGET_ID not in ledger_after_failure

    assert contract_wrapper._formal_success_ledger(
        [*responses, quote],
        requests,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        label="wrapper-quote-parity",
    )[TARGET_ID] == quote
    assert merge_reviewed._formal_post150_success_ledger(
        [*responses, quote],
        request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_hash=request["prompt_sha256"],
        label="merge-quote-parity",
    )[TARGET_ID] == quote
    assert suite_validator._formal_post150_success_ledger(
        [*responses, quote],
        request_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_hash=request["prompt_sha256"],
        label="validator-quote-parity",
    )[TARGET_ID] == quote


def test_legacy_saved_profile_remains_replayable() -> None:
    blocker_root = CONTRACT_ROOT / "blocker-rereview"
    requests = _read_jsonl(blocker_root / "requests.jsonl")
    responses = _read_jsonl(blocker_root / "responses.jsonl")
    legacy_id = "concept-review-c753ea69e724ecaf38ce"
    request = next(row for row in requests if row["custom_id"] == legacy_id)
    source_index = next(
        index
        for index, row in enumerate(responses)
        if row.get("custom_id") == legacy_id
        and row.get("status") == "failed"
        and runner._recover_latest_saved_same_concept_variant_subset(
            row=request,
            response_rows=responses[: index + 1],
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        )
        is not None
    )
    source_rows = responses[: source_index + 1]
    legacy = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert legacy is not None
    payload = json.loads(request["request"]["messages"][1]["content"])
    assert runner._validate_saved_post150_recovery(
        legacy,
        request=request,
        semantic_payload=payload,
        source_response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    ) == []

    persisted = next(
        row
        for row in responses[source_index + 1 :]
        if row.get("custom_id") == legacy_id
        and row.get("runner_version")
        == runner.POST150_SAVED_RECOVERY_RUNNER_VERSION
    )
    later_failure = deepcopy(responses[source_index])
    later_failure["completed_at"] = "2099-01-01T00:00:00+00:00"
    requests_by_id = {str(row["custom_id"]): row for row in requests}
    ledger = formal_rereview_success_ledger(
        [responses[source_index], persisted, later_failure],
        requests_by_id=requests_by_id,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        expected_prompt_sha256=request["prompt_sha256"],
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        additional_saved_profiles=(
            (
                runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
                runner.SAVED_RESPONSE_TRANSPORT,
            ),
        ),
        validate_live_result=lambda *_args: [],
        validate_saved_result=lambda result, current, rows: (
            runner._validate_saved_post150_recovery(
                result,
                request=current,
                semantic_payload=json.loads(
                    current["request"]["messages"][1]["content"]
                ),
                source_response_rows=rows,
                execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
            )
        ),
        validate_failed_result=_failed_validator(requests),
        label="legacy-saved-latest-failed",
    )
    assert legacy_id not in ledger


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("usage", "prompt_tokens"), False),
        (("usage", "completion_tokens"), 0.0),
        (("usage", "total_tokens"), False),
        (("usage_is_lower_bound",), 0),
        (
            (
                "saved_response_same_concept_variant_subset_recovery",
                "api_call_performed",
            ),
            0,
        ),
        (("prior_response_recovery", "api_call_performed"), 0),
        (("attempt",), 0.0),
        (("sequence",), 8.0),
    ),
)
def test_legacy_saved_profile_rejects_scalar_type_confusion(
    path: tuple[str, ...], value: object
) -> None:
    blocker_root = CONTRACT_ROOT / "blocker-rereview"
    requests = _read_jsonl(blocker_root / "requests.jsonl")
    responses = _read_jsonl(blocker_root / "responses.jsonl")
    legacy_id = "concept-review-c753ea69e724ecaf38ce"
    request = next(row for row in requests if row["custom_id"] == legacy_id)
    source_index = next(
        index
        for index, row in enumerate(responses)
        if row.get("custom_id") == legacy_id
        and row.get("status") == "failed"
        and runner._recover_latest_saved_same_concept_variant_subset(
            row=request,
            response_rows=responses[: index + 1],
            execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
        )
        is not None
    )
    source_rows = responses[: source_index + 1]
    legacy = runner._recover_latest_saved_same_concept_variant_subset(
        row=request,
        response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )
    assert legacy is not None
    changed = deepcopy(legacy)
    target: object = changed
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value
    payload = json.loads(request["request"]["messages"][1]["content"])
    assert runner._validate_saved_post150_recovery(
        changed,
        request=request,
        semantic_payload=payload,
        source_response_rows=source_rows,
        execution_contract_sha256=EXPECTED_EXECUTION_CONTRACT,
    )


def test_contract_wrapper_appends_quote_recovery_before_key_or_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, physical_responses, request, _payload, _source = _fixture()
    responses = _source_bound_prefix(physical_responses)
    monkeypatch.setattr(
        contract_wrapper,
        "_validate_requests",
        lambda: (requests, len(requests)),
    )
    monkeypatch.setattr(contract_wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(
        contract_wrapper,
        "validate_rereview_runtime",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        contract_wrapper.runner,
        "_acquire_execution_lock",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        contract_wrapper.runner,
        "_read_jsonl",
        lambda path: (
            responses
            if path == contract_wrapper.RESPONSES_PATH
            else []
            if path == contract_wrapper.runner.CALIBRATION_RESPONSES_PATH
            else requests
        ),
    )
    monkeypatch.delenv("QUOTE_KEY_SAVED_ONLY_KEY", raising=False)
    monkeypatch.setattr(
        contract_wrapper.runner,
        "_execute_one",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )
    appended: list[dict] = []

    class StopAfterSavedAppend(RuntimeError):
        pass

    def append_once(_path: Path, row: dict) -> None:
        appended.append(deepcopy(row))
        raise StopAfterSavedAppend

    monkeypatch.setattr(contract_wrapper.runner, "_append_jsonl", append_once)
    with pytest.raises(StopAfterSavedAppend):
        contract_wrapper.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
                "--api-key-env",
                "QUOTE_KEY_SAVED_ONLY_KEY",
            ]
        )
    assert len(appended) == 1
    assert appended[0]["custom_id"] == request["custom_id"] == TARGET_ID
    assert appended[0]["runner_version"] == (
        runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION
    )
    assert appended[0]["usage"]["total_tokens"] == 0
