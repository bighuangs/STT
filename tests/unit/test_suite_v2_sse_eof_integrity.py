from __future__ import annotations

import json
import io
import hashlib
import http.client
import ssl
import sys
import urllib.error
from copy import deepcopy
from pathlib import Path
from typing import Any

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

from review_contracts import (  # noqa: E402
    RUNNER_VERSION,
    SSEIntegrityError,
    SSE_COMPLETION_POLICY,
    TRANSPORT,
    _calibration_requests,
    _completed_calibrations_by_id,
    _canonicalize_same_response_inverse_edges,
    _complete_terminal_semantic_attempt,
    _dedupe_same_response_exact_variants,
    _execute_one,
    _project_fresh_calibration_abstention,
    _recover_prior_invalid_review,
    _review_schema_extra_paths,
    _sha256_json,
    _validate_against_payload,
    _validate_calibration_result,
    _validate_coverage_status_canonicalization_audit,
    _validate_exact_variant_deduplication_audit,
    _validate_inverse_edge_canonicalization_audit,
    _validate_invalid_attempt_raw_normalization_provenance,
    _validate_review,
)


def _projectable_abstain_review() -> dict[str, Any]:
    return {
        "verdict": "approve",
        "answerability": "abstain",
        "required_concepts": {
            "all_of": [
                {
                    "concept_key": "c01",
                    "canonical_text": "知识库未提供实时价格，无法确定",
                    "any_of": ["未提供实时价格", "无法确定当前价格"],
                    "evidence_labels": [],
                    "branch_labels": ["B1"],
                }
            ]
        },
        "critical_constraints": {
            "numerics": [],
            "identifiers": [],
            "polarity": {"required": "negative", "cues": ["无法确定"]},
        },
        "required_branches": [
            {
                "branch_label": "B1",
                "query_span": "这台设备今天在上海的现货价是多少",
                "required_concept_keys": ["c01"],
            }
        ],
        "evidence_concept_coverage": [],
        "review_notes": ["资料不足，只能拒答。"],
    }


def _semantic_payload() -> dict[str, Any]:
    return {
        "query": "默认接口是什么？",
        "reference_answer": "默认接口为 USB。",
        "unanswerable": False,
        "legacy_expected_terms": ["USB"],
        "forbidden_terms": [],
        "branches": [{"branch_label": "B1", "query_span": "默认接口是什么"}],
        "evidence": [
            {"evidence_label": "E1", "page": 1, "text": "默认接口：USB。"}
        ],
        "deterministic_candidate": {},
    }


def _valid_review() -> dict[str, Any]:
    return {
        "verdict": "approve",
        "answerability": "answer",
        "required_concepts": {
            "all_of": [
                {
                    "concept_key": "c01",
                    "canonical_text": "默认接口为 USB",
                    "any_of": ["默认接口为 USB", "USB"],
                    "evidence_labels": ["E1"],
                    "branch_labels": ["B1"],
                }
            ]
        },
        "critical_constraints": {
            "numerics": [],
            "identifiers": [{"value": "USB", "evidence_labels": ["E1"]}],
            "polarity": {"required": "not_applicable", "cues": []},
        },
        "required_branches": [
            {
                "branch_label": "B1",
                "query_span": "默认接口是什么",
                "required_concept_keys": ["c01"],
            }
        ],
        "evidence_concept_coverage": [
            {
                "evidence_label": "E1",
                "required_concept_keys": ["c01"],
                "status": "covered",
            }
        ],
        "review_notes": ["Evidence E1 directly supports the contract."],
    }


def _request_row() -> dict[str, Any]:
    return {
        "custom_id": "synthetic-sse-eof",
        "prompt_sha256": "synthetic-prompt",
        "request": {
            "model": "prepared-model",
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {
                    "role": "user",
                    "content": json.dumps(
                        _semantic_payload(), ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
        },
    }


def _event(*, content: str | None = None, finish_reason: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _usage_event() -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class _FakeResponse:
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}

    def __init__(self, events: list[dict[str, Any]], *, done: bool = False) -> None:
        self._events = events
        self._done = done

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        for event in self._events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n".encode("utf-8")
        if self._done:
            yield b"data: [DONE]\n"


def _run(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    *,
    done: bool = False,
) -> dict[str, Any]:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: _FakeResponse(events, done=done),
    )
    return _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )


def _retry_protocol_error() -> SSEIntegrityError:
    return SSEIntegrityError(
        "synthetic incomplete stream",
        audit={
            "completion_policy": SSE_COMPLETION_POLICY,
            "termination": "protocol_error",
            "done_seen": False,
            "finish_reason": None,
            "finish_reasons": [],
            "finish_reason_stop_seen": False,
            "event_count": 1,
            "content_event_count": 1,
            "content_chars": 10,
            "event_integrity": "failed",
            "usage_integrity": {"status": "missing"},
            "review_validation": "not_run",
        },
        validated_usage={},
    )


def _complete_call_result(kwargs: dict[str, Any], review: dict[str, Any]):
    kwargs["integrity_audit"].update(
        {
            "completion_policy": SSE_COMPLETION_POLICY,
            "termination": "done_sentinel",
            "done_seen": True,
            "finish_reason": "stop",
            "finish_reasons": ["stop"],
            "finish_reason_stop_seen": True,
            "event_count": 4,
            "content_event_count": 2,
            "content_chars": len(json.dumps(review, ensure_ascii=False)),
            "content_sha256": _sha256_json(review),
            "event_integrity": "passed",
            "usage_integrity": {
                "status": "passed",
                "usage_event_count": 1,
                "required_token_fields_valid": True,
                "total_matches_prompt_plus_completion": True,
                "at_or_after_finish_reason_stop": True,
            },
            "review_validation": "deferred_to_existing_validation",
        }
    )
    return review, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def _run_retry_sequence(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[str], *, max_attempts: int = 3
) -> tuple[dict[str, Any], list[str]]:
    observed: list[str] = []

    def fake_call_api(**kwargs):
        outcome = outcomes[len(observed)]
        observed.append(outcome)
        if outcome == "transport":
            raise _retry_protocol_error()
        if outcome == "semantic":
            incomplete = {
                "verdict": "revise",
                "answerability": "answer",
                "required_concepts": {"all_of": []},
            }
            return _complete_call_result(kwargs, incomplete)
        if outcome == "success":
            return _complete_call_result(kwargs, _valid_review())
        raise AssertionError(outcome)

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=max_attempts,
        retry_delay=0,
        calibration=False,
    )
    return result, observed


def _run_structural_normalization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_review = _valid_review()
    raw_review["verdict"] = "revise"
    concept = raw_review["required_concepts"]["all_of"][0]
    concept.pop("any_of")
    concept.pop("evidence_labels")
    concept.pop("branch_labels")

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, deepcopy(raw_review))

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )
    return result, raw_review


def _unanswerable_calibration_row() -> dict[str, Any]:
    return next(
        row
        for row in _calibration_requests(
            _request_row(),
            model="test-model",
            endpoint="https://example.invalid/chat/completions",
        )
        if str(row["custom_id"]).endswith("-unanswerable_no_fact")
    )


def test_formal_live_response_never_uses_saved_abstain_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _unanswerable_calibration_row()
    row.pop("expected")
    raw = _projectable_abstain_review()

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, deepcopy(raw))

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )

    assert result["status"] == "failed"
    assert result["invalid_review"] == raw
    assert "review" not in result
    assert "prior_response_recovery" not in result
    assert "saved_response_abstain_factual_constraint_recovery" not in result


def test_fresh_unanswerable_calibration_returns_projectable_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _unanswerable_calibration_row()
    raw = _projectable_abstain_review()

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, deepcopy(raw))

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )

    assert result["status"] == "calibration_projectable_pass"
    assert result["raw_review"] == raw
    assert "review" not in result
    assert result["projected_review"]["critical_constraints"]["polarity"] == {
        "required": "abstain_without_factual_claims",
        "cues": ["资料不足", "无法确定"],
    }
    audit = result["calibration_abstain_sentinel_projection"]
    assert audit["raw_model_contract_pass"] is False
    assert audit["raw_semantics_not_authoritative"] is True
    assert audit["projected_contract_pass"] is True
    assert audit["formal_response_written"] is False
    assert _validate_calibration_result(result, row) == []

    path = tmp_path / "calibration.jsonl"
    path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    assert _completed_calibrations_by_id(path, [row]) == {
        row["custom_id"]: result
    }


@pytest.mark.parametrize(
    ("event_count", "content_event_count"),
    [(506, 505), (2, 1)],
)
def test_calibration_ledger_accepts_single_noncontent_json_event_shape(
    monkeypatch: pytest.MonkeyPatch,
    event_count: int,
    content_event_count: int,
) -> None:
    row = _unanswerable_calibration_row()

    def fake_call_api(**kwargs):
        result = _complete_call_result(kwargs, _projectable_abstain_review())
        kwargs["integrity_audit"].update(
            event_count=event_count,
            content_event_count=content_event_count,
        )
        return result

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )

    assert _validate_calibration_result(result, row) == []


@pytest.mark.parametrize("content_event_count", [506, 507])
def test_calibration_ledger_requires_one_noncontent_json_event(
    monkeypatch: pytest.MonkeyPatch,
    content_event_count: int,
) -> None:
    row = _unanswerable_calibration_row()

    def fake_call_api(**kwargs):
        result = _complete_call_result(kwargs, _projectable_abstain_review())
        kwargs["integrity_audit"].update(
            event_count=506,
            content_event_count=content_event_count,
        )
        return result

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )

    assert _validate_calibration_result(result, row)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda result: (
            result.update(status="succeeded"),
            result.update(review=deepcopy(result["projected_review"])),
        ),
        lambda result: result.update(status="succeeded"),
        lambda result: result["projected_review"]["critical_constraints"][
            "polarity"
        ]["cues"].append("价格是2999元"),
        lambda result: result["calibration_abstain_sentinel_projection"].update(
            formal_response_written=True
        ),
        lambda result: result["calibration_abstain_sentinel_projection"].update(
            calibration_manifest_sha256="0" * 64
        ),
        lambda result: result["calibration_abstain_sentinel_projection"][
            "actions"
        ][0].update(json_paths=["/verdict"]),
        lambda result: result["sse_integrity"].update(finish_reason="length"),
        lambda result: result.update(attempt_usage=[]),
        lambda result: result.update(
            successful_response_usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        ),
        lambda result: result["sse_integrity"].update(event_integrity="failed"),
        lambda result: result["attempt_usage"][0].update(attempt=999),
        lambda result: result["attempt_usage"][0].update(prompt_tokens=-1),
        lambda result: result["attempt_usage"][0].update(total_tokens=999),
        lambda result: result["sse_integrity"].update(content_sha256="0" * 64),
        lambda result: result["sse_integrity"].update(done_seen=False),
        lambda result: result["sse_integrity"].update(completion_policy="other"),
        lambda result: result["sse_integrity"].update(
            event_count=1, content_event_count=999
        ),
        lambda result: result["sse_integrity"].update(content_chars=1),
        lambda result: result["sse_integrity"].update(extra="tamper"),
        lambda result: result.update(completed_at="not-a-time"),
    ],
)
def test_calibration_projectable_receipt_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tamper,
) -> None:
    row = _unanswerable_calibration_row()

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, _projectable_abstain_review())

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )
    tamper(result)

    assert _validate_calibration_result(result, row)


def test_stream_hash_is_diagnostic_but_parsed_and_raw_hashes_authorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _unanswerable_calibration_row()

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, _projectable_abstain_review())

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )
    result["sse_integrity"]["content_sha256"] = "0" * 64
    result["sse_integrity_attempts"][0]["content_sha256"] = "0" * 64
    result["calibration_abstain_sentinel_projection"][
        "terminal_content_sha256"
    ] = "0" * 64
    assert _validate_calibration_result(result, row) == []

    parsed_tamper = deepcopy(result)
    parsed_tamper["sse_integrity"]["parsed_review_sha256"] = "1" * 64
    assert _validate_calibration_result(parsed_tamper, row)
    raw_tamper = deepcopy(result)
    raw_tamper["calibration_abstain_sentinel_projection"][
        "raw_review_sha256"
    ] = "1" * 64
    assert _validate_calibration_result(raw_tamper, row)


def test_current_calibration_loader_never_reuses_prior_runner_rows(
    tmp_path: Path,
) -> None:
    row = _unanswerable_calibration_row()
    old = {
        "custom_id": str(row["custom_id"]).replace(
            RUNNER_VERSION, "2.27.0-saved-abstain-sentinel-recovery"
        ),
        "status": "succeeded",
        "runner_version": "2.27.0-saved-abstain-sentinel-recovery",
        "prompt_sha256": row["prompt_sha256"],
    }
    path = tmp_path / "calibration.jsonl"
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")

    assert _completed_calibrations_by_id(path, [row]) == {}


def test_calibration_loader_uses_latest_append_only_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = _unanswerable_calibration_row()

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, _projectable_abstain_review())

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    good = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )
    bad = deepcopy(good)
    bad["response_sha256"] = "0" * 64
    path = tmp_path / "calibration.jsonl"

    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in (good, bad))
        + "\n",
        encoding="utf-8",
    )
    assert _completed_calibrations_by_id(path, [row]) == {}

    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in (bad, good))
        + "\n",
        encoding="utf-8",
    )
    assert _completed_calibrations_by_id(path, [row]) == {row["custom_id"]: good}


def test_calibration_execution_contract_binds_model_and_endpoint() -> None:
    template = _request_row()
    first = _calibration_requests(
        template,
        model="model-a",
        endpoint="https://a.invalid/chat/completions",
    )
    same = _calibration_requests(
        template,
        model="model-a",
        endpoint="https://a.invalid/chat/completions",
    )
    model_drift = _calibration_requests(
        template,
        model="model-b",
        endpoint="https://a.invalid/chat/completions",
    )
    endpoint_drift = _calibration_requests(
        template,
        model="model-a",
        endpoint="https://b.invalid/chat/completions",
    )

    assert [row["custom_id"] for row in first] == [
        row["custom_id"] for row in same
    ]
    assert {row["custom_id"] for row in first}.isdisjoint(
        row["custom_id"] for row in model_drift
    )
    assert {row["custom_id"] for row in first}.isdisjoint(
        row["custom_id"] for row in endpoint_drift
    )
    assert first[0]["calibration_execution_contract_sha256"] != model_drift[0][
        "calibration_execution_contract_sha256"
    ]
    assert first[0]["calibration_execution_contract_sha256"] != endpoint_drift[0][
        "calibration_execution_contract_sha256"
    ]


@pytest.mark.parametrize(
    ("first_config", "second_config"),
    [
        ({"max_tokens": 1}, {"max_tokens": 999999}),
        ({}, {"top_p": 0.01}),
        ({"seed": 1}, {"seed": 2}),
    ],
)
def test_calibration_execution_contract_hashes_all_request_knobs(
    first_config: dict[str, Any], second_config: dict[str, Any]
) -> None:
    first_template = _request_row()
    second_template = deepcopy(first_template)
    first_template["request"].update(first_config)
    second_template["request"].update(second_config)

    first = _calibration_requests(
        first_template,
        model="model-a",
        endpoint="https://a.invalid/chat/completions",
    )
    second = _calibration_requests(
        second_template,
        model="model-a",
        endpoint="https://a.invalid/chat/completions",
    )

    assert first[0]["calibration_execution_contract_sha256"] != second[0][
        "calibration_execution_contract_sha256"
    ]
    assert {row["custom_id"] for row in first}.isdisjoint(
        row["custom_id"] for row in second
    )


@pytest.mark.parametrize(
    ("model", "endpoint"),
    [
        ("model-b", "https://a.invalid/chat/completions"),
        ("model-a", "https://b.invalid/chat/completions"),
    ],
)
def test_calibration_call_rejects_execution_contract_drift_before_api(
    monkeypatch: pytest.MonkeyPatch, model: str, endpoint: str
) -> None:
    row = _calibration_requests(
        _request_row(),
        model="model-a",
        endpoint="https://a.invalid/chat/completions",
    )[0]
    called = False

    def fake_call_api(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must reject before API")

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    with pytest.raises(ValueError, match="model/endpoint contract mismatch"):
        _execute_one(
            row=row,
            endpoint=endpoint,
            api_key="masked",
            model=model,
            timeout=10,
            max_attempts=1,
            retry_delay=0,
            calibration=True,
        )
    assert called is False


def test_raw_calibration_result_cannot_be_rebound_to_another_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _request_row()
    row_a = next(
        row
        for row in _calibration_requests(
            template,
            model="model-a",
            endpoint="https://a.invalid/chat/completions",
        )
        if str(row["custom_id"]).endswith("-unanswerable_no_fact")
    )
    row_b = next(
        row
        for row in _calibration_requests(
            template,
            model="model-b",
            endpoint="https://b.invalid/chat/completions",
        )
        if str(row["custom_id"]).endswith("-unanswerable_no_fact")
    )
    payload = json.loads(row_a["request"]["messages"][1]["content"])
    exact, actions, _errors = _project_fresh_calibration_abstention(
        _projectable_abstain_review(), payload, row_a["expected"]
    )
    assert actions

    def fake_call_api(**kwargs):
        return _complete_call_result(kwargs, deepcopy(exact))

    monkeypatch.setattr("review_contracts._call_api", fake_call_api)
    result = _execute_one(
        row=row_a,
        endpoint="https://a.invalid/chat/completions",
        api_key="masked",
        model="model-a",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=True,
    )
    assert result["status"] == "succeeded"
    assert _validate_calibration_result(result, row_a) == []

    rebound = deepcopy(result)
    rebound["custom_id"] = row_b["custom_id"]
    rebound["model"] = "model-b"
    assert _validate_calibration_result(rebound, row_b)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review, payload, expected: review.update(answerability="answer"),
        lambda review, payload, expected: payload.update(unanswerable=False),
        lambda review, payload, expected: expected.pop(
            "calibration_projectable_abstention"
        ),
        lambda review, payload, expected: review["critical_constraints"][
            "numerics"
        ].append({"value": "2999", "unit": "元", "evidence_labels": []}),
        lambda review, payload, expected: review["required_concepts"]["all_of"][
            0
        ].update(evidence_labels=["E1"]),
        lambda review, payload, expected: review[
            "evidence_concept_coverage"
        ].append(
            {
                "evidence_label": "E1",
                "required_concept_keys": [],
                "status": "uncovered",
            }
        ),
        lambda review, payload, expected: review["required_branches"].clear(),
        lambda review, payload, expected: review["required_concepts"]["all_of"].append(
            deepcopy(review["required_concepts"]["all_of"][0])
        ),
    ],
)
def test_calibration_projector_rejects_nonrepresentational_defects(mutator) -> None:
    row = _unanswerable_calibration_row()
    payload = json.loads(row["request"]["messages"][1]["content"])
    expected = deepcopy(row["expected"])
    review = _projectable_abstain_review()
    mutator(review, payload, expected)

    projected, actions, _errors = _project_fresh_calibration_abstention(
        review, payload, expected
    )

    assert projected == review
    assert actions == []


def test_invalid_attempt_persists_replayable_raw_normalization_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, raw_review = _run_structural_normalization_failure(monkeypatch)

    assert result["status"] == "failed"
    assert result["runner_version"] == RUNNER_VERSION
    item = result["invalid_attempts"][0]
    raw_hash = _sha256_json(raw_review)
    assert item["raw_review"] == raw_review
    assert item["raw_review_sha256"] == raw_hash
    assert item["pre_normalization_review_sha256"] == raw_hash
    assert item["review_sha256"] == _sha256_json(item["review"])
    assert item["normalization_provenance"] == {
        "policy": "strict-rejected-raw-normalization-provenance-v1",
        "raw_review_sha256": raw_hash,
        "pre_normalization_review_sha256": raw_hash,
        "normalized_review_sha256": item["review_sha256"],
        "actions": [
            "self_filled_any_of:c1",
            "crossref_filled_evidence_labels:c1",
            "crossref_filled_branch_labels:c1",
        ],
    }
    assert item["review"]["required_concepts"]["all_of"][0] == {
        "concept_key": "c01",
        "canonical_text": "默认接口为 USB",
        "any_of": ["默认接口为 USB"],
        "evidence_labels": ["E1"],
        "branch_labels": ["B1"],
    }
    assert item["validation_errors"] == [
        "strict runner rejects structural normalization: 3 action(s)"
    ]
    assert result["failure_attempts"][0]["response_sha256"] == raw_hash
    assert _validate_invalid_attempt_raw_normalization_provenance(
        item,
        result,
        _semantic_payload(),
    ) == []
    assert _complete_terminal_semantic_attempt(
        item,
        result,
        _semantic_payload(),
    )
    assert _recover_prior_invalid_review(
        row=_request_row(), response_rows=[result]
    ) is None

    legacy = deepcopy(result)
    legacy["runner_version"] = "2.22.0-saved-quote-fragment-inverse-composition"
    legacy_item = legacy["invalid_attempts"][0]
    for key in (
        "raw_review",
        "raw_review_sha256",
        "pre_normalization_review_sha256",
        "normalization_provenance",
    ):
        legacy_item.pop(key)
    legacy["failure_attempts"][0]["response_sha256"] = legacy_item[
        "review_sha256"
    ]
    assert _validate_invalid_attempt_raw_normalization_provenance(
        legacy_item,
        legacy,
        _semantic_payload(),
    ) == []
    assert _complete_terminal_semantic_attempt(
        legacy_item,
        legacy,
        _semantic_payload(),
    )
    assert _recover_prior_invalid_review(
        row=_request_row(), response_rows=[legacy]
    ) is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda item: item["raw_review"].update(verdict="block"),
        lambda item: item.update(raw_review_sha256="0" * 64),
        lambda item: item.update(pre_normalization_review_sha256="0" * 64),
        lambda item: item["review"].update(verdict="block"),
        lambda item: item.update(review_sha256="0" * 64),
        lambda item: item["normalization_provenance"].update(
            normalized_review_sha256="0" * 64
        ),
        lambda item: item["normalization_provenance"]["actions"].reverse(),
        lambda item: item["normalization_provenance"].update(
            policy="untrusted"
        ),
        lambda item: item.pop("raw_review"),
    ],
)
def test_invalid_attempt_raw_normalization_provenance_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tamper,
) -> None:
    result, _raw_review = _run_structural_normalization_failure(monkeypatch)
    item = result["invalid_attempts"][0]
    tamper(item)

    assert _validate_invalid_attempt_raw_normalization_provenance(
        item,
        result,
        _semantic_payload(),
    )
    assert not _complete_terminal_semantic_attempt(
        item,
        result,
        _semantic_payload(),
    )


def test_raw_provenance_remains_required_after_future_runner_bump_and_unknown_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _raw_review = _run_structural_normalization_failure(monkeypatch)
    item = result["invalid_attempts"][0]
    item["raw_review"]["verdict"] = "block"
    monkeypatch.setattr("review_contracts.RUNNER_VERSION", "2.24.0-simulated-future")

    assert _validate_invalid_attempt_raw_normalization_provenance(
        item,
        result,
        _semantic_payload(),
    )

    unknown = deepcopy(result)
    unknown["runner_version"] = "unknown-runner"
    unknown_item = unknown["invalid_attempts"][0]
    for key in (
        "raw_review",
        "raw_review_sha256",
        "pre_normalization_review_sha256",
        "normalization_provenance",
    ):
        unknown_item.pop(key)
    unknown["failure_attempts"][0]["response_sha256"] = unknown_item[
        "review_sha256"
    ]
    errors = _validate_invalid_attempt_raw_normalization_provenance(
        unknown_item,
        unknown,
        _semantic_payload(),
    )
    assert errors == [
        "invalid attempt runner is not authorized to omit raw normalization provenance"
    ]
    assert not _complete_terminal_semantic_attempt(
        unknown_item,
        unknown,
        _semantic_payload(),
    )


def test_retry_budget_stops_after_three_incomplete_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_retry_sequence(
        monkeypatch, ["transport", "transport", "transport"]
    )
    assert observed == ["transport", "transport", "transport"]
    assert result["status"] == "failed"
    assert result["retry_accounting"] == {
        "policy": "bounded-transport-semantic-failure-quotas-v1",
        "transport_failure_quota": 3,
        "semantic_failure_quota": 3,
        "hard_total_attempt_quota": 6,
        "total_attempts": 3,
        "transport_failure_count": 3,
        "semantic_failure_count": 0,
        "quota_exhausted": "transport",
    }
    assert {item["failure_category"] for item in result["failure_attempts"]} == {
        "transport"
    }
    assert all(item["response_sha256"] is None for item in result["failure_attempts"])


def test_retry_budget_allows_transport_failures_before_three_semantic_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_retry_sequence(
        monkeypatch,
        ["transport", "transport", "semantic", "semantic", "semantic"],
    )
    assert observed == ["transport", "transport", "semantic", "semantic", "semantic"]
    assert result["status"] == "failed"
    assert result["retry_accounting"]["total_attempts"] == 5
    assert result["retry_accounting"]["transport_failure_count"] == 2
    assert result["retry_accounting"]["semantic_failure_count"] == 3
    assert result["retry_accounting"]["quota_exhausted"] == "semantic"
    assert result["usage_is_lower_bound"] is True
    assert len(result["invalid_attempts"]) == 3
    semantic_failures = [
        item
        for item in result["failure_attempts"]
        if item["failure_category"] == "semantic"
    ]
    assert all(
        item["response_hash_status"] == "complete_json_sha256"
        and len(item["response_sha256"]) == 64
        for item in semantic_failures
    )
    assert result["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 60,
        "total_tokens": 90,
    }
    assert [item["attempt"] for item in result["failure_attempts"]] == [1, 2, 3, 4, 5]
    assert [item["failure_category"] for item in result["failure_attempts"]] == [
        "transport",
        "transport",
        "semantic",
        "semantic",
        "semantic",
    ]
    assert [item["attempt"] for item in result["attempt_usage"]] == [3, 4, 5]
    assert [item["response_hash_status"] for item in result["failure_attempts"]] == [
        "not_available_incomplete_transport_or_parser_failure",
        "not_available_incomplete_transport_or_parser_failure",
        "complete_json_sha256",
        "complete_json_sha256",
        "complete_json_sha256",
    ]


def test_retry_budget_semantic_first_mirror_preserves_attempt_and_usage_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_retry_sequence(
        monkeypatch,
        ["semantic", "semantic", "transport", "transport", "transport"],
    )
    assert observed == ["semantic", "semantic", "transport", "transport", "transport"]
    assert result["status"] == "failed"
    assert result["retry_accounting"]["transport_failure_count"] == 3
    assert result["retry_accounting"]["semantic_failure_count"] == 2
    assert result["retry_accounting"]["quota_exhausted"] == "transport"
    assert [item["attempt"] for item in result["failure_attempts"]] == [1, 2, 3, 4, 5]
    assert [item["failure_category"] for item in result["failure_attempts"]] == [
        "semantic",
        "semantic",
        "transport",
        "transport",
        "transport",
    ]
    assert [item["attempt"] for item in result["attempt_usage"]] == [1, 2]
    assert result["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 40,
        "total_tokens": 60,
    }
    assert result["usage_is_lower_bound"] is True


def test_retry_budget_transport_failures_do_not_consume_semantic_success_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_retry_sequence(
        monkeypatch, ["transport", "transport", "success"]
    )
    assert observed == ["transport", "transport", "success"]
    assert result["status"] == "succeeded"
    assert result["attempt"] == 3
    assert result["retry_accounting"]["transport_failure_count"] == 2
    assert result["retry_accounting"]["semantic_failure_count"] == 0
    assert result["retry_accounting"]["quota_exhausted"] is None
    assert result["usage_is_lower_bound"] is True
    assert all(
        item["failure_category"] == "transport"
        for item in result["failures_before_success"]
    )


def test_complete_terminal_usage_failure_cannot_be_counted_as_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_hash = "a" * 64

    def terminal_failure(**kwargs):
        del kwargs
        raise SSEIntegrityError(
            "complete terminal response rejected",
            audit={
                "termination": "done_sentinel",
                "finish_reason_stop_seen": True,
                "usage_integrity": {"status": "passed"},
                "response_sha256": response_hash,
            },
            validated_usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

    monkeypatch.setattr("review_contracts._call_api", terminal_failure)
    result = _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )
    assert result["status"] == "failed"
    assert result["retry_accounting"]["semantic_failure_count"] == 1
    assert result["retry_accounting"]["transport_failure_count"] == 0
    assert result["failure_attempts"][0]["failure_category"] == "semantic"
    assert result["failure_attempts"][0]["response_sha256"] == response_hash
    assert result["usage"]["total_tokens"] == 30
    assert result["usage_is_lower_bound"] is False


def test_retry_budget_hard_total_never_exceeds_six_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_retry_sequence(
        monkeypatch,
        ["transport", "semantic", "transport", "semantic", "transport", "semantic"],
    )
    # A category quota is necessarily reached by the fifth two-category
    # failure, so the independent hard cap of six remains a strict upper bound.
    assert len(observed) == 5
    assert result["status"] == "failed"
    assert result["retry_accounting"]["hard_total_attempt_quota"] == 6
    assert result["retry_accounting"]["total_attempts"] == 5
    assert result["retry_accounting"]["transport_failure_count"] == 3
    assert result["retry_accounting"]["semantic_failure_count"] == 2
    assert result["retry_accounting"]["quota_exhausted"] == "transport"


def test_retry_budget_rejects_category_quota_above_three() -> None:
    with pytest.raises(ValueError, match="max_attempts must be in"):
        _execute_one(
            row=_request_row(),
            endpoint="https://example.invalid/chat/completions",
            api_key="masked",
            model="test-model",
            timeout=10,
            max_attempts=4,
            retry_delay=0,
            calibration=False,
        )


def test_eof_integrity_runner_uses_four_fresh_calibration_ids() -> None:
    template = {
        "prompt_sha256": "synthetic-prompt",
        "request": {
            "model": "prepared-model",
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "{}"},
            ],
        },
    }

    custom_ids = [row["custom_id"] for row in _calibration_requests(template)]

    assert len(custom_ids) == len(set(custom_ids)) == 4
    assert all(
        custom_id.startswith(f"calibration-{RUNNER_VERSION}-")
        for custom_id in custom_ids
    )


def test_normal_eof_without_done_is_accepted_only_after_complete_terminal_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = json.dumps(_valid_review(), ensure_ascii=False, separators=(",", ":"))
    result = _run(
        monkeypatch,
        [
            _event(content=serialized[:97]),
            _event(content=serialized[97:]),
            _event(finish_reason="stop"),
            _usage_event(),
        ],
    )

    assert result["status"] == "succeeded"
    assert result["review"] == _valid_review()
    assert result["successful_response_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    assert result["runner_version"] == RUNNER_VERSION
    assert result["transport"] == "openai-compatible-sse-v4-strict-terminal-schema"
    assert RUNNER_VERSION == result["runner_version"]
    assert TRANSPORT == result["transport"]
    assert result["sse_completion_policy"] == SSE_COMPLETION_POLICY
    assert result["sse_integrity"] == {
        "completion_policy": "strict-terminal-usage-complete-json-v2",
        "termination": "normal_eof",
        "done_seen": False,
        "finish_reason": "stop",
        "finish_reasons": ["stop"],
        "finish_reason_stop_seen": True,
        "event_count": 4,
        "content_event_count": 2,
        "content_chars": len(serialized),
        "content_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "event_integrity": "passed",
        "usage_integrity": {
            "status": "passed",
            "usage_event_count": 1,
            "required_token_fields_valid": True,
            "total_matches_prompt_plus_completion": True,
            "at_or_after_finish_reason_stop": True,
        },
        "review_validation": "passed",
        "attempt": 1,
    }


def test_nonterminal_null_usage_fields_are_protocol_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = json.dumps(_valid_review(), ensure_ascii=False, separators=(",", ":"))
    content_event = _event(content=serialized)
    content_event["usage"] = None
    stop_event = _event(finish_reason="stop")
    stop_event["usage"] = None

    result = _run(monkeypatch, [content_event, stop_event, _usage_event()])

    assert result["status"] == "succeeded"
    assert result["sse_integrity"]["usage_integrity"]["usage_event_count"] == 1
    assert result["successful_response_usage"]["total_tokens"] == 30


def test_done_stream_accepts_role_and_content_null_usage_before_terminal_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = json.dumps(_valid_review(), ensure_ascii=False, separators=(",", ":"))
    role_event = {
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
        "usage": None,
    }
    content_event = _event(content=serialized)
    content_event["usage"] = None
    stop_event = _event(finish_reason="stop")
    stop_event["usage"] = None

    result = _run(
        monkeypatch,
        [role_event, content_event, stop_event, _usage_event()],
        done=True,
    )

    assert result["status"] == "succeeded"
    assert result["sse_integrity"]["done_seen"] is True
    assert result["sse_integrity"]["usage_integrity"]["status"] == "passed"


@pytest.mark.parametrize(
    "events,done",
    [
        ([{"choices": [], "usage": None}], False),
        (
            [
                _event(content=json.dumps(_valid_review(), ensure_ascii=False)),
                _event(finish_reason="stop"),
                _usage_event(),
                _usage_event(),
            ],
            True,
        ),
        (
            [
                _event(content=json.dumps(_valid_review(), ensure_ascii=False)),
                _event(finish_reason="stop"),
                _event(finish_reason="stop"),
                _usage_event(),
            ],
            True,
        ),
        (
            [
                _event(content=json.dumps(_valid_review(), ensure_ascii=False)),
                _usage_event(),
                _event(finish_reason="stop"),
            ],
            True,
        ),
        (
            [
                _event(content=json.dumps(_valid_review(), ensure_ascii=False)),
                _event(finish_reason="stop"),
            ],
            True,
        ),
        ([{"error": {"code": "synthetic_error"}, "choices": []}], True),
    ],
)
def test_null_only_duplicate_early_missing_usage_and_error_frames_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    done: bool,
) -> None:
    result = _run(monkeypatch, events, done=done)

    assert result["status"] == "failed"
    assert "review" not in result
    assert "invalid_review" not in result


def test_provider_error_receipt_keeps_only_safe_identifiers_and_message_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "Bearer sk-test-secret api_key=private-value"
    result = _run(
        monkeypatch,
        [
            {
                "error": {
                    "code": " Internal_Error ",
                    "type": "Server_Error",
                    "status": 503,
                    "message": message,
                },
                "choices": [],
            }
        ],
        done=True,
    )

    receipt = result["sse_integrity"]["provider_error_receipt"]
    assert result["status"] == "failed"
    assert result["failure_attempts"][0]["failure_category"] == "transport"
    assert receipt == {
        "policy": "allowlisted-provider-error-receipt-v2",
        "shape": "object",
        "code": "internal_error",
        "type": "server_error",
        "status": "503",
        "message_redacted": True,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "message_byte_length": len(message.encode("utf-8")),
        "unknown_field_count": 0,
    }
    serialized = json.dumps(result, ensure_ascii=False)
    assert message not in serialized
    assert "sk-test-secret" not in serialized
    assert "private-value" not in serialized


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "bad\tcode",
        "错误代码",
        "ｅｒｒｏｒ",
        "x" * 97,
        "code with spaces",
        "synthetic_error",
        "sk-abc123XYZ789",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.synthetic-signature",  # secret-scan: allow-test
        "a" * 32,
        "b" * 64,
        "bearer-token-secret",
        "dashscope-key-0123456789abcdef",
    ],
)
def test_provider_error_receipt_never_persists_nonstandard_code(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_code: str,
) -> None:
    result = _run(
        monkeypatch,
        [{"error": {"code": unsafe_code}, "choices": []}],
    )

    receipt = result["sse_integrity"]["provider_error_receipt"]
    encoded = unsafe_code.encode("utf-8")
    assert "code" not in receipt
    assert receipt["code_redacted"] is True
    assert receipt["code_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert receipt["code_byte_length"] == len(encoded)
    assert unsafe_code not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("code", "api_error"),
        ("type", "internal_error"),
        ("type", "sk-type-secret"),
        ("status", "503"),
        ("status", "bearer-token-secret"),
    ],
)
def test_provider_error_receipt_allowlist_is_field_specific(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    unsafe_value: str,
) -> None:
    result = _run(
        monkeypatch,
        [{"error": {field: unsafe_value}, "choices": []}],
    )

    receipt = result["sse_integrity"]["provider_error_receipt"]
    encoded = unsafe_value.encode("utf-8")
    assert field not in receipt
    assert receipt[f"{field}_redacted"] is True
    assert receipt[f"{field}_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert receipt[f"{field}_byte_length"] == len(encoded)
    assert unsafe_value not in json.dumps(result, ensure_ascii=False)


def test_provider_error_receipt_hashes_malicious_tokens_and_ignores_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malicious_code = "bad\nBearer hidden-token"
    malicious_type = "x" * 200
    malicious_status = {"api_key": "nested-secret"}
    nested_message = {"details": {"token": "message-secret"}}
    result = _run(
        monkeypatch,
        [
            {
                "error": {
                    "code": malicious_code,
                    "type": malicious_type,
                    "status": malicious_status,
                    "message": nested_message,
                    "details": {"authorization": "Bearer unknown-secret"},
                },
                "choices": [],
            }
        ],
    )

    receipt = result["sse_integrity"]["provider_error_receipt"]
    assert receipt["shape"] == "object"
    assert receipt["unknown_field_count"] == 1
    for field, raw_value in (
        ("code", malicious_code),
        ("type", malicious_type),
        ("status", malicious_status),
        ("message", nested_message),
    ):
        encoded = (
            raw_value.encode("utf-8")
            if isinstance(raw_value, str)
            else json.dumps(
                raw_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        assert field not in receipt
        assert receipt[f"{field}_redacted"] is True
        assert receipt[f"{field}_sha256"] == hashlib.sha256(encoded).hexdigest()
        assert receipt[f"{field}_byte_length"] == len(encoded)
    serialized = json.dumps(result, ensure_ascii=False)
    for secret in (
        "hidden-token",
        "nested-secret",
        "message-secret",
        "unknown-secret",
        "Bearer",
        "api_key",
        "authorization",
    ):
        assert secret not in serialized


def test_provider_error_null_is_absent_but_empty_object_fails_with_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = json.dumps(_valid_review(), ensure_ascii=False, separators=(",", ":"))
    content_event = _event(content=serialized)
    content_event["error"] = None
    succeeded = _run(
        monkeypatch,
        [content_event, _event(finish_reason="stop"), _usage_event()],
        done=True,
    )
    assert succeeded["status"] == "succeeded"

    failed = _run(monkeypatch, [{"error": {}, "choices": []}], done=True)
    assert failed["status"] == "failed"
    assert failed["sse_integrity"]["provider_error_receipt"] == {
        "policy": "allowlisted-provider-error-receipt-v2",
        "shape": "object",
        "unknown_field_count": 0,
    }


def test_protocol_error_termination_and_done_complete_json_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    null_tail = _run(monkeypatch, [{"choices": [], "usage": None}])
    assert null_tail["failure_termination"] == "protocol_error"
    assert null_tail["sse_integrity"]["termination"] == "protocol_error"

    serialized = json.dumps(_valid_review(), ensure_ascii=False, separators=(",", ":"))
    trailing_json = _run(
        monkeypatch,
        [
            _event(content=serialized + "{}"),
            _event(finish_reason="stop"),
            _usage_event(),
        ],
        done=True,
    )
    assert trailing_json["status"] == "failed"
    assert "complete parseable JSON" in trailing_json["error"]
    assert trailing_json["retry_accounting"]["semantic_failure_count"] == 1
    assert trailing_json["retry_accounting"]["transport_failure_count"] == 0
    assert trailing_json["failure_attempts"][0]["failure_category"] == "semantic"
    assert trailing_json["failure_attempts"][0]["response_hash_status"] == (
        "terminal_stream_content_sha256"
    )


def test_http_error_body_is_never_read_or_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_body = b"provider body containing SECRET_TOKEN and request payload"

    def raise_http_error(_request, timeout):
        del timeout
        raise urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "provider-specific reason",
            {},
            io.BytesIO(secret_body),
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    result = _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )

    serialized_result = json.dumps(result, ensure_ascii=False)
    assert result["status"] == "failed"
    assert result["failure_termination"] == "http_error"
    assert result["failure_attempts"][-1]["http_status"] == 429
    assert "SECRET_TOKEN" not in serialized_result
    assert "provider-specific reason" not in serialized_result


@pytest.mark.parametrize(
    "transport_error",
    [
        ConnectionResetError("connection reset"),
        http.client.RemoteDisconnected("remote disconnected"),
        ssl.SSLError("TLS stream failed"),
        OSError("socket read failed"),
    ],
)
def test_sse_read_os_errors_are_audited_and_consume_transport_quota(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: OSError,
) -> None:
    class InterruptedResponse:
        headers = {"Content-Type": "text/event-stream; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            yield f"data: {json.dumps(_event(content='{'))}\n".encode("utf-8")
            raise transport_error

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: InterruptedResponse(),
    )
    result = _execute_one(
        row=_request_row(),
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )
    assert result["status"] == "failed"
    assert result["failure_termination"] == "transport_exception"
    assert result["retry_accounting"]["transport_failure_count"] == 1
    assert result["retry_accounting"]["semantic_failure_count"] == 0
    assert result["usage_is_lower_bound"] is True
    failure = result["failure_attempts"][0]
    assert failure["failure_category"] == "transport"
    assert failure["response_hash_status"] == "partial_stream_content_sha256"
    assert len(failure["response_sha256"]) == 64
    assert result["sse_integrity"]["content_chars"] == 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"unexpected_semantic_fragment": "按钮"}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"canonical_text": ""}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"any_of": ["", "USB"]}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"any_of": ["USB", "usb"]}
        ),
        lambda review: review["required_branches"][0].pop("required_concept_keys"),
        lambda review: review["required_branches"][0].update(
            {"required_concept_keys": []}
        ),
        lambda review: review["evidence_concept_coverage"][0].update(
            {"status": "ambiguous"}
        ),
        lambda review: review["critical_constraints"]["numerics"].append(
            {"value": "10", "evidence_labels": ["E1"]}
        ),
        lambda review: review["critical_constraints"]["identifiers"][0].update(
            {"evidence_labels": "E1"}
        ),
        lambda review: review["critical_constraints"]["polarity"].update(
            {"required": "free_form_unknown"}
        ),
        lambda review: review["required_branches"].append(
            deepcopy(review["required_branches"][0])
        ),
        lambda review: review["evidence_concept_coverage"].append(
            deepcopy(review["evidence_concept_coverage"][0])
        ),
    ],
)
def test_strict_recursive_review_schema_rejects_nested_defects(mutator) -> None:
    review = _valid_review()
    mutator(review)

    assert _validate_review(review)
    if "unexpected_semantic_fragment" in json.dumps(review, ensure_ascii=False):
        assert _review_schema_extra_paths(review)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["required_branches"][0].update(
            {"query_span": "默认接口"}
        ),
        lambda review: review.update({"required_branches": []}),
        lambda review: review["required_branches"][0].update(
            {"required_concept_keys": []}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"branch_labels": []}
        ),
        lambda review: review["evidence_concept_coverage"][0].update(
            {"required_concept_keys": []}
        ),
        lambda review: review.update({"evidence_concept_coverage": []}),
    ],
)
def test_payload_alignment_requires_exact_branches_and_symmetric_edges(mutator) -> None:
    review = _valid_review()
    mutator(review)

    assert _validate_against_payload(review, _semantic_payload())


@pytest.mark.parametrize(
    "alias", ["not_required", "not_applicable", "not_needed", "not_covered"]
)
def test_complete_api_response_canonicalizes_only_zero_support_status_alias(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    payload = _semantic_payload()
    payload["evidence"].append(
        {"evidence_label": "E2", "page": 2, "text": "仅为相关图像。"}
    )
    review = _valid_review()
    review["evidence_concept_coverage"].append(
        {
            "evidence_label": "E2",
            "required_concept_keys": [],
            "status": alias,
        }
    )
    serialized = json.dumps(review, ensure_ascii=False, separators=(",", ":"))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: _FakeResponse(
            [
                _event(content=serialized),
                _event(finish_reason="stop"),
                _usage_event(),
            ],
            done=True,
        ),
    )
    row = _request_row()
    row["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )

    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=1,
        retry_delay=0,
        calibration=False,
    )

    assert result["status"] == "succeeded"
    assert result["attempt"] == 1
    assert result["review"]["evidence_concept_coverage"][1]["status"] == "uncovered"
    audit = result["same_response_coverage_status_canonicalization"]
    assert audit["api_call_performed"] is True
    assert audit["source"]["source_attempt"] == 1
    assert _validate_coverage_status_canonicalization_audit(result, payload) == []
    assert "invalid_attempts_before_success" not in result


def test_live_unsupported_concept_does_not_block_before_later_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _semantic_payload()
    payload["branches"].append(
        {"branch_label": "B2", "query_span": "还会造成什么后果"}
    )
    review = _valid_review()
    review["verdict"] = "revise"
    review["critical_constraints"]["identifiers"] = []
    review["required_concepts"]["all_of"].append(
        {
            "concept_key": "c02",
            "canonical_text": "未检查前置条件会导致无法正常打印",
            "any_of": ["未检查前置条件会导致无法正常打印"],
            "evidence_labels": [],
            "branch_labels": ["B2"],
        }
    )
    review["required_branches"].append(
        {
            "branch_label": "B2",
            "query_span": "还会造成什么后果",
            "required_concept_keys": ["c02"],
        }
    )
    review["review_notes"] = ["B2 的后果结论没有直接证据支持，必须进入数据修复。"]
    valid_review = deepcopy(review)
    valid_review["required_concepts"]["all_of"][1]["evidence_labels"] = ["E1"]
    valid_review["evidence_concept_coverage"][0]["required_concept_keys"].append(
        "c02"
    )
    valid_review["review_notes"] = ["E1 直接支持两个回答分支。"]
    serialized_reviews = iter(
        [
            json.dumps(review, ensure_ascii=False, separators=(",", ":")),
            json.dumps(valid_review, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: _FakeResponse(
            [
                _event(content=next(serialized_reviews)),
                _event(finish_reason="stop"),
                _usage_event(),
            ],
            done=True,
        ),
    )
    row = _request_row()
    row["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )

    result = _execute_one(
        row=row,
        endpoint="https://example.invalid/chat/completions",
        api_key="masked",
        model="test-model",
        timeout=10,
        max_attempts=2,
        retry_delay=0,
        calibration=False,
    )

    assert result["status"] == "succeeded"
    assert result["attempt"] == 2
    assert result["review"] == valid_review
    assert result["review"]["verdict"] == "revise"
    assert "same_response_unsupported_required_concept_block" not in result
    assert len(result["invalid_attempts_before_success"]) == 1
    assert result["invalid_attempts_before_success"][0]["review"] == review


def _inverse_edge_payload_and_review() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _semantic_payload()
    payload["branches"] = [
        {"branch_label": "B1", "query_span": "默认接口是什么"},
        {"branch_label": "B2", "query_span": "请明确说明"},
    ]
    payload["evidence"].append(
        {"evidence_label": "E2", "page": 2, "text": "USB 是默认接口。"}
    )
    review = _valid_review()
    review["required_branches"].append(
        {
            "branch_label": "B2",
            "query_span": "请明确说明",
            "required_concept_keys": ["c01"],
        }
    )
    review["evidence_concept_coverage"].append(
        {
            "evidence_label": "E2",
            "required_concept_keys": ["c01"],
            "status": "covered",
        }
    )
    return payload, review


def test_same_response_inverse_edge_canonicalization_uses_only_authoritative_edges() -> None:
    payload, review = _inverse_edge_payload_and_review()

    projected, actions, raw_errors = _canonicalize_same_response_inverse_edges(
        review, payload
    )

    assert set(raw_errors) == {
        "concept.branch_labels and required_branches edges must be symmetric",
        "concept.evidence_labels and evidence_concept_coverage edges must be symmetric",
    }
    assert projected["required_concepts"]["all_of"][0]["branch_labels"] == [
        "B1",
        "B2",
    ]
    assert projected["required_concepts"]["all_of"][0]["evidence_labels"] == [
        "E1",
        "E2",
    ]
    assert {action["authority"] for action in actions} == {
        "required_branches",
        "evidence_concept_coverage",
    }
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["required_branches"][1].update(
            {"required_concept_keys": []}
        ),
        lambda review: review["required_branches"][1].update(
            {"required_concept_keys": ["c99"]}
        ),
        lambda review: review["required_branches"][1].update(
            {"query_span": "漂移的span"}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"schema_extra": "semantic text"}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"canonical_text": ""}
        ),
        lambda review: review["evidence_concept_coverage"][1].update(
            {"required_concept_keys": []}
        ),
    ],
)
def test_inverse_edge_canonicalization_rejects_any_other_defect(mutator) -> None:
    payload, review = _inverse_edge_payload_and_review()
    mutator(review)

    projected, actions, raw_errors = _canonicalize_same_response_inverse_edges(
        review, payload
    )

    assert projected == review
    assert actions == []
    assert raw_errors


def test_saved_same_response_inverse_edges_recover_without_api_or_candidate_hydration() -> None:
    payload, review = _inverse_edge_payload_and_review()
    row = _request_row()
    row["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    raw_hash = hashlib.sha256(
        json.dumps(
            review, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    failed_row = {
        "custom_id": row["custom_id"],
        "status": "failed",
        "prompt_sha256": row["prompt_sha256"],
        "attempt": 1,
        "model": "test-model",
        "runner_version": "2.8.0-sse-strict-terminal-schema",
        "completed_at": "2026-08-11T00:00:00+00:00",
        "invalid_attempts": [
            {
                "attempt": 1,
                "review": review,
                "review_sha256": raw_hash,
                "validation_errors": [
                    "concept.branch_labels and required_branches edges must be symmetric",
                    "concept.evidence_labels and evidence_concept_coverage edges must be symmetric",
                ],
            }
        ],
    }

    recovered = _recover_prior_invalid_review(row=row, response_rows=[failed_row])

    assert recovered is not None
    assert recovered["status"] == "succeeded"
    assert recovered["attempt"] == 0
    assert recovered["usage"]["total_tokens"] == 0
    audit = recovered["same_response_inverse_edge_canonicalization"]
    assert audit["api_call_performed"] is False
    assert audit["raw_review_sha256"] == raw_hash
    assert audit["projected_review_sha256"] != raw_hash
    assert recovered["prior_response_recovery"]["mode"] == (
        "same_response_inverse_edge_canonicalization"
    )
    assert _validate_against_payload(recovered["review"], payload) == []
    assert _validate_inverse_edge_canonicalization_audit(
        recovered, payload, source_response_rows=[failed_row]
    ) == []
    assert _validate_inverse_edge_canonicalization_audit(recovered, payload)
    assert _validate_inverse_edge_canonicalization_audit(
        recovered,
        payload,
        source_response_rows=[failed_row, deepcopy(failed_row)],
    )

    tampered = deepcopy(recovered)
    tampered["same_response_inverse_edge_canonicalization"]["actions"][0][
        "before"
    ] = ["B9"]
    assert _validate_inverse_edge_canonicalization_audit(
        tampered, payload, source_response_rows=[failed_row]
    )
    tampered_source = deepcopy(failed_row)
    tampered_source["prompt_sha256"] = "different"
    assert _validate_inverse_edge_canonicalization_audit(
        recovered, payload, source_response_rows=[tampered_source]
    )
    unknown_runner_source = deepcopy(failed_row)
    unknown_runner_source["runner_version"] = "unknown-runner"
    forged_legacy = deepcopy(recovered)
    forged_legacy["runner_version"] = "2.9.0-same-response-inverse-edge"
    forged_legacy_source = forged_legacy[
        "same_response_inverse_edge_canonicalization"
    ]["source"]
    for key in ("custom_id", "prompt_sha256", "runner_version"):
        forged_legacy_source.pop(key)
    forged_legacy_source["response_row_sha256"] = _sha256_json(
        unknown_runner_source
    )
    assert any(
        "source row lineage" in error
        for error in _validate_inverse_edge_canonicalization_audit(
            forged_legacy,
            payload,
            source_response_rows=[unknown_runner_source],
        )
    )
    structural_source = deepcopy(failed_row)
    structural_source["invalid_attempts"][0]["validation_errors"].append(
        "strict runner rejects structural normalization: 1 action(s)"
    )
    forged = deepcopy(recovered)
    forged_source = forged["same_response_inverse_edge_canonicalization"]["source"]
    forged_source["response_row_sha256"] = _sha256_json(structural_source)
    forged_source["validation_errors"] = structural_source["invalid_attempts"][0][
        "validation_errors"
    ]
    assert any(
        "structural normalization" in error
        for error in _validate_inverse_edge_canonicalization_audit(
            forged, payload, source_response_rows=[structural_source]
        )
    )


def _duplicate_variant_review() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _semantic_payload()
    review = _valid_review()
    review["required_concepts"]["all_of"][0]["any_of"] = [
        "50HZ/60HZ(±2%)",
        "50Hz/60Hz(±2%)",
        "50HZ/60HZ ±2%",
        "50HZ/60HZ(±2%",
    ]
    return payload, review


def test_same_response_exact_variant_dedupe_preserves_nonidentical_variants() -> None:
    payload, review = _duplicate_variant_review()

    projected, actions, raw_errors = _dedupe_same_response_exact_variants(
        review, payload
    )

    assert raw_errors == ["concept[0].any_of entries must be unique"]
    assert projected["required_concepts"]["all_of"][0]["any_of"] == [
        "50HZ/60HZ(±2%)",
        "50HZ/60HZ ±2%",
        "50HZ/60HZ(±2%",
    ]
    assert len(actions) == 1
    assert actions[0]["removed"] == [
        {
            "index": 1,
            "value": "50Hz/60Hz(±2%)",
            "normalized": "50hz/60hz(±2%)",
            "retained_index": 0,
            "retained_value": "50HZ/60HZ(±2%)",
        }
    ]
    assert _validate_review(projected) == []
    assert _validate_against_payload(projected, payload) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda review: review["required_branches"][0].update(
            {"query_span": "漂移的span"}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"schema_extra": "semantic text"}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"canonical_text": ""}
        ),
        lambda review: review["required_concepts"]["all_of"][0].update(
            {"any_of": ["50HZ/60HZ(±2%)", "not equivalent"]}
        ),
    ],
)
def test_exact_variant_dedupe_rejects_other_defects_or_nonduplicates(mutator) -> None:
    payload, review = _duplicate_variant_review()
    mutator(review)

    projected, actions, _raw_errors = _dedupe_same_response_exact_variants(
        review, payload
    )

    assert projected == review
    assert actions == []


def test_saved_exact_variant_duplicate_recovers_with_replayable_audit() -> None:
    payload, review = _duplicate_variant_review()
    row = _request_row()
    row["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    raw_hash = hashlib.sha256(
        json.dumps(
            review, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    failed_row = {
        "custom_id": row["custom_id"],
        "status": "failed",
        "prompt_sha256": row["prompt_sha256"],
        "attempt": 1,
        "model": "test-model",
        "runner_version": "2.9.0-same-response-inverse-edge",
        "completed_at": "2026-08-11T00:00:00+00:00",
        "invalid_attempts": [
            {
                "attempt": 1,
                "review": review,
                "review_sha256": raw_hash,
                "validation_errors": [
                    "concept[0].any_of entries must be unique"
                ],
            }
        ],
    }

    recovered = _recover_prior_invalid_review(row=row, response_rows=[failed_row])

    assert recovered is not None
    assert recovered["attempt"] == 0
    assert recovered["prior_response_recovery"]["mode"] == (
        "same_response_exact_variant_deduplication"
    )
    audit = recovered["same_response_exact_variant_deduplication"]
    assert audit["api_call_performed"] is False
    assert audit["raw_review_sha256"] == raw_hash
    assert _validate_exact_variant_deduplication_audit(
        recovered, payload, source_response_rows=[failed_row]
    ) == []
    assert _validate_exact_variant_deduplication_audit(recovered, payload)
    assert _validate_exact_variant_deduplication_audit(
        recovered,
        payload,
        source_response_rows=[failed_row, deepcopy(failed_row)],
    )
    tampered = deepcopy(recovered)
    tampered["same_response_exact_variant_deduplication"]["actions"][0][
        "after"
    ].append("invented")
    assert _validate_exact_variant_deduplication_audit(
        tampered, payload, source_response_rows=[failed_row]
    )
    tampered_source = deepcopy(failed_row)
    tampered_source["prompt_sha256"] = "different"
    assert _validate_exact_variant_deduplication_audit(
        recovered, payload, source_response_rows=[tampered_source]
    )
    unknown_runner_source = deepcopy(failed_row)
    unknown_runner_source["runner_version"] = "unknown-runner"
    forged_legacy = deepcopy(recovered)
    forged_legacy["runner_version"] = "2.10.0-same-response-exact-dedupe"
    forged_legacy_source = forged_legacy[
        "same_response_exact_variant_deduplication"
    ]["source"]
    for key in ("custom_id", "prompt_sha256", "runner_version"):
        forged_legacy_source.pop(key)
    forged_legacy_source["response_row_sha256"] = _sha256_json(
        unknown_runner_source
    )
    assert any(
        "source row lineage" in error
        for error in _validate_exact_variant_deduplication_audit(
            forged_legacy,
            payload,
            source_response_rows=[unknown_runner_source],
        )
    )
    structural_source = deepcopy(failed_row)
    structural_source["invalid_attempts"][0]["validation_errors"].append(
        "strict runner rejects structural normalization: 1 action(s)"
    )
    forged = deepcopy(recovered)
    forged_source = forged["same_response_exact_variant_deduplication"]["source"]
    forged_source["response_row_sha256"] = _sha256_json(structural_source)
    forged_source["validation_errors"] = structural_source["invalid_attempts"][0][
        "validation_errors"
    ]
    assert any(
        "structural normalization" in error
        for error in _validate_exact_variant_deduplication_audit(
            forged, payload, source_response_rows=[structural_source]
        )
    )


def test_eof_rejects_the_933_character_fragment_without_persisting_a_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = '{"verdict":"'
    fragment = prefix + "x" * (933 - len(prefix))
    assert len(fragment) == 933

    result = _run(monkeypatch, [_event(content=fragment)])

    assert result["status"] == "failed"
    assert "content_chars=933" in result["error"]
    assert "review" not in result
    assert "invalid_review" not in result
    assert result["sse_integrity"]["done_seen"] is False
    assert result["sse_integrity"]["finish_reason"] is None
    assert result["sse_integrity"]["event_count"] == 1
    assert result["sse_integrity"]["content_sha256"] == hashlib.sha256(
        fragment.encode("utf-8")
    ).hexdigest()
    assert result["failure_attempts"][0]["failure_category"] == "transport"
    assert result["failure_attempts"][0]["response_sha256"] == result[
        "sse_integrity"
    ]["content_sha256"]
    assert result["sse_integrity"]["content_chars"] == 933
    assert result["sse_integrity"]["usage_integrity"]["status"] == "missing"
    assert fragment not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    ("mutate_review", "events_suffix", "error_fragment"),
    [
        (None, [_usage_event()], "finish_reason=stop"),
        (None, [_event(finish_reason="stop")], "usage"),
    ],
)
def test_eof_without_done_fails_closed_when_any_completion_proof_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    mutate_review,
    events_suffix: list[dict[str, Any]],
    error_fragment: str,
) -> None:
    review = _valid_review()
    if mutate_review is not None:
        mutate_review(review)
    serialized = json.dumps(review, ensure_ascii=False, separators=(",", ":"))

    result = _run(monkeypatch, [_event(content=serialized), *events_suffix])

    assert result["status"] == "failed"
    assert error_fragment in result["error"]
    assert "review" not in result
    assert "invalid_review" not in result


def test_validated_stop_eof_semantic_failure_persists_raw_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_review = _valid_review()
    raw_review["verdict"] = "revise"
    concept = raw_review["required_concepts"]["all_of"][0]
    concept.pop("any_of")
    concept.pop("evidence_labels")
    concept.pop("branch_labels")
    serialized = json.dumps(raw_review, ensure_ascii=False, separators=(",", ":"))

    result = _run(
        monkeypatch,
        [
            _event(content=serialized),
            _event(finish_reason="stop"),
            _usage_event(),
        ],
    )

    assert result["status"] == "failed"
    assert result["retry_accounting"]["semantic_failure_count"] == 1
    assert result["retry_accounting"]["transport_failure_count"] == 0
    item = result["invalid_attempts"][0]
    assert item["raw_review"] == raw_review
    assert item["raw_review_sha256"] == _sha256_json(raw_review)
    assert item["sse_integrity"]["termination"] == "validated_stop_eof"
    assert item["sse_integrity"]["review_validation"] == "failed"
    assert item["normalization_provenance"]["actions"] == [
        "self_filled_any_of:c1",
        "crossref_filled_evidence_labels:c1",
        "crossref_filled_branch_labels:c1",
    ]
    assert _validate_invalid_attempt_raw_normalization_provenance(
        item,
        result,
        _semantic_payload(),
    ) == []
    assert _complete_terminal_semantic_attempt(
        item,
        result,
        _semantic_payload(),
    )
    assert result["failure_attempts"][0] == {
        "attempt": 1,
        "termination": "review_validation_error",
        "error_type": "ValueError",
        "failure_category": "semantic",
        "response_sha256": _sha256_json(raw_review),
        "response_hash_status": "complete_json_sha256",
    }
    assert _recover_prior_invalid_review(
        row=_request_row(), response_rows=[result]
    ) is None
