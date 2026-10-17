from __future__ import annotations

import json
import inspect
import sys
from collections.abc import Callable
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

from review_release_integrity import (  # noqa: E402
    EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS,
    EXPECTED_REVIEW_PROMPT_SHA256,
    EXPECTED_REREVIEW_BASE_URL,
    EXPECTED_REREVIEW_MODEL,
    bind_rereview_calibration_receipt,
    contract_rereview_custom_id,
    rereview_calibration_ledger,
    rereview_calibration_custom_id,
    rereview_execution_contract_sha256,
    sha256_file,
    sha256_json,
    validate_blocker_rereview_prepared_receipt,
    validate_block_union_freshness,
    validate_contract_rereview_prepared_receipt,
    validate_proposal_validation_binding,
    validate_request_pack_identity,
    validate_rereview_request_policy,
    validate_rereview_runtime,
    strict_success_ledger,
)
import review_blocker_rereview as blocker_review  # noqa: E402
import review_contract_rereview as contract_review  # noqa: E402
import review_contracts as review_runner  # noqa: E402
import prepare_contract_rereview as contract_prepare  # noqa: E402


def _original_requests() -> list[dict]:
    return [
        json.loads(line)
        for line in (CONTRACT_ROOT / "review" / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def test_request_identity_recomputes_prompt_payload_custom_id_and_pack_sha() -> None:
    rows = _original_requests()
    receipt = validate_request_pack_identity(
        rows,
        expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=150,
        expected_pack_sha256=sha256_json(rows),
        pack_name="test full-suite pack",
    )
    assert receipt["case_count"] == 150

    prompt_drift = deepcopy(rows)
    prompt_drift[0]["request"]["messages"][0]["content"] += "\nchanged"
    with pytest.raises(RuntimeError, match="actual prompt SHA drift"):
        validate_request_pack_identity(
            prompt_drift,
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            pack_name="prompt drift",
        )

    payload_drift = deepcopy(rows)
    payload = json.loads(payload_drift[0]["request"]["messages"][1]["content"])
    payload["query"] += " changed"
    payload_drift[0]["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    with pytest.raises(RuntimeError, match="payload SHA drift"):
        validate_request_pack_identity(
            payload_drift,
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            pack_name="payload drift",
        )

    custom_id_drift = deepcopy(rows)
    custom_id_drift[0]["custom_id"] = "concept-review-not-payload-bound"
    with pytest.raises(RuntimeError, match="not payload-bound"):
        validate_request_pack_identity(
            custom_id_drift,
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            pack_name="custom-id drift",
        )

    with pytest.raises(RuntimeError, match="request-pack SHA drift"):
        validate_request_pack_identity(
            rows,
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            expected_pack_sha256="0" * 64,
            pack_name="pack drift",
        )


def test_contract_rereview_id_is_bound_to_original_request_and_payload() -> None:
    original = deepcopy(_original_requests()[0])
    original_custom_id = original["custom_id"]
    rereview_id = contract_rereview_custom_id(
        original_custom_id, original["payload_sha256"]
    )
    original["custom_id"] = rereview_id
    validate_request_pack_identity(
        [original],
        expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=1,
        custom_id_mode="contract_rereview",
        original_custom_ids_by_rereview_id={rereview_id: original_custom_id},
        pack_name="contract rereview",
    )
    with pytest.raises(RuntimeError, match="not payload-bound"):
        validate_request_pack_identity(
            [original],
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            custom_id_mode="contract_rereview",
            original_custom_ids_by_rereview_id={rereview_id: "wrong-original-id"},
            pack_name="contract rereview drift",
        )


def test_block_union_five_hash_freshness_rejects_visual_drift(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "requests.jsonl",
            "responses.jsonl",
            "visual.json",
            "base.json",
            "supplemental.json",
        )
    }
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    audit = {
        "review": {
            "request_pack_sha256": sha256_file(paths["requests.jsonl"]),
            "responses_sha256": sha256_file(paths["responses.jsonl"]),
        },
        "inputs": {
            "visual_audit_sha256": sha256_file(paths["visual.json"]),
            "base_proposals_sha256": sha256_file(paths["base.json"]),
            "supplemental_proposals_sha256": sha256_file(
                paths["supplemental.json"]
            ),
        },
    }
    kwargs = {
        "requests_path": paths["requests.jsonl"],
        "responses_path": paths["responses.jsonl"],
        "visual_audit_path": paths["visual.json"],
        "base_proposals_path": paths["base.json"],
        "supplemental_proposals_path": paths["supplemental.json"],
    }
    validate_block_union_freshness(audit, **kwargs)
    paths["visual.json"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="visual_audit_sha256"):
        validate_block_union_freshness(audit, **kwargs)


def test_proposal_validation_receipt_rejects_proposal_drift(tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.json"
    proposals.write_text("{}", encoding="utf-8")
    validation = {"proposal_file_sha256": sha256_file(proposals)}
    validate_proposal_validation_binding(validation, proposals_path=proposals)
    proposals.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="proposal file SHA drift"):
        validate_proposal_validation_binding(validation, proposals_path=proposals)


def _blocker_rereview_receipt() -> dict:
    return {
        "schema_version": "stt.eval.suite-v2-blocker-rereview-pack.v1",
        "status": "prepared_pending_independent_same_prompt_review",
        "case_count": EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
        "difficulty_counts": {"easy": 4, "medium": 5, "hard": 6},
        "prompt_sha256": EXPECTED_REVIEW_PROMPT_SHA256,
        "request_pack_sha256": "a" * 64,
        "execution_contract_sha256": "b" * 64,
        "validated_no_index_proposal_count": (
            EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
        ),
        "supplemental_reviewer_block_proposal_count": (
            EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        ),
        "data_blocker_case_count": EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
        "index_rebuild_required": False,
        "evidence_refresh_required": False,
        "superseded_review_reused": False,
        "write_performed": True,
    }


def test_blocker_rereview_receipt_requires_exact_13_plus_2_union() -> None:
    receipt = _blocker_rereview_receipt()
    validated = validate_blocker_rereview_prepared_receipt(receipt)
    assert validated == {
        "base_proposal_count": 13,
        "supplemental_proposal_count": 2,
        "data_blocker_case_count": 15,
        "request_pack_sha256": "a" * 64,
    }

    for field, value in (
        ("validated_no_index_proposal_count", 12),
        ("supplemental_reviewer_block_proposal_count", 1),
        ("data_blocker_case_count", 14),
        ("case_count", 14),
    ):
        drifted = {**receipt, field: value}
        with pytest.raises(RuntimeError, match=r"exactly 13 base \+ 2 supplemental"):
            validate_blocker_rereview_prepared_receipt(drifted)

    bad_difficulty_counts = {
        **receipt,
        "difficulty_counts": {"easy": 5, "medium": 5, "unexpected": 5},
    }
    with pytest.raises(RuntimeError, match="difficulty counts do not total 15"):
        validate_blocker_rereview_prepared_receipt(bad_difficulty_counts)


def test_blocker_rereview_receipt_rejects_old_review_reuse_and_prompt_drift() -> None:
    receipt = _blocker_rereview_receipt()
    reused = {**receipt, "superseded_review_reused": True}
    with pytest.raises(RuntimeError, match="superseded review reuse"):
        validate_blocker_rereview_prepared_receipt(reused)

    prompt_drift = {**receipt, "prompt_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="frozen full-suite prompt"):
        validate_blocker_rereview_prepared_receipt(prompt_drift)

    execution_contract_drift = {**receipt, "execution_contract_sha256": ""}
    with pytest.raises(RuntimeError, match="invalid execution-contract SHA"):
        validate_blocker_rereview_prepared_receipt(execution_contract_drift)


def test_blocker_rereview_contract_matches_current_13_plus_2_inputs() -> None:
    proposal_root = CONTRACT_ROOT.parent / "suite-v2-blocker-remediation"
    base = json.loads(
        (proposal_root / "proposals.json").read_text(encoding="utf-8")
    )
    supplemental = json.loads(
        (CONTRACT_ROOT / "reviewer-block-supplemental-proposals.json").read_text(
            encoding="utf-8"
        )
    )
    assert EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS == len(base["proposals"]) == 13
    assert (
        EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        == len(supplemental["proposals"])
        == 2
    )
    assert EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS == 15


def test_contract_rereview_receipt_requires_written_same_semantic_pack() -> None:
    receipt = {
        "schema_version": "stt.eval.suite-v2-contract-rereview-pack.v1",
        "status": "prepared_pending_independent_same-semantic_review",
        "case_count": 2,
        "parent_case_ids": ["case-1", "case-2"],
        "semantic_payload_unchanged": True,
        "dataset_cases_changed": False,
        "new_custom_ids": True,
        "same_discriminative_variant_gate_required": True,
        "write_performed": True,
    }
    validate_contract_rereview_prepared_receipt(receipt)
    for field, value in (
        ("write_performed", False),
        ("semantic_payload_unchanged", False),
        ("dataset_cases_changed", True),
        ("new_custom_ids", False),
    ):
        with pytest.raises(RuntimeError, match="not immutable/written"):
            validate_contract_rereview_prepared_receipt({**receipt, field: value})
    with pytest.raises(RuntimeError, match="case identity is invalid"):
        validate_contract_rereview_prepared_receipt(
            {**receipt, "parent_case_ids": ["case-1", "case-1"]}
        )


def test_contract_rereview_prepare_accepts_zero_missing_case_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = {
        "status": "passed",
        "review": {
            "successful_case_count": 150,
            "missing_case_count": 0,
        },
    }
    monkeypatch.setattr(contract_prepare, "_read_json", lambda _path: audit)

    class FreshnessGateReached(Exception):
        pass

    def stop_after_review_gate(*_args: object, **_kwargs: object) -> None:
        raise FreshnessGateReached("freshness gate reached")

    monkeypatch.setattr(
        contract_prepare,
        "validate_block_union_freshness",
        stop_after_review_gate,
    )

    with pytest.raises(FreshnessGateReached, match="freshness gate reached"):
        contract_prepare.prepare(write=False)


def _write_changed_pack(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True)
    (root / "requests.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        **_blocker_rereview_receipt(),
        "request_pack_sha256": sha256_json(rows),
        "execution_contract_sha256": rereview_execution_contract_sha256(
            runner_version=blocker_review.runner.RUNNER_VERSION,
            transport=blocker_review.runner.TRANSPORT,
            sse_completion_policy=blocker_review.runner.SSE_COMPLETION_POLICY,
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )


def test_changed_pack_keeps_prompt_but_requires_new_payload_bound_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = deepcopy(_original_requests()[:15])
    fresh = deepcopy(original)
    for index, row in enumerate(fresh):
        payload = json.loads(row["request"]["messages"][1]["content"])
        payload["changed_case_contract_test"] = index
        payload_sha256 = sha256_json(payload)
        row["request"]["messages"][1]["content"] = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        row["payload_sha256"] = payload_sha256
        row["custom_id"] = f"concept-review-{payload_sha256[:20]}"

    rereview_root = tmp_path / "fresh"
    _write_changed_pack(rereview_root, fresh)
    monkeypatch.setattr(blocker_review, "REREVIEW_ROOT", rereview_root)
    monkeypatch.setattr(blocker_review, "REQUESTS_PATH", rereview_root / "requests.jsonl")
    monkeypatch.setattr(blocker_review, "_validate_upstream_freshness", lambda _value: None)
    requests, count = blocker_review._validate_requests()
    assert count == len(requests) == 15
    assert {row["prompt_sha256"] for row in requests} == {
        EXPECTED_REVIEW_PROMPT_SHA256
    }
    assert not ({row["custom_id"] for row in requests} & {row["custom_id"] for row in original})

    colliding_root = tmp_path / "colliding"
    _write_changed_pack(colliding_root, original)
    monkeypatch.setattr(blocker_review, "REREVIEW_ROOT", colliding_root)
    monkeypatch.setattr(blocker_review, "REQUESTS_PATH", colliding_root / "requests.jsonl")
    with pytest.raises(RuntimeError, match="collide with superseded review ids"):
        blocker_review._validate_requests()


def test_paid_changed_review_fails_before_lock_when_original_150_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(blocker_review, "_validate_requests", lambda: ([], 15))
    monkeypatch.setattr(blocker_review, "_original_review_complete", lambda: False)
    monkeypatch.setattr(
        blocker_review,
        "validate_rereview_request_policy",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        blocker_review,
        "_read_json",
        lambda _path: {"execution_contract_sha256": "a" * 64},
    )

    def unexpected_lock(**_: object) -> None:
        raise AssertionError("execution lock must not be reached")

    monkeypatch.setattr(blocker_review.runner, "_acquire_execution_lock", unexpected_lock)
    with pytest.raises(SystemExit, match="original 150/150 review is complete"):
        blocker_review.main(
            [
                "--execute",
                "--acknowledge-cost",
                "--confirm-full-review",
                "--confirm-no-concurrent-review",
            ]
        )


def test_original_150_gate_rejects_duplicate_request_ids_and_external_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = deepcopy(_original_requests())
    duplicated = deepcopy(requests)
    duplicated[-1] = deepcopy(duplicated[0])

    monkeypatch.setattr(
        blocker_review.runner,
        "_read_jsonl",
        lambda path: duplicated if path == blocker_review.runner.REQUESTS_PATH else [],
    )
    assert blocker_review._original_review_complete() is False

    responses = [
        {
            "custom_id": str(row["custom_id"]),
            "status": "succeeded",
            "prompt_sha256": EXPECTED_REVIEW_PROMPT_SHA256,
            "review": {},
        }
        for row in requests
    ]
    responses.append(
        {
            "custom_id": "external-success",
            "status": "succeeded",
            "prompt_sha256": EXPECTED_REVIEW_PROMPT_SHA256,
            "review": {},
        }
    )
    monkeypatch.setattr(
        blocker_review.runner,
        "_read_jsonl",
        lambda path: requests
        if path == blocker_review.runner.REQUESTS_PATH
        else responses,
    )
    assert blocker_review._original_review_complete() is False


def test_rereview_execution_contract_freezes_model_endpoint_and_json_object() -> None:
    rows = deepcopy(_original_requests()[:2])
    digest = validate_rereview_request_policy(
        rows,
        runner_version="2.26.0-test",
        transport="openai-compatible-sse-v4-strict-terminal-schema",
        sse_completion_policy="strict-terminal-usage-complete-json-v2",
    )
    assert digest == rereview_execution_contract_sha256(
        runner_version="2.26.0-test",
        transport="openai-compatible-sse-v4-strict-terminal-schema",
        sse_completion_policy="strict-terminal-usage-complete-json-v2",
    )
    validate_rereview_runtime(
        model=EXPECTED_REREVIEW_MODEL,
        base_url=EXPECTED_REREVIEW_BASE_URL,
        execution_contract_sha256=digest,
        expected_execution_contract_sha256=digest,
    )

    for mutator in (
        lambda request: request.update(model="other-model"),
        lambda request: request.update(temperature=0.1),
        lambda request: request.update(response_format={"type": "json_schema"}),
        lambda request: request.update(max_tokens=100),
    ):
        drifted = deepcopy(rows)
        mutator(drifted[0]["request"])
        with pytest.raises(RuntimeError, match="frozen rereview request policy"):
            validate_rereview_request_policy(
                drifted,
                runner_version="2.26.0-test",
                transport="openai-compatible-sse-v4-strict-terminal-schema",
                sse_completion_policy="strict-terminal-usage-complete-json-v2",
            )

    with pytest.raises(RuntimeError, match="runtime model/base URL drift"):
        validate_rereview_runtime(
            model="other-model",
            base_url=EXPECTED_REREVIEW_BASE_URL,
            execution_contract_sha256=digest,
            expected_execution_contract_sha256=digest,
        )


def test_rereview_calibration_and_resume_are_execution_contract_bound() -> None:
    execution_contract = "a" * 64
    calibration_id = rereview_calibration_custom_id(
        "calibration-2.26.0-test-numeric", execution_contract
    )
    assert execution_contract[:16] in calibration_id
    success = {
        "custom_id": "case-1",
        "status": "succeeded",
        "prompt_sha256": EXPECTED_REVIEW_PROMPT_SHA256,
        "model": EXPECTED_REREVIEW_MODEL,
        "runner_version": "2.26.0-test",
        "transport": "openai-compatible-sse-v4-strict-terminal-schema",
        "execution_contract_sha256": execution_contract,
        "review": {"verdict": "approve"},
    }
    ledger = strict_success_ledger(
        [success],
        expected_custom_ids={"case-1"},
        execution_contract_sha256=execution_contract,
        expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
        expected_model=EXPECTED_REREVIEW_MODEL,
        expected_runner_version="2.26.0-test",
        expected_transport="openai-compatible-sse-v4-strict-terminal-schema",
        label="test resume",
    )
    assert set(ledger) == {"case-1"}

    for rows in (
        [{**success, "custom_id": "pack-external"}],
        [success, deepcopy(success)],
        [{**success, "execution_contract_sha256": "b" * 64}],
        [{**success, "model": "other-model"}],
    ):
        with pytest.raises(RuntimeError):
            strict_success_ledger(
                rows,
                expected_custom_ids={"case-1"},
                execution_contract_sha256=execution_contract,
                expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
                expected_model=EXPECTED_REREVIEW_MODEL,
                expected_runner_version="2.26.0-test",
                expected_transport="openai-compatible-sse-v4-strict-terminal-schema",
                label="tampered resume",
            )


def _current_runner_calibrations() -> tuple[list[dict], dict[str, dict]]:
    requests = review_runner._calibration_requests(
        _original_requests()[0],
        model=review_runner.DEFAULT_MODEL,
        endpoint=review_runner._endpoint(review_runner.DEFAULT_BASE_URL),
    )
    completed = review_runner._completed_calibrations_by_id(
        review_runner.CALIBRATION_RESPONSES_PATH,
        requests,
    )
    assert len(requests) == len(completed) == 4
    return requests, completed


def test_current_runner_calibration_envelopes_cover_raw_and_projectable() -> None:
    requests, completed = _current_runner_calibrations()
    request_by_id = {str(row["custom_id"]): row for row in requests}
    statuses = [str(row["status"]) for row in completed.values()]
    assert statuses.count("succeeded") == 3
    assert statuses.count("calibration_projectable_pass") == 1
    for custom_id, result in completed.items():
        assert review_runner._validate_calibration_result(
            result,
            request_by_id[custom_id],
        ) == []

        extra_field = {**deepcopy(result), "execution_contract_sha256": "a" * 64}
        assert any(
            "schema is invalid" in error
            for error in review_runner._validate_calibration_result(
                extra_field,
                request_by_id[custom_id],
            )
        )

        identity_tamper = {**deepcopy(result), "model": "other-model"}
        assert review_runner._validate_calibration_result(
            identity_tamper,
            request_by_id[custom_id],
        )


def test_rereview_calibration_ledger_accepts_raw_and_projectable_with_outer_binding() -> None:
    requests, completed = _current_runner_calibrations()
    execution_contract = "a" * 64
    persisted = [
        bind_rereview_calibration_receipt(
            result,
            execution_contract_sha256=execution_contract,
        )
        for result in completed.values()
    ]
    replayed = rereview_calibration_ledger(
        persisted,
        requests=requests,
        execution_contract_sha256=execution_contract,
        validate_core_result=review_runner._validate_calibration_result,
    )

    assert len(replayed) == 4
    assert sorted(row["status"] for row in replayed.values()) == [
        "calibration_projectable_pass",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert all(
        row["execution_contract_sha256"] == execution_contract
        for row in replayed.values()
    )


def test_rereview_calibration_ledger_latest_invalid_receipt_requires_rerun() -> None:
    requests, completed = _current_runner_calibrations()
    execution_contract = "a" * 64
    rows = [
        bind_rereview_calibration_receipt(
            completed[str(request["custom_id"])],
            execution_contract_sha256=execution_contract,
        )
        for request in requests
    ]
    target_id = str(requests[0]["custom_id"])
    invalid_latest = bind_rereview_calibration_receipt(
        {**deepcopy(completed[target_id]), "model": "other-model"},
        execution_contract_sha256=execution_contract,
    )
    replayed = rereview_calibration_ledger(
        [*rows, invalid_latest],
        requests=requests,
        execution_contract_sha256=execution_contract,
        validate_core_result=review_runner._validate_calibration_result,
    )
    assert target_id not in replayed
    assert len(replayed) == 3

    valid_rerun = bind_rereview_calibration_receipt(
        completed[target_id],
        execution_contract_sha256=execution_contract,
    )
    replayed = rereview_calibration_ledger(
        [*rows, invalid_latest, valid_rerun],
        requests=requests,
        execution_contract_sha256=execution_contract,
        validate_core_result=review_runner._validate_calibration_result,
    )
    assert set(replayed) == {str(row["custom_id"]) for row in requests}


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.update(unexpected_core_field=True),
        lambda row: row.update(model="other-model"),
    ],
)
def test_rereview_calibration_ledger_rejects_core_tamper(
    mutator: Callable[[dict], None],
) -> None:
    requests, completed = _current_runner_calibrations()
    execution_contract = "a" * 64
    target_id = str(requests[0]["custom_id"])
    result = deepcopy(completed[target_id])
    mutator(result)
    bound = bind_rereview_calibration_receipt(
        result,
        execution_contract_sha256=execution_contract,
    )
    replayed = rereview_calibration_ledger(
        [bound],
        requests=requests,
        execution_contract_sha256=execution_contract,
        validate_core_result=review_runner._validate_calibration_result,
    )
    assert target_id not in replayed


def test_rereview_calibration_ledger_rejects_outer_binding_drift() -> None:
    requests, completed = _current_runner_calibrations()
    target_id = str(requests[0]["custom_id"])
    bound = bind_rereview_calibration_receipt(
        completed[target_id],
        execution_contract_sha256="a" * 64,
    )
    replayed = rereview_calibration_ledger(
        [bound],
        requests=requests,
        execution_contract_sha256="b" * 64,
        validate_core_result=review_runner._validate_calibration_result,
    )
    assert target_id not in replayed


def test_post150_wrappers_share_calibration_ids_and_ledger_path() -> None:
    execution_contract = rereview_execution_contract_sha256(
        runner_version=review_runner.RUNNER_VERSION,
        transport=review_runner.TRANSPORT,
        sse_completion_policy=review_runner.SSE_COMPLETION_POLICY,
    )
    requests = review_runner._calibration_requests(_original_requests()[0])
    expected_ids = {
        rereview_calibration_custom_id(str(row["custom_id"]), execution_contract)
        for row in requests
    }
    assert len(expected_ids) == 4
    assert blocker_review.runner.CALIBRATION_RESPONSES_PATH == (
        contract_review.runner.CALIBRATION_RESPONSES_PATH
    )
    assert blocker_review.runner.CALIBRATION_RESPONSES_PATH == (
        review_runner.CALIBRATION_RESPONSES_PATH
    )


def test_post150_wrappers_use_only_dedicated_saved_formal_recovery() -> None:
    for wrapper in (blocker_review, contract_review):
        assert "_recover_prior_invalid_review" not in inspect.getsource(wrapper.main)
        assert "_recover_latest_saved_same_concept_variant_subset" in inspect.getsource(
            wrapper.main
        )
        assert "formal_rereview_success_ledger" in inspect.getsource(
            wrapper._formal_success_ledger
        )
