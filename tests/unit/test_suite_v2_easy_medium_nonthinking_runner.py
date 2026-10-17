from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
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

import merge_easy_medium_reviewed as merge  # noqa: E402
import review_contracts as legacy  # noqa: E402
import run_easy_medium_nonthinking as runner  # noqa: E402
import validate_easy_medium_reviewed as scoped_validate  # noqa: E402


FROZEN_ENDPOINT = legacy._endpoint("https://dashscope.aliyuncs.com/compatible-mode/v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> tuple[
    dict[str, runner.profile.PackRuntime], dict[str, Any]
]:
    runtimes, predecessor_receipt = runner._load_predecessor_bridge(
        runner.load_runtimes()
    )
    manifest = runner.build_current_profile(
        runtimes=runtimes,
        model="qwen3.7-max",
        endpoint=FROZEN_ENDPOINT,
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
        predecessor_bridge_receipt=predecessor_receipt,
    )
    return runtimes, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _minimal_core(
    custom_id: str, manifest: dict[str, Any], *, status: str = "succeeded"
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "status": status,
        "runner_version": legacy.RUNNER_VERSION,
        "transport": legacy.TRANSPORT,
        "sse_completion_policy": legacy.SSE_COMPLETION_POLICY,
        "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"],
        "completed_at": "2026-08-13T00:00:01+00:00",
    }


def test_real_dry_run_is_zero_write_zero_api_and_preserves_formal_ledgers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal_paths = (
        runner.blocker.RESPONSES_PATH,
        runner.contract.RESPONSES_PATH,
        legacy.CALIBRATION_RESPONSES_PATH,
    )
    before = {
        path: (path.read_bytes().count(b"\n"), _sha256(path)) for path in formal_paths
    }
    sidecar_before = {
        path: path.exists()
        for path in (
            runner.PROFILE_PATH,
            runner.JOURNAL_PATH,
            runner.RESPONSES_PATH,
            runner.CALIBRATION_RESPONSES_PATH,
            runner.SUMMARY_PATH,
        )
    }

    assert runner.main([]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "dry_run_only"
    assert payload["write_performed"] is False
    assert payload["api_calls"] == 0
    assert payload["hard_calls"] == 0
    assert payload["summary_preview"]["case_expected_count"] == 30
    assert payload["summary_preview"]["case_success_count"] == 8
    assert payload["summary_preview"]["profile_counts"][
        "predecessor_v1_success"
    ] == 2
    assert payload["summary_preview"]["predecessor_v1_case_usage"][
        "total_tokens"
    ] == 10906
    assert payload["summary_preview"]["predecessor_v1_calibration_usage"][
        "total_tokens"
    ] == 3696
    assert payload["summary_preview"]["profile_counts"]["v5_adopted_success"] == 1
    assert payload["summary_preview"]["v5_adopted_case_usage"][
        "total_tokens"
    ] == 2085
    assert payload["summary_preview"]["v5_source_calibration_usage"][
        "total_tokens"
    ] == 4632
    assert payload["summary_preview"]["usage"]["total_tokens"] == 47522
    assert payload["summary_preview"]["recorded_api_attempts"] == 31
    pending_ids = payload["summary_preview"]["contract"]["pending_custom_ids"]
    assert "concept-contract-rereview-52d20717a579d2f585a6" not in pending_ids
    assert "concept-contract-rereview-056e9d3e6c713db68790" not in pending_ids
    assert payload["summary_preview"]["contract"]["success_count"] == 4
    assert len(payload["summary_preview"]["contract"]["pending_custom_ids"]) == 22
    assert payload["summary_preview"]["profile_counts"][
        "legacy_by_runner_version"
    ][legacy.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION] == 1
    assert {
        path: (path.read_bytes().count(b"\n"), _sha256(path)) for path in formal_paths
    } == before
    assert {
        path: path.exists() for path in sidecar_before
    } == sidecar_before
    assert not legacy.EXECUTION_LOCK_PATH.exists()


def test_predecessor_success_projection_tamper_is_rejected() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["blocker"]
    assert runtime.predecessor_bridge is not None
    tampered_rows = [deepcopy(row) for row in runtime.predecessor_bridge.success_rows]
    tampered_rows[0]["review"]["review_notes"] = ["TAMPERED BRIDGE ROW"]
    tampered_bridge = replace(
        runtime.predecessor_bridge,
        success_rows=tuple(tampered_rows),
    )
    tampered_runtime = replace(runtime, predecessor_bridge=tampered_bridge)

    with pytest.raises(
        runner.profile.ProfileContractError,
        match="projection exact replay drifted",
    ):
        runner.profile._reduce_production_journal(
            runtime=tampered_runtime,
            execution_profile=manifest,
            journal_rows=[],
            response_rows=[],
            validate_core_result=runner.core_validator(
                kind="blocker", execution_profile=manifest
            ),
        )


@pytest.mark.parametrize("mutation", ["top_level", "foreign_id"])
def test_v5_adoption_receipt_rejects_co_rehashed_schema_or_foreign_id(
    mutation: str,
) -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["contract"]
    assert runtime.v5_adoption_bridge is not None
    receipt = deepcopy(runtime.v5_adoption_bridge.receipt)
    if mutation == "top_level":
        receipt["foreign_top_level"] = "must be rejected"
    else:
        receipt["adopted_result_envelope_sha256_by_id"]["FOREIGN-ID"] = "0" * 64
    receipt["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered_runtime = replace(
        runtime,
        v5_adoption_bridge=replace(
            runtime.v5_adoption_bridge,
            receipt=receipt,
        ),
    )
    tampered_manifest = deepcopy(manifest)
    tampered_manifest["v5_adoption_policy"]["bridge_receipt_sha256"] = receipt[
        "receipt_sha256"
    ]
    tampered_manifest["execution_profile_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in tampered_manifest.items()
            if key != "execution_profile_sha256"
        }
    )

    with pytest.raises(
        runner.profile.ProfileContractError,
        match="schema is not exact|source maps are not exact",
    ):
        runner.profile._validated_v5_adoption(
            tampered_runtime, tampered_manifest
        )


def test_v5_calibration_sequence_co_rehash_is_rejected() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["contract"]
    bridge = runtime.v5_adoption_bridge
    assert bridge is not None
    starts = [deepcopy(row) for row in bridge.calibration_started_events]
    envelopes = [deepcopy(row) for row in bridge.calibration_envelopes]
    completions = [deepcopy(row) for row in bridge.calibration_completed_events]
    receipt = deepcopy(bridge.receipt)
    for started, envelope, completed in zip(
        starts, envelopes, completions, strict=True
    ):
        custom_id = str(envelope["logical_custom_id"])
        started["sequence"] = 1
        envelope["sequence"] = 1
        envelope["started_event_sha256"] = runner.profile.sha256_json(started)
        completed["sequence"] = 1
        completed["result_envelope_sha256"] = runner.profile.sha256_json(envelope)
        receipt["calibration_started_event_sha256_by_id"][custom_id] = (
            runner.profile.sha256_json(started)
        )
        receipt["calibration_result_envelope_sha256_by_id"][custom_id] = (
            runner.profile.sha256_json(envelope)
        )
        receipt["calibration_completed_event_sha256_by_id"][custom_id] = (
            runner.profile.sha256_json(completed)
        )
    receipt["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered_runtime = replace(
        runtime,
        v5_adoption_bridge=replace(
            bridge,
            receipt=receipt,
            calibration_envelopes=tuple(envelopes),
            calibration_started_events=tuple(starts),
            calibration_completed_events=tuple(completions),
        ),
    )
    tampered_manifest = deepcopy(manifest)
    tampered_manifest["v5_adoption_policy"]["bridge_receipt_sha256"] = receipt[
        "receipt_sha256"
    ]
    tampered_manifest["execution_profile_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in tampered_manifest.items()
            if key != "execution_profile_sha256"
        }
    )

    with pytest.raises(
        runner.profile.ProfileContractError,
        match="calibration evidence exact replay failed",
    ):
        runner.profile._validated_v5_adoption(
            tampered_runtime, tampered_manifest
        )


def test_v5_calibration_completed_started_hash_co_rehash_is_rejected() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["contract"]
    bridge = runtime.v5_adoption_bridge
    assert bridge is not None
    completions = [deepcopy(row) for row in bridge.calibration_completed_events]
    receipt = deepcopy(bridge.receipt)
    target = completions[0]
    custom_id = str(target["logical_custom_id"])
    target["started_event_sha256"] = "0" * 64
    receipt["calibration_completed_event_sha256_by_id"][custom_id] = (
        runner.profile.sha256_json(target)
    )
    receipt["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered_runtime = replace(
        runtime,
        v5_adoption_bridge=replace(
            bridge,
            receipt=receipt,
            calibration_completed_events=tuple(completions),
        ),
    )
    tampered_manifest = deepcopy(manifest)
    tampered_manifest["v5_adoption_policy"]["bridge_receipt_sha256"] = receipt[
        "receipt_sha256"
    ]
    tampered_manifest["execution_profile_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in tampered_manifest.items()
            if key != "execution_profile_sha256"
        }
    )

    with pytest.raises(
        runner.profile.ProfileContractError,
        match="causal chain|calibration evidence exact replay failed",
    ):
        runner.profile._validated_v5_adoption(
            tampered_runtime, tampered_manifest
        )


def test_v5_extra_execution_summary_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "execution-summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "V5_SUMMARY_PATH", summary)
    with pytest.raises(
        runner.profile.ProfileContractError,
        match="physically absent",
    ):
        runner._load_v5_adoption_bridge(runner.load_runtimes())


def test_v5_production_operation_co_rehash_is_rejected() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["contract"]
    bridge = runtime.v5_adoption_bridge
    assert bridge is not None
    envelopes = [deepcopy(row) for row in bridge.source_envelopes]
    projections = [deepcopy(row) for row in bridge.success_rows]
    receipt = deepcopy(bridge.receipt)
    custom_id = runner.profile.V5_ADOPTION_RESULT_ONLY_ID
    envelope = envelopes[0]
    envelope["operation_id"] = "FOREIGN-OP"
    receipt["adopted_result_envelope_sha256_by_id"][custom_id] = (
        runner.profile.sha256_json(envelope)
    )
    projections[0] = runner.profile._project_v5_adopted_success(
        envelope, started_event=bridge.source_started_events[0]
    )
    receipt["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    tampered_runtime = replace(
        runtime,
        v5_adoption_bridge=replace(
            bridge,
            receipt=receipt,
            source_envelopes=tuple(envelopes),
            success_rows=tuple(projections),
        ),
    )
    tampered_manifest = deepcopy(manifest)
    tampered_manifest["v5_adoption_policy"]["bridge_receipt_sha256"] = receipt[
        "receipt_sha256"
    ]
    tampered_manifest["execution_profile_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in tampered_manifest.items()
            if key != "execution_profile_sha256"
        }
    )

    with pytest.raises(
        runner.profile.ProfileContractError,
        match="source exact replay failed",
    ):
        runner.profile._validated_v5_adoption(
            tampered_runtime, tampered_manifest
        )


def test_v6_failed_calibration_reconciles_then_stays_terminal_zero_api() -> None:
    runtimes, manifest = _manifest()
    assert manifest["profile_id"].endswith("-v6")
    assert (
        manifest["execution_profile_sha256"]
        != "cb306d77d4923e89857c05fc5b5ed6d93c93078a412edb35c2cd4f4999569578"
    )
    request = runner.profile.build_calibration_requests(
        runtimes["blocker"].pack.requests[0], manifest
    )[1]
    v2_path = CONTRACT_ROOT / "easy-medium-nonthinking-v2" / "calibration-responses.jsonl"
    source = legacy._read_jsonl(v2_path)[1]["core_result"]
    failed_core = deepcopy(source)
    failed_core["custom_id"] = request["custom_id"]
    failed_core["prompt_sha256"] = request["prompt_sha256"]
    expected_errors = [
        "critical_constraints.polarity.cues must be an array",
        "missing calibration polarity: not_applicable",
    ]
    for item in failed_core["invalid_attempts"]:
        item["validation_errors"] = expected_errors
    failed_core["error"] = "ValueError: " + "; ".join(expected_errors)
    validator = runner.core_validator(kind="blocker", execution_profile=manifest)
    assert list(validator(failed_core, request, True)) == []

    started = runner.profile.build_started_event(
        kind="calibration",
        logical_custom_id=request["custom_id"],
        effective_row=request,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    envelope = runner.profile.wrap_result(
        kind="calibration",
        logical_custom_id=request["custom_id"],
        effective_row=request,
        core_result=failed_core,
        execution_profile=manifest,
        started_event=started,
    )
    appended: list[dict[str, Any]] = []
    calls = 0

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    reconciled = runner.profile.drive_pack(
        runtime=runtimes["blocker"],
        execution_profile=manifest,
        journal_rows=[started],
        response_rows=[],
        calibration_response_rows=[envelope],
        append_journal=appended.append,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=None,
        validate_core_result=validator,
    )
    assert reconciled["status"] == "reconciled_result_without_api"
    assert calls == 0
    assert [row["event_type"] for row in appended] == ["completed"]

    terminal = runner.profile.drive_pack(
        runtime=runtimes["blocker"],
        execution_profile=manifest,
        journal_rows=[started, appended[0]],
        response_rows=[],
        calibration_response_rows=[envelope],
        append_journal=lambda _row: pytest.fail("terminal restart must not append"),
        append_response=lambda _row: pytest.fail("terminal restart must not append"),
        append_calibration_response=lambda _row: pytest.fail(
            "terminal restart must not append"
        ),
        execute_one=execute,
        now=None,
        validate_core_result=validator,
    )
    assert terminal["status"] == "calibration_failed"
    assert calls == 0


def test_real_v3_failed_core_fixture_is_v4_zero_api_reconcilable() -> None:
    runtimes, manifest = _manifest()
    v3_root = CONTRACT_ROOT / "easy-medium-nonthinking-v3"
    source_envelope = next(
        row
        for row in legacy._read_jsonl(v3_root / "calibration-responses.jsonl")
        if row.get("core_result", {}).get("status") == "failed"
    )
    request = runner.profile.build_calibration_requests(
        runtimes["blocker"].pack.requests[0], manifest
    )[3]
    failed_core = deepcopy(source_envelope["core_result"])
    failed_core["custom_id"] = request["custom_id"]
    failed_core["prompt_sha256"] = request["prompt_sha256"]
    for item in failed_core["invalid_attempts"]:
        item["validation_errors"] = [
            "missing calibration polarity: not_applicable"
        ]
    failed_core["error"] = (
        "ValueError: missing calibration polarity: not_applicable"
    )
    started = runner.profile.build_started_event(
        kind="calibration",
        logical_custom_id=request["custom_id"],
        effective_row=request,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    envelope = runner.profile.wrap_result(
        kind="calibration",
        logical_custom_id=request["custom_id"],
        effective_row=request,
        core_result=failed_core,
        execution_profile=manifest,
        started_event=started,
    )
    validator = runner.core_validator(kind="blocker", execution_profile=manifest)
    assert list(validator(failed_core, request, True)) == []
    appended: list[dict[str, Any]] = []
    calls = 0

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    result = runner.profile.drive_pack(
        runtime=runtimes["blocker"],
        execution_profile=manifest,
        journal_rows=[started],
        response_rows=[],
        calibration_response_rows=[envelope],
        append_journal=appended.append,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=None,
        validate_core_result=validator,
    )
    assert result["status"] == "reconciled_result_without_api"
    assert calls == 0
    assert [row["event_type"] for row in appended] == ["completed"]


@pytest.mark.parametrize(
    "mirror",
    ["invalid_attempt_embedded", "terminal"],
)
def test_v4_failed_calibration_rejects_parsed_review_sha_tamper(
    mirror: str,
) -> None:
    runtimes, manifest = _manifest()
    v3_root = CONTRACT_ROOT / "easy-medium-nonthinking-v3"
    source_envelope = next(
        row
        for row in legacy._read_jsonl(v3_root / "calibration-responses.jsonl")
        if row.get("core_result", {}).get("status") == "failed"
    )
    request = runner.profile.build_calibration_requests(
        runtimes["blocker"].pack.requests[0], manifest
    )[3]
    core = deepcopy(source_envelope["core_result"])
    core["custom_id"] = request["custom_id"]
    core["prompt_sha256"] = request["prompt_sha256"]
    for item in core["invalid_attempts"]:
        item["validation_errors"] = [
            "missing calibration polarity: not_applicable"
        ]
    core["error"] = "ValueError: missing calibration polarity: not_applicable"
    if mirror == "invalid_attempt_embedded":
        core["invalid_attempts"][0]["sse_integrity"][
            "parsed_review_sha256"
        ] = "0" * 64
        expected_error = "invalid-attempt SSE receipt"
    else:
        core["sse_integrity"]["parsed_review_sha256"] = "0" * 64
        expected_error = "terminal SSE receipt"
    errors = list(
        runner.core_validator(kind="blocker", execution_profile=manifest)(
            core, request, True
        )
    )
    assert errors == [f"calibration failed {expected_error} is invalid"]


@pytest.mark.parametrize(
    "argv,match",
    [
        (["--base-url", "https://example.invalid/v1"], "DashScope"),
        (["--model", "other-model"], "Qwen"),
    ],
)
def test_runner_refuses_nonfrozen_provider_identity_before_loading_packs(
    argv: list[str], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "load_runtimes",
        lambda: pytest.fail("pack/provider work must not start"),
    )
    with pytest.raises(SystemExit, match=match):
        runner.main(argv)


def test_production_adapter_passes_top_level_false_to_real_execute_boundary() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["blocker"]
    custom_id = next(
        value
        for value in runner.profile._selection(runtime.pack).target_custom_ids
        if value not in runner.profile._validate_runtime(runtime)["successful_by_id"]
    )
    base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
    effective = runner.profile.effective_request_row(base, manifest, kind="blocker")
    observed: dict[str, Any] = {}

    def spy(**kwargs: Any) -> dict[str, Any]:
        observed.update(deepcopy(kwargs))
        return {
            "custom_id": custom_id,
            "status": "failed",
            "runner_version": legacy.RUNNER_VERSION,
            "transport": legacy.TRANSPORT,
            "sse_completion_policy": legacy.SSE_COMPLETION_POLICY,
            "model": "qwen3.7-max",
            "prompt_sha256": effective["prompt_sha256"],
            "completed_at": "2026-08-13T00:00:01+00:00",
        }

    bound = runner.profile.execute_profile_request(
        effective,
        kind="blocker",
        calibration=False,
        execution_profile=manifest,
        api_key="masked-test-key",
        execute_one=spy,
    )

    assert observed["row"]["request"]["enable_thinking"] is False
    assert observed["row"]["request"]["model"] == "qwen3.7-max"
    assert observed["endpoint"] == FROZEN_ENDPOINT
    assert observed["timeout"] == 600.0
    assert observed["max_attempts"] == 3
    assert observed["retry_delay"] == 2.0
    assert observed["calibration"] is False
    assert bound["execution_contract_sha256"] == manifest[
        "execution_profile_sha256"
    ]
    assert bound["sequence"] == manifest["base_packs"]["blocker"][
        "target_sequence_by_custom_id"
    ][custom_id]


def test_real_execute_one_passes_top_level_false_to_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用本地provider spy穿透真实_execute_one，禁止测试发起网络调用。"""

    runtimes, manifest = _manifest()
    runtime = runtimes["blocker"]
    target_ids = set(runner.profile._selection(runtime.pack).target_custom_ids)
    source = next(
        deepcopy(row)
        for row in runtime.legacy_rows
        if row.get("custom_id") in target_ids and row.get("status") == "succeeded"
    )
    base = next(
        row for row in runtime.pack.requests if row["custom_id"] == source["custom_id"]
    )
    effective = runner.profile.effective_request_row(base, manifest, kind="blocker")
    observed: dict[str, Any] = {}

    def fake_call_api(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observed.update(deepcopy(kwargs))
        integrity_audit = kwargs["integrity_audit"]
        integrity_audit.clear()
        integrity_audit.update(deepcopy(source["sse_integrity"]))
        return deepcopy(source["review"]), deepcopy(source["successful_response_usage"])

    monkeypatch.setattr(legacy, "_call_api", fake_call_api)
    result = runner.profile.execute_profile_request(
        effective,
        kind="blocker",
        calibration=False,
        execution_profile=manifest,
        api_key="masked-test-key",
    )

    assert observed["request_body"]["enable_thinking"] is False
    assert observed["request_body"]["model"] == manifest["model"]
    assert result["status"] == "succeeded"
    assert result["execution_contract_sha256"] == manifest[
        "execution_profile_sha256"
    ]


def _v5_materializer_retry_fixture() -> tuple[
    dict[str, runner.profile.PackRuntime],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, int],
    dict[str, Any],
]:
    runtimes, manifest = _manifest()
    envelope = legacy._read_jsonl(runner.V5_RESPONSES_PATH)[0]
    good = deepcopy(envelope["core_result"]["review"])
    bad = deepcopy(good)
    bad["required_concepts"]["all_of"][0]["any_of"] = [
        bad["required_concepts"]["all_of"][0]["canonical_text"]
    ]
    custom_id = str(envelope["logical_custom_id"])
    runtime = runtimes["contract"]
    base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
    effective = runner.profile.effective_request_row(base, manifest, kind="contract")
    usage = deepcopy(envelope["core_result"]["successful_response_usage"])
    integrity = deepcopy(envelope["core_result"]["sse_integrity"])
    return runtimes, manifest, effective, good, usage, {"bad": bad, "integrity": integrity}


@pytest.mark.parametrize("eof", [False, True])
def test_materializer_failure_is_semantic_retry_then_success_with_exact_replay(
    eof: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _runtimes,
        manifest,
        effective,
        good,
        usage,
        fixture,
    ) = _v5_materializer_retry_fixture()
    reviews = [deepcopy(fixture["bad"]), deepcopy(good)]

    def fake_call_api(**kwargs: Any) -> tuple[dict[str, Any], dict[str, int]]:
        review = reviews.pop(0)
        errors = list(kwargs["eof_review_validator"](deepcopy(review))) if eof else []
        audit = deepcopy(fixture["integrity"])
        if eof:
            audit.update(
                {
                    "termination": "validated_stop_eof",
                    "done_seen": False,
                    "review_validation": "failed" if errors else "passed",
                    "content_chars": len(
                        json.dumps(review, ensure_ascii=False, separators=(",", ":"))
                    ),
                }
            )
        kwargs["integrity_audit"].update(audit)
        return review, deepcopy(usage)

    monkeypatch.setattr(legacy, "_call_api", fake_call_api)
    result = runner.profile.execute_profile_request(
        effective,
        kind="contract",
        calibration=False,
        execution_profile=manifest,
        api_key="masked-test-key",
    )

    assert result["status"] == "succeeded"
    assert result["attempt"] == 2
    assert result["retry_accounting"]["semantic_failure_count"] == 1
    invalid = result["invalid_attempts_before_success"]
    assert len(invalid) == 1
    assert invalid[0]["validation_errors"] == [
        "concept materialization failed: concept groups have no collision-free nonempty subset"
    ]
    assert result["usage"]["total_tokens"] == usage["total_tokens"] * 2
    assert list(
        runner.core_validator(kind="contract", execution_profile=manifest)(
            result, effective, False
        )
    ) == []


def test_three_materializer_failures_are_terminal_semantic_failure_with_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _runtimes,
        manifest,
        effective,
        _good,
        usage,
        fixture,
    ) = _v5_materializer_retry_fixture()

    def fake_call_api(**kwargs: Any) -> tuple[dict[str, Any], dict[str, int]]:
        kwargs["integrity_audit"].update(deepcopy(fixture["integrity"]))
        return deepcopy(fixture["bad"]), deepcopy(usage)

    monkeypatch.setattr(legacy, "_call_api", fake_call_api)
    result = runner.profile.execute_profile_request(
        effective,
        kind="contract",
        calibration=False,
        execution_profile=manifest,
        api_key="masked-test-key",
    )

    assert result["status"] == "failed"
    assert result["attempt"] == 3
    assert result["retry_accounting"]["quota_exhausted"] == "semantic"
    assert result["retry_accounting"]["semantic_failure_count"] == 3
    assert len(result["invalid_attempts"]) == 3
    assert result["usage"]["total_tokens"] == usage["total_tokens"] * 3
    assert list(
        runner.core_validator(kind="contract", execution_profile=manifest)(
            result, effective, False
        )
    ) == []


def test_real_validator_accepts_exact_transport_failure_without_usage() -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["blocker"]
    target_ids = set(runner.profile._selection(runtime.pack).target_custom_ids)
    raw = next(
        deepcopy(row)
        for row in runtime.legacy_rows
        if row.get("custom_id") in target_ids
        and row.get("status") == "failed"
        and "usage" not in row
    )
    raw.pop("execution_contract_sha256")
    raw.pop("sequence")
    raw["prompt_sha256"] = manifest["prompt_sha256"]
    bound = runner.profile.bind_production_core_result(
        raw,
        kind="blocker",
        logical_custom_id=raw["custom_id"],
        execution_profile=manifest,
    )
    base = next(
        row for row in runtime.pack.requests if row["custom_id"] == raw["custom_id"]
    )
    effective = runner.profile.effective_request_row(base, manifest, kind="blocker")

    assert (
        list(
            runner.core_validator(kind="blocker", execution_profile=manifest)(
                bound, effective, False
            )
        )
        == []
    )


def test_real_success_validator_always_replays_materializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtimes, manifest = _manifest()
    runtime = runtimes["blocker"]
    target_ids = set(runner.profile._selection(runtime.pack).target_custom_ids)
    raw = next(
        deepcopy(row)
        for row in runtime.legacy_rows
        if row.get("custom_id") in target_ids and row.get("status") == "succeeded"
    )
    raw.pop("execution_contract_sha256")
    raw.pop("sequence")
    bound = runner.profile.bind_production_core_result(
        raw,
        kind="blocker",
        logical_custom_id=raw["custom_id"],
        execution_profile=manifest,
    )
    base = next(
        row for row in runtime.pack.requests if row["custom_id"] == raw["custom_id"]
    )
    effective = runner.profile.effective_request_row(base, manifest, kind="blocker")
    monkeypatch.setattr(
        legacy,
        "_review_materialization_errors",
        lambda _review, _payload, **_kwargs: ["injected materializer failure"],
    )

    errors = list(
        runner.core_validator(kind="blocker", execution_profile=manifest)(
            bound, effective, False
        )
    )
    assert "injected materializer failure" in errors


def test_release_loader_absent_is_legacy_compatible_and_partial_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "PROFILE_PATH": tmp_path / "execution-profile.json",
        "JOURNAL_PATH": tmp_path / "journal.jsonl",
        "RESPONSES_PATH": tmp_path / "responses.jsonl",
        "CALIBRATION_RESPONSES_PATH": tmp_path / "calibration-responses.jsonl",
        "SUMMARY_PATH": tmp_path / "execution-summary.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    assert runner.load_release_snapshot() is None
    paths["SUMMARY_PATH"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.profile.ProfileContractError, match="partial"):
        runner.load_release_snapshot()
    paths["SUMMARY_PATH"].unlink()
    paths["JOURNAL_PATH"].write_text("", encoding="utf-8")
    with pytest.raises(runner.profile.ProfileContractError, match="partial"):
        runner.load_release_snapshot()


def test_release_loader_rejects_foreign_endpoint_even_when_profile_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtimes = runner.load_runtimes()
    foreign = runner.build_current_profile(
        runtimes=runtimes,
        model="qwen3.7-max",
        endpoint="https://example.invalid/v1/chat/completions",
        request_timeout=600.0,
        max_attempts=3,
        retry_delay=2.0,
    )
    paths = {
        "PROFILE_PATH": tmp_path / "execution-profile.json",
        "JOURNAL_PATH": tmp_path / "journal.jsonl",
        "RESPONSES_PATH": tmp_path / "responses.jsonl",
        "CALIBRATION_RESPONSES_PATH": tmp_path / "calibration-responses.jsonl",
        "SUMMARY_PATH": tmp_path / "execution-summary.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    runner._write_json_fsync(paths["PROFILE_PATH"], foreign)
    for name in ("JOURNAL_PATH", "RESPONSES_PATH", "CALIBRATION_RESPONSES_PATH"):
        paths[name].write_text("", encoding="utf-8")

    with pytest.raises(runner.profile.ProfileContractError, match="provider identity"):
        runner.load_release_snapshot()


def test_merge_consumer_uses_release_loader_and_absent_sidecars_preserve_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, filename in {
        "PROFILE_PATH": "execution-profile.json",
        "JOURNAL_PATH": "journal.jsonl",
        "RESPONSES_PATH": "responses.jsonl",
        "CALIBRATION_RESPONSES_PATH": "calibration-responses.jsonl",
        "SUMMARY_PATH": "execution-summary.json",
    }.items():
        monkeypatch.setattr(runner, name, tmp_path / filename)
    calls = 0
    original = runner.load_release_snapshot

    def spy() -> dict[str, Any] | None:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(merge.nonthinking, "load_release_snapshot", spy)
    readiness = merge.scope_readiness()
    assert readiness["missing_case_count"] == 27
    assert readiness["hard_rows_counted"] == 0
    assert calls == 2


def test_current_profile_preserves_old_runner_prompt_contract_and_hard_exclusion() -> None:
    runtimes, manifest = _manifest()
    assert manifest["legacy_runner_version"] == legacy.RUNNER_VERSION
    assert {
        manifest["base_packs"][kind]["base_execution_contract_sha256"]
        for kind in runner.profile.KINDS
    } == {"60694067e286a704a7644e70c63786c0c062e44be27373b8b79927dfc9ba9414"}
    assert manifest["base_prompt_sha256"] == runner.blocker.EXPECTED_PROMPT_SHA256
    assert manifest["prompt_sha256"] != runner.blocker.EXPECTED_PROMPT_SHA256
    assert manifest["request_overlay"] == {"enable_thinking": False}
    for kind in runner.profile.KINDS:
        selection = runner.profile._selection(runtimes[kind].pack)
        assert set(selection.target_difficulty_by_custom_id.values()) <= {
            "easy",
            "medium",
        }


def test_v6_uses_independent_output_root() -> None:
    assert runner.PROFILE_PATH.parent.name == "easy-medium-nonthinking-v6"
    assert runner.PROFILE_PATH.parent != CONTRACT_ROOT / "easy-medium-nonthinking-v4"


def test_completed_sidecar_is_consumed_by_shared_merge_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtimes, manifest = _manifest()
    paths = {
        "PROFILE_PATH": tmp_path / "execution-profile.json",
        "JOURNAL_PATH": tmp_path / "journal.jsonl",
        "RESPONSES_PATH": tmp_path / "responses.jsonl",
        "CALIBRATION_RESPONSES_PATH": tmp_path / "calibration-responses.jsonl",
        "SUMMARY_PATH": tmp_path / "execution-summary.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    real_core_validator = runner.core_validator

    def test_core_validator(**kwargs: Any) -> Any:
        if kwargs["execution_profile"].get("profile_id") == runner.profile.PROFILE_ID:
            return lambda _result, _request, _calibration: []
        return real_core_validator(**kwargs)

    monkeypatch.setattr(runner, "core_validator", test_core_validator)

    journal: list[dict[str, Any]] = []
    calibration_responses: list[dict[str, Any]] = []
    template = runtimes["blocker"].pack.requests[0]
    for row in runner.profile.build_calibration_requests(template, manifest):
        started = runner.profile.build_started_event(
            kind="calibration",
            logical_custom_id=row["custom_id"],
            effective_row=row,
            execution_profile=manifest,
            started_at="2026-08-13T00:00:00+00:00",
        )
        envelope = runner.profile.wrap_result(
            kind="calibration",
            logical_custom_id=row["custom_id"],
            effective_row=row,
            core_result=_minimal_core(row["custom_id"], manifest),
            execution_profile=manifest,
            started_event=started,
        )
        journal.extend(
            [
                started,
                runner.profile.build_completed_event(
                    started_event=started, result_envelope=envelope
                ),
            ]
        )
        calibration_responses.append(envelope)

    runtime = runtimes["contract"]
    old_success = runner.profile._validate_runtime(runtime)["successful_by_id"]
    adopted_success = set(
        runner.profile._validated_v4_adoption(runtime, manifest)[0]
    )
    adopted_success.update(
        runner.profile._validated_v5_adoption(runtime, manifest)[0]
    )
    custom_id = next(
        value
        for value in runner.profile._selection(runtime.pack).target_custom_ids
        if value not in old_success and value not in adopted_success
    )
    base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
    effective = runner.profile.effective_request_row(base, manifest, kind="contract")
    started = runner.profile.build_started_event(
        kind="contract",
        logical_custom_id=custom_id,
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    core = runner.profile.bind_production_core_result(
        _minimal_core(custom_id, manifest),
        kind="contract",
        logical_custom_id=custom_id,
        execution_profile=manifest,
    )
    response = runner.profile.wrap_result(
        kind="contract",
        logical_custom_id=custom_id,
        effective_row=effective,
        core_result=core,
        execution_profile=manifest,
        started_event=started,
    )
    journal.extend(
        [
            started,
            runner.profile.build_completed_event(
                started_event=started, result_envelope=response
            ),
        ]
    )
    runner._write_json_fsync(paths["PROFILE_PATH"], manifest)
    _write_jsonl(paths["JOURNAL_PATH"], journal)
    _write_jsonl(paths["RESPONSES_PATH"], [response])
    _write_jsonl(paths["CALIBRATION_RESPONSES_PATH"], calibration_responses)
    summary = runner.profile.build_scope_summary(
        runtimes=runtimes,
        execution_profile=manifest,
        response_rows=[response],
        calibration_response_rows=calibration_responses,
        journal_rows=journal,
        validate_core_result_by_kind={
            kind: (lambda _result, _request, _calibration: [])
            for kind in runner.profile.KINDS
        },
    )
    runner._write_json_fsync(paths["SUMMARY_PATH"], summary)

    snapshot = runner.load_release_snapshot()
    assert snapshot is not None
    assert custom_id in snapshot["combined_authoritative_by_kind"]["contract"]
    assert snapshot["combined_receipt_by_kind"]["contract"][
        "fresh_calibration_receipt"
    ]["required_case_count"] == 4
    readiness = merge.scope_readiness()
    assert readiness["missing_case_count"] == 21
    assert readiness["contract"]["nonthinking_combined_receipt"][
        "new_profile_success_count"
    ] == 1
    validation_fixture = deepcopy(readiness["contract"])
    selected_ids = sorted(validation_fixture["selected_response_row_sha256_by_id"])
    validation_fixture["scope_target_custom_ids"] = selected_ids
    validation_fixture["selected_case_count"] = len(selected_ids)
    validation_fixture["authoritative_success_count"] = len(selected_ids)
    validation_fixture["missing_case_count"] = 0
    validation_fixture["receipt_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in validation_fixture.items()
            if key != "receipt_sha256"
        }
    )
    validated_receipt = scoped_validate._validate_receipt_self_hash(
        validation_fixture, kind="contract"
    )
    assert validated_receipt["nonthinking_combined_receipt"][
        "fresh_calibration_receipt"
    ]["required_case_count"] == 4

    tampered = deepcopy(validation_fixture)
    tampered["nonthinking_combined_receipt"]["fresh_calibration_receipt"][
        "required_case_count"
    ] = 3
    combined = tampered["nonthinking_combined_receipt"]
    combined["fresh_calibration_receipt"]["receipt_sha256"] = runner.profile.sha256_json(
        {
            key: value
            for key, value in combined["fresh_calibration_receipt"].items()
            if key != "receipt_sha256"
        }
    )
    combined["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in combined.items() if key != "receipt_sha256"}
    )
    tampered["receipt_sha256"] = runner.profile.sha256_json(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(
        scoped_validate.ScopedValidationError,
        match="calibration|physical sidecars",
    ):
        scoped_validate._validate_receipt_self_hash(tampered, kind="contract")
