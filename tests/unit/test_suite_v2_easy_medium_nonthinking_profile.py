from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)


def _load_module():
    module_path = CONTRACT_ROOT / "easy_medium_nonthinking_profile.py"
    spec = importlib.util.spec_from_file_location(
        "easy_medium_nonthinking_profile_test", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile = _load_module()


def _request(custom_id: str, *, marker: str = "easy") -> dict[str, Any]:
    payload = {"query": marker, "reference_answer": marker}
    system_prompt = "system"
    return {
        "custom_id": custom_id,
        "payload_sha256": profile.sha256_json(payload),
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "request": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "model": "qwen3.7-max",
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    }


def _scope() -> dict[str, Any]:
    return profile.scope.scope_profile()


def _authority(
    custom_id: str, difficulty: str, *, marker: str
) -> dict[str, Any]:
    namespace = {"easy": "EASY", "medium": "MEDIUM", "hard": "HARD"}[
        difficulty
    ]
    return {
        "custom_id": custom_id,
        "case_id": f"P3500-{namespace}-V2-001",
        "parent_case_id": f"P3500-{namespace}-001",
        "difficulty": difficulty,
        "payload_sha256": profile.sha256_json(
            {"query": marker, "reference_answer": marker}
        ),
    }


def _mapping(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        key: authority[key]
        for key in ("custom_id", "case_id", "parent_case_id", "difficulty")
    }


def _packs() -> dict[str, profile.PackInput]:
    blocker_authority = (
        _authority("blocker-easy-1", "easy", marker="easy"),
        _authority("blocker-hard-1", "hard", marker="hard"),
    )
    contract_authority = (
        _authority("contract-medium-1", "medium", marker="medium"),
    )
    return {
        "blocker": profile.PackInput(
            requests=(_request("blocker-easy-1"), _request("blocker-hard-1", marker="hard")),
            request_map=tuple(_mapping(row) for row in blocker_authority),
            authority_rows=blocker_authority,
            base_execution_contract_sha256="b" * 64,
        ),
        "contract": profile.PackInput(
            requests=(_request("contract-medium-1", marker="medium"),),
            request_map=tuple(_mapping(row) for row in contract_authority),
            authority_rows=contract_authority,
            base_execution_contract_sha256="c" * 64,
        ),
    }


def _replace_pack(
    packs: dict[str, profile.PackInput], kind: str, **changes: Any
) -> dict[str, profile.PackInput]:
    result = deepcopy(packs)
    current = result[kind]
    result[kind] = profile.PackInput(
        requests=changes.get("requests", current.requests),
        request_map=changes.get("request_map", current.request_map),
        authority_rows=changes.get("authority_rows", current.authority_rows),
        base_execution_contract_sha256=changes.get(
            "base_execution_contract_sha256",
            current.base_execution_contract_sha256,
        ),
    )
    return result


def _manifest(**overrides: Any) -> dict[str, Any]:
    values = {
        "packs": _packs(),
        "scope_profile": _scope(),
        "model": "qwen3.7-max",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "request_timeout": 600.0,
        "max_attempts": 3,
        "retry_delay": 2.0,
    }
    values.update(overrides)
    return profile.build_execution_profile(**values)


def _core_result(custom_id: str, *, status: str = "succeeded") -> dict[str, Any]:
    result: dict[str, Any] = {
        "custom_id": custom_id,
        "status": status,
        "runner_version": profile.LEGACY_RUNNER_VERSION,
        "transport": profile.TRANSPORT,
        "sse_completion_policy": profile.SSE_COMPLETION_POLICY,
        "model": "qwen3.7-max",
        "prompt_sha256": hashlib.sha256(b"system").hexdigest(),
        "completed_at": "2026-08-13T00:00:01+00:00",
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "attempt_usage": [
            {"attempt": 1, "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        ],
    }
    if status == "succeeded":
        result.update({"attempt": 1, "review": {"verdict": "approve"}})
    else:
        result.update({"attempt": 3, "error": "provider failed"})
    return result


def _pack_runtime(
    *,
    kind: str,
    legacy_rows: tuple[dict[str, Any], ...] = (),
) -> profile.PackRuntime:
    pack = _packs()[kind]

    def replay(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
        successful = {
            str(row["custom_id"]): row
            for row in rows
            if row.get("status") == "succeeded"
        }
        return {
            "successful_by_id": successful,
            "retry_eligible_by_id": {},
            "invalidated_by_id": {},
            "retry_started_by_id": {},
            "terminal_by_id": {},
            "adopted_by_id": {},
            "episode_by_id": {},
        }

    return profile.PackRuntime(
        kind=kind,
        pack=pack,
        legacy_rows=legacy_rows,
        replay_legacy=replay,
    )


def _production_core(
    manifest: dict[str, Any], kind: str, custom_id: str, *, status: str = "succeeded"
) -> dict[str, Any]:
    core = _core_result(custom_id, status=status)
    core["prompt_sha256"] = manifest["prompt_sha256"]
    return profile.bind_production_core_result(
        core,
        kind=kind,
        logical_custom_id=custom_id,
        execution_profile=manifest,
    )


def _completed_calibration(
    manifest: dict[str, Any], template: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    journal: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for row in profile.build_calibration_requests(template, manifest):
        started = profile.build_started_event(
            kind="calibration",
            logical_custom_id=row["custom_id"],
            effective_row=row,
            execution_profile=manifest,
            started_at="2026-08-13T00:00:00+00:00",
        )
        core = _core_result(row["custom_id"])
        core["prompt_sha256"] = manifest["prompt_sha256"]
        result = profile.wrap_result(
            kind="calibration",
            logical_custom_id=row["custom_id"],
            effective_row=row,
            core_result=core,
            execution_profile=manifest,
            started_event=started,
        )
        journal.extend(
            [started, profile.build_completed_event(started_event=started, result_envelope=result)]
        )
        responses.append(result)
    return journal, responses


def _completed_production(
    manifest: dict[str, Any], runtime: profile.PackRuntime, *, status: str = "succeeded"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    custom_id = profile._selection(runtime.pack).target_custom_ids[0]
    base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
    effective = profile.effective_request_row(base, manifest, kind=runtime.kind)
    started = profile.build_started_event(
        kind=runtime.kind,
        logical_custom_id=custom_id,
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    envelope = profile.wrap_result(
        kind=runtime.kind,
        logical_custom_id=custom_id,
        effective_row=effective,
        core_result=_production_core(manifest, runtime.kind, custom_id, status=status),
        execution_profile=manifest,
        started_event=started,
    )
    completed = profile.build_completed_event(
        started_event=started, result_envelope=envelope
    )
    return [started, completed], [envelope]


def test_profile_overlays_exact_false_without_mutating_frozen_requests() -> None:
    packs = _packs()
    before = deepcopy(packs)
    manifest = _manifest(packs=packs)
    effective = profile.effective_request_row(
        packs["blocker"].requests[0], manifest, kind="blocker"
    )

    assert packs == before
    assert "enable_thinking" not in packs["blocker"].requests[0]["request"]
    assert effective["request"]["enable_thinking"] is False
    assert manifest["request_overlay"] == {"enable_thinking": False}
    assert manifest["base_packs"]["blocker"]["base_pack_sha256"] != (
        manifest["base_packs"]["blocker"]["effective_target_pack_sha256"]
    )
    prompt = effective["request"]["messages"][0]["content"]
    assert all(value in prompt for value in profile.POLARITY_REQUIRED_VALUES)
    calibration = profile.build_calibration_requests(
        packs["blocker"].requests[0], manifest
    )
    assert len(calibration) == 4
    assert calibration[-1]["expected"]["polarity"] == "not_applicable"
    assert calibration[-1]["custom_id"].endswith("not_applicable_polarity")


def test_prompt_contract_explicitly_requires_polarity_cues_array() -> None:
    manifest = _manifest()
    effective = profile.effective_request_row(
        _packs()["blocker"].requests[0], manifest, kind="blocker"
    )
    prompts = [effective["request"]["messages"][0]["content"]]
    prompts.extend(
        row["request"]["messages"][0]["content"]
        for row in profile.build_calibration_requests(
            _packs()["blocker"].requests[0], manifest
        )
    )

    for prompt in prompts:
        assert (
            "critical_constraints.polarity.cues MUST be a JSON array of strings"
            in prompt
        )
        assert "Do not omit the cues key; use [] when no polarity cue applies" in prompt


def test_v6_neutral_polarity_policy_and_case4_manifest_are_frozen() -> None:
    manifest = _manifest()
    assert manifest["schema_version"].endswith(".v6")
    assert manifest["profile_id"].endswith("-v6")
    assert manifest["runner_version"].endswith("-v6")
    prompt = profile.effective_request_row(
        _packs()["blocker"].requests[0], manifest, kind="blocker"
    )["request"]["messages"][0]["content"]
    assert (
        "Neutral descriptive, identifier, or procedural action-to-outcome facts"
    ) in prompt
    assert "no negation, prohibition, permission, obligation or requirement" in prompt
    assert "recommendation, comparison or contrast, or condition" in prompt
    assert (
        "a neutral fact being true MUST NOT by itself be classified as "
        '"affirmative"'
    ) in prompt
    cases = profile._v4_calibration_cases()
    assert cases[1].expected["polarity"] == "not_applicable"
    case4 = cases[3]
    assert case4.expected["polarity"] == "not_applicable"
    assert "max_concepts" not in case4.expected
    assert case4.payload["deterministic_candidate"] == {}


@pytest.mark.parametrize(
    "overlay",
    [{}, {"enable_thinking": True}, {"enable_thinking": False, "extra": 1}],
)
def test_profile_rejects_any_nonexact_overlay(overlay: dict[str, Any]) -> None:
    with pytest.raises(profile.ProfileContractError, match="overlay"):
        profile.build_execution_profile(
            packs=_packs(),
            scope_profile=_scope(),
            model="qwen3.7-max",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            request_timeout=600.0,
            max_attempts=3,
            retry_delay=2.0,
            request_overlay=overlay,
        )


def test_profile_digest_changes_for_endpoint_model_or_runtime_knob() -> None:
    base = _manifest()["execution_profile_sha256"]
    with pytest.raises(profile.ProfileContractError, match="model differs"):
        _manifest(model="qwen3.7-max-2026-08-01")
    changed_model_packs = deepcopy(_packs())
    for pack in changed_model_packs.values():
        for row in pack.requests:
            row["request"]["model"] = "qwen3.7-max-2026-08-01"
    assert (
        _manifest(
            packs=changed_model_packs,
            model="qwen3.7-max-2026-08-01",
        )["execution_profile_sha256"]
        != base
    )
    assert (
        _manifest(endpoint="https://example.invalid/v1/chat/completions")[
            "execution_profile_sha256"
        ]
        != base
    )
    assert _manifest(request_timeout=601.0)["execution_profile_sha256"] != base
    assert _manifest(max_attempts=2)["execution_profile_sha256"] != base
    assert _manifest(retry_delay=3.0)["execution_profile_sha256"] != base


def test_profile_digest_binds_full_pack_hard_row_target_sequence_prompt_and_contract() -> None:
    base = _manifest()["execution_profile_sha256"]

    changed_contract = _replace_pack(
        _packs(),
        "contract",
        base_execution_contract_sha256="d" * 64,
    )
    assert _manifest(packs=changed_contract)["execution_profile_sha256"] != base

    reordered = _packs()
    blocker = reordered["blocker"]
    reordered = _replace_pack(
        reordered,
        "blocker",
        requests=tuple(reversed(blocker.requests)),
    )
    assert _manifest(packs=reordered)["execution_profile_sha256"] != base

    hard_changed = _packs()
    blocker = hard_changed["blocker"]
    hard_requests = deepcopy(blocker.requests)
    hard_authority = deepcopy(blocker.authority_rows)
    payload = {"query": "hard-changed", "reference_answer": "hard-changed"}
    hard_requests[1]["request"]["messages"][1]["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    hard_requests[1]["payload_sha256"] = profile.sha256_json(payload)
    hard_authority[1]["payload_sha256"] = profile.sha256_json(payload)
    hard_changed = _replace_pack(
        hard_changed,
        "blocker",
        requests=tuple(hard_requests),
        authority_rows=tuple(hard_authority),
    )
    assert _manifest(packs=hard_changed)["execution_profile_sha256"] != base

    prompt_changed = _packs()
    for kind, pack in list(prompt_changed.items()):
        rows = deepcopy(pack.requests)
        for row in rows:
            row["request"]["messages"][0]["content"] = "system-v2"
            row["prompt_sha256"] = hashlib.sha256(b"system-v2").hexdigest()
        prompt_changed = _replace_pack(
            prompt_changed,
            kind,
            requests=tuple(rows),
        )
    assert _manifest(packs=prompt_changed)["execution_profile_sha256"] != base


def test_profile_digest_binds_calibration_manifest_transport_and_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _manifest()["execution_profile_sha256"]
    original_cases = profile.legacy._calibration_cases()
    changed_cases = deepcopy(original_cases)
    changed_cases[0] = replace(
        changed_cases[0],
        expected={**changed_cases[0].expected, "new_expected_field": "bound"},
    )
    monkeypatch.setattr(profile.legacy, "_calibration_cases", lambda: changed_cases)
    assert _manifest()["execution_profile_sha256"] != base
    monkeypatch.setattr(profile.legacy, "_calibration_cases", lambda: original_cases)

    original_transport = profile.TRANSPORT
    original_sse = profile.SSE_COMPLETION_POLICY
    monkeypatch.setattr(profile, "TRANSPORT", original_transport + "-changed")
    assert _manifest()["execution_profile_sha256"] != base
    monkeypatch.setattr(profile, "TRANSPORT", original_transport)
    monkeypatch.setattr(profile, "SSE_COMPLETION_POLICY", original_sse + "-changed")
    assert _manifest()["execution_profile_sha256"] != base


@pytest.mark.parametrize("field", list(_scope().keys()))
def test_profile_rejects_every_scope_field_drift(field: str) -> None:
    changed = deepcopy(_scope())
    value = changed[field]
    changed[field] = not value if isinstance(value, bool) else f"{value}-changed"
    with pytest.raises(profile.ProfileContractError, match="scope profile"):
        _manifest(scope_profile=changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["request"].update({"extra": 1}),
        lambda row: row["request"].update({"enable_thinking": True}),
        lambda row: row["request"].update({"response_format": {"type": "text"}}),
        lambda row: row["request"].update({"temperature": 1}),
    ],
)
def test_base_request_all_knob_drift_is_rejected(mutation) -> None:
    packs = _packs()
    blocker = packs["blocker"]
    rows = deepcopy(blocker.requests)
    mutation(rows[0])
    changed = _replace_pack(packs, "blocker", requests=tuple(rows))
    with pytest.raises(profile.ProfileContractError, match="base request"):
        _manifest(packs=changed)


def test_profile_internal_corehash_tamper_is_rejected_even_after_top_rehash() -> None:
    manifest = _manifest()
    manifest["base_packs"]["blocker"]["full_case_count"] += 1
    manifest["execution_profile_sha256"] = profile.sha256_json(
        {key: value for key, value in manifest.items() if key != "execution_profile_sha256"}
    )
    with pytest.raises(profile.ProfileContractError, match="target receipt"):
        profile.validate_execution_profile(manifest)


def test_effective_request_rejects_base_enable_thinking_or_config_drift() -> None:
    manifest = _manifest()
    row = deepcopy(_packs()["blocker"].requests[0])
    row["request"]["enable_thinking"] = False
    with pytest.raises(profile.ProfileContractError, match="frozen base request"):
        profile.effective_request_row(row, manifest, kind="blocker")

    row = deepcopy(_packs()["blocker"].requests[0])
    row["request"]["temperature"] = 1
    with pytest.raises(profile.ProfileContractError, match="base request"):
        profile.effective_request_row(row, manifest, kind="blocker")


def test_new_calibration_has_four_fresh_ids_and_never_reuses_legacy_ids() -> None:
    manifest = _manifest()
    rows = profile.build_calibration_requests(
        _packs()["blocker"].requests[0], manifest
    )
    assert len(rows) == 4
    assert len({row["custom_id"] for row in rows}) == 4
    assert all(manifest["execution_profile_sha256"][:16] in row["custom_id"] for row in rows)
    assert all(row["request"]["enable_thinking"] is False for row in rows)
    assert all(not row["custom_id"].startswith("calibration-2.32") for row in rows)


def test_provider_driver_receives_effective_false_and_skips_old_success_and_hard() -> None:
    manifest = _manifest()
    legacy_success = _core_result("blocker-easy-1")
    runtime = _pack_runtime(
        kind="blocker",
        legacy_rows=(legacy_success,),
    )
    calls: list[tuple[str, bool]] = []

    def execute(row: dict[str, Any], calibration: bool) -> dict[str, Any]:
        calls.append((str(row["custom_id"]), bool(row["request"]["enable_thinking"])))
        return _core_result(str(row["custom_id"]))

    state = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=[],
        response_rows=[],
        calibration_response_rows=[],
        append_journal=lambda _row: None,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=lambda: "2026-08-13T00:00:00+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )

    assert state["status"] == "succeeded"
    assert calls == []
    assert state["hard_calls"] == 0


def test_started_without_result_is_fail_stop_and_restart_calls_provider_zero_times() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    row = profile.effective_request_row(
        runtime.pack.requests[0], manifest, kind="contract"
    )
    started = profile.build_started_event(
        kind="contract",
        logical_custom_id=row["custom_id"],
        effective_row=row,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    calls = 0

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _core_result("contract-medium-1")

    state = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=[started],
        response_rows=[],
        calibration_response_rows=[],
        append_journal=lambda _row: None,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=lambda: "2026-08-13T00:00:02+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )

    assert state["status"] == "fail_stop_unknown_side_effect"
    assert calls == 0


def test_result_without_completion_is_reconciled_without_provider_call() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    effective = profile.effective_request_row(
        runtime.pack.requests[0], manifest, kind="contract"
    )
    started = profile.build_started_event(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    wrapped = profile.wrap_result(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        core_result=_production_core(manifest, "contract", effective["custom_id"]),
        execution_profile=manifest,
        started_event=started,
    )
    appended: list[dict[str, Any]] = []

    state = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=[started],
        response_rows=[wrapped],
        calibration_response_rows=[],
        append_journal=appended.append,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=lambda *_args: pytest.fail("provider must not be called"),
        now=lambda: "2026-08-13T00:00:02+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )

    assert state["status"] == "reconciled_result_without_api"
    assert appended[-1]["event_type"] == "completed"
    assert appended[-1]["result_envelope_sha256"] == profile.sha256_json(wrapped)


@pytest.mark.parametrize(
    "path,value",
    [
        (("execution_profile_sha256",), "0" * 64),
        (("model",), "other-model"),
        (("endpoint",), "https://example.invalid/v1/chat/completions"),
        (("request_overlay", "enable_thinking"), True),
        (("effective_request_sha256",), "1" * 64),
        (("core_result", "usage", "total_tokens"), 31),
        (("core_result", "model"), "other-model"),
        (("core_result", "sse_completion_policy"), "loose"),
    ],
)
def test_result_receipt_tamper_fails_closed(path: tuple[str, ...], value: Any) -> None:
    manifest = _manifest()
    effective = profile.effective_request_row(
        _packs()["contract"].requests[0], manifest, kind="contract"
    )
    started = profile.build_started_event(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    wrapped = profile.wrap_result(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        core_result=_production_core(manifest, "contract", effective["custom_id"]),
        execution_profile=manifest,
        started_event=started,
    )
    target: Any = wrapped
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    errors = profile.validate_result_envelope(
        wrapped,
        kind="contract",
        request_row=effective,
        execution_profile=manifest,
        started_event=started,
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert errors


def test_foreign_hard_unknown_status_duplicate_and_good_to_bad_are_rejected() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(
        kind="blocker",
        legacy_rows=(_core_result("blocker-easy-1"),),
    )
    hard_effective = profile._effective_from_base(_packs()["blocker"].requests[1])
    with pytest.raises(profile.ProfileContractError, match="sequence"):
        profile.build_started_event(
            kind="blocker",
            logical_custom_id="blocker-hard-1",
            effective_row=hard_effective,
            execution_profile=manifest,
            started_at="2026-08-13T00:00:00+00:00",
            allow_non_target_for_test=True,
        )

    effective = profile.effective_request_row(
        _packs()["blocker"].requests[0], manifest, kind="blocker"
    )
    started = profile.build_started_event(
        kind="blocker",
        logical_custom_id="blocker-easy-1",
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    with pytest.raises(profile.ProfileContractError, match="already succeeded"):
        profile.reduce_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=[started],
            response_rows=[],
            validate_core_result=lambda _result, _row, _calibration: [],
        )
    unknown = deepcopy(started)
    unknown["event_type"] = "mystery"
    with pytest.raises(profile.ProfileContractError, match="event"):
        profile.reduce_journal(
            runtime=_pack_runtime(kind="blocker"),
            execution_profile=manifest,
            journal_rows=[unknown],
            response_rows=[],
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_summary_discloses_mixed_profiles_usage_and_zero_hard_cost() -> None:
    manifest = _manifest()
    legacy = _core_result("blocker-easy-1")
    runtime = _pack_runtime(
        kind="blocker",
        legacy_rows=(legacy,),
    )
    summary = profile.build_scope_summary(
        runtimes={"blocker": runtime, "contract": _pack_runtime(kind="contract")},
        execution_profile=manifest,
        response_rows=[],
        calibration_response_rows=[],
        journal_rows=[],
        validate_core_result_by_kind={
            kind: (lambda _result, _row, _calibration: [])
            for kind in profile.KINDS
        },
    )
    assert summary["hard_reference_only"] is True
    assert summary["hard_calls"] == 0
    assert summary["hard_usage"]["total_tokens"] == 0
    assert (
        summary["profile_counts"]["legacy_by_runner_version"][
            profile.LEGACY_RUNNER_VERSION
        ]
        == 1
    )
    assert summary["usage"]["total_tokens"] == 30


def test_summary_routes_each_production_result_to_its_pack_validator() -> None:
    manifest = _manifest()
    runtimes = {
        "blocker": _pack_runtime(kind="blocker"),
        "contract": _pack_runtime(kind="contract"),
    }
    production_journal, production_responses = _completed_production(
        manifest, runtimes["contract"]
    )
    calibration_journal, calibration_responses = _completed_calibration(
        manifest, runtimes["blocker"].pack.requests[0]
    )
    observed: list[tuple[str, str, bool]] = []

    def validator(kind: str) -> Callable[[dict[str, Any], dict[str, Any], bool], list[str]]:
        def validate(
            result: dict[str, Any], request: dict[str, Any], calibration: bool
        ) -> list[str]:
            observed.append((kind, str(result["custom_id"]), calibration))
            return []

        return validate

    profile.build_scope_summary(
        runtimes=runtimes,
        execution_profile=manifest,
        response_rows=production_responses,
        calibration_response_rows=calibration_responses,
        journal_rows=[*production_journal, *calibration_journal],
        validate_core_result_by_kind={
            "blocker": validator("blocker"),
            "contract": validator("contract"),
        },
    )

    assert ("contract", "contract-medium-1", False) in observed
    assert ("blocker", "contract-medium-1", False) not in observed
    assert {
        (kind, custom_id)
        for kind, custom_id, calibration in observed
        if not calibration
    } == {("contract", "contract-medium-1")}


def test_new_success_requires_same_profile_completed_fresh_calibration_4_of_4() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    production_journal, production_responses = _completed_production(
        manifest, runtime
    )
    with pytest.raises(profile.ProfileContractError, match="calibration 4/4"):
        profile.reduce_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=production_journal,
            response_rows=production_responses,
            calibration_response_rows=[],
            validate_core_result=lambda _result, _row, _calibration: [],
        )

    cal_journal, cal_responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )
    with pytest.raises(profile.ProfileContractError, match="calibration 4/4"):
        profile.reduce_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=[*production_journal, *cal_journal[:-2]],
            response_rows=production_responses,
            calibration_response_rows=cal_responses[:-1],
            validate_core_result=lambda _result, _row, _calibration: [],
        )

    state = profile.reduce_journal(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=[*production_journal, *cal_journal],
        response_rows=production_responses,
        calibration_response_rows=cal_responses,
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert state["new_success_custom_ids"] == ["contract-medium-1"]
    assert state["fresh_calibration_receipt"]["required_case_count"] == 4


def test_result_without_completed_is_never_authoritative() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    journal, responses = _completed_production(manifest, runtime)
    state = profile._reduce_production_journal(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=journal[:1],
        response_rows=responses,
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert state["authoritative_success_by_id"] == {}
    assert state["reconcile_custom_ids"] == ["contract-medium-1"]


def test_journal_physical_completed_before_started_is_rejected() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    journal, responses = _completed_production(manifest, runtime)
    with pytest.raises(profile.ProfileContractError, match="precedes"):
        profile._reduce_production_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=list(reversed(journal)),
            response_rows=responses,
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_calibration_started_without_result_fail_stops_before_any_call() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    cal_row = profile.build_calibration_requests(
        runtime.pack.requests[0], manifest
    )[0]
    started = profile.build_started_event(
        kind="calibration",
        logical_custom_id=cal_row["custom_id"],
        effective_row=cal_row,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:00+00:00",
    )
    calls = 0

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _core_result("never")

    result = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=[started],
        response_rows=[],
        calibration_response_rows=[],
        append_journal=lambda _row: None,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=lambda: "2026-08-13T00:00:00+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert result["status"] == "fail_stop_unknown_side_effect"
    assert result["phase"] == "calibration"
    assert calls == 0


def test_calibration_result_without_started_is_rejected() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    _journal, responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )
    with pytest.raises(profile.ProfileContractError, match="started fence"):
        profile._calibration_state(
            template=runtime.pack.requests[0],
            execution_profile=manifest,
            journal_rows=[],
            response_rows=[responses[0]],
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_production_started_unknown_has_priority_over_missing_calibration() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    production_journal, _responses = _completed_production(manifest, runtime)
    calls = 0

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _core_result("never")

    result = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=production_journal[:1],
        response_rows=[],
        calibration_response_rows=[],
        append_journal=lambda _row: None,
        append_response=lambda _row: None,
        append_calibration_response=lambda _row: None,
        execute_one=execute,
        now=lambda: "2026-08-13T00:00:00+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert result["phase"] == "contract"
    assert result["status"] == "fail_stop_unknown_side_effect"
    assert calls == 0


def test_started_append_must_succeed_before_provider_execution() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    cal_journal, cal_responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )
    calls = 0

    def fail_append(_row: dict[str, Any]) -> None:
        raise OSError("fsync append failed")

    def execute(_row: dict[str, Any], _calibration: bool) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _core_result("never")

    with pytest.raises(OSError, match="fsync"):
        profile.drive_pack(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=cal_journal,
            response_rows=[],
            calibration_response_rows=cal_responses,
            append_journal=fail_append,
            append_response=lambda _row: None,
            append_calibration_response=lambda _row: None,
            execute_one=execute,
            now=lambda: "2026-08-13T00:00:00+00:00",
            validate_core_result=lambda _result, _row, _calibration: [],
        )
    assert calls == 0


def test_normal_provider_order_is_started_result_completed_and_effective_false() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    cal_journal, cal_responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )
    actions: list[str] = []

    def append_journal(row: dict[str, Any]) -> None:
        actions.append(str(row["event_type"]))

    def append_response(_row: dict[str, Any]) -> None:
        actions.append("result")

    def execute(row: dict[str, Any], calibration: bool) -> dict[str, Any]:
        assert actions[-1] == "started"
        assert calibration is False
        assert row["request"]["enable_thinking"] is False
        return _production_core(manifest, "contract", row["custom_id"])

    result = profile.drive_pack(
        runtime=runtime,
        execution_profile=manifest,
        journal_rows=cal_journal,
        response_rows=[],
        calibration_response_rows=cal_responses,
        append_journal=append_journal,
        append_response=append_response,
        append_calibration_response=lambda _row: actions.append("calibration-result"),
        execute_one=execute,
        now=lambda: "2026-08-13T00:00:00+00:00",
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert result["status"] == "succeeded"
    assert actions == ["started", "result", "completed"]


def test_fake_legacy_replay_cannot_forge_a_success_skip() -> None:
    manifest = _manifest()
    pack = _packs()["contract"]
    physical_failed = _core_result("contract-medium-1", status="failed")

    def dishonest(_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "successful_by_id": {"contract-medium-1": _core_result("contract-medium-1")},
            "retry_eligible_by_id": {},
            "invalidated_by_id": {},
            "retry_started_by_id": {},
            "terminal_by_id": {},
            "adopted_by_id": {},
            "episode_by_id": {},
        }

    runtime = profile.PackRuntime(
        kind="contract",
        pack=pack,
        legacy_rows=(physical_failed,),
        replay_legacy=dishonest,
    )
    with pytest.raises(profile.ProfileContractError, match="physical legacy rows"):
        profile._reduce_production_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=[],
            response_rows=[],
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_global_preflight_rejects_cross_pack_corehash_profile_tamper() -> None:
    manifest = _manifest()
    tampered = deepcopy(manifest)
    tampered["base_packs"]["contract"]["base_execution_contract_sha256"] = "d" * 64
    tampered["execution_profile_sha256"] = profile.sha256_json(
        {key: value for key, value in tampered.items() if key != "execution_profile_sha256"}
    )
    runtimes = {
        "blocker": _pack_runtime(kind="blocker"),
        "contract": _pack_runtime(kind="contract"),
    }
    with pytest.raises(profile.ProfileContractError, match="actual pack/profile"):
        profile.preflight_execution(
            runtimes=runtimes,
            execution_profile=tampered,
            journal_rows=[],
            response_rows=[],
            calibration_response_rows=[],
            validate_core_result_by_kind={
                kind: (lambda _result, _row, _calibration: [])
                for kind in profile.KINDS
            },
        )


def test_materializer_validator_failure_prevents_new_authoritative_success() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    production_journal, production_responses = _completed_production(
        manifest, runtime
    )
    cal_journal, cal_responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )

    def validator(
        _result: dict[str, Any], _row: dict[str, Any], calibration: bool
    ) -> list[str]:
        return [] if calibration else ["materializer rejected review"]

    with pytest.raises(profile.ProfileContractError, match="materializer"):
        profile.reduce_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=[*production_journal, *cal_journal],
            response_rows=production_responses,
            calibration_response_rows=cal_responses,
            validate_core_result=validator,
        )


def test_journal_timestamps_must_be_utc_and_monotonic() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    effective = profile.effective_request_row(
        runtime.pack.requests[0], manifest, kind="contract"
    )
    with pytest.raises(profile.ProfileContractError, match="UTC"):
        profile.build_started_event(
            kind="contract",
            logical_custom_id=effective["custom_id"],
            effective_row=effective,
            execution_profile=manifest,
            started_at="2026-08-13T00:00:00+08:00",
        )

    started = profile.build_started_event(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        execution_profile=manifest,
        started_at="2026-08-13T00:00:02+00:00",
    )
    core = _production_core(manifest, "contract", effective["custom_id"])
    core["completed_at"] = "2026-08-13T00:00:01+00:00"
    envelope = profile.wrap_result(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        core_result=core,
        execution_profile=manifest,
        started_event=started,
    )
    completed = profile.build_completed_event(
        started_event=started, result_envelope=envelope
    )
    with pytest.raises(profile.ProfileContractError, match="precedes"):
        profile._reduce_production_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=[started, completed],
            response_rows=[envelope],
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_success_then_bad_duplicate_result_is_rejected_fail_closed() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    journal, responses = _completed_production(manifest, runtime)
    success = responses[0]
    effective = profile.effective_request_row(
        runtime.pack.requests[0], manifest, kind="contract"
    )
    bad = profile.wrap_result(
        kind="contract",
        logical_custom_id=effective["custom_id"],
        effective_row=effective,
        core_result=_production_core(
            manifest, "contract", effective["custom_id"], status="failed"
        ),
        execution_profile=manifest,
        started_event=journal[0],
    )
    with pytest.raises(profile.ProfileContractError, match="duplicate result"):
        profile._reduce_production_journal(
            runtime=runtime,
            execution_profile=manifest,
            journal_rows=journal,
            response_rows=[success, bad],
            validate_core_result=lambda _result, _row, _calibration: [],
        )


def test_calibration_result_without_completion_is_reconcile_not_success() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    journal, responses = _completed_calibration(
        manifest, runtime.pack.requests[0]
    )
    state = profile._calibration_state(
        template=runtime.pack.requests[0],
        execution_profile=manifest,
        journal_rows=journal[:1],
        response_rows=responses[:1],
        validate_core_result=lambda _result, _row, _calibration: [],
    )
    assert state["succeeded_custom_ids"] == []
    assert state["reconcile_custom_ids"] == [responses[0]["logical_custom_id"]]


def test_legacy_calibration_rows_cannot_satisfy_fresh_profile() -> None:
    manifest = _manifest()
    runtime = _pack_runtime(kind="contract")
    old_rows = profile.legacy._read_jsonl(profile.legacy.CALIBRATION_RESPONSES_PATH)
    assert old_rows
    with pytest.raises(profile.ProfileContractError, match="foreign kind"):
        profile._calibration_state(
            template=runtime.pack.requests[0],
            execution_profile=manifest,
            journal_rows=[],
            response_rows=[old_rows[-1]],
            validate_core_result=lambda _result, _row, _calibration: [],
        )
