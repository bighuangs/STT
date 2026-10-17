"""运行独立的 Easy/Medium ``enable_thinking=false`` 复审 profile。

默认只做只读 dry-run。付费执行必须显式提供全部 gate，并复用旧执行锁；所有
新 profile、journal、response、calibration 与 summary 均写入独立目录，永不
追加旧 2.32 正式账本。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import easy_medium_nonthinking_profile as profile
import easy_medium_scope as scope
import review_blocker_rereview as blocker
import review_contract_rereview as contract
import review_contracts as legacy
import review_release_integrity as integrity


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "easy-medium-nonthinking-v6"
PROFILE_PATH = OUTPUT_ROOT / "execution-profile.json"
JOURNAL_PATH = OUTPUT_ROOT / "journal.jsonl"
RESPONSES_PATH = OUTPUT_ROOT / "responses.jsonl"
CALIBRATION_RESPONSES_PATH = OUTPUT_ROOT / "calibration-responses.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "execution-summary.json"
PREDECESSOR_ROOT = ROOT / "easy-medium-nonthinking"
PREDECESSOR_PROFILE_PATH = PREDECESSOR_ROOT / "execution-profile.json"
PREDECESSOR_JOURNAL_PATH = PREDECESSOR_ROOT / "journal.jsonl"
PREDECESSOR_RESPONSES_PATH = PREDECESSOR_ROOT / "responses.jsonl"
PREDECESSOR_CALIBRATION_RESPONSES_PATH = (
    PREDECESSOR_ROOT / "calibration-responses.jsonl"
)
PREDECESSOR_SUMMARY_PATH = PREDECESSOR_ROOT / "execution-summary.json"
PREDECESSOR_FILE_SHA256 = {
    "execution-profile.json": "3823a4c4f148494f2e5881191defb6dc0324c1218a6d3af1a8402d5b92b70934",
    "journal.jsonl": "e56129388add1e450b89927b54ff196b6510923e58625b7546578ae0619bed17",
    "responses.jsonl": "32b9836f403190ee565bf505e513fc939197073f1628dd920c4c33dbfb3d7bdb",
    "calibration-responses.jsonl": "41b22da80dbbeff18d29a3139ccbc2686c39c0153aef798b8518f42371c1a35d",
    "execution-summary.json": "fb8cb9980d0a7ec0b50ec69e297390d27353a3fab01f5a97ba9c21ec706bd570",
}
V4_ROOT = ROOT / "easy-medium-nonthinking-v4"
V4_PROFILE_PATH = V4_ROOT / "execution-profile.json"
V4_JOURNAL_PATH = V4_ROOT / "journal.jsonl"
V4_RESPONSES_PATH = V4_ROOT / "responses.jsonl"
V4_CALIBRATION_RESPONSES_PATH = V4_ROOT / "calibration-responses.jsonl"
V4_FILE_SHA256 = {
    "execution-profile.json": "83aa9afe4146668e4257c9ba0d5bf5e676c0a35159877d5306dbec54af1d8475",
    "journal.jsonl": "b2721962bc5f80b14272e925549b07a25769bfceb9b4c0daa366225302058230",
    "responses.jsonl": "5eff141d15511c293e0f7e199e0c17568f3cb006aecc2f5f7368fd04afe8adc2",
    "calibration-responses.jsonl": "6454d8347e3e08122ff1c0bbb0552af281a132efc3d2d2005f056f0bf1b963d3",
}
V5_ROOT = ROOT / "easy-medium-nonthinking-v5"
V5_PROFILE_PATH = V5_ROOT / "execution-profile.json"
V5_JOURNAL_PATH = V5_ROOT / "journal.jsonl"
V5_RESPONSES_PATH = V5_ROOT / "responses.jsonl"
V5_CALIBRATION_RESPONSES_PATH = V5_ROOT / "calibration-responses.jsonl"
V5_SUMMARY_PATH = V5_ROOT / "execution-summary.json"
V5_FILE_SHA256 = {
    "execution-profile.json": "468a974b81aa5bef16c86edf97e031fcb4c15e39ce744aade3c4f6caa9363c0d",
    "journal.jsonl": "5332ef98ba2e1f8a0332101d627a47e0b7771d80478e7451ae85fa14f1d603ca",
    "responses.jsonl": "b31012b47e29facd6873885618e2560cbda3d1eee10c8b68f18a947a9ad4ef5a",
    "calibration-responses.jsonl": "ca53fcd08ac49fe8c479991315f90bcc273f653066e5ec24830d83cafec24113",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise profile.ProfileContractError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return legacy._read_jsonl(path)


def _write_json_fsync(path: Path, value: dict[str, Any]) -> None:
    """用临时文件+replace持久化独立快照，并在replace前fsync。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _initialize_output_bundle(execution_profile: dict[str, Any]) -> None:
    """在首次started fence前持久化profile及三个独立append-only文件。"""

    if not PROFILE_PATH.exists():
        _write_json_fsync(PROFILE_PATH, execution_profile)
    for path in (
        JOURNAL_PATH,
        RESPONSES_PATH,
        CALIBRATION_RESPONSES_PATH,
    ):
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.flush()
            os.fsync(handle.fileno())


def _pack_context(kind: str) -> dict[str, Any]:
    """从两个现有wrapper的真实校验入口构造冻结pack与formal reducer。"""

    if kind == "blocker":
        wrapper = blocker
        requests, _expected = blocker._validate_requests()
        request_map = json.loads(blocker.REQUEST_MAP_PATH.read_text(encoding="utf-8"))
        if not isinstance(request_map, list):
            raise profile.ProfileContractError("blocker request-map must be an array")
        authority = blocker._scope_authority(request_map)
    elif kind == "contract":
        wrapper = contract
        requests, _expected = contract._validate_requests()
        request_map = json.loads(contract.REQUEST_MAP_PATH.read_text(encoding="utf-8"))
        if not isinstance(request_map, list):
            raise profile.ProfileContractError("contract request-map must be an array")
        authority = contract._scope_authority(requests)
    else:  # pragma: no cover - internal closed enum
        raise profile.ProfileContractError("unknown pack kind")
    prepared = _read_json(wrapper.REREVIEW_ROOT / "summary.json")
    execution_contract_sha256 = str(
        prepared.get("execution_contract_sha256") or ""
    )
    physical_rows = legacy._read_jsonl(wrapper.RESPONSES_PATH)
    pack = profile.PackInput(
        requests=tuple(deepcopy(requests)),
        request_map=tuple(deepcopy(request_map)),
        authority_rows=tuple(deepcopy(authority)),
        base_execution_contract_sha256=execution_contract_sha256,
    )

    def replay(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
        return wrapper._formal_state(
            rows,
            requests,
            execution_contract_sha256=execution_contract_sha256,
            label=f"easy-medium nonthinking {kind} legacy replay",
        )

    runtime = profile.PackRuntime(
        kind=kind,
        pack=pack,
        legacy_rows=tuple(deepcopy(physical_rows)),
        replay_legacy=replay,
    )
    return {
        "kind": kind,
        "wrapper": wrapper,
        "requests": requests,
        "runtime": runtime,
    }


def load_runtimes() -> dict[str, profile.PackRuntime]:
    """加载并严格校验两个真实full pack及其旧物理账本。"""

    contexts = {kind: _pack_context(kind) for kind in profile.KINDS}
    return {kind: contexts[kind]["runtime"] for kind in profile.KINDS}


def _load_predecessor_bridge(
    runtimes: dict[str, profile.PackRuntime],
) -> tuple[dict[str, profile.PackRuntime], dict[str, Any] | None]:
    """严格重放现存 v1；只桥接 completed success，failed 仅作为投影证据。"""

    paths = (
        PREDECESSOR_PROFILE_PATH,
        PREDECESSOR_JOURNAL_PATH,
        PREDECESSOR_RESPONSES_PATH,
        PREDECESSOR_CALIBRATION_RESPONSES_PATH,
        PREDECESSOR_SUMMARY_PATH,
    )
    existing = [path for path in paths if path.exists()]
    if not existing:
        return runtimes, None
    if len(existing) != len(paths):
        raise profile.ProfileContractError("partial predecessor v1 sidecar set")
    for path in paths:
        if _sidecar_file_sha256(path) != PREDECESSOR_FILE_SHA256[path.name]:
            raise profile.ProfileContractError(
                f"predecessor v1 physical sidecar hash drifted: {path.name}"
            )
    predecessor_profile = _read_json(PREDECESSOR_PROFILE_PATH)
    journal = _read_jsonl(PREDECESSOR_JOURNAL_PATH)
    responses = _read_jsonl(PREDECESSOR_RESPONSES_PATH)
    calibrations = _read_jsonl(PREDECESSOR_CALIBRATION_RESPONSES_PATH)
    successful_envelopes: dict[str, dict[str, Any]] = {}
    failed_envelopes: dict[str, dict[str, Any]] = {}
    projected_success: dict[str, dict[str, Any]] = {}
    for envelope in responses:
        kind = str(envelope.get("kind") or "")
        custom_id = str(envelope.get("logical_custom_id") or "")
        if kind not in profile.KINDS:
            raise profile.ProfileContractError("predecessor response kind is foreign")
        runtime = runtimes[kind]
        if custom_id not in profile._selection(runtime.pack).target_custom_ids:
            raise profile.ProfileContractError("predecessor response targets non-E/M id")
        base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
        predecessor_effective = deepcopy(base)
        predecessor_effective["request"]["enable_thinking"] = False
        started = next(
            (
                event
                for event in journal
                if event.get("event_type") == "started"
                and event.get("operation_id") == envelope.get("operation_id")
            ),
            None,
        )
        completed = next(
            (
                event
                for event in journal
                if event.get("event_type") == "completed"
                and event.get("operation_id") == envelope.get("operation_id")
            ),
            None,
        )
        if not isinstance(started, dict) or not isinstance(completed, dict):
            raise profile.ProfileContractError("predecessor response is not completed")
        core = deepcopy(envelope.get("core_result") or {})
        validator = core_validator(kind=kind, execution_profile={
            **predecessor_profile,
            "base_packs": predecessor_profile["base_packs"],
        })
        errors = list(validator(core, predecessor_effective, False))
        if errors:
            raise profile.ProfileContractError(
                "predecessor core strict replay failed: " + "; ".join(errors)
            )
        if core.get("status") == "succeeded":
            successful_envelopes[custom_id] = envelope
            projected_success[custom_id] = profile._project_predecessor_success(
                envelope
            )
        elif core.get("status") == "failed":
            failed_envelopes[custom_id] = envelope
        else:
            raise profile.ProfileContractError("predecessor production status is invalid")
    receipt = profile.predecessor_bridge_receipt(
        predecessor_profile=predecessor_profile,
        predecessor_journal_rows=journal,
        predecessor_response_rows=responses,
        predecessor_calibration_rows=calibrations,
        successful_envelope_by_id=successful_envelopes,
        failed_envelope_by_id=failed_envelopes,
        physical_sidecar_sha256={
            path.name: _sidecar_file_sha256(path) for path in paths
        },
    )
    bridged = {}
    for kind, runtime in runtimes.items():
        success_rows = tuple(
            deepcopy(projected_success[custom_id])
            for custom_id in profile._selection(runtime.pack).target_custom_ids
            if custom_id in projected_success
        )
        failed_rows = tuple(
            deepcopy(failed_envelopes[custom_id])
            for custom_id in profile._selection(runtime.pack).target_custom_ids
            if custom_id in failed_envelopes
        )
        source_envelopes = tuple(
            deepcopy(successful_envelopes[custom_id])
            for custom_id in profile._selection(runtime.pack).target_custom_ids
            if custom_id in successful_envelopes
        )
        source_cores = tuple(
            deepcopy(envelope["core_result"])
            for envelope in (*source_envelopes, *failed_rows)
        )
        bridged[kind] = profile.PackRuntime(
            kind=runtime.kind,
            pack=runtime.pack,
            legacy_rows=runtime.legacy_rows,
            replay_legacy=runtime.replay_legacy,
            predecessor_bridge=profile.PredecessorBridge(
                receipt=deepcopy(receipt),
                success_rows=success_rows,
                success_envelopes=source_envelopes,
                failed_envelopes=failed_rows,
                source_core_rows=source_cores,
                calibration_core_rows=tuple(
                    deepcopy(envelope["core_result"])
                    for envelope in calibrations
                ),
                completed_events=tuple(
                    deepcopy(event)
                    for event in journal
                    if event.get("event_type") == "completed"
                ),
                validate_predecessor_core=core_validator(
                    kind=kind,
                    execution_profile={
                        **predecessor_profile,
                        "base_packs": predecessor_profile["base_packs"],
                    },
                ),
            ),
        )
    return _load_v5_adoption_bridge(_load_v4_adoption_bridge(bridged)), receipt


def _load_v4_adoption_bridge(
    runtimes: dict[str, profile.PackRuntime],
) -> dict[str, profile.PackRuntime]:
    """精确重放v4两个已付费结果及4/4 calibration，不修改v4账本。"""

    paths = (
        V4_PROFILE_PATH,
        V4_JOURNAL_PATH,
        V4_RESPONSES_PATH,
        V4_CALIBRATION_RESPONSES_PATH,
    )
    if any(not path.exists() for path in paths):
        raise profile.ProfileContractError("v5 requires the complete pinned v4 source")
    for path in paths:
        if _sidecar_file_sha256(path) != V4_FILE_SHA256[path.name]:
            raise profile.ProfileContractError(
                f"v4 adoption physical sidecar hash drifted: {path.name}"
            )
    source_profile = _read_json(V4_PROFILE_PATH)
    journal = _read_jsonl(V4_JOURNAL_PATH)
    responses = _read_jsonl(V4_RESPONSES_PATH)
    calibrations = _read_jsonl(V4_CALIBRATION_RESPONSES_PATH)
    if (
        source_profile.get("profile_id") != profile.V4_PROFILE_ID
        or source_profile.get("runner_version") != profile.V4_PROFILE_RUNNER_VERSION
        or source_profile.get("execution_profile_sha256")
        != profile.V4_PROFILE_SHA256
        or profile.sha256_json(
            {
                key: value
                for key, value in source_profile.items()
                if key != "execution_profile_sha256"
            }
        )
        != profile.V4_PROFILE_SHA256
    ):
        raise profile.ProfileContractError("v4 adoption source profile is invalid")
    expected_ids = {
        "concept-contract-rereview-52d20717a579d2f585a6",
        "concept-contract-rereview-056e9d3e6c713db68790",
    }
    if {
        str(row.get("logical_custom_id") or "") for row in responses
    } != expected_ids or len(responses) != 2:
        raise profile.ProfileContractError("v4 adoption production set is not exact")
    starts = {
        str(row.get("operation_id") or ""): row
        for row in journal
        if row.get("event_type") == "started"
    }
    completions = {
        str(row.get("operation_id") or ""): row
        for row in journal
        if row.get("event_type") == "completed"
    }
    projections: dict[str, dict[str, Any]] = {}
    source_started: list[dict[str, Any]] = []
    source_completed: list[dict[str, Any]] = []
    for envelope in responses:
        custom_id = str(envelope["logical_custom_id"])
        kind = str(envelope.get("kind") or "")
        if kind != "contract":
            raise profile.ProfileContractError("v4 adopted result kind is invalid")
        runtime = runtimes[kind]
        base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
        effective = profile._effective_from_base(base)
        operation_id = str(envelope.get("operation_id") or "")
        started = starts.get(operation_id)
        completed = completions.get(operation_id)
        core = envelope.get("core_result")
        old_validator = core_validator(
            kind=kind,
            execution_profile=source_profile,
            materialization_policy_version=(
                profile.LEGACY_MATERIALIZATION_POLICY_VERSION
            ),
        )
        new_validator = core_validator(
            kind=kind,
            execution_profile=source_profile,
            materialization_policy_version=profile.MATERIALIZATION_POLICY_VERSION,
        )
        old_errors = (
            ["v4 adopted core is not an object"]
            if not isinstance(core, dict)
            else list(old_validator(deepcopy(core), effective, False))
        )
        new_errors = (
            ["v4 adopted core is not an object"]
            if not isinstance(core, dict)
            else list(new_validator(deepcopy(core), effective, False))
        )
        if (
            not isinstance(started, dict)
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v4"
            or envelope.get("execution_profile_sha256")
            != profile.V4_PROFILE_SHA256
            or envelope.get("profile_runner_version")
            != profile.V4_PROFILE_RUNNER_VERSION
            or envelope.get("started_event_sha256")
            != profile.sha256_json(started)
            or envelope.get("core_result_sha256") != profile.sha256_json(core)
            or envelope.get("base_request_sha256") != profile.sha256_json(base)
            or envelope.get("effective_request_sha256")
            != profile.sha256_json(effective)
            or old_errors != profile.V4_ADOPTION_OLD_ERRORS_BY_ID[custom_id]
            or new_errors
        ):
            raise profile.ProfileContractError(
                "v4 adopted production source strict replay failed"
                + (": " + "; ".join(new_errors) if new_errors else "")
            )
        if completed is not None and completed.get(
            "result_envelope_sha256"
        ) != profile.sha256_json(envelope):
            raise profile.ProfileContractError("v4 completed/result binding drifted")
        source_started.append(deepcopy(started))
        if completed is not None:
            source_completed.append(deepcopy(completed))
        projections[custom_id] = profile._project_v4_adopted_success(
            envelope, started_event=started, completed_event=completed
        )
    if set(row["logical_custom_id"] for row in source_completed) != {
        "concept-contract-rereview-52d20717a579d2f585a6"
    }:
        raise profile.ProfileContractError(
            "v4 adoption must preserve one completed and one result-only source"
        )

    calibration_requests = {
        row["custom_id"]: row
        for row in _build_v4_calibration_requests(
            runtimes["blocker"].pack.requests[0], source_profile
        )
    }
    calibration_base_requests: dict[str, dict[str, Any]] = {}
    calibration_template = runtimes["blocker"].pack.requests[0]
    for custom_id, request in calibration_requests.items():
        base_request = deepcopy(request)
        base_request["request"]["messages"][0] = deepcopy(
            calibration_template["request"]["messages"][0]
        )
        base_request["prompt_sha256"] = calibration_template["prompt_sha256"]
        base_request["request"].pop("enable_thinking", None)
        calibration_base_requests[custom_id] = base_request
    if len(calibrations) != 4 or {
        str(row.get("logical_custom_id") or "") for row in calibrations
    } != set(calibration_requests):
        raise profile.ProfileContractError("v4 calibration migration is not exact 4/4")
    calibration_started: list[dict[str, Any]] = []
    calibration_completed: list[dict[str, Any]] = []
    calibration_validator = core_validator(
        kind="blocker",
        execution_profile=source_profile,
        materialization_policy_version=(
            profile.LEGACY_MATERIALIZATION_POLICY_VERSION
        ),
    )
    for envelope in calibrations:
        custom_id = str(envelope["logical_custom_id"])
        request = calibration_requests[custom_id]
        operation_id = str(envelope.get("operation_id") or "")
        started = starts.get(operation_id)
        completed = completions.get(operation_id)
        core = envelope.get("core_result")
        errors = (
            ["v4 calibration core is not an object"]
            if not isinstance(core, dict)
            else list(calibration_validator(deepcopy(core), request, True))
        )
        if (
            not isinstance(started, dict)
            or not isinstance(completed, dict)
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v4"
            or envelope.get("execution_profile_sha256")
            != profile.V4_PROFILE_SHA256
            or envelope.get("started_event_sha256")
            != profile.sha256_json(started)
            or envelope.get("core_result_sha256") != profile.sha256_json(core)
            or started.get("base_request_sha256")
            != profile.sha256_json(calibration_base_requests[custom_id])
            or started.get("effective_request_sha256")
            != profile.sha256_json(request)
            or envelope.get("base_request_sha256")
            != profile.sha256_json(calibration_base_requests[custom_id])
            or envelope.get("effective_request_sha256")
            != profile.sha256_json(request)
            or completed.get("result_envelope_sha256")
            != profile.sha256_json(envelope)
            or core.get("status") not in profile.SUCCESS_STATUSES
            or errors
        ):
            raise profile.ProfileContractError(
                "v4 calibration migration strict replay failed"
                + (": " + "; ".join(errors) if errors else "")
            )
        calibration_started.append(deepcopy(started))
        calibration_completed.append(deepcopy(completed))
    receipt = profile.v4_adoption_bridge_receipt(
        source_profile=source_profile,
        source_envelopes=responses,
        source_started_events=source_started,
        source_completed_events=source_completed,
        calibration_envelopes=calibrations,
        calibration_base_requests=[
            calibration_base_requests[custom_id]
            for custom_id in calibration_requests
        ],
        calibration_requests=[
            calibration_requests[custom_id] for custom_id in calibration_requests
        ],
        calibration_started_events=calibration_started,
        calibration_completed_events=calibration_completed,
        physical_sidecar_sha256={
            path.name: _sidecar_file_sha256(path) for path in paths
        },
        old_materialization_errors_by_id=profile.V4_ADOPTION_OLD_ERRORS_BY_ID,
        new_materialization_errors_by_id={custom_id: [] for custom_id in expected_ids},
    )
    bridged: dict[str, profile.PackRuntime] = {}
    for kind, runtime in runtimes.items():
        kind_ids = set(profile._selection(runtime.pack).target_custom_ids)
        envelopes = tuple(
            deepcopy(row)
            for row in responses
            if str(row["logical_custom_id"]) in kind_ids
        )
        rows = tuple(
            deepcopy(projections[str(row["logical_custom_id"])]) for row in envelopes
        )
        started_events = tuple(
            deepcopy(starts[str(row["operation_id"])]) for row in envelopes
        )
        completed_events = tuple(
            deepcopy(completions[str(row["operation_id"])])
            for row in envelopes
            if str(row["operation_id"]) in completions
        )
        bridged[kind] = profile.PackRuntime(
            kind=runtime.kind,
            pack=runtime.pack,
            legacy_rows=runtime.legacy_rows,
            replay_legacy=runtime.replay_legacy,
            predecessor_bridge=runtime.predecessor_bridge,
            v4_adoption_bridge=profile.V4AdoptionBridge(
                receipt=deepcopy(receipt),
                source_profile=deepcopy(source_profile),
                success_rows=rows,
                source_envelopes=envelopes,
                source_started_events=started_events,
                source_completed_events=completed_events,
                source_core_rows=tuple(
                    deepcopy(row["core_result"]) for row in envelopes
                ),
                calibration_envelopes=tuple(deepcopy(row) for row in calibrations),
                calibration_base_requests=tuple(
                    deepcopy(calibration_base_requests[custom_id])
                    for custom_id in calibration_requests
                ),
                calibration_requests=tuple(
                    deepcopy(calibration_requests[custom_id])
                    for custom_id in calibration_requests
                ),
                calibration_started_events=tuple(calibration_started),
                calibration_completed_events=tuple(calibration_completed),
                validate_v4_source_core=core_validator(
                    kind=kind,
                    execution_profile=source_profile,
                    materialization_policy_version=(
                        profile.LEGACY_MATERIALIZATION_POLICY_VERSION
                    ),
                ),
                validate_v5_adoption_core=core_validator(
                    kind=kind,
                    execution_profile=source_profile,
                    materialization_policy_version=(
                        profile.MATERIALIZATION_POLICY_VERSION
                    ),
                ),
            ),
        )
    return bridged


def _load_v5_adoption_bridge(
    runtimes: dict[str, profile.PackRuntime],
) -> dict[str, profile.PackRuntime]:
    """精确重放v5的一个result-only结果及4/4 calibration。"""

    paths = (
        V5_PROFILE_PATH,
        V5_JOURNAL_PATH,
        V5_RESPONSES_PATH,
        V5_CALIBRATION_RESPONSES_PATH,
    )
    if V5_SUMMARY_PATH.exists():
        raise profile.ProfileContractError(
            "v5 adoption requires execution-summary.json to be physically absent"
        )
    if any(not path.exists() for path in paths):
        raise profile.ProfileContractError("v6 requires the complete pinned v5 source")
    for path in paths:
        if _sidecar_file_sha256(path) != V5_FILE_SHA256[path.name]:
            raise profile.ProfileContractError(
                f"v5 adoption physical sidecar hash drifted: {path.name}"
            )

    source_profile = _read_json(V5_PROFILE_PATH)
    journal = _read_jsonl(V5_JOURNAL_PATH)
    responses = _read_jsonl(V5_RESPONSES_PATH)
    calibrations = _read_jsonl(V5_CALIBRATION_RESPONSES_PATH)
    if (
        source_profile.get("profile_id") != profile.V5_PROFILE_ID
        or source_profile.get("runner_version") != profile.V5_PROFILE_RUNNER_VERSION
        or source_profile.get("execution_profile_sha256")
        != profile.V5_PROFILE_SHA256
        or profile.sha256_json(
            {
                key: value
                for key, value in source_profile.items()
                if key != "execution_profile_sha256"
            }
        )
        != profile.V5_PROFILE_SHA256
    ):
        raise profile.ProfileContractError("v5 adoption source profile is invalid")
    if (
        len(journal) != 9
        or len(responses) != 1
        or len(calibrations) != 4
        or set(responses[0]) != profile.V5_RESULT_ENVELOPE_KEYS
        or any(
            set(envelope) != profile.V5_RESULT_ENVELOPE_KEYS
            for envelope in calibrations
        )
        or str(responses[0].get("logical_custom_id") or "")
        != profile.V5_ADOPTION_RESULT_ONLY_ID
    ):
        raise profile.ProfileContractError("v5 physical source row counts/IDs are not exact")

    started_rows = [row for row in journal if row.get("event_type") == "started"]
    completed_rows = [row for row in journal if row.get("event_type") == "completed"]
    if (
        len(started_rows) != 5
        or len(completed_rows) != 4
        or len({str(row.get("operation_id") or "") for row in started_rows}) != 5
        or len({str(row.get("logical_custom_id") or "") for row in started_rows}) != 5
        or len({str(row.get("operation_id") or "") for row in completed_rows}) != 4
        or any(
            row.get("event_type") not in {"started", "completed"}
            or row.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or set(row)
            != (
                profile.V5_STARTED_EVENT_KEYS
                if row.get("event_type") == "started"
                else profile.V5_COMPLETED_EVENT_KEYS
            )
            for row in journal
        )
    ):
        raise profile.ProfileContractError("v5 journal event set is not exact 9 rows")
    starts = {str(row["operation_id"]): row for row in started_rows}
    completions = {str(row["operation_id"]): row for row in completed_rows}

    source_envelope = responses[0]
    custom_id = profile.V5_ADOPTION_RESULT_ONLY_ID
    if source_envelope.get("kind") != "contract":
        raise profile.ProfileContractError("v5 adopted result kind is invalid")
    runtime = runtimes["contract"]
    target_ids = set(profile._selection(runtime.pack).target_custom_ids)
    if custom_id not in target_ids:
        raise profile.ProfileContractError("v5 adopted result is outside E/M authority")
    base = next(row for row in runtime.pack.requests if row["custom_id"] == custom_id)
    effective = profile._effective_from_base(base)
    operation_id = str(source_envelope.get("operation_id") or "")
    started = starts.get(operation_id)
    core = source_envelope.get("core_result")
    old_validator = core_validator(
        kind="contract",
        execution_profile=source_profile,
        materialization_policy_version=profile.V5_MATERIALIZATION_POLICY_VERSION,
    )
    new_validator = core_validator(
        kind="contract",
        execution_profile=source_profile,
        materialization_policy_version=profile.MATERIALIZATION_POLICY_VERSION,
    )
    old_errors = (
        ["v5 adopted core is not an object"]
        if not isinstance(core, dict)
        else list(old_validator(deepcopy(core), effective, False))
    )
    new_errors = (
        ["v5 adopted core is not an object"]
        if not isinstance(core, dict)
        else list(new_validator(deepcopy(core), effective, False))
    )
    source_contract_profile = source_profile["base_packs"]["contract"]
    expected_production_sequence = source_contract_profile[
        "target_sequence_by_custom_id"
    ].get(custom_id)
    if (
        not isinstance(started, dict)
        or operation_id in completions
        or source_envelope.get("event_type") != "result"
        or source_envelope.get("kind") != "contract"
        or source_envelope.get("logical_custom_id") != custom_id
        or source_envelope.get("schema_version")
        != "stt.eval.easy-medium-nonthinking-result.v5"
        or started.get("schema_version")
        != "stt.eval.easy-medium-nonthinking-journal.v5"
        or started.get("event_type") != "started"
        or started.get("kind") != "contract"
        or started.get("logical_custom_id") != custom_id
        or started.get("operation_id") != operation_id
        or source_envelope.get("execution_profile_sha256")
        != profile.V5_PROFILE_SHA256
        or started.get("execution_profile_sha256")
        != profile.V5_PROFILE_SHA256
        or source_envelope.get("profile_runner_version")
        != profile.V5_PROFILE_RUNNER_VERSION
        or source_envelope.get("model") != source_profile.get("model")
        or source_envelope.get("endpoint") != source_profile.get("endpoint")
        or source_envelope.get("transport") != source_profile.get("transport")
        or source_envelope.get("sse_completion_policy")
        != source_profile.get("sse_completion_policy")
        or source_envelope.get("request_overlay")
        != source_profile.get("request_overlay")
        or type(expected_production_sequence) is not int
        or source_envelope.get("sequence") != expected_production_sequence
        or started.get("sequence") != expected_production_sequence
        or source_envelope.get("started_event_sha256")
        != profile.sha256_json(started)
        or source_envelope.get("core_result_sha256") != profile.sha256_json(core)
        or started.get("base_execution_contract_sha256")
        != source_contract_profile.get("base_execution_contract_sha256")
        or source_envelope.get("base_execution_contract_sha256")
        != source_contract_profile.get("base_execution_contract_sha256")
        or started.get("base_request_sha256") != profile.sha256_json(base)
        or started.get("effective_request_sha256")
        != profile.sha256_json(effective)
        or source_envelope.get("base_request_sha256") != profile.sha256_json(base)
        or source_envelope.get("effective_request_sha256")
        != profile.sha256_json(effective)
        or old_errors != profile.V5_ADOPTION_OLD_ERRORS_BY_ID[custom_id]
        or new_errors
    ):
        raise profile.ProfileContractError(
            "v5 adopted production source strict replay failed"
            + (": " + "; ".join(new_errors) if new_errors else "")
        )
    projection = profile._project_v5_adopted_success(
        source_envelope, started_event=started
    )

    calibration_request_rows = _build_v4_calibration_requests(
        runtimes["blocker"].pack.requests[0], source_profile
    )
    if (
        len(calibration_request_rows) != 4
        or len({str(row.get("custom_id") or "") for row in calibration_request_rows})
        != 4
    ):
        raise profile.ProfileContractError("v5 calibration requests are not exact 4/4")
    calibration_requests = {
        str(row["custom_id"]): row for row in calibration_request_rows
    }
    calibration_template = runtimes["blocker"].pack.requests[0]
    calibration_base_rows: list[dict[str, Any]] = []
    for request in calibration_request_rows:
        base_request = deepcopy(request)
        base_request["request"]["messages"][0] = deepcopy(
            calibration_template["request"]["messages"][0]
        )
        base_request["prompt_sha256"] = calibration_template["prompt_sha256"]
        base_request["request"].pop("enable_thinking", None)
        calibration_base_rows.append(base_request)
    if len({str(row["custom_id"]) for row in calibration_base_rows}) != 4:
        raise profile.ProfileContractError("v5 calibration base requests are not unique")
    calibration_base_requests = {
        str(row["custom_id"]): row for row in calibration_base_rows
    }
    calibration_ids = [
        str(row.get("logical_custom_id") or "") for row in calibrations
    ]
    if len(set(calibration_ids)) != 4 or set(calibration_ids) != set(
        calibration_requests
    ):
        raise profile.ProfileContractError("v5 calibration responses are not exact 4/4")

    calibration_started: list[dict[str, Any]] = []
    calibration_completed: list[dict[str, Any]] = []
    calibration_validator = core_validator(
        kind="blocker",
        execution_profile=source_profile,
        materialization_policy_version=profile.V5_MATERIALIZATION_POLICY_VERSION,
    )
    for envelope in calibrations:
        calibration_id = str(envelope["logical_custom_id"])
        request = calibration_requests[calibration_id]
        expected_sequence = int(calibration_id.split("-")[3])
        operation = str(envelope.get("operation_id") or "")
        calibration_start = starts.get(operation)
        calibration_complete = completions.get(operation)
        calibration_core = envelope.get("core_result")
        errors = (
            ["v5 calibration core is not an object"]
            if not isinstance(calibration_core, dict)
            else list(
                calibration_validator(deepcopy(calibration_core), request, True)
            )
        )
        if (
            not isinstance(calibration_start, dict)
            or not isinstance(calibration_complete, dict)
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v5"
            or calibration_start.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or calibration_complete.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or envelope.get("event_type") != "result"
            or calibration_start.get("event_type") != "started"
            or calibration_complete.get("event_type") != "completed"
            or any(
                row.get("kind") != "calibration"
                or row.get("logical_custom_id") != calibration_id
                or row.get("operation_id") != operation
                for row in (calibration_start, envelope, calibration_complete)
            )
            or envelope.get("execution_profile_sha256")
            != profile.V5_PROFILE_SHA256
            or calibration_start.get("execution_profile_sha256")
            != profile.V5_PROFILE_SHA256
            or calibration_complete.get("execution_profile_sha256")
            != profile.V5_PROFILE_SHA256
            or envelope.get("profile_runner_version")
            != profile.V5_PROFILE_RUNNER_VERSION
            or envelope.get("model") != source_profile.get("model")
            or envelope.get("endpoint") != source_profile.get("endpoint")
            or envelope.get("transport") != source_profile.get("transport")
            or envelope.get("sse_completion_policy")
            != source_profile.get("sse_completion_policy")
            or envelope.get("request_overlay")
            != source_profile.get("request_overlay")
            or envelope.get("sequence") != expected_sequence
            or calibration_start.get("sequence") != expected_sequence
            or calibration_complete.get("sequence") != expected_sequence
            or envelope.get("started_event_sha256")
            != profile.sha256_json(calibration_start)
            or calibration_complete.get("started_event_sha256")
            != profile.sha256_json(calibration_start)
            or envelope.get("core_result_sha256")
            != profile.sha256_json(calibration_core)
            or calibration_start.get("base_execution_contract_sha256") is not None
            or envelope.get("base_execution_contract_sha256") is not None
            or calibration_start.get("base_request_sha256")
            != profile.sha256_json(calibration_base_requests[calibration_id])
            or calibration_start.get("effective_request_sha256")
            != profile.sha256_json(request)
            or envelope.get("base_request_sha256")
            != profile.sha256_json(calibration_base_requests[calibration_id])
            or envelope.get("effective_request_sha256")
            != profile.sha256_json(request)
            or calibration_complete.get("result_envelope_sha256")
            != profile.sha256_json(envelope)
            or not isinstance(calibration_core, dict)
            or calibration_complete.get("result_status")
            != calibration_core.get("status")
            or calibration_core.get("status") not in profile.SUCCESS_STATUSES
            or errors
        ):
            raise profile.ProfileContractError(
                "v5 calibration source strict replay failed"
                + (": " + "; ".join(errors) if errors else "")
            )
        calibration_started.append(deepcopy(calibration_start))
        calibration_completed.append(deepcopy(calibration_complete))

    if (
        {str(row["logical_custom_id"]) for row in calibration_started}
        != set(calibration_requests)
        or {str(row["logical_custom_id"]) for row in calibration_completed}
        != set(calibration_requests)
    ):
        raise profile.ProfileContractError("v5 calibration event IDs are not exact 4/4")
    receipt = profile.v5_adoption_bridge_receipt(
        source_profile=source_profile,
        source_envelope=source_envelope,
        source_started_event=started,
        calibration_envelopes=calibrations,
        calibration_base_requests=calibration_base_rows,
        calibration_requests=calibration_request_rows,
        calibration_started_events=calibration_started,
        calibration_completed_events=calibration_completed,
        physical_sidecar_sha256={
            path.name: _sidecar_file_sha256(path) for path in paths
        },
        old_materialization_errors=old_errors,
        new_materialization_errors=new_errors,
    )

    bridged: dict[str, profile.PackRuntime] = {}
    for kind, current in runtimes.items():
        include_source = custom_id in set(
            profile._selection(current.pack).target_custom_ids
        )
        source_envelopes = (
            (deepcopy(source_envelope),) if include_source else ()
        )
        source_started_events = ((deepcopy(started),) if include_source else ())
        success_rows = ((deepcopy(projection),) if include_source else ())
        bridged[kind] = profile.PackRuntime(
            kind=current.kind,
            pack=current.pack,
            legacy_rows=current.legacy_rows,
            replay_legacy=current.replay_legacy,
            predecessor_bridge=current.predecessor_bridge,
            v4_adoption_bridge=current.v4_adoption_bridge,
            v5_adoption_bridge=profile.V5AdoptionBridge(
                receipt=deepcopy(receipt),
                source_profile=deepcopy(source_profile),
                success_rows=success_rows,
                source_envelopes=source_envelopes,
                source_started_events=source_started_events,
                source_core_rows=tuple(
                    deepcopy(row["core_result"]) for row in source_envelopes
                ),
                calibration_envelopes=tuple(deepcopy(row) for row in calibrations),
                calibration_base_requests=tuple(
                    deepcopy(row) for row in calibration_base_rows
                ),
                calibration_requests=tuple(
                    deepcopy(row) for row in calibration_request_rows
                ),
                calibration_started_events=tuple(calibration_started),
                calibration_completed_events=tuple(calibration_completed),
                validate_v5_source_core=core_validator(
                    kind=kind,
                    execution_profile=source_profile,
                    materialization_policy_version=(
                        profile.V5_MATERIALIZATION_POLICY_VERSION
                    ),
                ),
                validate_v6_adoption_core=core_validator(
                    kind=kind,
                    execution_profile=source_profile,
                    materialization_policy_version=(
                        profile.MATERIALIZATION_POLICY_VERSION
                    ),
                ),
            ),
        )
    return bridged


def _build_v4_calibration_requests(
    template: dict[str, Any], source_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """按冻结v4算法重建请求；不把v4 profile伪装成当前v5合同。"""

    effective = profile._effective_from_base(template)
    digest = str(source_profile["execution_profile_sha256"])
    contract_sha = profile.sha256_json(
        {
            "profile_sha256": digest,
            "model": source_profile["model"],
            "endpoint": source_profile["endpoint"],
            "transport": profile.TRANSPORT,
            "sse_completion_policy": profile.SSE_COMPLETION_POLICY,
            "prompt_sha256": source_profile["prompt_sha256"],
            "effective_request_config": {
                key: deepcopy(value)
                for key, value in effective["request"].items()
                if key not in {"messages", "model"}
            },
            "calibration_cases_manifest_sha256": profile.sha256_json(
                profile._calibration_manifest()
            ),
        }
    )
    system_message = deepcopy(effective["request"]["messages"][0])
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(profile._v4_calibration_cases(), start=1):
        rows.append(
            {
                "custom_id": (
                    f"em-nonthinking-{digest[:16]}-{index}-{case.name}"
                ),
                "prompt_sha256": source_profile["prompt_sha256"],
                "request": {
                    **deepcopy(effective["request"]),
                    "messages": [
                        system_message,
                        {
                            "role": "user",
                            "content": json.dumps(
                                case.payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                },
                "expected": deepcopy(case.expected),
                "calibration_execution_contract_sha256": contract_sha,
                "calibration_model": source_profile["model"],
                "calibration_endpoint": source_profile["endpoint"],
                "easy_medium_execution_profile_sha256": digest,
            }
        )
    return rows


def build_current_profile(
    *,
    runtimes: dict[str, profile.PackRuntime],
    model: str,
    endpoint: str,
    request_timeout: float,
    max_attempts: int,
    retry_delay: float,
    predecessor_bridge_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v4_receipts = {
        profile.sha256_json(runtime.v4_adoption_bridge.receipt): (
            runtime.v4_adoption_bridge.receipt
        )
        for runtime in runtimes.values()
        if runtime.v4_adoption_bridge is not None
    }
    if len(v4_receipts) > 1:
        raise profile.ProfileContractError("runtimes require at most one exact v4 adoption receipt")
    v5_receipts = {
        profile.sha256_json(runtime.v5_adoption_bridge.receipt): (
            runtime.v5_adoption_bridge.receipt
        )
        for runtime in runtimes.values()
        if runtime.v5_adoption_bridge is not None
    }
    if len(v5_receipts) > 1:
        raise profile.ProfileContractError(
            "runtimes require at most one exact v5 adoption receipt"
        )
    return profile.build_execution_profile(
        packs={kind: runtimes[kind].pack for kind in profile.KINDS},
        scope_profile=scope.scope_profile(),
        model=model,
        endpoint=endpoint,
        request_timeout=request_timeout,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        predecessor_bridge_receipt=predecessor_bridge_receipt,
        v4_adoption_bridge_receipt=(
            next(iter(v4_receipts.values())) if v4_receipts else None
        ),
        v5_adoption_bridge_receipt=(
            next(iter(v5_receipts.values())) if v5_receipts else None
        ),
    )


def _live_semantic_errors(
    result: dict[str, Any],
    request: dict[str, Any],
    *,
    materialization_policy_version: str = profile.MATERIALIZATION_POLICY_VERSION,
) -> list[str]:
    payload = json.loads(request["request"]["messages"][1]["content"])
    review = result.get("review")
    if not isinstance(review, dict):
        return ["new live formal review must be an object"]
    errors = legacy._validate_review(review)
    errors.extend(legacy._validate_against_payload(review, payload))
    errors.extend(legacy._validate_inverse_edge_canonicalization_audit(result, payload))
    errors.extend(legacy._validate_exact_variant_deduplication_audit(result, payload))
    errors.extend(legacy._validate_coverage_status_canonicalization_audit(result, payload))
    for item in result.get("invalid_attempts_before_success") or []:
        errors.extend(
            legacy._validate_invalid_attempt_raw_normalization_provenance(
                item,
                result,
                payload,
                extra_review_validator=lambda review, semantic_payload: (
                    legacy._review_materialization_errors(
                        review,
                        semantic_payload,
                        materialization_policy_version=(
                            materialization_policy_version
                        ),
                    )
                ),
            )
        )
    errors.extend(
        legacy._review_materialization_errors(
            review,
            payload,
            materialization_policy_version=materialization_policy_version,
        )
    )
    return errors


def _calibration_invalid_attempt_errors(
    item: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> list[str]:
    """重放calibration专属validation_errors，同时复用raw provenance校验。"""

    normalized = item.get("review")
    provenance = item.get("normalization_provenance")
    if not isinstance(normalized, dict) or not isinstance(provenance, dict):
        return ["calibration failed invalid-attempt review/provenance is invalid"]
    actions = provenance.get("actions")
    if not isinstance(actions, list):
        return ["calibration failed normalization actions are invalid"]

    # 通用provenance validator同时会重算production validation_errors；给其
    # 内存副本提供production等价值，只让它负责raw→normalized/hash/actions。
    generic_item = deepcopy(item)
    generic_errors = legacy._validate_review(normalized)
    if actions:
        generic_errors.append(
            "strict runner rejects structural normalization: "
            f"{len(actions)} action(s)"
        )
    generic_errors.extend(legacy._validate_against_payload(normalized, semantic_payload))
    if (
        (item.get("sse_integrity") or {}).get("termination")
        == "validated_stop_eof"
        and (item.get("sse_integrity") or {}).get("review_validation") == "failed"
        and not generic_errors
    ):
        generic_errors.append(
            "validated EOF prevalidation disagreed with authoritative validation"
        )
    generic_item["validation_errors"] = generic_errors
    errors = legacy._validate_invalid_attempt_raw_normalization_provenance(
        generic_item, source_row, semantic_payload
    )

    calibration_errors = legacy._validate_calibration(normalized, expected)
    if actions:
        calibration_errors.append(
            "strict runner rejects structural normalization: "
            f"{len(actions)} action(s)"
        )
    calibration_errors.extend(
        legacy._validate_against_payload(normalized, semantic_payload)
    )
    if (
        (item.get("sse_integrity") or {}).get("termination")
        == "validated_stop_eof"
        and (item.get("sse_integrity") or {}).get("review_validation") == "failed"
        and not calibration_errors
    ):
        calibration_errors.append(
            "validated EOF prevalidation disagreed with authoritative validation"
        )
    if item.get("validation_errors") != calibration_errors:
        errors.append("calibration failed validation-error receipt mismatch")
    return errors


def core_validator(
    *,
    kind: str,
    execution_profile: dict[str, Any],
    materialization_policy_version: str = profile.MATERIALIZATION_POLICY_VERSION,
) -> Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]]:
    """复用真实2.32 receipt/semantic/failure/materializer validators。"""

    if kind not in profile.KINDS:
        raise profile.ProfileContractError("core validator kind is invalid")

    def validate(
        result: dict[str, Any], request: dict[str, Any], calibration: bool
    ) -> Sequence[str]:
        if calibration:
            if result.get("status") != "failed":
                return legacy._validate_calibration_result(result, request)
            if "execution_contract_sha256" in result or "sequence" in result:
                return ["calibration failed core declares production-only bindings"]
            replay = deepcopy(result)
            replay["execution_contract_sha256"] = execution_profile[
                "execution_profile_sha256"
            ]
            replay["sequence"] = profile._expected_sequence(
                kind="calibration",
                logical_custom_id=str(request.get("custom_id") or ""),
                execution_profile=execution_profile,
            )
            # calibration SSE receipt额外保存parsed review SHA；先验证该冗余镜像，
            # 再在内存副本中剥离它以复用严格formal failed validator。
            invalid_by_attempt = {
                item.get("attempt"): item
                for item in replay.get("invalid_attempts") or []
                if isinstance(item, dict)
            }
            integrity_attempts = replay.get("sse_integrity_attempts")
            if not isinstance(integrity_attempts, list):
                return ["calibration failed SSE attempt ledger is invalid"]
            for integrity_row in integrity_attempts:
                if not isinstance(integrity_row, dict):
                    return ["calibration failed SSE attempt receipt is invalid"]
                invalid = invalid_by_attempt.get(integrity_row.get("attempt"))
                if (
                    not isinstance(invalid, dict)
                    or integrity_row.get("parsed_review_sha256")
                    != invalid.get("raw_review_sha256")
                ):
                    return ["calibration failed parsed review SHA mirror drifted"]
                integrity_row.pop("parsed_review_sha256", None)
                invalid_integrity = invalid.get("sse_integrity")
                if (
                    not isinstance(invalid_integrity, dict)
                    or invalid_integrity.get("parsed_review_sha256")
                    != invalid.get("raw_review_sha256")
                ):
                    return ["calibration failed invalid-attempt SSE receipt is invalid"]
                invalid_integrity.pop("parsed_review_sha256", None)
            terminal_integrity = replay.get("sse_integrity")
            final_invalid = invalid_by_attempt.get(replay.get("attempt"))
            if (
                not isinstance(terminal_integrity, dict)
                or not isinstance(final_invalid, dict)
                or terminal_integrity.get("parsed_review_sha256")
                != final_invalid.get("raw_review_sha256")
            ):
                return ["calibration failed terminal SSE receipt is invalid"]
            terminal_integrity.pop("parsed_review_sha256", None)
            if not integrity_attempts or terminal_integrity != integrity_attempts[-1]:
                return ["calibration failed terminal/attempt SSE mirror drifted"]
            return integrity.formal_rereview_failed_result_errors(
                replay,
                request,
                expected_sequence=int(replay["sequence"]),
                execution_contract_sha256=execution_profile[
                    "execution_profile_sha256"
                ],
                live_runner_version=legacy.RUNNER_VERSION,
                live_transport=legacy.TRANSPORT,
                validate_invalid_attempt=lambda item, source, payload: (
                    _calibration_invalid_attempt_errors(
                        item,
                        source,
                        payload,
                        expected=deepcopy(request["expected"]),
                    )
                ),
            )
        custom_id = str(request.get("custom_id") or "")
        expected_sequence = execution_profile["base_packs"][kind][
            "target_sequence_by_custom_id"
        ].get(custom_id)
        if type(expected_sequence) is not int:
            return ["new result custom id is outside the E/M target sequence"]
        if result.get("status") == "succeeded":
            errors = integrity._formal_live_receipt_errors(
                result,
                request,
                expected_sequence=expected_sequence,
            )
            if (
                result.get("execution_contract_sha256")
                != execution_profile["execution_profile_sha256"]
            ):
                errors.append("new live result execution profile contract drift")
            errors.extend(
                _live_semantic_errors(
                    result,
                    request,
                    materialization_policy_version=materialization_policy_version,
                )
            )
            return errors
        if result.get("status") == "failed":
            return integrity.formal_rereview_failed_result_errors(
                result,
                request,
                expected_sequence=expected_sequence,
                execution_contract_sha256=execution_profile[
                    "execution_profile_sha256"
                ],
                live_runner_version=legacy.RUNNER_VERSION,
                live_transport=legacy.TRANSPORT,
                validate_invalid_attempt=(
                    lambda item, source, payload: (
                        legacy._validate_invalid_attempt_raw_normalization_provenance(
                            item,
                            source,
                            payload,
                            extra_review_validator=lambda review, semantic_payload: (
                                legacy._review_materialization_errors(
                                    review,
                                    semantic_payload,
                                    materialization_policy_version=(
                                        materialization_policy_version
                                    ),
                                )
                            ),
                        )
                    )
                ),
            )
        return ["new production result status is invalid"]

    return validate


def _load_independent_ledgers() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    return (
        _read_jsonl(JOURNAL_PATH),
        _read_jsonl(RESPONSES_PATH),
        _read_jsonl(CALIBRATION_RESPONSES_PATH),
    )


def _sidecar_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_snapshot() -> dict[str, Any] | None:
    """供merge/validate/QA共享；无sidecar走旧路径，部分sidecar一律拒绝。"""

    core_paths = (
        PROFILE_PATH,
        JOURNAL_PATH,
        RESPONSES_PATH,
        CALIBRATION_RESPONSES_PATH,
    )
    existing = [path for path in core_paths if path.exists()]
    if not existing and not SUMMARY_PATH.exists():
        return None
    if len(existing) != len(core_paths):
        raise profile.ProfileContractError(
            "partial nonthinking sidecar set is not authoritative"
        )
    persisted = _read_json(PROFILE_PATH)
    profile.validate_execution_profile(persisted)
    expected_endpoint = legacy._endpoint(integrity.EXPECTED_REREVIEW_BASE_URL)
    if (
        persisted["model"] != integrity.EXPECTED_REREVIEW_MODEL
        or persisted["endpoint"] != expected_endpoint
    ):
        raise profile.ProfileContractError(
            "persisted nonthinking profile provider identity is not frozen"
        )
    runtimes, predecessor_receipt = _load_predecessor_bridge(load_runtimes())
    knobs = persisted["runtime_knobs"]
    computed = build_current_profile(
        runtimes=runtimes,
        model=str(persisted["model"]),
        endpoint=str(persisted["endpoint"]),
        request_timeout=float(knobs["request_timeout_seconds"]),
        max_attempts=int(knobs["max_attempts_per_failure_category"]),
        retry_delay=float(knobs["retry_delay_seconds"]),
        predecessor_bridge_receipt=predecessor_receipt,
    )
    if computed != persisted:
        raise profile.ProfileContractError(
            "persisted nonthinking profile does not match both current physical packs"
        )
    journal, responses, calibrations = _load_independent_ledgers()
    validators = {
        kind: core_validator(kind=kind, execution_profile=persisted)
        for kind in profile.KINDS
    }
    preflight = profile.preflight_execution(
        runtimes=runtimes,
        execution_profile=persisted,
        journal_rows=journal,
        response_rows=responses,
        calibration_response_rows=calibrations,
        validate_core_result_by_kind=validators,
    )
    if not preflight["api_safe"]:
        raise profile.ProfileContractError(
            "nonthinking release snapshot contains an uncertain/reconcile side effect"
        )
    summary = profile.build_scope_summary(
        runtimes=runtimes,
        execution_profile=persisted,
        response_rows=responses,
        calibration_response_rows=calibrations,
        journal_rows=journal,
        validate_core_result_by_kind=validators,
    )
    if responses:
        if not SUMMARY_PATH.exists():
            raise profile.ProfileContractError(
                "nonthinking results exist without their replayed summary"
            )
        persisted_summary = _read_json(SUMMARY_PATH)
        if persisted_summary != summary:
            raise profile.ProfileContractError(
                "persisted nonthinking summary differs from current replay"
            )
    elif SUMMARY_PATH.exists() and _read_json(SUMMARY_PATH) != summary:
        raise profile.ProfileContractError("empty nonthinking summary replay drifted")
    combined: dict[str, dict[str, dict[str, Any]]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for kind in profile.KINDS:
        rows, receipt = profile.combined_authoritative_view(
            runtime=runtimes[kind],
            execution_profile=persisted,
            journal_rows=journal,
            response_rows=responses,
            calibration_response_rows=calibrations,
            validate_core_result=validators[kind],
        )
        combined[kind] = rows
        receipts[kind] = receipt
    return {
        "execution_profile": persisted,
        "runtimes": runtimes,
        "combined_authoritative_by_kind": combined,
        "combined_receipt_by_kind": receipts,
        "summary": summary,
        "sidecar_sha256": {
            path.name: _sidecar_file_sha256(path)
            for path in (*core_paths,)
        },
    }


def _validate_persisted_profile(
    computed: dict[str, Any], *, ledgers_nonempty: bool
) -> None:
    if not PROFILE_PATH.exists():
        if ledgers_nonempty:
            raise profile.ProfileContractError(
                "independent ledgers exist without their persisted execution profile"
            )
        return
    persisted = _read_json(PROFILE_PATH)
    profile.validate_execution_profile(persisted)
    if persisted != computed:
        raise profile.ProfileContractError(
            "persisted execution profile differs from current physical packs/runtime"
        )


def _reconcile_without_api(
    preflight: dict[str, Any], *, append_journal: Callable[[dict[str, Any]], None]
) -> list[str]:
    reconciled: list[str] = []
    for kind in (*profile.KINDS, "calibration"):
        state = (
            preflight["calibration"]
            if kind == "calibration"
            else preflight["states"][kind]
        )
        for custom_id in state["reconcile_custom_ids"]:
            operation_id = next(
                operation
                for operation, event in state["started_by_operation_id"].items()
                if event["logical_custom_id"] == custom_id
            )
            completed = profile.build_completed_event(
                started_event=state["started_by_operation_id"][operation_id],
                result_envelope=state["result_by_operation_id"][operation_id],
            )
            append_journal(completed)
            reconciled.append(custom_id)
    return reconciled


def dry_run(
    *,
    runtimes: dict[str, profile.PackRuntime],
    execution_profile: dict[str, Any],
) -> dict[str, Any]:
    journal, responses, calibrations = _load_independent_ledgers()
    _validate_persisted_profile(
        execution_profile,
        ledgers_nonempty=bool(journal or responses or calibrations),
    )
    validators = {
        kind: core_validator(kind=kind, execution_profile=execution_profile)
        for kind in profile.KINDS
    }
    preflight = profile.preflight_execution(
        runtimes=runtimes,
        execution_profile=execution_profile,
        journal_rows=journal,
        response_rows=responses,
        calibration_response_rows=calibrations,
        validate_core_result_by_kind=validators,
    )
    summary = profile.build_scope_summary(
        runtimes=runtimes,
        execution_profile=execution_profile,
        response_rows=responses,
        calibration_response_rows=calibrations,
        journal_rows=journal,
        validate_core_result_by_kind=validators,
    )
    return {
        "status": "dry_run_only",
        "execution_profile_id": profile.PROFILE_ID,
        "execution_profile_sha256": execution_profile[
            "execution_profile_sha256"
        ],
        "execution_profile_path": str(PROFILE_PATH),
        "journal_path": str(JOURNAL_PATH),
        "responses_path": str(RESPONSES_PATH),
        "calibration_responses_path": str(CALIBRATION_RESPONSES_PATH),
        "summary_path": str(SUMMARY_PATH),
        "profile_persisted": PROFILE_PATH.exists(),
        "write_performed": False,
        "api_calls": 0,
        "hard_calls": 0,
        "api_safe_after_global_preflight": preflight["api_safe"],
        "unknown_side_effect_by_kind": preflight["unknown_side_effect_by_kind"],
        "reconcile_by_kind": preflight["reconcile_by_kind"],
        "summary_preview": summary,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--confirm-easy-medium-only", action="store_true")
    parser.add_argument("--confirm-no-concurrent-review", action="store_true")
    parser.add_argument("--api-key-env", default=legacy.DEFAULT_KEY_ENV)
    parser.add_argument("--base-url", default=integrity.EXPECTED_REREVIEW_BASE_URL)
    parser.add_argument("--model", default=integrity.EXPECTED_REREVIEW_MODEL)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.base_url != integrity.EXPECTED_REREVIEW_BASE_URL:
        raise SystemExit("nonthinking profile refuses a non-frozen DashScope base URL")
    if args.model != integrity.EXPECTED_REREVIEW_MODEL:
        raise SystemExit("nonthinking profile refuses a non-frozen Qwen model")
    endpoint = legacy._endpoint(args.base_url)
    runtimes, predecessor_receipt = _load_predecessor_bridge(load_runtimes())
    execution_profile = build_current_profile(
        runtimes=runtimes,
        model=args.model,
        endpoint=endpoint,
        request_timeout=args.request_timeout,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
        predecessor_bridge_receipt=predecessor_receipt,
    )
    preview = dry_run(runtimes=runtimes, execution_profile=execution_profile)
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    missing = [
        name
        for enabled, name in (
            (args.acknowledge_cost, "--acknowledge-cost"),
            (args.confirm_easy_medium_only, "--confirm-easy-medium-only"),
            (args.confirm_no_concurrent_review, "--confirm-no-concurrent-review"),
        )
        if not enabled
    ]
    if missing:
        raise SystemExit("paid execution refused; missing gate(s): " + ", ".join(missing))

    legacy._acquire_execution_lock(
        recover_stale=False,
        purpose="suite-v2 Easy/Medium explicit non-thinking rereview",
    )
    # 持锁后重新加载全部physical authority与独立账本，防TOCTOU与跨pack篡改。
    runtimes, predecessor_receipt = _load_predecessor_bridge(load_runtimes())
    execution_profile = build_current_profile(
        runtimes=runtimes,
        model=args.model,
        endpoint=endpoint,
        request_timeout=args.request_timeout,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
        predecessor_bridge_receipt=predecessor_receipt,
    )
    journal, responses, calibrations = _load_independent_ledgers()
    _validate_persisted_profile(
        execution_profile,
        ledgers_nonempty=bool(journal or responses or calibrations),
    )
    _initialize_output_bundle(execution_profile)
    validators = {
        kind: core_validator(kind=kind, execution_profile=execution_profile)
        for kind in profile.KINDS
    }
    preflight = profile.preflight_execution(
        runtimes=runtimes,
        execution_profile=execution_profile,
        journal_rows=journal,
        response_rows=responses,
        calibration_response_rows=calibrations,
        validate_core_result_by_kind=validators,
    )
    unknown = {
        kind: values
        for kind, values in preflight["unknown_side_effect_by_kind"].items()
        if values
    }
    if unknown:
        print(
            json.dumps(
                {
                    "status": "fail_stop_unknown_side_effect",
                    "api_calls": 0,
                    "hard_calls": 0,
                    "unknown_side_effect_by_kind": unknown,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    reconciled = _reconcile_without_api(
        preflight,
        append_journal=lambda row: legacy._append_jsonl(JOURNAL_PATH, row),
    )
    if reconciled:
        print(
            json.dumps(
                {
                    "status": "reconciled_result_without_api",
                    "custom_ids": reconciled,
                    "api_calls": 0,
                    "hard_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"paid execution refused; environment variable {args.api_key_env} is empty"
        )
    total_api_calls = 0
    pack_results: dict[str, Any] = {}
    for kind in profile.KINDS:
        journal, responses, calibrations = _load_independent_ledgers()

        def execute(row: dict[str, Any], calibration: bool) -> dict[str, Any]:
            return profile.execute_profile_request(
                row,
                kind="calibration" if calibration else kind,
                calibration=calibration,
                execution_profile=execution_profile,
                api_key=api_key,
            )

        result = profile.drive_pack(
            runtime=runtimes[kind],
            execution_profile=execution_profile,
            journal_rows=journal,
            response_rows=responses,
            calibration_response_rows=calibrations,
            append_journal=lambda row: legacy._append_jsonl(JOURNAL_PATH, row),
            append_response=lambda row: legacy._append_jsonl(RESPONSES_PATH, row),
            append_calibration_response=lambda row: legacy._append_jsonl(
                CALIBRATION_RESPONSES_PATH, row
            ),
            execute_one=execute,
            now=None,
            validate_core_result=validators[kind],
        )
        pack_results[kind] = result
        total_api_calls += int(result.get("api_calls") or 0)
        if result.get("status") != "succeeded":
            break
    journal, responses, calibrations = _load_independent_ledgers()
    summary = profile.build_scope_summary(
        runtimes=runtimes,
        execution_profile=execution_profile,
        response_rows=responses,
        calibration_response_rows=calibrations,
        journal_rows=journal,
        validate_core_result_by_kind=validators,
    )
    _write_json_fsync(SUMMARY_PATH, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "pack_results": pack_results,
                "api_calls": total_api_calls,
                "hard_calls": 0,
                "summary_path": str(SUMMARY_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
