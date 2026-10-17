#!/usr/bin/env python3
"""Safely manage the audited retrieval-profile v2 through supported local APIs.

The only permitted configuration change is a generic enlargement of the
candidate pool available to fusion and reranking:

* ``candidate_k``: 5 -> 15
* ``fusion.max_per_document``: 5 -> 15

``top_k`` and the complete ContextPack configuration remain frozen.  Every
mutating action is guarded by the three-difficulty baseline's exact v1 profile,
KB-scoped profile listings, append-only intent/result events, and post-action
pointer verification.  Credentials are used only in memory and are redacted
from all persisted HTTP snapshots.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
SUITE_ROOT = ROOT.parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (SUITE_ROOT, *SUITE_ROOT.parents)
        if (candidate / "pyproject.toml").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:  # pragma: no cover - layout guard
    raise RuntimeError("cannot locate repository root")

BASELINE_RUNNER_PATH = SUITE_ROOT / "run_live_baselines.py"
BASELINE_REGISTRY_PATH = SUITE_ROOT / "baselines" / "baselines.json"
STATE_ROOT = ROOT / "retrieval-profile-v2"
STATE_PATH = STATE_ROOT / "profile-state.json"
EVENTS_ROOT = STATE_ROOT / "events"
CHECKSUM_ROOT = STATE_ROOT / "checksums"
PROFILE_ID = "rpf_default"
SOURCE_VERSION = 1
TARGET_VERSION = 2
TARGET_CANDIDATE_K = 15
KB_ID_DEFAULT = "kb_2fad25538aab42d19baf79fba782a06c"
STATE_SCHEMA = "stt.eval.retrieval-profile-remediation.v2"


def _load_baseline_runner():
    spec = importlib.util.spec_from_file_location(
        "stt_profile_v2_baseline_runner", BASELINE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load baseline runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


br = _load_baseline_runner()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    br._write_json(path, value)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data(payload: dict[str, Any], operation: str) -> dict[str, Any]:
    value = br._envelope_data(payload, operation=operation)
    if not isinstance(value, dict):
        raise br.BaselineRunnerError(f"{operation} returned non-object data")
    return value


def _profile_identity(value: dict[str, Any]) -> tuple[str, int, str]:
    profile_id = value.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise br.BaselineRunnerError("retrieval profile has no logical profile_id")
    return profile_id, int(value.get("version") or 0), str(value.get("status") or "")


def _config(value: dict[str, Any]) -> dict[str, Any]:
    config = value.get("config")
    if not isinstance(config, dict):
        raise br.BaselineRunnerError("retrieval profile has no config object")
    return copy.deepcopy(config)


def _changed_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths: set[str] = set()
        for key in set(before) | set(after):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.add(child)
            else:
                paths.update(_changed_paths(before[key], after[key], child))
        return paths
    return set() if before == after else {prefix}


def _target_config(source: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(source)
    if int(target.get("top_k") or 0) != 5:
        raise br.BaselineRunnerError("frozen v1 top_k is not 5")
    if int(target.get("candidate_k") or 0) != 5:
        raise br.BaselineRunnerError("frozen v1 candidate_k is not 5")
    fusion = target.get("fusion")
    if not isinstance(fusion, dict):
        raise br.BaselineRunnerError("frozen v1 config has no fusion object")
    if int(fusion.get("max_per_document") or 0) != 5:
        raise br.BaselineRunnerError("frozen v1 fusion.max_per_document is not 5")
    target["candidate_k"] = TARGET_CANDIDATE_K
    fusion["max_per_document"] = TARGET_CANDIDATE_K
    expected = {"candidate_k", "fusion.max_per_document"}
    changed = _changed_paths(source, target)
    if changed != expected:
        raise br.BaselineRunnerError(
            f"unsafe retrieval profile diff: expected {sorted(expected)}, got {sorted(changed)}"
        )
    return target


def _baseline_expectation(kb_id: str) -> dict[str, Any]:
    registry = _read_json(BASELINE_REGISTRY_PATH)
    if not isinstance(registry, dict) or registry.get("status") != "succeeded":
        raise br.BaselineRunnerError("three-difficulty baseline registry is not succeeded")
    if registry.get("kb_id") != kb_id:
        raise br.BaselineRunnerError("requested KB differs from the frozen baseline registry")
    profiles: list[dict[str, Any]] = []
    records = registry.get("baselines") or {}
    for difficulty in ("easy", "medium", "hard"):
        profile = (
            (records.get(difficulty) or {})
            .get("config_snapshot", {})
            .get("retrieval_profile")
        )
        if not isinstance(profile, dict):
            raise br.BaselineRunnerError(
                f"baseline {difficulty} has no frozen retrieval profile"
            )
        if _profile_identity(profile)[:2] != (PROFILE_ID, SOURCE_VERSION):
            raise br.BaselineRunnerError(
                f"baseline {difficulty} does not freeze rpf_default v1"
            )
        profiles.append(profile)
    reference = profiles[0]
    for profile in profiles[1:]:
        if (
            _config(profile) != _config(reference)
            or profile.get("config_hash") != reference.get("config_hash")
        ):
            raise br.BaselineRunnerError(
                "three difficulty baselines do not share one retrieval profile"
            )
    return {
        "registry_path": BASELINE_REGISTRY_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "registry_sha256": br._sha256_file(BASELINE_REGISTRY_PATH),
        "registry_generated_at": registry.get("generated_at"),
        "config": _config(reference),
        "config_hash": reference.get("config_hash"),
        "canonical_config_sha256": _canonical_sha256(_config(reference)),
    }


def _list_profiles(
    client: Any, kb_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    envelope = client.request_json(
        "GET", f"/api/v1/knowledge-bases/{kb_id}/retrieval-profiles"
    )
    data = _data(envelope, "list retrieval profiles")
    current = data.get("current")
    items = data.get("items")
    if not isinstance(current, dict) or not isinstance(items, list):
        raise br.BaselineRunnerError("KB-scoped profile list is incomplete")
    if current.get("kb_id") != kb_id:
        raise br.BaselineRunnerError("current retrieval profile has the wrong KB")
    clean_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("kb_id") != kb_id:
            raise br.BaselineRunnerError("profile list contains a cross-KB row")
        clean_items.append(item)
    return envelope, data, current, clean_items


def _control_plane_revision(data: dict[str, Any]) -> int:
    value = data.get("control_plane_revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise br.BaselineRunnerError("KB profile list has no valid control-plane revision")
    return value


def _find_version(items: list[dict[str, Any]], version: int) -> dict[str, Any] | None:
    matches = [
        item
        for item in items
        if item.get("profile_id") == PROFILE_ID and int(item.get("version") or 0) == version
    ]
    if len(matches) > 1:
        raise br.BaselineRunnerError(f"duplicate KB-scoped rpf_default v{version}")
    return matches[0] if matches else None


def _reject_later_versions(items: list[dict[str, Any]]) -> None:
    if any(
        item.get("profile_id") == PROFILE_ID
        and int(item.get("version") or 0) > TARGET_VERSION
        for item in items
    ):
        raise br.BaselineRunnerError("a later rpf_default version already exists")


def _assert_profile(
    profile: dict[str, Any],
    *,
    kb_id: str,
    version: int,
    statuses: set[str],
    expected_config: dict[str, Any],
    expected_config_hash: str | None = None,
) -> None:
    identity = _profile_identity(profile)
    if profile.get("kb_id") != kb_id:
        raise br.BaselineRunnerError("retrieval profile belongs to the wrong KB")
    if identity[0] != PROFILE_ID or identity[1] != version or identity[2] not in statuses:
        raise br.BaselineRunnerError(
            f"unexpected rpf_default v{version} lifecycle identity: {identity}"
        )
    if _config(profile) != expected_config:
        raise br.BaselineRunnerError(f"live rpf_default v{version} config changed")
    if expected_config_hash is not None and profile.get("config_hash") != expected_config_hash:
        raise br.BaselineRunnerError(f"live rpf_default v{version} config hash changed")


def _new_state(
    *,
    kb_id: str,
    expectation: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
    adopted: bool,
) -> dict[str, Any]:
    source_config = _config(source)
    target_config = _config(target)
    return {
        "schema_version": STATE_SCHEMA,
        "kb_id": kb_id,
        "created_at": br._utc_now(),
        "status": "draft",
        "adopted_existing_draft": adopted,
        "baseline_registry": {
            "path": expectation["registry_path"],
            "sha256": expectation["registry_sha256"],
            "generated_at": expectation["registry_generated_at"],
        },
        "source": {
            "profile_id": PROFILE_ID,
            "version": SOURCE_VERSION,
            "status": source.get("status"),
            "config_hash": source.get("config_hash"),
            "canonical_config_sha256": _canonical_sha256(source_config),
            "config": source_config,
            "published_at": source.get("published_at"),
        },
        "target": {
            "profile_id": PROFILE_ID,
            "version": TARGET_VERSION,
            "status": target.get("status"),
            "config_hash": target.get("config_hash"),
            "canonical_config_sha256": _canonical_sha256(target_config),
            "config": target_config,
        },
        "changed_paths": sorted(_changed_paths(source_config, target_config)),
        "validations": {"previews": []},
    }


def _load_state(kb_id: str, expectation: dict[str, Any]) -> dict[str, Any]:
    if not STATE_PATH.is_file():
        raise br.BaselineRunnerError("profile-state.json is required before this action")
    state = _read_json(STATE_PATH)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != STATE_SCHEMA
        or state.get("kb_id") != kb_id
    ):
        raise br.BaselineRunnerError("saved profile state has the wrong schema or KB")
    registry = state.get("baseline_registry") or {}
    if registry.get("sha256") != expectation["registry_sha256"]:
        raise br.BaselineRunnerError("baseline registry changed after profile state was created")
    source = state.get("source")
    target = state.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise br.BaselineRunnerError("saved profile state is incomplete")
    if (source.get("profile_id"), source.get("version")) != (
        PROFILE_ID,
        SOURCE_VERSION,
    ) or (target.get("profile_id"), target.get("version")) != (
        PROFILE_ID,
        TARGET_VERSION,
    ):
        raise br.BaselineRunnerError("saved profile lineage is invalid")
    source_config = source.get("config")
    target_config = target.get("config")
    if not isinstance(source_config, dict) or not isinstance(target_config, dict):
        raise br.BaselineRunnerError("saved profile configs are invalid")
    if (
        source_config != expectation["config"]
        or source.get("config_hash") != expectation["config_hash"]
        or source.get("canonical_config_sha256") != _canonical_sha256(source_config)
    ):
        raise br.BaselineRunnerError("saved source differs from the frozen baseline v1")
    transformed = _target_config(source_config)
    if (
        target_config != transformed
        or target.get("canonical_config_sha256") != _canonical_sha256(target_config)
        or set(state.get("changed_paths") or [])
        != {"candidate_k", "fusion.max_per_document"}
    ):
        raise br.BaselineRunnerError("saved target is not the audited two-field transform")
    validations = state.get("validations")
    if not isinstance(validations, dict) or not isinstance(validations.get("previews"), list):
        raise br.BaselineRunnerError("saved validation receipts are invalid")
    return state


def _assert_target_against_state(
    profile: dict[str, Any], *, kb_id: str, state: dict[str, Any], statuses: set[str]
) -> None:
    target = state["target"]
    _assert_profile(
        profile,
        kb_id=kb_id,
        version=TARGET_VERSION,
        statuses=statuses,
        expected_config=target["config"],
        expected_config_hash=target["config_hash"],
    )


def _write_event(invocation_id: str, phase: str, payload: dict[str, Any]) -> Path:
    EVENTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = EVENTS_ROOT / f"{invocation_id}-{phase}.json"
    if path.exists():
        raise br.BaselineRunnerError("append-only profile event already exists")
    _write_json(path, payload)
    return path


def _has_audited_preview(state: dict[str, Any]) -> bool:
    target_hash = state["target"]["config_hash"]
    for receipt in state["validations"]["previews"]:
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "succeeded"
            or receipt.get("target_config_hash") != target_hash
        ):
            continue
        invocation_id = str(receipt.get("invocation_id") or "")
        event_path = EVENTS_ROOT / f"{invocation_id}-result.json"
        checksum_path = CHECKSUM_ROOT / f"{invocation_id}.sha256"
        if not event_path.is_file() or not checksum_path.is_file():
            continue
        event = _read_json(event_path)
        if (
            not isinstance(event, dict)
            or event.get("action") != "preview"
            or event.get("status") != "succeeded"
            or event.get("result") != receipt
        ):
            continue
        expected_line = (
            f"{br._sha256_file(event_path)}  "
            f"{event_path.relative_to(STATE_ROOT).as_posix()}"
        )
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
        if expected_line in checksum_lines:
            return True
    return False


def _intent(
    invocation_id: str,
    *,
    action: str,
    kb_id: str,
    current: dict[str, Any],
    control_plane_revision: int,
    expectation: dict[str, Any],
) -> None:
    _write_event(
        invocation_id,
        "intent",
        {
            "schema_version": "stt.eval.retrieval-profile-event.v1",
            "invocation_id": invocation_id,
            "phase": "intent",
            "action": action,
            "recorded_at": br._utc_now(),
            "kb_id": kb_id,
            "current_identity": list(_profile_identity(current)),
            "expected_control_plane_revision": control_plane_revision,
            "baseline_registry_sha256": expectation["registry_sha256"],
            "source_config_hash": expectation["config_hash"],
            "target_canonical_config_sha256": _canonical_sha256(
                _target_config(expectation["config"])
            ),
        },
    )


def _inspect(client: Any, kb_id: str, redactor: Any) -> dict[str, Any]:
    envelope, data, current, items = _list_profiles(client, kb_id)
    return {
        "inspected_at": br._utc_now(),
        "kb_id": kb_id,
        "control_plane_revision": _control_plane_revision(data),
        "current": redactor.clean(current),
        "items": redactor.clean(items),
        "response_sha256": _canonical_sha256(redactor.clean(envelope)),
    }


def _create(
    client: Any,
    kb_id: str,
    redactor: Any,
    invocation_id: str,
) -> dict[str, Any]:
    expectation = _baseline_expectation(kb_id)
    _, list_data, current, items = _list_profiles(client, kb_id)
    control_plane_revision = _control_plane_revision(list_data)
    source = _find_version(items, SOURCE_VERSION)
    if source is None:
        raise br.BaselineRunnerError("KB has no frozen rpf_default v1")
    _assert_profile(
        source,
        kb_id=kb_id,
        version=SOURCE_VERSION,
        statuses={"published"},
        expected_config=expectation["config"],
        expected_config_hash=expectation["config_hash"],
    )
    if _profile_identity(current) != (PROFILE_ID, SOURCE_VERSION, "published"):
        raise br.BaselineRunnerError("current pointer is not the audited published v1")
    lineage = [item for item in items if item.get("profile_id") == PROFILE_ID]
    _reject_later_versions(lineage)
    expected_target = _target_config(expectation["config"])
    target = _find_version(items, TARGET_VERSION)
    saved = _load_state(kb_id, expectation) if STATE_PATH.is_file() else None
    if target is not None:
        _assert_profile(
            target,
            kb_id=kb_id,
            version=TARGET_VERSION,
            statuses={"draft"},
            expected_config=expected_target,
            expected_config_hash=(saved or {}).get("target", {}).get("config_hash"),
        )
        if saved is not None:
            _assert_target_against_state(target, kb_id=kb_id, state=saved, statuses={"draft"})
            return saved
        state = _new_state(
            kb_id=kb_id,
            expectation=expectation,
            source=source,
            target=target,
            adopted=True,
        )
        state["control_plane_revision"] = control_plane_revision
        _write_json(STATE_PATH, redactor.clean(state))
        return state
    if saved is not None:
        raise br.BaselineRunnerError("saved v2 state exists but the live v2 row is missing")

    _intent(
        invocation_id,
        action="create",
        kb_id=kb_id,
        current=current,
        control_plane_revision=control_plane_revision,
        expectation=expectation,
    )
    body = {
        "profile_id": PROFILE_ID,
        "name": "rpf_default v2 candidate pool 15",
        "description": (
            "Increase fusion/reranker candidate pool from 5 to 15 while preserving "
            "top_k and ContextPack budgets."
        ),
        "config": expected_target,
        "expected_current_profile_id": PROFILE_ID,
        "expected_current_version": SOURCE_VERSION,
        "expected_current_config_hash": expectation["config_hash"],
        "expected_control_plane_revision": control_plane_revision,
        "expected_max_version": SOURCE_VERSION,
    }
    created_envelope = client.request_json(
        "POST",
        f"/api/v1/knowledge-bases/{kb_id}/retrieval-profiles",
        body=body,
    )
    created = _data(created_envelope, "create rpf_default v2")
    _assert_profile(
        created,
        kb_id=kb_id,
        version=TARGET_VERSION,
        statuses={"draft"},
        expected_config=expected_target,
    )
    _, after_data, after_current, after_items = _list_profiles(client, kb_id)
    after_revision = _control_plane_revision(after_data)
    if after_revision <= control_plane_revision:
        raise br.BaselineRunnerError("draft creation did not advance the control-plane revision")
    if _profile_identity(after_current) != (PROFILE_ID, SOURCE_VERSION, "published"):
        raise br.BaselineRunnerError("creating a draft unexpectedly changed the KB pointer")
    after_target = _find_version(after_items, TARGET_VERSION)
    if after_target is None:
        raise br.BaselineRunnerError("created v2 is absent from the KB-scoped list")
    _assert_profile(
        after_target,
        kb_id=kb_id,
        version=TARGET_VERSION,
        statuses={"draft"},
        expected_config=expected_target,
        expected_config_hash=created.get("config_hash"),
    )
    state = _new_state(
        kb_id=kb_id,
        expectation=expectation,
        source=source,
        target=after_target,
        adopted=False,
    )
    state["created_response_sha256"] = _canonical_sha256(redactor.clean(created_envelope))
    state["control_plane_revision"] = after_revision
    _write_json(STATE_PATH, redactor.clean(state))
    return state


def _preview(
    client: Any,
    kb_id: str,
    query: str,
    redactor: Any,
    invocation_id: str,
) -> dict[str, Any]:
    expectation = _baseline_expectation(kb_id)
    state = _load_state(kb_id, expectation)
    _, before_data, before_current, items = _list_profiles(client, kb_id)
    before_revision = _control_plane_revision(before_data)
    _reject_later_versions(items)
    target = _find_version(items, TARGET_VERSION)
    if target is None:
        raise br.BaselineRunnerError("audited v2 is missing")
    _assert_target_against_state(
        target, kb_id=kb_id, state=state, statuses={"draft", "published"}
    )
    before_identity = _profile_identity(before_current)
    allowed_currents = {
        (PROFILE_ID, SOURCE_VERSION, "published"),
        (PROFILE_ID, TARGET_VERSION, "published"),
    }
    if before_identity not in allowed_currents:
        raise br.BaselineRunnerError("KB pointer is outside the audited v1/v2 lifecycle")
    envelope = client.request_json(
        "POST",
        f"/api/v1/retrieval-profiles/{PROFILE_ID}/versions/{TARGET_VERSION}:preview",
        body={"kb_id": kb_id, "query": query, "retrieval_filter": None},
    )
    response = _data(envelope, "preview rpf_default v2")
    if (
        response.get("profile_id") != PROFILE_ID
        or int(response.get("profile_version") or 0) != TARGET_VERSION
        or response.get("profile_hash") != state["target"]["config_hash"]
        or response.get("query_variant") != "preview"
    ):
        raise br.BaselineRunnerError("preview returned the wrong profile identity")
    routes = response.get("routes")
    results = response.get("results")
    if (
        not isinstance(routes, list)
        or not routes
        or any(
            not isinstance(route, dict) or route.get("status") != "succeeded"
            for route in routes
        )
        or not isinstance(results, list)
        or not results
    ):
        raise br.BaselineRunnerError("preview did not return successful routes and evidence")
    _, after_data, after_current, _ = _list_profiles(client, kb_id)
    after_revision = _control_plane_revision(after_data)
    if _profile_identity(after_current) != before_identity:
        raise br.BaselineRunnerError("read-only preview changed the KB pointer")
    receipt = {
        "invocation_id": invocation_id,
        "previewed_at": br._utc_now(),
        "status": "succeeded",
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "query_length": len(query),
        "target_config_hash": state["target"]["config_hash"],
        "response_sha256": _canonical_sha256(redactor.clean(envelope)),
        "current_identity": list(before_identity),
        "control_plane_revision_before": before_revision,
        "control_plane_revision_after": after_revision,
        "route_count": len(routes),
        "result_count": len(results),
    }
    state["validations"]["previews"].append(receipt)
    state["last_validated_at"] = receipt["previewed_at"]
    _write_json(STATE_PATH, redactor.clean(state))
    return receipt


def _publish(
    client: Any,
    kb_id: str,
    redactor: Any,
    invocation_id: str,
) -> dict[str, Any]:
    expectation = _baseline_expectation(kb_id)
    state = _load_state(kb_id, expectation)
    _, list_data, current, items = _list_profiles(client, kb_id)
    control_plane_revision = _control_plane_revision(list_data)
    _reject_later_versions(items)
    source = _find_version(items, SOURCE_VERSION)
    target = _find_version(items, TARGET_VERSION)
    if source is None or target is None:
        raise br.BaselineRunnerError("audited v1/v2 lineage is incomplete")
    _assert_profile(
        source,
        kb_id=kb_id,
        version=SOURCE_VERSION,
        statuses={"published", "retired"},
        expected_config=state["source"]["config"],
        expected_config_hash=state["source"]["config_hash"],
    )
    _assert_target_against_state(
        target, kb_id=kb_id, state=state, statuses={"draft", "published"}
    )
    current_identity = _profile_identity(current)
    if current_identity == (PROFILE_ID, TARGET_VERSION, "published"):
        if source.get("status") != "retired" or target.get("status") != "published":
            raise br.BaselineRunnerError("published v2 lifecycle is inconsistent")
        state["status"] = "published"
        state["target"]["status"] = "published"
        state.setdefault("published_at", br._utc_now())
        state["reconciled_published_state"] = True
        _write_json(STATE_PATH, redactor.clean(state))
        return state
    if current_identity != (PROFILE_ID, SOURCE_VERSION, "published"):
        raise br.BaselineRunnerError("publish would replace an unaudited current profile")
    if target.get("status") != "draft" or source.get("status") != "published":
        raise br.BaselineRunnerError("publish requires published v1 plus audited draft v2")
    if not _has_audited_preview(state):
        raise br.BaselineRunnerError(
            "publish requires a checksummed append-only preview receipt for v2"
        )

    _intent(
        invocation_id,
        action="publish",
        kb_id=kb_id,
        current=current,
        control_plane_revision=control_plane_revision,
        expectation=expectation,
    )
    envelope = client.request_json(
        "POST",
        f"/api/v1/retrieval-profiles/{PROFILE_ID}/versions/{TARGET_VERSION}:publish",
        body={
            "kb_id": kb_id,
            "expected_current_profile_id": PROFILE_ID,
            "expected_current_version": SOURCE_VERSION,
            "expected_current_config_hash": state["source"]["config_hash"],
            "expected_control_plane_revision": control_plane_revision,
            "expected_max_version": TARGET_VERSION,
        },
    )
    published = _data(envelope, "publish rpf_default v2")
    _assert_target_against_state(
        published, kb_id=kb_id, state=state, statuses={"published"}
    )
    _, after_data, after_current, after_items = _list_profiles(client, kb_id)
    after_revision = _control_plane_revision(after_data)
    if after_revision <= control_plane_revision:
        raise br.BaselineRunnerError("publish did not advance the control-plane revision")
    _reject_later_versions(after_items)
    after_source = _find_version(after_items, SOURCE_VERSION)
    after_target = _find_version(after_items, TARGET_VERSION)
    if (
        _profile_identity(after_current) != (PROFILE_ID, TARGET_VERSION, "published")
        or after_source is None
        or after_source.get("status") != "retired"
        or after_target is None
        or after_target.get("status") != "published"
    ):
        raise br.BaselineRunnerError("publish postcondition failed")
    _assert_target_against_state(
        after_target, kb_id=kb_id, state=state, statuses={"published"}
    )
    state["status"] = "published"
    state["published_at"] = br._utc_now()
    state["target"]["status"] = "published"
    state["publish_response_sha256"] = _canonical_sha256(redactor.clean(envelope))
    state["control_plane_revision"] = after_revision
    _write_json(STATE_PATH, redactor.clean(state))
    return state


def _rollback(
    client: Any,
    kb_id: str,
    redactor: Any,
    invocation_id: str,
) -> dict[str, Any]:
    expectation = _baseline_expectation(kb_id)
    state = _load_state(kb_id, expectation)
    _, list_data, current, items = _list_profiles(client, kb_id)
    control_plane_revision = _control_plane_revision(list_data)
    source = _find_version(items, SOURCE_VERSION)
    target = _find_version(items, TARGET_VERSION)
    if source is None or target is None:
        raise br.BaselineRunnerError("audited v1/v2 lineage is incomplete")
    _reject_later_versions(items)
    _assert_profile(
        source,
        kb_id=kb_id,
        version=SOURCE_VERSION,
        statuses={"published", "retired"},
        expected_config=state["source"]["config"],
        expected_config_hash=state["source"]["config_hash"],
    )
    _assert_target_against_state(
        target, kb_id=kb_id, state=state, statuses={"published", "retired"}
    )
    current_identity = _profile_identity(current)
    if current_identity == (PROFILE_ID, SOURCE_VERSION, "published"):
        if source.get("status") != "published" or target.get("status") != "retired":
            raise br.BaselineRunnerError("rolled-back lifecycle is inconsistent")
        state["status"] = "rolled_back"
        state["source"]["status"] = "published"
        state["target"]["status"] = "retired"
        state.setdefault("rolled_back_at", br._utc_now())
        state["reconciled_rolled_back_state"] = True
        _write_json(STATE_PATH, redactor.clean(state))
        return state
    if current_identity != (PROFILE_ID, TARGET_VERSION, "published"):
        raise br.BaselineRunnerError("rollback refuses to replace an unaudited current profile")
    if source.get("status") != "retired" or source.get("published_at") is None:
        raise br.BaselineRunnerError("v1 is not a previously published rollback target")
    if target.get("status") != "published":
        raise br.BaselineRunnerError("current v2 lifecycle is inconsistent")

    _intent(
        invocation_id,
        action="rollback",
        kb_id=kb_id,
        current=current,
        control_plane_revision=control_plane_revision,
        expectation=expectation,
    )
    envelope = client.request_json(
        "POST",
        f"/api/v1/knowledge-bases/{kb_id}:rollback-retrieval-profile",
        body={
            "profile_id": PROFILE_ID,
            "version": SOURCE_VERSION,
            "expected_current_profile_id": PROFILE_ID,
            "expected_current_version": TARGET_VERSION,
            "expected_current_config_hash": state["target"]["config_hash"],
            "expected_control_plane_revision": control_plane_revision,
            "expected_max_version": TARGET_VERSION,
        },
    )
    rolled_back = _data(envelope, "rollback retrieval profile")
    _assert_profile(
        rolled_back,
        kb_id=kb_id,
        version=SOURCE_VERSION,
        statuses={"published"},
        expected_config=state["source"]["config"],
        expected_config_hash=state["source"]["config_hash"],
    )
    _, after_data, after_current, after_items = _list_profiles(client, kb_id)
    after_revision = _control_plane_revision(after_data)
    if after_revision <= control_plane_revision:
        raise br.BaselineRunnerError("rollback did not advance the control-plane revision")
    _reject_later_versions(after_items)
    after_source = _find_version(after_items, SOURCE_VERSION)
    after_target = _find_version(after_items, TARGET_VERSION)
    if (
        _profile_identity(after_current) != (PROFILE_ID, SOURCE_VERSION, "published")
        or after_source is None
        or after_source.get("status") != "published"
        or after_target is None
        or after_target.get("status") != "retired"
    ):
        raise br.BaselineRunnerError("rollback postcondition failed")
    state["status"] = "rolled_back"
    state["rolled_back_at"] = br._utc_now()
    state["source"]["status"] = "published"
    state["target"]["status"] = "retired"
    state["rollback_response_sha256"] = _canonical_sha256(redactor.clean(envelope))
    state["control_plane_revision"] = after_revision
    _write_json(STATE_PATH, redactor.clean(state))
    return state


def _write_checksum_receipt(invocation_id: str) -> Path:
    CHECKSUM_ROOT.mkdir(parents=True, exist_ok=True)
    target = CHECKSUM_ROOT / f"{invocation_id}.sha256"
    if target.exists():
        raise br.BaselineRunnerError("append-only checksum receipt already exists")
    files = sorted(
        (
            path
            for path in STATE_ROOT.rglob("*")
            if path.is_file() and path.resolve() != target.resolve()
        ),
        key=lambda path: path.relative_to(STATE_ROOT).as_posix(),
    )
    content = "".join(
        f"{br._sha256_file(path)}  {path.relative_to(STATE_ROOT).as_posix()}\n"
        for path in files
    )
    br._write_bytes(target, content.encode("utf-8"))
    return target


def _args(config: dict[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("inspect", "create", "preview", "publish", "rollback")
    )
    parser.add_argument("--query", help="required only for preview; persisted only as SHA-256")
    parser.add_argument(
        "--base-url",
        default=br._first_config(
            config,
            "STT_EVAL_BASE_URL",
            "STT_QUERY_BASE_URL",
            default=br.DEFAULT_BASE_URL,
        ),
    )
    parser.add_argument(
        "--kb-id",
        default=br._first_config(
            config, "STT_EVAL_KB_ID", "EVALUATION_KB_ID", default=KB_ID_DEFAULT
        ),
    )
    parser.add_argument("--request-timeout", type=br._positive_float, default=120.0)
    args = parser.parse_args()
    args.base_url = br._normalize_local_base_url(args.base_url)
    args.kb_id = br._safe_component(str(args.kb_id or ""), label="knowledge-base id")
    if args.action == "preview" and not str(args.query or "").strip():
        raise br.BaselineRunnerError("preview requires --query")
    if args.action != "preview" and args.query:
        raise br.BaselineRunnerError("--query is only accepted for preview")
    return args


def main() -> int:
    config = br._load_environment()
    args = _args(config)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    br.SUITE_ROOT = ROOT
    br.API_SNAPSHOT_ROOT = STATE_ROOT / "api-snapshots"

    email = br._first_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_EMAIL",
        "AUTH_BOOTSTRAP_ADMIN_EMAIL",
        default="admin@stt.local",
    )
    password = br._first_secret_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_PASSWORD",
        "AUTH_BOOTSTRAP_ADMIN_PASSWORD",
    )
    tenant_id = br._first_config(
        config, "STT_EVAL_TENANT_ID", "AUTH_BOOTSTRAP_TENANT_ID"
    ) or None
    if not password:
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.shared.config.settings_config import settings

        email = settings.auth_bootstrap_admin_email
        password = settings.auth_bootstrap_admin_password
    if not password:
        raise br.BaselineRunnerError("bootstrap admin password could not be resolved")

    redactor = br.Redactor(password)
    invocation_id = f"{br._timestamp_slug()}-{uuid.uuid4().hex[:8]}-{args.action}"
    recorder = br.ApiSnapshotRecorder(invocation_id, args.base_url, redactor)
    client = br.ApiClient(
        base_url=args.base_url,
        email=email,
        password=password,
        tenant_id=tenant_id,
        timeout_seconds=args.request_timeout,
        recorder=recorder,
        redactor=redactor,
    )
    error: Exception | None = None
    result: dict[str, Any] = {}
    try:
        client.login()
        if args.action == "inspect":
            result = _inspect(client, args.kb_id, redactor)
        elif args.action == "create":
            result = _create(client, args.kb_id, redactor, invocation_id)
        elif args.action == "preview":
            result = _preview(
                client,
                args.kb_id,
                str(args.query),
                redactor,
                invocation_id,
            )
        elif args.action == "publish":
            result = _publish(client, args.kb_id, redactor, invocation_id)
        else:
            result = _rollback(client, args.kb_id, redactor, invocation_id)
    except Exception as exc:
        error = exc
    finally:
        logout_error_type: str | None = None
        try:
            client.logout()
        except Exception as exc:
            logout_error_type = type(exc).__name__
        snapshot_info = recorder.close(
            status="failed" if error else "succeeded",
            error_type=type(error).__name__ if error else None,
        )
        _write_event(
            invocation_id,
            "result",
            {
                "schema_version": "stt.eval.retrieval-profile-event.v1",
                "invocation_id": invocation_id,
                "phase": "result",
                "action": args.action,
                "recorded_at": br._utc_now(),
                "status": "failed" if error else "succeeded",
                "error_type": type(error).__name__ if error else None,
                "logout_error_type": logout_error_type,
                "kb_id": args.kb_id,
                "result": redactor.clean(result),
                "api_snapshots": snapshot_info,
            },
        )
        _write_checksum_receipt(invocation_id)
    if error is not None:
        safe = redactor.text(str(error))
        raise br.BaselineRunnerError(f"{type(error).__name__}: {safe}") from error
    print(
        json.dumps(
            {
                "status": "succeeded",
                "action": args.action,
                "invocation_id": invocation_id,
                "profile": result.get("target")
                or result.get("current")
                or result.get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except br.BaselineRunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
