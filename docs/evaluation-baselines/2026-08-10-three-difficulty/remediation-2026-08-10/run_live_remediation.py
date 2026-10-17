#!/usr/bin/env python3
"""Run auditable live remediation candidates against the frozen three-level suite.

The script deliberately reuses the original baseline runner's authenticated,
redacted HTTP client and artifact verifier.  It never edits a dataset and it
never reads application databases directly.  Full-suite live candidates point
at the original root baselines.  A failed-case subset is intentionally created
as a standalone diagnostic run because the evaluation API requires a candidate
and its baseline to contain the same complete case set.  A final ``--as-root``
invocation can record a fresh post-remediation baseline after the quality
targets have been independently verified.

Examples (from the repository root)::

    python docs/evaluation-baselines/2026-08-10-three-difficulty/\
      remediation-2026-08-10/run_live_remediation.py \
      --label candidate-01 --selection baseline-failed

    python docs/evaluation-baselines/2026-08-10-three-difficulty/\
      remediation-2026-08-10/run_live_remediation.py \
      --label final-root --selection all --as-root

Use ``--resume`` with the same label after an interrupted poll or download.
Credentials are resolved exactly as in ``run_live_baselines.py`` and never
written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote


SCRIPT_PATH = Path(__file__).resolve()
REMEDIATION_ROOT = SCRIPT_PATH.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
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
DIFFICULTIES = ("easy", "medium", "hard")
INDEX_SCHEMA = "stt.eval.live-remediation-run-index.v2"
PROFILE_PROVENANCE_SCHEMA = "stt.eval.retrieval-profile-provenance.v1"


def _load_baseline_runner():
    spec = importlib.util.spec_from_file_location(
        "stt_three_difficulty_baseline_runner", BASELINE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError("cannot load the baseline runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


br = _load_baseline_runner()
_BR_WRITE_JSON = br._write_json
_BR_WRITE_BYTES = br._write_bytes


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    for attempt in range(1, 6):
        try:
            _BR_WRITE_JSON(path, value)
            return
        except FileNotFoundError:
            if attempt == 5:
                raise
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.05 * attempt)


def _write_bytes(path: Path, value: bytes) -> None:
    for attempt in range(1, 6):
        try:
            _BR_WRITE_BYTES(path, value)
            return
        except FileNotFoundError:
            if attempt == 5:
                raise
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.05 * attempt)


# The reused recorder and artifact copier resolve these helpers from their
# module globals.  Install the same bounded Windows transient retry there so a
# vanished antivirus-scanned temporary file cannot strand a completed run.
br._write_json = _write_json
br._write_bytes = _write_bytes


def _safe_label(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized or not __import__("re").fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", normalized):
        raise br.BaselineRunnerError(
            "--label must contain 1..64 lowercase letters, digits, dots, underscores or dashes"
        )
    return normalized


def _profile_config_hash(value: str) -> str:
    """规范化 CLI 传入的 profile config SHA-256。"""

    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise br.BaselineRunnerError(
            "--retrieval-profile-config-hash must be a 64-character SHA-256 hex digest"
        )
    return normalized


def _parse_args(config: dict[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live remediation candidates and archive verified artifacts under docs."
    )
    parser.add_argument("--label", required=True, type=_safe_label)
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
        default=br._first_config(config, "STT_EVAL_KB_ID", "EVALUATION_KB_ID"),
    )
    parser.add_argument(
        "--difficulty",
        choices=DIFFICULTIES,
        action="append",
        help="difficulty to run; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--selection",
        choices=("all", "baseline-failed"),
        default="baseline-failed",
        help=(
            "run all cases or only cases failed by the original baseline; "
            "a subset is stored as a standalone diagnostic run"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("live", "retrieval_only"),
        default="live",
        help="full query evaluation or an LLM-free retrieval validation",
    )
    parser.add_argument(
        "--retrieval-profile-id",
        help="explicit profile lineage to freeze (must be paired with --retrieval-profile-version)",
    )
    parser.add_argument(
        "--retrieval-profile-version",
        type=int,
        help="explicit profile version to freeze, including a draft under validation",
    )
    parser.add_argument(
        "--retrieval-profile-config-hash",
        type=_profile_config_hash,
        help=(
            "expected config_hash for the explicit KB-scoped profile; required for a new "
            "explicit-profile run"
        ),
    )
    parser.add_argument(
        "--as-root",
        action="store_true",
        help="create root runs with baseline_run_id=null after targets are verified",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--poll-timeout",
        type=br._positive_float,
        default=br.DEFAULT_POLL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval",
        type=br._positive_float,
        default=br.DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout",
        type=br._positive_float,
        default=br.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    args = parser.parse_args()
    args.base_url = br._normalize_local_base_url(args.base_url)
    args.kb_id = br._safe_component(str(args.kb_id or ""), label="knowledge-base id")
    args.difficulty = tuple(dict.fromkeys(args.difficulty or DIFFICULTIES))
    if (args.retrieval_profile_id is None) != (args.retrieval_profile_version is None):
        raise br.BaselineRunnerError(
            "--retrieval-profile-id and --retrieval-profile-version must be supplied together"
        )
    if args.retrieval_profile_config_hash is not None and args.retrieval_profile_id is None:
        raise br.BaselineRunnerError(
            "--retrieval-profile-config-hash requires an explicit retrieval profile"
        )
    if (
        args.retrieval_profile_id is not None
        and args.retrieval_profile_config_hash is None
        and not args.resume
    ):
        raise br.BaselineRunnerError(
            "new explicit-profile runs require --retrieval-profile-config-hash"
        )
    if args.retrieval_profile_version is not None and args.retrieval_profile_version < 1:
        raise br.BaselineRunnerError("--retrieval-profile-version must be positive")
    if args.as_root and (
        args.mode != "live" or args.selection != "all" or args.difficulty != DIFFICULTIES
    ):
        raise br.BaselineRunnerError(
            "--as-root requires --mode live, --selection all, and all three difficulties"
        )
    return args


def _run_kind(args: argparse.Namespace) -> str:
    if args.as_root:
        return "proposed_root_baseline"
    if args.mode == "retrieval_only":
        return "unpaired_retrieval_validation"
    if args.selection == "baseline-failed":
        return "standalone_failure_diagnostic"
    return "paired_candidate"


def _registry() -> dict[str, Any]:
    value = _read_json(BASELINE_REGISTRY_PATH)
    if not isinstance(value, dict) or value.get("status") != "succeeded":
        raise br.BaselineRunnerError("the original baseline registry is not succeeded")
    records = value.get("baselines")
    if not isinstance(records, dict) or any(level not in records for level in DIFFICULTIES):
        raise br.BaselineRunnerError("the original baseline registry is incomplete")
    return value


def _ordered_case_id_sha256(case_ids: list[str]) -> str:
    encoded = json.dumps(case_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _baseline_case_snapshot(difficulty: str, run_id: str, *, selection: str) -> dict[str, Any]:
    path = SUITE_ROOT / "runs" / difficulty / run_id / "api" / "results.json"
    payload = _read_json(path)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise br.BaselineRunnerError(f"missing archived baseline results for {difficulty}")
    case_ids: list[str] = []
    failed: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise br.BaselineRunnerError(f"invalid archived baseline result row for {difficulty}")
        case_id = str(row.get("case_id") or "")
        if not case_id or str(row.get("run_id") or "") != run_id:
            raise br.BaselineRunnerError(f"baseline result identity is invalid for {difficulty}")
        case_ids.append(case_id)
        if row.get("passed") is False:
            failed.append(case_id)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise br.BaselineRunnerError(f"invalid baseline case set for {difficulty}")
    selected = case_ids if selection == "all" else failed
    if not selected:
        raise br.BaselineRunnerError(f"no cases selected for {difficulty}")
    return {
        "baseline_results_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "baseline_results_sha256": br._sha256_file(path),
        "all_case_count": len(case_ids),
        "selected_case_ids": selected,
        "selected_case_ids_sha256": _ordered_case_id_sha256(selected),
        "request_case_ids": None if selection == "all" else selected,
    }


def _paired_baseline_run_id(args: argparse.Namespace, source_run_id: str) -> str | None:
    """Return a baseline only when the API's complete-pairing contract holds."""

    if args.as_root or args.mode != "live" or args.selection != "all":
        return None
    return source_run_id


def _new_index(args: argparse.Namespace, registry: dict[str, Any]) -> dict[str, Any]:
    now = br._utc_now()
    return {
        "schema_version": INDEX_SCHEMA,
        "label": args.label,
        "status": "in_progress",
        "run_kind": _run_kind(args),
        "mode": args.mode,
        "selection": args.selection,
        "as_root": bool(args.as_root),
        "planned_difficulties": list(args.difficulty),
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "retrieval_profile_config_hash": args.retrieval_profile_config_hash,
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "source_baseline_registry": {
            "path": BASELINE_REGISTRY_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": br._sha256_file(BASELINE_REGISTRY_PATH),
            "generated_at": registry.get("generated_at"),
        },
        "created_at": now,
        "updated_at": now,
        "difficulties": {},
    }


def _persist(index_path: Path, index: dict[str, Any]) -> None:
    index["updated_at"] = br._utc_now()
    _write_json(index_path, index)


def _prepare_index(
    args: argparse.Namespace, registry: dict[str, Any], index_path: Path
) -> dict[str, Any]:
    if args.resume:
        if not index_path.is_file():
            raise br.BaselineRunnerError("--resume requires an existing label index")
        index = _read_json(index_path)
        identity = (
            index.get("schema_version"),
            index.get("label"),
            index.get("run_kind"),
            index.get("selection"),
            index.get("mode"),
            bool(index.get("as_root")),
            tuple(index.get("planned_difficulties") or []),
            index.get("retrieval_profile_id"),
            index.get("retrieval_profile_version"),
            index.get("base_url"),
            index.get("kb_id"),
        )
        expected = (
            INDEX_SCHEMA,
            args.label,
            _run_kind(args),
            args.selection,
            args.mode,
            bool(args.as_root),
            args.difficulty,
            args.retrieval_profile_id,
            args.retrieval_profile_version,
            args.base_url,
            args.kb_id,
        )
        if identity != expected:
            raise br.BaselineRunnerError("--resume identity does not match the saved index")
        saved_profile_hash = index.get("retrieval_profile_config_hash")
        requested_profile_hash = args.retrieval_profile_config_hash
        if (
            saved_profile_hash is not None
            and requested_profile_hash is not None
            and saved_profile_hash != requested_profile_hash
        ):
            raise br.BaselineRunnerError(
                "--resume retrieval profile config hash does not match the saved index"
            )
        if saved_profile_hash is None and requested_profile_hash is not None:
            index["retrieval_profile_config_hash"] = requested_profile_hash
        frozen_registry = index.get("source_baseline_registry") or {}
        if frozen_registry.get("sha256") != br._sha256_file(BASELINE_REGISTRY_PATH):
            raise br.BaselineRunnerError(
                "cannot resume because the frozen baseline registry changed"
            )
        index["status"] = "in_progress"
        return index
    if index_path.exists():
        raise br.BaselineRunnerError(
            f"label {args.label!r} already exists; use --resume or choose another label"
        )
    return _new_index(args, registry)


def _profile_request_pin(
    args: argparse.Namespace,
    index: dict[str, Any],
) -> dict[str, Any] | None:
    """返回本次 preflight 必须解析出的 KB-scoped profile 身份。"""

    if args.retrieval_profile_id is not None:
        expected_hash = args.retrieval_profile_config_hash or index.get(
            "retrieval_profile_config_hash"
        )
        frozen = index.get("retrieval_profile_provenance") or {}
        if expected_hash is None:
            expected_hash = frozen.get("config_hash")
        return {
            "profile_id": args.retrieval_profile_id,
            "version": args.retrieval_profile_version,
            "config_hash": expected_hash,
        }
    frozen = index.get("retrieval_profile_provenance")
    if isinstance(frozen, dict) and frozen.get("profile_id"):
        return {
            "profile_id": frozen.get("profile_id"),
            "version": frozen.get("version"),
            "config_hash": frozen.get("config_hash"),
        }
    return None


def _server_profile_config_hash(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise br.BaselineRunnerError(
            "evaluation preflight returned an invalid retrieval profile config hash"
        )
    return normalized


def _assert_preflight_profile_identity(
    preflight: dict[str, Any],
    *,
    kb_id: str,
    expected_pin: dict[str, Any] | None,
) -> dict[str, Any]:
    """验证 evaluation 实际解析出的 KB/profile/hash，并返回可冻结回执。"""

    knowledge_base = preflight.get("knowledge_base") or {}
    profile = preflight.get("retrieval_profile") or {}
    selection = preflight.get("retrieval_profile_selection") or {}
    wrong: list[str] = []
    if knowledge_base.get("id") != kb_id:
        wrong.append("knowledge_base.id")
    if profile.get("kb_id") != kb_id:
        wrong.append("retrieval_profile.kb_id")

    profile_id = str(profile.get("profile_id") or "")
    version = profile.get("version")
    tenant_id = str(profile.get("tenant_id") or "")
    if not profile_id:
        wrong.append("retrieval_profile.profile_id")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        wrong.append("retrieval_profile.version")
    if not tenant_id:
        wrong.append("retrieval_profile.tenant_id")
    config_hash = _server_profile_config_hash(profile.get("config_hash"))

    if expected_pin is not None:
        if profile_id != expected_pin.get("profile_id"):
            wrong.append("retrieval_profile.profile_id")
        if version != expected_pin.get("version"):
            wrong.append("retrieval_profile.version")
        expected_hash = expected_pin.get("config_hash")
        if expected_hash is not None and config_hash != expected_hash:
            wrong.append("retrieval_profile.config_hash")

    if selection.get("profile_id") != profile_id:
        wrong.append("retrieval_profile_selection.profile_id")
    if selection.get("version") != version:
        wrong.append("retrieval_profile_selection.version")
    try:
        selection_hash = _server_profile_config_hash(selection.get("config_hash"))
    except br.BaselineRunnerError:
        selection_hash = ""
    if selection_hash != config_hash:
        wrong.append("retrieval_profile_selection.config_hash")
    if wrong:
        raise br.BaselineRunnerError(
            "evaluation preflight retrieval profile identity mismatch: "
            + ", ".join(dict.fromkeys(wrong))
        )

    identity = {
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "profile_id": profile_id,
        "version": version,
        "config_hash": config_hash,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": PROFILE_PROVENANCE_SCHEMA,
        **identity,
        "status": str(profile.get("status") or ""),
        "selection_source": str(selection.get("source") or ""),
        "identity_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _freeze_profile_provenance(
    index: dict[str, Any],
    entry: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """跨 difficulty/resume 冻结同一 profile provenance。"""

    identity_keys = (
        "schema_version",
        "tenant_id",
        "kb_id",
        "profile_id",
        "version",
        "config_hash",
        "identity_sha256",
    )
    for owner, container in (("run index", index), ("difficulty entry", entry)):
        existing = container.get("retrieval_profile_provenance")
        if existing is not None:
            if any(existing.get(key) != provenance.get(key) for key in identity_keys):
                raise br.BaselineRunnerError(
                    f"{owner} retrieval profile provenance changed before run creation"
                )
            continue
        container["retrieval_profile_provenance"] = provenance


def _poll(
    client: Any,
    *,
    difficulty: str,
    run_id: str,
    args: argparse.Namespace,
    index_path: Path,
    index: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = f"/api/v1/evaluation/runs/{quote(run_id, safe='')}"
    deadline = time.monotonic() + args.poll_timeout
    last: tuple[str, int, int] | None = None
    while True:
        envelope = client.request_json("GET", path)
        run = br._envelope_data(envelope, operation=f"poll {difficulty}")
        if not isinstance(run, dict) or str(run.get("id") or "") != run_id:
            raise br.BaselineRunnerError(f"poll {difficulty} returned the wrong run")
        current = (
            str(run.get("status") or ""),
            int(run.get("completed_cases") or 0),
            int(run.get("total_cases") or 0),
        )
        if current != last:
            print(f"[{difficulty}] {run_id}: {current[0]} ({current[1]}/{current[2]})", flush=True)
            entry = index["difficulties"][difficulty]
            entry.update(
                {"status": current[0], "completed_cases": current[1], "total_cases": current[2]}
            )
            _persist(index_path, index)
            last = current
        if current[0] in br.TERMINAL_RUN_STATUSES:
            return envelope, run
        if current[0] not in br.ACTIVE_RUN_STATUSES:
            raise br.BaselineRunnerError(f"unsupported run status {current[0]!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise br.BaselineRunnerError("poll timeout reached; rerun with --resume")
        time.sleep(min(args.poll_interval, remaining))


def _assert_entry_identity(entry: dict[str, Any], expected: dict[str, Any]) -> None:
    frozen_keys = (
        "difficulty",
        "dataset_id",
        "dataset_key",
        "dataset_version",
        "dataset_checksum",
        "dataset_source_path",
        "dataset_source_sha256",
        "source_baseline_run_id",
        "source_baseline_results_path",
        "source_baseline_results_sha256",
        "baseline_run_id",
        "mode",
        "retrieval_profile_id",
        "retrieval_profile_version",
        "retrieval_profile_config_hash",
        "selection",
        "selected_case_count",
        "selected_case_ids",
        "selected_case_ids_sha256",
    )
    mismatched = [
        key
        for key in frozen_keys
        if entry.get(key) != expected.get(key)
        and not (key == "retrieval_profile_config_hash" and entry.get(key) is None)
    ]
    if mismatched:
        raise br.BaselineRunnerError("saved difficulty identity changed: " + ", ".join(mismatched))


def _assert_run_identity(
    run: dict[str, Any],
    *,
    run_id: str,
    args: argparse.Namespace,
    expected: dict[str, Any],
    profile_provenance: dict[str, Any],
) -> None:
    checks = {
        "id": run_id,
        "dataset_id": expected["dataset_id"],
        "kb_id": args.kb_id,
        "mode": args.mode,
        "baseline_run_id": expected["baseline_run_id"],
        "total_cases": expected["selected_case_count"],
    }
    wrong = [key for key, value in checks.items() if run.get(key) != value]
    snapshot = run.get("config_snapshot") or {}
    selected = list(snapshot.get("selected_case_ids") or [])
    if selected != expected["selected_case_ids"]:
        wrong.append("config_snapshot.selected_case_ids")
    if snapshot.get("dataset_checksum") != expected["dataset_checksum"]:
        wrong.append("config_snapshot.dataset_checksum")
    profile = snapshot.get("retrieval_profile") or {}
    for key in ("kb_id", "profile_id", "version", "config_hash"):
        if profile.get(key) != profile_provenance.get(key):
            wrong.append(f"config_snapshot.retrieval_profile.{key}")
    if wrong:
        raise br.BaselineRunnerError(
            "run identity does not match the frozen request: " + ", ".join(wrong)
        )


def _assert_results_identity(results: Any, *, run_id: str, expected_case_ids: list[str]) -> None:
    if not isinstance(results, list):
        raise br.BaselineRunnerError("results payload is not a list")
    actual: list[str] = []
    for row in results:
        if not isinstance(row, dict) or str(row.get("run_id") or "") != run_id:
            raise br.BaselineRunnerError("result row has the wrong run identity")
        actual.append(str(row.get("case_id") or ""))
    if (
        any(not case_id for case_id in actual)
        or len(actual) != len(set(actual))
        or actual != expected_case_ids
    ):
        raise br.BaselineRunnerError("result case set does not match the frozen selection")


def _verify_archived_entry(entry: dict[str, Any]) -> None:
    receipt = entry.get("artifact_copy_manifest") or {}
    relative = str(receipt.get("path") or "")
    expected_sha = str(receipt.get("sha256") or "")
    if not relative or not expected_sha:
        raise br.BaselineRunnerError("archived entry has no frozen artifact receipt")
    manifest_path = (REMEDIATION_ROOT / relative).resolve()
    try:
        manifest_path.relative_to(REMEDIATION_ROOT.resolve())
    except ValueError as exc:
        raise br.BaselineRunnerError("archived artifact receipt escapes remediation root") from exc
    if not manifest_path.is_file() or br._sha256_file(manifest_path) != expected_sha:
        raise br.BaselineRunnerError("archived artifact receipt changed")
    manifest = _read_json(manifest_path)
    for copy in (manifest.get("copies") or {}).values():
        target = copy.get("target") or {}
        target_path = (REMEDIATION_ROOT / str(target.get("relative_path") or "")).resolve()
        try:
            target_path.relative_to(REMEDIATION_ROOT.resolve())
        except ValueError as exc:
            raise br.BaselineRunnerError("archived artifact escapes remediation root") from exc
        if not target_path.is_file() or br._sha256_file(target_path) != target.get("sha256"):
            raise br.BaselineRunnerError("archived artifact changed")


def _load_frozen_copy_manifest(
    manifest_path: Path, *, run: dict[str, Any], difficulty: str
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != run.get("id") or manifest.get("difficulty") != difficulty:
        raise br.BaselineRunnerError("existing artifact receipt has the wrong identity")
    copies = manifest.get("copies") or {}
    report_paths = run.get("report_paths") or {}
    if set(copies) != set(br.ARTIFACT_FILENAMES):
        raise br.BaselineRunnerError("existing artifact receipt is incomplete")
    for artifact, copy in copies.items():
        source = copy.get("source") or {}
        target = copy.get("target") or {}
        server = report_paths.get(artifact) or {}
        if source.get("server_sha256") != server.get("sha256"):
            raise br.BaselineRunnerError("server artifact digest changed on resume")
        target_path = (REMEDIATION_ROOT / str(target.get("relative_path") or "")).resolve()
        try:
            target_path.relative_to(REMEDIATION_ROOT.resolve())
        except ValueError as exc:
            raise br.BaselineRunnerError("artifact target escapes remediation root") from exc
        if not target_path.is_file() or br._sha256_file(target_path) != target.get("sha256"):
            raise br.BaselineRunnerError("existing artifact target changed")
    return manifest


def _process_difficulty(
    *,
    difficulty: str,
    args: argparse.Namespace,
    client: Any,
    redactor: Any,
    registry: dict[str, Any],
    index_path: Path,
    index: dict[str, Any],
    invocation_id: str,
) -> None:
    baseline = registry["baselines"][difficulty]
    dataset = baseline.get("dataset") or {}
    dataset_id = br._safe_component(str(dataset.get("id") or ""), label="dataset id")
    source_run_id = br._safe_component(str(baseline.get("run_id") or ""), label="baseline run id")
    source_relative = str(dataset.get("source_path") or "")
    source_path = (SUITE_ROOT / source_relative).resolve()
    try:
        source_path.relative_to(SUITE_ROOT.resolve())
    except ValueError as exc:
        raise br.BaselineRunnerError("dataset source escapes the frozen suite") from exc
    if not source_path.is_file():
        raise br.BaselineRunnerError(f"missing frozen dataset source for {difficulty}")
    actual_source_sha = br._sha256_file(source_path)
    if actual_source_sha != dataset.get("source_sha256"):
        raise br.BaselineRunnerError(f"frozen dataset source changed for {difficulty}")
    cases = _baseline_case_snapshot(difficulty, source_run_id, selection=args.selection)
    selected_case_ids = list(cases["selected_case_ids"])
    selected_count = len(selected_case_ids)
    if int(dataset.get("case_count") or 0) != int(cases["all_case_count"]):
        raise br.BaselineRunnerError(f"baseline case count changed for {difficulty}")
    case_ids = cases["request_case_ids"]
    expected_entry = {
        "difficulty": difficulty,
        "dataset_id": dataset_id,
        "dataset_key": dataset.get("key"),
        "dataset_version": dataset.get("version"),
        "dataset_checksum": dataset.get("checksum"),
        "dataset_source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_source_sha256": actual_source_sha,
        "source_baseline_run_id": source_run_id,
        "source_baseline_results_path": cases["baseline_results_path"],
        "source_baseline_results_sha256": cases["baseline_results_sha256"],
        "baseline_run_id": _paired_baseline_run_id(args, source_run_id),
        "mode": args.mode,
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "retrieval_profile_config_hash": args.retrieval_profile_config_hash,
        "selection": args.selection,
        "selected_case_count": selected_count,
        "selected_case_ids": selected_case_ids,
        "selected_case_ids_sha256": cases["selected_case_ids_sha256"],
    }
    existing = index["difficulties"].get(difficulty)
    if isinstance(existing, dict):
        _assert_entry_identity(existing, expected_entry)
        entry = existing
        if args.resume and entry.get("status") == "archived":
            _verify_archived_entry(entry)
            print(f"[{difficulty}] verified existing immutable archive", flush=True)
            return
    else:
        entry = {**expected_entry, "status": "preflight_pending"}
        index["difficulties"][difficulty] = entry
    _persist(index_path, index)

    preflight_body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "kb_id": args.kb_id,
        "mode": args.mode,
    }
    profile_pin = _profile_request_pin(args, index)
    if profile_pin is not None:
        preflight_body.update(
            {
                "retrieval_profile_id": profile_pin["profile_id"],
                "retrieval_profile_version": profile_pin["version"],
            }
        )
    preflight_envelope = client.request_json(
        "POST",
        "/api/v1/evaluation/preflight",
        body=preflight_body,
    )
    preflight = br._envelope_data(preflight_envelope, operation=f"preflight {difficulty}")
    if not isinstance(preflight, dict) or not preflight.get("ready"):
        raise br.BaselineRunnerError(f"{difficulty} live preflight is not ready")
    profile_provenance = _assert_preflight_profile_identity(
        preflight,
        kb_id=args.kb_id,
        expected_pin=profile_pin,
    )
    _freeze_profile_provenance(index, entry, profile_provenance)
    _persist(index_path, index)

    run_id = str(entry.get("run_id") or "") if args.resume else ""
    create_envelope: dict[str, Any] | None = None
    if run_id:
        run_id = br._safe_component(run_id, label="run id")
        initial_envelope = client.request_json(
            "GET", f"/api/v1/evaluation/runs/{quote(run_id, safe='')}"
        )
        initial_run = br._envelope_data(initial_envelope, operation=f"resume {difficulty}")
    else:
        body: dict[str, Any] = {
            "dataset_id": dataset_id,
            "kb_id": args.kb_id,
            "mode": args.mode,
            "name": (f"three-difficulty-remediation-{args.label}-{difficulty}-{args.mode}"),
            "baseline_run_id": _paired_baseline_run_id(args, source_run_id),
        }
        if case_ids is not None:
            body["case_ids"] = case_ids
        body.update(
            {
                "retrieval_profile_id": profile_provenance["profile_id"],
                "retrieval_profile_version": profile_provenance["version"],
            }
        )
        create_envelope = client.request_json("POST", "/api/v1/evaluation/runs", body=body)
        initial_envelope = create_envelope
        created = br._envelope_data(create_envelope, operation=f"create {difficulty}")
        if not isinstance(created, dict):
            raise br.BaselineRunnerError(f"create {difficulty} returned invalid data")
        run_id = br._safe_component(str(created.get("id") or ""), label="run id")
        initial_run = created
        entry.update({"run_id": run_id, "status": created.get("status")})
        _persist(index_path, index)
    if not isinstance(initial_run, dict):
        raise br.BaselineRunnerError(f"{difficulty} run snapshot is invalid")
    _assert_run_identity(
        initial_run,
        run_id=run_id,
        args=args,
        expected=expected_entry,
        profile_provenance=profile_provenance,
    )

    run_dir = br.RUNS_DIR / difficulty / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep the immutable API-copy leaf short enough for Windows MAX_PATH while
    # the full invocation identity remains frozen in the index and recorder.
    api_copy_key = invocation_id.rsplit("-", maxsplit=1)[-1]
    api_copy_dir = run_dir / "api" / api_copy_key
    api_copy_dir.mkdir(parents=True, exist_ok=False)
    _write_json(api_copy_dir / "preflight.json", redactor.clean(preflight_envelope))
    _write_json(
        api_copy_dir / ("run-create.json" if create_envelope is not None else "run-resume.json"),
        redactor.clean(initial_envelope),
    )

    final_envelope, final_run = _poll(
        client,
        difficulty=difficulty,
        run_id=run_id,
        args=args,
        index_path=index_path,
        index=index,
    )
    _write_json(api_copy_dir / "run-final.json", redactor.clean(final_envelope))
    if final_run.get("status") != "succeeded":
        raise br.BaselineRunnerError(
            f"{difficulty} run ended as {final_run.get('status')} ({final_run.get('error_code')})"
        )
    _assert_run_identity(
        final_run,
        run_id=run_id,
        args=args,
        expected=expected_entry,
        profile_provenance=profile_provenance,
    )

    encoded = quote(run_id, safe="")
    results_envelope = client.request_json("GET", f"/api/v1/evaluation/runs/{encoded}/results")
    comparison_envelope = client.request_json(
        "GET", f"/api/v1/evaluation/runs/{encoded}/comparison"
    )
    _write_json(api_copy_dir / "results.json", redactor.clean(results_envelope))
    _write_json(api_copy_dir / "comparison.json", redactor.clean(comparison_envelope))
    results = br._envelope_data(results_envelope, operation=f"results {difficulty}")
    _assert_results_identity(
        results,
        run_id=run_id,
        expected_case_ids=selected_case_ids,
    )
    manifest_path = run_dir / "artifact-copy-manifest.json"
    if manifest_path.is_file():
        copy_manifest = _load_frozen_copy_manifest(
            manifest_path, run=final_run, difficulty=difficulty
        )
    else:
        manifest_path, copy_manifest = br._archive_artifacts(
            client,
            difficulty=difficulty,
            run=final_run,
            allow_repair=False,
        )
    if not all(
        bool(value.get("source_target_sha256_match")) for value in copy_manifest["copies"].values()
    ):
        raise br.BaselineRunnerError(f"{difficulty} artifact verification failed")
    entry.update(
        redactor.clean(
            {
                "status": "archived",
                "completed_at": final_run.get("completed_at"),
                "completed_cases": final_run.get("completed_cases"),
                "total_cases": final_run.get("total_cases"),
                "gate_result": final_run.get("gate_result") or {},
                "metrics_summary": final_run.get("metrics_summary") or {},
                "comparison_summary": final_run.get("comparison_summary") or {},
                "config_snapshot": final_run.get("config_snapshot") or {},
                "run_directory": br._relative(run_dir),
                "artifact_copy_manifest": {
                    "path": br._relative(manifest_path),
                    "sha256": br._sha256_file(manifest_path),
                },
                "api_copy": {
                    "directory": br._relative(api_copy_dir),
                    "invocation_id": invocation_id,
                },
            }
        )
    )
    _persist(index_path, index)


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    return path.is_symlink() or bool(attributes & 0x400)


def _prepare_label_root(args: argparse.Namespace) -> Path:
    runs_root = REMEDIATION_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(runs_root):
        raise br.BaselineRunnerError("remediation runs root cannot be a reparse point")
    label_root = runs_root / args.label
    if label_root.resolve().parent != runs_root.resolve():
        raise br.BaselineRunnerError("label path escapes remediation runs root")
    if args.resume:
        if not label_root.is_dir() or _is_reparse_point(label_root):
            raise br.BaselineRunnerError("--resume requires a regular existing label directory")
    else:
        if label_root.exists():
            raise br.BaselineRunnerError(
                f"label {args.label!r} already exists; use --resume or choose another label"
            )
        label_root.mkdir(parents=False, exist_ok=False)
    return label_root


def _write_checksums(label_root: Path) -> None:
    target = label_root / "SHA256SUMS"
    for path in label_root.rglob("*"):
        if _is_reparse_point(path):
            raise br.BaselineRunnerError("refusing to checksum a reparse point")
    files = sorted(
        (
            path
            for path in label_root.rglob("*")
            if path.is_file() and path.resolve() != target.resolve()
        ),
        key=lambda path: path.relative_to(label_root).as_posix(),
    )
    content = "".join(
        f"{br._sha256_file(path)}  {path.relative_to(label_root).as_posix()}\n" for path in files
    )
    _write_bytes(target, content.encode("utf-8"))


def main() -> int:
    config = br._load_environment()
    args = _parse_args(config)
    registry = _registry()
    if str(registry.get("kb_id") or "") != args.kb_id:
        raise br.BaselineRunnerError("--kb-id does not match the frozen baseline registry")

    label_root = _prepare_label_root(args)
    index_path = label_root / "run-index.json"
    br.SUITE_ROOT = REMEDIATION_ROOT
    br.RUNS_DIR = label_root / "artifacts"
    br.API_SNAPSHOT_ROOT = label_root / "api-snapshots"
    index = _prepare_index(args, registry, index_path)

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
    tenant_id = br._first_config(config, "STT_EVAL_TENANT_ID", "AUTH_BOOTSTRAP_TENANT_ID") or None
    if not password:
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.shared.config.settings_config import settings

        email = settings.auth_bootstrap_admin_email
        password = settings.auth_bootstrap_admin_password
    if not password:
        raise br.BaselineRunnerError("bootstrap admin password could not be resolved")

    redactor = br.Redactor(password)
    invocation_id = f"{br._timestamp_slug()}-{uuid.uuid4().hex[:8]}"
    invocation = {
        "id": invocation_id,
        "started_at": br._utc_now(),
        "status": "running",
        "resume": bool(args.resume),
    }
    index.setdefault("invocations", []).append(invocation)
    _persist(index_path, index)
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
    try:
        client.login()
        for difficulty in args.difficulty:
            _process_difficulty(
                difficulty=difficulty,
                args=args,
                client=client,
                redactor=redactor,
                registry=registry,
                index_path=index_path,
                index=index,
                invocation_id=invocation_id,
            )
        incomplete = [
            difficulty
            for difficulty in args.difficulty
            if (index["difficulties"].get(difficulty) or {}).get("status") != "archived"
        ]
        if incomplete:
            raise br.BaselineRunnerError(
                "planned difficulties were not archived: " + ", ".join(incomplete)
            )
        index["status"] = "succeeded"
    except Exception as exc:
        error = exc
        index["status"] = "failed"
        index["error_type"] = type(exc).__name__
    finally:
        try:
            client.logout()
        except Exception as logout_exc:
            invocation["logout_error_type"] = type(logout_exc).__name__
        invocation["ended_at"] = br._utc_now()
        invocation["status"] = "failed" if error else "succeeded"
        if error is not None:
            invocation["error_type"] = type(error).__name__
        finalization_error: Exception | None = None
        try:
            invocation["api_snapshots"] = recorder.close(
                status=invocation["status"],
                error_type=invocation.get("error_type"),
            )
            _persist(index_path, redactor.clean(index))
            _write_checksums(label_root)
        except Exception as exc:
            finalization_error = exc
            if error is None:
                index["status"] = "failed"
                index["error_type"] = type(exc).__name__
    if error is not None:
        safe = redactor.text(str(error))
        raise br.BaselineRunnerError(f"{type(error).__name__}: {safe}") from error
    if finalization_error is not None:
        safe = redactor.text(str(finalization_error))
        raise br.BaselineRunnerError(
            f"finalization {type(finalization_error).__name__}: {safe}"
        ) from finalization_error
    print(f"Archived live remediation run under {label_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except br.BaselineRunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
