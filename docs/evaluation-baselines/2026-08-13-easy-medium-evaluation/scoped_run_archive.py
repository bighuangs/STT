#!/usr/bin/env python3
"""统一校验 easy/medium scoped run 的完整、不可晋级归档契约。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_easy_medium_live as live


DIFFICULTY_ENTRY_ALLOWED_FIELDS = frozenset(
    {
        "difficulty",
        "role",
        "mode",
        "baseline_run_id",
        "status",
        "dataset_id",
        "dataset_key",
        "dataset_version",
        "dataset_checksum",
        "dataset_source_path",
        "dataset_source_sha256",
        "selected_case_count",
        "selected_case_ids",
        "created_at",
        "completed_at",
        "completed_cases",
        "total_cases",
        "run_id",
        "run_directory",
        "artifact_copy_manifest",
        "server_artifact_receipt",
        "results_path",
        "results_sha256",
        "target_result",
        "target_required_correct_count",
        "target_passed",
        "gate_result_diagnostic_only",
        "metrics_summary_diagnostic_only",
        "comparison_summary_diagnostic_only",
        "config_snapshot",
    }
)
ARTIFACT_COPY_RECEIPT_FIELDS = frozenset({"path", "sha256"})
SERVER_ARTIFACT_RECEIPT_FIELDS = frozenset({"path", "sha256", "size_bytes"})


class ScopedRunArchiveError(RuntimeError):
    """表示运行身份或归档证据不能形成可信的 scoped execution snapshot。"""


def _fail(message: str) -> None:
    raise ScopedRunArchiveError(message)


def _expected_dataset_source_path(root: Path, path: Path) -> str:
    """正式根使用项目相对路径；临时契约夹具使用 scoped-root 相对路径。"""

    if root.resolve() != live.FINAL_ROOT.resolve():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as exc:  # pragma: no cover - suite loader 已限制路径
            raise ScopedRunArchiveError("dataset path escapes the scoped fixture root") from exc
    try:
        return path.resolve().relative_to(live.PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ScopedRunArchiveError("dataset path escapes the project root") from exc


def _expected_runner() -> dict[str, str]:
    return {
        "path": live._relative_project(live.SCRIPT_PATH),
        "sha256": live._sha256_file(
            live.SCRIPT_PATH,
            root=live.PROJECT_ROOT,
            label="scoped live runner",
        ),
        "shared_validation_path": live._relative_project(live.full.SCRIPT_PATH),
        "shared_validation_sha256": live._sha256_file(
            live.full.SCRIPT_PATH,
            root=live.PROJECT_ROOT,
            label="full live validation runtime",
        ),
        "shared_client_path": live._relative_project(live.SOURCE_RUNNER_PATH),
        "shared_client_sha256": live._sha256_file(
            live.SOURCE_RUNNER_PATH,
            root=live.PROJECT_ROOT,
            label="shared live runner",
        ),
    }


def _validate_completion(value: Any) -> None:
    expected = {
        "schema_version": live.COMPLETION_SCHEMA,
        "status": "succeeded",
        "role": live.RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "scope": live.scope.scope_profile(),
        "difficulty_order": list(live.DIFFICULTIES),
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != {*expected, "completed_at"}
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
        or not isinstance(value.get("completed_at"), str)
        or not value["completed_at"].strip()
    ):
        _fail("scope_completion does not exactly bind the succeeded scoped run")


def _validate_execution_identities(index: Mapping[str, Any]) -> None:
    try:
        current_code_revision = live.full._current_code_revision()
        current_runtime = live.full._local_runtime_identity()
    except live.full.FinalLiveError as exc:
        raise ScopedRunArchiveError(str(exc)) from exc
    if index.get("code_revision") != current_code_revision:
        _fail("code revision differs from the frozen scoped execution")
    if index.get("runtime_identity") != current_runtime:
        _fail("runtime identity differs from the frozen scoped execution")
    if index.get("runner") != _expected_runner():
        _fail("runner identity or code SHA differs from the frozen scoped execution")

    profile = index.get("retrieval_profile_identity")
    expected_profile_fields = {
        "profile_id",
        "version",
        "config_hash",
        "snapshot_sha256",
        "vector_read_set_digest",
    }
    if (
        not isinstance(profile, Mapping)
        or set(profile) != expected_profile_fields
        or profile.get("profile_id") != index.get("retrieval_profile_id")
        or profile.get("version") != index.get("retrieval_profile_version")
        or any(
            not isinstance(profile.get(field), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(profile.get(field)))
            for field in ("config_hash", "snapshot_sha256", "vector_read_set_digest")
        )
    ):
        _fail("retrieval profile identity is incomplete or inconsistent")

    preflight = index.get("preflight_identity")
    if (
        not isinstance(preflight, Mapping)
        or set(preflight) != {"invariant_sha256", "evaluation_mode", "checks"}
        or not re.fullmatch(r"[0-9a-f]{64}", str(preflight.get("invariant_sha256") or ""))
        or preflight.get("evaluation_mode") != "live"
        or not isinstance(preflight.get("checks"), Mapping)
        or not preflight["checks"]
        or any(value is not True for value in preflight["checks"].values())
    ):
        _fail("preflight identity is incomplete or not all-ready")
    execution = index.get("execution_config_identity")
    expected_execution_fields = {
        "snapshot_invariant_sha256",
        "config_version",
        "llm_model",
        "embedding_model",
        "reranker_model",
        "gate_policy_sha256",
        "evaluation_security_sha256",
        "vector_read_scopes_sha256",
        "telemetry_coverage_sha256",
    }
    if not isinstance(execution, Mapping) or set(execution) != expected_execution_fields:
        _fail("execution config identity is incomplete")


def validate_scoped_run_archive(
    index_path: Path,
    index: Mapping[str, Any],
    suite: Any,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """验证完整 execution snapshot，并返回已验证的逐难度结果证据。"""

    chosen_root = (root or suite.activation_path.parent).resolve()
    chosen_index = index_path.resolve()
    label = index.get("label")
    if not isinstance(label, str) or not label:
        _fail("run-index label is missing")
    try:
        if live.br._safe_component(label, label="scoped label") != label:
            _fail("run-index label is not canonical")
    except live.br.BaselineRunnerError as exc:
        raise ScopedRunArchiveError("run-index label is not safe") from exc
    expected_index = (chosen_root / "runs" / label / "run-index.json").resolve()
    if chosen_index != expected_index:
        _fail("run-index path does not match its frozen label")
    if not chosen_index.is_file():
        _fail("run-index file is missing")
    stored_index = live._read_json(
        chosen_index,
        root=chosen_root,
        label="scoped run index",
    )
    if not isinstance(stored_index, Mapping) or dict(stored_index) != dict(index):
        _fail("run-index argument differs from the physical frozen index")
    if (chosen_index.parent / "candidate-roots.json").exists():
        _fail("scoped run contains a forbidden candidate-roots artifact")

    expected_top_level = {
        "schema_version": live.INDEX_SCHEMA,
        "status": "succeeded",
        "label": label,
        "role": live.RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "scope": live.scope.scope_profile(),
        "mode": "live",
        "baseline_run_id": None,
        "selection": "all_scoped_cases",
        "planned_difficulties": list(live.DIFFICULTIES),
        "kb_id": live.EXPECTED_KB_ID,
        "source_index_identity": suite.source_index_identity,
        "static_checksums_sha256": suite.static_checksums_sha256,
    }
    changed = [
        field for field, expected in expected_top_level.items() if index.get(field) != expected
    ]
    for field in (
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
    ):
        if field in index and index[field] is not False:
            changed.append(field)
    if {"candidate_roots", "source_baseline_registry", "baselines"}.intersection(index):
        changed.append("forbidden promotion fields")
    if changed:
        _fail("run-index frozen identity changed: " + ", ".join(sorted(changed)))

    base_url = index.get("base_url")
    try:
        if not isinstance(base_url, str) or live.br._normalize_local_base_url(base_url) != base_url:
            _fail("base_url is not the exact normalized local endpoint")
    except (ValueError, live.br.BaselineRunnerError) as exc:
        raise ScopedRunArchiveError("base_url is not the exact normalized local endpoint") from exc

    _validate_execution_identities(index)
    _validate_completion(index.get("scope_completion"))

    expected_activation = {
        "path": suite.activation_path.resolve().relative_to(chosen_root).as_posix(),
        "sha256": suite.activation_sha256,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
    }
    if index.get("activation") != expected_activation:
        _fail("activation receipt changed")
    expected_quality = {
        "path": suite.quality_targets_path.resolve().relative_to(chosen_root).as_posix(),
        "sha256": suite.quality_targets_sha256,
        "schema_version": live.QUALITY_TARGETS_SCHEMA,
    }
    if index.get("quality_targets") != expected_quality:
        _fail("quality-target receipt changed")
    expected_target = {
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "strict_metrics_count_toward_target": False,
        "blind_audit_counts_toward_target": False,
        "correct_counts": live.TARGET_CORRECT_COUNTS,
        "case_count": live.EXPECTED_CASE_COUNT,
    }
    if index.get("target_contract") != expected_target:
        _fail("target contract changed")

    dataset_files = index.get("dataset_files")
    entries = index.get("difficulties")
    if (
        not isinstance(dataset_files, Mapping)
        or set(dataset_files) != set(live.DIFFICULTIES)
        or not isinstance(entries, Mapping)
        or set(entries) != set(live.DIFFICULTIES)
    ):
        _fail("archive must contain exactly easy and medium")

    verified: dict[str, Any] = {}
    all_case_ids: set[str] = set()
    label_root = chosen_index.parent
    for difficulty in live.DIFFICULTIES:
        spec = suite.datasets[difficulty]
        expected_dataset_receipt = {
            "path": spec.path.resolve().relative_to(chosen_root).as_posix(),
            "sha256": spec.file_sha256,
            "manifest_sha256": spec.manifest_sha256,
            "case_count": live.EXPECTED_CASE_COUNT,
            "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
            "version": live.EXPECTED_DATASET_VERSION,
        }
        if dataset_files[difficulty] != expected_dataset_receipt:
            _fail(f"dataset_files.{difficulty} changed")
        entry = entries[difficulty]
        run_id = entry.get("run_id") if isinstance(entry, Mapping) else None
        selected = entry.get("selected_case_ids") if isinstance(entry, Mapping) else None
        expected_entry = {
            "difficulty": difficulty,
            "role": live.RUN_ROLE,
            "mode": "live",
            "baseline_run_id": None,
            "status": "archived",
            "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
            "dataset_version": live.EXPECTED_DATASET_VERSION,
            "dataset_checksum": spec.manifest_sha256,
            "dataset_source_path": _expected_dataset_source_path(chosen_root, spec.path),
            "dataset_source_sha256": spec.file_sha256,
            "selected_case_count": live.EXPECTED_CASE_COUNT,
            "selected_case_ids": list(spec.case_ids),
            "completed_cases": live.EXPECTED_CASE_COUNT,
            "total_cases": live.EXPECTED_CASE_COUNT,
            "target_required_correct_count": live._minimum_correct(difficulty),
            "target_passed": True,
        }
        if (
            not isinstance(entry, Mapping)
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(entry.get("dataset_id"), str)
            or not entry["dataset_id"]
            or any(entry.get(field) != expected for field, expected in expected_entry.items())
        ):
            _fail(f"difficulties.{difficulty} frozen identity changed")
        authority_drift = live._promotion_authority_drift(entry)
        unexpected_entry_fields = set(entry) - DIFFICULTY_ENTRY_ALLOWED_FIELDS
        if authority_drift or unexpected_entry_fields:
            fields = sorted({*authority_drift, *unexpected_entry_fields})
            _fail(
                f"difficulties.{difficulty} contains forbidden identity fields: "
                + ", ".join(fields)
            )
        if all_case_ids.intersection(selected):
            _fail("selected case ids overlap across difficulties")
        all_case_ids.update(selected)

        expected_run_dir = label_root / "artifacts" / difficulty / run_id
        expected_run_relative = expected_run_dir.relative_to(chosen_root).as_posix()
        if entry.get("run_directory") != expected_run_relative:
            _fail(f"{difficulty} run directory receipt changed")
        artifact_copy_receipt = entry.get("artifact_copy_manifest")
        server_artifact_receipt = entry.get("server_artifact_receipt")
        if (
            not isinstance(artifact_copy_receipt, Mapping)
            or set(artifact_copy_receipt) != ARTIFACT_COPY_RECEIPT_FIELDS
            or not isinstance(server_artifact_receipt, Mapping)
            or set(server_artifact_receipt) != set(live.br.ARTIFACT_FILENAMES)
            or any(
                not isinstance(receipt, Mapping)
                or set(receipt) != SERVER_ARTIFACT_RECEIPT_FIELDS
                for receipt in server_artifact_receipt.values()
            )
        ):
            _fail(f"{difficulty} artifact receipt schema changed")
        try:
            manifest, results_path, results_sha256 = live.full._verify_artifact_chain(
                suite_root=chosen_root,
                label_root=label_root,
                entry=entry,
                difficulty=difficulty,
            )
            target = live.full._verify_archived_target(
                suite_root=chosen_root,
                label_root=label_root,
                entry=entry,
                difficulty=difficulty,
                spec=spec,
            )
            live.full._verify_archived_config_identity(
                index,
                entry,
                difficulty=difficulty,
                spec=spec,
            )
        except live.full.FinalLiveError as exc:
            raise ScopedRunArchiveError(f"{difficulty} archive validation failed: {exc}") from exc
        manifest_path = chosen_root / str(entry["artifact_copy_manifest"]["path"])
        if (
            manifest_path.resolve() != (expected_run_dir / "artifact-copy-manifest.json").resolve()
            or results_path.resolve() != (expected_run_dir / "results.jsonl").resolve()
            or entry.get("results_path") != results_path.relative_to(chosen_root).as_posix()
            or entry.get("results_sha256") != results_sha256
            or target.get("case_count") != live.EXPECTED_CASE_COUNT
            or target.get("correct_count", -1) < live._minimum_correct(difficulty)
        ):
            _fail(f"{difficulty} six-artifact/results receipt changed")
        rows = live._read_results_jsonl(results_path, root=chosen_root)
        if len(rows) != live.EXPECTED_CASE_COUNT:
            _fail(f"{difficulty} results must contain exactly 50 physical rows")
        verified[difficulty] = {
            "entry": entry,
            "manifest": manifest,
            "results_path": results_path,
            "results_sha256": results_sha256,
            "rows": rows,
            "target_result": target,
            "run_id": run_id,
        }
    if all_case_ids != {
        case_id
        for difficulty in live.DIFFICULTIES
        for case_id in suite.datasets[difficulty].case_ids
    }:
        _fail("archive does not bind the exact scoped 100-case universe")
    return {
        "index_path": chosen_index,
        "index_sha256": live._sha256_file(
            chosen_index,
            root=chosen_root,
            label="scoped run index",
        ),
        "difficulties": verified,
    }
