#!/usr/bin/env python3
"""执行只含 easy/medium 的不可晋级 live 评测，并冻结完整可重放证据。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote


SCRIPT_PATH = Path(__file__).absolute()
FINAL_ROOT = SCRIPT_PATH.parent
FULL_ROOT = FINAL_ROOT.parent / "2026-08-11-three-difficulty-final"
SCOPE_CONTRACT_ROOT = (
    FINAL_ROOT.parent
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)
sys.path.insert(0, str(FULL_ROOT))
import run_final_live as full  # noqa: E402

sys.path.insert(0, str(SCOPE_CONTRACT_ROOT))
import activate_easy_medium_evaluation as activation_builder  # noqa: E402
import easy_medium_scope as scope  # noqa: E402
import validate_easy_medium_reviewed as reviewed_validator  # noqa: E402


PROJECT_ROOT = full.PROJECT_ROOT
SOURCE_RUNNER_PATH = full.SOURCE_RUNNER_PATH
br = full.br
DIFFICULTIES = tuple(scope.SCOPE_DIFFICULTIES)
EXPECTED_SUITE_KEY = "easy-medium-live-2026-08-13-reviewed-v2"
EXPECTED_DATASET_VERSION = "2.0.0-reviewed-v2.easy-medium.20260813"
EXPECTED_DATASET_KEYS = {
    difficulty: f"pantum-p3500-{difficulty}-live-reviewed-v2-easy-medium-2026-08-13"
    for difficulty in DIFFICULTIES
}
EXPECTED_CASE_COUNT = 50
EXPECTED_TOTAL_CASE_COUNT = 100
TARGET_CORRECT_COUNTS = dict(scope.SCOPE_QUALITY_TARGETS)
ACTIVATION_SCHEMA = "stt.eval.easy-medium-scope-activation.v1"
ACTIVE_DATASET_SCHEMA = "stt.eval.dataset.v2-llm-reviewed-candidate-active"
INDEX_SCHEMA = "stt.eval.easy-medium-scoped-live-run-index.v1"
RUN_ROLE = "easy_medium_scoped_live_run"
COMPLETION_SCHEMA = "stt.eval.easy-medium-scoped-live-completion.v1"
CASE_CONTRACT = full.CASE_CONTRACT
AUTHORITATIVE_METRIC = full.AUTHORITATIVE_METRIC
QUALITY_TARGETS_SCHEMA = full.QUALITY_TARGETS_SCHEMA
EXPECTED_CASE_PASS_POLICY = full.EXPECTED_CASE_PASS_POLICY
EXPECTED_KB_ID = full.EXPECTED_KB_ID
EXPECTED_DOCUMENT_ID = full.EXPECTED_DOCUMENT_ID
EXPECTED_DOCUMENT_VERSION_ID = full.EXPECTED_DOCUMENT_VERSION_ID
EXPECTED_INDEX_VERSION_ID = full.EXPECTED_INDEX_VERSION_ID
EXPECTED_CHUNK_COUNT = full.EXPECTED_CHUNK_COUNT
STATIC_CHECKSUM_FILENAME = full.STATIC_CHECKSUM_FILENAME
TERMINAL_RUN_STATUSES = full.TERMINAL_RUN_STATUSES
ACTIVE_RUN_STATUSES = full.ACTIVE_RUN_STATUSES
ACTIVE_DATASET_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "suite_key",
        "dataset_key",
        "name",
        "description",
        "version",
        "difficulty",
        "difficulty_label",
        "minimum_case_count",
        "evaluation_mode_default",
        "scope",
        "status",
        "import_allowed",
        "candidate_run_allowed",
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
        "source",
        "generation",
        "gate_policy",
        "tags",
        "activation_derivation",
        "cases",
    }
)
ACTIVATION_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "suite_key",
        "scope",
        "import_allowed",
        "candidate_run_allowed",
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
        "evaluation_contract",
        "quality_targets",
        "source_and_index_integrity",
        "datasets",
    }
)
ACTIVATION_EVALUATION_CONTRACT_FIELDS = frozenset(
    {
        "case_schema_version",
        "authoritative_metric",
        "target_counts",
        "case_pass_policy",
    }
)
ACTIVATION_DATASET_ENTRY_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "manifest_sha256",
        "case_count",
        "dataset_key",
        "version",
        "input",
        "derivation_receipt_sha256",
    }
)
ACTIVATION_DATASET_INPUT_FIELDS = frozenset(
    {"path", "file_sha256", "manifest_sha256"}
)
NONPROMOTION_BOOLEAN_FIELDS = (
    "canonical",
    "promotion_allowed",
    "promotable",
    "baseline_registration_allowed",
    "canonical_promotion_allowed",
)
FORBIDDEN_PROMOTION_REFERENCE_FIELDS = (
    "candidate_roots",
    "source_baseline_registry",
    "source_registry",
    "baselines",
)
DatasetSpec = full.DatasetSpec
SuiteSnapshot = full.SuiteSnapshot
FinalRedactor = full.FinalRedactor


class ScopedLiveError(RuntimeError):
    """表示可安全展示且不会泄露凭据的 scoped 工作流拒绝。"""


def _promotion_authority_drift(value: Any, *, path: str = "$") -> tuple[str, ...]:
    """递归返回与不可晋级授权冲突的显式字段路径。"""

    changed: list[str] = []
    if isinstance(value, Mapping):
        for field, nested in value.items():
            nested_path = f"{path}.{field}"
            if field in NONPROMOTION_BOOLEAN_FIELDS and nested is not False:
                changed.append(nested_path)
            if field in FORBIDDEN_PROMOTION_REFERENCE_FIELDS:
                changed.append(nested_path)
            changed.extend(_promotion_authority_drift(nested, path=nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            changed.extend(_promotion_authority_drift(nested, path=f"{path}[{index}]"))
    return tuple(sorted(changed))


def _minimum_correct(difficulty: str) -> int:
    value = TARGET_CORRECT_COUNTS[difficulty]
    if type(value) is not int:
        raise ScopedLiveError(f"共享 scope 的 {difficulty} target 必须是整数")
    return value


def _sha256_json(value: Any) -> str:
    return full._sha256_json(value)


def _sha256_file(path: Path, *, root: Path, label: str) -> str:
    try:
        return full._sha256_file(path, root=root, label=label)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _assert_safe_path(
    path: Path,
    *,
    root: Path,
    label: str,
    require_exists: bool = True,
) -> Path:
    try:
        return full._assert_safe_path(
            path,
            root=root,
            label=label,
            require_exists=require_exists,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    try:
        return full._read_json(path, root=root, label=label)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _current_reviewed_release_validation(reviewed_root: Path) -> dict[str, Any]:
    """重跑当前 scoped release 门禁，拒绝只靠可重哈希 input receipt 的激活。"""

    try:
        return reviewed_validator.validate(reviewed_root)
    except (OSError, RuntimeError, ValueError, reviewed_validator.ScopedValidationError) as exc:
        raise ScopedLiveError(f"current reviewed release validation failed: {exc}") from exc


def _write_json(path: Path, value: Any, *, root: Path = FINAL_ROOT) -> None:
    try:
        full._write_bytes_safe(
            path,
            (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            root=root,
            label="scoped runner JSON output",
            overwrite=True,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _write_static_checksums(root: Path = FINAL_ROOT) -> Path:
    try:
        return full._write_static_checksums(root)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _validate_static_checksums(root: Path) -> str:
    try:
        return full._validate_static_checksums(root)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _relative_scope(path: Path, *, root: Path = FINAL_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ScopedLiveError(f"path escapes easy/medium scope root: {path}") from exc


def _relative_project(path: Path) -> str:
    try:
        return full._relative_project(path)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _expected_quality_targets() -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        minimum = _minimum_correct(difficulty)
        targets[difficulty] = {
            "required": True,
            "operator": ">",
            "threshold": (minimum - 0.5) / EXPECTED_CASE_COUNT,
            "case_count": EXPECTED_CASE_COUNT,
            "minimum_correct_cases": minimum,
        }
    return {
        "schema_version": QUALITY_TARGETS_SCHEMA,
        "suite_key": EXPECTED_SUITE_KEY,
        "primary_metric": {
            "name": "final_visible_answer_binary_accuracy",
            "production_path": "metrics.case_outcome.final_answer_accuracy",
            "description": "最终呈现给用户的答案按题二元判定正确或错误，再按难度汇总正确率。",
        },
        "independent_blind_review": {
            "role": "audit_only",
            "counts_toward_target": False,
            "description": "独立盲审仅用于审计生产评测结论，不替代或混入生产正确率。",
        },
        "targets": targets,
    }


def _load_dataset(
    root: Path,
    difficulty: str,
    activation_entry: Mapping[str, Any],
    source_index_identity: Mapping[str, Any],
    release_validation: Mapping[str, Any],
    reviewed_root: Path,
) -> DatasetSpec:
    expected_relative = f"datasets/{difficulty}.json"
    if activation_entry.get("path") != expected_relative:
        raise ScopedLiveError(f"activation dataset path must be {expected_relative}")
    path = _assert_safe_path(root / expected_relative, root=root, label=f"{difficulty} dataset")
    if not path.is_file():
        raise ScopedLiveError(f"missing dataset: {expected_relative}")
    file_sha256 = _sha256_file(path, root=root, label=f"{difficulty} dataset")
    if activation_entry.get("sha256") != file_sha256:
        raise ScopedLiveError(f"activation SHA does not match {expected_relative}")
    raw = _read_json(path, root=root, label=f"{difficulty} dataset")
    if not isinstance(raw, Mapping):
        raise ScopedLiveError(f"{expected_relative} must contain an object")
    manifest, request_body = full._manifest_from_raw(raw, difficulty)
    authority_drift = _promotion_authority_drift(manifest)
    if authority_drift:
        raise ScopedLiveError(
            f"{difficulty} manifest authorization semantics drifted: "
            + ", ".join(authority_drift)
        )
    unexpected = set(manifest) - ACTIVE_DATASET_ALLOWED_TOP_LEVEL_FIELDS
    if unexpected:
        raise ScopedLiveError(
            f"{difficulty} manifest has forbidden top-level fields: "
            + ", ".join(sorted(unexpected))
        )
    for required in ("schema_version", "suite_key", "dataset_key", "name", "version", "source"):
        if not manifest.get(required):
            raise ScopedLiveError(f"{difficulty} manifest is missing required field {required}")
    expected_identity = {
        "schema_version": ACTIVE_DATASET_SCHEMA,
        "suite_key": EXPECTED_SUITE_KEY,
        "dataset_key": EXPECTED_DATASET_KEYS[difficulty],
        "version": EXPECTED_DATASET_VERSION,
        "difficulty": difficulty,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ScopedLiveError(
            f"{difficulty} manifest schema or identity differs from the frozen scoped contract"
        )
    if str(manifest.get("evaluation_mode_default") or "live") != "live":
        raise ScopedLiveError(f"{difficulty} evaluation_mode_default must be live")
    expected_state = {
        "status": "candidate_active",
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
        "scope": scope.scope_profile(),
    }
    if any(manifest.get(key) != value for key, value in expected_state.items()):
        raise ScopedLiveError(f"{difficulty} manifest is not internally candidate-active")
    source = manifest.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != reviewed_validator.REVIEWED_SOURCE_ALLOWED_FIELDS
    ):
        raise ScopedLiveError(f"{difficulty} manifest source schema drifted")
    generation = manifest.get("generation")
    gate_policy = manifest.get("gate_policy")
    if (
        not isinstance(generation, Mapping)
        or generation.get("independent_validation_status") != "passed"
        or not isinstance(gate_policy, Mapping)
        or gate_policy.get("case_pass_policy") != EXPECTED_CASE_PASS_POLICY
        or gate_policy.get("case_pass_policy_activation_status") != "active_candidate"
        or gate_policy.get("case_pass_policy_materialization_required") is not False
    ):
        raise ScopedLiveError(f"{difficulty} manifest policy/validation activation state drifted")
    source = manifest.get("source")
    expected_source = {
        "relative_path": source_index_identity["source_path"],
        "content_sha256": source_index_identity["source_sha256"],
        "page_count": source_index_identity["source_page_count"],
        "kb_id": source_index_identity["kb_id"],
        "document_id": source_index_identity["document_id"],
        "document_version_id": source_index_identity["document_version_id"],
        "index_version_id": source_index_identity["index_version_id"],
    }
    if not isinstance(source, Mapping) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise ScopedLiveError(f"{difficulty} manifest source/index identity drifted")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise ScopedLiveError(f"{difficulty} must contain exactly 50 scoped cases")
    case_ids: list[str] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ScopedLiveError(f"{difficulty} case {position} is not an object")
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ScopedLiveError(f"{difficulty} case {position} has no case_id")
        groups = row.get("concept_groups")
        all_of = groups.get("all_of") if isinstance(groups, Mapping) else None
        if not isinstance(all_of, list) or not all_of:
            raise ScopedLiveError(f"{difficulty} case {case_id} has no concept_groups.all_of")
        for group_index, group in enumerate(all_of, start=1):
            variants = group.get("any_of") if isinstance(group, Mapping) else None
            if (
                not isinstance(variants, list)
                or not variants
                or any(not isinstance(value, str) or not value.strip() for value in variants)
            ):
                raise ScopedLiveError(
                    f"{difficulty} case {case_id} concept_groups.all_of[{group_index}] is invalid"
                )
        answer_contract = row.get("answer_contract")
        review = answer_contract.get("review") if isinstance(answer_contract, Mapping) else None
        if (
            not isinstance(answer_contract, Mapping)
            or answer_contract.get("contract_status") != "llm_reviewed_candidate_active"
            or not isinstance(review, Mapping)
            or review.get("independent_validation_status") != "passed"
        ):
            raise ScopedLiveError(f"{difficulty} case {case_id} is not candidate-active")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ScopedLiveError(f"{difficulty} case_id values are not unique")
    manifest_sha256 = _sha256_json(manifest)
    if (
        activation_entry.get("case_count") != EXPECTED_CASE_COUNT
        or activation_entry.get("dataset_key") != EXPECTED_DATASET_KEYS[difficulty]
        or activation_entry.get("version") != EXPECTED_DATASET_VERSION
        or activation_entry.get("manifest_sha256") != manifest_sha256
    ):
        raise ScopedLiveError(f"{difficulty} activation manifest receipt drifted")
    try:
        full._validate_derivation(
            root,
            manifest,
            activation_entry,
            difficulty=difficulty,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    derivation = manifest.get("activation_derivation")
    if not isinstance(derivation, Mapping):  # pragma: no cover - 上游校验已覆盖
        raise ScopedLiveError(f"{difficulty} activation derivation receipt is required")
    try:
        input_path, input_root = full._resolve_frozen_input_path(
            root,
            derivation.get("input_path"),
            label=f"{difficulty} reviewed activation input",
        )
        reviewed_manifest = full._read_json(
            input_path,
            root=input_root,
            label=f"{difficulty} reviewed activation input",
        )
        if not isinstance(reviewed_manifest, dict):
            raise ScopedLiveError(f"{difficulty} reviewed activation input is invalid")
        if input_path.parent.parent.resolve() != reviewed_root.resolve():
            raise ScopedLiveError(
                f"{difficulty} activation input differs from current reviewed release root"
            )
        release_datasets = release_validation.get("datasets")
        current_release_receipt = (
            release_datasets.get(difficulty)
            if isinstance(release_datasets, Mapping)
            else None
        )
        expected_release_receipt = {
            "path": f"datasets/{difficulty}.reviewed.json",
            "sha256": derivation.get("input_file_sha256"),
            "manifest_sha256": derivation.get("input_manifest_sha256"),
            "dataset_key": EXPECTED_DATASET_KEYS[difficulty],
            "version": EXPECTED_DATASET_VERSION,
            "case_count": EXPECTED_CASE_COUNT,
        }
        if (
            release_validation.get("status") != "passed"
            or release_validation.get("suite_key") != EXPECTED_SUITE_KEY
            or current_release_receipt != expected_release_receipt
        ):
            raise ScopedLiveError(
                f"{difficulty} activation input differs from current reviewed release"
            )
        expected_active = activation_builder._derive_active_manifest(
            reviewed_manifest,
            input_relative_path=str(derivation["input_path"]),
            input_file_sha256=str(derivation["input_file_sha256"]),
        )
    except (full.FinalLiveError, activation_builder.ScopedActivationError) as exc:
        raise ScopedLiveError(f"{difficulty} activation derivation replay failed: {exc}") from exc
    if manifest != expected_active:
        raise ScopedLiveError(
            f"{difficulty} activation derivation output differs from exact replay"
        )
    return DatasetSpec(
        difficulty=difficulty,
        path=path,
        file_sha256=file_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        request_body=request_body,
        case_ids=tuple(case_ids),
    )


def load_suite_snapshot(root: Path = FINAL_ROOT) -> SuiteSnapshot:
    root = _assert_safe_path(root, root=root, label="easy/medium scope root")
    try:
        activation_builder._assert_no_promotion_surfaces(root)
    except activation_builder.ScopedActivationError as exc:
        raise ScopedLiveError(str(exc)) from exc
    activation_path = _assert_safe_path(root / "activation.json", root=root, label="activation.json")
    activation = _read_json(activation_path, root=root, label="activation.json")
    if not isinstance(activation, dict):
        raise ScopedLiveError("activation.json must contain an object")
    authority_drift = _promotion_authority_drift(activation)
    if authority_drift:
        raise ScopedLiveError(
            "activation authorization semantics drifted: "
            + ", ".join(authority_drift)
        )
    unexpected = set(activation) - ACTIVATION_ALLOWED_TOP_LEVEL_FIELDS
    if unexpected:
        raise ScopedLiveError(
            "activation has forbidden top-level fields: "
            + ", ".join(sorted(unexpected))
        )
    expected_top_level = {
        "schema_version": ACTIVATION_SCHEMA,
        "suite_key": EXPECTED_SUITE_KEY,
        "scope": scope.scope_profile(),
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
    }
    for key, expected in expected_top_level.items():
        if activation.get(key) != expected:
            if key == "scope":
                raise ScopedLiveError("activation scope profile does not match the frozen contract")
            raise ScopedLiveError(f"activation {key} must remain {expected!r}")
    contract = activation.get("evaluation_contract")
    expected_targets = {
        difficulty: {
            "correct_count": _minimum_correct(difficulty),
            "case_count": EXPECTED_CASE_COUNT,
        }
        for difficulty in DIFFICULTIES
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != ACTIVATION_EVALUATION_CONTRACT_FIELDS
        or contract.get("case_schema_version") != CASE_CONTRACT
        or contract.get("authoritative_metric") != AUTHORITATIVE_METRIC
        or contract.get("target_counts") != expected_targets
        or contract.get("case_pass_policy") != EXPECTED_CASE_PASS_POLICY
    ):
        raise ScopedLiveError("activation evaluation contract does not match the exact scoped contract")
    quality_receipt = activation.get("quality_targets")
    if (
        not isinstance(quality_receipt, Mapping)
        or set(quality_receipt) != {"path", "sha256", "schema_version"}
        or quality_receipt.get("path") != "quality-targets.json"
    ):
        raise ScopedLiveError("activation scoped quality-target receipt is required")
    quality_path = _assert_safe_path(
        root / "quality-targets.json",
        root=root,
        label="scoped quality-target contract",
    )
    quality_sha256 = _sha256_file(
        quality_path,
        root=root,
        label="scoped quality-target contract",
    )
    quality_targets = _read_json(
        quality_path,
        root=root,
        label="scoped quality-target contract",
    )
    if (
        quality_receipt.get("schema_version") != QUALITY_TARGETS_SCHEMA
        or quality_receipt.get("sha256") != quality_sha256
        or quality_targets != _expected_quality_targets()
    ):
        raise ScopedLiveError("scoped quality-target schema, content, or SHA drifted")
    try:
        source_index_identity = full._load_source_index_identity(root, activation)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    entries = activation.get("datasets")
    if not isinstance(entries, Mapping) or set(entries) != set(DIFFICULTIES):
        raise ScopedLiveError("activation datasets must contain exactly easy and medium")
    for difficulty in DIFFICULTIES:
        entry = entries.get(difficulty)
        input_receipt = entry.get("input") if isinstance(entry, Mapping) else None
        if (
            not isinstance(entry, Mapping)
            or set(entry) != ACTIVATION_DATASET_ENTRY_FIELDS
            or not isinstance(input_receipt, Mapping)
            or set(input_receipt) != ACTIVATION_DATASET_INPUT_FIELDS
            or _promotion_authority_drift(entry)
        ):
            raise ScopedLiveError(f"activation dataset entry is invalid: {difficulty}")
    try:
        reviewed_input_path, _ = full._resolve_frozen_input_path(
            root,
            entries[DIFFICULTIES[0]]["input"]["path"],
            label="current reviewed release input",
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    reviewed_root = reviewed_input_path.parent.parent
    release_validation = _current_reviewed_release_validation(reviewed_root)
    reviewed_suite = _read_json(
        reviewed_root / reviewed_validator.SUITE_FILENAME,
        root=reviewed_root,
        label="current reviewed release suite",
    )
    reviewed_integrity = (
        reviewed_suite.get("source_and_index_integrity")
        if isinstance(reviewed_suite, Mapping)
        else None
    )
    current_review = (
        reviewed_integrity.get("review")
        if isinstance(reviewed_integrity, Mapping)
        else None
    )
    try:
        expected_review = {
            **dict(current_review),
            "scoped_validation_schema": release_validation["schema_version"],
            "scoped_validation_sha256": _sha256_json(release_validation),
            "scoped_reviewed_suite_path": _relative_project(
                reviewed_root / release_validation["suite"]["path"]
            ),
            "scoped_reviewed_suite_sha256": release_validation["suite"]["sha256"],
            "scoped_reviewed_summary_path": _relative_project(
                reviewed_root / release_validation["summary"]["path"]
            ),
            "scoped_reviewed_summary_sha256": release_validation["summary"]["sha256"],
            "scoped_reviewed_qa_path": _relative_project(
                reviewed_root / release_validation["qa"]["path"]
            ),
            "scoped_reviewed_qa_sha256": release_validation["qa"]["sha256"],
            "scoped_reviewed_checksums_path": _relative_project(
                reviewed_root / release_validation["checksums"]["path"]
            ),
            "scoped_reviewed_checksums_sha256": release_validation["checksums"][
                "sha256"
            ],
            "authoritative_scope_receipts_sha256": release_validation[
                "authoritative_scope_receipts_sha256"
            ],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ScopedLiveError("current reviewed release receipt is incomplete") from exc
    integrity = activation.get("source_and_index_integrity")
    source_integrity = integrity.get("source") if isinstance(integrity, Mapping) else None
    active_index_integrity = (
        integrity.get("active_index") if isinstance(integrity, Mapping) else None
    )
    review_integrity = integrity.get("review") if isinstance(integrity, Mapping) else None
    expected_review_fields = (
        reviewed_validator.REVIEW_INTEGRITY_REQUIRED_FIELDS
        | reviewed_validator.REVIEW_INTEGRITY_ACTIVATION_FIELDS
    )
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != reviewed_validator.SOURCE_AND_INDEX_INTEGRITY_FIELDS
        or not isinstance(source_integrity, Mapping)
        or set(source_integrity) != reviewed_validator.SOURCE_INTEGRITY_FIELDS
        or not isinstance(active_index_integrity, Mapping)
        or set(active_index_integrity)
        != reviewed_validator.ACTIVE_INDEX_INTEGRITY_FIELDS
        or not isinstance(review_integrity, Mapping)
        or set(review_integrity) != expected_review_fields
        or dict(review_integrity) != expected_review
        or _promotion_authority_drift(integrity)
        or _promotion_authority_drift(source_integrity)
        or _promotion_authority_drift(active_index_integrity)
        or _promotion_authority_drift(review_integrity)
    ):
        raise ScopedLiveError("activation source_and_index_integrity schema drifted")
    datasets: dict[str, DatasetSpec] = {}
    for difficulty in DIFFICULTIES:
        entry = entries.get(difficulty)
        datasets[difficulty] = _load_dataset(
            root,
            difficulty,
            entry,
            source_index_identity,
            release_validation,
            reviewed_root,
        )
    all_case_ids = [
        case_id for difficulty in DIFFICULTIES for case_id in datasets[difficulty].case_ids
    ]
    if len(all_case_ids) != EXPECTED_TOTAL_CASE_COUNT or len(set(all_case_ids)) != (
        EXPECTED_TOTAL_CASE_COUNT
    ):
        raise ScopedLiveError("case_id values must be globally unique across exactly 100 scoped cases")
    static_checksums_sha256 = _validate_static_checksums(root)
    activation_sha256 = _sha256_file(
        activation_path,
        root=root,
        label="activation.json",
    )
    identity = {
        "activation_sha256": activation_sha256,
        "suite_key": EXPECTED_SUITE_KEY,
        "scope": scope.scope_profile(),
        "dataset_files": {
            difficulty: datasets[difficulty].file_sha256 for difficulty in DIFFICULTIES
        },
        "dataset_keys": EXPECTED_DATASET_KEYS,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "case_contract": CASE_CONTRACT,
        "authoritative_metric": AUTHORITATIVE_METRIC,
        "quality_targets_sha256": quality_sha256,
        "source_index_identity": source_index_identity,
        "static_checksums_sha256": static_checksums_sha256,
    }
    return SuiteSnapshot(
        activation=activation,
        activation_path=activation_path,
        activation_sha256=activation_sha256,
        suite_key=EXPECTED_SUITE_KEY,
        suite_identity_sha256=_sha256_json(identity),
        datasets=datasets,
        quality_targets_path=quality_path,
        quality_targets_sha256=quality_sha256,
        source_index_identity=source_index_identity,
        static_checksums_sha256=static_checksums_sha256,
    )


class LabelLock:
    """持有随进程释放的单 label OS advisory lock。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.handle: Any | None = None
        self.path = FINAL_ROOT / "runs" / ".locks" / f"{label}.lock"

    def __enter__(self) -> LabelLock:
        runs_root = FINAL_ROOT / "runs"
        _assert_safe_path(
            runs_root,
            root=FINAL_ROOT,
            label="runs root",
            require_exists=False,
        )
        runs_root.mkdir(parents=True, exist_ok=True)
        lock_root = self.path.parent
        _assert_safe_path(
            lock_root,
            root=FINAL_ROOT,
            label="runner lock root",
            require_exists=False,
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        _assert_safe_path(
            self.path,
            root=FINAL_ROOT,
            label="runner lock file",
            require_exists=False,
        )
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - 当前工作区以 Windows 为目标
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise ScopedLiveError(f"another runner owns label {self.label!r}") from exc
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - 当前工作区以 Windows 为目标
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _new_index(
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    code_revision: str,
    *,
    root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    now = br._utc_now()
    return {
        "schema_version": INDEX_SCHEMA,
        "label": args.label,
        "role": RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "suite_key": suite.suite_key,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "scope": scope.scope_profile(),
        "mode": "live",
        "baseline_run_id": None,
        "selection": "all_scoped_cases",
        "planned_difficulties": list(DIFFICULTIES),
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "code_revision": code_revision,
        "runtime_identity": dict(args.runtime_identity),
        "runner": {
            "path": _relative_project(SCRIPT_PATH),
            "sha256": _sha256_file(SCRIPT_PATH, root=PROJECT_ROOT, label="scoped live runner"),
            "shared_validation_path": _relative_project(full.SCRIPT_PATH),
            "shared_validation_sha256": _sha256_file(
                full.SCRIPT_PATH,
                root=PROJECT_ROOT,
                label="full live validation runtime",
            ),
            "shared_client_path": _relative_project(SOURCE_RUNNER_PATH),
            "shared_client_sha256": _sha256_file(
                SOURCE_RUNNER_PATH,
                root=PROJECT_ROOT,
                label="shared live runner",
            ),
        },
        "activation": {
            "path": _relative_scope(suite.activation_path, root=root),
            "sha256": suite.activation_sha256,
            "candidate_run_allowed": True,
            "baseline_registration_allowed": False,
            "canonical_promotion_allowed": False,
            "promotable": False,
        },
        "quality_targets": {
            "path": _relative_scope(suite.quality_targets_path, root=root),
            "sha256": suite.quality_targets_sha256,
            "schema_version": QUALITY_TARGETS_SCHEMA,
        },
        "source_index_identity": dict(suite.source_index_identity),
        "static_checksums_sha256": suite.static_checksums_sha256,
        "dataset_files": {
            difficulty: {
                "path": _relative_scope(suite.datasets[difficulty].path, root=root),
                "sha256": suite.datasets[difficulty].file_sha256,
                "manifest_sha256": suite.datasets[difficulty].manifest_sha256,
                "case_count": suite.datasets[difficulty].case_count,
                "dataset_key": suite.datasets[difficulty].manifest["dataset_key"],
                "version": suite.datasets[difficulty].manifest["version"],
            }
            for difficulty in DIFFICULTIES
        },
        "target_contract": {
            "authoritative_metric": AUTHORITATIVE_METRIC,
            "strict_metrics_count_toward_target": False,
            "blind_audit_counts_toward_target": False,
            "correct_counts": TARGET_CORRECT_COUNTS,
            "case_count": EXPECTED_CASE_COUNT,
        },
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "invocations": [],
        "difficulties": {},
    }


def _persist_index(index_path: Path, index: dict[str, Any], *, root: Path = FINAL_ROOT) -> None:
    index["updated_at"] = br._utc_now()
    _write_json(index_path, index, root=root)


def _validate_frozen_index(
    index: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    code_revision: str,
    root: Path = FINAL_ROOT,
) -> None:
    expected = {
        "schema_version": INDEX_SCHEMA,
        "label": args.label,
        "role": RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "suite_key": suite.suite_key,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "scope": scope.scope_profile(),
        "mode": "live",
        "baseline_run_id": None,
        "selection": "all_scoped_cases",
        "planned_difficulties": list(DIFFICULTIES),
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "code_revision": code_revision,
        "runtime_identity": dict(args.runtime_identity),
    }
    changed = [key for key, value in expected.items() if index.get(key) != value]
    changed.extend(_promotion_authority_drift(index))
    runner = index.get("runner")
    runner_hashes = {
        "sha256": _sha256_file(SCRIPT_PATH, root=PROJECT_ROOT, label="scoped live runner"),
        "shared_validation_sha256": _sha256_file(
            full.SCRIPT_PATH,
            root=PROJECT_ROOT,
            label="full live validation runtime",
        ),
        "shared_client_sha256": _sha256_file(
            SOURCE_RUNNER_PATH,
            root=PROJECT_ROOT,
            label="shared live runner",
        ),
    }
    if not isinstance(runner, Mapping) or any(
        runner.get(key) != value for key, value in runner_hashes.items()
    ):
        changed.append("runner")
    activation = index.get("activation")
    expected_activation = {
        "path": _relative_scope(suite.activation_path, root=root),
        "sha256": suite.activation_sha256,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
    }
    if activation != expected_activation:
        changed.append("activation")
    quality = index.get("quality_targets")
    expected_quality = {
        "path": _relative_scope(suite.quality_targets_path, root=root),
        "sha256": suite.quality_targets_sha256,
        "schema_version": QUALITY_TARGETS_SCHEMA,
    }
    if quality != expected_quality:
        changed.append("quality_targets")
    if index.get("source_index_identity") != suite.source_index_identity:
        changed.append("source_index_identity")
    if index.get("static_checksums_sha256") != suite.static_checksums_sha256:
        changed.append("static_checksums_sha256")
    files = index.get("dataset_files")
    if not isinstance(files, Mapping) or set(files) != set(DIFFICULTIES):
        changed.append("dataset_files")
    else:
        for difficulty in DIFFICULTIES:
            item = files[difficulty]
            if (
                not isinstance(item, Mapping)
                or item.get("sha256") != suite.datasets[difficulty].file_sha256
                or item.get("manifest_sha256") != suite.datasets[difficulty].manifest_sha256
            ):
                changed.append(f"dataset_files.{difficulty}")
    if changed:
        raise ScopedLiveError(
            "refusing to continue because frozen scoped label identity changed: "
            + ", ".join(sorted(set(changed)))
        )


def _prepare_label(
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    code_revision: str,
    *,
    root: Path = FINAL_ROOT,
) -> tuple[Path, Path, dict[str, Any]]:
    runs_root = root / "runs"
    _assert_safe_path(
        runs_root,
        root=root,
        label="runs root",
        require_exists=False,
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    label_root = runs_root / args.label
    _assert_safe_path(
        label_root,
        root=runs_root,
        label="scoped label root",
        require_exists=False,
    )
    index_path = label_root / "run-index.json"
    if args.resume:
        if not label_root.is_dir() or not index_path.is_file():
            raise ScopedLiveError("--resume requires runs/<label>/run-index.json")
        try:
            full._tree_files(label_root, trusted_root=root, label="scoped label")
        except full.FinalLiveError as exc:
            raise ScopedLiveError(str(exc)) from exc
        index = _read_json(index_path, root=root, label="scoped run index")
        if not isinstance(index, dict):
            raise ScopedLiveError("run-index.json must contain an object")
        _validate_frozen_index(
            index,
            args=args,
            suite=suite,
            code_revision=code_revision,
            root=root,
        )
        index["status"] = "in_progress"
        return label_root, index_path, index
    if label_root.exists():
        raise ScopedLiveError("label already exists; use --resume or choose a new label")
    if args.difficulty != "easy":
        raise ScopedLiveError("a new scoped label must start with easy")
    label_root.mkdir(parents=False, exist_ok=False)
    index = _new_index(args, suite, code_revision, root=root)
    _persist_index(index_path, index, root=root)
    return label_root, index_path, index


def _target_count_from_rows(
    rows: Sequence[Any],
    *,
    expected_run_id: str,
    expected_case_ids: Sequence[str],
) -> dict[str, Any]:
    try:
        return full._target_count_from_rows(
            rows,
            expected_run_id=expected_run_id,
            expected_case_ids=expected_case_ids,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _read_results_jsonl(path: Path, *, root: Path = FINAL_ROOT) -> list[Any]:
    try:
        return full._read_results_jsonl(path, root=root)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _entry_target_passed(entry: Mapping[str, Any], difficulty: str) -> bool:
    target = entry.get("target_result")
    return (
        isinstance(target, Mapping)
        and target.get("case_count") == EXPECTED_CASE_COUNT
        and type(target.get("correct_count")) is int
        and target["correct_count"] >= _minimum_correct(difficulty)
    )


def _assert_progression(
    index: Mapping[str, Any],
    difficulty: str,
    *,
    suite: SuiteSnapshot,
    label_root: Path,
    root: Path = FINAL_ROOT,
) -> None:
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping):
        raise ScopedLiveError("run index has no difficulty map")
    position = DIFFICULTIES.index(difficulty)
    for preceding in DIFFICULTIES[:position]:
        entry = entries.get(preceding)
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise ScopedLiveError(f"{difficulty} requires archived {preceding} first")
        try:
            target = full._verify_archived_target(
                suite_root=root,
                label_root=label_root,
                entry=entry,
                difficulty=preceding,
                spec=suite.datasets[preceding],
            )
            full._verify_archived_config_identity(
                index,
                entry,
                difficulty=preceding,
                spec=suite.datasets[preceding],
            )
        except full.FinalLiveError as exc:
            raise ScopedLiveError(str(exc)) from exc
        required = _minimum_correct(preceding)
        if target["correct_count"] < required:
            raise ScopedLiveError(
                f"{difficulty} is blocked: {preceding} did not reach {required}/50"
            )
    for later in DIFFICULTIES[position + 1 :]:
        if later in entries:
            raise ScopedLiveError("scoped run-index difficulty order is invalid")
    if any(key not in DIFFICULTIES for key in entries):
        raise ScopedLiveError("scoped run-index contains a forbidden difficulty")


def _poll(
    client: Any,
    *,
    run_id: str,
    difficulty: str,
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
        if not isinstance(run, dict) or run.get("id") != run_id:
            raise ScopedLiveError("poll returned the wrong run identity")
        status = str(run.get("status") or "")
        completed = int(run.get("completed_cases") or 0)
        total = int(run.get("total_cases") or 0)
        display = (status, completed, total)
        if display != last:
            print(f"[{difficulty}] {status} ({completed}/{total})", flush=True)
            index["difficulties"][difficulty].update(
                {
                    "status": status,
                    "completed_cases": completed,
                    "total_cases": total,
                    "last_polled_at": br._utc_now(),
                }
            )
            _persist_index(index_path, index)
            last = display
        if status in TERMINAL_RUN_STATUSES:
            return envelope, run
        if status not in ACTIVE_RUN_STATUSES:
            raise ScopedLiveError(f"unsupported run status: {status!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ScopedLiveError("poll timeout; rerun the same difficulty with --resume")
        time.sleep(min(args.poll_interval, remaining))


def _load_existing_copy_manifest(
    manifest_path: Path,
    *,
    label_root: Path,
    difficulty: str,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value, _, _ = full._verify_artifact_chain(
            suite_root=FINAL_ROOT,
            label_root=label_root,
            entry={
                "run_id": run.get("id"),
                "artifact_copy_manifest": {
                    "path": _relative_scope(manifest_path),
                    "sha256": _sha256_file(
                        manifest_path,
                        root=FINAL_ROOT,
                        label="artifact-copy manifest",
                    ),
                },
            },
            difficulty=difficulty,
            run=run,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    return value


def _write_api_copy(
    run_dir: Path,
    invocation_id: str,
    filename: str,
    value: Any,
    redactor: Any,
) -> None:
    short_id = invocation_id.rsplit("-", maxsplit=1)[-1]
    _write_json(run_dir / "api" / short_id / filename, redactor.clean(value))


def _process_difficulty(
    *,
    difficulty: str,
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    label_root: Path,
    index_path: Path,
    index: dict[str, Any],
    client: Any,
    redactor: Any,
    invocation_id: str,
) -> None:
    _assert_progression(index, difficulty, suite=suite, label_root=label_root)
    spec = suite.datasets[difficulty]
    existing = index["difficulties"].get(difficulty)
    if isinstance(existing, Mapping) and existing.get("status") == "archived":
        if not args.resume:
            raise ScopedLiveError("archived difficulty requires --resume for local verification")
        try:
            full._verify_archived_target(
                suite_root=FINAL_ROOT,
                label_root=label_root,
                entry=existing,
                difficulty=difficulty,
                spec=spec,
            )
            full._verify_archived_config_identity(
                index,
                existing,
                difficulty=difficulty,
                spec=spec,
            )
        except full.FinalLiveError as exc:
            raise ScopedLiveError(str(exc)) from exc
        if not _entry_target_passed(existing, difficulty):
            required = _minimum_correct(difficulty)
            raise ScopedLiveError(
                f"{difficulty} remains below the required {required}/50 target; start a new label"
            )
        print(f"[{difficulty}] verified existing immutable scoped archive", flush=True)
        return

    entry = dict(existing) if isinstance(existing, Mapping) else {}
    entry.update(
        {
            "difficulty": difficulty,
            "role": RUN_ROLE,
            "mode": "live",
            "baseline_run_id": None,
            "dataset_source_path": _relative_project(spec.path),
            "dataset_source_sha256": spec.file_sha256,
            "selected_case_count": spec.case_count,
            "selected_case_ids": list(spec.case_ids),
            "status": entry.get("status") or "dataset_import_pending",
        }
    )
    index["difficulties"][difficulty] = entry
    _persist_index(index_path, index)
    prior_dataset_id = str(entry.get("dataset_id") or "")
    prior_dataset_checksum = str(entry.get("dataset_checksum") or "")
    if prior_dataset_id:
        dataset_envelope = client.request_json(
            "GET",
            f"/api/v1/evaluation/datasets/{quote(prior_dataset_id, safe='')}",
        )
    else:
        dataset_envelope = client.request_json(
            "POST",
            "/api/v1/evaluation/datasets",
            body=spec.request_body,
        )
    dataset = br._envelope_data(dataset_envelope, operation=f"dataset {difficulty}")
    if not isinstance(dataset, Mapping):
        raise ScopedLiveError("dataset API returned non-object data")
    try:
        dataset_id = full._assert_dataset_api(
            dataset,
            spec,
            prior_dataset_id,
            prior_dataset_checksum,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    dataset_checksum = str(dataset["checksum"])
    entry.update(
        {
            "dataset_id": dataset_id,
            "dataset_key": dataset.get("dataset_key"),
            "dataset_version": dataset.get("version"),
            "dataset_checksum": dataset_checksum,
            "status": "preflight_pending",
        }
    )
    _persist_index(index_path, index)
    preflight_body = {
        "dataset_id": dataset_id,
        "kb_id": args.kb_id,
        "mode": "live",
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
    }
    preflight_envelope = client.request_json(
        "POST",
        "/api/v1/evaluation/preflight",
        body=preflight_body,
    )
    preflight = br._envelope_data(preflight_envelope, operation=f"preflight {difficulty}")
    try:
        profile_identity = full._validate_preflight(
            preflight,
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            spec=spec,
            args=args,
            suite=suite,
        )
        full._freeze_or_validate_profile(index, profile_identity)
        full._freeze_or_validate_preflight(index, preflight)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    _persist_index(index_path, index)

    run_id = str(entry.get("run_id") or "") if args.resume else ""
    create_envelope: dict[str, Any] | None = None
    if run_id:
        run_id = br._safe_component(run_id, label="run id")
        initial_envelope = client.request_json(
            "GET",
            f"/api/v1/evaluation/runs/{quote(run_id, safe='')}",
        )
    else:
        body = {
            "dataset_id": dataset_id,
            "kb_id": args.kb_id,
            "mode": "live",
            "name": f"easy-medium-scope-{args.label}-{difficulty}-non-promotable",
            "baseline_run_id": None,
            "retrieval_profile_id": args.retrieval_profile_id,
            "retrieval_profile_version": args.retrieval_profile_version,
        }
        create_envelope = client.request_json("POST", "/api/v1/evaluation/runs", body=body)
        initial_envelope = create_envelope
    initial_run = br._envelope_data(initial_envelope, operation=f"run {difficulty}")
    if not isinstance(initial_run, Mapping):
        raise ScopedLiveError("run API returned non-object data")
    if not run_id:
        run_id = br._safe_component(str(initial_run.get("id") or ""), label="run id")
        entry.update(
            {
                "run_id": run_id,
                "status": str(initial_run.get("status") or "queued"),
                "created_at": initial_run.get("created_at"),
            }
        )
        _persist_index(index_path, index)
    snapshot = initial_run.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ScopedLiveError("run has no config_snapshot")
    try:
        full._freeze_or_validate_execution_config(index, snapshot)
        full._assert_run_identity(
            initial_run,
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            args=args,
            index=index,
            case_ids=spec.case_ids,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    _persist_index(index_path, index)

    run_dir = label_root / "artifacts" / difficulty / run_id
    _assert_safe_path(
        run_dir,
        root=FINAL_ROOT,
        label=f"{difficulty} run directory",
        require_exists=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_api_copy(run_dir, invocation_id, "dataset.json", dataset_envelope, redactor)
    _write_api_copy(run_dir, invocation_id, "preflight.json", preflight_envelope, redactor)
    _write_api_copy(
        run_dir,
        invocation_id,
        "run-create.json" if create_envelope is not None else "run-resume.json",
        initial_envelope,
        redactor,
    )
    final_envelope, final_run = _poll(
        client,
        run_id=run_id,
        difficulty=difficulty,
        args=args,
        index_path=index_path,
        index=index,
    )
    _write_api_copy(run_dir, invocation_id, "run-final.json", final_envelope, redactor)
    if final_run.get("status") != "succeeded":
        raise ScopedLiveError(
            f"{difficulty} run ended as {final_run.get('status')} "
            f"({final_run.get('error_code') or 'unknown_error'})"
        )
    try:
        full._assert_run_identity(
            final_run,
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            args=args,
            index=index,
            case_ids=spec.case_ids,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc

    encoded = quote(run_id, safe="")
    results_envelope = client.request_json("GET", f"/api/v1/evaluation/runs/{encoded}/results")
    comparison_envelope = client.request_json(
        "GET",
        f"/api/v1/evaluation/runs/{encoded}/comparison",
    )
    _write_api_copy(run_dir, invocation_id, "results.json", results_envelope, redactor)
    _write_api_copy(run_dir, invocation_id, "comparison.json", comparison_envelope, redactor)
    api_rows = br._envelope_data(results_envelope, operation=f"results {difficulty}")
    if not isinstance(api_rows, list):
        raise ScopedLiveError("results API returned non-list data")
    api_target = _target_count_from_rows(
        api_rows,
        expected_run_id=run_id,
        expected_case_ids=spec.case_ids,
    )
    manifest_path = run_dir / "artifact-copy-manifest.json"
    if manifest_path.is_file():
        _load_existing_copy_manifest(
            manifest_path,
            label_root=label_root,
            difficulty=difficulty,
            run=final_run,
        )
    else:
        manifest_path, _ = br._archive_artifacts(
            client,
            difficulty=difficulty,
            run=dict(final_run),
            allow_repair=bool(args.resume),
        )
    try:
        _, results_path, results_sha256 = full._verify_artifact_chain(
            suite_root=FINAL_ROOT,
            label_root=label_root,
            entry={
                "run_id": run_id,
                "artifact_copy_manifest": {
                    "path": _relative_scope(manifest_path),
                    "sha256": _sha256_file(
                        manifest_path,
                        root=FINAL_ROOT,
                        label="artifact-copy manifest",
                    ),
                },
            },
            difficulty=difficulty,
            run=final_run,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    archived_target = _target_count_from_rows(
        _read_results_jsonl(results_path),
        expected_run_id=run_id,
        expected_case_ids=spec.case_ids,
    )
    if archived_target != api_target:
        raise ScopedLiveError("archived target result differs from the results API")
    required = _minimum_correct(difficulty)
    target_passed = archived_target["correct_count"] >= required
    try:
        server_artifact_receipt = full._server_artifact_receipt(final_run)
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    entry.update(
        redactor.clean(
            {
                "status": "archived",
                "completed_at": final_run.get("completed_at"),
                "completed_cases": final_run.get("completed_cases"),
                "total_cases": final_run.get("total_cases"),
                "run_directory": _relative_scope(run_dir),
                "artifact_copy_manifest": {
                    "path": _relative_scope(manifest_path),
                    "sha256": _sha256_file(
                        manifest_path,
                        root=FINAL_ROOT,
                        label="artifact-copy manifest",
                    ),
                },
                "server_artifact_receipt": server_artifact_receipt,
                "results_path": _relative_scope(results_path),
                "results_sha256": results_sha256,
                "target_result": archived_target,
                "target_required_correct_count": required,
                "target_passed": target_passed,
                "gate_result_diagnostic_only": final_run.get("gate_result") or {},
                "metrics_summary_diagnostic_only": final_run.get("metrics_summary") or {},
                "comparison_summary_diagnostic_only": final_run.get("comparison_summary") or {},
                "config_snapshot": final_run.get("config_snapshot") or {},
            }
        )
    )
    _persist_index(index_path, index)
    print(
        f"[{difficulty}] non-promotable scoped archive: "
        f"{archived_target['correct_count']}/{archived_target['case_count']}",
        flush=True,
    )
    if not target_passed:
        raise ScopedLiveError(f"{difficulty} did not reach the required {required}/50 target")


def _finalize_scope_index(
    label_root: Path,
    index_path: Path,
    index: dict[str, Any],
    *,
    root: Path | None = None,
) -> None:
    chosen_root = root or label_root.parent.parent
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping) or set(entries) != set(DIFFICULTIES):
        return
    if any(
        not isinstance(entries[difficulty], Mapping)
        or entries[difficulty].get("status") != "archived"
        or not _entry_target_passed(entries[difficulty], difficulty)
        for difficulty in DIFFICULTIES
    ):
        return
    forbidden_path = label_root / "candidate-roots.json"
    if forbidden_path.exists():
        raise ScopedLiveError("scoped run refuses a candidate-roots.json artifact")
    if {"candidate_roots", "source_baseline_registry", "baselines"}.intersection(index):
        raise ScopedLiveError("scoped run index contains forbidden promotion fields")
    index["scope_completion"] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "succeeded",
        "role": RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "scope": scope.scope_profile(),
        "difficulty_order": list(DIFFICULTIES),
        "completed_at": br._utc_now(),
    }
    index["status"] = "succeeded"
    index["promotion_allowed"] = False
    _persist_index(index_path, index, root=chosen_root)


def _write_checksums(label_root: Path) -> None:
    destination = label_root / "SHA256SUMS"
    try:
        files = [
            path
            for path in full._tree_files(
                label_root,
                trusted_root=FINAL_ROOT,
                label="scoped label",
            )
            if path != destination
        ]
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc
    content = "".join(
        f"{_sha256_file(path, root=FINAL_ROOT, label='scoped artifact')}  "
        f"{path.relative_to(label_root).as_posix()}\n"
        for path in files
    ).encode("utf-8")
    try:
        full._write_bytes_safe(
            destination,
            content,
            root=FINAL_ROOT,
            label="scoped SHA256SUMS",
            overwrite=True,
        )
    except full.FinalLiveError as exc:
        raise ScopedLiveError(str(exc)) from exc


def _parse_args(config: Mapping[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, type=full._safe_label)
    parser.add_argument("--difficulty", required=True, choices=DIFFICULTIES)
    parser.add_argument("--retrieval-profile-id", required=True)
    parser.add_argument(
        "--retrieval-profile-version",
        required=True,
        type=full._positive_int,
    )
    parser.add_argument(
        "--base-url",
        default=br._first_config(
            dict(config),
            "STT_EVAL_BASE_URL",
            "STT_QUERY_BASE_URL",
            default=br.DEFAULT_BASE_URL,
        ),
    )
    parser.add_argument(
        "--kb-id",
        default=br._first_config(dict(config), "STT_EVAL_KB_ID", "EVALUATION_KB_ID"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--poll-timeout",
        type=br._positive_float,
        default=float(br.DEFAULT_POLL_TIMEOUT_SECONDS),
    )
    parser.add_argument(
        "--poll-interval",
        type=br._positive_float,
        default=float(br.DEFAULT_POLL_INTERVAL_SECONDS),
    )
    parser.add_argument(
        "--request-timeout",
        type=br._positive_float,
        default=float(br.DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )
    args = parser.parse_args()
    args.base_url = br._normalize_local_base_url(args.base_url)
    args.kb_id = br._safe_component(str(args.kb_id or ""), label="knowledge-base id")
    args.retrieval_profile_id = br._safe_component(
        str(args.retrieval_profile_id or ""),
        label="retrieval profile id",
    )
    return args


def _validate_cli_suite_identity(args: argparse.Namespace, suite: SuiteSnapshot) -> None:
    if args.kb_id != suite.source_index_identity["kb_id"]:
        raise ScopedLiveError("--kb-id does not match the KB frozen in both scoped manifests")


def _run(
    args: argparse.Namespace,
    config: dict[str, str],
    suite: SuiteSnapshot,
    code_revision: str,
) -> int:
    label_root, index_path, index = _prepare_label(args, suite, code_revision)
    _assert_progression(index, args.difficulty, suite=suite, label_root=label_root)
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
        raise ScopedLiveError("bootstrap admin password could not be resolved")
    br.SUITE_ROOT = FINAL_ROOT
    br.RUNS_DIR = label_root / "artifacts"
    br.API_SNAPSHOT_ROOT = label_root / "api-snapshots"
    redactor = FinalRedactor(password)
    invocation_id = f"{br._timestamp_slug()}-{uuid.uuid4().hex[:8]}"
    invocation = {
        "id": invocation_id,
        "difficulty": args.difficulty,
        "started_at": br._utc_now(),
        "status": "running",
        "resume": bool(args.resume),
    }
    index["invocations"].append(invocation)
    _persist_index(index_path, index)
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
        print("Authenticated to the loopback query API; credentials are redacted.", flush=True)
        _process_difficulty(
            difficulty=args.difficulty,
            args=args,
            suite=suite,
            label_root=label_root,
            index_path=index_path,
            index=index,
            client=client,
            redactor=redactor,
            invocation_id=invocation_id,
        )
        _finalize_scope_index(label_root, index_path, index)
        if index.get("status") != "succeeded":
            index["status"] = "in_progress"
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
            _persist_index(index_path, redactor.clean(index))
            _write_checksums(label_root)
        except Exception as exc:
            finalization_error = exc
    if error is not None:
        raise ScopedLiveError(f"{type(error).__name__}: {redactor.text(str(error))}") from error
    if finalization_error is not None:
        raise ScopedLiveError(
            f"finalization {type(finalization_error).__name__}: "
            f"{redactor.text(str(finalization_error))}"
        ) from finalization_error
    print(f"Scoped run index: {_relative_project(index_path)}", flush=True)
    return 0


def main() -> int:
    config = br._load_environment()
    args = _parse_args(config)
    suite = load_suite_snapshot()
    _validate_cli_suite_identity(args, suite)
    code_revision = full._current_code_revision()
    args.runtime_identity = full._local_runtime_identity()
    with LabelLock(args.label):
        return _run(args, config, suite, code_revision)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ScopedLiveError, full.FinalLiveError, br.BaselineRunnerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

