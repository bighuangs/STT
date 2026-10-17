#!/usr/bin/env python3
"""将已验证的 easy/medium reviewed 产物激活到独立候选评测根。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from activate_reviewed_suite import (
    EXPECTED_DOCUMENT_ID,
    EXPECTED_DOCUMENT_VERSION_ID,
    EXPECTED_INDEX_VERSION_ID,
    EXPECTED_KB_ID,
)
from easy_medium_scope import (
    SCOPE_BASELINE_REGISTRATION_ALLOWED,
    SCOPE_CANDIDATE_RUN_ALLOWED,
    SCOPE_CANONICAL_PROMOTION_ALLOWED,
    SCOPE_DIFFICULTIES,
    SCOPE_PROMOTABLE,
    SCOPE_QUALITY_TARGETS,
    scope_profile,
)
from validate_easy_medium_reviewed import (
    CHECKSUM_FILENAME as REVIEWED_CHECKSUM_FILENAME,
    DATASET_KEYS,
    PROJECT_ROOT,
    QA_FILENAME,
    REVIEWED_ROOT,
    SUITE_FILENAME,
    SUITE_KEY,
    SUMMARY_FILENAME,
    VERSION,
    ScopedValidationError,
    REVIEWED_MANIFEST_ALLOWED_TOP_LEVEL_FIELDS,
    _assert_safe_path,
    _read_json,
    _sha256_file,
    _sha256_json,
    validate as validate_reviewed,
)


ROOT = Path(__file__).absolute().parent
FINAL_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-13-easy-medium-evaluation"
)
ACTIVATION_SCHEMA = "stt.eval.easy-medium-scope-activation.v1"
QUALITY_TARGETS_SCHEMA = "stt.eval.quality-targets.v1"
DERIVATION_SCHEMA = "stt.eval.dataset-candidate-activation-derivation.v1"
DERIVATION_RULES_VERSION = "stt.eval.reviewed-suite-activation.v1"
ACTIVE_DATASET_SCHEMA = "stt.eval.dataset.v2-llm-reviewed-candidate-active"
STATIC_CHECKSUM_FILENAME = "SHA256SUMS"
STATIC_DYNAMIC_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    "runs",
    "baselines",
    "certification",
    "semantic-reviews",
    "service",
}
FORBIDDEN_PROMOTION_SURFACES = (
    "candidate-roots.json",
    "baselines",
    "certification",
)


class ScopedActivationError(RuntimeError):
    """表示 scoped activation 无法在不放宽门禁时完成。"""


def _assert_no_promotion_surfaces(root: Path) -> None:
    """拒绝 scoped final root 中任何候选晋级、基线或认证表面。"""

    for relative in FORBIDDEN_PROMOTION_SURFACES:
        path = root / relative
        if path.exists() or path.is_symlink():
            raise ScopedActivationError(
                f"forbidden promotion surface exists in scoped root: {relative}"
            )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative_project(path: Path) -> str:
    safe_path = _assert_safe_path(path, root=PROJECT_ROOT, label="scoped activation input")
    return safe_path.relative_to(Path(os.path.abspath(os.fspath(PROJECT_ROOT)))).as_posix()


def _derive_active_manifest(
    reviewed_manifest: dict[str, Any],
    *,
    input_relative_path: str,
    input_file_sha256: str,
) -> dict[str, Any]:
    """在不修改 reviewed 输入的前提下推导 candidate-active manifest。"""

    unexpected = set(reviewed_manifest) - REVIEWED_MANIFEST_ALLOWED_TOP_LEVEL_FIELDS
    if unexpected:
        raise ScopedActivationError(
            "reviewed manifest has forbidden top-level fields: "
            + ", ".join(sorted(unexpected))
        )
    active = deepcopy(reviewed_manifest)
    active["schema_version"] = ACTIVE_DATASET_SCHEMA
    active["status"] = "candidate_active"
    active["import_allowed"] = True
    active["candidate_run_allowed"] = SCOPE_CANDIDATE_RUN_ALLOWED
    active["promotable"] = SCOPE_PROMOTABLE
    active["baseline_registration_allowed"] = SCOPE_BASELINE_REGISTRATION_ALLOWED
    active["canonical_promotion_allowed"] = SCOPE_CANONICAL_PROMOTION_ALLOWED
    active["scope"] = scope_profile()
    generation = dict(active.get("generation") or {})
    generation["independent_validation_status"] = "passed"
    active["generation"] = generation
    gate_policy = dict(active.get("gate_policy") or {})
    gate_policy["case_pass_policy_activation_status"] = "active_candidate"
    gate_policy["case_pass_policy_materialization_required"] = False
    active["gate_policy"] = gate_policy
    cases = active.get("cases")
    if not isinstance(cases, list):
        raise ScopedActivationError("reviewed manifest has no cases")
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ScopedActivationError(f"reviewed case {position} is invalid")
        contract = case.get("answer_contract")
        if not isinstance(contract, dict) or contract.get("contract_status") != (
            "llm_reviewed_candidate"
        ):
            raise ScopedActivationError(f"reviewed case {position} is not approved")
        review = contract.get("review")
        if not isinstance(review, dict) or review.get("verdict") not in {
            "approve",
            "revise",
        }:
            raise ScopedActivationError(f"reviewed case {position} has no approved verdict")
        contract["contract_status"] = "llm_reviewed_candidate_active"
        review["independent_validation_status"] = "passed"
    receipt = {
        "schema_version": DERIVATION_SCHEMA,
        "input_path": input_relative_path,
        "input_file_sha256": input_file_sha256,
        "input_manifest_sha256": _sha256_json(reviewed_manifest),
        "output_state": "candidate_active",
        "rules_version": DERIVATION_RULES_VERSION,
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    active["activation_derivation"] = receipt
    return active


def _quality_targets() -> dict[str, Any]:
    expected = {
        str(key): int(value) for key, value in SCOPE_QUALITY_TARGETS.items()
    }
    if expected != {"easy": 48, "medium": 43}:
        raise ScopedActivationError("frozen easy/medium quality targets drifted")
    return {
        "schema_version": QUALITY_TARGETS_SCHEMA,
        "suite_key": SUITE_KEY,
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
        "targets": {
            "easy": {
                "required": True,
                "operator": ">",
                "threshold": 0.95,
                "case_count": 50,
                "minimum_correct_cases": 48,
            },
            "medium": {
                "required": True,
                "operator": ">",
                "threshold": 0.85,
                "case_count": 50,
                "minimum_correct_cases": 43,
            },
        },
    }


def _validate_source_integrity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScopedActivationError("scoped reviewed source/index receipt is required")
    source = value.get("source")
    active_index = value.get("active_index")
    review = value.get("review")
    source_path = str(source.get("path") or "") if isinstance(source, Mapping) else ""
    source_sha256 = str(source.get("sha256") or "") if isinstance(source, Mapping) else ""
    receipt_fields = (
        ("source_qa_summary_path", "source_qa_summary_sha256"),
        ("blocker_validation_path", "blocker_validation_sha256"),
    )
    if (
        not isinstance(source, Mapping)
        or not isinstance(active_index, Mapping)
        or not isinstance(review, Mapping)
        or not source_path
        or Path(source_path).is_absolute()
        or ".." in Path(source_path).parts
        or "\\" in source_path
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        or source.get("page_count") != 95
        or active_index.get("kb_id") != EXPECTED_KB_ID
        or active_index.get("document_id") != EXPECTED_DOCUMENT_ID
        or active_index.get("document_version_id") != EXPECTED_DOCUMENT_VERSION_ID
        or active_index.get("index_version_id") != EXPECTED_INDEX_VERSION_ID
        or active_index.get("frozen_chunk_count") != 218
        or any(
            not str(active_index.get(path_key) or "")
            or Path(str(active_index.get(path_key) or "")).is_absolute()
            or ".." in Path(str(active_index.get(path_key) or "")).parts
            or "\\" in str(active_index.get(path_key) or "")
            or re.fullmatch(
                r"[0-9a-f]{64}", str(active_index.get(sha_key) or "")
            )
            is None
            for path_key, sha_key in receipt_fields
        )
        or review.get("scope_id") != scope_profile()["scope_id"]
        or review.get("scoped_case_count") != 100
        or review.get("hard_reference_only") is not True
        or review.get("hard_imported_case_count") != 0
        or review.get("full_source_joined_case_count_reference_only") != 150
    ):
        raise ScopedActivationError("scoped reviewed source/index receipt is incomplete")
    return deepcopy(dict(value))


def build_activation(
    *,
    reviewed_root: Path = REVIEWED_ROOT,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """重放 scoped 校验并构造不具备 canonical 晋升权限的 activation。"""

    validation = validate_reviewed(reviewed_root)
    suite = _read_json(
        reviewed_root / SUITE_FILENAME,
        root=reviewed_root,
        label="scoped reviewed suite",
    )
    summary = _read_json(
        reviewed_root / SUMMARY_FILENAME,
        root=reviewed_root,
        label="scoped reviewed summary",
    )
    if not isinstance(suite, Mapping) or not isinstance(summary, Mapping):
        raise ScopedActivationError("scoped reviewed suite/summary is invalid")
    integrity = _validate_source_integrity(suite.get("source_and_index_integrity"))
    review_integrity = dict(integrity["review"])
    review_integrity.update(
        {
            "scoped_validation_schema": validation["schema_version"],
            "scoped_validation_sha256": _sha256_json(validation),
            "scoped_reviewed_suite_path": _safe_relative_project(
                reviewed_root / SUITE_FILENAME
            ),
            "scoped_reviewed_suite_sha256": validation["suite"]["sha256"],
            "scoped_reviewed_summary_path": _safe_relative_project(
                reviewed_root / SUMMARY_FILENAME
            ),
            "scoped_reviewed_summary_sha256": validation["summary"]["sha256"],
            "scoped_reviewed_qa_path": _safe_relative_project(reviewed_root / QA_FILENAME),
            "scoped_reviewed_qa_sha256": validation["qa"]["sha256"],
            "scoped_reviewed_checksums_path": _safe_relative_project(
                reviewed_root / REVIEWED_CHECKSUM_FILENAME
            ),
            "scoped_reviewed_checksums_sha256": validation["checksums"]["sha256"],
            "authoritative_scope_receipts_sha256": validation[
                "authoritative_scope_receipts_sha256"
            ],
            "scoped_case_count": 100,
            "hard_reference_only": True,
            "hard_imported_case_count": 0,
        }
    )
    integrity["review"] = review_integrity

    manifests: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for difficulty in SCOPE_DIFFICULTIES:
        input_path = reviewed_root / "datasets" / f"{difficulty}.reviewed.json"
        reviewed_manifest = _read_json(
            input_path,
            root=reviewed_root,
            label=f"{difficulty} scoped reviewed manifest",
        )
        input_file_sha256 = _sha256_file(
            input_path,
            root=reviewed_root,
            label=f"{difficulty} scoped reviewed manifest",
        )
        expected_source = {
            "relative_path": integrity["source"]["path"],
            "content_sha256": integrity["source"]["sha256"],
            "page_count": integrity["source"]["page_count"],
            "kb_id": integrity["active_index"]["kb_id"],
            "document_id": integrity["active_index"]["document_id"],
            "document_version_id": integrity["active_index"]["document_version_id"],
            "index_version_id": integrity["active_index"]["index_version_id"],
        }
        manifest_source = reviewed_manifest.get("source")
        if not isinstance(manifest_source, Mapping) or any(
            manifest_source.get(key) != expected
            for key, expected in expected_source.items()
        ):
            raise ScopedActivationError(
                f"{difficulty} reviewed manifest source/index identity drifted"
            )
        active = _derive_active_manifest(
            dict(reviewed_manifest),
            input_relative_path=_safe_relative_project(input_path),
            input_file_sha256=input_file_sha256,
        )
        output_bytes = _json_bytes(active)
        derivation = active["activation_derivation"]
        manifests[difficulty] = active
        entries[difficulty] = {
            "path": f"datasets/{difficulty}.json",
            "sha256": _sha256_bytes(output_bytes),
            "manifest_sha256": _sha256_json(active),
            "case_count": 50,
            "dataset_key": DATASET_KEYS[difficulty],
            "version": VERSION,
            "input": {
                "path": derivation["input_path"],
                "file_sha256": input_file_sha256,
                "manifest_sha256": _sha256_json(reviewed_manifest),
            },
            "derivation_receipt_sha256": derivation["receipt_sha256"],
        }

    quality_targets = _quality_targets()
    quality_sha256 = _sha256_bytes(_json_bytes(quality_targets))
    activation = {
        "schema_version": ACTIVATION_SCHEMA,
        "suite_key": SUITE_KEY,
        "scope": scope_profile(),
        "import_allowed": True,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "evaluation_contract": {
            "case_schema_version": "stt.eval.case.v3",
            "authoritative_metric": "metrics.case_outcome.final_answer_correct",
            "target_counts": {
                "easy": {"correct_count": 48, "case_count": 50},
                "medium": {"correct_count": 43, "case_count": 50},
            },
            "case_pass_policy": deepcopy(
                manifests["easy"]["gate_policy"]["case_pass_policy"]
            ),
        },
        "quality_targets": {
            "path": "quality-targets.json",
            "sha256": quality_sha256,
            "schema_version": QUALITY_TARGETS_SCHEMA,
        },
        "source_and_index_integrity": integrity,
        "datasets": entries,
    }
    preflight = {
        "status": "ready_to_activate",
        "scope": scope_profile(),
        "reviewed_validation": validation,
        "reviewed_validation_sha256": _sha256_json(validation),
        "case_count": 100,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "activation_sha256": _sha256_bytes(_json_bytes(activation)),
        "quality_targets_sha256": quality_sha256,
        "dataset_file_sha256": {
            difficulty: entries[difficulty]["sha256"]
            for difficulty in SCOPE_DIFFICULTIES
        },
    }
    return activation, quality_targets, manifests, preflight


def _static_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    safe_root = _assert_safe_path(root, root=root, label="scoped final root")
    for path in safe_root.rglob("*"):
        relative = path.relative_to(safe_root)
        if any(part in STATIC_DYNAMIC_DIRECTORIES for part in relative.parts):
            continue
        _assert_safe_path(path, root=safe_root, label="scoped static artifact")
        if path.is_file() and relative.as_posix() != STATIC_CHECKSUM_FILENAME:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(safe_root).as_posix())


def _checksum_bytes(root: Path, outputs: Mapping[str, bytes]) -> bytes:
    payloads: dict[str, bytes] = {}
    for path in _static_files(root):
        relative = path.relative_to(root).as_posix()
        payloads[relative] = path.read_bytes()
    payloads.update(outputs)
    payloads.pop(STATIC_CHECKSUM_FILENAME, None)
    return "".join(
        f"{_sha256_bytes(payloads[relative])}  {relative}\n"
        for relative in sorted(payloads)
    ).encode("utf-8")


def _write_exact_or_refuse(path: Path, content: bytes, *, root: Path) -> None:
    _assert_safe_path(
        path,
        root=root,
        label="scoped activation output",
        require_exists=False,
    )
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ScopedActivationError(f"refusing conflicting activation output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _assert_safe_path(
        temporary,
        root=root,
        label="scoped activation temporary",
        require_exists=False,
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != content:
        raise ScopedActivationError("scoped activation post-write verification failed")


def activate(
    *,
    write_final_root: bool,
    reviewed_root: Path = REVIEWED_ROOT,
    final_root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    """执行 dry-run 或将完全相同的 scoped activation 原子写入新根。"""

    activation, quality_targets, manifests, preflight = build_activation(
        reviewed_root=reviewed_root
    )
    result = {
        **preflight,
        "status": "activated" if write_final_root else "ready_to_activate",
        "write_performed": write_final_root,
        "final_root": str(final_root),
    }
    if not write_final_root:
        return result
    _assert_safe_path(
        final_root,
        root=PROJECT_ROOT,
        label="scoped final root",
        require_exists=False,
    )
    final_root.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(final_root, root=PROJECT_ROOT, label="scoped final root")
    _assert_no_promotion_surfaces(final_root)
    outputs = {
        "activation.json": _json_bytes(activation),
        "quality-targets.json": _json_bytes(quality_targets),
        **{
            f"datasets/{difficulty}.json": _json_bytes(manifests[difficulty])
            for difficulty in SCOPE_DIFFICULTIES
        },
    }
    checksum = _checksum_bytes(final_root, outputs)
    complete_outputs = {**outputs, STATIC_CHECKSUM_FILENAME: checksum}
    for relative, content in complete_outputs.items():
        target = final_root / relative
        if target.exists() and (not target.is_file() or target.read_bytes() != content):
            raise ScopedActivationError(f"refusing conflicting activation output: {target}")
    for relative, content in outputs.items():
        _write_exact_or_refuse(final_root / relative, content, root=final_root)
    _write_exact_or_refuse(
        final_root / STATIC_CHECKSUM_FILENAME,
        checksum,
        root=final_root,
    )
    result["written_files"] = {
        relative: _sha256_file(
            final_root / relative,
            root=final_root,
            label=f"scoped final {relative}",
        )
        for relative in complete_outputs
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-final-root",
        action="store_true",
        help="将已验证 scoped 产物写入独立 easy/medium 评测根",
    )
    args = parser.parse_args()
    try:
        result = activate(write_final_root=args.write_final_root)
    except (
        OSError,
        ValueError,
        ScopedValidationError,
        ScopedActivationError,
    ) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
