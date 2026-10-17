"""Fail-closed activation of the fully reviewed suite into the final live root.

The default is a dry run.  ``--write-final-root`` is the only write gate.  No
boolean is manually promoted: every activation field is derived from validated
reviewed artifacts, source/index receipts, and zero-unresolved-blocker checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from merge_reviewed_contracts import FROZEN_CASE_PASS_POLICY
from review_release_integrity import (
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    validate_block_union_freshness,
    validate_proposal_validation_binding,
)
from validate_suite_v2 import validate as validate_suite_v2


ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = ROOT.parents[4]
FINAL_ROOT = PROJECT_ROOT / "docs" / "evaluation-baselines" / "2026-08-11-three-difficulty-final"
REVIEWED_ROOT = ROOT / "reviewed"
SOURCE_QA_ROOT = ROOT.parent / "suite-source-evaluator-qa"
BLOCKER_ROOT = ROOT.parent / "suite-v2-blocker-remediation"
DIFFICULTIES = ("easy", "medium", "hard")
SUITE_KEY = "three-difficulty-live-2026-08-11-reviewed-v2"
VERSION = "2.0.0-reviewed-v2.20260811"
ACTIVE_DATASET_SCHEMA = "stt.eval.dataset.v2-llm-reviewed-candidate-active"
DERIVATION_SCHEMA = "stt.eval.dataset-candidate-activation-derivation.v1"
DERIVATION_RULES_VERSION = "stt.eval.reviewed-suite-activation.v1"
QUALITY_TARGETS_SCHEMA = "stt.eval.quality-targets.v1"
EXPECTED_KB_ID = "kb_2fad25538aab42d19baf79fba782a06c"
EXPECTED_DOCUMENT_ID = "doc_9353bf25fcfa4991a8a0425713a3c9f1"
EXPECTED_DOCUMENT_VERSION_ID = "dv_8d7fe6750ff5445e8dfbfd80b913b216"
EXPECTED_INDEX_VERSION_ID = "idx_fb5eeac133f540f4af31004dbde23151"
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
DATASET_KEYS = {
    difficulty: f"pantum-p3500-{difficulty}-live-reviewed-v2-2026-08-11"
    for difficulty in DIFFICULTIES
}
TARGET_COUNTS = {
    "easy": {"correct_count": 48, "case_count": 50},
    "medium": {"correct_count": 43, "case_count": 50},
    "hard": {"correct_count": None, "case_count": 50},
}


class ActivationError(RuntimeError):
    """Raised when a reviewed suite cannot be activated safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_safe_path(
    path: Path,
    *,
    root: Path,
    label: str,
    require_exists: bool = True,
) -> Path:
    root_absolute = _absolute(root)
    path_absolute = _absolute(path)
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ActivationError(f"{label} escapes its trusted root") from exc
    project_absolute = _absolute(PROJECT_ROOT)
    try:
        root_relative = root_absolute.relative_to(project_absolute)
    except ValueError:
        root_relative = None
    if root_relative is not None:
        for component in (Path(), *root_relative.parents[::-1], root_relative):
            candidate = project_absolute if component == Path() else project_absolute / component
            if candidate.exists() or candidate.is_symlink():
                if _is_reparse_point(candidate):
                    raise ActivationError(
                        f"{label} trusted-root chain contains a symlink or reparse point: "
                        f"{candidate}"
                    )
        try:
            root_absolute.resolve(strict=True).relative_to(project_absolute.resolve(strict=True))
        except ValueError as exc:
            raise ActivationError(f"{label} trusted root resolves outside repository") from exc
    for component in (Path(), *relative.parents[::-1], relative):
        candidate = root_absolute if component == Path() else root_absolute / component
        if candidate.exists() or candidate.is_symlink():
            if _is_reparse_point(candidate):
                raise ActivationError(f"{label} contains a symlink or reparse point: {candidate}")
    resolved_root = root_absolute.resolve(strict=True)
    resolved_path = path_absolute.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ActivationError(f"{label} resolves outside its trusted root") from exc
    if require_exists and not path_absolute.exists():
        raise ActivationError(f"{label} is missing: {path_absolute}")
    return path_absolute


def _sha256_file(path: Path, *, root: Path = PROJECT_ROOT, label: str = "file") -> str:
    safe_path = _assert_safe_path(path, root=root, label=label)
    if not safe_path.is_file():
        raise ActivationError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    with safe_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    _assert_safe_path(safe_path, root=root, label=label)
    return digest.hexdigest()


def _read_json(path: Path, *, root: Path = PROJECT_ROOT, label: str = "JSON file") -> Any:
    safe_path = _assert_safe_path(path, root=root, label=label)
    if not safe_path.is_file():
        raise ActivationError(f"{label} is not a regular file")
    try:
        value = json.loads(safe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActivationError(f"{label} is not valid JSON") from exc
    _assert_safe_path(safe_path, root=root, label=label)
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    return path.is_symlink() or bool(attributes & 0x400)


def _assert_safe_final_root() -> None:
    _assert_safe_path(
        FINAL_ROOT,
        root=PROJECT_ROOT,
        label="final activation root",
        require_exists=False,
    )
    if FINAL_ROOT.exists():
        _assert_safe_path(FINAL_ROOT, root=PROJECT_ROOT, label="final activation root")


def _safe_relative_project(path: Path) -> str:
    safe_path = _assert_safe_path(path, root=PROJECT_ROOT, label="activation input")
    return safe_path.relative_to(_absolute(PROJECT_ROOT)).as_posix()


def _derive_candidate_active_manifest(
    reviewed_manifest: dict[str, Any],
    *,
    difficulty: str,
    input_relative_path: str,
    input_file_sha256: str,
) -> dict[str, Any]:
    """Derive an importable candidate without mutating reviewed evidence."""

    active = deepcopy(reviewed_manifest)
    active["schema_version"] = ACTIVE_DATASET_SCHEMA
    active["status"] = "candidate_active"
    active["import_allowed"] = True
    active["candidate_run_allowed"] = True
    active["baseline_registration_allowed"] = False
    generation = dict(active.get("generation") or {})
    generation["independent_validation_status"] = "passed"
    active["generation"] = generation
    gate_policy = dict(active.get("gate_policy") or {})
    gate_policy["case_pass_policy"] = deepcopy(FROZEN_CASE_PASS_POLICY)
    gate_policy["case_pass_policy_activation_status"] = "active_candidate"
    gate_policy["case_pass_policy_materialization_required"] = False
    active["gate_policy"] = gate_policy
    cases = active.get("cases")
    if not isinstance(cases, list):
        raise ActivationError(f"reviewed {difficulty} manifest has no cases")
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ActivationError(f"reviewed {difficulty} case {position} is invalid")
        contract = case.get("answer_contract")
        if not isinstance(contract, dict):
            raise ActivationError(f"reviewed {difficulty} case {position} has no contract")
        if contract.get("contract_status") != "llm_reviewed_candidate":
            raise ActivationError(
                f"reviewed {difficulty} case {position} is not an approved candidate"
            )
        contract["contract_status"] = "llm_reviewed_candidate_active"
        review = contract.get("review")
        if not isinstance(review, dict) or review.get("verdict") == "block":
            raise ActivationError(
                f"reviewed {difficulty} case {position} has unresolved review state"
            )
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


def _assert_input_tree_safe(root: Path, *, label: str) -> None:
    safe_root = _assert_safe_path(root, root=PROJECT_ROOT, label=label)
    stack = [safe_root]
    while stack:
        directory = stack.pop()
        _assert_safe_path(directory, root=PROJECT_ROOT, label=label)
        for child in directory.iterdir():
            _assert_safe_path(child, root=PROJECT_ROOT, label=label)
            if child.is_dir():
                stack.append(child)
            elif not child.is_file():
                raise ActivationError(f"{label} contains an unsupported filesystem entry")


def _build() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    for input_root, label in (
        (ROOT, "suite review inputs"),
        (SOURCE_QA_ROOT, "source QA inputs"),
        (BLOCKER_ROOT, "blocker/index inputs"),
    ):
        _assert_input_tree_safe(input_root, label=label)
    try:
        validation = validate_suite_v2(require_reviewed=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ActivationError(f"suite-v2 final validator did not pass: {exc}") from exc
    reviewed_summary = _read_json(REVIEWED_ROOT / "summary.json")
    block_union_audit = _read_json(ROOT / "review-block-union-audit.json")
    try:
        validate_block_union_freshness(
            block_union_audit,
            requests_path=ROOT / "review" / "requests.jsonl",
            responses_path=ROOT / "review" / "responses.jsonl",
            visual_audit_path=ROOT / "source-visual-audit.json",
            base_proposals_path=BLOCKER_ROOT / "proposals.json",
            supplemental_proposals_path=ROOT / "reviewer-block-supplemental-proposals.json",
            label="activation block-union audit",
        )
    except RuntimeError as exc:
        raise ActivationError(str(exc)) from exc
    reviewed_qa = _read_json(REVIEWED_ROOT / "qa-reviewed.json")
    if (
        reviewed_summary.get("case_count") != 150
        or reviewed_summary.get("concept_groups_materialized_case_count") != 150
        or reviewed_summary.get("blocked_case_count") != 0
        or reviewed_summary.get("confirmed_source_visual_override_count") != 0
        or block_union_audit.get("status") != "passed"
    ):
        raise ActivationError("reviewed summary still has missing concepts or effective blockers")

    expected_data_ids = set(
        str(value) for value in block_union_audit.get("required_data_remediation_case_ids") or []
    )
    if (
        len(expected_data_ids) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
        or reviewed_summary.get("blocker_rereview_case_count")
        != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
    ):
        raise ActivationError("activation requires the exact 15-case blocker rereview receipt")
    applied_data_rows = [
        row for row in reviewed_qa if row.get("blocker_remediation_applied") is True
    ]
    applied_data_ids = {str(row.get("parent_case_id") or "") for row in applied_data_rows}
    unresolved_data_ids = sorted(
        (expected_data_ids - applied_data_ids)
        | {
            str(row.get("parent_case_id") or "")
            for row in applied_data_rows
            if row.get("verdict") == "block"
        }
    )
    if unresolved_data_ids:
        raise ActivationError(f"unresolved data blocker(s): {unresolved_data_ids}")

    contract_rereview_count = int(reviewed_summary.get("contract_rereview_case_count") or 0)
    weak_failure_count = 0
    contract_audit_sha256: str | None = None
    if contract_rereview_count:
        contract_audit_path = ROOT / "contract-rereview" / "audit.json"
        contract_audit = _read_json(contract_audit_path)
        weak_failure_count = int(contract_audit.get("reviewer_contract_failure_count") or 0)
        contract_audit_sha256 = _sha256_file(contract_audit_path)
        if contract_audit.get("status") != "passed" or weak_failure_count:
            raise ActivationError("same-semantic weak-contract rereview still has failures")

    source_qa_summary_path = SOURCE_QA_ROOT / "summary.json"
    source_qa_summary = _read_json(source_qa_summary_path)
    source_checks = source_qa_summary.get("input_checks") or {}
    blocker_validation_path = BLOCKER_ROOT / "validation.json"
    blocker_validation = _read_json(blocker_validation_path)
    active_index = blocker_validation.get("active_index") or {}
    if (
        source_checks.get("source_hash_matches") is not True
        or source_checks.get("source_page_count_matches") is not True
        or source_checks.get("joined_case_count") != 150
        or source_checks.get("indexed_evidence_count") != 218
        or blocker_validation.get("status") != "passed"
        or blocker_validation.get("selected_chunks_resolve_to_frozen_active_index") is not True
        or active_index.get("frozen_chunk_count") != 218
        or active_index.get("index_rebuild_required") is not False
        or active_index.get("evidence_refresh_required") is not False
    ):
        raise ActivationError("source/index integrity receipts are not activation-ready")
    try:
        validate_proposal_validation_binding(
            blocker_validation,
            proposals_path=BLOCKER_ROOT / "proposals.json",
            label="activation base proposal validation",
        )
    except RuntimeError as exc:
        raise ActivationError(str(exc)) from exc

    source_relative = str(source_checks.get("source_path") or "")
    source_path = _assert_safe_path(
        PROJECT_ROOT / source_relative,
        root=PROJECT_ROOT,
        label="source PDF",
    )
    source_sha256 = str(source_checks.get("source_sha256") or "")
    index_version_id = str(source_checks.get("index_version_id") or "")
    if (
        not source_path.is_file()
        or _sha256_file(source_path, label="source PDF") != source_sha256
        or active_index.get("index_version_id") != index_version_id
        or index_version_id != EXPECTED_INDEX_VERSION_ID
    ):
        raise ActivationError("source PDF or active-index identity drifted")

    manifest_payloads: dict[str, dict[str, Any]] = {}
    manifest_entries: dict[str, dict[str, Any]] = {}
    reviewed_cases = 0
    abstain_cases = 0
    global_case_ids: list[str] = []
    for difficulty in DIFFICULTIES:
        path = REVIEWED_ROOT / "datasets" / f"{difficulty}.reviewed.json"
        manifest = _read_json(path, label=f"reviewed {difficulty} manifest")
        cases = manifest.get("cases") or []
        source = manifest.get("source") or {}
        if (
            manifest.get("suite_key") != SUITE_KEY
            or manifest.get("dataset_key") != DATASET_KEYS[difficulty]
            or manifest.get("version") != VERSION
            or len(cases) != 50
            or (manifest.get("gate_policy") or {}).get("case_pass_policy")
            != FROZEN_CASE_PASS_POLICY
            or source.get("content_sha256") != source_sha256
            or source.get("relative_path") != source_relative
            or source.get("page_count") != source_checks.get("source_pdf_page_count")
            or source.get("kb_id") != EXPECTED_KB_ID
            or source.get("document_id") != EXPECTED_DOCUMENT_ID
            or source.get("document_version_id") != EXPECTED_DOCUMENT_VERSION_ID
            or source.get("index_version_id") != EXPECTED_INDEX_VERSION_ID
            or manifest.get("import_allowed") is not False
            or manifest.get("baseline_registration_allowed") is not False
        ):
            raise ActivationError(f"reviewed {difficulty} manifest identity/policy drift")
        for case in cases:
            case_id = str(case.get("case_id") or "")
            if not case_id:
                raise ActivationError(f"empty case_id in reviewed {difficulty} manifest")
            global_case_ids.append(case_id)
            groups = (case.get("concept_groups") or {}).get("all_of") or []
            if not groups:
                raise ActivationError(f"empty concept_groups in {case.get('case_id')}")
            answerability = str(
                ((case.get("answer_contract") or {}).get("answerability") or {}).get("required")
                or ""
            )
            if answerability == "abstain":
                abstain_cases += 1
        reviewed_cases += len(cases)
        input_file_sha256 = _sha256_file(path, label=f"reviewed {difficulty} manifest")
        input_relative_path = _safe_relative_project(path)
        active_manifest = _derive_candidate_active_manifest(
            manifest,
            difficulty=difficulty,
            input_relative_path=input_relative_path,
            input_file_sha256=input_file_sha256,
        )
        output_bytes = _json_bytes(active_manifest)
        output_file_sha256 = hashlib.sha256(output_bytes).hexdigest()
        if output_file_sha256 == input_file_sha256:
            raise ActivationError(f"{difficulty} activation did not derive a new manifest SHA")
        derivation = active_manifest["activation_derivation"]
        manifest_payloads[difficulty] = active_manifest
        manifest_entries[difficulty] = {
            "path": f"datasets/{difficulty}.json",
            "sha256": output_file_sha256,
            "manifest_sha256": _sha256_json(active_manifest),
            "case_count": 50,
            "dataset_key": DATASET_KEYS[difficulty],
            "version": VERSION,
            "input": {
                "path": input_relative_path,
                "file_sha256": input_file_sha256,
                "manifest_sha256": _sha256_json(manifest),
            },
            "derivation_receipt_sha256": derivation["receipt_sha256"],
        }
    if (
        reviewed_cases != 150
        or abstain_cases != 5
        or len(global_case_ids) != 150
        or len(set(global_case_ids)) != 150
    ):
        raise ActivationError("reviewed case/abstention count drift")
    if FROZEN_CASE_PASS_POLICY.get("abstention_factual_claim_maximum") != 0:
        raise ActivationError("case.v3 abstention factual maximum is not frozen at zero")

    integrity = {
        "source": {
            "path": source_relative,
            "sha256": source_sha256,
            "page_count": source_checks.get("source_pdf_page_count"),
        },
        "active_index": {
            "kb_id": EXPECTED_KB_ID,
            "document_id": EXPECTED_DOCUMENT_ID,
            "document_version_id": EXPECTED_DOCUMENT_VERSION_ID,
            "index_version_id": index_version_id,
            "frozen_chunk_count": 218,
            "source_qa_summary_path": source_qa_summary_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_qa_summary_sha256": _sha256_file(source_qa_summary_path),
            "blocker_validation_path": blocker_validation_path.relative_to(PROJECT_ROOT).as_posix(),
            "blocker_validation_sha256": _sha256_file(blocker_validation_path),
        },
        "review": {
            "block_union_audit_path": (ROOT / "review-block-union-audit.json")
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "block_union_audit_sha256": _sha256_file(ROOT / "review-block-union-audit.json"),
            "contract_rereview_audit_sha256": contract_audit_sha256,
            "unresolved_data_block_count": 0,
            "unresolved_weak_contract_count": weak_failure_count,
            "unresolved_visual_block_count": 0,
            "concept_groups_materialized_case_count": 150,
        },
    }
    quality_targets = {
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
            "hard": {
                "required": False,
                "operator": None,
                "threshold": None,
                "case_count": 50,
                "minimum_correct_cases": None,
            },
        },
    }
    quality_bytes = _json_bytes(quality_targets)
    quality_sha256 = hashlib.sha256(quality_bytes).hexdigest()
    activation = {
        "schema_version": "stt.eval.final-suite-activation.v1",
        "suite_key": SUITE_KEY,
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "evaluation_contract": {
            "case_schema_version": "stt.eval.case.v3",
            "authoritative_metric": "metrics.case_outcome.final_answer_correct",
            "target_counts": TARGET_COUNTS,
            "case_pass_policy": FROZEN_CASE_PASS_POLICY,
        },
        "quality_targets": {
            "path": "quality-targets.json",
            "sha256": quality_sha256,
            "schema_version": QUALITY_TARGETS_SCHEMA,
        },
        "source_and_index_integrity": integrity,
        "datasets": manifest_entries,
    }
    preflight = {
        "validator": validation,
        "reviewed_case_count": reviewed_cases,
        "unresolved_data_block_count": 0,
        "unresolved_weak_contract_count": weak_failure_count,
        "unresolved_visual_block_count": 0,
        "concept_groups_materialized_case_count": 150,
        "case_pass_policy_sha256": _sha256_json(FROZEN_CASE_PASS_POLICY),
        "activation_sha256": hashlib.sha256(_json_bytes(activation)).hexdigest(),
        "quality_targets_sha256": quality_sha256,
        "dataset_file_sha256": {
            difficulty: manifest_entries[difficulty]["sha256"] for difficulty in DIFFICULTIES
        },
    }
    return activation, quality_targets, manifest_payloads, preflight


def _write_exact_or_refuse(path: Path, content: bytes) -> None:
    _assert_safe_final_root()
    safe_path = _assert_safe_path(
        path,
        root=FINAL_ROOT,
        label="final activation artifact",
        require_exists=False,
    )
    _assert_safe_path(
        safe_path.parent,
        root=FINAL_ROOT,
        label="final activation parent",
        require_exists=False,
    )
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(safe_path.parent, root=FINAL_ROOT, label="final activation parent")
    if safe_path.exists():
        _assert_safe_path(safe_path, root=FINAL_ROOT, label="final activation artifact")
        if not safe_path.is_file() or safe_path.read_bytes() != content:
            raise ActivationError(
                f"refusing to overwrite non-identical final artifact: {safe_path}"
            )
        _assert_safe_path(safe_path, root=FINAL_ROOT, label="final activation artifact")
        return
    temporary = safe_path.with_name(f".{safe_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    _assert_safe_path(
        temporary,
        root=FINAL_ROOT,
        label="final activation temporary",
        require_exists=False,
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_safe_path(
            temporary,
            root=FINAL_ROOT,
            label="final activation temporary",
        )
        _assert_safe_path(
            safe_path.parent,
            root=FINAL_ROOT,
            label="final activation parent",
        )
        os.replace(temporary, safe_path)
        _assert_safe_path(safe_path, root=FINAL_ROOT, label="final activation artifact")
        if safe_path.read_bytes() != content:
            raise ActivationError("final activation artifact post-write verification failed")
        _assert_safe_path(safe_path, root=FINAL_ROOT, label="final activation artifact")
    finally:
        if temporary.exists() and not _is_reparse_point(temporary):
            temporary.unlink()


def _static_files(root: Path) -> list[Path]:
    safe_root = _assert_safe_path(root, root=root, label="final static root")
    files: list[Path] = []
    stack = [safe_root]
    while stack:
        directory = stack.pop()
        _assert_safe_path(directory, root=safe_root, label="final static directory")
        for child in directory.iterdir():
            relative = child.relative_to(safe_root)
            if any(part in STATIC_DYNAMIC_DIRECTORIES for part in relative.parts):
                continue
            if relative.as_posix() == STATIC_CHECKSUM_FILENAME:
                continue
            _assert_safe_path(child, root=safe_root, label="final static artifact")
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                files.append(child)
            else:
                raise ActivationError(f"unsupported final static artifact: {child}")
    return sorted(files, key=lambda item: item.relative_to(safe_root).as_posix())


def _static_checksum_bytes(overrides: dict[str, bytes]) -> bytes:
    digests: dict[str, str] = {}
    for path in _static_files(FINAL_ROOT):
        relative = path.relative_to(FINAL_ROOT).as_posix()
        if relative not in overrides:
            digests[relative] = _sha256_file(
                path,
                root=FINAL_ROOT,
                label=f"final static artifact {relative}",
            )
    for relative, content in overrides.items():
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in relative
            or not candidate.parts
            or any(part in STATIC_DYNAMIC_DIRECTORIES for part in candidate.parts)
            or relative == STATIC_CHECKSUM_FILENAME
        ):
            raise ActivationError(f"unsafe static activation output: {relative}")
        _assert_safe_path(
            FINAL_ROOT / candidate,
            root=FINAL_ROOT,
            label="final static activation output",
            require_exists=False,
        )
        digests[relative] = hashlib.sha256(content).hexdigest()
    return "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(digests.items())
    ).encode("utf-8")


def activate(*, write_final_root: bool) -> dict[str, Any]:
    activation, quality_targets, manifest_payloads, preflight = _build()
    result = {
        "status": "ready_to_activate" if not write_final_root else "activated",
        "write_performed": write_final_root,
        "final_root": str(FINAL_ROOT),
        **preflight,
    }
    if not write_final_root:
        return result
    _assert_safe_final_root()
    outputs = {
        **{
            f"datasets/{difficulty}.json": _json_bytes(manifest_payloads[difficulty])
            for difficulty in DIFFICULTIES
        },
        "quality-targets.json": _json_bytes(quality_targets),
        "activation.json": _json_bytes(activation),
    }
    static_checksums = _static_checksum_bytes(outputs)
    # Refuse all conflicts before creating any new activation output.
    for relative, content in {**outputs, STATIC_CHECKSUM_FILENAME: static_checksums}.items():
        target = FINAL_ROOT / relative
        _assert_safe_path(
            target,
            root=FINAL_ROOT,
            label="final activation output",
            require_exists=False,
        )
        if target.exists() and (not target.is_file() or target.read_bytes() != content):
            raise ActivationError(f"refusing conflicting final activation output: {target}")
        if target.exists():
            _assert_safe_path(target, root=FINAL_ROOT, label="final activation output")
    for relative, content in outputs.items():
        _write_exact_or_refuse(FINAL_ROOT / relative, content)
    _write_exact_or_refuse(FINAL_ROOT / STATIC_CHECKSUM_FILENAME, static_checksums)

    sys.path.insert(0, str(FINAL_ROOT))
    import run_final_live  # noqa: PLC0415

    snapshot = run_final_live.load_suite_snapshot(FINAL_ROOT)
    if snapshot.activation_sha256 != preflight["activation_sha256"]:
        raise ActivationError("final live runner did not reproduce activation SHA")
    result["final_live_snapshot_sha256"] = snapshot.suite_identity_sha256
    result["written_files"] = {
        "activation.json": _sha256_file(
            FINAL_ROOT / "activation.json",
            root=FINAL_ROOT,
            label="final activation",
        ),
        "quality-targets.json": _sha256_file(
            FINAL_ROOT / "quality-targets.json",
            root=FINAL_ROOT,
            label="final quality targets",
        ),
        STATIC_CHECKSUM_FILENAME: _sha256_file(
            FINAL_ROOT / STATIC_CHECKSUM_FILENAME,
            root=FINAL_ROOT,
            label="final static checksums",
        ),
        **{
            f"datasets/{difficulty}.json": _sha256_file(
                FINAL_ROOT / "datasets" / f"{difficulty}.json",
                root=FINAL_ROOT,
                label=f"final {difficulty} dataset",
            )
            for difficulty in DIFFICULTIES
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-final-root",
        action="store_true",
        help="copy the validated manifests and write activation/quality targets",
    )
    args = parser.parse_args()
    try:
        result = activate(write_final_root=args.write_final_root)
    except (OSError, ValueError, ActivationError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
