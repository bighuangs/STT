#!/usr/bin/env python3
"""只读校验 easy/medium 专属 reviewed 产物及其完整性收据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

import merge_reviewed_contracts as full
from easy_medium_scope import (
    SCOPE_BASELINE_REGISTRATION_ALLOWED,
    SCOPE_CANDIDATE_RUN_ALLOWED,
    SCOPE_CANONICAL_PROMOTION_ALLOWED,
    SCOPE_DIFFICULTIES,
    SCOPE_ID,
    SCOPE_PROMOTABLE,
    SCOPE_QUALITY_TARGETS,
    load_frozen_candidate_manifest_scope_authority,
    scope_profile,
)


FROZEN_CASE_PASS_POLICY = full.FROZEN_CASE_PASS_POLICY


ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = ROOT.parents[4]
REVIEWED_ROOT = ROOT / "easy-medium-reviewed"
SUITE_FILENAME = "suite-easy-medium-reviewed.json"
QA_FILENAME = "qa-reviewed.json"
SUMMARY_FILENAME = "summary.json"
CHECKSUM_FILENAME = "SHA256SUMS"
SUITE_SCHEMA = "stt.eval.easy-medium-reviewed-suite.v1"
SUMMARY_SCHEMA = "stt.eval.easy-medium-reviewed-summary.v1"
VALIDATION_SCHEMA = "stt.eval.easy-medium-reviewed-validation.v1"
SUITE_KEY = "easy-medium-live-2026-08-13-reviewed-v2"
VERSION = "2.0.0-reviewed-v2.easy-medium.20260813"
DATASET_KEYS = {
    difficulty: (
        f"pantum-p3500-{difficulty}-live-reviewed-v2-easy-medium-2026-08-13"
    )
    for difficulty in SCOPE_DIFFICULTIES
}
EXPECTED_CASE_COUNT_PER_DIFFICULTY = 50
EXPECTED_CASE_COUNT = EXPECTED_CASE_COUNT_PER_DIFFICULTY * len(SCOPE_DIFFICULTIES)
EXPECTED_BLOCKER_REREVIEW_COUNTS = {"easy": 4, "medium": 0}
EXPECTED_CONTRACT_REREVIEW_COUNTS = {"easy": 8, "medium": 18}
EXPECTED_RECEIPT_KINDS = ("original", "blocker", "contract")
REVIEWED_CONTRACT_VALIDATOR = Draft202012Validator(full._reviewed_contract_schema())
REVIEWED_MANIFEST_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
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
        "cases",
    }
)
REVIEWED_SUITE_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "suite_key",
        "version",
        "status",
        "scope",
        "import_allowed",
        "candidate_run_allowed",
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
        "datasets",
        "source_and_index_integrity",
    }
)
REVIEWED_SUMMARY_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "suite_key",
        "scope_id",
        "scope",
        "case_count",
        "blocked_case_count",
        "difficulty_case_counts",
        "blocker_rereview_counts",
        "contract_rereview_counts",
        "authoritative_scope_receipts",
        "hard_reference_only",
        "hard_imported_case_count",
        "candidate_run_allowed",
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
        "review_prompt_sha256",
        "verdict_counts",
        "source_and_index_integrity_sha256",
    }
)
SOURCE_AND_INDEX_INTEGRITY_FIELDS = frozenset({"source", "active_index", "review"})
SOURCE_INTEGRITY_FIELDS = frozenset({"path", "sha256", "page_count"})
REVIEWED_SOURCE_ALLOWED_FIELDS = frozenset(
    {
        "relative_path",
        "content_sha256",
        "page_count",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "layout",
        "logical_name",
        "original_logical_name",
        "parent_dataset_key",
        "parent_dataset_version",
        "parent_manifest_file_sha256",
        "parent_suite_key",
        "reviewed_from_published_evidence",
    }
)
ACTIVE_INDEX_INTEGRITY_FIELDS = frozenset(
    {
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "frozen_chunk_count",
        "source_qa_summary_path",
        "source_qa_summary_sha256",
        "blocker_validation_path",
        "blocker_validation_sha256",
    }
)
REVIEW_INTEGRITY_REQUIRED_FIELDS = frozenset(
    {
        "scope_id",
        "scoped_case_count",
        "hard_reference_only",
        "hard_imported_case_count",
        "full_source_joined_case_count_reference_only",
        "block_union_audit_path",
        "block_union_audit_sha256",
        "original_scope_receipt_sha256",
        "blocker_scope_receipt_sha256",
        "contract_scope_receipt_sha256",
    }
)
# Activation 在校验后补充这些只读验证收据，仍保持封闭字段集。
REVIEW_INTEGRITY_ACTIVATION_FIELDS = frozenset(
    {
        "scoped_validation_schema",
        "scoped_validation_sha256",
        "scoped_reviewed_suite_path",
        "scoped_reviewed_suite_sha256",
        "scoped_reviewed_summary_path",
        "scoped_reviewed_summary_sha256",
        "scoped_reviewed_qa_path",
        "scoped_reviewed_qa_sha256",
        "scoped_reviewed_checksums_path",
        "scoped_reviewed_checksums_sha256",
        "authoritative_scope_receipts_sha256",
    }
)


class ScopedValidationError(RuntimeError):
    """表示 scoped reviewed 产物未满足 fail-closed 门禁。"""


def _promotion_authority_drift_paths(value: Any, *, path: str = "$") -> tuple[str, ...]:
    """递归定位与不可晋级契约冲突的授权字段。"""

    boolean_fields = {
        "canonical",
        "promotion_allowed",
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
    }
    forbidden_reference_fields = {
        "candidate_roots",
        "source_baseline_registry",
        "source_registry",
        "baselines",
    }
    changed: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if key in boolean_fields and nested is not False:
                changed.append(nested_path)
            if key in forbidden_reference_fields:
                changed.append(nested_path)
            changed.extend(_promotion_authority_drift_paths(nested, path=nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            changed.extend(
                _promotion_authority_drift_paths(nested, path=f"{path}[{index}]")
            )
    return tuple(changed)


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
        raise ScopedValidationError(f"{label} escapes its trusted root") from exc
    for component in (Path(), *relative.parents[::-1], relative):
        candidate = root_absolute if component == Path() else root_absolute / component
        if candidate.exists() or candidate.is_symlink():
            attributes = int(getattr(os.lstat(candidate), "st_file_attributes", 0))
            if candidate.is_symlink() or bool(attributes & 0x400):
                raise ScopedValidationError(
                    f"{label} contains a symlink or reparse point: {candidate}"
                )
    if root_absolute.exists():
        try:
            path_absolute.resolve(strict=False).relative_to(
                root_absolute.resolve(strict=True)
            )
        except ValueError as exc:
            raise ScopedValidationError(f"{label} resolves outside trusted root") from exc
    if require_exists and not path_absolute.exists():
        raise ScopedValidationError(f"{label} is missing: {path_absolute}")
    return path_absolute


def _sha256_file(path: Path, *, root: Path, label: str) -> str:
    safe_path = _assert_safe_path(path, root=root, label=label)
    if not safe_path.is_file():
        raise ScopedValidationError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    with safe_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    _assert_safe_path(safe_path, root=root, label=label)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    safe_path = _assert_safe_path(path, root=root, label=label)
    try:
        value = json.loads(safe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScopedValidationError(f"{label} is not valid JSON") from exc
    _assert_safe_path(safe_path, root=root, label=label)
    return value


def _validate_checksums(root: Path) -> tuple[str, dict[str, str]]:
    checksum_path = _assert_safe_path(
        root / CHECKSUM_FILENAME,
        root=root,
        label="scoped reviewed SHA256SUMS",
    )
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ScopedValidationError("scoped reviewed SHA256SUMS row is invalid") from exc
        candidate = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in relative
            or relative in rows
        ):
            raise ScopedValidationError("scoped reviewed SHA256SUMS row is unsafe")
        rows[relative] = digest
    expected_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != checksum_path
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    )
    expected_relatives = {
        path.relative_to(root).as_posix() for path in expected_files
    }
    if set(rows) != expected_relatives:
        raise ScopedValidationError(
            "scoped reviewed SHA256SUMS does not exactly cover reviewed artifacts"
        )
    for path in expected_files:
        relative = path.relative_to(root).as_posix()
        if _sha256_file(path, root=root, label=relative) != rows[relative]:
            raise ScopedValidationError(f"scoped reviewed checksum mismatch: {relative}")
    return (
        _sha256_file(checksum_path, root=root, label="scoped reviewed SHA256SUMS"),
        rows,
    )


def _require_exact_scope(value: Any, *, label: str) -> dict[str, Any]:
    expected = scope_profile()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ScopedValidationError(f"{label} does not match the frozen scope profile")
    return expected


def _build_current_authoritative_scope_receipts() -> dict[str, dict[str, Any]]:
    """从当前物理账本独立重算三类权威收据，不信任 reviewed 摘要缓存。"""

    import merge_easy_medium_reviewed as scoped_merge

    readiness = scoped_merge.scope_readiness()
    if (
        readiness.get("status") != "ready"
        or readiness.get("missing_case_count") != 0
    ):
        raise ScopedValidationError(
            "current easy/medium rereview ledger is not release-ready"
        )
    _, _, original = scoped_merge._load_scope_original_reviews()
    blocker = readiness.get("blocker")
    contract = readiness.get("contract")
    if not all(isinstance(value, Mapping) for value in (original, blocker, contract)):
        raise ScopedValidationError("current authoritative scope receipts are invalid")
    return {
        "original": dict(original),
        "blocker": dict(blocker),
        "contract": dict(contract),
    }


def _build_current_case_lineage_authority() -> dict[str, tuple[str, str]]:
    """从冻结150题候选清单投影出E/M逐题lineage权威表。"""

    rows = load_frozen_candidate_manifest_scope_authority(ROOT)
    authority = {
        str(row["case_id"]): (
            str(row["parent_case_id"]),
            str(row["difficulty"]),
        )
        for row in rows
        if str(row.get("difficulty") or "") in SCOPE_DIFFICULTIES
    }
    if len(authority) != EXPECTED_CASE_COUNT:
        raise ScopedValidationError("frozen E/M case lineage authority is incomplete")
    return authority


def _build_frozen_candidate_source_authority() -> dict[str, dict[str, Any]]:
    """从代码内SHA锚定的候选manifest读取E/M source/index身份。"""

    # 先执行完整冻结文件、suite 与 QA 哈希校验，再读取同一批已锚定文件。
    try:
        load_frozen_candidate_manifest_scope_authority(ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScopedValidationError(
            f"frozen candidate source authority could not be loaded: {exc}"
        ) from exc
    authority: dict[str, dict[str, Any]] = {}
    for difficulty in SCOPE_DIFFICULTIES:
        manifest = _read_json(
            ROOT / "datasets" / f"{difficulty}.v2.json",
            root=ROOT,
            label=f"frozen {difficulty} candidate manifest",
        )
        source = manifest.get("source") if isinstance(manifest, Mapping) else None
        if not isinstance(source, Mapping):
            raise ScopedValidationError(
                f"frozen {difficulty} candidate source identity is missing"
            )
        authority[difficulty] = {
            "relative_path": source.get("relative_path"),
            "content_sha256": source.get("content_sha256"),
            "page_count": source.get("page_count"),
            "kb_id": source.get("kb_id"),
            "document_id": source.get("document_id"),
            "document_version_id": source.get("document_version_id"),
            "index_version_id": source.get("index_version_id"),
        }
    common_fields = (
        "relative_path",
        "content_sha256",
        "page_count",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
    )
    if any(
        authority["easy"].get(field) != authority["medium"].get(field)
        for field in common_fields
    ):
        raise ScopedValidationError("frozen E/M candidate source identity drifted")
    return authority


def _build_current_case_semantic_authority() -> dict[str, dict[str, Any]]:
    """从冻结候选与当前blocker账本重建逐题语义、证据和branch权威。"""

    import merge_easy_medium_reviewed as scoped_merge

    blockers, _, _ = scoped_merge._load_scope_rereviews()
    authority: dict[str, dict[str, Any]] = {}
    for difficulty in SCOPE_DIFFICULTIES:
        manifest = json.loads(
            (ROOT / "datasets" / f"{difficulty}.v2.json").read_text(encoding="utf-8")
        )
        cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
        if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT_PER_DIFFICULTY:
            raise ScopedValidationError(
                f"frozen {difficulty} semantic authority is incomplete"
            )
        for candidate in cases:
            metadata = candidate.get("metadata") if isinstance(candidate, Mapping) else None
            parent_case_id = (
                str(metadata.get("parent_case_id") or "")
                if isinstance(metadata, Mapping)
                else ""
            )
            active = blockers.get(parent_case_id, (candidate, None))[0]
            active_metadata = active.get("metadata") if isinstance(active, Mapping) else None
            case_id = str(active.get("case_id") or "") if isinstance(active, Mapping) else ""
            contract = active.get("answer_contract") if isinstance(active, Mapping) else None
            if (
                not case_id
                or case_id in authority
                or not isinstance(active_metadata, Mapping)
                or active_metadata.get("parent_case_id") != parent_case_id
                or active_metadata.get("difficulty") != difficulty
                or not isinstance(contract, Mapping)
            ):
                raise ScopedValidationError(
                    f"current {difficulty} semantic authority lineage drifted"
                )
            authority[case_id] = {
                "query": active.get("query"),
                "reference_answer": active.get("reference_answer"),
                "unanswerable": active.get("unanswerable"),
                "expected_terms": active.get("expected_terms"),
                "forbidden_terms": active.get("forbidden_terms"),
                "relevant_evidence": active.get("relevant_evidence"),
                "required_branch_spans": [
                    str(branch.get("query_span") or "")
                    for branch in contract.get("required_branches") or []
                ],
            }
    if len(authority) != EXPECTED_CASE_COUNT:
        raise ScopedValidationError("current E/M semantic authority is incomplete")
    return authority


def _build_current_case_review_authority() -> dict[str, dict[str, Any]]:
    """从物理账本重建逐题最终 contract 与被采用响应的权威谱系。"""

    import merge_easy_medium_reviewed as scoped_merge

    try:
        projected = scoped_merge.build_current_case_review_authority()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScopedValidationError(
            f"current case review authority could not be rebuilt: {exc}"
        ) from exc
    if not isinstance(projected, Mapping) or len(projected) != EXPECTED_CASE_COUNT:
        raise ScopedValidationError("current E/M case review authority is incomplete")
    authority: dict[str, dict[str, Any]] = {}
    expected_row_fields = {
        "parent_case_id",
        "difficulty",
        "answer_contract",
        "selected_response",
    }
    expected_response_fields = {
        "source",
        "custom_id",
        "row_sha256",
        "review_sha256",
        "prompt_sha256",
        "runner_version",
        "quote_key_recovery",
    }
    for raw_case_id, raw_row in projected.items():
        case_id = str(raw_case_id or "")
        if (
            not case_id
            or case_id in authority
            or not isinstance(raw_row, Mapping)
            or set(raw_row) != expected_row_fields
        ):
            raise ScopedValidationError("current E/M case review authority is malformed")
        row = dict(raw_row)
        selected = row.get("selected_response")
        contract = row.get("answer_contract")
        if (
            not isinstance(contract, Mapping)
            or not isinstance(selected, Mapping)
            or set(selected) != expected_response_fields
            or str(row.get("parent_case_id") or "") == ""
            or str(row.get("difficulty") or "") not in SCOPE_DIFFICULTIES
            or str(selected.get("source") or "") not in EXPECTED_RECEIPT_KINDS
            or str(selected.get("custom_id") or "") == ""
        ):
            raise ScopedValidationError(
                f"current case review authority is malformed: {case_id}"
            )
        authority[case_id] = row
    return authority


def _validate_case_review_authority_receipt_binding(
    authority: Mapping[str, Mapping[str, Any]],
    *,
    receipts: Mapping[str, Mapping[str, Any]],
) -> None:
    """证明逐题投影采用的响应就是当前三类账本收据中的物理行。"""

    for case_id, row in authority.items():
        selected = row.get("selected_response")
        if not isinstance(selected, Mapping):
            raise ScopedValidationError(
                f"current selected response provenance is malformed: {case_id}"
            )
        source = str(selected.get("source") or "")
        custom_id = str(selected.get("custom_id") or "")
        row_sha256 = str(selected.get("row_sha256") or "")
        receipt = receipts.get(source)
        receipt_rows = (
            receipt.get("selected_response_row_sha256_by_id")
            if isinstance(receipt, Mapping)
            else None
        )
        if (
            not isinstance(receipt_rows, Mapping)
            or receipt_rows.get(custom_id) != row_sha256
        ):
            raise ScopedValidationError(
                f"current case review authority is not bound to its ledger receipt: {case_id}"
            )


def _validate_receipt_self_hash(receipt: Any, *, kind: str) -> dict[str, Any]:
    """校验收据种类、自散列和目标ID集合的内部一致性。"""

    if not isinstance(receipt, Mapping):
        raise ScopedValidationError(f"scoped {kind} authoritative receipt is required")
    normalized = dict(receipt)
    claimed = normalized.pop("receipt_sha256", None)
    target_ids = normalized.get("scope_target_custom_ids")
    selected = normalized.get("selected_response_row_sha256_by_id")
    if (
        normalized.get("receipt_kind") != kind
        or normalized.get("scope_id") != SCOPE_ID
        or not isinstance(claimed, str)
        or claimed != _sha256_json(normalized)
        or not isinstance(target_ids, list)
        or len(target_ids) != len(set(str(value) for value in target_ids))
        or not isinstance(selected, Mapping)
        or set(str(value) for value in target_ids) != set(selected)
        or normalized.get("selected_case_count") != len(target_ids)
        or normalized.get("authoritative_success_count") != len(target_ids)
        or normalized.get("missing_case_count") != 0
    ):
        raise ScopedValidationError(
            f"scoped {kind} authoritative receipt self-hash/selected IDs drifted"
        )
    combined = normalized.get("nonthinking_combined_receipt")
    sidecar_sha256 = normalized.get("nonthinking_sidecar_sha256")
    if (combined is None) is not (sidecar_sha256 is None):
        raise ScopedValidationError(
            f"scoped {kind} nonthinking receipt/sidecar binding is incomplete"
        )
    if combined is not None:
        import easy_medium_nonthinking_profile as nonthinking_profile

        if not isinstance(combined, Mapping) or not isinstance(sidecar_sha256, Mapping):
            raise ScopedValidationError(
                f"scoped {kind} nonthinking receipt is malformed"
            )
        combined_value = dict(combined)
        combined_claimed = combined_value.pop("receipt_sha256", None)
        expected_combined_keys = {
            "execution_profile_id",
            "execution_profile_sha256",
            "legacy_execution_contract_sha256",
            "legacy_success_count",
            "predecessor_success_count",
            "predecessor_bridge_receipt_sha256",
            "v4_adoption_success_count",
            "v4_adopted_success_custom_ids",
            "v4_adoption_bridge_receipt_sha256",
            "v4_source_calibration_evidence_receipt",
            "v5_adoption_success_count",
            "v5_adopted_success_custom_ids",
            "v5_adoption_bridge_receipt_sha256",
            "v5_source_calibration_evidence_receipt",
            "materialization_policy",
            "new_profile_success_count",
            "fresh_calibration_receipt",
            "authoritative_success_count",
            "selected_response_sha256_by_id",
            "hard_rows_counted",
            "hard_usage_counted",
        }
        new_count = combined_value.get("new_profile_success_count")
        predecessor_count = combined_value.get("predecessor_success_count")
        predecessor_receipt = combined_value.get(
            "predecessor_bridge_receipt_sha256"
        )
        calibration = combined_value.get("fresh_calibration_receipt")
        adopted_count = combined_value.get("v4_adoption_success_count")
        adopted_ids = combined_value.get("v4_adopted_success_custom_ids")
        adopted_receipt = combined_value.get(
            "v4_adoption_bridge_receipt_sha256"
        )
        source_calibration_evidence = combined_value.get(
            "v4_source_calibration_evidence_receipt"
        )
        v5_adopted_count = combined_value.get("v5_adoption_success_count")
        v5_adopted_ids = combined_value.get("v5_adopted_success_custom_ids")
        v5_adopted_receipt = combined_value.get(
            "v5_adoption_bridge_receipt_sha256"
        )
        v5_source_calibration_evidence = combined_value.get(
            "v5_source_calibration_evidence_receipt"
        )
        if (
            set(combined_value) != expected_combined_keys
            or combined_claimed != _sha256_json(combined_value)
            or combined_value.get("execution_profile_id")
            != nonthinking_profile.PROFILE_ID
            or combined_value.get("legacy_execution_contract_sha256")
            != normalized.get("execution_contract_sha256")
            or type(new_count) is not int
            or new_count < 0
            or type(predecessor_count) is not int
            or predecessor_count < 0
            or type(adopted_count) is not int
            or adopted_count < 0
            or not isinstance(adopted_ids, list)
            or len(adopted_ids) != adopted_count
            or len(set(adopted_ids)) != adopted_count
            or not set(adopted_ids).issubset(set(selected))
            or type(v5_adopted_count) is not int
            or v5_adopted_count < 0
            or not isinstance(v5_adopted_ids, list)
            or len(v5_adopted_ids) != v5_adopted_count
            or len(set(v5_adopted_ids)) != v5_adopted_count
            or not set(v5_adopted_ids).issubset(set(selected))
            or set(adopted_ids) & set(v5_adopted_ids)
            or combined_value.get("materialization_policy")
            != {
                "policy_version": nonthinking_profile.MATERIALIZATION_POLICY_VERSION,
                "production_matcher": nonthinking_profile.PRODUCTION_MATCHER,
                "production_matcher_policy_version": (
                    nonthinking_profile.PRODUCTION_MATCHER_POLICY_VERSION
                ),
                "subset_solver_policy": dict(
                    nonthinking_profile.SUBSET_SOLVER_POLICY
                ),
            }
            or (
                predecessor_count > 0
                and (
                    not isinstance(predecessor_receipt, str)
                    or len(predecessor_receipt) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in predecessor_receipt
                    )
                )
            )
            or combined_value.get("authoritative_success_count")
            != normalized.get("authoritative_success_count")
            or combined_value.get("selected_response_sha256_by_id") != selected
            or combined_value.get("hard_rows_counted") != 0
            or combined_value.get("hard_usage_counted") != 0
            or set(sidecar_sha256)
            != {
                "execution-profile.json",
                "journal.jsonl",
                "responses.jsonl",
                "calibration-responses.jsonl",
            }
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in sidecar_sha256.values()
            )
        ):
            raise ScopedValidationError(
                f"scoped {kind} nonthinking combined receipt drifted"
            )
        if (
            not isinstance(adopted_receipt, str)
            or len(adopted_receipt) != 64
            or not isinstance(source_calibration_evidence, Mapping)
        ):
            raise ScopedValidationError(
                f"scoped {kind} v4 adoption lineage is incomplete"
            )
        if isinstance(source_calibration_evidence, Mapping):
            evidence_value = dict(source_calibration_evidence)
            evidence_claimed = evidence_value.pop("receipt_sha256", None)
            source_result_hashes = evidence_value.get(
                "source_result_envelope_sha256_by_id"
            )
            if (
                set(evidence_value)
                != {
                    "evidence_role",
                    "source_profile_sha256",
                    "source_result_envelope_sha256_by_id",
                    "authorizes_v5_calls",
                    "bridge_receipt_sha256",
                }
                or evidence_claimed != _sha256_json(evidence_value)
                or evidence_value.get("evidence_role")
                != "source_execution_precondition_only"
                or evidence_value.get("authorizes_v5_calls") is not False
                or evidence_value.get("source_profile_sha256")
                != nonthinking_profile.V4_PROFILE_SHA256
                or evidence_value.get("bridge_receipt_sha256") != adopted_receipt
                or not isinstance(source_result_hashes, Mapping)
                or len(source_result_hashes) != 4
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef" for character in value
                    )
                    for value in source_result_hashes.values()
                )
            ):
                raise ScopedValidationError(
                    f"scoped {kind} v4 source calibration evidence drifted"
                )
        if (
            not isinstance(v5_adopted_receipt, str)
            or len(v5_adopted_receipt) != 64
            or any(
                character not in "0123456789abcdef"
                for character in v5_adopted_receipt
            )
            or not isinstance(v5_source_calibration_evidence, Mapping)
        ):
            raise ScopedValidationError(
                f"scoped {kind} v5 adoption lineage is incomplete"
            )
        if isinstance(v5_source_calibration_evidence, Mapping):
            evidence_value = dict(v5_source_calibration_evidence)
            evidence_claimed = evidence_value.pop("receipt_sha256", None)
            source_result_hashes = evidence_value.get(
                "source_result_envelope_sha256_by_id"
            )
            if (
                set(evidence_value)
                != {
                    "evidence_role",
                    "source_profile_sha256",
                    "source_result_envelope_sha256_by_id",
                    "authorizes_v6_calls",
                    "bridge_receipt_sha256",
                }
                or evidence_claimed != _sha256_json(evidence_value)
                or evidence_value.get("evidence_role")
                != "source_execution_precondition_only"
                or evidence_value.get("authorizes_v6_calls") is not False
                or evidence_value.get("source_profile_sha256")
                != nonthinking_profile.V5_PROFILE_SHA256
                or evidence_value.get("bridge_receipt_sha256")
                != v5_adopted_receipt
                or not isinstance(source_result_hashes, Mapping)
                or set(source_result_hashes)
                != nonthinking_profile._expected_v5_calibration_ids()
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in source_result_hashes.values()
                )
            ):
                raise ScopedValidationError(
                    f"scoped {kind} v5 source calibration evidence drifted"
                )
        import run_easy_medium_nonthinking as nonthinking_runner

        physical_snapshot = nonthinking_runner.load_release_snapshot()
        if (
            physical_snapshot is None
            or physical_snapshot["combined_receipt_by_kind"].get(kind) != combined
            or physical_snapshot.get("sidecar_sha256") != sidecar_sha256
        ):
            raise ScopedValidationError(
                f"scoped {kind} nonthinking receipt does not replay physical sidecars"
            )
        if new_count > 0:
            if not isinstance(calibration, Mapping):
                raise ScopedValidationError(
                    f"scoped {kind} new nonthinking results lack calibration 4/4"
                )
            calibration_value = dict(calibration)
            calibration_claimed = calibration_value.pop("receipt_sha256", None)
            calibration_ids = calibration_value.get("succeeded_custom_ids")
            calibration_hashes = calibration_value.get(
                "result_envelope_sha256_by_id"
            )
            if (
                set(calibration_value)
                != {
                    "execution_profile_sha256",
                    "required_case_count",
                    "succeeded_custom_ids",
                    "result_envelope_sha256_by_id",
                }
                or calibration_claimed != _sha256_json(calibration_value)
                or calibration_value.get("execution_profile_sha256")
                != combined_value.get("execution_profile_sha256")
                or calibration_value.get("required_case_count") != 4
                or not isinstance(calibration_ids, list)
                or len(calibration_ids) != 4
                or len(set(calibration_ids)) != 4
                or not isinstance(calibration_hashes, Mapping)
                or set(calibration_hashes) != set(calibration_ids)
            ):
                raise ScopedValidationError(
                    f"scoped {kind} fresh calibration receipt drifted"
                )
        elif calibration is not None:
            raise ScopedValidationError(
                f"scoped {kind} calibration receipt exists without new successes"
            )
    return dict(receipt)


def _project_file_from_receipt(relative: Any, *, label: str) -> Path:
    text = str(relative or "")
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts or "\\" in text:
        raise ScopedValidationError(f"{label} path is unsafe")
    return _assert_safe_path(PROJECT_ROOT / candidate, root=PROJECT_ROOT, label=label)


def _validate_source_and_index_integrity(
    value: Any,
    *,
    summary: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    frozen_source_authority: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """交叉绑定源文件、源QA、冻结索引、逐难度manifest与复审收据。"""

    if not isinstance(value, Mapping):
        raise ScopedValidationError("source_and_index integrity receipt is required")
    integrity = dict(value)
    source = integrity.get("source")
    active_index = integrity.get("active_index")
    review = integrity.get("review")
    if not all(isinstance(item, Mapping) for item in (source, active_index, review)):
        raise ScopedValidationError("source_and_index integrity receipt is malformed")
    source = dict(source)
    active_index = dict(active_index)
    review = dict(review)
    if (
        set(integrity) != SOURCE_AND_INDEX_INTEGRITY_FIELDS
        or set(source) != SOURCE_INTEGRITY_FIELDS
        or set(active_index) != ACTIVE_INDEX_INTEGRITY_FIELDS
        or set(review) != REVIEW_INTEGRITY_REQUIRED_FIELDS
    ):
        raise ScopedValidationError("source_and_index integrity schema drifted")
    source_path = _project_file_from_receipt(source.get("path"), label="frozen source")
    source_summary_path = _project_file_from_receipt(
        active_index.get("source_qa_summary_path"),
        label="source QA summary",
    )
    blocker_validation_path = _project_file_from_receipt(
        active_index.get("blocker_validation_path"),
        label="blocker validation",
    )
    if (
        _sha256_file(source_path, root=PROJECT_ROOT, label="frozen source")
        != source.get("sha256")
        or _sha256_file(
            source_summary_path,
            root=PROJECT_ROOT,
            label="source QA summary",
        )
        != active_index.get("source_qa_summary_sha256")
        or _sha256_file(
            blocker_validation_path,
            root=PROJECT_ROOT,
            label="blocker validation",
        )
        != active_index.get("blocker_validation_sha256")
    ):
        raise ScopedValidationError("source_and_index file SHA receipt drifted")
    source_summary = _read_json(
        source_summary_path,
        root=PROJECT_ROOT,
        label="source QA summary",
    )
    blocker_validation = _read_json(
        blocker_validation_path,
        root=PROJECT_ROOT,
        label="blocker validation",
    )
    checks = source_summary.get("input_checks") if isinstance(source_summary, Mapping) else None
    blocker_index = (
        blocker_validation.get("active_index")
        if isinstance(blocker_validation, Mapping)
        else None
    )
    index_identity_fields = (
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "frozen_chunk_count",
    )
    blocker_index_fields = ("index_version_id", "frozen_chunk_count")
    if (
        not isinstance(checks, Mapping)
        or not isinstance(blocker_index, Mapping)
        or checks.get("source_path") != source.get("path")
        or checks.get("source_sha256") != source.get("sha256")
        or checks.get("source_pdf_page_count") != source.get("page_count")
        or checks.get("source_hash_matches") is not True
        or checks.get("source_page_count_matches") is not True
        or checks.get("joined_case_count") != 150
        or checks.get("indexed_evidence_count") != active_index.get("frozen_chunk_count")
        or checks.get("index_version_id") != active_index.get("index_version_id")
        or blocker_validation.get("status") != "passed"
        or blocker_validation.get("selected_chunks_resolve_to_frozen_active_index")
        is not True
        or any(
            blocker_index.get(field) != active_index.get(field)
            for field in blocker_index_fields
        )
        or blocker_index.get("index_rebuild_required") is not False
        or blocker_index.get("evidence_refresh_required") is not False
    ):
        raise ScopedValidationError("source_and_index semantic receipt drifted")
    expected_manifest_source = {
        "relative_path": source.get("path"),
        "content_sha256": source.get("sha256"),
        "page_count": source.get("page_count"),
        **{field: active_index.get(field) for field in index_identity_fields[:-1]},
    }
    if any(
        frozen_source_authority.get(difficulty) != expected_manifest_source
        for difficulty in SCOPE_DIFFICULTIES
    ):
        raise ScopedValidationError(
            "source_and_index identity drifted from frozen candidate manifests"
        )
    if any(
        not isinstance(manifest.get("source"), Mapping)
        or set(manifest["source"]) != REVIEWED_SOURCE_ALLOWED_FIELDS
        or any(
            manifest["source"].get(key) != expected
            for key, expected in expected_manifest_source.items()
        )
        for manifest in manifests.values()
    ):
        raise ScopedValidationError("dataset source_and_index identity drifted")
    if (
        summary.get("source_and_index_integrity_sha256") != _sha256_json(integrity)
        or review.get("scope_id") != SCOPE_ID
        or review.get("scoped_case_count") != EXPECTED_CASE_COUNT
        or review.get("hard_reference_only") is not True
        or review.get("hard_imported_case_count") != 0
        or review.get("original_scope_receipt_sha256")
        != receipts["original"].get("receipt_sha256")
        or review.get("blocker_scope_receipt_sha256")
        != receipts["blocker"].get("receipt_sha256")
        or review.get("contract_scope_receipt_sha256")
        != receipts["contract"].get("receipt_sha256")
    ):
        raise ScopedValidationError("source_and_index review receipt cross-binding drifted")
    return integrity


def _validate_manifest(
    manifest: Any,
    *,
    difficulty: str,
    semantic_authority: Mapping[str, Mapping[str, Any]],
    review_authority: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    if not isinstance(manifest, Mapping):
        raise ScopedValidationError(f"{difficulty} reviewed manifest must be an object")
    unexpected = set(manifest) - REVIEWED_MANIFEST_ALLOWED_TOP_LEVEL_FIELDS
    if unexpected:
        raise ScopedValidationError(
            f"{difficulty} reviewed manifest has forbidden top-level fields: "
            + ", ".join(sorted(unexpected))
        )
    authority_drift = _promotion_authority_drift_paths(manifest)
    if authority_drift:
        raise ScopedValidationError(
            f"{difficulty} reviewed authorization semantics drifted: "
            + ", ".join(authority_drift)
        )
    expected_identity = {
        "schema_version": "stt.eval.dataset.v2-llm-reviewed-candidate",
        "suite_key": SUITE_KEY,
        "dataset_key": DATASET_KEYS[difficulty],
        "version": VERSION,
        "difficulty": difficulty,
        "status": "reviewed_candidate",
        "import_allowed": False,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ScopedValidationError(f"{difficulty} reviewed manifest identity drifted")
    _require_exact_scope(manifest.get("scope"), label=f"{difficulty} manifest scope")
    gate_policy = manifest.get("gate_policy")
    generation = manifest.get("generation")
    if (
        not isinstance(gate_policy, Mapping)
        or gate_policy.get("case_pass_policy") != FROZEN_CASE_PASS_POLICY
        or gate_policy.get("case_pass_policy_activation_status")
        != "reserved_not_active"
        or gate_policy.get("case_pass_policy_materialization_required") is not True
        or not isinstance(generation, Mapping)
        or generation.get("llm_review_status") != "complete"
        or generation.get("independent_validation_status") != "pending"
        or generation.get("answer_contract_to_concept_groups_materialization_status")
        != "complete"
    ):
        raise ScopedValidationError(f"{difficulty} reviewed policy/generation drifted")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT_PER_DIFFICULTY:
        raise ScopedValidationError(f"{difficulty} must contain exactly 50 reviewed cases")
    case_ids: list[str] = []
    parent_ids: list[str] = []
    cases_by_id: dict[str, Mapping[str, Any]] = {}
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise ScopedValidationError(f"{difficulty} case {position} is invalid")
        case_id = str(case.get("case_id") or "")
        metadata = case.get("metadata")
        parent_case_id = (
            str(metadata.get("parent_case_id") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        contract = case.get("answer_contract")
        review = contract.get("review") if isinstance(contract, Mapping) else None
        groups = case.get("concept_groups")
        all_of = groups.get("all_of") if isinstance(groups, Mapping) else None
        if not case_id or not parent_case_id or case_id in cases_by_id:
            raise ScopedValidationError(f"{difficulty} case lineage is incomplete")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("difficulty") != difficulty
            or metadata.get("easy_medium_scope_id") != SCOPE_ID
        ):
            raise ScopedValidationError(
                f"{difficulty} case {case_id} metadata difficulty/scope drifted"
            )
        semantic = semantic_authority.get(case_id)
        if not isinstance(semantic, Mapping) or any(
            case.get(field) != semantic.get(field)
            for field in (
                "query",
                "reference_answer",
                "unanswerable",
                "expected_terms",
                "forbidden_terms",
                "relevant_evidence",
            )
        ):
            raise ScopedValidationError(
                f"{difficulty} case {case_id} semantic/evidence authority drifted"
            )
        if (
            not isinstance(contract, Mapping)
            or contract.get("contract_status") != "llm_reviewed_candidate"
            or not isinstance(review, Mapping)
            or review.get("verdict") not in {"approve", "revise"}
            or not isinstance(all_of, list)
            or not all_of
        ):
            raise ScopedValidationError(f"{difficulty} case {case_id} is not reviewed-ready")
        if any(
            not isinstance(group, Mapping)
            or not isinstance(group.get("any_of"), list)
            or not group["any_of"]
            for group in all_of
        ):
            raise ScopedValidationError(f"{difficulty} case {case_id} concept groups are invalid")
        schema_errors = list(REVIEWED_CONTRACT_VALIDATOR.iter_errors(contract))
        if schema_errors:
            raise ScopedValidationError(
                f"{difficulty} case {case_id} reviewed contract schema failure: "
                f"{schema_errors[0].message}"
            )
        evidence = case.get("relevant_evidence")
        if not isinstance(evidence, list):
            raise ScopedValidationError(
                f"{difficulty} case {case_id} relevant evidence is invalid"
            )
        try:
            full._validate_contract_coverage(
                dict(contract),
                evidence,
                case_id=case_id,
                expected_branch_spans=list(semantic["required_branch_spans"]),
            )
            expected_groups, expected_variant_audit = full._materialize_concept_groups(
                dict(contract),
                case_id=case_id,
                evidence=evidence,
            )
        except (RuntimeError, ValueError) as exc:
            raise ScopedValidationError(
                f"{difficulty} case {case_id} contract coverage/materializer failed: {exc}"
            ) from exc
        contract_hash = _sha256_json(contract)
        groups_hash = _sha256_json(groups)
        if groups != expected_groups:
            raise ScopedValidationError(
                f"{difficulty} case {case_id} concept_groups materialization drifted"
            )
        if (
            metadata.get("reviewed_contract_hash") != contract_hash
            or metadata.get("concept_groups_hash") != groups_hash
            or metadata.get("concept_groups_source_contract_hash") != contract_hash
            or metadata.get("concept_groups_materializer")
            != "merge_reviewed_contracts.py:v1"
            or metadata.get("concept_groups_variant_filter") != expected_variant_audit
        ):
            raise ScopedValidationError(
                f"{difficulty} case {case_id} metadata contract/groups hash drifted"
            )
        authoritative_review = review_authority.get(case_id)
        if (
            not isinstance(authoritative_review, Mapping)
            or contract != authoritative_review.get("answer_contract")
        ):
            raise ScopedValidationError(
                f"{difficulty} case {case_id} final contract drifted from current review authority"
            )
        case_ids.append(case_id)
        parent_ids.append(parent_case_id)
        cases_by_id[case_id] = case
    if len(case_ids) != len(set(case_ids)) or len(parent_ids) != len(set(parent_ids)):
        raise ScopedValidationError(f"{difficulty} reviewed lineage contains duplicates")
    return case_ids, parent_ids, cases_by_id


def validate(root: Path = REVIEWED_ROOT) -> dict[str, Any]:
    """验证 scoped reviewed 产物，并返回可嵌入 activation 的确定性收据。"""

    root = _assert_safe_path(root, root=root, label="scoped reviewed root")
    checksum_sha256, checksum_rows = _validate_checksums(root)
    suite_path = root / SUITE_FILENAME
    summary_path = root / SUMMARY_FILENAME
    qa_path = root / QA_FILENAME
    suite = _read_json(suite_path, root=root, label="scoped reviewed suite")
    summary = _read_json(summary_path, root=root, label="scoped reviewed summary")
    qa_rows = _read_json(qa_path, root=root, label="scoped reviewed QA")
    if not isinstance(suite, Mapping) or not isinstance(summary, Mapping):
        raise ScopedValidationError("scoped reviewed suite/summary must be objects")
    unexpected_suite = set(suite) - REVIEWED_SUITE_ALLOWED_TOP_LEVEL_FIELDS
    unexpected_summary = set(summary) - REVIEWED_SUMMARY_ALLOWED_TOP_LEVEL_FIELDS
    if unexpected_suite:
        raise ScopedValidationError(
            "scoped reviewed suite has forbidden top-level fields: "
            + ", ".join(sorted(unexpected_suite))
        )
    if unexpected_summary:
        raise ScopedValidationError(
            "scoped reviewed summary has forbidden top-level fields: "
            + ", ".join(sorted(unexpected_summary))
        )
    authority_drift = {
        *_promotion_authority_drift_paths(suite),
        *_promotion_authority_drift_paths(summary),
    }
    if authority_drift:
        raise ScopedValidationError(
            "scoped reviewed authorization semantics drifted: "
            + ", ".join(sorted(authority_drift))
        )
    expected_suite_identity = {
        "schema_version": SUITE_SCHEMA,
        "suite_key": SUITE_KEY,
        "version": VERSION,
        "status": "reviewed_candidate_pending_independent_validation",
    }
    if any(suite.get(key) != value for key, value in expected_suite_identity.items()):
        raise ScopedValidationError("scoped reviewed suite identity drifted")
    expected_summary_identity = {
        "schema_version": SUMMARY_SCHEMA,
        "suite_key": SUITE_KEY,
        "status": "reviewed_candidate_pending_independent_validation",
    }
    if any(summary.get(key) != value for key, value in expected_summary_identity.items()):
        raise ScopedValidationError("scoped reviewed summary identity drifted")
    _require_exact_scope(suite.get("scope"), label="reviewed suite scope")
    _require_exact_scope(summary.get("scope"), label="reviewed summary scope")
    expected_state = {
        "import_allowed": False,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
    }
    if any(suite.get(key) != value for key, value in expected_state.items()):
        raise ScopedValidationError("scoped reviewed suite permissions drifted")
    entries = suite.get("datasets")
    if not isinstance(entries, Mapping) or tuple(entries) != tuple(SCOPE_DIFFICULTIES):
        raise ScopedValidationError("scoped reviewed suite must contain only easy and medium")

    all_case_ids: list[str] = []
    all_parent_ids: list[str] = []
    expected_lineage_by_case: dict[str, tuple[str, str]] = {}
    reviewed_cases_by_id: dict[str, Mapping[str, Any]] = {}
    reviewed_manifests: dict[str, Mapping[str, Any]] = {}
    dataset_receipts: dict[str, dict[str, Any]] = {}
    current_semantic_authority = _build_current_case_semantic_authority()
    current_review_authority = _build_current_case_review_authority()
    frozen_source_authority = _build_frozen_candidate_source_authority()
    if set(current_review_authority) != set(current_semantic_authority):
        raise ScopedValidationError(
            "current semantic and review authority case coverage drifted"
        )
    for difficulty in SCOPE_DIFFICULTIES:
        relative = f"datasets/{difficulty}.reviewed.json"
        entry = entries.get(difficulty)
        if not isinstance(entry, Mapping) or entry.get("path") != relative:
            raise ScopedValidationError(f"{difficulty} reviewed dataset receipt is invalid")
        path = root / relative
        manifest = _read_json(path, root=root, label=f"{difficulty} reviewed dataset")
        case_ids, parent_ids, cases_by_id = _validate_manifest(
            manifest,
            difficulty=difficulty,
            semantic_authority=current_semantic_authority,
            review_authority=current_review_authority,
        )
        file_sha256 = _sha256_file(path, root=root, label=f"{difficulty} reviewed dataset")
        manifest_sha256 = _sha256_json(manifest)
        expected_entry = {
            "path": relative,
            "sha256": file_sha256,
            "manifest_sha256": manifest_sha256,
            "dataset_key": DATASET_KEYS[difficulty],
            "version": VERSION,
            "case_count": EXPECTED_CASE_COUNT_PER_DIFFICULTY,
        }
        if dict(entry) != expected_entry:
            raise ScopedValidationError(f"{difficulty} reviewed dataset receipt drifted")
        dataset_receipts[difficulty] = expected_entry
        reviewed_manifests[difficulty] = manifest
        reviewed_cases_by_id.update(cases_by_id)
        all_case_ids.extend(case_ids)
        all_parent_ids.extend(parent_ids)
        expected_lineage_by_case.update(
            {
                case_id: (parent_case_id, difficulty)
                for case_id, parent_case_id in zip(case_ids, parent_ids, strict=True)
            }
        )

    if (
        len(all_case_ids) != EXPECTED_CASE_COUNT
        or len(set(all_case_ids)) != EXPECTED_CASE_COUNT
        or len(all_parent_ids) != EXPECTED_CASE_COUNT
        or len(set(all_parent_ids)) != EXPECTED_CASE_COUNT
    ):
        raise ScopedValidationError("scoped reviewed datasets must cover 100 unique lineages")
    current_lineage_authority = _build_current_case_lineage_authority()
    if expected_lineage_by_case != current_lineage_authority:
        raise ScopedValidationError(
            "scoped reviewed case metadata lineage drifted from frozen authority"
        )
    if set(current_review_authority) != set(expected_lineage_by_case):
        raise ScopedValidationError(
            "scoped reviewed case coverage drifted from current review authority"
        )
    if not isinstance(qa_rows, list) or len(qa_rows) != EXPECTED_CASE_COUNT:
        raise ScopedValidationError("scoped reviewed QA must contain exactly 100 rows")
    qa_by_case: dict[str, Mapping[str, Any]] = {}
    for row in qa_rows:
        if not isinstance(row, Mapping):
            raise ScopedValidationError("scoped reviewed QA contains a non-object row")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in qa_by_case:
            raise ScopedValidationError("scoped reviewed QA case lineage is missing or duplicate")
        qa_by_case[case_id] = row
    if set(qa_by_case) != set(expected_lineage_by_case):
        raise ScopedValidationError("scoped reviewed QA case coverage drifted")
    for case_id, (parent_case_id, difficulty) in expected_lineage_by_case.items():
        row = qa_by_case[case_id]
        case = reviewed_cases_by_id[case_id]
        contract_hash = _sha256_json(case.get("answer_contract"))
        groups = case.get("concept_groups")
        groups_hash = _sha256_json(groups)
        authoritative_review = current_review_authority[case_id]
        selected = authoritative_review.get("selected_response")
        if (
            str(row.get("parent_case_id") or "") != parent_case_id
            or str(row.get("difficulty") or "") != difficulty
            or str(row.get("verdict") or "") not in {"approve", "revise"}
            or row.get("scope_id") != SCOPE_ID
            or row.get("answer_contract_hash") != contract_hash
            or row.get("concept_groups_hash") != groups_hash
            or row.get("concept_groups_source_contract_hash") != contract_hash
            or row.get("concept_group_count")
            != len(groups.get("all_of") if isinstance(groups, Mapping) else [])
            or not isinstance(selected, Mapping)
            or row.get("custom_id") != selected.get("custom_id")
            or row.get("prompt_sha256") != selected.get("prompt_sha256")
            or row.get("runner_version") != selected.get("runner_version")
            or row.get("response_sha256") != selected.get("review_sha256")
            or row.get("selected_response_source") != selected.get("source")
            or row.get("selected_response_sha256") != selected.get("row_sha256")
            or row.get("saved_response_quote_key_branch_span_recovery")
            != selected.get("quote_key_recovery")
        ):
            raise ScopedValidationError(
                "scoped reviewed QA lineage/verdict drifted or hash mismatched/"
                f"provenance drifted: {case_id}"
            )

    expected_summary = {
        "case_count": EXPECTED_CASE_COUNT,
        "blocked_case_count": 0,
        "difficulty_case_counts": {
            difficulty: EXPECTED_CASE_COUNT_PER_DIFFICULTY
            for difficulty in SCOPE_DIFFICULTIES
        },
        "blocker_rereview_counts": EXPECTED_BLOCKER_REREVIEW_COUNTS,
        "contract_rereview_counts": EXPECTED_CONTRACT_REREVIEW_COUNTS,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "scope_id": SCOPE_ID,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ScopedValidationError("scoped reviewed summary counts/policy drifted")
    selected_receipts = summary.get("authoritative_scope_receipts")
    if (
        not isinstance(selected_receipts, Mapping)
        or tuple(selected_receipts) != EXPECTED_RECEIPT_KINDS
    ):
        raise ScopedValidationError("scoped authoritative rereview receipts are required")
    current_receipts = _build_current_authoritative_scope_receipts()
    if tuple(current_receipts) != EXPECTED_RECEIPT_KINDS:
        raise ScopedValidationError("current authoritative receipt schema drifted")
    validated_receipts: dict[str, dict[str, Any]] = {}
    for kind in EXPECTED_RECEIPT_KINDS:
        receipt = _validate_receipt_self_hash(selected_receipts.get(kind), kind=kind)
        current = _validate_receipt_self_hash(current_receipts.get(kind), kind=kind)
        if receipt != current:
            raise ScopedValidationError(
                f"scoped {kind} authoritative receipt is stale or schema-drifted"
            )
        if any(
            receipt.get(field) != 0
            for field in (
                "hard_rows_counted",
                "hard_usage_counted",
                "hard_materialization_episodes_counted",
            )
        ):
            raise ScopedValidationError(f"scoped {kind} receipt counted hard activity")
        validated_receipts[kind] = receipt
    _validate_case_review_authority_receipt_binding(
        current_review_authority,
        receipts=validated_receipts,
    )
    _validate_source_and_index_integrity(
        suite.get("source_and_index_integrity"),
        summary=summary,
        receipts=validated_receipts,
        manifests=reviewed_manifests,
        frozen_source_authority=frozen_source_authority,
    )

    quality_targets = {
        str(key): int(value) for key, value in SCOPE_QUALITY_TARGETS.items()
    }
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "suite_key": SUITE_KEY,
        "scope": scope_profile(),
        "case_count": EXPECTED_CASE_COUNT,
        "difficulty_case_counts": {
            difficulty: EXPECTED_CASE_COUNT_PER_DIFFICULTY
            for difficulty in SCOPE_DIFFICULTIES
        },
        "quality_target_correct_counts": quality_targets,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "suite": {
            "path": SUITE_FILENAME,
            "sha256": _sha256_file(suite_path, root=root, label="reviewed suite"),
            "canonical_sha256": _sha256_json(suite),
        },
        "summary": {
            "path": SUMMARY_FILENAME,
            "sha256": _sha256_file(summary_path, root=root, label="reviewed summary"),
            "canonical_sha256": _sha256_json(summary),
        },
        "qa": {
            "path": QA_FILENAME,
            "sha256": _sha256_file(qa_path, root=root, label="reviewed QA"),
            "canonical_sha256": _sha256_json(qa_rows),
            "row_count": EXPECTED_CASE_COUNT,
        },
        "datasets": dataset_receipts,
        "checksums": {
            "path": CHECKSUM_FILENAME,
            "sha256": checksum_sha256,
            "covered_file_count": len(checksum_rows),
        },
        "authoritative_scope_receipts_sha256": _sha256_json(selected_receipts),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = validate()
    except (OSError, ValueError, ScopedValidationError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
