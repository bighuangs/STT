#!/usr/bin/env python3
"""为 easy/medium scoped run 构建不可晋级、证据绑定的错题诊断。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_easy_medium_live as live
import scoped_run_archive as run_archive


SCRIPT_PATH = Path(__file__).resolve()
FINAL_ROOT = SCRIPT_PATH.parent
CORE_PATH = live.FULL_ROOT / "failure_analysis.py"
_CORE_MODULE_NAME = "_stt_full_failure_analysis_core"
_spec = importlib.util.spec_from_file_location(_CORE_MODULE_NAME, CORE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - 固定仓库路径保证可加载
    raise RuntimeError(f"cannot load shared failure-analysis core: {CORE_PATH}")
core = importlib.util.module_from_spec(_spec)
sys.modules[_CORE_MODULE_NAME] = core
_spec.loader.exec_module(core)


RECORD_SCHEMA = "stt.eval.easy-medium-diagnostic-failure-record.v1"
SUMMARY_SCHEMA = "stt.eval.easy-medium-diagnostic-failure-summary.v1"
DIAGNOSTIC_ROLE = "non_promotable_easy_medium_failure_diagnostic"
ANNOTATIONS_FILENAME = "diagnostic-failure-analysis-annotations.jsonl"
RECORDS_FILENAME = "diagnostic-failure-analysis.jsonl"
SUMMARY_FILENAME = "diagnostic-failure-summary.json"
TAXONOMY = core.TAXONOMY
DIVERGENCE_STAGES = core.DIVERGENCE_STAGES
ANNOTATION_FIELDS = core.ANNOTATION_FIELDS
RECORD_FIELDS = core.RECORD_FIELDS | {
    "scope_id",
    "role",
    "canonical",
    "promotion_allowed",
}


class DiagnosticFailureError(RuntimeError):
    """表示 scoped 错题诊断在证据或身份校验上的失败。"""


def _safe_label(value: str) -> str:
    try:
        return core._safe_label(value)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        return core._relative(root, path, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _contained(root: Path, value: str, *, label: str) -> Path:
    try:
        return core._contained(root, value, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    try:
        return core._read_json(path, root=root, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _read_jsonl(path: Path, *, root: Path, label: str) -> list[dict[str, Any]]:
    try:
        return core._read_jsonl(path, root=root, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _sha256_file(path: Path, *, root: Path, label: str) -> str:
    return live._sha256_file(path, root=root, label=label)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, Any]) -> str:
    return live._sha256_json(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _load_run_evidence(
    root: Path,
    label: str,
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    label = _safe_label(label)
    expected_index = root / "runs" / label / "run-index.json"
    chosen_index = (index_path or expected_index).resolve()
    if chosen_index != expected_index.resolve():
        raise DiagnosticFailureError("run-index path does not match the selected label")
    index = _read_json(chosen_index, root=root, label="scoped diagnostic run index")
    if not isinstance(index, Mapping):
        raise DiagnosticFailureError("run-index must contain an object")
    try:
        suite = live.load_suite_snapshot(root)
        run_archive.validate_scoped_run_archive(
            chosen_index,
            index,
            suite,
            root=root,
        )
    except (
        live.ScopedLiveError,
        live.full.FinalLiveError,
        run_archive.ScopedRunArchiveError,
    ) as exc:
        raise DiagnosticFailureError(f"scoped run archive is invalid: {exc}") from exc
    expected_identity = {
        "schema_version": live.INDEX_SCHEMA,
        "status": "succeeded",
        "label": label,
        "role": live.RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "scope": live.scope.scope_profile(),
    }
    changed = [key for key, expected in expected_identity.items() if index.get(key) != expected]
    if changed:
        raise DiagnosticFailureError("run-index identity is invalid: " + ", ".join(changed))
    if {"candidate_roots", "source_baseline_registry", "baselines"}.intersection(index):
        raise DiagnosticFailureError("run-index identity contains forbidden promotion fields")
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping) or set(entries) != set(live.DIFFICULTIES):
        raise DiagnosticFailureError("run-index must contain exactly easy and medium")

    index_receipt = {
        "path": _relative(root, chosen_index, label="run index"),
        "sha256": _sha256_file(chosen_index, root=root, label="scoped diagnostic run index"),
    }
    results: dict[str, dict[str, Any]] = {}
    wrong: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_case_ids: set[str] = set()
    for difficulty in live.DIFFICULTIES:
        entry = entries[difficulty]
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise DiagnosticFailureError(f"{difficulty} run-index entry is not archived")
        run_id = str(entry.get("run_id") or "")
        selected = entry.get("selected_case_ids")
        if (
            not run_id
            or not isinstance(selected, list)
            or len(selected) != live.EXPECTED_CASE_COUNT
            or any(not isinstance(case_id, str) or not case_id for case_id in selected)
            or len(set(selected)) != live.EXPECTED_CASE_COUNT
            or entry.get("selected_case_count") != live.EXPECTED_CASE_COUNT
        ):
            raise DiagnosticFailureError(f"{difficulty} selected-case identity is invalid")
        if all_case_ids.intersection(selected):
            raise DiagnosticFailureError("case ids are not globally unique across scoped difficulties")
        all_case_ids.update(selected)
        results_path = _contained(
            root,
            str(entry.get("results_path") or ""),
            label=f"{difficulty} results",
        )
        actual_results_sha = _sha256_file(
            results_path,
            root=root,
            label=f"{difficulty} results",
        )
        if entry.get("results_sha256") != actual_results_sha:
            raise DiagnosticFailureError(f"{difficulty} results SHA changed")
        rows = live._read_results_jsonl(results_path, root=root)
        target = live._target_count_from_rows(
            rows,
            expected_run_id=run_id,
            expected_case_ids=tuple(selected),
        )
        if entry.get("target_result") != target:
            raise DiagnosticFailureError(f"{difficulty} target summary differs from results")
        by_case: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id") or "")
            if case_id in by_case:
                raise DiagnosticFailureError(f"{difficulty} results contain duplicate case ids")
            by_case[case_id] = row
        wrong_count = 0
        for case_id in selected:
            row = by_case[case_id]
            key = (difficulty, run_id, case_id)
            try:
                correct, strict = core._case_outcome(row, ":".join(key))
            except core.FailureAnalysisError as exc:
                raise DiagnosticFailureError(str(exc)) from exc
            if correct is False:
                wrong_count += 1
                wrong[key] = {
                    "strict_supported_case_pass": strict,
                    "result_row_sha256": live._sha256_json(row),
                    "results_receipt": {
                        "path": _relative(root, results_path, label=f"{difficulty} results"),
                        "sha256": actual_results_sha,
                    },
                }
        results[difficulty] = {
            "path": _relative(root, results_path, label=f"{difficulty} results"),
            "sha256": actual_results_sha,
            "run_id": run_id,
            "case_count": live.EXPECTED_CASE_COUNT,
            "wrong_case_count": wrong_count,
        }
    if len(all_case_ids) != live.EXPECTED_TOTAL_CASE_COUNT:
        raise DiagnosticFailureError("run-index does not cover exactly 100 unique scoped case ids")
    return {
        "index": index,
        "index_path": chosen_index,
        "index_receipt": index_receipt,
        "results": results,
        "wrong": wrong,
    }


def _validate_divergence(value: Any, *, case_key: str) -> dict[str, str]:
    try:
        return core._validate_divergence(value, case_key=case_key)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _validate_evidence(
    root: Path,
    value: Any,
    *,
    case_key: str,
) -> list[dict[str, str]]:
    try:
        return core._validate_evidence(root, value, case_key=case_key)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _nonempty_text(value: Any, *, label: str) -> str:
    try:
        return core._nonempty_text(value, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _record_from_annotation(
    root: Path,
    label: str,
    annotation: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    difficulty = str(annotation.get("difficulty") or "")
    run_id = str(annotation.get("run_id") or "")
    case_id = str(annotation.get("case_id") or "")
    case_key = f"{difficulty}:{run_id}:{case_id}"
    primary_cause = annotation.get("primary_cause")
    if primary_cause not in TAXONOMY:
        raise DiagnosticFailureError(f"{case_key} primary_cause is invalid")
    record = {
        "schema_version": RECORD_SCHEMA,
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "scope_id": live.scope.SCOPE_ID,
        "role": DIAGNOSTIC_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "difficulty": difficulty,
        "run_id": run_id,
        "case_id": case_id,
        "case_key": case_key,
        "final_answer_correct": False,
        "strict_supported_case_pass": evidence["strict_supported_case_pass"],
        "primary_cause": primary_cause,
        "earliest_divergence": _validate_divergence(
            annotation.get("earliest_divergence"),
            case_key=case_key,
        ),
        "mechanism": _nonempty_text(annotation.get("mechanism"), label=f"{case_key} mechanism"),
        "evidence": _validate_evidence(root, annotation.get("evidence"), case_key=case_key),
        "source_receipts": {
            "run_index": dict(run_evidence["index_receipt"]),
            "results": dict(evidence["results_receipt"]),
        },
        "result_row_sha256": evidence["result_row_sha256"],
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def _records_from_annotations(
    root: Path,
    label: str,
    annotations_path: Path,
    run_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    annotations = _read_jsonl(
        annotations_path,
        root=root,
        label="scoped diagnostic annotations",
    )
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for position, annotation in enumerate(annotations, start=1):
        if set(annotation) != ANNOTATION_FIELDS:
            raise DiagnosticFailureError(
                f"diagnostic annotation {position} fields differ from the frozen schema"
            )
        key = (
            str(annotation.get("difficulty") or ""),
            str(annotation.get("run_id") or ""),
            str(annotation.get("case_id") or ""),
        )
        if key in by_key:
            raise DiagnosticFailureError(
                f"diagnostic annotations contain duplicate case {':'.join(key)}"
            )
        by_key[key] = annotation
    expected_keys = set(run_evidence["wrong"])
    if set(by_key) != expected_keys:
        missing = sorted(":".join(key) for key in expected_keys - set(by_key))
        extra = sorted(":".join(key) for key in set(by_key) - expected_keys)
        raise DiagnosticFailureError(
            "diagnostic annotations must exactly cover scoped final_answer_correct=false; "
            f"missing={missing}, extra={extra}"
        )
    order = {difficulty: position for position, difficulty in enumerate(live.DIFFICULTIES)}
    ordered_keys = sorted(expected_keys, key=lambda key: (order[key[0]], key[2], key[1]))
    return [
        _record_from_annotation(
            root,
            label,
            by_key[key],
            evidence=run_evidence["wrong"][key],
            run_evidence=run_evidence,
        )
        for key in ordered_keys
    ]


def _validate_records(
    root: Path,
    label: str,
    records: Sequence[Mapping[str, Any]],
    run_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != RECORD_FIELDS:
            raise DiagnosticFailureError(
                f"diagnostic record {position} fields differ from the frozen schema"
            )
        key = (
            str(record.get("difficulty") or ""),
            str(record.get("run_id") or ""),
            str(record.get("case_id") or ""),
        )
        if key in by_key:
            raise DiagnosticFailureError(f"diagnostic records contain duplicate case {':'.join(key)}")
        by_key[key] = record
    expected_keys = set(run_evidence["wrong"])
    if set(by_key) != expected_keys:
        raise DiagnosticFailureError(
            "diagnostic records must exactly cover scoped final_answer_correct=false"
        )
    order = {difficulty: position for position, difficulty in enumerate(live.DIFFICULTIES)}
    ordered_keys = sorted(expected_keys, key=lambda key: (order[key[0]], key[2], key[1]))
    verified: list[dict[str, Any]] = []
    for key in ordered_keys:
        stored = by_key[key]
        evidence = run_evidence["wrong"][key]
        expected_identity = {
            "schema_version": RECORD_SCHEMA,
            "label": label,
            "suite_key": live.EXPECTED_SUITE_KEY,
            "scope_id": live.scope.SCOPE_ID,
            "role": DIAGNOSTIC_ROLE,
            "canonical": False,
            "promotion_allowed": False,
            "difficulty": key[0],
            "run_id": key[1],
            "case_id": key[2],
            "case_key": ":".join(key),
            "final_answer_correct": False,
            "strict_supported_case_pass": evidence["strict_supported_case_pass"],
            "source_receipts": {
                "run_index": dict(run_evidence["index_receipt"]),
                "results": dict(evidence["results_receipt"]),
            },
            "result_row_sha256": evidence["result_row_sha256"],
        }
        changed = [
            field for field, expected in expected_identity.items() if stored.get(field) != expected
        ]
        if changed:
            raise DiagnosticFailureError(
                f"{':'.join(key)} source/result binding changed: " + ", ".join(changed)
            )
        if stored.get("primary_cause") not in TAXONOMY:
            raise DiagnosticFailureError(f"{':'.join(key)} primary_cause is invalid")
        case_key = ":".join(key)
        _validate_divergence(stored.get("earliest_divergence"), case_key=case_key)
        _nonempty_text(stored.get("mechanism"), label=f"{case_key} mechanism")
        _validate_evidence(root, stored.get("evidence"), case_key=case_key)
        if stored.get("record_sha256") != _record_sha256(stored):
            raise DiagnosticFailureError(f"{case_key} record SHA is invalid")
        verified.append(dict(stored))
    return verified


def _summary(
    root: Path,
    label: str,
    records: Sequence[Mapping[str, Any]],
    run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    records_path = root / "runs" / label / RECORDS_FILENAME
    primary_counts = Counter(str(record["primary_cause"]) for record in records)
    stage_counts = Counter(str(record["earliest_divergence"]["stage"]) for record in records)
    pattern_keys: dict[tuple[str, str], list[str]] = {}
    for record in records:
        pattern = (
            str(record["primary_cause"]),
            str(record["earliest_divergence"]["stage"]),
        )
        pattern_keys.setdefault(pattern, []).append(str(record["case_key"]))
    cause_order = {value: position for position, value in enumerate(TAXONOMY)}
    stage_order = {value: position for position, value in enumerate(DIVERGENCE_STAGES)}
    ordered_patterns = sorted(
        pattern_keys,
        key=lambda key: (-len(pattern_keys[key]), cause_order[key[0]], stage_order[key[1]]),
    )
    high_frequency_patterns = [
        {
            "primary_cause": primary_cause,
            "earliest_divergence_stage": stage,
            "count": len(pattern_keys[(primary_cause, stage)]),
            "case_keys": pattern_keys[(primary_cause, stage)],
            "case_keys_sha256": live._sha256_json(pattern_keys[(primary_cause, stage)]),
        }
        for primary_cause, stage in ordered_patterns
    ]
    case_keys = [str(record["case_key"]) for record in records]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "scope": live.scope.scope_profile(),
        "role": DIAGNOSTIC_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "non_promotable_reason": "hard is reference-only and the scoped protocol forbids canonical promotion",
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "analysis_policy": {
            "wrong_case_selector": "final_answer_correct === false",
            "taxonomy_assignment": "analyst_supplied_not_automatically_inferred",
            "exact_wrong_case_coverage_required": True,
        },
        "taxonomy": list(TAXONOMY),
        "run_index": dict(run_evidence["index_receipt"]),
        "results": {
            difficulty: dict(run_evidence["results"][difficulty])
            for difficulty in live.DIFFICULTIES
        },
        "records_path": _relative(root, records_path, label="diagnostic records"),
        "records_sha256": core._sha256_bytes(_jsonl_bytes(records)),
        "record_count": len(records),
        "case_keys_sha256": live._sha256_json(case_keys),
        "primary_cause_counts": {name: primary_counts[name] for name in TAXONOMY},
        "divergence_stage_counts": {
            stage: stage_counts[stage] for stage in DIVERGENCE_STAGES
        },
        "high_frequency_patterns": high_frequency_patterns,
        "per_difficulty": {
            difficulty: {
                "case_count": live.EXPECTED_CASE_COUNT,
                "wrong_case_count": sum(
                    record["difficulty"] == difficulty for record in records
                ),
            }
            for difficulty in live.DIFFICULTIES
        },
        "evaluator_error_count": primary_counts["evaluator_error"],
    }


def build_diagnostic_artifacts(
    root: Path,
    label: str,
    annotations_path: Path,
    *,
    index_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从权威 scoped 错题全集生成确定性的诊断记录与摘要。"""

    chosen_root = root.resolve()
    run_evidence = _load_run_evidence(chosen_root, label, index_path=index_path)
    records = _records_from_annotations(
        chosen_root,
        label,
        annotations_path,
        run_evidence,
    )
    return records, _summary(chosen_root, label, records, run_evidence)


def _write_exact_or_refuse(path: Path, content: bytes, *, root: Path, label: str) -> None:
    try:
        core._write_exact_or_refuse(path, content, root=root, label=label)
    except core.FailureAnalysisError as exc:
        raise DiagnosticFailureError(str(exc)) from exc


def _artifact_receipt(root: Path, label: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    records_path = root / str(summary["records_path"])
    summary_path = root / "runs" / label / SUMMARY_FILENAME
    return {
        "role": DIAGNOSTIC_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "scope_id": live.scope.SCOPE_ID,
        "records_path": _relative(root, records_path, label="diagnostic records"),
        "records_sha256": _sha256_file(records_path, root=root, label="diagnostic records"),
        "summary_path": _relative(root, summary_path, label="diagnostic summary"),
        "summary_sha256": _sha256_file(summary_path, root=root, label="diagnostic summary"),
        "record_count": summary["record_count"],
        "case_keys_sha256": summary["case_keys_sha256"],
        "primary_cause_counts": dict(summary["primary_cause_counts"]),
        "evaluator_error_count": summary["evaluator_error_count"],
    }


def write_diagnostic_artifacts(
    root: Path,
    label: str,
    annotations_path: Path,
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """以 create-once/exact-idempotent 语义写入 scoped 诊断制品。"""

    chosen_root = root.resolve()
    records, summary = build_diagnostic_artifacts(
        chosen_root,
        label,
        annotations_path,
        index_path=index_path,
    )
    label_root = chosen_root / "runs" / label
    _write_exact_or_refuse(
        label_root / RECORDS_FILENAME,
        _jsonl_bytes(records),
        root=chosen_root,
        label="scoped diagnostic records",
    )
    _write_exact_or_refuse(
        label_root / SUMMARY_FILENAME,
        _json_bytes(summary),
        root=chosen_root,
        label="scoped diagnostic summary",
    )
    return validate_diagnostic_artifacts(chosen_root, label, index_path=index_path)


def validate_diagnostic_artifacts(
    root: Path,
    label: str,
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """重算错题全集、逐行绑定、聚合与不可晋级收据。"""

    chosen_root = root.resolve()
    label = _safe_label(label)
    run_evidence = _load_run_evidence(chosen_root, label, index_path=index_path)
    label_root = chosen_root / "runs" / label
    records_path = label_root / RECORDS_FILENAME
    summary_path = label_root / SUMMARY_FILENAME
    records = _read_jsonl(records_path, root=chosen_root, label="scoped diagnostic records")
    verified_records = _validate_records(
        chosen_root,
        label,
        records,
        run_evidence,
    )
    if records_path.read_bytes() != _jsonl_bytes(verified_records):
        raise DiagnosticFailureError("diagnostic JSONL is not in canonical order/encoding")
    summary = _read_json(summary_path, root=chosen_root, label="scoped diagnostic summary")
    expected_summary = _summary(chosen_root, label, verified_records, run_evidence)
    if not isinstance(summary, Mapping) or summary != expected_summary:
        raise DiagnosticFailureError("diagnostic summary does not recompute exactly")
    return _artifact_receipt(chosen_root, label, summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--annotations", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.label = _safe_label(args.label)
    except DiagnosticFailureError as exc:
        parser.error(str(exc))
    if args.annotations is None:
        args.annotations = FINAL_ROOT / "runs" / args.label / ANNOTATIONS_FILENAME
    elif not args.annotations.is_absolute():
        args.annotations = FINAL_ROOT / args.annotations
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate:
        receipt = validate_diagnostic_artifacts(FINAL_ROOT, args.label)
    else:
        receipt = write_diagnostic_artifacts(FINAL_ROOT, args.label, args.annotations)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticFailureError, live.ScopedLiveError, live.full.FinalLiveError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
