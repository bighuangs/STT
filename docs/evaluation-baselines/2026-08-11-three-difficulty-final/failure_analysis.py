#!/usr/bin/env python3
"""Build and validate evidence-bound analysis for every final-answer failure.

The command is local-only.  It never assigns a root cause: an analyst supplies
one annotation for each authoritative wrong answer, and this module binds those
annotations to the immutable run index, results files, and exact result rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
FINAL_ROOT = SCRIPT_PATH.parent
sys.path.insert(0, str(FINAL_ROOT))
import run_final_live as live  # noqa: E402


RECORD_SCHEMA = "stt.eval.failure-analysis-record.v1"
SUMMARY_SCHEMA = "stt.eval.failure-analysis-summary.v1"
TAXONOMY = (
    "retrieval_insufficient",
    "retrieval_noisy_or_unclear",
    "context_loss",
    "intermediate_process_loss",
    "evaluator_error",
    "other",
)
DIVERGENCE_STAGES = (
    "retrieval",
    "reranking",
    "context_assembly",
    "generation",
    "answer_postprocessing",
    "evaluation",
    "unknown",
)
ANNOTATION_FIELDS = {
    "difficulty",
    "run_id",
    "case_id",
    "primary_cause",
    "earliest_divergence",
    "mechanism",
    "evidence",
}
RECORD_FIELDS = {
    "schema_version",
    "label",
    "suite_key",
    "difficulty",
    "run_id",
    "case_id",
    "case_key",
    "final_answer_correct",
    "strict_supported_case_pass",
    "primary_cause",
    "earliest_divergence",
    "mechanism",
    "evidence",
    "source_receipts",
    "result_row_sha256",
    "record_sha256",
}


class FailureAnalysisError(RuntimeError):
    """Fail-closed failure-analysis contract error."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, root: Path, label: str) -> str:
    try:
        return live._sha256_file(path, root=root, label=label)
    except live.FinalLiveError as exc:
        raise FailureAnalysisError(str(exc)) from exc


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


def _safe_label(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise FailureAnalysisError("invalid label")
    return value


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise FailureAnalysisError(f"{label} escapes final suite root") from exc


def _contained(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    if not value or candidate.is_absolute():
        raise FailureAnalysisError(f"{label} path must be final-root-relative")
    try:
        return live._assert_safe_path(
            root / candidate,
            root=root,
            label=label,
        )
    except live.FinalLiveError as exc:
        raise FailureAnalysisError(str(exc)) from exc


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    try:
        return live._read_json(path, root=root, label=label)
    except live.FinalLiveError as exc:
        raise FailureAnalysisError(str(exc)) from exc


def _read_jsonl(path: Path, *, root: Path, label: str) -> list[dict[str, Any]]:
    safe_path = _contained(root, _relative(root, path, label=label), label=label)
    records: list[dict[str, Any]] = []
    try:
        with safe_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FailureAnalysisError(
                        f"{label} contains invalid JSON on row {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise FailureAnalysisError(f"{label} row {line_number} must contain an object")
                records.append(value)
    except OSError as exc:
        raise FailureAnalysisError(f"cannot read {label}: {safe_path}") from exc
    return records


def _case_outcome(row: Mapping[str, Any], case_key: str) -> tuple[bool, bool | None]:
    metrics = row.get("metrics")
    nested_metrics = metrics.get("metrics") if isinstance(metrics, Mapping) else None
    outcome = nested_metrics.get("case_outcome") if isinstance(nested_metrics, Mapping) else None
    correct = outcome.get("final_answer_correct") if isinstance(outcome, Mapping) else None
    strict = outcome.get("strict_supported_case_pass") if isinstance(outcome, Mapping) else None
    if type(correct) is not bool:
        raise FailureAnalysisError(f"{case_key} has no JSON boolean final_answer_correct")
    if strict is not None and type(strict) is not bool:
        raise FailureAnalysisError(
            f"{case_key} strict_supported_case_pass must be a JSON boolean or null"
        )
    return correct, strict


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
        raise FailureAnalysisError("run-index path does not match the selected label")
    index = _read_json(chosen_index, root=root, label="failure-analysis run index")
    if not isinstance(index, Mapping):
        raise FailureAnalysisError("run-index must contain an object")
    expected_identity = {
        "schema_version": live.INDEX_SCHEMA,
        "status": "succeeded",
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
    }
    changed = [key for key, expected in expected_identity.items() if index.get(key) != expected]
    if changed:
        raise FailureAnalysisError("run-index identity is invalid: " + ", ".join(changed))
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping) or set(entries) != set(live.DIFFICULTIES):
        raise FailureAnalysisError("run-index must contain exactly three difficulty entries")

    index_receipt = {
        "path": _relative(root, chosen_index, label="run index"),
        "sha256": _sha256_file(chosen_index, root=root, label="failure-analysis run index"),
    }
    results: dict[str, dict[str, Any]] = {}
    wrong: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_case_ids: set[str] = set()
    for difficulty in live.DIFFICULTIES:
        entry = entries[difficulty]
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise FailureAnalysisError(f"{difficulty} run-index entry is not archived")
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
            raise FailureAnalysisError(f"{difficulty} selected-case identity is invalid")
        duplicate_ids = all_case_ids.intersection(selected)
        if duplicate_ids:
            raise FailureAnalysisError("case ids are not globally unique across difficulties")
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
            raise FailureAnalysisError(f"{difficulty} results SHA changed")
        try:
            rows = live._read_results_jsonl(results_path, root=root)
            target = live._target_count_from_rows(
                rows,
                expected_run_id=run_id,
                expected_case_ids=tuple(selected),
            )
        except live.FinalLiveError as exc:
            raise FailureAnalysisError(str(exc)) from exc
        if entry.get("target_result") != target:
            raise FailureAnalysisError(f"{difficulty} target summary differs from results")
        by_case: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id") or "")
            if case_id in by_case:
                raise FailureAnalysisError(f"{difficulty} results contain duplicate case ids")
            by_case[case_id] = row
        wrong_count = 0
        for case_id in selected:
            row = by_case[case_id]
            key = (difficulty, run_id, case_id)
            correct, strict = _case_outcome(row, ":".join(key))
            if correct is False:
                wrong_count += 1
                wrong[key] = {
                    "row": row,
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
    if len(all_case_ids) != live.EXPECTED_CASE_COUNT * len(live.DIFFICULTIES):
        raise FailureAnalysisError("run-index does not cover exactly 150 unique case ids")
    return {
        "index": index,
        "index_path": chosen_index,
        "index_receipt": index_receipt,
        "results": results,
        "wrong": wrong,
    }


def _nonempty_text(value: Any, *, label: str, minimum: int = 8) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise FailureAnalysisError(f"{label} must contain at least {minimum} non-blank characters")
    return value.strip()


def _validate_divergence(value: Any, *, case_key: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"stage", "description"}:
        raise FailureAnalysisError(f"{case_key} earliest_divergence schema is invalid")
    stage = value.get("stage")
    if stage not in DIVERGENCE_STAGES:
        raise FailureAnalysisError(f"{case_key} earliest_divergence stage is invalid")
    return {
        "stage": str(stage),
        "description": _nonempty_text(
            value.get("description"),
            label=f"{case_key} earliest_divergence description",
        ),
    }


def _validate_evidence(
    root: Path,
    value: Any,
    *,
    case_key: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise FailureAnalysisError(f"{case_key} must contain evidence path/SHA receipts")
    verified: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for position, item in enumerate(value, start=1):
        if not isinstance(item, Mapping) or set(item) not in (
            {"path", "sha256"},
            {"path", "sha256", "description"},
        ):
            raise FailureAnalysisError(f"{case_key} evidence {position} schema is invalid")
        path = _contained(
            root,
            str(item.get("path") or ""),
            label=f"{case_key} evidence {position}",
        )
        relative = _relative(root, path, label=f"{case_key} evidence {position}")
        if relative in seen_paths:
            raise FailureAnalysisError(f"{case_key} evidence paths must be unique")
        seen_paths.add(relative)
        actual_sha = _sha256_file(
            path,
            root=root,
            label=f"{case_key} evidence {position}",
        )
        if item.get("sha256") != actual_sha:
            raise FailureAnalysisError(f"{case_key} evidence {position} SHA changed")
        receipt = {"path": relative, "sha256": actual_sha}
        if "description" in item:
            receipt["description"] = _nonempty_text(
                item.get("description"),
                label=f"{case_key} evidence {position} description",
                minimum=4,
            )
        verified.append(receipt)
    return verified


def _record_from_analysis(
    root: Path,
    label: str,
    analysis: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    difficulty = str(analysis.get("difficulty") or "")
    run_id = str(analysis.get("run_id") or "")
    case_id = str(analysis.get("case_id") or "")
    case_key = f"{difficulty}:{run_id}:{case_id}"
    primary_cause = analysis.get("primary_cause")
    if primary_cause not in TAXONOMY:
        raise FailureAnalysisError(f"{case_key} primary_cause is not one of the frozen six classes")
    record = {
        "schema_version": RECORD_SCHEMA,
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "difficulty": difficulty,
        "run_id": run_id,
        "case_id": case_id,
        "case_key": case_key,
        "final_answer_correct": False,
        "strict_supported_case_pass": evidence["strict_supported_case_pass"],
        "primary_cause": primary_cause,
        "earliest_divergence": _validate_divergence(
            analysis.get("earliest_divergence"), case_key=case_key
        ),
        "mechanism": _nonempty_text(analysis.get("mechanism"), label=f"{case_key} mechanism"),
        "evidence": _validate_evidence(
            root,
            analysis.get("evidence"),
            case_key=case_key,
        ),
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
        label="failure-analysis annotations",
    )
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for position, annotation in enumerate(annotations, start=1):
        if set(annotation) != ANNOTATION_FIELDS:
            raise FailureAnalysisError(
                f"failure annotation {position} fields differ from the frozen schema"
            )
        key = (
            str(annotation.get("difficulty") or ""),
            str(annotation.get("run_id") or ""),
            str(annotation.get("case_id") or ""),
        )
        if key in by_key:
            raise FailureAnalysisError(
                f"failure annotations contain duplicate case {':'.join(key)}"
            )
        by_key[key] = annotation
    expected_keys = set(run_evidence["wrong"])
    if set(by_key) != expected_keys:
        missing = sorted(":".join(key) for key in expected_keys - set(by_key))
        extra = sorted(":".join(key) for key in set(by_key) - expected_keys)
        raise FailureAnalysisError(
            "failure annotations must exactly cover final_answer_correct=false; "
            f"missing={missing}, extra={extra}"
        )
    difficulty_order = {
        difficulty: position for position, difficulty in enumerate(live.DIFFICULTIES)
    }
    ordered_keys = sorted(
        expected_keys,
        key=lambda key: (difficulty_order[key[0]], key[2], key[1]),
    )
    return [
        _record_from_analysis(
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
            raise FailureAnalysisError(
                f"failure-analysis record {position} fields differ from the frozen schema"
            )
        key = (
            str(record.get("difficulty") or ""),
            str(record.get("run_id") or ""),
            str(record.get("case_id") or ""),
        )
        if key in by_key:
            raise FailureAnalysisError(
                f"failure-analysis records contain duplicate case {':'.join(key)}"
            )
        by_key[key] = record
    expected_keys = set(run_evidence["wrong"])
    if set(by_key) != expected_keys:
        raise FailureAnalysisError(
            "failure-analysis records must exactly cover final_answer_correct=false"
        )
    difficulty_order = {
        difficulty: position for position, difficulty in enumerate(live.DIFFICULTIES)
    }
    ordered_keys = sorted(
        expected_keys,
        key=lambda key: (difficulty_order[key[0]], key[2], key[1]),
    )
    verified: list[dict[str, Any]] = []
    for key in ordered_keys:
        stored = by_key[key]
        evidence = run_evidence["wrong"][key]
        expected_identity = {
            "schema_version": RECORD_SCHEMA,
            "label": label,
            "suite_key": live.EXPECTED_SUITE_KEY,
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
            raise FailureAnalysisError(
                f"{':'.join(key)} source/result binding changed: " + ", ".join(changed)
            )
        primary_cause = stored.get("primary_cause")
        if primary_cause not in TAXONOMY:
            raise FailureAnalysisError(f"{':'.join(key)} primary_cause is invalid")
        case_key = ":".join(key)
        _validate_divergence(stored.get("earliest_divergence"), case_key=case_key)
        _nonempty_text(stored.get("mechanism"), label=f"{':'.join(key)} mechanism")
        _validate_evidence(root, stored.get("evidence"), case_key=case_key)
        if stored.get("record_sha256") != _record_sha256(stored):
            raise FailureAnalysisError(f"{':'.join(key)} record SHA is invalid")
        verified.append(dict(stored))
    return verified


def _summary(
    root: Path,
    label: str,
    records: Sequence[Mapping[str, Any]],
    run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    label_root = root / "runs" / label
    records_path = label_root / "failure-analysis.jsonl"
    records_sha = _sha256_bytes(_jsonl_bytes(records))
    primary_cause_counts = Counter(str(record["primary_cause"]) for record in records)
    divergence_stage_counts = Counter(
        str(record["earliest_divergence"]["stage"]) for record in records
    )
    pattern_case_keys: dict[tuple[str, str], list[str]] = {}
    for record in records:
        pattern = (
            str(record["primary_cause"]),
            str(record["earliest_divergence"]["stage"]),
        )
        pattern_case_keys.setdefault(pattern, []).append(str(record["case_key"]))
    cause_order = {value: position for position, value in enumerate(TAXONOMY)}
    stage_order = {value: position for position, value in enumerate(DIVERGENCE_STAGES)}
    ordered_patterns = sorted(
        pattern_case_keys,
        key=lambda key: (
            -len(pattern_case_keys[key]),
            cause_order[key[0]],
            stage_order[key[1]],
        ),
    )
    high_frequency_patterns = []
    for primary_cause, stage in ordered_patterns:
        pattern_keys = pattern_case_keys[(primary_cause, stage)]
        high_frequency_patterns.append(
            {
                "primary_cause": primary_cause,
                "earliest_divergence_stage": stage,
                "count": len(pattern_keys),
                "case_keys": pattern_keys,
                "case_keys_sha256": live._sha256_json(pattern_keys),
            }
        )
    per_difficulty = {
        difficulty: {
            "case_count": live.EXPECTED_CASE_COUNT,
            "wrong_case_count": sum(record["difficulty"] == difficulty for record in records),
        }
        for difficulty in live.DIFFICULTIES
    }
    evaluator_errors = primary_cause_counts["evaluator_error"]
    case_keys = [str(record["case_key"]) for record in records]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
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
        "records_path": _relative(root, records_path, label="failure-analysis records"),
        "records_sha256": records_sha,
        "record_count": len(records),
        "case_keys_sha256": live._sha256_json(case_keys),
        "primary_cause_counts": {name: primary_cause_counts[name] for name in TAXONOMY},
        "divergence_stage_counts": {
            stage: divergence_stage_counts[stage] for stage in DIVERGENCE_STAGES
        },
        "high_frequency_patterns": high_frequency_patterns,
        "per_difficulty": per_difficulty,
        "evaluator_error_count": evaluator_errors,
        "promotion_allowed": evaluator_errors == 0,
    }


def build_failure_artifacts(
    final_root: Path,
    label: str,
    annotations_path: Path,
    *,
    index_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic records/summary bytes from exact analyst annotations."""

    root = final_root.resolve()
    run_evidence = _load_run_evidence(root, label, index_path=index_path)
    records = _records_from_annotations(root, label, annotations_path, run_evidence)
    return records, _summary(root, label, records, run_evidence)


def _write_exact_or_refuse(path: Path, content: bytes, *, root: Path, label: str) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise FailureAnalysisError(f"cannot read existing {label}: {path}") from exc
        if existing != content:
            raise FailureAnalysisError(f"refusing to overwrite changed {label}: {path}")
        return
    try:
        live._write_bytes_safe(
            path,
            content,
            root=root,
            label=label,
            overwrite=False,
        )
    except live.FinalLiveError as exc:
        raise FailureAnalysisError(str(exc)) from exc


def _artifact_receipt(
    root: Path,
    label: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    records_path = root / str(summary["records_path"])
    summary_path = root / "runs" / label / "failure-summary.json"
    return {
        "records_path": _relative(root, records_path, label="failure-analysis records"),
        "records_sha256": _sha256_file(records_path, root=root, label="failure-analysis records"),
        "summary_path": _relative(root, summary_path, label="failure-analysis summary"),
        "summary_sha256": _sha256_file(summary_path, root=root, label="failure-analysis summary"),
        "record_count": summary["record_count"],
        "case_keys_sha256": summary["case_keys_sha256"],
        "primary_cause_counts": dict(summary["primary_cause_counts"]),
        "evaluator_error_count": summary["evaluator_error_count"],
        "promotion_allowed": summary["promotion_allowed"],
    }


def write_failure_artifacts(
    final_root: Path,
    label: str,
    annotations_path: Path,
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Write immutable failure-analysis JSONL and recomputed summary."""

    root = final_root.resolve()
    records, summary = build_failure_artifacts(
        root,
        label,
        annotations_path,
        index_path=index_path,
    )
    label_root = root / "runs" / label
    records_path = label_root / "failure-analysis.jsonl"
    summary_path = label_root / "failure-summary.json"
    _write_exact_or_refuse(
        records_path,
        _jsonl_bytes(records),
        root=root,
        label="failure-analysis records",
    )
    _write_exact_or_refuse(
        summary_path,
        _json_bytes(summary),
        root=root,
        label="failure-analysis summary",
    )
    return validate_failure_artifacts(root, label, index_path=index_path)


def validate_failure_artifacts(
    final_root: Path,
    label: str,
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute the exact wrong-case universe, bindings, aggregates, and receipt."""

    root = final_root.resolve()
    label = _safe_label(label)
    run_evidence = _load_run_evidence(root, label, index_path=index_path)
    label_root = root / "runs" / label
    records_path = label_root / "failure-analysis.jsonl"
    summary_path = label_root / "failure-summary.json"
    records = _read_jsonl(
        records_path,
        root=root,
        label="failure-analysis records",
    )
    verified_records = _validate_records(root, label, records, run_evidence)
    canonical_records = _jsonl_bytes(verified_records)
    try:
        actual_records = records_path.read_bytes()
    except OSError as exc:
        raise FailureAnalysisError("cannot read failure-analysis records") from exc
    if actual_records != canonical_records:
        raise FailureAnalysisError("failure-analysis JSONL is not in canonical order/encoding")
    summary = _read_json(summary_path, root=root, label="failure-analysis summary")
    if not isinstance(summary, Mapping):
        raise FailureAnalysisError("failure-analysis summary must contain an object")
    expected_summary = _summary(root, label, verified_records, run_evidence)
    if summary != expected_summary:
        raise FailureAnalysisError("failure-analysis summary does not recompute exactly")
    return _artifact_receipt(root, label, summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="analyst-authored exact wrong-case JSONL (default: runs/<label>/failure-analysis-annotations.jsonl)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write final immutable artifacts")
    mode.add_argument("--validate", action="store_true", help="validate existing artifacts")
    args = parser.parse_args(argv)
    try:
        args.label = _safe_label(args.label)
    except FailureAnalysisError as exc:
        parser.error(str(exc))
    if args.annotations is None:
        args.annotations = FINAL_ROOT / "runs" / args.label / "failure-analysis-annotations.jsonl"
    elif not args.annotations.is_absolute():
        args.annotations = FINAL_ROOT / args.annotations
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate:
        receipt = validate_failure_artifacts(FINAL_ROOT, args.label)
    else:
        receipt = write_failure_artifacts(FINAL_ROOT, args.label, args.annotations)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FailureAnalysisError, live.FinalLiveError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
