#!/usr/bin/env python3
"""Summarize the completed easy/medium/hard live baselines.

The script is deliberately post-processing only.  It never calls the API and
refuses to read run directories until ``baselines/baselines.json`` declares a
fully succeeded three-difficulty baseline set.  It writes two derived files:

* ``baseline-summary.json`` -- machine-readable, audit-oriented facts;
* ``README.md`` -- a technical report for maintainers and evaluators.

Run from the repository root after ``run_live_baselines.py`` has completed::

    uv run python docs/evaluation-baselines/2026-08-10-three-difficulty/\
        summarize_baselines.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard")
SUMMARY_SCHEMA = "stt.eval.three-difficulty-baseline-summary.v1"
EXPECTED_BASELINES_SCHEMA = "stt.eval.live-baselines.v1"
EXPECTED_COPY_SCHEMA = "stt.eval.artifact-copy-manifest.v1"

SCRIPT_PATH = Path(__file__).resolve()
SUITE_ROOT = SCRIPT_PATH.parent
BASELINES_PATH = SUITE_ROOT / "baselines" / "baselines.json"
QUALITY_PATH = SUITE_ROOT / "quality" / "quality-summary.json"
OUTPUT_JSON_PATH = SUITE_ROOT / "baseline-summary.json"
OUTPUT_README_PATH = SUITE_ROOT / "README.md"

LIMITATIONS = (
    "correctness、faithfulness、citation 与 abstention 来自确定性词项/证据启发式，"
    "不是独立 LLM judge，也不等价于人工语义评审。",
    "cost 只有在运行时保存了供应商定价快照后才可计算；当前未定价时必须保留为 unknown，"
    "不能按 0 成本解释。",
    "tokens 与 cost 的 unknown 会从均值和总量中排除；部分覆盖时的 total 不是全集估算。",
    "首次 baseline 没有候选配对，因此 baseline_recorded 只表示事实基线已保存，"
    "baseline_recorded_with_failures 更不表示质量通过。",
    "每个难度固定 50 题，报告为描述性快照；未计算置信区间，也不能据此推断其他文档或流量。",
    "失败原因切片按当前 case-pass 启发式反推；运行时 error_type/error_detail 才是异常根因的"
    "权威记录。",
)

REASON_LABELS = {
    "abstention_incorrect": "拒答判断错误",
    "abstention_unknown": "拒答判断未采集",
    "expected_term_recall_below_0.75": "期望词召回低于 0.75",
    "expected_term_recall_unknown": "期望词召回未采集",
    "forbidden_term_hit": "命中禁止词",
    "retrieval_recall_at_10_zero": "Recall@10 为 0",
    "retrieval_recall_at_10_unknown": "Recall@10 未采集",
    "case_gate_failed_without_diagnostic": "case 未通过但无可推导诊断",
}


class BaselineSummaryError(RuntimeError):
    """A stable, user-facing post-processing failure."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _get(value: Any, *path: str) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineSummaryError(f"cannot read valid JSON: {_relative(path)}") from exc


def _write_text(path: Path, content: str) -> None:
    """Atomically replace one derived artifact after all inputs are validated."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text(path, content)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BaselineSummaryError(f"cannot hash artifact: {_relative(path)}") from exc
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(SUITE_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise BaselineSummaryError(f"path escapes suite root: {resolved}") from exc


def _suite_path(relative: Any, *, label: str) -> Path:
    raw = str(relative or "").strip()
    candidate = Path(raw)
    if not raw or candidate.is_absolute():
        raise BaselineSummaryError(f"{label} must be a suite-relative path")
    resolved = (SUITE_ROOT / candidate).resolve()
    _relative(resolved)
    return resolved


def _safe_component(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
        raise BaselineSummaryError(f"unsafe {label}: {normalized!r}")
    return normalized


def _expected_sha(value: Any, *, label: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise BaselineSummaryError(f"invalid SHA-256 for {label}")
    return normalized


def _artifact_fact(
    path: Path,
    *,
    role: str,
    expected_sha256: Any = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineSummaryError(f"missing required artifact: {_relative(path)}")
    expected = _expected_sha(expected_sha256, label=_relative(path))
    actual = _sha256_file(path)
    if expected is not None and expected != actual:
        raise BaselineSummaryError(f"artifact checksum mismatch: {_relative(path)}")
    return {
        "path": _relative(path),
        "role": role,
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "expected_sha256": expected,
        "verification": "matched_expected" if expected is not None else "calculated_only",
    }


def _envelope_data(payload: Any, *, path: Path) -> Any:
    if not isinstance(payload, Mapping) or "data" not in payload:
        raise BaselineSummaryError(f"API snapshot has no ResourceEnvelope.data: {_relative(path)}")
    return payload["data"]


def _evaluated_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = result.get("metrics")
    if isinstance(nested, Mapping) and isinstance(nested.get("metrics"), Mapping):
        return nested
    return result


def _metric_mean(summary: Mapping[str, Any], *path: str) -> float | None:
    return _finite(_get(summary, *path, "mean"))


def _coverage(summary: Mapping[str, Any], *path: str) -> dict[str, int]:
    value = _mapping(_get(summary, *path))
    return {
        "known_case_count": _integer(value.get("count")),
        "unknown_case_count": _integer(value.get("unknown_count")),
    }


def _telemetry_status(total: float | None, known: int, unknown: int) -> str:
    if total is None or known == 0:
        return "unknown"
    return "partial" if unknown else "known"


def _metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    token_summary = _mapping(_get(summary, "runtime", "total_tokens"))
    cost_summary = _mapping(_get(summary, "runtime", "cost"))
    token_total = _finite(token_summary.get("total"))
    cost_total = _finite(cost_summary.get("total"))
    token_known = _integer(token_summary.get("count"))
    token_unknown = _integer(token_summary.get("unknown_count"))
    cost_known = _integer(cost_summary.get("count"))
    cost_unknown = _integer(cost_summary.get("unknown_count"))

    abstention_counts = _mapping(summary.get("abstention"))
    false_positive = _integer(abstention_counts.get("false_positive"))
    true_negative = _integer(abstention_counts.get("true_negative"))
    fpr_denominator = false_positive + true_negative
    false_positive_rate = (
        false_positive / fpr_denominator if fpr_denominator else None
    )

    return {
        "retrieval": {
            "recall_at_1": _metric_mean(
                summary, "metrics", "retrieval", "recall_at_k", "1"
            ),
            "recall_at_3": _metric_mean(
                summary, "metrics", "retrieval", "recall_at_k", "3"
            ),
            "recall_at_10": _metric_mean(
                summary, "metrics", "retrieval", "recall_at_k", "10"
            ),
            "mrr": _metric_mean(summary, "metrics", "retrieval", "mrr"),
            "ndcg_at_10": _metric_mean(
                summary, "metrics", "retrieval", "ndcg_at_k", "10"
            ),
        },
        "generation": {
            "correctness": _metric_mean(
                summary, "metrics", "generation", "correctness"
            ),
            "faithfulness": _metric_mean(
                summary, "metrics", "generation", "faithfulness"
            ),
        },
        "citation": {
            "coverage": _metric_mean(summary, "metrics", "citation", "coverage"),
            "precision": _metric_mean(summary, "metrics", "citation", "precision"),
            "recall": _metric_mean(summary, "metrics", "citation", "recall"),
        },
        "abstention": {
            "accuracy": _metric_mean(
                summary, "metrics", "abstention", "accuracy"
            ),
            "false_positive_rate": false_positive_rate,
            "false_positive_rate_denominator": fpr_denominator,
            "count": _integer(abstention_counts.get("count")),
            "unknown_count": _integer(abstention_counts.get("unknown_count")),
            "true_positive": _integer(abstention_counts.get("true_positive")),
            "false_positive": false_positive,
            "false_negative": _integer(abstention_counts.get("false_negative")),
            "true_negative": true_negative,
        },
        "runtime": {
            "latency_ms": {
                "mean": _finite(_get(summary, "runtime", "latency_ms", "mean")),
                "p95": _finite(_get(summary, "runtime", "latency_ms", "p95")),
                **_coverage(summary, "runtime", "latency_ms"),
            },
            "total_tokens": {
                "total": token_total,
                "known_case_count": token_known,
                "unknown_case_count": token_unknown,
                "status": _telemetry_status(token_total, token_known, token_unknown),
            },
            "cost": {
                "total": cost_total,
                "currency": cost_summary.get("currency"),
                "known_case_count": cost_known,
                "unknown_case_count": cost_unknown,
                "status": _telemetry_status(cost_total, cost_known, cost_unknown),
            },
        },
    }


def _recall_at_10(evaluated: Mapping[str, Any]) -> float | None:
    recalls = _mapping(_get(evaluated, "metrics", "retrieval", "recall_at_k"))
    direct = _finite(recalls.get("10"))
    if direct is not None:
        return direct
    numeric_keys = sorted(
        (int(str(key)) for key in recalls if str(key).isdigit()),
        reverse=True,
    )
    if not numeric_keys:
        return None
    return _finite(recalls.get(str(numeric_keys[0])))


def _failure_reasons(
    result: Mapping[str, Any], evaluated: Mapping[str, Any]
) -> list[str]:
    error_type = str(result.get("error_type") or "").strip()
    error_detail = str(result.get("error_detail") or "").strip()
    if error_type or error_detail:
        return [f"runtime_error:{error_type or 'unknown'}"]

    reasons: list[str] = []
    if bool(evaluated.get("unanswerable")):
        abstention = _finite(_get(evaluated, "metrics", "abstention", "correct"))
        if abstention is None:
            reasons.append("abstention_unknown")
        elif abstention < 1.0:
            reasons.append("abstention_incorrect")
    else:
        expected_recall = _finite(
            _get(evaluated, "metrics", "generation", "expected_term_recall")
        )
        forbidden_rate = _finite(
            _get(evaluated, "metrics", "generation", "forbidden_term_hit_rate")
        )
        recall_10 = _recall_at_10(evaluated)
        if expected_recall is None:
            reasons.append("expected_term_recall_unknown")
        elif expected_recall < 0.75:
            reasons.append("expected_term_recall_below_0.75")
        if forbidden_rate is not None and forbidden_rate > 0:
            reasons.append("forbidden_term_hit")
        if recall_10 is None:
            reasons.append("retrieval_recall_at_10_unknown")
        elif recall_10 <= 0:
            reasons.append("retrieval_recall_at_10_zero")
    return reasons or ["case_gate_failed_without_diagnostic"]


def _one_line(value: Any, *, limit: int = 500) -> str | None:
    normalized = " ".join(str(value or "").split())
    if not normalized:
        return None
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _failures(
    results: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    failed = [result for result in results if result.get("passed") is False]
    failed_ids = {str(result.get("case_id") or "") for result in failed}
    critical_ids = {
        str(case_id)
        for case_id in _sequence(gate.get("critical_failed_cases"))
        if str(case_id)
    }
    if not critical_ids <= failed_ids:
        missing = sorted(critical_ids - failed_ids)
        raise BaselineSummaryError(
            "gate critical_failed_cases are absent from failed API results: "
            + ", ".join(missing)
        )

    details: list[dict[str, Any]] = []
    slices: dict[str, list[str]] = defaultdict(list)
    for result in sorted(failed, key=lambda item: str(item.get("case_id") or "")):
        case_id = str(result.get("case_id") or "")
        evaluated = _evaluated_result(result)
        reasons = _failure_reasons(result, evaluated)
        for reason in reasons:
            slices[reason].append(case_id)
        details.append(
            {
                "case_id": case_id,
                "critical": case_id in critical_ids,
                "reasons": reasons,
                "error_type": _one_line(result.get("error_type")),
                "error_detail": _one_line(result.get("error_detail")),
            }
        )

    reason_slices: dict[str, Any] = {}
    for reason in sorted(slices):
        case_ids = sorted(set(slices[reason]))
        reason_slices[reason] = {
            "label": _reason_label(reason),
            "count": len(case_ids),
            "case_ids": case_ids,
            "critical_case_ids": sorted(set(case_ids) & critical_ids),
            "ordinary_case_ids": sorted(set(case_ids) - critical_ids),
        }
    return {
        "critical_case_ids": sorted(critical_ids),
        "ordinary_case_ids": sorted(failed_ids - critical_ids),
        "cases": details,
        "reason_slices": reason_slices,
    }


def _reason_label(reason: str) -> str:
    if reason.startswith("runtime_error:"):
        return "运行时异常：" + reason.split(":", 1)[1]
    return REASON_LABELS.get(reason, reason)


def _index_version_ids(config_snapshot: Mapping[str, Any]) -> list[str]:
    scopes = _sequence(config_snapshot.get("vector_read_scopes"))
    if not scopes:
        scopes = _sequence(_get(config_snapshot, "preflight", "vector_read_scopes"))
    return sorted(
        {
            str(_mapping(scope).get("index_version_id") or "").strip()
            for scope in scopes
            if str(_mapping(scope).get("index_version_id") or "").strip()
        }
    )


def _run_artifacts(
    run_dir: Path,
    *,
    difficulty: str,
    run_id: str,
    baseline: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    manifest_ref = _mapping(baseline.get("artifact_copy_manifest"))
    manifest_path = _suite_path(
        manifest_ref.get("path") or _relative(run_dir / "artifact-copy-manifest.json"),
        label="artifact copy manifest",
    )
    expected_manifest_path = (run_dir / "artifact-copy-manifest.json").resolve()
    if manifest_path != expected_manifest_path:
        raise BaselineSummaryError(
            f"{difficulty} artifact copy manifest is outside its run directory"
        )
    manifest_fact = _artifact_fact(
        manifest_path,
        role="artifact_copy_manifest",
        expected_sha256=manifest_ref.get("sha256"),
    )
    copy_manifest = _read_json(manifest_path)
    if not isinstance(copy_manifest, Mapping):
        raise BaselineSummaryError(f"{difficulty} artifact copy manifest must be an object")
    if copy_manifest.get("schema_version") != EXPECTED_COPY_SCHEMA:
        raise BaselineSummaryError(f"{difficulty} artifact copy manifest schema is unsupported")
    if copy_manifest.get("difficulty") != difficulty or copy_manifest.get("run_id") != run_id:
        raise BaselineSummaryError(f"{difficulty} artifact copy manifest identity mismatch")

    expected_by_path: dict[str, str] = {}
    copies = _mapping(copy_manifest.get("copies"))
    for artifact_name, raw_copy in copies.items():
        copy = _mapping(raw_copy)
        target = _mapping(copy.get("target"))
        target_path = _suite_path(
            target.get("relative_path"), label=f"{difficulty} {artifact_name} target"
        )
        try:
            target_path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise BaselineSummaryError(
                f"{difficulty} {artifact_name} target escapes its run directory"
            ) from exc
        expected = _expected_sha(
            target.get("sha256"), label=f"{difficulty} {artifact_name} target"
        )
        if expected is None or copy.get("source_target_sha256_match") is not True:
            raise BaselineSummaryError(
                f"{difficulty} {artifact_name} has no verified source/target checksum"
            )
        expected_by_path[_relative(target_path)] = expected

    required = (
        run_dir / "api" / "results.json",
        run_dir / "summary.json",
        run_dir / "report.json",
    )
    for path in required:
        if not path.is_file():
            raise BaselineSummaryError(f"missing required run artifact: {_relative(path)}")

    facts: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=_relative,
    ):
        relative = _relative(path)
        role = "archived_run_artifact"
        if path == manifest_path:
            facts.append(manifest_fact)
            continue
        if relative.endswith("/api/results.json"):
            role = "api_results_snapshot"
        elif relative.endswith("/summary.json"):
            role = "aggregate_summary"
        elif relative.endswith("/report.json"):
            role = "full_json_report"
        facts.append(
            _artifact_fact(
                path,
                role=role,
                expected_sha256=expected_by_path.get(relative),
            )
        )
    missing_targets = sorted(set(expected_by_path) - {fact["path"] for fact in facts})
    if missing_targets:
        raise BaselineSummaryError(
            f"{difficulty} archived targets are missing: {', '.join(missing_targets)}"
        )
    return facts, copy_manifest


def _load_difficulty(
    difficulty: str,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.get("difficulty") != difficulty:
        raise BaselineSummaryError(f"{difficulty} baseline difficulty mismatch")
    run_id = _safe_component(baseline.get("run_id"), label=f"{difficulty} run id")
    run_dir = (SUITE_ROOT / "runs" / difficulty / run_id).resolve()
    _relative(run_dir)
    declared_run_dir = str(baseline.get("run_directory") or "").strip()
    if declared_run_dir and _suite_path(
        declared_run_dir, label=f"{difficulty} run directory"
    ) != run_dir:
        raise BaselineSummaryError(f"{difficulty} baseline run directory mismatch")
    if baseline.get("status") != "succeeded":
        raise BaselineSummaryError(f"{difficulty} baseline run is not succeeded")

    artifact_facts, _ = _run_artifacts(
        run_dir,
        difficulty=difficulty,
        run_id=run_id,
        baseline=baseline,
    )
    api_results_path = run_dir / "api" / "results.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.json"
    api_results = _envelope_data(_read_json(api_results_path), path=api_results_path)
    if not isinstance(api_results, list):
        raise BaselineSummaryError(f"{difficulty} API results data must be a list")
    results: list[Mapping[str, Any]] = []
    for result in api_results:
        if not isinstance(result, Mapping):
            raise BaselineSummaryError(f"{difficulty} API result must be an object")
        if not isinstance(result.get("passed"), bool):
            raise BaselineSummaryError(
                f"{difficulty} result {result.get('case_id')} has non-boolean passed"
            )
        results.append(result)

    summary = _read_json(summary_path)
    report = _read_json(report_path)
    if not isinstance(summary, Mapping) or not isinstance(report, Mapping):
        raise BaselineSummaryError(f"{difficulty} summary/report must be JSON objects")
    if report.get("summary") != summary:
        raise BaselineSummaryError(f"{difficulty} report summary differs from summary.json")
    baseline_metrics = baseline.get("metrics_summary")
    if isinstance(baseline_metrics, Mapping) and baseline_metrics and baseline_metrics != summary:
        raise BaselineSummaryError(
            f"{difficulty} baseline metrics_summary differs from summary.json"
        )
    report_results = _sequence(report.get("results"))
    api_case_ids = [str(item.get("case_id") or "") for item in results]
    report_case_ids = [str(_mapping(item).get("case_id") or "") for item in report_results]
    if api_case_ids != report_case_ids or len(api_case_ids) != len(set(api_case_ids)):
        raise BaselineSummaryError(f"{difficulty} API/report result case sets differ or duplicate")

    run = _mapping(report.get("run"))
    if run.get("id") != run_id or run.get("status") != "succeeded":
        raise BaselineSummaryError(f"{difficulty} report run identity/status mismatch")
    mode = str(run.get("mode") or baseline.get("mode") or "")
    if mode != "live" or report.get("evaluation_scope") != "full":
        raise BaselineSummaryError(f"{difficulty} report is not a full/live run")
    if run.get("baseline_run_id") is not None or baseline.get("baseline_run_id") is not None:
        raise BaselineSummaryError(f"{difficulty} run is not a first baseline")

    gate = _mapping(report.get("gate")) or _mapping(baseline.get("gate_result"))
    baseline_gate = _mapping(baseline.get("gate_result"))
    if baseline_gate and dict(gate) != dict(baseline_gate):
        raise BaselineSummaryError(f"{difficulty} baseline/report gate results differ")
    if gate.get("status") not in {
        "baseline_recorded",
        "baseline_recorded_with_failures",
    }:
        raise BaselineSummaryError(f"{difficulty} has an unexpected first-baseline gate")

    dataset = _mapping(baseline.get("dataset"))
    case_count = len(results)
    declared_counts = {
        _integer(summary.get("case_count"), default=-1),
        _integer(dataset.get("case_count"), default=-1),
        _integer(baseline.get("result_count"), default=-1),
        _integer(run.get("total_cases"), default=-1),
        _integer(run.get("completed_cases"), default=-1),
    }
    if declared_counts != {case_count} or case_count < 50:
        raise BaselineSummaryError(
            f"{difficulty} case counts do not agree or contain fewer than 50 cases"
        )
    passed_count = sum(result.get("passed") is True for result in results)
    failed_count = case_count - passed_count
    error_count = sum(
        bool(result.get("error_type") or result.get("error_detail")) for result in results
    )
    if any(
        bool(result.get("error_type") or result.get("error_detail"))
        and result.get("passed") is True
        for result in results
    ):
        raise BaselineSummaryError(f"{difficulty} contains a passed case with runtime error")

    config_snapshot = _mapping(run.get("config_snapshot")) or _mapping(
        baseline.get("config_snapshot")
    )
    source_path = _suite_path(
        dataset.get("source_path"), label=f"{difficulty} dataset source"
    )
    source_fact = _artifact_fact(
        source_path,
        role="dataset_manifest",
        expected_sha256=dataset.get("source_sha256"),
    )
    artifact_facts.append(source_fact)

    return {
        "difficulty": difficulty,
        "run_id": run_id,
        "dataset": {
            "id": dataset.get("id"),
            "key": dataset.get("key"),
            "version": dataset.get("version"),
            "checksum": dataset.get("checksum"),
            "case_count": case_count,
            "source_path": source_fact["path"],
            "source_sha256": source_fact["sha256"],
        },
        "cases": {
            "total": case_count,
            "passed": passed_count,
            "failed": failed_count,
            "error": error_count,
        },
        "gate": {
            "status": gate.get("status"),
            "passed": gate.get("passed"),
            "reason": gate.get("reason"),
            "policy_version": gate.get("policy_version"),
            "critical_failed_cases": sorted(
                str(case_id)
                for case_id in _sequence(gate.get("critical_failed_cases"))
            ),
            "failures": list(_sequence(gate.get("failures"))),
            "interpretation": (
                "first baseline recorded; this is not a paired candidate quality pass"
                if gate.get("status") == "baseline_recorded"
                else "first baseline recorded with critical case failures; quality did not pass"
            ),
        },
        "metrics": _metrics(summary),
        "failures": _failures(results, gate),
        "runtime_identity": {
            "kb_id": baseline.get("kb_id") or run.get("kb_id"),
            "index_version_ids": _index_version_ids(config_snapshot),
            "llm_model": config_snapshot.get("llm_model"),
            "mode": mode,
            "code_revision": config_snapshot.get("code_revision"),
            "vector_read_set_digest": config_snapshot.get("vector_read_set_digest"),
        },
        "started_at": baseline.get("started_at") or run.get("started_at"),
        "completed_at": baseline.get("completed_at") or run.get("completed_at"),
        "artifacts": sorted(artifact_facts, key=lambda item: item["path"]),
    }


def _shared_scalar(
    difficulties: Mapping[str, Mapping[str, Any]], key: str
) -> dict[str, Any]:
    by_difficulty = {
        difficulty: _get(details, "runtime_identity", key)
        for difficulty, details in difficulties.items()
    }
    normalized = [str(value or "").strip() for value in by_difficulty.values()]
    values = sorted({value for value in normalized if value})
    consistent = len(values) == 1 and all(normalized)
    return {
        "consistent": consistent,
        "value": values[0] if consistent else None,
        "values": values,
        "by_difficulty": by_difficulty,
    }


def _shared_runtime(difficulties: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scalar = {
        key: _shared_scalar(difficulties, key)
        for key in ("kb_id", "llm_model", "mode")
    }
    indexes_by_difficulty = {
        difficulty: list(
            _sequence(_get(details, "runtime_identity", "index_version_ids"))
        )
        for difficulty, details in difficulties.items()
    }
    index_sets = [tuple(values) for values in indexes_by_difficulty.values()]
    indexes_consistent = bool(index_sets) and bool(index_sets[0]) and len(set(index_sets)) == 1
    indexes = {
        "consistent": indexes_consistent,
        "value": list(index_sets[0]) if indexes_consistent else None,
        "by_difficulty": indexes_by_difficulty,
    }
    return {
        **scalar,
        "index_version_ids": indexes,
        "all_consistent": all(item["consistent"] for item in scalar.values())
        and indexes_consistent,
    }


def _quality_audit(
    baselines: Mapping[str, Any],
    difficulties: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not QUALITY_PATH.is_file():
        raise BaselineSummaryError("quality/quality-summary.json is required")
    quality = _read_json(QUALITY_PATH)
    if not isinstance(quality, Mapping):
        raise BaselineSummaryError("quality summary must be an object")
    if quality.get("suite_key") != baselines.get("suite_key"):
        raise BaselineSummaryError("quality summary suite_key differs from baselines")

    artifact_facts = [_artifact_fact(QUALITY_PATH, role="quality_summary")]
    for key in ("deterministic_report", "online_audit_report", "adjudication_report"):
        relative = quality.get(key)
        if relative:
            artifact_facts.append(
                _artifact_fact(_suite_path(relative, label=key), role=key)
            )

    expected_checksums = _mapping(quality.get("dataset_checksums"))
    checksum_matches: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        expected = _expected_sha(
            expected_checksums.get(difficulty), label=f"quality {difficulty} dataset"
        )
        actual = str(_get(difficulties, difficulty, "dataset", "source_sha256") or "")
        checksum_matches[difficulty] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": expected is not None and expected == actual,
        }
    passed = (
        quality.get("ready_for_import") is True
        and quality.get("deterministic_status") == "passed"
        and quality.get("online_audit_status")
        in {"passed", "passed_after_adjudication"}
        and all(item["matched"] for item in checksum_matches.values())
    )
    return (
        {
            "path": _relative(QUALITY_PATH),
            "sha256": artifact_facts[0]["sha256"],
            "suite_key": quality.get("suite_key"),
            "ready_for_import": quality.get("ready_for_import"),
            "deterministic_status": quality.get("deterministic_status"),
            "online_audit_status": quality.get("online_audit_status"),
            "dataset_checksum_matches": checksum_matches,
            "passed": passed,
        },
        artifact_facts,
    )


def _deduplicate_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        path = str(artifact.get("path") or "")
        current = by_path.get(path)
        value = dict(artifact)
        if current is not None and current.get("sha256") != value.get("sha256"):
            raise BaselineSummaryError(f"conflicting checksums recorded for {path}")
        if current is None or value.get("expected_sha256"):
            by_path[path] = value
    return [by_path[path] for path in sorted(by_path)]


def build_summary() -> dict[str, Any]:
    if not BASELINES_PATH.is_file():
        raise BaselineSummaryError("baselines/baselines.json does not exist")
    baselines = _read_json(BASELINES_PATH)
    if not isinstance(baselines, Mapping):
        raise BaselineSummaryError("baselines.json must be an object")
    if baselines.get("schema_version") != EXPECTED_BASELINES_SCHEMA:
        raise BaselineSummaryError("baselines.json schema is unsupported")
    if baselines.get("status") != "succeeded":
        raise BaselineSummaryError(
            "baseline workflow is not fully succeeded; refusing to read run outputs"
        )
    if baselines.get("mode") != "live":
        raise BaselineSummaryError("baseline workflow mode must be live")
    order = tuple(str(item) for item in _sequence(baselines.get("difficulty_order")))
    records = _mapping(baselines.get("baselines"))
    if order != DIFFICULTIES or set(records) != set(DIFFICULTIES):
        raise BaselineSummaryError("baselines.json must contain easy, medium and hard")

    difficulty_summaries = {
        difficulty: _load_difficulty(difficulty, _mapping(records[difficulty]))
        for difficulty in DIFFICULTIES
    }
    shared_runtime = _shared_runtime(difficulty_summaries)
    quality_audit, quality_artifacts = _quality_audit(
        baselines, difficulty_summaries
    )
    artifacts: list[Mapping[str, Any]] = [
        _artifact_fact(BASELINES_PATH, role="baseline_registry"),
        _artifact_fact(SCRIPT_PATH, role="summarizer_source"),
        *quality_artifacts,
    ]
    for details in difficulty_summaries.values():
        artifacts.extend(_sequence(details.get("artifacts")))
    artifacts = _deduplicate_artifacts(artifacts)
    expected_artifacts = [
        artifact for artifact in artifacts if artifact.get("expected_sha256")
    ]
    artifact_integrity = {
        "all_expected_checksums_verified": bool(expected_artifacts)
        and all(
            artifact.get("verification") == "matched_expected"
            for artifact in expected_artifacts
        ),
        "expected_checksum_count": len(expected_artifacts),
        "calculated_only_count": sum(
            artifact.get("verification") == "calculated_only"
            for artifact in artifacts
        ),
        "files": artifacts,
    }

    kb_id = _get(shared_runtime, "kb_id", "value") or "<KB_ID>"
    suite_script = "docs/evaluation-baselines/2026-08-10-three-difficulty"
    return {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at": _utc_now(),
        "suite_key": baselines.get("suite_key"),
        "source_baselines": {
            "path": _relative(BASELINES_PATH),
            "sha256": _sha256_file(BASELINES_PATH),
            "generated_at": baselines.get("generated_at"),
            "status": baselines.get("status"),
        },
        "difficulty_order": list(DIFFICULTIES),
        "difficulties": difficulty_summaries,
        "shared_runtime": shared_runtime,
        "quality_audit": quality_audit,
        "artifact_integrity": artifact_integrity,
        "reproducibility": {
            "working_directory": "repository root",
            "commands": [
                f"uv run python {suite_script}/validate_suite.py --online-audit",
                f"uv run python {suite_script}/run_live_baselines.py --kb-id {kb_id} --overwrite",
                f"uv run python {suite_script}/summarize_baselines.py",
            ],
            "resume_command": (
                f"uv run python {suite_script}/run_live_baselines.py "
                f"--kb-id {kb_id} --resume"
            ),
        },
        "limitations": list(LIMITATIONS),
    }


def _md(value: Any) -> str:
    return (
        str(value if value is not None else "—")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", "<br>")
    )


def _fmt_metric(value: Any) -> str:
    number = _finite(value)
    return "unknown" if number is None else f"{number:.4f}"


def _fmt_number(value: Any, *, digits: int = 2) -> str:
    number = _finite(value)
    if number is None:
        return "unknown"
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _ids(values: Any) -> str:
    normalized = [str(value) for value in _sequence(values) if str(value)]
    return ", ".join(normalized) if normalized else "—"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_md(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_md(value) for value in row) + " |" for row in rows)
    return lines


def render_readme(payload: Mapping[str, Any]) -> str:
    difficulties = _mapping(payload.get("difficulties"))
    shared = _mapping(payload.get("shared_runtime"))
    quality = _mapping(payload.get("quality_audit"))
    integrity = _mapping(payload.get("artifact_integrity"))
    runs_ok = all(
        _get(difficulties, difficulty, "cases", "total") == 50
        and _get(difficulties, difficulty, "run_id")
        for difficulty in DIFFICULTIES
    )
    gate_statuses = ", ".join(
        f"{difficulty}={_get(difficulties, difficulty, 'gate', 'status')}"
        for difficulty in DIFFICULTIES
    )
    failure_counts = ", ".join(
        f"{difficulty}={_get(difficulties, difficulty, 'cases', 'failed')}"
        for difficulty in DIFFICULTIES
    )

    lines = [
        "# 三难度在线评测基线技术报告",
        "",
        "> 本文件由 `summarize_baselines.py` 从已归档制品生成，请勿手工改写数值。",
        "",
        "## 技术摘要",
        "",
        f"- 三个 full/live run 与各 50 题制品完整性：**{'通过' if runs_ok else '未通过'}**；"
        f"gate 状态为 `{_md(gate_statuses)}`。",
        f"- case 失败数为 `{_md(failure_counts)}`；失败 case 会在后文按关键/普通和原因切片列出。",
        f"- 同一 KB、index、模型与 mode 一致性审计："
        f"**{'通过' if shared.get('all_consistent') else '未通过'}**。",
        f"- 数据集确定性/在线质量审计：**{'通过' if quality.get('passed') else '未通过'}**；"
        f"有期望值的制品校验和："
        f"**{'全部匹配' if integrity.get('all_expected_checksums_verified') else '存在缺口'}**。",
        "- 这些是首次事实基线，不是候选相对基线的 paired quality gate 结论。",
        "",
        "## 运行与门禁事实",
        "",
    ]
    lines.extend(
        _table(
            (
                "难度",
                "dataset",
                "version",
                "run_id",
                "cases",
                "passed",
                "failed",
                "error",
                "gate",
                "gate passed",
            ),
            [
                (
                    difficulty,
                    _get(difficulties, difficulty, "dataset", "key"),
                    _get(difficulties, difficulty, "dataset", "version"),
                    _get(difficulties, difficulty, "run_id"),
                    _get(difficulties, difficulty, "cases", "total"),
                    _get(difficulties, difficulty, "cases", "passed"),
                    _get(difficulties, difficulty, "cases", "failed"),
                    _get(difficulties, difficulty, "cases", "error"),
                    _get(difficulties, difficulty, "gate", "status"),
                    _fmt_bool(_get(difficulties, difficulty, "gate", "passed")),
                )
                for difficulty in DIFFICULTIES
            ],
        )
    )
    lines.extend(["", "## 三种难度的质量指标", ""])
    lines.extend(
        _table(
            (
                "难度",
                "Recall@1",
                "Recall@3",
                "Recall@10",
                "MRR",
                "nDCG@10",
                "correctness",
                "faithfulness",
            ),
            [
                (
                    difficulty,
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "retrieval", "recall_at_1")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "retrieval", "recall_at_3")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "retrieval", "recall_at_10")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "retrieval", "mrr")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "retrieval", "ndcg_at_10")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "generation", "correctness")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "generation", "faithfulness")),
                )
                for difficulty in DIFFICULTIES
            ],
        )
    )
    lines.extend(["", "引用与拒答指标：", ""])
    lines.extend(
        _table(
            (
                "难度",
                "citation coverage",
                "citation precision",
                "citation recall",
                "abstention accuracy",
                "abstention FPR",
                "FPR denominator",
            ),
            [
                (
                    difficulty,
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "citation", "coverage")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "citation", "precision")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "citation", "recall")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "abstention", "accuracy")),
                    _fmt_metric(_get(difficulties, difficulty, "metrics", "abstention", "false_positive_rate")),
                    _get(difficulties, difficulty, "metrics", "abstention", "false_positive_rate_denominator"),
                )
                for difficulty in DIFFICULTIES
            ],
        )
    )
    lines.extend(["", "运行遥测：", ""])
    lines.extend(
        _table(
            (
                "难度",
                "latency mean (ms)",
                "latency p95 (ms)",
                "tokens total",
                "tokens unknown cases",
                "cost total",
                "cost status",
            ),
            [
                (
                    difficulty,
                    _fmt_number(_get(difficulties, difficulty, "metrics", "runtime", "latency_ms", "mean")),
                    _fmt_number(_get(difficulties, difficulty, "metrics", "runtime", "latency_ms", "p95")),
                    _fmt_number(_get(difficulties, difficulty, "metrics", "runtime", "total_tokens", "total"), digits=0),
                    _get(difficulties, difficulty, "metrics", "runtime", "total_tokens", "unknown_case_count"),
                    _fmt_number(_get(difficulties, difficulty, "metrics", "runtime", "cost", "total"), digits=6),
                    _get(difficulties, difficulty, "metrics", "runtime", "cost", "status"),
                )
                for difficulty in DIFFICULTIES
            ],
        )
    )

    lines.extend(["", "## 失败 case 与原因切片", ""])
    for difficulty in DIFFICULTIES:
        failures = _mapping(_get(difficulties, difficulty, "failures"))
        lines.extend(
            [
                f"### {difficulty}",
                "",
                f"- 关键失败：`{_md(_ids(failures.get('critical_case_ids')))}`",
                f"- 普通失败：`{_md(_ids(failures.get('ordinary_case_ids')))}`",
                "",
            ]
        )
        slices = _mapping(failures.get("reason_slices"))
        if slices:
            lines.extend(
                _table(
                    ("原因", "数量", "关键 case", "普通 case"),
                    [
                        (
                            _get(slices, reason, "label"),
                            _get(slices, reason, "count"),
                            _ids(_get(slices, reason, "critical_case_ids")),
                            _ids(_get(slices, reason, "ordinary_case_ids")),
                        )
                        for reason in sorted(slices)
                    ],
                )
            )
            lines.append("")
        else:
            lines.extend(["无失败 case。", ""])
        case_details = _sequence(failures.get("cases"))
        if case_details:
            lines.extend(
                _table(
                    ("case_id", "级别", "原因", "运行时错误"),
                    [
                        (
                            _mapping(item).get("case_id"),
                            "critical" if _mapping(item).get("critical") else "ordinary",
                            ", ".join(
                                _reason_label(str(reason))
                                for reason in _sequence(_mapping(item).get("reasons"))
                            ),
                            _mapping(item).get("error_detail") or _mapping(item).get("error_type") or "—",
                        )
                        for item in case_details
                    ],
                )
            )
            lines.append("")

    lines.extend(["## 同一运行上下文与质量审计", ""])
    runtime_rows = []
    for key, label in (
        ("kb_id", "KB"),
        ("index_version_ids", "index version(s)"),
        ("llm_model", "LLM model"),
        ("mode", "evaluation mode"),
    ):
        item = _mapping(shared.get(key))
        value = item.get("value")
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = ", ".join(str(part) for part in value)
        runtime_rows.append((label, value or "—", _fmt_bool(item.get("consistent"))))
    lines.extend(_table(("事实", "共享值", "三难度一致"), runtime_rows))
    lines.extend(
        [
            "",
            f"质量审计文件：`{_md(quality.get('path'))}`（SHA-256 `{_md(quality.get('sha256'))}`）。",
            "",
        ]
    )
    lines.extend(
        _table(
            ("检查", "结果"),
            (
                ("ready_for_import", _fmt_bool(quality.get("ready_for_import"))),
                ("deterministic_status", quality.get("deterministic_status")),
                ("online_audit_status", quality.get("online_audit_status")),
                ("dataset checksum matches", _fmt_bool(quality.get("passed"))),
            ),
        )
    )

    lines.extend(
        [
            "",
            "## 指标定义与方法",
            "",
            "- Recall@K、MRR 与 nDCG@10 基于标注相关证据在检索排名中的位置；表中为已知 case 的均值。",
            "- correctness、faithfulness 与 citation 指标由当前确定性评测器计算；unknown 不按 0 计入。",
            "- abstention accuracy 是已知拒答判断的正确率；FPR = false_positive / "
            "(false_positive + true_negative)，分母为可回答且拒答判断已知的 case。",
            "- live answerable case 的 `passed` 由期望词召回 ≥ 0.75、禁止词命中率为 0，"
            "以及可用时 Recall@10 > 0 的组合启发式决定；unanswerable case 由拒答正确性决定。",
            "- latency mean/p95 只使用已采集值；tokens/cost 同时报告 unknown case 数，避免把缺失值当 0。",
            "",
            "## 制品校验和",
            "",
            f"共有 {len(_sequence(integrity.get('files')))} 个输入制品；"
            f"{integrity.get('expected_checksum_count')} 个带期望 SHA-256，"
            f"{integrity.get('calculated_only_count')} 个仅记录本次计算值。",
            "",
        ]
    )
    lines.extend(
        _table(
            ("路径", "角色", "SHA-256", "验证"),
            [
                (
                    _mapping(item).get("path"),
                    _mapping(item).get("role"),
                    _mapping(item).get("sha256"),
                    _mapping(item).get("verification"),
                )
                for item in _sequence(integrity.get("files"))
            ],
        )
    )

    commands = _sequence(_get(payload, "reproducibility", "commands"))
    lines.extend(
        [
            "",
            "## 重现步骤",
            "",
            "从仓库根目录执行；在线基线会产生 LLM 调用费用：",
            "",
            "```powershell",
            *[str(command) for command in commands],
            "```",
            "",
            "中断续跑使用：",
            "",
            "```powershell",
            str(_get(payload, "reproducibility", "resume_command")),
            "```",
            "",
            "## 局限与解释边界",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in _sequence(payload.get("limitations")))
    lines.extend(
        [
            "",
            "## 建议的下一步",
            "",
            "1. 优先复核 critical failure 的检索证据、生成答案与引用绑定，再处理普通失败。",
            "2. 固化供应商计价版本与全链路 token collector 后，重新生成可审计的 cost 指标。",
            "3. 以这三次 run 为固定 baseline，分别运行同 case 集 candidate，使用 paired gate 判断回归。",
            "4. 若要形成语义质量结论，增加盲审人工样本或独立 LLM judge，并与当前启发式并列报告。",
            "",
            "## 待回答问题",
            "",
            "- 后续 candidate 是否继续固定相同 index read set 与模型版本，还是另建版本化基线？",
            "- 哪些 failure slice 应升级为发布阻断，哪些只进入质量改进队列？",
            "- token/cost 覆盖率达到多少后才允许用于难度间成本比较？",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        summary = build_summary()
        readme = render_readme(summary)
        _write_json(OUTPUT_JSON_PATH, summary)
        _write_text(OUTPUT_README_PATH, readme)
    except BaselineSummaryError as exc:
        print(f"Baseline summary failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {_relative(OUTPUT_JSON_PATH)} and {_relative(OUTPUT_README_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
