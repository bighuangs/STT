from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REMEDIATION_ROOT = HERE.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "pyproject.toml").is_file())

ORDER = ("easy", "medium", "hard")
LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
ROLE_LABELS = {"baseline": "原始基线", "final": "最终候选"}
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_REGISTRY = SUITE_ROOT / "baselines" / "baselines.json"
DEFAULT_DIAGNOSTICS = REMEDIATION_ROOT / "diagnostics" / "failure-diagnostics.json"
DEFAULT_JUDGE_SUMMARY = REMEDIATION_ROOT / "diagnostics" / "llm-review-summary.json"

DATASET_SQL = {
    "case_outcomes": (
        'SELECT "order", difficulty_run, difficulty, role, passed, failed, error, total '
        'FROM case_outcomes ORDER BY "order";'
    ),
    "run_facts": 'SELECT * FROM run_facts ORDER BY "order";',
    "quality_metrics": 'SELECT * FROM quality_metrics ORDER BY "order";',
    "runtime_telemetry": 'SELECT * FROM runtime_telemetry ORDER BY "order";',
    "identity_facts": 'SELECT * FROM identity_facts ORDER BY "order";',
    "diagnostic_attribution": 'SELECT * FROM diagnostic_attribution ORDER BY case_count DESC, "primary" ASC;',
    "diagnostic_semantic": "SELECT * FROM diagnostic_semantic ORDER BY case_count DESC, status ASC;",
    "diagnostic_failures": (
        "SELECT * FROM diagnostic_failures ORDER BY critical_order DESC, difficulty ASC, case_id ASC;"
    ),
    "final_failure_cases": (
        "SELECT * FROM final_failure_cases ORDER BY critical_order DESC, difficulty ASC, case_id ASC;"
    ),
    "semantic_judge": 'SELECT * FROM semantic_judge ORDER BY "order";',
    "validation_checks": 'SELECT * FROM validation_checks ORDER BY "order";',
    "input_inventory": 'SELECT * FROM input_inventory ORDER BY "order";',
}

QUERY_LABELS = {
    "case_outcomes": "Case outcome composition query",
    "run_facts": "Run and gate facts query",
    "quality_metrics": "Quality metric query",
    "runtime_telemetry": "Runtime telemetry query",
    "identity_facts": "Comparison identity query",
    "diagnostic_attribution": "Baseline diagnostic attribution query",
    "diagnostic_semantic": "Baseline semantic-proxy query",
    "diagnostic_failures": "Baseline case-diagnostic query",
    "final_failure_cases": "Final candidate failure-reason query",
    "semantic_judge": "Semantic judge audit query",
    "validation_checks": "Input validation query",
    "input_inventory": "Input inventory query",
}


class ReportInputError(RuntimeError):
    """Raised when a report input is incomplete, stale, or identity-incompatible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportInputError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON is missing: {source_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {source_path(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ReportInputError(f"report inputs must remain inside the repository: {resolved}") from exc


def resolve_input(value: str | None, default: Path) -> Path:
    if value is None:
        path = default
    else:
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    source_path(path)
    return path.resolve()


def resolve_scoped_path(scope_root: Path, value: str, *, label: str) -> Path:
    candidate = (scope_root / value).resolve()
    try:
        candidate.relative_to(scope_root.resolve())
    except ValueError as exc:
        raise ReportInputError(f"{label} escapes its archive root: {value}") from exc
    return candidate


def short(value: Any, width: int = 16) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= width else f"{text[:width]}…"


def wrap_hint(value: Any) -> str:
    return str(value).replace("/", "/\u200b").replace("_", "_\u200b").replace("-", "-\u200b")


def display_path(value: str) -> str:
    path = Path(value)
    parts = path.parts
    compact = value if len(value) <= 72 else f"…/{'/'.join(parts[-3:])}"
    return wrap_hint(compact)


def json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row {line_number} is not an object: {source_path(path)}")
            payload = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
            rows.append(
                {
                    "case_id": value.get("case_id"),
                    "run_id": value.get("run_id"),
                    "passed": value.get("passed") is True,
                    "error_type": value.get("error_type"),
                    "error_detail": value.get("error_detail"),
                    "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
                }
            )
    return rows


def nested(value: dict[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def sqlite_project(table_name: str, rows: list[dict[str, Any]], sql: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    columns = list(rows[0])
    require(all(list(row) == columns for row in rows), f"{table_name}: projected rows have inconsistent columns")
    definitions: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        sqlite_type = "REAL" if values and all(isinstance(value, (bool, int, float)) for value in values) else "TEXT"
        definitions.append(f'"{column}" {sqlite_type}')
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(f'CREATE TABLE "{table_name}" ({", ".join(definitions)})')
        placeholders = ", ".join("?" for _ in columns)
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        connection.executemany(
            f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
            [[row.get(column) for column in columns] for row in rows],
        )
        return [dict(row) for row in connection.execute(sql)]
    finally:
        connection.close()


def mean_metric(summary: dict[str, Any], *parts: str) -> float | None:
    value = nested(summary, "metrics", *parts)
    if isinstance(value, dict):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def runtime_stat(summary: dict[str, Any], family: str, statistic: str) -> float | int | None:
    value = nested(summary, "runtime", family, statistic)
    return value if isinstance(value, (int, float)) else None


def dataset_value(entry: dict[str, Any], key: str) -> Any:
    dataset = entry.get("dataset") if isinstance(entry.get("dataset"), dict) else {}
    return entry.get(f"dataset_{key}", dataset.get(key))


def outcome_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(row["passed"] for row in results)
    errors = sum(bool(row.get("error_type") or row.get("error_detail")) for row in results)
    return {"passed": passed, "failed": len(results) - passed - errors, "error": errors, "total": len(results)}


def active_index_ids(entry: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    config = entry.get("config_snapshot") if isinstance(entry.get("config_snapshot"), dict) else {}
    indexes = nested(config, "preflight", "active_indexes")
    if isinstance(indexes, list):
        values = [str(item.get("id")) for item in indexes if isinstance(item, dict) and item.get("id")]
        if values:
            return sorted(set(values))
    values: set[str] = set()
    for result in results:
        versions = nested(result, "metrics", "versions")
        if isinstance(versions, dict) and isinstance(versions.get("index_version_ids"), list):
            values.update(str(item) for item in versions["index_version_ids"] if item)
    return sorted(values)


def model_id(entry: dict[str, Any], results: list[dict[str, Any]]) -> str:
    config = entry.get("config_snapshot") if isinstance(entry.get("config_snapshot"), dict) else {}
    if config.get("llm_model"):
        return str(config["llm_model"])
    for result in results:
        value = nested(result, "metrics", "versions", "model_id")
        if value:
            return str(value)
    return "unknown"


def verify_copy_manifest(
    *,
    entry: dict[str, Any],
    scope_root: Path,
    artifact_dir: Path,
    difficulty: str,
    run_id: str,
    inventory: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    reference = entry.get("artifact_copy_manifest")
    require(isinstance(reference, dict), f"{difficulty}: artifact_copy_manifest reference is missing")
    reference_path = reference.get("path")
    require(isinstance(reference_path, str), f"{difficulty}: artifact_copy_manifest.path is missing")
    manifest_path = resolve_scoped_path(scope_root, reference_path, label=f"{difficulty} manifest path")
    require(manifest_path.is_file(), f"{difficulty}: copy manifest is missing: {source_path(manifest_path)}")
    actual_manifest_hash = sha256_file(manifest_path)
    require(
        actual_manifest_hash == reference.get("sha256"),
        f"{difficulty}: copy manifest SHA-256 does not match its registry/index reference",
    )
    manifest = read_json(manifest_path)
    require(manifest.get("run_id") == run_id, f"{difficulty}: copy manifest run_id mismatch")
    require(manifest.get("difficulty") == difficulty, f"{difficulty}: copy manifest difficulty mismatch")
    inventory.append(
        {
            "role": "artifact_copy_manifest",
            "difficulty": difficulty,
            "path": source_path(manifest_path),
            "sha256": actual_manifest_hash,
        }
    )

    resolved: dict[str, Path] = {}
    copies = manifest.get("copies") if isinstance(manifest.get("copies"), dict) else {}
    for key, filename in (("summary", "summary.json"), ("results", "results.jsonl")):
        copy = copies.get(key)
        require(isinstance(copy, dict), f"{difficulty}: {key} copy receipt is missing")
        require(copy.get("source_target_sha256_match") is True, f"{difficulty}: {key} archive receipt is not verified")
        target = copy.get("target") if isinstance(copy.get("target"), dict) else {}
        target_path_value = target.get("relative_path")
        require(isinstance(target_path_value, str), f"{difficulty}: {key} target path is missing")
        target_path = resolve_scoped_path(scope_root, target_path_value, label=f"{difficulty} {key} target")
        expected_path = (artifact_dir / filename).resolve()
        require(target_path == expected_path, f"{difficulty}: {key} receipt points outside the declared run directory")
        require(target_path.is_file(), f"{difficulty}: archived {key} file is missing")
        actual_hash = sha256_file(target_path)
        require(actual_hash == target.get("sha256"), f"{difficulty}: archived {key} SHA-256 mismatch")
        resolved[key] = target_path
        inventory.append(
            {
                "role": key,
                "difficulty": difficulty,
                "path": source_path(target_path),
                "sha256": actual_hash,
            }
        )
    return manifest_path, resolved["summary"], resolved["results"]


def load_record(
    *,
    entry: dict[str, Any],
    scope_root: Path,
    difficulty: str,
    role: str,
    expected_case_count: int,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    run_id = entry.get("run_id")
    require(isinstance(run_id, str) and run_id, f"{difficulty}: run_id is missing")
    run_directory = entry.get("run_directory")
    require(isinstance(run_directory, str), f"{difficulty}: run_directory is missing")
    artifact_dir = resolve_scoped_path(scope_root, run_directory, label=f"{difficulty} run_directory")
    require(artifact_dir.is_dir(), f"{difficulty}: archived run directory is missing: {source_path(artifact_dir)}")
    manifest_path, summary_path, results_path = verify_copy_manifest(
        entry=entry,
        scope_root=scope_root,
        artifact_dir=artifact_dir,
        difficulty=difficulty,
        run_id=run_id,
        inventory=inventory,
    )
    summary = read_json(summary_path)
    results = json_rows(results_path)
    require(summary.get("case_count") == expected_case_count, f"{difficulty}: summary case_count mismatch")
    require(len(results) == expected_case_count, f"{difficulty}: result row count mismatch")
    case_ids = [row.get("case_id") for row in results]
    require(all(isinstance(case_id, str) and case_id for case_id in case_ids), f"{difficulty}: blank case_id")
    require(len(set(case_ids)) == expected_case_count, f"{difficulty}: result case_ids are not unique")
    require(all(row.get("run_id") == run_id for row in results), f"{difficulty}: a result row has a different run_id")
    return {
        "difficulty": difficulty,
        "role": role,
        "entry": entry,
        "run_id": run_id,
        "summary": summary,
        "results": results,
        "outcomes": outcome_counts(results),
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "results_path": results_path,
        "case_ids": set(case_ids),
        "model_id": model_id(entry, results),
        "index_ids": active_index_ids(entry, results),
    }


def load_baselines(registry_path: Path, inventory: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = read_json(registry_path)
    registry_hash = sha256_file(registry_path)
    inventory.append(
        {"role": "baseline_registry", "difficulty": "all", "path": source_path(registry_path), "sha256": registry_hash}
    )
    require(registry.get("status") == "succeeded", "baseline registry is not succeeded")
    require(registry.get("mode") == "live", "baseline registry mode must be live")
    require(tuple(registry.get("difficulty_order", [])) == ORDER, "baseline difficulty_order must be easy/medium/hard")
    baselines = registry.get("baselines")
    require(isinstance(baselines, dict) and set(baselines) == set(ORDER), "baseline registry must contain three difficulties")

    records: list[dict[str, Any]] = []
    for difficulty in ORDER:
        entry = baselines[difficulty]
        require(isinstance(entry, dict), f"{difficulty}: baseline registry entry is invalid")
        require(entry.get("status") == "succeeded", f"{difficulty}: baseline run is not succeeded")
        require(entry.get("baseline_run_id") is None, f"{difficulty}: registered baseline must be a root run")
        require(entry.get("difficulty") == difficulty, f"{difficulty}: baseline difficulty mismatch")
        require(entry.get("mode") == "live", f"{difficulty}: baseline mode must be live")
        case_count = dataset_value(entry, "case_count")
        require(case_count == 50, f"{difficulty}: expected exactly 50 baseline cases")
        require(entry.get("result_count") == case_count, f"{difficulty}: baseline result_count mismatch")
        records.append(
            load_record(
                entry=entry,
                scope_root=SUITE_ROOT,
                difficulty=difficulty,
                role="baseline",
                expected_case_count=case_count,
                inventory=inventory,
            )
        )
    return registry, records


def load_final_records(
    *,
    label: str,
    registry: dict[str, Any],
    registry_path: Path,
    baseline_records: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    require(LABEL_RE.fullmatch(label) is not None, "final label may contain only letters, digits, '.', '_' and '-'")
    run_index_path = REMEDIATION_ROOT / "runs" / label / "run-index.json"
    run_index = read_json(run_index_path)
    inventory.append(
        {
            "role": "final_run_index",
            "difficulty": "all",
            "path": source_path(run_index_path),
            "sha256": sha256_file(run_index_path),
        }
    )
    require(run_index.get("label") == label, "final run-index label does not match --final-label")
    require(run_index.get("status") == "succeeded", "final run-index must be succeeded before report generation")
    require(run_index.get("run_kind") == "paired_candidate", "final run must be a paired_candidate")
    require(run_index.get("selection") == "all", "final run selection must be all")
    require(run_index.get("as_root") is False, "final run must not be recorded as a root baseline")
    require(run_index.get("mode") == registry.get("mode") == "live", "final and baseline modes must both be live")
    require(run_index.get("kb_id") == registry.get("kb_id"), "final and baseline knowledge-base ids differ")
    require(tuple(run_index.get("planned_difficulties", [])) == ORDER, "final run must plan easy/medium/hard")
    source_registry = run_index.get("source_baseline_registry")
    require(isinstance(source_registry, dict), "final run-index source_baseline_registry is missing")
    require(source_registry.get("path") == source_path(registry_path), "final run-index points to a different registry path")
    require(source_registry.get("sha256") == sha256_file(registry_path), "final run-index registry SHA-256 is stale")

    entries = run_index.get("difficulties")
    require(isinstance(entries, dict) and set(entries) == set(ORDER), "final run-index must contain three difficulties")
    baseline_by_difficulty = {record["difficulty"]: record for record in baseline_records}
    records: list[dict[str, Any]] = []
    for difficulty in ORDER:
        entry = entries[difficulty]
        baseline_entry = registry["baselines"][difficulty]
        baseline_record = baseline_by_difficulty[difficulty]
        require(isinstance(entry, dict), f"{difficulty}: final run entry is invalid")
        require(entry.get("status") in {"archived", "succeeded"}, f"{difficulty}: final run is not archived/succeeded")
        require(entry.get("difficulty") == difficulty, f"{difficulty}: final difficulty mismatch")
        require(entry.get("mode") == "live", f"{difficulty}: final mode must be live")
        require(entry.get("selection") == "all", f"{difficulty}: final selection must be all")
        require(entry.get("baseline_run_id") == baseline_entry.get("run_id"), f"{difficulty}: paired baseline_run_id mismatch")
        require(
            entry.get("source_baseline_run_id") == baseline_entry.get("run_id"),
            f"{difficulty}: source_baseline_run_id mismatch",
        )
        for key in ("id", "key", "version", "checksum"):
            require(dataset_value(entry, key) == dataset_value(baseline_entry, key), f"{difficulty}: dataset {key} mismatch")
        case_count = int(dataset_value(baseline_entry, "case_count"))
        require(entry.get("selected_case_count") == case_count, f"{difficulty}: selected_case_count mismatch")
        require(entry.get("completed_cases") == case_count, f"{difficulty}: completed_cases mismatch")
        require(entry.get("total_cases") == case_count, f"{difficulty}: total_cases mismatch")
        record = load_record(
            entry=entry,
            scope_root=REMEDIATION_ROOT,
            difficulty=difficulty,
            role="final",
            expected_case_count=case_count,
            inventory=inventory,
        )
        require(record["case_ids"] == baseline_record["case_ids"], f"{difficulty}: final and baseline case sets differ")
        selected_ids = entry.get("selected_case_ids")
        require(isinstance(selected_ids, list), f"{difficulty}: selected_case_ids is missing")
        require(set(selected_ids) == record["case_ids"], f"{difficulty}: selected_case_ids do not match archived results")
        records.append(record)
    return run_index, records, run_index_path


def verify_diagnostics(
    diagnostics_path: Path,
    registry: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = read_json(diagnostics_path)
    inventory.append(
        {
            "role": "baseline_failure_diagnostics",
            "difficulty": "all",
            "path": source_path(diagnostics_path),
            "sha256": sha256_file(diagnostics_path),
        }
    )
    require(nested(diagnostics, "scope", "mode") == "live", "diagnostics mode must be live")
    require(tuple(nested(diagnostics, "scope", "difficulties") or []) == ORDER, "diagnostics must cover three difficulties")
    require(nested(diagnostics, "scope", "read_only") is True, "diagnostics must be read-only")
    require(nested(diagnostics, "scope", "case_id_special_casing") is False, "diagnostics may not use case-id special casing")
    sources = diagnostics.get("sources")
    require(isinstance(sources, list), "diagnostics sources are missing")
    for source in sources:
        require(isinstance(source, dict), "diagnostics source entry is invalid")
        difficulty = source.get("difficulty")
        require(difficulty in ORDER, "diagnostics source has an invalid difficulty")
        source_file = resolve_scoped_path(SUITE_ROOT, str(source.get("path")), label="diagnostics source")
        require(source_file.is_file(), f"diagnostics source is missing: {source_path(source_file)}")
        actual_hash = sha256_file(source_file)
        require(actual_hash == source.get("sha256"), f"diagnostics source SHA-256 mismatch: {source_path(source_file)}")
        baseline = registry["baselines"][difficulty]
        if source.get("role") == "api_results":
            require(baseline["run_id"] in str(source.get("path")), f"{difficulty}: diagnostics are not tied to the registry run")
        if source.get("role") == "dataset_manifest":
            require(actual_hash == dataset_value(baseline, "source_sha256"), f"{difficulty}: diagnostic dataset SHA mismatch")
        inventory.append(
            {
                "role": f"diagnostics_{source.get('role', 'source')}",
                "difficulty": difficulty,
                "path": source_path(source_file),
                "sha256": actual_hash,
            }
        )
    return diagnostics


def verify_judge_summary(
    judge_path: Path,
    diagnostics: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = read_json(judge_path)
    inventory.append(
        {
            "role": "semantic_judge_summary",
            "difficulty": "all",
            "path": source_path(judge_path),
            "sha256": sha256_file(judge_path),
        }
    )
    require(summary.get("status") == "succeeded", "semantic judge summary is not succeeded")
    require(summary.get("credentials_recorded") is False, "semantic judge summary indicates recorded credentials")
    selected_ids = nested(summary, "plan", "selected_case_ids")
    require(isinstance(selected_ids, list), "semantic judge selected_case_ids are missing")
    diagnostic_ids = {case.get("case_id") for case in diagnostics.get("cases", []) if isinstance(case, dict)}
    require(set(selected_ids).issubset(diagnostic_ids), "semantic judge case set is not a subset of diagnosed failures")
    require(summary.get("unique_case_count") == len(set(selected_ids)), "semantic judge unique_case_count mismatch")
    responses_path = judge_path.with_name("llm-review-responses.jsonl")
    require(responses_path.is_file(), "semantic judge response archive is missing")
    response_hash = sha256_file(responses_path)
    require(response_hash == summary.get("output_sha256"), "semantic judge response SHA-256 mismatch")
    inventory.append(
        {
            "role": "semantic_judge_responses",
            "difficulty": "all",
            "path": source_path(responses_path),
            "sha256": response_hash,
        }
    )
    return summary


def quality_row(record: dict[str, Any], order: int) -> dict[str, Any]:
    summary = record["summary"]
    outcomes = record["outcomes"]
    abstention = summary.get("abstention") if isinstance(summary.get("abstention"), dict) else {}
    false_positive = abstention.get("false_positive")
    true_negative = abstention.get("true_negative")
    fpr_denominator = (
        false_positive + true_negative
        if isinstance(false_positive, int) and isinstance(true_negative, int)
        else None
    )
    return {
        "order": order,
        "difficulty": LABELS[record["difficulty"]],
        "role": ROLE_LABELS[record["role"]],
        "pass_rate": outcomes["passed"] / outcomes["total"] if outcomes["total"] else None,
        "recall_at_1": mean_metric(summary, "retrieval", "recall_at_k", "1"),
        "recall_at_3": mean_metric(summary, "retrieval", "recall_at_k", "3"),
        "recall_at_10": mean_metric(summary, "retrieval", "recall_at_k", "10"),
        "mrr": mean_metric(summary, "retrieval", "mrr"),
        "ndcg_at_10": mean_metric(summary, "retrieval", "ndcg_at_k", "10"),
        "correctness": mean_metric(summary, "generation", "correctness"),
        "faithfulness": mean_metric(summary, "generation", "faithfulness"),
        "citation_coverage": mean_metric(summary, "citation", "coverage"),
        "citation_precision": mean_metric(summary, "citation", "precision"),
        "citation_recall": mean_metric(summary, "citation", "recall"),
        "abstention_accuracy": mean_metric(summary, "abstention", "accuracy"),
        "abstention_fpr": false_positive / fpr_denominator if fpr_denominator else None,
    }


def runtime_row(record: dict[str, Any], order: int) -> dict[str, Any]:
    summary = record["summary"]
    cost_total = runtime_stat(summary, "cost", "total")
    cost_currency = nested(summary, "runtime", "cost", "currency")
    return {
        "order": order,
        "difficulty": LABELS[record["difficulty"]],
        "role": ROLE_LABELS[record["role"]],
        "latency_mean_ms": runtime_stat(summary, "latency_ms", "mean"),
        "latency_p95_ms": runtime_stat(summary, "latency_ms", "p95"),
        "tokens_total": runtime_stat(summary, "total_tokens", "total"),
        "tokens_unknown": runtime_stat(summary, "total_tokens", "unknown_count"),
        "cost_status": "unknown (price snapshot unavailable)" if cost_total is None else f"{cost_total} {cost_currency or ''}".strip(),
    }


def run_row(record: dict[str, Any], order: int) -> dict[str, Any]:
    entry = record["entry"]
    gate = entry.get("gate_result") if isinstance(entry.get("gate_result"), dict) else {}
    return {
        "order": order,
        "difficulty": LABELS[record["difficulty"]],
        "role": ROLE_LABELS[record["role"]],
        "run_id_short": short(record["run_id"], 20),
        "case_count": record["outcomes"]["total"],
        "archive_status": entry.get("status"),
        "mode": entry.get("mode"),
        "gate_status": gate.get("status", "unknown"),
        "gate_passed": "true" if gate.get("passed") is True else "false",
        "model": record["model_id"],
        "retrieval_profile": f"{entry.get('retrieval_profile_id', 'default')} v{entry.get('retrieval_profile_version', '?')}",
    }


def identity_row(record: dict[str, Any], registry: dict[str, Any], order: int) -> dict[str, Any]:
    entry = record["entry"]
    return {
        "order": order,
        "difficulty": LABELS[record["difficulty"]],
        "role": ROLE_LABELS[record["role"]],
        "kb_id": short(registry.get("kb_id"), 18),
        "dataset_key": wrap_hint(dataset_value(entry, "key")),
        "dataset_checksum": short(dataset_value(entry, "checksum"), 18),
        "model": record["model_id"],
        "index_ids": ", ".join(short(value, 14) for value in record["index_ids"]) or "unknown",
        "mode": entry.get("mode"),
    }


def final_failure_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        gate = record["entry"].get("gate_result") if isinstance(record["entry"].get("gate_result"), dict) else {}
        critical_ids = set(gate.get("critical_failed_cases") or [])
        for result in record["results"]:
            if result["passed"]:
                continue
            metrics = result["metrics"]
            outcome = metrics.get("case_outcome") if isinstance(metrics.get("case_outcome"), dict) else {}
            reasons = outcome.get("reason_codes") if isinstance(outcome.get("reason_codes"), list) else []
            if result.get("error_type") or result.get("error_detail"):
                reasons = ["execution_error", *reasons]
            if not reasons:
                reasons = ["historical_evaluator_failure"]
            rows.append(
                {
                    "critical_order": 1 if result["case_id"] in critical_ids else 0,
                    "difficulty": LABELS[record["difficulty"]],
                    "case_id": result["case_id"],
                    "critical": "true" if result["case_id"] in critical_ids else "false",
                    "final_answer_correct_proxy": outcome.get("final_answer_correct"),
                    "failure_reasons": ", ".join(wrap_hint(reason) for reason in reasons),
                }
            )
    return rows


def diagnostic_rows(diagnostics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall = nested(diagnostics, "summary", "primary_labels", "overall") or {}
    failed_count = int(overall.get("failed_case_count") or 0)
    attribution = [
        {
            "primary": label,
            "case_count": count,
            "share": count / failed_count if failed_count else None,
        }
        for label, count in sorted((overall.get("counts") or {}).items(), key=lambda item: (-item[1], item[0]))
    ]
    semantic_overall = nested(diagnostics, "summary", "semantic_correctness", "overall") or {}
    semantic = [
        {
            "status": status,
            "case_count": count,
            "share": count / failed_count if failed_count else None,
        }
        for status, count in sorted((semantic_overall.get("counts") or {}).items(), key=lambda item: (-item[1], item[0]))
    ]
    failures: list[dict[str, Any]] = []
    for case in diagnostics.get("cases", []):
        if not isinstance(case, dict):
            continue
        confidence = case.get("confidence") if isinstance(case.get("confidence"), dict) else {}
        semantic_correctness = (
            case.get("semantic_correctness") if isinstance(case.get("semantic_correctness"), dict) else {}
        )
        failures.append(
            {
                "critical_order": 1 if case.get("critical") else 0,
                "difficulty": LABELS.get(str(case.get("difficulty")), str(case.get("difficulty"))),
                "case_id": case.get("case_id"),
                "critical": "true" if case.get("critical") else "false",
                "primary": wrap_hint(case.get("primary")),
                "secondary": ", ".join(wrap_hint(value) for value in case.get("secondary", [])),
                "confidence": confidence.get("score"),
                "semantic_proxy": wrap_hint(semantic_correctness.get("status")),
                "historical_gate_reasons": ", ".join(
                    wrap_hint(value) for value in case.get("old_gate_reasons", [])
                ),
            }
        )
    return attribution, semantic, failures


def judge_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "order": 0,
            "item": "review coverage",
            "result": f"{summary.get('unique_case_count')} unique cases / {summary.get('record_count')} records",
            "status": summary.get("status"),
        }
    ]
    for index, (case_id, calibration) in enumerate((summary.get("calibration") or {}).items(), start=1):
        rows.append(
            {
                "order": index,
                "item": f"calibration {case_id}",
                "result": (
                    f"expected={calibration.get('expected_semantic_correct')}; "
                    f"actual={calibration.get('actual_semantic_correct')}"
                ),
                "status": "matched" if calibration.get("matched") else "mismatch",
            }
        )
    return rows


def combined_inventory_hash(inventory: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{row['path']}:{row['sha256']}" for row in sorted(inventory, key=lambda row: row["path"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_report_data(
    final_label: str | None = None,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
    judge_summary_path: Path = DEFAULT_JUDGE_SUMMARY,
) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    registry, baseline_records = load_baselines(registry_path, inventory)
    final_index: dict[str, Any] | None = None
    final_records: list[dict[str, Any]] = []
    final_index_path: Path | None = None
    if final_label:
        final_index, final_records, final_index_path = load_final_records(
            label=final_label,
            registry=registry,
            registry_path=registry_path,
            baseline_records=baseline_records,
            inventory=inventory,
        )
    diagnostics = verify_diagnostics(diagnostics_path, registry, inventory)
    judge_summary = verify_judge_summary(judge_summary_path, diagnostics, inventory)

    records = [*baseline_records, *final_records]
    case_outcomes: list[dict[str, Any]] = []
    run_facts: list[dict[str, Any]] = []
    quality_metrics: list[dict[str, Any]] = []
    runtime_telemetry: list[dict[str, Any]] = []
    identity_facts: list[dict[str, Any]] = []
    for difficulty_index, difficulty in enumerate(ORDER):
        for role_index, role in enumerate(("baseline", "final")):
            record = next((item for item in records if item["difficulty"] == difficulty and item["role"] == role), None)
            if record is None:
                continue
            order = difficulty_index * 2 + role_index
            outcome = record["outcomes"]
            case_outcomes.append(
                {
                    "order": order,
                    "difficulty_run": f"{LABELS[difficulty]}·{ROLE_LABELS[role]}",
                    "difficulty": LABELS[difficulty],
                    "role": ROLE_LABELS[role],
                    **outcome,
                }
            )
            run_facts.append(run_row(record, order))
            quality_metrics.append(quality_row(record, order))
            runtime_telemetry.append(runtime_row(record, order))
            identity_facts.append(identity_row(record, registry, order))

    diagnostic_attribution, diagnostic_semantic, diagnostic_failures = diagnostic_rows(diagnostics)
    validation_checks = [
        {"order": 0, "check": "baseline registry", "status": "passed", "detail": "3 × 50 live root runs"},
        {
            "order": 1,
            "check": "baseline artifacts",
            "status": "passed",
            "detail": "copy-manifest, summary and results SHA-256 verified for all difficulties",
        },
        {
            "order": 2,
            "check": "diagnostic lineage",
            "status": "passed",
            "detail": "diagnostic result/dataset hashes match the registered baselines",
        },
        {
            "order": 3,
            "check": "semantic judge archive",
            "status": "passed",
            "detail": "summary, selected-case subset and response SHA-256 verified; credentials_recorded=false",
        },
    ]
    if final_label:
        validation_checks.extend(
            [
                {
                    "order": 4,
                    "check": "final label completeness",
                    "status": "passed",
                    "detail": f"{final_label}: succeeded, three archived 50-case runs",
                },
                {
                    "order": 5,
                    "check": "paired identity",
                    "status": "passed",
                    "detail": "baseline_run_id, dataset, KB, mode and exact case sets match",
                },
                {
                    "order": 6,
                    "check": "final artifacts",
                    "status": "passed",
                    "detail": "copy-manifest, summary and results SHA-256 verified for all difficulties",
                },
            ]
        )
    else:
        validation_checks.append(
            {
                "order": 4,
                "check": "final label selection",
                "status": "pending",
                "detail": "no label auto-discovered; provide --final-label only after explicit approval",
            }
        )

    datasets = {
        "case_outcomes": case_outcomes,
        "run_facts": run_facts,
        "quality_metrics": quality_metrics,
        "runtime_telemetry": runtime_telemetry,
        "identity_facts": identity_facts,
        "diagnostic_attribution": diagnostic_attribution,
        "diagnostic_semantic": diagnostic_semantic,
        "diagnostic_failures": diagnostic_failures,
        "final_failure_cases": final_failure_rows(final_records),
        "semantic_judge": judge_rows(judge_summary),
        "validation_checks": validation_checks,
        "input_inventory": [
            {
                "order": index,
                **row,
                "path": display_path(row["path"]),
                "sha256": short(row["sha256"], 20),
            }
            for index, row in enumerate(
                sorted(inventory, key=lambda row: (row["role"], row["difficulty"], row["path"]))
            )
        ],
    }
    for dataset_id, sql in DATASET_SQL.items():
        datasets[dataset_id] = sqlite_project(dataset_id, datasets[dataset_id], sql)

    return {
        "registry": registry,
        "registry_path": registry_path,
        "baseline_records": baseline_records,
        "final_label": final_label,
        "final_index": final_index,
        "final_index_path": final_index_path,
        "final_records": final_records,
        "diagnostics": diagnostics,
        "diagnostics_path": diagnostics_path,
        "judge_summary": judge_summary,
        "judge_summary_path": judge_summary_path,
        "inventory": inventory,
        "inventory_hash": combined_inventory_hash(inventory),
        "datasets": datasets,
    }


def source_entry(
    source_id: str,
    label: str,
    path: Path,
    *,
    digest: str,
    generated_at: str | None,
    tables_used: list[str] | None = None,
    description: str | None = None,
    metric_definitions: list[str] | None = None,
    sql: str | None = None,
    engine: str = "Repository JSON archive",
    language: str = "json",
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "engine": engine,
        "id": f"sha256:{digest}",
        "language": language,
        "executed_at": generated_at,
        "description": description,
        "tables_used": tables_used or [source_path(path)],
        "filters": ["easy, medium and hard", "full live evaluation", "exact archived rows only"],
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    if sql:
        query["sql"] = sql
    return {"id": source_id, "label": label, "path": source_path(path), "query": query}


def table(
    table_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    columns: list[dict[str, Any]],
    *,
    sort_field: str,
    sort_direction: str = "asc",
    source_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": dataset,
        "density": "dense",
        "layout": "full",
        "sourceId": source_id or f"{dataset}-query",
        "defaultSort": {"field": sort_field, "direction": sort_direction},
        "columns": columns,
    }


def percent_text(value: float) -> str:
    return f"{value:.1%}"


def build_artifact(data: dict[str, Any]) -> dict[str, Any]:
    registry = data["registry"]
    final_label = data["final_label"]
    final_index = data["final_index"]
    datasets = data["datasets"]
    generated_at = (
        final_index.get("updated_at")
        if isinstance(final_index, dict) and final_index.get("updated_at")
        else registry.get("generated_at")
    )
    baseline_outcomes = [row for row in datasets["case_outcomes"] if row["role"] == ROLE_LABELS["baseline"]]
    final_outcomes = [row for row in datasets["case_outcomes"] if row["role"] == ROLE_LABELS["final"]]
    baseline_passes = sum(row["passed"] for row in baseline_outcomes)
    baseline_total = sum(row["total"] for row in baseline_outcomes)
    diagnostic_total = nested(data["diagnostics"], "summary", "failed_case_count") or len(datasets["diagnostic_failures"])
    top_attribution = datasets["diagnostic_attribution"][0] if datasets["diagnostic_attribution"] else None
    judge = data["judge_summary"]

    inventory_paths = [row["path"] for row in data["inventory"]]
    sources = [
        source_entry(
            "baseline-registry",
            "Three-difficulty baseline registry",
            data["registry_path"],
            digest=sha256_file(data["registry_path"]),
            generated_at=registry.get("generated_at"),
            description="Registered easy/medium/hard live root runs and their dataset, gate and artifact identities.",
        ),
        source_entry(
            "baseline-diagnostics",
            "Read-only baseline failure diagnostics",
            data["diagnostics_path"],
            digest=sha256_file(data["diagnostics_path"]),
            generated_at=data["diagnostics"].get("generated_from"),
            description="Case-level primary/secondary attribution and deterministic semantic-correctness proxy for failed baseline cases.",
        ),
        source_entry(
            "semantic-judge-summary",
            "Semantic judge review summary",
            data["judge_summary_path"],
            digest=sha256_file(data["judge_summary_path"]),
            generated_at=generated_at,
            description="qwen-plus review coverage and two-case answer-claims-isolation calibration receipt; no credentials are included.",
        ),
    ]
    if final_label and data["final_index_path"] is not None:
        sources.append(
            source_entry(
                "final-run-index",
                f"Approved final run index: {final_label}",
                data["final_index_path"],
                digest=sha256_file(data["final_index_path"]),
                generated_at=generated_at,
                description="Explicitly selected three-difficulty paired candidate index and archived artifact pointers.",
            )
        )
    sources.append(
        source_entry(
            "report-derivation",
            "Deterministic report projection",
            Path(__file__),
            digest=data["inventory_hash"],
            generated_at=generated_at,
            tables_used=inventory_paths,
            description="Validates registry/run/artifact identities and hashes, then projects bounded case, metric, diagnostic and judge tables.",
            metric_definitions=[
                "case pass rate = archived result rows with passed=true / all archived result rows",
                "failed excludes rows with error_type/error_detail; error is reported separately",
                "Recall@K, MRR, nDCG@10, generation and citation values are arithmetic means from archived summary.json",
                "abstention FPR = false_positive / (false_positive + true_negative)",
                "cost remains unknown when the archived runtime has no priced total",
            ],
            engine="Python 3 deterministic file projection",
            language="python",
        )
    )
    for dataset_id, sql in DATASET_SQL.items():
        if dataset_id == "final_failure_cases" and not final_label:
            continue
        sources.append(
            source_entry(
                f"{dataset_id}-query",
                QUERY_LABELS[dataset_id],
                Path(__file__),
                digest=data["inventory_hash"],
                generated_at=generated_at,
                tables_used=inventory_paths,
                description=f"SQLite ordering and bounded projection for snapshot.datasets.{dataset_id}.",
                sql=sql,
                engine="SQLite over validated repository JSON projections",
                language="sql",
            )
        )

    if final_outcomes:
        final_passes = sum(row["passed"] for row in final_outcomes)
        final_total = sum(row["total"] for row in final_outcomes)
        gate_passes = sum(
            record["entry"].get("gate_result", {}).get("passed") is True for record in data["final_records"]
        )
        technical_summary = (
            "## 技术摘要\n\n"
            f"显式最终 label `{final_label}` 已通过三档完整性、paired baseline、dataset/KB/mode/case-set identity 与归档 SHA-256 校验。"
            f"原始基线通过 {baseline_passes}/{baseline_total}，最终候选通过 {final_passes}/{final_total}，净变化 "
            f"{final_passes - baseline_passes:+d} case；三档 paired gate 中 {gate_passes}/3 通过。\n\n"
            f"历史基线的只读诊断覆盖 {diagnostic_total} 个 evaluator-failed case；语义 judge 摘要覆盖 "
            f"{judge.get('unique_case_count')} 个 case，answer-claims 隔离校准为 "
            f"{'通过' if judge.get('calibration_passed') else '未通过'}。这些诊断/judge 证据属于已登记基线失败，"
            "不会被误写成最终候选的逐题语义结论。"
        )
        comparison_note = (
            f"下图按难度并列原始基线与 `{final_label}`，每根柱固定 50 题并分解为 passed、failed、error。"
            "它回答的是 case 结果构成，而不是把不同量纲的质量指标混在同一图中；精确指标见后续表格。"
        )
        outcome_subtitle = "每档基线与显式最终候选均为同一 50-case 配对集合。"
    else:
        technical_summary = (
            "## 技术摘要\n\n"
            f"当前原始三难度基线通过 {baseline_passes}/{baseline_total}（简单/中等/困难各 50 题），"
            f"历史只读诊断覆盖 {diagnostic_total} 个 evaluator-failed case。"
            f"语义 judge 摘要覆盖 {judge.get('unique_case_count')} 个 case，answer-claims 隔离校准为 "
            f"{'通过' if judge.get('calibration_passed') else '未通过'}。\n\n"
            "尚未显式指定最终 label，因此本 artifact 为 `partial`：它只陈述注册基线、诊断和 judge 审计事实，"
            "不会自动发现或把 remediation/runs 下的任何 candidate 认定为最终结果。"
        )
        comparison_note = (
            "下图只显示已登记原始基线的 passed、failed、error 构成。最终候选必须通过 `--final-label` 显式选择；"
            "在此之前不绘制候选对比，也不形成修复效果结论。"
        )
        outcome_subtitle = "当前仅含已登记原始基线；最终 label 尚未选择。"

    if top_attribution:
        diagnostic_note = (
            "## 历史失败主要集中在中间链路损失\n\n"
            f"基线失败的互斥主归因中，`{top_attribution['primary']}` 最多："
            f"{top_attribution['case_count']}/{diagnostic_total}（{percent_text(top_attribution['share'])}）。"
            "表中的 semantic_proxy 是独立于 historical passed 的确定性参考/证据代理；它不是 LLM judge verdict。"
        )
    else:
        diagnostic_note = "## 历史失败诊断\n\n诊断文件未产生可展示的主归因行。"

    limitations = (
        "## 结论边界与稳健性\n\n"
        "- `passed` 是当次 archived evaluator 的 case 结果，不等同于“最终答案语义正确”；两者在表中保持分离。\n"
        "- correctness、faithfulness、citation 与 abstention 是当前确定性/启发式评测指标，不是盲审人工结论。\n"
        "- qwen-plus judge summary 证明 review 覆盖、两例校准及响应哈希，但摘要没有 verdict 分布，故本报告不从中反推语义正确率。\n"
        "- 历史 diagnostics 的 source hashes 明确绑定原始 baseline runs；除非另行生成 final-run diagnostics，否则不能把其逐题归因套用到最终候选。\n"
        "- Token unknown 保留为 unknown；没有供应商定价快照时 cost 也保持 unknown，不能按 0 解读。\n"
        "- 报告只覆盖这三个 50-case 数据集、冻结知识库/数据集身份和显式 label，不外推到其他文档或生产流量。"
    )

    if final_label:
        next_steps = (
            "## 下一步以 paired gate 和语义复核共同决策\n\n"
            "1. 逐档处理未通过的 paired gate 与 critical cases，区分 retrieval、generation、grounding 和 evaluator 原因。\n"
            "2. 为显式 final label 生成独立逐题 diagnostics/judge summary，再与 historical baseline diagnostics 并列，不覆盖旧证据。\n"
            "3. 固化模型计价与完整 Token collector 后，重新生成 artifact 并执行 portable HTML builder。"
        )
    else:
        next_steps = (
            "## 下一步先锁定最终 label，再生成 ready 报告\n\n"
            "1. 等在线 runner 完成并由负责人明确选择最终 label；不要用目录时间或名称自动推断。\n"
            "2. 用 `--final-label <LABEL>` 重跑生成器；任何缺档、identity 或 SHA-256 不一致都会阻止 ready artifact。\n"
            "3. 若需要最终候选的逐题语义结论，先生成与该 label 绑定的 diagnostics/judge summary，再执行 portable HTML builder。"
        )

    tables = [
        table(
            "run-facts",
            "运行与门禁事实",
            "每档固定 50 题；gate status 与运行成功分开呈现。",
            "run_facts",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "case_count", "label": "题数", "type": "number"},
                {"field": "gate_status", "label": "门禁状态", "type": "text"},
                {"field": "gate_passed", "label": "门禁通过", "type": "text"},
            ],
            sort_field="order",
        ),
        table(
            "retrieval-metrics",
            "Case 与检索指标",
            "比例使用 0–100% 显示；pass rate 直接由 archived result rows 计算。",
            "quality_metrics",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "pass_rate", "label": "Case pass", "format": "percent"},
                {"field": "recall_at_1", "label": "Recall@1", "format": "percent"},
                {"field": "recall_at_3", "label": "Recall@3", "format": "percent"},
                {"field": "recall_at_10", "label": "Recall@10", "format": "percent"},
            ],
            sort_field="order",
        ),
        table(
            "ranking-metrics",
            "检索排序指标",
            "MRR 与 nDCG@10 使用同一归档 evidence ranking。",
            "quality_metrics",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "mrr", "label": "MRR", "format": "percent"},
                {"field": "ndcg_at_10", "label": "nDCG@10", "format": "percent"},
            ],
            sort_field="order",
        ),
        table(
            "answer-metrics",
            "生成、引用与拒答指标",
            "比例使用 0–100% 显示；unknown 不按 0 计。",
            "quality_metrics",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "correctness", "label": "Correctness", "format": "percent"},
                {"field": "faithfulness", "label": "Faithfulness", "format": "percent"},
            ],
            sort_field="order",
        ),
        table(
            "citation-metrics",
            "引用指标",
            "Coverage、precision 与 recall 保持各自 archived 定义。",
            "quality_metrics",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "citation_coverage", "label": "Citation coverage", "format": "percent"},
                {"field": "citation_precision", "label": "Citation precision", "format": "percent"},
                {"field": "citation_recall", "label": "Citation recall", "format": "percent"},
            ],
            sort_field="order",
        ),
        table(
            "abstention-metrics",
            "拒答指标",
            "Accuracy 与 false-positive rate 分开呈现。",
            "quality_metrics",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "abstention_accuracy", "label": "Abstention accuracy", "format": "percent"},
                {"field": "abstention_fpr", "label": "Abstention FPR", "format": "percent"},
            ],
            sort_field="order",
        ),
        table(
            "latency-telemetry",
            "延迟",
            "Mean 与 P95 均来自 archived runtime summary。",
            "runtime_telemetry",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "latency_mean_ms", "label": "Mean latency ms", "type": "number"},
                {"field": "latency_p95_ms", "label": "P95 latency ms", "type": "number"},
            ],
            sort_field="order",
        ),
        table(
            "usage-telemetry",
            "Token 与成本覆盖",
            "成本缺少计价快照时保持 unknown；Token unknown 单独计数。",
            "runtime_telemetry",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "tokens_total", "label": "Known tokens total", "type": "number"},
                {"field": "tokens_unknown", "label": "Unknown token cases", "type": "number"},
                {"field": "cost_status", "label": "Cost", "type": "text"},
            ],
            sort_field="order",
        ),
        table(
            "identity-facts",
            "比较身份",
            "Dataset、KB 与 mode 必须相同；模型、索引和 profile 显式展示而不假定相同。",
            "identity_facts",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "dataset_checksum", "label": "Dataset SHA", "type": "text"},
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "index_ids", "label": "Index IDs", "type": "text"},
            ],
            sort_field="order",
        ),
        table(
            "diagnostic-attribution",
            "历史基线失败主归因",
            "每个 evaluator-failed case 仅计入一个 primary 类别。",
            "diagnostic_attribution",
            [
                {"field": "primary", "label": "Primary", "type": "text"},
                {"field": "case_count", "label": "Case count", "type": "number"},
                {"field": "share", "label": "Share", "format": "percent"},
            ],
            sort_field="case_count",
            sort_direction="desc",
            source_id="diagnostic_attribution-query",
        ),
        table(
            "diagnostic-failures",
            "历史基线逐题归因",
            "Primary/secondary 与 critical 标记；不是最终候选 verdict。",
            "diagnostic_failures",
            [
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "case_id", "label": "Case ID", "type": "text"},
                {"field": "critical", "label": "Critical", "type": "text"},
                {"field": "primary", "label": "Primary", "type": "text"},
                {"field": "secondary", "label": "Secondary", "type": "text"},
            ],
            sort_field="case_id",
            source_id="diagnostic_failures-query",
        ),
        table(
            "diagnostic-semantic-cases",
            "历史基线语义代理与旧门禁原因",
            "Semantic proxy 独立于 historical passed；仍不是 LLM judge verdict。",
            "diagnostic_failures",
            [
                {"field": "case_id", "label": "Case ID", "type": "text"},
                {"field": "semantic_proxy", "label": "Semantic proxy", "type": "text"},
                {"field": "historical_gate_reasons", "label": "Historical gate reasons", "type": "text"},
            ],
            sort_field="case_id",
            source_id="diagnostic_failures-query",
        ),
        table(
            "semantic-judge",
            "语义 judge 审计摘要",
            "仅展示覆盖与 answer-claims 隔离校准；摘要不含 verdict 分布。",
            "semantic_judge",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "item", "label": "检查", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
            sort_field="order",
            source_id="semantic_judge-query",
        ),
        table(
            "validation-checks",
            "输入与可重复性校验",
            "任何 required identity/hash 检查失败都会中止 artifact 生成。",
            "validation_checks",
            [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "check", "label": "检查", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
                {"field": "detail", "label": "证据", "type": "text"},
            ],
            sort_field="order",
        ),
        table(
            "input-inventory",
            "输入制品清单",
            "仓库相对路径与生成时复核的 SHA-256（显示截断，完整值用于 combined input hash）。",
            "input_inventory",
            [
                {"field": "role", "label": "角色", "type": "text"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "path", "label": "Path", "type": "text"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
            sort_field="role",
        ),
    ]
    if final_label:
        tables.append(
            table(
                "final-failure-cases",
                "最终候选 evaluator-failed cases",
                "final_answer_correct_proxy 来自确定性 case_outcome；不是 LLM judge verdict。",
                "final_failure_cases",
                [
                    {"field": "difficulty", "label": "难度", "type": "text"},
                    {"field": "case_id", "label": "Case ID", "type": "text"},
                    {"field": "critical", "label": "Critical", "type": "text"},
                    {"field": "final_answer_correct_proxy", "label": "Answer-correct proxy", "type": "text"},
                    {"field": "failure_reasons", "label": "Evaluator reasons", "type": "text"},
                ],
                sort_field="case_id",
            )
        )

    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# 三难度在线评测修复技术报告"},
        {"id": "technical-summary", "type": "markdown", "layout": "full", "body": technical_summary},
        {
            "id": "outcome-interpretation",
            "type": "markdown",
            "layout": "full",
            "body": f"## Case 结果构成按难度拉开\n\n{comparison_note}",
        },
        {"id": "outcome-chart", "type": "chart", "chartId": "case-outcomes", "layout": "full"},
        {
            "id": "exact-metrics",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 精确指标必须与门禁、覆盖率一起阅读\n\n"
                "下列表格给出 case、retrieval、generation、citation、abstention、latency 和 token 的精确归档值。"
                "门禁失败不等于运行失败；同样，单一 aggregate 改善也不能覆盖 critical case 或 metric coverage 缺口。"
            ),
        },
        {"id": "run-table", "type": "table", "tableId": "run-facts", "layout": "full"},
        {"id": "retrieval-table", "type": "table", "tableId": "retrieval-metrics", "layout": "full"},
        {"id": "ranking-table", "type": "table", "tableId": "ranking-metrics", "layout": "full"},
        {"id": "answer-table", "type": "table", "tableId": "answer-metrics", "layout": "full"},
        {"id": "citation-table", "type": "table", "tableId": "citation-metrics", "layout": "full"},
        {"id": "abstention-table", "type": "table", "tableId": "abstention-metrics", "layout": "full"},
        {"id": "latency-table", "type": "table", "tableId": "latency-telemetry", "layout": "full"},
        {"id": "usage-table", "type": "table", "tableId": "usage-telemetry", "layout": "full"},
        {"id": "diagnostic-note", "type": "markdown", "layout": "full", "body": diagnostic_note},
        {"id": "diagnostic-attribution-table", "type": "table", "tableId": "diagnostic-attribution", "layout": "full"},
        {"id": "diagnostic-failure-table", "type": "table", "tableId": "diagnostic-failures", "layout": "full"},
        {
            "id": "diagnostic-semantic-table",
            "type": "table",
            "tableId": "diagnostic-semantic-cases",
            "layout": "full",
        },
        {
            "id": "judge-note",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 语义 judge 已校准，但摘要不能替代逐题 verdict\n\n"
                f"审计使用 `{nested(judge, 'plan', 'model')}`，覆盖 {judge.get('unique_case_count')} 个基线失败 case。"
                "校准强制只把 visible ACTUAL_ANSWER 作为 answer claims，引用证据仅用于 grounding；"
                "EASY-001/EASY-006 的期望结果均匹配。"
            ),
        },
        {"id": "judge-table", "type": "table", "tableId": "semantic-judge", "layout": "full"},
    ]
    if final_label:
        blocks.extend(
            [
                {
                    "id": "final-failure-note",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 最终候选失败需按 evaluator reason 继续拆解\n\n"
                        "该表从显式 final label 的 archived results 动态提取失败与 critical case。"
                        "其中 answer-correct proxy 与 historical diagnostics/LLM judge 保持不同字段，避免把 evaluator passed 当成语义真值。"
                    ),
                },
                {"id": "final-failure-table", "type": "table", "tableId": "final-failure-cases", "layout": "full"},
            ]
        )
    blocks.extend(
        [
            {
                "id": "scope-definitions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 范围、数据与指标定义\n\n"
                    f"Suite 为 `{registry.get('suite_key')}`，KB 为 `{registry.get('kb_id')}`，范围为 `live/full`；"
                    "简单、中等、困难各 50 题。Case pass rate 直接按 archived `passed=true` 计数；"
                    "failed 排除 execution error。Recall@K/MRR/nDCG@10 基于标注证据排名；"
                    "correctness/faithfulness/citation/abstention 取 archived summary 的算术均值。"
                ),
            },
            {"id": "identity-table", "type": "table", "tableId": "identity-facts", "layout": "full"},
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 方法：先验证身份与哈希，再投影有界数据\n\n"
                    "生成器读取 baseline registry，逐档验证 copy manifest、summary、results 和 SHA-256；"
                    "显式 final label 还必须满足 succeeded、paired_candidate、selection=all、三档 50 case、"
                    "baseline_run_id/dataset/KB/mode/case-set 一致。随后动态投影 aggregate metrics、失败 reason、"
                    "baseline diagnostics 与 semantic judge summary。不会扫描目录选择“最新 candidate”。"
                ),
            },
            {"id": "validation-table", "type": "table", "tableId": "validation-checks", "layout": "full"},
            {"id": "inventory-table", "type": "table", "tableId": "input-inventory", "layout": "full"},
            {"id": "limitations", "type": "markdown", "layout": "full", "body": limitations},
            {"id": "next-steps", "type": "markdown", "layout": "full", "body": next_steps},
            {
                "id": "further-questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "最终发布是否要求三档 paired gate 全部通过？哪些 critical case 可以由独立语义 judge 推翻 evaluator false-negative？"
                    "何时能提供 final-label-specific diagnostics、完整 Token 覆盖和版本化价格快照？"
                ),
            },
        ]
    )

    status = "ready" if final_label else "partial"
    access_issues = (
        []
        if final_label
        else [
            {
                "id": "final-label-not-selected",
                "title": "最终 label 尚未选择",
                "message": "Required final comparison data is intentionally absent; rerun with --final-label after explicit approval.",
            }
        ]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "三难度在线评测修复技术报告",
            "description": (
                f"Baseline versus explicitly selected final label {final_label}."
                if final_label
                else "Baseline, failure diagnostics and semantic judge audit; final label intentionally pending."
            ),
            "generatedAt": generated_at,
            "sources": sources,
            "charts": [
                {
                    "id": "case-outcomes",
                    "title": "每档 50 题的结果构成",
                    "subtitle": outcome_subtitle,
                    "type": "stackedBar",
                    "dataset": "case_outcomes",
                    "encodings": {
                        "x": {"field": "difficulty_run", "type": "nominal", "label": "难度与运行角色"},
                        "y": {
                            "fields": ["passed", "failed", "error"],
                            "type": "quantitative",
                            "label": "Case 数",
                        },
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "sourceId": "case_outcomes-query",
                    "intent": "composition",
                    "question": "每个难度/运行角色的 50 个 case 如何分布为 passed、failed 和 error？",
                    "rationale": "固定分母的堆叠条形图适合比较结果构成；不把不同量纲质量指标画在一起。",
                }
            ],
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": status,
            "datasets": datasets,
            "accessIssues": access_issues,
        },
        "sources": sources,
        "package_info": {
            "authoring": {
                "generator": source_path(Path(__file__)),
                "final_label": final_label,
                "combined_input_sha256": data["inventory_hash"],
                "chart_map": [
                    {
                        "section": "Case 结果构成按难度拉开",
                        "question": "每档 50 case 的 passed/failed/error 构成是什么？",
                        "family": "composition",
                        "type": "stackedBar",
                        "fields": ["difficulty_run", "passed", "failed", "error"],
                        "palette_policy": "three outcome categories; non-color labels preserved in semantic table",
                    }
                ],
                "omitted_visuals": "Exact metrics, failures, methodology and source inventory remain tables because lookup/audit is the primary task.",
            }
        },
    }
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the canonical remediation technical-report artifact. No candidate is auto-selected; "
            "pass --final-label only after the final three-difficulty label is approved."
        )
    )
    parser.add_argument("--final-label", help="Explicit approved directory label under remediation-2026-08-10/runs")
    parser.add_argument("--baseline-registry", help="Repository-relative override for baselines/baselines.json")
    parser.add_argument("--diagnostics", help="Repository-relative override for failure-diagnostics.json")
    parser.add_argument("--semantic-judge-summary", help="Repository-relative override for llm-review-summary.json")
    parser.add_argument(
        "--output",
        default="artifact.json",
        help="Output path inside this technical-report directory (default: artifact.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_candidate = Path(args.output)
    output_path = output_candidate.resolve() if output_candidate.is_absolute() else (HERE / output_candidate).resolve()
    try:
        output_path.relative_to(HERE)
    except ValueError as exc:
        raise ReportInputError("--output must stay inside remediation-2026-08-10/technical-report") from exc
    data = collect_report_data(
        args.final_label,
        registry_path=resolve_input(args.baseline_registry, DEFAULT_REGISTRY),
        diagnostics_path=resolve_input(args.diagnostics, DEFAULT_DIAGNOSTICS),
        judge_summary_path=resolve_input(args.semantic_judge_summary, DEFAULT_JUDGE_SUMMARY),
    )
    artifact = build_artifact(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": source_path(output_path),
                "surface": artifact["surface"],
                "snapshot_status": artifact["snapshot"]["status"],
                "final_label": args.final_label,
                "combined_input_sha256": data["inventory_hash"],
                "artifact_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
