#!/usr/bin/env python3
"""Audit every archived remediation answer with a calibrated semantic judge.

The command is a dry run unless all three execution confirmations are present::

    python review_semantic_correctness.py --label candidate-01
    python review_semantic_correctness.py --label candidate-01 \
      --execute --acknowledge-cost --confirm-full-review

The judge sees only the query, the frozen reference answer, and the selected
visible answer.  ``--answer-stage final`` (the default) reviews the final answer;
``--answer-stage pre_grounding`` reviews the archived pre-grounding answer under
an isolated output identity.  Retrieved evidence, citation payloads, expected
terms, and production metrics are deliberately excluded.  Credentials and raw
provider responses are never written to disk.  Successful responses are
appended and fsynced one at a time so ``--resume`` can continue an interrupted
review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


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
if PROJECT_ROOT is None:  # pragma: no cover - repository layout guard
    raise RuntimeError("cannot locate repository root")

RUNS_ROOT = REMEDIATION_ROOT / "runs"
REVIEWS_ROOT = REMEDIATION_ROOT / "semantic-reviews"
BASELINE_REGISTRY_PATH = SUITE_ROOT / "baselines" / "baselines.json"
DIFFICULTIES = ("easy", "medium", "hard")
ANSWER_STAGES = ("final", "pre_grounding")
INDEX_SCHEMA = "stt.eval.live-remediation-run-index.v2"
RECORD_SCHEMA = "stt.eval.semantic-correctness-review-record.v1"
SUMMARY_SCHEMA = "stt.eval.semantic-correctness-review-summary.v1"
RECOVERY_SCHEMA = "stt.eval.semantic-correctness-review-recovery.v1"
JUDGE_MODEL = "qwen3.7-max"
JUDGE_TEMPERATURE = 0
PROMPT_VERSION = "stt.semantic-visible-answer-judge.v2"
REVIEW_ITERATION = "direct-answer-v3-qwen3.7-max"
CALIBRATION_EXPECTED = {
    "P3500-EASY-001": False,
    "P3500-EASY-006": True,
}

SYSTEM_PROMPT = """You are a strict semantic correctness judge for saved RAG answers.
Return exactly one JSON object and no surrounding prose.

VISIBLE-ANSWER ISOLATION (mandatory):
- Judge only the text in FINAL_VISIBLE_ANSWER.
- A reference answer states the expected meaning; it is not text produced by the system.
- Citation markers such as [E1] are markers only. They cannot import, imply, or complete any
  source text that is not explicitly stated in FINAL_VISIBLE_ANSWER.
- No retrieved evidence, source excerpt, expected-term list, or historical metric is available.
- First derive the claims explicitly stated by FINAL_VISIBLE_ANSWER, then compare those claims
  with the query and reference answer.
- Extra irrelevant material is acceptable only when it does not evade the question, introduce a
  material error, or obscure the requested answer.
- Do not infer a requested conclusion from merely related background facts. The conclusion must
  be explicitly stated in FINAL_VISIBLE_ANSWER.
- For a yes/no, required/not-required, enabled/disabled, allowed/forbidden, or other polarity
  question, FINAL_VISIBLE_ANSWER must unambiguously state the requested polarity. A statement
  that someone has a right, capability, or intention to act does not by itself answer whether a
  separate notice, permission, or requirement applies.
- semantic_correct=true only for a fully correct answer with no material omission or contradiction.
  A partially correct answer must be semantic_status=partial and semantic_correct=false.
- Do not invent case-id-specific rules. No case ID is supplied.

Required JSON keys:
- semantic_correct: boolean
- semantic_status: one of correct, partial, incorrect
- semantic_score: correct must be 1; partial must be strictly between 0 and 1;
  incorrect must be 0
- confidence_score: number from 0 to 1
- answer_claims: array of concise strings derived only from FINAL_VISIBLE_ANSWER
- missing_or_incorrect_aspects: array of concise strings
- reasoning_summary: concise string
"""


class ReviewError(RuntimeError):
    """A safe, user-facing review validation error."""


@dataclass(frozen=True)
class ReviewCase:
    phase: str
    difficulty: str
    case_id: str
    answer_stage: str
    query: str
    reference_answer: str
    final_answer: str
    production_correctness: float | None
    production_passed: bool | None
    source_results_path: str
    source_results_sha256: str
    source_identity_sha256: str

    @property
    def input_sha256(self) -> str:
        identity = {
            "query": self.query,
            "reference_answer": self.reference_answer,
            "final_visible_answer": self.final_answer,
        }
        # Preserve the historical final-stage identity so existing immutable
        # records remain resumable.  Non-final stages must be domain-separated.
        if self.answer_stage != "final":
            identity = {
                "answer_stage": self.answer_stage,
                "query": self.query,
                "reference_answer": self.reference_answer,
                "selected_visible_answer": self.final_answer,
            }
        return _sha256_json(identity)


@dataclass(frozen=True)
class DifficultyPlan:
    difficulty: str
    dataset_path: str
    dataset_sha256: str
    results_path: str
    results_sha256: str
    run_id: str
    cases: tuple[ReviewCase, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read valid JSON: {_relative_display(path)}") from exc


def _write_json_atomic(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _relative_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _contained_path(root: Path, relative: str, *, label: str) -> Path:
    if not relative:
        raise ReviewError(f"missing {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ReviewError(f"{label} path escapes its archive root") from exc
    return path


def _safe_label(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", normalized):
        raise argparse.ArgumentTypeError(
            "label must contain 1..64 lowercase letters, digits, dots, underscores or dashes"
        )
    return normalized


def _fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number from 0 to 1") from exc
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be a number from 0 to 1")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be positive") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _sanitize_endpoint(base_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ReviewError("judge base URL must be an http(s) URL")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    host = f"{hostname}:{parsed.port}" if parsed.port else hostname
    clean_path = parsed.path.rstrip("/")
    clean_base = urllib.parse.urlunsplit((parsed.scheme, host, clean_path, "", ""))
    endpoint = (
        clean_base if clean_base.endswith("/chat/completions") else clean_base + "/chat/completions"
    )
    return endpoint, clean_base


def _resolve_index(label: str | None, run_index: str | None) -> tuple[Path, dict[str, Any]]:
    if label:
        path = RUNS_ROOT / label / "run-index.json"
    elif run_index:
        candidate = Path(run_index).expanduser()
        path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    else:  # guarded by argument validation
        raise ReviewError("one of --label or --run-index is required")
    path = path.resolve()
    expected_runs_root = RUNS_ROOT.resolve()
    try:
        relative = path.relative_to(expected_runs_root)
    except ValueError as exc:
        raise ReviewError("run index must be under the remediation runs directory") from exc
    if len(relative.parts) != 2 or relative.parts[1] != "run-index.json":
        raise ReviewError("run index must be runs/<label>/run-index.json")
    if not path.is_file():
        raise ReviewError(f"remediation run index is missing: {_relative_display(path)}")
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ReviewError("remediation run index is not an object")
    index_label = str(value.get("label") or "")
    if index_label != relative.parts[0] or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,63}", index_label
    ):
        raise ReviewError("remediation run index label/path identity is invalid")
    if value.get("schema_version") != INDEX_SCHEMA:
        raise ReviewError("unsupported remediation run-index schema")
    if value.get("status") != "succeeded":
        raise ReviewError("remediation run index must have status=succeeded")
    if value.get("mode") != "live":
        raise ReviewError("semantic correctness review requires a live remediation run")
    return path, value


def _baseline_registry(index: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not BASELINE_REGISTRY_PATH.is_file():
        raise ReviewError("frozen baseline registry is missing")
    sha256 = _sha256_file(BASELINE_REGISTRY_PATH)
    frozen = index.get("source_baseline_registry")
    if not isinstance(frozen, Mapping) or frozen.get("sha256") != sha256:
        raise ReviewError("run index baseline-registry hash does not match the frozen registry")
    registry = _read_json(BASELINE_REGISTRY_PATH)
    if not isinstance(registry, dict) or registry.get("status") != "succeeded":
        raise ReviewError("frozen baseline registry is not succeeded")
    baselines = registry.get("baselines")
    if not isinstance(baselines, Mapping) or any(level not in baselines for level in DIFFICULTIES):
        raise ReviewError("frozen baseline registry is incomplete")
    return registry, sha256


def _artifact_results(
    *,
    archive_root: Path,
    containment_root: Path,
    manifest_relative: str,
    manifest_sha256: str,
    expected_run_id: str,
    expected_difficulty: str,
) -> tuple[Path, str]:
    manifest_path = _contained_path(archive_root, manifest_relative, label="artifact-copy manifest")
    try:
        manifest_path.relative_to(containment_root.resolve())
    except ValueError as exc:
        raise ReviewError("artifact-copy manifest is outside the selected run archive") from exc
    if not manifest_path.is_file() or _sha256_file(manifest_path) != manifest_sha256:
        raise ReviewError("artifact-copy manifest is missing or changed")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ReviewError("artifact-copy manifest is not an object")
    if manifest.get("run_id") != expected_run_id:
        raise ReviewError("artifact-copy manifest has the wrong run id")
    if manifest.get("difficulty") != expected_difficulty:
        raise ReviewError("artifact-copy manifest has the wrong difficulty")
    copies = manifest.get("copies")
    result_copy = copies.get("results") if isinstance(copies, Mapping) else None
    if not isinstance(result_copy, Mapping):
        raise ReviewError("artifact-copy manifest has no results receipt")
    if result_copy.get("source_target_sha256_match") is not True:
        raise ReviewError("results receipt did not verify source/target equality")
    target = result_copy.get("target")
    if not isinstance(target, Mapping):
        raise ReviewError("results receipt has no target")
    target_relative = str(target.get("relative_path") or "")
    target_sha256 = str(target.get("sha256") or "")
    results_path = _contained_path(archive_root, target_relative, label="results artifact")
    try:
        results_path.relative_to(containment_root.resolve())
    except ValueError as exc:
        raise ReviewError("results artifact is outside the selected run archive") from exc
    if not results_path.is_file() or _sha256_file(results_path) != target_sha256:
        raise ReviewError("archived results artifact is missing or changed")
    return results_path, target_sha256


def _dataset_cases(
    difficulty: str,
    entry: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[Path, str, dict[str, dict[str, Any]], list[str]]:
    baselines = registry["baselines"]
    baseline = baselines[difficulty]
    baseline_dataset = baseline.get("dataset") if isinstance(baseline, Mapping) else None
    if not isinstance(baseline_dataset, Mapping):
        raise ReviewError(f"baseline dataset identity is missing for {difficulty}")
    identity_keys = {
        "dataset_id": "id",
        "dataset_key": "key",
        "dataset_version": "version",
        "dataset_checksum": "checksum",
        "dataset_source_sha256": "source_sha256",
    }
    mismatched = [
        entry_key
        for entry_key, baseline_key in identity_keys.items()
        if entry.get(entry_key) != baseline_dataset.get(baseline_key)
    ]
    if mismatched:
        raise ReviewError(
            f"{difficulty} dataset identity differs from the frozen registry: "
            + ", ".join(mismatched)
        )
    canonical_path = _contained_path(
        SUITE_ROOT,
        str(baseline_dataset.get("source_path") or ""),
        label=f"{difficulty} dataset",
    )
    entry_path = _contained_path(
        PROJECT_ROOT,
        str(entry.get("dataset_source_path") or ""),
        label=f"{difficulty} dataset source",
    )
    if entry_path != canonical_path:
        raise ReviewError(f"{difficulty} dataset source path is not canonical")
    source_sha256 = str(baseline_dataset.get("source_sha256") or "")
    if not canonical_path.is_file() or _sha256_file(canonical_path) != source_sha256:
        raise ReviewError(f"{difficulty} frozen dataset source is missing or changed")
    dataset = _read_json(canonical_path)
    if not isinstance(dataset, Mapping):
        raise ReviewError(f"{difficulty} dataset is not an object")
    if (
        dataset.get("difficulty") != difficulty
        or dataset.get("dataset_key") != baseline_dataset.get("key")
        or dataset.get("version") != baseline_dataset.get("version")
    ):
        raise ReviewError(f"{difficulty} dataset metadata identity is invalid")
    rows = dataset.get("cases")
    if not isinstance(rows, list):
        raise ReviewError(f"{difficulty} dataset has no cases")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ReviewError(f"{difficulty} dataset contains an invalid case")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in by_id:
            raise ReviewError(f"{difficulty} dataset case ids are invalid")
        query = row.get("query")
        reference = row.get("reference_answer")
        if not isinstance(query, str) or not query.strip():
            raise ReviewError(f"{difficulty} case {case_id} has no query")
        if not isinstance(reference, str) or not reference.strip():
            raise ReviewError(f"{difficulty} case {case_id} has no reference answer")
        by_id[case_id] = row
    if len(by_id) != int(baseline_dataset.get("case_count") or 0):
        raise ReviewError(f"{difficulty} dataset case count changed")
    selected = entry.get("selected_case_ids")
    if not isinstance(selected, list) or not selected:
        raise ReviewError(f"{difficulty} run has no frozen selected case ids")
    selected_ids = [str(case_id) for case_id in selected]
    if (
        len(selected_ids) != len(set(selected_ids))
        or any(case_id not in by_id for case_id in selected_ids)
        or len(selected_ids) != int(entry.get("selected_case_count") or 0)
    ):
        raise ReviewError(f"{difficulty} selected case identity is invalid")
    return canonical_path, source_sha256, by_id, selected_ids


def _final_visible_answer(observation: Mapping[str, Any], case_id: str) -> str:
    envelope = observation.get("answer_envelope")
    envelope_answer = envelope.get("answer") if isinstance(envelope, Mapping) else None
    legacy_answer = observation.get("answer")
    if envelope_answer is not None and not isinstance(envelope_answer, str):
        raise ReviewError(f"{case_id} final answer-envelope value is not text")
    if legacy_answer is not None and not isinstance(legacy_answer, str):
        raise ReviewError(f"{case_id} legacy final answer value is not text")
    if isinstance(envelope_answer, str):
        if isinstance(legacy_answer, str) and legacy_answer != envelope_answer:
            raise ReviewError(f"{case_id} has conflicting final visible answer fields")
        return envelope_answer
    if isinstance(legacy_answer, str):
        return legacy_answer
    return ""


def _selected_visible_answer(
    observation: Mapping[str, Any], case_id: str, answer_stage: str
) -> str:
    if answer_stage == "final":
        return _final_visible_answer(observation, case_id)
    if answer_stage != "pre_grounding":  # pragma: no cover - argparse/internal guard
        raise ReviewError(f"unsupported answer stage: {answer_stage}")
    value = observation.get("pre_grounding_answer")
    # A query that failed before answer generation has no pre-grounding field;
    # its selected visible answer is honestly the empty answer, not the later
    # final-answer fallback.  This keeps every frozen case in the blind review.
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReviewError(f"{case_id} archived pre-grounding answer is not text")
    return value


def _production_correctness(row: Mapping[str, Any]) -> float | None:
    metrics = row.get("metrics")
    metrics_by_family = metrics.get("metrics") if isinstance(metrics, Mapping) else None
    generation = (
        metrics_by_family.get("generation") if isinstance(metrics_by_family, Mapping) else None
    )
    value = generation.get("correctness") if isinstance(generation, Mapping) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError("production generation.correctness is not numeric")
    number = float(value)
    if not 0 <= number <= 1:
        raise ReviewError("production generation.correctness is outside 0..1")
    return number


def _read_result_cases(
    *,
    phase: str,
    difficulty: str,
    results_path: Path,
    results_sha256: str,
    run_id: str,
    selected_ids: Sequence[str],
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    source_identity_sha256: str,
    answer_stage: str,
) -> tuple[ReviewCase, ...]:
    cases: list[ReviewCase] = []
    try:
        handle = results_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReviewError(f"cannot read results: {_relative_display(results_path)}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"invalid results JSONL row {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ReviewError(f"invalid results row {line_number}")
            case_id = str(row.get("case_id") or "")
            if row.get("run_id") != run_id:
                raise ReviewError(f"{difficulty} result row has the wrong run id")
            if case_id not in dataset_by_id:
                raise ReviewError(f"{difficulty} result row has an unknown case id")
            dataset_case = dataset_by_id[case_id]
            query = str(dataset_case["query"])
            reference_answer = str(dataset_case["reference_answer"])
            observation = row.get("observation")
            if not isinstance(observation, Mapping):
                raise ReviewError(f"{case_id} has no observation")
            for observed_query in (observation.get("query"),):
                if isinstance(observed_query, str) and observed_query != query:
                    raise ReviewError(f"{case_id} observation query differs from the dataset")
            metrics = row.get("metrics")
            metric_query = metrics.get("query") if isinstance(metrics, Mapping) else None
            if isinstance(metric_query, str) and metric_query != query:
                raise ReviewError(f"{case_id} metric query differs from the dataset")
            passed = row.get("passed")
            if passed not in {True, False, None}:
                raise ReviewError(f"{case_id} production passed value is invalid")
            cases.append(
                ReviewCase(
                    phase=phase,
                    difficulty=difficulty,
                    case_id=case_id,
                    answer_stage=answer_stage,
                    query=query,
                    reference_answer=reference_answer,
                    final_answer=_selected_visible_answer(observation, case_id, answer_stage),
                    production_correctness=_production_correctness(row),
                    production_passed=passed,
                    source_results_path=_relative_display(results_path),
                    source_results_sha256=results_sha256,
                    source_identity_sha256=source_identity_sha256,
                )
            )
    actual_ids = [case.case_id for case in cases]
    if actual_ids != list(selected_ids):
        raise ReviewError(f"{difficulty} results order/case set differs from the frozen selection")
    return tuple(cases)


def _build_target_plan(
    index_path: Path,
    index: Mapping[str, Any],
    registry: Mapping[str, Any],
    answer_stage: str,
) -> tuple[DifficultyPlan, ...]:
    difficulties = index.get("difficulties")
    if not isinstance(difficulties, Mapping):
        raise ReviewError("run index has no difficulty entries")
    planned = index.get("planned_difficulties")
    if planned != list(DIFFICULTIES):
        raise ReviewError("full semantic review requires all three planned difficulties")
    index_sha256 = _sha256_file(index_path)
    plans: list[DifficultyPlan] = []
    for difficulty in DIFFICULTIES:
        entry = difficulties.get(difficulty)
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise ReviewError(f"{difficulty} remediation artifacts are not archived")
        if entry.get("difficulty") != difficulty:
            raise ReviewError(f"{difficulty} run-index entry identity is invalid")
        run_id = str(entry.get("run_id") or "")
        if not run_id:
            raise ReviewError(f"{difficulty} run id is missing")
        dataset_path, dataset_sha, dataset_by_id, selected_ids = _dataset_cases(
            difficulty, entry, registry
        )
        manifest_receipt = entry.get("artifact_copy_manifest")
        if not isinstance(manifest_receipt, Mapping):
            raise ReviewError(f"{difficulty} artifact-copy receipt is missing")
        results_path, results_sha = _artifact_results(
            archive_root=REMEDIATION_ROOT,
            containment_root=index_path.parent,
            manifest_relative=str(manifest_receipt.get("path") or ""),
            manifest_sha256=str(manifest_receipt.get("sha256") or ""),
            expected_run_id=run_id,
            expected_difficulty=difficulty,
        )
        cases = _read_result_cases(
            phase="evaluation",
            difficulty=difficulty,
            results_path=results_path,
            results_sha256=results_sha,
            run_id=run_id,
            selected_ids=selected_ids,
            dataset_by_id=dataset_by_id,
            source_identity_sha256=index_sha256,
            answer_stage=answer_stage,
        )
        plans.append(
            DifficultyPlan(
                difficulty=difficulty,
                dataset_path=_relative_display(dataset_path),
                dataset_sha256=dataset_sha,
                results_path=_relative_display(results_path),
                results_sha256=results_sha,
                run_id=run_id,
                cases=cases,
            )
        )
    return tuple(plans)


def _build_calibration_cases(
    registry: Mapping[str, Any], registry_sha256: str
) -> tuple[ReviewCase, ...]:
    baseline = registry["baselines"]["easy"]
    if not isinstance(baseline, Mapping):
        raise ReviewError("easy baseline identity is invalid")
    dataset = baseline.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ReviewError("easy baseline dataset identity is missing")
    dataset_path = _contained_path(
        SUITE_ROOT, str(dataset.get("source_path") or ""), label="calibration dataset"
    )
    dataset_sha = str(dataset.get("source_sha256") or "")
    if not dataset_path.is_file() or _sha256_file(dataset_path) != dataset_sha:
        raise ReviewError("calibration dataset is missing or changed")
    dataset_value = _read_json(dataset_path)
    rows = dataset_value.get("cases") if isinstance(dataset_value, Mapping) else None
    if not isinstance(rows, list):
        raise ReviewError("calibration dataset has no cases")
    by_id = {
        str(row.get("case_id")): row for row in rows if isinstance(row, dict) and row.get("case_id")
    }
    if any(case_id not in by_id for case_id in CALIBRATION_EXPECTED):
        raise ReviewError("calibration dataset cases are missing")
    manifest_receipt = baseline.get("artifact_copy_manifest")
    if not isinstance(manifest_receipt, Mapping):
        raise ReviewError("calibration artifact-copy receipt is missing")
    run_id = str(baseline.get("run_id") or "")
    results_path, results_sha = _artifact_results(
        archive_root=SUITE_ROOT,
        containment_root=SUITE_ROOT,
        manifest_relative=str(manifest_receipt.get("path") or ""),
        manifest_sha256=str(manifest_receipt.get("sha256") or ""),
        expected_run_id=run_id,
        expected_difficulty="easy",
    )
    all_ids = [str(row.get("case_id")) for row in rows if isinstance(row, Mapping)]
    all_cases = _read_result_cases(
        phase="calibration",
        difficulty="easy",
        results_path=results_path,
        results_sha256=results_sha,
        run_id=run_id,
        selected_ids=all_ids,
        dataset_by_id=by_id,
        source_identity_sha256=registry_sha256,
        answer_stage="final",
    )
    selected_by_id = {case.case_id: case for case in all_cases}
    return tuple(selected_by_id[case_id] for case_id in CALIBRATION_EXPECTED)


def _messages(case: ReviewCase, *, second_review: bool = False) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "query": case.query,
        "reference_answer": case.reference_answer,
        "FINAL_VISIBLE_ANSWER": case.final_answer,
    }
    if second_review:
        payload["recheck_instruction"] = (
            "This is an independent second review. Re-derive the visible answer claims and "
            "re-evaluate from scratch; no first-review result or trigger signal is supplied."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _canonical_json(payload)},
    ]


def _request_payload(messages: Sequence[Mapping[str, str]]) -> tuple[bytes, str]:
    payload = {
        "model": JUDGE_MODEL,
        "messages": list(messages),
        "temperature": JUDGE_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return encoded, _sha256_bytes(encoded)


def _request_identity_sha256(
    *,
    answer_stage: str,
    messages: Sequence[Mapping[str, str]],
    payload_sha256: str,
    safe_base_url: str,
) -> str:
    """Bind a provider request to its local answer-stage audit domain.

    ``answer_stage`` is deliberately not added to the provider messages: the
    semantic judge receives only query/reference/selected-visible-answer text.
    """

    return _sha256_json(
        {
            "answer_stage": answer_stage,
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _sha256_json(list(messages)),
            "payload_sha256": payload_sha256,
            "safe_base_url": safe_base_url,
        }
    )


def _provider_request(
    *,
    endpoint: str,
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float,
    max_attempts: int,
    retry_delay: float,
) -> tuple[dict[str, Any], str, int]:
    encoded, payload_sha256 = _request_payload(messages)
    retryable_http = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ReviewError("judge response is not a JSON object")
            return value, payload_sha256, attempt
        except urllib.error.HTTPError as exc:
            if exc.code not in retryable_http or attempt == max_attempts:
                raise ReviewError(f"judge HTTP request failed with status {exc.code}") from exc
        except urllib.error.URLError as exc:
            if attempt == max_attempts:
                raise ReviewError("judge network request failed") from exc
        except TimeoutError as exc:
            # urllib may surface a socket read timeout directly instead of
            # wrapping it in URLError.  Treat both forms with the same bounded
            # retry policy so an incremental batch remains resumable.
            if attempt == max_attempts:
                raise ReviewError("judge network request timed out") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewError("judge returned an invalid JSON response envelope") from exc
        time.sleep(retry_delay * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReviewError(f"judge returned invalid {field}")
    return [item.strip() for item in value if item.strip()]


def _score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError(f"judge returned invalid {field}")
    number = float(value)
    if not 0 <= number <= 1:
        raise ReviewError(f"judge returned {field} outside 0..1")
    return number


def _judgment(response: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ReviewError("judge response has no choices")
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ReviewError("judge response content is not text")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ReviewError("judge response content is not a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ReviewError("judge response content is not an object")
    semantic_correct = value.get("semantic_correct")
    status = value.get("semantic_status")
    if semantic_correct not in {True, False}:
        raise ReviewError("judge returned invalid semantic_correct")
    if status not in {"correct", "partial", "incorrect"}:
        raise ReviewError("judge returned invalid semantic_status")
    if (status == "correct") is not semantic_correct:
        raise ReviewError("judge semantic_correct conflicts with semantic_status")
    reasoning = value.get("reasoning_summary")
    # The free-text explanation is redacted and does not participate in the
    # binary decision.  A provider omitting it must not strand an otherwise
    # valid, incrementally persisted review batch.
    reasoning_text = reasoning.strip() if isinstance(reasoning, str) else ""
    semantic_score = _score(value.get("semantic_score"), field="semantic_score")
    if (
        (status == "correct" and semantic_score != 1.0)
        or (status == "partial" and not 0 < semantic_score < 1)
        or (status == "incorrect" and semantic_score != 0.0)
    ):
        raise ReviewError("judge semantic_score conflicts with semantic_status")
    answer_claims = _string_list(value.get("answer_claims"), field="answer_claims")
    missing_aspects = _string_list(
        value.get("missing_or_incorrect_aspects"),
        field="missing_or_incorrect_aspects",
    )
    detail = {
        "answer_claims": answer_claims,
        "missing_or_incorrect_aspects": missing_aspects,
        "reasoning_summary": reasoning_text,
    }
    # Free-form model text can echo source material. Preserve only counts and a
    # deterministic digest; the actual phrases and rationale never reach disk.
    sanitized = {
        "semantic_correct": semantic_correct,
        "semantic_status": status,
        "semantic_score": semantic_score,
        "confidence_score": _score(value.get("confidence_score"), field="confidence_score"),
        "answer_claim_count": len(answer_claims),
        "missing_or_incorrect_aspect_count": len(missing_aspects),
        "reasoning_char_count": len(reasoning_text),
        "redacted_detail_sha256": _sha256_json(detail),
    }
    return sanitized, _sha256_text(content)


def _normalize_usage(value: Any) -> dict[str, int | None]:
    usage = value if isinstance(value, Mapping) else {}

    def token(*names: str) -> int | None:
        for name in names:
            candidate = usage.get(name)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                return candidate
        return None

    return {
        "input_tokens": token("prompt_tokens", "input_tokens"),
        "output_tokens": token("completion_tokens", "output_tokens"),
        "total_tokens": token("total_tokens"),
    }


def _saved_judgment(record: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    value = record.get("judgment")
    required = {
        "semantic_correct",
        "semantic_status",
        "semantic_score",
        "confidence_score",
        "answer_claim_count",
        "missing_or_incorrect_aspect_count",
        "reasoning_char_count",
        "redacted_detail_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ReviewError(f"saved judgment is invalid for {case_id}")
    semantic_correct = value.get("semantic_correct")
    status = value.get("semantic_status")
    if semantic_correct not in {True, False} or status not in {
        "correct",
        "partial",
        "incorrect",
    }:
        raise ReviewError(f"saved judgment status is invalid for {case_id}")
    if (status == "correct") is not semantic_correct:
        raise ReviewError(f"saved judgment binary/status conflict for {case_id}")
    score = _score(value.get("semantic_score"), field="saved semantic_score")
    _score(value.get("confidence_score"), field="saved confidence_score")
    if (
        (status == "correct" and score != 1.0)
        or (status == "partial" and not 0 < score < 1)
        or (status == "incorrect" and score != 0.0)
    ):
        raise ReviewError(f"saved judgment score/status conflict for {case_id}")
    for field in (
        "answer_claim_count",
        "missing_or_incorrect_aspect_count",
        "reasoning_char_count",
    ):
        if (
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or value[field] < 0
        ):
            raise ReviewError(f"saved judgment count is invalid for {case_id}")
    digest = value.get("redacted_detail_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReviewError(f"saved judgment detail hash is invalid for {case_id}")
    return value


def _second_review_reasons(
    case: ReviewCase,
    judgment: Mapping[str, Any],
    *,
    confidence_threshold: float,
    production_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if float(judgment["confidence_score"]) < confidence_threshold:
        reasons.append("low_judge_confidence")
    production_conflict_enabled = not (
        case.phase == "evaluation" and case.answer_stage == "pre_grounding"
    )
    if production_conflict_enabled and case.production_correctness is not None:
        production_binary = case.production_correctness >= production_threshold
        if bool(judgment["semantic_correct"]) != production_binary:
            reasons.append("production_correctness_conflict")
    return reasons


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record.get("phase") or ""),
        str(record.get("difficulty") or ""),
        str(record.get("case_id") or ""),
        int(record.get("review_index") or 0),
    )


def _record_hash(record: Mapping[str, Any]) -> str:
    return _sha256_json({key: value for key, value in record.items() if key != "record_sha256"})


def _load_records(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]]]:
    if not path.is_file():
        return [], {}
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReviewError("cannot read existing review records") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"invalid review JSONL row {line_number}") from exc
            if not isinstance(row, dict) or row.get("schema_version") != RECORD_SCHEMA:
                raise ReviewError(f"invalid review record {line_number}")
            if row.get("record_sha256") != _record_hash(row):
                raise ReviewError(f"review record hash mismatch at row {line_number}")
            try:
                key = _record_key(row)
            except (TypeError, ValueError) as exc:
                raise ReviewError(f"invalid review record key at row {line_number}") from exc
            if key in by_key or key[3] not in {1, 2}:
                raise ReviewError(f"duplicate or invalid review record at row {line_number}")
            rows.append(row)
            by_key[key] = row
    return rows, by_key


def _load_recovery_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = _read_json(path)
    events = value.get("events") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != RECOVERY_SCHEMA
        or not isinstance(events, list)
        or any(not isinstance(event, dict) for event in events)
    ):
        raise ReviewError("resume recovery audit is invalid")
    return list(events)


def _repair_incomplete_jsonl_tail(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ReviewError("cannot inspect review JSONL tail") from exc
    if not content or content.endswith(b"\n"):
        return None
    tail_offset = content.rfind(b"\n") + 1
    tail = content[tail_offset:]
    before_sha256 = _sha256_bytes(content)
    try:
        decoded = tail.decode("utf-8")
        json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        action = "truncated_incomplete_final_row"
        discarded_byte_count = len(tail)
        discarded_sha256 = _sha256_bytes(tail)
        try:
            with path.open("r+b") as handle:
                handle.truncate(tail_offset)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ReviewError("cannot repair incomplete review JSONL tail") from exc
    else:
        action = "appended_missing_final_newline"
        discarded_byte_count = 0
        discarded_sha256 = None
        try:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ReviewError("cannot complete review JSONL final newline") from exc
    return {
        "schema_version": RECOVERY_SCHEMA,
        "recovered_at": _utc_now(),
        "action": action,
        "before_sha256": before_sha256,
        "after_sha256": _sha256_file(path),
        "discarded_byte_count": discarded_byte_count,
        "discarded_sha256": discarded_sha256,
        "raw_tail_recorded": False,
    }


def _append_record(handle: TextIO, record: dict[str, Any]) -> None:
    record["record_sha256"] = _record_hash(record)
    handle.write(_canonical_json(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _validate_existing_record(
    record: Mapping[str, Any],
    *,
    case: ReviewCase,
    review_index: int,
    prompt_sha256: str,
    payload_sha256: str,
    trigger_reasons: Sequence[str],
    safe_base_url: str,
    confidence_threshold: float,
    production_threshold: float,
) -> None:
    request = record.get("request")
    source = record.get("source")
    policy = record.get("review_policy")
    legacy_final_identity = (
        case.answer_stage == "final"
        and record.get("answer_stage") is None
        and isinstance(request, Mapping)
        and request.get("answer_stage") is None
        and request.get("request_identity_sha256") is None
        and isinstance(source, Mapping)
        and source.get("answer_stage") is None
        and source.get("selected_visible_answer_sha256") is None
    )
    staged_identity = (
        isinstance(request, Mapping)
        and isinstance(source, Mapping)
        and record.get("answer_stage") == case.answer_stage
        and request.get("answer_stage") == case.answer_stage
        and source.get("answer_stage") == case.answer_stage
        and source.get("selected_visible_answer_sha256") == _sha256_text(case.final_answer)
        and source.get("selected_visible_answer_char_count") == len(case.final_answer)
        and request.get("request_identity_sha256")
        == _request_identity_sha256(
            answer_stage=case.answer_stage,
            messages=(
                _messages(case, second_review=review_index == 2)
                if review_index in {1, 2}
                else []
            ),
            payload_sha256=payload_sha256,
            safe_base_url=safe_base_url,
        )
    )
    checks = (
        isinstance(request, Mapping)
        and isinstance(source, Mapping)
        and isinstance(policy, Mapping)
        and (legacy_final_identity or staged_identity)
        and record.get("phase") == case.phase
        and record.get("difficulty") == case.difficulty
        and record.get("case_id") == case.case_id
        and record.get("review_index") == review_index
        and request.get("model") == JUDGE_MODEL
        and request.get("temperature") == JUDGE_TEMPERATURE
        and request.get("prompt_version") == PROMPT_VERSION
        and request.get("prompt_sha256") == prompt_sha256
        and request.get("payload_sha256") == payload_sha256
        and request.get("safe_base_url") == safe_base_url
        and source.get("input_sha256") == case.input_sha256
        and source.get("results_sha256") == case.source_results_sha256
        and source.get("identity_sha256") == case.source_identity_sha256
        and policy.get("low_confidence_threshold") == confidence_threshold
        and policy.get("production_correctness_threshold") == production_threshold
        and record.get("trigger_reasons") == list(trigger_reasons)
    )
    if not checks:
        raise ReviewError(f"resume identity changed for {case.case_id} review {review_index}")


def _new_record(
    *,
    case: ReviewCase,
    review_index: int,
    messages: Sequence[Mapping[str, str]],
    trigger_reasons: Sequence[str],
    endpoint: str,
    safe_base_url: str,
    api_key: str,
    timeout: float,
    max_attempts: int,
    retry_delay: float,
    confidence_threshold: float,
    production_threshold: float,
) -> dict[str, Any]:
    response, payload_sha256, attempt_count = _provider_request(
        endpoint=endpoint,
        api_key=api_key,
        messages=messages,
        timeout=timeout,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )
    judgment, content_sha256 = _judgment(response)
    source: dict[str, Any] = {
        "answer_stage": case.answer_stage,
        "results_path": case.source_results_path,
        "results_sha256": case.source_results_sha256,
        "identity_sha256": case.source_identity_sha256,
        "input_sha256": case.input_sha256,
        "query_sha256": _sha256_text(case.query),
        "reference_answer_sha256": _sha256_text(case.reference_answer),
        "selected_visible_answer_sha256": _sha256_text(case.final_answer),
        "query_char_count": len(case.query),
        "reference_answer_char_count": len(case.reference_answer),
        "selected_visible_answer_char_count": len(case.final_answer),
    }
    if case.answer_stage == "final":
        # Retain the historical aliases for downstream readers of final reviews.
        source["final_visible_answer_sha256"] = _sha256_text(case.final_answer)
        source["final_visible_answer_char_count"] = len(case.final_answer)
    return {
        "schema_version": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "phase": case.phase,
        "difficulty": case.difficulty,
        "case_id": case.case_id,
        "answer_stage": case.answer_stage,
        "review_index": review_index,
        "trigger_reasons": list(trigger_reasons),
        "source": source,
        "production": {
            "generation_correctness": case.production_correctness,
            "passed": case.production_passed,
        },
        "review_policy": {
            "low_confidence_threshold": confidence_threshold,
            "production_correctness_threshold": production_threshold,
        },
        "request": {
            "answer_stage": case.answer_stage,
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _sha256_json(list(messages)),
            "payload_sha256": payload_sha256,
            "request_identity_sha256": _request_identity_sha256(
                answer_stage=case.answer_stage,
                messages=messages,
                payload_sha256=payload_sha256,
                safe_base_url=safe_base_url,
            ),
            "safe_base_url": safe_base_url,
            "attempt_count": attempt_count,
            "visible_answer_isolation": True,
            "citations_cannot_import_unstated_information": True,
        },
        "judgment": judgment,
        "usage": _normalize_usage(response.get("usage")),
        "response_model": response.get("model") if isinstance(response.get("model"), str) else None,
        "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
        "response_content_sha256": content_sha256,
        "free_text_judgment_recorded": False,
        "raw_response_recorded": False,
        "credentials_recorded": False,
    }


def _review_case(
    *,
    case: ReviewCase,
    records: list[dict[str, Any]],
    by_key: dict[tuple[str, str, str, int], dict[str, Any]],
    output: TextIO,
    endpoint: str,
    safe_base_url: str,
    api_key: str,
    timeout: float,
    max_attempts: int,
    retry_delay: float,
    confidence_threshold: float,
    production_threshold: float,
) -> dict[str, Any]:
    first_messages = _messages(case)
    first_key = (case.phase, case.difficulty, case.case_id, 1)
    first = by_key.get(first_key)
    if first is None:
        first = _new_record(
            case=case,
            review_index=1,
            messages=first_messages,
            trigger_reasons=[],
            endpoint=endpoint,
            safe_base_url=safe_base_url,
            api_key=api_key,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            confidence_threshold=confidence_threshold,
            production_threshold=production_threshold,
        )
        _append_record(output, first)
        records.append(first)
        by_key[first_key] = first
    else:
        _validate_existing_record(
            first,
            case=case,
            review_index=1,
            prompt_sha256=_sha256_json(first_messages),
            payload_sha256=_request_payload(first_messages)[1],
            trigger_reasons=[],
            safe_base_url=safe_base_url,
            confidence_threshold=confidence_threshold,
            production_threshold=production_threshold,
        )
    first_judgment = _saved_judgment(first, case.case_id)
    reasons = _second_review_reasons(
        case,
        first_judgment,
        confidence_threshold=confidence_threshold,
        production_threshold=production_threshold,
    )
    second_key = (case.phase, case.difficulty, case.case_id, 2)
    second = by_key.get(second_key)
    if not reasons:
        if second is not None:
            raise ReviewError(f"unexpected saved second review for {case.case_id}")
        return first
    second_messages = _messages(case, second_review=True)
    if second is None:
        second = _new_record(
            case=case,
            review_index=2,
            messages=second_messages,
            trigger_reasons=reasons,
            endpoint=endpoint,
            safe_base_url=safe_base_url,
            api_key=api_key,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            confidence_threshold=confidence_threshold,
            production_threshold=production_threshold,
        )
        _append_record(output, second)
        records.append(second)
        by_key[second_key] = second
    else:
        _validate_existing_record(
            second,
            case=case,
            review_index=2,
            prompt_sha256=_sha256_json(second_messages),
            payload_sha256=_request_payload(second_messages)[1],
            trigger_reasons=reasons,
            safe_base_url=safe_base_url,
            confidence_threshold=confidence_threshold,
            production_threshold=production_threshold,
        )
    _saved_judgment(second, case.case_id)
    return second


def _validate_record_universe(
    records: Sequence[Mapping[str, Any]],
    calibration: Sequence[ReviewCase],
    plans: Sequence[DifficultyPlan],
) -> None:
    allowed = {
        (case.phase, case.difficulty, case.case_id)
        for case in (*calibration, *(case for plan in plans for case in plan.cases))
    }
    for record in records:
        identity = (
            str(record.get("phase") or ""),
            str(record.get("difficulty") or ""),
            str(record.get("case_id") or ""),
        )
        if identity not in allowed:
            raise ReviewError("saved review records contain a case outside the current run")


def _validate_calibration_precedes_evaluation(
    records: Sequence[Mapping[str, Any]],
    calibration: Sequence[ReviewCase],
    *,
    confidence_threshold: float,
    production_threshold: float,
) -> None:
    order = {case_id: index for index, case_id in enumerate(CALIBRATION_EXPECTED)}
    last_calibration_order = -1
    last_review_by_case: dict[str, int] = {}
    evaluation_seen = False
    by_key = {_record_key(record): record for record in records}
    for record in records:
        phase = record.get("phase")
        if phase == "evaluation":
            evaluation_seen = True
            continue
        if phase != "calibration":
            raise ReviewError("saved review record has an invalid phase")
        if evaluation_seen:
            raise ReviewError("saved calibration record appears after an evaluation record")
        case_id = str(record.get("case_id") or "")
        case_order = order.get(case_id)
        if case_order is None or case_order < last_calibration_order:
            raise ReviewError("saved calibration record order is invalid")
        review_index = int(record.get("review_index") or 0)
        if review_index != last_review_by_case.get(case_id, 0) + 1:
            raise ReviewError("saved calibration review order is invalid")
        last_calibration_order = case_order
        last_review_by_case[case_id] = review_index
    if not evaluation_seen:
        return
    calibration_by_id = {case.case_id: case for case in calibration}
    for case_id, expected in CALIBRATION_EXPECTED.items():
        case = calibration_by_id[case_id]
        first = by_key.get(("calibration", "easy", case_id, 1))
        if first is None:
            raise ReviewError("saved evaluation records exist before complete calibration")
        first_judgment = _saved_judgment(first, case_id)
        reasons = _second_review_reasons(
            case,
            first_judgment,
            confidence_threshold=confidence_threshold,
            production_threshold=production_threshold,
        )
        second = by_key.get(("calibration", "easy", case_id, 2))
        if bool(reasons) != (second is not None):
            raise ReviewError("saved evaluation records exist before final calibration review")
        final = second or first
        judgment = _saved_judgment(final, case_id)
        if judgment["semantic_correct"] is not expected:
            raise ReviewError("saved evaluation records follow a failed calibration")


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _aggregate(final_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    judgments = [record["judgment"] for record in final_records]
    correct_count = sum(judgment["semantic_correct"] is True for judgment in judgments)
    count = len(judgments)
    return {
        "case_count": count,
        "correct_count": correct_count,
        "incorrect_count": count - correct_count,
        "binary_accuracy": round(correct_count / count, 6) if count else None,
        "mean_semantic_score": _mean([float(judgment["semantic_score"]) for judgment in judgments]),
        "mean_confidence_score": _mean(
            [float(judgment["confidence_score"]) for judgment in judgments]
        ),
        "second_review_count": sum(record.get("review_index") == 2 for record in final_records),
        "production_correctness_conflict_review_count": sum(
            "production_correctness_conflict" in (record.get("trigger_reasons") or [])
            for record in final_records
        ),
    }


def _token_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("input_tokens", "output_tokens", "total_tokens")
    summary: dict[str, Any] = {"successful_call_count": len(records)}
    for field in fields:
        values = [
            usage.get(field)
            for record in records
            if isinstance((usage := record.get("usage")), Mapping)
        ]
        known = [value for value in values if isinstance(value, int)]
        summary[field] = sum(known)
        summary[field + "_unknown_call_count"] = len(records) - len(known)
    return summary


def _calibration_summary(
    finals: Mapping[str, Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for case_id, expected in CALIBRATION_EXPECTED.items():
        final = finals.get(case_id)
        judgment = final.get("judgment") if isinstance(final, Mapping) else None
        actual = judgment.get("semantic_correct") if isinstance(judgment, Mapping) else None
        result[case_id] = {
            "expected_semantic_correct": expected,
            "actual_semantic_correct": actual,
            "matched": actual is expected,
            "review_count": sum(
                record.get("phase") == "calibration" and record.get("case_id") == case_id
                for record in records
            ),
            "final_review_index": final.get("review_index") if isinstance(final, Mapping) else None,
            "semantic_score": (
                judgment.get("semantic_score") if isinstance(judgment, Mapping) else None
            ),
            "confidence_score": (
                judgment.get("confidence_score") if isinstance(judgment, Mapping) else None
            ),
            "final_record_sha256": (
                final.get("record_sha256") if isinstance(final, Mapping) else None
            ),
        }
    return result


def _summary(
    *,
    status: str,
    index_path: Path,
    index: Mapping[str, Any],
    registry_sha256: str,
    plans: Sequence[DifficultyPlan],
    all_records: Sequence[Mapping[str, Any]],
    calibration_finals: Mapping[str, Mapping[str, Any]],
    evaluation_finals: Mapping[tuple[str, str], Mapping[str, Any]],
    records_path: Path,
    recovery_path: Path,
    recovery_events: Sequence[Mapping[str, Any]],
    safe_base_url: str,
    confidence_threshold: float,
    production_threshold: float,
    answer_stage: str,
) -> dict[str, Any]:
    calibration = _calibration_summary(calibration_finals, all_records)
    per_difficulty: dict[str, Any] = {}
    final_target_records: list[Mapping[str, Any]] = []
    for plan in plans:
        finals = [
            evaluation_finals[(plan.difficulty, case.case_id)]
            for case in plan.cases
            if (plan.difficulty, case.case_id) in evaluation_finals
        ]
        difficulty_records = [
            record
            for record in all_records
            if record.get("phase") == "evaluation" and record.get("difficulty") == plan.difficulty
        ]
        difficulty_models = Counter(
            str(record.get("response_model") or "unknown") for record in difficulty_records
        )
        final_target_records.extend(finals)
        per_difficulty[plan.difficulty] = {
            **_aggregate(finals),
            "selected_visible_answer_empty_count": sum(
                not case.final_answer for case in plan.cases
            ),
            "response_model_counts": dict(sorted(difficulty_models.items())),
            "usage": _token_summary(difficulty_records),
            "run_id": plan.run_id,
            "dataset_path": plan.dataset_path,
            "dataset_sha256": plan.dataset_sha256,
            "results_path": plan.results_path,
            "results_sha256": plan.results_sha256,
        }
    response_models = Counter(
        str(record.get("response_model") or "unknown") for record in all_records
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": status,
        "generated_at": _utc_now(),
        "label": index.get("label"),
        "answer_stage": answer_stage,
        "review_iteration": REVIEW_ITERATION,
        "input": {
            "answer_stage": answer_stage,
            "run_index_path": _relative_display(index_path),
            "run_index_sha256": _sha256_file(index_path),
            "baseline_registry_path": _relative_display(BASELINE_REGISTRY_PATH),
            "baseline_registry_sha256": registry_sha256,
            "selection": index.get("selection"),
            "mode": index.get("mode"),
        },
        "judge": {
            "script_path": _relative_display(SCRIPT_PATH),
            "script_sha256": _sha256_file(SCRIPT_PATH),
            "requested_model": JUDGE_MODEL,
            "response_model_counts": dict(sorted(response_models.items())),
            "temperature": JUDGE_TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": _sha256_text(SYSTEM_PROMPT),
            "evaluation_answer_stage": answer_stage,
            "calibration_answer_stage": "final",
            "missing_selected_answer_treated_as_empty": True,
            "safe_base_url": safe_base_url,
            "visible_answer_only": True,
            "citation_content_supplied": False,
            "expected_terms_supplied": False,
            "production_metrics_supplied": False,
            "free_text_judgment_recorded": False,
            "query_reference_answer_text_recorded_in_review_output": False,
        },
        "review_policy": {
            "binary_correct_rule": "semantic_correct=true only",
            "low_confidence_second_review_below": confidence_threshold,
            "production_correctness_binary_threshold": production_threshold,
            "production_correctness_conflict_second_review_enabled": (
                answer_stage == "final"
            ),
            "second_review_is_final": True,
        },
        "calibration": calibration,
        "calibration_passed": all(item["matched"] for item in calibration.values()),
        "overall": _aggregate(final_target_records),
        "per_difficulty": per_difficulty,
        "usage": _token_summary(all_records),
        "record_count": len(all_records),
        "records_path": _relative_display(records_path),
        "records_sha256": _sha256_file(records_path) if records_path.is_file() else None,
        "resume_recovery": {
            "event_count": len(recovery_events),
            "audit_path": _relative_display(recovery_path) if recovery_path.is_file() else None,
            "audit_sha256": _sha256_file(recovery_path) if recovery_path.is_file() else None,
            "raw_truncated_tail_recorded": False,
        },
        "credentials_recorded": False,
        "raw_provider_responses_recorded": False,
    }


def _write_checksums(output_root: Path, paths: Sequence[Path]) -> None:
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in sorted(paths, key=lambda item: item.name)
    ]
    _write_text_atomic(output_root / "SHA256SUMS", "".join(lines))


def _review_output_root(label: str, answer_stage: str) -> Path:
    iteration_root = REVIEWS_ROOT / label / REVIEW_ITERATION
    legacy_artifacts = (
        iteration_root / "semantic-correctness-review.jsonl",
        iteration_root / "summary.json",
        iteration_root / "resume-recovery.json",
        iteration_root / "SHA256SUMS",
    )
    # Before answer stages existed, final reviews were written directly under
    # the iteration root.  Continue those exact records in place.  Every new
    # review uses an explicit stage component, so pre-grounding records can
    # never share a JSONL file with the historical final-answer review.
    if answer_stage == "final" and any(path.is_file() for path in legacy_artifacts):
        return iteration_root
    return iteration_root / answer_stage


def _self_test() -> None:
    endpoint, display = _sanitize_endpoint(
        "https://user:secret@example.invalid/v1?api_key=do-not-record"
    )
    assert endpoint == "https://example.invalid/v1/chat/completions"
    assert display == "https://example.invalid/v1"
    assert "secret" not in endpoint + display and "api_key" not in endpoint + display
    sample = ReviewCase(
        phase="evaluation",
        difficulty="easy",
        case_id="JOIN-ONLY",
        answer_stage="final",
        query="问题",
        reference_answer="答案",
        final_answer="最终回答 [E1]",
        production_correctness=0.0,
        production_passed=False,
        source_results_path="results.jsonl",
        source_results_sha256="a" * 64,
        source_identity_sha256="b" * 64,
    )
    messages = _messages(sample)
    serialized = _canonical_json(messages)
    assert "JOIN-ONLY" not in serialized
    assert "production" not in serialized
    assert "expected_terms" not in serialized
    assert "最终回答 [E1]" in serialized
    second_serialized = _canonical_json(_messages(sample, second_review=True))
    assert "prior_judgment" not in second_serialized
    fake_content = _canonical_json(
        {
            "semantic_correct": True,
            "semantic_status": "correct",
            "semantic_score": 1,
            "confidence_score": 0.9,
            "answer_claims": ["不得落盘的原文"],
            "missing_or_incorrect_aspects": [],
            "reasoning_summary": "不得落盘的自由文本",
        }
    )
    judgment, _ = _judgment({"choices": [{"message": {"content": fake_content}}]})
    assert "answer_claims" not in judgment and "reasoning_summary" not in judgment
    assert judgment["answer_claim_count"] == 1
    assert _second_review_reasons(
        sample,
        {"semantic_correct": True, "confidence_score": 0.9},
        confidence_threshold=0.8,
        production_threshold=0.8,
    ) == ["production_correctness_conflict"]
    pre_sample = ReviewCase(
        phase="evaluation",
        difficulty="easy",
        case_id="JOIN-ONLY",
        answer_stage="pre_grounding",
        query=sample.query,
        reference_answer=sample.reference_answer,
        final_answer=sample.final_answer,
        production_correctness=sample.production_correctness,
        production_passed=sample.production_passed,
        source_results_path=sample.source_results_path,
        source_results_sha256=sample.source_results_sha256,
        source_identity_sha256=sample.source_identity_sha256,
    )
    assert _messages(pre_sample) == messages
    assert pre_sample.input_sha256 != sample.input_sha256
    assert _selected_visible_answer(
        {"answer": "最终", "pre_grounding_answer": "生成后、落地前"},
        "STAGE-TEST",
        "pre_grounding",
    ) == "生成后、落地前"
    assert _second_review_reasons(
        pre_sample,
        {"semantic_correct": True, "confidence_score": 0.9},
        confidence_threshold=0.8,
        production_threshold=0.8,
    ) == []
    assert _second_review_reasons(
        pre_sample,
        {"semantic_correct": True, "confidence_score": 0.7},
        confidence_threshold=0.8,
        production_threshold=0.8,
    ) == ["low_judge_confidence"]
    payload_sha256 = _request_payload(messages)[1]
    final_request_identity = _request_identity_sha256(
        answer_stage="final",
        messages=messages,
        payload_sha256=payload_sha256,
        safe_base_url=display,
    )
    pre_request_identity = _request_identity_sha256(
        answer_stage="pre_grounding",
        messages=messages,
        payload_sha256=payload_sha256,
        safe_base_url=display,
    )
    assert final_request_identity != pre_request_identity
    assert _review_output_root("self-test-no-archive", "final").name == "final"
    assert _review_output_root("self-test-no-archive", "pre_grounding").name == "pre_grounding"
    print(
        json.dumps(
            {
                "status": "self_test_passed",
                "network_calls": 0,
                "model": JUDGE_MODEL,
                "temperature": JUDGE_TEMPERATURE,
                "answer_stages": list(ANSWER_STAGES),
                "pre_grounding_production_conflict_review": False,
                "prompt_template_sha256": _sha256_text(SYSTEM_PROMPT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--label", type=_safe_label)
    source.add_argument("--run-index", help="project-relative or absolute remediation run-index")
    parser.add_argument(
        "--answer-stage",
        choices=ANSWER_STAGES,
        default="final",
        help="archived visible answer to review (default: final)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--confirm-full-review", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--request-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--max-attempts", type=_positive_int, default=3)
    parser.add_argument("--retry-delay", type=_positive_float, default=1.0)
    parser.add_argument("--low-confidence-threshold", type=_fraction, default=0.8)
    parser.add_argument("--production-correctness-threshold", type=_fraction, default=0.8)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic static checks without reading an archive or calling a provider",
    )
    args = parser.parse_args()
    if not args.self_test and not (args.label or args.run_index):
        parser.error("one of --label or --run-index is required")
    if args.self_test and (args.label or args.run_index or args.execute):
        parser.error("--self-test cannot be combined with an archive or --execute")
    return args


def main() -> int:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return 0

    index_path, index = _resolve_index(args.label, args.run_index)
    registry, registry_sha256 = _baseline_registry(index)
    plans = _build_target_plan(index_path, index, registry, args.answer_stage)
    calibration_cases = _build_calibration_cases(registry, registry_sha256)
    target_count = sum(len(plan.cases) for plan in plans)
    # Prompt revisions are separate immutable review attempts.  A failed
    # calibration remains auditable instead of being overwritten by a later
    # judge contract.
    output_root = _review_output_root(str(index["label"]), args.answer_stage)
    records_path = output_root / "semantic-correctness-review.jsonl"
    summary_path = output_root / "summary.json"
    recovery_path = output_root / "resume-recovery.json"
    dry_plan = {
        "status": "dry_run",
        "network_calls": 0,
        "label": index["label"],
        "answer_stage": args.answer_stage,
        "calibration_answer_stage": "final",
        "run_index_path": _relative_display(index_path),
        "run_index_sha256": _sha256_file(index_path),
        "model": JUDGE_MODEL,
        "review_iteration": REVIEW_ITERATION,
        "temperature": JUDGE_TEMPERATURE,
        "calibration": [
            {"case_id": case_id, "expected_semantic_correct": expected}
            for case_id, expected in CALIBRATION_EXPECTED.items()
        ],
        "case_counts": {plan.difficulty: len(plan.cases) for plan in plans},
        "selected_visible_answer_empty_counts": {
            plan.difficulty: sum(not case.final_answer for case in plan.cases)
            for plan in plans
        },
        "target_case_count": target_count,
        "maximum_successful_calls_with_second_reviews": 2 * (target_count + len(calibration_cases)),
        "output_directory": _relative_display(output_root),
        "output_identity": {
            "label": index["label"],
            "review_iteration": REVIEW_ITERATION,
            "answer_stage": args.answer_stage,
        },
        "production_correctness_conflict_second_review_enabled": (
            args.answer_stage == "final"
        ),
        "resume": bool(args.resume),
        "execution_requires": [
            "--execute",
            "--acknowledge-cost",
            "--confirm-full-review",
        ],
    }
    if not args.execute:
        print(json.dumps(dry_plan, ensure_ascii=False, indent=2))
        return 0
    if not args.acknowledge_cost or not args.confirm_full_review:
        raise ReviewError("--execute requires --acknowledge-cost and --confirm-full-review")
    if not args.base_url:
        raise ReviewError("OPENAI_BASE_URL or --base-url is required")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ReviewError(f"API key environment variable is missing: {args.api_key_env}")
    endpoint, safe_base_url = _sanitize_endpoint(args.base_url)

    if output_root.exists() and not args.resume:
        raise ReviewError("semantic review output already exists; use --resume")
    if args.resume and not output_root.is_dir():
        raise ReviewError("--resume requires an existing semantic review output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    recovery_events = _load_recovery_events(recovery_path)
    if args.resume:
        recovery = _repair_incomplete_jsonl_tail(records_path)
        if recovery is not None:
            recovery_events.append(recovery)
            _write_json_atomic(
                recovery_path,
                {"schema_version": RECOVERY_SCHEMA, "events": recovery_events},
            )
    records, by_key = _load_records(records_path)
    _validate_record_universe(records, calibration_cases, plans)
    _validate_calibration_precedes_evaluation(
        records,
        calibration_cases,
        confidence_threshold=args.low_confidence_threshold,
        production_threshold=args.production_correctness_threshold,
    )
    calibration_finals: dict[str, Mapping[str, Any]] = {}
    evaluation_finals: dict[tuple[str, str], Mapping[str, Any]] = {}

    with records_path.open("a", encoding="utf-8", newline="\n") as output:
        for case in calibration_cases:
            final = _review_case(
                case=case,
                records=records,
                by_key=by_key,
                output=output,
                endpoint=endpoint,
                safe_base_url=safe_base_url,
                api_key=api_key,
                timeout=args.request_timeout,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                confidence_threshold=args.low_confidence_threshold,
                production_threshold=args.production_correctness_threshold,
            )
            calibration_finals[case.case_id] = final
            judgment = final["judgment"]
            print(
                f"[calibration] {case.case_id}: semantic_correct="
                f"{judgment['semantic_correct']} review={final['review_index']}",
                flush=True,
            )
            if judgment["semantic_correct"] is not CALIBRATION_EXPECTED[case.case_id]:
                summary = _summary(
                    status="calibration_failed",
                    index_path=index_path,
                    index=index,
                    registry_sha256=registry_sha256,
                    plans=plans,
                    all_records=records,
                    calibration_finals=calibration_finals,
                    evaluation_finals=evaluation_finals,
                    records_path=records_path,
                    recovery_path=recovery_path,
                    recovery_events=recovery_events,
                    safe_base_url=safe_base_url,
                    confidence_threshold=args.low_confidence_threshold,
                    production_threshold=args.production_correctness_threshold,
                    answer_stage=args.answer_stage,
                )
                _write_json_atomic(summary_path, summary)
                checksum_paths = [records_path, summary_path]
                if recovery_path.is_file():
                    checksum_paths.append(recovery_path)
                _write_checksums(output_root, checksum_paths)
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 3

        for plan in plans:
            for position, case in enumerate(plan.cases, start=1):
                final = _review_case(
                    case=case,
                    records=records,
                    by_key=by_key,
                    output=output,
                    endpoint=endpoint,
                    safe_base_url=safe_base_url,
                    api_key=api_key,
                    timeout=args.request_timeout,
                    max_attempts=args.max_attempts,
                    retry_delay=args.retry_delay,
                    confidence_threshold=args.low_confidence_threshold,
                    production_threshold=args.production_correctness_threshold,
                )
                evaluation_finals[(plan.difficulty, case.case_id)] = final
                print(
                    f"[{plan.difficulty}] {position}/{len(plan.cases)} {case.case_id}: "
                    f"semantic_correct={final['judgment']['semantic_correct']} "
                    f"review={final['review_index']}",
                    flush=True,
                )

    summary = _summary(
        status="succeeded",
        index_path=index_path,
        index=index,
        registry_sha256=registry_sha256,
        plans=plans,
        all_records=records,
        calibration_finals=calibration_finals,
        evaluation_finals=evaluation_finals,
        records_path=records_path,
        recovery_path=recovery_path,
        recovery_events=recovery_events,
        safe_base_url=safe_base_url,
        confidence_threshold=args.low_confidence_threshold,
        production_threshold=args.production_correctness_threshold,
        answer_stage=args.answer_stage,
    )
    _write_json_atomic(summary_path, summary)
    checksum_paths = [records_path, summary_path]
    if recovery_path.is_file():
        checksum_paths.append(recovery_path)
    _write_checksums(output_root, checksum_paths)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
