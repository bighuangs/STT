#!/usr/bin/env python3
"""Validate post-run certification and locally register canonical baselines.

This command performs no HTTP or model calls.  It is a dry run unless
``--write`` is supplied, and it never creates a legacy ``baselines.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
FINAL_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (FINAL_ROOT, *FINAL_ROOT.parents)
        if (candidate / "pyproject.toml").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:  # pragma: no cover - repository layout guard
    raise RuntimeError("cannot locate repository root")
sys.path.insert(0, str(FINAL_ROOT))
import run_final_live as live  # noqa: E402
import failure_analysis as failures  # noqa: E402


AUTHORIZATION_SCHEMA = "stt.eval.baseline-registration-authorization.v1"
CALIBRATION_SCHEMA = "stt.eval.evaluator-calibration-approval.v1"
BASELINE_INDEX_SCHEMA = "stt.eval.canonical-baseline-index.v1"
AUTHORIZATION_PATH = FINAL_ROOT / "certification" / "baseline-registration-authorization.json"
OUTPUT_PATH = FINAL_ROOT / "baselines" / "baseline-index.json"
CALIBRATION_FIXTURE_PATH = FINAL_ROOT / "contracts" / "blind-review-calibration.json"
REVIEW_RECORD_SCHEMA = "stt.eval.semantic-correctness-review-record.v1"
REVIEW_SUMMARY_SCHEMA = "stt.eval.semantic-correctness-review-summary.v1"
REVIEW_PROMPT_VERSION = "stt.semantic-visible-answer-judge.v2"
JUDGE_MODEL = "qwen3.7-max"


class PromotionError(RuntimeError):
    """Fail-closed local certification error."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read valid JSON: {path}") from exc


def _sha256_file(path: Path) -> str:
    return live._sha256_file(path, root=FINAL_ROOT, label="promotion evidence")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_sha256(record: Mapping[str, Any]) -> str:
    return live._sha256_json(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _visible_answer(row: Mapping[str, Any], case_id: str) -> str:
    observation = row.get("observation")
    if not isinstance(observation, Mapping):
        raise PromotionError(f"{case_id} archived result has no observation")
    envelope = observation.get("answer_envelope")
    envelope_answer = envelope.get("answer") if isinstance(envelope, Mapping) else None
    legacy_answer = observation.get("answer")
    if envelope_answer is not None and not isinstance(envelope_answer, str):
        raise PromotionError(f"{case_id} answer_envelope.answer is not text")
    if legacy_answer is not None and not isinstance(legacy_answer, str):
        raise PromotionError(f"{case_id} observation.answer is not text")
    if isinstance(envelope_answer, str):
        if isinstance(legacy_answer, str) and legacy_answer != envelope_answer:
            raise PromotionError(f"{case_id} final visible answer fields conflict")
        return envelope_answer
    return legacy_answer if isinstance(legacy_answer, str) else ""


def _validate_judgment(record: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    judgment = record.get("judgment")
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
    if not isinstance(judgment, Mapping) or set(judgment) != required:
        raise PromotionError(f"{case_id} blind judgment schema is invalid")
    correct = judgment.get("semantic_correct")
    status = judgment.get("semantic_status")
    score = judgment.get("semantic_score")
    confidence = judgment.get("confidence_score")
    if (
        type(correct) is not bool
        or status not in {"correct", "partial", "incorrect"}
        or (status == "correct") is not correct
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
        or (status == "correct" and float(score) != 1)
        or (status == "partial" and not 0 < float(score) < 1)
        or (status == "incorrect" and float(score) != 0)
    ):
        raise PromotionError(f"{case_id} blind judgment values conflict")
    for field in (
        "answer_claim_count",
        "missing_or_incorrect_aspect_count",
        "reasoning_char_count",
    ):
        value = judgment.get(field)
        if type(value) is not int or value < 0:
            raise PromotionError(f"{case_id} blind judgment count is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(judgment.get("redacted_detail_sha256") or "")):
        raise PromotionError(f"{case_id} blind judgment detail SHA is invalid")
    return judgment


def _relative_final(path: Path) -> str:
    try:
        return path.resolve().relative_to(FINAL_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PromotionError(f"path escapes final suite root: {path}") from exc


def _relative_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PromotionError(f"path escapes repository root: {path}") from exc


def _reject_reparse_chain(path: Path, *, label: str) -> None:
    absolute = path if path.is_absolute() else FINAL_ROOT / path
    try:
        relative = absolute.absolute().relative_to(FINAL_ROOT.absolute())
    except ValueError as exc:
        raise PromotionError(f"{label} escapes final suite root") from exc
    current = FINAL_ROOT.absolute()
    if current.exists() and live._is_reparse_point(current):
        raise PromotionError("final suite root cannot be a symlink or reparse point")
    for part in relative.parts:
        current = current / part
        if current.exists() and live._is_reparse_point(current):
            raise PromotionError(f"{label} contains a symlink or reparse point")


def _contained(relative: str, *, label: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise PromotionError(f"{label} path must be final-root-relative")
    lexical = FINAL_ROOT / relative
    _reject_reparse_chain(lexical, label=label)
    path = lexical.resolve()
    try:
        path.relative_to(FINAL_ROOT.resolve())
    except ValueError as exc:
        raise PromotionError(f"{label} path escapes final suite root") from exc
    return path


def _project_or_final_path(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        lexical = candidate
    elif candidate.parts and candidate.parts[0] == "docs":
        lexical = PROJECT_ROOT / candidate
    else:
        lexical = FINAL_ROOT / candidate
    path = lexical.resolve()
    try:
        path.relative_to(FINAL_ROOT.resolve())
    except ValueError as exc:
        raise PromotionError(f"{label} escapes final suite root") from exc
    _reject_reparse_chain(lexical, label=label)
    return path


def _receipt(
    value: Any,
    *,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PromotionError(f"authorization has no {label} receipt")
    if value.get("path") != _relative_final(expected_path):
        raise PromotionError(f"authorization {label} path does not match")
    if not expected_path.is_file() or value.get("sha256") != _sha256_file(expected_path):
        raise PromotionError(f"authorization {label} SHA does not match")


def _load_authorization(path: Path, label: str) -> dict[str, Any]:
    _reject_reparse_chain(path, label="authorization")
    if path.resolve() != AUTHORIZATION_PATH.resolve():
        try:
            path.resolve().relative_to((FINAL_ROOT / "certification").resolve())
        except ValueError as exc:
            raise PromotionError("authorization must remain under certification/") from exc
    value = _read_json(path)
    if not isinstance(value, dict):
        raise PromotionError("authorization must contain an object")
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "approved": True,
        "baseline_registration_allowed": True,
        "label": label,
        "suite_key": live.EXPECTED_SUITE_KEY,
    }
    changed = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if changed:
        raise PromotionError("authorization identity is invalid: " + ", ".join(changed))
    return value


def _validate_index(
    label: str,
    suite: live.SuiteSnapshot,
    authorization: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    index_path = FINAL_ROOT / "runs" / label / "run-index.json"
    index = _read_json(index_path)
    if not isinstance(index, dict):
        raise PromotionError("candidate run index must contain an object")
    expected = {
        "schema_version": live.INDEX_SCHEMA,
        "label": label,
        "role": "candidate_root_run",
        "canonical": False,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "mode": "live",
        "baseline_run_id": None,
        "status": "succeeded",
        "planned_difficulties": list(live.DIFFICULTIES),
    }
    changed = [key for key, expected_value in expected.items() if index.get(key) != expected_value]
    if changed:
        raise PromotionError("candidate run index is not promotable: " + ", ".join(changed))
    _receipt(
        authorization.get("activation"),
        expected_path=suite.activation_path,
        label="activation",
    )
    _receipt(
        authorization.get("candidate_run_index"),
        expected_path=index_path,
        label="candidate_run_index",
    )
    registry_receipt = index.get("candidate_roots")
    if not isinstance(registry_receipt, Mapping):
        raise PromotionError("candidate run index has no candidate-roots receipt")
    registry_path = _contained(str(registry_receipt.get("path") or ""), label="candidate roots")
    if not registry_path.is_file() or registry_receipt.get("sha256") != _sha256_file(registry_path):
        raise PromotionError("candidate-roots receipt changed")
    _receipt(
        authorization.get("candidate_roots"),
        expected_path=registry_path,
        label="candidate_roots",
    )
    registry = _read_json(registry_path)
    if not isinstance(registry, dict):
        raise PromotionError("candidate-roots registry must contain an object")
    if (
        registry.get("schema_version") != live.CANDIDATE_REGISTRY_SCHEMA
        or registry.get("status") != "succeeded"
        or registry.get("role") != "candidate_root_run"
        or registry.get("canonical") is not False
        or registry.get("label") != label
    ):
        raise PromotionError("candidate-roots registry identity is invalid")
    registry_baselines = registry.get("baselines")
    index_entries = index.get("difficulties")
    if not isinstance(registry_baselines, Mapping) or not isinstance(index_entries, Mapping):
        raise PromotionError("candidate roots or run-index difficulties are missing")
    for difficulty in live.DIFFICULTIES:
        candidate = registry_baselines.get(difficulty)
        entry = index_entries.get(difficulty)
        if not isinstance(candidate, Mapping) or not isinstance(entry, Mapping):
            raise PromotionError(f"candidate root identity is missing for {difficulty}")
        dataset = candidate.get("dataset")
        expected_dataset = {
            "id": entry.get("dataset_id"),
            "key": entry.get("dataset_key"),
            "version": entry.get("dataset_version"),
            "checksum": entry.get("dataset_checksum"),
            "case_count": entry.get("selected_case_count"),
            "source_sha256": entry.get("dataset_source_sha256"),
        }
        if (
            candidate.get("role") != "candidate_root_run"
            or candidate.get("canonical") is not False
            or candidate.get("difficulty") != difficulty
            or candidate.get("run_id") != entry.get("run_id")
            or candidate.get("baseline_run_id") is not None
            or candidate.get("artifact_copy_manifest") != entry.get("artifact_copy_manifest")
            or candidate.get("server_artifact_receipt") != entry.get("server_artifact_receipt")
            or candidate.get("target_result") != entry.get("target_result")
            or not isinstance(dataset, Mapping)
            or any(dataset.get(key) != value for key, value in expected_dataset.items())
        ):
            raise PromotionError(f"candidate root differs from run index for {difficulty}")
    return index_path, index, registry_path, registry


def _validate_results(
    index: Mapping[str, Any],
    authorization: Mapping[str, Any],
    suite: live.SuiteSnapshot,
) -> dict[str, dict[str, Any]]:
    entries = index.get("difficulties")
    receipts = authorization.get("results")
    if not isinstance(entries, Mapping) or not isinstance(receipts, Mapping):
        raise PromotionError("difficulty entries/results authorization are missing")
    if set(entries) != set(live.DIFFICULTIES) or set(receipts) != set(live.DIFFICULTIES):
        raise PromotionError(
            "run-index/results authorization must contain exactly three difficulties"
        )
    label_root = FINAL_ROOT / "runs" / str(index.get("label") or "")
    verified: dict[str, dict[str, Any]] = {}
    for difficulty in live.DIFFICULTIES:
        entry = entries.get(difficulty)
        receipt = receipts.get(difficulty)
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise PromotionError(f"{difficulty} candidate artifacts are not archived")
        if not isinstance(receipt, Mapping):
            raise PromotionError(f"authorization has no {difficulty} results receipt")
        spec = suite.datasets[difficulty]
        expected_dataset_identity = {
            "difficulty": difficulty,
            "role": "candidate_root_run",
            "mode": "live",
            "baseline_run_id": None,
            "dataset_key": spec.manifest["dataset_key"],
            "dataset_version": spec.manifest["version"],
            "dataset_source_path": _relative_project(spec.path),
            "dataset_source_sha256": spec.file_sha256,
            "selected_case_count": live.EXPECTED_CASE_COUNT,
            "selected_case_ids": list(spec.case_ids),
        }
        identity_changes = [
            key
            for key, expected_value in expected_dataset_identity.items()
            if entry.get(key) != expected_value
        ]
        if identity_changes:
            raise PromotionError(
                f"{difficulty} candidate identity changed: " + ", ".join(identity_changes)
            )
        try:
            live._verify_archived_config_identity(
                index,
                entry,
                difficulty=difficulty,
                spec=spec,
            )
        except live.FinalLiveError as exc:
            raise PromotionError(str(exc)) from exc
        try:
            _, path, sha256 = live._verify_artifact_chain(
                suite_root=FINAL_ROOT,
                label_root=label_root,
                entry=entry,
                difficulty=difficulty,
            )
        except live.FinalLiveError as exc:
            raise PromotionError(str(exc)) from exc
        if entry.get("results_path") != _relative_final(path):
            raise PromotionError(f"{difficulty} results path differs from artifact receipt")
        expected_receipt = {
            "path": _relative_final(path),
            "sha256": sha256,
            "run_id": entry.get("run_id"),
            "correct_count": (entry.get("target_result") or {}).get("correct_count"),
            "case_count": live.EXPECTED_CASE_COUNT,
        }
        changed = [
            key
            for key, expected_value in expected_receipt.items()
            if receipt.get(key) != expected_value
        ]
        if changed:
            raise PromotionError(
                f"authorization {difficulty} results differ: " + ", ".join(changed)
            )
        rows = live._read_results_jsonl(path, root=FINAL_ROOT)
        target = live._target_count_from_rows(
            rows,
            expected_run_id=str(entry.get("run_id") or ""),
            expected_case_ids=suite.datasets[difficulty].case_ids,
        )
        if target != entry.get("target_result") or sha256 != entry.get("results_sha256"):
            raise PromotionError(f"{difficulty} archived target evidence changed")
        required = live.TARGET_CORRECT_COUNTS[difficulty]
        if required is not None and target["correct_count"] < required:
            raise PromotionError(f"{difficulty} did not reach {required}/50")
        verified[difficulty] = {
            "path": _relative_final(path),
            "sha256": sha256,
            **target,
        }
    expected_targets = {
        "easy": {"correct_count": 48, "case_count": 50},
        "medium": {"correct_count": 43, "case_count": 50},
        "hard": {"correct_count": None, "case_count": 50},
    }
    if authorization.get("target_counts") != expected_targets:
        raise PromotionError("authorization target_counts do not match the frozen contract")
    return verified


def _validate_failure_analysis(
    authorization: Mapping[str, Any],
    *,
    label: str,
    index_path: Path,
) -> dict[str, Any]:
    """Recompute the exact wrong-case analysis and match its authorization receipt."""

    try:
        verified = failures.validate_failure_artifacts(
            FINAL_ROOT,
            label,
            index_path=index_path,
        )
    except failures.FailureAnalysisError as exc:
        raise PromotionError(f"failure analysis is invalid: {exc}") from exc
    expected = {"approved": True, **verified}
    receipt = authorization.get("failure_analysis")
    if not isinstance(receipt, Mapping) or dict(receipt) != expected:
        raise PromotionError("authorization failure_analysis receipt differs from current evidence")
    if verified["evaluator_error_count"] != 0 or verified["promotion_allowed"] is not True:
        raise PromotionError("failure analysis confirms an evaluator error; rerun is required")
    return expected


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise PromotionError(f"cannot read audit JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromotionError(f"invalid audit JSONL row {line_number}") from exc
            if not isinstance(value, dict):
                raise PromotionError(f"audit JSONL row {line_number} is not an object")
            records.append(value)
    return records


def _review_expectations(
    index: Mapping[str, Any], suite: live.SuiteSnapshot, index_path: Path
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, bool],
    dict[str, bool],
]:
    expectations: dict[tuple[str, str, str], dict[str, Any]] = {}
    production_labels: dict[str, bool] = {}
    for difficulty in live.DIFFICULTIES:
        entry = index["difficulties"][difficulty]
        results_path = _contained(str(entry["results_path"]), label=f"{difficulty} results")
        rows = live._read_results_jsonl(results_path, root=FINAL_ROOT)
        by_case = {str(row.get("case_id") or ""): row for row in rows if isinstance(row, Mapping)}
        manifest_rows = suite.datasets[difficulty].manifest["cases"]
        for case in manifest_rows:
            case_id = str(case["case_id"])
            row = by_case.get(case_id)
            if not isinstance(row, Mapping):
                raise PromotionError(f"missing archived result for {case_id}")
            query = str(case.get("query") or "")
            reference = str(case.get("reference_answer") or "")
            answer = _visible_answer(row, case_id)
            outcome = row["metrics"]["metrics"]["case_outcome"]["final_answer_correct"]
            production_labels[f"{difficulty}:{case_id}"] = outcome
            expectations[("evaluation", difficulty, case_id)] = {
                "query": query,
                "reference_answer": reference,
                "final_visible_answer": answer,
                "source_results_path": _relative_project(results_path),
                "source_results_sha256": entry["results_sha256"],
                "source_identity_sha256": _sha256_file(index_path),
            }
    fixture = _read_json(CALIBRATION_FIXTURE_PATH)
    rows = fixture.get("cases") if isinstance(fixture, Mapping) else None
    if (
        not isinstance(rows, list)
        or not rows
        or fixture.get("schema_version") != "stt.eval.semantic-judge-calibration.v1"
    ):
        raise PromotionError("frozen blind calibration fixture is invalid")
    fixture_sha = _sha256_file(CALIBRATION_FIXTURE_PATH)
    calibration_expected: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("expected_semantic_correct")) is not bool:
            raise PromotionError("frozen blind calibration case is invalid")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in calibration_expected:
            raise PromotionError("frozen blind calibration case ids are invalid")
        calibration_expected[case_id] = row["expected_semantic_correct"]
        expectations[("calibration", "easy", case_id)] = {
            "query": str(row.get("query") or ""),
            "reference_answer": str(row.get("reference_answer") or ""),
            "final_visible_answer": str(row.get("final_visible_answer") or ""),
            "source_results_path": _relative_project(CALIBRATION_FIXTURE_PATH),
            "source_results_sha256": fixture_sha,
            "source_identity_sha256": fixture_sha,
        }
    return expectations, calibration_expected, production_labels


def _validate_review_record(
    record: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    phase: str,
    difficulty: str,
    case_id: str,
    safe_base_url: str,
) -> tuple[int, Mapping[str, Any]]:
    request = record.get("request")
    source = record.get("source")
    review_index = record.get("review_index")
    if (
        record.get("schema_version") != REVIEW_RECORD_SCHEMA
        or record.get("phase") != phase
        or record.get("difficulty") != difficulty
        or record.get("case_id") != case_id
        or record.get("answer_stage") != "final"
        or review_index not in {1, 2}
        or record.get("credentials_recorded") is not False
        or record.get("raw_response_recorded") is not False
        or record.get("free_text_judgment_recorded") is not False
        or record.get("response_model") != JUDGE_MODEL
        or not isinstance(request, Mapping)
        or request.get("answer_stage") != "final"
        or request.get("model") != JUDGE_MODEL
        or request.get("temperature") != 0
        or request.get("prompt_version") != REVIEW_PROMPT_VERSION
        or request.get("safe_base_url") != safe_base_url
        or request.get("visible_answer_isolation") is not True
        or request.get("citations_cannot_import_unstated_information") is not True
        or type(request.get("attempt_count")) is not int
        or request.get("attempt_count") < 1
        or not isinstance(source, Mapping)
        or source.get("answer_stage") != "final"
    ):
        raise PromotionError(f"{case_id} blind review identity/provenance is invalid")
    for field in ("prompt_sha256", "payload_sha256", "request_identity_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(request.get(field) or "")):
            raise PromotionError(f"{case_id} blind request {field} is invalid")
    query = str(expected["query"])
    reference = str(expected["reference_answer"])
    answer = str(expected["final_visible_answer"])
    input_sha = live._sha256_json(
        {"query": query, "reference_answer": reference, "final_visible_answer": answer}
    )
    expected_source = {
        "answer_stage": "final",
        "results_path": expected["source_results_path"],
        "results_sha256": expected["source_results_sha256"],
        "identity_sha256": expected["source_identity_sha256"],
        "input_sha256": input_sha,
        "query_sha256": _sha256_text(query),
        "reference_answer_sha256": _sha256_text(reference),
        "selected_visible_answer_sha256": _sha256_text(answer),
        "selected_visible_answer_char_count": len(answer),
        "final_visible_answer_sha256": _sha256_text(answer),
        "final_visible_answer_char_count": len(answer),
        "query_char_count": len(query),
        "reference_answer_char_count": len(reference),
    }
    changed = [
        field
        for field, expected_value in expected_source.items()
        if source.get(field) != expected_value
    ]
    if changed:
        raise PromotionError(f"{case_id} blind source identity changed: " + ", ".join(changed))
    reasons = record.get("trigger_reasons")
    if (
        not isinstance(reasons, list)
        or any(
            reason not in {"low_judge_confidence", "production_correctness_conflict"}
            for reason in reasons
        )
        or (review_index == 1 and reasons)
        or (review_index == 2 and not reasons)
    ):
        raise PromotionError(f"{case_id} blind review trigger sequence is invalid")
    if record.get("record_sha256") != _record_sha256(record):
        raise PromotionError(f"{case_id} blind record SHA is invalid")
    return int(review_index), _validate_judgment(record, case_id)


def _validate_blind_audit(
    authorization: Mapping[str, Any],
    *,
    label: str,
    index_path: Path,
    registry_path: Path,
    index: Mapping[str, Any],
    suite: live.SuiteSnapshot,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, bool]]:
    receipt = authorization.get("final_blind_audit")
    if not isinstance(receipt, Mapping) or receipt.get("approved") is not True:
        raise PromotionError("authorization final_blind_audit must be approved")
    summary_path = _contained(str(receipt.get("summary_path") or ""), label="blind summary")
    records_path = _contained(str(receipt.get("records_path") or ""), label="blind records")
    if receipt.get("summary_sha256") != _sha256_file(summary_path):
        raise PromotionError("blind summary SHA does not match")
    if receipt.get("records_sha256") != _sha256_file(records_path):
        raise PromotionError("blind records SHA does not match")
    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise PromotionError("blind summary must contain an object")
    expected_summary = {
        "schema_version": REVIEW_SUMMARY_SCHEMA,
        "status": "succeeded",
        "label": label,
        "answer_stage": "final",
        "calibration_passed": True,
    }
    changed = [
        key
        for key, expected_value in expected_summary.items()
        if summary.get(key) != expected_value
    ]
    if changed:
        raise PromotionError("blind audit summary is not approved: " + ", ".join(changed))
    judge = summary.get("judge")
    policy = summary.get("review_policy")
    if (
        not isinstance(judge, Mapping)
        or judge.get("requested_model") != JUDGE_MODEL
        or judge.get("temperature") != 0
        or judge.get("evaluation_answer_stage") != "final"
        or judge.get("visible_answer_only") is not True
        or judge.get("citation_content_supplied") is not False
        or judge.get("expected_terms_supplied") is not False
        or judge.get("production_metrics_supplied") is not False
        or judge.get("prompt_version") != REVIEW_PROMPT_VERSION
        or not isinstance(policy, Mapping)
        or policy.get("binary_correct_rule") != "semantic_correct=true only"
        or summary.get("credentials_recorded") is not False
        or summary.get("raw_provider_responses_recorded") is not False
    ):
        raise PromotionError("blind audit isolation/policy contract is invalid")
    summary_input = summary.get("input")
    if not isinstance(summary_input, Mapping):
        raise PromotionError("blind summary has no input receipt")
    summary_index_path = _project_or_final_path(
        str(summary_input.get("run_index_path") or ""), label="blind run index"
    )
    if summary_index_path != index_path.resolve() or summary_input.get(
        "run_index_sha256"
    ) != _sha256_file(index_path):
        raise PromotionError("blind audit is not bound to the current run index")
    summary_registry_path = _project_or_final_path(
        str(summary_input.get("baseline_registry_path") or ""),
        label="blind candidate registry",
    )
    if summary_registry_path != registry_path.resolve() or summary_input.get(
        "baseline_registry_sha256"
    ) != _sha256_file(registry_path):
        raise PromotionError("blind audit is not bound to the candidate roots")
    summary_records_path = _project_or_final_path(
        str(summary.get("records_path") or ""), label="blind records summary"
    )
    if summary_records_path != records_path.resolve() or summary.get(
        "records_sha256"
    ) != _sha256_file(records_path):
        raise PromotionError("blind summary records receipt does not match")
    overall = summary.get("overall")
    per_difficulty = summary.get("per_difficulty")
    if not isinstance(overall, Mapping) or overall.get("case_count") != 150:
        raise PromotionError("blind audit must cover exactly 150 current-run cases")
    if not isinstance(per_difficulty, Mapping) or any(
        not isinstance(per_difficulty.get(difficulty), Mapping)
        or per_difficulty[difficulty].get("case_count") != 50
        for difficulty in live.DIFFICULTIES
    ):
        raise PromotionError("blind audit must cover 50 cases per difficulty")

    expectations, calibration_expected, production_labels = _review_expectations(
        index, suite, index_path
    )
    records = _load_jsonl(records_path)
    by_case: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    evaluation_seen = False
    safe_base_url = str(judge.get("safe_base_url") or "")
    if not safe_base_url:
        raise PromotionError("blind judge safe_base_url is missing")
    for record in records:
        phase = str(record.get("phase") or "")
        difficulty = str(record.get("difficulty") or "")
        case_id = str(record.get("case_id") or "")
        key = (phase, difficulty, case_id)
        expected = expectations.get(key)
        if expected is None:
            raise PromotionError("blind JSONL contains an unknown review case")
        if phase == "evaluation":
            evaluation_seen = True
        elif phase == "calibration":
            if evaluation_seen:
                raise PromotionError("blind calibration appears after evaluation")
        else:
            raise PromotionError("blind JSONL contains an unsupported phase")
        _validate_review_record(
            record,
            expected=expected,
            phase=phase,
            difficulty=difficulty,
            case_id=case_id,
            safe_base_url=safe_base_url,
        )
        by_case.setdefault(key, []).append(record)
    if set(by_case) != set(expectations):
        raise PromotionError("blind JSONL does not cover frozen calibration plus 150 cases")
    finals: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for key, case_records in by_case.items():
        indices = [int(record["review_index"]) for record in case_records]
        if indices not in ([1], [1, 2]):
            raise PromotionError(f"blind review indices are not contiguous for {key[2]}")
        finals[key] = case_records[-1]
    calibration_summary = summary.get("calibration")
    if not isinstance(calibration_summary, Mapping) or set(calibration_summary) != set(
        calibration_expected
    ):
        raise PromotionError("blind summary calibration universe differs from frozen fixture")
    for case_id, expected_value in calibration_expected.items():
        key = ("calibration", "easy", case_id)
        final = finals[key]
        item = calibration_summary.get(case_id)
        actual = final["judgment"]["semantic_correct"]
        if (
            actual is not expected_value
            or not isinstance(item, Mapping)
            or item.get("expected_semantic_correct") is not expected_value
            or item.get("actual_semantic_correct") is not expected_value
            or item.get("matched") is not True
            or item.get("review_count") != len(by_case[key])
            or item.get("final_review_index") != final["review_index"]
            or item.get("final_record_sha256") != final["record_sha256"]
        ):
            raise PromotionError(f"blind frozen calibration failed for {case_id}")
    evaluation_finals = {
        (difficulty, case_id): record
        for (phase, difficulty, case_id), record in finals.items()
        if phase == "evaluation"
    }
    if len(evaluation_finals) != 150:
        raise PromotionError("blind JSONL does not contain 150 final evaluation judgments")
    if summary.get("record_count") != len(records):
        raise PromotionError("blind summary record_count differs from its JSONL")
    overall_correct = sum(
        bool(record["judgment"]["semantic_correct"]) for record in evaluation_finals.values()
    )
    if (
        overall.get("correct_count") != overall_correct
        or overall.get("incorrect_count") != 150 - overall_correct
    ):
        raise PromotionError("blind summary aggregate differs from final JSONL judgments")
    for difficulty in live.DIFFICULTIES:
        expected_sha = index["difficulties"][difficulty]["results_sha256"]
        difficulty_records = [
            record for key, record in evaluation_finals.items() if key[0] == difficulty
        ]
        difficulty_correct = sum(
            bool(record["judgment"]["semantic_correct"]) for record in difficulty_records
        )
        difficulty_summary = per_difficulty[difficulty]
        if (
            difficulty_summary.get("correct_count") != difficulty_correct
            or difficulty_summary.get("incorrect_count") != 50 - difficulty_correct
        ):
            raise PromotionError(f"blind {difficulty} summary differs from final JSONL judgments")
        for key, record in evaluation_finals.items():
            if key[0] != difficulty:
                continue
            source = record.get("source")
            if not isinstance(source, Mapping) or source.get("results_sha256") != expected_sha:
                raise PromotionError(f"blind {difficulty} judgment is not bound to results SHA")
    blind_labels = {
        f"{difficulty}:{case_id}": bool(record["judgment"]["semantic_correct"])
        for (difficulty, case_id), record in evaluation_finals.items()
    }
    return (
        {
            "approved": True,
            "summary_path": _relative_final(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "records_path": _relative_final(records_path),
            "records_sha256": _sha256_file(records_path),
            "case_count": 150,
            "calibration_passed": True,
            "per_difficulty": {
                difficulty: {
                    "correct_count": per_difficulty[difficulty]["correct_count"],
                    "incorrect_count": per_difficulty[difficulty]["incorrect_count"],
                    "case_count": 50,
                }
                for difficulty in live.DIFFICULTIES
            },
            "calibration_fixture": {
                "path": _relative_final(CALIBRATION_FIXTURE_PATH),
                "sha256": _sha256_file(CALIBRATION_FIXTURE_PATH),
                "case_count": len(calibration_expected),
            },
        },
        blind_labels,
        production_labels,
    )


def _stable_split(case_keys: list[str]) -> dict[str, Any]:
    calibration: list[str] = []
    holdout: list[str] = []
    for key in sorted(case_keys):
        digest = hashlib.sha256(
            (
                "stt.eval.evaluator-calibration-split.v1\0" + live.EXPECTED_SUITE_KEY + "\0" + key
            ).encode("utf-8")
        ).digest()
        (calibration if int.from_bytes(digest[:8], "big") < 2**63 else holdout).append(key)
    if not calibration or not holdout:
        raise PromotionError("stable evaluator calibration split is degenerate")
    return {
        "algorithm": "sha256-suite-difficulty-case-v1",
        "calibration_fraction": 0.5,
        "calibration_case_count": len(calibration),
        "calibration_case_keys_sha256": live._sha256_json(calibration),
        "holdout_case_count": len(holdout),
        "holdout_case_keys_sha256": live._sha256_json(holdout),
        "calibration_case_keys": calibration,
        "holdout_case_keys": holdout,
    }


def _agreement_metrics(
    case_keys: list[str], production: Mapping[str, bool], blind: Mapping[str, bool]
) -> dict[str, Any]:
    true_positive = sum(production[key] and blind[key] for key in case_keys)
    true_negative = sum(not production[key] and not blind[key] for key in case_keys)
    false_positive = sum(production[key] and not blind[key] for key in case_keys)
    false_negative = sum(not production[key] and blind[key] for key in case_keys)
    count = len(case_keys)
    return {
        "case_count": count,
        "agreement_count": true_positive + true_negative,
        "agreement_rate": round((true_positive + true_negative) / count, 6),
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
    }


def _validate_adjudication(
    item: Mapping[str, Any],
    *,
    case_key: str,
    production_value: bool,
    blind_value: bool,
    index: Mapping[str, Any],
    blind_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    path = _contained(str(item.get("path") or ""), label=f"{case_key} adjudication")
    if not path.is_file() or item.get("sha256") != _sha256_file(path):
        raise PromotionError(f"{case_key} adjudication SHA does not match")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise PromotionError(f"{case_key} adjudication must contain an object")
    difficulty, case_id = case_key.split(":", maxsplit=1)
    entry = index["difficulties"][difficulty]
    expected = {
        "schema_version": "stt.eval.evaluator-blind-adjudication.v1",
        "status": "approved",
        "approved": True,
        "resolved": True,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "label": index.get("label"),
        "difficulty": difficulty,
        "case_id": case_id,
        "run_id": entry.get("run_id"),
        "results_sha256": entry.get("results_sha256"),
        "blind_records_sha256": blind_receipt.get("records_sha256"),
        "production_final_answer_correct": production_value,
        "blind_semantic_correct": blind_value,
    }
    changed = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if changed:
        raise PromotionError(f"{case_key} adjudication identity changed: " + ", ".join(changed))
    resolution = value.get("resolution")
    if resolution == "evaluator_error_requires_rerun":
        raise PromotionError(f"{case_key} confirms an evaluator error; rerun is required")
    if resolution != "production_correct_blind_judge_error":
        raise PromotionError(f"{case_key} disagreement is not safely resolved")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or len(rationale.strip()) < 20:
        raise PromotionError(f"{case_key} adjudication rationale is insufficient")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PromotionError(f"{case_key} adjudication has no evidence")
    for position, receipt in enumerate(evidence, start=1):
        if not isinstance(receipt, Mapping):
            raise PromotionError(f"{case_key} adjudication evidence {position} is invalid")
        evidence_path = _contained(
            str(receipt.get("path") or ""),
            label=f"{case_key} adjudication evidence {position}",
        )
        if not evidence_path.is_file() or receipt.get("sha256") != _sha256_file(evidence_path):
            raise PromotionError(f"{case_key} adjudication evidence {position} SHA changed")
    return {"path": _relative_final(path), "sha256": _sha256_file(path), "resolution": resolution}


def _validate_evaluator_calibration(
    authorization: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    index_path: Path,
    production_labels: Mapping[str, bool],
    blind_labels: Mapping[str, bool],
    blind_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = authorization.get("evaluator_calibration")
    if not isinstance(receipt, Mapping) or receipt.get("approved") is not True:
        raise PromotionError("authorization evaluator_calibration must be approved")
    path = _contained(str(receipt.get("path") or ""), label="evaluator calibration")
    if not path.is_file() or receipt.get("sha256") != _sha256_file(path):
        raise PromotionError("evaluator calibration SHA does not match")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise PromotionError("evaluator calibration must contain an object")
    expected = {
        "schema_version": CALIBRATION_SCHEMA,
        "status": "approved",
        "approved": True,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "case_contract": live.CASE_CONTRACT,
        "code_revision": index.get("code_revision"),
    }
    changed = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if changed:
        raise PromotionError("evaluator calibration is not approved: " + ", ".join(changed))
    if set(production_labels) != set(blind_labels) or len(production_labels) != 150:
        raise PromotionError("evaluator calibration label universe is not the current 150 cases")
    split = _stable_split(list(production_labels))
    split_public = {key: value for key, value in split.items() if not key.endswith("_case_keys")}
    metrics = {
        "calibration": _agreement_metrics(
            split["calibration_case_keys"], production_labels, blind_labels
        ),
        "holdout": _agreement_metrics(split["holdout_case_keys"], production_labels, blind_labels),
        "overall": _agreement_metrics(sorted(production_labels), production_labels, blind_labels),
    }
    if value.get("split") != split_public or value.get("metrics") != metrics:
        raise PromotionError("evaluator calibration split/metrics do not recompute exactly")
    policy = {
        "minimum_calibration_agreement_rate": 0.8,
        "minimum_holdout_agreement_rate": 0.8,
        "maximum_confirmed_evaluator_errors": 0,
        "all_disagreements_require_adjudication": True,
    }
    if value.get("acceptance_policy") != policy:
        raise PromotionError("evaluator calibration acceptance policy changed")
    if metrics["calibration"]["agreement_rate"] < 0.8 or metrics["holdout"]["agreement_rate"] < 0.8:
        raise PromotionError("evaluator agreement is below the frozen calibration threshold")
    source_receipts = value.get("source_receipts")
    expected_sources = {
        "run_index": {
            "path": _relative_final(index_path),
            "sha256": _sha256_file(index_path),
        },
        "blind_summary": {
            "path": blind_receipt["summary_path"],
            "sha256": blind_receipt["summary_sha256"],
        },
        "blind_records": {
            "path": blind_receipt["records_path"],
            "sha256": blind_receipt["records_sha256"],
        },
    }
    if source_receipts != expected_sources:
        raise PromotionError("evaluator calibration source receipts differ from current evidence")
    disagreements = sorted(
        key for key in production_labels if production_labels[key] is not blind_labels[key]
    )
    adjudications = value.get("adjudications")
    if not isinstance(adjudications, list) or len(adjudications) != len(disagreements):
        raise PromotionError("every evaluator/blind disagreement requires one adjudication")
    by_key: dict[str, Mapping[str, Any]] = {}
    for item in adjudications:
        if not isinstance(item, Mapping):
            raise PromotionError("evaluator adjudication receipt is invalid")
        case_key = str(item.get("case_key") or "")
        if not case_key or case_key in by_key:
            raise PromotionError("evaluator adjudication case keys are invalid")
        by_key[case_key] = item
    if set(by_key) != set(disagreements):
        raise PromotionError("evaluator adjudication universe differs from disagreements")
    verified_adjudications = [
        {
            "case_key": case_key,
            **_validate_adjudication(
                by_key[case_key],
                case_key=case_key,
                production_value=production_labels[case_key],
                blind_value=blind_labels[case_key],
                index=index,
                blind_receipt=blind_receipt,
            ),
        }
        for case_key in disagreements
    ]
    if (
        value.get("unresolved_disagreement_count") != 0
        or value.get("confirmed_evaluator_error_count") != 0
    ):
        raise PromotionError("evaluator calibration has unresolved/confirmed evaluator errors")
    return {
        "approved": True,
        "path": _relative_final(path),
        "sha256": _sha256_file(path),
        "split": split_public,
        "metrics": metrics,
        "disagreement_count": len(disagreements),
        "adjudications": verified_adjudications,
    }


def _canonical_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(candidate)
    value["role"] = "canonical_baseline"
    value["canonical"] = True
    return value


def build_baseline_index(label: str, authorization_path: Path) -> dict[str, Any]:
    suite = live.load_suite_snapshot(FINAL_ROOT)
    authorization = _load_authorization(authorization_path, label)
    index_path, index, registry_path, registry = _validate_index(label, suite, authorization)
    verified_results = _validate_results(index, authorization, suite)
    failure_analysis = _validate_failure_analysis(
        authorization,
        label=label,
        index_path=index_path,
    )
    blind, blind_labels, production_labels = _validate_blind_audit(
        authorization,
        label=label,
        index_path=index_path,
        registry_path=registry_path,
        index=index,
        suite=suite,
    )
    calibration = _validate_evaluator_calibration(
        authorization,
        index,
        index_path=index_path,
        production_labels=production_labels,
        blind_labels=blind_labels,
        blind_receipt=blind,
    )
    baselines = registry.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(live.DIFFICULTIES):
        raise PromotionError("candidate roots do not contain exactly three difficulties")
    return {
        "schema_version": BASELINE_INDEX_SCHEMA,
        "status": "succeeded",
        "role": "canonical_baseline_index",
        "canonical": True,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "label": label,
        "mode": "live",
        "kb_id": index["kb_id"],
        "generated_at": live.br._utc_now(),
        "difficulty_order": list(live.DIFFICULTIES),
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "certification": {
            "authorization": {
                "path": _relative_final(authorization_path),
                "sha256": _sha256_file(authorization_path),
            },
            "activation": {
                "path": _relative_final(suite.activation_path),
                "sha256": suite.activation_sha256,
                "candidate_baseline_registration_allowed": False,
            },
            "candidate_run_index": {
                "path": _relative_final(index_path),
                "sha256": _sha256_file(index_path),
            },
            "candidate_roots": {
                "path": _relative_final(registry_path),
                "sha256": _sha256_file(registry_path),
            },
            "verified_results": verified_results,
            "failure_analysis": failure_analysis,
            "final_blind_audit": blind,
            "evaluator_calibration": calibration,
        },
        "baselines": {
            difficulty: _canonical_record(baselines[difficulty]) for difficulty in live.DIFFICULTIES
        },
    }


def _safe_label(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise argparse.ArgumentTypeError("invalid label")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, type=_safe_label)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=AUTHORIZATION_PATH,
        help="final-root certification receipt (default: certification/baseline-registration-authorization.json)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write baselines/baseline-index.json after all local checks pass",
    )
    args = parser.parse_args()
    if not args.authorization.is_absolute():
        args.authorization = (FINAL_ROOT / args.authorization).resolve()
    return args


def main() -> int:
    args = _parse_args()
    with live.LabelLock("canonical-promotion"), live.LabelLock(args.label):
        return _main_locked(args)


def _main_locked(args: argparse.Namespace) -> int:
    payload = build_baseline_index(args.label, args.authorization)
    if not args.write:
        print(
            json.dumps(
                {
                    "status": "promotion_ready",
                    "network_calls": 0,
                    "write_performed": False,
                    "output_path": _relative_final(OUTPUT_PATH),
                    "label": args.label,
                    "verified_result_counts": {
                        difficulty: payload["certification"]["verified_results"][difficulty][
                            "correct_count"
                        ]
                        for difficulty in live.DIFFICULTIES
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if OUTPUT_PATH.exists():
        raise PromotionError("canonical baseline-index.json already exists; refusing to overwrite")
    _reject_reparse_chain(OUTPUT_PATH.parent, label="canonical baselines directory")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_chain(OUTPUT_PATH.parent, label="canonical baselines directory")
    if OUTPUT_PATH.parent.resolve().parent != FINAL_ROOT.resolve():
        raise PromotionError("canonical baselines directory escapes final suite root")
    live._write_json(OUTPUT_PATH, payload)
    checksums = OUTPUT_PATH.parent / "SHA256SUMS"
    live.br._write_bytes(
        checksums,
        f"{_sha256_file(OUTPUT_PATH)}  {OUTPUT_PATH.name}\n".encode("utf-8"),
    )
    print(f"Canonical baseline index: {_relative_project(OUTPUT_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PromotionError, live.FinalLiveError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
