#!/usr/bin/env python3
"""把冻结盲审器绑定到 easy/medium 的 100 题不可晋级归档。"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_easy_medium_live as live
import scoped_run_archive as run_archive


SCRIPT_PATH = Path(__file__).resolve()
FINAL_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = live.PROJECT_ROOT
REVIEWER_PATH = (
    FINAL_ROOT.parent
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "review_semantic_correctness.py"
)
CALIBRATION_PATH = FINAL_ROOT / "contracts" / "blind-review-calibration.json"
FROZEN_CALIBRATION_SHA256 = (
    "63baee6c212a604f90c0bff13027f3f8c73fef8778d1cdbad2dc1fe506317da6"
)
RECORD_SCHEMA = "stt.eval.easy-medium-blind-review-record.v1"
SUMMARY_SCHEMA = "stt.eval.easy-medium-blind-review-summary.v1"
RECOVERY_SCHEMA = "stt.eval.easy-medium-blind-review-recovery.v1"
BLIND_REVIEW_ROLE = "non_promotable_easy_medium_blind_review"
SOURCE_CONTRACT_ROLE = "scoped_succeeded_run_derived_blind_review_source"
TARGET_CASE_COUNT = live.EXPECTED_TOTAL_CASE_COUNT
NONPROMOTABLE_AUTHORITY = {
    "canonical": False,
    "promotion_allowed": False,
    "promotable": False,
    "baseline_registration_allowed": False,
    "canonical_promotion_allowed": False,
}
SCOPED_RECORD_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "phase",
        "difficulty",
        "case_id",
        "answer_stage",
        "review_index",
        "trigger_reasons",
        "source",
        "production",
        "review_policy",
        "request",
        "judgment",
        "usage",
        "response_model",
        "response_id",
        "response_content_sha256",
        "free_text_judgment_recorded",
        "raw_response_recorded",
        "credentials_recorded",
        "role",
        "scope_id",
        *NONPROMOTABLE_AUTHORITY,
        "counts_toward_primary_target",
        "record_sha256",
    }
)


class ScopedBlindReviewError(RuntimeError):
    """表示 scoped 盲审的身份、证据或隔离契约不成立。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reviewer() -> Any:
    if not REVIEWER_PATH.is_file():
        raise ScopedBlindReviewError(f"frozen semantic reviewer is missing: {REVIEWER_PATH}")
    module_name = "_stt_easy_medium_semantic_reviewer"
    spec = importlib.util.spec_from_file_location(module_name, REVIEWER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - 固定仓库布局保证可加载
        raise ScopedBlindReviewError("cannot load the frozen semantic reviewer")
    reviewer = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = reviewer
    spec.loader.exec_module(reviewer)
    return reviewer


def _frozen_calibration() -> tuple[list[dict[str, Any]], str]:
    if not CALIBRATION_PATH.is_file():
        raise ScopedBlindReviewError("frozen blind-review calibration fixture is missing")
    fixture_sha256 = _sha256_file(CALIBRATION_PATH)
    if fixture_sha256 != FROZEN_CALIBRATION_SHA256:
        raise ScopedBlindReviewError("frozen blind-review calibration SHA changed")
    try:
        value = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopedBlindReviewError("cannot read frozen blind-review calibration") from exc
    rows = value.get("cases") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "stt.eval.semantic-judge-calibration.v1"
        or not isinstance(rows, list)
        or len(rows) != 3
    ):
        raise ScopedBlindReviewError("frozen blind-review calibration fixture is invalid")
    expected_ids = {
        "CAL-FROZEN-POLARITY-OMISSION",
        "CAL-FROZEN-COMPLETE",
        "CAL-FROZEN-PARTIAL",
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ScopedBlindReviewError("frozen blind-review calibration case is invalid")
        case_id = row.get("case_id")
        text_fields = ("query", "reference_answer", "final_visible_answer")
        if (
            not isinstance(case_id, str)
            or case_id not in expected_ids
            or case_id in seen
            or type(row.get("expected_semantic_correct")) is not bool
            or any(not isinstance(row.get(field), str) or not row[field] for field in text_fields)
        ):
            raise ScopedBlindReviewError("frozen blind-review calibration case is invalid")
        seen.add(case_id)
        normalized.append(dict(row))
    if seen != expected_ids:
        raise ScopedBlindReviewError("frozen blind-review calibration case set changed")
    return normalized, fixture_sha256


def _validate_scoped_index(
    index_path: Path,
    index: Mapping[str, Any],
    suite: Any,
) -> None:
    try:
        run_archive.validate_scoped_run_archive(
            index_path,
            index,
            suite,
            root=FINAL_ROOT,
        )
    except run_archive.ScopedRunArchiveError as exc:
        raise ScopedBlindReviewError(f"scoped blind review refused run archive: {exc}") from exc


def _source_registry(index: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    entries = index["difficulties"]
    baselines: dict[str, Any] = {}
    for difficulty in live.DIFFICULTIES:
        entry = entries[difficulty]
        baselines[difficulty] = {
            "role": SOURCE_CONTRACT_ROLE,
            "canonical": False,
            "promotion_allowed": False,
            "dataset": {
                "id": entry["dataset_id"],
                "key": entry["dataset_key"],
                "version": entry["dataset_version"],
                "checksum": entry["dataset_checksum"],
                "source_path": entry["dataset_source_path"],
                "source_sha256": entry["dataset_source_sha256"],
                "case_count": live.EXPECTED_CASE_COUNT,
            },
        }
    registry = {
        "schema_version": "stt.eval.easy-medium-blind-review-source-contract.v1",
        "status": "succeeded",
        "role": SOURCE_CONTRACT_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "scope": live.scope.scope_profile(),
        "baselines": baselines,
    }
    return registry, live._sha256_json(registry)


def _validate_plans(plans: Sequence[Any]) -> None:
    if [plan.difficulty for plan in plans] != list(live.DIFFICULTIES):
        raise ScopedBlindReviewError("blind-review plan must contain exactly easy then medium")
    case_ids: list[str] = []
    for plan in plans:
        if len(plan.cases) != live.EXPECTED_CASE_COUNT:
            raise ScopedBlindReviewError(f"{plan.difficulty} blind-review plan must contain 50 cases")
        for case in plan.cases:
            if case.answer_stage != "final" or case.phase != "evaluation":
                raise ScopedBlindReviewError("blind-review plan accepts only final evaluation answers")
            case_ids.append(case.case_id)
    if len(case_ids) != TARGET_CASE_COUNT or len(set(case_ids)) != TARGET_CASE_COUNT:
        raise ScopedBlindReviewError("blind-review plan must contain exactly 100 unique cases")


def _configure_reviewer() -> Any:
    reviewer = _load_reviewer()
    calibration_rows, calibration_sha256 = _frozen_calibration()
    reviewer_source_root = (
        PROJECT_ROOT if FINAL_ROOT.resolve() == SCRIPT_PATH.parent else FINAL_ROOT.resolve()
    )
    reviewer.PROJECT_ROOT = reviewer_source_root
    reviewer.REMEDIATION_ROOT = FINAL_ROOT
    reviewer.SUITE_ROOT = reviewer_source_root
    reviewer.RUNS_ROOT = FINAL_ROOT / "runs"
    reviewer.REVIEWS_ROOT = FINAL_ROOT / "semantic-reviews"
    reviewer.BASELINE_REGISTRY_PATH = FINAL_ROOT / "contracts" / "run-derived-source.json"
    reviewer.DIFFICULTIES = tuple(live.DIFFICULTIES)
    reviewer.ANSWER_STAGES = ("final",)
    reviewer.INDEX_SCHEMA = live.INDEX_SCHEMA
    reviewer.RECORD_SCHEMA = RECORD_SCHEMA
    reviewer.SUMMARY_SCHEMA = SUMMARY_SCHEMA
    reviewer.RECOVERY_SCHEMA = RECOVERY_SCHEMA
    reviewer.CALIBRATION_EXPECTED = {
        str(row["case_id"]): bool(row["expected_semantic_correct"])
        for row in calibration_rows
    }

    original_resolve_index = reviewer._resolve_index
    original_build_target_plan = reviewer._build_target_plan
    original_new_record = reviewer._new_record
    original_validate_existing_record = reviewer._validate_existing_record
    original_summary = reviewer._summary
    source_production: dict[tuple[str, str, str], tuple[float | None, bool | None]] = {}

    def resolve_index(label: str | None, run_index: str | None) -> tuple[Path, dict[str, Any]]:
        path, index = original_resolve_index(label, run_index)
        try:
            suite = live.load_suite_snapshot(FINAL_ROOT)
            _validate_scoped_index(path, index, suite)
        except (live.ScopedLiveError, live.full.FinalLiveError) as exc:
            raise reviewer.ReviewError(str(exc)) from exc
        except ScopedBlindReviewError as exc:
            raise reviewer.ReviewError(str(exc)) from exc
        return path, index

    def baseline_registry(index: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            return _source_registry(index)
        except (KeyError, TypeError) as exc:
            raise reviewer.ReviewError("scoped run dataset source contract is incomplete") from exc

    def build_target_plan(
        index_path: Path,
        index: Mapping[str, Any],
        registry: Mapping[str, Any],
        answer_stage: str,
    ) -> tuple[Any, ...]:
        if answer_stage != "final":
            raise reviewer.ReviewError("scoped blind review accepts final visible answers only")
        plans = original_build_target_plan(index_path, index, registry, answer_stage)
        isolated_plans: list[Any] = []
        for plan in plans:
            isolated_cases: list[Any] = []
            for case in plan.cases:
                source_production[(case.phase, case.difficulty, case.case_id)] = (
                    case.production_correctness,
                    case.production_passed,
                )
                isolated_cases.append(
                    reviewer.ReviewCase(
                        phase=case.phase,
                        difficulty=case.difficulty,
                        case_id=case.case_id,
                        answer_stage=case.answer_stage,
                        query=case.query,
                        reference_answer=case.reference_answer,
                        final_answer=case.final_answer,
                        production_correctness=None,
                        production_passed=None,
                        source_results_path=case.source_results_path,
                        source_results_sha256=case.source_results_sha256,
                        source_identity_sha256=case.source_identity_sha256,
                    )
                )
            isolated_plans.append(
                reviewer.DifficultyPlan(
                    difficulty=plan.difficulty,
                    dataset_path=plan.dataset_path,
                    dataset_sha256=plan.dataset_sha256,
                    results_path=plan.results_path,
                    results_sha256=plan.results_sha256,
                    run_id=plan.run_id,
                    cases=tuple(isolated_cases),
                )
            )
        plans = tuple(isolated_plans)
        try:
            _validate_plans(plans)
        except ScopedBlindReviewError as exc:
            raise reviewer.ReviewError(str(exc)) from exc
        return plans

    def build_calibration_cases(registry: Any, registry_sha256: str) -> tuple[Any, ...]:
        del registry, registry_sha256
        return tuple(
            reviewer.ReviewCase(
                phase="calibration",
                difficulty="easy",
                case_id=str(row["case_id"]),
                answer_stage="final",
                query=str(row["query"]),
                reference_answer=str(row["reference_answer"]),
                final_answer=str(row["final_visible_answer"]),
                production_correctness=None,
                production_passed=None,
                source_results_path=CALIBRATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
                source_results_sha256=calibration_sha256,
                source_identity_sha256=calibration_sha256,
            )
            for row in calibration_rows
        )

    def scoped_new_record(**kwargs: Any) -> dict[str, Any]:
        record = original_new_record(**kwargs)
        case = kwargs["case"]
        production = source_production.get((case.phase, case.difficulty, case.case_id))
        if production is not None:
            record["production"] = {
                "generation_correctness": production[0],
                "passed": production[1],
                "audit_only": True,
                "influences_sampling_or_judgment": False,
            }
        record.update(
            {
                "role": BLIND_REVIEW_ROLE,
                "scope_id": live.scope.SCOPE_ID,
                **NONPROMOTABLE_AUTHORITY,
                "counts_toward_primary_target": False,
            }
        )
        return record

    def validate_scoped_existing_record(record: Mapping[str, Any], **kwargs: Any) -> None:
        original_validate_existing_record(record, **kwargs)
        if set(record) != SCOPED_RECORD_ALLOWED_FIELDS:
            raise reviewer.ReviewError("saved scoped blind-review record schema changed")
        case = kwargs["case"]
        production = source_production.get((case.phase, case.difficulty, case.case_id))
        if production is not None and record.get("production") != {
            "generation_correctness": production[0],
            "passed": production[1],
            "audit_only": True,
            "influences_sampling_or_judgment": False,
        }:
            raise reviewer.ReviewError("saved production audit receipt changed")
        expected = {
            "role": BLIND_REVIEW_ROLE,
            "scope_id": live.scope.SCOPE_ID,
            **NONPROMOTABLE_AUTHORITY,
            "counts_toward_primary_target": False,
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise reviewer.ReviewError("saved scoped blind-review record changed audit permissions")

    def scoped_summary(**kwargs: Any) -> dict[str, Any]:
        value = original_summary(**kwargs)
        source_sha256 = str(value["input"].pop("baseline_registry_sha256"))
        value["input"].pop("baseline_registry_path", None)
        value["input"]["source_contract"] = {
            "schema_version": "stt.eval.easy-medium-blind-review-source-contract.v1",
            "role": SOURCE_CONTRACT_ROLE,
            "sha256": source_sha256,
            "derived_from_succeeded_run_index": True,
            "candidate_roots_consumed": False,
        }
        value.update(
            {
                "schema_version": SUMMARY_SCHEMA,
                "role": BLIND_REVIEW_ROLE,
                "scope": live.scope.scope_profile(),
                **NONPROMOTABLE_AUTHORITY,
                "counts_toward_primary_target": False,
                "primary_target_counts": dict(live.TARGET_CORRECT_COUNTS),
                "planned_target_case_count": TARGET_CASE_COUNT,
            }
        )
        value["judge"].update(
            {
                "adapter_path": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "adapter_sha256": _sha256_file(SCRIPT_PATH),
                "shared_reviewer_path": REVIEWER_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "shared_reviewer_sha256": _sha256_file(REVIEWER_PATH),
                "calibration_path": CALIBRATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "calibration_sha256": calibration_sha256,
                "provider_input_fields": [
                    "query",
                    "reference_answer",
                    "FINAL_VISIBLE_ANSWER",
                ],
            }
        )
        value["review_policy"].update(
            {
                "audit_only": True,
                "counts_toward_primary_target": False,
                "promotion_allowed": False,
                "source_production_values_audit_only": True,
                "production_correctness_influences_sampling_or_judgment": False,
                "production_correctness_conflict_second_review_enabled": False,
            }
        )
        return value

    reviewer._resolve_index = resolve_index
    reviewer._baseline_registry = baseline_registry
    reviewer._build_target_plan = build_target_plan
    reviewer._build_calibration_cases = build_calibration_cases
    reviewer._new_record = scoped_new_record
    reviewer._validate_existing_record = validate_scoped_existing_record
    reviewer._summary = scoped_summary
    return reviewer


def _adapter_contract(reviewer: Any) -> dict[str, Any]:
    sample = reviewer.ReviewCase(
        phase="evaluation",
        difficulty="easy",
        case_id="PAYLOAD-ISOLATION",
        answer_stage="final",
        query="问题",
        reference_answer="参考答案",
        final_answer="最终可见答案",
        production_correctness=None,
        production_passed=None,
        source_results_path="results.jsonl",
        source_results_sha256="a" * 64,
        source_identity_sha256="b" * 64,
    )
    messages = reviewer._messages(sample)
    user_payload = json.loads(messages[1]["content"])
    expected_fields = {"query", "reference_answer", "FINAL_VISIBLE_ANSWER"}
    if set(user_payload) != expected_fields:
        raise ScopedBlindReviewError("judge provider payload field set drifted")
    serialized = messages[1]["content"]
    forbidden = ("case_id", "production", "expected_terms", "retrieved", "citation_payload")
    if any(value in serialized for value in forbidden):
        raise ScopedBlindReviewError("judge provider payload contains a forbidden side channel")
    _, calibration_sha256 = _frozen_calibration()
    return {
        "scope": live.scope.scope_profile(),
        "role": BLIND_REVIEW_ROLE,
        "difficulties": list(live.DIFFICULTIES),
        "target_case_count": TARGET_CASE_COUNT,
        "answer_stage": "final",
        "provider_input_fields": sorted(expected_fields),
        "run_index_status_required": "succeeded",
        "archived_results_required": True,
        "candidate_roots_consumed": False,
        **NONPROMOTABLE_AUTHORITY,
        "counts_toward_primary_target": False,
        "source_production_values_audit_only": True,
        "production_correctness_influences_sampling_or_judgment": False,
        "production_correctness_conflict_second_review_enabled": False,
        "calibration_sha256": calibration_sha256,
        "shared_reviewer_sha256": _sha256_file(REVIEWER_PATH),
        "adapter_sha256": _sha256_file(SCRIPT_PATH),
    }


def _enrich_local_output(value: Mapping[str, Any], reviewer: Any) -> dict[str, Any]:
    return {
        **dict(value),
        "scoped_blind_review_contract": _adapter_contract(reviewer),
        "network_calls": int(value.get("network_calls") or 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    chosen_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        reviewer = _configure_reviewer()
    except ScopedBlindReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    original_argv = sys.argv
    sys.argv = [str(SCRIPT_PATH), *chosen_argv]
    local_only = "--execute" not in chosen_argv
    try:
        if local_only:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = int(reviewer.main())
            try:
                value = json.loads(output.getvalue())
            except json.JSONDecodeError as exc:
                raise ScopedBlindReviewError("frozen reviewer emitted invalid local JSON") from exc
            print(json.dumps(_enrich_local_output(value, reviewer), ensure_ascii=False, indent=2))
            return result
        return int(reviewer.main())
    except (reviewer.ReviewError, ScopedBlindReviewError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
