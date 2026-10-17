#!/usr/bin/env python3
"""Bind the existing calibrated blind reviewer to a final candidate label.

The underlying reviewer, prompt, response validation, JSONL append/fsync, and
cost confirmations remain unchanged.  This adapter changes only the archive
roots and points its historical ``baseline registry`` input at the explicitly
non-canonical ``candidate-roots.json`` for the current label.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
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
REVIEWER_PATH = (
    FINAL_ROOT.parent
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "review_semantic_correctness.py"
)
INDEX_SCHEMA = "stt.eval.final-candidate-run-index.v1"
CALIBRATION_PATH = FINAL_ROOT / "contracts" / "blind-review-calibration.json"


def _load_reviewer() -> Any:
    if not REVIEWER_PATH.is_file():
        raise RuntimeError(f"existing semantic reviewer is missing: {REVIEWER_PATH}")
    spec = importlib.util.spec_from_file_location("stt_final_semantic_reviewer", REVIEWER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError("cannot load existing semantic reviewer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _label_from_argv(argv: list[str]) -> str:
    label: str | None = None
    run_index: Path | None = None
    for position, value in enumerate(argv):
        if value == "--label" and position + 1 < len(argv):
            label = argv[position + 1]
        elif value.startswith("--label="):
            label = value.split("=", maxsplit=1)[1]
        elif value == "--run-index" and position + 1 < len(argv):
            run_index = Path(argv[position + 1])
        elif value.startswith("--run-index="):
            run_index = Path(value.split("=", maxsplit=1)[1])
    if label is None and run_index is not None:
        path = run_index if run_index.is_absolute() else PROJECT_ROOT / run_index
        label = path.resolve().parent.name
    if label is None:
        # Let the underlying parser present its normal error for --self-test or
        # a missing source.  No archive-specific root is needed in that mode.
        return "self-test-no-archive"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", label):
        raise RuntimeError("invalid candidate label")
    return label


def main() -> int:
    label = _label_from_argv(sys.argv[1:])
    reviewer = _load_reviewer()
    reviewer.REMEDIATION_ROOT = FINAL_ROOT
    reviewer.SUITE_ROOT = FINAL_ROOT
    reviewer.RUNS_ROOT = FINAL_ROOT / "runs"
    reviewer.REVIEWS_ROOT = FINAL_ROOT / "semantic-reviews"
    reviewer.BASELINE_REGISTRY_PATH = FINAL_ROOT / "runs" / label / "candidate-roots.json"
    reviewer.INDEX_SCHEMA = INDEX_SCHEMA
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    rows = calibration.get("cases") if isinstance(calibration, dict) else None
    if (
        not isinstance(rows, list)
        or not rows
        or calibration.get("schema_version") != "stt.eval.semantic-judge-calibration.v1"
    ):
        raise RuntimeError("frozen blind-review calibration fixture is invalid")
    fixture_sha256 = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    expected: dict[str, bool] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("case_id"), str)
            or not row["case_id"]
            or row["case_id"] in expected
            or type(row.get("expected_semantic_correct")) is not bool
            or not isinstance(row.get("query"), str)
            or not isinstance(row.get("reference_answer"), str)
            or not isinstance(row.get("final_visible_answer"), str)
        ):
            raise RuntimeError("frozen blind-review calibration case is invalid")
        expected[row["case_id"]] = row["expected_semantic_correct"]
    reviewer.CALIBRATION_EXPECTED = expected

    def build_frozen_calibration_cases(registry: Any, registry_sha256: str) -> tuple[Any, ...]:
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
                production_correctness=(1.0 if row["expected_semantic_correct"] is True else 0.0),
                production_passed=bool(row["expected_semantic_correct"]),
                source_results_path=CALIBRATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
                source_results_sha256=fixture_sha256,
                source_identity_sha256=fixture_sha256,
            )
            for row in rows
        )

    reviewer._build_calibration_cases = build_frozen_calibration_cases
    try:
        return int(reviewer.main())
    except reviewer.ReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
