from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
import failure_analysis as failure  # noqa: E402
import promote_final_baselines as promote  # noqa: E402
import run_final_live as live  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result(run_id: str, case_id: str, *, correct: bool, strict: bool) -> dict:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "passed": strict,
        "observation": {"answer": f"visible answer for {case_id}"},
        "metrics": {
            "metrics": {
                "case_outcome": {
                    "final_answer_correct": correct,
                    "strict_supported_case_pass": strict,
                }
            }
        },
    }


def _run_evidence(root: Path, *, label: str = "candidate-01") -> tuple[Path, list[dict]]:
    entries: dict[str, dict] = {}
    annotations: list[dict] = []
    taxonomy = {
        "easy": ("retrieval_insufficient", "retrieval"),
        "medium": ("context_loss", "context_assembly"),
        "hard": ("intermediate_process_loss", "answer_postprocessing"),
    }
    for difficulty in live.DIFFICULTIES:
        run_id = f"erun_{difficulty}"
        case_ids = [f"{difficulty.upper()}-{number:03d}" for number in range(1, 51)]
        wrong_case_id = case_ids[-1]
        rows = [
            _result(
                run_id,
                case_id,
                correct=case_id != wrong_case_id,
                strict=case_id not in {wrong_case_id, case_ids[-2]},
            )
            for case_id in case_ids
        ]
        results_path = root / "runs" / label / "artifacts" / difficulty / run_id / "results.jsonl"
        _write_jsonl(results_path, rows)
        results_relative = results_path.relative_to(root).as_posix()
        entries[difficulty] = {
            "difficulty": difficulty,
            "status": "archived",
            "run_id": run_id,
            "selected_case_count": 50,
            "selected_case_ids": case_ids,
            "results_path": results_relative,
            "results_sha256": _sha(results_path),
            "target_result": live._target_count_from_rows(
                rows,
                expected_run_id=run_id,
                expected_case_ids=tuple(case_ids),
            ),
        }
        category, stage = taxonomy[difficulty]
        annotations.append(
            {
                "difficulty": difficulty,
                "run_id": run_id,
                "case_id": wrong_case_id,
                "primary_cause": category,
                "earliest_divergence": {
                    "stage": stage,
                    "description": "最早可验证偏差出现在该阶段，后续答案因此遗漏关键事实。",
                },
                "mechanism": "持久化结果中的候选证据与最终答案之间存在可复核的信息损失。",
                "evidence": [{"path": results_relative, "sha256": _sha(results_path)}],
            }
        )
    index_path = root / "runs" / label / "run-index.json"
    _write_json(
        index_path,
        {
            "schema_version": live.INDEX_SCHEMA,
            "status": "succeeded",
            "label": label,
            "suite_key": live.EXPECTED_SUITE_KEY,
            "difficulties": entries,
        },
    )
    return index_path, annotations


def _write_annotations(root: Path, label: str, rows: list[dict]) -> Path:
    path = root / "runs" / label / "failure-analysis-annotations.jsonl"
    _write_jsonl(path, rows)
    return path


def test_documented_taxonomy_and_authorization_receipt_match_runtime_contract() -> None:
    authorization = json.loads(
        (SUITE_DIR / "contracts" / "baseline-registration-authorization.example.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (SUITE_DIR / "contracts" / "failure-analysis-summary.example.json").read_text(
            encoding="utf-8"
        )
    )
    expected_taxonomy = [
        "retrieval_insufficient",
        "retrieval_noisy_or_unclear",
        "context_loss",
        "intermediate_process_loss",
        "evaluator_error",
        "other",
    ]
    assert list(failure.TAXONOMY) == expected_taxonomy
    assert summary["taxonomy"] == expected_taxonomy
    assert list(summary["primary_cause_counts"]) == expected_taxonomy
    assert list(authorization["failure_analysis"]["primary_cause_counts"]) == expected_taxonomy
    assert authorization["failure_analysis"]["evaluator_error_count"] == 0
    assert authorization["failure_analysis"]["promotion_allowed"] is True


def test_failure_artifacts_cover_exact_authoritative_wrong_set_and_recompute_summary(
    tmp_path: Path,
) -> None:
    label = "candidate-01"
    index_path, annotations = _run_evidence(tmp_path, label=label)
    annotations_path = _write_annotations(tmp_path, label, annotations)

    receipt = failure.write_failure_artifacts(tmp_path, label, annotations_path)
    verified = failure.validate_failure_artifacts(
        tmp_path,
        label,
        index_path=index_path,
    )

    assert receipt == verified
    assert verified["record_count"] == 3
    assert verified["evaluator_error_count"] == 0
    records_path = tmp_path / verified["records_path"]
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert {record["case_key"] for record in records} == {
        "easy:erun_easy:EASY-050",
        "medium:erun_medium:MEDIUM-050",
        "hard:erun_hard:HARD-050",
    }
    assert all(record["final_answer_correct"] is False for record in records)
    assert all(set(record["source_receipts"]) == {"run_index", "results"} for record in records)
    assert all(len(record["result_row_sha256"]) == 64 for record in records)

    summary = json.loads((tmp_path / verified["summary_path"]).read_text(encoding="utf-8"))
    assert summary["primary_cause_counts"] == {
        "retrieval_insufficient": 1,
        "retrieval_noisy_or_unclear": 0,
        "context_loss": 1,
        "intermediate_process_loss": 1,
        "evaluator_error": 0,
        "other": 0,
    }
    assert summary["divergence_stage_counts"]["retrieval"] == 1
    assert summary["high_frequency_patterns"][0]["count"] == 1
    assert all(
        pattern["case_keys_sha256"] == live._sha256_json(pattern["case_keys"])
        for pattern in summary["high_frequency_patterns"]
    )
    assert summary["promotion_allowed"] is True


def test_failure_annotations_reject_missing_extra_and_duplicate_cases(tmp_path: Path) -> None:
    label = "candidate-01"
    _, annotations = _run_evidence(tmp_path, label=label)

    missing_path = _write_annotations(tmp_path, label, annotations[:-1])
    with pytest.raises(failure.FailureAnalysisError, match="exactly cover"):
        failure.build_failure_artifacts(tmp_path, label, missing_path)

    duplicate_path = _write_annotations(tmp_path, label, [*annotations, annotations[0]])
    with pytest.raises(failure.FailureAnalysisError, match="duplicate"):
        failure.build_failure_artifacts(tmp_path, label, duplicate_path)

    extra = {
        **annotations[0],
        "case_id": "EASY-001",
    }
    extra_path = _write_annotations(tmp_path, label, [*annotations, extra])
    with pytest.raises(failure.FailureAnalysisError, match="exactly cover"):
        failure.build_failure_artifacts(tmp_path, label, extra_path)


def test_failure_validation_rejects_summary_and_source_drift(tmp_path: Path) -> None:
    label = "candidate-01"
    index_path, annotations = _run_evidence(tmp_path, label=label)
    annotations_path = _write_annotations(tmp_path, label, annotations)
    receipt = failure.write_failure_artifacts(tmp_path, label, annotations_path)

    summary_path = tmp_path / receipt["summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["primary_cause_counts"]["other"] = 1
    _write_json(summary_path, summary)
    with pytest.raises(failure.FailureAnalysisError, match="summary does not recompute"):
        failure.validate_failure_artifacts(tmp_path, label, index_path=index_path)

    _write_json(summary_path, failure.build_failure_artifacts(tmp_path, label, annotations_path)[1])
    results_path = tmp_path / "runs" / label / "artifacts" / "easy" / "erun_easy" / "results.jsonl"
    results_path.write_text(results_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(failure.FailureAnalysisError, match="results SHA"):
        failure.validate_failure_artifacts(tmp_path, label, index_path=index_path)


def test_promotion_requires_exact_failure_receipt_and_blocks_evaluator_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = "candidate-01"
    index_path, annotations = _run_evidence(tmp_path, label=label)
    annotations[0] = {**annotations[0], "primary_cause": "evaluator_error"}
    annotations_path = _write_annotations(tmp_path, label, annotations)
    receipt = failure.write_failure_artifacts(tmp_path, label, annotations_path)
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)

    authorization = {"failure_analysis": {"approved": True, **receipt}}
    with pytest.raises(promote.PromotionError, match="evaluator error"):
        promote._validate_failure_analysis(
            authorization,
            label=label,
            index_path=index_path,
        )

    clean_root = tmp_path / "clean"
    clean_index, clean_annotations = _run_evidence(clean_root, label=label)
    clean_annotations_path = _write_annotations(clean_root, label, clean_annotations)
    clean_receipt = failure.write_failure_artifacts(clean_root, label, clean_annotations_path)
    monkeypatch.setattr(promote, "FINAL_ROOT", clean_root)
    monkeypatch.setattr(promote, "PROJECT_ROOT", clean_root)
    changed_receipt = {"approved": True, **clean_receipt, "record_count": 4}
    with pytest.raises(promote.PromotionError, match="receipt differs"):
        promote._validate_failure_analysis(
            {"failure_analysis": changed_receipt},
            label=label,
            index_path=clean_index,
        )


@pytest.mark.parametrize("drift", ["duplicate_record", "summary", "results"])
def test_promotion_rejects_failure_artifact_universe_and_sha_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    label = "candidate-01"
    index_path, annotations = _run_evidence(tmp_path, label=label)
    annotations_path = _write_annotations(tmp_path, label, annotations)
    receipt = failure.write_failure_artifacts(tmp_path, label, annotations_path)
    authorization = {"failure_analysis": {"approved": True, **receipt}}
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)

    if drift == "duplicate_record":
        records_path = tmp_path / receipt["records_path"]
        first_row = records_path.read_text(encoding="utf-8").splitlines()[0]
        records_path.write_text(
            records_path.read_text(encoding="utf-8") + first_row + "\n",
            encoding="utf-8",
        )
    elif drift == "summary":
        summary_path = tmp_path / receipt["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["high_frequency_patterns"][0]["count"] += 1
        _write_json(summary_path, summary)
    else:
        results_path = (
            tmp_path / "runs" / label / "artifacts" / "medium" / "erun_medium" / "results.jsonl"
        )
        results_path.write_text(
            results_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

    with pytest.raises(promote.PromotionError, match="failure analysis is invalid"):
        promote._validate_failure_analysis(
            authorization,
            label=label,
            index_path=index_path,
        )
