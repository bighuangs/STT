from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import easy_medium_failure_analysis as failure  # noqa: E402
import run_easy_medium_live as live  # noqa: E402
from test_scoped_blind_review import _scoped_archive  # noqa: E402
from test_scoped_live_contract import (  # noqa: E402
    _fixture_current_reviewed_release_validation,
)

scope = live.scope


@pytest.fixture(autouse=True)
def _use_fixture_reviewed_release_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live,
        "_current_reviewed_release_validation",
        _fixture_current_reviewed_release_validation,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result(run_id: str, case_id: str, *, correct: bool) -> dict:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "metrics": {
            "metrics": {
                "case_outcome": {
                    "final_answer_correct": correct,
                    "strict_supported_case_pass": correct,
                }
            }
        },
    }


def _run_evidence(root: Path, label: str) -> tuple[Path, list[dict]]:
    index_path = _scoped_archive(root, label)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    annotations: list[dict] = []
    for difficulty in scope.SCOPE_DIFFICULTIES:
        entry = index["difficulties"][difficulty]
        run_id = entry["run_id"]
        case_ids = entry["selected_case_ids"]
        results_path = root / entry["results_path"]
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["passed"] = False
        rows[-1]["metrics"]["metrics"]["generation"]["correctness"] = 0.0
        rows[-1]["metrics"]["metrics"]["case_outcome"] = {
            "final_answer_correct": False,
            "strict_supported_case_pass": False,
        }
        _write_jsonl(results_path, rows)
        relative = results_path.relative_to(root).as_posix()
        results_sha256 = _sha(results_path)
        results_size = results_path.stat().st_size
        entry["results_sha256"] = results_sha256
        entry["server_artifact_receipt"]["results"].update(
            {"sha256": results_sha256, "size_bytes": results_size}
        )
        manifest_path = root / entry["artifact_copy_manifest"]["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_copy = manifest["copies"]["results"]
        result_copy["source"].update(
            {
                "server_sha256": results_sha256,
                "download_sha256": results_sha256,
                "server_size_bytes": results_size,
                "download_size_bytes": results_size,
            }
        )
        result_copy["target"].update(
            {"sha256": results_sha256, "size_bytes": results_size}
        )
        _write_json(manifest_path, manifest)
        entry["artifact_copy_manifest"]["sha256"] = _sha(manifest_path)
        entry["target_result"] = live._target_count_from_rows(
            rows,
            expected_run_id=run_id,
            expected_case_ids=case_ids,
        )
        entry["target_passed"] = True
        annotations.append(
            {
                "difficulty": difficulty,
                "run_id": run_id,
                "case_id": case_ids[-1],
                "primary_cause": "retrieval_insufficient",
                "earliest_divergence": {
                    "stage": "retrieval",
                    "description": "最早偏差发生在召回阶段，缺少回答所需的关键事实。",
                },
                "mechanism": "归档结果显示最终答案缺少冻结参考答案中的必要信息。",
                "evidence": [{"path": relative, "sha256": _sha(results_path)}],
            }
        )
    _write_json(index_path, index)
    return index_path, annotations


def test_diagnostic_failure_analysis_is_exact_and_non_promotable(tmp_path: Path) -> None:
    label = "scope-01"
    index_path, annotations = _run_evidence(tmp_path, label)
    annotations_path = tmp_path / "runs" / label / failure.ANNOTATIONS_FILENAME
    _write_jsonl(annotations_path, annotations)

    receipt = failure.write_diagnostic_artifacts(
        tmp_path,
        label,
        annotations_path,
        index_path=index_path,
    )
    verified = failure.validate_diagnostic_artifacts(tmp_path, label, index_path=index_path)

    assert receipt == verified
    assert receipt["record_count"] == 2
    assert receipt["promotion_allowed"] is False
    assert receipt["role"] == failure.DIAGNOSTIC_ROLE
    label_root = tmp_path / "runs" / label
    assert (label_root / failure.RECORDS_FILENAME).is_file()
    assert (label_root / failure.SUMMARY_FILENAME).is_file()
    assert not (label_root / "failure-analysis.jsonl").exists()
    assert not (label_root / "failure-summary.json").exists()
    summary = json.loads((label_root / failure.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["scope"] == scope.scope_profile()
    assert summary["promotion_allowed"] is False
    assert summary["canonical"] is False


def test_diagnostic_failure_analysis_rejects_hard_or_incomplete_scope(tmp_path: Path) -> None:
    label = "scope-01"
    index_path, annotations = _run_evidence(tmp_path, label)
    annotations_path = tmp_path / "runs" / label / failure.ANNOTATIONS_FILENAME
    _write_jsonl(
        annotations_path,
        [
            *annotations,
            {
                **annotations[0],
                "difficulty": "hard",
                "run_id": "erun_hard",
                "case_id": "HARD-001",
            },
        ],
    )
    with pytest.raises(failure.DiagnosticFailureError, match="exactly cover"):
        failure.build_diagnostic_artifacts(
            tmp_path,
            label,
            annotations_path,
            index_path=index_path,
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["difficulties"]["medium"]
    _write_json(index_path, index)
    _write_jsonl(annotations_path, annotations[:1])
    with pytest.raises(failure.DiagnosticFailureError, match="exactly easy and medium"):
        failure.build_diagnostic_artifacts(
            tmp_path,
            label,
            annotations_path,
            index_path=index_path,
        )


def test_diagnostic_failure_analysis_rejects_promotable_index(tmp_path: Path) -> None:
    label = "scope-01"
    index_path, annotations = _run_evidence(tmp_path, label)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["promotion_allowed"] = True
    _write_json(index_path, index)
    annotations_path = tmp_path / "runs" / label / failure.ANNOTATIONS_FILENAME
    _write_jsonl(annotations_path, annotations)

    with pytest.raises(failure.DiagnosticFailureError, match="identity"):
        failure.build_diagnostic_artifacts(
            tmp_path,
            label,
            annotations_path,
            index_path=index_path,
        )


@pytest.mark.parametrize(
    "field",
    (
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
    ),
)
def test_diagnostic_failure_analysis_rejects_top_level_promotion_authority(
    tmp_path: Path,
    field: str,
) -> None:
    label = "scope-01"
    index_path, annotations = _run_evidence(tmp_path, label)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[field] = True
    _write_json(index_path, index)
    annotations_path = tmp_path / "runs" / label / failure.ANNOTATIONS_FILENAME
    _write_jsonl(annotations_path, annotations)

    with pytest.raises(failure.DiagnosticFailureError, match="identity"):
        failure.build_diagnostic_artifacts(
            tmp_path,
            label,
            annotations_path,
            index_path=index_path,
        )


def test_diagnostic_failure_analysis_rejects_pseudo_100_case_universe(
    tmp_path: Path,
) -> None:
    label = "scope-01"
    index_path, annotations = _run_evidence(tmp_path, label)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for position, difficulty in enumerate(scope.SCOPE_DIFFICULTIES):
        entry = index["difficulties"][difficulty]
        fake_ids = [f"FAKE-{difficulty.upper()}-{number:03d}" for number in range(1, 51)]
        entry["selected_case_ids"] = fake_ids
        entry["config_snapshot"]["selected_case_ids"] = fake_ids
        results_path = tmp_path / entry["results_path"]
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        for row, fake_id in zip(rows, fake_ids, strict=True):
            row["case_id"] = fake_id
        _write_jsonl(results_path, rows)
        results_sha256 = _sha(results_path)
        results_size = results_path.stat().st_size
        entry["results_sha256"] = results_sha256
        entry["server_artifact_receipt"]["results"].update(
            {"sha256": results_sha256, "size_bytes": results_size}
        )
        manifest_path = tmp_path / entry["artifact_copy_manifest"]["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_copy = manifest["copies"]["results"]
        result_copy["source"].update(
            {
                "server_sha256": results_sha256,
                "download_sha256": results_sha256,
                "server_size_bytes": results_size,
                "download_size_bytes": results_size,
            }
        )
        result_copy["target"].update(
            {"sha256": results_sha256, "size_bytes": results_size}
        )
        _write_json(manifest_path, manifest)
        entry["artifact_copy_manifest"]["sha256"] = _sha(manifest_path)
        entry["target_result"] = live._target_count_from_rows(
            rows,
            expected_run_id=entry["run_id"],
            expected_case_ids=fake_ids,
        )
        annotations[position]["case_id"] = fake_ids[-1]
    _write_json(index_path, index)
    annotations_path = tmp_path / "runs" / label / failure.ANNOTATIONS_FILENAME
    _write_jsonl(annotations_path, annotations)

    with pytest.raises(failure.DiagnosticFailureError, match="case|suite|archive"):
        failure.build_diagnostic_artifacts(
            tmp_path,
            label,
            annotations_path,
            index_path=index_path,
        )
