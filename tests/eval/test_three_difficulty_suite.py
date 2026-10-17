from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.domain.evaluation import content_digest, validate_dataset_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
)
DIFFICULTIES = ("easy", "medium", "hard")
SUITE_KEY = "three-difficulty-live-2026-08-10"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_three_difficulty_suite_is_frozen_and_has_fifty_cases_per_level() -> None:
    suite = _load_json(SUITE_ROOT / "suite.json")
    assert suite["suite_key"] == SUITE_KEY
    assert suite["evaluation_mode"] == "live"
    assert tuple(suite["datasets"]) == DIFFICULTIES

    all_case_ids: set[str] = set()
    all_queries: set[str] = set()
    source_hashes: set[str] = set()
    for difficulty in DIFFICULTIES:
        manifest = _load_json(SUITE_ROOT / "datasets" / f"{difficulty}.json")
        validate_dataset_manifest(manifest)
        assert manifest["suite_key"] == SUITE_KEY
        assert manifest["difficulty"] == difficulty
        assert manifest["minimum_case_count"] == 50
        assert manifest["evaluation_mode_default"] == "live"
        assert len(manifest["cases"]) == 50
        assert suite["datasets"][difficulty]["case_count"] == 50
        assert suite["datasets"][difficulty]["checksum"] == content_digest(manifest)

        source = manifest["source"]
        source_path = PROJECT_ROOT / source["relative_path"]
        assert source_path.is_file()
        assert _file_sha256(source_path) == source["content_sha256"]
        source_hashes.add(source["content_sha256"])

        for case in manifest["cases"]:
            assert case["case_id"] not in all_case_ids
            all_case_ids.add(case["case_id"])
            normalized_query = "".join(str(case["query"]).lower().split())
            assert normalized_query not in all_queries
            all_queries.add(normalized_query)
            assert case["metadata"]["difficulty"] == difficulty
            assert case["metadata"]["suite_key"] == SUITE_KEY
            assert f"difficulty-{difficulty}" in case["tags"]

    assert len(all_case_ids) == 150
    assert len(all_queries) == 150
    assert source_hashes == {suite["source_sha256"]}


def test_difficulty_composition_and_evidence_cardinality_are_stable() -> None:
    manifests = {
        difficulty: _load_json(SUITE_ROOT / "datasets" / f"{difficulty}.json")
        for difficulty in DIFFICULTIES
    }
    for case in manifests["easy"]["cases"]:
        assert not case["unanswerable"]
        assert len(case["relevant_chunk_ids"]) == 1
        assert len(case["relevant_evidence"]) == 1
    for case in manifests["medium"]["cases"]:
        assert not case["unanswerable"]
        assert len(case["relevant_chunk_ids"]) == 1
        assert len(case["relevant_evidence"]) == 1

    hard_answerable = [case for case in manifests["hard"]["cases"] if not case["unanswerable"]]
    hard_unanswerable = [case for case in manifests["hard"]["cases"] if case["unanswerable"]]
    assert len(hard_answerable) == 45
    assert len(hard_unanswerable) == 5
    assert all(len(case["relevant_chunk_ids"]) == 2 for case in hard_answerable)
    assert all(len(case["relevant_evidence"]) == 2 for case in hard_answerable)
    assert all(not case["relevant_chunk_ids"] for case in hard_unanswerable)
    assert all(not case["relevant_evidence"] for case in hard_unanswerable)


def test_quality_gate_is_bound_to_the_current_dataset_files() -> None:
    summary = _load_json(SUITE_ROOT / "quality" / "quality-summary.json")
    assert summary["suite_key"] == SUITE_KEY
    assert summary["deterministic_status"] == "passed"
    assert summary["online_audit_status"] in {"passed", "passed_after_adjudication"}
    assert summary["ready_for_import"] is True
    for difficulty in DIFFICULTIES:
        path = SUITE_ROOT / "datasets" / f"{difficulty}.json"
        assert summary["dataset_checksums"][difficulty] == _file_sha256(path)
