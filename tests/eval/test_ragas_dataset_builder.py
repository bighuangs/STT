from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from evals.ragas_support.dataset import (
    LEXICAL_REQUIREMENT_NOTE,
    OUTPUT_FILENAMES,
    PROTECTED_CASE_ID,
    SAMPLE_SCHEMA_VERSION,
    build_ragas_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREE_DIFFICULTY_ROOT = (
    PROJECT_ROOT / "docs" / "evaluation-baselines" / "2026-08-10-three-difficulty"
)
HAK180_V1_4 = PROJECT_ROOT / "evals" / "datasets" / "hak180_safety_v1_4.json"


def _case(case_id: str, *, answerable: bool = True) -> dict[str, object]:
    evidence = []
    chunk_ids = []
    if answerable:
        evidence = [
            {
                "evidence_id": f"evidence-{case_id}",
                "chunk_id": f"chunk-{case_id}",
                "page": 1,
                "text": f"{case_id} 的冻结证据正文。",
                "content_hash": "original-full-content-hash",
            }
        ]
        chunk_ids = [f"chunk-{case_id}"]
    return {
        "case_id": case_id,
        "query": f"{case_id} 的问题是什么？",
        "conversation": [],
        "reference_answer": "冻结参考答案",
        "unanswerable": not answerable,
        "expected_terms": ["参考答案"],
        "forbidden_terms": [],
        "relevant_document_ids": ["document-1"],
        "relevant_document_names": ["manual.pdf"],
        "relevant_chunk_ids": chunk_ids,
        "relevant_evidence": evidence,
        "tags": ["manual", "single-span"],
        "sensitivity": "internal",
        "metadata": {"type": "direct_fact", "critical": False},
    }


def _write_manifest(
    path: Path,
    *,
    dataset_key: str,
    difficulty: str | None,
    cases: list[dict[str, object]],
) -> None:
    manifest: dict[str, object] = {
        "dataset_key": dataset_key,
        "version": "1.0.0",
        "source": {
            "relative_path": "doc/manual.pdf",
            "content_sha256": "frozen-source-sha256",
        },
        "cases": cases,
    }
    if difficulty is not None:
        manifest["difficulty"] = difficulty
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def _write_protected_easy(path: Path) -> None:
    # 保护用例尾部故意不是合法 JSON；构建器若尝试反序列化或累计后再解析就会失败。
    path.write_text(
        """{
          "dataset_key": "fixture-easy",
          "version": "1.0.0",
          "difficulty": "easy",
          "source": {"relative_path": "doc/manual.pdf", "content_sha256": "source-v1"},
          "cases": [
            {"case_id": "P3500-EASY-001", "query": "普通问题？",
             "reference_answer": "普通答案", "unanswerable": false,
             "expected_terms": ["普通答案"], "relevant_document_ids": ["doc-1"],
             "relevant_chunk_ids": ["chunk-1"],
             "relevant_evidence": [{"chunk_id": "chunk-1", "text": "普通证据"}],
             "tags": ["manual"], "sensitivity": "internal", "metadata": {}},
            {"case_id": "P3500-EASY-002", "secret": "TOP-SECRET-MUST-NOT-LEAK",
             "deserialization_bomb": not-valid-json}
          ]
        }""",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_builder_normalizes_assets_and_never_decodes_protected_case(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    easy = inputs / "easy.json"
    medium = inputs / "medium.json"
    hard = inputs / "hard.json"
    hak180 = inputs / "hak180.json"
    _write_protected_easy(easy)
    _write_manifest(
        medium,
        dataset_key="fixture-medium",
        difficulty="medium",
        cases=[_case("P3500-MEDIUM-001")],
    )
    _write_manifest(
        hard,
        dataset_key="fixture-hard",
        difficulty="hard",
        cases=[_case("P3500-HARD-001", answerable=False)],
    )
    _write_manifest(
        hak180,
        dataset_key="hak180-safety",
        difficulty=None,
        cases=[_case("HAK180-001")],
    )

    output = tmp_path / "output"
    stats = build_ragas_dataset(
        [easy, medium, hard, hak180],
        output_dir=output,
    )

    assert set(path.name for path in output.iterdir()) == set(OUTPUT_FILENAMES)
    assert stats["totals"] == {
        "source_cases": 5,
        "historical_denominator": 5,
        "protected_placeholders": 1,
        "evaluation_eligible_samples": 4,
        "judge_eligible_samples": 4,
        "answerable_samples": 3,
        "unanswerable_samples": 1,
        "relevance_judgments": 3,
        "lexical_requirements": 4,
    }
    assert stats["protected_policy"] == {
        "case_ids": [PROTECTED_CASE_ID],
        "historical_denominator_retained": True,
        "content_parsed": False,
        "samples_and_judge_excluded": True,
    }

    cases = _read_csv(output / "cases.csv")
    protected = next(row for row in cases if row["case_id"] == PROTECTED_CASE_ID)
    assert protected["query"] == ""
    assert protected["gold_answer"] == ""
    assert protected["source_type"] == "protected_placeholder"
    assert json.loads(protected["metadata_json"])["content_parsed"] is False

    samples = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(sample["schema_version"] == SAMPLE_SCHEMA_VERSION for sample in samples)
    assert PROTECTED_CASE_ID not in {sample["case_id"] for sample in samples}
    medium_sample = next(sample for sample in samples if sample["difficulty"] == "medium")
    assert medium_sample["reference_context_ids"] == ["chunk-P3500-MEDIUM-001"]
    assert medium_sample["reference_contexts"] == [
        "P3500-MEDIUM-001 的冻结证据正文。"
    ]
    assert medium_sample["judge_eligible"] is True
    assert "response" not in medium_sample
    assert "retrieved_contexts" not in medium_sample

    qrels = _read_csv(output / "relevance-judgments.csv")
    assert len(qrels) == 3
    assert {row["case_id"] for row in qrels}.isdisjoint({PROTECTED_CASE_ID})
    assert all(len(row["evidence_text_hash"]) == 64 for row in qrels)

    claims = _read_csv(output / "claims.csv")
    assert len(claims) == 4
    assert all(row["verification_method"] == "lexical_requirement" for row in claims)
    assert all(row["notes"] == LEXICAL_REQUIREMENT_NOTE for row in claims)

    combined_output = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    assert "TOP-SECRET-MUST-NOT-LEAK" not in combined_output


def test_builder_checks_output_sha256_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "medium.json"
    _write_manifest(
        source,
        dataset_key="fixture-medium",
        difficulty="medium",
        cases=[_case("MEDIUM-001")],
    )
    output = tmp_path / "output"
    build_ragas_dataset([source], output_dir=output)

    checksum_rows = {}
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checksum_rows[filename] = digest
    assert set(checksum_rows) == {
        "cases.csv",
        "claims.csv",
        "dataset-stats.json",
        "relevance-judgments.csv",
        "samples.jsonl",
    }
    for filename, expected in checksum_rows.items():
        actual = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert actual == expected

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        build_ragas_dataset([source], output_dir=output)
    overwritten = build_ragas_dataset([source], output_dir=output, overwrite=True)
    assert overwritten["totals"]["evaluation_eligible_samples"] == 1
    assert overwritten["totals"]["judge_eligible_samples"] == 1


def test_builder_fails_closed_when_case_id_is_not_first_field(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.json"
    source.write_text(
        '{"dataset_key":"unsafe","cases":'
        '[{"query":"不得先解析","case_id":"P3500-EASY-002"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="首字段必须是 case_id"):
        build_ragas_dataset([source], output_dir=tmp_path / "output")


def test_builder_supports_the_frozen_three_difficulty_and_hak180_sources(
    tmp_path: Path,
) -> None:
    stats = build_ragas_dataset(
        [
            THREE_DIFFICULTY_ROOT / "datasets" / "easy.json",
            THREE_DIFFICULTY_ROOT / "datasets" / "medium.json",
            THREE_DIFFICULTY_ROOT / "datasets" / "hard.json",
            HAK180_V1_4,
        ],
        output_dir=tmp_path / "normalized",
    )

    assert stats["totals"]["historical_denominator"] == 178
    assert stats["totals"]["protected_placeholders"] == 1
    assert stats["totals"]["evaluation_eligible_samples"] == 177
    assert stats["totals"]["judge_eligible_samples"] == 176
    assert stats["totals"]["answerable_samples"] == 170
    assert stats["totals"]["unanswerable_samples"] == 7
    assert stats["totals"]["relevance_judgments"] == 215

    samples = [
        json.loads(line)
        for line in (tmp_path / "normalized" / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sample_ids = {sample["case_id"] for sample in samples}
    assert len(sample_ids) == 177
    assert PROTECTED_CASE_ID not in sample_ids
    restricted = next(sample for sample in samples if sample["case_id"] == "HAK180-028")
    assert restricted["sensitivity"] == "restricted"
    assert restricted["judge_eligible"] is False
