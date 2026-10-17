from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
import promote_final_baselines as promote  # noqa: E402
import run_final_live as live  # noqa: E402
import self_check  # noqa: E402

ACTIVATION_DIR = (
    SUITE_DIR.parent
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)
sys.path.insert(0, str(ACTIVATION_DIR))
import activate_reviewed_suite as activate_suite  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(difficulty: str) -> dict:
    input_sha256 = ("1" if difficulty == "easy" else "2" if difficulty == "medium" else "3") * 64
    derivation = {
        "schema_version": "stt.eval.dataset-candidate-activation-derivation.v1",
        "input_path": f"reviewed/datasets/{difficulty}.reviewed.json",
        "input_file_sha256": input_sha256,
        "input_manifest_sha256": input_sha256,
        "output_state": "candidate_active",
        "rules_version": "stt.eval.reviewed-suite-activation.v1",
    }
    derivation["receipt_sha256"] = live._sha256_json(derivation)
    return {
        "schema_version": "stt.eval.dataset.v2-llm-reviewed-candidate-active",
        "suite_key": live.EXPECTED_SUITE_KEY,
        "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
        "name": f"Final {difficulty}",
        "version": live.EXPECTED_DATASET_VERSION,
        "difficulty": difficulty,
        "evaluation_mode_default": "live",
        "status": "candidate_active",
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "generation": {"independent_validation_status": "passed"},
        "gate_policy": {
            "case_pass_policy": live.EXPECTED_CASE_PASS_POLICY,
            "case_pass_policy_activation_status": "active_candidate",
            "case_pass_policy_materialization_required": False,
        },
        "activation_derivation": derivation,
        "source": {
            "kind": "test",
            "relative_path": "source.pdf",
            "content_sha256": "f" * 64,
            "page_count": 95,
            "kb_id": live.EXPECTED_KB_ID,
            "document_id": live.EXPECTED_DOCUMENT_ID,
            "document_version_id": live.EXPECTED_DOCUMENT_VERSION_ID,
            "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
        },
        "cases": [
            {
                "case_id": f"{difficulty.upper()}-{number:03d}",
                "query": f"query {number}",
                "reference_answer": f"answer {number}",
                "concept_groups": {
                    "all_of": [
                        {
                            "concept_key": f"concept-{number}",
                            "any_of": [f"answer {number}"],
                        }
                    ]
                },
                "answer_contract": {
                    "contract_status": "llm_reviewed_candidate_active",
                    "review": {"independent_validation_status": "passed"},
                },
            }
            for number in range(1, 51)
        ],
    }


def _quality_targets() -> dict:
    return {
        "schema_version": live.QUALITY_TARGETS_SCHEMA,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "primary_metric": {
            "name": "final_visible_answer_binary_accuracy",
            "production_path": "metrics.case_outcome.final_answer_accuracy",
            "description": "最终呈现给用户的答案按题二元判定正确或错误，再按难度汇总正确率。",
        },
        "independent_blind_review": {
            "role": "audit_only",
            "counts_toward_target": False,
            "description": "独立盲审仅用于审计生产评测结论，不替代或混入生产正确率。",
        },
        "targets": {
            "easy": {
                "required": True,
                "operator": ">",
                "threshold": 0.95,
                "case_count": 50,
                "minimum_correct_cases": 48,
            },
            "medium": {
                "required": True,
                "operator": ">",
                "threshold": 0.85,
                "case_count": 50,
                "minimum_correct_cases": 43,
            },
            "hard": {
                "required": False,
                "operator": None,
                "threshold": None,
                "case_count": 50,
                "minimum_correct_cases": None,
            },
        },
    }


def _suite(tmp_path: Path) -> dict:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"test source")
    source_sha256 = _sha(source_path)
    datasets: dict[str, dict] = {}
    for difficulty in live.DIFFICULTIES:
        input_path = tmp_path / "reviewed" / "datasets" / f"{difficulty}.reviewed.json"
        input_manifest = {"reviewed_input": difficulty}
        _write_json(input_path, input_manifest)
        path = tmp_path / "datasets" / f"{difficulty}.json"
        manifest = _manifest(difficulty)
        manifest["source"]["content_sha256"] = source_sha256
        manifest["activation_derivation"].update(
            {
                "input_file_sha256": _sha(input_path),
                "input_manifest_sha256": live._sha256_json(input_manifest),
            }
        )
        receipt = {
            key: value
            for key, value in manifest["activation_derivation"].items()
            if key != "receipt_sha256"
        }
        manifest["activation_derivation"]["receipt_sha256"] = live._sha256_json(receipt)
        _write_json(path, manifest)
        derivation = manifest["activation_derivation"]
        datasets[difficulty] = {
            "path": f"datasets/{difficulty}.json",
            "sha256": _sha(path),
            "manifest_sha256": live._sha256_json(manifest),
            "case_count": 50,
            "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
            "version": live.EXPECTED_DATASET_VERSION,
            "input": {
                "path": derivation["input_path"],
                "file_sha256": derivation["input_file_sha256"],
                "manifest_sha256": derivation["input_manifest_sha256"],
            },
            "derivation_receipt_sha256": derivation["receipt_sha256"],
        }
    source_receipt = {
        "input_checks": {
            "source_path": "source.pdf",
            "source_sha256": source_sha256,
            "source_hash_matches": True,
            "source_page_count_matches": True,
            "source_pdf_page_count": 95,
            "joined_case_count": 150,
            "indexed_evidence_count": 218,
            "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
        }
    }
    blocker_receipt = {
        "status": "passed",
        "selected_chunks_resolve_to_frozen_active_index": True,
        "active_index": {
            "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
            "frozen_chunk_count": 218,
            "index_rebuild_required": False,
            "evidence_refresh_required": False,
        },
    }
    _write_json(tmp_path / "receipts" / "source.json", source_receipt)
    _write_json(tmp_path / "receipts" / "blocker.json", blocker_receipt)
    quality_targets = _quality_targets()
    _write_json(tmp_path / "quality-targets.json", quality_targets)
    activation = {
        "schema_version": live.ACTIVATION_SCHEMA,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "evaluation_contract": {
            "case_schema_version": live.CASE_CONTRACT,
            "authoritative_metric": live.AUTHORITATIVE_METRIC,
            "target_counts": {
                "easy": {"correct_count": 48, "case_count": 50},
                "medium": {"correct_count": 43, "case_count": 50},
                "hard": {"correct_count": None, "case_count": 50},
            },
            "case_pass_policy": live.EXPECTED_CASE_PASS_POLICY,
        },
        "quality_targets": {
            "path": "quality-targets.json",
            "sha256": _sha(tmp_path / "quality-targets.json"),
            "schema_version": live.QUALITY_TARGETS_SCHEMA,
        },
        "source_and_index_integrity": {
            "source": {
                "path": "source.pdf",
                "sha256": source_sha256,
                "page_count": 95,
            },
            "active_index": {
                "kb_id": live.EXPECTED_KB_ID,
                "document_id": live.EXPECTED_DOCUMENT_ID,
                "document_version_id": live.EXPECTED_DOCUMENT_VERSION_ID,
                "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
                "frozen_chunk_count": 218,
                "source_qa_summary_path": "receipts/source.json",
                "source_qa_summary_sha256": _sha(tmp_path / "receipts" / "source.json"),
                "blocker_validation_path": "receipts/blocker.json",
                "blocker_validation_sha256": _sha(tmp_path / "receipts" / "blocker.json"),
            },
        },
        "datasets": datasets,
    }
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    return activation


def _result(case_id: str, correct: bool, *, run_id: str = "erun_test") -> dict:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "passed": not correct,
        "observation": {
            "answer": f"visible answer for {case_id}",
            "answer_envelope": {"answer": f"visible answer for {case_id}"},
        },
        "metrics": {
            "metrics": {
                "case_outcome": {
                    "final_answer_correct": correct,
                    "strict_supported_case_pass": not correct,
                }
            }
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact_entry(
    suite_root: Path,
    *,
    label: str,
    difficulty: str,
    correct_count: int,
) -> tuple[dict, Path]:
    run_id = f"erun_{difficulty}"
    artifact_dir = suite_root / "runs" / label / "artifacts" / difficulty / run_id
    rows = [
        _result(
            f"{difficulty.upper()}-{number:03d}",
            number <= correct_count,
            run_id=run_id,
        )
        for number in range(1, 51)
    ]
    payloads: dict[str, bytes] = {
        "report.json": b'{"status":"succeeded"}\n',
        "report.md": b"# succeeded\n",
        "report.html": b"<p>succeeded</p>\n",
        "summary.json": b'{"status":"succeeded"}\n',
        "manifest.json": b'{"status":"succeeded"}\n',
        "results.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode(
            "utf-8"
        ),
    }
    copies: dict[str, dict] = {}
    server_artifacts: dict[str, dict] = {}
    for artifact, filename in live.br.ARTIFACT_FILENAMES.items():
        path = artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        digest = _sha(path)
        size = path.stat().st_size
        server_path = f"output/evaluation/{run_id}/{filename}"
        server_artifacts[artifact] = {
            "path": server_path,
            "sha256": digest,
            "size_bytes": size,
        }
        copies[artifact] = {
            "filename": filename,
            "source": {
                "server_relative_path": server_path,
                "server_sha256": digest,
                "download_sha256": digest,
                "server_size_bytes": size,
                "download_size_bytes": size,
            },
            "target": {
                "relative_path": path.relative_to(suite_root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            },
            "source_target_sha256_match": True,
        }
    copy_manifest_path = artifact_dir / "artifact-copy-manifest.json"
    _write_json(
        copy_manifest_path,
        {
            "run_id": run_id,
            "difficulty": difficulty,
            "copies": copies,
        },
    )
    results_path = artifact_dir / "results.jsonl"
    target = {
        "correct_count": correct_count,
        "case_count": 50,
        "accuracy": round(correct_count / 50, 6),
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "strict_metrics_counted": False,
    }
    return {
        "difficulty": difficulty,
        "status": "archived",
        "run_id": run_id,
        "artifact_copy_manifest": {
            "path": copy_manifest_path.relative_to(suite_root).as_posix(),
            "sha256": _sha(copy_manifest_path),
        },
        "server_artifact_receipt": server_artifacts,
        "results_path": results_path.relative_to(suite_root).as_posix(),
        "results_sha256": _sha(results_path),
        "target_result": target,
    }, artifact_dir


def _preflight(spec: live.DatasetSpec) -> tuple[dict, SimpleNamespace]:
    config_hash = "a" * 64
    args = SimpleNamespace(
        kb_id=live.EXPECTED_KB_ID,
        retrieval_profile_id="rpf_test",
        retrieval_profile_version=7,
    )
    value = {
        "ready": True,
        "evaluation_mode": "live",
        "dataset": {
            "id": "eds_test",
            "key": spec.manifest["dataset_key"],
            "version": spec.manifest["version"],
            "checksum": "b" * 64,
        },
        "knowledge_base": {"id": args.kb_id, "status": "active"},
        "source_integrity": {
            "verified": True,
            "actual_sha256": spec.manifest["source"]["content_sha256"],
            "expected_sha256": spec.manifest["source"]["content_sha256"],
        },
        "matching_versions": [
            {
                "document_id": live.EXPECTED_DOCUMENT_ID,
                "active_version_id": live.EXPECTED_DOCUMENT_VERSION_ID,
                "active_index_version_id": live.EXPECTED_INDEX_VERSION_ID,
                "chunk_count": 218,
            }
        ],
        "active_indexes": [
            {
                "id": live.EXPECTED_INDEX_VERSION_ID,
                "kb_id": live.EXPECTED_KB_ID,
                "status": "active",
            }
        ],
        "vector_read_scopes": [
            {
                "kb_id": live.EXPECTED_KB_ID,
                "document_id": live.EXPECTED_DOCUMENT_ID,
                "document_version_id": live.EXPECTED_DOCUMENT_VERSION_ID,
                "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
            }
        ],
        "checks": {"dataset_ready": True, "knowledge_base_ready": True},
        "retrieval_profile_selection": {
            "profile_id": args.retrieval_profile_id,
            "version": args.retrieval_profile_version,
            "config_hash": config_hash,
            "source": "explicit",
        },
        "retrieval_profile": {
            "profile_id": args.retrieval_profile_id,
            "version": args.retrieval_profile_version,
            "config_hash": config_hash,
        },
        "vector_read_set_digest": "vector-digest-test",
    }
    return value, args


def test_activation_freezes_exact_reviewed_suite(tmp_path: Path) -> None:
    _suite(tmp_path)
    snapshot = live.load_suite_snapshot(tmp_path)
    assert snapshot.suite_key == live.EXPECTED_SUITE_KEY
    assert [snapshot.datasets[item].case_count for item in live.DIFFICULTIES] == [50, 50, 50]
    assert snapshot.datasets["medium"].manifest["version"] == live.EXPECTED_DATASET_VERSION


def test_activation_fails_closed_before_candidate_run(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    activation["candidate_run_allowed"] = False
    _write_json(tmp_path / "activation.json", activation)
    with pytest.raises(live.FinalLiveError, match="candidate_run_allowed"):
        live.load_suite_snapshot(tmp_path)


def test_activation_rejects_changed_dataset_hash(tmp_path: Path) -> None:
    _suite(tmp_path)
    path = tmp_path / "datasets" / "easy.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(live.FinalLiveError, match="activation SHA"):
        live.load_suite_snapshot(tmp_path)


def test_activation_requires_exact_frozen_v3_policy(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    activation["evaluation_contract"]["case_pass_policy"] = {
        **live.EXPECTED_CASE_PASS_POLICY,
        "answer_correctness_minimum": 0.74,
    }
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.FinalLiveError, match="case_pass_policy"):
        live.load_suite_snapshot(tmp_path)


def test_activation_requires_concept_groups_on_every_case(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    path = tmp_path / "datasets" / "easy.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["cases"][0].pop("concept_groups")
    _write_json(path, manifest)
    activation["datasets"]["easy"]["sha256"] = _sha(path)
    activation["datasets"]["easy"]["manifest_sha256"] = live._sha256_json(manifest)
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.FinalLiveError, match="concept_groups"):
        live.load_suite_snapshot(tmp_path)


def test_activation_requires_case_ids_unique_across_difficulties(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    path = tmp_path / "datasets" / "medium.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["cases"][0]["case_id"] = "EASY-001"
    _write_json(path, manifest)
    activation["datasets"]["medium"]["sha256"] = _sha(path)
    activation["datasets"]["medium"]["manifest_sha256"] = live._sha256_json(manifest)
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.FinalLiveError, match="globally unique"):
        live.load_suite_snapshot(tmp_path)


def test_activation_rejects_quality_target_hash_or_schema_drift(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    quality_path = tmp_path / "quality-targets.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["targets"]["easy"]["minimum_correct_cases"] = 47
    _write_json(quality_path, quality)
    activation["quality_targets"]["sha256"] = _sha(quality_path)
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.FinalLiveError, match="quality-target"):
        live.load_suite_snapshot(tmp_path)


def test_activation_rejects_incomplete_static_checksum_manifest(tmp_path: Path) -> None:
    _suite(tmp_path)
    checksum_path = tmp_path / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(live.FinalLiveError, match="SHA256SUMS"):
        live.load_suite_snapshot(tmp_path)


def test_activation_rejects_reparse_dataset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _suite(tmp_path)
    original = live._is_reparse_point
    monkeypatch.setattr(
        live,
        "_is_reparse_point",
        lambda path: Path(path).name == "easy.json" or original(Path(path)),
    )
    with pytest.raises(live.FinalLiveError, match="reparse point"):
        live.load_suite_snapshot(tmp_path)


def test_activation_derives_candidate_active_manifest_without_mutating_input() -> None:
    reviewed = _manifest("easy")
    reviewed["schema_version"] = "stt.eval.dataset.v2-llm-reviewed-candidate"
    reviewed["status"] = "reviewed_candidate_pending_independent_validation"
    reviewed["import_allowed"] = False
    reviewed.pop("candidate_run_allowed")
    reviewed["generation"]["independent_validation_status"] = "pending"
    reviewed["gate_policy"]["case_pass_policy_activation_status"] = "reserved_not_active"
    reviewed["gate_policy"]["case_pass_policy_materialization_required"] = True
    reviewed.pop("activation_derivation")
    for case in reviewed["cases"]:
        case["answer_contract"]["contract_status"] = "llm_reviewed_candidate"
        case["answer_contract"]["review"]["independent_validation_status"] = "pending"
    original = json.loads(json.dumps(reviewed))
    activated = activate_suite._derive_candidate_active_manifest(
        reviewed,
        difficulty="easy",
        input_relative_path="reviewed/datasets/easy.reviewed.json",
        input_file_sha256="a" * 64,
    )
    assert reviewed == original
    assert activated["status"] == "candidate_active"
    assert activated["import_allowed"] is True
    assert activated["candidate_run_allowed"] is True
    assert activated["baseline_registration_allowed"] is False
    assert activated["generation"]["independent_validation_status"] == "passed"
    assert activated["gate_policy"]["case_pass_policy_activation_status"] == "active_candidate"
    assert all(
        case["answer_contract"]["review"]["independent_validation_status"] == "passed"
        for case in activated["cases"]
    )
    receipt = activated["activation_derivation"]
    expected_receipt = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == activate_suite._sha256_json(expected_receipt)


def test_target_reads_only_final_answer_correct() -> None:
    case_ids = [f"CASE-{number:03d}" for number in range(50)]
    rows = [_result(case_id, position < 48) for position, case_id in enumerate(case_ids)]
    target = live._target_count_from_rows(
        rows,
        expected_run_id="erun_test",
        expected_case_ids=case_ids,
    )
    assert target["correct_count"] == 48
    assert target["accuracy"] == 0.96
    assert target["strict_metrics_counted"] is False


@pytest.mark.parametrize("bad_value", [None, 1, 0, "true"])
def test_target_fails_closed_for_missing_or_non_boolean_outcome(bad_value: object) -> None:
    row = _result("CASE-001", True)
    row["metrics"]["metrics"]["case_outcome"]["final_answer_correct"] = bad_value
    with pytest.raises(live.FinalLiveError, match="missing boolean"):
        live._target_count_from_rows(
            [row], expected_run_id="erun_test", expected_case_ids=["CASE-001"]
        )


def test_target_count_gates() -> None:
    easy_pass = {
        "status": "archived",
        "target_result": {"correct_count": 48, "case_count": 50},
    }
    medium_pass = {
        "status": "archived",
        "target_result": {"correct_count": 43, "case_count": 50},
    }
    assert live._entry_target_passed(easy_pass, "easy")
    assert live._entry_target_passed(medium_pass, "medium")
    assert live._entry_target_passed(
        {"target_result": {"correct_count": 0, "case_count": 50}}, "hard"
    )
    easy_fail = {
        "status": "archived",
        "target_result": {"correct_count": 47, "case_count": 50},
    }
    assert not live._entry_target_passed(easy_fail, "easy")


def test_profile_identity_cannot_change_mid_label() -> None:
    index: dict = {}
    frozen = {
        "profile_id": "rpf_default",
        "version": 2,
        "config_hash": "a" * 64,
        "snapshot_sha256": "b" * 64,
        "vector_read_set_digest": "vector-a",
    }
    live._freeze_or_validate_profile(index, frozen)
    live._freeze_or_validate_profile(index, dict(frozen))
    changed = {**frozen, "config_hash": "c" * 64}
    with pytest.raises(live.FinalLiveError, match="profile"):
        live._freeze_or_validate_profile(index, changed)


def test_server_dataset_checksum_cannot_change_on_resume(tmp_path: Path) -> None:
    _suite(tmp_path)
    spec = live.load_suite_snapshot(tmp_path).datasets["easy"]
    dataset = {
        "id": "eds_test",
        "dataset_key": live.EXPECTED_DATASET_KEYS["easy"],
        "version": live.EXPECTED_DATASET_VERSION,
        "case_count": 50,
        "checksum": "a" * 64,
    }
    with pytest.raises(live.FinalLiveError, match="checksum changed"):
        live._assert_dataset_api(
            dataset,
            spec,
            prior_id="eds_test",
            prior_checksum="b" * 64,
        )


def test_server_dataset_must_match_local_canonical_manifest(tmp_path: Path) -> None:
    _suite(tmp_path)
    spec = live.load_suite_snapshot(tmp_path).datasets["easy"]
    dataset = {
        "id": "eds_test",
        "dataset_key": live.EXPECTED_DATASET_KEYS["easy"],
        "version": live.EXPECTED_DATASET_VERSION,
        "case_count": 50,
        "checksum": "a" * 64,
        "manifest_json": spec.manifest,
    }
    with pytest.raises(live.FinalLiveError, match="canonical manifest checksum"):
        live._assert_dataset_api(dataset, spec)

    dataset["checksum"] = live._sha256_json(spec.manifest)
    dataset["manifest_json"] = {**spec.manifest, "name": "tampered"}
    with pytest.raises(live.FinalLiveError, match="manifest_json"):
        live._assert_dataset_api(dataset, spec)


def test_blind_calibration_fixture_is_candidate_independent() -> None:
    fixture = json.loads(
        (SUITE_DIR / "contracts" / "blind-review-calibration.json").read_text(encoding="utf-8")
    )
    rows = fixture["cases"]
    candidate_ids = {
        case["case_id"]
        for difficulty in live.DIFFICULTIES
        for case in _manifest(difficulty)["cases"]
    }
    assert rows
    assert all(row["case_id"] not in candidate_ids for row in rows)
    assert all(type(row["expected_semantic_correct"]) is bool for row in rows)


def test_minimal_handwritten_blind_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    calibration_path = tmp_path / "contracts" / "blind-review-calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_bytes(
        (SUITE_DIR / "contracts" / "blind-review-calibration.json").read_bytes()
    )
    monkeypatch.setattr(promote, "CALIBRATION_FIXTURE_PATH", calibration_path)
    _suite(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    label = "candidate-01"
    index_path = tmp_path / "runs" / label / "run-index.json"
    registry_path = tmp_path / "runs" / label / "candidate-roots.json"
    entries: dict[str, dict] = {}
    for difficulty in live.DIFFICULTIES:
        run_id = f"erun_{difficulty}"
        results_path = (
            tmp_path / "runs" / label / "artifacts" / difficulty / run_id / "results.jsonl"
        )
        rows = [
            _result(case_id, True, run_id=run_id) for case_id in suite.datasets[difficulty].case_ids
        ]
        _write_jsonl(results_path, rows)
        entries[difficulty] = {
            "run_id": run_id,
            "selected_case_ids": list(suite.datasets[difficulty].case_ids),
            "results_path": results_path.relative_to(tmp_path).as_posix(),
            "results_sha256": _sha(results_path),
        }
    index = {"difficulties": entries}
    _write_json(index_path, index)
    _write_json(registry_path, {"status": "succeeded"})
    records_path = tmp_path / "semantic-reviews" / label / "final" / "review.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "schema_version": "stt.eval.semantic-correctness-review-record.v1",
            "phase": "evaluation",
            "answer_stage": "final",
            "difficulty": difficulty,
            "case_id": case_id,
            "review_index": 1,
            "source": {
                "answer_stage": "final",
                "results_sha256": entries[difficulty]["results_sha256"],
            },
            "request": {
                "answer_stage": "final",
                "model": "qwen3.7-max",
                "temperature": 0,
                "prompt_version": promote.REVIEW_PROMPT_VERSION,
                "safe_base_url": "https://example.invalid/v1",
                "visible_answer_isolation": True,
                "citations_cannot_import_unstated_information": True,
            },
            "judgment": {"semantic_correct": True},
        }
        for difficulty in live.DIFFICULTIES
        for case_id in index["difficulties"][difficulty]["selected_case_ids"]
    ]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary_path = records_path.parent / "summary.json"
    summary = {
        "schema_version": "stt.eval.semantic-correctness-review-summary.v1",
        "status": "succeeded",
        "label": label,
        "answer_stage": "final",
        "calibration_passed": True,
        "judge": {
            "requested_model": "qwen3.7-max",
            "temperature": 0,
            "evaluation_answer_stage": "final",
            "visible_answer_only": True,
            "citation_content_supplied": False,
            "expected_terms_supplied": False,
            "production_metrics_supplied": False,
            "prompt_version": promote.REVIEW_PROMPT_VERSION,
            "safe_base_url": "https://example.invalid/v1",
        },
        "review_policy": {"binary_correct_rule": "semantic_correct=true only"},
        "credentials_recorded": False,
        "raw_provider_responses_recorded": False,
        "input": {
            "run_index_path": index_path.relative_to(tmp_path).as_posix(),
            "run_index_sha256": _sha(index_path),
            "baseline_registry_path": registry_path.relative_to(tmp_path).as_posix(),
            "baseline_registry_sha256": _sha(registry_path),
        },
        "overall": {"case_count": 150, "correct_count": 150, "incorrect_count": 0},
        "per_difficulty": {
            difficulty: {"case_count": 50, "correct_count": 50, "incorrect_count": 0}
            for difficulty in live.DIFFICULTIES
        },
        "record_count": 150,
        "records_path": records_path.relative_to(tmp_path).as_posix(),
        "records_sha256": _sha(records_path),
    }
    _write_json(summary_path, summary)
    authorization = {
        "final_blind_audit": {
            "approved": True,
            "summary_path": summary_path.relative_to(tmp_path).as_posix(),
            "summary_sha256": _sha(summary_path),
            "records_path": records_path.relative_to(tmp_path).as_posix(),
            "records_sha256": _sha(records_path),
        }
    }
    with pytest.raises(promote.PromotionError, match="identity/provenance|frozen calibration"):
        promote._validate_blind_audit(
            authorization,
            label=label,
            index_path=index_path,
            registry_path=registry_path,
            index=index,
            suite=suite,
        )


def test_arbitrary_hashed_evaluator_calibration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    evidence_path = tmp_path / "certification" / "calibration-details.json"
    _write_json(evidence_path, {"calibration": "complete"})
    approval_path = tmp_path / "certification" / "evaluator-calibration-approval.json"
    _write_json(
        approval_path,
        {
            "schema_version": promote.CALIBRATION_SCHEMA,
            "status": "approved",
            "approved": True,
            "suite_key": live.EXPECTED_SUITE_KEY,
            "case_contract": live.CASE_CONTRACT,
            "code_revision": "revision-a",
            "evidence": [
                {
                    "path": evidence_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(evidence_path),
                }
            ],
        },
    )
    labels = {
        f"{difficulty}:{difficulty.upper()}-{number:03d}": True
        for difficulty in live.DIFFICULTIES
        for number in range(1, 51)
    }
    index_path = tmp_path / "runs" / "candidate-01" / "run-index.json"
    index = {"code_revision": "revision-a", "label": "candidate-01"}
    _write_json(index_path, index)
    with pytest.raises(promote.PromotionError, match="split/metrics"):
        promote._validate_evaluator_calibration(
            {
                "evaluator_calibration": {
                    "approved": True,
                    "path": approval_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(approval_path),
                }
            },
            index,
            index_path=index_path,
            production_labels=labels,
            blind_labels=labels,
            blind_receipt={
                "summary_path": "blind/summary.json",
                "summary_sha256": "a" * 64,
                "records_path": "blind/records.jsonl",
                "records_sha256": "b" * 64,
            },
        )


def test_cached_target_cannot_bypass_predecessor_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _suite(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    label = "candidate-01"
    easy, _ = _artifact_entry(tmp_path, label=label, difficulty="easy", correct_count=47)
    easy["target_result"] = {
        **easy["target_result"],
        "correct_count": 48,
        "accuracy": 0.96,
    }
    monkeypatch.setattr(live, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(live, "_verify_archived_config_identity", lambda *args, **kwargs: None)
    with pytest.raises(live.FinalLiveError, match="archived target evidence changed"):
        live._assert_progression(
            {"difficulties": {"easy": easy}},
            "medium",
            suite=suite,
            label_root=tmp_path / "runs" / label,
        )


def test_all_six_artifacts_are_required(tmp_path: Path) -> None:
    label = "candidate-01"
    entry, artifact_dir = _artifact_entry(
        tmp_path, label=label, difficulty="easy", correct_count=48
    )
    (artifact_dir / "report.md").unlink()
    with pytest.raises(live.FinalLiveError, match="markdown target is missing"):
        live._verify_artifact_chain(
            suite_root=tmp_path,
            label_root=tmp_path / "runs" / label,
            entry=entry,
            difficulty="easy",
        )


def test_server_artifact_receipt_cannot_be_rewritten(tmp_path: Path) -> None:
    label = "candidate-01"
    entry, _ = _artifact_entry(tmp_path, label=label, difficulty="easy", correct_count=48)
    entry["server_artifact_receipt"]["markdown"]["sha256"] = "0" * 64
    with pytest.raises(live.FinalLiveError, match="markdown server receipt changed"):
        live._verify_artifact_chain(
            suite_root=tmp_path,
            label_root=tmp_path / "runs" / label,
            entry=entry,
            difficulty="easy",
        )


def test_execution_model_change_is_rejected_within_label() -> None:
    snapshot = {
        "config_version": "cfg-v1",
        "llm_model": "qwen3.7-max",
        "embedding_model": "bge-m3",
        "reranker_model": "bge-reranker-large",
        "code_revision": "revision-a",
        "evaluation_mode": "live",
        "gate_policy": {"enabled": True},
        "evaluation_security": {"mode": "live"},
        "vector_read_scopes": ["scope-a"],
        "telemetry_coverage": {"complete": True},
    }
    index: dict = {}
    live._freeze_or_validate_execution_config(index, snapshot)
    with pytest.raises(live.FinalLiveError, match="execution config changed"):
        live._freeze_or_validate_execution_config(
            index, {**snapshot, "llm_model": "different-model"}
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["checks"].__setitem__("dataset_ready", False), "checks"),
        (
            lambda value: value["retrieval_profile_selection"].__setitem__("source", "active"),
            "explicitly",
        ),
    ],
)
def test_preflight_fails_closed_for_failed_checks_or_implicit_profile(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    _suite(tmp_path)
    spec = live.load_suite_snapshot(tmp_path).datasets["easy"]
    preflight, args = _preflight(spec)
    mutation(preflight)
    with pytest.raises(live.FinalLiveError, match=message):
        live._validate_preflight(
            preflight,
            dataset_id="eds_test",
            dataset_checksum="b" * 64,
            spec=spec,
            args=args,
        )


def test_preflight_binds_manifest_kb_document_version_index_and_chunk_count(
    tmp_path: Path,
) -> None:
    _suite(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    spec = suite.datasets["easy"]
    preflight, args = _preflight(spec)
    identity = suite.source_index_identity
    preflight.update(
        {
            "source_integrity": {
                "verified": True,
                "actual_sha256": identity["source_sha256"],
                "expected_sha256": identity["source_sha256"],
            },
            "active_indexes": [
                {
                    "id": identity["index_version_id"],
                    "kb_id": identity["kb_id"],
                    "status": "active",
                }
            ],
            "matching_versions": [
                {
                    "document_id": identity["document_id"],
                    "active_version_id": identity["document_version_id"],
                    "active_index_version_id": identity["index_version_id"],
                    "chunk_count": 218,
                }
            ],
            "vector_read_scopes": [
                {
                    "kb_id": identity["kb_id"],
                    "document_id": identity["document_id"],
                    "document_version_id": identity["document_version_id"],
                    "index_version_id": identity["index_version_id"],
                }
            ],
        }
    )
    args.kb_id = identity["kb_id"]
    live._validate_preflight(
        preflight,
        dataset_id="eds_test",
        dataset_checksum="b" * 64,
        spec=spec,
        args=args,
        suite=suite,
    )
    preflight["matching_versions"][0]["chunk_count"] = 217
    with pytest.raises(live.FinalLiveError, match="source/index identity"):
        live._validate_preflight(
            preflight,
            dataset_id="eds_test",
            dataset_checksum="b" * 64,
            spec=spec,
            args=args,
            suite=suite,
        )


def test_cli_kb_id_must_match_all_canonical_manifests(tmp_path: Path) -> None:
    _suite(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    args = SimpleNamespace(kb_id="kb_wrong")
    with pytest.raises(live.FinalLiveError, match="--kb-id"):
        live._validate_cli_suite_identity(args, suite)


def test_self_check_defaults_to_real_suite_and_contract_only_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live, "FINAL_ROOT", tmp_path)
    assert self_check.main([]) == 2
    assert self_check.main(["--contract-only"]) == 0
    _suite(tmp_path)
    assert self_check.main([]) == 0


def test_generic_token_fields_are_redacted() -> None:
    cleaned = live.FinalRedactor().clean(
        {
            "token": "credential-token",  # secret-scan: allow-test
            "id_token": "credential-id",
            "session_token": "credential-session",
            "access_token": "credential-access",
        }
    )
    assert cleaned == {
        "token": "<redacted>",
        "id_token": "<redacted>",
        "session_token": "<redacted>",
        "access_token": "<redacted>",
    }


def test_label_lock_rejects_concurrent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live, "FINAL_ROOT", tmp_path)
    with live.LabelLock("candidate-01"):
        with pytest.raises(live.FinalLiveError, match="another runner owns label"):
            with live.LabelLock("candidate-01"):
                pass


def test_promotion_rejects_reparse_output_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    (tmp_path / "baselines").mkdir()
    monkeypatch.setattr(
        live,
        "_is_reparse_point",
        lambda path: Path(path).name == "baselines",
    )
    with pytest.raises(promote.PromotionError, match="reparse point"):
        promote._reject_reparse_chain(
            tmp_path / "baselines" / "baseline-index.json",
            label="canonical output",
        )


def test_evaluator_disagreement_without_adjudication_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(promote, "FINAL_ROOT", tmp_path)
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    labels = {
        f"{difficulty}:{difficulty.upper()}-{number:03d}": True
        for difficulty in live.DIFFICULTIES
        for number in range(1, 51)
    }
    blind = dict(labels)
    blind["easy:EASY-001"] = False
    split = promote._stable_split(list(labels))
    split_public = {key: value for key, value in split.items() if not key.endswith("_case_keys")}
    metrics = {
        "calibration": promote._agreement_metrics(split["calibration_case_keys"], labels, blind),
        "holdout": promote._agreement_metrics(split["holdout_case_keys"], labels, blind),
        "overall": promote._agreement_metrics(sorted(labels), labels, blind),
    }
    label = "candidate-01"
    index_path = tmp_path / "runs" / label / "run-index.json"
    index = {
        "code_revision": "revision-a",
        "label": label,
        "difficulties": {
            difficulty: {"run_id": f"erun_{difficulty}", "results_sha256": "a" * 64}
            for difficulty in live.DIFFICULTIES
        },
    }
    _write_json(index_path, index)
    blind_summary_path = tmp_path / "blind" / "summary.json"
    blind_records_path = tmp_path / "blind" / "records.jsonl"
    _write_json(blind_summary_path, {"status": "succeeded"})
    _write_jsonl(blind_records_path, [{"status": "succeeded"}])
    blind_receipt = {
        "summary_path": blind_summary_path.relative_to(tmp_path).as_posix(),
        "summary_sha256": _sha(blind_summary_path),
        "records_path": blind_records_path.relative_to(tmp_path).as_posix(),
        "records_sha256": _sha(blind_records_path),
    }
    approval_path = tmp_path / "certification" / "evaluator-calibration-approval.json"
    _write_json(
        approval_path,
        {
            "schema_version": promote.CALIBRATION_SCHEMA,
            "status": "approved",
            "approved": True,
            "suite_key": live.EXPECTED_SUITE_KEY,
            "case_contract": live.CASE_CONTRACT,
            "code_revision": "revision-a",
            "split": split_public,
            "metrics": metrics,
            "acceptance_policy": {
                "minimum_calibration_agreement_rate": 0.8,
                "minimum_holdout_agreement_rate": 0.8,
                "maximum_confirmed_evaluator_errors": 0,
                "all_disagreements_require_adjudication": True,
            },
            "source_receipts": {
                "run_index": {
                    "path": index_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(index_path),
                },
                "blind_summary": {
                    "path": blind_receipt["summary_path"],
                    "sha256": blind_receipt["summary_sha256"],
                },
                "blind_records": {
                    "path": blind_receipt["records_path"],
                    "sha256": blind_receipt["records_sha256"],
                },
            },
            "adjudications": [],
            "unresolved_disagreement_count": 0,
            "confirmed_evaluator_error_count": 0,
        },
    )
    with pytest.raises(promote.PromotionError, match="every evaluator/blind disagreement"):
        promote._validate_evaluator_calibration(
            {
                "evaluator_calibration": {
                    "approved": True,
                    "path": approval_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(approval_path),
                }
            },
            index,
            index_path=index_path,
            production_labels=labels,
            blind_labels=blind,
            blind_receipt=blind_receipt,
        )
