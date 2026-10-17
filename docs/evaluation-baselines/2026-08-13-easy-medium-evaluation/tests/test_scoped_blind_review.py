from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_easy_medium_semantic_correctness as blind  # noqa: E402
import run_easy_medium_live as live  # noqa: E402
import scoped_run_archive as run_archive  # noqa: E402
from test_scoped_live_contract import (  # noqa: E402
    _fixture_current_reviewed_release_validation,
    _sha,
    _suite,
    _write_json,
)


@pytest.fixture(autouse=True)
def _use_fixture_reviewed_release_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live,
        "_current_reviewed_release_validation",
        _fixture_current_reviewed_release_validation,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scoped_archive(root: Path, label: str = "scope-01") -> Path:
    _suite(root)
    suite = live.load_suite_snapshot(root)
    code_revision = live.full._current_code_revision()
    runtime_identity = live.full._local_runtime_identity()
    profile_snapshot = {
        "profile_id": "rpf_default",
        "version": 2,
        "config_hash": "c" * 64,
    }
    profile_identity = {
        **profile_snapshot,
        "snapshot_sha256": live._sha256_json(profile_snapshot),
        "vector_read_set_digest": "d" * 64,
    }
    entries: dict[str, dict] = {}
    config_snapshots: dict[str, dict] = {}
    for difficulty in live.DIFFICULTIES:
        spec = suite.datasets[difficulty]
        run_id = f"erun_{difficulty}"
        dataset_by_id = {row["case_id"]: row for row in spec.manifest["cases"]}
        rows = [
            {
                "run_id": run_id,
                "case_id": case_id,
                "passed": True,
                "observation": {
                    "query": dataset_by_id[case_id]["query"],
                    "answer_envelope": {"answer": f"answer {position}"},
                },
                "metrics": {
                    "query": dataset_by_id[case_id]["query"],
                    "metrics": {
                        "generation": {"correctness": 1.0},
                        "case_outcome": {
                            "final_answer_correct": True,
                            "strict_supported_case_pass": True,
                        },
                    },
                },
            }
            for position, case_id in enumerate(spec.case_ids, start=1)
        ]
        artifact_root = root / "runs" / label / "artifacts" / difficulty / run_id
        results_path = artifact_root / "results.jsonl"
        _write_jsonl(results_path, rows)
        artifacts: dict[str, Path] = {}
        for artifact, filename in live.br.ARTIFACT_FILENAMES.items():
            path = artifact_root / filename
            if artifact != "results":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"scoped {artifact}\n", encoding="utf-8")
            artifacts[artifact] = path
        server_receipt = {
            artifact: {
                "path": f"server/{difficulty}/{filename}",
                "sha256": _sha(artifacts[artifact]),
                "size_bytes": artifacts[artifact].stat().st_size,
            }
            for artifact, filename in live.br.ARTIFACT_FILENAMES.items()
        }
        copies = {
            artifact: {
                "filename": filename,
                "source": {
                    "server_relative_path": server_receipt[artifact]["path"],
                    "server_sha256": server_receipt[artifact]["sha256"],
                    "download_sha256": server_receipt[artifact]["sha256"],
                    "server_size_bytes": server_receipt[artifact]["size_bytes"],
                    "download_size_bytes": server_receipt[artifact]["size_bytes"],
                },
                "target": {
                    "relative_path": artifacts[artifact].relative_to(root).as_posix(),
                    "sha256": server_receipt[artifact]["sha256"],
                    "size_bytes": server_receipt[artifact]["size_bytes"],
                },
                "source_target_sha256_match": True,
            }
            for artifact, filename in live.br.ARTIFACT_FILENAMES.items()
        }
        manifest_path = artifact_root / "artifact-copy-manifest.json"
        _write_json(
            manifest_path,
            {
                "run_id": run_id,
                "difficulty": difficulty,
                "copies": copies,
            },
        )
        preflight = {
            "evaluation_mode": "live",
            "checks": {"dataset_ready": True, "knowledge_base_ready": True},
            "dataset": {
                "id": f"eds_{difficulty}",
                "key": live.EXPECTED_DATASET_KEYS[difficulty],
                "version": live.EXPECTED_DATASET_VERSION,
                "checksum": spec.manifest_sha256,
            },
        }
        config_snapshot = {
            **runtime_identity,
            "code_revision": code_revision,
            "dataset_version": live.EXPECTED_DATASET_VERSION,
            "dataset_checksum": spec.manifest_sha256,
            "evaluation_mode": "live",
            "selected_case_ids": list(spec.case_ids),
            "retrieval_profile": profile_snapshot,
            "vector_read_set_digest": profile_identity["vector_read_set_digest"],
            "preflight": preflight,
            "gate_policy": {},
            "evaluation_security": {},
            "vector_read_scopes": [],
            "telemetry_coverage": {},
        }
        config_snapshots[difficulty] = config_snapshot
        entries[difficulty] = {
            "difficulty": difficulty,
            "role": live.RUN_ROLE,
            "mode": "live",
            "baseline_run_id": None,
            "status": "archived",
            "run_id": run_id,
            "dataset_id": f"eds_{difficulty}",
            "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
            "dataset_version": live.EXPECTED_DATASET_VERSION,
            "dataset_checksum": spec.manifest_sha256,
            "dataset_source_path": spec.path.relative_to(root).as_posix(),
            "dataset_source_sha256": spec.file_sha256,
            "selected_case_count": live.EXPECTED_CASE_COUNT,
            "selected_case_ids": list(spec.case_ids),
            "completed_cases": live.EXPECTED_CASE_COUNT,
            "total_cases": live.EXPECTED_CASE_COUNT,
            "run_directory": artifact_root.relative_to(root).as_posix(),
            "artifact_copy_manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": _sha(manifest_path),
            },
            "server_artifact_receipt": server_receipt,
            "results_path": results_path.relative_to(root).as_posix(),
            "results_sha256": _sha(results_path),
            "target_result": live._target_count_from_rows(
                rows,
                expected_run_id=run_id,
                expected_case_ids=spec.case_ids,
            ),
            "target_required_correct_count": live._minimum_correct(difficulty),
            "target_passed": True,
            "config_snapshot": config_snapshot,
        }
    preflight_identity = {
        "invariant_sha256": live._sha256_json(
            live.full._preflight_invariant(config_snapshots["easy"]["preflight"])
        ),
        "evaluation_mode": "live",
        "checks": dict(config_snapshots["easy"]["preflight"]["checks"]),
    }
    execution_config_identity = live.full._execution_config_identity(config_snapshots["easy"])
    index_path = root / "runs" / label / "run-index.json"
    _write_json(
        index_path,
        {
            "schema_version": live.INDEX_SCHEMA,
            "status": "succeeded",
            "label": label,
            "role": live.RUN_ROLE,
            "canonical": False,
            "promotion_allowed": False,
            "suite_key": live.EXPECTED_SUITE_KEY,
            "suite_identity_sha256": suite.suite_identity_sha256,
            "scope": live.scope.scope_profile(),
            "mode": "live",
            "baseline_run_id": None,
            "selection": "all_scoped_cases",
            "planned_difficulties": list(live.DIFFICULTIES),
            "base_url": "http://127.0.0.1:8001",
            "kb_id": live.EXPECTED_KB_ID,
            "retrieval_profile_id": profile_identity["profile_id"],
            "retrieval_profile_version": profile_identity["version"],
            "code_revision": code_revision,
            "runtime_identity": runtime_identity,
            "runner": {
                "path": live._relative_project(live.SCRIPT_PATH),
                "sha256": live._sha256_file(
                    live.SCRIPT_PATH,
                    root=live.PROJECT_ROOT,
                    label="test scoped runner",
                ),
                "shared_validation_path": live._relative_project(live.full.SCRIPT_PATH),
                "shared_validation_sha256": live._sha256_file(
                    live.full.SCRIPT_PATH,
                    root=live.PROJECT_ROOT,
                    label="test shared validator",
                ),
                "shared_client_path": live._relative_project(live.SOURCE_RUNNER_PATH),
                "shared_client_sha256": live._sha256_file(
                    live.SOURCE_RUNNER_PATH,
                    root=live.PROJECT_ROOT,
                    label="test shared client",
                ),
            },
            "activation": {
                "path": suite.activation_path.relative_to(root).as_posix(),
                "sha256": suite.activation_sha256,
                "candidate_run_allowed": True,
                "baseline_registration_allowed": False,
                "canonical_promotion_allowed": False,
                "promotable": False,
            },
            "quality_targets": {
                "path": suite.quality_targets_path.relative_to(root).as_posix(),
                "sha256": suite.quality_targets_sha256,
                "schema_version": live.QUALITY_TARGETS_SCHEMA,
            },
            "target_contract": {
                "authoritative_metric": live.AUTHORITATIVE_METRIC,
                "strict_metrics_count_toward_target": False,
                "blind_audit_counts_toward_target": False,
                "correct_counts": live.TARGET_CORRECT_COUNTS,
                "case_count": live.EXPECTED_CASE_COUNT,
            },
            "scope_completion": {
                "schema_version": live.COMPLETION_SCHEMA,
                "status": "succeeded",
                "role": live.RUN_ROLE,
                "canonical": False,
                "promotion_allowed": False,
                "scope": live.scope.scope_profile(),
                "difficulty_order": list(live.DIFFICULTIES),
                "completed_at": "2026-08-13T12:00:00Z",
            },
            "source_index_identity": suite.source_index_identity,
            "static_checksums_sha256": suite.static_checksums_sha256,
            "retrieval_profile_identity": profile_identity,
            "preflight_identity": preflight_identity,
            "execution_config_identity": execution_config_identity,
            "dataset_files": {
                difficulty: {
                    "path": suite.datasets[difficulty].path.relative_to(root).as_posix(),
                    "sha256": suite.datasets[difficulty].file_sha256,
                    "manifest_sha256": suite.datasets[difficulty].manifest_sha256,
                    "case_count": live.EXPECTED_CASE_COUNT,
                    "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
                    "version": live.EXPECTED_DATASET_VERSION,
                }
                for difficulty in live.DIFFICULTIES
            },
            "difficulties": entries,
        },
    )
    return index_path


def test_scoped_blind_review_self_test_has_exact_isolated_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert blind.main(["--self-test"]) == 0
    value = json.loads(capsys.readouterr().out)
    contract = value["scoped_blind_review_contract"]

    assert value["network_calls"] == 0
    assert contract["provider_input_fields"] == [
        "FINAL_VISIBLE_ANSWER",
        "query",
        "reference_answer",
    ]
    assert contract["target_case_count"] == 100
    assert contract["candidate_roots_consumed"] is False
    assert contract["counts_toward_primary_target"] is False
    assert contract["promotion_allowed"] is False


def test_scoped_blind_review_dry_run_binds_exact_100_archived_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _scoped_archive(tmp_path)
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)

    assert blind.main(["--label", "scope-01"]) == 0
    value = json.loads(capsys.readouterr().out)
    contract = value["scoped_blind_review_contract"]

    assert value["status"] == "dry_run"
    assert value["network_calls"] == 0
    assert value["case_counts"] == {"easy": 50, "medium": 50}
    assert value["target_case_count"] == 100
    assert contract["run_index_status_required"] == "succeeded"
    assert contract["canonical"] is False


def test_scoped_blind_review_records_and_summary_are_audit_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = _scoped_archive(tmp_path)
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)
    reviewer = blind._configure_reviewer()
    _, index = reviewer._resolve_index("scope-01", None)
    registry, registry_sha256 = reviewer._baseline_registry(index)
    plans = reviewer._build_target_plan(index_path, index, registry, "final")
    assert all(
        case.production_correctness is None and case.production_passed is None
        for plan in plans
        for case in plan.cases
    )
    calibration = reviewer._build_calibration_cases(registry, registry_sha256)
    assert len(calibration) == 3

    fake_content = json.dumps(
        {
            "semantic_correct": True,
            "semantic_status": "correct",
            "semantic_score": 1,
            "confidence_score": 0.9,
            "answer_claims": ["answer"],
            "missing_or_incorrect_aspects": [],
            "reasoning_summary": "correct",
        }
    )
    monkeypatch.setattr(
        reviewer,
        "_provider_request",
        lambda **kwargs: (
            {
                "choices": [{"message": {"content": fake_content}}],
                "model": reviewer.JUDGE_MODEL,
                "id": "local-no-network",
            },
            "c" * 64,
            1,
        ),
    )
    case = plans[0].cases[0]
    record = reviewer._new_record(
        case=case,
        review_index=1,
        messages=reviewer._messages(case),
        trigger_reasons=[],
        endpoint="https://example.invalid/chat/completions",
        safe_base_url="https://example.invalid",
        api_key="not-used-by-local-stub",  # secret-scan: allow-test
        timeout=1,
        max_attempts=1,
        retry_delay=1,
        confidence_threshold=0.8,
        production_threshold=0.8,
    )
    assert record["canonical"] is False
    assert record["promotion_allowed"] is False
    assert record["counts_toward_primary_target"] is False
    assert record["production"]["audit_only"] is True
    assert record["production"]["influences_sampling_or_judgment"] is False

    summary = reviewer._summary(
        status="succeeded",
        index_path=index_path,
        index=index,
        registry_sha256=registry_sha256,
        plans=plans,
        all_records=[],
        calibration_finals={},
        evaluation_finals={},
        records_path=tmp_path / "semantic-reviews" / "records.jsonl",
        recovery_path=tmp_path / "semantic-reviews" / "recovery.json",
        recovery_events=[],
        safe_base_url="https://example.invalid",
        confidence_threshold=0.8,
        production_threshold=0.8,
        answer_stage="final",
    )
    assert summary["role"] == blind.BLIND_REVIEW_ROLE
    assert summary["canonical"] is False
    assert summary["promotion_allowed"] is False
    assert summary["promotable"] is False
    assert summary["baseline_registration_allowed"] is False
    assert summary["canonical_promotion_allowed"] is False
    assert summary["counts_toward_primary_target"] is False
    assert summary["planned_target_case_count"] == 100
    assert summary["input"]["source_contract"]["candidate_roots_consumed"] is False
    assert (
        summary["review_policy"]["production_correctness_conflict_second_review_enabled"]
        is False
    )
    assert (
        summary["review_policy"]["production_correctness_influences_sampling_or_judgment"]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    ["promotable", "hard", "short", "candidate_roots"],
)
def test_scoped_blind_review_rejects_scope_or_promotion_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    index_path = _scoped_archive(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if mutation == "promotable":
        index["promotion_allowed"] = True
    elif mutation == "hard":
        index["planned_difficulties"].append("hard")
        index["difficulties"]["hard"] = dict(index["difficulties"]["medium"])
    elif mutation == "short":
        index["difficulties"]["medium"]["selected_case_ids"].pop()
        index["difficulties"]["medium"]["selected_case_count"] = 49
    else:
        _write_json(index_path.parent / "candidate-roots.json", {"status": "succeeded"})
    _write_json(index_path, index)
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)

    assert blind.main(["--label", "scope-01"]) == 2
    assert "refused" in capsys.readouterr().err


def test_scoped_blind_review_rejects_results_receipt_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_path = _scoped_archive(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    results_path = tmp_path / index["difficulties"]["easy"]["results_path"]
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)

    assert blind.main(["--label", "scope-01"]) == 2
    assert "archive validation" in capsys.readouterr().err


def test_scoped_blind_review_rejects_execution_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index_path = _scoped_archive(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["code_revision"] = "tampered-code-revision"
    _write_json(index_path, index)
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)

    assert blind.main(["--label", "scope-01"]) == 2
    assert "code" in capsys.readouterr().err


def test_shared_archive_validator_rejects_every_execution_identity_layer(
    tmp_path: Path,
) -> None:
    index_path = _scoped_archive(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    original = json.loads(index_path.read_text(encoding="utf-8"))
    mutations = {
        "code": lambda value: value.__setitem__("code_revision", "tampered"),
        "runner": lambda value: value["runner"].__setitem__("sha256", "0" * 64),
        "runtime": lambda value: value["runtime_identity"].__setitem__(
            "llm_model", "tampered"
        ),
        "retrieval": lambda value: value["retrieval_profile_identity"].__setitem__(
            "config_hash", "0" * 64
        ),
        "preflight": lambda value: value["preflight_identity"].__setitem__(
            "invariant_sha256", "0" * 64
        ),
        "config": lambda value: value["execution_config_identity"].__setitem__(
            "snapshot_invariant_sha256", "0" * 64
        ),
        "static": lambda value: value.__setitem__("static_checksums_sha256", "0" * 64),
        "activation": lambda value: value["activation"].__setitem__("sha256", "0" * 64),
        "completion": lambda value: value["scope_completion"].__setitem__(
            "promotion_allowed", True
        ),
    }
    for mutation in mutations.values():
        changed = deepcopy(original)
        mutation(changed)
        _write_json(index_path, changed)
        with pytest.raises(run_archive.ScopedRunArchiveError):
            run_archive.validate_scoped_run_archive(
                index_path,
                changed,
                suite,
                root=tmp_path,
            )
    _write_json(index_path, original)


def test_shared_archive_validator_requires_six_artifacts_and_server_receipt(
    tmp_path: Path,
) -> None:
    index_path = _scoped_archive(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    original_index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / original_index["difficulties"]["easy"][
        "artifact_copy_manifest"
    ]["path"]
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed_manifest = deepcopy(original_manifest)
    changed_manifest["copies"].pop("json")
    _write_json(manifest_path, changed_manifest)
    missing_artifact_index = deepcopy(original_index)
    missing_artifact_index["difficulties"]["easy"]["artifact_copy_manifest"]["sha256"] = _sha(
        manifest_path
    )
    _write_json(index_path, missing_artifact_index)
    with pytest.raises(run_archive.ScopedRunArchiveError, match="six|all six"):
        run_archive.validate_scoped_run_archive(
            index_path,
            missing_artifact_index,
            suite,
            root=tmp_path,
        )
    _write_json(manifest_path, original_manifest)
    changed_receipt_index = deepcopy(original_index)
    changed_receipt_index["difficulties"]["easy"]["server_artifact_receipt"]["results"][
        "sha256"
    ] = "0" * 64
    _write_json(index_path, changed_receipt_index)
    with pytest.raises(run_archive.ScopedRunArchiveError, match="server receipt"):
        run_archive.validate_scoped_run_archive(
            index_path,
            changed_receipt_index,
            suite,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    (
        "promotable",
        "baseline_registration_allowed",
        "canonical_promotion_allowed",
    ),
)
def test_shared_archive_validator_rejects_top_level_promotion_authority(
    tmp_path: Path,
    field: str,
) -> None:
    index_path = _scoped_archive(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[field] = True
    _write_json(index_path, index)

    with pytest.raises(run_archive.ScopedRunArchiveError, match="identity"):
        run_archive.validate_scoped_run_archive(
            index_path,
            index,
            suite,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    (
        "promotable",
        "promotion_allowed",
        "canonical",
        "canonical_promotion_allowed",
        "baseline_registration_allowed",
    ),
)
def test_shared_archive_validator_rejects_nested_promotion_authority(
    tmp_path: Path,
    field: str,
) -> None:
    index_path = _scoped_archive(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["difficulties"]["easy"][field] = True
    _write_json(index_path, index)

    with pytest.raises(run_archive.ScopedRunArchiveError, match="difficulties.easy"):
        run_archive.validate_scoped_run_archive(
            index_path,
            index,
            suite,
            root=tmp_path,
        )


@pytest.mark.parametrize("receipt", ("artifact_copy_manifest", "server_artifact_receipt"))
def test_shared_archive_validator_rejects_artifact_receipt_schema_drift(
    tmp_path: Path,
    receipt: str,
) -> None:
    index_path = _scoped_archive(tmp_path)
    suite = live.load_suite_snapshot(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index["difficulties"]["easy"]
    if receipt == "artifact_copy_manifest":
        entry[receipt]["promotion_allowed"] = True
    else:
        entry[receipt]["results"]["canonical"] = True
    _write_json(index_path, index)

    with pytest.raises(run_archive.ScopedRunArchiveError, match="artifact|receipt"):
        run_archive.validate_scoped_run_archive(
            index_path,
            index,
            suite,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("promotable", True),
        ("canonical_promotion_allowed", True),
        ("baseline_registration_allowed", True),
        ("candidate_roots", {"easy": "forbidden"}),
        ("source_baseline_registry", "forbidden"),
        ("baselines", ["forbidden"]),
    ),
)
def test_scoped_existing_record_rejects_co_rehashed_promotion_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    index_path = _scoped_archive(tmp_path)
    monkeypatch.setattr(blind, "FINAL_ROOT", tmp_path)
    reviewer = blind._configure_reviewer()
    _, index = reviewer._resolve_index("scope-01", None)
    registry, registry_sha256 = reviewer._baseline_registry(index)
    plans = reviewer._build_target_plan(index_path, index, registry, "final")
    case = plans[0].cases[0]
    messages = reviewer._messages(case)
    payload_sha256 = "c" * 64
    fake_content = json.dumps(
        {
            "semantic_correct": True,
            "semantic_status": "correct",
            "semantic_score": 1,
            "confidence_score": 0.9,
            "answer_claims": ["answer"],
            "missing_or_incorrect_aspects": [],
            "reasoning_summary": "correct",
        }
    )
    monkeypatch.setattr(
        reviewer,
        "_provider_request",
        lambda **kwargs: (
            {
                "choices": [{"message": {"content": fake_content}}],
                "model": reviewer.JUDGE_MODEL,
                "id": "local-no-network",
            },
            payload_sha256,
            1,
        ),
    )
    record = reviewer._new_record(
        case=case,
        review_index=1,
        messages=messages,
        trigger_reasons=[],
        endpoint="https://example.invalid/chat/completions",
        safe_base_url="https://example.invalid",
        api_key="not-used-by-local-stub",  # secret-scan: allow-test
        timeout=1,
        max_attempts=1,
        retry_delay=1,
        confidence_threshold=0.8,
        production_threshold=0.8,
    )
    record[field] = value
    record["record_sha256"] = reviewer._record_hash(record)
    records_path = tmp_path / "semantic-reviews" / "records.jsonl"
    _write_jsonl(records_path, [record])
    loaded, _ = reviewer._load_records(records_path)
    assert loaded[0]["record_sha256"] == record["record_sha256"]

    with pytest.raises(reviewer.ReviewError, match="schema|permissions"):
        reviewer._validate_existing_record(
            record,
            case=case,
            review_index=1,
            prompt_sha256=reviewer._sha256_json(list(messages)),
            payload_sha256=payload_sha256,
            trigger_reasons=[],
            safe_base_url="https://example.invalid",
            confidence_threshold=0.8,
            production_threshold=0.8,
        )

def test_scoped_blind_review_second_review_ignores_production_conflict() -> None:
    reviewer = blind._configure_reviewer()
    case = reviewer.ReviewCase(
        phase="evaluation",
        difficulty="easy",
        case_id="EASY-PRODUCTION-CONFLICT",
        answer_stage="final",
        query="问题",
        reference_answer="参考答案",
        final_answer="最终答案",
        production_correctness=None,
        production_passed=None,
        source_results_path="results.jsonl",
        source_results_sha256="a" * 64,
        source_identity_sha256="b" * 64,
    )

    assert reviewer._second_review_reasons(
        case,
        {"semantic_correct": True, "confidence_score": 0.99},
        confidence_threshold=0.8,
        production_threshold=0.8,
    ) == []
