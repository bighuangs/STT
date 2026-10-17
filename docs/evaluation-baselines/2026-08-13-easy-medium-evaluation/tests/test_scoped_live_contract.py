from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest


SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))
import run_easy_medium_live as live  # noqa: E402
import activate_easy_medium_evaluation as activation_builder  # noqa: E402

scope = live.scope
_FIXTURE_RELEASE_VALIDATIONS: dict[Path, dict] = {}


def _fixture_current_reviewed_release_validation(root: Path) -> dict:
    return _FIXTURE_RELEASE_VALIDATIONS[root.resolve()]


@pytest.fixture(autouse=True)
def _use_fixture_reviewed_release_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live,
        "_current_reviewed_release_validation",
        _fixture_current_reviewed_release_validation,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(difficulty: str, *, source_sha256: str) -> dict:
    input_sha256 = ("1" if difficulty == "easy" else "2") * 64
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
        "name": f"Easy/medium {difficulty}",
        "version": live.EXPECTED_DATASET_VERSION,
        "difficulty": difficulty,
        "evaluation_mode_default": "live",
        "status": "candidate_active",
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
        "scope": scope.scope_profile(),
        "generation": {"independent_validation_status": "passed"},
        "gate_policy": {
            "case_pass_policy": live.EXPECTED_CASE_PASS_POLICY,
            "case_pass_policy_activation_status": "active_candidate",
            "case_pass_policy_materialization_required": False,
        },
        "activation_derivation": derivation,
        "source": {
            "relative_path": "source.pdf",
            "content_sha256": source_sha256,
            "page_count": 95,
            "kb_id": live.EXPECTED_KB_ID,
            "document_id": live.EXPECTED_DOCUMENT_ID,
            "document_version_id": live.EXPECTED_DOCUMENT_VERSION_ID,
            "index_version_id": live.EXPECTED_INDEX_VERSION_ID,
            "layout": "A4 portrait",
            "logical_name": "source.pdf",
            "original_logical_name": "source.pdf",
            "parent_dataset_key": f"fixture-{difficulty}-parent",
            "parent_dataset_version": "1.0.0",
            "parent_manifest_file_sha256": "f" * 64,
            "parent_suite_key": "fixture-parent-suite",
            "reviewed_from_published_evidence": True,
        },
        "cases": [
            {
                "case_id": f"{difficulty.upper()}-{number:03d}",
                "query": f"query {number}",
                "reference_answer": f"answer {number}",
                "metadata": {
                    "parent_case_id": f"{difficulty.upper()}-{number:03d}",
                    "difficulty": difficulty,
                },
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


def _suite(tmp_path: Path) -> dict:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"scoped source")
    source_sha256 = _sha(source_path)
    datasets: dict[str, dict] = {}
    reviewed_receipts: dict[str, dict] = {}
    for difficulty in scope.SCOPE_DIFFICULTIES:
        input_path = tmp_path / "reviewed" / "datasets" / f"{difficulty}.reviewed.json"
        active_template = _manifest(difficulty, source_sha256=source_sha256)
        input_manifest = {
            key: value
            for key, value in active_template.items()
            if key != "activation_derivation"
        }
        input_manifest.update(
            {
                "schema_version": "stt.eval.dataset.v2-llm-reviewed-candidate",
                "status": "llm_reviewed_candidate",
                "import_allowed": False,
            }
        )
        for case in input_manifest["cases"]:
            case["answer_contract"]["contract_status"] = "llm_reviewed_candidate"
            case["answer_contract"]["review"]["verdict"] = "approve"
        _write_json(input_path, input_manifest)
        path = tmp_path / "datasets" / f"{difficulty}.json"
        manifest = activation_builder._derive_active_manifest(
            input_manifest,
            input_relative_path=f"reviewed/datasets/{difficulty}.reviewed.json",
            input_file_sha256=_sha(input_path),
        )
        _write_json(path, manifest)
        derivation = manifest["activation_derivation"]
        reviewed_receipts[difficulty] = {
            "path": f"datasets/{difficulty}.reviewed.json",
            "sha256": _sha(input_path),
            "manifest_sha256": live._sha256_json(input_manifest),
            "case_count": 50,
            "dataset_key": live.EXPECTED_DATASET_KEYS[difficulty],
            "version": live.EXPECTED_DATASET_VERSION,
        }
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
    quality_targets = live._expected_quality_targets()
    _write_json(tmp_path / "quality-targets.json", quality_targets)
    reviewed_root = tmp_path / "reviewed"
    base_review = {
        "scope_id": scope.SCOPE_ID,
        "scoped_case_count": 100,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "full_source_joined_case_count_reference_only": 150,
        "block_union_audit_path": "reviewed/block-union-audit.json",
        "block_union_audit_sha256": "0" * 64,
        "original_scope_receipt_sha256": "1" * 64,
        "blocker_scope_receipt_sha256": "2" * 64,
        "contract_scope_receipt_sha256": "3" * 64,
    }
    reviewed_suite_path = reviewed_root / live.reviewed_validator.SUITE_FILENAME
    reviewed_summary_path = reviewed_root / live.reviewed_validator.SUMMARY_FILENAME
    reviewed_qa_path = reviewed_root / live.reviewed_validator.QA_FILENAME
    reviewed_checksums_path = reviewed_root / live.reviewed_validator.CHECKSUM_FILENAME
    _write_json(
        reviewed_suite_path,
        {"source_and_index_integrity": {"review": base_review}},
    )
    _write_json(reviewed_summary_path, {})
    _write_json(reviewed_qa_path, [])
    reviewed_checksums_path.write_text("fixture checksums\n", encoding="utf-8")
    release_validation = {
        "schema_version": "stt.eval.easy-medium-reviewed-validation.v1",
        "status": "passed",
        "suite_key": live.EXPECTED_SUITE_KEY,
        "suite": {
            "path": live.reviewed_validator.SUITE_FILENAME,
            "sha256": _sha(reviewed_suite_path),
        },
        "summary": {
            "path": live.reviewed_validator.SUMMARY_FILENAME,
            "sha256": _sha(reviewed_summary_path),
        },
        "qa": {
            "path": live.reviewed_validator.QA_FILENAME,
            "sha256": _sha(reviewed_qa_path),
        },
        "checksums": {
            "path": live.reviewed_validator.CHECKSUM_FILENAME,
            "sha256": _sha(reviewed_checksums_path),
        },
        "authoritative_scope_receipts_sha256": "9" * 64,
        "datasets": reviewed_receipts,
    }
    activation_review = {
        **base_review,
        "scoped_validation_schema": release_validation["schema_version"],
        "scoped_validation_sha256": live._sha256_json(release_validation),
        "scoped_reviewed_suite_path": live._relative_project(reviewed_suite_path),
        "scoped_reviewed_suite_sha256": release_validation["suite"]["sha256"],
        "scoped_reviewed_summary_path": live._relative_project(reviewed_summary_path),
        "scoped_reviewed_summary_sha256": release_validation["summary"]["sha256"],
        "scoped_reviewed_qa_path": live._relative_project(reviewed_qa_path),
        "scoped_reviewed_qa_sha256": release_validation["qa"]["sha256"],
        "scoped_reviewed_checksums_path": live._relative_project(reviewed_checksums_path),
        "scoped_reviewed_checksums_sha256": release_validation["checksums"]["sha256"],
        "authoritative_scope_receipts_sha256": release_validation[
            "authoritative_scope_receipts_sha256"
        ],
    }
    activation = {
        "schema_version": live.ACTIVATION_SCHEMA,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "scope": scope.scope_profile(),
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "promotable": False,
        "evaluation_contract": {
            "case_schema_version": live.CASE_CONTRACT,
            "authoritative_metric": live.AUTHORITATIVE_METRIC,
            "target_counts": {
                "easy": {"correct_count": 48, "case_count": 50},
                "medium": {"correct_count": 43, "case_count": 50},
            },
            "case_pass_policy": live.EXPECTED_CASE_PASS_POLICY,
        },
        "quality_targets": {
            "path": "quality-targets.json",
            "sha256": _sha(tmp_path / "quality-targets.json"),
            "schema_version": live.QUALITY_TARGETS_SCHEMA,
        },
        "source_and_index_integrity": {
            "source": {"path": "source.pdf", "sha256": source_sha256, "page_count": 95},
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
            "review": activation_review,
        },
        "datasets": datasets,
    }
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    _FIXTURE_RELEASE_VALIDATIONS[reviewed_root.resolve()] = release_validation
    return activation


def test_scope_activation_binds_exact_profile_and_100_cases(tmp_path: Path) -> None:
    _suite(tmp_path)
    snapshot = live.load_suite_snapshot(tmp_path)

    assert snapshot.suite_key == live.EXPECTED_SUITE_KEY
    assert tuple(snapshot.datasets) == scope.SCOPE_DIFFICULTIES
    assert sum(item.case_count for item in snapshot.datasets.values()) == 100
    assert snapshot.activation["scope"] == scope.scope_profile()


def test_scope_contract_has_one_authoritative_module_location() -> None:
    assert Path(scope.__file__).resolve() == (
        live.SCOPE_CONTRACT_ROOT / "easy_medium_scope.py"
    ).resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_registration_allowed", True),
        ("canonical_promotion_allowed", True),
        ("promotable", True),
    ],
)
def test_scope_activation_rejects_any_promotion_permission(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    activation = _suite(tmp_path)
    activation[field] = value
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match=field):
        live.load_suite_snapshot(tmp_path)


def test_scope_activation_rejects_hard_dataset_or_profile_drift(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    activation["datasets"]["hard"] = dict(activation["datasets"]["medium"])
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.ScopedLiveError, match="exactly easy and medium"):
        live.load_suite_snapshot(tmp_path)

    activation = _suite(tmp_path)
    activation["scope"] = {**scope.scope_profile(), "promotable": True}
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)
    with pytest.raises(live.ScopedLiveError, match="scope profile|authorization semantics"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_promotion_allowed", True),
        ("promotable", True),
        ("scope", {**scope.scope_profile(), "promotable": True}),
    ],
)
def test_dataset_manifest_rejects_promotion_or_scope_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    activation = _suite(tmp_path)
    dataset_path = tmp_path / "datasets" / "easy.json"
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(dataset_path, manifest)
    activation["datasets"]["easy"].update(
        {
            "sha256": _sha(dataset_path),
            "manifest_sha256": live._sha256_json(manifest),
        }
    )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="candidate-active|authorization semantics"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("promotion_allowed", True),
        ("canonical", True),
        ("candidate_roots", {"easy": "forbidden"}),
        ("source_baseline_registry", "forbidden"),
        ("baselines", ["forbidden"]),
    ),
)
def test_dataset_manifest_rejects_co_rehashed_unknown_promotion_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    activation = _suite(tmp_path)
    dataset_path = tmp_path / "datasets" / "easy.json"
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(dataset_path, manifest)
    activation["datasets"]["easy"].update(
        {
            "sha256": _sha(dataset_path),
            "manifest_sha256": live._sha256_json(manifest),
        }
    )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="top-level|authorization semantics"):
        live.load_suite_snapshot(tmp_path)


def test_dataset_manifest_rejects_co_rehashed_active_schema_drift(tmp_path: Path) -> None:
    activation = _suite(tmp_path)
    dataset_path = tmp_path / "datasets" / "easy.json"
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "evil.schema"
    _write_json(dataset_path, manifest)
    activation["datasets"]["easy"].update(
        {
            "sha256": _sha(dataset_path),
            "manifest_sha256": live._sha256_json(manifest),
        }
    )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="schema"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("concept_groups", "critical_constraints", "relevant_evidence", "review_verdict"),
)
def test_dataset_manifest_rejects_co_rehashed_active_derivation_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    activation = _suite(tmp_path)
    dataset_path = tmp_path / "datasets" / "easy.json"
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    if mutation == "concept_groups":
        case["concept_groups"]["all_of"][0]["any_of"].append("coordinated drift")
    elif mutation == "critical_constraints":
        case["critical_constraints"] = ["coordinated drift"]
    elif mutation == "relevant_evidence":
        case["relevant_evidence"] = [{"text": "coordinated drift"}]
    else:
        case["answer_contract"]["review"]["verdict"] = "coordinated_drift"
    _write_json(dataset_path, manifest)
    activation["datasets"]["easy"].update(
        {
            "sha256": _sha(dataset_path),
            "manifest_sha256": live._sha256_json(manifest),
        }
    )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="derivation"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("query", "concept_groups", "metadata_hard_lineage", "case_promotion"),
)
def test_dataset_manifest_rejects_reviewed_input_and_active_co_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    activation = _suite(tmp_path)
    input_path = tmp_path / "reviewed" / "datasets" / "easy.reviewed.json"
    reviewed = json.loads(input_path.read_text(encoding="utf-8"))
    case = reviewed["cases"][0]
    if mutation == "query":
        case["query"] = "coordinated changed query"
    elif mutation == "concept_groups":
        case["concept_groups"]["all_of"][0]["any_of"].append("coordinated drift")
    elif mutation == "metadata_hard_lineage":
        case["metadata"]["difficulty"] = "hard"
    else:
        case["promotion_allowed"] = True
    _write_json(input_path, reviewed)
    active = activation_builder._derive_active_manifest(
        reviewed,
        input_relative_path="reviewed/datasets/easy.reviewed.json",
        input_file_sha256=_sha(input_path),
    )
    dataset_path = tmp_path / "datasets" / "easy.json"
    _write_json(dataset_path, active)
    derivation = active["activation_derivation"]
    activation["datasets"]["easy"].update(
        {
            "sha256": _sha(dataset_path),
            "manifest_sha256": live._sha256_json(active),
            "input": {
                "path": derivation["input_path"],
                "file_sha256": derivation["input_file_sha256"],
                "manifest_sha256": derivation["input_manifest_sha256"],
            },
            "derivation_receipt_sha256": derivation["receipt_sha256"],
        }
    )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="current reviewed release|authorization"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("promotion_allowed", True),
        ("canonical", True),
        ("candidate_roots", {"easy": "forbidden"}),
        ("source_baseline_registry", "forbidden"),
        ("baselines", ["forbidden"]),
    ),
)
def test_activation_rejects_co_rehashed_unknown_promotion_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    activation = _suite(tmp_path)
    activation[field] = value
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="top-level|authorization semantics"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("layer", "field"),
    (
        ("evaluation_contract", "promotion_allowed"),
        ("evaluation_contract", "canonical"),
        ("dataset", "promotable"),
        ("dataset", "canonical_promotion_allowed"),
        ("source_and_index_integrity", "promotion_allowed"),
        ("quality_targets", "promotion_allowed"),
        ("quality_targets", "canonical"),
        ("quality_targets", "candidate_roots"),
    ),
)
def test_activation_rejects_nested_promotion_authority(
    tmp_path: Path,
    layer: str,
    field: str,
) -> None:
    activation = _suite(tmp_path)
    if layer == "dataset":
        activation["datasets"]["easy"][field] = True
    else:
        activation[layer][field] = (
            {"easy": "forbidden"} if field == "candidate_roots" else True
        )
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="authorization|contract|dataset|integrity"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    "field",
    (
        "scoped_validation_sha256",
        "scoped_reviewed_suite_sha256",
        "scoped_reviewed_summary_sha256",
        "scoped_reviewed_qa_sha256",
        "scoped_reviewed_checksums_sha256",
        "authoritative_scope_receipts_sha256",
    ),
)
def test_activation_rejects_stale_current_release_review_receipt(
    tmp_path: Path,
    field: str,
) -> None:
    activation = _suite(tmp_path)
    activation["source_and_index_integrity"]["review"][field] = "f" * 64
    _write_json(tmp_path / "activation.json", activation)
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="integrity schema"):
        live.load_suite_snapshot(tmp_path)


@pytest.mark.parametrize(
    "relative",
    (
        "candidate-roots.json",
        "baselines/canonical.json",
        "certification/promotion.json",
    ),
)
def test_loader_rejects_root_level_promotion_surface(
    tmp_path: Path,
    relative: str,
) -> None:
    _suite(tmp_path)
    _write_json(tmp_path / relative, {"promotion_allowed": True})
    live._write_static_checksums(tmp_path)

    with pytest.raises(live.ScopedLiveError, match="promotion surface"):
        live.load_suite_snapshot(tmp_path)


def test_cli_rejects_hard_before_any_live_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_easy_medium_live.py",
            "--label",
            "scope-01",
            "--difficulty",
            "hard",
            "--retrieval-profile-id",
            "rpf_default",
            "--retrieval-profile-version",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        live._parse_args({})


def test_scope_completion_is_non_promotable_and_never_creates_candidate_roots(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "runs" / "scope-01" / "run-index.json"
    index_path.parent.mkdir(parents=True)
    index = {
        "schema_version": live.INDEX_SCHEMA,
        "status": "in_progress",
        "role": live.RUN_ROLE,
        "canonical": False,
        "promotion_allowed": False,
        "scope": scope.scope_profile(),
        "difficulties": {
            "easy": {"status": "archived", "target_result": {"case_count": 50, "correct_count": 48}},
            "medium": {
                "status": "archived",
                "target_result": {"case_count": 50, "correct_count": 43},
            },
        },
    }

    live._finalize_scope_index(index_path.parent, index_path, index)

    assert index["status"] == "succeeded"
    assert index["promotion_allowed"] is False
    assert not (index_path.parent / "candidate-roots.json").exists()
    assert "candidate_roots" not in index
    assert "source_baseline_registry" not in index


def test_new_scope_label_cannot_start_with_medium(tmp_path: Path) -> None:
    _suite(tmp_path)
    snapshot = live.load_suite_snapshot(tmp_path)
    args = argparse.Namespace(
        label="scope-01",
        difficulty="medium",
        resume=False,
        base_url="http://127.0.0.1:8001",
        kb_id=live.EXPECTED_KB_ID,
        retrieval_profile_id="rpf_default",
        retrieval_profile_version=2,
        runtime_identity={},
    )

    with pytest.raises(live.ScopedLiveError, match="start with easy"):
        live._prepare_label(args, snapshot, "revision-test", root=tmp_path)


@pytest.mark.parametrize(
    "field",
    ("promotable", "baseline_registration_allowed", "canonical_promotion_allowed"),
)
def test_resume_rejects_in_progress_top_level_promotion_authority(
    tmp_path: Path,
    field: str,
) -> None:
    _suite(tmp_path)
    snapshot = live.load_suite_snapshot(tmp_path)
    runtime_identity = live.full._local_runtime_identity()
    args = argparse.Namespace(
        label="scope-01",
        difficulty="medium",
        resume=True,
        base_url="http://127.0.0.1:8001",
        kb_id=live.EXPECTED_KB_ID,
        retrieval_profile_id="rpf_default",
        retrieval_profile_version=2,
        runtime_identity=runtime_identity,
    )
    code_revision = live.full._current_code_revision()
    index = live._new_index(args, snapshot, code_revision, root=tmp_path)
    index[field] = True
    index_path = tmp_path / "runs" / "scope-01" / "run-index.json"
    _write_json(index_path, index)

    with pytest.raises(live.ScopedLiveError, match=field):
        live._prepare_label(args, snapshot, code_revision, root=tmp_path)
