from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)
if str(CONTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRACT_ROOT))

scope = importlib.import_module("easy_medium_scope")
merge = importlib.import_module("merge_easy_medium_reviewed")
validate = importlib.import_module("validate_easy_medium_reviewed")
activate = importlib.import_module("activate_easy_medium_evaluation")
full_activate = importlib.import_module("activate_reviewed_suite")
SCOPED_FINAL_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-13-easy-medium-evaluation"
)
if str(SCOPED_FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SCOPED_FINAL_ROOT))
live = importlib.import_module("run_easy_medium_live")


_FIXTURE_CASE_REVIEW_AUTHORITY: dict[str, dict] = {}
_FIXTURE_SOURCE_AUTHORITY: dict[str, dict] = {}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums(root: Path) -> None:
    checksum_path = root / validate.CHECKSUM_FILENAME
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _rewrite_manifest(root: Path, difficulty: str, manifest: dict) -> None:
    manifest_path = root / "datasets" / f"{difficulty}.reviewed.json"
    _write_json(manifest_path, manifest)
    suite_path = root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["datasets"][difficulty].update(
        {
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": validate._sha256_json(manifest),
        }
    )
    _write_json(suite_path, suite)
    _write_checksums(root)


def _fixture_selected_response(difficulty: str, number: int) -> tuple[str, str, str]:
    case_id = f"{difficulty.upper()}-V2-{number:03d}"
    if difficulty == "easy" and number <= 4:
        source = "blocker"
    elif (difficulty == "easy" and 5 <= number <= 12) or (
        difficulty == "medium" and number <= 18
    ):
        source = "contract"
    else:
        source = "original"
    custom_id = f"{source}-{difficulty}-{number:03d}"
    row_sha256 = hashlib.sha256(
        f"fixture-response:{source}:{case_id}".encode("utf-8")
    ).hexdigest()
    return source, custom_id, row_sha256


def _scope_receipt(kind: str, count: int) -> dict:
    if kind == "original":
        selected = [
            (
                f"original-{difficulty}-{number:03d}",
                hashlib.sha256(
                    (
                        "fixture-response:original:"
                        f"{difficulty.upper()}-V2-{number:03d}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for difficulty in scope.SCOPE_DIFFICULTIES
            for number in range(1, 51)
        ]
    elif kind == "blocker":
        selected = [
            (_fixture_selected_response("easy", number)[1], _fixture_selected_response("easy", number)[2])
            for number in range(1, 5)
        ]
    else:
        selected = [
            (_fixture_selected_response("easy", number)[1], _fixture_selected_response("easy", number)[2])
            for number in range(5, 13)
        ] + [
            (_fixture_selected_response("medium", number)[1], _fixture_selected_response("medium", number)[2])
            for number in range(1, 19)
        ]
    assert len(selected) == count
    receipt = {
        **scope.scope_profile(),
        "scope_id": scope.SCOPE_ID,
        "receipt_kind": kind,
        "scope_target_custom_ids": [custom_id for custom_id, _ in selected],
        "selected_case_count": count,
        "authoritative_success_count": count,
        "missing_case_count": 0,
        "hard_rows_counted": 0,
        "hard_usage_counted": 0,
        "hard_materialization_episodes_counted": 0,
        "selected_response_row_sha256_by_id": dict(selected),
        "requests_sha256": "1" * 64,
        "request_map_sha256_file": "2" * 64,
        "responses_sha256": "3" * 64,
    }
    receipt["receipt_sha256"] = validate._sha256_json(receipt)
    return receipt


def _fixture_receipts() -> dict[str, dict]:
    return {
        "original": _scope_receipt("original", 100),
        "blocker": _scope_receipt("blocker", 4),
        "contract": _scope_receipt("contract", 26),
    }


def _fixture_lineage_authority() -> dict[str, tuple[str, str]]:
    return {
        f"{difficulty.upper()}-V2-{number:03d}": (
            f"{difficulty.upper()}-{number:03d}",
            difficulty,
        )
        for difficulty in scope.SCOPE_DIFFICULTIES
        for number in range(1, 51)
    }


def _fixture_semantic_authority() -> dict[str, dict]:
    return {
        f"{difficulty.upper()}-V2-{number:03d}": {
            "query": "该打印机支持什么打印功能",
            "reference_answer": "支持自动双面打印。",
            "unanswerable": False,
            "expected_terms": None,
            "forbidden_terms": None,
            "relevant_evidence": [
                {
                    "evidence_id": f"ev-{difficulty}-{number:03d}",
                    "chunk_id": f"chunk-{difficulty}-{number:03d}",
                    "page": number,
                    "text": "设备支持自动双面打印功能。",
                }
            ],
            "required_branch_spans": ["该打印机支持什么打印功能"],
        }
        for difficulty in scope.SCOPE_DIFFICULTIES
        for number in range(1, 51)
    }


@pytest.fixture(autouse=True)
def _use_synthetic_current_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(validate, "_build_current_authoritative_scope_receipts"):
        monkeypatch.setattr(
            validate,
            "_build_current_authoritative_scope_receipts",
            _fixture_receipts,
        )
    if hasattr(validate, "_build_current_case_lineage_authority"):
        monkeypatch.setattr(
            validate,
            "_build_current_case_lineage_authority",
            _fixture_lineage_authority,
        )
    if hasattr(validate, "_build_current_case_semantic_authority"):
        monkeypatch.setattr(
            validate,
            "_build_current_case_semantic_authority",
            _fixture_semantic_authority,
        )
    if hasattr(validate, "_build_current_case_review_authority"):
        monkeypatch.setattr(
            validate,
            "_build_current_case_review_authority",
            lambda: deepcopy(_FIXTURE_CASE_REVIEW_AUTHORITY),
        )
    if hasattr(validate, "_build_frozen_candidate_source_authority"):
        monkeypatch.setattr(
            validate,
            "_build_frozen_candidate_source_authority",
            lambda: deepcopy(_FIXTURE_SOURCE_AUTHORITY),
        )


def _reviewed_fixture(root: Path) -> Path:
    _FIXTURE_CASE_REVIEW_AUTHORITY.clear()
    _FIXTURE_SOURCE_AUTHORITY.clear()
    profile = scope.scope_profile()
    receipts = _fixture_receipts()
    entries: dict[str, dict] = {}
    qa_rows: list[dict] = []
    upstream_root = root.parent / "upstream"
    source_path = upstream_root / "source.pdf"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"easy-medium scoped source")
    source_relative = source_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    source_sha256 = _sha256_file(source_path)
    _FIXTURE_SOURCE_AUTHORITY.update(
        {
            difficulty: {
                "relative_path": source_relative,
                "content_sha256": source_sha256,
                "page_count": 95,
                "kb_id": full_activate.EXPECTED_KB_ID,
                "document_id": full_activate.EXPECTED_DOCUMENT_ID,
                "document_version_id": full_activate.EXPECTED_DOCUMENT_VERSION_ID,
                "index_version_id": full_activate.EXPECTED_INDEX_VERSION_ID,
            }
            for difficulty in scope.SCOPE_DIFFICULTIES
        }
    )
    source = {
        "relative_path": source_relative,
        "content_sha256": source_sha256,
        "page_count": 95,
        "kb_id": full_activate.EXPECTED_KB_ID,
        "document_id": full_activate.EXPECTED_DOCUMENT_ID,
        "document_version_id": full_activate.EXPECTED_DOCUMENT_VERSION_ID,
        "index_version_id": full_activate.EXPECTED_INDEX_VERSION_ID,
        "layout": "A4 portrait",
        "logical_name": "source.pdf",
        "original_logical_name": "source.pdf",
        "parent_dataset_key": "fixture-parent-dataset",
        "parent_dataset_version": "1.0.0",
        "parent_manifest_file_sha256": "f" * 64,
        "parent_suite_key": "fixture-parent-suite",
        "reviewed_from_published_evidence": True,
    }
    for difficulty in scope.SCOPE_DIFFICULTIES:
        cases = []
        for number in range(1, 51):
            case_id = f"{difficulty.upper()}-V2-{number:03d}"
            parent_id = f"{difficulty.upper()}-{number:03d}"
            selected_source, selected_custom_id, selected_response_sha256 = (
                _fixture_selected_response(difficulty, number)
            )
            evidence = [
                {
                    "evidence_id": f"ev-{difficulty}-{number:03d}",
                    "chunk_id": f"chunk-{difficulty}-{number:03d}",
                    "page": number,
                    "text": "设备支持自动双面打印功能。",
                }
            ]
            branch_span = "该打印机支持什么打印功能"
            contract = {
                "schema_version": merge.full.REVIEWED_CONTRACT_SCHEMA,
                "contract_status": "llm_reviewed_candidate",
                "answerability": {
                    "required": "answer",
                    "unsupported_factual_additions_forbidden": True,
                    "contradictory_claims_forbidden": True,
                },
                "required_concepts": {
                    "all_of": [
                        {
                            "concept_id": "c01",
                            "canonical_text": "支持自动双面打印",
                            "any_of": ["自动双面打印", "支持双面打印"],
                            "derivation": {
                                "method": "full_suite_same_prompt_llm_review",
                                "reviewer_concept_key": "feature",
                                "custom_id": selected_custom_id,
                            },
                            "evidence_support": [
                                {
                                    "evidence_id": evidence[0]["evidence_id"],
                                    "method": "llm_reviewed_direct_support",
                                }
                            ],
                            "query_branch_spans": [branch_span],
                        }
                    ]
                },
                "critical_constraints": {
                    "numerics": [],
                    "identifiers": [],
                    "polarity": {},
                },
                "required_branches": [
                    {
                        "branch_id": "b01",
                        "query_span": branch_span,
                        "required_concept_ids": ["c01"],
                        "assignment_method": "full_suite_same_prompt_llm_review",
                    }
                ],
                "evidence_concept_coverage": [
                    {
                        "evidence_id": evidence[0]["evidence_id"],
                        "chunk_id": evidence[0]["chunk_id"],
                        "page": evidence[0]["page"],
                        "required_concept_ids": ["c01"],
                        "coverage_status": "covered",
                    }
                ],
                "review": {
                    "status": "llm_review_complete",
                    "verdict": "approve",
                    "llm_verdict": "approve",
                    "source_visual_override": False,
                    "model": "fixture",
                    "runner_version": "fixture",
                    "prompt_sha256": "a" * 64,
                    "custom_id": selected_custom_id,
                    "response_sha256": "b" * 64,
                    "review_notes": [],
                    "independent_validation_status": "pending",
                    "deterministic_candidate_review": {},
                },
            }
            groups, variant_audit = merge.full._materialize_concept_groups(
                contract,
                case_id=case_id,
                evidence=evidence,
            )
            contract_hash = validate._sha256_json(contract)
            groups_hash = validate._sha256_json(groups)
            cases.append(
                {
                    "case_id": case_id,
                    "query": branch_span,
                    "reference_answer": "支持自动双面打印。",
                    "unanswerable": False,
                    "relevant_evidence": evidence,
                    "metadata": {
                        "parent_case_id": parent_id,
                        "difficulty": difficulty,
                        "easy_medium_scope_id": scope.SCOPE_ID,
                        "reviewed_contract_hash": contract_hash,
                        "concept_groups_hash": groups_hash,
                        "concept_groups_source_contract_hash": contract_hash,
                        "concept_groups_materializer": "merge_reviewed_contracts.py:v1",
                        "concept_groups_variant_filter": variant_audit,
                    },
                    "answer_contract": contract,
                    "concept_groups": groups,
                }
            )
            qa_rows.append(
                {
                    "case_id": case_id,
                    "parent_case_id": parent_id,
                    "difficulty": difficulty,
                    "verdict": "approve",
                    "answer_contract_hash": contract_hash,
                    "concept_groups_hash": groups_hash,
                    "concept_groups_source_contract_hash": contract_hash,
                    "concept_group_count": len(groups["all_of"]),
                    "custom_id": contract["review"]["custom_id"],
                    "prompt_sha256": contract["review"]["prompt_sha256"],
                    "runner_version": contract["review"]["runner_version"],
                    "response_sha256": contract["review"]["response_sha256"],
                    "selected_response_source": selected_source,
                    "selected_response_sha256": selected_response_sha256,
                    "saved_response_quote_key_branch_span_recovery": None,
                    "scope_id": scope.SCOPE_ID,
                }
            )
            _FIXTURE_CASE_REVIEW_AUTHORITY[case_id] = {
                "parent_case_id": parent_id,
                "difficulty": difficulty,
                "answer_contract": deepcopy(contract),
                "selected_response": {
                    "source": selected_source,
                    "custom_id": contract["review"]["custom_id"],
                    "row_sha256": selected_response_sha256,
                    "review_sha256": contract["review"]["response_sha256"],
                    "prompt_sha256": contract["review"]["prompt_sha256"],
                    "runner_version": contract["review"]["runner_version"],
                    "quote_key_recovery": None,
                },
            }
        manifest = {
            "schema_version": "stt.eval.dataset.v2-llm-reviewed-candidate",
            "suite_key": validate.SUITE_KEY,
            "dataset_key": validate.DATASET_KEYS[difficulty],
            "name": f"Easy/medium {difficulty} reviewed fixture",
            "version": validate.VERSION,
            "difficulty": difficulty,
            "scope": profile,
            "status": "reviewed_candidate",
            "import_allowed": False,
            "candidate_run_allowed": True,
            "promotable": False,
            "baseline_registration_allowed": False,
            "canonical_promotion_allowed": False,
            "source": source,
            "generation": {
                "llm_review_status": "complete",
                "independent_validation_status": "pending",
                "answer_contract_to_concept_groups_materialization_status": "complete",
            },
            "gate_policy": {
                "case_pass_policy": merge.full.FROZEN_CASE_PASS_POLICY,
                "case_pass_policy_activation_status": "reserved_not_active",
                "case_pass_policy_materialization_required": True,
            },
            "cases": cases,
        }
        relative = f"datasets/{difficulty}.reviewed.json"
        path = root / relative
        _write_json(path, manifest)
        entries[difficulty] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "manifest_sha256": validate._sha256_json(manifest),
            "dataset_key": validate.DATASET_KEYS[difficulty],
            "version": validate.VERSION,
            "case_count": 50,
        }
    source_summary_path = upstream_root / "source-summary.json"
    source_summary = {
        "input_checks": {
            "source_path": source_relative,
            "source_sha256": source_sha256,
            "source_hash_matches": True,
            "source_page_count_matches": True,
            "source_pdf_page_count": 95,
            "joined_case_count": 150,
            "indexed_evidence_count": 218,
            "index_version_id": full_activate.EXPECTED_INDEX_VERSION_ID,
        }
    }
    _write_json(source_summary_path, source_summary)
    blocker_validation_path = upstream_root / "blocker-validation.json"
    blocker_validation = {
        "status": "passed",
        "selected_chunks_resolve_to_frozen_active_index": True,
        "active_index": {
            "kb_id": full_activate.EXPECTED_KB_ID,
            "document_id": full_activate.EXPECTED_DOCUMENT_ID,
            "document_version_id": full_activate.EXPECTED_DOCUMENT_VERSION_ID,
            "index_version_id": full_activate.EXPECTED_INDEX_VERSION_ID,
            "frozen_chunk_count": 218,
            "index_rebuild_required": False,
            "evidence_refresh_required": False,
        },
    }
    _write_json(blocker_validation_path, blocker_validation)
    integrity = {
        "source": {
            "path": source["relative_path"],
            "sha256": source_sha256,
            "page_count": 95,
        },
        "active_index": {
            "kb_id": full_activate.EXPECTED_KB_ID,
            "document_id": full_activate.EXPECTED_DOCUMENT_ID,
            "document_version_id": full_activate.EXPECTED_DOCUMENT_VERSION_ID,
            "index_version_id": full_activate.EXPECTED_INDEX_VERSION_ID,
            "frozen_chunk_count": 218,
            "source_qa_summary_path": source_summary_path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            "source_qa_summary_sha256": _sha256_file(source_summary_path),
            "blocker_validation_path": blocker_validation_path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            "blocker_validation_sha256": _sha256_file(blocker_validation_path),
        },
        "review": {
            "scope_id": scope.SCOPE_ID,
            "scoped_case_count": 100,
            "hard_reference_only": True,
            "hard_imported_case_count": 0,
            "full_source_joined_case_count_reference_only": 150,
            "block_union_audit_path": "fixture/block-union-audit.json",
            "block_union_audit_sha256": "0" * 64,
            "original_scope_receipt_sha256": receipts["original"]["receipt_sha256"],
            "blocker_scope_receipt_sha256": receipts["blocker"]["receipt_sha256"],
            "contract_scope_receipt_sha256": receipts["contract"]["receipt_sha256"],
        },
    }
    suite_document = {
        "schema_version": validate.SUITE_SCHEMA,
        "suite_key": validate.SUITE_KEY,
        "version": validate.VERSION,
        "status": "reviewed_candidate_pending_independent_validation",
        "scope": profile,
        "import_allowed": False,
        "candidate_run_allowed": True,
        "promotable": False,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "datasets": entries,
        "source_and_index_integrity": integrity,
    }
    _write_json(root / validate.SUITE_FILENAME, suite_document)
    summary = {
        "schema_version": merge.SUMMARY_SCHEMA,
        "status": "reviewed_candidate_pending_independent_validation",
        "suite_key": validate.SUITE_KEY,
        "scope_id": scope.SCOPE_ID,
        "scope": profile,
        "case_count": 100,
        "blocked_case_count": 0,
        "difficulty_case_counts": {"easy": 50, "medium": 50},
        "blocker_rereview_counts": {"easy": 4, "medium": 0},
        "contract_rereview_counts": {"easy": 8, "medium": 18},
        "authoritative_scope_receipts": receipts,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "candidate_run_allowed": True,
        "promotable": False,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "source_and_index_integrity_sha256": validate._sha256_json(integrity),
    }
    _write_json(root / validate.SUMMARY_FILENAME, summary)
    _write_json(root / validate.QA_FILENAME, qa_rows)
    _write_checksums(root)
    return root


def test_scope_profile_is_exact_and_rejects_unknown_fields() -> None:
    profile = scope.scope_profile()
    assert profile["candidate_run_allowed"] is True
    assert profile["canonical_promotion_allowed"] is False
    assert profile["hard_reference_only"] is True
    assert profile["difficulties"] == ["easy", "medium"]
    validate._require_exact_scope(profile, label="test")
    with pytest.raises(validate.ScopedValidationError, match="frozen scope profile"):
        validate._require_exact_scope({**profile, "hard": "ignored"}, label="test")


def test_scope_receipt_excludes_hard_rows_usage_and_episodes() -> None:
    easy_payload = {"query": "easy authority"}
    hard_payload = {"query": "hard authority"}
    requests = [
        {
            "custom_id": "easy-id",
            "payload_sha256": validate._sha256_json(easy_payload),
            "request": {
                "messages": [
                    {"role": "system", "content": "scope test"},
                    {"role": "user", "content": json.dumps(easy_payload)},
                ]
            },
        },
        {
            "custom_id": "hard-id",
            "payload_sha256": validate._sha256_json(hard_payload),
            "request": {
                "messages": [
                    {"role": "system", "content": "scope test"},
                    {"role": "user", "content": json.dumps(hard_payload)},
                ]
            },
        },
    ]
    request_map = [
        {
            "custom_id": "easy-id",
            "case_id": "EASY-V2-001",
            "parent_case_id": "EASY-001",
            "difficulty": "easy",
        },
        {
            "custom_id": "hard-id",
            "case_id": "HARD-V2-001",
            "parent_case_id": "HARD-001",
            "difficulty": "hard",
        },
    ]
    selection = scope.select_scope_requests(
        requests,
        request_map,
        scope.SCOPE_NAME,
        authority_rows=[
            {
                **row,
                "payload_sha256": requests[index]["payload_sha256"],
            }
            for index, row in enumerate(request_map)
        ],
    )
    easy = {
        "custom_id": "easy-id",
        "status": "succeeded",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    hard = {
        "custom_id": "hard-id",
        "status": "succeeded",
        "usage": {"prompt_tokens": 200, "completion_tokens": 300, "total_tokens": 500},
    }
    receipt = merge._authoritative_scope_receipt(
        kind="test",
        selection=selection,
        successful={"easy-id": easy, "hard-id": hard},
        physical_rows=[easy, hard],
        requests_path=CONTRACT_ROOT / "review" / "requests.jsonl",
        request_map_path=CONTRACT_ROOT / "review" / "request-map.json",
        responses_path=CONTRACT_ROOT / "review" / "responses.jsonl",
        execution_contract_sha256="d" * 64,
    )
    assert receipt["authoritative_success_count"] == 1
    assert receipt["selected_usage"]["total_tokens"] == 5
    assert receipt["selected_physical_row_count"] == 1
    assert receipt["hard_rows_counted"] == 0
    assert receipt["hard_usage_counted"] == 0
    assert receipt["hard_materialization_episodes_counted"] == 0


def test_scoped_merge_refuses_all_missing_em_rereviews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        merge,
        "scope_readiness",
        lambda: {
            "missing_case_count": 28,
            "blocker": {"missing_case_count": 2},
            "contract": {"missing_case_count": 26},
        },
    )
    with pytest.raises(merge.ScopedMergeError, match="28 case"):
        merge._load_scope_rereviews()


def test_blocker_readiness_rejects_difficulty_swap_against_candidate_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (merge.BLOCKER_REREVIEW_ROOT / "request-map.json").resolve()
    original_read_json = merge._read_json

    def tampered_read_json(path: Path) -> object:
        value = original_read_json(path)
        if path.resolve() != target:
            return value
        rows = deepcopy(value)
        easy = next(row for row in rows if row["difficulty"] == "easy")
        hard = next(row for row in rows if row["difficulty"] == "hard")
        easy["difficulty"], hard["difficulty"] = hard["difficulty"], easy["difficulty"]
        return rows

    monkeypatch.setattr(merge, "_read_json", tampered_read_json)
    with pytest.raises(RuntimeError, match="request-map.*authority"):
        merge._validate_full_pack_and_replay(
            kind="blocker",
            rereview_root=merge.BLOCKER_REREVIEW_ROOT,
        )


def test_contract_readiness_rejects_whole_lineage_swap_against_original_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (merge.CONTRACT_REREVIEW_ROOT / "request-map.json").resolve()
    original_read_json = merge._read_json

    def tampered_read_json(path: Path) -> object:
        value = original_read_json(path)
        if path.resolve() != target:
            return value
        rows = deepcopy(value)
        easy = next(row for row in rows if row["difficulty"] == "easy")
        hard = next(row for row in rows if row["difficulty"] == "hard")
        for field in ("case_id", "parent_case_id", "difficulty"):
            easy[field], hard[field] = hard[field], easy[field]
        return rows

    monkeypatch.setattr(merge, "_read_json", tampered_read_json)
    with pytest.raises(RuntimeError, match="request-map.*authority"):
        merge._validate_full_pack_and_replay(
            kind="contract",
            rereview_root=merge.CONTRACT_REREVIEW_ROOT,
        )


def test_build_reports_scope_gap_before_loading_original_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_scope() -> tuple:
        raise merge.ScopedMergeError(
            "easy/medium authoritative rereview is incomplete: 28 case(s) missing"
        )

    monkeypatch.setattr(merge, "_load_scope_rereviews", refuse_scope)
    monkeypatch.setattr(
        merge,
        "_load_scope_original_reviews",
        lambda: pytest.fail("original history must not load before scoped readiness"),
    )
    with pytest.raises(merge.ScopedMergeError, match="28 case"):
        merge.build()


def test_original_review_loader_strictly_adopts_only_em_ids() -> None:
    reviews, prompt_sha256, receipt = merge._load_scope_original_reviews()
    _, _, selection = merge._load_validated_original_scope_inputs()
    assert len(reviews) == 100
    assert set(reviews) == set(selection.target_custom_ids)
    assert prompt_sha256 == merge.full.EXPECTED_REVIEW_PROMPT_SHA256
    assert receipt["selected_case_count"] == 100
    assert receipt["authoritative_success_count"] == 100
    assert receipt["hard_rows_counted"] == 0
    assert receipt["hard_usage_counted"] == 0


def test_validate_and_activate_exact_easy_medium_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    receipt = validate.validate(reviewed_root)
    assert receipt["status"] == "passed"
    assert receipt["case_count"] == 100
    assert tuple(receipt["datasets"]) == ("easy", "medium")
    assert receipt["hard_imported_case_count"] == 0
    monkeypatch.setattr(
        activate,
        "_safe_relative_project",
        lambda path: f"fixture/{Path(path).name}",
    )
    activation, quality, manifests, preflight = activate.build_activation(
        reviewed_root=reviewed_root
    )
    assert activation["schema_version"] == activate.ACTIVATION_SCHEMA
    assert activation["scope"] == scope.scope_profile()
    assert tuple(activation["datasets"]) == ("easy", "medium")
    assert set(quality) == {
        "schema_version",
        "suite_key",
        "primary_metric",
        "independent_blind_review",
        "targets",
    }
    assert "hard" not in quality["targets"]
    assert quality["targets"]["easy"]["minimum_correct_cases"] == 48
    assert quality["targets"]["medium"]["minimum_correct_cases"] == 43
    assert activation["candidate_run_allowed"] is True
    assert activation["promotable"] is False
    assert activation["baseline_registration_allowed"] is False
    assert activation["canonical_promotion_allowed"] is False
    assert preflight["case_count"] == 100
    assert all(
        manifest["candidate_run_allowed"] is True
        and manifest["promotable"] is False
        and manifest["canonical_promotion_allowed"] is False
        for manifest in manifests.values()
    )


def test_validator_rejects_unknown_scope_field_after_valid_checksum(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite_document = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_document["scope"]["unknown_field"] = True
    _write_json(suite_path, suite_document)
    _write_checksums(reviewed_root)
    with pytest.raises(validate.ScopedValidationError, match="frozen scope profile"):
        validate.validate(reviewed_root)


def test_validator_rejects_checksum_tampering(tmp_path: Path) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(validate.ScopedValidationError, match="checksum mismatch"):
        validate.validate(reviewed_root)


def test_validator_rejects_unknown_review_verdict_with_valid_checksum(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["answer_contract"]["review"]["verdict"] = "unknown"
    _write_json(manifest_path, manifest)
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["datasets"]["easy"].update(
        {
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": validate._sha256_json(manifest),
        }
    )
    _write_json(suite_path, suite)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="not reviewed-ready"):
        validate.validate(reviewed_root)


def test_validator_rejects_qa_difficulty_swap_with_valid_checksum(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    easy_index = next(index for index, row in enumerate(qa) if row["difficulty"] == "easy")
    medium_index = next(
        index for index, row in enumerate(qa) if row["difficulty"] == "medium"
    )
    qa[easy_index]["difficulty"], qa[medium_index]["difficulty"] = (
        qa[medium_index]["difficulty"],
        qa[easy_index]["difficulty"],
    )
    _write_json(qa_path, qa)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="QA lineage/verdict drifted"):
        validate.validate(reviewed_root)


def test_activation_output_is_accepted_by_scoped_runner(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    final_root = tmp_path / "final"

    result = activate.activate(
        write_final_root=True,
        reviewed_root=reviewed_root,
        final_root=final_root,
    )
    snapshot = live.load_suite_snapshot(final_root)

    assert result["status"] == "activated"
    assert snapshot.activation["scope"] == scope.scope_profile()
    assert tuple(snapshot.datasets) == scope.SCOPE_DIFFICULTIES
    assert sum(item.case_count for item in snapshot.datasets.values()) == 100


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("difficulty", "medium"),
        ("easy_medium_scope_id", "tampered-scope"),
    ),
)
def test_validator_rejects_case_metadata_drift_with_valid_checksums(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0].setdefault("metadata", {})[field] = value
    _write_json(manifest_path, manifest)
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["datasets"]["easy"].update(
        {
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": validate._sha256_json(manifest),
        }
    )
    _write_json(suite_path, suite)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="metadata"):
        validate.validate(reviewed_root)


def test_validator_rejects_materialized_variant_drift_with_valid_hashes(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["concept_groups"]["all_of"][0]["any_of"] = [
        "unrelated variant"
    ]
    _write_json(manifest_path, manifest)
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["datasets"]["easy"].update(
        {
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": validate._sha256_json(manifest),
        }
    )
    _write_json(suite_path, suite)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="materializ"):
        validate.validate(reviewed_root)


def test_validator_rejects_receipt_selected_id_tamper_with_valid_checksums(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["authoritative_scope_receipts"]["blocker"][
        "selected_response_row_sha256_by_id"
    ]
    digest = rows.pop(next(iter(rows)))
    rows["foreign-id"] = digest
    _write_json(summary_path, summary)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="receipt"):
        validate.validate(reviewed_root)


def test_validator_rejects_constraint_unknown_evidence_after_hash_rewrite(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    case["answer_contract"]["critical_constraints"]["numerics"] = [
        {
            "value": "9",
            "unit": None,
            "evidence_ids": ["foreign-evidence"],
            "required_exact": True,
        }
    ]
    contract_hash = validate._sha256_json(case["answer_contract"])
    case["metadata"]["reviewed_contract_hash"] = contract_hash
    case["metadata"]["concept_groups_source_contract_hash"] = contract_hash
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0]["answer_contract_hash"] = contract_hash
    qa[0]["concept_groups_source_contract_hash"] = contract_hash
    _write_json(qa_path, qa)
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="coverage"):
        validate.validate(reviewed_root)


def test_validator_rejects_evidence_identity_drift_with_valid_checksums(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["relevant_evidence"][0]["evidence_id"] = "foreign-evidence"
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="semantic/evidence"):
        validate.validate(reviewed_root)


def test_validator_rejects_qa_contract_hash_tamper_with_valid_checksums(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0]["answer_contract_hash"] = "f" * 64
    _write_json(qa_path, qa)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="hash mismatched"):
        validate.validate(reviewed_root)


def test_validator_rejects_qa_quote_key_recovery_not_in_selected_response(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0]["saved_response_quote_key_branch_span_recovery"] = {
        "policy": "foreign-self-signed-audit"
    }
    _write_json(qa_path, qa)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="provenance"):
        validate.validate(reviewed_root)


def test_validator_rejects_co_tampered_reviewed_contract_against_current_ledger(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    concept = case["answer_contract"]["required_concepts"]["all_of"][0]
    concept["canonical_text"] = "设备"
    concept["any_of"] = ["设备"]
    groups, variant_audit = merge.full._materialize_concept_groups(
        case["answer_contract"],
        case_id=case["case_id"],
        evidence=case["relevant_evidence"],
    )
    case["concept_groups"] = groups
    contract_hash = validate._sha256_json(case["answer_contract"])
    groups_hash = validate._sha256_json(groups)
    case["metadata"].update(
        {
            "reviewed_contract_hash": contract_hash,
            "concept_groups_hash": groups_hash,
            "concept_groups_source_contract_hash": contract_hash,
            "concept_groups_variant_filter": variant_audit,
        }
    )
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0].update(
        {
            "answer_contract_hash": contract_hash,
            "concept_groups_hash": groups_hash,
            "concept_groups_source_contract_hash": contract_hash,
            "concept_group_count": len(groups["all_of"]),
        }
    )
    _write_json(qa_path, qa)
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="review authority"):
        validate.validate(reviewed_root)


def test_validator_rejects_co_tampered_review_and_qa_response_provenance(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    review = case["answer_contract"]["review"]
    review.update(
        {
            "model": "foreign-model",
            "custom_id": "foreign-custom-id",
            "response_sha256": "f" * 64,
        }
    )
    contract_hash = validate._sha256_json(case["answer_contract"])
    case["metadata"]["reviewed_contract_hash"] = contract_hash
    case["metadata"]["concept_groups_source_contract_hash"] = contract_hash
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0].update(
        {
            "answer_contract_hash": contract_hash,
            "concept_groups_source_contract_hash": contract_hash,
            "custom_id": "foreign-custom-id",
            "response_sha256": "f" * 64,
            "selected_response_source": "contract",
            "selected_response_sha256": "e" * 64,
        }
    )
    _write_json(qa_path, qa)
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="review authority|provenance"):
        validate.validate(reviewed_root)


def test_validator_rejects_co_tampered_parent_lineage_against_frozen_authority(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["metadata"]["parent_case_id"] = "FOREIGN-PARENT"
    qa_path = reviewed_root / validate.QA_FILENAME
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa[0]["parent_case_id"] = "FOREIGN-PARENT"
    _write_json(qa_path, qa)
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="frozen authority"):
        validate.validate(reviewed_root)


def test_validator_rejects_current_ledger_freshness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    current = _fixture_receipts()
    current["contract"]["responses_sha256"] = "d" * 64
    current["contract"].pop("receipt_sha256")
    current["contract"]["receipt_sha256"] = validate._sha256_json(current["contract"])
    monkeypatch.setattr(
        validate,
        "_build_current_authoritative_scope_receipts",
        lambda: current,
    )

    with pytest.raises(validate.ScopedValidationError, match="stale"):
        validate.validate(reviewed_root)


def test_validator_rejects_receipt_source_file_sha_tamper_after_self_rehash(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = summary["authoritative_scope_receipts"]["blocker"]
    receipt["requests_sha256"] = "e" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = validate._sha256_json(receipt)
    _write_json(summary_path, summary)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="stale"):
        validate.validate(reviewed_root)


def test_validator_rejects_source_and_index_sha_cross_binding_tamper(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["source_and_index_integrity"]["source"]["sha256"] = "0" * 64
    _write_json(suite_path, suite)
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_and_index_integrity_sha256"] = validate._sha256_json(
        suite["source_and_index_integrity"]
    )
    _write_json(summary_path, summary)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="file SHA"):
        validate.validate(reviewed_root)


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
def test_validator_rejects_co_rehashed_reviewed_promotion_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="top-level"):
        validate.validate(reviewed_root)


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
def test_activation_derivation_rejects_reviewed_promotion_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value

    with pytest.raises(activate.ScopedActivationError, match="top-level"):
        activate._derive_active_manifest(
            manifest,
            input_relative_path="reviewed/datasets/easy.reviewed.json",
            input_file_sha256=_sha256_file(manifest_path),
        )


@pytest.mark.parametrize("document", ("suite", "summary"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("promotion_allowed", True),
        ("canonical", True),
        ("candidate_roots", {"easy": "forbidden"}),
    ),
)
def test_validator_rejects_suite_or_summary_top_level_promotion_authority(
    tmp_path: Path,
    document: str,
    field: str,
    value: object,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    path = reviewed_root / (
        validate.SUITE_FILENAME if document == "suite" else validate.SUMMARY_FILENAME
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="top-level"):
        validate.validate(reviewed_root)


def test_validator_rejects_physical_source_co_tamper_against_candidate_anchor(
    tmp_path: Path,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT / suite["source_and_index_integrity"]["source"]["path"]
    source_path.write_bytes(b"coordinated replacement source")
    changed_sha256 = _sha256_file(source_path)
    source_summary_path = (
        PROJECT_ROOT
        / suite["source_and_index_integrity"]["active_index"]["source_qa_summary_path"]
    )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["input_checks"]["source_sha256"] = changed_sha256
    _write_json(source_summary_path, source_summary)
    integrity = suite["source_and_index_integrity"]
    integrity["source"]["sha256"] = changed_sha256
    integrity["active_index"]["source_qa_summary_sha256"] = _sha256_file(
        source_summary_path
    )
    _write_json(suite_path, suite)
    for difficulty in scope.SCOPE_DIFFICULTIES:
        manifest_path = reviewed_root / "datasets" / f"{difficulty}.reviewed.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"]["content_sha256"] = changed_sha256
        _write_json(manifest_path, manifest)
        suite["datasets"][difficulty].update(
            {
                "sha256": _sha256_file(manifest_path),
                "manifest_sha256": validate._sha256_json(manifest),
            }
        )
    _write_json(suite_path, suite)
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_and_index_integrity_sha256"] = validate._sha256_json(integrity)
    _write_json(summary_path, summary)
    _write_checksums(reviewed_root)

    with pytest.raises(validate.ScopedValidationError, match="frozen candidate"):
        validate.validate(reviewed_root)


@pytest.mark.parametrize("layer", ("integrity", "source", "active_index", "review"))
def test_validator_rejects_nested_source_integrity_promotion_authority(
    tmp_path: Path,
    layer: str,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    suite_path = reviewed_root / validate.SUITE_FILENAME
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    integrity = suite["source_and_index_integrity"]
    target = integrity if layer == "integrity" else integrity[layer]
    target["promotion_allowed"] = True
    _write_json(suite_path, suite)
    summary_path = reviewed_root / validate.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_and_index_integrity_sha256"] = validate._sha256_json(integrity)
    _write_json(summary_path, summary)
    _write_checksums(reviewed_root)

    with pytest.raises(
        validate.ScopedValidationError,
        match="integrity schema|authorization semantics",
    ):
        validate.validate(reviewed_root)


@pytest.mark.parametrize(
    "location",
    ("source", "generation", "gate_policy", "case", "metadata"),
)
def test_validator_recursively_rejects_manifest_promotion_authority(
    tmp_path: Path,
    location: str,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    manifest_path = reviewed_root / "datasets" / "easy.reviewed.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location in {"source", "generation", "gate_policy"}:
        target = manifest[location]
    elif location == "case":
        target = manifest["cases"][0]
    else:
        target = manifest["cases"][0]["metadata"]
    target["promotion_allowed"] = True
    _rewrite_manifest(reviewed_root, "easy", manifest)

    with pytest.raises(validate.ScopedValidationError, match="authorization semantics"):
        validate.validate(reviewed_root)


@pytest.mark.parametrize(
    "relative",
    (
        "candidate-roots.json",
        "baselines/canonical.json",
        "certification/promotion.json",
    ),
)
def test_activation_writer_rejects_root_level_promotion_surface(
    tmp_path: Path,
    relative: str,
) -> None:
    reviewed_root = _reviewed_fixture(tmp_path / "reviewed")
    final_root = tmp_path / "final"
    _write_json(final_root / relative, {"promotion_allowed": True})

    with pytest.raises(activate.ScopedActivationError, match="promotion surface"):
        activate.activate(
            write_final_root=True,
            reviewed_root=reviewed_root,
            final_root=final_root,
        )
