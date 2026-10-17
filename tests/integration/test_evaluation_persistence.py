"""TODO-18 评测、反馈与标注事实的真实 SQLite/Alembic 持久化测试。"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session

from app.domain.evaluation import content_digest
from app.evaluation.dataset_loader import load_dataset
from app.infra.persistence.sql.evaluation_repository import EvaluationRepository
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_TABLES = {
    "evaluation_datasets",
    "evaluation_cases",
    "evaluation_runs",
    "evaluation_results",
    "feedback",
    "annotation_tasks",
}


def _upgrade_to_0006(database_url: str) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    old_url = os.environ.get("CONTROL_DATABASE_URL")
    os.environ["CONTROL_DATABASE_URL"] = database_url
    try:
        command.upgrade(config, "20260721_0006")
    finally:
        if old_url is None:
            os.environ.pop("CONTROL_DATABASE_URL", None)
        else:
            os.environ["CONTROL_DATABASE_URL"] = old_url


def _engine_for(path: Path) -> Engine:
    database_url = f"sqlite:///{path.as_posix()}"
    _upgrade_to_0006(database_url)
    engine = create_engine(database_url, future=True)
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    return engine


@pytest.fixture(scope="module")
def evaluation_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    engine = _engine_for(tmp_path_factory.mktemp("evaluation-db") / "control.db")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session(evaluation_engine: Engine) -> Iterator[Session]:
    with Session(evaluation_engine, expire_on_commit=False) as database_session:
        yield database_session
        database_session.rollback()


def test_alembic_0006_creates_all_six_evaluation_tables(tmp_path: Path) -> None:
    engine = _engine_for(tmp_path / "migration-only.db")
    try:
        assert EVALUATION_TABLES <= set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "20260721_0006"
            )
    finally:
        engine.dispose()


def test_dataset_version_is_idempotent_for_same_content_and_immutable_for_changes(
    session: Session,
) -> None:
    manifest = load_dataset()
    repository = EvaluationRepository(session, tenant_id="tenant-immutable", actor_id="publisher")

    first = repository.seed_manifest(manifest)
    repeated = repository.seed_manifest(copy.deepcopy(manifest))

    assert repeated["id"] == first["id"]
    assert repeated["checksum"] == first["checksum"]
    assert repeated["manifest_json"]["gate_policy"] == manifest["gate_policy"]
    assert repeated["case_count"] == 28
    assert len(repository.list_datasets()) == 1

    changed = copy.deepcopy(manifest)
    changed["description"] += "（未经升版的修改）"
    with pytest.raises(ResourceConflictError, match="immutable"):
        repository.seed_manifest(changed)


def test_case_authored_contracts_round_trip_without_colliding_with_user_metadata(
    session: Session,
) -> None:
    manifest = copy.deepcopy(load_dataset())
    manifest["dataset_key"] = "concept-groups-persistence-test"
    authored_groups = {
        "all_of": [
            {
                "concept_key": "minimum_temperature",
                "any_of": ["10 °C", "10℃"],
            }
        ]
    }
    ordinary_metadata_groups = {
        "source": "user-metadata",
        "labels": ["voltage", "manual"],
    }
    authored_claims = {
        "contract_version": "stt.eval.authored-claims.v1",
        "atoms": [
            {
                "claim_key": "temperature_range",
                "requires_concept_keys": ["minimum_temperature", "maximum_temperature"],
                "citation_required": True,
                "source_alignment": {
                    "source_terms": ["10 °C", "32 °C"],
                    "anchors": [],
                },
            }
        ],
    }
    authored_case = manifest["cases"][0]
    source_evidence = authored_case["relevant_evidence"][0]
    authored_groups["contract_version"] = "stt.eval.authored-concepts.v2"
    authored_groups["all_of"][0]["source_alignment"] = {
        "mode": "required",
        "source_terms": ["10 °C"],
        "anchors": [
            {
                "evidence_id": source_evidence["evidence_id"],
                "text": source_evidence["text"],
            }
        ],
    }
    authored_case["concept_groups"] = authored_groups
    authored_case["authored_claims"] = authored_claims
    authored_case["metadata"] = {
        **dict(authored_case.get("metadata") or {}),
        "concept_groups": ordinary_metadata_groups,
        "authored_claims": {"source": "user-metadata"},
    }
    legacy_case = manifest["cases"][1]
    legacy_case["metadata"] = {
        **dict(legacy_case.get("metadata") or {}),
        "concept_groups": {"source": "ordinary-only"},
    }
    unchanged_manifest = copy.deepcopy(manifest)
    expected_checksum = content_digest(manifest)
    repository = EvaluationRepository(
        session, tenant_id="tenant-concept-groups", actor_id="publisher"
    )

    seeded = repository.seed_manifest(manifest)
    repeated = repository.seed_manifest(copy.deepcopy(manifest))
    fetched = repository.get_dataset(seeded["id"])
    run = repository.create_run(
        dataset_id=seeded["id"],
        kb_id="kb-concept-groups",
        name="concept groups round trip",
        mode="retrieval_only",
        baseline_run_id=None,
        config_snapshot={},
        case_ids=[authored_case["case_id"], legacy_case["case_id"]],
    )
    selected = repository.selected_cases_for_run(run["id"])

    assert manifest == unchanged_manifest
    assert repeated["id"] == seeded["id"]
    assert seeded["checksum"] == repeated["checksum"] == fetched["checksum"]
    assert fetched["checksum"] == expected_checksum
    for cases in (seeded["cases"], repeated["cases"], fetched["cases"], selected):
        by_id = {case["case_id"]: case for case in cases}
        persisted = by_id[authored_case["case_id"]]
        assert persisted["concept_groups"] == authored_groups
        assert persisted["authored_claims"] == authored_claims
        assert persisted["metadata"] == authored_case["metadata"]
        assert set(persisted["metadata"]) == set(authored_case["metadata"])

        without_top_level_groups = by_id[legacy_case["case_id"]]
        assert "concept_groups" not in without_top_level_groups
        assert "authored_claims" not in without_top_level_groups
        assert without_top_level_groups["metadata"] == legacy_case["metadata"]


def test_dataset_and_run_dtos_project_optional_suite_metadata(session: Session) -> None:
    manifest = copy.deepcopy(load_dataset())
    manifest.update(
        {
            "dataset_key": "three-difficulty-projection-test",
            "name": "Three difficulty projection test",
            "suite_key": "three-difficulty-live-2026-08-10",
            "difficulty": "medium",
            "minimum_case_count": len(manifest["cases"]),
            "evaluation_mode_default": "live",
        }
    )
    repository = EvaluationRepository(
        session, tenant_id="tenant-suite-projection", actor_id="publisher"
    )

    dataset = repository.seed_manifest(manifest)
    run = repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-three-difficulty",
        name="medium live baseline",
        mode="live",
        baseline_run_id=None,
        config_snapshot={},
    )

    expected = {
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": "medium",
        "minimum_case_count": len(manifest["cases"]),
        "evaluation_mode_default": "live",
    }
    assert {key: dataset[key] for key in expected} == expected
    assert {key: run[key] for key in expected} == expected


def test_baseline_and_candidate_must_use_the_same_mode(session: Session) -> None:
    repository = EvaluationRepository(session, tenant_id="tenant-mode", actor_id="evaluator")
    dataset = repository.seed_manifest(load_dataset())
    baseline = repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-hak180",
        name="retrieval baseline",
        mode="retrieval_only",
        baseline_run_id=None,
        config_snapshot={},
        case_ids=["HAK180-001"],
    )

    with pytest.raises(ResourceConflictError, match="same evaluation mode"):
        repository.create_run(
            dataset_id=dataset["id"],
            kb_id="kb-hak180",
            name="invalid live candidate",
            mode="live",
            baseline_run_id=baseline["id"],
            config_snapshot={},
            case_ids=["HAK180-001"],
        )


def test_baseline_and_candidate_must_use_the_same_complete_case_set(
    session: Session,
) -> None:
    repository = EvaluationRepository(session, tenant_id="tenant-paired-set", actor_id="evaluator")
    dataset = repository.seed_manifest(load_dataset())
    baseline = repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-hak180",
        name="paired baseline",
        mode="retrieval_only",
        baseline_run_id=None,
        config_snapshot={},
        case_ids=["HAK180-001", "HAK180-002"],
    )

    with pytest.raises(ResourceConflictError, match="same complete case set"):
        repository.create_run(
            dataset_id=dataset["id"],
            kb_id="kb-hak180",
            name="easy subset",
            mode="retrieval_only",
            baseline_run_id=baseline["id"],
            config_snapshot={},
            case_ids=["HAK180-001"],
        )

    candidate = repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-hak180",
        name="same complete set",
        mode="retrieval_only",
        baseline_run_id=baseline["id"],
        config_snapshot={},
        case_ids=["HAK180-002", "HAK180-001"],
    )
    assert candidate["total_cases"] == 2
    assert candidate["config_snapshot"]["selected_case_ids"] == [
        "HAK180-001",
        "HAK180-002",
    ]


def test_dataset_and_feedback_queries_are_tenant_isolated(session: Session) -> None:
    manifest = load_dataset()
    tenant_a = EvaluationRepository(session, tenant_id="tenant-a", actor_id="user-a")
    tenant_b = EvaluationRepository(session, tenant_id="tenant-b", actor_id="user-b")

    dataset_a = tenant_a.seed_manifest(manifest)
    dataset_b = tenant_b.seed_manifest(manifest)
    feedback_a = tenant_a.create_feedback({"kb_id": "kb-a", "answer_id": "answer-a", "rating": -1})
    tenant_b.create_feedback({"kb_id": "kb-b", "answer_id": "answer-b", "rating": 1})

    assert dataset_a["id"] != dataset_b["id"]
    assert [item["id"] for item in tenant_a.list_datasets()] == [dataset_a["id"]]
    assert [item["id"] for item in tenant_b.list_datasets()] == [dataset_b["id"]]
    assert [item["answer_id"] for item in tenant_a.list_feedback()] == ["answer-a"]
    assert [item["answer_id"] for item in tenant_b.list_feedback()] == ["answer-b"]
    with pytest.raises(ResourceNotFoundError):
        tenant_b.get_dataset(dataset_a["id"])
    with pytest.raises(ResourceNotFoundError):
        tenant_b.triage_feedback(feedback_a["id"], status="triaged", review_note="cross-tenant")


def test_run_and_case_results_survive_a_new_database_session(
    evaluation_engine: Engine,
) -> None:
    tenant_id = "tenant-run-persistence"
    with Session(evaluation_engine, expire_on_commit=False) as first_session:
        repository = EvaluationRepository(first_session, tenant_id=tenant_id, actor_id="evaluator")
        dataset = repository.seed_manifest(load_dataset())
        run = repository.create_run(
            dataset_id=dataset["id"],
            kb_id="kb-hak180",
            name="HAK180 persistence baseline",
            mode="offline",
            baseline_run_id=None,
            config_snapshot={"code_revision": "test-revision"},
            case_ids=["HAK180-001", "HAK180-002"],
        )
        repository.start_run(run["id"])
        repository.record_result(
            run["id"],
            "HAK180-001",
            observation={"answer": "10 °C 到 32 °C", "latency_ms": 12.5},
            metrics={"metrics": {"retrieval": {"recall_at_k": {"10": 1.0}}}},
            passed=True,
        )
        repository.record_result(
            run["id"],
            "HAK180-002",
            observation={"answer": "", "latency_ms": 8.0},
            metrics={"metrics": {"retrieval": {"recall_at_k": {"10": 0.0}}}},
            passed=False,
            error_type="AssertionError",
            error_detail="expected evidence was not retrieved",
        )
        completed_at = datetime(2026, 8, 15, 2, 23, 52, tzinfo=UTC)
        repository.complete_run(
            run["id"],
            metrics_summary={"case_count": 2},
            comparison_summary={},
            gate_result={"status": "baseline_recorded", "passed": True},
            report_paths={"json": {"path": "output/evaluation/report.json"}},
            completed_at=completed_at,
        )
        first_session.commit()
        run_id = run["id"]

    with Session(evaluation_engine, expire_on_commit=False) as second_session:
        repository = EvaluationRepository(second_session, tenant_id=tenant_id, actor_id="auditor")
        persisted_run = repository.get_run(run_id)
        persisted_results = repository.list_results(run_id)
        failed_results = repository.list_results(run_id, only_failed=True)

        assert persisted_run["status"] == "succeeded"
        assert persisted_run["progress"] == 1.0
        assert persisted_run["completed_cases"] == persisted_run["total_cases"] == 2
        assert persisted_run["config_snapshot"]["code_revision"] == "test-revision"
        assert persisted_run["gate_result"]["status"] == "baseline_recorded"
        assert persisted_run["completed_at"] == "2026-08-15T02:23:52+00:00"
        assert [item["case_id"] for item in persisted_results] == [
            "HAK180-001",
            "HAK180-002",
        ]
        assert [item["case_id"] for item in failed_results] == ["HAK180-002"]
        assert failed_results[0]["error_type"] == "AssertionError"

        with pytest.raises(ResourceNotFoundError):
            EvaluationRepository(
                second_session, tenant_id="tenant-other", actor_id="outsider"
            ).get_run(run_id)


def test_feedback_requires_two_distinct_reviewers_and_conflict_arbitration(
    session: Session,
) -> None:
    tenant_id = "tenant-feedback-workflow"
    submitter = EvaluationRepository(session, tenant_id=tenant_id, actor_id="submitter")
    feedback = submitter.create_feedback(
        {
            "kb_id": "kb-hak180",
            "answer_id": "answer-needs-correction",
            "rating": -1,
            "reason_codes": ["missing_evidence"],
            "comment": "没有引用温度范围证据",
            "corrected_answer": "应在 10 °C 到 32 °C 之间使用。",
        }
    )
    assert feedback["status"] == "pending"

    task = EvaluationRepository(
        session, tenant_id=tenant_id, actor_id="triager"
    ).create_annotation_task(
        feedback["id"],
        target_dataset_key="hak180-safety",
        proposed_case={"case_id": "HAK180-CANDIDATE-001", "query": "允许温度是多少？"},
    )
    assert task["status"] == "in_review"
    assert submitter.list_feedback()[0]["status"] == "in_review"

    reviewer_one = EvaluationRepository(session, tenant_id=tenant_id, actor_id="reviewer-one")
    first_review = reviewer_one.review_annotation(task["id"], decision="approve", note="证据可复核")
    assert first_review["status"] == "in_review"
    assert [review["reviewer_id"] for review in first_review["reviews"]] == ["reviewer-one"]

    with pytest.raises(ResourceConflictError, match="same reviewer"):
        reviewer_one.review_annotation(
            task["id"], decision="approve", note="不得重复充当第二位复核人"
        )

    reviewer_two = EvaluationRepository(session, tenant_id=tenant_id, actor_id="reviewer-two")
    conflicting = reviewer_two.review_annotation(
        task["id"], decision="reject", note="问题措辞需要调整"
    )
    assert conflicting["status"] == "conflict"
    assert [review["reviewer_id"] for review in conflicting["reviews"]] == [
        "reviewer-one",
        "reviewer-two",
    ]
    assert len({review["reviewer_id"] for review in conflicting["reviews"]}) == 2

    with pytest.raises(ResourceConflictError, match="arbiter must be independent"):
        reviewer_one.arbitrate_annotation(
            task["id"],
            decision="approve",
            note="审核人不能再充当仲裁人",
            resolved_case=None,
        )

    resolved_case = {
        "case_id": "HAK180-CANDIDATE-001",
        "query": "HAK 180 允许在什么温度范围内使用？",
    }
    arbitrated = EvaluationRepository(
        session, tenant_id=tenant_id, actor_id="arbiter"
    ).arbitrate_annotation(
        task["id"],
        decision="approve",
        note="采用更明确的产品名后批准",
        resolved_case=resolved_case,
    )

    assert arbitrated["status"] == "approved"
    assert arbitrated["arbiter_id"] == "arbiter"
    assert arbitrated["resolution"]["kind"] == "arbitration"
    assert arbitrated["resolution"]["resolved_case"] == resolved_case
    assert submitter.list_feedback()[0]["status"] == "accepted"


def test_feedback_result_binding_requires_same_kb_and_answer_id(
    session: Session,
) -> None:
    repository = EvaluationRepository(
        session, tenant_id="tenant-feedback-binding", actor_id="submitter"
    )
    dataset = repository.seed_manifest(load_dataset())
    run = repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-a",
        name="binding run",
        mode="live",
        baseline_run_id=None,
        config_snapshot={},
        case_ids=["HAK180-001"],
    )
    repository.start_run(run["id"])
    result = repository.record_result(
        run["id"],
        "HAK180-001",
        observation={"answer_id": "answer-a", "answer": "test"},
        metrics={},
        passed=True,
    )

    with pytest.raises(ResourceConflictError, match="knowledge base"):
        repository.create_feedback(
            {
                "kb_id": "kb-b",
                "answer_id": "answer-a",
                "evaluation_result_id": result["id"],
                "rating": -1,
            }
        )
    with pytest.raises(ResourceConflictError, match="answer_id"):
        repository.create_feedback(
            {
                "kb_id": "kb-a",
                "answer_id": "answer-b",
                "evaluation_result_id": result["id"],
                "rating": -1,
            }
        )

    created = repository.create_feedback(
        {
            "kb_id": "kb-a",
            "answer_id": "answer-a",
            "evaluation_result_id": result["id"],
            "rating": 1,
        }
    )
    assert created["evaluation_result_id"] == result["id"]
