from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

import app.application.evaluation_runtime_maintenance as maintenance
from app.application.evaluation_runtime_maintenance import (
    EVALUATION_INTERRUPTED_CODE,
    interrupt_owned_evaluation_runs,
    recover_stale_evaluation_runs,
    run_evaluation_recovery_loop,
)
from app.evaluation.dataset_loader import load_dataset
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.evaluation_repository import (
    EvaluationRepository,
    EvaluationRunOwnershipLost,
)
from app.infra.persistence.sql.models import (
    Base,
    EvaluationCaseModel,
    EvaluationDatasetModel,
    EvaluationResultModel,
    EvaluationRunModel,
)


def _configure(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'evaluation-recovery.db').as_posix()}"
    engine = configure_database(database_url)
    Base.metadata.create_all(
        engine,
        tables=[
            EvaluationDatasetModel.__table__,
            EvaluationCaseModel.__table__,
            EvaluationRunModel.__table__,
            EvaluationResultModel.__table__,
        ],
    )


def _running_run(*, tenant_id: str, owner_id: str | None, case_count: int = 2) -> dict:
    case_ids = [f"HAK180-{index:03d}" for index in range(1, case_count + 1)]
    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id=tenant_id,
            actor_id="evaluator",
        )
        dataset = repository.seed_manifest(load_dataset())
        run = repository.create_run(
            dataset_id=dataset["id"],
            kb_id="kb-hak180",
            name="recovery fixture",
            mode="retrieval_only",
            baseline_run_id=None,
            config_snapshot={"code_revision": "test"},
            case_ids=case_ids,
            execution_owner=owner_id,
        )
        return repository.start_run(run["id"], owner_id=owner_id)


def test_stale_running_run_is_interrupted_without_faking_completion(tmp_path) -> None:
    _configure(tmp_path)
    now = datetime.now(UTC)
    owner_id = "query-host:123:owner-a"
    run = _running_run(tenant_id="tenant-recovery", owner_id=owner_id)

    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id="tenant-recovery",
            actor_id="evaluator",
        )
        repository.record_result(
            run["id"],
            "HAK180-001",
            observation={"answer": "10 °C 到 32 °C", "latency_ms": 12.5},
            metrics={"metrics": {"retrieval": {"recall_at_k": {"10": 1.0}}}},
            passed=True,
            owner_id=owner_id,
        )
        session.execute(
            update(EvaluationRunModel)
            .where(EvaluationRunModel.id == run["id"])
            .values(updated_at=now - timedelta(minutes=5))
        )

    report = recover_stale_evaluation_runs(
        now=now,
        stale_after_seconds=60,
    )
    assert report == {"scanned": 1, "interrupted": 1, "raced": 0, "failed": 0}

    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id="tenant-recovery",
            actor_id="auditor",
        )
        persisted = repository.get_run(run["id"])
        assert persisted["status"] == "failed"
        assert persisted["completed_cases"] == 1
        assert persisted["total_cases"] == 2
        assert persisted["progress"] == 0.5
        assert persisted["error_code"] == EVALUATION_INTERRUPTED_CODE
        assert persisted["gate_result"]["status"] == "interrupted"
        assert persisted["gate_result"]["progress"] == {
            "completed_cases": 1,
            "total_cases": 2,
            "value": 0.5,
        }
        assert persisted["report_paths"] == {}
        assert [item["case_id"] for item in repository.list_results(run["id"])] == [
            "HAK180-001"
        ]
        with pytest.raises(EvaluationRunOwnershipLost):
            repository.record_result(
                run["id"],
                "HAK180-002",
                observation={"answer": "迟到结果"},
                metrics={},
                passed=True,
                owner_id=owner_id,
            )

    assert recover_stale_evaluation_runs(
        now=now,
        stale_after_seconds=60,
    )["scanned"] == 0
    dispose_database()


def test_heartbeat_protects_live_run_and_shutdown_interrupts_only_owner(tmp_path) -> None:
    _configure(tmp_path)
    now = datetime.now(UTC)
    owner_a = "query-host:123:owner-a"
    owner_b = "query-host:456:owner-b"
    run_a = _running_run(tenant_id="tenant-a", owner_id=owner_a, case_count=1)
    run_b = _running_run(tenant_id="tenant-b", owner_id=owner_b, case_count=1)

    with session_scope() as session:
        assert EvaluationRepository(
            session,
            tenant_id="tenant-a",
            actor_id="evaluator",
        ).heartbeat_run(run_a["id"], owner_id=owner_a)

    assert recover_stale_evaluation_runs(
        now=now + timedelta(seconds=30),
        stale_after_seconds=60,
    )["scanned"] == 0

    report = interrupt_owned_evaluation_runs(owner_a, now=now + timedelta(seconds=31))
    assert report == {"scanned": 1, "interrupted": 1, "raced": 0, "failed": 0}
    with session_scope() as session:
        assert EvaluationRepository(
            session,
            tenant_id="tenant-a",
            actor_id="auditor",
        ).get_run(run_a["id"])["status"] == "failed"
        assert EvaluationRepository(
            session,
            tenant_id="tenant-b",
            actor_id="auditor",
        ).get_run(run_b["id"])["status"] == "running"
    dispose_database()


def test_pre_heartbeat_running_run_falls_back_to_updated_at(tmp_path) -> None:
    """修复上线前的假 running 行没有 owner/heartbeat，也必须能被启动恢复。"""

    _configure(tmp_path)
    now = datetime.now(UTC)
    run = _running_run(tenant_id="tenant-legacy", owner_id=None, case_count=1)
    with session_scope() as session:
        session.execute(
            update(EvaluationRunModel)
            .where(EvaluationRunModel.id == run["id"])
            .values(
                gate_result={},
                updated_at=now - timedelta(minutes=5),
            )
        )

    report = recover_stale_evaluation_runs(now=now, stale_after_seconds=60)
    assert report["interrupted"] == 1
    with session_scope() as session:
        persisted = EvaluationRepository(
            session,
            tenant_id="tenant-legacy",
            actor_id="auditor",
        ).get_run(run["id"])
        assert persisted["status"] == "failed"
        assert persisted["error_code"] == EVALUATION_INTERRUPTED_CODE
        assert persisted["gate_result"]["execution"]["owner_id"] is None
    dispose_database()


def test_recovery_loop_keeps_scanning_until_shutdown(monkeypatch) -> None:
    calls = 0

    def fake_recover(**_kwargs):
        nonlocal calls
        calls += 1
        return {"scanned": 0, "interrupted": 0, "raced": 0, "failed": 0}

    monkeypatch.setattr(maintenance, "recover_stale_evaluation_runs", fake_recover)

    async def exercise() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_evaluation_recovery_loop(
                stop,
                interval_seconds=0.01,
                stale_after_seconds=1,
            )
        )
        while calls < 2:
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise())
    assert calls >= 2
