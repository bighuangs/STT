from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.domain.evaluation import (
    FORMAL_EVALUATION_SNAPSHOT_KEY,
    finalize_formal_evaluation_contract,
)
from scripts import run_formal_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path, *, formal: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "stt.eval.harness-config.v3",
                "config_id": "fixture-formal",
                "target_id": "query-graph-evaluation-target",
                "dataset_id": "fixture@1.0.0",
                "scenario_id": "live-query",
                "graders": ["retrieval", "generation", "abstention", "safety"],
                "failure_policy": "continue",
                "formal_evaluation": formal,
                "planning_admission_budgets": {
                    "scope": "per_case_query_plan_admission",
                    "max_plan_tokens": 4096,
                    "max_routes": 3,
                    "max_planned_cost_units": 1.0,
                },
                "usage_accounting": {
                    "provider_tokens": "unavailable",
                    "provider_cost_usd": "unavailable",
                    "retrieval_calls": "unavailable",
                    "enforcement": "not_enforced",
                },
                "allowed_intervention_dimensions": ["code_revision"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_live_execution_requires_explicit_paid_model_confirmation(tmp_path: Path) -> None:
    args = run_formal_evaluation._parse_args(
        [
            "--config",
            str(_config(tmp_path / "config.json")),
            "--kb-id",
            "kb_fixture",
            "--user-id",
            "user_fixture",
            "--execute",
        ]
    )
    config = run_formal_evaluation.load_harness_config(args.config)
    scenario = run_formal_evaluation.load_scenario(config.scenario_id)

    with pytest.raises(ValueError, match="allow-paid-model-calls"):
        run_formal_evaluation._validate_selection(args, config, scenario)


def test_profile_pin_requires_id_and_version_together(tmp_path: Path) -> None:
    args = run_formal_evaluation._parse_args(
        [
            "--config",
            str(_config(tmp_path / "config.json")),
            "--kb-id",
            "kb_fixture",
            "--user-id",
            "user_fixture",
            "--profile-id",
            "rpf_fixture",
        ]
    )
    config = run_formal_evaluation.load_harness_config(args.config)
    scenario = run_formal_evaluation.load_scenario(config.scenario_id)

    with pytest.raises(ValueError, match="provided together"):
        run_formal_evaluation._validate_selection(args, config, scenario)


def test_default_path_is_read_only_and_reports_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path / "config.json")
    asset = SimpleNamespace(
        dataset_id="fixture@1.0.0",
        sha256="a" * 64,
        manifest={"dataset_id": "fixture"},
    )
    monkeypatch.setattr(run_formal_evaluation, "load_dataset_asset", lambda *args, **kwargs: asset)
    monkeypatch.setattr(
        run_formal_evaluation,
        "_load_evaluation_service",
        lambda: pytest.fail("read-only preflight must not import the application service"),
    )

    assert (
        run_formal_evaluation.main(
            [
                "--config",
                str(config_path),
                "--project-root",
                str(tmp_path),
                "--kb-id",
                "kb_fixture",
                "--user-id",
                "user_fixture",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["dataset_sha256"] == "a" * 64
    assert len(payload["config_sha256"]) == 64
    assert payload["mode"] == "live"
    assert payload["scenario"]["observation_scope"] == "full"
    assert payload["planning_admission_budgets"]["max_routes"] == 3
    assert payload["usage_accounting"]["provider_cost_usd"] == "unavailable"
    assert payload["usage_accounting"]["enforcement"] == "not_enforced"
    assert payload["validation_scope"] == "config_scenario_and_dataset_asset_only"
    assert payload["will_write"] is False
    assert payload["paid_model_calls_allowed"] is False


def test_live_execution_uses_frozen_asset_and_durable_formal_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path / "config.json")
    asset = SimpleNamespace(
        dataset_id="fixture@1.0.0",
        sha256="b" * 64,
        manifest={"dataset_key": "fixture", "version": "1.0.0"},
    )
    calls: list[tuple[str, object]] = []
    durable_run: dict = {}

    class FakeService:
        @staticmethod
        def import_dataset(manifest):
            calls.append(("import", manifest))
            return {
                "id": "eds_fixture",
                "dataset_key": "fixture",
                "version": "1.0.0",
                "checksum": "c" * 64,
            }

        @staticmethod
        def evaluation_preflight(dataset_id, kb_id, **kwargs):
            del dataset_id, kb_id, kwargs
            pytest.fail("execute CLI must not perform a pre-create preflight")

        @staticmethod
        def create_run(**kwargs):
            calls.append(("create", kwargs))
            calls.append(("internal_preflight", kwargs["dataset_id"]))
            intent = kwargs["formal_evaluation_contract"]
            scopes = [
                {
                    "tenant_id": "tenant_default",
                    "kb_id": "kb_fixture",
                    "document_id": "doc_fixture",
                    "document_version_id": "dver_fixture",
                    "index_version_id": "idx_fixture",
                    "collection_name": "kb_chunks",
                    "partition_name": None,
                }
            ]
            snapshot = {
                "code_revision": "test-revision",
                "config_version": "cfg-test",
                "dataset_version": "1.0.0",
                "dataset_checksum": "c" * 64,
                "source_sha256": "b" * 64,
                "evaluation_mode": "live",
                "embedding_model": "bge-m3",
                "reranker_model": "bge-reranker",
                "retrieval_profile": {
                    "profile_id": "rpf_fixture",
                    "version": 3,
                    "config_hash": "d" * 64,
                    "config": {"query_planning": {}},
                },
                "vector_read_scopes": scopes,
                "vector_read_set_digest": "e" * 64,
                "evaluation_security": {"scope_authorized": True},
                "llm_model": "qwen-max",
                "gate_policy": {"schema_version": "gate.v1"},
                "preflight": {"ready": True, "checks": {"all": True}},
                "runtime_flags": {
                    "query_default_deadline_ms": 120000,
                    "llm_request_timeout_seconds": 60.0,
                    "llm_max_attempts": 3,
                    "llm_retry_base_delay_seconds": 0.6,
                    "llm_retry_max_delay_seconds": 8.0,
                },
                "telemetry_coverage": {
                    "tokens": "unavailable",
                    "cost": "unavailable",
                },
                FORMAL_EVALUATION_SNAPSHOT_KEY: intent,
                "selected_case_ids": ["CASE-001", "CASE-002"],
            }
            snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY] = finalize_formal_evaluation_contract(
                intent,
                config_snapshot=snapshot,
                run_id="erun_fixture",
                tenant_id="tenant_default",
                kb_id="kb_fixture",
                baseline_run_id=None,
                baseline_formal_contract_sha256=None,
                selected_case_ids=("CASE-001", "CASE-002"),
            )
            durable_run.update(
                {
                    "id": "erun_fixture",
                    "dataset_id": "eds_fixture",
                    "dataset_key": "fixture",
                    "dataset_version": "1.0.0",
                    "dataset_checksum": "c" * 64,
                    "kb_id": "kb_fixture",
                    "baseline_run_id": None,
                    "mode": "live",
                    "total_cases": 2,
                    "config_snapshot": snapshot,
                }
            )
            return dict(durable_run)

        @staticmethod
        def execute_run(run_id, principal):
            calls.append(("execute", run_id))

        @staticmethod
        def get_run(run_id):
            return {
                **durable_run,
                "id": run_id,
                "status": "succeeded",
                "metrics_summary": {"case_count": 2},
                "gate_result": {"status": "passed", "passed": True},
            }

    monkeypatch.setattr(run_formal_evaluation, "load_dataset_asset", lambda *args, **kwargs: asset)
    monkeypatch.setattr(run_formal_evaluation, "_load_evaluation_service", lambda: FakeService)

    assert (
        run_formal_evaluation.main(
            [
                "--config",
                str(config_path),
                "--project-root",
                str(tmp_path),
                "--kb-id",
                "kb_fixture",
                "--user-id",
                "user_fixture",
                "--profile-id",
                "rpf_fixture",
                "--profile-version",
                "3",
                "--case-id",
                "CASE-001",
                "--case-id",
                "CASE-002",
                "--execute",
                "--allow-paid-model-calls",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "erun_fixture"
    assert payload["completed_cases"] == 2
    assert payload["gate_passed"] is True
    create_call = next(value for name, value in calls if name == "create")
    assert create_call["retrieval_profile_id"] == "rpf_fixture"
    assert create_call["retrieval_profile_version"] == 3
    assert create_call["case_ids"] == ["CASE-001", "CASE-002"]
    assert create_call["mode"] == "live"
    assert create_call["formal_evaluation_contract"]["config_id"] == "fixture-formal"
    assert create_call["formal_evaluation_contract"]["planning_admission_budgets"] == {
        "scope": "per_case_query_plan_admission",
        "max_plan_tokens": 4096,
        "max_routes": 3,
        "max_planned_cost_units": 1.0,
    }
    assert [name for name, _ in calls].count("internal_preflight") == 1
    assert payload["usage_accounting"]["provider_tokens"] == "unavailable"
    assert payload["selected_cases"]["selected_case_count"] == 2
    assert payload["enforced_execution_controls"]["provider_max_attempts_per_request"] == 3
    assert payload["preflight"]["ready"] is True


def test_direct_cli_path_resolves_repository_packages() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_formal_evaluation.py"),
            "--config",
            str(PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_3_authority_v3.json"),
            "--project-root",
            str(PROJECT_ROOT),
            "--kb-id",
            "kb_fixture",
            "--user-id",
            "user_fixture",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "validated"
    assert payload["dataset_id"] == "hak180-safety@1.3.0"
    assert payload["will_write"] is False
