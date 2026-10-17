from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.domain.evaluation import (
    FORMAL_EVALUATION_SNAPSHOT_KEY,
    build_formal_evaluation_intent,
    build_legacy_formal_evaluation_contract,
    content_digest,
    finalize_formal_evaluation_contract,
    formal_comparison_snapshot_projection,
    formal_runtime_snapshot_projection,
    freeze_formal_evaluation_contract,
    freeze_formal_evaluation_intent,
)
from app.evaluation.dataset_loader import load_dataset
from app.infra.persistence.sql.evaluation_repository import EvaluationRepository
from app.infra.persistence.sql.models import (
    Base,
    EvaluationCaseModel,
    EvaluationDatasetModel,
    EvaluationResultModel,
    EvaluationRunModel,
)
from app.infra.persistence.sql.repositories import ResourceConflictError
from app.infrastructure.evaluation.query_targets import (
    _apply_formal_planning_admission_budgets,
)
from evals.configs import load_harness_config
from evals.scenarios import load_scenario
from scripts import run_formal_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _planning_admission_budgets() -> dict:
    return {
        "scope": "per_case_query_plan_admission",
        "max_plan_tokens": 4096,
        "max_routes": 3,
        "max_planned_cost_units": 1.0,
    }


def _usage_accounting() -> dict:
    return {
        "provider_tokens": "unavailable",
        "provider_cost_usd": "unavailable",
        "retrieval_calls": "unavailable",
        "enforcement": "not_enforced",
    }


def _config_payload(*, schema_version: str = "stt.eval.harness-config.v3") -> dict:
    payload = {
        "schema_version": schema_version,
        "config_id": "fixture-formal-v3",
        "target_id": "query-graph-evaluation-target",
        "dataset_id": "hak180-safety@1.0.0",
        "scenario_id": "live-query",
        "graders": ["retrieval", "generation", "abstention", "safety"],
        "failure_policy": "continue",
        "formal_evaluation": True,
    }
    if schema_version == "stt.eval.harness-config.v2":
        payload["budgets"] = {
            "scope": "per_case_query_plan",
            "token_budget": 4096,
            "tool_budget": 3,
            "cost_budget": 1.0,
            "cost_currency": "USD",
        }
    elif schema_version == "stt.eval.harness-config.v3":
        payload["planning_admission_budgets"] = _planning_admission_budgets()
        payload["usage_accounting"] = _usage_accounting()
        payload["allowed_intervention_dimensions"] = ["code_revision"]
    return payload


def _write_config(path: Path, payload: dict, *, indent: int | None = None) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return path


def _scenario() -> dict:
    value = {
        "scenario_id": "live-query",
        "observation_scope": "full",
        "allowed_side_effects": ["vector_read", "external_llm"],
        "external_model_allowed": True,
        "formal_evaluation": True,
    }
    return {**value, "scenario_sha256": content_digest(value)}


def _intent(
    *,
    record_id: str,
    record_checksum: str,
    config_id: str = "fixture-formal-v3",
    allowed_interventions: tuple[str, ...] = ("code_revision",),
) -> dict:
    config_payload = _config_payload()
    config_payload["config_id"] = config_id
    config_payload["allowed_intervention_dimensions"] = list(allowed_interventions)
    return build_formal_evaluation_intent(
        config_schema_version="stt.eval.harness-config.v3",
        config_id=config_id,
        config_sha256=content_digest(config_payload),
        target_id="query-graph-evaluation-target",
        dataset={
            "registry_id": "hak180-safety@1.0.0",
            "asset_sha256": "2" * 64,
            "record_id": record_id,
            "record_checksum": record_checksum,
            "dataset_key": "hak180-safety",
            "dataset_version": "1.0.0",
        },
        scenario=_scenario(),
        graders=["retrieval", "generation", "abstention", "safety"],
        failure_policy="continue",
        execution_mode="live",
        planning_admission_budgets=_planning_admission_budgets(),
        usage_accounting=_usage_accounting(),
        allowed_intervention_dimensions=allowed_interventions,
    )


def _runtime_snapshot(
    intent: dict,
    *,
    code_revision: str | None = None,
    profile_hash: str = "3" * 64,
) -> dict:
    vector_scopes = [
        {
            "tenant_id": "tenant-formal",
            "kb_id": "kb-formal",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "collection_name": "kb_chunks",
            "partition_name": None,
        }
    ]
    return {
        "code_revision": code_revision or "head:test;source-sha256:" + "1" * 64,
        "answering_contract": {
            "schema_version": "answering-output-contract.v1",
            "generation_output_token_cap": 2048,
        },
        "config_version": "cfg-test",
        "dataset_version": "1.0.0",
        "dataset_checksum": intent["dataset"]["record_checksum"],
        "source_sha256": intent["dataset"]["asset_sha256"],
        "evaluation_mode": "live",
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "retrieval_profile": {
            "profile_id": "rpf-formal",
            "version": 3,
            "config_hash": profile_hash,
            "config": {
                "timeout_ms": 8000,
                "query_planning": {
                    "token_budget": 8192,
                    "max_routes": 6,
                    "cost_budget": 2.0,
                },
                "context_pack": {"reserved_output_tokens": 1024},
            },
        },
        "vector_read_scopes": vector_scopes,
        "vector_read_set_digest": "4" * 64,
        "evaluation_security": {
            "scope_authorized": True,
            "external_llm_allowed": True,
            "authorized_scope": {"allowed_kb_ids": ["kb-formal"]},
        },
        "llm_model": "qwen-max",
        "gate_policy": {"schema_version": "evaluation-gate-policy.v1"},
        "preflight": {
            "ready": True,
            "checks": {"all": True},
            "retrieval_profile_selection": {
                "profile_id": "rpf-formal",
                "version": 3,
                "config_hash": profile_hash,
            },
        },
        "runtime_flags": {
            "query_default_deadline_ms": 120000,
            "llm_request_timeout_seconds": 60.0,
            "llm_max_attempts": 3,
            "llm_retry_base_delay_seconds": 0.6,
            "llm_retry_max_delay_seconds": 8.0,
            "reranker_failure_mode": "fail_closed",
            "vector_placement_mode": "shared_compatible",
            "vector_placement_policy_version": "placement.v1",
        },
        "telemetry_coverage": {"tokens": "unavailable", "cost": "unavailable"},
        FORMAL_EVALUATION_SNAPSHOT_KEY: intent,
    }


def _final_contract(
    intent: dict,
    snapshot: dict,
    *,
    run_id: str = "erun-formal",
    baseline_run_id: str | None = None,
    baseline_sha256: str | None = None,
    selected_case_ids: tuple[str, ...] = ("HAK180-001",),
) -> dict:
    return finalize_formal_evaluation_contract(
        intent,
        config_snapshot=snapshot,
        run_id=run_id,
        tenant_id="tenant-formal",
        kb_id="kb-formal",
        baseline_run_id=baseline_run_id,
        baseline_formal_contract_sha256=baseline_sha256,
        selected_case_ids=selected_case_ids,
    )


def _align_with_current_query_runtime(snapshot: dict) -> None:
    from app.shared.config.settings_config import settings

    snapshot["embedding_model"] = settings.bge_m3_path or settings.bge_m3
    snapshot["reranker_model"] = settings.bge_reranker_large
    snapshot["llm_model"] = settings.llm_default_model
    snapshot["runtime_flags"] = {
        "query_default_deadline_ms": settings.query_default_deadline_ms,
        "llm_request_timeout_seconds": settings.llm_request_timeout,
        "llm_max_attempts": settings.llm_max_attempts,
        "llm_retry_base_delay_seconds": settings.llm_retry_base_delay,
        "llm_retry_max_delay_seconds": settings.llm_retry_max_delay,
        "reranker_failure_mode": settings.bge_reranker_failure_mode,
        "vector_placement_mode": settings.vector_placement_mode,
        "vector_placement_policy_version": settings.vector_placement_policy_version,
    }


def _evaluation_engine(path: Path):
    engine = create_engine(f"sqlite:///{path.as_posix()}", future=True)
    Base.metadata.create_all(
        engine,
        tables=[
            EvaluationDatasetModel.__table__,
            EvaluationCaseModel.__table__,
            EvaluationRunModel.__table__,
            EvaluationResultModel.__table__,
        ],
    )
    return engine


def _formal_repository_run(
    repository: EvaluationRepository,
    dataset: dict,
    *,
    baseline_run_id: str | None = None,
    config_id: str = "fixture-formal-v3",
    allowed_interventions: tuple[str, ...] = ("code_revision",),
    code_revision: str | None = None,
    profile_hash: str = "3" * 64,
) -> dict:
    intent = _intent(
        record_id=dataset["id"],
        record_checksum=dataset["checksum"],
        config_id=config_id,
        allowed_interventions=allowed_interventions,
    )
    return repository.create_run(
        dataset_id=dataset["id"],
        kb_id="kb-formal",
        name="formal authority",
        mode="live",
        baseline_run_id=baseline_run_id,
        config_snapshot=_runtime_snapshot(
            intent,
            code_revision=code_revision,
            profile_hash=profile_hash,
        ),
        case_ids=["HAK180-001"],
    )


def test_v3_config_hash_is_canonical_and_names_planning_admission_limits(
    tmp_path: Path,
) -> None:
    payload = _config_payload()
    first = load_harness_config(_write_config(tmp_path / "first.json", payload, indent=2))
    second = load_harness_config(
        _write_config(
            tmp_path / "second.json",
            dict(reversed(list(payload.items()))),
            indent=None,
        )
    )

    assert first.schema_version == "stt.eval.harness-config.v3"
    assert first.canonical_sha256 == second.canonical_sha256 == content_digest(payload)
    assert first.budgets is None
    assert first.planning_admission_budgets.as_dict() == _planning_admission_budgets()
    assert first.usage_accounting.as_dict() == _usage_accounting()


@pytest.mark.parametrize(
    ("filename", "canonical_sha256"),
    [
        (
            "hak180_online_v1_1.json",
            "0715dcc91d9ae95cd79399626c70e2222940ab1b973a4e7885492059a2a368ee",
        ),
        (
            "hak180_online_v1_2.json",
            "3e1be6df7fe522d2e43aa4ea2e094990503d80b39cbf96861dc755a7e30f3718",
        ),
        (
            "hak180_online_v1_3.json",
            "f60cc14b12ffaa6a9d96c9eacf33bf29ba50ebf7725adf3b9b6a41140873d188",
        ),
    ],
)
def test_published_v2_config_assets_keep_their_historical_hash(
    filename: str,
    canonical_sha256: str,
) -> None:
    config = load_harness_config(PROJECT_ROOT / "evals" / "configs" / filename)

    assert config.schema_version == "stt.eval.harness-config.v2"
    assert config.canonical_sha256 == canonical_sha256


def test_v3_config_rejects_tool_or_actual_cost_budget_claims(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["planning_admission_budgets"]["tool_budget"] = 3

    with pytest.raises(ValueError, match="unknown tool_budget"):
        load_harness_config(_write_config(tmp_path / "ambiguous.json", payload))


def test_config_loader_rejects_duplicate_keys_before_hashing(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"stt.eval.harness-config.v3","config_id":"first","config_id":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate field: config_id"):
        load_harness_config(path)


@pytest.mark.parametrize(
    "schema_version",
    ["stt.eval.harness-config.v1", "stt.eval.harness-config.v2"],
)
def test_legacy_v1_v2_configs_load_but_formal_execution_fails_closed(
    tmp_path: Path,
    schema_version: str,
) -> None:
    path = _write_config(
        tmp_path / "legacy.json",
        _config_payload(schema_version=schema_version),
    )
    config = load_harness_config(path)
    scenario = load_scenario(config.scenario_id)
    args = run_formal_evaluation._parse_args(
        ["--config", str(path), "--kb-id", "kb_fixture", "--user-id", "user_fixture"]
    )

    assert config.schema_version == schema_version
    with pytest.raises(ValueError, match="v3 authority"):
        run_formal_evaluation._validate_selection(args, config, scenario)


def test_scenario_is_the_mode_authority_and_rejects_cli_drift(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "formal.json", _config_payload())
    config = load_harness_config(path)
    scenario = load_scenario(config.scenario_id)
    args = run_formal_evaluation._parse_args(
        [
            "--config",
            str(path),
            "--kb-id",
            "kb_fixture",
            "--user-id",
            "user_fixture",
            "--mode",
            "retrieval_only",
        ]
    )

    with pytest.raises(ValueError, match="conflicts with scenario"):
        run_formal_evaluation._validate_selection(args, config, scenario)


def test_formal_intent_is_self_hashing_and_rejects_semantic_tamper() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    assert freeze_formal_evaluation_intent(intent) == intent

    tampered = copy.deepcopy(intent)
    tampered["planning_admission_budgets"]["max_routes"] = 4
    with pytest.raises(ValueError, match="config sha256"):
        freeze_formal_evaluation_intent(tampered)


def test_formal_run_v2_binds_actual_runtime_authority() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    contract = _final_contract(intent, snapshot)

    authority = contract["run_authority"]
    assert authority["run_id"] == "erun-formal"
    assert authority["tenant_id"] == "tenant-formal"
    assert authority["kb_id"] == "kb-formal"
    assert authority["baseline"] == {
        "baseline_run_id": None,
        "baseline_formal_contract_sha256": None,
    }
    assert authority["selected_cases"]["selected_case_count"] == 1
    assert authority["retrieval_profile"] == {
        "profile_id": "rpf-formal",
        "version": 3,
        "config_hash": "3" * 64,
    }
    assert authority["vector_read_set"]["scopes"] == snapshot["vector_read_scopes"]
    assert authority["evaluation_security"] == snapshot["evaluation_security"]
    assert authority["runtime_flags"]["query_default_deadline_ms"] == 120000
    assert authority["model_ids"]["llm"] == "qwen-max"
    assert authority["enforced_execution_controls"] == {
        "query_plan_route_timeout_ms": 8000,
        "generation_output_token_cap": 2048,
        "provider_request_timeout_seconds": 60.0,
        "provider_max_attempts_per_request": 3,
        "provider_retry_base_delay_seconds": 0.6,
        "provider_retry_max_delay_seconds": 8.0,
    }


def test_formal_run_v2_preserves_historical_profile_derived_output_cap() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    snapshot.pop("answering_contract")

    contract = _final_contract(intent, snapshot)

    assert (
        contract["run_authority"]["enforced_execution_controls"]["generation_output_token_cap"]
        == 1024
    )


@pytest.mark.parametrize("qualified", [False, True])
def test_formal_run_v2_canonicalizes_vector_read_set_digest(qualified: bool) -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    digest = "4" * 64
    snapshot["vector_read_set_digest"] = f"sha256:{digest}" if qualified else digest
    snapshot["preflight"]["vector_read_set_digest"] = snapshot["vector_read_set_digest"]

    contract = _final_contract(intent, snapshot)

    authority = contract["run_authority"]
    assert authority["vector_read_set"]["digest"] == digest
    assert snapshot["vector_read_set_digest"] == (f"sha256:{digest}" if qualified else digest)
    assert snapshot["preflight"]["vector_read_set_digest"] == snapshot["vector_read_set_digest"]
    assert authority["preflight_sha256"] == content_digest(snapshot["preflight"])
    assert authority["runtime_snapshot_sha256"] == content_digest(
        formal_runtime_snapshot_projection(snapshot)
    )
    assert authority["comparison_snapshot_sha256"] == content_digest(
        formal_comparison_snapshot_projection(
            formal_runtime_snapshot_projection(snapshot),
            intent["allowed_intervention_dimensions"],
        )
    )
    assert "answering_contract" not in formal_comparison_snapshot_projection(
        formal_runtime_snapshot_projection(snapshot),
        intent["allowed_intervention_dimensions"],
    )
    durable_snapshot = {
        **snapshot,
        FORMAL_EVALUATION_SNAPSHOT_KEY: contract,
        "selected_case_ids": ["HAK180-001"],
    }
    assert (
        freeze_formal_evaluation_contract(
            contract,
            expected_runtime_snapshot=durable_snapshot,
        )["contract_sha256"]
        == contract["contract_sha256"]
    )


def test_vector_digest_normalization_does_not_rewrite_snapshot_hash_inputs() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    digest = "4" * 64
    bare_snapshot = _runtime_snapshot(intent)
    tagged_snapshot = _runtime_snapshot(intent)
    for snapshot, value in (
        (bare_snapshot, digest),
        (tagged_snapshot, f"sha256:{digest}"),
    ):
        snapshot["vector_read_set_digest"] = value
        snapshot["preflight"]["vector_read_set_digest"] = value

    bare_authority = _final_contract(intent, bare_snapshot)["run_authority"]
    tagged_authority = _final_contract(intent, tagged_snapshot)["run_authority"]

    assert (
        bare_authority["vector_read_set"]["digest"] == tagged_authority["vector_read_set"]["digest"]
    )
    for field in (
        "preflight_sha256",
        "runtime_snapshot_sha256",
        "comparison_snapshot_sha256",
    ):
        assert bare_authority[field] != tagged_authority[field]


@pytest.mark.parametrize(
    "digest",
    [
        "SHA256:" + "4" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "4" * 63,
        "sha512:" + "4" * 64,
        "sha256:" + "G" * 64,
        " sha256:" + "4" * 64,
        "sha256:" + "4" * 64 + " ",
        "sha256:sha256:" + "4" * 64,
        None,
        64,
    ],
)
def test_formal_run_v2_rejects_noncanonical_vector_read_set_digest(digest: object) -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    snapshot["vector_read_set_digest"] = digest

    with pytest.raises(ValueError, match="vector_read_set.digest"):
        _final_contract(intent, snapshot)


def test_contract_expectation_distinguishes_unset_baseline_from_expected_none() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    contract = _final_contract(intent, _runtime_snapshot(intent))
    tampered = copy.deepcopy(contract)
    tampered["run_authority"]["baseline"] = {
        "baseline_run_id": "erun-other",
        "baseline_formal_contract_sha256": "9" * 64,
    }
    tampered["contract_sha256"] = content_digest(
        {key: value for key, value in tampered.items() if key != "contract_sha256"}
    )

    with pytest.raises(ValueError, match="baseline run id"):
        freeze_formal_evaluation_contract(
            tampered,
            expected_baseline_run_id=None,
            expected_baseline_formal_contract_sha256=None,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("code_revision", "head:tampered"),
        ("config_version", "cfg-tampered"),
        ("llm_model", "different-model"),
        ("vector_read_set_digest", "9" * 64),
        ("runtime_flags", {"query_default_deadline_ms": 1}),
        ("evaluation_security", {"scope_authorized": False}),
    ],
)
def test_formal_contract_rejects_runtime_snapshot_tamper(
    field: str,
    replacement: object,
) -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    contract = _final_contract(intent, snapshot)
    durable = {
        **snapshot,
        FORMAL_EVALUATION_SNAPSHOT_KEY: contract,
        "selected_case_ids": ["HAK180-001"],
    }
    durable[field] = replacement

    with pytest.raises(ValueError, match="runtime authority"):
        freeze_formal_evaluation_contract(
            contract,
            expected_run_id="erun-formal",
            expected_kb_id="kb-formal",
            expected_selected_case_ids=["HAK180-001"],
            expected_runtime_snapshot=durable,
        )


def test_planning_admission_projection_does_not_claim_actual_usage_caps() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    _align_with_current_query_runtime(snapshot)
    contract = _final_contract(intent, snapshot)
    snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY] = contract
    snapshot["selected_case_ids"] = ["HAK180-001"]
    profile = {
        "query_planning": {
            "max_routes": 8,
            "token_budget": 32768,
            "cost_budget": 12.0,
            "allow_web": False,
        },
        "top_k": 12,
    }

    bounded = _apply_formal_planning_admission_budgets(
        profile,
        snapshot,
        expected_mode="live",
        expected_run_id="erun-formal",
        expected_kb_id="kb-formal",
    )

    assert bounded["query_planning"] == {
        "max_routes": 3,
        "token_budget": 4096,
        "cost_budget": 1.0,
        "allow_web": False,
    }
    assert intent["usage_accounting"] == _usage_accounting()
    assert "scenario" not in bounded
    assert "graders" not in bounded


def test_adapter_rejects_runtime_flag_drift_after_acceptance() -> None:
    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    snapshot = _runtime_snapshot(intent)
    contract = _final_contract(intent, snapshot)
    snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY] = contract
    snapshot["selected_case_ids"] = ["HAK180-001"]

    with pytest.raises(ValueError, match="runtime flags drifted"):
        _apply_formal_planning_admission_budgets(
            {"query_planning": {}},
            snapshot,
            expected_mode="live",
            expected_run_id="erun-formal",
            expected_kb_id="kb-formal",
        )


def test_application_create_run_performs_one_authoritative_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.application import evaluation_service

    intent = _intent(record_id="eds_fixture", record_checksum="5" * 64)
    preflight_calls: list[tuple] = []
    captured: dict = {}
    dataset = {
        "id": "eds_fixture",
        "dataset_key": "hak180-safety",
        "version": "1.0.0",
        "checksum": "5" * 64,
        "source": {"content_sha256": "2" * 64},
        "manifest_json": {},
    }

    class FakeRepository:
        def __init__(self, session):
            del session

        def get_dataset(self, dataset_id: str) -> dict:
            assert dataset_id == "eds_fixture"
            return dataset

        def create_run(self, **kwargs) -> dict:
            captured.update(kwargs)
            return {"id": "erun_fixture", "config_snapshot": kwargs["config_snapshot"]}

    @contextmanager
    def fake_session_scope():
        yield object()

    def fake_preflight(*args, **kwargs):
        preflight_calls.append((args, kwargs))
        return {
            "ready": True,
            "checks": {"all": True},
            "retrieval_profile": _runtime_snapshot(intent)["retrieval_profile"],
            "vector_read_scopes": _runtime_snapshot(intent)["vector_read_scopes"],
            "vector_read_set_digest": "4" * 64,
            "evaluation_security": _runtime_snapshot(intent)["evaluation_security"],
        }

    monkeypatch.setattr(evaluation_service, "evaluation_preflight", fake_preflight)
    monkeypatch.setattr(evaluation_service, "session_scope", fake_session_scope)
    monkeypatch.setattr(evaluation_service, "EvaluationRepository", FakeRepository)
    monkeypatch.setattr(evaluation_service, "_code_revision", lambda: "test-revision")
    monkeypatch.setattr(evaluation_service, "evaluation_execution_owner", lambda: "owner")

    evaluation_service.create_run(
        dataset_id="eds_fixture",
        kb_id="kb-formal",
        mode="live",
        name="formal handoff",
        baseline_run_id=None,
        case_ids=["HAK180-001"],
        formal_evaluation_contract=intent,
    )

    assert len(preflight_calls) == 1
    assert captured["config_snapshot"][FORMAL_EVALUATION_SNAPSHOT_KEY] == intent
    assert captured["config_snapshot"]["answering_contract"] == {
        "schema_version": "answering-output-contract.v1",
        "generation_output_token_cap": evaluation_service.MAX_PROPOSAL_BYTES,
    }
    assert captured["config_snapshot"]["runtime_flags"]["query_default_deadline_ms"] > 0


def test_repository_finalizes_intent_and_rejects_durable_runtime_tamper(
    tmp_path: Path,
) -> None:
    engine = _evaluation_engine(tmp_path / "formal-authority-v2.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            run = _formal_repository_run(repository, dataset)
            session.commit()
            run_id = run["id"]
            contract = run["config_snapshot"][FORMAL_EVALUATION_SNAPSHOT_KEY]
            assert contract["contract_version"] == "stt.eval.formal-run.v2"
            assert contract["run_authority"]["run_id"] == run_id
            assert repository.list_runs()[0]["id"] == run_id

            tampered_snapshot = copy.deepcopy(run["config_snapshot"])
            tampered_snapshot["runtime_flags"]["query_default_deadline_ms"] = 1
            session.execute(
                text("UPDATE evaluation_runs SET config_snapshot = :snapshot WHERE id = :run_id"),
                {
                    "snapshot": json.dumps(tampered_snapshot, ensure_ascii=False),
                    "run_id": run_id,
                },
            )
            session.commit()

        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="auditor",
            )
            with pytest.raises(ResourceConflictError, match="formal evaluation contract"):
                repository.get_run(run_id)
    finally:
        engine.dispose()


def test_repository_precreate_accepts_qualified_vector_digest_without_rewriting_snapshot(
    tmp_path: Path,
) -> None:
    engine = _evaluation_engine(tmp_path / "formal-qualified-vector-digest.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            intent = _intent(
                record_id=dataset["id"],
                record_checksum=dataset["checksum"],
            )
            snapshot = _runtime_snapshot(intent)
            tagged_digest = "sha256:" + "4" * 64
            snapshot["vector_read_set_digest"] = tagged_digest
            snapshot["preflight"]["vector_read_set_digest"] = tagged_digest

            run = repository.create_run(
                dataset_id=dataset["id"],
                kb_id="kb-formal",
                name="formal qualified vector digest",
                mode="live",
                baseline_run_id=None,
                config_snapshot=snapshot,
                case_ids=["HAK180-001"],
            )
            session.commit()

            durable_snapshot = run["config_snapshot"]
            authority = durable_snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY]["run_authority"]
            assert durable_snapshot["vector_read_set_digest"] == tagged_digest
            assert durable_snapshot["preflight"]["vector_read_set_digest"] == tagged_digest
            assert authority["vector_read_set"]["digest"] == "4" * 64
            assert session.scalar(text("SELECT COUNT(*) FROM evaluation_runs")) == 1
    finally:
        engine.dispose()


@pytest.mark.parametrize("tamper", ["none", "wrong_sha"])
def test_repository_rejects_baseline_none_or_sha_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    engine = _evaluation_engine(tmp_path / f"formal-baseline-{tamper}.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(repository, dataset)
            candidate = _formal_repository_run(
                repository,
                dataset,
                baseline_run_id=baseline["id"],
            )
            session.commit()
            candidate_id = candidate["id"]
            snapshot = copy.deepcopy(candidate["config_snapshot"])
            contract = snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY]
            if tamper == "none":
                contract["run_authority"]["baseline"] = {
                    "baseline_run_id": None,
                    "baseline_formal_contract_sha256": None,
                }
            else:
                contract["run_authority"]["baseline"]["baseline_formal_contract_sha256"] = "9" * 64
            contract["contract_sha256"] = content_digest(
                {key: value for key, value in contract.items() if key != "contract_sha256"}
            )
            session.execute(
                text("UPDATE evaluation_runs SET config_snapshot = :snapshot WHERE id = :run_id"),
                {
                    "snapshot": json.dumps(snapshot, ensure_ascii=False),
                    "run_id": candidate_id,
                },
            )
            session.commit()

        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="auditor",
            )
            with pytest.raises(ResourceConflictError, match="formal evaluation contract"):
                repository.get_run(candidate_id)
    finally:
        engine.dispose()


def test_candidate_creation_rejects_a_rehashed_tampered_baseline(tmp_path: Path) -> None:
    engine = _evaluation_engine(tmp_path / "formal-tampered-baseline-create.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(repository, dataset)
            snapshot = copy.deepcopy(baseline["config_snapshot"])
            contract = snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY]
            contract["run_authority"]["code_revision"] = "rehashed-tamper"
            contract["contract_sha256"] = content_digest(
                {key: value for key, value in contract.items() if key != "contract_sha256"}
            )
            session.execute(
                text("UPDATE evaluation_runs SET config_snapshot = :snapshot WHERE id = :run_id"),
                {
                    "snapshot": json.dumps(snapshot, ensure_ascii=False),
                    "run_id": baseline["id"],
                },
            )
            session.commit()

            with pytest.raises(ResourceConflictError, match="formal evaluation contract"):
                _formal_repository_run(
                    repository,
                    dataset,
                    baseline_run_id=baseline["id"],
                    code_revision="revision-b",
                )
    finally:
        engine.dispose()


def test_formal_candidate_rejects_different_intent(tmp_path: Path) -> None:
    engine = _evaluation_engine(tmp_path / "formal-baseline-intent.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(repository, dataset)

            with pytest.raises(ResourceConflictError, match="same formal evaluation intent"):
                _formal_repository_run(
                    repository,
                    dataset,
                    baseline_run_id=baseline["id"],
                    config_id="fixture-formal-v3-candidate",
                )
    finally:
        engine.dispose()


def test_paired_run_rejects_undeclared_code_intervention(tmp_path: Path) -> None:
    engine = _evaluation_engine(tmp_path / "formal-undeclared-code.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(
                repository,
                dataset,
                allowed_interventions=(),
                code_revision="revision-a",
            )

            with pytest.raises(ResourceConflictError, match="undeclared runtime intervention"):
                _formal_repository_run(
                    repository,
                    dataset,
                    baseline_run_id=baseline["id"],
                    allowed_interventions=(),
                    code_revision="revision-b",
                )
    finally:
        engine.dispose()


def test_paired_run_allows_only_the_declared_code_intervention(tmp_path: Path) -> None:
    engine = _evaluation_engine(tmp_path / "formal-declared-code.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(
                repository,
                dataset,
                code_revision="revision-a",
            )
            candidate = _formal_repository_run(
                repository,
                dataset,
                baseline_run_id=baseline["id"],
                code_revision="revision-b",
            )

            assert candidate["baseline_run_id"] == baseline["id"]
            assert (
                candidate["config_snapshot"][FORMAL_EVALUATION_SNAPSHOT_KEY]["run_authority"][
                    "code_revision"
                ]
                == "revision-b"
            )
    finally:
        engine.dispose()


def test_paired_run_rejects_change_outside_declared_dimension(tmp_path: Path) -> None:
    engine = _evaluation_engine(tmp_path / "formal-undeclared-profile.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            baseline = _formal_repository_run(repository, dataset)

            with pytest.raises(ResourceConflictError, match="undeclared runtime intervention"):
                _formal_repository_run(
                    repository,
                    dataset,
                    baseline_run_id=baseline["id"],
                    profile_hash="8" * 64,
                )
    finally:
        engine.dispose()


def test_historical_v1_contract_remains_readable_but_cannot_create_a_new_run(
    tmp_path: Path,
) -> None:
    engine = _evaluation_engine(tmp_path / "formal-legacy.db")
    try:
        with Session(engine, expire_on_commit=False) as session:
            repository = EvaluationRepository(
                session,
                tenant_id="tenant-formal",
                actor_id="formal-cli",
            )
            dataset = repository.seed_manifest(load_dataset())
            legacy_config = {
                **_config_payload(schema_version="stt.eval.harness-config.v2"),
                "config_id": "legacy-formal-v2",
            }
            legacy = build_legacy_formal_evaluation_contract(
                config_schema_version="stt.eval.harness-config.v2",
                config_id="legacy-formal-v2",
                config_sha256=content_digest(legacy_config),
                target_id="query-graph-evaluation-target",
                dataset={
                    "registry_id": "hak180-safety@1.0.0",
                    "asset_sha256": "2" * 64,
                    "record_id": dataset["id"],
                    "record_checksum": dataset["checksum"],
                    "dataset_key": dataset["dataset_key"],
                    "dataset_version": dataset["version"],
                },
                scenario=_scenario(),
                graders=["retrieval", "generation", "abstention", "safety"],
                failure_policy="continue",
                execution_mode="live",
                budgets=legacy_config["budgets"],
            )
            with pytest.raises(ResourceConflictError, match="server-finalized intent"):
                repository.create_run(
                    dataset_id=dataset["id"],
                    kb_id="kb-formal",
                    name="new legacy run",
                    mode="live",
                    baseline_run_id=None,
                    config_snapshot={FORMAL_EVALUATION_SNAPSHOT_KEY: legacy},
                    case_ids=["HAK180-001"],
                )

            historical = repository.create_run(
                dataset_id=dataset["id"],
                kb_id="kb-formal",
                name="historical shell",
                mode="live",
                baseline_run_id=None,
                config_snapshot={},
                case_ids=["HAK180-001"],
            )
            historical_snapshot = dict(historical["config_snapshot"])
            historical_snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY] = legacy
            session.execute(
                text("UPDATE evaluation_runs SET config_snapshot = :snapshot WHERE id = :run_id"),
                {
                    "snapshot": json.dumps(historical_snapshot, ensure_ascii=False),
                    "run_id": historical["id"],
                },
            )
            session.commit()

            assert repository.get_run(historical["id"])["id"] == historical["id"]
    finally:
        engine.dispose()
