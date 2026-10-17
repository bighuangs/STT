"""根级评测 harness 与生产目标之间的依赖边界测试。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app.evaluation.observation import (
    QueryGraphEvalAdapter,
    RetrievalOnlyEvalAdapter,
    StaticObservationAdapter,
)
from app.domain.evaluation_proof import (
    ProofEvaluationAuthority,
    ProofEvidenceAuthority,
    ProofRepresentationAuthority,
)
from evals.baselines import load_baseline_asset
from evals.configs import load_harness_config
from evals.datasets import load_dataset_asset
from evals.graders import ObservationContractGrader
from evals.harness import (
    EvaluationHarness,
    EvaluationRequest,
    EvaluationTarget,
    EvaluationTargetResult,
    ObservationContractError,
    ensure_observation_contract,
    TargetInput,
)
from evals.scenarios import load_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        target_input=TargetInput(
            query="安全检查",
            correlation_id="opaque-contract",
            sensitivity="internal",
        ),
        kb_id="kb-contract",
        run_id="run-contract",
        config={"config_version": "cfg-contract"},
    )


def test_static_target_runs_through_public_harness_contract() -> None:
    target = StaticObservationAdapter(
        {
            "opaque-contract": {
                "answer": "关闭电源后检查。",
                "retrieved": [{"chunk_id": "chunk-1"}],
            }
        }
    )
    harness = EvaluationHarness(
        target=target,
        graders=(ObservationContractGrader(),),
    )

    result = harness.run_case(
        _request(),
        grader_case={"case_id": "contract-1", "query": "安全检查"},
    )

    assert result["schema_version"] == "stt.eval.case-result.v1"
    assert result["observation"]["observation_schema"] == "stt.eval.observation.v2"
    assert result["grades"] == [{"grader": "observation_contract", "passed": True, "score": 1.0}]


def test_public_harness_unwraps_result_and_drops_raw_authority() -> None:
    raw_text = "RAW-AUTHORITY-MUST-NOT-REACH-GRADER-OR-RESULT"
    observation = StaticObservationAdapter(
        {"opaque-contract": {"answer": "关闭电源后检查。", "retrieved": []}}
    ).evaluate(_request())
    authority = ProofEvaluationAuthority(
        question_contract_json="{}",
        witness_catalog_json="{}",
        witness_rows_json=(),
        evidences=(ProofEvidenceAuthority("evd-harness", raw_text),),
        representations=(ProofRepresentationAuthority("rep-harness", raw_text),),
        source_ledger_complete=False,
        witness_ledger_complete=False,
    )

    class WrappedTarget:
        def evaluate(self, _request: EvaluationRequest) -> EvaluationTargetResult:
            return EvaluationTargetResult(
                observation=observation,
                proof_authority=authority,
            )

    result = EvaluationHarness(
        target=WrappedTarget(),
        graders=(ObservationContractGrader(),),
    ).run_case(
        _request(),
        grader_case={"case_id": "contract-wrapper", "query": "安全检查"},
    )

    assert result["observation"] == observation
    assert raw_text not in json.dumps(result, ensure_ascii=False)


def test_observation_contract_fails_closed_for_private_state_like_payload() -> None:
    with pytest.raises(ObservationContractError, match="observation_schema"):
        ensure_observation_contract(
            {
                "messages": [],
                "embedding_chunks": [],
                "reranked_docs": [],
            }
        )


def test_query_graph_target_implements_wrapped_public_evaluate_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = QueryGraphEvalAdapter()
    captured: dict = {}

    def fake_evaluate_result(case: dict, *, kb_id: str, run_id: str, config: dict):
        captured.update(
            case=case,
            kb_id=kb_id,
            run_id=run_id,
            config=config,
        )
        return EvaluationTargetResult(
            observation={"observation_schema": "stt.eval.observation.v3"}
        )

    monkeypatch.setattr(target, "_evaluate_result", fake_evaluate_result)

    assert isinstance(target, EvaluationTarget)
    result = target.evaluate(_request())
    assert isinstance(result, EvaluationTargetResult)
    assert result.observation == {"observation_schema": "stt.eval.observation.v3"}
    assert captured == {
        "case": {
            "query": "安全检查",
            "correlation_id": "opaque-contract",
            "sensitivity": "internal",
            "conversation": [],
        },
        "kb_id": "kb-contract",
        "run_id": "run-contract",
        "config": {"config_version": "cfg-contract"},
    }


def test_retrieval_target_keeps_mapping_public_evaluate_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = RetrievalOnlyEvalAdapter()
    captured: dict = {}

    def fake_observe(case: dict, *, kb_id: str, run_id: str, config: dict) -> dict:
        captured.update(case=case, kb_id=kb_id, run_id=run_id, config=config)
        return {"observation_schema": "stt.eval.observation.v2"}

    monkeypatch.setattr(target, "observe", fake_observe)

    assert isinstance(target, EvaluationTarget)
    assert target.evaluate(_request()) == {"observation_schema": "stt.eval.observation.v2"}
    assert captured["case"] == {
        "query": "安全检查",
        "correlation_id": "opaque-contract",
        "sensitivity": "internal",
        "conversation": [],
    }


def test_root_evals_never_imports_production_or_infrastructure_packages() -> None:
    forbidden = (
        "app.process",
        "app.rag",
        "app.infra",
        "app.infrastructure",
    )
    for path in sorted((PROJECT_ROOT / "evals").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        app_imports = {name for name in imports if name == "app" or name.startswith("app.")}
        assert app_imports <= {"app.application.evaluation_target"}, (
            f"{path} bypasses the public evaluation target contract: {sorted(app_imports)}"
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in forbidden
        ), f"{path} imports a forbidden production package: {imports}"


def test_legacy_observation_facade_contains_no_private_graph_import() -> None:
    source = (PROJECT_ROOT / "app/evaluation/observation.py").read_text(encoding="utf-8")
    assert "app.process" not in source
    assert "app.rag" not in source


def test_registered_dataset_and_baseline_are_content_addressed() -> None:
    dataset = load_dataset_asset(
        "hak180-safety@1.0.0",
        project_root=PROJECT_ROOT,
    )
    baseline = load_baseline_asset(
        "three-difficulty-live@2026-08-10",
        project_root=PROJECT_ROOT,
    )

    assert dataset.manifest["dataset_key"] == "hak180-safety"
    assert dataset.status == "frozen"
    assert baseline.payload["ready_for_import"] is True


def test_dataset_registry_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-eval.json"
    outside.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "stt.eval.dataset-registry.v1",
                "datasets": {
                    "escaped": {
                        "path": "../outside-eval.json",
                        "sha256": "0" * 64,
                        "status": "candidate",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inside project root"):
        load_dataset_asset(
            "escaped",
            project_root=tmp_path,
            registry_path=registry,
        )


def test_offline_config_is_explicitly_not_a_formal_evaluation() -> None:
    config = load_harness_config()
    scenario = load_scenario(config.scenario_id)

    assert config.formal_evaluation is False
    assert scenario.formal_evaluation is False
    assert scenario.external_model_allowed is False
    assert scenario.allowed_side_effects == ()
