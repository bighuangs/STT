from __future__ import annotations

import hashlib
from pathlib import Path

from app.application.evaluation_service import _case_decision
from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    validate_dataset_manifest,
)
from app.evaluation.metrics import evaluate_case
from evals.configs.loader import load_harness_config
from evals.datasets.registry import load_dataset_asset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "hak180-safety@1.3.0"
DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "hak180_safety_v1_3.json"
CONFIG_PATH = PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_3.json"
V1_SHA256 = "2294cc2a1f01a5bec991ca5bfdb16fba396b99ad2bffb0757139f91b9bc6fe9a"
V1_1_SHA256 = "fb54b8c8704f3606669b7d5ea88fcb73cce83666131201eaf79dee8f6e5c0a70"
V1_2_SHA256 = "3288e1cfff85e754c07a509711a176970e2b8bf2be2afda0bc20cb603a735d8a"
V1_3_SHA256 = "cd374fc01754f1e463f605608e0cdce484ac0e398ec6b5542b08ebc3ea30f851"
POLICY = {"policy_version": AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_and_decide(case: dict, observation: dict) -> tuple[dict, dict]:
    evaluated = evaluate_case(case, observation, case_policy=POLICY)
    decision = _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy=POLICY,
        observation=observation,
    )
    return evaluated, decision


def test_v1_3_is_content_addressed_and_keeps_all_prior_assets_immutable() -> None:
    asset = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT)

    assert asset.path == DATASET_PATH
    assert asset.sha256 == V1_3_SHA256
    assert _sha256(PROJECT_ROOT / "app/evaluation/datasets/hak180_safety_v1.json") == V1_SHA256
    assert _sha256(PROJECT_ROOT / "app/evaluation/datasets/hak180_safety_v1_1.json") == V1_1_SHA256
    assert _sha256(PROJECT_ROOT / "evals/datasets/hak180_safety_v1_2.json") == V1_2_SHA256
    assert asset.manifest["parent_dataset"] == {
        "dataset_id": "hak180-safety@1.2.0",
        "sha256": V1_2_SHA256,
    }
    assert asset.manifest["gate_policy"]["case_pass_policy"] == POLICY
    validate_dataset_manifest(asset.manifest)


def test_v1_3_has_one_explicit_source_bound_query_predicate_contract() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    ellipsis_groups: list[tuple[str, dict]] = []

    for case in manifest["cases"]:
        concept_groups = case.get("concept_groups")
        if not concept_groups:
            continue
        assert concept_groups["contract_version"] == "stt.eval.authored-concepts.v3"
        for group in concept_groups["all_of"]:
            if (group.get("satisfaction") or {}).get("mode") == "query_predicate_ellipsis":
                ellipsis_groups.append((case["case_id"], group))

    assert [case_id for case_id, _ in ellipsis_groups] == ["HAK180-019"]
    calibrated = next(case for case in manifest["cases"] if case["case_id"] == "HAK180-019")
    assert calibrated["authored_claims"]["contract_version"] == "stt.eval.authored-claims.v1"
    assert len(calibrated["authored_claims"]["atoms"]) == 4


def test_v1_3_hak019_ellipsis_passes_only_with_live_source_and_citations() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    case = next(item for item in manifest["cases"] if item["case_id"] == "HAK180-019")
    source_text = case["relevant_evidence"][0]["text"]
    candidate = {
        "evidence_id": "p1-hazard-all-circles",
        "chunk_id": "chunk-hazard-table",
        "label": "E1",
        "text": source_text,
    }
    observation = {
        "answer": (
            "以下部件 [E1]\n- 标签 [E1]\n- 金属薄片保持单元 [E1]\n- 选配件 [E1]\n- 包装材料 [E1]"
        ),
        "retrieved": [candidate],
        "context": [candidate],
    }

    evaluated, decision = _evaluate_and_decide(case, observation)

    assert evaluated["metrics"]["generation"]["concept_group_coverage"] == 1.0
    assert evaluated["metrics"]["generation"]["authored_claim_coverage"] == 1.0
    assert decision["final_answer_correct"] is True
    assert decision["strict_supported_case_pass"] is True


def test_v1_3_does_not_calibrate_away_a_real_hak018_omission() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    case = next(item for item in manifest["cases"] if item["case_id"] == "HAK180-018")
    source_text = case["relevant_evidence"][0]["text"]
    candidate = {"evidence_id": "p1-hazard-lead-table", "label": "E1", "text": source_text}
    observation = {
        "answer": "主电路板的铅列标记为 × [E1]。",
        "retrieved": [candidate],
        "context": [candidate],
    }

    _, decision = _evaluate_and_decide(case, observation)

    assert decision["final_answer_correct"] is False
    assert "concept_group_incomplete" in decision["reason_codes"]


def test_v1_3_expected_abstentions_are_strict_with_grounding_and_citation_na() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    cases = {case["case_id"]: case for case in manifest["cases"]}

    for case_id in ("HAK180-025", "HAK180-026"):
        case = cases[case_id]
        observation = {
            "answer": case["reference_answer"],
            "abstained": True,
            "retrieved": [],
            "context": [],
            "visible_answer_policy": {
                "strict_grounding_eligible": False,
                "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
            },
        }
        evaluated, decision = _evaluate_and_decide(case, observation)

        assert evaluated["metrics"]["generation"]["faithfulness"] is None
        assert evaluated["metrics"]["citation"]["coverage"] is None
        assert decision["final_answer_correct"] is True
        assert decision["strict_supported_case_pass"] is True


def test_v1_3_online_config_freezes_formal_v5_dataset() -> None:
    config = load_harness_config(CONFIG_PATH)

    assert config.config_id == "hak180-online-v1-3"
    assert config.dataset_id == DATASET_ID
    assert config.target_id == "query-graph-evaluation-target"
    assert config.scenario_id == "live-query"
    assert config.formal_evaluation is True
