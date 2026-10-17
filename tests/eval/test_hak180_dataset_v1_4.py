from __future__ import annotations

import hashlib
from pathlib import Path

from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
    freeze_case_pass_policy,
    validate_dataset_manifest,
)
from evals.configs.loader import load_harness_config
from evals.datasets.registry import load_dataset_asset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "hak180-safety@1.4.0"
DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "hak180_safety_v1_4.json"
FORMAL_CONFIG_PATH = PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_4_authority_v3.json"
PARENT_FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_3_authority_v3.json"
)
V1_SHA256 = "2294cc2a1f01a5bec991ca5bfdb16fba396b99ad2bffb0757139f91b9bc6fe9a"
V1_1_SHA256 = "fb54b8c8704f3606669b7d5ea88fcb73cce83666131201eaf79dee8f6e5c0a70"
V1_2_SHA256 = "3288e1cfff85e754c07a509711a176970e2b8bf2be2afda0bc20cb603a735d8a"
V1_3_SHA256 = "cd374fc01754f1e463f605608e0cdce484ac0e398ec6b5542b08ebc3ea30f851"
V1_4_SHA256 = "62c2646e9e9ea18c20d26dde2f9634d1db699e6ecf7c631dd3e7cba8b1653f20"
FORMAL_CONFIG_RAW_SHA256 = "eeea239b41a2b3f6042de9a8c887e8f0896a4404a2b12462ba2cd34f99bb160b"
FORMAL_CONFIG_CANONICAL_SHA256 = "ba35c528e7ae4329e84330db42f8dbaa0163476b15ddcfbf123f8fae7ab6af98"
POLICY = {"policy_version": TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_4_is_content_addressed_and_keeps_v1_through_v1_3_immutable() -> None:
    asset = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT)

    assert asset.path == DATASET_PATH
    assert asset.sha256 == V1_4_SHA256
    assert _sha256(PROJECT_ROOT / "app/evaluation/datasets/hak180_safety_v1.json") == V1_SHA256
    assert _sha256(PROJECT_ROOT / "app/evaluation/datasets/hak180_safety_v1_1.json") == V1_1_SHA256
    assert _sha256(PROJECT_ROOT / "evals/datasets/hak180_safety_v1_2.json") == V1_2_SHA256
    assert _sha256(PROJECT_ROOT / "evals/datasets/hak180_safety_v1_3.json") == V1_3_SHA256
    assert asset.manifest["parent_dataset"] == {
        "dataset_id": "hak180-safety@1.3.0",
        "sha256": V1_3_SHA256,
    }
    assert asset.manifest["gate_policy"]["case_pass_policy"] == POLICY
    validate_dataset_manifest(asset.manifest)


def test_v1_4_is_a_metadata_only_copy_of_all_28_parent_cases() -> None:
    parent = load_dataset_asset("hak180-safety@1.3.0", project_root=PROJECT_ROOT).manifest
    current = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    mutable_metadata = {"description", "version", "tags", "gate_policy", "parent_dataset"}

    assert len(current["cases"]) == 28
    assert current["cases"] == parent["cases"]
    assert current["source"] == parent["source"]
    assert current["calibration"] == parent["calibration"]
    assert {key: value for key, value in current.items() if key not in mutable_metadata} == {
        key: value for key, value in parent.items() if key not in mutable_metadata
    }


def test_v6_inherits_v5_thresholds_and_freezes_the_temporal_claim_contract() -> None:
    v5 = freeze_case_pass_policy({"policy_version": AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION})
    v6 = freeze_case_pass_policy(POLICY)

    assert TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION == "stt.eval.case.v6"
    assert TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION == "stt.eval.temporal-derived-claim.v1"
    assert {key: value for key, value in v6.items() if key != "policy_version"} == {
        key: value for key, value in v5.items() if key != "policy_version"
    }


def test_v1_4_formal_v3_config_preserves_planning_and_usage_boundaries() -> None:
    config = load_harness_config(FORMAL_CONFIG_PATH)
    parent = load_harness_config(PARENT_FORMAL_CONFIG_PATH)

    assert _sha256(FORMAL_CONFIG_PATH) == FORMAL_CONFIG_RAW_SHA256
    assert config.canonical_sha256 == FORMAL_CONFIG_CANONICAL_SHA256
    assert config.schema_version == "stt.eval.harness-config.v3"
    assert config.config_id == "hak180-online-v1-4-authority-v3"
    assert config.dataset_id == DATASET_ID
    assert config.target_id == parent.target_id == "query-graph-evaluation-target"
    assert config.scenario_id == parent.scenario_id == "live-query"
    assert config.formal_evaluation is parent.formal_evaluation is True
    assert config.budgets is parent.budgets is None
    assert config.planning_admission_budgets == parent.planning_admission_budgets
    assert config.usage_accounting == parent.usage_accounting
    assert (
        config.allowed_intervention_dimensions
        == parent.allowed_intervention_dimensions
        == ("code_revision",)
    )
