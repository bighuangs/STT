from __future__ import annotations

import hashlib
from pathlib import Path

from app.domain.evaluation import (
    CASE_PASS_POLICY_VERSION,
    SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
    validate_dataset_manifest,
)
from app.evaluation.metrics import evaluate_case
from evals.configs.loader import load_harness_config
from evals.datasets.registry import load_dataset_asset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "hak180-safety@1.2.0"
DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "hak180_safety_v1_2.json"
CONFIG_PATH = PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_2.json"
V1_1_PATH = PROJECT_ROOT / "app" / "evaluation" / "datasets" / "hak180_safety_v1_1.json"
V1_1_SHA256 = "fb54b8c8704f3606669b7d5ea88fcb73cce83666131201eaf79dee8f6e5c0a70"
V1_2_SHA256 = "3288e1cfff85e754c07a509711a176970e2b8bf2be2afda0bc20cb603a735d8a"


def _evaluate(case: dict, answer: str, *, policy_version: str) -> dict:
    evidence = " ".join(item["text"] for item in case.get("relevant_evidence") or [])
    candidate = {"chunk_id": f"source-{case['case_id']}", "text": evidence}
    return evaluate_case(
        case,
        {"answer": answer, "retrieved": [candidate], "context": [candidate]},
        case_policy={"policy_version": policy_version},
    )


def test_v1_2_is_content_addressed_and_keeps_v1_1_immutable() -> None:
    asset = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT)

    assert asset.path == DATASET_PATH
    assert asset.sha256 == V1_2_SHA256
    assert hashlib.sha256(V1_1_PATH.read_bytes()).hexdigest() == V1_1_SHA256
    assert asset.manifest["version"] == "1.2.0"
    assert asset.manifest["parent_dataset"] == {
        "dataset_id": "hak180-safety@1.1.0",
        "sha256": V1_1_SHA256,
    }
    assert asset.manifest["gate_policy"]["case_pass_policy"] == {
        "policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION
    }
    validate_dataset_manifest(asset.manifest)


def test_v1_2_required_groups_are_source_bound_and_reference_self_checks() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    authored = [case for case in manifest["cases"] if "concept_groups" in case]
    diagnostic_count = 0

    for case in authored:
        contract = case["concept_groups"]
        assert contract["contract_version"] == "stt.eval.authored-concepts.v2"
        required = []
        for group in contract["all_of"]:
            alignment = group["source_alignment"]
            if alignment["mode"] == "diagnostic":
                diagnostic_count += 1
                assert alignment["reason_code"] == ("REFERENCE_DETAIL_NOT_QREL_SUPPORTED")
            else:
                required.append(group)
                assert alignment["anchors"]
                assert alignment["source_terms"]
        assert required

        reference = _evaluate(
            case,
            case["reference_answer"],
            policy_version=SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        )
        assert reference["metrics"]["generation"]["concept_group_coverage"] == 1.0

    assert diagnostic_count == 3


def test_v1_2_joint_anchors_cannot_hide_hak014_or_hak019_omissions() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    cases = {case["case_id"]: case for case in manifest["cases"]}
    omissions = {
        "HAK180-014": "应使用无绒干抹布；不能使用易燃物品、喷雾剂以及含氨水的液体，否则可能导致火灾。",
        "HAK180-019": "标签、金属薄片保持单元和包装材料的六类有害物质均标为 ○。",
    }

    for case_id, answer in omissions.items():
        result = _evaluate(
            cases[case_id],
            answer,
            policy_version=SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        )
        assert result["metrics"]["generation"]["concept_group_coverage"] < 1.0


def test_v1_2_source_alignment_fields_do_not_change_v3_matcher_semantics() -> None:
    previous = load_dataset_asset("hak180-safety@1.1.0", project_root=PROJECT_ROOT).manifest
    current = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest

    for old_case, new_case in zip(previous["cases"], current["cases"], strict=True):
        if "concept_groups" not in old_case:
            continue
        source = " ".join(item["text"] for item in old_case["relevant_evidence"])
        old_generation = _evaluate(
            old_case,
            source,
            policy_version=CASE_PASS_POLICY_VERSION,
        )["metrics"]["generation"]
        new_generation = _evaluate(
            new_case,
            source,
            policy_version=CASE_PASS_POLICY_VERSION,
        )["metrics"]["generation"]
        assert new_generation == old_generation


def test_v1_2_online_config_freezes_formal_v4_dataset() -> None:
    config = load_harness_config(CONFIG_PATH)

    assert config.config_id == "hak180-online-v1-2"
    assert config.dataset_id == DATASET_ID
    assert config.target_id == "query-graph-evaluation-target"
    assert config.scenario_id == "live-query"
    assert config.formal_evaluation is True
