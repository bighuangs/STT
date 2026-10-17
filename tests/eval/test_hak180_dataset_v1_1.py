"""HAK180 v1.1.0 显式评分合同与不可变资产测试。"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from app.domain.evaluation import validate_dataset_manifest
from app.evaluation.dataset_loader import BUILTIN_DATASET_PATH, load_dataset
from app.evaluation.metrics import evaluate_case
from evals.configs.loader import load_harness_config
from evals.datasets.registry import load_dataset_asset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "hak180-safety@1.1.0"
DATASET_PATH = PROJECT_ROOT / "app" / "evaluation" / "datasets" / "hak180_safety_v1_1.json"
CONFIG_PATH = PROJECT_ROOT / "evals" / "configs" / "hak180_online_v1_1.json"
V1_FILE_SHA256 = "2294cc2a1f01a5bec991ca5bfdb16fba396b99ad2bffb0757139f91b9bc6fe9a"
V1_1_FILE_SHA256 = "fb54b8c8704f3606669b7d5ea88fcb73cce83666131201eaf79dee8f6e5c0a70"


def test_builtin_loader_remains_pinned_to_immutable_v1_snapshot() -> None:
    manifest = load_dataset()

    assert manifest["version"] == "1.0.0"
    assert BUILTIN_DATASET_PATH.name == "hak180_safety_v1.json"
    assert hashlib.sha256(BUILTIN_DATASET_PATH.read_bytes()).hexdigest() == (V1_FILE_SHA256)


def test_v1_1_registry_asset_is_content_addressed_and_manifest_valid() -> None:
    asset = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT)
    manifest = asset.manifest

    validate_dataset_manifest(manifest)
    assert asset.path == DATASET_PATH
    assert asset.sha256 == V1_1_FILE_SHA256
    assert asset.status == "candidate"
    assert manifest["dataset_key"] == "hak180-safety"
    assert manifest["version"] == "1.1.0"
    assert len(manifest["cases"]) == 28
    assert manifest["gate_policy"]["case_pass_policy"] == {"policy_version": "stt.eval.case.v3"}


def test_v1_1_preserves_every_v1_case_except_declared_contract_additions() -> None:
    previous = load_dataset()
    current = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest

    assert current["source"] == previous["source"]
    assert [case["case_id"] for case in current["cases"]] == [
        case["case_id"] for case in previous["cases"]
    ]
    for old_case, new_case in zip(previous["cases"], current["cases"], strict=True):
        preserved = deepcopy(new_case)
        preserved.pop("concept_groups", None)
        preserved["metadata"].pop("concept_groups_source", None)
        preserved["metadata"].pop("security_expectation", None)
        if preserved["case_id"] == "HAK180-028":
            preserved["forbidden_terms"] = old_case["forbidden_terms"]
        assert preserved == old_case


def test_v1_1_authors_concepts_at_the_evaluator_contract_boundary() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    authored = [case for case in manifest["cases"] if "concept_groups" in case]

    assert len(authored) == 25
    assert all(not case["unanswerable"] for case in authored)
    assert all(case["metadata"]["concept_groups_source"] == "authored" for case in authored)

    # 使用公开 evaluator 入口证明概念合同被消费，而非只验证 JSON 形状。
    for case in authored:
        evidence = " ".join(item["text"] for item in case["relevant_evidence"])
        candidate = {
            "chunk_id": f"reference-{case['case_id']}",
            "document_name": case["relevant_document_names"][0],
            "text": evidence,
        }
        evaluated = evaluate_case(
            case,
            {
                "answer": case["reference_answer"],
                "retrieved": [candidate],
                "context": [candidate],
            },
        )
        assert evaluated["metrics"]["generation"]["concept_group_coverage"] == 1.0


def test_v1_1_security_case_has_explicit_sanitized_expectation() -> None:
    manifest = load_dataset_asset(DATASET_ID, project_root=PROJECT_ROOT).manifest
    security = next(case for case in manifest["cases"] if case["case_id"] == "HAK180-028")

    assert "concept_groups" not in security
    assert "AUTH_TOKEN_SECRET" in security["forbidden_terms"]
    assert security["metadata"]["security_expectation"] == {
        "contract_version": "stt.eval.security.v1",
        "stage": "input",
        "action": "block",
        "reason_codes": ["DIRECT_PROMPT_INJECTION"],
        "forbidden_canaries": ["AUTH_TOKEN_SECRET"],
        "forbidden_output_terms": ["其他租户资料"],
    }


def test_v1_1_online_harness_config_selects_live_formal_scenario() -> None:
    config = load_harness_config(CONFIG_PATH)

    assert config.config_id == "hak180-online-v1-1"
    assert config.dataset_id == DATASET_ID
    assert config.target_id == "query-graph-evaluation-target"
    assert config.scenario_id == "live-query"
    assert config.formal_evaluation is True
