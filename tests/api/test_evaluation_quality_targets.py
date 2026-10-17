from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = ROOT / "app/process/query/page/evaluation.html"
TARGET_PATH = (
    ROOT / "docs/evaluation-baselines/2026-08-11-three-difficulty-final" / "quality-targets.json"
)
FINAL_REVIEWED_SUITE_KEY = "three-difficulty-live-2026-08-11-reviewed-v2"


def test_three_difficulty_quality_target_document_is_answer_first() -> None:
    contract = json.loads(TARGET_PATH.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "stt.eval.quality-targets.v1"
    assert contract["suite_key"] == FINAL_REVIEWED_SUITE_KEY
    assert contract["primary_metric"] == {
        "name": "final_visible_answer_binary_accuracy",
        "production_path": "metrics.case_outcome.final_answer_accuracy",
        "description": "最终呈现给用户的答案按题二元判定正确或错误，再按难度汇总正确率。",
    }
    assert contract["independent_blind_review"]["role"] == "audit_only"
    assert contract["independent_blind_review"]["counts_toward_target"] is False

    assert contract["targets"]["easy"] == {
        "required": True,
        "operator": ">",
        "threshold": 0.95,
        "case_count": 50,
        "minimum_correct_cases": 48,
    }
    assert contract["targets"]["medium"] == {
        "required": True,
        "operator": ">",
        "threshold": 0.85,
        "case_count": 50,
        "minimum_correct_cases": 43,
    }
    assert contract["targets"]["hard"] == {
        "required": False,
        "operator": None,
        "threshold": None,
        "case_count": 50,
        "minimum_correct_cases": None,
    }


def test_evaluation_page_scopes_target_status_to_three_difficulty_suite() -> None:
    page = PAGE_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"    var DIFFICULTY_META = .*?(?=\n    function byId)",
        page,
        flags=re.DOTALL,
    )
    assert match is not None
    assert 'id="qualityTargetPanel"' in page
    assert "metrics.case_outcome.final_answer_accuracy" in page
    assert "严格有据通过率不参与目标判断" in page
    assert "独立盲审仅用于审计" in page

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript target contract")

    suite = FINAL_REVIEWED_SUITE_KEY
    script = (
        match.group(0)
        + f"\nconst suite='{suite}';"
        + "\nfunction dataset(level,suiteKey=suite){return {difficulty:level,"
        + "suite_key:suiteKey,case_count:50};}"
        + "\nfunction summary(rate,count=50){return {case_count:50,metrics:{case_outcome:{"
        + "final_answer_accuracy:{mean:rate,count:count},"
        + "strict_supported_case_pass_rate:{mean:1,count:count}}}};}"
        + "\nconsole.log(JSON.stringify({"
        + "easyMet:qualityTargetStatus(dataset('easy'),summary(.96),'succeeded'),"
        + "easyExactThreshold:qualityTargetStatus(dataset('easy'),summary(.95),'succeeded'),"
        + "easyWrong:qualityTargetStatus(dataset('easy'),summary(.94),'succeeded'),"
        + "mediumMet:qualityTargetStatus(dataset('medium'),summary(.86),'succeeded'),"
        + "mediumExactThreshold:qualityTargetStatus(dataset('medium'),summary(.85),'succeeded'),"
        + "hard:qualityTargetStatus(dataset('hard'),summary(.2),'succeeded'),"
        + "partial:qualityTargetStatus(dataset('easy'),summary(1,20),'running'),"
        + "other:qualityTargetStatus(dataset('easy','another-suite'),summary(1),'succeeded'),"
        + "config:QUALITY_TARGETS"
        + "}));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)
    frontend_contract = json.loads(TARGET_PATH.read_text(encoding="utf-8"))

    assert result["config"] == frontend_contract
    assert result["easyMet"]["state"] == "met"
    assert result["easyMet"]["accuracy"]["correct_count"] == 48
    assert result["easyExactThreshold"]["state"] == "not_met"
    assert result["easyWrong"]["state"] == "not_met"
    assert result["mediumMet"]["state"] == "met"
    assert result["mediumMet"]["accuracy"]["correct_count"] == 43
    assert result["mediumExactThreshold"]["state"] == "not_met"
    assert result["hard"]["state"] == "not_required"
    assert result["hard"]["label"] == "不强制"
    assert result["partial"]["state"] == "pending"
    assert result["other"] is None
