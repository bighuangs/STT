from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_HTML = PROJECT_ROOT / "app" / "process" / "query" / "page" / "evaluation.html"


def test_failure_columns_use_disjoint_authoritative_boolean_buckets() -> None:
    source = EVALUATION_HTML.read_text(encoding="utf-8")
    match = re.search(
        r"^    function failureBucketFromOutcome\(outcome\) \{.*?^    \}",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    assert "value.final_answer_correct === false" in match.group(0)
    assert "value.strict_supported_case_pass === false" in match.group(0)
    assert "value.final_answer_correct !== false" in match.group(0)

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the inline frontend contract")
    cases = [
        {"final_answer_correct": True, "strict_supported_case_pass": False},
        {"final_answer_correct": False, "strict_supported_case_pass": True},
        {"final_answer_correct": False, "strict_supported_case_pass": False},
        {"strict_supported_case_pass": False},
        {},
        {"final_answer_correct": True, "strict_supported_case_pass": True},
    ]
    script = (
        match.group(0)
        + "\nprocess.stdout.write(JSON.stringify("
        + json.dumps(cases)
        + ".map(failureBucketFromOutcome)));"
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(completed.stdout) == ["strict", "final", "final", "strict", "", ""]


def test_failure_column_dom_keeps_final_and_strict_counts_separate() -> None:
    source = EVALUATION_HTML.read_text(encoding="utf-8")
    for element_id in (
        "finalFailureCountPill",
        "finalFailureList",
        "finalFailureEmpty",
        "strictFailureCountPill",
        "strictFailureList",
        "strictFailureEmpty",
    ):
        assert source.count(f'id="{element_id}"') == 1
    assert "state.results.filter(finalAnswerFailure)" in source
    assert "state.results.filter(strictSupportedDiagnosticFailure)" in source
