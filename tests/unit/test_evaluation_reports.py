import json
from datetime import UTC, datetime

import pytest

from app.application.evaluation_service import (
    _case_decision,
    _final_report_run_snapshot,
)
from app.evaluation.metrics import aggregate_results, evaluate_case
from app.evaluation.reports import (
    _completeness_self_review_audit_text,
    render_html_report,
    render_json_report,
    render_markdown_report,
)


def _payload():
    case = {
        "case_id": "xss-1",
        "query": "<script>alert(1)</script>",
        "reference_answer": "正常答案",
        "expected_terms": ["正常答案"],
        "relevant_chunk_ids": ["safe"],
        "tags": ["<img src=x onerror=alert(1)>"],
    }
    result = evaluate_case(
        case,
        {
            "answer": "正常答案。[safe]",
            "retrieved": [{"chunk_id": "safe", "text": "正常答案"}],
            "context": [{"chunk_id": "safe", "text": "正常答案"}],
        },
    )
    result["metrics"]["case_outcome"] = _case_decision(
        case,
        result,
        had_error=False,
    )
    return {
        "title": "<script>alert('title')</script>",
        "run_id": "run-1",
        "aggregate": aggregate_results([result]),
        "results": [result],
    }


def test_json_report_is_stable_valid_and_keeps_unknown_as_null():
    rendered = render_json_report(_payload())
    parsed = json.loads(rendered)
    assert parsed["results"][0]["runtime"]["total_tokens"] is None
    assert parsed["aggregate"]["runtime"]["cost"]["total"] is None
    assert rendered == render_json_report(_payload())


def test_final_report_run_snapshot_contains_the_durable_terminal_state() -> None:
    completed_at = datetime(2026, 8, 15, 2, 23, 52, tzinfo=UTC)
    summary = {"case_count": 28}
    gate = {"status": "failed", "passed": False}

    snapshot = _final_report_run_snapshot(
        {
            "id": "run-final",
            "status": "running",
            "progress": 1.0,
            "completed_cases": 28,
            "total_cases": 28,
            "metrics_summary": {},
            "comparison_summary": {},
            "gate_result": {"status": "running", "passed": None},
            "completed_at": None,
        },
        summary=summary,
        comparison={},
        gate=gate,
        completed_at=completed_at,
    )

    assert snapshot["status"] == "succeeded"
    assert snapshot["metrics_summary"] == summary
    assert snapshot["gate_result"] == gate
    assert snapshot["completed_at"] == "2026-08-15T02:23:52+00:00"
    assert snapshot["updated_at"] == snapshot["completed_at"]


def test_markdown_report_explicitly_displays_unknown_tokens_and_cost():
    rendered = render_markdown_report(_payload())
    assert "tokens total | unknown" in rendered
    assert "cost total | unknown" in rendered
    assert "strict supported case pass rate | 1" in rendered
    assert "final answer accuracy | 1" in rendered
    assert "case quality score | 1" in rendered
    assert "manifest 省略策略时使用 v2" in rendered
    assert "v3 候选概念合同" in rendered
    assert "| xss-1 | correct | passed | 1 | unknown |" in rendered
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered


def test_html_report_escapes_every_untrusted_value_and_displays_unknown():
    rendered = render_html_report(_payload())
    assert "<!doctype html>" in rendered
    assert "&lt;script&gt;alert(&#x27;title&#x27;)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<script>alert" not in rendered
    assert "<img src=x" not in rendered
    assert ">unknown</td>" in rendered
    assert ">strict supported case pass rate</td><td>1</td>" in rendered
    assert ">final answer accuracy</td><td>1</td>" in rendered
    assert "manifest 省略策略时使用 v2" in rendered
    assert "v3 候选概念合同" in rendered
    assert ">xss-1</td><td>correct</td><td>passed</td><td>1</td>" in rendered


def test_reports_serialize_and_display_visible_answer_degradation_audit() -> None:
    case = {
        "case_id": "generic-audit",
        "query": "标准容量是多少？",
        "reference_answer": "3000 页。",
        "expected_terms": ["3000 页"],
        "relevant_chunk_ids": ["chunk-audit"],
    }
    policy = {
        "schema_version": "visible-answer-policy.v1",
        "mode": "safe_pre_degraded",
        "status": "degraded",
        "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        "visible_answer_sha256": "a" * 64,
        "rejected_post_answer_sha256": "b" * 64,
        "trusted_citation_count": 1,
        "removed_untrusted_marker_count": 1,
        "strict_grounding_eligible": False,
    }
    observation = {
        "answer": "标准容量是 3000 页。[chunk-audit]",
        "retrieved": [{"chunk_id": "chunk-audit", "text": "标准容量是 3000 页。"}],
        "context": [{"chunk_id": "chunk-audit", "text": "标准容量是 3000 页。"}],
        "visible_answer_policy": policy,
        "completeness_self_review": {
            "schema_version": "completeness-self-review-audit.v1",
            "status": "committed",
            "calls_attempted": 1,
            "model_id": "model-a",
            "usage": {
                "status": "available",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
            "trigger_reason_codes": ["UNCOVERED_QUESTION_CLAUSES"],
            "prompt_sha256": "must-remain-archive-only",
        },
    }
    evaluated = evaluate_case(case, observation)
    evaluated["metrics"]["case_outcome"] = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=observation,
    )
    durable_result = {
        "case_id": case["case_id"],
        "observation": observation,
        "metrics": evaluated,
        "passed": False,
    }
    payload = {
        "summary": aggregate_results([evaluated]),
        "results": [durable_result],
    }

    parsed = json.loads(render_json_report(payload))
    persisted = parsed["results"][0]
    assert persisted["observation"]["visible_answer_policy"] == policy
    assert persisted["metrics"]["visible_answer_policy"] == policy
    assert (
        persisted["observation"]["completeness_self_review"]["prompt_sha256"]
        == "must-remain-archive-only"
    )
    assert persisted["metrics"]["metrics"]["case_outcome"]["strict_supported_case_pass"] is False
    assert persisted["metrics"]["metrics"]["case_outcome"]["final_answer_correct"] is True

    label = "答案完整性优先，部分内容未通过引用校验"
    markdown = render_markdown_report(payload)
    assert label in markdown
    assert "attempted=yes; status=committed; committed=yes; model=model-a" in markdown
    assert "usage=status=available,input_tokens=100,output_tokens=20,total_tokens=120" in markdown
    assert "trigger_codes=UNCOVERED_QUESTION_CLAUSES" in markdown
    assert "prompt_sha256" not in markdown
    html_report = render_html_report(payload)
    assert label in html_report
    main_html = html_report.split("<details>", 1)[0]
    assert "attempted=yes; status=committed; committed=yes; model=model-a" in main_html
    assert "prompt_sha256" not in main_html


@pytest.mark.parametrize(
    "status",
    [
        "committed_deterministic_set_cover",
        "committed_deterministic_insufficient_evidence",
        "current_complete",
    ],
)
def test_report_marks_runtime_completeness_success_statuses_as_committed(
    status: str,
) -> None:
    rendered = _completeness_self_review_audit_text(
        {
            "observation": {
                "completeness_self_review": {
                    "status": status,
                    "calls_attempted": 0,
                    "model_id": "not_applicable",
                    "usage": {"status": "not_applicable"},
                    "trigger_reason_codes": [],
                }
            }
        },
        {},
    )

    assert f"status={status}; committed=yes" in rendered


@pytest.mark.parametrize(
    ("scope", "mode", "expected_title"),
    [
        ("full", "live", "STT 全链路在线评测报告"),
        ("retrieval_only", "retrieval_only", "STT 仅检索评测报告"),
    ],
)
def test_default_report_title_distinguishes_live_and_retrieval_only(
    scope: str,
    mode: str,
    expected_title: str,
) -> None:
    payload = {
        "evaluation_scope": scope,
        "run": {"mode": mode},
        "summary": {},
    }

    assert render_markdown_report(payload).startswith(f"# {expected_title}\n")
    rendered_html = render_html_report(payload)
    assert f"<title>{expected_title}</title>" in rendered_html
    assert f"<h1>{expected_title}</h1>" in rendered_html


def test_report_displays_suite_metadata_projected_on_the_run() -> None:
    payload = {
        "evaluation_scope": "full",
        "run": {
            "mode": "live",
            "suite_key": "three-difficulty-live-2026-08-10",
            "difficulty": "hard",
            "minimum_case_count": 50,
            "evaluation_mode_default": "live",
        },
        "summary": {},
    }

    markdown = render_markdown_report(payload)
    assert "| suite_key | three-difficulty-live-2026-08-10 |" in markdown
    assert "| difficulty | hard |" in markdown
    assert "| minimum_case_count | 50 |" in markdown
    html_report = render_html_report(payload)
    assert ">suite_key</td><td>three-difficulty-live-2026-08-10</td>" in html_report
    assert ">difficulty</td><td>hard</td>" in html_report
