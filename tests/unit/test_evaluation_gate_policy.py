from __future__ import annotations

import pytest

from app.application import evaluation_service
from app.domain.evaluation import (
    CASE_PASS_POLICY_VERSION,
    DEFAULT_CASE_PASS_POLICY,
    DEFAULT_EVALUATION_GATE_POLICY,
    EVALUATION_GATE_POLICY_VERSION,
    PREVIOUS_CASE_PASS_POLICY_VERSION,
    SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
    freeze_case_pass_policy,
    freeze_evaluation_gate_policy,
)


CASE_IDS = [f"case-{index}" for index in range(1, 8)]


def _comparison(*, delta: float = -0.08) -> dict:
    metric = {
        "delta": delta,
        "paired_count": 7,
        "unknown_pair_count": 1,
    }
    return {
        "matched_case_count": len(CASE_IDS),
        "matched_case_ids": CASE_IDS,
        "baseline_only_case_ids": [],
        "candidate_only_case_ids": [],
        "metric_deltas": {
            "metrics.retrieval.recall_at_k.1": dict(metric),
            "metrics.retrieval.recall_at_k.3": dict(metric),
            "metrics.retrieval.recall_at_k.10": {
                **metric,
                "delta": 0.0,
            },
            "metrics.retrieval.mrr": dict(metric),
            "metrics.retrieval.ndcg_at_k.10": dict(metric),
        },
        "baseline": {"runtime": {"latency_ms": {"p95": 100.0}}},
        "candidate": {"runtime": {"latency_ms": {"p95": 90.0}}},
    }


def _cases() -> list[dict]:
    return [{"case_id": case_id, "metadata": {}} for case_id in CASE_IDS]


def _run(policy: dict) -> dict:
    return {
        "mode": "retrieval_only",
        "baseline_run_id": "erun-baseline",
        "config_snapshot": {"gate_policy": policy},
    }


def _summary() -> dict:
    return {
        "runtime": {
            "total_tokens": {"count": 0},
            "cost": {"count": 0},
        }
    }


def test_new_gate_policy_materializes_safe_defaults_and_preserves_extensions() -> None:
    frozen = freeze_evaluation_gate_policy(
        {
            "recall_at_1_max_drop": 0.01,
            "external_vision_allowed": False,
        }
    )

    assert frozen["policy_version"] == EVALUATION_GATE_POLICY_VERSION
    assert frozen["recall_at_1_max_drop"] == 0.01
    assert frozen["recall_at_3_max_drop"] == 0.02
    assert frozen["mrr_max_drop"] == 0.02
    assert frozen["ndcg_at_10_max_drop"] == 0.02
    assert frozen["answer_correctness_max_drop"] == 0.0
    assert frozen["case_pass_policy"] == DEFAULT_CASE_PASS_POLICY
    assert frozen["case_pass_policy"] is not DEFAULT_CASE_PASS_POLICY
    assert frozen["external_vision_allowed"] is False
    assert DEFAULT_EVALUATION_GATE_POLICY["recall_at_1_max_drop"] == 0.02


@pytest.mark.parametrize(
    "policy",
    [
        {"recall_at_1_max_drop": -0.01},
        {"mrr_max_drop": float("nan")},
        {"ndcg_at_10_max_drop": True},
        {"p95_latency_max_increase_ratio": 11},
        {"critical_failure_blocks": "yes"},
        {"policy_version": "stt.eval.gate.v999"},
        {"case_pass_policy": {"faithfulness_minimum": 0.0}},
        {
            "case_pass_policy": {
                "answer_correctness_weight": 0.4,
                "faithfulness_weight": 0.3,
                "citation_coverage_weight": 0.2,
                "retrieval_support_weight": 0.1,
            }
        },
    ],
)
def test_gate_policy_rejects_unsafe_or_unknown_values(policy: dict) -> None:
    with pytest.raises(ValueError, match="evaluation (?:gate|case pass)|gate policy"):
        freeze_evaluation_gate_policy(policy)


def test_v2_gate_blocks_early_rank_mrr_and_ndcg_regressions() -> None:
    policy = freeze_evaluation_gate_policy({})

    gate = evaluation_service._gate(
        _run(policy),
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=_comparison(),
    )

    assert gate["status"] == "failed"
    assert gate["passed"] is False
    assert gate["policy_version"] == EVALUATION_GATE_POLICY_VERSION
    assert {item["rule"] for item in gate["failures"]} == {
        "recall_at_1",
        "recall_at_3",
        "mrr",
        "ndcg_at_10",
    }
    assert all(item["max_drop"] == 0.02 for item in gate["failures"])
    assert all(item["paired_count"] == 7 for item in gate["failures"])
    assert all(item["unknown_pair_count"] == 1 for item in gate["failures"])


def test_historical_unversioned_snapshot_keeps_v1_gate_semantics() -> None:
    gate = evaluation_service._gate(
        _run({"recall_at_10_max_drop": 0.02}),
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=_comparison(),
    )

    assert gate["status"] == "passed"
    assert gate["passed"] is True
    assert gate["policy_version"] == "stt.eval.gate.v1"
    assert gate["failures"] == []


def test_gate_fails_closed_for_easy_subset_even_when_metrics_improve() -> None:
    comparison = _comparison(delta=0.5)
    comparison.update(
        {
            "matched_case_count": 1,
            "matched_case_ids": [CASE_IDS[0]],
            "baseline_only_case_ids": CASE_IDS[1:],
        }
    )

    gate = evaluation_service._gate(
        _run(freeze_evaluation_gate_policy({})),
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    pairing = next(item for item in gate["failures"] if item["rule"] == "paired_case_set")
    assert gate["passed"] is False
    assert pairing["reason"] == "incomplete_pairing"
    assert pairing["matched_case_count"] == 1
    assert pairing["baseline_only_case_ids"] == CASE_IDS[1:]


def test_gate_fails_closed_for_zero_overlap() -> None:
    comparison = _comparison(delta=0.5)
    comparison.update(
        {
            "matched_case_count": 0,
            "matched_case_ids": [],
            "baseline_only_case_ids": CASE_IDS,
            "candidate_only_case_ids": ["other-case"],
        }
    )

    gate = evaluation_service._gate(
        _run(freeze_evaluation_gate_policy({})),
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    pairing = next(item for item in gate["failures"] if item["rule"] == "paired_case_set")
    assert gate["passed"] is False
    assert pairing["reason"] == "zero_overlap"


def test_gate_accepts_the_same_complete_case_set_when_all_rules_pass() -> None:
    gate = evaluation_service._gate(
        _run(freeze_evaluation_gate_policy({})),
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=_comparison(delta=0.0),
    )

    assert gate["passed"] is True
    assert gate["failures"] == []


def test_noncritical_security_contract_failure_always_blocks_root_and_candidate() -> None:
    cases = _cases()
    cases[0]["metadata"] = {
        "security_expectation": {
            "contract_version": "stt.eval.security.v1",
            "stage": "input",
            "action": "block",
            "reason_codes": ["DIRECT_PROMPT_INJECTION"],
        },
        "critical": False,
    }
    results = [
        {
            "case_id": CASE_IDS[0],
            "passed": False,
            "metrics": {"metrics": {"safety": {"passed": False}}},
        }
    ]
    root_run = _run(freeze_evaluation_gate_policy({}))
    root_run.update({"mode": "live", "baseline_run_id": None})
    candidate_run = _run(freeze_evaluation_gate_policy({}))
    candidate_run["mode"] = "live"

    root_gate = evaluation_service._gate(
        root_run,
        cases=cases,
        results=results,
        summary=_summary(),
        comparison={},
    )
    candidate_gate = evaluation_service._gate(
        candidate_run,
        cases=cases,
        results=results,
        summary=_summary(),
        comparison=_comparison(delta=0.0),
    )

    assert root_gate["passed"] is False
    assert root_gate["safety_failed_cases"] == [CASE_IDS[0]]
    assert candidate_gate["passed"] is False
    assert {item["rule"] for item in candidate_gate["failures"]} == {"safety_contracts"}


def test_live_gate_blocks_final_answer_and_case_quality_regressions() -> None:
    comparison = _comparison(delta=0.0)
    comparison["metric_deltas"].update(
        {
            "metrics.case_outcome.final_answer_accuracy": {
                "delta": -0.01,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            },
            "metrics.case_outcome.quality_score": {
                "delta": -0.005,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            },
            "metrics.generation.faithfulness": {
                "delta": 0.0,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            },
        }
    )
    run = _run(freeze_evaluation_gate_policy({}))
    run["mode"] = "live"

    gate = evaluation_service._gate(
        run,
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    assert gate["passed"] is False
    assert gate["case_pass_policy_version"] == PREVIOUS_CASE_PASS_POLICY_VERSION
    assert {failure["rule"] for failure in gate["failures"]} == {
        "answer_correctness",
        "case_quality",
    }


def test_v3_gate_uses_contract_abstention_while_v2_keeps_legacy_metric() -> None:
    comparison = _comparison(delta=0.0)
    comparison["metric_deltas"].update(
        {
            "metrics.abstention.correct": {
                "delta": -1.0,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            },
            "metrics.abstention.contract_correct": {
                "delta": 0.0,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            },
        }
    )
    v3_run = _run(
        freeze_evaluation_gate_policy(
            {"case_pass_policy": {"policy_version": CASE_PASS_POLICY_VERSION}}
        )
    )
    v3_run["mode"] = "live"
    v2_run = _run(freeze_evaluation_gate_policy({}))
    v2_run["mode"] = "live"

    v3_gate = evaluation_service._gate(
        v3_run,
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )
    v2_gate = evaluation_service._gate(
        v2_run,
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    assert v3_gate["passed"] is True
    assert v2_gate["passed"] is False
    assert {failure["rule"] for failure in v2_gate["failures"]} == {"unanswerable_accuracy"}


def test_v4_gate_keeps_all_contract_and_final_quality_paired_guards() -> None:
    comparison = _comparison(delta=0.0)
    comparison["metric_deltas"].update(
        {
            metric: {
                "delta": -1.0,
                "paired_count": len(CASE_IDS),
                "unknown_pair_count": 0,
            }
            for metric in (
                "metrics.abstention.contract_correct",
                "metrics.case_outcome.final_answer_accuracy",
                "metrics.case_outcome.quality_score",
                "metrics.generation.faithfulness",
            )
        }
    )
    run = _run(
        freeze_evaluation_gate_policy(
            {"case_pass_policy": {"policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION}}
        )
    )
    run["mode"] = "live"

    gate = evaluation_service._gate(
        run,
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    assert gate["passed"] is False
    assert {failure["rule"] for failure in gate["failures"]} == {
        "unanswerable_accuracy",
        "answer_correctness",
        "case_quality",
        "faithfulness",
    }


def test_content_security_exception_projects_auditable_pre_retrieval_block() -> None:
    observation = evaluation_service._failed_case_observation(
        mode="live",
        config_snapshot={},
        error_type="QueryContentSecurityViolation",
        error_detail="用户输入被内容安全策略阻断：DIRECT_PROMPT_INJECTION",
    )

    assert observation["security"] == {
        "stage": "input",
        "action": "block",
        "reason_codes": ["DIRECT_PROMPT_INJECTION"],
        "retrieval_called": False,
    }
    assert observation["side_effects"]["external_llm"] is False


def test_historical_v2_snapshot_without_case_policy_is_not_reinterpreted() -> None:
    comparison = _comparison(delta=0.0)
    comparison["metric_deltas"]["metrics.case_outcome.final_answer_accuracy"] = {
        "delta": -1.0,
        "paired_count": len(CASE_IDS),
        "unknown_pair_count": 0,
    }
    historical_policy = {
        key: value
        for key, value in freeze_evaluation_gate_policy({}).items()
        if key != "case_pass_policy"
    }
    run = _run(historical_policy)
    run["mode"] = "live"

    gate = evaluation_service._gate(
        run,
        cases=_cases(),
        results=[],
        summary=_summary(),
        comparison=comparison,
    )

    assert gate["passed"] is True
    assert gate["case_pass_policy_version"] == "stt.eval.case.v1"


def test_case_policy_requires_correctness_to_be_the_majority_weight() -> None:
    frozen = freeze_case_pass_policy(None)

    assert frozen["policy_version"] == PREVIOUS_CASE_PASS_POLICY_VERSION
    assert frozen["answer_correctness_weight"] == 0.7
    assert sum(
        frozen[field]
        for field in (
            "answer_correctness_weight",
            "faithfulness_weight",
            "citation_coverage_weight",
            "retrieval_support_weight",
        )
    ) == pytest.approx(1.0)


def test_run_snapshot_freezes_complete_v2_policy_without_mutating_manifest() -> None:
    raw_policy = {"mrr_max_drop": 0.01, "custom_extension": "kept"}
    dataset = {
        "version": "1.0.0",
        "checksum": "dataset-checksum",
        "source": {"content_sha256": "source-checksum"},
        "manifest_json": {"gate_policy": raw_policy},
    }
    preflight = {
        "retrieval_profile": {},
        "vector_read_scopes": [{"index_version_id": "idx-v2"}],
        "vector_read_set_digest": "sha256:frozen",
        "evaluation_security": {},
    }

    snapshot = evaluation_service._snapshot(
        dataset,
        preflight,
        mode="retrieval_only",
    )

    assert raw_policy == {"mrr_max_drop": 0.01, "custom_extension": "kept"}
    assert snapshot["gate_policy"] == {
        **DEFAULT_EVALUATION_GATE_POLICY,
        "mrr_max_drop": 0.01,
        "custom_extension": "kept",
    }
    assert snapshot["vector_read_set_digest"] == "sha256:frozen"


@pytest.mark.parametrize(
    ("manifest", "expected_case_policy_version"),
    [
        ({}, PREVIOUS_CASE_PASS_POLICY_VERSION),
        (
            {
                "gate_policy": {
                    "case_pass_policy": {
                        "policy_version": PREVIOUS_CASE_PASS_POLICY_VERSION,
                    }
                }
            },
            PREVIOUS_CASE_PASS_POLICY_VERSION,
        ),
        (
            {
                "gate_policy": {
                    "case_pass_policy": {
                        "policy_version": CASE_PASS_POLICY_VERSION,
                    }
                }
            },
            CASE_PASS_POLICY_VERSION,
        ),
    ],
)
def test_run_snapshot_activates_v3_only_from_explicit_manifest_policy(
    manifest: dict,
    expected_case_policy_version: str,
) -> None:
    dataset = {
        "version": "1.0.0",
        "checksum": "dataset-checksum",
        "source": {"content_sha256": "source-checksum"},
        "manifest_json": manifest,
    }
    preflight = {
        "retrieval_profile": {},
        "vector_read_scopes": [{"index_version_id": "idx-v2"}],
        "vector_read_set_digest": "sha256:frozen",
        "evaluation_security": {},
    }

    snapshot = evaluation_service._snapshot(
        dataset,
        preflight,
        mode="live",
    )

    assert (
        snapshot["gate_policy"]["case_pass_policy"]["policy_version"]
        == expected_case_policy_version
    )
