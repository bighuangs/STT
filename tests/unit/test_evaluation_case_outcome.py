from __future__ import annotations

from app.application.evaluation_service import _case_decision
from app.domain.evaluation import (
    LEGACY_CASE_PASS_POLICY_VERSION,
    PREVIOUS_CASE_PASS_POLICY_VERSION,
    freeze_case_pass_policy,
)
from app.evaluation.metrics import aggregate_results, evaluate_case


def _yield_case() -> dict:
    # This wording is the generic form of the archived P3500-EASY-006 false
    # negative: the answer inserts a harmless grammatical particle ("约为"),
    # so exact expected-term matching used to score a correct answer as zero.
    return {
        "case_id": "generic-yield",
        "query": "标准容量耗材的平均打印量是多少？",
        "reference_answer": "约 3000 页。",
        "expected_terms": ["约 3000 页"],
        "forbidden_terms": ["5000 页"],
        "relevant_chunk_ids": ["chunk-yield"],
        "unanswerable": False,
    }


def _supported_observation(
    *,
    answer: str = "标准容量耗材的平均打印量约为 3000 页。[chunk-yield]",
    chunk_id: str = "chunk-yield",
) -> dict:
    evidence = {
        "chunk_id": chunk_id,
        "text": "标准容量耗材的平均打印量约为 3000 页。",
    }
    return {
        "answer": answer,
        "retrieved": [evidence],
        "context": [evidence],
    }


def test_answer_first_policy_fixes_grounded_semantic_false_negative() -> None:
    case = _yield_case()
    evaluated = evaluate_case(case, _supported_observation())

    generation = evaluated["metrics"]["generation"]
    assert generation["expected_term_recall"] == 0.0
    assert generation["reference_semantic_recall"] == 1.0
    assert generation["correctness"] == 1.0
    assert evaluated["metrics"]["generation"]["faithfulness"] == 1.0
    assert evaluated["metrics"]["citation"]["coverage"] == 1.0

    legacy = _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy={"policy_version": LEGACY_CASE_PASS_POLICY_VERSION},
    )
    current = _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy=freeze_case_pass_policy(None),
    )

    assert legacy["passed"] is False
    assert "expected_term_recall_below_minimum" in legacy["reason_codes"]
    assert current["passed"] is True
    assert current["final_answer_correctness"] == 1.0
    assert current["final_answer_correct"] is True
    assert current["final_answer_accuracy"] == 1.0
    assert current["strict_supported_case_pass"] is True
    assert current["quality_score"] == 1.0
    assert current["weights"]["answer_correctness"] == 0.7


def test_correct_but_unsupported_answer_never_passes() -> None:
    case = _yield_case()
    evaluated = evaluate_case(
        case,
        {
            "answer": "标准容量耗材的平均打印量约为 3000 页。",
            "retrieved": [],
            "context": [],
        },
    )

    assert evaluated["metrics"]["generation"]["correctness"] == 1.0
    outcome = _case_decision(case, evaluated, had_error=False)

    assert outcome["passed"] is False
    assert outcome["final_answer_correctness"] == 1.0
    assert outcome["final_answer_correct"] is True
    assert outcome["final_answer_accuracy"] == 1.0
    assert outcome["strict_supported_case_pass"] is False
    assert "faithfulness_below_minimum" in outcome["reason_codes"]
    assert "citation_coverage_below_minimum" in outcome["reason_codes"]
    evaluated["metrics"]["case_outcome"] = outcome
    summary = aggregate_results([evaluated])
    assert summary["metrics"]["case_outcome"]["final_answer_correctness"]["mean"] == 1.0
    assert summary["metrics"]["case_outcome"]["final_answer_accuracy"]["mean"] == 1.0
    assert summary["metrics"]["case_outcome"]["strict_supported_case_pass_rate"]["mean"] == 0.0


def _unanswerable_case() -> dict:
    return {
        "case_id": "generic-unanswerable",
        "query": "未声明的规格是多少？",
        "reference_answer": "资料未给出该规格，因此无法确定。",
        "unanswerable": True,
        "expected_terms": ["未给出", "无法确定"],
        "forbidden_terms": [],
        "relevant_evidence": [],
    }


def _v2_degraded_abstention_decision(answer: str, *, abstained: bool = True) -> dict:
    case = _unanswerable_case()
    observation = {
        "answer": answer,
        "abstained": abstained,
        "retrieved": [],
        "context": [],
        "visible_answer_policy": {
            "strict_grounding_eligible": False,
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        },
    }
    evaluated = evaluate_case(case, observation)
    return _case_decision(
        case,
        evaluated,
        had_error=False,
        case_policy={"policy_version": PREVIOUS_CASE_PASS_POLICY_VERSION},
        observation=observation,
    )


def test_v2_correct_fact_free_abstention_ignores_answerable_only_strictness() -> None:
    outcome = _v2_degraded_abstention_decision("资料未给出该规格，因此无法确定。")

    assert outcome["policy_version"] == PREVIOUS_CASE_PASS_POLICY_VERSION
    assert outcome["final_answer_correct"] is True
    assert outcome["strict_supported_case_pass"] is True
    assert outcome["passed"] is True
    assert outcome["reason_codes"] == []


def test_v2_incorrect_refusal_does_not_receive_strictness_exemption() -> None:
    outcome = _v2_degraded_abstention_decision(
        "资料未给出该规格，因此无法确定。",
        abstained=False,
    )

    assert outcome["final_answer_correct"] is False
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["passed"] is False
    assert "abstention_incorrect" in outcome["reason_codes"]


def test_v2_factual_abstention_does_not_receive_strictness_exemption() -> None:
    outcome = _v2_degraded_abstention_decision("资料未给出该规格，因此无法确定；设备支持卫星通信。")

    # v2 的旧正确性合同尚未把残留事实主张计入 final_answer_correct；显示层
    # 守卫仍必须拒绝放行，避免回移豁免扩大旧合同的安全缺口。
    assert outcome["final_answer_correct"] is True
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["passed"] is False
    assert "visible_answer_not_strict_grounding_eligible" in outcome["reason_codes"]


def test_completeness_preserved_answer_can_be_correct_but_never_strict() -> None:
    case = _yield_case()
    observation = {
        **_supported_observation(),
        "answer_decision": {
            "action": "PARTIAL_ANSWER",
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        },
        "visible_answer_policy": {
            "schema_version": "visible-answer-policy.v1",
            "mode": "safe_pre_degraded",
            "status": "degraded",
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
            "visible_answer_sha256": "a" * 64,
            "rejected_post_answer_sha256": "b" * 64,
            "trusted_citation_count": 1,
            "removed_untrusted_marker_count": 1,
            "strict_grounding_eligible": False,
        },
    }
    evaluated = evaluate_case(case, observation)

    # The evaluator projection makes the query-state audit durable even for
    # direct callers that decide the case after the observation is evaluated.
    assert evaluated["visible_answer_policy"] == observation["visible_answer_policy"]
    outcome = _case_decision(case, evaluated, had_error=False)

    assert outcome["final_answer_correctness"] == 1.0
    assert outcome["final_answer_correct"] is True
    assert outcome["final_answer_accuracy"] == 1.0
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["strict_supported_case_pass_rate"] == 0.0
    assert outcome["passed"] is False
    assert outcome["pass_rate"] == 0.0
    assert "visible_answer_not_strict_grounding_eligible" in outcome["reason_codes"]
    assert outcome["answer_display_audit"] == {
        "status": "degraded",
        "badge_code": "ANSWER_COMPLETENESS_PRESERVED_CITATION_DEGRADED",
        "label": "答案完整性优先，部分内容未通过引用校验",
        "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        "strict_grounding_eligible": False,
    }

    evaluated["metrics"]["case_outcome"] = outcome
    summary = aggregate_results([evaluated])
    assert summary["metrics"]["case_outcome"]["final_answer_accuracy"]["mean"] == 1.0
    assert summary["metrics"]["case_outcome"]["strict_supported_case_pass_rate"]["mean"] == 0.0


def test_completeness_reason_alone_forces_strict_false_without_changing_accuracy() -> None:
    case = _yield_case()
    observation = {
        **_supported_observation(),
        "answer_decision": {
            "action": "PARTIAL_ANSWER",
            "reason_code": "POST_GENERATION_COMPLETENESS_PRESERVED",
        },
    }
    evaluated = evaluate_case(case, observation)
    outcome = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=observation,
    )

    assert outcome["final_answer_correct"] is True
    assert outcome["final_answer_accuracy"] == 1.0
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["answer_display_audit"]["reason_code"] == (
        "POST_GENERATION_COMPLETENESS_PRESERVED"
    )


def test_explicit_strict_grounding_ineligible_flag_is_independently_authoritative() -> None:
    case = _yield_case()
    observation = {
        **_supported_observation(),
        "visible_answer_policy": {
            "schema_version": "visible-answer-policy.v1",
            "reason_code": "GENERIC_CITATION_DEGRADATION",
            "strict_grounding_eligible": False,
        },
    }
    evaluated = evaluate_case(case, observation)
    outcome = _case_decision(case, evaluated, had_error=False)

    assert outcome["final_answer_correct"] is True
    assert outcome["final_answer_accuracy"] == 1.0
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["answer_display_audit"]["reason_code"] == ("GENERIC_CITATION_DEGRADATION")


def test_missing_visible_answer_policy_keeps_legacy_observation_decision_unchanged() -> None:
    case = _yield_case()
    legacy_observation = _supported_observation()
    evaluated = evaluate_case(case, legacy_observation)

    assert "visible_answer_policy" not in evaluated
    outcome = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=legacy_observation,
    )

    assert outcome["passed"] is True
    assert outcome["strict_supported_case_pass"] is True
    assert outcome["final_answer_correct"] is True
    assert "answer_display_audit" not in outcome


def test_retrieval_label_miss_is_diagnostic_when_answer_is_independently_grounded() -> None:
    case = _yield_case()
    evaluated = evaluate_case(
        case,
        _supported_observation(
            answer="标准容量耗材的平均打印量约为 3000 页。[observed-chunk]",
            chunk_id="observed-chunk",
        ),
    )

    assert evaluated["metrics"]["retrieval"]["recall_at_k"]["10"] == 0.0
    assert evaluated["metrics"]["generation"]["faithfulness"] == 1.0
    assert evaluated["metrics"]["citation"]["coverage"] == 1.0
    outcome = _case_decision(case, evaluated, had_error=False)

    assert outcome["passed"] is True
    assert outcome["strict_supported_case_pass"] is True
    assert outcome["retrieval_support"] == 0.0
    assert outcome["diagnostic_codes"] == ["retrieval_ground_truth_miss"]


def test_grounded_but_wrong_answer_cannot_pass_on_support_metrics() -> None:
    case = _yield_case()
    evidence = {
        "chunk_id": "chunk-yield",
        "text": "另一个耗材的平均打印量约为 5000 页。",
    }
    evaluated = evaluate_case(
        case,
        {
            "answer": "平均打印量约为 5000 页。[chunk-yield]",
            "retrieved": [evidence],
            "context": [evidence],
        },
    )
    outcome = _case_decision(case, evaluated, had_error=False)

    assert evaluated["metrics"]["generation"]["faithfulness"] == 1.0
    assert outcome["answer_correctness"] == 0.0
    assert outcome["final_answer_correctness"] == 0.0
    assert outcome["final_answer_correct"] is False
    assert outcome["final_answer_accuracy"] == 0.0
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["passed"] is False
    assert "answer_correctness_below_minimum" in outcome["reason_codes"]


def test_expected_term_overlap_cannot_hide_reversed_requirement() -> None:
    case = {
        "case_id": "generic-polarity",
        "query": "操作前应如何处理安全盖？",
        "reference_answer": "操作前必须打开安全盖。",
        "expected_terms": ["打开安全盖"],
        "relevant_chunk_ids": ["chunk-polarity"],
        "unanswerable": False,
    }
    evidence = {
        "chunk_id": "chunk-polarity",
        "text": "操作前不得打开安全盖。",
    }
    evaluated = evaluate_case(
        case,
        {
            "answer": "操作前不得打开安全盖。[chunk-polarity]",
            "retrieved": [evidence],
            "context": [evidence],
        },
    )
    outcome = _case_decision(case, evaluated, had_error=False)

    assert evaluated["metrics"]["generation"]["expected_term_recall"] == 1.0
    assert evaluated["metrics"]["generation"]["reference_constraints_preserved"] is False
    assert outcome["answer_correctness"] == 0.0
    assert outcome["final_answer_correctness"] == 0.0
    assert outcome["final_answer_correct"] is False
    assert outcome["final_answer_accuracy"] == 0.0
    assert outcome["strict_supported_case_pass"] is False
    assert outcome["passed"] is False
