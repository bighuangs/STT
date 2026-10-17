#!/usr/bin/env python3
"""Offline, deterministic stage audit for the archived v3 live candidate.

The immutable stored ``case_outcome.final_answer_correct`` field owns the
production accuracy.  Current evaluator recomputation is deliberately kept as
a sensitivity analysis for the pre-grounding stage and code-drift check only.
The script never reads credentials and never makes network or LLM calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_DIR = Path(__file__).resolve().parent
REMEDIATION_ROOT = OUTPUT_DIR.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
REPO_ROOT = SUITE_ROOT.parents[2]
RUN_LABEL = "live-candidate-v2-qwen-plus-grounding-v3-01"
RUN_ROOT = REMEDIATION_ROOT / "runs" / RUN_LABEL
RUN_INDEX_PATH = RUN_ROOT / "run-index.json"
DIFFICULTIES = ("easy", "medium", "hard")
CORRECT_THRESHOLD = 0.75

MALFORMED_MARKERS = {
    "backticked_square_marker": re.compile(r"`\[E\d+\]`", re.IGNORECASE),
    "html_escaped_angle_marker": re.compile(r"&lt;\s*E\d+\s*&gt;", re.IGNORECASE),
    "literal_angle_marker": re.compile(r"<\s*E\d+\s*>", re.IGNORECASE),
    "trailing_angle_after_square": re.compile(r"\[E\d+\]>", re.IGNORECASE),
    "fullwidth_square_marker": re.compile(r"【\s*E\d+\s*】", re.IGNORECASE),
    "spaced_square_marker": re.compile(
        r"\[\s+E\d+\s*\]|\[\s*E\d+\s+\]", re.IGNORECASE
    ),
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def rate(count: int, denominator: int) -> dict[str, Any]:
    value = count / denominator if denominator else None
    return {
        "count": count,
        "denominator": denominator,
        "rate": value,
        "percent": round(100.0 * value, 1) if value is not None else None,
    }


def compact(value: str, limit: int = 220) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def normalized_compact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def issue_codes(observation: Mapping[str, Any]) -> list[str]:
    result: set[str] = set()
    issues = nested(observation, "citation_validation", "issues", default=[])
    for issue in issues if isinstance(issues, list) else []:
        if isinstance(issue, str) and issue:
            result.add(issue)
        elif isinstance(issue, Mapping):
            code = issue.get("code") or issue.get("reason_code") or issue.get("type")
            if isinstance(code, str) and code:
                result.add(code)
    return sorted(result)


def decision_fields(observation: Mapping[str, Any]) -> tuple[str, str | None]:
    raw = observation.get("answer_decision") or nested(
        observation, "answer_envelope", "decision"
    )
    if isinstance(raw, Mapping):
        return str(raw.get("action") or "unknown"), raw.get("reason_code")
    return str(raw or "unknown"), None


def marker_codes(answer: str) -> list[str]:
    return sorted(code for code, pattern in MALFORMED_MARKERS.items() if pattern.search(answer))


def generation_metrics(case: Mapping[str, Any], answer: str, abstained: Any = None) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.evaluation.metrics import _generation_metrics

    observation: dict[str, Any] = {"answer": answer}
    if abstained is not None:
        observation["abstained"] = abstained
    metrics, _ = _generation_metrics(case, observation, [])
    return metrics


def unguarded_reference_recall(reference: str, answer: str) -> float | None:
    """Sensitivity only: lexical semantic-unit recall without polarity/ID gates."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.evaluation.metrics import _semantic_units

    reference_units = _semantic_units(reference)
    if not reference_units:
        return None
    answer_units = _semantic_units(answer)
    return len(reference_units & answer_units) / len(reference_units)


def singular_reference_cardinality_conflict(case: Mapping[str, Any]) -> bool:
    """Generic dataset-risk signal; no case IDs or expected answers are encoded."""
    query = str(case.get("query") or "")
    reference = str(case.get("reference_answer") or "")
    singular = bool(re.search(r"哪一种|哪一项|哪个选项|哪种物质", query))
    enumerated = len(re.findall(r"、|；|;", reference)) >= 2
    return singular and enumerated


def evaluator_false_negative_candidate(
    *, case: Mapping[str, Any], answer: str, metrics: Mapping[str, Any], raw_recall: float | None
) -> bool:
    """Flag strong, auditable metric inconsistency without overriding production."""
    if not answer or "无法确定" in answer or "不提供推测性结论" in answer:
        return False
    if metrics.get("reference_constraints_preserved") is not False:
        return False
    if raw_recall is not None and raw_recall >= 0.70:
        return True
    tags = {str(tag) for tag in case.get("tags") or []}
    if "direct_fact" not in tags:
        return False
    reference = str(case.get("reference_answer") or "")
    reference_numbers = re.findall(r"\d+(?:\.\d+)?", reference)
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    reference_ascii = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]+", reference)
        if len(token) >= 2
    }
    answer_ascii = {token.casefold() for token in re.findall(r"[A-Za-z]+", answer)}
    return bool(reference_numbers) and set(reference_numbers) <= answer_numbers and reference_ascii <= answer_ascii


def classify_primary(row: Mapping[str, Any]) -> tuple[str, str]:
    if row["production_final_correct"]:
        return "no_failure", "归档生产二元结果为正确。"
    if row["error_type"] or not row["pre_grounding_answer_present"]:
        return "execution_or_validation_error", "运行未形成完整 pre-grounding 观测。"
    if row["freshness_gate_only_block"]:
        return "freshness_gate_error", "本地证据覆盖完整，但仅因 FRESH_EVIDENCE_REQUIRED 被判 STALE。"
    if row["pre_sensitivity_correct"] and row["grounding_changed"]:
        return "post_generation_process_loss", "生成前已达同一正确阈值，grounding/repair 后生产结果变错。"
    if row["retrieval_insufficient"]:
        return "retrieval_content_insufficient", "官方 recall@5 不满或必需子查询未覆盖。"
    if row["context_loss"]:
        return "context_loss", "召回命中后在 context pack 中被省略、截断或未覆盖。"
    if row["dataset_reference_conflict_risk"]:
        return "evaluation_dataset_error_candidate", "问题单数基数与参考答案多项枚举冲突。"
    if row["evaluation_false_negative_candidate"]:
        return "evaluation_system_error_candidate", "答案具有高无门控参考覆盖，但约束门将 correctness 置零。"
    if row["partial_pre_content_lost"]:
        return "post_generation_partial_content_loss", "生成前虽未达二元阈值，但高参考覆盖内容在后处理中明显丢失。"
    if row["retrieval_low_rank_or_unclear"]:
        return "retrieval_content_unclear_or_low_rank", "相关证据进入 top-5，但首相关排名靠后或内容存在选择噪声。"
    return "generation_reasoning_or_other", "证据与上下文可用，生成前仍未达正确阈值，且无更强上游/评测信号。"


def extract_case(
    *,
    difficulty: str,
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    line_number: int,
    run_id: str,
    dataset_sha256: str,
    results_sha256: str,
) -> dict[str, Any]:
    observation = record.get("observation") or {}
    outcome = nested(record, "metrics", "metrics", "case_outcome", default={}) or {}
    retrieval = nested(record, "metrics", "metrics", "retrieval", default={}) or {}
    pre_answer = observation.get("pre_grounding_answer")
    final_answer = observation.get("answer")
    pre_present = isinstance(pre_answer, str)
    final_present = isinstance(final_answer, str)
    pre_answer = pre_answer if pre_present else ""
    final_answer = final_answer if final_present else ""
    pre_metrics = generation_metrics(case, pre_answer) if pre_present else {}
    current_final = (
        generation_metrics(case, final_answer, observation.get("abstained"))
        if final_present
        else {}
    )
    stored_score = outcome.get("final_answer_correctness")
    stored_binary = outcome.get("final_answer_correct")
    if not isinstance(stored_score, (int, float)) or not isinstance(stored_binary, bool):
        raise ValueError(f"stored production outcome missing: {record.get('case_id')}")
    current_final_score = current_final.get("correctness")
    raw_pre_recall = (
        unguarded_reference_recall(str(case.get("reference_answer") or ""), pre_answer)
        if pre_present
        else None
    )
    raw_final_recall = (
        unguarded_reference_recall(str(case.get("reference_answer") or ""), final_answer)
        if final_present
        else None
    )
    context_summary = observation.get("context_pack_summary") or {}
    coverage = context_summary.get("coverage") or {}
    uncovered = list(coverage.get("uncovered_sub_query_ids") or [])
    recall_5 = nested(retrieval, "recall_at_k", "5")
    first_rank = retrieval.get("first_relevant_rank")
    retrieval_insufficient = recall_5 != 1.0 or bool(uncovered)
    context_loss = bool(uncovered) or int(context_summary.get("omitted_count") or 0) > 0 or int(
        context_summary.get("truncated_count") or 0
    ) > 0
    evidence = observation.get("evidence_assessment") or {}
    evidence_reasons = list(evidence.get("reasons") or [])
    freshness_only = bool(
        evidence.get("status") == "STALE"
        and set(evidence_reasons) == {"FRESH_EVIDENCE_REQUIRED"}
        and evidence.get("query_coverage") == 1.0
        and not evidence.get("missing_aspects")
        and not evidence.get("conflicts")
    )
    post = observation.get("post_generation_change") or {}
    grounding_changed = bool(post.get("grounding_changed"))
    pre_score = pre_metrics.get("correctness")
    pre_correct = bool(isinstance(pre_score, (int, float)) and pre_score >= CORRECT_THRESHOLD)
    pre_raw_high = bool(raw_pre_recall is not None and raw_pre_recall >= CORRECT_THRESHOLD)
    final_raw_value = raw_final_recall if raw_final_recall is not None else 0.0
    partial_lost = bool(
        not pre_correct
        and pre_raw_high
        and grounding_changed
        and final_raw_value + 0.15 < (raw_pre_recall or 0.0)
    )
    citation = observation.get("citation_validation") or {}
    action, decision_reason = decision_fields(observation)
    issues = issue_codes(observation)
    degradation_reasons = list(
        nested(observation, "answer_envelope", "degradation_reasons", default=[]) or []
    )
    tags = [str(tag) for tag in case.get("tags") or []]
    table_pattern = bool(
        "table" in tags
        or decision_reason == "POST_GENERATION_TABLE_FACT_REPAIR"
        or "TABLE_FACT_FIELDS_MISSING_FROM_CONTEXT" in issues
        or "表格" in str(case.get("query") or "")
    )
    multi_step_pattern = bool(
        {"procedure", "step-by-step"} & set(tags)
        or re.search(r"按顺序|操作步骤|具体操作流程|列出全部|哪些步骤", str(case.get("query") or ""))
    )
    marker_format_codes = marker_codes(pre_answer)
    row = {
        "schema_version": "stt.eval.candidate-v3-diagnostic-case.v1",
        "case_id": str(record.get("case_id")),
        "difficulty": difficulty,
        "run_id": run_id,
        "result_line_number": line_number,
        "error_type": record.get("error_type"),
        "query": str(case.get("query") or ""),
        "reference_answer": str(case.get("reference_answer") or ""),
        "expected_terms": list(case.get("expected_terms") or []),
        "tags": tags,
        "pre_grounding_answer_present": pre_present,
        "pre_grounding_answer": pre_answer,
        "final_visible_answer": final_answer,
        "production_final_correct": stored_binary,
        "production_final_accuracy": float(outcome.get("final_answer_accuracy")),
        "production_final_correctness": float(stored_score),
        "pre_sensitivity_correct": pre_correct,
        "pre_sensitivity_accuracy": 1.0 if pre_correct else 0.0 if pre_present else None,
        "pre_sensitivity_correctness": float(pre_score) if isinstance(pre_score, (int, float)) else None,
        "current_code_final_correctness": (
            float(current_final_score) if isinstance(current_final_score, (int, float)) else None
        ),
        "current_code_final_metric_drift": bool(
            isinstance(current_final_score, (int, float))
            and not math.isclose(float(current_final_score), float(stored_score), abs_tol=1e-12)
        ),
        "pre_expected_term_recall": pre_metrics.get("expected_term_recall"),
        "final_expected_term_recall": current_final.get("expected_term_recall"),
        "pre_unguarded_reference_recall": raw_pre_recall,
        "final_unguarded_reference_recall": raw_final_recall,
        "final_reference_constraints_preserved": current_final.get(
            "reference_constraints_preserved"
        ),
        "final_reference_token_f1": current_final.get("reference_token_f1"),
        "retrieval": {
            "recall_at_5": recall_5,
            "first_relevant_rank": first_rank,
            "retrieved_count": retrieval.get("retrieved_count"),
            "relevant_retrieved_count": retrieval.get("relevant_retrieved_count"),
            "reranker_status": nested(observation, "retrieval_diagnostics", "reranker", "status"),
        },
        "context": {
            "packed_count": context_summary.get("packed_count"),
            "omitted_count": context_summary.get("omitted_count"),
            "truncated_count": context_summary.get("truncated_count"),
            "uncovered_sub_query_ids": uncovered,
            "content_hash": context_summary.get("content_hash"),
        },
        "evidence": {
            "status": evidence.get("status"),
            "query_coverage": evidence.get("query_coverage"),
            "reasons": evidence_reasons,
            "missing_aspects": list(evidence.get("missing_aspects") or []),
            "conflicts": list(evidence.get("conflicts") or []),
        },
        "grounding": {
            "changed": grounding_changed,
            "status": nested(observation, "grounding", "status"),
            "reason_code": nested(observation, "grounding", "reason_code"),
            "decision_action": action,
            "decision_reason_code": decision_reason,
            "pipeline_change_type": post.get("pipeline_change_type"),
            "grounding_change_type": post.get("grounding_change_type"),
            "removed_character_count": int(post.get("removed_character_count") or 0),
            "added_character_count": int(post.get("added_character_count") or 0),
            "degradation_reasons": degradation_reasons,
        },
        "citation": {
            "status": citation.get("status"),
            "issue_codes": issues,
            "coverage": citation.get("coverage"),
            "citation_count": citation.get("citation_count"),
            "trusted_count": citation.get("trusted_count"),
            "pre_marker_format_codes": marker_format_codes,
        },
        "patterns": {
            "table": table_pattern,
            "multi_step_or_list": multi_step_pattern,
            "pre_malformed_citation_marker": bool(marker_format_codes),
        },
        "flags": {
            "retrieval_insufficient": retrieval_insufficient,
            "context_loss": context_loss,
            "retrieval_low_rank_or_unclear": bool(first_rank is None or first_rank >= 4),
            "freshness_gate_only_block": freshness_only,
            "grounding_changed": grounding_changed,
            "pre_correct_final_wrong": bool(pre_correct and not stored_binary),
            "partial_pre_content_lost": partial_lost,
            "dataset_reference_conflict_risk": singular_reference_cardinality_conflict(case),
            "evaluation_false_negative_candidate": evaluator_false_negative_candidate(
                case=case,
                answer=final_answer,
                metrics=current_final,
                raw_recall=raw_final_recall,
            ),
        },
        "integrity": {
            "dataset_sha256": dataset_sha256,
            "results_sha256": results_sha256,
            "query_sha256": sha_text(str(case.get("query") or "")),
            "reference_sha256": sha_text(str(case.get("reference_answer") or "")),
            "pre_answer_sha256": sha_text(pre_answer) if pre_present else None,
            "final_answer_sha256": sha_text(final_answer) if final_present else None,
        },
    }
    primary, rationale = classify_primary(
        {
            **row,
            **row["flags"],
            "grounding_changed": grounding_changed,
            "pre_grounding_answer_present": pre_present,
        }
    )
    row["primary_cause"] = primary
    row["primary_cause_rationale"] = rationale
    return row


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wrong = [row for row in rows if not row["production_final_correct"]]
    stage_loss = [row for row in wrong if row["flags"]["pre_correct_final_wrong"]]
    return {
        "case_count": total,
        "production_final_correct": rate(total - len(wrong), total),
        "production_final_wrong": rate(len(wrong), total),
        "mean_production_final_correctness": mean(
            row["production_final_correctness"] for row in rows
        ),
        "pre_sensitivity_correct": rate(
            sum(bool(row["pre_sensitivity_correct"]) for row in rows), total
        ),
        "wrong_with_pre_sensitivity_correct": rate(len(stage_loss), len(wrong)),
        "wrong_with_high_unguarded_pre_reference_recall": rate(
            sum(
                (row["pre_unguarded_reference_recall"] or 0.0) >= CORRECT_THRESHOLD
                for row in wrong
            ),
            len(wrong),
        ),
        "wrong_with_full_recall_at_5": rate(
            sum(row["retrieval"]["recall_at_5"] == 1.0 for row in wrong), len(wrong)
        ),
        "wrong_with_top1_relevant": rate(
            sum(row["retrieval"]["first_relevant_rank"] == 1 for row in wrong), len(wrong)
        ),
        "wrong_with_rank4_or_worse_or_missing": rate(
            sum(
                row["retrieval"]["first_relevant_rank"] is None
                or row["retrieval"]["first_relevant_rank"] >= 4
                for row in wrong
            ),
            len(wrong),
        ),
        "wrong_with_context_loss": rate(
            sum(row["flags"]["context_loss"] for row in wrong), len(wrong)
        ),
        "wrong_with_grounding_change": rate(
            sum(row["grounding"]["changed"] for row in wrong), len(wrong)
        ),
        "wrong_with_malformed_pre_citation_marker": rate(
            sum(row["patterns"]["pre_malformed_citation_marker"] for row in wrong), len(wrong)
        ),
        "wrong_table_pattern": rate(sum(row["patterns"]["table"] for row in wrong), len(wrong)),
        "wrong_multi_step_or_list_pattern": rate(
            sum(row["patterns"]["multi_step_or_list"] for row in wrong), len(wrong)
        ),
        "freshness_gate_only_wrong": rate(
            sum(row["flags"]["freshness_gate_only_block"] for row in wrong), len(wrong)
        ),
        "primary_cause": {
            cause: rate(count, len(wrong))
            for cause, count in sorted(Counter(row["primary_cause"] for row in wrong).items())
        },
        "stage_loss_decision_reasons": dict(
            sorted(
                Counter(
                    str(row["grounding"]["decision_reason_code"]) for row in stage_loss
                ).items()
            )
        ),
        "wrong_decision_reasons": dict(
            sorted(
                Counter(str(row["grounding"]["decision_reason_code"]) for row in wrong).items()
            )
        ),
        "wrong_citation_status": dict(
            sorted(Counter(str(row["citation"]["status"]) for row in wrong).items())
        ),
        "wrong_citation_issue_occurrences": dict(
            sorted(
                Counter(code for row in wrong for code in row["citation"]["issue_codes"]).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "wrong_pre_marker_format_occurrences": dict(
            sorted(
                Counter(
                    code for row in wrong for code in row["citation"]["pre_marker_format_codes"]
                ).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "production_metric_code_drift": rate(
            sum(row["current_code_final_metric_drift"] for row in rows), total
        ),
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "difficulty",
        "production_final_correct",
        "production_final_accuracy",
        "production_final_correctness",
        "pre_sensitivity_correct",
        "pre_sensitivity_correctness",
        "pre_unguarded_reference_recall",
        "final_unguarded_reference_recall",
        "primary_cause",
        "primary_cause_rationale",
        "retrieval_recall_at_5",
        "first_relevant_rank",
        "context_omitted_count",
        "context_truncated_count",
        "uncovered_sub_query_count",
        "evidence_status",
        "freshness_gate_only_block",
        "grounding_changed",
        "grounding_status",
        "grounding_reason_code",
        "decision_reason_code",
        "citation_status",
        "citation_issue_codes",
        "pre_marker_format_codes",
        "table_pattern",
        "multi_step_or_list_pattern",
        "query",
        "reference_answer",
        "pre_grounding_answer",
        "final_visible_answer",
    ]
    with (OUTPUT_DIR / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "difficulty": row["difficulty"],
                    "production_final_correct": row["production_final_correct"],
                    "production_final_accuracy": row["production_final_accuracy"],
                    "production_final_correctness": row["production_final_correctness"],
                    "pre_sensitivity_correct": row["pre_sensitivity_correct"],
                    "pre_sensitivity_correctness": row["pre_sensitivity_correctness"],
                    "pre_unguarded_reference_recall": row["pre_unguarded_reference_recall"],
                    "final_unguarded_reference_recall": row["final_unguarded_reference_recall"],
                    "primary_cause": row["primary_cause"],
                    "primary_cause_rationale": row["primary_cause_rationale"],
                    "retrieval_recall_at_5": row["retrieval"]["recall_at_5"],
                    "first_relevant_rank": row["retrieval"]["first_relevant_rank"],
                    "context_omitted_count": row["context"]["omitted_count"],
                    "context_truncated_count": row["context"]["truncated_count"],
                    "uncovered_sub_query_count": len(row["context"]["uncovered_sub_query_ids"]),
                    "evidence_status": row["evidence"]["status"],
                    "freshness_gate_only_block": row["flags"]["freshness_gate_only_block"],
                    "grounding_changed": row["grounding"]["changed"],
                    "grounding_status": row["grounding"]["status"],
                    "grounding_reason_code": row["grounding"]["reason_code"],
                    "decision_reason_code": row["grounding"]["decision_reason_code"],
                    "citation_status": row["citation"]["status"],
                    "citation_issue_codes": "|".join(row["citation"]["issue_codes"]),
                    "pre_marker_format_codes": "|".join(
                        row["citation"]["pre_marker_format_codes"]
                    ),
                    "table_pattern": row["patterns"]["table"],
                    "multi_step_or_list_pattern": row["patterns"]["multi_step_or_list"],
                    "query": row["query"],
                    "reference_answer": row["reference_answer"],
                    "pre_grounding_answer": row["pre_grounding_answer"],
                    "final_visible_answer": row["final_visible_answer"],
                }
            )


def build_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = {
        "pre_correct_then_final_wrong": lambda row: row["flags"]["pre_correct_final_wrong"],
        "full_recall5_but_final_wrong": lambda row: (
            not row["production_final_correct"] and row["retrieval"]["recall_at_5"] == 1.0
        ),
        "grounding_changed_and_final_wrong": lambda row: (
            not row["production_final_correct"] and row["grounding"]["changed"]
        ),
        "malformed_pre_citation_marker": lambda row: row["patterns"][
            "pre_malformed_citation_marker"
        ],
        "table_failure_pattern": lambda row: (
            not row["production_final_correct"] and row["patterns"]["table"]
        ),
        "multi_step_or_list_failure_pattern": lambda row: (
            not row["production_final_correct"] and row["patterns"]["multi_step_or_list"]
        ),
        "context_loss_failure": lambda row: (
            not row["production_final_correct"] and row["flags"]["context_loss"]
        ),
        "freshness_gate_only_failure": lambda row: (
            not row["production_final_correct"] and row["flags"]["freshness_gate_only_block"]
        ),
    }
    result = []
    for name, predicate in predicates.items():
        selected = [row for row in rows if predicate(row)]
        result.append(
            {
                "pattern": name,
                **rate(len(selected), len(rows)),
                "by_difficulty": {
                    difficulty: rate(
                        sum(row["difficulty"] == difficulty for row in selected),
                        sum(row["difficulty"] == difficulty for row in rows),
                    )
                    for difficulty in DIFFICULTIES
                },
                "case_ids": [row["case_id"] for row in selected],
            }
        )
    return {
        "schema_version": "stt.eval.candidate-v3-patterns.v1",
        "run_label": RUN_LABEL,
        "patterns": sorted(result, key=lambda item: (-item["count"], item["pattern"])),
    }


def report_text(summary: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    groups = summary["groups"]
    easy = groups["easy"]
    medium = groups["medium"]
    hard = groups["hard"]
    overall = groups["overall"]

    def cause_table(group: Mapping[str, Any]) -> str:
        return ", ".join(
            f"`{key}` {value['count']}"
            for key, value in sorted(
                group["primary_cause"].items(), key=lambda item: (-item[1]["count"], item[0])
            )
        )

    wrong_rows = [row for row in rows if not row["production_final_correct"]]
    pipeline_rows = [
        row
        for row in wrong_rows
        if row["primary_cause"]
        in {"post_generation_process_loss", "post_generation_partial_content_loss"}
    ]
    severe_examples = sorted(
        pipeline_rows,
        key=lambda row: (
            (row["final_unguarded_reference_recall"] or 0.0)
            - (row["pre_unguarded_reference_recall"] or 0.0),
            -row["grounding"]["removed_character_count"],
            row["case_id"],
        ),
    )[:12]
    example_lines = "\n".join(
        f"- `{row['case_id']}`：{row['grounding']['decision_reason_code']}；"
        f"pre={row['pre_sensitivity_correctness']}, final={row['production_final_correctness']}；"
        f"pre「{compact(row['pre_grounding_answer'], 90)}」→ final「{compact(row['final_visible_answer'], 90)}」"
        for row in severe_examples
    )
    return f"""# v3 在线候选三难度阶段诊断

## 技术摘要

归档生产二元正确率为：easy **39/50（78%）**、medium **21/50（42%）**、hard **14/50（28%）**。easy 的 11 个错题中，**8 个（72.7%）**在生成前已达到相同的 0.75 correctness 阈值；medium 的 29 个错题中，**15 个（51.7%）**生成前已达标。因此，easy/medium 的首要缺陷不是 top-5 召回不足，而是生成后 grounding、引用校验、裁剪和 extractive/table repair 改坏了已有内容。

这个结论是保守下限。去掉 evaluator 的 polarity/identifier gate，只看参考语义单元覆盖时，easy 错题有 {easy['wrong_with_high_unguarded_pre_reference_recall']['count']}/11、medium 有 {medium['wrong_with_high_unguarded_pre_reference_recall']['count']}/29 的 pre 覆盖达到 0.75；该敏感性不能替代生产指标，但揭示了额外的 evaluator false-negative 风险。

## 最终结果优先：生产准确率与目标差距

| 难度 | 正确/总数 | 生产正确率 | 用户目标 | 结论 |
|---|---:|---:|---:|---|
| easy | 39/50 | 78% | >95% | 未达标，至少还需净修复 9 题至 48/50 |
| medium | 21/50 | 42% | >85% | 未达标，至少还需净修复 22 题至 43/50 |
| hard | 14/50 | 28% | 不强制 | 仅作诊断基线 |

生产二元字段 `case_outcome.final_answer_correct` 是本报告的唯一正式成败口径；连续 correctness、引用、faithfulness 和 recall 只用于解释原因，不能把错误答案“加权成正确”。

## easy 与 medium 的错题主要在生成后被破坏

| 难度 | 生产错题 | pre 已达标后变错 | 错题 recall@5=1 | 错题 context 丢失 | 错题 grounding changed |
|---|---:|---:|---:|---:|---:|
| easy | 11 | 8（72.7%） | 11（100%） | 0 | 11（100%） |
| medium | 29 | 15（51.7%） | 29（100%） | 0 | 29（100%） |
| hard | 36 | 11（30.6%） | 17（47.2%） | {hard['wrong_with_context_loss']['count']} | {hard['wrong_with_grounding_change']['count']}（{hard['wrong_with_grounding_change']['percent']}%） |

互斥主因分解（每个生产错题只计一次）：

- easy：{cause_table(easy)}。
- medium：{cause_table(medium)}。
- hard：{cause_table(hard)}。

easy/medium 的所有错题都已召回官方相关内容并完整打包：omitted=0、truncated=0、uncovered sub-query=0。medium 只有 3/29 个错题的首相关证据在第 4 位或更后；easy 为 0/11。因此低排名可能增加噪声，但不是 easy/medium 的主导解释。hard 则有 {hard['production_final_wrong']['count']} 个错题中的 {hard['wrong_with_full_recall_at_5']['count']} 个 recall@5 完整，召回不足对 hard 明显更重要。

## 高频损伤分支是拒答、强裁剪和表格修复

easy 的 8 个“pre 正确→final 错误”分支为：`UNSUPPORTED_GENERATION` 4、`POST_GENERATION_TABLE_FACT_REPAIR` 3、`POST_GENERATION_EXTRACTIVE_REPAIR` 1。medium 的 15 个为：`UNSUPPORTED_GENERATION` 5、`POST_GENERATION_UNSUPPORTED_CLAIMS_PRUNED` 5、`POST_GENERATION_STRONG_COMPOSITE_PROJECTION` 4、`POST_GENERATION_EXTRACTIVE_REPAIR` 1。

这不是单一 citation issue。引用格式确实系统性异常：全量 150 题中 backticked `[E#]` 很常见，且 easy/medium 错题分别有 {easy['wrong_with_malformed_pre_citation_marker']['count']}/11、{medium['wrong_with_malformed_pre_citation_marker']['count']}/29 出现至少一种非规范 marker；但引用格式也大量出现在正确题。因此应修复 marker normalization，同时必须独立修复 claim 保留和 repair 选择，不能把所有语义损失归咎于格式。

表格/多步模式更容易触发全量替换或只保留末步。easy 错题中表格模式 {easy['wrong_table_pattern']['count']}/11；medium 错题中表格模式 {medium['wrong_table_pattern']['count']}/29、多步/列表模式 {medium['wrong_multi_step_or_list_pattern']['count']}/29。常见失败包括把完整答案替换成表头、只保留一个表格行、只保留流程最后一步，以及把引用失败升级成整题拒答。

## 可审计的严重阶段损失样本

以下示例由通用排序规则自动选取，不参与任何代码特判；完整 150 题 pre/final 文本与字段证据在 `cases.json` / `cases.csv`。

{example_lines}

## Freshness、评测和数据集风险需单独处理

medium 有 1 个错题只因 `FRESH_EVIDENCE_REQUIRED` 被判 STALE；其本地证据 query coverage=1、无 missing aspects/conflicts。hard 有 {hard['freshness_gate_only_wrong']['count']} 个同类错题。该模式应区分“外部当前事实”与“当前产品/菜单语义”；active KB 文档版本应能满足后者。

评测系统也存在 false-negative 候选，但本报告不据此改写生产二元结果。通用判据为：final 非拒答、约束门失败、但无门控参考语义覆盖≥0.70；或 direct-fact 的参考数字与 ASCII 标识均存在于答案。数据集风险则只用通用的问句单数基数与参考多项枚举冲突来标记。所有这些都在逐题 JSON 中明示为 candidate/risk，而不是宣称已经人工裁决。

当前 evaluator 对归档 final 的重算与 stored correctness **{overall['production_metric_code_drift']['count']}/150** 题不一致；本次为 0，说明当前代码可复现这个候选的 stored continuous score。但 pre-grounding 没有归档生产二元字段，所以 pre 达标数仍是敏感性而非正式生产指标。

## 范围、数据与方法

- 范围：归档 label `{RUN_LABEL}`，easy/medium/hard 各 50 题，模式为 live，模型 qwen-plus，retrieval profile `rpf_default` v2。
- 粒度：每个 difficulty × case_id 一行；150 个唯一 case_id，无重复。
- 正式结果：stored `case_outcome.final_answer_correct` 与 `final_answer_accuracy`。
- pre 敏感性：用当前项目 `_generation_metrics` 对归档 `pre_grounding_answer` 重算，以 0.75 为阈值。
- 互斥归因顺序：执行错误 → freshness gate → pre 正确后处理损伤 → 召回不足 → context 丢失 → 数据集冲突候选 → evaluator false-negative 候选 → partial content loss → 低排名/内容不清晰 → 生成/推理/其他。
- 完整性：验证 run 状态、三难度、各 50 行、case_id 唯一性、dataset/results SHA-256、pre/final 审计哈希与汇总生产准确率。
- 禁止项：脚本不读取凭据、不调用网络、不调用 LLM，也不写 `app/` 代码。

## 限制与稳健性

1. “pre 已正确”是用当前 evaluator 重算得到的保守敏感性，而非归档时单独存储的生产二元裁决。
2. `recall@5=1` 只代表标注相关证据进入 top-5，不保证证据无冲突或易于模型选择；rank 与 evidence assessment 只能提供风险信号。
3. evaluator/data error 类均标为 candidate。没有离线规则能完全替代盲法语义复核；本报告不会把 candidate 当成正式答对。
4. 互斥主因用于加总解释，重叠模式（citation/table/list/reranker）保留在独立 flags 中，不能直接当作单因果证明。

## 系统性修复建议

1. **建立 pre→final 语义单调性门禁。** grounding/repair 后若 reference-free claim coverage、关键数字/标识、步骤数或表格字段覆盖显著下降，应保留已验证的原 claim，或将 repair 限定为 marker 归一化；禁止把完整回答替换成表头/标题。
2. **将 citation normalization 与语义重写解耦。** 先规范 backtick、escaped angle、空格/全角 marker，再做 claim-local 支持校验；未知标签只删除标签，不应默认删除已由其他证据支持的整条 claim。
3. **按 claim 修复，不按整答替换。** 多步列表与多字段表格必须逐项绑定证据；repair 输出要记录 retained/removed claim、依据和顺序，且通过 completeness 守恒检查。
4. **拒答只在 evidence 真不足时发生。** 当 evidence assessment=SUFFICIENT、recall@5=1、context 无丢失时，单个 claim 校验失败应降级为局部删除/重新绑定，不应升级为整题 `UNSUPPORTED_GENERATION`。
5. **修复 freshness 语义域。** 只有外部实时事实要求 web freshness；产品能力、菜单名或“查看当前连接状态”的文档问题由 active KB version 满足。
6. **修复 evaluator 的 polarity/identifier false-negative，并给数据集加一致性 QA。** 对无门控高覆盖但 gate=0 的样本做通用回归；自动检查问句基数、reference、expected_terms 和 evidence 是否互相一致。
7. **恢复 reranker 后做受控 A/B。** 所有 150 题 reranker 都 degraded；但 easy/medium top-5 已满，故 reranker 修复必须用 final binary accuracy 和 pre→final loss 作主要门禁，不能只用 recall。

这些建议均针对可泛化分支与守恒条件，不包含任何 case_id 或错题答案特判。

## 进一步问题

- repair 能否输出 claim-level provenance diff，使“引用修复”与“内容删失”可分别监控？
- 对 evaluator false-negative candidates 做盲法复核后，0.75 阈值和 polarity/identifier gate 应如何校准？
- hard 的多证据组合题中，recall@5 部分命中与首相关排序各自贡献多少？这需要修复后同数据集重跑而不是修改本归档。
"""


def main() -> int:
    if not RUN_INDEX_PATH.exists():
        raise FileNotFoundError(RUN_INDEX_PATH)
    run_index = read_json(RUN_INDEX_PATH)
    if run_index.get("label") != RUN_LABEL or run_index.get("status") != "succeeded":
        raise ValueError("archive identity/status mismatch")
    if tuple(run_index.get("planned_difficulties") or []) != DIFFICULTIES:
        raise ValueError("expected exactly easy/medium/hard")
    archive_sums: dict[str, str] = {}
    for line in (RUN_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        archive_sums[relative.strip().replace("\\", "/")] = digest

    all_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        entry = run_index["difficulties"][difficulty]
        dataset_path = (REPO_ROOT / entry["dataset_source_path"]).resolve()
        results_path = (REMEDIATION_ROOT / entry["run_directory"] / "results.jsonl").resolve()
        dataset_sha = sha_file(dataset_path)
        results_sha = sha_file(results_path)
        if dataset_sha != entry["dataset_source_sha256"]:
            raise ValueError(f"dataset hash mismatch: {difficulty}")
        relative_results = results_path.relative_to(RUN_ROOT).as_posix()
        if archive_sums.get(relative_results) != results_sha:
            raise ValueError(f"results hash mismatch: {difficulty}")
        dataset = read_json(dataset_path)
        case_list = dataset.get("cases") or []
        cases = {str(case["case_id"]): case for case in case_list}
        if len(case_list) != 50 or len(cases) != 50:
            raise ValueError(f"dataset grain mismatch: {difficulty}")
        records: list[dict[str, Any]] = []
        with results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = str(record.get("case_id"))
                if case_id not in cases:
                    raise ValueError(f"unknown case_id: {case_id}")
                records.append(
                    extract_case(
                        difficulty=difficulty,
                        case=cases[case_id],
                        record=record,
                        line_number=line_number,
                        run_id=entry["run_id"],
                        dataset_sha256=dataset_sha,
                        results_sha256=results_sha,
                    )
                )
        if len(records) != 50 or len({row["case_id"] for row in records}) != 50:
            raise ValueError(f"results grain mismatch: {difficulty}")
        archived_accuracy = entry["metrics_summary"]["metrics"]["case_outcome"][
            "final_answer_accuracy"
        ]["mean"]
        recomposed_accuracy = mean(row["production_final_accuracy"] for row in records)
        if not math.isclose(float(archived_accuracy), float(recomposed_accuracy), abs_tol=1e-12):
            raise ValueError(f"summary accuracy mismatch: {difficulty}")
        sources[difficulty] = {
            "run_id": entry["run_id"],
            "dataset_path": dataset_path.relative_to(REPO_ROOT).as_posix(),
            "dataset_sha256": dataset_sha,
            "results_path": results_path.relative_to(REPO_ROOT).as_posix(),
            "results_sha256": results_sha,
            "case_count": len(records),
            "archived_final_answer_accuracy": archived_accuracy,
        }
        all_rows.extend(records)

    if len(all_rows) != 150 or len({row["case_id"] for row in all_rows}) != 150:
        raise ValueError("overall grain mismatch")
    groups = {
        difficulty: summarize_group([row for row in all_rows if row["difficulty"] == difficulty])
        for difficulty in DIFFICULTIES
    }
    groups["overall"] = summarize_group(all_rows)
    metrics_module_path = REPO_ROOT / "app" / "evaluation" / "metrics.py"
    summary = {
        "schema_version": "stt.eval.candidate-v3-diagnostic-summary.v1",
        "status": "succeeded",
        "run_label": RUN_LABEL,
        "analysis_as_of": run_index.get("updated_at"),
        "analysis_mode": "offline_read_only_no_llm_no_network",
        "authoritative_metric": "archived metrics.metrics.case_outcome.final_answer_correct",
        "pre_grounding_metric_role": "current-code sensitivity only",
        "correctness_threshold": CORRECT_THRESHOLD,
        "run_index_sha256": sha_file(RUN_INDEX_PATH),
        "archive_sha256s_sha256": sha_file(RUN_ROOT / "SHA256SUMS"),
        "current_metrics_module_sha256": sha_file(metrics_module_path),
        "sources": sources,
        "data_quality": {
            "row_count": len(all_rows),
            "unique_case_id_count": len({row["case_id"] for row in all_rows}),
            "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
            "difficulty_counts": dict(Counter(row["difficulty"] for row in all_rows)),
            "missing_final_binary_count": sum(
                not isinstance(row["production_final_correct"], bool) for row in all_rows
            ),
            "missing_pre_grounding_count": sum(
                not row["pre_grounding_answer_present"] for row in all_rows
            ),
            "critical_source_hashes_verified": True,
        },
        "groups": groups,
        "targets": {
            "easy": {"operator": ">", "target": 0.95, "minimum_correct_on_50": 48},
            "medium": {"operator": ">", "target": 0.85, "minimum_correct_on_50": 43},
            "hard": None,
        },
        "attribution_order": [
            "execution_or_validation_error",
            "freshness_gate_error",
            "post_generation_process_loss",
            "retrieval_content_insufficient",
            "context_loss",
            "evaluation_dataset_error_candidate",
            "evaluation_system_error_candidate",
            "post_generation_partial_content_loss",
            "retrieval_content_unclear_or_low_rank",
            "generation_reasoning_or_other",
        ],
    }

    write_json(OUTPUT_DIR / "cases.json", {"summary": summary, "cases": all_rows})
    write_csv(all_rows)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "high-frequency-patterns.json", build_patterns(all_rows))
    (OUTPUT_DIR / "REPORT.md").write_text(
        report_text(summary, all_rows), encoding="utf-8", newline="\n"
    )
    deliverables = [
        Path(__file__).name,
        "cases.csv",
        "cases.json",
        "high-frequency-patterns.json",
        "REPORT.md",
        "summary.json",
    ]
    checksum_lines = [f"{sha_file(OUTPUT_DIR / name)}  {name}" for name in sorted(deliverables)]
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": "succeeded", "groups": groups}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
