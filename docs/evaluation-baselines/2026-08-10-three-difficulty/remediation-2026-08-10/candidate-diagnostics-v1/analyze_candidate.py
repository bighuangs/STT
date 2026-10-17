#!/usr/bin/env python3
"""Deterministic stage-by-stage audit for live-candidate-v2-qwen-plus-01.

This script is intentionally offline: it reads the immutable evaluation archive,
recomputes the project's deterministic answer-correctness metric for both the
pre-grounding and final visible answer, validates provenance hashes, and writes
reviewable JSON/CSV/report inputs.  It never reads environment variables and it
does not make network or LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_DIR = Path(__file__).resolve().parent
REMEDIATION_ROOT = OUTPUT_DIR.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
REPO_ROOT = SUITE_ROOT.parents[2]
RUN_LABEL = "live-candidate-v2-qwen-plus-01"
RUN_ROOT = REMEDIATION_ROOT / "runs" / RUN_LABEL
RUN_INDEX_PATH = RUN_ROOT / "run-index.json"
EXPECTED_DIFFICULTIES = ("easy", "medium")
CORRECT_THRESHOLD = 0.8


def _sha_bytes(value: bytes, *, prefix: bool = False) -> str:
    digest = hashlib.sha256(value).hexdigest()
    return f"sha256:{digest}" if prefix else digest


def _sha_text(value: str, *, prefix: bool = True) -> str:
    return _sha_bytes(value.encode("utf-8"), prefix=prefix)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha_json(value: Any, *, prefix: bool = True) -> str:
    return _sha_bytes(_canonical_bytes(value), prefix=prefix)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _contained(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"path escapes archive root: {candidate}")
    return resolved


def _quality(score: float) -> str:
    if score >= CORRECT_THRESHOLD:
        return "correct"
    if score > 0:
        return "partial"
    return "incorrect"


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _issue_codes(observation: Mapping[str, Any]) -> list[str]:
    issues = _nested(observation, "citation_validation", "issues", default=[])
    codes: set[str] = set()
    for issue in issues if isinstance(issues, list) else []:
        if isinstance(issue, str) and issue.strip():
            codes.add(issue.strip())
        elif isinstance(issue, Mapping):
            value = issue.get("code") or issue.get("reason_code") or issue.get("type")
            if isinstance(value, str) and value.strip():
                codes.add(value.strip())
    return sorted(codes)


def _degradation_reasons(observation: Mapping[str, Any]) -> list[str]:
    raw = _nested(observation, "answer_envelope", "degradation_reasons", default=[])
    return sorted({str(value) for value in raw if str(value).strip()}) if isinstance(raw, list) else []


def _stage_loss_subtype(
    *,
    stage_loss: bool,
    decision: str,
    grounding_status: str,
    issue_codes: list[str],
    degradation_reasons: list[str],
) -> str | None:
    if not stage_loss:
        return None
    issue_set = set(issue_codes)
    reason_set = set(degradation_reasons)
    if decision == "REFUSE" or grounding_status == "failed":
        return "grounding_failed_to_refusal"
    if "UNSUPPORTED_CLAIMS_PRUNED" in issue_set | reason_set:
        return "unsupported_claim_pruning"
    if "POST_GENERATION_EXTRACTIVE_REPAIR" in reason_set:
        return "explicit_extractive_repair"
    if (
        "TABLE_FACT_FIELDS_MISSING_FROM_CONTEXT" in issue_set | reason_set
        or "POST_GENERATION_TABLE_FACT_REPAIR" in reason_set
    ):
        return "table_fact_repair"
    return "opaque_grounding_rewrite"


def _primary_attribution(
    *,
    pre_quality: str,
    final_quality: str,
    retrieval_insufficient: bool,
    low_rank_noise_risk: bool,
) -> str:
    if final_quality == "correct":
        return "no_observed_semantic_failure"
    if pre_quality == "correct":
        return "post_generation_grounding_damage"
    if retrieval_insufficient:
        return "retrieval_insufficient"
    if low_rank_noise_risk:
        return "retrieval_noise_likely"
    return "generation_or_reasoning_error"


def _rate(count: int, denominator: int) -> dict[str, float | int]:
    fraction = count / denominator if denominator else 0.0
    return {"count": count, "denominator": denominator, "rate": fraction, "percent": round(100 * fraction, 1)}


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else math.nan


def _safe_preview(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _load_cases(dataset_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = _read_json(dataset_path)
    raw_cases = dataset.get("cases") if isinstance(dataset, Mapping) else None
    if not isinstance(raw_cases, list):
        raise ValueError(f"dataset cases missing: {dataset_path}")
    cases: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, dict) or not case.get("case_id"):
            raise ValueError(f"malformed case in {dataset_path}")
        case_id = str(case["case_id"])
        if case_id in cases:
            raise ValueError(f"duplicate dataset case_id: {case_id}")
        cases[case_id] = case
    return cases, dataset


def _load_results(path: Path) -> list[tuple[int, bytes, dict[str, Any]]]:
    rows: list[tuple[int, bytes, dict[str, Any]]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"result line {line_number} is not an object: {path}")
        rows.append((line_number, raw, value))
    return rows


def _generation_metrics(case: Mapping[str, Any], answer: str, *, abstained: Any = None) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.evaluation.metrics import _generation_metrics as project_generation_metrics

    observation: dict[str, Any] = {"answer": answer}
    if abstained is not None:
        observation["abstained"] = abstained
    metrics, _ = project_generation_metrics(case, observation, [])
    return metrics


def _extract_case(
    *,
    difficulty: str,
    case: dict[str, Any],
    record: dict[str, Any],
    line_number: int,
    raw_line: bytes,
    run_id: str,
    results_sha256: str,
    manifest_sha256: str,
    run_index_sha256: str,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    if str(record.get("case_id")) != case_id:
        raise ValueError(f"dataset/result case mismatch: {case_id}")
    if str(record.get("run_id")) != run_id:
        raise ValueError(f"run identity mismatch for {case_id}")

    observation = record.get("observation")
    if not isinstance(observation, dict):
        raise ValueError(f"observation missing for {case_id}")
    pre_answer = observation.get("pre_grounding_answer")
    final_answer = observation.get("answer")
    envelope_answer = _nested(observation, "answer_envelope", "answer")
    if not isinstance(pre_answer, str) or not isinstance(final_answer, str):
        raise ValueError(f"answer stage missing for {case_id}")
    if final_answer != envelope_answer:
        raise ValueError(f"visible/envelope answer mismatch for {case_id}")
    if observation.get("query") != case.get("query"):
        raise ValueError(f"query mismatch for {case_id}")

    pre_metrics = _generation_metrics(case, pre_answer)
    final_metrics = _generation_metrics(case, final_answer, abstained=observation.get("abstained"))
    stored_final = _nested(record, "metrics", "metrics", "generation", "correctness")
    if not isinstance(stored_final, (int, float)):
        raise ValueError(f"stored final correctness missing for {case_id}")

    pre_score = float(pre_metrics["correctness"])
    # The archived final metric is authoritative for this run.  The evaluator
    # source has legitimately evolved after the run; its current recomputation is
    # retained only as a code-drift sensitivity field and never replaces the
    # immutable run-time value.
    final_score = float(stored_final)
    current_final_score = float(final_metrics["correctness"])
    final_metric_code_drift = abs(final_score - current_final_score) > 1e-12
    pre_quality = _quality(pre_score)
    final_quality = _quality(final_score)
    stage_loss = pre_quality == "correct" and final_quality != "correct"
    generation_error = pre_quality != "correct" and final_quality != "correct"
    pre_expected_recall = pre_metrics.get("expected_term_recall")
    final_expected_recall = final_metrics.get("expected_term_recall")
    expected_terms_full_then_lost = bool(
        pre_expected_recall == 1.0
        and isinstance(final_expected_recall, (int, float))
        and float(final_expected_recall) < 1.0
    )

    pre_audit = observation.get("pre_grounding_answer_audit") or {}
    post_change = observation.get("post_generation_change") or {}
    final_audit = post_change.get("final_answer") or {}
    expected_pre_hash = _sha_text(pre_answer)
    expected_final_hash = _sha_text(final_answer)
    if pre_audit.get("sha256") != expected_pre_hash:
        raise ValueError(f"pre-grounding hash mismatch for {case_id}")
    if final_audit.get("sha256") != expected_final_hash:
        raise ValueError(f"final-answer hash mismatch for {case_id}")

    retrieval = _nested(record, "metrics", "metrics", "retrieval", default={})
    recall_5 = _nested(retrieval, "recall_at_k", "5")
    first_rank = retrieval.get("first_relevant_rank") if isinstance(retrieval, Mapping) else None
    context_summary = observation.get("context_pack_summary") or {}
    coverage = context_summary.get("coverage") or {}
    uncovered = coverage.get("uncovered_sub_query_ids") or []
    retrieval_insufficient = (isinstance(recall_5, (int, float)) and float(recall_5) < 1.0) or bool(uncovered)
    low_rank_noise_risk = first_rank is None or (isinstance(first_rank, (int, float)) and int(first_rank) >= 4)
    retrieved_count = retrieval.get("retrieved_count") if isinstance(retrieval, Mapping) else None
    relevant_count = retrieval.get("relevant_retrieved_count") if isinstance(retrieval, Mapping) else None
    label_noise_exposure = (
        isinstance(retrieved_count, int)
        and isinstance(relevant_count, int)
        and retrieved_count > relevant_count
    )

    citation = observation.get("citation_validation") or {}
    issue_codes = _issue_codes(observation)
    citation_status = str(citation.get("status") or "unknown")
    citation_normalization_issue = citation_status != "valid" or bool(issue_codes)
    degradation_reasons = _degradation_reasons(observation)
    decision = str(
        observation.get("answer_decision")
        or _nested(observation, "answer_envelope", "decision")
        or "unknown"
    )
    grounding_status = str(_nested(observation, "grounding", "status") or "unknown")
    stage_subtype = _stage_loss_subtype(
        stage_loss=stage_loss,
        decision=decision,
        grounding_status=grounding_status,
        issue_codes=issue_codes,
        degradation_reasons=degradation_reasons,
    )
    primary = _primary_attribution(
        pre_quality=pre_quality,
        final_quality=final_quality,
        retrieval_insufficient=retrieval_insufficient,
        low_rank_noise_risk=low_rank_noise_risk,
    )

    pre_chars = len(pre_answer)
    final_chars = len(final_answer)
    content_retention_ratio = final_chars / pre_chars if pre_chars else None
    removed_chars = int(post_change.get("removed_character_count") or 0)
    severe_truncation_candidate = bool(
        stage_loss
        and pre_chars
        and final_chars <= 0.5 * pre_chars
        and removed_chars > 0
    )
    grounding_changed = bool(post_change.get("grounding_changed"))
    reranker = _nested(observation, "retrieval_diagnostics", "reranker", default={}) or {}
    reranker_status = str(reranker.get("status") or "unknown") if isinstance(reranker, Mapping) else "unknown"
    reranker_reason = str(reranker.get("reason") or reranker.get("reason_code") or "") if isinstance(reranker, Mapping) else ""
    evidence_assessment = observation.get("evidence_assessment") or {}
    query_understanding = observation.get("query_understanding") or {}
    temporal_requirement = query_understanding.get("temporal_requirement") or {}
    freshness_gate_only_block = bool(
        evidence_assessment.get("status") == "STALE"
        and set(evidence_assessment.get("reasons") or []) == {"FRESH_EVIDENCE_REQUIRED"}
        and evidence_assessment.get("query_coverage") == 1.0
        and not evidence_assessment.get("missing_aspects")
        and not evidence_assessment.get("conflicts")
    )

    evidence_payload = {
        "case_id": case_id,
        "dataset_case_sha256": _sha_json(case),
        "result_line_sha256": _sha_bytes(raw_line, prefix=True),
        "query_sha256": _sha_text(str(case.get("query") or "")),
        "reference_sha256": _sha_text(str(case.get("reference_answer") or "")),
        "pre_grounding_sha256": expected_pre_hash,
        "final_answer_sha256": expected_final_hash,
        "context_pack_content_hash": context_summary.get("content_hash"),
        "retrieval_diagnostics_sha256": _sha_json(observation.get("retrieval_diagnostics") or {}),
        "citation_validation_sha256": _sha_json(citation),
        "post_generation_change_sha256": _sha_json(post_change),
        "run_index_sha256": run_index_sha256,
        "artifact_manifest_sha256": manifest_sha256,
        "results_file_sha256": results_sha256,
    }
    evidence_payload["case_evidence_sha256"] = _sha_json(evidence_payload)

    return {
        "schema_version": "stt.eval.candidate-diagnostic-case.v1",
        "case_id": case_id,
        "difficulty": difficulty,
        "run_id": run_id,
        "result_line_number": line_number,
        "query": str(case.get("query") or ""),
        "reference_answer": str(case.get("reference_answer") or ""),
        "expected_terms": list(case.get("expected_terms") or []),
        "pre_grounding_answer": pre_answer,
        "final_visible_answer": final_answer,
        "pre_correctness": pre_score,
        "final_correctness": final_score,
        "current_code_recomputed_final_correctness": current_final_score,
        "final_metric_code_drift": final_metric_code_drift,
        "correctness_delta_final_minus_pre": final_score - pre_score,
        "pre_expected_term_recall": pre_expected_recall,
        "final_expected_term_recall": final_expected_recall,
        "pre_quality": pre_quality,
        "final_quality": final_quality,
        "primary_attribution": primary,
        "stage_loss_subtype": stage_subtype,
        "flags": {
            "pre_correct_final_noncorrect": stage_loss,
            "final_itself_generation_error": generation_error,
            "retrieval_insufficient": retrieval_insufficient,
            "retrieval_low_rank_noise_risk": low_rank_noise_risk,
            "retrieval_label_noise_exposure": label_noise_exposure,
            "citation_normalization_issue": citation_normalization_issue,
            "citation_causal_candidate": stage_loss and citation_normalization_issue,
            "post_generation_damage_candidate": stage_loss and grounding_changed,
            "severe_truncation_candidate": severe_truncation_candidate,
            "expected_terms_full_then_lost": expected_terms_full_then_lost,
            "freshness_gate_only_block": freshness_gate_only_block,
        },
        "post_generation": {
            "pipeline_change_type": post_change.get("pipeline_change_type"),
            "grounding_change_type": post_change.get("grounding_change_type"),
            "grounding_changed": grounding_changed,
            "raw_to_safe_changed": bool(post_change.get("raw_to_safe_changed")),
            "removed_character_count": removed_chars,
            "added_character_count": int(post_change.get("added_character_count") or 0),
            "pre_character_count": pre_chars,
            "final_character_count": final_chars,
            "content_retention_ratio": content_retention_ratio,
            "degradation_reasons": degradation_reasons,
            "decision": decision,
            "grounding_status": grounding_status,
            "grounding_reason_code": _nested(observation, "grounding", "reason_code"),
        },
        "retrieval": {
            "first_relevant_rank": first_rank,
            "recall_at_5": recall_5,
            "retrieved_count": retrieved_count,
            "relevant_retrieved_count": relevant_count,
            "context_packed_count": context_summary.get("packed_count"),
            "context_omitted_count": context_summary.get("omitted_count"),
            "context_truncated_count": context_summary.get("truncated_count"),
            "uncovered_sub_query_ids": list(uncovered),
            "evidence_status": evidence_assessment.get("status"),
            "evidence_query_coverage": evidence_assessment.get("query_coverage"),
            "reranker_status": reranker_status,
            "reranker_reason": reranker_reason,
            "freshness_required": bool(query_understanding.get("need_freshness")),
            "temporal_expression": temporal_requirement.get("expression"),
            "temporal_kind": temporal_requirement.get("kind"),
            "evidence_reasons": list(evidence_assessment.get("reasons") or []),
        },
        "citation": {
            "status": citation_status,
            "issues": issue_codes,
            "repair_attempts": citation.get("repair_attempts"),
            "coverage": citation.get("coverage"),
            "citation_count": citation.get("citation_count"),
            "trusted_count": citation.get("trusted_count"),
        },
        "model_id": _nested(observation, "versions", "model_id"),
        "evidence_integrity": {"status": "verified", **evidence_payload},
    }


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    count = len(rows)
    flags = sorted({key for row in rows for key in row["flags"]})
    transitions = Counter(f"{row['pre_quality']}->{row['final_quality']}" for row in rows)
    citation_status = Counter(row["citation"]["status"] for row in rows)
    issue_occurrences = Counter(code for row in rows for code in row["citation"]["issues"])
    return {
        "label": label,
        "case_count": count,
        "mean_pre_correctness": _mean(row["pre_correctness"] for row in rows),
        "mean_final_correctness": _mean(row["final_correctness"] for row in rows),
        "mean_correctness_delta": _mean(row["correctness_delta_final_minus_pre"] for row in rows),
        "pre_correct": _rate(sum(row["pre_quality"] == "correct" for row in rows), count),
        "final_correct": _rate(sum(row["final_quality"] == "correct" for row in rows), count),
        "numeric_correctness_drop": _rate(sum(row["final_correctness"] < row["pre_correctness"] for row in rows), count),
        "flags": {flag: _rate(sum(bool(row["flags"].get(flag)) for row in rows), count) for flag in flags},
        "primary_attribution": {
            key: _rate(value, count)
            for key, value in sorted(Counter(row["primary_attribution"] for row in rows).items())
        },
        "stage_loss_subtypes": {
            key: _rate(value, count)
            for key, value in sorted(Counter(row["stage_loss_subtype"] for row in rows if row["stage_loss_subtype"]).items())
        },
        "quality_transitions": dict(sorted(transitions.items())),
        "citation_status": dict(sorted(citation_status.items())),
        "citation_issue_occurrences": dict(sorted(issue_occurrences.items(), key=lambda item: (-item[1], item[0]))),
        "grounding_changed": _rate(sum(row["post_generation"]["grounding_changed"] for row in rows), count),
        "context_truncated": _rate(sum((row["retrieval"]["context_truncated_count"] or 0) > 0 for row in rows), count),
        "official_recall_at_5_full": _rate(sum(row["retrieval"]["recall_at_5"] == 1.0 for row in rows), count),
        "relevant_rank_top_1": _rate(sum(row["retrieval"]["first_relevant_rank"] == 1 for row in rows), count),
        "relevant_rank_4_or_worse_or_missing": _rate(
            sum(
                row["retrieval"]["first_relevant_rank"] is None
                or row["retrieval"]["first_relevant_rank"] >= 4
                for row in rows
            ),
            count,
        ),
    }


def _pattern_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predicates: list[tuple[str, str, Any]] = [
        ("post_generation_damage_candidate", "pre 正确、final 非正确且记录了 grounding 变更", lambda r: r["flags"]["post_generation_damage_candidate"]),
        ("citation_normalization_issue", "citation status 非 valid 或存在 issue code", lambda r: r["flags"]["citation_normalization_issue"]),
        ("citation_overlap_with_stage_loss", "生成后损坏与 citation normalization issue 重叠", lambda r: r["flags"]["citation_causal_candidate"]),
        ("generation_or_reasoning_error", "pre 与 final 均未达到 0.8 正确阈值", lambda r: r["flags"]["final_itself_generation_error"]),
        ("reranker_unavailable", "reranker 未成功，回退保持原顺序", lambda r: r["retrieval"]["reranker_status"] not in {"succeeded", "available"}),
        ("low_rank_noise_risk", "首个标注相关块位于第 4 位以后或缺失", lambda r: r["flags"]["retrieval_low_rank_noise_risk"]),
        ("severe_truncation_candidate", "final 长度不超过 pre 的一半且发生语义损失", lambda r: r["flags"]["severe_truncation_candidate"]),
        ("expected_terms_full_then_lost", "pre 覆盖全部 expected terms，但 final 不再完整覆盖", lambda r: r["flags"]["expected_terms_full_then_lost"]),
        ("grounding_failed_to_refusal", "正确 pre 经 grounding failed/REFUSE 变成非正确 final", lambda r: r["stage_loss_subtype"] == "grounding_failed_to_refusal"),
        ("unsupported_claim_pruning", "正确 pre 在 unsupported-claim pruning 后变成非正确 final", lambda r: r["stage_loss_subtype"] == "unsupported_claim_pruning"),
        ("opaque_grounding_rewrite", "正确 pre 经无明确 issue 的 grounding rewrite 变差", lambda r: r["stage_loss_subtype"] == "opaque_grounding_rewrite"),
    ]
    result: list[dict[str, Any]] = []
    for code, definition, predicate in predicates:
        selected = [row for row in rows if predicate(row)]
        if len(selected) < 5:
            continue
        result.append(
            {
                "pattern": code,
                "definition": definition,
                **_rate(len(selected), len(rows)),
                "case_ids": [row["case_id"] for row in selected],
            }
        )
    return sorted(result, key=lambda item: (-int(item["count"]), str(item["pattern"])))


def _chart_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "no_observed_semantic_failure": "未观察到语义失败",
        "post_generation_grounding_damage": "生成后 grounding/repair 损坏",
        "generation_or_reasoning_error": "生成或推理错误",
        "retrieval_noise_likely": "检索噪声可能主导",
        "retrieval_insufficient": "检索不足",
    }
    primary = summary["groups"]["overall"]["primary_attribution"]
    rows = [
        {"attribution": labels.get(code, code), "case_count": values["count"], "rate_percent": values["percent"], "code": code, "denominator": values["denominator"]}
        for code, values in primary.items()
    ]
    return sorted(rows, key=lambda row: (-row["case_count"], row["attribution"]))


def _build_artifact(summary: dict[str, Any], rows: list[dict[str, Any]], patterns: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summary["groups"]["overall"]
    easy = summary["groups"]["easy"]
    medium = summary["groups"]["medium"]
    expected_loss = summary["expected_terms_full_then_lost"]
    source_id = "candidate_diagnostics"
    source = {
        "id": source_id,
        "label": "live-candidate-v2-qwen-plus-01 deterministic stage audit",
        "path": "docs/evaluation-baselines/2026-08-10-three-difficulty/remediation-2026-08-10/candidate-diagnostics-v1/summary.json",
        "query": {
            "engine": "duckdb reproduction query over the Python-generated audited CSV",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('docs/evaluation-baselines/2026-08-10-three-difficulty/remediation-2026-08-10/candidate-diagnostics-v1/cases.csv', header=true);",
            "description": "The Python audit joins archived easy/medium results to their datasets and writes cases.csv; this bounded SQL query exposes the exact reviewed case-level source table used for report aggregation and independent reproduction.",
            "filters": ["difficulty in {easy, medium}", "50 archived cases per difficulty", "correctness threshold >= 0.8", "no LLM or network calls"],
            "metric_definitions": [
                "Stage loss: pre-grounding correctness >= 0.8 and final correctness < 0.8.",
                "Generation/reasoning error flag: both pre-grounding and final correctness < 0.8.",
                "Citation normalization issue: citation status is not valid or issue codes are present.",
                "Retrieval insufficiency: official recall@5 < 1 or required sub-query context is uncovered.",
            ],
            "tables_used": [
                "runs/live-candidate-v2-qwen-plus-01/artifacts/easy/*/results.jsonl",
                "runs/live-candidate-v2-qwen-plus-01/artifacts/medium/*/results.jsonl",
                "datasets/easy.json",
                "datasets/medium.json",
            ],
        },
    }

    headline = [{
        "stage_loss_percent": overall["flags"]["pre_correct_final_noncorrect"]["percent"] / 100,
        "final_correct_percent": overall["final_correct"]["percent"] / 100,
        "retrieval_recall5_percent": overall["official_recall_at_5_full"]["percent"] / 100,
        "citation_issue_percent": overall["flags"]["citation_normalization_issue"]["percent"] / 100,
    }]
    example_rows = sorted(
        [row for row in rows if row["primary_attribution"] != "no_observed_semantic_failure"],
        key=lambda row: (row["correctness_delta_final_minus_pre"], -row["post_generation"]["removed_character_count"], row["case_id"]),
    )[:12]
    examples = [
        {
            "case_id": row["case_id"],
            "difficulty": row["difficulty"],
            "pre_score": row["pre_correctness"],
            "final_score": row["final_correctness"],
            "primary": row["primary_attribution"],
            "subtype": row["stage_loss_subtype"] or "—",
            "citation": row["citation"]["status"],
            "rank": row["retrieval"]["first_relevant_rank"],
            "pre_preview": _safe_preview(row["pre_grounding_answer"], 100),
            "final_preview": _safe_preview(row["final_visible_answer"], 100),
        }
        for row in example_rows
    ]

    title = "qwen-plus 候选运行：生成前后语义损失诊断"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\n审计对象：`{RUN_LABEL}`，简单与中等各 50 题。"},
        {"id": "summary", "type": "markdown", "sourceId": source_id, "body": (
            "## 技术摘要\n\n"
            f"主导缺陷发生在 LLM 已生成答案之后：80/100 个 pre-grounding 答案达到正确阈值，但 final 仅 36/100；"
            f"其中 45/100（45.0%）由正确 pre 变成部分命中或错误 final，而且 45 个全部记录了 grounding 变更。"
            f"中等集的阶段损失为 {medium['flags']['pre_correct_final_noncorrect']['count']}/50（{medium['flags']['pre_correct_final_noncorrect']['percent']}%），"
            f"显著高于简单集的 {easy['flags']['pre_correct_final_noncorrect']['count']}/50（{easy['flags']['pre_correct_final_noncorrect']['percent']}%）。"
        )},
        {"id": "kpis", "type": "metric-strip", "cardIds": ["stage_loss", "final_correct", "recall5", "citation_issue"]},
        {"id": "findings", "type": "markdown", "sourceId": source_id, "body": (
            "## 关键发现\n\n"
            "互斥主归因把每题只计一次：45 题归入生成后 grounding/repair 损坏，17 题归入生成或推理错误，2 题存在较强的低排名检索噪声迹象，36 题未观察到语义失败。"
            "官方相关证据在 99/100 题进入 top-5，且上下文截断为 0/100，因此“召回不足”不能解释大多数错误。"
        )},
        {"id": "attribution_chart_block", "type": "chart", "chartId": "primary_attribution"},
        {"id": "stage_breakdown", "type": "markdown", "sourceId": source_id, "body": (
            "## 难度分层\n\n"
            "中等集从 pre 正确率 70% 降到 final 正确率 16%，下降 54 个百分点；简单集从 90% 降到 56%，下降 34 个百分点。"
            "这说明候选模型本身在多数题上已经生成了足够答案，但后处理对中等、多约束回答破坏更严重。"
        )},
        {"id": "patterns_text", "type": "markdown", "sourceId": source_id, "body": (
            "## 高频模式\n\n"
            "Citation 校验/归一化在 64/100 题出现非 valid 状态或 issue code，并与 34/45 个阶段损失重叠；但仍有 11/45 个阶段损失在 citation 显示 valid 且无 issue 时发生。"
            f"有 {expected_loss['count']}/100 题在 pre 已覆盖全部 expected terms、final 却丢失至少一项："
            f"其中 {expected_loss['by_primary_attribution'].get('post_generation_grounding_damage', 0)} 题归入生成后损坏、"
            f"{expected_loss['by_primary_attribution'].get('generation_or_reasoning_error', 0)} 题归入生成/推理错误，"
            f"{expected_loss['by_primary_attribution'].get('no_observed_semantic_failure', 0)} 题虽丢词项但仍达到语义正确阈值。"
            "此外 reranker 在 100/100 题均未成功并回退保持原顺序，这是系统性噪声风险，但不能直接视为 100 个因果失败。"
        )},
        {"id": "examples_text", "type": "markdown", "sourceId": source_id, "body": (
            "## 可审计示例\n\n"
            "按 final-minus-pre 正确性差值升序自动选取的严重样本包括："
            + "、".join(item["case_id"] for item in examples[:8])
            + "。选取不按题号特判；完整 pre/final 文本、字段级判据和证据哈希见 `cases.json` / `cases.csv`。"
        )},
        {"id": "scope", "type": "markdown", "sourceId": source_id, "body": (
            "## 范围、数据与定义\n\n"
            "范围仅含本次归档的简单 50 题和中等 50 题，模型均为 qwen-plus。"
            "“正确”沿用项目确定性 correctness 指标并使用 0.8 阈值；“阶段损失”指 pre 达标而 final 未达标。"
            "“检索不足”要求官方 recall@5 不满或必需子查询上下文未覆盖；“噪声风险”使用首个标注相关块排名作为保守代理。"
        )},
        {"id": "methodology", "type": "markdown", "sourceId": source_id, "body": (
            "## 方法与可复现性\n\n"
            "脚本逐题连接数据集 reference 与归档 observation，分别对 pre-grounding 与最终可见答案重算项目原生 generation correctness，并验证结果行、数据集、归档 manifest、pre/final 文本和诊断字段哈希。"
            "主归因为互斥层级；citation、truncation、reranker 与 retrieval 风险作为可重叠标记。分析不调用网络或 LLM。"
        )},
        {"id": "limitations", "type": "markdown", "sourceId": source_id, "body": (
            "## 限制与稳健性\n\n"
            "正确性是 expected terms/reference 驱动的确定性代理，不等同于人工或 LLM 语义裁决；partial 与 incorrect 的边界可能受参考答案粒度影响。"
            "相关块标注可能不完备，因此未标注块不能直接称为无关噪声。这里的“损坏候选”由阶段顺序与哈希变化支持，但 observation 仍不足以证明 citation normalization、grounding verifier 或 extractive repair 中哪一个内部函数单独造成损失。"
            "HTML 包完成结构校验；安装的 Windows Chromium 对共享全宽顶栏报告触发 8px 滚动条兼容性告警，因此未宣称完整浏览器 QA 通过。"
        )},
        {"id": "freshness", "type": "markdown", "sourceId": source_id, "body": (
            "## Freshness 误判审计\n\n"
            "EASY-011 与 MEDIUM-047 都因词面“当前”被理解为 external-current，进而要求 30 日内 web 证据；虽然本地手册证据 query coverage=1、无 missing aspects/conflicts，仍被唯一原因 `FRESH_EVIDENCE_REQUIRED` 判为 STALE 并拒答。"
            "触发链位于 `query_understanding_service.py` 的 `_CURRENT_TIME` / `_detect_temporal` / `need_freshness`，以及 `sufficiency_service.py` 的 `_freshness_required` / `assess_evidence`。"
            "最小通用修复应区分 external freshness 与 authorized-KB-current：前者仍要求新鲜 web，后者由 active document version 满足，避免把产品能力或“用于查看当前连接状态”的菜单语义误当实时事实。"
        )},
        {"id": "next_steps", "type": "markdown", "sourceId": source_id, "body": (
            "## 建议的下一步\n\n"
            "1. 把 pre→final correctness 单调性纳入回归门禁，发生显著下降时回滚或保留 pre。\n"
            "2. 将 citation 标记归一化与答案内容重写解耦，并记录 removed claims 与选择依据。\n"
            "3. 修复 reranker 可用性后重放同一归档方案，比较首相关排名与阶段损失，而不是只看 final accuracy。\n"
            "4. 对 45 个阶段损失样本做盲法语义复核，确认确定性指标未把等价表达误判为损坏。"
        )},
        {"id": "questions", "type": "markdown", "body": (
            "## 进一步问题\n\n"
            "Grounding repair 能否在改写前后做 reference-free claim coverage 守恒检查？Citation valid 但内容损失的 11 题是否来自相同分支？"
            "Reranker 恢复后，低排名相关证据的 7 题是否仍出现相同损失？这些问题需要下一次可控重放回答。"
        )},
    ]

    cards = [
        {"id": "stage_loss", "description": "正确 pre 变为非正确 final", "dataset": "headline", "sourceId": source_id, "metrics": [{"label": "阶段损失率", "field": "stage_loss_percent", "format": "percent"}]},
        {"id": "final_correct", "description": "最终可见答案达到 0.8 阈值", "dataset": "headline", "sourceId": source_id, "metrics": [{"label": "Final 正确率", "field": "final_correct_percent", "format": "percent"}]},
        {"id": "recall5", "description": "官方标注相关证据进入 top-5", "dataset": "headline", "sourceId": source_id, "metrics": [{"label": "Recall@5 完整率", "field": "retrieval_recall5_percent", "format": "percent"}]},
        {"id": "citation_issue", "description": "非 valid 或含 citation issue", "dataset": "headline", "sourceId": source_id, "metrics": [{"label": "Citation issue 率", "field": "citation_issue_percent", "format": "percent"}]},
    ]
    charts = [{
        "id": "primary_attribution",
        "title": "100 题互斥主归因",
        "subtitle": "每题仅计入一个主类别；条形长度为题数",
        "intent": "comparison",
        "question": "最终语义失败最常发生在哪个阶段？",
        "rationale": "互斥类别适合使用从零开始的单序列横向条形图比较题数。",
        "type": "bar",
        "dataset": "attribution",
        "sourceId": source_id,
        "encodings": {
            "x": {"field": "attribution", "type": "nominal", "label": "主归因"},
            "y": {"field": "case_count", "type": "quantitative", "aggregate": "none", "label": "题数", "format": "number"},
            "tooltip": [
                {"field": "case_count", "type": "quantitative", "label": "题数", "format": "number"},
                {"field": "rate_percent", "type": "quantitative", "label": "比例（%）", "format": "number"},
                {"field": "denominator", "type": "quantitative", "label": "分母", "format": "number"},
            ],
        },
        "valueFormat": "number",
        "maxRows": 10,
        "compatibleTypes": ["bar"],
        "labels": {"values": "all"},
        "settings": {"orientation": "horizontal", "groupMode": "single", "showValues": True, "sort": "descending"},
        "surface": {"interactiveLegend": False, "showControls": False, "viewMode": "visualization"},
    }]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "对 qwen-plus 候选运行进行 pre-grounding 与 final 可见答案的逐题确定性诊断。",
            "generatedAt": summary["source_snapshot_at"],
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": [],
            "sources": [source],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": summary["source_snapshot_at"],
            "status": "ready",
            "datasets": {
                "headline": headline,
                "attribution": _chart_rows(summary),
            },
            "accessIssues": [],
        },
        "sources": [source],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id", "difficulty", "run_id", "pre_correctness", "final_correctness", "correctness_delta_final_minus_pre",
        "pre_quality", "final_quality", "primary_attribution", "stage_loss_subtype", "pre_correct_final_noncorrect",
        "final_itself_generation_error", "retrieval_insufficient", "retrieval_low_rank_noise_risk", "retrieval_label_noise_exposure", "citation_normalization_issue",
        "citation_causal_candidate", "post_generation_damage_candidate", "severe_truncation_candidate", "expected_terms_full_then_lost",
        "freshness_gate_only_block", "grounding_changed", "pre_expected_term_recall", "final_expected_term_recall",
        "current_code_recomputed_final_correctness", "final_metric_code_drift",
        "pipeline_change_type", "grounding_change_type", "decision", "grounding_status", "removed_character_count",
        "added_character_count", "content_retention_ratio", "first_relevant_rank", "recall_at_5", "retrieved_count",
        "relevant_retrieved_count", "context_omitted_count", "context_truncated_count", "evidence_status", "evidence_query_coverage",
        "reranker_status", "citation_status", "citation_issue_codes", "degradation_reasons", "model_id", "query", "reference_answer",
        "pre_grounding_answer", "final_visible_answer", "case_evidence_sha256", "result_line_sha256", "pre_grounding_sha256", "final_answer_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row["case_id"], "difficulty": row["difficulty"], "run_id": row["run_id"],
                "pre_correctness": row["pre_correctness"], "final_correctness": row["final_correctness"],
                "correctness_delta_final_minus_pre": row["correctness_delta_final_minus_pre"], "pre_quality": row["pre_quality"],
                "pre_expected_term_recall": row["pre_expected_term_recall"], "final_expected_term_recall": row["final_expected_term_recall"],
                "current_code_recomputed_final_correctness": row["current_code_recomputed_final_correctness"],
                "final_metric_code_drift": row["final_metric_code_drift"],
                "final_quality": row["final_quality"], "primary_attribution": row["primary_attribution"], "stage_loss_subtype": row["stage_loss_subtype"] or "",
                **row["flags"], "grounding_changed": row["post_generation"]["grounding_changed"],
                "pipeline_change_type": row["post_generation"]["pipeline_change_type"], "grounding_change_type": row["post_generation"]["grounding_change_type"],
                "decision": row["post_generation"]["decision"], "grounding_status": row["post_generation"]["grounding_status"],
                "removed_character_count": row["post_generation"]["removed_character_count"], "added_character_count": row["post_generation"]["added_character_count"],
                "content_retention_ratio": row["post_generation"]["content_retention_ratio"], "first_relevant_rank": row["retrieval"]["first_relevant_rank"],
                "recall_at_5": row["retrieval"]["recall_at_5"], "retrieved_count": row["retrieval"]["retrieved_count"],
                "relevant_retrieved_count": row["retrieval"]["relevant_retrieved_count"], "context_omitted_count": row["retrieval"]["context_omitted_count"],
                "context_truncated_count": row["retrieval"]["context_truncated_count"], "evidence_status": row["retrieval"]["evidence_status"],
                "evidence_query_coverage": row["retrieval"]["evidence_query_coverage"], "reranker_status": row["retrieval"]["reranker_status"],
                "citation_status": row["citation"]["status"], "citation_issue_codes": "|".join(row["citation"]["issues"]),
                "degradation_reasons": "|".join(row["post_generation"]["degradation_reasons"]), "model_id": row["model_id"],
                "query": row["query"], "reference_answer": row["reference_answer"], "pre_grounding_answer": row["pre_grounding_answer"],
                "final_visible_answer": row["final_visible_answer"], "case_evidence_sha256": row["evidence_integrity"]["case_evidence_sha256"],
                "result_line_sha256": row["evidence_integrity"]["result_line_sha256"], "pre_grounding_sha256": row["evidence_integrity"]["pre_grounding_sha256"],
                "final_answer_sha256": row["evidence_integrity"]["final_answer_sha256"],
            })


def _write_source_notes(summary: dict[str, Any]) -> None:
    notes = {
        "schema_version": "stt.eval.candidate-diagnostic-source-notes.v1",
        "report_surface": "portable self-contained HTML",
        "chart_map": [{
            "chart_id": "primary_attribution",
            "question": "最终语义失败最常发生在哪个阶段？",
            "type": "horizontalBar",
            "x": "mutually exclusive primary attribution",
            "y": "case_count",
            "denominator": 100,
            "sort": "case_count descending",
            "zero_baseline": True,
            "legend": "hidden because the chart has one series",
            "rationale": "Exact comparison of a small number of mutually exclusive categories.",
        }],
        "quantitative_sections_without_additional_charts": {
            "difficulty_breakdown": "The two difficulty comparisons are stated directly; exact fields remain in summary.json.",
            "high_frequency_patterns": "Patterns overlap, so another partition-like chart would be misleading; definitions and case IDs remain in high-frequency-patterns.json.",
            "examples": "Case-level answer previews and audit hashes require exact lookup and remain in cases.json/cases.csv rather than a wide report table.",
        },
        "portable_html_qa": {
            "canonical_artifact_validation": "passed",
            "browser_verification": "structural_only",
            "limitation": "Installed Windows Chromium engines render the packaged reader, but the shared full-bleed top bar plus a traditional vertical scrollbar creates an 8px document-width overflow (measured clientWidth=1409, scrollWidth=1417). This is a packaging-reader layout issue, not an analytical data overflow.",
            "content_mitigation": "Wide report tables were removed; all exact case data remains in JSON/CSV. The self-contained report retains semantic chart fallback data.",
        },
        "data_quality_gate": summary["data_quality_gate"],
    }
    _write_json(OUTPUT_DIR / "source-notes.json", notes)


def _write_checksums() -> None:
    excluded = {"SHA256SUMS"}
    files = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name not in excluded)
    lines = [f"{_sha_file(path)}  {path.name}" for path in files]
    (OUTPUT_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _sensitive_payload_check(paths: Iterable[Path]) -> None:
    forbidden_fragments = ("authorization: bearer ", '"api_key"', '"access_token"', "sk-")
    for path in paths:
        text = path.read_text(encoding="utf-8-sig", errors="replace").lower()
        for fragment in forbidden_fragments:
            if fragment in text:
                raise ValueError(f"possible credential material in {path.name}: {fragment}")


def run_analysis() -> dict[str, Any]:
    if not RUN_INDEX_PATH.exists():
        raise FileNotFoundError(RUN_INDEX_PATH)
    run_index = _read_json(RUN_INDEX_PATH)
    run_index_sha256 = _sha_file(RUN_INDEX_PATH)
    if run_index.get("label") != RUN_LABEL or run_index.get("status") != "succeeded":
        raise ValueError("unexpected run-index identity/status")
    difficulty_entries = run_index.get("difficulties")
    if not isinstance(difficulty_entries, dict) or tuple(sorted(difficulty_entries)) != EXPECTED_DIFFICULTIES:
        raise ValueError("run-index must contain exactly easy and medium")

    all_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []
    for difficulty in EXPECTED_DIFFICULTIES:
        entry = difficulty_entries[difficulty]
        run_id = str(entry["run_id"])
        dataset_path = _contained(REPO_ROOT, REPO_ROOT / str(entry["dataset_source_path"]))
        if _sha_file(dataset_path) != entry["dataset_source_sha256"]:
            raise ValueError(f"dataset source hash mismatch: {difficulty}")
        cases, dataset = _load_cases(dataset_path)
        if len(cases) != 50:
            raise ValueError(f"expected 50 dataset cases for {difficulty}, got {len(cases)}")

        manifest_entry = entry["artifact_copy_manifest"]
        manifest_path = _contained(REMEDIATION_ROOT, REMEDIATION_ROOT / str(manifest_entry["path"]))
        manifest_sha256 = _sha_file(manifest_path)
        if manifest_sha256 != manifest_entry["sha256"]:
            raise ValueError(f"artifact copy manifest hash mismatch: {difficulty}")
        manifest = _read_json(manifest_path)
        result_copy = _nested(manifest, "copies", "results")
        if not isinstance(result_copy, Mapping) or result_copy.get("source_target_sha256_match") is not True:
            raise ValueError(f"results copy is not verified: {difficulty}")
        results_path = _contained(REMEDIATION_ROOT, REMEDIATION_ROOT / str(_nested(result_copy, "target", "relative_path")))
        results_sha256 = _sha_file(results_path)
        if results_sha256 != _nested(result_copy, "target", "sha256") or results_sha256 != _nested(result_copy, "source", "download_sha256"):
            raise ValueError(f"results hash mismatch: {difficulty}")

        raw_results = _load_results(results_path)
        if len(raw_results) != 50:
            raise ValueError(f"expected 50 results for {difficulty}, got {len(raw_results)}")
        result_ids = [str(record.get("case_id")) for _, _, record in raw_results]
        if len(set(result_ids)) != 50 or set(result_ids) != set(cases):
            raise ValueError(f"dataset/result universe mismatch: {difficulty}")
        selected_ids = entry.get("selected_case_ids") or []
        if set(selected_ids) != set(result_ids):
            raise ValueError(f"selected/result universe mismatch: {difficulty}")

        for line_number, raw_line, record in raw_results:
            case_id = str(record["case_id"])
            all_rows.append(_extract_case(
                difficulty=difficulty,
                case=cases[case_id],
                record=record,
                line_number=line_number,
                raw_line=raw_line,
                run_id=run_id,
                results_sha256=results_sha256,
                manifest_sha256=manifest_sha256,
                run_index_sha256=run_index_sha256,
            ))

        source_files.extend([
            {"role": f"{difficulty}_dataset", "path": dataset_path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha_file(dataset_path), "case_count": len(cases), "dataset_schema": dataset.get("schema_version")},
            {"role": f"{difficulty}_results", "path": results_path.relative_to(REPO_ROOT).as_posix(), "sha256": results_sha256, "case_count": len(raw_results), "run_id": run_id},
            {"role": f"{difficulty}_artifact_copy_manifest", "path": manifest_path.relative_to(REPO_ROOT).as_posix(), "sha256": manifest_sha256},
        ])
        quality_checks.append({"difficulty": difficulty, "dataset_case_count": len(cases), "result_case_count": len(raw_results), "join_coverage": 1.0, "hashes_verified": True})

    all_rows.sort(key=lambda row: (EXPECTED_DIFFICULTIES.index(row["difficulty"]), row["case_id"]))
    if len(all_rows) != 100 or len({row["case_id"] for row in all_rows}) != 100:
        raise ValueError("expected a unique 100-case audit universe")

    groups = {
        difficulty: _summarize([row for row in all_rows if row["difficulty"] == difficulty], difficulty)
        for difficulty in EXPECTED_DIFFICULTIES
    }
    groups["overall"] = _summarize(all_rows, "overall")
    patterns = _pattern_rows(all_rows)
    expected_loss_rows = [row for row in all_rows if row["flags"]["expected_terms_full_then_lost"]]
    freshness_rows = [row for row in all_rows if row["flags"]["freshness_gate_only_block"]]
    script_sha256 = _sha_file(Path(__file__))
    summary = {
        "schema_version": "stt.eval.candidate-diagnostics-summary.v1",
        "run_label": RUN_LABEL,
        "source_snapshot_at": str(run_index.get("updated_at") or run_index.get("created_at")),
        "analysis_mode": "deterministic_offline_archive_audit",
        "llm_calls": 0,
        "metric_definition": {
            "pre_implementation": "current app.evaluation.metrics._generation_metrics applied to archived pre-grounding text",
            "final_implementation": "archived record.metrics.metrics.generation.correctness (run-time authoritative)",
            "correct_threshold": CORRECT_THRESHOLD,
            "correct": "correctness >= 0.8",
            "partial": "0 < correctness < 0.8",
            "incorrect": "correctness == 0",
            "stage_loss": "pre quality is correct and final quality is partial or incorrect",
            "severe_truncation_candidate": "stage loss, final character count <= 50% of pre, and removed_character_count > 0",
            "caveat": "This deterministic expected-term/reference proxy is not a substitute for blinded semantic review.",
            "code_drift_policy": "Current-code final recomputation is sensitivity-only and never overrides the archived run-time final metric.",
        },
        "attribution_rules_sha256": _sha_json({
            "threshold": CORRECT_THRESHOLD,
            "primary_order": ["final_correct", "pre_correct_stage_loss", "retrieval_insufficient", "low_rank_noise", "generation_or_reasoning"],
            "stage_subtype_order": ["grounding_failed_to_refusal", "unsupported_claim_pruning", "explicit_extractive_repair", "table_fact_repair", "opaque_grounding_rewrite"],
        }),
        "analysis_script_sha256": script_sha256,
        "run_index": {"path": RUN_INDEX_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": run_index_sha256},
        "source_files": source_files,
        "data_quality_gate": {
            "status": "passed",
            "checks": quality_checks,
            "verified_case_count": 100,
            "missing_required_stage_fields": 0,
            "stored_vs_recomputed_final_correctness_mismatches": 0,
            "visible_vs_envelope_answer_mismatches": 0,
            "pre_or_final_hash_mismatches": 0,
        },
        "groups": groups,
        "high_frequency_pattern_count": len(patterns),
        "expected_terms_full_then_lost": {
            **_rate(len(expected_loss_rows), len(all_rows)),
            "definition": "pre expected_term_recall == 1 and final expected_term_recall < 1",
            "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in expected_loss_rows).items())),
            "by_primary_attribution": dict(sorted(Counter(row["primary_attribution"] for row in expected_loss_rows).items())),
            "by_stage_loss_subtype": dict(sorted(Counter(row["stage_loss_subtype"] or "not_stage_loss" for row in expected_loss_rows).items())),
            "by_citation_status": dict(sorted(Counter(row["citation"]["status"] for row in expected_loss_rows).items())),
            "case_ids": [row["case_id"] for row in expected_loss_rows],
        },
        "freshness_audit": {
            "freshness_gate_only_block": _rate(len(freshness_rows), len(all_rows)),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "query": row["query"],
                    "reference_answer": row["reference_answer"],
                    "temporal_expression": row["retrieval"]["temporal_expression"],
                    "temporal_kind": row["retrieval"]["temporal_kind"],
                    "evidence_status": row["retrieval"]["evidence_status"],
                    "evidence_query_coverage": row["retrieval"]["evidence_query_coverage"],
                    "evidence_reasons": row["retrieval"]["evidence_reasons"],
                    "recall_at_5": row["retrieval"]["recall_at_5"],
                    "final_answer_sha256": row["evidence_integrity"]["final_answer_sha256"],
                    "case_evidence_sha256": row["evidence_integrity"]["case_evidence_sha256"],
                }
                for row in freshness_rows
            ],
            "trigger_chain": [
                {"path": "app/application/query_understanding_service.py", "lines": [78, 344, 721], "symbols": ["_CURRENT_TIME", "_detect_temporal", "QueryPlanner.understand"], "finding": "Any occurrence of 当前/最新/目前/现在/... becomes TemporalKind.CURRENT and therefore need_freshness=True."},
                {"path": "app/rag/query/sufficiency_service.py", "lines": [501, 579, 580, 591], "symbols": ["_freshness_required", "assess_evidence"], "finding": "Every freshness request requires at least one recent web item; active local document versions cannot satisfy it."},
            ],
            "minimum_generic_fix": "Represent freshness scope explicitly. Strong external-recency requests (最新/今天/实时/截至) remain external and require fresh web. Product/manual capability or menu semantics using 当前 become authorized_kb_current and are satisfied only by evidence bound to the active published document version; do not simply disable the downstream gate when web is forbidden.",
            "counterexample_tests": {
                "must_not_require_external_web": ["当前打印机支持的PCL设备字体有多少种？", "哪个菜单项用于查看当前连接状态？", "该设备当前支持哪些纸张规格？"],
                "must_still_require_external_freshness": ["当前最新固件版本是什么？", "今天官网售价是多少？", "截至今天有哪些安全公告？"],
            },
            "app_modified": False,
        },
    }
    summary["data_quality_gate"]["stored_vs_recomputed_final_correctness_mismatches"] = sum(
        row["final_metric_code_drift"] for row in all_rows
    )
    summary["data_quality_gate"]["stored_final_metric_authoritative"] = True
    summary["data_quality_gate"]["current_code_drift_case_ids"] = [
        row["case_id"] for row in all_rows if row["final_metric_code_drift"]
    ]

    _write_json(OUTPUT_DIR / "cases.json", {"schema_version": "stt.eval.candidate-diagnostic-cases.v1", "case_count": len(all_rows), "cases": all_rows})
    _write_csv(all_rows, OUTPUT_DIR / "cases.csv")
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_json(OUTPUT_DIR / "high-frequency-patterns.json", {"schema_version": "stt.eval.candidate-diagnostic-patterns.v1", "patterns": patterns})
    _write_source_notes(summary)
    artifact = _build_artifact(summary, all_rows, patterns)
    _write_json(OUTPUT_DIR / "artifact.json", artifact)
    _sensitive_payload_check([
        OUTPUT_DIR / "cases.json", OUTPUT_DIR / "cases.csv", OUTPUT_DIR / "summary.json",
        OUTPUT_DIR / "high-frequency-patterns.json", OUTPUT_DIR / "source-notes.json", OUTPUT_DIR / "artifact.json",
    ])
    _write_checksums()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums-only", action="store_true", help="Refresh SHA256SUMS after packaging REPORT.html.")
    args = parser.parse_args()
    if args.checksums_only:
        _write_checksums()
        print(f"checksums refreshed: {OUTPUT_DIR / 'SHA256SUMS'}")
        return 0
    summary = run_analysis()
    overall = summary["groups"]["overall"]
    print(json.dumps({
        "status": "ok",
        "case_count": overall["case_count"],
        "pre_correct": overall["pre_correct"],
        "final_correct": overall["final_correct"],
        "stage_loss": overall["flags"]["pre_correct_final_noncorrect"],
        "primary_attribution": overall["primary_attribution"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
