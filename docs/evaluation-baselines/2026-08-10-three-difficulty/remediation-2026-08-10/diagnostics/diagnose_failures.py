"""Deterministically diagnose failed cases in the three live baselines.

The classifier is stage- and signal-based.  It never branches on a case id and
does not modify datasets, run artifacts, gates, or production code.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard")
TAXONOMY = (
    "retrieval_insufficient",
    "retrieval_unclear_or_noisy",
    "context_loss",
    "intermediate_pipeline_loss",
    "evaluator_error",
    "generation_or_reasoning",
    "dataset_reference_issue",
    "other",
)
SCHEMA_VERSION = "stt.eval.failure-diagnostics.v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "baseline-summary.json"


class DiagnosticError(RuntimeError):
    """Raised when source artifacts are incomplete or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DiagnosticError(f"required source is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"invalid JSON at {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _one_line(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("m²", "m2").replace("㎡", "m2").replace("℃", "°c")
    return "".join(
        char
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _fuzzy_coverage(needle: Any, haystack: Any) -> float:
    """Return the fraction of normalized needle aligned in ordered blocks."""
    expected = _normalize_text(needle)
    actual = _normalize_text(haystack)
    if not expected:
        return 1.0
    if not actual:
        return 0.0
    matcher = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(1.0, matched / len(expected))


def _coverage_summary(terms: Sequence[Any], text: Any) -> dict[str, Any]:
    scores = [_fuzzy_coverage(term, text) for term in terms if str(term or "").strip()]
    return {
        "term_count": len(scores),
        "average": sum(scores) / len(scores) if scores else 1.0,
        "minimum": min(scores) if scores else 1.0,
        "scores": scores,
    }


def _candidate_ids(candidate: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("chunk_id", "source_chunk_id", "evidence_id"):
        if candidate.get(key):
            values.add(str(candidate[key]))
    evidence = _mapping(candidate.get("evidence"))
    for key in ("chunk_id", "source_chunk_id", "evidence_id"):
        if evidence.get(key):
            values.add(str(evidence[key]))
    return values


def _candidate_hashes(candidate: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for owner in (candidate, _mapping(candidate.get("evidence"))):
        if owner.get("content_hash"):
            values.add(str(owner["content_hash"]))
    return values


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    evidence = _mapping(candidate.get("evidence"))
    return str(
        candidate.get("security_safe_content")
        or candidate.get("content")
        or candidate.get("text")
        or evidence.get("content")
        or ""
    )


def _stage_signal(
    items: Sequence[Any],
    ground_truth_ids: set[str],
    ground_truth_hashes: set[str],
) -> dict[str, Any]:
    direct_positions: list[int] = []
    equivalent_positions: list[int] = []
    texts: list[str] = []
    for index, raw in enumerate(items, start=1):
        candidate = _mapping(raw)
        texts.append(_candidate_text(candidate))
        if ground_truth_ids & _candidate_ids(candidate):
            direct_positions.append(index)
        elif ground_truth_hashes & _candidate_hashes(candidate):
            equivalent_positions.append(index)
    return {
        "candidate_count": len(items),
        "direct_ground_truth_present": bool(direct_positions),
        "content_equivalent_present": bool(equivalent_positions),
        "ground_truth_present": bool(direct_positions or equivalent_positions),
        "direct_positions": direct_positions,
        "content_equivalent_positions": equivalent_positions,
        "combined_text": "\n".join(texts),
    }


def _old_gate_reasons(result: Mapping[str, Any]) -> list[str]:
    if result.get("error_type") or result.get("error_detail"):
        return [f"runtime_error:{result.get('error_type') or 'unknown'}"]
    evaluated = _mapping(result.get("metrics"))
    metric_groups = _mapping(evaluated.get("metrics"))
    if bool(evaluated.get("unanswerable")):
        correct = _finite(_mapping(metric_groups.get("abstention")).get("correct"))
        if correct is None:
            return ["abstention_unknown"]
        return [] if correct >= 1.0 else ["abstention_incorrect"]
    reasons: list[str] = []
    generation = _mapping(metric_groups.get("generation"))
    retrieval = _mapping(metric_groups.get("retrieval"))
    expected_recall = _finite(generation.get("expected_term_recall"))
    forbidden_rate = _finite(generation.get("forbidden_term_hit_rate"))
    recall_map = _mapping(retrieval.get("recall_at_k"))
    recall_10 = _finite(recall_map.get("10"))
    if expected_recall is None:
        reasons.append("expected_term_recall_unknown")
    elif expected_recall < 0.75:
        reasons.append("expected_term_recall_below_0.75")
    if forbidden_rate is not None and forbidden_rate > 0:
        reasons.append("forbidden_term_hit")
    if recall_10 is None:
        reasons.append("retrieval_recall_at_10_unknown")
    elif recall_10 <= 0:
        reasons.append("retrieval_recall_at_10_zero")
    return reasons or ["case_gate_failed_without_diagnostic"]


def _confidence(level: str, basis: str) -> dict[str, Any]:
    score = {"high": 0.9, "medium": 0.7, "low": 0.45}[level]
    return {"level": level, "score": score, "basis": basis}


def _semantic_correctness(signals: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate answer correctness independently from the historical pass flag.

    This deliberately combines reference-answer and authorized-evidence checks;
    expected terms are one signal, not the sole truth source.
    """
    answer_terms = _mapping(signals.get("answer_term_proxy"))
    term_average = _finite(answer_terms.get("average")) or 0.0
    term_minimum = _finite(answer_terms.get("minimum")) or 0.0
    reference = _finite(signals.get("answer_reference_proxy")) or 0.0
    target_reference = _finite(signals.get("target_reference_proxy")) or 0.0
    abstained = bool(signals.get("abstained"))
    expected_recall = _finite(signals.get("expected_term_recall"))

    if not abstained and (
        reference >= 0.90
        or (term_average >= 0.90 and term_minimum >= 0.70 and target_reference >= 0.70)
    ):
        return {
            "status": "likely_correct",
            "score": max(reference, term_average),
            "basis": (
                "actual answer closely covers the reference answer and/or expected concepts "
                "supported by authorized evidence"
            ),
            "method": "deterministic_reference_and_evidence_proxy",
            "independent_of_historical_pass": True,
        }
    if not abstained and (
        reference >= 0.70
        or term_average >= 0.70
        or (expected_recall is not None and expected_recall >= 0.75)
    ):
        return {
            "status": "partially_correct",
            "score": max(reference, term_average, expected_recall or 0.0),
            "basis": "actual answer covers material but not all reference/evidence-supported aspects",
            "method": "deterministic_reference_and_evidence_proxy",
            "independent_of_historical_pass": True,
        }
    if abstained or max(reference, term_average) < 0.50:
        return {
            "status": "likely_incorrect_or_incomplete",
            "score": max(reference, term_average),
            "basis": "actual answer abstained or omitted most reference/evidence-supported content",
            "method": "deterministic_reference_and_evidence_proxy",
            "independent_of_historical_pass": True,
        }
    return {
        "status": "unresolved",
        "score": max(reference, term_average),
        "basis": "textual evidence is mixed and requires semantic review",
        "method": "deterministic_reference_and_evidence_proxy",
        "independent_of_historical_pass": True,
    }


def _classify(signals: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    stages = _mapping(signals.get("stages"))
    retrieved = _mapping(stages.get("retrieved"))
    reranked = _mapping(stages.get("reranked"))
    context = _mapping(stages.get("context"))
    sources = _mapping(stages.get("sources"))
    recall_10 = _finite(signals.get("retrieval_recall_at_10"))
    expected_recall = _finite(signals.get("expected_term_recall"))
    answer_proxy = _finite(_mapping(signals.get("answer_term_proxy")).get("average")) or 0.0
    reference_proxy = _finite(signals.get("answer_reference_proxy")) or 0.0
    target_proxy = _finite(_mapping(signals.get("target_term_proxy")).get("average")) or 0.0
    candidate_proxy = _finite(_mapping(signals.get("retrieved_term_proxy")).get("average")) or 0.0
    decision_reason = str(signals.get("decision_reason") or "")
    abstained = bool(signals.get("abstained"))
    error_present = bool(signals.get("error_present"))
    post_generation = decision_reason.startswith("POST_GENERATION_")
    evaluator_signal = (
        expected_recall is not None
        and expected_recall < 0.75
        and (answer_proxy >= 0.90 or reference_proxy >= 0.90)
    )

    if error_present:
        return "other", [], _confidence("high", "API result contains a runtime error")

    if decision_reason == "QUERY_SCOPE_AMBIGUOUS":
        return (
            "intermediate_pipeline_loss",
            ["generation_or_reasoning"],
            _confidence("high", "scope decision stopped retrieval for a concrete evaluated query"),
        )

    if recall_10 is not None and recall_10 <= 0 and not retrieved.get("ground_truth_present"):
        equivalent_or_answered = (
            retrieved.get("content_equivalent_present")
            or candidate_proxy >= 0.90
            or (expected_recall is not None and expected_recall >= 0.75)
            or answer_proxy >= 0.90
        )
        if equivalent_or_answered:
            return (
                "dataset_reference_issue",
                ["evaluator_error"],
                _confidence(
                    "high" if retrieved.get("content_equivalent_present") else "medium",
                    "ground-truth id recall is zero while equivalent content or the expected answer is present",
                ),
            )
        secondary = ["dataset_reference_issue"] if target_proxy < 0.60 else []
        return (
            "retrieval_insufficient",
            secondary,
            _confidence("high", "no ground-truth or content-equivalent item reached retrieved candidates"),
        )

    if retrieved.get("ground_truth_present") and not reranked.get("ground_truth_present"):
        return (
            "retrieval_unclear_or_noisy",
            ["context_loss"],
            _confidence("high", "ground truth was retrieved but did not survive reranking"),
        )

    if reranked.get("ground_truth_present") and not context.get("ground_truth_present"):
        return (
            "context_loss",
            ["retrieval_unclear_or_noisy"],
            _confidence("high", "ground truth survived reranking but was absent from packed context"),
        )

    if context.get("ground_truth_present") and not sources.get("ground_truth_present"):
        if decision_reason == "UNSUPPORTED_GENERATION":
            secondary = ["intermediate_pipeline_loss"]
            if recall_10 is not None and 0 < recall_10 < 1:
                secondary.append("retrieval_unclear_or_noisy")
            return (
                "generation_or_reasoning",
                secondary,
                _confidence(
                    "high" if signals.get("evidence_assessment_status") == "SUFFICIENT" else "medium",
                    "ground truth reached context but generation was rejected as unsupported and abstained",
                ),
            )
        secondary = ["generation_or_reasoning"]
        if recall_10 is not None and 0 < recall_10 < 1:
            secondary.append("retrieval_unclear_or_noisy")
        return (
            "intermediate_pipeline_loss",
            secondary,
            _confidence(
                "high" if abstained or decision_reason == "UNSUPPORTED_GENERATION" else "medium",
                "ground truth reached context but did not survive into final answer sources",
            ),
        )

    if evaluator_signal:
        secondary = ["intermediate_pipeline_loss"] if post_generation else ["generation_or_reasoning"]
        return (
            "evaluator_error",
            secondary,
            _confidence(
                "high",
                "deterministic gate failed while normalized answer coverage is at least 0.90",
            ),
        )

    if recall_10 is not None and 0 < recall_10 < 1:
        secondary = ["generation_or_reasoning"]
        if post_generation:
            secondary.insert(0, "intermediate_pipeline_loss")
        return (
            "retrieval_unclear_or_noisy",
            secondary,
            _confidence("high", "retrieval recall is partial rather than complete"),
        )

    if post_generation:
        secondary = ["generation_or_reasoning"]
        if answer_proxy >= 0.75 or reference_proxy >= 0.75:
            secondary.append("evaluator_error")
        return (
            "intermediate_pipeline_loss",
            secondary,
            _confidence(
                "medium",
                "a named post-generation repair/projection changed an evidence-complete answer path",
            ),
        )

    if context.get("ground_truth_present"):
        secondary = []
        if answer_proxy >= 0.75 or reference_proxy >= 0.75:
            secondary.append("evaluator_error")
        return (
            "generation_or_reasoning",
            secondary,
            _confidence("medium", "ground truth reached context but the final response missed gate terms"),
        )

    if recall_10 is None:
        return (
            "other",
            ["retrieval_insufficient"],
            _confidence("low", "retrieval diagnostics are incomplete"),
        )

    return (
        "other",
        ["generation_or_reasoning"],
        _confidence("low", "available deterministic signals do not isolate one stage"),
    )


def _diagnose_case(
    difficulty: str,
    result: Mapping[str, Any],
    case: Mapping[str, Any],
    critical_ids: set[str],
) -> dict[str, Any]:
    observation = _mapping(result.get("observation"))
    evaluated = _mapping(result.get("metrics"))
    metric_groups = _mapping(evaluated.get("metrics"))
    generation = _mapping(metric_groups.get("generation"))
    retrieval_metrics = _mapping(metric_groups.get("retrieval"))
    recall_at_k = _mapping(retrieval_metrics.get("recall_at_k"))
    answer_decision = _mapping(observation.get("answer_decision"))
    evidence_assessment = _mapping(observation.get("evidence_assessment"))
    grounding = _mapping(observation.get("grounding"))

    relevant_evidence = [_mapping(item) for item in _sequence(case.get("relevant_evidence"))]
    ground_truth_ids = {
        str(value) for value in _sequence(case.get("relevant_chunk_ids")) if value
    }
    ground_truth_hashes: set[str] = set()
    for item in relevant_evidence:
        for key in ("chunk_id", "evidence_id"):
            if item.get(key):
                ground_truth_ids.add(str(item[key]))
        if item.get("content_hash"):
            ground_truth_hashes.add(str(item["content_hash"]))

    stage_names = ("retrieved", "rrf", "reranked", "context", "sources")
    stages = {
        name: _stage_signal(
            _sequence(observation.get(name)), ground_truth_ids, ground_truth_hashes
        )
        for name in stage_names
    }
    expected_terms = [str(value) for value in _sequence(case.get("expected_terms"))]
    answer = str(observation.get("answer") or "")
    reference_answer = str(case.get("reference_answer") or "")
    ground_truth_text = "\n".join(str(item.get("text") or "") for item in relevant_evidence)
    signals: dict[str, Any] = {
        "error_present": bool(result.get("error_type") or result.get("error_detail")),
        "retrieval_recall_at_10": _finite(recall_at_k.get("10")),
        "expected_term_recall": _finite(generation.get("expected_term_recall")),
        "answer_term_proxy": _coverage_summary(expected_terms, answer),
        "answer_reference_proxy": _fuzzy_coverage(reference_answer, answer),
        "target_term_proxy": _coverage_summary(expected_terms, ground_truth_text),
        "target_reference_proxy": _fuzzy_coverage(reference_answer, ground_truth_text),
        "retrieved_term_proxy": _coverage_summary(
            expected_terms, _mapping(stages.get("retrieved")).get("combined_text", "")
        ),
        "decision_reason": answer_decision.get("reason_code"),
        "decision_action": answer_decision.get("action"),
        "abstained": bool(observation.get("abstained")),
        "evidence_assessment_status": evidence_assessment.get("status"),
        "grounding_status": grounding.get("status"),
        "stages": stages,
    }
    primary, secondary, confidence = _classify(signals)
    semantic_correctness = _semantic_correctness(signals)
    if primary not in TAXONOMY or any(label not in TAXONOMY for label in secondary):
        raise DiagnosticError(f"invalid taxonomy label for {result.get('case_id')}")

    stage_code = "".join(
        name[0].upper() if _mapping(stages.get(name)).get("ground_truth_present") else "-"
        for name in ("retrieved", "reranked", "context", "sources")
    )
    evidence_lines = [
        f"old_gate={','.join(_old_gate_reasons(result))}",
        f"stage={stage_code}; r10={signals['retrieval_recall_at_10']}",
        (
            "expected_recall="
            f"{signals['expected_term_recall']}; normalized_proxy="
            f"{signals['answer_term_proxy']['average']:.3f}"
        ),
        (
            f"decision={signals['decision_reason'] or 'unknown'}; "
            f"assessment={signals['evidence_assessment_status'] or 'unknown'}; "
            f"grounding={signals['grounding_status'] or 'unknown'}"
        ),
    ]

    return {
        "case_id": str(result.get("case_id") or case.get("case_id") or ""),
        "difficulty": difficulty,
        "critical": str(result.get("case_id") or "") in critical_ids,
        "passed": result.get("passed"),
        "primary": primary,
        "secondary": list(dict.fromkeys(label for label in secondary if label != primary)),
        "confidence": confidence,
        "semantic_correctness": semantic_correctness,
        "old_gate_reasons": _old_gate_reasons(result),
        "evidence_summary": evidence_lines,
        "evidence": {
            "query": _one_line(case.get("query"), 600),
            "reference_answer": _one_line(reference_answer, 1000),
            "expected_terms": expected_terms,
            "answer": _one_line(answer, 1600),
            "relevant_evidence_excerpt": _one_line(ground_truth_text, 1200),
            "metrics": {
                "retrieval_recall_at_1": _finite(recall_at_k.get("1")),
                "retrieval_recall_at_3": _finite(recall_at_k.get("3")),
                "retrieval_recall_at_10": signals["retrieval_recall_at_10"],
                "expected_term_recall": signals["expected_term_recall"],
                "generation_correctness": _finite(generation.get("correctness")),
                "generation_faithfulness": _finite(generation.get("faithfulness")),
            },
            "text_checks": {
                "answer_term_proxy": signals["answer_term_proxy"],
                "answer_reference_proxy": signals["answer_reference_proxy"],
                "target_term_proxy": signals["target_term_proxy"],
                "target_reference_proxy": signals["target_reference_proxy"],
                "retrieved_term_proxy": signals["retrieved_term_proxy"],
                "method": "NFKC normalized ordered-block character coverage; diagnostic proxy, not a semantic judge",
            },
            "stage_trace": {
                name: {
                    key: value
                    for key, value in _mapping(stages.get(name)).items()
                    if key != "combined_text"
                }
                for name in stage_names
            },
            "stage_code": stage_code,
            "decision": {
                "action": signals["decision_action"],
                "reason_code": signals["decision_reason"],
                "abstained": signals["abstained"],
                "degradation_reasons": _sequence(
                    _mapping(observation.get("answer_envelope")).get("degradation_reasons")
                ),
            },
            "assessment": {
                "status": signals["evidence_assessment_status"],
                "reasons": _sequence(evidence_assessment.get("reasons")),
                "missing_aspects": _sequence(evidence_assessment.get("missing_aspects")),
                "query_coverage": evidence_assessment.get("query_coverage"),
                "packed_evidence_count": evidence_assessment.get("packed_evidence_count"),
                "grounding_status": signals["grounding_status"],
                "grounding_reason": grounding.get("reason_code"),
            },
            "ground_truth": {
                "ids": sorted(ground_truth_ids),
                "content_hashes": sorted(ground_truth_hashes),
            },
            "tags": [str(value) for value in _sequence(case.get("tags"))],
            "error_type": result.get("error_type"),
            "error_detail": _one_line(result.get("error_detail"), 800) or None,
        },
    }


def _counts_by_difficulty(cases: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for difficulty in (*DIFFICULTIES, "overall"):
        selected = cases if difficulty == "overall" else [c for c in cases if c["difficulty"] == difficulty]
        counts: Counter[str] = Counter()
        for case in selected:
            value = case.get(field)
            if isinstance(value, Sequence) and not isinstance(value, str):
                counts.update(str(item) for item in value)
            elif value:
                counts[str(value)] += 1
        output[difficulty] = {
            "failed_case_count": len(selected),
            "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }
    return output


def _pattern_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    ids: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for case in cases:
        evidence = _mapping(case.get("evidence"))
        decision = _mapping(evidence.get("decision"))
        key = (
            str(case.get("difficulty")),
            str(evidence.get("stage_code") or "----"),
            str(decision.get("reason_code") or "unknown"),
        )
        counter[key] += 1
        ids[key].append(str(case.get("case_id")))
    return [
        {
            "difficulty": key[0],
            "stage_code": key[1],
            "decision_reason": key[2],
            "count": count,
            "case_ids": sorted(ids[key]),
        }
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _tag_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for difficulty in (*DIFFICULTIES, "overall"):
        selected = cases if difficulty == "overall" else [c for c in cases if c["difficulty"] == difficulty]
        counts: Counter[str] = Counter()
        for case in selected:
            counts.update(str(tag) for tag in _mapping(case.get("evidence")).get("tags", []))
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "difficulty": difficulty,
                    "tag": tag,
                    "count": count,
                    "failed_case_count": len(selected),
                    "share": count / len(selected) if selected else 0.0,
                }
            )
    return output


def _summary_markdown(report: Mapping[str, Any]) -> str:
    cases = [dict(case) for case in _sequence(report.get("cases"))]
    primary = _mapping(_mapping(report.get("summary")).get("primary_labels"))
    overall = _mapping(primary.get("overall"))
    overall_counts = _mapping(overall.get("counts"))
    top_labels = sorted(overall_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:4]
    total = len(cases)
    high = sum(1 for case in cases if _mapping(case.get("confidence")).get("level") == "high")
    medium = sum(1 for case in cases if _mapping(case.get("confidence")).get("level") == "medium")
    low = total - high - medium
    semantic_groups = _mapping(_mapping(report.get("summary")).get("semantic_correctness"))
    semantic_overall = _mapping(_mapping(semantic_groups.get("overall")).get("counts"))

    lines = [
        "# 三难度 live 基线逐题失败诊断",
        "",
        "## 技术摘要",
        "",
        (
            f"本报告只读诊断 easy 22、medium 29、hard 41，共 {total} 个旧 gate 失败 case。"
            f"确定性阶段证据给出高/中/低置信度 {high}/{medium}/{low} 个；"
            f"独立答案判断为 likely_correct {semantic_overall.get('likely_correct', 0)}、"
            f"partially_correct {semantic_overall.get('partially_correct', 0)}、"
            f"likely_incorrect_or_incomplete {semantic_overall.get('likely_incorrect_or_incomplete', 0)}、"
            f"unresolved {semantic_overall.get('unresolved', 0)}。"
            "结论是描述性根因归因，不是因果实验，也未调用 LLM judge。"
        ),
        "",
        "主要归因：" + "；".join(f"`{label}` {count} 个" for label, count in top_labels) + "。",
        "",
        "## 失败主要集中在哪些阶段",
        "",
        "| primary | easy | medium | hard | overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in TAXONOMY:
        values = []
        for difficulty in (*DIFFICULTIES, "overall"):
            values.append(int(_mapping(_mapping(primary.get(difficulty)).get("counts")).get(label, 0)))
        lines.append(f"| `{label}` | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")

    top_patterns = [dict(row) for row in _sequence(report.get("patterns"))[:10]]
    generic_tags = {
        "live-baseline", "manual", "printer", "zh-CN",
        "difficulty-easy", "difficulty-medium", "difficulty-hard",
    }
    top_tags = [
        dict(row)
        for row in _sequence(report.get("case_tags"))
        if _mapping(row).get("difficulty") == "overall"
        and _mapping(row).get("tag") not in generic_tags
    ][:10]
    lines.extend(
        [
            "",
            "## 高频标签与链路模式",
            "",
            "| difficulty | stage | decision | count |",
            "|---|---|---|---:|",
        ]
    )
    for row in top_patterns:
        lines.append(
            f"| {row['difficulty']} | `{row['stage_code']}` | `{row['decision_reason']}` | {row['count']} |"
        )
    lines.extend(["", "| task tag | failed cases | share of 92 |", "|---|---:|---:|"])
    for row in top_tags:
        lines.append(f"| `{row['tag']}` | {row['count']} | {float(row['share']):.1%} |")

    lines.extend(
        [
            "",
            (
                "阶段码按 `retrieved/reranked/context/sources` 顺序显示，字母表示 ground-truth id "
                "或等价 content hash 在该阶段存在，`-` 表示不存在。该表示能区分检索不足、重排/上下文丢失和最终答案链路丢失。"
            ),
            "",
            "## easy 与 medium 的逐题证据",
            "",
            "以下是本次修复优先范围；完整结构化证据、原文切片和阶段位置见 `failure-diagnostics.json`。",
            "",
        ]
    )
    for difficulty in ("easy", "medium"):
        lines.extend(
            [
                f"### {difficulty}（{sum(1 for c in cases if c['difficulty'] == difficulty)} 个）",
                "",
                "| case | critical | semantic correctness | primary | secondary | confidence | evidence slice |",
                "|---|:---:|---|---|---|:---:|---|",
            ]
        )
        for case in cases:
            if case["difficulty"] != difficulty:
                continue
            secondary = ", ".join(f"`{value}`" for value in case.get("secondary", [])) or "—"
            evidence = "; ".join(case.get("evidence_summary", [])[:3]).replace("|", "\\|")
            semantic = _mapping(case.get("semantic_correctness")).get("status")
            lines.append(
                f"| {case['case_id']} | {'Y' if case['critical'] else 'N'} | `{semantic}` | `{case['primary']}` | "
                f"{secondary} | {_mapping(case['confidence']).get('level')} | {evidence} |"
            )

    hard_counts = _mapping(_mapping(primary.get("hard")).get("counts"))
    lines.extend(
        [
            "",
            "## hard 作为同链路对照",
            "",
            "hard 的 41 个失败也按同一无题号规则分析，用于验证归因规则没有只针对 easy/medium 调参。",
            "",
            "| primary | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(hard_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| `{label}` | {count} |")

    lines.extend(
        [
            "",
            "## 范围、数据与标签定义",
            "",
            "- 范围：三条已保存的 `live/full` 基线，仅分析 API `results.json` 中 `passed=false` 的 case。",
            "- 粒度：一行一个失败 case；primary 互斥，secondary 可多选。",
            "- `semantic_correctness` 独立于历史 `passed`，同时参考 reference answer、授权 evidence 和实际 answer；expected terms 不是唯一真值。",
            "- `retrieval_insufficient`：ground truth/等价内容未进入 retrieved。",
            "- `retrieval_unclear_or_noisy`：只召回部分 ground truth，或 retrieved 后在 rerank 丢失。",
            "- `context_loss`：ground truth 通过 rerank 但未进入 packed context。",
            "- `intermediate_pipeline_loss`：context 已有 ground truth，但 final sources/decision/repair 未保留，或 scope 决策提前终止。",
            "- `evaluator_error`：旧 exact-term gate 失败，但 NFKC 规范化的答案覆盖代理达到 0.90。",
            "- `generation_or_reasoning`：证据已进 context，最终作答仍未覆盖目标，且无更强的阶段丢失证据。",
            "- `dataset_reference_issue`：ground-truth id recall 为 0，但等价内容或预期答案存在；或目标证据本身不能充分支持 expected terms。",
            "- `other`：运行错误或证据不足以定位到上述阶段。",
            "",
            "## 方法与可复现性",
            "",
            "脚本以 dataset manifest 的 relevant chunk/evidence id 与 content hash 为 ground truth，逐级核对 `retrieved → reranked → context → sources`。旧 gate 原因按现有 summarizer 的阈值重算；文本等价仅使用通用 NFKC、单位规范化和有序字符块覆盖率，不含 case id 分支。运行：",
            "",
            "```powershell",
            "uv run python docs/evaluation-baselines/2026-08-10-three-difficulty/remediation-2026-08-10/diagnostics/diagnose_failures.py",
            "```",
            "",
            "未绘制只有三个难度点的分类图：这种图不会比精确频数表增加诊断信息。频率明细分别保存在 `label-frequency.csv`、`pattern-frequency.csv` 和 `tag-frequency.csv`。",
            "",
            "## 限制、不确定性与稳健性",
            "",
            "- 文本覆盖代理不是语义 judge；`evaluator_error`、`dataset_reference_issue` 和部分 post-generation 归因需要人工或 LLM 复核。",
            "- saved results 没有修复前原始生成草稿，因此 post-generation repair 与 generation 本身的贡献不能完全拆分。",
            "- ground-truth id 不命中但内容等价时，归为 reference/evaluator 风险；这不证明 dataset 一定错误。",
            "- 本报告诊断旧 gate 失败，不重新定义 pass，也不代表修复后指标会自动提升。",
            "- 三难度共享同一 KB/index/model/mode，减少运行环境混杂，但没有重复运行来估计随机波动。",
            "",
            "## 建议的下一步",
            "",
            "1. 先处理高置信度 `intermediate_pipeline_loss`：对 scope、abstention、post-generation repair 做阶段回放，确保 ground truth 已在 context 时不会被无依据丢弃。",
            "2. 对高置信度 `retrieval_insufficient`/`context_loss` 建立不含 case id 的查询与排序回归测试。",
            "3. 将 `evaluator_error` 与 `dataset_reference_issue` 队列做盲审；确认后再调整通用 normalization/semantic judge 或修订 reference，不直接改题以迎合当前答案。",
            "4. 修复后用同一三套数据、同一 KB/index/model/mode 重跑，并比较 primary 标签是否迁移，而不只看总 pass rate。",
            "",
            "## 仍需回答的问题",
            "",
            "- post-generation repair 前的原始生成文本能否从 trace 或日志恢复，以区分生成失败与修复丢失？",
            "- ground-truth chunk id 是否应允许 content-hash/duplicate-group 等价匹配？",
            "- 旧 expected-term gate 是否需要独立语义 judge 作为复核层，而不是替换可审计的确定性 gate？",
            "",
            "## 数据来源",
            "",
            "来源和 SHA-256 固化在 `failure-diagnostics.json` 的 `sources` 字段。LLM 复核输入为 `llm-review-queue.jsonl`；本次没有调用，亦没有生成 LLM judgment。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise DiagnosticError("unexpected diagnostic schema")
    cases = _sequence(report.get("cases"))
    identifiers = [str(_mapping(case).get("case_id") or "") for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise DiagnosticError("duplicate case ids in diagnostic output")
    expected_counts = {"easy": 22, "medium": 29, "hard": 41}
    actual_counts = Counter(str(_mapping(case).get("difficulty")) for case in cases)
    if dict(actual_counts) != expected_counts:
        raise DiagnosticError(f"failed-case counts changed: {dict(actual_counts)} != {expected_counts}")
    for raw in cases:
        case = _mapping(raw)
        if case.get("primary") not in TAXONOMY:
            raise DiagnosticError(f"missing primary label: {case.get('case_id')}")
        confidence = _mapping(case.get("confidence"))
        if confidence.get("level") not in {"high", "medium", "low"}:
            raise DiagnosticError(f"invalid confidence: {case.get('case_id')}")
        semantic = _mapping(case.get("semantic_correctness"))
        if semantic.get("status") not in {
            "likely_correct", "partially_correct", "likely_incorrect_or_incomplete", "unresolved"
        }:
            raise DiagnosticError(f"invalid semantic correctness: {case.get('case_id')}")


def build_report() -> dict[str, Any]:
    summary = _read_json(SUMMARY_PATH)
    if summary.get("schema_version") != "stt.eval.three-difficulty-baseline-summary.v1":
        raise DiagnosticError("baseline-summary.json has an unsupported schema")
    source_baselines = _mapping(summary.get("source_baselines"))
    if source_baselines.get("status") != "succeeded":
        raise DiagnosticError("all live baselines must be succeeded before diagnosis")

    sources: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    difficulties = _mapping(summary.get("difficulties"))
    for difficulty in DIFFICULTIES:
        baseline = _mapping(difficulties.get(difficulty))
        run_id = str(baseline.get("run_id") or "")
        if not run_id:
            raise DiagnosticError(f"missing run id for {difficulty}")
        dataset_path = ROOT / "datasets" / f"{difficulty}.json"
        results_path = ROOT / "runs" / difficulty / run_id / "api" / "results.json"
        dataset = _read_json(dataset_path)
        payload = _read_json(results_path)
        dataset_cases = {
            str(_mapping(case).get("case_id")): _mapping(case)
            for case in _sequence(_mapping(dataset).get("cases"))
        }
        results = _sequence(_mapping(payload).get("data"))
        critical_ids = {
            str(value)
            for value in _sequence(_mapping(baseline.get("gate")).get("critical_failed_cases"))
        }
        for raw in results:
            result = _mapping(raw)
            if result.get("passed") is not False:
                continue
            case_id = str(result.get("case_id") or "")
            if case_id not in dataset_cases:
                raise DiagnosticError(f"result case absent from dataset: {case_id}")
            cases.append(_diagnose_case(difficulty, result, dataset_cases[case_id], critical_ids))
        for role, path in (("dataset_manifest", dataset_path), ("api_results", results_path)):
            sources.append(
                {
                    "difficulty": difficulty,
                    "role": role,
                    "path": _relative(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    cases.sort(key=lambda case: (DIFFICULTIES.index(case["difficulty"]), case["case_id"]))
    patterns = _pattern_rows(cases)
    tags = _tag_rows(cases)
    summary_block = {
        "failed_case_count": len(cases),
        "focus_case_count": sum(1 for case in cases if case["difficulty"] in {"easy", "medium"}),
        "critical_failed_case_count": sum(1 for case in cases if case["critical"]),
        "primary_labels": _counts_by_difficulty(cases, "primary"),
        "secondary_labels": _counts_by_difficulty(cases, "secondary"),
        "confidence": _counts_by_difficulty(
            [{**case, "confidence_level": _mapping(case.get("confidence")).get("level")} for case in cases],
            "confidence_level",
        ),
        "semantic_correctness": _counts_by_difficulty(
            [
                {
                    **case,
                    "semantic_status": _mapping(case.get("semantic_correctness")).get("status"),
                }
                for case in cases
            ],
            "semantic_status",
        ),
        "decision_reasons": {
            difficulty: dict(
                Counter(
                    str(_mapping(_mapping(case.get("evidence")).get("decision")).get("reason_code") or "unknown")
                    for case in cases
                    if difficulty == "overall" or case["difficulty"] == difficulty
                ).most_common()
            )
            for difficulty in (*DIFFICULTIES, "overall")
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_from": summary.get("generated_at"),
        "suite_key": summary.get("suite_key"),
        "scope": {
            "mode": "live",
            "evaluation_scope": "full",
            "difficulties": list(DIFFICULTIES),
            "population": "saved API results with passed=false",
            "focus": ["easy", "medium"],
            "read_only": True,
            "llm_judge_called": False,
            "case_id_special_casing": False,
        },
        "taxonomy": list(TAXONOMY),
        "method": {
            "classifier": "deterministic-stage-diagnostics.v1",
            "stage_order": ["retrieved", "reranked", "context", "sources"],
            "text_proxy": "NFKC normalized ordered-block character coverage",
            "primary_is_mutually_exclusive": True,
            "secondary_may_repeat_across_cases": True,
            "visual_omission_reason": "Only three difficulty aggregates exist; exact label/pattern tables are more informative than a three-point categorical chart.",
        },
        "summary": summary_block,
        "patterns": patterns,
        "case_tags": tags,
        "cases": cases,
        "sources": sorted(sources, key=lambda item: (item["difficulty"], item["role"])),
        "limitations": [
            "The normalized text proxy is heuristic and is not a semantic or LLM judge.",
            "Saved results do not expose the pre-repair generation draft, so generation and post-generation effects can overlap.",
            "Content-equivalent retrieval can indicate a dataset/evaluator reference issue but does not prove the dataset is wrong.",
            "The analysis explains old gate failures without redefining pass/fail or estimating causal lift.",
        ],
    }
    _validate(report)
    return report


def write_outputs(report: Mapping[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [dict(case) for case in _sequence(report.get("cases"))]
    _write_json(OUTPUT_DIR / "failure-diagnostics.json", report)
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        evidence = _mapping(case.get("evidence"))
        metrics = _mapping(evidence.get("metrics"))
        decision = _mapping(evidence.get("decision"))
        confidence = _mapping(case.get("confidence"))
        semantic = _mapping(case.get("semantic_correctness"))
        case_rows.append(
            {
                "difficulty": case.get("difficulty"),
                "case_id": case.get("case_id"),
                "critical": case.get("critical"),
                "primary": case.get("primary"),
                "secondary": ";".join(case.get("secondary", [])),
                "confidence": confidence.get("level"),
                "confidence_score": confidence.get("score"),
                "confidence_basis": confidence.get("basis"),
                "semantic_correctness": semantic.get("status"),
                "semantic_correctness_score": semantic.get("score"),
                "semantic_correctness_basis": semantic.get("basis"),
                "old_gate_reasons": ";".join(case.get("old_gate_reasons", [])),
                "stage_code": evidence.get("stage_code"),
                "retrieval_recall_at_10": metrics.get("retrieval_recall_at_10"),
                "expected_term_recall": metrics.get("expected_term_recall"),
                "decision_reason": decision.get("reason_code"),
                "abstained": decision.get("abstained"),
                "query": evidence.get("query"),
                "reference_answer": evidence.get("reference_answer"),
                "answer": evidence.get("answer"),
                "evidence_summary": " | ".join(case.get("evidence_summary", [])),
            }
        )
    _write_csv(
        OUTPUT_DIR / "failure-diagnostics.csv",
        case_rows,
        (
            "difficulty", "case_id", "critical", "primary", "secondary", "confidence",
            "confidence_score", "confidence_basis", "old_gate_reasons", "stage_code",
            "semantic_correctness", "semantic_correctness_score", "semantic_correctness_basis",
            "retrieval_recall_at_10", "expected_term_recall", "decision_reason", "abstained",
            "query", "reference_answer", "answer", "evidence_summary",
        ),
    )

    label_rows: list[dict[str, Any]] = []
    for label_kind in ("primary_labels", "secondary_labels", "confidence", "semantic_correctness"):
        groups = _mapping(_mapping(report.get("summary")).get(label_kind))
        for difficulty in (*DIFFICULTIES, "overall"):
            group = _mapping(groups.get(difficulty))
            failed_count = int(group.get("failed_case_count") or 0)
            for label, count in _mapping(group.get("counts")).items():
                label_rows.append(
                    {
                        "kind": label_kind,
                        "difficulty": difficulty,
                        "label": label,
                        "count": count,
                        "failed_case_count": failed_count,
                        "share": int(count) / failed_count if failed_count else 0.0,
                    }
                )
    _write_csv(
        OUTPUT_DIR / "label-frequency.csv",
        label_rows,
        ("kind", "difficulty", "label", "count", "failed_case_count", "share"),
    )
    pattern_rows = [
        {**dict(row), "case_ids": ";".join(_mapping(row).get("case_ids", []))}
        for row in _sequence(report.get("patterns"))
    ]
    _write_csv(
        OUTPUT_DIR / "pattern-frequency.csv",
        pattern_rows,
        ("difficulty", "stage_code", "decision_reason", "count", "case_ids"),
    )
    _write_csv(
        OUTPUT_DIR / "tag-frequency.csv",
        [dict(row) for row in _sequence(report.get("case_tags"))],
        ("difficulty", "tag", "count", "failed_case_count", "share"),
    )

    review_path = OUTPUT_DIR / "llm-review-queue.jsonl"
    temporary = review_path.with_suffix(review_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            confidence = _mapping(case.get("confidence"))
            if confidence.get("level") == "high" and case.get("primary") not in {
                "evaluator_error", "dataset_reference_issue"
            }:
                continue
            payload = {
                "case_id": case.get("case_id"),
                "difficulty": case.get("difficulty"),
                "allowed_labels": list(TAXONOMY),
                "deterministic_diagnosis": {
                    "primary": case.get("primary"),
                    "secondary": case.get("secondary"),
                    "confidence": confidence,
                    "semantic_correctness": case.get("semantic_correctness"),
                },
                "evidence": case.get("evidence"),
                "instruction": (
                    "Blindly review the evidence, choose one primary and zero or more secondary labels, "
                    "quote the decisive saved-result fields, and distinguish evaluator/reference issues "
                    "from model/pipeline failures. Do not use case-id-specific rules."
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, review_path)
    (OUTPUT_DIR / "REPORT.md").write_text(_summary_markdown(report), encoding="utf-8")

    checksums = []
    for name in (
        "diagnose_failures.py", "llm_judge_review.py",
        "failure-diagnostics.json", "failure-diagnostics.csv", "label-frequency.csv",
        "pattern-frequency.csv", "tag-frequency.csv", "llm-review-queue.jsonl", "REPORT.md",
    ):
        path = OUTPUT_DIR / name
        checksums.append(f"{_sha256(path)}  {name}")
    (OUTPUT_DIR / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")


def main() -> int:
    report = build_report()
    write_outputs(report)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "case_count": len(_sequence(report.get("cases"))),
                "output_dir": str(OUTPUT_DIR),
                "llm_judge_called": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
