#!/usr/bin/env python3
"""Read-only partial diagnosis for archived easy/medium in live-cand-max-mono-01.

Hard is intentionally excluded. Stored final-answer binary outcomes are
authoritative; pre-grounding correctness is recomputed with the current project
evaluator at the same 0.75 threshold. No network or LLM calls are made.
"""

from __future__ import annotations

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
RUN_LABEL = "live-cand-max-mono-01"
RUN_ROOT = REMEDIATION_ROOT / "runs" / RUN_LABEL
RUN_INDEX_PATH = RUN_ROOT / "run-index.json"
DIFFICULTIES = ("easy", "medium")
THRESHOLD = 0.75
TARGET_COUNTS = {"easy": 48, "medium": 43}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        "percent": round(value * 100.0, 1) if value is not None else None,
    }


def generation_metrics(case: Mapping[str, Any], answer: str) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.evaluation.metrics import _generation_metrics

    metrics, _ = _generation_metrics(case, {"answer": answer}, [])
    return metrics


def issue_codes(observation: Mapping[str, Any]) -> list[str]:
    result: set[str] = set()
    for issue in nested(observation, "citation_validation", "issues", default=[]) or []:
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


def extract_row(
    *,
    difficulty: str,
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    line_number: int,
    dataset_sha256: str,
    results_sha256: str,
) -> dict[str, Any]:
    observation = record.get("observation") or {}
    outcome = nested(record, "metrics", "metrics", "case_outcome", default={}) or {}
    retrieval = nested(record, "metrics", "metrics", "retrieval", default={}) or {}
    pre_answer = observation.get("pre_grounding_answer")
    if not isinstance(pre_answer, str):
        raise ValueError(f"pre-grounding answer missing: {record.get('case_id')}")
    final_answer = observation.get("answer")
    if not isinstance(final_answer, str):
        raise ValueError(f"final answer missing: {record.get('case_id')}")
    pre_metrics = generation_metrics(case, pre_answer)
    pre_score = float(pre_metrics["correctness"])
    pre_correct = pre_score >= THRESHOLD
    stored_final_score = outcome.get("final_answer_correctness")
    stored_final_correct = outcome.get("final_answer_correct")
    if not isinstance(stored_final_score, (int, float)) or not isinstance(
        stored_final_correct, bool
    ):
        raise ValueError(f"stored outcome missing: {record.get('case_id')}")
    context = observation.get("context_pack_summary") or {}
    coverage = context.get("coverage") or {}
    uncovered = list(coverage.get("uncovered_sub_query_ids") or [])
    action, decision_reason = decision_fields(observation)
    row = {
        "schema_version": "stt.eval.max-mono-partial-case.v1",
        "case_id": str(record.get("case_id")),
        "difficulty": difficulty,
        "line_number": line_number,
        "query": str(case.get("query") or ""),
        "pre_grounding_answer": pre_answer,
        "final_visible_answer": final_answer,
        "pre_correctness_recomputed": pre_score,
        "pre_correct_recomputed": pre_correct,
        "stored_final_correctness": float(stored_final_score),
        "stored_final_correct": stored_final_correct,
        "transition": (
            "pre_correct_to_final_correct"
            if pre_correct and stored_final_correct
            else "pre_correct_to_final_wrong"
            if pre_correct
            else "pre_wrong_to_final_correct"
            if stored_final_correct
            else "pre_wrong_to_final_wrong"
        ),
        "grounding": {
            "changed": bool(nested(observation, "post_generation_change", "grounding_changed")),
            "status": nested(observation, "grounding", "status"),
            "reason_code": nested(observation, "grounding", "reason_code"),
            "decision_action": action,
            "decision_reason_code": decision_reason,
            "degradation_reasons": list(
                nested(observation, "answer_envelope", "degradation_reasons", default=[])
                or []
            ),
            "removed_character_count": int(
                nested(observation, "post_generation_change", "removed_character_count") or 0
            ),
            "added_character_count": int(
                nested(observation, "post_generation_change", "added_character_count") or 0
            ),
        },
        "citation": {
            "status": nested(observation, "citation_validation", "status"),
            "issue_codes": issue_codes(observation),
        },
        "retrieval": {
            "recall_at_5": nested(retrieval, "recall_at_k", "5"),
            "first_relevant_rank": retrieval.get("first_relevant_rank"),
            "retrieved_count": retrieval.get("retrieved_count"),
            "relevant_retrieved_count": retrieval.get("relevant_retrieved_count"),
        },
        "context": {
            "packed_count": context.get("packed_count"),
            "omitted_count": context.get("omitted_count"),
            "truncated_count": context.get("truncated_count"),
            "uncovered_sub_query_ids": uncovered,
            "content_hash": context.get("content_hash"),
        },
        "evidence": {
            "status": nested(observation, "evidence_assessment", "status"),
            "reasons": list(
                nested(observation, "evidence_assessment", "reasons", default=[]) or []
            ),
        },
        "integrity": {
            "dataset_sha256": dataset_sha256,
            "results_sha256": results_sha256,
            "pre_answer_sha256": sha_text(pre_answer),
            "final_answer_sha256": sha_text(final_answer),
        },
    }
    return row


def summarize(rows: list[dict[str, Any]], difficulty: str) -> dict[str, Any]:
    total = len(rows)
    final_correct = sum(row["stored_final_correct"] for row in rows)
    pre_correct = sum(row["pre_correct_recomputed"] for row in rows)
    pre_to_wrong = sum(row["transition"] == "pre_correct_to_final_wrong" for row in rows)
    pre_wrong_to_final = sum(row["transition"] == "pre_wrong_to_final_correct" for row in rows)
    oracle_union = sum(
        row["pre_correct_recomputed"] or row["stored_final_correct"] for row in rows
    )
    wrong = [row for row in rows if not row["stored_final_correct"]]
    target_count = TARGET_COUNTS[difficulty]
    transitions = Counter(row["transition"] for row in rows)
    return {
        "case_count": total,
        "stored_final_correct": rate(final_correct, total),
        "pre_grounding_correct_recomputed": rate(pre_correct, total),
        "pre_correct_to_final_wrong": rate(pre_to_wrong, total),
        "pre_wrong_to_final_correct": rate(pre_wrong_to_final, total),
        "transition_counts": dict(sorted(transitions.items())),
        "mean_stored_final_correctness": mean(row["stored_final_correctness"] for row in rows),
        "mean_pre_correctness_recomputed": mean(
            row["pre_correctness_recomputed"] for row in rows
        ),
        "preserve_pre_scenario": {
            "correct_count": pre_correct,
            "accuracy": pre_correct / total,
            "target_correct_count": target_count,
            "target_met": pre_correct >= target_count,
            "shortfall_cases": max(0, target_count - pre_correct),
            "delta_vs_stored_final_cases": pre_correct - final_correct,
        },
        "oracle_pre_or_final_scenario": {
            "correct_count": oracle_union,
            "accuracy": oracle_union / total,
            "target_correct_count": target_count,
            "target_met": oracle_union >= target_count,
            "shortfall_cases": max(0, target_count - oracle_union),
        },
        "all_grounding_decision_reasons": dict(
            sorted(
                Counter(str(row["grounding"]["decision_reason_code"]) for row in rows).items()
            )
        ),
        "wrong_grounding_decision_reasons": dict(
            sorted(
                Counter(str(row["grounding"]["decision_reason_code"]) for row in wrong).items()
            )
        ),
        "wrong_grounding_status": dict(
            sorted(Counter(str(row["grounding"]["status"]) for row in wrong).items())
        ),
        "wrong_grounding_reason_codes": dict(
            sorted(Counter(str(row["grounding"]["reason_code"]) for row in wrong).items())
        ),
        "wrong_grounding_changed": rate(
            sum(row["grounding"]["changed"] for row in wrong), len(wrong)
        ),
        "wrong_recall_at_5_full": rate(
            sum(row["retrieval"]["recall_at_5"] == 1.0 for row in wrong), len(wrong)
        ),
        "wrong_first_relevant_rank": dict(
            sorted(Counter(str(row["retrieval"]["first_relevant_rank"]) for row in wrong).items())
        ),
        "wrong_context_omitted": rate(
            sum(int(row["context"]["omitted_count"] or 0) > 0 for row in wrong), len(wrong)
        ),
        "wrong_context_truncated": rate(
            sum(int(row["context"]["truncated_count"] or 0) > 0 for row in wrong), len(wrong)
        ),
        "wrong_context_uncovered_subquery": rate(
            sum(bool(row["context"]["uncovered_sub_query_ids"]) for row in wrong),
            len(wrong),
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
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "difficulty",
        "stored_final_correct",
        "stored_final_correctness",
        "pre_correct_recomputed",
        "pre_correctness_recomputed",
        "transition",
        "grounding_changed",
        "grounding_status",
        "grounding_reason_code",
        "decision_reason_code",
        "citation_status",
        "citation_issue_codes",
        "recall_at_5",
        "first_relevant_rank",
        "context_omitted_count",
        "context_truncated_count",
        "uncovered_sub_query_count",
        "query",
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
                    "stored_final_correct": row["stored_final_correct"],
                    "stored_final_correctness": row["stored_final_correctness"],
                    "pre_correct_recomputed": row["pre_correct_recomputed"],
                    "pre_correctness_recomputed": row["pre_correctness_recomputed"],
                    "transition": row["transition"],
                    "grounding_changed": row["grounding"]["changed"],
                    "grounding_status": row["grounding"]["status"],
                    "grounding_reason_code": row["grounding"]["reason_code"],
                    "decision_reason_code": row["grounding"]["decision_reason_code"],
                    "citation_status": row["citation"]["status"],
                    "citation_issue_codes": "|".join(row["citation"]["issue_codes"]),
                    "recall_at_5": row["retrieval"]["recall_at_5"],
                    "first_relevant_rank": row["retrieval"]["first_relevant_rank"],
                    "context_omitted_count": row["context"]["omitted_count"],
                    "context_truncated_count": row["context"]["truncated_count"],
                    "uncovered_sub_query_count": len(row["context"]["uncovered_sub_query_ids"]),
                    "query": row["query"],
                    "pre_grounding_answer": row["pre_grounding_answer"],
                    "final_visible_answer": row["final_visible_answer"],
                }
            )


def report(summary: Mapping[str, Any]) -> str:
    easy = summary["difficulties"]["easy"]
    medium = summary["difficulties"]["medium"]
    return f"""# live-cand-max-mono-01 easy/medium partial 诊断

> 状态：**partial**。本快照只覆盖已经 archived 的 easy 与 medium；hard 明确不在范围内。

## 结论

完全保留 pre-grounding 答案仍不能达到目标：easy 为 **45/50（90%）**，低于 48/50 三题；medium 为 **42/50（84%）**，低于 43/50 一题。因此 pre-preservation 是必要修复，但不是充分修复。

更强的逐题 oracle（每题只要 pre 或 final 任一正确就计正确）也只有 easy **46/50（92%）**、medium **42/50（84%）**，仍分别差 2 题和 1 题。这证明仅设计一个“选择 pre 还是 final”的门控也无法达到目标；剩余 both-wrong 题必须通过通用生成/评测流程修复。

## 正式 final、pre 重算与转移矩阵

| 难度 | Stored final 正确 | 同 evaluator pre 正确 | pre正确→final错误 | pre错误→final正确 | pre/final oracle | 目标 |
|---|---:|---:|---:|---:|---:|---:|
| easy | 40/50（80%） | 45/50（90%） | 6 | 1 | 46/50（92%） | 48/50 |
| medium | 35/50（70%） | 42/50（84%） | 7 | 0 | 42/50（84%） | 43/50 |

easy 完全保留 pre 相对 stored final 净增加 5 题；medium 净增加 7 题。pre-preservation 对两集均有明确价值，但不能单独通过目标门槛。

## 错题 grounding 分支

easy 的 10 个 stored 错题全部发生 grounding change：`POST_GENERATION_UNSUPPORTED_CLAIMS_PRUNED` 9、`POST_GENERATION_EXTRACTIVE_REPAIR` 1；grounding status 为 degraded 9、passed 1。

medium 的 15 个 stored 错题也全部发生 grounding change：`POST_GENERATION_UNSUPPORTED_CLAIMS_PRUNED` 12、`POST_GENERATION_COMPOSITE_CONSTRAINT_REPAIR` 1、`UNSUPPORTED_GENERATION` 1、`POST_GENERATION_EXTRACTIVE_REPAIR` 1；grounding status 为 degraded 13、failed 1、passed 1。

高频缺陷已经集中为 claim pruning：easy 错题 90%、medium 错题 80% 进入 unsupported-claim pruning。需要校验被删 claim 是否已有其他 trusted citation、是否只是 marker 形式不规范，以及删除后是否破坏列表/多字段完整性。

## 错题召回与上下文

| 难度 | 错题数 | Recall@5=1 | 首相关 rank=1 | Omitted>0 | Truncated>0 | Uncovered sub-query>0 |
|---|---:|---:|---:|---:|---:|---:|
| easy | 10 | 10 | 9 | 0 | 0 | 0 |
| medium | 15 | 15 | 14 | 0 | 0 | 2 |

easy 的错题不能归因于 top-5 召回或 context 丢失。medium 也全部 recall@5=1 且没有 omitted/truncated；只有 2 题存在 uncovered sub-query，应单独作为上下文覆盖缺口，而不是解释其余 13 题。

## 方法与口径

- Final 正式口径：归档 `case_outcome.final_answer_correct`。
- Pre 口径：当前项目 `_generation_metrics` 对归档 `pre_grounding_answer` 重算，阈值与 case policy 相同，correctness ≥0.75。
- 粒度：easy/medium 各 50 个唯一 case_id；只读取 archived `results.jsonl` 与固定数据集。
- 分析不调用网络或 LLM，不读取凭据，不修改 `app/`，也不修改既有 candidate-v3 产物。
- 这是 partial 快照；不能据此报告 hard 或整个三难度运行已经完成。

## 下一步判断

1. 先用通用 completeness/claim-retention 门禁消除 13 个 pre正确→final错误中的可避免损失。
2. easy 仍需至少修复 3 个 pre-wrong；即使使用 pre/final oracle 也需修复 2 个 both-wrong。
3. medium 仍需至少修复 1 个 pre-wrong/both-wrong；先检查 2 个 uncovered sub-query，再分析其余 evidence 已足但生成失败的题。
4. 任何新候选仍以 stored final binary accuracy 为发布门禁，不能用 pre 理论值替代正式结果。
"""


def main() -> int:
    run_index = read_json(RUN_INDEX_PATH)
    if run_index.get("label") != RUN_LABEL:
        raise ValueError("run label mismatch")
    all_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        entry = run_index["difficulties"][difficulty]
        if entry.get("status") != "archived" or entry.get("completed_cases") != 50:
            raise ValueError(f"difficulty is not archived: {difficulty}")
        dataset_path = (REPO_ROOT / entry["dataset_source_path"]).resolve()
        results_path = (REMEDIATION_ROOT / entry["run_directory"] / "results.jsonl").resolve()
        dataset_sha = sha_file(dataset_path)
        results_sha = sha_file(results_path)
        if dataset_sha != entry["dataset_source_sha256"]:
            raise ValueError(f"dataset sha mismatch: {difficulty}")
        dataset = read_json(dataset_path)
        cases = {str(case["case_id"]): case for case in dataset.get("cases") or []}
        rows: list[dict[str, Any]] = []
        with results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = str(record.get("case_id"))
                rows.append(
                    extract_row(
                        difficulty=difficulty,
                        case=cases[case_id],
                        record=record,
                        line_number=line_number,
                        dataset_sha256=dataset_sha,
                        results_sha256=results_sha,
                    )
                )
        if len(rows) != 50 or len({row["case_id"] for row in rows}) != 50:
            raise ValueError(f"grain mismatch: {difficulty}")
        stored_summary_accuracy = entry["metrics_summary"]["metrics"]["case_outcome"][
            "final_answer_accuracy"
        ]["mean"]
        recomposed = mean(float(row["stored_final_correct"]) for row in rows)
        if not math.isclose(float(stored_summary_accuracy), float(recomposed), abs_tol=1e-12):
            raise ValueError(f"stored accuracy mismatch: {difficulty}")
        sources[difficulty] = {
            "run_id": entry["run_id"],
            "dataset_path": dataset_path.relative_to(REPO_ROOT).as_posix(),
            "dataset_sha256": dataset_sha,
            "results_path": results_path.relative_to(REPO_ROOT).as_posix(),
            "results_sha256": results_sha,
            "archived_case_count": len(rows),
        }
        all_rows.extend(rows)
    if len(all_rows) != 100 or len({row["case_id"] for row in all_rows}) != 100:
        raise ValueError("partial snapshot grain mismatch")
    summary = {
        "schema_version": "stt.eval.max-mono-partial-summary.v1",
        "status": "partial",
        "run_label": RUN_LABEL,
        "included_difficulties": list(DIFFICULTIES),
        "excluded_difficulties": ["hard"],
        "partial_reason": "hard was not awaited; only archived easy/medium are included",
        "authoritative_final_metric": "archived case_outcome.final_answer_correct",
        "pre_metric": "current _generation_metrics correctness >= 0.75",
        "metrics_module_sha256": sha_file(REPO_ROOT / "app" / "evaluation" / "metrics.py"),
        "sources": sources,
        "data_quality": {
            "row_count": len(all_rows),
            "unique_case_id_count": len({row["case_id"] for row in all_rows}),
            "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
            "difficulty_counts": dict(Counter(row["difficulty"] for row in all_rows)),
            "critical_dataset_hashes_verified": True,
            "stored_summary_accuracy_reconciled": True,
        },
        "difficulties": {
            difficulty: summarize(
                [row for row in all_rows if row["difficulty"] == difficulty], difficulty
            )
            for difficulty in DIFFICULTIES
        },
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(
        OUTPUT_DIR / "cases.json",
        {
            "schema_version": "stt.eval.max-mono-partial-cases.v1",
            "status": "partial",
            "summary": summary,
            "cases": all_rows,
        },
    )
    write_csv(all_rows)
    (OUTPUT_DIR / "REPORT.md").write_text(report(summary), encoding="utf-8", newline="\n")
    deliverables = ["REPORT.md", "analyze_partial.py", "cases.csv", "cases.json", "summary.json"]
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(f"{sha_file(OUTPUT_DIR / name)}  {name}" for name in sorted(deliverables))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["difficulties"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
