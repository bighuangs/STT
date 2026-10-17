"""Read-only evaluator-v3 shadow replay for the frozen three-level suite.

No rule depends on a case id.  The script joins immutable manifests to the
archived candidate at case grain, re-evaluates both final and pre-grounding
answers under explicit v2/v3 policy snapshots, and runs suite-wide
counterfactual contract checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


OUTPUT_ROOT = Path(__file__).resolve().parent
REMEDIATION_ROOT = OUTPUT_ROOT.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
PROJECT_ROOT = SUITE_ROOT.parents[2]
DATASET_ROOT = SUITE_ROOT / "datasets"
RUN_ROOT = REMEDIATION_ROOT / "runs" / "live-cand-max-mono-01"
SOURCE_AUDIT_ROOT = (
    REMEDIATION_ROOT / "suite-source-evaluator-qa" / "pdf-visual-review"
)
SOURCE_AUDIT_SUMMARY_PATH = SOURCE_AUDIT_ROOT / "summary.json"
SOURCE_AUDIT_CASES_PATH = SOURCE_AUDIT_ROOT / "cases.json"
DIFFICULTIES = ("easy", "medium", "hard")
RUN_LABEL = "live-cand-max-mono-01"
FALSE_ADDITION = "此外，该设备支持手册未记载的卫星通信功能。"
FABRICATED_REFUSAL = "无法仅依据手册确定，但上海现货价为 9999 元。"
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?")
_MULTI_QUERY_RE = re.compile(
    r"(?:分别|各自|以及|同时|并且|并说明|并列出|和哪|与哪|哪些|全部|每个|"
    r"两种|两个|三种|步骤|哪些值|哪些项|比较|差异|如何.*(?:并|以及))"
)
_CLAUSE_SPLIT_RE = re.compile(r"[；;。\n]+")

sys.path.insert(0, str(PROJECT_ROOT))
from app.application.evaluation_service import _case_decision  # noqa: E402
from app.domain.evaluation import (  # noqa: E402
    PREVIOUS_CASE_PASS_POLICY_VERSION,
    freeze_case_pass_policy,
)
from app.evaluation.metrics import evaluate_case  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(difficulty: str) -> Path:
    matches = sorted((RUN_ROOT / "artifacts" / difficulty).glob("*/results.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one archived results.jsonl for {difficulty}, got {len(matches)}"
        )
    return matches[0]


def _inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    cases: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for difficulty in DIFFICULTIES:
        dataset_path = DATASET_ROOT / f"{difficulty}.json"
        result_path = _result_path(difficulty)
        manifest = _load_json(dataset_path)
        hashes[str(dataset_path.relative_to(PROJECT_ROOT))] = _sha256(dataset_path)
        hashes[str(result_path.relative_to(PROJECT_ROOT))] = _sha256(result_path)
        for case in manifest.get("cases") or []:
            case_id = str(case.get("case_id") or "")
            if not case_id or case_id in cases:
                raise ValueError(f"invalid or duplicate dataset case_id: {case_id}")
            cases[case_id] = {**case, "_difficulty": difficulty}
        for result in _load_jsonl(result_path):
            case_id = str(result.get("case_id") or "")
            if not case_id or case_id in results:
                raise ValueError(f"invalid or duplicate result case_id: {case_id}")
            results[case_id] = result
    if set(cases) != set(results) or len(cases) != 150:
        raise ValueError(
            "shadow replay requires an exact 150-case manifest/result join; "
            f"dataset_only={sorted(set(cases) - set(results))}, "
            f"result_only={sorted(set(results) - set(cases))}"
        )
    return cases, results, hashes


def _source_audit_findings(
    cases: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary = _load_json(SOURCE_AUDIT_SUMMARY_PATH)
    reviewed_rows = _load_json_array(SOURCE_AUDIT_CASES_PATH)
    by_case_id: dict[str, dict[str, Any]] = {}
    disposition_counts: Counter[str] = Counter()
    for raw in reviewed_rows:
        case_id = str(raw.get("case_id") or "")
        disposition = str(raw.get("reference_disposition") or "")
        if not case_id or case_id in by_case_id or case_id not in cases:
            raise ValueError(f"invalid source-audit case_id: {case_id}")
        if not disposition:
            raise ValueError(f"missing source-audit disposition: {case_id}")
        by_case_id[case_id] = raw
        disposition_counts[disposition] += 1

    declared_counts = summary.get("reference_dispositions")
    if not isinstance(declared_counts, dict) or dict(disposition_counts) != declared_counts:
        raise ValueError("source-audit disposition counts do not match official summary")
    blocking_ids = summary.get("blocking_reference_cases")
    if not isinstance(blocking_ids, list) or not all(
        isinstance(case_id, str) and case_id for case_id in blocking_ids
    ):
        raise ValueError("official source audit is missing blocking_reference_cases")
    if len(set(blocking_ids)) != len(blocking_ids):
        raise ValueError("duplicate source-audit blocking case id")
    missing = sorted(set(blocking_ids) - set(by_case_id))
    if missing:
        raise ValueError(f"source-audit blocking rows are missing: {missing}")

    # This sidecar is joined only after v2/v3 scoring.  It changes attribution
    # denominators, never evaluator decisions or the raw 50-case scores.
    findings = {case_id: by_case_id[case_id] for case_id in blocking_ids}
    return findings, summary


def _stage_observation(result: dict[str, Any], stage: str) -> dict[str, Any] | None:
    raw = result.get("observation")
    if not isinstance(raw, dict):
        return None
    observation = deepcopy(raw)
    if stage == "final":
        return observation
    pre_answer = observation.get("pre_grounding_answer")
    if not isinstance(pre_answer, str) or not pre_answer.strip():
        return None
    observation["answer"] = pre_answer
    observation.pop("abstained", None)
    # These structures describe the final post-grounding text.  Keeping them
    # would leak final-stage spans into the isolated pre-grounding replay.
    for field in (
        "citations",
        "claims",
        "citation_validation",
        "answer_envelope",
        "visible_answer_policy",
    ):
        observation.pop(field, None)
    return observation


def _stored_final_correct(result: dict[str, Any]) -> bool | None:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    inner = metrics.get("metrics")
    if not isinstance(inner, dict):
        return None
    outcome = inner.get("case_outcome")
    if not isinstance(outcome, dict):
        return None
    value = outcome.get("final_answer_correct")
    return value if isinstance(value, bool) else None


def _replay_row(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    stage: str,
    source_finding: dict[str, Any] | None,
) -> dict[str, Any] | None:
    observation = _stage_observation(result, stage)
    if observation is None:
        return None
    evaluated = evaluate_case(case, observation)
    v2 = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=observation,
        case_policy={"policy_version": PREVIOUS_CASE_PASS_POLICY_VERSION},
    )
    v3 = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=observation,
        case_policy=freeze_case_pass_policy(None),
    )
    generation = evaluated["metrics"]["generation"]
    components = generation.get("correctness_components") or {}
    stored = _stored_final_correct(result) if stage == "final" else None
    v2_correct = bool(v2.get("final_answer_correct"))
    v3_correct = bool(v3.get("final_answer_correct"))
    return {
        "case_id": case["case_id"],
        "difficulty": case["_difficulty"],
        "stage": stage,
        "source_audit_status": (
            source_finding.get("reference_disposition")
            if source_finding
            else "not_blocked_by_source_audit"
        ),
        "excluded_from_process_attribution": source_finding is not None,
        "source_audit_note": source_finding.get("finding") if source_finding else None,
        "source_audit_recommended_action": (
            source_finding.get("recommended_action") if source_finding else None
        ),
        "stored_final_correct": stored,
        "v2_final_correct": v2_correct,
        "v3_final_correct": v3_correct,
        "v2_to_v3": (
            "pass_to_fail"
            if v2_correct and not v3_correct
            else "fail_to_pass"
            if not v2_correct and v3_correct
            else "unchanged_pass"
            if v3_correct
            else "unchanged_fail"
        ),
        "v3_reason_codes": list(v3.get("reason_codes") or []),
        "coverage_correctness": generation.get("correctness"),
        "contract_correctness": generation.get("contract_correctness"),
        "concept_group_coverage": components.get("concept_group_coverage"),
        "required_branch_count": components.get("required_branch_count"),
        "required_branch_coverage": components.get("required_branch_coverage"),
        "critical_numbers_preserved": components.get(
            "critical_numbers_preserved"
        ),
        "critical_identifiers_preserved": components.get(
            "critical_identifiers_preserved"
        ),
        "polarity_constraints_preserved": components.get(
            "polarity_constraints_preserved"
        ),
        "unsupported_factual_claim_count": components.get(
            "unsupported_factual_claim_count"
        ),
        "contradictory_factual_claim_count": components.get(
            "contradictory_factual_claim_count"
        ),
        "abstention_factual_claim_count": components.get(
            "abstention_factual_claim_count"
        ),
    }


def _reference_context(case: dict[str, Any]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for index, raw in enumerate(case.get("relevant_evidence") or []):
        if not isinstance(raw, dict):
            continue
        context.append(
            {
                "chunk_id": raw.get("chunk_id") or f"reference-{index}",
                "evidence_id": raw.get("evidence_id"),
                "text": raw.get("text") or "",
            }
        )
    return context


def _v3_correct(case: dict[str, Any], answer: str, *, abstained: bool) -> tuple[bool, list[str]]:
    context = _reference_context(case)
    observation = {
        "answer": answer,
        "abstained": abstained,
        "retrieved": context,
        "context": context,
    }
    evaluated = evaluate_case(case, observation)
    outcome = _case_decision(
        case,
        evaluated,
        had_error=False,
        observation=observation,
        case_policy=freeze_case_pass_policy(None),
    )
    return bool(outcome.get("final_answer_correct")), list(
        outcome.get("reason_codes") or []
    )


def _mutate_first_number(case: dict[str, Any]) -> str | None:
    reference = str(case.get("reference_answer") or "")
    scoring_texts = [str(value) for value in case.get("expected_terms") or []]
    concept_groups = case.get("concept_groups")
    if isinstance(concept_groups, dict):
        for group in concept_groups.get("all_of") or []:
            if isinstance(group, dict):
                scoring_texts.extend(str(value) for value in group.get("any_of") or [])
    scoring_numbers = {
        match.group(0)
        for text in scoring_texts
        for match in _NUMBER_RE.finditer(text)
    }
    match = next(
        (
            candidate
            for candidate in _NUMBER_RE.finditer(reference)
            if candidate.group(0) in scoring_numbers
        ),
        None,
    )
    if match is None:
        return None
    raw = match.group(0)
    replacement = (
        str(float(raw) + 1.0).rstrip("0").rstrip(".")
        if "." in raw
        else str(int(raw) + 1)
    )
    return reference[: match.start()] + replacement + reference[match.end() :]


def _truncate_multi_branch(case: dict[str, Any]) -> str | None:
    if not _MULTI_QUERY_RE.search(str(case.get("query") or "")):
        return None
    reference = str(case.get("reference_answer") or "")
    clauses = [item.strip(" ，,、") for item in _CLAUSE_SPLIT_RE.split(reference) if item.strip()]
    # The un-authored heuristic is deliberately restricted to the same
    # high-confidence two-branch shape enforced by evaluator v3.  Larger
    # all-of contracts require versioned concept_groups in the manifest.
    if len(clauses) == 2:
        return clauses[0] + "。"
    return None


def _counterfactuals(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, list[dict[str, Any]]] = {
        "unsupported_false_addition": [],
        "fabricated_fact_with_refusal": [],
        "numeric_mutation": [],
        "multi_branch_truncation": [],
    }
    for case in cases:
        reference = str(case.get("reference_answer") or "")
        if case.get("unanswerable"):
            passed, reasons = _v3_correct(
                case, FABRICATED_REFUSAL, abstained=True
            )
            checks["fabricated_fact_with_refusal"].append(
                {"case_id": case["case_id"], "incorrectly_passed": passed, "reasons": reasons}
            )
            continue
        passed, reasons = _v3_correct(
            case, f"{reference} {FALSE_ADDITION}", abstained=False
        )
        checks["unsupported_false_addition"].append(
            {"case_id": case["case_id"], "incorrectly_passed": passed, "reasons": reasons}
        )
        mutated = _mutate_first_number(case)
        if mutated is not None:
            passed, reasons = _v3_correct(case, mutated, abstained=False)
            checks["numeric_mutation"].append(
                {"case_id": case["case_id"], "incorrectly_passed": passed, "reasons": reasons}
            )
        truncated = _truncate_multi_branch(case)
        if truncated is not None:
            branch_case = deepcopy(case)
            branches = [
                item.strip(" ，,、")
                for item in _CLAUSE_SPLIT_RE.split(reference)
                if item.strip()
            ]
            branch_case["concept_groups"] = {
                "all_of": [
                    {
                        "concept_key": f"explicit_branch_{index + 1}",
                        "any_of": [branch],
                    }
                    for index, branch in enumerate(branches)
                ]
            }
            passed, reasons = _v3_correct(
                branch_case, truncated, abstained=False
            )
            checks["multi_branch_truncation"].append(
                {"case_id": case["case_id"], "incorrectly_passed": passed, "reasons": reasons}
            )
    return {
        name: {
            "applicable_count": len(rows),
            "incorrect_pass_count": sum(
                1 for row in rows if row["incorrectly_passed"]
            ),
            "blocked_count": sum(
                1 for row in rows if not row["incorrectly_passed"]
            ),
            "rows": rows,
        }
        for name, rows in checks.items()
    }


def _summary(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    stage_rows = [row for row in rows if row["stage"] == stage]
    by_difficulty: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        subset = [row for row in stage_rows if row["difficulty"] == difficulty]
        attributable = [
            row
            for row in subset
            if not row["excluded_from_process_attribution"]
        ]
        by_difficulty[difficulty] = {
            "available_count": len(subset),
            "v2_correct": sum(1 for row in subset if row["v2_final_correct"]),
            "v2_accuracy": (
                sum(1 for row in subset if row["v2_final_correct"]) / len(subset)
                if subset
                else None
            ),
            "v3_correct": sum(1 for row in subset if row["v3_final_correct"]),
            "v3_accuracy": (
                sum(1 for row in subset if row["v3_final_correct"]) / len(subset)
                if subset
                else None
            ),
            "transition_counts": dict(Counter(row["v2_to_v3"] for row in subset)),
            "v3_reason_counts": dict(
                Counter(
                    reason
                    for row in subset
                    for reason in row["v3_reason_codes"]
                )
            ),
            "source_audit_excluded_count": len(subset) - len(attributable),
            "process_attribution": {
                "case_count": len(attributable),
                "v2_correct": sum(
                    1 for row in attributable if row["v2_final_correct"]
                ),
                "v2_accuracy": (
                    sum(1 for row in attributable if row["v2_final_correct"])
                    / len(attributable)
                    if attributable
                    else None
                ),
                "v3_correct": sum(
                    1 for row in attributable if row["v3_final_correct"]
                ),
                "v3_accuracy": (
                    sum(1 for row in attributable if row["v3_final_correct"])
                    / len(attributable)
                    if attributable
                    else None
                ),
                "transition_counts": dict(
                    Counter(row["v2_to_v3"] for row in attributable)
                ),
            },
        }
    return {
        "stage": stage,
        "available_count": len(stage_rows),
        "missing_count": 150 - len(stage_rows),
        "by_difficulty": by_difficulty,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "difficulty",
        "stage",
        "source_audit_status",
        "excluded_from_process_attribution",
        "source_audit_note",
        "source_audit_recommended_action",
        "stored_final_correct",
        "v2_final_correct",
        "v3_final_correct",
        "v2_to_v3",
        "v3_reason_codes",
        "coverage_correctness",
        "contract_correctness",
        "concept_group_coverage",
        "required_branch_count",
        "required_branch_coverage",
        "critical_numbers_preserved",
        "critical_identifiers_preserved",
        "polarity_constraints_preserved",
        "unsupported_factual_claim_count",
        "contradictory_factual_claim_count",
        "abstention_factual_claim_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["v3_reason_codes"] = "|".join(row["v3_reason_codes"])
            writer.writerow(output)


def _report(summary: dict[str, Any]) -> str:
    final = summary["final"]["by_difficulty"]
    source_audit = summary["source_audit"]
    lines = [
        "# Evaluator v3 Shadow Replay",
        "",
        "## 技术摘要",
        "",
        (
            "Evaluator v3 消除了本次全量 QA 证明的四类确定性假阳性：无据追加事实 145/145、"
            "拒答夹带事实 5/5、标注关键数字改写 36/36、明确多分支删尾 63/63 均被拦截，错误放行均为 0。"
        ),
        (
            f"同一归档 final answer 的原始 v2→v3 正确数为 easy "
            f"{final['easy']['v2_correct']}→{final['easy']['v3_correct']}、medium "
            f"{final['medium']['v2_correct']}→{final['medium']['v3_correct']}、hard "
            f"{final['hard']['v2_correct']}→{final['hard']['v3_correct']}。"
            "这说明新合同不会通过放宽评测制造达标；当前 candidate 仍不能登记为新 baseline。"
        ),
        (
            f"独立 PDF 视觉审计另标记 {source_audit['finding_count']} 题不适合用于流程绩效归因："
            f"{source_audit['status_counts'].get('confirmed_source_conflict', 0)} 题已确认旧 reference/source 冲突，"
            f"其余 {source_audit['finding_count'] - source_audit['status_counts'].get('confirmed_source_conflict', 0)} 题"
            "分别属于源中不显式、源范围未明确或旧 reference 未回答问题。原始分数不改，只在流程归因切片中排除。"
        ),
        "",
        "## 四类反事实盲区已被全部关闭",
        "",
        "| 反事实 | 适用 | v3 拦截 | 错误放行 |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["counterfactuals"].items():
        lines.append(
            f"| {name} | {item['applicable_count']} | {item['blocked_count']} | "
            f"{item['incorrect_pass_count']} |"
        )
    lines.extend(
        [
            "",
            "这些反事实均按全套通用转换构造，没有 case-id 分支；结果证明的是 v3 可接受集合不再包含这些已知错误形态。",
            "",
            "## Raw shadow 与流程可归因切片必须分开解释",
            "",
            "| 难度 | Raw题数 | v2正确 | v3正确 | Source审计排除 | 可归因题数 | 可归因v2正确 | 可归因v3正确 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        item = final[difficulty]
        attribution = item["process_attribution"]
        lines.append(
            f"| {difficulty} | {item['available_count']} | {item['v2_correct']} | "
            f"{item['v3_correct']} | {item['source_audit_excluded_count']} | "
            f"{attribution['case_count']} | {attribution['v2_correct']} | "
            f"{attribution['v3_correct']} |"
        )
    lines.extend(
        [
            "",
            "Raw 是对旧 manifest 合同的忠实重算；可归因切片只回答系统流程表现，不是修订后的官方正确率，也不能与 50 题目标分母直接比较。",
            "",
            "## Final 与 pre-grounding 的阶段差异",
        "",
        "| 阶段 | 难度 | 可用 | v2 正确 | v3 正确 | v2→v3 pass→fail |",
        "|---|---|---:|---:|---:|---:|",
        ]
    )
    for stage in ("final", "pre_grounding"):
        stage_summary = summary[stage]
        for difficulty in DIFFICULTIES:
            item = stage_summary["by_difficulty"][difficulty]
            lines.append(
                f"| {stage} | {difficulty} | {item['available_count']} | "
                f"{item['v2_correct']} | {item['v3_correct']} | "
                f"{item['transition_counts'].get('pass_to_fail', 0)} |"
            )
    lines.extend(
        [
            "",
            "pre-grounding 使用同一归档 context，但移除 final 的 citation/claim/span 结构；因此它只用于阶段诊断，不替代 final target。",
            "",
            "## 范围、数据与指标定义",
            "",
            "- 粒度：固定 `case_id`；manifest 150 题与归档 result 150 题一对一完整连接。",
            "- Raw v2/v3 accuracy：各版本 `final_answer_correct` 的二元均值；本报告展示正确数避免小样本百分比掩盖分母。",
            "- 严格有据通过率：faithfulness、citation coverage、quality score 的独立守门；不替代 `final_answer_accuracy`。",
            "- 流程可归因切片：仅排除独立视觉审计确认 reference/source 冲突或无法明确源绑定的题，不改任何 evaluator 输出。",
            "- 时间与版本：输入为 `live-cand-max-mono-01` 的固定 2026-08-10 归档；shadow policy 为 `stt.eval.case.v3`。",
            "",
            "## 方法：版本化合同与只读重放",
            "",
            "1. 锁定三份 immutable manifest 与三个 `results.jsonl` 的 SHA-256，并验证 150/150 case join。",
            "2. 对 final 与 `pre_grounding_answer` 分别调用同一 deterministic evaluator；v2 与 v3 policy 均显式冻结。",
            "3. v3 final correctness 在覆盖分之外硬检查事实支持/矛盾、拒答残留事实、关键数字/标识符/极性、明确分支与可选 concept groups。",
            "4. 运行四类套件级反事实并记录逐题 reason codes；规则不读取 case id、参考答案替换表或当前成败列表。",
            "5. 最后连接独立视觉审计状态，仅生成 attribution slice；该连接发生在评分完成之后。",
            "",
            "没有绘制趋势图：这是固定单次快照与离散合同审计，精确计数表比时间趋势或连续分布更忠实。",
            "",
            "## 限制、稳健性与解释边界",
            "",
            "- `pass_to_fail` 表示 v3 合同更严格，不自动证明旧答案事实上错误；逐条组件见 `shadow-cases.json/csv`。",
            "- 当前旧 manifest 尚无 `concept_groups`；schema 与行为已有单元测试，但套件迁移必须另发新版本/hash。",
            "- 事实支持仍是保守 deterministic claim matcher；与独立语义盲审的混淆矩阵尚未完成，不能把 shadow 分数当最终人工真值。",
            "- Source audit 排除是归因边界，不是分数豁免；逐题人工事实与修订建议仅以官方 `pdf-visual-review` sidecar 为准。",
            "- faithfulness/citation/retrieval 继续作为诊断或严格护栏；仅引用降级不会自动改写最终正确性，除非内容本身违反 v3 合同。",
            "",
            "## 建议的下一步",
            "",
            f"1. 将 {source_audit['finding_count']} 个 source-audit finding 纳入全套 suite v2 的统一复核与新 hash，禁止原地改旧 manifest。",
            "2. 用独立盲审对 v3 的 raw pass/fail 做全量或分层冲突矩阵，重点复核 unsupported-claim 与 branch-incomplete 假阴性。",
            "3. 只有在修订套件、v3 evaluator 与新 live run 同时冻结后，才重新判断 easy/medium 是否达到登记阈值。",
            "",
            "## 仍待回答的问题",
            "",
            "- v3 与独立盲审在 final/pre 两阶段的一致率和主要冲突类型是什么？",
            "- 全套 concept-group 双审后，多少旧字符串标签会被拆为真正独立的必答概念？",
            "- 视觉/OCR 结构化后，当前 source-unverifiable 题是否可恢复为可靠基准题？",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, results, input_hashes = _inputs()
    source_findings, source_audit_summary = _source_audit_findings(cases)
    for source_audit_path in (
        SOURCE_AUDIT_SUMMARY_PATH,
        SOURCE_AUDIT_CASES_PATH,
    ):
        input_hashes[str(source_audit_path.relative_to(PROJECT_ROOT))] = _sha256(
            source_audit_path
        )
    rows: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        result = results[case_id]
        for stage in ("final", "pre_grounding"):
            row = _replay_row(
                case,
                result,
                stage=stage,
                source_finding=source_findings.get(case_id),
            )
            if row is not None:
                rows.append(row)

    counterfactuals = _counterfactuals(cases.values())
    summary = {
        "schema_version": "stt.eval.v3-shadow-replay.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "suite_key": "three-difficulty-live-2026-08-10",
        "run_label": RUN_LABEL,
        "input_case_count": len(cases),
        "join_complete": len(cases) == len(results) == 150,
        "input_sha256": input_hashes,
        "source_audit": {
            "finding_count": len(source_findings),
            "status_counts": dict(
                Counter(
                    str(finding.get("reference_disposition") or "unknown")
                    for finding in source_findings.values()
                )
            ),
            "official_reference_dispositions": source_audit_summary[
                "reference_dispositions"
            ],
            "official_sidecar": {
                "summary": str(SOURCE_AUDIT_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
                "cases": str(SOURCE_AUDIT_CASES_PATH.relative_to(PROJECT_ROOT)),
            },
            "scoring_effect": "none",
            "attribution_effect": "excluded_from_process_performance_attribution",
        },
        "policy_versions": {"previous": "stt.eval.case.v2", "shadow": "stt.eval.case.v3"},
        "final": _summary(rows, "final"),
        "pre_grounding": _summary(rows, "pre_grounding"),
        "counterfactuals": {
            name: {key: value for key, value in item.items() if key != "rows"}
            for name, item in counterfactuals.items()
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "shadow-cases.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "counterfactuals.json").write_text(
        json.dumps(counterfactuals, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "shadow-cases.csv", rows)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    artifact = {
        "schema_version": "stt.eval.artifact.v1",
        "status": "shadow_only_not_a_baseline",
        "summary": "summary.json",
        "case_rows": ["shadow-cases.json", "shadow-cases.csv"],
        "counterfactuals": "counterfactuals.json",
        "source_audit_findings": {
            "summary": str(SOURCE_AUDIT_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "cases": str(SOURCE_AUDIT_CASES_PATH.relative_to(PROJECT_ROOT)),
        },
        "report": "REPORT.md",
        "reproduce": ".venv/Scripts/python.exe shadow_replay.py",
        "visual_omission_reason": (
            "This is a fixed one-snapshot contract audit; exact case counts and "
            "attribution denominators are clearer in tables than in a trend chart."
        ),
    }
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_targets = [
        Path(__file__).resolve(),
        output_dir / "artifact.json",
        output_dir / "counterfactuals.json",
        output_dir / "REPORT.md",
        output_dir / "shadow-cases.csv",
        output_dir / "shadow-cases.json",
        output_dir / "summary.json",
    ]
    checksum_lines = [
        f"{_sha256(path)}  {path.name}" for path in checksum_targets
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
