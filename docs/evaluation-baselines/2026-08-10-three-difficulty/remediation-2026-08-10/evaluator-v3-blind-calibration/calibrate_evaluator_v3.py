from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
REM_DIR = OUTPUT_DIR.parent
BASELINE_DIR = REM_DIR.parent

SHADOW_CASES_PATH = REM_DIR / "evaluator-v3-shadow-replay" / "shadow-cases.json"
SHADOW_SUMMARY_PATH = REM_DIR / "evaluator-v3-shadow-replay" / "summary.json"
SHADOW_CHECKSUMS_PATH = REM_DIR / "evaluator-v3-shadow-replay" / "SHA256SUMS"
PRE_REVIEW_DIR = (
    REM_DIR
    / "semantic-reviews"
    / "live-cand-max-mono-01"
    / "direct-answer-v3-qwen3.7-max"
    / "pre_grounding"
)
PRE_RECORDS_PATH = PRE_REVIEW_DIR / "semantic-correctness-review.jsonl"
PRE_SUMMARY_PATH = PRE_REVIEW_DIR / "summary.json"
PRIOR_FINAL_REVIEW_DIR = (
    REM_DIR
    / "semantic-reviews"
    / "live-candidate-v2-qwen-plus-grounding-v3-01"
    / "direct-answer-v3-qwen3.7-max"
)
PRIOR_FINAL_RECORDS_PATH = (
    PRIOR_FINAL_REVIEW_DIR / "semantic-correctness-review.jsonl"
)
PRIOR_FINAL_SUMMARY_PATH = PRIOR_FINAL_REVIEW_DIR / "summary.json"
PDF_REVIEW_DIR = REM_DIR / "suite-source-evaluator-qa" / "pdf-visual-review"
PDF_SUMMARY_PATH = PDF_REVIEW_DIR / "summary.json"
PDF_CASES_PATH = PDF_REVIEW_DIR / "cases.json"

TARGET_REASON_CODES = (
    "unsupported_factual_claim",
    "required_branch_incomplete",
    "polarity_mismatch",
    "critical_number_mismatch",
    "critical_identifier_mismatch",
    "contradictory_factual_claim",
    "abstention_incorrect",
    "abstention_contains_factual_claim",
)
SPLIT_VERSION = "stt.eval.v3-blind-content-sha-split.v1"

_ASCII_TOKEN_RE = re.compile(r"[a-z]+(?:[._/-]?[a-z0-9]+)*|\d+(?:\.\d+)?", re.I)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_CITATION_RE = re.compile(r"\[(?:E|e)\d+(?:\s*[,，、]\s*(?:E|e)?\d+)*\]")


def _find_repo_root() -> Path:
    for candidate in SCRIPT_PATH.parents:
        if (candidate / "app").is_dir() and (candidate / "docs").is_dir():
            return candidate
    raise RuntimeError("repository root not found")


REPO_ROOT = _find_repo_root()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _content_sha_partition(
    query_sha256: str, reference_sha256: str
) -> tuple[str, str, float]:
    """Create a label-free 50/50 partition from semantic input identity."""

    split_sha256 = _sha256_text(
        f"{SPLIT_VERSION}\0{query_sha256}\0{reference_sha256}"
    )
    bucket = int(split_sha256[:16], 16) / 2**64
    partition = "calibration" if bucket < 0.5 else "holdout"
    return partition, split_sha256, round(bucket, 12)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes(path))


def _load_json(path: Path) -> Any:
    return json.loads(_read_bytes(path).decode("utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _read_bytes(path).decode("utf-8").splitlines()
        if line.strip()
    ]


def _verify_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in _read_bytes(path).decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split("  ", 1)
        target = path.parent / filename
        if _sha256_file(target) != digest:
            raise ValueError(f"checksum mismatch: {target}")
        checksums[filename] = digest
    return checksums


def _path_from_repo(value: str) -> Path:
    normalized = value.replace("/", "\\") if "\\" in str(REPO_ROOT) else value
    path = (REPO_ROOT / normalized).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path escapes repository: {value}")
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _canonical_record_hash(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _latest_evaluation_records(
    records: Iterable[dict[str, Any]], *, expected_stage: str | None
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("phase") != "evaluation":
            continue
        if expected_stage is not None and record.get("answer_stage") != expected_stage:
            continue
        if record.get("record_sha256") != _canonical_record_hash(record):
            raise ValueError(f"invalid review record hash: {record.get('case_id')}")
        case_id = str(record["case_id"])
        previous = selected.get(case_id)
        if previous is None or int(record.get("review_index", 0)) > int(
            previous.get("review_index", 0)
        ):
            selected[case_id] = record
    return selected


def _units(value: str) -> set[str]:
    text = _CITATION_RE.sub("", value).casefold()
    units = {match.group(0) for match in _ASCII_TOKEN_RE.finditer(text)}
    for block in _CJK_RE.findall(text):
        if len(block) == 1:
            units.add(block)
        else:
            units.update(block[index : index + 2] for index in range(len(block) - 1))
    return units


def _coverage(source: str, target: str) -> float | None:
    source_units = _units(source)
    if not source_units:
        return None
    return round(len(source_units & _units(target)) / len(source_units), 4)


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    for key in ("content", "text", "quote"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    evidence = candidate.get("evidence")
    if isinstance(evidence, Mapping):
        for key in ("content", "text", "quote"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _best_context_excerpt(
    observation: Mapping[str, Any], reference: str, answer: str, *, limit: int = 900
) -> tuple[str, float | None, str | None]:
    context = observation.get("context")
    if not isinstance(context, list):
        return "", None, None
    candidates: list[tuple[float, int, str, str | None]] = []
    for index, raw in enumerate(context):
        if not isinstance(raw, Mapping):
            continue
        text = _candidate_text(raw)
        if not text:
            continue
        reference_coverage = _coverage(reference, text) or 0.0
        answer_coverage = _coverage(answer, text) or 0.0
        label = raw.get("context_label")
        candidates.append(
            (
                reference_coverage * 2 + answer_coverage,
                -index,
                text,
                str(label) if label is not None else None,
            )
        )
    if not candidates:
        return "", None, None
    _, _, text, label = max(candidates, key=lambda item: (item[0], item[1]))
    excerpt = text[:limit].rstrip()
    if len(text) > limit:
        excerpt += "…"
    return excerpt, _coverage(reference, text), label


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _confusion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [
        row
        for row in rows
        if row["blind_semantic_correct"] is not None
        and not row["excluded_from_metrics"]
    ]
    counts = Counter(row["confusion_bucket"] for row in labeled)
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]
    return {
        "eligible_case_count": sum(not row["excluded_from_metrics"] for row in rows),
        "labeled_case_count": len(labeled),
        "label_coverage": _rate(
            len(labeled), sum(not row["excluded_from_metrics"] for row in rows)
        ),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "accuracy": _rate(tp + tn, len(labeled)),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "false_negative_rate": _rate(fn, tp + fn),
        "false_positive_rate": _rate(fp, fp + tn),
        "blind_semantic_correct_count": tp + fn,
        "blind_semantic_incorrect_count": fp + tn,
        "v3_correct_count": tp + fp,
        "v3_incorrect_count": fn + tn,
    }


def _by_difficulty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        difficulty: _confusion_summary(
            [row for row in rows if row["difficulty"] == difficulty]
        )
        for difficulty in ("easy", "medium", "hard")
    }


def _by_partition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        partition: {
            "overall": _confusion_summary(
                [row for row in rows if row["partition"] == partition]
            ),
            "by_difficulty": _by_difficulty(
                [row for row in rows if row["partition"] == partition]
            ),
        }
        for partition in ("calibration", "holdout")
    }


def _reason_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["blind_semantic_correct"] is not None
        and not row["excluded_from_metrics"]
    ]
    semantic_correct_count = sum(bool(row["blind_semantic_correct"]) for row in eligible)
    false_negatives = [row for row in eligible if row["confusion_bucket"] == "FN"]
    true_negatives = [row for row in eligible if row["confusion_bucket"] == "TN"]
    all_reasons = sorted(
        {reason for row in eligible for reason in row["v3_reason_codes"]}
    )
    by_reason: dict[str, Any] = {}
    for reason in all_reasons:
        flagged = [row for row in eligible if reason in row["v3_reason_codes"]]
        false_negative_count = sum(
            row["confusion_bucket"] == "FN" for row in flagged
        )
        true_negative_count = sum(row["confusion_bucket"] == "TN" for row in flagged)
        high_reference_evidence_false_negative_count = sum(
            row["confusion_bucket"] == "FN"
            and (row["reference_coverage_in_visible_answer"] or 0.0) >= 0.8
            and (row["reference_coverage_in_best_context"] or 0.0) >= 0.8
            for row in flagged
        )
        by_reason[reason] = {
            "flagged_count": len(flagged),
            "false_negative_count": false_negative_count,
            "true_negative_count": true_negative_count,
            "high_reference_evidence_false_negative_count": (
                high_reference_evidence_false_negative_count
            ),
            "false_negative_share_among_flags": _rate(
                false_negative_count, len(flagged)
            ),
            "share_of_all_false_negatives": _rate(
                false_negative_count, len(false_negatives)
            ),
            "semantic_correct_recall_loss": _rate(
                false_negative_count, semantic_correct_count
            ),
            "hard_gate_calibration": (
                "unsafe_as_independent_hard_gate_on_this_sample"
                if false_negative_count > 0
                else "no_false_negative_observed_on_this_sample"
            ),
        }
    false_negative_combinations = Counter(
        "|".join(sorted(row["v3_reason_codes"])) or "<none>"
        for row in false_negatives
    )
    return {
        "false_negative_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in false_negatives
                    for reason in row["v3_reason_codes"]
                ).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "true_negative_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in true_negatives
                    for reason in row["v3_reason_codes"]
                ).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "false_negative_reason_combinations": [
            {"reason_codes": key.split("|") if key != "<none>" else [], "count": count}
            for key, count in sorted(
                false_negative_combinations.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "by_reason": by_reason,
        "target_reason_codes": {
            reason: by_reason.get(
                reason,
                {
                    "flagged_count": 0,
                    "false_negative_count": 0,
                    "true_negative_count": 0,
                    "high_reference_evidence_false_negative_count": 0,
                    "false_negative_share_among_flags": None,
                    "share_of_all_false_negatives": None,
                    "semantic_correct_recall_loss": None,
                    "hard_gate_calibration": "not_observed",
                },
            )
            for reason in TARGET_REASON_CODES
        },
    }


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_report(summary: Mapping[str, Any]) -> str:
    pre = summary["calibration"]["pre_grounding"]
    final = summary["calibration"]["final"]
    lines = [
        "# Evaluator v3 × 独立盲审校准",
        "",
        "## 结论",
        "",
        f"总体评估：**{summary['assessment']['status']}**。"
        f"排除 {summary['exclusions']['blocking_reference_case_count']} 个已由 PDF 视觉复核确认的数据/参考阻断项后，"
        f"pre-grounding 共 {pre['overall']['labeled_case_count']} 个独立盲审标签；"
        f"v3 precision={pre['overall']['precision']}, recall={pre['overall']['recall']}, "
        f"accuracy={pre['overall']['accuracy']}。",
        "",
        "**生产默认建议：否。** pre-grounding 仍存在盲审代理标签错分，且当前 run 的 final "
        "独立盲审覆盖不足；final 是主指标，因此 exact-answer 子集不能替代全量认证。",
        "",
        "独立盲审是校准代理标签而非绝对真值。该结果显示当前 v3 仍存在双向错分，"
        "但不能单独证明每一个盲审判定都正确。",
        "",
        "## Pre-grounding confusion matrix",
        "",
        "| 范围 | TP | FP | FN | TN | Precision | Recall | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, values in [("overall", pre["overall"]), *pre["by_difficulty"].items()]:
        matrix = values["confusion_matrix"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    scope,
                    matrix["true_positive"],
                    matrix["false_positive"],
                    matrix["false_negative"],
                    matrix["true_negative"],
                    values["precision"],
                    values["recall"],
                    values["accuracy"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "正类定义：独立盲审认为最终可见答案语义正确；预测正类定义：evaluator v3 认为正确。",
            "",
            "## 稳定内容 SHA 分区",
            "",
            "分区只使用 query/reference_answer 的 SHA-256，不使用 case_id、难度、盲审标签或 evaluator 输出。",
            "",
            "| 分区 | 可用标签 | TP | FP | FN | TN | Precision | Recall | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for partition in ("calibration", "holdout"):
        values = pre["by_partition"][partition]["overall"]
        matrix = values["confusion_matrix"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    partition,
                    values["labeled_case_count"],
                    matrix["true_positive"],
                    matrix["false_positive"],
                    matrix["false_negative"],
                    matrix["true_negative"],
                    values["precision"],
                    values["recall"],
                    values["accuracy"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "该分区是确定且无标签泄漏的，但 evaluator 修订前已使用全量聚合反馈，"
            "因此 holdout 不宣称为前瞻性完全未触碰样本。",
            "",
            "## 重点硬门禁的误杀校准",
            "",
            "| reason_code | 触发 | 盲审正确但被杀 | 其中参考在答案/上下文覆盖均≥0.8 | 触发中误杀占比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    target_reasons = pre["reason_analysis"]["target_reason_codes"]
    for reason in TARGET_REASON_CODES:
        values = target_reasons[reason]
        lines.append(
            f"| {reason} | {values['flagged_count']} | "
            f"{values['false_negative_count']} | "
            f"{values['high_reference_evidence_false_negative_count']} | "
            f"{values['false_negative_share_among_flags']} |"
        )
    lines.extend(
        [
            "",
            "其中“参考在答案/上下文覆盖均≥0.8”只证明核心参考内容具有强词面证据，"
            "不自动证明答案中的每一条扩展事实都正确。即便如此，这些统计与下方逐题复核共同说明："
            "reason code 可以保留为诊断信号，但在本样本上不能未经语义校准就作为独立一票否决。",
            "",
            "## Final 阶段标签覆盖限制",
            "",
            f"当前 candidate 没有 150 条独立 final 盲审。只对 {final['overall']['labeled_case_count']} 个"
            f"排除阻断后的 exact-visible-answer identity 样本复用已有语义标签，覆盖率 "
            f"{final['overall']['label_coverage']}。其余 "
            f"{final['blind_label_availability']['eligible_unlabeled_count']} 条保持 unknown，"
            "未使用另一个 run 的不同答案标签。该子集由答案未变化/恰好同答决定，存在选择偏差，"
            "不可外推为 final 全量 precision/recall。",
            "",
            "| exact-answer 范围 | 标签 | TP | FP | FN | TN | Precision | Recall | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope, values in [
        ("overall", final["overall"]),
        ("calibration", final["by_partition"]["calibration"]["overall"]),
        ("holdout", final["by_partition"]["holdout"]["overall"]),
    ]:
        matrix = values["confusion_matrix"]
        lines.append(
            f"| {scope} | {values['labeled_case_count']} | "
            f"{matrix['true_positive']} | {matrix['false_positive']} | "
            f"{matrix['false_negative']} | {matrix['true_negative']} | "
            f"{values['precision']} | {values['recall']} | {values['accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## 策略冻结建议",
            "",
            "- **全局缺省与旧 suite：显式冻结 v2。** 当前旧 suite 没有 authored concept_groups；"
            "不得让省略 case_pass_policy 的新 run 自动落到 v3。旧 suite 的 gate_policy 应显式写入 "
            "`case_pass_policy.policy_version=stt.eval.case.v2`。",
            "- **全量 reviewed concept suite：只允许显式 opt-in v3。** 仅在所有可回答题均有独立审阅的 "
            "concept_groups、不可回答题的 absence contract 已审阅、manifest/hash 已冻结且所有 review gate "
            "通过后，才可在该 suite 的 gate_policy 显式写入 `stt.eval.case.v3`；先作为 shadow/candidate，"
            "不能继承为所有 run 的全局默认。",
            "- **不可重解释历史运行。** 已冻结的 v1/v2/v3 run 继续按其 config snapshot 回放。",
            "- **转正门禁。** v3 只有在新 suite 的 current-run final 全量独立盲审和稳定 holdout 均通过后，"
            "才能用于 baseline registration；当前 33/141 exact-answer 子集不足以转正。",
            "",
            "## 代表样本证据复核",
            "",
        ]
    )
    for sample in summary["representative_samples"]:
        lines.extend(
            [
                f"### {sample['case_id']} · {sample['confusion_bucket']} · "
                f"{sample['primary_reason_code']} · {sample['partition']}",
                "",
                f"- 问题：{sample['query']}",
                f"- 参考：{sample['reference_answer']}",
                f"- pre-grounding 答案：{sample['answer']}",
                f"- 判定：blind_semantic_correct={sample['blind_semantic_correct']}；"
                f"v3_final_correct={sample['v3_final_correct']}。",
                f"- 最相关授权上下文（{sample['context_label'] or '无标签'}）：{sample['context_excerpt']}",
                f"- 可核指标：reference→answer coverage={sample['reference_coverage_in_answer']}；"
                f"reference→context coverage={sample['reference_coverage_in_context']}；"
                f"v3 reasons={', '.join(sample['v3_reason_codes'])}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 数据完整性与方法边界",
            "",
            f"- case join：{summary['integrity']['case_join_complete']}；"
            f"shadow 行数={summary['integrity']['shadow_row_count']}；"
            f"pre 独立盲审正式记录={summary['integrity']['pre_blind_case_count']}。",
            f"- 排除项：{', '.join(summary['exclusions']['blocking_reference_case_ids'])}。",
            "- 只按 case_id join，并验证 dataset query/reference hash、可见答案 hash、review record hash 与输入文件 SHA-256。",
            "- final 仅在同一 case 的 query/reference/visible-answer 三者语义输入完全同一时复用标签；不同答案绝不跨 run 迁移标签。",
            "- 反事实合同门禁仍全部拦截 fabricated refusal fact、numeric mutation、multi-branch truncation 与 unsupported false addition；这不能替代真实 final 语义校准。",
            "- 本报告不修改 app、metrics 或 tests；修复后必须重新运行 shadow replay 和本脚本。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    pre_summary = _load_json(PRE_SUMMARY_PATH)
    prior_final_summary = _load_json(PRIOR_FINAL_SUMMARY_PATH)
    shadow_summary = _load_json(SHADOW_SUMMARY_PATH)
    shadow_checksums = _verify_checksum_manifest(SHADOW_CHECKSUMS_PATH)
    pdf_summary = _load_json(PDF_SUMMARY_PATH)
    pdf_cases = _load_json(PDF_CASES_PATH)
    pdf_by_case = {str(item["case_id"]): item for item in pdf_cases}
    blocked = set(pdf_summary["blocking_reference_cases"])
    if pre_summary.get("status") != "succeeded":
        raise ValueError("pre-grounding blind review is not complete")
    if pre_summary.get("records_sha256") != _sha256_file(PRE_RECORDS_PATH):
        raise ValueError("pre-grounding review file does not match its summary")
    if prior_final_summary.get("records_sha256") != _sha256_file(
        PRIOR_FINAL_RECORDS_PATH
    ):
        raise ValueError("prior final review file does not match its summary")
    if (
        pre_summary["judge"]["requested_model"]
        != prior_final_summary["judge"]["requested_model"]
        or pre_summary["judge"]["prompt_version"]
        != prior_final_summary["judge"]["prompt_version"]
    ):
        raise ValueError("blind reviews use different judge contracts")
    if not shadow_summary.get("join_complete") or shadow_summary.get(
        "input_case_count"
    ) != 150:
        raise ValueError("shadow replay summary is not a complete 150-case join")
    if shadow_checksums.get("shadow-cases.json") != _sha256_file(SHADOW_CASES_PATH):
        raise ValueError("shadow cases are not pinned by the shadow checksum manifest")
    if shadow_checksums.get("summary.json") != _sha256_file(SHADOW_SUMMARY_PATH):
        raise ValueError("shadow summary is not pinned by the shadow checksum manifest")
    for relative_path, expected_sha256 in shadow_summary["input_sha256"].items():
        source_path = _path_from_repo(relative_path.replace("\\", "/"))
        if _sha256_file(source_path) != expected_sha256:
            raise ValueError(f"shadow input hash mismatch: {relative_path}")

    pre_records = _latest_evaluation_records(
        _load_jsonl(PRE_RECORDS_PATH), expected_stage="pre_grounding"
    )
    prior_final_records = _latest_evaluation_records(
        _load_jsonl(PRIOR_FINAL_RECORDS_PATH), expected_stage=None
    )
    if len(pre_records) != 150 or len(prior_final_records) != 150:
        raise ValueError("expected 150 formal records in both blind-review inputs")

    datasets: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    input_paths = {
        SHADOW_CASES_PATH,
        SHADOW_SUMMARY_PATH,
        PRE_RECORDS_PATH,
        PRE_SUMMARY_PATH,
        PRIOR_FINAL_RECORDS_PATH,
        PRIOR_FINAL_SUMMARY_PATH,
        PDF_SUMMARY_PATH,
        PDF_CASES_PATH,
        SHADOW_CHECKSUMS_PATH,
    }
    for difficulty in ("easy", "medium", "hard"):
        detail = pre_summary["per_difficulty"][difficulty]
        dataset_path = _path_from_repo(detail["dataset_path"])
        results_path = _path_from_repo(detail["results_path"])
        input_paths.update({dataset_path, results_path})
        if _sha256_file(dataset_path) != detail["dataset_sha256"]:
            raise ValueError(f"dataset hash mismatch: {difficulty}")
        if _sha256_file(results_path) != detail["results_sha256"]:
            raise ValueError(f"results hash mismatch: {difficulty}")
        dataset = _load_json(dataset_path)
        cases = dataset["cases"]
        if len(cases) != 50:
            raise ValueError(f"expected 50 cases: {difficulty}")
        for case in cases:
            case_id = str(case["case_id"])
            if case_id in datasets:
                raise ValueError(f"duplicate dataset case: {case_id}")
            datasets[case_id] = case
        for row in _load_jsonl(results_path):
            case_id = str(row["case_id"])
            if case_id in results:
                raise ValueError(f"duplicate result case: {case_id}")
            results[case_id] = row

    expected_case_ids = set(datasets)
    if (
        len(expected_case_ids) != 150
        or set(results) != expected_case_ids
        or set(pre_records) != expected_case_ids
        or set(prior_final_records) != expected_case_ids
    ):
        raise ValueError("dataset/results/blind-review join is incomplete")

    shadow_rows = _load_json(SHADOW_CASES_PATH)
    shadow_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in shadow_rows:
        key = (str(row["case_id"]), str(row["stage"]))
        if key in shadow_by_key:
            raise ValueError(f"duplicate shadow row: {key}")
        shadow_by_key[key] = row
    expected_shadow_keys = {
        (case_id, stage)
        for case_id in expected_case_ids
        for stage in ("final", "pre_grounding")
    }
    if set(shadow_by_key) != expected_shadow_keys:
        raise ValueError("shadow replay must contain final and pre_grounding for 150 cases")
    if not blocked <= expected_case_ids or len(blocked) != 9:
        raise ValueError("expected exactly nine blocking reference cases")

    case_rows: list[dict[str, Any]] = []
    observations: dict[str, Mapping[str, Any]] = {}
    for case_id in sorted(expected_case_ids):
        case = datasets[case_id]
        result = results[case_id]
        observation = result["observation"]
        observations[case_id] = observation
        query = str(case.get("query") or "")
        reference = str(case.get("reference_answer") or "")
        query_sha256 = _sha256_text(query)
        reference_sha256 = _sha256_text(reference)
        partition, partition_sha256, partition_bucket = _content_sha_partition(
            query_sha256, reference_sha256
        )
        if pre_records[case_id]["source"]["query_sha256"] != query_sha256:
            raise ValueError(f"pre blind query hash mismatch: {case_id}")
        if (
            pre_records[case_id]["source"]["reference_answer_sha256"]
            != reference_sha256
        ):
            raise ValueError(f"pre blind reference hash mismatch: {case_id}")

        answers = {
            "pre_grounding": str(observation.get("pre_grounding_answer") or ""),
            "final": str(observation.get("answer") or ""),
        }
        answer_hashes = {stage: _sha256_text(answer) for stage, answer in answers.items()}
        if (
            answer_hashes["pre_grounding"]
            != pre_records[case_id]["source"]["selected_visible_answer_sha256"]
        ):
            raise ValueError(f"pre blind answer hash mismatch: {case_id}")

        for stage in ("pre_grounding", "final"):
            blind_label: bool | None = None
            blind_source: str | None = None
            blind_record_sha256: str | None = None
            if stage == "pre_grounding":
                record = pre_records[case_id]
                blind_label = bool(record["judgment"]["semantic_correct"])
                blind_source = "current_run_pre_grounding_blind_review"
                blind_record_sha256 = str(record["record_sha256"])
            elif answer_hashes["final"] == answer_hashes["pre_grounding"]:
                record = pre_records[case_id]
                blind_label = bool(record["judgment"]["semantic_correct"])
                blind_source = "same_case_exact_visible_answer_as_pre_grounding"
                blind_record_sha256 = str(record["record_sha256"])
            else:
                record = prior_final_records[case_id]
                source = record["source"]
                query_matches = source["query_sha256"] == query_sha256
                reference_matches = (
                    source["reference_answer_sha256"] == reference_sha256
                )
                answer_matches = (
                    source["final_visible_answer_sha256"] == answer_hashes["final"]
                )
                if query_matches and reference_matches and answer_matches:
                    blind_label = bool(record["judgment"]["semantic_correct"])
                    blind_source = "prior_run_exact_semantic_input_identity"
                    blind_record_sha256 = str(record["record_sha256"])

            shadow = shadow_by_key[(case_id, stage)]
            v3_correct = bool(shadow["v3_final_correct"])
            _, reference_context_coverage, _ = _best_context_excerpt(
                observation, reference, answers[stage]
            )
            confusion_bucket = None
            if blind_label is not None and case_id not in blocked:
                confusion_bucket = (
                    "TP"
                    if v3_correct and blind_label
                    else "FP"
                    if v3_correct and not blind_label
                    else "FN"
                    if not v3_correct and blind_label
                    else "TN"
                )
            pdf_case = pdf_by_case.get(case_id)
            case_rows.append(
                {
                    "case_id": case_id,
                    "difficulty": str(case.get("difficulty") or shadow["difficulty"]),
                    "stage": stage,
                    "partition": partition,
                    "partition_sha256": partition_sha256,
                    "partition_bucket": partition_bucket,
                    "excluded_from_metrics": case_id in blocked,
                    "exclusion_reason": (
                        "pdf_visual_review_blocking_reference_case"
                        if case_id in blocked
                        else None
                    ),
                    "reference_disposition": (
                        pdf_case.get("reference_disposition") if pdf_case else None
                    ),
                    "blind_label_available": blind_label is not None,
                    "blind_semantic_correct": blind_label,
                    "blind_label_source": blind_source,
                    "blind_record_sha256": blind_record_sha256,
                    "v3_final_correct": v3_correct,
                    "confusion_bucket": confusion_bucket,
                    "v3_reason_codes": sorted(set(shadow["v3_reason_codes"])),
                    "contract_correctness": shadow.get("contract_correctness"),
                    "coverage_correctness": shadow.get("coverage_correctness"),
                    "unsupported_factual_claim_count": shadow.get(
                        "unsupported_factual_claim_count"
                    ),
                    "contradictory_factual_claim_count": shadow.get(
                        "contradictory_factual_claim_count"
                    ),
                    "abstention_factual_claim_count": shadow.get(
                        "abstention_factual_claim_count"
                    ),
                    "required_branch_count": shadow.get("required_branch_count"),
                    "required_branch_coverage": shadow.get(
                        "required_branch_coverage"
                    ),
                    "critical_numbers_preserved": shadow.get(
                        "critical_numbers_preserved"
                    ),
                    "critical_identifiers_preserved": shadow.get(
                        "critical_identifiers_preserved"
                    ),
                    "polarity_constraints_preserved": shadow.get(
                        "polarity_constraints_preserved"
                    ),
                    "query_sha256": query_sha256,
                    "reference_answer_sha256": reference_sha256,
                    "visible_answer_sha256": answer_hashes[stage],
                    "visible_answer_char_count": len(answers[stage]),
                    "reference_coverage_in_visible_answer": _coverage(
                        reference, answers[stage]
                    ),
                    "reference_coverage_in_best_context": reference_context_coverage,
                    "final_equals_pre_grounding": (
                        answer_hashes["final"] == answer_hashes["pre_grounding"]
                    ),
                }
            )

    pre_rows = [row for row in case_rows if row["stage"] == "pre_grounding"]
    final_rows = [row for row in case_rows if row["stage"] == "final"]
    pre_overall = _confusion_summary(pre_rows)
    final_overall = _confusion_summary(final_rows)
    pre_by_partition = _by_partition(pre_rows)
    final_by_partition = _by_partition(final_rows)
    final_provenance = Counter(
        row["blind_label_source"]
        for row in final_rows
        if row["blind_label_source"] is not None
        and not row["excluded_from_metrics"]
    )

    representative_samples: list[dict[str, Any]] = []
    already_selected: set[str] = set()

    def add_representative(row: dict[str, Any], primary_reason: str) -> None:
        case_id = row["case_id"]
        if case_id in already_selected:
            return
        case = datasets[case_id]
        answer = str(observations[case_id].get("pre_grounding_answer") or "")
        reference = str(case.get("reference_answer") or "")
        excerpt, context_coverage, context_label = _best_context_excerpt(
            observations[case_id], reference, answer
        )
        representative_samples.append(
            {
                "case_id": case_id,
                "difficulty": row["difficulty"],
                "partition": row["partition"],
                "confusion_bucket": row["confusion_bucket"],
                "primary_reason_code": primary_reason,
                "v3_reason_codes": row["v3_reason_codes"],
                "query": str(case.get("query") or ""),
                "reference_answer": reference,
                "answer": answer,
                "context_label": context_label,
                "context_excerpt": excerpt,
                "reference_coverage_in_answer": _coverage(reference, answer),
                "reference_coverage_in_context": context_coverage,
                "blind_semantic_correct": row["blind_semantic_correct"],
                "v3_final_correct": row["v3_final_correct"],
            }
        )
        already_selected.add(case_id)

    per_reason_limit = {
        "unsupported_factual_claim": 3,
        "required_branch_incomplete": 2,
        "polarity_mismatch": 2,
        "critical_number_mismatch": 2,
        "critical_identifier_mismatch": 1,
        "contradictory_factual_claim": 2,
        "abstention_incorrect": 2,
        "abstention_contains_factual_claim": 1,
    }
    for reason in TARGET_REASON_CODES:
        candidates = [
            row
            for row in pre_rows
            if row["confusion_bucket"] == "FN" and reason in row["v3_reason_codes"]
        ]
        ranked: list[tuple[float, float, int, str, dict[str, Any]]] = []
        for row in candidates:
            case_id = row["case_id"]
            case = datasets[case_id]
            answer = str(observations[case_id].get("pre_grounding_answer") or "")
            reference = str(case.get("reference_answer") or "")
            excerpt, context_coverage, _ = _best_context_excerpt(
                observations[case_id], reference, answer
            )
            ranked.append(
                (
                    _coverage(reference, answer) or 0.0,
                    context_coverage or 0.0,
                    -len(answer),
                    case_id,
                    row,
                )
            )
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected_count = 0
        for _, _, _, case_id, row in ranked:
            if case_id in already_selected:
                continue
            add_representative(row, reason)
            selected_count += 1
            if selected_count >= per_reason_limit[reason]:
                break

    false_negatives = [row for row in pre_rows if row["confusion_bucket"] == "FN"]
    false_negatives.sort(
        key=lambda row: (
            -(row["reference_coverage_in_visible_answer"] or 0.0),
            -(row["reference_coverage_in_best_context"] or 0.0),
            row["partition_sha256"],
        )
    )
    for row in false_negatives:
        if sum(
            sample["confusion_bucket"] == "FN" for sample in representative_samples
        ) >= 5:
            break
        add_representative(row, "coverage_or_contract_false_negative")

    false_positives = [row for row in pre_rows if row["confusion_bucket"] == "FP"]
    false_positives.sort(
        key=lambda row: (row["partition"], row["difficulty"], row["partition_sha256"])
    )
    for row in false_positives[:5]:
        add_representative(row, "v3_pass_blind_semantic_incorrect")

    input_hashes = {
        _relative(path): _sha256_file(path) for path in sorted(input_paths)
    }
    summary = {
        "schema_version": "stt.eval.v3-blind-calibration.v2",
        "suite_key": shadow_summary["suite_key"],
        "run_label": shadow_summary["run_label"],
        "policy_versions": shadow_summary["policy_versions"],
        "analysis_policy": {
            "positive_class": "independent_blind_semantic_correct",
            "predicted_positive": "evaluator_v3_final_correct",
            "pre_grounding_label_policy": "current run independent qwen3.7-max blind review",
            "final_label_policy": (
                "reuse only when case_id, query, reference and exact visible answer are identical; "
                "otherwise unknown"
            ),
            "blocking_reference_policy": "exclude nine PDF visual-review blockers",
            "blind_judge_role": "independent calibration proxy, not absolute ground truth",
            "partition": {
                "version": SPLIT_VERSION,
                "input": "sha256(query) + sha256(reference_answer)",
                "rule": "first 64 digest bits / 2^64; calibration if <0.5 else holdout",
                "uses_case_id": False,
                "uses_difficulty": False,
                "uses_labels_or_evaluator_output": False,
                "caveat": (
                    "the split is deterministic and label-free, but evaluator revisions used "
                    "aggregate full-set feedback before this report; holdout is not claimed as "
                    "prospectively untouched"
                ),
            },
        },
        "inputs": {
            "sha256": input_hashes,
            "script_path": _relative(SCRIPT_PATH),
            "script_sha256": _sha256_file(SCRIPT_PATH),
            "pre_blind_judge": {
                "model": pre_summary["judge"]["requested_model"],
                "prompt_version": pre_summary["judge"]["prompt_version"],
                "records_path": _relative(PRE_RECORDS_PATH),
                "record_count": pre_summary["record_count"],
            },
            "prior_final_blind_judge": {
                "run_label": prior_final_summary["label"],
                "records_path": _relative(PRIOR_FINAL_RECORDS_PATH),
                "use": "exact semantic-input identity rows only",
            },
        },
        "integrity": {
            "case_join_complete": True,
            "dataset_case_count": len(datasets),
            "authored_concept_group_case_count": sum(
                "concept_groups" in case for case in datasets.values()
            ),
            "result_case_count": len(results),
            "pre_blind_case_count": len(pre_records),
            "prior_final_blind_case_count": len(prior_final_records),
            "shadow_row_count": len(shadow_rows),
            "shadow_stage_case_count": {
                stage: sum(row["stage"] == stage for row in shadow_rows)
                for stage in ("final", "pre_grounding")
            },
            "blind_record_hashes_verified": True,
            "query_reference_hashes_verified": True,
            "pre_visible_answer_hashes_verified": True,
            "shadow_checksum_manifest_verified": True,
            "shadow_summary_input_hashes_verified": True,
            "input_file_hashes_recorded": True,
            "partition_is_content_sha_only": True,
            "partition_case_counts_before_exclusion": dict(
                sorted(
                    Counter(
                        row["partition"]
                        for row in pre_rows
                    ).items()
                )
            ),
            "partition_case_counts_after_exclusion": dict(
                sorted(
                    Counter(
                        row["partition"]
                        for row in pre_rows
                        if not row["excluded_from_metrics"]
                    ).items()
                )
            ),
        },
        "exclusions": {
            "blocking_reference_case_count": len(blocked),
            "blocking_reference_case_ids": sorted(blocked),
            "source": _relative(PDF_SUMMARY_PATH),
            "dispositions": {
                case_id: pdf_by_case[case_id]["reference_disposition"]
                for case_id in sorted(blocked)
            },
        },
        "calibration": {
            "pre_grounding": {
                "overall": pre_overall,
                "by_difficulty": _by_difficulty(pre_rows),
                "by_partition": pre_by_partition,
                "reason_analysis": _reason_analysis(pre_rows),
                "label_scope": "complete after explicit PDF blocker exclusions",
            },
            "final": {
                "overall": final_overall,
                "by_difficulty": _by_difficulty(final_rows),
                "by_partition": final_by_partition,
                "reason_analysis": _reason_analysis(final_rows),
                "blind_label_availability": {
                    "eligible_case_count": final_overall["eligible_case_count"],
                    "eligible_labeled_count": final_overall["labeled_case_count"],
                    "eligible_unlabeled_count": (
                        final_overall["eligible_case_count"]
                        - final_overall["labeled_case_count"]
                    ),
                    "provenance_counts": dict(sorted(final_provenance.items())),
                    "full_current_run_final_blind_review_available": False,
                    "selection_bias": (
                        "label availability depends on exact answer identity; partial final metrics "
                        "must not be generalized to all 150 cases"
                    ),
                },
                "label_scope": "partial exact-semantic-input identity subset only",
            },
        },
        "representative_samples": representative_samples,
        "assessment": {
            "status": "not_ready_for_production_default",
            "pre_grounding_calibration_conclusion": (
                f"{pre_overall['confusion_matrix']['false_negative']} false negatives and "
                f"{pre_overall['confusion_matrix']['false_positive']} false positives versus "
                "the independent blind proxy after official source exclusions"
            ),
            "final_calibration_conclusion": (
                "not certifiable without a complete current-run final blind review"
            ),
            "production_default_recommended": False,
            "production_default_blockers": [
                "pre-grounding blind-proxy mismatches remain after official source exclusions",
                "current-run final blind labels are unavailable for most eligible cases",
                "the final exact-answer subset is selection-biased and cannot be extrapolated",
            ],
            "production_baseline_gate": (
                "blocked_by_pre_proxy_mismatch_and_incomplete_final_label_coverage"
            ),
            "policy_freeze_recommendation": {
                "global_default_when_case_policy_omitted": "stt.eval.case.v2",
                "legacy_suite_without_authored_concept_groups": (
                    "explicitly_pin_stt.eval.case.v2"
                ),
                "fully_reviewed_concept_suite": (
                    "explicit_stt.eval.case.v3_opt_in_for_shadow_candidate_only"
                ),
                "v3_opt_in_preconditions": [
                    "all answerable cases have independently reviewed concept_groups",
                    "all unanswerable cases have independently reviewed absence contracts",
                    "all suite review gates pass and manifest hashes are frozen",
                    "current-run final full blind review and stable holdout pass before baseline registration",
                ],
                "historical_runs": "preserve_frozen_snapshot_policy_without_reinterpretation",
            },
            "synthetic_contract_gates": shadow_summary["counterfactuals"],
        },
    }

    cases_json_path = OUTPUT_DIR / "cases.json"
    cases_csv_path = OUTPUT_DIR / "cases.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    _write_json(cases_json_path, case_rows)
    _write_json(summary_path, summary)

    csv_fields = [
        "case_id",
        "difficulty",
        "stage",
        "partition",
        "partition_sha256",
        "partition_bucket",
        "excluded_from_metrics",
        "reference_disposition",
        "blind_label_available",
        "blind_semantic_correct",
        "blind_label_source",
        "v3_final_correct",
        "confusion_bucket",
        "v3_reason_codes",
        "contract_correctness",
        "coverage_correctness",
        "unsupported_factual_claim_count",
        "contradictory_factual_claim_count",
        "abstention_factual_claim_count",
        "required_branch_count",
        "required_branch_coverage",
        "critical_numbers_preserved",
        "critical_identifiers_preserved",
        "polarity_constraints_preserved",
        "reference_coverage_in_visible_answer",
        "reference_coverage_in_best_context",
        "visible_answer_sha256",
        "final_equals_pre_grounding",
    ]
    with cases_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in case_rows:
            csv_row = {key: row.get(key) for key in csv_fields}
            csv_row["v3_reason_codes"] = "|".join(row["v3_reason_codes"])
            writer.writerow(csv_row)

    report_path.write_text(
        _render_report(summary), encoding="utf-8", newline="\n"
    )
    checksum_paths = [
        SCRIPT_PATH,
        summary_path,
        cases_json_path,
        cases_csv_path,
        report_path,
    ]
    checksum_lines = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(checksum_paths, key=lambda path: path.name)
    ]
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": summary["assessment"]["status"],
                "pre_grounding": pre_overall,
                "final_partial": final_overall,
                "output_dir": _relative(OUTPUT_DIR),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
