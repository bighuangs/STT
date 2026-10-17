"""Safe JSON, Markdown and standalone HTML renderers for evaluation output."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


_UNKNOWN = "unknown"
_COMPLETENESS_PRESERVED_REASON = "POST_GENERATION_COMPLETENESS_PRESERVED"
_COMPLETENESS_PRESERVED_LABEL = "答案完整性优先，部分内容未通过引用校验"


def _json_default(value: Any) -> str:
    """Keep adapter UUID/datetime values serializable without executing markup."""

    return str(value)


def render_json_report(payload: Any) -> str:
    """Render stable, UTF-8 friendly JSON suitable for an immutable artifact."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _summary(payload: Any) -> Mapping[str, Any]:
    root = _mapping(payload)
    for key in ("aggregate", "summary"):
        if isinstance(root.get(key), Mapping):
            return root[key]
    return root


def _get(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _number(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return _UNKNOWN
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _UNKNOWN
    if not math.isfinite(number):
        return _UNKNOWN
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _safe_markdown(value: Any) -> str:
    # Escaping HTML makes Markdown safe even when its renderer permits raw HTML.
    return (
        html.escape(str(value), quote=True)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _mean(summary: Mapping[str, Any], *path: str) -> str:
    return _number(_get(summary, *path, "mean"))


def _report_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("case count", _number(summary.get("case_count"))),
        (
            "strict supported case pass rate",
            _mean(
                summary,
                "metrics",
                "case_outcome",
                "strict_supported_case_pass_rate",
            ),
        ),
        ("case quality score", _mean(summary, "metrics", "case_outcome", "quality_score")),
        ("MRR", _mean(summary, "metrics", "retrieval", "mrr")),
        (
            "final answer accuracy",
            _mean(summary, "metrics", "case_outcome", "final_answer_accuracy"),
        ),
        (
            "final answer contract score",
            _mean(summary, "metrics", "case_outcome", "final_answer_correctness"),
        ),
        (
            "coverage-only answer score (diagnostic)",
            _mean(summary, "metrics", "generation", "correctness"),
        ),
        ("faithfulness", _mean(summary, "metrics", "generation", "faithfulness")),
        ("citation precision", _mean(summary, "metrics", "citation", "precision")),
        ("citation recall", _mean(summary, "metrics", "citation", "recall")),
        ("citation coverage", _mean(summary, "metrics", "citation", "coverage")),
    ]
    recall_values = _get(summary, "metrics", "retrieval", "recall_at_k")
    if isinstance(recall_values, Mapping):
        for k in sorted(
            recall_values, key=lambda value: int(value) if str(value).isdigit() else str(value)
        ):
            rows.append((f"Recall@{k}", _number(_get(_mapping(recall_values[k]), "mean"))))
    rows.extend(
        (
            ("latency mean (ms)", _number(_get(summary, "runtime", "latency_ms", "mean"))),
            ("latency p95 (ms)", _number(_get(summary, "runtime", "latency_ms", "p95"))),
            # Missing telemetry must remain explicit; zero would mean a free call.
            ("tokens total", _number(_get(summary, "runtime", "total_tokens", "total"))),
            ("cost total", _number(_get(summary, "runtime", "cost", "total"))),
            ("cost currency", str(_get(summary, "runtime", "cost", "currency") or _UNKNOWN)),
        )
    )
    return rows


def _metadata(payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key in (
        "run_id",
        "evaluation_scope",
        "dataset_id",
        "dataset_version",
        "baseline_run_id",
        "candidate_run_id",
        "created_at",
    ):
        if key in payload and payload[key] is not None:
            rows.append((key, payload[key]))
    run = _mapping(payload.get("run"))
    for key in (
        "suite_key",
        "difficulty",
        "minimum_case_count",
        "evaluation_mode_default",
    ):
        value = payload.get(key)
        if value is None:
            value = run.get(key)
        if value is not None:
            rows.append((key, value))
    return rows


def _report_title(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("title")
    if explicit:
        return str(explicit)
    run = _mapping(payload.get("run"))
    scope = str(payload.get("evaluation_scope") or run.get("mode") or "")
    if scope == "retrieval_only":
        return "STT 仅检索评测报告"
    if scope in {"full", "live"}:
        return "STT 全链路在线评测报告"
    return "STT 离线评测报告"


def _comparison(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("comparison", "paired_diff"):
        if isinstance(payload.get(key), Mapping):
            return payload[key]
    if isinstance(payload.get("metric_deltas"), Mapping):
        return payload
    return {}


def _evaluated_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept both a raw evaluator result and its durable repository envelope."""

    nested = result.get("metrics")
    if isinstance(nested, Mapping) and isinstance(nested.get("metrics"), Mapping):
        return nested
    return result


def _visible_answer_audit_text(result: Mapping[str, Any], evaluated: Mapping[str, Any]) -> str:
    """Describe the explicit degraded-visible-answer policy, if present."""

    outcome = _mapping(_get(evaluated, "metrics", "case_outcome"))
    audit = _mapping(outcome.get("answer_display_audit"))
    audit_degraded = audit.get("status") == "degraded"
    label = str(audit.get("label") or _COMPLETENESS_PRESERVED_LABEL)

    observation = _mapping(result.get("observation"))
    policy: Mapping[str, Any] = {}
    for candidate in (
        observation.get("visible_answer_policy"),
        evaluated.get("visible_answer_policy"),
        outcome.get("visible_answer_policy"),
    ):
        if isinstance(candidate, Mapping) and candidate:
            policy = candidate
            break
    reason_code = str(
        policy.get("reason_code") or policy.get("reason") or audit.get("reason_code") or ""
    )
    decision = _mapping(observation.get("answer_decision"))
    if not reason_code:
        reason_code = str(decision.get("reason_code") or decision.get("reason") or "")
    if (
        not audit_degraded
        and policy.get("strict_grounding_eligible") is not False
        and reason_code != _COMPLETENESS_PRESERVED_REASON
    ):
        return _UNKNOWN

    details: list[str] = []
    if reason_code:
        details.append(reason_code)
    mode = policy.get("mode")
    if mode:
        details.append(f"mode={mode}")
    trusted_count = policy.get("trusted_citation_count")
    if isinstance(trusted_count, int) and not isinstance(trusted_count, bool):
        details.append(f"trusted_citations={trusted_count}")
    removed_count = policy.get("removed_untrusted_marker_count")
    if isinstance(removed_count, int) and not isinstance(removed_count, bool):
        details.append(f"removed_markers={removed_count}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return label + suffix


def _completeness_self_review_audit_text(
    result: Mapping[str, Any], evaluated: Mapping[str, Any]
) -> str:
    """Render only the approved non-CoT completeness-review audit fields."""

    observation = _mapping(result.get("observation"))
    review = observation.get("completeness_self_review")
    if not isinstance(review, Mapping) or not review:
        review = evaluated.get("completeness_self_review")
    if not isinstance(review, Mapping) or not review:
        return _UNKNOWN

    calls = review.get("calls_attempted")
    attempted = isinstance(calls, (int, float)) and not isinstance(calls, bool) and calls > 0
    status = str(review.get("status") or _UNKNOWN)
    committed_statuses = {
        "committed",
        "committed_insufficient_evidence",
        "committed_deterministic_set_cover",
        "committed_deterministic_insufficient_evidence",
        "current_complete",
    }
    model = str(review.get("model_id") or _UNKNOWN)
    usage = _mapping(review.get("usage"))
    usage_parts = [f"status={usage.get('status') or _UNKNOWN}"]
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            usage_parts.append(f"{key}={_number(value)}")
    trigger_codes = review.get("trigger_reason_codes")
    triggers = (
        ",".join(str(code) for code in trigger_codes)
        if isinstance(trigger_codes, Sequence) and not isinstance(trigger_codes, (str, bytes))
        else _UNKNOWN
    )
    return (
        f"attempted={'yes' if attempted else 'no'}; "
        f"status={status}; committed={'yes' if status in committed_statuses else 'no'}; "
        f"model={model}; usage={','.join(usage_parts)}; "
        f"trigger_codes={triggers or _UNKNOWN}"
    )


def render_markdown_report(payload: Any) -> str:
    """Render a compact report that remains safe in HTML-enabled Markdown."""

    root = _mapping(payload)
    summary = _summary(payload)
    title = _safe_markdown(_report_title(root))
    lines = [f"# {title}", ""]
    metadata = _metadata(root)
    if metadata:
        lines.extend(["## 运行事实", "", "| 字段 | 值 |", "|---|---|"])
        lines.extend(
            f"| {_safe_markdown(key)} | {_safe_markdown(value)} |" for key, value in metadata
        )
        lines.append("")

    lines.extend(["## 汇总指标", "", "| 指标 | 值 |", "|---|---:|"])
    lines.extend(
        f"| {_safe_markdown(name)} | {_safe_markdown(value)} |"
        for name, value in _report_rows(summary)
    )
    lines.extend(
        [
            "",
            (
                "`final_answer_accuracy` 按运行快照中的版本化 case policy 判定；manifest 省略策略时使用 v2。"
                "仅显式启用的 v3 候选概念合同会把无证据/矛盾事实、拒答夹带事实、关键数字/标识符/极性错误及明确多分支漏答计为最终错误。"
                "严格有据通过率仍是独立护栏；仅引用降级不会自动改写最终正确性，除非答案内容本身违反运行快照中的正确性合同。"
            ),
            "",
        ]
    )

    by_tag = summary.get("by_tag")
    if isinstance(by_tag, Mapping) and by_tag:
        lines.extend(
            [
                "## Tag 切片",
                "",
                "| Tag | Cases | Coverage score | Faithfulness | Recall@5 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for tag in sorted(by_tag, key=str):
            detail = _mapping(by_tag[tag])
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    _safe_markdown(tag),
                    _number(detail.get("case_count")),
                    _mean(detail, "metrics", "generation", "correctness"),
                    _mean(detail, "metrics", "generation", "faithfulness"),
                    _mean(detail, "metrics", "retrieval", "recall_at_k", "5"),
                )
            )
        lines.append("")

    comparison = _comparison(root)
    deltas = comparison.get("metric_deltas")
    if isinstance(deltas, Mapping) and deltas:
        lines.extend(
            [
                "## 基线与候选配对差异",
                "",
                "| 指标 | Baseline | Candidate | Delta | Improved | Regressed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in sorted(deltas, key=str):
            detail = _mapping(deltas[name])
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    _safe_markdown(name),
                    _number(detail.get("baseline_mean")),
                    _number(detail.get("candidate_mean")),
                    _number(detail.get("delta")),
                    _number(detail.get("improved_cases")),
                    _number(detail.get("regressed_cases")),
                )
            )
        lines.append("")

    results = root.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)) and results:
        lines.extend(
            [
                "## 用例明细",
                "",
                "| Case | Final answer correct | Strict supported pass | Quality | Reasons | Visible answer audit | Completeness self-review audit | Scope | Applicable | Tags | Final answer contract score | Faithfulness | MRR | Latency(ms) | Tokens | Cost |",
                "|---|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for raw in results:
            result = _mapping(raw)
            evaluated = _evaluated_result(result)
            outcome = _mapping(_get(evaluated, "metrics", "case_outcome"))
            passed = outcome.get("strict_supported_case_pass")
            if not isinstance(passed, bool):
                passed = result.get("passed")
            if not isinstance(passed, bool):
                passed = outcome.get("passed")
            reasons = ", ".join(str(reason) for reason in outcome.get("reason_codes", []) or [])
            tags = ", ".join(str(tag) for tag in evaluated.get("tags", []) or [])
            applicable = _get(evaluated, "metrics", "retrieval", "applicable")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _safe_markdown(result.get("case_id", _UNKNOWN)),
                    _safe_markdown(
                        "correct"
                        if outcome.get("final_answer_correct") is True
                        else (
                            "incorrect"
                            if outcome.get("final_answer_correct") is False
                            else _UNKNOWN
                        )
                    ),
                    _safe_markdown(
                        "passed" if passed is True else ("failed" if passed is False else _UNKNOWN)
                    ),
                    _number(outcome.get("quality_score")),
                    _safe_markdown(reasons or _UNKNOWN),
                    _safe_markdown(_visible_answer_audit_text(result, evaluated)),
                    _safe_markdown(_completeness_self_review_audit_text(result, evaluated)),
                    _safe_markdown(evaluated.get("evaluation_scope", "full")),
                    "N/A" if applicable is False else _safe_markdown(applicable),
                    _safe_markdown(tags),
                    _number(outcome.get("final_answer_correctness")),
                    _number(_get(evaluated, "metrics", "generation", "faithfulness")),
                    _number(_get(evaluated, "metrics", "retrieval", "mrr")),
                    _number(_get(evaluated, "runtime", "latency_ms")),
                    _number(_get(evaluated, "runtime", "total_tokens")),
                    _number(_get(evaluated, "runtime", "cost")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "".join(f"<th>{html.escape(str(value), quote=True)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value), quote=True)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def render_html_report(payload: Any) -> str:
    """Render a standalone HTML report with every payload value escaped."""

    root = _mapping(payload)
    summary = _summary(payload)
    title = html.escape(_report_title(root), quote=True)
    sections: list[str] = []
    metadata = _metadata(root)
    if metadata:
        sections.append("<h2>运行事实</h2>" + _html_table(("字段", "值"), metadata))
    sections.append("<h2>汇总指标</h2>" + _html_table(("指标", "值"), _report_rows(summary)))
    sections.append(
        "<p><code>final_answer_accuracy</code> 按运行快照中的版本化 case policy 判定；"
        "manifest 省略策略时使用 v2。仅显式启用的 v3 候选概念合同会把无证据/矛盾事实、拒答夹带事实、"
        "关键数字/标识符/极性错误及明确多分支漏答计为最终错误。严格有据通过率仍是独立护栏；"
        "仅引用降级不会自动改写最终正确性，除非答案内容本身违反运行快照中的正确性合同。</p>"
    )

    by_tag = summary.get("by_tag")
    if isinstance(by_tag, Mapping) and by_tag:
        rows = []
        for tag in sorted(by_tag, key=str):
            detail = _mapping(by_tag[tag])
            rows.append(
                (
                    tag,
                    _number(detail.get("case_count")),
                    _mean(detail, "metrics", "generation", "correctness"),
                    _mean(detail, "metrics", "generation", "faithfulness"),
                    _mean(detail, "metrics", "retrieval", "recall_at_k", "5"),
                )
            )
        sections.append(
            "<h2>Tag 切片</h2>"
            + _html_table(("Tag", "Cases", "Coverage score", "Faithfulness", "Recall@5"), rows)
        )

    comparison = _comparison(root)
    deltas = comparison.get("metric_deltas")
    if isinstance(deltas, Mapping) and deltas:
        rows = []
        for name in sorted(deltas, key=str):
            detail = _mapping(deltas[name])
            rows.append(
                (
                    name,
                    _number(detail.get("baseline_mean")),
                    _number(detail.get("candidate_mean")),
                    _number(detail.get("delta")),
                    _number(detail.get("improved_cases")),
                    _number(detail.get("regressed_cases")),
                )
            )
        sections.append(
            "<h2>基线与候选配对差异</h2>"
            + _html_table(("指标", "Baseline", "Candidate", "Delta", "Improved", "Regressed"), rows)
        )

    results = root.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)) and results:
        rows = []
        for raw in results:
            result = _mapping(raw)
            evaluated = _evaluated_result(result)
            outcome = _mapping(_get(evaluated, "metrics", "case_outcome"))
            passed = outcome.get("strict_supported_case_pass")
            if not isinstance(passed, bool):
                passed = result.get("passed")
            if not isinstance(passed, bool):
                passed = outcome.get("passed")
            applicable = _get(evaluated, "metrics", "retrieval", "applicable")
            rows.append(
                (
                    result.get("case_id", _UNKNOWN),
                    (
                        "correct"
                        if outcome.get("final_answer_correct") is True
                        else (
                            "incorrect"
                            if outcome.get("final_answer_correct") is False
                            else _UNKNOWN
                        )
                    ),
                    "passed" if passed is True else ("failed" if passed is False else _UNKNOWN),
                    _number(outcome.get("quality_score")),
                    ", ".join(str(reason) for reason in outcome.get("reason_codes", []) or [])
                    or _UNKNOWN,
                    _visible_answer_audit_text(result, evaluated),
                    _completeness_self_review_audit_text(result, evaluated),
                    evaluated.get("evaluation_scope", "full"),
                    "N/A" if applicable is False else applicable,
                    ", ".join(str(tag) for tag in evaluated.get("tags", []) or []),
                    _number(outcome.get("final_answer_correctness")),
                    _number(_get(evaluated, "metrics", "generation", "faithfulness")),
                    _number(_get(evaluated, "metrics", "retrieval", "mrr")),
                    _number(_get(evaluated, "runtime", "latency_ms")),
                    _number(_get(evaluated, "runtime", "total_tokens")),
                    _number(_get(evaluated, "runtime", "cost")),
                )
            )
        sections.append(
            "<h2>用例明细</h2>"
            + _html_table(
                (
                    "Case",
                    "Final answer correct",
                    "Strict supported pass",
                    "Quality",
                    "Reasons",
                    "Visible answer audit",
                    "Completeness self-review audit",
                    "Scope",
                    "Applicable",
                    "Tags",
                    "Final answer contract score",
                    "Faithfulness",
                    "MRR",
                    "Latency(ms)",
                    "Tokens",
                    "Cost",
                ),
                rows,
            )
        )

    # The complete artifact is escaped too: adapter diagnostics are untrusted.
    raw_json = html.escape(render_json_report(payload), quote=True)
    sections.append(f"<details><summary>完整 JSON</summary><pre>{raw_json}</pre></details>")
    styles = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem;color:#172033}
table{border-collapse:collapse;width:100%;margin:.75rem 0 1.5rem}
th,td{border:1px solid #d9dfeb;padding:.5rem .65rem;text-align:left;vertical-align:top}
th{background:#f3f6fb}h1,h2{color:#153b70}pre{white-space:pre-wrap;word-break:break-word}
details{margin-top:2rem}code{background:#f3f6fb;padding:.1rem .25rem}
""".strip()
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        f"<style>{styles}</style></head><body><h1>{title}</h1>{''.join(sections)}</body></html>"
    )
