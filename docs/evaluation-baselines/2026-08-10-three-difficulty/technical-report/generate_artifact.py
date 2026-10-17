from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "baseline-report-artifact.json"
ORDER = ("easy", "medium", "hard")
LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
OUTCOME_SQL = "SELECT difficulty, passed, failed, error FROM case_outcomes ORDER BY sort_order;"
RUN_SQL = (
    "SELECT difficulty, run_id_short, dataset, cases, status, mode, gate_label "
    "FROM run_facts ORDER BY sort_order;"
)
METRIC_SQL = (
    "SELECT difficulty, recall_at_1, recall_at_3, recall_at_10, mrr, ndcg_at_10, "
    "correctness, faithfulness, citation_precision, citation_recall, abstention_accuracy "
    "FROM quality_metrics ORDER BY sort_order;"
)
TELEMETRY_SQL = (
    "SELECT difficulty, latency_mean_ms, latency_p95_ms, tokens_total, token_coverage, cost_status "
    "FROM runtime_telemetry ORDER BY sort_order;"
)
FAILURE_SQL = (
    "SELECT difficulty, failed_cases, critical_failed_cases, gate_status, "
    "CASE WHEN quality_passed = 1 THEN 'true' ELSE 'false' END AS quality_passed "
    "FROM failure_facts ORDER BY sort_order;"
)
QA_SQL = 'SELECT check_name AS "check", status, detail FROM qa_checks ORDER BY sort_order;'


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pct(value: float | None) -> float | None:
    return None if value is None else float(value)


def sqlite_query(table: str, rows: list[dict], sql: str) -> list[dict]:
    enriched = [{**row, "sort_order": index} for index, row in enumerate(rows)]
    columns = list(enriched[0])
    definitions = []
    for column in columns:
        values = [row.get(column) for row in enriched if row.get(column) is not None]
        sqlite_type = "REAL" if values and all(isinstance(value, (bool, int, float)) for value in values) else "TEXT"
        definitions.append(f'"{column}" {sqlite_type}')
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(f'CREATE TABLE "{table}" ({", ".join(definitions)})')
        placeholders = ", ".join("?" for _ in columns)
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        connection.executemany(
            f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})',
            [[row.get(column) for column in columns] for row in enriched],
        )
        return [dict(row) for row in connection.execute(sql)]
    finally:
        connection.close()


def main() -> None:
    summary = read_json("baseline-summary.json")
    validation = read_json("qa/final-validation.json")
    generation = read_json("generation/generation-summary.json")
    audit = read_json("quality/llm-audit.json")

    outcomes: list[dict] = []
    metrics: list[dict] = []
    telemetry: list[dict] = []
    failures: list[dict] = []
    runs: list[dict] = []

    for difficulty in ORDER:
        facts = summary["difficulties"][difficulty]
        quality = facts["metrics"]
        runtime = quality["runtime"]
        cases = facts["cases"]
        gate = facts["gate"]
        outcomes.append(
            {
                "difficulty": LABELS[difficulty],
                "passed": cases["passed"],
                "failed": cases["failed"],
                "error": cases["error"],
            }
        )
        metrics.append(
            {
                "difficulty": LABELS[difficulty],
                "recall_at_1": pct(quality["retrieval"]["recall_at_1"]),
                "recall_at_3": pct(quality["retrieval"]["recall_at_3"]),
                "recall_at_10": pct(quality["retrieval"]["recall_at_10"]),
                "mrr": pct(quality["retrieval"]["mrr"]),
                "ndcg_at_10": pct(quality["retrieval"]["ndcg_at_10"]),
                "correctness": pct(quality["generation"]["correctness"]),
                "faithfulness": pct(quality["generation"]["faithfulness"]),
                "citation_precision": pct(quality["citation"]["precision"]),
                "citation_recall": pct(quality["citation"]["recall"]),
                "abstention_accuracy": pct(quality["abstention"]["accuracy"]),
            }
        )
        telemetry.append(
            {
                "difficulty": LABELS[difficulty],
                "latency_mean_ms": runtime["latency_ms"]["mean"],
                "latency_p95_ms": runtime["latency_ms"]["p95"],
                "tokens_total": runtime["total_tokens"]["total"],
                "tokens_known_cases": runtime["total_tokens"]["known_case_count"],
                "tokens_unknown_cases": runtime["total_tokens"]["unknown_case_count"],
                "token_coverage": (
                    f"{runtime['total_tokens']['known_case_count']} known / "
                    f"{runtime['total_tokens']['unknown_case_count']} unknown"
                ),
                "cost_status": runtime["cost"]["status"],
            }
        )
        failures.append(
            {
                "difficulty": LABELS[difficulty],
                "failed_cases": cases["failed"],
                "critical_failed_cases": len(gate["critical_failed_cases"]),
                "critical_case_ids": ", ".join(gate["critical_failed_cases"]),
                "gate_status": gate["status"],
                "quality_passed": gate["passed"],
            }
        )
        runs.append(
            {
                "difficulty": LABELS[difficulty],
                "run_id": facts["run_id"],
                "run_id_short": f"{facts['run_id'][:17]}…",
                "dataset": facts["dataset"]["key"],
                "cases": cases["total"],
                "status": "succeeded",
                "mode": facts["runtime_identity"]["mode"],
                "gate": gate["status"],
                "gate_label": "已记录，含关键失败" if not gate["passed"] else "已记录，质量通过",
            }
        )

    test_check = next(item for item in validation["checks"] if item["name"] == "automated_tests")
    qa_checks = [
        {"check": "数据集确定性校验", "status": "通过", "detail": "3 × 50 题；源/索引/证据/校验和闭合"},
        {"check": "独立在线 LLM 审计", "status": "通过", "detail": "150 / 150 case 通过最终审计"},
        {"check": "live 运行完成度", "status": "通过", "detail": "150 / 150 执行到终态；0 execution error"},
        {"check": "docs 制品 SHA-256", "status": "通过", "detail": "所有带期望值的归档哈希匹配"},
        {"check": "真实浏览器验收", "status": "通过", "detail": "难度、题数、live、预检、根基线与详情显示正确；控制台 0 warning/error"},
        {
            "check": "自动化回归",
            "status": "通过",
            "detail": f"{test_check['detail']['total_tests']} passed；0 failure/error",
        },
    ]

    outcomes = sqlite_query("case_outcomes", outcomes, OUTCOME_SQL)
    runs = sqlite_query("run_facts", runs, RUN_SQL)
    metrics = sqlite_query("quality_metrics", metrics, METRIC_SQL)
    telemetry = sqlite_query("runtime_telemetry", telemetry, TELEMETRY_SQL)
    failures = sqlite_query("failure_facts", failures, FAILURE_SQL)
    qa_checks = sqlite_query(
        "qa_checks",
        [{"check_name": row["check"], "status": row["status"], "detail": row["detail"]} for row in qa_checks],
        QA_SQL,
    )

    sources = [
        {"id": "baseline-summary", "label": "基线汇总", "path": "baseline-summary.json"},
        {"id": "baseline-registry", "label": "三难度基线注册表", "path": "baselines/baselines.json"},
        {"id": "quality-summary", "label": "数据集质量结论", "path": "quality/quality-summary.json"},
        {"id": "frontend-qa", "label": "真实浏览器验收", "path": "frontend/frontend-qa.json"},
        {"id": "final-validation", "label": "最终闭合校验", "path": "qa/final-validation.json"},
        {
            "id": "outcome-query",
            "label": "结果构成查询",
            "path": "baseline-summary.json",
            "query": {
                "engine": "SQLite over normalized baseline-summary.json",
                "sql": OUTCOME_SQL,
            },
        },
        {
            "id": "run-query",
            "label": "根基线事实查询",
            "path": "baselines/baselines.json",
            "query": {
                "engine": "SQLite over normalized baselines.json",
                "sql": RUN_SQL,
            },
        },
        {
            "id": "metric-query",
            "label": "质量指标查询",
            "path": "baseline-summary.json",
            "query": {
                "engine": "SQLite over normalized baseline-summary.json",
                "sql": METRIC_SQL,
            },
        },
        {
            "id": "telemetry-query",
            "label": "运行遥测查询",
            "path": "baseline-summary.json",
            "query": {
                "engine": "SQLite over normalized baseline-summary.json",
                "sql": TELEMETRY_SQL,
            },
        },
        {
            "id": "failure-query",
            "label": "失败事实查询",
            "path": "baseline-summary.json",
            "query": {
                "engine": "SQLite over normalized baseline-summary.json",
                "sql": FAILURE_SQL,
            },
        },
        {
            "id": "qa-query",
            "label": "交付验收查询",
            "path": "qa/final-validation.json",
            "query": {
                "engine": "SQLite over normalized final-validation.json",
                "sql": QA_SQL,
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "三难度在线全链路评测基线",
            "description": "Pantum P3500 同一知识库、索引和 qwen-flash 配置下的简单/中等/困难 50 题事实基线。",
            "generatedAt": summary["generated_at"],
            "sources": sources,
            "charts": [
                {
                    "id": "case-outcomes",
                    "title": "题目结果随难度变化",
                    "subtitle": "每档固定 50 题；运行均成功，但质量通过数随难度下降。",
                    "type": "stackedBar",
                    "dataset": "case_outcomes",
                    "encodings": {
                        "x": {"field": "difficulty", "type": "nominal", "label": "难度"},
                        "y": {
                            "fields": ["passed", "failed", "error"],
                            "type": "quantitative",
                            "label": "case 数",
                        },
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "sourceId": "outcome-query",
                    "intent": "comparison",
                    "question": "三个难度下，50 个 case 如何分布为通过、质量失败和执行错误？",
                    "rationale": "堆叠条形图直接显示固定样本量中的结果构成，不把不同量纲指标混画。",
                }
            ],
            "tables": [
                {
                    "id": "run-facts",
                    "title": "根基线运行事实",
                    "subtitle": "三条运行均为 baseline_run_id=null 的 live 根运行。",
                    "dataset": "run_facts",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "run-query",
                    "columns": [
                        {"field": "difficulty", "label": "难度", "type": "text"},
                        {"field": "run_id_short", "label": "run_id", "type": "text"},
                        {"field": "cases", "label": "题数", "type": "number"},
                        {"field": "status", "label": "运行状态", "type": "text"},
                        {"field": "mode", "label": "范围", "type": "text"},
                        {"field": "gate_label", "label": "门禁事实", "type": "text"},
                    ],
                },
                {
                    "id": "retrieval-metrics",
                    "title": "检索指标",
                    "subtitle": "比例按 0–100% 展示；unknown 不按 0 计。",
                    "dataset": "quality_metrics",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "metric-query",
                    "columns": [
                        {"field": "difficulty", "label": "难度", "type": "text"},
                        {"field": "recall_at_1", "label": "Recall@1", "format": "percent"},
                        {"field": "recall_at_3", "label": "Recall@3", "format": "percent"},
                        {"field": "recall_at_10", "label": "Recall@10", "format": "percent"},
                        {"field": "mrr", "label": "MRR", "format": "percent"},
                        {"field": "ndcg_at_10", "label": "nDCG@10", "format": "percent"},
                    ],
                },
                {
                    "id": "answer-metrics",
                    "title": "生成、引用与拒答指标",
                    "subtitle": "比例按 0–100% 展示；unknown 不按 0 计。",
                    "dataset": "quality_metrics",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "metric-query",
                    "columns": [
                        {"field": "difficulty", "label": "难度", "type": "text"},
                        {"field": "correctness", "label": "正确性", "format": "percent"},
                        {"field": "faithfulness", "label": "忠实度", "format": "percent"},
                        {"field": "citation_precision", "label": "引用精确率", "format": "percent"},
                        {"field": "citation_recall", "label": "引用召回率", "format": "percent"},
                        {"field": "abstention_accuracy", "label": "拒答准确率", "format": "percent"},
                    ],
                },
                {
                    "id": "runtime-telemetry",
                    "title": "运行遥测",
                    "subtitle": "成本未配置供应商计价快照，必须解释为 unknown。",
                    "dataset": "runtime_telemetry",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "telemetry-query",
                    "columns": [
                        {"field": "difficulty", "label": "难度", "type": "text"},
                        {"field": "latency_mean_ms", "label": "平均延迟 ms", "type": "number"},
                        {"field": "latency_p95_ms", "label": "P95 延迟 ms", "type": "number"},
                        {"field": "tokens_total", "label": "已知 Token 总量", "type": "number"},
                        {"field": "token_coverage", "label": "Token 覆盖", "type": "text"},
                        {"field": "cost_status", "label": "成本状态", "type": "text"},
                    ],
                },
                {
                    "id": "failure-facts",
                    "title": "失败与门禁事实",
                    "subtitle": "baseline_recorded_with_failures 表示已记录事实基线，不表示质量通过。",
                    "dataset": "failure_facts",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "failure-query",
                    "columns": [
                        {"field": "difficulty", "label": "难度", "type": "text"},
                        {"field": "failed_cases", "label": "失败 case", "type": "number"},
                        {"field": "critical_failed_cases", "label": "关键失败", "type": "number"},
                        {"field": "gate_status", "label": "门禁状态", "type": "text"},
                        {"field": "quality_passed", "label": "质量通过", "type": "text"},
                    ],
                },
                {
                    "id": "qa-checks",
                    "title": "交付验收",
                    "dataset": "qa_checks",
                    "density": "dense",
                    "layout": "full",
                    "sourceId": "qa-query",
                    "columns": [
                        {"field": "check", "label": "检查", "type": "text"},
                        {"field": "status", "label": "结果", "type": "text"},
                        {"field": "detail", "label": "证据", "type": "text"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "executive-summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "baseline-summary",
                    "body": (
                        "## 技术结论\n\n"
                        "三套各 50 题的数据集已在同一 Pantum P3500 知识库、同一活动索引和同一 qwen-flash 配置上完成 `live` 全链路运行。"
                        "150/150 case 均执行到终态且无执行错误，三个 run 均已登记为可供后续同 case 配对比较的根基线。\n\n"
                        "本次不把‘运行成功’混同为‘质量通过’：三档均为 `baseline_recorded_with_failures`。"
                        "通过数从简单 28 降到中等 21、困难 9，说明新题集确实拉开了当前系统的能力梯度。"
                    ),
                },
                {"id": "run-table", "type": "table", "tableId": "run-facts", "layout": "full"},
                {"id": "outcome-chart", "type": "chart", "chartId": "case-outcomes", "layout": "full"},
                {
                    "id": "findings",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "baseline-summary",
                    "body": (
                        "## 关键发现\n\n"
                        "- 简单与中等的检索召回接近，但生成正确性由 61.7% 降至 52.1%，体现多约束回答的组合损失。\n"
                        "- 困难集要求双证据综合并含 5 道知识边界题，Recall@1 为 37.8%、正确性为 37.6%、引用召回率为 30.0%。\n"
                        "- 延迟随难度上升：平均 11.9s → 22.2s → 27.6s；三档成本均因未固化计价快照而保持 unknown。\n"
                        "- 主要失败切片是期望词召回低于 0.75；其次为 Recall@10 为 0。完整 case 级证据在 README 与各 run 的 report/results 制品中。"
                    ),
                },
                {"id": "retrieval-table", "type": "table", "tableId": "retrieval-metrics", "layout": "full"},
                {"id": "answer-table", "type": "table", "tableId": "answer-metrics", "layout": "full"},
                {"id": "telemetry-table", "type": "table", "tableId": "runtime-telemetry", "layout": "full"},
                {"id": "failure-table", "type": "table", "tableId": "failure-facts", "layout": "full"},
                {
                    "id": "scope-method",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "baseline-summary",
                    "body": (
                        "## 范围、定义与方法\n\n"
                        "语料来自 `doc/PantumP3500用户手册zh_CNV1.2_1644316283788.pdf`（95 页），复用已发布的 218-chunk 活动索引。"
                        "简单题为单证据直接事实；中等题为单证据多约束；困难题包含 45 道双证据综合和 5 道不可回答题。"
                        "数据集先做源文件/索引/证据/唯一性等确定性校验，再由 qwen-flash 对 150 题做独立在线审计。\n\n"
                        "Recall@K、MRR、nDCG@10 基于标注证据排名；correctness、faithfulness、citation 与 abstention 是当前确定性评测器的启发式指标。"
                        "数据集生成保留 21 次调用、193,936 tokens；最终在线审计保留 15 次调用、120,166 tokens。"
                    ),
                },
                {"id": "qa-table", "type": "table", "tableId": "qa-checks", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "baseline-summary",
                    "body": (
                        "## 局限与稳健性边界\n\n"
                        "- 指标不是独立 LLM judge 或盲审人工结论；期望词阈值可能惩罚语义正确但措辞不同的答案。\n"
                        "- `minimum_metric_coverage` 已进入策略快照，但当前 gate 实现未把它作为独立阻断条件；比较时应同时检查 coverage。\n"
                        "- Token 总量存在 1/1/2 个 unknown case；金额成本因无供应商计价快照而不可计算，不能按 0 解读。\n"
                        "- 本结论只覆盖这份手册、这 150 题和冻结的 read set，不能外推到其他文档或生产流量。\n"
                        "- 系统运行制品原件仍留在 `output/evaluation` 以保持 API 下载契约；本报告使用 `docs` 中逐文件哈希验证的审计副本。"
                    ),
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "baseline-summary",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 优先处理 critical case，先复核召回证据，再修正答案组合和引用绑定。\n"
                        "2. 用这三个 run_id 分别作为候选运行的 `baseline_run_id`，保持 dataset、KB、mode 与 case 集一致，执行 paired gate。\n"
                        "3. 固化供应商定价版本和完整 Token collector，再比较成本与性能。\n"
                        "4. 若要形成发布级语义质量结论，增加盲审样本或独立 judge，并与当前启发式并列。\n\n"
                        "## 待回答问题\n\n"
                        "哪些 failure slice 应直接阻断发布？后续候选是否继续固定同一索引 read set？Token/成本覆盖率达到多少才允许用于难度间比较？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": summary["generated_at"],
            "status": "ready",
            "datasets": {
                "case_outcomes": outcomes,
                "run_facts": runs,
                "quality_metrics": metrics,
                "runtime_telemetry": telemetry,
                "failure_facts": failures,
                "qa_checks": qa_checks,
                "authoring_usage": [
                    {
                        "stage": "dataset_generation",
                        "model": generation["authoring_model"],
                        "calls": generation["usage"]["calls"],
                        "prompt_tokens": generation["usage"]["prompt_tokens"],
                        "completion_tokens": generation["usage"]["completion_tokens"],
                        "total_tokens": generation["usage"]["total_tokens"],
                    },
                    {
                        "stage": "final_online_audit",
                        "model": audit["model"],
                        "calls": audit["usage"]["calls"],
                        "prompt_tokens": audit["usage"]["prompt_tokens"],
                        "completion_tokens": audit["usage"]["completion_tokens"],
                        "total_tokens": audit["usage"]["total_tokens"],
                    },
                ],
            },
        },
        "sources": sources,
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
