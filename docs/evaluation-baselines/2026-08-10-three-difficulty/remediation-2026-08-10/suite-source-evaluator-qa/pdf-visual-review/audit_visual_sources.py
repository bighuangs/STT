from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
QA_DIR = THIS_DIR.parent
SUITE_ROOT = QA_DIR.parents[1]
DATASET_DIR = SUITE_ROOT / "datasets"
SOURCE_RELATIVE_PATH = "doc/PantumP3500用户手册zh_CNV1.2_1644316283788.pdf"
SOURCE_SHA256 = "c4c7b2a434f3f86c4988d3961ac7304970029e71b0201fb6c10ca722b63bce2b"


PAGE_NOTES: dict[int, dict[str, str]] = {
    4: {
        "layout": "text+embedded-label+flowchart",
        "note": "激光警示标签和回收流程为嵌入图像；标签可见文本为 INVISIBLE LASER RADIATION。",
    },
    7: {"layout": "symbol+text", "note": "废弃电气设备符号旁有明确送往指定收集点的文字。"},
    8: {
        "layout": "structured-table",
        "note": "RoHS 部件×物质矩阵清晰可见；仅铅(Pb)列在金属、电线电缆、电路板行出现×。",
    },
    14: {
        "layout": "numbered-diagram+table",
        "note": "后视图编号及名称表清晰；编号1对应 USB 接口。",
    },
    18: {
        "layout": "embedded-report-screenshots",
        "note": "演示页和打印机信息页为小字号嵌入图，含月打印量及硒鼓剩余量。",
    },
    19: {
        "layout": "embedded-report-screenshots",
        "note": "菜单结构页和网络配置页为小字号嵌入图。",
    },
    20: {"layout": "embedded-table-screenshot", "note": "Wi-Fi 热点列表为嵌入表格图。"},
    23: {"layout": "body-text", "note": "纸张与特殊介质说明为正文。"},
    24: {"layout": "table+procedure-diagram", "note": "介质规则表和装纸示意图清晰。"},
    25: {
        "layout": "procedure-diagram+note",
        "note": "展开纸张、250张上限与超量后卡纸/不进纸提示位于相邻页。",
    },
    28: {
        "layout": "diagram+table",
        "note": "非打印区域以阴影明确标识，并列出上下/左右边距。",
    },
    30: {"layout": "installer-screenshot+procedure", "note": "USB/有线安装流程和安装器界面。"},
    32: {"layout": "installer-screenshot+procedure", "note": "新无线网络安装流程和安装器界面。"},
    33: {"layout": "installer-screenshot+procedure", "note": "Wi-Fi 热点安装流程和安装器界面。"},
    40: {"layout": "body-text", "note": "IPv6 地址类型说明为正文，但未直接绑定到两种无线连接模式。"},
    41: {"layout": "network-diagram+body-text", "note": "基础结构与 Wi-Fi 热点连接拓扑图清晰。"},
    43: {"layout": "installer-screenshot+body-text", "note": "WPA/WPA2 密钥长度及显示密码行为清晰。"},
    44: {"layout": "procedure+icons", "note": "WPS 和 Wi-Fi 热点连接步骤清晰。"},
    48: {"layout": "body-text+web-example", "note": "内嵌 Web 服务器访问说明含 IP 地址示例。"},
    58: {"layout": "control-panel-diagram", "note": "省墨菜单操作以控制面板示意图呈现。"},
    59: {"layout": "control-panel+ui-screenshot", "note": "省墨开关和打印首选项界面清晰。"},
    60: {"layout": "control-panel-diagram", "note": "恢复出厂设置菜单操作清晰。"},
    66: {"layout": "tray-diagram", "note": "多功能纸盒和选配纸盒安装示意。"},
    67: {
        "layout": "tray-diagram+ui-screenshot",
        "note": "选配纸盒通信失败提示及进纸选择说明位于相邻页。",
    },
    76: {"layout": "toner-replacement-diagrams", "note": "碳粉盒更换步骤以连续图示呈现。"},
    79: {"layout": "jam-removal-diagrams", "note": "标准纸盒卡纸处理图示。"},
    80: {"layout": "jam-removal-diagrams", "note": "多功能纸盒卡纸处理图示。"},
    81: {"layout": "jam-removal-diagrams", "note": "选配纸盒卡纸处理图示。"},
    82: {"layout": "jam-removal-diagrams+warning", "note": "中部卡纸及感光鼓遮盖提示。"},
    83: {"layout": "jam-removal-diagrams+warning", "note": "中部卡纸后续操作及高温区域警示。"},
    86: {"layout": "image-defect-table", "note": "图像缺陷表从本页开始，跨相邻页连续排版。"},
    87: {
        "layout": "image-defect-table",
        "note": "周期性痕迹、页面歪斜等原因和解决办法清晰。",
    },
    88: {"layout": "image-defect-table", "note": "打印全黑版等原因和解决办法清晰。"},
    90: {"layout": "menu-hierarchy-table", "note": "菜单结构表从本页开始。"},
    91: {
        "layout": "menu-hierarchy-table",
        "note": "网络信息三级菜单位于相邻页；说明合并 chunk 的 page 字段可指向起始页。",
    },
}


REFERENCE_FINDINGS: dict[str, dict[str, object]] = {
    "P3500-EASY-004": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [8],
        "finding": "问题问哪一种物质，表格仅铅(Pb)在部分部件为×；旧参考列出全部六种物质。",
        "recommended_action": "将答案合同修为铅(Pb)，并明确超限部件为金属、电线电缆、电路板。",
    },
    "P3500-EASY-049": {
        "reference_disposition": "not_explicit_in_source",
        "visual_pages_checked": [90, 91],
        "finding": "菜单表存在横向分隔线，但源页没有‘四条水平横线代表文本行’的图例或定义。",
        "recommended_action": "人工重写为源文档可直接判定的问题，或移出确定性评分集。",
    },
    "P3500-HARD-008": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [86, 87, 88],
        "finding": "周期性痕迹的源解决办法是清洁/更换硒鼓或更换定影组件；旧参考声称与页面歪斜共享清洁进纸通道/调整介质。",
        "recommended_action": "分开建立两类故障的 all-of 概念，删除无源的重叠结论。",
    },
    "P3500-HARD-015": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [28, 59],
        "finding": "源文明确‘阴影部分表示非打印区域’，旧参考称白色空白区域；省墨分支有明确支持。",
        "recommended_action": "将第一分支修为图中阴影边缘区域，保留省墨分支。",
    },
    "P3500-HARD-021": {
        "reference_disposition": "source_scope_not_explicit",
        "visual_pages_checked": [40, 41],
        "finding": "源分别描述 IPv6 地址类型和两种无线拓扑，但没有把 DHCPv6/路由器地址与基础结构、FE80 与热点模式建立一一合同。",
        "recommended_action": "避免要求无明确源绑定的推断；拆为地址类型题与连接拓扑题。",
    },
    "P3500-HARD-037": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [4],
        "finding": "激光标签可见文字为 INVISIBLE LASER RADIATION，旧参考误写为 MAILABLE LASER RADIATION。",
        "recommended_action": "按标签原文修正标识符；评测合同要求关键标识符全保留。",
    },
    "P3500-HARD-040": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [7, 8],
        "finding": "第二问要求列出超限部件；表格答案为金属、电线电缆、电路板（均为铅），旧参考只泛述‘部分部件’。",
        "recommended_action": "补全第二分支的部件与物质绑定。",
    },
    "P3500-HARD-041": {
        "reference_disposition": "confirmed_source_conflict",
        "visual_pages_checked": [14, 18],
        "finding": "后视图编号表明确标签1为 USB 接口，不是左侧后部竖向通风口；硒鼓100%分支有支持。",
        "recommended_action": "删除错误图号绑定，或改问标签1对应的 USB 接口。",
    },
    "P3500-HARD-042": {
        "reference_disposition": "reference_does_not_answer_query",
        "visual_pages_checked": [18, 19],
        "finding": "源支持月负荷和 IP 数值，但问题要求判断是否仍在可接受工作负载；旧参考只复述数字和IP，没有给出判断。",
        "recommended_action": "参考答案需明确：接近4,000页仍在推荐范围内且远低于50,000页最大月负荷；IP与负荷判断无关。",
    },
}


ADJACENT_PAGE_FINDINGS: dict[str, list[int]] = {
    "P3500-HARD-012": [25],
    "P3500-HARD-032": [67],
    "P3500-EASY-046": [87],
    "P3500-MEDIUM-045": [88],
    "P3500-HARD-008": [87, 88],
    "P3500-MEDIUM-047": [91],
}


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest_cases() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for difficulty in ("easy", "medium", "hard"):
        payload = _json(DATASET_DIR / f"{difficulty}.json")
        assert isinstance(payload, dict)
        for case in payload["cases"]:
            result[case["case_id"]] = case
    assert len(result) == 150
    return result


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    qa_cases = _json(QA_DIR / "cases.json")
    assert isinstance(qa_cases, list) and len(qa_cases) == 150
    manifests = _load_manifest_cases()
    candidates = [row for row in qa_cases if "source_ambiguity" in row["qa_classifications"]]
    assert len(candidates) == 39
    assert sum("PDF_TEXT_EXTRACTION_TERM_GAP" in row["issue_codes"] for row in candidates) == 37

    output: list[dict[str, object]] = []
    for row in candidates:
        case_id = row["case_id"]
        manifest = manifests[case_id]
        declared_pages = sorted({int(evidence["page"]) for evidence in manifest["relevant_evidence"]})
        adjacent_pages = ADJACENT_PAGE_FINDINGS.get(case_id, [])
        finding = REFERENCE_FINDINGS.get(case_id)
        checked_pages = (
            list(finding["visual_pages_checked"])
            if finding
            else sorted(set(declared_pages + adjacent_pages))
        )
        missing_page_notes = [page for page in checked_pages if page not in PAGE_NOTES]
        assert not missing_page_notes, (case_id, missing_page_notes)
        output.append(
            {
                "case_id": case_id,
                "difficulty": row["difficulty"],
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "issue_codes": row["issue_codes"],
                "declared_evidence_pages": declared_pages,
                "visual_pages_checked": checked_pages,
                "adjacent_page_material_required": bool(adjacent_pages),
                "visual_source_disposition": "visual_material_present",
                "reference_disposition": (
                    finding["reference_disposition"] if finding else "not_fully_adjudicated"
                ),
                "finding": (
                    finding["finding"]
                    if finding
                    else "视觉材料存在；本审计只排除‘源文件缺失’，不把存在性当作参考答案已被完整验证。"
                ),
                "recommended_action": (
                    finding["recommended_action"]
                    if finding
                    else "在 suite v2 概念合同中继续按分支、关键数值/标识符和证据绑定审阅。"
                ),
                "page_observations": [
                    {"page": page, **PAGE_NOTES[page]} for page in checked_pages
                ],
            }
        )

    output.sort(key=lambda item: item["case_id"])
    dispositions = Counter(str(item["reference_disposition"]) for item in output)
    summary: dict[str, object] = {
        "schema_version": "stt.eval.pdf-visual-source-review.v1",
        "suite_key": "three-difficulty-live-2026-08-10",
        "source": {"path": SOURCE_RELATIVE_PATH, "sha256": SOURCE_SHA256, "page_count": 95},
        "review_scope": {
            "source_ambiguity_candidate_count": len(output),
            "pdf_text_extraction_gap_count": sum(
                "PDF_TEXT_EXTRACTION_TERM_GAP" in item["issue_codes"] for item in output
            ),
            "unique_declared_page_count": len(
                {page for item in output for page in item["declared_evidence_pages"]}
            ),
            "visual_material_present_count": sum(
                item["visual_source_disposition"] == "visual_material_present" for item in output
            ),
            "adjacent_page_material_case_count": sum(
                bool(item["adjacent_page_material_required"]) for item in output
            ),
        },
        "reference_dispositions": dict(sorted(dispositions.items())),
        "blocking_reference_cases": [
            item["case_id"]
            for item in output
            if item["reference_disposition"]
            in {
                "confirmed_source_conflict",
                "not_explicit_in_source",
                "source_scope_not_explicit",
                "reference_does_not_answer_query",
            }
        ],
        "conclusion": {
            "source_missing_claim": False,
            "layout_extraction_note": "37项文本提取缺口均有对应视觉材料；缺口不能再被解释为PDF源文件不存在。",
            "reference_quality_note": "视觉材料存在不代表旧参考正确；已确认的冲突/不显式/不作答题必须在suite v2修订后才能进入新基线。",
            "page_binding_note": "至少6题的关键信息位于声明页的相邻页，反映合并chunk以起始页归属，预检不应把它误报为源缺失。",
        },
    }
    return output, summary


def _write_csv(rows: list[dict[str, object]]) -> None:
    columns = [
        "case_id",
        "difficulty",
        "declared_evidence_pages",
        "visual_pages_checked",
        "adjacent_page_material_required",
        "visual_source_disposition",
        "reference_disposition",
        "finding",
        "recommended_action",
    ]
    with (THIS_DIR / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], list)
                    else row[key]
                    for key in columns
                }
            )


def _write_report(summary: dict[str, object]) -> None:
    counts = summary["reference_dispositions"]
    blockers = summary["blocking_reference_cases"]
    text = f"""# PDF 视觉源证据审计

## 结论

39 条 `source_ambiguity` 候选的对应视觉材料全部存在，因此不能把这些题笼统归因为“PDF 源文件缺失”。其中 37 条文本抽取缺口主要落在嵌入截图、表格、流程图或跨页合并 chunk；同时，视觉复核确认旧参考合同本身存在需要阻断新基线的问题。

## 审计范围

- 来源：`{SOURCE_RELATIVE_PATH}`（SHA256 `{SOURCE_SHA256}`，95页）
- source ambiguity 候选：39/150
- PDF text extraction gap：37/150
- 视觉材料存在：39/39
- 关键材料落在相邻页：{summary['review_scope']['adjacent_page_material_case_count']}/39

## 参考合同处置

- confirmed_source_conflict：{counts.get('confirmed_source_conflict', 0)}
- not_explicit_in_source：{counts.get('not_explicit_in_source', 0)}
- source_scope_not_explicit：{counts.get('source_scope_not_explicit', 0)}
- reference_does_not_answer_query：{counts.get('reference_does_not_answer_query', 0)}
- 尚未在本视觉步骤完整裁定：{counts.get('not_fully_adjudicated', 0)}

需阻断旧参考直接沿用的 case：{', '.join(blockers)}。

## 方法与限制

使用 Poppler 将候选声明页以 110 DPI 渲染为 PNG，并对表格、截图、图号、警示标签、菜单结构和相邻页连续内容进行人工视觉复核。`cases.json` 逐题保存声明页、实际检查页、版式类型、结论和建议。此步骤只裁定可见源材料与明确冲突；未标为冲突的题仍需 suite v2 概念合同审阅，不能把 `visual_material_present` 解读为参考答案已被完整批准。

渲染 PNG 位于工作区临时目录，不作为权威基线制品；权威来源仍是原 PDF 及其 SHA256。
"""
    (THIS_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows, summary = build()
    (THIS_DIR / "cases.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (THIS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(rows)
    _write_report(summary)
    files = ["REPORT.md", "audit_visual_sources.py", "cases.csv", "cases.json", "summary.json"]
    checksums = "".join(f"{_sha256(THIS_DIR / name)}  {name}\n" for name in files)
    (THIS_DIR / "SHA256SUMS").write_text(checksums, encoding="utf-8")


if __name__ == "__main__":
    main()
