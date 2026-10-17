#!/usr/bin/env python3
"""Deterministic diagnosis of the archived pre-grounding blind-review errors.

This script is intentionally read-only with respect to application code and the
fixed datasets.  It joins the blind semantic-review verdicts to archived query
observations and dataset evidence, derives stage-presence facts, then applies an
explicit human-reviewed adjudication table to the 30 blind-incorrect cases.

The blind semantic verdict is never inferred from production pass/fail fields.
The script does not read credentials, call a network service, or invoke an LLM.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_DIR = Path(__file__).resolve().parent
REMEDIATION_ROOT = OUTPUT_DIR.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
REPO_ROOT = SUITE_ROOT.parents[2]

RUN_LABEL = "live-cand-max-mono-01"
REVIEW_ITERATION = "direct-answer-v3-qwen3.7-max"
REVIEW_DIR = (
    REMEDIATION_ROOT
    / "semantic-reviews"
    / RUN_LABEL
    / REVIEW_ITERATION
    / "pre_grounding"
)
REVIEW_SUMMARY_PATH = REVIEW_DIR / "summary.json"
REVIEW_RECORDS_PATH = REVIEW_DIR / "semantic-correctness-review.jsonl"
DIFFICULTIES = ("easy", "medium", "hard")

CATEGORY_LABELS = {
    "retrieval_content_insufficient": "召回内容不足",
    "retrieval_content_unclear_or_noisy": "召回内容噪声/不清晰",
    "context_loss": "上下文丢失",
    "intermediate_process_loss": "流程中间内容丢失",
    "generation_reasoning": "生成/推理",
    "evaluation_system_error_candidate": "评测系统错误候选",
    "dataset_reference": "dataset/reference",
    "source_ambiguity": "source ambiguity",
    "freshness_execution": "freshness/execution",
    "other": "other",
}

PATTERN_FAMILIES: dict[str, dict[str, Any]] = {
    "reference_conflicts_with_or_exceeds_source": {
        "label": "reference与源证据冲突或要求无证据推断",
        "case_ids": (
            "P3500-EASY-004",
            "P3500-HARD-007",
            "P3500-HARD-009",
            "P3500-HARD-021",
            "P3500-HARD-027",
            "P3500-HARD-044",
        ),
    },
    "underspecified_or_scope_mixed_question": {
        "label": "问句条件不足、范围错位或混合不同章节",
        "case_ids": (
            "P3500-EASY-038",
            "P3500-HARD-001",
            "P3500-HARD-004",
            "P3500-HARD-010",
            "P3500-HARD-011",
        ),
    },
    "reference_does_not_answer_requested_output": {
        "label": "reference未回答所要求的实体清单或判断",
        "case_ids": ("P3500-HARD-040", "P3500-HARD-042"),
    },
    "generation_completeness_omission": {
        "label": "生成答案在完整context下漏步骤或字段",
        "case_ids": (
            "P3500-MEDIUM-014",
            "P3500-MEDIUM-021",
            "P3500-MEDIUM-048",
        ),
    },
    "generation_type_or_identifier_binding": {
        "label": "生成答案混淆字段类型或未原样保留标识符",
        "case_ids": ("P3500-MEDIUM-031", "P3500-HARD-037"),
    },
    "generation_overconservative_refusal": {
        "label": "生成答案拒绝证据已支持的低风险归纳",
        "case_ids": ("P3500-HARD-024",),
    },
    "relevant_branch_lost_before_rerank_or_context": {
        "label": "已召回的关键分支未进入rerank/context",
        "case_ids": (
            "P3500-HARD-003",
            "P3500-HARD-006",
            "P3500-HARD-013",
            "P3500-HARD-045",
        ),
    },
    "top_reranked_chunk_dropped_before_context": {
        "label": "reranked首位相关块在context构造时丢失",
        "case_ids": ("P3500-MEDIUM-015", "P3500-HARD-019"),
    },
    "flattened_table_structure_ambiguous": {
        "label": "扁平化表格行列边界或标签值绑定不清",
        "case_ids": ("P3500-EASY-047", "P3500-HARD-008"),
    },
    "visual_control_missing_from_extraction": {
        "label": "视觉按钮标识未进入文本抽取",
        "case_ids": ("P3500-EASY-031",),
    },
    "conflicting_values_across_source_sections": {
        "label": "源文档不同章节存在冲突数值",
        "case_ids": ("P3500-MEDIUM-003",),
    },
    "freshness_misroute_and_missing_fact": {
        "label": "静态快照误走freshness且第二事实未召回",
        "case_ids": ("P3500-HARD-041",),
    },
}
PATTERN_FAMILY_BY_CASE = {
    case_id: family
    for family, details in PATTERN_FAMILIES.items()
    for case_id in details["case_ids"]
}

# Explicit human-reviewed adjudications.  The entries describe generic failure
# modes and contain no proposed case-specific application fixes.
ADJUDICATIONS: dict[str, dict[str, Any]] = {
    "P3500-EASY-004": {
        "primary": "dataset_reference",
        "secondary": ["evaluation_system_error_candidate", "source_ambiguity"],
        "confidence": 0.98,
        "mechanism": (
            "问题使用单数“哪一种”，数据行的×只落在铅(Pb)列；reference却枚举了六个表头物质。"
            "pre回答依据数据行给出Pb，盲审失败主要由题目/reference基数与表格事实不一致造成。"
        ),
        "evidence": [
            "相关表格块在retrieved/reranked/context均为第1位，pack无省略或截断。",
            "pre回答明确给出Pb及三个含×的部件；reference列出Pb、Hg、Cd、Cr(VI)、PBB、PBDE。",
        ],
        "pattern": "reference_enumerates_headers_instead_of_answer_rows",
        "self_review": {
            "would_trigger": True,
            "trigger": "单数问句、表头枚举与数据行×标记的基数冲突检查。",
            "coverage_likelihood": "unlikely",
            "rationale": "不读金标的表格自审会继续确认Pb，而不会诚实地改成有缺陷的六项reference。",
        },
    },
    "P3500-EASY-031": {
        "primary": "source_ambiguity",
        "secondary": ["dataset_reference", "evaluation_system_error_candidate"],
        "confidence": 0.99,
        "mechanism": (
            "索引文本把按钮图标抽取成空白，源证据与reference都只剩“点击 按钮”。"
            "pre回答已如实说明原文未给出按钮名称，因此在现有文本上下文中不存在可恢复的按钮标识。"
        ),
        "evidence": [
            "相关块全程第1位且pack完整，排除召回和上下文丢失。",
            "固定证据原文为“2.点击 按钮”，reference同样没有按钮名称。",
        ],
        "pattern": "visual_control_label_missing_from_extraction",
        "self_review": {
            "would_trigger": True,
            "trigger": "答案显式报告关键UI控件名称缺失。",
            "coverage_likelihood": "unlikely",
            "rationale": "纯文本自审无法从空白处恢复图标；需要通用视觉/OCR补全或修订题集。",
        },
    },
    "P3500-EASY-038": {
        "primary": "dataset_reference",
        "secondary": ["generation_reasoning", "evaluation_system_error_candidate"],
        "confidence": 0.90,
        "mechanism": (
            "同一流程连续三次按OK，分别进入纸盒设置界面、设置界面和纸张类型界面；问句未限定是第几次。"
            "pre首句选择第一次，但正文也完整包含reference所指的第二次，失败由问句歧义与直答聚焦共同造成。"
        ),
        "evidence": [
            "唯一相关块在三阶段均第1位，包含三次OK及三个不同界面。",
            "pre首句为“纸盒设置界面”，第三条明确写出再次按OK进入“设置界面”。",
        ],
        "pattern": "ambiguous_repeated_action_reference_selects_one_step",
        "self_review": {
            "would_trigger": True,
            "trigger": "同一动作对应多个阶段，首句与后续步骤出现多个候选直答。",
            "coverage_likelihood": "likely",
            "rationale": "无需金标即可改为条件化说明三次OK，并把“选定纸盒后的第二次OK”直答为设置界面。",
        },
    },
    "P3500-EASY-047": {
        "primary": "source_ambiguity",
        "secondary": ["dataset_reference", "evaluation_system_error_candidate"],
        "confidence": 0.85,
        "mechanism": (
            "扁平化故障表在黑色竖条行后同时出现背部进纸通道和机器内部进纸通道，行/列边界不清。"
            "pre保留三项清洁动作，而reference只保留前两项。"
        ),
        "evidence": [
            "相关表格块位于context第2位且pack完整。",
            "同一抽取块的解决办法文本连续包含三条清洁动作，缺少可靠的结构边界。",
        ],
        "pattern": "flattened_table_row_boundary_ambiguous",
        "self_review": {
            "would_trigger": True,
            "trigger": "表格行中出现近义重复位置与项目边界不清。",
            "coverage_likelihood": "possible",
            "rationale": "结构感知自审可能区分主行与相邻项目，但仅凭当前扁平文本不能可靠删除第三项。",
        },
    },
    "P3500-MEDIUM-003": {
        "primary": "source_ambiguity",
        "secondary": ["retrieval_content_unclear_or_noisy", "dataset_reference"],
        "confidence": 0.95,
        "mechanism": (
            "演示页写50,000 page monthly duty cycle，而同一手册规格表写最大月打印量80,000页。"
            "两条均进入生成上下文，pre将“最大月工作负荷”解释为规格表80,000，reference固定为演示页50,000。"
        ),
        "evidence": [
            "reference相关演示页在context第1位；冲突规格表在context第5位。",
            "pre同时正确保留750–4,000推荐量，并显式选择80,000作为最大月打印量。",
        ],
        "pattern": "conflicting_values_across_source_sections",
        "self_review": {
            "would_trigger": True,
            "trigger": "同一指标出现50,000与80,000两个数值。",
            "coverage_likelihood": "possible",
            "rationale": "无金标自审可并列说明营销页duty cycle与规格表最大月打印量，但不能武断选定一项。",
        },
    },
    "P3500-MEDIUM-014": {
        "primary": "generation_reasoning",
        "secondary": [],
        "confidence": 0.98,
        "mechanism": (
            "完整安装流程已在首个上下文块中，但pre只保留界面中后段和部分前置条件，"
            "漏掉开机、插入光盘、运行Autorun.exe、语言和具体型号等关键步骤。"
        ),
        "evidence": [
            "相关块在retrieved/reranked/context均第1位，pack无损。",
            "pre未覆盖reference中的Autorun.exe、中文(简体)和Pantum P3500DW Series PCL6。",
        ],
        "pattern": "generation_omits_required_procedure_steps",
        "self_review": {
            "would_trigger": True,
            "trigger": "流程编号覆盖和关键实体守恒检查。",
            "coverage_likelihood": "likely",
            "rationale": "完整编号流程已在上下文中，通用步骤覆盖自审可发现缺项。",
        },
    },
    "P3500-MEDIUM-015": {
        "primary": "intermediate_process_loss",
        "secondary": ["retrieval_content_unclear_or_noisy"],
        "confidence": 0.99,
        "mechanism": (
            "Mac安装相关块已被召回并进入reranked第1位，但构造生成context时被四个Windows安装块替换。"
            "evidence assessor仍报SUFFICIENT，导致pre错误地断言没有Mac步骤。"
        ),
        "evidence": [
            "固定相关chunk在retrieved第1位、reranked第1位、context中缺失。",
            "context的4个块均为Windows流程；pack对这4块无截断，说明损失发生在pack之前。",
        ],
        "pattern": "reranked_relevant_chunk_dropped_before_context",
        "self_review": {
            "would_trigger": True,
            "trigger": "查询要求Mac而生成上下文全部为Windows，域一致性检查失败。",
            "coverage_likelihood": "unlikely",
            "rationale": "答案阶段自审看不到已丢失的Mac块；需要先修复rerank→context守恒。",
        },
    },
    "P3500-MEDIUM-021": {
        "primary": "generation_reasoning",
        "secondary": [],
        "confidence": 0.98,
        "mechanism": "源证据完整列出四步，pre按序输出前三步但漏掉“开启计算机或移动设备”。",
        "evidence": [
            "相关块在context第2位且明确含步骤4。",
            "pack无省略/截断；pre停在步骤3后转为注意事项。",
        ],
        "pattern": "generation_omits_final_procedure_step",
        "self_review": {
            "would_trigger": True,
            "trigger": "源流程编号1–4与答案编号1–3不守恒。",
            "coverage_likelihood": "likely",
            "rationale": "通用编号完整性自审可直接发现并补回已见步骤4。",
        },
    },
    "P3500-MEDIUM-031": {
        "primary": "generation_reasoning",
        "secondary": [],
        "confidence": 0.95,
        "mechanism": (
            "pre完整列出两个纸盒及交集，但把配置哨兵“自动*”当成纸张尺寸加入交集；"
            "reference只列实际尺寸。"
        ),
        "evidence": [
            "相关表格在context第2位且pack完整。",
            "pre与reference的实质差异是额外的“自动*”。",
        ],
        "pattern": "generation_treats_configuration_sentinel_as_dimension_value",
        "self_review": {
            "would_trigger": True,
            "trigger": "枚举值类型检查：自动是配置模式，不是纸张尺寸。",
            "coverage_likelihood": "likely",
            "rationale": "无需金标即可按字段类型剔除非尺寸哨兵。",
        },
    },
    "P3500-MEDIUM-048": {
        "primary": "generation_reasoning",
        "secondary": ["retrieval_content_unclear_or_noisy", "source_ambiguity"],
        "confidence": 0.98,
        "mechanism": (
            "首个规格上下文完整包含温度、电压、频率和电流，pre却遗漏50HZ/60HZ(±2%)，"
            "并额外引入安装注意事项中的10–32.5°C，削弱直接性。"
        ),
        "evidence": [
            "相关规格块在三阶段均第1位，pack无损。",
            "pre保留5–35°C、AC220–240V和4A，但未输出频率字段。",
        ],
        "pattern": "generation_omits_one_table_field_and_adds_nearby_noise",
        "self_review": {
            "would_trigger": True,
            "trigger": "复合问题字段清单与答案字段守恒检查。",
            "coverage_likelihood": "likely",
            "rationale": "频率字段已在同一表格行中，可由通用表格字段自审补齐。",
        },
    },
    "P3500-HARD-001": {
        "primary": "dataset_reference",
        "secondary": ["generation_reasoning", "evaluation_system_error_candidate"],
        "confidence": 0.93,
        "mechanism": (
            "问句比较出纸托盘与多功能纸盒导纸板，reference却引入未被问及的标准纸盒宽度导纸板。"
            "pre按字面正确说明出纸托盘不调宽、多功能导纸板调两侧宽度。"
        ),
        "evidence": [
            "首个产品视图块同时包含三种部件定义，pack完整。",
            "reference第三部分回答标准纸盒宽度导纸板，而不是问句中的出纸托盘。",
        ],
        "pattern": "reference_answers_unasked_third_component",
        "self_review": {
            "would_trigger": True,
            "trigger": "问题实体与reference/候选实体数量不一致。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审会坚持比较问句中的两个部件，不应添加未问部件。",
        },
    },
    "P3500-HARD-003": {
        "primary": "intermediate_process_loss",
        "secondary": ["dataset_reference"],
        "confidence": 0.95,
        "mechanism": (
            "通用故障前置检查块已在retrieved第1位，但reranked/context只保留了卡纸操作及装纸噪声。"
            "pre因此断言不存在前置条件；reference中的未检查后果本身也只是概括性推断。"
        ),
        "evidence": [
            "两个标注相关块在retrieved第1、17位；context仅保留第17位的卡纸步骤块。",
            "pack无省略/截断，损失发生于rerank/选择阶段。",
        ],
        "pattern": "first_relevant_chunk_lost_during_rerank",
        "self_review": {
            "would_trigger": True,
            "trigger": "问题要求前置检查，而上下文只含事后处理，意图覆盖不完整。",
            "coverage_likelihood": "unlikely",
            "rationale": "答案自审无法重建已被reranker丢弃的检查清单。",
        },
    },
    "P3500-HARD-004": {
        "primary": "dataset_reference",
        "secondary": ["source_ambiguity", "evaluation_system_error_candidate"],
        "confidence": 0.98,
        "mechanism": (
            "额外的“看不到纸张→打开标准纸盒”步骤来自选配纸盒卡纸章节，不是多功能纸盒章节。"
            "题目/reference把两个不同章节拼接成同一流程；pre已给出步骤但诚实标注章节不匹配。"
        ),
        "evidence": [
            "多功能纸盒卡纸与选配纸盒卡纸两个块均进入context。",
            "固定证据标题清楚区分10.1.2与10.1.3，reference未保留这个适用范围差异。",
        ],
        "pattern": "query_combines_steps_from_different_procedure_scope",
        "self_review": {
            "would_trigger": True,
            "trigger": "跨章节步骤的适用对象不一致。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审应保留适用范围警告，而非把选配纸盒步骤伪装成多功能步骤。",
        },
    },
    "P3500-HARD-006": {
        "primary": "intermediate_process_loss",
        "secondary": ["retrieval_content_unclear_or_noisy"],
        "confidence": 0.98,
        "mechanism": (
            "双面卡纸与软件故障两个相关块均已召回，但软件故障块未进入reranked/context，"
            "pre只能回答第一分支并拒绝第二分支。"
        ),
        "evidence": [
            "相关块在retrieved第1、16位；reranked/context只命中第1位双面卡纸块。",
            "evidence assessor仍将两个子问题都标为覆盖，形成错误的SUFFICIENT。",
        ],
        "pattern": "second_branch_relevant_chunk_lost_during_rerank",
        "self_review": {
            "would_trigger": True,
            "trigger": "复合问题第二分支在答案中明确标为证据不足。",
            "coverage_likelihood": "unlikely",
            "rationale": "需要重新引入已召回的软件故障块，纯答案自审不能补出措施。",
        },
    },
    "P3500-HARD-007": {
        "primary": "dataset_reference",
        "secondary": ["source_ambiguity", "evaluation_system_error_candidate"],
        "confidence": 0.98,
        "mechanism": (
            "源表明确的原因共同点是“内部部件故障”；打印多页进纸的解决办法并不含清洁。"
            "reference却称共同点是介质状态且两者都包含介质处理和清洁，pre更贴近表格事实。"
        ),
        "evidence": [
            "相关故障表在context第1位且pack完整。",
            "多页进纸解决办法只含分离介质、推荐介质和干燥介质，没有清洁动作。",
        ],
        "pattern": "reference_claims_nonexistent_solution_overlap",
        "self_review": {
            "would_trigger": True,
            "trigger": "两行原因/解决办法集合交集计算。",
            "coverage_likelihood": "unlikely",
            "rationale": "集合自审会确认pre的内部部件共同点，不会生成源中不存在的清洁交集。",
        },
    },
    "P3500-HARD-008": {
        "primary": "source_ambiguity",
        "secondary": ["generation_reasoning"],
        "confidence": 0.85,
        "mechanism": (
            "扁平表格把故障名称与下一组原因/办法错位呈现，pre将“周期性痕迹”绑定到前一组碳粉盒/定影组件办法，"
            "而reference按后一组介质安装/进纸通道解释。"
        ),
        "evidence": [
            "唯一context块包含所有字段但有连续空单元格和错位行。",
            "无召回或pack丢失；错误发生在结构不清条件下的行绑定。",
        ],
        "pattern": "flattened_table_label_value_alignment_ambiguous",
        "self_review": {
            "would_trigger": True,
            "trigger": "故障标签与相邻原因/办法的结构对齐置信度低。",
            "coverage_likelihood": "possible",
            "rationale": "结构化重建可能修复绑定，但纯线性文本自审无法稳定判定行边界。",
        },
    },
    "P3500-HARD-009": {
        "primary": "dataset_reference",
        "secondary": ["source_ambiguity", "evaluation_system_error_candidate"],
        "confidence": 0.95,
        "mechanism": (
            "打印全黑版的办法是正确安装或更换碳粉盒，黑色竖条的办法是清洁反光玻璃/进纸通道并规范介质。"
            "reference声称两者共同涉及清洁和介质规范，与源表不符。"
        ),
        "evidence": [
            "两个关键表格块均进入context且pack完整。",
            "pre给出原因和清洁项，并把可能共通点解释为碳粉盒处理；reference的共通点没有源支持。",
        ],
        "pattern": "reference_invents_cross_row_overlap",
        "self_review": {
            "would_trigger": True,
            "trigger": "两故障解决办法集合交集检查。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审不会把仅一侧出现的清洁/介质规范说成共同点。",
        },
    },
    "P3500-HARD-010": {
        "primary": "dataset_reference",
        "secondary": ["evaluation_system_error_candidate"],
        "confidence": 0.98,
        "mechanism": (
            "纸张类型位于每一种纸盒二级菜单下，问句未指定纸盒；pre列出多功能、标准、选配一和选配二，"
            "reference却只给多功能纸盒。"
        ),
        "evidence": [
            "两个菜单结构相关块都进入context。",
            "pre的层级说明与表头一级/二级/三级结构一致。",
        ],
        "pattern": "reference_arbitrarily_selects_one_of_multiple_valid_menu_branches",
        "self_review": {
            "would_trigger": True,
            "trigger": "未指定目标纸盒但存在多个同构二级菜单。",
            "coverage_likelihood": "unlikely",
            "rationale": "通用自审应列出所有有效分支或请求限定，而不是猜测多功能纸盒。",
        },
    },
    "P3500-HARD-011": {
        "primary": "dataset_reference",
        "secondary": ["source_ambiguity", "evaluation_system_error_candidate"],
        "confidence": 0.95,
        "mechanism": (
            "标准与多功能纸盒都支持A4、80g/m²普通纸；题目未说明页数或大批量。"
            "reference以250页对60页推导应选标准纸盒，pre则依据另一个源规则指出装有介质时多功能纸盒优先。"
        ),
        "evidence": [
            "两个规格块和打印优先级块均进入context。",
            "两种纸盒均满足尺寸与克重约束，选择取决于未给出的数量/装纸状态。",
        ],
        "pattern": "underspecified_choice_with_multiple_valid_options",
        "self_review": {
            "would_trigger": True,
            "trigger": "两个候选都满足显式约束，缺少决定性条件。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审应条件化回答，而不能假定“大批量”。",
        },
    },
    "P3500-HARD-013": {
        "primary": "intermediate_process_loss",
        "secondary": ["retrieval_content_unclear_or_noisy"],
        "confidence": 0.98,
        "mechanism": (
            "标准与多功能纸盒两个相关块均已召回，但多功能纸盒的完整放置说明未进入reranked/context；"
            "context只剩较窄的“已打印纸空白面朝上”说明，pre因此拒绝新纸放置方向。"
        ),
        "evidence": [
            "多功能完整说明相关chunk在retrieved第19位，reranked/context缺失。",
            "pack无截断；pre准确暴露了缺失分支。",
        ],
        "pattern": "second_source_lost_before_context_related_snippet_survives",
        "self_review": {
            "would_trigger": True,
            "trigger": "问题明确要求两份证据，答案只完整覆盖标准纸盒。",
            "coverage_likelihood": "unlikely",
            "rationale": "需恢复完整多功能纸盒块，答案自审本身没有足够事实。",
        },
    },
    "P3500-HARD-019": {
        "primary": "intermediate_process_loss",
        "secondary": ["retrieval_content_unclear_or_noisy"],
        "confidence": 0.99,
        "mechanism": (
            "Mac驱动安装块在retrieved和reranked均第1位，但未进入最终context，后者被通用无线配置块占据。"
            "pre因此错误声称提示窗口处理证据不存在。"
        ),
        "evidence": [
            "相关Mac安装chunk：retrieved第1位、reranked第1位、context缺失。",
            "context pack对现有4块无截断，证明损失在reranked→context边界。",
        ],
        "pattern": "top_reranked_chunk_dropped_during_context_construction",
        "self_review": {
            "would_trigger": True,
            "trigger": "查询指定驱动安装后的提示窗口，context没有安装流程。",
            "coverage_likelihood": "unlikely",
            "rationale": "需要修复context构建守恒，纯自审无法恢复“是/否”规则。",
        },
    },
    "P3500-HARD-021": {
        "primary": "dataset_reference",
        "secondary": [
            "retrieval_content_insufficient",
            "source_ambiguity",
            "evaluation_system_error_candidate",
        ],
        "confidence": 0.92,
        "mechanism": (
            "源文档只定义三类通用IPv6地址，没有把DHCPv6/路由器地址专属于基础结构、"
            "也没有把FE80专属于热点模式；reference作了未被证据支持的模式映射。"
        ),
        "evidence": [
            "连接拓扑相关块进入context，但IPv6地址类型块未进入retrieved/context。",
            "即便补回IPv6块，源文本也没有reference要求的两模式专属关系。",
        ],
        "pattern": "reference_requires_unsupported_cross_document_inference",
        "self_review": {
            "would_trigger": True,
            "trigger": "模式到地址分配的因果映射缺少显式桥接证据。",
            "coverage_likelihood": "unlikely",
            "rationale": "诚实自审应拒绝该映射；不能为匹配reference而补造关系。",
        },
    },
    "P3500-HARD-024": {
        "primary": "generation_reasoning",
        "secondary": ["source_ambiguity"],
        "confidence": 0.90,
        "mechanism": (
            "context同时给出热点直连步骤和WPS需路由器支持的前提，足以概括适用场景；"
            "pre只复述操作条件并过度保守地拒绝场景归纳。"
        ),
        "evidence": [
            "两个固定相关块都在context中，另有热点直连和WPS支持说明。",
            "pre已说出WPS前提，却没有把计算机/移动设备直接连接热点概括为第一场景。",
        ],
        "pattern": "generation_refuses_supported_low_risk_synthesis",
        "self_review": {
            "would_trigger": True,
            "trigger": "答案包含两个方法前提但未回答“分别适用什么场景”。",
            "coverage_likelihood": "likely",
            "rationale": "意图覆盖自审可在不增加外部事实的情况下完成低风险归纳。",
        },
    },
    "P3500-HARD-027": {
        "primary": "dataset_reference",
        "secondary": ["evaluation_system_error_candidate"],
        "confidence": 0.99,
        "mechanism": (
            "源文档区分“打印失败”和“未检测到任何打印机”：前者检查Wi-Fi连接，后者检查打印机电源/开机。"
            "reference把前者原因错配给后者；pre给出的电源原因与源文档一致。"
        ),
        "evidence": [
            "Mopria使用块与版本要求块都在context前2位。",
            "同一源块明确写“未检测到任何打印机，请检查打印机是否已接通电源，以及打印机是否已打开”。",
        ],
        "pattern": "reference_swaps_two_distinct_failure_conditions",
        "self_review": {
            "would_trigger": True,
            "trigger": "条件句主语严格对齐：打印失败与未检测到打印机。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审会继续选择电源/开机条件，不应改成错误的Wi-Fi条件。",
        },
    },
    "P3500-HARD-037": {
        "primary": "generation_reasoning",
        "secondary": ["source_ambiguity"],
        "confidence": 0.95,
        "mechanism": (
            "两条证据均完整进入context，pre正确回答回收地点，但把标签改写成中文含义，"
            "没有按问题要求给出可识别的CAUTION CLASS 3B英文标签。"
        ),
        "evidence": [
            "回收与激光标签相关块分别在context第1、2位。",
            "英文标签原文完整可见；pre只输出概括性中文。",
        ],
        "pattern": "generation_paraphrases_identifier_instead_of_extracting_it",
        "self_review": {
            "would_trigger": True,
            "trigger": "问题问“何种标签”，答案未保留源中的标签标识符。",
            "coverage_likelihood": "likely",
            "rationale": "通用标识符/专名保真自审可从已见文本直接摘录。",
        },
    },
    "P3500-HARD-040": {
        "primary": "dataset_reference",
        "secondary": ["retrieval_content_insufficient", "evaluation_system_error_candidate"],
        "confidence": 0.98,
        "mechanism": (
            "问题要求列出哪些部件超限，但固定相关证据只含表头和备注，没有部件数据行；"
            "reference也没有列部件，只把问题改写成“部分部件含有害物质”。"
        ),
        "evidence": [
            "法规说明与表头/备注块进入context，具体×数据行未进入生成context。",
            "pre完整回答回收做法，并明确说明不能据现有Evidence列出部件。",
        ],
        "pattern": "reference_does_not_answer_requested_table_entity_list",
        "self_review": {
            "would_trigger": True,
            "trigger": "问题请求实体列表，但source/reference都只有泛化陈述。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审应继续暴露证据缺口，不能把泛化陈述当作部件列表。",
        },
    },
    "P3500-HARD-041": {
        "primary": "freshness_execution",
        "secondary": [
            "retrieval_content_insufficient",
            "dataset_reference",
            "source_ambiguity",
        ],
        "confidence": 0.90,
        "mechanism": (
            "静态手册快照问题因“当前运行/理想条件”被判STALE；同时第二个固定相关打印机信息页未被召回。"
            "即使硒鼓100%被恢复，该状态也不足以证明整机运行理想，reference仅重复两个事实。"
        ),
        "evidence": [
            "evidence assessment状态为STALE；标签1相关块进入context，硒鼓100%相关块未进入retrieved/context。",
            "reference没有回答“是否处于理想条件”的判断标准。",
        ],
        "pattern": "static_snapshot_misrouted_as_freshness_and_second_fact_missing",
        "self_review": {
            "would_trigger": True,
            "trigger": "静态文档任务被freshness阻断且复合问题缺少第二事实。",
            "coverage_likelihood": "unlikely",
            "rationale": "答案自审无法恢复未召回快照，也不能从100%硒鼓推导整机理想。",
        },
    },
    "P3500-HARD-042": {
        "primary": "dataset_reference",
        "secondary": ["source_ambiguity", "evaluation_system_error_candidate"],
        "confidence": 0.95,
        "mechanism": (
            "问题要求判断接近推荐上限是否仍可接受，reference却只复述打印量范围和无关IP，未给判断。"
            "pre也复述了全部事实但因缺少评估标准拒绝结论。"
        ),
        "evidence": [
            "两个固定相关块都在context前2位且pack完整。",
            "pre包含50,000、750–4,000和192.168.0.102；失败不是事实缺失，而是题目/reference没有定义判断规则。",
        ],
        "pattern": "reference_repeats_inputs_without_answering_decision_question",
        "self_review": {
            "would_trigger": True,
            "trigger": "问句要求判断，reference/source只提供阈值和无关网络字段。",
            "coverage_likelihood": "possible",
            "rationale": "自审可给条件化判断（不超过4,000仍在推荐范围），但这超出reference文本且尚未验证盲审结果。",
        },
    },
    "P3500-HARD-044": {
        "primary": "dataset_reference",
        "secondary": ["evaluation_system_error_candidate"],
        "confidence": 0.99,
        "mechanism": (
            "60页上限明确属于多功能纸盒，选配纸盒证据只说普通纸且不得放特殊介质。"
            "reference把多功能纸盒的60页限制错误移植到选配纸盒；pre选择多功能纸盒并分别说明两者约束。"
        ),
        "evidence": [
            "选配与多功能自动双面打印两个块都在context前2位。",
            "只有多功能块出现“最多只能放置60页”。",
        ],
        "pattern": "reference_attaches_numeric_limit_to_wrong_entity",
        "self_review": {
            "would_trigger": True,
            "trigger": "数字60页的实体绑定检查。",
            "coverage_likelihood": "unlikely",
            "rationale": "无金标自审会确认数字属于多功能纸盒，不应制造错误绑定。",
        },
    },
    "P3500-HARD-045": {
        "primary": "intermediate_process_loss",
        "secondary": ["dataset_reference"],
        "confidence": 0.95,
        "mechanism": (
            "抬起托纸板建议与10.1.6目录块已在retrieved第10、13位，但二者都未进入reranked/context；"
            "pre看到的是相邻章节和详细卡纸操作，因而否认存在精确建议。"
        ),
        "evidence": [
            "两个固定相关chunk均在retrieved命中，reranked/context命中数为0。",
            "pack无省略/截断，损失发生在reranker前后；reference把纸张堆积直接映射到10.1.6也带推断性。",
        ],
        "pattern": "both_required_chunks_lost_during_rerank",
        "self_review": {
            "would_trigger": True,
            "trigger": "查询点名的建议和目录条目均未出现在答案引用上下文。",
            "coverage_likelihood": "unlikely",
            "rationale": "需恢复两个已召回块；答案自审无法可靠补回原句和章节名。",
        },
    },
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def compact(value: Any, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def hit_ranks(items: Iterable[Mapping[str, Any]], relevant: set[str]) -> list[int]:
    return [
        index
        for index, item in enumerate(items, start=1)
        if str(item.get("chunk_id") or "") in relevant
    ]


def unique_chunk_ids(items: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id not in seen:
            result.append(chunk_id)
            seen.add(chunk_id)
    return result


def selected_review_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("phase") != "evaluation":
            continue
        case_id = str(record.get("case_id") or "")
        prior = selected.get(case_id)
        if prior is None or int(record.get("review_index") or 0) > int(
            prior.get("review_index") or 0
        ):
            selected[case_id] = record
    return selected


def source_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def build_case(
    *,
    difficulty: str,
    dataset: Mapping[str, Any],
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    review: Mapping[str, Any],
    dataset_sha256: str,
    results_sha256: str,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    adjudication = ADJUDICATIONS[case_id]
    observation = result.get("observation") or {}
    pre_answer = str(observation.get("pre_grounding_answer") or "")
    relevant_chunk_ids = [str(value) for value in case.get("relevant_chunk_ids") or []]
    relevant = set(relevant_chunk_ids)
    retrieved = observation.get("retrieved") or []
    reranked = observation.get("reranked") or []
    context = observation.get("context") or []
    retrieved_ranks = hit_ranks(retrieved, relevant)
    reranked_ranks = hit_ranks(reranked, relevant)
    context_ranks = hit_ranks(context, relevant)
    pack_summary = observation.get("context_pack_summary") or {}
    pack_omitted = int(pack_summary.get("omitted_count") or 0)
    pack_truncated = int(pack_summary.get("truncated_count") or 0)
    context_pack_loss = pack_omitted > 0 or pack_truncated > 0
    evidence_assessment = observation.get("evidence_assessment") or {}
    query_plan = observation.get("query_plan") or {}
    sub_queries = [
        {
            "sub_query_id": str(item.get("sub_query_id") or item.get("id") or ""),
            "query": str(item.get("query") or ""),
        }
        for item in query_plan.get("sub_queries") or []
        if isinstance(item, Mapping)
    ]
    relevant_evidence = [
        {
            "evidence_id": item.get("evidence_id"),
            "chunk_id": item.get("chunk_id"),
            "page": item.get("page"),
            "content_hash": item.get("content_hash"),
            "match_terms": list(item.get("match_terms") or []),
            "text": str(item.get("text") or item.get("quote") or ""),
        }
        for item in case.get("relevant_evidence") or []
    ]
    judgment = review.get("judgment") or {}
    source = dataset.get("source") or {}
    primary = str(adjudication["primary"])
    secondary = [str(value) for value in adjudication.get("secondary") or []]
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "semantic_label_source": "blind_pre_grounding_review_only",
        "blind_review": {
            "semantic_correct": bool(judgment.get("semantic_correct")),
            "semantic_status": judgment.get("semantic_status"),
            "semantic_score": judgment.get("semantic_score"),
            "confidence_score": judgment.get("confidence_score"),
            "answer_claim_count": judgment.get("answer_claim_count"),
            "missing_or_incorrect_aspect_count": judgment.get(
                "missing_or_incorrect_aspect_count"
            ),
            "record_sha256": review.get("record_sha256"),
            "free_text_reason_available": False,
        },
        "fixed_case": {
            "dataset_sha256": dataset_sha256,
            "query": str(case.get("query") or ""),
            "reference_answer": str(case.get("reference_answer") or ""),
            "expected_terms": list(case.get("expected_terms") or []),
            "tags": list(case.get("tags") or []),
            "relevant_chunk_ids": relevant_chunk_ids,
            "relevant_document_ids": list(case.get("relevant_document_ids") or []),
            "relevant_document_names": list(case.get("relevant_document_names") or []),
            "relevant_evidence": relevant_evidence,
            "source": {
                "relative_path": source.get("relative_path"),
                "content_sha256": source.get("content_sha256"),
                "document_id": source.get("document_id"),
                "document_version_id": source.get("document_version_id"),
                "index_version_id": source.get("index_version_id"),
            },
        },
        "archived_pre_grounding": {
            "results_sha256": results_sha256,
            "result_id": result.get("id"),
            "pre_grounding_answer": pre_answer,
            "pre_grounding_answer_sha256": sha_text(pre_answer),
            "query_plan": {
                "disposition": query_plan.get("disposition"),
                "sub_queries": sub_queries,
            },
            "evidence_assessment": {
                "status": evidence_assessment.get("status"),
                "query_coverage": evidence_assessment.get("query_coverage"),
                "missing_aspects": list(evidence_assessment.get("missing_aspects") or []),
                "reasons": list(evidence_assessment.get("reasons") or []),
                "sub_query_coverage": evidence_assessment.get("sub_query_coverage") or {},
            },
            "retrieval_to_context": {
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "context_count": len(context),
                "relevant_retrieved_ranks": retrieved_ranks,
                "relevant_reranked_ranks": reranked_ranks,
                "relevant_context_ranks": context_ranks,
                "relevant_retrieved_count": len(set(unique_chunk_ids(retrieved)) & relevant),
                "relevant_reranked_count": len(set(unique_chunk_ids(reranked)) & relevant),
                "relevant_context_count": len(set(unique_chunk_ids(context)) & relevant),
                "relevant_missing_from_retrieved": sorted(
                    relevant - set(unique_chunk_ids(retrieved))
                ),
                "relevant_missing_from_reranked": sorted(
                    relevant - set(unique_chunk_ids(reranked))
                ),
                "relevant_missing_from_context": sorted(
                    relevant - set(unique_chunk_ids(context))
                ),
                "context_pack": {
                    "packed_count": pack_summary.get("packed_count"),
                    "omitted_count": pack_omitted,
                    "truncated_count": pack_truncated,
                    "used_tokens": pack_summary.get("used_tokens"),
                    "whole_prompt_tokens": pack_summary.get("whole_prompt_tokens"),
                    "context_pack_loss_detected": context_pack_loss,
                },
                "context_inventory": [
                    {
                        "rank": index,
                        "chunk_id": item.get("chunk_id"),
                        "document_id": item.get("document_id"),
                        "title": item.get("title") or item.get("file_title"),
                        "text_preview": compact(item.get("content") or item.get("text"), 600),
                    }
                    for index, item in enumerate(context, start=1)
                ],
            },
        },
        "diagnosis": {
            "primary_category": primary,
            "primary_category_label": CATEGORY_LABELS[primary],
            "secondary_categories": secondary,
            "secondary_category_labels": [CATEGORY_LABELS[value] for value in secondary],
            "pattern": adjudication["pattern"],
            "pattern_family": PATTERN_FAMILY_BY_CASE[case_id],
            "pattern_family_label": PATTERN_FAMILIES[
                PATTERN_FAMILY_BY_CASE[case_id]
            ]["label"],
            "failure_mechanism": adjudication["mechanism"],
            "stage_evidence": list(adjudication["evidence"]),
            "confidence_score": adjudication["confidence"],
            "confidence_level": (
                "high"
                if float(adjudication["confidence"]) >= 0.90
                else "medium"
            ),
            "pre_stage_post_generation_loss_applicable": False,
            "note": (
                "该诊断止于pre_grounding；pre之后的grounding/final损伤不在本次主因归属中。"
            ),
        },
        "gold_independent_self_review": adjudication["self_review"],
    }


def summarize(cases: list[dict[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    primary = Counter(case["diagnosis"]["primary_category"] for case in cases)
    secondary = Counter(
        category
        for case in cases
        for category in case["diagnosis"]["secondary_categories"]
    )
    case_level_patterns = Counter(case["diagnosis"]["pattern"] for case in cases)
    pattern_family_rows = []
    for family, details in PATTERN_FAMILIES.items():
        member_ids = sorted(
            case["case_id"]
            for case in cases
            if case["diagnosis"]["pattern_family"] == family
        )
        pattern_family_rows.append(
            {
                "pattern_family": family,
                "label": details["label"],
                "count": len(member_ids),
                "rate": len(member_ids) / len(cases),
                "case_ids": member_ids,
            }
        )
    pattern_family_rows.sort(
        key=lambda item: (-item["count"], item["pattern_family"])
    )
    per_difficulty: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        rows = [case for case in cases if case["difficulty"] == difficulty]
        per_difficulty[difficulty] = {
            "blind_incorrect_count": len(rows),
            "primary_categories": dict(
                sorted(
                    Counter(
                        case["diagnosis"]["primary_category"] for case in rows
                    ).items()
                )
            ),
            "relevant_all_retrieved_count": sum(
                not case["archived_pre_grounding"]["retrieval_to_context"][
                    "relevant_missing_from_retrieved"
                ]
                for case in rows
            ),
            "context_pack_loss_count": sum(
                bool(
                    case["archived_pre_grounding"]["retrieval_to_context"][
                        "context_pack"
                    ]["context_pack_loss_detected"]
                )
                for case in rows
            ),
        }

    easy_rows = [case for case in cases if case["difficulty"] == "easy"]
    easy_self_review = Counter(
        case["gold_independent_self_review"]["coverage_likelihood"]
        for case in easy_rows
    )
    easy_current_correct = int(summary["per_difficulty"]["easy"]["correct_count"])
    easy_total = int(summary["per_difficulty"]["easy"]["case_count"])
    easy_required_correct = 48
    evidence_status = Counter(
        str(case["archived_pre_grounding"]["evidence_assessment"]["status"])
        for case in cases
    )
    all_pack_clean = all(
        not case["archived_pre_grounding"]["retrieval_to_context"]["context_pack"][
            "context_pack_loss_detected"
        ]
        for case in cases
    )
    return {
        "schema_version": "stt.eval.pre-grounding-blind-diagnostics-summary.v1",
        "source_snapshot": {
            "run_label": RUN_LABEL,
            "review_iteration": REVIEW_ITERATION,
            "answer_stage": "pre_grounding",
            "review_status": summary.get("status"),
            "review_record_count": summary.get("record_count"),
            "evaluation_case_count": summary["overall"]["case_count"],
            "blind_incorrect_count": summary["overall"]["incorrect_count"],
            "judge_model": summary["judge"]["requested_model"],
            "visible_answer_only": summary["judge"]["visible_answer_only"],
            "production_metrics_supplied": summary["judge"]["production_metrics_supplied"],
            "free_text_judgment_recorded": summary["judge"][
                "free_text_judgment_recorded"
            ],
        },
        "blind_accuracy": {
            "overall": {
                "correct_count": summary["overall"]["correct_count"],
                "case_count": summary["overall"]["case_count"],
                "binary_accuracy": summary["overall"]["binary_accuracy"],
            },
            "per_difficulty": {
                difficulty: {
                    "correct_count": summary["per_difficulty"][difficulty][
                        "correct_count"
                    ],
                    "incorrect_count": summary["per_difficulty"][difficulty][
                        "incorrect_count"
                    ],
                    "case_count": summary["per_difficulty"][difficulty]["case_count"],
                    "binary_accuracy": summary["per_difficulty"][difficulty][
                        "binary_accuracy"
                    ],
                }
                for difficulty in DIFFICULTIES
            },
        },
        "primary_categories": {
            category: {
                "label": CATEGORY_LABELS[category],
                **rate(primary.get(category, 0), len(cases)),
            }
            for category in CATEGORY_LABELS
        },
        "secondary_categories": {
            category: {
                "label": CATEGORY_LABELS[category],
                "count": secondary.get(category, 0),
            }
            for category in CATEGORY_LABELS
        },
        "per_difficulty_diagnostics": per_difficulty,
        "high_frequency_patterns": [
            item for item in pattern_family_rows if item["count"] >= 2
        ],
        "all_pattern_families": pattern_family_rows,
        "case_level_patterns": [
            {"pattern": pattern, "count": count, "rate": count / len(cases)}
            for pattern, count in case_level_patterns.most_common()
        ],
        "stage_checks": {
            "evidence_assessment_status": dict(sorted(evidence_status.items())),
            "context_pack_loss_count": sum(
                case["archived_pre_grounding"]["retrieval_to_context"][
                    "context_pack"
                ]["context_pack_loss_detected"]
                for case in cases
            ),
            "all_30_context_packs_have_zero_omitted_and_truncated": all_pack_clean,
            "intermediate_process_loss_count": primary.get(
                "intermediate_process_loss", 0
            ),
            "post_pre_grounding_loss_used_as_cause": False,
        },
        "easy_target_gap": {
            "target_rule": ">95%",
            "case_count": easy_total,
            "minimum_correct_count": easy_required_correct,
            "blind_pre_grounding_correct_count": easy_current_correct,
            "blind_pre_grounding_accuracy": easy_current_correct / easy_total,
            "additional_correct_needed": max(0, easy_required_correct - easy_current_correct),
            "incorrect_pool_primary_categories": dict(
                sorted(
                    Counter(
                        case["diagnosis"]["primary_category"] for case in easy_rows
                    ).items()
                )
            ),
            "incorrect_cases": [
                {
                    "case_id": case["case_id"],
                    "primary_category": case["diagnosis"]["primary_category"],
                    "pattern": case["diagnosis"]["pattern"],
                    "self_review_coverage_likelihood": case[
                        "gold_independent_self_review"
                    ]["coverage_likelihood"],
                }
                for case in easy_rows
            ],
            "gold_independent_self_review_coverage": dict(sorted(easy_self_review.items())),
            "likely_recoverable_count": easy_self_review.get("likely", 0),
            "possible_but_unverified_count": easy_self_review.get("possible", 0),
            "verified_recoverable_count": 0,
            "conclusion": (
                "四个easy盲审错误由2个dataset/reference与2个source ambiguity组成；"
                "关闭目标差距需净恢复2题。无金标自审明确likely覆盖1题、possible覆盖1题，"
                "但没有任何修复重跑证据，因此不得声称已达到48/50。"
            ),
        },
        "data_quality_checks": {
            "blind_incorrect_joined": len(cases),
            "unique_case_ids": len({case["case_id"] for case in cases}),
            "expected_difficulty_counts": {"easy": 4, "medium": 6, "hard": 20},
            "actual_difficulty_counts": dict(
                sorted(Counter(case["difficulty"] for case in cases).items())
            ),
            "production_pass_fail_used_for_semantic_label_or_diagnosis": False,
            "application_or_dataset_files_modified": False,
            "network_or_llm_calls_made": False,
        },
    }


def report_markdown(cases: list[dict[str, Any]], patterns: Mapping[str, Any]) -> str:
    primary_rows = [
        (
            CATEGORY_LABELS[category],
            details["count"],
            f'{details["rate"]:.1%}',
        )
        for category, details in patterns["primary_categories"].items()
        if details["count"]
    ]
    lines = [
        "# qwen3.7-max pre-grounding 盲审错题诊断",
        "",
        "## 技术摘要",
        "",
        "本次只读诊断以 `live-cand-max-mono-01` 的 qwen3.7-max pre-grounding 盲审为唯一语义标签：150 题中 120 题正确、30 题错误，二元准确率 **80.0%**。分难度为 easy **46/50（92.0%）**、medium **44/50（88.0%）**、hard **30/50（60.0%）**。153 条调用记录包含 150 条 evaluation 与 3 条 calibration；校准通过。",
        "",
        "30 个错误的互斥主因是：`dataset/reference` 13（43.3%）、生成/推理 6（20.0%）、流程中间内容丢失 6（20.0%）、`source ambiguity` 4（13.3%）、`freshness/execution` 1（3.3%）。召回不足、召回噪声/不清晰、上下文丢失和评测系统错误候选都没有被强行设为 primary；它们只在存在更强主因时作为 secondary 风险保留。",
        "",
        "最重要的阶段结论是：30 题的 context pack 均为 `omitted=0`、`truncated=0`，所以没有证据支持“已选上下文在pack阶段丢失”。但有 6 题的固定相关 chunk 已进入 retrieved（部分已进入 reranked），却没有进入生成 context，属于 pre-grounding 之前真实的流程中间丢失。pre之后的grounding/final变化没有参与本报告归因。",
        "",
        "## 主因分解显示题集/源质量风险高于模型生成缺陷",
        "",
        "| 互斥主因 | 错题数 | 占30题 |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {label} | {count} | {share} |" for label, count, share in primary_rows)
    lines.extend(
        [
            "",
            "`dataset/reference` 与 `source ambiguity` 合计 17/30（56.7%）。这不等于把系统错误洗成题目错误：每题都同时检查了固定query、reference、相关证据和pre回答，只有在reference不回答问题、数字/实体绑定错误、跨章节作用域混用、或源抽取本身缺失视觉/表格结构时才归入这两类。",
            "",
            "生成/推理 6 题的共同点是相关事实已完整进入context且pack无损，但答案漏步骤、漏字段、把配置哨兵当尺寸、过度保守拒绝可支持的低风险归纳，或把标签标识改写成模糊含义。流程中间丢失 6 题则有可核验的stage ranks，不能归咎于生成模型。",
            "",
            "## 跨题高频模式",
            "",
            "下表只列至少出现2次的通用模式；这些模式在脚本中先验定义并覆盖全部30题，避免用每题唯一标签冒充‘高频’。单例模式仍完整保留在 `high-frequency-patterns.json` 的 `all_pattern_families` 中。",
            "",
            "| 通用模式 | 题数 | 占30题 | cases |",
            "|---|---:|---:|---|",
        ]
    )
    for item in patterns["high_frequency_patterns"]:
        case_ids = "、".join(f'`{case_id}`' for case_id in item["case_ids"])
        lines.append(
            f'| {item["label"]} | {item["count"]} | '
            f'{item["rate"]:.1%} | {case_ids} |'
        )
    lines.extend(
        [
            "",
            "## easy距离目标2题，但自审尚不足以证明达标",
            "",
            "easy 的盲审pre正确数为46；`>95%` 在50题上要求至少48题，仍差2题。4个错误池由2个 `dataset/reference` 和2个 `source ambiguity` 组成：",
            "",
            "| case | 通用模式 | 无金标自审触发 | 潜在覆盖 |",
            "|---|---|---|---|",
        ]
    )
    for case in [item for item in cases if item["difficulty"] == "easy"]:
        review = case["gold_independent_self_review"]
        lines.append(
            f'| `{case["case_id"]}` | `{case["diagnosis"]["pattern"]}` | '
            f'{"是" if review["would_trigger"] else "否"} | {review["coverage_likelihood"]} |'
        )
    lines.extend(
        [
            "",
            "不读取reference的通用自审对 `EASY-038` 为 **likely**：同一OK键对应三个阶段，答案可条件化消歧；对 `EASY-047` 仅为 **possible**：需要可靠恢复表格行边界。`EASY-004` 的pre与数据行一致，自审不应为了金标改成六项；`EASY-031` 的按钮图标已从文本抽取中丢失，纯文本自审无法恢复。由此只能说“1题可能高概率、另1题可能但未验证”，不能声称修复后必然48/50。",
            "",
            "## 六个流程中间丢失均发生在pack之前",
            "",
            "| case | retrieved相关rank | reranked相关rank | context相关rank | 结果 |",
            "|---|---|---|---|---|",
        ]
    )
    for case in [
        item
        for item in cases
        if item["diagnosis"]["primary_category"] == "intermediate_process_loss"
    ]:
        stage = case["archived_pre_grounding"]["retrieval_to_context"]
        lines.append(
            f'| `{case["case_id"]}` | {stage["relevant_retrieved_ranks"]} | '
            f'{stage["relevant_reranked_ranks"]} | {stage["relevant_context_ranks"]} | '
            f'{case["diagnosis"]["pattern"]} |'
        )
    lines.extend(
        [
            "",
            "由于这6题的pack本身没有省略或截断，修复点应位于rerank、context选择、sub-query覆盖判定和SUFFICIENT门禁之间：只要固定相关chunk已经被召回，就应对每个必需分支执行相关chunk守恒检查；不能让assessor在关键块缺失时仍报告全部子问题已覆盖。",
            "",
            "## 逐题主因与失败链",
            "",
            "下表覆盖全部30题。`评测系统错误候选`只作为secondary信号：盲审模型本身没有记录自由文本理由，不能仅凭结果反推judge错误。完整query、reference、pre回答、源证据、各阶段rank和置信度见 `cases.json`。",
            "",
            "| case | 主因 | secondary | 失败链（摘要） | 置信度 |",
            "|---|---|---|---|---:|",
        ]
    )
    for case in cases:
        diagnosis = case["diagnosis"]
        secondary = "、".join(diagnosis["secondary_category_labels"]) or "—"
        lines.append(
            f'| `{case["case_id"]}` | {diagnosis["primary_category_label"]} | '
            f'{secondary} | {compact(diagnosis["failure_mechanism"], 150)} | '
            f'{diagnosis["confidence_score"]:.2f} |'
        )
    lines.extend(
        [
            "",
            "## 范围、数据与方法",
            "",
            "- 分析粒度：blind evaluation 的 `difficulty × case_id`，150个唯一case；只对30个最终blind incorrect做因果裁决。",
            "- 语义标签：只使用pre-grounding blind review的 `judgment.semantic_correct`；没有使用归档生产 `passed`、final correctness或strict pass来决定标签或主因。",
            "- 数据连接：review record → 同case归档 `observation.pre_grounding_answer` / query plan / retrieved / reranked / context / context pack → 固定dataset query/reference/relevant evidence → 原PDF身份与索引版本。",
            "- 自动派生：输入SHA、case唯一性、各难度数量、相关chunk的stage ranks、pack omitted/truncated、evidence assessment状态。",
            "- 人工裁决：`ADJUDICATIONS`显式记录互斥primary、可重叠secondary、失败机制、阶段证据、置信度及无金标自审判断；脚本不会把启发式信号伪装成自动因果。",
            "- 禁止项：无网络/LLM调用，不读取凭据，不修改 `app/` 或 `datasets/`。",
            "",
            "## 限制与稳健性",
            "",
            "1. Blind review输出刻意不保存自由文本reasoning，只保存semantic status/score、置信度和计数；逐题原因是基于归档阶段证据的独立重建，不是provider理由的复述。",
            "2. 固定相关chunk标注是检索覆盖的控制源，但不自动证明reference正确；本报告因此把dataset/reference与source ambiguity分开。",
            "3. `评测系统错误候选`不是已确认judge错误。它表示固定reference/证据可能导致blind verdict失真，需要在修订后的整套题集上重新盲审。",
            "4. 自审覆盖是counterfactual可触发性分析，没有运行修复后的模型；`likely`和`possible`都不是通过结果。",
            "5. 本报告只解释pre-grounding错误，不代表最终用户可见答案的准确率，也不替代完整live重跑。",
            "",
            "## 通用修复与验证建议",
            "",
            "1. 对全部150题执行系统化query/reference/evidence一致性门禁：实体范围、单复数基数、数字到实体绑定、条件句主语、跨章节适用范围和决策问题是否被reference实际回答。若修订，必须版本化整套suite并重新建立基线，不能只改失败题。",
            "2. 在retrieval→rerank→context边界增加按必需sub-query的相关证据守恒：已召回的关键块若被丢弃，assessor不得报告SUFFICIENT，应触发一次通用重排/补取。",
            "3. 对procedure/table/composite答案增加reference-free completeness自审：编号数量守恒、字段覆盖、实体/数字绑定、配置哨兵类型、标识符原样保留和首句与正文一致性。",
            "4. 表格与UI图片索引保留结构：行/列坐标、图标alt/OCR、相邻行边界和视觉标签；结构置信度低时应输出条件化答案或请求更强证据。",
            "5. freshness只用于真实动态事实。静态手册快照中的“当前页面/当前状态示例”不应单独触发STALE；同时不能把文档示例状态当成真实设备状态。",
            "6. 任何修复都需用同一固定或正式版本化套件重新进行pre盲审与最终答案盲审，并以最终答案正确率为主门禁；当前产物不声称任何修复已经生效。",
            "",
            "## 进一步问题",
            "",
            "- 对6个中间丢失案例，context选择为何会丢掉reranked第1位或已召回的第二分支？应通过统一trace字段记录每个chunk的淘汰原因。",
            "- 对4个source ambiguity案例，原PDF视觉重建能否稳定恢复按钮图标与表格行边界？若不能，这些题不应作为可判定基线题。",
            "- 对13个dataset/reference案例，执行全套一致性规则后会有多少正确题也被同类规则命中？只有全量审计才能避免失败题特判。",
            "",
        ]
    )
    return "\n".join(lines)


def write_cases_csv(cases: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "difficulty",
        "blind_status",
        "blind_score",
        "blind_confidence",
        "primary_category",
        "primary_category_label",
        "secondary_categories",
        "pattern",
        "pattern_family",
        "pattern_family_label",
        "diagnosis_confidence",
        "retrieved_relevant_ranks",
        "reranked_relevant_ranks",
        "context_relevant_ranks",
        "context_pack_omitted_count",
        "context_pack_truncated_count",
        "evidence_status",
        "query",
        "reference_answer",
        "pre_grounding_answer",
        "failure_mechanism",
        "stage_evidence",
        "self_review_would_trigger",
        "self_review_coverage_likelihood",
        "self_review_rationale",
    ]
    with (OUTPUT_DIR / "cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            review = case["blind_review"]
            diagnosis = case["diagnosis"]
            archived = case["archived_pre_grounding"]
            stage = archived["retrieval_to_context"]
            pack = stage["context_pack"]
            self_review = case["gold_independent_self_review"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "difficulty": case["difficulty"],
                    "blind_status": review["semantic_status"],
                    "blind_score": review["semantic_score"],
                    "blind_confidence": review["confidence_score"],
                    "primary_category": diagnosis["primary_category"],
                    "primary_category_label": diagnosis["primary_category_label"],
                    "secondary_categories": "|".join(
                        diagnosis["secondary_categories"]
                    ),
                    "pattern": diagnosis["pattern"],
                    "pattern_family": diagnosis["pattern_family"],
                    "pattern_family_label": diagnosis["pattern_family_label"],
                    "diagnosis_confidence": diagnosis["confidence_score"],
                    "retrieved_relevant_ranks": json.dumps(
                        stage["relevant_retrieved_ranks"], ensure_ascii=False
                    ),
                    "reranked_relevant_ranks": json.dumps(
                        stage["relevant_reranked_ranks"], ensure_ascii=False
                    ),
                    "context_relevant_ranks": json.dumps(
                        stage["relevant_context_ranks"], ensure_ascii=False
                    ),
                    "context_pack_omitted_count": pack["omitted_count"],
                    "context_pack_truncated_count": pack["truncated_count"],
                    "evidence_status": archived["evidence_assessment"]["status"],
                    "query": case["fixed_case"]["query"],
                    "reference_answer": case["fixed_case"]["reference_answer"],
                    "pre_grounding_answer": archived["pre_grounding_answer"],
                    "failure_mechanism": diagnosis["failure_mechanism"],
                    "stage_evidence": " | ".join(diagnosis["stage_evidence"]),
                    "self_review_would_trigger": self_review["would_trigger"],
                    "self_review_coverage_likelihood": self_review[
                        "coverage_likelihood"
                    ],
                    "self_review_rationale": self_review["rationale"],
                }
            )


def write_checksums(paths: list[Path]) -> None:
    lines = [
        f"{sha_file(path)}  {Path(os.path.relpath(path, OUTPUT_DIR)).as_posix()}"
        for path in paths
    ]
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    summary = read_json(REVIEW_SUMMARY_PATH)
    require(summary.get("status") == "succeeded", "blind review did not succeed")
    require(summary.get("answer_stage") == "pre_grounding", "wrong answer stage")
    require(summary.get("record_count") == 153, "expected 153 review records")
    require(summary["overall"]["case_count"] == 150, "expected 150 evaluation cases")
    require(summary["overall"]["incorrect_count"] == 30, "expected 30 blind errors")
    require(summary["judge"]["visible_answer_only"] is True, "judge not answer-only")
    require(
        summary["judge"]["production_metrics_supplied"] is False,
        "production metrics were supplied to judge",
    )
    require(
        summary["judge"]["free_text_judgment_recorded"] is False,
        "unexpected free-text judgment",
    )
    require(
        sha_file(REVIEW_RECORDS_PATH) == summary["records_sha256"],
        "review records SHA mismatch",
    )
    run_index_path = source_path(summary["input"]["run_index_path"])
    registry_path = source_path(summary["input"]["baseline_registry_path"])
    require(
        sha_file(run_index_path) == summary["input"]["run_index_sha256"],
        "run index SHA mismatch",
    )
    require(
        sha_file(registry_path) == summary["input"]["baseline_registry_sha256"],
        "baseline registry SHA mismatch",
    )

    records = read_jsonl(REVIEW_RECORDS_PATH)
    require(len(records) == 153, "review JSONL line count mismatch")
    reviews = selected_review_records(records)
    require(len(reviews) == 150, "expected 150 selected evaluation records")
    incorrect_reviews = {
        case_id: review
        for case_id, review in reviews.items()
        if not bool((review.get("judgment") or {}).get("semantic_correct"))
    }
    require(len(incorrect_reviews) == 30, "selected blind incorrect count mismatch")
    require(
        set(incorrect_reviews) == set(ADJUDICATIONS),
        "adjudication case set does not match blind incorrect cases",
    )
    family_memberships = [
        case_id
        for details in PATTERN_FAMILIES.values()
        for case_id in details["case_ids"]
    ]
    require(
        len(family_memberships) == len(set(family_memberships)),
        "pattern-family case memberships overlap",
    )
    require(
        set(family_memberships) == set(ADJUDICATIONS),
        "pattern-family case set does not match adjudications",
    )

    cases: list[dict[str, Any]] = []
    input_paths = [
        Path(__file__).resolve(),
        REVIEW_SUMMARY_PATH,
        REVIEW_RECORDS_PATH,
        run_index_path,
        registry_path,
    ]
    for difficulty in DIFFICULTIES:
        difficulty_summary = summary["per_difficulty"][difficulty]
        dataset_path = source_path(difficulty_summary["dataset_path"])
        results_path = source_path(difficulty_summary["results_path"])
        require(
            sha_file(dataset_path) == difficulty_summary["dataset_sha256"],
            f"{difficulty} dataset SHA mismatch",
        )
        require(
            sha_file(results_path) == difficulty_summary["results_sha256"],
            f"{difficulty} results SHA mismatch",
        )
        dataset = read_json(dataset_path)
        dataset_cases = {item["case_id"]: item for item in dataset["cases"]}
        result_rows = read_jsonl(results_path)
        require(len(result_rows) == 50, f"{difficulty} expected 50 results")
        results = {item["case_id"]: item for item in result_rows}
        require(len(results) == 50, f"{difficulty} duplicate result case_id")
        for case_id in sorted(incorrect_reviews):
            if incorrect_reviews[case_id].get("difficulty") != difficulty:
                continue
            require(case_id in dataset_cases, f"dataset missing {case_id}")
            require(case_id in results, f"results missing {case_id}")
            # Only observation is used; result.metrics and result.passed are not
            # consulted anywhere in semantic labeling or causal adjudication.
            cases.append(
                build_case(
                    difficulty=difficulty,
                    dataset=dataset,
                    case=dataset_cases[case_id],
                    result={
                        "id": results[case_id].get("id"),
                        "observation": results[case_id].get("observation") or {},
                    },
                    review=incorrect_reviews[case_id],
                    dataset_sha256=difficulty_summary["dataset_sha256"],
                    results_sha256=difficulty_summary["results_sha256"],
                )
            )
        input_paths.extend([dataset_path, results_path])

    cases.sort(key=lambda item: (DIFFICULTIES.index(item["difficulty"]), item["case_id"]))
    require(len(cases) == 30, "diagnostic row count mismatch")
    require(len({case["case_id"] for case in cases}) == 30, "duplicate diagnostic case")
    actual_counts = Counter(case["difficulty"] for case in cases)
    require(actual_counts == {"easy": 4, "medium": 6, "hard": 20}, "difficulty counts mismatch")
    require(
        all(not case["blind_review"]["semantic_correct"] for case in cases),
        "correct blind case leaked into diagnostics",
    )
    require(
        all(
            not case["archived_pre_grounding"]["retrieval_to_context"][
                "context_pack"
            ]["context_pack_loss_detected"]
            for case in cases
        ),
        "unexpected context-pack loss in 30-case cohort",
    )

    cases_payload = {
        "schema_version": "stt.eval.pre-grounding-blind-diagnostic-cases.v1",
        "run_label": RUN_LABEL,
        "review_iteration": REVIEW_ITERATION,
        "answer_stage": "pre_grounding",
        "case_count": len(cases),
        "method": {
            "semantic_label": "blind judgment.semantic_correct only",
            "production_pass_fail_used": False,
            "human_adjudication_table": "ADJUDICATIONS in diagnose_pre_grounding_blind_failures.py",
            "post_pre_grounding_process_considered": False,
        },
        "cases": cases,
    }
    patterns = summarize(cases, summary)
    write_json(OUTPUT_DIR / "cases.json", cases_payload)
    write_cases_csv(cases)
    write_json(OUTPUT_DIR / "high-frequency-patterns.json", patterns)
    (OUTPUT_DIR / "REPORT.md").write_text(
        report_markdown(cases, patterns), encoding="utf-8", newline="\n"
    )
    generated = [
        Path(__file__).resolve(),
        OUTPUT_DIR / "cases.json",
        OUTPUT_DIR / "cases.csv",
        OUTPUT_DIR / "high-frequency-patterns.json",
        OUTPUT_DIR / "REPORT.md",
    ]
    # Include immutable inputs after generated artifacts for end-to-end audit.
    checksum_paths = generated + list(dict.fromkeys(input_paths[1:]))
    write_checksums(checksum_paths)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "case_count": len(cases),
                "primary": {
                    key: value["count"]
                    for key, value in patterns["primary_categories"].items()
                },
                "output_dir": str(OUTPUT_DIR),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
