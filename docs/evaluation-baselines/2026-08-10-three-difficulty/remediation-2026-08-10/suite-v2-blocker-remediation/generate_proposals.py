from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REMEDIATION_DIR = THIS_DIR.parent
SUITE_ROOT = REMEDIATION_DIR.parent
WORKSPACE_ROOT = SUITE_ROOT.parents[2]
DATASET_DIR = SUITE_ROOT / "datasets"
VISUAL_AUDIT = REMEDIATION_DIR / "suite-source-evaluator-qa" / "pdf-visual-review" / "cases.json"
SOURCE_PDF = WORKSPACE_ROOT / "doc" / "PantumP3500用户手册zh_CNV1.2_1644316283788.pdf"

SCHEMA_VERSION = "stt.eval.suite-v2-blocker-remediation-proposal.v2-no-index-refresh"
SOURCE_PDF_SHA256 = "c4c7b2a434f3f86c4988d3961ac7304970029e71b0201fb6c10ca722b63bce2b"
TARGET_IDS = {
    "P3500-EASY-004",
    "P3500-EASY-049",
    "P3500-HARD-003",
    "P3500-HARD-007",
    "P3500-HARD-008",
    "P3500-HARD-009",
    "P3500-HARD-011",
    "P3500-HARD-015",
    "P3500-HARD-021",
    "P3500-HARD-037",
    "P3500-HARD-040",
    "P3500-HARD-041",
    "P3500-HARD-042",
}
NO_INDEX_REPLACEMENT_IDS = {
    "P3500-EASY-004",
    "P3500-HARD-003",
    "P3500-HARD-007",
    "P3500-HARD-009",
    "P3500-HARD-011",
    "P3500-HARD-037",
    "P3500-HARD-040",
}
REVIEWER_BLOCK_TARGET_IDS = {
    "P3500-HARD-003",
    "P3500-HARD-007",
    "P3500-HARD-009",
    "P3500-HARD-011",
}
EXPECTED_INPUT_HASHES = {
    "datasets/easy.json": "1a3cacbb59b3aa7f354a569f828ad6214351f07cd3d0923a798e98cc33c41b2a",
    "datasets/medium.json": "89c570d10282342fd7facec9bd7d20af61337b5ecaef0038157ed824c25c0369",
    "datasets/hard.json": "e5be4d2495618e819b9758fa753bb10d553037ca10b26dc6ccd8521404600a04",
    "pdf-visual-review/cases.json": "9d4e1ae706093173a13c47d063c64e9b0b24fb5314b23b662a2b86226b07f76a",
    "source.pdf": SOURCE_PDF_SHA256,
}


def _concept(concept_key: str, *variants: str) -> dict[str, object]:
    return {"concept_key": concept_key, "any_of": list(variants)}


DEFINITIONS: dict[str, dict[str, Any]] = {
    "P3500-EASY-004": {
        "disposition": "replacement",
        "blocker_reason": "旧参考与PDF表格冲突，而当前活动chunk又缺失矩阵数据行；为避免重建索引，改用同一手册中已完整入索引的直接事实。",
        "source_facts": [
            {
                "statement": "产品视图编号1为出纸槽，用于存放打印出来的纸张。",
                "pdf_pages": [13],
                "evidence_ids": ["ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6"],
            },
        ],
        "query": "产品视图中编号1的出纸槽用于什么？",
        "reference_answer": "用于存放打印出来的纸张。",
        "expected_terms": ["存放打印出来的纸张"],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "output_slot_function",
                    "用于存放打印出来的纸张",
                    "存放打印出的纸张",
                )
            ]
        },
        "evidence_ids": ["ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6"],
        "evidence_match_terms": {
            "ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6": [
                "出纸槽",
                "用于存放打印出来的纸张",
            ]
        },
        "binding_overrides": {
            "ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6": {
                "visual_pages": [13],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含编号1、出纸槽名称及其用途，无需视觉补写或重建索引。",
            }
        },
        "required_branches": [{"branch_id": "b01", "concept_keys": ["output_slot_function"]}],
        "critical_constraints": {},
        "difficulty_equivalence": "原题属于单证据直接事实；替换题同样只需从一个完整表格行读取一个部件功能，信息量与推理深度均保持easy。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-EASY-049": {
        "disposition": "replacement",
        "blocker_reason": "“四条水平横线代表文本行”来自自动图像描述，源页没有该图例或定义，原题不可确定评分。",
        "source_facts": [
            {
                "statement": "本章展示的是最全的菜单结构。",
                "pdf_pages": [90],
                "evidence_ids": ["ev_a15a520f37b5352b1aa0265c875cfca852ce6e0d"],
            },
            {
                "statement": "实际使用机型的菜单结构可能与手册所列结构存在差异。",
                "pdf_pages": [90],
                "evidence_ids": ["ev_a15a520f37b5352b1aa0265c875cfca852ce6e0d"],
            },
        ],
        "query": "手册对本章菜单结构与实际使用机型的菜单结构差异作了什么说明？",
        "reference_answer": "本章展示的是最全的菜单结构；实际使用机型的菜单结构可能与手册所列结构存在差异。",
        "expected_terms": ["最全的菜单结构", "实际使用机型可能存在差异"],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept("chapter_scope", "最全的菜单结构", "完整菜单结构"),
                _concept(
                    "model_variance",
                    "实际使用机型的菜单结构可能存在差异",
                    "实际机型菜单可能与所列结构不同",
                ),
            ]
        },
        "evidence_ids": ["ev_a15a520f37b5352b1aa0265c875cfca852ce6e0d"],
        "evidence_match_terms": {
            "ev_a15a520f37b5352b1aa0265c875cfca852ce6e0d": [
                "最全的菜单结构",
                "实际使用机型的菜单结构可能与以下的菜单结构存在差异",
            ]
        },
        "binding_overrides": {
            "ev_a15a520f37b5352b1aa0265c875cfca852ce6e0d": {
                "visual_pages": [90],
                "integrity": "usable_text_scope_only",
                "note": "只使用chunk中的正文注释；不使用其自动生成的图标语义描述。",
            }
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": ["chapter_scope", "model_variance"],
            }
        ],
        "critical_constraints": {},
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-003": {
        "disposition": "semantic_replacement",
        "blocker_reason": "旧题把通用故障排除前检查误写成标准纸盒卡纸专属条件，并要求证据未直接说明的漏检后果。",
        "source_review_reason": "后果分支缺乏直接证据；活动证据只完整支持处理常见故障前的八项通用检查。",
        "source_facts": [
            {
                "statement": "处理常见故障前应检查电源和开关、保护零件、激光碳粉盒、出纸槽、纸张、接口电缆、驱动程序和计算机端口。",
                "pdf_pages": [79],
                "evidence_ids": ["ev_ee696de1bd6dcdeb9c673828918e142287d5df1e"],
            }
        ],
        "query": "在处理常见故障之前，首先应检查哪些情况？",
        "reference_answer": "检查电源线是否连接正确且打印机电源开关是否已打开；所有保护零件是否已拆除；激光碳粉盒是否已正确安装；出纸槽是否已完全合上；纸张是否已正确放入纸盒；接口电缆线是否已正确连接打印机和计算机；是否已选择并安装正确的打印机驱动程序；计算机端口是否已安装并连接到正确的打印机端口。",
        "expected_terms": [
            "电源线是否连接正确",
            "打印机电源开关是否已打开",
            "保护零件是否已拆除",
            "激光碳粉盒是否已正确安装",
            "出纸槽是否已完全合上",
            "纸张是否已正确放入纸盒",
            "接口电缆线是否已正确连接打印机和计算机",
            "正确的打印机驱动程序",
            "计算机端口是否已安装并连接到正确的打印机端口",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "power_and_switch",
                    "电源线连接正确且电源开关已打开",
                    "检查电源线和打印机电源开关",
                ),
                _concept(
                    "protective_parts_removed", "所有保护零件已拆除", "检查保护零件是否已拆除"
                ),
                _concept("toner_cartridge_installed", "激光碳粉盒已正确安装", "检查激光碳粉盒安装"),
                _concept("output_slot_closed", "出纸槽已完全合上", "检查出纸槽是否完全合上"),
                _concept("paper_loaded", "纸张已正确放入纸盒", "检查纸张是否正确装入纸盒"),
                _concept(
                    "interface_cable_connected",
                    "接口电缆线已正确连接打印机和计算机",
                    "检查打印机与计算机的接口电缆",
                ),
                _concept(
                    "correct_driver_installed",
                    "已选择并安装正确的打印机驱动程序",
                    "检查正确驱动程序是否已安装",
                ),
                _concept(
                    "computer_port_connected",
                    "计算机端口已安装并连接到正确的打印机端口",
                    "检查计算机端口与正确打印机端口的连接",
                ),
            ]
        },
        "evidence_ids": ["ev_ee696de1bd6dcdeb9c673828918e142287d5df1e"],
        "evidence_match_terms": {
            "ev_ee696de1bd6dcdeb9c673828918e142287d5df1e": [
                "在处理常见故障之前",
                "电源线是否连接正确",
                "正确的打印机驱动程序",
                "计算机端口是否已安装并连接到正确的打印机端口",
            ]
        },
        "binding_overrides": {
            "ev_ee696de1bd6dcdeb9c673828918e142287d5df1e": {
                "visual_pages": [79],
                "integrity": "active_index_complete",
                "note": "活动chunk直接、完整列出八项通用检查；替换题不再要求无源后果，也不再把它们限定为标准纸盒卡纸专属条件。",
            }
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": [
                    "power_and_switch",
                    "protective_parts_removed",
                    "toner_cartridge_installed",
                    "output_slot_closed",
                    "paper_loaded",
                    "interface_cable_connected",
                    "correct_driver_installed",
                    "computer_port_connected",
                ],
            }
        ],
        "critical_constraints": {},
        "difficulty_equivalence": "替换题仍要求从同一故障排除段落完整汇总八项检查，保留多要点完整性要求，但删除无源推断。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-007": {
        "disposition": "semantic_replacement",
        "blocker_reason": "旧参考声称卡纸与打印多页进纸的解决办法均包含清洁，但打印多页进纸行没有清洁措施，比较结论失实。",
        "source_review_reason": "共同解决办法中的“清洁”不受证据支持；打印多页进纸自身的原因和处理措施均有完整直接证据。",
        "source_facts": [
            {
                "statement": "打印多页进纸的原因是静电量过大、介质受潮或粘合、内部部件故障；处理包括重新分离介质、使用推荐介质或更好的干燥介质。",
                "pdf_pages": [85],
                "evidence_ids": ["ev_c9895eb1d2eb948d42cb6b092ab0031d3cc40919"],
            }
        ],
        "query": "打印多页进纸的故障原因有哪些，应如何处理？",
        "reference_answer": "原因包括打印介质含静电量过大、打印介质受潮或粘合在一起、内部部件故障。处理时可将打印介质重新分离以消除部分静电，使用推荐打印介质；还可重新分离打印介质或使用更好的干燥打印介质。",
        "expected_terms": [
            "打印介质含静电量过大",
            "打印介质受潮或粘合在一起",
            "内部部件故障",
            "将打印介质重新分离",
            "消除部分静电",
            "使用推荐打印介质",
            "使用更好的干燥打印介质",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept("multi_feed_static", "打印介质含静电量过大", "介质静电过大"),
                _concept("multi_feed_damp_or_stuck", "打印介质受潮或粘合在一起", "介质受潮或粘合"),
                _concept("multi_feed_internal_fault", "内部部件故障", "打印机内部部件故障"),
                _concept(
                    "separate_media_for_static",
                    "将打印介质重新分离，可以消除部分静电",
                    "重新分离介质以消除静电",
                ),
                _concept("recommended_media", "建议使用推荐打印介质", "使用推荐的打印介质"),
                _concept(
                    "dry_media_or_reseparate",
                    "重新将打印介质分离或使用更好的干燥打印介质",
                    "重新分离介质或改用更干燥的介质",
                ),
            ]
        },
        "evidence_ids": ["ev_c9895eb1d2eb948d42cb6b092ab0031d3cc40919"],
        "evidence_match_terms": {
            "ev_c9895eb1d2eb948d42cb6b092ab0031d3cc40919": [
                "打印多页进纸",
                "打印介质含静电量过大",
                "将打印介质重新分离,可以消除部分静电",
                "使用更好的干燥打印介质",
            ]
        },
        "binding_overrides": {
            "ev_c9895eb1d2eb948d42cb6b092ab0031d3cc40919": {
                "visual_pages": [85],
                "integrity": "active_index_complete",
                "note": "活动chunk完整保留打印多页进纸行的原因与解决办法；替换题不再要求无效的跨行共同点。",
            }
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": [
                    "multi_feed_static",
                    "multi_feed_damp_or_stuck",
                    "multi_feed_internal_fault",
                ],
            },
            {
                "branch_id": "b02",
                "concept_keys": [
                    "separate_media_for_static",
                    "recommended_media",
                    "dry_media_or_reseparate",
                ],
            },
        ],
        "critical_constraints": {},
        "difficulty_equivalence": "替换题仍需跨故障表的原因列和解决办法列完整汇总六个语义要点，保留hard级多分支完整性要求。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-009": {
        "disposition": "semantic_replacement",
        "blocker_reason": "旧题要求活动证据中不存在的“打印全黑版”原因，并据此继续比较；reference混入无源原因和共通性结论。",
        "source_review_reason": "“全黑版”分支完全无证据；黑色竖条行的原因与解决办法在活动chunk中直接可答。",
        "source_facts": [
            {
                "statement": "黑色竖条行直接列出碳粉盒、反光玻璃、进纸通道、介质规格和转印电压方面的原因，以及四项处理措施。",
                "pdf_pages": [86],
                "evidence_ids": ["ev_f0e6029144c83f20f961249e2fdc5c60e757470b"],
            }
        ],
        "query": "出现黑色竖条时，手册列出的故障原因和解决办法分别有哪些？",
        "reference_answer": "原因包括激光碳粉盒脏污、激光碳粉盒内部部件损坏、机器内部激光器反光玻璃脏污、进纸通道脏污、使用了不符合规格（如受潮或太粗糙）的打印介质，以及打印内部转印电压异常。解决办法是清洁机器背部激光器反光玻璃、清洁打印机背部进纸通道、使用规格范围内的打印介质，并清洁机器内部进纸通道。",
        "expected_terms": [
            "激光碳粉盒脏污",
            "激光碳粉盒内部部件损坏",
            "机器内部激光器反光玻璃脏污",
            "进纸通道脏污",
            "不符合使用规格的打印介质",
            "打印内部转印电压异常",
            "清洁机器背部激光器反光玻璃",
            "清洁打印机背部进纸通道",
            "使用规格范围内的打印介质",
            "清洁机器内部进纸通道",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept("black_stripe_toner_dirty", "激光碳粉盒脏污", "碳粉盒脏污"),
                _concept(
                    "black_stripe_toner_damage", "激光碳粉盒内部部件损坏", "碳粉盒内部部件损坏"
                ),
                _concept(
                    "black_stripe_reflector_dirty",
                    "机器内部激光器反光玻璃脏污",
                    "激光器反光玻璃脏污",
                ),
                _concept("black_stripe_feed_path_dirty", "进纸通道脏污", "机器进纸通道脏污"),
                _concept(
                    "black_stripe_media_out_of_spec",
                    "使用了不符合使用规格的打印介质",
                    "打印介质不符合规格",
                ),
                _concept(
                    "black_stripe_transfer_voltage", "打印内部转印电压异常", "内部转印电压异常"
                ),
                _concept(
                    "clean_rear_reflector", "清洁机器背部激光器反光玻璃", "清洁背部激光器反光玻璃"
                ),
                _concept("clean_rear_feed_path", "清洁打印机背部进纸通道", "清洁背部进纸通道"),
                _concept(
                    "use_in_spec_media", "请使用规格范围内的打印介质", "使用符合规格的打印介质"
                ),
                _concept("clean_internal_feed_path", "清洁机器内部进纸通道", "清洁内部进纸通道"),
            ]
        },
        "evidence_ids": ["ev_f0e6029144c83f20f961249e2fdc5c60e757470b"],
        "evidence_match_terms": {
            "ev_f0e6029144c83f20f961249e2fdc5c60e757470b": [
                "黑色竖条",
                "机器内部激光器反光玻璃脏污",
                "打印内部转印电压异常",
                "清洁机器背部激光器反光玻璃",
                "清洁机器内部进纸通道",
            ]
        },
        "binding_overrides": {
            "ev_f0e6029144c83f20f961249e2fdc5c60e757470b": {
                "visual_pages": [86],
                "integrity": "active_index_complete",
                "note": "活动chunk直接包含黑色竖条行的全部目标原因和解决办法；替换题彻底移除无源的全黑版分支。",
            }
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": [
                    "black_stripe_toner_dirty",
                    "black_stripe_toner_damage",
                    "black_stripe_reflector_dirty",
                    "black_stripe_feed_path_dirty",
                    "black_stripe_media_out_of_spec",
                    "black_stripe_transfer_voltage",
                ],
            },
            {
                "branch_id": "b02",
                "concept_keys": [
                    "clean_rear_reflector",
                    "clean_rear_feed_path",
                    "use_in_spec_media",
                    "clean_internal_feed_path",
                ],
            },
        ],
        "critical_constraints": {},
        "difficulty_equivalence": "替换题仍需在噪声较多的图像缺陷表行中区分原因与解决办法并完整汇总十个要点，保留hard级表格解析负担。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-011": {
        "disposition": "semantic_replacement",
        "blocker_reason": "A4、80 g/m²普通纸同时满足标准纸盒和多功能纸盒规格；旧题未给打印量却要求唯一选盒，并把容量差推断为“不适合大批量”。",
        "source_review_reason": "证据只提供介质规格和容量数值，没有“大批量”阈值；可直接比较两纸盒的适配性与容量。",
        "source_facts": [
            {
                "statement": "标准纸盒和多功能纸盒均支持A4普通纸，80 g/m²落在两者克重范围内。",
                "pdf_pages": [22],
                "evidence_ids": ["ev_a4337cb04380459fd3ae4238ec0cfa8cde6acb85"],
            },
            {
                "statement": "标准纸盒克重范围60至105 g/m²、容量250页；多功能纸盒克重范围60至200 g/m²、容量60页，因此标准纸盒容量更大。",
                "pdf_pages": [22],
                "evidence_ids": ["ev_a4337cb04380459fd3ae4238ec0cfa8cde6acb85"],
            },
        ],
        "query": "标准纸盒和多功能纸盒是否都支持A4、80 g/m²普通纸？两者的介质克重范围和最大容量分别是多少，哪个纸盒容量更大？",
        "reference_answer": "两种纸盒都支持A4、80 g/m²普通纸。标准纸盒的介质克重范围为60~105 g/m²，最大容量250页；多功能纸盒的介质克重范围为60~200 g/m²，最大容量60页。标准纸盒容量更大。",
        "expected_terms": [
            "两种纸盒都支持",
            "60~105 g/m²",
            "250页",
            "60~200 g/m²",
            "60页",
            "标准纸盒容量更大",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "both_trays_support_a4_80_plain",
                    "两种纸盒都支持A4、80 g/m²普通纸",
                    "标准纸盒和多功能纸盒均支持A4 80克普通纸",
                ),
                _concept(
                    "standard_tray_weight",
                    "标准纸盒介质克重60~105 g/m²",
                    "标准纸盒支持60至105克每平方米",
                ),
                _concept("standard_tray_capacity", "标准纸盒最大容量250页", "标准纸盒可装250页"),
                _concept(
                    "multipurpose_tray_weight",
                    "多功能纸盒介质克重60~200 g/m²",
                    "多功能纸盒支持60至200克每平方米",
                ),
                _concept(
                    "multipurpose_tray_capacity", "多功能纸盒最大容量60页", "多功能纸盒可装60页"
                ),
                _concept(
                    "standard_capacity_larger",
                    "标准纸盒容量更大",
                    "标准纸盒的最大容量高于多功能纸盒",
                ),
            ]
        },
        "evidence_ids": ["ev_a4337cb04380459fd3ae4238ec0cfa8cde6acb85"],
        "evidence_match_terms": {
            "ev_a4337cb04380459fd3ae4238ec0cfa8cde6acb85": [
                "标准纸盒 | 介质尺寸 | 自动*、A4",
                "标准纸盒 | 介质克重 | 60~105 g/m2",
                "标准纸盒 | 纸盒最大容量 | 250页",
                "多功能纸盒 | 介质克重 | 60~200 g/m2",
                "多功能纸盒 | 纸盒最大容量 | 60页",
            ]
        },
        "binding_overrides": {
            "ev_a4337cb04380459fd3ae4238ec0cfa8cde6acb85": {
                "visual_pages": [22],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含两纸盒的介质类型、A4尺寸、克重范围和容量；替换题只要求直接适配判断及数值比较。",
            }
        },
        "required_branches": [
            {"branch_id": "b01", "concept_keys": ["both_trays_support_a4_80_plain"]},
            {
                "branch_id": "b02",
                "concept_keys": ["standard_tray_weight", "standard_tray_capacity"],
            },
            {
                "branch_id": "b03",
                "concept_keys": ["multipurpose_tray_weight", "multipurpose_tray_capacity"],
            },
            {"branch_id": "b04", "concept_keys": ["standard_capacity_larger"]},
        ],
        "critical_constraints": {"required_numerics": [60, 105, 200, 250]},
        "difficulty_equivalence": "替换题仍要求跨同一规格表的类型、尺寸、克重和容量列完成四分支比较，保留hard级多字段综合。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-008": {
        "disposition": "corrected",
        "blocker_reason": "旧参考发生跨行错配，并虚构页面歪斜与周期性痕迹共享解决办法。",
        "source_facts": [
            {
                "statement": "页面歪斜的原因是打印介质未正确安装、机器进纸通道脏污。",
                "pdf_pages": [87],
                "evidence_ids": ["ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd"],
            },
            {
                "statement": "周期性痕迹应清洁或更换激光碳粉盒，并联系奔图客服更换定影组件。",
                "pdf_pages": [87],
                "evidence_ids": ["ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd"],
            },
            {
                "statement": "两类故障的源解决办法没有重叠，处理对象分别是介质/进纸通道与碳粉盒/定影组件。",
                "pdf_pages": [87],
                "evidence_ids": ["ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd"],
            },
        ],
        "query": "页面歪斜的故障原因有哪些？若出现周期性痕迹，应采取什么解决办法？两者在解决办法上是否存在重叠？",
        "reference_answer": "页面歪斜的原因是未正确安装打印介质、机器进纸通道脏污。周期性痕迹应清洁或更换新激光碳粉盒，并联系奔图客服维修更换新的定影组件。两者的解决办法没有重叠：前者针对介质安装和进纸通道，后者针对碳粉盒和定影组件。",
        "expected_terms": [
            "未正确安装打印介质",
            "机器进纸通道脏污",
            "清洁或更换新激光碳粉盒",
            "更换新的定影组件",
            "没有重叠",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept("skew_media_install", "未正确安装打印介质", "打印介质安装不正确"),
                _concept("skew_feed_path", "机器进纸通道脏污", "进纸通道脏污"),
                _concept(
                    "periodic_toner_action",
                    "清洁或更换新激光碳粉盒",
                    "清洁或更换碳粉盒",
                ),
                _concept(
                    "periodic_fuser_action",
                    "联系奔图客服维修更换新的定影组件",
                    "联系奔图客服更换定影组件",
                ),
                _concept("solution_overlap", "没有重叠", "解决办法不重叠", "两者处理对象不同"),
            ]
        },
        "evidence_ids": ["ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd"],
        "evidence_match_terms": {
            "ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd": [
                "未正确安装打印介质",
                "机器进纸通道脏污",
                "清洁或更换新激光碳粉盒",
                "更换新的定影组件",
            ]
        },
        "binding_overrides": {
            "ev_7aa7ff92a8f1f7536c5bcd7560f48d004672babd": {
                "visual_pages": [87],
                "integrity": "usable_merged_chunk",
                "note": "chunk声明页为86，目标表格行实际位于PDF第87页；旧第二chunk不含本题事实，已从提案中移除。",
            }
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": ["skew_media_install", "skew_feed_path"],
            },
            {
                "branch_id": "b02",
                "concept_keys": ["periodic_toner_action", "periodic_fuser_action"],
            },
            {"branch_id": "b03", "concept_keys": ["solution_overlap"]},
        ],
        "critical_constraints": {"required_polarity": ["不存在重叠"]},
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-015": {
        "disposition": "corrected",
        "blocker_reason": "源文明确阴影部分为非打印区域，旧参考却把白色空白区域判为非打印区域。",
        "source_facts": [
            {
                "statement": "图中阴影部分表示非打印区域，位于内外矩形边界之间的边缘带。",
                "pdf_pages": [28],
                "evidence_ids": ["ev_c53f37b3138ccaf59c675c3df8719977e4efd1f3"],
            },
            {
                "statement": "打印首选项选择省墨模式时，输出颜色为较淡。",
                "pdf_pages": [59],
                "evidence_ids": ["ev_30a38253df74711824964334f75ba970ae2c5a21"],
            },
        ],
        "query": "在打印文档时，非打印区域位于何处？若希望输出颜色较淡，应在打印首选项中选择何种模式？请结合图示与界面设置说明作答。",
        "reference_answer": "非打印区域是图中内外矩形边界之间的阴影部分；若希望输出颜色较淡，应在打印首选项中选择“省墨”模式。",
        "expected_terms": ["阴影部分", "省墨", "较淡"],
        "forbidden_terms": ["白色空白区域是非打印区域"],
        "concept_groups": {
            "all_of": [
                _concept("non_print_area", "阴影部分", "内外矩形边界之间的阴影区域"),
                _concept("toner_save_mode", "省墨模式", "省墨"),
                _concept("lighter_output", "输出颜色较淡", "颜色较淡"),
            ]
        },
        "evidence_ids": [
            "ev_c53f37b3138ccaf59c675c3df8719977e4efd1f3",
            "ev_30a38253df74711824964334f75ba970ae2c5a21",
        ],
        "evidence_match_terms": {
            "ev_c53f37b3138ccaf59c675c3df8719977e4efd1f3": ["阴影部分表示非打印区域"],
            "ev_30a38253df74711824964334f75ba970ae2c5a21": [
                "打印首选项",
                "省墨",
                "较淡",
            ],
        },
        "binding_overrides": {
            "ev_c53f37b3138ccaf59c675c3df8719977e4efd1f3": {
                "visual_pages": [28],
                "integrity": "text_authoritative_visual_caption_conflict",
                "note": "chunk正文的‘阴影部分表示非打印区域’正确；自动图像描述误称中间区域为白色空白，不得用该描述反转正文。",
            },
            "ev_30a38253df74711824964334f75ba970ae2c5a21": {
                "visual_pages": [59],
                "integrity": "usable",
                "note": "正文和界面均明确省墨模式与较淡输出。",
            },
        },
        "required_branches": [
            {"branch_id": "b01", "concept_keys": ["non_print_area"]},
            {
                "branch_id": "b02",
                "concept_keys": ["toner_save_mode", "lighter_output"],
            },
        ],
        "critical_constraints": {
            "required_polarity": ["阴影区域是非打印区域"],
            "forbidden_claims": ["白色空白区域是非打印区域"],
        },
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-021": {
        "disposition": "replacement",
        "blocker_reason": "源文分别说明IPv6地址类型和无线拓扑，但没有建立原题要求的一一模式映射。",
        "source_facts": [
            {
                "statement": "IPv6本地链路地址自行配置并以FE80开头；有状态地址由DHCPv6服务器配置；无状态地址由路由器自动配置。",
                "pdf_pages": [40],
                "evidence_ids": ["ev_fe59ce7fd78c8da261d5c26ad45f82e91651d528"],
            },
            {
                "statement": "基础结构模式通过接入点连接设备；Wi-Fi热点模式由计算机直接无线连接打印机热点。",
                "pdf_pages": [41],
                "evidence_ids": ["ev_2d9e4f1c45df7a19098ae3cf12e7c8ef1ed1df12"],
            },
            {
                "statement": "手册没有把任一IPv6地址类型排他地绑定到某一种无线连接模式。",
                "pdf_pages": [40, 41],
                "evidence_ids": [
                    "ev_fe59ce7fd78c8da261d5c26ad45f82e91651d528",
                    "ev_2d9e4f1c45df7a19098ae3cf12e7c8ef1ed1df12",
                ],
            },
        ],
        "query": "手册分别列出三种IPv6地址类型和两种无线连接模式。请说明三种IPv6地址的获得方式或识别特征，并概括基础结构与Wi-Fi热点模式的连接拓扑；手册是否明确规定某一种地址类型只属于某一种无线模式？",
        "reference_answer": "IPv6本地链路地址由设备自行配置，以FE80开头；有状态地址由DHCPv6服务器配置，网络中没有DHCPv6服务器时不能分配；无状态地址由网络路由器自动配置。基础结构模式通过接入点（无线路由器）连接无线网络打印机和计算机；Wi-Fi热点模式由计算机直接通过无线网络连接打印机热点。手册没有把某一种IPv6地址类型排他地绑定到某一种无线模式，因此不能作一一映射。",
        "expected_terms": [
            "FE80",
            "DHCPv6服务器",
            "网络路由器自动配置",
            "接入点",
            "打印机热点",
            "不能作一一映射",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "local_link", "自行配置的本地IPv6地址，以FE80开头", "本地链路地址以FE80开头"
                ),
                _concept("stateful", "有状态地址由DHCPv6服务器配置", "DHCPv6服务器配置有状态地址"),
                _concept("stateless", "无状态地址由网络路由器自动配置", "路由器自动配置无状态地址"),
                _concept(
                    "infrastructure_topology",
                    "基础结构模式通过接入点连接设备",
                    "通过无线路由器连接打印机和计算机",
                ),
                _concept(
                    "hotspot_topology",
                    "计算机直接无线连接打印机热点",
                    "Wi-Fi热点模式直接连接打印机",
                ),
                _concept(
                    "no_exclusive_mapping",
                    "不能作一一映射",
                    "手册未规定地址类型只属于某一种无线模式",
                ),
            ]
        },
        "evidence_ids": [
            "ev_fe59ce7fd78c8da261d5c26ad45f82e91651d528",
            "ev_2d9e4f1c45df7a19098ae3cf12e7c8ef1ed1df12",
        ],
        "evidence_match_terms": {
            "ev_fe59ce7fd78c8da261d5c26ad45f82e91651d528": [
                "IPv6本地链路地址",
                "FE80",
                "有状态地址",
                "DHCPv6服务器",
                "无状态地址",
                "网络路由器自动配置",
            ],
            "ev_2d9e4f1c45df7a19098ae3cf12e7c8ef1ed1df12": [
                "基础结构连接模式",
                "Wi-Fi热点连接模式",
                "接入点",
                "打印机热点",
            ],
        },
        "binding_overrides": {
            "ev_fe59ce7fd78c8da261d5c26ad45f82e91651d528": {
                "visual_pages": [40],
                "integrity": "usable",
                "note": "IPv6三类地址定义为正文。",
            },
            "ev_2d9e4f1c45df7a19098ae3cf12e7c8ef1ed1df12": {
                "visual_pages": [41],
                "integrity": "usable_merged_chunk",
                "note": "chunk声明页为40，拓扑图和标签实际位于PDF第41页。",
            },
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": ["local_link", "stateful", "stateless"],
            },
            {
                "branch_id": "b02",
                "concept_keys": ["infrastructure_topology", "hotspot_topology"],
            },
            {"branch_id": "b03", "concept_keys": ["no_exclusive_mapping"]},
        ],
        "critical_constraints": {
            "required_identifiers": ["FE80", "DHCPv6"],
            "required_polarity": ["没有排他一一映射"],
        },
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-037": {
        "disposition": "replacement",
        "blocker_reason": "激光标签关键标识符在当前活动chunk中被误OCR，无法仅修参考而保持索引可回答；替换为两个现有完整环保正文chunk的路径比较题。",
        "source_facts": [
            {
                "statement": "法规信息要求把废弃设备送到指定收集点，以回收利用废弃的电气和电子设备。",
                "pdf_pages": [7],
                "evidence_ids": ["ev_70e1da742ddd8d2b506890422625113c71836783"],
            },
            {
                "statement": "奔图回收体系允许把废弃打印机和耗材交给当地奔图维修中心统一回收。",
                "pdf_pages": [4],
                "evidence_ids": ["ev_82c71a223525922e9158416347c996f49c5f6a94"],
            },
            {
                "statement": "维修中心之后把废弃产品交由国家认定、具备废弃物处理资质的机构正确回收和再利用。",
                "pdf_pages": [4],
                "evidence_ids": ["ev_82c71a223525922e9158416347c996f49c5f6a94"],
            },
        ],
        "query": "手册给出两条废弃物处理路径：法规信息对一般废弃设备要求送往哪里、目的是什么？奔图回收体系中，废弃打印机和耗材先交给哪里，之后由何类机构处理？两条路径体现的共同处理原则是什么？",
        "reference_answer": "法规信息要求将废弃设备送到指定的收集点，以回收利用废弃的电气和电子设备。奔图回收体系中，废弃打印机和耗材先交给当地奔图维修中心统一回收，再由国家认定、具备废弃物处理资质的机构进行正确回收和再利用。两条路径都要求避免随意混同丢弃，将废弃物交给指定或具资质的渠道集中回收利用。",
        "expected_terms": [
            "指定的收集点",
            "回收利用废弃的电气和电子设备",
            "当地奔图维修中心",
            "具备废弃物处理资质的机构",
            "集中回收利用",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "general_collection_point",
                    "送到指定的收集点",
                    "交给指定收集点",
                ),
                _concept(
                    "general_recycling_purpose",
                    "回收利用废弃的电气和电子设备",
                    "回收废弃电气电子设备",
                ),
                _concept(
                    "pantum_first_hop",
                    "当地奔图维修中心",
                    "奔图维修中心",
                ),
                _concept(
                    "pantum_second_hop",
                    "国家认定、具备废弃物处理资质的机构",
                    "具备废弃物处理资质的处理机构",
                ),
                _concept(
                    "shared_principle",
                    "交给指定或具资质的渠道集中回收利用",
                    "不得随意丢弃，应通过指定渠道回收",
                ),
            ]
        },
        "evidence_ids": [
            "ev_70e1da742ddd8d2b506890422625113c71836783",
            "ev_82c71a223525922e9158416347c996f49c5f6a94",
        ],
        "evidence_match_terms": {
            "ev_70e1da742ddd8d2b506890422625113c71836783": [
                "送到指定的收集点",
                "回收利用废弃的电气和电子设备",
            ],
            "ev_82c71a223525922e9158416347c996f49c5f6a94": [
                "当地奔图维修中心",
                "具备废弃物处理资质的处理机构",
                "回收、再利用处理",
            ],
        },
        "binding_overrides": {
            "ev_70e1da742ddd8d2b506890422625113c71836783": {
                "visual_pages": [7],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含不得随意丢弃、指定收集点和回收目的。",
            },
            "ev_82c71a223525922e9158416347c996f49c5f6a94": {
                "visual_pages": [4],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含维修中心、具资质处理机构及回收再利用路径。",
            },
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": [
                    "general_collection_point",
                    "general_recycling_purpose",
                ],
            },
            {
                "branch_id": "b02",
                "concept_keys": ["pantum_first_hop", "pantum_second_hop"],
            },
            {"branch_id": "b03", "concept_keys": ["shared_principle"]},
        ],
        "critical_constraints": {},
        "difficulty_equivalence": "原题是跨两类证据的双分支综合题；替换题仍需联合两个不同页面/chunk，重建两段处理路径并抽象共同原则，分支数和综合推理负荷保持hard。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-040": {
        "disposition": "replacement",
        "blocker_reason": "原题所需RoHS矩阵数据行未进入当前活动chunk；为避免重建索引，改用两个现有完整正文/表格chunk的标准适用范围综合题。",
        "source_facts": [
            {
                "statement": "法规信息说明产品符合EN 55022的B级范围和EN 60950的安全要求。",
                "pdf_pages": [7],
                "evidence_ids": ["ev_1b3eb2878fb886a49d96992d9063ab2a5f185fe3"],
            },
            {
                "statement": "激光碳粉盒寿命表的平均打印量基于ISO/IEC 19752：标准容量约3000页、高容量约6000页、超大容量约10000页。",
                "pdf_pages": [15],
                "evidence_ids": ["ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233"],
            },
        ],
        "query": "手册在法规信息和激光碳粉盒寿命表中分别引用EN 55022、EN 60950和ISO/IEC 19752。请说明前两个标准在手册中支持什么结论，ISO/IEC 19752用于什么数据并列出三种容量的平均打印量；这些标准能否相互替代？",
        "reference_answer": "EN 55022支持本产品符合B级范围的结论，EN 60950支持产品安全要求。ISO/IEC 19752是激光碳粉盒平均打印量的数据依据：标准容量约3000页、高容量约6000页、超大容量约10000页。它们不能相互替代，因为前两项用于产品范围/安全合规，ISO/IEC 19752用于耗材打印量测定，适用对象不同。",
        "expected_terms": [
            "EN 55022的B级范围",
            "EN 60950的安全要求",
            "ISO/IEC 19752",
            "约3000页",
            "约6000页",
            "约10000页",
            "不能相互替代",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "en_55022_scope",
                    "符合EN 55022的B级范围",
                    "EN 55022 B级范围",
                ),
                _concept(
                    "en_60950_safety",
                    "符合EN 60950的安全要求",
                    "EN 60950安全要求",
                ),
                _concept(
                    "iso_yield_basis",
                    "ISO/IEC 19752是平均打印量的数据依据",
                    "打印量基于ISO/IEC 19752",
                ),
                _concept(
                    "three_capacity_yields",
                    "标准容量约3000页、高容量约6000页、超大容量约10000页",
                    "3000页、6000页、10000页",
                ),
                _concept(
                    "standards_not_interchangeable",
                    "不能相互替代，适用对象不同",
                    "产品合规标准与耗材打印量标准不能替代",
                ),
            ]
        },
        "evidence_ids": [
            "ev_1b3eb2878fb886a49d96992d9063ab2a5f185fe3",
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233",
        ],
        "evidence_match_terms": {
            "ev_1b3eb2878fb886a49d96992d9063ab2a5f185fe3": [
                "EN 55022",
                "B 级范围",
                "EN 60950",
                "安全要求",
            ],
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233": [
                "ISO/IEC 19752",
                "约 3000 页",
                "约 6000 页",
                "约 10000 页",
            ],
        },
        "binding_overrides": {
            "ev_1b3eb2878fb886a49d96992d9063ab2a5f185fe3": {
                "visual_pages": [7],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含EN 55022 B级范围和EN 60950安全要求。",
            },
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233": {
                "visual_pages": [15],
                "integrity": "active_index_complete",
                "note": "活动chunk完整包含ISO/IEC 19752依据及三种容量的平均打印量。",
            },
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": [
                    "en_55022_scope",
                    "en_60950_safety",
                ],
            },
            {
                "branch_id": "b02",
                "concept_keys": ["iso_yield_basis", "three_capacity_yields"],
            },
            {
                "branch_id": "b03",
                "concept_keys": ["standards_not_interchangeable"],
            },
        ],
        "critical_constraints": {
            "required_numerics": [3000, 6000, 10000],
            "required_identifiers": ["EN 55022", "EN 60950", "ISO/IEC 19752"],
            "required_polarity": ["不能相互替代"],
        },
        "difficulty_equivalence": "原题是双证据多分支综合；替换题仍需联合两个不同页面/chunk，保持三个标准标识符、三个关键数值、范围映射和否定性比较结论，约束密度不低于原hard题。",
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-041": {
        "disposition": "replacement",
        "blocker_reason": "原题把后视图编号1预设为通风口，但源表明确编号1是USB接口；原题的‘理想条件’也无判定合同。",
        "source_facts": [
            {
                "statement": "产品视图编号4是前盖，打开前盖可取出激光碳粉盒。",
                "pdf_pages": [13],
                "evidence_ids": ["ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6"],
            },
            {
                "statement": "高容量激光碳粉盒平均打印量约6000页。",
                "pdf_pages": [15],
                "evidence_ids": ["ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233"],
            },
            {
                "statement": "超大容量激光碳粉盒平均打印量约10000页；打印量数据基于ISO/IEC 19752。",
                "pdf_pages": [15],
                "evidence_ids": ["ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233"],
            },
        ],
        "query": "若维修人员需要从机身取出激光碳粉盒，并比较高容量与超大容量碳粉盒的平均打印量，应操作产品视图中的哪个部件、该部件有什么用途？两种容量的平均打印量各是多少，数据依据什么标准？",
        "reference_answer": "应操作产品视图中的编号4前盖；打开前盖可取出激光碳粉盒。高容量激光碳粉盒平均打印量约6000页，超大容量激光碳粉盒约10000页，以上数据基于ISO/IEC 19752标准。",
        "expected_terms": [
            "编号4前盖",
            "取出激光碳粉盒",
            "约6000页",
            "约10000页",
            "ISO/IEC 19752",
        ],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept("access_component", "编号4前盖", "前盖"),
                _concept("access_function", "打开前盖可取出激光碳粉盒", "取出激光碳粉盒"),
                _concept("high_capacity", "高容量约6000页", "高容量激光碳粉盒平均打印量约6000页"),
                _concept(
                    "ultra_capacity", "超大容量约10000页", "超大容量激光碳粉盒平均打印量约10000页"
                ),
                _concept("yield_standard", "ISO/IEC 19752", "ISO IEC 19752"),
            ]
        },
        "evidence_ids": [
            "ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6",
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233",
        ],
        "evidence_match_terms": {
            "ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6": [
                "前盖",
                "可取出激光碳粉盒",
            ],
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233": [
                "高容量激光碳粉盒",
                "约 6000 页",
                "超大容量激光碳粉盒",
                "约 10000 页",
                "ISO/IEC 19752",
            ],
        },
        "binding_overrides": {
            "ev_cdb645f23ba94eaf3de759176aea0bc17c8986b6": {
                "visual_pages": [13],
                "integrity": "usable",
                "note": "产品视图编号、部件名称和用途表完整。",
            },
            "ev_9afc450d3fbb7fc5a1ec87df11e33dc94a275233": {
                "visual_pages": [15],
                "integrity": "usable",
                "note": "三种容量及平均打印量表完整；提案仅使用高容量和超大容量两行。",
            },
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": ["access_component", "access_function"],
            },
            {
                "branch_id": "b02",
                "concept_keys": ["high_capacity", "ultra_capacity", "yield_standard"],
            },
        ],
        "critical_constraints": {
            "required_numerics": [6000, 10000],
            "required_identifiers": ["4", "ISO/IEC 19752"],
        },
        "import_gate": "ready_after_suite_v2_review",
    },
    "P3500-HARD-042": {
        "disposition": "corrected",
        "blocker_reason": "旧参考只复述打印量和IP，没有回答是否处于可接受工作负载范围。",
        "source_facts": [
            {
                "statement": "推荐月打印量为750至4000页，每月工作负荷为50000页。",
                "pdf_pages": [18],
                "evidence_ids": ["ev_1b5a1b1cccfcbe046898717ba0b26b5960f4f7e6"],
            },
            {
                "statement": "网络配置页的无线IPv4地址为192.168.0.102。",
                "pdf_pages": [19],
                "evidence_ids": ["ev_1cd266d7d173f9d3fa0218dc1ecad2d25bb48a8a"],
            },
            {
                "statement": "手册没有给出IP地址影响工作负载阈值的规则；工作负载判断使用的是页数指标。",
                "pdf_pages": [18, 19],
                "evidence_ids": [
                    "ev_1b5a1b1cccfcbe046898717ba0b26b5960f4f7e6",
                    "ev_1cd266d7d173f9d3fa0218dc1ecad2d25bb48a8a",
                ],
            },
        ],
        "query": "根据演示页中列出的50,000页每月工作负荷和750至4,000页推荐月打印量，以及网络配置信息页中显示的无线IP地址为192.168.0.102，若某用户在无线网络下使用该打印机进行连续打印，其实际打印量接近推荐上限时，应如何判断设备是否仍处于可接受的工作负载范围内？",
        "reference_answer": "实际月打印量接近但不超过4000页时，仍在750至4000页的推荐月打印量范围内，并且远低于50000页的每月工作负荷，因此仍处于可接受范围。无线IP地址192.168.0.102只说明网络配置，与工作负载判断无关。",
        "expected_terms": ["仍在推荐范围内", "远低于50000页", "IP地址与工作负载判断无关"],
        "forbidden_terms": [],
        "concept_groups": {
            "all_of": [
                _concept(
                    "recommended_range",
                    "接近但不超过4000页仍在750至4000页推荐范围内",
                    "仍在推荐月打印量范围内",
                ),
                _concept(
                    "duty_cycle_margin", "远低于50000页每月工作负荷", "未超过50000页月工作负荷"
                ),
                _concept(
                    "ip_irrelevance", "IP地址与工作负载判断无关", "无线IP不影响打印量阈值判断"
                ),
            ]
        },
        "evidence_ids": [
            "ev_1b5a1b1cccfcbe046898717ba0b26b5960f4f7e6",
            "ev_1cd266d7d173f9d3fa0218dc1ecad2d25bb48a8a",
        ],
        "evidence_match_terms": {
            "ev_1b5a1b1cccfcbe046898717ba0b26b5960f4f7e6": [
                "50,000 page monthly duty cycle",
                "750 to 4,000 recommended monthly page volume",
            ],
            "ev_1cd266d7d173f9d3fa0218dc1ecad2d25bb48a8a": [
                "无线网络 IPv4 信息",
                "192.168.0.102",
            ],
        },
        "binding_overrides": {
            "ev_1b5a1b1cccfcbe046898717ba0b26b5960f4f7e6": {
                "visual_pages": [18],
                "integrity": "usable",
                "note": "月工作负荷和推荐月打印量位于演示页。",
            },
            "ev_1cd266d7d173f9d3fa0218dc1ecad2d25bb48a8a": {
                "visual_pages": [19],
                "integrity": "usable",
                "note": "无线IPv4地址位于网络配置信息页；它是题干干扰信息，不是工作负载阈值。",
            },
        },
        "required_branches": [
            {
                "branch_id": "b01",
                "concept_keys": ["recommended_range", "duty_cycle_margin"],
            },
            {"branch_id": "b02", "concept_keys": ["ip_irrelevance"]},
        ],
        "critical_constraints": {
            "required_numerics": [750, 4000, 50000],
            "required_polarity": ["仍在可接受范围"],
        },
        "import_gate": "ready_after_suite_v2_review",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_paths() -> dict[str, Path]:
    return {
        "datasets/easy.json": DATASET_DIR / "easy.json",
        "datasets/medium.json": DATASET_DIR / "medium.json",
        "datasets/hard.json": DATASET_DIR / "hard.json",
        "pdf-visual-review/cases.json": VISUAL_AUDIT,
        "source.pdf": SOURCE_PDF,
    }


def _validate_concept_groups(case_id: str, groups: object) -> None:
    assert isinstance(groups, dict) and set(groups) == {"all_of"}, case_id
    all_of = groups["all_of"]
    assert isinstance(all_of, list) and all_of, case_id
    keys: set[str] = set()
    signatures: set[tuple[str, ...]] = set()
    for group in all_of:
        assert isinstance(group, dict), case_id
        assert set(group) == {"concept_key", "any_of"}, (case_id, group)
        key = group["concept_key"]
        variants = group["any_of"]
        assert isinstance(key, str) and key.strip() and key not in keys, (case_id, key)
        assert isinstance(variants, list) and variants, (case_id, key)
        assert all(isinstance(item, str) and item.strip() for item in variants), key
        normalized = tuple(sorted(" ".join(item.split()).casefold() for item in variants))
        assert len(normalized) == len(set(normalized)), (case_id, key)
        assert normalized not in signatures, (case_id, key)
        keys.add(key)
        signatures.add(normalized)


def _build_catalog(
    datasets: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for payload in datasets.values():
        for case in payload["cases"]:
            case_id = str(case["case_id"])
            assert case_id not in cases
            cases[case_id] = case
            for item in case.get("relevant_evidence", []):
                evidence_id = str(item["evidence_id"])
                prior = evidence.get(evidence_id)
                if prior is not None:
                    identity_fields = ("chunk_id", "page", "text", "content_hash")
                    assert all(prior.get(field) == item.get(field) for field in identity_fields), (
                        evidence_id
                    )
                    continue
                evidence[evidence_id] = item
    return cases, evidence


def _build_proposed_case(
    *,
    original: dict[str, Any],
    definition: dict[str, Any],
    evidence_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    proposed = copy.deepcopy(original)
    proposed["query"] = definition["query"]
    proposed["reference_answer"] = definition["reference_answer"]
    proposed["expected_terms"] = definition["expected_terms"]
    proposed["forbidden_terms"] = definition["forbidden_terms"]
    selected_evidence = [
        copy.deepcopy(evidence_catalog[evidence_id]) for evidence_id in definition["evidence_ids"]
    ]
    evidence_match_terms = definition.get("evidence_match_terms", {})
    for item in selected_evidence:
        evidence_id = str(item["evidence_id"])
        if evidence_id in evidence_match_terms:
            item["match_terms"] = list(evidence_match_terms[evidence_id])
    proposed["relevant_evidence"] = selected_evidence
    proposed["relevant_chunk_ids"] = list(
        dict.fromkeys(str(item["chunk_id"]) for item in selected_evidence)
    )
    proposed["concept_groups"] = definition["concept_groups"]
    for tag in ("suite-v2", "blocker-remediation-proposal"):
        if tag not in proposed["tags"]:
            proposed["tags"].append(tag)
    metadata = proposed.setdefault("metadata", {})
    metadata["blocker_remediation_disposition"] = definition["disposition"]
    metadata["parent_case_id"] = original["case_id"]
    metadata["proposal_schema_version"] = SCHEMA_VERSION
    if definition["disposition"] == "semantic_replacement":
        metadata["lineage_semantic_replacement"] = True
    return proposed


def _run_project_manifest_validator(
    *,
    datasets: dict[str, dict[str, Any]],
    proposals: list[dict[str, Any]],
) -> None:
    workspace = str(WORKSPACE_ROOT)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    module = importlib.import_module("app.domain.evaluation")
    validator = module.validate_dataset_manifest
    by_difficulty: dict[str, dict[str, dict[str, Any]]] = {
        "easy": {},
        "medium": {},
        "hard": {},
    }
    for row in proposals:
        by_difficulty[row["difficulty"]][row["case_id"]] = row["proposed_case"]
    for difficulty, manifest in datasets.items():
        patched = copy.deepcopy(manifest)
        replacements = by_difficulty[difficulty]
        patched["cases"] = [replacements.get(case["case_id"], case) for case in patched["cases"]]
        validator(patched)
        assert len(patched["cases"]) == 50


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_paths = _input_paths()
    actual_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    assert actual_hashes == EXPECTED_INPUT_HASHES, actual_hashes

    datasets = {
        difficulty: _load_json(DATASET_DIR / f"{difficulty}.json")
        for difficulty in ("easy", "medium", "hard")
    }
    assert {name: len(payload["cases"]) for name, payload in datasets.items()} == {
        "easy": 50,
        "medium": 50,
        "hard": 50,
    }
    all_cases, evidence_catalog = _build_catalog(datasets)
    visual_rows = {row["case_id"]: row for row in _load_json(VISUAL_AUDIT)}
    assert set(DEFINITIONS) == TARGET_IDS
    assert (TARGET_IDS - REVIEWER_BLOCK_TARGET_IDS) <= set(visual_rows)
    assert REVIEWER_BLOCK_TARGET_IDS.isdisjoint(visual_rows)

    proposals: list[dict[str, Any]] = []
    for case_id in sorted(TARGET_IDS):
        definition = DEFINITIONS[case_id]
        original = all_cases[case_id]
        proposed_case = _build_proposed_case(
            original=original,
            definition=definition,
            evidence_catalog=evidence_catalog,
        )
        _validate_concept_groups(case_id, proposed_case["concept_groups"])
        concept_keys = {group["concept_key"] for group in proposed_case["concept_groups"]["all_of"]}
        branch_keys = {
            key for branch in definition["required_branches"] for key in branch["concept_keys"]
        }
        assert concept_keys == branch_keys, (case_id, concept_keys, branch_keys)
        assert proposed_case["case_id"] == original["case_id"]
        assert proposed_case["metadata"]["difficulty"] == original["metadata"]["difficulty"]
        if definition["disposition"] == "semantic_replacement":
            assert proposed_case["query"] != original["query"], case_id
            assert proposed_case["reference_answer"] != original["reference_answer"], case_id
            assert proposed_case["metadata"]["lineage_semantic_replacement"] is True
            assert definition.get("source_review_reason"), case_id
        configured_match_terms = definition["evidence_match_terms"]
        assert set(configured_match_terms) == set(definition["evidence_ids"]), case_id
        for evidence in proposed_case["relevant_evidence"]:
            evidence_text = "".join(str(evidence["text"]).split()).casefold()
            match_terms = evidence["match_terms"]
            assert match_terms, (case_id, evidence["evidence_id"])
            assert all("".join(term.split()).casefold() in evidence_text for term in match_terms), (
                case_id,
                evidence["evidence_id"],
                match_terms,
            )

        bindings: list[dict[str, Any]] = []
        for evidence_id in definition["evidence_ids"]:
            evidence = evidence_catalog[evidence_id]
            override = definition["binding_overrides"][evidence_id]
            bindings.append(
                {
                    "evidence_id": evidence_id,
                    "chunk_id": evidence["chunk_id"],
                    "declared_page": evidence["page"],
                    "visual_pages": override["visual_pages"],
                    "content_hash": evidence["content_hash"],
                    "integrity": override["integrity"],
                    "note": override["note"],
                }
            )

        visual_row = visual_rows.get(case_id)
        proposals.append(
            {
                "case_id": case_id,
                "difficulty": original["metadata"]["difficulty"],
                "suite_v2_target_selector": {"parent_case_id": case_id},
                "disposition": definition["disposition"],
                "blocker_reason": definition["blocker_reason"],
                "trigger": (
                    "full_suite_reviewer_block"
                    if case_id in REVIEWER_BLOCK_TARGET_IDS
                    else "pdf_visual_audit_block"
                ),
                "source_review_verdict": (
                    "block" if case_id in REVIEWER_BLOCK_TARGET_IDS else None
                ),
                "source_review_reason": definition.get("source_review_reason"),
                "visual_audit_disposition": (
                    visual_row["reference_disposition"]
                    if visual_row is not None
                    else "not_applicable_reviewer_block"
                ),
                "original_visual_pages_checked": (
                    visual_row["visual_pages_checked"] if visual_row is not None else []
                ),
                "source_facts": definition["source_facts"],
                "source_bindings": bindings,
                "difficulty_equivalence": definition.get("difficulty_equivalence"),
                "proposed_case": proposed_case,
                "contract_draft": {
                    "required_branches": definition["required_branches"],
                    "critical_constraints": definition["critical_constraints"],
                },
                "import_gate": {
                    "status": definition["import_gate"],
                    "requires_full_suite_same_prompt_review": True,
                    "requires_live_result_review": True,
                    "must_not_use_production_answers_for_authoring": True,
                },
                "audit": {
                    "original_case_sha256": _canonical_hash(original),
                    "proposed_case_sha256": _canonical_hash(proposed_case),
                    "semantic_replacement_payload_sha256": (
                        _canonical_hash(
                            {
                                "query": proposed_case["query"],
                                "reference_answer": proposed_case["reference_answer"],
                                "relevant_evidence": proposed_case["relevant_evidence"],
                                "concept_groups": proposed_case["concept_groups"],
                                "required_branches": definition["required_branches"],
                                "critical_constraints": definition["critical_constraints"],
                            }
                        )
                        if definition["disposition"] == "semantic_replacement"
                        else None
                    ),
                    "authoring_inputs": [
                        "source PDF pages",
                        "immutable dataset manifests",
                        (
                            "full-suite reviewer block reason"
                            if case_id in REVIEWER_BLOCK_TARGET_IDS
                            else "PDF visual audit sidecar"
                        ),
                    ],
                    "excluded_inputs": [
                        "production model answers",
                        "run pass/fail labels",
                        "per-case production metrics",
                    ],
                },
            }
        )

    disposition_counts = Counter(row["disposition"] for row in proposals)
    difficulty_counts = Counter(row["difficulty"] for row in proposals)
    assert disposition_counts == {
        "corrected": 3,
        "replacement": 6,
        "semantic_replacement": 4,
    }
    assert difficulty_counts == {"easy": 2, "hard": 11}
    assert len({row["case_id"] for row in proposals}) == 13
    assert all(row["import_gate"]["status"] == "ready_after_suite_v2_review" for row in proposals)
    assert all(
        binding["integrity"] != "visual_refresh_required"
        for row in proposals
        for binding in row["source_bindings"]
    )
    no_index_replacements = {
        row["case_id"]: row for row in proposals if row["case_id"] in NO_INDEX_REPLACEMENT_IDS
    }
    assert set(no_index_replacements) == NO_INDEX_REPLACEMENT_IDS
    assert all(
        row["disposition"] in {"replacement", "semantic_replacement"}
        and bool(row["difficulty_equivalence"])
        and all(
            binding["integrity"] == "active_index_complete" for binding in row["source_bindings"]
        )
        for row in no_index_replacements.values()
    )

    index_version_ids = {payload["source"]["index_version_id"] for payload in datasets.values()}
    frozen_chunk_counts = {
        int(payload["generation"]["source_evidence_count"]) for payload in datasets.values()
    }
    assert index_version_ids == {"idx_fb5eeac133f540f4af31004dbde23151"}
    assert frozen_chunk_counts == {218}

    _run_project_manifest_validator(datasets=datasets, proposals=proposals)

    patched_counts = {name: len(payload["cases"]) for name, payload in datasets.items()}
    assert patched_counts == {"easy": 50, "medium": 50, "hard": 50}
    post_build_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    assert post_build_hashes == actual_hashes

    validation = {
        "schema_version": "stt.eval.suite-v2-blocker-remediation-validation.v2-no-index-refresh",
        "status": "passed",
        "target_case_count": len(proposals),
        "target_case_ids": sorted(TARGET_IDS),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "full_suite_case_counts_preserved": patched_counts,
        "full_suite_total_preserved": sum(patched_counts.values()),
        "case_ids_preserved": True,
        "difficulty_preserved": True,
        "evidence_ids_resolve_to_immutable_inputs": True,
        "evidence_match_terms_present_in_source_text": True,
        "selected_chunks_resolve_to_frozen_active_index": True,
        "no_index_replacement_case_ids": sorted(NO_INDEX_REPLACEMENT_IDS),
        "active_index": {
            "index_version_id": next(iter(index_version_ids)),
            "frozen_chunk_count": next(iter(frozen_chunk_counts)),
            "index_rebuild_required": False,
            "evidence_refresh_required": False,
        },
        "concept_groups_schema_valid": True,
        "project_manifest_validator_passed": True,
        "all_concepts_assigned_to_branches": True,
        "semantic_replacement_case_ids": sorted(REVIEWER_BLOCK_TARGET_IDS),
        "semantic_replacement_query_and_reference_changed": True,
        "semantic_replacement_payload_hashes_present": True,
        "same_prompt_independent_rereview_required": True,
        "superseded_review_reuse_forbidden": True,
        "input_hashes_verified_before_and_after": actual_hashes,
        "write_scope": [str(THIS_DIR.relative_to(WORKSPACE_ROOT)).replace("\\", "/")],
        "network_or_llm_calls": False,
        "production_answers_read": False,
        "suite_v2_request_pack_modified": False,
        "application_or_evaluator_code_modified": False,
        "import_readiness": {
            "ready_after_suite_v2_review": sum(
                row["import_gate"]["status"] == "ready_after_suite_v2_review" for row in proposals
            ),
            "requires_index_evidence_refresh": sum(
                row["import_gate"]["status"] == "requires_index_evidence_refresh"
                for row in proposals
            ),
        },
    }
    return proposals, validation


def _write_csv(proposals: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "difficulty",
        "disposition",
        "blocker_reason",
        "proposed_query",
        "proposed_reference_answer",
        "pdf_pages",
        "chunk_ids",
        "evidence_ids",
        "difficulty_equivalence",
        "import_gate",
    ]
    with (THIS_DIR / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in proposals:
            pages = sorted({page for fact in row["source_facts"] for page in fact["pdf_pages"]})
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "difficulty": row["difficulty"],
                    "disposition": row["disposition"],
                    "blocker_reason": row["blocker_reason"],
                    "proposed_query": row["proposed_case"]["query"],
                    "proposed_reference_answer": row["proposed_case"]["reference_answer"],
                    "pdf_pages": json.dumps(pages, ensure_ascii=False),
                    "chunk_ids": json.dumps(
                        row["proposed_case"]["relevant_chunk_ids"], ensure_ascii=False
                    ),
                    "evidence_ids": json.dumps(
                        [item["evidence_id"] for item in row["source_bindings"]],
                        ensure_ascii=False,
                    ),
                    "difficulty_equivalence": row["difficulty_equivalence"] or "",
                    "import_gate": row["import_gate"]["status"],
                }
            )


def _write_report(proposals: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    corrected = [row["case_id"] for row in proposals if row["disposition"] == "corrected"]
    replaced = [row["case_id"] for row in proposals if row["disposition"] == "replacement"]
    semantic_replaced = [
        row["case_id"] for row in proposals if row["disposition"] == "semantic_replacement"
    ]
    refresh = [
        row["case_id"]
        for row in proposals
        if row["import_gate"]["status"] == "requires_index_evidence_refresh"
    ]
    assert not refresh
    lines = [
        "# suite v2 PDF blocker 修复提案",
        "",
        "## 结论",
        "",
        f"{len(proposals)} 个视觉或全套语义审查 blocker 已形成隔离提案：{len(corrected)} 题修正、{len(replaced)} 题替换、{len(semantic_replaced)} 题语义替换。所有提案保留原 case ID 和难度，因此 easy/medium/hard 仍各 50 题、总计 150 题。",
        "",
        f"可修正：{', '.join(corrected)}。",
        "",
        f"必须替换：{', '.join(replaced)}。",
        "",
        f"语义替换：{', '.join(semantic_replaced)}。",
        "",
        f"全部{len(proposals)}题均只绑定当前冻结218-chunk活动索引中已存在且文本完整的证据；无需重建索引或刷新证据即可进入suite-v2同prompt复审。",
        "",
        "## 隔离与诚实性约束",
        "",
        "- 本产物只读取原始PDF、不可变旧数据集和PDF视觉审计侧车。",
        "- 未读取任何生产模型答案、运行通过/失败标签或逐题生产指标。",
        "- 未调用网络或LLM，未修改 `app/`、旧 `datasets/`、evaluator 或正在运行的 suite-v2 request pack。",
        "- replacement/semantic_replacement 只在原题无确定源合同、嵌入错误前提或当前活动索引无法完整承载源事实时使用；没有把任何线上错题答案写成程序特判。",
        "",
        "## 逐题提案",
        "",
    ]
    for row in proposals:
        action = {
            "corrected": "修正",
            "replacement": "替换",
            "semantic_replacement": "语义替换",
        }[row["disposition"]]
        lines.extend(
            [
                f"### {row['case_id']} - {action}",
                "",
                f"- 阻断原因：{row['blocker_reason']}",
                f"- 触发来源：`{row['trigger']}`",
                f"- 提案问题：{row['proposed_case']['query']}",
                f"- 提案参考：{row['proposed_case']['reference_answer']}",
                f"- 导入门：`{row['import_gate']['status']}`",
            ]
        )
        if row["difficulty_equivalence"]:
            lines.append(f"- 难度等价：{row['difficulty_equivalence']}")
        lines.extend(["- 源事实：", ""])
        for fact in row["source_facts"]:
            evidence_ids = ", ".join(fact["evidence_ids"])
            pages = ", ".join(str(page) for page in fact["pdf_pages"])
            lines.append(f"  - {fact['statement']}（PDF p.{pages}; evidence {evidence_ids}）")
        lines.extend(["", "- 证据绑定：", ""])
        for binding in row["source_bindings"]:
            visual_pages = ", ".join(str(page) for page in binding["visual_pages"])
            lines.append(
                "  - "
                f"`{binding['chunk_id']}` / `{binding['evidence_id']}` / "
                f"声明页 {binding['declared_page']} / 视觉页 {visual_pages} / "
                f"`{binding['integrity']}`：{binding['note']}"
            )
        concepts = ", ".join(
            group["concept_key"] for group in row["proposed_case"]["concept_groups"]["all_of"]
        )
        lines.extend(["", f"- concept groups（all-of）：{concepts}。", ""])

    lines.extend(
        [
            "## 验证结果",
            "",
            f"- 状态：`{validation['status']}`",
            f"- 处置计数：{json.dumps(validation['disposition_counts'], ensure_ascii=False)}",
            f"- 难度计数：{json.dumps(validation['difficulty_counts'], ensure_ascii=False)}",
            f"- 全套题数：{validation['full_suite_total_preserved']}（easy/medium/hard 均为50）",
            "- case ID、难度、证据ID解析、concept_groups结构和分支覆盖均通过确定性校验。",
            f"- 将{len(proposals)}个提案以内存替换方式应用到旧manifest后，项目 `validate_dataset_manifest` 校验通过，未写回旧数据集。",
            f"- {len(proposals)}题所选chunk均解析到活动索引 `idx_fb5eeac133f540f4af31004dbde23151` 的冻结218-chunk证据集合；索引重建/证据刷新需求均为0。",
            "- 4个语义替换均同时改变query与reference，并生成新的proposed-case SHA和semantic-replacement payload SHA；旧review不得复用。",
            "- 输入文件在生成前后SHA256一致。",
            "",
            "## 使用限制",
            "",
            f"本目录是修复提案，不是已批准数据集。{len(proposals)}题都不需要索引重建或证据刷新，但仍须使用suite-v2相同prompt完成替换题独立复审，并在合并后重新执行全套确定性校验，才能进入候选集。",
            "",
        ]
    )
    (THIS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    proposals, validation = build()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "suite_key": "three-difficulty-live-2026-08-10",
        "source": {
            "relative_path": "doc/PantumP3500用户手册zh_CNV1.2_1644316283788.pdf",
            "sha256": SOURCE_PDF_SHA256,
            "page_count": 95,
        },
        "authoring_policy": {
            "production_answers_read": False,
            "production_run_labels_read": False,
            "network_or_llm_calls": False,
            "case_specific_program_rules": False,
            "replacement_only_when_contract_invalid_or_active_index_incomplete": True,
            "active_index_rebuild_required": False,
            "semantic_replacement_requires_new_query_and_reference": True,
            "superseded_review_must_not_be_reused": True,
            "same_prompt_independent_rereview_required": True,
        },
        "proposals": proposals,
    }
    proposals_path = THIS_DIR / "proposals.json"
    proposals_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation["proposal_file_sha256"] = _sha256(proposals_path)
    (THIS_DIR / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(proposals)
    _write_report(proposals, validation)

    artifact_names = [
        "REPORT.md",
        "cases.csv",
        "generate_proposals.py",
        "proposals.json",
        "validation.json",
    ]
    checksums = "".join(f"{_sha256(THIS_DIR / name)}  {name}\n" for name in artifact_names)
    (THIS_DIR / "SHA256SUMS").write_text(checksums, encoding="utf-8")


if __name__ == "__main__":
    main()
