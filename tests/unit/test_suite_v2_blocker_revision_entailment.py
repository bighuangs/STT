from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMEDIATION_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-blocker-remediation"
)
sys.path.insert(0, str(REMEDIATION_ROOT))

import generate_proposals as generator  # noqa: E402


EXPECTED_CHANGED_IDS = {
    "P3500-HARD-009",
    "P3500-HARD-021",
    "P3500-HARD-037",
    "P3500-HARD-040",
    "P3500-HARD-042",
}
UNCHANGED_PROPOSED_CASE_HASHES = {
    "P3500-EASY-004": "d202493c1b0605c405b42751b70b07fc2bc5b9533c9daddcb0b3326d214a0313",
    "P3500-EASY-049": "2fd03a7ce7f1bdead1a5e1f9a5b0d6424c621fb8f4979478eac2c2cbc11e1183",
    "P3500-HARD-003": "724309d3eb91eb32d46cfe4b42db4069c68d58041c4060fb4546b217d7c23dd3",
    "P3500-HARD-007": "18dbad6972dc99741a3be9d0e9e117449c4097b82e4450a0bcac5ffd623d1776",
    "P3500-HARD-008": "23364bbec7f57d481433a06eba3df5f7662e8e231ca0814dbc090ecf71379868",
    "P3500-HARD-011": "ec9c5e7968cb9a10aa98c0f3de74255c1a688131eb977321362cb7bd65319cc0",
    "P3500-HARD-015": "890b9ef11939b220d894622953521b21225e141b876bdc2d8b21fef3249b4895",
    "P3500-HARD-041": "ea893ba705ae94129f017ecfd0b6d2b83475b577302c7c50630554a563357621",
}


def _by_id() -> tuple[dict[str, dict], dict]:
    proposals, validation = generator.build()
    return {row["case_id"]: row for row in proposals}, validation


def test_revision_changes_exactly_five_proposed_cases() -> None:
    rows, validation = _by_id()
    assert generator.REVISION_CHANGED_PARENT_IDS == EXPECTED_CHANGED_IDS
    assert set(validation["revision"]["changed_parent_case_ids"]) == EXPECTED_CHANGED_IDS
    assert set(validation["revision"]["unchanged_parent_case_ids"]) == set(
        UNCHANGED_PROPOSED_CASE_HASHES
    )
    for case_id, expected_hash in UNCHANGED_PROPOSED_CASE_HASHES.items():
        assert rows[case_id]["audit"]["proposed_case_sha256"] == expected_hash


def test_revised_question_contracts_remove_unsupported_claims() -> None:
    rows, _ = _by_id()
    expected = {
        "P3500-HARD-009": {
            "query": "出现黑色竖条时，手册列出的故障原因和解决办法分别有哪些？",
            "reference": "原因包括激光碳粉盒脏污、激光碳粉盒内部部件损坏、机器内部激光器反光玻璃脏污、进纸通道脏污。解决办法是清洁机器背部激光器反光玻璃，并清洁打印机背部进纸通道。",
            "expected_terms": [
                "激光碳粉盒脏污",
                "激光碳粉盒内部部件损坏",
                "机器内部激光器反光玻璃脏污",
                "进纸通道脏污",
                "清洁机器背部激光器反光玻璃",
                "清洁打印机背部进纸通道",
            ],
        },
        "P3500-HARD-021": {
            "query": "手册分别列出三种IPv6地址类型和两种无线连接模式。请说明三种IPv6地址的获得方式或识别特征，并概括基础结构与Wi-Fi热点模式的连接拓扑。",
            "reference": "IPv6本地链路地址由设备自行配置，以FE80开头；有状态地址由DHCPv6服务器配置，网络中没有DHCPv6服务器时不能分配；无状态地址由网络路由器自动配置。基础结构模式通过接入点（无线路由器）连接无线网络打印机和计算机；Wi-Fi热点模式由计算机直接通过无线网络连接打印机热点。",
            "expected_terms": [
                "FE80",
                "DHCPv6服务器",
                "网络路由器自动配置",
                "接入点",
                "打印机热点",
            ],
        },
        "P3500-HARD-037": {
            "query": "手册给出两条废弃物处理路径：法规信息对一般废弃设备要求送往哪里、目的是什么？奔图回收体系中，废弃打印机和耗材先交给哪里，之后由何类机构处理？两条路径分别通过哪些专门渠道实现回收利用？",
            "reference": "法规信息要求将废弃设备送到指定的收集点，以回收利用废弃的电气和电子设备。奔图回收体系中，废弃打印机和耗材先交给当地奔图维修中心统一回收，再由国家认定、具备废弃物处理资质的机构进行正确回收和再利用。第一条路径通过指定收集点回收利用，第二条通过奔图维修中心和具备资质的处理机构统一回收、再利用。",
            "expected_terms": [
                "指定的收集点",
                "回收利用废弃的电气和电子设备",
                "当地奔图维修中心",
                "具备废弃物处理资质的机构",
                "第一条路径通过指定收集点回收利用",
                "第二条通过奔图维修中心和具备资质的处理机构统一回收、再利用",
            ],
        },
        "P3500-HARD-040": {
            "query": "手册在法规信息和激光碳粉盒寿命表中分别引用EN 55022、EN 60950和ISO/IEC 19752。请说明前两个标准在手册中支持什么结论，ISO/IEC 19752用于什么数据并列出三种容量的平均打印量。",
            "reference": "EN 55022支持本产品符合B级范围的结论，EN 60950支持产品安全要求。ISO/IEC 19752是激光碳粉盒平均打印量的数据依据：标准容量约3000页、高容量约6000页、超大容量约10000页。",
            "expected_terms": [
                "EN 55022的B级范围",
                "EN 60950的安全要求",
                "ISO/IEC 19752",
                "约3000页",
                "约6000页",
                "约10000页",
            ],
        },
        "P3500-HARD-042": {
            "query": "演示页给出750至4000页推荐月打印量和50000页每月工作负荷。若实际月打印量为3900页，它是否处于推荐范围内，且是否低于每月工作负荷？网络配置信息页显示的无线网络IPv4地址是多少？",
            "reference": "3900页处于750至4000页的推荐月打印量范围内，并低于50000页的每月工作负荷。网络配置信息页显示的无线网络IPv4地址是192.168.0.102。",
            "expected_terms": [
                "3900页处于750至4000页的推荐月打印量范围内",
                "低于50000页的每月工作负荷",
                "192.168.0.102",
            ],
        },
    }
    for case_id, contract in expected.items():
        case = rows[case_id]["proposed_case"]
        assert case["query"] == contract["query"]
        assert case["reference_answer"] == contract["reference"]
        assert case["expected_terms"] == contract["expected_terms"]


def test_entailment_receipts_are_top_level_and_exactly_render_reference() -> None:
    rows, validation = _by_id()
    assert validation["evidence_entailment"]["status"] == "passed"
    for row in rows.values():
        assert "evidence_entailment_receipt" in row
        assert "evidence_entailment_receipt" not in row["proposed_case"]
        receipt = row["evidence_entailment_receipt"]
        assert "".join(claim["claim_text"] for claim in receipt["claim_inventory"]) == row[
            "proposed_case"
        ]["reference_answer"]


def test_inference_receipts_use_replayable_allowlisted_operators() -> None:
    rows, _ = _by_id()
    expected_modes = {
        "P3500-HARD-008": {"finite_enumerated_set_disjoint"},
        "P3500-HARD-011": {"interval_contains", "numeric_gt"},
        "P3500-HARD-037": {"direct_conjunction"},
        "P3500-HARD-042": {"interval_contains", "numeric_lt"},
    }
    for case_id, required_modes in expected_modes.items():
        receipt = rows[case_id]["evidence_entailment_receipt"]
        modes = set(generator._iter_entailment_modes(receipt["claim_inventory"]))
        assert required_modes <= modes
        assert modes <= generator.ALLOWED_ENTAILMENT_MODES


@pytest.mark.parametrize(
    "mode",
    ["open_world_absence", "nonassociation", "noninterchangeability"],
)
def test_entailment_gate_rejects_open_world_modes(mode: str) -> None:
    rows, _ = _by_id()
    receipt = copy.deepcopy(rows["P3500-EASY-004"]["evidence_entailment_receipt"])
    receipt["claim_inventory"][0]["entailment"]["mode"] = mode
    with pytest.raises(AssertionError, match="unsupported entailment mode"):
        generator._validate_evidence_entailment_receipt(
            receipt=receipt,
            proposed_case=rows["P3500-EASY-004"]["proposed_case"],
        )


@pytest.mark.parametrize(
    "claim_text",
    [
        "手册没有说明两者存在关联。",
        "该IP地址与工作负载判断无关。",
        "这些标准不能相互替代。",
    ],
)
def test_entailment_gate_rejects_open_world_assertion_text(claim_text: str) -> None:
    rows, _ = _by_id()
    receipt = copy.deepcopy(rows["P3500-EASY-004"]["evidence_entailment_receipt"])
    receipt["claim_inventory"][0]["claim_text"] = claim_text
    proposed_case = copy.deepcopy(rows["P3500-EASY-004"]["proposed_case"])
    proposed_case["reference_answer"] = claim_text
    with pytest.raises(AssertionError, match="open-world assertion rejected"):
        generator._validate_evidence_entailment_receipt(
            receipt=receipt,
            proposed_case=proposed_case,
        )
