from __future__ import annotations

import copy
from typing import Any

import app.domain.evaluation_proof as evaluation_proof_module
from app.application.query_answering.proposals import (
    AnswerDecision,
    AnswerPlanProposal,
    ClaimProposal,
    PlanReasonCode,
    RenderingRule,
)
from app.domain.answering._base import stable_content_hash
from app.domain.evaluation_proof import (
    OBSERVATION_SCHEMA_V3,
    answer_proof_evaluation,
    build_proof_evaluation_authority,
    project_proof_observation,
)
import test_proof_carrying_answer_v2 as proof_fixtures


_QUERY = "打印全黑版的故障原因是什么？解决办法中建议采取哪两项措施？"
_TABLE = (
    "| 故障现象 | 故障原因 | 解决办法 |\n"
    "| --- | --- | --- |\n"
    "| 打印全黑版 | 未正确安装激光碳粉盒。激光碳粉盒内部损坏。"
    "机器内部充电异常,未给激光碳粉盒充电。 | "
    "确保正确安装激光碳粉盒。建议更换原装激光碳粉盒。 |\n"
    "| 碳粉晕开 | 使用了不符合规格的介质。机器内部脏污。 | "
    "使用规格内介质。清洁机器内部。 |\n"
    "| 水平条纹 | 激光碳粉盒未正确安装。机器内部部件损坏。 | "
    "确保正确安装激光碳粉盒。 |"
)


def _table_data_row_ids(state: dict[str, Any], witness_catalog: list[dict[str, Any]]) -> list[str]:
    prompt_ids = {str(item["witness_id"]) for item in witness_catalog}
    return [
        str(item["witness_id"])
        for item in state["witness_catalog"]["witnesses"]
        if item.get("atom_kind") == "table_data_row" and str(item["witness_id"]) in prompt_ids
    ]


class _WrongProjectedRowsModel(proof_fixtures._AcceptingModel):
    """把 cell 内两项恶意降级为另外两条完整 data row。"""

    def select_plan(self, state, *, question_contract, witness_catalog):
        rows = _table_data_row_ids(state, witness_catalog)
        assert len(rows) == 3
        first, closed = question_contract.required_obligation_ids
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=(
                ClaimProposal(
                    obligation_id=first,
                    witness_id=rows[0],
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                ),
                ClaimProposal(
                    obligation_id=closed,
                    witness_id=rows[1],
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                ),
                ClaimProposal(
                    obligation_id=closed,
                    witness_id=rows[2],
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                ),
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _TwoRowClosedCountModel(proof_fixtures._AcceptingModel):
    """普通表格 CLOSED_COUNT 仍把每条 data row 作为一个业务 item。"""

    def select_plan(self, state, *, question_contract, witness_catalog):
        rows = _table_data_row_ids(state, witness_catalog)
        assert len(rows) == 2
        obligation_id = question_contract.required_obligation_ids[0]
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=obligation_id,
                    witness_id=witness_id,
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for witness_id in rows
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


def _rehash_downgraded_proof(observation: dict[str, Any]) -> None:
    proof = observation["answer_proof"]
    closed = proof["obligation_proofs"][1]
    for item in closed["proven_items"]:
        serialized = evaluation_proof_module._proven_item_serialization(item)
        payload = {
            key: value
            for key, value in serialized.items()
            if key not in {"item_ref_id", "item_ref_hash"}
        }
        item["item_ref_hash"] = stable_content_hash(payload)
        item["item_ref_id"] = "pit_" + item["item_ref_hash"].removeprefix("sha256:")[:40]

    serialized = evaluation_proof_module._obligation_proof_serialization(closed)
    payload = {
        key: value
        for key, value in serialized.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    closed["obligation_proof_hash"] = stable_content_hash(payload)
    closed["obligation_proof_id"] = (
        "opr_" + closed["obligation_proof_hash"].removeprefix("sha256:")[:40]
    )
    proof["proof_hash"] = stable_content_hash(
        evaluation_proof_module._answer_proof_identity_payload(proof)
    )
    proof["proof_id"] = "apr_" + proof["proof_hash"].removeprefix("sha256:")[:40]


def test_closed_count_projected_cell_scope_rejects_wrong_rows_as_cardinality_items() -> None:
    state = proof_fixtures._run_to_render(
        proof_fixtures._state(_TABLE, original_query=_QUERY),
        _WrongProjectedRowsModel(),
    )

    closed = next(
        item
        for item in state["answer_proof"]["obligation_proofs"]
        if item["closure"] == "closed_count"
    )
    assert state["answer_proof"]["strict_eligible"] is False
    assert closed["outcome"] != "answered"
    assert all(
        item["segmentation_method"] != "markdown_table_data_row.v1"
        for item in closed["proven_items"]
    )


def test_evaluator_rejects_projected_cell_scope_downgraded_to_data_rows() -> None:
    state = proof_fixtures._run_to_render(
        proof_fixtures._state(_TABLE, original_query=_QUERY),
        proof_fixtures._ProjectedTableCellModel(),
    )
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    downgraded = copy.deepcopy(observation)
    closed = downgraded["answer_proof"]["obligation_proofs"][1]
    for item in closed["proven_items"]:
        item["segmentation_method"] = "markdown_table_data_row.v1"
    _rehash_downgraded_proof(downgraded)

    audit = answer_proof_evaluation(downgraded, proof_authority=authority)

    assert audit is not None
    assert audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in audit["reason_codes"]
    assert "answer_proof_obligation_schema_invalid" not in audit["reason_codes"]
    assert "answer_proof_closed_count_table_cell_invalid" in audit["reason_codes"]


def test_plain_two_row_closed_count_keeps_markdown_data_row_items() -> None:
    table = "| 项目 | 解决办法 |\n| --- | --- |\n| 卡纸 | 取出卡住的纸张。 |\n| 缺纸 | 补充纸张。 |"
    state = proof_fixtures._run_to_render(
        proof_fixtures._state(table, original_query="请列出两个项目"),
        _TwoRowClosedCountModel(),
    )

    closed = state["answer_proof"]["obligation_proofs"][0]
    assert state["answer_proof"]["strict_eligible"] is True
    assert closed["outcome"] == "answered"
    assert closed["proven_item_count"] == 2
    assert {item["segmentation_method"] for item in closed["proven_items"]} == {
        "markdown_table_data_row.v1"
    }
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    audit = answer_proof_evaluation(
        observation,
        proof_authority=build_proof_evaluation_authority(state),
    )
    assert audit is not None
    assert audit["strict_graph_valid"] is True
