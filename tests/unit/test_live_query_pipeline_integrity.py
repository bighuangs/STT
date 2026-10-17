from __future__ import annotations

from dataclasses import dataclass

from app.application.query_answering.builders import build_question_contract
from app.application.model_usage_service import record_model_call
from app.domain.query_planning import PlanningPolicy
from app.evaluation.observation import _packed_context
from app.process.query.agent.state import create_query_default_state
from app.rag.query.query_planning_service import build_query_plan_from_state


def test_multihop_decomposition_keeps_context_for_anaphoric_clause() -> None:
    query = "设备安装前应核对哪些条件？如果未核对这些条件会有什么风险？"
    state = create_query_default_state(
        session_id="integrity-plan",
        original_query=query,
        rewritten_query=query,
        questions=[query],
        kb_id="kb-integrity",
        web_search_allowed=False,
        evaluation_mode=True,
        is_stream=False,
    )

    plan = build_query_plan_from_state(
        state,
        tenant_id="tenant-integrity",
        scope_id="scope-integrity",
        policy=PlanningPolicy(allow_hyde=False, allow_web=False),
    )

    retrieval_queries = [
        item.query for item in plan.sub_queries if item.purpose.value == "retrieval"
    ]
    assert retrieval_queries == [
        "设备安装前应核对哪些条件",
        "设备安装前应核对哪些条件；如果未核对这些条件会有什么风险",
    ]


def test_multihop_decomposition_keeps_explicit_conditional_branch_independent() -> None:
    plan = _plan(
        "双面打印单元卡纸时，若纸张未随单元一起取出，应如何处理？"
        "若软件故障导致打印机不执行打印作业，应采取哪些措施？"
    )

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == [
        "双面打印单元卡纸时，若纸张未随单元一起取出，应如何处理",
        "若软件故障导致打印机不执行打印作业，应采取哪些措施",
    ]


def test_multihop_decomposition_keeps_direct_anaphoric_clause_context() -> None:
    plan = _plan("设备安装失败时应先检查电源？这会排除什么问题？")

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == [
        "设备安装失败时应先检查电源",
        "设备安装失败时应先检查电源；这会排除什么问题",
    ]


def test_multihop_decomposition_drops_pure_response_instruction_clause() -> None:
    for instruction in (
        "请分别依据两份证据中的具体操作说明作答。",
        "请结合两条证据中的说明作答。",
        "请根据安装向导界面描述。",
        "请按顺序说明操作步骤。",
        "请描述完整流程。",
        "请列出全部内容。",
    ):
        plan = _plan("标准纸盒应如何装纸？多功能纸盒应如何装纸？" + instruction)

        retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
        assert retrieval == ["标准纸盒应如何装纸", "多功能纸盒应如何装纸"]


def test_single_information_facet_projects_trailing_response_instruction() -> None:
    for query, information_need in (
        ("设备启用自动寻址后有哪些注意事项？请结合证据说明。", "设备启用自动寻址后有哪些注意事项"),
        ("配置基础网络前需要了解哪些信息？请根据文档内容回答。", "配置基础网络前需要了解哪些信息"),
    ):
        plan = _plan(query)

        retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
        assert retrieval == [information_need]
        assert [facet.information_need for facet in plan.facet_ledger.facets] == [information_need]


def test_single_action_order_tail_shapes_contract_without_entering_retrieval() -> None:
    plan = _plan("设备发生卡纸时应如何处理？请按顺序说明操作步骤。")

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == ["设备发生卡纸时应如何处理"]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "设备发生卡纸时应如何处理"
    ]

    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})
    assert contract.obligations[0].required_output_form.value == "ordered_steps"


def test_distributed_order_instruction_projects_to_existing_action_facets() -> None:
    plan = _plan("标准纸盒应如何装纸？多功能纸盒应如何装纸？请分别按顺序说明操作步骤。")

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == ["标准纸盒应如何装纸", "多功能纸盒应如何装纸"]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "标准纸盒应如何装纸；请按顺序说明操作步骤",
        "多功能纸盒应如何装纸；请按顺序说明操作步骤",
    ]
    assert [facet.closure.value for facet in plan.facet_ledger.facets] == [
        "open_enumeration",
        "open_enumeration",
    ]

    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})
    assert [item.required_output_form.value for item in contract.obligations] == [
        "ordered_steps",
        "ordered_steps",
    ]


def test_scope_dimension_is_not_mistaken_for_response_instruction() -> None:
    plan = _plan("设备手册中的 Wi-Fi 热点连接模式与基础结构连接模式有何区别？请从连接方式上说明。")

    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "设备手册中的 Wi-Fi 热点连接模式与基础结构连接模式有何区别",
        "请从连接方式上说明",
    ]


def test_shared_predicate_attribute_group_keeps_its_entity_and_scenario_scope() -> None:
    query = "在节能模式下，该设备的休眠时间、唤醒方式和指示灯状态分别是什么？"
    plan = _plan(query)

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == [query]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "在节能模式下，该设备的休眠时间、唤醒方式和指示灯状态分别是什么"
    ]

    sourced = _plan(query + "请根据文档内容回答。")
    sourced_retrieval = [
        item.query for item in sourced.sub_queries if item.purpose.value == "retrieval"
    ]
    assert sourced_retrieval == ["在节能模式下，该设备的休眠时间、唤醒方式和指示灯状态分别是什么"]


def test_independent_second_attribute_keeps_its_own_scope() -> None:
    plan = _plan("设备的接口数量是多少？另一台设备的固件版本是什么？")

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == ["设备的接口数量是多少", "另一台设备的固件版本是什么"]


def test_result_set_dependent_follow_up_is_one_relational_projection() -> None:
    for query, expected_closure, expected_count in (
        (
            "列表中哪些接口处于启用状态？它们的端口号是什么？",
            "open_enumeration",
            None,
        ),
        (
            "资产表中异常状态有哪些？对应的处理建议包括哪些操作？",
            "open_enumeration",
            None,
        ),
        (
            "菜单中包含哪些状态项？其中哪个用于显示当前连接状态？",
            "open_enumeration",
            None,
        ),
    ):
        plan = _plan(query)
        retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]

        assert retrieval == [query]
        assert len(plan.facet_ledger.facets) == 1
        facet = plan.facet_ledger.facets[0]
        assert facet.necessity.value == "required"
        assert facet.closure.value == expected_closure
        assert facet.expected_count == expected_count

        contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})
        assert len(contract.obligations) == 1
        assert contract.obligations[0].required_output_form.value == "list"


def test_relational_follow_up_count_stays_local_to_its_projected_table_column() -> None:
    query = "故障表中打印中断的原因是什么？解决办法中建议采取哪两项措施？"
    plan = _plan(query)

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == [
        "故障表中打印中断的原因是什么",
        "故障表中打印中断的原因是什么；解决办法中建议采取哪两项措施",
    ]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "故障表中打印中断的原因是什么",
        "解决办法中建议采取哪两项措施",
    ]
    assert [facet.closure.value for facet in plan.facet_ledger.facets] == [
        "single",
        "closed_count",
    ]
    assert [facet.expected_count for facet in plan.facet_ledger.facets] == [None, 2]

    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})
    assert [item.required_output_form.value for item in contract.obligations] == [
        "fact",
        "list",
    ]


def test_follow_up_intersection_is_one_relational_projection() -> None:
    plan = _plan("根据规格表，纸盒 A 和纸盒 B 分别支持哪些尺寸？请列出两者都支持的尺寸。")

    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval == ["根据规格表，纸盒 A 和纸盒 B 分别支持哪些尺寸？请列出两者都支持的尺寸。"]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "根据规格表，纸盒 A 和纸盒 B 分别支持哪些尺寸？请列出两者都支持的尺寸"
    ]


def _plan(query: str):
    state = create_query_default_state(
        session_id="integrity-ambiguity",
        original_query=query,
        rewritten_query=query,
        questions=[query],
        kb_id="kb-integrity",
        web_search_allowed=False,
        evaluation_mode=True,
        is_stream=False,
    )
    return build_query_plan_from_state(
        state,
        tenant_id="tenant-integrity",
        scope_id="scope-integrity",
        policy=PlanningPolicy(allow_hyde=False, allow_web=False),
    )


def test_single_kb_explicit_domain_referent_is_not_treated_as_bare() -> None:
    plan = _plan("该产品是否适合室外使用？")

    assert plan.disposition.value == "retrieve"
    assert plan.understanding.ambiguity.is_ambiguous is False


def test_plural_referent_uses_collection_anchor_from_previous_clause() -> None:
    plan = _plan("可用热点有哪些？它们分别支持什么频段？")

    assert plan.disposition.value == "retrieve"
    assert plan.understanding.ambiguity.is_ambiguous is False
    retrieval = [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert retrieval[1] == "可用热点有哪些；它们分别支持什么频段"


def test_truly_bare_referent_still_requires_clarification() -> None:
    plan = _plan("它怎么用？")

    assert plan.disposition.value == "need_clarification"
    assert plan.understanding.ambiguity.missing_slots == ("target",)


def test_explicit_comparison_pair_does_not_require_duplicate_entity_extraction() -> None:
    plan = _plan("比较有线连接与无线连接：两者各适用于什么场景？")

    assert plan.disposition.value == "retrieve"
    assert plan.understanding.ambiguity.is_ambiguous is False


def test_comparison_without_named_targets_still_requires_clarification() -> None:
    plan = _plan("比较一下哪个好？")

    assert plan.disposition.value == "need_clarification"
    assert plan.understanding.ambiguity.missing_slots == ("comparison_targets",)


def test_packed_context_uses_actual_prompt_text_and_keeps_source_identity() -> None:
    final_state = {
        "context_pack": {
            "packed_blocks": [
                {
                    "label": "E1",
                    "text": "预算内实际文本",
                    "representation_id": "rep_" + "a" * 40,
                    "truncation": {"reasons": ["token_budget"]},
                    "inclusion_reason": "sub_query_coverage",
                }
            ]
        },
        "context_aliases": [
            {
                "label": "E1",
                "packed_text": "预算内实际文本",
                "document": {
                    "chunk_id": "chunk-1",
                    "evidence_id": "ev_" + "b" * 40,
                    "content": "未截断的完整原文不得伪装成模型上下文",
                },
            }
        ],
    }

    context = _packed_context(final_state)

    assert context[0]["chunk_id"] == "chunk-1"
    assert context[0]["content"] == "预算内实际文本"
    assert context[0]["text"] == "预算内实际文本"
    assert context[0]["context_truncation"] == {"reasons": ["token_budget"]}


@dataclass
class _Response:
    content: str
    usage_metadata: dict[str, int]


def test_usage_summary_carries_resolved_model_without_principal() -> None:
    state = {"model_usage_summaries": []}
    response = _Response(
        content="ok",
        usage_metadata={"input_tokens": 3, "output_tokens": 2},
    )

    record_model_call(
        state,
        [response],
        scene="answer_generation",
        model_id="frozen-model",
    )

    assert state["model_usage_summaries"] == [
        {
            "scene": "answer_generation",
            "call_status": "succeeded",
            "usage_status": "available",
            "model_id": "frozen-model",
            "input_tokens": 3,
            "output_tokens": 2,
        }
    ]
