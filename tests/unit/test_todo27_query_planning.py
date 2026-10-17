from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json

import pytest
from pydantic import ValidationError

from app.application.query_understanding_service import (
    DeterministicQueryPlanner,
    QueryPlanningInput,
    project_chapter_title_retrieval_expression,
    validate_intent_preserving_rewrite,
)
from app.application.query_answering.builders import build_question_contract
from app.domain.answering.question import (
    ObligationKind,
    QuestionContract,
    RequiredOutputForm,
)
from app.domain.query_planning import (
    AuthorizedScopeRef,
    ConversationTurn,
    FacetClosure,
    FacetNecessity,
    FacetPolarity,
    MAX_QUERY_LENGTH,
    PlanDisposition,
    PlanningPolicy,
    QueryIntent,
    QueryPlan,
    QueryRoute,
    QueryRouteType,
    RewriteStatus,
    RouteFallback,
    RouteFilter,
    TemporalKind,
    validate_plan_for_execution,
)
from app.rag.query.query_planning_service import (
    build_query_plan_from_state,
    query_plan_state_patch,
)
from app.process.query.agent.state import create_query_default_state


def _scope(
    *,
    tenant_id: str = "tenant-a",
    kb_id: str = "kb-hak180",
) -> AuthorizedScopeRef:
    return AuthorizedScopeRef(
        scope_id=f"scope-{tenant_id}-{kb_id}",
        tenant_id=tenant_id,
        kb_ids=(kb_id,),
    )


def _request(
    query: str,
    *,
    policy: PlanningPolicy | None = None,
    history: tuple[ConversationTurn, ...] = (),
    rewrite_candidate: str | None = None,
    authorized_scope_ref: AuthorizedScopeRef | None = None,
) -> QueryPlanningInput:
    return QueryPlanningInput(
        original_query=query,
        history=history,
        rewrite_candidate=rewrite_candidate,
        authorized_scope_ref=authorized_scope_ref or _scope(),
        profile_version=7,
        policy=policy or PlanningPolicy(),
        deadline_at=datetime.now(UTC) + timedelta(seconds=8),
    )


def test_query_plan_schema_is_strict_and_json_serializable():
    plan = DeterministicQueryPlanner().plan(_request("在使用 HAK180 之前需要做什么？"))

    schema = QueryPlan.model_json_schema()
    payload = plan.model_dump(mode="json")
    restored = QueryPlan.model_validate_json(plan.model_dump_json())

    assert schema["additionalProperties"] is False
    assert "understanding" in schema["properties"]
    assert "authorized_scope_ref" in schema["properties"]
    assert payload["planner_version"] == "query-planner-rules/1.6.0"
    assert payload["facet_ledger"]["schema_version"] == "query-facet-ledger.v2"
    assert payload["facet_ledger"]["extractor_version"] == "query-facet-rules/1.4.0"
    assert restored == plan

    payload["untrusted_extra"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryPlan.model_validate(payload)


def test_hak180_simple_question_uses_only_required_local_route():
    planner = DeterministicQueryPlanner()
    request = _request("在使用 HAK180 之前需要做什么？")

    first = planner.plan(request)
    second = planner.plan(request)

    assert first.plan_id == second.plan_id
    assert first.disposition == PlanDisposition.RETRIEVE
    assert first.understanding.intent == QueryIntent.PROCEDURE
    assert first.understanding.entities == ("HAK180",)
    assert first.understanding.original_query == request.original_query
    assert first.understanding.display_query == request.original_query
    assert first.understanding.standalone_query == request.original_query
    assert len(first.sub_queries) == 1
    assert [route.route_type for route in first.routes] == [QueryRouteType.LOCAL_HYBRID]
    assert first.routes[0].required is True
    assert "simple_query_avoids_hyde" in first.decision_reasons
    assert "web_not_needed_without_freshness" in first.decision_reasons


@pytest.mark.parametrize(
    ("query", "expected_path"),
    [
        ("第８．４节的标题是什么？", "8.4"),
        ("4.2.1章的标题是什么？", "4.2.1"),
        ("What is the heading of section 3.2?", "3.2"),
    ],
)
def test_chapter_title_retrieval_projection_uses_index_structure_alias(
    query: str,
    expected_path: str,
):
    projected = project_chapter_title_retrieval_expression(query)

    assert projected.startswith(query)
    assert projected.endswith(f"标题路径：{expected_path}")
    assert project_chapter_title_retrieval_expression(projected) == projected


@pytest.mark.parametrize(
    "query",
    [
        "版本 1.2.3 的标题是什么？",
        "v2.10.0-beta 的标题是什么？",
        "IP 地址 192.168.0.1 的标题是什么？",
        "圆周率 3.14 的标题是什么？",
        "12.50 元商品的标题是什么？",
        "37.5 ℃ 对应什么标题？",
        "2026.08.25 的标题是什么？",
        "8.4. 纸盒与纸张类型",
        "4.2.1章包含哪些内容？",
        "7.3.2节包含哪些内容？文档标题是什么？",
        "What is the document title? See section 9.4.",
        "第8章第4节的题名是什么？",
    ],
)
def test_chapter_title_retrieval_projection_rejects_unanchored_numbers(query: str):
    assert project_chapter_title_retrieval_expression(query) == query


def test_chapter_title_projection_respects_query_contract_length():
    suffix = " 4.2.1章的标题"
    query = "x" * (MAX_QUERY_LENGTH - len(suffix)) + suffix

    assert len(query) == MAX_QUERY_LENGTH
    assert project_chapter_title_retrieval_expression(query) == query


def test_chapter_title_plan_preserves_user_query_and_existing_retrieval_shape():
    query = "4.2.1章的标题是什么？"
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert plan.understanding.original_query == query
    assert plan.understanding.display_query == query
    assert plan.understanding.standalone_query == query
    assert [(item.sub_query_id, item.purpose.value) for item in plan.sub_queries] == [
        ("sq_primary", "retrieval")
    ]
    assert plan.sub_queries[0].query == f"{query} 标题路径：4.2.1"
    assert plan.facet_ledger.facets[0].retrieval_expression == plan.sub_queries[0].query
    assert [route.route_type for route in plan.routes] == [QueryRouteType.LOCAL_HYBRID]
    assert plan.routes[0].sub_query_ids == ("sq_primary",)


@pytest.mark.parametrize(
    "policy",
    [
        PlanningPolicy(total_timeout_ms=30_000),
        PlanningPolicy(total_timeout_ms=250),
    ],
    ids=["published-default-retrieval-budget", "minimum-policy-budget"],
)
def test_required_local_route_uses_bounded_total_policy_timeout(
    policy: PlanningPolicy,
):
    plan = DeterministicQueryPlanner().plan(
        _request("在使用 HAK180 之前需要做什么？", policy=policy)
    )

    local_route = next(
        route for route in plan.routes if route.route_type == QueryRouteType.LOCAL_HYBRID
    )

    assert local_route.required is True
    assert local_route.timeout_ms == policy.total_timeout_ms
    assert local_route.timeout_ms <= plan.policy.total_timeout_ms


@pytest.mark.parametrize(
    ("query", "intent", "response_key"),
    [
        ("你好！", QueryIntent.GREETING, "query.greeting"),
        ("如何在页面里切换知识库？", QueryIntent.UI_HELP, "query.ui_help"),
        ("怎么在当前页面上传文档？", QueryIntent.UI_HELP, "query.ui_help"),
        ("清空对话按钮在哪里？", QueryIntent.UI_HELP, "query.ui_help"),
    ],
)
def test_direct_intents_never_trigger_retrieval(
    query: str,
    intent: QueryIntent,
    response_key: str,
):
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert plan.disposition == PlanDisposition.DIRECT_ANSWER
    assert plan.understanding.intent == intent
    assert plan.understanding.need_retrieval is False
    assert plan.sub_queries == ()
    assert plan.routes == ()
    assert plan.response_key == response_key


@pytest.mark.parametrize(
    "query",
    [
        "驱动搜索完成后应如何选择打印机？请展示手册中的安装界面图片，并标明页码。",
        "设备安装界面如何操作？",
        "PX900 驱动程序如何安装？",
        "根据说明书，网页配置界面的按钮在哪里？请给出引用。",
        "路由器后台登录界面如何操作？",
    ],
)
def test_manual_or_device_ui_language_still_requires_retrieval(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert plan.disposition == PlanDisposition.RETRIEVE
    assert plan.understanding.intent != QueryIntent.UI_HELP
    assert plan.understanding.need_retrieval is True
    assert plan.sub_queries
    assert plan.routes
    assert plan.response_key is None


@pytest.mark.parametrize(
    ("query", "missing_slot"),
    [
        ("它怎么处理？", "target"),
        ("哪个更好？", "comparison_targets"),
        ("HAK180", "information_need"),
    ],
)
def test_critical_ambiguity_returns_actionable_clarification_without_search(
    query: str,
    missing_slot: str,
):
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert plan.disposition == PlanDisposition.NEED_CLARIFICATION
    assert plan.routes == ()
    assert plan.sub_queries == ()
    assert plan.understanding.ambiguity.is_ambiguous is True
    assert missing_slot in plan.understanding.ambiguity.missing_slots
    assert plan.understanding.ambiguity.clarification_question


def test_history_snapshot_can_resolve_pronoun_without_reading_a_repository():
    plan = DeterministicQueryPlanner().plan(
        _request(
            "它如何关机？",
            history=(
                ConversationTurn(role="user", text="HAK180 应该如何启动？"),
                ConversationTurn(role="assistant", text="请先检查总电源。"),
            ),
        )
    )

    assert plan.disposition == PlanDisposition.RETRIEVE
    assert plan.understanding.standalone_query == "HAK180如何关机？"
    assert plan.understanding.rewrite.status == RewriteStatus.ACCEPTED
    assert plan.understanding.entities == ("HAK180",)


@pytest.mark.parametrize(
    ("original", "candidate", "reason"),
    [
        (
            "HAK180 是否可以不接地使用？",
            "HAK180 可以接地使用。",
            "rewrite_changed_negation",
        ),
        (
            "HAK180 的压力上限是多少？",
            "设备的压力上限是多少？",
            "rewrite_lost_identifier_or_number",
        ),
        (
            "HAK180 为什么无法启动？",
            "HAK180 如何启动？",
            "rewrite_changed_negation",
        ),
    ],
)
def test_intent_changing_rewrite_is_rejected(
    original: str,
    candidate: str,
    reason: str,
):
    accepted, _, reasons = validate_intent_preserving_rewrite(original, candidate)
    plan = DeterministicQueryPlanner().plan(_request(original, rewrite_candidate=candidate))

    assert accepted is False
    assert reason in reasons
    assert plan.understanding.standalone_query == original
    assert plan.understanding.rewrite.status == RewriteStatus.REJECTED


def test_multihop_decomposition_is_bounded_and_hyde_is_never_evidence():
    policy = PlanningPolicy(max_subqueries=3, allow_hyde=True)
    plan = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 如何安装？同时如何执行日常维护？另外如何校准？",
            policy=policy,
        )
    )

    assert plan.understanding.needs_multihop is True
    assert len(plan.sub_queries) == policy.max_subqueries
    assert [item.sub_query_id for item in plan.sub_queries] == [
        "sq_part_1",
        "sq_part_2",
        "sq_synthesis",
    ]
    assert plan.sub_queries[-1].depends_on == ("sq_part_1", "sq_part_2")
    assert "subqueries_truncated_by_budget" in plan.limitations
    hyde = next(route for route in plan.routes if route.route_type == QueryRouteType.HYDE)
    assert hyde.required is False
    assert hyde.evidence_eligible is False
    assert "sq_synthesis" not in hyde.sub_query_ids


def test_coordinated_each_question_is_split_into_independent_retrieval_aspects():
    plan = DeterministicQueryPlanner().plan(
        _request(
            "首次安装打印头和首次使用校准时，各有哪些必须遵守的注意事项？请标明手册页码。",
            policy=PlanningPolicy(max_subqueries=4, allow_hyde=False),
        )
    )

    retrieval = [item for item in plan.sub_queries if item.purpose.value == "retrieval"]
    assert plan.understanding.needs_multihop is True
    assert [item.query for item in retrieval] == [
        "首次安装打印头有哪些必须遵守的注意事项？请标明手册页码",
        "首次使用校准有哪些必须遵守的注意事项？请标明手册页码",
    ]
    assert plan.sub_queries[-1].sub_query_id == "sq_synthesis"
    assert plan.sub_queries[-1].depends_on == ("sq_part_1", "sq_part_2")
    local = next(route for route in plan.routes if route.route_type == QueryRouteType.LOCAL_HYBRID)
    assert local.sub_query_ids == ("sq_part_1", "sq_part_2")
    assert len(plan.facet_ledger.facets) == 2
    assert [facet.sub_query_ids for facet in plan.facet_ledger.facets] == [
        ("sq_part_1",),
        ("sq_part_2",),
    ]


def test_facet_ids_are_stable_across_order_punctuation_and_connector_perturbations():
    variants = (
        "HAK180 如何安装？同时 HAK180 如何维护？",
        "HAK180 如何维护；另外 HAK180 如何安装。",
        "HAK180 如何安装，并且 HAK180 如何维护",
    )

    ledgers = [DeterministicQueryPlanner().plan(_request(query)).facet_ledger for query in variants]
    expected_ids = {facet.facet_id for facet in ledgers[0].facets}

    assert len(expected_ids) == 2
    assert all({facet.facet_id for facet in ledger.facets} == expected_ids for ledger in ledgers)
    assert all(len(ledger.facets) == 2 for ledger in ledgers)


def test_ordinary_coordinated_objects_remain_one_information_facet():
    plan = DeterministicQueryPlanner().plan(_request("HAK180 安装时如何检查打印头和墨盒？"))

    assert len(plan.facet_ledger.facets) == 1
    assert plan.facet_ledger.facets[0].sub_query_ids == ("sq_primary",)
    assert "打印头和墨盒" in plan.facet_ledger.facets[0].information_need


def test_aspect_splitter_does_not_split_adverb_but_splits_two_complete_comma_clauses():
    joint = DeterministicQueryPlanner().plan(_request("设备是否同时支持打印和扫描？"))
    separate = DeterministicQueryPlanner().plan(_request("打印头在哪里，墨盒在哪里？"))

    assert [item.query for item in joint.sub_queries if item.purpose.value == "retrieval"] == [
        "设备是否同时支持打印和扫描？"
    ]
    assert len(joint.facet_ledger.facets) == 1
    assert [facet.information_need for facet in separate.facet_ledger.facets] == [
        "打印头在哪里",
        "墨盒在哪里",
    ]


def test_coordinated_each_supports_yiji_and_preserves_shared_question_suffix():
    plan = DeterministicQueryPlanner().plan(_request("打印头以及墨盒分别如何安装？"))

    assert [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"] == [
        "打印头如何安装",
        "墨盒如何安装",
    ]
    assert [facet.sub_query_ids for facet in plan.facet_ledger.facets] == [
        ("sq_part_1",),
        ("sq_part_2",),
    ]


def test_facet_canonicalization_preserves_semantic_technical_symbols_without_collision():
    plan = DeterministicQueryPlanner().plan(_request("C++ 和 C# 分别如何配置？"))

    assert [facet.information_need for facet in plan.facet_ledger.facets] == [
        "C++如何配置",
        "C#如何配置",
    ]
    assert len({facet.facet_id for facet in plan.facet_ledger.facets}) == 2
    assert [facet.sub_query_ids for facet in plan.facet_ledger.facets] == [
        ("sq_part_1",),
        ("sq_part_2",),
    ]


def test_conditional_and_negative_facets_preserve_answer_requirements():
    conditional = DeterministicQueryPlanner().plan(
        _request("HAK180 使用前应检查哪些条件？如果未满足这些条件，应如何处理？")
    )
    positive = DeterministicQueryPlanner().plan(_request("HAK180 可以在通风环境中使用吗？"))
    negative = DeterministicQueryPlanner().plan(_request("HAK180 不得在潮湿环境中使用吗？"))

    assert [facet.necessity for facet in conditional.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.CONDITIONAL,
    ]
    assert conditional.facet_ledger.facets[1].polarity == FacetPolarity.NEGATIVE
    assert positive.facet_ledger.facets[0].polarity == FacetPolarity.AFFIRMATIVE
    assert negative.facet_ledger.facets[0].polarity == FacetPolarity.NEGATIVE
    assert positive.facet_ledger.facets[0].facet_id != negative.facet_ledger.facets[0].facet_id


@pytest.mark.parametrize(
    "query",
    [
        "如果状态灯持续闪烁，设备应如何处理？",
        "若网络连接中断，应执行哪些检查？",
        "如未完成初始化，下一步怎么操作？",
        "当保护模式启动时，如何恢复运行？",
        "When startup fails, what should the operator check?",
    ],
)
def test_conditional_primary_facet_is_promoted_to_required(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert plan.disposition == PlanDisposition.RETRIEVE
    assert len(plan.facet_ledger.facets) == 1
    assert plan.facet_ledger.facets[0].necessity == FacetNecessity.REQUIRED


def test_all_conditional_facets_promote_only_the_primary_facet():
    plan = DeterministicQueryPlanner().plan(
        _request("如果状态灯持续闪烁，应如何处理？若自动恢复失败，应执行什么操作？")
    )

    assert [facet.necessity for facet in plan.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.CONDITIONAL,
    ]


def test_explicit_scenario_question_is_required_but_dependent_branch_remains_conditional():
    scenario = DeterministicQueryPlanner().plan(
        _request("当无线连接中断时，应检查哪些项目？之后还需记录什么信息？")
    )
    dependent = DeterministicQueryPlanner().plan(
        _request("设备支持扩展纸盒吗？如果支持，其容量是多少？")
    )

    assert [facet.necessity for facet in scenario.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.REQUIRED,
    ]
    assert [facet.necessity for facet in dependent.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.CONDITIONAL,
    ]


@pytest.mark.parametrize(
    "query",
    [
        "设备型号是什么？当设备发生卡纸时，应如何处理？",
        "设备型号是什么？如果该设备支持双面打印，应如何设置？",
        "设备型号是什么？若软件故障导致打印机停止工作，应如何处理？",
    ],
)
def test_later_explicit_scenario_is_required_instead_of_positionally_conditional(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query))

    assert [facet.necessity for facet in plan.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.REQUIRED,
    ]


def test_repeated_binary_answer_state_is_a_conditional_dependency() -> None:
    plan = DeterministicQueryPlanner().plan(
        _request("设备支持扩展纸盒吗？如果该设备支持扩展纸盒，其容量是多少？")
    )

    assert [facet.necessity for facet in plan.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.CONDITIONAL,
    ]


@pytest.mark.parametrize(
    "dependent_clause",
    [
        "如果支持，其容量是多少？",
        "如果该设备支持，其容量是多少？",
        "如果未满足这些条件，应如何处理？",
        "若自动恢复失败，应执行什么操作？",
    ],
)
def test_answer_dependent_conditions_remain_conditional(dependent_clause: str):
    plan = DeterministicQueryPlanner().plan(_request(f"设备应先完成基础检查吗？{dependent_clause}"))

    assert [facet.necessity for facet in plan.facet_ledger.facets] == [
        FacetNecessity.REQUIRED,
        FacetNecessity.CONDITIONAL,
    ]


@pytest.mark.parametrize(
    "query",
    [
        "安装驱动前需要完成哪些具体操作？",
        "输入密码后应执行哪些后续操作？",
        "在管理界面中如何启用网络协议？",
        "设备卡纸时应如何处理？请按顺序说明。",
    ],
)
def test_action_collection_builds_an_ordered_procedure_contract(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query))
    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})

    assert len(contract.obligations) == 1
    assert contract.obligations[0].kind == ObligationKind.PROCEDURE
    assert contract.obligations[0].required_output_form == RequiredOutputForm.ORDERED_STEPS


def test_compound_action_collection_builds_an_ordered_procedure_contract():
    plan = DeterministicQueryPlanner().plan(_request("取出卡纸后还需进行哪些检查和操作？"))
    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})

    assert plan.facet_ledger.facets[0].closure == FacetClosure.OPEN_ENUMERATION
    assert contract.obligations[0].kind == ObligationKind.PROCEDURE
    assert contract.obligations[0].required_output_form == RequiredOutputForm.ORDERED_STEPS


@pytest.mark.parametrize(
    ("query", "closure", "kind", "output_form"),
    [
        (
            "设备支持哪些网络协议？请按顺序列出。",
            FacetClosure.OPEN_ENUMERATION,
            ObligationKind.ENUMERATION,
            RequiredOutputForm.LIST,
        ),
        (
            "哪个操作步骤最容易失败？",
            FacetClosure.SINGLE,
            ObligationKind.FACT,
            RequiredOutputForm.FACT,
        ),
        (
            "装入纸张前应采取什么操作以避免卡纸？",
            FacetClosure.SINGLE,
            ObligationKind.FACT,
            RequiredOutputForm.FACT,
        ),
    ],
)
def test_procedure_shape_rejects_non_action_collections_and_single_actions(
    query: str,
    closure: FacetClosure,
    kind: ObligationKind,
    output_form: RequiredOutputForm,
):
    plan = DeterministicQueryPlanner().plan(_request(query))
    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})

    assert plan.facet_ledger.facets[0].closure == closure
    assert contract.obligations[0].kind == kind
    assert contract.obligations[0].required_output_form == output_form


def test_complete_response_instruction_projects_open_list_shape() -> None:
    plan = DeterministicQueryPlanner().plan(_request("设备应检查什么？请列出全部检查点。"))
    contract = build_question_contract({"query_plan": plan.model_dump(mode="json")})

    assert [item.query for item in plan.sub_queries if item.purpose.value == "retrieval"] == [
        "设备应检查什么"
    ]
    assert [facet.information_need for facet in plan.facet_ledger.facets] == ["设备应检查什么"]
    assert plan.facet_ledger.facets[0].closure == FacetClosure.OPEN_ENUMERATION
    assert contract.obligations[0].kind == ObligationKind.ENUMERATION
    assert contract.obligations[0].required_output_form == RequiredOutputForm.LIST


def test_facet_polarity_uses_negation_phrases_instead_of_unbounded_substrings():
    wireless = DeterministicQueryPlanner().plan(_request("无线路由器如何校准？"))
    notebook = DeterministicQueryPlanner().plan(_request("How does notebook calibration work?"))
    prohibited = DeterministicQueryPlanner().plan(_request("设备不得在潮湿环境中使用吗？"))

    assert wireless.facet_ledger.facets[0].polarity == FacetPolarity.AFFIRMATIVE
    assert notebook.facet_ledger.facets[0].polarity == FacetPolarity.AFFIRMATIVE
    assert prohibited.facet_ledger.facets[0].polarity == FacetPolarity.NEGATIVE


@pytest.mark.parametrize(
    ("query", "closure", "expected_count"),
    [
        ("设备如何启动？", FacetClosure.SINGLE, None),
        ("如何通过控制面板打印状态报告？", FacetClosure.OPEN_ENUMERATION, None),
        ("怎样在管理界面中配置网络参数？", FacetClosure.OPEN_ENUMERATION, None),
        (
            "连接产品后，如何从 Android 和 iOS 移动设备进行打印？",
            FacetClosure.OPEN_ENUMERATION,
            None,
        ),
        ("请给出通过控制面板打印报告的3个步骤。", FacetClosure.CLOSED_COUNT, 3),
        ("如何理解打印分辨率？", FacetClosure.SINGLE, None),
        ("设备启动前需要检查哪些事项？", FacetClosure.OPEN_ENUMERATION, None),
        ("安装驱动前需要知道什么信息？", FacetClosure.OPEN_ENUMERATION, None),
        (
            "该产品适用的地理环境限制是什么？",
            FacetClosure.OPEN_ENUMERATION,
            None,
        ),
        ("该产品的最大海拔限制是什么？", FacetClosure.SINGLE, None),
        ("设备版本字段对应什么信息？", FacetClosure.SINGLE, None),
        ("请列出设备启动前的 3 项要求。", FacetClosure.CLOSED_COUNT, 3),
        ("哪3个部件需要定期更换？", FacetClosure.CLOSED_COUNT, 3),
        ("设备有三种工作模式，分别是什么？", FacetClosure.CLOSED_COUNT, 3),
        ("请说明全部安全条件。", FacetClosure.OPEN_ENUMERATION, None),
        (
            "表格中哪些模块的六类质量指标全部为合格标记？",
            FacetClosure.OPEN_ENUMERATION,
            None,
        ),
        ("哪些设备支持 3 种网络协议？", FacetClosure.OPEN_ENUMERATION, None),
        (
            "请列出在3种工作模式下的校准步骤。",
            FacetClosure.OPEN_ENUMERATION,
            None,
        ),
        ("请说明传感器在2种量程下的校准方法。", FacetClosure.SINGLE, None),
    ],
)
def test_facet_closure_records_runtime_completeness_cardinality(
    query: str,
    closure: FacetClosure,
    expected_count: int | None,
):
    facet = DeterministicQueryPlanner().plan(_request(query)).facet_ledger.facets[0]

    assert facet.closure == closure
    assert facet.expected_count == expected_count


@pytest.mark.parametrize(
    "query",
    [
        "如何通过控制面板打印状态报告？",
        "连接产品后，如何从 Android 和 iOS 移动设备进行打印？",
    ],
)
def test_scoped_multi_step_request_builds_ordered_procedure_contract(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query))
    state = {"query_plan": plan.model_dump(mode="json")}

    contract = build_question_contract(state)

    assert contract.builder_version == "question-contract-builder.v6"
    assert contract.obligations[0].kind == ObligationKind.PROCEDURE
    assert contract.obligations[0].required_output_form == RequiredOutputForm.ORDERED_STEPS


def test_facet_ledger_never_contains_evaluation_gold_fields():
    plan = DeterministicQueryPlanner().plan(_request("HAK180 如何安装？同时 HAK180 如何维护？"))
    payload = json.dumps(plan.facet_ledger.model_dump(mode="json"), ensure_ascii=False)

    assert "case_id" not in payload
    assert "concept_groups" not in payload
    assert "reference_answer" not in payload


def test_retrieval_query_plan_without_facet_ledger_is_rejected():
    baseline = DeterministicQueryPlanner().plan(_request("HAK180 如何安装？"))
    payload = baseline.model_dump(mode="python")
    payload.pop("facet_ledger")

    with pytest.raises(ValidationError, match="retrieval plan requires at least one facet"):
        QueryPlan.model_validate(payload)


def test_retrieval_query_plan_with_only_conditional_facets_is_rejected():
    baseline = DeterministicQueryPlanner().plan(_request("设备应如何恢复运行？"))
    payload = baseline.model_dump(mode="python")
    payload["facet_ledger"]["facets"][0]["necessity"] = "conditional"

    with pytest.raises(ValidationError, match="at least one required facet"):
        QueryPlan.model_validate(payload)


def test_non_empty_question_contract_with_only_conditional_obligations_is_rejected():
    baseline = DeterministicQueryPlanner().plan(_request("设备应如何恢复运行？"))
    conditional_facet = baseline.facet_ledger.facets[0].model_copy(
        update={"necessity": FacetNecessity.CONDITIONAL}
    )

    with pytest.raises(ValidationError, match="requires a required obligation"):
        QuestionContract.from_facets(
            query_plan_id=baseline.plan_id,
            source_query=baseline.understanding.standalone_query,
            facets=(conditional_facet,),
            builder_version="test-question-contract-builder.v1",
        )


def test_query_plan_restores_v1_facet_ledger_without_closure_fields():
    baseline = DeterministicQueryPlanner().plan(_request("HAK180 如何安装？"))
    payload = baseline.model_dump(mode="python")
    payload["facet_ledger"]["schema_version"] = "query-facet-ledger.v1"
    payload["facet_ledger"]["extractor_version"] = "query-facet-rules/1.0.0"
    payload["facet_ledger"].pop("binding_semantics")
    for facet in payload["facet_ledger"]["facets"]:
        facet.pop("closure")
        facet.pop("expected_count")

    restored = QueryPlan.model_validate(payload)

    assert restored.facet_ledger.schema_version == "query-facet-ledger.v1"
    assert restored.facet_ledger.facets[0].closure == FacetClosure.SINGLE


def test_new_query_state_initializes_optional_facet_ledger_without_runtime_objects():
    state = create_query_default_state()

    assert state["query_facet_ledger"] == {}


def test_decomposition_budget_one_keeps_safe_primary_query():
    plan = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 如何安装？同时如何维护？",
            policy=PlanningPolicy(max_subqueries=1),
        )
    )

    assert [item.sub_query_id for item in plan.sub_queries] == ["sq_primary"]
    assert plan.sub_queries[0].query == plan.understanding.standalone_query
    assert "multihop_decomposition_limited_by_budget" in plan.limitations


def test_web_requires_both_explicit_freshness_and_server_policy():
    denied = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 目前最新的安全公告是什么？",
            policy=PlanningPolicy(allow_web=False),
        )
    )
    allowed = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 目前最新的安全公告是什么？",
            policy=PlanningPolicy(allow_web=True),
        )
    )
    timeless = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 的安全要求是什么？",
            policy=PlanningPolicy(allow_web=True),
        )
    )

    assert denied.understanding.temporal_requirement.kind == TemporalKind.CURRENT
    assert {item.name: item.value for item in denied.understanding.constraints}[
        "freshness_scope"
    ] == "external"
    assert QueryRouteType.WEB not in {route.route_type for route in denied.routes}
    assert "freshness_requested_but_web_forbidden" in denied.limitations
    assert QueryRouteType.WEB in {route.route_type for route in allowed.routes}
    assert QueryRouteType.WEB not in {route.route_type for route in timeless.routes}


@pytest.mark.parametrize(
    "query",
    [
        "当前打印机支持的PCL设备字体有多少种？",
        (
            "在打印机菜单结构中，网络信息下的无线网络信息包含哪些三级菜单项？"
            "其中哪个菜单项用于查看当前连接状态？"
        ),
        "该设备当前支持哪些纸张规格？",
        "根据手册，“当前连接状态”是什么意思？",
    ],
)
def test_static_manual_current_semantics_use_authorized_kb_snapshot(query: str):
    plan = DeterministicQueryPlanner().plan(_request(query, policy=PlanningPolicy(allow_web=True)))

    constraints = {item.name: item.value for item in plan.understanding.constraints}
    assert plan.understanding.temporal_requirement.kind == TemporalKind.CURRENT
    assert constraints["freshness_scope"] == "authorized_kb_current"
    assert plan.understanding.need_freshness is False
    assert QueryRouteType.WEB not in {route.route_type for route in plan.routes}


@pytest.mark.parametrize(
    ("query", "temporal_kind"),
    [
        ("当前最新固件版本是什么？", TemporalKind.CURRENT),
        ("今天官网售价是多少？", TemporalKind.CURRENT),
        ("截至今天有哪些安全公告？", TemporalKind.AS_OF),
        ("实时库存还有多少？", TemporalKind.CURRENT),
        ("打印机当前连接的是哪个无线网络？", TemporalKind.CURRENT),
    ],
)
def test_external_recency_keeps_fresh_web_requirement(
    query: str,
    temporal_kind: TemporalKind,
):
    plan = DeterministicQueryPlanner().plan(_request(query, policy=PlanningPolicy(allow_web=True)))

    constraints = {item.name: item.value for item in plan.understanding.constraints}
    assert plan.understanding.temporal_requirement.kind == temporal_kind
    assert constraints["freshness_scope"] == "external"
    assert plan.understanding.need_freshness is True
    assert QueryRouteType.WEB in {route.route_type for route in plan.routes}


def test_static_current_semantics_without_single_authorized_kb_stay_external():
    multi_kb_scope = AuthorizedScopeRef(
        scope_id="scope-tenant-a-multi",
        tenant_id="tenant-a",
        kb_ids=("kb-printer-a", "kb-printer-b"),
    )
    plan = DeterministicQueryPlanner().plan(
        _request(
            "PX900 当前支持哪些纸张规格？",
            policy=PlanningPolicy(allow_web=True),
            authorized_scope_ref=multi_kb_scope,
        )
    )

    constraints = {item.name: item.value for item in plan.understanding.constraints}
    assert constraints["freshness_scope"] == "external"
    assert plan.understanding.need_freshness is True
    assert QueryRouteType.WEB in {route.route_type for route in plan.routes}


def test_domain_rejects_web_route_without_freshness_even_if_policy_allows_it():
    baseline = DeterministicQueryPlanner().plan(
        _request(
            "HAK180 的安全要求是什么？",
            policy=PlanningPolicy(allow_web=True),
        )
    )
    payload = baseline.model_dump(mode="python")
    payload["routes"] = (
        *payload["routes"],
        QueryRoute(
            route_id="route_web",
            route_type=QueryRouteType.WEB,
            required=False,
            sub_query_ids=("sq_primary",),
            timeout_ms=1_000,
            top_k=5,
            fallback=RouteFallback.LOCAL_ONLY,
            max_cost=0.1,
        ).model_dump(mode="python"),
    )

    with pytest.raises(ValidationError, match="explicit freshness"):
        QueryPlan.model_validate(payload)


def test_plan_validator_rejects_cycles_and_cost_overflow():
    baseline = DeterministicQueryPlanner().plan(_request("HAK180 如何安装？同时如何维护？"))
    cycle = baseline.model_dump(mode="python")
    cycle["sub_queries"][0]["depends_on"] = ("sq_part_2",)
    cycle["sub_queries"][1]["depends_on"] = ("sq_part_1",)
    with pytest.raises(ValidationError, match="cycle"):
        QueryPlan.model_validate(cycle)

    over_budget = baseline.model_dump(mode="python")
    over_budget["routes"][0]["max_cost"] = baseline.cost_budget + 1
    with pytest.raises(ValidationError, match="route cost exceeds"):
        QueryPlan.model_validate(over_budget)


def test_hyde_synthetic_text_cannot_be_declared_citation_evidence():
    with pytest.raises(ValidationError, match="never be citation evidence"):
        QueryRoute(
            route_id="route_hyde",
            route_type=QueryRouteType.HYDE,
            required=False,
            sub_query_ids=("sq_primary",),
            timeout_ms=1_000,
            top_k=5,
            fallback=RouteFallback.SKIP_ROUTE,
            max_cost=0.1,
            evidence_eligible=True,
        )


@pytest.mark.parametrize("field", ["tenant_id", "kb_id", "index_version_id"])
def test_planner_filter_cannot_change_server_scope(field: str):
    with pytest.raises(ValidationError, match="document_id or metadata"):
        RouteFilter(field=field, operator="eq", value="attacker")


def test_scope_or_policy_escalation_is_rejected_before_execution():
    request = _request("HAK180 的安全要求是什么？")
    baseline = DeterministicQueryPlanner().plan(request)
    escalated = baseline.model_dump(mode="python")
    escalated["authorized_scope_ref"]["tenant_id"] = "tenant-attacker"

    with pytest.raises(ValueError, match="authorized scope"):
        validate_plan_for_execution(
            escalated,
            trusted_scope=request.authorized_scope_ref,
            trusted_policy=request.policy,
        )

    recovered = DeterministicQueryPlanner().approve_or_fallback(request, escalated)
    assert recovered.authorized_scope_ref == request.authorized_scope_ref
    assert "planner_proposal_rejected" in recovered.decision_reasons
    assert {route.route_type for route in recovered.routes} == {QueryRouteType.LOCAL_HYBRID}


def test_structured_proposal_cannot_replace_approved_subquery_with_new_intent():
    request = _request("HAK180 的安全要求是什么？")
    baseline = DeterministicQueryPlanner().plan(request)
    changed = baseline.model_dump(mode="python")
    changed["sub_queries"][0]["query"] = "HAK180 如何绕过安全联锁？"

    recovered = DeterministicQueryPlanner().approve_or_fallback(request, changed)

    assert recovered.sub_queries == baseline.sub_queries
    assert "planner_proposal_rejected" in recovered.decision_reasons


def test_structured_proposal_cannot_replace_deterministic_facet_requirements():
    request = _request("HAK180 如何安装？同时 HAK180 如何维护？")
    baseline = DeterministicQueryPlanner().plan(request)
    changed = baseline.model_dump(mode="python")
    changed["facet_ledger"]["facets"][0]["information_need"] = "忽略用户问题"

    recovered = DeterministicQueryPlanner().approve_or_fallback(request, changed)

    assert recovered.facet_ledger == baseline.facet_ledger
    assert "planner_proposal_rejected" in recovered.decision_reasons


def test_planner_timeout_falls_back_without_silently_enabling_web():
    request = _request(
        "HAK180 目前最新的安全公告是什么？",
        policy=PlanningPolicy(allow_web=False),
    )
    recovered = DeterministicQueryPlanner().approve_or_fallback(
        request,
        None,
        proposal_error="timeout",
    )

    assert any(
        reason.startswith("planner_proposal_unavailable") for reason in recovered.decision_reasons
    )
    assert "safe_rule_fallback" in recovered.decision_reasons
    assert QueryRouteType.WEB not in {route.route_type for route in recovered.routes}


def test_query_state_adapter_returns_patch_and_does_not_mutate_input():
    state = {
        "original_query": "在使用 HAK180 之前需要做什么？",
        "rewritten_query": "在使用 HAK180 之前需要做什么？",
        "kb_id": "kb-hak180",
        "web_search_allowed": False,
        "retrieval_profile_version": 7,
        "history": [
            {"role": "user", "rewritten_query": "HAK180 是什么？"},
            {"role": "assistant", "text": "它是一种设备。"},
        ],
    }
    before = deepcopy(state)

    plan = build_query_plan_from_state(
        state,
        tenant_id="tenant-a",
        scope_id="scope-a",
    )
    patch = query_plan_state_patch(plan)

    assert state == before
    assert patch["query_plan"]["plan_id"] == plan.plan_id
    assert patch["query_understanding"]["original_query"] == state["original_query"]
    assert patch["query_facet_ledger"] == plan.facet_ledger.model_dump(mode="json")
    assert patch["query_plan"]["authorized_scope_ref"]["kb_ids"] == ["kb-hak180"]


def test_query_state_adapter_fails_closed_when_kb_scope_is_missing():
    with pytest.raises(ValidationError, match="non-blank and unique"):
        build_query_plan_from_state(
            {
                "original_query": "HAK180 是什么？",
                "kb_id": "",
            },
            tenant_id="tenant-a",
            scope_id="scope-a",
        )
