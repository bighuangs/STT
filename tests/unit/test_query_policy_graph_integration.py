from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from app.application.query_understanding_service import (
    DeterministicQueryPlanner,
    QueryPlanningInput,
)
from app.domain.query_planning import (
    AuthorizedScopeRef,
    PlanningPolicy,
    QueryRouteType,
    RouteFallback,
)
from app.domain.query_resilience import RequiredRouteError
from app.process.query.agent.nodes import node_retrieval_policy as policy_node
from app.process.query.agent.nodes import node_search_embedding as local_node
from app.process.query.agent.state import create_query_default_state


def _plan(query: str, *, allow_web: bool = False):
    return DeterministicQueryPlanner().plan(
        QueryPlanningInput(
            original_query=query,
            authorized_scope_ref=AuthorizedScopeRef(
                scope_id="scope-query-policy-integration",
                tenant_id="tenant-a",
                kb_ids=("kb-hak180",),
            ),
            profile_version=3,
            policy=PlanningPolicy(
                allow_web=allow_web,
                total_timeout_ms=1_000,
                token_budget=1_000,
                cost_budget=1.0,
            ),
            deadline_at=datetime.now(UTC) + timedelta(seconds=2),
        )
    )


def _state(plan) -> dict:
    return create_query_default_state(
        session_id="session-policy",
        conversation_id="conversation-policy",
        turn_id="turn-policy",
        query_run_id="qrun-policy",
        trace_id="qrun-policy",
        tenant_id="tenant-a",
        user_id="user-a",
        kb_id="kb-hak180",
        original_query=plan.understanding.original_query,
        rewritten_query=plan.understanding.standalone_query,
        questions=[plan.understanding.standalone_query],
        query_plan=plan.model_dump(mode="json"),
        deadline_at=plan.deadline_at.isoformat(),
        retrieval_profile_id="rpf-policy",
        retrieval_profile_version=3,
        is_stream=False,
    )


def _disable_progress(monkeypatch) -> None:
    monkeypatch.setattr(policy_node, "add_running_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(policy_node, "add_done_task", lambda *_args, **_kwargs: None)


def test_legacy_local_node_fails_immediately_when_plan_marks_route_required(
    monkeypatch,
) -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")
    monkeypatch.setattr(
        local_node,
        "search_by_embedding",
        lambda _state, **_kwargs: (_ for _ in ()).throw(
            ConnectionError("milvus unavailable")
        ),
    )
    monkeypatch.setattr(local_node, "add_running_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(local_node, "add_done_task", lambda *_args, **_kwargs: None)

    with pytest.raises(RequiredRouteError) as captured:
        local_node.node_search_embedding(_state(plan))

    assert captured.value.route_id == "route_local_hybrid"
    assert captured.value.error_code == "LOCAL_RETRIEVAL_FAILED"


def test_production_graph_routes_retrieval_through_policy_node() -> None:
    from app.process.query.agent.main_graph import (
        build_query_graph,
        node_question_contract_after_router,
        node_query_rewrite_after_router,
    )

    plan = _plan("在使用 HAK180 之前需要做什么？")
    assert node_query_rewrite_after_router(_state(plan)) == "node_question_contract"
    assert node_question_contract_after_router(_state(plan)) == "node_retrieval_policy"
    graph_nodes = build_query_graph().nodes
    assert "node_retrieval_policy" in graph_nodes
    assert "node_search_embedding" not in graph_nodes
    assert "node_search_embedding_hyde" not in graph_nodes
    assert "node_web_search_mcp" not in graph_nodes


def test_manual_install_image_question_routes_through_retrieval_contract() -> None:
    """产品手册中的“界面”不是 STT 页面帮助，必须进入真实检索链。"""

    from app.process.query.agent.main_graph import (
        node_question_contract_after_router,
        node_query_rewrite_after_router,
    )

    plan = _plan(
        "驱动搜索完成后应如何选择打印机？"
        "请展示手册中的安装界面图片，并标明页码。"
    )

    assert plan.disposition.value == "retrieve"
    assert plan.understanding.need_retrieval is True
    assert plan.response_key is None
    assert {route.route_type.value for route in plan.routes} >= {"local_hybrid"}
    assert node_query_rewrite_after_router(_state(plan)) == "node_question_contract"
    assert node_question_contract_after_router(_state(plan)) == "node_retrieval_policy"


def test_policy_node_executes_real_local_handler_and_returns_safe_summary(
    monkeypatch,
) -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")
    _disable_progress(monkeypatch)

    def local(state, **_kwargs):
        state["embedding_retrieval_diagnostics"] = {
            "query_variant": "base",
            "status": "succeeded",
            "candidate_count": 1,
        }
        return [{"chunk_id": "chunk-hak180"}]

    monkeypatch.setattr(policy_node, "search_by_embedding", local)
    result = policy_node.node_retrieval_policy(_state(plan))

    assert result["embedding_chunks"] == [{"chunk_id": "chunk-hak180"}]
    assert result["query_execution_summary"]["answer_disposition"] == "answer"
    assert result["query_execution_summary"]["outcomes"][0]["status"] == "succeeded"
    assert "payload" not in result["query_execution_summary"]["outcomes"][0]


def test_policy_node_times_out_optional_web_without_hiding_local_evidence(
    monkeypatch,
) -> None:
    plan = _plan("HAK180 当前最新的安全要求是什么？", allow_web=True)
    plan = plan.model_copy(
        update={
            "routes": tuple(
                route.model_copy(update={"timeout_ms": 100}) if not route.required else route
                for route in plan.routes
            )
        }
    )
    _disable_progress(monkeypatch)

    def local(state, **_kwargs):
        state["embedding_retrieval_diagnostics"] = {
            "query_variant": "base",
            "status": "succeeded",
            "candidate_count": 1,
        }
        return [{"chunk_id": "chunk-local"}]

    release_web = Event()
    web_started = Event()
    web_finished = Event()

    def slow_web(_state, count=10):
        web_started.set()
        release_web.wait(timeout=2)
        web_finished.set()
        return [{"url": "https://example.com", "title": "late"}]

    monkeypatch.setattr(policy_node, "search_by_embedding", local)
    monkeypatch.setattr(policy_node, "search_by_web", slow_web)
    monkeypatch.setattr(policy_node, "authorize_external_call", lambda *_args, **_kwargs: None)

    try:
        result = policy_node.node_retrieval_policy(_state(plan))

        assert web_started.is_set()
        # route timeout 已经返回；物理同步线程尚未结束且其迟到结果没有提交。
        assert web_finished.is_set() is False
        assert result["embedding_chunks"] == [{"chunk_id": "chunk-local"}]
        assert result["web_search_docs"] == []
        assert result["web_retrieval_diagnostics"]["status"] == "timed_out"
        assert result["query_execution_summary"]["degradation_reasons"] == [
            "optional_route_timeout"
        ]
        assert result["query_execution_summary"]["answer_disposition"] == "partial"
    finally:
        # Python 无法强杀同步 SDK 线程；显式放行隔离线程，避免影响后续测试退出。
        release_web.set()
        assert web_finished.wait(timeout=1)


def test_policy_node_propagates_route_local_cancellation_to_optional_hyde(
    monkeypatch,
) -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")
    local_route = plan.routes[0].model_copy(update={"timeout_ms": 500})
    hyde_route = local_route.model_copy(
        update={
            "route_id": "route_hyde_test",
            "route_type": QueryRouteType.HYDE,
            "required": False,
            "timeout_ms": 100,
            "fallback": RouteFallback.SKIP_ROUTE,
            "evidence_eligible": False,
        }
    )
    plan = plan.model_copy(update={"routes": (local_route, hyde_route)})
    _disable_progress(monkeypatch)
    hyde_cancelled = Event()

    def local(state, **_kwargs):
        state["embedding_retrieval_diagnostics"] = {
            "status": "succeeded",
            "candidate_count": 1,
        }
        return [{"chunk_id": "chunk-local"}]

    def late_hyde(_state, *, cancellation, **_kwargs):
        while not cancellation():
            hyde_cancelled.wait(timeout=0.005)
        hyde_cancelled.set()
        return [{"chunk_id": "chunk-late"}]

    monkeypatch.setattr(policy_node, "search_by_embedding", local)
    monkeypatch.setattr(policy_node, "search_embedding_hyde", late_hyde)

    result = policy_node.node_retrieval_policy(_state(plan))

    assert hyde_cancelled.wait(timeout=0.2)
    assert result["embedding_chunks"] == [{"chunk_id": "chunk-local"}]
    assert result["hyde_embedding_chunks"] == []
    assert result["hyde_retrieval_diagnostics"]["status"] == "timed_out"
    assert result["query_execution_summary"]["answer_disposition"] == "partial"
