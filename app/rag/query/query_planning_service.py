"""TODO-27 查询图的无副作用 QueryPlan 适配入口。

本文件只把既有 state 快照转换为领域输入，不读取 Mongo/SQL，也不修改公共
``QueryGraphState``。主图接入时可将返回的 JSON 对象写入自己的专属状态字段。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.application.query_understanding_service import (
    DeterministicQueryPlanner,
    QueryPlanningInput,
)
from app.domain.query_planning import (
    AuthorizedScopeRef,
    ConversationTurn,
    PlanningPolicy,
    QueryPlan,
    RouteFilter,
)


def _history_snapshot(raw_history: list[dict] | tuple[dict, ...]) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    for item in raw_history[-20:]:
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        text = item.get("rewritten_query") if role == "user" else item.get("text")
        normalized = str(text or "").strip()
        if normalized:
            turns.append(ConversationTurn(role=role, text=normalized))
    return tuple(turns)


def build_query_plan_from_state(
    state: dict[str, Any],
    *,
    tenant_id: str,
    scope_id: str,
    document_ids: tuple[str, ...] = (),
    policy: PlanningPolicy | None = None,
    route_filters: tuple[RouteFilter, ...] = (),
    deadline_at: datetime | None = None,
) -> QueryPlan:
    """从调用方已加载的 state/history 生成经过验证的计划。

    ``tenant_id`` 和 ``scope_id`` 必须来自认证/授权层，而非客户端请求。KB 仅取
    当前会话已经固定的 ``state["kb_id"]``；空 KB 会在领域模型入口 fail closed。
    """

    kb_id = str(state.get("kb_id") or "").strip()
    trusted_policy = policy or PlanningPolicy(
        allow_web=bool(state.get("web_search_allowed", False)),
    )
    request = QueryPlanningInput(
        original_query=str(state.get("original_query") or ""),
        history=_history_snapshot(tuple(state.get("history") or ())),
        rewrite_candidate=str(state.get("rewritten_query") or "").strip() or None,
        authorized_scope_ref=AuthorizedScopeRef(
            scope_id=scope_id,
            tenant_id=tenant_id,
            kb_ids=(kb_id,),
            document_ids=document_ids,
        ),
        profile_version=max(0, int(state.get("retrieval_profile_version") or 0)),
        policy=trusted_policy,
        deadline_at=deadline_at,
        route_filters=route_filters,
    )
    return DeterministicQueryPlanner().plan(request)


def query_plan_state_patch(plan: QueryPlan) -> dict[str, dict[str, Any]]:
    """返回可由编排层显式合并的 JSON patch，不原地修改 state。"""

    return {
        "query_understanding": plan.understanding.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "query_facet_ledger": plan.facet_ledger.model_dump(
            mode="json",
        ),
        "query_plan": plan.model_dump(mode="json", exclude_none=True),
    }


__all__ = ["build_query_plan_from_state", "query_plan_state_patch"]
