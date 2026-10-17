"""由 TODO-10 QueryPolicyExecutor 托管真实在线检索路线。

QueryPlan 仍是唯一的路线事实；本节点不会新增 local、HyDE 或 Web 路线。同步 SDK
运行在有界共享线程池中，因此执行器可以在 route deadline 到达时停止等待并阻止
结果进入 RRF/生成。Python 无法强杀已经进入第三方同步 SDK 的线程；超时后的底层
调用可能在隔离的 state 副本上继续到 SDK 自身超时，但不会再提交候选或查询事件。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
import re
import sys
from typing import Any, Callable

from app.application.query_policy_service import (
    CancellationSignal,
    CircuitBreaker,
    QueryPolicyExecutor,
    RouteInvocation,
    RouteInvocationError,
)
from app.application.query_security_pipeline import authorize_external_call
from app.domain.query_planning import QueryPlan, QueryRoute, QueryRouteType
from app.domain.query_resilience import DegradationReason, RouteOutcome, RouteUsage
from app.rag.query.embedding_search_service import search_by_embedding
from app.rag.query.hyde_search_service import search_embedding_hyde
from app.rag.query.web_search_service import search_by_web
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id


_SAFE_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
# 多个 QueryRun 共享有界池和 route 级 breaker，避免每次运行重新得到“全新熔断器”。
_SYNC_ROUTE_POOL = ThreadPoolExecutor(
    max_workers=12,
    thread_name_prefix="query-route",
)
_ROUTE_CIRCUIT_BREAKERS = {
    route_type.value: CircuitBreaker()
    for route_type in (
        QueryRouteType.LOCAL_HYBRID,
        QueryRouteType.HYDE,
        QueryRouteType.WEB,
    )
}


def _error_code(exc: BaseException, default: str) -> str:
    candidate = str(getattr(exc, "code", "") or getattr(exc, "reason_code", "")).upper()
    return candidate if _SAFE_ERROR_CODE.fullmatch(candidate) else default


async def _run_sync(function: Callable[[], RouteInvocation]) -> RouteInvocation:
    """在共享有界池执行同步依赖，并传播 Principal/trace ContextVar。"""

    loop = asyncio.get_running_loop()
    context = copy_context()
    return await loop.run_in_executor(_SYNC_ROUTE_POOL, context.run, function)


def _route_usage(route: QueryRoute, *, llm_calls: int = 0) -> RouteUsage:
    # 现有 SDK 不回传统一 token/cost；用计划中的 max_cost 作保守上界并明确 estimated。
    return RouteUsage(
        llm_calls=llm_calls,
        cost=route.max_cost,
        estimated=True,
    )


def _payload(
    *,
    chunks_key: str,
    chunks: list[dict[str, Any]],
    diagnostics_key: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunks_key": chunks_key,
        "chunks": chunks,
        "diagnostics_key": diagnostics_key,
        "diagnostics": diagnostics,
    }


def _local_invocation(
    state: dict[str, Any],
    route: QueryRoute,
    cancellation: CancellationSignal,
) -> RouteInvocation:
    scoped = deepcopy(state)
    try:
        chunks = search_by_embedding(scoped, cancellation=cancellation.is_set)
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
    except RouteInvocationError:
        raise
    except Exception as exc:
        raise RouteInvocationError(
            _error_code(exc, "LOCAL_RETRIEVAL_FAILED"),
        ) from exc
    diagnostics = {
        "route": route.route_type.value,
        "route_id": route.route_id,
        **dict(scoped.get("embedding_retrieval_diagnostics") or {}),
    }
    return RouteInvocation(
        payload=_payload(
            chunks_key="embedding_chunks",
            chunks=chunks,
            diagnostics_key="embedding_retrieval_diagnostics",
            diagnostics=diagnostics,
        ),
        candidate_count=len(chunks),
        usage=_route_usage(route),
    )


def _hyde_invocation(
    state: dict[str, Any],
    route: QueryRoute,
    cancellation: CancellationSignal,
) -> RouteInvocation:
    scoped = deepcopy(state)
    try:
        chunks = search_embedding_hyde(
            scoped,
            cancellation=cancellation.is_set,
            timeout_seconds=max(0.001, route.timeout_ms / 1000.0),
        )
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
    except RouteInvocationError:
        raise
    except Exception as exc:
        raise RouteInvocationError(
            _error_code(exc, "HYDE_FAILED"),
        ) from exc
    diagnostics = {
        "route": route.route_type.value,
        "route_id": route.route_id,
        **dict(scoped.get("hyde_retrieval_diagnostics") or {}),
    }
    return RouteInvocation(
        payload=_payload(
            chunks_key="hyde_embedding_chunks",
            chunks=chunks,
            diagnostics_key="hyde_retrieval_diagnostics",
            diagnostics=diagnostics,
        ),
        candidate_count=len(chunks),
        usage=_route_usage(route, llm_calls=1),
    )


def _web_invocation(
    state: dict[str, Any],
    route: QueryRoute,
    cancellation: CancellationSignal,
) -> RouteInvocation:
    scoped = deepcopy(state)
    try:
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
        authorize_external_call(
            scoped,
            purpose="web_search",
            provider="dashscope-web",
        )
        plan = scoped.get("query_plan") or {}
        subqueries = {
            str(item.get("sub_query_id")): str(item.get("query") or "")
            for item in (plan.get("sub_queries") or [])
            if item.get("purpose") == "retrieval" and item.get("sub_query_id") and item.get("query")
        }
        approved = [
            (sub_query_id, subqueries[sub_query_id])
            for sub_query_id in route.sub_query_ids
            if sub_query_id in subqueries
        ]
        if not approved:
            approved = [(None, str(scoped.get("rewritten_query") or ""))]
        pages: list[dict[str, Any]] = []
        subquery_diagnostics: list[dict[str, Any]] = []
        for sub_query_id, query in approved:
            if cancellation.is_set():
                raise RouteInvocationError("QUERY_CANCELLED")
            current = search_by_web({**scoped, "rewritten_query": query}, count=10)
            if cancellation.is_set():
                raise RouteInvocationError("QUERY_CANCELLED")
            pages.extend({**document, "sub_query_id": sub_query_id} for document in current)
            subquery_diagnostics.append(
                {
                    "sub_query_id": sub_query_id,
                    "status": "succeeded",
                    "candidate_count": len(current),
                }
            )
    except RouteInvocationError:
        raise
    except Exception as exc:
        reason = (
            DegradationReason.SECURITY_POLICY_BLOCKED
            if getattr(exc, "reason_code", None)
            else DegradationReason.OPTIONAL_ROUTE_FAILED
        )
        raise RouteInvocationError(
            _error_code(exc, "WEB_RETRIEVAL_FAILED"),
            degradation_reason=reason,
        ) from exc
    diagnostics = {
        "route": route.route_type.value,
        "route_id": route.route_id,
        "query_variant": "web",
        "status": "succeeded",
        "subqueries": subquery_diagnostics,
        "candidate_count": len(pages),
    }
    return RouteInvocation(
        payload=_payload(
            chunks_key="web_search_docs",
            chunks=pages,
            diagnostics_key="web_retrieval_diagnostics",
            diagnostics=diagnostics,
        ),
        candidate_count=len(pages),
        usage=_route_usage(route),
    )


def _handlers(state: dict[str, Any]):
    async def local(
        route: QueryRoute,
        cancellation: CancellationSignal,
    ) -> RouteInvocation:
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
        return await _run_sync(lambda: _local_invocation(state, route, cancellation))

    async def hyde(
        route: QueryRoute,
        cancellation: CancellationSignal,
    ) -> RouteInvocation:
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
        return await _run_sync(lambda: _hyde_invocation(state, route, cancellation))

    async def web(
        route: QueryRoute,
        cancellation: CancellationSignal,
    ) -> RouteInvocation:
        if cancellation.is_set():
            raise RouteInvocationError("QUERY_CANCELLED")
        return await _run_sync(lambda: _web_invocation(state, route, cancellation))

    return {
        QueryRouteType.LOCAL_HYBRID.value: local,
        QueryRouteType.HYDE.value: hyde,
        QueryRouteType.WEB.value: web,
    }


def _failure_diagnostics(outcome: RouteOutcome) -> dict[str, Any]:
    return {
        "route": outcome.route_type,
        "route_id": outcome.route_id,
        "status": outcome.status.value,
        "candidate_count": 0,
        "error_code": outcome.error_code,
        "degradation_reason": (
            outcome.degradation_reason.value if outcome.degradation_reason else None
        ),
    }


@node_log("node_retrieval_policy")
def node_retrieval_policy(state: dict[str, Any]) -> dict[str, Any]:
    """只执行 QueryPlan 已批准的路线，并在进入 RRF 前落实失败语义。"""

    plan = QueryPlan.model_validate(state.get("query_plan") or {})
    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    summary = asyncio.run(
        QueryPolicyExecutor(
            _handlers(state),
            circuit_breakers=_ROUTE_CIRCUIT_BREAKERS,
        ).execute(plan)
    )
    patch: dict[str, Any] = {
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "embedding_retrieval_diagnostics": {},
        "hyde_retrieval_diagnostics": {},
        "web_retrieval_diagnostics": {},
    }
    diagnostics_key_by_route = {
        QueryRouteType.LOCAL_HYBRID.value: "embedding_retrieval_diagnostics",
        QueryRouteType.HYDE.value: "hyde_retrieval_diagnostics",
        QueryRouteType.WEB.value: "web_retrieval_diagnostics",
    }
    for outcome in summary.outcomes:
        payload = outcome.payload if isinstance(outcome.payload, dict) else None
        if payload is not None:
            chunks_key = str(payload["chunks_key"])
            diagnostics_key = str(payload["diagnostics_key"])
            patch[chunks_key] = list(payload["chunks"])
            patch[diagnostics_key] = dict(payload["diagnostics"])
            continue
        diagnostics_key = diagnostics_key_by_route.get(outcome.route_type)
        if diagnostics_key:
            patch[diagnostics_key] = _failure_diagnostics(outcome)

    safe_summary = summary.model_dump(mode="json")
    for outcome in safe_summary["outcomes"]:
        outcome.pop("payload", None)
    patch["query_execution_summary"] = safe_summary
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return patch


__all__ = ["node_retrieval_policy"]
