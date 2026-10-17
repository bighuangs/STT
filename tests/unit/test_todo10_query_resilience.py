from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from app.application.query_policy_service import QueryPolicyExecutor, RouteInvocation
from app.application.query_understanding_service import (
    DeterministicQueryPlanner,
    QueryPlanningInput,
)
from app.domain.query_planning import AuthorizedScopeRef, PlanningPolicy
from app.domain.query_resilience import (
    BudgetExceededError,
    BudgetLedger,
    DegradationReason,
    QueryBudget,
    RequiredRouteError,
    RouteUsage,
)


def _plan(query: str, *, allow_web: bool = False):
    return DeterministicQueryPlanner().plan(
        QueryPlanningInput(
            original_query=query,
            authorized_scope_ref=AuthorizedScopeRef(
                scope_id="scope-tenant-a-kb-hak180",
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
            deadline_at=datetime.now(UTC) + timedelta(seconds=1),
        )
    )


def test_budget_ledger_never_silently_exceeds_server_policy() -> None:
    ledger = BudgetLedger(
        QueryBudget(deadline_ms=1_000, max_llm_calls=1, max_tokens=100, max_cost=0.1)
    )
    ledger.record(
        RouteUsage(llm_calls=1, input_tokens=40, output_tokens=20, cost=0.08)
    )

    assert ledger.can_start(max_cost=0.03) == (
        False,
        DegradationReason.BUDGET_EXHAUSTED,
    )
    with pytest.raises(BudgetExceededError, match="max_llm_calls"):
        ledger.record(RouteUsage(llm_calls=1))


def test_required_local_route_succeeds_and_is_the_only_evidence_source() -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")

    async def local(route, cancellation):
        assert route.required is True
        assert cancellation.is_set() is False
        return RouteInvocation(
            payload=[{"chunk_id": "hak180-p3"}],
            candidate_count=1,
            usage=RouteUsage(),
        )

    summary = asyncio.run(
        QueryPolicyExecutor({"local_hybrid": local}).execute(plan)
    )

    assert summary.answer_disposition == "answer"
    assert summary.evidence_count == 1
    assert summary.degraded is False
    assert [item.route_type for item in summary.outcomes] == ["local_hybrid"]


def test_optional_route_timeout_degrades_but_does_not_hide_local_result() -> None:
    plan = _plan("HAK180 当前最新的安全要求是什么？", allow_web=True)
    assert any(not route.required for route in plan.routes)
    routes = tuple(
        route.model_copy(update={"timeout_ms": 10})
        if not route.required
        else route
        for route in plan.routes
    )
    plan = plan.model_copy(update={"routes": routes})

    async def local(route, cancellation):
        return RouteInvocation(payload=["local"], candidate_count=1)

    async def slow(route, cancellation):
        await asyncio.sleep(0.05)
        return RouteInvocation(payload=["external"], candidate_count=1)

    handlers = {
        "local_hybrid": local,
        "hyde": slow,
        "web": slow,
    }
    summary = asyncio.run(QueryPolicyExecutor(handlers).execute(plan))

    assert summary.answer_disposition == "partial"
    assert summary.evidence_count == 1
    assert summary.degraded is True
    assert DegradationReason.OPTIONAL_ROUTE_TIMEOUT in summary.degradation_reasons


def test_required_route_failure_is_retryable_failure_not_web_substitution() -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")

    async def failed(route, cancellation):
        raise ConnectionError("milvus unavailable")

    with pytest.raises(RequiredRouteError) as captured:
        asyncio.run(QueryPolicyExecutor({"local_hybrid": failed}).execute(plan))

    assert captured.value.route_id == "route_local_hybrid"
    assert captured.value.error_code == "ROUTE_EXECUTION_FAILED"


def test_required_timeout_cooperatively_stops_the_late_sync_handler() -> None:
    plan = _plan("在使用 HAK180 之前需要做什么？")
    plan = plan.model_copy(
        update={
            "routes": tuple(
                route.model_copy(update={"timeout_ms": 10})
                for route in plan.routes
            )
        }
    )
    cancellation_observed = Event()

    def blocking_local(route, cancellation):
        del route
        while not cancellation.is_set():
            cancellation_observed.wait(timeout=0.005)
        cancellation_observed.set()
        return RouteInvocation(payload=[], candidate_count=0)

    with pytest.raises(RequiredRouteError) as captured:
        asyncio.run(
            QueryPolicyExecutor({"local_hybrid": blocking_local}).execute(plan)
        )

    assert captured.value.error_code == "ROUTE_TIMEOUT"
    assert cancellation_observed.wait(timeout=0.2)


def test_optional_timeout_cancels_only_its_own_late_sync_handler() -> None:
    plan = _plan("HAK180 当前最新的安全要求是什么？", allow_web=True)
    local_route = plan.routes[0].model_copy(update={"timeout_ms": 200})
    optional_route = plan.routes[-1].model_copy(
        update={"required": False, "timeout_ms": 10}
    )
    plan = plan.model_copy(update={"routes": (local_route, optional_route)})
    optional_cancelled = Event()
    local_saw_cancellation = [False]

    def local(route, cancellation):
        del route
        cancellation_observed = cancellation.is_set()
        local_saw_cancellation[0] = cancellation_observed
        return RouteInvocation(payload=["local"], candidate_count=1)

    def optional(route, cancellation):
        del route
        while not cancellation.is_set():
            optional_cancelled.wait(timeout=0.005)
        optional_cancelled.set()
        return RouteInvocation(payload=["late"], candidate_count=1)

    summary = asyncio.run(
        QueryPolicyExecutor(
            {
                local_route.route_type.value: local,
                optional_route.route_type.value: optional,
            }
        ).execute(plan)
    )

    assert local_saw_cancellation == [False]
    assert optional_cancelled.wait(timeout=0.2)
    assert summary.evidence_count == 1
    assert summary.answer_disposition == "partial"
    assert summary.outcomes[-1].error_code == "OPTIONAL_ROUTE_TIMEOUT"
