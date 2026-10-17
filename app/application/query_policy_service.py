"""TODO-10 受控并发路由执行器。

调用者把已经验证的 QueryPlan 和每条 route 的实现传入。执行器负责 deadline、
bulkhead、circuit breaker、预算、必选/可选失败语义和确定性聚合；它不会决定新增
路线，因此安全策略不能被运行时 fallback 绕过。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import iscoroutinefunction
from threading import Event, Lock
from time import monotonic
from typing import Any, Protocol

from app.domain.query_planning import QueryPlan, QueryRoute
from app.domain.query_resilience import (
    BudgetLedger,
    DegradationReason,
    QueryBudget,
    QueryExecutionSummary,
    RequiredRouteError,
    RouteExecutionStatus,
    RouteOutcome,
    RouteUsage,
    stable_degradation_reasons,
)


@dataclass(frozen=True)
class RouteInvocation:
    payload: Any
    candidate_count: int = 0
    usage: RouteUsage = RouteUsage()


class RouteInvocationError(RuntimeError):
    """路线适配器给执行器的稳定失败，不携带查询或证据正文。"""

    def __init__(
        self,
        error_code: str,
        *,
        degradation_reason: DegradationReason = DegradationReason.OPTIONAL_ROUTE_FAILED,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.degradation_reason = degradation_reason


class CancellationSignal(Protocol):
    """同步/异步路线共享的最小取消合同。"""

    def is_set(self) -> bool: ...


class _LinkedRouteCancellation:
    """路线局部取消与 QueryRun 全局取消的逻辑或。"""

    def __init__(self, parent: Event) -> None:
        self._parent = parent
        self._local = Event()

    def is_set(self) -> bool:
        return self._local.is_set() or self._parent.is_set()

    def set(self) -> None:
        self._local.set()


RouteHandler = Callable[
    [QueryRoute, CancellationSignal],
    RouteInvocation | Awaitable[RouteInvocation],
]


class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 3, recovery_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if monotonic() - self.opened_at >= self.recovery_seconds:
                self.failures = 0
                self.opened_at = None
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = monotonic()


def _effective_deadline_ms(plan: QueryPlan) -> int:
    """取发布策略与 QueryRun 绝对截止时间的交集。"""

    deadline_ms = int(plan.policy.total_timeout_ms)
    if plan.deadline_at is None:
        return deadline_ms
    deadline = plan.deadline_at
    if deadline.tzinfo is None:  # QueryPlan 自身也会 fail closed，此处仅防御。
        deadline = deadline.replace(tzinfo=UTC)
    remaining_ms = int((deadline.astimezone(UTC) - datetime.now(UTC)).total_seconds() * 1000)
    if remaining_ms < 100:
        raise TimeoutError("query deadline exceeded before route execution")
    return min(deadline_ms, remaining_ms)


class QueryPolicyExecutor:
    def __init__(
        self,
        handlers: Mapping[str, RouteHandler],
        *,
        circuit_breakers: Mapping[str, CircuitBreaker] | None = None,
    ) -> None:
        self.handlers = dict(handlers)
        self.circuit_breakers = dict(circuit_breakers or {})

    async def execute(
        self,
        plan: QueryPlan,
        *,
        cancellation: Event | None = None,
    ) -> QueryExecutionSummary:
        """并发执行计划，结果顺序始终与 QueryPlan 一致。"""

        # 路线可以进入 ``to_thread``/第三方同步 SDK；threading.Event 才能作为
        # 受支持的跨线程协作取消信号。异步 handler 只使用 set/is_set，不阻塞等待。
        cancellation = cancellation or Event()
        budget = QueryBudget(
            deadline_ms=_effective_deadline_ms(plan),
            max_llm_calls=max(1, len(plan.routes) + 1),
            max_tokens=plan.token_budget,
            max_cost=plan.cost_budget,
            max_parallel_routes=min(3, max(1, len(plan.routes))),
        )
        ledger = BudgetLedger(budget)
        semaphore = asyncio.Semaphore(budget.max_parallel_routes)
        tasks = [
            asyncio.create_task(
                self._execute_route(route, ledger, semaphore, cancellation),
                name=f"query-route:{route.route_id}",
            )
            for route in plan.routes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[RouteOutcome] = []
        required_failure: RequiredRouteError | None = None
        for route, result in zip(plan.routes, results, strict=True):
            if isinstance(result, RequiredRouteError):
                required_failure = required_failure or result
                outcomes.append(
                    RouteOutcome(
                        route_id=route.route_id,
                        route_type=route.route_type.value,
                        required=True,
                        status=RouteExecutionStatus.FAILED,
                        latency_ms=ledger.elapsed_ms,
                        error_code=result.error_code,
                    )
                )
            elif isinstance(result, BaseException):
                if route.required:
                    required_failure = required_failure or RequiredRouteError(
                        route.route_id, "ROUTE_EXECUTION_FAILED"
                    )
                    outcomes.append(
                        RouteOutcome(
                            route_id=route.route_id,
                            route_type=route.route_type.value,
                            required=True,
                            status=RouteExecutionStatus.FAILED,
                            latency_ms=ledger.elapsed_ms,
                            error_code="ROUTE_EXECUTION_FAILED",
                        )
                    )
                else:
                    outcomes.append(
                        RouteOutcome(
                            route_id=route.route_id,
                            route_type=route.route_type.value,
                            required=False,
                            status=RouteExecutionStatus.FAILED,
                            latency_ms=ledger.elapsed_ms,
                            error_code="OPTIONAL_ROUTE_FAILED",
                            degradation_reason=DegradationReason.OPTIONAL_ROUTE_FAILED,
                        )
                    )
            else:
                outcomes.append(result)

        if required_failure is not None:
            cancellation.set()
            raise required_failure

        evidence_count = sum(item.candidate_count for item in outcomes)
        if evidence_count == 0:
            reasons = stable_degradation_reasons(outcomes)
            if DegradationReason.NO_EVIDENCE not in reasons:
                reasons = (*reasons, DegradationReason.NO_EVIDENCE)
            disposition = "abstain"
        else:
            reasons = stable_degradation_reasons(outcomes)
            disposition = "partial" if reasons else "answer"
        return QueryExecutionSummary(
            policy_version=plan.policy.policy_version,
            planner_version=plan.planner_version,
            outcomes=tuple(outcomes),
            degraded=bool(reasons),
            degradation_reasons=reasons,
            total_latency_ms=ledger.elapsed_ms,
            usage=ledger.snapshot(),
            evidence_count=evidence_count,
            answer_disposition=disposition,
        )

    async def _execute_route(
        self,
        route: QueryRoute,
        ledger: BudgetLedger,
        semaphore: asyncio.Semaphore,
        cancellation: Event,
    ) -> RouteOutcome:
        started = monotonic()
        if cancellation.is_set():
            return self._non_success(
                route,
                RouteExecutionStatus.CANCELLED,
                started,
                "QUERY_CANCELLED",
                DegradationReason.BUDGET_EXHAUSTED,
            )
        allowed, reason = ledger.can_start(max_cost=route.max_cost)
        if not allowed:
            if route.required:
                raise RequiredRouteError(route.route_id, reason.value.upper())
            return self._non_success(
                route,
                RouteExecutionStatus.SKIPPED,
                started,
                reason.value.upper(),
                reason,
            )
        breaker = self.circuit_breakers.setdefault(route.route_type.value, CircuitBreaker())
        if not breaker.allow():
            if route.required:
                raise RequiredRouteError(route.route_id, "CIRCUIT_OPEN")
            return self._non_success(
                route,
                RouteExecutionStatus.SKIPPED,
                started,
                "CIRCUIT_OPEN",
                DegradationReason.CIRCUIT_OPEN,
            )
        handler = self.handlers.get(route.route_type.value)
        if handler is None:
            if route.required:
                raise RequiredRouteError(route.route_id, "ROUTE_HANDLER_MISSING")
            return self._non_success(
                route,
                RouteExecutionStatus.SKIPPED,
                started,
                "ROUTE_HANDLER_MISSING",
                DegradationReason.OPTIONAL_ROUTE_FAILED,
            )
        route_cancellation = _LinkedRouteCancellation(cancellation)
        timeout = min(route.timeout_ms, max(1.0, ledger.remaining_ms)) / 1000.0
        try:
            async with semaphore:
                if iscoroutinefunction(handler):
                    invocation = await asyncio.wait_for(
                        handler(route, route_cancellation),
                        timeout=timeout,
                    )
                else:
                    invocation = await asyncio.wait_for(
                        asyncio.to_thread(handler, route, route_cancellation),
                        timeout=timeout,
                    )
            if not isinstance(invocation, RouteInvocation):
                raise TypeError("route handler must return RouteInvocation")
            ledger.record(invocation.usage)
            breaker.success()
            return RouteOutcome(
                route_id=route.route_id,
                route_type=route.route_type.value,
                required=route.required,
                status=RouteExecutionStatus.SUCCEEDED,
                latency_ms=(monotonic() - started) * 1000.0,
                candidate_count=invocation.candidate_count,
                usage=invocation.usage,
                payload=invocation.payload,
            )
        except TimeoutError as exc:
            breaker.failure()
            # 每条路线都有自己的取消信号。可选路线超时只停止自身的迟到线程，
            # 不得误伤仍可能成功的必选本地路线。
            route_cancellation.set()
            if route.required:
                # ``asyncio.wait_for`` 只能取消等待同步线程的 Future，无法强杀已经
                # 进入 embedding / Milvus SDK 的线程。必选路线超时意味着整个计划
                # 已经失败，立即广播协作取消，让隔离线程在下一个依赖边界停止，
                # 避免 QueryRun 终态后继续占用 CPU 或向 Milvus 提交迟到查询。
                cancellation.set()
                raise RequiredRouteError(route.route_id, "ROUTE_TIMEOUT") from exc
            return self._non_success(
                route,
                RouteExecutionStatus.TIMED_OUT,
                started,
                "OPTIONAL_ROUTE_TIMEOUT",
                DegradationReason.OPTIONAL_ROUTE_TIMEOUT,
            )
        except RouteInvocationError as exc:
            breaker.failure()
            if route.required:
                cancellation.set()
                raise RequiredRouteError(route.route_id, exc.error_code) from exc
            return self._non_success(
                route,
                RouteExecutionStatus.FAILED,
                started,
                exc.error_code,
                exc.degradation_reason,
            )
        except RequiredRouteError:
            raise
        except Exception as exc:
            breaker.failure()
            if route.required:
                cancellation.set()
                raise RequiredRouteError(route.route_id, "ROUTE_EXECUTION_FAILED") from exc
            return self._non_success(
                route,
                RouteExecutionStatus.FAILED,
                started,
                "OPTIONAL_ROUTE_FAILED",
                DegradationReason.OPTIONAL_ROUTE_FAILED,
            )
        finally:
            # 覆盖 task 外部取消（CancelledError）等不进入 Exception 分支的退出。
            route_cancellation.set()

    @staticmethod
    def _non_success(
        route: QueryRoute,
        status: RouteExecutionStatus,
        started: float,
        error_code: str,
        reason: DegradationReason,
    ) -> RouteOutcome:
        return RouteOutcome(
            route_id=route.route_id,
            route_type=route.route_type.value,
            required=route.required,
            status=status,
            latency_ms=(monotonic() - started) * 1000.0,
            error_code=error_code,
            degradation_reason=reason,
        )


__all__ = [
    "CancellationSignal",
    "CircuitBreaker",
    "QueryPolicyExecutor",
    "RouteHandler",
    "RouteInvocation",
    "RouteInvocationError",
]
