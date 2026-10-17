"""P3 查询预算、路由执行和故障降级领域契约。

本模块不调用网络、模型、数据库或 LangGraph。它把 TODO-10 中原先隐含在节点里的
deadline、预算、必选/可选分支和拒答下限固化为可验证合同。执行器只能消费已经由
TODO-27 生成并校验的 QueryPlan，不能自行开启 Web/HyDE 或扩大作用域。
"""

from __future__ import annotations

from enum import StrEnum
from time import monotonic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RouteExecutionStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class DegradationReason(StrEnum):
    OPTIONAL_ROUTE_FAILED = "optional_route_failed"
    OPTIONAL_ROUTE_TIMEOUT = "optional_route_timeout"
    CIRCUIT_OPEN = "circuit_open"
    BULKHEAD_FULL = "bulkhead_full"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RATE_LIMITED = "rate_limited"
    SECURITY_POLICY_BLOCKED = "security_policy_blocked"
    NO_EVIDENCE = "no_evidence"
    RERANK_FALLBACK = "rerank_fallback"


class QueryBudget(BaseModel):
    """一次 QueryRun 的硬预算；所有值都来自服务端发布策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deadline_ms: int = Field(default=8_000, ge=100, le=120_000)
    max_llm_calls: int = Field(default=3, ge=0, le=64)
    max_tokens: int = Field(default=4_096, ge=0, le=1_000_000)
    max_cost: float = Field(default=1.0, ge=0.0, le=100_000.0)
    max_parallel_routes: int = Field(default=3, ge=1, le=16)


class RouteUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RouteOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(min_length=1, max_length=64)
    route_type: str = Field(min_length=1, max_length=64)
    required: bool
    status: RouteExecutionStatus
    latency_ms: float = Field(ge=0.0)
    candidate_count: int = Field(default=0, ge=0)
    usage: RouteUsage = Field(default_factory=RouteUsage)
    error_code: str | None = Field(default=None, max_length=128)
    degradation_reason: DegradationReason | None = None
    payload: Any = None

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> "RouteOutcome":
        if self.status == RouteExecutionStatus.SUCCEEDED and self.error_code:
            raise ValueError("successful route cannot carry error_code")
        if self.status in {
            RouteExecutionStatus.FAILED,
            RouteExecutionStatus.TIMED_OUT,
            RouteExecutionStatus.SKIPPED,
            RouteExecutionStatus.CANCELLED,
        } and not (self.error_code or self.degradation_reason):
            raise ValueError("non-success route requires a stable reason")
        if self.required and self.degradation_reason in {
            DegradationReason.OPTIONAL_ROUTE_FAILED,
            DegradationReason.OPTIONAL_ROUTE_TIMEOUT,
        }:
            raise ValueError("required route cannot use optional degradation reason")
        return self


class QueryExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(min_length=1, max_length=128)
    planner_version: str = Field(min_length=1, max_length=128)
    outcomes: tuple[RouteOutcome, ...]
    degraded: bool
    degradation_reasons: tuple[DegradationReason, ...] = ()
    total_latency_ms: float = Field(ge=0.0)
    usage: RouteUsage
    evidence_count: int = Field(default=0, ge=0)
    answer_disposition: str = Field(default="answer", pattern=r"^(answer|partial|clarify|abstain)$")

    @model_validator(mode="after")
    def validate_summary(self) -> "QueryExecutionSummary":
        if self.degraded != bool(self.degradation_reasons):
            raise ValueError("degraded flag must match degradation_reasons")
        if self.answer_disposition == "answer" and self.evidence_count == 0:
            raise ValueError("answer disposition requires evidence")
        if any(
            item.required and item.status != RouteExecutionStatus.SUCCEEDED
            for item in self.outcomes
        ):
            raise ValueError("summary cannot hide a failed required route")
        return self


class BudgetExceededError(RuntimeError):
    """继续执行会突破服务端预算。"""


class RequiredRouteError(RuntimeError):
    """必选路线失败，禁止用可选 Web 等路线伪装为成功。"""

    def __init__(self, route_id: str, error_code: str) -> None:
        super().__init__(f"required route failed: {route_id} ({error_code})")
        self.route_id = route_id
        self.error_code = error_code


class BudgetLedger:
    """线程安全由上层保证的单运行预算账本。

    先用 ``can_start`` 检查预留预算，结束后再用 ``record`` 记真实用量。账本永不
    静默截断；越界立即抛出稳定异常，便于执行器取消剩余分支。
    """

    def __init__(self, budget: QueryBudget) -> None:
        self.budget = budget
        self.started_at = monotonic()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.estimated = False

    @property
    def elapsed_ms(self) -> float:
        return (monotonic() - self.started_at) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.budget.deadline_ms - self.elapsed_ms)

    def can_start(
        self,
        *,
        max_cost: float = 0.0,
        llm_calls: int = 0,
        tokens: int = 0,
    ) -> tuple[bool, DegradationReason | None]:
        if self.remaining_ms <= 0:
            return False, DegradationReason.DEADLINE_EXCEEDED
        if (
            self.llm_calls + llm_calls > self.budget.max_llm_calls
            or self.input_tokens + self.output_tokens + tokens > self.budget.max_tokens
            or self.cost + max_cost > self.budget.max_cost + 1e-9
        ):
            return False, DegradationReason.BUDGET_EXHAUSTED
        return True, None

    def record(self, usage: RouteUsage) -> None:
        next_llm_calls = self.llm_calls + usage.llm_calls
        next_tokens = self.input_tokens + self.output_tokens + usage.total_tokens
        next_cost = self.cost + usage.cost
        if next_llm_calls > self.budget.max_llm_calls:
            raise BudgetExceededError("max_llm_calls exceeded")
        if next_tokens > self.budget.max_tokens:
            raise BudgetExceededError("max_tokens exceeded")
        if next_cost > self.budget.max_cost + 1e-9:
            raise BudgetExceededError("max_cost exceeded")
        self.llm_calls = next_llm_calls
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost = next_cost
        self.estimated = self.estimated or usage.estimated

    def snapshot(self) -> RouteUsage:
        return RouteUsage(
            llm_calls=self.llm_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=round(self.cost, 8),
            estimated=self.estimated,
        )


def stable_degradation_reasons(
    outcomes: tuple[RouteOutcome, ...] | list[RouteOutcome],
) -> tuple[DegradationReason, ...]:
    """按首次出现顺序返回去重原因，避免并发完成顺序污染响应。"""

    return tuple(
        dict.fromkeys(
            item.degradation_reason
            for item in outcomes
            if item.degradation_reason is not None
        )
    )


__all__ = [
    "BudgetExceededError",
    "BudgetLedger",
    "DegradationReason",
    "QueryBudget",
    "QueryExecutionSummary",
    "RequiredRouteError",
    "RouteExecutionStatus",
    "RouteOutcome",
    "RouteUsage",
    "stable_degradation_reasons",
]
