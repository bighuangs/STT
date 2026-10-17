"""Query 侧同步模型调用的受限瞬时重试基础设施边界。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Any, TypeVar
from uuid import uuid4

import httpx
import openai

from app.application.model_ports import QueryModelCallTransportUnavailable
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger


_T = TypeVar("_T")
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_AUDIT_SCHEMA_VERSION = "query-model-call-retry-audit.v1"


class QueryModelCallDeadlineExceeded(TimeoutError):
    """逻辑模型调用在开始前或完成后越过了冻结截止时间。"""

    code = "QUERY_MODEL_CALL_DEADLINE_EXCEEDED"

    def __init__(self) -> None:
        super().__init__("query model logical call deadline exceeded")


@dataclass(frozen=True, slots=True)
class QueryModelRetryPolicy:
    """一次逻辑调用内的 transport 重试预算。"""

    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    total_timeout_seconds: float

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds cannot exceed max_delay_seconds")
        if self.total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be positive")


def configured_query_model_retry_policy() -> QueryModelRetryPolicy:
    """从冻结配置构造小次数、有限总时长的默认策略。"""

    max_attempts = min(3, max(1, int(settings.llm_max_attempts)))
    base_delay = max(0.0, float(settings.llm_retry_base_delay))
    max_delay = max(base_delay, float(settings.llm_retry_max_delay))
    retry_wait = sum(min(max_delay, base_delay * (2**offset)) for offset in range(max_attempts - 1))
    transport_upper_bound = (
        max(0.001, float(settings.llm_request_timeout)) * max_attempts + retry_wait
    )
    query_upper_bound = max(
        0.001,
        float(settings.query_default_deadline_ms) / 1000.0,
    )
    return QueryModelRetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        total_timeout_seconds=min(transport_upper_bound, query_upper_bound),
    )


def _runtime_remaining_seconds(
    state: dict[str, Any] | None,
    *,
    utc_now: Callable[[], datetime],
) -> float | None:
    if not isinstance(state, dict):
        return None
    raw_deadline = state.get("deadline_at")
    if raw_deadline in {None, ""}:
        return None
    if isinstance(raw_deadline, datetime):
        deadline = raw_deadline
    else:
        deadline = datetime.fromisoformat(str(raw_deadline).replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline_at must include timezone")
    return (deadline.astimezone(UTC) - utc_now().astimezone(UTC)).total_seconds()


def _retry_class(exc: Exception) -> str | None:
    """只识别幂等调用中明确允许重试的 transport/provider 瞬时错误。"""

    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(
        exc,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ),
    ):
        return "timeout"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.APIStatusError):
        status_code = int(getattr(exc, "status_code", 0) or 0)
        if status_code in _RETRYABLE_STATUS_CODES:
            return "rate_limit" if status_code == 429 else "server_error"
        return None
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ProxyError)):
        return "connection"
    if isinstance(exc, httpx.NetworkError):
        return "connection"
    return None


def _append_state_audit(
    state: dict[str, Any] | None,
    audit: dict[str, Any],
) -> None:
    if not isinstance(state, dict):
        return
    existing = state.get("query_model_call_audits")
    if existing is None:
        existing = []
        state["query_model_call_audits"] = existing
    if isinstance(existing, list):
        existing.append(audit)


def invoke_query_model_call(
    state: dict[str, Any] | None,
    *,
    scene: str,
    model_id: str,
    invoke: Callable[[float], _T],
    retryable_error_types: tuple[type[Exception], ...] = (),
    audit: dict[str, Any] | None = None,
    policy: QueryModelRetryPolicy | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    utc_now: Callable[[], datetime] | None = None,
) -> _T:
    """执行一次逻辑模型调用，并在同一边界内聚合有界 attempts。

    默认只重试瞬时 transport/provider 错误；调用方可显式提供不含正文的结构化
    响应错误类型。审计不会记录 prompt、response、异常消息或 provider body。
    SDK 自身必须保持 ``max_retries=0``，避免双重重试。
    """

    retry_policy = policy or configured_query_model_retry_policy()
    clock = monotonic or time.monotonic
    sleeper = sleep or time.sleep
    wall_clock = utc_now or (lambda: datetime.now(UTC))
    started_at = clock()
    configured_timeout = retry_policy.total_timeout_seconds
    runtime_remaining = _runtime_remaining_seconds(state, utc_now=wall_clock)
    if runtime_remaining is None:
        timeout_seconds = configured_timeout
        deadline_source = "configured_logical_call"
    else:
        timeout_seconds = min(configured_timeout, runtime_remaining)
        deadline_source = "query_run"
    logical_deadline = started_at + max(0.0, timeout_seconds)

    target_audit = audit if isinstance(audit, dict) else {}
    target_audit.update(
        {
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "logical_call_id": str(target_audit.get("logical_call_id") or f"qmc_{uuid4().hex}"),
            "scene": scene,
            "model_id": model_id or "unknown",
            "status": "running",
            "max_attempts": retry_policy.max_attempts,
            "provider_attempt_count": 0,
            "retry_count": 0,
            "deadline_source": deadline_source,
            "logical_timeout_ms": max(0, int(timeout_seconds * 1000)),
            "total_elapsed_ms": 0,
            "attempts": [],
        }
    )
    _append_state_audit(state, target_audit)

    def finalize(status: str) -> None:
        target_audit["status"] = status
        target_audit["provider_attempt_count"] = len(target_audit["attempts"])
        target_audit["retry_count"] = max(
            0,
            len(target_audit["attempts"]) - 1,
        )
        target_audit["total_elapsed_ms"] = max(
            0,
            int((clock() - started_at) * 1000),
        )

    if timeout_seconds <= 0:
        finalize("deadline_exhausted")
        raise QueryModelCallDeadlineExceeded()

    for attempt_number in range(1, retry_policy.max_attempts + 1):
        if clock() >= logical_deadline:
            finalize("deadline_exhausted")
            raise QueryModelCallDeadlineExceeded()
        remaining_before_attempt = logical_deadline - clock()
        if remaining_before_attempt <= 0:
            finalize("deadline_exhausted")
            raise QueryModelCallDeadlineExceeded()
        attempt_started = clock()
        attempt_audit: dict[str, Any] = {
            "attempt": attempt_number,
            "status": "running",
            "elapsed_ms": 0,
            "timeout_ms": max(1, int(remaining_before_attempt * 1000)),
        }
        target_audit["attempts"].append(attempt_audit)
        try:
            response = invoke(max(0.001, remaining_before_attempt))
        except Exception as exc:
            retry_class = _retry_class(exc)
            if (
                retry_class is None
                and retryable_error_types
                and isinstance(exc, retryable_error_types)
            ):
                retry_class = "invalid_response"
            attempt_audit.update(
                {
                    "status": "retryable_failure" if retry_class else "failed",
                    "error_type": type(exc).__name__,
                    "retryable": retry_class is not None,
                    "retry_class": retry_class or "not_retryable",
                    "elapsed_ms": max(0, int((clock() - attempt_started) * 1000)),
                }
            )
            if retry_class is None:
                finalize("failed_non_retryable")
                raise
            if clock() >= logical_deadline:
                attempt_audit["status"] = "deadline_exceeded"
                finalize("deadline_exhausted")
                raise QueryModelCallDeadlineExceeded() from exc
            if attempt_number >= retry_policy.max_attempts:
                finalize("failed_attempt_limit")
                if retry_class in {"connection", "timeout"}:
                    raise QueryModelCallTransportUnavailable() from exc
                raise
            delay = min(
                retry_policy.max_delay_seconds,
                retry_policy.base_delay_seconds * (2 ** (attempt_number - 1)),
            )
            remaining = logical_deadline - clock()
            if remaining <= delay:
                finalize("deadline_exhausted")
                raise QueryModelCallDeadlineExceeded() from exc
            attempt_audit["backoff_ms"] = max(0, int(delay * 1000))
            logger.warning(
                "query model retry scheduled: scene={}, attempt={}/{}, "
                "error_type={}, retry_class={}, backoff_ms={}",
                scene,
                attempt_number,
                retry_policy.max_attempts,
                type(exc).__name__,
                retry_class,
                attempt_audit["backoff_ms"],
            )
            sleeper(delay)
            continue
        attempt_audit.update(
            {
                "status": "succeeded",
                "elapsed_ms": max(0, int((clock() - attempt_started) * 1000)),
            }
        )
        if clock() > logical_deadline:
            attempt_audit["status"] = "deadline_exceeded"
            finalize("deadline_exhausted")
            raise QueryModelCallDeadlineExceeded()
        finalize("succeeded")
        return response

    raise AssertionError("query model retry loop terminated unexpectedly")


__all__ = [
    "QueryModelCallDeadlineExceeded",
    "QueryModelCallTransportUnavailable",
    "QueryModelRetryPolicy",
    "configured_query_model_retry_policy",
    "invoke_query_model_call",
]
