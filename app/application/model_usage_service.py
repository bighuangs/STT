"""在线查询模型调用的统一 usage 采集。

每次真实 provider 调用都产生一条调用事实；provider 未返回 token 时仍记录
``calls=1``、``estimated=True``，避免观测面把“未知用量”误当成“没有调用”。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Any, TypeVar
from uuid import uuid4

from app.application.observability_service import (
    UsageRecord,
    metric_registry,
    usage_ledger,
)
from app.infra.config import settings
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.security.principal import current_principal


_T = TypeVar("_T")
_ANSWERING_SCENES = frozenset(
    {"answer_plan_selection", "answer_plan_semantic_review"}
)


def usage_numbers(value: object) -> tuple[int, int]:
    usage = getattr(value, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(value, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    try:
        return max(0, int(input_tokens or 0)), max(0, int(output_tokens or 0))
    except (TypeError, ValueError):
        return 0, 0


def _logical_call_id(state: dict[str, Any], scene: str) -> str | None:
    receipts = state.get("answer_prompt_receipts")
    if not isinstance(receipts, list):
        return None
    for item in reversed(receipts):
        if (
            isinstance(item, dict)
            and item.get("scene") == scene
            and isinstance(item.get("logical_call_id"), str)
        ):
            return item["logical_call_id"]
    return None


def _usage_id(state: dict[str, Any], scene: str) -> str:
    logical_call_id = _logical_call_id(state, scene)
    if logical_call_id is None:
        return f"use_{uuid4().hex}"
    payload = f"{state.get('query_run_id') or ''}:{logical_call_id}:{scene}"
    return f"use_{sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def _aggregate_public_usage(state: dict[str, Any]) -> dict[str, Any]:
    """聚合整个 QueryRun；answer retry 使用逐 attempt 保守账本，避免漏算。"""

    summaries = state.get("model_usage_summaries")
    summaries = summaries if isinstance(summaries, list) else []
    ledger = state.get("answer_model_ledger")
    ledger = ledger if isinstance(ledger, dict) else {}
    ledger_attempts = int(ledger.get("provider_attempt_count") or 0)
    input_tokens = 0
    output_tokens = 0
    unknown = False
    for item in summaries:
        if not isinstance(item, dict):
            unknown = True
            continue
        if ledger_attempts and item.get("scene") in _ANSWERING_SCENES:
            continue
        input_tokens += max(0, int(item.get("input_tokens") or 0))
        output_tokens += max(0, int(item.get("output_tokens") or 0))
        unknown = unknown or item.get("usage_status") != "available"
    if ledger_attempts:
        input_tokens += max(0, int(ledger.get("accounted_input_tokens") or 0))
        output_tokens += max(0, int(ledger.get("accounted_output_tokens") or 0))
        estimated = not bool(ledger.get("usage_metadata_complete", False))
        status = "estimated" if estimated or unknown else "available"
        reason = (
            "one or more provider attempts used frozen token upper bounds"
            if status == "estimated"
            else "token usage available; unit price is not configured"
        )
        return {
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": None,
            "currency": None,
            "reason": reason,
        }
    if summaries and not unknown:
        return {
            "status": "available",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": None,
            "currency": None,
            "reason": "token usage available; unit price is not configured",
        }
    return {
        "status": "unavailable",
        "reason": "one or more model calls lack complete token usage metadata",
    }


def _telemetry_failed(state: dict[str, Any], *, stage: str, exc: Exception) -> None:
    """持久化无正文错误摘要；telemetry 永不反向改变 provider 结果。"""

    errors = state.setdefault("usage_telemetry_errors", [])
    if isinstance(errors, list):
        errors.append(
            {
                "schema_version": "usage-telemetry-error.v1",
                "stage": stage,
                "error_type": type(exc).__name__,
            }
        )
def record_model_call(
    state: dict[str, Any] | None,
    responses: Sequence[object],
    *,
    scene: str,
    call_status: str = "succeeded",
    provider: str = "configured-llm",
    model_id: str | None = None,
) -> dict[str, Any]:
    """记录一次调用；返回适合 AnswerEnvelope 使用的安全 usage 摘要。"""

    resolved_model_id = model_id or settings.llm_default_model or "unknown"
    observed = [usage_numbers(item) for item in responses]
    input_tokens = max((item[0] for item in observed), default=0)
    output_tokens = max((item[1] for item in observed), default=0)
    available = bool(input_tokens or output_tokens)
    snapshot = state if isinstance(state, dict) else {}
    try:
        metric_registry.increment(
            "model_calls_total",
            labels={"scene": scene, "status": call_status},
        )
        if available:
            metric_registry.observe(
                "model_tokens",
                input_tokens + output_tokens,
                labels={"scene": scene, "direction": "total"},
            )
    except Exception as exc:
        _telemetry_failed(snapshot, stage="metrics", exc=exc)

    if call_status != "succeeded":
        summary = {
            "status": "unavailable",
            "reason": "model call failed before complete usage metadata was available",
        }
        unit_price_version = "call-failed"
    elif not available:
        summary = {
            "status": "unavailable",
            "reason": "provider did not return token usage",
        }
        unit_price_version = "provider-unavailable"
    else:
        summary = {
            "status": "available",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": None,
            "currency": None,
            "reason": "token usage available; unit price is not configured",
        }
        unit_price_version = "unconfigured"

    summaries = snapshot.setdefault("model_usage_summaries", [])
    if isinstance(summaries, list):
        summaries.append(
            {
                "scene": scene,
                "call_status": call_status,
                "usage_status": summary["status"],
                "model_id": resolved_model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    aggregate = _aggregate_public_usage(snapshot)
    try:
        principal = current_principal()
        query_run_id = str(snapshot.get("query_run_id") or "")
        if principal is None or not query_run_id.startswith("qry_"):
            return aggregate
        record = UsageRecord(
            usage_id=_usage_id(snapshot, scene),
            tenant_id=principal.tenant_id,
            kb_id=str(snapshot.get("kb_id") or "") or None,
            query_run_id=query_run_id,
            provider=provider,
            model_id=resolved_model_id,
            scene=scene,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unit_price_version=unit_price_version,
            cost=0.0,
            estimated=True,
        )
        usage_ledger.append(record)
        if snapshot.get("runtime_managed_history"):
            with session_scope() as session:
                QueryReliabilityRepository(
                    session,
                    tenant_id=principal.tenant_id,
                ).append_usage(record)
    except Exception as exc:
        _telemetry_failed(snapshot, stage="usage_ledger", exc=exc)
    return aggregate


def invoke_model_call(
    state: dict[str, Any] | None,
    *,
    scene: str,
    invoke: Callable[[], _T],
) -> _T:
    """执行一次非流式 provider 调用并确保成功/失败都产生调用事实。"""

    try:
        response = invoke()
    except Exception:
        record_model_call(state, (), scene=scene, call_status="failed")
        raise
    record_model_call(state, (response,), scene=scene)
    return response


__all__ = [
    "invoke_model_call",
    "record_model_call",
    "usage_numbers",
]
