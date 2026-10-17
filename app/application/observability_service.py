"""TODO-14 Trace、Metric、Audit 与 Usage 的最小可信底座。

正文、Prompt、向量、凭据和原始问题默认不会进入观测记录。此实现提供进程内
collector 及稳定领域合同；SQL append-only repository 由 P3 持久化层承接。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


_trace_context: ContextVar[tuple[str, str] | None] = ContextVar("stt_trace_context", default=None)
_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "prompt",
    "content",
    "query",
    "question",
    "vector",
    "embedding",
}
_HIGH_CARDINALITY_LABELS = {
    "request_id",
    "trace_id",
    "span_id",
    "query_run_id",
    "session_id",
    "conversation_id",
    "document_id",
    "chunk_id",
    "user_id",
}


def safe_hash(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def redact(value: Any, *, key: str = "") -> Any:
    """递归生成 JSON-safe 脱敏副本，不修改调用方对象。"""

    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_KEYS):
        if value in (None, "", (), [], {}):
            return value
        raw = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        )
        return {"redacted": True, "hash": safe_hash(raw), "length": len(raw)}
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, key=key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    status: str
    attributes: dict[str, Any]
    error_code: str | None = None


@dataclass
class _HistogramAggregate:
    """常量内存直方图摘要；长期进程不会为每次观测保留一个样本。"""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)


class TraceRecorder:
    def __init__(self, *, max_spans: int = 10_000) -> None:
        self.max_spans = max_spans
        self._records: list[SpanRecord] = []
        self._lock = RLock()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[str, str]]:
        parent = _trace_context.get()
        selected_trace_id = trace_id or (parent[0] if parent else f"trc_{uuid4().hex}")
        span_id = f"spn_{uuid4().hex[:16]}"
        token = _trace_context.set((selected_trace_id, span_id))
        started_at = datetime.now(UTC)
        started = monotonic()
        status = "ok"
        error_code: str | None = None
        try:
            yield selected_trace_id, span_id
        except Exception as exc:
            status = "error"
            error_code = getattr(exc, "error_code", None) or type(exc).__name__.upper()
            raise
        finally:
            ended_at = datetime.now(UTC)
            record = SpanRecord(
                trace_id=selected_trace_id,
                span_id=span_id,
                parent_span_id=parent[1] if parent else None,
                name=name,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=(monotonic() - started) * 1000.0,
                status=status,
                attributes=redact(dict(attributes or {})),
                error_code=error_code,
            )
            with self._lock:
                self._records.append(record)
                if len(self._records) > self.max_spans:
                    del self._records[: len(self._records) - self.max_spans]
            _trace_context.reset(token)

    def records(self, *, trace_id: str | None = None) -> tuple[SpanRecord, ...]:
        with self._lock:
            values = tuple(self._records)
        if trace_id is None:
            return values
        return tuple(item for item in values if item.trace_id == trace_id)


class MetricRegistry:
    """低基数 counter/histogram；高基数标签在写入时直接拒绝。"""

    def __init__(self, *, max_series: int = 5_000) -> None:
        if max_series < 1:
            raise ValueError("max_series must be positive")
        self.max_series = max_series
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            _HistogramAggregate,
        ] = {}
        self._lock = RLock()

    def _reserve_series(
        self,
        key: tuple[str, tuple[tuple[str, str], ...]],
        target: Mapping,
    ) -> None:
        if key not in target and len(self._counters) + len(self._histograms) >= self.max_series:
            raise ValueError("metric series limit exceeded")

    @staticmethod
    def _labels(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
        labels = labels or {}
        forbidden = set(labels) & _HIGH_CARDINALITY_LABELS
        if forbidden:
            raise ValueError(f"high-cardinality metric labels are forbidden: {sorted(forbidden)}")
        if len(labels) > 8:
            raise ValueError("metric labels cannot exceed 8 dimensions")
        normalized = tuple(sorted((str(key), str(value)) for key, value in labels.items()))
        if any(len(value) > 64 for _, value in normalized):
            raise ValueError("metric label value is too long")
        return normalized

    def increment(
        self, name: str, value: float = 1.0, *, labels: Mapping[str, str] | None = None
    ) -> None:
        if value < 0:
            raise ValueError("counter increment cannot be negative")
        key = (name, self._labels(labels))
        with self._lock:
            self._reserve_series(key, self._counters)
            self._counters[key] += value

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        key = (name, self._labels(labels))
        with self._lock:
            self._reserve_series(key, self._histograms)
            aggregate = self._histograms.setdefault(key, _HistogramAggregate())
            aggregate.observe(float(value))

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._counters.items())
            ]
            histograms = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "count": aggregate.count,
                    "min": aggregate.minimum,
                    "max": aggregate.maximum,
                    "avg": aggregate.total / aggregate.count,
                }
                for (name, labels), aggregate in sorted(self._histograms.items())
                if aggregate.count
            ]
        return {"counters": counters, "histograms": histograms}


class UsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    usage_id: str = Field(pattern=r"^use_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str | None = Field(default=None, max_length=64)
    query_run_id: str | None = Field(default=None, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=255)
    scene: str = Field(min_length=1, max_length=64)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    calls: int = Field(default=1, ge=1)
    currency: str = Field(default="CNY", min_length=3, max_length=3)
    unit_price_version: str = Field(min_length=1, max_length=64)
    cost: float = Field(default=0.0, ge=0.0)
    estimated: bool
    fallback_sequence: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UsageLedger:
    def __init__(self, *, max_records: int = 10_000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self._records: list[UsageRecord] = []
        self._lock = RLock()

    def append(self, record: UsageRecord) -> None:
        with self._lock:
            if any(item.usage_id == record.usage_id for item in self._records):
                return
            self._records.append(record)
            if len(self._records) > self.max_records:
                del self._records[: len(self._records) - self.max_records]

    def aggregate(self, tenant_id: str, *, query_run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            rows = [
                item
                for item in self._records
                if item.tenant_id == tenant_id
                and (query_run_id is None or item.query_run_id == query_run_id)
            ]
        return {
            "tenant_id": tenant_id,
            "query_run_id": query_run_id,
            "calls": sum(item.calls for item in rows),
            "input_tokens": sum(item.input_tokens for item in rows),
            "output_tokens": sum(item.output_tokens for item in rows),
            "cache_read_tokens": sum(item.cache_read_tokens for item in rows),
            "cost": round(sum(item.cost for item in rows), 8),
            "estimated": any(item.estimated for item in rows),
            "record_count": len(rows),
        }


@dataclass(frozen=True)
class SafeAuditEvent:
    event_id: str
    tenant_id: str | None
    actor_id: str | None
    action: str
    resource_type: str | None
    resource_id_hash: str | None
    result: str
    request_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str | None,
        actor_id: str | None,
        action: str,
        resource_type: str | None,
        resource_id: str | None,
        result: str,
        request_id: str | None,
        detail: Mapping[str, Any] | None = None,
    ) -> "SafeAuditEvent":
        return cls(
            event_id=f"aud_{uuid4().hex}",
            tenant_id=tenant_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id_hash=safe_hash(resource_id) if resource_id else None,
            result=result,
            request_id=request_id,
            detail=redact(dict(detail or {})),
        )


trace_recorder = TraceRecorder()
metric_registry = MetricRegistry()
usage_ledger = UsageLedger()


__all__ = [
    "MetricRegistry",
    "SafeAuditEvent",
    "SpanRecord",
    "TraceRecorder",
    "UsageLedger",
    "UsageRecord",
    "metric_registry",
    "redact",
    "safe_hash",
    "trace_recorder",
    "usage_ledger",
]
