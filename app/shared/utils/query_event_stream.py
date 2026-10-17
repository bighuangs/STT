"""TODO-31 基于持久 QueryEvent 的 SSE 投影。

该适配器不保存队列、不拥有 QueryRun，也不会在浏览器断开时自动取消运行。
每次轮询都从持久 EventStore 的有界窗口读取，因此跨 worker 和断线续传共享同一
事实源；慢消费者只会被明确 reset，不会使生产者内存无限增长。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
import json
from typing import Any

from app.domain.query_runtime import EventPage, QueryEvent
from app.infra.persistence.sql.query_runtime_repository import (
    EventCursorExpired,
    EventCursorInvalid,
)


EventPageReader = Callable[
    [str, int, str | None, int],
    EventPage | Awaitable[EventPage],
]
DisconnectProbe = Callable[[], bool | Awaitable[bool]]


@dataclass(frozen=True)
class DurableStreamPolicy:
    batch_size: int = 100
    max_lag_events: int = 1000
    max_batch_bytes: int = 1_048_576
    poll_interval_seconds: float = 0.25
    heartbeat_interval_seconds: float = 15.0

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        if self.max_lag_events < self.batch_size:
            raise ValueError("max_lag_events must be at least batch_size")
        if self.max_batch_bytes < 1024:
            raise ValueError("max_batch_bytes is too small")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")


def encode_sse(
    event: str,
    data: dict[str, Any],
    *,
    event_id: str | None = None,
    retry_ms: int | None = None,
) -> str:
    """生成规范 SSE frame；JSON 被固定在单个 data 行，便于代理转发。"""

    safe_event = event.replace("\r", "").replace("\n", "")
    fields: list[str] = []
    if event_id is not None:
        fields.append(f"id: {event_id.replace(chr(13), '').replace(chr(10), '')}")
    if retry_ms is not None:
        fields.append(f"retry: {max(0, retry_ms)}")
    fields.append(f"event: {safe_event}")
    fields.append(
        "data: "
        + json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )
    return "\n".join(fields) + "\n\n"


def encode_query_event(event: QueryEvent) -> str:
    return encode_sse(
        event.type.value,
        event.model_dump(mode="json"),
        event_id=event.event_id,
    )


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


async def durable_sse_generator(
    *,
    query_run_id: str,
    read_page: EventPageReader,
    last_event_id: str | None = None,
    initial_after_seq: int = 0,
    policy: DurableStreamPolicy | None = None,
    is_disconnected: DisconnectProbe | None = None,
) -> AsyncGenerator[str, None]:
    """将 EventStore 投影为支持 Last-Event-ID 的有界 SSE。

    ``read_page`` 必须在内部使用 tenant/user 作用域仓储或等价授权查询。生成器
    默认仅断开订阅，不取消运行；显式取消应调用 QueryRun DELETE/cancel API。
    """

    if initial_after_seq < 0:
        raise ValueError("initial_after_seq must be non-negative")
    stream_policy = policy or DurableStreamPolicy()
    # 标准浏览器自动重连用 Last-Event-ID；手工新建 EventSource 无法设置 header，
    # 因此允许受校验的 after_seq query cursor。两者并存时事件 ID 是权威游标。
    after_seq = 0 if last_event_id else initial_after_seq
    resume_id = last_event_id
    last_emit_monotonic = asyncio.get_running_loop().time()
    yield encode_sse(
        "stream.ready",
        {
            "schema_version": "query-event.v1",
            "query_run_id": query_run_id,
            "resume_from": resume_id,
            "resume_after_seq": after_seq,
        },
        retry_ms=max(100, int(stream_policy.poll_interval_seconds * 1000)),
    )

    while True:
        if is_disconnected is not None and await _resolve(is_disconnected()):
            return
        try:
            page = await _resolve(
                read_page(
                    query_run_id,
                    after_seq,
                    resume_id,
                    stream_policy.batch_size,
                )
            )
        except EventCursorExpired:
            yield encode_sse(
                "stream.reset",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "code": "EVENT_CURSOR_EXPIRED",
                    "message": "事件续传窗口已过期，请读取 QueryRun 快照后重新订阅。",
                    "last_delivered_seq": after_seq,
                },
            )
            return
        except EventCursorInvalid:
            yield encode_sse(
                "stream.reset",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "code": "EVENT_CURSOR_INVALID",
                    "message": "Last-Event-ID 不属于当前 QueryRun。",
                    "last_delivered_seq": after_seq,
                },
            )
            return
        resume_id = None
        if page.query_run_id != query_run_id:
            yield encode_sse(
                "stream.reset",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "code": "EVENT_SCOPE_MISMATCH",
                    "message": "事件读取结果与订阅运行不一致。",
                    "last_delivered_seq": after_seq,
                },
            )
            return
        if page.lag_events > stream_policy.max_lag_events:
            yield encode_sse(
                "stream.reset",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "code": "SLOW_CONSUMER",
                    "message": "消费者落后事件数超过上限，请使用最新运行快照重连。",
                    "last_delivered_seq": after_seq,
                    "first_available_seq": page.first_available_seq,
                    "last_available_seq": page.last_available_seq,
                },
            )
            return
        batch_bytes = sum(event.size_bytes for event in page.events)
        if batch_bytes > stream_policy.max_batch_bytes:
            yield encode_sse(
                "stream.reset",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "code": "BATCH_BYTES_EXCEEDED",
                    "message": "待发送事件批次超过连接字节预算，请缩小读取窗口后重连。",
                    "last_delivered_seq": after_seq,
                },
            )
            return
        for event in page.events:
            if event.seq <= after_seq:
                # EventStore 的重复读取不会造成浏览器重复展示。
                continue
            yield encode_query_event(event)
            after_seq = event.seq
            last_emit_monotonic = asyncio.get_running_loop().time()
        if page.terminal:
            return
        monotonic = asyncio.get_running_loop().time()
        if monotonic - last_emit_monotonic >= stream_policy.heartbeat_interval_seconds:
            # heartbeat 不持久化、无 event id，因此不会改变续传游标。
            yield encode_sse(
                "heartbeat",
                {
                    "schema_version": "query-event.v1",
                    "query_run_id": query_run_id,
                    "at": datetime.now(UTC).isoformat(),
                    "last_delivered_seq": after_seq,
                },
            )
            last_emit_monotonic = monotonic
        await asyncio.sleep(stream_policy.poll_interval_seconds)


__all__ = [
    "DurableStreamPolicy",
    "durable_sse_generator",
    "encode_query_event",
    "encode_sse",
]
