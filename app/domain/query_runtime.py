"""TODO-24/25/31 查询运行、会话一致性与可靠事件领域契约。

本模块刻意不依赖 FastAPI、LangGraph、SQLAlchemy 或具体消息队列。查询运行的
事实身份、状态转换、历史快照和事件协议因此可以被 HTTP、Worker、SSE 以及离线
恢复流程共同使用，而不会再次退化为以 ``session_id`` 索引进程内字典。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain._state_machine import InvalidStateTransition, ensure_transition


QUERY_RUNTIME_SCHEMA_VERSION = "query-runtime.v1"
QUERY_EVENT_SCHEMA_VERSION = "query-event.v1"
HISTORY_SNAPSHOT_SCHEMA_VERSION = "conversation-history.v1"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class TurnStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    READY_TO_COMMIT = "ready_to_commit"
    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class QueryRunStatus(StrEnum):
    ACCEPTED = "accepted"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    PACKING = "packing"
    GENERATING = "generating"
    VALIDATING = "validating"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


RUN_TERMINAL_STATUSES = frozenset(
    {
        QueryRunStatus.COMPLETED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
        QueryRunStatus.CANCELLED,
    }
)

TURN_TERMINAL_STATUSES = frozenset(
    {
        TurnStatus.COMMITTED,
        TurnStatus.FAILED,
        TurnStatus.TIMED_OUT,
        TurnStatus.CANCELLED,
    }
)

_RUN_TRANSITIONS = {
    QueryRunStatus.ACCEPTED: {
        QueryRunStatus.PLANNING,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.PLANNING: {
        QueryRunStatus.RETRIEVING,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.RETRIEVING: {
        QueryRunStatus.PACKING,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.PACKING: {
        QueryRunStatus.GENERATING,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.GENERATING: {
        QueryRunStatus.VALIDATING,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.VALIDATING: {
        QueryRunStatus.COMPLETED,
        QueryRunStatus.CANCEL_REQUESTED,
        QueryRunStatus.FAILED,
        QueryRunStatus.TIMED_OUT,
    },
    QueryRunStatus.CANCEL_REQUESTED: {QueryRunStatus.CANCELLED},
    QueryRunStatus.COMPLETED: set(),
    QueryRunStatus.FAILED: set(),
    QueryRunStatus.TIMED_OUT: set(),
    QueryRunStatus.CANCELLED: set(),
}

_TURN_TRANSITIONS = {
    TurnStatus.PENDING: {
        TurnStatus.RUNNING,
        TurnStatus.FAILED,
        TurnStatus.TIMED_OUT,
        TurnStatus.CANCELLED,
    },
    TurnStatus.RUNNING: {
        TurnStatus.READY_TO_COMMIT,
        TurnStatus.FAILED,
        TurnStatus.TIMED_OUT,
        TurnStatus.CANCELLED,
    },
    TurnStatus.READY_TO_COMMIT: {TurnStatus.COMMITTED},
    # 显式 retry 可复用原 Turn 身份，但不会复用 QueryRun 身份。
    TurnStatus.FAILED: {TurnStatus.RUNNING},
    TurnStatus.TIMED_OUT: {TurnStatus.RUNNING},
    TurnStatus.CANCELLED: {TurnStatus.RUNNING},
    TurnStatus.COMMITTED: set(),
}


def ensure_run_transition(current: QueryRunStatus, target: QueryRunStatus) -> None:
    """校验 QueryRun 状态机；终态不可逆，取消请求一旦 CAS 成功便优先。"""

    ensure_transition(current, target, _RUN_TRANSITIONS)


def ensure_turn_transition(current: TurnStatus, target: TurnStatus) -> None:
    """校验 Turn 状态机。"""

    ensure_transition(current, target, _TURN_TRANSITIONS)


def new_runtime_id(prefix: Literal["conv", "turn", "msg", "qry"]) -> str:
    """生成可读且不泄露租户/用户信息的随机身份。"""

    return f"{prefix}_{uuid4().hex}"


def canonical_hash(value: Any) -> str:
    """生成跨进程稳定 JSON 哈希，用于请求幂等和历史快照一致性。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-runtime.v1"] = QUERY_RUNTIME_SCHEMA_VERSION
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=64)
    title: str = Field(default="", max_length=512)
    status: ConversationStatus = ConversationStatus.ACTIVE
    revision: int = Field(default=0, ge=0)
    next_turn_seq: int = Field(default=1, ge=1)
    resolved_turn_seq: int = Field(default=0, ge=0)
    history_version: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archived_at: datetime | None = None

    @model_validator(mode="after")
    def validate_sequence_frontier(self) -> "Conversation":
        if self.resolved_turn_seq >= self.next_turn_seq:
            raise ValueError("resolved_turn_seq must be lower than next_turn_seq")
        if self.status == ConversationStatus.ARCHIVED and self.archived_at is None:
            raise ValueError("archived conversation requires archived_at")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-runtime.v1"] = QUERY_RUNTIME_SCHEMA_VERSION
    turn_id: str = Field(pattern=r"^turn_[0-9a-f]{32}$")
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    client_turn_id: str = Field(min_length=1, max_length=128)
    seq: int = Field(ge=1)
    base_history_seq: int = Field(ge=0)
    history_version: int = Field(ge=0)
    status: TurnStatus
    input_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    user_content: str = Field(min_length=1, max_length=1_000_000)
    active_query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    created_at: datetime
    committed_at: datetime | None = None
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_turn_timestamps(self) -> "Turn":
        if self.status == TurnStatus.COMMITTED and self.committed_at is None:
            raise ValueError("committed turn requires committed_at")
        if self.status in TURN_TERMINAL_STATUSES and self.resolved_at is None:
            raise ValueError("terminal turn requires resolved_at")
        return self


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-runtime.v1"] = QUERY_RUNTIME_SCHEMA_VERSION
    message_id: str = Field(pattern=r"^msg_[0-9a-f]{32}$")
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    turn_id: str = Field(pattern=r"^turn_[0-9a-f]{32}$")
    query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    role: Literal["user", "assistant", "system", "tool"]
    status: Literal["committed", "partial", "discarded"]
    content: str = Field(max_length=1_000_000)
    ordinal: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    committed_at: datetime | None = None


class QueryRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-runtime.v1"] = QUERY_RUNTIME_SCHEMA_VERSION
    query_run_id: str = Field(pattern=r"^qry_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    turn_id: str = Field(pattern=r"^turn_[0-9a-f]{32}$")
    attempt: int = Field(ge=1)
    parent_query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    replay_of_query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    status: QueryRunStatus
    current_stage: str = Field(default="accepted", min_length=1, max_length=64)
    revision: int = Field(ge=0)
    event_sequence: int = Field(ge=0)
    profile_version: str = Field(default="", max_length=128)
    index_snapshot: dict[str, Any] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    history_snapshot_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    history_base_turn_seq: int = Field(default=0, ge=0)
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = Field(default=None, max_length=128)
    safe_error_message: str | None = Field(default=None, max_length=1024)
    retryable: bool = False
    trace_id: str = Field(min_length=1, max_length=128)
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> "QueryRun":
        if self.status in RUN_TERMINAL_STATUSES and self.finished_at is None:
            raise ValueError("terminal query run requires finished_at")
        if self.status == QueryRunStatus.COMPLETED and self.result is None:
            raise ValueError("completed query run requires result")
        if self.status == QueryRunStatus.CANCEL_REQUESTED and self.cancel_requested_at is None:
            raise ValueError("cancel_requested run requires cancel_requested_at")
        if self.status in {QueryRunStatus.FAILED, QueryRunStatus.TIMED_OUT}:
            if not self.error_code or not self.safe_error_message:
                raise ValueError("failed/timed-out query run requires safe error fields")
        return self


class HistorySnapshotMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    turn_id: str
    turn_seq: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    committed_at: datetime


class ConversationHistorySnapshot(BaseModel):
    """一次 QueryRun 唯一可见的不可变历史输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["conversation-history.v1"] = HISTORY_SNAPSHOT_SCHEMA_VERSION
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    base_turn_seq: int = Field(ge=0)
    history_version: int = Field(ge=0)
    messages: tuple[HistorySnapshotMessage, ...] = Field(default_factory=tuple)
    summary: dict[str, Any] | None = None
    token_count: int = Field(ge=0)
    truncation_reason: str | None = Field(default=None, max_length=256)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def conservative_token_count(text: str) -> int:
    """无 tokenizer 时采用偏保守估算；组合根可注入模型真实 tokenizer。"""

    if not text:
        return 0
    # 中文字符通常接近 1 token，拉丁文本通常约 4 chars/token；取更保守值。
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    other = len(text) - cjk
    return cjk + math.ceil(other / 3)


def build_history_snapshot(
    *,
    conversation_id: str,
    tenant_id: str,
    base_turn_seq: int,
    history_version: int,
    messages: list[HistorySnapshotMessage],
    max_messages: int = 40,
    max_tokens: int = 4096,
    token_counter: Callable[[str], int] = conservative_token_count,
) -> ConversationHistorySnapshot:
    """按已提交顺序保留最近消息，并显式记录截断原因与稳定哈希。"""

    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    ordered = sorted(
        messages,
        key=lambda item: (item.turn_seq, item.ordinal, item.committed_at, item.message_id),
    )
    reasons: list[str] = []
    if len(ordered) > max_messages:
        ordered = ordered[-max_messages:]
        reasons.append("max_messages")
    selected_reversed: list[HistorySnapshotMessage] = []
    total = 0
    for message in reversed(ordered):
        cost = token_counter(message.content)
        if selected_reversed and total + cost > max_tokens:
            reasons.append("max_tokens")
            break
        if not selected_reversed and cost > max_tokens:
            # 最近一条消息也不能突破硬预算；不做静默字符串截断。
            reasons.append("latest_message_exceeds_token_budget")
            break
        selected_reversed.append(message)
        total += cost
    selected = tuple(reversed(selected_reversed))
    payload = {
        "schema_version": HISTORY_SNAPSHOT_SCHEMA_VERSION,
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "base_turn_seq": base_turn_seq,
        "history_version": history_version,
        "messages": [item.model_dump(mode="json") for item in selected],
        "summary": None,
        "token_count": total,
        "truncation_reason": ",".join(dict.fromkeys(reasons)) or None,
    }
    payload["content_hash"] = canonical_hash(payload)
    return ConversationHistorySnapshot.model_validate(payload)


class QueryEventType(StrEnum):
    RUN_ACCEPTED = "run.accepted"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    STAGE_STARTED = "stage.started"
    RETRIEVAL_SUMMARY = "retrieval.summary"
    ROUTE_DEGRADED = "route.degraded"
    ANSWER_DELTA = "answer.delta"
    CITATION_UPDATE = "citation.update"
    USAGE_UPDATE = "usage.update"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_TIMED_OUT = "run.timed_out"


TERMINAL_EVENT_TYPES = frozenset(
    {
        QueryEventType.RUN_COMPLETED,
        QueryEventType.RUN_FAILED,
        QueryEventType.RUN_CANCELLED,
        QueryEventType.RUN_TIMED_OUT,
    }
)


class QueryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-event.v1"] = QUERY_EVENT_SCHEMA_VERSION
    event_id: str = Field(pattern=r"^evt_[0-9a-f]{16}_[0-9]{20}$")
    query_run_id: str = Field(pattern=r"^qry_[0-9a-f]{32}$")
    seq: int = Field(ge=1)
    type: QueryEventType
    occurred_at: datetime
    stage: str | None = Field(default=None, max_length=64)
    data: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(min_length=1, max_length=128)
    visibility: Literal["public", "internal"] = "public"
    persistence_class: Literal["durable", "compactable"] = "durable"
    expires_at: datetime | None = None
    size_bytes: int = Field(ge=0)

    @field_validator("occurred_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("event timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_delta_semantics(self) -> "QueryEvent":
        if self.type == QueryEventType.ANSWER_DELTA:
            if self.data.get("operation") not in {"append", "replace"}:
                raise ValueError("answer.delta requires append or replace operation")
            if not isinstance(self.data.get("text"), str):
                raise ValueError("answer.delta requires text")
        return self


class EventPage(BaseModel):
    """EventStore 的有界读取结果；用于 SSE 背压判定和断线续传。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_run_id: str
    after_seq: int = Field(ge=0)
    events: tuple[QueryEvent, ...] = Field(default_factory=tuple)
    first_available_seq: int | None = Field(default=None, ge=1)
    last_available_seq: int = Field(default=0, ge=0)
    terminal: bool = False

    @property
    def lag_events(self) -> int:
        return max(0, self.last_available_seq - self.after_seq)


__all__ = [
    "Conversation",
    "ConversationHistorySnapshot",
    "ConversationStatus",
    "EventPage",
    "HISTORY_SNAPSHOT_SCHEMA_VERSION",
    "HistorySnapshotMessage",
    "InvalidStateTransition",
    "Message",
    "QUERY_EVENT_SCHEMA_VERSION",
    "QUERY_RUNTIME_SCHEMA_VERSION",
    "QueryEvent",
    "QueryEventType",
    "QueryRun",
    "QueryRunStatus",
    "RUN_TERMINAL_STATUSES",
    "TERMINAL_EVENT_TYPES",
    "TURN_TERMINAL_STATUSES",
    "Turn",
    "TurnStatus",
    "build_history_snapshot",
    "canonical_hash",
    "conservative_token_count",
    "ensure_run_transition",
    "ensure_turn_transition",
    "new_runtime_id",
]
