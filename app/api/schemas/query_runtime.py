"""TODO-24/25/26/31 可靠查询 HTTP 契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.query_runtime import (
    Conversation,
    EventPage,
    Message,
    QueryRun,
    QueryRunStatus,
    Turn,
    TurnStatus,
)
from app.domain.query_security import RequestedDocument
from app.domain.retrieval import FilterNode
from app.application.query_cache_service import QueryReplayManifest, ReplayResult


class QueryRunAcceptRequest(BaseModel):
    """客户端只表达期望范围；授权、发布版本和配置快照均由服务端冻结。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=100_000)
    kb_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    documents: tuple[RequestedDocument, ...] = Field(
        default_factory=tuple,
        max_length=10_000,
    )
    business_filter: FilterNode | None = None
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^conv_[0-9a-f]{32}$",
    )
    client_turn_id: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_title: str = Field(default="", max_length=512)
    workspace_id: str | None = Field(default=None, min_length=1, max_length=64)
    web_search_requested: bool = False
    deadline_seconds: int | None = Field(default=None, ge=1, le=3600)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query cannot be blank")
        return normalized

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("kb_ids cannot contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("kb_ids cannot contain duplicates")
        return normalized


class QueryRunCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int | None = Field(default=None, ge=0)
    reason_code: str = Field(
        default="USER_REQUESTED",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9_]{0,127}$",
    )


class QueryRunRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    deadline_seconds: int | None = Field(default=None, ge=1, le=3600)


class QueryRunReplayRequest(BaseModel):
    """只读重放；当前在线接口仅允许固定旧快照且不再次外呼模型。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_context"] = "fixed_context"


class AuthorizedScopeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_hash: str
    workspace_id: str
    kb_ids: tuple[str, ...]
    document_restricted: bool
    active_index_count: int = Field(ge=0)
    policy_version: str
    entitlement_version: str


class RetrievalProfileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    version: int = Field(ge=1)
    config_hash: str


class QueryRunAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "query-runtime.v1"
    conversation_id: str
    turn_id: str
    query_run_id: str
    attempt: int = Field(ge=1)
    status: QueryRunStatus
    revision: int = Field(ge=0)
    created: bool
    events_url: str
    stream_url: str
    result_url: str
    authorized_scope: AuthorizedScopeSummary
    retrieval_profile: RetrievalProfileIdentity


class QueryRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "query-runtime.v1"
    query_run_id: str
    conversation_id: str
    turn_id: str
    attempt: int
    parent_query_run_id: str | None = None
    replay_of_query_run_id: str | None = None
    status: QueryRunStatus
    current_stage: str
    revision: int
    event_sequence: int
    history_snapshot_hash: str | None = None
    history_base_turn_seq: int
    trace_id: str
    result_available: bool
    retryable: bool
    error_code: str | None = None
    safe_error_message: str | None = None
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, run: QueryRun) -> "QueryRunResponse":
        return cls(
            query_run_id=run.query_run_id,
            conversation_id=run.conversation_id,
            turn_id=run.turn_id,
            attempt=run.attempt,
            parent_query_run_id=run.parent_query_run_id,
            replay_of_query_run_id=run.replay_of_query_run_id,
            status=run.status,
            current_stage=run.current_stage,
            revision=run.revision,
            event_sequence=run.event_sequence,
            history_snapshot_hash=run.history_snapshot_hash,
            history_base_turn_seq=run.history_base_turn_seq,
            trace_id=run.trace_id,
            result_available=run.status == QueryRunStatus.COMPLETED and run.result is not None,
            retryable=run.retryable,
            error_code=run.error_code,
            safe_error_message=run.safe_error_message,
            deadline_at=run.deadline_at,
            cancel_requested_at=run.cancel_requested_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class QueryRunResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "query-result.v1"
    query_run_id: str
    status: QueryRunStatus
    result: dict[str, Any]


class QueryRunCancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_run_id: str
    accepted: bool
    status: QueryRunStatus
    revision: int


class QueryEventPageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: EventPage


class ConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[Conversation, ...]
    limit: int
    offset: int
    status_filter: Literal["active", "archived", "all"] = "active"
    search: str = ""


class ConversationMutationRequest(BaseModel):
    """会话状态操作必须携带客户端最后读取到的 revision。"""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)


class ConversationMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-mutation.v1"] = "conversation-mutation.v1"
    conversation: Conversation


class ConversationDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-mutation.v1"] = "conversation-mutation.v1"
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    deleted: Literal[True] = True
    revision: int = Field(ge=1)


class ConversationRunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    items: tuple[QueryRunResponse, ...]


class ConversationMessageResponse(BaseModel):
    """面向最终用户的已提交会话消息，不暴露 partial/tool/system 内部记录。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-message.v1"] = "conversation-message.v1"
    message_id: str = Field(pattern=r"^msg_[0-9a-f]{32}$")
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    turn_id: str = Field(pattern=r"^turn_[0-9a-f]{32}$")
    query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    role: Literal["user", "assistant"]
    status: Literal["committed"] = "committed"
    content: str = Field(max_length=1_000_000)
    ordinal: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    committed_at: datetime

    @classmethod
    def from_domain(
        cls,
        message: Message,
        *,
        metadata: dict[str, Any],
    ) -> "ConversationMessageResponse":
        if message.role not in {"user", "assistant"} or message.status != "committed":
            raise ValueError("conversation transcript only accepts committed user/assistant")
        if message.committed_at is None:
            raise ValueError("committed conversation message requires committed_at")
        return cls(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            turn_id=message.turn_id,
            query_run_id=message.query_run_id,
            role=message.role,
            content=message.content,
            ordinal=message.ordinal,
            metadata=metadata,
            created_at=message.created_at,
            committed_at=message.committed_at,
        )


class ConversationTurnResponse(BaseModel):
    """在途与失败轮次的公开投影，不返回 prompt、快照、trace 或内部错误。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-turn.v1"] = "conversation-turn.v1"
    turn_id: str = Field(pattern=r"^turn_[0-9a-f]{32}$")
    conversation_id: str = Field(pattern=r"^conv_[0-9a-f]{32}$")
    seq: int = Field(ge=1)
    status: TurnStatus
    user_content: str = Field(max_length=100_000)
    query_run_id: str | None = Field(default=None, pattern=r"^qry_[0-9a-f]{32}$")
    query_run_status: QueryRunStatus | None = None
    current_stage: str | None = Field(default=None, max_length=64)
    retryable: bool = False
    error_code: str | None = Field(default=None, max_length=128)
    safe_error_message: str | None = Field(default=None, max_length=1024)
    created_at: datetime
    resolved_at: datetime | None = None

    @classmethod
    def from_domain(
        cls,
        turn: Turn,
        *,
        run: QueryRun | None,
    ) -> "ConversationTurnResponse":
        return cls(
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            seq=turn.seq,
            status=turn.status,
            user_content=turn.user_content,
            query_run_id=run.query_run_id if run is not None else turn.active_query_run_id,
            query_run_status=run.status if run is not None else None,
            current_stage=run.current_stage if run is not None else None,
            retryable=run.retryable if run is not None else False,
            error_code=run.error_code if run is not None else None,
            safe_error_message=run.safe_error_message if run is not None else None,
            created_at=turn.created_at,
            resolved_at=turn.resolved_at,
        )


class ConversationMessageListResponse(BaseModel):
    """当前登录用户仍有权访问的完整、持久化会话 transcript。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-transcript.v1"] = "conversation-transcript.v1"
    conversation: Conversation
    items: tuple[ConversationMessageResponse, ...]
    turns: tuple[ConversationTurnResponse, ...] = Field(default_factory=tuple)


class QueryReplayManifestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: QueryReplayManifest


class QueryRunReplayResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: ReplayResult


__all__ = [
    "AuthorizedScopeSummary",
    "ConversationListResponse",
    "ConversationDeleteResponse",
    "ConversationMessageListResponse",
    "ConversationMessageResponse",
    "ConversationMutationRequest",
    "ConversationMutationResponse",
    "ConversationRunListResponse",
    "ConversationTurnResponse",
    "QueryEventPageResponse",
    "QueryRunAcceptRequest",
    "QueryRunAcceptedResponse",
    "QueryRunCancelRequest",
    "QueryRunCancelResponse",
    "QueryRunResponse",
    "QueryRunReplayRequest",
    "QueryRunReplayResponse",
    "QueryRunResultResponse",
    "QueryRunRetryRequest",
    "RetrievalProfileIdentity",
    "QueryReplayManifestResponse",
]
