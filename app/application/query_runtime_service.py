"""TODO-24/25/31 查询运行应用服务。

该服务是查询图与持久运行控制面的边界：组合根负责开启 SQL 事务并构造
``QueryRuntimeRepository``，节点只接收稳定的 run identity、冻结历史和协作式
``CancellationToken``，不得重新按 session 查询历史或创建自己的 SSE 队列。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.domain.query_runtime import (
    Conversation,
    ConversationHistorySnapshot,
    QueryEvent,
    QueryEventType,
    QueryRun,
    QueryRunStatus,
    Turn,
)
from app.infra.persistence.sql.query_runtime_repository import QueryRuntimeRepository


class QueryCancellationRequested(RuntimeError):
    """协作式取消检查命中；调用方应释放下游资源并终结为 CANCELLED。"""

    def __init__(self, query_run_id: str) -> None:
        super().__init__(f"query run cancellation requested: {query_run_id}")
        self.query_run_id = query_run_id


@dataclass(frozen=True)
class AcceptedQuery:
    conversation: Conversation
    turn: Turn
    run: QueryRun
    history: ConversationHistorySnapshot
    created: bool

    def graph_identity(self) -> dict[str, Any]:
        """返回可注入 QueryGraphState 的不可变身份，不包含用户可覆盖 scope。"""

        return {
            "conversation_id": self.conversation.conversation_id,
            # 兼容期 session_id 只做展示别名；task/event/cancel 必须使用 query_run_id。
            "session_id": self.conversation.conversation_id,
            "turn_id": self.turn.turn_id,
            "query_run_id": self.run.query_run_id,
            "tenant_id": self.run.tenant_id,
            "user_id": self.run.user_id,
            "kb_id": self.conversation.kb_id,
            "history": [
                {
                    "message_id": message.message_id,
                    "turn_id": message.turn_id,
                    "role": message.role,
                    "text": message.content,
                    "committed_at": message.committed_at.isoformat(),
                }
                for message in self.history.messages
            ],
            "history_snapshot_hash": self.history.content_hash,
            "history_snapshot_loaded": True,
            "history_base_turn_seq": self.history.base_turn_seq,
            "history_version": self.history.history_version,
            "trace_id": self.run.trace_id,
        }


class CancellationToken:
    """数据库事实驱动的协作式取消令牌。

    每个耗时节点及外部调用前后调用 ``raise_if_cancelled``。SDK 支持原生取消时，
    上层可在捕获异常后同时取消 HTTP/LLM stream；不支持时必须设置硬超时并丢弃
    迟到结果。
    """

    def __init__(self, repository: QueryRuntimeRepository, query_run_id: str) -> None:
        self._repository = repository
        self.query_run_id = query_run_id

    @property
    def is_cancelled(self) -> bool:
        return self._repository.cancellation_requested(self.query_run_id)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise QueryCancellationRequested(self.query_run_id)


class QueryRuntimeService:
    def __init__(self, repository: QueryRuntimeRepository) -> None:
        self.repository = repository

    def accept(
        self,
        *,
        kb_id: str,
        client_turn_id: str,
        user_content: str,
        conversation_id: str | None = None,
        conversation_title: str = "",
        idempotency_key: str | None = None,
        profile_version: str = "",
        index_snapshot: dict[str, Any] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        deadline_at: datetime | None = None,
        max_history_messages: int = 40,
        max_history_tokens: int = 4096,
    ) -> AcceptedQuery:
        if conversation_id is None:
            conversation = self.repository.create_conversation(
                kb_id=kb_id,
                title=conversation_title,
            )
        else:
            conversation = self.repository.get_conversation(conversation_id)
            if conversation.kb_id != kb_id:
                raise ValueError("conversation knowledge base is immutable")
        turn, run, history, created = self.repository.begin_turn(
            conversation_id=conversation.conversation_id,
            client_turn_id=client_turn_id,
            user_content=user_content,
            idempotency_key=idempotency_key,
            profile_version=profile_version,
            index_snapshot=index_snapshot,
            config_snapshot=config_snapshot,
            deadline_at=deadline_at,
            max_history_messages=max_history_messages,
            max_history_tokens=max_history_tokens,
        )
        return AcceptedQuery(
            conversation=self.repository.get_conversation(conversation.conversation_id),
            turn=turn,
            run=run,
            history=history,
            created=created,
        )

    def cancellation_token(self, query_run_id: str) -> CancellationToken:
        self.repository.get_run(query_run_id)
        return CancellationToken(self.repository, query_run_id)

    def enter_stage(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        stage: QueryRunStatus,
    ) -> QueryRun:
        if stage not in {
            QueryRunStatus.PLANNING,
            QueryRunStatus.RETRIEVING,
            QueryRunStatus.PACKING,
            QueryRunStatus.GENERATING,
            QueryRunStatus.VALIDATING,
        }:
            raise ValueError("enter_stage only accepts non-terminal execution stages")
        return self.repository.transition_run(
            query_run_id,
            expected_revision=expected_revision,
            target=stage,
            stage=stage.value,
        )

    def emit(
        self,
        query_run_id: str,
        *,
        event_type: QueryEventType,
        data: dict[str, Any],
        stage: str | None = None,
        visibility: str = "public",
        persistence_class: str = "durable",
    ) -> QueryEvent:
        if event_type in {
            QueryEventType.RUN_ACCEPTED,
            QueryEventType.RUN_CANCEL_REQUESTED,
            QueryEventType.RUN_COMPLETED,
            QueryEventType.RUN_FAILED,
            QueryEventType.RUN_CANCELLED,
            QueryEventType.RUN_TIMED_OUT,
        }:
            raise ValueError("run lifecycle events are emitted by state transition methods")
        return self.repository.append_event(
            query_run_id,
            event_type=event_type,
            data=data,
            stage=stage,
            visibility=visibility,
            persistence_class=persistence_class,
        )

    def complete(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        answer: str,
        answer_envelope: dict[str, Any] | None = None,
        public_result: dict[str, Any] | None = None,
    ) -> QueryRun:
        result = dict(public_result or {})
        if answer_envelope is not None:
            result["answer_envelope"] = answer_envelope
        return self.repository.complete_run(
            query_run_id,
            expected_revision=expected_revision,
            answer=answer,
            result=result,
        )

    def fail(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        error_code: str,
        safe_message: str,
        retryable: bool,
        error_detail_ref: str | None = None,
        timed_out: bool = False,
    ) -> QueryRun:
        return self.repository.fail_run(
            query_run_id,
            expected_revision=expected_revision,
            error_code=error_code,
            safe_message=safe_message,
            retryable=retryable,
            error_detail_ref=error_detail_ref,
            timed_out=timed_out,
        )

    def request_cancel(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        reason_code: str = "USER_REQUESTED",
    ) -> tuple[QueryRun, bool]:
        return self.repository.request_cancel(
            query_run_id,
            expected_revision=expected_revision,
            reason_code=reason_code,
        )

    def acknowledge_cancel(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
    ) -> QueryRun:
        return self.repository.finalize_cancel(
            query_run_id,
            expected_revision=expected_revision,
        )

    def retry(
        self,
        query_run_id: str,
        *,
        idempotency_key: str,
        deadline_at: datetime | None = None,
    ) -> tuple[QueryRun, ConversationHistorySnapshot, bool]:
        return self.repository.retry_run(
            query_run_id,
            idempotency_key=idempotency_key,
            deadline_at=deadline_at,
        )


__all__ = [
    "AcceptedQuery",
    "CancellationToken",
    "QueryCancellationRequested",
    "QueryRuntimeService",
]
