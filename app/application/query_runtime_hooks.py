"""把既有 LangGraph 节点进度投影到持久 QueryRun。

旧 task_utils 仍可服务兼容页面；当可靠查询入口设置本上下文时，每个节点开始和
结束都会检查取消，并按状态机推进持久阶段。钩子只保存 tenant/user/run 身份，
每次写入使用短事务，保证另一个进程的 SSE 订阅立即可见。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterator

from app.application.query_runtime_service import QueryCancellationRequested
from app.domain.query_runtime import (
    QueryEventType,
    QueryRunStatus,
    RUN_TERMINAL_STATUSES,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.query_runtime_repository import QueryRuntimeRepository


_STAGE_ORDER = (
    QueryRunStatus.ACCEPTED,
    QueryRunStatus.PLANNING,
    QueryRunStatus.RETRIEVING,
    QueryRunStatus.PACKING,
    QueryRunStatus.GENERATING,
    QueryRunStatus.VALIDATING,
)
_NODE_STAGE = {
    "node_query_rewrite": QueryRunStatus.PLANNING,
    "node_search_embedding": QueryRunStatus.RETRIEVING,
    "node_search_embedding_hyde": QueryRunStatus.RETRIEVING,
    "node_web_search_mcp": QueryRunStatus.RETRIEVING,
    "node_rrf": QueryRunStatus.RETRIEVING,
    "node_rerank": QueryRunStatus.RETRIEVING,
    "node_question_contract": QueryRunStatus.PLANNING,
    "node_context_pack": QueryRunStatus.PACKING,
    "node_evidence_witness": QueryRunStatus.PACKING,
    "node_answer_plan": QueryRunStatus.GENERATING,
    "node_answer_verify": QueryRunStatus.VALIDATING,
    "node_answer_coverage": QueryRunStatus.VALIDATING,
    "node_answer_render": QueryRunStatus.VALIDATING,
    "node_answer_commit": QueryRunStatus.VALIDATING,
}


@dataclass
class QueryRuntimeProgressHook:
    tenant_id: str
    user_id: str
    query_run_id: str
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def node_started(self, node_name: str) -> None:
        with self._lock:
            self._advance(_NODE_STAGE.get(node_name))
            self._raise_if_cancelled()

    def node_finished(self, node_name: str) -> None:
        with self._lock:
            self._raise_if_cancelled()

    def advance_validating(self) -> None:
        with self._lock:
            self._advance(QueryRunStatus.VALIDATING)
            self._raise_if_cancelled()

    def answer_delta(self, delta: str) -> None:
        """把模型增量持久化为运行级事件，并在写入前后检查取消。"""

        if not delta:
            return
        with self._lock:
            self._raise_if_cancelled()
            with session_scope() as session:
                repository = QueryRuntimeRepository(
                    session,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                )
                repository.append_event(
                    self.query_run_id,
                    event_type=QueryEventType.ANSWER_DELTA,
                    stage=QueryRunStatus.GENERATING.value,
                    data={"operation": "append", "text": delta},
                )
            self._raise_if_cancelled()

    def _advance(self, target: QueryRunStatus | None) -> None:
        if target is None:
            return
        with session_scope() as session:
            repository = QueryRuntimeRepository(
                session,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
            )
            run = repository.get_run(self.query_run_id)
            if run.status in RUN_TERMINAL_STATUSES:
                return
            if run.status == QueryRunStatus.CANCEL_REQUESTED:
                raise QueryCancellationRequested(self.query_run_id)
            try:
                current_index = _STAGE_ORDER.index(run.status)
                target_index = _STAGE_ORDER.index(target)
            except ValueError:
                return
            if target_index <= current_index:
                return
            for stage in _STAGE_ORDER[current_index + 1 : target_index + 1]:
                run = repository.transition_run(
                    self.query_run_id,
                    expected_revision=run.revision,
                    target=stage,
                    stage=stage.value,
                )

    def _raise_if_cancelled(self) -> None:
        with session_scope() as session:
            run = QueryRuntimeRepository(
                session,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
            ).get_run(self.query_run_id)
            if run.status in {
                QueryRunStatus.CANCEL_REQUESTED,
                QueryRunStatus.CANCELLED,
            }:
                raise QueryCancellationRequested(self.query_run_id)


_hook_context: ContextVar[QueryRuntimeProgressHook | None] = ContextVar(
    "stt_query_runtime_progress_hook",
    default=None,
)


@contextmanager
def bind_query_runtime_hook(hook: QueryRuntimeProgressHook) -> Iterator[None]:
    token = _hook_context.set(hook)
    try:
        yield
    finally:
        _hook_context.reset(token)


def notify_query_node_started(node_name: str) -> None:
    hook = _hook_context.get()
    if hook is not None:
        hook.node_started(node_name)


def notify_query_node_finished(node_name: str) -> None:
    hook = _hook_context.get()
    if hook is not None:
        hook.node_finished(node_name)


def notify_query_answer_delta(delta: str) -> None:
    hook = _hook_context.get()
    if hook is not None:
        hook.answer_delta(delta)


__all__ = [
    "QueryRuntimeProgressHook",
    "bind_query_runtime_hook",
    "notify_query_answer_delta",
    "notify_query_node_finished",
    "notify_query_node_started",
]
