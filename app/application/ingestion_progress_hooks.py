"""把导入节点内部的批次进度投影到持久 Job。

LangGraph 只会在同步节点返回后产出节点事件。对于 CPU 密集的向量化节点，这会
让页面长时间停留在节点起始进度。这里通过 ``ContextVar`` 绑定当前租约身份，
使底层批次循环可以复用既有 ``stage_started`` 事件上报细粒度进度，同时避免把
``lease_token`` 放入 Graph state 或 checkpoint。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

from app.application.ingestion_job_service import start_job_stage


EMBEDDING_PROGRESS_START = 0.72
EMBEDDING_PROGRESS_END = 0.86
_EMBEDDING_PROGRESS_LAST_RUNNING = EMBEDDING_PROGRESS_END - 0.000001


@dataclass
class IngestionProgressHook:
    """持有一个 Worker 当前 Job 的租约保护进度写入口。"""

    job_id: str
    lease_token: str
    _last_embedding_bucket: int = field(
        default=int(EMBEDDING_PROGRESS_START * 100),
        init=False,
        repr=False,
    )
    _last_embedding_completed: int = field(default=-1, init=False, repr=False)

    def embedding_batch_progress(
        self,
        completed: int,
        total: int,
        failed_batches: int = 0,
    ) -> None:
        """按可见百分比节流持久化向量批次进度。

        节点完成前的值严格小于 86%；86% 仍由 ``import_runner`` 在节点真正返回
        后写入。相同整数百分比不重复写库，但最终批一定上报，便于页面显示
        ``completed == total``。底层 ``start_job_stage`` 会校验租约与取消状态。
        """

        if total <= 0:
            return
        bounded_total = int(total)
        bounded_completed = max(0, min(int(completed), bounded_total))
        if bounded_completed <= self._last_embedding_completed:
            return

        ratio = bounded_completed / bounded_total
        progress = min(
            EMBEDDING_PROGRESS_START
            + (EMBEDDING_PROGRESS_END - EMBEDDING_PROGRESS_START) * ratio,
            _EMBEDDING_PROGRESS_LAST_RUNNING,
        )
        bucket = int(progress * 100)
        is_final_batch = bounded_completed == bounded_total
        if bucket <= self._last_embedding_bucket and not is_final_batch:
            self._last_embedding_completed = bounded_completed
            return

        detail = {
            "node": "node_bge_embedding",
            "phase": "batch",
            "completed": bounded_completed,
            "total": bounded_total,
            "failed_batches": max(0, int(failed_batches)),
        }
        start_job_stage(
            self.job_id,
            self.lease_token,
            "node_bge_embedding",
            progress,
            detail,
        )
        self._last_embedding_bucket = max(self._last_embedding_bucket, bucket)
        self._last_embedding_completed = bounded_completed


_hook_context: ContextVar[IngestionProgressHook | None] = ContextVar(
    "stt_ingestion_progress_hook",
    default=None,
)


@contextmanager
def bind_ingestion_progress_hook(hook: IngestionProgressHook) -> Iterator[None]:
    """在当前 Graph 执行上下文绑定进度钩子，并保证异常时复位。"""

    token = _hook_context.set(hook)
    try:
        yield
    finally:
        _hook_context.reset(token)


def notify_ingestion_embedding_batch_progress(
    completed: int,
    total: int,
    failed_batches: int = 0,
) -> None:
    """由 embedding service 调用；非持久 Worker 上下文中安全地无操作。"""

    hook = _hook_context.get()
    if hook is not None:
        hook.embedding_batch_progress(completed, total, failed_batches)


__all__ = [
    "EMBEDDING_PROGRESS_END",
    "EMBEDDING_PROGRESS_START",
    "IngestionProgressHook",
    "bind_ingestion_progress_hook",
    "notify_ingestion_embedding_batch_progress",
]
