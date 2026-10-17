"""TODO-24/31 查询进程重启后的持久运行收敛。

FastAPI BackgroundTasks 不跨进程存活，因此服务启动时必须处理控制库中已超过
``deadline_at`` 的非终态 QueryRun。维护任务只根据持久状态和 revision 做 CAS：
取消请求收敛为 CANCELLED，其余执行态收敛为 TIMED_OUT；不会尝试猜测或提交迟到
答案。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from app.domain._state_machine import InvalidStateTransition
from app.domain.query_runtime import QueryRunStatus, RUN_TERMINAL_STATUSES
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.infra.persistence.sql.query_runtime_models import QueryRunModel
from app.infra.persistence.sql.query_runtime_repository import (
    QueryRuntimeRepository,
    StaleQueryRevision,
)
from app.infra.persistence.sql.repositories import ResourceConflictError
from app.shared.runtime.logger import logger


_RECOVERABLE_STATUSES = (
    QueryRunStatus.ACCEPTED,
    QueryRunStatus.PLANNING,
    QueryRunStatus.RETRIEVING,
    QueryRunStatus.PACKING,
    QueryRunStatus.GENERATING,
    QueryRunStatus.VALIDATING,
    QueryRunStatus.CANCEL_REQUESTED,
)


def recover_expired_query_runs(
    *,
    now: datetime | None = None,
    limit: int = 1000,
) -> dict[str, int]:
    """终结一批已过期持久运行，返回不含用户正文的维护统计。"""

    if not 1 <= limit <= 10_000:
        raise ValueError("query runtime recovery limit must be in 1..10000")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("query runtime recovery timestamp must include timezone")
    stats = {
        "scanned": 0,
        "timed_out": 0,
        "cancelled": 0,
        "raced": 0,
        "failed": 0,
    }
    with session_scope() as session:
        candidates = session.execute(
            select(
                QueryRunModel.id,
                QueryRunModel.tenant_id,
                QueryRunModel.user_id,
            )
            .where(
                QueryRunModel.status.in_([status.value for status in _RECOVERABLE_STATUSES]),
                QueryRunModel.deadline_at.is_not(None),
                QueryRunModel.deadline_at <= current,
            )
            .order_by(QueryRunModel.deadline_at, QueryRunModel.id)
            .limit(limit)
        ).all()
        stats["scanned"] = len(candidates)
        for query_run_id, tenant_id, user_id in candidates:
            try:
                with session.begin_nested():
                    repository = QueryRuntimeRepository(
                        session,
                        tenant_id=tenant_id,
                        user_id=user_id,
                    )
                    run = repository.get_run(query_run_id)
                    if run.status in RUN_TERMINAL_STATUSES:
                        stats["raced"] += 1
                        continue
                    if run.deadline_at is None:
                        stats["raced"] += 1
                        continue
                    deadline = (
                        run.deadline_at
                        if run.deadline_at.tzinfo is not None
                        else run.deadline_at.replace(tzinfo=UTC)
                    )
                    if deadline > current:
                        stats["raced"] += 1
                        continue
                    if run.status == QueryRunStatus.CANCEL_REQUESTED:
                        repository.finalize_cancel(
                            query_run_id,
                            expected_revision=run.revision,
                        )
                        QueryReliabilityRepository(
                            session,
                            tenant_id=tenant_id,
                        ).complete_idempotency(query_run_id, 409)
                        stats["cancelled"] += 1
                    else:
                        repository.fail_run(
                            query_run_id,
                            expected_revision=run.revision,
                            error_code="QUERY_TIMED_OUT_AFTER_RESTART",
                            safe_message=("查询在服务重启或执行中断后超过截止时间，可安全重试。"),
                            retryable=True,
                            error_detail_ref=f"trace:{run.trace_id}",
                            timed_out=True,
                        )
                        QueryReliabilityRepository(
                            session,
                            tenant_id=tenant_id,
                        ).complete_idempotency(query_run_id, 504)
                        stats["timed_out"] += 1
            except (
                InvalidStateTransition,
                ResourceConflictError,
                StaleQueryRevision,
            ):
                # 正在运行的 worker 可能先完成/取消；CAS 失败即接受胜出事实。
                stats["raced"] += 1
            except Exception:
                stats["failed"] += 1
                logger.exception(
                    "查询运行启动恢复失败：query_run_id={}",
                    query_run_id,
                )
    return stats


async def run_query_runtime_recovery_loop(
    stop: asyncio.Event,
    *,
    interval_seconds: float,
) -> None:
    """周期收敛后台执行丢失后的过期运行；多进程并发由 CAS 保证安全。"""

    if interval_seconds <= 0:
        raise ValueError("query runtime recovery interval must be positive")
    while not stop.is_set():
        try:
            report = await asyncio.to_thread(recover_expired_query_runs)
            if report["scanned"]:
                logger.info("查询运行周期恢复完成：{}", report)
        except Exception:
            # 单次控制库故障不应杀死维护循环；健康检查会独立暴露依赖状态。
            logger.exception("查询运行周期恢复失败")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


__all__ = [
    "recover_expired_query_runs",
    "run_query_runtime_recovery_loop",
]
