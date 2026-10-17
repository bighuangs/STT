"""评测 BackgroundTask 崩溃后的持久运行收敛。

评测当前由查询进程内的 ``BackgroundTasks`` 执行，不能跨进程恢复 Python
调用栈。执行器会把 owner/heartbeat 写入既有 ``gate_result`` JSON，并通过
``updated_at`` 作为数据库无关的 CAS 栅栏。恢复器只把失去心跳的 queued/running
运行终结为 interrupted/failed，保留真实进度与已落盘 case；绝不补写结果或报告。
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.evaluation_repository import EvaluationRepository
from app.infra.persistence.sql.models import EvaluationRunModel
from app.shared.runtime.logger import logger


EVALUATION_INTERRUPTED_CODE = "EVALUATION_RUN_INTERRUPTED"
DEFAULT_HEARTBEAT_SECONDS = 10.0
DEFAULT_STALE_AFTER_SECONDS = 90.0

_PROCESS_OWNER = (
    f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
)


def evaluation_execution_owner() -> str:
    return _PROCESS_OWNER


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _owner_from_gate(gate_result: object) -> str | None:
    if not isinstance(gate_result, dict):
        return None
    execution = gate_result.get("execution")
    if not isinstance(execution, dict):
        return None
    value = str(execution.get("owner_id") or "").strip()
    return value or None


def recover_stale_evaluation_runs(
    *,
    now: datetime | None = None,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    limit: int = 1000,
) -> dict[str, int]:
    """终结一批已失去心跳的评测运行，不伪造剩余 case 或完成态。"""

    if stale_after_seconds <= 0:
        raise ValueError("evaluation stale timeout must be positive")
    if not 1 <= limit <= 10_000:
        raise ValueError("evaluation recovery limit must be in 1..10000")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("evaluation recovery timestamp must include timezone")
    cutoff = current - timedelta(seconds=stale_after_seconds)
    stats = {"scanned": 0, "interrupted": 0, "raced": 0, "failed": 0}
    with session_scope() as session:
        candidates = session.execute(
            select(
                EvaluationRunModel.id,
                EvaluationRunModel.tenant_id,
                EvaluationRunModel.updated_at,
            )
            .where(
                EvaluationRunModel.status.in_(("queued", "running")),
                EvaluationRunModel.updated_at <= cutoff,
            )
            .order_by(EvaluationRunModel.updated_at, EvaluationRunModel.id)
            .limit(limit)
        ).all()
        stats["scanned"] = len(candidates)
        for run_id, tenant_id, updated_at in candidates:
            try:
                with session.begin_nested():
                    interrupted = EvaluationRepository(
                        session,
                        tenant_id=tenant_id,
                        actor_id="evaluation-recovery",
                    ).interrupt_run(
                        run_id,
                        expected_updated_at=_aware(updated_at),
                        error_code=EVALUATION_INTERRUPTED_CODE,
                        reason=(
                            "评测执行进程已中断且心跳超时；已保留真实 case 结果与进度，"
                            "未执行的 case 未被伪造。请新建运行重试。"
                        ),
                        now=current,
                    )
                stats["interrupted" if interrupted else "raced"] += 1
            except Exception:
                stats["failed"] += 1
                # 维护边界只记录稳定定位字段，禁止异常正文与 traceback 进入日志。
                logger.warning("评测运行崩溃恢复失败：run_id={}", run_id)
    return stats


def interrupt_owned_evaluation_runs(
    owner_id: str,
    *,
    now: datetime | None = None,
    limit: int = 1000,
) -> dict[str, int]:
    """正常关闭时明确终结本进程仍拥有的运行。"""

    if not owner_id.strip():
        raise ValueError("evaluation owner id is required")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("evaluation shutdown timestamp must include timezone")
    stats = {"scanned": 0, "interrupted": 0, "raced": 0, "failed": 0}
    with session_scope() as session:
        candidates = session.execute(
            select(
                EvaluationRunModel.id,
                EvaluationRunModel.tenant_id,
                EvaluationRunModel.updated_at,
                EvaluationRunModel.gate_result,
            )
            .where(EvaluationRunModel.status.in_(("queued", "running")))
            .order_by(EvaluationRunModel.updated_at, EvaluationRunModel.id)
            .limit(limit)
        ).all()
        owned = [item for item in candidates if _owner_from_gate(item.gate_result) == owner_id]
        stats["scanned"] = len(owned)
        for run_id, tenant_id, updated_at, _gate_result in owned:
            try:
                with session.begin_nested():
                    interrupted = EvaluationRepository(
                        session,
                        tenant_id=tenant_id,
                        actor_id="evaluation-shutdown",
                    ).interrupt_run(
                        run_id,
                        expected_updated_at=_aware(updated_at),
                        expected_owner=owner_id,
                        error_code=EVALUATION_INTERRUPTED_CODE,
                        reason=(
                            "评测所属查询进程已停止；已保留真实 case 结果与进度，"
                            "未执行的 case 未被伪造。请新建运行重试。"
                        ),
                        now=current,
                    )
                stats["interrupted" if interrupted else "raced"] += 1
            except Exception:
                stats["failed"] += 1
                # 维护边界只记录稳定定位字段，禁止异常正文与 traceback 进入日志。
                logger.warning("评测运行关闭收敛失败：run_id={}", run_id)
    return stats


async def run_evaluation_recovery_loop(
    stop: asyncio.Event,
    *,
    interval_seconds: float,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> None:
    """周期回收 OOM/kill 后停止更新心跳的评测运行。"""

    if interval_seconds <= 0:
        raise ValueError("evaluation recovery interval must be positive")
    while not stop.is_set():
        try:
            report = await asyncio.to_thread(
                recover_stale_evaluation_runs,
                stale_after_seconds=stale_after_seconds,
            )
            if report["scanned"]:
                logger.info("评测运行周期恢复完成：{}", report)
        except Exception:
            logger.warning("评测运行周期恢复失败")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "DEFAULT_STALE_AFTER_SECONDS",
    "EVALUATION_INTERRUPTED_CODE",
    "evaluation_execution_owner",
    "interrupt_owned_evaluation_runs",
    "recover_stale_evaluation_runs",
    "run_evaluation_recovery_loop",
]
