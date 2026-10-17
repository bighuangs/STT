"""TODO-02 独立持久队列 Worker。

启动：python -m app.worker.ingestion_worker
"""

from __future__ import annotations

import os
import platform
import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable

from app.application.control_plane_service import bootstrap_control_plane
from app.bootstrap.observability import observability_lifespan
from app.application.import_runner import run_graph_task
from app.application.ingestion_job_service import (
    cancel_claimed_job,
    claim_next_job,
    fail_job,
    get_job_payload,
    heartbeat_job,
    mark_job_running,
    schedule_job_retry,
    worker_heartbeat,
)
from app.application.job_errors import classify_job_error, retry_delay_seconds
from app.infra.persistence.sql.job_repository import (
    JobCancelledError,
    LeaseLostError,
)
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger
from app.infra.health import close_runtime_resources, validate_startup


def default_worker_id() -> str:
    configured = settings.ingestion_worker_id.strip()
    if configured:
        return configured
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class IngestionWorker:
    def __init__(
        self,
        *,
        worker_id: str | None = None,
        executor: Callable[[dict, str], dict] = run_graph_task,
        lease_seconds: int | None = None,
        heartbeat_seconds: int | None = None,
    ):
        self.worker_id = worker_id or default_worker_id()
        self.executor = executor
        self.lease_seconds = lease_seconds or settings.ingestion_lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or settings.ingestion_heartbeat_seconds
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("INGESTION_HEARTBEAT_SECONDS must be smaller than lease seconds")
        self.stop_event = threading.Event()

    def _worker_metadata(self) -> dict:
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "lease_seconds": self.lease_seconds,
        }

    def _heartbeat_loop(self, job_id: str, lease_token: str, done: threading.Event) -> None:
        while not done.wait(self.heartbeat_seconds):
            try:
                if not heartbeat_job(job_id, lease_token, self.lease_seconds):
                    logger.warning(f"[{job_id}] 心跳续租失败，Worker 将停止提交后续结果")
                    return
                worker_heartbeat(
                    self.worker_id,
                    "busy",
                    job_id,
                    self._worker_metadata(),
                )
            except Exception as exc:
                # 暂时数据库抖动不能直接杀执行；租约栅栏会拒绝过期 Worker 的最终写入。
                logger.error(f"[{job_id}] 心跳写入失败：{exc}")

    def run_once(self) -> bool:
        claimed = claim_next_job(self.worker_id, self.lease_seconds)
        if claimed is None:
            worker_heartbeat(self.worker_id, "idle", None, self._worker_metadata())
            return False

        job_id = claimed["id"]
        lease_token = claimed["lease_token"]
        logger.info(f"[{job_id}] {self.worker_id} 已领取租约")
        running = mark_job_running(job_id, lease_token)
        if running["status"] == "cancelled":
            worker_heartbeat(self.worker_id, "idle", None, self._worker_metadata())
            return True
        payload = get_job_payload(job_id)
        heartbeat_done = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, lease_token, heartbeat_done),
            name=f"heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self.executor(payload, lease_token)
        except JobCancelledError as exc:
            cancel_claimed_job(job_id, lease_token, str(exc))
            logger.info(f"[{job_id}] 已在安全点取消")
        except LeaseLostError as exc:
            # 新 Worker 已接管；旧执行器不得再改写状态。
            logger.warning(f"[{job_id}] {exc}")
        except BaseException as exc:
            classified = classify_job_error(exc)
            logger.exception(f"[{job_id}] 执行失败（{classified.category}）：{exc}")
            try:
                if classified.retryable:
                    schedule_job_retry(
                        job_id,
                        lease_token,
                        error_category=classified.category,
                        error_code=classified.code,
                        safe_message=classified.safe_message,
                        delay_seconds=retry_delay_seconds(
                            payload["run_attempt"],
                            settings.ingestion_retry_base_seconds,
                            settings.ingestion_retry_max_seconds,
                        ),
                    )
                else:
                    fail_job(
                        job_id,
                        lease_token,
                        error_category=classified.category,
                        error_code=classified.code,
                        safe_message=classified.safe_message,
                    )
            except LeaseLostError:
                logger.warning(f"[{job_id}] 记录失败结果时租约已被接管")
        finally:
            heartbeat_done.set()
            heartbeat_thread.join(timeout=max(1, self.heartbeat_seconds + 1))
            worker_heartbeat(self.worker_id, "idle", None, self._worker_metadata())
        return True

    def _run_forever(self) -> None:
        bootstrap_control_plane()
        validate_startup("worker")
        worker_heartbeat(self.worker_id, "idle", None, self._worker_metadata())
        logger.info(
            f"持久导入 Worker 已启动：{self.worker_id}，租约 {self.lease_seconds}s，"
            f"心跳 {self.heartbeat_seconds}s"
        )
        while not self.stop_event.is_set():
            try:
                handled = self.run_once()
            except Exception as exc:
                logger.exception(f"Worker 调度循环异常：{exc}")
                handled = False
            if not handled:
                self.stop_event.wait(settings.ingestion_worker_poll_seconds)
        worker_heartbeat(self.worker_id, "stopped", None, self._worker_metadata())
        close_runtime_resources()
        logger.info(f"持久导入 Worker 已停止：{self.worker_id}")

    def run_forever(self) -> None:
        """在统一组合根生命周期内运行持久导入 Worker。"""

        with observability_lifespan("ingestion-worker"):
            self._run_forever()

    def stop(self, *_args) -> None:
        self.stop_event.set()


def main() -> None:
    worker = IngestionWorker()
    signal.signal(signal.SIGINT, worker.stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
