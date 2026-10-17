"""导入任务领域状态。"""

from enum import StrEnum

from app.domain._state_machine import ensure_transition


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    # TODO-01 兼容值：旧代码/历史行仍可读取，新的 Worker 不再写入。
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    # TODO-01 兼容值：对外序列化会同时给出 terminal=True。
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS = {
    IngestionJobStatus.QUEUED: {
        IngestionJobStatus.LEASED,
        IngestionJobStatus.PROCESSING,
        IngestionJobStatus.CANCELLED,
        IngestionJobStatus.FAILED,
    },
    IngestionJobStatus.LEASED: {
        IngestionJobStatus.RUNNING,
        IngestionJobStatus.RETRY_WAIT,
        IngestionJobStatus.FAILED,
        IngestionJobStatus.CANCELLED,
    },
    IngestionJobStatus.RUNNING: {
        IngestionJobStatus.RETRY_WAIT,
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED,
        IngestionJobStatus.CANCELLED,
    },
    IngestionJobStatus.RETRY_WAIT: {
        IngestionJobStatus.LEASED,
        IngestionJobStatus.CANCELLED,
        IngestionJobStatus.FAILED,
    },
    IngestionJobStatus.PROCESSING: {
        IngestionJobStatus.COMPLETED,
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED,
        IngestionJobStatus.CANCELLED,
    },
    IngestionJobStatus.SUCCEEDED: set(),
    IngestionJobStatus.COMPLETED: set(),
    IngestionJobStatus.FAILED: set(),
    IngestionJobStatus.CANCELLED: set(),
}


def ensure_ingestion_job_transition(current: str, target: str) -> None:
    ensure_transition(
        IngestionJobStatus(current), IngestionJobStatus(target), _TRANSITIONS
    )
