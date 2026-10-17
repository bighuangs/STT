"""TODO-02 持久导入任务应用服务。"""

from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.job_repository import DurableJobRepository


def claim_next_job(worker_id: str, lease_seconds: int) -> dict | None:
    with session_scope() as session:
        return DurableJobRepository(session).claim_next(worker_id, lease_seconds)


def get_job_payload(job_id: str) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).payload(job_id)


def mark_job_running(job_id: str, lease_token: str) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).mark_running(job_id, lease_token)


def heartbeat_job(job_id: str, lease_token: str, lease_seconds: int) -> bool:
    with session_scope() as session:
        return DurableJobRepository(session).heartbeat(job_id, lease_token, lease_seconds)


def is_cancellation_requested(job_id: str, lease_token: str) -> bool:
    with session_scope() as session:
        return DurableJobRepository(session).cancellation_requested(job_id, lease_token)


def update_job_stage(job_id: str, lease_token: str, stage: str, progress: float, detail: dict | None = None) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).update_stage(
            job_id, lease_token, stage, progress, detail
        )


def start_job_stage(job_id: str, lease_token: str, stage: str, progress: float, detail: dict | None = None) -> dict:
    """持久化正在执行的节点，但不把该节点误记为已完成。"""
    with session_scope() as session:
        return DurableJobRepository(session).start_stage(
            job_id, lease_token, stage, progress, detail
        )


def record_job_checkpoint(job_id: str, lease_token: str, **metadata) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).record_checkpoint(
            job_id, lease_token, **metadata
        )


def record_index_manifest(job_id: str, lease_token: str, manifest: dict) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).record_manifest(job_id, lease_token, manifest)


def get_latest_checkpoint(job_id: str) -> dict | None:
    with session_scope() as session:
        return DurableJobRepository(session).latest_checkpoint(job_id)


def schedule_job_retry(job_id: str, lease_token: str, **error) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).schedule_retry(job_id, lease_token, **error)


def fail_job(job_id: str, lease_token: str, **error) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).fail(job_id, lease_token, **error)


def complete_job(job_id: str, lease_token: str, chunk_count: int, token_count: int = 0) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).complete(
            job_id, lease_token, chunk_count=chunk_count, token_count=token_count
        )


def cancel_job(job_id: str, reason: str = "requested by user") -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).request_cancel(job_id, reason)


def cancel_claimed_job(job_id: str, lease_token: str, reason: str) -> dict:
    with session_scope() as session:
        return DurableJobRepository(session).cancel_claimed(job_id, lease_token, reason)


def list_jobs(kb_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
    with session_scope() as session:
        return DurableJobRepository(session).list_jobs(kb_id=kb_id, status=status, limit=limit)


def list_job_events(job_id: str, after_sequence: int = 0, limit: int = 500) -> list[dict]:
    with session_scope() as session:
        return DurableJobRepository(session).events(job_id, after_sequence, limit)


def worker_heartbeat(worker_id: str, status: str, current_job_id: str | None, metadata: dict | None = None) -> None:
    with session_scope() as session:
        DurableJobRepository(session).worker_heartbeat(
            worker_id, status=status, current_job_id=current_job_id, metadata=metadata
        )


def list_workers() -> list[dict]:
    with session_scope() as session:
        return DurableJobRepository(session).list_workers()
