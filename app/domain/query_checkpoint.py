"""Query 节点安全边界 checkpoint 的持久化契约。

该契约只保证已提交节点边界可恢复。节点一旦标记 ``started`` 而没有原子提交下一
checkpoint，就必须收敛为 interrupted，禁止把潜在外部副作用伪装成 exactly-once。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


QUERY_CHECKPOINT_SCHEMA_VERSION = "query-checkpoint.v2"
QUERY_STATE_SCHEMA_VERSION = "query-state.v3"
QUERY_CHECKPOINT_NAMESPACE = "query-runtime-proof-v2"


class QueryExecutionStatus(StrEnum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


QUERY_EXECUTION_TERMINAL_STATUSES = frozenset(
    {
        QueryExecutionStatus.COMPLETED,
        QueryExecutionStatus.INTERRUPTED,
        QueryExecutionStatus.FAILED,
        QueryExecutionStatus.CANCELLED,
        QueryExecutionStatus.TIMED_OUT,
    }
)


class QueryNodeAttemptStatus(StrEnum):
    SCHEDULED = "scheduled"
    STARTED = "started"
    CHECKPOINTED = "checkpointed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class QueryExecutionRecord:
    query_run_id: str
    tenant_id: str
    user_id: str
    auth_type: str
    credential_id: str
    identity_version: int | None
    roles_digest: str
    scopes_digest: str
    thread_id: str
    checkpoint_namespace: str
    workflow_version: str
    state_schema_version: str
    checkpoint_schema_version: str
    status: QueryExecutionStatus
    revision: int
    fencing_token: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    heartbeat_at: datetime | None
    next_node: str | None
    checkpoint_sequence: int
    attempt_sequence: int
    recovery_count: int
    safe_error_code: str | None
    created_at: datetime | None
    updated_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class QueryExecutionLease:
    query_run_id: str
    lease_owner: str
    lease_token: str = field(repr=False)
    fencing_token: int
    revision: int
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class QueryCheckpointRecord:
    checkpoint_id: str
    query_run_id: str
    sequence: int
    checkpoint_namespace: str
    workflow_version: str
    state_schema_version: str
    checkpoint_schema_version: str
    completed_node: str | None
    next_node: str | None
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)
    ciphertext_sha256: str
    size_bytes: int
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class QueryNodeAttemptRecord:
    attempt_id: str
    query_run_id: str
    sequence: int
    node_name: str
    effect_class: str
    status: QueryNodeAttemptStatus
    checkpoint_sequence_before: int
    checkpoint_sequence_after: int | None
    fencing_token: int
    lease_owner: str
    revision: int
    safe_error_code: str | None
    scheduled_at: datetime | None
    started_at: datetime | None
    checkpointed_at: datetime | None
    finished_at: datetime | None


__all__ = [
    "QUERY_CHECKPOINT_NAMESPACE",
    "QUERY_CHECKPOINT_SCHEMA_VERSION",
    "QUERY_EXECUTION_TERMINAL_STATUSES",
    "QUERY_STATE_SCHEMA_VERSION",
    "QueryCheckpointRecord",
    "QueryExecutionLease",
    "QueryExecutionRecord",
    "QueryExecutionStatus",
    "QueryNodeAttemptRecord",
    "QueryNodeAttemptStatus",
]
