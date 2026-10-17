"""Query 的诚实 safe-boundary 耐久执行器。

执行器在 handler 前持久化 ``started``，在 handler 返回后原子提交密文 state 与后继
游标。重启若看到 started 但 checkpoint 未前进，只会终止为 interrupted，绝不会
自动重放可能已产生 provider/citation/history 副作用的 active node。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from app.agent_system.agents.query.workflow import (
    QUERY_ENTRY_NODE,
    QUERY_NODE_REGISTRY,
    QUERY_WORKFLOW_VERSION,
    next_query_node,
    validate_query_workflow,
)
from app.domain.query_checkpoint import (
    QUERY_CHECKPOINT_NAMESPACE,
    QUERY_CHECKPOINT_SCHEMA_VERSION,
    QUERY_STATE_SCHEMA_VERSION,
    QueryCheckpointRecord,
    QueryExecutionLease,
    QueryExecutionRecord,
    QueryExecutionStatus,
    QueryNodeAttemptRecord,
    QueryNodeAttemptStatus,
)


class EncryptedCheckpointPayload(Protocol):
    nonce: bytes
    ciphertext: bytes
    ciphertext_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CheckpointCipherContext:
    tenant_id: str
    user_id: str
    query_run_id: str
    sequence: int
    completed_node: str | None
    next_node: str | None
    checkpoint_namespace: str
    workflow_version: str
    state_schema_version: str
    checkpoint_schema_version: str


class CheckpointCipher(Protocol):
    def encrypt_state(
        self,
        state: Mapping[str, Any],
        *,
        context: CheckpointCipherContext,
    ) -> EncryptedCheckpointPayload: ...

    def decrypt_state(
        self,
        encrypted: EncryptedCheckpointPayload,
        *,
        context: CheckpointCipherContext,
    ) -> dict[str, Any]: ...


class QueryCheckpointStore(Protocol):
    def initialize_execution(
        self,
        query_run_id: str,
        *,
        thread_id: str,
        checkpoint_namespace: str,
        workflow_version: str,
        state_schema_version: str,
        checkpoint_schema_version: str,
        next_node: str,
        encrypted: EncryptedCheckpointPayload,
        auth_type: str = "internal",
        credential_id: str = "internal",
        identity_version: int | None = None,
        roles_digest: str = "sha256:internal",
        scopes_digest: str = "sha256:internal",
    ) -> tuple[QueryExecutionRecord, bool]: ...

    def claim_execution(
        self,
        query_run_id: str,
        *,
        lease_owner: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> QueryExecutionLease: ...

    def get_execution(self, query_run_id: str) -> QueryExecutionRecord: ...

    def latest_checkpoint(self, query_run_id: str) -> QueryCheckpointRecord: ...

    def latest_attempt(self, query_run_id: str) -> QueryNodeAttemptRecord | None: ...

    def abandon_unstarted_attempt(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord: ...

    def interrupt_started_attempt(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryExecutionRecord: ...

    def schedule_attempt(
        self,
        lease: QueryExecutionLease,
        *,
        node_name: str,
        effect_class: str,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord: ...

    def mark_attempt_started(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord: ...

    def commit_attempt_checkpoint(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        completed_node: str,
        next_node: str | None,
        encrypted: EncryptedCheckpointPayload,
        now: datetime | None = None,
    ) -> tuple[QueryExecutionRecord, QueryCheckpointRecord, QueryNodeAttemptRecord]: ...

    def mark_attempt_failed(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        safe_error_code: str,
        now: datetime | None = None,
    ) -> QueryExecutionRecord: ...

    def abort_execution(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus,
        safe_error_code: str,
        now: datetime | None = None,
    ) -> QueryExecutionRecord: ...

    def finish_execution(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus = QueryExecutionStatus.COMPLETED,
        now: datetime | None = None,
    ) -> QueryExecutionRecord: ...


StoreScopeFactory = Callable[[], AbstractContextManager[QueryCheckpointStore]]
QueryNodeHandler = Callable[[dict[str, Any]], Mapping[str, Any]]
BoundaryGuard = Callable[[Mapping[str, Any]], None]


class DurableQueryExecutionError(RuntimeError):
    """不会携带 checkpoint 正文的可靠执行错误基类。"""


class QueryInterruptedDuringNode(DurableQueryExecutionError):
    """旧 worker 已进入 node，但没有提交下一安全边界。"""


class QueryCheckpointCompatibilityError(DurableQueryExecutionError):
    """checkpoint 的 schema/workflow/游标版本不兼容。"""


class QueryCheckpointLoadError(DurableQueryExecutionError):
    """checkpoint 密文损坏或身份认证失败。"""


class QueryExecutionDeadlineExceeded(DurableQueryExecutionError):
    """在进入下一节点前已超过绝对 deadline。"""

    def __init__(self, lease: QueryExecutionLease) -> None:
        super().__init__("query execution deadline exceeded at a safe boundary")
        self.lease = lease


class QueryExecutionCancelled(DurableQueryExecutionError):
    """边界 guard 检测到持久取消请求。"""

    def __init__(self, lease: QueryExecutionLease) -> None:
        super().__init__("query execution cancelled at a safe boundary")
        self.lease = lease


class QueryBoundaryCancellationRequested(DurableQueryExecutionError):
    """boundary guard 仅在读到持久取消事实时抛出的内部信号。"""


@dataclass(frozen=True, slots=True)
class DurableQueryExecutionResult:
    state: dict[str, Any] = field(repr=False)
    lease: QueryExecutionLease = field(repr=False)
    checkpoint_sequence: int
    recovered: bool


@dataclass(frozen=True, slots=True)
class DurableQueryPreflight:
    """cache/provider 前已认证且没有 active-node uncertainty 的安全边界。"""

    state: dict[str, Any] = field(repr=False)
    lease: QueryExecutionLease = field(repr=False)
    execution: QueryExecutionRecord = field(repr=False)
    checkpoint_sequence: int
    recovered: bool


class DurableQueryExecutor:
    """逐节点执行共享 registry，并把每个已完成节点提交为密文边界。"""

    def __init__(
        self,
        *,
        store_scope: StoreScopeFactory,
        cipher: CheckpointCipher,
        node_handlers: Mapping[str, QueryNodeHandler],
        lease_owner: str,
        lease_duration: timedelta = timedelta(minutes=5),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        validate_query_workflow(set(node_handlers))
        if not lease_owner or len(lease_owner) > 128:
            raise ValueError("durable query executor lease_owner is invalid")
        if lease_duration <= timedelta(0):
            raise ValueError("durable query executor lease_duration must be positive")
        self._store_scope = store_scope
        self._cipher = cipher
        self._handlers = dict(node_handlers)
        self._lease_owner = lease_owner
        self._lease_duration = lease_duration
        self._now = now or (lambda: datetime.now(UTC))

    def initialize(
        self,
        *,
        query_run_id: str,
        tenant_id: str,
        user_id: str,
        state: Mapping[str, Any],
        auth_type: str = "internal",
        credential_id: str = "internal",
        identity_version: int | None = None,
        roles_digest: str = "sha256:internal",
        scopes_digest: str = "sha256:internal",
    ) -> bool:
        """在返回 accepted 前调用；保证 0 node call 的 initial checkpoint 已提交。"""

        self._validate_state_identity(state, tenant_id, user_id, query_run_id)
        context = self._context(
            tenant_id=tenant_id,
            user_id=user_id,
            query_run_id=query_run_id,
            sequence=0,
            completed_node=None,
            next_node=QUERY_ENTRY_NODE,
        )
        encrypted = self._cipher.encrypt_state(state, context=context)
        with self._store_scope() as store:
            _execution, created = store.initialize_execution(
                query_run_id,
                thread_id=f"query:{query_run_id}",
                checkpoint_namespace=QUERY_CHECKPOINT_NAMESPACE,
                workflow_version=QUERY_WORKFLOW_VERSION,
                state_schema_version=QUERY_STATE_SCHEMA_VERSION,
                checkpoint_schema_version=QUERY_CHECKPOINT_SCHEMA_VERSION,
                next_node=QUERY_ENTRY_NODE,
                encrypted=encrypted,
                auth_type=auth_type,
                credential_id=credential_id,
                identity_version=identity_version,
                roles_digest=roles_digest,
                scopes_digest=scopes_digest,
            )
        return created

    def preflight(
        self,
        *,
        query_run_id: str,
        tenant_id: str,
        user_id: str,
        lease: QueryExecutionLease | None = None,
    ) -> DurableQueryPreflight:
        """在 cache/singleflight/provider 前认证 checkpoint 并收敛 active attempt。"""

        lease = lease or self.claim(query_run_id)
        if lease.query_run_id != query_run_id:
            raise QueryCheckpointCompatibilityError("query execution lease belongs to another run")
        with self._store_scope() as store:
            execution = store.get_execution(query_run_id)
            checkpoint = store.latest_checkpoint(query_run_id)
            latest_attempt = store.latest_attempt(query_run_id)

        recovered = execution.recovery_count > 0 or checkpoint.sequence > 0
        self._validate_persisted_versions(execution, checkpoint, lease)
        if latest_attempt is not None and latest_attempt.checkpoint_sequence_after is None:
            if latest_attempt.status == QueryNodeAttemptStatus.STARTED:
                with self._store_scope() as store:
                    store.interrupt_started_attempt(
                        lease,
                        latest_attempt.attempt_id,
                        now=self._now(),
                    )
                raise QueryInterruptedDuringNode(
                    "query interrupted during an active node; automatic replay is forbidden"
                )
            if latest_attempt.status == QueryNodeAttemptStatus.SCHEDULED:
                with self._store_scope() as store:
                    store.abandon_unstarted_attempt(
                        lease,
                        latest_attempt.attempt_id,
                        now=self._now(),
                    )

        try:
            state = self._cipher.decrypt_state(
                checkpoint,
                context=self._checkpoint_context(execution, checkpoint),
            )
        except Exception as exc:
            self._abort(
                lease,
                status=QueryExecutionStatus.FAILED,
                safe_error_code="QUERY_CHECKPOINT_CORRUPT",
            )
            raise QueryCheckpointLoadError("query checkpoint could not be authenticated") from exc
        self._validate_state_identity(state, tenant_id, user_id, query_run_id)
        expected_next = next_query_node(checkpoint.completed_node, state)
        if expected_next != checkpoint.next_node or expected_next != execution.next_node:
            self._abort(
                lease,
                status=QueryExecutionStatus.FAILED,
                safe_error_code="QUERY_CHECKPOINT_ROUTE_MISMATCH",
            )
            raise QueryCheckpointCompatibilityError(
                "query checkpoint route does not match the workflow contract"
            )
        self._check_deadline(state, lease)
        return DurableQueryPreflight(
            state=state,
            lease=lease,
            execution=execution,
            checkpoint_sequence=checkpoint.sequence,
            recovered=recovered,
        )

    def execute(
        self,
        *,
        query_run_id: str,
        tenant_id: str,
        user_id: str,
        boundary_guard: BoundaryGuard | None = None,
        lease: QueryExecutionLease | None = None,
        preflight: DurableQueryPreflight | None = None,
    ) -> DurableQueryExecutionResult:
        """从最近已提交边界继续；active node 残留会 fail-closed 且零重放。"""

        prepared = preflight or self.preflight(
            query_run_id=query_run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            lease=lease,
        )
        if prepared.execution.query_run_id != query_run_id:
            raise QueryCheckpointCompatibilityError("query preflight belongs to another run")
        if lease is not None and prepared.lease != lease:
            raise QueryCheckpointCompatibilityError("query preflight lease changed")
        lease = prepared.lease
        execution = prepared.execution
        state = dict(prepared.state)
        recovered = prepared.recovered

        while execution.next_node is not None:
            self._check_deadline(state, lease)
            if boundary_guard is not None:
                try:
                    boundary_guard(state)
                except QueryBoundaryCancellationRequested as exc:
                    raise QueryExecutionCancelled(lease) from exc
            node_name = execution.next_node
            definition = QUERY_NODE_REGISTRY[node_name]
            with self._store_scope() as store:
                attempt = store.schedule_attempt(
                    lease,
                    node_name=node_name,
                    effect_class=definition.effect_class,
                    now=self._now(),
                )
            with self._store_scope() as store:
                store.mark_attempt_started(lease, attempt.attempt_id, now=self._now())

            try:
                patch = self._handlers[node_name](state)
                if not isinstance(patch, Mapping):
                    raise TypeError("query node must return a state mapping")
                state.update(dict(patch))
                next_node = next_query_node(node_name, state)
                next_sequence = execution.checkpoint_sequence + 1
                encrypted = self._cipher.encrypt_state(
                    state,
                    context=self._context(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        query_run_id=query_run_id,
                        sequence=next_sequence,
                        completed_node=node_name,
                        next_node=next_node,
                    ),
                )
            except Exception:
                with self._store_scope() as store:
                    store.mark_attempt_failed(
                        lease,
                        attempt.attempt_id,
                        safe_error_code="QUERY_NODE_EXECUTION_FAILED",
                        now=self._now(),
                    )
                raise

            with self._store_scope() as store:
                execution, checkpoint, _attempt = store.commit_attempt_checkpoint(
                    lease,
                    attempt.attempt_id,
                    completed_node=node_name,
                    next_node=next_node,
                    encrypted=encrypted,
                    now=self._now(),
                )

        return DurableQueryExecutionResult(
            state=state,
            lease=lease,
            checkpoint_sequence=execution.checkpoint_sequence,
            recovered=recovered,
        )

    def finish_completed(self, result: DurableQueryExecutionResult) -> QueryExecutionRecord:
        """QueryRun/result/terminal event 提交成功后释放同一 fence。"""

        with self._store_scope() as store:
            return store.finish_execution(
                result.lease,
                status=QueryExecutionStatus.COMPLETED,
                now=self._now(),
            )

    def finish_stopped(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus,
        safe_error_code: str,
    ) -> QueryExecutionRecord:
        if status not in {
            QueryExecutionStatus.CANCELLED,
            QueryExecutionStatus.TIMED_OUT,
        }:
            raise ValueError("durable query stop status is invalid")
        with self._store_scope() as store:
            return store.abort_execution(
                lease,
                status=status,
                safe_error_code=safe_error_code,
                now=self._now(),
            )

    def claim(self, query_run_id: str) -> QueryExecutionLease:
        """在任何 cache/provider 决策前取得唯一 execution fence。"""

        with self._store_scope() as store:
            return store.claim_execution(
                query_run_id,
                lease_owner=self._lease_owner,
                lease_duration=self._lease_duration,
                now=self._now(),
            )

    def _validate_persisted_versions(
        self,
        execution: QueryExecutionRecord,
        checkpoint: QueryCheckpointRecord,
        lease: QueryExecutionLease,
    ) -> None:
        expected = (
            QUERY_CHECKPOINT_NAMESPACE,
            QUERY_WORKFLOW_VERSION,
            QUERY_STATE_SCHEMA_VERSION,
            QUERY_CHECKPOINT_SCHEMA_VERSION,
        )
        actual_execution = (
            execution.checkpoint_namespace,
            execution.workflow_version,
            execution.state_schema_version,
            execution.checkpoint_schema_version,
        )
        actual_checkpoint = (
            checkpoint.checkpoint_namespace,
            checkpoint.workflow_version,
            checkpoint.state_schema_version,
            checkpoint.checkpoint_schema_version,
        )
        if actual_execution != expected or actual_checkpoint != expected:
            self._abort(
                lease,
                status=QueryExecutionStatus.FAILED,
                safe_error_code="QUERY_CHECKPOINT_VERSION_UNSUPPORTED",
            )
            raise QueryCheckpointCompatibilityError("query checkpoint version is not supported")

    def _abort(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus,
        safe_error_code: str,
    ) -> None:
        with self._store_scope() as store:
            store.abort_execution(
                lease,
                status=status,
                safe_error_code=safe_error_code,
                now=self._now(),
            )

    def _check_deadline(
        self,
        state: Mapping[str, Any],
        lease: QueryExecutionLease,
    ) -> None:
        raw = state.get("deadline_at")
        if raw in {None, ""}:
            return
        if isinstance(raw, datetime):
            deadline = raw
        else:
            try:
                deadline = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError as exc:
                raise QueryCheckpointCompatibilityError(
                    "query checkpoint deadline is invalid"
                ) from exc
        if deadline.tzinfo is None:
            raise QueryCheckpointCompatibilityError(
                "query checkpoint deadline must include timezone"
            )
        if deadline.astimezone(UTC) <= self._now().astimezone(UTC):
            raise QueryExecutionDeadlineExceeded(lease)

    @staticmethod
    def _validate_state_identity(
        state: Mapping[str, Any],
        tenant_id: str,
        user_id: str,
        query_run_id: str,
    ) -> None:
        if (
            state.get("tenant_id") != tenant_id
            or state.get("user_id") != user_id
            or state.get("query_run_id") != query_run_id
        ):
            raise QueryCheckpointCompatibilityError(
                "query checkpoint state identity does not match the execution"
            )

    @staticmethod
    def _context(
        *,
        tenant_id: str,
        user_id: str,
        query_run_id: str,
        sequence: int,
        completed_node: str | None,
        next_node: str | None,
    ) -> CheckpointCipherContext:
        return CheckpointCipherContext(
            tenant_id=tenant_id,
            user_id=user_id,
            query_run_id=query_run_id,
            sequence=sequence,
            completed_node=completed_node,
            next_node=next_node,
            checkpoint_namespace=QUERY_CHECKPOINT_NAMESPACE,
            workflow_version=QUERY_WORKFLOW_VERSION,
            state_schema_version=QUERY_STATE_SCHEMA_VERSION,
            checkpoint_schema_version=QUERY_CHECKPOINT_SCHEMA_VERSION,
        )

    @staticmethod
    def _checkpoint_context(
        execution: QueryExecutionRecord,
        checkpoint: QueryCheckpointRecord,
    ) -> CheckpointCipherContext:
        return CheckpointCipherContext(
            tenant_id=execution.tenant_id,
            user_id=execution.user_id,
            query_run_id=execution.query_run_id,
            sequence=checkpoint.sequence,
            completed_node=checkpoint.completed_node,
            next_node=checkpoint.next_node,
            checkpoint_namespace=checkpoint.checkpoint_namespace,
            workflow_version=checkpoint.workflow_version,
            state_schema_version=checkpoint.state_schema_version,
            checkpoint_schema_version=checkpoint.checkpoint_schema_version,
        )


__all__ = [
    "BoundaryGuard",
    "CheckpointCipher",
    "CheckpointCipherContext",
    "DurableQueryExecutionError",
    "DurableQueryExecutionResult",
    "DurableQueryPreflight",
    "DurableQueryExecutor",
    "QueryCheckpointCompatibilityError",
    "QueryCheckpointLoadError",
    "QueryBoundaryCancellationRequested",
    "QueryExecutionCancelled",
    "QueryExecutionDeadlineExceeded",
    "QueryInterruptedDuringNode",
]
