"""可靠 Query 的耐久组合与跨进程恢复适配器。

本模块把 HTTP 之外的密钥解析、身份绑定、SQL checkpoint store 组合和恢复扫描
集中在严格 ``app.infrastructure`` 边界。它只保证已提交节点边界可恢复；看到
``started`` 且无后继 checkpoint 的节点时，执行器会终止而不会重放副作用。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4
from threading import Event as ThreadEvent, RLock, Thread

from sqlalchemy import and_, or_, select

from app.agent_system.agents.query.durable_executor import DurableQueryExecutor
from app.agent_system.agents.query.workflow import (
    QUERY_ENTRY_NODE,
    QUERY_WORKFLOW_VERSION,
)
from app.domain.query_checkpoint import (
    QUERY_CHECKPOINT_NAMESPACE,
    QUERY_CHECKPOINT_SCHEMA_VERSION,
    QUERY_EXECUTION_TERMINAL_STATUSES,
    QUERY_STATE_SCHEMA_VERSION,
    QueryExecutionRecord,
    QueryExecutionStatus,
)
from app.domain.query_runtime import QueryRun, QueryRunStatus, RUN_TERMINAL_STATUSES, canonical_hash
from app.foundation.query_checkpoint_crypto import (
    QueryCheckpointCipher,
    QueryCheckpointEncryptionContext,
    QueryCheckpointKeyInvalid,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import (
    ApiCredentialModel,
    TenantMembershipModel,
    TenantModel,
    UserModel,
)
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.infra.persistence.sql.query_runtime_models import (
    QueryRunExecutionModel,
    QueryRunModel,
)
from app.infra.persistence.sql.query_runtime_repository import (
    QueryCheckpointRepository,
    QueryRuntimeRepository,
    StaleQueryRevision,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.security.principal import Principal
from app.shared.runtime.logger import logger


class SecretValue(Protocol):
    """最小 SecretStr port，避免基础设施读取全局 settings。"""

    def get_secret_value(self) -> str: ...


class QueryDurabilityConfigurationError(RuntimeError):
    """可靠 Query 的独立 checkpoint key 缺失或无效。"""


class QueryResumeIdentityError(RuntimeError):
    """恢复身份已撤销或与接收时绑定不一致。"""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class QueryRecoveryStopped(RuntimeError):
    """恢复 worker 在取得新 lease 前收到进程停止 fence。"""


@dataclass(frozen=True, slots=True)
class QueryAuthBinding:
    """可恢复但不含 credential secret 的最小身份引用。"""

    auth_type: str
    credential_id: str
    identity_version: int
    roles_digest: str
    scopes_digest: str


@dataclass(frozen=True, slots=True)
class QueryRecoveryCandidate:
    query_run_id: str
    tenant_id: str
    user_id: str
    execution_present: bool
    execution_status: QueryExecutionStatus | None
    execution_safe_error_code: str | None
    run_terminal: bool


QueryNodeHandler = Callable[[dict[str, Any]], Mapping[str, Any]]
ExecuteRecoveredQuery = Callable[[str, Principal, str | None], None]


class DurableQueryRecoveryController:
    """把“停止领取”与 execution claim 串行化，并在关闭时主动 fence。"""

    def __init__(self, *, owner_id: str | None = None) -> None:
        self.owner_id = owner_id or f"query-recovery:{uuid4().hex}"
        self._claim_lock = RLock()
        self._stop_requested = ThreadEvent()
        self._stopping = False

    @contextmanager
    def claim_gate(self):
        """调用方必须在上下文内完成数据库 claim 提交。"""

        if self._stop_requested.is_set():
            raise QueryRecoveryStopped("query recovery is stopping")
        with self._claim_lock:
            if self._stop_requested.is_set():
                raise QueryRecoveryStopped("query recovery is stopping")
            if self._stopping:
                raise QueryRecoveryStopped("query recovery is stopping")
            yield

    def stop_and_interrupt(self) -> dict[str, int]:
        """禁止新 claim，并把本进程已领取运行收敛为诚实 interrupted 终态。"""

        self._stop_requested.set()
        with self._claim_lock:
            self._stopping = True
        stats = {"scanned": 0, "interrupted": 0, "raced": 0, "failed": 0}
        with session_scope() as session:
            owned = session.execute(
                select(
                    QueryRunExecutionModel.query_run_id,
                    QueryRunExecutionModel.tenant_id,
                    QueryRunExecutionModel.user_id,
                ).where(
                    QueryRunExecutionModel.status == QueryExecutionStatus.RUNNING.value,
                    QueryRunExecutionModel.lease_owner == self.owner_id,
                )
            ).all()
        stats["scanned"] = len(owned)
        for query_run_id, tenant_id, user_id in owned:
            try:
                with session_scope() as session:
                    runtime = QueryRuntimeRepository(
                        session,
                        tenant_id=str(tenant_id),
                        user_id=str(user_id),
                    )
                    run = runtime.get_run(str(query_run_id))
                    checkpoint_repository = QueryCheckpointRepository(
                        session,
                        tenant_id=str(tenant_id),
                        user_id=str(user_id),
                    )
                    checkpoint_repository.interrupt_owned_execution(
                        run.query_run_id,
                        lease_owner=self.owner_id,
                        safe_error_code="QUERY_INTERRUPTED_DURING_SHUTDOWN",
                    )
                    if run.status not in RUN_TERMINAL_STATUSES:
                        if run.status == QueryRunStatus.CANCEL_REQUESTED:
                            runtime.finalize_cancel(
                                run.query_run_id,
                                expected_revision=run.revision,
                            )
                            response_status = 409
                        else:
                            runtime.fail_run(
                                run.query_run_id,
                                expected_revision=run.revision,
                                error_code="QUERY_INTERRUPTED_DURING_SHUTDOWN",
                                safe_message=(
                                    "查询在服务关闭时中断，未自动重放 active node；请安全重试。"
                                ),
                                retryable=True,
                                error_detail_ref=f"trace:{run.trace_id}",
                            )
                            response_status = 500
                        QueryReliabilityRepository(
                            session,
                            tenant_id=str(tenant_id),
                        ).complete_idempotency(run.query_run_id, response_status)
                    checkpoint_repository.reconcile_terminal_run(str(query_run_id))
                stats["interrupted"] += 1
            except (ResourceConflictError, ResourceNotFoundError, StaleQueryRevision):
                stats["raced"] += 1
            except Exception:
                stats["failed"] += 1
                logger.error(
                    "query recovery shutdown fence failed: code={} query_run_id={} stage={}",
                    "QUERY_RECOVERY_SHUTDOWN_FENCE_FAILED",
                    str(query_run_id),
                    "terminalize",
                )
        return stats

    def stop_and_interrupt_bounded(self, *, timeout_seconds: float) -> dict[str, int]:
        """有界等待 shutdown fence；DB 不可用时留下稳定 residual 供 hard-stop。"""

        if timeout_seconds <= 0:
            raise ValueError("query recovery shutdown timeout must be positive")
        self._stop_requested.set()
        completed = ThreadEvent()
        holder: dict[str, dict[str, int]] = {}

        def fence_owned() -> None:
            try:
                holder["report"] = self.stop_and_interrupt()
            except Exception:
                holder["report"] = {
                    "scanned": 0,
                    "interrupted": 0,
                    "raced": 0,
                    "failed": 1,
                }
            finally:
                completed.set()

        worker = Thread(
            target=fence_owned,
            name="query-checkpoint-shutdown-fence",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=timeout_seconds)
        if not completed.is_set():
            return {
                "scanned": 0,
                "interrupted": 0,
                "raced": 0,
                "failed": 0,
                "timed_out": 1,
            }
        return {**holder["report"], "timed_out": 0}


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _identity_digest(values: tuple[str, ...]) -> str:
    return canonical_hash(sorted(values))


def build_query_checkpoint_cipher(
    configured: SecretValue | str | None,
) -> QueryCheckpointCipher:
    """显式解析 purpose-separated key；缺失或非法一律 fail-closed。"""

    if configured is None:
        raise QueryDurabilityConfigurationError("query checkpoint encryption key is unavailable")
    raw = configured if isinstance(configured, str) else configured.get_secret_value()
    if not raw.strip():
        raise QueryDurabilityConfigurationError("query checkpoint encryption key is unavailable")
    try:
        return QueryCheckpointCipher(raw)
    except QueryCheckpointKeyInvalid as exc:
        raise QueryDurabilityConfigurationError(
            "query checkpoint encryption key is invalid"
        ) from exc


def bind_query_execution_identity(session, principal: Principal) -> QueryAuthBinding:
    """接收时从控制库绑定身份；不信任请求 Principal 中的权限副本。"""

    tenant = session.get(TenantModel, principal.tenant_id)
    user = session.get(UserModel, principal.user_id)
    membership = session.scalar(
        select(TenantMembershipModel).where(
            TenantMembershipModel.tenant_id == principal.tenant_id,
            TenantMembershipModel.user_id == principal.user_id,
            TenantMembershipModel.status == "active",
        )
    )
    if (
        tenant is None
        or tenant.status != "active"
        or user is None
        or user.status != "active"
        or membership is None
    ):
        raise QueryResumeIdentityError(
            "QUERY_EXECUTION_IDENTITY_REVOKED",
            "当前身份或租户成员关系不可用于可靠查询。",
        )

    if principal.auth_type == "api_key":
        credential = session.get(ApiCredentialModel, principal.credential_id)
        expires_at = _aware(credential.expires_at) if credential is not None else None
        if (
            credential is None
            or credential.status != "active"
            or credential.tenant_id != principal.tenant_id
            or credential.user_id != principal.user_id
            or (expires_at is not None and expires_at <= datetime.now(UTC))
        ):
            raise QueryResumeIdentityError(
                "QUERY_EXECUTION_CREDENTIAL_REVOKED",
                "当前 API credential 不可用于可靠查询。",
            )
        roles = ("ServiceAccount",)
        scopes = tuple(str(item) for item in (credential.scopes or ()))
    elif principal.auth_type in {"bearer", "password", "refresh"}:
        roles = (str(membership.role),)
        # 当前 bearer/password/refresh 认证没有独立 scope 源；非空 scope
        # 不能在不持久化 token 的前提下安全重建，因此接收时即拒绝。
        scopes = ()
    else:
        raise QueryResumeIdentityError(
            "QUERY_EXECUTION_AUTH_TYPE_UNSUPPORTED",
            "当前认证类型不支持跨进程可靠查询。",
        )
    if tuple(principal.roles) != roles or tuple(principal.scopes) != scopes:
        raise QueryResumeIdentityError(
            "QUERY_EXECUTION_AUTHORITY_CHANGED",
            "当前角色或授权范围已变化。",
        )
    return QueryAuthBinding(
        auth_type=principal.auth_type,
        credential_id=principal.credential_id,
        identity_version=int(user.token_version),
        roles_digest=_identity_digest(roles),
        scopes_digest=_identity_digest(scopes),
    )


def recover_query_principal(execution: QueryExecutionRecord) -> Principal:
    """恢复时只从 credential identity 重建当前 Principal，不信 checkpoint 权限。"""

    with session_scope() as session:
        tenant = session.get(TenantModel, execution.tenant_id)
        user = session.get(UserModel, execution.user_id)
        membership = session.scalar(
            select(TenantMembershipModel).where(
                TenantMembershipModel.tenant_id == execution.tenant_id,
                TenantMembershipModel.user_id == execution.user_id,
                TenantMembershipModel.status == "active",
            )
        )
        if (
            tenant is None
            or tenant.status != "active"
            or user is None
            or user.status != "active"
            or membership is None
            or execution.identity_version != int(user.token_version)
        ):
            raise QueryResumeIdentityError(
                "QUERY_RESUME_IDENTITY_REVOKED",
                "查询恢复时身份或成员关系已失效。",
            )
        if execution.auth_type == "api_key":
            credential = session.get(ApiCredentialModel, execution.credential_id)
            expires_at = _aware(credential.expires_at) if credential is not None else None
            if (
                credential is None
                or credential.status != "active"
                or credential.tenant_id != execution.tenant_id
                or credential.user_id != execution.user_id
                or (expires_at is not None and expires_at <= datetime.now(UTC))
            ):
                raise QueryResumeIdentityError(
                    "QUERY_RESUME_CREDENTIAL_REVOKED",
                    "查询恢复时 API credential 已失效。",
                )
            roles = ("ServiceAccount",)
            scopes = tuple(str(item) for item in (credential.scopes or ()))
        elif execution.auth_type in {"bearer", "password", "refresh"}:
            roles = (str(membership.role),)
            scopes = ()
        else:
            raise QueryResumeIdentityError(
                "QUERY_RESUME_AUTH_TYPE_UNSUPPORTED",
                "查询恢复时认证类型不受支持。",
            )
    if (
        _identity_digest(roles) != execution.roles_digest
        or _identity_digest(scopes) != execution.scopes_digest
    ):
        raise QueryResumeIdentityError(
            "QUERY_RESUME_AUTHORITY_CHANGED",
            "查询恢复时角色或授权范围已变化。",
        )
    return Principal(
        user_id=execution.user_id,
        tenant_id=execution.tenant_id,
        roles=roles,
        scopes=scopes,
        auth_type=execution.auth_type,
        credential_id=execution.credential_id,
        request_id=f"recovery_{uuid4().hex}",
    )


def initialize_query_execution(
    session,
    *,
    principal: Principal,
    run: QueryRun,
    state: dict[str, Any],
    cipher: QueryCheckpointCipher,
    auth_binding: QueryAuthBinding,
) -> bool:
    """与 QueryRun 接收事务一起写入 sequence=0 密文 checkpoint。"""

    context = QueryCheckpointEncryptionContext(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        query_run_id=run.query_run_id,
        sequence=0,
        completed_node=None,
        next_node=QUERY_ENTRY_NODE,
        checkpoint_namespace=QUERY_CHECKPOINT_NAMESPACE,
        workflow_version=QUERY_WORKFLOW_VERSION,
        state_schema_version=QUERY_STATE_SCHEMA_VERSION,
        checkpoint_schema_version=QUERY_CHECKPOINT_SCHEMA_VERSION,
    )
    encrypted = cipher.encrypt_state(state, context=context)
    _execution, created = QueryCheckpointRepository(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
    ).initialize_execution(
        run.query_run_id,
        thread_id=f"query:{run.query_run_id}",
        checkpoint_namespace=QUERY_CHECKPOINT_NAMESPACE,
        workflow_version=QUERY_WORKFLOW_VERSION,
        state_schema_version=QUERY_STATE_SCHEMA_VERSION,
        checkpoint_schema_version=QUERY_CHECKPOINT_SCHEMA_VERSION,
        next_node=QUERY_ENTRY_NODE,
        encrypted=encrypted,
        auth_type=auth_binding.auth_type,
        credential_id=auth_binding.credential_id,
        identity_version=auth_binding.identity_version,
        roles_digest=auth_binding.roles_digest,
        scopes_digest=auth_binding.scopes_digest,
    )
    return created


def build_durable_query_executor(
    *,
    principal: Principal,
    cipher: QueryCheckpointCipher,
    node_handlers: Mapping[str, QueryNodeHandler],
    lease_owner: str | None = None,
) -> DurableQueryExecutor:
    """组合 session-scoped store；handler registry 由外层显式注入。"""

    @contextmanager
    def store_scope():
        with session_scope() as session:
            yield QueryCheckpointRepository(
                session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

    return DurableQueryExecutor(
        store_scope=store_scope,
        cipher=cipher,
        node_handlers=node_handlers,
        lease_owner=lease_owner or f"query-worker:{uuid4().hex}",
        lease_duration=timedelta(minutes=5),
    )


def _recovery_candidates(
    *,
    now: datetime,
    limit: int,
) -> tuple[QueryRecoveryCandidate, ...]:
    nonterminal_runs = tuple(
        status.value for status in QueryRunStatus if status not in RUN_TERMINAL_STATUSES
    )
    terminal_runs = tuple(status.value for status in RUN_TERMINAL_STATUSES)
    terminal_executions = tuple(status.value for status in QUERY_EXECUTION_TERMINAL_STATUSES)
    terminal_execution_matches_run = or_(
        and_(
            QueryRunModel.status == QueryRunStatus.COMPLETED.value,
            QueryRunExecutionModel.status == QueryExecutionStatus.COMPLETED.value,
        ),
        and_(
            QueryRunModel.status == QueryRunStatus.FAILED.value,
            or_(
                QueryRunExecutionModel.status == QueryExecutionStatus.FAILED.value,
                and_(
                    QueryRunModel.error_code.in_(
                        (
                            "QUERY_INTERRUPTED_DURING_NODE",
                            "QUERY_INTERRUPTED_DURING_SHUTDOWN",
                        )
                    ),
                    QueryRunExecutionModel.status == QueryExecutionStatus.INTERRUPTED.value,
                ),
            ),
        ),
        and_(
            QueryRunModel.status == QueryRunStatus.TIMED_OUT.value,
            QueryRunExecutionModel.status == QueryExecutionStatus.TIMED_OUT.value,
        ),
        and_(
            QueryRunModel.status == QueryRunStatus.CANCELLED.value,
            QueryRunExecutionModel.status == QueryExecutionStatus.CANCELLED.value,
        ),
    )
    with session_scope() as session:
        rows = session.execute(
            select(
                QueryRunModel.id,
                QueryRunModel.tenant_id,
                QueryRunModel.user_id,
                QueryRunModel.status,
                QueryRunExecutionModel.query_run_id,
                QueryRunExecutionModel.status,
                QueryRunExecutionModel.safe_error_code,
            )
            .outerjoin(
                QueryRunExecutionModel,
                QueryRunExecutionModel.query_run_id == QueryRunModel.id,
            )
            .where(
                or_(
                    and_(
                        QueryRunModel.status.in_(terminal_runs),
                        QueryRunExecutionModel.query_run_id.is_not(None),
                        or_(
                            QueryRunExecutionModel.lease_owner.is_not(None),
                            QueryRunExecutionModel.next_node.is_not(None),
                            ~terminal_execution_matches_run,
                        ),
                    ),
                    and_(
                        QueryRunModel.status.in_(nonterminal_runs),
                        or_(
                            QueryRunExecutionModel.query_run_id.is_(None),
                            QueryRunExecutionModel.status == QueryExecutionStatus.READY.value,
                            QueryRunExecutionModel.status.in_(terminal_executions),
                            and_(
                                QueryRunExecutionModel.status == QueryExecutionStatus.RUNNING.value,
                                or_(
                                    QueryRunExecutionModel.lease_expires_at.is_(None),
                                    QueryRunExecutionModel.lease_expires_at <= now,
                                ),
                            ),
                        ),
                    ),
                )
            )
            .order_by(QueryRunModel.created_at, QueryRunModel.id)
            .limit(limit)
        ).all()
    return tuple(
        QueryRecoveryCandidate(
            query_run_id=str(query_run_id),
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            execution_present=execution_id is not None,
            execution_status=(
                QueryExecutionStatus(str(execution_status))
                if execution_status is not None
                else None
            ),
            execution_safe_error_code=(
                str(execution_safe_error_code) if execution_safe_error_code is not None else None
            ),
            run_terminal=str(run_status) in terminal_runs,
        )
        for (
            query_run_id,
            tenant_id,
            user_id,
            run_status,
            execution_id,
            execution_status,
            execution_safe_error_code,
        ) in rows
    )


def _terminalize_unrecoverable(
    candidate: QueryRecoveryCandidate,
    *,
    code: str,
    safe_message: str,
    response_status: int,
) -> None:
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=candidate.tenant_id,
            user_id=candidate.user_id,
        )
        run = runtime.get_run(candidate.query_run_id)
        if run.status not in RUN_TERMINAL_STATUSES:
            terminal_response_status = response_status
            if run.status == QueryRunStatus.CANCEL_REQUESTED:
                runtime.finalize_cancel(
                    candidate.query_run_id,
                    expected_revision=run.revision,
                )
                terminal_response_status = 409
            else:
                runtime.fail_run(
                    candidate.query_run_id,
                    expected_revision=run.revision,
                    error_code=code,
                    safe_message=safe_message,
                    retryable=True,
                    error_detail_ref=f"trace:{run.trace_id}",
                )
            QueryReliabilityRepository(
                session,
                tenant_id=candidate.tenant_id,
            ).complete_idempotency(candidate.query_run_id, terminal_response_status)
        if candidate.execution_present:
            QueryCheckpointRepository(
                session,
                tenant_id=candidate.tenant_id,
                user_id=candidate.user_id,
            ).reconcile_terminal_run(candidate.query_run_id)


def _converge_terminal_execution(candidate: QueryRecoveryCandidate) -> None:
    """收敛 execution 已终态、QueryRun 仍非终态的反向 crash seam。"""

    execution_status = candidate.execution_status
    if execution_status is None or execution_status not in QUERY_EXECUTION_TERMINAL_STATUSES:
        raise ValueError("query recovery candidate is not terminal")
    with session_scope() as session:
        runtime = QueryRuntimeRepository(
            session,
            tenant_id=candidate.tenant_id,
            user_id=candidate.user_id,
        )
        run = runtime.get_run(candidate.query_run_id)
        if run.status in RUN_TERMINAL_STATUSES:
            QueryCheckpointRepository(
                session,
                tenant_id=candidate.tenant_id,
                user_id=candidate.user_id,
            ).reconcile_terminal_run(candidate.query_run_id)
            return

        response_status = 500
        if (
            execution_status == QueryExecutionStatus.CANCELLED
            and run.status == QueryRunStatus.CANCEL_REQUESTED
        ):
            runtime.finalize_cancel(
                candidate.query_run_id,
                expected_revision=run.revision,
            )
            response_status = 409
        else:
            timed_out = execution_status == QueryExecutionStatus.TIMED_OUT
            if timed_out:
                error_code = candidate.execution_safe_error_code or "QUERY_TIMED_OUT"
                safe_message = "查询执行已超过截止时间，请安全重试。"
                response_status = 504
            elif execution_status == QueryExecutionStatus.INTERRUPTED:
                error_code = "QUERY_INTERRUPTED_DURING_NODE"
                safe_message = "查询在可能产生外部副作用的节点中断，未自动重放；请安全重试。"
            elif execution_status == QueryExecutionStatus.FAILED:
                error_code = candidate.execution_safe_error_code or "QUERY_EXECUTION_FAILED"
                safe_message = "查询执行失败，请安全重试。"
            else:
                error_code = "QUERY_EXECUTION_TERMINAL_INCONSISTENT"
                safe_message = "查询终态提交不完整，请创建安全重试。"
            runtime.fail_run(
                candidate.query_run_id,
                expected_revision=run.revision,
                error_code=error_code,
                safe_message=safe_message,
                retryable=True,
                error_detail_ref=f"trace:{run.trace_id}",
                timed_out=timed_out,
            )
        QueryReliabilityRepository(
            session,
            tenant_id=candidate.tenant_id,
        ).complete_idempotency(candidate.query_run_id, response_status)
        QueryCheckpointRepository(
            session,
            tenant_id=candidate.tenant_id,
            user_id=candidate.user_id,
        ).reconcile_terminal_run(candidate.query_run_id)


def recover_durable_query_executions(
    execute: ExecuteRecoveredQuery,
    *,
    now: datetime | None = None,
    limit: int = 1000,
) -> dict[str, int]:
    """扫描 READY/过期 RUNNING，并以当前身份恢复；不盲重放 active node。"""

    if not 1 <= limit <= 10_000:
        raise ValueError("query checkpoint recovery limit must be in 1..10000")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("query checkpoint recovery timestamp must include timezone")
    candidates = _recovery_candidates(now=current, limit=limit)
    stats = {
        "scanned": len(candidates),
        "resumed": 0,
        "reconciled": 0,
        "rejected": 0,
        "raced": 0,
        "failed": 0,
    }
    for candidate in candidates:
        try:
            if candidate.run_terminal:
                with session_scope() as session:
                    QueryCheckpointRepository(
                        session,
                        tenant_id=candidate.tenant_id,
                        user_id=candidate.user_id,
                    ).reconcile_terminal_run(candidate.query_run_id)
                stats["reconciled"] += 1
                continue
            if not candidate.execution_present:
                _terminalize_unrecoverable(
                    candidate,
                    code="QUERY_CHECKPOINT_NOT_INITIALIZED",
                    safe_message="查询缺少可验证的初始 checkpoint，请安全重试。",
                    response_status=500,
                )
                stats["rejected"] += 1
                continue
            if candidate.execution_status in QUERY_EXECUTION_TERMINAL_STATUSES:
                _converge_terminal_execution(candidate)
                stats["reconciled"] += 1
                continue
            with session_scope() as session:
                execution = QueryCheckpointRepository(
                    session,
                    tenant_id=candidate.tenant_id,
                    user_id=candidate.user_id,
                ).get_execution(candidate.query_run_id)
            try:
                principal = recover_query_principal(execution)
            except QueryResumeIdentityError as exc:
                _terminalize_unrecoverable(
                    candidate,
                    code=exc.code,
                    safe_message=exc.safe_message,
                    response_status=403,
                )
                stats["rejected"] += 1
                continue
            execute(candidate.query_run_id, principal, None)
            stats["resumed"] += 1
        except QueryRecoveryStopped:
            stats["raced"] += 1
            break
        except (ResourceConflictError, ResourceNotFoundError, StaleQueryRevision):
            stats["raced"] += 1
        except Exception:
            stats["failed"] += 1
            # 不记录 traceback：底层异常可能携带 provider/state/credential 正文。
            logger.error(
                "query checkpoint recovery failed: code={} query_run_id={} stage={}",
                "QUERY_CHECKPOINT_RECOVERY_FAILED",
                candidate.query_run_id,
                "candidate",
            )
    return stats


async def run_durable_query_recovery_loop(
    stop: asyncio.Event,
    *,
    execute: ExecuteRecoveredQuery,
    interval_seconds: float,
    batch_limit: int = 1,
) -> None:
    """周期恢复安全边界；每轮最多执行一个运行，避免批量拖住停机。"""

    if interval_seconds <= 0:
        raise ValueError("query checkpoint recovery interval must be positive")
    if not 1 <= batch_limit <= 100:
        raise ValueError("query checkpoint recovery batch_limit must be in 1..100")
    while not stop.is_set():
        completed = ThreadEvent()
        report_holder: dict[str, dict[str, int]] = {}

        def run_batch() -> None:
            try:
                report_holder["report"] = recover_durable_query_executions(
                    execute,
                    limit=batch_limit,
                )
            except Exception:
                report_holder["error"] = {"failed": 1}
            finally:
                completed.set()

        # daemon thread 允许进程管理器在 grace 期后 hard-stop。关闭路径会先用
        # DurableQueryRecoveryController fence 本进程 lease/active attempt。
        Thread(
            target=run_batch,
            name="query-checkpoint-recovery-batch",
            daemon=True,
        ).start()
        while not completed.is_set() and not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.05)
            except TimeoutError:
                continue
        if stop.is_set():
            return
        report = report_holder.get("report")
        if report is None:
            logger.error(
                "query checkpoint recovery loop failed: code={} stage={}",
                "QUERY_CHECKPOINT_RECOVERY_LOOP_FAILED",
                "batch",
            )
        elif report["scanned"]:
            logger.info("可靠查询 checkpoint 周期恢复完成：{}", report)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


__all__ = [
    "DurableQueryRecoveryController",
    "QueryAuthBinding",
    "QueryDurabilityConfigurationError",
    "QueryResumeIdentityError",
    "QueryRecoveryStopped",
    "bind_query_execution_identity",
    "build_durable_query_executor",
    "build_query_checkpoint_cipher",
    "initialize_query_execution",
    "recover_durable_query_executions",
    "recover_query_principal",
    "run_durable_query_recovery_loop",
]
