"""TODO-24/25/31 查询运行控制面 ORM 模型。

模型放在独立模块以降低大型 ``models.py`` 的合并冲突。运行仓储导入本模块后，
所有表仍注册到同一个 ``Base.metadata``；Alembic 迁移应显式创建这些表。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import Mapped, mapped_column

from app.infra.persistence.sql.models import Base, TimestampMixin


class QueryConversationModel(Base, TimestampMixin):
    __tablename__ = "query_conversations"
    __table_args__ = (
        Index(
            "ix_query_conversation_owner_updated",
            "tenant_id",
            "user_id",
            "updated_at",
        ),
        CheckConstraint("revision >= 0", name="ck_query_conversation_revision"),
        CheckConstraint("next_turn_seq >= 1", name="ck_query_conversation_next_turn"),
        CheckConstraint(
            "resolved_turn_seq >= 0 AND resolved_turn_seq < next_turn_seq",
            name="ck_query_conversation_frontier",
        ),
        CheckConstraint("history_version >= 0", name="ck_query_conversation_history_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_turn_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    resolved_turn_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    history_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryTurnModel(Base, TimestampMixin):
    __tablename__ = "query_turns"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "client_turn_id",
            name="uq_query_turn_client_identity",
        ),
        UniqueConstraint("conversation_id", "seq", name="uq_query_turn_sequence"),
        Index("ix_query_turn_conversation_status", "conversation_id", "status", "seq"),
        CheckConstraint("seq >= 1", name="ck_query_turn_seq"),
        CheckConstraint("base_history_seq >= 0", name="ck_query_turn_base_history"),
        CheckConstraint("history_version >= 0", name="ck_query_turn_history_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("query_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    client_turn_id: Mapped[str] = mapped_column(String(128), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    base_history_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    history_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    input_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    user_content: Mapped[str] = mapped_column(Text, nullable=False)
    active_query_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ready_result_json: Mapped[dict | None] = mapped_column(JSON)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryMessageModel(Base):
    __tablename__ = "query_messages"
    __table_args__ = (
        UniqueConstraint("turn_id", "ordinal", name="uq_query_message_turn_ordinal"),
        Index(
            "ix_query_message_history",
            "conversation_id",
            "turn_seq",
            "ordinal",
        ),
        CheckConstraint("turn_seq >= 1", name="ck_query_message_turn_seq"),
        CheckConstraint("ordinal >= 0", name="ck_query_message_ordinal"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("query_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_id: Mapped[str] = mapped_column(
        ForeignKey("query_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    turn_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryRunModel(Base, TimestampMixin):
    __tablename__ = "query_runs"
    __table_args__ = (
        UniqueConstraint("turn_id", "attempt", name="uq_query_run_turn_attempt"),
        UniqueConstraint(
            "turn_id",
            "idempotency_key",
            name="uq_query_run_retry_idempotency",
        ),
        Index("ix_query_run_owner_status", "tenant_id", "user_id", "status"),
        Index("ix_query_run_conversation_created", "conversation_id", "created_at"),
        CheckConstraint("attempt >= 1", name="ck_query_run_attempt"),
        CheckConstraint("revision >= 0", name="ck_query_run_revision"),
        CheckConstraint("event_sequence >= 0", name="ck_query_run_event_sequence"),
        CheckConstraint(
            "history_base_turn_seq >= 0",
            name="ck_query_run_history_base_turn",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("query_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_id: Mapped[str] = mapped_column(
        ForeignKey("query_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_query_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    replay_of_query_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="accepted")
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="accepted")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    profile_version: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    index_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    config_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    history_snapshot_hash: Mapped[str | None] = mapped_column(String(71))
    history_base_turn_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    history_snapshot_json: Mapped[dict | None] = mapped_column(JSON)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    safe_error_message: Mapped[str | None] = mapped_column(String(1024))
    error_detail_ref: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    result_json: Mapped[dict | None] = mapped_column(JSON)


_CHECKPOINT_CIPHERTEXT_TYPE = LargeBinary().with_variant(LONGBLOB(), "mysql")


class QueryRunExecutionModel(Base, TimestampMixin):
    """可靠 Query 专用的节点边界执行游标与 lease/fence。"""

    __tablename__ = "query_run_executions"
    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "checkpoint_namespace",
            name="uq_query_execution_thread_namespace",
        ),
        Index(
            "ix_query_execution_recovery",
            "status",
            "lease_expires_at",
        ),
        Index(
            "ix_query_execution_owner",
            "tenant_id",
            "user_id",
            "status",
        ),
        CheckConstraint("revision >= 0", name="ck_query_execution_revision"),
        CheckConstraint("fencing_token >= 0", name="ck_query_execution_fence"),
        CheckConstraint(
            "checkpoint_sequence >= 0",
            name="ck_query_execution_checkpoint_seq",
        ),
        CheckConstraint(
            "attempt_sequence >= 0",
            name="ck_query_execution_attempt_seq",
        ),
        CheckConstraint("recovery_count >= 0", name="ck_query_execution_recovery_count"),
        CheckConstraint(
            "status IN ('ready', 'running', 'completed', 'interrupted', "
            "'failed', 'cancelled', 'timed_out')",
            name="ck_query_execution_status",
        ),
    )

    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False)
    credential_id: Mapped[str] = mapped_column(String(128), nullable=False)
    identity_version: Mapped[int | None] = mapped_column(Integer)
    roles_digest: Mapped[str] = mapped_column(String(71), nullable=False)
    scopes_digest: Mapped[str] = mapped_column(String(71), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    checkpoint_namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    state_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token_hash: Mapped[str | None] = mapped_column(String(71))
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_node: Mapped[str | None] = mapped_column(String(64))
    checkpoint_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recovery_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    safe_error_code: Mapped[str | None] = mapped_column(String(128))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryRunCheckpointModel(Base):
    """仅保存密文；查询正文、中间证据与答案不得进入普通 JSON/Text 列。"""

    __tablename__ = "query_run_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "query_run_id",
            "sequence",
            name="uq_query_checkpoint_run_sequence",
        ),
        Index(
            "ix_query_checkpoint_latest",
            "query_run_id",
            "sequence",
        ),
        CheckConstraint("sequence >= 0", name="ck_query_checkpoint_sequence"),
        CheckConstraint("size_bytes >= 16", name="ck_query_checkpoint_size"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_run_executions.query_run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    state_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_node: Mapped[str | None] = mapped_column(String(64))
    next_node: Mapped[str | None] = mapped_column(String(64))
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(_CHECKPOINT_CIPHERTEXT_TYPE, nullable=False)
    ciphertext_sha256: Mapped[str] = mapped_column(String(71), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class QueryNodeAttemptModel(Base):
    """节点调用前先持久化 started；没有后继 checkpoint 时禁止自动重放。"""

    __tablename__ = "query_node_attempts"
    __table_args__ = (
        UniqueConstraint(
            "query_run_id",
            "sequence",
            name="uq_query_node_attempt_sequence",
        ),
        Index(
            "ix_query_node_attempt_active",
            "query_run_id",
            "status",
            "sequence",
        ),
        CheckConstraint("sequence >= 1", name="ck_query_node_attempt_sequence"),
        CheckConstraint("revision >= 0", name="ck_query_node_attempt_revision"),
        CheckConstraint(
            "checkpoint_sequence_before >= 0",
            name="ck_query_node_attempt_checkpoint_before",
        ),
        CheckConstraint(
            "checkpoint_sequence_after IS NULL OR "
            "checkpoint_sequence_after > checkpoint_sequence_before",
            name="ck_query_node_attempt_checkpoint_after",
        ),
        CheckConstraint("fencing_token >= 1", name="ck_query_node_attempt_fence"),
        CheckConstraint(
            "status IN ('scheduled', 'started', 'checkpointed', 'interrupted', 'failed')",
            name="ck_query_node_attempt_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_run_executions.query_run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(String(64), nullable=False)
    effect_class: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="scheduled")
    checkpoint_sequence_before: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_sequence_after: Mapped[int | None] = mapped_column(Integer)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    lease_owner: Mapped[str] = mapped_column(String(128), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    safe_error_code: Mapped[str | None] = mapped_column(String(128))
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpointed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryEventModel(Base):
    __tablename__ = "query_events"
    __table_args__ = (
        UniqueConstraint("query_run_id", "seq", name="uq_query_event_run_sequence"),
        UniqueConstraint(
            "query_run_id",
            "operation_id",
            name="uq_query_event_run_operation",
        ),
        Index("ix_query_event_run_occurred", "query_run_id", "occurred_at"),
        Index("ix_query_event_expiry", "expires_at"),
        CheckConstraint("seq >= 1", name="ck_query_event_seq"),
        CheckConstraint("size_bytes >= 0", name="ck_query_event_size"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_run_id: Mapped[str] = mapped_column(
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    operation_id: Mapped[str | None] = mapped_column(String(128))
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    stage: Mapped[str | None] = mapped_column(String(64))
    data_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    visibility: Mapped[str] = mapped_column(String(16), nullable=False, default="public")
    persistence_class: Mapped[str] = mapped_column(String(16), nullable=False, default="durable")
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="query-event.v1"
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = [
    "QueryConversationModel",
    "QueryEventModel",
    "QueryMessageModel",
    "QueryNodeAttemptModel",
    "QueryRunCheckpointModel",
    "QueryRunExecutionModel",
    "QueryRunModel",
    "QueryTurnModel",
]
