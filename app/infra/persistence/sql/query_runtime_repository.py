"""TODO-24/25/31 tenant/user-scoped 查询运行、历史和事件 SQL 仓储。

所有运行状态写入均使用 ``revision`` CAS；会话 turn 序号和提交游标也使用
``revision`` CAS。浏览器连接、进程内队列和 ``session_id`` 都不是事实源。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import hmac
import json
import re
from secrets import token_urlsafe
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from app.domain.query_runtime import (
    Conversation,
    ConversationHistorySnapshot,
    ConversationStatus,
    EventPage,
    HistorySnapshotMessage,
    Message,
    QueryEvent,
    QueryEventType,
    QueryRun,
    QueryRunStatus,
    RUN_TERMINAL_STATUSES,
    TERMINAL_EVENT_TYPES,
    TURN_TERMINAL_STATUSES,
    Turn,
    TurnStatus,
    build_history_snapshot,
    canonical_hash,
    ensure_run_transition,
    ensure_turn_transition,
    new_runtime_id,
)
from app.domain.query_checkpoint import (
    QUERY_EXECUTION_TERMINAL_STATUSES,
    QueryCheckpointRecord,
    QueryExecutionLease,
    QueryExecutionRecord,
    QueryExecutionStatus,
    QueryNodeAttemptRecord,
    QueryNodeAttemptStatus,
)
from app.foundation.query_checkpoint_crypto import EncryptedQueryCheckpoint
from app.infra.persistence.sql.query_runtime_models import (
    QueryConversationModel,
    QueryEventModel,
    QueryMessageModel,
    QueryNodeAttemptModel,
    QueryRunCheckpointModel,
    QueryRunExecutionModel,
    QueryRunModel,
    QueryTurnModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)


class StaleQueryRevision(ResourceConflictError):
    """CAS 版本已被另一 worker 修改，调用方应重新读取后决定是否重试。"""


class EventCursorInvalid(ResourceConflictError):
    """Last-Event-ID 不属于当前运行，禁止静默跨 run 续传。"""


class EventCursorExpired(ResourceConflictError):
    """续传游标已超过事件保留期，需要客户端获取运行快照后重置。"""


class QueryExecutionLeaseBusy(ResourceConflictError):
    """另一个 worker 仍持有未过期 lease。"""


class QueryExecutionFenceLost(StaleQueryRevision):
    """worker 的 lease/fencing token 已过期，禁止再提交节点结果。"""


class QueryExecutionNotInitialized(ResourceConflictError):
    """可靠 Query 尚未持久化 initial checkpoint。"""


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _json_size(value: dict[str, Any]) -> int:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("query runtime payload must be JSON serializable") from exc
    return len(payload.encode("utf-8"))


_TERMINAL_EVENT_BY_STATUS = {
    QueryRunStatus.COMPLETED: QueryEventType.RUN_COMPLETED,
    QueryRunStatus.FAILED: QueryEventType.RUN_FAILED,
    QueryRunStatus.TIMED_OUT: QueryEventType.RUN_TIMED_OUT,
    QueryRunStatus.CANCELLED: QueryEventType.RUN_CANCELLED,
}


class QueryRuntimeRepository:
    """构造时固定 tenant/user；任何 ID 查询都不会扩大该作用域。"""

    def __init__(
        self,
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
        max_event_bytes: int = 1_048_576,
    ) -> None:
        if not tenant_id.strip() or not user_id.strip():
            raise ValueError("query runtime repository requires tenant_id and user_id")
        self.session = session
        self.tenant_id = tenant_id.strip()
        self.user_id = user_id.strip()
        self.max_event_bytes = max_event_bytes

    # ------------------------------------------------------------------
    # Conversation / Turn / immutable history snapshot
    # ------------------------------------------------------------------
    def create_conversation(
        self,
        *,
        kb_id: str,
        title: str = "",
        conversation_id: str | None = None,
    ) -> Conversation:
        if not 1 <= len(kb_id) <= 64:
            raise ValueError("kb_id length must be between 1 and 64")
        if len(title) > 512:
            raise ValueError("conversation title is too long")
        identifier = conversation_id or new_runtime_id("conv")
        if re.fullmatch(r"conv_[0-9a-f]{32}", identifier) is None:
            raise ValueError("conversation_id must use the conv_<32 lowercase hex> form")
        existing = self.session.scalar(
            select(QueryConversationModel).where(
                QueryConversationModel.id == identifier,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
            )
        )
        if existing is not None:
            if existing.kb_id != kb_id:
                raise ResourceConflictError(
                    "conversation identity already belongs to another knowledge base"
                )
            return self._conversation(existing)
        collision = self.session.get(QueryConversationModel, identifier)
        if collision is not None:
            # 不向另一 tenant/user 暴露该 UUID 的存在。
            raise ResourceNotFoundError("query conversation not found")
        row = QueryConversationModel(
            id=identifier,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            kb_id=kb_id,
            title=title.strip(),
            status=ConversationStatus.ACTIVE.value,
            revision=0,
            next_turn_seq=1,
            resolved_turn_seq=0,
            history_version=0,
        )
        self.session.add(row)
        self.session.flush()
        return self._conversation(row)

    def get_conversation(self, conversation_id: str) -> Conversation:
        return self._conversation(self._conversation_row(conversation_id))

    def list_conversations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
        allowed_kb_ids: tuple[str, ...] | None = None,
        status_filter: str | None = None,
        search: str = "",
    ) -> list[Conversation]:
        if not 1 <= limit <= 200:
            raise ValueError("conversation limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("conversation offset cannot be negative")
        if allowed_kb_ids is not None:
            normalized_kb_ids = tuple(dict.fromkeys(item.strip() for item in allowed_kb_ids))
            if any(not item for item in normalized_kb_ids):
                raise ValueError("allowed conversation kb ids cannot contain blanks")
            # 权限过滤必须在 offset/limit 之前进入 SQL；先分页再在 Python 中删掉
            # 撤权 KB 会造成漏页，并让客户端无法稳定遍历当前可见的全部会话。
            if not normalized_kb_ids:
                return []
        if status_filter is None:
            status_filter = "all" if include_archived else ConversationStatus.ACTIVE.value
        if status_filter not in {
            "all",
            ConversationStatus.ACTIVE.value,
            ConversationStatus.ARCHIVED.value,
        }:
            raise ValueError("conversation status filter is invalid")
        normalized_search = search.strip()
        if len(normalized_search) > 200:
            raise ValueError("conversation search is too long")
        criteria = [
            QueryConversationModel.tenant_id == self.tenant_id,
            QueryConversationModel.user_id == self.user_id,
        ]
        if status_filter != "all":
            criteria.append(QueryConversationModel.status == status_filter)
        if allowed_kb_ids is not None:
            criteria.append(QueryConversationModel.kb_id.in_(normalized_kb_ids))
        if normalized_search:
            escaped = (
                normalized_search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            pattern = f"%{escaped}%"
            criteria.append(
                or_(
                    QueryConversationModel.title.ilike(pattern, escape="\\"),
                    QueryConversationModel.id.ilike(pattern, escape="\\"),
                    QueryConversationModel.kb_id.ilike(pattern, escape="\\"),
                )
            )
        rows = self.session.scalars(
            select(QueryConversationModel)
            .where(*criteria)
            .order_by(
                QueryConversationModel.updated_at.desc(),
                QueryConversationModel.id.asc(),
            )
            .offset(offset)
            .limit(limit)
        ).all()
        return [self._conversation(row) for row in rows]

    def list_conversation_kb_ids(
        self,
        *,
        include_archived: bool = False,
        status_filter: str | None = None,
    ) -> tuple[str, ...]:
        """列出当前 tenant/user 历史实际引用的 KB，供实时 entitlement 求交。

        该方法只返回资源标识候选集，不代表授权结果。调用方必须再与当前授权
        事实求交，并把结果传回 :meth:`list_conversations` 参与 SQL 分页。
        """

        if status_filter is None:
            status_filter = "all" if include_archived else ConversationStatus.ACTIVE.value
        if status_filter not in {
            "all",
            ConversationStatus.ACTIVE.value,
            ConversationStatus.ARCHIVED.value,
        }:
            raise ValueError("conversation status filter is invalid")
        criteria = [
            QueryConversationModel.tenant_id == self.tenant_id,
            QueryConversationModel.user_id == self.user_id,
        ]
        if status_filter != "all":
            criteria.append(QueryConversationModel.status == status_filter)
        rows = self.session.scalars(
            select(QueryConversationModel.kb_id)
            .where(*criteria)
            .distinct()
            .order_by(QueryConversationModel.kb_id.asc())
        ).all()
        return tuple(rows)

    def archive_conversation(
        self,
        conversation_id: str,
        *,
        expected_revision: int,
    ) -> Conversation:
        row = self._conversation_row(conversation_id)
        if row.revision != expected_revision:
            raise StaleQueryRevision("conversation revision changed")
        if row.status == ConversationStatus.ARCHIVED.value:
            return self._conversation(row)
        active = self.session.scalar(
            select(func.count(QueryRunModel.id)).where(
                QueryRunModel.conversation_id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.status.not_in([status.value for status in RUN_TERMINAL_STATUSES]),
            )
        )
        if active:
            raise ResourceConflictError("conversation with active query runs cannot be archived")
        now = _now()
        result = self.session.execute(
            update(QueryConversationModel)
            .where(
                QueryConversationModel.id == row.id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
                QueryConversationModel.revision == expected_revision,
            )
            .values(
                status=ConversationStatus.ARCHIVED.value,
                archived_at=now,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            raise StaleQueryRevision("conversation revision changed")
        self.session.expire(row)
        return self._conversation(row)

    def restore_conversation(
        self,
        conversation_id: str,
        *,
        expected_revision: int,
    ) -> Conversation:
        """以 CAS 恢复当前 owner 的归档会话；不会创建或复制历史事实。"""

        row = self._conversation_row(conversation_id)
        if row.revision != expected_revision:
            raise StaleQueryRevision("conversation revision changed")
        if row.status == ConversationStatus.ACTIVE.value:
            return self._conversation(row)
        now = _now()
        result = self.session.execute(
            update(QueryConversationModel)
            .where(
                QueryConversationModel.id == row.id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
                QueryConversationModel.status == ConversationStatus.ARCHIVED.value,
                QueryConversationModel.revision == expected_revision,
            )
            .values(
                status=ConversationStatus.ACTIVE.value,
                archived_at=None,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            raise StaleQueryRevision("conversation revision changed")
        self.session.expire(row)
        return self._conversation(row)

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        expected_revision: int,
    ) -> int:
        """永久删除已归档会话；数据库外键级联清理 turn/run/event/message。"""

        row = self._conversation_row(conversation_id)
        if row.revision != expected_revision:
            raise StaleQueryRevision("conversation revision changed")
        if row.status != ConversationStatus.ARCHIVED.value:
            raise ResourceConflictError("conversation must be archived before deletion")
        active = self.session.scalar(
            select(func.count(QueryRunModel.id)).where(
                QueryRunModel.conversation_id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.status.not_in([status.value for status in RUN_TERMINAL_STATUSES]),
            )
        )
        if active:
            raise ResourceConflictError("conversation with active query runs cannot be deleted")
        result = self.session.execute(
            delete(QueryConversationModel).where(
                QueryConversationModel.id == row.id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
                QueryConversationModel.status == ConversationStatus.ARCHIVED.value,
                QueryConversationModel.revision == expected_revision,
            )
        )
        if result.rowcount != 1:
            raise StaleQueryRevision("conversation revision changed")
        return expected_revision + 1

    def begin_turn(
        self,
        *,
        conversation_id: str,
        client_turn_id: str,
        user_content: str,
        idempotency_key: str | None = None,
        profile_version: str = "",
        index_snapshot: dict[str, Any] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        deadline_at: datetime | None = None,
        max_history_messages: int = 40,
        max_history_tokens: int = 4096,
    ) -> tuple[Turn, QueryRun, ConversationHistorySnapshot, bool]:
        """事务创建 Turn + 首次 QueryRun；相同 client turn 返回原身份。

        返回值最后一项表示是否为本次新建。调用方应在同一个事务提交，避免
        ``run`` 已可见但 ``run.accepted`` 事件尚未可见。
        """

        if not 1 <= len(client_turn_id) <= 128:
            raise ValueError("client_turn_id length must be between 1 and 128")
        if not user_content.strip() or len(user_content) > 1_000_000:
            raise ValueError("user_content must contain 1 to 1000000 characters")
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 256:
            raise ValueError("idempotency_key length must be between 1 and 256")
        if len(profile_version) > 128:
            raise ValueError("profile_version is too long")
        if deadline_at is not None and deadline_at.tzinfo is None:
            raise ValueError("deadline_at must include timezone")
        _json_size(index_snapshot or {})
        _json_size(config_snapshot or {})
        conversation = self._conversation_row(conversation_id)
        if conversation.status != ConversationStatus.ACTIVE.value:
            raise ResourceConflictError("archived conversation does not accept new turns")
        self._drain_frontier(conversation)
        self.session.refresh(conversation)
        request_hash = canonical_hash(
            {
                "user_content": user_content,
                "profile_version": profile_version,
                "index_snapshot": index_snapshot or {},
                "config_snapshot": config_snapshot or {},
            }
        )
        existing_turn = self.session.scalar(
            select(QueryTurnModel).where(
                QueryTurnModel.conversation_id == conversation.id,
                QueryTurnModel.client_turn_id == client_turn_id,
            )
        )
        if existing_turn is not None:
            if existing_turn.input_hash != request_hash:
                raise ResourceConflictError(
                    "client_turn_id already belongs to another request payload"
                )
            existing_run = self.session.scalar(
                select(QueryRunModel)
                .where(
                    QueryRunModel.turn_id == existing_turn.id,
                    QueryRunModel.tenant_id == self.tenant_id,
                    QueryRunModel.user_id == self.user_id,
                )
                .order_by(QueryRunModel.attempt.desc())
            )
            if existing_run is None:
                raise ResourceConflictError("idempotent turn has no query run")
            snapshot = self._snapshot_from_run(existing_run)
            return (
                self._turn(existing_turn),
                self._run(existing_run),
                snapshot,
                False,
            )

        seq = conversation.next_turn_seq
        revision = conversation.revision
        now = _now()
        allocated = self.session.execute(
            update(QueryConversationModel)
            .where(
                QueryConversationModel.id == conversation.id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
                QueryConversationModel.revision == revision,
                QueryConversationModel.next_turn_seq == seq,
            )
            .values(
                next_turn_seq=seq + 1,
                revision=revision + 1,
                updated_at=now,
            )
        )
        if allocated.rowcount != 1:
            raise StaleQueryRevision("conversation changed while allocating the next turn sequence")
        turn = QueryTurnModel(
            id=new_runtime_id("turn"),
            conversation_id=conversation.id,
            client_turn_id=client_turn_id,
            seq=seq,
            base_history_seq=conversation.resolved_turn_seq,
            history_version=conversation.history_version,
            status=TurnStatus.RUNNING.value,
            input_hash=request_hash,
            user_content=user_content,
        )
        self.session.add(turn)
        # 没有 ORM relationship 时 SQLAlchemy 的 unit-of-work 不保证按 FK 依赖排序；
        # 先落 Turn，再创建引用它的 QueryRun。
        self.session.flush()
        run_id = new_runtime_id("qry")
        run = QueryRunModel(
            id=run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            conversation_id=conversation.id,
            turn_id=turn.id,
            attempt=1,
            idempotency_key=idempotency_key,
            status=QueryRunStatus.ACCEPTED.value,
            current_stage=QueryRunStatus.ACCEPTED.value,
            revision=0,
            event_sequence=0,
            profile_version=profile_version,
            index_snapshot_json=index_snapshot or {},
            config_snapshot_json=config_snapshot or {},
            history_base_turn_seq=turn.base_history_seq,
            deadline_at=deadline_at,
            trace_id=run_id,
        )
        turn.active_query_run_id = run.id
        self.session.add(run)
        self.session.flush()
        snapshot = self.load_history_snapshot(
            conversation.id,
            base_turn_seq=turn.base_history_seq,
            history_version=turn.history_version,
            max_messages=max_history_messages,
            max_tokens=max_history_tokens,
        )
        run.history_snapshot_hash = snapshot.content_hash
        run.history_snapshot_json = snapshot.model_dump(mode="json")
        self.session.flush()
        self.append_event(
            run.id,
            event_type=QueryEventType.RUN_ACCEPTED,
            data={
                "status": QueryRunStatus.ACCEPTED.value,
                "turn_id": turn.id,
                "conversation_id": conversation.id,
                "attempt": 1,
            },
        )
        return self._turn(turn), self._run(run), snapshot, True

    def load_history_snapshot(
        self,
        conversation_id: str,
        *,
        base_turn_seq: int | None = None,
        history_version: int | None = None,
        max_messages: int = 40,
        max_tokens: int = 4096,
    ) -> ConversationHistorySnapshot:
        conversation = self._conversation_row(conversation_id)
        base = conversation.resolved_turn_seq if base_turn_seq is None else base_turn_seq
        if base > conversation.resolved_turn_seq:
            raise ResourceConflictError(
                "history snapshot cannot read beyond the resolved turn frontier"
            )
        rows = self.session.execute(
            select(QueryMessageModel, QueryTurnModel.seq)
            .join(QueryTurnModel, QueryTurnModel.id == QueryMessageModel.turn_id)
            .where(
                QueryMessageModel.tenant_id == self.tenant_id,
                QueryMessageModel.conversation_id == conversation.id,
                QueryMessageModel.status == "committed",
                QueryTurnModel.status == TurnStatus.COMMITTED.value,
                QueryTurnModel.seq <= base,
            )
            .order_by(QueryTurnModel.seq, QueryMessageModel.ordinal)
        ).all()
        messages = [
            HistorySnapshotMessage(
                message_id=row.id,
                turn_id=row.turn_id,
                turn_seq=turn_seq,
                ordinal=row.ordinal,
                role=row.role,
                content=row.content,
                committed_at=_aware(row.committed_at) or _aware(row.created_at) or _now(),
            )
            for row, turn_seq in rows
        ]
        return build_history_snapshot(
            conversation_id=conversation.id,
            tenant_id=self.tenant_id,
            base_turn_seq=base,
            history_version=(
                conversation.history_version if history_version is None else history_version
            ),
            messages=messages,
            max_messages=max_messages,
            max_tokens=max_tokens,
        )

    def get_turn(self, turn_id: str) -> Turn:
        return self._turn(self._turn_row(turn_id))

    def list_messages(self, conversation_id: str) -> list[Message]:
        conversation = self._conversation_row(conversation_id)
        rows = self.session.scalars(
            select(QueryMessageModel)
            .where(
                QueryMessageModel.tenant_id == self.tenant_id,
                QueryMessageModel.conversation_id == conversation.id,
            )
            .order_by(QueryMessageModel.turn_seq, QueryMessageModel.ordinal)
        ).all()
        return [self._message(row) for row in rows]

    def list_turns(self, conversation_id: str) -> list[Turn]:
        """返回会话全部轮次，供 HTTP 层投影在途/失败安全视图。"""

        conversation = self._conversation_row(conversation_id)
        rows = self.session.scalars(
            select(QueryTurnModel)
            .where(QueryTurnModel.conversation_id == conversation.id)
            .order_by(QueryTurnModel.seq.asc(), QueryTurnModel.id.asc())
        ).all()
        return [self._turn(row) for row in rows]

    # ------------------------------------------------------------------
    # QueryRun CAS state machine / result / retry / cancel
    # ------------------------------------------------------------------
    def get_run(self, query_run_id: str) -> QueryRun:
        return self._run(self._run_row(query_run_id))

    def get_history_snapshot(
        self,
        query_run_id: str,
    ) -> ConversationHistorySnapshot:
        return self._snapshot_from_run(self._run_row(query_run_id))

    def list_runs(self, conversation_id: str) -> list[QueryRun]:
        conversation = self._conversation_row(conversation_id)
        rows = self.session.scalars(
            select(QueryRunModel)
            .where(
                QueryRunModel.conversation_id == conversation.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
            )
            .order_by(QueryRunModel.created_at, QueryRunModel.attempt)
        ).all()
        return [self._run(row) for row in rows]

    def transition_run(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        target: QueryRunStatus,
        stage: str | None = None,
    ) -> QueryRun:
        row = self._run_row(query_run_id)
        current = QueryRunStatus(row.status)
        if current == target:
            if row.revision != expected_revision:
                raise StaleQueryRevision(
                    f"query run revision changed from expected {expected_revision}"
                )
            return self._run(row)
        ensure_run_transition(current, target)
        now = _now()
        values: dict[str, Any] = {
            "status": target.value,
            "current_stage": stage or target.value,
            "revision": expected_revision + 1,
            "updated_at": now,
        }
        if row.started_at is None and target == QueryRunStatus.PLANNING:
            values["started_at"] = now
        if target in RUN_TERMINAL_STATUSES:
            values["finished_at"] = now
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.revision == expected_revision,
                QueryRunModel.status == current.value,
            )
            .values(**values)
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision(
                f"query run revision changed from expected {expected_revision}"
            )
        self.session.expire(row)
        updated = self._run(row)
        self.append_event(
            row.id,
            event_type=QueryEventType.STAGE_STARTED,
            stage=updated.current_stage,
            data={"status": updated.status.value},
        )
        return updated

    def complete_run(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        answer: str,
        result: dict[str, Any] | None = None,
    ) -> QueryRun:
        if len(answer) > 1_000_000:
            raise ValueError("answer exceeds the maximum persisted size")
        row = self._run_row(query_run_id)
        current = QueryRunStatus(row.status)
        if current == QueryRunStatus.COMPLETED:
            return self._run(row)
        ensure_run_transition(current, QueryRunStatus.COMPLETED)
        payload = dict(result or {})
        payload["answer"] = answer
        if _json_size(payload) > 5 * 1024 * 1024:
            raise ValueError("query result exceeds the maximum persisted size")
        now = _now()
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.revision == expected_revision,
                QueryRunModel.status == current.value,
            )
            .values(
                status=QueryRunStatus.COMPLETED.value,
                current_stage=QueryRunStatus.COMPLETED.value,
                result_json=payload,
                finished_at=now,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision(
                "complete lost a race to cancel, timeout, failure or another completion"
            )
        turn = self._turn_row(row.turn_id)
        if turn.active_query_run_id != row.id:
            raise ResourceConflictError("query run is no longer the active turn attempt")
        ensure_turn_transition(TurnStatus(turn.status), TurnStatus.READY_TO_COMMIT)
        turn.status = TurnStatus.READY_TO_COMMIT.value
        turn.ready_result_json = payload
        self.session.flush()
        self.session.expire(row)
        event = self.append_event(
            row.id,
            event_type=QueryEventType.RUN_COMPLETED,
            stage=QueryRunStatus.COMPLETED.value,
            data={
                "status": QueryRunStatus.COMPLETED.value,
                "result_available": True,
            },
        )
        if event.type != QueryEventType.RUN_COMPLETED:
            raise ResourceConflictError("completed run did not persist its terminal event")
        conversation = self._conversation_row(row.conversation_id)
        self._drain_frontier(conversation)
        self.session.expire(row)
        return self._run(row)

    def fail_run(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        error_code: str,
        safe_message: str,
        retryable: bool,
        error_detail_ref: str | None = None,
        timed_out: bool = False,
    ) -> QueryRun:
        if not error_code or len(error_code) > 128:
            raise ValueError("error_code length must be between 1 and 128")
        if not safe_message or len(safe_message) > 1024:
            raise ValueError("safe_message length must be between 1 and 1024")
        target = QueryRunStatus.TIMED_OUT if timed_out else QueryRunStatus.FAILED
        row = self._run_row(query_run_id)
        current = QueryRunStatus(row.status)
        if current == target:
            return self._run(row)
        ensure_run_transition(current, target)
        now = _now()
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.revision == expected_revision,
                QueryRunModel.status == current.value,
            )
            .values(
                status=target.value,
                current_stage=target.value,
                error_code=error_code,
                safe_error_message=safe_message,
                error_detail_ref=error_detail_ref,
                retryable=retryable,
                finished_at=now,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision(
                "failure lost a race to cancel, completion or another terminal state"
            )
        turn = self._turn_row(row.turn_id)
        if turn.active_query_run_id == row.id:
            turn_target = TurnStatus.TIMED_OUT if timed_out else TurnStatus.FAILED
            ensure_turn_transition(TurnStatus(turn.status), turn_target)
            turn.status = turn_target.value
            turn.resolved_at = now
        self.session.flush()
        self.session.expire(row)
        self.append_event(
            row.id,
            event_type=(QueryEventType.RUN_TIMED_OUT if timed_out else QueryEventType.RUN_FAILED),
            stage=target.value,
            data={
                "status": target.value,
                "error": {
                    "code": error_code,
                    "message": safe_message,
                    "retryable": retryable,
                },
            },
        )
        self._drain_when_a_later_turn_is_ready(row.conversation_id, turn.seq)
        self.session.expire(row)
        return self._run(row)

    def retry_run(
        self,
        query_run_id: str,
        *,
        idempotency_key: str,
        deadline_at: datetime | None = None,
    ) -> tuple[QueryRun, ConversationHistorySnapshot, bool]:
        if not 1 <= len(idempotency_key) <= 256:
            raise ValueError("retry idempotency_key length must be between 1 and 256")
        if deadline_at is not None and deadline_at.tzinfo is None:
            raise ValueError("deadline_at must include timezone")
        previous = self._run_row(query_run_id)
        previous_status = QueryRunStatus(previous.status)
        if previous_status not in {
            QueryRunStatus.FAILED,
            QueryRunStatus.TIMED_OUT,
            QueryRunStatus.CANCELLED,
        }:
            raise ResourceConflictError("only a failed/cancelled/timed-out run can be retried")
        if not previous.retryable and previous_status != QueryRunStatus.CANCELLED:
            raise ResourceConflictError("query run is not retryable")
        existing = self.session.scalar(
            select(QueryRunModel).where(
                QueryRunModel.turn_id == previous.turn_id,
                QueryRunModel.idempotency_key == idempotency_key,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
            )
        )
        if existing is not None:
            return self._run(existing), self._snapshot_from_run(existing), False
        turn = self._turn_row(previous.turn_id)
        conversation = self._conversation_row(previous.conversation_id)
        if conversation.resolved_turn_seq >= turn.seq:
            raise ResourceConflictError(
                "resolved turn cannot be retried in-place; create an explicit branch/new turn"
            )
        if turn.status not in {
            TurnStatus.FAILED.value,
            TurnStatus.TIMED_OUT.value,
            TurnStatus.CANCELLED.value,
        }:
            raise ResourceConflictError("turn is not in a retryable state")
        attempt = (
            self.session.scalar(
                select(func.max(QueryRunModel.attempt)).where(QueryRunModel.turn_id == turn.id)
            )
            or 0
        ) + 1
        run_id = new_runtime_id("qry")
        run = QueryRunModel(
            id=run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            conversation_id=conversation.id,
            turn_id=turn.id,
            attempt=attempt,
            parent_query_run_id=previous.id,
            replay_of_query_run_id=previous.id,
            idempotency_key=idempotency_key,
            status=QueryRunStatus.ACCEPTED.value,
            current_stage=QueryRunStatus.ACCEPTED.value,
            revision=0,
            event_sequence=0,
            profile_version=previous.profile_version,
            index_snapshot_json=previous.index_snapshot_json,
            config_snapshot_json=previous.config_snapshot_json,
            history_snapshot_hash=previous.history_snapshot_hash,
            history_base_turn_seq=previous.history_base_turn_seq,
            history_snapshot_json=previous.history_snapshot_json,
            deadline_at=deadline_at or previous.deadline_at,
            trace_id=run_id,
        )
        ensure_turn_transition(TurnStatus(turn.status), TurnStatus.RUNNING)
        turn.status = TurnStatus.RUNNING.value
        turn.active_query_run_id = run.id
        turn.ready_result_json = None
        turn.resolved_at = None
        self.session.add(run)
        self.session.flush()
        self.append_event(
            run.id,
            event_type=QueryEventType.RUN_ACCEPTED,
            data={
                "status": QueryRunStatus.ACCEPTED.value,
                "turn_id": turn.id,
                "conversation_id": conversation.id,
                "attempt": attempt,
                "retry_of": previous.id,
            },
        )
        return self._run(run), self._snapshot_from_run(run), True

    def request_cancel(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
        reason_code: str = "USER_REQUESTED",
    ) -> tuple[QueryRun, bool]:
        if not reason_code or len(reason_code) > 128:
            raise ValueError("cancel reason_code length must be between 1 and 128")
        row = self._run_row(query_run_id)
        current = QueryRunStatus(row.status)
        if current in RUN_TERMINAL_STATUSES:
            return self._run(row), False
        if current == QueryRunStatus.CANCEL_REQUESTED:
            return self._run(row), False
        ensure_run_transition(current, QueryRunStatus.CANCEL_REQUESTED)
        now = _now()
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.revision == expected_revision,
                QueryRunModel.status == current.value,
            )
            .values(
                status=QueryRunStatus.CANCEL_REQUESTED.value,
                current_stage=QueryRunStatus.CANCEL_REQUESTED.value,
                cancel_requested_at=now,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision(
                "cancel lost a race to completion, timeout or another transition"
            )
        self.session.expire(row)
        self.append_event(
            row.id,
            event_type=QueryEventType.RUN_CANCEL_REQUESTED,
            stage=QueryRunStatus.CANCEL_REQUESTED.value,
            data={"status": QueryRunStatus.CANCEL_REQUESTED.value, "reason": reason_code},
        )
        self.session.expire(row)
        return self._run(row), True

    def finalize_cancel(
        self,
        query_run_id: str,
        *,
        expected_revision: int,
    ) -> QueryRun:
        row = self._run_row(query_run_id)
        current = QueryRunStatus(row.status)
        if current == QueryRunStatus.CANCELLED:
            return self._run(row)
        ensure_run_transition(current, QueryRunStatus.CANCELLED)
        now = _now()
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == row.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.revision == expected_revision,
                QueryRunModel.status == QueryRunStatus.CANCEL_REQUESTED.value,
            )
            .values(
                status=QueryRunStatus.CANCELLED.value,
                current_stage=QueryRunStatus.CANCELLED.value,
                finished_at=now,
                retryable=True,
                revision=expected_revision + 1,
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision("query run changed before cancellation completed")
        turn = self._turn_row(row.turn_id)
        if turn.active_query_run_id == row.id:
            ensure_turn_transition(TurnStatus(turn.status), TurnStatus.CANCELLED)
            turn.status = TurnStatus.CANCELLED.value
            turn.resolved_at = now
        self.session.flush()
        self.session.expire(row)
        self.append_event(
            row.id,
            event_type=QueryEventType.RUN_CANCELLED,
            stage=QueryRunStatus.CANCELLED.value,
            data={"status": QueryRunStatus.CANCELLED.value},
        )
        self._drain_when_a_later_turn_is_ready(row.conversation_id, turn.seq)
        self.session.expire(row)
        return self._run(row)

    def get_result(self, query_run_id: str) -> dict[str, Any]:
        row = self._run_row(query_run_id)
        if row.status != QueryRunStatus.COMPLETED.value or row.result_json is None:
            raise ResourceConflictError("query result is not available")
        return dict(row.result_json)

    def cancellation_requested(self, query_run_id: str) -> bool:
        status = self.session.scalar(
            select(QueryRunModel.status).where(
                QueryRunModel.id == query_run_id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
            )
        )
        if status is None:
            raise ResourceNotFoundError("query run not found")
        return status in {
            QueryRunStatus.CANCEL_REQUESTED.value,
            QueryRunStatus.CANCELLED.value,
        }

    # ------------------------------------------------------------------
    # Durable QueryEvent log
    # ------------------------------------------------------------------
    def append_event(
        self,
        query_run_id: str,
        *,
        event_type: QueryEventType,
        data: dict[str, Any],
        stage: str | None = None,
        operation_id: str | None = None,
        visibility: str = "public",
        persistence_class: str = "durable",
        retention: timedelta | None = None,
        occurred_at: datetime | None = None,
    ) -> QueryEvent:
        if stage is not None and len(stage) > 64:
            raise ValueError("event stage is too long")
        if operation_id is not None and not 1 <= len(operation_id) <= 128:
            raise ValueError("event operation_id length must be between 1 and 128")
        run = self._run_row(query_run_id)
        run_status = QueryRunStatus(run.status)
        if event_type in TERMINAL_EVENT_TYPES:
            expected_type = _TERMINAL_EVENT_BY_STATUS.get(run_status)
            if expected_type != event_type:
                raise ResourceConflictError(
                    "terminal event type does not match persisted query run status"
                )
            existing_terminal = self.session.scalar(
                select(QueryEventModel).where(
                    QueryEventModel.query_run_id == run.id,
                    QueryEventModel.event_type.in_([event.value for event in TERMINAL_EVENT_TYPES]),
                )
            )
            if existing_terminal is not None:
                if existing_terminal.event_type == event_type.value:
                    return self._event(existing_terminal)
                raise ResourceConflictError("query run already has another terminal event")
        if visibility not in {"public", "internal"}:
            raise ValueError("event visibility must be public or internal")
        if persistence_class not in {"durable", "compactable"}:
            raise ValueError("event persistence_class is invalid")
        if event_type == QueryEventType.ANSWER_DELTA:
            if data.get("operation") not in {"append", "replace"}:
                raise ValueError("answer.delta requires append or replace operation")
            if not isinstance(data.get("text"), str):
                raise ValueError("answer.delta requires text")
        size = _json_size(data)
        if size > self.max_event_bytes:
            raise ValueError("query event exceeds max_event_bytes")
        if operation_id is not None:
            existing_operation = self.session.scalar(
                select(QueryEventModel).where(
                    QueryEventModel.query_run_id == run.id,
                    QueryEventModel.tenant_id == self.tenant_id,
                    QueryEventModel.operation_id == operation_id,
                )
            )
            if existing_operation is not None:
                identity = (
                    existing_operation.event_type,
                    existing_operation.stage,
                    existing_operation.visibility,
                    existing_operation.persistence_class,
                    existing_operation.data_json or {},
                )
                expected_identity = (
                    event_type.value,
                    stage,
                    visibility,
                    persistence_class,
                    data,
                )
                if identity != expected_identity:
                    raise ResourceConflictError(
                        "query event operation_id was reused with another payload"
                    )
                return self._event(existing_operation)
        now = occurred_at or _now()
        if now.tzinfo is None:
            raise ValueError("event occurred_at must include timezone")
        ttl = retention or (
            timedelta(days=7) if persistence_class == "durable" else timedelta(hours=24)
        )
        expected_seq = run.event_sequence
        changed = self.session.execute(
            update(QueryRunModel)
            .where(
                QueryRunModel.id == run.id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
                QueryRunModel.event_sequence == expected_seq,
            )
            .values(event_sequence=expected_seq + 1, updated_at=now)
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision("query event sequence changed; retry append transaction")
        seq = expected_seq + 1
        digest = canonical_hash({"query_run_id": run.id})[7:23]
        event_id = f"evt_{digest}_{seq:020d}"
        row = QueryEventModel(
            id=event_id,
            tenant_id=self.tenant_id,
            query_run_id=run.id,
            seq=seq,
            event_type=event_type.value,
            operation_id=operation_id,
            occurred_at=now,
            stage=stage,
            data_json=data,
            trace_id=run.trace_id,
            visibility=visibility,
            persistence_class=persistence_class,
            schema_version="query-event.v1",
            expires_at=now + ttl,
            size_bytes=size,
        )
        self.session.add(row)
        self.session.flush()
        self.session.expire(run)
        return self._event(row)

    def read_event_page(
        self,
        query_run_id: str,
        *,
        after_seq: int = 0,
        last_event_id: str | None = None,
        limit: int = 100,
        public_only: bool = True,
        now: datetime | None = None,
    ) -> EventPage:
        run = self._run_row(query_run_id)
        if limit < 1 or limit > 1000:
            raise ValueError("event page limit must be between 1 and 1000")
        current = now or _now()
        if last_event_id is not None:
            cursor = self.session.scalar(
                select(QueryEventModel).where(
                    QueryEventModel.id == last_event_id,
                    QueryEventModel.query_run_id == run.id,
                    QueryEventModel.tenant_id == self.tenant_id,
                )
            )
            if cursor is None:
                raise EventCursorInvalid("Last-Event-ID does not belong to this query run")
            if cursor.expires_at is not None and _aware(cursor.expires_at) <= current:
                raise EventCursorExpired("Last-Event-ID is outside event retention")
            if public_only and cursor.visibility != "public":
                raise EventCursorInvalid("Last-Event-ID is not visible to this subscriber")
            after_seq = cursor.seq
        criteria = [
            QueryEventModel.query_run_id == run.id,
            QueryEventModel.tenant_id == self.tenant_id,
            (QueryEventModel.expires_at.is_(None) | (QueryEventModel.expires_at > current)),
        ]
        if public_only:
            criteria.append(QueryEventModel.visibility == "public")
        first_available = self.session.scalar(
            select(func.min(QueryEventModel.seq)).where(*criteria)
        )
        last_available = (
            self.session.scalar(select(func.max(QueryEventModel.seq)).where(*criteria)) or 0
        )
        if first_available is not None and after_seq < first_available - 1:
            raise EventCursorExpired(
                "requested event sequence is outside the retained event window"
            )
        rows = self.session.scalars(
            select(QueryEventModel)
            .where(*criteria, QueryEventModel.seq > after_seq)
            .order_by(QueryEventModel.seq)
            .limit(limit)
        ).all()
        events = tuple(self._event(row) for row in rows)
        consumed_seq = events[-1].seq if events else after_seq
        return EventPage(
            query_run_id=run.id,
            after_seq=after_seq,
            events=events,
            first_available_seq=first_available,
            last_available_seq=last_available,
            terminal=(
                QueryRunStatus(run.status) in RUN_TERMINAL_STATUSES
                and consumed_seq >= last_available
            ),
        )

    def prune_expired_events(self, *, now: datetime | None = None) -> int:
        current = now or _now()
        result = self.session.execute(
            delete(QueryEventModel).where(
                QueryEventModel.tenant_id == self.tenant_id,
                QueryEventModel.query_run_id.in_(
                    select(QueryRunModel.id).where(
                        QueryRunModel.tenant_id == self.tenant_id,
                        QueryRunModel.user_id == self.user_id,
                    )
                ),
                QueryEventModel.expires_at <= current,
            )
        )
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------
    # Internal ordering and serialization
    # ------------------------------------------------------------------
    def _drain_when_a_later_turn_is_ready(
        self,
        conversation_id: str,
        terminal_turn_seq: int,
    ) -> None:
        later_ready = self.session.scalar(
            select(func.count(QueryTurnModel.id)).where(
                QueryTurnModel.conversation_id == conversation_id,
                QueryTurnModel.seq > terminal_turn_seq,
                QueryTurnModel.status == TurnStatus.READY_TO_COMMIT.value,
            )
        )
        if later_ready:
            self._drain_frontier(self._conversation_row(conversation_id))

    def _drain_frontier(self, conversation: QueryConversationModel) -> None:
        """按 seq 原子推进 resolved frontier；失败轮次不写入默认历史。"""

        while True:
            self.session.refresh(conversation)
            next_seq = conversation.resolved_turn_seq + 1
            turn = self.session.scalar(
                select(QueryTurnModel).where(
                    QueryTurnModel.conversation_id == conversation.id,
                    QueryTurnModel.seq == next_seq,
                )
            )
            if turn is None or turn.status in {
                TurnStatus.PENDING.value,
                TurnStatus.RUNNING.value,
            }:
                return
            status = TurnStatus(turn.status)
            if status not in TURN_TERMINAL_STATUSES and status != TurnStatus.READY_TO_COMMIT:
                return
            revision = conversation.revision
            now = _now()
            history_increment = 1 if status == TurnStatus.READY_TO_COMMIT else 0
            advanced = self.session.execute(
                update(QueryConversationModel)
                .where(
                    QueryConversationModel.id == conversation.id,
                    QueryConversationModel.tenant_id == self.tenant_id,
                    QueryConversationModel.user_id == self.user_id,
                    QueryConversationModel.revision == revision,
                    QueryConversationModel.resolved_turn_seq == next_seq - 1,
                )
                .values(
                    resolved_turn_seq=next_seq,
                    history_version=conversation.history_version + history_increment,
                    revision=revision + 1,
                    updated_at=now,
                )
            )
            if advanced.rowcount != 1:
                raise StaleQueryRevision(
                    "conversation commit frontier changed in another transaction"
                )
            if status == TurnStatus.READY_TO_COMMIT:
                result = dict(turn.ready_result_json or {})
                answer = str(result.get("answer") or "")
                turn.status = TurnStatus.COMMITTED.value
                turn.committed_at = now
                turn.resolved_at = now
                existing = self.session.scalar(
                    select(func.count(QueryMessageModel.id)).where(
                        QueryMessageModel.turn_id == turn.id
                    )
                )
                if existing:
                    raise ResourceConflictError("turn already owns messages before atomic commit")
                self.session.add_all(
                    [
                        QueryMessageModel(
                            id=new_runtime_id("msg"),
                            tenant_id=self.tenant_id,
                            conversation_id=conversation.id,
                            turn_id=turn.id,
                            query_run_id=turn.active_query_run_id,
                            turn_seq=turn.seq,
                            role="user",
                            status="committed",
                            content=turn.user_content,
                            ordinal=0,
                            metadata_json={},
                            created_at=turn.created_at,
                            committed_at=now,
                        ),
                        QueryMessageModel(
                            id=new_runtime_id("msg"),
                            tenant_id=self.tenant_id,
                            conversation_id=conversation.id,
                            turn_id=turn.id,
                            query_run_id=turn.active_query_run_id,
                            turn_seq=turn.seq,
                            role="assistant",
                            status="committed",
                            content=answer,
                            ordinal=1,
                            metadata_json={
                                key: value for key, value in result.items() if key != "answer"
                            },
                            created_at=now,
                            committed_at=now,
                        ),
                    ]
                )
            elif turn.resolved_at is None:
                turn.resolved_at = now
            self.session.flush()
            self.session.expire(conversation)

    def _conversation_row(self, conversation_id: str) -> QueryConversationModel:
        row = self.session.scalar(
            select(QueryConversationModel)
            .where(
                QueryConversationModel.id == conversation_id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError("query conversation not found")
        return row

    def _turn_row(self, turn_id: str) -> QueryTurnModel:
        row = self.session.scalar(
            select(QueryTurnModel)
            .join(
                QueryConversationModel,
                QueryConversationModel.id == QueryTurnModel.conversation_id,
            )
            .where(
                QueryTurnModel.id == turn_id,
                QueryConversationModel.tenant_id == self.tenant_id,
                QueryConversationModel.user_id == self.user_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError("query turn not found")
        return row

    def _run_row(self, query_run_id: str) -> QueryRunModel:
        row = self.session.scalar(
            select(QueryRunModel)
            .where(
                QueryRunModel.id == query_run_id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError("query run not found")
        return row

    @staticmethod
    def _conversation(row: QueryConversationModel) -> Conversation:
        return Conversation(
            conversation_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            kb_id=row.kb_id,
            title=row.title,
            status=row.status,
            revision=row.revision,
            next_turn_seq=row.next_turn_seq,
            resolved_turn_seq=row.resolved_turn_seq,
            history_version=row.history_version,
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
            archived_at=_aware(row.archived_at),
        )

    @staticmethod
    def _turn(row: QueryTurnModel) -> Turn:
        return Turn(
            turn_id=row.id,
            conversation_id=row.conversation_id,
            client_turn_id=row.client_turn_id,
            seq=row.seq,
            base_history_seq=row.base_history_seq,
            history_version=row.history_version,
            status=row.status,
            input_hash=row.input_hash,
            user_content=row.user_content,
            active_query_run_id=row.active_query_run_id,
            created_at=_aware(row.created_at),
            committed_at=_aware(row.committed_at),
            resolved_at=_aware(row.resolved_at),
        )

    @staticmethod
    def _message(row: QueryMessageModel) -> Message:
        return Message(
            message_id=row.id,
            conversation_id=row.conversation_id,
            turn_id=row.turn_id,
            query_run_id=row.query_run_id,
            role=row.role,
            status=row.status,
            content=row.content,
            ordinal=row.ordinal,
            metadata=row.metadata_json or {},
            created_at=_aware(row.created_at),
            committed_at=_aware(row.committed_at),
        )

    @staticmethod
    def _run(row: QueryRunModel) -> QueryRun:
        return QueryRun(
            query_run_id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            conversation_id=row.conversation_id,
            turn_id=row.turn_id,
            attempt=row.attempt,
            parent_query_run_id=row.parent_query_run_id,
            replay_of_query_run_id=row.replay_of_query_run_id,
            idempotency_key=row.idempotency_key,
            status=row.status,
            current_stage=row.current_stage,
            revision=row.revision,
            event_sequence=row.event_sequence,
            profile_version=row.profile_version,
            index_snapshot=row.index_snapshot_json or {},
            config_snapshot=row.config_snapshot_json or {},
            history_snapshot_hash=row.history_snapshot_hash,
            history_base_turn_seq=row.history_base_turn_seq,
            deadline_at=_aware(row.deadline_at),
            cancel_requested_at=_aware(row.cancel_requested_at),
            started_at=_aware(row.started_at),
            finished_at=_aware(row.finished_at),
            error_code=row.error_code,
            safe_error_message=row.safe_error_message,
            retryable=row.retryable,
            trace_id=row.trace_id,
            result=row.result_json,
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )

    @staticmethod
    def _event(row: QueryEventModel) -> QueryEvent:
        return QueryEvent(
            event_id=row.id,
            query_run_id=row.query_run_id,
            seq=row.seq,
            type=row.event_type,
            occurred_at=_aware(row.occurred_at),
            stage=row.stage,
            data=row.data_json or {},
            trace_id=row.trace_id,
            visibility=row.visibility,
            persistence_class=row.persistence_class,
            expires_at=_aware(row.expires_at),
            size_bytes=row.size_bytes,
        )

    @staticmethod
    def _snapshot_from_run(row: QueryRunModel) -> ConversationHistorySnapshot:
        if not row.history_snapshot_json:
            raise ResourceConflictError("query run has no frozen history snapshot")
        snapshot = ConversationHistorySnapshot.model_validate(row.history_snapshot_json)
        if snapshot.content_hash != row.history_snapshot_hash:
            raise ResourceConflictError("query run history snapshot hash mismatch")
        return snapshot


class QueryCheckpointRepository:
    """tenant/user-scoped encrypted checkpoint、attempt 与 lease/fence 仓储。"""

    def __init__(
        self,
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
    ) -> None:
        if not tenant_id.strip() or not user_id.strip():
            raise ValueError("query checkpoint repository requires tenant_id and user_id")
        self.session = session
        self.tenant_id = tenant_id.strip()
        self.user_id = user_id.strip()

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
        encrypted: EncryptedQueryCheckpoint,
        auth_type: str = "internal",
        credential_id: str = "internal",
        identity_version: int | None = None,
        roles_digest: str = "sha256:internal",
        scopes_digest: str = "sha256:internal",
    ) -> tuple[QueryExecutionRecord, bool]:
        """在任何节点开始前原子写入 execution 和 sequence=0 checkpoint。"""

        if not thread_id or len(thread_id) > 128:
            raise ValueError("query execution thread_id is invalid")
        if not next_node or len(next_node) > 64:
            raise ValueError("query execution entry node is invalid")
        self._owned_run_row(query_run_id)
        existing = self.session.scalar(
            select(QueryRunExecutionModel).where(
                QueryRunExecutionModel.query_run_id == query_run_id,
                QueryRunExecutionModel.tenant_id == self.tenant_id,
                QueryRunExecutionModel.user_id == self.user_id,
            )
        )
        if existing is not None:
            identity = (
                existing.auth_type,
                existing.credential_id,
                existing.identity_version,
                existing.roles_digest,
                existing.scopes_digest,
                existing.thread_id,
                existing.checkpoint_namespace,
                existing.workflow_version,
                existing.state_schema_version,
                existing.checkpoint_schema_version,
            )
            expected = (
                auth_type,
                credential_id,
                identity_version,
                roles_digest,
                scopes_digest,
                thread_id,
                checkpoint_namespace,
                workflow_version,
                state_schema_version,
                checkpoint_schema_version,
            )
            if identity != expected:
                raise ResourceConflictError(
                    "query execution already exists with incompatible identity or version"
                )
            return self._execution(existing), False

        execution = QueryRunExecutionModel(
            query_run_id=query_run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            auth_type=auth_type,
            credential_id=credential_id,
            identity_version=identity_version,
            roles_digest=roles_digest,
            scopes_digest=scopes_digest,
            thread_id=thread_id,
            checkpoint_namespace=checkpoint_namespace,
            workflow_version=workflow_version,
            state_schema_version=state_schema_version,
            checkpoint_schema_version=checkpoint_schema_version,
            status=QueryExecutionStatus.READY.value,
            revision=0,
            fencing_token=0,
            next_node=next_node,
            checkpoint_sequence=0,
            attempt_sequence=0,
            recovery_count=0,
        )
        self.session.add(execution)
        self.session.flush()
        checkpoint = QueryRunCheckpointModel(
            id=new_runtime_id("qcp"),
            tenant_id=self.tenant_id,
            query_run_id=query_run_id,
            sequence=0,
            checkpoint_namespace=checkpoint_namespace,
            workflow_version=workflow_version,
            state_schema_version=state_schema_version,
            checkpoint_schema_version=checkpoint_schema_version,
            completed_node=None,
            next_node=next_node,
            nonce=encrypted.nonce,
            ciphertext=encrypted.ciphertext,
            ciphertext_sha256=encrypted.ciphertext_sha256,
            size_bytes=encrypted.size_bytes,
        )
        self.session.add(checkpoint)
        self.session.flush()
        return self._execution(execution), True

    def get_execution(self, query_run_id: str) -> QueryExecutionRecord:
        return self._execution(self._execution_row(query_run_id))

    def latest_checkpoint(self, query_run_id: str) -> QueryCheckpointRecord:
        execution = self._execution_row(query_run_id)
        row = self.session.scalar(
            select(QueryRunCheckpointModel)
            .where(
                QueryRunCheckpointModel.query_run_id == execution.query_run_id,
                QueryRunCheckpointModel.tenant_id == self.tenant_id,
            )
            .order_by(QueryRunCheckpointModel.sequence.desc())
            .limit(1)
        )
        if row is None:
            raise QueryExecutionNotInitialized("query execution has no checkpoint")
        if row.sequence != execution.checkpoint_sequence:
            raise ResourceConflictError("query execution checkpoint cursor is inconsistent")
        return self._checkpoint(row)

    def latest_attempt(self, query_run_id: str) -> QueryNodeAttemptRecord | None:
        execution = self._execution_row(query_run_id)
        row = self.session.scalar(
            select(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
            )
            .order_by(QueryNodeAttemptModel.sequence.desc())
            .limit(1)
        )
        return None if row is None else self._attempt(row)

    def claim_execution(
        self,
        query_run_id: str,
        *,
        lease_owner: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> QueryExecutionLease:
        """用 revision CAS 取得 lease 并递增单调 fencing token。"""

        if not lease_owner or len(lease_owner) > 128:
            raise ValueError("query execution lease_owner is invalid")
        if lease_duration <= timedelta(0):
            raise ValueError("query execution lease duration must be positive")
        current = now or _now()
        if current.tzinfo is None:
            raise ValueError("query execution claim time must include timezone")
        row = self._execution_row(query_run_id)
        status = QueryExecutionStatus(row.status)
        if status in QUERY_EXECUTION_TERMINAL_STATUSES:
            raise ResourceConflictError("query execution is already terminal")
        expires = _aware(row.lease_expires_at)
        if row.lease_owner is not None and expires is not None and expires > current:
            raise QueryExecutionLeaseBusy("query execution lease is still active")
        raw_token = token_urlsafe(32)
        token_hash = self._lease_token_hash(raw_token)
        next_fence = row.fencing_token + 1
        next_revision = row.revision + 1
        lease_expires_at = current + lease_duration
        recovery_increment = 1 if row.fencing_token > 0 else 0
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(
                QueryRunExecutionModel.query_run_id == row.query_run_id,
                QueryRunExecutionModel.tenant_id == self.tenant_id,
                QueryRunExecutionModel.user_id == self.user_id,
                QueryRunExecutionModel.revision == row.revision,
                QueryRunExecutionModel.fencing_token == row.fencing_token,
                QueryRunExecutionModel.status == row.status,
                or_(
                    QueryRunExecutionModel.lease_owner.is_(None),
                    QueryRunExecutionModel.lease_expires_at.is_(None),
                    QueryRunExecutionModel.lease_expires_at <= current,
                ),
            )
            .execution_options(synchronize_session=False)
            .values(
                status=QueryExecutionStatus.RUNNING.value,
                revision=next_revision,
                fencing_token=next_fence,
                lease_owner=lease_owner,
                lease_token_hash=token_hash,
                lease_expires_at=lease_expires_at,
                heartbeat_at=current,
                recovery_count=row.recovery_count + recovery_increment,
                safe_error_code=None,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionLeaseBusy("query execution claim lost a CAS race")
        self.session.expire(row)
        return QueryExecutionLease(
            query_run_id=query_run_id,
            lease_owner=lease_owner,
            lease_token=raw_token,
            fencing_token=next_fence,
            revision=next_revision,
            expires_at=lease_expires_at,
        )

    def assert_lease(
        self,
        lease: QueryExecutionLease,
        *,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        row = self._execution_row(lease.query_run_id)
        self._assert_lease_row(row, lease, now=now)
        return self._execution(row)

    def schedule_attempt(
        self,
        lease: QueryExecutionLease,
        *,
        node_name: str,
        effect_class: str,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord:
        current = now or _now()
        row = self._execution_row(lease.query_run_id)
        self._assert_lease_row(row, lease, now=current)
        if row.next_node != node_name:
            raise ResourceConflictError("query execution next node changed before scheduling")
        latest = self._latest_attempt_row(row.query_run_id)
        if latest is not None and latest.status in {
            QueryNodeAttemptStatus.SCHEDULED.value,
            QueryNodeAttemptStatus.STARTED.value,
        }:
            raise ResourceConflictError("query execution already has an active node attempt")
        sequence = row.attempt_sequence + 1
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(row, lease))
            .values(
                attempt_sequence=sequence,
                revision=row.revision + 1,
                heartbeat_at=current,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed before scheduling")
        attempt = QueryNodeAttemptModel(
            id=new_runtime_id("qat"),
            tenant_id=self.tenant_id,
            query_run_id=row.query_run_id,
            sequence=sequence,
            node_name=node_name,
            effect_class=effect_class,
            status=QueryNodeAttemptStatus.SCHEDULED.value,
            checkpoint_sequence_before=row.checkpoint_sequence,
            fencing_token=lease.fencing_token,
            lease_owner=lease.lease_owner,
            revision=0,
            scheduled_at=current,
        )
        self.session.add(attempt)
        self.session.flush()
        return self._attempt(attempt)

    def mark_attempt_started(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord:
        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        attempt = self._attempt_row(lease.query_run_id, attempt_id)
        # STARTED 是允许进入有副作用 handler 的唯一授权点。必须先在同一事务内
        # CAS execution fence，消除 assert 后被取消/恢复者换 fence 的 TOCTOU。
        execution_changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                revision=execution.revision + 1,
                heartbeat_at=current,
                updated_at=current,
            )
        )
        if execution_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed before node start")
        changed = self.session.execute(
            update(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt.id,
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
                QueryNodeAttemptModel.status == QueryNodeAttemptStatus.SCHEDULED.value,
                QueryNodeAttemptModel.revision == attempt.revision,
                QueryNodeAttemptModel.fencing_token == lease.fencing_token,
                QueryNodeAttemptModel.lease_owner == lease.lease_owner,
            )
            .values(
                status=QueryNodeAttemptStatus.STARTED.value,
                started_at=current,
                revision=attempt.revision + 1,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query node attempt changed before start")
        self.session.expire(attempt)
        return self._attempt(attempt)

    def abandon_unstarted_attempt(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryNodeAttemptRecord:
        """新 fence 可安全放弃旧 scheduled attempt；其 handler 从未获准进入。"""

        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        attempt = self._attempt_row(lease.query_run_id, attempt_id)
        if attempt.status != QueryNodeAttemptStatus.SCHEDULED.value:
            raise ResourceConflictError("only a scheduled node attempt can be abandoned")
        changed = self.session.execute(
            update(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt.id,
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
                QueryNodeAttemptModel.status == QueryNodeAttemptStatus.SCHEDULED.value,
                QueryNodeAttemptModel.revision == attempt.revision,
            )
            .values(
                status=QueryNodeAttemptStatus.INTERRUPTED.value,
                safe_error_code="QUERY_ATTEMPT_ABANDONED_BEFORE_START",
                finished_at=current,
                revision=attempt.revision + 1,
            )
        )
        if changed.rowcount != 1:
            raise StaleQueryRevision("scheduled query node attempt changed during recovery")
        self.session.expire(attempt)
        return self._attempt(attempt)

    def commit_attempt_checkpoint(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        completed_node: str,
        next_node: str | None,
        encrypted: EncryptedQueryCheckpoint,
        now: datetime | None = None,
    ) -> tuple[QueryExecutionRecord, QueryCheckpointRecord, QueryNodeAttemptRecord]:
        """以同一事务提交 attempt=checkpointed、密文 state 与后继游标。"""

        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        attempt = self._attempt_row(lease.query_run_id, attempt_id)
        if execution.next_node != completed_node or attempt.node_name != completed_node:
            raise ResourceConflictError("query node/checkpoint boundary identity mismatch")
        if attempt.status != QueryNodeAttemptStatus.STARTED.value:
            raise ResourceConflictError("query node attempt is not started")
        if attempt.checkpoint_sequence_before != execution.checkpoint_sequence:
            raise ResourceConflictError("query node attempt started from a stale checkpoint")
        next_sequence = execution.checkpoint_sequence + 1
        checkpoint = QueryRunCheckpointModel(
            id=new_runtime_id("qcp"),
            tenant_id=self.tenant_id,
            query_run_id=execution.query_run_id,
            sequence=next_sequence,
            checkpoint_namespace=execution.checkpoint_namespace,
            workflow_version=execution.workflow_version,
            state_schema_version=execution.state_schema_version,
            checkpoint_schema_version=execution.checkpoint_schema_version,
            completed_node=completed_node,
            next_node=next_node,
            nonce=encrypted.nonce,
            ciphertext=encrypted.ciphertext,
            ciphertext_sha256=encrypted.ciphertext_sha256,
            size_bytes=encrypted.size_bytes,
            created_at=current,
        )
        attempt_changed = self.session.execute(
            update(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt.id,
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
                QueryNodeAttemptModel.status == QueryNodeAttemptStatus.STARTED.value,
                QueryNodeAttemptModel.revision == attempt.revision,
                QueryNodeAttemptModel.fencing_token == lease.fencing_token,
                QueryNodeAttemptModel.lease_owner == lease.lease_owner,
            )
            .values(
                status=QueryNodeAttemptStatus.CHECKPOINTED.value,
                checkpoint_sequence_after=next_sequence,
                checkpointed_at=current,
                finished_at=current,
                revision=attempt.revision + 1,
            )
        )
        if attempt_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query node attempt lost its fence before checkpoint")
        execution_changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                checkpoint_sequence=next_sequence,
                next_node=next_node,
                revision=execution.revision + 1,
                heartbeat_at=current,
                updated_at=current,
            )
        )
        if execution_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution lost its fence before checkpoint")
        self.session.add(checkpoint)
        self.session.flush()
        self.session.expire(execution)
        self.session.expire(attempt)
        return self._execution(execution), self._checkpoint(checkpoint), self._attempt(attempt)

    def interrupt_started_attempt(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        """恢复时终止未提交 active node；绝不调用该节点 handler。"""

        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        attempt = self._attempt_row(lease.query_run_id, attempt_id)
        if attempt.status != QueryNodeAttemptStatus.STARTED.value:
            raise ResourceConflictError("query node attempt is not an interrupted active node")
        attempt_changed = self.session.execute(
            update(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt.id,
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
                QueryNodeAttemptModel.status == QueryNodeAttemptStatus.STARTED.value,
                QueryNodeAttemptModel.revision == attempt.revision,
            )
            .values(
                status=QueryNodeAttemptStatus.INTERRUPTED.value,
                safe_error_code="QUERY_INTERRUPTED_DURING_NODE",
                finished_at=current,
                revision=attempt.revision + 1,
            )
        )
        if attempt_changed.rowcount != 1:
            raise StaleQueryRevision("active query node attempt changed during recovery")
        execution_changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                status=QueryExecutionStatus.INTERRUPTED.value,
                revision=execution.revision + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                safe_error_code="QUERY_INTERRUPTED_DURING_NODE",
                finished_at=current,
                updated_at=current,
            )
        )
        if execution_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed during interruption")
        self.session.expire(execution)
        return self._execution(execution)

    def mark_attempt_failed(
        self,
        lease: QueryExecutionLease,
        attempt_id: str,
        *,
        safe_error_code: str,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        if not safe_error_code or len(safe_error_code) > 128:
            raise ValueError("query node safe_error_code is invalid")
        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        attempt = self._attempt_row(lease.query_run_id, attempt_id)
        if attempt.status not in {
            QueryNodeAttemptStatus.SCHEDULED.value,
            QueryNodeAttemptStatus.STARTED.value,
        }:
            raise ResourceConflictError("query node attempt cannot transition to failed")
        attempt_changed = self.session.execute(
            update(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt.id,
                QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
                QueryNodeAttemptModel.status == attempt.status,
                QueryNodeAttemptModel.revision == attempt.revision,
                QueryNodeAttemptModel.fencing_token == lease.fencing_token,
            )
            .values(
                status=QueryNodeAttemptStatus.FAILED.value,
                safe_error_code=safe_error_code,
                finished_at=current,
                revision=attempt.revision + 1,
            )
        )
        if attempt_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query node attempt changed before failure")
        execution_changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                status=QueryExecutionStatus.FAILED.value,
                revision=execution.revision + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                safe_error_code=safe_error_code,
                finished_at=current,
                updated_at=current,
            )
        )
        if execution_changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed before failure")
        self.session.expire(execution)
        return self._execution(execution)

    def abort_execution(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus,
        safe_error_code: str,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        """在尚未进入节点或 checkpoint 无法读取时安全终止执行游标。"""

        if status not in {
            QueryExecutionStatus.FAILED,
            QueryExecutionStatus.CANCELLED,
            QueryExecutionStatus.TIMED_OUT,
        }:
            raise ValueError("query execution abort status is invalid")
        if not safe_error_code or len(safe_error_code) > 128:
            raise ValueError("query execution safe_error_code is invalid")
        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                status=status.value,
                revision=execution.revision + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                safe_error_code=safe_error_code,
                finished_at=current,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed before abort")
        self.session.expire(execution)
        return self._execution(execution)

    def finish_execution(
        self,
        lease: QueryExecutionLease,
        *,
        status: QueryExecutionStatus = QueryExecutionStatus.COMPLETED,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        """终态由可靠 API 在 QueryRun 终态提交后调用，仍受同一 fence 保护。"""

        if status not in {
            QueryExecutionStatus.COMPLETED,
            QueryExecutionStatus.CANCELLED,
            QueryExecutionStatus.TIMED_OUT,
        }:
            raise ValueError("query execution finish status is invalid")
        current = now or _now()
        execution = self._execution_row(lease.query_run_id)
        self._assert_lease_row(execution, lease, now=current)
        if status == QueryExecutionStatus.COMPLETED and execution.next_node is not None:
            raise ResourceConflictError("query execution cannot complete before the final boundary")
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(*self._lease_predicates(execution, lease))
            .values(
                status=status.value,
                revision=execution.revision + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                finished_at=current,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution fence changed before finish")
        self.session.expire(execution)
        return self._execution(execution)

    def reconcile_terminal_run(
        self,
        query_run_id: str,
        *,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        """以 QueryRun 终态为权威，原子 fence 掉任何迟到 worker。"""

        current = now or _now()
        run = self._owned_run_row(query_run_id)
        target_by_run = {
            QueryRunStatus.COMPLETED.value: QueryExecutionStatus.COMPLETED,
            QueryRunStatus.FAILED.value: QueryExecutionStatus.FAILED,
            QueryRunStatus.TIMED_OUT.value: QueryExecutionStatus.TIMED_OUT,
            QueryRunStatus.CANCELLED.value: QueryExecutionStatus.CANCELLED,
        }
        target = target_by_run.get(run.status)
        if target is None:
            raise ResourceConflictError("query run is not terminal")
        execution = self._execution_row(query_run_id)
        active_attempt = self._latest_attempt_row(query_run_id)
        if active_attempt is not None and active_attempt.status in {
            QueryNodeAttemptStatus.SCHEDULED.value,
            QueryNodeAttemptStatus.STARTED.value,
        }:
            attempt_changed = self.session.execute(
                update(QueryNodeAttemptModel)
                .where(
                    QueryNodeAttemptModel.id == active_attempt.id,
                    QueryNodeAttemptModel.query_run_id == execution.query_run_id,
                    QueryNodeAttemptModel.tenant_id == self.tenant_id,
                    QueryNodeAttemptModel.status == active_attempt.status,
                    QueryNodeAttemptModel.revision == active_attempt.revision,
                )
                .values(
                    status=QueryNodeAttemptStatus.INTERRUPTED.value,
                    safe_error_code="QUERY_ATTEMPT_TERMINAL_RECONCILED",
                    finished_at=current,
                    revision=active_attempt.revision + 1,
                )
            )
            if attempt_changed.rowcount != 1:
                raise StaleQueryRevision("query node attempt changed during terminal reconcile")
        # QueryRun 是公开终态权威；execution 还保留更精确的内部原因（例如
        # interrupted active node）。已有无 lease 的 execution 终态无需抹平。
        preserves_interruption = (
            run.status == QueryRunStatus.FAILED.value
            and run.error_code
            in {
                "QUERY_INTERRUPTED_DURING_NODE",
                "QUERY_INTERRUPTED_DURING_SHUTDOWN",
            }
            and execution.status == QueryExecutionStatus.INTERRUPTED.value
        )
        if (
            (execution.status == target.value or preserves_interruption)
            and execution.lease_owner is None
            and execution.next_node is None
        ):
            return self._execution(execution)
        execution_target = QueryExecutionStatus.INTERRUPTED if preserves_interruption else target
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(
                QueryRunExecutionModel.query_run_id == execution.query_run_id,
                QueryRunExecutionModel.tenant_id == self.tenant_id,
                QueryRunExecutionModel.user_id == self.user_id,
                QueryRunExecutionModel.revision == execution.revision,
                QueryRunExecutionModel.fencing_token == execution.fencing_token,
            )
            .execution_options(synchronize_session=False)
            .values(
                status=execution_target.value,
                revision=execution.revision + 1,
                fencing_token=execution.fencing_token + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                next_node=None,
                safe_error_code=run.error_code,
                finished_at=current,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution changed during terminal reconcile")
        self.session.expire(execution)
        return self._execution(execution)

    def interrupt_owned_execution(
        self,
        query_run_id: str,
        *,
        lease_owner: str,
        safe_error_code: str,
        now: datetime | None = None,
    ) -> QueryExecutionRecord:
        """进程关闭时按精确 owner fence 自己的 lease，不需要回显 lease token。"""

        if not lease_owner or not safe_error_code:
            raise ValueError("query execution shutdown identity is incomplete")
        current = now or _now()
        execution = self._execution_row(query_run_id)
        if (
            execution.status == QueryExecutionStatus.INTERRUPTED.value
            and execution.lease_owner is None
        ):
            return self._execution(execution)
        changed = self.session.execute(
            update(QueryRunExecutionModel)
            .where(
                QueryRunExecutionModel.query_run_id == execution.query_run_id,
                QueryRunExecutionModel.tenant_id == self.tenant_id,
                QueryRunExecutionModel.user_id == self.user_id,
                QueryRunExecutionModel.status == QueryExecutionStatus.RUNNING.value,
                QueryRunExecutionModel.lease_owner == lease_owner,
                QueryRunExecutionModel.revision == execution.revision,
                QueryRunExecutionModel.fencing_token == execution.fencing_token,
            )
            .execution_options(synchronize_session=False)
            .values(
                status=QueryExecutionStatus.INTERRUPTED.value,
                revision=execution.revision + 1,
                fencing_token=execution.fencing_token + 1,
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
                heartbeat_at=current,
                next_node=None,
                safe_error_code=safe_error_code,
                finished_at=current,
                updated_at=current,
            )
        )
        if changed.rowcount != 1:
            raise QueryExecutionFenceLost("query execution changed during shutdown interruption")
        self.session.expire(execution)
        return self._execution(execution)

    def _owned_run_row(self, query_run_id: str) -> QueryRunModel:
        row = self.session.scalar(
            select(QueryRunModel).where(
                QueryRunModel.id == query_run_id,
                QueryRunModel.tenant_id == self.tenant_id,
                QueryRunModel.user_id == self.user_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError("query run not found")
        return row

    def _execution_row(self, query_run_id: str) -> QueryRunExecutionModel:
        row = self.session.scalar(
            select(QueryRunExecutionModel)
            .where(
                QueryRunExecutionModel.query_run_id == query_run_id,
                QueryRunExecutionModel.tenant_id == self.tenant_id,
                QueryRunExecutionModel.user_id == self.user_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            self._owned_run_row(query_run_id)
            raise QueryExecutionNotInitialized("query execution is not initialized")
        return row

    def _attempt_row(self, query_run_id: str, attempt_id: str) -> QueryNodeAttemptModel:
        row = self.session.scalar(
            select(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.id == attempt_id,
                QueryNodeAttemptModel.query_run_id == query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError("query node attempt not found")
        return row

    def _latest_attempt_row(self, query_run_id: str) -> QueryNodeAttemptModel | None:
        return self.session.scalar(
            select(QueryNodeAttemptModel)
            .where(
                QueryNodeAttemptModel.query_run_id == query_run_id,
                QueryNodeAttemptModel.tenant_id == self.tenant_id,
            )
            .order_by(QueryNodeAttemptModel.sequence.desc())
            .limit(1)
            .execution_options(populate_existing=True)
        )

    def _assert_lease_row(
        self,
        row: QueryRunExecutionModel,
        lease: QueryExecutionLease,
        *,
        now: datetime | None = None,
    ) -> None:
        current = now or _now()
        expires = _aware(row.lease_expires_at)
        token_matches = row.lease_token_hash is not None and hmac.compare_digest(
            row.lease_token_hash,
            self._lease_token_hash(lease.lease_token),
        )
        if (
            row.status != QueryExecutionStatus.RUNNING.value
            or row.lease_owner != lease.lease_owner
            or row.fencing_token != lease.fencing_token
            or not token_matches
            or expires is None
            or expires <= current
        ):
            raise QueryExecutionFenceLost("query execution lease or fence is no longer valid")

    def _lease_predicates(
        self,
        row: QueryRunExecutionModel,
        lease: QueryExecutionLease,
    ) -> tuple[Any, ...]:
        return (
            QueryRunExecutionModel.query_run_id == row.query_run_id,
            QueryRunExecutionModel.tenant_id == self.tenant_id,
            QueryRunExecutionModel.user_id == self.user_id,
            QueryRunExecutionModel.status == QueryExecutionStatus.RUNNING.value,
            QueryRunExecutionModel.revision == row.revision,
            QueryRunExecutionModel.fencing_token == lease.fencing_token,
            QueryRunExecutionModel.lease_owner == lease.lease_owner,
            QueryRunExecutionModel.lease_token_hash == self._lease_token_hash(lease.lease_token),
        )

    @staticmethod
    def _lease_token_hash(raw_token: str) -> str:
        return f"sha256:{sha256(raw_token.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _execution(row: QueryRunExecutionModel) -> QueryExecutionRecord:
        return QueryExecutionRecord(
            query_run_id=row.query_run_id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            auth_type=row.auth_type,
            credential_id=row.credential_id,
            identity_version=row.identity_version,
            roles_digest=row.roles_digest,
            scopes_digest=row.scopes_digest,
            thread_id=row.thread_id,
            checkpoint_namespace=row.checkpoint_namespace,
            workflow_version=row.workflow_version,
            state_schema_version=row.state_schema_version,
            checkpoint_schema_version=row.checkpoint_schema_version,
            status=QueryExecutionStatus(row.status),
            revision=row.revision,
            fencing_token=row.fencing_token,
            lease_owner=row.lease_owner,
            lease_expires_at=_aware(row.lease_expires_at),
            heartbeat_at=_aware(row.heartbeat_at),
            next_node=row.next_node,
            checkpoint_sequence=row.checkpoint_sequence,
            attempt_sequence=row.attempt_sequence,
            recovery_count=row.recovery_count,
            safe_error_code=row.safe_error_code,
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
            finished_at=_aware(row.finished_at),
        )

    @staticmethod
    def _checkpoint(row: QueryRunCheckpointModel) -> QueryCheckpointRecord:
        return QueryCheckpointRecord(
            checkpoint_id=row.id,
            query_run_id=row.query_run_id,
            sequence=row.sequence,
            checkpoint_namespace=row.checkpoint_namespace,
            workflow_version=row.workflow_version,
            state_schema_version=row.state_schema_version,
            checkpoint_schema_version=row.checkpoint_schema_version,
            completed_node=row.completed_node,
            next_node=row.next_node,
            nonce=bytes(row.nonce),
            ciphertext=bytes(row.ciphertext),
            ciphertext_sha256=row.ciphertext_sha256,
            size_bytes=row.size_bytes,
            created_at=_aware(row.created_at),
        )

    @staticmethod
    def _attempt(row: QueryNodeAttemptModel) -> QueryNodeAttemptRecord:
        return QueryNodeAttemptRecord(
            attempt_id=row.id,
            query_run_id=row.query_run_id,
            sequence=row.sequence,
            node_name=row.node_name,
            effect_class=row.effect_class,
            status=QueryNodeAttemptStatus(row.status),
            checkpoint_sequence_before=row.checkpoint_sequence_before,
            checkpoint_sequence_after=row.checkpoint_sequence_after,
            fencing_token=row.fencing_token,
            lease_owner=row.lease_owner,
            revision=row.revision,
            safe_error_code=row.safe_error_code,
            scheduled_at=_aware(row.scheduled_at),
            started_at=_aware(row.started_at),
            checkpointed_at=_aware(row.checkpointed_at),
            finished_at=_aware(row.finished_at),
        )


__all__ = [
    "EventCursorExpired",
    "EventCursorInvalid",
    "QueryCheckpointRepository",
    "QueryExecutionFenceLost",
    "QueryExecutionLeaseBusy",
    "QueryExecutionNotInitialized",
    "QueryRuntimeRepository",
    "StaleQueryRevision",
]
