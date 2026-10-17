"""TODO-17 可恢复、幂等的数据源增量同步核心。

本模块只编排 ConnectorPort、控制库仓储端口与 DocumentVersionBoundary。后者必须由
TODO-01/TODO-02 的正式文档版本与持久任务队列实现；连接器不得直接写向量索引。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import hashlib
from typing import Protocol, runtime_checkable
from uuid import uuid4

from app.domain.connectors import (
    ChangePage,
    ConnectorCapabilities,
    ConnectorScope,
    ConnectorValidation,
    DataSource,
    DataSourceStatus,
    DeletePropagationPolicy,
    DeletePropagationReceipt,
    DocumentVersionReceipt,
    FetchedSourceObject,
    SourceBindingStatus,
    SourceChange,
    SourceChangeKind,
    SourceItem,
    SourceItemBinding,
    SourceSnapshot,
    SourceTombstone,
    SyncAction,
    SyncCursor,
    SyncEvent,
    SyncEventStatus,
    SyncOperation,
    SyncPlan,
    SyncRun,
    SyncRunStatus,
    canonical_sha256,
    sync_idempotency_key,
    utc_now,
)


class ConnectorSyncError(RuntimeError):
    code = "CONNECTOR_SYNC_ERROR"


class ConnectorDisabledError(ConnectorSyncError):
    code = "CONNECTOR_DISABLED"


class ExternalConnectorDisabledError(ConnectorSyncError):
    code = "EXTERNAL_CONNECTORS_DISABLED"


class ConnectorConfigurationError(ConnectorSyncError):
    code = "CONNECTOR_CONFIG_INVALID"


class ConnectorScopeError(ConnectorSyncError):
    code = "CONNECTOR_SCOPE_MISMATCH"


class EventConflictError(ConnectorSyncError):
    code = "SYNC_EVENT_CONFLICT"


class ContentIntegrityError(ConnectorSyncError):
    code = "SOURCE_CONTENT_INTEGRITY_ERROR"


class SourceObjectPolicyError(ConnectorSyncError):
    code = "SOURCE_OBJECT_POLICY_REJECTED"


class ReconciliationSafetyError(ConnectorSyncError):
    code = "RECONCILIATION_SAFETY_STOP"


class ConnectorNotFoundError(ConnectorSyncError):
    code = "CONNECTOR_NOT_FOUND"


class ConnectorRateLimitedError(ConnectorSyncError):
    code = "CONNECTOR_RATE_LIMITED"

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            f"connector rate limited; retry after {self.retry_after_seconds:g} seconds"
        )


@runtime_checkable
class ConnectorPort(Protocol):
    @property
    def capabilities(self) -> ConnectorCapabilities: ...

    def validate_config(self, data_source: DataSource) -> ConnectorValidation: ...

    def list_changes(
        self,
        data_source: DataSource,
        cursor: SyncCursor,
        limit: int,
    ) -> ChangePage: ...

    def fetch(
        self,
        data_source: DataSource,
        item: SourceItem,
    ) -> FetchedSourceObject: ...

    def ack(self, data_source: DataSource, cursor: SyncCursor) -> None: ...

    def snapshot(self, data_source: DataSource) -> SourceSnapshot: ...


@runtime_checkable
class ConnectorSyncRepository(Protocol):
    """控制库端口。

    ``commit_page`` 必须在一个事务中写入事件、binding 与 cursor；任一唯一约束冲突都应
    整体回滚。所有读取方法必须使用完整 scope，不提供无 tenant/KB 的管理旁路。
    """

    def get_cursor(self, scope: ConnectorScope, data_source_id: str) -> SyncCursor: ...

    def list_bindings(
        self,
        scope: ConnectorScope,
        data_source_id: str,
    ) -> tuple[SourceItemBinding, ...]: ...

    def list_events(
        self,
        scope: ConnectorScope,
        data_source_id: str,
    ) -> tuple[SyncEvent, ...]: ...

    def find_event(
        self,
        scope: ConnectorScope,
        data_source_id: str,
        event_id: str,
    ) -> SyncEvent | None: ...

    def create_run(self, run: SyncRun) -> None: ...

    def complete_run(self, run: SyncRun) -> None: ...

    def fail_run(self, run: SyncRun) -> None: ...

    def commit_page(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        expected_cursor: SyncCursor,
        next_cursor: SyncCursor,
        bindings: Iterable[SourceItemBinding],
        events: Iterable[SyncEvent],
    ) -> None: ...

    def replace_binding(self, binding: SourceItemBinding) -> None: ...


@runtime_checkable
class DocumentVersionBoundary(Protocol):
    """正式 DocumentVersion/IngestionJob 路径的防腐层。"""

    def create_version_and_job(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        item: SourceItem,
        fetched: FetchedSourceObject,
        content_sha256: str,
        existing_document_id: str | None,
        idempotency_key: str,
        sync_run_id: str,
    ) -> DocumentVersionReceipt: ...

    def update_metadata(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        document_id: str,
        item: SourceItem,
        idempotency_key: str,
        sync_run_id: str,
    ) -> None: ...

    def propagate_delete(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        document_id: str,
        policy: DeletePropagationPolicy,
        purge_after: datetime,
        idempotency_key: str,
        sync_run_id: str,
    ) -> DeletePropagationReceipt: ...


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, ConnectorPort] = {}

    def register(self, key: str, connector: ConnectorPort) -> None:
        normalized = key.strip()
        if not normalized:
            raise ValueError("connector key must not be empty")
        if normalized in self._connectors:
            raise ValueError(f"connector already registered: {normalized}")
        if not isinstance(connector, ConnectorPort):
            raise TypeError("connector does not satisfy ConnectorPort")
        self._connectors[normalized] = connector

    def get(self, key: str) -> ConnectorPort:
        try:
            return self._connectors[key]
        except KeyError as exc:
            raise ConnectorNotFoundError(f"connector is not registered: {key}") from exc


class IncrementalSyncPlanner:
    version = "connector-sync-planner/1.0.0"

    @staticmethod
    def _plan_hash(
        *,
        source: DataSource,
        cursor: SyncCursor,
        to_cursor: str,
        reconciliation: bool,
        operations: Iterable[SyncOperation],
    ) -> str:
        return canonical_sha256(
            {
                "planner": IncrementalSyncPlanner.version,
                "scope": source.scope.model_dump(mode="json"),
                "data_source_id": source.data_source_id,
                "config_version": source.config_version,
                "from_cursor": cursor.value,
                "to_cursor": to_cursor,
                "reconciliation": reconciliation,
                "operations": [
                    operation.model_dump(
                        mode="json",
                        exclude={
                            "ordinal": True,
                            "prior_binding": True,
                            "change": {"occurred_at"},
                        },
                    )
                    for operation in operations
                ],
            }
        )

    @staticmethod
    def _action_for(
        source: DataSource,
        change: SourceChange,
        binding: SourceItemBinding | None,
    ) -> tuple[SyncAction, str]:
        if change.kind == SourceChangeKind.DELETE:
            if binding is None or binding.status == SourceBindingStatus.PURGED:
                return SyncAction.NOOP, "remote tombstone has no live local binding"
            mapping = {
                DeletePropagationPolicy.DISABLE: SyncAction.DISABLE_DOCUMENT,
                DeletePropagationPolicy.SOFT_DELETE: SyncAction.SOFT_DELETE_DOCUMENT,
                DeletePropagationPolicy.PURGE: SyncAction.PURGE_DOCUMENT,
            }
            return mapping[source.sync_policy.delete_policy], (
                f"propagate remote deletion using {source.sync_policy.delete_policy.value}"
            )

        item = change.item
        assert item is not None
        if binding is None or binding.status in {
            SourceBindingStatus.PURGED,
            SourceBindingStatus.SOFT_DELETED,
        }:
            return SyncAction.CREATE_VERSION, "new or previously removed source item"

        same_hash = bool(
            item.content_sha256
            and binding.content_sha256
            and item.content_sha256 == binding.content_sha256
        )
        same_version = item.remote_version == binding.remote_version
        same_metadata = (
            item.logical_name == binding.logical_name and item.locator == binding.locator
        )
        if same_hash and same_metadata:
            return SyncAction.NOOP, "content hash and tracked metadata are unchanged"
        if same_hash and not same_metadata:
            return SyncAction.UPDATE_METADATA, "content unchanged; name or locator changed"
        if same_version and same_metadata:
            return SyncAction.NOOP, "remote version and tracked metadata are unchanged"
        return SyncAction.UPDATE_VERSION, "remote version or content hash changed"

    def plan_incremental(
        self,
        *,
        source: DataSource,
        cursor: SyncCursor,
        page: ChangePage,
        bindings: Iterable[SourceItemBinding],
        prior_events: dict[str, SyncEvent] | None = None,
        run_id: str,
        dry_run: bool = False,
    ) -> SyncPlan:
        self._validate_scope(source, cursor, bindings)
        binding_map = {binding.source_item_id: binding for binding in bindings}
        prior_events = prior_events or {}
        deduplicated: dict[str, SourceChange] = {}
        for change in sorted(page.changes, key=lambda item: (item.sequence, item.event_id)):
            existing = deduplicated.get(change.event_id)
            if existing is not None and existing.fingerprint != change.fingerprint:
                raise EventConflictError(
                    f"event {change.event_id!r} was reused with different payload"
                )
            deduplicated.setdefault(change.event_id, change)

        operations: list[SyncOperation] = []
        for ordinal, change in enumerate(deduplicated.values()):
            prior = prior_events.get(change.event_id)
            if prior is not None:
                if prior.event_fingerprint != change.fingerprint:
                    raise EventConflictError(
                        f"persisted event {change.event_id!r} has a different fingerprint"
                    )
                action, reason = (
                    SyncAction.ALREADY_APPLIED,
                    "event was committed by an earlier sync run",
                )
            else:
                action, reason = self._action_for(
                    source,
                    change,
                    binding_map.get(change.source_item_id),
                )
            operations.append(
                SyncOperation(
                    ordinal=ordinal,
                    event_id=change.event_id,
                    event_fingerprint=change.fingerprint,
                    source_item_id=change.source_item_id,
                    action=action,
                    reason=reason,
                    idempotency_key=sync_idempotency_key(
                        source.data_source_id,
                        change,
                    ),
                    change=change,
                    prior_binding=binding_map.get(change.source_item_id),
                )
            )

        plan_hash = self._plan_hash(
            source=source,
            cursor=cursor,
            to_cursor=page.next_cursor,
            reconciliation=False,
            operations=operations,
        )
        return SyncPlan(
            scope=source.scope,
            data_source_id=source.data_source_id,
            run_id=run_id,
            from_cursor=cursor.value,
            to_cursor=page.next_cursor,
            has_more=page.has_more,
            dry_run=dry_run,
            operations=tuple(operations),
            plan_hash=plan_hash,
        )

    def plan_reconciliation(
        self,
        *,
        source: DataSource,
        cursor: SyncCursor,
        snapshot: SourceSnapshot,
        bindings: Iterable[SourceItemBinding],
        prior_events: dict[str, SyncEvent] | None = None,
        run_id: str,
        dry_run: bool = False,
    ) -> SyncPlan:
        binding_tuple = tuple(bindings)
        self._validate_scope(source, cursor, binding_tuple)
        if not snapshot.complete:
            raise ReconciliationSafetyError(
                "incomplete snapshot cannot drive deletion reconciliation"
            )
        if (
            not snapshot.items
            and binding_tuple
            and not source.sync_policy.allow_empty_snapshot_deletes
        ):
            raise ReconciliationSafetyError("empty snapshot deletion is blocked by policy")

        item_map = {item.source_item_id: item for item in snapshot.items}
        changes: list[SourceChange] = []
        for sequence, item in enumerate(
            sorted(snapshot.items, key=lambda value: value.source_item_id)
        ):
            fingerprint = canonical_sha256(
                {
                    "source": source.data_source_id,
                    "cursor": snapshot.cursor,
                    "item": item.model_dump(mode="json"),
                }
            )
            changes.append(
                SourceChange(
                    event_id=f"reconcile:{fingerprint[:32]}",
                    sequence=sequence,
                    kind=SourceChangeKind.UPDATE,
                    item=item,
                    occurred_at=snapshot.observed_at,
                )
            )
        next_sequence = len(changes)
        for binding in sorted(binding_tuple, key=lambda value: value.source_item_id):
            if binding.source_item_id in item_map:
                continue
            fingerprint = canonical_sha256(
                {
                    "source": source.data_source_id,
                    "cursor": snapshot.cursor,
                    "missing": binding.source_item_id,
                    "remote_version": binding.remote_version,
                }
            )
            changes.append(
                SourceChange(
                    event_id=f"reconcile:{fingerprint[:32]}",
                    sequence=next_sequence,
                    kind=SourceChangeKind.DELETE,
                    tombstone=SourceTombstone(
                        source_item_id=binding.source_item_id,
                        remote_version=binding.remote_version,
                        prior_locator=binding.locator,
                        reason="missing_from_complete_snapshot",
                    ),
                    occurred_at=snapshot.observed_at,
                )
            )
            next_sequence += 1

        incremental = self.plan_incremental(
            source=source,
            cursor=cursor,
            page=ChangePage(changes=tuple(changes), next_cursor=snapshot.cursor),
            bindings=binding_tuple,
            prior_events=prior_events,
            run_id=run_id,
            dry_run=dry_run,
        )
        plan_hash = self._plan_hash(
            source=source,
            cursor=cursor,
            to_cursor=snapshot.cursor,
            reconciliation=True,
            operations=incremental.operations,
        )
        return incremental.model_copy(update={"reconciliation": True, "plan_hash": plan_hash})

    @staticmethod
    def _validate_scope(
        source: DataSource,
        cursor: SyncCursor,
        bindings: Iterable[SourceItemBinding],
    ) -> None:
        if cursor.scope != source.scope or cursor.data_source_id != source.data_source_id:
            raise ConnectorScopeError("cursor is outside the data source scope")
        for binding in bindings:
            if binding.scope != source.scope or binding.data_source_id != source.data_source_id:
                raise ConnectorScopeError("binding is outside the data source scope")


class ConnectorSyncService:
    """执行单个 change page；调度器可按 ``has_more`` 重复调用。"""

    def __init__(
        self,
        *,
        registry: ConnectorRegistry,
        repository: ConnectorSyncRepository,
        document_versions: DocumentVersionBoundary,
        external_connectors_enabled: bool = False,
        planner: IncrementalSyncPlanner | None = None,
    ) -> None:
        self._registry = registry
        self._repository = repository
        self._document_versions = document_versions
        self._external_connectors_enabled = external_connectors_enabled
        self._planner = planner or IncrementalSyncPlanner()

    def dry_run(
        self,
        source: DataSource,
        *,
        reconciliation: bool = False,
        run_id: str | None = None,
    ) -> SyncPlan:
        connector = self._connector_for(source)
        cursor, bindings, prior_events = self._load_state(source)
        resolved_run_id = run_id or f"dry_{uuid4().hex}"
        if reconciliation:
            snapshot = connector.snapshot(source)
            return self._planner.plan_reconciliation(
                source=source,
                cursor=cursor,
                snapshot=snapshot,
                bindings=bindings,
                prior_events=prior_events,
                run_id=resolved_run_id,
                dry_run=True,
            )
        page = connector.list_changes(
            source,
            cursor,
            source.sync_policy.max_changes_per_run,
        )
        self._validate_page_progress(cursor, page)
        return self._planner.plan_incremental(
            source=source,
            cursor=cursor,
            page=page,
            bindings=bindings,
            prior_events=prior_events,
            run_id=resolved_run_id,
            dry_run=True,
        )

    def run_once(
        self,
        source: DataSource,
        *,
        reconciliation: bool = False,
        run_id: str | None = None,
    ) -> tuple[SyncRun, SyncPlan]:
        resolved_run_id = run_id or f"sync_{uuid4().hex}"
        cursor = self._repository.get_cursor(source.scope, source.data_source_id)
        run = SyncRun(
            run_id=resolved_run_id,
            scope=source.scope,
            data_source_id=source.data_source_id,
            status=SyncRunStatus.RUNNING,
            from_cursor=cursor.value,
        )
        self._repository.create_run(run)
        events: list[SyncEvent] = []
        changed_bindings: list[SourceItemBinding] = []
        committed = False
        try:
            connector = self._connector_for(source)
            loaded_cursor, bindings, prior_events = self._load_state(source)
            if loaded_cursor != cursor:
                raise ConnectorSyncError("sync cursor changed before planning")
            if reconciliation:
                snapshot = connector.snapshot(source)
                plan = self._planner.plan_reconciliation(
                    source=source,
                    cursor=cursor,
                    snapshot=snapshot,
                    bindings=bindings,
                    prior_events=prior_events,
                    run_id=resolved_run_id,
                )
            else:
                page = connector.list_changes(
                    source,
                    cursor,
                    source.sync_policy.max_changes_per_run,
                )
                self._validate_page_progress(cursor, page)
                plan = self._planner.plan_incremental(
                    source=source,
                    cursor=cursor,
                    page=page,
                    bindings=bindings,
                    prior_events=prior_events,
                    run_id=resolved_run_id,
                )
            run = run.model_copy(update={"plan_hash": plan.plan_hash})
            for operation in plan.operations:
                event, binding = self._execute_operation(
                    source=source,
                    connector=connector,
                    operation=operation,
                    run_id=resolved_run_id,
                )
                events.append(event)
                if binding is not None:
                    changed_bindings.append(binding)

            next_cursor = cursor.advance(plan.to_cursor)
            self._repository.commit_page(
                scope=source.scope,
                data_source_id=source.data_source_id,
                expected_cursor=cursor,
                next_cursor=next_cursor,
                bindings=changed_bindings,
                events=events,
            )
            committed = True
            # 远端 ack 必须晚于控制库事务；ack 失败只会导致重放，event idempotency 会兜底。
            connector.ack(source, next_cursor)
            applied = sum(event.status == SyncEventStatus.APPLIED for event in events)
            skipped = len(events) - applied
            completed = run.model_copy(
                update={
                    "status": SyncRunStatus.SUCCEEDED,
                    "to_cursor": plan.to_cursor,
                    "applied_count": applied,
                    "skipped_count": skipped,
                    "finished_at": utc_now(),
                }
            )
            self._repository.complete_run(completed)
            return completed, plan
        except Exception as exc:
            failed = run.model_copy(
                update={
                    "status": (SyncRunStatus.PARTIAL if committed else SyncRunStatus.FAILED),
                    "to_cursor": plan.to_cursor if committed else None,
                    "applied_count": (
                        sum(event.status == SyncEventStatus.APPLIED for event in events)
                        if committed
                        else 0
                    ),
                    "skipped_count": (
                        sum(event.status == SyncEventStatus.SKIPPED for event in events)
                        if committed
                        else 0
                    ),
                    "failed_count": 1,
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "error_detail": str(exc)[:1000],
                    "finished_at": utc_now(),
                }
            )
            self._repository.fail_run(failed)
            raise

    def mark_version_published(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        source_item_id: str,
        document_version_id: str,
    ) -> SourceItemBinding:
        """由 TODO-03 发布成功事件调用，禁止在 fetch 完成时提前标 active。"""

        bindings = {
            binding.source_item_id: binding
            for binding in self._repository.list_bindings(scope, data_source_id)
        }
        binding = bindings.get(source_item_id)
        if binding is None:
            raise ConnectorScopeError("binding not found in the requested scope")
        if binding.pending_document_version_id != document_version_id:
            raise ConnectorScopeError(
                "published version does not match binding.pending_document_version_id"
            )
        published = binding.model_copy(
            update={
                "active_document_version_id": document_version_id,
                "pending_document_version_id": None,
                "status": SourceBindingStatus.ACTIVE,
            }
        )
        self._repository.replace_binding(published)
        return published

    def _connector_for(self, source: DataSource) -> ConnectorPort:
        if source.status != DataSourceStatus.ACTIVE:
            raise ConnectorDisabledError("data source is not active")
        connector = self._registry.get(source.connector_key)
        if connector.capabilities.external_network and not self._external_connectors_enabled:
            raise ExternalConnectorDisabledError(
                "external connector execution is disabled by default"
            )
        validation = connector.validate_config(source)
        if not validation.valid:
            raise ConnectorConfigurationError("; ".join(validation.errors))
        if source.sync_policy.map_remote_acl and not connector.capabilities.acl:
            raise ConnectorConfigurationError(
                "data source requires ACL mapping but connector does not declare ACL support"
            )
        if source.sync_policy.allow_partial_cursor_advance:
            raise ConnectorConfigurationError(
                "partial cursor advance requires a durable failed-event retry ledger; "
                "the strict core only advances after the complete page succeeds"
            )
        return connector

    def _load_state(
        self,
        source: DataSource,
    ) -> tuple[SyncCursor, tuple[SourceItemBinding, ...], dict[str, SyncEvent]]:
        cursor = self._repository.get_cursor(source.scope, source.data_source_id)
        bindings = self._repository.list_bindings(
            source.scope,
            source.data_source_id,
        )
        prior_events = {
            event.event_id: event
            for event in self._repository.list_events(
                source.scope,
                source.data_source_id,
            )
            if event.status in {SyncEventStatus.APPLIED, SyncEventStatus.SKIPPED}
        }
        return cursor, bindings, prior_events

    @staticmethod
    def _validate_page_progress(cursor: SyncCursor, page: ChangePage) -> None:
        if page.has_more and page.next_cursor == cursor.value:
            raise ConnectorSyncError("connector returned has_more without advancing the cursor")

    def _execute_operation(
        self,
        *,
        source: DataSource,
        connector: ConnectorPort,
        operation: SyncOperation,
        run_id: str,
    ) -> tuple[SyncEvent, SourceItemBinding | None]:
        prior = self._repository.find_event(
            source.scope,
            source.data_source_id,
            operation.event_id,
        )
        if prior is not None:
            if prior.event_fingerprint != operation.event_fingerprint:
                raise EventConflictError(
                    f"event {operation.event_id!r} conflicts with persisted payload"
                )
            return (
                SyncEvent(
                    scope=source.scope,
                    data_source_id=source.data_source_id,
                    run_id=run_id,
                    event_id=operation.event_id,
                    event_fingerprint=operation.event_fingerprint,
                    source_item_id=operation.source_item_id,
                    action=SyncAction.ALREADY_APPLIED,
                    status=SyncEventStatus.SKIPPED,
                    idempotency_key=operation.idempotency_key,
                    document_id=prior.document_id,
                    document_version_id=prior.document_version_id,
                    ingestion_job_id=prior.ingestion_job_id,
                    detail="replayed event skipped",
                ),
                None,
            )

        if operation.action in {SyncAction.CREATE_VERSION, SyncAction.UPDATE_VERSION}:
            return self._upsert(source, connector, operation, run_id)
        if operation.action == SyncAction.UPDATE_METADATA:
            return self._metadata_update(source, operation, run_id)
        if operation.action in {
            SyncAction.DISABLE_DOCUMENT,
            SyncAction.SOFT_DELETE_DOCUMENT,
            SyncAction.PURGE_DOCUMENT,
        }:
            return self._delete(source, operation, run_id)
        return (
            SyncEvent(
                scope=source.scope,
                data_source_id=source.data_source_id,
                run_id=run_id,
                event_id=operation.event_id,
                event_fingerprint=operation.event_fingerprint,
                source_item_id=operation.source_item_id,
                action=operation.action,
                status=SyncEventStatus.SKIPPED,
                idempotency_key=operation.idempotency_key,
                document_id=(
                    operation.prior_binding.document_id
                    if operation.prior_binding is not None
                    else None
                ),
                detail=operation.reason,
            ),
            (
                operation.prior_binding.model_copy(
                    update={
                        "last_seen_at": utc_now(),
                        **(
                            {
                                "remote_version": operation.change.item.remote_version,
                                "content_sha256": (
                                    operation.change.item.content_sha256
                                    or operation.prior_binding.content_sha256
                                ),
                                "logical_name": operation.change.item.logical_name,
                                "locator": operation.change.item.locator,
                            }
                            if operation.change.item is not None
                            else {}
                        ),
                    }
                )
                if operation.prior_binding is not None
                else None
            ),
        )

    def _upsert(
        self,
        source: DataSource,
        connector: ConnectorPort,
        operation: SyncOperation,
        run_id: str,
    ) -> tuple[SyncEvent, SourceItemBinding]:
        item = operation.change.item
        assert item is not None
        self._enforce_item_policy(source, connector.capabilities, item)
        fetched = connector.fetch(source, item)
        if (
            fetched.source_item_id != item.source_item_id
            or fetched.remote_version != item.remote_version
        ):
            raise ContentIntegrityError("fetch result identity/version differs from change")
        if fetched.mime_type.split(";", 1)[0].strip().lower() != item.mime_type:
            raise ContentIntegrityError("fetch result MIME differs from change")
        if len(fetched.content) > source.sync_policy.max_object_bytes:
            raise SourceObjectPolicyError("fetched object exceeds data-source size policy")
        if len(fetched.content) > connector.capabilities.max_object_bytes:
            raise SourceObjectPolicyError("fetched object exceeds connector capability")
        digest = hashlib.sha256(fetched.content).hexdigest()
        if item.content_sha256 is not None and digest != item.content_sha256:
            raise ContentIntegrityError("fetched content hash differs from remote change")
        receipt = self._document_versions.create_version_and_job(
            scope=source.scope,
            data_source_id=source.data_source_id,
            item=item,
            fetched=fetched,
            content_sha256=digest,
            existing_document_id=(
                operation.prior_binding.document_id if operation.prior_binding is not None else None
            ),
            idempotency_key=operation.idempotency_key,
            sync_run_id=run_id,
        )
        binding = SourceItemBinding(
            scope=source.scope,
            data_source_id=source.data_source_id,
            source_item_id=item.source_item_id,
            document_id=receipt.document_id,
            active_document_version_id=(
                operation.prior_binding.active_document_version_id
                if operation.prior_binding is not None
                else None
            ),
            pending_document_version_id=receipt.document_version_id,
            remote_version=item.remote_version,
            content_sha256=digest,
            logical_name=item.logical_name,
            locator=item.locator,
            status=SourceBindingStatus.PENDING,
        )
        return (
            SyncEvent(
                scope=source.scope,
                data_source_id=source.data_source_id,
                run_id=run_id,
                event_id=operation.event_id,
                event_fingerprint=operation.event_fingerprint,
                source_item_id=item.source_item_id,
                action=operation.action,
                status=SyncEventStatus.APPLIED,
                idempotency_key=operation.idempotency_key,
                document_id=receipt.document_id,
                document_version_id=receipt.document_version_id,
                ingestion_job_id=receipt.ingestion_job_id,
                detail="document version and ingestion job registered",
            ),
            binding,
        )

    def _metadata_update(
        self,
        source: DataSource,
        operation: SyncOperation,
        run_id: str,
    ) -> tuple[SyncEvent, SourceItemBinding]:
        item = operation.change.item
        binding = operation.prior_binding
        assert item is not None and binding is not None
        self._document_versions.update_metadata(
            scope=source.scope,
            data_source_id=source.data_source_id,
            document_id=binding.document_id,
            item=item,
            idempotency_key=operation.idempotency_key,
            sync_run_id=run_id,
        )
        changed = binding.model_copy(
            update={
                "remote_version": item.remote_version,
                "logical_name": item.logical_name,
                "locator": item.locator,
                "last_seen_at": utc_now(),
            }
        )
        return (
            SyncEvent(
                scope=source.scope,
                data_source_id=source.data_source_id,
                run_id=run_id,
                event_id=operation.event_id,
                event_fingerprint=operation.event_fingerprint,
                source_item_id=item.source_item_id,
                action=operation.action,
                status=SyncEventStatus.APPLIED,
                idempotency_key=operation.idempotency_key,
                document_id=binding.document_id,
                detail="document metadata updated without creating a content version",
            ),
            changed,
        )

    def _delete(
        self,
        source: DataSource,
        operation: SyncOperation,
        run_id: str,
    ) -> tuple[SyncEvent, SourceItemBinding]:
        binding = operation.prior_binding
        assert binding is not None
        purge_after = utc_now() + timedelta(seconds=source.sync_policy.deletion_grace_seconds)
        receipt = self._document_versions.propagate_delete(
            scope=source.scope,
            data_source_id=source.data_source_id,
            document_id=binding.document_id,
            policy=source.sync_policy.delete_policy,
            purge_after=purge_after,
            idempotency_key=operation.idempotency_key,
            sync_run_id=run_id,
        )
        changed = binding.model_copy(
            update={
                "status": receipt.state,
                "disabled_at": utc_now(),
                "purge_after": (
                    purge_after if receipt.state != SourceBindingStatus.PURGED else None
                ),
                "pending_document_version_id": None,
            }
        )
        return (
            SyncEvent(
                scope=source.scope,
                data_source_id=source.data_source_id,
                run_id=run_id,
                event_id=operation.event_id,
                event_fingerprint=operation.event_fingerprint,
                source_item_id=operation.source_item_id,
                action=operation.action,
                status=SyncEventStatus.APPLIED,
                idempotency_key=operation.idempotency_key,
                document_id=binding.document_id,
                detail=f"remote deletion propagated as {receipt.state.value}",
            ),
            changed,
        )

    @staticmethod
    def _enforce_item_policy(
        source: DataSource,
        capabilities: ConnectorCapabilities,
        item: SourceItem,
    ) -> None:
        if item.size_bytes is not None:
            if item.size_bytes > source.sync_policy.max_object_bytes:
                raise SourceObjectPolicyError("source item exceeds data-source size policy")
            if item.size_bytes > capabilities.max_object_bytes:
                raise SourceObjectPolicyError("source item exceeds connector capability")
        allowed = source.sync_policy.allowed_mime_types
        supported = capabilities.supported_mime_types
        if allowed and item.mime_type not in allowed:
            raise SourceObjectPolicyError(
                f"MIME {item.mime_type!r} is not allowed by data-source policy"
            )
        if supported and item.mime_type not in supported:
            raise SourceObjectPolicyError(f"MIME {item.mime_type!r} is not supported by connector")


__all__ = [
    "ConnectorConfigurationError",
    "ConnectorDisabledError",
    "ConnectorNotFoundError",
    "ConnectorPort",
    "ConnectorRegistry",
    "ConnectorRateLimitedError",
    "ConnectorScopeError",
    "ConnectorSyncError",
    "ConnectorSyncRepository",
    "ConnectorSyncService",
    "ContentIntegrityError",
    "DocumentVersionBoundary",
    "EventConflictError",
    "ExternalConnectorDisabledError",
    "IncrementalSyncPlanner",
    "ReconciliationSafetyError",
    "SourceObjectPolicyError",
]
