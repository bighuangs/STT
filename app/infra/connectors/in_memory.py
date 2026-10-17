"""TODO-17 端口的线程安全内存实现。

仅用于单元测试、本地演练和作为 SQL adapter 的可执行语义参考，不宣称跨进程持久化。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from threading import RLock

from app.application.connector_sync_service import (
    ConnectorScopeError,
    EventConflictError,
)
from app.domain.connectors import (
    ConnectorScope,
    DeletePropagationPolicy,
    DeletePropagationReceipt,
    DocumentVersionReceipt,
    FetchedSourceObject,
    SourceBindingStatus,
    SourceItem,
    SourceItemBinding,
    SyncCursor,
    SyncEvent,
    SyncRun,
    canonical_sha256,
)


def _scope_key(scope: ConnectorScope, data_source_id: str) -> tuple[str, str, str]:
    return scope.tenant_id, scope.kb_id, data_source_id


class InMemoryConnectorSyncRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._cursors: dict[tuple[str, str, str], SyncCursor] = {}
        self._bindings: dict[tuple[str, str, str, str], SourceItemBinding] = {}
        self._events: dict[tuple[str, str, str, str], SyncEvent] = {}
        self._runs: dict[tuple[str, str, str, str], SyncRun] = {}

    def get_cursor(
        self,
        scope: ConnectorScope,
        data_source_id: str,
    ) -> SyncCursor:
        with self._lock:
            key = _scope_key(scope, data_source_id)
            cursor = self._cursors.get(key)
            if cursor is None:
                cursor = SyncCursor(
                    scope=scope,
                    data_source_id=data_source_id,
                    value="0",
                )
            return cursor.model_copy(deep=True)

    def list_bindings(
        self,
        scope: ConnectorScope,
        data_source_id: str,
    ) -> tuple[SourceItemBinding, ...]:
        prefix = _scope_key(scope, data_source_id)
        with self._lock:
            return tuple(
                binding.model_copy(deep=True)
                for key, binding in sorted(self._bindings.items())
                if key[:3] == prefix
            )

    def list_events(
        self,
        scope: ConnectorScope,
        data_source_id: str,
    ) -> tuple[SyncEvent, ...]:
        prefix = _scope_key(scope, data_source_id)
        with self._lock:
            return tuple(
                event.model_copy(deep=True)
                for key, event in sorted(self._events.items())
                if key[:3] == prefix
            )

    def find_event(
        self,
        scope: ConnectorScope,
        data_source_id: str,
        event_id: str,
    ) -> SyncEvent | None:
        with self._lock:
            event = self._events.get((*_scope_key(scope, data_source_id), event_id))
            return event.model_copy(deep=True) if event is not None else None

    def create_run(self, run: SyncRun) -> None:
        key = (*_scope_key(run.scope, run.data_source_id), run.run_id)
        with self._lock:
            if key in self._runs:
                raise ValueError(f"sync run already exists: {run.run_id}")
            self._runs[key] = run.model_copy(deep=True)

    def complete_run(self, run: SyncRun) -> None:
        self._replace_run(run)

    def fail_run(self, run: SyncRun) -> None:
        self._replace_run(run)

    def _replace_run(self, run: SyncRun) -> None:
        key = (*_scope_key(run.scope, run.data_source_id), run.run_id)
        with self._lock:
            if key not in self._runs:
                raise KeyError(f"sync run does not exist: {run.run_id}")
            self._runs[key] = run.model_copy(deep=True)

    def get_run(
        self,
        scope: ConnectorScope,
        data_source_id: str,
        run_id: str,
    ) -> SyncRun | None:
        with self._lock:
            run = self._runs.get((*_scope_key(scope, data_source_id), run_id))
            return run.model_copy(deep=True) if run is not None else None

    def commit_page(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        expected_cursor: SyncCursor,
        next_cursor: SyncCursor,
        bindings: Iterable[SourceItemBinding],
        events: Iterable[SyncEvent],
    ) -> None:
        prefix = _scope_key(scope, data_source_id)
        binding_tuple = tuple(bindings)
        event_tuple = tuple(events)
        with self._lock:
            current = self._cursors.get(
                prefix,
                SyncCursor(scope=scope, data_source_id=data_source_id, value="0"),
            )
            if current != expected_cursor:
                raise RuntimeError("sync cursor compare-and-swap failed")
            if next_cursor.scope != scope or next_cursor.data_source_id != data_source_id:
                raise ConnectorScopeError("next cursor is outside commit scope")

            bindings_copy = dict(self._bindings)
            events_copy = dict(self._events)
            for binding in binding_tuple:
                if binding.scope != scope or binding.data_source_id != data_source_id:
                    raise ConnectorScopeError("binding is outside commit scope")
                bindings_copy[(*prefix, binding.source_item_id)] = binding.model_copy(deep=True)
            for event in event_tuple:
                if event.scope != scope or event.data_source_id != data_source_id:
                    raise ConnectorScopeError("event is outside commit scope")
                key = (*prefix, event.event_id)
                existing = events_copy.get(key)
                if existing is not None and existing.event_fingerprint != event.event_fingerprint:
                    raise EventConflictError(
                        f"event {event.event_id!r} conflicts with committed payload"
                    )
                if existing is None:
                    events_copy[key] = event.model_copy(deep=True)

            # Atomic publication of the copied state models one control-database transaction.
            self._bindings = bindings_copy
            self._events = events_copy
            self._cursors[prefix] = next_cursor.model_copy(deep=True)

    def replace_binding(self, binding: SourceItemBinding) -> None:
        key = (
            binding.scope.tenant_id,
            binding.scope.kb_id,
            binding.data_source_id,
            binding.source_item_id,
        )
        with self._lock:
            existing = self._bindings.get(key)
            if existing is None:
                raise KeyError("binding does not exist in requested scope")
            if existing.document_id != binding.document_id:
                raise ConnectorScopeError("binding document identity cannot be replaced")
            self._bindings[key] = binding.model_copy(deep=True)


class InMemoryDocumentVersionBoundary:
    """模拟正式版本+任务路径，所有副作用都以 idempotency key 去重。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._version_receipts: dict[str, DocumentVersionReceipt] = {}
        self._version_fingerprints: dict[str, str] = {}
        self._metadata_fingerprints: dict[str, str] = {}
        self._delete_receipts: dict[str, DeletePropagationReceipt] = {}
        self.contents: dict[str, bytes] = {}
        self.create_calls = 0
        self.metadata_calls = 0
        self.delete_calls = 0

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
    ) -> DocumentVersionReceipt:
        fingerprint = canonical_sha256(
            {
                "scope": scope.model_dump(mode="json"),
                "data_source_id": data_source_id,
                "item": item.model_dump(mode="json"),
                "content_sha256": content_sha256,
                "existing_document_id": existing_document_id,
            }
        )
        del sync_run_id
        with self._lock:
            prior = self._version_receipts.get(idempotency_key)
            if prior is not None:
                if self._version_fingerprints[idempotency_key] != fingerprint:
                    raise EventConflictError(
                        "idempotency key was reused with different version payload"
                    )
                return prior.model_copy(deep=True)
            digest = canonical_sha256(
                {
                    "scope": scope.model_dump(mode="json"),
                    "source": data_source_id,
                    "item": item.source_item_id,
                }
            )
            document_id = existing_document_id or f"doc_{digest[:24]}"
            version_digest = canonical_sha256(
                {"document_id": document_id, "idempotency_key": idempotency_key}
            )
            receipt = DocumentVersionReceipt(
                document_id=document_id,
                document_version_id=f"dv_{version_digest[:24]}",
                ingestion_job_id=f"job_{version_digest[24:48]}",
                created=True,
            )
            self._version_receipts[idempotency_key] = receipt
            self._version_fingerprints[idempotency_key] = fingerprint
            self.contents[receipt.document_version_id] = bytes(fetched.content)
            self.create_calls += 1
            return receipt.model_copy(deep=True)

    def update_metadata(
        self,
        *,
        scope: ConnectorScope,
        data_source_id: str,
        document_id: str,
        item: SourceItem,
        idempotency_key: str,
        sync_run_id: str,
    ) -> None:
        fingerprint = canonical_sha256(
            {
                "scope": scope.model_dump(mode="json"),
                "source": data_source_id,
                "document": document_id,
                "item": item.model_dump(mode="json"),
            }
        )
        del sync_run_id
        with self._lock:
            prior = self._metadata_fingerprints.get(idempotency_key)
            if prior is not None and prior != fingerprint:
                raise EventConflictError(
                    "idempotency key was reused with different metadata payload"
                )
            if prior is None:
                self._metadata_fingerprints[idempotency_key] = fingerprint
                self.metadata_calls += 1

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
    ) -> DeletePropagationReceipt:
        del scope, data_source_id, purge_after, sync_run_id
        with self._lock:
            prior = self._delete_receipts.get(idempotency_key)
            if prior is not None:
                return prior.model_copy(deep=True)
            state = {
                DeletePropagationPolicy.DISABLE: SourceBindingStatus.DISABLED,
                DeletePropagationPolicy.SOFT_DELETE: SourceBindingStatus.SOFT_DELETED,
                DeletePropagationPolicy.PURGE: SourceBindingStatus.PURGED,
            }[policy]
            receipt = DeletePropagationReceipt(
                document_id=document_id,
                state=state,
                changed=True,
            )
            self._delete_receipts[idempotency_key] = receipt
            self.delete_calls += 1
            return receipt.model_copy(deep=True)


__all__ = [
    "InMemoryConnectorSyncRepository",
    "InMemoryDocumentVersionBoundary",
]
