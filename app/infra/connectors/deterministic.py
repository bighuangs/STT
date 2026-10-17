"""分页、游标、重放和失败均可确定性控制的离线连接器。"""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from app.application.connector_sync_service import ConnectorRateLimitedError
from app.domain.connectors import (
    ChangePage,
    ConnectorCapabilities,
    ConnectorValidation,
    DataSource,
    FetchedSourceObject,
    SourceChange,
    SourceItem,
    SourceSnapshot,
    SyncCursor,
    utc_now,
)


class DeterministicFakeConnector:
    """可用于合同测试和本地演练，不进行任何网络访问。"""

    def __init__(
        self,
        changes: Iterable[SourceChange] = (),
        *,
        objects: dict[tuple[str, str], bytes] | None = None,
        snapshot_items: Iterable[SourceItem] | None = None,
        fail_fetch_once_for: Iterable[str] = (),
        fail_ack_once: bool = False,
        retry_after_once: float | None = None,
    ) -> None:
        self._lock = RLock()
        self._changes = tuple(
            sorted(changes, key=lambda change: (change.sequence, change.event_id))
        )
        self._objects = dict(objects or {})
        self._snapshot_items = tuple(snapshot_items) if snapshot_items is not None else None
        self._fail_fetch_once_for = set(fail_fetch_once_for)
        self._fail_ack_once = fail_ack_once
        self._retry_after_once = retry_after_once
        self.fetch_calls: list[tuple[str, str]] = []
        self.acked_cursors: list[str] = []
        self.list_calls: list[tuple[str, int]] = []

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            incremental_cursor=True,
            delete_events=True,
            folders=True,
            acl=False,
            webhook=False,
            external_network=False,
            max_object_bytes=1024 * 1024 * 1024,
            supported_mime_types=(
                "application/json",
                "application/pdf",
                "text/markdown",
                "text/plain",
            ),
        )

    def validate_config(self, data_source: DataSource) -> ConnectorValidation:
        errors = (
            ("fake connector rejects force_invalid=true",)
            if data_source.config.get("force_invalid") is True
            else ()
        )
        return ConnectorValidation(
            valid=not errors,
            capabilities=self.capabilities,
            checks=("offline", "readable", "no-inline-secret"),
            errors=errors,
        )

    def list_changes(
        self,
        data_source: DataSource,
        cursor: SyncCursor,
        limit: int,
    ) -> ChangePage:
        del data_source
        with self._lock:
            if self._retry_after_once is not None:
                retry_after = self._retry_after_once
                self._retry_after_once = None
                raise ConnectorRateLimitedError(retry_after)
            try:
                offset = int(cursor.value or "0")
            except ValueError as exc:
                raise ValueError("fake connector cursor must be an integer offset") from exc
            if offset < 0 or offset > len(self._changes):
                raise ValueError("fake connector cursor is outside the change log")
            bounded_limit = max(1, limit)
            end = min(offset + bounded_limit, len(self._changes))
            self.list_calls.append((cursor.value, bounded_limit))
            return ChangePage(
                changes=self._changes[offset:end],
                next_cursor=str(end),
                has_more=end < len(self._changes),
            )

    def fetch(
        self,
        data_source: DataSource,
        item: SourceItem,
    ) -> FetchedSourceObject:
        del data_source
        with self._lock:
            key = (item.source_item_id, item.remote_version)
            self.fetch_calls.append(key)
            if item.source_item_id in self._fail_fetch_once_for:
                self._fail_fetch_once_for.remove(item.source_item_id)
                raise OSError(f"injected fetch failure for {item.source_item_id}")
            try:
                content = self._objects[key]
            except KeyError as exc:
                raise FileNotFoundError(f"no fake content registered for {key}") from exc
            return FetchedSourceObject(
                source_item_id=item.source_item_id,
                remote_version=item.remote_version,
                content=content,
                mime_type=item.mime_type,
                logical_name=item.logical_name,
                metadata=item.metadata,
            )

    def ack(self, data_source: DataSource, cursor: SyncCursor) -> None:
        del data_source
        with self._lock:
            if self._fail_ack_once:
                self._fail_ack_once = False
                raise OSError("injected ack failure")
            self.acked_cursors.append(cursor.value)

    def snapshot(self, data_source: DataSource) -> SourceSnapshot:
        del data_source
        with self._lock:
            items = self._snapshot_items
            if items is None:
                latest: dict[str, SourceItem] = {}
                for change in self._changes:
                    if change.item is not None:
                        latest[change.item.source_item_id] = change.item
                    elif change.tombstone is not None:
                        latest.pop(change.tombstone.source_item_id, None)
                items = tuple(latest.values())
            return SourceSnapshot(
                items=tuple(sorted(items, key=lambda item: item.source_item_id)),
                cursor=str(len(self._changes)),
                complete=True,
                observed_at=utc_now(),
            )

    def set_snapshot(self, items: Iterable[SourceItem]) -> None:
        with self._lock:
            self._snapshot_items = tuple(items)
