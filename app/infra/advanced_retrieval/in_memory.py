"""TODO-16 生命周期端口的确定性纯内存实现。"""

from __future__ import annotations

from threading import RLock

from app.domain.advanced_retrieval import AdvancedIndexSnapshot


class InMemoryAdvancedIndexRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[str, AdvancedIndexSnapshot] = {}

    def save_snapshot(self, snapshot: AdvancedIndexSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.index.advanced_index_id] = snapshot

    def get_snapshot(self, advanced_index_id: str) -> AdvancedIndexSnapshot | None:
        with self._lock:
            return self._snapshots.get(advanced_index_id)

    def list_snapshots(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[AdvancedIndexSnapshot, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        snapshot
                        for snapshot in self._snapshots.values()
                        if snapshot.index.tenant_id == tenant_id and snapshot.index.kb_id == kb_id
                    ),
                    key=lambda item: (
                        item.index.index_type,
                        item.index.version,
                        item.index.advanced_index_id,
                    ),
                )
            )


__all__ = ["InMemoryAdvancedIndexRepository"]
