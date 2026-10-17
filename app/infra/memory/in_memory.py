"""TODO-22 仓储和派生资源端口的线程安全内存实现。"""

from __future__ import annotations

from threading import RLock

from app.application.memory_service import MemoryStateConflictError
from app.domain.memory import (
    LongTermMemory,
    MemoryAuditEvent,
    MemoryNamespace,
    MemoryStatus,
)


def _namespace_key(namespace: MemoryNamespace) -> tuple[str, str, str]:
    return namespace.tenant_id, namespace.subject_user_id, namespace.kb_id


def _memory_key(
    namespace: MemoryNamespace,
    memory_id: str,
) -> tuple[str, str, str, str]:
    return (*_namespace_key(namespace), memory_id)


class InMemoryMemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._memories: dict[tuple[str, str, str, str], LongTermMemory] = {}
        self._candidate_keys: dict[tuple[str, str, str, str], str] = {}
        self._audits: dict[tuple[str, str, str, str], MemoryAuditEvent] = {}

    def create_proposed(
        self,
        memory: LongTermMemory,
        audit: MemoryAuditEvent,
    ) -> LongTermMemory:
        if memory.status != MemoryStatus.PROPOSED:
            raise MemoryStateConflictError("create_proposed requires proposed status")
        self._validate_audit(memory, audit)
        key = _memory_key(memory.namespace, memory.memory_id)
        candidate_key = (*_namespace_key(memory.namespace), memory.candidate_id)
        with self._lock:
            existing_id = self._candidate_keys.get(candidate_key)
            if existing_id is not None:
                existing = self._memories[_memory_key(memory.namespace, existing_id)]
                return existing.model_copy(deep=True)
            if key in self._memories:
                raise MemoryStateConflictError("memory id already exists")
            self._memories[key] = memory.model_copy(deep=True)
            self._candidate_keys[candidate_key] = memory.memory_id
            self._append_audit_locked(audit)
            return memory.model_copy(deep=True)

    def get(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> LongTermMemory | None:
        with self._lock:
            memory = self._memories.get(_memory_key(namespace, memory_id))
            return memory.model_copy(deep=True) if memory is not None else None

    def list(
        self,
        namespace: MemoryNamespace,
        *,
        statuses: tuple[MemoryStatus, ...] | None = None,
    ) -> tuple[LongTermMemory, ...]:
        prefix = _namespace_key(namespace)
        accepted = set(statuses) if statuses is not None else None
        with self._lock:
            values = [
                memory
                for key, memory in self._memories.items()
                if key[:3] == prefix and (accepted is None or memory.status in accepted)
            ]
            values.sort(key=lambda memory: (memory.created_at, memory.memory_id))
            return tuple(memory.model_copy(deep=True) for memory in values)

    def activate_with_supersede(
        self,
        *,
        proposed: LongTermMemory,
        active: LongTermMemory,
        audit: MemoryAuditEvent,
        supersede_audits: tuple[MemoryAuditEvent, ...],
    ) -> tuple[LongTermMemory, tuple[LongTermMemory, ...]]:
        if active.status != MemoryStatus.ACTIVE:
            raise MemoryStateConflictError("activation target must be active")
        if proposed.namespace != active.namespace or proposed.memory_id != active.memory_id:
            raise MemoryStateConflictError("activation cannot change memory identity/scope")
        self._validate_audit(active, audit)
        key = _memory_key(proposed.namespace, proposed.memory_id)
        with self._lock:
            current = self._memories.get(key)
            if current != proposed:
                raise MemoryStateConflictError("memory compare-and-swap failed")
            conflicts = [
                memory
                for other_key, memory in self._memories.items()
                if other_key[:3] == _namespace_key(active.namespace)
                and memory.memory_id != active.memory_id
                and memory.status == MemoryStatus.ACTIVE
                and memory.type == active.type
                and memory.conflict_key == active.conflict_key
            ]
            audit_by_memory = {item.memory_id: item for item in supersede_audits}
            if set(audit_by_memory) != {item.memory_id for item in conflicts}:
                raise MemoryStateConflictError(
                    "supersede audit set does not match active conflicts"
                )
            memories_copy = dict(self._memories)
            superseded: list[LongTermMemory] = []
            for conflict in conflicts:
                replacement = conflict.model_copy(
                    update={
                        "status": MemoryStatus.SUPERSEDED,
                        "valid_to": active.valid_from,
                        "updated_at": active.updated_at,
                    }
                )
                memories_copy[_memory_key(conflict.namespace, conflict.memory_id)] = replacement
                superseded.append(replacement)
            memories_copy[key] = active.model_copy(deep=True)

            # Validate audit uniqueness before publishing any copied state.
            audit_values = (audit, *supersede_audits)
            for item in audit_values:
                audit_key = (*_namespace_key(item.namespace), item.audit_id)
                if audit_key in self._audits:
                    raise MemoryStateConflictError("duplicate memory audit id")
            self._memories = memories_copy
            for item in audit_values:
                self._append_audit_locked(item)
            return (
                active.model_copy(deep=True),
                tuple(item.model_copy(deep=True) for item in superseded),
            )

    def replace(
        self,
        *,
        current: LongTermMemory,
        replacement: LongTermMemory,
        audit: MemoryAuditEvent,
    ) -> LongTermMemory:
        if current.namespace != replacement.namespace or current.memory_id != replacement.memory_id:
            raise MemoryStateConflictError("replace cannot change memory identity/scope")
        self._validate_audit(replacement, audit)
        key = _memory_key(current.namespace, current.memory_id)
        with self._lock:
            persisted = self._memories.get(key)
            if persisted != current:
                raise MemoryStateConflictError("memory compare-and-swap failed")
            audit_key = (*_namespace_key(audit.namespace), audit.audit_id)
            if audit_key in self._audits:
                raise MemoryStateConflictError("duplicate memory audit id")
            self._memories[key] = replacement.model_copy(deep=True)
            self._append_audit_locked(audit)
            return replacement.model_copy(deep=True)

    def list_audits(
        self,
        namespace: MemoryNamespace,
    ) -> tuple[MemoryAuditEvent, ...]:
        prefix = _namespace_key(namespace)
        with self._lock:
            audits = [audit for key, audit in self._audits.items() if key[:3] == prefix]
            audits.sort(key=lambda audit: (audit.occurred_at, audit.audit_id))
            return tuple(audit.model_copy(deep=True) for audit in audits)

    def append_audit(self, audit: MemoryAuditEvent) -> None:
        with self._lock:
            self._append_audit_locked(audit)

    def _append_audit_locked(self, audit: MemoryAuditEvent) -> None:
        key = (*_namespace_key(audit.namespace), audit.audit_id)
        if key in self._audits:
            raise MemoryStateConflictError("duplicate memory audit id")
        self._audits[key] = audit.model_copy(deep=True)

    @staticmethod
    def _validate_audit(
        memory: LongTermMemory,
        audit: MemoryAuditEvent,
    ) -> None:
        if audit.namespace != memory.namespace or audit.memory_id != memory.memory_id:
            raise MemoryStateConflictError("audit is outside memory scope")
        forbidden = {"content", "structured_value", "source_excerpt"}
        if forbidden.intersection(audit.metadata):
            raise MemoryStateConflictError("audit metadata contains memory content")


class InMemoryMemoryDerivativeStore:
    """模拟独立的向量、缓存和摘要资源，便于验证遗忘传播完整性。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._vectors: set[tuple[str, str, str, str]] = set()
        self._cache: set[tuple[str, str, str, str]] = set()
        self._summaries: set[tuple[str, str, str, str]] = set()

    def index(self, memory: LongTermMemory) -> None:
        if memory.status != MemoryStatus.ACTIVE:
            raise MemoryStateConflictError("only active memory can be indexed")
        key = _memory_key(memory.namespace, memory.memory_id)
        with self._lock:
            self._vectors.add(key)
            self._cache.add(key)
            self._summaries.add(key)

    def delete(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> tuple[str, ...]:
        key = _memory_key(namespace, memory_id)
        deleted: list[str] = []
        with self._lock:
            for name, target in (
                ("vector", self._vectors),
                ("cache", self._cache),
                ("summary", self._summaries),
            ):
                if key in target:
                    target.remove(key)
                    deleted.append(name)
        return tuple(deleted)

    def resources_for(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> tuple[str, ...]:
        key = _memory_key(namespace, memory_id)
        with self._lock:
            found = [
                name
                for name, target in (
                    ("vector", self._vectors),
                    ("cache", self._cache),
                    ("summary", self._summaries),
                )
                if key in target
            ]
        return tuple(found)


__all__ = ["InMemoryMemoryDerivativeStore", "InMemoryMemoryRepository"]
