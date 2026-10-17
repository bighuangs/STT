"""TODO-32 查询幂等、Singleflight、分层缓存与重放。

任何原始问题、历史正文和权限列表只用于本地计算 SHA-256，不会进入 cache key 或
日志。调用方必须先完成授权再查缓存；版本或 scope 任一变化都会自然形成新键。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from threading import RLock
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


T = TypeVar("T")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"sha256:{sha256(raw.encode('utf-8')).hexdigest()}"


class QueryFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "query-fingerprint/1"
    tenant_hash: str
    principal_scope_hash: str
    normalized_query_hash: str
    conversation_history_hash: str
    authorized_scope_hash: str
    filters_hash: str
    index_versions_hash: str
    query_profile_version: str
    planner_version: str
    embedding_model_version: str
    rerank_model_version: str
    generator_model_version: str
    prompt_version: str
    policy_version: str
    web_snapshot_policy: str
    locale: str = "zh-CN"

    @field_validator(
        "tenant_hash",
        "principal_scope_hash",
        "normalized_query_hash",
        "conversation_history_hash",
        "authorized_scope_hash",
        "filters_hash",
        "index_versions_hash",
    )
    @classmethod
    def require_hash(cls, value: str) -> str:
        if len(value) != 71 or not value.startswith("sha256:"):
            raise ValueError("fingerprint sensitive fields must be SHA-256 digests")
        return value

    @property
    def key(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    @classmethod
    def from_request(
        cls,
        *,
        tenant_id: str,
        principal_scope: Any,
        normalized_query: str,
        conversation_history: Any,
        authorized_scope: Any,
        filters: Any,
        index_versions: Any,
        query_profile_version: str,
        planner_version: str,
        embedding_model_version: str,
        rerank_model_version: str,
        generator_model_version: str,
        prompt_version: str,
        policy_version: str,
        web_snapshot_policy: str,
        locale: str = "zh-CN",
    ) -> "QueryFingerprint":
        return cls(
            tenant_hash=canonical_hash(tenant_id),
            principal_scope_hash=canonical_hash(principal_scope),
            normalized_query_hash=canonical_hash(" ".join(normalized_query.split())),
            conversation_history_hash=canonical_hash(conversation_history),
            authorized_scope_hash=canonical_hash(authorized_scope),
            filters_hash=canonical_hash(filters),
            index_versions_hash=canonical_hash(index_versions),
            query_profile_version=query_profile_version,
            planner_version=planner_version,
            embedding_model_version=embedding_model_version,
            rerank_model_version=rerank_model_version,
            generator_model_version=generator_model_version,
            prompt_version=prompt_version,
            policy_version=policy_version,
            web_snapshot_policy=web_snapshot_policy,
            locale=locale,
        )


class IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class IdempotencyRecord:
    tenant_hash: str
    key_hash: str
    payload_hash: str
    query_run_id: str
    expires_at: datetime


class IdempotencyRegistry:
    """进程内参考实现；SQL repository 复用相同语义。"""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], IdempotencyRecord] = {}
        self._lock = RLock()

    def resolve(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        payload: Any,
        create_run_id: Callable[[], str],
        ttl: timedelta = timedelta(hours=24),
    ) -> tuple[str, bool]:
        now = datetime.now(UTC)
        tenant_hash = canonical_hash(tenant_id)
        key_hash = canonical_hash(idempotency_key)
        payload_hash = canonical_hash(payload)
        lookup = (tenant_hash, key_hash)
        with self._lock:
            existing = self._records.get(lookup)
            if existing is not None and existing.expires_at > now:
                if existing.payload_hash != payload_hash:
                    raise IdempotencyConflictError(
                        "同一 Idempotency-Key 对应了不同请求载荷"
                    )
                return existing.query_run_id, False
            run_id = create_run_id()
            self._records[lookup] = IdempotencyRecord(
                tenant_hash=tenant_hash,
                key_hash=key_hash,
                payload_hash=payload_hash,
                query_run_id=run_id,
                expires_at=now + ttl,
            )
            return run_id, True


class AsyncSingleflight(Generic[T]):
    """同一 fingerprint 只有一个 producer；waiter 取消不会取消 producer。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._flights: dict[str, asyncio.Task[T]] = {}

    async def run(self, key: str, producer: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        async with self._lock:
            task = self._flights.get(key)
            is_producer = task is None
            if task is None:
                task = asyncio.create_task(producer(), name=f"singleflight:{key[-12:]}")
                self._flights[key] = task
                task.add_done_callback(
                    lambda completed, lookup=key: asyncio.create_task(
                        self._remove(lookup, completed)
                    )
                )
        return await asyncio.shield(task), is_producer

    async def _remove(self, key: str, task: asyncio.Task[T]) -> None:
        async with self._lock:
            if self._flights.get(key) is task:
                self._flights.pop(key, None)


@dataclass(frozen=True)
class CacheEntry(Generic[T]):
    value: T
    tenant_hash: str
    authorized_scope_hash: str
    created_at: datetime
    expires_at: datetime
    classification: str


class ScopedTTLCache(Generic[T]):
    def __init__(self, *, max_entries: int = 1_000) -> None:
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry[T]] = {}
        self._lock = RLock()

    def put(
        self,
        fingerprint: QueryFingerprint,
        value: T,
        *,
        ttl: timedelta,
        classification: str,
    ) -> bool:
        if classification in {"confidential", "restricted"}:
            return False
        if ttl.total_seconds() <= 0:
            raise ValueError("cache ttl must be positive")
        now = datetime.now(UTC)
        entry = CacheEntry(
            value=value,
            tenant_hash=fingerprint.tenant_hash,
            authorized_scope_hash=fingerprint.authorized_scope_hash,
            created_at=now,
            expires_at=now + ttl,
            classification=classification,
        )
        with self._lock:
            if len(self._entries) >= self.max_entries and fingerprint.key not in self._entries:
                oldest = min(self._entries, key=lambda item: self._entries[item].created_at)
                self._entries.pop(oldest, None)
            self._entries[fingerprint.key] = entry
        return True

    def get(
        self,
        fingerprint: QueryFingerprint,
        *,
        authorize: Callable[[], bool],
    ) -> tuple[T | None, str]:
        # 授权必须先于 key lookup，避免通过命中时间探测无权资源。
        if not authorize():
            return None, "bypass_unauthorized"
        with self._lock:
            entry = self._entries.get(fingerprint.key)
            if entry is None:
                return None, "miss"
            if entry.expires_at <= datetime.now(UTC):
                self._entries.pop(fingerprint.key, None)
                return None, "stale"
            if (
                entry.tenant_hash != fingerprint.tenant_hash
                or entry.authorized_scope_hash != fingerprint.authorized_scope_hash
            ):
                return None, "bypass_scope_mismatch"
            return entry.value, "hit"

    def invalidate(self, predicate: Callable[[CacheEntry[T]], bool]) -> int:
        with self._lock:
            keys = [key for key, value in self._entries.items() if predicate(value)]
            for key in keys:
                self._entries.pop(key, None)
        return len(keys)


class QueryReplayManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "query-replay/1"
    query_run_id: str = Field(min_length=1, max_length=64)
    fingerprint: QueryFingerprint
    request_hash: str
    history_snapshot_ref: str | None = None
    query_plan: dict[str, Any]
    route_result_refs: tuple[str, ...] = ()
    context_pack_ref: str | None = None
    answer_envelope: dict[str, Any]
    event_stream_hash: str
    external_snapshot_refs: tuple[str, ...] = ()
    random_seed: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("request_hash", "event_stream_hash")
    @classmethod
    def require_manifest_hash(cls, value: str) -> str:
        if len(value) != 71 or not value.startswith("sha256:"):
            raise ValueError("replay manifest hashes must be SHA-256")
        return value


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_of: str
    mode: str = Field(pattern=r"^(fixed_context|paired_pipeline)$")
    answer_envelope: dict[str, Any]
    changed_versions: dict[str, tuple[str, str]] = Field(default_factory=dict)
    committed_to_conversation: bool = False
    external_side_effects: bool = False


class ReplayService:
    def replay(
        self,
        manifest: QueryReplayManifest,
        *,
        mode: str,
        generator: Callable[[QueryReplayManifest], Mapping[str, Any]],
        changed_versions: Mapping[str, tuple[str, str]] | None = None,
    ) -> ReplayResult:
        if mode not in {"fixed_context", "paired_pipeline"}:
            raise ValueError("unsupported replay mode")
        answer = dict(generator(manifest))
        return ReplayResult(
            replay_of=manifest.query_run_id,
            mode=mode,
            answer_envelope=answer,
            changed_versions=dict(changed_versions or {}),
            committed_to_conversation=False,
            external_side_effects=False,
        )


__all__ = [
    "AsyncSingleflight",
    "IdempotencyConflictError",
    "IdempotencyRegistry",
    "QueryFingerprint",
    "QueryReplayManifest",
    "ReplayResult",
    "ReplayService",
    "ScopedTTLCache",
    "canonical_hash",
]
