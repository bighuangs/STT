from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.application.query_cache_service import (
    AsyncSingleflight,
    IdempotencyConflictError,
    IdempotencyRegistry,
    QueryFingerprint,
    QueryReplayManifest,
    ReplayService,
    ScopedTTLCache,
    canonical_hash,
)


def _fingerprint(**overrides) -> QueryFingerprint:
    values = {
        "tenant_id": "tenant-a",
        "principal_scope": ["query:execute", "kb:hak180"],
        "normalized_query": "在使用 HAK180 之前需要做什么？",
        "conversation_history": [],
        "authorized_scope": {"kb_ids": ["kb-hak180"], "index": "idx-v1"},
        "filters": {},
        "index_versions": ["idx-v1"],
        "query_profile_version": "profile-v1",
        "planner_version": "planner-v1",
        "embedding_model_version": "bge-m3-v1",
        "rerank_model_version": "rerank-v1",
        "generator_model_version": "qwen-v1",
        "prompt_version": "answer-v1",
        "policy_version": "policy-v1",
        "web_snapshot_policy": "disabled",
    }
    values.update(overrides)
    return QueryFingerprint.from_request(**values)


def test_idempotency_reuses_run_but_conflicting_payload_is_rejected() -> None:
    registry = IdempotencyRegistry()
    sequence = iter(("qrun-1", "qrun-2"))

    first = registry.resolve(
        tenant_id="tenant-a",
        idempotency_key="client-turn-1",
        payload={"query_hash": canonical_hash("HAK180")},
        create_run_id=lambda: next(sequence),
    )
    second = registry.resolve(
        tenant_id="tenant-a",
        idempotency_key="client-turn-1",
        payload={"query_hash": canonical_hash("HAK180")},
        create_run_id=lambda: next(sequence),
    )

    assert first == ("qrun-1", True)
    assert second == ("qrun-1", False)
    with pytest.raises(IdempotencyConflictError):
        registry.resolve(
            tenant_id="tenant-a",
            idempotency_key="client-turn-1",
            payload={"query_hash": canonical_hash("另一问题")},
            create_run_id=lambda: "qrun-3",
        )


def test_singleflight_100_waiters_execute_exactly_one_producer() -> None:
    async def scenario():
        flight: AsyncSingleflight[int] = AsyncSingleflight()
        calls = 0

        async def producer() -> int:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return 42

        values = await asyncio.gather(
            *(flight.run("same-fingerprint", producer) for _ in range(100))
        )
        return calls, values

    calls, values = asyncio.run(scenario())
    assert calls == 1
    assert {item[0] for item in values} == {42}
    assert sum(1 for _, is_producer in values if is_producer) == 1


def test_cache_requires_authorization_and_all_versions_participate_in_key() -> None:
    cache: ScopedTTLCache[dict] = ScopedTTLCache()
    original = _fingerprint()
    assert cache.put(
        original,
        {"answer": "先阅读产品安全手册"},
        ttl=timedelta(minutes=5),
        classification="internal",
    )

    assert cache.get(original, authorize=lambda: False) == (
        None,
        "bypass_unauthorized",
    )
    assert cache.get(original, authorize=lambda: True)[1] == "hit"
    for changed in (
        {"tenant_id": "tenant-b"},
        {"principal_scope": ["query:execute"]},
        {"index_versions": ["idx-v2"]},
        {"query_profile_version": "profile-v2"},
        {"generator_model_version": "qwen-v2"},
        {"prompt_version": "answer-v2"},
        {"policy_version": "policy-v2"},
    ):
        assert cache.get(_fingerprint(**changed), authorize=lambda: True)[1] == "miss"

    assert cache.put(
        _fingerprint(normalized_query="restricted"),
        {"answer": "secret"},
        ttl=timedelta(minutes=5),
        classification="restricted",
    ) is False


def test_replay_is_explicitly_side_effect_free() -> None:
    fingerprint = _fingerprint()
    manifest = QueryReplayManifest(
        query_run_id="qrun-hak180",
        fingerprint=fingerprint,
        request_hash=canonical_hash({"query": fingerprint.normalized_query_hash}),
        query_plan={"plan_id": "qpl-1"},
        route_result_refs=("obj://route/1",),
        context_pack_ref="obj://context/1",
        answer_envelope={"schema_version": "answer.v1", "text": "旧答案"},
        event_stream_hash=canonical_hash([{"seq": 1, "type": "final"}]),
    )

    result = ReplayService().replay(
        manifest,
        mode="fixed_context",
        generator=lambda _: {"schema_version": "answer.v1", "text": "重放答案"},
    )

    assert result.replay_of == "qrun-hak180"
    assert result.committed_to_conversation is False
    assert result.external_side_effects is False

