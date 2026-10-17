"""TODO-09 production query orchestration tests."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.application.retrieval_service import (
    _deadline_from_state,
    profile_from_state,
    rerank_candidates,
)
from app.domain.retrieval import RetrievalEngineError, default_retrieval_profile
from app.security.principal import Principal, reset_current_principal, set_current_principal


class _FakeReranker:
    def compute_score(self, pairs, normalize=True):
        assert normalize is True
        return [0.2 if "普通" in pair[1] else 0.9 for pair in pairs]


class _MemoryFailingReranker:
    def compute_score(self, pairs, **kwargs):
        del pairs, kwargs
        raise MemoryError("simulated reranker OOM")


def _principal() -> Principal:
    return Principal(
        user_id="user-retrieval",
        tenant_id="tenant-retrieval",
        roles=("KBAdmin",),
        scopes=(),
        auth_type="test",
        credential_id="cred-retrieval",
    )


def _state() -> dict:
    profile = default_retrieval_profile("tenant-retrieval").model_copy(
        update={"candidate_k": 2, "top_k": 1}
    )
    config = profile.model_dump(
        mode="json",
        exclude={"profile_id", "version", "tenant_id", "status"},
    )
    return {
        "kb_id": "kb-retrieval",
        "original_query": "HAK180 安全事项",
        "rewritten_query": "HAK180 有哪些安全事项？",
        "retrieval_profile": config,
        "retrieval_profile_id": "rpf-retrieval",
        "retrieval_profile_version": 3,
        "retrieval_profile_hash": "a" * 64,
        "retrieval_diagnostics": {"rrf": {"candidate_count": 2}},
    }


def test_profile_identity_is_taken_from_pinned_session_not_mutable_config():
    token = set_current_principal(_principal())
    try:
        profile = profile_from_state(_state())
    finally:
        reset_current_principal(token)
    assert profile.profile_id == "rpf-retrieval"
    assert profile.version == 3
    assert profile.tenant_id == "tenant-retrieval"
    assert profile.candidate_k == 2
    assert profile.top_k == 1


def test_query_run_deadline_is_restored_as_utc_for_retrieval_adapter() -> None:
    restored = _deadline_from_state({"deadline_at": "2026-08-07T18:30:00+08:00"})

    assert restored == datetime(2026, 8, 7, 10, 30, tzinfo=UTC)


def test_retrieval_rejects_naive_query_run_deadline() -> None:
    with pytest.raises(RetrievalEngineError) as caught:
        _deadline_from_state({"deadline_at": "2026-08-07T18:30:00"})

    assert caught.value.code == "RETRIEVAL_INVALID_DEADLINE"
    assert caught.value.retryable is False


def test_hyde_cancellation_after_llm_prevents_late_embedding_and_milvus(
    monkeypatch,
) -> None:
    from app.rag.query import hyde_search_service

    cancelled = [False]

    def completed_after_route_timeout(*_args, **_kwargs):
        cancelled[0] = True
        return "迟到的假设答案"

    monkeypatch.setattr(
        hyde_search_service,
        "generate_hyde_answer",
        completed_after_route_timeout,
    )
    monkeypatch.setattr(
        hyde_search_service,
        "retrieve_candidates",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled HyDE must not continue to embedding/Milvus"
        ),
    )

    with pytest.raises(RetrievalEngineError) as caught:
        hyde_search_service.search_chunks_with_hyde(
            query_text="如何启动？",
            state={},
            cancellation=lambda: cancelled[0],
        )

    assert caught.value.code == "RETRIEVAL_CANCELLED"


def test_hyde_forwards_one_route_deadline_to_llm_and_retrieval(monkeypatch) -> None:
    from app.rag.query import hyde_search_service

    llm_timeouts: list[float | None] = []
    retrieval_deadlines: list[datetime | None] = []

    def generate(*_args, timeout_seconds=None, **_kwargs):
        llm_timeouts.append(timeout_seconds)
        return "假设答案"

    def retrieve(*_args, deadline_at=None, **_kwargs):
        retrieval_deadlines.append(deadline_at)
        return [], {"status": "empty"}

    monkeypatch.setattr(hyde_search_service, "generate_hyde_answer", generate)
    monkeypatch.setattr(hyde_search_service, "retrieve_candidates", retrieve)
    started = datetime.now(UTC)

    result = hyde_search_service.search_embedding_hyde(
        {"original_query": "如何启动？", "query_plan": {}, "doc_ids": []},
        cancellation=lambda: False,
        timeout_seconds=0.5,
    )

    assert result == []
    assert llm_timeouts and 0 < float(llm_timeouts[0] or 0) <= 0.5
    assert len(retrieval_deadlines) == 1
    assert retrieval_deadlines[0] is not None
    assert started < retrieval_deadlines[0] <= started + timedelta(seconds=1)


def test_reranker_port_applies_profile_top_k_and_keeps_diagnostics(monkeypatch):
    from app.application import retrieval_service

    monkeypatch.setattr(
        retrieval_service.llm_provider,
        "reranker_model",
        lambda: _FakeReranker(),
    )
    candidates = [
        {
            "chunk_id": "chunk-ordinary",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "content": "普通说明",
            "retrieval_observations": [
                {"route": "base.hybrid", "raw_score": 0.8, "rank": 1}
            ],
        },
        {
            "chunk_id": "chunk-safety",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "content": "断电并等待设备完全停止后再维护。",
            "retrieval_observations": [
                {"route": "hyde.hybrid", "raw_score": 0.7, "rank": 2}
            ],
        },
    ]
    state = _state()
    token = set_current_principal(_principal())
    try:
        result = rerank_candidates(state, candidates)
    finally:
        reset_current_principal(token)
    assert [item["chunk_id"] for item in result] == ["chunk-safety"]
    assert result[0]["rerank_score"] == 0.9
    assert state["retrieval_diagnostics"]["reranker"]["input_count"] == 2
    assert state["retrieval_diagnostics"]["reranker"]["output_count"] == 1


def test_reranker_memory_fallback_preserves_fusion_order_and_source_metadata(monkeypatch):
    from app.application import retrieval_service

    monkeypatch.setattr(
        retrieval_service.llm_provider,
        "reranker_model",
        lambda: _MemoryFailingReranker(),
    )
    candidates = [
        {
            "chunk_id": "chunk-first",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "content": "融合顺序第一条",
            "score": 0.88,
            "sentinel_metadata": {"must": "survive"},
        },
        {
            "chunk_id": "chunk-second",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-2",
            "document_version_id": "dver-2",
            "index_version_id": "idx-2",
            "content": "融合顺序第二条",
            "score": 0.77,
        },
    ]
    state = _state()
    token = set_current_principal(_principal())
    try:
        result = rerank_candidates(state, candidates)
    finally:
        reset_current_principal(token)

    assert result == [candidates[0]]
    assert "rerank_score" not in result[0]
    assert result[0]["sentinel_metadata"] == {"must": "survive"}
    diagnostics = state["retrieval_diagnostics"]["reranker"]
    assert diagnostics["status"] == "degraded"
    assert diagnostics["failure_code"] == "RERANKER_UNAVAILABLE"
    assert diagnostics["fallback"] == "preserve_order"
    assert diagnostics["batch_count"] == 1


def test_reranker_load_failure_preserves_order_diagnostics_and_uses_cooldown(monkeypatch):
    from app.application import retrieval_service

    calls = 0

    def fail_to_load():
        nonlocal calls
        calls += 1
        raise OSError(1455, "页面文件太小，无法完成操作")

    monkeypatch.setattr(retrieval_service, "_reranker_load_failure_at", None)
    monkeypatch.setattr(retrieval_service, "_reranker_load_failure_type", None)
    monkeypatch.setattr(retrieval_service, "_reranker_clock", lambda: 100.0)
    monkeypatch.setattr(retrieval_service.llm_provider, "reranker_model", fail_to_load)
    candidates = [
        {
            "chunk_id": "chunk-first",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "content": "融合顺序第一条",
            "score": 0.88,
            "retrieval_observations": [
                {"route": "base.hybrid", "raw_score": 0.88, "rank": 1}
            ],
            "sentinel_metadata": {"must": "survive"},
        },
        {
            "chunk_id": "chunk-second",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-2",
            "document_version_id": "dver-2",
            "index_version_id": "idx-2",
            "content": "融合顺序第二条",
            "score": 0.77,
        },
    ]

    token = set_current_principal(_principal())
    try:
        first_state = _state()
        first = rerank_candidates(first_state, candidates)
        second_state = _state()
        second = rerank_candidates(second_state, candidates)
    finally:
        reset_current_principal(token)

    assert calls == 1
    assert first == second == [candidates[0]]
    assert first[0]["sentinel_metadata"] == {"must": "survive"}
    assert "rerank_score" not in first[0]
    assert first_state["retrieval_diagnostics"]["rrf"] == {"candidate_count": 2}
    first_diagnostics = first_state["retrieval_diagnostics"]["reranker"]
    assert first_diagnostics["status"] == "degraded"
    assert first_diagnostics["phase"] == "load"
    assert first_diagnostics["error_type"] == "OSError"
    assert first_diagnostics["failure_cached"] is False
    assert first_diagnostics["batch_count"] == 0
    second_diagnostics = second_state["retrieval_diagnostics"]["reranker"]
    assert second_diagnostics["phase"] == "load"
    assert second_diagnostics["error_type"] == "OSError"
    assert second_diagnostics["failure_cached"] is True
    assert second_diagnostics["cooldown_remaining_ms"] == 60_000


def test_reranker_load_failure_error_mode_is_typed_and_cooldown_avoids_retry(monkeypatch):
    from app.application import retrieval_service

    calls = 0

    def fail_to_load():
        nonlocal calls
        calls += 1
        raise OSError(1455, "页面文件太小，无法完成操作")

    monkeypatch.setattr(retrieval_service, "_reranker_load_failure_at", None)
    monkeypatch.setattr(retrieval_service, "_reranker_load_failure_type", None)
    monkeypatch.setattr(retrieval_service, "_reranker_clock", lambda: 200.0)
    monkeypatch.setattr(retrieval_service.llm_provider, "reranker_model", fail_to_load)
    monkeypatch.setattr(
        retrieval_service,
        "reranker_config",
        SimpleNamespace(failure_mode="error"),
    )
    candidates = [
        {
            "chunk_id": "chunk-first",
            "tenant_id": "tenant-retrieval",
            "kb_id": "kb-retrieval",
            "document_id": "doc-1",
            "document_version_id": "dver-1",
            "index_version_id": "idx-1",
            "content": "融合顺序第一条",
            "score": 0.88,
        }
    ]

    token = set_current_principal(_principal())
    try:
        with pytest.raises(RetrievalEngineError) as first_caught:
            rerank_candidates(_state(), candidates)
        with pytest.raises(RetrievalEngineError) as second_caught:
            rerank_candidates(_state(), candidates)
    finally:
        reset_current_principal(token)

    assert calls == 1
    assert first_caught.value.code == "RERANKER_UNAVAILABLE"
    assert first_caught.value.retryable is True
    assert first_caught.value.route == "reranker"
    assert first_caught.value.details == {
        "phase": "load",
        "error_type": "OSError",
        "failure_cached": False,
    }
    assert second_caught.value.details["phase"] == "load"
    assert second_caught.value.details["error_type"] == "OSError"
    assert second_caught.value.details["failure_cached"] is True
    assert second_caught.value.details["cooldown_remaining_ms"] == 60_000
