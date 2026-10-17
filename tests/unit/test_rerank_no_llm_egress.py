from __future__ import annotations

from copy import deepcopy

import pytest

from app.domain.retrieval import (
    RerankerConfig,
    RerankRequest,
    RetrievalEngineError,
    RetrievalHit,
    RouteScore,
)
from app.infra.retrieval.reranker_adapter import BGERerankerAdapter
from app.rag.query import rerank_service
from app.shared.config.settings_config import AppSettings


class CharacterTokenizer:
    def __init__(self) -> None:
        self.encoded_character_counts: list[int] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded_character_counts.append(len(text))
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(item) for item in token_ids)


class RecordingReranker:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.received_pairs: list[list[str]] = []

    def compute_score(self, pairs: list[list[str]], *, normalize: bool) -> list[float]:
        assert normalize is True
        self.received_pairs = pairs
        return [0.75 for _ in pairs]


def test_long_evidence_is_truncated_locally_without_chat_model_egress(monkeypatch):
    canary = "CANARY-INDIRECT-PROMPT-MUST-NOT-LEAVE"
    original_text = "安全操作说明。" * 100 + canary
    candidates = [
        {
            "chunk_id": "chk_safe_1",
            "text": original_text,
            "content_hash": "sha256:" + "a" * 64,
        }
    ]
    original = deepcopy(candidates)
    reranker = RecordingReranker()

    def forbidden_chat(*args, **kwargs):
        raise AssertionError(f"rerank must not call chat model: args={args}, kwargs={kwargs}")

    monkeypatch.setattr(rerank_service.llm_provider, "chat", forbidden_chat)
    monkeypatch.setattr(rerank_service.llm_provider, "reranker_model", lambda: reranker)

    result = rerank_service.score_and_sort_chunks(
        {"original_query": "在使用 HAK180 之前需要做什么？"},
        candidates,
    )

    assert result[0]["rerank_score"] == 0.75
    assert candidates[0]["text"] == original[0]["text"]
    assert candidates[0]["content_hash"] == original[0]["content_hash"]
    assert len(reranker.received_pairs) == 1
    question, evidence = reranker.received_pairs[0]
    assert canary not in evidence
    assert (
        len(reranker.tokenizer.encode(question, add_special_tokens=False))
        + len(reranker.tokenizer.encode(evidence, add_special_tokens=False))
        + rerank_service.RERANK_SPECIAL_TOKENS
        <= rerank_service.RERANK_MAX_INPUT_TOKENS
    )


def test_extreme_question_and_evidence_are_both_bounded_without_mutation():
    reranker = RecordingReranker()
    question = "问" * 2_000
    evidence = "证据" * 2_000
    candidates = [{"text": evidence, "content_hash": "sha256:" + "b" * 64}]
    original = deepcopy(candidates)

    pairs = rerank_service.build_question_pairs(question, candidates, reranker)

    bounded_question, bounded_evidence = pairs[0]
    assert bounded_question
    assert bounded_evidence
    assert candidates == original
    assert (
        len(reranker.tokenizer.encode(bounded_question, add_special_tokens=False))
        + len(reranker.tokenizer.encode(bounded_evidence, add_special_tokens=False))
        + rerank_service.RERANK_SPECIAL_TOKENS
        <= rerank_service.RERANK_MAX_INPUT_TOKENS
    )


def test_production_reranker_adapter_uses_local_tokenizer_hard_budget():
    canary = "CANARY-TAIL-MUST-BE-TRUNCATED"
    original_content = "维护前先断电。" * 100 + canary
    model = RecordingReranker()
    adapter = BGERerankerAdapter(model=model)
    hit = RetrievalHit(
        chunk_id="chunk-1",
        tenant_id="tenant-default",
        kb_id="kb-hak180",
        document_id="doc-hak180",
        document_version_id="dver-hak180",
        index_version_id="idx-hak180",
        content=original_content,
        content_hash="sha256:" + "c" * 64,
        scores=(RouteScore(route="hybrid", raw_score=0.8, rank=1),),
    )

    result = adapter.rerank(
        RerankRequest(
            query="在使用 HAK180 之前需要做什么？",
            candidates=(hit,),
            config=RerankerConfig(max_input_tokens=64),
            trace_id="trace-rerank-no-egress",
        )
    )

    question, evidence = model.received_pairs[0]
    assert canary not in evidence
    assert (
        len(model.tokenizer.encode(question, add_special_tokens=False))
        + len(model.tokenizer.encode(evidence, add_special_tokens=False))
        + rerank_service.RERANK_SPECIAL_TOKENS
        <= 64
    )
    assert result.hits[0].content == original_content
    assert result.hits[0].content_hash == hit.content_hash


class BoundedBatchReranker:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.calls: list[dict] = []

    def compute_score(
        self,
        pairs: list[list[str]],
        *,
        normalize: bool,
        batch_size: int,
        query_max_length: int,
        max_length: int,
    ) -> list[float]:
        self.calls.append(
            {
                "pairs": deepcopy(pairs),
                "normalize": normalize,
                "batch_size": batch_size,
                "query_max_length": query_max_length,
                "max_length": max_length,
            }
        )
        return [float(pair[1].split(maxsplit=1)[0].removeprefix("doc-")) for pair in pairs]


def _rerank_hit(index: int, *, content: str | None = None) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{index:02d}",
        tenant_id="tenant-default",
        kb_id="kb-hak180",
        document_id=f"doc-{index:02d}",
        document_version_id="dver-hak180",
        index_version_id="idx-hak180",
        content=content or f"doc-{index:02d} " + "证" * 10_000,
        content_hash=f"sha256:{index:064x}",
        scores=(RouteScore(route="hybrid", raw_score=1.0 - index / 100, rank=index + 1),),
    )


def test_production_adapter_reranks_twenty_candidates_in_memory_bounded_batches() -> None:
    model = BoundedBatchReranker()
    candidates = tuple(_rerank_hit(index) for index in range(20))
    adapter = BGERerankerAdapter(
        model=model,
        batch_size=2,
        max_input_tokens=512,
        query_max_tokens=128,
        document_max_tokens=376,
    )

    result = adapter.rerank(
        RerankRequest(
            query="问" * 10_000,
            candidates=candidates,
            config=RerankerConfig(min_top_k=20, max_top_k=20, max_input_tokens=4096),
            trace_id="trace-rerank-bounded-twenty",
        )
    )

    assert result.status == "succeeded"
    assert result.batch_size == 2
    assert result.batch_count == 10
    assert result.truncated_count == 20
    assert result.max_input_tokens == 512
    assert len(model.calls) == 10
    assert all(len(call["pairs"]) <= 2 for call in model.calls)
    assert all(call["batch_size"] == 2 for call in model.calls)
    assert all(call["query_max_length"] == 128 for call in model.calls)
    assert all(call["max_length"] == 512 for call in model.calls)
    assert all(
        len(model.tokenizer.encode(question, add_special_tokens=False))
        + len(model.tokenizer.encode(document, add_special_tokens=False))
        + rerank_service.RERANK_SPECIAL_TOKENS
        <= 512
        for call in model.calls
        for question, document in call["pairs"]
    )
    assert [hit.chunk_id for hit in result.hits] == [
        f"chunk-{index:02d}" for index in reversed(range(20))
    ]
    assert all(hit.content == candidates[int(hit.chunk_id[-2:])].content for hit in result.hits)
    # 10,000 字符原文不会整段进入 Rust/Python tokenizer。
    assert max(model.tokenizer.encoded_character_counts) <= 3_008


class MemoryFailingReranker:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.calls = 0

    def compute_score(self, pairs, **kwargs):
        del pairs, kwargs
        self.calls += 1
        raise MemoryError("simulated bounded reranker OOM")


def test_memory_error_degrades_to_original_fusion_order_without_metadata_mutation() -> None:
    model = MemoryFailingReranker()
    candidates = tuple(_rerank_hit(index, content=f"doc-{index:02d} 正文") for index in range(4))
    adapter = BGERerankerAdapter(model=model, batch_size=2, failure_mode="preserve_order")

    result = adapter.rerank(
        RerankRequest(
            query="安全操作",
            candidates=candidates,
            config=RerankerConfig(min_top_k=1, max_top_k=4),
            trace_id="trace-rerank-memory-fallback",
        )
    )

    assert result.status == "degraded"
    assert result.failure_code == "RERANKER_UNAVAILABLE"
    assert result.hits == candidates
    assert result.batch_count == 1
    assert model.calls == 1
    assert all(
        not any(score.route == "reranker" for score in hit.scores)
        for hit in result.hits
    )


def test_error_failure_mode_returns_typed_retryable_failure() -> None:
    adapter = BGERerankerAdapter(
        model=MemoryFailingReranker(),
        batch_size=1,
        failure_mode="error",
    )

    with pytest.raises(RetrievalEngineError) as caught:
        adapter.rerank(
            RerankRequest(
                query="安全操作",
                candidates=(_rerank_hit(0, content="doc-00 正文"),),
                config=RerankerConfig(),
                trace_id="trace-rerank-memory-error",
            )
        )

    assert getattr(caught.value, "code", None) == "RERANKER_UNAVAILABLE"
    assert getattr(caught.value, "retryable", None) is True
    assert caught.value.details == {"batch": 1, "error_type": "MemoryError"}


def test_reranker_runtime_defaults_and_cross_field_budget_are_validated() -> None:
    settings = AppSettings()

    assert settings.reranker.batch_size == 2
    assert settings.reranker.max_input_tokens == 512
    assert settings.reranker.query_max_tokens == 128
    assert settings.reranker.document_max_tokens == 376
    assert settings.reranker.failure_mode == "preserve_order"
    assert settings.reranker.min_available_physical_mib == 1536
    assert settings.reranker.min_available_commit_mib == 4096
    with pytest.raises(ValueError, match="QUERY_MAX_TOKENS"):
        AppSettings(
            bge_reranker_max_input_tokens=512,
            bge_reranker_query_max_tokens=128,
            bge_reranker_document_max_tokens=377,
        )
