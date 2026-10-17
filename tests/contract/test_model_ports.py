from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

import pytest

from tests.fakes import FakeEmbedding, FakeLLM, FakeReranker, InjectedModelFailure


@runtime_checkable
class ChatPort(Protocol):
    def invoke(self, messages: list[Any] | str, **kwargs: Any) -> str: ...


@runtime_checkable
class EmbeddingPort(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class RerankerPort(Protocol):
    def compute_score(
        self, pairs: list[list[str]], normalize: bool = True
    ) -> list[float]: ...


def test_fakes_satisfy_runtime_model_ports():
    assert isinstance(FakeLLM(), ChatPort)
    assert isinstance(FakeEmbedding(), EmbeddingPort)
    assert isinstance(FakeReranker(), RerankerPort)


def test_llm_is_deterministic_and_records_normalized_calls():
    first = FakeLLM()
    second = FakeLLM()
    assert first.invoke(["system", "同一个问题"]) == second.invoke(
        ["system", "同一个问题"]
    )
    assert first.calls == [("system", "同一个问题")]


def test_llm_async_contract_uses_the_same_fault_plan():
    model = FakeLLM(responses=["ok"], timeout_on_calls={2})
    assert asyncio.run(model.ainvoke("first")) == "ok"
    with pytest.raises(TimeoutError, match="call 2"):
        asyncio.run(model.ainvoke("second"))


def test_embedding_dimension_normalization_and_batch_order():
    model = FakeEmbedding(dimension=6)
    vectors = model.embed_documents(["alpha", "beta", "alpha"])
    assert len(vectors) == 3
    assert all(len(vector) == 6 for vector in vectors)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0, abs=1e-9)


def test_reranker_order_is_deterministic_and_normalized():
    model = FakeReranker()
    scores = model.compute_score(
        [["压力 测量", "压力 测量 方法"], ["压力 测量", "温度 校准"]],
        normalize=True,
    )
    assert scores[0] > scores[1]
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.parametrize("factory", [FakeLLM, FakeEmbedding, FakeReranker])
def test_every_fake_can_inject_failure_on_an_exact_call(factory):
    model = factory(fail_on_calls={2})
    if isinstance(model, FakeLLM):
        call = lambda: model.invoke("input")
    elif isinstance(model, FakeEmbedding):
        call = lambda: model.embed_documents(["input"])
    else:
        call = lambda: model.compute_score([["q", "d"]])
    call()
    with pytest.raises(InjectedModelFailure, match="call 2"):
        call()
