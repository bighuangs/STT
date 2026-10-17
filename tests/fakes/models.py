"""Small deterministic model fakes with explicit failure injection."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


class InjectedModelFailure(RuntimeError):
    """Raised when a fake is configured to fail a particular call."""


@dataclass
class _FaultPlan:
    fail_on_calls: set[int] = field(default_factory=set)
    timeout_on_calls: set[int] = field(default_factory=set)
    call_count: int = 0

    def before_call(self) -> int:
        self.call_count += 1
        if self.call_count in self.timeout_on_calls:
            raise TimeoutError(f"injected timeout on call {self.call_count}")
        if self.call_count in self.fail_on_calls:
            raise InjectedModelFailure(f"injected failure on call {self.call_count}")
        return self.call_count


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    return str(content if content is not None else value)


@dataclass
class FakeLLM:
    """Chat fake returning queued responses or a stable echo of the request."""

    responses: list[str] = field(default_factory=list)
    fail_on_calls: set[int] = field(default_factory=set)
    timeout_on_calls: set[int] = field(default_factory=set)
    calls: list[tuple[str, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._faults = _FaultPlan(set(self.fail_on_calls), set(self.timeout_on_calls))

    def invoke(self, messages: Sequence[Any] | str, **_: Any) -> str:
        call_no = self._faults.before_call()
        normalized = (
            (_text(messages),)
            if isinstance(messages, str)
            else tuple(_text(message) for message in messages)
        )
        self.calls.append(normalized)
        if call_no <= len(self.responses):
            return self.responses[call_no - 1]
        digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:12]
        return f"fake-llm:{digest}"

    async def ainvoke(self, messages: Sequence[Any] | str, **kwargs: Any) -> str:
        return self.invoke(messages, **kwargs)

    def stream(self, messages: Sequence[Any] | str, **kwargs: Any) -> Iterable[str]:
        yield self.invoke(messages, **kwargs)


@dataclass
class FakeEmbedding:
    """Hash-based, L2-normalized embeddings that never load a model or network."""

    dimension: int = 8
    fail_on_calls: set[int] = field(default_factory=set)
    timeout_on_calls: set[int] = field(default_factory=set)
    calls: list[tuple[str, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self._faults = _FaultPlan(set(self.fail_on_calls), set(self.timeout_on_calls))

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values = [
            ((seed[index % len(seed)] / 255.0) * 2.0) - 1.0
            for index in range(self.dimension)
        ]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 12) for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._faults.before_call()
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        self.calls.append(tuple(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class _FakeTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[str]:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+|[^\s]", text.lower())
        return ["[CLS]", *tokens, "[SEP]"] if add_special_tokens else tokens


@dataclass
class FakeReranker:
    """Lexical-overlap reranker compatible with ``compute_score`` callers."""

    fail_on_calls: set[int] = field(default_factory=set)
    timeout_on_calls: set[int] = field(default_factory=set)
    calls: list[tuple[tuple[str, str], ...]] = field(default_factory=list, init=False)
    tokenizer: _FakeTokenizer = field(default_factory=_FakeTokenizer, init=False)

    def __post_init__(self) -> None:
        self._faults = _FaultPlan(set(self.fail_on_calls), set(self.timeout_on_calls))

    def compute_score(
        self,
        pairs: list[list[str]] | list[tuple[str, str]],
        normalize: bool = True,
    ) -> list[float]:
        self._faults.before_call()
        normalized_pairs = tuple((str(query), str(document)) for query, document in pairs)
        self.calls.append(normalized_pairs)
        scores: list[float] = []
        for query, document in normalized_pairs:
            query_tokens = set(self.tokenizer.encode(query))
            document_tokens = set(self.tokenizer.encode(document))
            union = query_tokens | document_tokens
            score = len(query_tokens & document_tokens) / len(union) if union else 0.0
            scores.append(round(score if normalize else score * 10.0, 12))
        return scores
