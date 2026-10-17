"""默认关闭的模型适配器与纯本地确定性测试适配器。

这里不发起 HTTP，不读取 credential。真实 provider adapter 必须由组合根显式注入；
未注入时使用 ``DisabledRemoteModelAdapter`` fail closed。
"""

from __future__ import annotations

import hashlib
import math

from app.application.model_ports import (
    EmbeddingModelRequest,
    EmbeddingModelResponse,
    ProviderCallError,
    RerankModelRequest,
    RerankModelResponse,
    TextModelRequest,
    TextModelResponse,
    VisionModelRequest,
)
from app.domain.model_routing import ModelUsage


class DisabledRemoteModelAdapter:
    def _disabled(self) -> None:
        raise ProviderCallError(
            "policy_denied",
            "remote model adapter is disabled",
            retryable=False,
        )

    def invoke_chat(self, request: TextModelRequest) -> TextModelResponse:
        del request
        self._disabled()

    def invoke_vision(self, request: VisionModelRequest) -> TextModelResponse:
        del request
        self._disabled()

    def embed(self, request: EmbeddingModelRequest) -> EmbeddingModelResponse:
        del request
        self._disabled()

    def rerank(self, request: RerankModelRequest) -> RerankModelResponse:
        del request
        self._disabled()


class DeterministicLocalModelAdapter:
    """仅供离线测试的可复现 adapter，不代表生产模型能力。"""

    def invoke_chat(self, request: TextModelRequest) -> TextModelResponse:
        self._require_capability(request.snapshot.capability, "chat")
        text = "\n".join(message.content for message in request.messages)
        return TextModelResponse(
            text=f"deterministic:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
            usage=ModelUsage(input_tokens=len(text), output_tokens=1),
        )

    def invoke_vision(self, request: VisionModelRequest) -> TextModelResponse:
        self._require_capability(request.snapshot.capability, "vision")
        text = "|".join(request.image_refs)
        return TextModelResponse(
            text=f"vision:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
            usage=ModelUsage(input_tokens=len(text), output_tokens=1),
        )

    def embed(self, request: EmbeddingModelRequest) -> EmbeddingModelResponse:
        self._require_capability(request.snapshot.capability, "embedding")
        dimension = request.snapshot.embedding_dimension
        if dimension is None:
            raise ProviderCallError(
                "invalid_response",
                "embedding snapshot has no dimension",
                retryable=False,
            )
        vectors = tuple(self._vector(text, dimension) for text in request.texts)
        return EmbeddingModelResponse(
            vectors=vectors,
            dimension=dimension,
            usage=ModelUsage(input_tokens=sum(len(text) for text in request.texts)),
        )

    def rerank(self, request: RerankModelRequest) -> RerankModelResponse:
        self._require_capability(request.snapshot.capability, "rerank")
        query_tokens = set(request.query.lower().split())
        scores = tuple(
            len(query_tokens & set(document.lower().split())) / max(1, len(query_tokens))
            for document in request.documents
        )
        return RerankModelResponse(
            scores=scores,
            usage=ModelUsage(input_tokens=len(request.query) + sum(map(len, request.documents))),
        )

    @staticmethod
    def _vector(text: str, dimension: int) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(dimension)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return tuple(value / norm for value in raw)

    @staticmethod
    def _require_capability(actual: str, expected: str) -> None:
        if actual != expected:
            raise ProviderCallError(
                "policy_denied",
                f"snapshot capability must be {expected}",
                retryable=False,
            )


__all__ = ["DeterministicLocalModelAdapter", "DisabledRemoteModelAdapter"]
