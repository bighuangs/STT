"""TODO-13 四类模型的供应商无关端口与标准化输入输出。"""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.model_routing import ModelErrorCode, ModelSnapshot, ModelUsage


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=1_000_000)
    name: str | None = Field(default=None, max_length=128)


class TextModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: ModelSnapshot
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    response_schema: dict[str, Any] | None = None
    timeout_ms: int = Field(default=30_000, ge=1, le=600_000)
    stream: bool = False


class TextModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    usage: ModelUsage
    finish_reason: str = Field(default="stop", max_length=128)
    provider_request_id: str | None = Field(default=None, max_length=256)


class VisionModelRequest(TextModelRequest):
    image_refs: tuple[str, ...] = Field(min_length=1)


class EmbeddingModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: ModelSnapshot
    texts: tuple[str, ...] = Field(min_length=1)
    timeout_ms: int = Field(default=30_000, ge=1, le=600_000)


class EmbeddingModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vectors: tuple[tuple[float, ...], ...]
    dimension: int = Field(ge=1, le=65_536)
    usage: ModelUsage

    @model_validator(mode="after")
    def validate_vectors(self) -> "EmbeddingModelResponse":
        if not self.vectors:
            raise ValueError("embedding response requires vectors")
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("every vector must match response dimension")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("embedding vectors must contain finite values")
        return self


class RerankModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: ModelSnapshot
    query: str = Field(min_length=1, max_length=100_000)
    documents: tuple[str, ...] = Field(min_length=1)
    timeout_ms: int = Field(default=30_000, ge=1, le=600_000)


class RerankModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: tuple[float, ...]
    usage: ModelUsage

    @field_validator("scores")
    @classmethod
    def finite_scores(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("rerank response requires scores")
        if any(not math.isfinite(score) for score in value):
            raise ValueError("rerank scores must be finite")
        return value


class QueryModelCallTransportUnavailable(ConnectionError):
    """Query 模型传输重试耗尽后的稳定、无正文应用层错误。"""

    code = "QUERY_MODEL_CALL_TRANSPORT_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__("query model transport unavailable")


class ProviderCallError(RuntimeError):
    """适配器必须只向应用层暴露这一稳定错误。"""

    def __init__(
        self,
        code: ModelErrorCode,
        message: str,
        *,
        retryable: bool,
        safe_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        self.safe_details = dict(safe_details or {})


@runtime_checkable
class ChatModelPort(Protocol):
    def invoke_chat(self, request: TextModelRequest) -> TextModelResponse: ...


@runtime_checkable
class VisionModelPort(Protocol):
    def invoke_vision(self, request: VisionModelRequest) -> TextModelResponse: ...


@runtime_checkable
class EmbeddingModelPort(Protocol):
    def embed(self, request: EmbeddingModelRequest) -> EmbeddingModelResponse: ...


@runtime_checkable
class RerankerModelPort(Protocol):
    def rerank(self, request: RerankModelRequest) -> RerankModelResponse: ...


__all__ = [
    "ChatMessage",
    "ChatModelPort",
    "EmbeddingModelPort",
    "EmbeddingModelRequest",
    "EmbeddingModelResponse",
    "ProviderCallError",
    "RerankModelRequest",
    "RerankModelResponse",
    "RerankerModelPort",
    "TextModelRequest",
    "TextModelResponse",
    "VisionModelPort",
    "VisionModelRequest",
]
