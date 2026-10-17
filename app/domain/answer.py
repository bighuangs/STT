"""可信回答的版本化领域契约（TODO-11）。

``AnswerEnvelope`` 是同步接口、历史记录、离线评测以及未来流式终态的共同
业务对象。模型输出只能提供答案文本和临时 ``[E<n>]`` handle；这里保存的
claim/citation/source 关系均由服务端根据本次 ``ContextPack`` 校验后构造。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ANSWER_SCHEMA_VERSION = "answer.v1"

AnswerFinishReason = Literal[
    "stop",
    "no_evidence",
    "insufficient_evidence",
    "clarification_required",
    "conflicting_evidence",
    "stale_evidence",
    "direct_answer",
    "policy",
    "cancelled",
    "error",
]
AnswerDecisionAction = Literal[
    "ANSWER",
    "PARTIAL_ANSWER",
    "CLARIFY",
    "REFUSE",
    "DIRECT_ANSWER",
]
GroundingStatus = Literal["passed", "degraded", "failed", "not_applicable"]
ConfidenceLevel = Literal["high", "medium", "low", "unknown"]
CitationValidation = Literal["valid", "degraded", "invalid"]


class AnswerClaim(BaseModel):
    """答案中的一个可核验 claim 及其服务端绑定的 citation。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^clm_[0-9]+$")
    text: str = Field(min_length=1, max_length=100_000)
    answer_start: int = Field(ge=0)
    answer_end: int = Field(gt=0)
    claim_type: Literal["factual", "caveat", "uncertainty"] = "factual"
    citation_ids: tuple[str, ...] = Field(default_factory=tuple)
    supported: bool = False

    @model_validator(mode="after")
    def validate_span(self) -> "AnswerClaim":
        if self.answer_end <= self.answer_start:
            raise ValueError("claim answer_end must be greater than answer_start")
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("claim citation_ids must not contain duplicates")
        if self.supported and not self.citation_ids:
            raise ValueError("supported claim requires at least one citation")
        return self


class AnswerCitation(BaseModel):
    """服务端验证后的 claim → Evidence 关系。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str = Field(pattern=r"^cit_[0-9]+$")
    claim_id: str = Field(pattern=r"^clm_[0-9]+$")
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    answer_start: int = Field(ge=0)
    answer_end: int = Field(gt=0)
    validation: CitationValidation
    relation: Literal["SUPPORTS", "CONTRADICTS", "CONTEXT"] = "SUPPORTS"
    verifier_score: float = Field(ge=0.0, le=1.0)
    verifier_version: str = Field(default="citation-lexical.v1", min_length=1, max_length=128)
    source_quote_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_span(self) -> "AnswerCitation":
        if self.answer_end <= self.answer_start:
            raise ValueError("citation answer_end must be greater than answer_start")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("citation reason_codes must not contain duplicates")
        return self


class AnswerSource(BaseModel):
    """允许返回给调用方的 Evidence 展示 DTO，不携带完整正文或内部 metadata。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    source_type: Literal["local", "web"]
    source_label: str = Field(default="", max_length=1024)
    knowledge_base_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=64)
    document_version_id: str | None = Field(default=None, max_length=64)
    index_version_id: str | None = Field(default=None, max_length=64)
    chunk_id: str | None = Field(default=None, max_length=128)
    url: str | None = Field(default=None, max_length=8192)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    location_label: str = Field(default="位置不可用", max_length=2048)
    provenance_spans: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    preview_url: str | None = Field(default=None, max_length=4096)
    modality: Literal["text", "image", "audio", "video"] = "text"
    asset: dict[str, Any] | None = None
    derivation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_source_identity(self) -> "AnswerSource":
        if self.source_type == "local":
            required = (
                self.knowledge_base_id,
                self.document_id,
                self.document_version_id,
                self.index_version_id,
                self.chunk_id,
            )
            if any(value is None for value in required):
                raise ValueError("local answer source requires complete version identity")
            if self.url is not None:
                raise ValueError("local answer source cannot expose a remote URL")
        else:
            if self.url is None:
                raise ValueError("web answer source requires url")
            if self.modality != "text" or self.asset is not None or self.derivation:
                raise ValueError("web answer source cannot expose a local media asset")
        if self.modality == "text" and (self.asset is not None or self.derivation):
            raise ValueError("text answer source cannot carry multimodal metadata")
        if self.modality != "text" and self.asset is None:
            raise ValueError("multimodal answer source requires asset metadata")
        return self


class AnswerConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: ConfidenceLevel = "unknown"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    method: str = Field(default="deterministic-evidence.v1", max_length=128)


class AnswerGrounding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GroundingStatus
    claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    claim_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claims: tuple[str, ...] = Field(default_factory=tuple)
    verifier_version: str = Field(default="grounding.v1", max_length=128)

    @model_validator(mode="after")
    def validate_counts(self) -> "AnswerGrounding":
        if self.supported_claim_count > self.claim_count:
            raise ValueError("supported_claim_count cannot exceed claim_count")
        return self


class AnswerVersions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["answer.v1"] = ANSWER_SCHEMA_VERSION
    prompt_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    model_id: str = Field(default="unknown", max_length=512)
    config_version: str = Field(default="cfg_unknown", max_length=128)
    index_version_ids: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_profile_id: str = Field(default="", max_length=128)
    retrieval_profile_version: int = Field(default=0, ge=0)
    query_plan_version: str = Field(default="query-plan.v1", max_length=128)
    fusion_version: str = Field(default="fusion.v1", max_length=128)
    context_pack_version: str = Field(default="context-pack.v1", max_length=128)
    assessor_version: str = Field(default="evidence-assessor.v1", max_length=128)


class AnswerUsage(BaseModel):
    """Token/cost 无法由 provider 提供时必须显式标记 unavailable。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "unavailable", "estimated"] = "unavailable"
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0.0)
    currency: str | None = Field(default=None, max_length=16)
    reason: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_availability(self) -> "AnswerUsage":
        if self.status == "unavailable" and not self.reason:
            raise ValueError("unavailable usage requires reason")
        return self


class AnswerEnvelope(BaseModel):
    """查询终态的唯一权威表示。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["answer.v1"] = ANSWER_SCHEMA_VERSION
    answer_id: str = Field(pattern=r"^ans_[0-9a-f]{32}$")
    query_run_id: str = Field(pattern=r"^qry_[0-9a-f]{32}$")
    session_id: str = Field(min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(max_length=1_000_000)
    decision: AnswerDecisionAction
    claims: tuple[AnswerClaim, ...] = Field(default_factory=tuple)
    citations: tuple[AnswerCitation, ...] = Field(default_factory=tuple)
    sources: tuple[AnswerSource, ...] = Field(default_factory=tuple)
    confidence: AnswerConfidence
    grounding: AnswerGrounding
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = Field(default_factory=tuple)
    finish_reason: AnswerFinishReason
    versions: AnswerVersions
    usage: AnswerUsage
    trace_id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> "AnswerEnvelope":
        if len(self.degradation_reasons) != len(set(self.degradation_reasons)):
            raise ValueError("degradation_reasons must not contain duplicates")

        claim_ids = {claim.claim_id for claim in self.claims}
        citation_ids = {citation.citation_id for citation in self.citations}
        claims_by_id = {claim.claim_id: claim for claim in self.claims}
        citations_by_id = {citation.citation_id: citation for citation in self.citations}
        source_ids = {source.evidence_id for source in self.sources}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim_id must be unique")
        if len(citation_ids) != len(self.citations):
            raise ValueError("citation_id must be unique")
        if len(source_ids) != len(self.sources):
            raise ValueError("source evidence_id must be unique")

        for claim in self.claims:
            if claim.answer_end > len(self.answer):
                raise ValueError("claim span exceeds answer length")
            if self.answer[claim.answer_start : claim.answer_end] != claim.text:
                raise ValueError("claim text must equal its answer span")
            if not set(claim.citation_ids).issubset(citation_ids):
                raise ValueError("claim references an unknown citation")
            if any(
                citations_by_id[citation_id].claim_id != claim.claim_id
                for citation_id in claim.citation_ids
            ):
                raise ValueError("claim cannot reference a citation owned by another claim")
        for citation in self.citations:
            if citation.claim_id not in claim_ids:
                raise ValueError("citation references an unknown claim")
            if citation.evidence_id not in source_ids:
                raise ValueError("citation references evidence outside source whitelist")
            if citation.answer_end > len(self.answer):
                raise ValueError("citation span exceeds answer length")
            owner = claims_by_id[citation.claim_id]
            if (
                citation.answer_start != owner.answer_start
                or citation.answer_end != owner.answer_end
            ):
                raise ValueError("citation span must exactly match its owner claim span")

        trusted = {
            citation.citation_id
            for citation in self.citations
            if citation.validation in {"valid", "degraded"}
        }
        for claim in self.claims:
            expected = bool(set(claim.citation_ids) & trusted)
            if claim.supported != expected:
                raise ValueError("claim supported flag disagrees with trusted citations")

        if self.degraded != bool(self.degradation_reasons):
            raise ValueError("degraded must match degradation_reasons emptiness")
        if self.decision in {"ANSWER", "PARTIAL_ANSWER"} and not self.answer.strip():
            raise ValueError("answer decisions require non-empty answer text")
        if self.decision == "ANSWER" and self.grounding.status == "failed":
            raise ValueError("failed grounding cannot be emitted as ANSWER")
        return self


__all__ = [
    "ANSWER_SCHEMA_VERSION",
    "AnswerCitation",
    "AnswerClaim",
    "AnswerConfidence",
    "AnswerDecisionAction",
    "AnswerEnvelope",
    "AnswerFinishReason",
    "AnswerGrounding",
    "AnswerSource",
    "AnswerUsage",
    "AnswerVersions",
]
