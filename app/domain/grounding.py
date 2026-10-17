"""证据充分性与回答决策领域对象（TODO-30）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EvidenceStatus = Literal["SUFFICIENT", "PARTIAL", "NONE", "CONFLICTING", "STALE"]
DecisionAction = Literal[
    "ANSWER",
    "PARTIAL_ANSWER",
    "CLARIFY",
    "REFUSE",
    "DIRECT_ANSWER",
]


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EvidenceStatus
    query_coverage: float = Field(ge=0.0, le=1.0)
    sub_query_coverage: dict[str, bool] = Field(default_factory=dict)
    source_diversity: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    packed_evidence_count: int = Field(default=0, ge=0)
    top_scores: dict[str, float] = Field(default_factory=dict)
    trust_summary: dict[str, Any] = Field(default_factory=dict)
    freshness_summary: dict[str, Any] = Field(default_factory=dict)
    missing_aspects: tuple[str, ...] = Field(default_factory=tuple)
    conflicts: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    assessor_version: str = "evidence-assessor.v1"

    @field_validator("top_scores")
    @classmethod
    def finite_scores(cls, value: dict[str, float]) -> dict[str, float]:
        import math

        if any(not math.isfinite(float(score)) for score in value.values()):
            raise ValueError("top_scores must contain finite values")
        return value


class AnswerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: DecisionAction
    allowed_claim_scope: tuple[str, ...] = Field(default_factory=tuple)
    caveats: tuple[str, ...] = Field(default_factory=tuple)
    reason_code: str = Field(min_length=1, max_length=128)
    user_message: str = Field(default="", max_length=100_000)
    repair_budget: int = Field(default=0, ge=0, le=2)
    decision_version: str = "answer-decision.v1"

    @model_validator(mode="after")
    def validate_terminal_message(self) -> "AnswerDecision":
        if self.action in {"CLARIFY", "REFUSE", "DIRECT_ANSWER"} and not self.user_message:
            raise ValueError(f"{self.action} requires user_message")
        return self


__all__ = [
    "AnswerDecision",
    "DecisionAction",
    "EvidenceAssessment",
    "EvidenceStatus",
]
