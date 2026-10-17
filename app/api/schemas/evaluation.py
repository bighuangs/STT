"""TODO-18 评测、反馈与人工标注 API schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.evaluation import freeze_evaluation_gate_policy


class DatasetImportRequest(BaseModel):
    """发布一个不可变的评测集版本。"""

    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any]
    parent_dataset_id: str | None = Field(default=None, min_length=1, max_length=96)

    @field_validator("manifest")
    @classmethod
    def validate_gate_policy(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Keep the original manifest for its immutable checksum. This check only
        # turns invalid thresholds or versions into a stable API-side 422. v3
        # remains an explicit manifest opt-in; omission is materialized as v2.
        freeze_evaluation_gate_policy(value.get("gate_policy"))
        return value


class EvaluationPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=96)
    kb_id: str = Field(min_length=1, max_length=96)
    mode: Literal["live", "retrieval_only"] = "retrieval_only"
    retrieval_profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    retrieval_profile_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def complete_profile_pin(self) -> "EvaluationPreflightRequest":
        if (self.retrieval_profile_id is None) != (
            self.retrieval_profile_version is None
        ):
            raise ValueError(
                "retrieval_profile_id 与 retrieval_profile_version 必须同时提供"
            )
        return self


class EvaluationRunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=96)
    kb_id: str = Field(min_length=1, max_length=96)
    mode: Literal["live", "retrieval_only"] = "live"
    name: str | None = Field(default=None, min_length=1, max_length=255)
    baseline_run_id: str | None = Field(default=None, min_length=1, max_length=96)
    case_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)
    retrieval_profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    retrieval_profile_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def complete_profile_pin(self) -> "EvaluationRunCreateRequest":
        if (self.retrieval_profile_id is None) != (
            self.retrieval_profile_version is None
        ):
            raise ValueError(
                "retrieval_profile_id 与 retrieval_profile_version 必须同时提供"
            )
        return self

    @field_validator("case_ids")
    @classmethod
    def validate_case_ids(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 255 for value in normalized):
            raise ValueError("case_ids 中每个标识必须包含 1..255 个字符")
        if len(normalized) != len(set(normalized)):
            raise ValueError("case_ids 不能重复")
        return normalized


class FeedbackCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=96)
    answer_id: str = Field(min_length=1, max_length=255)
    evaluation_result_id: str | None = Field(default=None, min_length=1, max_length=96)
    session_id: str | None = Field(default=None, min_length=1, max_length=255)
    rating: Literal[-1, 0, 1]
    reason_codes: list[str] = Field(default_factory=list, max_length=20)
    comment: str = Field(default="", max_length=4000)
    selected_citation: dict[str, Any] = Field(default_factory=dict)
    corrected_answer: str = Field(default="", max_length=12000)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 64 for value in normalized):
            raise ValueError("reason_codes 中每项必须包含 1..64 个字符")
        if len(normalized) != len(set(normalized)):
            raise ValueError("reason_codes 不能重复")
        return normalized


class FeedbackTriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # accepted 必须经过双人标注审核或冲突仲裁，不能由普通分诊直接写入。
    status: Literal["triaged", "in_review", "rejected"]
    review_note: str = Field(default="", max_length=4000)


class AnnotationTaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(min_length=1, max_length=96)
    target_dataset_key: str = Field(min_length=1, max_length=255)
    proposed_case: dict[str, Any]


class AnnotationReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=4000)


class AnnotationArbitrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=4000)
    resolved_case: dict[str, Any] | None = None


EvaluationArtifact = Literal[
    "json",
    "markdown",
    "html",
    "summary",
    "manifest",
    "results",
    "comparison",
]
