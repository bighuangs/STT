"""回答契约的机器可判定失败语义。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from app.domain.answering._base import SHA256_PATTERN, AnsweringContractModel


class FailureStage(StrEnum):
    """失败发生的确定性契约阶段。"""

    QUESTION = "question"
    WITNESS = "witness"
    PLAN = "plan"
    CLAIM_PROOF = "claim_proof"
    OBLIGATION_PROOF = "obligation_proof"
    ANSWER_PROOF = "answer_proof"


class Retryability(StrEnum):
    """失败恢复所需的最小重试范围。"""

    NONE = "none"
    SAME_CONTEXT = "same_context"
    NEW_RETRIEVAL = "new_retrieval"
    NEW_GENERATION = "new_generation"


class AnsweringFailureCode(StrEnum):
    """跨层稳定、不可携带自由诊断正文的失败枚举。"""

    QUESTION_EMPTY = "question_empty"
    QUESTION_HASH_MISMATCH = "question_hash_mismatch"
    OBLIGATION_DUPLICATE = "obligation_duplicate"
    OBLIGATION_SOURCE_INVALID = "obligation_source_invalid"
    WITNESS_OUTSIDE_CONTEXT_PACK = "witness_outside_context_pack"
    WITNESS_OUTSIDE_AUTHORIZED_SCOPE = "witness_outside_authorized_scope"
    WITNESS_SPAN_INVALID = "witness_span_invalid"
    WITNESS_SPAN_HASH_MISMATCH = "witness_span_hash_mismatch"
    WITNESS_SOURCE_MAPPING_UNPROVEN = "witness_source_mapping_unproven"
    WITNESS_CATALOG_TRUNCATED = "witness_catalog_truncated"
    WITNESS_CATALOG_INDETERMINATE = "witness_catalog_indeterminate"
    PLAN_UNKNOWN_OBLIGATION = "plan_unknown_obligation"
    PLAN_UNKNOWN_WITNESS = "plan_unknown_witness"
    PLAN_UNSUPPORTED_RENDERING_RULE = "plan_unsupported_rendering_rule"
    PLAN_DECISION_INCONSISTENT = "plan_decision_inconsistent"
    CLAIM_RENDER_MISMATCH = "claim_render_mismatch"
    CLAIM_GROUNDING_FAILED = "claim_grounding_failed"
    CLAIM_RELEVANCE_FAILED = "claim_relevance_failed"
    CLAIM_PROOF_INDETERMINATE = "claim_proof_indeterminate"
    OBLIGATION_UNCOVERED = "obligation_uncovered"
    OBLIGATION_CARDINALITY_MISMATCH = "obligation_cardinality_mismatch"
    OBLIGATION_CLOSURE_UNPROVEN = "obligation_closure_unproven"
    OBLIGATION_ABSENCE_UNPROVEN = "obligation_absence_unproven"
    OBLIGATION_CONFLICT_UNRESOLVED = "obligation_conflict_unresolved"
    ANSWER_COMPLETENESS_FAILED = "answer_completeness_failed"
    ANSWER_COMPLETENESS_INDETERMINATE = "answer_completeness_indeterminate"
    REFUSAL_NOT_AUTHORIZED = "refusal_not_authorized"
    FINAL_ANSWER_HASH_MISMATCH = "final_answer_hash_mismatch"
    PROOF_GRAPH_INCONSISTENT = "proof_graph_inconsistent"
    VERIFIER_INTERNAL_ERROR = "verifier_internal_error"


class AnsweringFailure(AnsweringContractModel):
    """可持久化但不泄露正文的结构化失败。"""

    stage: FailureStage
    code: AnsweringFailureCode
    subject_id: str | None = Field(default=None, min_length=1, max_length=128)
    retryability: Retryability = Retryability.NONE
    diagnostic_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("subject_id")
    @classmethod
    def validate_subject_id(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("subject_id must not contain surrounding whitespace")
        return value


__all__ = [
    "AnsweringFailure",
    "AnsweringFailureCode",
    "FailureStage",
    "Retryability",
]
