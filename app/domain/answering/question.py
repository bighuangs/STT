"""由原始 QueryFacet 派生的问题义务契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Mapping, Self

from pydantic import Field, field_validator, model_validator

from app.domain.answering._base import (
    FACET_ID_PATTERN,
    SHA256_PATTERN,
    AnsweringContractModel,
    require_unique,
    sha256_digest,
    stable_content_hash,
    stable_identifier,
)
from app.domain.query_planning import (
    FacetClosure,
    FacetNecessity,
    FacetPolarity,
    FacetSupportStatus,
    QueryFacet,
)


QUESTION_OBLIGATION_SCHEMA_VERSION = "question-obligation.v1"
QUESTION_CONTRACT_SCHEMA_VERSION = "question-contract.v1"


class ObligationKind(StrEnum):
    """义务的确定性回答形态类别。"""

    FACT = "fact"
    BINARY = "binary"
    ENUMERATION = "enumeration"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"


class RequiredOutputForm(StrEnum):
    """服务端渲染器必须满足的输出形态。"""

    FACT = "fact"
    DIRECT_VERDICT = "direct_verdict"
    LIST = "list"
    ORDERED_STEPS = "ordered_steps"
    COMPARISON = "comparison"


def _obligation_payload(
    *,
    obligation_id: str,
    information_need: str,
    retrieval_expression: str,
    necessity: FacetNecessity,
    polarity: FacetPolarity,
    closure: FacetClosure,
    expected_count: int | None,
    sub_query_ids: tuple[str, ...],
    kind: ObligationKind,
    required_output_form: RequiredOutputForm,
) -> dict[str, Any]:
    return {
        "schema_version": QUESTION_OBLIGATION_SCHEMA_VERSION,
        "obligation_id": obligation_id,
        "information_need": information_need,
        "retrieval_expression": retrieval_expression,
        "necessity": necessity.value,
        "polarity": polarity.value,
        "closure": closure.value,
        "expected_count": expected_count,
        "sub_query_ids": list(sub_query_ids),
        "kind": kind.value,
        "required_output_form": required_output_form.value,
    }


class QuestionObligation(AnsweringContractModel):
    """原始 facet 的不可变语义快照；obligation_id 即 facet_id。"""

    schema_version: Literal["question-obligation.v1"] = QUESTION_OBLIGATION_SCHEMA_VERSION
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    information_need: str = Field(min_length=1, max_length=16_384)
    retrieval_expression: str = Field(min_length=1, max_length=16_384)
    necessity: FacetNecessity = FacetNecessity.REQUIRED
    polarity: FacetPolarity = FacetPolarity.AFFIRMATIVE
    closure: FacetClosure = FacetClosure.SINGLE
    expected_count: int | None = Field(default=None, ge=1, le=128)
    sub_query_ids: tuple[str, ...] = Field(default=(), max_length=8)
    kind: ObligationKind = ObligationKind.FACT
    required_output_form: RequiredOutputForm = RequiredOutputForm.FACT
    obligation_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("information_need", "retrieval_expression")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("obligation text cannot be blank")
        return value

    @field_validator("sub_query_ids")
    @classmethod
    def validate_sub_query_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(value, field_name="sub_query_ids")
        if any(not item.startswith("sq_") for item in value):
            raise ValueError("sub_query_ids must use the sq_ identity namespace")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _obligation_payload(
            obligation_id=self.obligation_id,
            information_need=self.information_need,
            retrieval_expression=self.retrieval_expression,
            necessity=self.necessity,
            polarity=self.polarity,
            closure=self.closure,
            expected_count=self.expected_count,
            sub_query_ids=self.sub_query_ids,
            kind=self.kind,
            required_output_form=self.required_output_form,
        )

    @model_validator(mode="after")
    def validate_semantics_and_hash(self) -> Self:
        if self.closure == FacetClosure.CLOSED_COUNT:
            if self.expected_count is None:
                raise ValueError("closed_count obligation requires expected_count")
        elif self.expected_count is not None:
            raise ValueError("only closed_count obligation may carry expected_count")

        expected_forms = {
            ObligationKind.BINARY: RequiredOutputForm.DIRECT_VERDICT,
            ObligationKind.ENUMERATION: RequiredOutputForm.LIST,
            ObligationKind.PROCEDURE: RequiredOutputForm.ORDERED_STEPS,
            ObligationKind.COMPARISON: RequiredOutputForm.COMPARISON,
        }
        expected_form = expected_forms.get(self.kind)
        if expected_form is not None and self.required_output_form != expected_form:
            raise ValueError("obligation kind disagrees with required_output_form")
        if self.closure != FacetClosure.SINGLE and self.required_output_form not in {
            RequiredOutputForm.LIST,
            RequiredOutputForm.ORDERED_STEPS,
            RequiredOutputForm.COMPARISON,
        }:
            raise ValueError("multi-item closure requires a structured multi-item output")
        if self.closure == FacetClosure.SINGLE and self.required_output_form in {
            RequiredOutputForm.LIST,
            RequiredOutputForm.ORDERED_STEPS,
            RequiredOutputForm.COMPARISON,
        }:
            raise ValueError("single closure cannot require a multi-item output form")
        if stable_content_hash(self._identity_payload()) != self.obligation_hash:
            raise ValueError("obligation_hash does not match the facet snapshot")
        return self

    @classmethod
    def from_facet(
        cls,
        facet: QueryFacet,
        *,
        kind: ObligationKind | None = None,
        required_output_form: RequiredOutputForm | None = None,
    ) -> "QuestionObligation":
        """从未经运行期绑定的原始 facet 建立唯一义务。"""

        if facet.support_status != FacetSupportStatus.UNASSESSED or facet.evidence_ids:
            raise ValueError("question obligations must be derived from an unassessed facet")
        resolved_kind = kind or (
            ObligationKind.ENUMERATION
            if facet.closure != FacetClosure.SINGLE
            else ObligationKind.FACT
        )
        resolved_form = required_output_form or {
            ObligationKind.BINARY: RequiredOutputForm.DIRECT_VERDICT,
            ObligationKind.ENUMERATION: RequiredOutputForm.LIST,
            ObligationKind.PROCEDURE: RequiredOutputForm.ORDERED_STEPS,
            ObligationKind.COMPARISON: RequiredOutputForm.COMPARISON,
        }.get(resolved_kind, RequiredOutputForm.FACT)
        payload = _obligation_payload(
            obligation_id=facet.facet_id,
            information_need=facet.information_need,
            retrieval_expression=facet.retrieval_expression,
            necessity=facet.necessity,
            polarity=facet.polarity,
            closure=facet.closure,
            expected_count=facet.expected_count,
            sub_query_ids=facet.sub_query_ids,
            kind=resolved_kind,
            required_output_form=resolved_form,
        )
        return cls(**payload, obligation_hash=stable_content_hash(payload))


def _question_payload(
    *,
    query_plan_id: str,
    source_query: str,
    source_query_hash: str,
    builder_version: str,
    obligations: tuple[QuestionObligation, ...],
) -> dict[str, Any]:
    return {
        "schema_version": QUESTION_CONTRACT_SCHEMA_VERSION,
        "query_plan_id": query_plan_id,
        "source_query": source_query,
        "source_query_hash": source_query_hash,
        "builder_version": builder_version,
        "obligations": [item.to_dict() for item in obligations],
    }


class QuestionContract(AnsweringContractModel):
    """回答规划、证明和完整性校验共享的唯一问题契约。"""

    schema_version: Literal["question-contract.v1"] = QUESTION_CONTRACT_SCHEMA_VERSION
    contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    contract_hash: str = Field(pattern=SHA256_PATTERN)
    query_plan_id: str = Field(pattern=r"^qpl_[0-9a-f]{32}$")
    source_query: str = Field(min_length=1, max_length=16_384)
    source_query_hash: str = Field(pattern=SHA256_PATTERN)
    builder_version: str = Field(min_length=1, max_length=128)
    obligations: tuple[QuestionObligation, ...] = Field(default=(), max_length=32)

    @field_validator("source_query", "builder_version")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("question contract text fields must be trimmed and non-blank")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _question_payload(
            query_plan_id=self.query_plan_id,
            source_query=self.source_query,
            source_query_hash=self.source_query_hash,
            builder_version=self.builder_version,
            obligations=self.obligations,
        )

    @model_validator(mode="after")
    def validate_identity_graph(self) -> Self:
        identifiers = tuple(item.obligation_id for item in self.obligations)
        require_unique(identifiers, field_name="obligation_ids")
        if self.obligations and not any(
            item.necessity == FacetNecessity.REQUIRED for item in self.obligations
        ):
            raise ValueError("non-empty question contract requires a required obligation")
        if self.source_query_hash != sha256_digest(self.source_query):
            raise ValueError("source_query_hash does not match source_query")
        payload = self._identity_payload()
        if self.contract_hash != stable_content_hash(payload):
            raise ValueError("contract_hash does not match the question contract")
        if self.contract_id != stable_identifier("qct_", payload):
            raise ValueError("contract_id does not match the question contract")
        return self

    @property
    def required_obligation_ids(self) -> tuple[str, ...]:
        """按原始 facet 顺序返回所有强制义务。"""

        return tuple(
            item.obligation_id
            for item in self.obligations
            if item.necessity == FacetNecessity.REQUIRED
        )

    @classmethod
    def from_facets(
        cls,
        *,
        query_plan_id: str,
        source_query: str,
        facets: tuple[QueryFacet, ...],
        builder_version: str,
        kind_by_obligation: Mapping[str, ObligationKind] | None = None,
        output_form_by_obligation: Mapping[str, RequiredOutputForm] | None = None,
    ) -> "QuestionContract":
        """从原始 facet 序列建立可重放契约，不复制运行期 Evidence 绑定。"""

        kinds = kind_by_obligation or {}
        output_forms = output_form_by_obligation or {}
        obligations = tuple(
            QuestionObligation.from_facet(
                facet,
                kind=kinds.get(facet.facet_id),
                required_output_form=output_forms.get(facet.facet_id),
            )
            for facet in facets
        )
        source_query_hash = sha256_digest(source_query)
        payload = _question_payload(
            query_plan_id=query_plan_id,
            source_query=source_query,
            source_query_hash=source_query_hash,
            builder_version=builder_version,
            obligations=obligations,
        )
        return cls(
            **payload,
            contract_id=stable_identifier("qct_", payload),
            contract_hash=stable_content_hash(payload),
        )

    def validate_source_facets(self, facets: tuple[QueryFacet, ...]) -> Self:
        """验证反序列化契约仍与同序原始 facets 精确一致。"""

        if len(facets) != len(self.obligations):
            raise ValueError("question contract facet count mismatch")
        for obligation, facet in zip(self.obligations, facets, strict=True):
            expected = QuestionObligation.from_facet(
                facet,
                kind=obligation.kind,
                required_output_form=obligation.required_output_form,
            )
            if obligation != expected:
                raise ValueError("question obligation does not match its source facet")
        return self


__all__ = [
    "QUESTION_CONTRACT_SCHEMA_VERSION",
    "QUESTION_OBLIGATION_SCHEMA_VERSION",
    "ObligationKind",
    "QuestionContract",
    "QuestionObligation",
    "RequiredOutputForm",
]
