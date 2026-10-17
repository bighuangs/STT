"""只引用义务与 Witness 的确定性回答计划。"""

from __future__ import annotations

from enum import StrEnum
import re
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from app.domain.answering._base import (
    CLAIM_ID_PATTERN,
    FACET_ID_PATTERN,
    SHA256_PATTERN,
    WITNESS_ID_PATTERN,
    AnsweringContractModel,
    require_unique,
    stable_content_hash,
    stable_identifier,
)
from app.domain.answering.question import QuestionContract, RequiredOutputForm
from app.domain.answering.witness import WitnessCatalogReceipt


CLAIM_PLAN_SCHEMA_VERSION = "claim-plan.v1"
ANSWER_PLAN_SCHEMA_VERSION = "answer-plan.v1"


class AnswerDecision(StrEnum):
    """AnswerPlan 的业务决定。"""

    ANSWER = "answer"
    PARTIAL = "partial"
    REFUSE = "refuse"
    CLARIFY = "clarify"
    DIRECT_ANSWER = "direct_answer"


class ClaimKind(StrEnum):
    """服务端对渲染单元的分类。"""

    FACTUAL = "factual"
    DIRECT_VERDICT = "direct_verdict"
    UNCERTAINTY = "uncertainty"
    INSUFFICIENCY = "insufficiency"


class RenderingRule(StrEnum):
    """允许渲染器执行的封闭规则集。"""

    EXACT_WITNESS = "exact_witness"
    DETERMINISTIC_TEMPLATE = "deterministic_template"
    SYSTEM_NOTICE = "system_notice"


def _claim_payload(
    *,
    claim_id: str,
    obligation_id: str | None,
    witness_ids: tuple[str, ...],
    claim_kind: ClaimKind,
    rendering_rule: RenderingRule,
    template_id: str | None,
    ordinal: int,
) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_PLAN_SCHEMA_VERSION,
        "claim_id": claim_id,
        "obligation_id": obligation_id,
        "witness_ids": list(witness_ids),
        "claim_kind": claim_kind.value,
        "rendering_rule": rendering_rule.value,
        "template_id": template_id,
        "ordinal": ordinal,
    }


class ClaimPlan(AnsweringContractModel):
    """不含模型事实正文的单个渲染计划。"""

    schema_version: Literal["claim-plan.v1"] = CLAIM_PLAN_SCHEMA_VERSION
    claim_id: str = Field(pattern=CLAIM_ID_PATTERN)
    claim_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str | None = Field(default=None, pattern=FACET_ID_PATTERN)
    witness_ids: tuple[str, ...] = Field(default=(), max_length=16)
    claim_kind: ClaimKind
    rendering_rule: RenderingRule
    template_id: str | None = Field(default=None, min_length=1, max_length=128)
    ordinal: int = Field(ge=0, le=63)

    @field_validator("witness_ids")
    @classmethod
    def validate_witness_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(value, field_name="witness_ids")
        if any(re.fullmatch(WITNESS_ID_PATTERN, item) is None for item in value):
            raise ValueError("claim witness_ids contain an invalid identity")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _claim_payload(
            claim_id=self.claim_id,
            obligation_id=self.obligation_id,
            witness_ids=self.witness_ids,
            claim_kind=self.claim_kind,
            rendering_rule=self.rendering_rule,
            template_id=self.template_id,
            ordinal=self.ordinal,
        )

    @model_validator(mode="after")
    def validate_rendering_contract(self) -> Self:
        if self.claim_id != f"clm_{self.ordinal + 1}":
            raise ValueError("claim_id must be the one-based identity of ordinal")
        if self.rendering_rule == RenderingRule.EXACT_WITNESS:
            if (
                self.claim_kind != ClaimKind.FACTUAL
                or len(self.witness_ids) != 1
                or self.template_id is not None
                or self.obligation_id is None
            ):
                raise ValueError("exact_witness requires one factual witness and obligation")
        elif self.rendering_rule == RenderingRule.DETERMINISTIC_TEMPLATE:
            if (
                self.claim_kind not in {ClaimKind.FACTUAL, ClaimKind.DIRECT_VERDICT}
                or not self.witness_ids
                or self.template_id is None
                or self.obligation_id is None
            ):
                raise ValueError(
                    "deterministic_template requires factual witnesses, template and obligation"
                )
        else:
            if (
                self.claim_kind not in {ClaimKind.UNCERTAINTY, ClaimKind.INSUFFICIENCY}
                or self.witness_ids
                or self.template_id is None
            ):
                raise ValueError("system_notice requires a notice kind, template and no witnesses")
        if self.claim_hash != stable_content_hash(self._identity_payload()):
            raise ValueError("claim_hash does not match ClaimPlan")
        return self

    @property
    def is_substantive(self) -> bool:
        """事实或直接裁决才算回答义务；系统提示不算覆盖。"""

        return self.claim_kind in {ClaimKind.FACTUAL, ClaimKind.DIRECT_VERDICT}

    @classmethod
    def build(
        cls,
        *,
        ordinal: int,
        obligation_id: str | None,
        witness_ids: tuple[str, ...],
        claim_kind: ClaimKind,
        rendering_rule: RenderingRule,
        template_id: str | None = None,
    ) -> "ClaimPlan":
        """由服务端分配唯一 claim_id 并生成稳定计划。"""

        claim_id = f"clm_{ordinal + 1}"
        payload = _claim_payload(
            claim_id=claim_id,
            obligation_id=obligation_id,
            witness_ids=witness_ids,
            claim_kind=claim_kind,
            rendering_rule=rendering_rule,
            template_id=template_id,
            ordinal=ordinal,
        )
        return cls(**payload, claim_hash=stable_content_hash(payload))


def _answer_plan_payload(
    *,
    question_contract_id: str,
    question_contract_hash: str,
    context_pack_hash: str | None,
    witness_catalog_id: str | None,
    witness_catalog_hash: str | None,
    decision: AnswerDecision,
    claims: tuple[ClaimPlan, ...],
    unresolved_obligation_ids: tuple[str, ...],
    proposal_schema_version: str | None,
    prompt_contract_version: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": ANSWER_PLAN_SCHEMA_VERSION,
        "question_contract_id": question_contract_id,
        "question_contract_hash": question_contract_hash,
        "context_pack_hash": context_pack_hash,
        "witness_catalog_id": witness_catalog_id,
        "witness_catalog_hash": witness_catalog_hash,
        "decision": decision.value,
        "claims": [item.to_dict() for item in claims],
        "unresolved_obligation_ids": list(unresolved_obligation_ids),
        "proposal_schema_version": proposal_schema_version,
        "prompt_contract_version": prompt_contract_version,
    }


class AnswerPlan(AnsweringContractModel):
    """Provider 选择经服务端投影后的权威回答计划。"""

    schema_version: Literal["answer-plan.v1"] = ANSWER_PLAN_SCHEMA_VERSION
    plan_id: str = Field(pattern=r"^apn_[0-9a-f]{40}$")
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    context_pack_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    witness_catalog_id: str | None = Field(
        default=None,
        pattern=r"^wcat_[0-9a-f]{40}$",
    )
    witness_catalog_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decision: AnswerDecision
    claims: tuple[ClaimPlan, ...] = Field(default=(), max_length=64)
    unresolved_obligation_ids: tuple[str, ...] = Field(default=(), max_length=32)
    proposal_schema_version: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_contract_version: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("unresolved_obligation_ids")
    @classmethod
    def validate_unresolved_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="unresolved_obligation_ids")

    def _identity_payload(self) -> dict[str, Any]:
        return _answer_plan_payload(
            question_contract_id=self.question_contract_id,
            question_contract_hash=self.question_contract_hash,
            context_pack_hash=self.context_pack_hash,
            witness_catalog_id=self.witness_catalog_id,
            witness_catalog_hash=self.witness_catalog_hash,
            decision=self.decision,
            claims=self.claims,
            unresolved_obligation_ids=self.unresolved_obligation_ids,
            proposal_schema_version=self.proposal_schema_version,
            prompt_contract_version=self.prompt_contract_version,
        )

    @model_validator(mode="after")
    def validate_local_graph(self) -> Self:
        claim_ids = tuple(item.claim_id for item in self.claims)
        require_unique(claim_ids, field_name="claim_ids")
        if tuple(item.ordinal for item in self.claims) != tuple(range(len(self.claims))):
            raise ValueError("claim ordinals must be contiguous and zero-based")

        reference_fields = (
            self.context_pack_hash,
            self.witness_catalog_id,
            self.witness_catalog_hash,
        )
        model_fields = (self.proposal_schema_version, self.prompt_contract_version)
        non_retrieval = self.decision in {
            AnswerDecision.CLARIFY,
            AnswerDecision.DIRECT_ANSWER,
        }
        substantive = tuple(item for item in self.claims if item.is_substantive)
        if non_retrieval:
            if any(value is not None for value in (*reference_fields, *model_fields)):
                raise ValueError("non-retrieval plan cannot carry retrieval/model references")
            if not self.claims or substantive or self.unresolved_obligation_ids:
                raise ValueError("non-retrieval plan requires only system notices")
        else:
            if any(value is None for value in (*reference_fields, *model_fields)):
                raise ValueError("retrieval plan requires catalog and model contract references")
            if self.decision == AnswerDecision.ANSWER:
                if not substantive or self.unresolved_obligation_ids:
                    raise ValueError("answer decision requires substantive complete claims")
            elif self.decision == AnswerDecision.PARTIAL:
                if not substantive or not self.unresolved_obligation_ids:
                    raise ValueError("partial decision requires claims and unresolved obligations")
            elif substantive or not self.unresolved_obligation_ids or not self.claims:
                raise ValueError(
                    "refuse decision requires unresolved obligations and a system notice"
                )

        if self.plan_hash != stable_content_hash(self._identity_payload()):
            raise ValueError("plan_hash does not match AnswerPlan")
        if self.plan_id != stable_identifier("apn_", self._identity_payload()):
            raise ValueError("plan_id does not match AnswerPlan")
        return self

    @property
    def substantive_claims(self) -> tuple[ClaimPlan, ...]:
        """返回真正回答义务的事实或裁决计划。"""

        return tuple(item for item in self.claims if item.is_substantive)

    @classmethod
    def build(
        cls,
        *,
        question_contract: QuestionContract,
        catalog: WitnessCatalogReceipt | None,
        decision: AnswerDecision,
        claims: tuple[ClaimPlan, ...],
        unresolved_obligation_ids: tuple[str, ...] = (),
        proposal_schema_version: str | None = None,
        prompt_contract_version: str | None = None,
    ) -> "AnswerPlan":
        """构建计划并立即执行 QuestionContract/Catalog 交叉校验。"""

        payload = _answer_plan_payload(
            question_contract_id=question_contract.contract_id,
            question_contract_hash=question_contract.contract_hash,
            context_pack_hash=catalog.context_pack_hash if catalog else None,
            witness_catalog_id=catalog.catalog_id if catalog else None,
            witness_catalog_hash=catalog.catalog_hash if catalog else None,
            decision=decision,
            claims=claims,
            unresolved_obligation_ids=unresolved_obligation_ids,
            proposal_schema_version=proposal_schema_version,
            prompt_contract_version=prompt_contract_version,
        )
        plan = cls(
            **payload,
            plan_id=stable_identifier("apn_", payload),
            plan_hash=stable_content_hash(payload),
        )
        return plan.validate_against(question_contract=question_contract, catalog=catalog)

    def validate_against(
        self,
        *,
        question_contract: QuestionContract,
        catalog: WitnessCatalogReceipt | None,
    ) -> Self:
        """校验计划只引用冻结问题与目录中的唯一身份。"""

        if (
            self.question_contract_id != question_contract.contract_id
            or self.question_contract_hash != question_contract.contract_hash
        ):
            raise ValueError("AnswerPlan references a different QuestionContract")
        known_obligations = {item.obligation_id for item in question_contract.obligations}
        required = set(question_contract.required_obligation_ids)

        if self.decision in {AnswerDecision.CLARIFY, AnswerDecision.DIRECT_ANSWER}:
            if question_contract.obligations or catalog is not None:
                raise ValueError("non-retrieval plan requires an empty question contract")
            if any(item.obligation_id is not None for item in self.claims):
                raise ValueError("non-retrieval system notice cannot invent an obligation")
            return self

        if catalog is None:
            raise ValueError("retrieval plan requires WitnessCatalogReceipt")
        if (
            catalog.question_contract_id != question_contract.contract_id
            or catalog.question_contract_hash != question_contract.contract_hash
            or self.context_pack_hash != catalog.context_pack_hash
            or self.witness_catalog_id != catalog.catalog_id
            or self.witness_catalog_hash != catalog.catalog_hash
        ):
            raise ValueError("AnswerPlan catalog references are inconsistent")

        known_witnesses = set(catalog.witness_ids)
        obligations_by_id = {
            item.obligation_id: item for item in question_contract.obligations
        }
        for claim in self.claims:
            if claim.obligation_id is None:
                if claim.rendering_rule != RenderingRule.SYSTEM_NOTICE:
                    raise ValueError("only system_notice may omit obligation_id")
            elif claim.obligation_id not in known_obligations:
                raise ValueError("AnswerPlan references an unknown obligation")
            if not set(claim.witness_ids).issubset(known_witnesses):
                raise ValueError("AnswerPlan references an unknown witness")
            if claim.is_substantive and claim.obligation_id is not None:
                obligation = obligations_by_id[claim.obligation_id]
                expected_template = {
                    RequiredOutputForm.DIRECT_VERDICT: "template.verbatim-verdict.v1",
                    RequiredOutputForm.LIST: "template.verbatim-list.v1",
                    RequiredOutputForm.ORDERED_STEPS: "template.verbatim-steps.v1",
                    RequiredOutputForm.COMPARISON: "template.verbatim-comparison.v1",
                }.get(obligation.required_output_form)
                if expected_template is None:
                    if claim.rendering_rule not in {
                        RenderingRule.EXACT_WITNESS,
                        RenderingRule.DETERMINISTIC_TEMPLATE,
                    }:
                        raise ValueError("fact obligation has an invalid rendering rule")
                elif (
                    claim.rendering_rule != RenderingRule.DETERMINISTIC_TEMPLATE
                    or claim.template_id != expected_template
                    or (
                        obligation.required_output_form == RequiredOutputForm.DIRECT_VERDICT
                        and claim.claim_kind != ClaimKind.DIRECT_VERDICT
                    )
                    or (
                        obligation.required_output_form != RequiredOutputForm.DIRECT_VERDICT
                        and claim.claim_kind != ClaimKind.FACTUAL
                    )
                ):
                    raise ValueError("ClaimPlan does not satisfy required_output_form")

        witness_rank = {item: index for index, item in enumerate(catalog.witness_ids)}
        for obligation in question_contract.obligations:
            if obligation.required_output_form != RequiredOutputForm.ORDERED_STEPS:
                continue
            ordered_claims = tuple(
                item
                for item in self.substantive_claims
                if item.obligation_id == obligation.obligation_id
            )
            positions = tuple(witness_rank[item.witness_ids[0]] for item in ordered_claims)
            if positions != tuple(sorted(positions)) or len(positions) != len(set(positions)):
                raise ValueError("ordered-step claims must follow frozen Witness order")

        substantive_obligations = {
            item.obligation_id for item in self.substantive_claims if item.obligation_id is not None
        }
        unresolved = set(self.unresolved_obligation_ids)
        if not unresolved.issubset(required):
            raise ValueError("only required obligations may be unresolved")
        overlap = substantive_obligations & unresolved
        if overlap and self.decision != AnswerDecision.PARTIAL:
            raise ValueError("only a partial plan may keep a supported obligation unresolved")

        covered_required = substantive_obligations & required
        if self.decision == AnswerDecision.ANSWER:
            if covered_required != required:
                raise ValueError("answer decision must cover every required obligation")
        elif self.decision == AnswerDecision.PARTIAL:
            if not covered_required or not unresolved or covered_required | unresolved != required:
                raise ValueError(
                    "partial decision must cover or explicitly leave unresolved every "
                    "required obligation"
                )
        else:
            if not required or unresolved != required or substantive_obligations:
                raise ValueError("refuse decision must leave every required obligation unresolved")
        return self


__all__ = [
    "ANSWER_PLAN_SCHEMA_VERSION",
    "CLAIM_PLAN_SCHEMA_VERSION",
    "AnswerDecision",
    "AnswerPlan",
    "ClaimKind",
    "ClaimPlan",
    "RenderingRule",
]
