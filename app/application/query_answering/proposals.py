"""Query answering provider 的最小、严格 Application proposal 契约。

这些类型只承载模型允许提出的选择。事实正文、引用标签、内容哈希、claim ID、
Evidence/service ID 和最终 AnswerPlan 均由确定性外壳持有或生成，不能进入模型输出。
"""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION = "answer-plan-proposal.v3"
REVIEW_PROPOSAL_SCHEMA_VERSION = "answer-plan-review-proposal.v3"
MAX_REPAIR_ATTEMPTS = 2
MAX_PROPOSAL_CLAIMS = 16
# 16 项 wire proposal 的最坏规范 JSON 为 2,561 UTF-8 字节。provider 输出 token
# 与本地 Parser 共用该最小充分上限；字节计数也为 tokenizer 提供保守上界。
MAX_PROPOSAL_BYTES = 2_561
MAX_WITNESS_CATALOG_ITEMS = 256

_OBLIGATION_ID_PATTERN = r"^qf_[0-9a-f]{32}$"
_WITNESS_ID_PATTERN = r"^wit_[0-9a-f]{40}$"


class ProviderWireFailureReason(StrEnum):
    """Provider wire 失败的低基数、安全分类。"""

    WIRE_CONTRACT_INVALID = "WIRE_CONTRACT_INVALID"
    INVALID_JSON = "INVALID_JSON"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    NON_FINITE_JSON_NUMBER = "NON_FINITE_JSON_NUMBER"
    INVALID_UNICODE = "INVALID_UNICODE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    ROOT_NOT_OBJECT = "ROOT_NOT_OBJECT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    ALLOWLIST_VIOLATION = "ALLOWLIST_VIOLATION"
    REPAIR_STATE_INVALID = "REPAIR_STATE_INVALID"


class ProviderWireContractError(ValueError):
    """仅表示 provider wire payload 可通过重试纠正的契约错误。"""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ProviderWireFailureReason = (ProviderWireFailureReason.WIRE_CONTRACT_INVALID),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ProposalValidationError(ProviderWireContractError):
    """模型 proposal 无法安全进入确定性回答流水线。"""


class AnswerDecision(StrEnum):
    """模型可选择的回答动作。"""

    ANSWER = "answer"
    PARTIAL = "partial"
    REFUSE = "refuse"


class RenderingRule(StrEnum):
    """模型可选择、由确定性 renderer 实现的正文规则。"""

    EXACT_WITNESS = "exact_witness"
    DETERMINISTIC_TEMPLATE = "deterministic_template"


class PlanReasonCode(StrEnum):
    """回答动作的低基数、无正文原因。"""

    ANSWER_SUPPORTED = "ANSWER_SUPPORTED"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    REQUIRED_OBLIGATION_UNSUPPORTED = "REQUIRED_OBLIGATION_UNSUPPORTED"
    NO_SUPPORTING_WITNESS = "NO_SUPPORTING_WITNESS"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    WITNESS_CONFLICT = "WITNESS_CONFLICT"


class ReviewDecision(StrEnum):
    """语义 reviewer 对当前 plan proposal 的处理动作。"""

    ACCEPT = "accept"
    REPAIR = "repair"
    REJECT = "reject"


class ReviewReasonCode(StrEnum):
    """语义审核的低基数、无正文原因。"""

    PLAN_SEMANTICALLY_VALID = "PLAN_SEMANTICALLY_VALID"
    OBLIGATION_WITNESS_MISMATCH = "OBLIGATION_WITNESS_MISMATCH"
    WITNESS_NOT_SUPPORTIVE = "WITNESS_NOT_SUPPORTIVE"
    RENDERING_RULE_MISMATCH = "RENDERING_RULE_MISMATCH"
    REQUIRED_OBLIGATION_MISSING = "REQUIRED_OBLIGATION_MISSING"
    REPLACEMENT_SELECTION_PROPOSED = "REPLACEMENT_SELECTION_PROPOSED"
    NO_SEMANTICALLY_VALID_PLAN = "NO_SEMANTICALLY_VALID_PLAN"
    REPAIR_LIMIT_REACHED = "REPAIR_LIMIT_REACHED"


class _StrictProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClaimProposal(_StrictProposal):
    """一个 obligation/witness/rendering_rule 的原子选择。"""

    obligation_id: str = Field(pattern=_OBLIGATION_ID_PATTERN)
    witness_id: str = Field(pattern=_WITNESS_ID_PATTERN)
    rendering_rule: RenderingRule


def _validate_unique_claims(claims: tuple[ClaimProposal, ...]) -> None:
    """校验 proposal 中的原子选择没有重复或一对多渲染歧义。"""

    if len(claims) != len(set(claims)):
        raise ValueError("claim proposal selections must be unique")
    claim_pairs = tuple((item.obligation_id, item.witness_id) for item in claims)
    if len(claim_pairs) != len(set(claim_pairs)):
        raise ValueError("one obligation/witness pair may select only one rendering rule")


class AnswerPlanWireProposal(_StrictProposal):
    """provider 只可输出决定与 opaque ID 选择。"""

    decision: AnswerDecision
    claims: tuple[ClaimProposal, ...] = Field(max_length=MAX_PROPOSAL_CLAIMS)

    @model_validator(mode="after")
    def validate_decision(self) -> "AnswerPlanWireProposal":
        _validate_unique_claims(self.claims)
        if self.decision == AnswerDecision.REFUSE:
            if self.claims:
                raise ValueError("refuse requires no claims")
            return self
        if not self.claims:
            raise ValueError("answer and partial require at least one claim")
        return self


class ReviewWireProposal(_StrictProposal):
    """reviewer 的最小 wire 输出；decision 分支在 wire 层即严格互斥。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "decision": {"const": "accept"},
                        "claims": {"maxItems": 0},
                    }
                },
                {
                    "properties": {
                        "decision": {"const": "repair"},
                        "claims": {"minItems": 1},
                    }
                },
                {
                    "properties": {
                        "decision": {"const": "reject"},
                        "claims": {"maxItems": 0},
                    }
                },
            ]
        },
    )

    decision: ReviewDecision
    claims: tuple[ClaimProposal, ...] = Field(max_length=MAX_PROPOSAL_CLAIMS)

    @model_validator(mode="after")
    def validate_structure(self) -> "ReviewWireProposal":
        _validate_unique_claims(self.claims)
        if self.decision == ReviewDecision.ACCEPT:
            if self.claims:
                raise ValueError("accept cannot carry replacement claims")
            return self
        if self.decision == ReviewDecision.REPAIR:
            if not self.claims:
                raise ValueError("repair requires replacement claims")
            return self
        if self.claims:
            raise ValueError("reject cannot carry claims")
        return self


class AnswerPlanProposal(_StrictProposal):
    """服务端规范化的回答计划建议，不是领域 AnswerPlan。"""

    decision: AnswerDecision
    claims: tuple[ClaimProposal, ...] = Field(max_length=MAX_PROPOSAL_CLAIMS)
    reason_codes: tuple[PlanReasonCode, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_decision(self) -> "AnswerPlanProposal":
        _validate_unique_claims(self.claims)
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("plan reason codes must be unique")

        reasons = set(self.reason_codes)
        if self.decision == AnswerDecision.ANSWER:
            if not self.claims or reasons != {PlanReasonCode.ANSWER_SUPPORTED}:
                raise ValueError("answer requires claims and only ANSWER_SUPPORTED")
            return self

        if self.decision == AnswerDecision.PARTIAL:
            allowed = {
                PlanReasonCode.PARTIAL_SUPPORT,
                PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
                PlanReasonCode.WITNESS_CONFLICT,
            }
            if (
                not self.claims
                or PlanReasonCode.PARTIAL_SUPPORT not in reasons
                or not reasons.issubset(allowed)
            ):
                raise ValueError("partial requires claims and a bounded partial-support reason set")
            return self

        allowed = {
            PlanReasonCode.NO_SUPPORTING_WITNESS,
            PlanReasonCode.EVIDENCE_INSUFFICIENT,
            PlanReasonCode.WITNESS_CONFLICT,
        }
        if self.claims or not reasons.issubset(allowed):
            raise ValueError("refuse requires no claims and only refusal reason codes")
        return self


class ReviewProposal(_StrictProposal):
    """服务端规范化审核结果；repair claims 是完整的安全候选替换集。"""

    decision: ReviewDecision
    claims: tuple[ClaimProposal, ...] = Field(max_length=MAX_PROPOSAL_CLAIMS)
    reason_codes: tuple[ReviewReasonCode, ...] = Field(min_length=1, max_length=8)
    closed_obligation_ids: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_decision(self) -> "ReviewProposal":
        _validate_unique_claims(self.claims)
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("review reason codes must be unique")
        if len(self.closed_obligation_ids) != len(set(self.closed_obligation_ids)):
            raise ValueError("closed obligation IDs must be unique")
        if any(
            re.fullmatch(_OBLIGATION_ID_PATTERN, item) is None
            for item in self.closed_obligation_ids
        ):
            raise ValueError("closed obligation IDs contain an invalid identity")

        reasons = set(self.reason_codes)
        if self.decision == ReviewDecision.ACCEPT:
            if self.claims or reasons != {ReviewReasonCode.PLAN_SEMANTICALLY_VALID}:
                raise ValueError(
                    "accept requires no replacement claims and only PLAN_SEMANTICALLY_VALID"
                )
            return self

        repairable = {
            ReviewReasonCode.OBLIGATION_WITNESS_MISMATCH,
            ReviewReasonCode.WITNESS_NOT_SUPPORTIVE,
            ReviewReasonCode.RENDERING_RULE_MISMATCH,
            ReviewReasonCode.REQUIRED_OBLIGATION_MISSING,
            ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,
        }
        if self.decision == ReviewDecision.REPAIR:
            if not self.claims or self.closed_obligation_ids or not reasons.issubset(repairable):
                raise ValueError(
                    "repair requires replacement claims and only repairable reason codes"
                )
            return self

        terminal = {
            ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,
            ReviewReasonCode.REPAIR_LIMIT_REACHED,
        }
        if self.claims or self.closed_obligation_ids or not reasons.issubset(terminal):
            raise ValueError("reject requires no claims and only terminal review reason codes")
        return self


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProposalValidationError(
                "duplicate JSON key is forbidden",
                reason_code=ProviderWireFailureReason.DUPLICATE_JSON_KEY,
            )
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> Any:
    raise ProposalValidationError(
        "non-finite JSON number is forbidden",
        reason_code=ProviderWireFailureReason.NON_FINITE_JSON_NUMBER,
    )


def _decode_json_object(raw_json: str | bytes | bytearray) -> dict[str, Any]:
    try:
        encoded_size = len(raw_json.encode("utf-8")) if isinstance(raw_json, str) else len(raw_json)
    except UnicodeEncodeError as exc:
        raise ProposalValidationError(
            "proposal contains invalid Unicode",
            reason_code=ProviderWireFailureReason.INVALID_UNICODE,
        ) from exc
    if encoded_size > MAX_PROPOSAL_BYTES:
        raise ProposalValidationError(
            "proposal exceeds the bounded JSON size",
            reason_code=ProviderWireFailureReason.PAYLOAD_TOO_LARGE,
        )
    try:
        value = json.loads(
            raw_json,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except ProposalValidationError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(
            "proposal is not valid JSON",
            reason_code=ProviderWireFailureReason.INVALID_JSON,
        ) from exc
    if not isinstance(value, dict):
        raise ProposalValidationError(
            "proposal root must be a JSON object",
            reason_code=ProviderWireFailureReason.ROOT_NOT_OBJECT,
        )
    return value


def parse_answer_plan_proposal_json(
    raw_json: str | bytes | bytearray,
) -> AnswerPlanProposal:
    """解析服务端 canonical JSON；重复键、未知字段和枚举均 fail closed。"""

    try:
        return AnswerPlanProposal.model_validate(_decode_json_object(raw_json))
    except ProposalValidationError:
        raise
    except ValueError as exc:
        raise ProposalValidationError(
            "answer plan proposal failed strict validation",
            reason_code=ProviderWireFailureReason.SCHEMA_INVALID,
        ) from exc


def parse_answer_plan_wire_proposal_json(
    raw_json: str | bytes | bytearray,
) -> AnswerPlanWireProposal:
    """解析最小 plan wire JSON；不接受 provider 自报原因。"""

    try:
        return AnswerPlanWireProposal.model_validate(_decode_json_object(raw_json))
    except ProposalValidationError:
        raise
    except ValueError as exc:
        raise ProposalValidationError(
            "answer plan wire proposal failed strict validation",
            reason_code=ProviderWireFailureReason.SCHEMA_INVALID,
        ) from exc


def project_answer_plan_wire_proposal(
    proposal: AnswerPlanWireProposal,
) -> AnswerPlanProposal:
    """按 wire 决定确定性派生服务端低基数原因。"""

    reasons = {
        AnswerDecision.ANSWER: (PlanReasonCode.ANSWER_SUPPORTED,),
        AnswerDecision.PARTIAL: (
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
        AnswerDecision.REFUSE: (PlanReasonCode.NO_SUPPORTING_WITNESS,),
    }[proposal.decision]
    return AnswerPlanProposal(
        decision=proposal.decision,
        claims=proposal.claims,
        reason_codes=reasons,
    )


def parse_review_proposal_json(
    raw_json: str | bytes | bytearray,
    *,
    repair_attempt: Literal[0, 1, 2],
) -> ReviewProposal:
    """解析服务端 canonical review JSON，并确定性关闭 repair 边。"""

    if repair_attempt not in {0, 1, 2}:
        raise ProposalValidationError(
            "repair_attempt must be 0, 1, or 2",
            reason_code=ProviderWireFailureReason.REPAIR_STATE_INVALID,
        )
    try:
        proposal = ReviewProposal.model_validate(_decode_json_object(raw_json))
    except ProposalValidationError:
        raise
    except ValueError as exc:
        raise ProposalValidationError(
            "review proposal failed strict validation",
            reason_code=ProviderWireFailureReason.SCHEMA_INVALID,
        ) from exc
    if repair_attempt >= MAX_REPAIR_ATTEMPTS and proposal.decision == ReviewDecision.REPAIR:
        return ReviewProposal(
            decision=ReviewDecision.REJECT,
            claims=(),
            reason_codes=(ReviewReasonCode.REPAIR_LIMIT_REACHED,),
        )
    if (
        repair_attempt < MAX_REPAIR_ATTEMPTS
        and ReviewReasonCode.REPAIR_LIMIT_REACHED in proposal.reason_codes
    ):
        raise ProposalValidationError(
            "non-terminal review cannot report REPAIR_LIMIT_REACHED",
            reason_code=ProviderWireFailureReason.REPAIR_STATE_INVALID,
        )
    return proposal


def parse_review_wire_proposal_json(
    raw_json: str | bytes | bytearray,
) -> ReviewWireProposal:
    """解析最小 review wire JSON，并拒绝矛盾 decision 字段组合。"""

    try:
        return ReviewWireProposal.model_validate(_decode_json_object(raw_json))
    except ProposalValidationError:
        raise
    except ValueError as exc:
        raise ProposalValidationError(
            "review wire proposal failed strict validation",
            reason_code=ProviderWireFailureReason.SCHEMA_INVALID,
        ) from exc


def project_review_wire_proposal(
    proposal: ReviewWireProposal,
    *,
    repair_attempt: Literal[0, 1, 2],
    required_closed_obligation_ids: Collection[str],
) -> ReviewProposal:
    """把已验证的 review wire 确定性投影为服务端 canonical proposal。"""

    if repair_attempt not in {0, 1, 2}:
        raise ValueError("repair_attempt must be 0, 1, or 2")

    if proposal.decision == ReviewDecision.ACCEPT:
        return ReviewProposal(
            decision=ReviewDecision.ACCEPT,
            claims=(),
            reason_codes=(ReviewReasonCode.PLAN_SEMANTICALLY_VALID,),
            closed_obligation_ids=tuple(required_closed_obligation_ids),
        )

    if proposal.decision == ReviewDecision.REPAIR:
        if repair_attempt < MAX_REPAIR_ATTEMPTS:
            return ReviewProposal(
                decision=ReviewDecision.REPAIR,
                claims=proposal.claims,
                reason_codes=(ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,),
                closed_obligation_ids=(),
            )
        return ReviewProposal(
            decision=ReviewDecision.REJECT,
            claims=(),
            reason_codes=(ReviewReasonCode.REPAIR_LIMIT_REACHED,),
            closed_obligation_ids=(),
        )

    return ReviewProposal(
        decision=ReviewDecision.REJECT,
        claims=(),
        reason_codes=(ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,),
        closed_obligation_ids=(),
    )


def enforce_repair_budget(
    proposal: ReviewProposal,
    *,
    repair_attempt: Literal[0, 1, 2],
) -> None:
    """确保全流程最多只有两次显式 repair。"""

    if repair_attempt >= MAX_REPAIR_ATTEMPTS and proposal.decision == ReviewDecision.REPAIR:
        raise ProposalValidationError(
            "repair proposal exceeds bounded repair limit",
            reason_code=ProviderWireFailureReason.REPAIR_STATE_INVALID,
        )


def validate_claim_allowlists(
    claims: Collection[ClaimProposal],
    *,
    allowed_obligation_ids: Collection[str],
    allowed_witness_ids: Collection[str],
) -> None:
    """确定性校验 provider 只能选择当前输入中已提供的 opaque ID。"""

    obligations = frozenset(allowed_obligation_ids)
    witnesses = frozenset(allowed_witness_ids)
    unknown_obligations = sorted(
        {claim.obligation_id for claim in claims if claim.obligation_id not in obligations}
    )
    unknown_witnesses = sorted(
        {claim.witness_id for claim in claims if claim.witness_id not in witnesses}
    )
    if unknown_obligations or unknown_witnesses:
        raise ProposalValidationError(
            "proposal selected an identifier outside the input allowlists",
            reason_code=ProviderWireFailureReason.ALLOWLIST_VIOLATION,
        )


def validate_review_wire_allowlists(
    proposal: ReviewWireProposal,
    *,
    allowed_obligation_ids: Collection[str],
    allowed_witness_ids: Collection[str],
) -> None:
    """在安全投影丢弃字段前校验 wire 中出现过的全部 opaque ID。"""

    validate_claim_allowlists(
        proposal.claims,
        allowed_obligation_ids=allowed_obligation_ids,
        allowed_witness_ids=allowed_witness_ids,
    )


__all__ = [
    "ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION",
    "MAX_PROPOSAL_BYTES",
    "MAX_PROPOSAL_CLAIMS",
    "MAX_REPAIR_ATTEMPTS",
    "MAX_WITNESS_CATALOG_ITEMS",
    "REVIEW_PROPOSAL_SCHEMA_VERSION",
    "AnswerDecision",
    "AnswerPlanWireProposal",
    "AnswerPlanProposal",
    "ClaimProposal",
    "PlanReasonCode",
    "ProviderWireContractError",
    "ProviderWireFailureReason",
    "ProposalValidationError",
    "RenderingRule",
    "ReviewDecision",
    "ReviewProposal",
    "ReviewReasonCode",
    "ReviewWireProposal",
    "enforce_repair_budget",
    "parse_answer_plan_proposal_json",
    "parse_answer_plan_wire_proposal_json",
    "parse_review_proposal_json",
    "parse_review_wire_proposal_json",
    "project_answer_plan_wire_proposal",
    "project_review_wire_proposal",
    "validate_claim_allowlists",
    "validate_review_wire_allowlists",
]
