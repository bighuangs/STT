"""配置化 AnsweringModelPort 实现。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel

from app.application.query_answering.proposals import (
    AnswerDecision as ProposalDecision,
    AnswerPlanProposal,
    AnswerPlanWireProposal,
    RenderingRule as ProposalRenderingRule,
    ReviewWireProposal,
    ReviewProposal,
    parse_answer_plan_wire_proposal_json,
    parse_review_wire_proposal_json,
    project_answer_plan_wire_proposal,
    project_review_wire_proposal,
    validate_claim_allowlists,
    validate_review_wire_allowlists,
)
from app.domain.answering.plan import AnswerPlan, RenderingRule
from app.domain.answering.enumeration_closure import (
    QueryAnchoredInlineActionChainClosureReceipt,
    QueryAnchoredProcedureListClosureReceipt,
    QueryAnchoredScopedBulletListClosureReceipt,
    QueryAnchoredScopedProcedureListClosureReceipt,
    QueryAnchoredScopedTableSelectionClosureReceipt,
    QueryAnchoredTableRowClosureReceipt,
    SourceEnumerationClosureReceipt,
    parse_source_enumeration_closure_receipt,
    source_closure_witness_ids,
)
from app.domain.answering.fact_set_closure import QueryAnchoredScopedFactSetClosureReceipt
from app.domain.answering.question import QuestionContract
from app.domain.answering.witness import WitnessCatalogReceipt, WitnessUniverseStatus
from app.domain.query_planning import FacetClosure
from app.infrastructure.answering.model_gateway import (
    AnsweringPromptBudgetError,
    invoke_structured_answer_model,
)
from app.infrastructure.answering.prompts import (
    ANSWERING_OUTPUT_TOKEN_CAP,
    PLAN_PROMPT_VERSION,
    REVIEW_PROMPT_VERSION,
    build_plan_prompt_payload,
    build_review_prompt_payload,
)


def _output_cap(state: Mapping[str, Any]) -> int:
    context = state.get("context_pack")
    budget = context.get("budget") if isinstance(context, Mapping) else None
    raw = budget.get("reserved_output") if isinstance(budget, Mapping) else None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < ANSWERING_OUTPUT_TOKEN_CAP:
        raise AnsweringPromptBudgetError(
            "ContextPack did not reserve the configured answering output cap"
        )
    return ANSWERING_OUTPUT_TOKEN_CAP


def _witness_catalog_receipt(state: Mapping[str, Any]) -> Mapping[str, Any]:
    bundle = state.get("witness_catalog")
    receipt = bundle.get("receipt") if isinstance(bundle, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise AnsweringPromptBudgetError(
            "answering requires a frozen WitnessCatalog closure receipt"
        )
    return receipt


def _required_closed_obligation_ids(
    *,
    question_contract: QuestionContract,
    answer_plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    source_closed_witness_ids: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """根据权威 plan 与 catalog 确定性派生 canonical required closure。"""

    if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
        return ()
    source_closed_witness_ids = source_closed_witness_ids or {}
    claims_by_obligation: dict[str, list[object]] = {}
    for claim in answer_plan.substantive_claims:
        if claim.obligation_id is not None:
            claims_by_obligation.setdefault(claim.obligation_id, []).append(claim)
    unresolved = set(answer_plan.unresolved_obligation_ids)
    required = set(question_contract.required_obligation_ids)
    closed: list[str] = []
    for obligation in question_contract.obligations:
        claims = claims_by_obligation.get(obligation.obligation_id, [])
        if obligation.obligation_id not in required or obligation.obligation_id in unresolved:
            continue
        if obligation.closure == FacetClosure.SINGLE and len(claims) != 1:
            continue
        if (
            obligation.closure == FacetClosure.CLOSED_COUNT
            and len(claims) != obligation.expected_count
        ):
            continue
        if obligation.closure == FacetClosure.OPEN_ENUMERATION:
            witness_ids = tuple(
                witness_id for claim in claims for witness_id in getattr(claim, "witness_ids", ())
            )
            if witness_ids != source_closed_witness_ids.get(obligation.obligation_id):
                continue
        closed.append(obligation.obligation_id)
    return tuple(closed)


def _validated_source_closed_witness_ids(
    witness_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """重放 prompt flags 与 owner receipt 的完整多 Witness 绑定。"""

    receipts_by_obligation: dict[str, SourceEnumerationClosureReceipt] = {}
    receipt_ids: set[str] = set()
    declared_pairs: set[tuple[str, str]] = set()
    for witness in witness_catalog:
        witness_id = str(witness.get("witness_id") or "")
        declared_ids = tuple(witness.get("source_closed_open_obligation_ids") or ())
        if len(declared_ids) != len(set(declared_ids)):
            raise ValueError("source closure prompt flags must be unique per witness")
        declared_pairs.update((str(obligation_id), witness_id) for obligation_id in declared_ids)
        receipts = tuple(witness.get("source_enumeration_closure_receipts") or ())
        for raw_receipt in receipts:
            receipt = (
                raw_receipt
                if isinstance(
                    raw_receipt,
                    (
                        QueryAnchoredTableRowClosureReceipt,
                        QueryAnchoredInlineActionChainClosureReceipt,
                        QueryAnchoredProcedureListClosureReceipt,
                        QueryAnchoredScopedBulletListClosureReceipt,
                        QueryAnchoredScopedProcedureListClosureReceipt,
                        QueryAnchoredScopedTableSelectionClosureReceipt,
                        QueryAnchoredScopedFactSetClosureReceipt,
                    ),
                )
                else parse_source_enumeration_closure_receipt(raw_receipt)
            )
            member_ids = source_closure_witness_ids(receipt)
            if member_ids[0] != witness_id:
                raise ValueError("source closure receipt belongs to another witness")
            if receipt.receipt_id in receipt_ids:
                raise ValueError("source closure receipt is duplicated across witness views")
            if receipt.obligation_id in receipts_by_obligation:
                raise ValueError("each open obligation may carry at most one source receipt")
            receipt_ids.add(receipt.receipt_id)
            receipts_by_obligation[receipt.obligation_id] = receipt
    expected_pairs = {
        (obligation_id, witness_id)
        for obligation_id, receipt in receipts_by_obligation.items()
        for witness_id in source_closure_witness_ids(receipt)
    }
    if declared_pairs != expected_pairs:
        raise ValueError("source closure prompt flags differ from validated receipts")
    return {
        obligation_id: source_closure_witness_ids(receipt)
        for obligation_id, receipt in receipts_by_obligation.items()
    }


def _validate_review_plan_binding(
    *,
    plan_proposal: AnswerPlanProposal,
    answer_plan: AnswerPlan,
) -> None:
    """确认 prompt proposal 与 state 中权威 plan 的可审核选择完全一致。"""

    expected_decision = ProposalDecision(answer_plan.decision.value)
    if plan_proposal.decision != expected_decision:
        raise ValueError("semantic reviewer proposal decision differs from AnswerPlan")
    expected_claims = tuple(
        (
            str(item.obligation_id),
            item.witness_ids[0],
            (
                ProposalRenderingRule.EXACT_WITNESS
                if item.rendering_rule == RenderingRule.EXACT_WITNESS
                else ProposalRenderingRule.DETERMINISTIC_TEMPLATE
            ),
        )
        for item in answer_plan.substantive_claims
    )
    proposed_claims = tuple(
        (item.obligation_id, item.witness_id, item.rendering_rule) for item in plan_proposal.claims
    )
    if sorted(proposed_claims) != sorted(expected_claims):
        raise ValueError("semantic reviewer proposal claims differ from AnswerPlan")


def _review_accept_contract(
    state: Mapping[str, Any],
    *,
    question_contract: QuestionContract,
    plan_proposal: AnswerPlanProposal,
    witness_catalog: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """冻结当前 review 调用的权威 plan/catalog 与精确 accept closure。"""

    receipt_payload = _witness_catalog_receipt(state)
    catalog = WitnessCatalogReceipt.from_dict(receipt_payload)
    answer_plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
    answer_plan.validate_against(
        question_contract=question_contract,
        catalog=catalog,
    )
    _validate_review_plan_binding(
        plan_proposal=plan_proposal,
        answer_plan=answer_plan,
    )
    required_closed_obligation_ids = _required_closed_obligation_ids(
        question_contract=question_contract,
        answer_plan=answer_plan,
        catalog=catalog,
        source_closed_witness_ids=_validated_source_closed_witness_ids(witness_catalog),
    )
    return receipt_payload, required_closed_obligation_ids


class ConfiguredAnsweringModelAdapter:
    """使用当前配置模型生成并严格解析 ID-only proposal。"""

    plan_prompt_version = PLAN_PROMPT_VERSION
    review_prompt_version = REVIEW_PROMPT_VERSION

    def select_plan(
        self,
        state: dict[str, Any],
        *,
        question_contract: QuestionContract,
        witness_catalog: Sequence[Mapping[str, Any]],
    ) -> AnswerPlanProposal:
        payload = build_plan_prompt_payload(
            question_contract,
            witness_catalog,
            witness_catalog_receipt=_witness_catalog_receipt(state),
        )
        allowed_obligation_ids = tuple(item.obligation_id for item in question_contract.obligations)
        allowed_witness_ids = tuple(str(item["witness_id"]) for item in witness_catalog)

        def parse_wire(raw: str) -> AnswerPlanWireProposal:
            wire = parse_answer_plan_wire_proposal_json(raw)
            validate_claim_allowlists(
                wire.claims,
                allowed_obligation_ids=allowed_obligation_ids,
                allowed_witness_ids=allowed_witness_ids,
            )
            return wire

        def project_wire(wire: BaseModel) -> AnswerPlanProposal:
            if not isinstance(wire, AnswerPlanWireProposal):
                raise ValueError("plan wire parser returned an unexpected server type")
            return project_answer_plan_wire_proposal(wire)

        return invoke_structured_answer_model(
            state,
            scene="answer_plan_selection",
            prompt=payload["messages"],
            response_model=AnswerPlanProposal,
            response_parser=parse_wire,
            prompt_version=PLAN_PROMPT_VERSION,
            output_token_cap=_output_cap(state),
            wire_response_model=AnswerPlanWireProposal,
            response_projector=project_wire,
        )

    def review_plan(
        self,
        state: dict[str, Any],
        *,
        question_contract: QuestionContract,
        witness_catalog: Sequence[Mapping[str, Any]],
        plan_proposal: AnswerPlanProposal,
        repair_attempt: int,
    ) -> ReviewProposal:
        if repair_attempt not in {0, 1, 2}:
            raise ValueError("repair_attempt must be 0, 1, or 2")
        bounded_attempt = cast(Literal[0, 1, 2], repair_attempt)
        receipt_payload, required_closed_obligation_ids = _review_accept_contract(
            state,
            question_contract=question_contract,
            plan_proposal=plan_proposal,
            witness_catalog=witness_catalog,
        )
        payload = build_review_prompt_payload(
            question_contract,
            witness_catalog,
            plan_proposal,
            witness_catalog_receipt=receipt_payload,
            repair_attempt=bounded_attempt,
        )
        allowed_obligation_ids = tuple(item.obligation_id for item in question_contract.obligations)
        allowed_witness_ids = tuple(str(item["witness_id"]) for item in witness_catalog)

        def parse_wire(raw: str) -> ReviewWireProposal:
            wire = parse_review_wire_proposal_json(raw)
            validate_review_wire_allowlists(
                wire,
                allowed_obligation_ids=allowed_obligation_ids,
                allowed_witness_ids=allowed_witness_ids,
            )
            return wire

        def project_wire(wire: BaseModel) -> ReviewProposal:
            if not isinstance(wire, ReviewWireProposal):
                raise ValueError("review wire parser returned an unexpected server type")
            return project_review_wire_proposal(
                wire,
                repair_attempt=bounded_attempt,
                required_closed_obligation_ids=required_closed_obligation_ids,
            )

        return invoke_structured_answer_model(
            state,
            scene="answer_plan_semantic_review",
            prompt=payload["messages"],
            response_model=ReviewProposal,
            response_parser=parse_wire,
            prompt_version=REVIEW_PROMPT_VERSION,
            output_token_cap=_output_cap(state),
            wire_response_model=ReviewWireProposal,
            response_projector=project_wire,
        )


configured_answering_model_adapter = ConfiguredAnsweringModelAdapter()


__all__ = [
    "ConfiguredAnsweringModelAdapter",
    "configured_answering_model_adapter",
]
