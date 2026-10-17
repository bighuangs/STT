"""Provider proposal 到权威 AnswerPlan 的确定性投影。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.application.query_answering.proposals import (
    ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION,
    AnswerDecision as ProposalDecision,
    AnswerPlanProposal,
    PlanReasonCode,
    RenderingRule as ProposalRenderingRule,
    ReviewDecision,
    ReviewProposal,
    validate_claim_allowlists,
)
from app.domain.answering.plan import (
    AnswerDecision,
    AnswerPlan,
    ClaimKind,
    ClaimPlan,
    RenderingRule,
)
from app.domain.answering.question import QuestionContract, RequiredOutputForm
from app.domain.answering.enumeration_closure import (
    SourceEnumerationClosureReceipt,
    source_closure_witness_ids,
)
from app.domain.answering.witness import WitnessCatalogReceipt, WitnessUniverseStatus
from app.domain.query_planning import FacetClosure, QueryPlan


_DIRECT_NOTICE_TEMPLATES = {
    "query.greeting": "notice.greeting.v1",
    "query.ui_help": "notice.ui-help.v1",
}


def _refusal_notice_claim(catalog: WitnessCatalogReceipt) -> ClaimPlan:
    template_id = (
        "notice.no-support.v1"
        if catalog.universe_status.value == "complete"
        else "notice.insufficient-context.v1"
    )
    return ClaimPlan.build(
        ordinal=0,
        obligation_id=None,
        witness_ids=(),
        claim_kind=ClaimKind.INSUFFICIENCY,
        rendering_rule=RenderingRule.SYSTEM_NOTICE,
        template_id=template_id,
    )


def non_retrieval_answer_plan(
    state: Mapping[str, Any],
    question_contract: QuestionContract,
) -> AnswerPlan | None:
    """为澄清与 UI/问候构造无模型 system_notice 计划。"""

    query_plan = QueryPlan.model_validate(state.get("query_plan") or {})
    if query_plan.disposition.value == "retrieve":
        return None
    if query_plan.disposition.value == "need_clarification":
        decision = AnswerDecision.CLARIFY
        template_id = "notice.clarification.v1"
    else:
        decision = AnswerDecision.DIRECT_ANSWER
        template_id = _DIRECT_NOTICE_TEMPLATES.get(
            str(query_plan.response_key or ""),
            "notice.direct-help.v1",
        )
    claim = ClaimPlan.build(
        ordinal=0,
        obligation_id=None,
        witness_ids=(),
        claim_kind=ClaimKind.UNCERTAINTY,
        rendering_rule=RenderingRule.SYSTEM_NOTICE,
        template_id=template_id,
    )
    return AnswerPlan.build(
        question_contract=question_contract,
        catalog=None,
        decision=decision,
        claims=(claim,),
    )


def _template_for_obligation(
    question_contract: QuestionContract,
    obligation_id: str,
) -> tuple[ClaimKind, str]:
    obligation = next(
        item for item in question_contract.obligations if item.obligation_id == obligation_id
    )
    if obligation.required_output_form == RequiredOutputForm.DIRECT_VERDICT:
        return ClaimKind.DIRECT_VERDICT, "template.verbatim-verdict.v1"
    return ClaimKind.FACTUAL, {
        RequiredOutputForm.LIST: "template.verbatim-list.v1",
        RequiredOutputForm.ORDERED_STEPS: "template.verbatim-steps.v1",
        RequiredOutputForm.COMPARISON: "template.verbatim-comparison.v1",
    }.get(obligation.required_output_form, "template.verbatim-fact.v1")


def project_answer_plan(
    *,
    proposal: AnswerPlanProposal,
    question_contract: QuestionContract,
    catalog: WitnessCatalogReceipt,
    prompt_contract_version: str,
) -> AnswerPlan:
    """校验 ID allowlist，并为所有内部 claim/hash 分配服务端身份。"""

    validate_claim_allowlists(
        proposal.claims,
        allowed_obligation_ids=[item.obligation_id for item in question_contract.obligations],
        allowed_witness_ids=catalog.witness_ids,
    )
    claims: list[ClaimPlan] = []
    seen: set[tuple[str, str]] = set()
    obligation_rank = {
        item.obligation_id: index for index, item in enumerate(question_contract.obligations)
    }
    obligation_by_id = {item.obligation_id: item for item in question_contract.obligations}
    witness_rank = {item: index for index, item in enumerate(catalog.witness_ids)}
    proposal_rank = {id(item): index for index, item in enumerate(proposal.claims)}
    selected_claims = sorted(
        proposal.claims,
        key=lambda item: (
            obligation_rank[item.obligation_id],
            (
                witness_rank[item.witness_id]
                if obligation_by_id[item.obligation_id].required_output_form
                == RequiredOutputForm.ORDERED_STEPS
                else proposal_rank[id(item)]
            ),
        ),
    )
    for selected in selected_claims:
        key = (selected.obligation_id, selected.witness_id)
        if key in seen:
            raise ValueError("proposal repeats an obligation/witness pair")
        seen.add(key)
        obligation = next(
            item
            for item in question_contract.obligations
            if item.obligation_id == selected.obligation_id
        )
        if (
            selected.rendering_rule == ProposalRenderingRule.EXACT_WITNESS
            and obligation.required_output_form == RequiredOutputForm.FACT
        ):
            kind = ClaimKind.FACTUAL
            rule = RenderingRule.EXACT_WITNESS
            template_id = None
        else:
            kind, template_id = _template_for_obligation(
                question_contract,
                selected.obligation_id,
            )
            rule = RenderingRule.DETERMINISTIC_TEMPLATE
        claims.append(
            ClaimPlan.build(
                ordinal=len(claims),
                obligation_id=selected.obligation_id,
                witness_ids=(selected.witness_id,),
                claim_kind=kind,
                rendering_rule=rule,
                template_id=template_id,
            )
        )

    required = set(question_contract.required_obligation_ids)
    covered = {
        claim.obligation_id
        for claim in claims
        if claim.obligation_id in required and claim.is_substantive
    }
    unresolved = tuple(
        obligation_id
        for obligation_id in question_contract.required_obligation_ids
        if obligation_id not in covered
    )
    if proposal.decision == ProposalDecision.REFUSE or not covered:
        decision = AnswerDecision.REFUSE
        claims = [_refusal_notice_claim(catalog)]
        unresolved = question_contract.required_obligation_ids
    elif proposal.decision == ProposalDecision.PARTIAL:
        # Provider 的 partial 是语义判断。即使单个 obligation 已选择 Witness，
        # 也不能把“义务内部分覆盖”静默升级为完整 answer。
        decision = AnswerDecision.PARTIAL
        unresolved = question_contract.required_obligation_ids
    elif unresolved:
        # Provider 声称完整回答，但服务端覆盖投影发现仍有义务缺失时降级。
        decision = AnswerDecision.PARTIAL
    else:
        decision = AnswerDecision.ANSWER

    return AnswerPlan.build(
        question_contract=question_contract,
        catalog=catalog,
        decision=decision,
        claims=tuple(claims),
        unresolved_obligation_ids=unresolved,
        proposal_schema_version=ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION,
        prompt_contract_version=prompt_contract_version,
    )


def project_review_repair_proposal(
    *,
    review: ReviewProposal,
    question_contract: QuestionContract,
    catalog: WitnessCatalogReceipt,
    source_closure_by_obligation: Mapping[str, SourceEnumerationClosureReceipt],
) -> AnswerPlanProposal:
    """把 reviewer 的完整 replacement 确定性转换为下一版 plan proposal。"""

    if review.decision != ReviewDecision.REPAIR:
        raise ValueError("review repair projection requires a repair decision")
    validate_claim_allowlists(
        review.claims,
        allowed_obligation_ids=[item.obligation_id for item in question_contract.obligations],
        allowed_witness_ids=catalog.witness_ids,
    )
    claims_by_obligation: dict[str, int] = {}
    for item in review.claims:
        claims_by_obligation[item.obligation_id] = (
            claims_by_obligation.get(item.obligation_id, 0) + 1
        )
    required_ids = set(question_contract.required_obligation_ids)
    required_obligations = tuple(
        item for item in question_contract.obligations if item.obligation_id in required_ids
    )
    selected_witness_ids_by_obligation = {
        obligation.obligation_id: tuple(
            item.witness_id
            for item in sorted(
                (
                    claim
                    for claim in review.claims
                    if claim.obligation_id == obligation.obligation_id
                ),
                key=lambda claim: catalog.witness_ids.index(claim.witness_id),
            )
        )
        for obligation in required_obligations
    }
    required_shape_closed = all(
        (
            obligation.closure == FacetClosure.SINGLE
            and claims_by_obligation.get(obligation.obligation_id) == 1
        )
        or (
            obligation.closure == FacetClosure.CLOSED_COUNT
            and claims_by_obligation.get(obligation.obligation_id) == obligation.expected_count
        )
        or (
            obligation.closure == FacetClosure.OPEN_ENUMERATION
            and (source_receipt := source_closure_by_obligation.get(obligation.obligation_id))
            is not None
            and selected_witness_ids_by_obligation[obligation.obligation_id]
            == source_closure_witness_ids(source_receipt)
        )
        for obligation in required_obligations
    )
    complete_candidate = (
        catalog.universe_status == WitnessUniverseStatus.COMPLETE
        and bool(required_obligations)
        and required_shape_closed
    )
    decision = ProposalDecision.ANSWER if complete_candidate else ProposalDecision.PARTIAL
    reasons = (
        (PlanReasonCode.ANSWER_SUPPORTED,)
        if decision == ProposalDecision.ANSWER
        else (
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        )
    )
    return AnswerPlanProposal(
        decision=decision,
        claims=review.claims,
        reason_codes=reasons,
    )


def refusal_answer_plan(
    *,
    question_contract: QuestionContract,
    catalog: WitnessCatalogReceipt,
    prompt_contract_version: str,
) -> AnswerPlan:
    """把语义复核的终止结论收敛为无事实 claim 的拒答计划。"""

    return AnswerPlan.build(
        question_contract=question_contract,
        catalog=catalog,
        decision=AnswerDecision.REFUSE,
        claims=(_refusal_notice_claim(catalog),),
        unresolved_obligation_ids=question_contract.required_obligation_ids,
        proposal_schema_version=ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION,
        prompt_contract_version=prompt_contract_version,
    )


__all__ = [
    "non_retrieval_answer_plan",
    "project_answer_plan",
    "project_review_repair_proposal",
    "refusal_answer_plan",
]
