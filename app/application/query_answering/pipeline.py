"""Proof-Carrying Answer v2 的应用层用例编排。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.application.query_answering.proposals import (
    AnswerDecision as ProposalDecision,
    AnswerPlanProposal,
    ClaimProposal,
    MAX_REPAIR_ATTEMPTS,
    PlanReasonCode,
    RenderingRule as ProposalRenderingRule,
    ReviewDecision,
    ReviewProposal,
    ReviewReasonCode,
    validate_claim_allowlists,
)
from app.application.query_answering.builders import (
    build_question_contract,
    build_witness_catalog_bundle,
    collect_source_enumeration_closure_receipts,
    resolve_witness_prompt_views,
)
from app.application.query_answering.atomizer import top_level_list_item_body_span
from app.application.query_answering.planning import (
    non_retrieval_answer_plan,
    project_answer_plan,
    project_review_repair_proposal,
    refusal_answer_plan,
)
from app.application.query_answering.ports import (
    AnswerCommitPort,
    AnswerRenderPort,
    AnsweringModelPort,
)
from app.application.query_answering.rendering import (
    render_answer_and_proof,
    replay_answer_derivation,
)
from app.domain.answering.plan import AnswerDecision, AnswerPlan
from app.domain.answering.enumeration_closure import (
    QueryAnchoredInlineActionChainClosureReceipt,
    SourceEnumerationClosureReceipt,
    source_closure_witness_ids,
)
from app.domain.answering.item_boundary import (
    ClosedCountTableCellProjectionStatus,
    markdown_table_data_row_body_span,
    resolve_closed_count_markdown_table_cell_projection,
)
from app.domain.answering.proof import (
    AnswerProof,
    ItemSegmentationMethod,
    ObligationClosureStatus,
    ObligationOutcome,
    ProofVerdict,
    ProvenItemRef,
    SemanticReviewClosureReceipt,
    SemanticReviewTerminalReceipt,
    semantic_review_receipt_from_dict,
)
from app.domain.answering.question import QuestionContract
from app.domain.answering.witness import (
    EvidenceWitness,
    WitnessCatalogReceipt,
    WitnessAtomKind,
)
from app.domain.answer import AnswerEnvelope
from app.domain.answering._base import (
    sha256_digest,
    stable_content_hash,
    stable_identifier,
)
from app.domain.evidence import CitationFact, citation_fact_bundle_hash
from app.domain.query_planning import FacetClosure


class AnswerProofNotCommittableError(RuntimeError):
    """最终 proof 不是可交付终态，必须在任何外部副作用前失败。"""

    code = "ANSWER_PROOF_NOT_COMMITTABLE"
    safe_message = "回答证明未达到可安全交付状态。"
    retryable = False


_SEMANTIC_REVIEW_VERIFIER_VERSION = "answer-plan-review.v1"


def _local_review_attestation(
    *,
    plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    review: ReviewProposal,
) -> dict[str, str]:
    proposal_payload = review.model_dump(mode="json")
    proposal_hash = stable_content_hash(proposal_payload)
    prompt_payload = {
        "method": "deterministic-empty-witness-review.v1",
        "answer_plan_id": plan.plan_id,
        "answer_plan_hash": plan.plan_hash,
        "catalog_id": catalog.catalog_id,
        "catalog_hash": catalog.catalog_hash,
    }
    call_payload = {**prompt_payload, "review_proposal_hash": proposal_hash}
    return {
        "model_id": "deterministic-empty-witness-review.v1",
        "prompt_version": "not-applicable.empty-witness.v1",
        "logical_call_id": stable_identifier("lrv_", call_payload),
        "prompt_hash": stable_content_hash(prompt_payload),
        "response_hash": proposal_hash,
        "proposal_hash": proposal_hash,
    }


def _provider_review_attestation(
    state: Mapping[str, Any],
    *,
    review: ReviewProposal,
    prompt_version: str,
) -> dict[str, str]:
    successful = next(
        (
            item
            for item in reversed(state.get("answer_prompt_receipts") or [])
            if isinstance(item, Mapping)
            and item.get("scene") == "answer_plan_semantic_review"
            and item.get("status") == "succeeded"
        ),
        None,
    )
    if successful is None:
        raise ValueError("semantic review lacks a successful provider call receipt")
    parsed = successful.get("parsed_proposal")
    if not isinstance(parsed, Mapping):
        raise ValueError("semantic review call receipt lacks its parsed proposal")
    parsed_review = ReviewProposal.model_validate(parsed)
    expected_payload = review.model_dump(mode="json")
    if parsed_review.model_dump(mode="json") != expected_payload:
        raise ValueError("semantic review call receipt belongs to another proposal")
    if str(successful.get("prompt_version") or "") != prompt_version:
        raise ValueError("semantic review prompt version differs from the configured reviewer")
    result = {
        "model_id": str(successful.get("model_id") or ""),
        "prompt_version": str(successful.get("prompt_version") or ""),
        "logical_call_id": str(successful.get("logical_call_id") or ""),
        "prompt_hash": str(successful.get("prompt_sha256") or ""),
        "response_hash": str(successful.get("response_sha256") or ""),
        "proposal_hash": stable_content_hash(expected_payload),
    }
    if any(not value for value in result.values()):
        raise ValueError("semantic review call receipt is incomplete")
    return result


def _terminal_review_receipt(
    state: Mapping[str, Any],
    *,
    answer_plan: AnswerPlan,
    reviewed_answer_plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    review: ReviewProposal,
    reason_code: str,
) -> SemanticReviewTerminalReceipt:
    """把实际 provider review 绑定为最终无事实正文的弃答终态。"""

    verification = state.get("answer_verification")
    if not isinstance(verification, Mapping):
        raise ValueError("terminal review receipt lacks verification metadata")
    prompt_version = str(verification.get("review_prompt_version") or "")
    attestation = _provider_review_attestation(
        state,
        review=review,
        prompt_version=prompt_version,
    )
    return SemanticReviewTerminalReceipt.build(
        answer_plan=answer_plan,
        reviewed_answer_plan=reviewed_answer_plan,
        catalog=catalog,
        decision=review.decision.value,
        reason_code=reason_code,
        reviewed_closed_obligation_ids=review.closed_obligation_ids,
        review_prompt_version=attestation["prompt_version"],
        model_id=attestation["model_id"],
        verifier_version=_SEMANTIC_REVIEW_VERIFIER_VERSION,
        review_logical_call_id=attestation["logical_call_id"],
        review_prompt_hash=attestation["prompt_hash"],
        review_response_hash=attestation["response_hash"],
        review_proposal_hash=attestation["proposal_hash"],
    )


def _validate_review_receipt_attestation(
    state: Mapping[str, Any],
    *,
    plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    receipt: SemanticReviewClosureReceipt | SemanticReviewTerminalReceipt,
) -> None:
    review = ReviewProposal.model_validate(state.get("answer_plan_review") or {})
    if stable_content_hash(review.model_dump(mode="json")) != receipt.review_proposal_hash:
        raise ValueError("semantic review receipt differs from the review proposal")
    if isinstance(receipt, SemanticReviewClosureReceipt):
        if (
            review.decision != ReviewDecision.ACCEPT
            or review.closed_obligation_ids != receipt.closed_obligation_ids
        ):
            raise ValueError("semantic closure differs from the accepted review proposal")
        reviewed_plan = plan
    else:
        reviewed_plan = AnswerPlan.from_dict(state.get("answer_reviewed_plan") or {})
        if (
            review.decision.value != receipt.decision
            or review.closed_obligation_ids != receipt.reviewed_closed_obligation_ids
        ):
            raise ValueError("terminal review receipt differs from its reviewer decision")
        receipt.validate_against(
            answer_plan=plan,
            reviewed_answer_plan=reviewed_plan,
            catalog=catalog,
        )
    if receipt.review_logical_call_id.startswith("lrv_"):
        expected = _local_review_attestation(plan=plan, catalog=catalog, review=review)
        if (
            not isinstance(receipt, SemanticReviewClosureReceipt)
            or catalog.witness_count != 0
            or plan.decision != AnswerDecision.REFUSE
        ):
            raise ValueError("local empty-witness review attestation is not applicable")
    else:
        expected = _provider_review_attestation(
            state,
            review=review,
            prompt_version=receipt.review_prompt_version,
        )
    actual = {
        "model_id": receipt.model_id,
        "prompt_version": receipt.review_prompt_version,
        "logical_call_id": receipt.review_logical_call_id,
        "prompt_hash": receipt.review_prompt_hash,
        "response_hash": receipt.review_response_hash,
        "proposal_hash": receipt.review_proposal_hash,
    }
    if actual != expected:
        raise ValueError("semantic closure attestation differs from its review call")


def validate_committable_answer(state: Mapping[str, Any]) -> AnswerProof:
    """在交付边界重放完整 proof 图并验证最终正文。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
    proof = AnswerProof.from_dict(state.get("answer_proof") or {})
    catalog: WitnessCatalogReceipt | None = None
    if plan.decision not in {AnswerDecision.CLARIFY, AnswerDecision.DIRECT_ANSWER}:
        catalog = _catalog(state)
    resolved_views = (
        resolve_witness_prompt_views(state, _bundle(state)) if catalog is not None else []
    )
    proof.validate_against(
        answer_plan=plan,
        question_contract=question,
        catalog=catalog,
        witnesses=_witnesses(state) if catalog is not None else (),
        witness_texts=(
            {str(item["witness_id"]): str(item["content"]) for item in resolved_views}
            if catalog is not None
            else None
        ),
        validated_source_enumeration_closure_receipts=(
            collect_source_enumeration_closure_receipts(resolved_views)
            if catalog is not None
            else ()
        ),
    ).validate_answer_text(str(state.get("answer") or ""))
    if not proof.is_committable:
        raise AnswerProofNotCommittableError(
            f"answer proof is not committable: {proof.overall_status.value}"
        )
    if catalog is not None:
        try:
            state_closure = semantic_review_receipt_from_dict(
                state.get("answer_review_closure_receipt") or {}
            )
            if (
                proof.semantic_review_closure is None
                or state_closure.to_dict() != proof.semantic_review_closure.to_dict()
            ):
                raise ValueError("proof and query state carry different review closures")
            _validate_review_receipt_attestation(
                state,
                plan=plan,
                catalog=catalog,
                receipt=state_closure,
            )
        except (TypeError, ValueError) as exc:
            raise AnswerProofNotCommittableError(
                "semantic review closure cannot be replayed"
            ) from exc
    answer_id = str(state.get("answer_id") or "")
    query_run_id = str(state.get("query_run_id") or "")
    answer = str(state.get("answer") or "")
    expected_answer, expected_claim_proofs, expected_citation_specs = replay_answer_derivation(
        state,
        plan=plan,
        bundle=_bundle(state) if catalog is not None else None,
    )
    if answer != expected_answer or [item.to_dict() for item in proof.claim_proofs] != [
        item.to_dict() for item in expected_claim_proofs
    ]:
        raise AnswerProofNotCommittableError(
            "answer text or claim proof is not derived from the frozen Witness graph"
        )
    if proof.answer_id != answer_id:
        raise AnswerProofNotCommittableError("proof answer_id differs from query state")
    envelope = AnswerEnvelope.model_validate(state.get("answer_envelope") or {})
    expected_envelope_decision = {
        AnswerDecision.ANSWER: "ANSWER",
        AnswerDecision.PARTIAL: "PARTIAL_ANSWER",
        AnswerDecision.REFUSE: "REFUSE",
        AnswerDecision.CLARIFY: "CLARIFY",
        AnswerDecision.DIRECT_ANSWER: "DIRECT_ANSWER",
    }[plan.decision]
    if (
        envelope.answer_id != answer_id
        or envelope.query_run_id != query_run_id
        or envelope.answer != answer
        or envelope.decision != expected_envelope_decision
    ):
        raise AnswerProofNotCommittableError("answer envelope identity graph is inconsistent")
    raw_citations = state.get("citations") or []
    raw_pending_facts = state.get("pending_citation_facts") or []
    if not isinstance(raw_citations, list) or not isinstance(raw_pending_facts, list):
        raise AnswerProofNotCommittableError("citation bundles must be ordered lists")
    citation_fields = {
        "schema_version",
        "citation_id",
        "answer_id",
        "evidence_id",
        "answer_span",
        "source_quote",
        "quote_hash",
        "source_label",
        "validation_status",
        "reason_codes",
        "ordinal",
    }
    try:
        pending_facts = [CitationFact.model_validate(item) for item in raw_pending_facts]
        public_facts = [
            CitationFact.model_validate(
                {key: value for key, value in item.items() if key in citation_fields}
            )
            for item in raw_citations
            if isinstance(item, Mapping)
        ]
    except ValueError as exc:
        raise AnswerProofNotCommittableError("citation fact bundle is invalid") from exc
    expected_citation_count = len(plan.substantive_claims)
    public_claim_ids = [
        str(item.get("claim_id") or "") for item in raw_citations if isinstance(item, Mapping)
    ]
    if (
        len(public_facts) != len(raw_citations)
        or any(
            not isinstance(item, Mapping) or not str(item.get("citation_id") or "")
            for item in (*raw_citations, *raw_pending_facts)
        )
        or len(pending_facts) != expected_citation_count
        or len(public_facts) != expected_citation_count
        or public_claim_ids != [item.claim_id for item in plan.substantive_claims]
        or any(item.answer_id != answer_id for item in (*pending_facts, *public_facts))
        or any(item.validation_status != "valid" for item in pending_facts)
        or [item.model_dump(mode="json", exclude_none=False) for item in pending_facts]
        != [item.model_dump(mode="json", exclude_none=False) for item in public_facts]
    ):
        raise AnswerProofNotCommittableError("citation identity graph is inconsistent")
    if len(expected_citation_specs) != len(raw_citations) or any(
        not isinstance(actual, Mapping)
        or str(actual.get("claim_id") or "") != str(expected["claim_id"])
        or str(actual.get("obligation_id") or "") != str(expected["obligation_id"] or "")
        or str(actual.get("witness_id") or "") != str(expected["witness_id"])
        or str(actual.get("evidence_id") or "") != str(expected["evidence_id"])
        or str(actual.get("claim_text") or "") != str(expected["content"])
        or str(actual.get("source_quote") or "") != str(expected["source_quote"])
        or actual.get("answer_span") != expected["answer_span"].model_dump(mode="json")
        for expected, actual in zip(
            expected_citation_specs,
            raw_citations,
            strict=True,
        )
    ):
        raise AnswerProofNotCommittableError(
            "citation text is not derived from the frozen Witness graph"
        )
    if len(envelope.claims) != expected_citation_count or len(envelope.citations) != (
        expected_citation_count
    ):
        raise AnswerProofNotCommittableError("answer envelope citation graph is inconsistent")
    for claim_plan, fact, raw_citation, envelope_claim, envelope_citation in zip(
        plan.substantive_claims,
        public_facts,
        raw_citations,
        envelope.claims,
        envelope.citations,
        strict=True,
    ):
        if (
            envelope_claim.claim_id != claim_plan.claim_id
            or envelope_claim.answer_start != fact.answer_span.start
            or envelope_claim.answer_end != fact.answer_span.end
            or envelope_claim.text != str(raw_citation.get("claim_text") or "")
            or not envelope_claim.supported
            or envelope_citation.claim_id != claim_plan.claim_id
            or envelope_citation.evidence_id != fact.evidence_id
            or envelope_citation.answer_start != fact.answer_span.start
            or envelope_citation.answer_end != fact.answer_span.end
            or envelope_citation.source_quote_hash != fact.quote_hash
            or envelope_citation.validation != "valid"
        ):
            raise AnswerProofNotCommittableError("answer envelope citation graph is inconsistent")
    citation_bundle_hash = citation_fact_bundle_hash(pending_facts)
    render_receipt = state.get("answer_render_receipt")
    if not isinstance(render_receipt, Mapping) or (
        str(render_receipt.get("schema_version") or "") != "answer-render-receipt.v3"
        or str(render_receipt.get("answer_id") or "") != answer_id
        or str(render_receipt.get("query_run_id") or "") != query_run_id
        or str(render_receipt.get("answer_sha256") or "") != sha256_digest(answer)
        or str(render_receipt.get("proof_id") or "") != proof.proof_id
        or str(render_receipt.get("proof_hash") or "") != proof.proof_hash
        or render_receipt.get("citation_count") != expected_citation_count
        or str(render_receipt.get("citation_fact_bundle_hash") or "") != citation_bundle_hash
    ):
        raise AnswerProofNotCommittableError("answer render receipt identity is inconsistent")
    return proof


def _bundle(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("witness_catalog")
    if not isinstance(value, Mapping):
        raise ValueError("query state is missing witness catalog bundle")
    return value


def _catalog(state: Mapping[str, Any]) -> WitnessCatalogReceipt:
    return WitnessCatalogReceipt.from_dict(_bundle(state).get("receipt") or {})


def _witnesses(state: Mapping[str, Any]) -> tuple[EvidenceWitness, ...]:
    bundle = _bundle(state)
    witnesses = tuple(
        EvidenceWitness.from_dict(item)
        for item in bundle.get("witnesses") or []
        if isinstance(item, Mapping)
    )
    _catalog(state).validate_witnesses(witnesses)
    return witnesses


def _source_closure_by_obligation(
    views: list[Mapping[str, Any]],
) -> dict[str, SourceEnumerationClosureReceipt]:
    receipts = collect_source_enumeration_closure_receipts(views)
    result = {item.obligation_id: item for item in receipts}
    if len(result) != len(receipts):
        raise ValueError("source closure obligations must be unique")
    return result


def _obligation_coverage_receipts(
    *,
    question: QuestionContract,
    plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    witnesses: tuple[EvidenceWitness, ...],
    witness_texts: Mapping[str, str],
    source_closure_by_obligation: Mapping[str, SourceEnumerationClosureReceipt],
    closed_obligation_ids: set[str],
    absence_authorized: bool,
    abstention_authorized: bool,
    conflict_requested: bool,
) -> list[dict[str, Any]]:
    """生成不含正文的逐义务闭包与基数回执。"""

    witness_by_id = {item.witness_id: item for item in witnesses}
    claims_by_obligation: dict[str, list[Any]] = {}
    for claim in plan.substantive_claims:
        if claim.obligation_id is not None:
            claims_by_obligation.setdefault(claim.obligation_id, []).append(claim)
    unresolved = set(plan.unresolved_obligation_ids)
    receipts: list[dict[str, Any]] = []
    for obligation in question.obligations:
        claims = claims_by_obligation.get(obligation.obligation_id, [])
        candidate_receipt = source_closure_by_obligation.get(obligation.obligation_id)
        table_cell_projection = (
            resolve_closed_count_markdown_table_cell_projection(
                question=question,
                obligation=obligation,
                witnesses=witnesses,
                witness_texts=witness_texts,
            )
            if obligation.closure == FacetClosure.CLOSED_COUNT
            else None
        )
        proven_items: list[ProvenItemRef] = []
        seen_item_hashes: set[str] = set()
        for claim in claims:
            for witness_id in claim.witness_ids:
                witness = witness_by_id[witness_id]
                witness_text = witness_texts[witness_id]
                item_spans: tuple[tuple[int, int], ...]
                if isinstance(
                    candidate_receipt,
                    QueryAnchoredInlineActionChainClosureReceipt,
                ) and candidate_receipt.witness_ids == (witness_id,):
                    ordered_actions = sorted(
                        zip(
                            candidate_receipt.action_ordinals,
                            candidate_receipt.packed_action_starts,
                            candidate_receipt.packed_action_ends,
                            strict=True,
                        ),
                        key=lambda item: item[0],
                    )
                    item_spans = tuple(
                        (
                            start - candidate_receipt.packed_witness_start,
                            end - candidate_receipt.packed_witness_start,
                        )
                        for _ordinal, start, end in ordered_actions
                    )
                    method = ItemSegmentationMethod.INLINE_ACTION_SPAN
                elif obligation.closure == FacetClosure.SINGLE:
                    item_spans = ((0, len(witness_text)),)
                    method = ItemSegmentationMethod.SINGLE_WITNESS_ATOM
                elif witness.atom_kind == WitnessAtomKind.LIST_ITEM:
                    item_span = top_level_list_item_body_span(witness_text)
                    item_spans = (item_span,) if item_span is not None else ()
                    method = ItemSegmentationMethod.TOP_LEVEL_LIST_ENTRY
                elif witness.atom_kind == WitnessAtomKind.TABLE_DATA_ROW:
                    if (
                        table_cell_projection is not None
                        and table_cell_projection.status
                        == ClosedCountTableCellProjectionStatus.VALID
                    ):
                        item_spans = (
                            table_cell_projection.item_spans
                            if table_cell_projection.witness_id == witness.witness_id
                            else ()
                        )
                        method = ItemSegmentationMethod.MARKDOWN_TABLE_PROJECTED_CELL_ITEM
                    elif (
                        table_cell_projection is None
                        or table_cell_projection.status
                        == ClosedCountTableCellProjectionStatus.NOT_APPLICABLE
                    ):
                        item_span = markdown_table_data_row_body_span(witness_text)
                        item_spans = (item_span,) if item_span is not None else ()
                        method = ItemSegmentationMethod.MARKDOWN_TABLE_DATA_ROW
                    else:
                        item_spans = ()
                        method = ItemSegmentationMethod.MARKDOWN_TABLE_PROJECTED_CELL_ITEM
                else:
                    item_spans = ()
                    method = ItemSegmentationMethod.TOP_LEVEL_LIST_ENTRY
                if (
                    not item_spans
                    and obligation.closure == FacetClosure.OPEN_ENUMERATION
                    and witness_text.strip()
                ):
                    # 开放枚举只证明当前 Witness 原子是一个已支持的部分项；
                    # 该回退不参与 CLOSED_COUNT 基数闭包，也不能把开放集合升级为完整回答。
                    item_spans = ((0, len(witness_text)),)
                    method = ItemSegmentationMethod.OPEN_ENUMERATION_WITNESS_ATOM
                if not item_spans:
                    continue
                for item_span in item_spans:
                    item = ProvenItemRef.build(
                        obligation_id=obligation.obligation_id,
                        claim_id=claim.claim_id,
                        witness=witness,
                        witness_text=witness_text,
                        relative_start=item_span[0],
                        relative_end=item_span[1],
                        segmentation_method=method,
                    )
                    if (
                        method != ItemSegmentationMethod.INLINE_ACTION_SPAN
                        and item.normalized_item_hash in seen_item_hashes
                    ):
                        continue
                    seen_item_hashes.add(item.normalized_item_hash)
                    proven_items.append(item)

        itemized_claim_ids = {item.claim_id for item in proven_items}
        supporting_claim_ids = [
            item.claim_id
            for item in claims
            if (
                table_cell_projection is None
                or table_cell_projection.status
                == ClosedCountTableCellProjectionStatus.NOT_APPLICABLE
                or item.claim_id in itemized_claim_ids
            )
        ]
        closure_proven = False
        source_closure_receipt: SourceEnumerationClosureReceipt | None = None
        closure_status = ObligationClosureStatus.INDETERMINATE
        if supporting_claim_ids:
            if obligation.closure == FacetClosure.SINGLE:
                closure_proven = (
                    catalog.universe_status.value == "complete"
                    and obligation.obligation_id not in unresolved
                    and obligation.obligation_id in closed_obligation_ids
                    and len(claims) == 1
                    and len(proven_items) == 1
                )
            elif obligation.closure == FacetClosure.CLOSED_COUNT:
                closure_proven = (
                    catalog.universe_status.value == "complete"
                    and obligation.obligation_id not in unresolved
                    and obligation.obligation_id in closed_obligation_ids
                    and len(proven_items) == obligation.expected_count
                    and len(proven_items) >= len(claims)
                    and itemized_claim_ids == {item.claim_id for item in claims}
                )
                if not closure_proven:
                    closure_status = ObligationClosureStatus.INCOMPLETE
            else:
                selected_witness_ids = tuple(
                    witness_id for claim in claims for witness_id in claim.witness_ids
                )
                proven_witness_ids = tuple(item.witness_id for item in proven_items)
                if isinstance(
                    candidate_receipt,
                    QueryAnchoredInlineActionChainClosureReceipt,
                ):
                    closure_proven = (
                        catalog.universe_status.value == "complete"
                        and obligation.obligation_id not in unresolved
                        and obligation.obligation_id in closed_obligation_ids
                        and len(claims) == 1
                        and selected_witness_ids == candidate_receipt.witness_ids
                        and proven_witness_ids
                        == candidate_receipt.witness_ids * candidate_receipt.action_count
                        and len(proven_items) == candidate_receipt.action_count
                        and tuple(item.claim_id for item in proven_items)
                        == (claims[0].claim_id,) * candidate_receipt.action_count
                    )
                else:
                    closure_proven = (
                        catalog.universe_status.value == "complete"
                        and obligation.obligation_id not in unresolved
                        and obligation.obligation_id in closed_obligation_ids
                        and candidate_receipt is not None
                        and selected_witness_ids == source_closure_witness_ids(candidate_receipt)
                        and proven_witness_ids == selected_witness_ids
                        and len(proven_items) == len(claims)
                        and {item.claim_id for item in proven_items}
                        == {item.claim_id for item in claims}
                    )
                if closure_proven:
                    source_closure_receipt = candidate_receipt
            if closure_proven:
                outcome = ObligationOutcome.ANSWERED
                closure_status = ObligationClosureStatus.PROVEN_COMPLETE
            else:
                outcome = ObligationOutcome.PARTIALLY_ANSWERED
            status = ProofVerdict.PROVEN
            failure_codes: list[str] = []
        elif absence_authorized:
            outcome = ObligationOutcome.EVIDENCE_ABSENT
            closure_status = ObligationClosureStatus.NOT_APPLICABLE
            status = ProofVerdict.PROVEN
            failure_codes = []
        elif abstention_authorized:
            outcome = ObligationOutcome.INSUFFICIENT_RETRIEVED_EVIDENCE
            closure_status = ObligationClosureStatus.NOT_APPLICABLE
            status = ProofVerdict.PROVEN
            failure_codes = []
        else:
            outcome = ObligationOutcome.INDETERMINATE
            status = ProofVerdict.INDETERMINATE
            failure_codes = [
                (
                    "obligation_conflict_unresolved"
                    if conflict_requested
                    else "obligation_absence_unproven"
                )
            ]

        receipts.append(
            {
                "obligation_id": obligation.obligation_id,
                "status": status.value,
                "outcome": outcome.value,
                "supporting_claim_ids": supporting_claim_ids,
                "contradiction_witness_ids": [],
                "closure": obligation.closure.value,
                "required_output_form": obligation.required_output_form.value,
                "expected_item_count": obligation.expected_count,
                "proven_item_count": len(proven_items),
                "proven_items": [item.to_dict() for item in proven_items],
                "closure_status": closure_status.value,
                "source_enumeration_closure_receipt": (
                    source_closure_receipt.to_dict() if source_closure_receipt is not None else None
                ),
                "failure_codes": failure_codes,
            }
        )
    return receipts


def _normalize_plan_before_review(
    *,
    question: QuestionContract,
    plan: AnswerPlan,
    catalog: WitnessCatalogReceipt,
    witnesses: tuple[EvidenceWitness, ...],
    witness_texts: Mapping[str, str],
    source_closure_by_obligation: Mapping[str, SourceEnumerationClosureReceipt],
) -> tuple[AnswerPlan, tuple[str, ...]]:
    """在语义复核前收敛服务端可判定的 item/基数边界。"""

    if plan.decision not in {AnswerDecision.ANSWER, AnswerDecision.PARTIAL}:
        return plan, ()
    structural = _obligation_coverage_receipts(
        question=question,
        plan=plan,
        catalog=catalog,
        witnesses=witnesses,
        witness_texts=witness_texts,
        source_closure_by_obligation=source_closure_by_obligation,
        closed_obligation_ids={item.obligation_id for item in question.obligations},
        absence_authorized=False,
        abstention_authorized=False,
        conflict_requested=False,
    )
    structurally_unresolved = {
        item["obligation_id"]
        for item in structural
        if item["obligation_id"] in set(question.required_obligation_ids)
        and item["outcome"] != ObligationOutcome.ANSWERED.value
    }
    unresolved_ids = set(plan.unresolved_obligation_ids) | structurally_unresolved
    unresolved = tuple(
        obligation_id
        for obligation_id in question.required_obligation_ids
        if obligation_id in unresolved_ids
    )
    if unresolved == plan.unresolved_obligation_ids:
        return plan, ()
    return (
        AnswerPlan.build(
            question_contract=question,
            catalog=catalog,
            decision=AnswerDecision.PARTIAL,
            claims=plan.claims,
            unresolved_obligation_ids=unresolved,
            proposal_schema_version=plan.proposal_schema_version,
            prompt_contract_version=plan.prompt_contract_version,
        ),
        unresolved,
    )


def freeze_question_contract(state: dict[str, Any]) -> dict[str, Any]:
    """在检索前冻结原始问题义务。"""

    contract = build_question_contract(state)
    state["question_contract"] = contract.to_dict()
    return state


def freeze_witness_catalog(state: dict[str, Any]) -> dict[str, Any]:
    """在 ContextPack 后冻结精确 EvidenceWitness 目录。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    state["witness_catalog"] = build_witness_catalog_bundle(state, question)
    return state


def _empty_catalog_proposal() -> AnswerPlanProposal:
    return AnswerPlanProposal(
        decision=ProposalDecision.REFUSE,
        claims=(),
        reason_codes=(PlanReasonCode.NO_SUPPORTING_WITNESS,),
    )


def plan_answer(
    state: dict[str, Any],
    *,
    model_port: AnsweringModelPort,
) -> dict[str, Any]:
    """把初始 selector 或 reviewer replacement 投影为权威 AnswerPlan。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    state["answer_review_closure_receipt"] = {}
    direct = non_retrieval_answer_plan(state, question)
    if direct is not None:
        state["answer_plan_proposal"] = {
            "schema_version": "non-retrieval-answer-plan.v2",
            "decision": direct.decision.value,
            "claims": [],
            "reason_codes": ["RETRIEVAL_NOT_REQUIRED"],
        }
        state["answer_plan"] = direct.to_dict()
        state["answer_repair_requested"] = False
        state["answer_repair_transition"] = {}
        state["answer_repair_transitions"] = []
        return state

    bundle = _bundle(state)
    catalog = _catalog(state)
    views = resolve_witness_prompt_views(state, bundle)
    source_closure_by_obligation = _source_closure_by_obligation(views)
    repair_feedback = state.get("answer_repair_feedback")
    repair_review: ReviewProposal | None = None
    repair_source_plan: AnswerPlan | None = None
    repair_attestation: dict[str, str] | None = None
    if repair_feedback:
        current_repair_count = int(state.get("answer_repair_count") or 0)
        if not 0 <= current_repair_count < MAX_REPAIR_ATTEMPTS:
            raise ValueError("answer repair count is outside the bounded state space")
        repair_review = ReviewProposal.model_validate(repair_feedback)
        if repair_review.decision != ReviewDecision.REPAIR:
            raise ValueError("answer repair feedback must carry a repair decision")
        repair_source_plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
        reviewed_source_plan = AnswerPlan.from_dict(state.get("answer_reviewed_plan") or {})
        if (
            reviewed_source_plan.plan_id != repair_source_plan.plan_id
            or reviewed_source_plan.plan_hash != repair_source_plan.plan_hash
        ):
            raise ValueError("answer repair feedback belongs to a different reviewed plan")
        repair_attestation = _provider_review_attestation(
            state,
            review=repair_review,
            prompt_version=model_port.review_prompt_version,
        )
        applied_repair_count = current_repair_count + 1
        state["answer_repair_count"] = applied_repair_count
    state["answer_repair_requested"] = False
    if repair_review is not None:
        proposal = project_review_repair_proposal(
            review=repair_review,
            question_contract=question,
            catalog=catalog,
            source_closure_by_obligation=source_closure_by_obligation,
        )
    elif not views:
        proposal = _empty_catalog_proposal()
    else:
        proposal = model_port.select_plan(
            state,
            question_contract=question,
            witness_catalog=views,
        )
        validate_claim_allowlists(
            proposal.claims,
            allowed_obligation_ids=[item.obligation_id for item in question.obligations],
            allowed_witness_ids=catalog.witness_ids,
        )
    answer_plan = project_answer_plan(
        proposal=proposal,
        question_contract=question,
        catalog=catalog,
        prompt_contract_version=model_port.plan_prompt_version,
    )
    answer_plan, structurally_unresolved = _normalize_plan_before_review(
        question=question,
        plan=answer_plan,
        catalog=catalog,
        witnesses=_witnesses(state),
        witness_texts={str(item["witness_id"]): str(item["content"]) for item in views},
        source_closure_by_obligation=source_closure_by_obligation,
    )
    canonical_claims = tuple(
        ClaimProposal(
            obligation_id=str(item.obligation_id),
            witness_id=item.witness_ids[0],
            rendering_rule=(
                ProposalRenderingRule.EXACT_WITNESS
                if item.rendering_rule.value == "exact_witness"
                else ProposalRenderingRule.DETERMINISTIC_TEMPLATE
            ),
        )
        for item in answer_plan.substantive_claims
    )
    canonical_decision = ProposalDecision(answer_plan.decision.value)
    canonical_reasons = {
        ProposalDecision.ANSWER: (PlanReasonCode.ANSWER_SUPPORTED,),
        ProposalDecision.PARTIAL: (
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
        ProposalDecision.REFUSE: (PlanReasonCode.NO_SUPPORTING_WITNESS,),
    }[canonical_decision]
    canonical_proposal = AnswerPlanProposal(
        decision=canonical_decision,
        claims=canonical_claims,
        reason_codes=canonical_reasons,
    )
    state["answer_repair_feedback"] = {}
    state["answer_plan_raw_proposal"] = proposal.model_dump(mode="json")
    state["answer_plan_proposal"] = canonical_proposal.model_dump(mode="json")
    normalizations = [
        {
            "code": "REQUIRED_OUTPUT_FORM_NORMALIZATION",
            "obligation_id": source.obligation_id,
            "from_rule": source.rendering_rule.value,
            "to_rule": target.rendering_rule.value,
        }
        for source, target in zip(proposal.claims, canonical_claims, strict=True)
        if source.rendering_rule != target.rendering_rule
    ]
    if structurally_unresolved:
        normalizations.append(
            {
                "code": "LOCAL_CLOSURE_DOWNGRADE",
                "unresolved_obligation_ids": list(structurally_unresolved),
                "from_decision": "answer",
                "to_decision": "partial",
            }
        )
    if repair_review is not None:
        assert repair_source_plan is not None
        assert repair_attestation is not None
        proposal_payload = proposal.model_dump(mode="json")
        normalizations.append(
            {
                "code": "SEMANTIC_REVIEW_REPAIR_PROJECTION",
                "from_plan_id": repair_source_plan.plan_id,
                "to_plan_id": answer_plan.plan_id,
            }
        )
        transition = {
            "schema_version": "answer-repair-transition.v1",
            "status": "applied",
            "repair_count": applied_repair_count,
            "source_plan_id": repair_source_plan.plan_id,
            "source_plan_hash": repair_source_plan.plan_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "source_review_logical_call_id": repair_attestation["logical_call_id"],
            "source_review_prompt_hash": repair_attestation["prompt_hash"],
            "source_review_response_hash": repair_attestation["response_hash"],
            "source_review_proposal_hash": repair_attestation["proposal_hash"],
            "replacement_proposal_hash": stable_content_hash(proposal_payload),
            "resulting_plan_id": answer_plan.plan_id,
            "resulting_plan_hash": answer_plan.plan_hash,
        }
        raw_history = state.get("answer_repair_transitions") or []
        if not isinstance(raw_history, list) or any(
            not isinstance(item, Mapping) for item in raw_history
        ):
            raise ValueError("answer repair transition history is invalid")
        history = [dict(item) for item in raw_history]
        prior_counts = [int(item.get("repair_count") or 0) for item in history]
        expected_prior_counts = list(range(1, applied_repair_count))
        if prior_counts != expected_prior_counts:
            raise ValueError("answer repair transition history is not contiguous")
        for index, item in enumerate(history):
            if (
                item.get("schema_version") != "answer-repair-transition.v1"
                or item.get("status") != "applied"
                or item.get("catalog_id") != catalog.catalog_id
                or item.get("catalog_hash") != catalog.catalog_hash
            ):
                raise ValueError("answer repair transition history binding is invalid")
            if index:
                previous = history[index - 1]
                if previous.get("resulting_plan_id") != item.get("source_plan_id") or previous.get(
                    "resulting_plan_hash"
                ) != item.get("source_plan_hash"):
                    raise ValueError("answer repair transition history chain is broken")
        if history and (
            history[-1].get("resulting_plan_id") != repair_source_plan.plan_id
            or history[-1].get("resulting_plan_hash") != repair_source_plan.plan_hash
        ):
            raise ValueError("answer repair transition history does not bind the current plan")
        review_call_ids = [str(item.get("source_review_logical_call_id") or "") for item in history]
        review_call_ids.append(str(transition["source_review_logical_call_id"]))
        if any(not item for item in review_call_ids) or len(review_call_ids) != len(
            set(review_call_ids)
        ):
            raise ValueError("answer repair transition review calls are not unique")
        history.append(transition)
        state["answer_repair_transition"] = transition
        state["answer_repair_transitions"] = history
    else:
        state["answer_repair_transition"] = {}
    state["answer_plan_normalizations"] = normalizations
    state["answer_plan"] = answer_plan.to_dict()
    return state


def verify_answer_plan(
    state: dict[str, Any],
    *,
    model_port: AnsweringModelPort,
) -> dict[str, Any]:
    """让独立 reviewer 判断 obligation↔witness 语义关系。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
    if plan.decision in {AnswerDecision.CLARIFY, AnswerDecision.DIRECT_ANSWER}:
        state["answer_review_closure_receipt"] = {}
        state["answer_plan_review"] = {
            "schema_version": "answer-plan-review-receipt.v2",
            "decision": "accept",
            "claims": [],
            "reason_codes": ["NON_RETRIEVAL_NOT_APPLICABLE"],
        }
        state["answer_verification"] = {
            "schema_version": "answer-plan-verification.v2",
            "status": "accepted",
            "semantic_review": "not_applicable",
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
        }
        return state

    bundle = _bundle(state)
    catalog = _catalog(state)
    views = resolve_witness_prompt_views(state, bundle)
    deterministic_review = not views and plan.decision == AnswerDecision.REFUSE
    if deterministic_review:
        review = ReviewProposal(
            decision=ReviewDecision.ACCEPT,
            claims=(),
            reason_codes=(ReviewReasonCode.PLAN_SEMANTICALLY_VALID,),
        )
    else:
        raw_proposal = state.get("answer_plan_proposal")
        proposal = AnswerPlanProposal.model_validate(raw_proposal or {})
        projected_claims = tuple(
            (
                item.obligation_id,
                item.witness_ids[0],
                (
                    "exact_witness"
                    if item.rendering_rule.value == "exact_witness"
                    else "deterministic_template"
                ),
            )
            for item in plan.substantive_claims
        )
        proposed_claims = tuple(
            (item.obligation_id, item.witness_id, item.rendering_rule.value)
            for item in proposal.claims
        )
        if sorted(proposed_claims) != sorted(projected_claims):
            raise ValueError("semantic reviewer input differs from the projected AnswerPlan")
        canonical_claims = tuple(
            ClaimProposal(
                obligation_id=str(item.obligation_id),
                witness_id=item.witness_ids[0],
                rendering_rule=(
                    ProposalRenderingRule.EXACT_WITNESS
                    if item.rendering_rule.value == "exact_witness"
                    else ProposalRenderingRule.DETERMINISTIC_TEMPLATE
                ),
            )
            for item in plan.substantive_claims
        )
        if plan.decision == AnswerDecision.ANSWER:
            canonical_decision = ProposalDecision.ANSWER
            canonical_reasons = (PlanReasonCode.ANSWER_SUPPORTED,)
        elif plan.decision == AnswerDecision.PARTIAL:
            canonical_decision = ProposalDecision.PARTIAL
            canonical_reasons = (
                PlanReasonCode.PARTIAL_SUPPORT,
                PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
            )
        else:
            canonical_decision = ProposalDecision.REFUSE
            canonical_reasons = (PlanReasonCode.EVIDENCE_INSUFFICIENT,)
        proposal = AnswerPlanProposal(
            decision=canonical_decision,
            claims=canonical_claims,
            reason_codes=canonical_reasons,
        )
        repair_attempt = int(state.get("answer_repair_count") or 0)
        if not 0 <= repair_attempt <= MAX_REPAIR_ATTEMPTS:
            raise ValueError("answer repair count is outside the bounded state space")
        review = model_port.review_plan(
            state,
            question_contract=question,
            witness_catalog=views,
            plan_proposal=proposal,
            repair_attempt=repair_attempt,
        )
        validate_claim_allowlists(
            review.claims,
            allowed_obligation_ids=[item.obligation_id for item in question.obligations],
            allowed_witness_ids=catalog.witness_ids,
        )
    known_obligations = {item.obligation_id for item in question.obligations}
    closed_obligations = set(review.closed_obligation_ids)
    if not closed_obligations.issubset(known_obligations):
        raise ValueError("review closed_obligation_ids contain an unknown obligation")
    if plan.decision == AnswerDecision.REFUSE and closed_obligations:
        raise ValueError("refusal review cannot certify an answered obligation closure")
    if closed_obligations & set(plan.unresolved_obligation_ids):
        raise ValueError("review cannot close an obligation left unresolved by the plan")
    state["answer_plan_review"] = review.model_dump(mode="json")
    state["answer_reviewed_plan"] = plan.to_dict()
    state["answer_verification"] = {
        "schema_version": "answer-plan-verification.v2",
        "status": review.decision.value,
        "reason_codes": [item.value for item in review.reason_codes],
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "catalog_id": catalog.catalog_id,
        "reviewed_claim_count": len(plan.claims),
        "closed_obligation_ids": list(review.closed_obligation_ids),
        "review_prompt_version": (
            "not-applicable.empty-witness.v1"
            if deterministic_review
            else model_port.review_prompt_version
        ),
    }
    if review.decision == ReviewDecision.ACCEPT:
        attestation = (
            _local_review_attestation(plan=plan, catalog=catalog, review=review)
            if deterministic_review
            else _provider_review_attestation(
                state,
                review=review,
                prompt_version=model_port.review_prompt_version,
            )
        )
        closure_receipt = SemanticReviewClosureReceipt.build(
            answer_plan=plan,
            catalog=catalog,
            closed_obligation_ids=review.closed_obligation_ids,
            review_prompt_version=attestation["prompt_version"],
            model_id=attestation["model_id"],
            verifier_version=_SEMANTIC_REVIEW_VERIFIER_VERSION,
            review_logical_call_id=attestation["logical_call_id"],
            review_prompt_hash=attestation["prompt_hash"],
            review_response_hash=attestation["response_hash"],
            review_proposal_hash=attestation["proposal_hash"],
        )
        state["answer_review_closure_receipt"] = closure_receipt.to_dict()
    else:
        state["answer_review_closure_receipt"] = {}
    return state


def decide_answer_coverage(state: dict[str, Any]) -> dict[str, Any]:
    """收敛 reviewer 结论并开放最多两次显式 repair 回边。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
    if plan.decision in {AnswerDecision.CLARIFY, AnswerDecision.DIRECT_ANSWER}:
        state["answer_coverage_decision"] = {
            "schema_version": "answer-coverage-decision.v2",
            "status": "accepted",
            "reason_code": "NON_RETRIEVAL_NOT_APPLICABLE",
            "required_count": 0,
            "covered_count": 0,
            "repair_count": 0,
        }
        state["answer_repair_requested"] = False
        return state

    catalog = _catalog(state)
    witnesses = _witnesses(state)
    views = resolve_witness_prompt_views(state, _bundle(state))
    witness_texts = {str(item["witness_id"]): str(item["content"]) for item in views}
    source_closure_by_obligation = _source_closure_by_obligation(views)
    review = ReviewProposal.model_validate(state.get("answer_plan_review") or {})
    reviewed_plan = AnswerPlan.from_dict(state.get("answer_reviewed_plan") or {})
    if reviewed_plan.plan_id != plan.plan_id or reviewed_plan.plan_hash != plan.plan_hash:
        raise ValueError("coverage decision received a different reviewed AnswerPlan")
    repair_count = int(state.get("answer_repair_count") or 0)
    if not 0 <= repair_count <= MAX_REPAIR_ATTEMPTS:
        raise ValueError("answer repair count is outside the bounded state space")
    status = "accepted"
    reason_code = "ANSWER_PLAN_PROVEN"
    semantic_accepted = review.decision == ReviewDecision.ACCEPT
    if review.decision == ReviewDecision.REPAIR and repair_count < MAX_REPAIR_ATTEMPTS:
        status = "repair"
        reason_code = "SEMANTIC_REPAIR_REQUESTED"
        state["answer_repair_requested"] = True
        state["answer_repair_feedback"] = review.model_dump(mode="json")
    elif review.decision == ReviewDecision.REPAIR:
        plan = refusal_answer_plan(
            question_contract=question,
            catalog=catalog,
            prompt_contract_version=str(plan.prompt_contract_version),
        )
        state["answer_plan"] = plan.to_dict()
        state["answer_review_closure_receipt"] = _terminal_review_receipt(
            state,
            answer_plan=plan,
            reviewed_answer_plan=reviewed_plan,
            catalog=catalog,
            review=review,
            reason_code="REPAIR_LIMIT_REACHED",
        ).to_dict()
        state["answer_repair_requested"] = False
        status = "accepted"
        reason_code = "REPAIR_LIMIT_REACHED"
    elif review.decision == ReviewDecision.REJECT:
        terminal_reason = (
            "REPAIR_LIMIT_REACHED"
            if ReviewReasonCode.REPAIR_LIMIT_REACHED in review.reason_codes
            else "SEMANTIC_REVIEW_REJECTED"
        )
        plan = refusal_answer_plan(
            question_contract=question,
            catalog=catalog,
            prompt_contract_version=str(plan.prompt_contract_version),
        )
        state["answer_plan"] = plan.to_dict()
        state["answer_review_closure_receipt"] = _terminal_review_receipt(
            state,
            answer_plan=plan,
            reviewed_answer_plan=reviewed_plan,
            catalog=catalog,
            review=review,
            reason_code=terminal_reason,
        ).to_dict()
        state["answer_repair_requested"] = False
        status = "accepted"
        reason_code = terminal_reason
    else:
        state["answer_repair_requested"] = False

    raw_proposal = state.get("answer_plan_proposal")
    proposal_reason_codes: set[str] = set()
    if isinstance(raw_proposal, Mapping):
        proposal_reason_codes = {str(item) for item in raw_proposal.get("reason_codes") or []}
    # ContextPack 完整不等于检索全集完备；在没有专门的 exhaustive retrieval
    # closure receipt 前，任何检索拒答都不能被证明为 Evidence 不存在。
    absence_authorized = False
    abstention_authorized = (
        status == "accepted"
        and plan.decision == AnswerDecision.REFUSE
        and bool(state.get("answer_review_closure_receipt"))
    )
    obligation_receipts = _obligation_coverage_receipts(
        question=question,
        plan=plan,
        catalog=catalog,
        witnesses=witnesses,
        witness_texts=witness_texts,
        source_closure_by_obligation=source_closure_by_obligation,
        closed_obligation_ids=set(review.closed_obligation_ids),
        absence_authorized=absence_authorized,
        abstention_authorized=abstention_authorized,
        conflict_requested=PlanReasonCode.WITNESS_CONFLICT.value in proposal_reason_codes,
    )

    required_ids = set(question.required_obligation_ids)
    unresolved_from_receipts = tuple(
        item["obligation_id"]
        for item in obligation_receipts
        if item["obligation_id"] in required_ids
        and item["outcome"] != ObligationOutcome.ANSWERED.value
    )
    planned_unresolved = set(plan.unresolved_obligation_ids)
    actual_unresolved = set(unresolved_from_receipts)
    if status == "accepted" and actual_unresolved != planned_unresolved:
        # 该差异表示服务端闭包不变量已破坏；复制原 claims 不会产生新 Observation，
        # 也没有可绑定的 provider repair 收据，因此必须直接收敛为可审计安全终态。
        plan = refusal_answer_plan(
            question_contract=question,
            catalog=catalog,
            prompt_contract_version=str(plan.prompt_contract_version),
        )
        state["answer_plan"] = plan.to_dict()
        state["answer_review_closure_receipt"] = _terminal_review_receipt(
            state,
            answer_plan=plan,
            reviewed_answer_plan=reviewed_plan,
            catalog=catalog,
            review=review,
            reason_code="OBLIGATION_CLOSURE_UNPROVEN",
        ).to_dict()
        state["answer_repair_requested"] = False
        state["answer_repair_feedback"] = {}
        semantic_accepted = False
        status = "accepted"
        reason_code = "OBLIGATION_CLOSURE_UNPROVEN"
        obligation_receipts = _obligation_coverage_receipts(
            question=question,
            plan=plan,
            catalog=catalog,
            witnesses=witnesses,
            witness_texts=witness_texts,
            source_closure_by_obligation=source_closure_by_obligation,
            closed_obligation_ids=set(),
            absence_authorized=False,
            abstention_authorized=True,
            conflict_requested=False,
        )
    elif plan.decision == AnswerDecision.REFUSE:
        if abstention_authorized:
            if reason_code == "ANSWER_PLAN_PROVEN":
                reason_code = "INSUFFICIENT_RETRIEVED_EVIDENCE"
        elif not absence_authorized:
            status = "indeterminate"
            reason_code = (
                "WITNESS_CONFLICT_UNPROVEN"
                if PlanReasonCode.WITNESS_CONFLICT.value in proposal_reason_codes
                else "REFUSAL_ABSENCE_UNPROVEN"
            )

    covered = {
        str(item["obligation_id"])
        for item in obligation_receipts
        if item["outcome"]
        in {
            ObligationOutcome.ANSWERED.value,
            ObligationOutcome.PARTIALLY_ANSWERED.value,
        }
    }
    state["answer_coverage_decision"] = {
        "schema_version": "answer-coverage-decision.v4",
        "status": status,
        "reason_code": reason_code,
        "required_count": len(question.required_obligation_ids),
        "covered_count": len(covered),
        "unresolved_obligation_ids": list(plan.unresolved_obligation_ids),
        "repair_count": repair_count,
        "catalog_universe_status": catalog.universe_status.value,
        "semantic_review_accepted": semantic_accepted,
        "proposal_reason_codes": sorted(proposal_reason_codes),
        "obligations": obligation_receipts,
    }
    return state


def render_proven_answer(
    state: dict[str, Any],
    *,
    render_port: AnswerRenderPort,
) -> dict[str, Any]:
    """确定性渲染并生成 proof、citation 和公开 envelope。"""

    return render_answer_and_proof(state, render_port=render_port)


def commit_proven_answer(
    state: dict[str, Any],
    *,
    commit_port: AnswerCommitPort,
) -> dict[str, Any]:
    """在 render/proof checkpoint 后执行最终交付。"""

    validate_committable_answer(state)
    state["answer_commit_receipt"] = commit_port(state)
    return state


__all__ = [
    "AnswerProofNotCommittableError",
    "commit_proven_answer",
    "decide_answer_coverage",
    "freeze_question_contract",
    "freeze_witness_catalog",
    "plan_answer",
    "render_proven_answer",
    "validate_committable_answer",
    "verify_answer_plan",
]
