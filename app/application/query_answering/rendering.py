"""从已验证 AnswerPlan 确定性渲染正文、引用和 AnswerProof。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from app.application.query_answering.builders import (
    collect_source_enumeration_closure_receipts,
    resolve_witness_prompt_views,
)
from app.application.query_answering.planning import refusal_answer_plan
from app.application.query_answering.ports import AnswerRenderPort
from app.application.query_answering.proposals import MAX_REPAIR_ATTEMPTS
from app.application.query_security_pipeline import secure_answer_state
from app.domain.answering.failures import (
    AnsweringFailure,
    AnsweringFailureCode,
    FailureStage,
    Retryability,
)
from app.domain.answering.enumeration_closure import (
    parse_source_enumeration_closure_receipt,
)
from app.domain.answering.item_boundary import top_level_list_item_body_span
from app.domain.answering.plan import AnswerDecision, AnswerPlan, RenderingRule
from app.domain.answering.proof import (
    AnswerProof,
    ClaimProof,
    CompletenessProofStatus,
    ObligationClosureStatus,
    ObligationOutcome,
    ObligationProof,
    OverallProofStatus,
    ProofDimensionStatus,
    ProofRelation,
    ProofVerdict,
    ProvenItemRef,
    SemanticReviewClosureReceipt,
    SemanticReviewTerminalReceipt,
    VerifierRef,
    semantic_review_receipt_from_dict,
)
from app.domain.answering.question import QuestionContract, RequiredOutputForm
from app.domain.answering.witness import (
    EvidenceSpanRef,
    EvidenceWitness,
    SpanEquivalence,
    WitnessCatalogReceipt,
)
from app.domain.evidence import (
    AnswerSpan,
    CitationFact,
    Evidence,
    citation_fact_bundle_hash,
    sha256_digest,
)
from app.domain.query_planning import FacetClosure, QueryPlan
from app.domain.safe_representation import SafeRepresentationDerivationReceipt


_CLAIM_VERIFIER = VerifierRef(
    verifier_id="deterministic-span-renderer",
    verifier_version="proof-carrying-renderer.v2",
)
_SEMANTIC_VERIFIER = VerifierRef(
    verifier_id="structured-semantic-review",
    verifier_version="answer-plan-review.v1",
)
_COVERAGE_VERIFIER = VerifierRef(
    verifier_id="question-obligation-coverage",
    verifier_version="answer-coverage.v4",
)
_SafeReplayCache = dict[tuple[str, str, str], SafeRepresentationDerivationReceipt]


def _notice_text(state: Mapping[str, Any], plan: AnswerPlan) -> str:
    query_plan = QueryPlan.model_validate(state.get("query_plan") or {})
    template_id = plan.claims[0].template_id if plan.claims else None
    if template_id == "notice.greeting.v1":
        return "您好，我是掌柜智库助手。请告诉我您想查询的知识库问题。"
    if template_id == "notice.ui-help.v1":
        return (
            "请先登录并选择当前知识库，再描述要查询的对象、问题和必要范围；"
            "系统只使用该知识库中已发布的证据回答。"
        )
    if template_id == "notice.clarification.v1":
        question = query_plan.understanding.ambiguity.clarification_question
        return str(question or "请补充问题中缺少的对象或范围。")
    if template_id == "notice.no-support.v1":
        return "当前已检索证据中没有足以支持该问题的内容。"
    if template_id == "notice.insufficient-context.v1":
        return "当前可用证据不足以可靠回答；证据目录不完整，无法据此断言相关信息不存在。"
    return "请描述需要查询的知识库问题。"


def _claim_prefix(template_id: str | None, ordinal: int, *, content: str) -> str:
    if template_id in {"template.verbatim-list.v1", "template.verbatim-comparison.v1"}:
        return "- "
    if template_id == "template.verbatim-steps.v1":
        if top_level_list_item_body_span(content) is not None:
            return ""
        return f"{ordinal + 1}. "
    return ""


def _decision_projection(decision: AnswerDecision) -> str:
    return {
        AnswerDecision.ANSWER: "ANSWER",
        AnswerDecision.PARTIAL: "PARTIAL_ANSWER",
        AnswerDecision.REFUSE: "REFUSE",
        AnswerDecision.CLARIFY: "CLARIFY",
        AnswerDecision.DIRECT_ANSWER: "DIRECT_ANSWER",
    }[decision]


def _alias_source_labels(state: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in state.get("context_aliases") or []:
        if not isinstance(item, Mapping) or not isinstance(item.get("evidence"), Mapping):
            continue
        evidence = item["evidence"]
        result[str(evidence.get("evidence_id") or "")] = str(
            evidence.get("source_label") or item.get("label") or "证据"
        )
    return result


def _span_alias(
    state: Mapping[str, Any],
    span_ref: EvidenceSpanRef,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in state.get("context_aliases") or []
        if isinstance(item, Mapping)
        and isinstance(item.get("evidence"), Mapping)
        and str(item["evidence"].get("evidence_id") or "") == span_ref.evidence_id
    ]
    if len(matches) != 1:
        raise ValueError("citation span does not resolve to exactly one ContextPack alias")
    return matches[0]


def _validate_render_span(
    state: Mapping[str, Any],
    *,
    span_ref: EvidenceSpanRef,
    canonical_text: str,
    safe_replay_cache: _SafeReplayCache,
) -> str:
    """在 renderer 边界重验 typed equivalence 并解析 canonical quote。"""

    alias = _span_alias(state, span_ref)
    packed_text = str(alias.get("packed_text") or "")
    if span_ref.equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED:
        document = alias.get("document")
        if not isinstance(document, Mapping):
            raise ValueError("security-safe citation is missing its source document")
        raw_receipt = document.get("safe_representation_derivation")
        raw_safe = document.get("security_safe_content")
        if raw_receipt is None or raw_safe is None:
            raise ValueError("security-safe citation is missing its derivation receipt")
        safe_text = str(raw_safe)
        receipt = SafeRepresentationDerivationReceipt.from_dict(raw_receipt)
        cache_key = (
            receipt.canonical_content_hash,
            receipt.safe_representation_hash,
            receipt.receipt_hash,
        )
        if receipt.canonical_content_hash != sha256_digest(
            canonical_text
        ) or receipt.safe_representation_hash != sha256_digest(safe_text):
            raise ValueError("security-safe citation active artifact hash mismatch")
        cached = safe_replay_cache.get(cache_key)
        if cached is None:
            receipt.validate_artifacts(
                canonical_text=canonical_text,
                safe_text=safe_text,
            )
            safe_replay_cache[cache_key] = receipt
        elif cached != receipt:
            raise ValueError("security-safe citation replay cache identity collision")
        if (
            span_ref.safe_derivation_transform_version != receipt.transform_version
            or span_ref.safe_derivation_receipt_hash != receipt.receipt_hash
        ):
            raise ValueError("security-safe citation derivation identity mismatch")
    span_ref.validate_texts(canonical_text=canonical_text, packed_text=packed_text)
    return span_ref.resolve_canonical_span(canonical_text)


def _render_claims(
    state: Mapping[str, Any],
    plan: AnswerPlan,
    bundle: Mapping[str, Any],
    *,
    resolved_views: list[dict[str, Any]] | None = None,
    safe_replay_cache: _SafeReplayCache | None = None,
) -> tuple[str, list[ClaimProof], list[dict[str, Any]]]:
    replay_cache = safe_replay_cache if safe_replay_cache is not None else {}
    materialized_views = (
        resolved_views
        if resolved_views is not None
        else resolve_witness_prompt_views(
            state,
            bundle,
            safe_replay_cache=replay_cache,
        )
    )
    views = {item["witness_id"]: item for item in materialized_views}
    answer_parts: list[str] = []
    claim_proofs: list[ClaimProof] = []
    citation_specs: list[dict[str, Any]] = []
    obligation_ordinals: dict[str, int] = defaultdict(int)
    cursor = 0
    for claim in plan.claims:
        if claim.rendering_rule == RenderingRule.SYSTEM_NOTICE:
            raise ValueError("system_notice must use the non-retrieval renderer")
        view = views.get(claim.witness_ids[0])
        if view is None:
            raise ValueError("AnswerPlan witness cannot be resolved")
        content = str(view["content"]).strip()
        if not content:
            raise ValueError("resolved Witness content cannot be blank")
        obligation_id = str(claim.obligation_id or "")
        obligation_ordinal = obligation_ordinals[obligation_id]
        prefix = _claim_prefix(claim.template_id, obligation_ordinal, content=content)
        obligation_ordinals[obligation_id] += 1
        label = str(view["label"])
        separator = "\n" if answer_parts else ""
        answer_parts.append(separator)
        cursor += len(separator)
        claim_start = cursor
        answer_parts.append(prefix)
        cursor += len(prefix)
        citation_start = cursor
        answer_parts.append(content)
        cursor += len(content)
        citation_end = cursor
        suffix = f" [{label}]"
        answer_parts.append(suffix)
        cursor += len(suffix)
        proof_span = AnswerSpan(start=claim_start, end=cursor)
        citation_span = AnswerSpan(start=citation_start, end=citation_end)
        span_ref = view["witness"].span_ref
        alias = _span_alias(state, span_ref)
        canonical_evidence = Evidence.model_validate(alias["evidence"])
        source_quote = _validate_render_span(
            state,
            span_ref=span_ref,
            canonical_text=canonical_evidence.content,
            safe_replay_cache=replay_cache,
        )
        claim_proofs.append(
            ClaimProof.build(
                claim_id=claim.claim_id,
                obligation_id=claim.obligation_id,
                status=ProofVerdict.PROVEN,
                method=claim.rendering_rule,
                witness_ids=claim.witness_ids,
                answer_span=proof_span,
                rendered_text_hash=sha256_digest(prefix + content + suffix),
                relation=ProofRelation.SUPPORTS,
                verifier=_CLAIM_VERIFIER,
            )
        )
        citation_specs.append(
            {
                "claim_id": claim.claim_id,
                "obligation_id": claim.obligation_id,
                "witness_id": claim.witness_ids[0],
                "evidence_id": str(view["evidence_id"]),
                "label": label,
                "content": content,
                "source_quote": source_quote,
                "span_ref": span_ref.to_dict(),
                "answer_span": citation_span,
            }
        )
    return "".join(answer_parts), claim_proofs, citation_specs


def _obligation_proofs(
    *,
    question: QuestionContract,
    catalog: WitnessCatalogReceipt,
    coverage: Mapping[str, Any],
) -> tuple[ObligationProof, ...]:
    if coverage.get("schema_version") != "answer-coverage-decision.v4":
        raise ValueError("answer coverage receipt schema is unsupported")
    raw_receipts = coverage.get("obligations")
    if not isinstance(raw_receipts, list):
        raise ValueError("answer coverage receipt is missing obligation proofs")
    receipt_by_id = {
        str(item.get("obligation_id") or ""): item
        for item in raw_receipts
        if isinstance(item, Mapping)
    }
    if tuple(receipt_by_id) != tuple(item.obligation_id for item in question.obligations):
        raise ValueError("answer coverage obligations are missing, duplicated or reordered")
    proofs: list[ObligationProof] = []
    for obligation in question.obligations:
        receipt = receipt_by_id[obligation.obligation_id]
        proven_items = tuple(
            ProvenItemRef.from_dict(item)
            for item in receipt.get("proven_items") or ()
            if isinstance(item, Mapping)
        )
        if int(receipt.get("proven_item_count") or 0) != len(proven_items):
            raise ValueError("answer coverage item count is inconsistent")
        raw_source_closure = receipt.get("source_enumeration_closure_receipt")
        source_closure = (
            parse_source_enumeration_closure_receipt(raw_source_closure)
            if isinstance(raw_source_closure, Mapping)
            else None
        )
        proofs.append(
            ObligationProof.build(
                obligation_id=obligation.obligation_id,
                status=ProofVerdict(str(receipt.get("status") or "")),
                outcome=ObligationOutcome(str(receipt.get("outcome") or "")),
                supporting_claim_ids=tuple(receipt.get("supporting_claim_ids") or ()),
                contradiction_witness_ids=tuple(receipt.get("contradiction_witness_ids") or ()),
                closure=FacetClosure(str(receipt.get("closure") or "")),
                required_output_form=RequiredOutputForm(
                    str(receipt.get("required_output_form") or "")
                ),
                expected_item_count=receipt.get("expected_item_count"),
                proven_items=proven_items,
                closure_status=ObligationClosureStatus(str(receipt.get("closure_status") or "")),
                source_enumeration_closure_receipt=source_closure,
                catalog=catalog,
                verifier=_COVERAGE_VERIFIER,
                failure_codes=tuple(
                    AnsweringFailureCode(str(item)) for item in receipt.get("failure_codes") or ()
                ),
            )
        )
    return tuple(proofs)


def _accepted_abstention_coverage(
    *,
    question: QuestionContract,
    catalog: WitnessCatalogReceipt,
    reason_code: str,
    repair_count: int,
) -> dict[str, Any]:
    """为不暴露事实正文的终态构造逐义务本地不足回执。"""

    return {
        "schema_version": "answer-coverage-decision.v4",
        "status": "accepted",
        "reason_code": reason_code,
        "required_count": len(question.required_obligation_ids),
        "covered_count": 0,
        "unresolved_obligation_ids": list(question.required_obligation_ids),
        "repair_count": max(0, repair_count),
        "catalog_universe_status": catalog.universe_status.value,
        "semantic_review_accepted": False,
        "proposal_reason_codes": [],
        "obligations": [
            {
                "obligation_id": obligation.obligation_id,
                "status": ProofVerdict.PROVEN.value,
                "outcome": ObligationOutcome.INSUFFICIENT_RETRIEVED_EVIDENCE.value,
                "supporting_claim_ids": [],
                "contradiction_witness_ids": [],
                "closure": obligation.closure.value,
                "required_output_form": obligation.required_output_form.value,
                "expected_item_count": obligation.expected_count,
                "proven_item_count": 0,
                "proven_items": [],
                "closure_status": ObligationClosureStatus.NOT_APPLICABLE.value,
                "failure_codes": [],
            }
            for obligation in question.obligations
        ],
    }


def _citation_bundle(
    state: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    render_port: AnswerRenderPort,
    safe_replay_cache: _SafeReplayCache | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[CitationFact]]:
    replay_cache = safe_replay_cache if safe_replay_cache is not None else {}
    evidence_ids = {str(item["evidence_id"]) for item in specs}
    evidences = render_port.revalidate_cited_evidence(state, evidence_ids)
    source_labels = _alias_source_labels(state)
    facts: list[CitationFact] = []
    citations: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(specs, start=1):
        evidence = evidences[str(spec["evidence_id"])]
        content = str(spec["content"])
        span_ref = EvidenceSpanRef.from_dict(spec.get("span_ref") or {})
        if span_ref.evidence_id != str(spec["evidence_id"]):
            raise ValueError("citation span and Evidence identity disagree")
        source_quote = _validate_render_span(
            state,
            span_ref=span_ref,
            canonical_text=str(evidence.get("content") or ""),
            safe_replay_cache=replay_cache,
        )
        if source_quote != str(spec.get("source_quote") or ""):
            raise ValueError("citation canonical quote changed after answer derivation")
        fact = CitationFact(
            answer_id=str(state["answer_id"]),
            evidence_id=str(spec["evidence_id"]),
            answer_span=spec["answer_span"],
            source_quote=source_quote,
            quote_hash=sha256_digest(source_quote),
            source_label=source_labels.get(str(spec["evidence_id"]), str(spec["label"])),
            validation_status="valid",
            reason_codes=[],
            ordinal=ordinal,
        )
        facts.append(fact)
        item = fact.model_dump(mode="json", exclude_none=True)
        item.update(
            {
                "claim_id": spec["claim_id"],
                "obligation_id": spec["obligation_id"],
                "witness_id": spec["witness_id"],
                "label": spec["label"],
                "claim_text": content,
                "support_score": 1.0,
                "support_method": {
                    SpanEquivalence.EXACT_UTF8: "ANSWER_PROOF_EXACT_WITNESS",
                    SpanEquivalence.NFKC_EQUIVALENT: "ANSWER_PROOF_NFKC_WITNESS",
                    SpanEquivalence.SECURITY_SAFE_DERIVED: ("ANSWER_PROOF_SECURITY_SAFE_WITNESS"),
                }[span_ref.equivalence],
                "preview_url": f"/api/v1/evidence/{spec['evidence_id']}",
            }
        )
        citations.append(item)
    sources = render_port.build_answer_sources(
        list(state.get("context_aliases") or []), evidence_ids
    )
    return citations, sources, facts


def _non_retrieval_proof(
    state: dict[str, Any],
    *,
    question: QuestionContract,
    plan: AnswerPlan,
) -> AnswerProof:
    answer = _notice_text(state, plan)
    span = AnswerSpan(start=0, end=len(answer))
    claim = plan.claims[0]
    claim_proof = ClaimProof.build(
        claim_id=claim.claim_id,
        obligation_id=None,
        status=ProofVerdict.PROVEN,
        method=RenderingRule.SYSTEM_NOTICE,
        witness_ids=(),
        answer_span=span,
        rendered_text_hash=sha256_digest(answer),
        relation=ProofRelation.NOT_APPLICABLE,
        verifier=_CLAIM_VERIFIER,
    )
    state["answer"] = answer
    return AnswerProof.build(
        answer_plan=plan,
        question_contract=question,
        catalog=None,
        answer_id=str(state["answer_id"]),
        answer_text=answer,
        overall_status=OverallProofStatus.NOT_APPLICABLE,
        grounding_status=ProofDimensionStatus.NOT_APPLICABLE,
        relevance_status=ProofDimensionStatus.NOT_APPLICABLE,
        completeness_status=CompletenessProofStatus.NOT_APPLICABLE,
        claim_proofs=(claim_proof,),
        obligation_proofs=(),
        verifier_versions=(_CLAIM_VERIFIER,),
    )


def _refusal_claim_proof(answer: str, plan: AnswerPlan) -> ClaimProof:
    """为拒答系统提示建立覆盖完整正文的非事实证明。"""

    claim = plan.claims[0]
    return ClaimProof.build(
        claim_id=claim.claim_id,
        obligation_id=None,
        status=ProofVerdict.PROVEN,
        method=RenderingRule.SYSTEM_NOTICE,
        witness_ids=(),
        answer_span=AnswerSpan(start=0, end=len(answer)),
        rendered_text_hash=sha256_digest(answer),
        relation=ProofRelation.NOT_APPLICABLE,
        verifier=_CLAIM_VERIFIER,
    )


def replay_answer_derivation(
    state: Mapping[str, Any],
    *,
    plan: AnswerPlan,
    bundle: Mapping[str, Any] | None,
) -> tuple[str, tuple[ClaimProof, ...], tuple[dict[str, Any], ...]]:
    """从冻结模板与 Witness 逐字重放可见正文、claim proof 和引用规格。"""

    if plan.decision in {AnswerDecision.ANSWER, AnswerDecision.PARTIAL}:
        if bundle is None:
            raise ValueError("factual answer replay requires a Witness catalog bundle")
        answer, claim_proofs, specs = _render_claims(
            state,
            plan,
            bundle,
            safe_replay_cache={},
        )
        return answer, tuple(claim_proofs), tuple(specs)
    answer = _notice_text(state, plan)
    return answer, (_refusal_claim_proof(answer, plan),), ()


def render_answer_and_proof(
    state: dict[str, Any],
    *,
    render_port: AnswerRenderPort,
) -> dict[str, Any]:
    """渲染最终答案和待提交 CitationFact，并构造唯一权威证明图。"""

    state["answer_id"] = str(state.get("answer_id") or render_port.new_answer_id())
    state["query_run_id"] = str(state.get("query_run_id") or render_port.new_query_run_id())
    question = QuestionContract.from_dict(state.get("question_contract") or {})
    plan = AnswerPlan.from_dict(state.get("answer_plan") or {})
    coverage = state.get("answer_coverage_decision")
    if not isinstance(coverage, Mapping):
        raise ValueError("answer coverage decision is missing")
    semantic_accepted = str(coverage.get("status") or "") == "accepted"

    if plan.decision in {AnswerDecision.CLARIFY, AnswerDecision.DIRECT_ANSWER}:
        proof = _non_retrieval_proof(state, question=question, plan=plan)
        citations: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        pending_facts: list[CitationFact] = []
    else:
        bundle = state.get("witness_catalog")
        if not isinstance(bundle, Mapping):
            raise ValueError("retrieval answer requires witness catalog bundle")
        catalog = WitnessCatalogReceipt.from_dict(bundle.get("receipt") or {})
        raw_closure = state.get("answer_review_closure_receipt")
        semantic_closure = (
            semantic_review_receipt_from_dict(raw_closure)
            if isinstance(raw_closure, Mapping) and raw_closure
            else None
        )
        if isinstance(semantic_closure, SemanticReviewClosureReceipt):
            semantic_closure.validate_against(answer_plan=plan, catalog=catalog)
        elif isinstance(semantic_closure, SemanticReviewTerminalReceipt):
            semantic_closure.validate_against(
                answer_plan=plan,
                reviewed_answer_plan=AnswerPlan.from_dict(state.get("answer_reviewed_plan") or {}),
                catalog=catalog,
            )
        if semantic_accepted and semantic_closure is None:
            raise ValueError("accepted retrieval plan lacks a bound semantic review receipt")
        witnesses = tuple(
            EvidenceWitness.from_dict(item)
            for item in bundle.get("witnesses") or []
            if isinstance(item, Mapping)
        )
        catalog.validate_witnesses(witnesses)
        safe_replay_cache: _SafeReplayCache = {}
        resolved_views = resolve_witness_prompt_views(
            state,
            bundle,
            safe_replay_cache=safe_replay_cache,
        )
        validated_source_closure_receipts = collect_source_enumeration_closure_receipts(
            resolved_views
        )
        witness_texts = {str(item["witness_id"]): str(item["content"]) for item in resolved_views}
        if plan.decision in {AnswerDecision.ANSWER, AnswerDecision.PARTIAL}:
            if not semantic_accepted:
                raise ValueError("factual AnswerPlan was not accepted by semantic review")
            answer, claim_proofs, specs = _render_claims(
                state,
                plan,
                bundle,
                resolved_views=resolved_views,
                safe_replay_cache=safe_replay_cache,
            )
            state["answer"] = answer
            secure_answer_state(state)
            if str(state.get("answer") or "") != answer:
                if not isinstance(semantic_closure, SemanticReviewClosureReceipt):
                    raise ValueError("output security transition lacks an accepted review closure")
                reviewed_plan = plan
                refusal_plan = refusal_answer_plan(
                    question_contract=question,
                    catalog=catalog,
                    prompt_contract_version=str(plan.prompt_contract_version),
                )
                terminal_receipt = SemanticReviewTerminalReceipt.build(
                    answer_plan=refusal_plan,
                    reviewed_answer_plan=reviewed_plan,
                    catalog=catalog,
                    decision="accept",
                    reason_code="OUTPUT_SECURITY_TRANSFORMED",
                    reviewed_closed_obligation_ids=(semantic_closure.closed_obligation_ids),
                    review_prompt_version=semantic_closure.review_prompt_version,
                    model_id=semantic_closure.model_id,
                    verifier_version=semantic_closure.verifier_version,
                    review_logical_call_id=semantic_closure.review_logical_call_id,
                    review_prompt_hash=semantic_closure.review_prompt_hash,
                    review_response_hash=semantic_closure.review_response_hash,
                    review_proposal_hash=semantic_closure.review_proposal_hash,
                )
                state["answer_reviewed_plan"] = reviewed_plan.to_dict()
                state["answer_plan"] = refusal_plan.to_dict()
                state["answer_review_closure_receipt"] = terminal_receipt.to_dict()
                state["answer_coverage_decision"] = _accepted_abstention_coverage(
                    question=question,
                    catalog=catalog,
                    reason_code="OUTPUT_SECURITY_TRANSFORMED",
                    repair_count=int(state.get("answer_repair_count") or 0),
                )
                state["citations"] = []
                state["sources"] = []
                state["pending_citation_facts"] = []
                return render_answer_and_proof(state, render_port=render_port)
            citations, sources, pending_facts = _citation_bundle(
                state,
                specs,
                render_port=render_port,
                safe_replay_cache=safe_replay_cache,
            )
            obligation_proofs = _obligation_proofs(
                question=question,
                catalog=catalog,
                coverage=coverage,
            )
            overall = (
                OverallProofStatus.ACCEPTED
                if plan.decision == AnswerDecision.ANSWER
                else OverallProofStatus.ACCEPTED_PARTIAL
            )
            completeness = (
                CompletenessProofStatus.COMPLETE
                if overall == OverallProofStatus.ACCEPTED
                else CompletenessProofStatus.INCOMPLETE
            )
            proof = AnswerProof.build(
                answer_plan=plan,
                question_contract=question,
                catalog=catalog,
                witnesses=witnesses,
                witness_texts=witness_texts,
                validated_source_enumeration_closure_receipts=(validated_source_closure_receipts),
                semantic_review_closure=semantic_closure,
                answer_id=str(state["answer_id"]),
                answer_text=answer,
                overall_status=overall,
                grounding_status=ProofDimensionStatus.PASSED,
                relevance_status=ProofDimensionStatus.PASSED,
                completeness_status=completeness,
                claim_proofs=tuple(claim_proofs),
                obligation_proofs=obligation_proofs,
                verifier_versions=(
                    _CLAIM_VERIFIER,
                    _SEMANTIC_VERIFIER,
                    _COVERAGE_VERIFIER,
                ),
            )
        else:
            answer = _notice_text(state, plan)
            state["answer"] = answer
            citations, sources = [], []
            pending_facts = []
            obligation_proofs = _obligation_proofs(
                question=question,
                catalog=catalog,
                coverage=coverage,
            )
            required = set(question.required_obligation_ids)
            required_proofs = tuple(
                item for item in obligation_proofs if item.obligation_id in required
            )
            complete_refusal = (
                bool(required)
                and semantic_accepted
                and all(
                    item.status == ProofVerdict.PROVEN
                    and item.outcome
                    in {ObligationOutcome.EVIDENCE_ABSENT, ObligationOutcome.CONFLICTING}
                    for item in required_proofs
                )
            )
            accepted_abstention = (
                bool(required)
                and semantic_accepted
                and all(
                    item.status == ProofVerdict.PROVEN
                    and item.outcome == ObligationOutcome.INSUFFICIENT_RETRIEVED_EVIDENCE
                    for item in required_proofs
                )
            )
            failures = ()
            overall = OverallProofStatus.ACCEPTED_REFUSAL
            completeness = CompletenessProofStatus.COMPLETE
            grounding = relevance = ProofDimensionStatus.NOT_APPLICABLE
            if accepted_abstention:
                overall = OverallProofStatus.ACCEPTED_ABSTENTION
                completeness = CompletenessProofStatus.INDETERMINATE
            elif not complete_refusal:
                overall = OverallProofStatus.INDETERMINATE
                completeness = CompletenessProofStatus.INDETERMINATE
                grounding = relevance = ProofDimensionStatus.INDETERMINATE
                failures = (
                    AnsweringFailure(
                        stage=FailureStage.ANSWER_PROOF,
                        code=AnsweringFailureCode.REFUSAL_NOT_AUTHORIZED,
                        retryability=Retryability.NEW_RETRIEVAL,
                    ),
                )
            proof = AnswerProof.build(
                answer_plan=plan,
                question_contract=question,
                catalog=catalog,
                witnesses=witnesses,
                witness_texts=witness_texts,
                validated_source_enumeration_closure_receipts=(validated_source_closure_receipts),
                semantic_review_closure=semantic_closure,
                answer_id=str(state["answer_id"]),
                answer_text=answer,
                overall_status=overall,
                grounding_status=grounding,
                relevance_status=relevance,
                completeness_status=completeness,
                claim_proofs=(_refusal_claim_proof(answer, plan),),
                obligation_proofs=obligation_proofs,
                verifier_versions=(
                    _CLAIM_VERIFIER,
                    _SEMANTIC_VERIFIER,
                    _COVERAGE_VERIFIER,
                ),
                failures=failures,
            )

    state["citations"] = citations
    state["sources"] = sources
    state["pending_citation_facts"] = [
        item.model_dump(mode="json", exclude_none=True) for item in pending_facts
    ]
    state["citation_validation"] = {
        "status": "valid" if citations else "not_applicable",
        "coverage": 1.0 if citations else 0.0,
        "citation_count": len(citations),
        "trusted_count": len(citations),
        "issues": [],
        "verifier_version": "answer-proof-verifier.v3",
    }
    state["answer_proof"] = proof.to_dict()
    state["answer_decision"] = {
        "action": _decision_projection(plan.decision),
        "reason_code": str(
            (state.get("answer_coverage_decision") or {}).get("reason_code")
            or "ANSWER_PROOF_FINALIZED"
        ),
        "repair_budget": max(
            0,
            MAX_REPAIR_ATTEMPTS - int(state.get("answer_repair_count") or 0),
        ),
    }
    supported_claims = len(citations)
    state["grounding"] = {
        "status": ("passed" if supported_claims else "not_applicable"),
        "claim_count": supported_claims,
        "supported_claim_count": supported_claims,
        "claim_coverage": 1.0 if supported_claims else 0.0,
        "unsupported_claims": [],
        "verifier_version": "answer-proof-verifier.v3",
    }
    state["evidence_assessment"] = {
        "status": (
            "SUFFICIENT"
            if plan.decision == AnswerDecision.ANSWER
            else "PARTIAL"
            if plan.decision == AnswerDecision.PARTIAL
            else "NONE"
        ),
        "query_coverage": 1.0 if plan.decision == AnswerDecision.ANSWER else 0.0,
        "candidate_count": len((state.get("witness_catalog") or {}).get("witnesses") or []),
        "packed_evidence_count": len(state.get("context_aliases") or []),
        "reasons": [state["answer_decision"]["reason_code"]],
        "assessor_version": "answer-proof-verifier.v3",
    }
    state["image_urls"] = render_port.cited_image_urls(state)
    envelope = render_port.build_answer_envelope(state)
    state["answer_envelope"] = envelope.model_dump(mode="json")
    state["answer_render_receipt"] = {
        "schema_version": "answer-render-receipt.v3",
        "status": "rendered",
        "answer_id": str(state["answer_id"]),
        "query_run_id": str(state["query_run_id"]),
        "answer_sha256": sha256_digest(str(state.get("answer") or "")),
        "proof_id": proof.proof_id,
        "proof_hash": proof.proof_hash,
        "claim_count": len(proof.claim_proofs),
        "citation_count": len(citations),
        "citation_fact_bundle_hash": citation_fact_bundle_hash(pending_facts),
    }
    return state


__all__ = ["render_answer_and_proof", "replay_answer_derivation"]
