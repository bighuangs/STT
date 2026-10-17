"""从 AnswerProof 与确定性渲染结果投影公开 AnswerEnvelope。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.answer import (
    AnswerCitation,
    AnswerClaim,
    AnswerConfidence,
    AnswerEnvelope,
    AnswerGrounding,
    AnswerSource,
    AnswerUsage,
    AnswerVersions,
)
from app.shared.config.settings_config import settings


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _usage(value: Any) -> AnswerUsage:
    raw = _mapping(value)
    status = str(raw.get("status") or "unavailable")
    if status not in {"available", "unavailable", "estimated"}:
        status = "unavailable"
    return AnswerUsage(
        status=status,  # type: ignore[arg-type]
        input_tokens=raw.get("input_tokens"),
        output_tokens=raw.get("output_tokens"),
        total_tokens=raw.get("total_tokens"),
        cost=raw.get("cost"),
        currency=raw.get("currency"),
        reason=(
            str(raw.get("reason") or "provider did not return usage")
            if status == "unavailable"
            else raw.get("reason")
        ),
    )


def _claims_and_citations(
    answer: str,
    raw_citations: list[dict[str, Any]],
) -> tuple[tuple[AnswerClaim, ...], tuple[AnswerCitation, ...]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for citation in raw_citations:
        span = _mapping(citation.get("answer_span"))
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(answer)):
            continue
        grouped.setdefault((start, end), []).append(citation)

    claims: list[AnswerClaim] = []
    citations: list[AnswerCitation] = []
    for ordinal, ((start, end), group) in enumerate(sorted(grouped.items()), start=1):
        claim_id = f"clm_{ordinal}"
        citation_ids: list[str] = []
        supported = False
        for raw in group:
            citation_id = f"cit_{len(citations) + 1}"
            citation_ids.append(citation_id)
            validation = str(raw.get("validation_status") or "invalid")
            if validation not in {"valid", "degraded", "invalid"}:
                validation = "invalid"
            supported = supported or validation in {"valid", "degraded"}
            citations.append(
                AnswerCitation(
                    citation_id=citation_id,
                    claim_id=claim_id,
                    evidence_id=str(raw.get("evidence_id") or ""),
                    answer_start=start,
                    answer_end=end,
                    validation=validation,  # type: ignore[arg-type]
                    relation="SUPPORTS",
                    verifier_score=1.0 if validation == "valid" else 0.0,
                    verifier_version="answer-proof-verifier.v3",
                    source_quote_hash=(
                        str(raw.get("quote_hash")) if raw.get("quote_hash") else None
                    ),
                    reason_codes=tuple(
                        dict.fromkeys(str(code) for code in raw.get("reason_codes") or [] if code)
                    ),
                )
            )
        claims.append(
            AnswerClaim(
                claim_id=claim_id,
                text=answer[start:end],
                answer_start=start,
                answer_end=end,
                citation_ids=tuple(citation_ids),
                supported=supported,
            )
        )
    return tuple(claims), tuple(citations)


def _sources(raw_sources: list[dict[str, Any]], cited_ids: set[str]) -> tuple[AnswerSource, ...]:
    result: list[AnswerSource] = []
    for raw in raw_sources:
        if str(raw.get("evidence_id") or "") not in cited_ids:
            continue
        candidate = {
            key: raw.get(key)
            for key in AnswerSource.model_fields
            if key in raw and raw.get(key) is not None
        }
        result.append(AnswerSource.model_validate(candidate))
    return tuple(result)


def _finish_reason(action: str, reason_code: str) -> str:
    if action == "DIRECT_ANSWER":
        return "direct_answer"
    if action == "CLARIFY":
        return "clarification_required"
    if action == "REFUSE":
        if reason_code in {"NO_EVIDENCE", "PROVEN_ABSENT"}:
            return "no_evidence"
        if reason_code == "CONFLICTING_EVIDENCE":
            return "conflicting_evidence"
        return "insufficient_evidence" if "EVIDENCE" in reason_code else "policy"
    if action == "PARTIAL_ANSWER":
        return "insufficient_evidence"
    return "stop"


def _index_versions(state: Mapping[str, Any], sources: tuple[AnswerSource, ...]) -> tuple[str, ...]:
    identifiers = [
        str(scope.get("index_version_id") or "")
        for scope in state.get("vector_read_scopes") or []
        if isinstance(scope, Mapping)
    ]
    identifiers.extend(str(source.index_version_id or "") for source in sources)
    return tuple(dict.fromkeys(value for value in identifiers if value))


def build_proof_answer_envelope(state: dict[str, Any]) -> AnswerEnvelope:
    """只消费最终 proof/render 产物，不再从自由正文反推 claim 图。"""

    answer = str(state.get("answer") or "")
    decision = _mapping(state.get("answer_decision"))
    proof = _mapping(state.get("answer_proof"))
    action = str(decision.get("action") or "REFUSE")
    if action not in {"ANSWER", "PARTIAL_ANSWER", "CLARIFY", "REFUSE", "DIRECT_ANSWER"}:
        action = "REFUSE"
    raw_citations = [
        dict(item) for item in state.get("citations") or [] if isinstance(item, Mapping)
    ]
    claims, citations = _claims_and_citations(answer, raw_citations)
    cited_ids = {citation.evidence_id for citation in citations}
    sources = _sources(
        [dict(item) for item in state.get("sources") or [] if isinstance(item, Mapping)],
        cited_ids,
    )

    supported = sum(claim.supported for claim in claims)
    coverage = supported / max(1, len(claims))
    strict_eligible = bool(proof.get("strict_eligible", False))
    if action in {"CLARIFY", "REFUSE", "DIRECT_ANSWER"}:
        grounding_status = "not_applicable"
    elif strict_eligible and claims and supported == len(claims):
        grounding_status = "passed"
    elif supported:
        grounding_status = "degraded"
    else:
        grounding_status = "failed"
        if action == "ANSWER":
            action = "PARTIAL_ANSWER"
    grounding = AnswerGrounding(
        status=grounding_status,  # type: ignore[arg-type]
        claim_count=len(claims),
        supported_claim_count=supported,
        claim_coverage=round(coverage, 4),
        unsupported_claims=tuple(claim.text for claim in claims if not claim.supported),
        verifier_version="answer-proof-verifier.v3",
    )

    degradation_reasons: tuple[str, ...] = ()
    if action == "PARTIAL_ANSWER":
        degradation_reasons = (str(decision.get("reason_code") or "PARTIAL_EVIDENCE"),)
    prompt_receipts = [
        item for item in state.get("answer_prompt_receipts") or [] if isinstance(item, Mapping)
    ]
    prompt_hash = str(prompt_receipts[-1].get("prompt_sha256") or "") if prompt_receipts else ""
    versions = AnswerVersions(
        prompt_hash=prompt_hash or None,
        model_id=str(state.get("llm_model_id") or settings.llm_default_model or "unknown"),
        config_version=str(state.get("config_version") or "cfg_unknown"),
        index_version_ids=_index_versions(state, sources),
        retrieval_profile_id=str(state.get("retrieval_profile_id") or ""),
        retrieval_profile_version=int(state.get("retrieval_profile_version") or 0),
        query_plan_version=str(
            _mapping(state.get("query_plan")).get("schema_version") or "query-plan.v1"
        ),
        fusion_version=str(
            _mapping(state.get("fusion_result")).get("schema_version") or "fusion.v1"
        ),
        context_pack_version=str(
            _mapping(state.get("context_pack")).get("schema_version") or "context-pack.v1"
        ),
        assessor_version="answer-proof-verifier.v3",
    )
    confidence = AnswerConfidence(
        level=("high" if strict_eligible else "medium" if supported else "low"),
        score=round(coverage, 4),
        reasons=("ANSWER_PROOF_STRICT_ELIGIBLE",)
        if strict_eligible
        else ("ANSWER_PROOF_DEGRADED",),
        method="answer-proof.v3",
    )
    return AnswerEnvelope(
        answer_id=str(state["answer_id"]),
        query_run_id=str(state["query_run_id"]),
        session_id=str(state.get("conversation_id") or state["session_id"]),
        message_id=str(state.get("message_id") or state["answer_id"]),
        answer=answer,
        decision=action,  # type: ignore[arg-type]
        claims=claims,
        citations=citations,
        sources=sources,
        confidence=confidence,
        grounding=grounding,
        degraded=bool(degradation_reasons),
        degradation_reasons=degradation_reasons,
        finish_reason=_finish_reason(action, str(decision.get("reason_code") or "")),  # type: ignore[arg-type]
        versions=versions,
        usage=_usage(state.get("usage")),
        trace_id=str(state.get("trace_id") or state["query_run_id"]),
    )
