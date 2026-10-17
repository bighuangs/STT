"""把 TODO-30 的可信终态固化为 ``AnswerEnvelope v1``（TODO-11）。"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import re
from typing import Any
from uuid import uuid4

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
from app.domain.evidence import sha256_digest
from app.shared.config.settings_config import settings


_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def new_query_run_id() -> str:
    return f"qry_{uuid4().hex}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _span(citation: dict[str, Any]) -> tuple[int, int] | None:
    raw = _dict(citation.get("answer_span"))
    try:
        start, end = int(raw["start"]), int(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if 0 <= start < end else None


def _build_claims_and_citations(
    answer: str,
    raw_citations: list[dict[str, Any]],
    allowed_evidence_ids: set[str],
) -> tuple[list[AnswerClaim], list[AnswerCitation]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for citation in raw_citations:
        span = _span(citation)
        evidence_id = str(citation.get("evidence_id") or "")
        if span is None or span[1] > len(answer) or evidence_id not in allowed_evidence_ids:
            continue
        grouped.setdefault(span, []).append(citation)

    claims: list[AnswerClaim] = []
    citations: list[AnswerCitation] = []
    covered_spans: list[tuple[int, int]] = []
    for claim_index, ((start, end), group) in enumerate(sorted(grouped.items()), start=1):
        claim_id = f"clm_{claim_index}"
        citation_ids: list[str] = []
        trusted = False
        for citation in group:
            citation_id = f"cit_{len(citations) + 1}"
            citation_ids.append(citation_id)
            validation = str(citation.get("validation_status") or "invalid")
            if validation not in {"valid", "degraded", "invalid"}:
                validation = "invalid"
            trusted = trusted or validation in {"valid", "degraded"}
            citations.append(
                AnswerCitation(
                    citation_id=citation_id,
                    claim_id=claim_id,
                    evidence_id=str(citation["evidence_id"]),
                    answer_start=start,
                    answer_end=end,
                    validation=validation,  # type: ignore[arg-type]
                    relation="SUPPORTS",
                    verifier_score=max(0.0, min(1.0, float(citation.get("support_score") or 0.0))),
                    source_quote_hash=(
                        str(citation.get("quote_hash")) if citation.get("quote_hash") else None
                    ),
                    reason_codes=tuple(
                        dict.fromkeys(
                            str(item) for item in (citation.get("reason_codes") or []) if item
                        )
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
                supported=trusted,
            )
        )
        covered_spans.append((start, end))

    # 生成后校验识别到的无引用句也进入 claim 图；调用方不必从中文回答反推
    # “哪些句子没有支持”。短礼貌语/标题不提升为 factual claim。
    for match in _SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        start, end = match.start(), match.end()
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        if end <= start or len(re.sub(r"\W+", "", sentence, flags=re.UNICODE)) < 6:
            continue
        if any(left < end and right > start for left, right in covered_spans):
            continue
        claims.append(
            AnswerClaim(
                claim_id=f"clm_{len(claims) + 1}",
                text=answer[start:end],
                answer_start=start,
                answer_end=end,
                citation_ids=(),
                supported=False,
                claim_type=(
                    "uncertainty"
                    if any(token in sentence for token in ("不足", "无法", "不确定", "请补充"))
                    else "factual"
                ),
            )
        )
    return claims, citations


def _source_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in state.get("evidences") or []:
        evidence = _dict(raw)
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id:
            result[evidence_id] = evidence
    for document in state.get("reranked_docs") or []:
        evidence = _dict(_dict(document).get("evidence"))
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id:
            result.setdefault(evidence_id, evidence)
    return result


def _build_sources(state: dict[str, Any], cited_evidence_ids: set[str]) -> list[AnswerSource]:
    evidences = _source_map(state)
    display_sources = {
        str(item.get("evidence_id") or ""): _dict(item)
        for item in state.get("sources") or []
        if isinstance(item, dict)
    }
    result: list[AnswerSource] = []
    for evidence_id in sorted(cited_evidence_ids):
        evidence = evidences.get(evidence_id)
        if not evidence:
            continue
        source = _dict(evidence.get("source"))
        display = display_sources.get(evidence_id, {})
        source_type = str(source.get("source_type") or "local")
        result.append(
            AnswerSource(
                evidence_id=evidence_id,
                source_type=source_type,  # type: ignore[arg-type]
                source_label=str(
                    display.get("source_label") or evidence.get("source_label") or evidence_id
                ),
                knowledge_base_id=source.get("kb_id"),
                document_id=source.get("document_id"),
                document_version_id=source.get("document_version_id"),
                index_version_id=source.get("index_version_id"),
                chunk_id=source.get("chunk_id"),
                url=source.get("source_url"),
                content_hash=str(evidence.get("content_hash") or ""),
                location_label=str(display.get("location_label") or "位置不可用"),
                provenance_spans=tuple(evidence.get("provenance_spans") or []),
                preview_url=str(display.get("preview_url") or f"/api/v1/evidence/{evidence_id}"),
                modality=str(evidence.get("modality") or "text"),  # type: ignore[arg-type]
                asset=_dict(evidence.get("asset")) or None,
                derivation=_dict(evidence.get("derivation")) or None,
            )
        )
    return result


def _finish_reason(decision: dict[str, Any]) -> str:
    action = str(decision.get("action") or "REFUSE")
    reason = str(decision.get("reason_code") or "")
    if action == "DIRECT_ANSWER":
        return "direct_answer"
    if action == "CLARIFY":
        return "clarification_required"
    if action == "REFUSE":
        if reason in {"NO_EVIDENCE", "UNSUPPORTED_GENERATION"}:
            return "no_evidence" if reason == "NO_EVIDENCE" else "insufficient_evidence"
        return "policy"
    if reason == "CONFLICTING_EVIDENCE":
        return "conflicting_evidence"
    if reason == "STALE_EVIDENCE":
        return "stale_evidence"
    if action == "PARTIAL_ANSWER":
        return "insufficient_evidence"
    return "stop"


def _confidence(
    decision: dict[str, Any],
    grounding: dict[str, Any],
    assessment: dict[str, Any],
) -> AnswerConfidence:
    action = str(decision.get("action") or "REFUSE")
    grounding_status = str(grounding.get("status") or "not_applicable")
    coverage = float(grounding.get("claim_coverage") or 0.0)
    if action == "ANSWER" and grounding_status == "passed" and coverage >= 0.8:
        return AnswerConfidence(
            level="high",
            score=round(coverage, 4),
            reasons=("TRUSTED_CLAIM_COVERAGE_HIGH",),
        )
    if action in {"ANSWER", "PARTIAL_ANSWER"} and grounding_status in {
        "passed",
        "degraded",
    }:
        return AnswerConfidence(
            level="medium" if coverage >= 0.5 else "low",
            score=round(coverage, 4),
            reasons=tuple(assessment.get("reasons") or ("PARTIAL_GROUNDING",)),
        )
    if action == "DIRECT_ANSWER":
        return AnswerConfidence(
            level="unknown",
            reasons=("DIRECT_NON_RETRIEVAL_RESPONSE",),
        )
    return AnswerConfidence(
        level="low",
        score=0.0,
        reasons=tuple(assessment.get("reasons") or (decision.get("reason_code") or "NO_EVIDENCE",)),
    )


def _usage(raw: Any) -> AnswerUsage:
    value = _dict(raw)
    status = str(value.get("status") or "unavailable")
    if status not in {"available", "unavailable", "estimated"}:
        status = "unavailable"
    return AnswerUsage(
        status=status,  # type: ignore[arg-type]
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
        total_tokens=value.get("total_tokens"),
        cost=value.get("cost"),
        currency=value.get("currency"),
        reason=(
            str(value.get("reason") or "provider did not return usage")
            if status == "unavailable"
            else value.get("reason")
        ),
    )


def _version(value: Any, default: str) -> str:
    if isinstance(value, dict):
        raw = value.get("schema_version") or value.get("version")
    else:
        raw = value
    return str(raw or default)


def _index_version_ids(
    state: dict[str, Any],
    sources: list[AnswerSource],
) -> tuple[str, ...]:
    """记录查询真正冻结的索引读集，而不只记录最终被引用的来源。

    ``vector_read_scopes`` 是 QueryRun 接受时由服务端冻结并在检索前再次校验的
    物理读集。回答可能因模型错标引用而被拒绝，但这不会改变本次检索实际读取
    过哪个索引版本。仅从 ``sources`` 推导会让这类可审计终态错误地得到空版本。
    旧入口没有冻结读集时，继续以可信来源作为兼容回退。
    """

    identifiers: list[str] = []
    for raw_scope in state.get("vector_read_scopes") or []:
        scope = _dict(raw_scope)
        identifier = str(scope.get("index_version_id") or "")
        if identifier:
            identifiers.append(identifier)
    snapshot = _dict(state.get("index_snapshot"))
    for key in ("active_indexes", "vector_read_scopes"):
        for raw_scope in snapshot.get(key) or []:
            scope = _dict(raw_scope)
            identifier = str(scope.get("index_version_id") or "")
            if identifier:
                identifiers.append(identifier)
    identifiers.extend(
        str(item.index_version_id) for item in sources if item.index_version_id
    )
    return tuple(dict.fromkeys(identifiers))


def _degradation_reasons(
    state: dict[str, Any],
    decision: dict[str, Any],
    grounding: dict[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if decision.get("action") == "PARTIAL_ANSWER":
        reasons.append(str(decision.get("reason_code") or "PARTIAL_ANSWER"))
    if grounding.get("status") in {"degraded", "failed"}:
        reasons.append(f"GROUNDING_{str(grounding['status']).upper()}")
    assessment = _dict(state.get("evidence_assessment"))
    trust = _dict(assessment.get("trust_summary"))
    if trust.get("failed_routes"):
        reasons.append("RETRIEVAL_ROUTE_DEGRADED")
    return tuple(dict.fromkeys(reasons))


def build_answer_envelope(state: dict[str, Any]) -> AnswerEnvelope:
    """从已完成 citation/grounding 的状态构建并再次验证终态图。"""

    state["query_run_id"] = str(state.get("query_run_id") or new_query_run_id())
    answer = str(state.get("answer") or "")
    decision = _dict(state.get("answer_decision")) or {
        "action": "REFUSE",
        "reason_code": "NO_EVIDENCE",
    }
    grounding_raw = _dict(state.get("grounding")) or {
        "status": "not_applicable",
        "unsupported_claims": [],
    }

    evidence_map = _source_map(state)
    raw_citations = [
        _dict(item) for item in (state.get("citations") or []) if isinstance(item, dict)
    ]
    cited_ids = {
        str(item.get("evidence_id") or "")
        for item in raw_citations
        if str(item.get("evidence_id") or "") in evidence_map
    }
    sources = _build_sources(state, cited_ids)
    allowed_ids = {item.evidence_id for item in sources}
    claims, citations = _build_claims_and_citations(answer, raw_citations, allowed_ids)

    trusted_count = sum(citation.validation in {"valid", "degraded"} for citation in citations)
    supported_claims = sum(claim.supported for claim in claims)
    claim_count = len(claims)
    computed_coverage = supported_claims / max(1, claim_count)
    grounding_status = str(grounding_raw.get("status") or "not_applicable")
    if decision.get("action") in {"CLARIFY", "REFUSE", "DIRECT_ANSWER"}:
        grounding_status = "not_applicable"
    elif trusted_count == 0:
        grounding_status = "failed"
    elif computed_coverage < 1.0 or grounding_status == "degraded":
        grounding_status = "degraded"
    else:
        grounding_status = "passed"
    grounding = AnswerGrounding(
        status=grounding_status,  # type: ignore[arg-type]
        claim_count=claim_count,
        supported_claim_count=supported_claims,
        claim_coverage=round(computed_coverage, 4),
        unsupported_claims=tuple(claim.text for claim in claims if not claim.supported),
    )

    index_ids = _index_version_ids(state, sources)
    prompt = str(state.get("prompt") or "")
    versions = AnswerVersions(
        prompt_hash=sha256_digest(prompt) if prompt else None,
        # 以本次运行实际解析的冻结模型为准；旧入口没有该字段时兼容进程默认值。
        model_id=str(state.get("llm_model_id") or settings.llm_default_model or "unknown"),
        config_version=str(state.get("config_version") or "cfg_unknown"),
        index_version_ids=index_ids,
        retrieval_profile_id=str(state.get("retrieval_profile_id") or ""),
        retrieval_profile_version=int(state.get("retrieval_profile_version") or 0),
        query_plan_version=_version(state.get("query_plan"), "query-plan.v1"),
        fusion_version=_version(state.get("fusion_result"), "fusion.v1"),
        context_pack_version=_version(state.get("context_pack"), "context-pack.v1"),
        assessor_version=str(
            _dict(state.get("evidence_assessment")).get("assessor_version")
            or "evidence-assessor.v1"
        ),
    )
    degradation_reasons = _degradation_reasons(state, decision, grounding_raw)
    envelope = AnswerEnvelope(
        answer_id=str(state["answer_id"]),
        query_run_id=state["query_run_id"],
        session_id=str(state.get("conversation_id") or state["session_id"]),
        message_id=str(state.get("message_id") or state["answer_id"]),
        answer=answer,
        decision=str(decision.get("action") or "REFUSE"),  # type: ignore[arg-type]
        claims=tuple(claims),
        citations=tuple(citations),
        sources=tuple(sources),
        confidence=_confidence(
            decision,
            grounding.model_dump(mode="json"),
            _dict(state.get("evidence_assessment")),
        ),
        grounding=grounding,
        degraded=bool(degradation_reasons),
        degradation_reasons=degradation_reasons,
        finish_reason=_finish_reason(decision),  # type: ignore[arg-type]
        versions=versions,
        usage=_usage(state.get("usage")),
        trace_id=str(state.get("trace_id") or state["query_run_id"]),
    )
    return envelope


def legacy_envelope(
    *,
    session_id: str,
    answer: str,
    answer_id: str | None = None,
    created_at: datetime | None = None,
    seed: str = "",
) -> AnswerEnvelope | None:
    """把旧 text-only assistant history 映射为显式降级的只读 v1 对象。"""

    if not answer:
        return None
    digest = sha256(f"{session_id}|{seed}|{answer}".encode("utf-8")).hexdigest()
    stable_answer_id = (
        answer_id
        if answer_id and re.fullmatch(r"ans_[0-9a-f]{32}", answer_id)
        else f"ans_{digest[:32]}"
    )
    return AnswerEnvelope(
        answer_id=stable_answer_id,
        query_run_id=f"qry_{digest[32:64]}",
        session_id=session_id or "legacy",
        message_id=stable_answer_id,
        answer=answer,
        decision="ANSWER",
        confidence=AnswerConfidence(level="unknown", reasons=("LEGACY_TEXT_ONLY",)),
        grounding=AnswerGrounding(
            status="not_applicable",
            unsupported_claims=(answer,),
        ),
        degraded=True,
        degradation_reasons=("LEGACY_TEXT_ONLY",),
        finish_reason="stop",
        versions=AnswerVersions(),
        usage=AnswerUsage(status="unavailable", reason="legacy history has no usage"),
        trace_id=f"qry_{digest[32:64]}",
        created_at=created_at or datetime.now(UTC),
    )


__all__ = [
    "build_answer_envelope",
    "legacy_envelope",
    "new_query_run_id",
]
