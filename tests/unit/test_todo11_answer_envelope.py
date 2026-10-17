from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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
from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
    sha256_digest,
)
from app.rag.query.answer_envelope_service import (
    build_answer_envelope,
    legacy_envelope,
)


ANSWER_ID = "ans_" + "a" * 32
QUERY_RUN_ID = "qry_" + "b" * 32
EVIDENCE_ID = "ev_" + "c" * 40
OTHER_EVIDENCE_ID = "ev_" + "d" * 40
CONTENT_HASH = "sha256:" + "e" * 64
QUOTE_HASH = "sha256:" + "f" * 64


def _source(evidence_id: str = EVIDENCE_ID) -> AnswerSource:
    return AnswerSource(
        evidence_id=evidence_id,
        source_type="local",
        source_label="hak180产品安全手册.pdf",
        knowledge_base_id="kb_hak180",
        document_id="doc_hak180",
        document_version_id="dv_hak180_v1",
        index_version_id="idx_hak180_v1",
        chunk_id=f"chunk-{evidence_id[-4:]}",
        content_hash=CONTENT_HASH,
        location_label="第1页",
        provenance_spans=(
            {
                "block_id": "block-1",
                "availability": "exact",
                "coordinate_space": "normalized_page",
                "page_no": 1,
                "bbox": [0.1, 0.1, 0.8, 0.3],
            },
        ),
        preview_url=f"/api/v1/evidence/{evidence_id}",
    )


def _claim(
    *,
    claim_id: str = "clm_1",
    text: str = "设备必须可靠接地。",
    start: int = 0,
    end: int | None = None,
    citation_ids: tuple[str, ...] = ("cit_1",),
    supported: bool = True,
) -> AnswerClaim:
    return AnswerClaim(
        claim_id=claim_id,
        text=text,
        answer_start=start,
        answer_end=len(text) if end is None else end,
        citation_ids=citation_ids,
        supported=supported,
    )


def _citation(
    *,
    citation_id: str = "cit_1",
    claim_id: str = "clm_1",
    evidence_id: str = EVIDENCE_ID,
    start: int = 0,
    end: int = len("设备必须可靠接地。"),
    validation: str = "valid",
) -> AnswerCitation:
    return AnswerCitation(
        citation_id=citation_id,
        claim_id=claim_id,
        evidence_id=evidence_id,
        answer_start=start,
        answer_end=end,
        validation=validation,
        verifier_score=0.98,
        source_quote_hash=QUOTE_HASH,
        reason_codes=("EXACT_CONTENT_MATCH",),
    )


def _envelope(**overrides) -> AnswerEnvelope:
    value = {
        "answer_id": ANSWER_ID,
        "query_run_id": QUERY_RUN_ID,
        "session_id": "sess-todo11",
        "message_id": "msg-todo11",
        "answer": "设备必须可靠接地。",
        "decision": "ANSWER",
        "claims": (_claim(),),
        "citations": (_citation(),),
        "sources": (_source(),),
        "confidence": AnswerConfidence(
            level="high",
            score=1.0,
            reasons=("TRUSTED_CLAIM_COVERAGE_HIGH",),
        ),
        "grounding": AnswerGrounding(
            status="passed",
            claim_count=1,
            supported_claim_count=1,
            claim_coverage=1.0,
        ),
        "degraded": False,
        "degradation_reasons": (),
        "finish_reason": "stop",
        "versions": AnswerVersions(
            prompt_hash="sha256:" + "1" * 64,
            model_id="test-model",
            config_version="cfg-test",
            index_version_ids=("idx_hak180_v1",),
            retrieval_profile_id="rpf-default",
            retrieval_profile_version=1,
        ),
        "usage": AnswerUsage(
            status="available",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        ),
        "trace_id": "trace-todo11",
        "created_at": datetime(2026, 7, 24, 18, 0, tzinfo=UTC),
    }
    value.update(overrides)
    return AnswerEnvelope.model_validate(value)


def _local_evidence(content: str = "设备启动前必须可靠接地。") -> Evidence:
    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id="tenant_default",
            kb_id="kb_hak180",
            document_id="doc_hak180",
            document_version_id="dv_hak180_v1",
            index_version_id="idx_hak180_v1",
            chunk_id="chunk-grounding",
            source_block_ids=["block-grounding"],
        ),
        content=content,
        provenance_spans=[
            EvidenceLocationSpan(
                block_id="block-grounding",
                availability="exact",
                coordinate_space="normalized_page",
                page_no=1,
                bbox=(0.1, 0.1, 0.8, 0.3),
            )
        ],
        source_label="hak180产品安全手册.pdf",
    )


def _normal_state() -> tuple[dict, Evidence]:
    answer = "设备启动前必须可靠接地。"
    evidence = _local_evidence(answer)
    state = {
        "answer_id": ANSWER_ID,
        "query_run_id": QUERY_RUN_ID,
        "session_id": "sess-todo11",
        "message_id": "msg-todo11",
        "answer": answer,
        "answer_decision": {"action": "ANSWER", "reason_code": "SUFFICIENT_EVIDENCE"},
        "grounding": {
            "status": "passed",
            "claim_coverage": 1.0,
            "unsupported_claims": [],
        },
        # prepare_query_context leaves only ContextPack Evidence in these fields.
        "evidences": [evidence.model_dump(mode="json", exclude_none=True)],
        "reranked_docs": [
            {
                "evidence_id": evidence.evidence_id,
                "evidence": evidence.model_dump(mode="json", exclude_none=True),
            }
        ],
        "citations": [
            {
                "evidence_id": evidence.evidence_id,
                "answer_span": {"start": 0, "end": len(answer)},
                "validation_status": "valid",
                "support_score": 0.99,
                "quote_hash": sha256_digest(answer),
                "reason_codes": ["EXACT_CONTENT_MATCH", "EXACT_CONTENT_MATCH"],
            }
        ],
        "sources": [
            {
                "evidence_id": evidence.evidence_id,
                "source_label": "HAK180 安全手册",
                "location_label": "第1页",
                "preview_url": f"/api/v1/evidence/{evidence.evidence_id}",
            }
        ],
        "prompt": "system + evidence + question",
        "config_version": "cfg-todo11",
        "retrieval_profile_id": "rpf-default",
        "retrieval_profile_version": 3,
        "query_plan": {"schema_version": "query-plan.v1"},
        "fusion_result": {"schema_version": "fusion.v1"},
        "context_pack": {"schema_version": "context-pack.v1"},
        "evidence_assessment": {
            "assessor_version": "evidence-assessor.v1",
            "reasons": [],
            "trust_summary": {"failed_routes": []},
        },
        "usage": {
            "status": "available",
            "input_tokens": 400,
            "output_tokens": 40,
            "total_tokens": 440,
            "cost": 0.01,
            "currency": "CNY",
        },
        "trace_id": "trace-todo11",
    }
    return state, evidence


def test_answer_schema_is_closed_versioned_and_machine_readable() -> None:
    schema = AnswerEnvelope.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "answer.v1"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {
        "answer_id",
        "query_run_id",
        "session_id",
        "message_id",
        "answer",
        "decision",
        "confidence",
        "grounding",
        "finish_reason",
        "versions",
        "usage",
        "trace_id",
    }

    payload = _envelope().model_dump(mode="json")
    payload["unexpected_internal_prompt"] = "secret"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnswerEnvelope.model_validate(payload)

    payload = _envelope().model_dump(mode="json")
    payload["decision"] = "MODEL_DECIDES"
    with pytest.raises(ValidationError, match="Input should be"):
        AnswerEnvelope.model_validate(payload)

    payload = _envelope().model_dump(mode="json")
    payload["created_at"] = "2026-07-24T18:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        AnswerEnvelope.model_validate(payload)


def test_domain_rejects_unknown_sources_invalid_claim_text_and_supported_mismatch() -> None:
    with pytest.raises(ValidationError, match="outside source whitelist"):
        _envelope(citations=(_citation(evidence_id=OTHER_EVIDENCE_ID),))

    with pytest.raises(ValidationError, match="claim text must equal"):
        _envelope(claims=(_claim(text="篡改后的 claim", end=len("设备必须可靠接地。")),))

    with pytest.raises(ValidationError, match="supported flag disagrees"):
        _envelope(claims=(_claim(supported=False),))


def test_domain_rejects_cross_claim_citation_wiring() -> None:
    answer = "设备必须可靠接地。启动前检查急停按钮。"
    split = len("设备必须可靠接地。")
    claims = (
        _claim(
            claim_id="clm_1",
            text=answer[:split],
            start=0,
            end=split,
            # cit_2 belongs to clm_2 and must not be accepted by clm_1.
            citation_ids=("cit_2",),
        ),
        _claim(
            claim_id="clm_2",
            text=answer[split:],
            start=split,
            end=len(answer),
            citation_ids=("cit_1",),
        ),
    )
    citations = (
        _citation(
            citation_id="cit_1",
            claim_id="clm_1",
            start=0,
            end=split,
        ),
        _citation(
            citation_id="cit_2",
            claim_id="clm_2",
            start=split,
            end=len(answer),
        ),
    )
    with pytest.raises(ValidationError, match="citation.*claim"):
        _envelope(
            answer=answer,
            claims=claims,
            citations=citations,
            grounding=AnswerGrounding(
                status="passed",
                claim_count=2,
                supported_claim_count=2,
                claim_coverage=1.0,
            ),
        )


def test_domain_rejects_citation_span_outside_its_claim() -> None:
    answer = "设备必须可靠接地。启动前检查急停按钮。"
    split = len("设备必须可靠接地。")
    with pytest.raises(ValidationError, match="citation.*span"):
        _envelope(
            answer=answer,
            claims=(
                _claim(
                    text=answer[:split],
                    start=0,
                    end=split,
                ),
            ),
            citations=(
                # Citation claims to support clm_1 but points at the second sentence.
                _citation(start=split, end=len(answer)),
            ),
        )


def test_normal_envelope_builds_claim_citation_source_and_version_graph() -> None:
    state, evidence = _normal_state()
    envelope = build_answer_envelope(state)

    assert envelope.schema_version == "answer.v1"
    assert envelope.answer_id == ANSWER_ID
    assert envelope.query_run_id == QUERY_RUN_ID
    assert envelope.decision == "ANSWER"
    assert envelope.finish_reason == "stop"
    assert envelope.grounding.status == "passed"
    assert envelope.grounding.claim_count == 1
    assert envelope.grounding.supported_claim_count == 1
    assert envelope.grounding.claim_coverage == 1.0
    assert envelope.confidence.level == "high"
    assert envelope.confidence.score == 1.0

    claim = envelope.claims[0]
    citation = envelope.citations[0]
    source = envelope.sources[0]
    assert claim.text == state["answer"]
    assert claim.supported is True
    assert claim.citation_ids == (citation.citation_id,)
    assert citation.claim_id == claim.claim_id
    assert citation.evidence_id == evidence.evidence_id
    assert citation.answer_start == claim.answer_start
    assert citation.answer_end == claim.answer_end
    assert citation.reason_codes == ("EXACT_CONTENT_MATCH",)
    assert source.evidence_id == evidence.evidence_id
    assert source.document_version_id == "dv_hak180_v1"
    assert source.index_version_id == "idx_hak180_v1"
    assert source.chunk_id == "chunk-grounding"
    assert source.location_label == "第1页"
    assert source.preview_url == f"/api/v1/evidence/{evidence.evidence_id}"

    assert envelope.versions.prompt_hash == sha256_digest(state["prompt"])
    assert envelope.versions.config_version == "cfg-todo11"
    assert envelope.versions.index_version_ids == ("idx_hak180_v1",)
    assert envelope.versions.retrieval_profile_id == "rpf-default"
    assert envelope.versions.retrieval_profile_version == 3
    assert envelope.versions.context_pack_version == "context-pack.v1"
    assert envelope.usage.total_tokens == 440
    assert envelope.degraded is False


def test_service_filters_citations_outside_context_evidence_whitelist() -> None:
    state, evidence = _normal_state()
    state["citations"].append(
        {
            "evidence_id": OTHER_EVIDENCE_ID,
            "answer_span": {"start": 0, "end": len(state["answer"])},
            "validation_status": "valid",
            "support_score": 1.0,
            "quote_hash": QUOTE_HASH,
        }
    )

    envelope = build_answer_envelope(state)

    assert [item.evidence_id for item in envelope.sources] == [evidence.evidence_id]
    assert [item.evidence_id for item in envelope.citations] == [evidence.evidence_id]
    assert OTHER_EVIDENCE_ID not in envelope.model_dump_json()


@pytest.mark.parametrize(
    ("action", "reason_code", "answer", "finish_reason", "confidence"),
    [
        (
            "REFUSE",
            "NO_EVIDENCE",
            "当前知识库中没有足够证据回答该问题。",
            "no_evidence",
            "low",
        ),
        (
            "DIRECT_ANSWER",
            "GREETING",
            "你好，我可以帮助查询知识库内容。",
            "direct_answer",
            "unknown",
        ),
    ],
)
def test_refusal_and_direct_answer_have_explicit_non_grounded_outcomes(
    action: str,
    reason_code: str,
    answer: str,
    finish_reason: str,
    confidence: str,
) -> None:
    envelope = build_answer_envelope(
        {
            "answer_id": ANSWER_ID,
            "query_run_id": QUERY_RUN_ID,
            "session_id": "sess-todo11",
            "answer": answer,
            "answer_decision": {
                "action": action,
                "reason_code": reason_code,
            },
            "grounding": {"status": "failed"},
            "evidences": [],
            "reranked_docs": [],
            "citations": [],
            "sources": [],
            "usage": {
                "status": "unavailable",
                "reason": "generation was not invoked",
            },
        }
    )

    assert envelope.decision == action
    assert envelope.finish_reason == finish_reason
    assert envelope.grounding.status == "not_applicable"
    assert envelope.confidence.level == confidence
    assert envelope.citations == ()
    assert envelope.sources == ()
    assert envelope.usage.status == "unavailable"


def test_refusal_keeps_frozen_index_read_set_for_replay_audit() -> None:
    envelope = build_answer_envelope(
        {
            "answer_id": ANSWER_ID,
            "query_run_id": QUERY_RUN_ID,
            "session_id": "sess-refusal-index-audit",
            "answer": "当前证据不足以形成可验证的回答。",
            "answer_decision": {
                "action": "REFUSE",
                "reason_code": "UNSUPPORTED_GENERATION",
            },
            "grounding": {"status": "failed"},
            "evidences": [],
            "reranked_docs": [],
            "citations": [],
            "sources": [],
            "vector_read_scopes": [
                {
                    "tenant_id": "tenant_default",
                    "kb_id": "kb_dolphin",
                    "document_id": "doc_dolphin",
                    "document_version_id": "dv_dolphin_v1",
                    "index_version_id": "idx_dolphin_v2",
                    "collection_name": "stt_chunks_dolphin",
                }
            ],
            "usage": {
                "status": "unavailable",
                "reason": "generation result was rejected by grounding",
            },
        }
    )

    assert envelope.sources == ()
    assert envelope.citations == ()
    assert envelope.versions.index_version_ids == ("idx_dolphin_v2",)


def test_legacy_text_history_is_stable_explicitly_degraded_and_read_only_compatible() -> None:
    created_at = datetime(2026, 7, 24, 18, 30, tzinfo=UTC)
    first = legacy_envelope(
        session_id="sess-legacy",
        answer="旧历史只保存了答案文本。",
        answer_id="not-a-valid-answer-id",
        created_at=created_at,
        seed="mongo-object-id",
    )
    second = legacy_envelope(
        session_id="sess-legacy",
        answer="旧历史只保存了答案文本。",
        answer_id="not-a-valid-answer-id",
        created_at=created_at,
        seed="mongo-object-id",
    )

    assert first is not None
    assert second is not None
    assert first == second
    assert first.answer_id.startswith("ans_")
    assert first.query_run_id.startswith("qry_")
    assert first.decision == "ANSWER"
    assert first.degraded is True
    assert first.degradation_reasons == ("LEGACY_TEXT_ONLY",)
    assert first.confidence.reasons == ("LEGACY_TEXT_ONLY",)
    assert first.grounding.status == "not_applicable"
    assert first.sources == ()
    assert first.citations == ()
    assert first.created_at == created_at
    assert legacy_envelope(session_id="sess-legacy", answer="") is None

    valid_id = "ans_" + "9" * 32
    preserved = legacy_envelope(
        session_id="sess-legacy",
        answer="已有合法答案身份。",
        answer_id=valid_id,
    )
    assert preserved is not None
    assert preserved.answer_id == valid_id


def test_answer_envelope_json_roundtrip_preserves_unicode_tuples_and_timestamp() -> None:
    envelope = _envelope()
    encoded = envelope.model_dump_json()
    restored = AnswerEnvelope.model_validate_json(encoded)

    assert "设备必须可靠接地" in encoded
    assert restored == envelope
    assert isinstance(restored.claims, tuple)
    assert isinstance(restored.citations, tuple)
    assert isinstance(restored.sources, tuple)
    assert restored.created_at.tzinfo is not None


def test_nested_schema_invariants_reject_invalid_usage_and_source_identity() -> None:
    with pytest.raises(ValidationError, match="requires reason"):
        AnswerUsage(status="unavailable")

    with pytest.raises(ValidationError, match="complete version identity"):
        AnswerSource(
            evidence_id=EVIDENCE_ID,
            source_type="local",
            content_hash=CONTENT_HASH,
        )

    with pytest.raises(ValidationError, match="web answer source requires url"):
        AnswerSource(
            evidence_id=EVIDENCE_ID,
            source_type="web",
            content_hash=CONTENT_HASH,
        )

    with pytest.raises(ValidationError, match="supported claim requires"):
        AnswerClaim(
            claim_id="clm_1",
            text="有事实主张",
            answer_start=0,
            answer_end=5,
            supported=True,
        )
