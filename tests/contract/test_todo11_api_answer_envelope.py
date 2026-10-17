from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.http import query_server
from app.api.schemas.query import HistoryItem, QueryResponse
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
from app.security.principal import Principal
from app.shared.utils.sse_utils import SSEEvent


ANSWER_ID = "ans_" + "a" * 32
QUERY_RUN_ID = "qry_" + "b" * 32
EVIDENCE_ID = "ev_" + "c" * 40
CONTENT_HASH = "sha256:" + "d" * 64
QUOTE_HASH = "sha256:" + "e" * 64
ANSWER_TEXT = "设备启动前必须可靠接地。"


def _principal() -> Principal:
    return Principal(
        user_id="user-contract",
        tenant_id="tenant-default",
        roles=("TenantAdmin",),
        scopes=("*",),
        auth_type="test",
        credential_id="credential-contract",
    )


def _envelope() -> AnswerEnvelope:
    claim = AnswerClaim(
        claim_id="clm_1",
        text=ANSWER_TEXT,
        answer_start=0,
        answer_end=len(ANSWER_TEXT),
        citation_ids=("cit_1",),
        supported=True,
    )
    citation = AnswerCitation(
        citation_id="cit_1",
        claim_id=claim.claim_id,
        evidence_id=EVIDENCE_ID,
        answer_start=claim.answer_start,
        answer_end=claim.answer_end,
        validation="valid",
        verifier_score=1.0,
        source_quote_hash=QUOTE_HASH,
        reason_codes=("EXACT_CONTENT_MATCH",),
    )
    source = AnswerSource(
        evidence_id=EVIDENCE_ID,
        source_type="local",
        source_label="hak180产品安全手册.pdf",
        knowledge_base_id="kb-hak180",
        document_id="doc-hak180",
        document_version_id="dv-hak180-v1",
        index_version_id="idx-hak180-v1",
        chunk_id="chunk-hak180-grounding",
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
        preview_url=f"/api/v1/evidence/{EVIDENCE_ID}",
    )
    return AnswerEnvelope(
        answer_id=ANSWER_ID,
        query_run_id=QUERY_RUN_ID,
        session_id="sess-contract",
        message_id="msg-contract",
        answer=ANSWER_TEXT,
        decision="ANSWER",
        claims=(claim,),
        citations=(citation,),
        sources=(source,),
        confidence=AnswerConfidence(
            level="high",
            score=1.0,
            reasons=("TRUSTED_CLAIM_COVERAGE_HIGH",),
        ),
        grounding=AnswerGrounding(
            status="passed",
            claim_count=1,
            supported_claim_count=1,
            claim_coverage=1.0,
        ),
        finish_reason="stop",
        versions=AnswerVersions(
            prompt_hash="sha256:" + "f" * 64,
            model_id="test-model",
            config_version="cfg-contract",
            index_version_ids=("idx-hak180-v1",),
            retrieval_profile_id="rpf-default",
            retrieval_profile_version=1,
        ),
        usage=AnswerUsage(
            status="unavailable",
            reason="contract test has no provider usage",
        ),
        trace_id="trace-contract",
        created_at=datetime(2026, 7, 24, 19, 0, tzinfo=UTC),
    )


def _final_state(envelope: AnswerEnvelope) -> dict:
    return {
        "answer": envelope.answer,
        "image_urls": [],
        "answer_id": envelope.answer_id,
        "citations": [
            {
                "label": "E1",
                "evidence_id": EVIDENCE_ID,
                "validation_status": "valid",
            }
        ],
        "sources": [{"label": "E1", "evidence_id": EVIDENCE_ID}],
        "citation_validation": {"status": "valid", "coverage": 1.0},
        "retrieval_profile_id": "rpf-default",
        "retrieval_profile_version": 1,
        "retrieval_profile_hash": "sha256:" + "1" * 64,
        "retrieval_diagnostics": {"status": "completed"},
        "query_plan": {"schema_version": "query-plan.v1"},
        "fusion_diagnostics": {"schema_version": "fusion.v1"},
        "context_pack_summary": {"schema_version": "context-pack.v1"},
        "evidence_assessment": {"status": "sufficient"},
        "answer_decision": {"action": "ANSWER"},
        "answer_envelope": envelope.model_dump(mode="json"),
    }


def test_query_response_strictly_deserializes_nested_answer_envelope() -> None:
    raw = _envelope().model_dump(mode="json")
    response = QueryResponse(
        message="处理完成！",
        session_id="sess-contract",
        answer=ANSWER_TEXT,
        answer_envelope=raw,
    )

    assert isinstance(response.answer_envelope, AnswerEnvelope)
    assert response.answer_envelope.model_dump(mode="json") == raw

    wrong_version = deepcopy(raw)
    wrong_version["schema_version"] = "answer.v999"
    with pytest.raises(ValidationError, match="answer.v1"):
        QueryResponse(
            message="处理完成！",
            session_id="sess-contract",
            answer=ANSWER_TEXT,
            answer_envelope=wrong_version,
        )

    leaked_internal_field = deepcopy(raw)
    leaked_internal_field["system_prompt"] = "must never cross the API boundary"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryResponse(
            message="处理完成！",
            session_id="sess-contract",
            answer=ANSWER_TEXT,
            answer_envelope=leaked_internal_field,
        )


def test_sync_and_sse_final_project_the_identical_answer_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _envelope()
    final_state = _final_state(envelope)
    monkeypatch.setattr(
        query_server,
        "register_query_session",
        lambda *_: {"web_search_allowed": False, "retrieval_profile": {}},
    )
    monkeypatch.setattr(query_server, "invoke_query", lambda **_: deepcopy(final_state))
    monkeypatch.setattr(query_server, "get_task_result", lambda *_: None)
    monkeypatch.setattr(query_server, "get_done_task_list", lambda *_: ["answer"])

    sync = query_server.execute_query(
        "启动前要做什么？",
        session_id="sess-contract",
        kb_id="kb-hak180",
        principal=_principal(),
    )
    assert isinstance(sync.answer_envelope, AnswerEnvelope)

    events: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        query_server,
        "push_to_session",
        lambda _session, event, payload: events.append((event, payload)),
    )
    query_server.run_stream_query_background(
        "sess-contract",
        "启动前要做什么？",
        "kb-hak180",
        _principal(),
        False,
    )

    event, payload = events[-1]
    assert event == SSEEvent.FINAL
    streamed = AnswerEnvelope.model_validate(payload["answer_envelope"])
    assert payload["answer"] == streamed.answer == sync.answer
    assert payload["answer_id"] == streamed.answer_id == sync.answer_id
    assert streamed == sync.answer_envelope
    assert set(streamed.model_dump(mode="json")) == set(
        sync.answer_envelope.model_dump(mode="json")
    )


def test_sse_final_rejects_an_invalid_answer_envelope_instead_of_publishing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _final_state(_envelope())
    malformed["answer_envelope"]["claims"][0]["text"] = "伪造且不匹配答案区间"
    monkeypatch.setattr(query_server, "invoke_query", lambda **_: malformed)
    monkeypatch.setattr(query_server, "get_task_result", lambda *_: None)
    events: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        query_server,
        "push_to_session",
        lambda _session, event, payload: events.append((event, payload)),
    )
    monkeypatch.setattr(query_server, "update_task_status", lambda *_: None)

    query_server.run_stream_query_background(
        "sess-invalid-envelope",
        "问题",
        "kb-hak180",
        _principal(),
        False,
    )

    assert events
    assert events[-1][0] == SSEEvent.ERROR
    assert all(event != SSEEvent.FINAL for event, _ in events)


def test_history_item_is_strict_and_server_stably_adapts_legacy_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _envelope().model_dump(mode="json")
    item = HistoryItem(
        _id="mongo-new",
        session_id="sess-contract",
        role="assistant",
        text=ANSWER_TEXT,
        answer_id=ANSWER_ID,
        answer_envelope=raw,
    )
    assert isinstance(item.answer_envelope, AnswerEnvelope)

    malformed = deepcopy(raw)
    malformed["citations"][0]["evidence_id"] = "ev_" + "0" * 40
    with pytest.raises(ValidationError, match="outside source whitelist"):
        HistoryItem(
            _id="mongo-invalid",
            session_id="sess-contract",
            role="assistant",
            text=ANSWER_TEXT,
            answer_envelope=malformed,
        )

    records = [
        {
            "_id": "mongo-new",
            "session_id": "sess-contract",
            "role": "assistant",
            "text": ANSWER_TEXT,
            "answer_id": ANSWER_ID,
            "answer_envelope": raw,
            "ts": 1_753_353_600.0,
        },
        {
            "_id": "mongo-legacy",
            "session_id": "sess-contract",
            "role": "assistant",
            "text": "旧记录只有答案文本。",
            "answer_id": "legacy-invalid-id",
            "ts": 1_753_353_601.0,
        },
        {
            "_id": "mongo-corrupt-envelope",
            "session_id": "sess-contract",
            "role": "assistant",
            "text": "损坏的新格式应稳定回退到旧文本。",
            "answer_id": "legacy-invalid-id",
            "answer_envelope": malformed,
            "ts": 1_753_353_602.0,
        },
        {
            "_id": "mongo-user",
            "session_id": "sess-contract",
            "role": "user",
            "text": "用户问题",
            "ts": 1_753_353_603.0,
        },
    ]
    monkeypatch.setattr(
        query_server.history_repository,
        "list_recent",
        lambda _session, limit=10: deepcopy(records[:limit]),
    )

    first = query_server.build_history_response("sess-contract", limit=10)
    second = query_server.build_history_response("sess-contract", limit=10)
    assert first == second
    assert first.items[0].answer_envelope == _envelope()

    legacy = first.items[1].answer_envelope
    corrupt_fallback = first.items[2].answer_envelope
    assert legacy is not None
    assert corrupt_fallback is not None
    assert legacy.degraded is True
    assert legacy.degradation_reasons == ("LEGACY_TEXT_ONLY",)
    assert corrupt_fallback.degraded is True
    assert corrupt_fallback.answer == "损坏的新格式应稳定回退到旧文本。"
    assert first.items[3].answer_envelope is None
    assert legacy.answer_id == second.items[1].answer_envelope.answer_id
    assert legacy.query_run_id == second.items[1].answer_envelope.query_run_id
