from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.api.http import query_runtime_routes, query_server
from app.domain.query_runtime import QueryRunStatus
from app.domain.evidence import citation_fact_bundle_hash
from app.infra.persistence.sql.models import Base
from app.infra.persistence.sql.query_runtime_models import (
    QueryConversationModel,
    QueryEventModel,
    QueryMessageModel,
    QueryRunModel,
    QueryTurnModel,
)
from app.infra.persistence.sql.query_runtime_repository import QueryRuntimeRepository
from app.infrastructure.answering import delivery


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _registered_routes() -> set[tuple[str, str]]:
    return {
        (route.path, method)
        for route in query_server.app.routes
        for method in (getattr(route, "methods", None) or set())
    }


def test_legacy_query_stream_and_mongo_history_http_routes_are_retired() -> None:
    routes = _registered_routes()

    assert ("/query", "POST") not in routes
    assert ("/stream/{session_id}", "GET") not in routes
    assert ("/history/{session_id}", "GET") not in routes
    assert ("/history/{session_id}", "DELETE") not in routes
    assert ("/api/v1/query-runs", "POST") in routes


def test_chat_page_has_no_legacy_answer_or_history_fallback() -> None:
    page = (PROJECT_ROOT / "app/process/query/page/chat.html").read_text(encoding="utf-8")

    assert "runtimeApiJson('/query'" not in page
    assert "`${API_BASE}/stream/" not in page
    assert "`${API_BASE}/history/" not in page
    assert "runLegacyQuery" not in page


def test_legacy_graph_invocation_fails_before_graph_execution() -> None:
    with pytest.raises(
        query_server.LegacyQueryPathRetiredError,
        match="use POST /api/v1/query-runs",
    ):
        query_server.invoke_query(
            session_id="session-retired",
            query="不会进入查询图",
            is_stream=False,
        )


def test_legacy_delivery_fails_before_any_persistence_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("旧回答路径不得触发引用或终态持久化")

    monkeypatch.setattr(delivery, "revalidate_cited_evidence", unexpected)
    monkeypatch.setattr(delivery, "persist_verified_citations", unexpected)

    with pytest.raises(
        delivery.LegacyAnswerDeliveryRetiredError,
        match="durable QueryRun runtime",
    ):
        delivery.commit_answer_delivery(
            {
                "evaluation_mode": False,
                "runtime_managed_history": False,
                "pending_citation_facts": [{"malformed": "must not be parsed"}],
            }
        )


@pytest.mark.parametrize(
    ("state", "expected_status", "delegated"),
    [
        ({"evaluation_mode": True}, "evaluated", False),
        ({"runtime_managed_history": True}, "delegated", True),
    ],
)
def test_formal_runtime_and_evaluation_keep_the_proof_aware_commit_path(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, bool],
    expected_status: str,
    delegated: bool,
) -> None:
    persisted: list[tuple[dict, dict, list]] = []
    monkeypatch.setattr(delivery, "revalidate_cited_evidence", lambda *_args: {})
    monkeypatch.setattr(
        delivery,
        "persist_verified_citations",
        lambda current, *, evidences, facts: persisted.append((current, evidences, facts)),
    )

    empty_bundle_hash = citation_fact_bundle_hash([])
    receipt = delivery.commit_answer_delivery(
        {
            **state,
            "pending_citation_facts": [],
            "answer_render_receipt": {
                "schema_version": "answer-render-receipt.v3",
                "citation_count": 0,
                "citation_fact_bundle_hash": empty_bundle_hash,
            },
        }
    )

    assert receipt == {
        "schema_version": "answer-commit-receipt.v2",
        "status": expected_status,
        "stream_chunk_count": 0,
        "task_result_written": False,
        "history_written": False,
        "citation_fact_count": 0,
        "citation_fact_bundle_hash": empty_bundle_hash,
        "runtime_terminal_write_delegated": delegated,
    }
    assert len(persisted) == 1


def test_durable_query_runtime_terminal_projection_retains_answer_proof() -> None:
    proof = {
        "schema_version": "answer-proof.v3",
        "status": "accepted_strict",
        "strict_eligible": True,
    }

    result = query_runtime_routes._public_result(
        {
            "answer": "已证明回答",
            "answer_proof": proof,
            "prompt": "不得持久化的 prompt",
            "context_pack": {"packed_blocks": ["不得持久化的证据正文"]},
        }
    )

    assert result["answer_proof"] == proof
    assert "prompt" not in result
    assert "context_pack" not in result


def test_durable_query_runtime_atomically_persists_proof_with_assistant_message(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'proof-runtime.db').as_posix()}", future=True)
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(
        engine,
        tables=[
            QueryConversationModel.__table__,
            QueryTurnModel.__table__,
            QueryRunModel.__table__,
            QueryMessageModel.__table__,
            QueryEventModel.__table__,
        ],
    )
    proof = {
        "schema_version": "answer-proof.v3",
        "status": "accepted_strict",
        "strict_eligible": True,
    }

    try:
        with Session(engine) as session:
            repository = QueryRuntimeRepository(
                session,
                tenant_id="tenant-proof",
                user_id="user-proof",
            )
            conversation = repository.create_conversation(kb_id="kb-proof")
            _, run, _, _ = repository.begin_turn(
                conversation_id=conversation.conversation_id,
                client_turn_id="proof-turn",
                user_content="证明是否随答案保存？",
            )
            for stage in (
                QueryRunStatus.PLANNING,
                QueryRunStatus.RETRIEVING,
                QueryRunStatus.PACKING,
                QueryRunStatus.GENERATING,
                QueryRunStatus.VALIDATING,
            ):
                run = repository.transition_run(
                    run.query_run_id,
                    expected_revision=run.revision,
                    target=stage,
                )

            completed = repository.complete_run(
                run.query_run_id,
                expected_revision=run.revision,
                answer="已证明回答",
                result={"answer_proof": proof},
            )
            assistant = repository.list_messages(conversation.conversation_id)[1]

            assert completed.result is not None
            assert completed.result["answer_proof"] == proof
            assert assistant.content == "已证明回答"
            assert assistant.metadata["answer_proof"] == proof
    finally:
        engine.dispose()
