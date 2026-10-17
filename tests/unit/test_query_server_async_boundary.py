from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from app.api.http import query_server
from app.api.schemas.query import QueryRequest, QueryResponse
from app.security.principal import (
    Principal,
    current_principal,
    reset_current_principal,
    set_current_principal,
)


def _principal() -> Principal:
    return Principal(
        user_id="user-query-boundary",
        tenant_id="tenant-query-boundary",
        roles=("viewer",),
        scopes=("query:execute",),
        auth_type="session",
        credential_id="credential-query-boundary",
        request_id="request-query-boundary",
    )


def _request(principal: Principal) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(principal=principal))


def test_non_stream_query_runs_sync_graph_off_event_loop_with_security_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    principal = _principal()
    event_loop_thread = threading.get_ident()
    captured: dict[str, object] = {}

    def fake_execute_query(**kwargs: object) -> QueryResponse:
        captured["thread_id"] = threading.get_ident()
        captured["context_principal"] = current_principal()
        captured["kwargs"] = kwargs
        return QueryResponse(
            message="处理完成！",
            session_id="session-threadpool",
            answer="安全结果",
        )

    monkeypatch.setattr(query_server, "execute_query", fake_execute_query)
    token = set_current_principal(principal)
    try:
        response = asyncio.run(
            query_server.query(
                QueryRequest(query="测试异步边界", kb_id="kb-query-boundary"),
                BackgroundTasks(),
                _request(principal),
            )
        )
    finally:
        reset_current_principal(token)

    assert captured["thread_id"] != event_loop_thread
    assert captured["context_principal"] == principal
    assert captured["kwargs"] == {
        "query": "测试异步边界",
        "session_id": None,
        "kb_id": "kb-query-boundary",
        "principal": principal,
        "retrieval_filter": None,
    }
    assert response.session_id == "session-threadpool"
    assert response.answer == "安全结果"


def test_non_stream_query_propagates_sync_execution_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class QueryExecutionFailure(RuntimeError):
        pass

    execution_thread: list[int] = []

    def fail_execute_query(**_kwargs: object) -> QueryResponse:
        execution_thread.append(threading.get_ident())
        raise QueryExecutionFailure("QUERY_EXECUTION_FAILED")

    monkeypatch.setattr(query_server, "execute_query", fail_execute_query)
    event_loop_thread = threading.get_ident()

    with pytest.raises(QueryExecutionFailure, match="QUERY_EXECUTION_FAILED"):
        asyncio.run(
            query_server.query(
                QueryRequest(query="触发同步执行失败"),
                BackgroundTasks(),
                _request(_principal()),
            )
        )

    assert execution_thread and execution_thread[0] != event_loop_thread


def test_stream_query_keeps_existing_background_dispatch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_thread: list[int] = []

    def fake_start_stream_query(**_kwargs: object) -> QueryResponse:
        dispatch_thread.append(threading.get_ident())
        return QueryResponse(message="结果正在处理中...", session_id="session-stream")

    monkeypatch.setattr(query_server, "start_stream_query", fake_start_stream_query)
    monkeypatch.setattr(
        query_server,
        "execute_query",
        lambda **_kwargs: pytest.fail("流式路径不得调用非流式 execute_query"),
    )
    event_loop_thread = threading.get_ident()

    response = asyncio.run(
        query_server.query(
            QueryRequest(query="流式查询", is_stream=True),
            BackgroundTasks(),
            _request(_principal()),
        )
    )

    assert dispatch_thread == [event_loop_thread]
    assert response.session_id == "session-stream"
