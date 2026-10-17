from __future__ import annotations

from contextlib import contextmanager

from app.application import model_usage_service
from app.application.observability_service import UsageLedger
from app.rag.query import hyde_search_service, query_rewrite_service
from app.security.principal import Principal
from langchain_core.messages import AIMessage


class _ResponseWithoutUsage:
    content = "answer"
    response_metadata = {}


class _ResponseWithUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.content = "answer"
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def test_provider_without_token_metadata_still_records_one_estimated_call(
    monkeypatch,
) -> None:
    ledger = UsageLedger(max_records=10)
    principal = Principal(
        user_id="user_a",
        tenant_id="tenant_a",
        roles=("TenantAdmin",),
        scopes=("query:execute",),
        auth_type="password",
        credential_id="cred_a",
        request_id="req_a",
    )
    monkeypatch.setattr(model_usage_service, "usage_ledger", ledger)
    monkeypatch.setattr(model_usage_service, "current_principal", lambda: principal)
    state = {
        "query_run_id": "qry_0123456789abcdef",
        "kb_id": "kb_a",
        "runtime_managed_history": False,
    }

    state["usage"] = model_usage_service.record_model_call(
        state,
        [_ResponseWithoutUsage()],
        scene="answer_plan_selection",
        model_id="frozen-model",
    )

    assert state["usage"]["status"] == "unavailable"
    summary = ledger.aggregate(
        "tenant_a",
        query_run_id="qry_0123456789abcdef",
    )
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0
    assert summary["estimated"] is True
    assert summary["record_count"] == 1


def test_public_usage_aggregates_multiple_logical_calls() -> None:
    state: dict = {"model_usage_summaries": []}

    model_usage_service.record_model_call(
        state,
        [_ResponseWithUsage(3, 2)],
        scene="query_rewrite",
        model_id="frozen-model",
    )
    aggregate = model_usage_service.record_model_call(
        state,
        [_ResponseWithUsage(5, 4)],
        scene="hyde_generation",
        model_id="frozen-model",
    )

    assert aggregate["status"] == "available"
    assert aggregate["input_tokens"] == 8
    assert aggregate["output_tokens"] == 6
    assert aggregate["total_tokens"] == 14


def test_sql_usage_telemetry_failure_is_fail_open(
    monkeypatch,
) -> None:
    principal = Principal(
        user_id="user_a",
        tenant_id="tenant_a",
        roles=("TenantAdmin",),
        scopes=("query:execute",),
        auth_type="password",
        credential_id="cred_a",
        request_id="req_a",
    )
    monkeypatch.setattr(model_usage_service, "current_principal", lambda: principal)

    @contextmanager
    def broken_session_scope():
        raise RuntimeError("synthetic SQL telemetry outage")
        yield

    monkeypatch.setattr(model_usage_service, "session_scope", broken_session_scope)
    state = {
        "query_run_id": "qry_0123456789abcdef",
        "runtime_managed_history": True,
    }

    aggregate = model_usage_service.record_model_call(
        state,
        [_ResponseWithUsage(3, 2)],
        scene="answer_plan_selection",
        model_id="frozen-model",
    )

    assert aggregate["status"] == "available"
    assert state["usage_telemetry_errors"] == [
        {
            "schema_version": "usage-telemetry-error.v1",
            "stage": "usage_ledger",
            "error_type": "RuntimeError",
        }
    ]


def test_query_llm_stages_use_the_shared_usage_collector(monkeypatch) -> None:
    scenes: list[str] = []
    responses = {
        "query_rewrite": "独立问题",
        "cross_language": "English question",
        "keyword_extraction": "启动，接地",
        "metadata_filter": '{"logic":"and","conditions":[]}',
        "hyde_generation": "设备启动前应可靠接地。",
    }

    def fake_invoke(state, *, scene, invoke):
        assert state is marker
        scenes.append(scene)
        return AIMessage(content=responses[scene])

    marker: dict = {"external_llm_allowed": True}
    monkeypatch.setattr(query_rewrite_service, "invoke_model_call", fake_invoke)
    monkeypatch.setattr(hyde_search_service, "invoke_model_call", fake_invoke)

    assert (
        query_rewrite_service.full_question(
            [{"role": "user", "rewritten_query": "上一问"}],
            "继续呢？",
            state=marker,
        )
        == "独立问题"
    )
    assert query_rewrite_service.cross_languages(
        "问题",
        ["英语"],
        state=marker,
    ) == ["问题", "English question"]
    assert (
        query_rewrite_service.keyword_extraction(
            "问题",
            state=marker,
        )
        == "启动，接地"
    )
    assert query_rewrite_service.gen_meta_filter(
        {"产品": {"HAK180": ["doc_a"]}},
        "问题",
        state=marker,
    ) == {"logic": "and", "conditions": []}
    assert (
        hyde_search_service.generate_hyde_answer(
            "问题",
            state=marker,
        )
        == "设备启动前应可靠接地。"
    )
    assert scenes == [
        "query_rewrite",
        "cross_language",
        "keyword_extraction",
        "metadata_filter",
        "hyde_generation",
    ]
