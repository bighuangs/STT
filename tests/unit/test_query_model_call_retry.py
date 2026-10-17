from __future__ import annotations

import json

import httpx
import openai
import pytest

from app.infrastructure.model_calls.query_retry import (
    QueryModelCallDeadlineExceeded,
    QueryModelCallTransportUnavailable,
    QueryModelRetryPolicy,
    invoke_query_model_call,
)
from app.shared.model.lm_utils import _build_sync_http_client


def _policy(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.0,
    max_delay: float = 0.0,
    total_timeout: float = 10.0,
) -> QueryModelRetryPolicy:
    return QueryModelRetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        total_timeout_seconds=total_timeout,
    )


def test_transient_connect_timeout_retries_inside_one_logical_call() -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    provider_calls = 0
    audit: dict = {}
    state: dict = {}

    def invoke(_remaining_seconds: float) -> str:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise httpx.ConnectTimeout(
                "sensitive provider detail must not be audited",
                request=request,
            )
        return "ok"

    result = invoke_query_model_call(
        state,
        scene="answer_generation",
        model_id="test-model",
        invoke=invoke,
        audit=audit,
        policy=_policy(),
    )

    assert result == "ok"
    assert provider_calls == 2
    assert audit["status"] == "succeeded"
    assert audit["provider_attempt_count"] == 2
    assert audit["retry_count"] == 1
    assert [item["status"] for item in audit["attempts"]] == [
        "retryable_failure",
        "succeeded",
    ]
    assert state["query_model_call_audits"] == [audit]
    assert "sensitive provider detail" not in json.dumps(audit)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid response schema"),
        openai.BadRequestError(
            "bad request body must not be audited",
            response=httpx.Response(
                400,
                request=httpx.Request(
                    "POST",
                    "https://provider.invalid/v1/chat/completions",
                ),
            ),
            body={"sensitive": "body"},
        ),
        httpx.LocalProtocolError("local request construction defect must not be retried"),
    ],
)
def test_non_retryable_errors_fail_after_one_attempt(error: Exception) -> None:
    provider_calls = 0
    audit: dict = {}

    def invoke(_remaining_seconds: float) -> None:
        nonlocal provider_calls
        provider_calls += 1
        raise error

    with pytest.raises(type(error)):
        invoke_query_model_call(
            {},
            scene="answer_completeness_self_review",
            model_id="test-model",
            invoke=invoke,
            audit=audit,
            policy=_policy(max_attempts=3),
        )

    assert provider_calls == 1
    assert audit["status"] == "failed_non_retryable"
    assert audit["provider_attempt_count"] == 1
    assert audit["retry_count"] == 0
    serialized = json.dumps(audit)
    assert "bad request body" not in serialized
    assert "sensitive" not in serialized
    assert "local request construction defect" not in serialized


def test_retry_attempt_limit_is_enforced() -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    provider_calls = 0
    audit: dict = {}

    def invoke(_remaining_seconds: float) -> None:
        nonlocal provider_calls
        provider_calls += 1
        raise openai.APITimeoutError(request)

    with pytest.raises(QueryModelCallTransportUnavailable) as captured:
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            audit=audit,
            policy=_policy(max_attempts=2),
        )

    assert provider_calls == 2
    assert audit["status"] == "failed_attempt_limit"
    assert audit["provider_attempt_count"] == 2
    assert audit["retry_count"] == 1
    assert isinstance(captured.value.__cause__, openai.APITimeoutError)


@pytest.mark.parametrize(
    "transport_error",
    [
        openai.APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://provider.invalid/v1/chat/completions",
            ),
        ),
        httpx.ConnectError(
            "synthetic network detail must not escape",
            request=httpx.Request(
                "POST",
                "https://provider.invalid/v1/chat/completions",
            ),
        ),
        httpx.RemoteProtocolError("synthetic remote protocol detail must not escape"),
        httpx.ProxyError("synthetic proxy detail must not escape"),
    ],
)
def test_network_retry_exhaustion_raises_stable_transport_error(
    transport_error: Exception,
) -> None:
    """网络重试耗尽后只暴露稳定项目异常，同时保留原始 cause。"""

    audit: dict = {}

    def invoke(_remaining_seconds: float) -> None:
        raise transport_error

    with pytest.raises(ConnectionError) as captured:
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            audit=audit,
            policy=_policy(max_attempts=2),
        )

    assert type(captured.value).__name__ == "QueryModelCallTransportUnavailable"
    assert captured.value.__cause__ is transport_error
    assert str(captured.value) == "query model transport unavailable"
    assert audit["status"] == "failed_attempt_limit"
    assert audit["provider_attempt_count"] == 2
    assert "synthetic network detail" not in json.dumps(audit)
    assert "synthetic remote protocol detail" not in json.dumps(audit)
    assert "synthetic proxy detail" not in json.dumps(audit)


def test_non_network_retry_failure_is_not_reclassified_as_transport_unavailable() -> None:
    """非网络异常继续保持原类型，避免把实现错误伪装成瞬时依赖故障。"""

    error = ValueError("synthetic contract defect")

    def invoke(_remaining_seconds: float) -> None:
        raise error

    with pytest.raises(ValueError) as captured:
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            policy=_policy(max_attempts=2),
        )

    assert captured.value is error
    assert type(captured.value).__name__ != "QueryModelCallTransportUnavailable"


def test_retry_does_not_start_attempt_past_logical_deadline() -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    current = [0.0]
    provider_calls = 0
    audit: dict = {}

    def monotonic() -> float:
        return current[0]

    def sleep(seconds: float) -> None:
        current[0] += seconds

    def invoke(_remaining_seconds: float) -> None:
        nonlocal provider_calls
        provider_calls += 1
        current[0] += 0.1
        raise httpx.ConnectError("connection unavailable", request=request)

    with pytest.raises(QueryModelCallDeadlineExceeded):
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            audit=audit,
            policy=_policy(
                max_attempts=3,
                base_delay=0.3,
                max_delay=0.3,
                total_timeout=0.45,
            ),
            monotonic=monotonic,
            sleep=sleep,
        )

    assert provider_calls == 2
    assert audit["status"] == "deadline_exhausted"
    assert audit["provider_attempt_count"] == 2
    assert current[0] == pytest.approx(0.5)


def test_attempt_receives_remaining_logical_deadline_as_transport_budget() -> None:
    current = [0.0]
    received: list[float] = []

    def monotonic() -> float:
        return current[0]

    def invoke(remaining_seconds: float) -> None:
        received.append(remaining_seconds)
        current[0] += 0.08

    with pytest.raises(QueryModelCallDeadlineExceeded):
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            policy=_policy(max_attempts=1, total_timeout=0.01),
            monotonic=monotonic,
        )

    assert received == [pytest.approx(0.01)]


def test_last_retryable_attempt_crossing_deadline_uses_typed_timeout() -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    current = [0.0]
    audit: dict = {}

    def monotonic() -> float:
        return current[0]

    def invoke(_remaining_seconds: float) -> None:
        current[0] = 2.0
        raise httpx.ReadTimeout("provider exceeded budget", request=request)

    with pytest.raises(QueryModelCallDeadlineExceeded):
        invoke_query_model_call(
            {},
            scene="answer_generation",
            model_id="test-model",
            invoke=invoke,
            audit=audit,
            policy=_policy(max_attempts=1, total_timeout=1.0),
            monotonic=monotonic,
        )

    assert audit["status"] == "deadline_exhausted"
    assert audit["attempts"][0]["status"] == "deadline_exceeded"


def test_sync_http_transport_has_all_timeout_dimensions() -> None:
    client = _build_sync_http_client()
    try:
        assert client.timeout.connect is not None
        assert client.timeout.read is not None
        assert client.timeout.write is not None
        assert client.timeout.pool is not None
    finally:
        client.close()
