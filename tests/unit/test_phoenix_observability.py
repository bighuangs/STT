from __future__ import annotations

import os
from collections.abc import Iterator
from logging import WARNING

import pytest
from pydantic import ValidationError

from app.config.observability import PhoenixObservabilitySettings
from app.infrastructure.observability import phoenix
from app.infrastructure.observability.phoenix import PhoenixLifecycleState


class _FakeProvider:
    def __init__(self, *, shutdown_error: Exception | None = None) -> None:
        self.shutdown_calls = 0
        self.shutdown_error = shutdown_error

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.fixture(autouse=True)
def reset_adapter_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(phoenix, "_provider", None)
    yield
    monkeypatch.setattr(phoenix, "_provider", None)


def test_settings_are_disabled_and_secret_is_hidden_by_default() -> None:
    settings = PhoenixObservabilitySettings()

    assert settings.enabled is False
    assert settings.batch is True
    assert settings.export_endpoint == "http://localhost:6006/v1/traces"
    assert "api_key_value" not in repr(settings)


def test_settings_accept_official_project_alias_and_normalize_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOENIX_PROJECT", "stt-eval")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "ignored-alias")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.example/prefix/")

    settings = PhoenixObservabilitySettings()

    assert settings.project_name == "stt-eval"
    assert settings.export_endpoint == "https://phoenix.example/prefix/v1/traces"


@pytest.mark.parametrize(
    "endpoint",
    (
        "file:///tmp/traces",
        "https://user:secret@phoenix.example",
        "https://phoenix.example?api_key=secret",
    ),
)
def test_settings_reject_unsafe_collector_endpoints(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        PhoenixObservabilitySettings(collector_endpoint=endpoint)


def test_disabled_initialize_does_not_import_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> object:
        raise AssertionError("禁用时不应加载 Phoenix SDK")

    monkeypatch.setattr(phoenix, "_load_register", fail_if_loaded)

    result = phoenix.initialize(PhoenixObservabilitySettings(enabled=False))

    assert result.state is PhoenixLifecycleState.DISABLED
    assert result.active is False
    assert phoenix.is_initialized() is False


def test_invalid_environment_configuration_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOENIX_ENABLED", "true")
    monkeypatch.setenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        "https://phoenix.example?api_key=must-not-leak",
    )

    result = phoenix.initialize()

    assert result.state is PhoenixLifecycleState.INITIALIZATION_FAILED
    assert result.error_type == "ValidationError"
    assert phoenix.is_initialized() is False


def test_initialize_is_idempotent_and_passes_masked_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    calls: list[dict[str, object]] = []

    def register(**kwargs: object) -> _FakeProvider:
        calls.append(kwargs)
        for name in phoenix._REDACTION_ENVIRONMENT:
            assert os.environ[name] == "true"
        return provider

    monkeypatch.setattr(phoenix, "_load_register", lambda: register)
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "false")
    settings = PhoenixObservabilitySettings(
        enabled=True,
        collector_endpoint="https://phoenix.example/base",
        project_name="stt-test",
        api_key_value="super-secret",
    )
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in settings.model_dump_json()

    first = phoenix.initialize(settings)
    second = phoenix.initialize(settings)

    assert first.state is PhoenixLifecycleState.INITIALIZED
    assert second.state is PhoenixLifecycleState.ALREADY_INITIALIZED
    assert len(calls) == 1
    assert calls[0] == {
        "endpoint": "https://phoenix.example/base/v1/traces",
        "project_name": "stt-test",
        "protocol": "http/protobuf",
        "batch": True,
        "auto_instrument": True,
        "api_key": "super-secret",
        "set_global_tracer_provider": True,
        "verbose": False,
    }
    assert os.environ["OPENINFERENCE_HIDE_INPUTS"] == "false"
    for name in phoenix._REDACTION_ENVIRONMENT[1:]:
        assert name not in os.environ


def test_missing_dependency_and_sdk_failure_are_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = PhoenixObservabilitySettings(enabled=True)

    def missing() -> object:
        raise ModuleNotFoundError("phoenix")

    monkeypatch.setattr(phoenix, "_load_register", missing)
    missing_result = phoenix.initialize(settings)

    assert missing_result.state is PhoenixLifecycleState.DEPENDENCY_MISSING
    assert missing_result.error_type == "ModuleNotFoundError"
    assert phoenix.is_initialized() is False

    def broken_register(**_: object) -> _FakeProvider:
        raise ConnectionError("Authorization: Bearer super-secret")

    monkeypatch.setattr(phoenix, "_load_register", lambda: broken_register)
    caplog.set_level(WARNING, logger=phoenix.__name__)
    failure_result = phoenix.initialize(settings)

    assert failure_result.state is PhoenixLifecycleState.INITIALIZATION_FAILED
    assert failure_result.error_type == "ConnectionError"
    assert "super-secret" not in repr(failure_result)
    assert "super-secret" not in caplog.text
    assert phoenix.is_initialized() is False


def test_shutdown_is_idempotent_and_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _FakeProvider(shutdown_error=TimeoutError("collector secret"))
    monkeypatch.setattr(phoenix, "_provider", provider)

    first = phoenix.shutdown()
    second = phoenix.shutdown()

    assert first.state is PhoenixLifecycleState.SHUTDOWN_FAILED
    assert first.error_type == "TimeoutError"
    assert second.state is PhoenixLifecycleState.ALREADY_SHUTDOWN
    assert provider.shutdown_calls == 1
    assert phoenix.is_initialized() is False
