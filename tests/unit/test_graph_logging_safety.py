from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect
import re
from collections.abc import Iterator

import pytest

from app.shared.runtime.logger import logger, node_log, step_log


def _sensitive_values() -> dict[str, str]:
    """拆分构造合成值，避免失败报告展示完整正文。"""

    return {
        "query": "".join(("SYNTH", "-QUERY-", "LOG-BOUNDARY")),
        "evidence": "".join(("SYNTH", "-EVIDENCE-", "LOG-BOUNDARY")),
        "credential": "".join(("sk-test-", "SYNTH-", "LOG-BOUNDARY")),
    }


def _exception_with_sensitive_cause(values: dict[str, str]) -> ValueError:
    cause = RuntimeError(values["query"])
    error = ValueError(f"{values['evidence']}|{values['credential']}")
    error.__cause__ = cause
    return error


def _raise_node_error(_state: dict, error: Exception) -> None:
    raise error


async def _raise_node_error_async(_state: dict, error: Exception) -> None:
    raise error


def _raise_step_error(error: Exception) -> None:
    raise error


async def _raise_step_error_async(error: Exception) -> None:
    raise error


def _return_node_value(_state: dict, value: object) -> object:
    return value


async def _return_node_value_async(_state: dict, value: object) -> object:
    return value


def _return_step_value(value: object) -> object:
    return value


async def _return_step_value_async(value: object) -> object:
    return value


class _StringGuardError(Exception):
    def __init__(self) -> None:
        super().__init__()
        self.string_conversion_called = False

    def __str__(self) -> str:
        self.string_conversion_called = True
        raise AssertionError("异常正文不应在安全日志边界被读取")


@contextmanager
def _capture_error_logs() -> Iterator[list[str]]:
    entries: list[str] = []
    sink_id = logger.add(
        lambda message: entries.append(str(message)),
        level="ERROR",
        format="{message}",
        colorize=False,
        enqueue=False,
        backtrace=True,
        diagnose=True,
    )
    try:
        yield entries
    finally:
        logger.remove(sink_id)


@contextmanager
def _capture_log_records() -> Iterator[list[dict]]:
    records: list[dict] = []
    sink_id = logger.add(
        lambda message: records.append(dict(message.record)),
        level="INFO",
        format="{message}",
        colorize=False,
        enqueue=False,
        backtrace=True,
        diagnose=True,
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _assert_no_sensitive_values(entries: list[str], values: dict[str, str]) -> None:
    serialized = "".join(entries)
    if any(value in serialized for value in values.values()):
        pytest.fail("Graph 装饰器日志包含合成敏感异常正文", pytrace=False)


def test_graph_logging_source_forbids_exception_tracebacks() -> None:
    module = inspect.getmodule(node_log)
    assert module is not None
    source = inspect.getsource(module)
    assert "logger.exception(" not in source
    assert "exc_info=" not in source
    assert "str(exc)" not in source
    assert "repr(exc)" not in source


@pytest.mark.parametrize("is_async", [False, True])
def test_node_log_sanitizes_cause_chain_and_reraises_same_exception(is_async: bool) -> None:
    values = _sensitive_values()
    error = _exception_with_sensitive_cause(values)
    target = _raise_node_error_async if is_async else _raise_node_error
    wrapped = node_log("safe_node")(target)

    with _capture_error_logs() as entries, pytest.raises(ValueError) as caught:
        if is_async:
            asyncio.run(wrapped({"trace_id": "trace-safe-001"}, error))
        else:
            wrapped({"trace_id": "trace-safe-001"}, error)

    assert caught.value is error
    assert getattr(error, "origin_node") == "safe_node"
    assert getattr(error, "origin_function") == target.__name__
    assert getattr(error, "origin_exception_type") == "ValueError"
    assert re.fullmatch(r"[0-9a-f]{64}", getattr(error, "origin_fingerprint"))
    assert len(entries) == 1
    _assert_no_sensitive_values(entries, values)
    entry = entries[0]
    assert "stage=node" in entry
    assert "node=safe_node" in entry
    assert "function=" in entry
    assert "exception_type=ValueError" in entry
    assert "trace_id=trace-safe-001" in entry
    assert re.search(r"fingerprint=[0-9a-f]{64}\b", entry)
    assert f"fingerprint={getattr(error, 'origin_fingerprint')}" in entry
    assert "Traceback" not in entry


@pytest.mark.parametrize("is_async", [False, True])
def test_step_log_sanitizes_cause_chain_and_reraises_same_exception(is_async: bool) -> None:
    values = _sensitive_values()
    error = _exception_with_sensitive_cause(values)
    target = _raise_step_error_async if is_async else _raise_step_error
    wrapped = step_log("safe_step")(target)

    with _capture_error_logs() as entries, pytest.raises(ValueError) as caught:
        if is_async:
            asyncio.run(wrapped(error))
        else:
            wrapped(error)

    assert caught.value is error
    assert len(entries) == 1
    _assert_no_sensitive_values(entries, values)
    entry = entries[0]
    assert "stage=step" in entry
    assert "step=safe_step" in entry
    assert "function=" in entry
    assert "exception_type=ValueError" in entry
    assert "correlation_id=-" in entry
    assert re.search(r"fingerprint=[0-9a-f]{64}\b", entry)
    assert "Traceback" not in entry


def test_failure_fingerprint_does_not_depend_on_exception_body() -> None:
    wrapped = step_log("stable_step")(_raise_step_error)
    entries: list[str] = []
    with _capture_error_logs() as captured:
        for suffix in ("first", "second"):
            with pytest.raises(ValueError):
                wrapped(ValueError(suffix))
        entries.extend(captured)

    fingerprints = [re.search(r"fingerprint=([0-9a-f]{64})\b", item) for item in entries]
    assert all(match is not None for match in fingerprints)
    assert len({match.group(1) for match in fingerprints if match is not None}) == 1


def test_node_failure_metadata_rejects_free_text_node_name() -> None:
    values = _sensitive_values()
    error = ValueError(values["query"])
    wrapped = node_log(f"unsafe node {values['credential']}")(_raise_node_error)

    with _capture_error_logs() as entries, pytest.raises(ValueError) as caught:
        wrapped({"trace_id": "trace-safe-002"}, error)

    assert caught.value is error
    assert getattr(error, "origin_node") == "unknown_node"
    assert getattr(error, "origin_function") == "_raise_node_error"
    assert getattr(error, "origin_exception_type") == "ValueError"
    assert re.fullmatch(r"[0-9a-f]{64}", getattr(error, "origin_fingerprint"))
    _assert_no_sensitive_values(entries, values)


def test_safe_failure_logging_never_converts_exception_to_text() -> None:
    error = _StringGuardError()
    wrapped = step_log("string_guard_step")(_raise_step_error)

    with _capture_error_logs() as entries, pytest.raises(_StringGuardError) as caught:
        wrapped(error)

    assert caught.value is error
    assert error.string_conversion_called is False
    assert len(entries) == 1
    assert "exception_type=_StringGuardError" in entries[0]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("should_fail", [False, True])
def test_node_log_keeps_decorated_function_source_context(
    is_async: bool,
    should_fail: bool,
) -> None:
    target = (
        _raise_node_error_async
        if is_async and should_fail
        else _return_node_value_async
        if is_async
        else _raise_node_error
        if should_fail
        else _return_node_value
    )
    wrapped = node_log("source_node")(target)
    value = ValueError("synthetic") if should_fail else object()

    with _capture_log_records() as records:
        if should_fail:
            with pytest.raises(ValueError):
                if is_async:
                    asyncio.run(wrapped({"trace_id": "trace-source-001"}, value))
                else:
                    wrapped({"trace_id": "trace-source-001"}, value)
        elif is_async:
            assert asyncio.run(wrapped({"trace_id": "trace-source-001"}, value)) is value
        else:
            assert wrapped({"trace_id": "trace-source-001"}, value) is value

    assert len(records) == 2
    assert {record["name"] for record in records} == {"test_graph_logging_safety.py"}
    assert {record["function"] for record in records} == {target.__name__}
    assert {record["line"] for record in records} == {target.__code__.co_firstlineno}
    assert all(record["name"] != "contextlib.py" for record in records)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("should_fail", [False, True])
def test_step_log_keeps_decorated_function_source_context(
    is_async: bool,
    should_fail: bool,
) -> None:
    target = (
        _raise_step_error_async
        if is_async and should_fail
        else _return_step_value_async
        if is_async
        else _raise_step_error
        if should_fail
        else _return_step_value
    )
    wrapped = step_log("source_step")(target)
    value = ValueError("synthetic") if should_fail else object()

    with _capture_log_records() as records:
        if should_fail:
            with pytest.raises(ValueError):
                if is_async:
                    asyncio.run(wrapped(value))
                else:
                    wrapped(value)
        elif is_async:
            assert asyncio.run(wrapped(value)) is value
        else:
            assert wrapped(value) is value

    assert len(records) == 2
    assert {record["name"] for record in records} == {"test_graph_logging_safety.py"}
    assert {record["function"] for record in records} == {target.__name__}
    assert {record["line"] for record in records} == {target.__code__.co_firstlineno}
    assert all(record["name"] != "contextlib.py" for record in records)
