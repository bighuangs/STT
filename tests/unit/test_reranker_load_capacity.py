from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.shared.model import reranker_utils


def _config(*, physical_mib: int = 1536, commit_mib: int = 4096) -> SimpleNamespace:
    return SimpleNamespace(
        bge_reranker_large="./models/test-reranker",
        bge_reranker_device="cpu",
        bge_reranker_fp16=False,
        batch_size=2,
        query_max_tokens=128,
        max_input_tokens=512,
        min_available_physical_mib=physical_mib,
        min_available_commit_mib=commit_mib,
    )


def test_low_capacity_fails_before_flag_reranker_native_initialization(monkeypatch) -> None:
    constructor_calls = 0

    def forbidden_constructor(**kwargs):
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError(f"native constructor must not be called: {kwargs}")

    monkeypatch.setattr(reranker_utils, "_reranker_model", None)
    monkeypatch.setattr(reranker_utils, "reranker_config", _config())
    monkeypatch.setattr(reranker_utils, "FlagReranker", forbidden_constructor)
    monkeypatch.setattr(
        reranker_utils,
        "_memory_capacity_snapshot",
        lambda: reranker_utils.MemoryCapacitySnapshot(
            available_physical=512 * 1024 * 1024,
            available_commit=1024 * 1024 * 1024,
            source="test_snapshot",
        ),
    )

    with pytest.raises(reranker_utils.RerankerCapacityError) as caught:
        reranker_utils.get_reranker_model()

    assert constructor_calls == 0
    assert "available_physical_mib=512" in str(caught.value)
    assert "available_commit_mib=1024" in str(caught.value)
    assert "./models/test-reranker" not in str(caught.value)


def test_sufficient_capacity_allows_flag_reranker_initialization(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(reranker_utils, "_reranker_model", None)
    monkeypatch.setattr(reranker_utils, "reranker_config", _config())
    monkeypatch.setattr(reranker_utils, "FlagReranker", constructor)
    monkeypatch.setattr(
        reranker_utils,
        "_memory_capacity_snapshot",
        lambda: reranker_utils.MemoryCapacitySnapshot(
            available_physical=2048 * 1024 * 1024,
            available_commit=6144 * 1024 * 1024,
            source="test_snapshot",
        ),
    )

    assert reranker_utils.get_reranker_model() is sentinel
    assert calls == [
        {
            "model_name_or_path": "./models/test-reranker",
            "device": "cpu",
            "use_fp16": False,
            "batch_size": 2,
            "query_max_length": 128,
            "max_length": 512,
        }
    ]


def test_loaded_singleton_bypasses_capacity_guard_and_constructor(monkeypatch) -> None:
    sentinel = object()

    def forbidden_capacity_check():
        raise AssertionError("loaded singleton must bypass capacity check")

    def forbidden_constructor(**kwargs):
        raise AssertionError(f"loaded singleton must bypass constructor: {kwargs}")

    monkeypatch.setattr(reranker_utils, "_reranker_model", sentinel)
    monkeypatch.setattr(reranker_utils, "_memory_capacity_snapshot", forbidden_capacity_check)
    monkeypatch.setattr(reranker_utils, "FlagReranker", forbidden_constructor)

    assert reranker_utils.get_reranker_model() is sentinel
