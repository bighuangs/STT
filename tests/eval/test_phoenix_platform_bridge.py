"""Phoenix 平台桥接只测试纯映射与授权门禁，不发起网络请求。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.application.evaluation_target import EvaluationRequest
from evals.platforms.phoenix import (
    build_experiment_task,
    build_phoenix_examples,
    publish_dataset,
    run_experiment,
)


def _case(case_id: str = "case-1") -> dict:
    return {
        "case_id": case_id,
        "query": "什么是 RAG？",
        "reference_answer": "检索增强生成",
        "expected_terms": ["检索", "生成"],
        "difficulty": "easy",
        "category": "definition",
    }


def test_build_examples_uses_stable_ids_and_provenance() -> None:
    examples = build_phoenix_examples(
        [_case()],
        dataset_version="v1",
        dataset_checksum="sha256:abc",
    )

    assert examples == [
        {
            "id": "case-1",
            "input": {
                "case": {
                    "case_id": "case-1",
                    "query": "什么是 RAG？",
                }
            },
            "output": {
                "reference_answer": "检索增强生成",
                "expected_terms": ["检索", "生成"],
            },
            "metadata": {
                "case_id": "case-1",
                "dataset_version": "v1",
                "dataset_checksum": "sha256:abc",
                "difficulty": "easy",
                "category": "definition",
            },
        }
    ]


def test_build_examples_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError, match="duplicate case_id"):
        build_phoenix_examples(
            [_case(), _case()],
            dataset_version="v1",
            dataset_checksum="sha256:abc",
        )


def test_publish_dataset_accepts_injected_client_without_network() -> None:
    calls: list[dict] = []

    class Datasets:
        def create_dataset(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(name=kwargs["name"], version_id="version-1", example_count=1)

    receipt = publish_dataset(
        name="stt-golden",
        description="离线黄金集",
        cases=[_case()],
        dataset_version="v1",
        dataset_checksum="sha256:abc",
        client=SimpleNamespace(datasets=Datasets()),
    )

    assert receipt == {
        "name": "stt-golden",
        "version_id": "version-1",
        "example_count": 1,
    }
    assert calls[0]["examples"][0]["id"] == "case-1"


def test_examples_match_installed_phoenix_client_contract(monkeypatch) -> None:
    """通过官方客户端的公开入口校验 example 形状，但替换最终 HTTP 上传。"""

    from phoenix.client import Client

    dataset_api = Client(base_url="http://127.0.0.1:6006").datasets
    captured: dict = {}
    sentinel = object()

    def fake_upload(_self, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(dataset_api._guard, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(type(dataset_api), "_upload_json_dataset", fake_upload)

    result = dataset_api.create_dataset(
        name="stt-local-contract-check",
        examples=build_phoenix_examples(
            [_case()],
            dataset_version="v1",
            dataset_checksum="sha256:abc",
        ),
    )

    assert result is sentinel
    assert captured["inputs"] == [{"case": {"case_id": "case-1", "query": "什么是 RAG？"}}]
    assert captured["outputs"] == [
        {"reference_answer": "检索增强生成", "expected_terms": ["检索", "生成"]}
    ]


def test_experiment_task_calls_only_stable_target_contract() -> None:
    requests: list[EvaluationRequest] = []

    class Target:
        def evaluate(self, request: EvaluationRequest) -> dict:
            requests.append(request)
            return {"answer": "Retrieval-Augmented Generation"}

    task = build_experiment_task(
        Target(),
        kb_id="kb-1",
        run_id="run-1",
        config={"config_version": "cfg-1"},
    )

    target_case = {"case_id": "case-1", "query": "什么是 RAG？"}
    assert task({"case": target_case}) == {"answer": "Retrieval-Augmented Generation"}
    assert requests[0].case["case_id"] == "case-1"
    assert requests[0].kb_id == "kb-1"


def test_experiment_execution_requires_explicit_authorization() -> None:
    with pytest.raises(PermissionError, match="explicit authorization"):
        run_experiment(
            dataset=object(),
            target=SimpleNamespace(evaluate=lambda request: {}),
            kb_id="kb-1",
            run_id="run-1",
            config={},
            experiment_name="not-run",
        )
