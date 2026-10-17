from __future__ import annotations

import hashlib

import pytest

from evals.ragas_support.materializer import (
    ContextMaterializationError,
    ObservationSource,
    materialize_judge_observations,
)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample(*, sensitivity: str = "internal", eligible: bool = True) -> dict:
    return {
        "case_id": "case-1",
        "sensitivity": sensitivity,
        "judge_eligible": eligible,
    }


def _observation(texts: list[str]) -> dict:
    return {
        "case_id": "case-1",
        "observation": {
            "answer": "系统答案",
            "context": [
                {
                    "chunk_id": f"chunk-{index}",
                    "content_hash": _hash(text),
                    "document_version_id": "version-1",
                }
                for index, text in enumerate(texts, start=1)
            ],
        },
    }


class _FakeClient:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    def query(self, **_: object) -> list[dict]:
        self.calls += 1
        return [
            {
                "source_chunk_id": f"chunk-{index}",
                "content_with_weight": text,
                "content_hash": _hash(text),
                "document_version_id": "version-1",
            }
            for index, text in enumerate(self.texts, start=1)
        ]


def test_materializer_preserves_rank_order_and_verifies_content_hashes() -> None:
    texts = ["证据一", "证据二"]
    client = _FakeClient(texts)

    rows = materialize_judge_observations(
        [_sample()],
        [ObservationSource("collection-1", [_observation(texts)])],
        case_ids=["case-1"],
        client=client,
    )

    assert rows == [
        {
            "case_id": "case-1",
            "response": "系统答案",
            "retrieved_context_ids": ["chunk-1", "chunk-2"],
            "retrieved_contexts": texts,
        }
    ]
    assert client.calls == 1


@pytest.mark.parametrize(
    ("sensitivity", "eligible"),
    [("restricted", True), ("protected", True), ("internal", False)],
)
def test_materializer_rejects_non_eligible_cases_before_query(
    sensitivity: str,
    eligible: bool,
) -> None:
    client = _FakeClient(["证据"])

    with pytest.raises(ContextMaterializationError, match="ineligible"):
        materialize_judge_observations(
            [_sample(sensitivity=sensitivity, eligible=eligible)],
            [ObservationSource("collection-1", [_observation(["证据"])])],
            case_ids=["case-1"],
            client=client,
        )

    assert client.calls == 0


def test_materializer_fails_visible_on_hash_mismatch() -> None:
    client = _FakeClient(["被篡改的证据"])

    with pytest.raises(ContextMaterializationError, match="missing"):
        materialize_judge_observations(
            [_sample()],
            [ObservationSource("collection-1", [_observation(["原始证据"])])],
            case_ids=["case-1"],
            client=client,
        )
