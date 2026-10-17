from __future__ import annotations

from typing import Any

import pytest

from app.infra.vectorstore.write_compatibility import current_chunks_profile
from app.rag.import_ import index_service


class _Schema:
    def add_field(self, **_: Any) -> None:
        return None


class _IndexParams:
    def add_index(self, **_: Any) -> None:
        return None


class _CreateClient:
    def __init__(self) -> None:
        self.created = False

    def has_collection(self, *args: Any, **kwargs: Any) -> bool:
        return self.created

    def create_schema(self, **_: Any) -> _Schema:
        return _Schema()

    def prepare_index_params(self) -> _IndexParams:
        return _IndexParams()

    def create_collection(self, **_: Any) -> None:
        self.created = True


def test_managed_first_write_creates_only_from_exact_persisted_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _CreateClient()
    collection = "stt_chunks_k_managed_exact"
    index_service._verified_collections.discard(collection)
    monkeypatch.setattr(index_service.milvus_gateway, "client", lambda: client)
    profile = current_chunks_profile(
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3",
    )

    index_service.prepare_chunks_collection(
        collection,
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3",
        managed_profile=profile,
    )

    assert client.created is True


def test_managed_contract_drift_fails_before_milvus_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _CreateClient()
    collection = "stt_chunks_k_managed_drift"
    index_service._verified_collections.discard(collection)
    monkeypatch.setattr(index_service.milvus_gateway, "client", lambda: client)
    stale_profile = current_chunks_profile(
        vector_dim=4,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3",
    )

    with pytest.raises(Exception, match="compatibility|drift"):
        index_service.prepare_chunks_collection(
            collection,
            vector_dim=3,
            schema_version="chunks_v1",
            embedding_model_id="bge-m3",
            managed_profile=stale_profile,
        )

    assert client.created is False
