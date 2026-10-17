from __future__ import annotations

import pytest

from app.infra.vectorstore.milvus_topology_inspector import (
    IDENTITY_OUTPUT_FIELDS,
    MilvusReadOnlyTopologyInspector,
)


class _ReadOnlyClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.calls: list[tuple[str, dict]] = []

    def has_collection(self, **kwargs):
        self.calls.append(("has_collection", kwargs))
        return self.exists

    def describe_collection(self, **kwargs):
        self.calls.append(("describe_collection", kwargs))
        return {"fields": [{"name": "tenant_id", "type": "VARCHAR"}]}

    def get_collection_stats(self, **kwargs):
        self.calls.append(("get_collection_stats", kwargs))
        return {"row_count": "2"}

    def list_partitions(self, **kwargs):
        self.calls.append(("list_partitions", kwargs))
        return ["p_0001", "_default"]

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        return ["dense_idx"]

    def describe_index(self, **kwargs):
        self.calls.append(("describe_index", kwargs))
        return {
            "index_name": kwargs["index_name"],
            "field_name": "dense_vector",
            "metric_type": "IP",
        }

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return [
            {
                "id": "chunk_1",
                "tenant_id": "tenant_a",
                "kb_id": "kb_a",
                "doc_id": "doc_a",
                "document_version_id": "dv_a",
                "index_version_id": "idx_a",
                "available_int": 1,
                # 即使驱动意外返回额外字段，InventoryEntity 也不会保留正文。
                "content": "secret body",
            }
        ]


def test_inspector_exposes_only_read_operations_and_identity_fields():
    client = _ReadOnlyClient()
    inspector = MilvusReadOnlyTopologyInspector(client)
    snapshot = inspector.inspect_collection(
        physical_database="default",
        physical_collection="kb_chunks",
        sample_limit=50,
    )
    assert snapshot.entity_count == 2
    assert snapshot.partitions == ("_default", "p_0001")
    assert snapshot.sampled_entities[0].chunk_id == "chunk_1"
    assert "content" not in snapshot.sampled_entities[0].model_dump()
    assert {name for name, _ in client.calls} == {
        "has_collection",
        "describe_collection",
        "get_collection_stats",
        "list_partitions",
        "list_indexes",
        "describe_index",
        "query",
    }
    query = next(payload for name, payload in client.calls if name == "query")
    assert query["output_fields"] == list(IDENTITY_OUTPUT_FIELDS)
    assert MilvusReadOnlyTopologyInspector._available("0") is False
    assert MilvusReadOnlyTopologyInspector._available("1") is True


def test_inspector_does_not_treat_unavailable_as_missing():
    class _Unavailable(_ReadOnlyClient):
        def has_collection(self, **kwargs):
            raise TimeoutError("Milvus timeout")

    inspector = MilvusReadOnlyTopologyInspector(_Unavailable())
    with pytest.raises(TimeoutError, match="timeout"):
        inspector.inspect_collection(
            physical_database="default",
            physical_collection="kb_chunks",
            sample_limit=10,
        )


def test_inspector_rejects_database_switch_and_missing_collection():
    inspector = MilvusReadOnlyTopologyInspector(_ReadOnlyClient())
    with pytest.raises(ValueError, match="different"):
        inspector.inspect_collection(
            physical_database="tenant_other",
            physical_collection="kb_chunks",
            sample_limit=10,
        )
    with pytest.raises(LookupError, match="does not exist"):
        MilvusReadOnlyTopologyInspector(_ReadOnlyClient(exists=False)).inspect_collection(
            physical_database="default",
            physical_collection="kb_chunks",
            sample_limit=10,
        )
