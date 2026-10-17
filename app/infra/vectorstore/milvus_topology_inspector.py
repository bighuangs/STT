"""Milvus 拓扑只读检查器。

公开表面刻意只有 describe/list/query sample，不包含 create/insert/delete/drop/
compact。调用失败会原样抛出，不能把连接超时误判为 collection 不存在。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any

from app.domain.vector_topology import (
    CollectionInventorySnapshot,
    InventoryEntity,
)


IDENTITY_OUTPUT_FIELDS = (
    "chunk_id",
    "id",
    "tenant_id",
    "kb_id",
    "doc_id",
    "document_version_id",
    "index_version_id",
    "available_int",
)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


class MilvusReadOnlyTopologyInspector:
    """在一个已绑定 database 的 MilvusClient 上生成安全 inventory。"""

    def __init__(self, client: Any, *, configured_database: str = "default") -> None:
        if client is None:
            raise RuntimeError("Milvus client is not configured")
        self._client = client
        self._configured_database = configured_database

    def inspect_collection(
        self,
        *,
        physical_database: str,
        physical_collection: str,
        sample_limit: int,
    ) -> CollectionInventorySnapshot:
        if physical_database != self._configured_database:
            raise ValueError("read-only inspector is bound to a different Milvus database")
        if sample_limit < 1 or sample_limit > 10_000:
            raise ValueError("sample_limit must be between 1 and 10000")

        # 不捕获连接/权限/超时异常，确保 readiness 和审计 fail-closed。
        exists = self._client.has_collection(collection_name=physical_collection)
        if not exists:
            raise LookupError(f"Milvus collection does not exist: {physical_collection}")

        description = self._client.describe_collection(collection_name=physical_collection)
        stats = self._client.get_collection_stats(collection_name=physical_collection)
        row_count = self._row_count(stats)
        partitions = tuple(
            sorted(
                self._partition_name(item)
                for item in self._client.list_partitions(collection_name=physical_collection)
            )
        )
        indexes = self._index_descriptions(physical_collection)
        rows = self._client.query(
            collection_name=physical_collection,
            filter="",
            output_fields=list(IDENTITY_OUTPUT_FIELDS),
            limit=sample_limit,
        )
        sampled = tuple(self._identity(item) for item in (rows or ()))
        return CollectionInventorySnapshot(
            physical_database=physical_database,
            physical_collection=physical_collection,
            entity_count=row_count,
            partitions=partitions,
            indexes=indexes,
            schema_snapshot=_json_safe(description),
            sampled_entities=sampled,
            # embedding model/revision/instruction 不是 Milvus schema 事实，必须与
            # SQL compatibility_payload 合并后才能生成完整实际指纹。
            actual_fingerprint=None,
        )

    def _index_descriptions(self, collection_name: str) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        index_names = self._client.list_indexes(collection_name=collection_name)
        for item in index_names or ():
            name = str(_value(item, "index_name", item))
            detail = self._client.describe_index(
                collection_name=collection_name,
                index_name=name,
            )
            result.append(_json_safe(detail))
        return tuple(
            sorted(
                result,
                key=lambda item: str(item.get("index_name") or item.get("field_name")),
            )
        )

    @staticmethod
    def _row_count(stats: Any) -> int:
        raw = _value(stats, "row_count", _value(stats, "num_entities"))
        if raw is None and isinstance(stats, Mapping):
            raw = stats.get("row_count")
        try:
            result = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Milvus collection stats are missing row_count") from exc
        if result < 0:
            raise ValueError("Milvus row_count cannot be negative")
        return result

    @staticmethod
    def _partition_name(item: Any) -> str:
        value = _value(item, "partition_name", _value(item, "name", item))
        result = str(value).strip()
        if not result:
            raise ValueError("Milvus returned a blank partition name")
        return result

    @staticmethod
    def _identity(item: Any) -> InventoryEntity:
        chunk_id = _value(item, "chunk_id") or _value(item, "id")
        available = _value(item, "available_int")
        return InventoryEntity(
            chunk_id=str(chunk_id) if chunk_id not in (None, "") else None,
            tenant_id=(
                str(_value(item, "tenant_id"))
                if _value(item, "tenant_id") not in (None, "")
                else None
            ),
            kb_id=(str(_value(item, "kb_id")) if _value(item, "kb_id") not in (None, "") else None),
            document_id=(
                str(_value(item, "doc_id")) if _value(item, "doc_id") not in (None, "") else None
            ),
            document_version_id=(
                str(_value(item, "document_version_id"))
                if _value(item, "document_version_id") not in (None, "")
                else None
            ),
            index_version_id=(
                str(_value(item, "index_version_id"))
                if _value(item, "index_version_id") not in (None, "")
                else None
            ),
            available=MilvusReadOnlyTopologyInspector._available(available),
        )

    @staticmethod
    def _available(value: Any) -> bool | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        normalized = str(value).strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
        raise ValueError("Milvus available_int is not boolean-like")


__all__ = [
    "IDENTITY_OUTPUT_FIELDS",
    "MilvusReadOnlyTopologyInspector",
]
