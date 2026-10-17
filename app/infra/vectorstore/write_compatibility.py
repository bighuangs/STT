"""Milvus 写入目标解析与物理兼容性门禁。

该模块位于真实 ``insert/delete`` 之前，解决两个容易被混淆的问题：

* SQL placement 决定“写到哪里”，调用方不能通过 state 覆盖物理 collection；
* Milvus ``describe_collection/describe_index/list_partitions`` 决定“那里是否仍兼容”。

历史 ``kb_chunks``/版本化 collection 没有 ``vector_placements`` 记录。它们只能走
显式 ``CONTROL_PLANE_LEGACY`` 策略，而且 collection 必须与 SQL IndexVersion 的
服务端 alias 一致；这不是对任意客户端 collection 名的兼容开关。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from sqlalchemy import select

from app.domain.vector_topology import (
    CompatibilityFingerprint,
    CompatibilityProfile,
    DenseVectorContract,
    FamilyStatus,
    PartitionContract,
    ScalarFieldContract,
    SparseVectorContract,
    compatibility_fingerprint,
    require_compatible,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import IndexVersionModel, KnowledgeBaseModel
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorPlacementModel,
)


class VectorWriteGateError(RuntimeError):
    """写入目标或物理结构不能被证明安全。"""


class LegacyWritePolicy(StrEnum):
    DENY = "deny"
    CONTROL_PLANE_LEGACY = "control_plane_legacy"


@dataclass(frozen=True)
class ResolvedVectorWriteTarget:
    physical_database: str
    physical_collection: str
    partition_bucket: str | None
    expected_profile: CompatibilityProfile | None
    expected_fingerprint: CompatibilityFingerprint | None
    placement_id: str | None
    placement_epoch: int | None
    legacy_policy: LegacyWritePolicy
    legacy_reason: str | None = None

    @property
    def managed(self) -> bool:
        return self.placement_id is not None


@dataclass(frozen=True)
class VectorWriteGateResult:
    physical_database: str
    physical_collection: str
    compatibility_fingerprint: CompatibilityFingerprint
    managed: bool
    legacy_policy: LegacyWritePolicy
    inspected_partitions: tuple[str, ...]


_TYPE_NAMES = {
    1: "BOOL",
    2: "INT8",
    3: "INT16",
    4: "INT32",
    5: "INT64",
    10: "FLOAT",
    11: "DOUBLE",
    20: "STRING",
    21: "VARCHAR",
    23: "JSON",
    101: "FLOAT_VECTOR",
    102: "BINARY_VECTOR",
    103: "FLOAT16_VECTOR",
    104: "SPARSE_FLOAT_VECTOR",
}

_CURRENT_SCALAR_FIELDS: dict[str, tuple[str, int | None, bool]] = {
    "chunk_id": ("INT64", None, True),
    "doc_id": ("VARCHAR", 4096, False),
    "kb_id": ("VARCHAR", 4096, False),
    "tenant_id": ("VARCHAR", 4096, False),
    "document_version_id": ("VARCHAR", 4096, False),
    "index_version_id": ("VARCHAR", 4096, False),
    "evidence_id": ("VARCHAR", 64, False),
    "content_hash": ("VARCHAR", 80, False),
    "source_chunk_id": ("VARCHAR", 128, False),
    "location_status": ("VARCHAR", 16, False),
    "primary_page_no": ("INT64", None, False),
    "docnm_kwd": ("VARCHAR", 4096, False),
    "content_with_weight": ("VARCHAR", 65535, False),
    "type": ("VARCHAR", 32, False),
    "img_id": ("VARCHAR", 4096, False),
    "id": ("VARCHAR", 64, False),
    "create_time": ("VARCHAR", 32, False),
    "create_timestamp_flt": ("DOUBLE", None, False),
    "available_int": ("INT64", None, False),
    "position_int": ("INT64", None, False),
    "important_kwd": ("VARCHAR", 4096, False),
    "important_tks": ("VARCHAR", 4096, False),
    "question_kwd": ("VARCHAR", 4096, False),
    "question_tks": ("VARCHAR", 4096, False),
    "meta_fields": ("VARCHAR", 4096, False),
    "tag_kwd": ("VARCHAR", 4096, False),
}

_LEGACY_REQUIRED_FIELDS = frozenset(
    {
        "chunk_id",
        "doc_id",
        "kb_id",
        "docnm_kwd",
        "content_with_weight",
        "dense_vector",
        "sparse_vector",
        "type",
        "img_id",
        "id",
        "create_time",
        "create_timestamp_flt",
        "available_int",
        "position_int",
        "important_kwd",
        "important_tks",
        "question_kwd",
        "question_tks",
        "meta_fields",
        "tag_kwd",
    }
)
_KNOWN_FIELDS = frozenset((*_CURRENT_SCALAR_FIELDS, "dense_vector", "sparse_vector"))


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalize_type(value: Any) -> str:
    raw = getattr(value, "value", value)
    if isinstance(raw, int):
        return _TYPE_NAMES.get(raw, f"UNKNOWN_{raw}")
    text = str(raw).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text.isdigit():
        return _TYPE_NAMES.get(int(text), f"UNKNOWN_{text}")
    return text


def _field_params(field: Any) -> Mapping[str, Any]:
    params = _value(field, "params", {}) or {}
    type_params = _value(field, "type_params", {}) or {}
    merged: dict[str, Any] = {}
    if isinstance(params, Mapping):
        merged.update(params)
    if isinstance(type_params, Mapping):
        merged.update(type_params)
    return merged


def _integer_param(field: Any, name: str) -> int | None:
    raw = _value(field, name)
    if raw is None:
        raw = _field_params(field).get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise VectorWriteGateError(f"Milvus 字段 {name} 不是整数") from exc


def _description_fields(description: Any) -> dict[str, Any]:
    raw_fields = _value(description, "fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise VectorWriteGateError("Milvus describe_collection 缺少 fields")
    result: dict[str, Any] = {}
    for field in raw_fields:
        name = str(_value(field, "name", _value(field, "field_name", ""))).strip()
        if not name or name in result:
            raise VectorWriteGateError("Milvus schema 含空白或重复字段")
        result[name] = field
    return result


def _partition_names(client: Any, collection_name: str) -> tuple[str, ...]:
    try:
        raw = client.list_partitions(collection_name=collection_name)
    except TypeError:
        raw = client.list_partitions(collection_name)
    names = []
    for item in raw or ():
        name = str(_value(item, "partition_name", _value(item, "name", item))).strip()
        if not name:
            raise VectorWriteGateError("Milvus 返回空 partition 名")
        names.append(name)
    if not names:
        raise VectorWriteGateError("Milvus collection 没有可证明的 partition")
    return tuple(sorted(set(names)))


def _index_details(client: Any, collection_name: str) -> dict[str, dict[str, Any]]:
    try:
        names = client.list_indexes(collection_name=collection_name)
    except TypeError:
        names = client.list_indexes(collection_name)
    result: dict[str, dict[str, Any]] = {}
    for item in names or ():
        index_name = str(_value(item, "index_name", item)).strip()
        if not index_name:
            raise VectorWriteGateError("Milvus 返回空 index 名")
        try:
            detail = client.describe_index(
                collection_name=collection_name,
                index_name=index_name,
            )
        except TypeError:
            detail = client.describe_index(collection_name, index_name)
        field_name = str(_value(detail, "field_name", "")).strip()
        params = _value(detail, "params", {}) or {}
        if not isinstance(params, Mapping):
            raise VectorWriteGateError(f"Milvus index {index_name} params 非对象")
        normalized_params = dict(params)
        index_param = _value(detail, "index_param", {}) or {}
        if isinstance(index_param, Mapping):
            normalized_params.update(index_param)
        index_type = _value(detail, "index_type", normalized_params.pop("index_type", ""))
        metric = _value(
            detail,
            "metric_type",
            _value(detail, "metric", normalized_params.pop("metric_type", "")),
        )
        # SDK 有时把构建参数再包一层 ``params``。
        nested = normalized_params.pop("params", None)
        if isinstance(nested, Mapping):
            normalized_params.update(nested)
        for parameter_name in (
            "inverted_index_algo",
            "drop_ratio_build",
            "nlist",
            "M",
            "efConstruction",
        ):
            value = _value(detail, parameter_name)
            if value is not None:
                normalized_params[parameter_name] = value
        if not field_name or field_name in result:
            raise VectorWriteGateError("Milvus vector field 缺少唯一 index")
        result[field_name] = {
            "index_name": index_name,
            "index_type": str(index_type).strip().upper(),
            "metric": str(metric).strip().upper(),
            "params": normalized_params,
        }
    return result


def _dynamic_enabled(description: Any) -> bool:
    raw = _value(
        description,
        "enable_dynamic_field",
        _value(description, "enable_dynamic_fields", False),
    )
    return bool(raw)


def current_chunks_profile(
    *,
    vector_dim: int,
    schema_version: str,
    embedding_model_id: str,
    scalar_names: Sequence[str] | None = None,
    dense_index_type: str = "AUTOINDEX",
    dense_metric: str = "IP",
    dense_index_params: Mapping[str, Any] | None = None,
    sparse_index_type: str = "SPARSE_INVERTED_INDEX",
    sparse_metric: str = "IP",
    sparse_index_params: Mapping[str, Any] | None = None,
) -> CompatibilityProfile:
    """构造当前写链完整兼容合同；legacy 可传远端已知字段子集。"""

    selected = tuple(sorted(scalar_names or _CURRENT_SCALAR_FIELDS))
    unknown = set(selected).difference(_CURRENT_SCALAR_FIELDS)
    if unknown:
        raise VectorWriteGateError(f"未知 legacy scalar 字段: {sorted(unknown)}")
    scalar_fields = tuple(
        ScalarFieldContract(
            name=name,
            data_type=_CURRENT_SCALAR_FIELDS[name][0],
            primary_key=_CURRENT_SCALAR_FIELDS[name][2],
            nullable=False,
            max_length=_CURRENT_SCALAR_FIELDS[name][1],
        )
        for name in selected
    )
    model_id = embedding_model_id.strip() or "bge-m3"
    schema = schema_version.strip() or "chunks_v1"
    return CompatibilityProfile(
        schema_version=schema,
        chunk_identity_schema_version="chunk-id.v1",
        evidence_identity_schema_version="evidence.v1",
        dynamic_fields_enabled=True,
        scalar_fields=scalar_fields,
        dense=DenseVectorContract(
            field_name="dense_vector",
            dimension=vector_dim,
            provider="configured",
            model_id=model_id,
            revision=model_id,
            tokenizer=model_id,
            normalization="l2",
            metric=dense_metric,
            index_type=dense_index_type,
            index_params=dict(dense_index_params or {}),
        ),
        sparse=SparseVectorContract(
            field_name="sparse_vector",
            algorithm=f"{model_id}:lexical",
            analyzer=model_id,
            metric=sparse_metric,
            index_type=sparse_index_type,
            index_params=dict(
                sparse_index_params
                if sparse_index_params is not None
                else {"inverted_index_algo": "DAAT_MAXSCORE"}
            ),
        ),
        partition=PartitionContract(strategy="none", num_partitions=1),
        preprocessing_version=schema,
    )


def _observed_profile(
    *,
    expected: CompatibilityProfile,
    description: Any,
    indexes: Mapping[str, Mapping[str, Any]],
    partitions: tuple[str, ...],
    legacy: bool,
) -> CompatibilityProfile:
    fields = _description_fields(description)
    names = set(fields)
    if not names <= _KNOWN_FIELDS:
        raise VectorWriteGateError(f"Milvus schema 含未授权字段: {sorted(names - _KNOWN_FIELDS)}")
    if legacy:
        if not _LEGACY_REQUIRED_FIELDS <= names:
            raise VectorWriteGateError(
                f"legacy Milvus schema 缺少必需字段: {sorted(_LEGACY_REQUIRED_FIELDS - names)}"
            )
        scalar_names = sorted(names.difference({"dense_vector", "sparse_vector"}))
    else:
        expected_names = {item.name for item in expected.scalar_fields} | {
            expected.dense.field_name
        }
        if expected.sparse is not None:
            expected_names.add(expected.sparse.field_name)
        if names != expected_names:
            raise VectorWriteGateError(
                "managed Milvus schema 字段漂移: "
                f"missing={sorted(expected_names - names)}, extra={sorted(names - expected_names)}"
            )
        scalar_names = sorted(item.name for item in expected.scalar_fields)

    observed_scalars: list[ScalarFieldContract] = []
    expected_scalars = {item.name: item for item in expected.scalar_fields}
    for name in scalar_names:
        field = fields[name]
        contract = expected_scalars.get(name)
        if contract is None:
            known_type, known_length, primary = _CURRENT_SCALAR_FIELDS[name]
            contract = ScalarFieldContract(
                name=name,
                data_type=known_type,
                max_length=known_length,
                primary_key=primary,
            )
        # ``MilvusClient.describe_collection`` 的字段类型键随 SDK 版本存在三种
        # 等价形态：``data_type``、``datatype`` 与 ``type``。当前 pymilvus
        # 返回的是 ``{"type": DataType.VARCHAR}``；若只读取前两种键，所有
        # 字段都会被归一化为空字符串，进而把同一个物理 schema 误判为整体漂移。
        # 这里只扩展输入形态，后续仍统一交给 ``_normalize_type`` 严格比较，
        # 未知枚举和值依然 fail-closed，不会放宽真实类型漂移。
        data_type = _normalize_type(
            _value(
                field,
                "data_type",
                _value(field, "datatype", _value(field, "type", "")),
            )
        )
        max_length = _integer_param(field, "max_length")
        if data_type != "VARCHAR":
            max_length = None
        observed_scalars.append(
            contract.model_copy(
                update={
                    "data_type": data_type,
                    "max_length": max_length,
                    "primary_key": bool(
                        _value(
                            field,
                            "is_primary",
                            _value(field, "primary_key", contract.primary_key),
                        )
                    ),
                    "nullable": bool(_value(field, "nullable", contract.nullable)),
                }
            )
        )

    dense_field = fields.get(expected.dense.field_name)
    if dense_field is None:
        raise VectorWriteGateError("Milvus dense vector 字段缺失")
    dense_index = indexes.get(expected.dense.field_name)
    if dense_index is None:
        raise VectorWriteGateError("Milvus dense vector index 缺失")
    dense = expected.dense.model_copy(
        update={
            "dimension": _integer_param(dense_field, "dim") or 0,
            "metric": dense_index["metric"],
            "index_type": dense_index["index_type"],
            "index_params": dict(dense_index["params"]),
        }
    )

    sparse = None
    if expected.sparse is not None:
        sparse_field = fields.get(expected.sparse.field_name)
        if sparse_field is None:
            raise VectorWriteGateError("Milvus sparse vector 字段缺失")
        sparse_index = indexes.get(expected.sparse.field_name)
        if sparse_index is None:
            raise VectorWriteGateError("Milvus sparse vector index 缺失")
        sparse = expected.sparse.model_copy(
            update={
                "metric": sparse_index["metric"],
                "index_type": sparse_index["index_type"],
                "index_params": dict(sparse_index["params"]),
            }
        )

    partition = expected.partition
    if partition.strategy == "none":
        if len(partitions) != 1:
            raise VectorWriteGateError("无分区合同对应的 Milvus partition 数不为 1")
    elif len(partitions) != partition.num_partitions:
        raise VectorWriteGateError(
            "Milvus partition 数与兼容合同不一致: "
            f"expected={partition.num_partitions}, actual={len(partitions)}"
        )

    return expected.model_copy(
        update={
            "dynamic_fields_enabled": _dynamic_enabled(description),
            "scalar_fields": tuple(observed_scalars),
            "dense": dense,
            "sparse": sparse,
        }
    )


class MilvusWriteCompatibilityGate:
    """每次持久导入写入前都重新读取物理事实，不缓存漂移判断。"""

    def __init__(self, client: Any, *, configured_database: str = "default") -> None:
        if client is None:
            raise VectorWriteGateError("Milvus client 未配置")
        self._client = client
        self._database = configured_database

    def verify(
        self,
        target: ResolvedVectorWriteTarget,
        *,
        vector_dim: int,
        schema_version: str,
        embedding_model_id: str,
    ) -> VectorWriteGateResult:
        if target.physical_database != self._database:
            raise VectorWriteGateError("Milvus client 与 placement database 不一致")
        if not self._client.has_collection(collection_name=target.physical_collection):
            raise VectorWriteGateError("目标 Milvus collection 不存在")
        # 任何 describe/index/partition 调用异常都向上抛出；网络短停不能被解释为兼容。
        description = self._client.describe_collection(collection_name=target.physical_collection)
        indexes = _index_details(self._client, target.physical_collection)
        partitions = _partition_names(self._client, target.physical_collection)

        if target.managed:
            if target.expected_profile is None or target.expected_fingerprint is None:
                raise VectorWriteGateError("managed placement 缺少持久兼容合同")
            expected_profile = target.expected_profile
            expected_fingerprint = target.expected_fingerprint
            # 防止 SQL payload 与其三层 digest 被分别篡改。
            generated = compatibility_fingerprint(expected_profile)
            require_compatible(expected_fingerprint, generated)
            legacy = False
        else:
            if target.legacy_policy != LegacyWritePolicy.CONTROL_PLANE_LEGACY:
                raise VectorWriteGateError("未管理 collection 的 legacy 写入策略为 deny")
            fields = _description_fields(description)
            scalar_names = sorted(set(fields).intersection(_CURRENT_SCALAR_FIELDS))
            expected_profile = current_chunks_profile(
                vector_dim=vector_dim,
                schema_version=schema_version,
                embedding_model_id=embedding_model_id,
                scalar_names=scalar_names,
            )
            expected_fingerprint = compatibility_fingerprint(expected_profile)
            legacy = True

        observed_profile = _observed_profile(
            expected=expected_profile,
            description=description,
            indexes=indexes,
            partitions=partitions,
            legacy=legacy,
        )
        observed_fingerprint = compatibility_fingerprint(observed_profile)
        require_compatible(expected_fingerprint, observed_fingerprint)
        if target.partition_bucket is not None and target.partition_bucket not in partitions:
            raise VectorWriteGateError("placement partition 不存在于 Milvus collection")
        return VectorWriteGateResult(
            physical_database=target.physical_database,
            physical_collection=target.physical_collection,
            compatibility_fingerprint=observed_fingerprint,
            managed=target.managed,
            legacy_policy=target.legacy_policy,
            inspected_partitions=partitions,
        )


def resolve_import_vector_target(
    state: Mapping[str, Any],
    *,
    fallback_collection: str,
) -> ResolvedVectorWriteTarget:
    """用 SQL job/index/placement 事实解析导入写目标。

    只有持久 Worker state（带 job/index/tenant/kb）会进入此函数。旧的离线单测或
    手工工具没有完整控制面身份时，应由调用方显式选择是否跳过门禁。
    """

    tenant_id = str(state.get("tenant_id") or "").strip()
    kb_id = str(state.get("kb_id") or "").strip()
    index_version_id = str(state.get("index_version_id") or "").strip()
    requested_collection = str(state.get("collection_name") or fallback_collection).strip()
    if not all((tenant_id, kb_id, index_version_id, requested_collection)):
        raise VectorWriteGateError("持久导入缺少 tenant/kb/index/collection 身份")

    with session_scope() as session:
        index = session.scalar(
            select(IndexVersionModel)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == IndexVersionModel.kb_id)
            .where(
                IndexVersionModel.id == index_version_id,
                IndexVersionModel.kb_id == kb_id,
                KnowledgeBaseModel.tenant_id == tenant_id,
            )
        )
        if index is None:
            raise VectorWriteGateError("导入 index version 不属于 tenant/kb")
        if requested_collection != index.collection_alias:
            raise VectorWriteGateError("Worker state collection 与 SQL index alias 不一致")

        placement_rows = session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(
                VectorPlacementModel.tenant_id == tenant_id,
                VectorPlacementModel.kb_id == kb_id,
                VectorPlacementModel.index_version_id == index_version_id,
            )
        ).all()
        if not placement_rows:
            if index.vector_governance_mode != "legacy":
                raise VectorWriteGateError(
                    "managed index 缺少 vector placement，拒绝回退 legacy collection"
                )
            return ResolvedVectorWriteTarget(
                physical_database="default",
                physical_collection=index.collection_alias,
                partition_bucket=None,
                expected_profile=None,
                expected_fingerprint=None,
                placement_id=None,
                placement_epoch=None,
                legacy_policy=LegacyWritePolicy.CONTROL_PLANE_LEGACY,
                legacy_reason="migration_marked_pre_control_plane_index",
            )

        writable = [
            (placement, family)
            for placement, family in placement_rows
            if placement.writable and placement.write_slot == "active"
        ]
        if len(writable) != 1:
            raise VectorWriteGateError("存在 placement 记录但没有唯一 active writable placement")
        placement, family = writable[0]
        if family.status != FamilyStatus.ACTIVE.value:
            raise VectorWriteGateError("writable vector family 不是 active")
        profile = CompatibilityProfile.model_validate(family.compatibility_payload)
        persisted_fingerprint = CompatibilityFingerprint(
            compatibility=family.compatibility_fingerprint,
            schema_fingerprint=family.schema_fingerprint,
            index_fingerprint=family.index_fingerprint,
            canonical_payload=family.compatibility_payload,
        )
        return ResolvedVectorWriteTarget(
            physical_database=family.physical_database,
            physical_collection=family.physical_collection,
            partition_bucket=placement.partition_bucket,
            expected_profile=profile,
            expected_fingerprint=persisted_fingerprint,
            placement_id=placement.id,
            placement_epoch=placement.epoch,
            legacy_policy=LegacyWritePolicy.DENY,
        )


__all__ = [
    "LegacyWritePolicy",
    "MilvusWriteCompatibilityGate",
    "ResolvedVectorWriteTarget",
    "VectorWriteGateError",
    "VectorWriteGateResult",
    "current_chunks_profile",
    "resolve_import_vector_target",
]
