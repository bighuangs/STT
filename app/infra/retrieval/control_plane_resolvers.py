"""TODO-09 控制库作用域与结构化元数据过滤解析器。

客户端过滤树只在 SQL 元数据目录中求出文档集合，绝不编译或透传为 Milvus
表达式；向量库表达式只由 :class:`MilvusRetrievalAdapter` 根据服务端 active
facts 构造，因此过滤值即使包含引号或 ``or 1 == 1`` 也只是普通数据。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select

from app.application.index_lifecycle_service import active_index_scopes
from app.domain.query_runtime import canonical_hash
from app.domain.vector_topology import FamilyStatus
from app.domain.retrieval import (
    FilterNode,
    FilterPredicate,
    RetrievalEngineError,
    RetrievalScope,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.models import IndexVersionModel
from app.infra.retrieval.milvus_adapter import ActiveIndexScope


_VECTOR_READ_SET_SCHEMA_VERSION = "vector-read-set.v1"


def serialize_active_scopes(
    scopes: tuple[ActiveIndexScope, ...] | list[ActiveIndexScope],
) -> list[dict[str, Any]]:
    """生成可持久化、顺序稳定的物理读集。

    QueryRun 接受阶段使用该结构冻结 collection/partition；执行阶段只能消费这个
    服务端快照，不能再次读取当前 placement 后悄然切换到另一个物理目标。
    """

    rows = [item.model_dump(mode="json") for item in scopes]
    return sorted(
        rows,
        key=lambda item: (
            item["tenant_id"],
            item["kb_id"],
            item["document_id"],
            item["document_version_id"],
            item["index_version_id"],
            item["collection_name"],
            item.get("partition_name") or "",
        ),
    )


def active_scope_digest(
    scopes: tuple[ActiveIndexScope, ...] | list[ActiveIndexScope],
) -> str:
    """为完整物理读集生成可重放、防篡改摘要。"""

    return canonical_hash(
        {
            "schema_version": _VECTOR_READ_SET_SCHEMA_VERSION,
            "scopes": serialize_active_scopes(scopes),
        }
    )


def freeze_vector_read_set(
    scope: RetrievalScope,
) -> tuple[tuple[ActiveIndexScope, ...], list[dict[str, Any]], str]:
    """解析并冻结一个认证检索范围的物理向量读集。

    在线 QueryRun 与离线评测必须共用这一入口，避免某个调用方只冻结逻辑
    ``index_version_id``，却遗漏真正执行所需的 collection/partition。空读集在
    接受阶段直接失败关闭，不能等到逐 case 执行时才暴露，也不能回退到其他 KB
    或历史索引的 collection。
    """

    resolved = tuple(SqlActiveScopeResolver().resolve(scope))
    if not resolved:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "活动索引没有可冻结的 vector read set",
            retryable=False,
            details={"kb_ids": sorted(scope.kb_ids)},
        )
    rows = serialize_active_scopes(list(resolved))
    return resolved, rows, active_scope_digest(list(resolved))


class FrozenActiveScopeResolver:
    """只返回 QueryRun 接受时冻结的物理目标，并对快照做 fail-closed 校验。"""

    def __init__(
        self,
        rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        expected_digest: str,
    ) -> None:
        try:
            scopes = tuple(ActiveIndexScope.model_validate(item) for item in rows)
        except Exception as exc:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "冻结的 vector read set 结构无效",
                retryable=False,
            ) from exc
        if not scopes:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "冻结的 vector read set 不能为空",
                retryable=False,
            )
        logical_keys = {
            (
                item.tenant_id,
                item.kb_id,
                item.document_id,
                item.document_version_id,
                item.index_version_id,
            )
            for item in scopes
        }
        if len(logical_keys) != len(scopes):
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "冻结的 vector read set 含重复逻辑索引身份",
                retryable=False,
            )
        actual_digest = active_scope_digest(list(scopes))
        if not expected_digest or actual_digest != expected_digest:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "冻结的 vector read set 摘要不匹配",
                retryable=False,
            )
        self._scopes = scopes

    def resolve(self, scope: RetrievalScope) -> tuple[ActiveIndexScope, ...]:
        tenants = {item.tenant_id for item in self._scopes}
        knowledge_bases = {item.kb_id for item in self._scopes}
        if tenants != {scope.tenant_id} or knowledge_bases != set(scope.kb_ids):
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "冻结的 vector read set 与认证检索范围不一致",
                retryable=False,
            )
        return self._scopes


class SqlActiveScopeResolver:
    """把租户/KB 授权范围解析成精确的活动文档版本和索引版本。"""

    def resolve(self, scope: RetrievalScope) -> tuple[ActiveIndexScope, ...]:
        active_facts = [item for kb_id in scope.kb_ids for item in active_index_scopes(kb_id)]
        index_ids = {
            str(item.get("index_version_id") or "")
            for item in active_facts
            if item.get("index_version_id")
        }
        placements: dict[
            tuple[str, str, str],
            list[tuple[VectorPlacementModel, VectorCollectionFamilyModel]],
        ] = defaultdict(list)
        placement_indexes: set[str] = set()
        governance_modes: dict[str, str] = {}
        if index_ids:
            with session_scope() as session:
                governance_modes = {
                    str(index_id): str(governance_mode)
                    for index_id, governance_mode in session.execute(
                        select(
                            IndexVersionModel.id,
                            IndexVersionModel.vector_governance_mode,
                        ).where(IndexVersionModel.id.in_(index_ids))
                    ).all()
                }
                physical_rows = session.execute(
                    select(VectorPlacementModel, VectorCollectionFamilyModel)
                    .join(
                        VectorCollectionFamilyModel,
                        VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
                    )
                    .where(
                        VectorPlacementModel.index_version_id.in_(index_ids),
                    )
                ).all()
                for placement, family in physical_rows:
                    placement_indexes.add(placement.index_version_id)
                    placements[
                        (
                            placement.tenant_id,
                            placement.kb_id,
                            placement.index_version_id,
                        )
                    ].append((placement, family))
        missing_index_ids = index_ids.difference(governance_modes)
        if missing_index_ids:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "活动索引的 vector 治理代际不存在",
                retryable=False,
                details={"index_version_ids": sorted(missing_index_ids)},
            )

        rows: list[ActiveIndexScope] = []
        allowed_kbs = set(scope.kb_ids)
        for item in active_facts:
            kb_id = str(item.get("kb_id") or "")
            index_version_id = str(item.get("index_version_id") or "")
            if str(item.get("tenant_id") or "") != scope.tenant_id or kb_id not in allowed_kbs:
                raise RetrievalEngineError(
                    "RETRIEVAL_STALE_SCOPE",
                    "控制库返回了认证范围外的活动索引",
                    retryable=False,
                    details={"kb_id": kb_id},
                )
            candidates = placements.get(
                (scope.tenant_id, kb_id, index_version_id),
                [],
            )
            if not candidates and index_version_id in placement_indexes:
                raise RetrievalEngineError(
                    "RETRIEVAL_STALE_SCOPE",
                    "vector placement 身份与活动索引不一致",
                    retryable=False,
                    details={"kb_id": kb_id},
                )
            if candidates:
                readable = [
                    (placement, family)
                    for placement, family in candidates
                    if placement.readable and placement.read_slot == "active"
                ]
                if len(readable) != 1:
                    raise RetrievalEngineError(
                        "RETRIEVAL_STALE_SCOPE",
                        "活动索引没有唯一可读 vector placement",
                        retryable=False,
                        details={"kb_id": kb_id},
                    )
                placement, family = readable[0]
                if family.status not in {
                    FamilyStatus.ACTIVE.value,
                    FamilyStatus.DRAINING.value,
                }:
                    raise RetrievalEngineError(
                        "RETRIEVAL_STALE_SCOPE",
                        "可读 vector family 已失效",
                        retryable=False,
                        details={"kb_id": kb_id},
                    )
                if family.physical_database != "default":
                    # 当前 MilvusGateway 绑定单 database；不能静默把 dedicated
                    # database 路由到默认库同名 collection。
                    raise RetrievalEngineError(
                        "RETRIEVAL_UNAVAILABLE",
                        "当前检索客户端不支持该 vector placement database",
                        retryable=False,
                        details={"kb_id": kb_id},
                    )
                collection_name = family.physical_collection
                partition_name = placement.partition_bucket
            else:
                # 明确的旧数据兼容：只有迁移已标记为 legacy 且整个 index 从未
                # 建立 placement 时才回退 TODO-03 的服务端 collection_alias。
                # managed 索引漏建 placement 或存在坏 placement 时一律失败关闭，
                # 避免把新租户的数据重新写/读入共享 kb_chunks。
                if governance_modes.get(index_version_id) != "legacy":
                    raise RetrievalEngineError(
                        "RETRIEVAL_STALE_SCOPE",
                        "managed 活动索引缺少可读 vector placement，拒绝回退共享 collection",
                        retryable=False,
                        details={
                            "kb_id": kb_id,
                            "index_version_id": index_version_id,
                        },
                    )
                collection_name = str(item["collection_name"])
                partition_name = None
            rows.append(
                ActiveIndexScope(
                    tenant_id=scope.tenant_id,
                    kb_id=kb_id,
                    document_id=str(item["document_id"]),
                    document_version_id=str(item["document_version_id"]),
                    index_version_id=index_version_id,
                    collection_name=collection_name,
                    partition_name=partition_name,
                )
            )
        return tuple(rows)


def _compare(values: list[Any], predicate: FilterPredicate) -> bool:
    operator = predicate.operator
    expected = predicate.value
    if operator == "exists":
        return bool(values) is bool(expected)
    if not values:
        return False
    if operator == "eq":
        return any(value == expected for value in values)
    if operator == "ne":
        return all(value != expected for value in values)
    if operator == "in":
        assert isinstance(expected, tuple)
        return any(value in expected for value in values)
    if operator == "not_in":
        assert isinstance(expected, tuple)
        return all(value not in expected for value in values)
    if operator == "contains":
        return any(str(expected) in str(value) for value in values)
    if operator == "starts_with":
        return any(str(value).startswith(str(expected)) for value in values)
    if operator == "ends_with":
        return any(str(value).endswith(str(expected)) for value in values)

    def ordered(value: Any) -> bool:
        try:
            if operator == "gt":
                return value > expected
            if operator == "gte":
                return value >= expected
            if operator == "lt":
                return value < expected
            if operator == "lte":
                return value <= expected
        except TypeError:
            return False
        return False

    return any(ordered(value) for value in values)


class SqlMetadataFilterResolver:
    """在有界 SQL 目录上计算 Filter AST，不扫描 chunk、不接触向量 SDK。"""

    def resolve_document_ids(
        self,
        scope: RetrievalScope,
        filters: FilterNode,
    ) -> tuple[str, ...]:
        active_rows = SqlActiveScopeResolver().resolve(scope)
        universe = {row.document_id for row in active_rows}
        if not universe:
            return ()

        metadata: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        with session_scope() as session:
            repository = RetrievalProfileRepository(
                session, tenant_id=scope.tenant_id, actor_id=scope.user_id
            )
            for kb_id in scope.kb_ids:
                index_ids = {row.index_version_id for row in active_rows if row.kb_id == kb_id}
                document_ids = {row.document_id for row in active_rows if row.kb_id == kb_id}
                for entry in repository.load_metadata_entries(
                    kb_id,
                    index_version_ids=index_ids,
                    document_ids=document_ids,
                ):
                    metadata[entry.document_id][entry.metadata_key].append(entry.value_json)

        def evaluate(node: FilterNode) -> set[str]:
            if isinstance(node, FilterPredicate):
                if node.field == "document_id":
                    return {
                        document_id for document_id in universe if _compare([document_id], node)
                    }
                key = node.field.removeprefix("metadata.")
                return {
                    document_id
                    for document_id in universe
                    if _compare(metadata[document_id].get(key, []), node)
                }

            children = [evaluate(child) for child in node.conditions]
            if node.operator == "not":
                return universe.difference(children[0])
            if node.operator == "or":
                return set().union(*children)
            result = set(universe)
            for child in children:
                result.intersection_update(child)
            return result

        return tuple(sorted(evaluate(filters)))


__all__ = [
    "FrozenActiveScopeResolver",
    "SqlActiveScopeResolver",
    "SqlMetadataFilterResolver",
    "active_scope_digest",
    "freeze_vector_read_set",
    "serialize_active_scopes",
]
