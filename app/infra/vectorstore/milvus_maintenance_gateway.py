"""默认关闭 mutation 的真实 Milvus maintenance adapter。

GC 的 count/sample/delete 可直接用 MilvusClient 完成。migration copy/validation
必须由部署注入批复制和检索探针回调；缺少回调时 fail-closed，绝不把“能 insert”
误当成“已验证可切读”。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any

from app.application.vector_maintenance_service import (
    CopyReceipt,
    DeleteReceipt,
    MigrationExecutionContext,
    VectorMaintenanceError,
)
from app.domain.vector_topology import (
    CompatibilityFingerprint,
    CompatibilityProfile,
    MigrationValidation,
    VectorGcPlan,
    VectorIdentity,
)
from app.infra.vectorstore.write_compatibility import (
    LegacyWritePolicy,
    MilvusWriteCompatibilityGate,
    ResolvedVectorWriteTarget,
)


MigrationCopyExecutor = Callable[
    [Any, MigrationExecutionContext, str, int],
    CopyReceipt,
]
MigrationValidationExecutor = Callable[
    [Any, MigrationExecutionContext, CopyReceipt],
    MigrationValidation,
]


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _index_scope_filter(context: MigrationExecutionContext) -> str:
    scope = context.scope
    return " and ".join(
        (
            f"tenant_id == {_quoted(scope.tenant_id)}",
            f"kb_id == {_quoted(scope.kb_id)}",
            f"index_version_id == {_quoted(scope.index_version_id)}",
        )
    )


def _extract_count(rows: Any) -> int:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise VectorMaintenanceError("Milvus count 查询没有返回聚合行")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise VectorMaintenanceError("Milvus count 聚合行格式非法")
    raw = row.get("count(*)", row.get("count"))
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise VectorMaintenanceError("Milvus count 聚合值非法") from exc
    if count < 0:
        raise VectorMaintenanceError("Milvus count 不能为负数")
    return count


class GuardedMilvusMaintenanceGateway:
    """真实 SDK adapter；构造时必须显式 ``allow_mutations=True`` 才能写。"""

    def __init__(
        self,
        client: Any,
        *,
        configured_database: str = "default",
        allow_mutations: bool = False,
        migration_copy_executor: MigrationCopyExecutor | None = None,
        migration_validation_executor: MigrationValidationExecutor | None = None,
    ) -> None:
        if client is None:
            raise VectorMaintenanceError("Milvus client 未配置")
        self._client = client
        self._database = configured_database
        self._allow_mutations = allow_mutations
        self._copy_executor = migration_copy_executor
        self._validation_executor = migration_validation_executor

    @property
    def mutations_enabled(self) -> bool:
        return self._allow_mutations

    def inspect_fingerprint(
        self,
        *,
        physical_database: str,
        physical_collection: str,
        expected: CompatibilityFingerprint,
    ) -> CompatibilityFingerprint:
        profile = CompatibilityProfile.model_validate(expected.canonical_payload)
        result = MilvusWriteCompatibilityGate(
            self._client,
            configured_database=self._database,
        ).verify(
            ResolvedVectorWriteTarget(
                physical_database=physical_database,
                physical_collection=physical_collection,
                partition_bucket=None,
                expected_profile=profile,
                expected_fingerprint=expected,
                placement_id="maintenance-inspection",
                placement_epoch=1,
                legacy_policy=LegacyWritePolicy.DENY,
            ),
            vector_dim=profile.dense.dimension,
            schema_version=profile.schema_version,
            embedding_model_id=profile.dense.model_id,
        )
        return result.compatibility_fingerprint

    def count_gc_scope(self, plan: VectorGcPlan) -> int:
        return self._count(
            collection_name=plan.scope.physical_collection,
            expression=plan.scope.milvus_filter(),
        )

    def sample_gc_identities(
        self,
        plan: VectorGcPlan,
        *,
        limit: int,
    ) -> tuple[VectorIdentity, ...]:
        rows = self._client.query(
            collection_name=plan.scope.physical_collection,
            filter=plan.scope.milvus_filter(),
            output_fields=[
                "tenant_id",
                "kb_id",
                "doc_id",
                "document_version_id",
                "index_version_id",
            ],
            limit=limit,
        )
        identities = []
        for row in rows or ():
            identities.append(
                VectorIdentity(
                    tenant_id=str(row.get("tenant_id") or ""),
                    kb_id=str(row.get("kb_id") or ""),
                    document_id=str(row.get("doc_id") or ""),
                    document_version_id=str(row.get("document_version_id") or ""),
                    index_version_id=str(row.get("index_version_id") or ""),
                )
            )
        return tuple(identities)

    def delete_gc_scope(
        self,
        plan: VectorGcPlan,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> DeleteReceipt:
        self._require_mutations(operation_id, fencing_token)
        result = self._client.delete(
            collection_name=plan.scope.physical_collection,
            filter=plan.scope.milvus_filter(),
        )
        return DeleteReceipt(
            acknowledged=result is not None,
            deleted_count=self._delete_count(result),
            # compaction 是独立可恢复阶段；本 adapter 不把 delete ack 冒充空间回收。
            compaction_status="not_requested",
        )

    def count_migration_source(self, context: MigrationExecutionContext) -> int:
        return self._count(
            collection_name=context.scope.source_collection,
            expression=_index_scope_filter(context),
        )

    def count_migration_target(self, context: MigrationExecutionContext) -> int:
        return self._count(
            collection_name=context.scope.target_collection,
            expression=_index_scope_filter(context),
        )

    def copy_migration_scope(
        self,
        context: MigrationExecutionContext,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> CopyReceipt:
        self._require_mutations(operation_id, fencing_token)
        if self._copy_executor is None:
            raise VectorMaintenanceError("migration batch copy executor 未配置")
        return self._copy_executor(
            self._client,
            context,
            operation_id,
            fencing_token,
        )

    def validate_migration(
        self,
        context: MigrationExecutionContext,
        receipt: CopyReceipt,
    ) -> MigrationValidation:
        if self._validation_executor is None:
            raise VectorMaintenanceError("migration search/payload validation executor 未配置")
        return self._validation_executor(self._client, context, receipt)

    def delete_unpublished_migration_target(
        self,
        context: MigrationExecutionContext,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> DeleteReceipt:
        self._require_mutations(operation_id, fencing_token)
        result = self._client.delete(
            collection_name=context.scope.target_collection,
            filter=_index_scope_filter(context),
        )
        return DeleteReceipt(
            acknowledged=result is not None,
            deleted_count=self._delete_count(result),
            compaction_status="not_requested",
        )

    def _count(self, *, collection_name: str, expression: str) -> int:
        rows = self._client.query(
            collection_name=collection_name,
            filter=expression,
            output_fields=["count(*)"],
        )
        return _extract_count(rows)

    def _require_mutations(self, operation_id: str, fencing_token: int) -> None:
        if not self._allow_mutations:
            raise VectorMaintenanceError("Milvus mutation 默认关闭")
        if not operation_id.strip() or fencing_token < 1:
            raise VectorMaintenanceError("Milvus mutation 缺少 operation/fencing")

    @staticmethod
    def _delete_count(result: Any) -> int | None:
        if not isinstance(result, Mapping):
            return None
        raw = result.get("delete_count")
        if raw is None:
            return None
        try:
            count = int(raw)
        except (TypeError, ValueError) as exc:
            raise VectorMaintenanceError("Milvus delete_count 非法") from exc
        if count < 0:
            raise VectorMaintenanceError("Milvus delete_count 不能为负数")
        return count


__all__ = ["GuardedMilvusMaintenanceGateway"]
