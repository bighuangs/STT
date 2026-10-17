"""受租约和 fencing 保护的 Milvus GC / migration Worker handler。

本模块没有默认真实写入口。调用者必须同时提供：

* SQL 持久租约仓储；
* 明确启用 mutation 的 Milvus execution port；
* 已持久化且通过的 dry-run / migration manifest。

任何物理漂移、数量不一致、租约丢失或无法对账都会在 delete/copy 前 fail-closed。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.domain.vector_topology import (
    CompatibilityFingerprint,
    GcReconciliation,
    GcRunStatus,
    MigrationStatus,
    MigrationValidation,
    VectorGcPlan,
    VectorIdentity,
    VectorMigrationManifest,
    reconcile_gc,
    require_compatible,
)


class VectorMaintenanceError(RuntimeError):
    """维护动作被安全门禁阻断。"""


class VectorMaintenanceTransientError(VectorMaintenanceError):
    """外部依赖短停，可在新租约下安全重试。"""


class VectorMaintenanceLeaseLost(VectorMaintenanceError):
    """持久租约或 fencing 已失效。"""


@dataclass(frozen=True)
class VectorMaintenanceLease:
    run_id: str
    owner_id: str
    lease_token: str
    fencing_token: int
    lease_expires_at: datetime


@dataclass(frozen=True)
class GcExecutionContext:
    plan: VectorGcPlan
    expected_fingerprint: CompatibilityFingerprint
    dry_run_observed_count: int
    dry_run_safe: bool
    dry_run_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class VectorIndexMigrationScope:
    tenant_id: str
    kb_id: str
    index_version_id: str
    source_database: str
    source_collection: str
    target_database: str
    target_collection: str


@dataclass(frozen=True)
class MigrationExecutionContext:
    manifest: VectorMigrationManifest
    scope: VectorIndexMigrationScope
    source_fingerprint: CompatibilityFingerprint
    target_fingerprint: CompatibilityFingerprint
    status: MigrationStatus
    placement_epoch: int


@dataclass(frozen=True)
class DeleteReceipt:
    acknowledged: bool
    deleted_count: int | None
    compaction_status: str = "not_requested"


@dataclass(frozen=True)
class CopyReceipt:
    acknowledged: bool
    copied_count: int
    source_chunk_ids_sha256: str
    target_chunk_ids_sha256: str


class VectorMaintenanceRepository(Protocol):
    def claim_gc(
        self, gc_run_id: str, *, owner_id: str, lease_seconds: int
    ) -> tuple[VectorMaintenanceLease, GcExecutionContext]: ...

    def assert_gc_lease(self, lease: VectorMaintenanceLease) -> None: ...

    def finish_gc(
        self,
        lease: VectorMaintenanceLease,
        *,
        reconciliation: GcReconciliation,
        status: GcRunStatus,
        result: dict,
    ) -> None: ...

    def fail_gc(
        self,
        lease: VectorMaintenanceLease,
        *,
        status: GcRunStatus,
        error_code: str,
        safe_detail: str,
    ) -> None: ...

    def claim_migration(
        self, migration_run_id: str, *, owner_id: str, lease_seconds: int
    ) -> tuple[VectorMaintenanceLease, MigrationExecutionContext]: ...

    def assert_migration_lease(self, lease: VectorMaintenanceLease) -> None: ...

    def finish_migration_backfill(
        self,
        lease: VectorMaintenanceLease,
        *,
        validation: MigrationValidation,
        copy_receipt: CopyReceipt,
    ) -> None: ...

    def finish_migration_rollback(
        self,
        lease: VectorMaintenanceLease,
        *,
        removed_count: int,
    ) -> None: ...

    def fail_migration(
        self,
        lease: VectorMaintenanceLease,
        *,
        status: MigrationStatus,
        error_code: str,
        safe_detail: str,
    ) -> None: ...


class MilvusVectorMaintenancePort(Protocol):
    """唯一允许执行 Milvus mutation 的窄端口。"""

    @property
    def mutations_enabled(self) -> bool: ...

    def inspect_fingerprint(
        self,
        *,
        physical_database: str,
        physical_collection: str,
        expected: CompatibilityFingerprint,
    ) -> CompatibilityFingerprint: ...

    def count_gc_scope(self, plan: VectorGcPlan) -> int: ...

    def sample_gc_identities(
        self, plan: VectorGcPlan, *, limit: int
    ) -> tuple[VectorIdentity, ...]: ...

    def delete_gc_scope(
        self,
        plan: VectorGcPlan,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> DeleteReceipt: ...

    def count_migration_source(self, context: MigrationExecutionContext) -> int: ...

    def count_migration_target(self, context: MigrationExecutionContext) -> int: ...

    def copy_migration_scope(
        self,
        context: MigrationExecutionContext,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> CopyReceipt: ...

    def validate_migration(
        self,
        context: MigrationExecutionContext,
        receipt: CopyReceipt,
    ) -> MigrationValidation: ...

    def delete_unpublished_migration_target(
        self,
        context: MigrationExecutionContext,
        *,
        operation_id: str,
        fencing_token: int,
    ) -> DeleteReceipt: ...


def _same_identity(plan: VectorGcPlan, identities: tuple[VectorIdentity, ...]) -> bool:
    expected = VectorIdentity.model_validate(
        plan.scope.model_dump(
            include={
                "tenant_id",
                "kb_id",
                "document_id",
                "document_version_id",
                "index_version_id",
            }
        )
    )
    return all(identity == expected for identity in identities)


class VectorMaintenanceWorkerHandler:
    """一次调用处理一个持久 run；不包含轮询线程，便于嵌入现有 Worker。"""

    def __init__(
        self,
        repository: VectorMaintenanceRepository,
        gateway: MilvusVectorMaintenancePort,
        *,
        lease_seconds: int = 60,
    ) -> None:
        if lease_seconds < 5 or lease_seconds > 3600:
            raise ValueError("vector maintenance lease_seconds must be in 5..3600")
        self._repository = repository
        self._gateway = gateway
        self._lease_seconds = lease_seconds

    def run_gc(self, gc_run_id: str, *, owner_id: str) -> GcRunStatus:
        lease, context = self._repository.claim_gc(
            gc_run_id,
            owner_id=owner_id,
            lease_seconds=self._lease_seconds,
        )
        plan = context.plan
        try:
            if not self._gateway.mutations_enabled:
                raise VectorMaintenanceError("Milvus mutation 未显式启用")
            if not context.dry_run_safe:
                raise VectorMaintenanceError("持久 GC dry-run 未通过")
            if context.dry_run_observed_count != plan.expected_count:
                raise VectorMaintenanceError("持久 GC dry-run 数量已不一致")

            self._repository.assert_gc_lease(lease)
            actual = self._gateway.inspect_fingerprint(
                physical_database=plan.scope.physical_database,
                physical_collection=plan.scope.physical_collection,
                expected=context.expected_fingerprint,
            )
            require_compatible(context.expected_fingerprint, actual)

            matched_before = self._gateway.count_gc_scope(plan)
            if matched_before != plan.expected_count:
                raise VectorMaintenanceError("GC 执行前 matched count 与计划不一致")
            sampled = self._gateway.sample_gc_identities(
                plan,
                limit=max(1, min(matched_before, 1000)),
            )
            if not _same_identity(plan, sampled):
                raise VectorMaintenanceError("GC 执行前发现跨 scope 身份")

            # mutation 前最后一次持久 fencing 校验。delete 使用完整五层 identity
            # filter 且 operation_id 稳定，超时重试仍是幂等的。
            self._repository.assert_gc_lease(lease)
            receipt = self._gateway.delete_gc_scope(
                plan,
                operation_id=plan.idempotency_key,
                fencing_token=lease.fencing_token,
            )
            if receipt.deleted_count is not None and receipt.deleted_count > plan.expected_count:
                raise VectorMaintenanceError("Milvus 报告超预期 delete 数量")
            remaining = self._gateway.count_gc_scope(plan)
            reconciliation = GcReconciliation(
                expected_count=plan.expected_count,
                matched_before=matched_before,
                delete_acknowledged=receipt.acknowledged,
                remaining_after=remaining,
                compaction_status=receipt.compaction_status,
            )
            outcome = reconcile_gc(reconciliation)
            self._repository.assert_gc_lease(lease)
            self._repository.finish_gc(
                lease,
                reconciliation=reconciliation,
                status=outcome.status,
                result=outcome.model_dump(mode="json"),
            )
            return outcome.status
        except VectorMaintenanceLeaseLost:
            raise
        except VectorMaintenanceTransientError as exc:
            self._repository.fail_gc(
                lease,
                status=GcRunStatus.FAILED_RETRYABLE,
                error_code="VECTOR_DEPENDENCY_UNAVAILABLE",
                safe_detail=str(exc),
            )
            return GcRunStatus.FAILED_RETRYABLE
        except Exception as exc:
            self._repository.fail_gc(
                lease,
                status=GcRunStatus.BLOCKED_MANUAL,
                error_code="VECTOR_GC_GUARD_BLOCKED",
                safe_detail=str(exc),
            )
            return GcRunStatus.BLOCKED_MANUAL

    def run_migration_backfill(
        self,
        migration_run_id: str,
        *,
        owner_id: str,
    ) -> MigrationStatus:
        lease, context = self._repository.claim_migration(
            migration_run_id,
            owner_id=owner_id,
            lease_seconds=self._lease_seconds,
        )
        try:
            if not self._gateway.mutations_enabled:
                raise VectorMaintenanceError("Milvus mutation 未显式启用")
            self._repository.assert_migration_lease(lease)
            source_actual = self._gateway.inspect_fingerprint(
                physical_database=context.scope.source_database,
                physical_collection=context.scope.source_collection,
                expected=context.source_fingerprint,
            )
            target_actual = self._gateway.inspect_fingerprint(
                physical_database=context.scope.target_database,
                physical_collection=context.scope.target_collection,
                expected=context.target_fingerprint,
            )
            require_compatible(context.source_fingerprint, source_actual)
            require_compatible(context.target_fingerprint, target_actual)
            if (
                context.target_fingerprint.compatibility
                != context.manifest.target_compatibility_fingerprint
            ):
                raise VectorMaintenanceError("migration target compatibility 漂移")
            source_count = self._gateway.count_migration_source(context)
            if source_count != context.manifest.expected_count:
                raise VectorMaintenanceError("migration source count 与 manifest 不一致")
            target_count = self._gateway.count_migration_target(context)
            if target_count:
                raise VectorMaintenanceError("migration target 非空，须先走受控 target rollback")

            self._repository.assert_migration_lease(lease)
            receipt = self._gateway.copy_migration_scope(
                context,
                operation_id=context.manifest.migration_run_id,
                fencing_token=lease.fencing_token,
            )
            if not receipt.acknowledged:
                raise VectorMaintenanceTransientError("Milvus backfill 未确认")
            if receipt.copied_count > context.manifest.expected_count:
                raise VectorMaintenanceError("Milvus 报告超预期 copy 数量")
            self._repository.assert_migration_lease(lease)
            validation = self._gateway.validate_migration(context, receipt)
            if not validation.passed:
                raise VectorMaintenanceError(
                    "migration validation 失败: " + ",".join(validation.failure_reasons())
                )
            self._repository.finish_migration_backfill(
                lease,
                validation=validation,
                copy_receipt=receipt,
            )
            return MigrationStatus.READY_TO_SWITCH
        except VectorMaintenanceLeaseLost:
            raise
        except VectorMaintenanceTransientError as exc:
            self._repository.fail_migration(
                lease,
                status=MigrationStatus.FAILED_RETRYABLE,
                error_code="VECTOR_DEPENDENCY_UNAVAILABLE",
                safe_detail=str(exc),
            )
            return MigrationStatus.FAILED_RETRYABLE
        except Exception as exc:
            self._repository.fail_migration(
                lease,
                status=MigrationStatus.BLOCKED_MANUAL,
                error_code="VECTOR_MIGRATION_GUARD_BLOCKED",
                safe_detail=str(exc),
            )
            return MigrationStatus.BLOCKED_MANUAL

    def rollback_unpublished_target(
        self,
        migration_run_id: str,
        *,
        owner_id: str,
    ) -> MigrationStatus:
        """清理由失败 backfill 产生的 target 数据；永不触碰 source/read placement。"""

        lease, context = self._repository.claim_migration(
            migration_run_id,
            owner_id=owner_id,
            lease_seconds=self._lease_seconds,
        )
        try:
            if not self._gateway.mutations_enabled:
                raise VectorMaintenanceError("Milvus mutation 未显式启用")
            if context.status in {
                MigrationStatus.SWITCHED,
                MigrationStatus.ROLLBACK_WINDOW,
                MigrationStatus.SOURCE_GC_ELIGIBLE,
                MigrationStatus.COMPLETED,
            }:
                raise VectorMaintenanceError("已切读 migration 不能清 unpublished target")
            self._repository.assert_migration_lease(lease)
            target_count = self._gateway.count_migration_target(context)
            if target_count > context.manifest.expected_count:
                raise VectorMaintenanceError("target count 超出 manifest，禁止自动回滚删除")
            receipt = self._gateway.delete_unpublished_migration_target(
                context,
                operation_id=f"{context.manifest.migration_run_id}:rollback-target",
                fencing_token=lease.fencing_token,
            )
            if not receipt.acknowledged or (
                receipt.deleted_count is not None and receipt.deleted_count > target_count
            ):
                raise VectorMaintenanceError("target rollback delete 无法精确对账")
            remaining = self._gateway.count_migration_target(context)
            if remaining:
                raise VectorMaintenanceTransientError("target rollback 尚有残留")
            self._repository.finish_migration_rollback(
                lease,
                removed_count=target_count,
            )
            return MigrationStatus.TARGET_CREATED
        except VectorMaintenanceLeaseLost:
            raise
        except VectorMaintenanceTransientError as exc:
            self._repository.fail_migration(
                lease,
                status=MigrationStatus.FAILED_RETRYABLE,
                error_code="VECTOR_TARGET_ROLLBACK_PARTIAL",
                safe_detail=str(exc),
            )
            return MigrationStatus.FAILED_RETRYABLE
        except Exception as exc:
            self._repository.fail_migration(
                lease,
                status=MigrationStatus.BLOCKED_MANUAL,
                error_code="VECTOR_TARGET_ROLLBACK_BLOCKED",
                safe_detail=str(exc),
            )
            return MigrationStatus.BLOCKED_MANUAL


__all__ = [
    "CopyReceipt",
    "DeleteReceipt",
    "GcExecutionContext",
    "MigrationExecutionContext",
    "MilvusVectorMaintenancePort",
    "VectorIndexMigrationScope",
    "VectorMaintenanceError",
    "VectorMaintenanceLease",
    "VectorMaintenanceLeaseLost",
    "VectorMaintenanceRepository",
    "VectorMaintenanceTransientError",
    "VectorMaintenanceWorkerHandler",
]
