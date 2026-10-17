"""TODO-37 向量拓扑控制面编排。

应用服务只产生/推进可持久化的 placement、GC 和 migration 事实；它本身不调用
Milvus delete/drop/create。破坏性动作必须由 TODO-02 的持久 worker 在校验 dry-run、
lease fencing 和 checkpoint 后执行。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol

from app.domain.vector_topology import (
    CollectionInventorySnapshot,
    CompatibilityFingerprint,
    CompatibilityProfile,
    FamilyStatus,
    GcDryRunDecision,
    GcDryRunObservation,
    GcEligibility,
    GcOutcome,
    GcProtectionFacts,
    GcReconciliation,
    GcRunStatus,
    MigrationStatus,
    MigrationValidation,
    PlacementDecision,
    PlacementPolicy,
    PlacementRequest,
    PlacementStatus,
    QueryPhysicalReadSet,
    TopologyInventoryReport,
    VectorCollectionFamily,
    VectorGcPlan,
    VectorGcScope,
    VectorMigrationManifest,
    VectorPlacement,
    audit_inventory,
    canonical_json,
    compatibility_fingerprint,
    ensure_gc_transition,
    ensure_migration_transition,
    evaluate_gc_eligibility,
    freeze_query_read_set,
    reconcile_gc,
    require_compatible,
    resolve_placement,
    validate_gc_dry_run,
)


class VectorTopologyRepository(Protocol):
    """SQL repository 必须实现的最小控制面接口。"""

    def find_family(
        self,
        *,
        physical_database: str,
        physical_collection: str,
    ) -> VectorCollectionFamily | None: ...

    def save_family(self, family: VectorCollectionFamily) -> None: ...

    def save_placement(self, placement: VectorPlacement) -> None: ...

    def activate_planned_placement(
        self,
        *,
        family_id: str,
        placement_id: str,
        readable: bool,
        writable: bool,
    ) -> None: ...

    def list_read_placements(
        self,
        *,
        tenant_id: str,
        kb_ids: Sequence[str],
        index_version_ids: Sequence[str],
    ) -> Sequence[VectorPlacement]: ...

    def save_gc_plan(self, plan: VectorGcPlan) -> None: ...

    def save_gc_decision(
        self,
        *,
        gc_run_id: str,
        status: GcRunStatus,
        payload: dict,
    ) -> None: ...

    def save_migration(
        self,
        manifest: VectorMigrationManifest,
        *,
        status: MigrationStatus,
        payload: dict,
    ) -> None: ...

    def atomic_switch_placement(
        self,
        *,
        migration_run_id: str,
        source_family_id: str,
        target_family_id: str,
        expected_epoch: int,
    ) -> int: ...


class ReadOnlyMilvusInventoryPort(Protocol):
    """刻意不暴露 create/insert/delete/drop 的只读 Milvus 端口。"""

    def inspect_collection(
        self,
        *,
        physical_database: str,
        physical_collection: str,
        sample_limit: int,
    ) -> CollectionInventorySnapshot: ...


class VectorTopologyService:
    """安全地生成 topology 事实并把副作用留给持久 worker。"""

    def __init__(
        self,
        repository: VectorTopologyRepository,
        *,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._clock = clock

    @staticmethod
    def _stable_id(prefix: str, value: object) -> str:
        digest = sha256(canonical_json(value).encode("utf-8")).hexdigest()[:32]
        return f"{prefix}_{digest}"

    def plan_placement(
        self,
        *,
        policy: PlacementPolicy,
        request: PlacementRequest,
        profile: CompatibilityProfile,
    ) -> tuple[VectorCollectionFamily, VectorPlacement, PlacementDecision]:
        """幂等解析 family/placement；既有 family 必须完整指纹一致。"""

        fingerprint = compatibility_fingerprint(profile)
        decision = resolve_placement(policy, request, fingerprint)
        family = self._repository.find_family(
            physical_database=decision.physical_database,
            physical_collection=decision.physical_collection,
        )
        if family is not None:
            actual = CompatibilityFingerprint(
                compatibility=family.compatibility_fingerprint,
                schema_fingerprint=family.schema_fingerprint,
                index_fingerprint=family.index_fingerprint,
                canonical_payload=family.compatibility_payload,
            )
            require_compatible(fingerprint, actual)
            if family.status not in {
                FamilyStatus.CREATING,
                FamilyStatus.VALIDATING,
                FamilyStatus.ACTIVE,
            }:
                raise ValueError(
                    f"vector family is not writable: {family.family_id}/{family.status}"
                )
        else:
            family = VectorCollectionFamily(
                family_id=self._stable_id(
                    "vf",
                    {
                        "database": decision.physical_database,
                        "collection": decision.physical_collection,
                        "fingerprint": fingerprint.compatibility,
                    },
                ),
                physical_database=decision.physical_database,
                physical_collection=decision.physical_collection,
                placement_mode=decision.mode,
                compatibility_fingerprint=fingerprint.compatibility,
                schema_fingerprint=fingerprint.schema_fingerprint,
                index_fingerprint=fingerprint.index_fingerprint,
                compatibility_payload=fingerprint.canonical_payload,
                partition_strategy=profile.partition.strategy,
                num_partitions=profile.partition.num_partitions,
                status=FamilyStatus.CREATING,
                epoch=1,
            )
            self._repository.save_family(family)

        placement = VectorPlacement(
            placement_id=self._stable_id(
                "vp",
                {
                    "tenant_id": request.tenant_id,
                    "kb_id": request.kb_id,
                    "index_version_id": request.index_version_id,
                    "family_id": family.family_id,
                    "policy_id": policy.policy_id,
                    "policy_version": policy.version,
                },
            ),
            tenant_id=request.tenant_id,
            kb_id=request.kb_id,
            index_version_id=request.index_version_id,
            family_id=family.family_id,
            physical_database=family.physical_database,
            physical_collection=family.physical_collection,
            partition_bucket=decision.partition_bucket,
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            reason_codes=decision.reason_codes,
            status=PlacementStatus.PLANNED,
            epoch=family.epoch,
            readable=False,
            writable=False,
        )
        self._repository.save_placement(placement)
        return family, placement, decision

    def activate_planned_placement(
        self,
        *,
        family_id: str,
        placement_id: str,
        readable: bool,
        writable: bool = True,
    ) -> None:
        """把已规划目标原子提升为受管读写目标。

        该动作只推进 SQL 控制面状态，不访问 Milvus。真实写入仍须经过
        ``MilvusWriteCompatibilityGate``；首次创建也只能按 family 中已冻结的完整
        兼容合同执行。
        """

        if not writable:
            raise ValueError("new managed index must have one writable placement")
        self._repository.activate_planned_placement(
            family_id=family_id,
            placement_id=placement_id,
            readable=readable,
            writable=writable,
        )

    def freeze_read_set(
        self,
        *,
        tenant_id: str,
        kb_ids: Sequence[str],
        index_version_ids: Sequence[str],
    ) -> QueryPhysicalReadSet:
        """从 SQL active placement 固定一次 QueryRun 的物理读集合。"""

        if not tenant_id or not kb_ids or not index_version_ids:
            raise ValueError("tenant, KB and index scopes are required")
        placements = tuple(
            self._repository.list_read_placements(
                tenant_id=tenant_id,
                kb_ids=tuple(kb_ids),
                index_version_ids=tuple(index_version_ids),
            )
        )
        for placement in placements:
            if (
                placement.tenant_id != tenant_id
                or placement.kb_id not in kb_ids
                or placement.index_version_id not in index_version_ids
            ):
                raise ValueError("repository returned an out-of-scope vector placement")
        resolved_indexes = {item.index_version_id for item in placements if item.readable}
        if resolved_indexes != set(index_version_ids):
            raise ValueError("one or more requested index versions lack an active placement")
        return freeze_query_read_set(placements)

    def create_gc_plan(
        self,
        *,
        scope: VectorGcScope,
        expected_count: int,
        protection: GcProtectionFacts,
        drop_family_requested: bool = False,
    ) -> tuple[VectorGcPlan, GcEligibility]:
        """只创建 dry-run 计划；保护事实不满足时仍保存阻断证据。"""

        eligibility = evaluate_gc_eligibility(protection, now=self._clock())
        identity = {
            "scope": scope.model_dump(mode="json"),
            "expected_count": expected_count,
            "drop_family_requested": drop_family_requested,
        }
        can_plan = eligibility.eligible and (
            not drop_family_requested or eligibility.drop_family_allowed
        )
        plan = VectorGcPlan(
            gc_run_id=self._stable_id("vgc", identity),
            idempotency_key=self._stable_id("vgc-idem", identity),
            scope=scope,
            expected_count=expected_count,
            drop_family_requested=drop_family_requested,
            status=(GcRunStatus.PLANNED if can_plan else GcRunStatus.BLOCKED_MANUAL),
            dry_run=True,
        )
        self._repository.save_gc_plan(plan)
        return plan, eligibility

    def validate_gc_plan(
        self,
        *,
        plan: VectorGcPlan,
        observation: GcDryRunObservation,
        protection: GcProtectionFacts,
    ) -> GcDryRunDecision:
        eligibility = evaluate_gc_eligibility(protection, now=self._clock())
        decision = validate_gc_dry_run(plan, observation, eligibility)
        if plan.status != GcRunStatus.BLOCKED_MANUAL:
            ensure_gc_transition(plan.status, decision.status)
        self._repository.save_gc_decision(
            gc_run_id=plan.gc_run_id,
            status=decision.status,
            payload={
                "dry_run": True,
                "observation": observation.model_dump(mode="json"),
                "eligibility": eligibility.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            },
        )
        return decision

    def record_gc_reconciliation(
        self,
        *,
        gc_run_id: str,
        current_status: GcRunStatus,
        reconciliation: GcReconciliation,
    ) -> GcOutcome:
        outcome = reconcile_gc(reconciliation)
        ensure_gc_transition(current_status, outcome.status)
        self._repository.save_gc_decision(
            gc_run_id=gc_run_id,
            status=outcome.status,
            payload={
                "dry_run": False,
                "reconciliation": reconciliation.model_dump(mode="json"),
                "outcome": outcome.model_dump(mode="json"),
            },
        )
        return outcome

    def create_migration_manifest(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        index_version_id: str,
        source_family_id: str,
        target_family_id: str,
        expected_count: int,
        chunk_ids_sha256: str,
        target_compatibility_fingerprint: str,
        rollback_deadline: datetime,
    ) -> VectorMigrationManifest:
        identity = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "index_version_id": index_version_id,
            "source_family_id": source_family_id,
            "target_family_id": target_family_id,
            "target_compatibility_fingerprint": target_compatibility_fingerprint,
        }
        manifest = VectorMigrationManifest(
            migration_run_id=self._stable_id("vm", identity),
            tenant_id=tenant_id,
            kb_id=kb_id,
            index_version_id=index_version_id,
            source_family_id=source_family_id,
            target_family_id=target_family_id,
            expected_count=expected_count,
            chunk_ids_sha256=chunk_ids_sha256,
            target_compatibility_fingerprint=target_compatibility_fingerprint,
            rollback_deadline=rollback_deadline,
        )
        self._repository.save_migration(
            manifest,
            status=MigrationStatus.PLANNED,
            payload={"dry_run": True, "manifest": manifest.model_dump(mode="json")},
        )
        return manifest

    def advance_migration(
        self,
        *,
        manifest: VectorMigrationManifest,
        current_status: MigrationStatus,
        target_status: MigrationStatus,
        validation: MigrationValidation | None = None,
        rollback_pinned: bool = False,
        expected_epoch: int | None = None,
    ) -> int | None:
        """推进持久状态；切读只能走 repository 的 epoch-fenced 原子事务。"""

        ensure_migration_transition(
            current_status,
            target_status,
            validation=validation,
            rollback_deadline=manifest.rollback_deadline,
            rollback_pinned=rollback_pinned,
            now=self._clock(),
        )
        next_epoch: int | None = None
        if target_status == MigrationStatus.SWITCHED:
            if expected_epoch is None:
                raise ValueError("atomic placement switch requires expected_epoch")
            next_epoch = self._repository.atomic_switch_placement(
                migration_run_id=manifest.migration_run_id,
                source_family_id=manifest.source_family_id,
                target_family_id=manifest.target_family_id,
                expected_epoch=expected_epoch,
            )
        self._repository.save_migration(
            manifest,
            status=target_status,
            payload={
                "validation": (validation.model_dump(mode="json") if validation else None),
                "rollback_pinned": rollback_pinned,
                "placement_epoch": next_epoch,
            },
        )
        return next_epoch

    @staticmethod
    def audit_collection(
        inspector: ReadOnlyMilvusInventoryPort,
        *,
        physical_database: str,
        physical_collection: str,
        expected_fingerprint: CompatibilityFingerprint | None = None,
        sample_limit: int = 1000,
    ) -> TopologyInventoryReport:
        if sample_limit < 1 or sample_limit > 10_000:
            raise ValueError("sample_limit must be between 1 and 10000")
        snapshot = inspector.inspect_collection(
            physical_database=physical_database,
            physical_collection=physical_collection,
            sample_limit=sample_limit,
        )
        if (
            snapshot.physical_database != physical_database
            or snapshot.physical_collection != physical_collection
        ):
            raise ValueError("inventory port returned a different physical target")
        return audit_inventory(
            snapshot,
            expected_fingerprint=expected_fingerprint,
        )


__all__ = [
    "ReadOnlyMilvusInventoryPort",
    "VectorTopologyRepository",
    "VectorTopologyService",
]
