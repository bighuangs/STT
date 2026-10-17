"""Vector maintenance 的 SQL 租约、fencing 与结果持久化。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from secrets import token_urlsafe

from sqlalchemy import or_, update

from app.application.vector_maintenance_service import (
    CopyReceipt,
    GcExecutionContext,
    MigrationExecutionContext,
    VectorIndexMigrationScope,
    VectorMaintenanceLease,
    VectorMaintenanceLeaseLost,
)
from app.domain.vector_topology import (
    CompatibilityFingerprint,
    GcReconciliation,
    GcRunStatus,
    MigrationStatus,
    MigrationValidation,
    VectorGcPlan,
    VectorGcScope,
    VectorMigrationManifest,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorGcRunModel,
    VectorMigrationRunModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _token_hash(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(row: VectorCollectionFamilyModel) -> CompatibilityFingerprint:
    return CompatibilityFingerprint(
        compatibility=row.compatibility_fingerprint,
        schema_fingerprint=row.schema_fingerprint,
        index_fingerprint=row.index_fingerprint,
        canonical_payload=row.compatibility_payload,
    )


class SqlVectorMaintenanceRepository:
    """每个方法使用短事务；外部 Milvus RPC 期间不占用 SQL transaction。"""

    @staticmethod
    def _lease(
        run_id: str,
        owner_id: str,
        token: str,
        fencing_token: int,
        expires_at: datetime,
    ) -> VectorMaintenanceLease:
        return VectorMaintenanceLease(
            run_id=run_id,
            owner_id=owner_id,
            lease_token=token,
            fencing_token=fencing_token,
            lease_expires_at=_aware(expires_at) or expires_at,
        )

    def claim_gc(
        self,
        gc_run_id: str,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> tuple[VectorMaintenanceLease, GcExecutionContext]:
        if not owner_id.strip():
            raise ValueError("vector maintenance owner_id is required")
        now = _now()
        token = token_urlsafe(32)
        with session_scope() as session:
            row = session.get(VectorGcRunModel, gc_run_id)
            if row is None:
                raise ResourceNotFoundError("vector GC run not found")
            expires = _aware(row.lease_expires_at)
            if expires is not None and expires > now:
                raise ResourceConflictError("vector GC run has an active lease")
            if row.status not in {
                GcRunStatus.DRY_RUN_VALIDATED.value,
                GcRunStatus.FAILED_RETRYABLE.value,
                GcRunStatus.PARTIAL.value,
            }:
                raise ResourceConflictError(f"vector GC run is not claimable: {row.status}")
            family = session.get(VectorCollectionFamilyModel, row.family_id)
            if family is None:
                raise ResourceConflictError("vector GC family is missing")
            payload = dict(row.result_json or {})
            decision = dict(payload.get("decision") or {})
            observation = dict(payload.get("observation") or {})
            if (
                payload.get("dry_run") is not True
                or decision.get("safe_to_execute") is not True
                or decision.get("status") != GcRunStatus.DRY_RUN_VALIDATED.value
            ):
                raise ResourceConflictError("vector GC run lacks a persisted validated dry-run")

            current_fencing = int(row.fencing_token)
            next_fencing = current_fencing + 1
            next_attempt = row.attempt + 1
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            claimed = session.execute(
                update(VectorGcRunModel)
                .where(
                    VectorGcRunModel.id == gc_run_id,
                    VectorGcRunModel.fencing_token == current_fencing,
                    VectorGcRunModel.status.in_(
                        (
                            GcRunStatus.DRY_RUN_VALIDATED.value,
                            GcRunStatus.FAILED_RETRYABLE.value,
                            GcRunStatus.PARTIAL.value,
                        )
                    ),
                    or_(
                        VectorGcRunModel.lease_expires_at.is_(None),
                        VectorGcRunModel.lease_expires_at <= now,
                    ),
                )
                .values(
                    lease_owner=owner_id,
                    lease_token_hash=_token_hash(token),
                    fencing_token=next_fencing,
                    lease_expires_at=lease_expires_at,
                    attempt=next_attempt,
                    status=GcRunStatus.RUNNING.value,
                    error_code=None,
                    error_detail=None,
                )
            )
            if claimed.rowcount != 1:
                raise ResourceConflictError("vector GC lease claim raced")
            plan = VectorGcPlan(
                gc_run_id=row.id,
                idempotency_key=row.idempotency_key,
                scope=VectorGcScope.model_validate(row.scope_json),
                expected_count=row.expected_count,
                drop_family_requested=bool(payload.get("drop_family_requested", False)),
                status=GcRunStatus.DRY_RUN_VALIDATED,
                dry_run=False,
                attempt=next_attempt,
                fencing_token=next_fencing,
            )
            lease = self._lease(
                row.id,
                owner_id,
                token,
                next_fencing,
                lease_expires_at,
            )
            context = GcExecutionContext(
                plan=plan,
                expected_fingerprint=_fingerprint(family),
                dry_run_observed_count=int(observation.get("matched_count", -1)),
                dry_run_safe=True,
                dry_run_reason_codes=tuple(decision.get("reason_codes") or ()),
            )
            return lease, context

    def assert_gc_lease(self, lease: VectorMaintenanceLease) -> None:
        with session_scope() as session:
            row = session.get(VectorGcRunModel, lease.run_id)
            if not self._owns(row, lease, expected_status=GcRunStatus.RUNNING.value):
                raise VectorMaintenanceLeaseLost("vector GC lease/fencing lost")

    def finish_gc(
        self,
        lease: VectorMaintenanceLease,
        *,
        reconciliation: GcReconciliation,
        status: GcRunStatus,
        result: dict,
    ) -> None:
        with session_scope() as session:
            row = session.get(VectorGcRunModel, lease.run_id)
            if not self._owns(row, lease, expected_status=GcRunStatus.RUNNING.value):
                raise VectorMaintenanceLeaseLost("vector GC lease/fencing lost")
            assert row is not None
            row.status = status.value
            row.matched_before = reconciliation.matched_before
            row.delete_acknowledged = reconciliation.delete_acknowledged
            row.remaining_after = reconciliation.remaining_after
            row.compaction_status = reconciliation.compaction_status
            row.result_json = {
                "dry_run": False,
                "reconciliation": reconciliation.model_dump(mode="json"),
                "outcome": result,
                "fencing_token": lease.fencing_token,
            }
            row.completed_at = (
                _now()
                if status
                in {
                    GcRunStatus.SUCCEEDED,
                    GcRunStatus.BLOCKED_MANUAL,
                    GcRunStatus.CANCELLED,
                }
                else None
            )
            self._clear_lease(row)

    def fail_gc(
        self,
        lease: VectorMaintenanceLease,
        *,
        status: GcRunStatus,
        error_code: str,
        safe_detail: str,
    ) -> None:
        with session_scope() as session:
            row = session.get(VectorGcRunModel, lease.run_id)
            if not self._owns(row, lease, expected_status=GcRunStatus.RUNNING.value):
                raise VectorMaintenanceLeaseLost("vector GC lease/fencing lost")
            assert row is not None
            row.status = status.value
            row.error_code = error_code[:128]
            row.error_detail = safe_detail[:1024]
            if status == GcRunStatus.BLOCKED_MANUAL:
                row.completed_at = _now()
            self._clear_lease(row)

    def claim_migration(
        self,
        migration_run_id: str,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> tuple[VectorMaintenanceLease, MigrationExecutionContext]:
        if not owner_id.strip():
            raise ValueError("vector maintenance owner_id is required")
        now = _now()
        token = token_urlsafe(32)
        with session_scope() as session:
            row = session.get(VectorMigrationRunModel, migration_run_id)
            if row is None:
                raise ResourceNotFoundError("vector migration run not found")
            expires = _aware(row.lease_expires_at)
            if expires is not None and expires > now:
                raise ResourceConflictError("vector migration run has an active lease")
            original_status = MigrationStatus(row.status)
            if original_status not in {
                MigrationStatus.TARGET_CREATED,
                MigrationStatus.BACKFILLING,
                MigrationStatus.FAILED_RETRYABLE,
            }:
                raise ResourceConflictError(f"vector migration is not claimable: {row.status}")
            source = session.get(
                VectorCollectionFamilyModel,
                row.source_family_id,
            )
            target = session.get(
                VectorCollectionFamilyModel,
                row.target_family_id,
            )
            if source is None or target is None:
                raise ResourceConflictError("migration source/target family is missing")

            current_fencing = int(row.fencing_token)
            next_fencing = current_fencing + 1
            next_attempt = row.attempt + 1
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            claimed = session.execute(
                update(VectorMigrationRunModel)
                .where(
                    VectorMigrationRunModel.id == migration_run_id,
                    VectorMigrationRunModel.fencing_token == current_fencing,
                    VectorMigrationRunModel.status.in_(
                        (
                            MigrationStatus.TARGET_CREATED.value,
                            MigrationStatus.BACKFILLING.value,
                            MigrationStatus.FAILED_RETRYABLE.value,
                        )
                    ),
                    or_(
                        VectorMigrationRunModel.lease_expires_at.is_(None),
                        VectorMigrationRunModel.lease_expires_at <= now,
                    ),
                )
                .values(
                    lease_owner=owner_id,
                    lease_token_hash=_token_hash(token),
                    fencing_token=next_fencing,
                    lease_expires_at=lease_expires_at,
                    attempt=next_attempt,
                    status=MigrationStatus.BACKFILLING.value,
                    error_code=None,
                    error_detail=None,
                )
            )
            if claimed.rowcount != 1:
                raise ResourceConflictError("vector migration lease claim raced")
            manifest = VectorMigrationManifest(
                migration_run_id=row.id,
                tenant_id=row.tenant_id,
                kb_id=row.kb_id,
                index_version_id=row.index_version_id,
                source_family_id=row.source_family_id,
                target_family_id=row.target_family_id,
                expected_count=row.expected_count,
                chunk_ids_sha256=row.chunk_ids_sha256,
                target_compatibility_fingerprint=(row.target_compatibility_fingerprint),
                rollback_deadline=row.rollback_deadline,
            )
            lease = self._lease(
                row.id,
                owner_id,
                token,
                next_fencing,
                lease_expires_at,
            )
            context = MigrationExecutionContext(
                manifest=manifest,
                scope=VectorIndexMigrationScope(
                    tenant_id=row.tenant_id,
                    kb_id=row.kb_id,
                    index_version_id=row.index_version_id,
                    source_database=source.physical_database,
                    source_collection=source.physical_collection,
                    target_database=target.physical_database,
                    target_collection=target.physical_collection,
                ),
                source_fingerprint=_fingerprint(source),
                target_fingerprint=_fingerprint(target),
                status=original_status,
                placement_epoch=row.placement_epoch,
            )
            return lease, context

    def assert_migration_lease(self, lease: VectorMaintenanceLease) -> None:
        with session_scope() as session:
            row = session.get(VectorMigrationRunModel, lease.run_id)
            if not self._owns(
                row,
                lease,
                expected_status=MigrationStatus.BACKFILLING.value,
            ):
                raise VectorMaintenanceLeaseLost("vector migration lease/fencing lost")

    def finish_migration_backfill(
        self,
        lease: VectorMaintenanceLease,
        *,
        validation: MigrationValidation,
        copy_receipt: CopyReceipt,
    ) -> None:
        if not validation.passed:
            raise ValueError("cannot persist a failed migration validation as ready")
        with session_scope() as session:
            row = session.get(VectorMigrationRunModel, lease.run_id)
            if not self._owns(
                row,
                lease,
                expected_status=MigrationStatus.BACKFILLING.value,
            ):
                raise VectorMaintenanceLeaseLost("vector migration lease/fencing lost")
            assert row is not None
            row.status = MigrationStatus.READY_TO_SWITCH.value
            row.validation_json = validation.model_dump(mode="json")
            row.checkpoint_json = {
                "copy_receipt": {
                    "acknowledged": copy_receipt.acknowledged,
                    "copied_count": copy_receipt.copied_count,
                    "source_chunk_ids_sha256": (copy_receipt.source_chunk_ids_sha256),
                    "target_chunk_ids_sha256": (copy_receipt.target_chunk_ids_sha256),
                },
                "fencing_token": lease.fencing_token,
            }
            self._clear_lease(row)

    def finish_migration_rollback(
        self,
        lease: VectorMaintenanceLease,
        *,
        removed_count: int,
    ) -> None:
        with session_scope() as session:
            row = session.get(VectorMigrationRunModel, lease.run_id)
            if not self._owns(
                row,
                lease,
                expected_status=MigrationStatus.BACKFILLING.value,
            ):
                raise VectorMaintenanceLeaseLost("vector migration lease/fencing lost")
            assert row is not None
            row.status = MigrationStatus.TARGET_CREATED.value
            row.checkpoint_json = {
                "target_rollback_removed_count": removed_count,
                "fencing_token": lease.fencing_token,
            }
            row.validation_json = {}
            self._clear_lease(row)

    def fail_migration(
        self,
        lease: VectorMaintenanceLease,
        *,
        status: MigrationStatus,
        error_code: str,
        safe_detail: str,
    ) -> None:
        with session_scope() as session:
            row = session.get(VectorMigrationRunModel, lease.run_id)
            if not self._owns(
                row,
                lease,
                expected_status=MigrationStatus.BACKFILLING.value,
            ):
                raise VectorMaintenanceLeaseLost("vector migration lease/fencing lost")
            assert row is not None
            row.status = status.value
            row.error_code = error_code[:128]
            row.error_detail = safe_detail[:1024]
            if status == MigrationStatus.BLOCKED_MANUAL:
                row.completed_at = _now()
            self._clear_lease(row)

    @staticmethod
    def _owns(
        row: VectorGcRunModel | VectorMigrationRunModel | None,
        lease: VectorMaintenanceLease,
        *,
        expected_status: str,
    ) -> bool:
        if row is None:
            return False
        expires = _aware(row.lease_expires_at)
        return bool(
            row.status == expected_status
            and row.lease_owner == lease.owner_id
            and row.lease_token_hash == _token_hash(lease.lease_token)
            and row.fencing_token == lease.fencing_token
            and expires is not None
            and expires > _now()
        )

    @staticmethod
    def _clear_lease(
        row: VectorGcRunModel | VectorMigrationRunModel,
    ) -> None:
        row.lease_owner = None
        row.lease_token_hash = None
        row.lease_expires_at = None


__all__ = ["SqlVectorMaintenanceRepository"]
