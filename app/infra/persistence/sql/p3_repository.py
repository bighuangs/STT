"""P3 查询可靠性、用量与向量拓扑 SQL 仓储。

所有公开读取都显式要求 tenant/scope；缓存授权必须在调用仓储前完成，仓储仍会把
tenant 与 authorized_scope_hash 加入查询条件作为第二道防线。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.application.observability_service import UsageRecord
from app.application.query_cache_service import (
    IdempotencyConflictError,
    QueryReplayManifest,
    canonical_hash,
)
from app.domain.vector_topology import (
    FamilyStatus,
    GcRunStatus,
    MigrationStatus,
    PlacementMode,
    PlacementStatus,
    VectorCollectionFamily,
    VectorGcPlan,
    VectorMigrationManifest,
    VectorPlacement,
)
from app.infra.persistence.sql.p3_models import (
    ModelUsageRecordModel,
    QueryCacheEntryModel,
    QueryIdempotencyKeyModel,
    QueryReplayManifestModel,
    QuerySingleflightLeaseModel,
    VectorCollectionFamilyModel,
    VectorGcRunModel,
    VectorMigrationRunModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.query_runtime_models import QueryRunModel
from app.infra.persistence.sql.primitives import (
    ResourceConflictError,
    ResourceNotFoundError,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _token_hash(value: str) -> str:
    return canonical_hash(value)


class QueryReliabilityRepository:
    def __init__(self, session: Session, *, tenant_id: str) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def claim_idempotency(
        self,
        *,
        idempotency_key: str,
        payload_hash: str,
        query_run_id: str,
        ttl: timedelta = timedelta(hours=24),
    ) -> tuple[str, bool]:
        """同 key/同 payload 返回原 run；同 key/不同 payload 稳定冲突。"""

        key_hash = canonical_hash(idempotency_key)
        now = _now()

        def existing_record() -> QueryIdempotencyKeyModel | None:
            return self.session.scalar(
                select(QueryIdempotencyKeyModel).where(
                    QueryIdempotencyKeyModel.tenant_id == self.tenant_id,
                    QueryIdempotencyKeyModel.key_hash == key_hash,
                )
            )

        row = existing_record()
        if row is not None and _aware(row.expires_at) > now:
            if row.payload_hash != payload_hash:
                raise IdempotencyConflictError("同一 Idempotency-Key 对应了不同请求载荷")
            return row.query_run_id, False
        if row is not None:
            self.session.delete(row)
            self.session.flush()
        run = self.session.scalar(
            select(QueryRunModel).where(
                QueryRunModel.id == query_run_id,
                QueryRunModel.tenant_id == self.tenant_id,
            )
        )
        if run is None:
            raise ResourceNotFoundError("query run not found in tenant scope")
        candidate = QueryIdempotencyKeyModel(
            id=f"qid_{uuid4().hex}",
            tenant_id=self.tenant_id,
            key_hash=key_hash,
            payload_hash=payload_hash,
            query_run_id=query_run_id,
            expires_at=now + ttl,
        )
        try:
            with self.session.begin_nested():
                self.session.add(candidate)
                self.session.flush()
        except IntegrityError:
            concurrent = existing_record()
            if concurrent is None:
                raise
            if concurrent.payload_hash != payload_hash:
                raise IdempotencyConflictError("同一 Idempotency-Key 对应了不同请求载荷")
            return concurrent.query_run_id, False
        return query_run_id, True

    def complete_idempotency(self, query_run_id: str, response_status: int) -> None:
        row = self.session.scalar(
            select(QueryIdempotencyKeyModel).where(
                QueryIdempotencyKeyModel.tenant_id == self.tenant_id,
                QueryIdempotencyKeyModel.query_run_id == query_run_id,
            )
        )
        if row is not None:
            row.response_status = response_status
            self.session.flush()

    def put_cache(
        self,
        *,
        cache_key: str,
        authorized_scope_hash: str,
        layer: str,
        value: dict[str, Any],
        versions: dict[str, Any],
        classification: str,
        ttl: timedelta,
    ) -> bool:
        if classification not in {"public", "internal"}:
            return False
        if ttl.total_seconds() <= 0:
            raise ValueError("cache ttl must be positive")
        row = self.session.get(QueryCacheEntryModel, cache_key)
        if row is not None and row.tenant_id != self.tenant_id:
            # SHA-256 碰撞或调用方 key 严重错误时 fail closed。
            raise ResourceConflictError("cache key already belongs to another tenant")
        if row is None:
            row = QueryCacheEntryModel(
                cache_key=cache_key,
                tenant_id=self.tenant_id,
                authorized_scope_hash=authorized_scope_hash,
                layer=layer,
                value_json=value,
                versions_json=versions,
                classification=classification,
                expires_at=_now() + ttl,
            )
            self.session.add(row)
        else:
            row.authorized_scope_hash = authorized_scope_hash
            row.layer = layer
            row.value_json = value
            row.versions_json = versions
            row.classification = classification
            row.expires_at = _now() + ttl
        self.session.flush()
        return True

    def get_cache(
        self,
        *,
        cache_key: str,
        authorized_scope_hash: str,
    ) -> tuple[dict[str, Any] | None, str]:
        row = self.session.scalar(
            select(QueryCacheEntryModel).where(
                QueryCacheEntryModel.cache_key == cache_key,
                QueryCacheEntryModel.tenant_id == self.tenant_id,
                QueryCacheEntryModel.authorized_scope_hash == authorized_scope_hash,
            )
        )
        if row is None:
            return None, "miss"
        if _aware(row.expires_at) <= _now():
            self.session.delete(row)
            self.session.flush()
            return None, "stale"
        row.hit_count += 1
        self.session.flush()
        return dict(row.value_json), "hit"

    def invalidate_scope(self, authorized_scope_hash: str) -> int:
        result = self.session.execute(
            delete(QueryCacheEntryModel).where(
                QueryCacheEntryModel.tenant_id == self.tenant_id,
                QueryCacheEntryModel.authorized_scope_hash == authorized_scope_hash,
            )
        )
        return int(result.rowcount or 0)

    def acquire_singleflight(
        self,
        *,
        fingerprint: str,
        owner_id: str,
        lease_seconds: int = 30,
    ) -> dict[str, Any]:
        if lease_seconds < 1 or lease_seconds > 300:
            raise ValueError("singleflight lease_seconds must be in 1..300")
        now = _now()
        token = token_urlsafe(32)
        row = self.session.get(QuerySingleflightLeaseModel, fingerprint)
        if row is not None and _aware(row.lease_expires_at) > now and row.owner_id != owner_id:
            return {
                "acquired": False,
                "owner_id": row.owner_id,
                "fencing_token": row.fencing_token,
                "lease_expires_at": _aware(row.lease_expires_at).isoformat(),
                "result_cache_key": row.result_cache_key,
            }
        next_fencing = (row.fencing_token + 1) if row else 1
        if row is None:
            row = QuerySingleflightLeaseModel(
                fingerprint=fingerprint,
                tenant_id=self.tenant_id,
                owner_id=owner_id,
                lease_token_hash=_token_hash(token),
                fencing_token=next_fencing,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self.session.add(row)
        else:
            if row.tenant_id != self.tenant_id:
                raise ResourceConflictError("singleflight fingerprint belongs to another tenant")
            row.owner_id = owner_id
            row.lease_token_hash = _token_hash(token)
            row.fencing_token = next_fencing
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.result_cache_key = None
        self.session.flush()
        return {
            "acquired": True,
            "lease_token": token,
            "fencing_token": next_fencing,
            "lease_expires_at": row.lease_expires_at.isoformat(),
        }

    def complete_singleflight(
        self,
        *,
        fingerprint: str,
        owner_id: str,
        lease_token: str,
        fencing_token: int,
        result_cache_key: str,
    ) -> None:
        row = self.session.scalar(
            select(QuerySingleflightLeaseModel).where(
                QuerySingleflightLeaseModel.fingerprint == fingerprint,
                QuerySingleflightLeaseModel.tenant_id == self.tenant_id,
            )
        )
        if (
            row is None
            or row.owner_id != owner_id
            or row.lease_token_hash != _token_hash(lease_token)
            or row.fencing_token != fencing_token
            or _aware(row.lease_expires_at) <= _now()
        ):
            raise ResourceConflictError("singleflight lease is stale or not owned")
        row.result_cache_key = result_cache_key
        self.session.flush()

    def abandon_singleflight(
        self,
        *,
        fingerprint: str,
        owner_id: str,
        lease_token: str,
        fencing_token: int,
    ) -> bool:
        """Producer 失败/取消时只释放自己持有的 fenced lease。

        迟到 producer 不能删除新 owner 的租约；waiter 可在失败后立即竞选，
        无需一直等待旧 lease 自然过期。
        """

        row = self.session.scalar(
            select(QuerySingleflightLeaseModel).where(
                QuerySingleflightLeaseModel.fingerprint == fingerprint,
                QuerySingleflightLeaseModel.tenant_id == self.tenant_id,
            )
        )
        if (
            row is None
            or row.owner_id != owner_id
            or row.lease_token_hash != _token_hash(lease_token)
            or row.fencing_token != fencing_token
        ):
            return False
        self.session.delete(row)
        self.session.flush()
        return True

    def save_replay_manifest(
        self,
        manifest: QueryReplayManifest,
        *,
        ttl: timedelta | None = timedelta(days=30),
    ) -> dict[str, Any]:
        run = self.session.scalar(
            select(QueryRunModel).where(
                QueryRunModel.id == manifest.query_run_id,
                QueryRunModel.tenant_id == self.tenant_id,
            )
        )
        if run is None:
            raise ResourceNotFoundError("query run not found in tenant scope")
        existing = self.session.scalar(
            select(QueryReplayManifestModel).where(
                QueryReplayManifestModel.tenant_id == self.tenant_id,
                QueryReplayManifestModel.query_run_id == manifest.query_run_id,
            )
        )
        payload = manifest.model_dump(mode="json")
        if existing is not None:
            if existing.manifest_json != payload:
                raise ResourceConflictError("query replay manifest is immutable")
            return self._replay_dict(existing)
        row = QueryReplayManifestModel(
            id=f"qrp_{uuid4().hex}",
            tenant_id=self.tenant_id,
            query_run_id=manifest.query_run_id,
            fingerprint=manifest.fingerprint.key,
            request_hash=manifest.request_hash,
            manifest_json=payload,
            event_stream_hash=manifest.event_stream_hash,
            object_refs_json=[
                value
                for value in (
                    manifest.history_snapshot_ref,
                    *manifest.route_result_refs,
                    manifest.context_pack_ref,
                    *manifest.external_snapshot_refs,
                )
                if value
            ],
            expires_at=(_now() + ttl) if ttl else None,
        )
        self.session.add(row)
        self.session.flush()
        return self._replay_dict(row)

    def get_replay_manifest(self, query_run_id: str) -> QueryReplayManifest:
        row = self.session.scalar(
            select(QueryReplayManifestModel).where(
                QueryReplayManifestModel.tenant_id == self.tenant_id,
                QueryReplayManifestModel.query_run_id == query_run_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError("query replay manifest not found")
        if row.expires_at is not None and _aware(row.expires_at) <= _now():
            raise ResourceNotFoundError("query replay manifest expired")
        return QueryReplayManifest.model_validate(row.manifest_json)

    @staticmethod
    def _replay_dict(row: QueryReplayManifestModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "query_run_id": row.query_run_id,
            "fingerprint": row.fingerprint,
            "event_stream_hash": row.event_stream_hash,
            "object_refs": list(row.object_refs_json),
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        }

    def append_usage(self, record: UsageRecord) -> None:
        if record.tenant_id != self.tenant_id:
            raise ResourceNotFoundError("usage record is outside tenant scope")
        if self.session.get(ModelUsageRecordModel, record.usage_id) is not None:
            return
        self.session.add(
            ModelUsageRecordModel(
                id=record.usage_id,
                tenant_id=record.tenant_id,
                kb_id=record.kb_id,
                query_run_id=record.query_run_id,
                provider=record.provider,
                model_id=record.model_id,
                scene=record.scene,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cache_read_tokens=record.cache_read_tokens,
                calls=record.calls,
                currency=record.currency,
                unit_price_version=record.unit_price_version,
                cost=record.cost,
                estimated=record.estimated,
                fallback_sequence=record.fallback_sequence,
                created_at=record.created_at,
            )
        )
        self.session.flush()

    def usage_summary(self, *, query_run_id: str | None = None) -> dict[str, Any]:
        criteria = [ModelUsageRecordModel.tenant_id == self.tenant_id]
        if query_run_id is not None:
            criteria.append(ModelUsageRecordModel.query_run_id == query_run_id)
        rows = self.session.scalars(select(ModelUsageRecordModel).where(*criteria)).all()
        return {
            "tenant_id": self.tenant_id,
            "query_run_id": query_run_id,
            "calls": sum(row.calls for row in rows),
            "input_tokens": sum(row.input_tokens for row in rows),
            "output_tokens": sum(row.output_tokens for row in rows),
            "cache_read_tokens": sum(row.cache_read_tokens for row in rows),
            "cost": round(sum(row.cost for row in rows), 8),
            "estimated": any(row.estimated for row in rows),
            "record_count": len(rows),
        }


class SqlVectorTopologyRepository:
    """VectorTopologyService 的 SQL adapter；不暴露任何 Milvus 写操作。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def find_family(
        self, *, physical_database: str, physical_collection: str
    ) -> VectorCollectionFamily | None:
        row = self.session.scalar(
            select(VectorCollectionFamilyModel).where(
                VectorCollectionFamilyModel.physical_database == physical_database,
                VectorCollectionFamilyModel.physical_collection == physical_collection,
            )
        )
        return self._family(row) if row else None

    def save_family(self, family: VectorCollectionFamily) -> None:
        row = self.session.get(VectorCollectionFamilyModel, family.family_id)
        if row is None:
            self.session.add(
                VectorCollectionFamilyModel(
                    id=family.family_id,
                    physical_database=family.physical_database,
                    physical_collection=family.physical_collection,
                    placement_mode=family.placement_mode.value,
                    compatibility_fingerprint=family.compatibility_fingerprint,
                    schema_fingerprint=family.schema_fingerprint,
                    index_fingerprint=family.index_fingerprint,
                    compatibility_payload=family.compatibility_payload,
                    partition_strategy=family.partition_strategy,
                    num_partitions=family.num_partitions,
                    status=family.status.value,
                    epoch=family.epoch,
                )
            )
        elif self._family(row) != family:
            raise ResourceConflictError("vector family is immutable outside lifecycle transitions")
        self.session.flush()

    def save_placement(self, placement: VectorPlacement) -> None:
        row = self.session.get(VectorPlacementModel, placement.placement_id)
        policy_id = getattr(placement, "policy_id", "placement-policy/legacy")
        policy_version = int(getattr(placement, "policy_version", 0))
        reasons = list(getattr(placement, "reason_codes", ()))
        if row is None:
            self.session.add(
                VectorPlacementModel(
                    id=placement.placement_id,
                    tenant_id=placement.tenant_id,
                    kb_id=placement.kb_id,
                    index_version_id=placement.index_version_id,
                    family_id=placement.family_id,
                    partition_bucket=placement.partition_bucket,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    placement_reason_json=reasons,
                    status=placement.status.value,
                    readable=placement.readable,
                    writable=placement.writable,
                    read_slot="active" if placement.readable else None,
                    write_slot="active" if placement.writable else None,
                    epoch=placement.epoch,
                )
            )
        else:
            if (
                row.tenant_id != placement.tenant_id
                or row.kb_id != placement.kb_id
                or row.index_version_id != placement.index_version_id
                or row.family_id != placement.family_id
            ):
                raise ResourceConflictError("vector placement identity is immutable")
        self.session.flush()

    def activate_planned_placement(
        self,
        *,
        family_id: str,
        placement_id: str,
        readable: bool,
        writable: bool,
    ) -> None:
        """CAS 风格推进 family/placement；歧义槽位一律拒绝。"""

        family = self.session.get(VectorCollectionFamilyModel, family_id)
        placement = self.session.get(VectorPlacementModel, placement_id)
        if family is None or placement is None or placement.family_id != family_id:
            raise ResourceNotFoundError("planned vector family/placement not found")
        if family.status not in {
            FamilyStatus.CREATING.value,
            FamilyStatus.ACTIVE.value,
        }:
            raise ResourceConflictError(f"vector family cannot be activated from {family.status}")
        if placement.status not in {
            PlacementStatus.PLANNED.value,
            PlacementStatus.ACTIVE.value,
        }:
            raise ResourceConflictError(
                f"vector placement cannot be activated from {placement.status}"
            )
        slot_checks = []
        if readable:
            slot_checks.append(VectorPlacementModel.readable.is_(True))
        if writable:
            slot_checks.append(VectorPlacementModel.writable.is_(True))
        conflicts = self.session.scalars(
            select(VectorPlacementModel).where(
                VectorPlacementModel.index_version_id == placement.index_version_id,
                VectorPlacementModel.id != placement.id,
                or_(*slot_checks),
            )
        ).all()
        if conflicts:
            raise ResourceConflictError("index already has another active vector placement slot")
        family.status = FamilyStatus.ACTIVE.value
        placement.status = PlacementStatus.ACTIVE.value
        placement.readable = readable
        placement.read_slot = "active" if readable else None
        placement.writable = writable
        placement.write_slot = "active" if writable else None
        self.session.flush()

    def list_read_placements(
        self,
        *,
        tenant_id: str,
        kb_ids,
        index_version_ids,
    ) -> list[VectorPlacement]:
        rows = self.session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(
                VectorPlacementModel.tenant_id == tenant_id,
                VectorPlacementModel.kb_id.in_(tuple(kb_ids)),
                VectorPlacementModel.index_version_id.in_(tuple(index_version_ids)),
                VectorPlacementModel.readable.is_(True),
                VectorPlacementModel.read_slot == "active",
            )
        ).all()
        return [self._placement(placement, family) for placement, family in rows]

    def save_gc_plan(self, plan: VectorGcPlan) -> None:
        existing = self.session.get(VectorGcRunModel, plan.gc_run_id)
        if existing is not None:
            if existing.scope_json != plan.scope.model_dump(mode="json"):
                raise ResourceConflictError("vector GC idempotency payload changed")
            return
        self.session.add(
            VectorGcRunModel(
                id=plan.gc_run_id,
                idempotency_key=plan.idempotency_key,
                tenant_id=plan.scope.tenant_id,
                kb_id=plan.scope.kb_id,
                document_id=plan.scope.document_id,
                document_version_id=plan.scope.document_version_id,
                index_version_id=plan.scope.index_version_id,
                family_id=plan.scope.family_id,
                scope_json=plan.scope.model_dump(mode="json"),
                expected_count=plan.expected_count,
                status=plan.status.value,
                attempt=plan.attempt,
                fencing_token=plan.fencing_token,
                result_json={"dry_run": plan.dry_run},
            )
        )
        self.session.flush()

    def save_gc_decision(self, *, gc_run_id: str, status: GcRunStatus, payload: dict) -> None:
        row = self.session.get(VectorGcRunModel, gc_run_id)
        if row is None:
            raise ResourceNotFoundError("vector GC run not found")
        row.status = status.value
        row.result_json = payload
        observation = payload.get("observation") or {}
        reconciliation = payload.get("reconciliation") or {}
        row.matched_before = observation.get("matched_count", reconciliation.get("matched_before"))
        row.delete_acknowledged = reconciliation.get("delete_acknowledged")
        row.remaining_after = reconciliation.get("remaining_after")
        row.compaction_status = reconciliation.get("compaction_status")
        if status in {
            GcRunStatus.SUCCEEDED,
            GcRunStatus.BLOCKED_MANUAL,
            GcRunStatus.CANCELLED,
        }:
            row.completed_at = _now()
        self.session.flush()

    def save_migration(
        self,
        manifest: VectorMigrationManifest,
        *,
        status: MigrationStatus,
        payload: dict,
    ) -> None:
        row = self.session.get(VectorMigrationRunModel, manifest.migration_run_id)
        if row is None:
            row = VectorMigrationRunModel(
                id=manifest.migration_run_id,
                tenant_id=manifest.tenant_id,
                kb_id=manifest.kb_id,
                index_version_id=manifest.index_version_id,
                source_family_id=manifest.source_family_id,
                target_family_id=manifest.target_family_id,
                expected_count=manifest.expected_count,
                chunk_ids_sha256=manifest.chunk_ids_sha256,
                target_compatibility_fingerprint=(manifest.target_compatibility_fingerprint),
                status=status.value,
                checkpoint_json=payload,
                validation_json={},
                rollback_deadline=manifest.rollback_deadline,
            )
            self.session.add(row)
        else:
            row.status = status.value
            row.checkpoint_json = payload
            if payload.get("validation") is not None:
                row.validation_json = payload["validation"]
            if payload.get("placement_epoch") is not None:
                row.placement_epoch = int(payload["placement_epoch"])
        if status in {
            MigrationStatus.COMPLETED,
            MigrationStatus.ROLLED_BACK,
            MigrationStatus.BLOCKED_MANUAL,
            MigrationStatus.CANCELLED,
        }:
            row.completed_at = _now()
        self.session.flush()

    def atomic_switch_placement(
        self,
        *,
        migration_run_id: str,
        source_family_id: str,
        target_family_id: str,
        expected_epoch: int,
    ) -> int:
        migration = self.session.get(VectorMigrationRunModel, migration_run_id)
        if migration is None:
            raise ResourceNotFoundError("vector migration run not found")
        source = self.session.scalar(
            select(VectorPlacementModel).where(
                VectorPlacementModel.tenant_id == migration.tenant_id,
                VectorPlacementModel.kb_id == migration.kb_id,
                VectorPlacementModel.index_version_id == migration.index_version_id,
                VectorPlacementModel.family_id == source_family_id,
                VectorPlacementModel.readable.is_(True),
            )
        )
        target = self.session.scalar(
            select(VectorPlacementModel).where(
                VectorPlacementModel.tenant_id == migration.tenant_id,
                VectorPlacementModel.kb_id == migration.kb_id,
                VectorPlacementModel.index_version_id == migration.index_version_id,
                VectorPlacementModel.family_id == target_family_id,
            )
        )
        if source is None or target is None:
            raise ResourceConflictError("source/target vector placement is incomplete")
        if source.epoch != expected_epoch:
            raise ResourceConflictError("vector placement epoch conflict")
        next_epoch = expected_epoch + 1
        source.readable = False
        source.writable = False
        source.read_slot = None
        source.write_slot = None
        source.status = PlacementStatus.SOURCE.value
        source.epoch = next_epoch
        target.readable = True
        target.writable = True
        target.read_slot = "active"
        target.write_slot = "active"
        target.status = PlacementStatus.ACTIVE.value
        target.epoch = next_epoch
        migration.placement_epoch = next_epoch
        self.session.flush()
        return next_epoch

    @staticmethod
    def _family(row: VectorCollectionFamilyModel) -> VectorCollectionFamily:
        return VectorCollectionFamily(
            family_id=row.id,
            physical_database=row.physical_database,
            physical_collection=row.physical_collection,
            placement_mode=PlacementMode(row.placement_mode),
            compatibility_fingerprint=row.compatibility_fingerprint,
            schema_fingerprint=row.schema_fingerprint,
            index_fingerprint=row.index_fingerprint,
            compatibility_payload=row.compatibility_payload,
            partition_strategy=row.partition_strategy,
            num_partitions=row.num_partitions,
            status=row.status,
            epoch=row.epoch,
        )

    @staticmethod
    def _placement(
        row: VectorPlacementModel,
        family: VectorCollectionFamilyModel,
    ) -> VectorPlacement:
        payload = {
            "placement_id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "index_version_id": row.index_version_id,
            "family_id": row.family_id,
            "physical_database": family.physical_database,
            "physical_collection": family.physical_collection,
            "partition_bucket": row.partition_bucket,
            "status": row.status,
            "epoch": row.epoch,
            "readable": row.readable,
            "writable": row.writable,
        }
        fields = VectorPlacement.model_fields
        if "policy_id" in fields:
            payload.update(
                {
                    "policy_id": row.policy_id,
                    "policy_version": row.policy_version,
                    "reason_codes": tuple(row.placement_reason_json),
                }
            )
        return VectorPlacement.model_validate(payload)


__all__ = ["QueryReliabilityRepository", "SqlVectorTopologyRepository"]
