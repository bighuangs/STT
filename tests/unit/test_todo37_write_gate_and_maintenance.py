from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.application.vector_maintenance_service import (
    CopyReceipt,
    DeleteReceipt,
    GcExecutionContext,
    MigrationExecutionContext,
    VectorIndexMigrationScope,
    VectorMaintenanceLease,
    VectorMaintenanceLeaseLost,
    VectorMaintenanceTransientError,
    VectorMaintenanceWorkerHandler,
)
from app.domain.vector_topology import (
    GcRunStatus,
    MigrationStatus,
    MigrationValidation,
    VectorGcPlan,
    VectorGcScope,
    VectorIdentity,
    VectorMigrationManifest,
    compatibility_fingerprint,
)
from app.infra.vectorstore.write_compatibility import (
    LegacyWritePolicy,
    MilvusWriteCompatibilityGate,
    ResolvedVectorWriteTarget,
    current_chunks_profile,
)
from app.infra.vectorstore.milvus_maintenance_gateway import (
    GuardedMilvusMaintenanceGateway,
)


def _profile():
    return current_chunks_profile(
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3@rev1",
    )


class _PhysicalClient:
    def __init__(
        self,
        *,
        dim: int = 3,
        fail_describe: bool = False,
        sdk_type_key: bool = False,
    ) -> None:
        self.dim = dim
        self.fail_describe = fail_describe
        self.sdk_type_key = sdk_type_key

    def has_collection(self, **_: Any) -> bool:
        return True

    def describe_collection(self, **_: Any) -> dict[str, Any]:
        if self.fail_describe:
            raise TimeoutError("milvus temporarily unavailable")
        profile = _profile()
        numeric_types = {
            "BOOL": 1,
            "INT8": 2,
            "INT16": 3,
            "INT32": 4,
            "INT64": 5,
            "FLOAT": 10,
            "DOUBLE": 11,
            "STRING": 20,
            "VARCHAR": 21,
            "JSON": 23,
            "FLOAT_VECTOR": 101,
            "SPARSE_FLOAT_VECTOR": 104,
        }
        type_key = "type" if self.sdk_type_key else "data_type"
        fields = []
        for field in profile.scalar_fields:
            fields.append(
                {
                    "name": field.name,
                    type_key: (
                        numeric_types[field.data_type]
                        if self.sdk_type_key
                        else field.data_type
                    ),
                    "is_primary": field.primary_key,
                    "nullable": field.nullable,
                    "params": (
                        {"max_length": field.max_length}
                        if field.max_length is not None
                        else {}
                    ),
                }
            )
        fields.extend(
            [
                {
                    "name": "dense_vector",
                    type_key: 101 if self.sdk_type_key else "FLOAT_VECTOR",
                    "params": {"dim": self.dim},
                },
                {
                    "name": "sparse_vector",
                    type_key: 104 if self.sdk_type_key else "SPARSE_FLOAT_VECTOR",
                },
            ]
        )
        return {"enable_dynamic_field": True, "fields": fields}

    def list_indexes(self, **_: Any) -> list[str]:
        return ["dense_vector_index", "sparse_vector_index"]

    def describe_index(self, *, index_name: str, **_: Any) -> dict[str, Any]:
        if index_name == "dense_vector_index":
            return {
                "field_name": "dense_vector",
                "index_type": "AUTOINDEX",
                "metric_type": "IP",
                "params": {},
            }
        return {
            "field_name": "sparse_vector",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }

    def list_partitions(self, **_: Any) -> list[str]:
        return ["_default"]


def _managed_target():
    profile = _profile()
    return ResolvedVectorWriteTarget(
        physical_database="default",
        physical_collection="managed_chunks",
        partition_bucket=None,
        expected_profile=profile,
        expected_fingerprint=compatibility_fingerprint(profile),
        placement_id="vp_1",
        placement_epoch=7,
        legacy_policy=LegacyWritePolicy.DENY,
    )


def test_write_gate_uses_describe_index_and_partition_and_rejects_drift():
    result = MilvusWriteCompatibilityGate(_PhysicalClient()).verify(
        _managed_target(),
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3@rev1",
    )
    assert result.managed is True
    assert result.inspected_partitions == ("_default",)

    with pytest.raises(Exception, match="compatibility|dimension|drift"):
        MilvusWriteCompatibilityGate(_PhysicalClient(dim=4)).verify(
            _managed_target(),
            vector_dim=3,
            schema_version="chunks_v1",
            embedding_model_id="bge-m3@rev1",
        )


def test_write_gate_accepts_current_pymilvus_type_key_without_weakening_drift_checks():
    """当前 pymilvus 用 ``type`` + 数字枚举返回字段类型。"""

    result = MilvusWriteCompatibilityGate(_PhysicalClient(sdk_type_key=True)).verify(
        _managed_target(),
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3@rev1",
    )
    assert result.managed is True

    with pytest.raises(Exception, match="compatibility|dimension|drift"):
        MilvusWriteCompatibilityGate(
            _PhysicalClient(dim=4, sdk_type_key=True)
        ).verify(
            _managed_target(),
            vector_dim=3,
            schema_version="chunks_v1",
            embedding_model_id="bge-m3@rev1",
        )


def test_write_gate_has_explicit_legacy_policy_and_dependency_fail_closed():
    legacy = ResolvedVectorWriteTarget(
        physical_database="default",
        physical_collection="kb_chunks",
        partition_bucket=None,
        expected_profile=None,
        expected_fingerprint=None,
        placement_id=None,
        placement_epoch=None,
        legacy_policy=LegacyWritePolicy.CONTROL_PLANE_LEGACY,
        legacy_reason="pre-placement control-plane alias",
    )
    result = MilvusWriteCompatibilityGate(_PhysicalClient()).verify(
        legacy,
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3@rev1",
    )
    assert result.legacy_policy == LegacyWritePolicy.CONTROL_PLANE_LEGACY

    with pytest.raises(TimeoutError):
        MilvusWriteCompatibilityGate(_PhysicalClient(fail_describe=True)).verify(
            legacy,
            vector_dim=3,
            schema_version="chunks_v1",
            embedding_model_id="bge-m3@rev1",
        )


def _gc_plan() -> VectorGcPlan:
    return VectorGcPlan(
        gc_run_id="vgc_1",
        idempotency_key="vgc-idem-1",
        scope=VectorGcScope(
            tenant_id="tenant_a",
            kb_id="kb_a",
            document_id="doc_a",
            document_version_id="dv_a",
            index_version_id="idx_a",
            family_id="vf_a",
            physical_database="default",
            physical_collection="chunks_a",
        ),
        expected_count=2,
        status=GcRunStatus.DRY_RUN_VALIDATED,
    )


def _identity() -> VectorIdentity:
    return VectorIdentity(
        tenant_id="tenant_a",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_a",
        index_version_id="idx_a",
    )


def _migration_context(
    *, status: MigrationStatus = MigrationStatus.TARGET_CREATED
) -> MigrationExecutionContext:
    digest = "a" * 64
    fingerprint = compatibility_fingerprint(_profile())
    manifest = VectorMigrationManifest(
        migration_run_id="vm_1",
        tenant_id="tenant_a",
        kb_id="kb_a",
        index_version_id="idx_a",
        source_family_id="vf_source",
        target_family_id="vf_target",
        expected_count=2,
        chunk_ids_sha256=digest,
        target_compatibility_fingerprint=fingerprint.compatibility,
        rollback_deadline=datetime.now(UTC) + timedelta(days=1),
    )
    return MigrationExecutionContext(
        manifest=manifest,
        scope=VectorIndexMigrationScope(
            tenant_id="tenant_a",
            kb_id="kb_a",
            index_version_id="idx_a",
            source_database="default",
            source_collection="source_chunks",
            target_database="default",
            target_collection="target_chunks",
        ),
        source_fingerprint=fingerprint,
        target_fingerprint=fingerprint,
        status=status,
        placement_epoch=4,
    )


class _FakeRepository:
    def __init__(self) -> None:
        self.gc_status: GcRunStatus | None = None
        self.migration_status: MigrationStatus | None = None
        self.lease_checks = 0
        self.lose_lease_at: int | None = None
        self.migration_context = _migration_context()

    @staticmethod
    def _lease(run_id: str) -> VectorMaintenanceLease:
        return VectorMaintenanceLease(
            run_id=run_id,
            owner_id="worker-a",
            lease_token="secret-token",
            fencing_token=9,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )

    def claim_gc(self, gc_run_id: str, **_: Any):
        return self._lease(gc_run_id), GcExecutionContext(
            plan=_gc_plan(),
            expected_fingerprint=compatibility_fingerprint(_profile()),
            dry_run_observed_count=2,
            dry_run_safe=True,
            dry_run_reason_codes=("scope_and_count_reconciled",),
        )

    def assert_gc_lease(self, _: VectorMaintenanceLease) -> None:
        self.lease_checks += 1
        if self.lose_lease_at == self.lease_checks:
            raise VectorMaintenanceLeaseLost("lost")

    def finish_gc(
        self,
        _: VectorMaintenanceLease,
        *,
        status: GcRunStatus,
        **__: Any,
    ) -> None:
        self.gc_status = status

    def fail_gc(
        self,
        _: VectorMaintenanceLease,
        *,
        status: GcRunStatus,
        **__: Any,
    ) -> None:
        self.gc_status = status

    def claim_migration(self, migration_run_id: str, **_: Any):
        return self._lease(migration_run_id), self.migration_context

    def assert_migration_lease(self, _: VectorMaintenanceLease) -> None:
        return None

    def finish_migration_backfill(self, _: VectorMaintenanceLease, **__: Any) -> None:
        self.migration_status = MigrationStatus.READY_TO_SWITCH

    def finish_migration_rollback(self, _: VectorMaintenanceLease, **__: Any) -> None:
        self.migration_status = MigrationStatus.TARGET_CREATED

    def fail_migration(
        self,
        _: VectorMaintenanceLease,
        *,
        status: MigrationStatus,
        **__: Any,
    ) -> None:
        self.migration_status = status


class _FakeGateway:
    mutations_enabled = True

    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.gc_delete_calls = 0
        self.source_delete_calls = 0
        self.target_delete_calls = 0
        self.gc_count_calls = 0
        self.gc_deleted = False
        self.target_count = 0

    def inspect_fingerprint(self, *, expected, **_: Any):
        if self.mode == "outage":
            raise VectorMaintenanceTransientError("short outage")
        if self.mode == "drift":
            changed = _profile().model_copy(
                update={"dense": _profile().dense.model_copy(update={"revision": "drift"})}
            )
            return compatibility_fingerprint(changed)
        return expected

    def count_gc_scope(self, _: VectorGcPlan) -> int:
        self.gc_count_calls += 1
        if self.mode == "residual" and self.gc_count_calls > 1:
            return 1
        if self.gc_deleted:
            return 0
        return 2

    def sample_gc_identities(self, *_: Any, **__: Any):
        return (_identity(),)

    def delete_gc_scope(self, *_: Any, **__: Any) -> DeleteReceipt:
        self.gc_delete_calls += 1
        deleted = 3 if self.mode == "overdelete" else 2
        self.gc_deleted = True
        return DeleteReceipt(
            acknowledged=True,
            deleted_count=deleted,
            compaction_status="completed",
        )

    def count_migration_source(self, _: MigrationExecutionContext) -> int:
        return 2

    def count_migration_target(self, _: MigrationExecutionContext) -> int:
        return self.target_count

    def copy_migration_scope(self, *_: Any, **__: Any) -> CopyReceipt:
        if self.mode == "copy_outage":
            self.target_count = 1
            raise VectorMaintenanceTransientError("copy interrupted")
        self.target_count = 2
        return CopyReceipt(
            acknowledged=True,
            copied_count=2,
            source_chunk_ids_sha256="a" * 64,
            target_chunk_ids_sha256="a" * 64,
        )

    def validate_migration(
        self,
        context: MigrationExecutionContext,
        _: CopyReceipt,
    ) -> MigrationValidation:
        digest = context.manifest.chunk_ids_sha256
        fingerprint = context.manifest.target_compatibility_fingerprint
        return MigrationValidation(
            expected_count=2,
            actual_count=2,
            expected_chunk_ids_sha256=digest,
            actual_chunk_ids_sha256=digest,
            expected_compatibility_fingerprint=fingerprint,
            actual_compatibility_fingerprint=fingerprint,
            sampled_payloads_match=True,
            sampled_search_passed=True,
            cross_tenant_hit_count=0,
        )

    def delete_unpublished_migration_target(self, *_: Any, **__: Any) -> DeleteReceipt:
        self.target_delete_calls += 1
        deleted = self.target_count
        self.target_count = 0
        return DeleteReceipt(acknowledged=True, deleted_count=deleted)


@pytest.mark.parametrize(
    ("mode", "expected", "delete_calls"),
    (
        ("outage", GcRunStatus.FAILED_RETRYABLE, 0),
        ("drift", GcRunStatus.BLOCKED_MANUAL, 0),
        ("overdelete", GcRunStatus.BLOCKED_MANUAL, 1),
        ("residual", GcRunStatus.PARTIAL, 1),
        ("success", GcRunStatus.SUCCEEDED, 1),
    ),
)
def test_gc_fault_injection_never_widens_delete(mode, expected, delete_calls):
    repository = _FakeRepository()
    gateway = _FakeGateway(mode)
    status = VectorMaintenanceWorkerHandler(repository, gateway).run_gc(
        "vgc_1",
        owner_id="worker-a",
    )
    assert status == expected
    assert gateway.gc_delete_calls == delete_calls
    assert gateway.source_delete_calls == 0


def test_gc_lost_fencing_prevents_mutation():
    repository = _FakeRepository()
    repository.lose_lease_at = 2
    gateway = _FakeGateway()
    with pytest.raises(VectorMaintenanceLeaseLost):
        VectorMaintenanceWorkerHandler(repository, gateway).run_gc(
            "vgc_1",
            owner_id="worker-a",
        )
    assert gateway.gc_delete_calls == 0


def test_migration_interruption_requires_exact_target_rollback_and_keeps_source():
    repository = _FakeRepository()
    gateway = _FakeGateway("copy_outage")
    handler = VectorMaintenanceWorkerHandler(repository, gateway)
    assert (
        handler.run_migration_backfill("vm_1", owner_id="worker-a")
        == MigrationStatus.FAILED_RETRYABLE
    )
    assert gateway.target_count == 1
    assert gateway.source_delete_calls == 0

    repository.migration_context = _migration_context(status=MigrationStatus.FAILED_RETRYABLE)
    assert (
        handler.rollback_unpublished_target("vm_1", owner_id="worker-a")
        == MigrationStatus.TARGET_CREATED
    )
    assert gateway.target_delete_calls == 1
    assert gateway.target_count == 0
    assert gateway.source_delete_calls == 0


def test_migration_rollback_blocks_unexpected_target_volume_without_delete():
    repository = _FakeRepository()
    gateway = _FakeGateway()
    gateway.target_count = 3
    status = VectorMaintenanceWorkerHandler(
        repository,
        gateway,
    ).rollback_unpublished_target("vm_1", owner_id="worker-a")
    assert status == MigrationStatus.BLOCKED_MANUAL
    assert gateway.target_delete_calls == 0
    assert gateway.source_delete_calls == 0


class _DeleteCaptureClient:
    def __init__(self) -> None:
        self.deletes: list[dict[str, Any]] = []

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.deletes.append(kwargs)
        return {"delete_count": 2}


def test_real_milvus_adapter_is_disabled_by_default_and_uses_full_scope_filter():
    client = _DeleteCaptureClient()
    disabled = GuardedMilvusMaintenanceGateway(client)
    with pytest.raises(Exception, match="默认关闭"):
        disabled.delete_gc_scope(
            _gc_plan(),
            operation_id="vgc-idem-1",
            fencing_token=1,
        )
    assert client.deletes == []

    enabled = GuardedMilvusMaintenanceGateway(
        client,
        allow_mutations=True,
    )
    receipt = enabled.delete_gc_scope(
        _gc_plan(),
        operation_id="vgc-idem-1",
        fencing_token=2,
    )
    assert receipt.deleted_count == 2
    expression = client.deletes[0]["filter"]
    for field in (
        "tenant_id",
        "kb_id",
        "doc_id",
        "document_version_id",
        "index_version_id",
    ):
        assert field in expression
