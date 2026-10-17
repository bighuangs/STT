from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from app.application.vector_topology_service import VectorTopologyService
from app.domain.vector_topology import (
    CollectionInventorySnapshot,
    CompatibilityDriftError,
    CompatibilityFingerprint,
    CompatibilityProfile,
    DenseVectorContract,
    GcDryRunObservation,
    GcProtectionFacts,
    GcReconciliation,
    GcRunStatus,
    InventoryEntity,
    MigrationStatus,
    MigrationValidation,
    PartitionContract,
    PlacementMode,
    PlacementPolicy,
    PlacementRequest,
    PlacementStatus,
    ScalarFieldContract,
    SparseVectorContract,
    VectorCollectionFamily,
    VectorGcPlan,
    VectorGcScope,
    VectorIdentity,
    VectorPlacement,
    VectorTopologyError,
    audit_inventory,
    compatibility_fingerprint,
    ensure_migration_transition,
    evaluate_gc_eligibility,
    freeze_query_read_set,
    reconcile_gc,
    require_compatible,
    resolve_placement,
    validate_gc_dry_run,
)


def _profile(**updates: Any) -> CompatibilityProfile:
    values: dict[str, Any] = {
        "schema_version": "chunks.v2",
        "chunk_identity_schema_version": "chunk-id.v1",
        "evidence_identity_schema_version": "evidence.v1",
        "dynamic_fields_enabled": False,
        "scalar_fields": (
            ScalarFieldContract(name="tenant_id", data_type="varchar", max_length=64),
            ScalarFieldContract(name="kb_id", data_type="varchar", max_length=64),
            ScalarFieldContract(name="available_int", data_type="int8"),
        ),
        "dense": DenseVectorContract(
            field_name="dense_vector",
            dimension=1024,
            provider="BAAI",
            model_id="bge-m3",
            revision="rev-2026-07",
            tokenizer="bge-m3-tokenizer",
            normalization="l2",
            query_instruction="query:",
            document_instruction="passage:",
            metric="ip",
            index_type="autoinDEX",
            index_params={"nlist": 1024, "M": 16},
        ),
        "sparse": SparseVectorContract(
            field_name="sparse_vector",
            algorithm="bge-m3-lexical",
            analyzer="bge-m3-tokenizer",
            metric="ip",
            index_type="sparse_inverted_index",
            index_params={"drop_ratio_build": 0.2},
        ),
        "partition": PartitionContract(
            strategy="bounded_hash",
            key_field="tenant_id",
            num_partitions=64,
        ),
        "preprocessing_version": "normalize.v1",
        "quantization_version": "none",
    }
    values.update(updates)
    return CompatibilityProfile(**values)


def _policy(**updates: Any) -> PlacementPolicy:
    values = {
        "policy_id": "placement.default",
        "version": 3,
        "shared_partition_count": 64,
        "tenant_entity_threshold": 1_000,
        "kb_entity_threshold": 800,
        "hot_qps_threshold": 20,
    }
    values.update(updates)
    return PlacementPolicy(**values)


def _request(**updates: Any) -> PlacementRequest:
    values = {
        "tenant_id": "tenant_a",
        "kb_id": "kb_hak180",
        "index_version_id": "idx_v1",
    }
    values.update(updates)
    return PlacementRequest(**values)


def _scope(**updates: Any) -> VectorGcScope:
    values = {
        "tenant_id": "tenant_a",
        "kb_id": "kb_hak180",
        "document_id": "doc_manual",
        "document_version_id": "dv_1",
        "index_version_id": "idx_v1",
        "family_id": "vf_1",
        "physical_database": "default",
        "physical_collection": "stt_chunks_c_deadbeef",
    }
    values.update(updates)
    return VectorGcScope(**values)


def _validation(**updates: Any) -> MigrationValidation:
    digest = "a" * 64
    values = {
        "expected_count": 32,
        "actual_count": 32,
        "expected_chunk_ids_sha256": digest,
        "actual_chunk_ids_sha256": digest,
        "expected_compatibility_fingerprint": "b" * 64,
        "actual_compatibility_fingerprint": "b" * 64,
        "sampled_payloads_match": True,
        "sampled_search_passed": True,
        "cross_tenant_hit_count": 0,
    }
    values.update(updates)
    return MigrationValidation(**values)


def test_fingerprint_is_canonical_but_changes_for_every_semantic_contract():
    original = compatibility_fingerprint(_profile())
    reversed_fields = compatibility_fingerprint(
        _profile(scalar_fields=tuple(reversed(_profile().scalar_fields)))
    )
    assert original == reversed_fields

    variations = (
        _profile(dense=_profile().dense.model_copy(update={"revision": "different"})),
        _profile(dense=_profile().dense.model_copy(update={"dimension": 768})),
        _profile(dense=_profile().dense.model_copy(update={"metric": "COSINE"})),
        _profile(
            dense=_profile().dense.model_copy(update={"index_params": {"nlist": 2048, "M": 16}})
        ),
        _profile(
            partition=PartitionContract(
                strategy="bounded_hash",
                key_field="tenant_id",
                num_partitions=128,
            )
        ),
        _profile(preprocessing_version="normalize.v2"),
    )
    for variation in variations:
        assert compatibility_fingerprint(variation).compatibility != original.compatibility


def test_require_compatible_reports_field_path_and_fails_closed():
    expected = compatibility_fingerprint(_profile())
    actual = compatibility_fingerprint(
        _profile(dense=_profile().dense.model_copy(update={"revision": "rev-other"}))
    )
    with pytest.raises(CompatibilityDriftError) as caught:
        require_compatible(expected, actual)
    assert "$.dense.revision:value" in caught.value.differences


def test_fingerprint_payload_cannot_be_forged_independently_of_digest():
    fingerprint = compatibility_fingerprint(_profile())
    with pytest.raises(ValidationError, match="does not match"):
        CompatibilityFingerprint(
            compatibility=fingerprint.compatibility,
            schema_fingerprint=fingerprint.schema_fingerprint,
            index_fingerprint=fingerprint.index_fingerprint,
            canonical_payload={
                **fingerprint.canonical_payload,
                "preprocessing_version": "tampered",
            },
        )


def test_placement_is_server_derived_bounded_and_not_client_controllable():
    fingerprint = compatibility_fingerprint(_profile())
    first = resolve_placement(_policy(), _request(), fingerprint)
    second = resolve_placement(_policy(), _request(), fingerprint)
    assert first == second
    assert first.mode == PlacementMode.SHARED_COMPATIBLE
    assert first.partition_bucket is not None
    assert 0 <= int(first.partition_bucket.removeprefix("p_")) < 64
    assert "tenant_a" not in first.physical_collection
    assert "kb_hak180" not in first.physical_collection

    with pytest.raises(ValidationError):
        PlacementRequest.model_validate(
            {
                **_request().model_dump(),
                "physical_collection": "attacker_collection",
            }
        )


@pytest.mark.parametrize(
    ("updates", "mode"),
    (
        ({"compliance_tier": "isolated"}, PlacementMode.CLUSTER_DEDICATED),
        ({"special_vector_contract": True}, PlacementMode.KB_DEDICATED),
        ({"estimated_kb_entities": 800}, PlacementMode.KB_DEDICATED),
        ({"estimated_tenant_entities": 1_000}, PlacementMode.TENANT_DEDICATED),
        ({"estimated_qps": 20}, PlacementMode.TENANT_DEDICATED),
    ),
)
def test_placement_policy_escalates_isolation_tier(updates, mode):
    decision = resolve_placement(
        _policy(),
        _request(**updates),
        compatibility_fingerprint(_profile()),
    )
    assert decision.mode == mode
    if mode != PlacementMode.SHARED_COMPATIBLE:
        assert decision.partition_bucket is None


def test_gc_scope_requires_full_identity_and_escapes_filter_values():
    with pytest.raises(ValidationError):
        VectorGcScope.model_validate(
            {
                key: value
                for key, value in _scope().model_dump().items()
                if key != "document_version_id"
            }
        )
    scope = _scope(document_id='doc_"quoted')
    expression = scope.milvus_filter()
    assert 'doc_id == "doc_\\"quoted"' in expression
    for field in (
        "tenant_id",
        "kb_id",
        "doc_id",
        "document_version_id",
        "index_version_id",
    ):
        assert field in expression


@pytest.mark.parametrize(
    "updates",
    (
        {"active_binding": True},
        {"rollback_pinned": True},
        {"legal_hold": True},
        {"rollback_expires_at": datetime(2030, 1, 1, tzinfo=UTC)},
    ),
)
def test_gc_is_blocked_by_every_protection_boundary(updates):
    facts = GcProtectionFacts(
        placement_mode=PlacementMode.KB_DEDICATED,
        family_reference_count=0,
        **updates,
    )
    eligibility = evaluate_gc_eligibility(
        facts,
        now=datetime(2026, 7, 25, tzinfo=UTC),
    )
    assert eligibility.eligible is False
    assert eligibility.drop_family_allowed is False


def test_shared_family_can_delete_exact_scope_but_never_drop_family():
    facts = GcProtectionFacts(
        placement_mode=PlacementMode.SHARED_COMPATIBLE,
        family_reference_count=5,
    )
    eligibility = evaluate_gc_eligibility(facts)
    assert eligibility.eligible is True
    assert eligibility.drop_family_allowed is False

    plan = VectorGcPlan(
        gc_run_id="gc_1",
        idempotency_key="gc-idem-1",
        scope=_scope(),
        expected_count=3,
        drop_family_requested=False,
    )
    identity = VectorIdentity.model_validate(
        _scope().model_dump(
            include={
                "tenant_id",
                "kb_id",
                "document_id",
                "document_version_id",
                "index_version_id",
            }
        )
    )
    decision = validate_gc_dry_run(
        plan,
        GcDryRunObservation(
            matched_count=3,
            sampled_identities=(identity,),
        ),
        eligibility,
    )
    assert decision.safe_to_execute is True
    assert decision.status == GcRunStatus.DRY_RUN_VALIDATED


def test_gc_blocks_count_or_cross_scope_identity_mismatch():
    plan = VectorGcPlan(
        gc_run_id="gc_1",
        idempotency_key="gc-idem-1",
        scope=_scope(),
        expected_count=3,
    )
    eligibility = evaluate_gc_eligibility(
        GcProtectionFacts(
            placement_mode=PlacementMode.SHARED_COMPATIBLE,
            family_reference_count=1,
        )
    )
    foreign = VectorIdentity(
        tenant_id="tenant_b",
        kb_id="kb_hak180",
        document_id="doc_manual",
        document_version_id="dv_1",
        index_version_id="idx_v1",
    )
    decision = validate_gc_dry_run(
        plan,
        GcDryRunObservation(matched_count=4, sampled_identities=(foreign,)),
        eligibility,
    )
    assert decision.safe_to_execute is False
    assert set(decision.reason_codes) >= {
        "cross_scope_identity",
        "matched_count_mismatch",
    }


def test_gc_does_not_claim_physical_reclaim_before_compaction():
    pending = reconcile_gc(
        GcReconciliation(
            expected_count=3,
            matched_before=3,
            delete_acknowledged=True,
            remaining_after=0,
            compaction_status="pending",
        )
    )
    assert pending.status == GcRunStatus.AWAITING_COMPACTION
    assert pending.logical_delete_verified is True
    assert pending.physical_reclaimed is False

    completed = reconcile_gc(
        GcReconciliation(
            expected_count=3,
            matched_before=3,
            delete_acknowledged=True,
            remaining_after=0,
            compaction_status="completed",
        )
    )
    assert completed.status == GcRunStatus.SUCCEEDED
    assert completed.physical_reclaimed is True


def test_migration_cannot_skip_validation_or_close_rollback_window_early():
    with pytest.raises(VectorTopologyError, match="invalid vector migration"):
        ensure_migration_transition(
            MigrationStatus.BACKFILLING,
            MigrationStatus.SWITCHED,
        )
    with pytest.raises(VectorTopologyError, match="validation failed"):
        ensure_migration_transition(
            MigrationStatus.VALIDATING,
            MigrationStatus.READY_TO_SWITCH,
            validation=_validation(actual_count=31),
        )
    ensure_migration_transition(
        MigrationStatus.VALIDATING,
        MigrationStatus.READY_TO_SWITCH,
        validation=_validation(),
    )
    deadline = datetime(2026, 7, 26, tzinfo=UTC)
    with pytest.raises(VectorTopologyError, match="still open"):
        ensure_migration_transition(
            MigrationStatus.ROLLBACK_WINDOW,
            MigrationStatus.SOURCE_GC_ELIGIBLE,
            rollback_deadline=deadline,
            now=deadline - timedelta(seconds=1),
        )
    ensure_migration_transition(
        MigrationStatus.ROLLBACK_WINDOW,
        MigrationStatus.SOURCE_GC_ELIGIBLE,
        rollback_deadline=deadline,
        now=deadline,
    )


def test_query_read_set_is_fixed_and_rejects_ambiguous_index_placement():
    placement = VectorPlacement(
        placement_id="vp_1",
        tenant_id="tenant_a",
        kb_id="kb_hak180",
        index_version_id="idx_v1",
        family_id="vf_1",
        physical_database="default",
        physical_collection="stt_chunks_c_deadbeef",
        partition_bucket="p_0001",
        policy_id="placement.default",
        policy_version=3,
        reason_codes=("compatible_shared_default",),
        status=PlacementStatus.ACTIVE,
        epoch=7,
        readable=True,
        writable=True,
    )
    read_set = freeze_query_read_set((placement,))
    assert read_set.routes[0].placement_epoch == 7
    assert read_set.routes[0].index_version_ids == ("idx_v1",)
    assert len(read_set.digest) == 64

    other = placement.model_copy(
        update={
            "placement_id": "vp_2",
            "family_id": "vf_2",
            "physical_collection": "stt_chunks_c_cafebabe",
        }
    )
    with pytest.raises(VectorTopologyError, match="ambiguous"):
        freeze_query_read_set((placement, other))


def test_inventory_is_read_only_and_reports_missing_duplicates_and_drift():
    actual = compatibility_fingerprint(_profile())
    expected = compatibility_fingerprint(
        _profile(dense=_profile().dense.model_copy(update={"revision": "new"}))
    )
    snapshot = CollectionInventorySnapshot(
        physical_database="default",
        physical_collection="kb_chunks",
        entity_count=100,
        partitions=("_default",),
        indexes=({"index_name": "dense_vector_index"},),
        sampled_entities=(
            InventoryEntity(
                chunk_id="chunk_1",
                tenant_id="tenant_a",
                kb_id="kb_hak180",
                document_id="doc_manual",
                document_version_id="dv_1",
                index_version_id="idx_v1",
                available=True,
            ),
            InventoryEntity(chunk_id="chunk_1"),
        ),
        actual_fingerprint=actual,
    )
    report = audit_inventory(snapshot, expected_fingerprint=expected)
    assert report.dry_run is True
    assert report.destructive_actions == ()
    assert report.missing_identity_rows == (1,)
    assert report.duplicate_chunk_ids == ("chunk_1",)
    assert "$.dense.revision:value" in report.drift_paths


class _Repository:
    def __init__(self) -> None:
        self.family: VectorCollectionFamily | None = None
        self.placements: list[VectorPlacement] = []
        self.gc_plans: list[VectorGcPlan] = []
        self.events: list[tuple[str, Any]] = []

    def find_family(self, **_kwargs):
        return self.family

    def save_family(self, family):
        self.family = family

    def save_placement(self, placement):
        self.placements.append(placement)

    def list_read_placements(self, **_kwargs):
        return [item for item in self.placements if item.readable]

    def save_gc_plan(self, plan):
        self.gc_plans.append(plan)

    def save_gc_decision(self, **kwargs):
        self.events.append(("gc", kwargs))

    def save_migration(self, manifest, **kwargs):
        self.events.append(("migration", (manifest, kwargs)))

    def atomic_switch_placement(self, **_kwargs):
        return 8


def test_application_service_creates_idempotent_server_owned_facts():
    repository = _Repository()
    service = VectorTopologyService(repository)
    family, placement, decision = service.plan_placement(
        policy=_policy(),
        request=_request(),
        profile=_profile(),
    )
    assert repository.family == family
    assert repository.placements == [placement]
    assert family.compatibility_payload == compatibility_fingerprint(_profile()).canonical_payload
    assert decision.physical_collection == family.physical_collection
    assert placement.policy_id == decision.policy_id
    assert placement.policy_version == decision.policy_version
    assert placement.reason_codes == decision.reason_codes
    assert placement.readable is False
    assert placement.writable is False


def test_application_read_set_requires_every_requested_index():
    repository = _Repository()
    service = VectorTopologyService(repository)
    with pytest.raises(ValueError, match="lack an active placement"):
        service.freeze_read_set(
            tenant_id="tenant_a",
            kb_ids=("kb_hak180",),
            index_version_ids=("idx_v1",),
        )
