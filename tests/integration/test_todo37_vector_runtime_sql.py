from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.retrieval import RetrievalScope
from app.domain.vector_topology import (
    GcRunStatus,
    MigrationStatus,
    VectorGcScope,
    compatibility_fingerprint,
)
from app.infra.persistence.sql.database import (
    configure_database,
    initialize_control_database,
    session_scope,
)
from app.infra.persistence.sql.models import (
    DocumentIndexBindingModel,
    DocumentModel,
    DocumentVersionModel,
    IndexVersionModel,
    IngestionJobModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorGcRunModel,
    VectorMigrationRunModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.vector_maintenance_repository import (
    SqlVectorMaintenanceRepository,
)
from app.infra.persistence.sql.index_lifecycle_repository import (
    IndexLifecycleRepository,
)
from app.infra.retrieval.control_plane_resolvers import (
    SqlActiveScopeResolver,
    freeze_vector_read_set,
)
from app.infra.vectorstore.write_compatibility import current_chunks_profile
from app.infra.vectorstore.write_compatibility import resolve_import_vector_target
from app.application.vector_maintenance_service import VectorMaintenanceLeaseLost


def _seed_database() -> None:
    profile = current_chunks_profile(
        vector_dim=3,
        schema_version="chunks_v1",
        embedding_model_id="bge-m3@rev1",
    )
    fingerprint = compatibility_fingerprint(profile)
    now = datetime.now(UTC)
    scope = VectorGcScope(
        tenant_id="tenant_a",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_a",
        index_version_id="idx_a",
        family_id="vf_source",
        physical_database="default",
        physical_collection="physical_chunks_source",
    )
    with session_scope() as session:
        session.add(TenantModel(id="tenant_a", name="Tenant A", status="active"))
        session.flush()
        session.add(
            WorkspaceModel(
                id="workspace_a",
                tenant_id="tenant_a",
                name="Workspace A",
                status="active",
            )
        )
        session.flush()
        session.add(
            KnowledgeBaseModel(
                id="kb_a",
                tenant_id="tenant_a",
                workspace_id="workspace_a",
                name="KB A",
                status="active",
                created_by="user_a",
            )
        )
        session.flush()
        session.add(
            DocumentModel(
                id="doc_a",
                kb_id="kb_a",
                logical_name="manual.pdf",
                status="ready",
                active_version_id="dv_a",
            )
        )
        session.add(
            DocumentVersionModel(
                id="dv_a",
                document_id="doc_a",
                version_no=1,
                content_sha256="1" * 64,
                pipeline_fingerprint="2" * 64,
                size_bytes=1,
                mime_type="application/pdf",
                object_key="test/manual.pdf",
                status="ready",
            )
        )
        session.flush()
        session.add(
            IndexVersionModel(
                id="idx_a",
                kb_id="kb_a",
                version_no=1,
                embedding_model_id="bge-m3@rev1",
                vector_dim=3,
                schema_version="chunks_v1",
                status="active",
                active_slot="active",
                collection_alias="legacy_alias_must_not_win",
                validation_status="validated",
            )
        )
        session.flush()
        session.add(
            DocumentIndexBindingModel(
                document_version_id="dv_a",
                index_version_id="idx_a",
                status="published",
                chunk_count=2,
                token_count=2,
            )
        )
        for family_id, collection in (
            ("vf_source", "physical_chunks_source"),
            ("vf_target", "physical_chunks_target"),
        ):
            session.add(
                VectorCollectionFamilyModel(
                    id=family_id,
                    physical_database="default",
                    physical_collection=collection,
                    placement_mode="kb_dedicated",
                    compatibility_fingerprint=fingerprint.compatibility,
                    schema_fingerprint=fingerprint.schema_fingerprint,
                    index_fingerprint=fingerprint.index_fingerprint,
                    compatibility_payload=fingerprint.canonical_payload,
                    partition_strategy="none",
                    num_partitions=1,
                    status="active",
                    epoch=1,
                )
            )
        session.flush()
        session.add(
            VectorPlacementModel(
                id="vp_source",
                tenant_id="tenant_a",
                kb_id="kb_a",
                index_version_id="idx_a",
                family_id="vf_source",
                policy_id="policy",
                policy_version=1,
                placement_reason_json=["test"],
                status="active",
                readable=True,
                writable=True,
                read_slot="active",
                write_slot="active",
                epoch=1,
            )
        )
        session.add(
            VectorGcRunModel(
                id="vgc_1",
                idempotency_key="vgc-idem-1",
                tenant_id="tenant_a",
                kb_id="kb_a",
                document_id="doc_a",
                document_version_id="dv_a",
                index_version_id="idx_a",
                family_id="vf_source",
                scope_json=scope.model_dump(mode="json"),
                expected_count=2,
                status=GcRunStatus.DRY_RUN_VALIDATED.value,
                attempt=0,
                fencing_token=0,
                result_json={
                    "dry_run": True,
                    "observation": {"matched_count": 2},
                    "decision": {
                        "status": GcRunStatus.DRY_RUN_VALIDATED.value,
                        "safe_to_execute": True,
                        "reason_codes": ["scope_and_count_reconciled"],
                    },
                },
            )
        )
        session.add(
            VectorMigrationRunModel(
                id="vm_1",
                tenant_id="tenant_a",
                kb_id="kb_a",
                index_version_id="idx_a",
                source_family_id="vf_source",
                target_family_id="vf_target",
                expected_count=2,
                chunk_ids_sha256="a" * 64,
                target_compatibility_fingerprint=fingerprint.compatibility,
                status=MigrationStatus.TARGET_CREATED.value,
                checkpoint_json={},
                validation_json={},
                rollback_deadline=now + timedelta(days=1),
                placement_epoch=1,
                attempt=0,
                fencing_token=0,
            )
        )


def test_active_scope_prefers_persistent_physical_placement(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'placement.db').as_posix()}")
    initialize_control_database()
    _seed_database()
    scopes = SqlActiveScopeResolver().resolve(
        RetrievalScope(
            tenant_id="tenant_a",
            user_id="user_a",
            kb_ids=("kb_a",),
        )
    )
    assert len(scopes) == 1
    assert scopes[0].collection_name == "physical_chunks_source"
    assert scopes[0].tenant_id == "tenant_a"
    assert scopes[0].kb_id == "kb_a"
    assert scopes[0].document_version_id == "dv_a"
    assert scopes[0].index_version_id == "idx_a"

    write_target = resolve_import_vector_target(
        {
            "tenant_id": "tenant_a",
            "kb_id": "kb_a",
            "index_version_id": "idx_a",
            "collection_name": "legacy_alias_must_not_win",
        },
        fallback_collection="kb_chunks",
    )
    assert write_target.managed is True
    assert write_target.physical_collection == "physical_chunks_source"
    assert write_target.placement_epoch == 1


def test_active_scope_only_allows_explicit_legacy_without_placement(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'placement-generation.db').as_posix()}")
    initialize_control_database()
    _seed_database()
    scope = RetrievalScope(
        tenant_id="tenant_a",
        user_id="user_a",
        kb_ids=("kb_a",),
    )
    with session_scope() as session:
        placement = session.get(VectorPlacementModel, "vp_source")
        assert placement is not None
        session.delete(placement)

    with pytest.raises(
        Exception,
        match="managed 活动索引缺少可读 vector placement",
    ):
        SqlActiveScopeResolver().resolve(scope)

    with session_scope() as session:
        index = session.get(IndexVersionModel, "idx_a")
        assert index is not None
        index.vector_governance_mode = "legacy"

    scopes = SqlActiveScopeResolver().resolve(scope)
    assert len(scopes) == 1
    assert scopes[0].collection_name == "legacy_alias_must_not_win"
    assert scopes[0].partition_name is None


def test_rebuild_publish_freezes_only_the_new_active_placement(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'rebuild-publish.db').as_posix()}")
    initialize_control_database()
    _seed_database()
    with session_scope() as session:
        session.add(
            IndexVersionModel(
                id="idx_v2",
                kb_id="kb_a",
                version_no=2,
                embedding_model_id="bge-m3@rev1",
                vector_dim=3,
                schema_version="chunks_v1",
                status="staging",
                active_slot=None,
                collection_alias="physical_chunks_source",
                vector_governance_mode="managed",
                previous_version_id="idx_a",
                build_kind="rebuild",
                validation_status="validated",
            )
        )
        session.flush()
        session.add(
            DocumentIndexBindingModel(
                document_version_id="dv_a",
                index_version_id="idx_v2",
                status="staged",
                chunk_count=2,
                token_count=2,
            )
        )
        session.add(
            VectorPlacementModel(
                id="vp_v2",
                tenant_id="tenant_a",
                kb_id="kb_a",
                index_version_id="idx_v2",
                family_id="vf_source",
                policy_id="policy",
                policy_version=1,
                placement_reason_json=["rebuild"],
                status="active",
                readable=False,
                writable=True,
                read_slot=None,
                write_slot="active",
                epoch=1,
            )
        )
        session.flush()
        published = IndexLifecycleRepository(session).publish("idx_v2")
        assert published["status"] == "active"

    resolved, rows, digest = freeze_vector_read_set(
        RetrievalScope(
            tenant_id="tenant_a",
            user_id="user_a",
            kb_ids=("kb_a",),
        )
    )

    assert len(resolved) == len(rows) == 1
    assert resolved[0].index_version_id == "idx_v2"
    assert resolved[0].collection_name == "physical_chunks_source"
    assert rows[0]["index_version_id"] == "idx_v2"
    assert "idx_a" not in str(rows)
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_sql_gc_and_migration_leases_are_persistent_and_fenced(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'maintenance.db').as_posix()}")
    initialize_control_database()
    _seed_database()
    repository = SqlVectorMaintenanceRepository()

    gc_lease, gc_context = repository.claim_gc(
        "vgc_1",
        owner_id="worker-a",
        lease_seconds=60,
    )
    assert gc_lease.fencing_token == 1
    assert gc_context.dry_run_safe is True
    repository.assert_gc_lease(gc_lease)
    with pytest.raises(Exception, match="active lease"):
        repository.claim_gc(
            "vgc_1",
            owner_id="worker-b",
            lease_seconds=60,
        )

    with session_scope() as session:
        row = session.get(VectorGcRunModel, "vgc_1")
        assert row is not None
        row.fencing_token += 1
    with pytest.raises(VectorMaintenanceLeaseLost):
        repository.assert_gc_lease(gc_lease)

    migration_lease, context = repository.claim_migration(
        "vm_1",
        owner_id="worker-a",
        lease_seconds=60,
    )
    assert migration_lease.fencing_token == 1
    assert context.scope.source_collection == "physical_chunks_source"
    assert context.scope.target_collection == "physical_chunks_target"
    repository.assert_migration_lease(migration_lease)


def test_index_manifest_is_fenced_to_physical_write_placement(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'manifest.db').as_posix()}")
    initialize_control_database()
    _seed_database()
    fingerprint = compatibility_fingerprint(
        current_chunks_profile(
            vector_dim=3,
            schema_version="chunks_v1",
            embedding_model_id="bge-m3@rev1",
        )
    )
    with session_scope() as session:
        job = IngestionJobModel(
            id="job_a",
            document_version_id="dv_a",
            index_version_id="idx_a",
            status="running",
            stage="node_import_milvus",
            attempt=1,
            run_attempt=1,
            max_attempts=5,
        )
        session.add(job)
        session.flush()
        result = IndexLifecycleRepository(session).record_manifest(
            job,
            {
                "collection_name": "physical_chunks_source",
                "vector_dim": 3,
                "vector_placement_id": "vp_source",
                "vector_placement_epoch": 1,
                "vector_compatibility_fingerprint": fingerprint.compatibility,
                "vector_legacy_policy": "deny",
                "input_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "expected_chunk_count": 2,
                "written_chunk_count": 2,
                "searchable_chunk_count": 2,
                "chunk_ids_sha256": "d" * 64,
                "chunk_ids": ["chunk-1", "chunk-2"],
                "object_keys": [],
                "validation_status": "validated",
                "validation_errors": [],
            },
        )
        assert result["collection_name"] == "physical_chunks_source"
