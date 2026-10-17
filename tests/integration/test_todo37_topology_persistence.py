"""TODO-37 vector topology control facts on real SQLite/Alembic."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from app.application.vector_topology_service import VectorTopologyService
from app.domain.vector_topology import (
    CompatibilityProfile,
    DenseVectorContract,
    GcProtectionFacts,
    GcRunStatus,
    PartitionContract,
    PlacementMode,
    PlacementPolicy,
    PlacementRequest,
    ScalarFieldContract,
    VectorGcScope,
)
from app.infra.persistence.sql.models import (
    IndexVersionModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.p3_models import (
    VectorGcRunModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.p3_repository import SqlVectorTopologyRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _database(tmp_path: Path):
    url = f"sqlite:///{(tmp_path / 'topology.db').as_posix()}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, "head")
    return create_engine(url, future=True)


def _seed(session: Session) -> None:
    session.add(TenantModel(id="tenant_a", name="Tenant A", status="active"))
    session.add(
        WorkspaceModel(
            id="workspace_a",
            tenant_id="tenant_a",
            name="Default",
            status="active",
        )
    )
    session.add(
        KnowledgeBaseModel(
            id="kb_hak180",
            tenant_id="tenant_a",
            workspace_id="workspace_a",
            name="HAK180",
            status="active",
            created_by="tester",
        )
    )
    session.add(
        IndexVersionModel(
            id="idx_v1",
            kb_id="kb_hak180",
            version_no=1,
            embedding_model_id="BAAI/bge-m3",
            vector_dim=1024,
            schema_version="chunks.v2",
            status="active",
            active_slot="active",
            collection_alias="kb_chunks",
        )
    )
    session.commit()


def _profile() -> CompatibilityProfile:
    return CompatibilityProfile(
        schema_version="chunks.v2",
        chunk_identity_schema_version="chunk-id.v1",
        evidence_identity_schema_version="evidence.v1",
        scalar_fields=(
            ScalarFieldContract(name="tenant_id", data_type="VARCHAR", max_length=64),
            ScalarFieldContract(name="kb_id", data_type="VARCHAR", max_length=64),
            ScalarFieldContract(name="available_int", data_type="INT8"),
        ),
        dense=DenseVectorContract(
            field_name="dense_vector",
            dimension=1024,
            provider="BAAI",
            model_id="bge-m3",
            revision="rev-2026-07",
            tokenizer="bge-m3-tokenizer",
            normalization="l2",
            metric="IP",
            index_type="AUTOINDEX",
        ),
        partition=PartitionContract(
            strategy="bounded_hash",
            key_field="tenant_id",
            num_partitions=64,
        ),
        preprocessing_version="normalize.v1",
    )


def test_migration_and_repository_persist_auditable_topology_facts(tmp_path: Path):
    engine = _database(tmp_path)
    try:
        assert {
            "vector_collection_families",
            "vector_placements",
            "vector_gc_runs",
            "vector_migration_runs",
        } <= set(inspect(engine).get_table_names())
        with Session(engine, expire_on_commit=False) as session:
            _seed(session)
            repository = SqlVectorTopologyRepository(session)
            service = VectorTopologyService(repository)
            arguments = {
                "policy": PlacementPolicy(
                    policy_id="placement.default",
                    version=7,
                    shared_partition_count=64,
                ),
                "request": PlacementRequest(
                    tenant_id="tenant_a",
                    kb_id="kb_hak180",
                    index_version_id="idx_v1",
                ),
                "profile": _profile(),
            }
            family, placement, decision = service.plan_placement(**arguments)
            repeated = service.plan_placement(**arguments)
            session.commit()

            assert repeated == (family, placement, decision)
            assert decision.mode == PlacementMode.SHARED_COMPATIBLE
            persisted = session.get(VectorPlacementModel, placement.placement_id)
            assert persisted is not None
            assert persisted.policy_id == "placement.default"
            assert persisted.policy_version == 7
            assert persisted.placement_reason_json == ["compatible_shared_default"]
            assert persisted.partition_bucket == decision.partition_bucket
    finally:
        engine.dispose()


def test_read_set_and_gc_scope_are_persisted_fail_closed(tmp_path: Path):
    engine = _database(tmp_path)
    try:
        with Session(engine, expire_on_commit=False) as session:
            _seed(session)
            repository = SqlVectorTopologyRepository(session)
            service = VectorTopologyService(repository)
            _family, placement, _decision = service.plan_placement(
                policy=PlacementPolicy(
                    policy_id="placement.default",
                    version=1,
                    shared_partition_count=64,
                ),
                request=PlacementRequest(
                    tenant_id="tenant_a",
                    kb_id="kb_hak180",
                    index_version_id="idx_v1",
                ),
                profile=_profile(),
            )
            row = session.get(VectorPlacementModel, placement.placement_id)
            row.status = "active"
            row.readable = True
            row.writable = True
            row.read_slot = "active"
            row.write_slot = "active"
            session.flush()

            read_set = service.freeze_read_set(
                tenant_id="tenant_a",
                kb_ids=("kb_hak180",),
                index_version_ids=("idx_v1",),
            )
            assert read_set.routes[0].index_version_ids == ("idx_v1",)
            assert read_set.routes[0].physical_collection.startswith("stt_chunks_c_")

            plan, eligibility = service.create_gc_plan(
                scope=VectorGcScope(
                    tenant_id="tenant_a",
                    kb_id="kb_hak180",
                    document_id="doc_hak180",
                    document_version_id="dv_hak180",
                    index_version_id="idx_v1",
                    family_id=placement.family_id,
                    physical_database=placement.physical_database,
                    physical_collection=placement.physical_collection,
                ),
                expected_count=32,
                protection=GcProtectionFacts(
                    active_binding=True,
                    placement_mode=PlacementMode.SHARED_COMPATIBLE,
                    family_reference_count=1,
                ),
            )
            session.commit()

            assert eligibility.eligible is False
            assert plan.status == GcRunStatus.BLOCKED_MANUAL
            persisted_gc = session.scalar(
                select(VectorGcRunModel).where(VectorGcRunModel.id == plan.gc_run_id)
            )
            assert persisted_gc is not None
            assert (
                persisted_gc.tenant_id,
                persisted_gc.kb_id,
                persisted_gc.document_id,
                persisted_gc.document_version_id,
                persisted_gc.index_version_id,
            ) == (
                "tenant_a",
                "kb_hak180",
                "doc_hak180",
                "dv_hak180",
                "idx_v1",
            )
            assert persisted_gc.status == "blocked_manual"
    finally:
        engine.dispose()
