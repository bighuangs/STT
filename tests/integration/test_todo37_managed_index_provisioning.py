from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, delete, select, text

from app.infra.persistence.sql.database import (
    configure_database,
    initialize_control_database,
    session_scope,
)
from app.infra.persistence.sql.index_lifecycle_repository import (
    IndexLifecycleRepository,
)
from app.infra.persistence.sql.models import IndexVersionModel
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.repositories import ControlRepository
from app.infra.vectorstore.write_compatibility import (
    VectorWriteGateError,
    resolve_import_vector_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fresh_database(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'managed-placement.db').as_posix()}")
    initialize_control_database()


def test_fresh_index_is_managed_and_has_one_persistent_target(tmp_path) -> None:
    _fresh_database(tmp_path)
    with session_scope() as session:
        kb = ControlRepository(session).ensure_defaults()
        index = ControlRepository(session).get_active_index(kb.id)
        placement = session.scalar(
            select(VectorPlacementModel).where(VectorPlacementModel.index_version_id == index.id)
        )
        assert index.vector_governance_mode == "managed"
        assert index.collection_alias.startswith("stt_chunks_k_")
        assert placement is not None
        assert placement.readable is True
        assert placement.writable is True
        assert placement.read_slot == placement.write_slot == "active"
        family = session.get(VectorCollectionFamilyModel, placement.family_id)
        assert family is not None
        assert family.status == "active"
        assert family.physical_collection == index.collection_alias
        index_id = index.id
        kb_id = kb.id

    target = resolve_import_vector_target(
        {
            "tenant_id": "tenant_default",
            "kb_id": kb_id,
            "index_version_id": index_id,
            "collection_name": index.collection_alias,
        },
        fallback_collection="kb_chunks",
    )
    assert target.managed is True
    assert target.physical_collection.startswith("stt_chunks_k_")


def test_managed_missing_placement_fails_closed_but_marked_legacy_is_compatible(
    tmp_path,
) -> None:
    _fresh_database(tmp_path)
    with session_scope() as session:
        kb = ControlRepository(session).ensure_defaults()
        index = ControlRepository(session).get_active_index(kb.id)
        identity = {
            "tenant_id": "tenant_default",
            "kb_id": kb.id,
            "index_version_id": index.id,
            "collection_name": index.collection_alias,
        }
        session.execute(
            delete(VectorPlacementModel).where(VectorPlacementModel.index_version_id == index.id)
        )

    with pytest.raises(VectorWriteGateError, match="拒绝回退"):
        resolve_import_vector_target(identity, fallback_collection="kb_chunks")

    with session_scope() as session:
        index = session.get(IndexVersionModel, identity["index_version_id"])
        assert index is not None
        index.vector_governance_mode = "legacy"
    target = resolve_import_vector_target(identity, fallback_collection="kb_chunks")
    assert target.managed is False
    assert target.legacy_reason == "migration_marked_pre_control_plane_index"


def test_rebuild_publish_and_rollback_switch_vector_slots_atomically(tmp_path) -> None:
    _fresh_database(tmp_path)
    with session_scope() as session:
        kb = ControlRepository(session).ensure_defaults()
        old_index = ControlRepository(session).get_active_index(kb.id)
        old_id = old_index.id
        lifecycle = IndexLifecycleRepository(session)
        rebuild = lifecycle.create_rebuild(kb.id)
        new_id = rebuild["index_version"]["id"]
        new_placement = session.scalar(
            select(VectorPlacementModel).where(VectorPlacementModel.index_version_id == new_id)
        )
        assert new_placement is not None
        assert new_placement.writable is True
        assert new_placement.readable is False
        lifecycle.publish(new_id)

        old_placement = session.scalar(
            select(VectorPlacementModel).where(VectorPlacementModel.index_version_id == old_id)
        )
        assert old_placement is not None
        assert old_placement.status == "retired"
        assert old_placement.readable is old_placement.writable is False
        assert new_placement.status == "active"
        assert new_placement.readable is new_placement.writable is True

        lifecycle.rollback(kb.id)
        assert old_placement.status == "active"
        assert old_placement.readable is old_placement.writable is True
        assert new_placement.status == "retired"
        assert new_placement.readable is new_placement.writable is False


@pytest.mark.parametrize("resume_after_column_add", [False, True])
def test_0012_marks_only_preexisting_rows_legacy_and_defaults_new_rows_managed(
    tmp_path,
    resume_after_column_add: bool,
) -> None:
    suffix = "resume" if resume_after_column_add else "clean"
    url = f"sqlite:///{(tmp_path / f'upgrade-generation-{suffix}.db').as_posix()}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, "20260725_0011")
    engine = create_engine(url, future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, name, status) "
                "VALUES ('tenant_upgrade', 'Tenant Upgrade', 'active')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO workspaces (id, tenant_id, name, status) "
                "VALUES ('workspace_upgrade', 'tenant_upgrade', 'Default', 'active')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO knowledge_bases "
                "(id, tenant_id, workspace_id, name, description, status, created_by, "
                "data_classification, external_llm_allowed, web_search_allowed) "
                "VALUES ('kb_upgrade', 'tenant_upgrade', 'workspace_upgrade', "
                "'Upgrade KB', '', 'active', 'tester', 'internal', 1, 0)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO index_versions "
                "(id, kb_id, version_no, embedding_model_id, vector_dim, schema_version, "
                "status, active_slot, collection_alias, build_kind, validation_status, "
                "manifest_summary) "
                "VALUES ('idx_old', 'kb_upgrade', 1, 'bge-m3', 3, 'chunks_v1', "
                "'active', 'active', 'kb_chunks', 'initial', 'validated', '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, kb_id, logical_name, source_type, status) "
                "VALUES ('doc_upgrade', 'kb_upgrade', 'manual.pdf', 'upload', 'ready')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions "
                "(id, document_id, version_no, content_sha256, size_bytes, mime_type, "
                "object_key, parser_config_snapshot, status, pipeline_fingerprint) "
                "VALUES ('dv_upgrade', 'doc_upgrade', 1, :content_sha256, 1, "
                "'application/pdf', 'manual.pdf', '{}', 'ready', :pipeline_fingerprint)"
            ),
            {
                "content_sha256": "1" * 64,
                "pipeline_fingerprint": "2" * 64,
            },
        )
        connection.execute(
            text(
                "INSERT INTO document_index_bindings "
                "(document_version_id, index_version_id, chunk_count, token_count, status) "
                "VALUES ('dv_upgrade', 'idx_old', 1, 1, 'published')"
            )
        )
        if resume_after_column_add:
            # 模拟 SQLite 非事务 DDL 在第一阶段成功、Alembic version 尚未推进
            # 时进程退出；下一次启动必须能安全续跑而非 duplicate column。
            connection.execute(
                text(
                    "ALTER TABLE index_versions ADD COLUMN vector_governance_mode "
                    "VARCHAR(16) DEFAULT 'legacy' NOT NULL"
                )
            )
    command.upgrade(config, "head")
    with engine.begin() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == "20260725_0012"
        )
        assert (
            connection.execute(
                text("SELECT vector_governance_mode FROM index_versions WHERE id = 'idx_old'")
            ).scalar_one()
            == "legacy"
        )
        assert (
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM document_index_bindings "
                    "WHERE document_version_id = 'dv_upgrade' AND index_version_id = 'idx_old'"
                )
            ).scalar_one()
            == 1
        )
        assert connection.execute(text("PRAGMA foreign_key_check")).fetchall() == []
        connection.execute(
            text(
                "INSERT INTO index_versions "
                "(id, kb_id, version_no, embedding_model_id, vector_dim, schema_version, "
                "status, collection_alias, build_kind, validation_status, manifest_summary) "
                "VALUES ('idx_new', 'kb_upgrade', 2, 'bge-m3', 3, 'chunks_v1', "
                "'staging', 'new_chunks', 'rebuild', 'pending', '{}')"
            )
        )
        assert (
            connection.execute(
                text("SELECT vector_governance_mode FROM index_versions WHERE id = 'idx_new'")
            ).scalar_one()
            == "managed"
        )
    engine.dispose()

    if not resume_after_column_add:
        # downgrade 也会重建被 binding 引用的表；必须保持 revision、引用行和
        # foreign_key_check 同时正确，而不是只验证 upgrade。
        command.downgrade(config, "20260725_0011")
        downgraded = create_engine(url, future=True)
        with downgraded.begin() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == "20260725_0011"
            )
            assert (
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM pragma_table_info('index_versions') "
                        "WHERE name = 'vector_governance_mode'"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM document_index_bindings "
                        "WHERE document_version_id = 'dv_upgrade' "
                        "AND index_version_id = 'idx_old'"
                    )
                ).scalar_one()
                == 1
            )
            assert connection.execute(text("PRAGMA foreign_key_check")).fetchall() == []
        downgraded.dispose()
        command.upgrade(config, "head")
        reupgraded = create_engine(url, future=True)
        with reupgraded.begin() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == "20260725_0012"
            )
            assert connection.execute(text("PRAGMA foreign_key_check")).fetchall() == []
        reupgraded.dispose()
