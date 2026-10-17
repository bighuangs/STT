"""TODO-09 检索配置版本和元数据目录的真实 SQLite 合同测试。"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, MetaData, create_engine, event, func, inspect, select
from sqlalchemy.orm import Session

from app.infra.persistence.sql.models import (
    DocumentModel,
    DocumentVersionModel,
    IndexVersionModel,
    KnowledgeBaseModel,
    MetadataCatalogEntryModel,
    QuerySessionModel,
    RetrievalProfileModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.infra.persistence.sql.retrieval_repository import (
    RetrievalProfileRepository,
    default_retrieval_profile_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _upgrade(database_url: str, target: str = "head") -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, target)


def _engine_for(path: Path) -> Engine:
    url = f"sqlite:///{path.as_posix()}"
    _upgrade(url)
    engine = create_engine(url, future=True)
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    return engine


def _seed_legacy_0012_builtin_timeout(engine: Engine) -> None:
    """Seed only the rows required by the v13 migration using the v12 schema.

    This test intentionally stops Alembic at 0012.  Reflecting that database
    keeps the fixture independent from columns subsequently added to the ORM
    models (for example the workbench ``revision`` columns in 0014).
    """
    metadata = MetaData()
    metadata.reflect(
        bind=engine,
        only=("tenants", "workspaces", "knowledge_bases", "retrieval_profiles"),
    )
    tenants = metadata.tables["tenants"]
    workspaces = metadata.tables["workspaces"]
    knowledge_bases = metadata.tables["knowledge_bases"]
    retrieval_profiles = metadata.tables["retrieval_profiles"]

    old_config = default_retrieval_profile_config()
    old_config["timeout_ms"] = 30_000
    canonical = json.dumps(
        old_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    profile_id = "rpv_builtin_timeout_v1"
    with engine.begin() as connection:
        connection.execute(
            tenants.insert().values(
                id="tenant_a",
                name="Tenant a",
                status="active",
            )
        )
        connection.execute(
            workspaces.insert().values(
                id="workspace_a",
                tenant_id="tenant_a",
                name="Default",
                status="active",
            )
        )
        connection.execute(
            knowledge_bases.insert().values(
                id="kb_a",
                tenant_id="tenant_a",
                workspace_id="workspace_a",
                name="KB a",
                description="",
                status="active",
                retrieval_config_id=profile_id,
                created_by="tester",
            )
        )
        connection.execute(
            retrieval_profiles.insert().values(
                id=profile_id,
                tenant_id="tenant_a",
                kb_id="kb_a",
                profile_id="rpf_default",
                version=1,
                name="内建兼容检索配置",
                description="TODO-09 自动创建；保留既有默认值。",
                status="published",
                config_json=old_config,
                config_hash=sha256(canonical.encode("utf-8")).hexdigest(),
                created_by="system",
                published_at=datetime.now(UTC),
            )
        )


def _seed(session: Session) -> None:
    for suffix in ("a", "b"):
        tenant_id = f"tenant_{suffix}"
        kb_id = f"kb_{suffix}"
        session.add(TenantModel(id=tenant_id, name=f"Tenant {suffix}", status="active"))
        session.add(
            WorkspaceModel(
                id=f"workspace_{suffix}",
                tenant_id=tenant_id,
                name="Default",
                status="active",
            )
        )
        session.flush()
        session.add(
            KnowledgeBaseModel(
                id=kb_id,
                tenant_id=tenant_id,
                workspace_id=f"workspace_{suffix}",
                name=f"KB {suffix}",
                status="active",
                created_by="tester",
            )
        )
        session.flush()
        session.add(
            IndexVersionModel(
                id=f"index_{suffix}",
                kb_id=kb_id,
                version_no=1,
                embedding_model_id="BAAI/bge-m3",
                vector_dim=1024,
                schema_version="stt-chunks-v2",
                status="active",
                active_slot="active",
                collection_alias=f"kb_chunks_{suffix}",
                manifest_summary={"dense": {"metric_type": "IP"}},
            )
        )
    session.flush()
    session.add(
        DocumentModel(
            id="doc_a",
            kb_id="kb_a",
            logical_name="manual.pdf",
            source_type="upload",
            status="ready",
            active_version_id="version_a",
        )
    )
    session.flush()
    session.add(
        DocumentVersionModel(
            id="version_a",
            document_id="doc_a",
            version_no=1,
            content_sha256="a" * 64,
            pipeline_fingerprint="pipeline",
            size_bytes=42,
            mime_type="application/pdf",
            object_key="manual.pdf",
            parser_config_snapshot={},
            status="published",
        )
    )
    session.flush()


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database_engine = _engine_for(tmp_path / "todo09.db")
    with Session(database_engine) as session:
        _seed(session)
        session.commit()
    try:
        yield database_engine
    finally:
        database_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session
        database_session.rollback()


def test_migration_creates_profiles_catalog_and_session_pin_columns(
    tmp_path: Path,
) -> None:
    engine = _engine_for(tmp_path / "migration.db")
    try:
        inspector = inspect(engine)
        assert {"retrieval_profiles", "metadata_catalog_entries"} <= set(
            inspector.get_table_names()
        )
        query_columns = {column["name"] for column in inspector.get_columns("query_sessions")}
        assert {
            "retrieval_profile_id",
            "retrieval_profile_version",
            "retrieval_profile_hash",
        } <= query_columns
    finally:
        engine.dispose()


def test_publish_rollback_tenant_isolation_and_session_pin(session: Session) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a", actor_id="admin_a")
    source = default_retrieval_profile_config()
    first = repository.create_draft(
        "kb_a",
        "baseline",
        source,
        profile_id="profile_search",
    )
    source["candidate_k"] = 999
    assert first.config_json["candidate_k"] == 5
    published_first = repository.publish("kb_a", first.profile_id, first.version, "admin_a")
    assert published_first.status == "published"
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == first.id

    pinned_first = repository.resolve_for_session("session_old", "kb_a", "user_a")
    second_config = copy.deepcopy(pinned_first.config_json)
    second_config["candidate_k"] = 30
    second = repository.create_draft(
        "kb_a",
        "candidate",
        second_config,
        profile_id=first.profile_id,
    )
    assert second.version == first.version + 1
    repository.publish("kb_a", second.profile_id, second.version)
    assert first.status == "retired"

    # 旧会话精确解析退役版本，新会话采用刚发布版本。
    assert repository.resolve_for_session("session_old").id == first.id
    assert repository.resolve_for_session("session_new", "kb_a", "user_a").id == second.id
    old_session = session.get(QuerySessionModel, "session_old")
    assert old_session.retrieval_profile_hash == first.config_hash

    restored = repository.rollback("kb_a", first.profile_id, first.version)
    assert restored.id == first.id
    assert restored.status == "published"
    assert second.status == "retired"

    other_tenant = RetrievalProfileRepository(session, tenant_id="tenant_b")
    with pytest.raises(ResourceNotFoundError):
        other_tenant.get(first.profile_id, first.version)
    with pytest.raises(ResourceNotFoundError):
        other_tenant.resolve_for_kb("kb_a")


def test_publish_rejects_incompatible_active_index(session: Session) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a")
    config = default_retrieval_profile_config()
    config["dense"]["dimension"] = 768
    incompatible = repository.create_draft("kb_a", "wrong dim", config)
    with pytest.raises(ResourceConflictError, match="dimension"):
        repository.publish("kb_a", incompatible.profile_id, incompatible.version)
    assert incompatible.status == "draft"
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id is None


def test_profile_get_requires_kb_scope_when_tenant_identity_is_ambiguous(
    session: Session,
) -> None:
    session.add(
        KnowledgeBaseModel(
            id="kb_a_peer",
            tenant_id="tenant_a",
            workspace_id="workspace_a",
            name="KB a peer",
            status="active",
            created_by="tester",
        )
    )
    session.flush()
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a")
    profile_id = "profile_shared_identity"
    first = repository.create_draft(
        "kb_a",
        "first scoped profile",
        default_retrieval_profile_config(),
        profile_id=profile_id,
    )

    # 租户内仍只有一个匹配时，未带 kb_id 的旧调用保持兼容。
    assert repository.get(profile_id, first.version).id == first.id

    second_config = default_retrieval_profile_config()
    second_config["candidate_k"] = 15
    second = repository.create_draft(
        "kb_a_peer",
        "second scoped profile",
        second_config,
        profile_id=profile_id,
    )
    assert first.version == second.version == 1
    assert repository.get(profile_id, 1, kb_id="kb_a").id == first.id
    assert repository.get(profile_id, 1, kb_id="kb_a_peer").id == second.id

    with pytest.raises(ResourceConflictError, match="kb_id is required"):
        repository.get(profile_id, 1)


def test_control_plane_compare_and_swap_rejects_stale_profile_mutations(
    session: Session,
) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a")
    baseline = repository.create_draft(
        "kb_a",
        "baseline",
        default_retrieval_profile_config(),
        profile_id="profile_cas",
    )
    repository.publish("kb_a", baseline.profile_id, baseline.version)
    candidate_config = copy.deepcopy(baseline.config_json)
    candidate_config["candidate_k"] = 15
    candidate = repository.create_draft(
        "kb_a",
        "candidate",
        candidate_config,
        profile_id=baseline.profile_id,
        expected_current_profile_id=baseline.profile_id,
        expected_current_version=baseline.version,
        expected_current_config_hash=baseline.config_hash,
        expected_max_version=baseline.version,
    )
    assert candidate.version == 2

    with pytest.raises(ResourceConflictError, match="version lineage changed"):
        repository.create_draft(
            "kb_a",
            "stale duplicate",
            candidate_config,
            profile_id=baseline.profile_id,
            expected_current_profile_id=baseline.profile_id,
            expected_current_version=baseline.version,
            expected_current_config_hash=baseline.config_hash,
            expected_max_version=baseline.version,
        )
    assert repository.get(baseline.profile_id, kb_id="kb_a").version == candidate.version

    with pytest.raises(ResourceConflictError, match="current retrieval profile changed"):
        repository.publish(
            "kb_a",
            candidate.profile_id,
            candidate.version,
            expected_current_profile_id=baseline.profile_id,
            expected_current_version=baseline.version,
            expected_current_config_hash="0" * 64,
        )
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == baseline.id

    repository.publish(
        "kb_a",
        candidate.profile_id,
        candidate.version,
        expected_current_profile_id=baseline.profile_id,
        expected_current_version=baseline.version,
        expected_current_config_hash=baseline.config_hash,
    )
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == candidate.id

    with pytest.raises(ResourceConflictError, match="current retrieval profile changed"):
        repository.rollback(
            "kb_a",
            baseline.profile_id,
            baseline.version,
            expected_current_profile_id=baseline.profile_id,
            expected_current_version=baseline.version,
            expected_current_config_hash=baseline.config_hash,
        )

    repository.rollback(
        "kb_a",
        baseline.profile_id,
        baseline.version,
        expected_current_profile_id=candidate.profile_id,
        expected_current_version=candidate.version,
        expected_current_config_hash=candidate.config_hash,
    )
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == baseline.id


def test_two_sessions_reject_stale_revision_and_later_draft_lineage(
    engine: Engine,
) -> None:
    with Session(engine, expire_on_commit=False) as setup:
        repository = RetrievalProfileRepository(setup, tenant_id="tenant_a")
        baseline = repository.create_draft(
            "kb_a",
            "baseline",
            default_retrieval_profile_config(),
            profile_id="profile_serialized",
        )
        repository.publish("kb_a", baseline.profile_id, baseline.version)
        candidate_config = copy.deepcopy(baseline.config_json)
        candidate_config["candidate_k"] = 15
        revision_before_candidate = repository.control_plane_revision("kb_a")
        candidate = repository.create_draft(
            "kb_a",
            "candidate",
            candidate_config,
            profile_id=baseline.profile_id,
            expected_current_profile_id=baseline.profile_id,
            expected_current_version=baseline.version,
            expected_current_config_hash=baseline.config_hash,
            expected_control_plane_revision=revision_before_candidate,
            expected_max_version=baseline.version,
        )
        setup.commit()
        baseline_identity = (baseline.profile_id, baseline.version, baseline.config_hash)
        candidate_identity = (candidate.profile_id, candidate.version)

    session_a = Session(engine, expire_on_commit=False)
    session_b = Session(engine, expire_on_commit=False)
    try:
        repository_a = RetrievalProfileRepository(session_a, tenant_id="tenant_a")
        repository_b = RetrievalProfileRepository(session_b, tenant_id="tenant_a")
        stale_revision = repository_a.control_plane_revision("kb_a")
        # End A's read transaction while retaining the token a concurrent caller
        # would have received from the profile-list response.
        session_a.rollback()

        later_config = copy.deepcopy(candidate_config)
        later_config["candidate_k"] = 20
        later = repository_b.create_draft(
            "kb_a",
            "later draft",
            later_config,
            profile_id=baseline_identity[0],
            expected_current_profile_id=baseline_identity[0],
            expected_current_version=baseline_identity[1],
            expected_current_config_hash=baseline_identity[2],
            expected_control_plane_revision=stale_revision,
            expected_max_version=candidate_identity[1],
        )
        session_b.commit()
        assert later.version == 3

        with pytest.raises(ResourceConflictError, match="control-plane revision changed"):
            repository_a.publish(
                "kb_a",
                candidate_identity[0],
                candidate_identity[1],
                expected_current_profile_id=baseline_identity[0],
                expected_current_version=baseline_identity[1],
                expected_current_config_hash=baseline_identity[2],
                expected_control_plane_revision=stale_revision,
                expected_max_version=candidate_identity[1],
            )
        session_a.rollback()

        current_revision = repository_a.control_plane_revision("kb_a")
        with pytest.raises(ResourceConflictError, match="version lineage changed"):
            repository_a.publish(
                "kb_a",
                candidate_identity[0],
                candidate_identity[1],
                expected_current_profile_id=baseline_identity[0],
                expected_current_version=baseline_identity[1],
                expected_current_config_hash=baseline_identity[2],
                expected_control_plane_revision=current_revision,
                expected_max_version=candidate_identity[1],
            )
        session_a.rollback()

        with pytest.raises(ResourceConflictError, match="version lineage changed"):
            repository_a.rollback(
                "kb_a",
                baseline_identity[0],
                baseline_identity[1],
                expected_current_profile_id=baseline_identity[0],
                expected_current_version=baseline_identity[1],
                expected_current_config_hash=baseline_identity[2],
                expected_control_plane_revision=current_revision,
                expected_max_version=candidate_identity[1],
            )
        session_a.rollback()
    finally:
        session_a.close()
        session_b.close()

    with Session(engine) as verify:
        repository = RetrievalProfileRepository(verify, tenant_id="tenant_a")
        current = repository.resolve_for_kb("kb_a", allow_default=False)
        assert (current.profile_id, current.version) == baseline_identity[:2]
        assert repository.get(candidate_identity[0], candidate_identity[1]).status == "draft"
        assert repository.get(later.profile_id, later.version).status == "draft"


def test_legacy_profile_clients_remain_supported_and_advance_kb_revision(
    session: Session,
) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a")
    initial_revision = repository.control_plane_revision("kb_a")
    baseline = repository.create_draft(
        "kb_a",
        "legacy baseline",
        default_retrieval_profile_config(),
        profile_id="profile_legacy",
    )
    assert repository.control_plane_revision("kb_a") == initial_revision + 1
    repository.publish("kb_a", baseline.profile_id, baseline.version)
    assert repository.control_plane_revision("kb_a") == initial_revision + 2

    candidate_config = copy.deepcopy(baseline.config_json)
    candidate_config["candidate_k"] = 9
    candidate = repository.create_draft(
        "kb_a",
        "legacy candidate",
        candidate_config,
        profile_id=baseline.profile_id,
    )
    assert repository.control_plane_revision("kb_a") == initial_revision + 3
    repository.publish("kb_a", candidate.profile_id, candidate.version)
    assert repository.control_plane_revision("kb_a") == initial_revision + 4
    repository.rollback("kb_a", baseline.profile_id, baseline.version)
    assert repository.control_plane_revision("kb_a") == initial_revision + 5
    assert repository.resolve_for_kb("kb_a", allow_default=False).id == baseline.id


def test_evaluation_can_pin_authorized_draft_without_changing_published_pointer(
    session: Session,
) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a", actor_id="admin_a")
    baseline = repository.create_draft(
        "kb_a",
        "baseline",
        default_retrieval_profile_config(),
        profile_id="profile_ab",
    )
    repository.publish("kb_a", baseline.profile_id, baseline.version)
    candidate_config = copy.deepcopy(baseline.config_json)
    candidate_config["fusion"]["subquery_coverage_bonus_weight"] = 0.2
    candidate = repository.create_draft(
        "kb_a",
        "candidate",
        candidate_config,
        profile_id=baseline.profile_id,
    )
    assert baseline.config_json["fusion"]["subquery_coverage_bonus_weight"] == 0.0
    assert candidate.config_hash != baseline.config_hash
    published_pointer = session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id

    resolved = repository.resolve_for_evaluation(
        "kb_a",
        profile_id=candidate.profile_id,
        version=candidate.version,
    )

    assert resolved.id == candidate.id
    assert resolved.status == "draft"
    assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == published_pointer
    old_session = repository.resolve_for_session("evaluation-old-session", "kb_a", "user-a")
    assert old_session.id == baseline.id
    assert (
        session.get(QuerySessionModel, "evaluation-old-session").retrieval_profile_hash
        == baseline.config_hash
    )
    other_tenant = RetrievalProfileRepository(session, tenant_id="tenant_b")
    with pytest.raises(ResourceNotFoundError):
        other_tenant.resolve_for_evaluation(
            "kb_a",
            profile_id=candidate.profile_id,
            version=candidate.version,
        )


def test_metadata_catalog_replace_is_idempotent_and_tenant_scoped(
    session: Session,
) -> None:
    repository = RetrievalProfileRepository(session, tenant_id="tenant_a")
    metadata = {
        "department": ["safety", "factory", "safety"],
        "level": 3,
        "effective": True,
        "_source": {"page": 1},
        "empty": None,
    }
    created = repository.replace_metadata_entries(
        "kb_a",
        "doc_a",
        "version_a",
        "index_a",
        metadata,
    )
    assert created == 4
    assert repository.load_metadata_catalog("kb_a") == {
        "department": ["factory", "safety"],
        "effective": [True],
        "level": [3],
    }

    repeated = repository.replace_metadata_entries(
        "kb_a",
        "doc_a",
        "version_a",
        "index_a",
        metadata,
    )
    assert repeated == created
    assert session.scalar(select(func.count(MetadataCatalogEntryModel.id))) == created

    other_tenant = RetrievalProfileRepository(session, tenant_id="tenant_b")
    with pytest.raises(ResourceNotFoundError):
        other_tenant.load_metadata_catalog("kb_a")


def test_builtin_timeout_migration_versions_profile_without_mutating_v1(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "builtin-timeout.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    _upgrade(database_url, "20260725_0012")
    engine = create_engine(database_url, future=True)
    try:
        _seed_legacy_0012_builtin_timeout(engine)
    finally:
        engine.dispose()

    _upgrade(database_url)
    migrated_engine = create_engine(database_url, future=True)
    try:
        with Session(migrated_engine) as session:
            rows = session.scalars(
                select(RetrievalProfileModel)
                .where(
                    RetrievalProfileModel.kb_id == "kb_a",
                    RetrievalProfileModel.profile_id == "rpf_default",
                )
                .order_by(RetrievalProfileModel.version)
            ).all()
            assert [(row.version, row.status) for row in rows] == [
                (1, "retired"),
                (2, "published"),
            ]
            assert rows[0].config_json["timeout_ms"] == 30_000
            assert rows[1].config_json["timeout_ms"] == 90_000
            assert rows[1].created_by == "system-timeout-v13"
            assert session.get(KnowledgeBaseModel, "kb_a").retrieval_config_id == rows[1].id
    finally:
        migrated_engine.dispose()
