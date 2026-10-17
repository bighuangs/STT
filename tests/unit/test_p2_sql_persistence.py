"""P2 控制面 SQL 持久化的隔离、不可变性与生命周期合同测试。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session

from app.domain.connectors import (
    ConnectorScope,
    SourceBindingStatus,
    SourceItemBinding,
)
from app.domain.knowledge_governance import MetadataTarget
from app.domain.memory import (
    LongTermMemory,
    MemoryConsent,
    MemoryNamespace,
    MemoryProvenance,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    MemoryType,
)
from app.domain.multimodal import MediaAsset, MediaDerivation, MediaLocation
from app.infra.persistence.sql.models import (
    DocumentModel,
    DocumentVersionModel,
    EvidenceRecordModel,
    IndexVersionModel,
    KnowledgeBaseModel,
    LongTermMemoryModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.p2_repository import (
    CapabilityResourceRepository,
    CapabilityRunRepository,
    ConnectorStateRepository,
    LongTermMemoryRepository,
    MediaAssetRepository,
    MetadataValueRepository,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.security.permissions import is_allowed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def _alembic(database_url: str, target: str, *, downgrade: bool = False) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    operation = command.downgrade if downgrade else command.upgrade
    operation(config, target)


def _seed(session: Session) -> None:
    for tenant_id in ("tenant_a", "tenant_b"):
        suffix = tenant_id[-1]
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
    for kb_id, tenant_id, workspace_id in (
        ("kb_a", "tenant_a", "workspace_a"),
        ("kb_a2", "tenant_a", "workspace_a"),
        ("kb_b", "tenant_b", "workspace_b"),
    ):
        session.add(
            KnowledgeBaseModel(
                id=kb_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                name=kb_id,
                status="active",
                created_by="tester",
            )
        )
    session.flush()
    for suffix, kb_id in (("a", "kb_a"), ("a2", "kb_a2"), ("b", "kb_b")):
        document_id = f"doc_{suffix}"
        session.add(
            DocumentModel(
                id=document_id,
                kb_id=kb_id,
                logical_name=f"manual-{suffix}.pdf",
                source_type="upload",
                status="ready",
            )
        )
        session.flush()
        session.add(
            DocumentVersionModel(
                id=f"dv_{suffix}",
                document_id=document_id,
                version_no=1,
                content_sha256=(suffix[0] * 64),
                pipeline_fingerprint=("f" * 64),
                size_bytes=1024,
                mime_type="application/pdf",
                object_key=f"documents/{suffix}.pdf",
                parser_config_snapshot={},
                status="published",
                published_at=NOW,
            )
        )
    session.commit()


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    path = tmp_path_factory.mktemp("p2-sql") / "control.db"
    database_url = f"sqlite:///{path.as_posix()}"
    _alembic(database_url, "head")
    result = create_engine(database_url, future=True)
    event.listen(
        result,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    with Session(result) as session:
        _seed(session)
    yield result
    result.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    connection = engine.connect()
    transaction = connection.begin()
    value = Session(bind=connection)
    try:
        yield value
    finally:
        value.close()
        transaction.rollback()
        connection.close()


def _resource(
    session: Session,
    *,
    tenant_id: str = "tenant_a",
    kb_id: str | None = "kb_a",
    logical_id: str = "route-default",
    version: str = "1",
    resource_type: str = "advanced_index",
    schema_version: str = "p2.v1",
    payload: dict | None = None,
) -> tuple[CapabilityResourceRepository, dict]:
    repository = CapabilityResourceRepository(
        session,
        tenant_id=tenant_id,
        actor_id=f"admin_{tenant_id[-1]}",
    )
    created = repository.create(
        resource_type=resource_type,
        logical_id=logical_id,
        version=version,
        schema_version=schema_version,
        payload=payload or {"enabled": False},
        kb_id=kb_id,
    )
    return repository, created


def _published_metadata_schema(
    session: Session,
    *,
    logical_id: str,
    field: dict,
) -> dict:
    payload = {
        "schema_version": "metadata-schema.v1",
        "schema_id": logical_id,
        "tenant_id": "tenant_a",
        "kb_id": "kb_a",
        "version": 1,
        "revision": 1,
        "status": "published",
        "fields": [field],
        "created_by": "reviewer_a",
        "created_at": NOW.isoformat(),
        "published_at": NOW.isoformat(),
    }
    resources, schema = _resource(
        session,
        logical_id=logical_id,
        resource_type="metadata_schema",
        schema_version="metadata-schema.v1",
        payload=payload,
    )
    resources.publish(schema["id"])
    return schema


def _add_chunk_evidence(
    session: Session,
    *,
    chunk_id: str = "chunk_safety",
) -> EvidenceRecordModel:
    session.add(
        IndexVersionModel(
            id="idx_a_metadata",
            kb_id="kb_a",
            version_no=1,
            embedding_model_id="embedding-test",
            vector_dim=8,
            schema_version="milvus-test.v1",
            status="active",
            active_slot="active",
            collection_alias="kb_chunks",
            activated_at=NOW,
            build_kind="initial",
            validation_status="validated",
            manifest_summary={},
        )
    )
    session.flush()
    evidence = EvidenceRecordModel(
        id="ev_" + "a" * 40,
        schema_version="evidence.v1",
        source_type="local",
        tenant_id="tenant_a",
        kb_id="kb_a",
        document_id="doc_a",
        document_version_id="dv_a",
        index_version_id="idx_a_metadata",
        chunk_id=chunk_id,
        content="HAK180 使用前必须检查保护接地。",
        content_hash="sha256:" + "a" * 64,
        source_block_ids_json=["block_safety"],
        provenance_spans_json=[
            {
                "block_id": "block_safety",
                "availability": "unavailable",
                "coordinate_space": "none",
                "reason_code": "test_fixture",
            }
        ],
        heading_path_json=["使用前准备"],
        asset_refs_json=[],
        source_label="hak180产品安全手册.pdf",
        primary_block_id="block_safety",
        primary_availability="unavailable",
        primary_coordinate_space="none",
        primary_reason_code="test_fixture",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _memory(
    *,
    memory_id: str = "mem_a",
    candidate_id: str = "mcand_a",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    kb_id: str = "kb_a",
    content: str = "请使用中文回答",
) -> LongTermMemory:
    return LongTermMemory(
        memory_id=memory_id,
        candidate_id=candidate_id,
        namespace=MemoryNamespace(
            tenant_id="tenant_a",
            subject_user_id="user_a",
            kb_id=kb_id,
        ),
        type=MemoryType.PREFERENCE,
        content=content,
        structured_value={"key": "language", "value": "zh-CN"} if content else {},
        conflict_key="preference:language",
        provenance=MemoryProvenance(
            source_kind=MemorySourceKind.USER_MESSAGE,
            session_id="session_a",
            source_message_ids=("message_a",),
            extractor_version="memory-rules/1.0.0",
            source_excerpt_hash="a" * 64,
        ),
        confidence=0.98,
        sensitivity=MemorySensitivity.PERSONAL,
        consent=(
            MemoryConsent.CONFIRMED
            if status == MemoryStatus.ACTIVE
            else MemoryConsent.EXPLICIT_REQUEST
        ),
        status=status,
        valid_from=NOW,
        expires_at=NOW + timedelta(days=30),
        created_by="user_a",
        embedding_version=("memory-lexical/1.0.0" if status == MemoryStatus.ACTIVE else None),
        created_at=NOW,
        updated_at=NOW,
    )


def _document_image(*, tenant_id: str = "tenant_a") -> MediaAsset:
    suffix = "a" if tenant_id == "tenant_a" else "b"
    return MediaAsset(
        tenant_id=tenant_id,
        kb_id=f"kb_{suffix}",
        document_id=f"doc_{suffix}",
        document_version_id=f"dv_{suffix}",
        source_kind="document_asset",
        modality="image",
        media_type="image/png",
        content_hash=f"sha256:{suffix * 64}",
        size_bytes=2048,
        width=800,
        height=600,
        location=MediaLocation(page_no=1, bbox=(0.1, 0.2, 0.8, 0.9)),
        classification="internal",
        object_key=f"media/{suffix}/diagram.png",
        created_at=NOW,
    )


def test_capability_resources_are_tenant_scoped_and_versions_are_immutable(
    session: Session,
) -> None:
    repository, first = _resource(session, payload={"route": "toc"})

    replay = repository.create(
        resource_type="advanced_index",
        logical_id="route-default",
        version="1",
        schema_version="p2.v1",
        payload={"route": "toc"},
        kb_id="kb_a",
    )
    assert replay["id"] == first["id"]
    with pytest.raises(ResourceConflictError, match="another payload"):
        repository.create(
            resource_type="advanced_index",
            logical_id="route-default",
            version="1",
            schema_version="p2.v1",
            payload={"route": "graph"},
            kb_id="kb_a",
        )

    other_kb = repository.create(
        resource_type="advanced_index",
        logical_id="route-default",
        version="1",
        schema_version="p2.v1",
        payload={"route": "graph"},
        kb_id="kb_a2",
    )
    assert other_kb["id"] != first["id"]

    published = repository.publish(first["id"], enable=True)
    assert published["status"] == "published"
    assert published["enabled"] is True
    _, second = _resource(
        session,
        logical_id="route-default",
        version="2",
        payload={"route": "toc", "depth": 3},
    )
    repository.publish(second["id"], enable=True)
    assert repository.as_dict(repository.get(first["id"]))["status"] == "retired"

    other_tenant = CapabilityResourceRepository(
        session,
        tenant_id="tenant_b",
        actor_id="admin_b",
    )
    with pytest.raises(ResourceNotFoundError):
        other_tenant.get(first["id"])


def test_capability_run_idempotency_is_bound_to_input_and_resource(
    session: Session,
) -> None:
    resources, resource = _resource(
        session,
        logical_id="agent-flow",
        resource_type="agent_workflow",
    )
    resources.publish(resource["id"])
    runs = CapabilityRunRepository(session, tenant_id="tenant_a")
    first = runs.create(
        run_type="agent",
        trace_id="trace-a",
        input_payload={"question": "HAK180 如何使用？"},
        input_summary={"question_length": 12},
        resource_id=resource["id"],
        idempotency_key="idem-a",
    )
    replay = runs.create(
        run_type="agent",
        trace_id="trace-retry",
        input_payload={"question": "HAK180 如何使用？"},
        input_summary={"question_length": 12},
        resource_id=resource["id"],
        idempotency_key="idem-a",
    )
    assert replay["id"] == first["id"]
    with pytest.raises(ResourceConflictError, match="another input"):
        runs.create(
            run_type="agent",
            trace_id="trace-conflict",
            input_payload={"question": "另一问题"},
            input_summary={},
            resource_id=resource["id"],
            idempotency_key="idem-a",
        )


def test_published_manual_metadata_cannot_be_overwritten_by_auto(
    session: Session,
) -> None:
    schema_payload = {
        "schema_version": "metadata-schema.v1",
        "schema_id": "manual-metadata",
        "tenant_id": "tenant_a",
        "kb_id": "kb_a",
        "version": 1,
        "revision": 3,
        "status": "published",
        "fields": [
            {
                "key": "product",
                "value_type": "string",
                "description": "产品型号",
                "scope": "document",
                "required": True,
                "enum_values": ["HAK180", "HAK190"],
            },
            {
                "key": "edition",
                "value_type": "integer",
                "description": "文档版次",
                "scope": "document_version",
            },
        ],
        "created_by": "reviewer_a",
        "created_at": NOW.isoformat(),
        "published_at": NOW.isoformat(),
    }
    resources, schema = _resource(
        session,
        logical_id="manual-metadata",
        resource_type="metadata_schema",
        schema_version="metadata-schema.v1",
        payload=schema_payload,
    )
    resources.publish(schema["id"])
    target = MetadataTarget(
        tenant_id="tenant_a",
        kb_id="kb_a",
        scope="document",
        document_id="doc_a",
    )
    manual_repository = MetadataValueRepository(
        session,
        tenant_id="tenant_a",
        actor_id="reviewer_a",
    )
    manual = manual_repository.propose(
        schema_resource_id=schema["id"],
        target=target,
        field_key="product",
        value=" HAK180 ",
        source="manual",
        confidence=None,
        provenance={"source": "operator"},
    )
    assert manual["value"] == "HAK180"
    manual_repository.publish(manual["id"], review_reason="verified from manual")

    automatic_repository = MetadataValueRepository(
        session,
        tenant_id="tenant_a",
        actor_id="extractor",
    )
    automatic = automatic_repository.propose(
        schema_resource_id=schema["id"],
        target=target,
        field_key="product",
        value="HAK190",
        source="auto",
        confidence=0.91,
        provenance={"source": "model"},
        enhancement={
            "run_id": "run-a",
            "prompt_id": "metadata",
            "prompt_version": "1",
            "model_id": "local-model",
        },
    )
    with pytest.raises(ResourceConflictError, match="cannot overwrite"):
        automatic_repository.publish(automatic["id"])
    assert manual_repository._get(manual["id"]).status == "published"
    assert automatic_repository._get(automatic["id"]).status == "proposed"

    with pytest.raises(ResourceConflictError, match="field was not found"):
        manual_repository.propose(
            schema_resource_id=schema["id"],
            target=target,
            field_key="unknown",
            value="value",
            source="manual",
            confidence=None,
        )
    with pytest.raises(ResourceConflictError, match="outside the published enum"):
        manual_repository.propose(
            schema_resource_id=schema["id"],
            target=target,
            field_key="product",
            value="HAK999",
            source="manual",
            confidence=None,
        )
    version_target = MetadataTarget(
        tenant_id="tenant_a",
        kb_id="kb_a",
        scope="document_version",
        document_id="doc_a",
        document_version_id="dv_a",
    )
    with pytest.raises(ResourceConflictError, match="scope does not match"):
        manual_repository.propose(
            schema_resource_id=schema["id"],
            target=version_target,
            field_key="product",
            value="HAK180",
            source="manual",
            confidence=None,
        )
    with pytest.raises(ResourceConflictError, match="integer metadata requires"):
        manual_repository.propose(
            schema_resource_id=schema["id"],
            target=version_target,
            field_key="edition",
            value="1",
            source="manual",
            confidence=None,
        )
    with pytest.raises(ResourceConflictError, match="requires confidence"):
        automatic_repository.propose(
            schema_resource_id=schema["id"],
            target=target,
            field_key="product",
            value="HAK190",
            source="auto",
            confidence=None,
        )


def test_compliance_metadata_publish_requires_explicit_authorization_signal(
    session: Session,
) -> None:
    schema = _published_metadata_schema(
        session,
        logical_id="compliance-metadata",
        field={
            "key": "risk_level",
            "value_type": "integer",
            "description": "风险等级",
            "scope": "document",
            "compliance_controlled": True,
        },
    )
    repository = MetadataValueRepository(
        session,
        tenant_id="tenant_a",
        actor_id="reviewer_a",
    )
    proposal = repository.propose(
        schema_resource_id=schema["id"],
        target=MetadataTarget(
            tenant_id="tenant_a",
            kb_id="kb_a",
            scope="document",
            document_id="doc_a",
        ),
        field_key="risk_level",
        value=3,
        source="manual",
        confidence=None,
    )

    compliance_action = "metadata:compliance"
    assert is_allowed(("KBAdmin",), (), compliance_action) is False
    assert is_allowed(("TenantAdmin",), (), compliance_action) is True
    assert (
        is_allowed(
            ("ServiceAccount",),
            (compliance_action,),
            compliance_action,
        )
        is True
    )
    assert repository.requires_compliance_permission(proposal["id"]) is True
    with pytest.raises(
        ResourceConflictError,
        match="COMPLIANCE_PERMISSION_REQUIRED",
    ):
        repository.publish(proposal["id"])
    assert repository._get(proposal["id"]).status == "proposed"

    published = repository.publish(
        proposal["id"],
        allow_compliance_controlled=True,
    )
    assert published["status"] == "published"
    assert published["reviewed_by"] == "reviewer_a"


def test_chunk_metadata_requires_live_evidence_in_exact_scope_at_propose_and_publish(
    session: Session,
) -> None:
    schema = _published_metadata_schema(
        session,
        logical_id="chunk-metadata",
        field={
            "key": "safety_topic",
            "value_type": "string",
            "description": "安全主题",
            "scope": "chunk",
        },
    )
    repository = MetadataValueRepository(
        session,
        tenant_id="tenant_a",
        actor_id="reviewer_a",
    )
    target = MetadataTarget(
        tenant_id="tenant_a",
        kb_id="kb_a",
        scope="chunk",
        document_id="doc_a",
        document_version_id="dv_a",
        chunk_id="chunk_safety",
    )

    with pytest.raises(
        ResourceNotFoundError,
        match="chunk evidence not found",
    ):
        repository.propose(
            schema_resource_id=schema["id"],
            target=target,
            field_key="safety_topic",
            value="保护接地",
            source="manual",
            confidence=None,
        )

    evidence = _add_chunk_evidence(session)
    proposal = repository.propose(
        schema_resource_id=schema["id"],
        target=target,
        field_key="safety_topic",
        value="保护接地",
        source="manual",
        confidence=None,
    )
    assert proposal["status"] == "proposed"

    evidence.chunk_id = "chunk_moved_after_proposal"
    session.flush()
    with pytest.raises(
        ResourceNotFoundError,
        match="chunk evidence not found",
    ):
        repository.publish(proposal["id"])
    assert repository._get(proposal["id"]).status == "proposed"

    evidence.chunk_id = "chunk_safety"
    session.flush()
    assert repository.publish(proposal["id"])["status"] == "published"


def test_connector_binding_scope_and_cursor_optimistic_lock(session: Session) -> None:
    repository = ConnectorStateRepository(
        session,
        tenant_id="tenant_a",
        kb_id="kb_a",
    )
    binding = SourceItemBinding(
        scope=ConnectorScope(tenant_id="tenant_a", kb_id="kb_a"),
        data_source_id="source-manuals",
        source_item_id="hak180",
        document_id="doc_a",
        active_document_version_id="dv_a",
        remote_version="etag-1",
        content_sha256="a" * 64,
        logical_name="hak180产品安全手册.pdf",
        locator="/manuals/hak180.pdf",
        status=SourceBindingStatus.ACTIVE,
        last_seen_at=NOW,
    )
    saved = repository.save_binding(binding)
    assert saved["active_document_version_id"] == "dv_a"
    assert saved["status"] == "active"
    assert (
        ConnectorStateRepository(
            session,
            tenant_id="tenant_a",
            kb_id="kb_a2",
        ).get_binding("source-manuals", "hak180")
        is None
    )

    first = repository.acknowledge_cursor(
        data_source_id="source-manuals",
        value="cursor-1",
        expected_generation=0,
        committed_at=NOW,
    )
    assert first["generation"] == 1
    with pytest.raises(ResourceConflictError, match="generation mismatch"):
        repository.acknowledge_cursor(
            data_source_id="source-manuals",
            value="stale-write",
            expected_generation=0,
        )
    second = repository.acknowledge_cursor(
        data_source_id="source-manuals",
        value="cursor-2",
        expected_generation=1,
    )
    assert second["generation"] == 2
    assert second["value"] == "cursor-2"


def test_memory_is_scoped_and_forget_physically_redacts_user_content(
    session: Session,
) -> None:
    repository = LongTermMemoryRepository(
        session,
        tenant_id="tenant_a",
        subject_user_id="user_a",
        kb_id="kb_a",
    )
    memory = _memory()
    saved = repository.save(memory)
    original_hash = saved["content_hash"]
    assert [item["memory_id"] for item in repository.list_recallable(now=NOW)] == [memory.memory_id]

    with pytest.raises(ResourceNotFoundError):
        LongTermMemoryRepository(
            session,
            tenant_id="tenant_a",
            subject_user_id="user_b",
            kb_id="kb_a",
        ).get(memory.memory_id)
    with pytest.raises(ResourceNotFoundError):
        LongTermMemoryRepository(
            session,
            tenant_id="tenant_a",
            subject_user_id="user_a",
            kb_id="kb_a2",
        ).get(memory.memory_id)

    forgotten = repository.forget(memory.memory_id, at=NOW + timedelta(hours=1))
    assert forgotten["status"] == "forgotten"
    assert forgotten["consent"] == "withdrawn"
    assert forgotten["content"] == ""
    assert forgotten["structured_value"] == {}
    assert forgotten["provenance"] == {}
    assert forgotten["embedding_version"] is None
    assert forgotten["content_hash"] == original_hash
    assert repository.list_recallable(now=NOW) == []

    persisted = session.get(LongTermMemoryModel, memory.memory_id)
    assert persisted is not None
    assert persisted.content == ""
    assert persisted.structured_value == {}
    assert persisted.provenance_json == {}


@pytest.mark.parametrize("status", [MemoryStatus.EXPIRED, MemoryStatus.REJECTED])
def test_memory_terminal_statuses_are_persistable(
    session: Session,
    status: MemoryStatus,
) -> None:
    repository = LongTermMemoryRepository(
        session,
        tenant_id="tenant_a",
        subject_user_id="user_a",
        kb_id="kb_a",
    )
    memory = _memory(
        memory_id=f"mem_{status.value}",
        candidate_id=f"mcand_{status.value}",
        status=status,
        content="已失效的候选记忆",
    )
    assert repository.save(memory)["status"] == status.value
    assert repository.list_recallable(now=NOW) == []


def test_media_assets_enforce_scope_and_keep_auditable_derivation_chain(
    session: Session,
) -> None:
    repository = MediaAssetRepository(
        session,
        tenant_id="tenant_a",
        kb_id="kb_a",
    )
    asset = _document_image()
    saved = repository.save(asset)
    assert saved["asset"]["document_version_id"] == "dv_a"
    assert saved["derivations"] == []

    derivation = MediaDerivation(
        asset_id=asset.asset_id,
        kind="vision_description",
        text="图中显示 HAK180 的安全抬升姿势。",
        provider_id="local",
        model_id="vision-test",
        prompt_version="vision/1",
        model_snapshot_hash=f"sha256:{'f' * 64}",
        confidence=0.95,
        created_at=NOW,
    )
    with_derivation = repository.append_derivation(asset.asset_id, derivation)
    assert [item["derivation_id"] for item in with_derivation["derivations"]] == [
        derivation.derivation_id
    ]
    replay = repository.append_derivation(asset.asset_id, derivation)
    assert len(replay["derivations"]) == 1

    with pytest.raises(ResourceNotFoundError):
        MediaAssetRepository(
            session,
            tenant_id="tenant_b",
            kb_id="kb_b",
        ).retire(asset.asset_id)
    with pytest.raises(ResourceNotFoundError):
        MediaAssetRepository(
            session,
            tenant_id="tenant_a",
            kb_id="kb_a2",
        ).retire(asset.asset_id)

    query_attachment = MediaAsset(
        tenant_id="tenant_a",
        owner_user_id="user_a",
        source_kind="query_attachment",
        modality="image",
        media_type="image/png",
        content_hash=f"sha256:{'c' * 64}",
        size_bytes=512,
        width=320,
        height=240,
        classification="internal",
        object_key="query/user-a/image.png",
        created_at=NOW,
    )
    owner_repository = MediaAssetRepository(
        session,
        tenant_id="tenant_a",
        subject_user_id="user_a",
    )
    owner_repository.save(query_attachment)
    with pytest.raises(ResourceNotFoundError):
        MediaAssetRepository(
            session,
            tenant_id="tenant_a",
            subject_user_id="user_b",
        ).retire(query_attachment.asset_id)
    with pytest.raises(ResourceNotFoundError):
        repository.retire(query_attachment.asset_id)

    repository.retire(asset.asset_id)
    with pytest.raises(ResourceConflictError, match="active media"):
        repository.append_derivation(asset.asset_id, derivation)

    document = session.get(DocumentModel, "doc_a")
    assert document is not None
    document.deleted_at = NOW
    session.flush()
    with pytest.raises(ResourceNotFoundError, match="media document not found"):
        repository.save(_document_image())
    document.deleted_at = None
    session.flush()


def test_migration_can_round_trip_upgrade_and_downgrade(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    _alembic(database_url, "head")
    engine = create_engine(database_url, future=True)
    try:
        assert {
            "capability_resources",
            "metadata_value_records",
            "connector_bindings",
            "connector_cursors",
            "long_term_memories",
            "media_assets",
        }.issubset(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    _alembic(database_url, "20260724_0009", downgrade=True)
    engine = create_engine(database_url, future=True)
    try:
        assert "capability_resources" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()

    _alembic(database_url, "head")
    engine = create_engine(database_url, future=True)
    try:
        assert "long_term_memories" in inspect(engine).get_table_names()
    finally:
        engine.dispose()
