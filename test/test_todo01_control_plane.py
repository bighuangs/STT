from hashlib import sha256
from pathlib import Path

import pytest

from app.application import control_plane_service as service
from app.domain._state_machine import InvalidStateTransition
from app.domain.document import ensure_document_transition
from app.infra.persistence.sql.database import configure_database
from app.infra.persistence.sql.repositories import ResourceConflictError, ResourceNotFoundError


@pytest.fixture()
def control_db(tmp_path: Path):
    db_path = tmp_path / "control.db"
    configure_database(f"sqlite:///{db_path.as_posix()}")
    service.bootstrap_control_plane()
    return db_path


def test_default_resources_and_unique_kb(control_db: Path):
    rows = service.list_knowledge_bases()
    assert len(rows) == 1
    assert rows[0]["id"] == "kb_default"
    assert rows[0]["active_index_version_id"].startswith("idx_")

    created = service.create_knowledge_base("产品手册", "网页验证知识库")
    assert created["status"] == "active"
    assert created["document_count"] == 0
    assert len(service.list_index_versions(created["id"])) == 1

    with pytest.raises(ResourceConflictError):
        service.create_knowledge_base("产品手册", "重复名称")


def test_document_version_job_binding_and_dedup(control_db: Path, tmp_path: Path):
    kb = service.create_knowledge_base("版本测试库")
    source = tmp_path / "manual.md"
    content_v1 = b"# Manual\nversion one"
    source.write_bytes(content_v1)

    first = service.register_uploaded_document(
        kb_id=kb["id"],
        logical_name="manual.md",
        content_sha256=sha256(content_v1).hexdigest(),
        size_bytes=len(content_v1),
        mime_type="text/markdown",
        object_key=str(source),
        parser_config_snapshot={"chunk_token_num": 256},
    )
    assert first["document"]["id"] != first["ingestion_job"]["id"]
    assert first["document_version"]["version_no"] == 1
    assert first["should_process"] is True

    job_id = first["ingestion_job"]["id"]
    service.start_ingestion_job(job_id)
    service.update_ingestion_job_stage(job_id, "node_chunk_merge", 0.4)
    service.update_ingestion_job_stage(job_id, "node_bge_embedding", 0.7)
    service.update_ingestion_job_stage(job_id, "node_import_milvus", 0.95)
    completed = service.complete_ingestion_job(job_id, chunk_count=7, token_count=123)
    assert completed["status"] == "completed"
    with pytest.raises(ResourceConflictError, match="immutable"):
        service.create_ingestion_job(first["document_version"]["id"])

    detail = service.get_document(first["document"]["id"])
    assert detail["status"] == "ready"
    assert detail["active_version_id"] == first["document_version"]["id"]
    assert detail["chunk_count"] == 7
    assert detail["active_index_version_id"] == first["index_version"]["id"]

    duplicate = service.register_uploaded_document(
        kb_id=kb["id"],
        logical_name="manual.md",
        content_sha256=sha256(content_v1).hexdigest(),
        size_bytes=len(content_v1),
        mime_type="text/markdown",
        object_key=str(source),
    )
    assert duplicate["deduplicated"] is True
    assert duplicate["document_version"]["id"] == first["document_version"]["id"]
    assert duplicate["ingestion_job"]["id"] == job_id
    assert duplicate["should_process"] is False

    content_v2 = b"# Manual\nversion two"
    source_v2 = tmp_path / "manual-v2.md"
    source_v2.write_bytes(content_v2)
    second = service.register_uploaded_document(
        kb_id=kb["id"],
        document_id=first["document"]["id"],
        logical_name="manual-v2.md",
        content_sha256=sha256(content_v2).hexdigest(),
        size_bytes=len(content_v2),
        mime_type="text/markdown",
        object_key=str(source_v2),
    )
    assert second["document"]["id"] == first["document"]["id"]
    assert second["document_version"]["version_no"] == 2
    assert second["document_version"]["id"] != first["document_version"]["id"]


def test_illegal_state_transition_is_rejected(control_db: Path):
    with pytest.raises(InvalidStateTransition):
        ensure_document_transition("draft", "ready")


def test_active_job_conflict_and_failed_job_retry(control_db: Path, tmp_path: Path):
    source = tmp_path / "retry.md"
    source.write_text("retry", encoding="utf-8")
    registered = service.register_uploaded_document(
        kb_id="kb_default",
        logical_name=source.name,
        content_sha256=sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        mime_type="text/markdown",
        object_key=str(source),
    )
    version_id = registered["document_version"]["id"]
    with pytest.raises(ResourceConflictError, match="active ingestion job"):
        service.create_ingestion_job(version_id)

    failed = service.fail_ingestion_job(
        registered["ingestion_job"]["id"], "TEST_FAILURE", "expected"
    )
    assert failed["status"] == "failed"
    assert service.get_document(registered["document"]["id"])["status"] == "failed"
    retry = service.create_ingestion_job(version_id)
    assert retry["attempt"] == 2
    assert retry["status"] == "queued"


def test_milvus_mapping_uses_stable_domain_ids():
    from app.rag.import_.index_service import _build_milvus_chunks

    rows = _build_milvus_chunks(
        {
            "task_id": "job_123",
            "job_id": "job_123",
            "document_id": "doc_123",
            "document_version_id": "dv_123",
            "index_version_id": "idx_123",
            "tenant_id": "tenant_default",
            "kb_id": "kb_123",
            "file_title": "sample.md",
            "chunks": [{"content": "hello", "sparse_vector": {1: 0.2}}],
            "toc": {},
        }
    )
    assert rows[0]["doc_id"] == "doc_123"
    assert rows[0]["doc_id"] != "job_123"
    assert rows[0]["document_version_id"] == "dv_123"
    assert rows[0]["index_version_id"] == "idx_123"
    assert rows[0]["tenant_id"] == "tenant_default"


def test_legacy_migration_can_report_quarantine_and_rollback(control_db: Path):
    from scripts.migrate_legacy_control_plane import build_legacy_groups

    documents, quarantine = build_legacy_groups(
        [
            {"chunk_id": 1, "doc_id": "legacy_doc", "kb_id": "legacy_kb", "docnm_kwd": "旧文档"},
            {"chunk_id": 2, "doc_id": "legacy_doc", "kb_id": "legacy_kb", "docnm_kwd": "旧文档"},
            {"chunk_id": 3, "doc_id": "", "kb_id": "legacy_kb", "docnm_kwd": "孤儿"},
        ]
    )
    assert documents[0]["chunk_count"] == 2
    assert quarantine[0]["reason"] == "missing_doc_id_or_kb_id"

    result = service.import_legacy_document(
        **documents[0], collection_name="kb_chunks", batch_id="batch_test"
    )
    assert result["action"] == "created"
    assert service.get_document("legacy_doc")["chunk_count"] == 2
    assert service.rollback_legacy_batch("batch_test") == 1
    with pytest.raises(ResourceNotFoundError):
        service.get_document("legacy_doc")
