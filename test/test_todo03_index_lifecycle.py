"""TODO-03 staging manifest、原子发布、回滚、查询隔离和清理回归。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select

from app.application import control_plane_service as control
from app.application import ingestion_job_service as jobs
from app.application.index_lifecycle_service import (
    active_index_scopes,
    create_index_rebuild,
    delete_document_with_cleanup,
    delete_knowledge_base_with_cleanup,
    execute_cleanup,
    get_index_lifecycle,
    publish_index,
    rollback_index,
    set_document_enabled,
)
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.models import (
    CleanupJobModel,
    DocumentModel,
    IndexVersionModel,
)
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorPlacementModel,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.import_ import index_service
from app.rag.query.active_scope_filter import build_active_scope_expr


@pytest.fixture()
def todo03_db(tmp_path: Path):
    configure_database(f"sqlite:///{(tmp_path / 'todo03.db').as_posix()}")
    control.bootstrap_control_plane()
    return tmp_path


def _published_document(root: Path, kb_id: str = "kb_default") -> dict:
    source = root / f"published-{kb_id}.md"
    source.write_text("# published\nactive content", encoding="utf-8")
    registered = control.register_uploaded_document(
        kb_id=kb_id,
        logical_name=source.name,
        content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        mime_type="text/markdown",
        object_key=str(source),
    )
    job_id = registered["ingestion_job"]["id"]
    control.start_ingestion_job(job_id)
    control.update_ingestion_job_stage(job_id, "node_import_milvus", 0.95)
    control.complete_ingestion_job(job_id, chunk_count=2)
    return registered


def _finish_cow_job(claimed: dict, target: dict, chunk_count: int = 2) -> dict:
    jobs.mark_job_running(claimed["id"], claimed["lease_token"])
    jobs.update_job_stage(
        claimed["id"],
        claimed["lease_token"],
        "node_import_milvus",
        0.95,
    )
    version = control.get_document_version(claimed["document_version_id"])
    with session_scope() as session:
        placement, family = session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(VectorPlacementModel.index_version_id == target["id"])
        ).one()
    chunk_ids = [f"{claimed['id']}-{offset}" for offset in range(chunk_count)]
    jobs.record_index_manifest(
        claimed["id"],
        claimed["lease_token"],
        {
            "document_version_id": version["id"],
            "index_version_id": target["id"],
            "input_sha256": version["content_sha256"],
            "config_sha256": hashlib.sha256(claimed["id"].encode()).hexdigest(),
            "expected_chunk_count": chunk_count,
            "written_chunk_count": chunk_count,
            "searchable_chunk_count": chunk_count,
            "chunk_ids_sha256": hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest(),
            "chunk_ids": chunk_ids,
            "object_keys": [],
            "vector_dim": target["vector_dim"],
            "collection_name": target["collection_alias"],
            "vector_placement_id": placement.id,
            "vector_placement_epoch": placement.epoch,
            "vector_compatibility_fingerprint": family.compatibility_fingerprint,
            "validation_status": "validated",
            "validation_errors": [],
        },
    )
    return jobs.complete_job(
        claimed["id"],
        claimed["lease_token"],
        chunk_count,
    )


def test_rebuild_stays_invisible_until_atomic_publish_then_can_rollback(todo03_db: Path):
    registered = _published_document(todo03_db)
    old_index_id = registered["index_version"]["id"]
    old_scope = active_index_scopes("kb_default")
    assert [scope["index_version_id"] for scope in old_scope] == [old_index_id]

    rebuild = create_index_rebuild("kb_default")
    target = rebuild["index_version"]
    assert target["status"] == "staging"
    assert target["previous_version_id"] == old_index_id
    assert len(rebuild["jobs"]) == 1

    claimed = jobs.claim_next_job("todo03-worker", 60)
    jobs.mark_job_running(claimed["id"], claimed["lease_token"])
    with session_scope() as session:
        placement, family = session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(VectorPlacementModel.index_version_id == target["id"])
        ).one()
    jobs.record_index_manifest(
        claimed["id"],
        claimed["lease_token"],
        {
            "document_version_id": registered["document_version"]["id"],
            "index_version_id": target["id"],
            "input_sha256": registered["document_version"]["content_sha256"],
            "config_sha256": hashlib.sha256(b"{}").hexdigest(),
            "expected_chunk_count": 2,
            "written_chunk_count": 2,
            "searchable_chunk_count": 2,
            "chunk_ids_sha256": hashlib.sha256(b"a\nb").hexdigest(),
            "chunk_ids": ["a", "b"],
            "object_keys": ["imagetemps-dv/image.jpg"],
            "vector_dim": 1024,
            "collection_name": target["collection_alias"],
            "vector_placement_id": placement.id,
            "vector_placement_epoch": placement.epoch,
            "vector_compatibility_fingerprint": family.compatibility_fingerprint,
            "validation_status": "validated",
            "validation_errors": [],
        },
    )
    jobs.complete_job(claimed["id"], claimed["lease_token"], 2)

    # 即使所有 staging job 已成功，显式 publish 前查询仍只看到旧 index。
    assert active_index_scopes("kb_default")[0]["index_version_id"] == old_index_id
    assert get_index_lifecycle(target["id"])["validation_status"] == "validated"

    published = publish_index(target["id"])
    assert published["status"] == "active"
    assert active_index_scopes("kb_default")[0]["index_version_id"] == target["id"]
    retired_coverage = get_index_lifecycle(old_index_id)["document_coverage"]
    assert retired_coverage["expected_documents"] == 1
    assert retired_coverage["bound_documents"] == 1
    assert retired_coverage["covered_documents"] == 1
    assert published["document_coverage"]["bound_documents"] == 1
    assert published["document_coverage"]["validated_documents"] == 1
    assert (
        control.get_document(registered["document"]["id"])["active_index_version_id"]
        == target["id"]
    )

    rolled_back = rollback_index("kb_default")
    assert rolled_back["active"]["id"] == old_index_id
    assert active_index_scopes("kb_default")[0]["index_version_id"] == old_index_id
    assert (
        control.get_document(registered["document"]["id"])["active_index_version_id"]
        == old_index_id
    )


def test_new_document_version_keeps_published_version_queryable(todo03_db: Path):
    first = _published_document(todo03_db)
    document_id = first["document"]["id"]
    active_version_id = first["document_version"]["id"]
    active_index_id = first["index_version"]["id"]

    source = todo03_db / "published-kb_default-v2.md"
    source.write_text("# published\nversion two is still staging", encoding="utf-8")
    second = control.register_uploaded_document(
        kb_id="kb_default",
        document_id=document_id,
        logical_name=source.name,
        content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        mime_type="text/markdown",
        object_key=str(source),
    )

    # v2 的 job/version 表达处理中；Document 继续表达当前 published v1 可用。
    detail = control.get_document(document_id)
    assert second["index_version"]["id"] != active_index_id
    assert second["index_version"]["status"] == "staging"
    assert (
        get_index_lifecycle(second["index_version"]["id"])["build_kind"]
        == "document_update"
    )
    assert second["ingestion_job"]["index_version_id"] == second["index_version"]["id"]
    assert second["document_version"]["status"] == "uploaded"
    assert second["ingestion_job"]["status"] == "queued"
    assert detail["status"] == "ready"
    assert detail["active_version_id"] == active_version_id
    assert active_index_scopes("kb_default") == [
        {
            "tenant_id": "tenant_default",
            "kb_id": "kb_default",
            "document_id": document_id,
            "document_version_id": active_version_id,
            "index_version_id": active_index_id,
            "collection_name": first["index_version"]["collection_alias"],
        }
    ]

    # 兼容修复前已落库的在途状态：精确 active_version/binding 仍必须可查询。
    with session_scope() as session:
        session.get(DocumentModel, document_id).status = "processing"
    assert active_index_scopes("kb_default")[0]["document_version_id"] == active_version_id


def test_document_update_cow_validates_then_atomically_switches_read_set(todo03_db: Path):
    first = _published_document(todo03_db)
    document_id = first["document"]["id"]
    old_version_id = first["document_version"]["id"]
    old_index_id = first["index_version"]["id"]

    source = todo03_db / "cow-v2.md"
    source.write_text("# published\ncopy on write version two", encoding="utf-8")
    second = control.register_uploaded_document(
        kb_id="kb_default",
        document_id=document_id,
        logical_name=source.name,
        content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        mime_type="text/markdown",
        object_key=str(source),
        parser_config_snapshot={"parser": "cow-v2"},
    )
    target = second["index_version"]
    assert target["status"] == "staging"
    target_lifecycle = get_index_lifecycle(target["id"])
    assert target_lifecycle["validation_status"] == "pending"
    assert target_lifecycle["previous_version_id"] == old_index_id
    assert get_index_lifecycle(old_index_id)["validation_status"] == "validated"

    claimed = jobs.claim_next_job("cow-worker", 60)
    assert claimed["id"] == second["ingestion_job"]["id"]
    jobs.mark_job_running(claimed["id"], claimed["lease_token"])
    jobs.update_job_stage(
        claimed["id"],
        claimed["lease_token"],
        "node_import_milvus",
        0.95,
    )
    with session_scope() as session:
        placement, family = session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(VectorPlacementModel.index_version_id == target["id"])
        ).one()
    jobs.record_index_manifest(
        claimed["id"],
        claimed["lease_token"],
        {
            "document_version_id": second["document_version"]["id"],
            "index_version_id": target["id"],
            "input_sha256": second["document_version"]["content_sha256"],
            "config_sha256": hashlib.sha256(b"cow-v2").hexdigest(),
            "expected_chunk_count": 3,
            "written_chunk_count": 3,
            "searchable_chunk_count": 3,
            "chunk_ids_sha256": hashlib.sha256(b"a\nb\nc").hexdigest(),
            "chunk_ids": ["a", "b", "c"],
            "object_keys": [],
            "vector_dim": target["vector_dim"],
            "collection_name": target["collection_alias"],
            "vector_placement_id": placement.id,
            "vector_placement_epoch": placement.epoch,
            "vector_compatibility_fingerprint": family.compatibility_fingerprint,
            "validation_status": "validated",
            "validation_errors": [],
        },
    )
    completed = jobs.complete_job(claimed["id"], claimed["lease_token"], 3)
    assert completed["stage"] == "validated"
    assert get_index_lifecycle(target["id"])["validation_status"] == "validated"
    # Individual job completion cannot mutate the published document/index pointers.
    assert control.get_document(document_id)["active_version_id"] == old_version_id
    assert active_index_scopes("kb_default")[0]["index_version_id"] == old_index_id
    assert get_index_lifecycle(old_index_id)["validation_status"] == "validated"

    published = publish_index(target["id"])
    assert published["status"] == "active"
    assert published["validation_status"] == "validated"
    assert control.get_document(document_id)["active_version_id"] == second["document_version"]["id"]
    assert active_index_scopes("kb_default") == [
        {
            "tenant_id": "tenant_default",
            "kb_id": "kb_default",
            "document_id": document_id,
            "document_version_id": second["document_version"]["id"],
            "index_version_id": target["id"],
            "collection_name": target["collection_alias"],
        }
    ]
    rolled_back = rollback_index("kb_default")
    assert rolled_back["active"]["id"] == old_index_id
    assert control.get_document(document_id)["active_version_id"] == old_version_id
    assert active_index_scopes("kb_default")[0]["document_version_id"] == old_version_id


def test_active_scope_fails_closed_when_active_index_is_not_validated(todo03_db: Path):
    registered = _published_document(todo03_db)
    assert active_index_scopes("kb_default")
    with session_scope() as session:
        index = session.get(IndexVersionModel, registered["index_version"]["id"])
        assert index is not None
        index.validation_status = "pending"
    assert active_index_scopes("kb_default") == []


def test_document_update_cow_rebuilds_other_documents_before_publish(todo03_db: Path):
    # Register both documents while the bootstrap index is empty.  Their frozen bootstrap
    # job admission keeps the existing multi-file first-upload behavior intact.
    registered: list[dict] = []
    for name, content in (("alpha.md", b"alpha v1"), ("beta.md", b"beta v1")):
        path = todo03_db / name
        path.write_bytes(content)
        registered.append(
            control.register_uploaded_document(
                kb_id="kb_default",
                logical_name=name,
                content_sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                mime_type="text/markdown",
                object_key=str(path),
            )
        )
    old_index_id = registered[0]["index_version"]["id"]
    assert {item["index_version"]["id"] for item in registered} == {old_index_id}
    for item in registered:
        job_id = item["ingestion_job"]["id"]
        control.start_ingestion_job(job_id)
        control.update_ingestion_job_stage(job_id, "node_import_milvus", 0.95)
        control.complete_ingestion_job(job_id, chunk_count=2)
    assert len(active_index_scopes("kb_default")) == 2

    alpha_v2 = todo03_db / "alpha-v2.md"
    alpha_v2.write_text("alpha v2", encoding="utf-8")
    update = control.register_uploaded_document(
        kb_id="kb_default",
        document_id=registered[0]["document"]["id"],
        logical_name="alpha.md",
        content_sha256=hashlib.sha256(alpha_v2.read_bytes()).hexdigest(),
        size_bytes=alpha_v2.stat().st_size,
        mime_type="text/markdown",
        object_key=str(alpha_v2),
        parser_config_snapshot={"parser": "cow-multi"},
    )
    target = update["index_version"]
    lifecycle = get_index_lifecycle(target["id"])
    assert lifecycle["manifest_summary"]["expected_documents"] == 2
    assert lifecycle["manifest_summary"]["target_document_versions"] == {
        registered[0]["document"]["id"]: update["document_version"]["id"],
        registered[1]["document"]["id"]: registered[1]["document_version"]["id"],
    }

    first_claim = jobs.claim_next_job("cow-multi-worker", 60)
    _finish_cow_job(first_claim, target)
    assert get_index_lifecycle(target["id"])["validation_status"] == "pending"
    # The old two-document read set remains complete until all target manifests succeed.
    assert {
        item["index_version_id"] for item in active_index_scopes("kb_default")
    } == {old_index_id}
    second_claim = jobs.claim_next_job("cow-multi-worker", 60)
    _finish_cow_job(second_claim, target)
    assert get_index_lifecycle(target["id"])["validation_status"] == "validated"

    publish_index(target["id"])
    scopes = active_index_scopes("kb_default")
    assert len(scopes) == 2
    assert {item["index_version_id"] for item in scopes} == {target["id"]}
    version_by_document = {
        item["document_id"]: item["document_version_id"] for item in scopes
    }
    assert version_by_document == {
        registered[0]["document"]["id"]: update["document_version"]["id"],
        registered[1]["document"]["id"]: registered[1]["document_version"]["id"],
    }


def test_new_index_document_set_does_not_change_retired_index_denominator(
    todo03_db: Path,
):
    first = _published_document(todo03_db)
    old_index_id = first["index_version"]["id"]

    second_source = todo03_db / "second-document.md"
    second_source.write_text("# second\na document added only in v2", encoding="utf-8")
    second = control.register_uploaded_document(
        kb_id="kb_default",
        logical_name=second_source.name,
        content_sha256=hashlib.sha256(second_source.read_bytes()).hexdigest(),
        size_bytes=second_source.stat().st_size,
        mime_type="text/markdown",
        object_key=str(second_source),
    )
    target = second["index_version"]
    assert target["id"] != old_index_id

    first_claim = jobs.claim_next_job("coverage-worker", 60)
    _finish_cow_job(first_claim, target)
    second_claim = jobs.claim_next_job("coverage-worker", 60)
    _finish_cow_job(second_claim, target)
    published = publish_index(target["id"])

    retired = get_index_lifecycle(old_index_id)
    assert retired["status"] == "retired"
    assert retired["document_coverage"] == {
        "expected_documents": 1,
        "covered_documents": 1,
        "bound_documents": 1,
        "validated_documents": 0,
        "coverage_basis": "index_bindings_legacy",
    }
    assert published["document_coverage"] == {
        "expected_documents": 2,
        "covered_documents": 2,
        "bound_documents": 2,
        "validated_documents": 2,
        "coverage_basis": "target_document_versions",
    }


def test_disable_and_soft_delete_remove_visibility_before_cleanup(todo03_db: Path):
    registered = _published_document(todo03_db)
    document_id = registered["document"]["id"]
    assert active_index_scopes("kb_default")

    disabled = set_document_enabled(document_id, False)
    assert disabled["status"] == "disabled"
    assert active_index_scopes("kb_default") == []
    assert set_document_enabled(document_id, True)["status"] == "ready"

    deleted = delete_document_with_cleanup(document_id, "soft")
    assert deleted["document"]["status"] == "deleted"
    assert deleted["cleanup_job"]["status"] == "queued"
    assert deleted["cleanup_job"]["plan"]["original_paths"] == []
    targets = deleted["cleanup_job"]["plan"]["milvus_filters"]
    assert targets
    assert all(
        set(item["identity"])
        == {
            "tenant_id",
            "kb_id",
            "document_id",
            "document_version_id",
            "index_version_id",
        }
        for item in targets
    )
    assert all(
        all(
            field in item["filter"]
            for field in (
                "tenant_id",
                "kb_id",
                "doc_id",
                "document_version_id",
                "index_version_id",
            )
        )
        for item in targets
    )
    assert active_index_scopes("kb_default") == []


def test_knowledge_base_delete_closes_visibility_and_creates_recursive_cleanup(todo03_db: Path):
    kb = control.create_knowledge_base("delete-me", "TODO-03 KB cleanup")
    _published_document(todo03_db, kb["id"])
    assert active_index_scopes(kb["id"])

    deleted = delete_knowledge_base_with_cleanup(kb["id"], "soft")
    assert deleted["knowledge_base"]["status"] == "deleted"
    assert deleted["cleanup_job"]["target_kind"] == "knowledge_base"
    assert deleted["cleanup_job"]["status"] == "queued"
    assert deleted["cleanup_job"]["plan"]["original_paths"] == []
    targets = deleted["cleanup_job"]["plan"]["milvus_filters"]
    assert targets
    assert all(item["identity"]["kb_id"] == kb["id"] for item in targets)
    assert all(item["identity"]["tenant_id"] == "tenant_default" for item in targets)
    assert all(
        "document_version_id" in item["filter"] and "index_version_id" in item["filter"]
        for item in targets
    )
    assert active_index_scopes(kb["id"]) == []
    # 不指定 KB 的全局检索同样不能看见已删除知识库。
    assert all(scope["kb_id"] != kb["id"] for scope in active_index_scopes())


def test_cleanup_executor_uses_five_dimensional_filter(monkeypatch, todo03_db: Path):
    registered = _published_document(todo03_db)
    deleted = delete_document_with_cleanup(
        registered["document"]["id"],
        "soft",
    )

    class CleanupClient:
        def __init__(self):
            self.calls = []

        def has_collection(self, _collection):
            return True

        def delete(self, **kwargs):
            self.calls.append(kwargs)
            return {"delete_count": 2}

    client = CleanupClient()
    monkeypatch.setattr(milvus_gateway, "client", lambda: client)

    result = execute_cleanup(deleted["cleanup_job"]["id"])

    assert result["status"] == "succeeded"
    assert client.calls
    assert all(
        all(
            field in call["filter"]
            for field in (
                "tenant_id",
                "kb_id",
                "doc_id",
                "document_version_id",
                "index_version_id",
            )
        )
        for call in client.calls
    )


def test_cleanup_executor_rejects_legacy_doc_only_plan(
    monkeypatch,
    todo03_db: Path,
):
    registered = _published_document(todo03_db)
    deleted = delete_document_with_cleanup(
        registered["document"]["id"],
        "soft",
    )
    cleanup_id = deleted["cleanup_job"]["id"]
    with session_scope() as session:
        cleanup = session.get(CleanupJobModel, cleanup_id)
        cleanup.plan_json = {
            "milvus_filters": [
                {
                    "collection": "kb_chunks",
                    "filter": (f'doc_id == "{registered["document"]["id"]}"'),
                }
            ],
            "object_keys": [],
            "original_paths": [],
        }

    class RefusingClient:
        def has_collection(self, _collection):
            raise AssertionError("unsafe cleanup must not reach Milvus")

    monkeypatch.setattr(milvus_gateway, "client", lambda: RefusingClient())

    result = execute_cleanup(cleanup_id)

    assert result["status"] == "partial"
    assert "five-dimensional identity" in result["error_detail"]


def test_cleanup_executor_rejects_tampered_filter_even_with_full_identity(
    monkeypatch,
    todo03_db: Path,
):
    registered = _published_document(todo03_db)
    deleted = delete_document_with_cleanup(
        registered["document"]["id"],
        "soft",
    )
    cleanup_id = deleted["cleanup_job"]["id"]
    with session_scope() as session:
        cleanup = session.get(CleanupJobModel, cleanup_id)
        plan = deepcopy(cleanup.plan_json)
        target = plan["milvus_filters"][0]
        target["filter"] = f'kb_id == "{target["identity"]["kb_id"]}"'
        cleanup.plan_json = plan

    class RefusingClient:
        def has_collection(self, _collection):
            raise AssertionError("tampered cleanup must not reach Milvus")

    monkeypatch.setattr(milvus_gateway, "client", lambda: RefusingClient())

    result = execute_cleanup(cleanup_id)

    assert result["status"] == "partial"
    assert "identity/filter does not match" in result["error_detail"]


def test_cleanup_executor_rejects_identity_outside_job_target(
    monkeypatch,
    todo03_db: Path,
):
    registered = _published_document(todo03_db)
    deleted = delete_document_with_cleanup(
        registered["document"]["id"],
        "soft",
    )
    cleanup_id = deleted["cleanup_job"]["id"]
    with session_scope() as session:
        cleanup = session.get(CleanupJobModel, cleanup_id)
        plan = deepcopy(cleanup.plan_json)
        target = plan["milvus_filters"][0]
        target["identity"]["document_id"] = "doc_other"
        target["filter"] = (
            f'tenant_id == "{target["identity"]["tenant_id"]}" and '
            f'kb_id == "{target["identity"]["kb_id"]}" and '
            'doc_id == "doc_other" and '
            f'document_version_id == "{target["identity"]["document_version_id"]}" and '
            f'index_version_id == "{target["identity"]["index_version_id"]}"'
        )
        cleanup.plan_json = plan

    class RefusingClient:
        def has_collection(self, _collection):
            raise AssertionError("out-of-scope cleanup must not reach Milvus")

    monkeypatch.setattr(milvus_gateway, "client", lambda: RefusingClient())

    result = execute_cleanup(cleanup_id)

    assert result["status"] == "partial"
    assert "identity/filter does not match" in result["error_detail"]


def test_active_scope_filter_cannot_be_overridden_by_client_doc_ids():
    scopes = [
        {
            "tenant_id": "tenant_default",
            "kb_id": "kb_1",
            "document_id": "doc_active",
            "document_version_id": "dv_active",
            "index_version_id": "idx_active",
            "collection_name": "chunks",
        }
    ]
    expression = build_active_scope_expr(scopes, ["doc_active", "doc_inactive"])
    assert "doc_active" in expression
    assert "doc_inactive" not in expression
    assert "dv_active" in expression
    assert "idx_active" in expression
    assert build_active_scope_expr([], None).endswith('doc_id == "__stt_no_active_document__"')


class FakeMilvusClient:
    def __init__(self):
        self.rows: list[dict] = []
        self.flush_count = 0

    def has_collection(self, _name):
        return True

    def describe_collection(self, _name):
        fields = {
            "chunk_id",
            "doc_id",
            "kb_id",
            "docnm_kwd",
            "content_with_weight",
            "dense_vector",
            "sparse_vector",
            "type",
            "img_id",
            "id",
            "create_time",
            "create_timestamp_flt",
            "available_int",
            "position_int",
            "important_kwd",
            "important_tks",
            "question_kwd",
            "question_tks",
            "meta_fields",
            "tag_kwd",
        }
        return {"fields": [{"name": name} for name in fields]}

    def delete(self, *, collection_name, filter):
        self.rows = []
        return {"delete_count": 0}

    def insert(self, *, collection_name, data):
        self.rows = [{**row, "chunk_id": index + 1} for index, row in enumerate(data)]
        return {"insert_count": len(data), "ids": [row["chunk_id"] for row in self.rows]}

    def flush(self, *, collection_name):
        self.flush_count += 1

    def query(self, *, collection_name, filter, output_fields, limit):
        return [{key: row.get(key) for key in output_fields} for row in self.rows[:limit]]


def test_milvus_staging_replay_is_deterministic_and_manifest_is_complete(monkeypatch):
    fake = FakeMilvusClient()
    monkeypatch.setattr(milvus_gateway, "client", lambda: fake)
    monkeypatch.setattr(
        index_service,
        "record_evidence_batch",
        lambda evidences, **_: [
            evidence.model_dump(mode="json", exclude_none=True) for evidence in evidences
        ],
    )
    index_service._verified_collections.clear()
    state = {
        "task_id": "job_1",
        "document_id": "doc_1",
        "document_version_id": "dv_1",
        "index_version_id": "idx_1",
        "tenant_id": "tenant_default",
        "kb_id": "kb_1",
        "collection_name": "chunks",
        "vector_dim": 3,
        "file_title": "manual.md",
        "chunks": [
            {
                "content": "alpha",
                "type": "text",
                "dense_vector": [1.0, 0.0, 0.0],
                "sparse_vector": {1: 1.0},
            },
            {
                "content": "beta",
                "type": "text",
                "dense_vector": [0.0, 1.0, 0.0],
                "sparse_vector": {2: 1.0},
                "img_id": "imagetemps-dv_1/a.jpg",
            },
        ],
        "toc": {"items": [{"level": 1, "title": "目录"}], "content_with_weight": "[]"},
        "uploaded_object_keys": ["kb-images/dv_1/idx_1/a.jpg"],
    }
    first = index_service.index_chunks(state)
    first_ids = list(first["index_manifest"]["chunk_ids"])
    second = index_service.index_chunks(state)
    manifest = second["index_manifest"]
    assert manifest["validation_status"] == "validated"
    assert manifest["expected_chunk_count"] == 2
    assert manifest["written_chunk_count"] == 2
    assert manifest["searchable_chunk_count"] == 2
    assert manifest["chunk_ids"] == first_ids
    assert len(fake.rows) == 2
    assert fake.flush_count == 2
    assert manifest["object_keys"] == ["kb-images/dv_1/idx_1/a.jpg"]
    assert second["toc"]["chunk_id"] is None
