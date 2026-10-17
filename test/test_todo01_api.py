from pathlib import Path

from fastapi.testclient import TestClient

from app.application.control_plane_service import bootstrap_control_plane, fail_ingestion_job
from app.api.http import import_server
from app.infra.persistence.sql.database import configure_database


def login_admin(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@stt.local", "password": "Stt-Admin-ChangeMe!"},
    )
    assert response.status_code == 200


def test_control_plane_api_and_upload(monkeypatch, tmp_path: Path):
    configure_database(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    bootstrap_control_plane()
    monkeypatch.setattr(import_server, "run_graph_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(import_server, "ORIGINALS_ROOT", tmp_path / "originals")

    with TestClient(import_server.app) as client:
        html = client.get("/html")
        assert html.status_code == 200
        assert "创建或选择知识库" in html.text
        assert "/api/v1/knowledge-bases" in html.text
        assert "formData.append('file', file)" in html.text
        assert "stt.selectedKnowledgeBaseId" in html.text
        assert "restoreJobCards(jobs)" in html.text
        assert "重新入队" in html.text
        assert "创建全量重建" in html.text
        assert "索引一致性发布、重建、回滚与清理" in html.text
        assert client.get("/api/v1/knowledge-bases").status_code == 401
        login_admin(client)

        create = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "网页验收库", "description": "API test"},
            headers={"X-Request-ID": "req_test"},
        )
        assert create.status_code == 201
        assert create.headers["X-Request-ID"] == "req_test"
        kb = create.json()["data"]

        duplicate = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "网页验收库", "description": "duplicate"},
        )
        assert duplicate.status_code == 409

        invalid_config = client.post(
            f"/api/v1/knowledge-bases/{kb['id']}/documents",
            files={"file": ("sample.md", b"# sample", "text/markdown")},
            data={"parser_config": "[]"},
        )
        assert invalid_config.status_code == 422

        upload = client.post(
            f"/api/v1/knowledge-bases/{kb['id']}/documents",
            files={"file": ("sample.md", b"# sample\nhello", "text/markdown")},
        )
        assert upload.status_code == 202
        payload = upload.json()
        assert payload["document_id"].startswith("doc_")
        assert payload["document_version_id"].startswith("dv_")
        assert payload["ingestion_job_id"].startswith("job_")
        assert payload["document_id"] != payload["ingestion_job_id"]

        job = client.get(f"/api/v1/ingestion-jobs/{payload['ingestion_job_id']}")
        assert job.status_code == 200
        assert job.json()["data"]["status"] == "queued"

        document = client.get(f"/api/v1/documents/{payload['document_id']}")
        assert document.status_code == 200
        detail = document.json()["data"]
        assert detail["logical_name"] == "sample.md"
        assert len(detail["versions"]) == 1

        listed = client.get(f"/api/v1/knowledge-bases/{kb['id']}/documents")
        assert listed.status_code == 200
        assert len(listed.json()["data"]) == 1

        failed = fail_ingestion_job(
            payload["ingestion_job_id"], "TEST_FAILURE", "retry endpoint test"
        )
        assert failed["status"] == "failed"
        retried = client.post(
            f"/api/v1/ingestion-jobs/{payload['ingestion_job_id']}/retry"
        )
        assert retried.status_code == 202
        retried_job = retried.json()["data"]
        assert retried_job["id"] != payload["ingestion_job_id"]
        assert retried_job["document_version_id"] == payload["document_version_id"]
        assert retried_job["status"] == "queued"

        retry_active = client.post(
            f"/api/v1/ingestion-jobs/{retried_job['id']}/retry"
        )
        assert retry_active.status_code == 409


def test_todo03_empty_index_rebuild_publish_and_rollback_api(tmp_path: Path):
    """空知识库也走同一套显式发布协议，便于验证 API 路由和原子槽切换。"""
    configure_database(f"sqlite:///{(tmp_path / 'todo03-api.db').as_posix()}")
    bootstrap_control_plane()

    with TestClient(import_server.app) as client:
        login_admin(client)
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO03 空库", "description": "index lifecycle API"},
        )
        assert created.status_code == 201
        kb_id = created.json()["data"]["id"]

        indexes = client.get(f"/api/v1/knowledge-bases/{kb_id}/index-versions")
        assert indexes.status_code == 200
        old_active = indexes.json()["data"][0]
        assert old_active["status"] == "active"

        rebuilt = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/index-versions", json={}
        )
        assert rebuilt.status_code == 202
        staging = rebuilt.json()["data"]["index_version"]
        assert rebuilt.json()["data"]["jobs"] == []
        assert staging["status"] == "staging"
        assert staging["validation_status"] == "validated"
        assert staging["previous_version_id"] == old_active["id"]

        published = client.post(f"/api/v1/index-versions/{staging['id']}:publish")
        assert published.status_code == 200
        assert published.json()["data"]["status"] == "active"

        rolled_back = client.post(
            f"/api/v1/knowledge-bases/{kb_id}:rollback-index"
        )
        assert rolled_back.status_code == 200
        assert rolled_back.json()["data"]["active"]["id"] == old_active["id"]

        cleanup = client.get("/api/v1/cleanup-jobs")
        assert cleanup.status_code == 200
        assert cleanup.json()["data"] == []

        deleted_kb = client.delete(f"/api/v1/knowledge-bases/{kb_id}?mode=soft")
        assert deleted_kb.status_code == 200
        assert deleted_kb.json()["data"]["knowledge_base"]["status"] == "deleted"
        assert deleted_kb.json()["data"]["cleanup_job"]["target_kind"] == "knowledge_base"
