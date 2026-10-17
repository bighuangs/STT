from __future__ import annotations

from hashlib import sha256

from fastapi.testclient import TestClient
import pytest

from app.api.http import import_server, query_server
from app.application import control_plane_service as control
from app.application.ingestion_job_service import worker_heartbeat
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.models import DocumentModel
from app.infra.persistence.sql.repositories import ResourceConflictError


ADMIN = {
    "email": "admin@stt.local",
    "password": "Stt-Admin-ChangeMe!",  # secret-scan: allow-test
}


def _login(
    client: TestClient,
    *,
    email: str = ADMIN["email"],
    password: str = ADMIN["password"],
    tenant_id: str | None = None,
):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password, "tenant_id": tenant_id},
    )
    assert response.status_code == 200, response.text
    return response


def test_ui_assets_are_public_but_bootstrap_requires_login(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-assets.db').as_posix()}")
    with TestClient(import_server.app) as client:
        page = client.get("/html")
        css = client.get("/ui-assets/stt-console.css")
        js = client.get("/ui-assets/stt-console.js")
        nav_icons = client.get("/ui-assets/lucide-nav.svg")
        bootstrap = client.get("/api/v1/ui/bootstrap")

        assert page.status_code == 200
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert js.status_code == 200
        assert "javascript" in js.headers["content-type"]
        assert nav_icons.status_code == 200
        assert nav_icons.headers["content-type"].startswith("image/svg+xml")
        assert "<symbol" in nav_icons.text
        assert 'id="message-circle"' in nav_icons.text
        assert bootstrap.status_code == 401
        for response in (page, css, js, nav_icons):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["x-frame-options"] == "DENY"
            assert "default-src 'self'" in response.headers["content-security-policy"]


def test_ui_bootstrap_is_tenant_and_entitlement_scoped_on_both_services(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-scope.db').as_posix()}")
    with TestClient(import_server.app) as client:
        first_login = _login(client)
        tenant_a = first_login.json()["principal"]["tenant_id"]
        kb_a_response = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 租户 A 知识库"},
        )
        assert kb_a_response.status_code == 201
        kb_a = kb_a_response.json()["data"]["id"]

        tenant_b_response = client.post(
            "/api/v1/admin/tenants",
            json={"name": "TODO38 租户 B"},
        )
        assert tenant_b_response.status_code == 200
        tenant_b = tenant_b_response.json()["id"]
        _login(client, tenant_id=tenant_b)
        kb_b_response = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 租户 B 知识库"},
        )
        assert kb_b_response.status_code == 201
        kb_b = kb_b_response.json()["data"]["id"]

        _login(client, tenant_id=tenant_a)
        import_bootstrap = client.get("/api/v1/ui/bootstrap")
        assert import_bootstrap.status_code == 200
        payload = import_bootstrap.json()
        visible = {item["id"] for item in payload["knowledge_bases"]}
        assert kb_a in visible
        assert kb_b not in visible
        assert payload["identity"]["tenant_id"] == tenant_a
        assert payload["service"] == "import"
        assert payload["schema_version"] == "stt.ui.bootstrap.v1"
        assert "*" in payload["permissions"]
        assert client.get(f"/api/v1/knowledge-bases/{kb_b}").status_code == 404

    with TestClient(query_server.app) as client:
        _login(client, tenant_id=tenant_a)
        query_bootstrap = client.get("/api/v1/ui/bootstrap")
        assert query_bootstrap.status_code == 200
        query_payload = query_bootstrap.json()
        assert query_payload["service"] == "query"
        assert {item["id"] for item in query_payload["knowledge_bases"]} == visible


def test_ui_bootstrap_does_not_turn_frontend_hiding_into_authorization(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-rbac.db').as_posix()}")
    viewer_password = "Viewer-TODO38-Password!"  # secret-scan: allow-test
    with TestClient(import_server.app) as client:
        login = _login(client)
        tenant_id = login.json()["principal"]["tenant_id"]
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 管理员知识库"},
        )
        assert created.status_code == 201
        user = client.post(
            "/api/v1/admin/users",
            json={
                "email": "todo38-viewer@stt.local",
                "password": viewer_password,
                "display_name": "TODO38 Viewer",
                "role": "Viewer",
            },
        )
        assert user.status_code == 200

        _login(
            client,
            email="todo38-viewer@stt.local",
            password=viewer_password,
            tenant_id=tenant_id,
        )
        bootstrap = client.get("/api/v1/ui/bootstrap")
        assert bootstrap.status_code == 200
        assert "kb:create" not in bootstrap.json()["permissions"]
        denied = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "前端隐藏也不能越权创建"},
        )
        assert denied.status_code == 403


def test_ui_bootstrap_keeps_active_version_queryable_while_next_version_processes(
    tmp_path,
):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-active-version.db').as_posix()}")
    with TestClient(import_server.app) as client:
        _login(client)
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 活动版本兼容知识库"},
        )
        assert created.status_code == 201, created.text
        kb_id = created.json()["data"]["id"]

        v1 = b"# manual\npublished version one"
        first = control.register_uploaded_document(
            kb_id=kb_id,
            logical_name="manual.md",
            content_sha256=sha256(v1).hexdigest(),
            size_bytes=len(v1),
            mime_type="text/markdown",
            object_key=str(tmp_path / "manual-v1.md"),
        )
        job_id = first["ingestion_job"]["id"]
        control.start_ingestion_job(job_id)
        control.update_ingestion_job_stage(job_id, "node_import_milvus", 0.95)
        control.complete_ingestion_job(job_id, chunk_count=2)

        v2 = b"# manual\nversion two is processing"
        second = control.register_uploaded_document(
            kb_id=kb_id,
            document_id=first["document"]["id"],
            logical_name="manual.md",
            content_sha256=sha256(v2).hexdigest(),
            size_bytes=len(v2),
            mime_type="text/markdown",
            object_key=str(tmp_path / "manual-v2.md"),
        )
        assert second["document"]["status"] == "ready"

        canonical_bootstrap = client.get("/api/v1/ui/bootstrap")
        assert canonical_bootstrap.status_code == 200, canonical_bootstrap.text
        canonical_summary = next(
            item
            for item in canonical_bootstrap.json()["knowledge_bases"]
            if item["id"] == kb_id
        )
        assert canonical_summary["readiness"] == "ready"
        assert canonical_summary["readiness_message"] == "可查询（有文档正在更新）"
        assert canonical_summary["queryable_document_count"] == 1
        assert canonical_summary["processing_document_count"] == 1

        # 模拟修复前已由旧服务写入的真实在途状态；bootstrap 不能因此把 v1
        # 误报为不可查询。
        with session_scope() as session:
            session.get(DocumentModel, first["document"]["id"]).status = "processing"

        bootstrap = client.get("/api/v1/ui/bootstrap")
        assert bootstrap.status_code == 200, bootstrap.text
        summary = next(
            item for item in bootstrap.json()["knowledge_bases"] if item["id"] == kb_id
        )
        assert summary["readiness"] == "ready"
        assert summary["readiness_message"] == "可查询（有文档正在更新）"
        assert summary["ready_document_count"] == 1
        assert summary["queryable_document_count"] == 1
        assert summary["processing_document_count"] == 1

        # COW staging 不允许旧测试 helper 绕过 validated manifest + durable lease
        # 直接发布；正式 Worker 完成后仍需显式发布 index 才切换文档版本。
        second_job_id = second["ingestion_job"]["id"]
        control.start_ingestion_job(second_job_id)
        control.update_ingestion_job_stage(second_job_id, "node_import_milvus", 0.95)
        with pytest.raises(ResourceConflictError, match="validated manifest"):
            control.complete_ingestion_job(second_job_id, chunk_count=3)
        indexes = client.get(f"/api/v1/knowledge-bases/{kb_id}/index-versions")
        assert indexes.status_code == 200, indexes.text
        active_index = next(
            item for item in indexes.json()["data"] if item["status"] == "active"
        )
        assert active_index["document_coverage"] == {
            "expected_documents": 1,
            "covered_documents": 1,
            "bound_documents": 1,
            "validated_documents": 0,
            "coverage_basis": "index_bindings_legacy",
        }
        staging_index = next(
            item for item in indexes.json()["data"] if item["status"] == "staging"
        )
        assert staging_index["validation_status"] == "pending"
        assert staging_index["document_coverage"] == {
            "expected_documents": 1,
            "covered_documents": 0,
            "bound_documents": 0,
            "validated_documents": 0,
            "coverage_basis": "target_document_versions",
        }


@pytest.mark.parametrize("access_cookie_state", ["missing", "invalid"])
def test_logout_is_idempotent_and_clears_both_cookies_without_valid_access(
    tmp_path,
    access_cookie_state: str,
):
    configure_database(
        f"sqlite:///{(tmp_path / f'todo38-logout-{access_cookie_state}.db').as_posix()}"
    )
    with TestClient(import_server.app) as client:
        login = _login(client)
        refresh_token = login.json()["refresh_token"]

        # 模拟最需要 logout 兜底的两种浏览器状态：access cookie 已被浏览器
        # 删除，或仍存在但签名/内容已经失效；refresh cookie 保持有效。
        client.cookies.clear()
        client.cookies.set(
            "stt_refresh_token",
            refresh_token,
            domain="testserver.local",
            path="/api/v1/auth",
        )
        if access_cookie_state == "invalid":
            client.cookies.set(
                "stt_access_token",
                "invalid-or-expired-access-token",
                domain="testserver.local",
                path="/",
            )

        logout = client.post("/api/v1/auth/logout")

        assert logout.status_code == 200, logout.text
        assert logout.json() == {"message": "已退出当前浏览器会话"}
        set_cookie_headers = "\n".join(logout.headers.get_list("set-cookie")).lower()
        assert "stt_access_token=" in set_cookie_headers
        assert "stt_refresh_token=" in set_cookie_headers
        assert set_cookie_headers.count("max-age=0") == 2
        remaining_cookie_names = {cookie.name for cookie in client.cookies.jar}
        assert "stt_access_token" not in remaining_cookie_names
        assert "stt_refresh_token" not in remaining_cookie_names

        # 清 Cookie 之外还必须撤销服务端 refresh 会话，避免旧值被重新注入。
        refresh = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert refresh.status_code == 401


def test_cleanup_jobs_are_kb_filtered_and_keep_scope_after_soft_delete(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-cleanup-scope.db').as_posix()}")
    with TestClient(import_server.app) as client:
        login = _login(client)
        tenant_id = login.json()["principal"]["tenant_id"]
        created_a = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 待清理知识库 A"},
        )
        created_b = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 待清理知识库 B"},
        )
        assert created_a.status_code == 201, created_a.text
        assert created_b.status_code == 201, created_b.text
        kb_a = created_a.json()["data"]["id"]
        kb_b = created_b.json()["data"]["id"]

        deleted_a = client.delete(f"/api/v1/knowledge-bases/{kb_a}?mode=soft")
        deleted_b = client.delete(f"/api/v1/knowledge-bases/{kb_b}?mode=soft")
        assert deleted_a.status_code == 200, deleted_a.text
        assert deleted_b.status_code == 200, deleted_b.text
        cleanup_a_id = deleted_a.json()["data"]["cleanup_job"]["id"]
        cleanup_b_id = deleted_b.json()["data"]["cleanup_job"]["id"]

        # 目标知识库已经软删，kb_id 过滤仍必须在服务端识别其稳定归属。
        filtered_a = client.get(f"/api/v1/cleanup-jobs?kb_id={kb_a}&limit=100")
        assert filtered_a.status_code == 200, filtered_a.text
        rows_a = filtered_a.json()["data"]
        assert [item["id"] for item in rows_a] == [cleanup_a_id]
        assert cleanup_b_id not in {item["id"] for item in rows_a}
        assert rows_a[0]["scope"] == {
            "tenant_id": tenant_id,
            "kb_id": kb_a,
            "kb_name": "TODO38 待清理知识库 A",
            "document_id": None,
            "document_name": None,
            "index_version_id": None,
            "display_name": "TODO38 待清理知识库 A",
        }

        filtered_b = client.get(f"/api/v1/cleanup-jobs?kb_id={kb_b}&limit=100")
        assert filtered_b.status_code == 200, filtered_b.text
        rows_b = filtered_b.json()["data"]
        assert [item["id"] for item in rows_b] == [cleanup_b_id]
        assert rows_b[0]["scope"]["tenant_id"] == tenant_id
        assert rows_b[0]["scope"]["kb_id"] == kb_b
        assert rows_b[0]["scope"]["display_name"] == "TODO38 待清理知识库 B"


def test_ingestion_workers_do_not_expose_other_tenant_active_jobs(
    tmp_path,
    monkeypatch,
):
    configure_database(f"sqlite:///{(tmp_path / 'todo38-worker-scope.db').as_posix()}")
    monkeypatch.setattr(import_server, "ORIGINALS_ROOT", tmp_path / "originals")

    with TestClient(import_server.app) as client:
        login_a = _login(client)
        tenant_a = login_a.json()["principal"]["tenant_id"]
        created_a = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 Worker 租户 A"},
        )
        assert created_a.status_code == 201, created_a.text
        kb_a = created_a.json()["data"]["id"]
        uploaded_a = client.post(
            f"/api/v1/knowledge-bases/{kb_a}/documents",
            files={"file": ("tenant-a.md", b"# tenant A", "text/markdown")},
        )
        assert uploaded_a.status_code == 202, uploaded_a.text
        job_a = uploaded_a.json()["ingestion_job_id"]

        created_tenant_b = client.post(
            "/api/v1/admin/tenants",
            json={"name": "TODO38 Worker 租户 B"},
        )
        assert created_tenant_b.status_code == 200, created_tenant_b.text
        tenant_b = created_tenant_b.json()["id"]
        _login(client, tenant_id=tenant_b)
        created_b = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO38 Worker 租户 B"},
        )
        assert created_b.status_code == 201, created_b.text
        kb_b = created_b.json()["data"]["id"]
        uploaded_b = client.post(
            f"/api/v1/knowledge-bases/{kb_b}/documents",
            files={"file": ("tenant-b.md", b"# tenant B", "text/markdown")},
        )
        assert uploaded_b.status_code == 202, uploaded_b.text
        job_b = uploaded_b.json()["ingestion_job_id"]

        worker_heartbeat(
            "todo38-worker-a",
            status="running",
            current_job_id=job_a,
            metadata={"tenant_canary": tenant_a},
        )
        worker_heartbeat(
            "todo38-worker-b",
            status="running",
            current_job_id=job_b,
            metadata={"tenant_canary": tenant_b},
        )
        worker_heartbeat(
            "todo38-worker-idle",
            status="idle",
            current_job_id=None,
            metadata={"host": "platform-internal"},
        )

        _login(client, tenant_id=tenant_a)
        workers_a = client.get("/api/v1/ingestion-workers")
        assert workers_a.status_code == 200, workers_a.text
        assert {
            (item["id"], item["current_job_id"])
            for item in workers_a.json()["data"]
        } == {("todo38-worker-a", job_a)}

        _login(client, tenant_id=tenant_b)
        workers_b = client.get("/api/v1/ingestion-workers")
        assert workers_b.status_code == 200, workers_b.text
        assert {
            (item["id"], item["current_job_id"])
            for item in workers_b.json()["data"]
        } == {("todo38-worker-b", job_b)}
