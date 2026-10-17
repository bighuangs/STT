from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.api.http import import_server, query_server
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.models import DocumentModel, DocumentVersionModel
from app.security.http import action_for


ADMIN = {
    "email": "admin@stt.local",
    "password": "Stt-Admin-ChangeMe!",
    "tenant_id": "tenant_default",
}
EDITOR = {
    "email": "p2-editor@stt.local",
    "password": "P2-Editor-ChangeMe!",  # secret-scan: allow-test
    "tenant_id": "tenant_default",
}
KB_ADMIN = {
    "email": "p2-kb-admin@stt.local",
    "password": "P2-KBAdmin-ChangeMe!",  # secret-scan: allow-test
    "tenant_id": "tenant_default",
}


def _make_png_1x1() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


PNG_1X1 = _make_png_1x1()


def _login(
    client: TestClient,
    credentials: dict[str, str] | None = None,
) -> None:
    response = client.post("/api/v1/auth/login", json=credentials or ADMIN)
    assert response.status_code == 200, response.text


def _create_kb(database_path: Path) -> str:
    configure_database(f"sqlite:///{database_path.as_posix()}")
    with TestClient(import_server.app) as client:
        _login(client)
        response = client.post(
            "/api/v1/knowledge-bases",
            json={
                "name": "P2 HAK180 验收知识库",
                "external_llm_allowed": True,
                "web_search_allowed": False,
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["data"]["id"]


def _memory_policy(kb_id: str) -> dict:
    return {
        "resource_type": "memory_policy",
        "logical_id": "p2-test-memory-policy",
        "version": "1.0.0",
        "schema_version": "memory-policy.v1",
        "kb_id": kb_id,
        "payload": {
            "auto_candidate_extraction_enabled": False,
            "auto_activation_enabled": False,
            "preference_ttl_days": 365,
            "profile_fact_ttl_days": 180,
            "episodic_ttl_days": 30,
            "max_context_tokens": 256,
            "max_hits": 8,
            "embedding_version": "memory-lexical/1.0.0",
        },
    }


def _compliance_metadata_schema(kb_id: str) -> dict:
    return {
        "resource_type": "metadata_schema",
        "logical_id": "p2-compliance-metadata",
        "version": "1",
        "schema_version": "metadata-schema.v1",
        "kb_id": kb_id,
        "payload": {
            "schema_version": "metadata-schema.v1",
            "schema_id": "p2-compliance-metadata",
            "tenant_id": "tenant_default",
            "kb_id": kb_id,
            "version": 1,
            "revision": 1,
            "status": "published",
            "fields": [
                {
                    "key": "risk_level",
                    "value_type": "integer",
                    "description": "风险等级",
                    "scope": "document",
                    "compliance_controlled": True,
                }
            ],
            "created_by": "bootstrap-admin",
            "created_at": "2026-07-24T00:00:00+00:00",
            "published_at": "2026-07-24T00:00:00+00:00",
        },
    }


def _create_document_version(database_path: Path, kb_id: str) -> tuple[str, str]:
    configure_database(f"sqlite:///{database_path.as_posix()}")
    document_id = "doc_p2_image"
    version_id = "dv_p2_image"
    with session_scope() as session:
        session.add(
            DocumentModel(
                id=document_id,
                kb_id=kb_id,
                logical_name="hak180-page1.png",
                source_type="upload",
                status="ready",
            )
        )
        session.flush()
        session.add(
            DocumentVersionModel(
                id=version_id,
                document_id=document_id,
                version_no=1,
                content_sha256="a" * 64,
                pipeline_fingerprint="b" * 64,
                size_bytes=len(PNG_1X1),
                mime_type="image/png",
                object_key="test/hak180-page1.png",
                parser_config_snapshot={},
                status="published",
            )
        )
    return document_id, version_id


def test_p2_routes_have_explicit_rbac_actions() -> None:
    assert action_for("GET", "/api/v1/p2/capabilities") == "p2:read"
    assert action_for("POST", "/api/v1/p2/self-check") == "p2:run"
    assert action_for("POST", "/api/v1/p2/model-routing:preview") == "p2:run"
    assert action_for("POST", "/api/v1/p2/resources") == "p2:manage"
    assert action_for("POST", "/api/v1/p2/resources/cap_1:publish") == "p2:manage"


def test_p2_page_self_check_hak180_and_persistent_resource_lifecycle(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "p2-api.db"
    kb_id = _create_kb(database_path)

    configure_database(f"sqlite:///{database_path.as_posix()}")
    with TestClient(query_server.app) as client:
        page = client.get("/p2/html")
        assert page.status_code == 200
        assert "P2 高级能力实验台" in page.text
        assert "保存演示策略会写控制库，额外需要 p2:manage" in page.text
        assert 'id="runPersistBtn"' in page.text
        assert 'id="runPersistBtn" type="button" class="secondary" disabled' in page.text
        assert client.get("/api/v1/p2/capabilities").status_code == 401
        _login(client)

        catalog = client.get("/api/v1/p2/capabilities")
        assert catalog.status_code == 200, catalog.text
        assert len(catalog.json()["data"]["capabilities"]) == 7
        assert catalog.json()["data"]["execution_policy"]["all_new_routes_default_disabled"]

        self_check = client.post(
            "/api/v1/p2/self-check",
            json={"kb_id": kb_id, "persist_demo_resources": False},
        )
        assert self_check.status_code == 200, self_check.text
        assert self_check.json()["data"]["passed"] is True
        checks = self_check.json()["data"]["checks"]
        assert [item["todo"] for item in checks] == [
            "TODO-13",
            "TODO-15",
            "TODO-16",
            "TODO-17",
            "TODO-21",
            "TODO-22",
            "TODO-23",
        ]

        dataset = client.get("/api/v1/p2/hak180-multimodal-dataset")
        assert dataset.status_code == 200, dataset.text
        assert dataset.json()["data"]["runtime_verification"]["pdf_hash_matches"]
        assert len(dataset.json()["data"]["cases"]) == 4

        created = client.post("/api/v1/p2/resources", json=_memory_policy(kb_id))
        assert created.status_code == 201, created.text
        resource = created.json()["data"]
        assert resource["status"] == "draft"
        assert resource["enabled"] is False
        replay = client.post("/api/v1/p2/resources", json=_memory_policy(kb_id))
        assert replay.status_code == 201
        assert replay.json()["data"]["id"] == resource["id"]

        published = client.post(
            f"/api/v1/p2/resources/{resource['id']}:publish",
            json={"enable": False},
        )
        assert published.status_code == 200, published.text
        assert published.json()["data"]["status"] == "published"
        enabled = client.post(
            f"/api/v1/p2/resources/{resource['id']}:enable",
            json={"enabled": True},
        )
        assert enabled.status_code == 200, enabled.text
        assert enabled.json()["data"]["enabled"] is True

    # 新 TestClient 模拟刷新/服务会话重建，事实仍来自 SQLite 控制库。
    configure_database(f"sqlite:///{database_path.as_posix()}")
    with TestClient(query_server.app) as client:
        _login(client)
        listing = client.get(f"/api/v1/p2/resources?resource_type=memory_policy&kb_id={kb_id}")
        assert listing.status_code == 200, listing.text
        items = listing.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["status"] == "published"
        assert items[0]["enabled"] is True


def test_editor_can_run_self_check_but_cannot_persist_demo_resources(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "p2-editor-self-check.db"
    kb_id = _create_kb(database_path)
    configure_database(f"sqlite:///{database_path.as_posix()}")

    with TestClient(query_server.app) as client:
        _login(client)
        created_editor = client.post(
            "/api/v1/admin/users",
            json={
                "email": EDITOR["email"],
                "password": EDITOR["password"],
                "display_name": "P2 Editor",
                "role": "Editor",
            },
        )
        assert created_editor.status_code == 200, created_editor.text
        assert created_editor.json()["role"] == "Editor"

    with TestClient(query_server.app) as client:
        _login(client, EDITOR)

        # persist_demo_resources 默认 false；Editor 的 p2:run 足以执行普通自检。
        self_check = client.post(
            "/api/v1/p2/self-check",
            json={"kb_id": kb_id},
        )
        assert self_check.status_code == 200, self_check.text
        assert self_check.json()["data"]["passed"] is True
        assert "persisted_demo_resource" not in self_check.json()["data"]

        # 同一路由一旦请求写控制库，就必须额外通过 p2:manage。
        denied = client.post(
            "/api/v1/p2/self-check",
            json={"kb_id": kb_id, "persist_demo_resources": True},
        )
        assert denied.status_code == 403, denied.text
        assert denied.json()["error"]["code"] == "FORBIDDEN"
        assert "p2:manage" in denied.json()["error"]["message"]

        listing = client.get(f"/api/v1/p2/resources?resource_type=memory_policy&kb_id={kb_id}")
        assert listing.status_code == 200, listing.text
        assert listing.json()["data"]["items"] == []


def test_p2_resource_scope_and_payload_are_server_validated(tmp_path: Path) -> None:
    database_path = tmp_path / "p2-scope.db"
    kb_id = _create_kb(database_path)
    configure_database(f"sqlite:///{database_path.as_posix()}")

    with TestClient(query_server.app) as client:
        _login(client)
        wrong_schema = _memory_policy(kb_id)
        wrong_schema["schema_version"] = "free-json.v1"
        response = client.post("/api/v1/p2/resources", json=wrong_schema)
        assert response.status_code == 422

        unsafe = _memory_policy(kb_id)
        unsafe["payload"]["max_context_tokens"] = 100_000
        response = client.post("/api/v1/p2/resources", json=unsafe)
        assert response.status_code == 422

        missing_outer_kb = _memory_policy(kb_id)
        missing_outer_kb.pop("kb_id")
        response = client.post("/api/v1/p2/resources", json=missing_outer_kb)
        assert response.status_code == 422

        tenant_resource_with_kb = {
            "resource_type": "model_route",
            "logical_id": "route_with_invalid_kb_scope",
            "version": "1.0.0",
            "schema_version": "model-routing.v1",
            "kb_id": kb_id,
            "payload": {},
        }
        response = client.post("/api/v1/p2/resources", json=tenant_resource_with_kb)
        assert response.status_code == 422

        kb_payload_without_outer_kb = {
            "resource_type": "metadata_schema",
            "logical_id": "schema_without_outer_kb",
            "version": "1.0.0",
            "schema_version": "metadata-schema.v1",
            "payload": {},
        }
        response = client.post("/api/v1/p2/resources", json=kb_payload_without_outer_kb)
        assert response.status_code == 422

        missing_kb = _memory_policy("kb_from_another_tenant")
        response = client.post("/api/v1/p2/resources", json=missing_kb)
        assert response.status_code == 404


def test_document_asset_inspection_validates_document_version_chain(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "p2-image-scope.db"
    kb_id = _create_kb(database_path)
    document_id, version_id = _create_document_version(database_path, kb_id)
    configure_database(f"sqlite:///{database_path.as_posix()}")
    params = {
        "source_kind": "document_asset",
        "kb_id": kb_id,
        "document_id": document_id,
        "document_version_id": version_id,
        "page_no": 1,
        "x0": 0,
        "y0": 0,
        "x1": 1,
        "y1": 1,
    }

    with TestClient(query_server.app) as client:
        _login(client)
        mismatched = client.post(
            "/api/v1/p2/multimodal/images:inspect",
            params={**params, "document_version_id": "dv_other_document"},
            files={"file": ("page.png", PNG_1X1, "image/png")},
        )
        assert mismatched.status_code == 404, mismatched.text

        accepted = client.post(
            "/api/v1/p2/multimodal/images:inspect",
            params=params,
            files={"file": ("page.png", PNG_1X1, "image/png")},
        )
        assert accepted.status_code == 200, accepted.text
        asset = accepted.json()["data"]["asset"]
        assert asset["kb_id"] == kb_id
        assert asset["document_id"] == document_id
        assert asset["document_version_id"] == version_id
        assert accepted.json()["data"]["bytes_persisted"] is False


def test_compliance_metadata_publish_returns_403_and_audits_denial(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "p2-compliance-http.db"
    kb_id = _create_kb(database_path)
    document_id, _ = _create_document_version(database_path, kb_id)
    configure_database(f"sqlite:///{database_path.as_posix()}")

    with TestClient(query_server.app) as client:
        _login(client)
        schema = client.post(
            "/api/v1/p2/resources",
            json=_compliance_metadata_schema(kb_id),
        )
        assert schema.status_code == 201, schema.text
        schema_id = schema.json()["data"]["id"]
        published_schema = client.post(
            f"/api/v1/p2/resources/{schema_id}:publish",
            json={"enable": False},
        )
        assert published_schema.status_code == 200, published_schema.text
        created_admin = client.post(
            "/api/v1/admin/users",
            json={
                "email": KB_ADMIN["email"],
                "password": KB_ADMIN["password"],
                "display_name": "P2 KB Admin",
                "role": "KBAdmin",
            },
        )
        assert created_admin.status_code == 200, created_admin.text

    with TestClient(query_server.app) as client:
        _login(client, KB_ADMIN)
        proposed = client.post(
            "/api/v1/p2/metadata-values",
            json={
                "schema_resource_id": schema_id,
                "target": {
                    "tenant_id": "tenant_default",
                    "kb_id": kb_id,
                    "scope": "document",
                    "document_id": document_id,
                },
                "field_key": "risk_level",
                "value": 3,
                "source": "manual",
            },
        )
        assert proposed.status_code == 201, proposed.text
        value_id = proposed.json()["data"]["id"]

        denied = client.post(f"/api/v1/p2/metadata-values/{value_id}:publish")
        assert denied.status_code == 403, denied.text
        assert denied.json()["error"]["code"] == "FORBIDDEN"
        assert "metadata:compliance" in denied.json()["error"]["message"]

        events = client.get("/api/v1/audit-events?limit=20")
        assert events.status_code == 200, events.text
        assert any(
            item["action"] == "metadata:compliance" and item["result"] == "denied"
            for item in events.json()["items"]
        )

    with TestClient(query_server.app) as client:
        _login(client)
        published = client.post(f"/api/v1/p2/metadata-values/{value_id}:publish")
        assert published.status_code == 200, published.text
        assert published.json()["data"]["status"] == "published"
