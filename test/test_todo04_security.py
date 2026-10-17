from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from app.api.http import import_server
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.models import ApiCredentialModel, QuerySessionModel


ADMIN = {"email": "admin@stt.local", "password": "Stt-Admin-ChangeMe!"}


def _login(client: TestClient, email: str, password: str, tenant_id: str | None = None):
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password, "tenant_id": tenant_id})
    assert response.status_code == 200, response.text
    return response


def test_authentication_rbac_tenant_isolation_and_api_key(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'todo04.db').as_posix()}")
    with TestClient(import_server.app) as client:
        assert client.get("/api/v1/knowledge-bases").status_code == 401
        login = _login(client, **ADMIN)
        assert "HttpOnly" in login.headers.get("set-cookie", "")
        default_tenant = login.json()["principal"]["tenant_id"]

        kb_a = client.post("/api/v1/knowledge-bases", json={"name": "租户 A 私有知识库"})
        assert kb_a.status_code == 201
        kb_a_id = kb_a.json()["data"]["id"]

        viewer_password = "Viewer-Password-2026!"  # secret-scan: allow-test
        viewer = client.post("/api/v1/admin/users", json={
            "email": "viewer@stt.local", "password": viewer_password,
            "display_name": "只读验证用户", "role": "Viewer",
        })
        assert viewer.status_code == 200

        key_created = client.post("/api/v1/api-keys", json={"name": "test", "scopes": ["kb:read"]})
        assert key_created.status_code == 200
        key_value = key_created.json()["api_key"]
        key_id = key_created.json()["id"]
        with session_scope() as session:
            stored = session.scalar(select(ApiCredentialModel).where(ApiCredentialModel.id == key_id))
            assert stored is not None
            assert key_value not in stored.secret_hash
        assert client.get("/api/v1/auth/me", headers={"X-API-Key": key_value}).status_code == 200
        assert client.delete(f"/api/v1/api-keys/{key_id}").status_code == 200
        assert client.get("/api/v1/auth/me", headers={"X-API-Key": key_value}).status_code == 401

        tenant_b = client.post("/api/v1/admin/tenants", json={"name": "租户 B"})
        assert tenant_b.status_code == 200
        tenant_b_id = tenant_b.json()["id"]
        _login(client, ADMIN["email"], ADMIN["password"], tenant_b_id)
        kb_b = client.post("/api/v1/knowledge-bases", json={"name": "租户 B 私有知识库"})
        assert kb_b.status_code == 201
        kb_b_id = kb_b.json()["data"]["id"]

        _login(client, "viewer@stt.local", viewer_password, default_tenant)
        assert client.get(f"/api/v1/knowledge-bases/{kb_a_id}").status_code == 200
        assert client.post("/api/v1/knowledge-bases", json={"name": "越权创建"}).status_code == 403
        assert client.post(
            f"/api/v1/knowledge-bases/{kb_a_id}/documents",
            files={"file": ("forbidden.md", b"secret", "text/markdown")},
        ).status_code == 403
        # 跨租户资源统一伪装成不存在，避免枚举另一个租户的 ID。
        assert client.get(f"/api/v1/knowledge-bases/{kb_b_id}").status_code == 404


def test_kb_external_llm_policy_is_server_side_deny(tmp_path):
    from app.application.query_security_pipeline import (
        QueryEgressDenied,
        authorize_external_call,
    )
    from app.security.service import register_query_session
    from app.security.principal import Principal, set_current_principal, reset_current_principal

    configure_database(f"sqlite:///{(tmp_path / 'todo04-policy.db').as_posix()}")
    with TestClient(import_server.app) as client:
        login = _login(client, **ADMIN)
        principal_data = login.json()["principal"]
        created = client.post("/api/v1/knowledge-bases", json={
            "name": "敏感知识库", "data_classification": "restricted",
            "external_llm_allowed": False, "web_search_allowed": False,
        })
        kb_id = created.json()["data"]["id"]

    principal = Principal(
        user_id=principal_data["user_id"], tenant_id=principal_data["tenant_id"],
        roles=tuple(principal_data["roles"]), scopes=(), auth_type="test", credential_id="test",
    )
    token = set_current_principal(principal)
    try:
        session_policy = register_query_session(
            principal,
            "session-policy",
            kb_id,
        )
        # 禁止外发不能连本地检索会话也一并禁掉；真正的安全边界是在每一次
        # 外部调用前消费服务端冻结策略，并保持默认拒绝。
        assert session_policy["external_llm_allowed"] is False
        state = {
            **session_policy,
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
        }
        with pytest.raises(QueryEgressDenied) as denied:
            authorize_external_call(
                state,
                purpose="llm_generation",
                provider="configured-llm",
            )
        assert denied.value.reason_code == "EGRESS_DISABLED"
    finally:
        reset_current_principal(token)


def test_query_history_unknown_or_foreign_session_is_404_not_500(tmp_path):
    from app.api.http import query_server

    configure_database(f"sqlite:///{(tmp_path / 'todo04-query.db').as_posix()}")
    with TestClient(query_server.app) as client:
        _login(client, **ADMIN)
        response = client.get("/history/session-from-another-user")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"


def test_query_session_keeps_immutable_kb_binding_and_returns_stable_conflict_code(tmp_path):
    from app.api.http import query_server
    from app.security.principal import Principal, reset_current_principal, set_current_principal
    from app.security.service import SessionKnowledgeBaseConflict, register_query_session

    configure_database(f"sqlite:///{(tmp_path / 'todo04-session-kb.db').as_posix()}")
    with TestClient(import_server.app) as client:
        login = _login(client, **ADMIN)
        principal_data = login.json()["principal"]
        first = client.post("/api/v1/knowledge-bases", json={"name": "会话知识库 A"})
        second = client.post("/api/v1/knowledge-bases", json={"name": "会话知识库 B"})
        assert first.status_code == 201
        assert second.status_code == 201
        kb_a = first.json()["data"]["id"]
        kb_b = second.json()["data"]["id"]

    principal = Principal(
        user_id=principal_data["user_id"],
        tenant_id=principal_data["tenant_id"],
        roles=tuple(principal_data["roles"]),
        scopes=(),
        auth_type="test",
        credential_id="test",
    )
    token = set_current_principal(principal)
    try:
        register_query_session(principal, "session-fixed-kb", kb_a)
        register_query_session(principal, "session-fixed-kb", kb_a)
        with pytest.raises(SessionKnowledgeBaseConflict):
            register_query_session(principal, "session-fixed-kb", kb_b)
        register_query_session(principal, "session-for-kb-b", kb_b)
    finally:
        reset_current_principal(token)

    with session_scope() as session:
        fixed = session.get(QuerySessionModel, "session-fixed-kb")
        switched = session.get(QuerySessionModel, "session-for-kb-b")
        assert fixed is not None and fixed.kb_id == kb_a
        assert switched is not None and switched.kb_id == kb_b

    # HTTP 合同提供稳定错误码；前端据此仅换新会话重试一次，不依赖中文文案。
    with TestClient(query_server.app) as client:
        _login(client, **ADMIN)
        conflict = client.post(
            "/query",
            json={
                "query": "在使用 HAK180 之前需要做什么？",
                "session_id": "session-fixed-kb",
                "kb_id": kb_b,
                "is_stream": True,
            },
        )
        assert conflict.status_code == 403
        assert conflict.json()["error"]["code"] == "SESSION_KB_CONFLICT"


def test_query_page_scopes_sessions_by_principal_and_kb():
    html = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "process"
        / "query"
        / "page"
        / "chat.html"
    ).read_text(encoding="utf-8")

    assert "stt.query.sessions.v1" in html
    assert "JSON.stringify([currentPrincipal.tenant_id, currentPrincipal.user_id, kbId])" in html
    assert "activateSessionForCurrentKb({forceNew: true})" in html
    assert "code === 'SESSION_KB_CONFLICT' && allowSessionRecovery" in html
    assert "localStorage.removeItem(LEGACY_QUERY_SESSION_KEY)" in html
    assert "kbIdInput.addEventListener('change', async () =>" in html
