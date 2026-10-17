"""当前用户完整历史会话与 transcript 的授权/API 合同。"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import delete

import app.api.http.query_runtime_routes as runtime_routes
from app.domain.query_runtime import QueryRunStatus
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.models import (
    Base,
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.query_runtime_models import QueryMessageModel
from app.infra.persistence.sql.query_runtime_repository import QueryRuntimeRepository
from app.infra.persistence.sql.repositories import new_id
from app.security.principal import Principal
from app.security.http import action_for


TENANT_A = "tenant_history_a"
TENANT_B = "tenant_history_b"
USER_A = "user_history_a"
USER_B = "user_history_b"
KB_OWNER = "kb_history_owner"
KB_ACL = "kb_history_acl"
KB_REVOKED = "kb_history_revoked"
KB_DENIED = "kb_history_denied"
KB_OTHER_TENANT = "kb_history_other_tenant"


def _principal(
    *,
    tenant_id: str = TENANT_A,
    user_id: str = USER_A,
) -> Principal:
    return Principal(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=("Viewer",),
        scopes=("query:execute",),
        auth_type="password",
        credential_id=f"cred_{user_id}",
        request_id="req_history",
    )


def _app(principal_ref: dict[str, Principal]) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def inject_principal(request: Request, call_next):
        request.state.request_id = "req_history"
        request.state.principal = principal_ref["value"]
        return await call_next(request)

    runtime_routes.register_query_runtime_routes(app)
    return app


def _seed_control_plane() -> str:
    revoked_acl_id = new_id("acl")
    with session_scope() as session:
        session.add_all(
            [
                TenantModel(id=TENANT_A, name="History Tenant A", status="active"),
                TenantModel(id=TENANT_B, name="History Tenant B", status="active"),
                WorkspaceModel(
                    id="workspace_history_a",
                    tenant_id=TENANT_A,
                    name="History Workspace A",
                    status="active",
                ),
                WorkspaceModel(
                    id="workspace_history_b",
                    tenant_id=TENANT_B,
                    name="History Workspace B",
                    status="active",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                KnowledgeBaseModel(
                    id=KB_OWNER,
                    tenant_id=TENANT_A,
                    workspace_id="workspace_history_a",
                    name="History Owner KB",
                    status="active",
                    created_by=USER_A,
                ),
                KnowledgeBaseModel(
                    id=KB_ACL,
                    tenant_id=TENANT_A,
                    workspace_id="workspace_history_a",
                    name="History ACL KB",
                    status="active",
                    created_by="another_user",
                ),
                KnowledgeBaseModel(
                    id=KB_REVOKED,
                    tenant_id=TENANT_A,
                    workspace_id="workspace_history_a",
                    name="History Revoked KB",
                    status="active",
                    created_by="another_user",
                ),
                KnowledgeBaseModel(
                    id=KB_DENIED,
                    tenant_id=TENANT_A,
                    workspace_id="workspace_history_a",
                    name="History Denied KB",
                    status="active",
                    created_by="another_user",
                ),
                KnowledgeBaseModel(
                    id=KB_OTHER_TENANT,
                    tenant_id=TENANT_B,
                    workspace_id="workspace_history_b",
                    name="History Other Tenant KB",
                    status="active",
                    created_by=USER_A,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                KnowledgeBaseAclModel(
                    id=new_id("acl"),
                    kb_id=KB_ACL,
                    principal_type="user",
                    principal_id=USER_A,
                    role="reader",
                ),
                KnowledgeBaseAclModel(
                    id=revoked_acl_id,
                    kb_id=KB_REVOKED,
                    principal_type="user",
                    principal_id=USER_A,
                    role="reader",
                ),
            ]
        )
    return revoked_acl_id


def _advance_and_complete(
    repository: QueryRuntimeRepository,
    *,
    conversation_id: str,
    turn_number: int,
    metadata: dict | None = None,
) -> tuple[str, str]:
    _, run, _, _ = repository.begin_turn(
        conversation_id=conversation_id,
        client_turn_id=f"history-turn-{turn_number}",
        user_content=f"历史问题 {turn_number}",
    )
    for target in (
        QueryRunStatus.PLANNING,
        QueryRunStatus.RETRIEVING,
        QueryRunStatus.PACKING,
        QueryRunStatus.GENERATING,
        QueryRunStatus.VALIDATING,
    ):
        run = repository.transition_run(
            run.query_run_id,
            expected_revision=run.revision,
            target=target,
        )
    result = dict(metadata or {})
    repository.complete_run(
        run.query_run_id,
        expected_revision=run.revision,
        answer=f"历史答案 {turn_number}",
        result=result,
    )
    return run.turn_id, run.query_run_id


def test_history_directory_and_transcript_are_owner_and_entitlement_scoped(
    tmp_path,
) -> None:
    engine = configure_database(
        f"sqlite:///{(tmp_path / 'conversation-history.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    revoked_acl_id = _seed_control_plane()
    visible_ids: list[str] = []
    with session_scope() as session:
        repository = QueryRuntimeRepository(session, tenant_id=TENANT_A, user_id=USER_A)
        transcript = repository.create_conversation(
            kb_id=KB_OWNER,
            title="可回放的完整会话",
        )
        visible_ids.append(transcript.conversation_id)
        _advance_and_complete(repository, conversation_id=transcript.conversation_id, turn_number=1)
        last_turn_id, _ = _advance_and_complete(
            repository,
            conversation_id=transcript.conversation_id,
            turn_number=2,
            metadata={
                "citations": [{"evidence_id": "ev_history_1", "label": "E1"}],
                "image_urls": ["/api/v1/evidence/ev_history_1/images/0"],
                "answer_envelope": {"grounded": True},
                "sources": [{"title": "用户手册", "page": 7}],
                "prompt": "must-not-leak",
                "context_pack": {"raw": "must-not-leak"},
            },
        )
        now = datetime.now(UTC)
        session.add_all(
            [
                QueryMessageModel(
                    id=new_id("msg"),
                    tenant_id=TENANT_A,
                    conversation_id=transcript.conversation_id,
                    turn_id=last_turn_id,
                    query_run_id=None,
                    turn_seq=2,
                    role="system",
                    status="committed",
                    content="内部系统消息",
                    ordinal=2,
                    metadata_json={"prompt": "must-not-leak"},
                    created_at=now,
                    committed_at=now,
                ),
                QueryMessageModel(
                    id=new_id("msg"),
                    tenant_id=TENANT_A,
                    conversation_id=transcript.conversation_id,
                    turn_id=last_turn_id,
                    query_run_id=None,
                    turn_seq=2,
                    role="assistant",
                    status="partial",
                    content="未提交的流式片段",
                    ordinal=3,
                    metadata_json={},
                    created_at=now,
                    committed_at=None,
                ),
            ]
        )
        visible_ids.append(
            repository.create_conversation(kb_id=KB_OWNER, title="空 transcript").conversation_id
        )
        visible_ids.append(
            repository.create_conversation(kb_id=KB_ACL, title="ACL 会话").conversation_id
        )
        archived = repository.create_conversation(kb_id=KB_OWNER, title="已归档会话")
        archived = repository.archive_conversation(
            archived.conversation_id,
            expected_revision=archived.revision,
        )
        revoked = repository.create_conversation(kb_id=KB_REVOKED, title="稍后撤权")
        denied = repository.create_conversation(kb_id=KB_DENIED, title="从未授权")

        other_user = QueryRuntimeRepository(session, tenant_id=TENANT_A, user_id=USER_B)
        other_user_conversation = other_user.create_conversation(
            kb_id=KB_OWNER,
            title="其他用户会话",
        )
        other_tenant = QueryRuntimeRepository(session, tenant_id=TENANT_B, user_id=USER_A)
        other_tenant_conversation = other_tenant.create_conversation(
            kb_id=KB_OTHER_TENANT,
            title="其他租户会话",
        )

    # 会话产生后撤销 ACL，历史所有权不得继续充当 KB 读取授权。
    with session_scope() as session:
        session.execute(
            delete(KnowledgeBaseAclModel).where(KnowledgeBaseAclModel.id == revoked_acl_id)
        )

    principal_ref = {"value": _principal()}
    with TestClient(_app(principal_ref)) as client:
        directory = client.get("/api/v1/conversations", params={"limit": 200})
        assert directory.status_code == 200, directory.text
        listed_ids = [item["conversation_id"] for item in directory.json()["items"]]
        assert set(listed_ids) == set(visible_ids)
        assert revoked.conversation_id not in listed_ids
        assert denied.conversation_id not in listed_ids
        assert archived.conversation_id not in listed_ids
        assert other_user_conversation.conversation_id not in listed_ids
        assert other_tenant_conversation.conversation_id not in listed_ids

        # entitlement 先进入 SQL WHERE，再应用 offset/limit；逐页遍历不丢项或重复。
        paged_ids: list[str] = []
        for offset in range(0, len(listed_ids), 2):
            page = client.get(
                "/api/v1/conversations",
                params={"limit": 2, "offset": offset},
            )
            assert page.status_code == 200, page.text
            paged_ids.extend(item["conversation_id"] for item in page.json()["items"])
        assert paged_ids == listed_ids
        assert len(paged_ids) == len(set(paged_ids))

        archived_directory = client.get(
            "/api/v1/conversations",
            params={"limit": 200, "include_archived": True},
        )
        archived_ids = {
            item["conversation_id"] for item in archived_directory.json()["items"]
        }
        assert archived.conversation_id in archived_ids
        assert revoked.conversation_id not in archived_ids

        response = client.get(
            f"/api/v1/conversations/{transcript.conversation_id}/messages"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["schema_version"] == "conversation-transcript.v1"
        assert body["conversation"]["conversation_id"] == transcript.conversation_id
        assert [(item["role"], item["content"]) for item in body["items"]] == [
            ("user", "历史问题 1"),
            ("assistant", "历史答案 1"),
            ("user", "历史问题 2"),
            ("assistant", "历史答案 2"),
        ]
        assistant_metadata = body["items"][-1]["metadata"]
        assert assistant_metadata["citations"][0]["evidence_id"] == "ev_history_1"
        assert assistant_metadata["image_urls"] == [
            "/api/v1/evidence/ev_history_1/images/0"
        ]
        assert assistant_metadata["answer_envelope"] == {"grounded": True}
        assert assistant_metadata["sources"] == [{"title": "用户手册", "page": 7}]
        assert "prompt" not in assistant_metadata
        assert "context_pack" not in assistant_metadata
        assert all(item["status"] == "committed" for item in body["items"])
        assert all(item["role"] in {"user", "assistant"} for item in body["items"])

        empty = client.get(
            f"/api/v1/conversations/{visible_ids[1]}/messages"
        )
        assert empty.status_code == 200
        assert empty.json()["items"] == []

        for hidden_id in (
            revoked.conversation_id,
            denied.conversation_id,
            other_user_conversation.conversation_id,
            other_tenant_conversation.conversation_id,
            "conv_" + "f" * 32,
        ):
            hidden = client.get(f"/api/v1/conversations/{hidden_id}/messages")
            assert hidden.status_code == 404
            assert hidden.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"

        revoked_runs = client.get(
            f"/api/v1/conversations/{revoked.conversation_id}/query-runs"
        )
        assert revoked_runs.status_code == 404
        assert revoked_runs.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"

        principal_ref["value"] = _principal(user_id="user_without_history")
        no_history = client.get("/api/v1/conversations")
        assert no_history.status_code == 200
        assert no_history.json()["items"] == []

    dispose_database()


def test_history_entitlement_dependency_failure_is_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    engine = configure_database(
        f"sqlite:///{(tmp_path / 'history-authz-down.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    _seed_control_plane()
    with session_scope() as session:
        conversation = QueryRuntimeRepository(
            session,
            tenant_id=TENANT_A,
            user_id=USER_A,
        ).create_conversation(kb_id=KB_OWNER, title="授权依赖故障")

    def unavailable(*args, **kwargs):
        raise RuntimeError("authorization database unavailable")

    monkeypatch.setattr(runtime_routes.SqlEntitlementResolver, "resolve", unavailable)
    principal_ref = {"value": _principal()}
    with TestClient(_app(principal_ref)) as client:
        directory = client.get("/api/v1/conversations")
        assert directory.status_code == 503
        assert directory.json()["error"]["code"] == "ENTITLEMENT_UNAVAILABLE"
        transcript = client.get(
            f"/api/v1/conversations/{conversation.conversation_id}/messages"
        )
        assert transcript.status_code == 503
        assert transcript.json()["error"]["code"] == "ENTITLEMENT_UNAVAILABLE"

    dispose_database()


def test_todo12_history_search_lifecycle_cas_and_public_failed_turn(tmp_path) -> None:
    engine = configure_database(
        f"sqlite:///{(tmp_path / 'todo12-query-history.db').as_posix()}"
    )
    Base.metadata.create_all(bind=engine)
    _seed_control_plane()
    with session_scope() as session:
        repository = QueryRuntimeRepository(session, tenant_id=TENANT_A, user_id=USER_A)
        managed = repository.create_conversation(
            kb_id=KB_OWNER,
            title="HAK180 接地验证",
        )
        failed = repository.create_conversation(
            kb_id=KB_OWNER,
            title="HAK180 失败轮次",
        )
        _, run, _, _ = repository.begin_turn(
            conversation_id=failed.conversation_id,
            client_turn_id="todo12-failed-turn",
            user_content="使用 HAK180 之前需要做什么？",
        )
        repository.fail_run(
            run.query_run_id,
            expected_revision=run.revision,
            error_code="RETRIEVAL_TIMEOUT",
            safe_message="检索暂时超时，请稍后重试。",
            retryable=True,
            error_detail_ref="s3://must-not-leak/internal-trace",
        )

    assert action_for("GET", "/api/v1/conversations") == "history:read"
    assert (
        action_for("POST", f"/api/v1/conversations/{managed.conversation_id}:archive")
        == "history:delete"
    )
    assert (
        action_for("POST", f"/api/v1/conversations/{managed.conversation_id}:restore")
        == "history:delete"
    )
    assert (
        action_for("DELETE", f"/api/v1/conversations/{managed.conversation_id}")
        == "history:delete"
    )

    principal_ref = {"value": _principal()}
    with TestClient(_app(principal_ref)) as client:
        search = client.get(
            "/api/v1/conversations",
            params={"status": "active", "search": "接地"},
        )
        assert search.status_code == 200, search.text
        assert [item["conversation_id"] for item in search.json()["items"]] == [
            managed.conversation_id
        ]
        assert search.json()["status_filter"] == "active"
        assert search.json()["search"] == "接地"

        transcript = client.get(
            f"/api/v1/conversations/{failed.conversation_id}/messages"
        )
        assert transcript.status_code == 200, transcript.text
        body = transcript.json()
        assert body["items"] == []
        assert len(body["turns"]) == 1
        public_turn = body["turns"][0]
        assert public_turn["user_content"] == "使用 HAK180 之前需要做什么？"
        assert public_turn["query_run_status"] == "failed"
        assert public_turn["safe_error_message"] == "检索暂时超时，请稍后重试。"
        serialized = str(public_turn)
        for secret in ("input_hash", "history_snapshot", "trace_id", "s3://"):
            assert secret not in serialized

        principal_ref["value"] = _principal(user_id=USER_B)
        hidden = client.post(
            f"/api/v1/conversations/{managed.conversation_id}:archive",
            json={"expected_revision": 0},
        )
        assert hidden.status_code == 404

        principal_ref["value"] = _principal()
        archived = client.post(
            f"/api/v1/conversations/{managed.conversation_id}:archive",
            json={"expected_revision": 0},
        )
        assert archived.status_code == 200, archived.text
        assert archived.json()["conversation"]["status"] == "archived"
        assert archived.json()["conversation"]["revision"] == 1

        stale = client.post(
            f"/api/v1/conversations/{managed.conversation_id}:restore",
            json={"expected_revision": 0},
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "CONVERSATION_REVISION_CONFLICT"

        restored = client.post(
            f"/api/v1/conversations/{managed.conversation_id}:restore",
            json={"expected_revision": 1},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["conversation"]["status"] == "active"
        assert restored.json()["conversation"]["revision"] == 2

        archived_again = client.post(
            f"/api/v1/conversations/{managed.conversation_id}:archive",
            json={"expected_revision": 2},
        )
        assert archived_again.status_code == 200
        deleted = client.request(
            "DELETE",
            f"/api/v1/conversations/{managed.conversation_id}",
            json={"expected_revision": 3},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["deleted"] is True
        assert client.get(
            f"/api/v1/conversations/{managed.conversation_id}/messages"
        ).status_code == 404

    dispose_database()


def test_todo12_workbench_deep_links_and_inline_source_headers(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "联想海豚用户手册.pdf"
    source.write_bytes(b"%PDF-1.4\n% todo12\n")
    monkeypatch.setattr(
        runtime_routes,
        "_authorized_inline_source",
        lambda **_: (source, "application/pdf"),
    )
    principal_ref = {"value": _principal()}
    with TestClient(_app(principal_ref)) as client:
        for path in ("/workbench/chat", "/workbench/retrieval"):
            response = client.get(path)
            assert response.status_code == 200
            assert "可解释检索测试台" in response.text

        response = client.get(
            "/api/v1/evidence/ev_" + "2" * 40 + "/source-inline"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/pdf")
        assert response.headers["content-disposition"].startswith("inline;")
        assert response.headers["x-frame-options"] == "SAMEORIGIN"
        assert "object_key" not in response.text


def test_todo12_inline_source_requires_verified_immutable_original(
    tmp_path,
    monkeypatch,
) -> None:
    originals = tmp_path / "output" / "originals"
    originals.mkdir(parents=True)
    outside = tmp_path / ".env"
    outside_bytes = b"AUTH_TOKEN_SECRET=must-not-leak"
    outside.write_bytes(outside_bytes)
    source = originals / "联想海豚用户手册.pdf"
    source_bytes = b"%PDF-1.4\n% verified todo12 source\n"
    source.write_bytes(source_bytes)

    monkeypatch.setattr(runtime_routes, "ORIGINALS_ROOT", originals)
    monkeypatch.setattr(
        runtime_routes,
        "get_authorized_evidence",
        lambda _: {
            "source": {
                "source_type": "local",
                "tenant_id": TENANT_A,
                "kb_id": KB_OWNER,
                "document_version_id": "ver_todo12_inline",
            }
        },
    )
    monkeypatch.setattr(
        runtime_routes,
        "_conversation_kb_is_currently_entitled",
        lambda **_: True,
    )
    version_ref = {
        "value": {
            "object_key": str(outside),
            "size_bytes": len(outside_bytes),
            "content_sha256": sha256(outside_bytes).hexdigest(),
            "mime_type": "text/plain",
        }
    }
    monkeypatch.setattr(
        runtime_routes,
        "get_document_version",
        lambda _: version_ref["value"],
    )

    principal_ref = {"value": _principal()}
    evidence_url = "/api/v1/evidence/ev_" + "3" * 40 + "/source-inline"
    with TestClient(_app(principal_ref)) as client:
        outside_response = client.get(evidence_url)
        assert outside_response.status_code == 404
        assert outside_response.json()["error"]["code"] == (
            "EVIDENCE_SOURCE_UNAVAILABLE"
        )
        assert b"AUTH_TOKEN_SECRET" not in outside_response.content

        version_ref["value"] = {
            "object_key": str(source),
            "size_bytes": len(source_bytes),
            "content_sha256": sha256(source_bytes).hexdigest(),
            "mime_type": "application/pdf",
        }
        valid_response = client.get(evidence_url)
        assert valid_response.status_code == 200
        assert valid_response.content == source_bytes
        assert valid_response.headers["content-disposition"].startswith("inline;")

        source.write_bytes(b"X" + source_bytes[1:])
        tampered_response = client.get(evidence_url)
        assert tampered_response.status_code == 409
        assert tampered_response.json()["error"]["code"] == (
            "SOURCE_OBJECT_INTEGRITY_ERROR"
        )
