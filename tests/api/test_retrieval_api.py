"""TODO-09 HTTP route and authorization contract tests."""

from contextlib import contextmanager
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.http import retrieval_routes
from app.api.http.retrieval_routes import register_retrieval_routes
from app.domain.retrieval import RetrievalEngineError
from app.security.http import action_for


class _FakeRepository:
    def __init__(self):
        self.resolve_allow_default = None
        self.last_mutation = None
        self.last_get = None
        self.current = SimpleNamespace(
            id="rpv-current",
            profile_id="rpf-default",
            version=1,
            status="published",
        )

    @staticmethod
    def profile_dict(row):
        return {
            "id": row.id,
            "profile_id": row.profile_id,
            "version": row.version,
            "status": row.status,
            "config": {"routes": ["hybrid"], "candidate_k": 5, "top_k": 5},
            "config_hash": "a" * 64,
        }

    def resolve_for_kb(self, _kb_id, allow_default=True):
        self.resolve_allow_default = allow_default
        return self.current

    @staticmethod
    def control_plane_revision(_kb_id):
        return 7

    def list(self, **_kwargs):
        return [self.current]

    def get(self, profile_id, version, **kwargs):
        self.last_get = (profile_id, version, kwargs)
        return SimpleNamespace(
            id=f"rpv-{version}",
            profile_id=profile_id,
            version=version,
            status="draft",
        )

    def create_draft(self, kb_id, name, config, **kwargs):
        self.last_mutation = ("create", kb_id, name, config, kwargs)
        return SimpleNamespace(
            id="rpv-created",
            profile_id=kwargs.get("profile_id") or "rpf-created",
            version=2,
            status="draft",
        )

    def publish(self, kb_id, profile_id, version, **kwargs):
        self.last_mutation = ("publish", kb_id, profile_id, version, kwargs)
        return SimpleNamespace(
            id="rpv-published",
            profile_id=profile_id,
            version=version,
            status="published",
        )

    def rollback(self, kb_id, profile_id, version, **kwargs):
        self.last_mutation = ("rollback", kb_id, profile_id, version, kwargs)
        return SimpleNamespace(
            id="rpv-rolled-back",
            profile_id=profile_id,
            version=version,
            status="published",
        )


def _app(monkeypatch) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def inject_request_context(request: Request, call_next):
        request.state.request_id = "req-test"
        request.state.principal = SimpleNamespace(tenant_id="tenant-test", user_id="user-test")
        return await call_next(request)

    register_retrieval_routes(app)

    @contextmanager
    def fake_session_scope():
        yield object()

    repository = _FakeRepository()
    app.state.fake_retrieval_repository = repository
    monkeypatch.setattr(retrieval_routes, "session_scope", fake_session_scope)
    monkeypatch.setattr(retrieval_routes, "_repository", lambda _request, _session: repository)
    monkeypatch.setattr(
        retrieval_routes,
        "preview_retrieval_profile",
        lambda **kwargs: {
            "profile_id": kwargs["profile_record"]["profile_id"],
            "profile_version": kwargs["profile_record"]["version"],
            "routes": [{"route": "hybrid", "status": "succeeded"}],
        },
    )
    return app


def test_retrieval_profile_routes_and_rbac_actions(monkeypatch):
    app = _app(monkeypatch)
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if "retrieval-profile" in route.path
    }
    assert len(actual) == 6
    assert action_for("GET", "/api/v1/knowledge-bases/kb-1/retrieval-profiles") == "kb:read"
    assert action_for("POST", "/api/v1/knowledge-bases/kb-1/retrieval-profiles") == "kb:update"
    assert action_for("POST", "/api/v1/retrieval-profiles/rpf/versions/1:preview") == "kb:update"
    assert (
        action_for("POST", "/api/v1/knowledge-bases/kb-1:rollback-retrieval-profile") == "kb:update"
    )


def test_list_and_preview_return_profile_identity_and_route_diagnostics(monkeypatch):
    app = _app(monkeypatch)
    with TestClient(app) as client:
        listing = client.get("/api/v1/knowledge-bases/kb-1/retrieval-profiles")
        preview = client.post(
            "/api/v1/retrieval-profiles/rpf-experiment/versions/2:preview",
            json={"kb_id": "kb-1", "query": "HAK180 有哪些安全注意事项？"},
        )
    assert listing.status_code == 200
    assert app.state.fake_retrieval_repository.resolve_allow_default is False
    assert listing.json()["data"]["control_plane_revision"] == 7
    assert listing.json()["data"]["current"]["profile_id"] == "rpf-default"
    assert preview.status_code == 200
    assert preview.json()["data"]["profile_id"] == "rpf-experiment"
    assert preview.json()["data"]["profile_version"] == 2
    assert preview.json()["data"]["routes"][0]["route"] == "hybrid"


def test_get_profile_requires_and_forwards_knowledge_base_identity(monkeypatch):
    app = _app(monkeypatch)
    repository = app.state.fake_retrieval_repository
    with TestClient(app) as client:
        missing_scope = client.get("/api/v1/retrieval-profiles/rpf-experiment/versions/2")
        scoped = client.get(
            "/api/v1/retrieval-profiles/rpf-experiment/versions/2",
            params={"kb_id": "kb-1"},
        )

    assert missing_scope.status_code == 422
    assert scoped.status_code == 200
    assert scoped.json()["data"]["profile_id"] == "rpf-experiment"
    assert repository.last_get == (
        "rpf-experiment",
        2,
        {"kb_id": "kb-1"},
    )


def test_profile_mutation_routes_forward_revision_claims_and_accept_legacy_bodies(
    monkeypatch,
):
    app = _app(monkeypatch)
    repository = app.state.fake_retrieval_repository
    with TestClient(app) as client:
        legacy = client.post(
            "/api/v1/knowledge-bases/kb-1/retrieval-profiles",
            json={"name": "legacy", "config": {}, "profile_id": "rpf-default"},
        )
        assert legacy.status_code == 201, legacy.text
        assert repository.last_mutation[-1]["expected_control_plane_revision"] is None

        publish = client.post(
            "/api/v1/retrieval-profiles/rpf-default/versions/2:publish",
            json={
                "kb_id": "kb-1",
                "expected_current_profile_id": "rpf-default",
                "expected_current_version": 1,
                "expected_current_config_hash": "a" * 64,
                "expected_control_plane_revision": 7,
                "expected_max_version": 2,
            },
        )
        assert publish.status_code == 200, publish.text
        assert repository.last_mutation[-1]["expected_control_plane_revision"] == 7
        assert repository.last_mutation[-1]["expected_max_version"] == 2

        rollback = client.post(
            "/api/v1/knowledge-bases/kb-1:rollback-retrieval-profile",
            json={
                "profile_id": "rpf-default",
                "version": 1,
                "expected_current_profile_id": "rpf-default",
                "expected_current_version": 2,
                "expected_current_config_hash": "b" * 64,
                "expected_control_plane_revision": 8,
                "expected_max_version": 2,
            },
        )
        assert rollback.status_code == 200, rollback.text
        assert repository.last_mutation[-1]["expected_control_plane_revision"] == 8
        assert repository.last_mutation[-1]["expected_max_version"] == 2


def test_preview_rejects_unbounded_or_empty_query(monkeypatch):
    app = _app(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/retrieval-profiles/rpf/versions/1:preview",
            json={"kb_id": "kb-1", "query": ""},
        )
    assert response.status_code == 422


def test_preview_maps_missing_active_index_to_safe_business_conflict(monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(
        retrieval_routes,
        "preview_retrieval_profile",
        lambda **_kwargs: (_ for _ in ()).throw(
            RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "managed placement=secret_collection 缺少内部绑定",
                retryable=False,
                details={"collection": "secret_collection"},
            )
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/retrieval-profiles/rpf/versions/1:preview",
            json={"kb_id": "kb-unpublished", "query": "如何安装打印机？"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "request_id": "req-test",
        "error": {
            "code": "INDEX_NOT_READY",
            "message": (
                "当前知识库还没有可查询的已发布索引。请等待文档处理完成，确认索引已发布后再试。"
            ),
            "retryable": False,
        },
    }
    assert "secret_collection" not in response.text
