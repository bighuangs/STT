import importlib
from collections import Counter

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import inspect

from app.infra.health.service import HealthManager
from app.infra.persistence.sql.database import configure_database, initialize_control_database
from app.shared.config.settings_config import AppSettings, SettingsProvider


def _settings(**overrides) -> AppSettings:
    return AppSettings(_env_file=None, **overrides)


def test_config_rejects_invalid_numeric_and_production_defaults():
    with pytest.raises(ValidationError, match="INGESTION_HEARTBEAT_SECONDS"):
        _settings(ingestion_heartbeat_seconds=60, ingestion_lease_seconds=60)
    with pytest.raises(ValidationError, match="生产环境"):
        _settings(app_env="production")


def test_secret_repr_snapshot_and_config_version_never_contain_values():
    secret = "sk-super-secret-value-1234567890"  # secret-scan: allow-test
    password = "Admin-Password-Secret-2026!"  # secret-scan: allow-test
    first = _settings(openai_api_key_value=secret, auth_bootstrap_admin_password_value=password)
    rotated_same_version = _settings(
        openai_api_key_value="sk-another-secret-value-1234567890",  # secret-scan: allow-test
        auth_bootstrap_admin_password_value=password,
    )
    rotated_new_version = _settings(
        openai_api_key_value="sk-another-secret-value-1234567890",  # secret-scan: allow-test
        openai_api_key_version="vault-kid-2",
        auth_bootstrap_admin_password_value=password,
    )
    rendered = repr(first) + str(first.redacted_snapshot())
    assert secret not in rendered
    assert password not in rendered
    assert "**********" in repr(first)
    assert first.config_version == rotated_same_version.config_version
    assert first.config_version != rotated_new_version.config_version


def test_new_prefixed_environment_wins_without_logging_value(monkeypatch):
    old = "old-secret-value-with-at-least-32-chars"
    new = "new-secret-value-with-at-least-32-chars"
    monkeypatch.setenv("AUTH_TOKEN_SECRET", old)
    monkeypatch.setenv("STT_AUTH_TOKEN_SECRET", new)
    with pytest.warns(DeprecationWarning, match="STT_AUTH_TOKEN_SECRET") as warning:
        loaded = SettingsProvider().get()
    assert loaded.auth_token_secret == new
    assert old not in str(warning[0].message)
    assert new not in str(warning[0].message)


def test_legacy_provider_import_has_no_stdout_side_effect(capsys):
    import app.infra.config.providers as providers

    importlib.reload(providers)
    assert capsys.readouterr().out == ""


def test_health_checks_are_cached_and_optional_failure_only_degrades():
    calls = Counter()

    def database_probe():
        calls["database"] += 1
        return {"status": "healthy"}

    def optional_probe():
        calls["optional"] += 1
        raise RuntimeError("Authorization: Bearer should-not-leak")

    settings = _settings(
        import_required_dependencies_csv="control_database",
        health_cache_ttl_seconds=30,
    )
    manager = HealthManager(settings, {"control_database": database_probe, "optional": optional_probe})
    first = manager.check("import")
    second = manager.check("import")
    manager.close()

    assert first["ready"] is True and first["status"] == "degraded"
    assert second["cached"] is True
    assert calls == Counter({"database": 1, "optional": 1})
    assert "should-not-leak" not in str(first)
    assert "[REDACTED]" in first["dependencies"]["optional"]["message"]


def test_required_dependency_failure_returns_not_ready():
    settings = _settings(query_required_dependencies_csv="mongo")
    manager = HealthManager(settings, {"mongo": lambda: {"status": "unhealthy", "error_code": "AUTH_FAILED"}})
    result = manager.check("query", force=True)
    manager.close()
    assert result["ready"] is False
    assert result["status"] == "not_ready"
    assert result["required_failed"] == ["mongo"]


def test_health_http_protocol_and_admin_boundary(tmp_path, monkeypatch):
    from app.api.http import health_routes, import_server

    database_url = f"sqlite:///{(tmp_path / 'todo05-http.db').as_posix()}"
    configure_database(database_url)

    result = {
        "status": "ready", "ready": True, "role": "import", "cached": False,
        "config_version": "cfg_test", "checked_at": "2026-07-16T00:00:00+00:00", "latency_ms": 1,
        "required_failed": [], "optional_failed": [],
        "dependencies": {"control_database": {
            "status": "healthy", "required": True, "latency_ms": 1,
            "last_success_at": "2026-07-16T00:00:00+00:00",
        }},
    }

    class FakeHealth:
        def liveness(self, role):
            return {"status": "alive", "role": role, "config_version": "cfg_test"}

        def check(self, role, force=False, include_config=False):
            payload = dict(result)
            if include_config:
                payload["config"] = {"security": {"token_secret_version": "vault-kid-1"}}
            return payload

    monkeypatch.setattr(health_routes, "health_manager", FakeHealth())
    with TestClient(import_server.app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["ready"] is True
        assert client.get("/health/dependencies").status_code == 401
        login = client.post("/api/v1/auth/login", json={
            "email": "admin@stt.local", "password": "Stt-Admin-ChangeMe!", "tenant_id": None,
        })
        assert login.status_code == 200
        dependencies = client.get("/health/dependencies")
        assert dependencies.status_code == 200
        assert dependencies.json()["config_version"] == "cfg_test"


def test_migration_adds_config_trace_columns(tmp_path):
    engine = configure_database(f"sqlite:///{(tmp_path / 'todo05-migration.db').as_posix()}")
    initialize_control_database()
    inspector = inspect(engine)
    job_columns = {column["name"] for column in inspector.get_columns("ingestion_jobs")}
    session_columns = {column["name"] for column in inspector.get_columns("query_sessions")}
    assert "config_version" in job_columns
    assert "config_version" in session_columns


def test_log_redactor_removes_headers_and_known_secret_shapes():
    from app.shared.runtime.logger import redact_log_text

    rendered = redact_log_text("Authorization: Bearer abc.def.ghi X-API-Key=very-secret-key-value")
    assert "abc.def.ghi" not in rendered
    assert "very-secret-key-value" not in rendered
    assert rendered.count("[REDACTED]") >= 2


def test_console_logger_disables_traceback_local_variable_diagnostics(monkeypatch):
    from app.shared.runtime import logger as logger_module

    captured: list[dict] = []

    class FakeLogger:
        @staticmethod
        def remove():
            return None

        @staticmethod
        def add(sink, **kwargs):
            captured.append({"sink": sink, **kwargs})
            return len(captured)

    monkeypatch.setattr(logger_module, "_loguru_logger", FakeLogger())
    monkeypatch.setattr(logger_module, "LOG_CONSOLE_ENABLE", True)
    monkeypatch.setattr(logger_module, "LOG_FILE_ENABLE", False)

    logger_module.init_logger()

    assert len(captured) == 1
    assert captured[0]["backtrace"] is False
    assert captured[0]["diagnose"] is False


def test_auth_secret_dual_key_rotation_window(monkeypatch):
    from app.security import tokens

    old_secret = "old-signing-secret-with-at-least-32-characters"  # secret-scan: allow-test
    new_secret = "new-signing-secret-with-at-least-32-characters"  # secret-scan: allow-test
    old_settings = _settings(auth_token_secret_value=old_secret)
    monkeypatch.setattr(tokens, "settings", old_settings)
    token, _, _ = tokens.issue_token(
        user_id="usr_test", tenant_id="tenant_test", token_version=1, token_type="access"
    )

    rotating = _settings(
        auth_token_secret_value=new_secret,
        auth_previous_token_secret_value=old_secret,
        auth_previous_token_secret_version="vault-kid-old",
    )
    monkeypatch.setattr(tokens, "settings", rotating)
    assert tokens.decode_token(token, "access")["sub"] == "usr_test"
    assert len(tokens.token_digests(token)) == 2


def test_query_header_keeps_health_and_kb_controls_in_responsive_rows():
    from pathlib import Path

    html = Path("app/process/query/page/chat.html").read_text(encoding="utf-8")
    assert "topbar-main" in html
    assert "control-grid" in html
    assert "auth-controls" in html
    assert "query-controls" in html
    assert html.count('id="apiPill"') == 1
    assert html.count('id="kbIdInput"') == 1
    assert '<select class="stt-select" id="kbIdInput"' in html
    assert "查询服务未就绪" in html
