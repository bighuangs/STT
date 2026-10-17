"""上传传输文件名与知识库逻辑文件名的边界契约。"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from app.api.http import import_server
from app.application.control_plane_service import bootstrap_control_plane
from app.infra.persistence.sql.database import configure_database


ADMIN = {"email": "admin@stt.local", "password": "Stt-Admin-ChangeMe!"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/login", json=ADMIN)
    assert response.status_code == 200, response.text


def test_filename_normalization_preserves_chinese_and_blocks_unsafe_names() -> None:
    assert (
        import_server._safe_upload_filename("PantumP3500用户手册zh_CNV1.2.pdf")
        == "PantumP3500用户手册zh_CNV1.2.pdf"
    )
    assert import_server._safe_upload_filename(r"C:\fakepath\用户手册.pdf") == "用户手册.pdf"

    for value in ("../用户手册.pdf", r"C:\fakepath\用户手册.pdf", "坏\x01名字.pdf", ""):
        with pytest.raises(HTTPException) as captured:
            import_server._safe_upload_filename(value, logical_name=True)
        assert captured.value.status_code == 422


def test_document_upload_uses_explicit_logical_name_and_version_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """临时传输文件名不得覆盖首版逻辑名，后续版本也不得重命名文档。"""

    configure_database(f"sqlite:///{(tmp_path / 'filename-contract.db').as_posix()}")
    bootstrap_control_plane()
    originals_root = tmp_path / "originals"
    monkeypatch.setattr(import_server, "ORIGINALS_ROOT", originals_root)
    monkeypatch.setattr(import_server, "validate_startup", lambda _service: None)
    monkeypatch.setattr(import_server, "close_runtime_resources", lambda: None)

    logical_name = "PantumP3500用户手册zh_CNV1.2_1644316283788.md"
    with TestClient(import_server.app) as client:
        _login(client)
        created = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "中文逻辑文件名契约库"},
        )
        assert created.status_code == 201, created.text
        kb_id = created.json()["data"]["id"]

        first = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            files={"file": ("source.md", b"# Pantum P3500\n\nfirst", "text/markdown")},
            data={"parser_config": "{}", "logical_name": logical_name},
            headers={"Idempotency-Key": "filename-contract-first"},
        )
        assert first.status_code == 202, first.text
        document_id = first.json()["document_id"]
        assert client.get(f"/api/v1/documents/{document_id}").json()["data"]["logical_name"] == logical_name

        replay = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            files={"file": ("source.md", b"# Pantum P3500\n\nfirst", "text/markdown")},
            data={"parser_config": "{}", "logical_name": logical_name},
            headers={"Idempotency-Key": "filename-contract-first"},
        )
        assert replay.status_code == 202, replay.text
        assert replay.json()["idempotency_replayed"] is True
        assert replay.json()["document_id"] == document_id

        second = client.post(
            f"/api/v1/documents/{document_id}/versions",
            files={
                "file": (
                    "renamed-transport-v2.md",
                    b"# Pantum P3500\n\nsecond",
                    "text/markdown",
                )
            },
            data={"parser_config": "{}"},
        )
        assert second.status_code == 202, second.text
        assert second.json()["version_no"] == 2
        detail = client.get(f"/api/v1/documents/{document_id}").json()["data"]
        assert detail["logical_name"] == logical_name
        assert [version["version_no"] for version in detail["versions"]] == [2, 1]

        for unsafe_name in ("../越权.md", "坏\x01名字.md"):
            rejected = client.post(
                f"/api/v1/knowledge-bases/{kb_id}/documents",
                files={"file": ("source.md", b"unsafe", "text/markdown")},
                data={"parser_config": "{}", "logical_name": unsafe_name},
            )
            assert rejected.status_code == 422, rejected.text

        safe_transport = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            files={"file": ("../../transport.md", b"# safe", "text/markdown")},
            data={"parser_config": "{}"},
        )
        assert safe_transport.status_code == 202, safe_transport.text
        safe_document = client.get(
            f"/api/v1/documents/{safe_transport.json()['document_id']}"
        ).json()["data"]
        assert safe_document["logical_name"] == "transport.md"
        assert not (tmp_path / "transport.md").exists()


def test_import_page_sends_browser_file_name_as_transport_and_logical_name() -> None:
    html = (Path(import_server.PROJECT_ROOT) / "app/process/import_/page/import.html").read_text(
        encoding="utf-8"
    )
    assert "form.append('file', file, file.name);" in html
    assert "if (!documentId) form.append('logical_name', file.name);" in html

