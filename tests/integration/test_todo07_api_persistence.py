"""TODO-07 HTTP、pipeline 版本与解析制品持久化集成验收。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.api.http import import_server
from app.application import document_parsing_service
from app.application.control_plane_service import (
    bootstrap_control_plane,
    create_knowledge_base,
    get_document,
    get_document_version,
    register_uploaded_document,
)
from app.application.document_parsing_service import apply_parsing_to_graph_state
from app.application.document_parsing_service import resolved_parser_config_snapshot
from app.application.parse_artifact_store import build_parse_cache_key
from app.infra.persistence.sql.database import configure_database


ADMIN = {"email": "admin@stt.local", "password": "Stt-Admin-ChangeMe!"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/login", json=ADMIN)
    assert response.status_code == 200, response.text


def _configure_control_database(path: Path) -> None:
    configure_database(f"sqlite:///{path.as_posix()}")
    bootstrap_control_plane()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_parsing_capabilities_preview_cache_and_stable_corrupt_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """HTTP 验证全程离线，并验证失败上传不会留下文档事实。"""

    configure_database(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    originals_root = tmp_path / "originals"
    artifact_root = tmp_path / "parsed-artifacts"
    monkeypatch.setattr(import_server, "ORIGINALS_ROOT", originals_root)
    monkeypatch.setattr(document_parsing_service, "PARSE_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(import_server, "validate_startup", lambda _service: None)
    monkeypatch.setattr(import_server, "close_runtime_resources", lambda: None)

    with TestClient(import_server.app) as client:
        assert client.get("/api/v1/parsing/capabilities").status_code == 401
        _login(client)

        capabilities_response = client.get("/api/v1/parsing/capabilities")
        assert capabilities_response.status_code == 200, capabilities_response.text
        capabilities = capabilities_response.json()["data"]
        assert capabilities["parsed_schema_version"] == "parsed-document.v1"
        supported_formats = {
            format_name
            for parser in capabilities["parsers"]
            for format_name in parser["formats"]
        }
        assert {
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "markdown",
            "html",
            "csv",
            "txt",
        } <= supported_formats
        assert {
            template["template_id"]
            for template in capabilities["chunking_templates"]
        } == {"general", "manual", "paper", "qa", "table", "presentation"}

        markdown = (
            "# 安全说明\n\n"
            "## 接地要求\n\n"
            "设备必须可靠接地。\n\n"
            "- 断电后维护\n"
            "- 使用合适的防护设备\n"
        ).encode("utf-8")
        preview_form = {
            "parser_config": json.dumps(
                {
                    "chunking_template": "manual",
                    "chunk_target_tokens": 96,
                    "persist_parsed_artifact_to_minio": False,
                }
            ),
            "max_blocks": "20",
        }
        first = client.post(
            "/api/v1/parsing/preview",
            files={"file": ("safety.md", markdown, "text/markdown")},
            data=preview_form,
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        assert first_payload["detected_format"] == "markdown"
        assert first_payload["parser_id"] == "local.markdown"
        assert first_payload["chunking_template_id"] == "manual"
        assert first_payload["chunking_template_version"] == "1.1.1"
        assert first_payload["parsed_schema_version"] == "parsed-document.v1"
        assert first_payload["total_blocks"] >= 4
        assert first_payload["total_chunks"] >= 1
        assert first_payload["artifact"]["cache_hit"] is False
        assert first_payload["chunks"][0]["provenance"]["source_block_ids"]
        assert first_payload["chunks"][0]["provenance"]["locations"]
        assert first_payload["chunks"][0]["provenance"]["parser_id"] == "local.markdown"
        assert first_payload["chunks"][0]["provenance"]["template_id"] == "manual"

        second = client.post(
            "/api/v1/parsing/preview",
            files={"file": ("safety.md", markdown, "text/markdown")},
            data=preview_form,
        )
        assert second.status_code == 200, second.text
        second_payload = second.json()
        assert second_payload["artifact"]["cache_hit"] is True
        assert second_payload["artifact"]["cache_key"] == first_payload["artifact"]["cache_key"]
        assert second_payload["artifact"]["sha256"] == first_payload["artifact"]["sha256"]
        assert second_payload["chunks"] == first_payload["chunks"]

        disguised = client.post(
            "/api/v1/parsing/preview",
            files={
                "file": (
                    "not-really.pdf",
                    b"# Markdown heading\n\n- first\n- second\n\n[reference](manual.md)",
                    "application/pdf",
                )
            },
            data={"parser_config": "{}", "max_blocks": "20"},
        )
        assert disguised.status_code == 200, disguised.text
        disguised_payload = disguised.json()
        assert disguised_payload["detected_format"] == "markdown"
        warning_codes = {warning["code"] for warning in disguised_payload["warnings"]}
        assert {"PARSE_EXTENSION_MISMATCH", "PARSE_MIME_MISMATCH"} <= warning_codes

        kb_response = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "TODO-07 损坏文件验收库"},
        )
        assert kb_response.status_code == 201, kb_response.text
        kb_id = kb_response.json()["data"]["id"]
        corrupt = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            files={"file": ("corrupt.docx", b"PK\x03\x04not-a-zip", "application/zip")},
            data={"parser_config": "{}"},
        )
        assert corrupt.status_code == 422, corrupt.text
        assert corrupt.json()["error"] == {
            "code": "PARSE_CORRUPT_ARCHIVE",
            "message": "文件具有 ZIP 签名但不是可读取的 Office 文档。",
            "retryable": False,
        }
        listed = client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
        assert listed.status_code == 200
        assert listed.json()["data"] == []
        assert not originals_root.exists() or not any(
            path.is_file() for path in originals_root.rglob("*")
        )


def test_pipeline_fingerprint_distinguishes_config_and_is_order_independent(
    tmp_path: Path,
) -> None:
    """相同字节的版本身份由规范化 pipeline 配置共同决定。"""

    _configure_control_database(tmp_path / "fingerprint.db")
    kb_id = create_knowledge_base("TODO-07 pipeline 指纹库")["id"]
    content = b"# Same bytes\n\nThe parser configuration changes the result.\n"
    source = tmp_path / "same.md"
    source.write_bytes(content)
    common = {
        "kb_id": kb_id,
        "logical_name": source.name,
        "content_sha256": _sha256(content),
        "size_bytes": len(content),
        "mime_type": "text/markdown",
        "object_key": str(source),
    }
    config_a = {
        "parser_id": "local.markdown",
        "chunking_template": "general",
        "chunk": {"target_tokens": 128, "overlap_tokens": 16},
    }
    config_same_different_order = {
        "chunk": {"overlap_tokens": 16, "target_tokens": 128},
        "chunking_template": "general",
        "parser_id": "local.markdown",
    }
    config_b = {
        "parser_id": "local.markdown",
        "chunking_template": "manual",
        "chunk": {"target_tokens": 128, "overlap_tokens": 16},
    }

    first = register_uploaded_document(**common, parser_config_snapshot=config_a)
    replay = register_uploaded_document(
        **common,
        parser_config_snapshot=config_same_different_order,
    )
    changed = register_uploaded_document(**common, parser_config_snapshot=config_b)

    first_version = first["document_version"]
    replay_version = replay["document_version"]
    changed_version = changed["document_version"]
    assert replay["deduplicated"] is True
    assert replay["reused_job"] is True
    assert replay_version["id"] == first_version["id"]
    assert replay_version["pipeline_fingerprint"] == first_version["pipeline_fingerprint"]
    assert changed["deduplicated"] is False
    assert changed_version["id"] != first_version["id"]
    assert changed_version["version_no"] == 2
    assert changed_version["pipeline_fingerprint"] != first_version["pipeline_fingerprint"]

    document = get_document(first["document"]["id"])
    assert sorted(version["version_no"] for version in document["versions"]) == [1, 2]
    assert len({version["pipeline_fingerprint"] for version in document["versions"]}) == 2


def test_pdf_control_normalization_reaches_preview_and_import_version_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """PDF 预览不泄漏控制码，正式上传也不能复用 1.1.2 pipeline。"""

    configure_database(f"sqlite:///{(tmp_path / 'pdf-controls.db').as_posix()}")
    originals_root = tmp_path / "originals"
    artifact_root = tmp_path / "parsed-artifacts"
    monkeypatch.setattr(import_server, "ORIGINALS_ROOT", originals_root)
    monkeypatch.setattr(document_parsing_service, "PARSE_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(import_server, "validate_startup", lambda _service: None)
    monkeypatch.setattr(import_server, "close_runtime_resources", lambda: None)

    pdf_buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()
    source = tmp_path / "PantumP3500.pdf"
    source.write_bytes(pdf_bytes)

    extracted = tmp_path / "mineru-result"
    extracted.mkdir()
    markdown = extracted / "pantum.md"
    markdown.write_text("# 纸张规格", encoding="utf-8")
    content_list = extracted / "pantum_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "80 g/m²\u0001普通纸，最多 250 张。\n保留\t结构。",
                    "page_idx": 67,
                    "bbox": [100, 100, 900, 200],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {
            **state,
            "md_path": str(markdown),
            "mineru_content_list_path": str(content_list),
        },
    )

    requested_config = {
        "chunking_template": "manual",
        "persist_parsed_artifact_to_minio": False,
    }
    resolved, profile = resolved_parser_config_snapshot(
        source,
        declared_mime="application/pdf",
        parser_config=requested_config,
    )
    assert profile.parser_version == "1.1.3"
    old_resolved = {
        **resolved,
        "parser_version": "1.1.2",
        "chunking_template_version": "1.1.0",
    }
    digest = _sha256(pdf_bytes)

    with TestClient(import_server.app) as client:
        _login(client)
        preview = client.post(
            "/api/v1/parsing/preview",
            files={"file": (source.name, pdf_bytes, "application/pdf")},
            data={
                "parser_config": json.dumps(requested_config),
                "max_blocks": "20",
            },
        )
        assert preview.status_code == 200, preview.text
        preview_payload = preview.json()
        assert preview_payload["parser_version"] == "1.1.3"
        assert preview_payload["chunking_template_version"] == "1.1.1"
        assert preview_payload["blocks"][0]["text"] == (
            "80 g/m²普通纸，最多 250 张。\n保留\t结构。"
        )
        assert "\u0001" not in json.dumps(
            preview_payload["chunks"], ensure_ascii=False
        )
        assert preview_payload["chunks"][0]["provenance"]["parser_version"] == (
            "1.1.3"
        )

        kb_response = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "PDF 控制字符 pipeline 验收库"},
        )
        assert kb_response.status_code == 201, kb_response.text
        kb_id = kb_response.json()["data"]["id"]
        legacy = register_uploaded_document(
            kb_id=kb_id,
            logical_name=source.name,
            content_sha256=digest,
            size_bytes=len(pdf_bytes),
            mime_type="application/pdf",
            object_key=str(source),
            parser_config_snapshot=old_resolved,
        )
        uploaded = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            files={"file": (source.name, pdf_bytes, "application/pdf")},
            data={"parser_config": json.dumps(requested_config)},
        )
        assert uploaded.status_code == 202, uploaded.text
        upload_payload = uploaded.json()
        assert upload_payload["parser_version"] == "1.1.3"
        assert upload_payload["chunking_template_version"] == "1.1.1"
        assert upload_payload["version_no"] == 2
        assert upload_payload["deduplicated"] is False

        detail = get_document(upload_payload["document_id"])
        versions = sorted(detail["versions"], key=lambda item: item["version_no"])
        assert versions[0]["id"] == legacy["document_version"]["id"]
        assert versions[0]["parser_config_snapshot"]["parser_version"] == "1.1.2"
        assert versions[0]["parser_config_snapshot"]["chunking_template_version"] == (
            "1.1.0"
        )
        assert versions[1]["parser_config_snapshot"]["parser_version"] == "1.1.3"
        assert versions[1]["parser_config_snapshot"]["chunking_template_version"] == (
            "1.1.1"
        )
        assert versions[0]["pipeline_fingerprint"] != versions[1]["pipeline_fingerprint"]

    common_cache = {
        "input_sha256": digest,
        "parser_id": "mineru.pdf",
        "parser_config": requested_config,
    }
    assert build_parse_cache_key(
        **common_cache,
        parser_version="1.1.2",
    ) != build_parse_cache_key(
        **common_cache,
        parser_version="1.1.3",
    )


def test_graph_parsing_records_idempotent_parse_artifact_and_chunking_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Worker 解析完成后，控制面能追溯解析制品、切块运行与块级来源。"""

    _configure_control_database(tmp_path / "artifacts.db")
    monkeypatch.setattr(
        document_parsing_service,
        "PARSE_ARTIFACT_ROOT",
        tmp_path / "parsed-artifacts",
    )
    kb_id = create_knowledge_base("TODO-07 制品审计库")["id"]
    source = tmp_path / "manual.md"
    content = (
        "# HAK180 产品安全手册\n\n"
        "## 危险电压\n\n"
        "断电后等待五分钟再打开设备。\n\n"
        "## 保护接地\n\n"
        "通电前必须连接保护地。\n"
    ).encode("utf-8")
    source.write_bytes(content)
    parser_config = {
        "chunking_template": "manual",
        "chunk_target_tokens": 96,
        "chunk_overlap_tokens": 12,
        "persist_parsed_artifact": True,
        "persist_parsed_artifact_to_minio": False,
    }
    registered = register_uploaded_document(
        kb_id=kb_id,
        logical_name=source.name,
        content_sha256=_sha256(content),
        size_bytes=len(content),
        mime_type="text/markdown",
        object_key=str(source),
        parser_config_snapshot=parser_config,
    )
    version_id = registered["document_version"]["id"]
    state = {
        "local_file_path": str(source),
        "document_version_id": version_id,
        "parser_config": parser_config,
    }

    first_state = apply_parsing_to_graph_state(dict(state))
    second_state = apply_parsing_to_graph_state(dict(state))
    assert first_state["structured_chunks_ready"] is True
    assert first_state["parsed_artifact_cache_hit"] is False
    assert second_state["parsed_artifact_cache_hit"] is True
    assert first_state["chunks"] == second_state["chunks"]
    assert first_state["chunks"][0]["provenance"]["source_block_ids"]
    assert first_state["chunks"][0]["provenance"]["locations"]
    assert first_state["chunks"][0]["provenance"]["template_id"] == "manual"

    detail = get_document_version(version_id)
    assert len(detail["parse_artifacts"]) == 1
    assert len(detail["chunking_runs"]) == 1
    artifact = detail["parse_artifacts"][0]
    run = detail["chunking_runs"][0]
    assert artifact["id"] == first_state["parsed_document_summary"]["parse_artifact_id"]
    assert artifact["artifact_uri"].startswith("parse-cache://")
    assert artifact["input_sha256"] == _sha256(content)
    assert artifact["parser_id"] == "local.markdown"
    assert artifact["schema_version"] == "parsed-document.v1"
    assert artifact["metrics"]["cache_hit"] is True
    assert artifact["chunking_runs"] == [run]
    assert run["id"] == first_state["parsed_document_summary"]["chunking_run_id"]
    assert run["parse_artifact_id"] == artifact["id"]
    assert run["template_id"] == "manual"
    assert run["chunk_count"] == len(first_state["chunks"])
    assert Path(run["artifact_uri"]).is_file()
