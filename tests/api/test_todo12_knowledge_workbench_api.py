from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.http import import_server
from app.application import control_plane_service as control
from app.application import knowledge_workbench_service as workbench
from app.application.parse_artifact_store import (
    LocalParseArtifactStore,
    build_parse_cache_key,
    canonical_json,
)
from app.domain.parsing import ParsedBlock, ParsedDocument, SourceLocation
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.models import (
    DocumentChunkingRunModel,
    DocumentVersionModel,
    IndexBuildManifestModel,
    IngestionJobEventModel,
    IngestionJobModel,
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.repositories import DEFAULT_WORKSPACE_ID, new_id
from app.security.signed_cursor import decode_signed_cursor, encode_signed_cursor


ADMIN = {
    "email": "admin@stt.local",
    "password": "Stt-Admin-ChangeMe!",  # secret-scan: allow-test
    "tenant_id": "tenant_default",
}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/login", json=ADMIN)
    assert response.status_code == 200, response.text


def _login_as(
    client: TestClient, *, email: str, password: str, tenant_id: str
) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password, "tenant_id": tenant_id},
    )
    assert response.status_code == 200, response.text


def _create_kb(client: TestClient, name: str) -> str:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": name, "description": "TODO-12 workbench fixture"},
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for item in value.values()
            for key in _all_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _tamper_cursor(value: str) -> str:
    payload, signature = value.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    return f"{payload}.{replacement}{signature[1:]}"


def _seed_document(
    tmp_path: Path,
    monkeypatch,
    *,
    kb_id: str,
) -> dict:
    originals = tmp_path / "originals"
    artifacts = tmp_path / "parsed-artifacts"
    originals.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workbench, "ORIGINALS_ROOT", originals)
    monkeypatch.setattr(workbench, "PARSE_ARTIFACT_ROOT", artifacts)

    source = b"%PDF-TODO12-HAK180-workbench-fixture"
    source_path = originals / "hak180-product-safety.pdf"
    source_path.write_bytes(source)
    source_hash = sha256(source).hexdigest()
    registered = control.register_uploaded_document(
        kb_id=kb_id,
        logical_name="hak180产品安全手册.pdf",
        content_sha256=source_hash,
        size_bytes=len(source),
        mime_type="application/pdf",
        object_key=str(source_path),
        parser_config_snapshot={
            "parser_id": "todo12-fixture",
            "parser_version": "1.0.0",
            "template_id": "manual",
            "template_version": "1.0.0",
            "provider": {
                "api_token": "must-never-cross-the-workbench-boundary",
                "region": "cn-test",
            },
        },
    )
    with session_scope() as session:
        initial_event = session.scalar(
            select(IngestionJobEventModel).where(
                IngestionJobEventModel.job_id
                == registered["ingestion_job"]["id"]
            )
        )
        assert initial_event is not None
        initial_event.payload = {
            "nested": {
                "artifact": {
                    "sha256": "safe-public-integrity-fact",
                    "uri": "s3://private/object",
                },
                "bucket": "private-bucket",
                "object_key": "private/path",
                "path": "C:/private/source.pdf",
                "safe": "visible",
            },
            "lease_token": "must-never-leak",
        }
        job = session.get(IngestionJobModel, registered["ingestion_job"]["id"])
        assert job is not None
        job.error_category = "system"
        job.error_code = "TODO12_FIXTURE_FAILURE"
        job.error_detail = (
            r"worker failed at C:\private\stt\source.pdf; "
            "token=private-worker-token; object_key=private/object"
        )
        session.add(
            IndexBuildManifestModel(
                id=new_id("manifest"),
                job_id=job.id,
                document_version_id=registered["document_version"]["id"],
                index_version_id=registered["index_version"]["id"],
                input_sha256=source_hash,
                config_sha256=sha256(b"todo12-private-config").hexdigest(),
                expected_chunk_count=2,
                written_chunk_count=1,
                searchable_chunk_count=1,
                chunk_ids_sha256=sha256(b"chunk_1").hexdigest(),
                chunk_ids=["chunk_1"],
                object_keys=["private/index/object"],
                vector_dim=2,
                collection_name="private_collection",
                validation_status="failed",
                validation_errors=[
                    {
                        "code": "VECTOR_COUNT_MISMATCH",
                        "safe": "公开一致性校验摘要",
                        "path": r"C:\private\index\manifest.json",
                        "nested": {
                            "object_key": "private/index/object",
                            "source_uri": "s3://private/source",
                            "token": "private-manifest-token",  # secret-scan: allow-test
                        },
                    }
                ],
            )
        )
    version_id = registered["document_version"]["id"]
    parsed = ParsedDocument(
        parser_id="todo12-fixture",
        parser_version="1.0.0",
        source_sha256=source_hash,
        source_format="pdf",
        source_mime_type="application/pdf",
        blocks=[
            ParsedBlock(
                block_id="block_1",
                type="heading",
                text="在使用 HAK180 之前需要做什么？",
                location=SourceLocation(
                    page_no=1,
                    bbox=[0.1, 0.1, 0.9, 0.2],
                    coordinate_space="normalized_page",
                ),
                structured_data={
                    "safe": "公开结构化数据",
                    "file_path": r"C:\private\parsed\block.json",
                    "auth_token": "private-block-token",
                    "nested": {
                        "object_key": "private/block/object",
                        "source_uri": "s3://private/block",
                    },
                },
                order=0,
            ),
            ParsedBlock(
                block_id="block_2",
                type="text",
                text="使用前应阅读安全说明并检查设备和电源。",
                location=SourceLocation(
                    page_no=1,
                    bbox=[0.1, 0.25, 0.9, 0.5],
                    coordinate_space="normalized_page",
                ),
                parent_heading_path=["使用前检查"],
                order=1,
            ),
        ],
    )
    cache_key = build_parse_cache_key(
        input_sha256=source_hash,
        parser_id=parsed.parser_id,
        parser_version=parsed.parser_version,
        parser_config={},
    )
    parse_info = LocalParseArtifactStore(artifacts).save(cache_key, parsed)
    parse_record = control.record_document_parse_artifact(
        document_version_id=version_id,
        input_sha256=source_hash,
        parser_id=parsed.parser_id,
        parser_version=parsed.parser_version,
        schema_version=parsed.schema_version,
        status="succeeded",
        artifact_uri=f"parse-cache://{cache_key}",
        artifact_sha256=parse_info.sha256,
        artifact_size=parse_info.size_bytes,
        warnings=[{"code": "FIXTURE_WARNING", "message": "safe warning"}],
        metrics={"block_count": 2, "private_metric": "must not leak"},
    )
    chunks = [
        {
            "chunk_id": "chunk_1",
            "content": "在使用 HAK180 之前需要阅读安全说明。",
            "type": "text",
            "source_block_ids": ["block_1"],
            "location": [{"page_no": 1, "bbox": [0.1, 0.1, 0.9, 0.2]}],
            "metadata": {
                "section": "使用前检查",
                "object_key": "private/chunk/path",
                "nested": {"access_token": "must-never-leak", "safe": "visible"},
            },
            "dense_vector": [0.1, 0.2],
        },
        {
            "chunk_id": "chunk_2",
            "content": "检查设备外观和电源连接。",
            "type": "text",
            "source_block_ids": ["block_2"],
            "location": [{"page_no": 1, "bbox": [0.1, 0.25, 0.9, 0.5]}],
            "metadata": {"section": "使用前检查"},
            "sparse_vector": {"1": 0.3},
        },
    ]
    chunk_payload = canonical_json(chunks)
    chunk_path = artifacts / "chunks" / f"{version_id}.json"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(chunk_payload)
    chunk_record = control.record_document_chunking_run(
        parse_artifact_id=parse_record["id"],
        template_id="manual",
        template_version="1.0.0",
        config_hash=sha256(b"todo12-chunk-config").hexdigest(),
        status="succeeded",
        artifact_uri=str(chunk_path),
        artifact_sha256=sha256(chunk_payload).hexdigest(),
        artifact_size=len(chunk_payload),
        chunk_count=len(chunks),
    )
    return {
        "registered": registered,
        "source": source,
        "source_path": source_path,
        "source_hash": source_hash,
        "parsed": parsed,
        "parse_record": parse_record,
        "chunks": chunks,
        "chunk_payload": chunk_payload,
        "chunk_record": chunk_record,
        "artifacts": artifacts,
    }


def test_workbench_cursor_etag_range_and_safe_aggregate_views(
    tmp_path: Path, monkeypatch
) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo12-workbench-api.db').as_posix()}")
    with TestClient(import_server.app) as client:
        _login(client)
        kb_ids = [
            _create_kb(client, f"TODO12 Cursor KB {index}") for index in range(3)
        ]
        first = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={"q": "TODO12 Cursor", "sort": "name_asc", "limit": 2},
        )
        assert first.status_code == 200, first.text
        first_page = first.json()["data"]
        assert first_page["schema_version"] == "workbench-page.v1"
        assert first_page["total"] == 3
        assert len(first_page["items"]) == 2
        assert first_page["next_cursor"]
        second = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={
                "q": "TODO12 Cursor",
                "sort": "name_asc",
                "limit": 2,
                "cursor": first_page["next_cursor"],
            },
        )
        assert second.status_code == 200, second.text
        assert len(second.json()["data"]["items"]) == 1
        assert {
            item["id"]
            for item in first_page["items"] + second.json()["data"]["items"]
        } == set(kb_ids)
        mismatched_cursor = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={
                "q": "different filter",
                "sort": "name_asc",
                "cursor": first_page["next_cursor"],
            },
        )
        assert mismatched_cursor.status_code == 422
        assert mismatched_cursor.json()["error"]["code"] == "INVALID_CURSOR"
        forged_cursor = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={
                "q": "TODO12 Cursor",
                "sort": "name_asc",
                "cursor": _tamper_cursor(first_page["next_cursor"]),
            },
        )
        assert forged_cursor.status_code == 422
        assert forged_cursor.json()["error"]["code"] == "INVALID_CURSOR"

        timestamp_page = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={"q": "TODO12 Cursor", "sort": "updated_desc", "limit": 1},
        ).json()["data"]
        invalid_time_payload = decode_signed_cursor(
            timestamp_page["next_cursor"],
            purpose="knowledge-workbench-keyset-v1",
        )
        invalid_time_payload["p"][0] = "not-an-iso-8601-timestamp"
        invalid_time_cursor = encode_signed_cursor(
            invalid_time_payload,
            purpose="knowledge-workbench-keyset-v1",
        )
        invalid_time = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={
                "q": "TODO12 Cursor",
                "sort": "updated_desc",
                "cursor": invalid_time_cursor,
            },
        )
        assert invalid_time.status_code == 422
        assert invalid_time.json()["error"]["code"] == "INVALID_CURSOR"

        kb_id = kb_ids[0]
        detail = client.get(f"/api/v1/workbench/knowledge-bases/{kb_id}")
        assert detail.status_code == 200, detail.text
        etag = detail.headers["etag"]
        assert detail.json()["data"]["etag"] == etag
        missing = client.patch(
            f"/api/v1/workbench/knowledge-bases/{kb_id}",
            json={"description": "missing CAS"},
        )
        assert missing.status_code == 428
        assert missing.json()["error"]["code"] == "PRECONDITION_REQUIRED"
        stale = client.patch(
            f"/api/v1/workbench/knowledge-bases/{kb_id}",
            headers={"If-Match": '"kb:stale:r0"'},
            json={"description": "stale CAS"},
        )
        assert stale.status_code == 412
        assert stale.headers["etag"] == etag
        updated = client.patch(
            f"/api/v1/workbench/knowledge-bases/{kb_id}",
            headers={"If-Match": etag},
            json={"description": "updated through If-Match"},
        )
        assert updated.status_code == 200, updated.text
        assert updated.headers["etag"] != etag

        fixture = _seed_document(tmp_path, monkeypatch, kb_id=kb_id)
        document_id = fixture["registered"]["document"]["id"]
        version_id = fixture["registered"]["document_version"]["id"]
        documents = client.get(
            f"/api/v1/workbench/knowledge-bases/{kb_id}/documents"
        )
        assert documents.status_code == 200, documents.text
        assert documents.json()["data"]["total"] == 1
        assert documents.json()["data"]["items"][0]["id"] == document_id
        forbidden = {
            "source_uri",
            "object_key",
            "artifact_uri",
            "access_token",
            "api_token",
            "auth_token",
            "bucket",
            "bucket_name",
            "file_path",
            "local_uri",
            "lease_token",
            "collection_name",
            "path",
            "token",
            "uri",
        }
        assert not forbidden & _all_keys(documents.json())

        version = client.get(
            f"/api/v1/workbench/document-versions/{version_id}"
        )
        assert version.status_code == 200, version.text
        assert not forbidden & _all_keys(version.json())
        assert version.json()["data"]["source_url"].endswith(
            f"/document-versions/{version_id}/source"
        )
        assert version.json()["data"]["parser_config"]["provider"] == {
            "region": "cn-test"
        }
        blocks = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/blocks",
            params={"limit": 1},
        )
        assert blocks.status_code == 200, blocks.text
        assert blocks.json()["data"]["total"] == 2
        assert blocks.json()["data"]["next_cursor"]
        assert blocks.json()["data"]["items"][0]["structured_data"] == {
            "nested": {},
            "safe": "公开结构化数据",
        }
        forged_artifact_cursor = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/blocks",
            params={
                "limit": 1,
                "cursor": _tamper_cursor(
                    blocks.json()["data"]["next_cursor"]
                ),
            },
        )
        assert forged_artifact_cursor.status_code == 422
        assert forged_artifact_cursor.json()["error"]["code"] == "INVALID_CURSOR"
        invalid_offset_payload = decode_signed_cursor(
            blocks.json()["data"]["next_cursor"],
            purpose="knowledge-workbench-artifact-v1",
        )
        invalid_offset_payload["offset"] = True
        invalid_offset = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/blocks",
            params={
                "limit": 1,
                "cursor": encode_signed_cursor(
                    invalid_offset_payload,
                    purpose="knowledge-workbench-artifact-v1",
                ),
            },
        )
        assert invalid_offset.status_code == 422
        assert invalid_offset.json()["error"]["code"] == "INVALID_CURSOR"
        chunks = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/chunks",
            params={"q": "HAK180"},
        )
        assert chunks.status_code == 200, chunks.text
        assert chunks.json()["data"]["total"] == 1
        assert "dense_vector" not in chunks.text
        assert chunks.json()["data"]["items"][0]["metadata"]["nested"] == {
            "safe": "visible"
        }
        assert chunks.json()["data"]["items"][0]["source_block_ids"] == [
            "block_1"
        ]
        assert chunks.json()["data"]["items"][0]["location"] == [
            {"page_no": 1, "bbox": [0.1, 0.1, 0.9, 0.2]}
        ]

        source = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/source",
            headers={"Range": "bytes=0-3"},
        )
        assert source.status_code == 206, source.text
        assert source.content == fixture["source"][:4]
        assert source.headers["content-range"] == (
            f"bytes 0-3/{len(fixture['source'])}"
        )
        assert source.headers["accept-ranges"] == "bytes"
        assert source.headers["x-frame-options"] == "SAMEORIGIN"
        assert "frame-ancestors 'self'" in source.headers[
            "content-security-policy"
        ]
        assert source.headers["x-content-type-options"] == "nosniff"
        invalid_range = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/source",
            headers={"Range": "bytes=999999-"},
        )
        assert invalid_range.status_code == 416

        job_id = fixture["registered"]["ingestion_job"]["id"]
        job = client.get(f"/api/v1/workbench/jobs/{job_id}")
        assert job.status_code == 200, job.text
        assert job.json()["data"]["document"]["id"] == document_id
        assert "events" in job.json()["data"]
        assert "checkpoints" in job.json()["data"]
        assert "manifest" in job.json()["data"]
        assert not forbidden & _all_keys(job.json())
        assert job.json()["data"]["error_code"] == "TODO12_FIXTURE_FAILURE"
        assert job.json()["data"]["error_detail"] == (
            "任务执行未成功；详细诊断信息仅在服务端可见。"
        )
        assert job.json()["data"]["events"][0]["payload"] == {
            "nested": {
                "artifact": {"sha256": "safe-public-integrity-fact"},
                "safe": "visible",
            }
        }
        assert job.json()["data"]["manifest"]["validation_errors"] == [
            {
                "code": "VECTOR_COUNT_MISMATCH",
                "nested": {},
                "safe": "公开一致性校验摘要",
            }
        ]
        for private_value in (
            "private-worker-token",
            "private-manifest-token",
            "private-block-token",
            "private/index/object",
            "private\\stt\\source.pdf",
            "s3://private/source",
        ):
            assert private_value not in blocks.text
            assert private_value not in job.text

        fixture["source_path"].write_bytes(b"tampered")
        tampered = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/source"
        )
        assert tampered.status_code == 409
        assert tampered.json()["error"]["code"] == "SOURCE_OBJECT_INTEGRITY_ERROR"


def test_chunks_project_standard_nested_provenance_and_filter_by_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configure_database(
        f"sqlite:///{(tmp_path / 'todo12-nested-provenance.db').as_posix()}"
    )
    with TestClient(import_server.app) as client:
        _login(client)
        kb_id = _create_kb(client, "TODO12 Nested Provenance KB")
        fixture = _seed_document(tmp_path, monkeypatch, kb_id=kb_id)
        version_id = fixture["registered"]["document_version"]["id"]
        nested_chunks = [
            {
                "chunk_id": "chunk_nested_paper_table",
                "content": "标准纸盒最多可装入 250 张 80 g/m² 纸。",
                "type": "table",
                "images": ["images/paper-tray.jpg"],
                "provenance": {
                    "source_block_ids": ["block_paper_table", "block_paper_note"],
                    "heading_path": ["纸张与打印介质", "纸张规格"],
                    "locations": [
                        {
                            "block_id": "block_paper_table",
                            "page_no": 25,
                            "bbox": [0.11, 0.22, 0.88, 0.64],
                            "coordinate_space": "normalized_page",
                        }
                    ],
                    "parser_id": "mineru.pdf",
                    "parser_version": "1.1.2",
                },
            }
        ]
        payload = canonical_json(nested_chunks)
        chunk_path = Path(fixture["chunk_record"]["artifact_uri"])
        chunk_path.write_bytes(payload)
        with session_scope() as session:
            chunk_run = session.get(
                DocumentChunkingRunModel,
                fixture["chunk_record"]["id"],
            )
            assert chunk_run is not None
            chunk_run.artifact_sha256 = sha256(payload).hexdigest()
            chunk_run.artifact_size = len(payload)
            chunk_run.chunk_count = 1

        response = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/chunks",
            params={"page_no": 25},
        )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["total"] == 1
        item = data["items"][0]
        assert item["source_block_ids"] == [
            "block_paper_table",
            "block_paper_note",
        ]
        assert item["parent_heading_path"] == ["纸张与打印介质", "纸张规格"]
        assert item["location"] == [
            {
                "block_id": "block_paper_table",
                "page_no": 25,
                "bbox": [0.11, 0.22, 0.88, 0.64],
                "coordinate_space": "normalized_page",
            }
        ]
        assert item["has_images"] is True

        missing_page = client.get(
            f"/api/v1/workbench/document-versions/{version_id}/chunks",
            params={"page_no": 26},
        )
        assert missing_page.status_code == 200, missing_page.text
        assert missing_page.json()["data"]["total"] == 0


def test_chunk_draft_is_persistent_cas_protected_and_publishes_new_version(
    tmp_path: Path, monkeypatch
) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo12-chunk-draft.db').as_posix()}")
    with TestClient(import_server.app) as client:
        _login(client)
        kb_id = _create_kb(client, "TODO12 Chunk Draft KB")
        fixture = _seed_document(tmp_path, monkeypatch, kb_id=kb_id)
        base_version_id = fixture["registered"]["document_version"]["id"]
        created = client.post(
            f"/api/v1/workbench/document-versions/{base_version_id}/chunk-drafts",
            json={"base_chunking_run_id": fixture["chunk_record"]["id"]},
        )
        assert created.status_code == 201, created.text
        draft = created.json()["data"]
        draft_id = draft["id"]
        etag_1 = created.headers["etag"]

        missing = client.patch(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks/chunk_1",
            json={"chunk": {"content": "缺少 If-Match"}},
        )
        assert missing.status_code == 428
        replaced = client.patch(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks/chunk_1",
            headers={"If-Match": etag_1},
            json={
                "chunk": {
                    "content": "在使用 HAK180 前，必须阅读安全说明并检查设备。",
                    "type": "text",
                    "source_block_ids": ["block_1"],
                    "metadata": {"reviewed": True},
                }
            },
        )
        assert replaced.status_code == 200, replaced.text
        etag_2 = replaced.headers["etag"]
        replacement_id = replaced.json()["data"]["operation_result"][
            "output_chunk_ids"
        ][0]
        stale = client.patch(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks/chunk_1",
            headers={"If-Match": etag_1},
            json={"chunk": {"content": "stale mutation must be rejected"}},
        )
        assert stale.status_code == 412
        assert stale.headers["etag"] == etag_2

        split = client.post(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks/chunk_2:split",
            headers={"If-Match": etag_2},
            json={
                "chunks": [
                    {"content": "检查设备外观。", "source_block_ids": ["block_2"]},
                    {"content": "检查电源连接。", "source_block_ids": ["block_2"]},
                ]
            },
        )
        assert split.status_code == 200, split.text
        split_ids = split.json()["data"]["operation_result"]["output_chunk_ids"]
        assert len(split_ids) == 2
        merged = client.post(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks:merge",
            headers={"If-Match": split.headers["etag"]},
            json={
                "chunk_ids": [replacement_id, split_ids[0]],
                "chunk": {
                    "content": "使用前阅读安全说明并检查设备外观。",
                    "source_block_ids": ["block_1", "block_2"],
                },
            },
        )
        assert merged.status_code == 200, merged.text
        draft_chunks = client.get(
            f"/api/v1/workbench/chunk-drafts/{draft_id}/chunks"
        )
        assert draft_chunks.status_code == 200, draft_chunks.text
        assert draft_chunks.json()["data"]["total"] == 2
        assert any(
            item["content"] == "使用前阅读安全说明并检查设备外观。"
            for item in draft_chunks.json()["data"]["items"]
        )

        published = client.post(
            f"/api/v1/workbench/chunk-drafts/{draft_id}:publish",
            headers={"If-Match": merged.headers["etag"]},
            json={"note": "TODO-12 manual chunk correction"},
        )
        assert published.status_code == 202, published.text
        result = published.json()["data"]
        assert result["status"] == "publishing"
        assert result["published_document_version_id"] != base_version_id
        assert result["published_job_id"] == result["job"]["id"]
        assert result["job"]["operation"] == "chunk_override"
        assert fixture["chunk_record"]["artifact_sha256"] == sha256(
            fixture["chunk_payload"]
        ).hexdigest()

        # Re-register the parse fact that the real worker creates for the new immutable
        # DocumentVersion, then run only the deterministic pre-embedding override hook.
        new_version_id = result["published_document_version_id"]
        with session_scope() as session:
            new_version = session.get(DocumentVersionModel, new_version_id)
            assert new_version is not None
            worker_config = dict(new_version.parser_config_snapshot)
        new_parse = control.record_document_parse_artifact(
            document_version_id=new_version_id,
            input_sha256=fixture["source_hash"],
            parser_id="todo12-fixture",
            parser_version="1.0.0",
            schema_version="parsed-document.v1",
            status="succeeded",
            artifact_uri=fixture["parse_record"]["artifact_uri"],
            artifact_sha256=fixture["parse_record"]["artifact_sha256"],
            artifact_size=fixture["parse_record"]["artifact_size"],
        )
        state = {
            "document_version_id": new_version_id,
            "parser_config": worker_config,
            "parsed_document_summary": {"parse_artifact_id": new_parse["id"]},
        }
        overridden = workbench.apply_chunk_override_for_worker(
            state, fixture["chunks"]
        )
        assert len(overridden) == 2
        assert all("dense_vector" not in item for item in overridden)
        assert all("sparse_vector" not in item for item in overridden)
        assert any(
            item["content"] == "使用前阅读安全说明并检查设备外观。"
            for item in overridden
        )
        assert state["parsed_document_summary"]["chunk_override_draft_id"] == draft_id

        diff = client.get(
            f"/api/v1/workbench/document-versions/{base_version_id}/diff",
            params={"other_version_id": new_version_id},
        )
        assert diff.status_code == 200, diff.text
        assert diff.json()["data"]["summary"]["changed_hunk_count"] > 0

        # Publishing is idempotent and never creates another version/job.
        replay = client.post(
            f"/api/v1/workbench/chunk-drafts/{draft_id}:publish",
            headers={"If-Match": published.headers["etag"]},
            json={"note": "ignored replay"},
        )
        assert replay.status_code == 202, replay.text
        assert replay.json()["data"]["published_job_id"] == result["published_job_id"]


def test_openapi_marks_business_required_if_match_headers(tmp_path: Path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo12-openapi.db').as_posix()}")
    with TestClient(import_server.app) as client:
        specification = client.get("/openapi.json").json()
    expected = {
        ("/api/v1/workbench/knowledge-bases/{kb_id}", "patch"),
        (
            "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}",
            "patch",
        ),
        (
            "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}:split",
            "post",
        ),
        ("/api/v1/workbench/chunk-drafts/{draft_id}/chunks:merge", "post"),
        (
            "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}",
            "delete",
        ),
        ("/api/v1/workbench/chunk-drafts/{draft_id}:publish", "post"),
    }
    for path, method in expected:
        assert specification["paths"][path][method]["x-stt-required-headers"] == [
            "If-Match"
        ]

    schemas = specification["components"]["schemas"]
    assert not any(schema == {} for schema in schemas.values())
    json_value = schemas["WorkbenchJsonValue-Output"]
    assert {branch.get("type") for branch in json_value["anyOf"]} >= {
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "null",
    }
    for path, operations in specification["paths"].items():
        if not path.startswith("/api/v1/workbench") or path.endswith("/source"):
            continue
        for operation in operations.values():
            if not isinstance(operation, dict) or "responses" not in operation:
                continue
            success = next(
                response
                for code, response in operation["responses"].items()
                if str(code).startswith("2")
            )
            response_schema = success["content"]["application/json"]["schema"]
            reference = response_schema.get("$ref", "")
            assert reference.startswith("#/components/schemas/WorkbenchEnvelope_")
            envelope = schemas[reference.rsplit("/", 1)[-1]]
            data_schema = envelope["properties"]["data"]
            assert data_schema.get("$ref", "").startswith("#/components/schemas/")


def test_workbench_authorizes_before_pagination_and_applies_acl_role_matrix(
    tmp_path: Path,
) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo12-rbac.db').as_posix()}")
    tenant_id = "tenant_default"
    users = {
        "viewer": (
            "todo12-viewer@stt.local",
            "TODO12-Viewer-Password!",  # secret-scan: allow-test
            "Viewer",
        ),
        "editor": (
            "todo12-editor@stt.local",
            "TODO12-Editor-Password!",  # secret-scan: allow-test
            "Editor",
        ),
        "kb_admin": (
            "todo12-kbadmin@stt.local",
            "TODO12-KBAdmin-Password!",  # secret-scan: allow-test
            "KBAdmin",
        ),
        "owner": (
            "todo12-owner@stt.local",
            "TODO12-Owner-Password!",  # secret-scan: allow-test
            "Viewer",
        ),
    }
    with TestClient(import_server.app) as client:
        _login(client)
        denied_kb = _create_kb(client, "TODO12 Matrix 00 denied")
        allowed_kb = _create_kb(client, "TODO12 Matrix 99 allowed")
        user_ids: dict[str, str] = {}
        for key, (email, password, role) in users.items():
            response = client.post(
                "/api/v1/admin/users",
                json={
                    "email": email,
                    "password": password,
                    "display_name": f"TODO12 {key}",
                    "role": role,
                },
            )
            assert response.status_code == 200, response.text
            user_ids[key] = response.json()["user_id"]

        other_tenant = client.post(
            "/api/v1/admin/tenants", json={"name": "TODO12 Other Tenant"}
        )
        assert other_tenant.status_code == 200, other_tenant.text
        other_tenant_id = other_tenant.json()["id"]
        _login_as(
            client,
            email=ADMIN["email"],
            password=ADMIN["password"],
            tenant_id=other_tenant_id,
        )
        other_kb = _create_kb(client, "TODO12 Matrix other tenant")
        _login_as(
            client,
            email=ADMIN["email"],
            password=ADMIN["password"],
            tenant_id=tenant_id,
        )

        owner_kb = new_id("kb")
        with session_scope() as session:
            session.add(
                KnowledgeBaseModel(
                    id=owner_kb,
                    tenant_id=tenant_id,
                    workspace_id=DEFAULT_WORKSPACE_ID,
                    name="TODO12 Matrix owner",
                    description="owner visibility fixture",
                    status="active",
                    created_by=user_ids["owner"],
                )
            )
            session.add_all(
                [
                    KnowledgeBaseAclModel(
                        id=new_id("acl"),
                        kb_id=allowed_kb,
                        principal_type="user",
                        principal_id=user_ids["viewer"],
                        role="reader",
                    ),
                    KnowledgeBaseAclModel(
                        id=new_id("acl"),
                        kb_id=allowed_kb,
                        principal_type="user",
                        principal_id=user_ids["editor"],
                        role="editor",
                    ),
                    KnowledgeBaseAclModel(
                        id=new_id("acl"),
                        kb_id=allowed_kb,
                        principal_type="user",
                        principal_id=user_ids["kb_admin"],
                        role="admin",
                    ),
                ]
            )

        viewer_email, viewer_password, _ = users["viewer"]
        _login_as(
            client,
            email=viewer_email,
            password=viewer_password,
            tenant_id=tenant_id,
        )
        page = client.get(
            "/api/v1/workbench/knowledge-bases",
            params={"q": "TODO12 Matrix", "sort": "name_asc", "limit": 1},
        )
        assert page.status_code == 200, page.text
        payload = page.json()["data"]
        # denied_kb sorts first globally; it must be removed by SQL authorization
        # before LIMIT is applied, not filtered out after an empty first page.
        assert payload["total"] == 1
        assert [item["id"] for item in payload["items"]] == [allowed_kb]
        assert payload["next_cursor"] is None
        assert payload["items"][0]["actions"] == ["read"]
        assert client.get(
            f"/api/v1/workbench/knowledge-bases/{denied_kb}"
        ).status_code == 404
        assert client.get(
            f"/api/v1/workbench/knowledge-bases/{other_kb}"
        ).status_code == 404

        editor_email, editor_password, _ = users["editor"]
        _login_as(
            client,
            email=editor_email,
            password=editor_password,
            tenant_id=tenant_id,
        )
        editor_actions = client.get(
            f"/api/v1/workbench/knowledge-bases/{allowed_kb}"
        ).json()["data"]["actions"]
        assert "upload_document" in editor_actions
        assert "create_chunk_draft" in editor_actions
        assert "update" not in editor_actions
        assert "delete_document" not in editor_actions

        kb_admin_email, kb_admin_password, _ = users["kb_admin"]
        _login_as(
            client,
            email=kb_admin_email,
            password=kb_admin_password,
            tenant_id=tenant_id,
        )
        admin_actions = client.get(
            f"/api/v1/workbench/knowledge-bases/{allowed_kb}"
        ).json()["data"]["actions"]
        assert {
            "read",
            "update",
            "upload_document",
            "enable_document",
            "disable_document",
            "delete_document",
            "publish_index",
        } <= set(admin_actions)

        owner_email, owner_password, _ = users["owner"]
        _login_as(
            client,
            email=owner_email,
            password=owner_password,
            tenant_id=tenant_id,
        )
        owned = client.get(f"/api/v1/workbench/knowledge-bases/{owner_kb}")
        assert owned.status_code == 200, owned.text
        assert owned.json()["data"]["access"]["grant_role"] == "owner"
        # Ownership does not widen the tenant membership's global permission set.
        assert owned.json()["data"]["actions"] == ["read"]
