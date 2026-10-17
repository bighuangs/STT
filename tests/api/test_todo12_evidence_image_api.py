from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from app.api.http import query_server
from app.infra.object_storage.minio_gateway import minio_gateway
from app.infra.persistence.sql.repositories import ResourceNotFoundError
from app.rag.query.evidence_asset_service import evidence_image_assets
from app.security.principal import Principal


EVIDENCE_ID = "ev_" + "2" * 40


def _request() -> Request:
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    request.state.principal = Principal(
        user_id="user_media",
        tenant_id="tenant_media",
        roles=("Viewer",),
        scopes=("document:read",),
        auth_type="password",
        credential_id="cred_media",
    )
    return request


def _evidence() -> dict:
    return {
        "evidence_id": EVIDENCE_ID,
        "source": {
            "source_type": "local",
            "tenant_id": "tenant_media",
            "kb_id": "kb_media",
            "document_id": "doc_media",
            "document_version_id": "dver_media",
            "index_version_id": "idx_media",
            "chunk_id": "chunk_media",
        },
        "asset_refs": ["imagetemps-dver_media/idx_media/page-7.jpg"],
        "content": "图片证据",
        "provenance_spans": [{"page_no": 7}],
    }


def test_evidence_authorization_rechecks_current_kb_entitlement(monkeypatch) -> None:
    monkeypatch.setattr(query_server, "get_authorized_evidence", lambda _: _evidence())

    class AllowedResolver:
        def resolve(self, **_):
            return SimpleNamespace(grants=(SimpleNamespace(kb_id="kb_media"),))

    monkeypatch.setattr(query_server, "SqlEntitlementResolver", AllowedResolver)
    assert query_server._authorized_evidence_for_request(_request(), EVIDENCE_ID)[
        "evidence_id"
    ] == EVIDENCE_ID

    class DeniedResolver:
        def resolve(self, **_):
            return SimpleNamespace(grants=())

    monkeypatch.setattr(query_server, "SqlEntitlementResolver", DeniedResolver)
    with pytest.raises(ResourceNotFoundError):
        query_server._authorized_evidence_for_request(_request(), EVIDENCE_ID)


def test_evidence_preview_projects_only_same_origin_image_identity(monkeypatch) -> None:
    asset = evidence_image_assets(_evidence())[0]
    monkeypatch.setattr(
        query_server,
        "_authorized_evidence_for_request",
        lambda *_: _evidence(),
    )
    monkeypatch.setattr(
        query_server,
        "evidence_image_projection",
        lambda _: [
            {
                "image_id": asset.image_id,
                "url": asset.preview_url(EVIDENCE_ID),
                "media_type": asset.media_type,
            }
        ],
    )
    payload = query_server.evidence_preview(_request(), EVIDENCE_ID)

    assert payload["image_urls"] == [payload["images"][0]["url"]]
    assert payload["images"][0]["url"].startswith(
        f"/api/v1/evidence/{EVIDENCE_ID}/images/img_"
    )
    assert "dver_media/idx_media" not in payload["images"][0]["url"]


def test_evidence_image_streams_manifested_object_and_closes_response(monkeypatch) -> None:
    evidence = _evidence()
    asset = evidence_image_assets(evidence)[0]
    monkeypatch.setattr(
        query_server,
        "_authorized_evidence_for_request",
        lambda *_: evidence,
    )
    monkeypatch.setattr(query_server, "image_asset_is_manifested", lambda *_: True)

    class ObjectResponse:
        closed = False
        released = False

        def stream(self, *, amt):
            assert amt == 64 * 1024
            yield b"jpeg-bytes"

        def close(self):
            self.closed = True

        def release_conn(self):
            self.released = True

    object_response = ObjectResponse()

    class Client:
        def stat_object(self, bucket, object_key):
            assert bucket == minio_gateway.bucket_name
            assert object_key == asset.object_key
            return SimpleNamespace(size=10, etag="abc")

        def get_object(self, bucket, object_key):
            assert bucket == minio_gateway.bucket_name
            assert object_key == asset.object_key
            return object_response

    monkeypatch.setattr(minio_gateway, "client", lambda: Client())
    response = query_server.evidence_image_preview(_request(), EVIDENCE_ID, asset.image_id)

    async def consume() -> bytes:
        chunks = [chunk async for chunk in response.body_iterator]
        return b"".join(chunks)

    assert asyncio.run(consume()) == b"jpeg-bytes"
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"].startswith("private")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert object_response.closed is True
    assert object_response.released is True


def test_evidence_image_rejects_forged_id_and_unmanifested_asset(monkeypatch) -> None:
    evidence = _evidence()
    asset = evidence_image_assets(evidence)[0]
    monkeypatch.setattr(
        query_server,
        "_authorized_evidence_for_request",
        lambda *_: evidence,
    )

    with pytest.raises(ResourceNotFoundError):
        query_server.evidence_image_preview(_request(), EVIDENCE_ID, "img_forged")

    monkeypatch.setattr(query_server, "image_asset_is_manifested", lambda *_: False)
    with pytest.raises(ResourceNotFoundError):
        query_server.evidence_image_preview(_request(), EVIDENCE_ID, asset.image_id)
