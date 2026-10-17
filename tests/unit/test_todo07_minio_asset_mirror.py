"""TODO-07 解析制品及其二进制资产的 MinIO 镜像回归。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.application.parse_artifact_store import (
    LocalParseArtifactStore,
    MinIOParseArtifactMirror,
    build_parse_cache_key,
)
from app.domain.parsing import (
    DocumentParseError,
    ParsedAsset,
    ParsedBlock,
    ParsedDocument,
    SourceLocation,
)


SOURCE_SHA256 = "a" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\nTODO-07-image-payload"


class FakeNoSuchKey(Exception):
    code = "NoSuchKey"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class InMemoryMinIOClient:
    """仅实现 mirror 使用的最小 MinIO 协议，完全不访问网络。"""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_order: list[str] = []
        self.responses: list[FakeResponse] = []

    def put_object(
        self,
        bucket_name: str,
        object_key: str,
        data,
        length: int,
        *,
        content_type: str,
    ) -> None:
        del content_type
        payload = data.read()
        assert len(payload) == length
        self.objects[(bucket_name, object_key)] = payload
        self.put_order.append(object_key)

    def get_object(self, bucket_name: str, object_key: str) -> FakeResponse:
        try:
            payload = self.objects[(bucket_name, object_key)]
        except KeyError as exc:
            raise FakeNoSuchKey(object_key) from exc
        response = FakeResponse(payload)
        self.responses.append(response)
        return response


def _cache_key() -> str:
    return build_parse_cache_key(
        input_sha256=SOURCE_SHA256,
        parser_id="test.asset-parser",
        parser_version="1.0.0",
        parser_config={"language": "zh-CN"},
    )


def _document(asset_payload: bytes = PNG_BYTES) -> ParsedDocument:
    return ParsedDocument(
        parser_id="test.asset-parser",
        parser_version="1.0.0",
        source_sha256=SOURCE_SHA256,
        source_format="docx",
        source_mime_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        blocks=[
            ParsedBlock(
                block_id="image-block",
                type="image",
                text="安全接线示意图",
                location=SourceLocation(page_no=2),
                order=0,
                asset_refs=["asset-img"],
            )
        ],
        assets=[
            ParsedAsset(
                asset_id="asset-img",
                object_key="assets/diagram.png",
                sha256=hashlib.sha256(asset_payload).hexdigest(),
                media_type="image/png",
                width=16,
                height=8,
                size_bytes=len(asset_payload),
            )
        ],
    )


def _prepare_local_artifact(
    root: Path,
    *,
    document: ParsedDocument | None = None,
    asset_payload: bytes | None = PNG_BYTES,
) -> tuple[LocalParseArtifactStore, str, Path, object]:
    store = LocalParseArtifactStore(root)
    key = _cache_key()
    parsed = document or _document()
    info = store.save(key, parsed)
    work_dir = store.artifact_path(key).parent / f"{key}.work"
    if asset_payload is not None:
        asset_path = work_dir / "assets" / "diagram.png"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(asset_payload)
    return store, key, work_dir, info


def _remote_keys(client: InMemoryMinIOClient, bucket: str) -> tuple[str, str, str]:
    keys = {key for stored_bucket, key in client.objects if stored_bucket == bucket}
    json_key = next(key for key in keys if key.endswith(".json"))
    digest_key = next(key for key in keys if key.endswith(".sha256"))
    asset_key = next(key for key in keys if ".assets/" in key)
    return json_key, digest_key, asset_key


def test_mirror_uploads_document_and_assets_then_rehydrates_empty_store(
    tmp_path: Path,
) -> None:
    source_store, key, work_dir, info = _prepare_local_artifact(tmp_path / "source")
    document = _document()
    client = InMemoryMinIOClient()
    mirror = MinIOParseArtifactMirror(client, "todo07")

    mirrored_info = mirror.upload(info, document=document, work_dir=work_dir)
    json_key, digest_key, asset_key = _remote_keys(client, "todo07")

    assert mirrored_info.object_key == json_key
    assert client.put_order == [asset_key, json_key, digest_key]
    assert client.put_order[-1].endswith(".sha256")
    remote_document = json.loads(client.objects[("todo07", json_key)])
    assert remote_document["assets"] == [
        document.assets[0].model_dump(mode="json")
    ]
    assert client.objects[("todo07", asset_key)] == PNG_BYTES
    assert client.objects[("todo07", digest_key)].decode("ascii") == info.sha256

    target_store = LocalParseArtifactStore(tmp_path / "empty-target")
    assert target_store.load(key) is None
    downloaded = mirror.download_into(key, target_store)

    assert downloaded is not None
    downloaded_document, downloaded_info = downloaded
    assert downloaded_document == document
    assert downloaded_info.cache_hit is True
    assert downloaded_info.object_key == json_key
    assert target_store.load(key) is not None
    restored_asset = (
        target_store.artifact_path(key).parent
        / f"{key}.work"
        / "assets"
        / "diagram.png"
    )
    assert restored_asset.read_bytes() == PNG_BYTES
    assert all(response.closed and response.released for response in client.responses)
    assert Path(source_store.artifact_path(key)).is_file()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing", "PARSE_ARTIFACT_ASSET_MISSING"),
        ("tampered", "PARSE_ARTIFACT_ASSET_INTEGRITY_ERROR"),
    ],
)
def test_mirror_download_fails_closed_for_missing_or_tampered_asset(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    _, key, work_dir, info = _prepare_local_artifact(tmp_path / "source")
    client = InMemoryMinIOClient()
    mirror = MinIOParseArtifactMirror(client, "todo07")
    mirror.upload(info, document=_document(), work_dir=work_dir)
    _, _, asset_key = _remote_keys(client, "todo07")
    if mutation == "missing":
        del client.objects[("todo07", asset_key)]
    else:
        client.objects[("todo07", asset_key)] = b"tampered-asset"
    target_store = LocalParseArtifactStore(tmp_path / f"target-{mutation}")

    with pytest.raises(DocumentParseError) as captured:
        mirror.download_into(key, target_store)

    assert captured.value.code == expected_code
    assert target_store.load(key) is None


@pytest.mark.parametrize(
    ("asset_payload", "expected_code"),
    [
        (None, "PARSE_ARTIFACT_ASSET_MISSING"),
        (b"locally-tampered", "PARSE_ARTIFACT_ASSET_INTEGRITY_ERROR"),
    ],
)
def test_mirror_upload_fails_before_commit_for_bad_local_asset(
    tmp_path: Path,
    asset_payload: bytes | None,
    expected_code: str,
) -> None:
    _, _, work_dir, info = _prepare_local_artifact(
        tmp_path / "source",
        asset_payload=asset_payload,
    )
    client = InMemoryMinIOClient()
    mirror = MinIOParseArtifactMirror(client, "todo07")

    with pytest.raises(DocumentParseError) as captured:
        mirror.upload(info, document=_document(), work_dir=work_dir)

    assert captured.value.code == expected_code
    assert client.objects == {}
    assert not any(key.endswith(".sha256") for key in client.put_order)


def test_mirror_rejects_tampered_document_before_writing_local_cache(
    tmp_path: Path,
) -> None:
    _, key, work_dir, info = _prepare_local_artifact(tmp_path / "source")
    client = InMemoryMinIOClient()
    mirror = MinIOParseArtifactMirror(client, "todo07")
    mirror.upload(info, document=_document(), work_dir=work_dir)
    json_key, _, _ = _remote_keys(client, "todo07")
    client.objects[("todo07", json_key)] += b"\n"
    target_store = LocalParseArtifactStore(tmp_path / "target")

    with pytest.raises(DocumentParseError) as captured:
        mirror.download_into(key, target_store)

    assert captured.value.code == "PARSE_ARTIFACT_INTEGRITY_ERROR"
    assert target_store.load(key) is None

