"""TODO-07 解析制品缓存与完整性校验。

解析缓存与切块模板刻意分离：缓存键不包含 template id/version，因此同一份
ParsedDocument 切换模板时不会重新调用 MinerU/OCR。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
import uuid

from app.domain.parsing import DocumentParseError, ParseArtifactInfo, ParsedDocument


PARSE_SCHEMA_VERSION = "parsed-document.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def extraction_config(parser_config: dict | None) -> dict:
    """只保留会改变“解析结构”的参数，明确排除切块/LLM/向量参数。"""

    source = parser_config or {}
    prefixes = ("mineru_", "ocr_", "parser_", "zip_")
    exact = {
        "language",
        "page_start",
        "page_end",
        "html_keep_links",
        "office_include_notes",
        "max_table_rows_per_block",
    }
    return {
        key: source[key]
        for key in sorted(source)
        if key in exact or key.startswith(prefixes)
    }


def build_parse_cache_key(
    *,
    input_sha256: str,
    parser_id: str,
    parser_version: str,
    parser_config: dict | None,
) -> str:
    payload = {
        "input_sha256": input_sha256,
        "parser_id": parser_id,
        "parser_version": parser_version,
        "schema_version": PARSE_SCHEMA_VERSION,
        "extraction_config": extraction_config(parser_config),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class LocalParseArtifactStore:
    """受根目录约束、原子写入且 fail-closed 的本地解析制品仓库。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _path(self, cache_key: str, suffix: str) -> Path:
        if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
            raise DocumentParseError("INVALID_CACHE_KEY", "解析缓存键格式非法")
        target = (self.root / cache_key[:2] / f"{cache_key}{suffix}").resolve()
        if self.root not in target.parents:
            raise DocumentParseError("UNSAFE_ARTIFACT_PATH", "解析制品路径越界")
        return target

    def artifact_path(self, cache_key: str) -> Path:
        return self._path(cache_key, ".json")

    def load(self, cache_key: str) -> tuple[ParsedDocument, ParseArtifactInfo] | None:
        target = self._path(cache_key, ".json")
        digest_path = self._path(cache_key, ".sha256")
        if not target.exists() and not digest_path.exists():
            return None
        if not target.is_file() or not digest_path.is_file():
            raise DocumentParseError(
                "PARSE_ARTIFACT_INCOMPLETE",
                "解析制品或校验文件缺失，拒绝使用不完整缓存",
            )
        payload = target.read_bytes()
        expected = digest_path.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(payload).hexdigest()
        if expected != actual:
            raise DocumentParseError(
                "PARSE_ARTIFACT_INTEGRITY_ERROR",
                "解析制品 SHA-256 校验失败",
            )
        try:
            document = ParsedDocument.model_validate_json(payload)
        except Exception as exc:
            raise DocumentParseError(
                "PARSE_ARTIFACT_SCHEMA_ERROR",
                "解析制品不符合 parsed-document.v1 契约",
            ) from exc
        return document, ParseArtifactInfo(
            uri=str(target),
            sha256=actual,
            size_bytes=len(payload),
            cache_key=cache_key,
            cache_hit=True,
        )

    def save(self, cache_key: str, document: ParsedDocument) -> ParseArtifactInfo:
        target = self._path(cache_key, ".json")
        digest_path = self._path(cache_key, ".sha256")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(document.model_dump(mode="json"))
        digest = hashlib.sha256(payload).hexdigest()
        nonce = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        tmp_target = target.parent / f".json-{nonce}.tmp"
        tmp_digest = target.parent / f".sha-{nonce}.tmp"
        try:
            tmp_target.write_bytes(payload)
            tmp_digest.write_text(digest, encoding="ascii")
            os.replace(tmp_target, target)
            os.replace(tmp_digest, digest_path)
        finally:
            tmp_target.unlink(missing_ok=True)
            tmp_digest.unlink(missing_ok=True)
        return ParseArtifactInfo(
            uri=str(target),
            sha256=digest,
            size_bytes=len(payload),
            cache_key=cache_key,
            cache_hit=False,
        )


class MinIOParseArtifactMirror:
    """解析制品的 MinIO 镜像；对象名只由受校验的 cache key 构造。"""

    prefix = "parsed-artifacts/v1"

    def __init__(self, client, bucket_name: str) -> None:
        self.client = client
        self.bucket_name = bucket_name

    def _keys(self, cache_key: str) -> tuple[str, str]:
        if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
            raise DocumentParseError("INVALID_CACHE_KEY", "解析缓存键格式非法")
        base = f"{self.prefix}/{cache_key[:2]}/{cache_key}"
        return f"{base}.json", f"{base}.sha256"

    def _asset_key(self, cache_key: str, asset_sha256: str, object_key: str) -> str:
        self._keys(cache_key)
        if len(asset_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in asset_sha256
        ):
            raise DocumentParseError(
                "PARSE_ARTIFACT_ASSET_INTEGRITY_ERROR",
                "解析资产 SHA-256 格式非法",
            )
        suffix = PurePosixPath(object_key).suffix.lower()
        if not suffix or len(suffix) > 9 or not suffix[1:].isalnum():
            suffix = ".bin"
        return (
            f"{self.prefix}/{cache_key[:2]}/{cache_key}.assets/"
            f"{asset_sha256}{suffix}"
        )

    @staticmethod
    def _local_asset_path(work_dir: Path, object_key: str) -> Path:
        relative = PurePosixPath(object_key)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise DocumentParseError(
                "PARSE_ARTIFACT_ASSET_PATH_UNSAFE",
                "解析资产路径非法，拒绝读写缓存资产",
            )
        root = work_dir.resolve()
        target = (root / Path(*relative.parts)).resolve()
        if root not in target.parents:
            raise DocumentParseError(
                "PARSE_ARTIFACT_ASSET_PATH_UNSAFE",
                "解析资产路径越出缓存工作目录",
            )
        return target

    def _read_object(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket_name, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def upload(
        self,
        info: ParseArtifactInfo,
        *,
        document: ParsedDocument | None = None,
        work_dir: str | Path | None = None,
    ) -> ParseArtifactInfo:
        object_key, digest_key = self._keys(info.cache_key)
        payload = Path(info.uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != info.sha256:
            raise DocumentParseError(
                "PARSE_ARTIFACT_INTEGRITY_ERROR",
                "待上传解析制品的 SHA-256 与登记值不一致",
            )
        parsed = document or ParsedDocument.model_validate_json(payload)
        asset_root = (
            Path(work_dir).resolve()
            if work_dir is not None
            else Path(info.uri).parent / f"{info.cache_key}.work"
        )
        # 先写内容寻址资产，再写 JSON，最后用 sha256 sidecar 作为提交标记。
        # 跨 Worker 下载时只有 sidecar 完整存在且所有资产哈希匹配才会命中。
        for asset in parsed.assets:
            source = self._local_asset_path(asset_root, asset.object_key)
            try:
                asset_payload = source.read_bytes()
            except OSError as exc:
                raise DocumentParseError(
                    "PARSE_ARTIFACT_ASSET_MISSING",
                    f"解析资产缺失，无法持久化：{asset.asset_id}",
                ) from exc
            if hashlib.sha256(asset_payload).hexdigest() != asset.sha256:
                raise DocumentParseError(
                    "PARSE_ARTIFACT_ASSET_INTEGRITY_ERROR",
                    f"解析资产 SHA-256 校验失败：{asset.asset_id}",
                )
            asset_key = self._asset_key(info.cache_key, asset.sha256, asset.object_key)
            self.client.put_object(
                self.bucket_name,
                asset_key,
                io.BytesIO(asset_payload),
                len(asset_payload),
                content_type=asset.media_type,
            )
        digest_payload = info.sha256.encode("ascii")
        self.client.put_object(
            self.bucket_name,
            object_key,
            io.BytesIO(payload),
            len(payload),
            content_type="application/json",
        )
        self.client.put_object(
            self.bucket_name,
            digest_key,
            io.BytesIO(digest_payload),
            len(digest_payload),
            content_type="text/plain",
        )
        return info.model_copy(update={"object_key": object_key})

    def download_into(
        self,
        cache_key: str,
        store: LocalParseArtifactStore,
    ) -> tuple[ParsedDocument, ParseArtifactInfo] | None:
        object_key, digest_key = self._keys(cache_key)
        try:
            payload = self._read_object(object_key)
            expected = self._read_object(digest_key).decode("ascii").strip()
        except Exception as exc:
            # S3/MinIO 的 not-found 类型跨 SDK 版本不稳定；仅把明确不存在视为 miss。
            code = getattr(exc, "code", "")
            if code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                return None
            raise
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise DocumentParseError(
                "PARSE_ARTIFACT_INTEGRITY_ERROR",
                "MinIO 解析制品 SHA-256 校验失败",
            )
        try:
            document = ParsedDocument.model_validate_json(payload)
        except Exception as exc:
            raise DocumentParseError(
                "PARSE_ARTIFACT_SCHEMA_ERROR",
                "MinIO 解析制品不符合 parsed-document.v1 契约",
            ) from exc
        work_dir = store.artifact_path(cache_key).parent / f"{cache_key}.work"
        for asset in document.assets:
            asset_key = self._asset_key(cache_key, asset.sha256, asset.object_key)
            try:
                asset_payload = self._read_object(asset_key)
            except Exception as exc:
                raise DocumentParseError(
                    "PARSE_ARTIFACT_ASSET_MISSING",
                    f"远端解析资产缺失或不可读取：{asset.asset_id}",
                ) from exc
            if (
                hashlib.sha256(asset_payload).hexdigest() != asset.sha256
                or (asset.size_bytes and len(asset_payload) != asset.size_bytes)
            ):
                raise DocumentParseError(
                    "PARSE_ARTIFACT_ASSET_INTEGRITY_ERROR",
                    f"远端解析资产完整性校验失败：{asset.asset_id}",
                )
            target = self._local_asset_path(work_dir, asset.object_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.parent / f".asset-{uuid.uuid4().hex[:12]}.tmp"
            try:
                temporary.write_bytes(asset_payload)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        local_info = store.save(cache_key, document)
        return document, local_info.model_copy(
            update={"cache_hit": True, "object_key": object_key}
        )
