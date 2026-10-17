"""Evidence 绑定图片资产的可信解析与同源预览身份。

浏览器不能把正文里的 Markdown URL、模型输出或任意 MinIO URL 当作授权事实。
本模块只接受已经持久化在 ``Evidence.asset_refs`` 中、位于本服务图片前缀下的
光栅图对象，并把对象键投影为不泄露存储路径的稳定图片 ID。真正读取时仍需由
HTTP 层重新校验当前 principal、KB entitlement 与 index manifest。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from app.infra.object_storage.minio_gateway import minio_gateway
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import IndexBuildManifestModel
from sqlalchemy import select


_RASTER_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@dataclass(frozen=True, slots=True)
class EvidenceImageAsset:
    """经路径白名单校验的 Evidence 图片资产。"""

    image_id: str
    object_key: str
    media_type: str

    def preview_url(self, evidence_id: str) -> str:
        return f"/api/v1/evidence/{evidence_id}/images/{self.image_id}"


def _safe_relative_parts(value: str) -> tuple[str, ...] | None:
    decoded = unquote(value).replace("\\", "/").strip("/")
    parts = PurePosixPath(decoded).parts
    # 旧索引是 <document>/<file>，新索引是
    # <document_version>/<index_version>/<file>；两者都必须至少包含目录与文件。
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    if any("/" in part or "\\" in part for part in parts):
        return None
    return tuple(parts)


def trusted_image_object_key(raw_ref: object) -> tuple[str, str] | None:
    """把受信 ``asset_ref`` 解析为 ``(object_key, media_type)``。

    接受导入链生成的 ``imagetemps-...`` 身份以及本服务旧版 MinIO 直链；外部
    域名、用户信息、路径穿越、SVG/HTML 和非图片对象一律拒绝。
    """

    raw = str(raw_ref or "").strip()
    if not raw:
        return None

    image_prefix = minio_gateway.image_dir.strip("/")
    relative: str
    if raw.startswith("imagetemps-"):
        relative = raw[len("imagetemps-") :]
    else:
        parsed = urlsplit(raw)
        configured = urlsplit(minio_gateway.build_image_url("__scope__", "asset.png"))
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.scheme != configured.scheme
            or parsed.netloc.lower() != configured.netloc.lower()
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        decoded_path = unquote(parsed.path).replace("\\", "/")
        prefix = f"/{minio_gateway.bucket_name}/{image_prefix}/"
        if not decoded_path.startswith(prefix):
            return None
        relative = decoded_path[len(prefix) :]

    parts = _safe_relative_parts(relative)
    if parts is None:
        return None
    suffix = PurePosixPath(parts[-1]).suffix.lower()
    media_type = _RASTER_MEDIA_TYPES.get(suffix)
    if media_type is None:
        return None
    object_key = "/".join((image_prefix, *parts))
    return object_key, media_type


def evidence_image_assets(evidence: dict) -> list[EvidenceImageAsset]:
    """按 Evidence 原始顺序返回去重后的可信图片资产。"""

    evidence_id = str(evidence.get("evidence_id") or "").strip()
    if not evidence_id:
        return []
    refs = evidence.get("asset_refs")
    if not isinstance(refs, list):
        return []

    output: list[EvidenceImageAsset] = []
    seen: set[str] = set()
    for raw_ref in refs:
        parsed = trusted_image_object_key(raw_ref)
        # ParsedDocument 的 block ref 会保留为 ``images/<hash>.jpg``；实际上传对象
        # 则固定在该 Evidence 的 document_version/index_version 目录。只有来源身份
        # 完整时才允许做这一步确定性投影，最终读取还必须通过 manifest 精确成员校验。
        if parsed is None:
            raw_parts = _safe_relative_parts(str(raw_ref or ""))
            source = evidence.get("source") or {}
            document_version_id = str(source.get("document_version_id") or "")
            index_version_id = str(source.get("index_version_id") or "")
            if (
                source.get("source_type") == "local"
                and raw_parts is not None
                and len(raw_parts) == 2
                and raw_parts[0] == "images"
                and document_version_id
                and index_version_id
                and "/" not in document_version_id
                and "\\" not in document_version_id
                and "/" not in index_version_id
                and "\\" not in index_version_id
            ):
                parsed = trusted_image_object_key(
                    f"imagetemps-{document_version_id}/{index_version_id}/{raw_parts[1]}"
                )
        if parsed is None:
            continue
        object_key, media_type = parsed
        if object_key in seen:
            continue
        seen.add(object_key)
        digest = sha256(f"{evidence_id}\0{object_key}".encode("utf-8")).hexdigest()[:32]
        output.append(
            EvidenceImageAsset(
                image_id=f"img_{digest}",
                object_key=object_key,
                media_type=media_type,
            )
        )
    return output


def _validated_manifest_object_keys(evidence: dict) -> frozenset[str]:
    """返回 Evidence 固定版本的 validated manifest 对象键集合。"""

    source = evidence.get("source") if isinstance(evidence, dict) else None
    if not isinstance(source, dict) or source.get("source_type") != "local":
        return frozenset()
    document_version_id = str(source.get("document_version_id") or "")
    index_version_id = str(source.get("index_version_id") or "")
    if not document_version_id or not index_version_id:
        return frozenset()
    with session_scope() as session:
        rows = session.scalars(
            select(IndexBuildManifestModel).where(
                IndexBuildManifestModel.document_version_id == document_version_id,
                IndexBuildManifestModel.index_version_id == index_version_id,
                IndexBuildManifestModel.validation_status == "validated",
            )
        ).all()
    return frozenset(
        str(object_key)
        for row in rows
        for object_key in (row.object_keys or ())
        if str(object_key).strip()
    )


def manifested_evidence_image_assets(evidence: dict) -> list[EvidenceImageAsset]:
    """只返回已经进入该 Evidence 固定版本 validated manifest 的图片。

    ``asset_refs`` 可能同时保留解析期 ``images/...``、旧版 ``imagetemps`` 和
    新版版本化身份。解析期引用不等于上传成功事实；若在此处直接投影，会生成
    一个路由读取时必然被 manifest 防线拒绝的“伪可用”URL。发布给客户端前先
    与 manifest 取交集，读取端仍会再次执行同一校验以抵御伪造 ID 和权限变化。
    """

    manifest_object_keys = _validated_manifest_object_keys(evidence)
    if not manifest_object_keys:
        return []
    return [
        asset
        for asset in evidence_image_assets(evidence)
        if asset.object_key in manifest_object_keys
    ]


def evidence_image_projection(evidence: dict) -> list[dict[str, str]]:
    """返回可安全放进 Evidence 预览 JSON 的站内图片摘要。"""

    evidence_id = str(evidence.get("evidence_id") or "")
    return [
        {
            "image_id": asset.image_id,
            "url": asset.preview_url(evidence_id),
            "media_type": asset.media_type,
        }
        for asset in manifested_evidence_image_assets(evidence)
    ]


def image_asset_is_manifested(evidence: dict, asset: EvidenceImageAsset) -> bool:
    """确认对象键属于该 Evidence 固定的文档/索引版本 manifest。"""

    return asset.object_key in _validated_manifest_object_keys(evidence)


__all__ = [
    "EvidenceImageAsset",
    "evidence_image_assets",
    "evidence_image_projection",
    "image_asset_is_manifested",
    "manifested_evidence_image_assets",
    "trusted_image_object_key",
]
