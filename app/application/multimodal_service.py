"""TODO-23 安全媒体摄取、派生链、Evidence 转换与跨模态排序。"""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import PurePosixPath
from typing import Protocol
import hashlib
import math
import warnings

from PIL import Image, UnidentifiedImageError

from app.domain.evidence import (
    Evidence,
    EvidenceAssetRef,
    EvidenceDerivation,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.domain.multimodal import (
    MediaAsset,
    MediaDerivation,
    MediaIngestPolicy,
    MediaLocation,
    MultimodalCandidate,
    MultimodalRetrievalProfile,
)


class MediaValidationError(ValueError):
    """媒体字节、类型或资源上限不满足服务端策略。"""


class MediaPolicyError(PermissionError):
    """数据分级或外发策略拒绝模型派生。"""


class AssetObjectStore(Protocol):
    """只接收内容寻址对象键；实现必须保证 tenant 隔离。"""

    def put(self, object_key: str, content: bytes, *, media_type: str) -> None: ...

    def get(self, object_key: str) -> bytes: ...

    def delete(self, object_key: str) -> None: ...


class InMemoryAssetObjectStore:
    """确定性测试/本地预览存储，不在进程重启后声称持久。"""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}

    def put(self, object_key: str, content: bytes, *, media_type: str) -> None:
        existing = self._objects.get(object_key)
        if existing is not None and existing[0] != content:
            raise MediaValidationError("content-addressed object key collision")
        self._objects[object_key] = (bytes(content), media_type)

    def get(self, object_key: str) -> bytes:
        try:
            return self._objects[object_key][0]
        except KeyError as exc:
            raise KeyError("media object not found") from exc

    def delete(self, object_key: str) -> None:
        self._objects.pop(object_key, None)


_SIGNATURES = {
    "image/png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "image/gif": lambda data: data.startswith((b"GIF87a", b"GIF89a")),
    "image/bmp": lambda data: data.startswith(b"BM"),
    "image/tiff": lambda data: data.startswith((b"II*\x00", b"MM\x00*")),
    "image/webp": lambda data: data.startswith(b"RIFF") and data[8:12] == b"WEBP",
}
_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def sniff_image(
    content: bytes,
    *,
    claimed_media_type: str,
    policy: MediaIngestPolicy,
) -> tuple[str, int, int]:
    """用魔数和 Pillow 双重验证图片，不信任文件扩展名或请求 MIME。"""

    if not content:
        raise MediaValidationError("empty image is not allowed")
    if len(content) > policy.max_image_bytes:
        raise MediaValidationError("image exceeds max_image_bytes")
    detected = next(
        (media_type for media_type, check in _SIGNATURES.items() if check(content)),
        None,
    )
    claimed = claimed_media_type.strip().lower()
    if detected is None or detected not in policy.allowed_image_types:
        raise MediaValidationError("unsupported or unsafe image signature")
    if claimed != detected:
        raise MediaValidationError("claimed MIME does not match image signature")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise MediaValidationError("image dimensions must be positive")
                if width * height > policy.max_image_pixels:
                    raise MediaValidationError("image exceeds max_image_pixels")
                image.verify()
    except MediaValidationError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise MediaValidationError("image payload is corrupt or unsafe") from exc
    return detected, width, height


class MultimodalService:
    """媒体能力入口；任何模型调用必须由上层传入已完成的派生结果。"""

    def __init__(
        self,
        *,
        policy: MediaIngestPolicy | None = None,
        object_store: AssetObjectStore | None = None,
    ) -> None:
        self.policy = policy or MediaIngestPolicy()
        self.object_store = object_store or InMemoryAssetObjectStore()

    def ingest_image(
        self,
        content: bytes,
        *,
        claimed_media_type: str,
        tenant_id: str,
        source_kind: str,
        classification: str = "internal",
        kb_id: str | None = None,
        document_id: str | None = None,
        document_version_id: str | None = None,
        owner_user_id: str | None = None,
        page_no: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> MediaAsset:
        media_type, width, height = sniff_image(
            content,
            claimed_media_type=claimed_media_type,
            policy=self.policy,
        )
        digest = hashlib.sha256(content).hexdigest()
        suffix = _EXTENSIONS[media_type]
        safe_tenant = tenant_id.strip().replace("/", "_").replace("\\", "_")
        object_key = str(
            PurePosixPath("multimodal") / safe_tenant / digest[:2] / f"{digest}{suffix}"
        )
        asset = MediaAsset(
            tenant_id=tenant_id,
            kb_id=kb_id,
            document_id=document_id,
            document_version_id=document_version_id,
            owner_user_id=owner_user_id,
            source_kind=source_kind,
            modality="image",
            media_type=media_type,
            content_hash=f"sha256:{digest}",
            size_bytes=len(content),
            width=width,
            height=height,
            location=MediaLocation(page_no=page_no, bbox=bbox),
            classification=classification,
            object_key=object_key,
        )
        self.object_store.put(object_key, content, media_type=media_type)
        return asset

    def record_derivation(
        self,
        asset: MediaAsset,
        *,
        kind: str,
        text: str,
        provider_id: str,
        model_id: str,
        prompt_version: str,
        model_snapshot_hash: str,
        confidence: float | None = None,
        external_processing: bool = False,
    ) -> MediaDerivation:
        if asset.status != "active":
            raise MediaPolicyError("only active media can produce a derivation")
        if external_processing and (
            not self.policy.external_processing_allowed
            or asset.classification not in self.policy.allowed_external_classifications
        ):
            raise MediaPolicyError(
                "external media processing is forbidden by classification policy"
            )
        return MediaDerivation(
            asset_id=asset.asset_id,
            kind=kind,
            text=text,
            provider_id=provider_id,
            model_id=model_id,
            prompt_version=prompt_version,
            model_snapshot_hash=model_snapshot_hash,
            confidence=confidence,
            external_processing=external_processing,
        )

    @staticmethod
    def query_attachment_context(
        asset: MediaAsset,
        derivation: MediaDerivation,
        *,
        tenant_id: str,
        user_id: str,
        max_chars: int = 8_000,
    ) -> dict:
        """为 QueryPlan 提供有限派生文本，不把附件伪造成 KB Evidence。"""

        if asset.source_kind != "query_attachment":
            raise MediaValidationError("query attachment context requires attachment asset")
        if asset.tenant_id != tenant_id or asset.owner_user_id != user_id:
            raise MediaPolicyError("query attachment scope mismatch")
        if derivation.asset_id != asset.asset_id:
            raise MediaValidationError("derivation does not belong to attachment")
        bounded = derivation.text[:max_chars]
        return {
            "schema_version": "query-attachment-context.v1",
            "asset_id": asset.asset_id,
            "modality": asset.modality,
            "text": bounded,
            "truncated": len(bounded) != len(derivation.text),
            "derivation_id": derivation.derivation_id,
            "classification": asset.classification,
        }

    @staticmethod
    def to_evidence(
        asset: MediaAsset,
        derivation: MediaDerivation,
        *,
        index_version_id: str,
        chunk_id: str,
        source_block_id: str,
        source_label: str,
    ) -> Evidence:
        """把已发布文档媒体转换为可由 AnswerEnvelope 引用的 Evidence。"""

        if asset.source_kind != "document_asset":
            raise MediaValidationError("only published document assets become evidence")
        if derivation.asset_id != asset.asset_id:
            raise MediaValidationError("derivation does not belong to asset")
        if any(
            value is None for value in (asset.kb_id, asset.document_id, asset.document_version_id)
        ):
            raise MediaValidationError("document asset scope is incomplete")
        if asset.location.page_no is not None:
            span = EvidenceLocationSpan(
                block_id=source_block_id,
                availability="exact",
                coordinate_space="normalized_page",
                page_no=asset.location.page_no,
                bbox=asset.location.bbox,
            )
        else:
            span = EvidenceLocationSpan(
                block_id=source_block_id,
                availability="exact",
                coordinate_space="media_timeline",
                timestamp_start=asset.location.timestamp_start,
                timestamp_end=asset.location.timestamp_end,
                frame_no=asset.location.frame_no,
            )
        return Evidence(
            source=EvidenceSourceIdentity(
                source_type="local",
                tenant_id=asset.tenant_id,
                kb_id=asset.kb_id,
                document_id=asset.document_id,
                document_version_id=asset.document_version_id,
                index_version_id=index_version_id,
                chunk_id=chunk_id,
                source_block_ids=[source_block_id],
            ),
            content=derivation.text,
            provenance_spans=[span],
            asset_refs=[asset.asset_id],
            modality=asset.modality,
            asset=EvidenceAssetRef(
                asset_id=asset.asset_id,
                modality=asset.modality,
                media_type=asset.media_type,
                content_hash=asset.content_hash,
                size_bytes=asset.size_bytes,
            ),
            derivation=EvidenceDerivation(
                kind=derivation.kind,
                source_asset_id=asset.asset_id,
                provider_id=derivation.provider_id,
                model_id=derivation.model_id,
                prompt_version=derivation.prompt_version,
                output_hash=derivation.content_hash,
                confidence=derivation.confidence,
                external_processing=derivation.external_processing,
            ),
            source_label=source_label,
        )


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_multimodal_candidates(
    candidates: list[MultimodalCandidate],
    profile: MultimodalRetrievalProfile,
) -> tuple[list[MultimodalCandidate], dict]:
    """确定性跨模态校准与配额；关闭时不产生隐藏候选。"""

    if not profile.enabled:
        return [], {
            "schema_version": "multimodal-ranking.v1",
            "enabled": False,
            "reason": "profile_disabled",
            "input_count": len(candidates),
            "selected_count": 0,
        }
    if profile.status != "published":
        raise MediaPolicyError("online multimodal ranking requires published profile")
    allowed = [item for item in candidates if item.modality in profile.modalities]
    text_norm = _min_max([item.text_score for item in allowed])
    modality_norm = _min_max([item.modality_score for item in allowed])
    weighted: list[MultimodalCandidate] = []
    for item, text_score, modality_score in zip(allowed, text_norm, modality_norm, strict=True):
        modality_weight = profile.route_weights.get(item.modality, 0.0)
        text_weight = profile.route_weights.get("text", 0.0)
        denominator = text_weight + modality_weight
        final = (
            (text_score * text_weight + modality_score * modality_weight) / denominator
            if denominator > 0
            else 0.0
        )
        weighted.append(item.model_copy(update={"final_score": round(final, 8)}))
    weighted.sort(key=lambda item: (-item.final_score, item.evidence_id, item.asset_id))
    per_asset: defaultdict[str, int] = defaultdict(int)
    selected: list[MultimodalCandidate] = []
    dropped_by_quota = 0
    for item in weighted:
        if per_asset[item.asset_id] >= profile.max_per_asset:
            dropped_by_quota += 1
            continue
        per_asset[item.asset_id] += 1
        selected.append(item.model_copy(update={"rank": len(selected) + 1}))
        if len(selected) >= profile.max_candidates:
            break
    diagnostics = {
        "schema_version": "multimodal-ranking.v1",
        "enabled": True,
        "input_count": len(candidates),
        "allowed_count": len(allowed),
        "selected_count": len(selected),
        "dropped_by_modality": len(candidates) - len(allowed),
        "dropped_by_asset_quota": dropped_by_quota,
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
    }
    return selected, diagnostics


__all__ = [
    "AssetObjectStore",
    "InMemoryAssetObjectStore",
    "MediaPolicyError",
    "MediaValidationError",
    "MultimodalService",
    "rank_multimodal_candidates",
    "sniff_image",
]
