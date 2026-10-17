"""TODO-23 多模态资产、派生内容、检索配置与评测契约。

本模块只描述事实，不负责调用 Vision/OCR/ASR。原始资产与模型派生文本始终
分层保存；任何派生文本都必须携带 provider/model/prompt 快照，并能回到
tenant、KB、文档版本和页面/时间位置。
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MULTIMODAL_SCHEMA_VERSION = "multimodal.v1"
MediaModality = Literal["image", "audio", "video"]
MediaSourceKind = Literal["document_asset", "query_attachment"]
MediaStatus = Literal["active", "quarantined", "retired", "deleted"]
DataClassification = Literal["public", "internal", "confidential", "restricted"]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be blank")
    return normalized


class MediaLocation(BaseModel):
    """图片页面区域或音视频时间轴位置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_no: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    timestamp_start: float | None = Field(default=None, ge=0)
    timestamp_end: float | None = Field(default=None, gt=0)
    frame_no: int | None = Field(default=None, ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        normalized = tuple(round(float(item), 6) for item in value)
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError("bbox values must be finite")
        x0, y0, x1, y1 = normalized
        if min(normalized) < 0 or max(normalized) > 1 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must be normalized and satisfy x0<x1, y0<y1")
        return normalized

    @model_validator(mode="after")
    def validate_location(self) -> "MediaLocation":
        if self.bbox is not None and self.page_no is None:
            raise ValueError("bbox requires page_no")
        if self.timestamp_end is not None and self.timestamp_start is None:
            raise ValueError("timestamp_end requires timestamp_start")
        if (
            self.timestamp_start is not None
            and self.timestamp_end is not None
            and self.timestamp_end <= self.timestamp_start
        ):
            raise ValueError("timestamp_end must be greater than timestamp_start")
        if self.page_no is not None and self.timestamp_start is not None:
            raise ValueError("one location cannot mix page and timeline coordinates")
        return self


class MediaAsset(BaseModel):
    """不可变媒体资产元数据；不包含字节和密钥。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["multimodal.v1"] = MULTIMODAL_SCHEMA_VERSION
    asset_id: str | None = Field(default=None, pattern=r"^asset_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=64)
    document_version_id: str | None = Field(default=None, max_length=64)
    owner_user_id: str | None = Field(default=None, max_length=128)
    source_kind: MediaSourceKind
    modality: MediaModality
    media_type: str = Field(min_length=3, max_length=255)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=10 * 1024 * 1024 * 1024)
    width: int | None = Field(default=None, ge=1, le=1_000_000)
    height: int | None = Field(default=None, ge=1, le=1_000_000)
    duration_seconds: float | None = Field(default=None, gt=0)
    location: MediaLocation = Field(default_factory=MediaLocation)
    classification: DataClassification = "internal"
    status: MediaStatus = "active"
    object_key: str = Field(min_length=1, max_length=4096)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "owner_user_id",
        "object_key",
    )
    @classmethod
    def normalize_identifiers(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _identifier(value, info.field_name)

    @field_validator("media_type")
    @classmethod
    def normalize_mime(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "/" not in normalized:
            raise ValueError("media_type must be a MIME type")
        return normalized

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone")
        return value

    @model_validator(mode="after")
    def validate_scope_and_identity(self) -> "MediaAsset":
        if self.source_kind == "document_asset":
            if any(
                value is None for value in (self.kb_id, self.document_id, self.document_version_id)
            ):
                raise ValueError("document asset requires KB/document/document-version scope")
            if self.owner_user_id is not None:
                raise ValueError("document asset cannot use query attachment owner")
            if self.modality == "image" and (
                self.location.page_no is None or self.location.bbox is None
            ):
                raise ValueError("document image requires an exact normalized page region")
        else:
            if self.owner_user_id is None:
                raise ValueError("query attachment requires owner_user_id")
            if any(
                value is not None
                for value in (self.kb_id, self.document_id, self.document_version_id)
            ):
                raise ValueError("query attachment cannot masquerade as a published document asset")

        if self.modality == "image":
            if self.width is None or self.height is None:
                raise ValueError("image asset requires width and height")
            if self.duration_seconds is not None or self.location.timestamp_start is not None:
                raise ValueError("image asset cannot contain timeline fields")
        else:
            if self.duration_seconds is None:
                raise ValueError("audio/video asset requires duration_seconds")
            if self.location.timestamp_start is None:
                raise ValueError("audio/video asset requires an exact timeline span")
        payload = {
            "tenant_id": self.tenant_id,
            "kb_id": self.kb_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "owner_user_id": self.owner_user_id,
            "source_kind": self.source_kind,
            "content_hash": self.content_hash,
            "location": self.location.model_dump(mode="json"),
        }
        expected = f"asset_{sha256(_canonical_json(payload).encode()).hexdigest()[:32]}"
        if self.asset_id is None:
            object.__setattr__(self, "asset_id", expected)
        elif self.asset_id != expected:
            raise ValueError("asset_id does not match immutable scope and content")
        return self


class MediaDerivation(BaseModel):
    """由 OCR、Vision、表格模型或 ASR 产生的文本事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["multimodal.v1"] = MULTIMODAL_SCHEMA_VERSION
    derivation_id: str | None = Field(default=None, pattern=r"^drv_[0-9a-f]{32}$")
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    kind: Literal["ocr", "vision_description", "table_structure", "asr", "summary"]
    text: str = Field(min_length=1, max_length=4_000_000)
    content_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    provider_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    prompt_version: str = Field(min_length=1, max_length=128)
    model_snapshot_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    external_processing: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("derivation text cannot be blank")
        return value

    @field_validator("provider_id", "model_id", "prompt_version")
    @classmethod
    def normalized_ids(cls, value: str) -> str:
        return _identifier(value, "derivation identifier")

    @model_validator(mode="after")
    def validate_hashes(self) -> "MediaDerivation":
        expected_hash = f"sha256:{sha256(self.text.encode()).hexdigest()}"
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", expected_hash)
        elif self.content_hash != expected_hash:
            raise ValueError("derivation content_hash does not match text")
        payload = {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "content_hash": expected_hash,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "model_snapshot_hash": self.model_snapshot_hash,
        }
        expected_id = f"drv_{sha256(_canonical_json(payload).encode()).hexdigest()[:32]}"
        if self.derivation_id is None:
            object.__setattr__(self, "derivation_id", expected_id)
        elif self.derivation_id != expected_id:
            raise ValueError("derivation_id does not match derivation snapshot")
        return self


class MediaIngestPolicy(BaseModel):
    """媒体输入与外发的服务端策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_image_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
    )
    max_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=50_000_000, ge=1)
    external_processing_allowed: bool = False
    allowed_external_classifications: tuple[DataClassification, ...] = ("public",)

    @model_validator(mode="after")
    def unique_policy_values(self) -> "MediaIngestPolicy":
        if len(self.allowed_image_types) != len(set(self.allowed_image_types)):
            raise ValueError("allowed image MIME types must be unique")
        if len(self.allowed_external_classifications) != len(
            set(self.allowed_external_classifications)
        ):
            raise ValueError("allowed external classifications must be unique")
        return self


class MultimodalRetrievalProfile(BaseModel):
    """独立发布的实验路线；默认关闭，不能因代码存在而自动执行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["multimodal-profile.v1"] = "multimodal-profile.v1"
    profile_id: str = Field(default="multimodal-default", min_length=1, max_length=128)
    version: int = Field(default=1, ge=1)
    enabled: bool = False
    status: Literal["draft", "published", "retired"] = "draft"
    modalities: tuple[MediaModality, ...] = ("image",)
    route_weights: dict[str, float] = Field(default_factory=lambda: {"text": 1.0, "image": 0.0})
    max_candidates: int = Field(default=20, ge=1, le=1000)
    max_per_asset: int = Field(default=2, ge=1, le=100)
    minimum_paired_gain: float = Field(default=0.02, ge=0.0, le=1.0)

    @field_validator("route_weights")
    @classmethod
    def valid_weights(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for route, raw in value.items():
            weight = float(raw)
            if not math.isfinite(weight) or not 0 <= weight <= 100:
                raise ValueError("multimodal route weights must be finite in [0, 100]")
            normalized[_identifier(str(route), "route")] = weight
        if not any(weight > 0 for weight in normalized.values()):
            raise ValueError("at least one multimodal route weight must be positive")
        return normalized

    @model_validator(mode="after")
    def published_profile_gate(self) -> "MultimodalRetrievalProfile":
        if self.enabled and self.status != "published":
            raise ValueError("enabled multimodal profile must be published")
        if len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must be unique")
        return self


class MultimodalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    modality: MediaModality
    text_score: float = 0.0
    modality_score: float = 0.0
    final_score: float = 0.0
    rank: int | None = Field(default=None, ge=1)

    @field_validator("text_score", "modality_score", "final_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("candidate scores must be finite")
        return float(value)


class MultimodalEvaluationReport(BaseModel):
    """进入默认路线前的配对评测门禁结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=128)
    baseline_recall: float = Field(ge=0.0, le=1.0)
    candidate_recall: float = Field(ge=0.0, le=1.0)
    baseline_citation_location_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_citation_location_accuracy: float = Field(ge=0.0, le=1.0)
    latency_delta_ms: float
    cost_delta: float = Field(ge=0.0)
    minimum_gain: float = Field(default=0.02, ge=0.0, le=1.0)
    passed: bool | None = None

    @model_validator(mode="after")
    def calculate_gate(self) -> "MultimodalEvaluationReport":
        values = (
            self.latency_delta_ms,
            self.cost_delta,
            self.baseline_recall,
            self.candidate_recall,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("evaluation numbers must be finite")
        expected = (
            self.candidate_recall - self.baseline_recall >= self.minimum_gain
            and self.candidate_citation_location_accuracy
            >= self.baseline_citation_location_accuracy
        )
        if self.passed is None:
            object.__setattr__(self, "passed", expected)
        elif self.passed != expected:
            raise ValueError("passed must match the deterministic quality gate")
        return self


__all__ = [
    "MULTIMODAL_SCHEMA_VERSION",
    "MediaAsset",
    "MediaDerivation",
    "MediaIngestPolicy",
    "MediaLocation",
    "MultimodalCandidate",
    "MultimodalEvaluationReport",
    "MultimodalRetrievalProfile",
]
