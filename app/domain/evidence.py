"""TODO-08 块级证据、位置与引用的稳定领域契约。

本模块刻意不依赖检索引擎、LLM 或 HTTP。``Evidence`` 的身份只由发布事实、
稳定 chunk 和内容决定；RRF/rerank 的分数与名次属于一次查询观察，不参与
``evidence_id``，因此重排不会改变引用目标。
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
import math
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
CELL_RANGE_PATTERN = re.compile(
    r"^[A-Z]{1,3}[1-9][0-9]*(?::[A-Z]{1,3}[1-9][0-9]*)?$"
)
EVIDENCE_SCHEMA_VERSION = "evidence.v1"
CITATION_SCHEMA_VERSION = "citation.v1"

Availability = Literal["exact", "derived", "unavailable"]
CoordinateSpace = Literal[
    "none",
    "pdf_points",
    "normalized_page",
    "image_pixels",
    "slide_points",
    "spreadsheet_cells",
    "text_lines",
    "media_timeline",
]
ValidationStatus = Literal["valid", "invalid", "degraded"]
EvidenceModality = Literal["text", "image", "audio", "video"]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_digest(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return f"sha256:{sha256(payload).hexdigest()}"


def _normalized_identifiers(values: list[str], *, field_name: str) -> list[str]:
    normalized = [value.strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} cannot contain an empty identifier")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


class EvidenceSourceIdentity(BaseModel):
    """不可歧义的本地或网页来源身份。

    local 来源依靠控制库外键闭合；web 来源以租户、URL 与抓取内容 hash 固定
    快照，可作为查询期 evidence，但不能成为跨租户共享的隐式公共资源。
    """

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local", "web"] = "local"
    tenant_id: str | None = Field(default=None, max_length=64)
    kb_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=64)
    document_version_id: str | None = Field(default=None, max_length=64)
    index_version_id: str | None = Field(default=None, max_length=64)
    chunk_id: str | None = Field(default=None, max_length=128)
    source_block_ids: list[str] = Field(default_factory=list, max_length=2048)
    source_url: str | None = Field(default=None, max_length=8192)
    content_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    fetched_at: datetime | None = None

    @field_validator(
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "chunk_id",
    )
    @classmethod
    def strip_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("source identifier cannot be blank")
        return value

    @field_validator("source_block_ids")
    @classmethod
    def normalize_block_ids(cls, value: list[str]) -> list[str]:
        return (
            _normalized_identifiers(value, field_name="source_block_ids") if value else []
        )

    @field_validator("source_url")
    @classmethod
    def normalize_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("web source_url must use http or https")
        return value

    @model_validator(mode="after")
    def validate_source_kind(self) -> "EvidenceSourceIdentity":
        local_resource_fields = (
            self.kb_id,
            self.document_id,
            self.document_version_id,
            self.index_version_id,
            self.chunk_id,
        )
        if self.source_type == "local":
            if (
                self.tenant_id is None
                or any(value is None for value in local_resource_fields)
                or not self.source_block_ids
            ):
                raise ValueError(
                    "local source requires tenant/kb/document/version/index/chunk/block IDs"
                )
            if self.source_url is not None or self.content_hash is not None or self.fetched_at:
                raise ValueError("local source cannot contain web snapshot fields")
        else:
            if (
                self.tenant_id is None
                or self.source_url is None
                or self.content_hash is None
            ):
                raise ValueError(
                    "web source requires tenant_id, source_url and content_hash"
                )
            if any(value is not None for value in local_resource_fields):
                raise ValueError("web source cannot contain local control-plane IDs")
            if self.fetched_at is None or self.fetched_at.tzinfo is None:
                raise ValueError("web fetched_at is required and must include a timezone")
        return self


class EvidenceLocationSpan(BaseModel):
    """一个来源块的位置事实。

    ``availability`` 明确区分解析器直接给出的坐标、可靠推导的位置和无法定位。
    bbox 顺序固定为 ``[x0, y0, x1, y1]``；空间坐标统一采用“页面/图像按
    展示旋转后的左上角为原点，x 向右、y 向下”，单位由 ``coordinate_space``
    决定。无法可靠换算到该约定的 parser 必须返回 unavailable。
    """

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1, max_length=128)
    availability: Availability
    coordinate_space: CoordinateSpace = "none"
    page_no: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    slide_no: int | None = Field(default=None, ge=1)
    sheet: str | None = Field(default=None, max_length=255)
    cell_range: str | None = Field(default=None, max_length=64)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    timestamp_start: float | None = Field(default=None, ge=0)
    timestamp_end: float | None = Field(default=None, gt=0)
    frame_no: int | None = Field(default=None, ge=0)
    reason_code: str | None = Field(default=None, max_length=128)
    derivation: str | None = Field(default=None, max_length=255)

    @field_validator("block_id")
    @classmethod
    def normalize_block_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("block_id cannot be blank")
        return value

    @field_validator("sheet", "reason_code", "derivation")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("cell_range")
    @classmethod
    def normalize_cell_range(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().replace("$", "").upper()
        if not CELL_RANGE_PATTERN.fullmatch(normalized):
            raise ValueError("cell_range must use A1 or A1:B2 notation")
        return normalized

    @field_validator("bbox")
    @classmethod
    def normalize_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        normalized = tuple(round(float(item), 6) for item in value)
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError("bbox values must be finite")
        x0, y0, x1, y1 = normalized
        if min(normalized) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must satisfy 0 <= x0 < x1 and 0 <= y0 < y1")
        return normalized

    @model_validator(mode="after")
    def validate_location_semantics(self) -> "EvidenceLocationSpan":
        locator_values = (
            self.page_no,
            self.bbox,
            self.slide_no,
            self.sheet,
            self.cell_range,
            self.line_start,
            self.line_end,
            self.timestamp_start,
            self.timestamp_end,
            self.frame_no,
        )
        has_locator = any(value is not None for value in locator_values)

        if self.availability == "unavailable":
            if has_locator or self.coordinate_space != "none":
                raise ValueError("unavailable location cannot contain coordinates")
            if not self.reason_code:
                raise ValueError("unavailable location requires reason_code")
            if self.derivation:
                raise ValueError("unavailable location cannot contain derivation")
            return self

        if not has_locator:
            raise ValueError(f"{self.availability} location requires at least one locator")
        if self.reason_code:
            raise ValueError("available location cannot contain reason_code")
        if self.availability == "derived" and not self.derivation:
            raise ValueError("derived location requires derivation")
        if self.availability == "exact" and self.derivation:
            raise ValueError("exact location cannot contain derivation")

        if self.page_no is not None and self.slide_no is not None:
            raise ValueError("one location span cannot mix page_no and slide_no")
        if self.sheet is not None and (self.page_no is not None or self.slide_no is not None):
            raise ValueError("one location span cannot mix sheet and page/slide")
        if self.timestamp_start is not None and any(
            value is not None
            for value in (
                self.page_no,
                self.slide_no,
                self.sheet,
                self.line_start,
            )
        ):
            raise ValueError("timeline coordinates cannot mix with document coordinates")
        if self.cell_range is not None and self.sheet is None:
            raise ValueError("cell_range requires sheet")
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")

        if self.bbox is not None:
            if self.page_no is None and self.slide_no is None:
                raise ValueError("bbox requires page_no or slide_no")
            if self.coordinate_space not in {
                "pdf_points",
                "normalized_page",
                "image_pixels",
                "slide_points",
            }:
                raise ValueError("bbox requires an explicit spatial coordinate_space")
            if self.coordinate_space == "slide_points" and self.slide_no is None:
                raise ValueError("slide_points coordinates require slide_no")
            if self.coordinate_space == "normalized_page" and any(
                item > 1 for item in self.bbox
            ):
                raise ValueError("normalized_page bbox values must be within [0, 1]")
        elif self.coordinate_space in {
            "pdf_points",
            "normalized_page",
            "image_pixels",
            "slide_points",
        }:
            raise ValueError("spatial coordinate_space requires bbox")

        if self.cell_range is not None and self.coordinate_space != "spreadsheet_cells":
            raise ValueError("cell_range requires spreadsheet_cells coordinate_space")
        if self.coordinate_space == "spreadsheet_cells" and (
            self.sheet is None or self.cell_range is None
        ):
            raise ValueError("spreadsheet_cells requires sheet and cell_range")
        if self.line_start is not None and self.coordinate_space != "text_lines":
            raise ValueError("line positions require text_lines coordinate_space")
        if self.coordinate_space == "text_lines" and self.line_start is None:
            raise ValueError("text_lines requires line_start")
        if self.timestamp_end is not None and self.timestamp_start is None:
            raise ValueError("timestamp_end requires timestamp_start")
        if (
            self.timestamp_start is not None
            and self.timestamp_end is not None
            and self.timestamp_end <= self.timestamp_start
        ):
            raise ValueError("timestamp_end must be greater than timestamp_start")
        if self.coordinate_space == "media_timeline" and (
            self.timestamp_start is None or self.timestamp_end is None
        ):
            raise ValueError("media_timeline requires start and end timestamps")
        if self.timestamp_start is not None and self.coordinate_space != "media_timeline":
            raise ValueError("timeline positions require media_timeline coordinate_space")
        if self.frame_no is not None and self.coordinate_space != "media_timeline":
            raise ValueError("frame_no requires media_timeline coordinate_space")
        return self


class RetrievalScore(BaseModel):
    """一次查询的可变观察；不参与 Evidence 身份。"""

    model_config = ConfigDict(extra="forbid")

    route: str = Field(min_length=1, max_length=64)
    raw_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_score", "rrf_score", "rerank_score")
    @classmethod
    def validate_finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval scores must be finite")
        return value


def deterministic_evidence_id(
    source: EvidenceSourceIdentity, content_hash: str
) -> str:
    """由发布来源和内容计算稳定 ID，不包含 rank/score/位置展示字段。"""

    if source.source_type == "local":
        payload = {
            "source_type": "local",
            "tenant_id": source.tenant_id,
            "kb_id": source.kb_id,
            "document_id": source.document_id,
            "document_version_id": source.document_version_id,
            "index_version_id": source.index_version_id,
            "chunk_id": source.chunk_id,
            "content_hash": content_hash,
        }
    else:
        payload = {
            "source_type": "web",
            "tenant_id": source.tenant_id,
            "source_url": source.source_url,
            "content_hash": source.content_hash,
        }
    return f"ev_{sha256(canonical_json(payload).encode('utf-8')).hexdigest()[:40]}"


class EvidenceAssetRef(BaseModel):
    """非文本证据所指向的不可变原始资产。

    ``object_key`` 等内部存储地址不进入该对象；调用方只能通过受授权的
    Evidence 预览接口取得内容。``content_hash`` 用于验证资产没有被替换。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=r"^asset[-_][A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
    modality: Literal["image", "audio", "video"]
    media_type: str = Field(min_length=3, max_length=255)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1, le=10 * 1024 * 1024 * 1024)

    @field_validator("media_type")
    @classmethod
    def normalize_media_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "/" not in normalized:
            raise ValueError("media_type must be a MIME type")
        return normalized


class EvidenceDerivation(BaseModel):
    """OCR/VLM/ASR 等派生内容的可复现快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ocr", "vision_description", "table_structure", "asr", "summary"]
    source_asset_id: str = Field(
        pattern=r"^asset[-_][A-Za-z0-9][A-Za-z0-9._-]{5,127}$"
    )
    provider_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    prompt_version: str = Field(min_length=1, max_length=128)
    output_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    external_processing: bool = False

    @field_validator("provider_id", "model_id", "prompt_version")
    @classmethod
    def normalize_derivation_identifiers(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("derivation identifiers cannot be blank")
        return normalized


class Evidence(BaseModel):
    """可复现的块级证据。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["evidence.v1"] = EVIDENCE_SCHEMA_VERSION
    evidence_id: str | None = Field(default=None, pattern=r"^ev_[0-9a-f]{40}$")
    source: EvidenceSourceIdentity
    content: str = Field(min_length=1, max_length=4_000_000)
    content_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    provenance_spans: list[EvidenceLocationSpan] = Field(default_factory=list, max_length=4096)
    heading_path: list[str] = Field(default_factory=list, max_length=128)
    asset_refs: list[str] = Field(default_factory=list, max_length=2048)
    modality: EvidenceModality = "text"
    asset: EvidenceAssetRef | None = None
    derivation: EvidenceDerivation | None = None
    source_label: str = Field(default="", max_length=1024)
    retrieval: RetrievalScore | None = None

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence content cannot be blank")
        return value

    @field_validator("heading_path", "asset_refs")
    @classmethod
    def normalize_string_lists(cls, value: list[str], info) -> list[str]:
        return _normalized_identifiers(value, field_name=info.field_name) if value else []

    @field_validator("source_label")
    @classmethod
    def normalize_source_label(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> "Evidence":
        expected_hash = sha256_digest(self.content)
        if self.content_hash is None:
            self.content_hash = expected_hash
        elif self.content_hash != expected_hash:
            raise ValueError("content_hash does not match evidence content")
        if self.source.source_type == "web" and self.source.content_hash != self.content_hash:
            raise ValueError("web source content_hash must match evidence content")

        expected_id = deterministic_evidence_id(self.source, self.content_hash)
        if self.evidence_id is None:
            self.evidence_id = expected_id
        elif self.evidence_id != expected_id:
            raise ValueError("evidence_id does not match stable source identity")

        if self.source.source_type == "web" and not self.provenance_spans:
            self.provenance_spans = [
                EvidenceLocationSpan(
                    block_id="web",
                    availability="unavailable",
                    reason_code="web_location_unavailable",
                )
            ]
        if not self.provenance_spans:
            raise ValueError("local evidence requires provenance_spans")
        if self.source.source_type == "web" and any(
            span.availability != "unavailable" for span in self.provenance_spans
        ):
            raise ValueError("web evidence locations must be unavailable")

        span_block_ids = [span.block_id for span in self.provenance_spans]
        if len(span_block_ids) != len(set(span_block_ids)):
            raise ValueError("provenance_spans must contain at most one span per block_id")
        if self.source.source_type == "local" and set(span_block_ids) != set(
            self.source.source_block_ids
        ):
            raise ValueError(
                "provenance_spans must represent every source_block_id exactly once"
            )
        if self.modality == "text":
            if self.asset is not None or self.derivation is not None:
                raise ValueError(
                    "text evidence cannot silently carry a multimodal asset or derivation"
                )
        else:
            if self.source.source_type != "local":
                raise ValueError("multimodal evidence currently requires a local source")
            if self.asset is None:
                raise ValueError("non-text evidence requires an immutable asset reference")
            if self.asset.modality != self.modality:
                raise ValueError("evidence modality must match asset modality")
            if self.asset.asset_id not in self.asset_refs:
                raise ValueError("multimodal asset must be present in asset_refs")
            if self.derivation is not None:
                if self.derivation.source_asset_id != self.asset.asset_id:
                    raise ValueError("derivation must point to the evidence asset")
                if self.derivation.output_hash != self.content_hash:
                    raise ValueError(
                        "derived evidence content must match derivation output_hash"
                    )
        return self

    def persistent_copy(self) -> "Evidence":
        """检索分数是查询态，控制库只保存不可变证据事实。"""

        return self.model_copy(update={"retrieval": None}, deep=True)

    def primary_location_span(self) -> EvidenceLocationSpan:
        """统一选择主位置：exact 优先，其次 derived，最后 unavailable。"""

        rank = {"exact": 0, "derived": 1, "unavailable": 2}
        return min(
            enumerate(self.provenance_spans),
            key=lambda item: (rank[item[1].availability], item[0]),
        )[1]


class AnswerSpan(BaseModel):
    """答案中的 ``[start, end)`` Unicode code-point 偏移（Python 字符索引语义）。"""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_half_open_range(self) -> "AnswerSpan":
        if self.end <= self.start:
            raise ValueError("answer span must satisfy start < end")
        return self


def deterministic_citation_id(
    *,
    answer_id: str,
    evidence_id: str,
    answer_span: AnswerSpan,
    quote_hash: str,
) -> str:
    payload = {
        "answer_id": answer_id,
        "evidence_id": evidence_id,
        "answer_span": [answer_span.start, answer_span.end],
        "quote_hash": quote_hash,
    }
    return f"cit_{sha256(canonical_json(payload).encode('utf-8')).hexdigest()[:40]}"


class CitationFact(BaseModel):
    """服务端校验并持久化的引用事实；客户端不能替换 Evidence 内容。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["citation.v1"] = CITATION_SCHEMA_VERSION
    citation_id: str | None = Field(default=None, pattern=r"^cit_[0-9a-f]{40}$")
    answer_id: str = Field(min_length=1, max_length=128)
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    answer_span: AnswerSpan
    source_quote: str = Field(min_length=1, max_length=8000)
    quote_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_label: str = Field(min_length=1, max_length=1024)
    validation_status: ValidationStatus
    reason_codes: list[str] = Field(default_factory=list, max_length=64)
    ordinal: int = Field(default=0, ge=0)

    @field_validator("answer_id", "source_label", "source_quote")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("citation text fields cannot be blank")
        return value

    @field_validator("reason_codes")
    @classmethod
    def normalize_reason_codes(cls, value: list[str]) -> list[str]:
        return _normalized_identifiers(value, field_name="reason_codes") if value else []

    @model_validator(mode="after")
    def validate_status_and_identity(self) -> "CitationFact":
        if sha256_digest(self.source_quote) != self.quote_hash:
            raise ValueError("quote_hash does not match source_quote")
        if self.validation_status == "valid" and self.reason_codes:
            raise ValueError("valid citation cannot contain reason_codes")
        if self.validation_status != "valid" and not self.reason_codes:
            raise ValueError("invalid/degraded citation requires reason_codes")
        expected_id = deterministic_citation_id(
            answer_id=self.answer_id,
            evidence_id=self.evidence_id,
            answer_span=self.answer_span,
            quote_hash=self.quote_hash,
        )
        if self.citation_id is None:
            self.citation_id = expected_id
        elif self.citation_id != expected_id:
            raise ValueError("citation_id does not match citation fact")
        return self


def citation_fact_bundle_hash(facts: Sequence[CitationFact]) -> str:
    """计算有序 CitationFact 集合的规范哈希，并拒绝身份或序号漂移。"""

    citation_ids = [str(item.citation_id or "") for item in facts]
    if len(citation_ids) != len(set(citation_ids)) or any(not item for item in citation_ids):
        raise ValueError("citation fact identities must be present and unique")
    if [item.ordinal for item in facts] != list(range(1, len(facts) + 1)):
        raise ValueError("citation fact ordinals must be contiguous and one-based")
    payload = [item.model_dump(mode="json", exclude_none=False) for item in facts]
    return sha256_digest(canonical_json(payload))
