"""TODO-28 确定性候选治理：校准、融合、去重、多样性与诊断。

本模块刻意不依赖 LangGraph、Milvus、SQL 或 HTTP。调用方只需要把各召回路线的
候选列表交给 :func:`fuse_route_results`；模块会逐条校验，保留 Evidence 身份和
所有原始分数观察，并返回有界、可解释、可重复的排序结果。

第一版同时提供两个可比较的基线：

* ``rrf``：只使用路线内排名，不假设异构 raw score 可比；
* ``weighted``：先按路线执行版本化校准，再进行带权融合。

近重复只会在同一 source type（且本地证据在同一 tenant/KB）内合并，Web 快照
不会因为正文相似而与本地 Evidence 合并。所有 tie-break 都使用稳定身份，不依赖
``dict`` 插入顺序或 Python 对象地址。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from itertools import islice
import json
import math
import re
from statistics import fmean, pstdev
from typing import Any, Iterable, Literal, Mapping, Sequence
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FusionMethod = Literal["rrf", "weighted"]
CalibrationMethod = Literal["rank", "min_max", "z_score", "range"]
ScoreDirection = Literal["higher_is_better", "lower_is_better"]
DecisionStatus = Literal["selected", "dropped", "invalid"]

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}
_SCORE_FIELDS = (
    "raw_score",
    "dense_score",
    "sparse_score",
    "bm25_score",
    "provider_score",
    "retrieval_score",
    "rrf_score",
    "rerank_score",
    "score",
)
_LOGICAL_BLOCK_OCCURRENCE = re.compile(
    r"^(block|paragraph|para|line|row|cell|slide|sheet)[_-]?0*(\d+)$",
    re.IGNORECASE,
)


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _normalized_hash(value: str) -> str:
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("content_hash must be a SHA-256 digest")
    return f"sha256:{digest}"


def _content_hash(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def _stable_digest(value: Any, *, length: int = 32) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:length]


def canonicalize_web_uri(uri: str) -> str:
    """生成可重复的 Web URL 身份，同时移除常见追踪参数和 fragment。"""

    raw = uri.strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("web candidate uri must be an absolute HTTP(S) URL")
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    authority = hostname
    if port is not None and not default_port:
        authority = f"{hostname}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((parsed.scheme.lower(), authority, path, query, ""))


class ScoreSemantics(BaseModel):
    """一个 raw score 的可解释语义，不假设不同 metric 天然可比。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(default="unknown", min_length=1, max_length=128)
    direction: ScoreDirection = "higher_is_better"
    range_min: float | None = None
    range_max: float | None = None
    model_version: str | None = Field(default=None, max_length=256)

    @field_validator("range_min", "range_max")
    @classmethod
    def finite_range(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score semantic range must be finite")
        return value

    @model_validator(mode="after")
    def ordered_range(self) -> "ScoreSemantics":
        if (
            self.range_min is not None
            and self.range_max is not None
            and self.range_max <= self.range_min
        ):
            raise ValueError("score semantic range_max must be greater than range_min")
        return self


class RouteObservation(BaseModel):
    """候选在某条路线/子查询中的原始观察。

    ``raw_scores`` 永不被校准值、RRF 或 rerank 覆盖。校准结果只出现在最终
    ``RankedEvidence.calibrated_scores`` 中。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(min_length=1, max_length=128)
    sub_query_id: str | None = Field(default=None, max_length=128)
    route_rank: int = Field(ge=1, le=1_000_000)
    raw_scores: dict[str, float] = Field(default_factory=dict)
    score_semantics: dict[str, ScoreSemantics] = Field(default_factory=dict)
    primary_score_key: str | None = Field(default=None, max_length=128)
    retrieved_at: datetime | None = None

    @field_validator("route_id")
    @classmethod
    def safe_route_id(cls, value: str) -> str:
        normalized = value.strip()
        if not _SAFE_NAME.fullmatch(normalized):
            raise ValueError("route_id contains unsupported characters")
        return normalized

    @field_validator("sub_query_id")
    @classmethod
    def safe_sub_query_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or not _SAFE_NAME.fullmatch(normalized):
            raise ValueError("sub_query_id contains unsupported characters")
        return normalized

    @field_validator("raw_scores")
    @classmethod
    def finite_raw_scores(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, score in value.items():
            name = str(key).strip()
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError("raw score name contains unsupported characters")
            normalized[name] = _finite_float(score, field_name=f"raw_scores.{name}")
        return normalized

    @field_validator("score_semantics")
    @classmethod
    def safe_semantics(cls, value: dict[str, ScoreSemantics]) -> dict[str, ScoreSemantics]:
        for key in value:
            if not _SAFE_NAME.fullmatch(str(key)):
                raise ValueError("score semantics name contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_primary_score(self) -> "RouteObservation":
        if self.primary_score_key is not None:
            if self.primary_score_key not in self.raw_scores:
                raise ValueError("primary_score_key must exist in raw_scores")
        if self.retrieved_at is not None and self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must include a timezone")
        return self

    def stable_id(self) -> str:
        return "obs_" + _stable_digest(self.model_dump(mode="json"), length=24)

    def primary_score(self) -> tuple[str, float, ScoreSemantics] | None:
        if not self.raw_scores:
            return None
        if self.primary_score_key is not None:
            key = self.primary_score_key
        else:
            key = next(
                (name for name in _SCORE_FIELDS if name in self.raw_scores),
                sorted(self.raw_scores)[0],
            )
        return (
            key,
            self.raw_scores[key],
            self.score_semantics.get(key, ScoreSemantics()),
        )


class RetrievalCandidate(BaseModel):
    """融合层统一候选，保留稳定 Evidence/source 身份和完整路线观察。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str | None = Field(default=None, pattern=r"^ev_[0-9a-f]{40}$")
    source_type: str = Field(default="local", min_length=1, max_length=64)
    tenant_id: str | None = Field(default=None, max_length=64)
    kb_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=128)
    document_version_id: str | None = Field(default=None, max_length=128)
    index_version_id: str | None = Field(default=None, max_length=128)
    chunk_id: str | None = Field(default=None, max_length=128)
    content: str = Field(min_length=1, max_length=4_000_000)
    content_hash: str | None = None
    title: str | None = Field(default=None, max_length=2048)
    uri: str | None = Field(default=None, max_length=8192)
    canonical_uri: str | None = Field(default=None, max_length=8192)
    page_no: int | None = Field(default=None, ge=1, le=10_000_000)
    bbox: tuple[float, float, float, float] | None = None
    source_block_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=2048)
    provenance_spans: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        max_length=4096,
    )
    heading_path: tuple[str, ...] = Field(default_factory=tuple, max_length=128)
    asset_refs: tuple[str, ...] = Field(default_factory=tuple, max_length=2048)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: tuple[RouteObservation, ...] = Field(min_length=1, max_length=512)
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_payload: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_stable_content_fields(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        content = str(data.get("content") or "")
        if content:
            raw_hash = data.get("content_hash")
            data["content_hash"] = (
                _normalized_hash(str(raw_hash)) if raw_hash else _content_hash(content)
            )
        source_type = str(data.get("source_type") or "local").strip().lower()
        data["source_type"] = source_type
        uri = data.get("uri")
        if source_type == "web" and uri:
            data["canonical_uri"] = canonicalize_web_uri(str(data.get("canonical_uri") or uri))
        return data

    @field_validator(
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "chunk_id",
        "title",
        "uri",
        "canonical_uri",
    )
    @classmethod
    def strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("source_type")
    @classmethod
    def safe_source_type(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("source_type contains unsupported characters")
        return value

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate content cannot be blank")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str | None) -> str | None:
        return _normalized_hash(value) if value is not None else None

    @field_validator("bbox")
    @classmethod
    def finite_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not all(math.isfinite(item) for item in value):
            raise ValueError("bbox values must be finite")
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("bbox must have positive width and height")
        return value

    @model_validator(mode="after")
    def validate_source_identity(self) -> "RetrievalCandidate":
        if self.content_hash != _content_hash(self.content):
            raise ValueError("content_hash does not match candidate content")
        if self.source_type == "web" and not self.canonical_uri:
            raise ValueError("web candidate requires an absolute uri")
        return self

    def identity_quality(self) -> Literal["evidence", "source", "surrogate"]:
        if self.evidence_id:
            return "evidence"
        if self.source_type == "web" and self.canonical_uri:
            return "source"
        if self.chunk_id and self.document_version_id and self.index_version_id:
            return "source"
        return "surrogate"

    def identity_key(self) -> str:
        if self.evidence_id:
            return f"evidence:{self.evidence_id}"
        if self.source_type == "web":
            return f"web:{self.canonical_uri}:{self.content_hash}"
        if self.chunk_id and self.document_version_id and self.index_version_id:
            return ":".join(
                (
                    "local",
                    self.tenant_id or "-",
                    self.kb_id or "-",
                    self.document_version_id,
                    self.index_version_id,
                    self.chunk_id,
                )
            )
        payload = {
            "source_type": self.source_type,
            "tenant_id": self.tenant_id,
            "kb_id": self.kb_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "index_version_id": self.index_version_id,
            "chunk_id": self.chunk_id,
            "canonical_uri": self.canonical_uri,
            "content_hash": self.content_hash,
        }
        return f"surrogate:{_stable_digest(payload)}"

    def ranking_tiebreak_key(self) -> str:
        """返回跨索引重建稳定的语义/来源排序键。

        Evidence、document version、index version 和 chunk id 都属于一次发布或
        重建的物理身份，不能用于同分裁决。内容哈希、逻辑 document、页码/坐标
        与 Web canonical URI 在同一来源重建时稳定，可用于确定性排序。
        """

        logical_occurrences: list[tuple[str, int]] = []
        for block_id in self.source_block_ids:
            matched = _LOGICAL_BLOCK_OCCURRENCE.fullmatch(block_id.strip())
            if matched:
                logical_occurrences.append(
                    (matched.group(1).casefold(), int(matched.group(2)))
                )
        stable_spans = []
        for span in self.provenance_spans:
            stable_spans.append(
                {
                    key: span.get(key)
                    for key in (
                        "availability",
                        "coordinate_space",
                        "page_no",
                        "bbox",
                        "slide_no",
                        "sheet",
                        "cell_range",
                        "line_start",
                        "line_end",
                        "timestamp_start",
                        "timestamp_end",
                        "frame_no",
                    )
                    if span.get(key) is not None
                }
            )
        payload = {
            "source_type": self.source_type,
            "tenant_id": self.tenant_id,
            "kb_id": self.kb_id,
            "document_id": self.document_id,
            "canonical_uri": self.canonical_uri,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "heading_path": self.heading_path,
            "content_hash": self.content_hash,
            # Parser block ordinals and locators express logical occurrence without
            # depending on evidence/index/version/chunk identities.
            "logical_block_occurrences": sorted(set(logical_occurrences)),
            "stable_provenance_spans": stable_spans,
        }
        return f"rank:{_stable_digest(payload, length=64)}"

    def document_group(self) -> str:
        if self.source_type == "web":
            return f"web:{self.canonical_uri}"
        return f"local:{self.document_id or self.document_version_id or self.identity_key()}"


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CalibrationMethod = "rank"
    version: str = Field(default="rank.v1", min_length=1, max_length=128)


class FusionConfig(BaseModel):
    """一次融合运行的不可变配置快照。

    ``max_input_candidates`` 同时限制 CPU、近重复比较和诊断体积。超限输入会被
    显式计数，不会继续消费潜在无限 iterable。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["retrieval-fusion.v1"] = "retrieval-fusion.v1"
    method: FusionMethod = "rrf"
    output_k: int = Field(default=10, ge=1, le=1000)
    max_routes: int = Field(default=32, ge=1, le=256)
    max_input_candidates: int = Field(default=2000, ge=1, le=20_000)
    rrf_k: int = Field(default=60, ge=1, le=100_000)
    default_route_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    route_weights: dict[str, float] = Field(default_factory=dict)
    default_calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    route_calibrations: dict[str, CalibrationConfig] = Field(default_factory=dict)
    observation_aggregation: Literal["max", "mean"] = "max"
    near_duplicate_threshold: float | None = Field(default=0.94, ge=0.0, le=1.0)
    near_duplicate_min_chars: int = Field(default=40, ge=1, le=100_000)
    max_similarity_chars: int = Field(default=12_000, ge=256, le=100_000)
    max_near_duplicate_comparisons: int = Field(
        default=50_000,
        ge=0,
        le=2_000_000,
    )
    max_per_document: int = Field(default=3, ge=1, le=1000)
    source_quotas: dict[str, int] = Field(default_factory=dict)
    mmr_lambda: float = Field(default=0.8, ge=0.0, le=1.0)
    subquery_coverage_bonus_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    subquery_coverage_bonus_cap: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("default_route_weight")
    @classmethod
    def finite_default_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("default_route_weight must be finite")
        return value

    @field_validator("route_weights")
    @classmethod
    def valid_route_weights(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, raw_weight in value.items():
            route = str(key).strip()
            if not _SAFE_NAME.fullmatch(route):
                raise ValueError("route weight name contains unsupported characters")
            weight = _finite_float(raw_weight, field_name=f"route_weights.{route}")
            if not 0 <= weight <= 100:
                raise ValueError("route weights must be within [0, 100]")
            normalized[route] = weight
        return normalized

    @field_validator("route_calibrations")
    @classmethod
    def valid_calibration_routes(
        cls, value: dict[str, CalibrationConfig]
    ) -> dict[str, CalibrationConfig]:
        for route in value:
            if not _SAFE_NAME.fullmatch(str(route)):
                raise ValueError("route calibration name contains unsupported characters")
        return value

    @field_validator("source_quotas")
    @classmethod
    def valid_source_quotas(cls, value: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for key, raw_quota in value.items():
            source = str(key).strip()
            if not _SAFE_NAME.fullmatch(source):
                raise ValueError("source quota name contains unsupported characters")
            if isinstance(raw_quota, bool):
                raise ValueError("source quotas must be integers within [1, 1000]")
            numeric_quota = float(raw_quota)
            if not numeric_quota.is_integer() or not 1 <= numeric_quota <= 1000:
                raise ValueError("source quotas must be integers within [1, 1000]")
            normalized[source] = int(numeric_quota)
        return normalized

    @model_validator(mode="after")
    def validate_windows(self) -> "FusionConfig":
        if self.output_k > self.max_input_candidates:
            raise ValueError("output_k cannot exceed max_input_candidates")
        if self.default_route_weight == 0 and not any(
            weight > 0 for weight in self.route_weights.values()
        ):
            raise ValueError("fusion requires a positive route weight")
        return self

    def route_weight(self, route_id: str) -> float:
        if route_id in self.route_weights:
            return self.route_weights[route_id]
        segments = route_id.split(".")
        for end in range(len(segments) - 1, 0, -1):
            prefix = ".".join(segments[:end])
            if prefix in self.route_weights:
                return self.route_weights[prefix]
        return self.default_route_weight

    def calibration_for(self, route_id: str) -> CalibrationConfig:
        if route_id in self.route_calibrations:
            return self.route_calibrations[route_id]
        segments = route_id.split(".")
        for end in range(len(segments) - 1, 0, -1):
            prefix = ".".join(segments[:end])
            if prefix in self.route_calibrations:
                return self.route_calibrations[prefix]
        return self.default_calibration


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_ref: str = Field(min_length=1, max_length=512)
    status: DecisionStatus
    stage: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=128)
    duplicate_group_id: str | None = Field(default=None, max_length=64)
    final_rank: int | None = Field(default=None, ge=1)


class RouteFusionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str
    observation_count: int = Field(ge=0)
    contributing_candidate_count: int = Field(ge=0)
    raw_score_count: int = Field(ge=0)
    raw_score_min: float | None = None
    raw_score_max: float | None = None
    calibration_method: CalibrationMethod
    calibration_version: str
    calibration_fallback_count: int = Field(ge=0)


class FusionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["retrieval-fusion-diagnostics.v1"] = "retrieval-fusion-diagnostics.v1"
    fusion_method: FusionMethod
    input_count: int = Field(ge=0)
    valid_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    overflow_count: int = Field(ge=0)
    input_truncated: bool
    identity_derived_count: int = Field(ge=0)
    exact_duplicate_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    canonical_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    near_duplicate_comparisons: int = Field(ge=0)
    near_duplicate_scan_truncated: bool
    invalid_reasons: dict[str, int] = Field(default_factory=dict)
    routes: tuple[RouteFusionDiagnostics, ...] = ()
    decisions: tuple[CandidateDecision, ...] = ()


class RankedEvidence(BaseModel):
    """最终候选；candidate 中的 Evidence identity/raw score 原样保留。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: RetrievalCandidate
    duplicate_group_id: str
    duplicate_member_refs: tuple[str, ...]
    calibrated_scores: dict[str, float]
    fusion_contributions: dict[str, float]
    fusion_score: float
    diversity_score: float
    final_rank: int = Field(ge=1)
    selection_reasons: tuple[str, ...]

    @field_validator(
        "calibrated_scores",
        "fusion_contributions",
    )
    @classmethod
    def finite_score_maps(cls, value: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(item) for item in value.values()):
            raise ValueError("ranked score maps must contain finite values")
        return value

    @field_validator("fusion_score", "diversity_score")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ranked scores must be finite")
        return value


class FusionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ranked: tuple[RankedEvidence, ...] = ()
    diagnostics: FusionDiagnostics


class _InputCandidate:
    __slots__ = ("candidate", "candidate_ref")

    def __init__(self, candidate: RetrievalCandidate, candidate_ref: str) -> None:
        self.candidate = candidate
        self.candidate_ref = candidate_ref


class _CandidateGroup:
    __slots__ = (
        "candidate",
        "canonical_ref",
        "member_refs",
        "exact_duplicate_refs",
        "near_duplicate_refs",
    )

    def __init__(self, item: _InputCandidate) -> None:
        self.candidate = item.candidate
        self.canonical_ref = item.candidate_ref
        self.member_refs = [item.candidate_ref]
        self.exact_duplicate_refs: list[str] = []
        self.near_duplicate_refs: list[str] = []

    def group_id(self) -> str:
        return "dup_" + _stable_digest(sorted(self.member_refs), length=24)


def _candidate_priority(item: _InputCandidate) -> tuple[Any, ...]:
    quality_rank = {"evidence": 0, "source": 1, "surrogate": 2}
    candidate = item.candidate
    provenance_count = sum(
        value is not None
        for value in (
            candidate.document_id,
            candidate.document_version_id,
            candidate.index_version_id,
            candidate.chunk_id,
            candidate.page_no,
            candidate.bbox,
            candidate.provenance_spans or None,
        )
    )
    return (
        quality_rank[candidate.identity_quality()],
        -provenance_count,
        candidate.identity_key(),
        item.candidate_ref,
    )


def _merge_observations(
    first: tuple[RouteObservation, ...],
    second: tuple[RouteObservation, ...],
) -> tuple[RouteObservation, ...]:
    by_id = {item.stable_id(): item for item in (*first, *second)}
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (
                item.route_id,
                item.sub_query_id or "",
                item.route_rank,
                item.stable_id(),
            ),
        )
    )


def _merge_group_candidate(
    group: _CandidateGroup,
    item: _InputCandidate,
    *,
    duplicate_kind: Literal["exact", "near"],
) -> None:
    current = _InputCandidate(group.candidate, group.canonical_ref)
    chosen = min((current, item), key=_candidate_priority)
    merged_observations = _merge_observations(
        group.candidate.observations,
        item.candidate.observations,
    )
    group.candidate = chosen.candidate.model_copy(update={"observations": merged_observations})
    old_canonical_ref = group.canonical_ref
    group.canonical_ref = chosen.candidate_ref
    group.member_refs.append(item.candidate_ref)
    if duplicate_kind == "exact":
        duplicate_ref = (
            item.candidate_ref if chosen.candidate_ref == old_canonical_ref else old_canonical_ref
        )
        group.exact_duplicate_refs.append(duplicate_ref)
    else:
        duplicate_ref = (
            item.candidate_ref if chosen.candidate_ref == old_canonical_ref else old_canonical_ref
        )
        group.near_duplicate_refs.append(duplicate_ref)


def _safe_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    result = int(value)
    if result < 1:
        raise ValueError("route rank must be greater than zero")
    return result


def _raw_scores_from_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    nested = value.get("raw_scores")
    raw_scores: dict[str, float] = {}
    if isinstance(nested, Mapping):
        for key, raw in nested.items():
            if raw is not None:
                raw_scores[str(key)] = _finite_float(
                    raw,
                    field_name=f"raw_scores.{key}",
                )
    for key in _SCORE_FIELDS:
        raw = value.get(key)
        if raw is not None and key not in raw_scores:
            raw_scores[key] = _finite_float(raw, field_name=key)
    return raw_scores


def _semantics_from_mapping(
    value: Any,
    raw_scores: Mapping[str, float] | None = None,
) -> dict[str, ScoreSemantics]:
    if not isinstance(value, Mapping):
        return {}
    if "metric" in value or "direction" in value:
        score_key = next(
            (key for key in _SCORE_FIELDS if raw_scores and key in raw_scores),
            "raw_score",
        )
        return {score_key: ScoreSemantics.model_validate(value)}
    result: dict[str, ScoreSemantics] = {}
    for key, raw in value.items():
        result[str(key)] = (
            raw if isinstance(raw, ScoreSemantics) else ScoreSemantics.model_validate(raw)
        )
    return result


def candidate_from_mapping(
    value: Mapping[str, Any],
    *,
    route_id: str,
    route_rank: int,
) -> RetrievalCandidate:
    """把现有 query graph 字典安全适配为统一候选。

    已有 ``retrieval_observations`` 会逐条保留；若不存在，则由外层 route 和候选
    位置创建一条观察。该函数不修改输入字典。
    """

    raw_evidence = value.get("evidence") if isinstance(value.get("evidence"), Mapping) else {}
    raw_source = (
        raw_evidence.get("source") if isinstance(raw_evidence.get("source"), Mapping) else {}
    )
    content = str(
        raw_evidence.get("content")
        or value.get("content")
        or value.get("text")
        or value.get("snippet")
        or ""
    )
    raw_observations = value.get("retrieval_observations")
    if not raw_observations and isinstance(raw_evidence.get("retrieval"), Mapping):
        raw_observations = [raw_evidence["retrieval"]]
    observations: list[RouteObservation] = []
    if isinstance(raw_observations, Sequence) and not isinstance(raw_observations, (str, bytes)):
        for raw_observation in raw_observations:
            if not isinstance(raw_observation, Mapping):
                raise ValueError("retrieval_observations entries must be objects")
            observation_route = str(
                raw_observation.get("route_id") or raw_observation.get("route") or route_id
            )
            observation_scores = _raw_scores_from_mapping(raw_observation)
            observations.append(
                RouteObservation(
                    route_id=observation_route,
                    sub_query_id=raw_observation.get("sub_query_id")
                    or raw_observation.get("query_variant"),
                    route_rank=_safe_int(
                        raw_observation.get("route_rank") or raw_observation.get("rank"),
                        default=route_rank,
                    ),
                    raw_scores=observation_scores,
                    score_semantics=_semantics_from_mapping(
                        raw_observation.get("score_semantics"),
                        observation_scores,
                    ),
                    primary_score_key=raw_observation.get("primary_score_key"),
                    retrieved_at=raw_observation.get("retrieved_at"),
                )
            )
    if not observations:
        candidate_scores = _raw_scores_from_mapping(value)
        observations.append(
            RouteObservation(
                route_id=route_id,
                sub_query_id=value.get("sub_query_id"),
                route_rank=route_rank,
                raw_scores=candidate_scores,
                score_semantics=_semantics_from_mapping(
                    value.get("score_semantics"),
                    candidate_scores,
                ),
                primary_score_key=value.get("primary_score_key"),
                retrieved_at=value.get("retrieved_at"),
            )
        )

    raw_source_type = str(
        raw_source.get("source_type") or value.get("source_type") or value.get("type") or "local"
    ).lower()
    if raw_source_type in {"web", "internet"}:
        source_type = "web"
    elif raw_source_type in {"local", "milvus", "kb", "knowledge_base"}:
        source_type = "local"
    else:
        source_type = raw_source_type
    uri = (
        raw_source.get("source_url")
        or value.get("uri")
        or value.get("source_url")
        or value.get("url")
    )
    raw_spans = raw_evidence.get("provenance_spans")
    provenance_spans = (
        tuple(dict(span) for span in raw_spans if isinstance(span, Mapping))
        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, (str, bytes))
        else ()
    )
    primary_span = min(
        provenance_spans,
        key=lambda span: (
            {"exact": 0, "derived": 1, "unavailable": 2}.get(
                str(span.get("availability")),
                3,
            ),
            str(span.get("block_id") or ""),
        ),
        default={},
    )
    raw_bbox = value.get("bbox") or primary_span.get("bbox")
    bbox = tuple(raw_bbox) if isinstance(raw_bbox, (list, tuple)) else None
    page_no = value.get("page_no") or value.get("page") or primary_span.get("page_no")
    return RetrievalCandidate(
        evidence_id=raw_evidence.get("evidence_id") or value.get("evidence_id"),
        source_type=source_type,
        tenant_id=raw_source.get("tenant_id") or value.get("tenant_id"),
        kb_id=raw_source.get("kb_id") or value.get("kb_id"),
        document_id=raw_source.get("document_id")
        or value.get("document_id")
        or value.get("doc_id"),
        document_version_id=raw_source.get("document_version_id")
        or value.get("document_version_id"),
        index_version_id=raw_source.get("index_version_id") or value.get("index_version_id"),
        chunk_id=raw_source.get("chunk_id")
        or value.get("chunk_id")
        or value.get("source_chunk_id"),
        content=content,
        content_hash=raw_evidence.get("content_hash") or value.get("content_hash"),
        title=raw_evidence.get("source_label") or value.get("title") or value.get("file_title"),
        uri=uri,
        canonical_uri=value.get("canonical_uri"),
        page_no=page_no,
        bbox=bbox,
        source_block_ids=tuple(raw_source.get("source_block_ids") or ()),
        provenance_spans=provenance_spans,
        heading_path=tuple(raw_evidence.get("heading_path") or ()),
        asset_refs=tuple(raw_evidence.get("asset_refs") or ()),
        trust_score=value.get("trust_score", 0.5),
        observations=tuple(observations),
        attributes=dict(value.get("attributes") or value.get("metadata") or {}),
        evidence_payload=dict(raw_evidence) if raw_evidence else None,
    )


def _invalid_reason(error: Exception) -> str:
    message = str(error).lower()
    if "finite" in message:
        return "non_finite_score"
    if "content" in message and ("blank" in message or "at least 1" in message):
        return "missing_content"
    if "web candidate" in message or "url" in message:
        return "invalid_web_identity"
    if "route" in message:
        return "invalid_route_observation"
    if "content_hash" in message:
        return "content_hash_mismatch"
    return "schema_validation_failed"


def _deduplicate_exact(
    candidates: list[_InputCandidate],
) -> list[_CandidateGroup]:
    groups: dict[str, _CandidateGroup] = {}
    for item in sorted(
        candidates,
        key=lambda entry: (entry.candidate.identity_key(), entry.candidate_ref),
    ):
        key = item.candidate.identity_key()
        group = groups.get(key)
        if group is None:
            groups[key] = _CandidateGroup(item)
        else:
            _merge_group_candidate(group, item, duplicate_kind="exact")
    return [groups[key] for key in sorted(groups)]


def _similarity_window(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head_chars = max_chars // 2
    return content[:head_chars] + content[-(max_chars - head_chars) :]


def _normalized_text_shingles(content: str, *, max_chars: int) -> frozenset[str]:
    bounded = _similarity_window(content, max_chars)
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", "", bounded.casefold())
    if len(normalized) < 3:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[index : index + 3] for index in range(len(normalized) - 2))


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _same_near_duplicate_scope(
    first: RetrievalCandidate,
    second: RetrievalCandidate,
) -> bool:
    if first.source_type != second.source_type:
        return False
    if first.source_type == "web":
        return True
    return first.tenant_id == second.tenant_id and first.kb_id == second.kb_id


def _deduplicate_near(
    groups: list[_CandidateGroup],
    config: FusionConfig,
) -> tuple[list[_CandidateGroup], int, bool]:
    if config.near_duplicate_threshold is None:
        return groups, 0, False
    output: list[_CandidateGroup] = []
    shingles: list[frozenset[str]] = []
    comparisons = 0
    truncated = False
    for group in groups:
        candidate = group.candidate
        current_shingles = _normalized_text_shingles(
            candidate.content,
            max_chars=config.max_similarity_chars,
        )
        match_index: int | None = None
        if len(candidate.content.strip()) >= config.near_duplicate_min_chars:
            for index, existing in enumerate(output):
                if comparisons >= config.max_near_duplicate_comparisons:
                    truncated = True
                    break
                if not _same_near_duplicate_scope(candidate, existing.candidate):
                    continue
                comparisons += 1
                if _jaccard(current_shingles, shingles[index]) >= config.near_duplicate_threshold:
                    match_index = index
                    break
        if match_index is None:
            output.append(group)
            shingles.append(current_shingles)
            continue

        target = output[match_index]
        incoming = _InputCandidate(group.candidate, group.canonical_ref)
        _merge_group_candidate(target, incoming, duplicate_kind="near")
        target.member_refs.extend(ref for ref in group.member_refs if ref != group.canonical_ref)
        target.exact_duplicate_refs.extend(group.exact_duplicate_refs)
        target.near_duplicate_refs.extend(group.near_duplicate_refs)
        shingles[match_index] = _normalized_text_shingles(
            target.candidate.content,
            max_chars=config.max_similarity_chars,
        )
    return output, comparisons, truncated


class _ObservationRecord:
    __slots__ = (
        "group_index",
        "observation",
        "observation_ref",
        "raw_value",
        "semantics",
    )

    def __init__(
        self,
        *,
        group_index: int,
        observation: RouteObservation,
        observation_ref: str,
    ) -> None:
        self.group_index = group_index
        self.observation = observation
        self.observation_ref = observation_ref
        primary = observation.primary_score()
        self.raw_value = primary[1] if primary is not None else None
        self.semantics = primary[2] if primary is not None else ScoreSemantics()


def _rank_calibration(records: Sequence[_ObservationRecord]) -> list[float]:
    max_rank = max((record.observation.route_rank for record in records), default=1)
    if max_rank == 1:
        return [1.0 for _ in records]
    return [1.0 - ((record.observation.route_rank - 1) / (max_rank - 1)) for record in records]


def _calibrate_route(
    records: Sequence[_ObservationRecord],
    calibration: CalibrationConfig,
) -> tuple[list[float], int]:
    rank_values = _rank_calibration(records)
    if calibration.method == "rank":
        return rank_values, 0

    comparable: list[float | None] = []
    for record in records:
        value = record.raw_value
        if value is None:
            comparable.append(None)
        elif record.semantics.direction == "lower_is_better":
            comparable.append(-value)
        else:
            comparable.append(value)
    available = [value for value in comparable if value is not None]
    fallback_count = len(comparable) - len(available)
    if not available:
        return rank_values, len(records)

    calibrated: list[float] = []
    if calibration.method == "min_max":
        minimum, maximum = min(available), max(available)
        for index, value in enumerate(comparable):
            if value is None:
                calibrated.append(rank_values[index])
            elif maximum == minimum:
                calibrated.append(0.5)
            else:
                calibrated.append((value - minimum) / (maximum - minimum))
        return calibrated, fallback_count

    if calibration.method == "z_score":
        mean = fmean(available)
        deviation = pstdev(available)
        for index, value in enumerate(comparable):
            if value is None:
                calibrated.append(rank_values[index])
            elif deviation == 0:
                calibrated.append(0.5)
            else:
                z_score = max(-30.0, min(30.0, (value - mean) / deviation))
                calibrated.append(1.0 / (1.0 + math.exp(-z_score)))
        return calibrated, fallback_count

    # range 校准只在该 observation 明确声明范围时使用；缺范围时按 rank fail-safe。
    for index, record in enumerate(records):
        value = record.raw_value
        minimum = record.semantics.range_min
        maximum = record.semantics.range_max
        if value is None or minimum is None or maximum is None:
            calibrated.append(rank_values[index])
            fallback_count += 1 if value is not None else 0
            continue
        bounded = min(max(value, minimum), maximum)
        score = (bounded - minimum) / (maximum - minimum)
        if record.semantics.direction == "lower_is_better":
            score = 1.0 - score
        calibrated.append(score)
    return calibrated, fallback_count


def _build_calibration(
    groups: Sequence[_CandidateGroup],
    config: FusionConfig,
    approved_subquery_fingerprints: Mapping[str, str],
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[RouteFusionDiagnostics],
]:
    records_by_route: dict[str, list[_ObservationRecord]] = defaultdict(list)
    for group_index, group in enumerate(groups):
        for observation in group.candidate.observations:
            observation_ref = (
                f"{observation.route_id}:{observation.sub_query_id or '-'}:"
                f"{observation.stable_id()}"
            )
            records_by_route[observation.route_id].append(
                _ObservationRecord(
                    group_index=group_index,
                    observation=observation,
                    observation_ref=observation_ref,
                )
            )

    calibrated_by_group: list[dict[str, float]] = [dict() for _ in groups]
    contributions_by_group: list[dict[str, float]] = [dict() for _ in groups]
    route_diagnostics: list[RouteFusionDiagnostics] = []
    for route_id in sorted(records_by_route):
        records = sorted(
            records_by_route[route_id],
            key=lambda item: (
                item.observation.route_rank,
                item.observation_ref,
                groups[item.group_index].candidate.identity_key(),
            ),
        )
        calibration = config.calibration_for(route_id)
        calibrated_values, fallback_count = _calibrate_route(records, calibration)
        route_group_values: dict[int, list[float]] = defaultdict(list)
        for record, calibrated in zip(records, calibrated_values, strict=True):
            calibrated_by_group[record.group_index][record.observation_ref] = calibrated
            route_group_values[record.group_index].append(calibrated)

        weight = config.route_weight(route_id)
        for group_index, values in route_group_values.items():
            route_records = [record for record in records if record.group_index == group_index]
            if config.method == "rrf":
                observation_values = [
                    weight / (config.rrf_k + record.observation.route_rank)
                    for record in route_records
                ]
            elif config.observation_aggregation == "mean":
                observation_values = [weight * value for value in values]
            else:
                observation_values = [weight * value for value in values]
            route_contribution = _aggregate_observation_coverage(
                route_records,
                observation_values,
                config,
                approved_subquery_fingerprints,
            )
            contributions_by_group[group_index][route_id] = route_contribution

        raw_values = [record.raw_value for record in records if record.raw_value is not None]
        route_diagnostics.append(
            RouteFusionDiagnostics(
                route_id=route_id,
                observation_count=len(records),
                contributing_candidate_count=sum(
                    1
                    for contributions in contributions_by_group
                    if contributions.get(route_id, 0.0) > 0
                ),
                raw_score_count=len(raw_values),
                raw_score_min=min(raw_values) if raw_values else None,
                raw_score_max=max(raw_values) if raw_values else None,
                calibration_method=calibration.method,
                calibration_version=calibration.version,
                calibration_fallback_count=fallback_count,
            )
        )
    return calibrated_by_group, contributions_by_group, route_diagnostics


def _aggregate_observation_coverage(
    records: Sequence[_ObservationRecord],
    values: Sequence[float],
    config: FusionConfig,
    approved_subquery_fingerprints: Mapping[str, str],
) -> float:
    """聚合单 route 观察，并对真实的多子查询覆盖给予有界奖励。

    每个 ``sub_query_id`` 只保留最强一次观察，因此重复提交相同 query 不会放大
    分数。没有明确 sub_query_id 的兼容观察不声明额外覆盖。奖励最多为主分数的
    ``subquery_coverage_bonus_cap``，即使 planner 拆出很多子查询也无法刷高权重。
    """

    if not values:
        return 0.0
    if config.observation_aggregation == "mean" and config.method != "rrf":
        base = fmean(values)
    else:
        base = max(values)
    if config.subquery_coverage_bonus_weight <= 0:
        return base
    strongest_by_fingerprint: dict[str, float] = {}
    for record, value in zip(records, values, strict=True):
        subquery_id = record.observation.sub_query_id
        fingerprint = approved_subquery_fingerprints.get(subquery_id or "")
        if fingerprint:
            strongest_by_fingerprint[fingerprint] = max(
                value,
                strongest_by_fingerprint.get(fingerprint, 0.0),
            )
    if len(strongest_by_fingerprint) < 2 or base <= 0:
        return base
    ordered = sorted(strongest_by_fingerprint.values(), reverse=True)
    bonus = min(
        base * config.subquery_coverage_bonus_cap,
        config.subquery_coverage_bonus_weight * sum(ordered[1:]),
    )
    return base + bonus


def normalized_query_fingerprint(query: str) -> str:
    """生成与空白、大小写和标点无关的子查询语义指纹。"""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    canonical = "".join(character for character in normalized if character.isalnum())
    if not canonical:
        return ""
    return "query:" + sha256(canonical.encode("utf-8")).hexdigest()


def _fusion_scores(
    contributions_by_group: Sequence[dict[str, float]],
    config: FusionConfig,
) -> list[float]:
    if config.method == "rrf":
        return [sum(contributions.values()) for contributions in contributions_by_group]
    active_routes = {
        route
        for contributions in contributions_by_group
        for route in contributions
        if config.route_weight(route) > 0
    }
    denominator = sum(config.route_weight(route) for route in active_routes)
    if denominator <= 0:
        return [0.0 for _ in contributions_by_group]
    return [sum(contributions.values()) / denominator for contributions in contributions_by_group]


def _relevance_scale(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum, maximum = min(scores), max(scores)
    if maximum == minimum:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _select_diverse(
    groups: Sequence[_CandidateGroup],
    fusion_scores: Sequence[float],
    config: FusionConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[int, str]]:
    relevance = _relevance_scale(fusion_scores)
    token_sets = [
        _normalized_text_shingles(
            group.candidate.content,
            max_chars=config.max_similarity_chars,
        )
        for group in groups
    ]
    remaining = set(range(len(groups)))
    selected: list[tuple[int, float, tuple[str, ...]]] = []
    document_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    while remaining and len(selected) < config.output_k:
        eligible: list[tuple[float, float, str, int, float, tuple[str, ...]]] = []
        for index in remaining:
            candidate = groups[index].candidate
            document_key = candidate.document_group()
            if document_counts[document_key] >= config.max_per_document:
                continue
            source_quota = config.source_quotas.get(candidate.source_type)
            if source_quota is not None and source_counts[candidate.source_type] >= source_quota:
                continue
            max_similarity = max(
                (
                    _jaccard(token_sets[index], token_sets[selected_index])
                    for selected_index, _, _ in selected
                ),
                default=0.0,
            )
            diversity_score = (
                config.mmr_lambda * relevance[index] - (1.0 - config.mmr_lambda) * max_similarity
            )
            reasons = ["highest_fusion" if not selected else "mmr_selected"]
            if document_counts[document_key] == 0:
                reasons.append("document_novelty")
            if source_counts[candidate.source_type] == 0:
                reasons.append("source_novelty")
            eligible.append(
                (
                    -diversity_score,
                    -fusion_scores[index],
                    candidate.ranking_tiebreak_key(),
                    index,
                    diversity_score,
                    tuple(reasons),
                )
            )
        if not eligible:
            break
        _, _, _, chosen_index, diversity_score, reasons = min(eligible)
        selected.append((chosen_index, diversity_score, reasons))
        remaining.remove(chosen_index)
        candidate = groups[chosen_index].candidate
        document_counts[candidate.document_group()] += 1
        source_counts[candidate.source_type] += 1

    dropped_reasons: dict[int, str] = {}
    for index in remaining:
        candidate = groups[index].candidate
        if document_counts[candidate.document_group()] >= config.max_per_document:
            dropped_reasons[index] = "document_quota_exceeded"
        elif (
            candidate.source_type in config.source_quotas
            and source_counts[candidate.source_type] >= config.source_quotas[candidate.source_type]
        ):
            dropped_reasons[index] = "source_quota_exceeded"
        else:
            dropped_reasons[index] = "outside_top_k"
    return selected, dropped_reasons


def _known_length(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, OverflowError):
        return None


def fuse_route_results(
    route_results: Mapping[
        str,
        Iterable[RetrievalCandidate | Mapping[str, Any] | Any],
    ],
    config: FusionConfig | None = None,
    *,
    approved_subquery_fingerprints: Mapping[str, str] | None = None,
) -> FusionResult:
    """融合多路候选并返回稳定排序与逐候选诊断。

    路线按名称排序后处理，候选的 ``route_rank`` 仍由各路线列表位置决定。单条坏
    数据只记为 invalid；空 route、全部无效或全部被配额过滤时返回合法空结果。
    """

    fusion_config = config or FusionConfig()
    input_candidates: list[_InputCandidate] = []
    decisions: dict[str, CandidateDecision] = {}
    invalid_reasons: Counter[str] = Counter()
    input_count = 0
    invalid_count = 0
    overflow_count = 0
    input_truncated = False

    sorted_routes = sorted(
        ((str(route).strip(), values) for route, values in route_results.items()),
        key=lambda item: item[0],
    )
    if len(sorted_routes) > fusion_config.max_routes:
        for _, omitted_values in sorted_routes[fusion_config.max_routes :]:
            known = _known_length(omitted_values)
            overflow_count += known if known is not None else 1
        sorted_routes = sorted_routes[: fusion_config.max_routes]
        input_truncated = True

    remaining_capacity = fusion_config.max_input_candidates
    for route_id, values in sorted_routes:
        if not _SAFE_NAME.fullmatch(route_id):
            input_count += 1
            invalid_count += 1
            invalid_reasons["invalid_route_id"] += 1
            candidate_ref = f"invalid-route:{_stable_digest(route_id)}"
            decisions[candidate_ref] = CandidateDecision(
                candidate_ref=candidate_ref,
                status="invalid",
                stage="validation",
                reason="invalid_route_id",
            )
            continue
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            input_count += 1
            invalid_count += 1
            invalid_reasons["route_results_not_iterable"] += 1
            candidate_ref = f"{route_id}:invalid-container"
            decisions[candidate_ref] = CandidateDecision(
                candidate_ref=candidate_ref,
                status="invalid",
                stage="validation",
                reason="route_results_not_iterable",
            )
            continue
        if remaining_capacity <= 0:
            known = _known_length(values)
            overflow_count += known if known is not None else 1
            input_truncated = True
            continue

        known = _known_length(values)
        items = list(islice(iter(values), remaining_capacity + 1))
        overflow_here = max(0, len(items) - remaining_capacity)
        if overflow_here:
            items = items[:remaining_capacity]
            if known is not None:
                overflow_here = max(overflow_here, known - remaining_capacity)
            overflow_count += overflow_here
            input_truncated = True
        for route_rank, raw_candidate in enumerate(items, start=1):
            input_count += 1
            candidate_ref = f"{route_id}:{route_rank}"
            try:
                if isinstance(raw_candidate, RetrievalCandidate):
                    candidate = raw_candidate
                elif isinstance(raw_candidate, Mapping):
                    candidate = candidate_from_mapping(
                        raw_candidate,
                        route_id=route_id,
                        route_rank=route_rank,
                    )
                else:
                    raise ValueError("candidate must be an object")
            except (TypeError, ValueError) as exc:
                invalid_count += 1
                reason = _invalid_reason(exc)
                invalid_reasons[reason] += 1
                decisions[candidate_ref] = CandidateDecision(
                    candidate_ref=candidate_ref,
                    status="invalid",
                    stage="validation",
                    reason=reason,
                )
                continue
            identity_ref = candidate.identity_key()
            if len(identity_ref) > 384:
                identity_ref = f"identity_{_stable_digest(identity_ref)}"
            stable_ref = f"{candidate_ref}:{identity_ref}"
            if len(stable_ref) > 500:
                stable_ref = f"{candidate_ref}:identity_{_stable_digest(candidate.identity_key())}"
            input_candidates.append(_InputCandidate(candidate, stable_ref))
        remaining_capacity -= len(items)

    if not input_candidates:
        return FusionResult(
            diagnostics=FusionDiagnostics(
                fusion_method=fusion_config.method,
                input_count=input_count,
                valid_count=0,
                invalid_count=invalid_count,
                overflow_count=overflow_count,
                input_truncated=input_truncated,
                identity_derived_count=0,
                exact_duplicate_count=0,
                near_duplicate_count=0,
                canonical_count=0,
                selected_count=0,
                near_duplicate_comparisons=0,
                near_duplicate_scan_truncated=False,
                invalid_reasons=dict(sorted(invalid_reasons.items())),
                decisions=tuple(decisions[key] for key in sorted(decisions)),
            )
        )

    exact_groups = _deduplicate_exact(input_candidates)
    exact_duplicate_count = len(input_candidates) - len(exact_groups)
    groups, near_comparisons, near_scan_truncated = _deduplicate_near(
        exact_groups,
        fusion_config,
    )
    near_duplicate_count = len(exact_groups) - len(groups)
    calibrated, contributions, route_diagnostics = _build_calibration(
        groups,
        fusion_config,
        approved_subquery_fingerprints or {},
    )
    fusion_scores = _fusion_scores(contributions, fusion_config)
    selected, dropped_reasons = _select_diverse(
        groups,
        fusion_scores,
        fusion_config,
    )

    ranked: list[RankedEvidence] = []
    selected_indexes = {index for index, _, _ in selected}
    rank_by_index = {index: rank for rank, (index, _, _) in enumerate(selected, start=1)}
    for final_rank, (index, diversity_score, reasons) in enumerate(selected, start=1):
        group = groups[index]
        ranked.append(
            RankedEvidence(
                candidate=group.candidate,
                duplicate_group_id=group.group_id(),
                duplicate_member_refs=tuple(sorted(set(group.member_refs))),
                calibrated_scores=dict(sorted(calibrated[index].items())),
                fusion_contributions=dict(sorted(contributions[index].items())),
                fusion_score=fusion_scores[index],
                diversity_score=diversity_score,
                final_rank=final_rank,
                selection_reasons=reasons,
            )
        )

    for index, group in enumerate(groups):
        group_id = group.group_id()
        canonical_status: DecisionStatus = "selected" if index in selected_indexes else "dropped"
        canonical_reason = (
            "selected" if index in selected_indexes else dropped_reasons.get(index, "outside_top_k")
        )
        decisions[group.canonical_ref] = CandidateDecision(
            candidate_ref=group.canonical_ref,
            status=canonical_status,
            stage="selection",
            reason=canonical_reason,
            duplicate_group_id=group_id,
            final_rank=rank_by_index.get(index),
        )
        for candidate_ref in sorted(set(group.exact_duplicate_refs)):
            if candidate_ref == group.canonical_ref:
                continue
            decisions[candidate_ref] = CandidateDecision(
                candidate_ref=candidate_ref,
                status="dropped",
                stage="exact_dedup",
                reason="exact_duplicate",
                duplicate_group_id=group_id,
            )
        for candidate_ref in sorted(set(group.near_duplicate_refs)):
            if candidate_ref == group.canonical_ref:
                continue
            decisions[candidate_ref] = CandidateDecision(
                candidate_ref=candidate_ref,
                status="dropped",
                stage="near_dedup",
                reason="near_duplicate",
                duplicate_group_id=group_id,
            )
        for candidate_ref in group.member_refs:
            if candidate_ref not in decisions:
                decisions[candidate_ref] = CandidateDecision(
                    candidate_ref=candidate_ref,
                    status="dropped",
                    stage="dedup",
                    reason="duplicate_member",
                    duplicate_group_id=group_id,
                )

    return FusionResult(
        ranked=tuple(ranked),
        diagnostics=FusionDiagnostics(
            fusion_method=fusion_config.method,
            input_count=input_count,
            valid_count=len(input_candidates),
            invalid_count=invalid_count,
            overflow_count=overflow_count,
            input_truncated=input_truncated,
            identity_derived_count=sum(
                item.candidate.identity_quality() == "surrogate" for item in input_candidates
            ),
            exact_duplicate_count=exact_duplicate_count,
            near_duplicate_count=near_duplicate_count,
            canonical_count=len(groups),
            selected_count=len(ranked),
            near_duplicate_comparisons=near_comparisons,
            near_duplicate_scan_truncated=near_scan_truncated,
            invalid_reasons=dict(sorted(invalid_reasons.items())),
            routes=tuple(route_diagnostics),
            decisions=tuple(decisions[key] for key in sorted(decisions)),
        ),
    )


def ranked_evidence_to_mapping(ranked: RankedEvidence) -> dict[str, Any]:
    """把类型化结果还原为现有 rerank/answer 节点可消费的无损字典。

    该适配器是 TODO-28 接入旧查询图的唯一建议出口：``score`` 代表新的融合分，
    但所有 raw score 仍保存在 ``retrieval_observations[].raw_scores``，不会被
    ``score`` 覆盖。TODO-08 的完整 ``evidence`` 快照也原样透传。
    """

    candidate = ranked.candidate
    observations: list[dict[str, Any]] = []
    for observation in candidate.observations:
        primary = observation.primary_score()
        value = {
            "route": observation.route_id,
            "sub_query_id": observation.sub_query_id,
            "rank": observation.route_rank,
            "raw_scores": dict(observation.raw_scores),
            "score_semantics": {
                key: semantics.model_dump(mode="json", exclude_none=True)
                for key, semantics in observation.score_semantics.items()
            },
            "primary_score_key": observation.primary_score_key,
            "retrieved_at": observation.retrieved_at,
        }
        if primary is not None:
            value["raw_score"] = primary[1]
        observations.append({key: item for key, item in value.items() if item is not None})

    result: dict[str, Any] = {
        "evidence_id": candidate.evidence_id,
        "source_type": candidate.source_type,
        "tenant_id": candidate.tenant_id,
        "kb_id": candidate.kb_id,
        "document_id": candidate.document_id,
        "document_version_id": candidate.document_version_id,
        "index_version_id": candidate.index_version_id,
        "chunk_id": candidate.chunk_id or candidate.identity_key(),
        "source_chunk_id": candidate.chunk_id,
        "content": candidate.content,
        "text": candidate.content,
        "content_hash": candidate.content_hash,
        "title": candidate.title,
        "file_title": candidate.title,
        "uri": candidate.uri,
        "url": candidate.uri,
        "type": "web" if candidate.source_type == "web" else "milvus",
        "page_no": candidate.page_no,
        "bbox": list(candidate.bbox) if candidate.bbox else None,
        "source_block_ids": list(candidate.source_block_ids),
        "provenance_spans": list(candidate.provenance_spans),
        "heading_path": list(candidate.heading_path),
        "asset_refs": list(candidate.asset_refs),
        "metadata": dict(candidate.attributes),
        "evidence": dict(candidate.evidence_payload)
        if candidate.evidence_payload is not None
        else None,
        "retrieval_observations": observations,
        "duplicate_group_id": ranked.duplicate_group_id,
        "duplicate_member_refs": list(ranked.duplicate_member_refs),
        "calibrated_scores": dict(ranked.calibrated_scores),
        "fusion_contributions": dict(ranked.fusion_contributions),
        "fusion_score": ranked.fusion_score,
        "score": ranked.fusion_score,
        "diversity_score": ranked.diversity_score,
        "rank": ranked.final_rank,
        "selection_reasons": list(ranked.selection_reasons),
    }
    return {key: value for key, value in result.items() if value is not None}


def fusion_result_to_mappings(result: FusionResult) -> list[dict[str, Any]]:
    """批量转换融合结果，保持 ``final_rank`` 顺序。"""

    return [ranked_evidence_to_mapping(item) for item in result.ranked]


__all__ = [
    "CalibrationConfig",
    "CandidateDecision",
    "FusionConfig",
    "FusionDiagnostics",
    "FusionResult",
    "RankedEvidence",
    "normalized_query_fingerprint",
    "RetrievalCandidate",
    "RouteFusionDiagnostics",
    "RouteObservation",
    "ScoreSemantics",
    "candidate_from_mapping",
    "canonicalize_web_uri",
    "fuse_route_results",
    "fusion_result_to_mappings",
    "ranked_evidence_to_mapping",
]
