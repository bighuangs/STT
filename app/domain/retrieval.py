"""TODO-09 检索请求、过滤器、配置快照与结果的稳定领域契约。

本模块不依赖 Milvus、FastAPI、SQLAlchemy 或具体模型实现。查询编排只消费这里的
对象和 ``app.application.retrieval_ports`` 中的端口，因此向量库和重排模型可以
替换。租户、知识库、活动索引和可检索状态始终由服务端作用域解析器追加；客户端
过滤 AST 不能表达或覆盖这些安全边界。
"""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FILTER_MAX_DEPTH = 5
FILTER_MAX_PREDICATES = 32
FILTER_MAX_VALUES = 100
FILTER_MAX_STRING_LENGTH = 1024

RetrievalRoute = Literal["dense", "sparse", "hybrid", "question"]
ProfileStatus = Literal["draft", "published", "retired"]
FilterBooleanOperator = Literal["and", "or", "not"]
FilterOperator = Literal[
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts_with",
    "ends_with",
    "exists",
]
FilterScalar: TypeAlias = str | int | float | bool | None
EngineErrorCode = Literal[
    "RETRIEVAL_UNAVAILABLE",
    "RETRIEVAL_TIMEOUT",
    "RETRIEVAL_DIMENSION_MISMATCH",
    "RETRIEVAL_STALE_SCOPE",
    "RETRIEVAL_FILTER_UNSUPPORTED",
    "RETRIEVAL_INVALID_RESPONSE",
    "RERANKER_UNAVAILABLE",
    "RERANKER_INVALID_RESPONSE",
]

_SAFE_METADATA_FIELD = re.compile(
    r"^metadata\.(?!_)[A-Za-z][A-Za-z0-9_.-]{0,126}[A-Za-z0-9]$|"
    r"^metadata\.(?!_)[A-Za-z]$"
)
_ROUTE_SCORE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_PROTECTED_FILTER_FIELDS = {
    "tenant",
    "tenant_id",
    "kb",
    "kb_id",
    "knowledge_base_id",
    "index",
    "index_id",
    "index_version_id",
    "document_version_id",
    "available",
    "available_int",
    "searchable",
    "collection",
    "collection_name",
}
_SCALAR_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts_with",
    "ends_with",
}
_SEQUENCE_OPERATORS = {"in", "not_in"}


def _strip_identifier(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be blank")
    return normalized


def _validate_filter_scalar(value: Any) -> None:
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError("filter value must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("filter numeric values must be finite")
    if isinstance(value, str) and len(value) > FILTER_MAX_STRING_LENGTH:
        raise ValueError(
            f"filter string values cannot exceed {FILTER_MAX_STRING_LENGTH} characters"
        )


class FilterPredicate(BaseModel):
    """客户端可表达的单个过滤谓词。

    仅允许稳定的 ``document_id`` 和 ``metadata.<key>``。metadata 的实际 key
    还必须由控制库目录解析器确认存在；领域层只负责拒绝结构注入和安全字段越权。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["predicate"] = "predicate"
    field: str = Field(min_length=1, max_length=136)
    operator: FilterOperator
    value: FilterScalar | tuple[FilterScalar, ...]

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        normalized = value.strip()
        lowered = normalized.lower()
        root = lowered.split(".", 1)[0]
        if lowered in _PROTECTED_FILTER_FIELDS or root in _PROTECTED_FILTER_FIELDS:
            raise ValueError(f"filter field is controlled by the server: {normalized}")
        if normalized == "document_id":
            return normalized
        if not _SAFE_METADATA_FIELD.fullmatch(normalized):
            raise ValueError("filter field must be document_id or a safe metadata.<key> path")
        return normalized

    @model_validator(mode="after")
    def validate_operator_value(self) -> "FilterPredicate":
        value = self.value
        if self.operator in _SEQUENCE_OPERATORS:
            if not isinstance(value, tuple):
                raise ValueError(f"{self.operator} requires an array value")
            if not value or len(value) > FILTER_MAX_VALUES:
                raise ValueError(
                    f"{self.operator} requires between 1 and {FILTER_MAX_VALUES} values"
                )
            for item in value:
                _validate_filter_scalar(item)
            return self

        if self.operator == "exists":
            if not isinstance(value, bool):
                raise ValueError("exists requires a boolean value")
            return self

        if isinstance(value, tuple):
            raise ValueError(f"{self.operator} requires a scalar value")
        _validate_filter_scalar(value)

        if self.operator in {"contains", "starts_with", "ends_with"} and not isinstance(value, str):
            raise ValueError(f"{self.operator} requires a string value")
        if self.operator in {"gt", "gte", "lt", "lte"} and (
            value is None or isinstance(value, bool)
        ):
            raise ValueError(f"{self.operator} requires a number or string value")
        return self


class FilterGroup(BaseModel):
    """由白名单谓词组成的布尔过滤树。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["group"] = "group"
    operator: FilterBooleanOperator = "and"
    conditions: tuple["FilterPredicate | FilterGroup", ...] = Field(
        min_length=1,
        max_length=FILTER_MAX_PREDICATES,
    )

    @model_validator(mode="after")
    def validate_tree_shape(self) -> "FilterGroup":
        if self.operator == "not" and len(self.conditions) != 1:
            raise ValueError("not filter group requires exactly one condition")
        validate_filter_ast(self)
        return self


FilterNode: TypeAlias = FilterPredicate | FilterGroup


def validate_filter_ast(
    root: FilterNode,
    *,
    max_depth: int = FILTER_MAX_DEPTH,
    max_predicates: int = FILTER_MAX_PREDICATES,
) -> FilterNode:
    """验证过滤树复杂度并返回原对象，便于应用层在入口处显式调用。

    depth 从根节点的 1 开始计算；一个谓词也占一层。限制同时约束解析成本和
    控制库查询规模，避免深层 ``or/not`` 或超大 ``in`` 制造拒绝服务。
    """

    if max_depth < 1 or max_predicates < 1:
        raise ValueError("filter limits must be greater than zero")
    predicate_count = 0
    stack: list[tuple[FilterNode, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"filter AST depth cannot exceed {max_depth}")
        if isinstance(node, FilterPredicate):
            predicate_count += 1
            if predicate_count > max_predicates:
                raise ValueError(f"filter AST cannot contain more than {max_predicates} predicates")
            continue
        stack.extend((child, depth + 1) for child in node.conditions)
    return root


def iter_filter_predicates(root: FilterNode) -> tuple[FilterPredicate, ...]:
    """按输入顺序展开谓词，供元数据目录解析器使用。"""

    predicates: list[FilterPredicate] = []

    def visit(node: FilterNode) -> None:
        if isinstance(node, FilterPredicate):
            predicates.append(node)
            return
        for child in node.conditions:
            visit(child)

    visit(validate_filter_ast(root))
    return tuple(predicates)


FilterGroup.model_rebuild()


class DenseSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: Literal["dense_vector"] = "dense_vector"
    metric: Literal["IP"] = "IP"
    dimension: int = Field(default=1024, ge=1, le=65_536)
    weight: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("dense weight must be finite")
        return value


class SparseSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: Literal["sparse_vector"] = "sparse_vector"
    metric: Literal["IP"] = "IP"
    weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sparse weight must be finite")
        return value


class HybridSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fusion: Literal["weighted"] = "weighted"
    normalize_scores: bool = True


class RouteWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(min_length=1, max_length=64)
    weight: float = Field(default=1.0, ge=0.0, le=100.0)

    @field_validator("route")
    @classmethod
    def safe_route_name(cls, value: str) -> str:
        value = value.strip()
        if not _ROUTE_SCORE_NAME.fullmatch(value):
            raise ValueError("route weight name contains unsupported characters")
        return value

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RRF route weight must be finite")
        return value


class RRFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    k: int = Field(default=60, ge=1, le=100_000)
    default_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    route_weights: tuple[RouteWeight, ...] = (
        RouteWeight(route="base", weight=1.0),
        RouteWeight(route="hyde", weight=1.0),
    )

    @field_validator("default_weight")
    @classmethod
    def finite_default_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RRF default weight must be finite")
        return value

    @model_validator(mode="after")
    def unique_route_weights(self) -> "RRFConfig":
        names = [item.route for item in self.route_weights]
        if len(names) != len(set(names)):
            raise ValueError("RRF route weights must not contain duplicate routes")
        if (
            self.enabled
            and self.default_weight == 0
            and not any(item.weight > 0 for item in self.route_weights)
        ):
            raise ValueError("enabled RRF requires at least one positive route weight")
        return self

    def weight_for(self, route: str) -> float:
        for item in self.route_weights:
            if item.route == route:
                return item.weight
        return self.default_weight


class RerankerConfig(BaseModel):
    """重排器的可发布快照；默认值与现有 BGE 动态 Top-K 行为一致。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    model_id: str = Field(
        default="./models/bge-reranker-v2-m3",
        min_length=1,
        max_length=512,
    )
    normalize_scores: bool = True
    min_top_k: int = Field(default=1, ge=1, le=1000)
    max_top_k: int = Field(default=10, ge=1, le=1000)
    gap_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    gap_absolute: float = Field(default=2.0, ge=0.0)
    max_input_tokens: int = Field(default=512, ge=32, le=1_000_000)
    # ``summarize`` 仅为读取旧 profile 保留；生产适配器统一按本地 reranker
    # tokenizer 确定性截断。新 profile 默认 truncate，重排前绝不隐式调用 LLM。
    long_text_strategy: Literal["summarize", "truncate"] = "truncate"

    @model_validator(mode="after")
    def validate_top_k_and_gaps(self) -> "RerankerConfig":
        if self.max_top_k < self.min_top_k:
            raise ValueError("reranker max_top_k must be >= min_top_k")
        if not math.isfinite(self.gap_ratio) or not math.isfinite(self.gap_absolute):
            raise ValueError("reranker gap thresholds must be finite")
        return self


class MetadataFilterPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    allow_document_id: bool = True
    allow_metadata: bool = True
    max_depth: int = Field(default=FILTER_MAX_DEPTH, ge=1, le=10)
    max_predicates: int = Field(default=FILTER_MAX_PREDICATES, ge=1, le=256)
    max_values_per_predicate: int = Field(default=FILTER_MAX_VALUES, ge=1, le=1000)


class FusionCalibrationPolicy(BaseModel):
    """TODO-28 的版本化单路分数校准策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["rank", "min_max", "z_score", "range"] = "rank"
    version: str = Field(default="rank.v1", min_length=1, max_length=128)


class FusionPolicy(BaseModel):
    """多路召回进入 Reranker 前的统一治理快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["rrf", "weighted"] = "rrf"
    route_weights: dict[str, float] = Field(default_factory=dict)
    default_calibration: FusionCalibrationPolicy = Field(default_factory=FusionCalibrationPolicy)
    route_calibrations: dict[str, FusionCalibrationPolicy] = Field(default_factory=dict)
    # None 与 TODO-09 旧行为兼容；发布新 profile 时可在配对评测后显式开启。
    near_duplicate_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    max_per_document: int = Field(default=5, ge=1, le=1000)
    source_quotas: dict[str, int] = Field(default_factory=dict)
    mmr_lambda: float = Field(default=0.8, ge=0.0, le=1.0)
    # 同一路候选覆盖多个不同子查询时给予小幅奖励；同一子查询的重复 observation
    # 只计一次。奖励相对 strongest observation 有上限，避免拆分 query 刷分。
    # 缺字段的历史 profile 必须保持原始 max/mean 聚合，不能在 hash 不变时静默
    # 改排序。只有显式创建的新不可变版本才启用 coverage bonus。
    subquery_coverage_bonus_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    subquery_coverage_bonus_cap: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("route_weights")
    @classmethod
    def validate_fusion_weights(cls, value: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for route, raw in value.items():
            name = str(route).strip()
            if not _ROUTE_SCORE_NAME.fullmatch(name):
                raise ValueError("fusion route weight name is invalid")
            weight = float(raw)
            if not math.isfinite(weight) or not 0 <= weight <= 100:
                raise ValueError("fusion route weights must be finite within [0, 100]")
            result[name] = weight
        return result

    @field_validator("route_calibrations")
    @classmethod
    def validate_calibration_names(
        cls, value: dict[str, FusionCalibrationPolicy]
    ) -> dict[str, FusionCalibrationPolicy]:
        if any(not _ROUTE_SCORE_NAME.fullmatch(str(route)) for route in value):
            raise ValueError("fusion route calibration name is invalid")
        return value

    @field_validator("source_quotas")
    @classmethod
    def validate_source_quotas(cls, value: dict[str, int]) -> dict[str, int]:
        result: dict[str, int] = {}
        for source, quota in value.items():
            name = str(source).strip()
            if not _ROUTE_SCORE_NAME.fullmatch(name):
                raise ValueError("fusion source quota name is invalid")
            if isinstance(quota, bool) or not 1 <= int(quota) <= 1000:
                raise ValueError("fusion source quota must be within [1, 1000]")
            result[name] = int(quota)
        return result


class ContextPackProfile(BaseModel):
    """TODO-29 模型窗口、装包配额与父邻扩展策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_context_limit: int = Field(default=8192, ge=1024, le=2_000_000)
    reserved_output_tokens: int = Field(default=1024, ge=64, le=500_000)
    safety_margin_tokens: int = Field(default=256, ge=0, le=100_000)
    # 回答阶段当前最多 4 条 logical calls，每条最多 3 次 provider attempts；默认
    # 12,000,000 覆盖 4 * 3 * Qwen3.7-Max 1M context 的硬上界。
    answer_run_token_budget: int = Field(default=12_000_000, ge=8_192, le=24_000_000)
    evidence_character_budget: int = Field(default=80_000, ge=1024, le=4_000_000)
    max_blocks: int = Field(default=20, ge=1, le=10_000)
    max_blocks_per_document: int = Field(default=5, ge=1, le=10_000)
    max_blocks_per_source: int = Field(default=8, ge=1, le=10_000)
    expansion_mode: Literal["none", "parent", "neighbors", "parent_and_neighbors"] = "none"
    neighbor_window: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def validate_context_reservations(self) -> "ContextPackProfile":
        if self.reserved_output_tokens + self.safety_margin_tokens >= self.model_context_limit:
            raise ValueError("context output reserve and safety margin exhaust model window")
        return self


class EvidenceGatePolicy(BaseModel):
    """TODO-30 生成前/后校验的发布策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_query_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_trusted_evidence: int = Field(default=1, ge=1, le=1000)
    freshness_max_age_days: int = Field(default=30, ge=0, le=36500)
    minimum_claim_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    # 两项默认关闭以保持已发布 v1 profile 的行为不变；新 profile 必须显式
    # 校准并冻结到 hash 后，才能把检索 lineage 与独立 relevance proof 分离。
    require_lexical_support_when_reranker_degraded: bool = False
    min_normalized_rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)


class QueryPlanningProfile(BaseModel):
    """TODO-27 规划器的服务端预算；Web 权限仍由 KB 外发策略取交集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_hyde: bool = True
    max_subqueries: int = Field(default=4, ge=1, le=8)
    max_dependency_depth: int = Field(default=2, ge=1, le=4)
    max_routes: int = Field(default=3, ge=1, le=8)
    token_budget: int = Field(default=4096, ge=128, le=131_072)
    cost_budget: float = Field(default=1.0, ge=0.05, le=1000.0)


class RetrievalProfile(BaseModel):
    """查询期只读的检索配置版本。

    数据库中的 draft 可以通过“创建新版本”修改；一旦发布，查询会把 profile_id
    和 version 固定到会话。模型本身设置为 frozen，避免在一次调用中原地改写。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["retrieval-profile.v1"] = "retrieval-profile.v1"
    profile_id: str = Field(default="retrieval-default", min_length=1, max_length=128)
    version: int = Field(default=1, ge=1)
    tenant_id: str = Field(default="tenant_default", min_length=1, max_length=64)
    status: ProfileStatus = "published"
    routes: tuple[RetrievalRoute, ...] = ("hybrid",)
    candidate_k: int = Field(default=5, ge=1, le=1000)
    top_k: int = Field(default=5, ge=1, le=1000)
    score_threshold: float | None = None
    dense: DenseSearchConfig = Field(default_factory=DenseSearchConfig)
    sparse: SparseSearchConfig = Field(default_factory=SparseSearchConfig)
    hybrid: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    metadata_filter: MetadataFilterPolicy = Field(default_factory=MetadataFilterPolicy)
    fusion: FusionPolicy = Field(default_factory=FusionPolicy)
    context_pack: ContextPackProfile = Field(default_factory=ContextPackProfile)
    evidence_gate: EvidenceGatePolicy = Field(default_factory=EvidenceGatePolicy)
    query_planning: QueryPlanningProfile = Field(default_factory=QueryPlanningProfile)
    deduplicate_by: Literal["chunk_id", "content_hash"] = "chunk_id"
    parent_expansion_field: str | None = None
    # 首次加载本地 BGE-M3/稀疏模型、以及导入 Worker 并行编码时会有明显的
    # CPU/内存带宽竞争。内建配置给本地召回 90 秒有界预算；自定义 profile
    # 仍可在 100ms..300s 内明确收紧或放宽，执行器和 SDK 都会落实该值。
    timeout_ms: int = Field(default=90_000, ge=100, le=300_000)

    @field_validator("profile_id", "tenant_id")
    @classmethod
    def normalize_profile_identifiers(cls, value: str, info) -> str:
        return _strip_identifier(value, name=info.field_name)

    @field_validator("routes")
    @classmethod
    def unique_routes(cls, value: tuple[RetrievalRoute, ...]) -> tuple[RetrievalRoute, ...]:
        if not value:
            raise ValueError("retrieval profile requires at least one route")
        if len(value) != len(set(value)):
            raise ValueError("retrieval profile routes must be unique")
        return value

    @field_validator("score_threshold")
    @classmethod
    def finite_threshold(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score_threshold must be finite")
        return value

    @field_validator("parent_expansion_field")
    @classmethod
    def safe_parent_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            raise ValueError("parent_expansion_field must be a safe field identifier")
        return value

    @model_validator(mode="after")
    def validate_retrieval_window(self) -> "RetrievalProfile":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if "hybrid" in self.routes and self.dense.weight + self.sparse.weight <= 0:
            raise ValueError("hybrid route requires a positive dense or sparse weight")
        if "dense" in self.routes and self.dense.weight <= 0:
            raise ValueError("dense route requires a positive dense weight")
        if "sparse" in self.routes and self.sparse.weight <= 0:
            raise ValueError("sparse route requires a positive sparse weight")
        return self


def default_retrieval_profile(
    tenant_id: str,
    *,
    profile_id: str = "retrieval-default",
    version: int = 1,
    status: ProfileStatus = "published",
) -> RetrievalProfile:
    """构造与 TODO-09 前现状兼容的默认配置。

    现状为 BGE-M3 dense 0.9 + sparse 0.1、召回/返回 5、RRF k=60、
    base/HyDE 等权和 BGE reranker 动态 1..10。
    """

    return RetrievalProfile(
        profile_id=profile_id,
        version=version,
        tenant_id=tenant_id,
        status=status,
    )


class RetrievalScope(BaseModel):
    """由认证上下文产生的作用域；不能从过滤 AST 推导。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    kb_ids: tuple[str, ...] = Field(min_length=1, max_length=64)

    @field_validator("tenant_id", "user_id")
    @classmethod
    def normalize_scope_identifiers(cls, value: str, info) -> str:
        return _strip_identifier(value, name=info.field_name)

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_strip_identifier(item, name="kb_id") for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("kb_ids must not contain duplicates")
        return normalized


class RetrievalRequest(BaseModel):
    """传给 ``RetrievalPort`` 的完整、可审计请求。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=32_000)
    variants: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    scope: RetrievalScope
    profile: RetrievalProfile
    execution_mode: Literal["online", "preview"] = "online"
    filters: FilterNode | None = None
    document_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10_000)
    trace_id: str = Field(min_length=1, max_length=128)
    deadline_at: datetime | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("retrieval query cannot be blank")
        return value

    @field_validator("variants")
    @classmethod
    def normalize_variants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value if item.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("query variants must not contain duplicates")
        return normalized

    @field_validator("document_ids")
    @classmethod
    def normalize_document_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_strip_identifier(item, name="document_id") for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("document_ids must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_profile_and_filter_scope(self) -> "RetrievalRequest":
        if self.scope.tenant_id != self.profile.tenant_id:
            raise ValueError("retrieval profile tenant must match request tenant")
        if self.execution_mode == "online" and self.profile.status != "published":
            raise ValueError("online retrieval only accepts a published profile")
        if self.filters is not None:
            if not self.profile.metadata_filter.enabled:
                raise ValueError("retrieval profile does not allow client filters")
            validate_filter_ast(
                self.filters,
                max_depth=self.profile.metadata_filter.max_depth,
                max_predicates=self.profile.metadata_filter.max_predicates,
            )
            for predicate in iter_filter_predicates(self.filters):
                if (
                    predicate.field == "document_id"
                    and not self.profile.metadata_filter.allow_document_id
                ):
                    raise ValueError("profile does not allow document_id filters")
                if (
                    predicate.field.startswith("metadata.")
                    and not self.profile.metadata_filter.allow_metadata
                ):
                    raise ValueError("profile does not allow metadata filters")
                if (
                    isinstance(predicate.value, tuple)
                    and len(predicate.value) > self.profile.metadata_filter.max_values_per_predicate
                ):
                    raise ValueError("filter value count exceeds the retrieval profile limit")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must include a timezone")
        return self


class RouteScore(BaseModel):
    """一条命中在单路中的原始观察，禁止压扁成一个不可解释分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(min_length=1, max_length=64)
    raw_score: float | None = None
    rank: int = Field(ge=1)
    rrf_score: float | None = None
    explanation: str | None = Field(default=None, max_length=1000)

    @field_validator("route")
    @classmethod
    def safe_route_name(cls, value: str) -> str:
        value = value.strip()
        if not _ROUTE_SCORE_NAME.fullmatch(value):
            raise ValueError("route score name contains unsupported characters")
        return value

    @field_validator("raw_score", "rrf_score")
    @classmethod
    def finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("route scores must be finite")
        return value


class RetrievalHit(BaseModel):
    """检索引擎边界上的标准块，不携带 Milvus entity 对象。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=128)
    evidence_id: str | None = Field(
        default=None,
        pattern=r"^ev_[0-9a-f]{40}$",
    )
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    document_version_id: str = Field(min_length=1, max_length=64)
    index_version_id: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=4_000_000)
    content_hash: str | None = Field(
        default=None,
        pattern=r"^(?:sha256:)?[0-9a-f]{64}$",
    )
    title: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scores: tuple[RouteScore, ...] = Field(min_length=1)
    final_score: float | None = None

    @field_validator(
        "chunk_id",
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
    )
    @classmethod
    def normalize_hit_identifiers(cls, value: str, info) -> str:
        return _strip_identifier(value, name=info.field_name)

    @field_validator("content")
    @classmethod
    def require_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieval hit content cannot be blank")
        return value

    @field_validator("final_score")
    @classmethod
    def finite_final_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("final retrieval score must be finite")
        return value

    @model_validator(mode="after")
    def unique_route_scores(self) -> "RetrievalHit":
        names = [score.route for score in self.scores]
        if len(names) != len(set(names)):
            raise ValueError("retrieval hit must contain at most one score per route")
        return self


class RouteOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(min_length=1, max_length=64)
    status: Literal["succeeded", "empty", "failed", "skipped"]
    candidate_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    error_code: EngineErrorCode | None = None
    message: str | None = Field(default=None, max_length=1000)

    @field_validator("route")
    @classmethod
    def safe_route_name(cls, value: str) -> str:
        value = value.strip()
        if not _ROUTE_SCORE_NAME.fullmatch(value):
            raise ValueError("route outcome name contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_failure(self) -> "RouteOutcome":
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed route outcome requires error_code")
        if self.status != "failed" and self.error_code is not None:
            raise ValueError("only failed route outcome can contain error_code")
        return self


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: int = Field(ge=1)
    engine: str = Field(min_length=1, max_length=64)
    hits: tuple[RetrievalHit, ...] = Field(default_factory=tuple)
    route_outcomes: tuple[RouteOutcome, ...] = Field(default_factory=tuple)
    elapsed_ms: int = Field(default=0, ge=0)


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=32_000)
    candidates: tuple[RetrievalHit, ...]
    config: RerankerConfig
    trace_id: str = Field(min_length=1, max_length=128)


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    hits: tuple[RetrievalHit, ...]
    elapsed_ms: int = Field(default=0, ge=0)
    status: Literal["succeeded", "degraded"] = "succeeded"
    failure_code: EngineErrorCode | None = None
    message: str | None = Field(default=None, max_length=1000)
    batch_size: int = Field(default=0, ge=0, le=64)
    batch_count: int = Field(default=0, ge=0)
    truncated_count: int = Field(default=0, ge=0)
    max_input_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_degradation(self) -> "RerankResult":
        if self.status == "degraded" and self.failure_code is None:
            raise ValueError("degraded rerank result requires failure_code")
        if self.status == "succeeded" and self.failure_code is not None:
            raise ValueError("successful rerank result cannot contain failure_code")
        return self


class RetrievalEngineError(RuntimeError):
    """检索/重排适配器向查询编排暴露的稳定、可观测失败。"""

    def __init__(
        self,
        code: EngineErrorCode,
        message: str,
        *,
        retryable: bool,
        route: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        self.route = route
        self.details = dict(details or {})
