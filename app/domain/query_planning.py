"""TODO-27 查询理解与受控查询计划的领域契约。

本模块刻意不依赖 LangGraph、数据库、LLM 或具体检索引擎。调用方必须把历史快照、
已授权作用域和检索策略作为输入传入；planner 既不能自行读取历史，也不能通过计划
扩大 tenant/KB/document 范围。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PLANNER_VERSION = "query-planner-rules/1.6.0"
FACET_LEDGER_SCHEMA_VERSION = "query-facet-ledger.v2"
LEGACY_FACET_LEDGER_SCHEMA_VERSION = "query-facet-ledger.v1"
FACET_EXTRACTOR_VERSION = "query-facet-rules/1.4.0"
EMPTY_QUERY_SHA256 = f"sha256:{hashlib.sha256(b'').hexdigest()}"
MAX_QUERY_LENGTH = 4096
MAX_ENTITIES = 32
MAX_CONSTRAINTS = 32
MAX_DECISION_REASONS = 32
MAX_QUERY_FACETS = 32
MAX_FACET_EVIDENCE_IDS = 128


class QueryIntent(StrEnum):
    GREETING = "greeting"
    UI_HELP = "ui_help"
    KNOWLEDGE_FACT = "knowledge_fact"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    TEMPORAL_FACT = "temporal_fact"


class QueryLanguage(StrEnum):
    ZH = "zh"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TemporalKind(StrEnum):
    NONE = "none"
    CURRENT = "current"
    AS_OF = "as_of"
    RANGE = "range"


class PlanDisposition(StrEnum):
    DIRECT_ANSWER = "direct_answer"
    NEED_CLARIFICATION = "need_clarification"
    RETRIEVE = "retrieve"


class SubQueryPurpose(StrEnum):
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"


class QueryRouteType(StrEnum):
    LOCAL_HYBRID = "local_hybrid"
    HYDE = "hyde"
    WEB = "web"


class RouteFallback(StrEnum):
    FAIL_PLAN = "fail_plan"
    SKIP_ROUTE = "skip_route"
    LOCAL_ONLY = "local_only"


class RewriteStatus(StrEnum):
    ORIGINAL = "original"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FacetNecessity(StrEnum):
    """信息需求是否无条件要求回答。"""

    REQUIRED = "required"
    CONDITIONAL = "conditional"


class FacetPolarity(StrEnum):
    """信息需求中的显式否定方向。"""

    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class FacetClosure(StrEnum):
    """回答该信息需求所需的集合闭合语义。"""

    SINGLE = "single"
    OPEN_ENUMERATION = "open_enumeration"
    CLOSED_COUNT = "closed_count"


class FacetSupportStatus(StrEnum):
    """Facet 与已装包检索分支的确定性绑定状态。

    字段名为兼容 ``query-facet-ledger.v1`` 保留。即使值为 ``supported``，也只
    表示该 facet 的检索分支有 Evidence 进入 ContextPack，绝不代表 Evidence
    语义蕴含答案；生成后仍必须经过独立 grounding/verifier。
    """

    UNASSESSED = "unassessed"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    SUPPORTED = "supported"


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("conversation text cannot be blank")
        return normalized


class AuthorizedScopeRef(BaseModel):
    """执行器信任的授权作用域快照。

    ``scope_id`` 应由认证/授权层生成。显式保留 tenant/KB/document 是为了审计和
    执行前逐项比对，而不是允许 planner 修改这些值。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    document_ids: tuple[str, ...] = Field(default=(), max_length=512)

    @field_validator("scope_id", "tenant_id")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scope identifiers cannot be blank")
        return normalized

    @field_validator("kb_ids", "document_ids")
    @classmethod
    def normalize_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip() for item in values if item.strip()))
        if len(normalized) != len(values):
            raise ValueError("scope identifiers must be non-blank and unique")
        return normalized


class PlanningPolicy(BaseModel):
    """由服务端检索配置和 KB 数据外发策略合成的不可变规划策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="planning-policy/1", min_length=1, max_length=128)
    allow_hyde: bool = True
    allow_web: bool = False
    max_subqueries: int = Field(default=4, ge=1, le=8)
    max_dependency_depth: int = Field(default=2, ge=1, le=4)
    max_routes: int = Field(default=3, ge=1, le=8)
    total_timeout_ms: int = Field(default=8_000, ge=250, le=120_000)
    token_budget: int = Field(default=4_096, ge=128, le=131_072)
    cost_budget: float = Field(default=1.0, ge=0.05, le=1_000.0)
    default_top_k: int = Field(default=8, ge=1, le=100)
    max_top_k: int = Field(default=40, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_top_k(self) -> "PlanningPolicy":
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot exceed max_top_k")
        return self


class TemporalRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TemporalKind = TemporalKind.NONE
    expression: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_expression(self) -> "TemporalRequirement":
        if self.kind == TemporalKind.NONE and self.expression is not None:
            raise ValueError("non-temporal query cannot carry a temporal expression")
        if self.kind != TemporalKind.NONE and not (self.expression or "").strip():
            raise ValueError("temporal query requires the source expression")
        return self


class QueryConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    operator: Literal["eq", "as_of", "between"] = "eq"
    value: str = Field(min_length=1, max_length=256)


class QueryAmbiguity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_ambiguous: bool = False
    missing_slots: tuple[str, ...] = Field(default=(), max_length=8)
    clarification_question: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_consistency(self) -> "QueryAmbiguity":
        if self.is_ambiguous:
            if not self.missing_slots:
                raise ValueError("ambiguous query requires at least one missing slot")
            if not (self.clarification_question or "").strip():
                raise ValueError("ambiguous query requires a clarification question")
        elif self.missing_slots or self.clarification_question is not None:
            raise ValueError("unambiguous query cannot carry clarification fields")
        return self


class RewriteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RewriteStatus
    semantic_preservation_score: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(min_length=1, max_length=8)


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    display_query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    standalone_query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    language: QueryLanguage
    intent: QueryIntent
    temporal_requirement: TemporalRequirement = Field(default_factory=TemporalRequirement)
    entities: tuple[str, ...] = Field(default=(), max_length=MAX_ENTITIES)
    constraints: tuple[QueryConstraint, ...] = Field(default=(), max_length=MAX_CONSTRAINTS)
    ambiguity: QueryAmbiguity = Field(default_factory=QueryAmbiguity)
    need_retrieval: bool
    need_freshness: bool = False
    needs_multihop: bool = False
    rewrite: RewriteDecision

    @field_validator("original_query", "display_query", "standalone_query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query cannot be blank")
        return normalized

    @field_validator("entities")
    @classmethod
    def normalize_entities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip() for item in values if item.strip()))
        if len(normalized) != len(values):
            raise ValueError("entities must be non-blank and unique")
        return normalized

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "QueryUnderstanding":
        if self.display_query != self.original_query:
            raise ValueError("display_query must preserve the normalized original query")
        if self.ambiguity.is_ambiguous and self.need_retrieval:
            raise ValueError("ambiguous query must be clarified before retrieval")
        if self.intent in {QueryIntent.GREETING, QueryIntent.UI_HELP} and self.need_retrieval:
            raise ValueError("greeting/UI help must not trigger knowledge retrieval")
        if self.need_freshness and self.temporal_requirement.kind == TemporalKind.NONE:
            raise ValueError("freshness requires a temporal requirement")
        return self


class QuerySubQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sub_query_id: str = Field(pattern=r"^sq_[a-z0-9_]{1,48}$")
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    purpose: SubQueryPurpose = SubQueryPurpose.RETRIEVAL
    depends_on: tuple[str, ...] = Field(default=(), max_length=8)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("sub-query cannot be blank")
        return normalized

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("sub-query dependencies must be unique")
        return values


class QueryFacet(BaseModel):
    """由用户问题确定性抽取的一个独立信息需求。

    ``support_status`` 与 ``evidence_ids`` 是运行期可更新字段。Planner 只发布
    ``unassessed``；ContextPack 只能依据真实的 sub-query/Evidence 绑定更新它，
    不能把该状态当作语义正确性判定。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    facet_id: str = Field(pattern=r"^qf_[0-9a-f]{32}$")
    information_need: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    retrieval_expression: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    necessity: FacetNecessity = FacetNecessity.REQUIRED
    polarity: FacetPolarity = FacetPolarity.AFFIRMATIVE
    closure: FacetClosure = FacetClosure.SINGLE
    expected_count: int | None = Field(default=None, ge=1, le=MAX_FACET_EVIDENCE_IDS)
    sub_query_ids: tuple[str, ...] = Field(default=(), max_length=8)
    support_status: FacetSupportStatus = FacetSupportStatus.UNASSESSED
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=MAX_FACET_EVIDENCE_IDS)

    @field_validator("information_need", "retrieval_expression")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("facet text cannot be blank")
        return normalized

    @field_validator("sub_query_ids")
    @classmethod
    def normalize_sub_query_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip() for item in values)
        if any(not item for item in normalized):
            raise ValueError("facet sub-query IDs cannot be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("facet sub-query IDs must be unique")
        return normalized

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip() for item in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("facet evidence IDs must be unique")
        if any(re.fullmatch(r"ev_[0-9a-f]{40}", item) is None for item in normalized):
            raise ValueError("facet evidence IDs must be stable Evidence identifiers")
        return normalized

    @model_validator(mode="after")
    def validate_support_binding(self) -> "QueryFacet":
        if self.closure == FacetClosure.CLOSED_COUNT:
            if self.expected_count is None:
                raise ValueError("closed-count facet requires expected_count")
        elif self.expected_count is not None:
            raise ValueError("only closed-count facet may carry expected_count")
        if self.support_status in {
            FacetSupportStatus.UNASSESSED,
            FacetSupportStatus.UNSUPPORTED,
        }:
            if self.evidence_ids:
                raise ValueError("unassessed/unsupported facet cannot bind Evidence")
        elif not self.evidence_ids:
            raise ValueError("partial/supported facet requires bound Evidence")
        return self


class FacetLedger(BaseModel):
    """可序列化、可版本化的查询信息需求账本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-facet-ledger.v1", "query-facet-ledger.v2"] = (
        FACET_LEDGER_SCHEMA_VERSION
    )
    extractor_version: str = Field(
        default=FACET_EXTRACTOR_VERSION,
        min_length=1,
        max_length=128,
    )
    source_query_hash: str = Field(
        default=EMPTY_QUERY_SHA256,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    binding_semantics: Literal["retrieval_branch_lineage_only"] = "retrieval_branch_lineage_only"
    facets: tuple[QueryFacet, ...] = Field(default=(), max_length=MAX_QUERY_FACETS)

    @model_validator(mode="after")
    def validate_unique_facets(self) -> "FacetLedger":
        identifiers = [item.facet_id for item in self.facets]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("facet IDs must be unique")
        return self


class RouteFilter(BaseModel):
    """Planner 可携带的非授权过滤条件。

    这里只允许文档 ID 和 metadata 字段。tenant/KB/index/availability 仍必须由
    TODO-26 的服务端过滤编译器注入，不能出现在 planner 输出中。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1, max_length=136)
    operator: Literal["eq", "in"]
    value: str | tuple[str, ...]

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        normalized = value.strip()
        if normalized == "document_id":
            return normalized
        if re.fullmatch(r"metadata\.(?!_)[A-Za-z][A-Za-z0-9_.-]{0,126}", normalized):
            return normalized
        raise ValueError("planner filter can only target document_id or metadata.<key>")

    @model_validator(mode="after")
    def validate_value_shape(self) -> "RouteFilter":
        if self.operator == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in filter requires a non-empty array")
        elif isinstance(self.value, tuple):
            raise ValueError("eq filter requires a scalar string")
        return self


class QueryRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(pattern=r"^route_[a-z0-9_]{1,48}$")
    route_type: QueryRouteType
    required: bool
    sub_query_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    timeout_ms: int = Field(ge=100, le=120_000)
    top_k: int = Field(ge=1, le=1_000)
    filters: tuple[RouteFilter, ...] = Field(default=(), max_length=32)
    fallback: RouteFallback
    max_cost: float = Field(ge=0.0, le=1_000.0)
    evidence_eligible: bool = True

    @field_validator("sub_query_ids")
    @classmethod
    def unique_sub_queries(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("route sub_query_ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_route_semantics(self) -> "QueryRoute":
        if self.required and self.fallback != RouteFallback.FAIL_PLAN:
            raise ValueError("required route must fail the plan when unavailable")
        if not self.required and self.fallback == RouteFallback.FAIL_PLAN:
            raise ValueError("optional route cannot fail the complete plan")
        if self.route_type == QueryRouteType.HYDE and self.evidence_eligible:
            raise ValueError("HyDE synthetic text can never be citation evidence")
        if self.route_type != QueryRouteType.HYDE and not self.evidence_eligible:
            raise ValueError("only HyDE route may be marked evidence-ineligible")
        return self


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(pattern=r"^qpl_[0-9a-f]{32}$")
    planner_version: str = Field(min_length=1, max_length=128)
    profile_version: int = Field(ge=0)
    disposition: PlanDisposition
    understanding: QueryUnderstanding
    facet_ledger: FacetLedger = Field(default_factory=FacetLedger)
    sub_queries: tuple[QuerySubQuery, ...] = Field(default=(), max_length=8)
    routes: tuple[QueryRoute, ...] = Field(default=(), max_length=8)
    authorized_scope_ref: AuthorizedScopeRef
    policy: PlanningPolicy
    deadline_at: datetime | None = None
    token_budget: int = Field(ge=128, le=131_072)
    cost_budget: float = Field(ge=0.05, le=1_000.0)
    decision_reasons: tuple[str, ...] = Field(min_length=1, max_length=MAX_DECISION_REASONS)
    confidence: float = Field(ge=0.0, le=1.0)
    response_key: str | None = Field(default=None, max_length=128)
    limitations: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_complete_plan(self) -> "QueryPlan":
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")
        if self.token_budget > self.policy.token_budget:
            raise ValueError("plan token budget exceeds trusted policy")
        if self.cost_budget > self.policy.cost_budget:
            raise ValueError("plan cost budget exceeds trusted policy")

        if self.disposition == PlanDisposition.RETRIEVE:
            if not self.understanding.need_retrieval:
                raise ValueError("retrieval plan requires need_retrieval=true")
            if not self.facet_ledger.facets:
                raise ValueError("retrieval plan requires at least one facet")
            if not any(
                facet.necessity == FacetNecessity.REQUIRED for facet in self.facet_ledger.facets
            ):
                raise ValueError("retrieval plan requires at least one required facet")
            if not self.sub_queries or not self.routes:
                raise ValueError("retrieval plan requires sub-queries and routes")
            if not any(route.required for route in self.routes):
                raise ValueError("retrieval plan requires at least one required route")
            if self.response_key is not None:
                raise ValueError("retrieval plan cannot carry a direct response key")
        else:
            if self.sub_queries or self.routes:
                raise ValueError("non-retrieval plan cannot carry sub-queries or routes")
            if self.disposition == PlanDisposition.NEED_CLARIFICATION:
                if not self.understanding.ambiguity.is_ambiguous:
                    raise ValueError("clarification plan requires an ambiguous query")
                if self.response_key is not None:
                    raise ValueError("clarification uses the structured clarification question")
            elif self.response_key is None:
                raise ValueError("direct answer plan requires a response key")

        if len(self.sub_queries) > self.policy.max_subqueries:
            raise ValueError("sub-query count exceeds trusted policy")
        if len(self.routes) > self.policy.max_routes:
            raise ValueError("route count exceeds trusted policy")
        if sum(route.max_cost for route in self.routes) > self.cost_budget + 1e-9:
            raise ValueError("route cost exceeds plan cost budget")

        sub_query_ids = [item.sub_query_id for item in self.sub_queries]
        if len(sub_query_ids) != len(set(sub_query_ids)):
            raise ValueError("sub-query IDs must be unique")
        facet_sub_query_ids = {
            sub_query_id
            for facet in self.facet_ledger.facets
            for sub_query_id in facet.sub_query_ids
        }
        unknown_facet_sub_queries = facet_sub_query_ids - set(sub_query_ids)
        if unknown_facet_sub_queries:
            raise ValueError(
                "facet ledger references unknown sub-query IDs: "
                f"{sorted(unknown_facet_sub_queries)}"
            )
        if any(
            facet.support_status != FacetSupportStatus.UNASSESSED or facet.evidence_ids
            for facet in self.facet_ledger.facets
        ):
            raise ValueError("query plan facet ledger must be unassessed")
        if self.facet_ledger.facets:
            query_hash = (
                "sha256:"
                + hashlib.sha256(self.understanding.standalone_query.encode("utf-8")).hexdigest()
            )
            if self.facet_ledger.source_query_hash != query_hash:
                raise ValueError("facet ledger source query hash does not match the plan")
        route_ids = [item.route_id for item in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route IDs must be unique")
        route_types = [item.route_type for item in self.routes]
        if len(route_types) != len(set(route_types)):
            raise ValueError("route types must be unique")

        known = set(sub_query_ids)
        dependency_graph: dict[str, tuple[str, ...]] = {}
        purpose_by_id: dict[str, SubQueryPurpose] = {}
        for item in self.sub_queries:
            if item.sub_query_id in item.depends_on:
                raise ValueError("sub-query cannot depend on itself")
            unknown = set(item.depends_on) - known
            if unknown:
                raise ValueError(f"sub-query depends on unknown IDs: {sorted(unknown)}")
            dependency_graph[item.sub_query_id] = item.depends_on
            purpose_by_id[item.sub_query_id] = item.purpose
        _validate_dag(dependency_graph, self.policy.max_dependency_depth)

        for route in self.routes:
            unknown = set(route.sub_query_ids) - known
            if unknown:
                raise ValueError(f"route references unknown sub-query IDs: {sorted(unknown)}")
            if any(
                purpose_by_id[item] != SubQueryPurpose.RETRIEVAL for item in route.sub_query_ids
            ):
                raise ValueError("routes may only execute retrieval sub-queries")
            if route.timeout_ms > self.policy.total_timeout_ms:
                raise ValueError("route timeout exceeds trusted policy")
            if route.top_k > self.policy.max_top_k:
                raise ValueError("route top_k exceeds trusted policy")
            if route.route_type == QueryRouteType.WEB and not self.policy.allow_web:
                raise ValueError("Web route is forbidden by trusted policy")
            if route.route_type == QueryRouteType.WEB and not self.understanding.need_freshness:
                raise ValueError("Web route requires an explicit freshness need")
            if route.route_type == QueryRouteType.HYDE and not self.policy.allow_hyde:
                raise ValueError("HyDE route is forbidden by trusted policy")
        return self


def _validate_dag(graph: dict[str, tuple[str, ...]], max_depth: int) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()
    depths: dict[str, int] = {}

    def visit(node: str) -> int:
        if node in visiting:
            raise ValueError("sub-query dependency graph contains a cycle")
        if node in visited:
            return depths[node]
        visiting.add(node)
        depth = 1
        for dependency in graph[node]:
            depth = max(depth, visit(dependency) + 1)
        visiting.remove(node)
        visited.add(node)
        depths[node] = depth
        if depth > max_depth:
            raise ValueError("sub-query dependency depth exceeds trusted policy")
        return depth

    for node in graph:
        visit(node)


def canonical_plan_input(value: Any) -> str:
    """返回生成稳定 plan_id 所需的规范 JSON。"""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)

    def json_default(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json", exclude_none=True)
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def make_plan_id(value: Any) -> str:
    digest = hashlib.sha256(canonical_plan_input(value).encode("utf-8")).hexdigest()
    return f"qpl_{digest[:32]}"


def validate_plan_for_execution(
    raw_plan: QueryPlan | dict[str, Any],
    *,
    trusted_scope: AuthorizedScopeRef,
    trusted_policy: PlanningPolicy,
) -> QueryPlan:
    """校验外部/LLM 计划并与服务端事实逐项比对。

    Pydantic schema 校验只能证明结构合法；本函数进一步证明 proposal 没有自行修改
    授权范围或放宽 Web/HyDE/预算策略。执行器必须调用本函数而不是直接反序列化。
    """

    plan = raw_plan if isinstance(raw_plan, QueryPlan) else QueryPlan.model_validate(raw_plan)
    if plan.authorized_scope_ref != trusted_scope:
        raise ValueError("query plan attempted to change the authorized scope")
    if plan.policy != trusted_policy:
        raise ValueError("query plan attempted to change the trusted planning policy")
    return plan


__all__ = [
    "AuthorizedScopeRef",
    "ConversationTurn",
    "FACET_EXTRACTOR_VERSION",
    "FACET_LEDGER_SCHEMA_VERSION",
    "LEGACY_FACET_LEDGER_SCHEMA_VERSION",
    "FacetClosure",
    "FacetLedger",
    "FacetNecessity",
    "FacetPolarity",
    "FacetSupportStatus",
    "MAX_QUERY_FACETS",
    "MAX_FACET_EVIDENCE_IDS",
    "PlanDisposition",
    "PlanningPolicy",
    "PLANNER_VERSION",
    "QueryAmbiguity",
    "QueryConstraint",
    "QueryFacet",
    "QueryIntent",
    "QueryLanguage",
    "QueryPlan",
    "QueryRoute",
    "QueryRouteType",
    "QuerySubQuery",
    "QueryUnderstanding",
    "RewriteDecision",
    "RewriteStatus",
    "RouteFallback",
    "RouteFilter",
    "SubQueryPurpose",
    "TemporalKind",
    "TemporalRequirement",
    "make_plan_id",
    "validate_plan_for_execution",
]
