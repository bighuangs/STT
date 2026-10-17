"""TODO-16 TOC、RAPTOR 与 GraphRAG 的隔离领域契约。

高级索引只负责路由到真实 ``Evidence``。目录标题、生成摘要、实体和关系都不是
可直接引用的原文；它们必须携带可校验 provenance，最终候选的引用身份始终来自
``Evidence.evidence_id``。
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.evidence import Evidence


ADVANCED_INDEX_SCHEMA_VERSION = "advanced-index.v1"
ADVANCED_SNAPSHOT_SCHEMA_VERSION = "advanced-snapshot.v1"
ADVANCED_RESULT_SCHEMA_VERSION = "advanced-retrieval-result.v1"

AdvancedIndexType = Literal["toc", "raptor", "graphrag"]
AdvancedIndexStatus = Literal[
    "staging",
    "validated",
    "active",
    "stale",
    "retired",
    "failed",
]
AdvancedRoute = Literal["toc", "raptor", "graphrag"]
AdvancedRouteStatus = Literal["succeeded", "empty", "skipped", "failed"]
AdvancedRouteReason = Literal[
    "ROUTE_COMPLETED",
    "NO_MATCH",
    "FEATURE_DISABLED",
    "ROUTE_NOT_REQUESTED",
    "SNAPSHOT_MISSING",
    "INDEX_NOT_ACTIVE",
    "SOURCE_INDEX_MISMATCH",
    "INDEX_STALE",
    "INTEGRITY_ERROR",
    "BUDGET_EXHAUSTED",
]
StructureNodeType = Literal["toc", "heading", "content"]
ProvenanceKind = Literal[
    "toc_node",
    "heading_node",
    "raptor_summary",
    "graph_entity",
    "graph_relation",
    "graph_community",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_EVIDENCE_ID = re.compile(r"^ev_[0-9a-f]{40}$")


def _identifier(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized or not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} is not a safe identifier")
    return normalized


def _required_text(value: str, *, name: str, limit: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be blank")
    if len(normalized) > limit:
        raise ValueError(f"{name} cannot exceed {limit} characters")
    return normalized


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def _unique_ids(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _unique_evidence_ids(
    values: tuple[str, ...],
    *,
    name: str,
) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{name} requires at least one Evidence ID")
    if any(not _EVIDENCE_ID.fullmatch(value) for value in values):
        raise ValueError(f"{name} must contain stable Evidence IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


class AdvancedFeatureFlags(BaseModel):
    """所有高级路线默认关闭；评测通过后按 KB 显式开启。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    toc_enabled: bool = False
    raptor_enabled: bool = False
    graphrag_enabled: bool = False

    def enabled(self, route: AdvancedRoute) -> bool:
        return {
            "toc": self.toc_enabled,
            "raptor": self.raptor_enabled,
            "graphrag": self.graphrag_enabled,
        }[route]


class AdvancedRetrievalBudget(BaseModel):
    """一次高级检索的确定性硬预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_total_candidates: int = Field(default=20, ge=1, le=1000)
    max_candidates_per_route: int = Field(default=10, ge=1, le=1000)
    toc_node_limit: int = Field(default=8, ge=1, le=1000)
    toc_expansion_limit: int = Field(default=30, ge=1, le=10_000)
    raptor_summary_limit: int = Field(default=5, ge=1, le=1000)
    raptor_leaf_limit: int = Field(default=20, ge=1, le=10_000)
    graph_entity_limit: int = Field(default=12, ge=1, le=10_000)
    graph_edge_limit: int = Field(default=20, ge=1, le=10_000)
    graph_hops: int = Field(default=1, ge=1, le=2)
    max_provenance_per_candidate: int = Field(default=8, ge=1, le=100)


class AdvancedRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=32_000)
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    active_source_index_version_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    requested_routes: tuple[AdvancedRoute, ...] = ()
    planner_reasons: dict[AdvancedRoute, str] = Field(default_factory=dict)
    flags: AdvancedFeatureFlags = Field(default_factory=AdvancedFeatureFlags)
    budget: AdvancedRetrievalBudget = Field(default_factory=AdvancedRetrievalBudget)

    @field_validator(
        "tenant_id",
        "kb_id",
        "active_source_index_version_id",
        "trace_id",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return _required_text(value, name="query", limit=32_000)

    @field_validator("requested_routes")
    @classmethod
    def unique_routes(
        cls,
        value: tuple[AdvancedRoute, ...],
    ) -> tuple[AdvancedRoute, ...]:
        if len(value) != len(set(value)):
            raise ValueError("requested advanced routes must be unique")
        return value


class AdvancedEvaluationGate(BaseModel):
    """高级索引发布前必须固定的离线评测与回滚证明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str = Field(min_length=1, max_length=128)
    holdout_dataset_id: str = Field(min_length=1, max_length=128)
    passed: bool = False
    quality_delta: float = 0.0
    citation_coverage_delta: float = 0.0
    p95_latency_ms: int = Field(default=0, ge=0)
    estimated_cost_per_query: float = Field(default=0.0, ge=0.0)
    estimated_storage_bytes: int = Field(default=0, ge=0)
    budget_approved: bool = False
    rollback_ready: bool = False
    report_uri: str | None = Field(default=None, max_length=4096)

    @field_validator(
        "quality_delta",
        "citation_coverage_delta",
        "estimated_cost_per_query",
    )
    @classmethod
    def finite_metrics(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("advanced evaluation metrics must be finite")
        return value

    @field_validator("evaluation_id", "holdout_dataset_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_passed_gate(self) -> "AdvancedEvaluationGate":
        if self.passed and (
            not self.budget_approved or not self.rollback_ready or not self.report_uri
        ):
            raise ValueError("passed evaluation requires budget approval, rollback and report")
        return self


class AdvancedIndexVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-index.v1"] = ADVANCED_INDEX_SCHEMA_VERSION
    advanced_index_id: str = Field(min_length=1, max_length=128)
    index_type: AdvancedIndexType
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    scope: Literal["document", "kb"]
    document_id: str | None = Field(default=None, max_length=128)
    version: int = Field(ge=1)
    source_index_version_id: str = Field(min_length=1, max_length=128)
    source_document_version_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    status: AdvancedIndexStatus = "staging"
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    evaluation: AdvancedEvaluationGate | None = None
    created_by: str = Field(min_length=1, max_length=128)
    created_at: datetime
    published_at: datetime | None = None
    stale_reason: str | None = Field(default=None, max_length=2000)

    @field_validator(
        "advanced_index_id",
        "tenant_id",
        "kb_id",
        "document_id",
        "source_index_version_id",
        "created_by",
    )
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, name=info.field_name)

    @field_validator("source_document_version_ids")
    @classmethod
    def normalize_document_versions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_ids(value, name="source_document_version_ids")

    @field_validator("created_at", "published_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "AdvancedIndexVersion":
        if self.scope == "document" and self.document_id is None:
            raise ValueError("document-scoped advanced index requires document_id")
        if self.scope == "kb" and self.document_id is not None:
            raise ValueError("KB-scoped advanced index cannot contain document_id")
        if self.status == "active":
            if self.published_at is None:
                raise ValueError("active advanced index requires published_at")
            if self.evaluation is None or not self.evaluation.passed:
                raise ValueError("active advanced index requires a passed evaluation")
        elif self.published_at is not None:
            raise ValueError("only an active advanced index can contain published_at")
        if self.status == "stale" and not self.stale_reason:
            raise ValueError("stale advanced index requires stale_reason")
        if self.status != "stale" and self.stale_reason:
            raise ValueError("only stale advanced index can contain stale_reason")
        return self


class StructureNode(BaseModel):
    """目录/标题节点；``source_evidence_ids`` 永远指向正文 Evidence。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1, max_length=128)
    node_type: StructureNodeType
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=128)
    document_version_id: str = Field(min_length=1, max_length=128)
    source_index_version_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=2000)
    toc_path: tuple[str, ...] = Field(default_factory=tuple, max_length=128)
    order: int = Field(ge=0)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    parent_node_id: str | None = Field(default=None, max_length=128)
    child_node_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10_000)
    previous_node_id: str | None = Field(default=None, max_length=128)
    next_node_id: str | None = Field(default=None, max_length=128)

    @field_validator(
        "node_id",
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "source_index_version_id",
        "parent_node_id",
        "previous_node_id",
        "next_node_id",
    )
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, name=info.field_name)

    @field_validator("child_node_ids")
    @classmethod
    def normalize_children(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_ids(value, name="child_node_ids")

    @field_validator("source_evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="source_evidence_ids")

    @field_validator("toc_path")
    @classmethod
    def normalize_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_required_text(item, name="TOC path item", limit=2000) for item in value)
        return normalized

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, name="title", limit=2000)

    @model_validator(mode="after")
    def validate_semantics(self) -> "StructureNode":
        if self.node_type in {"toc", "heading"} and self.title is None:
            raise ValueError("TOC/heading nodes require a title")
        references = {
            value
            for value in (
                self.parent_node_id,
                self.previous_node_id,
                self.next_node_id,
            )
            if value is not None
        } | set(self.child_node_ids)
        if self.node_id in references:
            raise ValueError("structure node cannot reference itself")
        if self.toc_path and self.title and self.toc_path[-1] != self.title:
            raise ValueError("the last TOC path item must equal the node title")
        return self


class StructureSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advanced_index_id: str = Field(min_length=1, max_length=128)
    source_index_version_id: str = Field(min_length=1, max_length=128)
    nodes: tuple[StructureNode, ...] = Field(min_length=1, max_length=1_000_000)

    @field_validator("advanced_index_id", "source_index_version_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_graph(self) -> "StructureSnapshot":
        by_id = {item.node_id: item for item in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("structure node IDs must be unique")
        if any(item.source_index_version_id != self.source_index_version_id for item in self.nodes):
            raise ValueError("structure nodes must use the snapshot source index")
        order_keys = [(item.document_version_id, item.order) for item in self.nodes]
        if len(order_keys) != len(set(order_keys)):
            raise ValueError("structure node order must be unique per document version")
        for node in self.nodes:
            referenced = [
                value
                for value in (
                    node.parent_node_id,
                    node.previous_node_id,
                    node.next_node_id,
                    *node.child_node_ids,
                )
                if value is not None
            ]
            missing = [value for value in referenced if value not in by_id]
            if missing:
                raise ValueError(
                    f"structure node {node.node_id} has dangling references: {missing}"
                )
            for reference in referenced:
                other = by_id[reference]
                if (
                    other.tenant_id,
                    other.kb_id,
                    other.document_id,
                    other.document_version_id,
                ) != (
                    node.tenant_id,
                    node.kb_id,
                    node.document_id,
                    node.document_version_id,
                ):
                    raise ValueError("structure edges cannot cross document publication scope")
            if node.parent_node_id is not None:
                parent = by_id[node.parent_node_id]
                if node.node_id not in parent.child_node_ids:
                    raise ValueError("parent/child structure references must be symmetric")
            for child_id in node.child_node_ids:
                if by_id[child_id].parent_node_id != node.node_id:
                    raise ValueError("child/parent structure references must be symmetric")
            if node.next_node_id is not None:
                if by_id[node.next_node_id].previous_node_id != node.node_id:
                    raise ValueError("next/previous structure references must be symmetric")
            if node.previous_node_id is not None:
                if by_id[node.previous_node_id].next_node_id != node.node_id:
                    raise ValueError("previous/next structure references must be symmetric")
        for start in by_id:
            visited: set[str] = set()
            cursor: str | None = start
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("structure parent graph cannot contain a cycle")
                visited.add(cursor)
                cursor = by_id[cursor].parent_node_id
        return self


class RaptorSummaryNode(BaseModel):
    """生成摘要只作路由；叶子 Evidence 是唯一可引用来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary_id: str = Field(min_length=1, max_length=128)
    level: int = Field(ge=1, le=100)
    text: str = Field(min_length=1, max_length=1_000_000)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    source_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    child_summary_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=100_000)
    cluster_method: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)

    @field_validator(
        "summary_id",
        "cluster_method",
        "prompt_id",
        "prompt_version",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("source_evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="source_evidence_ids")

    @field_validator("source_chunk_ids", "child_summary_ids")
    @classmethod
    def normalize_reference_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique_ids(value, name=info.field_name)

    @field_validator("text", "model_id")
    @classmethod
    def normalize_text(cls, value: str, info) -> str:
        limit = 1_000_000 if info.field_name == "text" else 512
        return _required_text(value, name=info.field_name, limit=limit)

    @model_validator(mode="after")
    def validate_self_reference(self) -> "RaptorSummaryNode":
        if self.summary_id in self.child_summary_ids:
            raise ValueError("RAPTOR summary cannot reference itself")
        return self


class RaptorSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advanced_index_id: str = Field(min_length=1, max_length=128)
    tree_version: str = Field(min_length=1, max_length=128)
    source_index_version_id: str = Field(min_length=1, max_length=128)
    summaries: tuple[RaptorSummaryNode, ...] = Field(min_length=1, max_length=1_000_000)

    @field_validator("advanced_index_id", "tree_version", "source_index_version_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_tree(self) -> "RaptorSnapshot":
        by_id = {item.summary_id: item for item in self.summaries}
        if len(by_id) != len(self.summaries):
            raise ValueError("RAPTOR summary IDs must be unique")
        for summary in self.summaries:
            missing = [child for child in summary.child_summary_ids if child not in by_id]
            if missing:
                raise ValueError(
                    f"RAPTOR summary {summary.summary_id} has missing children: {missing}"
                )
            for child_id in summary.child_summary_ids:
                child = by_id[child_id]
                if child.level >= summary.level:
                    raise ValueError("RAPTOR child summary level must be below its parent")
                if not set(child.source_evidence_ids).issubset(summary.source_evidence_ids):
                    raise ValueError(
                        "RAPTOR parent provenance must contain all child leaf Evidence"
                    )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(summary_id: str) -> None:
            if summary_id in visiting:
                raise ValueError("RAPTOR summary graph cannot contain a cycle")
            if summary_id in visited:
                return
            visiting.add(summary_id)
            for child_id in by_id[summary_id].child_summary_ids:
                visit(child_id)
            visiting.remove(summary_id)
            visited.add(summary_id)

        for summary_id in by_id:
            visit(summary_id)
        return self


class GraphEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1, max_length=128)
    canonical_name: str = Field(min_length=1, max_length=1000)
    entity_type: str = Field(min_length=1, max_length=128)
    aliases: tuple[str, ...] = Field(default_factory=tuple, max_length=1000)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)

    @field_validator("entity_id", "entity_type")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("canonical_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _required_text(value, name="canonical_name", limit=1000)

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_required_text(item, name="entity alias", limit=1000) for item in value)
        if len(normalized) != len(set(item.casefold() for item in normalized)):
            raise ValueError("entity aliases must be unique")
        return normalized

    @field_validator("source_evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="source_evidence_ids")


class GraphRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str = Field(min_length=1, max_length=128)
    source_entity_id: str = Field(min_length=1, max_length=128)
    target_entity_id: str = Field(min_length=1, max_length=128)
    relation_type: str = Field(min_length=1, max_length=128)
    statement: str = Field(min_length=1, max_length=4000)
    inferred: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)

    @field_validator(
        "relation_id",
        "source_entity_id",
        "target_entity_id",
        "relation_type",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        return _required_text(value, name="statement", limit=4000)

    @field_validator("source_evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="source_evidence_ids")

    @field_validator("confidence")
    @classmethod
    def finite_confidence(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("relation confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_relation(self) -> "GraphRelation":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("graph relation cannot be a self-loop")
        if self.inferred and self.confidence is None:
            raise ValueError("inferred graph relation requires confidence")
        return self


class GraphCommunity(BaseModel):
    """社区摘要仍是生成路由文本，必须保留关系与叶子来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    community_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2000)
    summary: str = Field(min_length=1, max_length=100_000)
    entity_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    relation_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=100_000)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)

    @field_validator("community_id", "prompt_id", "prompt_version")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @field_validator("entity_ids", "relation_ids")
    @classmethod
    def normalize_reference_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique_ids(value, name=info.field_name)

    @field_validator("source_evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="source_evidence_ids")


class GraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advanced_index_id: str = Field(min_length=1, max_length=128)
    graph_version: str = Field(min_length=1, max_length=128)
    source_index_version_id: str = Field(min_length=1, max_length=128)
    entities: tuple[GraphEntity, ...] = Field(min_length=1, max_length=1_000_000)
    relations: tuple[GraphRelation, ...] = Field(default_factory=tuple, max_length=2_000_000)
    communities: tuple[GraphCommunity, ...] = Field(
        default_factory=tuple,
        max_length=1_000_000,
    )

    @field_validator("advanced_index_id", "graph_version", "source_index_version_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _identifier(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_graph(self) -> "GraphSnapshot":
        entities = {item.entity_id: item for item in self.entities}
        relations = {item.relation_id: item for item in self.relations}
        communities = {item.community_id: item for item in self.communities}
        if len(entities) != len(self.entities):
            raise ValueError("graph entity IDs must be unique")
        if len(relations) != len(self.relations):
            raise ValueError("graph relation IDs must be unique")
        if len(communities) != len(self.communities):
            raise ValueError("graph community IDs must be unique")
        for relation in self.relations:
            if (
                relation.source_entity_id not in entities
                or relation.target_entity_id not in entities
            ):
                raise ValueError(
                    f"graph relation {relation.relation_id} references a missing entity"
                )
            shared_entity_provenance = set(
                entities[relation.source_entity_id].source_evidence_ids
            ) & set(entities[relation.target_entity_id].source_evidence_ids)
            if not set(relation.source_evidence_ids).issubset(shared_entity_provenance):
                raise ValueError("graph relation provenance must support both endpoint entities")
        for community in self.communities:
            if not set(community.entity_ids).issubset(entities):
                raise ValueError(f"graph community {community.community_id} has missing entities")
            if not set(community.relation_ids).issubset(relations):
                raise ValueError(f"graph community {community.community_id} has missing relations")
            required_evidence = {
                evidence_id
                for relation_id in community.relation_ids
                for evidence_id in relations[relation_id].source_evidence_ids
            }
            if not required_evidence.issubset(community.source_evidence_ids):
                raise ValueError("graph community provenance must contain all relation Evidence")
        return self


class AdvancedIndexSnapshot(BaseModel):
    """高级索引制品及其完整 provenance 边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-snapshot.v1"] = ADVANCED_SNAPSHOT_SCHEMA_VERSION
    index: AdvancedIndexVersion
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=2_000_000)
    structure: StructureSnapshot | None = None
    raptor: RaptorSnapshot | None = None
    graph: GraphSnapshot | None = None

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_evidence_ids(value, name="evidence_ids")

    @model_validator(mode="after")
    def validate_payload_and_provenance(self) -> "AdvancedIndexSnapshot":
        payloads = {
            "toc": self.structure,
            "raptor": self.raptor,
            "graphrag": self.graph,
        }
        if payloads[self.index.index_type] is None:
            raise ValueError("advanced snapshot is missing its type-specific payload")
        if sum(value is not None for value in payloads.values()) != 1:
            raise ValueError("advanced snapshot must contain exactly one route payload")
        payload = payloads[self.index.index_type]
        assert payload is not None
        if payload.advanced_index_id != self.index.advanced_index_id:
            raise ValueError("advanced payload ID must match its index version")
        if payload.source_index_version_id != self.index.source_index_version_id:
            raise ValueError("advanced payload source index must match its index version")

        declared = set(self.evidence_ids)
        referenced: set[str] = set()
        if self.structure is not None:
            referenced.update(
                evidence_id
                for node in self.structure.nodes
                for evidence_id in node.source_evidence_ids
            )
        if self.raptor is not None:
            referenced.update(
                evidence_id
                for summary in self.raptor.summaries
                for evidence_id in summary.source_evidence_ids
            )
        if self.graph is not None:
            referenced.update(
                evidence_id
                for entity in self.graph.entities
                for evidence_id in entity.source_evidence_ids
            )
            referenced.update(
                evidence_id
                for relation in self.graph.relations
                for evidence_id in relation.source_evidence_ids
            )
            referenced.update(
                evidence_id
                for community in self.graph.communities
                for evidence_id in community.source_evidence_ids
            )
        if not referenced.issubset(declared):
            raise ValueError("advanced snapshot contains provenance outside its Evidence manifest")
        return self


class AdvancedSupportingProvenance(BaseModel):
    """同一路线、同一 Evidence 的次级路由依据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_kind: ProvenanceKind
    routing_node_id: str = Field(min_length=1, max_length=128)
    routing_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=2000)
    generated_routing_text: bool = False

    @field_validator("routing_node_id")
    @classmethod
    def normalize_node_id(cls, value: str) -> str:
        return _identifier(value, name="routing_node_id")

    @field_validator("routing_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("advanced supporting score must be finite")
        return value


class AdvancedRouteObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: AdvancedRoute
    provenance_kind: ProvenanceKind
    routing_node_id: str = Field(min_length=1, max_length=128)
    routing_score: float = Field(ge=0.0, le=1.0)
    content_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    explanation: str = Field(min_length=1, max_length=2000)
    generated_routing_text: bool = False
    supporting_provenance: tuple[AdvancedSupportingProvenance, ...] = Field(
        default_factory=tuple,
        max_length=100,
    )

    @field_validator("routing_node_id")
    @classmethod
    def normalize_node_id(cls, value: str) -> str:
        return _identifier(value, name="routing_node_id")

    @field_validator("routing_score", "content_score", "combined_score")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("advanced retrieval scores must be finite")
        return value

    @model_validator(mode="after")
    def unique_supporting_nodes(self) -> "AdvancedRouteObservation":
        node_ids = [item.routing_node_id for item in self.supporting_provenance]
        if self.routing_node_id in node_ids:
            raise ValueError("primary route node cannot repeat as supporting provenance")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("supporting route node IDs must be unique")
        return self


class AdvancedRetrievalCandidate(BaseModel):
    """高级路线最终只能返回真实 Evidence。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: Evidence
    observations: tuple[AdvancedRouteObservation, ...] = Field(min_length=1, max_length=100)
    final_score: float = Field(ge=0.0, le=1.0)

    @field_validator("final_score")
    @classmethod
    def finite_final_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("advanced retrieval final score must be finite")
        return value

    @model_validator(mode="after")
    def validate_real_local_evidence(self) -> "AdvancedRetrievalCandidate":
        if self.evidence.source.source_type != "local":
            raise ValueError("advanced local indexes can only return local Evidence")
        if self.evidence.evidence_id is None:
            raise ValueError("advanced candidate requires a stable Evidence ID")
        routes = [item.route for item in self.observations]
        if len(routes) != len(set(routes)):
            raise ValueError("candidate may contain at most one observation per route")
        return self


class AdvancedRouteOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: AdvancedRoute
    status: AdvancedRouteStatus
    reason: AdvancedRouteReason
    matched_node_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    omitted_missing_evidence: int = Field(default=0, ge=0)
    budget_omitted_count: int = Field(default=0, ge=0)
    message: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_reason(self) -> "AdvancedRouteOutcome":
        if self.status == "succeeded" and self.reason != "ROUTE_COMPLETED":
            raise ValueError("successful route must use ROUTE_COMPLETED")
        if self.status == "empty" and self.reason != "NO_MATCH":
            raise ValueError("empty route must use NO_MATCH")
        return self


class AdvancedRetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_routes: tuple[AdvancedRoute, ...]
    enabled_routes: tuple[AdvancedRoute, ...]
    outcomes: tuple[AdvancedRouteOutcome, ...]
    source_index_version_id: str
    active_source_index_version_id: str
    used_candidate_budget: int = Field(ge=0)
    max_candidate_budget: int = Field(ge=1)
    degraded_to_baseline: bool


class AdvancedRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-retrieval-result.v1"] = ADVANCED_RESULT_SCHEMA_VERSION
    trace_id: str = Field(min_length=1, max_length=128)
    tenant_id: str
    kb_id: str
    candidates: tuple[AdvancedRetrievalCandidate, ...]
    diagnostics: AdvancedRetrievalDiagnostics


__all__ = [
    "AdvancedEvaluationGate",
    "AdvancedFeatureFlags",
    "AdvancedIndexSnapshot",
    "AdvancedIndexStatus",
    "AdvancedIndexType",
    "AdvancedIndexVersion",
    "AdvancedRetrievalBudget",
    "AdvancedRetrievalCandidate",
    "AdvancedRetrievalDiagnostics",
    "AdvancedRetrievalRequest",
    "AdvancedRetrievalResult",
    "AdvancedRoute",
    "AdvancedRouteObservation",
    "AdvancedRouteOutcome",
    "AdvancedSupportingProvenance",
    "GraphCommunity",
    "GraphEntity",
    "GraphRelation",
    "GraphSnapshot",
    "RaptorSnapshot",
    "RaptorSummaryNode",
    "StructureNode",
    "StructureSnapshot",
]
