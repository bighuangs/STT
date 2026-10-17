"""TODO-29 deterministic ContextPack construction.

This module is deliberately independent from the query graph and persistence layer.
Its input is the already-authorized, already-hydrated ``Evidence`` produced by
``prepare_reranked_evidence``.  It never queries a parent or neighbour implicitly:
optional expansion is possible only through an explicit resolver which must return
another real, stable ``Evidence`` in the same local publication scope.

The builder owns the evidence part of the generation budget.  The prompt builder
which consumes a ``ContextPack`` must still perform a final whole-prompt token check
after adding the real system prompt, history and question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.evidence import Evidence, canonical_json, sha256_digest
from app.domain.query_planning import (
    FACET_LEDGER_SCHEMA_VERSION,
    FacetLedger,
    FacetNecessity,
    FacetSupportStatus,
    MAX_FACET_EVIDENCE_IDS,
)


CONTEXT_PACK_SCHEMA_VERSION = "context-pack.v2"
LEGACY_CONTEXT_PACK_SCHEMA_VERSION = "context-pack.v1"
DEFAULT_CONTEXT_STRATEGY_VERSION = "context-pack-builder.v2"
TRUNCATION_MARKER = "\n…[证据正文因上下文预算被截断]"

ExpansionMode = Literal["none", "parent", "neighbors", "parent_and_neighbors"]
ExpansionRelation = Literal["parent", "previous", "next"]
Placement = Literal["front", "middle", "tail"]
TrustLevel = Literal["controlled_local", "external_web"]
InclusionReason = Literal[
    "sub_query_coverage",
    "ranked_value",
    "expanded_parent",
    "expanded_neighbor",
]
OmissionReason = Literal[
    "duplicate_evidence",
    "duplicate_content",
    "near_duplicate_content",
    "document_quota",
    "source_quota",
    "max_blocks",
    "token_budget",
    "character_budget",
    "expansion_scope_mismatch",
    "expansion_relation_not_requested",
    "expansion_out_of_window",
    "expansion_self_reference",
]


class ContextPackInputError(ValueError):
    """The caller did not supply a complete, stable Evidence contract."""


class ContextBudget(BaseModel):
    """Resolved model-window allocation.

    ``evidence_budget`` is explicit in the persisted contract.  Callers may use
    :meth:`from_window` to derive it from the actual model window and reservations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_context_limit: int = Field(ge=1)
    reserved_system: int = Field(default=0, ge=0)
    reserved_history: int = Field(default=0, ge=0)
    reserved_output: int = Field(default=0, ge=0)
    safety_margin: int = Field(default=0, ge=0)
    evidence_budget: int = Field(ge=0)
    evidence_character_budget: int | None = Field(default=None, ge=1)

    @classmethod
    def from_window(
        cls,
        *,
        model_context_limit: int,
        reserved_system: int = 0,
        reserved_history: int = 0,
        reserved_output: int = 0,
        safety_margin: int = 0,
        evidence_budget: int | None = None,
        evidence_character_budget: int | None = None,
    ) -> "ContextBudget":
        fixed = reserved_system + reserved_history + reserved_output + safety_margin
        available = model_context_limit - fixed
        if available < 0:
            raise ValueError("context reservations exceed model_context_limit")
        resolved = available if evidence_budget is None else evidence_budget
        return cls(
            model_context_limit=model_context_limit,
            reserved_system=reserved_system,
            reserved_history=reserved_history,
            reserved_output=reserved_output,
            safety_margin=safety_margin,
            evidence_budget=resolved,
            evidence_character_budget=evidence_character_budget,
        )

    @model_validator(mode="after")
    def validate_allocation(self) -> "ContextBudget":
        allocated = (
            self.reserved_system
            + self.reserved_history
            + self.reserved_output
            + self.safety_margin
            + self.evidence_budget
        )
        if allocated > self.model_context_limit:
            raise ValueError("context reservations plus evidence budget exceed model limit")
        return self


class ContextPackPolicy(BaseModel):
    """Versioned, deterministic packing policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_version: str = Field(
        default=DEFAULT_CONTEXT_STRATEGY_VERSION,
        min_length=1,
        max_length=128,
    )
    max_blocks: int = Field(default=20, ge=1, le=10_000)
    max_blocks_per_document: int | None = Field(default=5, ge=1, le=10_000)
    max_blocks_per_source: int | None = Field(default=8, ge=1, le=10_000)
    max_block_characters: int = Field(default=16_000, ge=32, le=4_000_000)
    min_truncated_tokens: int = Field(default=16, ge=1, le=100_000)
    near_duplicate_threshold: float | None = Field(default=0.96, ge=0.5, le=1.0)
    near_duplicate_scan_characters: int = Field(default=20_000, ge=128, le=200_000)
    expansion_mode: ExpansionMode = "none"
    neighbor_window: int = Field(default=1, ge=1, le=20)
    expansion_score_discount: float = Field(default=0.95, ge=0.0, le=1.0)


class ExpansionRequest(BaseModel):
    """Authorized publication scope passed to a parent/neighbour resolver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    document_version_id: str = Field(min_length=1, max_length=64)
    index_version_id: str = Field(min_length=1, max_length=64)
    chunk_id: str = Field(min_length=1, max_length=128)
    relations: tuple[ExpansionRelation, ...]
    neighbor_window: int = Field(ge=1, le=20)


class ExpansionCandidate(BaseModel):
    """A resolver response.

    The expanded unit must be a genuine ``Evidence``.  Free-form text is not
    accepted because it would have no stable citation target or publication scope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    relation: ExpansionRelation
    distance: int = Field(default=0, ge=0, le=20)
    evidence: Evidence
    score: float | None = None
    sub_query_ids: tuple[str, ...] = ()

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("expansion score must be finite")
        return value

    @field_validator("sub_query_ids")
    @classmethod
    def normalize_sub_queries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, field_name="sub_query_ids")

    @model_validator(mode="after")
    def validate_distance(self) -> "ExpansionCandidate":
        if self.relation == "parent" and self.distance != 0:
            raise ValueError("parent expansion distance must be zero")
        if self.relation in {"previous", "next"} and self.distance < 1:
            raise ValueError("neighbor expansion distance must be positive")
        return self


@runtime_checkable
class ContextExpansionResolver(Protocol):
    """Port implemented later by an authorized version-scoped data adapter."""

    def resolve(self, request: ExpansionRequest) -> Sequence[ExpansionCandidate]:
        """Return only existing Evidence; an empty result means no expansion exists."""


@runtime_checkable
class TokenCounter(Protocol):
    """Tokenizer boundary used by the deterministic packer."""

    @property
    def tokenizer_id(self) -> str: ...

    @property
    def tokenizer_version(self) -> str: ...

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class TiktokenTokenCounter:
    """Exact counter for a named tiktoken encoding, with an optional safety factor."""

    encoding_name: str = "cl100k_base"
    safety_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.safety_multiplier) or self.safety_multiplier < 1.0:
            raise ValueError("safety_multiplier must be finite and >= 1")
        # Fail during construction rather than half-way through a query.
        tiktoken.get_encoding(self.encoding_name)

    @property
    def tokenizer_id(self) -> str:
        return f"tiktoken:{self.encoding_name}"

    @property
    def tokenizer_version(self) -> str:
        return str(getattr(tiktoken, "__version__", "unknown"))

    def count(self, text: str) -> int:
        raw = len(tiktoken.get_encoding(self.encoding_name).encode(text))
        return int(ceil(raw * self.safety_multiplier))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        raw_limit = max(0, int(max_tokens // self.safety_multiplier))
        encoding = tiktoken.get_encoding(self.encoding_name)
        return encoding.decode(encoding.encode(text)[:raw_limit])


@dataclass(frozen=True)
class Utf8ByteUpperBoundCounter:
    """Dependency-free conservative fallback.

    Modern BPE tokens encode at least one UTF-8 byte, so byte count is a safe upper
    bound.  It is intentionally pessimistic and must be recorded as such.
    """

    @property
    def tokenizer_id(self) -> str:
        return "utf8-byte-upper-bound"

    @property
    def tokenizer_version(self) -> str:
        return "v1"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        payload = text.encode("utf-8")[:max_tokens]
        return payload.decode("utf-8", errors="ignore")


class ContextRelationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: ExpansionRelation
    anchor_evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    distance: int = Field(ge=0, le=20)


class BlockTruncation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasons: tuple[Literal["block_character_limit", "token_budget", "character_budget"], ...]
    original_characters: int = Field(ge=1)
    packed_characters: int = Field(ge=1)
    original_tokens: int = Field(ge=1)
    packed_tokens: int = Field(ge=1)
    marker: str = TRUNCATION_MARKER


class PackedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    representation_id: str = Field(pattern=r"^rep_[0-9a-f]{40}$")
    original_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=4_100_000)
    content_token_count: int = Field(ge=1)
    token_count: int = Field(ge=1)
    character_count: int = Field(ge=1)
    source_label: str = Field(default="", max_length=1024)
    document_id: str | None = Field(default=None, max_length=64)
    source_key: str = Field(min_length=1, max_length=8192)
    score: float
    hit_spans: tuple[dict[str, Any], ...]
    related_refs: tuple[ContextRelationRef, ...] = ()
    sub_query_ids: tuple[str, ...] = ()
    placement: Placement
    inclusion_reason: InclusionReason
    trust_level: TrustLevel
    truncation: BlockTruncation | None = None

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("packed block score must be finite")
        return value


class OmittedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    reason: OmissionReason
    detail: str = Field(default="", max_length=1000)
    duplicate_of_evidence_id: str | None = Field(
        default=None,
        pattern=r"^ev_[0-9a-f]{40}$",
    )
    estimated_tokens: int | None = Field(default=None, ge=0)
    sub_query_ids: tuple[str, ...] = ()


class CitationLabelMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    representation_id: str = Field(pattern=r"^rep_[0-9a-f]{40}$")


class FacetCoverageSummary(BaseModel):
    """不含问题正文的 Facet 检索分支绑定摘要。

    ``covered_facet_ids`` 是 v1 兼容字段，含义与
    ``branch_bound_facet_ids`` 相同，均不证明 Evidence 语义支持答案。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-facet-ledger.v1", "query-facet-ledger.v2"] = (
        FACET_LEDGER_SCHEMA_VERSION
    )
    binding_semantics: Literal["retrieval_branch_lineage_only"] = "retrieval_branch_lineage_only"
    required_facet_ids: tuple[str, ...] = ()
    conditional_facet_ids: tuple[str, ...] = ()
    branch_bound_facet_ids: tuple[str, ...] = ()
    covered_facet_ids: tuple[str, ...] = ()
    uncovered_facet_ids: tuple[str, ...] = ()
    uncovered_required_facet_ids: tuple[str, ...] = ()
    partial_facet_ids: tuple[str, ...] = ()
    unsupported_facet_ids: tuple[str, ...] = ()
    unassessed_facet_ids: tuple[str, ...] = ()
    evidence_count_by_facet: dict[str, int] = Field(default_factory=dict)


class ContextCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_sub_query_ids: tuple[str, ...] = ()
    packed_sub_query_ids: tuple[str, ...] = ()
    uncovered_sub_query_ids: tuple[str, ...] = ()
    evidence_count_by_sub_query: dict[str, int] = Field(default_factory=dict)
    facet_coverage: FacetCoverageSummary = Field(default_factory=FacetCoverageSummary)


class ContextPackDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_candidate_count: int = Field(ge=0)
    expansion_candidate_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    packed_count: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    truncated_count: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    saved_tokens: int = Field(ge=0)
    token_utilization: float = Field(ge=0.0, le=1.0)
    character_utilization: float | None = Field(default=None, ge=0.0, le=1.0)
    expansion_status: Literal[
        "disabled",
        "resolver_unavailable",
        "completed",
        "completed_with_issues",
    ]
    expansion_issues: tuple[str, ...] = ()
    facet_count: int = Field(default=0, ge=0)
    required_facet_count: int = Field(default=0, ge=0)
    supported_facet_count: int = Field(default=0, ge=0)
    branch_bound_facet_count: int = Field(default=0, ge=0)
    uncovered_required_facet_count: int = Field(default=0, ge=0)


class ContextPack(BaseModel):
    """Serializable, replayable evidence context passed to prompt construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context-pack.v1", "context-pack.v2"] = CONTEXT_PACK_SCHEMA_VERSION
    query_run_id: str = Field(min_length=1, max_length=128)
    tokenizer_id: str = Field(min_length=1, max_length=255)
    tokenizer_version: str = Field(min_length=1, max_length=128)
    strategy_version: str = Field(min_length=1, max_length=128)
    budget: ContextBudget
    packed_blocks: tuple[PackedBlock, ...]
    omitted: tuple[OmittedEvidence, ...]
    label_mapping: tuple[CitationLabelMapping, ...]
    coverage: ContextCoverage
    facet_ledger: FacetLedger = Field(default_factory=FacetLedger)
    context_text: str
    used_tokens: int = Field(ge=0)
    used_characters: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    diagnostics: ContextPackDiagnostics

    @model_validator(mode="after")
    def validate_pack_invariants(self) -> "ContextPack":
        labels = [block.label for block in self.packed_blocks]
        if labels != [f"E{index}" for index in range(1, len(labels) + 1)]:
            raise ValueError("packed block labels must be contiguous and ordered")
        mapped = [(item.label, item.evidence_id) for item in self.label_mapping]
        expected = [(item.label, item.evidence_id) for item in self.packed_blocks]
        if mapped != expected:
            raise ValueError("citation label mapping must match packed block order")
        if self.used_tokens > self.budget.evidence_budget:
            raise ValueError("ContextPack exceeds evidence token budget")
        if (
            self.budget.evidence_character_budget is not None
            and self.used_characters > self.budget.evidence_character_budget
        ):
            raise ValueError("ContextPack exceeds evidence character budget")
        if self.coverage.facet_coverage != _facet_coverage_summary(self.facet_ledger):
            raise ValueError("facet coverage summary must match the bound facet ledger")
        return self


@dataclass(frozen=True)
class ContextPackBuildResult:
    """Pack plus aliases compatible with TODO-08 citation validation."""

    pack: ContextPack
    citation_aliases: tuple[dict[str, Any], ...]


@dataclass
class _Candidate:
    evidence: Evidence
    document: dict[str, Any]
    score: float
    original_rank: int
    sub_query_ids: tuple[str, ...]
    relation: ExpansionRelation | None = None
    anchor_evidence_id: str | None = None
    distance: int = 0

    @property
    def evidence_id(self) -> str:
        assert self.evidence.evidence_id is not None
        return self.evidence.evidence_id

    @property
    def prompt_content(self) -> str:
        """返回通过内容安全门禁的模型输入表示。

        Evidence 本体仍保留原始正文/hash 以维持引用完整性；只有 ContextPack
        渲染、预算和近重复判断消费该安全表示。
        """

        secured = self.document.get("security_safe_content")
        return str(secured) if secured is not None else self.evidence.content


@dataclass(frozen=True)
class _FittedText:
    text: str
    rendered: str
    reasons: tuple[Literal["block_character_limit", "token_budget", "character_budget"], ...]
    original_tokens: int


def _unique_identifiers(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"{field_name} cannot contain blank identifiers")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} cannot contain duplicate identifiers")
    return normalized


def _extract_sub_query_ids(document: Mapping[str, Any]) -> tuple[str, ...]:
    raw: Any = document.get("sub_query_ids")
    # 兼容 TODO-27/28 早期查询图使用过的无下划线字段；新链路统一写
    # ``sub_query_ids``。兼容读取使旧缓存/重放输入不会静默丢失 coverage。
    if raw is None:
        raw = document.get("subquery_ids")
    if raw is None and document.get("sub_query_id") is not None:
        raw = (document["sub_query_id"],)
    evidence = document.get("evidence")
    retrieval = evidence.get("retrieval") if isinstance(evidence, Mapping) else None
    metadata = retrieval.get("metadata") if isinstance(retrieval, Mapping) else None
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("sub_query_ids")
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, Sequence):
        raise ContextPackInputError("sub_query_ids must be a string sequence")
    return _unique_identifiers(raw, field_name="sub_query_ids")


def _candidate_score(document: Mapping[str, Any], evidence: Evidence) -> float:
    values = (
        document.get("rerank_score"),
        document.get("score"),
        document.get("rrf_score"),
        document.get("retrieval_score"),
        evidence.retrieval.rerank_score if evidence.retrieval else None,
        evidence.retrieval.rrf_score if evidence.retrieval else None,
        evidence.retrieval.raw_score if evidence.retrieval else None,
    )
    for raw in values:
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            return value
    return 0.0


def _normalize_candidates(documents: Sequence[Mapping[str, Any]]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for rank, raw_document in enumerate(documents, start=1):
        document = dict(raw_document)
        raw_evidence = document.get("evidence")
        if raw_evidence is None:
            raise ContextPackInputError(
                f"candidate at rank {rank} has no hydrated Evidence; "
                "call prepare_reranked_evidence first"
            )
        try:
            evidence = Evidence.model_validate(raw_evidence)
        except ValueError as exc:
            raise ContextPackInputError(
                f"candidate at rank {rank} contains invalid Evidence"
            ) from exc
        candidates.append(
            _Candidate(
                evidence=evidence,
                document=document,
                score=_candidate_score(document, evidence),
                original_rank=rank,
                sub_query_ids=_extract_sub_query_ids(document),
            )
        )
    return candidates


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, str, int]:
    return (-candidate.score, candidate.evidence_id, candidate.original_rank)


def _requested_relations(policy: ContextPackPolicy) -> tuple[ExpansionRelation, ...]:
    if policy.expansion_mode == "none":
        return ()
    if policy.expansion_mode == "parent":
        return ("parent",)
    if policy.expansion_mode == "neighbors":
        return ("previous", "next")
    return ("parent", "previous", "next")


def _local_scope(evidence: Evidence) -> tuple[str, str, str, str, str] | None:
    source = evidence.source
    if source.source_type != "local":
        return None
    assert source.tenant_id is not None
    assert source.kb_id is not None
    assert source.document_id is not None
    assert source.document_version_id is not None
    assert source.index_version_id is not None
    return (
        source.tenant_id,
        source.kb_id,
        source.document_id,
        source.document_version_id,
        source.index_version_id,
    )


def _expansion_request(
    candidate: _Candidate,
    relations: tuple[ExpansionRelation, ...],
    policy: ContextPackPolicy,
) -> ExpansionRequest | None:
    source = candidate.evidence.source
    if source.source_type != "local":
        return None
    assert source.tenant_id is not None
    assert source.kb_id is not None
    assert source.document_id is not None
    assert source.document_version_id is not None
    assert source.index_version_id is not None
    assert source.chunk_id is not None
    return ExpansionRequest(
        anchor_evidence_id=candidate.evidence_id,
        tenant_id=source.tenant_id,
        kb_id=source.kb_id,
        document_id=source.document_id,
        document_version_id=source.document_version_id,
        index_version_id=source.index_version_id,
        chunk_id=source.chunk_id,
        relations=relations,
        neighbor_window=policy.neighbor_window,
    )


def _collect_expansions(
    base_candidates: Sequence[_Candidate],
    *,
    resolver: ContextExpansionResolver | None,
    policy: ContextPackPolicy,
) -> tuple[list[_Candidate], list[OmittedEvidence], list[str], str]:
    relations = _requested_relations(policy)
    if not relations:
        return [], [], [], "disabled"
    if resolver is None:
        return [], [], ["EXPANSION_RESOLVER_UNAVAILABLE"], "resolver_unavailable"

    accepted: list[_Candidate] = []
    omitted: list[OmittedEvidence] = []
    issues: list[str] = []
    next_rank = len(base_candidates) + 1
    for anchor in sorted(base_candidates, key=_candidate_sort_key):
        request = _expansion_request(anchor, relations, policy)
        if request is None:
            issues.append(f"EXPANSION_UNSUPPORTED_SOURCE:{anchor.evidence_id}")
            continue
        try:
            resolved = resolver.resolve(request)
        except Exception as exc:  # resolver failures must not remove the original hit
            issues.append(f"EXPANSION_RESOLVER_FAILED:{anchor.evidence_id}:{type(exc).__name__}")
            continue
        validated: list[ExpansionCandidate] = []
        for raw in resolved:
            try:
                item = ExpansionCandidate.model_validate(raw)
            except ValueError as exc:
                issues.append(
                    f"EXPANSION_INVALID_RESPONSE:{anchor.evidence_id}:{type(exc).__name__}"
                )
                continue
            validated.append(item)
        validated.sort(
            key=lambda item: (
                item.relation,
                item.distance,
                item.evidence.evidence_id or "",
            )
        )
        for item in validated:
            evidence_id = item.evidence.evidence_id
            assert evidence_id is not None
            reason: OmissionReason | None = None
            detail = ""
            if item.anchor_evidence_id != anchor.evidence_id or evidence_id == anchor.evidence_id:
                reason = "expansion_self_reference"
                detail = "expansion must reference its requested anchor and a distinct Evidence"
            elif item.relation not in relations:
                reason = "expansion_relation_not_requested"
                detail = f"relation {item.relation} was not requested"
            elif item.relation in {"previous", "next"} and item.distance > policy.neighbor_window:
                reason = "expansion_out_of_window"
                detail = (
                    f"neighbor distance {item.distance} exceeds window {policy.neighbor_window}"
                )
            elif _local_scope(item.evidence) != _local_scope(anchor.evidence):
                reason = "expansion_scope_mismatch"
                detail = "expanded Evidence is outside anchor publication scope"
            if reason is not None:
                omitted.append(
                    OmittedEvidence(
                        evidence_id=evidence_id,
                        reason=reason,
                        detail=detail,
                        sub_query_ids=item.sub_query_ids,
                    )
                )
                continue

            document = {
                "evidence_id": evidence_id,
                "evidence": item.evidence.model_dump(mode="json", exclude_none=True),
                "title": item.evidence.source_label,
                "text": item.evidence.content,
                "content": item.evidence.content,
                "type": item.evidence.source.source_type,
                "context_relation": item.relation,
                "anchor_evidence_id": item.anchor_evidence_id,
            }
            score = (
                item.score
                if item.score is not None
                else anchor.score * policy.expansion_score_discount
            )
            accepted.append(
                _Candidate(
                    evidence=item.evidence,
                    document=document,
                    score=score,
                    original_rank=next_rank,
                    sub_query_ids=item.sub_query_ids or anchor.sub_query_ids,
                    relation=item.relation,
                    anchor_evidence_id=item.anchor_evidence_id,
                    distance=item.distance,
                )
            )
            next_rank += 1

    status = "completed_with_issues" if issues or omitted else "completed"
    return accepted, omitted, sorted(issues), status


def _normalized_text(value: str, *, limit: int) -> str:
    return "".join(character.lower() for character in value[:limit] if character.isalnum())


def _shingles(value: str, *, width: int = 5) -> frozenset[str]:
    if len(value) <= width:
        return frozenset({value}) if value else frozenset()
    return frozenset(value[index : index + width] for index in range(len(value) - width + 1))


def _near_duplicate(
    left: str,
    right: str,
    *,
    threshold: float,
) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    length_ratio = min(len(left), len(right)) / max(len(left), len(right))
    if length_ratio < threshold:
        return False
    left_shingles = _shingles(left)
    right_shingles = _shingles(right)
    union = left_shingles | right_shingles
    if not union:
        return False
    return len(left_shingles & right_shingles) / len(union) >= threshold


def _merge_candidate_context(retained: _Candidate, duplicate: _Candidate) -> None:
    retained.sub_query_ids = tuple(
        dict.fromkeys((*retained.sub_query_ids, *duplicate.sub_query_ids))
    )
    if duplicate.score > retained.score:
        retained.score = duplicate.score


def _deduplicate_candidates(
    candidates: Sequence[_Candidate],
    *,
    policy: ContextPackPolicy,
    counter: TokenCounter,
) -> tuple[list[_Candidate], list[OmittedEvidence]]:
    retained: list[_Candidate] = []
    omitted: list[OmittedEvidence] = []
    by_evidence: dict[str, _Candidate] = {}
    by_hash: dict[str, _Candidate] = {}
    normalized: dict[str, str] = {}
    for candidate in sorted(candidates, key=_candidate_sort_key):
        evidence_id = candidate.evidence_id
        duplicate = by_evidence.get(evidence_id)
        reason: OmissionReason | None = None
        if duplicate is not None:
            reason = "duplicate_evidence"
        else:
            content_hash = candidate.evidence.content_hash
            assert content_hash is not None
            duplicate = by_hash.get(content_hash)
            if duplicate is not None:
                reason = "duplicate_content"
        if duplicate is None and policy.near_duplicate_threshold is not None:
            current = _normalized_text(
                candidate.prompt_content,
                limit=policy.near_duplicate_scan_characters,
            )
            for existing in retained:
                existing_text = normalized[existing.evidence_id]
                if _near_duplicate(
                    current,
                    existing_text,
                    threshold=policy.near_duplicate_threshold,
                ):
                    duplicate = existing
                    reason = "near_duplicate_content"
                    break
        if duplicate is not None and reason is not None:
            _merge_candidate_context(duplicate, candidate)
            omitted.append(
                OmittedEvidence(
                    evidence_id=evidence_id,
                    reason=reason,
                    duplicate_of_evidence_id=duplicate.evidence_id,
                    estimated_tokens=counter.count(candidate.prompt_content),
                    sub_query_ids=candidate.sub_query_ids,
                )
            )
            continue

        retained.append(candidate)
        by_evidence[evidence_id] = candidate
        assert candidate.evidence.content_hash is not None
        by_hash[candidate.evidence.content_hash] = candidate
        normalized[evidence_id] = _normalized_text(
            candidate.prompt_content,
            limit=policy.near_duplicate_scan_characters,
        )
    return retained, omitted


def _coverage_order(
    candidates: Sequence[_Candidate],
    required_sub_query_ids: tuple[str, ...],
) -> tuple[list[_Candidate], set[str]]:
    ranked = sorted(candidates, key=_candidate_sort_key)
    selected: list[_Candidate] = []
    selected_ids: set[str] = set()
    coverage_selected: set[str] = set()
    for sub_query_id in required_sub_query_ids:
        if any(sub_query_id in item.sub_query_ids for item in selected):
            continue
        candidate = next(
            (
                item
                for item in ranked
                if item.evidence_id not in selected_ids and sub_query_id in item.sub_query_ids
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.evidence_id)
            coverage_selected.add(candidate.evidence_id)
    selected.extend(item for item in ranked if item.evidence_id not in selected_ids)
    return selected, coverage_selected


def _location_label(evidence: Evidence) -> str:
    labels: list[str] = []
    for span in evidence.provenance_spans:
        if span.availability == "unavailable":
            continue
        if span.page_no:
            labels.append(f"第{span.page_no}页")
        elif span.slide_no:
            labels.append(f"第{span.slide_no}张幻灯片")
        elif span.sheet:
            cell = f" {span.cell_range}" if span.cell_range else ""
            labels.append(f"工作表 {span.sheet}{cell}")
        elif span.line_start:
            labels.append(f"第{span.line_start}–{span.line_end or span.line_start}行")
    return "、".join(dict.fromkeys(labels)) if labels else "位置不可用"


def _relation_label(candidate: _Candidate) -> str:
    if candidate.relation == "parent":
        return "父块"
    if candidate.relation == "previous":
        return f"前邻块（距离 {candidate.distance}）"
    if candidate.relation == "next":
        return f"后邻块（距离 {candidate.distance}）"
    return "原始命中块"


def _render_block(label: str, candidate: _Candidate, text: str) -> str:
    source = candidate.evidence.source_label or candidate.evidence_id
    return "\n".join(
        (
            f"[{label}]（仅可用此标签引用）",
            f"来源：{source}",
            f"位置：{_location_label(candidate.evidence)}",
            f"上下文关系：{_relation_label(candidate)}",
            "证据正文：",
            text,
            f"[/{label}]",
        )
    )


def _joined_context(rendered_blocks: Sequence[str]) -> str:
    return "\n\n".join(rendered_blocks)


def _fits(
    rendered_blocks: Sequence[str],
    rendered: str,
    *,
    budget: ContextBudget,
    counter: TokenCounter,
) -> bool:
    context = _joined_context((*rendered_blocks, rendered))
    if counter.count(context) > budget.evidence_budget:
        return False
    return (
        budget.evidence_character_budget is None or len(context) <= budget.evidence_character_budget
    )


def _fit_text(
    candidate: _Candidate,
    label: str,
    rendered_blocks: Sequence[str],
    *,
    budget: ContextBudget,
    policy: ContextPackPolicy,
    counter: TokenCounter,
) -> _FittedText | None:
    original = candidate.prompt_content
    original_tokens = counter.count(original)
    reasons: list[Literal["block_character_limit", "token_budget", "character_budget"]] = []
    bounded = original
    if len(bounded) > policy.max_block_characters:
        bounded = bounded[: policy.max_block_characters]
        reasons.append("block_character_limit")
    displayed = bounded + (TRUNCATION_MARKER if reasons else "")
    rendered = _render_block(label, candidate, displayed)
    if _fits(rendered_blocks, rendered, budget=budget, counter=counter):
        return _FittedText(
            text=displayed,
            rendered=rendered,
            reasons=tuple(reasons),
            original_tokens=original_tokens,
        )

    current_context = _joined_context(rendered_blocks)
    if counter.count(current_context) >= budget.evidence_budget:
        return None
    if (
        budget.evidence_character_budget is not None
        and len(current_context) >= budget.evidence_character_budget
    ):
        return None

    if "token_budget" not in reasons:
        reasons.append("token_budget")
    if budget.evidence_character_budget is not None:
        empty_with_marker = _render_block(label, candidate, TRUNCATION_MARKER)
        if len(_joined_context((*rendered_blocks, empty_with_marker))) >= (
            budget.evidence_character_budget
        ):
            if "character_budget" not in reasons:
                reasons.append("character_budget")

    # Find a deterministic extractive prefix.  Every trial counts the complete
    # rendered context, so wrapper/separator tokens are never forgotten.
    low = 0
    high = len(bounded)
    best_text = ""
    while low <= high:
        middle = (low + high) // 2
        prefix = bounded[:middle].rstrip()
        trial_text = prefix + TRUNCATION_MARKER if prefix else TRUNCATION_MARKER
        trial_rendered = _render_block(label, candidate, trial_text)
        if _fits(rendered_blocks, trial_rendered, budget=budget, counter=counter):
            best_text = prefix
            low = middle + 1
        else:
            high = middle - 1

    if not best_text.strip():
        return None
    if counter.count(best_text) < policy.min_truncated_tokens:
        return None
    final_text = best_text + TRUNCATION_MARKER
    final_rendered = _render_block(label, candidate, final_text)
    if not _fits(rendered_blocks, final_rendered, budget=budget, counter=counter):
        return None
    if budget.evidence_character_budget is not None and len(final_text) < len(bounded):
        if "character_budget" not in reasons:
            reasons.append("character_budget")
    return _FittedText(
        text=final_text,
        rendered=final_rendered,
        reasons=tuple(dict.fromkeys(reasons)),
        original_tokens=original_tokens,
    )


def _source_and_document_keys(candidate: _Candidate) -> tuple[str, str, str | None]:
    source = candidate.evidence.source
    if source.source_type == "local":
        assert source.document_id is not None
        key = f"local:{source.document_id}"
        return key, key, source.document_id
    assert source.source_url is not None
    parsed = urlsplit(source.source_url)
    source_key = f"web-domain:{parsed.hostname or parsed.netloc}"
    return f"web-url:{source.source_url}", source_key, None


def _related_refs(candidate: _Candidate) -> tuple[ContextRelationRef, ...]:
    if candidate.relation is None or candidate.anchor_evidence_id is None:
        return ()
    return (
        ContextRelationRef(
            relation=candidate.relation,
            anchor_evidence_id=candidate.anchor_evidence_id,
            distance=candidate.distance,
        ),
    )


def _representation_id(
    candidate: _Candidate,
    text: str,
    *,
    strategy_version: str,
) -> str:
    payload = {
        "evidence_id": candidate.evidence_id,
        "content_hash": candidate.evidence.content_hash,
        "text_hash": sha256_digest(text),
        "relation": candidate.relation,
        "anchor_evidence_id": candidate.anchor_evidence_id,
        "distance": candidate.distance,
        "strategy_version": strategy_version,
    }
    return f"rep_{sha256_digest(canonical_json(payload)).split(':', 1)[1][:40]}"


def _inclusion_reason(
    candidate: _Candidate,
    *,
    coverage_selected: set[str],
) -> InclusionReason:
    if candidate.relation == "parent":
        return "expanded_parent"
    if candidate.relation in {"previous", "next"}:
        return "expanded_neighbor"
    if candidate.evidence_id in coverage_selected:
        return "sub_query_coverage"
    return "ranked_value"


def _omission_for_no_fit(
    candidate: _Candidate,
    rendered_blocks: Sequence[str],
    *,
    budget: ContextBudget,
    counter: TokenCounter,
) -> OmittedEvidence:
    current_context = _joined_context(rendered_blocks)
    character_exhausted = (
        budget.evidence_character_budget is not None
        and len(current_context) >= budget.evidence_character_budget
    )
    reason: OmissionReason = "character_budget" if character_exhausted else "token_budget"
    return OmittedEvidence(
        evidence_id=candidate.evidence_id,
        reason=reason,
        detail="not enough budget for the minimum extractive evidence prefix",
        estimated_tokens=counter.count(candidate.prompt_content),
        sub_query_ids=candidate.sub_query_ids,
    )


def _normalize_facet_ledger(
    raw: FacetLedger | Mapping[str, Any] | None,
) -> FacetLedger:
    if raw is None:
        return FacetLedger()
    if isinstance(raw, FacetLedger):
        return raw
    if not isinstance(raw, Mapping):
        raise ContextPackInputError("facet_ledger must be a mapping")
    try:
        return FacetLedger.model_validate(dict(raw))
    except (TypeError, ValueError) as exc:
        raise ContextPackInputError(f"invalid facet_ledger: {exc}") from exc


def _bind_facet_ledger(
    ledger: FacetLedger,
    blocks: Sequence[PackedBlock],
) -> FacetLedger:
    """仅按真实分支标记绑定 Evidence，不执行语义支持判断。"""

    covered_sub_queries = {sub_query_id for block in blocks for sub_query_id in block.sub_query_ids}
    facets = []
    for facet in ledger.facets:
        required_sub_queries = set(facet.sub_query_ids)
        evidence_ids = tuple(
            dict.fromkeys(
                block.evidence_id
                for block in blocks
                if required_sub_queries.intersection(block.sub_query_ids)
            )
        )
        covered = required_sub_queries.intersection(covered_sub_queries)
        if not required_sub_queries:
            status = FacetSupportStatus.UNASSESSED
            evidence_ids = ()
        elif not evidence_ids:
            status = FacetSupportStatus.UNSUPPORTED
        elif covered == required_sub_queries:
            status = FacetSupportStatus.SUPPORTED
        else:
            status = FacetSupportStatus.PARTIAL
        facets.append(
            facet.__class__.model_validate(
                {
                    **facet.model_dump(mode="python"),
                    "support_status": status,
                    "evidence_ids": evidence_ids,
                }
            )
        )
    return FacetLedger(
        schema_version=ledger.schema_version,
        extractor_version=ledger.extractor_version,
        source_query_hash=ledger.source_query_hash,
        binding_semantics=ledger.binding_semantics,
        facets=tuple(facets),
    )


def _facet_coverage_summary(ledger: FacetLedger) -> FacetCoverageSummary:
    required = tuple(
        facet.facet_id for facet in ledger.facets if facet.necessity == FacetNecessity.REQUIRED
    )
    conditional = tuple(
        facet.facet_id for facet in ledger.facets if facet.necessity == FacetNecessity.CONDITIONAL
    )
    covered = tuple(
        facet.facet_id
        for facet in ledger.facets
        if facet.support_status == FacetSupportStatus.SUPPORTED
    )
    partial = tuple(
        facet.facet_id
        for facet in ledger.facets
        if facet.support_status == FacetSupportStatus.PARTIAL
    )
    unsupported = tuple(
        facet.facet_id
        for facet in ledger.facets
        if facet.support_status == FacetSupportStatus.UNSUPPORTED
    )
    unassessed = tuple(
        facet.facet_id
        for facet in ledger.facets
        if facet.support_status == FacetSupportStatus.UNASSESSED
    )
    covered_set = set(covered)
    uncovered = tuple(
        facet.facet_id for facet in ledger.facets if facet.facet_id not in covered_set
    )
    required_set = set(required)
    return FacetCoverageSummary(
        schema_version=ledger.schema_version,
        binding_semantics=ledger.binding_semantics,
        required_facet_ids=required,
        conditional_facet_ids=conditional,
        branch_bound_facet_ids=covered,
        covered_facet_ids=covered,
        uncovered_facet_ids=uncovered,
        uncovered_required_facet_ids=tuple(
            facet_id for facet_id in uncovered if facet_id in required_set
        ),
        partial_facet_ids=partial,
        unsupported_facet_ids=unsupported,
        unassessed_facet_ids=unassessed,
        evidence_count_by_facet={
            facet.facet_id: len(facet.evidence_ids) for facet in ledger.facets
        },
    )


def _coverage(
    required: tuple[str, ...],
    blocks: Sequence[PackedBlock],
    *,
    facet_ledger: FacetLedger,
) -> ContextCoverage:
    counts = {
        sub_query_id: sum(1 for block in blocks if sub_query_id in block.sub_query_ids)
        for sub_query_id in required
    }
    packed = tuple(sub_query_id for sub_query_id in required if counts[sub_query_id] > 0)
    uncovered = tuple(sub_query_id for sub_query_id in required if counts[sub_query_id] == 0)
    return ContextCoverage(
        required_sub_query_ids=required,
        packed_sub_query_ids=packed,
        uncovered_sub_query_ids=uncovered,
        evidence_count_by_sub_query=counts,
        facet_coverage=_facet_coverage_summary(facet_ledger),
    )


def _pack_hash_payload(
    *,
    tokenizer_id: str,
    tokenizer_version: str,
    strategy_version: str,
    budget: ContextBudget,
    blocks: Sequence[PackedBlock],
    omitted: Sequence[OmittedEvidence],
    coverage: ContextCoverage,
    facet_ledger: FacetLedger,
    context_text: str,
) -> dict[str, Any]:
    """Hash replay-relevant content, excluding the per-execution query_run_id."""

    return {
        "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
        "tokenizer_id": tokenizer_id,
        "tokenizer_version": tokenizer_version,
        "strategy_version": strategy_version,
        "budget": budget.model_dump(mode="json"),
        "packed_blocks": [block.model_dump(mode="json", exclude={"score"}) for block in blocks],
        "omitted": [item.model_dump(mode="json") for item in omitted],
        "coverage": coverage.model_dump(mode="json"),
        "facet_ledger": facet_ledger.model_dump(mode="json"),
        "context_text": context_text,
    }


def replay_context_pack_content_hash(pack: ContextPack) -> str:
    """从完整可重放字段重算 ContextPack 内容身份。"""

    payload = _pack_hash_payload(
        tokenizer_id=pack.tokenizer_id,
        tokenizer_version=pack.tokenizer_version,
        strategy_version=pack.strategy_version,
        budget=pack.budget,
        blocks=pack.packed_blocks,
        omitted=pack.omitted,
        coverage=pack.coverage,
        facet_ledger=pack.facet_ledger,
        context_text=pack.context_text,
    )
    return sha256_digest(canonical_json(payload))


def build_context_pack(
    documents: Sequence[Mapping[str, Any]],
    *,
    query_run_id: str,
    budget: ContextBudget,
    counter: TokenCounter | None = None,
    policy: ContextPackPolicy | None = None,
    required_sub_query_ids: Sequence[str] = (),
    facet_ledger: FacetLedger | Mapping[str, Any] | None = None,
    expansion_resolver: ContextExpansionResolver | None = None,
) -> ContextPackBuildResult:
    """Build a deterministic, budget-bounded ContextPack.

    ``documents`` must be the output of ``prepare_reranked_evidence`` (or an
    equivalent trusted adapter).  Missing/invalid Evidence is rejected rather than
    silently promoted from arbitrary text.
    """

    if not str(query_run_id).strip():
        raise ContextPackInputError("query_run_id cannot be blank")
    if len(str(query_run_id)) > 128:
        raise ContextPackInputError("query_run_id cannot exceed 128 characters")
    counter = counter or TiktokenTokenCounter()
    policy = policy or ContextPackPolicy()
    required = _unique_identifiers(
        required_sub_query_ids,
        field_name="required_sub_query_ids",
    )
    input_facet_ledger = _normalize_facet_ledger(facet_ledger)
    if input_facet_ledger.facets and policy.max_blocks > MAX_FACET_EVIDENCE_IDS:
        raise ContextPackInputError(
            "max_blocks exceeds the facet Evidence binding limit "
            f"({policy.max_blocks}>{MAX_FACET_EVIDENCE_IDS})"
        )

    base = _normalize_candidates(documents)
    expansions, expansion_omitted, expansion_issues, expansion_status = _collect_expansions(
        base,
        resolver=expansion_resolver,
        policy=policy,
    )
    candidates, deduplicated = _deduplicate_candidates(
        (*base, *expansions),
        policy=policy,
        counter=counter,
    )
    ordered, coverage_selected = _coverage_order(candidates, required)

    omitted: list[OmittedEvidence] = [*expansion_omitted, *deduplicated]
    packed_intermediate: list[tuple[_Candidate, _FittedText, str, str, str | None]] = []
    rendered_blocks: list[str] = []
    document_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for candidate in ordered:
        if len(packed_intermediate) >= policy.max_blocks:
            omitted.append(
                OmittedEvidence(
                    evidence_id=candidate.evidence_id,
                    reason="max_blocks",
                    estimated_tokens=counter.count(candidate.prompt_content),
                    sub_query_ids=candidate.sub_query_ids,
                )
            )
            continue
        document_key, source_key, document_id = _source_and_document_keys(candidate)
        if (
            policy.max_blocks_per_document is not None
            and document_counts.get(document_key, 0) >= policy.max_blocks_per_document
        ):
            omitted.append(
                OmittedEvidence(
                    evidence_id=candidate.evidence_id,
                    reason="document_quota",
                    detail=f"document quota={policy.max_blocks_per_document}",
                    estimated_tokens=counter.count(candidate.prompt_content),
                    sub_query_ids=candidate.sub_query_ids,
                )
            )
            continue
        if (
            policy.max_blocks_per_source is not None
            and source_counts.get(source_key, 0) >= policy.max_blocks_per_source
        ):
            omitted.append(
                OmittedEvidence(
                    evidence_id=candidate.evidence_id,
                    reason="source_quota",
                    detail=f"source quota={policy.max_blocks_per_source}",
                    estimated_tokens=counter.count(candidate.prompt_content),
                    sub_query_ids=candidate.sub_query_ids,
                )
            )
            continue

        label = f"E{len(packed_intermediate) + 1}"
        fitted = _fit_text(
            candidate,
            label,
            rendered_blocks,
            budget=budget,
            policy=policy,
            counter=counter,
        )
        if fitted is None:
            omitted.append(
                _omission_for_no_fit(
                    candidate,
                    rendered_blocks,
                    budget=budget,
                    counter=counter,
                )
            )
            continue
        packed_intermediate.append((candidate, fitted, label, source_key, document_id))
        rendered_blocks.append(fitted.rendered)
        document_counts[document_key] = document_counts.get(document_key, 0) + 1
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

    blocks: list[PackedBlock] = []
    aliases: list[dict[str, Any]] = []
    total_blocks = len(packed_intermediate)
    for index, (candidate, fitted, label, source_key, document_id) in enumerate(
        packed_intermediate
    ):
        if index == 0:
            placement: Placement = "front"
        elif index == total_blocks - 1:
            placement = "tail"
        else:
            placement = "middle"
        representation_id = _representation_id(
            candidate,
            fitted.text,
            strategy_version=policy.strategy_version,
        )
        truncation = None
        if fitted.reasons:
            truncation = BlockTruncation(
                reasons=fitted.reasons,
                original_characters=len(candidate.prompt_content),
                packed_characters=len(fitted.text),
                original_tokens=fitted.original_tokens,
                packed_tokens=counter.count(fitted.text),
            )
        assert candidate.evidence.content_hash is not None
        block = PackedBlock(
            label=label,
            evidence_id=candidate.evidence_id,
            representation_id=representation_id,
            original_content_hash=candidate.evidence.content_hash,
            text=fitted.text,
            content_token_count=counter.count(fitted.text),
            token_count=counter.count(fitted.rendered),
            character_count=len(fitted.rendered),
            source_label=candidate.evidence.source_label,
            document_id=document_id,
            source_key=source_key,
            score=candidate.score,
            hit_spans=tuple(
                span.model_dump(mode="json", exclude_none=True)
                for span in candidate.evidence.provenance_spans
            ),
            related_refs=_related_refs(candidate),
            sub_query_ids=candidate.sub_query_ids,
            placement=placement,
            inclusion_reason=_inclusion_reason(
                candidate,
                coverage_selected=coverage_selected,
            ),
            trust_level=(
                "controlled_local"
                if candidate.evidence.source.source_type == "local"
                else "external_web"
            ),
            truncation=truncation,
        )
        blocks.append(block)
        aliases.append(
            {
                "label": label,
                "evidence": candidate.evidence.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "document": candidate.document,
            }
        )

    context_text = _joined_context(rendered_blocks)
    used_tokens = counter.count(context_text)
    used_characters = len(context_text)
    bound_facet_ledger = _bind_facet_ledger(input_facet_ledger, blocks)
    coverage = _coverage(required, blocks, facet_ledger=bound_facet_ledger)
    label_mapping = tuple(
        CitationLabelMapping(
            label=block.label,
            evidence_id=block.evidence_id,
            representation_id=block.representation_id,
        )
        for block in blocks
    )
    estimated_input_tokens = sum(
        counter.count(item.prompt_content) for item in (*base, *expansions)
    )
    token_utilization = used_tokens / budget.evidence_budget if budget.evidence_budget else 0.0
    character_utilization = (
        used_characters / budget.evidence_character_budget
        if budget.evidence_character_budget
        else None
    )
    diagnostics = ContextPackDiagnostics(
        input_candidate_count=len(base),
        expansion_candidate_count=len(expansions),
        deduplicated_count=len(deduplicated),
        packed_count=len(blocks),
        omitted_count=len(omitted),
        truncated_count=sum(block.truncation is not None for block in blocks),
        estimated_input_tokens=estimated_input_tokens,
        saved_tokens=max(0, estimated_input_tokens - used_tokens),
        token_utilization=min(1.0, token_utilization),
        character_utilization=(
            min(1.0, character_utilization) if character_utilization is not None else None
        ),
        expansion_status=expansion_status,
        expansion_issues=tuple(expansion_issues),
        facet_count=len(bound_facet_ledger.facets),
        required_facet_count=len(coverage.facet_coverage.required_facet_ids),
        supported_facet_count=len(coverage.facet_coverage.covered_facet_ids),
        branch_bound_facet_count=len(coverage.facet_coverage.branch_bound_facet_ids),
        uncovered_required_facet_count=len(coverage.facet_coverage.uncovered_required_facet_ids),
    )
    hash_payload = _pack_hash_payload(
        tokenizer_id=counter.tokenizer_id,
        tokenizer_version=counter.tokenizer_version,
        strategy_version=policy.strategy_version,
        budget=budget,
        blocks=blocks,
        omitted=omitted,
        coverage=coverage,
        facet_ledger=bound_facet_ledger,
        context_text=context_text,
    )
    pack = ContextPack(
        query_run_id=str(query_run_id).strip(),
        tokenizer_id=counter.tokenizer_id,
        tokenizer_version=counter.tokenizer_version,
        strategy_version=policy.strategy_version,
        budget=budget,
        packed_blocks=tuple(blocks),
        omitted=tuple(omitted),
        label_mapping=label_mapping,
        coverage=coverage,
        facet_ledger=bound_facet_ledger,
        context_text=context_text,
        used_tokens=used_tokens,
        used_characters=used_characters,
        content_hash=sha256_digest(canonical_json(hash_payload)),
        diagnostics=diagnostics,
    )
    return ContextPackBuildResult(pack=pack, citation_aliases=tuple(aliases))


__all__ = [
    "CONTEXT_PACK_SCHEMA_VERSION",
    "ContextBudget",
    "ContextExpansionResolver",
    "ContextPack",
    "ContextPackBuildResult",
    "ContextPackDiagnostics",
    "replay_context_pack_content_hash",
    "ContextPackInputError",
    "ContextPackPolicy",
    "ContextRelationRef",
    "ExpansionCandidate",
    "ExpansionRequest",
    "FacetCoverageSummary",
    "LEGACY_CONTEXT_PACK_SCHEMA_VERSION",
    "OmittedEvidence",
    "PackedBlock",
    "TiktokenTokenCounter",
    "TokenCounter",
    "Utf8ByteUpperBoundCounter",
    "build_context_pack",
]
