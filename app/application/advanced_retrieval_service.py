"""TODO-16 高级索引生命周期、完整性校验与离线确定性检索。

该模块不修改现有查询图。调用方后续可把结果作为普通 ``Evidence`` 路线送入
TODO-28 融合；摘要、TOC 和图节点只贡献路由观察，绝不会成为伪造 Evidence。
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import re
from typing import Any, Protocol, runtime_checkable

from app.domain.advanced_retrieval import (
    AdvancedIndexSnapshot,
    AdvancedRetrievalCandidate,
    AdvancedRetrievalDiagnostics,
    AdvancedRetrievalRequest,
    AdvancedRetrievalResult,
    AdvancedRoute,
    AdvancedRouteObservation,
    AdvancedRouteOutcome,
    AdvancedSupportingProvenance,
    GraphSnapshot,
    RaptorSnapshot,
    StructureSnapshot,
)
from app.domain.evidence import Evidence


class AdvancedRetrievalError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        missing_evidence_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.missing_evidence_count = missing_evidence_count


@runtime_checkable
class AdvancedIndexRepository(Protocol):
    def save_snapshot(self, snapshot: AdvancedIndexSnapshot) -> None: ...

    def get_snapshot(self, advanced_index_id: str) -> AdvancedIndexSnapshot | None: ...

    def list_snapshots(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[AdvancedIndexSnapshot, ...]: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _validated_copy(model, **updates):
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return type(model).model_validate(payload)


def validate_advanced_snapshot(
    snapshot: AdvancedIndexSnapshot,
    evidence_by_id: Mapping[str, Evidence],
) -> None:
    """发布/查询共用的强完整性校验。

    缺失、跨租户、跨 KB、跨普通索引或已变更文档版本都会失败；调用方只能降级，
    不能把高级节点文本包装成 Evidence 继续回答。
    """

    invalid_keys = [key for key, evidence in evidence_by_id.items() if evidence.evidence_id != key]
    if invalid_keys:
        raise AdvancedRetrievalError(
            "EVIDENCE_CATALOG_KEY_MISMATCH",
            "Evidence catalog keys must equal stable evidence_id",
        )
    missing = [
        evidence_id for evidence_id in snapshot.evidence_ids if evidence_id not in evidence_by_id
    ]
    if missing:
        raise AdvancedRetrievalError(
            "MISSING_PROVENANCE",
            "advanced index references unavailable Evidence",
            missing_evidence_count=len(missing),
        )
    allowed_document_versions = set(snapshot.index.source_document_version_ids)
    for evidence_id in snapshot.evidence_ids:
        evidence = evidence_by_id[evidence_id]
        source = evidence.source
        if source.source_type != "local":
            raise AdvancedRetrievalError(
                "NON_LOCAL_PROVENANCE",
                "advanced local index cannot reference Web Evidence",
            )
        if source.tenant_id != snapshot.index.tenant_id or source.kb_id != snapshot.index.kb_id:
            raise AdvancedRetrievalError(
                "CROSS_SCOPE_PROVENANCE",
                "advanced index Evidence crosses tenant or KB scope",
            )
        if source.index_version_id != snapshot.index.source_index_version_id:
            raise AdvancedRetrievalError(
                "SOURCE_INDEX_MISMATCH",
                "advanced index Evidence belongs to a different source index",
            )
        if source.document_version_id not in allowed_document_versions:
            raise AdvancedRetrievalError(
                "DOCUMENT_VERSION_MISMATCH",
                "advanced index Evidence belongs to an undeclared document version",
            )
        if snapshot.index.scope == "document" and source.document_id != snapshot.index.document_id:
            raise AdvancedRetrievalError(
                "DOCUMENT_SCOPE_MISMATCH",
                "document-scoped advanced index contains another document",
            )


class AdvancedIndexLifecycleService:
    """staging → validated → active、stale、rollback 的隔离生命周期。"""

    def __init__(
        self,
        repository: AdvancedIndexRepository,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.repository = repository
        self.clock = clock

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise AdvancedRetrievalError(
                "INVALID_CLOCK",
                "advanced index clock must include timezone",
            )
        return value.astimezone(timezone.utc)

    def register_staging(self, snapshot: AdvancedIndexSnapshot) -> AdvancedIndexSnapshot:
        if snapshot.index.status != "staging":
            raise AdvancedRetrievalError(
                "INDEX_NOT_STAGING",
                "new advanced snapshot must be staging",
            )
        if self.repository.get_snapshot(snapshot.index.advanced_index_id) is not None:
            raise AdvancedRetrievalError(
                "INDEX_ALREADY_EXISTS",
                "advanced index ID already exists",
            )
        self.repository.save_snapshot(snapshot)
        return snapshot

    def validate(
        self,
        advanced_index_id: str,
        *,
        evidence_by_id: Mapping[str, Evidence],
    ) -> AdvancedIndexSnapshot:
        snapshot = self.repository.get_snapshot(advanced_index_id)
        if snapshot is None:
            raise AdvancedRetrievalError("INDEX_NOT_FOUND", "advanced index was not found")
        if snapshot.index.status != "staging":
            raise AdvancedRetrievalError(
                "INDEX_NOT_STAGING",
                "only a staging advanced index can be validated",
            )
        validate_advanced_snapshot(snapshot, evidence_by_id)
        updated = _validated_copy(
            snapshot,
            index=_validated_copy(snapshot.index, status="validated"),
        )
        self.repository.save_snapshot(updated)
        return updated

    def publish(
        self,
        advanced_index_id: str,
        *,
        active_source_index_version_id: str,
    ) -> AdvancedIndexSnapshot:
        snapshot = self.repository.get_snapshot(advanced_index_id)
        if snapshot is None:
            raise AdvancedRetrievalError("INDEX_NOT_FOUND", "advanced index was not found")
        if snapshot.index.status != "validated":
            raise AdvancedRetrievalError(
                "INDEX_NOT_VALIDATED",
                "only a validated advanced index can be published",
            )
        if snapshot.index.source_index_version_id != active_source_index_version_id:
            raise AdvancedRetrievalError(
                "SOURCE_INDEX_MISMATCH",
                "advanced index source is no longer the active ordinary index",
            )
        gate = snapshot.index.evaluation
        if gate is None or not gate.passed:
            raise AdvancedRetrievalError(
                "EVALUATION_GATE_FAILED",
                "advanced index cannot publish before its holdout gate passes",
            )
        for previous in self.repository.list_snapshots(
            snapshot.index.tenant_id,
            snapshot.index.kb_id,
        ):
            if (
                previous.index.index_type == snapshot.index.index_type
                and previous.index.status == "active"
                and previous.index.advanced_index_id != snapshot.index.advanced_index_id
            ):
                self.repository.save_snapshot(
                    _validated_copy(
                        previous,
                        index=_validated_copy(
                            previous.index,
                            status="retired",
                            published_at=None,
                        ),
                    )
                )
        active = _validated_copy(
            snapshot,
            index=_validated_copy(
                snapshot.index,
                status="active",
                published_at=self._now(),
            ),
        )
        self.repository.save_snapshot(active)
        return active

    def mark_stale_for_source_change(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        active_source_index_version_id: str,
        reason: str = "SOURCE_INDEX_REPLACED",
    ) -> tuple[AdvancedIndexSnapshot, ...]:
        stale: list[AdvancedIndexSnapshot] = []
        for snapshot in self.repository.list_snapshots(tenant_id, kb_id):
            if (
                snapshot.index.status == "active"
                and snapshot.index.source_index_version_id != active_source_index_version_id
            ):
                updated = _validated_copy(
                    snapshot,
                    index=_validated_copy(
                        snapshot.index,
                        status="stale",
                        published_at=None,
                        stale_reason=reason,
                    ),
                )
                self.repository.save_snapshot(updated)
                stale.append(updated)
        return tuple(stale)

    def rollback(
        self,
        advanced_index_id: str,
        *,
        active_source_index_version_id: str,
    ) -> AdvancedIndexSnapshot:
        target = self.repository.get_snapshot(advanced_index_id)
        if target is None:
            raise AdvancedRetrievalError("INDEX_NOT_FOUND", "rollback index was not found")
        if target.index.status not in {"retired", "validated"}:
            raise AdvancedRetrievalError(
                "ROLLBACK_TARGET_INVALID",
                "rollback target must be retired or validated",
            )
        if target.index.source_index_version_id != active_source_index_version_id:
            raise AdvancedRetrievalError(
                "SOURCE_INDEX_MISMATCH",
                "cannot rollback an index built from an inactive ordinary index",
            )
        gate = target.index.evaluation
        if gate is None or not gate.passed or not gate.rollback_ready:
            raise AdvancedRetrievalError(
                "ROLLBACK_NOT_READY",
                "rollback target has no approved rollback evidence",
            )
        for current in self.repository.list_snapshots(
            target.index.tenant_id,
            target.index.kb_id,
        ):
            if (
                current.index.index_type == target.index.index_type
                and current.index.status == "active"
            ):
                self.repository.save_snapshot(
                    _validated_copy(
                        current,
                        index=_validated_copy(
                            current.index,
                            status="retired",
                            published_at=None,
                        ),
                    )
                )
        active = _validated_copy(
            target,
            index=_validated_copy(
                target.index,
                status="active",
                published_at=self._now(),
                stale_reason=None,
            ),
        )
        self.repository.save_snapshot(active)
        return active


_WORD = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")


def _tokens(value: str) -> frozenset[str]:
    normalized = value.casefold()
    result: set[str] = set()
    for match in _WORD.findall(normalized):
        if re.fullmatch(r"[\u3400-\u9fff]+", match):
            result.update(match)
            result.update(match[index : index + 2] for index in range(len(match) - 1))
        else:
            result.add(match)
    return frozenset(result)


def _score(query_tokens: frozenset[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    target = _tokens(text)
    if not target:
        return 0.0
    overlap = len(query_tokens & target)
    if not overlap:
        return 0.0
    coverage = overlap / len(query_tokens)
    precision = overlap / len(target)
    return min(1.0, 0.8 * coverage + 0.2 * precision)


def _evidence_content_score(query_tokens: frozenset[str], evidence: Evidence) -> float:
    return _score(query_tokens, evidence.content[:100_000])


def _combined(routing: float, content: float) -> float:
    return min(1.0, round(0.6 * routing + 0.4 * content, 9))


def _route_type(route: AdvancedRoute) -> str:
    return {"toc": "toc", "raptor": "raptor", "graphrag": "graphrag"}[route]


class _RouteCandidates:
    def __init__(self, *, max_provenance_per_candidate: int) -> None:
        self.observations: dict[str, AdvancedRouteObservation] = {}
        self.evidence: dict[str, Evidence] = {}
        self.matched_nodes = 0
        self.budget_omitted = 0
        self.max_provenance_per_candidate = max_provenance_per_candidate

    def add(
        self,
        evidence: Evidence,
        observation: AdvancedRouteObservation,
    ) -> None:
        evidence_id = evidence.evidence_id
        assert evidence_id is not None
        previous = self.observations.get(evidence_id)
        if previous is None:
            self.observations[evidence_id] = observation
            self.evidence[evidence_id] = evidence
            return
        new_wins = (
            observation.combined_score,
            observation.routing_node_id,
        ) > (
            previous.combined_score,
            previous.routing_node_id,
        )
        winner = observation if new_wins else previous
        loser = previous if new_wins else observation
        support_by_id = {
            item.routing_node_id: item
            for item in (
                *winner.supporting_provenance,
                *loser.supporting_provenance,
            )
            if item.routing_node_id != winner.routing_node_id
        }
        if loser.routing_node_id != winner.routing_node_id:
            support_by_id[loser.routing_node_id] = AdvancedSupportingProvenance(
                provenance_kind=loser.provenance_kind,
                routing_node_id=loser.routing_node_id,
                routing_score=loser.routing_score,
                explanation=loser.explanation,
                generated_routing_text=loser.generated_routing_text,
            )
        support_limit = max(0, self.max_provenance_per_candidate - 1)
        ordered_support = sorted(
            support_by_id.values(),
            key=lambda item: (-item.routing_score, item.routing_node_id),
        )
        self.budget_omitted += max(0, len(ordered_support) - support_limit)
        supporting = tuple(ordered_support[:support_limit])
        self.observations[evidence_id] = _validated_copy(
            winner,
            supporting_provenance=supporting,
        )
        self.evidence[evidence_id] = evidence

    def trim(self, limit: int) -> None:
        ordered = sorted(
            self.observations,
            key=lambda evidence_id: (
                -self.observations[evidence_id].combined_score,
                evidence_id,
            ),
        )
        self.budget_omitted += max(0, len(ordered) - limit)
        keep = set(ordered[:limit])
        self.observations = {
            evidence_id: observation
            for evidence_id, observation in self.observations.items()
            if evidence_id in keep
        }
        self.evidence = {
            evidence_id: evidence
            for evidence_id, evidence in self.evidence.items()
            if evidence_id in keep
        }


def _search_toc(
    request: AdvancedRetrievalRequest,
    snapshot: StructureSnapshot,
    evidence_by_id: Mapping[str, Evidence],
) -> _RouteCandidates:
    result = _RouteCandidates(
        max_provenance_per_candidate=request.budget.max_provenance_per_candidate
    )
    query_tokens = _tokens(request.query)
    by_id = {node.node_id: node for node in snapshot.nodes}
    scored = []
    for node in snapshot.nodes:
        if node.node_type not in {"toc", "heading"}:
            continue
        routing_text = " > ".join(node.toc_path or ((node.title or ""),))
        score = _score(query_tokens, routing_text)
        if score:
            scored.append((score, node))
    scored.sort(key=lambda item: (-item[0], item[1].order, item[1].node_id))
    selected = scored[: request.budget.toc_node_limit]
    result.matched_nodes = len(selected)
    expanded = 0
    for routing_score, node in selected:
        queue = deque([node.node_id, *node.child_node_ids])
        if node.previous_node_id is not None:
            queue.append(node.previous_node_id)
        if node.next_node_id is not None:
            queue.append(node.next_node_id)
        visited: set[str] = set()
        while queue and expanded < request.budget.toc_expansion_limit:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            expanded += 1
            expanded_node = by_id[node_id]
            for evidence_id in expanded_node.source_evidence_ids:
                evidence = evidence_by_id[evidence_id]
                content_score = _evidence_content_score(query_tokens, evidence)
                combined = _combined(routing_score, content_score)
                result.add(
                    evidence,
                    AdvancedRouteObservation(
                        route="toc",
                        provenance_kind=("toc_node" if node.node_type == "toc" else "heading_node"),
                        routing_node_id=node.node_id,
                        routing_score=round(routing_score, 9),
                        content_score=round(content_score, 9),
                        combined_score=combined,
                        rank=1,
                        explanation=(
                            f"TOC path={' > '.join(node.toc_path or (node.title or '',))}; "
                            f"parent_score={routing_score:.4f}; "
                            f"content_score={content_score:.4f}; "
                            f"expanded_node={expanded_node.node_id}"
                        ),
                        generated_routing_text=False,
                    ),
                )
    if queue if selected else False:
        result.budget_omitted += len(queue)
    result.trim(request.budget.max_candidates_per_route)
    return result


def _search_raptor(
    request: AdvancedRetrievalRequest,
    snapshot: RaptorSnapshot,
    evidence_by_id: Mapping[str, Evidence],
) -> _RouteCandidates:
    result = _RouteCandidates(
        max_provenance_per_candidate=request.budget.max_provenance_per_candidate
    )
    query_tokens = _tokens(request.query)
    scored = [(_score(query_tokens, summary.text), summary) for summary in snapshot.summaries]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], -item[1].level, item[1].summary_id))
    selected = scored[: request.budget.raptor_summary_limit]
    result.matched_nodes = len(selected)
    leaf_count = 0
    for routing_score, summary in selected:
        for evidence_id in summary.source_evidence_ids:
            if leaf_count >= request.budget.raptor_leaf_limit:
                result.budget_omitted += 1
                continue
            leaf_count += 1
            evidence = evidence_by_id[evidence_id]
            content_score = _evidence_content_score(query_tokens, evidence)
            combined = _combined(routing_score, content_score)
            result.add(
                evidence,
                AdvancedRouteObservation(
                    route="raptor",
                    provenance_kind="raptor_summary",
                    routing_node_id=summary.summary_id,
                    routing_score=round(routing_score, 9),
                    content_score=round(content_score, 9),
                    combined_score=combined,
                    rank=1,
                    explanation=(
                        f"summary_level={summary.level}; "
                        f"summary_score={routing_score:.4f}; "
                        f"leaf_score={content_score:.4f}; "
                        "citation_target=leaf_evidence"
                    ),
                    generated_routing_text=True,
                ),
            )
    result.trim(request.budget.max_candidates_per_route)
    return result


def _search_graph(
    request: AdvancedRetrievalRequest,
    snapshot: GraphSnapshot,
    evidence_by_id: Mapping[str, Evidence],
) -> _RouteCandidates:
    result = _RouteCandidates(
        max_provenance_per_candidate=request.budget.max_provenance_per_candidate
    )
    query_tokens = _tokens(request.query)
    entity_scores: dict[str, float] = {}
    for entity in snapshot.entities:
        text = " ".join((entity.canonical_name, entity.entity_type, *entity.aliases))
        score = _score(query_tokens, text)
        if score:
            entity_scores[entity.entity_id] = score
    selected_entities = dict(
        sorted(
            entity_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[: request.budget.graph_entity_limit]
    )
    relation_scores: list[tuple[float, Any]] = []
    frontier = set(selected_entities)
    reached = set(frontier)
    for _ in range(request.budget.graph_hops):
        next_frontier: set[str] = set()
        for relation in snapshot.relations:
            if (
                relation.source_entity_id not in frontier
                and relation.target_entity_id not in frontier
            ):
                continue
            score = max(
                selected_entities.get(relation.source_entity_id, 0.0),
                selected_entities.get(relation.target_entity_id, 0.0),
                _score(
                    query_tokens,
                    f"{relation.relation_type} {relation.statement}",
                ),
            )
            if score:
                relation_scores.append((score, relation))
                next_frontier.update((relation.source_entity_id, relation.target_entity_id))
        frontier = next_frontier - reached
        reached.update(next_frontier)
        if not frontier:
            break
    relation_scores = sorted(
        {relation.relation_id: (score, relation) for score, relation in relation_scores}.values(),
        key=lambda item: (-item[0], item[1].relation_id),
    )[: request.budget.graph_edge_limit]
    community_scores = [
        (
            _score(query_tokens, f"{community.title} {community.summary}"),
            community,
        )
        for community in snapshot.communities
    ]
    community_scores = sorted(
        (item for item in community_scores if item[0] > 0),
        key=lambda item: (-item[0], item[1].community_id),
    )[: request.budget.graph_entity_limit]
    result.matched_nodes = len(selected_entities) + len(relation_scores) + len(community_scores)

    entities = {item.entity_id: item for item in snapshot.entities}
    sources: list[tuple[float, str, str, tuple[str, ...], bool, str]] = []
    for entity_id, routing_score in selected_entities.items():
        entity = entities[entity_id]
        sources.append(
            (
                routing_score,
                entity.entity_id,
                "graph_entity",
                entity.source_evidence_ids,
                False,
                f"entity={entity.canonical_name}",
            )
        )
    for routing_score, relation in relation_scores:
        confidence = (
            f"; inferred_confidence={relation.confidence:.4f}"
            if relation.inferred and relation.confidence is not None
            else ""
        )
        sources.append(
            (
                routing_score,
                relation.relation_id,
                "graph_relation",
                relation.source_evidence_ids,
                relation.inferred,
                (
                    f"relation={relation.source_entity_id}"
                    f"-[{relation.relation_type}]->{relation.target_entity_id}"
                    f"{confidence}"
                ),
            )
        )
    for routing_score, community in community_scores:
        sources.append(
            (
                routing_score,
                community.community_id,
                "graph_community",
                community.source_evidence_ids,
                True,
                f"community={community.title}; citation_target=leaf_evidence",
            )
        )
    for routing_score, node_id, kind, evidence_ids, generated, detail in sources:
        for evidence_id in evidence_ids:
            evidence = evidence_by_id[evidence_id]
            content_score = _evidence_content_score(query_tokens, evidence)
            combined = _combined(routing_score, content_score)
            result.add(
                evidence,
                AdvancedRouteObservation(
                    route="graphrag",
                    provenance_kind=kind,
                    routing_node_id=node_id,
                    routing_score=round(routing_score, 9),
                    content_score=round(content_score, 9),
                    combined_score=combined,
                    rank=1,
                    explanation=(
                        f"{detail}; graph_score={routing_score:.4f}; "
                        f"content_score={content_score:.4f}"
                    ),
                    generated_routing_text=generated,
                ),
            )
    result.trim(request.budget.max_candidates_per_route)
    return result


class AdvancedRetrievalService:
    """在三条独立 feature flag 后运行高级路线并安全降级。"""

    _ROUTES: tuple[AdvancedRoute, ...] = ("toc", "raptor", "graphrag")

    def search(
        self,
        request: AdvancedRetrievalRequest,
        *,
        snapshots: Sequence[AdvancedIndexSnapshot],
        evidence_by_id: Mapping[str, Evidence],
    ) -> AdvancedRetrievalResult:
        selected: dict[AdvancedRoute, AdvancedIndexSnapshot] = {}
        duplicate_routes: set[AdvancedRoute] = set()
        for snapshot in snapshots:
            if (
                snapshot.index.tenant_id != request.tenant_id
                or snapshot.index.kb_id != request.kb_id
            ):
                continue
            route = snapshot.index.index_type
            if route in selected:
                duplicate_routes.add(route)
            else:
                selected[route] = snapshot

        route_candidates: dict[AdvancedRoute, _RouteCandidates] = {}
        outcomes: list[AdvancedRouteOutcome] = []
        enabled_routes: list[AdvancedRoute] = []
        requested = set(request.requested_routes)
        for route in self._ROUTES:
            if route not in requested:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="ROUTE_NOT_REQUESTED",
                        message="query planner did not request this route",
                    )
                )
                continue
            if not request.flags.enabled(route):
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="FEATURE_DISABLED",
                        message="advanced route feature flag is disabled",
                    )
                )
                continue
            enabled_routes.append(route)
            snapshot = selected.get(route)
            if snapshot is None:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="SNAPSHOT_MISSING",
                        message="no advanced index snapshot is available",
                    )
                )
                continue
            if route in duplicate_routes:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="failed",
                        reason="INTEGRITY_ERROR",
                        message="multiple snapshots were supplied for one route",
                    )
                )
                continue
            if snapshot.index.status == "stale":
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="INDEX_STALE",
                        message=snapshot.index.stale_reason or "advanced index is stale",
                    )
                )
                continue
            if snapshot.index.status != "active":
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="INDEX_NOT_ACTIVE",
                        message="advanced index is not active",
                    )
                )
                continue
            if snapshot.index.source_index_version_id != request.active_source_index_version_id:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="skipped",
                        reason="SOURCE_INDEX_MISMATCH",
                        message="advanced index source is not the active ordinary index",
                    )
                )
                continue
            try:
                validate_advanced_snapshot(snapshot, evidence_by_id)
                if route == "toc":
                    assert snapshot.structure is not None
                    found = _search_toc(request, snapshot.structure, evidence_by_id)
                elif route == "raptor":
                    assert snapshot.raptor is not None
                    found = _search_raptor(request, snapshot.raptor, evidence_by_id)
                else:
                    assert snapshot.graph is not None
                    found = _search_graph(request, snapshot.graph, evidence_by_id)
            except AdvancedRetrievalError as exc:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="failed",
                        reason="INTEGRITY_ERROR",
                        omitted_missing_evidence=exc.missing_evidence_count,
                        message=exc.safe_message,
                    )
                )
                continue
            except Exception:
                # 高级路线永远不能拖垮普通检索。此处不把异常文本暴露给客户端，
                # 生产接入时由 trace/logger 记录完整异常。
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="failed",
                        reason="INTEGRITY_ERROR",
                        message="advanced route failed validation or execution",
                    )
                )
                continue
            route_candidates[route] = found
            if found.observations:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="succeeded",
                        reason="ROUTE_COMPLETED",
                        matched_node_count=found.matched_nodes,
                        candidate_count=len(found.observations),
                        budget_omitted_count=found.budget_omitted,
                        message=request.planner_reasons.get(
                            route,
                            "route explicitly enabled",
                        ),
                    )
                )
            else:
                outcomes.append(
                    AdvancedRouteOutcome(
                        route=route,
                        status="empty",
                        reason="NO_MATCH",
                        matched_node_count=found.matched_nodes,
                        budget_omitted_count=found.budget_omitted,
                        message="advanced route found no matching original Evidence",
                    )
                )

        by_evidence: dict[str, list[AdvancedRouteObservation]] = defaultdict(list)
        evidence_catalog: dict[str, Evidence] = {}
        for route in self._ROUTES:
            found = route_candidates.get(route)
            if found is None:
                continue
            ranked_ids = sorted(
                found.observations,
                key=lambda evidence_id: (
                    -found.observations[evidence_id].combined_score,
                    evidence_id,
                ),
            )
            for rank, evidence_id in enumerate(ranked_ids, start=1):
                observation = _validated_copy(
                    found.observations[evidence_id],
                    rank=rank,
                )
                by_evidence[evidence_id].append(observation)
                evidence_catalog[evidence_id] = found.evidence[evidence_id]

        candidates: list[AdvancedRetrievalCandidate] = []
        for evidence_id, observations in by_evidence.items():
            observations.sort(key=lambda item: self._ROUTES.index(item.route))
            # 多路一致命中使用 noisy-or；它有界、单调且不伪装成引擎原始分数。
            complement = 1.0
            for observation in observations:
                complement *= 1.0 - observation.combined_score
            final_score = min(1.0, round(1.0 - complement, 9))
            candidates.append(
                AdvancedRetrievalCandidate(
                    evidence=evidence_catalog[evidence_id],
                    observations=tuple(observations),
                    final_score=final_score,
                )
            )
        candidates.sort(
            key=lambda item: (
                -item.final_score,
                item.evidence.evidence_id or "",
            )
        )
        budget_omitted = max(
            0,
            len(candidates) - request.budget.max_total_candidates,
        )
        candidates = candidates[: request.budget.max_total_candidates]
        if budget_omitted:
            for index, outcome in enumerate(outcomes):
                if outcome.status == "succeeded":
                    outcomes[index] = _validated_copy(
                        outcome,
                        budget_omitted_count=(outcome.budget_omitted_count + budget_omitted),
                    )
                    break
        degraded = not candidates
        source_index = (
            next(iter(selected.values())).index.source_index_version_id
            if selected
            else request.active_source_index_version_id
        )
        return AdvancedRetrievalResult(
            trace_id=request.trace_id,
            tenant_id=request.tenant_id,
            kb_id=request.kb_id,
            candidates=tuple(candidates),
            diagnostics=AdvancedRetrievalDiagnostics(
                requested_routes=request.requested_routes,
                enabled_routes=tuple(enabled_routes),
                outcomes=tuple(outcomes),
                source_index_version_id=source_index,
                active_source_index_version_id=request.active_source_index_version_id,
                used_candidate_budget=len(candidates),
                max_candidate_budget=request.budget.max_total_candidates,
                degraded_to_baseline=degraded,
            ),
        )


__all__ = [
    "AdvancedIndexLifecycleService",
    "AdvancedIndexRepository",
    "AdvancedRetrievalError",
    "AdvancedRetrievalService",
    "validate_advanced_snapshot",
]
