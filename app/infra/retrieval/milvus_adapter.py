"""Milvus 检索端口适配器。

适配器通过构造参数接收 embedding、Milvus 门面、活动索引解析器和元数据目录
解析器。模块自身不导入 pymilvus；这既让查询核心保持后端无关，也让契约测试
无需启动 Docker 即可覆盖与生产适配器相同的边界行为。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import json
import math
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.application.retrieval_ports import RetrievalPort
from app.domain.retrieval import (
    FilterNode,
    RetrievalEngineError,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
    RetrievalRoute,
    RetrievalScope,
    RouteOutcome,
    RouteScore,
)


DEFAULT_OUTPUT_FIELDS = [
    "chunk_id",
    "id",
    "source_chunk_id",
    "evidence_id",
    "content_hash",
    "content_with_weight",
    "content",
    "title",
    "parent_title",
    "docnm_kwd",
    "kb_id",
    "tenant_id",
    "doc_id",
    "document_version_id",
    "index_version_id",
    "meta_fields",
]


class ActiveIndexScope(BaseModel):
    """由控制库解析出的当前可搜索发布事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    document_version_id: str = Field(min_length=1, max_length=64)
    index_version_id: str = Field(min_length=1, max_length=64)
    collection_name: str = Field(min_length=1, max_length=255)
    partition_name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator(
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "collection_name",
    )
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("active index scope identifiers cannot be blank")
        return normalized

    @field_validator("partition_name")
    @classmethod
    def normalize_partition(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("active index partition cannot be blank")
        return normalized


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> Any: ...


class MilvusGatewayPort(Protocol):
    def create_requests(
        self,
        dense_vector: list[float],
        sparse_vector: Any,
        *,
        expr: str | None = None,
        limit: int = 5,
    ) -> Sequence[Any]: ...

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[Any],
        ranker_weights: tuple[float, float] = (0.5, 0.5),
        norm_score: bool = False,
        limit: int = 5,
        output_fields: list[str] | None = None,
        search_params: dict | None = None,
        partition_names: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any: ...


class ActiveScopeResolver(Protocol):
    def resolve(self, scope: RetrievalScope) -> Sequence[ActiveIndexScope]: ...


class MetadataFilterResolver(Protocol):
    """在 SQL 元数据目录中计算满足 AST 的文档 ID，不生成 Milvus 表达式。"""

    def resolve_document_ids(
        self,
        scope: RetrievalScope,
        filters: FilterNode,
    ) -> Sequence[str]: ...


class MilvusHitMapper(Protocol):
    def to_hit(
        self,
        raw_hit: Any,
        *,
        route: str,
        rank: int,
    ) -> RetrievalHit: ...


def _raw_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                value = getter(key)
                return default if value is None else value
            except Exception:
                pass
    return getattr(container, key, default)


class DefaultMilvusHitMapper:
    """把 pymilvus 命中转成后端无关模型；该类本身不知道 pymilvus 类型。"""

    def to_hit(
        self,
        raw_hit: Any,
        *,
        route: str,
        rank: int,
    ) -> RetrievalHit:
        entity = _raw_value(raw_hit, "entity", raw_hit)
        chunk_id = (
            _raw_value(entity, "source_chunk_id")
            or _raw_value(entity, "chunk_id")
            or _raw_value(entity, "id")
            or _raw_value(raw_hit, "id")
        )
        content = _raw_value(entity, "content_with_weight") or _raw_value(entity, "content")
        raw_score = _raw_value(raw_hit, "distance", _raw_value(raw_hit, "score"))
        if raw_score is None:
            raise ValueError("Milvus hit is missing distance")

        known_fields = {
            "source_chunk_id",
            "chunk_id",
            "id",
            "evidence_id",
            "content_hash",
            "content_with_weight",
            "content",
            "title",
            "parent_title",
            "docnm_kwd",
            "kb_id",
            "tenant_id",
            "doc_id",
            "document_version_id",
            "index_version_id",
            "meta_fields",
        }
        metadata: dict[str, Any] = {}
        if isinstance(entity, Mapping):
            metadata = {str(key): value for key, value in entity.items() if key not in known_fields}
        meta_fields = _raw_value(entity, "meta_fields")
        if isinstance(meta_fields, Mapping):
            metadata.update(meta_fields)
        elif isinstance(meta_fields, str) and meta_fields.strip():
            try:
                decoded = json.loads(meta_fields)
                if isinstance(decoded, Mapping):
                    metadata.update(decoded)
            except (TypeError, ValueError):
                metadata["meta_fields_raw"] = meta_fields

        return RetrievalHit(
            chunk_id=str(chunk_id or ""),
            evidence_id=_raw_value(entity, "evidence_id"),
            tenant_id=str(_raw_value(entity, "tenant_id") or ""),
            kb_id=str(_raw_value(entity, "kb_id") or ""),
            document_id=str(_raw_value(entity, "doc_id") or ""),
            document_version_id=str(_raw_value(entity, "document_version_id") or ""),
            index_version_id=str(_raw_value(entity, "index_version_id") or ""),
            content=str(content or ""),
            content_hash=_raw_value(entity, "content_hash"),
            title=(
                _raw_value(entity, "title")
                or _raw_value(entity, "parent_title")
                or _raw_value(entity, "docnm_kwd")
            ),
            metadata=metadata,
            scores=(
                RouteScore(
                    route=route,
                    raw_score=float(raw_score),
                    rank=rank,
                ),
            ),
        )


def _quoted(value: str) -> str:
    """Milvus 字符串字面量采用 JSON 转义，不能直接拼接未转义值。"""

    return json.dumps(value, ensure_ascii=False)


def build_mandatory_scope_expr(scopes: Sequence[ActiveIndexScope]) -> str:
    """只根据服务端活动发布事实构造过滤表达式。"""

    if not scopes:
        raise ValueError("at least one active index scope is required")
    clauses = []
    for scope in scopes:
        clauses.append(
            "("
            f"tenant_id == {_quoted(scope.tenant_id)} and "
            f"kb_id == {_quoted(scope.kb_id)} and "
            f"doc_id == {_quoted(scope.document_id)} and "
            f"document_version_id == {_quoted(scope.document_version_id)} and "
            f"index_version_id == {_quoted(scope.index_version_id)} and "
            "available_int == 1"
            ")"
        )
    return "(" + " or ".join(clauses) + ")"


def _hit_distance(raw_hit: Any) -> float:
    value = _raw_value(raw_hit, "distance", _raw_value(raw_hit, "score"))
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Milvus hit distance is not numeric") from exc
    if not math.isfinite(score):
        raise ValueError("Milvus hit distance is not finite")
    return score


def _is_timeout_exception(exc: BaseException) -> bool:
    """识别 Python、gRPC/pymilvus 常见的有界调用超时，不依赖 SDK 类型。"""

    if isinstance(exc, TimeoutError):
        return True
    marker = f"{type(exc).__name__} {exc}".casefold()
    return any(
        token in marker
        for token in ("deadline exceeded", "deadline_exceeded", "timed out", "timeout")
    )


class MilvusRetrievalAdapter(RetrievalPort):
    """使用 BGE-M3 向量和 Milvus 实现可配置多路召回。"""

    engine_name = "milvus"

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        gateway: MilvusGatewayPort,
        scope_resolver: ActiveScopeResolver,
        filter_resolver: MetadataFilterResolver | None = None,
        hit_mapper: MilvusHitMapper | None = None,
        cancellation: Callable[[], bool] | None = None,
        clock=time.monotonic,
    ) -> None:
        self._embedding = embedding
        self._gateway = gateway
        self._scope_resolver = scope_resolver
        self._filter_resolver = filter_resolver
        self._hit_mapper = hit_mapper or DefaultMilvusHitMapper()
        self._cancellation = cancellation or (lambda: False)
        self._clock = clock

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        started = self._clock()
        self._check_deadline(request, started)
        scopes = self._resolve_scopes(request)
        scopes = self._apply_document_filters(request, scopes)
        if not scopes:
            return RetrievalResult(
                trace_id=request.trace_id,
                profile_id=request.profile.profile_id,
                profile_version=request.profile.version,
                engine=self.engine_name,
                route_outcomes=tuple(
                    RouteOutcome(route=route, status="empty") for route in request.profile.routes
                ),
                elapsed_ms=self._elapsed_ms(started),
            )

        grouped_scopes: dict[tuple[str, str | None], list[ActiveIndexScope]] = {}
        for scope in scopes:
            grouped_scopes.setdefault(
                (scope.collection_name, scope.partition_name),
                [],
            ).append(scope)

        route_hits: list[RetrievalHit] = []
        outcomes: list[RouteOutcome] = []
        for route in request.profile.routes:
            route_started = self._clock()
            hits = self._search_route(
                request,
                route,
                grouped_scopes,
                started=started,
            )
            route_hits.extend(hits)
            outcomes.append(
                RouteOutcome(
                    route=route,
                    status="succeeded" if hits else "empty",
                    candidate_count=len(hits),
                    latency_ms=self._elapsed_ms(route_started),
                )
            )

        hits = self._fuse_hits(request, route_hits)
        return RetrievalResult(
            trace_id=request.trace_id,
            profile_id=request.profile.profile_id,
            profile_version=request.profile.version,
            engine=self.engine_name,
            hits=hits,
            route_outcomes=tuple(outcomes),
            elapsed_ms=self._elapsed_ms(started),
        )

    def _resolve_scopes(self, request: RetrievalRequest) -> tuple[ActiveIndexScope, ...]:
        try:
            raw_scopes = self._scope_resolver.resolve(request.scope)
            scopes = tuple(
                item
                if isinstance(item, ActiveIndexScope)
                else ActiveIndexScope.model_validate(item)
                for item in raw_scopes
            )
        except RetrievalEngineError:
            raise
        except Exception as exc:
            raise RetrievalEngineError(
                "RETRIEVAL_UNAVAILABLE",
                "活动索引作用域解析失败",
                retryable=True,
            ) from exc

        if not scopes:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "知识库没有可搜索的活动索引",
                retryable=False,
            )
        allowed_kbs = set(request.scope.kb_ids)
        for scope in scopes:
            if scope.tenant_id != request.scope.tenant_id or scope.kb_id not in allowed_kbs:
                raise RetrievalEngineError(
                    "RETRIEVAL_STALE_SCOPE",
                    "活动索引作用域与认证上下文不一致",
                    retryable=False,
                    details={"kb_id": scope.kb_id},
                )
        return scopes

    def _apply_document_filters(
        self,
        request: RetrievalRequest,
        scopes: tuple[ActiveIndexScope, ...],
    ) -> tuple[ActiveIndexScope, ...]:
        selected_ids: set[str] | None = set(request.document_ids) if request.document_ids else None
        if request.filters is not None:
            if self._filter_resolver is None:
                raise RetrievalEngineError(
                    "RETRIEVAL_FILTER_UNSUPPORTED",
                    "当前检索适配器没有配置元数据目录解析器",
                    retryable=False,
                )
            try:
                resolved_ids = {
                    str(item).strip()
                    for item in self._filter_resolver.resolve_document_ids(
                        request.scope,
                        request.filters,
                    )
                    if str(item).strip()
                }
            except RetrievalEngineError:
                raise
            except Exception as exc:
                raise RetrievalEngineError(
                    "RETRIEVAL_FILTER_UNSUPPORTED",
                    "元数据过滤条件无法解析",
                    retryable=False,
                ) from exc
            selected_ids = (
                resolved_ids if selected_ids is None else selected_ids.intersection(resolved_ids)
            )

        if selected_ids is None:
            return scopes
        return tuple(scope for scope in scopes if scope.document_id in selected_ids)

    def _search_route(
        self,
        request: RetrievalRequest,
        route: RetrievalRoute,
        grouped_scopes: Mapping[
            tuple[str, str | None],
            Sequence[ActiveIndexScope],
        ],
        *,
        started: float,
    ) -> tuple[RetrievalHit, ...]:
        texts = (
            tuple(dict.fromkeys((request.query, *request.variants)))
            if route == "question"
            else (request.query,)
        )
        all_hits: list[tuple[Any, str]] = []
        for variant_index, text in enumerate(texts):
            self._check_deadline(request, started)
            dense_vector, sparse_vector = self._embed(request, text, route)
            # embedding 是同步且不可抢占的第三方调用。它返回时必须重新检查
            # QueryRun/route 预算，不能让已经超时的隔离线程继续请求 Milvus。
            self._check_deadline(request, started, route=route)
            score_route = f"question.{variant_index + 1}" if route == "question" else route
            for (
                collection_name,
                partition_name,
            ), collection_scopes in grouped_scopes.items():
                try:
                    reqs = list(
                        self._gateway.create_requests(
                            dense_vector=dense_vector,
                            sparse_vector=sparse_vector,
                            expr=build_mandatory_scope_expr(collection_scopes),
                            limit=request.profile.candidate_k,
                        )
                    )
                    if len(reqs) != 2:
                        raise ValueError("Milvus gateway must return dense and sparse requests")
                    response = self._gateway.hybrid_search(
                        collection_name=collection_name,
                        reqs=reqs,
                        ranker_weights=self._weights_for(request, route),
                        norm_score=request.profile.hybrid.normalize_scores,
                        limit=request.profile.candidate_k,
                        output_fields=DEFAULT_OUTPUT_FIELDS,
                        partition_names=([partition_name] if partition_name is not None else None),
                        timeout_seconds=self._remaining_timeout_seconds(request, started),
                    )
                except RetrievalEngineError:
                    raise
                except Exception as exc:
                    if _is_timeout_exception(exc):
                        raise RetrievalEngineError(
                            "RETRIEVAL_TIMEOUT",
                            "Milvus 检索超过剩余查询预算",
                            retryable=True,
                            route=route,
                        ) from exc
                    raise RetrievalEngineError(
                        "RETRIEVAL_UNAVAILABLE",
                        "Milvus 检索调用失败",
                        retryable=True,
                        route=route,
                    ) from exc
                if response is None:
                    raise RetrievalEngineError(
                        "RETRIEVAL_UNAVAILABLE",
                        "Milvus 检索未返回结果",
                        retryable=True,
                        route=route,
                    )
                # pymilvus 的 timeout 约束传输/服务端等待；此处再做一次本地绝对
                # deadline 与协作取消检查，拒绝任何迟到结果进入候选集。
                self._check_deadline(request, started, route=route)
                try:
                    batch = response[0]
                except (IndexError, KeyError, TypeError) as exc:
                    raise RetrievalEngineError(
                        "RETRIEVAL_INVALID_RESPONSE",
                        "Milvus 返回结构不符合检索契约",
                        retryable=False,
                        route=route,
                    ) from exc
                all_hits.extend((raw_hit, score_route) for raw_hit in batch)

        try:
            all_hits.sort(key=lambda item: _hit_distance(item[0]), reverse=True)
        except ValueError as exc:
            raise RetrievalEngineError(
                "RETRIEVAL_INVALID_RESPONSE",
                "Milvus 返回了非法分数",
                retryable=False,
                route=route,
            ) from exc

        mapped: list[RetrievalHit] = []
        route_seen: set[tuple[str, str]] = set()
        for raw_hit, score_route in all_hits:
            raw_score = _hit_distance(raw_hit)
            if (
                request.profile.score_threshold is not None
                and raw_score < request.profile.score_threshold
            ):
                continue
            try:
                hit = self._hit_mapper.to_hit(
                    raw_hit,
                    route=score_route,
                    rank=len(mapped) + 1,
                )
            except Exception as exc:
                raise RetrievalEngineError(
                    "RETRIEVAL_INVALID_RESPONSE",
                    "Milvus 命中无法转换为标准检索块",
                    retryable=False,
                    route=route,
                ) from exc
            identity = (
                hit.tenant_id,
                hit.kb_id,
                hit.document_id,
                hit.document_version_id,
                hit.index_version_id,
            )
            active_identities = {
                (
                    scope.tenant_id,
                    scope.kb_id,
                    scope.document_id,
                    scope.document_version_id,
                    scope.index_version_id,
                )
                for scopes in grouped_scopes.values()
                for scope in scopes
            }
            if identity not in active_identities:
                raise RetrievalEngineError(
                    "RETRIEVAL_STALE_SCOPE",
                    "Milvus 命中不属于当前活动索引",
                    retryable=False,
                    route=route,
                    details={"chunk_id": hit.chunk_id},
                )
            duplicate_key = (score_route, hit.chunk_id)
            if duplicate_key in route_seen:
                continue
            route_seen.add(duplicate_key)
            mapped.append(hit)
            if len(mapped) >= request.profile.candidate_k * len(texts):
                break
        self._check_deadline(request, started)
        return tuple(mapped)

    def _embed(
        self,
        request: RetrievalRequest,
        text: str,
        route: RetrievalRoute,
    ) -> tuple[list[float], Any]:
        try:
            embedded = self._embedding.embed_documents([text])
            dense_batch = embedded["dense"]
            sparse_batch = embedded["sparse"]
            dense = [float(value) for value in dense_batch[0]]
            sparse = sparse_batch[0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RetrievalEngineError(
                "RETRIEVAL_INVALID_RESPONSE",
                "Embedding 返回结构不符合 dense/sparse 契约",
                retryable=False,
                route=route,
            ) from exc
        except Exception as exc:
            raise RetrievalEngineError(
                "RETRIEVAL_UNAVAILABLE",
                "Embedding 服务不可用",
                retryable=True,
                route=route,
            ) from exc

        if len(dense) != request.profile.dense.dimension:
            raise RetrievalEngineError(
                "RETRIEVAL_DIMENSION_MISMATCH",
                "查询向量维度与发布的检索配置不一致",
                retryable=False,
                route=route,
                details={
                    "expected": request.profile.dense.dimension,
                    "actual": len(dense),
                },
            )
        if not all(math.isfinite(value) for value in dense):
            raise RetrievalEngineError(
                "RETRIEVAL_INVALID_RESPONSE",
                "Embedding 返回了非有限 dense 数值",
                retryable=False,
                route=route,
            )
        return dense, sparse

    @staticmethod
    def _weights_for(
        request: RetrievalRequest,
        route: RetrievalRoute,
    ) -> tuple[float, float]:
        if route == "dense":
            return (1.0, 0.0)
        if route == "sparse":
            return (0.0, 1.0)
        return (request.profile.dense.weight, request.profile.sparse.weight)

    @staticmethod
    def _fuse_hits(
        request: RetrievalRequest,
        route_hits: Sequence[RetrievalHit],
    ) -> tuple[RetrievalHit, ...]:
        by_identity: dict[str, RetrievalHit] = {}
        scores_by_identity: dict[str, list[RouteScore]] = {}
        for hit in route_hits:
            identity = (
                hit.content_hash
                if request.profile.deduplicate_by == "content_hash" and hit.content_hash
                else hit.chunk_id
            )
            by_identity.setdefault(identity, hit)
            known_routes = {score.route for score in scores_by_identity.get(identity, [])}
            scores_by_identity.setdefault(identity, []).extend(
                score for score in hit.scores if score.route not in known_routes
            )

        fused: list[RetrievalHit] = []
        for identity, hit in by_identity.items():
            observations: list[RouteScore] = []
            final_score = 0.0
            raw_scores: list[float] = []
            for score in scores_by_identity[identity]:
                weighted_route = "question" if score.route.startswith("question.") else score.route
                contribution = (
                    request.profile.rrf.weight_for(weighted_route)
                    / (request.profile.rrf.k + score.rank)
                    if request.profile.rrf.enabled
                    else None
                )
                if contribution is not None:
                    final_score += contribution
                if score.raw_score is not None:
                    raw_scores.append(score.raw_score)
                observations.append(score.model_copy(update={"rrf_score": contribution}))
            if not request.profile.rrf.enabled:
                final_score = max(raw_scores, default=0.0)
            fused.append(
                hit.model_copy(
                    update={
                        "scores": tuple(observations),
                        "final_score": final_score,
                    },
                    deep=True,
                )
            )

        fused.sort(key=lambda item: (-(item.final_score or 0.0), item.chunk_id))
        # RetrievalPort 只负责候选召回；最终 top_k 必须由外层 RRF/reranker 应用，
        # 否则 candidate_k > top_k 的配置会在重排前被意外截断。
        return tuple(fused[: request.profile.candidate_k])

    def _check_deadline(
        self,
        request: RetrievalRequest,
        started: float,
        *,
        route: RetrievalRoute | None = None,
    ) -> None:
        if self._cancellation():
            raise RetrievalEngineError(
                "RETRIEVAL_CANCELLED",
                "检索已取消",
                retryable=False,
                route=route,
            )
        expired = self._elapsed_ms(started) > request.profile.timeout_ms
        if request.deadline_at is not None:
            now = datetime.now(tz=request.deadline_at.tzinfo)
            expired = expired or now >= request.deadline_at
        if expired:
            raise RetrievalEngineError(
                "RETRIEVAL_TIMEOUT",
                "检索超过发布配置允许的时间",
                retryable=True,
                route=route,
            )

    def _remaining_timeout_seconds(
        self,
        request: RetrievalRequest,
        started: float,
    ) -> float:
        """返回可交给同步 Milvus SDK 的剩余有界预算。"""

        self._check_deadline(request, started)
        remaining_ms = request.profile.timeout_ms - self._elapsed_ms(started)
        if request.deadline_at is not None:
            now = datetime.now(tz=request.deadline_at.tzinfo)
            absolute_remaining_ms = int(
                (request.deadline_at - now).total_seconds() * 1000
            )
            remaining_ms = min(remaining_ms, absolute_remaining_ms)
        if remaining_ms <= 0:
            self._check_deadline(request, started)
            raise RetrievalEngineError(
                "RETRIEVAL_TIMEOUT",
                "检索超过发布配置允许的时间",
                retryable=True,
            )
        # pymilvus 接受秒数；保留毫秒精度并确保正值。
        return max(0.001, remaining_ms / 1000.0)

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))
