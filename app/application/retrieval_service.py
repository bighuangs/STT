"""TODO-09 查询编排与检索端口的组合根。

查询图只调用本模块，不再导入 Milvus 门面。这里负责把会话固定的配置快照和认证
主体变成领域请求、调用生产适配器、恢复 TODO-08 Evidence，并输出可安全返回给
前端的路线诊断。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from threading import Lock
import time
from uuid import uuid4

from pydantic import TypeAdapter

from app.application.query_understanding_service import (
    extract_chapter_title_references,
)
from app.application.query_security_pipeline import consume_authorized_scope
from app.domain.evidence import RetrievalScore
from app.domain.retrieval import (
    FilterNode,
    RerankRequest,
    RetrievalEngineError,
    RetrievalHit,
    RetrievalProfile,
    RetrievalRequest,
    RetrievalScope,
    RouteScore,
    default_retrieval_profile,
)
from app.infra.llm.providers import llm_provider
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.repositories import ControlRepository
from app.infra.retrieval.control_plane_resolvers import (
    FrozenActiveScopeResolver,
    SqlActiveScopeResolver,
    SqlMetadataFilterResolver,
)
from app.infra.retrieval.milvus_adapter import MilvusRetrievalAdapter
from app.infra.retrieval.reranker_adapter import BGERerankerAdapter
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.query.evidence_service import hydrate_retrieved_hits
from app.security.principal import current_principal
from app.shared.config.reranker_config import reranker_config


_FILTER_ADAPTER = TypeAdapter(FilterNode)

# FlagEmbedding 会在首次调用时打开并反序列化整份 safetensors 权重。Windows
# 页面文件不足等进程级资源故障若在批量评测的每个 case 都立即重试，不但没有
# 恢复机会，还会持续放大内存压力。这里仅缓存“加载失败”一小段时间；冷却结束
# 后自动允许重试，成功加载后也立即清空失败状态，不会永久掩盖可恢复故障。
RERANKER_LOAD_FAILURE_COOLDOWN_SECONDS = 60.0
_reranker_load_lock = Lock()
_reranker_load_failure_at: float | None = None
_reranker_load_failure_type: str | None = None
_reranker_clock = time.monotonic


class _RerankerLoadCooldownError(RuntimeError):
    """同一进程内的短时加载熔断；保留最初异常类型供安全诊断。"""

    def __init__(self, original_error_type: str, remaining_seconds: float) -> None:
        super().__init__("reranker model load is temporarily in cooldown")
        self.original_error_type = original_error_type
        self.remaining_seconds = max(0.0, remaining_seconds)


def _load_reranker_model():
    """串行化惰性加载，并对最近一次失败施加可恢复的进程内冷却。"""

    global _reranker_load_failure_at, _reranker_load_failure_type
    with _reranker_load_lock:
        now = _reranker_clock()
        if _reranker_load_failure_at is not None:
            elapsed = now - _reranker_load_failure_at
            if elapsed < RERANKER_LOAD_FAILURE_COOLDOWN_SECONDS:
                raise _RerankerLoadCooldownError(
                    _reranker_load_failure_type or "Exception",
                    RERANKER_LOAD_FAILURE_COOLDOWN_SECONDS - elapsed,
                )
        try:
            model = llm_provider.reranker_model()
        except Exception as exc:
            _reranker_load_failure_at = _reranker_clock()
            _reranker_load_failure_type = type(exc).__name__
            raise
        _reranker_load_failure_at = None
        _reranker_load_failure_type = None
        return model


def _reranker_load_error_details(exc: Exception) -> dict[str, object]:
    cached = isinstance(exc, _RerankerLoadCooldownError)
    error_type = exc.original_error_type if cached else type(exc).__name__
    details: dict[str, object] = {
        "phase": "load",
        "error_type": error_type,
        "failure_cached": cached,
    }
    if cached:
        details["cooldown_remaining_ms"] = int(exc.remaining_seconds * 1000)
    return details


def profile_from_state(state: dict) -> RetrievalProfile:
    """把数据库配置快照与不可变身份合成领域对象。"""

    principal = current_principal()
    tenant_id = principal.tenant_id if principal else ""
    config = dict(state.get("retrieval_profile") or {})
    profile_id = str(state.get("retrieval_profile_id") or "")
    version = int(state.get("retrieval_profile_version") or 0)
    if not config:
        if not tenant_id:
            raise RetrievalEngineError(
                "RETRIEVAL_STALE_SCOPE",
                "缺少认证主体与会话检索配置",
                retryable=False,
            )
        return default_retrieval_profile(
            tenant_id,
            profile_id=profile_id or "rpf_default",
            version=version or 1,
        )
    config.update(
        {
            "profile_id": profile_id or str(config.get("profile_id") or "rpf_default"),
            "version": version or int(config.get("version") or 1),
            "tenant_id": tenant_id or str(config.get("tenant_id") or ""),
            # 在线 state 只会由 resolve_for_session 注入已发布版本，数据库状态是事实源。
            "status": "published",
        }
    )
    return RetrievalProfile.model_validate(config)


def _parse_filter(raw: dict | None) -> FilterNode | None:
    if not raw:
        return None
    return _FILTER_ADAPTER.validate_python(raw)


def _adapter(
    state: dict | None = None,
    *,
    cancellation: Callable[[], bool] | None = None,
) -> MilvusRetrievalAdapter:
    snapshot = dict(state or {})
    frozen_rows = snapshot.get("vector_read_scopes")
    frozen_digest = snapshot.get("vector_read_set_digest")
    if frozen_rows is None and frozen_digest is None:
        scope_resolver = SqlActiveScopeResolver()
    elif not isinstance(frozen_rows, list) or not isinstance(frozen_digest, str):
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "冻结的 vector read set 不完整",
            retryable=False,
        )
    else:
        scope_resolver = FrozenActiveScopeResolver(
            frozen_rows,
            expected_digest=frozen_digest,
        )
    return MilvusRetrievalAdapter(
        embedding=llm_provider,
        gateway=milvus_gateway,
        scope_resolver=scope_resolver,
        filter_resolver=SqlMetadataFilterResolver(),
        cancellation=cancellation,
    )


def _deadline_from_state(state: dict) -> datetime | None:
    """恢复 QueryRun 冻结的绝对截止时间，供同步适配器二次校验。"""

    raw = state.get("deadline_at")
    if raw is None or raw == "":
        return None
    try:
        value = raw if isinstance(raw, datetime) else datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise RetrievalEngineError(
            "RETRIEVAL_INVALID_DEADLINE",
            "查询截止时间格式无效",
            retryable=False,
        ) from exc
    if value.tzinfo is None:
        raise RetrievalEngineError(
            "RETRIEVAL_INVALID_DEADLINE",
            "查询截止时间必须包含时区",
            retryable=False,
        )
    return value.astimezone(UTC)


def _diagnostics(result, *, variant: str) -> dict:
    return {
        "trace_id": result.trace_id,
        "engine": result.engine,
        "query_variant": variant,
        "profile_id": result.profile_id,
        "profile_version": result.profile_version,
        "elapsed_ms": result.elapsed_ms,
        "routes": [
            item.model_dump(mode="json", exclude_none=True) for item in result.route_outcomes
        ],
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "final_score": hit.final_score,
                "scores": [
                    score.model_dump(mode="json", exclude_none=True) for score in hit.scores
                ],
            }
            for hit in result.hits
        ],
    }


def retrieve_exact_chapter_heading_candidates(
    state: dict,
    *,
    query_text: str,
    sub_query_id: str | None = None,
) -> tuple[list[dict], dict]:
    """在冻结读集内为单一章节标题问题补充精确 Evidence 候选。

    该路线只处理 extractor 判定为高置信、且仅含一个章节路径的问题。当前 Evidence
    没有可靠的 heading-owner 标记，因此只有唯一 Evidence row 才可返回；多 row、
    扫描上限或冻结范围缺失都会回退到常规检索。
    """

    chapter_paths = extract_chapter_title_references(query_text)
    base_diagnostics: dict[str, object] = {
        "route": "base.heading_exact",
        "requested_paths": list(chapter_paths),
        "candidate_count": 0,
    }
    if not chapter_paths:
        return [], {**base_diagnostics, "status": "not_applicable"}
    if len(chapter_paths) != 1:
        return [], {
            **base_diagnostics,
            "status": "fallback",
            "reason": "multiple_chapter_references",
        }

    frozen_rows = state.get("vector_read_scopes")
    frozen_digest = state.get("vector_read_set_digest")
    if frozen_rows is None and frozen_digest is None:
        return [], {
            **base_diagnostics,
            "status": "fallback",
            "reason": "frozen_scope_unavailable",
        }
    if not isinstance(frozen_rows, list) or not isinstance(frozen_digest, str):
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "冻结的 vector read set 不完整",
            retryable=False,
        )

    authorized_scope = consume_authorized_scope(state, "heading_exact")
    principal = current_principal()
    if principal is None:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "章节标题检索必须在认证主体上下文中执行",
            retryable=False,
        )
    kb_id = str(state.get("kb_id") or "").strip()
    if not kb_id:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "章节标题检索缺少知识库范围",
            retryable=False,
        )

    logical_scope = RetrievalScope(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        kb_ids=(kb_id,),
    )
    frozen_scopes = FrozenActiveScopeResolver(
        frozen_rows,
        expected_digest=frozen_digest,
    ).resolve(logical_scope)
    scope_identities = tuple(
        (
            item.tenant_id,
            item.kb_id,
            item.document_id,
            item.document_version_id,
            item.index_version_id,
        )
        for item in frozen_scopes
    )
    if authorized_scope is not None and any(
        not authorized_scope.permits(
            tenant_id=tenant_id,
            kb_id=scope_kb_id,
            document_id=document_id,
            document_version_id=document_version_id,
            index_version_id=index_version_id,
        )
        for (
            tenant_id,
            scope_kb_id,
            document_id,
            document_version_id,
            index_version_id,
        ) in scope_identities
    ):
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "冻结的章节标题读集超出授权范围",
            retryable=False,
        )

    document_ids = tuple(
        dict.fromkeys(str(value).strip() for value in (state.get("doc_ids") or []) if value)
    )
    with session_scope() as session:
        repository = ControlRepository(session, tenant_id=principal.tenant_id)
        rows, scanned_count, scan_truncated = (
            repository.find_exact_terminal_heading_evidence(
                chapter_paths=chapter_paths,
                scope_identities=scope_identities,
                document_ids=document_ids,
            )
        )
        if scan_truncated:
            return [], {
                **base_diagnostics,
                "status": "fallback",
                "reason": "scan_limit_exceeded",
                "scanned_count": scanned_count,
            }
        if not rows:
            return [], {
                **base_diagnostics,
                "status": "not_found",
                "scanned_count": scanned_count,
                "matched_row_count": 0,
                "matched_document_count": 0,
            }
        matched_documents = {str(row.document_id or "") for row in rows}
        if len(rows) != 1:
            return [], {
                **base_diagnostics,
                "status": "fallback",
                "reason": (
                    "ambiguous_documents"
                    if len(matched_documents) != 1
                    else "ambiguous_heading_evidence"
                ),
                "scanned_count": scanned_count,
                "matched_row_count": len(rows),
                "matched_document_count": len(matched_documents),
            }

        evidence = repository.evidence_dict(rows[0])

    retrieval = RetrievalScore(
        route="base.heading_exact",
        raw_score=1.0,
        rank=1,
        metadata={"match_kind": "exact_terminal_heading"},
    ).model_dump(mode="json", exclude_none=True)
    observation = {**retrieval, "sub_query_id": sub_query_id}
    evidence = {**evidence, "retrieval": retrieval}
    source = evidence["source"]
    candidate = {
        "chunk_id": source["chunk_id"],
        "source_chunk_id": source["chunk_id"],
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "content": evidence["content"],
        "text": evidence["content"],
        "title": evidence.get("source_label"),
        "file_title": evidence.get("source_label"),
        "tenant_id": source["tenant_id"],
        "kb_id": source["kb_id"],
        "document_id": source["document_id"],
        "document_version_id": source["document_version_id"],
        "index_version_id": source["index_version_id"],
        "retrieval_score": 1.0,
        "score": 1.0,
        "type": "milvus",
        "url": None,
        "evidence": evidence,
        "retrieval_route": "base.heading_exact",
        "retrieval_observations": [observation],
        "sub_query_ids": [sub_query_id] if sub_query_id else [],
    }
    return [candidate], {
        **base_diagnostics,
        "status": "succeeded",
        "candidate_count": 1,
        "scanned_count": scanned_count,
        "matched_row_count": len(rows),
        "matched_document_count": 1,
    }


def retrieve_candidates(
    state: dict,
    *,
    query_text: str,
    query_variant: str,
    sub_query_id: str | None = None,
    cancellation: Callable[[], bool] | None = None,
    deadline_at: datetime | None = None,
) -> tuple[list[dict], dict]:
    """执行一次 query variant 的候选检索并恢复受控 Evidence。"""

    authorized_scope = consume_authorized_scope(
        state,
        "hyde" if query_variant == "hyde" else "dense",
    )
    if query_variant != "hyde":
        # MilvusRetrievalAdapter 的本地路线是 dense+sparse 同一个强制表达式。
        consume_authorized_scope(state, "sparse")
    principal = current_principal()
    if principal is None:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "检索必须在认证主体上下文中执行",
            retryable=False,
        )
    kb_id = str(state.get("kb_id") or "").strip()
    if not kb_id:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "检索缺少知识库范围",
            retryable=False,
        )
    profile = profile_from_state(state)
    query_run_deadline = _deadline_from_state(state)
    if deadline_at is not None:
        if deadline_at.tzinfo is None:
            raise RetrievalEngineError(
                "RETRIEVAL_INVALID_DEADLINE",
                "路线截止时间必须包含时区",
                retryable=False,
            )
        deadline_at = deadline_at.astimezone(UTC)
        effective_deadline = (
            min(query_run_deadline, deadline_at)
            if query_run_deadline is not None
            else deadline_at
        )
    else:
        effective_deadline = query_run_deadline
    request = RetrievalRequest(
        query=query_text,
        variants=tuple(state.get("questions") or ()),
        scope=RetrievalScope(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            kb_ids=(kb_id,),
        ),
        profile=profile,
        filters=_parse_filter(state.get("retrieval_filter")),
        document_ids=tuple(str(item) for item in (state.get("doc_ids") or ()) if item),
        trace_id=f"ret_{uuid4().hex}",
        deadline_at=effective_deadline,
    )
    result = _adapter(state, cancellation=cancellation).search(request)
    diagnostics = _diagnostics(result, variant=query_variant)
    raw_hits: list[dict] = []
    observations_by_chunk: dict[str, list[dict]] = {}
    for hit in result.hits:
        observations = [
            {
                **score.model_dump(mode="json", exclude_none=True),
                "route": f"{query_variant}.{score.route}",
                "query_variant": query_variant,
                "sub_query_id": sub_query_id,
            }
            for score in hit.scores
        ]
        observations_by_chunk[hit.chunk_id] = observations
        raw_hits.append(
            {
                "chunk_id": hit.chunk_id,
                "source_chunk_id": hit.chunk_id,
                "evidence_id": hit.evidence_id,
                "content_hash": hit.content_hash,
                "content": hit.content,
                "title": hit.title,
                "file_title": hit.title,
                "tenant_id": hit.tenant_id,
                "kb_id": hit.kb_id,
                "document_id": hit.document_id,
                "document_version_id": hit.document_version_id,
                "index_version_id": hit.index_version_id,
                "retrieval_score": hit.final_score,
                "score": hit.final_score,
                "type": "milvus",
                "url": None,
            }
        )

    hydrated = hydrate_retrieved_hits(
        raw_hits,
        kb_id=kb_id,
        route=query_variant,
    )
    if authorized_scope is not None:
        hydrated = [
            item
            for item in hydrated
            if authorized_scope.permits(
                tenant_id=str(item.get("tenant_id") or ""),
                kb_id=str(item.get("kb_id") or ""),
                document_id=str(item.get("document_id") or ""),
                document_version_id=str(item.get("document_version_id") or ""),
                index_version_id=str(item.get("index_version_id") or ""),
            )
        ]
    for item in hydrated:
        item["retrieval_observations"] = observations_by_chunk.get(
            str(item.get("chunk_id") or ""), item.get("retrieval_observations") or []
        )
    diagnostics["hydrated_count"] = len(hydrated)
    diagnostics["discarded_by_evidence"] = len(raw_hits) - len(hydrated)
    return hydrated, diagnostics


def rerank_candidates(
    state: dict,
    candidates: list[dict],
    *,
    text_preparer=None,
) -> list[dict]:
    """经 ``RerankerPort`` 重排 legacy 查询图候选，并保留原字典载荷。"""

    consume_authorized_scope(state, "rerank")
    if not candidates:
        return []
    profile = profile_from_state(state)
    by_key: dict[str, dict] = {}
    domain_hits: list[RetrievalHit] = []
    principal = current_principal()
    for position, candidate in enumerate(candidates, start=1):
        raw_key = str(
            candidate.get("chunk_id")
            or candidate.get("evidence_id")
            or candidate.get("url")
            or f"candidate-{position}"
        )
        key = (
            raw_key
            if len(raw_key) <= 128
            else f"candidate_{sha256(raw_key.encode('utf-8')).hexdigest()}"
        )
        if key in by_key:
            key = f"candidate_{sha256(f'{raw_key}:{position}'.encode('utf-8')).hexdigest()}"
        content = str(candidate.get("text") or candidate.get("content") or "").strip()
        if not content:
            continue
        raw_hash = str(candidate.get("content_hash") or "")
        if not (
            len(raw_hash.removeprefix("sha256:")) == 64
            and all(ch in "0123456789abcdef" for ch in raw_hash.removeprefix("sha256:").lower())
        ):
            raw_hash = f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"
        observations: list[RouteScore] = []
        seen_routes: set[str] = set()
        for observation in candidate.get("retrieval_observations") or []:
            route = str(observation.get("route") or "retrieval")[:64]
            if route in seen_routes:
                continue
            seen_routes.add(route)
            observations.append(
                RouteScore(
                    route=route,
                    raw_score=observation.get("raw_score"),
                    rank=max(1, int(observation.get("rank") or position)),
                    rrf_score=observation.get("rrf_score"),
                )
            )
        if not observations:
            observations.append(
                RouteScore(
                    route=str(candidate.get("type") or "retrieval")[:64],
                    raw_score=candidate.get("retrieval_score", candidate.get("score")),
                    rank=position,
                )
            )
        evidence_id = candidate.get("evidence_id")
        hit_data = {
            "chunk_id": key,
            "tenant_id": str(
                candidate.get("tenant_id") or (principal.tenant_id if principal else "web")
            ),
            "kb_id": str(candidate.get("kb_id") or state.get("kb_id") or "web"),
            "document_id": str(candidate.get("document_id") or key[:64]),
            "document_version_id": str(candidate.get("document_version_id") or key[:64]),
            "index_version_id": str(candidate.get("index_version_id") or key[:64]),
            "content": content,
            "content_hash": raw_hash,
            "title": candidate.get("title") or candidate.get("file_title"),
            "scores": tuple(observations),
            "final_score": candidate.get("score"),
        }
        if evidence_id:
            hit_data["evidence_id"] = evidence_id
        hit = RetrievalHit.model_validate(hit_data)
        by_key[key] = candidate
        domain_hits.append(hit)

    if not domain_hits:
        return []
    load_started = _reranker_clock()
    try:
        reranker_model = _load_reranker_model()
    except Exception as exc:
        failure_details = _reranker_load_error_details(exc)
        if reranker_config.failure_mode == "error":
            raise RetrievalEngineError(
                "RERANKER_UNAVAILABLE",
                "重排模型加载失败",
                retryable=True,
                route="reranker",
                details=failure_details,
            ) from exc

        # 与模型推理阶段的 preserve_order 行为保持一致：只复制进入领域层的
        # 原候选，既不写入伪造 rerank_score，也不丢失 Evidence/provenance 等
        # 任意源诊断字段。
        output = [dict(by_key[hit.chunk_id]) for hit in domain_hits[: profile.top_k]]
        diagnostics = dict(state.get("retrieval_diagnostics") or {})
        diagnostics["reranker"] = {
            "status": "degraded",
            "model_id": profile.reranker.model_id,
            "elapsed_ms": max(0, int((_reranker_clock() - load_started) * 1000)),
            "input_count": len(domain_hits),
            "output_count": len(output),
            "batch_size": reranker_config.batch_size,
            "batch_count": 0,
            "truncated_count": 0,
            "max_input_tokens": min(
                profile.reranker.max_input_tokens,
                reranker_config.max_input_tokens,
            ),
            "failure_code": "RERANKER_UNAVAILABLE",
            "message": "重排模型不可用，已保留融合候选顺序",
            "fallback": "preserve_order",
            **failure_details,
        }
        state["retrieval_diagnostics"] = diagnostics
        return output

    reranker = BGERerankerAdapter(
        model=reranker_model,
        text_preparer=text_preparer,
        batch_size=reranker_config.batch_size,
        max_input_tokens=reranker_config.max_input_tokens,
        query_max_tokens=reranker_config.query_max_tokens,
        document_max_tokens=reranker_config.document_max_tokens,
        failure_mode=reranker_config.failure_mode,
    )
    result = reranker.rerank(
        RerankRequest(
            query=str(state.get("rewritten_query") or state.get("original_query") or ""),
            candidates=tuple(domain_hits),
            config=profile.reranker,
            trace_id=f"rerank_{uuid4().hex}",
        )
    )
    output: list[dict] = []
    for hit in result.hits[: profile.top_k]:
        source = dict(by_key[hit.chunk_id])
        if result.status == "degraded":
            output.append(source)
            continue
        rerank_score = next(
            (score.raw_score for score in hit.scores if score.route == "reranker"),
            hit.final_score,
        )
        source["rerank_score"] = rerank_score
        source["score"] = rerank_score
        source["retrieval_observations"] = [
            score.model_dump(mode="json", exclude_none=True) for score in hit.scores
        ]
        output.append(source)
    diagnostics = dict(state.get("retrieval_diagnostics") or {})
    diagnostics["reranker"] = {
        "status": result.status,
        "model_id": result.model_id,
        "elapsed_ms": result.elapsed_ms,
        "input_count": len(domain_hits),
        "output_count": len(output),
        "batch_size": result.batch_size,
        "batch_count": result.batch_count,
        "truncated_count": result.truncated_count,
        "max_input_tokens": result.max_input_tokens,
    }
    if result.status == "degraded":
        diagnostics["reranker"].update(
            {
                "failure_code": result.failure_code,
                "message": result.message,
                "fallback": "preserve_order",
            }
        )
    state["retrieval_diagnostics"] = diagnostics
    return output


def preview_retrieval_profile(
    *,
    kb_id: str,
    query: str,
    profile_record: dict,
    raw_filter: dict | None = None,
) -> dict:
    """用 draft/published 精确版本执行只读预览，不改变 KB 发布指针。"""

    principal = current_principal()
    if principal is None:
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE", "预览必须在认证主体上下文执行", retryable=False
        )
    config = dict(profile_record.get("config") or {})
    config.update(
        {
            "profile_id": profile_record["profile_id"],
            "version": int(profile_record["version"]),
            "tenant_id": principal.tenant_id,
            "status": profile_record["status"],
        }
    )
    profile = RetrievalProfile.model_validate(config)
    request = RetrievalRequest(
        query=query,
        scope=RetrievalScope(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            kb_ids=(kb_id,),
        ),
        profile=profile,
        execution_mode="preview",
        filters=_parse_filter(raw_filter),
        trace_id=f"preview_{uuid4().hex}",
    )
    result = _adapter().search(request)
    return {
        **_diagnostics(result, variant="preview"),
        "profile_hash": profile_record.get("config_hash"),
        "results": [
            {
                "chunk_id": hit.chunk_id,
                "evidence_id": hit.evidence_id,
                "document_id": hit.document_id,
                "title": hit.title,
                "content_preview": hit.content[:500],
                "final_score": hit.final_score,
                "scores": [
                    score.model_dump(mode="json", exclude_none=True) for score in hit.scores
                ],
            }
            for hit in result.hits
        ],
    }


__all__ = [
    "preview_retrieval_profile",
    "profile_from_state",
    "rerank_candidates",
    "retrieve_candidates",
]
