import json
from collections.abc import Callable

from app.application.retrieval_service import (
    retrieve_candidates,
    retrieve_exact_chapter_heading_candidates,
)
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log, logger


# ====================== 检索配置 ======================
# 默认返回的最大知识库片段数量
RETRIEVAL_DEFAULT_LIMIT = 5
# 混合检索权重：dense向量权重 0.9，sparse向量权重 0.1
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)


@step_log("build_retrieval_text")
def build_retrieval_text(state: QueryGraphState) -> str:
    """
    组装最终用于向量检索的查询文本（对齐 ragflow `" ".join(questions)`）。

    优先使用查询改写阶段产出的 questions（含跨语言变体 + 末位追加的关键词），
    回退到 rewritten_query，再回退到 original_query。
    """
    questions = state.get("questions") or []
    text = " ".join(q for q in questions if q).strip()
    return text or state.get("rewritten_query") or state.get("original_query") or ""


@step_log("validate_retrieval_state")
def validate_retrieval_state(state: QueryGraphState) -> tuple[str, list[str]]:
    """
    校验检索必须的核心参数是否存在。

    检索改由「查询改写」阶段驱动：必须有可检索文本（questions / rewritten_query /
    original_query 至少其一）；doc_ids 为可选的元数据过滤范围（空=不过滤）。

    Returns:
        tuple[str, list[str]]: (检索文本, 候选文档ID列表)
    """
    query_text = build_retrieval_text(state)
    if not query_text:
        logger.error("检索文本为空（questions/rewritten_query/original_query 均缺失），无法继续业务!")
        raise ValueError("检索文本为空,无法继续业务!")

    doc_ids = state.get("doc_ids") or []
    return query_text, doc_ids


@step_log("build_doc_id_expr")
def build_doc_id_expr(doc_ids: list[str] | None) -> str | None:
    """
    构造 Milvus 检索过滤表达式（对齐 ragflow：始终排除不可检索块，并可选限定文档范围）。

    - available_int == 1：仅检索「可被用户检索」的块，排除 TOC 目录块等内部块
      （available_int=0），对齐 ragflow 检索时的 available_int >= 1 过滤。
    - doc_id in [...]：元数据过滤（apply_meta_data_filter）命中的候选文档ID限定检索范围；
      doc_ids 为空时不加该条件，对全库检索。
    """
    available_expr = "available_int == 1"
    if not doc_ids:
        return available_expr
    # 仅保留给旧调用方的兼容帮助函数；生产查询已使用 RetrievalPort，
    # 且所有值均以 JSON 字面量转义。
    values = ", ".join(json.dumps(str(item), ensure_ascii=False) for item in doc_ids)
    return f"{available_expr} and doc_id in [{values}]"


@step_log("normalize_retrieved_chunk")
def normalize_retrieved_chunk(chunk: dict) -> dict:
    """
    将 Milvus 原始检索结果，统一格式化为查询链内部标准结构
    确保后续所有节点使用相同的数据结构
    """
    entity = chunk.get("entity", chunk)
    source_chunk_id = (
        entity.get("source_chunk_id")
        or entity.get("id")
        or entity.get("chunk_id")
    )
    return {
        # Milvus auto-PK 只用于诊断；RRF/Citation 必须使用应用稳定 chunk identity。
        "chunk_id": source_chunk_id,
        "source_chunk_id": source_chunk_id,
        "vector_pk": chunk.get("id") if chunk.get("entity") is not None else entity.get("chunk_id"),
        "evidence_id": entity.get("evidence_id"),
        "content_hash": entity.get("content_hash"),
        "item_name": entity.get("item_name", ""),  # 归属主体名称
        "title": entity.get("title"),  # 片段标题
        "parent_title": entity.get("parent_title"),  # 父标题/章节
        "part": entity.get("part"),  # 部分标识
        "file_title": entity.get("docnm_kwd") or entity.get("file_title"),
        "content": entity.get("content_with_weight") or entity.get("content", ""),
        "document_id": entity.get("doc_id"),
        "kb_id": entity.get("kb_id"),
        "tenant_id": entity.get("tenant_id"),
        "document_version_id": entity.get("document_version_id"),
        "index_version_id": entity.get("index_version_id"),
        "retrieval_score": chunk.get("distance", 0.0),
        "score": chunk.get("distance", 0.0),  # 当前阶段兼容字段
        "type": "milvus",  # 来源类型（向量库）
        "url": None,  # 附件URL（无）
    }


@step_log("search_chunks")
def search_chunks(
    query_text: str,
    doc_ids: list[str] | None = None,
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
    kb_id: str | None = None,
    route: str = "dense",
    state: dict | None = None,
) -> list[dict]:
    """
    基于检索文本执行混合向量检索，可选按 doc_ids 限定文档范围。
    这是本地知识库的核心检索逻辑。

    Args:
        query_text: 用于检索的查询文本（改写后的问题/变体拼接）
        doc_ids: 可选的候选文档ID列表，用于限定检索范围（空=全库检索）
        limit: 最多返回几条片段

    Returns:
        标准化后的知识库片段列表
    """
    compatibility_state = dict(state or {})
    compatibility_state.update(
        {
            "kb_id": kb_id or compatibility_state.get("kb_id") or "",
            "doc_ids": list(doc_ids or compatibility_state.get("doc_ids") or []),
        }
    )
    chunks, _ = retrieve_candidates(
        compatibility_state,
        query_text=query_text,
        query_variant=route,
    )
    # limit 仅为旧接口的进一步收窄；实际候选窗口由发布 Profile 控制。
    return chunks[: max(0, int(limit))]



@step_log("search_by_embedding")
def search_by_embedding(
    state: QueryGraphState,
    *,
    cancellation: Callable[[], bool] | None = None,
) -> list[dict]:
    """
    向量检索服务：
    1. 使用查询改写阶段产出的检索文本与候选文档范围（doc_ids）
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    query_text, _doc_ids = validate_retrieval_state(state)
    plan = state.get("query_plan") or {}
    subqueries = {
        str(item.get("sub_query_id")): str(item.get("query") or "")
        for item in (plan.get("sub_queries") or [])
        if item.get("purpose") == "retrieval"
        and item.get("sub_query_id")
        and item.get("query")
    }
    local_route = next(
        (
            route
            for route in (plan.get("routes") or [])
            if route.get("route_type") == "local_hybrid"
        ),
        None,
    )
    approved = [
        (sub_query_id, subqueries[sub_query_id])
        for sub_query_id in ((local_route or {}).get("sub_query_ids") or [])
        if sub_query_id in subqueries
    ]
    if not approved:
        approved = [(None, query_text)]

    chunks: list[dict] = []
    diagnostics_by_subquery: list[dict] = []
    for sub_query_id, sub_query in approved:
        current, diagnostics = retrieve_candidates(
            state,
            query_text=sub_query,
            query_variant="base",
            sub_query_id=sub_query_id,
            cancellation=cancellation,
        )
        exact_candidates, exact_diagnostics = retrieve_exact_chapter_heading_candidates(
            state,
            query_text=sub_query,
            sub_query_id=sub_query_id,
        )
        exact_chunk_ids = {
            str(item.get("chunk_id") or "") for item in exact_candidates if item.get("chunk_id")
        }
        current = exact_candidates + [
            item
            for item in current
            if str(item.get("chunk_id") or "") not in exact_chunk_ids
        ]
        for item in current:
            item["sub_query_ids"] = [sub_query_id] if sub_query_id else []
        chunks.extend(current)
        diagnostics_by_subquery.append(
            {
                "sub_query_id": sub_query_id,
                "query": sub_query,
                **diagnostics,
                "heading_exact": exact_diagnostics,
            }
        )
    state["embedding_retrieval_diagnostics"] = {
        "query_variant": "base",
        "status": "succeeded",
        "subqueries": diagnostics_by_subquery,
        "candidate_count": len(chunks),
    }
    return chunks
