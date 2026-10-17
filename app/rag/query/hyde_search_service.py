from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from langchain_core.messages import HumanMessage

from langchain_core.output_parsers import StrOutputParser
from app.application.model_usage_service import invoke_model_call
from app.application.query_security_pipeline import authorize_external_call
from app.application.retrieval_service import retrieve_candidates
from app.domain.retrieval import RetrievalEngineError
from app.infra.llm.providers import llm_provider
from app.rag.query.embedding_search_service import build_retrieval_text
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt


_INVALID_HYDE_ANSWER_MARKERS = (
    "请提供您想要查询的内容",
    "请提供您的查询",
    "请提供查询内容",
    "请提供问题",
    "rewritten_query",
)


CancellationProbe = Callable[[], bool]


def _raise_if_cancelled(cancellation: CancellationProbe | None) -> None:
    if cancellation is not None and cancellation():
        raise RetrievalEngineError(
            "RETRIEVAL_CANCELLED",
            "HyDE 路线已取消",
            retryable=False,
            route="question",
        )


def _remaining_seconds(deadline_at: datetime | None) -> float | None:
    if deadline_at is None:
        return None
    return max(0.001, (deadline_at - datetime.now(UTC)).total_seconds())


def normalize_hyde_answer(answer: str | None) -> str:
    """清洗并校验 HyDE 模型输出，无效时返回空字符串以触发原查询回退。"""
    normalized = (answer or "").strip()
    if not normalized:
        logger.warning("HyDE未生成有效内容，将回退为原检索文本")
        return ""
    if any(marker in normalized for marker in _INVALID_HYDE_ANSWER_MARKERS):
        logger.warning("HyDE返回了泛化询问或未渲染占位符，将回退为原检索文本")
        return ""
    return normalized


def build_hyde_query(query_text: str, hyde_answer: str | None) -> str:
    """构建 HyDE 检索文本；假设答案无效时仅使用原查询。"""
    normalized_answer = normalize_hyde_answer(hyde_answer)
    return f"{query_text}\n{normalized_answer}" if normalized_answer else query_text


@step_log("validate_retrieval_state")
def validate_retrieval_state(state: dict) -> tuple[str, list[str]]:
    """
    校验 HyDE 检索所需核心参数合法性
    确保前置「查询改写」流程已正常产出检索文本
    Args:
        state: 查询流程图全局状态
    Returns:
        tuple: (检索文本, 候选文档ID列表)
    """
    query_text = build_retrieval_text(state)
    # 改写后的检索文本为必填参数，缺失则无法执行检索
    if not query_text:
        logger.error(
            "检索文本为空（questions/rewritten_query/original_query 均缺失），无法继续业务!"
        )
        raise ValueError("检索文本为空,无法继续业务!")
    doc_ids = state.get("doc_ids") or []
    return query_text, doc_ids


@step_log("generate_hyde_answer")
def generate_hyde_answer(
    rewritten_query: str,
    *,
    state: dict | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """
    生成HYDE假设答案（核心语义增强能力）
    不用于最终回复，仅用于丰富检索语义、优化向量匹配效果
    Args:
        rewritten_query: 改写后的标准用户问句
    Returns:
        str: 大模型生成的场景化、标准化假设回答文本
    """
    if state is not None:
        authorize_external_call(
            state,
            purpose="llm_generation",
            provider="configured-llm",
        )
    # 加载HYDE专属提示词模板，传入优化后问句
    prompt_str = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    # 构造大模型对话请求
    messages = [HumanMessage(content=prompt_str)]
    # 调用大模型生成假设答案，纯文本输出
    client = llm_provider.chat()
    response = invoke_model_call(
        state,
        scene="hyde_generation",
        invoke=(
            (lambda: client.invoke(messages, timeout=timeout_seconds))
            if timeout_seconds is not None
            else (lambda: client.invoke(messages))
        ),
    )
    answer = StrOutputParser().invoke(response)
    return (answer or "").strip()


@step_log("search_chunks_with_hyde")
def search_chunks_with_hyde(
    *,
    query_text: str,
    doc_ids: list[str] | None = None,
    limit: int = 5,
    kb_id: str | None = None,
    state: dict | None = None,
    query_variant: str = "hyde",
    sub_query_id: str | None = None,
    cancellation: CancellationProbe | None = None,
    timeout_seconds: float | None = None,
    deadline_at: datetime | None = None,
) -> list[dict]:
    """
    执行 HyDE 增强检索核心逻辑
    拼接原始检索文本与模型假设答案，构造高语义检索文本，完成知识库召回
    Args:
        query_text: 前置「查询改写」流程产出的检索文本
        doc_ids: 可选的候选文档ID列表，用于限定检索范围（空=全库检索）
        limit: 检索返回知识片段最大数量
    Returns:
        list[dict]: HyDE 增强检索后的知识片段列表
    """
    # 大模型基于用户问题生成假设性标准答案，扩充弱语义信息
    _raise_if_cancelled(cancellation)
    hyde_answer = generate_hyde_answer(
        query_text,
        state=state,
        timeout_seconds=_remaining_seconds(deadline_at) or timeout_seconds,
    )
    # 同步 LLM 请求无法被 Python 强杀；若路线在调用期间超时，返回后必须在
    # embedding/Milvus 之前停止，不能产生迟到检索和候选。
    _raise_if_cancelled(cancellation)
    # 拼接原检索文本与假设答案；无效 HyDE 输出自动回退为原查询。
    hybrid_query = build_hyde_query(query_text, hyde_answer)
    # 复用基础检索能力，基于增强后的文本执行 Milvus 混合检索
    compatibility_state = dict(state or {})
    compatibility_state.update(
        {
            "kb_id": kb_id or compatibility_state.get("kb_id") or "",
            "doc_ids": list(doc_ids or compatibility_state.get("doc_ids") or []),
        }
    )
    chunks, diagnostics = retrieve_candidates(
        compatibility_state,
        query_text=hybrid_query,
        query_variant=query_variant,
        sub_query_id=sub_query_id,
        cancellation=cancellation,
        deadline_at=deadline_at,
    )
    if state is not None:
        state["hyde_retrieval_diagnostics"] = diagnostics
    return chunks[: max(0, int(limit))]


@step_log("search_embedding_hyde")
def search_embedding_hyde(
    state: dict,
    *,
    cancellation: CancellationProbe | None = None,
    timeout_seconds: float | None = None,
) -> list[dict]:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    作为多路检索的补充支路，专门优化短问句、口语化问句漏召回问题
    Args:
        state: 查询流程图全局状态
    Returns:
        list[dict]: 标准化的增强检索知识片段列表
    """
    # 校验检索前置参数完整性
    query_text, doc_ids = validate_retrieval_state(state)
    plan = state.get("query_plan") or {}
    subqueries = {
        str(item.get("sub_query_id")): str(item.get("query") or "")
        for item in (plan.get("sub_queries") or [])
        if item.get("purpose") == "retrieval" and item.get("sub_query_id") and item.get("query")
    }
    hyde_route = next(
        (route for route in (plan.get("routes") or []) if route.get("route_type") == "hyde"),
        None,
    )
    approved = [
        (sub_query_id, subqueries[sub_query_id])
        for sub_query_id in ((hyde_route or {}).get("sub_query_ids") or [])
        if sub_query_id in subqueries
    ]
    if not approved:
        approved = [(None, query_text)]

    chunks: list[dict] = []
    diagnostics_by_subquery: list[dict] = []
    route_deadline_at = (
        datetime.now(UTC) + timedelta(seconds=timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    for sub_query_id, sub_query in approved:
        _raise_if_cancelled(cancellation)
        current = search_chunks_with_hyde(
            query_text=sub_query,
            doc_ids=doc_ids,
            kb_id=state.get("kb_id"),
            state=state,
            query_variant="hyde",
            sub_query_id=sub_query_id,
            cancellation=cancellation,
            timeout_seconds=timeout_seconds,
            deadline_at=route_deadline_at,
        )
        for item in current:
            item["subquery_ids"] = [sub_query_id] if sub_query_id else []
        chunks.extend(current)
        diagnostics_by_subquery.append(
            {
                "sub_query_id": sub_query_id,
                "query": sub_query,
                **dict(state.get("hyde_retrieval_diagnostics") or {}),
            }
        )
    state["hyde_retrieval_diagnostics"] = {
        "query_variant": "hyde",
        "status": "succeeded",
        "subqueries": diagnostics_by_subquery,
        "candidate_count": len(chunks),
    }
    return chunks
