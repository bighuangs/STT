import inspect

from app.application.retrieval_service import rerank_candidates
from app.domain.retrieval import RetrievalEngineError
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.config.reranker_config import reranker_config
from app.shared.runtime.logger import step_log, logger

# ====================== 重排全局配置 ======================
RERANK_MAX_TOPK: int = 10  # 动态截断最多保留10条结果
RERANK_MIN_TOPK: int = 1  # 动态截断最少保留1条结果
RERANK_GAP_RATIO: float = 0.5  # 相邻分数相对下降 50% 时截断
RERANK_GAP_ABS: float = 2  # 分数断崖绝对值阈值
RERANK_BATCH_SIZE: int = reranker_config.batch_size
RERANK_MAX_INPUT_TOKENS: int = reranker_config.max_input_tokens
RERANK_QUERY_MAX_TOKENS: int = reranker_config.query_max_tokens
RERANK_DOCUMENT_MAX_TOKENS: int = reranker_config.document_max_tokens
RERANK_SPECIAL_TOKENS: int = 8  # 为 pair/instruction/model 特殊 token 留硬余量
RERANK_PRETOKEN_CHARS_PER_TOKEN: int = 8


@step_log("validate_rerank_inputs")
def validate_rerank_inputs(state: dict) -> tuple[list[dict], list[dict]]:
    """
    校验重排节点输入是否合法
    允许本地结果为空 或 联网结果为空，但不允许两者都为空
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    # TODO-28 后 Web 已在统一融合层消费，禁止在 rerank 前重复追加。
    if state.get("fusion_result"):
        web_search_docs = []

    return rrf_chunks, web_search_docs


@step_log("merge_rrf_and_web")
def merge_rrf_and_web(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    """
    统一合并本地知识库结果 + 联网搜索结果
    统一字段格式，方便后续重排模型统一打分
    """
    final_chunk_list: list[dict] = []

    # 处理本地知识库结果
    for chunk in rrf_chunks or []:
        # 透传稳定 provenance，避免 rerank 后无法计算检索指标或下钻版本。
        final_chunk_list.append(
            {
                **chunk,
                "title": chunk.get("title") or chunk.get("file_title"),
                "text": chunk.get("content") or chunk.get("text") or "",
                "url": chunk.get("url"),
                "type": chunk.get("type", "milvus"),
                "retrieval_score": chunk.get("retrieval_score"),
                "rrf_score": chunk.get("rrf_score", chunk.get("score", 0.0)),
                "score": chunk.get("rrf_score", chunk.get("score", 0.0)),
            }
        )

    # 处理联网搜索网页结果
    for doc in web_search_docs or []:
        final_chunk_list.append(
            {
                "title": doc.get("title"),
                "text": doc.get("snippet"),  # 网页结果取摘要snippet
                "url": doc.get("url"),  # 网页保留URL
                "type": "web",
                "retrieval_score": doc.get("score"),
                "rrf_score": None,
                "score": 0.0,
            }
        )

    return final_chunk_list


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_token_budget(tokenizer, text: str, token_budget: int) -> str:
    """只在本地按 tokenizer 确定性截断，不改写原 Evidence，也不调用外部模型。"""

    if token_budget <= 0 or not text:
        return ""
    max_chars = max(token_budget, token_budget * RERANK_PRETOKEN_CHARS_PER_TOKEN)
    candidate = text[:max_chars]

    # 优先按 token ID 解码，避免中文/英文字符比例的经验估算。部分轻量 fake
    # tokenizer 没有 decode，此时退回字符前缀二分；两条路径都会用同一 tokenizer
    # 复核最终预算。
    token_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if len(token_ids) <= token_budget:
        return candidate
    decode = getattr(tokenizer, "decode", None)
    truncated = ""
    if callable(decode):
        try:
            truncated = str(
                decode(
                    token_ids[:token_budget],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            truncated = str(decode(token_ids[:token_budget]))

    if not truncated or _token_count(tokenizer, truncated) > token_budget:
        low, high, best = 0, len(candidate), 0
        while low <= high:
            middle = (low + high) // 2
            if _token_count(tokenizer, candidate[:middle]) <= token_budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        truncated = candidate[:best]

    # 某些 tokenizer 的 decode 会因规范化产生额外 token；逐字符收敛保证硬上限。
    while truncated and _token_count(tokenizer, truncated) > token_budget:
        truncated = truncated[:-1]
    return truncated


@step_log("score_and_sort_chunks")
def build_question_pairs(question: str, final_chunk_list: list[dict], reranker) -> list[list[str]]:
    """
    构建重排模型输入对：[问题, 文本]
    自动处理超长文本：使用 reranker 自身 tokenizer 做确定性、有界截断。

    原始 Evidence 字典和正文保持不变；截断结果只存在于本地 reranker 输入中，
    不会经过聊天模型或任何外发通道。
    """
    tokenizer = reranker.tokenizer

    available_tokens = RERANK_MAX_INPUT_TOKENS - RERANK_SPECIAL_TOKENS
    question_budget = min(RERANK_QUERY_MAX_TOKENS, max(1, available_tokens - 1))
    question_for_rerank = _truncate_to_token_budget(tokenizer, question, question_budget)
    question_tokens = tokenizer.encode(question_for_rerank, add_special_tokens=False)
    answer_budget = min(
        RERANK_DOCUMENT_MAX_TOKENS,
        max(1, available_tokens - len(question_tokens)),
    )
    question_pairs: list[list[str]] = []

    for item in final_chunk_list:
        answer = item.get("text") or ""
        answer_for_rerank = _truncate_to_token_budget(tokenizer, answer, answer_budget)
        question_pairs.append([question_for_rerank, answer_for_rerank])

    return question_pairs


@step_log("score_and_sort_chunks")
def score_and_sort_chunks(state: dict, final_chunk_list: list[dict]) -> list[dict]:
    """
    调用重排模型对所有候选文档打分，并按分数从高到低排序
    """
    if not final_chunk_list:
        return []

    # 获取用户查询问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""

    # 获取重排模型实例
    reranker = llm_provider.reranker_model()

    # 构建模型输入对
    question_pairs = build_question_pairs(rewritten_query, final_chunk_list, reranker)

    # 小批次模型打分。显式传 max_length，避免 FlagEmbedding 回落到内部宽批次和
    # 隐式长度；对旧 fake/兼容模型仅传它声明支持的参数。
    score_list: list[float] = []
    values = {
        "normalize": True,
        "batch_size": RERANK_BATCH_SIZE,
        "query_max_length": RERANK_QUERY_MAX_TOKENS,
        "max_length": RERANK_MAX_INPUT_TOKENS,
    }
    try:
        parameters = inspect.signature(reranker.compute_score).parameters.values()
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        names = {parameter.name for parameter in parameters}
        score_kwargs = values if accepts_kwargs else {
            name: value for name, value in values.items() if name in names
        }
    except (TypeError, ValueError):
        score_kwargs = {"normalize": True}
    try:
        for start in range(0, len(question_pairs), RERANK_BATCH_SIZE):
            batch = question_pairs[start : start + RERANK_BATCH_SIZE]
            raw_scores = reranker.compute_score(batch, **score_kwargs)
            if isinstance(raw_scores, (float, int)):
                raw_scores = [raw_scores]
            batch_scores = [float(score) for score in raw_scores]
            if len(batch_scores) != len(batch):
                raise RetrievalEngineError(
                    "RERANKER_INVALID_RESPONSE",
                    "重排模型返回的分数数量无效",
                    retryable=False,
                    route="reranker",
                )
            score_list.extend(batch_scores)
    except RetrievalEngineError:
        raise
    except Exception as exc:
        if reranker_config.failure_mode == "preserve_order":
            logger.warning(
                "重排模型调用失败，legacy 路径保留融合顺序：{}",
                type(exc).__name__,
            )
            return final_chunk_list
        raise RetrievalEngineError(
            "RERANKER_UNAVAILABLE",
            "重排模型调用失败",
            retryable=True,
            route="reranker",
        ) from exc

    # 分数写入，将重排模型输出的原始分数四舍五入保留 4 位小数
    for score, chunk in zip(score_list, final_chunk_list):
        chunk["rerank_score"] = round(float(score), 4)
        chunk["score"] = chunk["rerank_score"]

    # 降序排序
    final_chunk_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return final_chunk_list


@step_log("dynamic_topk")
def dynamic_topk(chunk_list_score_sorted: list[dict]) -> list[dict]:
    """
    动态值截断：根据分数断崖自动决定保留多少条结果
    不是固定取前N条，而是找到分数突变的位置截断
    避免把明显低质量的长尾候选一并带入答案生成阶段
    """
    min_topk = RERANK_MIN_TOPK
    max_topk = min(RERANK_MAX_TOPK, len(chunk_list_score_sorted))
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    topk = max_topk  # 默认取最大条数

    # 遍历寻找分数断崖
    if topk > min_topk:
        for index in range(min_topk - 1, max_topk - 1):
            score_1 = chunk_list_score_sorted[index].get("score", 0.0)
            score_2 = chunk_list_score_sorted[index + 1].get("score", 0.0)
            # 连续的分数对的分数断崖绝对值
            abs_gap = abs(score_1 - score_2)
            ratio_score = max(0.0, score_1 - score_2) / max(abs(score_1), 1e-7)

            if ratio_score >= gap_ratio or abs_gap >= max_gap:
                topk = index + 1
                break
    return chunk_list_score_sorted[:topk]


@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
        每个元素都是：
        1. 第一个位置是当前问题；
        2. 第二个位置是候选文档正文。
        模型会对每一对分别输出一个相关性分数。
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表
    """

    # 1.校验输入
    rrf_chunks, web_search_docs = validate_rerank_inputs(state)

    # 2.统一格式合并两路结果
    merged = merge_rrf_and_web(rrf_chunks, web_search_docs)

    # 3. 通过 RerankerPort 执行；模型、阈值、动态 Top-K 和最终 top_k 均来自
    # 会话固定的 RetrievalProfile。适配器使用本地 reranker tokenizer 做硬预算
    # 截断；这里不再注册任何聊天模型摘要或可外发的 text_preparer。
    return rerank_candidates(state, merged)
