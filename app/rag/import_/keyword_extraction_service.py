"""
Service: 关键词提取 (keyword_extraction_service)

参考 ragflow rag/svr/task_executor.py:doc_keyword_extraction (L592-641)
       ragflow rag/prompts/generator.py:keyword_extraction (L302-322)
       ragflow rag/prompts/keyword_prompt.md（已迁移至 app/resources/prompts/keyword_extraction.prompt）

核心功能：遍历 state["chunks"]，为每个 chunk 调用 LLM 提取 topn 个关键词，写入
    chunk["key_words"]: list[str]
    chunk["key_tokens"]: list[str]（jieba 切分结果，便于后续 Milvus 稀疏向量扩展）

业务逻辑：
    1. 读取 parser_config["auto_keywords"]（topn）；<=0 时跳过整段
    2. 对每个非空 chunk 异步并发调用 LLM（LangChain ChatOpenAI）
    3. prompt 加载自 app/resources/prompts/keyword_extraction.prompt（由 load_prompt 渲染）
    4. 响应按 , ，; ；、\\r\\n 多种分隔符切分，过滤空白与空串
    5. 错误 / 超时 / LLM 返回 **ERROR** 时，置 chunk["key_words"] = []，不抛异常
       （与 ragflow 一致：失败兜底，不阻断主流程）
    6. 用 jieba 精确模式切分关键词，作为 key_tokens

对齐 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
    - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
    - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
    - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
    - 分词器（ragflow 用 rag_tokenizer.tokenize）—— STT 用 jieba 精确模式替代（中文友好、词粒度切分）
"""
import asyncio
import re
from typing import Any

import jieba
from langchain_core.messages import HumanMessage, SystemMessage

jieba.setLogLevel("INFO")  # 与 app/rag/import_/entry_service.py:5 保持一致

from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.llm_limiter import ainvoke_with_retry


# 关键词提示词文件名（位于 app/resources/prompts/）
# 模板原为 ragflow rag/prompts/keyword_prompt.md（去掉 jinja 语法）
_KEYWORD_PROMPT_NAME = "keyword_extraction"

# 关键词分隔符：与 ragflow task_executor.py:623 保持一致（中英文逗号/分号/顿号/换行）
_KEYWORD_DELIMITER = re.compile(r"[,，;；、\r\n]+")


def _build_keyword_prompt(content: str, topn: int) -> str:
    """
    加载并渲染关键词提取 prompt。

    Args:
        content: 待分析的 chunk 原文（已 strip）
        topn: 提取关键词数量

    Returns:
        完整 prompt 字符串（直接作为 system message 发送给 LLM）
    """
    return load_prompt(_KEYWORD_PROMPT_NAME, topn=topn, content=content)


# ---------------------------------------------------------------------------
# Jieba 分词（中文优先；与 ragflow rag_tokenizer 中文切分语义对齐）
# ---------------------------------------------------------------------------
# 注：import jieba / jieba.setLogLevel 已在文件顶部完成，此处直接使用 jieba 即可


def _tokenize_keywords(keywords: list[str]) -> list[str]:
    """
    将关键词列表用 jieba 精确模式切分，展平为 token 字符串列表。

    与 ragflow 的 `rag_tokenizer.tokenize(" ".join(important_kwd))` 行为对齐：
    中文友好、词粒度切分（而非字粒度），对后续 Milvus 稀疏向量扩展更友好。

    Returns:
        list[str]: 切分后的 token 列表，已去空字符串
    """
    if not keywords:
        return []
    text = " ".join(keywords)
    return [tok for tok in jieba.lcut(text, cut_all=False) if tok and tok.strip()]


# ---------------------------------------------------------------------------
# 同步 LLM 调用封装（内部 async，外面走 asyncio.run）
# ---------------------------------------------------------------------------
async def _call_llm_for_keywords(content: str, topn: int) -> str:
    """
    异步调用 LLM 返回关键词原始字符串（已去 <think> 块，未切分）。

    使用 LangChain ChatOpenAI（与 STT 现有 LLM 栈一致）。

    并发限流：已通过 chat_limiter 实现（见下方 `async with chat_limiter`）。
    TODO(对齐 ragflow): 后续加入 LLM 缓存（get_llm_cache / set_llm_cache）
    TODO(对齐 ragflow): 后续加入取消机制（has_canceled）

    Args:
        content: chunk 原文
        topn: 关键词数量

    Returns:
        LLM 原始输出字符串；出错时返回 ""
    """
    try:
        llm = llm_provider.chat(json_mode=False)
        prompt_text = _build_keyword_prompt(content, topn)
        msg = [
            SystemMessage(content=prompt_text),
            HumanMessage(content="输出："),
        ]
        # 并发限流 + 失败指数退避重试（吸收代理/网络抖动导致的连接错误）
        resp = await ainvoke_with_retry(llm, msg)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 与 ragflow generator.py:319 对齐：去掉思考链残留
        text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL).strip()
        if text.find("**ERROR**") >= 0:
            return ""
        return text
    except Exception as e:
        logger.exception(f"[keyword_extraction] LLM 调用失败: {e}")
        return ""


def _parse_keywords(raw: str) -> list[str]:
    """
    把 LLM 原始输出解析为关键词列表。

    切分规则与 ragflow task_executor.py:623 完全一致：
        r"[,，;；、\\r\\n]+"

    Returns:
        清洗后的关键词字符串列表
    """
    if not raw:
        return []
    parts = _KEYWORD_DELIMITER.split(raw)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# 对单个 chunk 提取关键词（异步）
# ---------------------------------------------------------------------------
async def _extract_for_chunk(chunk: dict, topn: int, content_max_chars: int = 4000) -> None:
    """
    异步处理单个 chunk，原地写入 key_words 与 key_tokens。

    不抛异常：失败时写入空列表，保证主流程不被单个 chunk 拖垮。

    Args:
        chunk: 单个 chunk 字典（必须有 "content" 字段）
        topn: 关键词数量
        content_max_chars: LLM 输入内容截断上限（从 parser_config.llm_content_max_chars 读取）
    """
    content = (chunk.get("content") or "").strip()
    if not content:
        chunk["key_words"] = []
        chunk["key_tokens"] = []
        return

    # 内容过长时截断，避免 LLM 上下文溢出（与 ragflow message_fit_in 行为近似）
    if len(content) > content_max_chars:
        content = content[:content_max_chars]

    raw = await _call_llm_for_keywords(content, topn)
    kws = _parse_keywords(raw)
    # 截断到 topn（防止 LLM 偶尔多输出）
    kws = kws[:topn]

    chunk["key_words"] = kws
    chunk["key_tokens"] = _tokenize_keywords(kws)


# ---------------------------------------------------------------------------
# 主入口（节点调用）
# ---------------------------------------------------------------------------
@step_log("extract_keywords")
def extract_keywords(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档关键词提取主入口（同步包装器）。

    业务流程：
        1. 读取 parser_config["auto_keywords"]（topn）；<=0 时跳过整段
        2. 为每个 chunk 调度异步 LLM 任务，asyncio.gather 并发执行
        3. 任一 chunk 异常被本地 try/except 吞掉，主流程不中断
        4. 原地修改 chunks[i]["key_words"] 与 chunks[i]["key_tokens"]

    Args:
        state: LangGraph 流程状态

    Returns:
        更新后的 state（chunks 已原地追加 key_words / key_tokens）
    """
    parser_config = state.get("parser_config", {}) or {}
    topn_raw = parser_config.get("auto_keywords", 0) or 0
    try:
        topn = int(topn_raw)
    except (TypeError, ValueError):
        topn = 0
    # LLM 输入内容截断上限：避免 LLM 上下文溢出（与 ragflow message_fit_in 行为近似）
    content_max_chars = int(parser_config.get("llm_content_max_chars", 4000))

    chunks = state.get("chunks", []) or []
    if topn <= 0 or not chunks:
        logger.info(
            f"[extract_keywords] 已跳过：topn={topn}, chunks={len(chunks)} "
            f"(auto_keywords<=0 或无 chunks)"
        )
        # 即使跳过也保证字段存在性
        for c in chunks:
            c.setdefault("key_words", [])
            c.setdefault("key_tokens", [])
        return state

    logger.info(
        f"[extract_keywords] 开始为 {len(chunks)} 个 chunk 并发提取关键词"
        f"（topn={topn}, content_max_chars={content_max_chars}）"
    )

    async def _run_all() -> None:
        tasks = [_extract_for_chunk(c, topn, content_max_chars) for c in chunks]
        # 局部 try/except 已在 _extract_for_chunk 内部处理；此处不抛
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run_all())

    # 统计
    succ = sum(1 for c in chunks if c.get("key_words"))
    logger.info(
        f"[extract_keywords] 完成：{succ}/{len(chunks)} 个 chunk 成功生成关键词"
    )
    return state
