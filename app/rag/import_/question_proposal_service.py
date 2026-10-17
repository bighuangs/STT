"""
Service: 问题生成 / 配套提问 (question_proposal_service)

参考 ragflow rag/svr/task_executor.py:doc_question_proposal (L650-695)
       ragflow rag/prompts/generator.py:question_proposal (L327-347)
       ragflow rag/prompts/question_prompt.md（已迁移至 app/resources/prompts/question_proposal.prompt）

核心功能：遍历 state["chunks"]，为每个 chunk 调用 LLM 提取 topn 个候选问题，
写入 chunk 字段（命名与 ragflow 1:1 对齐，便于后续 Milvus 字段映射）：
    chunk["question_kwd"]: list[str]（LLM 原始问题列表，按行切分）
    chunk["question_tks"]: list[str]（jieba 词粒度切分结果，用于 Milvus 稀疏向量扩展）

业务逻辑：
    1. 读取 parser_config["auto_questions"]（topn）；<=0 时跳过整段
    2. 对每个非空 chunk 异步并发调用 LLM（LangChain ChatOpenAI）
    3. prompt 加载自 app/resources/prompts/question_proposal.prompt（由 load_prompt 渲染）
    4. 响应按 \\n 切分（一行一题，与 ragflow 一致），过滤空白、空串、编号前缀
    5. 错误 / 超时 / LLM 返回 **ERROR** 时，置 question_kwd=[]，不抛异常
       （与 ragflow 一致：失败兜底，不阻断主流程）
    6. 用 jieba 精确模式切分问题文本，作为 question_tks

对齐 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
    - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
    - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
    - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
    - 分词器（ragflow 用 rag_tokenizer.tokenize）—— STT 用 jieba 精确模式替代（中文友好、词粒度切分）
"""
import asyncio
import re

import jieba
from langchain_core.messages import HumanMessage, SystemMessage

jieba.setLogLevel("INFO")  # 与 app/rag/import_/entry_service.py:5 保持一致

from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.llm_limiter import ainvoke_with_retry


# 问题生成提示词文件名（位于 app/resources/prompts/）
# 模板原为 ragflow rag/prompts/question_prompt.md（去掉 jinja 语法）
_QUESTION_PROMPT_NAME = "question_proposal"

# 问题切分：与 ragflow task_executor.py:678 `cached.split("\n")` 完全一致
_QUESTION_LINE_SPLIT = re.compile(r"[\r\n]+")

# 编号前缀清理：去掉 LLM 偶尔输出的 "1." "2)" "1、" "①" 等行首编号
#   - r"^\s*(?:\d+[.\)、]|[①②③④⑤⑥⑦⑧⑨⑩])+\s*" 匹配 "1." / "2)" / "3、" / "①" 等
#   - 多级编号如 "1.1." / "1)" 也兼容
_LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.\)、]?|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*"
)


# ---------------------------------------------------------------------------
# Jieba 分词（中文优先；与 ragflow rag_tokenizer 中文切分语义对齐）
# ---------------------------------------------------------------------------
# 注：import jieba / jieba.setLogLevel 已在文件顶部完成，此处直接使用 jieba 即可


def _tokenize_questions(questions: list[str]) -> list[str]:
    """
    将问题列表用 jieba 精确模式切分，展平为 token 字符串列表。

    与 ragflow 的 `rag_tokenizer.tokenize("\n".join(d["question_kwd"]))` 行为对齐：
    中文友好、词粒度切分（而非字粒度），对后续 Milvus 稀疏向量扩展更友好。

    Returns:
        list[str]: 切分后的 token 列表，已去空字符串
    """
    if not questions:
        return []
    text = "\n".join(questions)
    return [tok for tok in jieba.lcut(text, cut_all=False) if tok and tok.strip()]


def _build_question_prompt(content: str, topn: int) -> str:
    """
    加载并渲染问题生成 prompt。

    Args:
        content: 待分析的 chunk 原文（已 strip）
        topn: 生成问题数量

    Returns:
        完整 prompt 字符串（直接作为 system message 发送给 LLM）
    """
    return load_prompt(_QUESTION_PROMPT_NAME, topn=topn, content=content)


# ---------------------------------------------------------------------------
# 异步 LLM 调用封装
# ---------------------------------------------------------------------------
async def _call_llm_for_questions(content: str, topn: int) -> str:
    """
    异步调用 LLM 返回问题原始字符串（已去 <think> 块，未切分）。

    使用 LangChain ChatOpenAI（与 STT 现有 LLM 栈一致）。

    并发限流：已通过 chat_limiter 实现（见下方 `async with chat_limiter`）。
    TODO(对齐 ragflow): 后续加入 LLM 缓存（get_llm_cache / set_llm_cache）
    TODO(对齐 ragflow): 后续加入取消机制（has_canceled）

    Args:
        content: chunk 原文
        topn: 生成问题数量

    Returns:
        LLM 原始输出字符串（一行一题）；出错时返回 ""
    """
    try:
        llm = llm_provider.chat(json_mode=False)
        prompt_text = _build_question_prompt(content, topn)
        msg = [
            SystemMessage(content=prompt_text),
            HumanMessage(content="输出："),
        ]
        # 并发限流 + 失败指数退避重试（吸收代理/网络抖动导致的连接错误）
        resp = await ainvoke_with_retry(llm, msg)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 与 ragflow generator.py:344 对齐：去掉思考链残留
        text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL).strip()
        if text.find("**ERROR**") >= 0:
            return ""
        return text
    except Exception as e:
        logger.exception(f"[question_proposal] LLM 调用失败: {e}")
        return ""


def _parse_questions(raw: str) -> list[str]:
    """
    把 LLM 原始输出解析为问题列表。

    切分规则与 ragflow task_executor.py:678 `cached.split("\\n")` 对齐：
    1. 按换行切分
    2. strip 空白
    3. 去掉行首编号前缀（"1. " "2) " "① " 等）—— ragflow 不做这步但生产中很有用
    4. 过滤空串

    Returns:
        清洗后的问题字符串列表
    """
    if not raw:
        return []
    parts = _QUESTION_LINE_SPLIT.split(raw)
    cleaned: list[str] = []
    for p in parts:
        line = p.strip()
        if not line:
            continue
        line = _LEADING_NUMBER_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


# ---------------------------------------------------------------------------
# 对单个 chunk 生成问题（异步）
# ---------------------------------------------------------------------------
async def _propose_for_chunk(chunk: dict, topn: int, content_max_chars: int = 4000) -> None:
    """
    异步处理单个 chunk，原地写入 question_kwd 与 question_tks。

    不抛异常：失败时写入空列表，保证主流程不被单个 chunk 拖垮。

    Args:
        chunk: 单个 chunk 字典（必须有 "content" 字段）
        topn: 问题数量
        content_max_chars: LLM 输入内容截断上限（从 parser_config.llm_content_max_chars 读取）
    """
    content = (chunk.get("content") or "").strip()
    if not content:
        chunk["question_kwd"] = []
        chunk["question_tks"] = []
        return

    # 内容过长时截断，避免 LLM 上下文溢出（与 ragflow message_fit_in 行为近似）
    if len(content) > content_max_chars:
        content = content[:content_max_chars]

    raw = await _call_llm_for_questions(content, topn)
    questions = _parse_questions(raw)
    # 截断到 topn（防止 LLM 偶尔多输出）
    questions = questions[:topn]

    chunk["question_kwd"] = questions
    chunk["question_tks"] = _tokenize_questions(questions)


# ---------------------------------------------------------------------------
# 主入口（节点调用）
# ---------------------------------------------------------------------------
@step_log("propose_questions")
def propose_questions(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 配套提问 / 设问生成主入口（同步包装器）。

    业务流程：
        1. 读取 parser_config["auto_questions"]（topn）；<=0 时跳过整段
        2. 为每个 chunk 调度异步 LLM 任务，asyncio.gather 并发执行
        3. 任一 chunk 异常被本地 try/except 吞掉，主流程不中断
        4. 原地修改 chunks[i]["question_kwd"] 与 chunks[i]["question_tks"]

    Args:
        state: LangGraph 流程状态

    Returns:
        更新后的 state（chunks 已原地追加 question_kwd / question_tks）
    """
    parser_config = state.get("parser_config", {}) or {}
    topn_raw = parser_config.get("auto_questions", 0) or 0
    try:
        topn = int(topn_raw)
    except (TypeError, ValueError):
        topn = 0
    # LLM 输入内容截断上限（与 keyword/tagging service 策略一致）
    content_max_chars = int(parser_config.get("llm_content_max_chars", 4000))

    chunks = state.get("chunks", []) or []
    if topn <= 0 or not chunks:
        logger.info(
            f"[propose_questions] 已跳过：topn={topn}, chunks={len(chunks)} "
            f"(auto_questions<=0 或无 chunks)"
        )
        # 即使跳过也保证字段存在性
        for c in chunks:
            c.setdefault("question_kwd", [])
            c.setdefault("question_tks", [])
        return state

    logger.info(
        f"[propose_questions] 开始为 {len(chunks)} 个 chunk 并发生成问题"
        f"（topn={topn}, content_max_chars={content_max_chars}）"
    )

    async def _run_all() -> None:
        tasks = [_propose_for_chunk(c, topn, content_max_chars) for c in chunks]
        # 局部 try/except 已在 _propose_for_chunk 内部处理；此处不抛
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run_all())

    # 统计
    succ = sum(1 for c in chunks if c.get("question_kwd"))
    logger.info(
        f"[propose_questions] 完成：{succ}/{len(chunks)} 个 chunk 成功生成问题"
    )
    return state
