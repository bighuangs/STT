"""
Service: 内容打标 (content_tagging_service)

参考 ragflow rag/svr/task_executor.py:doc_content_tagging (L862-992)
       ragflow rag/prompts/generator.py:content_tagging (L433-480)
       ragflow rag/prompts/content_tagging_prompt.md

核心功能：
    - 遍历 state["chunks"]，根据 parser_config 中的标签配置，
      调用 LLM 为每个 chunk 打上相关标签，结果写入 chunk["tags"]: list[str]。
    - 与 ragflow 关键差异：
        - STT 暂不支持 tag_kb_ids 向量检索匹配（阶段 A），统一走 LLM（阶段 B）
        - STT 标签列表来自 parser_config["tags"]（而非从参考知识库拉取）
        - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
        - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
        - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
"""
import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.llm_limiter import ainvoke_with_retry

# 内容打标提示词文件名（位于 app/resources/prompts/）
_CONTENT_TAGGING_PROMPT_NAME = "content_tagging"

# 从 LLM 返回的 JSON 中提取的有效分数下限
# 默认值 1（与 ragflow 1-10 分制对齐）；实际值由主入口 tag_content 从 parser_config["tag_min_score"] 读取
_DEFAULT_MIN_SCORE = 1

# 默认 topn：每个 chunk 最多打几个标签
# 实际值由主入口 tag_content 从 parser_config["topn_tags"] 读取
_DEFAULT_TOPN_TAGS = 3


# ---------------------------------------------------------------------------
# JSON 解析辅助：处理 LLM 返回的含重复 key 的 JSON
# ---------------------------------------------------------------------------

def _extract_json_string(text: str, start: int) -> tuple:
    """
    从 text 起始位置提取 JSON 字符串值。

    Args:
        text: 原始文本
        start: text[start] 必须是 '"'

    Returns:
        (string_value, total_chars_consumed)；解析失败返回 (None, 0)
    """
    if start >= len(text) or text[start] != '"':
        return None, 0
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return text[start + 1 : i], i - start + 1
        i += 1
    return None, 0


def _extract_json_value(text: str) -> tuple:
    """
    从 text 起始位置提取一个 JSON 值。

    支持：数字、字符串、true、false、null、对象、数组。

    Returns:
        (parsed_value, end_offset)；解析失败返回 (None, 0)
    """
    text = text.lstrip()
    if not text:
        return None, 0
    c = text[0]
    if c == '"':
        return _extract_json_string(text, 0)
    if c.isdigit() or c == "-":
        i = 0
        while i < len(text) and (text[i].isdigit() or text[i] in ".-+eE"):
            i += 1
        num_str = text[:i]
        try:
            return int(num_str), i
        except ValueError:
            try:
                return float(num_str), i
            except ValueError:
                return None, 0
    if text.startswith("true"):
        return True, 4
    if text.startswith("false"):
        return False, 5
    if text.startswith("null"):
        return None, 4
    if c == "{":
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[: i + 1]), i + 1
        return None, 0
    if c == "[":
        depth = 0
        for i, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[: i + 1]), i + 1
        return None, 0
    return None, 0


def _parse_json_with_duplicate_keys(text: str) -> dict | None:
    """
    解析可能含重复 key 的 JSON，保留最后一个 key 的值。

    LLM 在 json_mode 下可能输出 {"标签": 8, "标签": 9}，标准库 json.loads
    会抛 "multiple repeat" 异常。本函数逐字符扫描文本，提取所有 key-value 对，
    再倒序去重（后出现的 key 覆盖前面的），最终返回干净的 dict。

    Args:
        text: JSON 字符串（已去除 markdown 代码块包裹）

    Returns:
        解析后的 dict；若无法解析返回 None
    """
    try:
        kv_pairs: list[tuple[str, Any]] = []
        i = 0
        while i < len(text):
            # 跳过空白
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            if text[i] != '"':
                i += 1
                continue

            # 提取 key
            key, key_consumed = _extract_json_string(text, i)
            if key is None:
                i += 1
                continue

            # 找 ":"
            colon = text.find(":", i + key_consumed)
            if colon == -1:
                i += 1
                continue

            # 提取 value
            val, val_consumed = _extract_json_value(text[colon + 1 :])
            if val is None:
                i = colon + 1
                continue

            kv_pairs.append((key, val))
            i = colon + 1 + val_consumed

            # 跳过逗号
            while i < len(text) and text[i] in " \t\n\r,":
                i += 1

        if not kv_pairs:
            return None

        # 去重：倒序遍历，后出现的 key 覆盖前面的
        seen: dict = {}
        for k, v in reversed(kv_pairs):
            seen[k] = v
        return seen
    except (json.JSONDecodeError, TypeError, RecursionError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Prompt 渲染
# ---------------------------------------------------------------------------

def _build_content_tagging_prompt(content: str, all_tags: list[str], examples: list[dict], topn: int) -> str:
    """
    加载并渲染内容打标 prompt。

    与 ragflow generator.py:442-456 对齐：
        1. 读取 Jinja 模板
        2. 为每个 example 的 tags 字段序列化为 JSON（tags_json）
        3. 渲染 template，返回完整 prompt 字符串

    Args:
        content: 待打标的 chunk 原文
        all_tags: 参考标签列表（来自 parser_config["tags"]）
        examples: few-shot 示例列表，每项含 content / tags
        topn: 每个 chunk 最多打几个标签

    Returns:
        完整 prompt 字符串
    """
    from app.shared.runtime.logger import PROJECT_ROOT

    prompt_path = PROJECT_ROOT / "app" / "resources" / "prompts" / f"{_CONTENT_TAGGING_PROMPT_NAME}.prompt"
    raw = prompt_path.read_text(encoding="utf-8")

    # 为每个 example 构造 tags_json 字段（与 ragflow generator.py:444-445 对齐）
    for ex in examples:
        tags = ex.get("tags", {}) or {}
        ex["tags_json"] = json.dumps(tags, indent=2, ensure_ascii=False)

    # Jinja2 渲染
    from jinja2 import Template
    template = Template(raw, autoescape=False)
    rendered = template.render(
        topn=topn,
        all_tags=all_tags,
        examples=examples,
        content=content,
    )
    return rendered


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

async def _call_llm_for_tags(
    content: str,
    all_tags: list[str],
    examples: list[dict],
    topn: int,
) -> dict[str, int] | None:
    """
    调用 LLM 为单个 chunk 打标签（JSON 输出模式）。

    与 ragflow generator.py:content_tagging (L433-480) 对齐：
        1. 渲染 prompt（all_tags + examples + content）
        2. 使用 json_mode=True 强制返回合法 JSON
        3. 去掉 RAGFlow 思考链残留标签
        4. 解析 JSON，返回 {tag: score, ...} 字典；出错时返回 None

    Args:
        content: chunk 原文
        all_tags: 参考标签列表
        examples: few-shot 示例列表
        topn: 每个 chunk 最多打几个标签

    Returns:
        {tag: score, ...} 字典；出错时返回 None
    """
    try:
        llm = llm_provider.chat(json_mode=True)
        prompt_text = _build_content_tagging_prompt(content, all_tags, examples, topn)
        msg = [
            SystemMessage(content=prompt_text),
            HumanMessage(content="输出："),
        ]
        # 并发限流 + 失败指数退避重试（吸收代理/网络抖动导致的连接错误）
        resp = await ainvoke_with_retry(llm, msg)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        # 与 ragflow generator.py:459 对齐：去掉思考链残留
        text = re.sub(r"^.*\n**RESULT**\n", "", text, flags=re.DOTALL).strip()

        # 清理 Markdown 代码块包裹的 JSON（即使 json_mode=True，部分模型仍可能返回 ```json ... ```）
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        if "**ERROR**" in text:
            logger.warning(f"[tag_content] LLM 返回错误标记: {text[:200]}")
            return None

        # 解析 JSON：先尝试标准库，若遇重复 key 异常则用自定义解析器容错
        # 重复 key 场景：LLM 可能输出 {"产品介绍": 8, "产品介绍": 9}，标准库直接抛 multiple repeat
        result: dict | None = None
        try:
            result = json.loads(text)
        except json.JSONDecodeError as je:
            if "multiple repeat" in str(je):
                logger.warning(f"[tag_content] JSON 含重复 key，尝试修复: {text[:200]}")
                result = _parse_json_with_duplicate_keys(text)
            else:
                # 非重复 key 的普通解析错误，尝试兜底提取
                logger.warning(f"[tag_content] JSON 解析失败，尝试兜底提取: {text[:200]}")
                try:
                    first_brace = text.index("{")
                    last_brace = text.rindex("}")
                    result = json.loads(text[first_brace:last_brace + 1])
                except (ValueError, json.JSONDecodeError):
                    logger.warning(f"[tag_content] 兜底提取也失败: {text[:200]}")
                    return None

        # 过滤：只保留分数 >= MIN_SCORE 的标签
        if not isinstance(result, dict):
            return None
        # MIN_SCORE 已通过 _tag_for_chunk 的 min_score 参数传入，filter 逻辑下沉到 caller
        filtered = {k: int(v) for k, v in result.items() if int(v or 0) >= 1}
        return filtered if filtered else None

    except Exception as e:
        err_str = str(e)
        if "multiple repeat" in err_str:
            # 尝试用自定义解析器修复重复 key
            logger.warning(f"[tag_content] LLM 返回含重复 key 的 JSON，尝试修复: {text[:200]}")
            result = _parse_json_with_duplicate_keys(text)
            if isinstance(result, dict):
                # MIN_SCORE 在主入口 _tag_for_chunk 阶段统一过滤，此处仅做基础解析
                filtered = {k: int(v) for k, v in result.items() if int(v or 0) >= 1}
                return filtered if filtered else None
            logger.warning(f"[tag_content] 自定义解析器也失败: {e}")
        else:
            logger.warning(f"[tag_content] LLM 调用失败: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 对单个 chunk 打标签（异步）
# ---------------------------------------------------------------------------

async def _tag_for_chunk(
    chunk: dict,
    all_tags: list[str],
    topn: int,
    examples: list[dict],
    min_score: int = 1,
    content_max_chars: int = 4000,
) -> None:
    """
    异步处理单个 chunk，原地写入 tags。

    不抛异常：失败时写入空列表，保证主流程不被单个 chunk 拖垮。

    与 ragflow task_executor.py:doc_content_tagging (L927-975) 对齐：
        - 步骤 1：检查 LLM 缓存（STT 暂未实现，跳过）
        - 步骤 2：缓存未命中，调用 LLM 获取标签
        - 步骤 3：将结果写入 chunk["tags"]（列表格式，取 topn 个最高分标签）

    Args:
        chunk: 单个 chunk 字典（必须有 "content" 字段）
        all_tags: 参考标签列表
        topn: 每个 chunk 最多打几个标签
        examples: few-shot 示例列表
        min_score: 标签评分下限（从 parser_config.tag_min_score 读取）
        content_max_chars: LLM 输入内容截断上限（从 parser_config.llm_content_max_chars 读取）
    """
    content = (chunk.get("content") or "").strip()
    if not content:
        chunk["tags"] = []
        return

    # 内容过长时截断，避免 LLM 上下文溢出（与 keyword/question service 策略一致）
    if len(content) > content_max_chars:
        content = content[:content_max_chars]

    result = await _call_llm_for_tags(content, all_tags, examples, topn)
    if result is None:
        chunk["tags"] = []
        return

    # 过滤：丢弃不在参考标签集内的幻觉标签（LLM 偶发幻觉的兜底）
    all_tags_set = set(all_tags)
    result = {k: v for k, v in result.items() if k in all_tags_set}

    # 取出 topn 个最高分标签（与 ragflow task_executor.py:res 逻辑一致）
    sorted_tags = sorted(result.items(), key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, score in sorted_tags[:topn] if score >= min_score]
    chunk["tags"] = top_tags


# ---------------------------------------------------------------------------
# 主入口（节点调用）
# ---------------------------------------------------------------------------

@step_log("tag_content")
def tag_content(state: ImportGraphState) -> ImportGraphState:
    """
    遍历 state["chunks"]，根据 parser_config 中的标签配置逐 chunk 打标签，
    结果写入每个 chunk["tags"]: list[str]。

    parser_config 标签配置字段（与 ragflow parser_config 对齐）：

        parser_config["tag_kb_ids"]: list[str]
            - 参考知识库 ID 列表（RAGFlow 用于从已有知识库拉取标签集）
            - STT 暂不支持此功能，标签列表改由 parser_config["tags"] 提供
            - 为保持接口兼容，当 tag_kb_ids 非空但 tags 为空时，记录警告但不阻塞流程

        parser_config["tags"]: list[str]
            - 标签列表，LLM 从中选取最相关的标签打给每个 chunk
            - 若为空且 tag_kb_ids 也为空：不生成，跳过
            - 支持从 parser_config["tag_kb_ids"] 拉取（RAGFlow 行为）

        parser_config["topn_tags"]: int
            - 每个 chunk 最多打几个标签，默认 3
            - 若配置 <=0，使用 DEFAULT_TOPN_TAGS

    字段映射（与 ragflow task_executor.py:tag_content 对齐）：
        - chunk["tags"]: list[str]（标签列表，非 dict）

    业务逻辑：
        1. 读取 parser_config["tags"] 配置；若为空，检查 tag_kb_ids 并记录警告
        2. 读取 topn_tags 配置（默认 3）
        3. 构造 few-shot 示例（STT 暂未实现向量匹配阶段，统一用 LLM）
        4. 对每个 chunk 并发调用 LLM，提取标签列表
        5. 错误兜底：LLM 调用失败时，写入空列表
        6. 统计完成数量

    对齐 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
        - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
        - 并发限流（ragflow 用 chat_limiter）—— STT 暂未实现，使用简单 asyncio.gather
        - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
        - tag_kb_ids 向量匹配（ragflow 阶段 A）—— STT 暂未实现，统一走 LLM

    Args:
        state: LangGraph 状态字典

    Returns:
        state（原地修改 chunks）
    """
    parser_config = state.get("parser_config") or {}

    # -------------------------------------------------------------------------
    # 步骤 1：读取标签配置
    # -------------------------------------------------------------------------
    tag_kb_ids = parser_config.get("tag_kb_ids", []) or []
    tags = parser_config.get("tags", []) or []

    if not tags:
        if tag_kb_ids:
            # STT 暂不支持从参考知识库拉取标签，仅记录警告，保持兼容
            logger.warning(
                f"[tag_content] tag_kb_ids={tag_kb_ids} 已配置，"
                f"但 STT 暂不支持从此拉取标签列表；请同时配置 parser_config['tags']"
            )
        logger.info("[tag_content] tags 配置为空，跳过内容打标")
        return state

    # -------------------------------------------------------------------------
    # 步骤 2：读取 topn / min_score / content_max_chars 配置
    # -------------------------------------------------------------------------
    topn_raw = parser_config.get("topn_tags", _DEFAULT_TOPN_TAGS)
    try:
        topn = int(topn_raw)
    except (TypeError, ValueError):
        topn = _DEFAULT_TOPN_TAGS

    if topn <= 0:
        topn = _DEFAULT_TOPN_TAGS

    min_score_raw = parser_config.get("tag_min_score", _DEFAULT_MIN_SCORE)
    try:
        min_score = int(min_score_raw)
    except (TypeError, ValueError):
        min_score = _DEFAULT_MIN_SCORE

    content_max_chars = int(parser_config.get("llm_content_max_chars", 4000))

    # -------------------------------------------------------------------------
    # 步骤 3：构造 few-shot 示例（STT 暂未实现向量匹配阶段）
    # -------------------------------------------------------------------------
    # RAGFlow 行为：阶段 A 向量匹配命中的块作为 examples，供阶段 B 的 LLM 参考
    # STT 暂未实现向量匹配阶段，examples 暂时为空；
    # 若后续实现了 tag_kb_ids 向量匹配，可在此补充 examples 构造逻辑
    examples: list[dict] = []

    # -------------------------------------------------------------------------
    # 步骤 4：为每个 chunk 并发调用 LLM
    # -------------------------------------------------------------------------
    chunks = state.get("chunks", [])
    if not chunks:
        logger.info("[tag_content] chunks 为空，跳过")
        return state

    logger.info(
        f"[tag_content] 开始为 {len(chunks)} 个 chunk 并发打标签"
        f"（topn={topn}, min_score={min_score}, content_max_chars={content_max_chars}, "
        f"标签总数={len(tags)}, examples={len(examples)}）"
    )

    async def _run_all() -> None:
        tasks = [
            _tag_for_chunk(c, tags, topn, examples, min_score, content_max_chars)
            for c in chunks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run_all())

    # -------------------------------------------------------------------------
    # 步骤 5：统计
    # -------------------------------------------------------------------------
    succ = sum(1 for c in chunks if c.get("tags"))
    logger.info(
        f"[tag_content] 完成：{succ}/{len(chunks)} 个 chunk 成功生成标签"
    )
    return state
