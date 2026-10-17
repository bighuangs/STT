"""
Service: 文档目录构建 (toc_service)

参考 ragflow rag/svr/task_executor.py:build_TOC (L1009-1113)
       ragflow rag/prompts/generator.py:run_toc_from_text (L1185-1279)
       ragflow rag/prompts/generator.py:gen_toc_from_text (L1132-1150)
       ragflow rag/prompts/generator.py:assign_toc_levels (L1109-1122)
       ragflow rag/prompts/toc_from_text_system.md
       ragflow rag/prompts/toc_from_text_user.md
       ragflow rag/prompts/assign_toc_levels.md

核心功能：
    - 为整篇文档生成结构化目录（TOC），结果写入 state["toc"]: dict
    - 不修改 state["chunks"]，TOC 与 chunks 解耦，由下游入库节点读取

与 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
    - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
    - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
    - LLM 输入字段：ragflow 使用 chunk["content_with_weight"]，STT 使用 chunk["content"]

字段约定（与 ragflow task_executor.py:build_TOC L1104-1112 一致）：
    state["toc"] = {
        "doc_id": 文档ID,
        "items": [
            {"level": str, "title": str, "ids": [chunk_id, ...]},
            ...
        ],
        "content_with_weight": str,  # items 序列化的 JSON
        "available_int": 0,          # 0=不可被用户检索
        "page_num_int": [100000000], # 极大值，使 TOC 排在所有内容块之后
        "toc_kwd": "toc",            # 前端/下游据此识别
    }
"""
import asyncio
import copy
import json
import re
from typing import Any

import xxhash
from langchain_core.messages import HumanMessage, SystemMessage

from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger, step_log
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.llm_limiter import chat_limiter


# Prompt 文件名（位于 app/resources/prompts/）
_TOC_FROM_TEXT_SYSTEM_PROMPT_NAME = "toc_from_text_system"
_TOC_FROM_TEXT_USER_PROMPT_NAME = "toc_from_text_user"
_ASSIGN_TOC_LEVELS_PROMPT_NAME = "assign_toc_levels"

# 标题最大字符数限制（参考 ragflow generator.py:1222-1228，超过则视为无效）
# 兜底常量；实际由 build_toc 从 parser_config["toc_title_max_len_cn"] / ["toc_title_max_len_en"] 读取
_TITLE_MAX_LEN_CN = 25
_TITLE_MAX_LEN_EN = 80

# TOC 块 page_num_int 默认值：极大值，使 TOC 排在所有内容块之后
# 兜底常量；实际由 build_toc 从 parser_config["toc_page_num_int"] 读取
_TOC_PAGE_NUM_INT = 100000000

# TOC 块类型标记（前端/下游据此识别）
_TOC_KWD = "toc"


# ---------------------------------------------------------------------------
# Prompt 加载（与 content_tagging_service 保持一致风格：直接读 .prompt 文件）
# ---------------------------------------------------------------------------

def _load_prompt(prompt_name: str) -> str:
    """
    加载指定名称的 prompt 模板。

    Args:
        prompt_name: prompt 文件名（不含 .prompt 后缀）

    Returns:
        模板字符串
    """
    from app.shared.runtime.logger import PROJECT_ROOT

    prompt_path = PROJECT_ROOT / "app" / "resources" / "prompts" / f"{prompt_name}.prompt"
    return prompt_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 标题层级分配：LLM 调用
# ---------------------------------------------------------------------------

async def _assign_toc_levels(titles: list[str]) -> list[dict]:
    """
    调用 LLM 为每个标题分配层级（与 ragflow generator.py:assign_toc_levels L1109-1122 对齐）。

    Args:
        titles: 标题列表（已按文档顺序去重）

    Returns:
        [{"level": "1", "title": "..."}, ...] 列表
    """
    if not titles:
        return []

    last_error = ""
    for attempt in range(3):
        try:
            llm = llm_provider.chat(json_mode=True)
            system_template = _load_prompt(_ASSIGN_TOC_LEVELS_PROMPT_NAME)
            msg = [
                SystemMessage(content=system_template),
                HumanMessage(content=json.dumps(titles, ensure_ascii=False)),
            ]
            # chat_limiter 限制本事件循环内并发 LLM 调用数（阶段 2：层级分配）
            async with chat_limiter:
                resp = await llm.ainvoke(msg)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            # 清理 Markdown 代码块包裹
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.DOTALL)
            text = re.sub(r"\s*```$", "", text).strip()

            result = json.loads(text)
            if not isinstance(result, list):
                return []

            # 规范化：ragflow 也做这一步（generator.py:1247-1258），接受 dict / [level, title] 两种格式
            normalized = []
            for item in result:
                if isinstance(item, dict):
                    normalized.append({
                        "level": str(item.get("level", "0")),
                        "title": str(item.get("title", "")),
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    normalized.append({
                        "level": str(item[0]),
                        "title": str(item[1]),
                    })
            return normalized
        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                wait = 2 ** attempt
                logger.info(f"[_assign_toc_levels] 调用失败（尝试 {attempt + 1}/3），{wait}s 后重试: {e}")
                await asyncio.sleep(wait)
    logger.warning(f"[_assign_toc_levels] 调用失败: {last_error}", exc_info=True)
    return []


# ---------------------------------------------------------------------------
# 阶段 1：从 chunk 列表中提取标题
# ---------------------------------------------------------------------------

async def _gen_toc_from_text(chunk_sections: list[list[dict]], progress_callback=None) -> list[dict]:
    """
    并发调用 LLM 从 chunk 列表中提取 TOC 标题（对齐 ragflow generator.py:gen_toc_from_text L1132-1150）。

    Args:
        chunk_sections: 分批后的 chunk 列表，每批元素为 {idx_str: chunk_text}
        progress_callback: 进度回调（可选）

    Returns:
        toc 列表，每项 {"title": str, "chunk_id": str}
    """
    if not chunk_sections:
        return []

    system_template = _load_prompt(_TOC_FROM_TEXT_SYSTEM_PROMPT_NAME)
    user_template = _load_prompt(_TOC_FROM_TEXT_USER_PROMPT_NAME)

    all_titles: list[dict] = []

    async def _run_one(section: list[dict]) -> list[dict]:
        if progress_callback:
            progress_callback(msg="")
        last_error = ""
        for attempt in range(3):
            try:
                llm = llm_provider.chat(json_mode=True)
                text_input = "\n".join(
                    json.dumps(d, ensure_ascii=False) for d in section
                )
                rendered_user = user_template.replace("{{text}}", text_input)
                msg = [
                    SystemMessage(content=system_template),
                    HumanMessage(content=rendered_user),
                ]
                # chat_limiter 限制本事件循环内并发 LLM 调用数（阶段 1：标题抽取，多批并发）
                async with chat_limiter:
                    resp = await llm.ainvoke(msg)
                text = resp.content if isinstance(resp.content, str) else str(resp.content)
                # 清理 Markdown 代码块
                text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.DOTALL)
                text = re.sub(r"\s*```$", "", text).strip()

                data = json.loads(text)
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
                return []
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.info(f"[_run_one] 调用失败（尝试 {attempt + 1}/3），{wait}s 后重试: {e}")
                    await asyncio.sleep(wait)
        logger.warning(f"[_gen_toc_from_text] 调用失败: {last_error}")
        return []

    tasks = [asyncio.create_task(_run_one(section)) for section in chunk_sections if section]
    if not tasks:
        return []

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"[_gen_toc_from_text] gather 失败: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for r in results:
        if isinstance(r, list):
            all_titles.extend(r)
    return all_titles


def _split_chunks_for_toc(
    chunks_text: list[str], max_length_chars: int = 3000
) -> list[list[dict]]:
    """
    按字符预算将 chunk 打包为批次（STT 简化版，参考 ragflow generator.py:split_chunks L1155-1180）。

    与 ragflow 差异：ragflow 按 token 预算（上限 1024 token）切分，STT 用字符预算近似
    （避免引入 tokenizer 依赖）。预算越小批次越多→阶段 1 并发度越高。
    单个 chunk 超过 max_length 也不拆，避免破坏语义。

    Args:
        chunks_text: 纯文本 chunk 列表（按位置）
        max_length_chars: 单批字符上限

    Returns:
        批次列表，每批元素为 {idx_str: chunk_text}
    """
    result: list[list[dict]] = []
    batch: list[dict] = []
    batch_chars = 0

    for idx, text in enumerate(chunks_text):
        t = len(text)
        if batch_chars + t > max_length_chars and batch:
            result.append(batch)
            batch = []
            batch_chars = 0
        batch.append({str(idx): text})
        batch_chars += t
    if batch:
        result.append(batch)
    return result


def _is_pure_punctuation_or_number(s: str) -> bool:
    """
    判断字符串是否仅由数字/标点/空白组成（与 ragflow generator.py:1230 re.match 一致）。
    这样的 title 通常是页码、章节号残留，不应纳入 TOC。
    """
    return bool(re.match(r"^[0-9,.()/ \-]+$", s)) if s else True


# ---------------------------------------------------------------------------
# 两阶段合并执行（单一事件循环，解决 httpx 连接池跨循环 Connection error）
# ---------------------------------------------------------------------------

async def _run_toc_phases(
    section_batches: list[list[dict]],
    toc_title_max_len_cn: int,
    toc_title_max_len_en: int,
) -> tuple[list[dict], list[dict]]:
    """
    在单一事件循环内完成 TOC 两阶段 LLM 调用：
      阶段1 提取标题 → 同步过滤 → 阶段2 分配层级

    根本原因：两次独立的 asyncio.run() 各自创建新事件循环；ChatOpenAI/httpx
    连接池绑定首次创建的循环，跨循环复用缓存客户端时 TCP 连接已过期，
    导致"Connection error"并浪费 1+2=3s 重试等待。
    合并为单次 asyncio.run() 后，两阶段共享同一连接池，无需重连。

    Returns:
        (filtered, toc_with_levels)：
          filtered       — 通过有效性过滤的标题条目列表
          toc_with_levels — LLM 返回的带层级标题列表（filtered 为空时返回 []）
    """
    # 阶段 1：并发提取标题
    raw_titles = await _gen_toc_from_text(section_batches)

    # 阶段 1 结果过滤（纯 CPU 逻辑，直接在协程内执行，无需切回同步层）
    filtered: list[dict] = []
    for item in raw_titles:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title or title == "-1":
            continue
        is_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in title)
        max_len = toc_title_max_len_cn if is_chinese else toc_title_max_len_en
        if len(title) > max_len:
            continue
        if _is_pure_punctuation_or_number(title):
            continue
        chunk_val = item.get("chunk_id")
        if chunk_val is None or str(chunk_val).strip() == "":
            continue
        filtered.append(item)

    if not filtered:
        return filtered, []

    # 阶段 2：为过滤后标题分配层级（与阶段 1 共享事件循环，连接池不中断）
    titles_only = [x["title"] for x in filtered]
    toc_with_levels = await _assign_toc_levels(titles_only)
    return filtered, toc_with_levels


# ---------------------------------------------------------------------------
# 主入口（节点调用）
# ---------------------------------------------------------------------------

@step_log("build_toc")
def build_toc(state: ImportGraphState) -> ImportGraphState:
    """
    为整篇文档生成结构化 TOC，写入 state["toc"]。

    parser_config 字段（与 ragflow parser_config.toc_extraction 对齐）：
        parser_config["toc_extraction"]: bool
            - True / >0：启用 TOC 生成
            - False / 0 / 缺失：不生成，跳过

    业务逻辑：
        1. 检查 parser_config["toc_extraction"] 开关；未启用则跳过
        2. 按位置排序 chunks，提取纯文本列表
        3. 阶段 1：分批调用 LLM 从 chunk 文本中提取标题
        4. 过滤无效标题（-1 / 纯数字标点 / 超长）
        5. 阶段 2：调用 LLM 为标题分配层级
        6. 合并两个阶段结果，把 chunk_id 映射回真实 chunk["id"]，并填充跨块 ids
        7. 构造 TOC 块元数据，写入 state["toc"]

    对齐 ragflow 的关键差异（STT 暂未实现的部分，TODO 注释标注）：
        - LLM 缓存（ragflow 用 get_llm_cache / set_llm_cache）—— STT 暂未实现
        - 取消机制（ragflow 用 has_canceled）—— STT 暂未实现
        - LLM 限流（ragflow 用 chat_limiter）—— 已实现：见 llm_limiter.chat_limiter
        - LLM 输入字段：ragflow 使用 chunk["content_with_weight"]，STT 使用 chunk["content"]

    Args:
        state: LangGraph 状态字典

    Returns:
        state（state["toc"] 已被填充，未启用时保持空 dict）
    """
    parser_config = state.get("parser_config") or {}

    # -------------------------------------------------------------------------
    # 步骤 1：检查开关
    # -------------------------------------------------------------------------
    toc_extraction = parser_config.get("toc_extraction", False)
    if not toc_extraction:
        logger.info("[build_toc] toc_extraction=False，跳过目录生成")
        return state

    # 超参数：从 parser_config 读取，统一使用默认值兜底
    toc_title_max_len_cn = int(parser_config.get("toc_title_max_len_cn", _TITLE_MAX_LEN_CN))
    toc_title_max_len_en = int(parser_config.get("toc_title_max_len_en", _TITLE_MAX_LEN_EN))
    toc_batch_max_chars = int(parser_config.get("toc_batch_max_chars", 3000))
    toc_page_num_int = int(parser_config.get("toc_page_num_int", _TOC_PAGE_NUM_INT))

    chunks = state.get("chunks", [])
    if not chunks:
        logger.info("[build_toc] chunks 为空，跳过目录生成")
        return state

    doc_id = state.get("task_id", "") or ""
    logger.info(f"[build_toc] 开始为 {len(chunks)} 个 chunk 生成目录")

    # -------------------------------------------------------------------------
    # 步骤 2：按位置排序 + 兜底生成缺失的 chunk["id"]
    # -------------------------------------------------------------------------
    # 与 ragflow task_executor.py:1044-1047 对齐：按 page_num + top 排序
    # STT 的 chunks 暂无 page_num_int / top_int 字段，按原顺序处理即可
    ordered = list(chunks)
    for idx, c in enumerate(ordered):
        if "id" not in c or not c.get("id"):
            # 兜底：与 image_minio_service 同样的 hash 公式，保证 id 一致
            c["id"] = xxhash.xxh64(
                ((c.get("content") or "") + doc_id).encode("utf-8", "surrogatepass")
            ).hexdigest()

    chunks_text = [c.get("content") or "" for c in ordered]
    # 只取非空文本参与 TOC（hyperlinks 等可跳过）
    section_batches = _split_chunks_for_toc(chunks_text, max_length_chars=toc_batch_max_chars)

    # -------------------------------------------------------------------------
    # 步骤 3-5：两阶段 LLM 调用（单一事件循环，消除跨循环 Connection error，节省 ~3s 重试）
    # -------------------------------------------------------------------------
    filtered, toc_with_levels = asyncio.run(
        _run_toc_phases(section_batches, toc_title_max_len_cn, toc_title_max_len_en)
    )

    if not filtered:
        logger.info("[build_toc] LLM 未生成有效 TOC 标题")
        return state

    if not toc_with_levels:
        logger.warning("[build_toc] 层级分配失败，回退为单层目录")
        toc_with_levels = [{"level": "1", "title": item["title"]} for item in filtered]

    # 与 ragflow generator.py:1269-1277 对齐：按索引合并
    merged: list[dict] = []
    for i, src_item in enumerate(filtered):
        if i >= len(toc_with_levels):
            level_item = {"level": "1", "title": src_item.get("title", "")}
        else:
            level_item = toc_with_levels[i]
        merged.append({
            "level": level_item.get("level", "1"),
            "title": level_item.get("title", src_item.get("title", "")),
            "chunk_id": src_item.get("chunk_id", ""),
        })

    # -------------------------------------------------------------------------
    # 步骤 6：映射 chunk_id → 真实 chunk["id"]，并填充跨块 ids
    # -------------------------------------------------------------------------
    items: list[dict] = []
    for ii, item in enumerate(merged):
        chunk_val = item.get("chunk_id", "")
        try:
            curr_idx = int(chunk_val)
        except (ValueError, TypeError):
            logger.warning(f"[build_toc] 索引 {ii}：chunk_id={chunk_val!r} 不是有效整数，跳过")
            continue

        if curr_idx < 0 or curr_idx >= len(ordered):
            logger.warning(
                f"[build_toc] 索引 {ii}：chunk_id {curr_idx} 超出范围 [0, {len(ordered)})，跳过"
            )
            continue

        # 第一个块 ID 必入
        item_ids = [ordered[curr_idx]["id"]]

        # 若有下一个条目，计算跨块范围
        if ii + 1 < len(merged):
            next_val = merged[ii + 1].get("chunk_id", "")
            try:
                next_idx = int(next_val)
                for jj in range(curr_idx + 1, min(next_idx + 1, len(ordered))):
                    item_ids.append(ordered[jj]["id"])
            except (ValueError, TypeError):
                pass

        items.append({
            "level": item.get("level", "1"),
            "title": item.get("title", ""),
            "ids": item_ids,
        })

    if not items:
        logger.info("[build_toc] 映射后无有效 TOC 条目")
        return state

    # -------------------------------------------------------------------------
    # 步骤 7：构造 TOC 块元数据，写入 state["toc"]
    # -------------------------------------------------------------------------
    # 与 ragflow task_executor.py:1104-1112 对齐：复制最后一个 chunk 作为模板，
    # # STT 改为基于现有 chunk 字段构造最小骨架块
    content_with_weight = json.dumps(items, ensure_ascii=False)
    toc_block = {
        "doc_id": doc_id,
        "items": items,
        "content_with_weight": content_with_weight,
        "available_int": 0,
        "page_num_int": [toc_page_num_int],
        "toc_kwd": _TOC_KWD,
    }
    # 与 ragflow 一致：TOC 块 id 用 xxhash64(content_with_weight + doc_id)
    toc_block["id"] = xxhash.xxh64(
        (content_with_weight + doc_id).encode("utf-8", "surrogatepass")
    ).hexdigest()

    state["toc"] = toc_block
    logger.info(
        f"[build_toc] 完成：生成 {len(items)} 个 TOC 条目，"
        f"id={toc_block['id'][:16]}…"
    )
    return state
