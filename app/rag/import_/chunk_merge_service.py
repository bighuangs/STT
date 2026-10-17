"""
文档块合并服务
在 metadata_extract_service 完成粗切分之后执行，只负责合并：
1. 基于 token 数做单 pass 累积合并（短块合并、短标题保护）
2. 支持块间重叠（overlapped_percent）
3. special chunk（type=hyperlinks/table）原样保留不参与合并
4. 备份结果到 chunks.json

注意：
- metadata_extract_service 已完成粗切分，此处不做切分，只做合并。
- hyperlinks/table 等 special chunk 由 link_table_chunk_service 生成，
  merge_and_backup_chunks 中原样追加到合并结果末尾。

参考 RAGFlow rag/app/naive.py:1601-1653 的 Markdown 合并逻辑（naive_merge）和
rag/nlp/__init__.py:1423 的通用合并逻辑（naive_merge）。
"""
import re
import json
from pathlib import Path
from typing import Any

import tiktoken

from app.shared.runtime.logger import logger, step_log


# ---------------------------------------------------------------------------
# token 计数（与 RAGFlow common/token_utils.py:29 保持一致）
# ---------------------------------------------------------------------------

_encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """返回字符串的 token 数量（tiktoken cl100k_base 编码）。参考 RAGFlow common/token_utils.py:29。"""
    try:
        return len(_encoder.encode(string))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 短标题处理（用户定制语义）
# ---------------------------------------------------------------------------

# 短标题处理策略（与 RAGFlow naive.py:1623 一致）：
#   - 当前块累加 token 超过限制时 flush 输出
#   - 但若当前块是短 Markdown 标题（几乎只有标题无正文），则继续合并，避免标题孤零零成为独立块


def _is_short_header(text: str, max_tokens: int = 50) -> bool:
    """
    判断累积块是否本质上仅为一篇短 Markdown 标题（几乎无正文）。
    用于在 flush 时保护孤立的短标题不被过早截断。

    规则：第 1 行是 # 标题 AND 总 token 数 < max_tokens
    """
    if not text or not text.strip():
        return False
    first_line = text.split("\n", 1)[0]
    if not re.match(r"^#{1,6}\s+", first_line.strip()):
        return False
    return num_tokens_from_string(text) < max_tokens


# ---------------------------------------------------------------------------
# 核心合并逻辑（单 pass token 累积合并，仅合并，不切分）
# ---------------------------------------------------------------------------

@step_log("refine_chunks")
def refine_chunks(
    sections: list[dict[str, Any]],
    chunk_token_num: int,
    overlapped_percent: int = 0,
    short_header_max_tokens: int = 50,
) -> list[dict[str, Any]]:
    """
    基于 token 数做单 pass 累积合并，生成标准化 chunks。

    重要：本函数只做合并，不做切分。超长 section 由 metadata_extract_service 预先处理。

    参考 RAGFlow rag/app/naive.py:1606-1650：
        - 当前块 token 超过 chunk_token_num 时 flush 输出
        - 短标题处理（naive.py 一致）：遇到短 Markdown 标题时不独立成块，继续与下一段合并
        - 重叠：overlapped_percent > 0 时，新块开头取前一块末尾的 N% 字符

    :param sections: 粗切分后的 section 列表，每个 dict 包含 content/title/images/type 等字段
    :param chunk_token_num: 单块最大 token 数
    :param overlapped_percent: 块间重叠百分比
    :param short_header_max_tokens: 短标题判定阈值（与 parser_config.short_header_max_tokens 对齐）
    :return: 合并后的 chunks 列表，每项为 {"content": ..., "images": ..., "file_title": ...}
            type=hyperlinks/table 等 special chunk 保持原样追加到末尾，不参与合并
    """
    if not sections:
        return []

    # 单独保留 special chunk（type=hyperlinks/table 等），不参与合并
    special_chunks: list[dict[str, Any]] = []
    regular_sections: list[dict[str, Any]] = []
    for sec in sections:
        if sec.get("type") in ("hyperlinks", "table"):
            special_chunks.append(sec)
        else:
            regular_sections.append(sec)

    merged = _merge_regular_chunks(
        regular_sections, chunk_token_num, overlapped_percent, short_header_max_tokens
    )
    return merged + special_chunks


def _merge_regular_chunks(
    sections: list[dict[str, Any]],
    chunk_token_num: int,
    overlapped_percent: int,
    short_header_max_tokens: int = 50,
) -> list[dict[str, Any]]:
    """
    对普通文本 chunks 做 token 级合并，仅合并不切分，special chunk 不传入此函数。
    与 ragflow/rag/app/naive.py:1615-1650 逻辑完全一致。
    """
    merged_texts: list[str] = []
    merged_images: list[Any] = []
    current_text = ""
    current_tokens = 0
    current_image: Any = None

    for sec in sections:
        content = sec.get("content", "") or ""
        if not content.strip():
            continue

        sec_tokens = num_tokens_from_string(content)
        raw_sec_image = sec.get("images")
        # 规范化：images 字段可能是 None / str / list[str] / list[list]，
        # 全部展开为 list[str]（与 _enrich_single 的 _resolve_image_urls 逻辑一致）
        if raw_sec_image:
            def _flatten(x):
                if isinstance(x, list):
                    result = []
                    for v in x:
                        result.extend(_flatten(v))
                    return result
                return [x]
            sec_image = [v for v in _flatten(raw_sec_image) if v]
        else:
            sec_image = None

        # 核心合并策略（与 naive.py:1623 一致）：
        # - 当前块累加 token 超过限制时 flush 输出
        # - 但若当前块是短标题（几乎只有标题无正文），则继续合并，避免标题孤零零成为独立块
        if current_text and not _is_short_header(current_text, max_tokens=short_header_max_tokens) \
                and current_tokens + sec_tokens > chunk_token_num:
            merged_texts.append(current_text)
            merged_images.append(current_image)

            # 计算重叠部分（末尾 overlap_len 个字符），保留在下一个块开头
            if overlapped_percent > 0:
                overlap_len = int(len(current_text) * overlapped_percent / 100)
                if overlap_len > 0:
                    current_text = current_text[-overlap_len:]
                    current_tokens = num_tokens_from_string(current_text)
                    current_image = current_image
                else:
                    current_text = ""
                    current_tokens = 0
                    current_image = None
            else:
                current_text = ""
                current_tokens = 0
                current_image = None

        # 累加当前 section 的文本和图片
        if current_text:
            current_text += "\n" + content
        else:
            current_text = content
        current_tokens += sec_tokens

        if sec_image:
            # sec_image 已被规范化为 list[str]，直接去重合并
            img_list = (current_image or []) + sec_image
            seen = []
            for v in img_list:
                if v not in seen:
                    seen.append(v)
            current_image = seen if seen else None

    # 处理最后一个块
    if current_text:
        merged_texts.append(current_text)
        merged_images.append(current_image)

    logger.info(
        f"[_merge_regular_chunks] 输入sections={len(sections)}, "
        f"输出chunks={len(merged_texts)}, "
        f"字符数={[len(t) for t in merged_texts]}"
    )

    # 重建带元数据的 chunks
    result: list[dict[str, Any]] = []
    file_title = sections[0].get("file_title", "") if sections else ""

    for text, imgs in zip(merged_texts, merged_images):
        if not text.strip():
            continue
        result.append({
            "content": text,
            "images": imgs if imgs else None,
            "file_title": file_title,
        })

    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@step_log("merge_and_backup_chunks")
def merge_and_backup_chunks(state: dict) -> dict:
    """
    对 metadata_extract_service 粗切分后的结果做 token 级合并，写入 state["chunks"]。

    参考 RAGFlow rag/app/naive.py:1601-1722：
        1. 从 state["chunks"] 读取粗切分结果（每项为 {"content": ..., "images": ..., "type": ...}）
        2. refine_chunks 做 token 级合并（images 同步合并，special chunk 原样保留）
        3. backup_chunks 持久化到本地 JSON
        4. 结果写回 state["chunks"]（每项含 images，special chunk 保持原有 type 字段）
    """
    chunks = state.get("chunks", [])
    if not chunks:
        logger.info("[merge_and_backup_chunks] chunks 为空，跳过")
        return state

    parser_config = state.get("parser_config", {})
    chunk_token_num = max(1, int(parser_config.get("chunk_char_num", 512)))
    overlapped_percent = int(parser_config.get("overlapped_percent", 0))
    short_header_max_tokens = int(parser_config.get("short_header_max_tokens", 50))

    # 构建 sections（special chunk 携带 type 字段以跳过合并）
    sections: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        sections.append({
            "content": content,
            "images": chunk.get("images") if isinstance(chunk, dict) else None,
            "title": f"chunk-{i + 1}",
            "parent_title": "",
            "file_title": state.get("file_title", ""),
            "type": chunk.get("type") if isinstance(chunk, dict) else None,
        })

    logger.info(
        f"[merge_and_backup_chunks] chunk_token_num={chunk_token_num}, "
        f"overlapped_percent={overlapped_percent}, "
        f"short_header_max_tokens={short_header_max_tokens}, "
        f"sections总量={len(sections)}"
    )

    merged = refine_chunks(
        sections, chunk_token_num, overlapped_percent, short_header_max_tokens
    )

    md_path = state.get("md_path", "")
    backup_chunks(merged, md_path)
    state["chunks"] = merged
    return state


@step_log("backup_chunks")
def backup_chunks(chunks: list[dict], md_path: str) -> None:
    """备份文档切块结果到本地 JSON 文件。"""
    if not chunks:
        return
    save_path = Path(md_path).parent / "chunks.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
