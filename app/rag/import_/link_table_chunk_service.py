"""
超链接提取与表格结构化切块服务
在 node_md_img 与 node_chunk_merge 之间执行，负责：
1. 从 Markdown 内容中提取所有超链接（href + display_text），生成 "hyperlinks" 类型 chunk
2. 将 state["tables"] 中的原始表格结构化为可检索文本，按 batch_size=10 分批生成 "table" 类型 chunk
3. 操作主体为 state["chunks"]，md_content 不再被修改（表格已由 metadata_extract_service 预先提取）

表格内嵌图片存储：表格被预先抽离到 state["tables"] 列表时，丢失了正文中的行号上下文，
因此这里走与正文 chunks 完全相同的存储结构（images: list[str] | None），
由后续 bind_img_id_to_chunks 统一处理为 img_id——保持 chunk 形态一致性。

参考 RAGFlow naive.py:1530-1536 的流程：
    - 超链接提取：markdown → md_to_html → get_hyperlink_urls
    - 表格切块：tokenize_table 风格，按 batch_size=10 分批
"""

import re
from pathlib import Path
from typing import Any

from app.shared.runtime.logger import logger, step_log
from app.rag.import_.metadata_extract_service import extract_table_image_urls


# ---------------------------------------------------------------------------
# 超链接提取（参考 naive.py:968-1006）
# ---------------------------------------------------------------------------

def md_to_html(md_text: str):
    """
    将 Markdown 文本转换为 BeautifulSoup HTML 对象。
    参考 naive.py:968-977。
    """
    try:
        import markdown
        from bs4 import BeautifulSoup
        html = markdown.markdown(md_text)
        return BeautifulSoup(html, "html.parser")
    except ImportError:
        return None


def get_hyperlink_urls(soup) -> set:
    """
    从 BeautifulSoup HTML 对象中提取所有超链接的 href 属性值并去重。
    参考 naive.py:994-1006。
    """
    if not soup:
        return set()
    return set(a.get("href") for a in soup.find_all("a") if a.get("href"))


@step_log("build_hyperlinks_chunk")
def build_hyperlinks_chunk(md_content: str, file_title: str) -> dict:
    """
    从 Markdown 内容中提取所有超链接，生成 "hyperlinks" 类型 chunk。

    流程（参考 naive.py:1530-1535）：
        1. 遍历每个 section_text，markdown.markdown 转 HTML
        2. BeautifulSoup.find_all("a") 提取 href
        3. 合并去重

    :param md_content: Markdown 原始文本
    :param file_title: 文档标题
    :return: 超链接 chunk，格式为 {"type": "hyperlinks", "content": str}
    """
    if not md_content:
        return {"type": "hyperlinks", "content": ""}

    all_urls: set = set()
    for section_text in md_content.split("\n"):
        if not section_text.strip():
            continue
        soup = md_to_html(section_text)
        urls = get_hyperlink_urls(soup)
        all_urls.update(urls)

    # 过滤无效 URL
    valid_urls = sorted(
        u for u in all_urls
        if u and not u.startswith("#") and not u.startswith("mailto:") and not u.startswith("data:")
    )

    content = "\n".join(valid_urls) if valid_urls else ""
    logger.info(f"[build_hyperlinks_chunk] 提取超链接 {len(valid_urls)} 个")

    return {
        "type": "hyperlinks",
        "content": content,
        "file_title": file_title,
    }


# ---------------------------------------------------------------------------
# 表格结构化（参考 rag/nlp/__init__.py:505-542 tokenize_table）
# ---------------------------------------------------------------------------

@step_log("parse_single_table")
def parse_single_table(table_raw: str) -> dict[str, Any] | None:
    """
    将原始 Markdown 表格文本解析为结构化字典。

    支持三种格式：
        1. 标准有边框表格：| col1 | col2 | ...
        2. 无边框纯文本表格：col1 | col2 | ...
        3. HTML 表格：<table>...</table>

    同时提取表格内嵌的图片 URL（Markdown ![](url) / HTML <img src=url>），
    存入 image_urls 字段，供 tokenize_table_chunks 写入 chunk.images 字段，
    与正文 chunks 保持完全相同的图片存储结构（由 bind_img_id_to_chunks 后续处理）。

    :param table_raw: 原始表格文本
    :return: {"headers": [...], "rows": [[...], ...], "table_type": str, "image_urls": [str, ...]}，
             解析失败返回 None
    """
    table_raw = table_raw.strip()
    if not table_raw:
        return None

    # 提取表格内嵌图片 URL（在所有解析路径之前统一处理，与解析逻辑解耦）
    image_urls = extract_table_image_urls(table_raw)

    if table_raw.lower().startswith("<table"):
        parsed = _parse_html_table(table_raw)
        if parsed is None:
            return None
        parsed["image_urls"] = image_urls
        return parsed

    lines = [l for l in table_raw.splitlines() if l.strip() and "|" in l]
    if len(lines) < 2:
        return None

    has_border = any(re.search(r"\|\s*[:\-]+\s*\|", line) for line in lines)

    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip("| ").split("|")]
        if not all(re.match(r"^\s*[:\-]+\s*$", c) for c in cells):
            rows.append(cells)

    if len(rows) < 2:
        return None

    return {
        "headers": rows[0],
        "rows": rows[1:],
        "table_type": "bordered" if has_border else "plain",
        "image_urls": image_urls,
    }


def _parse_html_table(html: str) -> dict[str, Any] | None:
    """内部函数：解析 HTML 表格。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        headers: list[str] = []
        rows: list[list[str]] = []

        thead = soup.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
        else:
            first_row = soup.find("tr")
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(["th", "td"])]

        tbody = soup.find("tbody") or soup
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        if len(rows) < 2:
            return None

        return {"headers": headers, "rows": rows, "table_type": "html"}
    except ImportError:
        logger.warning("[_parse_html_table] BeautifulSoup 未安装，HTML 表格解析跳过")
        return None


@step_log("tokenize_table_chunks")
def tokenize_table_chunks(
    tables: list[str],
    file_title: str,
    batch_size: int = 10,
    row_delimiter: str = "； ",
) -> list[dict]:
    """
    将原始表格列表转换为可检索的 "table" 类型 chunk 列表。

    流程（参考 rag/nlp/__init__.py:505-542 tokenize_table）：
        1. 逐个解析原始表格文本 → 结构化字典（含 image_urls）
        2. 每 batch_size 行合并为一个 chunk（用 row_delimiter 连接每行："列1: 值1; 列2: 值2"）
        3. 每个 chunk 标记 doc_type="table"，并附加 images 字段
        4. 表格内嵌图片：解析阶段提取的 image_urls 整体关联到该表格所有 batch chunks
           （图片归属到表格而非某行，符合 batch 内行合并的语义）

    与正文 chunks 的存储结构完全一致：
        {"content": str, "images": list[str]|None, "type": "table", "file_title": str}
        → 后续 bind_img_id_to_chunks 可无差别地为所有 chunk 绑定 img_id。

    :param tables: state["tables"]，即 extract_tables_and_remainder 返回的原始表格列表
    :param file_title: 文档标题
    :param batch_size: 每批合并的行数，默认 10（与 RAGFlow tokenize_table 保持一致）
    :param row_delimiter: 表格行间分隔符，默认 "； "（与 RAGFlow tokenize_table 保持一致）
    :return: 表格 chunk 列表
    """
    if not tables:
        return []

    chunks: list[dict] = []
    for idx, table_raw in enumerate(tables):
        parsed = parse_single_table(table_raw)
        if not parsed:
            logger.warning(f"[tokenize_table_chunks] 表格 {idx + 1} 解析失败，跳过")
            continue

        headers = parsed.get("headers", [])
        rows = parsed.get("rows", [])
        image_urls = parsed.get("image_urls") or []
        if not headers or not rows:
            logger.warning(f"[tokenize_table_chunks] 表格 {idx + 1} 无有效数据，跳过")
            continue

        # 将每行转为可拼接文本，参考 rag/nlp/__init__.py:530-533
        # vision model 解析结果格式：rows = [["张三","25"],["李四","30"]]
        # 直接拼接每行数据，行间用 row_delimiter 分隔
        # 示例：batch_size=2 → "张三；25；李四；30"
        row_texts: list[str] = []
        for row in rows:
            filtered = [c.strip() for c in row if c.strip()]
            if filtered:
                row_texts.append("；".join(filtered))

        if not row_texts:
            continue

        # 按 batch_size 分批合并（参考 rag/nlp/__init__.py:530-533）
        # RAGFlow 格式：直接拼接列值，行间用 row_delimiter 分隔
        for i in range(0, len(row_texts), batch_size):
            batch = row_texts[i : i + batch_size]
            content = row_delimiter.join(batch)
            chunks.append({
                "type": "table",
                "content": content,
                "file_title": file_title,
                # 与正文 chunks 同构：表格内图片以 URL 列表存储，bind_img_id_to_chunks
                # 会取第一张图片的 img_id 绑定到 chunk
                "images": image_urls if image_urls else None,
            })

    logger.info(f"[tokenize_table_chunks] 生成 {len(chunks)} 个 table chunk")
    return chunks


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@step_log("process_hyperlinks_and_tables")
def process_hyperlinks_and_tables(state: dict) -> dict:
    """
    超链接提取与表格结构化切块服务总入口。

    读取 state["md_content"] 和 state["tables"]，执行：
        1. 超链接提取 → build_hyperlinks_chunk → 追加到 state["chunks"]
        2. 表格结构化切块 → tokenize_table_chunks → 追加到 state["chunks"]
        3. 表格已被 metadata_extract_service 预先提取，chunks 内容不含表格，无需再修改 md_content

    关于图片字段（与正文 chunks 同构）：
        - hyperlinks chunk 不含图片，无 images 字段
        - table chunk 携带 images 字段（内嵌图片 URL 列表），
          后续 node_image_upload → bind_img_id_to_chunks 会无差别处理

    :param state: LangGraph 流程状态字典
    :return: 更新后的 state
    """
    md_content = state.get("md_content", "")
    md_path = state.get("md_path", "")
    tables: list = state.get("tables", [])
    file_title = state.get("file_title", "")

    # 从 md_path 补充读取 md_content（用于超链接提取）
    if not md_content and md_path:
        try:
            md_path_obj = Path(md_path)
            if md_path_obj.exists():
                md_content = md_path_obj.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[process_hyperlinks_and_tables] 读取 md_path 失败: {e}")

    if not md_content:
        logger.info("[process_hyperlinks_and_tables] md_content 为空，仅生成空的 hyperlinks chunk")
        hl_chunk = {
            "type": "hyperlinks",
            "content": "",
            "file_title": file_title,
        }
    else:
        # 1. 超链接提取 → 生成 "hyperlinks" chunk
        hl_chunk = build_hyperlinks_chunk(md_content, file_title)

    # 2. 表格结构化切块 → 生成 "table" 类型 chunks（自动绑定表格内嵌图片）
    #    batch_size / row_delimiter 从 parser_config 读取
    parser_config = state.get("parser_config", {}) or {}
    table_batch_size = int(parser_config.get("table_batch_size", 10))
    table_row_delimiter = str(parser_config.get("table_row_delimiter", "； "))
    tbl_chunks = tokenize_table_chunks(
        tables, file_title,
        batch_size=table_batch_size,
        row_delimiter=table_row_delimiter,
    )

    # 3. 追加到 state["chunks"]（操作主体为 chunks，md_content 不再被修改）
    existing_chunks = state.get("chunks", [])
    new_chunks = list(existing_chunks) + [hl_chunk] + tbl_chunks
    state["chunks"] = new_chunks

    # 统计：表格 chunk 中含图片的数量（用于诊断表格内嵌图片是否被正确捕获）
    tbl_with_images = sum(1 for c in tbl_chunks if c.get("images"))
    tbl_image_count = sum(len(c.get("images") or []) for c in tbl_chunks)
    logger.info(
        f"[process_hyperlinks_and_tables] 完成，"
        f"新增 hyperlinks chunk=1，table chunks={len(tbl_chunks)}"
        f"（含图片的 table chunk={tbl_with_images}，关联图片总数={tbl_image_count}），"
        f"chunks总量={len(new_chunks)}"
    )
    return state
