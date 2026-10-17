"""
全局元数据提取服务
从 PDF/MD 文件中提取全局元数据（标题、作者、关键词、摘要、页数、创建时间等）
以及粗切分结果（element_sections）。
"""

import re
from pathlib import Path
from typing import Any

from app.shared.runtime.logger import logger, step_log
from app.process.import_.agent.state import ImportGraphState


def find_codec(blob: bytes) -> str:
    """
    根据二进制数据自动检测字符编码。
    优先使用 chardet 置信度判断，ascii 统一归为 utf-8，
    否则逐一尝试常见编码表作为兜底。
    """
    import chardet

    detected = chardet.detect(blob[:1024])
    if detected["confidence"] > 0.5:
        if detected["encoding"] == "ascii":
            return "utf-8"

    all_codecs = ["utf-8", "gbk", "gb2312", "gb18030", "big5", "latin-1"]
    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


@step_log("load_document_text")
def load_document_text(state: ImportGraphState) -> str:
    """
    第一步：根据输入数据的来源决定如何读取文本内容。
    优先级：
        1. state["md_content"] 已有内容 → 直接返回
        2. state["md_path"] 文件路径存在 → 从文件加载（二进制 + find_codec 自动检测编码）
        3. 均不存在 → 抛出异常
    """
    md_content = state.get("md_content", "")
    md_path = state.get("md_path")

    if md_content:
        return md_content

    if md_path:
        md_path_obj = Path(md_path)
        if md_path_obj.exists() and md_path_obj.is_file():
            binary = md_path_obj.read_bytes()
            encoding = find_codec(binary)
            return binary.decode(encoding, errors="ignore")

    raise ValueError("无法获取文档内容：md_content 和 md_path 均不存在")


def _replace_tables_with_rendered_html(
    working_text: str,
    tables: list,
    pattern: re.Pattern,
    separate_tables: bool,
) -> str:
    """
    局部辅助函数：遍历正则匹配结果，提取表格原文并存入 tables，
    根据 separate_tables 决定是删除表格还是替换为 HTML 渲染结果。
    """
    new_text = ""
    last_end = 0
    for match in pattern.finditer(working_text):
        raw_table = match.group()
        tables.append(raw_table)
        if separate_tables:
            new_text += working_text[last_end : match.start()] + "\n\n"
        else:
            from markdown import markdown
            html_table = markdown(raw_table, extensions=["markdown.extensions.tables"])
            new_text += working_text[last_end : match.start()] + html_table + "\n\n"
        last_end = match.end()
    new_text += working_text[last_end:]
    return new_text


@step_log("extract_tables_and_remainder")
def extract_tables_and_remainder(
    markdown_text: str, separate_tables: bool = True
) -> tuple[str, list]:
    """
    第二步：对已读取的 Markdown 文本进行结构抽取，首先提取表格内容。
    参考 ragflow/deepdoc/parser/markdown_parser.py:34-160。

    处理流程：
        1. 初始化 tables 列表和 working_text 副本
        2. 识别 Markdown 标准表格（有边框 |---| 形式）
        3. 识别无边框 Markdown 表格
        4. 简化 HTML 表格标签（如 <table class="xxx"> → <table>）
        5. 识别并提取 HTML 表格（支持 body/html 包装标签）

    :param markdown_text: 原始 Markdown 文本
    :param separate_tables: 是否将表格从文本中分离出来（True=表格移入 tables，正文留空）
    :return: (去除/替换表格后的文本, 提取到的表格列表)
    """
    tables: list = []
    working_text = markdown_text

    # 标准 Markdown 表格（有边框 |---| 形式）
    if "|" in markdown_text:
        border_table_pattern = re.compile(
            r"""
            (?:\n|^)
            (?:\|.*?\|.*?\|.*?\n)
            (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
            (?:\|.*?\|.*?\|.*?\n)+
            """,
            re.VERBOSE,
        )
        working_text = _replace_tables_with_rendered_html(
            working_text, tables, border_table_pattern, separate_tables
        )

        # 无边框 Markdown 表格
        no_border_table_pattern = re.compile(
            r"""
            (?:\n|^)
            (?:\S.*?\|.*?\n)
            (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
            (?:\S.*?\|.*?\n)+
            """,
            re.VERBOSE,
        )
        working_text = _replace_tables_with_rendered_html(
            working_text, tables, no_border_table_pattern, separate_tables
        )

    # 简化 HTML 表格标签（去除 class/id/style 等属性）
    TAGS = ["table", "td", "tr", "th", "tbody", "thead", "div"]

    def _replace_tag(m: re.Match) -> str:
        tag_name = re.match(r"<(\w+)", m.group()).group(1)
        return "<{}>".format(tag_name)

    tag_pattern = re.compile(rf"<(?:{'|'.join(TAGS)})[^>]*>", re.IGNORECASE)
    working_text = re.sub(tag_pattern, _replace_tag, working_text)

    # HTML 表格提取（三种常见包装形式）
    if "<table>" in working_text.lower():
        html_table_pattern = re.compile(
            r"""
        (?:\n|^)
        \s*
        (?:
            # case1: <html><body><table>...</table></body></html>
            (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
            |
            # case2: <body><table>...</table></body>
            (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
            |
            # case3: only <table>...</table>
            (?:<table[^>]*>.*?</table>)
        )
        \s*
        (?=\n|$)
        """,
            re.VERBOSE | re.DOTALL | re.IGNORECASE,
        )

        def _replace_html_tables() -> None:
            nonlocal working_text
            new_text = ""
            last_end = 0
            for match in html_table_pattern.finditer(working_text):
                raw_table = match.group()
                tables.append(raw_table)
                if separate_tables:
                    new_text += working_text[last_end : match.start()] + "\n\n"
                else:
                    new_text += working_text[last_end : match.start()] + raw_table + "\n\n"
                last_end = match.end()
            new_text += working_text[last_end:]
            working_text = new_text

        _replace_html_tables()

    return working_text, tables


@step_log("extract_image_urls_with_lines")
def extract_image_urls_with_lines(text: str) -> list[dict]:
    """
    第三步：从 Markdown 原文中提取图片 URL 及其所在行号。
    参考 ragflow/rag/app/naive.py:1009-1058。

    用正则表达式按行扫描 Markdown 图片语法和 HTML img 标签，
    用 BeautifulSoup 兜底复杂 HTML，收集图片 URL 与行号的映射。

    :param text: Markdown 原文
    :return: 图片引用列表，每项为 {"url": str, "line": int}
    """
    md_img_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
    html_img_re = re.compile(r'src=["\']([^"\'>\s]+)', re.IGNORECASE)
    urls = []
    seen = set()
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        for url in md_img_re.findall(line):
            if (url, idx) not in seen:
                urls.append({"url": url, "line": idx})
                seen.add((url, idx))
        for url in html_img_re.findall(line):
            if (url, idx) not in seen:
                urls.append({"url": url, "line": idx})
                seen.add((url, idx))

    # BeautifulSoup 兜底：处理复杂 HTML 中的 <img> 标签
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        newline_offsets = [m.start() for m in re.finditer(r"\n", text)] + [len(text)]
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src")
            if not src:
                continue

            tag_str = str(img_tag)
            pos = text.find(tag_str)
            if pos == -1:
                pos = max(text.find(src), 0)
            line_no = 0
            for i, off in enumerate(newline_offsets):
                if pos <= off:
                    line_no = i
                    break
            if (src, line_no) not in seen:
                urls.append({"url": src, "line": line_no})
                seen.add((src, line_no))
    except ImportError:
        pass

    return urls


@step_log("extract_table_image_urls")
def extract_table_image_urls(table_raw: str) -> list[str]:
    """
    从单张原始表格文本中提取所有图片 URL（去重，保持首次出现顺序）。

    表格与正文走的是不同的存储路径：
        - 正文 chunks 在 metadata_extract 阶段就以 line 范围绑定 image_refs，
          因此可保留 (url, line) 行级映射供后置关联使用。
        - 表格 chunks 由 link_table_chunk_service 独立从 state["tables"] 列表中
          生成，失去了原始行号上下文（且批量生成时会跨多行合并），因此只保留
          纯 URL 列表即可——后续 bind_img_id_to_chunks 同样以首张图片绑定到
          chunk，与正文 chunk 的存储结构（images: list[str] | None）完全一致。

    支持的图片语法：
        1. Markdown：![alt](url)
        2. HTML：<img src="url" ...>

    :param table_raw: 单张表格的原始文本
    :return: 图片 URL 列表（去重）；无图则返回空列表
    """
    if not table_raw or not table_raw.strip():
        return []

    md_img_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
    html_img_re = re.compile(r'src=["\']([^"\'>\s]+)', re.IGNORECASE)

    urls: list[str] = []
    seen: set[str] = set()
    for url in md_img_re.findall(table_raw):
        if url not in seen:
            urls.append(url)
            seen.add(url)
    for url in html_img_re.findall(table_raw):
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


@step_log("extract_document_metadata")
def extract_document_metadata(state: ImportGraphState) -> ImportGraphState:
    """
    全局元数据提取服务：
    1. 读取 md_content 或 md_path，获取文档内容
    2. 提取表格内容，分离正文
    3. 提取图片 URL（含行号）
    4. 粗切分 + 细切分：通过 parser_config["delimiter"] 区分，均由 extract_elements 统一处理
       - delimiter=None：粗切分，按 Markdown 块元素切分（标题、代码块、列表、引用、段落）
       - delimiter 有值：细切分，按子分隔符进一步切分（参考 naive.py:1133）
    5. 将切分结果写入 state["chunks"]（每项为 {"content": str, "images": list[str]|None}）

    :param state: LangGraph 流程状态字典
    :return: 更新后的 state
    """
    # 步骤1：获取文档文本
    txt = load_document_text(state)

    # 步骤2：提取表格并分离正文
    # separate_tables 从 parser_config["separate_tables"] 读取，默认 True
    # 注：parser_config 需在步骤 2 之前从 state 取出（Python 词法作用域规则：
    #     函数体内一旦对变量赋值即被视为局部变量，所以必须先赋值再使用）
    parser_config = state.get("parser_config", {}) or {}
    separate_tables = bool(parser_config.get("separate_tables", True))
    remainder, tables = extract_tables_and_remainder(f"{txt}\n", separate_tables=separate_tables)
    state["tables"] = tables

    # 步骤3：提取图片 URL（含行号）
    # 重要：基于 remainder 而非 txt，因为表格移除会导致行号偏移
    # 图片关联使用 remainder 的行号基准，与 MarkdownElementExtractor 保持一致
    image_refs = extract_image_urls_with_lines(remainder)
    state["image_refs"] = image_refs
    logger.info(f"[extract_document_metadata] image_refs 数量: {len(image_refs)}")
    if image_refs:
        logger.info(f"[extract_document_metadata] image_refs 前5个: {image_refs[:5]}")

    # 步骤4：粗切分 + 细切分——通过 delimiter 参数区分行为，均由 extract_elements 统一处理
    #   - delimiter=None：按 Markdown 块元素识别（标题、代码块、列表、引用、段落），对应粗切分
    #   - delimiter 有值：按 delimiter 子分隔符正则切分全文本，对应细切分
    # 参考 naive.py:1131-1133，__call__ 中只调用一次 extract_elements(delimiter, include_meta=True)
    # 注：parser_config 已在步骤 2 开头从 state 取出，此处直接复用
    delimiter = parser_config.get("delimiter")
    # chunk_char_num：从 parser_config 读取单块最大字符数（与 state.py 默认 512 对齐）
    # 当前在 extract_elements 阶段不应用此限制（ragflow 的粗切分也不限制，超长块由
    # 后续 chunk_merge_service.refine_chunks 阶段用 chunk_char_num 控制 token 累积）。
    # 字段保留在主入口读取仅为"暴露给流程审计/单元测试"使用。
    chunk_char_num = int(parser_config.get("chunk_char_num", 512))
    logger.debug(
        f"[extract_document_metadata] parser_config 关键参数: "
        f"chunk_char_num={chunk_char_num}, delimiter={delimiter!r}"
    )
    # 若想保留表格注释掉使用remainder，使用txt
    # extractor = MarkdownElementExtractor(txt)
    extractor = MarkdownElementExtractor(remainder)
    sections = extractor.extract_elements(delimiter=delimiter, include_meta=True)

    # 步骤5：将细切分结果写入 state["chunks"]，images 作为内置字段
    # sections 已是 {"content": ..., "start_line": ..., "end_line": ...} 格式
    chunks = []
    for element in sections:
        # 关联图片：收集落在该块行号范围内的所有图片 URL
        start_line = element["start_line"]
        end_line = element["end_line"]
        urls_in_chunk = [
            ref["url"] for ref in image_refs if start_line <= ref["line"] <= end_line
        ]
        chunks.append({
            "content": element["content"],
            "images": urls_in_chunk if urls_in_chunk else None,
        })

    state["chunks"] = chunks

    # TODO: 基于 remainder 文本和 tables 提取元数据字段
    state["doc_metadata"] = {}

    return state


# ---------------------------------------------------------------------------
# MarkdownElementExtractor — 参考 ragflow/deepdoc/parser/markdown_parser.py:163
# 用于粗切分：从 Markdown 原文按块元素（标题、代码块、列表、引用、段落）切分
# ---------------------------------------------------------------------------


class MarkdownElementExtractor:
    """
    从 Markdown 文本中提取结构化元素块（标题、代码块、列表块、引用块、文本块）。
    参考 ragflow/deepdoc/parser/markdown_parser.py:163-423。
    """

    def __init__(self, markdown_content: str):
        self.markdown_content = markdown_content
        self.lines = markdown_content.split("\n")

    def get_delimiters(self, delimiters: str) -> str:
        """
        将 delimiters 参数中的反引号包裹的子串提取为正则转义后的候选分隔符列表，
        按长度降序排列后用 | 连接，用于精确匹配切分边界。
        """
        toks = re.findall(r"`([^`]+)`", delimiters)
        toks = sorted(set(toks), key=lambda x: -len(x))
        return "|".join(re.escape(t) for t in toks if t)

    @staticmethod
    def _strip_image_refs(content: str) -> str:
        """
        清除 content 中的 Markdown 图片语法和 HTML img 标签，避免 URL 文本混入向量。
        仅移除图片标记本身，保留其所在行的其余文本内容。
        """
        # Markdown 图片：![alt](url) 和 ![alt][ref] 引用式
        content = re.sub(r"!\[[^\]]*\]\([^)\s]+\)", "", content)
        content = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", "", content)
        # HTML <img ... src="..." ...> 和 <img ... src='...' ...>
        content = re.sub(r"<img[^>]+src=[\"'][^\"']+[\"'][^>]*>", "", content, flags=re.IGNORECASE)
        return content

    def extract_elements(
        self, delimiter: str | None = None, include_meta: bool = False
    ) -> list[dict[str, Any] | str]:
        """
        提取 Markdown 文档中的结构化元素块（粗切分 + 细切分共用此方法）。

        :param delimiter: 分隔符字符串。
                         - None 或空字符串：按 Markdown 块元素自动识别（粗切分），
                           识别标题（#）、代码块（```）、列表（-/*/数字.）、引用（>）、段落
                         - 有值：按 delimiter 中声明的子分隔符正则切分全文本（细切分），
                           参考 naive.py:1133，如 "\n!?。；！？" 按句子边界切分
        :param include_meta: True 返回 dict（含 type/content/start_line/end_line），
                             False 只返回 content 字符串
        :return: 元素块列表
        """
        sections = []
        i = 0

        if delimiter:
            dels = self.get_delimiters(delimiter)
            if dels:
                text = "\n".join(self.lines)
                if include_meta:
                    pattern = re.compile(dels)
                    last_end = 0
                    for m in pattern.finditer(text):
                        part = text[last_end : m.start()]
                        if part and part.strip():
                            sections.append(
                                {
                                    "content": self._strip_image_refs(part.strip()),
                                    "start_line": text.count("\n", 0, last_end),
                                    "end_line": text.count("\n", 0, m.start()),
                                }
                            )
                        last_end = m.end()
                    part = text[last_end:]
                    if part and part.strip():
                        sections.append(
                            {
                                "content": self._strip_image_refs(part.strip()),
                                "start_line": text.count("\n", 0, last_end),
                                "end_line": text.count("\n", 0, len(text)),
                            }
                        )
                else:
                    parts = re.split(dels, text)
                    sections = [self._strip_image_refs(p.strip()) for p in parts if p and p.strip()]
                return sections

        while i < len(self.lines):
            line = self.lines[i]

            if re.match(r"^#{1,6}\s+.*$", line):
                element = self._extract_header(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith("```"):
                element = self._extract_code_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif re.match(r"^\s*[-*+]\s+.*$", line) or re.match(
                r"^\s*\d+\.\s+.*$", line
            ):
                element = self._extract_list_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith(">"):
                element = self._extract_blockquote(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip():
                element = self._extract_text_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            else:
                i += 1

        # 过滤空块：但包含图片引用的行即使 strip 后为空也保留（确保图片关联生效）
        if include_meta:
            _has_image_ref = re.compile(r"!\[[^\]]*\]\([^)\s]+\)")
            sections = [
                s for s in sections
                if s["content"].strip()
                or _has_image_ref.search("\n".join(self.lines[s["start_line"] : s["end_line"] + 1]))
            ]
        else:
            sections = [s for s in sections if s.strip()]
        return sections

    def _extract_header(self, start: int) -> dict[str, Any]:
        content = self.lines[start]
        return {
            "type": "header",
            "content": self._strip_image_refs(content),
            "start_line": start,
            "end_line": start,
        }

    def _extract_code_block(self, start: int) -> dict[str, Any]:
        end_pos = start
        for i in range(start + 1, len(self.lines)):
            end_pos = i
            if self.lines[i].strip().startswith("```"):
                break
        return {
            "type": "code_block",
            "content": self._strip_image_refs("\n".join(self.lines[start : end_pos + 1])),
            "start_line": start,
            "end_line": end_pos,
        }

    def _extract_list_block(self, start: int) -> dict[str, Any]:
        end_pos = start
        i = start
        while i < len(self.lines):
            line = self.lines[i].strip()
            if not line:
                break
            if not (
                re.match(r"^\s*[-*+]\s+", line)
                or re.match(r"^\s*\d+\.\s+", line)
                or (i > start and re.match(r"^\s{2,}[-*+]\s+", line))
                or (i > start and re.match(r"^\s{2,}\d+\.\s+", line))
                or (i > start and re.match(r"^\s+\w+", line))
            ):
                break
            end_pos = i
            i += 1
        return {
            "type": "list_block",
            "content": self._strip_image_refs("\n".join(self.lines[start : end_pos + 1])),
            "start_line": start,
            "end_line": end_pos,
        }

    def _extract_blockquote(self, start: int) -> dict[str, Any]:
        end_pos = start
        i = start
        while i < len(self.lines):
            if not self.lines[i].strip().startswith(">"):
                break
            end_pos = i
            i += 1
        return {
            "type": "blockquote",
            "content": self._strip_image_refs("\n".join(self.lines[start : end_pos + 1])),
            "start_line": start,
            "end_line": end_pos,
        }

    def _extract_text_block(self, start: int) -> dict[str, Any]:
        end_pos = start
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip():
                break
            if re.match(r"^#{1,6}\s+", line):
                break
            if line.strip().startswith("```"):
                break
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
                break
            if line.strip().startswith(">"):
                break
            end_pos = i
            i += 1
        return {
            "type": "text_block",
            "content": self._strip_image_refs("\n".join(self.lines[start : end_pos + 1])),
            "start_line": start,
            "end_line": end_pos,
        }
