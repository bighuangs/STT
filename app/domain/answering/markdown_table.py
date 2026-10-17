"""严格、可重放的 Markdown 与窄 OCR 表格块解析。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata

from app.domain.answering.witness import PACKED_TRUNCATION_MARKER


_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_OCR_HEADER_RE = re.compile(r"^[ \t]*表头[ \t]*[:：][ \t]*(?P<body>.*?)[ \t]*$")
_MERGED_CELL_RE = re.compile(r"(?:rowspan|colspan)\s*=", re.IGNORECASE)
_DITTO_MARKERS = frozenset(
    {
        '"',
        "''",
        "^",
        "↑",
        "〃",
        "同上",
        "同前",
        "同前项",
        "同左",
        "见上",
        "如上",
        "same",
        "sameasabove",
        "ditto",
    }
)


class MarkdownTableParseMode(StrEnum):
    """表格在原始文本中的受支持物理布局。"""

    STANDARD = "standard"
    OCR_HEADER_DATA_BEFORE_SEPARATOR = "ocr_header_data_before_separator"


class MarkdownTableParseError(ValueError):
    """已经识别的表格块违反不可恢复的结构不变量。"""


@dataclass(frozen=True, slots=True)
class PhysicalLine:
    """不包含换行符的原始物理行及其半开字符区间。"""

    index: int
    start: int
    end: int
    text: str

    @property
    def span(self) -> tuple[int, int]:
        """返回该物理行在完整文本中的半开字符区间。"""

        return self.start, self.end


@dataclass(frozen=True, slots=True)
class MarkdownTableCell:
    """去除外侧空白后的单元格原文及其精确字符区间。"""

    start: int
    end: int
    text: str
    normalized_text: str

    @property
    def span(self) -> tuple[int, int]:
        """返回单元格在完整文本中的半开字符区间。"""

        return self.start, self.end


@dataclass(frozen=True, slots=True)
class MarkdownTableRow:
    """一个严格简单表格行；OCR 表头允许带显式 ``表头：`` 前缀。"""

    line: PhysicalLine
    cells: tuple[MarkdownTableCell, ...]
    is_separator: bool

    @property
    def start(self) -> int:
        """返回完整物理行的起始字符坐标。"""

        return self.line.start

    @property
    def end(self) -> int:
        """返回完整物理行的结束字符坐标。"""

        return self.line.end

    @property
    def values(self) -> tuple[str, ...]:
        """返回已去除外侧空白、但未改写内容的单元格值。"""

        return tuple(cell.text for cell in self.cells)


@dataclass(frozen=True, slots=True)
class MarkdownTableBlock:
    """从完整 authority 文本中识别出的不可变严格表格块。"""

    start: int
    end: int
    header: MarkdownTableRow
    separator: MarkdownTableRow
    data_rows: tuple[MarkdownTableRow, ...]
    column_count: int
    parse_mode: MarkdownTableParseMode

    @property
    def span(self) -> tuple[int, int]:
        """返回表格块从显式表头到末数据行的半开字符区间。"""

        return self.start, self.end

    @property
    def line_indexes(self) -> tuple[int, ...]:
        """按原始物理顺序返回该块占用的行号。"""

        rows = (self.header, self.separator, *self.data_rows)
        return tuple(sorted(row.line.index for row in rows))


def _normalize_cell(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _physical_lines(value: str) -> tuple[PhysicalLine, ...]:
    lines: list[PhysicalLine] = []
    start = 0
    for delimiter in re.finditer(r"\r\n|\r|\n", value):
        lines.append(
            PhysicalLine(
                index=len(lines),
                start=start,
                end=delimiter.start(),
                text=value[start : delimiter.start()],
            )
        )
        start = delimiter.end()
    if start < len(value):
        lines.append(
            PhysicalLine(
                index=len(lines),
                start=start,
                end=len(value),
                text=value[start:],
            )
        )
    return tuple(lines)


def _trimmed_bounds(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def _parse_cells(
    line: PhysicalLine,
    *,
    body_start: int = 0,
    body_end: int | None = None,
    allow_empty_cells: bool = False,
) -> tuple[MarkdownTableCell, ...] | None:
    """解析一段简单 pipe 行，同时保留每个 cell 的原始坐标。"""

    resolved_end = len(line.text) if body_end is None else body_end
    start, end = _trimmed_bounds(line.text, body_start, resolved_end)
    if start >= end:
        return None
    body = line.text[start:end]
    if "|" not in body or "\\|" in body or "`" in body:
        return None
    if line.text[start] == "|":
        start += 1
    if end > start and line.text[end - 1] == "|":
        end -= 1
    start, end = _trimmed_bounds(line.text, start, end)
    if start >= end:
        return None

    cells: list[MarkdownTableCell] = []
    cursor = start
    while True:
        delimiter = line.text.find("|", cursor, end)
        raw_end = delimiter if delimiter >= 0 else end
        cell_start, cell_end = _trimmed_bounds(line.text, cursor, raw_end)
        if cell_start >= cell_end and not allow_empty_cells:
            return None
        cell_text = line.text[cell_start:cell_end]
        cells.append(
            MarkdownTableCell(
                start=line.start + cell_start,
                end=line.start + cell_end,
                text=cell_text,
                normalized_text=_normalize_cell(cell_text),
            )
        )
        if delimiter < 0:
            break
        cursor = delimiter + 1
    return tuple(cells) if len(cells) >= 2 else None


def _row_from_cells(
    line: PhysicalLine,
    cells: tuple[MarkdownTableCell, ...],
) -> MarkdownTableRow:
    return MarkdownTableRow(
        line=line,
        cells=cells,
        is_separator=all(_SEPARATOR_CELL_RE.fullmatch(cell.text) for cell in cells),
    )


def _parse_standard_row(
    line: PhysicalLine,
    *,
    allow_empty_cells: bool = False,
) -> MarkdownTableRow | None:
    cells = _parse_cells(line, allow_empty_cells=allow_empty_cells)
    return None if cells is None else _row_from_cells(line, cells)


def _parse_ocr_header(line: PhysicalLine) -> MarkdownTableRow | None:
    match = _OCR_HEADER_RE.fullmatch(line.text)
    if match is None:
        return None
    cells = _parse_cells(line, body_start=match.start("body"), body_end=match.end("body"))
    if cells is None:
        return None
    row = _row_from_cells(line, cells)
    if row.is_separator:
        return None
    normalized = tuple(cell.normalized_text for cell in cells)
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        return None
    return row


def _has_ocr_header_prefix(line: PhysicalLine) -> bool:
    return _OCR_HEADER_RE.fullmatch(line.text) is not None


def _is_table_like(line: PhysicalLine) -> bool:
    return "|" in line.text


def _is_strict_data_row(row: MarkdownTableRow, *, column_count: int) -> bool:
    if row.is_separator or len(row.cells) != column_count:
        return False
    if not any(cell.normalized_text for cell in row.cells):
        return False
    for cell in row.cells:
        if cell.normalized_text in _DITTO_MARKERS or _MERGED_CELL_RE.search(cell.text):
            return False
    return True


def _skip_table_like_fragment(lines: tuple[PhysicalLine, ...], start: int) -> int:
    cursor = start
    while cursor < len(lines) and _is_table_like(lines[cursor]):
        cursor += 1
    return cursor


def _failed_ocr_next_cursor(
    lines: tuple[PhysicalLine, ...],
    *,
    cursor: int,
) -> int:
    """跳过坏 OCR 片段，但保留紧邻的独立表头供下一轮扫描。"""

    next_index = cursor + 1
    if next_index >= len(lines):
        return next_index
    next_line = lines[next_index]
    if _has_ocr_header_prefix(next_line):
        return next_index
    stripped = next_line.text.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        return next_index
    return _skip_table_like_fragment(lines, next_index)


def _raise_if_invalid_continuation(
    *,
    line: PhysicalLine,
    row: MarkdownTableRow | None,
    column_count: int,
) -> None:
    if row is not None and row.is_separator:
        raise MarkdownTableParseError(
            f"accepted table has a secondary separator at physical line {line.index}"
        )
    if row is not None and len(row.cells) != column_count:
        raise MarkdownTableParseError(
            f"accepted table has column drift at physical line {line.index}"
        )
    if _is_table_like(line):
        raise MarkdownTableParseError(
            f"accepted table has an empty, ditto, merged, or invalid row at physical line "
            f"{line.index}"
        )


def _scan_standard_candidate(
    lines: tuple[PhysicalLine, ...],
    cursor: int,
) -> tuple[MarkdownTableBlock | None, int]:
    if cursor + 1 >= len(lines):
        return None, cursor + 1
    header = _parse_standard_row(lines[cursor])
    separator = _parse_standard_row(lines[cursor + 1])
    if (
        header is None
        or header.is_separator
        or separator is None
        or not separator.is_separator
        or len(header.cells) != len(separator.cells)
    ):
        return None, cursor + 1

    column_count = len(header.cells)
    data_rows: list[MarkdownTableRow] = []
    data_cursor = cursor + 2
    while data_cursor < len(lines):
        line = lines[data_cursor]
        # 首行必须给出完整记录，后续行允许显式空单元格。空值仍保留为结构行，
        # 下游 selector 会拒绝把包含空值的行签成完整查询记录。
        row = _parse_standard_row(line, allow_empty_cells=bool(data_rows))
        if row is not None and _is_strict_data_row(row, column_count=column_count):
            data_rows.append(row)
            data_cursor += 1
            continue
        if data_rows:
            _raise_if_invalid_continuation(
                line=line,
                row=row,
                column_count=column_count,
            )
        break

    if not data_rows:
        # 稀疏版式经常产生只有 header/separator 的伪表；它不是 authority 表格。
        return None, cursor + 2
    return (
        MarkdownTableBlock(
            start=header.start,
            end=data_rows[-1].end,
            header=header,
            separator=separator,
            data_rows=tuple(data_rows),
            column_count=column_count,
            parse_mode=MarkdownTableParseMode.STANDARD,
        ),
        data_cursor,
    )


def _scan_ocr_candidate(
    lines: tuple[PhysicalLine, ...],
    cursor: int,
) -> tuple[MarkdownTableBlock | None, int]:
    header = _parse_ocr_header(lines[cursor])
    if header is None:
        # 显式 OCR 前缀拥有其连续片段，失败时不得从首数据行降级成标准表。
        return None, _failed_ocr_next_cursor(lines, cursor=cursor)
    if cursor + 3 >= len(lines):
        return None, _failed_ocr_next_cursor(lines, cursor=cursor)

    first_data = _parse_standard_row(lines[cursor + 1])
    separator = _parse_standard_row(lines[cursor + 2])
    column_count = len(header.cells)
    if (
        first_data is None
        or not _is_strict_data_row(first_data, column_count=column_count)
        or separator is None
        or not separator.is_separator
        or len(separator.cells) != column_count
    ):
        return None, _failed_ocr_next_cursor(lines, cursor=cursor)

    post_separator_rows: list[MarkdownTableRow] = []
    data_cursor = cursor + 3
    while data_cursor < len(lines):
        line = lines[data_cursor]
        row = _parse_standard_row(line, allow_empty_cells=bool(post_separator_rows))
        if row is not None and _is_strict_data_row(row, column_count=column_count):
            post_separator_rows.append(row)
            data_cursor += 1
            continue
        if post_separator_rows:
            _raise_if_invalid_continuation(
                line=line,
                row=row,
                column_count=column_count,
            )
        break

    if not post_separator_rows:
        return None, _failed_ocr_next_cursor(lines, cursor=cursor)
    data_rows = (first_data, *post_separator_rows)
    return (
        MarkdownTableBlock(
            start=header.start,
            end=data_rows[-1].end,
            header=header,
            separator=separator,
            data_rows=data_rows,
            column_count=column_count,
            parse_mode=MarkdownTableParseMode.OCR_HEADER_DATA_BEFORE_SEPARATOR,
        ),
        data_cursor,
    )


def scan_markdown_table_blocks(text: str) -> tuple[MarkdownTableBlock, ...]:
    """扫描完整文本并返回全部严格表格；歧义只会拒绝，不会猜测恢复。"""

    if PACKED_TRUNCATION_MARKER.strip() in text:
        raise MarkdownTableParseError("table authority contains a truncation marker")
    lines = _physical_lines(text)
    tables: list[MarkdownTableBlock] = []
    cursor = 0
    while cursor < len(lines):
        if _has_ocr_header_prefix(lines[cursor]):
            table, next_cursor = _scan_ocr_candidate(lines, cursor)
        else:
            table, next_cursor = _scan_standard_candidate(lines, cursor)
        if table is not None:
            tables.append(table)
        cursor = max(next_cursor, cursor + 1)
    return tuple(tables)


__all__ = [
    "MarkdownTableBlock",
    "MarkdownTableCell",
    "MarkdownTableParseError",
    "MarkdownTableParseMode",
    "MarkdownTableRow",
    "PhysicalLine",
    "scan_markdown_table_blocks",
]
