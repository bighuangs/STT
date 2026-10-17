"""可重放的业务 item 句法边界；只接受显式、查询锚定的结构边界。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata

from app.domain.answering.markdown_table import (
    MarkdownTableBlock,
    MarkdownTableParseError,
    MarkdownTableParseMode,
    scan_markdown_table_blocks,
)
from app.domain.answering.question import QuestionContract, QuestionObligation
from app.domain.answering.witness import EvidenceWitness, SpanEquivalence, WitnessAtomKind


_TOP_LEVEL_LIST_RE = re.compile(
    r"^(?:[-*•+]\s+|\d{1,3}(?:[.)．](?=\s|[^\d\s])|、(?!\d))\s*|"
    r"[一二三四五六七八九十]+[、.)．]\s*)"
)
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_MARKDOWN_IMAGE_ONLY_RE = re.compile(
    r"^!\[[^\]\r\n]*\]\([^\s()\r\n]+(?:\s+[\"'][^\"'\r\n]*[\"'])?\)$"
)
_IMAGE_DERIVATION_METADATA_START_RE = re.compile(
    r"^(?:视觉类型|图像类型|一般图形内容|轴/图例/标签|数据点|标题/注释)\s*[:：]?"
    r"|^(?:the\s+image|visual\s+type|data\s+points?)\s*[:：]",
    re.IGNORECASE,
)
_TABLE_CELL_ITEM_DELIMITER_RE = re.compile(r"[。！？!?；;]")
_TABLE_CELL_EMPTY_ITEM_RE = re.compile(r"[。！？!?；;]\s*[。！？!?；;]")
_TABLE_CELL_OPEN_REFERENCE_RE = re.compile(
    r"(?:参见|参考|详见|另见|更多(?:内容|信息|措施|操作)?|"
    r"包括但不限于|诸如|例如|比如|其余|其他(?:措施|操作|内容|事项)?)"
)
_TABLE_CELL_OPEN_TAIL_RE = re.compile(r"(?:等|等等|等措施|等操作|等内容|等事项)[。！？!?]?$")
_TABLE_CELL_DEPENDENT_PREFIX_RE = re.compile(r"^(?:并且?|以及|和|或|且|同时)[,，、\s]*")


class ClosedCountTableCellProjectionStatus(StrEnum):
    """CLOSED_COUNT 表格单元格投影的可信三态。"""

    NOT_APPLICABLE = "not_applicable"
    VALID = "valid"
    PROJECTED_SCOPE_INVALID = "projected_scope_invalid"


@dataclass(frozen=True, slots=True)
class ClosedCountTableCellProjection:
    """从完整 Witness 集合重放出的唯一目标行、目标列与 item 边界。"""

    status: ClosedCountTableCellProjectionStatus
    witness_id: str | None = None
    item_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        valid = self.status == ClosedCountTableCellProjectionStatus.VALID
        if valid != bool(self.witness_id and self.item_spans):
            raise ValueError("valid table-cell projection requires one witness and item spans")
        if not valid and (self.witness_id is not None or self.item_spans):
            raise ValueError("non-valid table-cell projection cannot carry witness or spans")


@dataclass(frozen=True, slots=True)
class _StandardTableWitnessGroup:
    """一个由连续 Witness 重建的标准 Markdown 表。"""

    table: MarkdownTableBlock
    data_witnesses: tuple[EvidenceWitness, ...]
    trusted: bool


def _markdown_table_cells(text: str) -> tuple[tuple[str, ...], tuple[int, int]] | None:
    """解析不含转义/代码片段的简单 Markdown 表格行。"""

    if "|" not in text or "\\|" in text or "`" in text:
        return None
    start = 0
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    if text[start] == "|":
        start += 1
    if end > start and text[end - 1] == "|":
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    body = text[start:end]
    cells = tuple(item.strip() for item in body.split("|"))
    if len(cells) < 2 or any(not item for item in cells):
        return None
    return cells, (start, end)


def markdown_table_row_shape(text: str) -> tuple[int, bool] | None:
    """返回列数及是否为 separator；只接受简单、可重放的表格句法。"""

    parsed = _markdown_table_cells(text)
    if parsed is None:
        return None
    cells, _ = parsed
    return len(cells), all(_TABLE_SEPARATOR_CELL_RE.fullmatch(item) for item in cells)


def markdown_table_data_row_body_span(text: str) -> tuple[int, int] | None:
    """返回已识别 Markdown data row 去除外侧管道后的正文坐标。"""

    parsed = _markdown_table_cells(text)
    if parsed is None:
        return None
    cells, span = parsed
    if all(_TABLE_SEPARATOR_CELL_RE.fullmatch(item) for item in cells):
        return None
    return span


def _table_anchor_normalized(value: str) -> str:
    """生成只用于显式 query/header/cell 包含判断的保守规范串。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith(("P", "S"))
    )


def _is_explicit_table_selector(value: str) -> bool:
    """拒绝单字符、纯数字等容易偶然命中的表格 selector。"""

    if not value or value.isnumeric():
        return False
    if re.search(r"[\u3400-\u9fff]", value):
        return len(value) >= 2
    return len(value) >= 4


def _explicit_cell_item_spans(value: str, *, expected_count: int) -> tuple[tuple[int, int], ...]:
    """按显式句末/分号边界切分封闭 cell，返回 cell 内相对坐标。"""

    if (
        expected_count < 2
        or not value.strip()
        or "\r" in value
        or "\n" in value
        or _TABLE_CELL_EMPTY_ITEM_RE.search(value)
        or _TABLE_CELL_OPEN_REFERENCE_RE.search(value)
        or _TABLE_CELL_OPEN_TAIL_RE.search(value.strip())
    ):
        return ()

    spans: list[tuple[int, int]] = []
    cursor = 0
    for delimiter in _TABLE_CELL_ITEM_DELIMITER_RE.finditer(value):
        start = cursor
        end = delimiter.end()
        while start < delimiter.start() and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if start >= delimiter.start():
            return ()
        spans.append((start, end))
        cursor = delimiter.end()
    trailing_start = cursor
    trailing_end = len(value)
    while trailing_start < trailing_end and value[trailing_start].isspace():
        trailing_start += 1
    while trailing_end > trailing_start and value[trailing_end - 1].isspace():
        trailing_end -= 1
    if trailing_start < trailing_end:
        spans.append((trailing_start, trailing_end))

    if len(spans) != expected_count:
        return ()
    normalized_items: list[str] = []
    for start, end in spans:
        raw = value[start:end]
        body = raw.rstrip("。！？!?；;").strip()
        normalized = _table_anchor_normalized(body)
        if (
            len(normalized) < 2
            or _TABLE_CELL_DEPENDENT_PREFIX_RE.match(body)
            or _TABLE_CELL_OPEN_REFERENCE_RE.search(body)
            or _TABLE_CELL_OPEN_TAIL_RE.search(body)
        ):
            return ()
        normalized_items.append(normalized)
    if len(set(normalized_items)) != len(normalized_items):
        return ()
    return tuple(spans)


def _coordinate_span(
    witness: EvidenceWitness,
    *,
    coordinate: str,
) -> tuple[int, int]:
    if coordinate == "packed":
        return witness.span_ref.packed_start, witness.span_ref.packed_end
    if coordinate == "canonical":
        return witness.span_ref.canonical_start, witness.span_ref.canonical_end
    raise ValueError("unsupported table item coordinate space")


def _physically_adjacent(
    left: EvidenceWitness,
    right: EvidenceWitness,
    *,
    coordinate: str,
) -> bool:
    _, left_end = _coordinate_span(left, coordinate=coordinate)
    right_start, _ = _coordinate_span(right, coordinate=coordinate)
    return right_start - left_end in {1, 2}


def _standard_table_witness_groups(
    *,
    witnesses: Sequence[EvidenceWitness],
    witness_texts: Mapping[str, str],
    coordinate: str,
) -> tuple[_StandardTableWitnessGroup, ...]:
    """从完整目录中重建所有连续标准 Markdown 表，不依赖模型选择。"""

    by_artifact: dict[tuple[str, str], list[EvidenceWitness]] = {}
    for witness in witnesses:
        key = (witness.span_ref.evidence_id, witness.span_ref.representation_id)
        by_artifact.setdefault(key, []).append(witness)

    groups: list[_StandardTableWitnessGroup] = []
    for artifact_witnesses in by_artifact.values():
        ordered = sorted(
            artifact_witnesses,
            key=lambda item: _coordinate_span(item, coordinate=coordinate)[0],
        )
        position = 0
        while position + 2 < len(ordered):
            header = ordered[position]
            separator = ordered[position + 1]
            first_data = ordered[position + 2]
            if (
                header.atom_kind != WitnessAtomKind.TABLE_ROW
                or separator.atom_kind != WitnessAtomKind.TABLE_ROW
                or first_data.atom_kind != WitnessAtomKind.TABLE_DATA_ROW
                or not _physically_adjacent(header, separator, coordinate=coordinate)
                or not _physically_adjacent(separator, first_data, coordinate=coordinate)
            ):
                position += 1
                continue

            end = position + 3
            while (
                end < len(ordered)
                and ordered[end].atom_kind == WitnessAtomKind.TABLE_DATA_ROW
                and _physically_adjacent(ordered[end - 1], ordered[end], coordinate=coordinate)
            ):
                end += 1
            table_witnesses = tuple(ordered[position:end])
            synthetic_text = "\n".join(witness_texts[item.witness_id] for item in table_witnesses)
            try:
                tables = scan_markdown_table_blocks(synthetic_text)
            except MarkdownTableParseError:
                position = end
                continue
            if (
                len(tables) == 1
                and tables[0].parse_mode == MarkdownTableParseMode.STANDARD
                and len(tables[0].data_rows) == len(table_witnesses) - 2
                and tables[0].header.line.text == witness_texts[header.witness_id]
                and tables[0].separator.line.text == witness_texts[separator.witness_id]
                and tuple(row.line.text for row in tables[0].data_rows)
                == tuple(witness_texts[item.witness_id] for item in table_witnesses[2:])
            ):
                groups.append(
                    _StandardTableWitnessGroup(
                        table=tables[0],
                        data_witnesses=table_witnesses[2:],
                        trusted=all(
                            item.span_ref.equivalence != SpanEquivalence.SECURITY_SAFE_DERIVED
                            for item in table_witnesses
                        ),
                    )
                )
            position = end
    return tuple(groups)


def _projection_from_table_group(
    group: _StandardTableWitnessGroup,
    *,
    information_need: str,
    source_query: str,
    expected_count: int,
    witness_texts: Mapping[str, str],
) -> ClosedCountTableCellProjection | None:
    """识别单表内是否存在 query-anchored cell scope，并严格解析它。"""

    table = group.table
    target_columns = tuple(
        index
        for index, cell in enumerate(table.header.cells)
        if (
            (header := _table_anchor_normalized(cell.text))
            and len(header) >= 2
            and header in information_need
        )
    )
    if not target_columns:
        return None

    matching_columns_by_row: dict[int, set[int]] = {}
    target_column_set = set(target_columns)
    for row_index, row in enumerate(table.data_rows):
        for column_index, cell in enumerate(row.cells):
            if column_index in target_column_set:
                continue
            selector = _table_anchor_normalized(cell.text)
            if _is_explicit_table_selector(selector) and selector in source_query:
                matching_columns_by_row.setdefault(row_index, set()).add(column_index)
    if not matching_columns_by_row:
        return None
    if not group.trusted or len(target_columns) != 1 or len(matching_columns_by_row) != 1:
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.PROJECTED_SCOPE_INVALID
        )

    selected_row_index, selector_columns = next(iter(matching_columns_by_row.items()))
    if len(selector_columns) != 1:
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.PROJECTED_SCOPE_INVALID
        )
    target_column = target_columns[0]
    selected_row = table.data_rows[selected_row_index]
    selected_witness = group.data_witnesses[selected_row_index]
    selected_text = witness_texts[selected_witness.witness_id]
    target_cell = selected_row.cells[target_column]
    local_spans = _explicit_cell_item_spans(
        target_cell.text,
        expected_count=expected_count,
    )
    if not local_spans:
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.PROJECTED_SCOPE_INVALID
        )
    cell_relative_start = target_cell.start - selected_row.start
    item_spans = tuple(
        (cell_relative_start + start, cell_relative_start + end) for start, end in local_spans
    )
    if any(not 0 <= start < end <= len(selected_text) for start, end in item_spans):
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.PROJECTED_SCOPE_INVALID
        )
    return ClosedCountTableCellProjection(
        status=ClosedCountTableCellProjectionStatus.VALID,
        witness_id=selected_witness.witness_id,
        item_spans=item_spans,
    )


def resolve_closed_count_markdown_table_cell_projection(
    *,
    question: QuestionContract,
    obligation: QuestionObligation,
    witnesses: Sequence[EvidenceWitness],
    witness_texts: Mapping[str, str],
    coordinate: str = "packed",
) -> ClosedCountTableCellProjection:
    """从全部可信表格中解析唯一 query row、target column 与 cell items。"""

    expected_count = obligation.expected_count
    if (
        obligation.closure.value != "closed_count"
        or expected_count is None
        or expected_count < 2
        or set(witness_texts) != {item.witness_id for item in witnesses}
    ):
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.NOT_APPLICABLE
        )

    information_need = _table_anchor_normalized(obligation.information_need)
    source_query = _table_anchor_normalized(question.source_query)
    candidates = tuple(
        candidate
        for group in _standard_table_witness_groups(
            witnesses=witnesses,
            witness_texts=witness_texts,
            coordinate=coordinate,
        )
        if (
            candidate := _projection_from_table_group(
                group,
                information_need=information_need,
                source_query=source_query,
                expected_count=expected_count,
                witness_texts=witness_texts,
            )
        )
        is not None
    )
    if not candidates:
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.NOT_APPLICABLE
        )
    if len(candidates) != 1 or candidates[0].status != ClosedCountTableCellProjectionStatus.VALID:
        return ClosedCountTableCellProjection(
            status=ClosedCountTableCellProjectionStatus.PROJECTED_SCOPE_INVALID
        )
    return candidates[0]


def closed_count_markdown_table_cell_item_spans(
    *,
    question: QuestionContract,
    obligation: QuestionObligation,
    selected_witness: EvidenceWitness,
    witnesses: Sequence[EvidenceWitness],
    witness_texts: Mapping[str, str],
    coordinate: str = "packed",
) -> tuple[tuple[int, int], ...]:
    """兼容入口：仅为全局解析出的唯一目标 Witness 返回全部 item span。"""

    projection = resolve_closed_count_markdown_table_cell_projection(
        question=question,
        obligation=obligation,
        witnesses=witnesses,
        witness_texts=witness_texts,
        coordinate=coordinate,
    )
    if (
        projection.status != ClosedCountTableCellProjectionStatus.VALID
        or projection.witness_id != selected_witness.witness_id
    ):
        return ()
    return projection.item_spans


def top_level_list_item_body_span(text: str) -> tuple[int, int] | None:
    """返回列表 marker 之后的非空正文半开区间。"""

    match = _TOP_LEVEL_LIST_RE.match(text)
    if match is None:
        return None
    start = match.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def is_markdown_image_only_line(text: str) -> bool:
    """判断一行是否只含一个可确定边界的 Markdown 图片。"""

    return _MARKDOWN_IMAGE_ONLY_RE.fullmatch(text.strip()) is not None


def is_image_derivation_metadata_start(text: str) -> bool:
    """识别图片派生说明的显式起始标签，避免把图例编号误作操作步骤。"""

    return _IMAGE_DERIVATION_METADATA_START_RE.search(text.strip()) is not None


__all__ = [
    "ClosedCountTableCellProjection",
    "ClosedCountTableCellProjectionStatus",
    "closed_count_markdown_table_cell_item_spans",
    "is_image_derivation_metadata_start",
    "is_markdown_image_only_line",
    "markdown_table_data_row_body_span",
    "markdown_table_row_shape",
    "resolve_closed_count_markdown_table_cell_projection",
    "top_level_list_item_body_span",
]
