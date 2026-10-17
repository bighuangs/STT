"""保留原始字符坐标的 Evidence 原子切分。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from app.domain.answering.item_boundary import top_level_list_item_body_span
from app.domain.answering.markdown_table import scan_markdown_table_blocks
from app.domain.safe_representation import find_markdown_image_spans


_LINE_RE = re.compile(r"[^\r\n]+")
_HTML_ENTITY_SUFFIX_RE = re.compile(r"&(?:#[xX][0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]{1,31});$")
_ASCII_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "vs.",
    "u.s.",
)
ProtectedSafeSpan = tuple[int, int] | tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EvidenceAtom:
    """``text[start:end]`` 的一个不改写原文的半开区间。"""

    start: int
    end: int
    kind: str

    def resolve(self, text: str) -> str:
        return text[self.start : self.end]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _protected_span_at(
    boundary: int,
    protected_spans: Sequence[tuple[int, int, int | None]],
) -> tuple[int, int] | None:
    low = 0
    high = len(protected_spans)
    while low < high:
        middle = (low + high) // 2
        if protected_spans[middle][0] < boundary:
            low = middle + 1
        else:
            high = middle
    if low == 0:
        return None
    protected_start, protected_end, safe_step = protected_spans[low - 1]
    if not protected_start < boundary < protected_end:
        return None
    if safe_step is not None:
        delta = boundary - protected_start
        if delta % safe_step == 0:
            return None
        step_start = protected_start + (delta // safe_step) * safe_step
        return step_start, min(step_start + safe_step, protected_end)
    return protected_start, protected_end


def _bounded_spans(
    text: str,
    start: int,
    end: int,
    max_characters: int,
    protected_spans: Sequence[tuple[int, int, int | None]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_characters:
        boundary = cursor + max_characters
        floor = cursor + max(64, max_characters // 2)
        candidates = [
            text.rfind(marker, floor, boundary)
            for marker in ("。", "；", ";", "，", ",", " ", "\t")
        ]
        split = max(candidates, default=-1)
        if split < floor:
            split = boundary
        else:
            split += 1
        protected = _protected_span_at(split, protected_spans)
        if protected is not None:
            protected_start, protected_end = protected
            if protected_start > cursor:
                split = protected_start
            elif protected_end > end:
                split = end
            elif protected_end - cursor <= max_characters:
                split = protected_end
            else:
                raise ValueError("non-linear safe derivation span exceeds witness atom limit")
        trimmed = _trim_span(text, cursor, split)
        if trimmed is not None:
            spans.append(trimmed)
        cursor = split
    terminal_protected = _protected_span_at(end, protected_spans)
    if terminal_protected is not None and terminal_protected[0] > cursor:
        prefix = _trim_span(text, cursor, terminal_protected[0])
        if prefix is not None:
            spans.append(prefix)
        partial = _trim_span(text, terminal_protected[0], end)
        if partial is not None:
            spans.append(partial)
    else:
        trimmed = _trim_span(text, cursor, end)
        if trimmed is not None:
            spans.append(trimmed)
    return spans


def _sentence_spans(line: str) -> list[tuple[int, int]]:
    """按中英文明确句界切分，同时避开常见缩写和人名首字母。"""

    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(line):
        boundary = char in "。！？!?；;"
        if char == ";" and _HTML_ENTITY_SUFFIX_RE.search(line[: index + 1]):
            boundary = False
        if char == "." and (index + 1 == len(line) or line[index + 1].isspace()):
            prefix = line[: index + 1].lower()
            token_match = re.search(r"([A-Za-z]+)\.$", line[: index + 1])
            initial = bool(token_match and len(token_match.group(1)) == 1)
            boundary = not initial and not prefix.endswith(_ASCII_ABBREVIATIONS)
        if not boundary:
            continue
        end = index + 1
        span = _trim_span(line, start, end)
        if span is not None:
            spans.append(span)
        start = end
    tail = _trim_span(line, start, len(line))
    if tail is not None:
        spans.append(tail)
    return spans


def _markdown_table_kinds(
    text: str,
) -> dict[tuple[int, int], str]:
    """将领域解析器确认的完整表格物理行投影为 atom kind。"""

    kinds: dict[tuple[int, int], str] = {}
    for table in scan_markdown_table_blocks(text):
        for row in (table.header, table.separator):
            span = _trim_span(text, row.start, row.end)
            if span is not None:
                kinds[span] = "table_row"
        for row in table.data_rows:
            span = _trim_span(text, row.start, row.end)
            if span is not None:
                # 含显式空单元格的行证明了表格物理边界，但并非完整可选记录。
                kinds[span] = (
                    "table_data_row"
                    if all(cell.normalized_text for cell in row.cells)
                    else "table_row"
                )
    return kinds


def atomize_evidence_text(
    text: str,
    *,
    max_atom_characters: int = 1_800,
    protected_spans: Sequence[ProtectedSafeSpan] = (),
) -> tuple[EvidenceAtom, ...]:
    """按表格行、列表项或句子切分，并保留每个字符的原始坐标。"""

    if max_atom_characters < 64:
        raise ValueError("max_atom_characters must be at least 64")
    normalized_protected = tuple(
        sorted(
            ((item[0], item[1], item[2] if len(item) == 3 else None) for item in protected_spans),
            key=lambda item: (item[0], item[1]),
        )
    )
    if any(
        start < 0 or end <= start or (safe_step is not None and safe_step <= 0)
        for start, end, safe_step in normalized_protected
    ) or any(
        current[0] < previous[1]
        for previous, current in zip(
            normalized_protected,
            normalized_protected[1:],
            strict=False,
        )
    ):
        raise ValueError("protected safe derivation spans must be ordered and disjoint")
    atoms: list[EvidenceAtom] = []
    lines: list[tuple[int, int]] = []
    for line_match in _LINE_RE.finditer(text):
        trimmed_line = _trim_span(text, line_match.start(), line_match.end())
        if trimmed_line is not None:
            lines.append(trimmed_line)
    table_kinds = _markdown_table_kinds(text)
    for line_start, line_end in lines:
        line = text[line_start:line_end]
        if (line_start, line_end) in table_kinds:
            pieces = [(line_start, line_end, table_kinds[(line_start, line_end)])]
        elif top_level_list_item_body_span(line) is not None:
            pieces = [(line_start, line_end, "list_item")]
        elif find_markdown_image_spans(line):
            pieces = [(line_start, line_end, "media_derivation")]
        else:
            pieces = []
            for sentence_start, sentence_end in _sentence_spans(line):
                span = _trim_span(text, line_start + sentence_start, line_start + sentence_end)
                if span is not None:
                    pieces.append((*span, "sentence"))
        for start, end, kind in pieces:
            for bounded_start, bounded_end in _bounded_spans(
                text,
                start,
                end,
                max_atom_characters,
                normalized_protected,
            ):
                atoms.append(EvidenceAtom(bounded_start, bounded_end, kind))
    return tuple(atoms)


__all__ = [
    "EvidenceAtom",
    "ProtectedSafeSpan",
    "atomize_evidence_text",
    "top_level_list_item_body_span",
]
