from __future__ import annotations

import pytest

from app.application.query_answering.atomizer import atomize_evidence_text
from app.domain.answering.markdown_table import MarkdownTableParseError
from app.domain.answering.witness import PACKED_TRUNCATION_MARKER


def _resolved_atoms(text: str) -> list[tuple[str, str]]:
    return [(atom.resolve(text), atom.kind) for atom in atomize_evidence_text(text)]


def test_standard_table_uses_shared_parser_classification() -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 甲 |\n普通说明。"

    assert _resolved_atoms(text) == [
        ("| 名称 | 值 |", "table_row"),
        ("| --- | --- |", "table_row"),
        ("| Alpha | 甲 |", "table_data_row"),
        ("普通说明。", "sentence"),
    ]


def test_ocr_first_and_post_separator_rows_are_all_countable_data() -> None:
    text = "表头：型号 | 功能\nAlpha | 功能甲\n--- | ---\nBeta | 功能乙\nGamma | 功能丙"

    assert _resolved_atoms(text) == [
        ("表头：型号 | 功能", "table_row"),
        ("Alpha | 功能甲", "table_data_row"),
        ("--- | ---", "table_row"),
        ("Beta | 功能乙", "table_data_row"),
        ("Gamma | 功能丙", "table_data_row"),
    ]


def test_incomplete_standard_row_preserves_structure_without_becoming_data() -> None:
    text = (
        "| 故障现象 | 原因 | 办法 |\n"
        "| --- | --- | --- |\n"
        "| 打印发白 | 碳粉不足 | 调整浓度 |\n"
        "| 黑色背景 |  |  |"
    )

    assert _resolved_atoms(text) == [
        ("| 故障现象 | 原因 | 办法 |", "table_row"),
        ("| --- | --- | --- |", "table_row"),
        ("| 打印发白 | 碳粉不足 | 调整浓度 |", "table_data_row"),
        ("| 黑色背景 |  |  |", "table_row"),
    ]


def test_table_projection_matches_trimmed_atom_spans_with_crlf() -> None:
    text = "  | 名称 | 值 |  \r\n | --- | --- | \r\n  | Alpha | 甲 |  "

    atoms = atomize_evidence_text(text)

    assert [atom.resolve(text) for atom in atoms] == [
        "| 名称 | 值 |",
        "| --- | --- |",
        "| Alpha | 甲 |",
    ]
    assert [atom.kind for atom in atoms] == [
        "table_row",
        "table_row",
        "table_data_row",
    ]


def test_sparse_pseudo_table_produces_no_data_row_and_keeps_other_kinds() -> None:
    text = "| 菜单 | 说明 |\n| --- | --- |\n普通说明。\n- 列表项\n![示意图](image.png)"

    atoms = atomize_evidence_text(text)

    assert all(atom.kind != "table_data_row" for atom in atoms)
    assert ("普通说明。", "sentence") in _resolved_atoms(text)
    assert ("- 列表项", "list_item") in _resolved_atoms(text)
    assert ("![示意图](image.png)", "media_derivation") in _resolved_atoms(text)


@pytest.mark.parametrize(
    "tail",
    [
        "| --- | --- |",
        "| Beta | 乙 | 多余 |",
        "| Beta | 同上 |",
    ],
)
def test_shared_parser_failures_propagate_without_partial_classification(tail: str) -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 甲 |\n" + tail

    with pytest.raises(MarkdownTableParseError):
        atomize_evidence_text(text)


def test_truncated_table_authority_fails_before_atom_classification() -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 甲 |" + PACKED_TRUNCATION_MARKER

    with pytest.raises(MarkdownTableParseError, match="truncation marker"):
        atomize_evidence_text(text)
