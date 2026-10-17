from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.domain.answering.markdown_table import (
    MarkdownTableParseError,
    MarkdownTableParseMode,
    scan_markdown_table_blocks,
)
from app.domain.answering.witness import PACKED_TRUNCATION_MARKER


def test_scans_standard_table_with_exact_line_and_cell_spans() -> None:
    text = "前言\r\n  | 名称 | 值 |  \r\n| :--- | ---: |\r\n| Alpha |  甲  |\r\n尾注"

    tables = scan_markdown_table_blocks(text)

    assert len(tables) == 1
    table = tables[0]
    assert table.parse_mode is MarkdownTableParseMode.STANDARD
    assert table.column_count == 2
    assert table.span == (text.index("  | 名称"), text.index("\r\n尾注"))
    assert table.line_indexes == (1, 2, 3)
    assert table.header.line.text == "  | 名称 | 值 |  "
    assert table.header.values == ("名称", "值")
    assert table.separator.values == (":---", "---:")
    assert table.data_rows[0].values == ("Alpha", "甲")
    for row in (table.header, table.separator, *table.data_rows):
        assert text[row.start : row.end] == row.line.text
        for cell in row.cells:
            assert text[cell.start : cell.end] == cell.text
    assert table.header.cells[0].span == (
        text.index("名称"),
        text.index("名称") + len("名称"),
    )


def test_scans_narrow_ocr_table_and_preserves_physical_and_logical_order() -> None:
    text = (
        "标题\n表头：纸盒 | 支持的纸张\n多功能纸盒 | A4\n--- | ---\n纸盒 1 | A3\n纸盒 2 | A5\n结尾"
    )

    table = scan_markdown_table_blocks(text)[0]

    assert table.parse_mode is MarkdownTableParseMode.OCR_HEADER_DATA_BEFORE_SEPARATOR
    assert table.header.values == ("纸盒", "支持的纸张")
    assert tuple(row.values for row in table.data_rows) == (
        ("多功能纸盒", "A4"),
        ("纸盒 1", "A3"),
        ("纸盒 2", "A5"),
    )
    assert table.separator.line.index == 3
    assert tuple(row.line.index for row in table.data_rows) == (2, 4, 5)
    assert table.line_indexes == (1, 2, 3, 4, 5)
    assert text[table.start : table.end].startswith("表头：")
    assert text[table.header.cells[0].start : table.header.cells[0].end] == "纸盒"


def test_skips_sparse_pseudo_table_without_poisoning_later_table() -> None:
    text = (
        "| 菜单 | 说明 |\n"
        "| --- | --- |\n"
        "此区域没有数据行。\n"
        "\n"
        "| 型号 | 功能 |\n"
        "| --- | --- |\n"
        "| Alpha | 功能甲 |"
    )

    tables = scan_markdown_table_blocks(text)

    assert len(tables) == 1
    assert tables[0].header.values == ("型号", "功能")
    assert tables[0].data_rows[0].values == ("Alpha", "功能甲")


def test_truncation_marker_is_globally_fail_closed() -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 甲 |" + PACKED_TRUNCATION_MARKER

    with pytest.raises(MarkdownTableParseError, match="truncation marker"):
        scan_markdown_table_blocks(text)


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        ("| --- | --- |", "secondary separator"),
        ("| Beta | 乙 | 多余 |", "column drift"),
        ("| Beta | 同上 |", "empty, ditto, merged, or invalid"),
        ('| Beta | <td rowspan="2">乙</td> |', "empty, ditto, merged, or invalid"),
    ],
)
def test_accepted_standard_table_rejects_ambiguous_continuations(
    tail: str,
    message: str,
) -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 甲 |\n" + tail

    with pytest.raises(MarkdownTableParseError, match=message):
        scan_markdown_table_blocks(text)


def test_standard_table_keeps_later_rows_with_explicit_empty_cells() -> None:
    text = (
        "| 故障现象 | 原因 | 办法 |\n"
        "| --- | --- | --- |\n"
        "| 打印发白 | 碳粉不足 | 调整浓度 |\n"
        "| 黑色背景 |  |  |"
    )

    table = scan_markdown_table_blocks(text)[0]

    assert tuple(row.values for row in table.data_rows) == (
        ("打印发白", "碳粉不足", "调整浓度"),
        ("黑色背景", "", ""),
    )


def test_invalid_first_data_row_does_not_create_a_table() -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| Alpha | 同上 |\n普通正文"

    assert scan_markdown_table_blocks(text) == ()


@pytest.mark.parametrize(
    "text",
    [
        "表头：名称 | 名称\nAlpha | 甲\n--- | ---\nBeta | 乙",
        "表头：名称 | 值\nAlpha | 甲\n--- | ---",
        "表头：名称 | 值\nAlpha | 同上\n--- | ---\nBeta | 乙",
        "表头：名称 | 值\nAlpha | 甲\n--- | ---\nBeta | ",
    ],
)
def test_invalid_ocr_fragment_is_not_recovered_as_a_standard_table(text: str) -> None:
    assert scan_markdown_table_blocks(text) == ()


def test_ocr_header_requires_adjacency_and_does_not_claim_a_later_standard_fragment() -> None:
    text = "表头：名称 | 值\n\nAlpha | 甲\n--- | ---\nBeta | 乙"

    tables = scan_markdown_table_blocks(text)

    assert len(tables) == 1
    assert tables[0].parse_mode is MarkdownTableParseMode.STANDARD
    assert tables[0].header.values == ("Alpha", "甲")


def test_ocr_metadata_preambles_do_not_hide_a_following_explicit_standard_table() -> None:
    text = (
        "表头：设备报告 |  |  |\n"
        "表头：[热点列表] |  |  | [第 1 页]\n"
        "| 序号 | 热点名称 | 加密方式 | 信号强度 |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | Test-AP2 | WPA/WPA2 | 60% |\n"
        "| 2 | Test-AP3 | WPA/WPA2 | 60% |"
    )

    tables = scan_markdown_table_blocks(text)

    assert len(tables) == 1
    assert tables[0].parse_mode is MarkdownTableParseMode.STANDARD
    assert tables[0].header.values == ("序号", "热点名称", "加密方式", "信号强度")
    assert tuple(row.values for row in tables[0].data_rows) == (
        ("1", "Test-AP2", "WPA/WPA2", "60%"),
        ("2", "Test-AP3", "WPA/WPA2", "60%"),
    )


def test_ocr_table_rejects_column_drift_after_becoming_valid() -> None:
    text = "表头：名称 | 值\nAlpha | 甲\n--- | ---\nBeta | 乙\nGamma | 丙 | 多余"

    with pytest.raises(MarkdownTableParseError, match="column drift"):
        scan_markdown_table_blocks(text)


def test_ocr_table_keeps_later_rows_with_empty_unselected_cells() -> None:
    text = (
        "表头：一级菜单 | 二级菜单 | 三级菜单\n"
        "| 1.系统设置 | 1.语言设置 | 1.中文 |\n"
        "| --- | --- | --- |\n"
        "| 4.网络信息 | 2.无线网络信息 | 1.连接状态 |\n"
        "| 5.打印信息 | 1.演示页 |  |"
    )

    table = scan_markdown_table_blocks(text)[0]

    assert tuple(row.values for row in table.data_rows) == (
        ("1.系统设置", "1.语言设置", "1.中文"),
        ("4.网络信息", "2.无线网络信息", "1.连接状态"),
        ("5.打印信息", "1.演示页", ""),
    )


def test_multiple_tables_have_disjoint_exact_spans() -> None:
    text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n边界\n表头：C | D\n3 | 4\n--- | ---\n5 | 6"

    first, second = scan_markdown_table_blocks(text)

    assert first.end < second.start
    assert text[first.start : first.end] == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert text[second.start : second.end] == "表头：C | D\n3 | 4\n--- | ---\n5 | 6"


def test_result_models_are_immutable() -> None:
    table = scan_markdown_table_blocks("| A | B |\n| --- | --- |\n| 1 | 2 |")[0]

    with pytest.raises(FrozenInstanceError):
        table.start = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        table.header.cells[0].text = "changed"  # type: ignore[misc]
