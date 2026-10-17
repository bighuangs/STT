"""TODO-07 多格式解析、格式探测与安全边界的完全离线测试。"""

from __future__ import annotations

import socket
import struct
import warnings
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

from app.domain.parsing import DocumentParseError
from app.rag.import_.parsing.registry import default_parser_registry


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"


def _zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o100644 << 16
    return info


def _write_zip(
    path: Path,
    entries: Iterable[tuple[str, str | bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> Path:
    """用固定元数据生成可重复的 OOXML/恶意 ZIP 测试输入。"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in entries:
                payload = value.encode("utf-8") if isinstance(value, str) else value
                archive.writestr(_zip_info(name, compression), payload)
    return path


def _content_types(*overrides: tuple[str, str]) -> str:
    nodes = "".join(
        f'<Override PartName="/{part}" ContentType="{content_type}"/>'
        for part, content_type in overrides
    )
    return (
        f'<Types xmlns="{CONTENT_TYPES}">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{nodes}</Types>"
    )


def _docx_entries() -> list[tuple[str, str]]:
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:a="{A}">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>安全说明</w:t></w:r></w:p>
    <w:p><w:r><w:t>操作前必须佩戴护目镜。</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>项目</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>数值</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>额定压力</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>16 MPa</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rIdExternal" Type="urn:test:hyperlink" Target="https://example.invalid/tracker" TargetMode="External"/>
</Relationships>"""
    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "word/document.xml",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                )
            ),
        ),
        ("word/document.xml", document),
        ("word/_rels/document.xml.rels", relationships),
    ]


def _pptx_entries() -> list[tuple[str, str]]:
    presentation = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="{P}" xmlns:r="{R}">
  <p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst>
</p:presentation>"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rId1" Type="urn:test:slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId2" Type="urn:test:slide" Target="slides/slide2.xml"/>
</Relationships>"""

    def slide(title: str, body: str, *, with_table: bool = False) -> str:
        table = ""
        if with_table:
            table = """
      <p:graphicFrame><a:graphic><a:graphicData><a:tbl>
        <a:tr><a:tc><a:txBody><a:p><a:r><a:t>项目</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>要求</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
        <a:tr><a:tc><a:txBody><a:p><a:r><a:t>防护</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>护目镜</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
      </a:tbl></a:graphicData></a:graphic></p:graphicFrame>"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
  <p:sp><p:nvSpPr><p:nvPr/></p:nvSpPr><p:txBody><a:p><a:r><a:t>{body}</a:t></a:r></a:p></p:txBody></p:sp>
  {table}
</p:spTree></p:cSld></p:sld>"""

    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "ppt/presentation.xml",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
                ),
                ("ppt/slides/slide1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"),
                ("ppt/slides/slide2.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"),
            ),
        ),
        ("ppt/presentation.xml", presentation),
        ("ppt/_rels/presentation.xml.rels", relationships),
        ("ppt/slides/slide1.xml", slide("安全培训", "第一张幻灯片", with_table=True)),
        ("ppt/slides/slide2.xml", slide("维护步骤", "第二张幻灯片")),
    ]


def _xlsx_entries() -> list[tuple[str, str]]:
    workbook = f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="安全参数" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{PKG_REL}"><Relationship Id="rId1" Type="urn:test:worksheet" Target="worksheets/sheet1.xml"/></Relationships>"""
    sheet = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{S}"><sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>项目</t></is></c><c r="B1" t="inlineStr"><is><t>数值</t></is></c></row>
  <row r="2"><c r="A2" t="inlineStr"><is><t>校验值</t></is></c><c r="B2"><f>1+1</f><v>2</v></c></row>
</sheetData></worksheet>"""
    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "xl/workbook.xml",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
                ),
                ("xl/worksheets/sheet1.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"),
            ),
        ),
        ("xl/workbook.xml", workbook),
        ("xl/_rels/workbook.xml.rels", relationships),
        ("xl/worksheets/sheet1.xml", sheet),
    ]


def _set_encrypted_flag(path: Path) -> None:
    """将现有 ZIP 的本地头和中央目录通用标志 bit 0 置为 1。"""

    payload = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        cursor = 0
        while (position := payload.find(signature, cursor)) >= 0:
            start = position + flag_offset
            flags = struct.unpack_from("<H", payload, start)[0] | 0x1
            struct.pack_into("<H", payload, start, flags)
            cursor = position + 4
    path.write_bytes(payload)


def _parse(path: Path, tmp_path: Path, **kwargs):
    return default_parser_registry.parse(path, work_dir=tmp_path / "work", **kwargs)


@pytest.mark.parametrize(
    ("name", "mime", "payload", "expected_format", "expected_types"),
    [
        (
            "manual.md",
            "text/markdown",
            "# 安全说明\n\n设备必须接地。\n\n| 项目 | 数值 |\n| --- | --- |\n| 压力 | 16 MPa |\n",
            "markdown",
            {"heading", "text", "table"},
        ),
        ("notes.txt", "text/plain", "第一段。\n\n第二段。\n", "txt", {"text"}),
        (
            "manual.html",
            "text/html",
            "<!doctype html><html><body><h1>安全说明</h1><p>设备必须接地。</p><table><tr><th>项目</th><th>数值</th></tr><tr><td>压力</td><td>16 MPa</td></tr></table></body></html>",
            "html",
            {"heading", "text", "table"},
        ),
        (
            "parameters.csv",
            "text/csv",
            "项目,数值\n压力,16 MPa\n危险公式,=1+1\n",
            "csv",
            {"table"},
        ),
    ],
)
def test_text_formats_are_detected_and_parsed_with_structure(
    tmp_path: Path,
    name: str,
    mime: str,
    payload: str,
    expected_format: str,
    expected_types: set[str],
) -> None:
    source = tmp_path / name
    source.write_text(payload, encoding="utf-8")

    parsed = _parse(source, tmp_path, declared_mime=mime)

    assert parsed.source_format == expected_format
    assert parsed.source_mime_type
    expected_parser_id = "local.text" if expected_format == "txt" else f"local.{expected_format}"
    assert parsed.parser_id == expected_parser_id
    assert expected_types.issubset({block.type for block in parsed.blocks})
    assert [block.order for block in parsed.blocks] == list(range(len(parsed.blocks)))
    assert len({block.block_id for block in parsed.blocks}) == len(parsed.blocks)
    assert any("安全" in block.text or "第一段" in block.text or "项目" in block.text for block in parsed.blocks)
    if expected_format == "csv":
        assert parsed.blocks[0].structured_data["headers"] == ["项目", "数值"]
        assert "PARSE_CSV_FORMULA_NOT_EXECUTED" in {warning.code for warning in parsed.warnings}


@pytest.mark.parametrize(
    ("name", "entries", "expected_format"),
    [
        ("manual.docx", _docx_entries(), "docx"),
        ("slides.pptx", _pptx_entries(), "pptx"),
        ("sheet.xlsx", _xlsx_entries(), "xlsx"),
    ],
)
def test_office_formats_preserve_heading_table_slide_and_sheet_evidence(
    tmp_path: Path,
    name: str,
    entries: list[tuple[str, str]],
    expected_format: str,
) -> None:
    source = _write_zip(tmp_path / name, entries)

    parsed = _parse(source, tmp_path)

    assert parsed.source_format == expected_format
    assert parsed.parser_id == f"local.{expected_format}"
    if expected_format == "docx":
        assert [block.type for block in parsed.blocks] == ["heading", "text", "table"]
        assert parsed.blocks[1].parent_heading_path == ["安全说明"]
        assert parsed.blocks[2].structured_data == {
            "headers": ["项目", "数值"],
            "rows": [["额定压力", "16 MPa"]],
        }
        assert "PARSE_EXTERNAL_LINK_SKIPPED" in {warning.code for warning in parsed.warnings}
    elif expected_format == "pptx":
        assert parsed.metadata["slide_count"] == 2
        assert {block.location.slide_no for block in parsed.blocks} == {1, 2}
        assert any(block.type == "heading" and block.text == "安全培训" for block in parsed.blocks)
        assert any(block.type == "table" for block in parsed.blocks)
    else:
        table = parsed.blocks[0]
        assert table.type == "table"
        assert table.location.sheet == "安全参数"
        assert table.location.cell_range == "A1:B2"
        assert table.structured_data["formulas"] == {"B2": "1+1"}
        assert "PARSE_FORMULA_NOT_EVALUATED" in {warning.code for warning in parsed.warnings}


def test_content_signature_wins_over_fake_extension_and_declared_mime(tmp_path: Path) -> None:
    source = _write_zip(tmp_path / "disguised.pdf", _docx_entries())

    parsed = _parse(source, tmp_path, declared_mime="application/pdf")

    assert parsed.source_format == "docx"
    warning_codes = {warning.code for warning in parsed.warnings}
    assert warning_codes >= {"PARSE_EXTENSION_MISMATCH", "PARSE_MIME_MISMATCH"}


def test_html_active_content_is_removed_and_external_resources_are_never_fetched(
    tmp_path: Path,
    block_external_network: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def reject_connect(_socket: socket.socket, address):
        calls.append(address)
        raise AssertionError("HTML parser attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_connect)
    source = tmp_path / "unsafe.html"
    source.write_text(
        """<!doctype html><html><head><title>安全页面</title><style>.x{display:none}</style></head>
<body><script>SECRET_SCRIPT()</script><iframe src="https://example.invalid/frame">SECRET_FRAME</iframe>
<h1>安全说明</h1><p onclick="SECRET_EVENT()">正文<a href="https://example.invalid/a">外部链接</a></p>
<img src="https://example.invalid/pixel.png" alt="远程图片"></body></html>""",
        encoding="utf-8",
    )

    parsed = _parse(source, tmp_path)
    rendered = "\n".join(block.text for block in parsed.blocks)
    warning_codes = {warning.code for warning in parsed.warnings}

    assert parsed.metadata["title"] == "安全页面"
    assert "安全说明" in rendered and "正文" in rendered
    assert "SECRET_SCRIPT" not in rendered
    assert "SECRET_FRAME" not in rendered
    assert "SECRET_EVENT" not in rendered
    assert calls == []
    assert warning_codes >= {
        "PARSE_HTML_ACTIVE_CONTENT_REMOVED",
        "PARSE_EXTERNAL_LINK_SKIPPED",
        "PARSE_EXTERNAL_RESOURCE_SKIPPED",
    }


@pytest.mark.parametrize(
    ("case_name", "factory", "config", "expected_code"),
    [
        (
            "path-traversal",
            lambda path: _write_zip(path, [("../escape.xml", "x"), *_docx_entries()]),
            {},
            "PARSE_ZIP_PATH_TRAVERSAL",
        ),
        (
            "duplicate-entry",
            lambda path: _write_zip(
                path,
                [*_docx_entries(), ("word/document.xml", _docx_entries()[1][1])],
            ),
            {},
            "PARSE_ZIP_DUPLICATE_ENTRY",
        ),
        (
            "compression-bomb",
            lambda path: _write_zip(
                path,
                [*_docx_entries(), ("word/high-ratio.bin", b"A" * 50_000)],
                compression=zipfile.ZIP_DEFLATED,
            ),
            {"max_compression_ratio": 2.0},
            "PARSE_ZIP_BOMB",
        ),
        (
            "active-xml",
            lambda path: _write_zip(
                path,
                [
                    _docx_entries()[0],
                    (
                        "word/document.xml",
                        f'<!DOCTYPE w:document [<!ENTITY x "boom">]><w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>',
                    ),
                ],
            ),
            {},
            "PARSE_XML_ACTIVE_CONTENT",
        ),
    ],
)
def test_office_security_boundaries_fail_with_stable_codes(
    tmp_path: Path,
    case_name: str,
    factory,
    config: dict,
    expected_code: str,
) -> None:
    source = factory(tmp_path / f"{case_name}.docx")

    with pytest.raises(DocumentParseError) as captured:
        _parse(source, tmp_path, config=config)

    assert captured.value.code == expected_code
    assert captured.value.retryable is False


def test_encrypted_office_archive_is_rejected_before_reading_content(tmp_path: Path) -> None:
    source = _write_zip(tmp_path / "encrypted.docx", _docx_entries())
    _set_encrypted_flag(source)

    with pytest.raises(DocumentParseError) as captured:
        _parse(source, tmp_path)

    assert captured.value.code == "PARSE_ENCRYPTED_DOCUMENT"
    assert captured.value.retryable is False


def test_truncated_zip_and_plain_archive_have_distinct_stable_errors(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.docx"
    truncated.write_bytes(b"PK\x03\x04truncated")
    plain_zip = _write_zip(tmp_path / "plain.docx", [("readme.txt", "not office")])

    with pytest.raises(DocumentParseError) as corrupt:
        _parse(truncated, tmp_path)
    with pytest.raises(DocumentParseError) as unsupported:
        _parse(plain_zip, tmp_path)

    assert corrupt.value.code == "PARSE_CORRUPT_ARCHIVE"
    assert unsupported.value.code == "PARSE_UNSUPPORTED_ARCHIVE"
    assert unsupported.value.status_code == 415
