"""TODO-07 多格式解析、切块模板与解析缓存的离线验收脚本。

默认测试 Markdown/TXT/HTML/CSV/DOCX/PPTX/XLSX，不访问网络、数据库、
MinIO 或向量库。传入 ``--pdf`` 时只验证 PDF 内容探测；只有显式增加
``--parse-pdf`` 才会调用已配置的 MinerU 服务。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import sys
import tempfile
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.document_parsing_service import (  # noqa: E402
    parse_and_chunk_document,
    resolve_parser_profile,
)
from app.application.parse_artifact_store import LocalParseArtifactStore  # noqa: E402
from app.domain.parsing import DocumentParseError, ParsedDocument  # noqa: E402
from app.rag.import_.chunking_templates import (  # noqa: E402
    chunk_parsed_document_with_profile,
    list_chunking_templates,
)


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
TEMPLATE_IDS = {"general", "manual", "paper", "qa", "table", "presentation"}


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _write_zip(path: Path, entries: Iterable[tuple[str, str | bytes]]) -> Path:
    """以固定 ZIP 元数据创建可复现的最小 OOXML 输入。"""

    with zipfile.ZipFile(path, "w") as archive:
        for name, value in entries:
            payload = value.encode("utf-8") if isinstance(value, str) else value
            archive.writestr(_zip_info(name), payload)
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
<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
    <w:r><w:t>安全说明</w:t></w:r></w:p>
  <w:p><w:r><w:t>操作前必须断电并佩戴护目镜。</w:t></w:r></w:p>
  <w:tbl>
    <w:tr><w:tc><w:p><w:r><w:t>项目</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>数值</w:t></w:r></w:p></w:tc></w:tr>
    <w:tr><w:tc><w:p><w:r><w:t>额定压力</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>16 MPa</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
</w:body></w:document>"""
    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "word/document.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document.main+xml",
                )
            ),
        ),
        ("word/document.xml", document),
    ]


def _pptx_entries() -> list[tuple[str, str]]:
    presentation = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>
  <p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/>
</p:sldIdLst></p:presentation>"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rId1" Type="urn:test:slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId2" Type="urn:test:slide" Target="slides/slide2.xml"/>
</Relationships>"""

    def slide(title: str, body: str) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
    <p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
  <p:sp><p:nvSpPr><p:nvPr/></p:nvSpPr>
    <p:txBody><a:p><a:r><a:t>{body}</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>"""

    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "ppt/presentation.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation.main+xml",
                ),
                (
                    "ppt/slides/slide1.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.slide+xml",
                ),
                (
                    "ppt/slides/slide2.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.slide+xml",
                ),
            ),
        ),
        ("ppt/presentation.xml", presentation),
        ("ppt/_rels/presentation.xml.rels", relationships),
        ("ppt/slides/slide1.xml", slide("安全培训", "第一张幻灯片")),
        ("ppt/slides/slide2.xml", slide("维护步骤", "第二张幻灯片")),
    ]


def _xlsx_entries() -> list[tuple[str, str]]:
    workbook = f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{S}" xmlns:r="{R}"><sheets>
  <sheet name="安全参数" sheetId="1" r:id="rId1"/>
</sheets></workbook>"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rId1" Type="urn:test:worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    sheet = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{S}"><sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>项目</t></is></c>
    <c r="B1" t="inlineStr"><is><t>数值</t></is></c></row>
  <row r="2"><c r="A2" t="inlineStr"><is><t>额定压力</t></is></c>
    <c r="B2"><v>16</v></c></row>
</sheetData></worksheet>"""
    return [
        (
            "[Content_Types].xml",
            _content_types(
                (
                    "xl/workbook.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet.main+xml",
                ),
                (
                    "xl/worksheets/sheet1.xml",
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.worksheet+xml",
                ),
            ),
        ),
        ("xl/workbook.xml", workbook),
        ("xl/_rels/workbook.xml.rels", relationships),
        ("xl/worksheets/sheet1.xml", sheet),
    ]


def _make_samples(root: Path) -> dict[str, tuple[Path, str | None, set[str]]]:
    root.mkdir(parents=True, exist_ok=True)
    markdown = root / "manual.md"
    markdown.write_text(
        "# 安全说明\n\n设备必须接地。\n\n| 项目 | 数值 |\n| --- | --- |\n| 压力 | 16 MPa |\n",
        encoding="utf-8",
    )
    text = root / "notes.txt"
    text.write_text("第一段安全要求。\n\n第二段维护要求。\n", encoding="utf-8")
    html = root / "manual.html"
    html.write_text(
        "<!doctype html><html><body><h1>安全说明</h1><p>设备必须接地。</p>"
        "<table><tr><th>项目</th><th>数值</th></tr>"
        "<tr><td>压力</td><td>16 MPa</td></tr></table></body></html>",
        encoding="utf-8",
    )
    csv = root / "parameters.csv"
    csv.write_text("项目,数值\n压力,16 MPa\n危险公式,=1+1\n", encoding="utf-8")
    docx = _write_zip(root / "manual.docx", _docx_entries())
    pptx = _write_zip(root / "slides.pptx", _pptx_entries())
    xlsx = _write_zip(root / "sheet.xlsx", _xlsx_entries())
    return {
        "markdown": (markdown, "text/markdown", {"heading", "text", "table"}),
        "txt": (text, "text/plain", {"text"}),
        "html": (html, "text/html", {"heading", "text", "table"}),
        "csv": (csv, "text/csv", {"table"}),
        "docx": (docx, None, {"heading", "text", "table"}),
        "pptx": (pptx, None, {"heading", "text"}),
        "xlsx": (xlsx, None, {"table"}),
    }


def _offline_config(template_id: str = "general") -> dict[str, object]:
    return {
        "chunking_template": template_id,
        "persist_parsed_artifact_to_minio": False,
        "require_parsed_artifact_minio": False,
        "chunk_token_num": 96,
        "chunk_overlap_tokens": 0,
    }


def _verify_formats(
    samples: dict[str, tuple[Path, str | None, set[str]]],
    store: LocalParseArtifactStore,
) -> dict[str, ParsedDocument]:
    documents: dict[str, ParsedDocument] = {}
    for expected_format, (path, mime, expected_types) in samples.items():
        outcome = parse_and_chunk_document(
            path,
            declared_mime=mime,
            parser_config=_offline_config(),
            artifact_store=store,
        )
        actual_types = {block.type for block in outcome.document.blocks}
        assert outcome.profile.detected_format == expected_format
        assert expected_types <= actual_types
        assert outcome.document.blocks
        assert outcome.chunks
        assert [block.order for block in outcome.document.blocks] == list(
            range(len(outcome.document.blocks))
        )
        assert len({block.block_id for block in outcome.document.blocks}) == len(
            outcome.document.blocks
        )
        documents[expected_format] = outcome.document
        print(
            f"  [通过] {path.name:<18} -> {outcome.profile.parser_id} "
            f"| blocks={len(outcome.document.blocks)} | chunks={len(outcome.chunks)}"
        )
    return documents


def _verify_templates(document: ParsedDocument) -> None:
    available = {info.template_id for info in list_chunking_templates()}
    assert available == TEMPLATE_IDS
    known_blocks = {block.block_id for block in document.blocks}
    for template_id in sorted(TEMPLATE_IDS):
        first, profile = chunk_parsed_document_with_profile(
            document,
            template_id=template_id,
            file_title="TODO-07 验收手册",
            config={"target_tokens": 96, "overlap_tokens": 0},
        )
        second, _ = chunk_parsed_document_with_profile(
            document,
            template_id=template_id,
            file_title="TODO-07 验收手册",
            config={"target_tokens": 96, "overlap_tokens": 0},
        )
        assert first
        assert [item["chunk_id"] for item in first] == [
            item["chunk_id"] for item in second
        ]
        assert [item["position_int"] for item in first] == list(range(len(first)))
        assert len({item["chunk_id"] for item in first}) == len(first)
        for chunk in first:
            provenance = chunk["provenance"]
            assert chunk["chunk_id"].startswith("chk_")
            assert chunk["content"].strip()
            assert provenance["parser_id"] == document.parser_id
            assert provenance["parser_version"] == document.parser_version
            assert provenance["template_id"] == template_id
            assert provenance["template_version"] == profile["template_version"]
            assert set(provenance["source_block_ids"]) <= known_blocks
            assert provenance["source_block_ids"]
        print(
            f"  [通过] {template_id:<12} version={profile['template_version']} "
            f"| chunks={len(first)} | ID 稳定且 provenance 完整"
        )


def _verify_cache(docx: Path, store: LocalParseArtifactStore) -> None:
    first = parse_and_chunk_document(
        docx,
        parser_config=_offline_config("general"),
        artifact_store=store,
    )
    second = parse_and_chunk_document(
        docx,
        parser_config=_offline_config("general"),
        artifact_store=store,
    )
    changed_template = parse_and_chunk_document(
        docx,
        parser_config=_offline_config("manual"),
        artifact_store=store,
    )
    assert first.artifact.cache_hit is False
    assert second.artifact.cache_hit is True
    assert changed_template.artifact.cache_hit is True
    assert first.artifact.cache_key == second.artifact.cache_key
    assert first.artifact.cache_key == changed_template.artifact.cache_key
    assert first.artifact.sha256 == changed_template.artifact.sha256
    assert {chunk["provenance"]["template_id"] for chunk in changed_template.chunks} == {
        "manual"
    }
    print("  [通过] 第二次解析命中缓存；切换 general -> manual 仍复用同一解析制品")


def _verify_malicious_zip(root: Path, store: LocalParseArtifactStore) -> None:
    malicious = _write_zip(
        root / "malicious.docx",
        [*_docx_entries(), ("../escape.xml", "不应写出目标目录")],
    )
    try:
        parse_and_chunk_document(
            malicious,
            parser_config=_offline_config(),
            artifact_store=store,
        )
    except DocumentParseError as exc:
        assert exc.code == "PARSE_ZIP_PATH_TRAVERSAL"
        assert exc.retryable is False
        print(f"  [通过] 恶意 ZIP 被稳定拒绝：{exc.code}")
        return
    raise AssertionError("恶意 ZIP 未被拒绝")


def _verify_pdf(
    pdf_path: Path,
    *,
    parse_pdf: bool,
    store: LocalParseArtifactStore,
) -> None:
    if not pdf_path.is_file():
        raise AssertionError(f"PDF 不存在：{pdf_path}")
    config = _offline_config("manual")
    profile = resolve_parser_profile(
        pdf_path,
        declared_mime="application/pdf",
        parser_config=config,
    )
    assert profile.detected_format == "pdf"
    assert profile.parser_id == "mineru.pdf"
    print(
        f"  [通过] PDF 仅本地探测：{pdf_path.name} -> "
        f"{profile.parser_id}@{profile.parser_version}"
    )
    if not parse_pdf:
        print("  [跳过] 未传 --parse-pdf，没有上传 PDF，也没有调用 MinerU")
        return
    outcome = parse_and_chunk_document(
        pdf_path,
        declared_mime="application/pdf",
        parser_config=config,
        artifact_store=store,
    )
    assert outcome.document.blocks and outcome.chunks
    print(
        f"  [通过] PDF 完整解析：blocks={len(outcome.document.blocks)} "
        f"| chunks={len(outcome.chunks)} | cache_hit={outcome.artifact.cache_hit}"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线验证 TODO-07 多格式解析、六类模板、缓存和安全拒绝。"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="可选 PDF 路径（例如 doc/hak180产品安全手册.pdf）；默认只做本地探测。",
    )
    parser.add_argument(
        "--parse-pdf",
        action="store_true",
        help="显式允许调用 MinerU 对 --pdf 做完整解析；未设置时绝不调用外部服务。",
    )
    args = parser.parse_args()
    if args.parse_pdf and args.pdf is None:
        parser.error("--parse-pdf 必须与 --pdf PATH 一起使用")
    return args


def main() -> int:
    args = _arguments()
    print("TODO-07 验收开始（默认完全离线）")
    try:
        temp_parent = ROOT / ".test-artifacts"
        temp_parent.mkdir(parents=True, exist_ok=True)
        # 保留一段有代表性的工作路径，以覆盖 Windows 长路径制品原子写入回归。
        # TemporaryDirectory 退出时会完整清理本次样例、缓存与规范化中间文件。
        with tempfile.TemporaryDirectory(prefix="stt-todo07-", dir=temp_parent) as temp:
            root = Path(temp)
            store = LocalParseArtifactStore(root / "parse-cache")
            samples = _make_samples(root / "samples")

            print("\n1. 七种格式的内容探测与结构解析")
            documents = _verify_formats(samples, store)

            print("\n2. 六类切块模板、溯源与稳定 ID")
            _verify_templates(documents["docx"])

            print("\n3. 解析制品缓存与模板解耦")
            _verify_cache(
                samples["docx"][0],
                LocalParseArtifactStore(root / "cache-verification"),
            )

            print("\n4. 压缩包安全边界")
            _verify_malicious_zip(root / "samples", store)

            if args.pdf is not None:
                print("\n5. 用户指定 PDF")
                _verify_pdf(
                    args.pdf.expanduser().resolve(),
                    parse_pdf=args.parse_pdf,
                    store=store,
                )
    except (AssertionError, DocumentParseError, OSError, ValueError) as exc:
        code = f" [{exc.code}]" if isinstance(exc, DocumentParseError) else ""
        print(f"\n[失败]{code} {exc}", file=sys.stderr)
        return 1

    print("\nTODO-07 验收通过：格式、结构、模板、provenance、缓存与安全边界均符合预期。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
