"""TODO-08 MinerU PDF 页码、坐标与安全降级的离线测试。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter

from app.domain.parsing import DocumentParseError, SourceLocation
from app.rag.import_ import pdf_parse_service
from app.rag.import_.parsing.base import DetectionResult, ParseRequest
from app.rag.import_.pdf_parser_adapter import MinerUPdfParser


def _request(tmp_path: Path) -> ParseRequest:
    source = tmp_path / "hak180.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with source.open("wb") as output:
        writer.write(output)
    return ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="a" * 64,
    )


def test_source_location_requires_declared_normalized_coordinate_space() -> None:
    location = SourceLocation(
        page_no=3,
        bbox=[0.1, 0.2, 0.8, 0.9],
        coordinate_space="normalized_page",
    )
    assert location.bbox == [0.1, 0.2, 0.8, 0.9]
    with pytest.raises(ValueError):
        SourceLocation(page_no=3, bbox=[100, 200, 800, 900])
    with pytest.raises(ValueError):
        SourceLocation(page_no=3, bbox=[0.1, 0.2, 1.2, 0.9], coordinate_space="normalized_page")
    with pytest.raises(ValueError):
        SourceLocation(page_no=3, bbox=[0.1, 0.2, 0.1, 0.9], coordinate_space="normalized_page")
    with pytest.raises(ValueError):
        SourceLocation(page_no=3, bbox=[0.1, 0.2, 0.8, 0.2], coordinate_space="normalized_page")


def test_parse_service_records_only_unique_non_v2_content_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    state = {
        "task_id": "parse-aabbccddeeff0011",
        "pdf_path": str(request.path),
        "local_dir": str(tmp_path / "output"),
        "mineru_content_list_path": "stale.json",
    }
    captured: dict[str, Path] = {}
    include_duplicate = False

    def fake_download(_url, local_dir, stem, **_kwargs):
        extract_root = Path(local_dir) / stem
        extract_root.mkdir(parents=True)
        markdown = extract_root / "hak180.md"
        markdown.write_text("# 安全手册", encoding="utf-8")
        selected = extract_root / "hak180_content_list.json"
        selected.write_text("[]", encoding="utf-8")
        (extract_root / "hak180_content_list_v2.json").write_text(
            "[]",
            encoding="utf-8",
        )
        if include_duplicate:
            (extract_root / "duplicate_content_list.json").write_text(
                "[]",
                encoding="utf-8",
            )
        captured["selected"] = selected
        return markdown

    monkeypatch.setattr(pdf_parse_service, "upload_pdf_and_poll", lambda *args, **kwargs: "zip")
    monkeypatch.setattr(
        pdf_parse_service,
        "download_and_extract_markdown",
        fake_download,
    )
    parsed = pdf_parse_service.parse_pdf_to_markdown(state)
    assert parsed["mineru_content_list_path"] == str(captured["selected"].resolve())

    include_duplicate = True
    parsed = pdf_parse_service.parse_pdf_to_markdown(state)
    assert "mineru_content_list_path" not in parsed


def test_mineru_presign_payload_keeps_pdf_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        tmp_path
        / "dff3aaa813342799.0123456789abcdef.empty-password.pdf"
    )
    source.write_bytes(b"%PDF fixture")
    captured: dict = {}

    class BusinessFailure:
        status_code = 200
        text = "business failure"

        @staticmethod
        def json() -> dict:
            return {"code": -1, "msg": "stop after payload capture"}

    def fake_post(url, *, headers, json, timeout):
        captured.update(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return BusinessFailure()

    monkeypatch.setattr(
        pdf_parse_service,
        "mineru_gateway",
        SimpleNamespace(base_url="https://mineru.example/api/v4", api_key="secret"),
    )
    monkeypatch.setattr(pdf_parse_service.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="stop after payload capture"):
        pdf_parse_service.upload_pdf_and_poll(source)

    assert captured["json"]["files"] == [{"name": source.name}]
    assert captured["json"]["files"][0]["name"].endswith(".pdf")


def test_mineru_artifact_stem_is_bounded_and_extracts_in_long_work_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / f"{'very-long-source-name-' * 6}.pdf"
    state = {
        "task_id": "parse-dff3aaa813342799",
        "pdf_path": str(source),
        "local_dir": str(tmp_path / "unused"),
    }
    artifact_stem = pdf_parse_service._build_mineru_artifact_stem(state, source)
    assert artifact_stem.startswith("pdf-")
    assert len(artifact_stem) == 24
    assert artifact_stem.isascii()
    assert artifact_stem != pdf_parse_service._build_mineru_artifact_stem(
        state,
        source,
    )

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result/full.md", "# HAK180\n\n必须可靠接地。")
        archive.writestr("result/hak180_content_list.json", "[]")
        archive.writestr(
            "388d6297-2994-41eb-90c1-b15997a5ce5f_origin.pdf",
            b"%PDF fixture",
        )

    class ZipResponse:
        content = archive_bytes.getvalue()

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(
        pdf_parse_service.requests,
        "get",
        lambda *args, **kwargs: ZipResponse(),
    )
    # pytest 的 Windows tmp_path 已经接近真实 ``<64hex>.work`` 根长度；
    # 再追加一个短 cache 目录即可覆盖本次 118 字符工作根形态。
    local_dir = tmp_path / f"cache-work-{'a' * 16}"
    expected_origin = (
        local_dir
        / artifact_stem
        / "388d6297-2994-41eb-90c1-b15997a5ce5f_origin.pdf"
    )
    assert len(str(expected_origin.resolve())) <= 240
    markdown = pdf_parse_service.download_and_extract_markdown(
        "https://mineru.example/result.zip",
        local_dir,
        artifact_stem,
    )

    assert markdown.read_text(encoding="utf-8").startswith("# HAK180")
    assert markdown.resolve().is_relative_to((local_dir / artifact_stem).resolve())
    assert (local_dir / artifact_stem / "result" / "hak180_content_list.json").is_file()


def test_mineru_download_rejects_uncontrolled_artifact_stem_before_network(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="制品目录名不符合安全格式"):
        pdf_parse_service.download_and_extract_markdown(
            "https://mineru.example/result.zip",
            tmp_path,
            "../escape",
        )


@pytest.mark.parametrize(
    "member_name",
    [
        "result/CON.txt",
        "result/figure.png:payload",
        "result/trailing-dot.",
        "result/trailing-space ",
    ],
)
def test_safe_mineru_zip_rejects_windows_unsafe_member_names(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, b"unsafe")

    with pytest.raises(ValueError, match="Windows 不安全文件名"):
        pdf_parse_service._safe_extract_mineru_zip(
            archive_path,
            tmp_path / "extract",
        )


def test_safe_mineru_zip_rejects_case_insensitive_duplicate_names(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "duplicates.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("result/full.md", "first")
        archive.writestr("RESULT/FULL.MD", "second")

    with pytest.raises(ValueError, match="重复条目"):
        pdf_parse_service._safe_extract_mineru_zip(
            archive_path,
            tmp_path / "extract",
        )


def test_mineru_zip_download_error_does_not_rethrow_capability_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_url = "https://mineru.example/result.zip?signature=must-not-leak"

    def fail_download(*args, **kwargs):
        raise pdf_parse_service.requests.ConnectionError(
            f"failed to fetch {capability_url}"
        )

    monkeypatch.setattr(pdf_parse_service.requests, "get", fail_download)
    with pytest.raises(RuntimeError) as captured:
        pdf_parse_service.download_and_extract_markdown(
            capability_url,
            tmp_path,
            "pdf-0123456-0123456789ab",
        )

    assert capability_url not in str(captured.value)
    assert "MinerU ZIP 下载失败" in str(captured.value)


def test_mineru_content_list_preserves_hak_page_bbox_order_and_safe_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    extracted = tmp_path / "extracted"
    (extracted / "images").mkdir(parents=True)
    (extracted / "images" / "warning.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    markdown = extracted / "hak180.md"
    markdown.write_text("# 降级正文", encoding="utf-8")
    content_list = extracted / "hak180_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {"type": "header", "text": "页眉", "page_idx": 0, "bbox": [0, 0, 1000, 50]},
                {"type": "text", "text_level": 1, "text": "产品安全", "page_idx": 1, "bbox": [100, 120, 900, 200]},
                {"type": "text", "text": "必须佩戴护目镜。", "page_idx": 1, "bbox": [100, 250, 900, 340]},
                {"type": "image", "image_caption": ["警告标识"], "img_path": "images/warning.png", "page_idx": 2, "bbox": [200, 300, 600, 700]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {
            **state,
            "md_path": str(markdown),
            "mineru_content_list_path": str(content_list),
        },
    )

    parsed = MinerUPdfParser().parse(request)

    assert parsed.parser_version == "1.1.3"
    assert [block.text for block in parsed.blocks] == ["产品安全", "必须佩戴护目镜。", "警告标识"]
    assert parsed.blocks[0].location.page_no == 2
    assert parsed.blocks[0].location.bbox == [0.1, 0.12, 0.9, 0.2]
    assert parsed.blocks[0].location.coordinate_space == "normalized_page"
    assert parsed.blocks[1].parent_heading_path == ["产品安全"]
    assert parsed.blocks[2].location.page_no == 3
    assert len(parsed.assets) == 1 and parsed.blocks[2].asset_refs == [parsed.assets[0].asset_id]


def test_repository_hak180_recovers_mineru_mixed_font_inline_gap_from_same_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2] / "doc" / "hak180产品安全手册.pdf"
    assert source.is_file()
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.3",
        ),
        source_sha256="dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765",
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    markdown = extracted / "hak180.md"
    markdown.write_text("本表格依据 的规定编制。", encoding="utf-8")
    content_list = extracted / "hak180_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "本表格依据 的规定编制。",
                    "page_idx": 0,
                    "bbox": [749, 755, 873, 768],
                },
                {
                    "type": "text",
                    "text": "与原文无关的 MinerU 文本。",
                    "page_idx": 0,
                    "bbox": [749, 755, 873, 768],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {
            **state,
            "md_path": str(markdown),
            "mineru_content_list_path": str(content_list),
        },
    )

    parsed = MinerUPdfParser().parse(request)

    assert "SJ/T 11364" in parsed.blocks[0].text
    assert parsed.blocks[0].location == SourceLocation(
        page_no=1,
        bbox=[0.749, 0.755, 0.873, 0.768],
        coordinate_space="normalized_page",
    )
    assert parsed.blocks[1].text == "与原文无关的 MinerU 文本。"
    recovered_warnings = [
        warning
        for warning in parsed.warnings
        if warning.code == "PARSE_PDF_TEXT_LAYER_GAP_RECOVERED"
    ]
    assert len(recovered_warnings) == 1
    assert recovered_warnings[0].location == SourceLocation(page_no=1)
    assert "共 1 块" in recovered_warnings[0].message


def test_mineru_content_list_removes_only_disallowed_c0_c1_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    extracted = tmp_path / "extracted-controls"
    extracted.mkdir()
    markdown = extracted / "pantum.md"
    markdown.write_text("# 降级正文", encoding="utf-8")
    content_list = extracted / "pantum_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "80 g/m²\u0001普通纸\n第二行\t对齐\r\n中文😀",
                    "page_idx": 67,
                    "bbox": [100, 100, 900, 200],
                },
                {
                    "type": "table",
                    "text": "介质\u0085规格",
                    "table_caption": ["纸张\u007f类型"],
                    "table_body": "<table>\n<tr><td>60\u009f页</td></tr>\n</table>",
                    "page_idx": 67,
                    "bbox": [100, 250, 900, 600],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {
            **state,
            "md_path": str(markdown),
            "mineru_content_list_path": str(content_list),
        },
    )

    parsed = MinerUPdfParser().parse(request)

    assert parsed.parser_version == "1.1.3"
    assert parsed.blocks[0].text == "80 g/m²普通纸\n第二行\t对齐\r\n中文😀"
    assert "介质规格" in parsed.blocks[1].text
    assert "纸张类型" in parsed.blocks[1].text
    assert "60页" in parsed.blocks[1].text
    assert parsed.blocks[1].structured_data == {
        "table_body": "<table>\n<tr><td>60页</td></tr>\n</table>",
        "table_caption": ["纸张类型"],
    }
    serialized = parsed.model_dump_json()
    assert all(
        character in {"\n", "\r", "\t"}
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
        for character in parsed.blocks[0].text + parsed.blocks[1].text
    )
    assert "中文😀" in serialized


def test_mineru_markdown_fallback_uses_same_control_character_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    markdown = tmp_path / "pantum-fallback.md"
    markdown.write_text(
        "# 纸张\u0001规格\n\n80 g/m²\u0085普通纸\n\n保留\t制表符与中文😀。",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {**state, "md_path": str(markdown)},
    )

    parsed = MinerUPdfParser().parse(request)

    texts = "\n".join(block.text for block in parsed.blocks)
    heading_paths = "\n".join(
        heading
        for block in parsed.blocks
        for heading in block.parent_heading_path
    )
    assert "纸张规格" in texts
    assert "80 g/m²普通纸" in texts
    assert "保留\t制表符与中文😀" in texts
    assert "\u0001" not in texts + heading_paths
    assert "\u0085" not in texts + heading_paths


def test_owner_only_encrypted_pdf_is_normalized_before_mineru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "owner-only.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(user_password="", owner_password="owner-secret")
    with source.open("wb") as output:
        writer.write(output)
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="b" * 64,
    )
    markdown = tmp_path / "result.md"
    markdown.write_text("# 可读取手册", encoding="utf-8")
    captured: dict = {}

    def fake_parse(state: dict) -> dict:
        captured.update(state)
        normalized = PdfReader(state["pdf_path"])
        assert normalized.is_encrypted is False
        return {**state, "md_path": str(markdown)}

    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        fake_parse,
    )

    parsed = MinerUPdfParser().parse(request)

    assert Path(captured["pdf_path"]) != source
    assert Path(captured["pdf_path"]).name.startswith("normalized-")
    assert Path(captured["pdf_path"]).suffix == ".pdf"
    assert len(Path(captured["pdf_path"]).name) <= 48
    assert Path(captured["pdf_path"]).name.isascii()
    assert Path(captured["pdf_path"]).exists() is False
    assert "PARSE_PDF_EMPTY_PASSWORD_NORMALIZED" in {
        warning.code for warning in parsed.warnings
    }


def test_pdf_requiring_password_is_rejected_before_mineru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "password.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(user_password="required-secret", owner_password="owner-secret")
    with source.open("wb") as output:
        writer.write(output)
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="c" * 64,
    )
    called = False

    def forbidden_call(_state: dict) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        forbidden_call,
    )

    with pytest.raises(DocumentParseError) as captured:
        MinerUPdfParser().parse(request)

    assert captured.value.code == "PARSE_ENCRYPTED_DOCUMENT"
    assert called is False


def test_encrypt_text_without_encryption_dictionary_is_not_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "literal-encrypt.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Subject": "literal /Encrypt text"})
    with source.open("wb") as output:
        writer.write(output)
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="d" * 64,
    )
    markdown = tmp_path / "plain.md"
    markdown.write_text("# 普通 PDF", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: captured.update(state) or {**state, "md_path": str(markdown)},
    )

    parsed = MinerUPdfParser().parse(request)

    assert Path(captured["pdf_path"]) == source
    assert "PARSE_PDF_EMPTY_PASSWORD_NORMALIZED" not in {
        warning.code for warning in parsed.warnings
    }


def test_corrupt_encryption_structure_has_distinct_error(tmp_path: Path) -> None:
    source = tmp_path / "corrupt-encrypt.pdf"
    source.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Encrypt true >>\nendobj\n%%EOF")
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="e" * 64,
    )

    with pytest.raises(DocumentParseError) as captured:
        MinerUPdfParser().parse(request)

    assert captured.value.code == "PARSE_CORRUPT_PDF"


def test_generic_mineru_password_word_is_not_misclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda _state: (_ for _ in ()).throw(RuntimeError("upstream password field missing")),
    )

    with pytest.raises(DocumentParseError) as captured:
        MinerUPdfParser().parse(request)

    assert captured.value.code == "PARSE_PDF_FAILED"


def test_repository_hak180_owner_only_pdf_reaches_mineru_as_plain_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).resolve().parents[2] / "doc" / "hak180产品安全手册.pdf"
    assert source.is_file()
    request = ParseRequest(
        path=source,
        work_dir=tmp_path / "work",
        detection=DetectionResult(
            format="pdf",
            mime_type="application/pdf",
            parser_id="mineru.pdf",
            parser_version="1.1.1",
        ),
        source_sha256="dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765",
    )
    markdown = tmp_path / "hak180.md"
    markdown.write_text("# HAK180 产品安全手册", encoding="utf-8")
    captured: dict = {}

    def fake_parse(state: dict) -> dict:
        captured.update(state)
        assert PdfReader(state["pdf_path"]).is_encrypted is False
        return {**state, "md_path": str(markdown)}

    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        fake_parse,
    )

    parsed = MinerUPdfParser().parse(request)

    assert Path(captured["pdf_path"]) != source
    assert Path(captured["pdf_path"]).exists() is False
    assert "PARSE_PDF_EMPTY_PASSWORD_NORMALIZED" in {
        warning.code for warning in parsed.warnings
    }


def test_bad_or_missing_location_never_fabricates_page_and_markdown_fallback_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    markdown = extracted / "hak180.md"
    markdown.write_text("# 安全说明\n\n设备必须接地。", encoding="utf-8")
    content_list = extracted / "hak180_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {"type": "text", "text": "坐标缺失", "page_idx": 4},
                {"type": "text", "text": "坐标非法", "page_idx": 5, "bbox": [0, 0, 1100, 20]},
                {"type": "text", "text": "坐标退化", "page_idx": 6, "bbox": [100, 100, 100, 200]},
                {"type": "text", "text": "页码非法", "page_idx": "5", "bbox": [0, 0, 100, 100]},
                {"type": "image", "img_path": "../outside.png", "page_idx": 0, "bbox": [0, 0, 100, 100]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {**state, "md_path": str(markdown), "mineru_content_list_path": str(content_list)},
    )
    parsed = MinerUPdfParser().parse(request)
    assert parsed.blocks[0].location == SourceLocation(page_no=5)
    assert parsed.blocks[1].location == SourceLocation(page_no=6)
    assert parsed.blocks[2].location == SourceLocation(page_no=7)
    assert parsed.blocks[3].location == SourceLocation()
    codes = {warning.code for warning in parsed.warnings}
    assert {"PARSE_PDF_BBOX_UNAVAILABLE", "PARSE_PDF_BBOX_INVALID", "PARSE_PDF_PAGE_INDEX_INVALID", "PARSE_PDF_ASSET_OUTSIDE_ROOT"} <= codes

    monkeypatch.setattr(
        "app.rag.import_.pdf_parser_adapter.parse_pdf_to_markdown",
        lambda state: {**state, "md_path": str(markdown)},
    )
    fallback = MinerUPdfParser().parse(request)
    assert fallback.blocks
    assert "PARSE_PDF_CONTENT_LIST_UNAVAILABLE" in {warning.code for warning in fallback.warnings}
