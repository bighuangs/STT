"""TODO-07 六类切块模板、溯源和解析制品缓存回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.application.parse_artifact_store import (
    LocalParseArtifactStore,
    build_parse_cache_key,
)
from app.domain.parsing import DocumentParseError, ParsedBlock, ParsedDocument, SourceLocation
from app.rag.import_.chunking_templates import (
    MANUAL_TABLE_AWARE_BASE_VERSION,
    MANUAL_TABLE_AWARE_VERSION,
    chunk_parsed_document,
    chunk_parsed_document_with_profile,
    count_tokens,
    default_chunking_template_registry,
    list_chunking_templates,
)


SOURCE_SHA256 = "a" * 64
TEMPLATE_IDS = {"general", "manual", "paper", "qa", "table", "presentation"}


def _document(
    blocks: list[ParsedBlock] | None = None,
    *,
    source_format: str = "markdown",
) -> ParsedDocument:
    if blocks is None:
        blocks = [
            ParsedBlock(
                block_id="heading-safety",
                type="heading",
                text="安全说明",
                location=SourceLocation(page_no=1),
                order=0,
            ),
            ParsedBlock(
                block_id="safety-body",
                type="text",
                text="操作前必须断电并泄压。",
                location=SourceLocation(page_no=1),
                parent_heading_path=["安全说明"],
                order=1,
            ),
            ParsedBlock(
                block_id="pressure-table",
                type="table",
                text="项目\t数值\n额定压力\t16 MPa",
                structured_data={"headers": ["项目", "数值"], "rows": [["额定压力", "16 MPa"]]},
                location=SourceLocation(page_no=2),
                parent_heading_path=["安全说明"],
                order=2,
            ),
            ParsedBlock(
                block_id="question-one",
                type="heading",
                text="问题：如何泄压？",
                location=SourceLocation(page_no=3),
                order=3,
            ),
            ParsedBlock(
                block_id="answer-one",
                type="text",
                text="缓慢打开泄压阀。",
                location=SourceLocation(page_no=3),
                parent_heading_path=["问题：如何泄压？"],
                order=4,
            ),
            ParsedBlock(
                block_id="question-two",
                type="heading",
                text="问题：额定压力是多少？",
                location=SourceLocation(page_no=4),
                order=5,
            ),
            ParsedBlock(
                block_id="answer-two",
                type="text",
                text="额定压力为 16 MPa。",
                location=SourceLocation(page_no=4),
                parent_heading_path=["问题：额定压力是多少？"],
                order=6,
            ),
        ]
    return ParsedDocument(
        parser_id="test.parser",
        parser_version="1.2.3",
        source_sha256=SOURCE_SHA256,
        source_format=source_format,
        source_mime_type="text/markdown",
        metadata={"title": "TODO-07 测试手册"},
        blocks=blocks,
    )


def _boundary_document(kind: str) -> ParsedDocument:
    if kind == "slide":
        blocks = [
            ParsedBlock(
                block_id="slide-one",
                type="text",
                text="第一页内容",
                location=SourceLocation(slide_no=1),
                order=0,
            ),
            ParsedBlock(
                block_id="slide-two",
                type="text",
                text="第二页内容",
                location=SourceLocation(slide_no=2),
                order=1,
            ),
        ]
        return _document(blocks, source_format="pptx")
    blocks = [
        ParsedBlock(
            block_id="sheet-one",
            type="text",
            text="参数表一",
            location=SourceLocation(sheet="参数一", cell_range="A1:B2"),
            order=0,
        ),
        ParsedBlock(
            block_id="sheet-two",
            type="text",
            text="参数表二",
            location=SourceLocation(sheet="参数二", cell_range="A1:B2"),
            order=1,
        ),
    ]
    return _document(blocks, source_format="xlsx")


def test_all_six_versioned_templates_are_available_and_deterministic() -> None:
    infos = list_chunking_templates()
    assert {info.template_id for info in infos} == TEMPLATE_IDS
    assert {
        info.version for info in infos if info.template_id == "manual"
    } == {
        "1.0.0",
        MANUAL_TABLE_AWARE_BASE_VERSION,
        MANUAL_TABLE_AWARE_VERSION,
    }
    assert {
        info.version for info in infos if info.template_id != "manual"
    } == {"1.0.0"}
    document = _document()
    known_ids = {block.block_id for block in document.blocks}

    for template_id in sorted(TEMPLATE_IDS):
        first, first_profile = chunk_parsed_document_with_profile(
            document,
            template_id=template_id,
            config={"target_tokens": 64, "overlap_tokens": 0},
        )
        second = chunk_parsed_document(
            document,
            template_id=template_id,
            config={"target_tokens": 64, "overlap_tokens": 0},
        )

        assert first == second
        assert first_profile["template_id"] == template_id
        expected_version = (
            MANUAL_TABLE_AWARE_VERSION if template_id == "manual" else "1.0.0"
        )
        assert first_profile["template_version"] == expected_version
        assert [chunk["position_int"] for chunk in first] == list(range(len(first)))
        assert len({chunk["chunk_id"] for chunk in first}) == len(first)
        for chunk in first:
            assert chunk["id"] == chunk["chunk_id"]
            assert chunk["chunk_id"].startswith("chk_")
            assert chunk["content"].strip()
            provenance = chunk["provenance"]
            assert provenance["parser_id"] == "test.parser"
            assert provenance["parser_version"] == "1.2.3"
            assert provenance["template_id"] == template_id
            assert provenance["template_version"] == expected_version
            assert set(provenance["source_block_ids"]) <= known_ids
            assert all(location["block_id"] in known_ids for location in provenance["locations"])


@pytest.mark.parametrize(("kind", "location_key"), [("slide", "slide_no"), ("sheet", "sheet")])
@pytest.mark.parametrize("template_id", sorted(TEMPLATE_IDS))
def test_no_template_merges_different_slides_or_sheets(
    kind: str,
    location_key: str,
    template_id: str,
) -> None:
    chunks = chunk_parsed_document(
        _boundary_document(kind),
        template_id=template_id,
        config={"target_tokens": 64, "overlap_tokens": 0},
    )

    observed: set[object] = set()
    for chunk in chunks:
        values = {
            location[location_key]
            for location in chunk["provenance"]["locations"]
            if location_key in location
        }
        assert len(values) <= 1
        observed.update(values)
    assert len(observed) == 2


def test_manual_template_inherits_heading_and_keeps_page_provenance() -> None:
    document = _document()
    first = chunk_parsed_document(
        document,
        template_id="manual",
        config={"target_tokens": 64, "overlap_tokens": 0},
    )
    second = chunk_parsed_document(
        document,
        template_id="manual",
        config={"target_tokens": 64, "overlap_tokens": 0},
    )

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    safety = next(chunk for chunk in first if "操作前必须断电" in chunk["content"])
    assert "安全说明" in safety["content"]
    assert safety["provenance"]["heading_path"] == ["安全说明"]
    assert {item.get("page_no") for item in safety["provenance"]["locations"]} == {1}
    table = next(chunk for chunk in first if chunk["type"] == "table")
    assert "| 项目 | 数值 |" in table["content"]
    assert set(table["provenance"]["source_block_ids"]) >= {
        "heading-safety",
        "pressure-table",
    }


def _hak180_table_document() -> ParsedDocument:
    table_body = """<table>
      <tr><td>型号</td><td colspan="6">有害物质</td></tr>
      <tr><td>HAK180</td><td>铅</td><td>汞</td><td>镉</td><td>六价铬</td><td>多溴联苯</td><td>多溴二苯醚</td></tr>
      <tr><td>部件名称</td><td>(Pb)</td><td>(Hg)</td><td>(Cd)</td><td>(Cr(VI))</td><td>(PBB)</td><td>(PBDE)</td></tr>
      <tr><td>框架L单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>框架R单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>中框架单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>框架</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>顶盖单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>进纸器单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>热熔器</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>盖板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>标签</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>金属薄片保持单元</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>主电路板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>低压电源电路板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>选配件</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
      <tr><td>包装材料</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr>
    </table>"""
    return _document(
        [
            ParsedBlock(
                block_id="hazard-heading",
                type="heading",
                text="产品中有害物质的名称及含量",
                location=SourceLocation(
                    page_no=1,
                    bbox=(0.746, 0.405, 0.869, 0.42),
                    coordinate_space="normalized_page",
                ),
                parent_heading_path=["HAK 180 烫金机", "设备和电源线"],
                order=0,
            ),
            ParsedBlock(
                block_id="hazard-table",
                type="table",
                text=table_body,
                structured_data={
                    "table_body": table_body,
                    "table_caption": [],
                    "table_footnote": [],
                },
                location=SourceLocation(
                    page_no=1,
                    bbox=(0.747, 0.426, 0.975, 0.752),
                    coordinate_space="normalized_page",
                ),
                parent_heading_path=["HAK 180 烫金机", "产品中有害物质的名称及含量"],
                order=1,
            ),
        ]
    )


def test_manual_v1_1_hak180_html_table_is_not_cut_inside_tags_and_keeps_late_rows() -> None:
    document = _hak180_table_document()
    first = chunk_parsed_document(
        document,
        template_id="manual",
        template_version=MANUAL_TABLE_AWARE_VERSION,
        config={"target_tokens": 512, "overlap_tokens": 0},
    )
    second = chunk_parsed_document(
        document,
        template_id="manual",
        template_version=MANUAL_TABLE_AWARE_VERSION,
        config={"target_tokens": 512, "overlap_tokens": 0},
    )

    assert first == second
    assert len(first) == 1
    table = first[0]
    assert table["type"] == "table"
    assert "铅" in table["content"] and "(Pb)" in table["content"]
    assert "主电路板" in table["content"] and "低压电源电路板" in table["content"]
    assert "<table" not in table["content"]
    assert "</td>" not in table["content"]
    assert "colspan=" not in table["content"]
    assert "```json" not in table["content"]
    assert count_tokens(table["content"]) <= 512
    assert set(table["provenance"]["source_block_ids"]) >= {
        "hazard-heading",
        "hazard-table",
    }
    assert {item["block_id"] for item in table["provenance"]["locations"]} == {
        "hazard-heading",
        "hazard-table",
    }
    table_location = next(
        item
        for item in table["provenance"]["locations"]
        if item["block_id"] == "hazard-table"
    )
    assert table_location == {
        "block_id": "hazard-table",
        "page_no": 1,
        "bbox": [0.747, 0.426, 0.975, 0.752],
        "coordinate_space": "normalized_page",
    }


def test_manual_v1_1_repeats_complex_html_headers_under_small_token_budget() -> None:
    chunks = chunk_parsed_document(
        _hak180_table_document(),
        template_id="manual",
        template_version=MANUAL_TABLE_AWARE_VERSION,
        config={"target_tokens": 160, "overlap_tokens": 0},
    )

    assert len(chunks) > 1
    assert all(chunk["type"] == "table" for chunk in chunks)
    assert all(count_tokens(chunk["content"]) <= 160 for chunk in chunks)
    assert all("型号" in chunk["content"] for chunk in chunks)
    assert all("铅" in chunk["content"] and "(Pb)" in chunk["content"] for chunk in chunks)
    assert all("<td" not in chunk["content"] for chunk in chunks)
    assert all(
        "hazard-table" in chunk["provenance"]["source_block_ids"]
        for chunk in chunks
    )


def test_manual_v1_1_attaches_headings_and_only_keeps_terminal_heading() -> None:
    document = _document(
        [
            ParsedBlock(
                block_id="root-heading",
                type="heading",
                text="HAK 180 烫金机",
                location=SourceLocation(page_no=1),
                order=0,
            ),
            ParsedBlock(
                block_id="device-heading",
                type="heading",
                text="设备",
                parent_heading_path=["HAK 180 烫金机", "警告"],
                location=SourceLocation(page_no=1),
                order=1,
            ),
            ParsedBlock(
                block_id="device-body",
                type="text",
                text="请先阅读这本手册再操作设备。",
                parent_heading_path=["HAK 180 烫金机", "设备"],
                location=SourceLocation(page_no=1),
                order=2,
            ),
            ParsedBlock(
                block_id="terminal-heading",
                type="heading",
                text="附录",
                parent_heading_path=["HAK 180 烫金机"],
                location=SourceLocation(page_no=1),
                order=3,
            ),
        ]
    )

    chunks = chunk_parsed_document(
        document,
        template_id="manual",
        template_version=MANUAL_TABLE_AWARE_VERSION,
        config={"target_tokens": 64, "overlap_tokens": 0},
    )

    body = next(chunk for chunk in chunks if "请先阅读" in chunk["content"])
    assert body["type"] == "text"
    assert "设备" in body["content"]
    assert set(body["provenance"]["source_block_ids"]) >= {
        "root-heading",
        "device-heading",
        "device-body",
    }
    assert sum(chunk["type"] == "heading" for chunk in chunks) == 1
    assert next(chunk for chunk in chunks if chunk["type"] == "heading")["content"] == "## 附录"


def test_manual_v1_0_remains_addressable_with_its_original_chunk_identity() -> None:
    document = _hak180_table_document()
    first = chunk_parsed_document(
        document,
        template_id="manual",
        template_version="1.0.0",
        config={"target_tokens": 160, "overlap_tokens": 0},
    )
    second = chunk_parsed_document(
        document,
        template_id="manual",
        template_version="1.0.0",
        config={"target_tokens": 160, "overlap_tokens": 0},
    )
    upgraded = chunk_parsed_document(
        document,
        template_id="manual",
        template_version=MANUAL_TABLE_AWARE_VERSION,
        config={"target_tokens": 160, "overlap_tokens": 0},
    )

    assert first == second
    assert {chunk["provenance"]["template_version"] for chunk in first} == {"1.0.0"}
    assert [chunk["chunk_id"] for chunk in first] != [
        chunk["chunk_id"] for chunk in upgraded
    ]
    assert any("<table" in chunk["content"] for chunk in first)


def test_template_identity_is_part_of_stable_chunk_id() -> None:
    document = _document(
        [
            ParsedBlock(
                block_id="only",
                type="text",
                text="同一份解析内容。",
                location=SourceLocation(page_no=1),
                order=0,
            )
        ]
    )
    general = chunk_parsed_document(document, template_id="general")[0]
    manual = chunk_parsed_document(document, template_id="manual")[0]

    assert general["content"] == manual["content"]
    assert general["chunk_id"] != manual["chunk_id"]


def test_unknown_template_and_invalid_config_have_stable_errors() -> None:
    with pytest.raises(DocumentParseError) as missing:
        default_chunking_template_registry.resolve("missing")
    with pytest.raises(DocumentParseError) as invalid:
        chunk_parsed_document(
            _document(),
            template_id="general",
            config={"target_tokens": 32, "overlap_tokens": 32},
        )

    assert missing.value.code == "CHUNK_TEMPLATE_NOT_FOUND"
    assert invalid.value.code == "CHUNK_CONFIG_INVALID"
    assert missing.value.retryable is False
    assert invalid.value.retryable is False


def test_parse_cache_key_excludes_chunk_template_but_tracks_extraction_inputs() -> None:
    common = {
        "input_sha256": SOURCE_SHA256,
        "parser_id": "local.docx",
        "parser_version": "1.0.0",
    }
    general = build_parse_cache_key(
        **common,
        parser_config={
            "template_id": "general",
            "chunking": {"target_tokens": 128},
            "chunk_token_num": 128,
            "language": "zh-CN",
        },
    )
    manual = build_parse_cache_key(
        **common,
        parser_config={
            "template_id": "manual",
            "chunking": {"target_tokens": 512},
            "chunk_token_num": 512,
            "language": "zh-CN",
        },
    )
    different_language = build_parse_cache_key(
        **common,
        parser_config={"template_id": "manual", "language": "en-US"},
    )
    different_parser_version = build_parse_cache_key(
        **{**common, "parser_version": "1.0.1"},
        parser_config={"language": "zh-CN"},
    )

    assert general == manual
    assert different_language != general
    assert different_parser_version != general


def test_switching_templates_reuses_one_parse_artifact(tmp_path: Path) -> None:
    store = LocalParseArtifactStore(tmp_path / "parse-cache")
    calls = 0

    def load_or_parse(parser_config: dict) -> ParsedDocument:
        nonlocal calls
        cache_key = build_parse_cache_key(
            input_sha256=SOURCE_SHA256,
            parser_id="test.parser",
            parser_version="1.2.3",
            parser_config=parser_config,
        )
        cached = store.load(cache_key)
        if cached is not None:
            return cached[0]
        calls += 1
        parsed = _document()
        store.save(cache_key, parsed)
        return parsed

    general_document = load_or_parse(
        {"language": "zh-CN", "template_id": "general", "chunk_token_num": 64}
    )
    general_chunks = chunk_parsed_document(general_document, template_id="general")
    manual_document = load_or_parse(
        {"language": "zh-CN", "template_id": "manual", "chunk_token_num": 256}
    )
    manual_chunks = chunk_parsed_document(manual_document, template_id="manual")

    assert calls == 1
    assert general_document == manual_document
    assert general_chunks[0]["provenance"]["parser_id"] == "test.parser"
    assert manual_chunks[0]["provenance"]["parser_id"] == "test.parser"
    assert {chunk["provenance"]["template_id"] for chunk in general_chunks} == {"general"}
    assert {chunk["provenance"]["template_id"] for chunk in manual_chunks} == {"manual"}


def test_parse_artifact_round_trip_and_tampering_fail_closed(tmp_path: Path) -> None:
    store = LocalParseArtifactStore(tmp_path / "parse-cache")
    key = build_parse_cache_key(
        input_sha256=SOURCE_SHA256,
        parser_id="test.parser",
        parser_version="1.2.3",
        parser_config={"language": "zh-CN"},
    )
    saved = store.save(key, _document())
    loaded = store.load(key)

    assert saved.cache_hit is False
    assert loaded is not None
    loaded_document, loaded_info = loaded
    assert loaded_document == _document()
    assert loaded_info.cache_hit is True
    assert loaded_info.sha256 == saved.sha256

    artifact = Path(saved.uri)
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(DocumentParseError) as captured:
        store.load(key)

    assert captured.value.code == "PARSE_ARTIFACT_INTEGRITY_ERROR"
    assert captured.value.retryable is False


def test_incomplete_artifact_and_unsafe_cache_key_are_rejected(tmp_path: Path) -> None:
    store = LocalParseArtifactStore(tmp_path / "parse-cache")
    key = "b" * 64
    store.save(key, _document())
    store.artifact_path(key).with_suffix(".sha256").unlink()

    with pytest.raises(DocumentParseError) as incomplete:
        store.load(key)
    with pytest.raises(DocumentParseError) as invalid:
        store.load("../outside")

    assert incomplete.value.code == "PARSE_ARTIFACT_INCOMPLETE"
    assert invalid.value.code == "INVALID_CACHE_KEY"
