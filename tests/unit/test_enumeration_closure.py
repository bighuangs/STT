from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from app.domain.answering._base import sha256_digest
from app.domain.answering.enumeration_closure import (
    QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION,
    QUERY_ANCHOR_NORMALIZATION_VERSION,
    QueryAnchoredInlineActionChainClosureReceipt,
    QueryAnchoredProcedureListClosureReceipt,
    QueryAnchoredScopedBulletListClosureReceipt,
    QueryAnchoredScopedProcedureListClosureReceipt,
    QueryAnchoredScopedTableSelectionClosureReceipt,
    QueryAnchoredTableRowClosureReceipt,
    ScopedListAuthorityRepresentation,
    ScopedBulletAuthorityRepresentation,
    ScopedTableAuthorityRepresentation,
    parse_source_enumeration_closure_receipt,
)
from app.domain.answering.item_boundary import top_level_list_item_body_span
from app.domain.answering.markdown_table import (
    MarkdownTableParseError,
    scan_markdown_table_blocks,
)
from app.domain.answering.question import (
    ObligationKind,
    QuestionContract,
    QuestionObligation,
    RequiredOutputForm,
)
from app.domain.answering.witness import (
    EvidenceSpanRef,
    EvidenceWitness,
    SpanEquivalence,
    WitnessAtomKind,
    WitnessCatalogOmission,
    WitnessCatalogReceipt,
    WitnessOmissionCode,
    WitnessUniverseStatus,
)
from app.domain.query_planning import FacetClosure, QueryFacet
from app.domain.safe_representation import derive_safe_representation


_OBLIGATION_ID = "qf_" + "a" * 32
_EVIDENCE_ID = "ev_" + "b" * 40
_REPRESENTATION_ID = "rep_" + "c" * 40
_OTHER_EVIDENCE_ID = "ev_" + "e" * 40
_OTHER_REPRESENTATION_ID = "rep_" + "f" * 40
_CONTEXT_PACK_HASH = sha256_digest("enumeration-closure-context")
_AUTHORIZED_SCOPE_HASH = "scope_" + "d" * 64


@dataclass(frozen=True)
class _Artifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witness: EvidenceWitness
    canonical_text: str
    packed_text: str

    def build_receipt(
        self,
        *,
        catalog_texts: tuple[str, ...] | None = None,
    ) -> QueryAnchoredTableRowClosureReceipt:
        return QueryAnchoredTableRowClosureReceipt.build(
            question=self.question,
            catalog=self.catalog,
            obligation=self.obligation,
            witness=self.witness,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=self.canonical_text,
            packed_text=self.packed_text,
            catalog_texts=catalog_texts or (self.packed_text,),
        )


def _artifacts(
    *,
    row: str = "| ACME-42 | 双面打印 |",
    packed_row: str | None = None,
    source_query: str = "请列出 ACME 42 的全部功能",
    information_need: str = "ACME-42 的开放功能列表",
    closure: FacetClosure = FacetClosure.OPEN_ENUMERATION,
    atom_kind: WitnessAtomKind = WitnessAtomKind.TABLE_DATA_ROW,
    universe_status: WitnessUniverseStatus = WitnessUniverseStatus.COMPLETE,
    header_row: str = "| 标识 | 值 |",
    separator_row: str = "| --- | --- |",
    packed_header_row: str | None = None,
    packed_separator_row: str | None = None,
    rows_before: tuple[str, ...] = (),
    rows_after: tuple[str, ...] = (),
    packed_rows_before: tuple[str, ...] | None = None,
    packed_rows_after: tuple[str, ...] | None = None,
    document_prefix: str = "",
    packed_document_prefix: str | None = None,
    document_suffix: str = "",
    packed_document_suffix: str | None = None,
    row_prefix: str = "",
    row_suffix: str = "",
) -> _Artifacts:
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "1" * 32,
        source_query=source_query,
        facets=(
            QueryFacet(
                facet_id=_OBLIGATION_ID,
                information_need=information_need,
                retrieval_expression=information_need,
                closure=closure,
            ),
        ),
        builder_version="enumeration-closure-test.v1",
    )
    obligation = question.obligations[0]

    resolved_packed_row = packed_row if packed_row is not None else row
    resolved_packed_header = packed_header_row if packed_header_row is not None else header_row
    resolved_packed_separator = (
        packed_separator_row if packed_separator_row is not None else separator_row
    )
    resolved_packed_rows_before = (
        packed_rows_before if packed_rows_before is not None else rows_before
    )
    resolved_packed_rows_after = packed_rows_after if packed_rows_after is not None else rows_after
    resolved_packed_prefix = (
        packed_document_prefix if packed_document_prefix is not None else document_prefix
    )
    resolved_packed_suffix = (
        packed_document_suffix if packed_document_suffix is not None else document_suffix
    )

    def _with_line_boundary(value: str) -> str:
        if value and not value.endswith(("\n", "\r")):
            return value + "\n"
        return value

    canonical_prefix = _with_line_boundary(document_prefix)
    packed_prefix = _with_line_boundary(resolved_packed_prefix)
    canonical_lines = [
        header_row,
        separator_row,
        *rows_before,
        f"{row_prefix}{row}{row_suffix}",
        *rows_after,
    ]
    packed_lines = [
        resolved_packed_header,
        resolved_packed_separator,
        *resolved_packed_rows_before,
        f"{row_prefix}{resolved_packed_row}{row_suffix}",
        *resolved_packed_rows_after,
    ]
    canonical_text = canonical_prefix + "\n".join(canonical_lines) + document_suffix
    packed_text = packed_prefix + "\n".join(packed_lines) + resolved_packed_suffix
    canonical_start = (
        len(canonical_prefix)
        + sum(len(line) + 1 for line in canonical_lines[: 2 + len(rows_before)])
        + len(row_prefix)
    )
    packed_start = (
        len(packed_prefix)
        + sum(len(line) + 1 for line in packed_lines[: 2 + len(resolved_packed_rows_before)])
        + len(row_prefix)
    )
    equivalence = (
        SpanEquivalence.EXACT_UTF8
        if row == resolved_packed_row
        else SpanEquivalence.NFKC_EQUIVALENT
    )
    span_ref = EvidenceSpanRef.from_texts(
        evidence_id=_EVIDENCE_ID,
        representation_id=_REPRESENTATION_ID,
        canonical_text=canonical_text,
        packed_text=packed_text,
        canonical_start=canonical_start,
        canonical_end=canonical_start + len(row),
        packed_start=packed_start,
        packed_end=packed_start + len(resolved_packed_row),
        equivalence=equivalence,
    )
    witness = EvidenceWitness.build(
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        span_ref=span_ref,
        atom_kind=atom_kind,
        ordinal=0,
    )
    omissions = (
        ()
        if universe_status == WitnessUniverseStatus.COMPLETE
        else (
            WitnessCatalogOmission(
                code=WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED,
                count=1,
            ),
        )
    )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=(witness,),
        universe_status=universe_status,
        omissions=omissions,
        builder_version="enumeration-closure-test.v1",
    )
    return _Artifacts(
        question=question,
        obligation=obligation,
        catalog=catalog,
        witness=witness,
        canonical_text=canonical_text,
        packed_text=packed_text,
    )


def test_builds_immutable_receipt_and_replays_all_bound_artifacts() -> None:
    row = "| ＡＣＭＥ－４２ | 双面打印 |"
    artifacts = _artifacts(
        row=row,
        packed_row="| ACME-42 | 双面打印 |",
        rows_before=("| Other-1 | 单面打印 |",),
        rows_after=("| Other-2 | 彩色打印 |",),
    )

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION
    assert receipt.normalization_version == QUERY_ANCHOR_NORMALIZATION_VERSION
    assert receipt.receipt_id.startswith("qtrc_")
    assert receipt.question_contract_id == artifacts.question.contract_id
    assert receipt.question_contract_hash == artifacts.question.contract_hash
    assert receipt.catalog_id == artifacts.catalog.catalog_id
    assert receipt.catalog_hash == artifacts.catalog.catalog_hash
    assert receipt.obligation_id == artifacts.obligation.obligation_id
    assert receipt.obligation_hash == artifacts.obligation.obligation_hash
    assert receipt.witness_id == artifacts.witness.witness_id
    assert receipt.witness_hash == artifacts.witness.witness_hash
    assert receipt.evidence_id == _EVIDENCE_ID
    assert receipt.evidence_content_hash == artifacts.witness.span_ref.evidence_content_hash
    assert receipt.row_hash == sha256_digest(row)
    assert receipt.table_hash == sha256_digest(artifacts.canonical_text)
    assert receipt.packed_table_hash == sha256_digest(artifacts.packed_text)
    assert receipt.header_hash == sha256_digest("| 标识 | 值 |")
    assert receipt.separator_hash == sha256_digest("| --- | --- |")
    assert receipt.table_start == 0
    assert receipt.table_end == len(artifacts.canonical_text)
    assert receipt.table_column_count == 2
    assert receipt.table_data_row_count == 3
    assert receipt.witness_table_row_index == 1
    assert receipt.key_hash == sha256_digest("acme42")
    assert receipt.key_cell_index == 0
    assert receipt.key_occurrence_count == 1
    assert receipt.catalog_representation_count == 1
    assert receipt.catalog_key_mention_count == 1
    assert receipt.canonical_non_table_character_count == 0
    assert receipt.packed_non_table_character_count == 0
    assert receipt.anchor_field == "source_query"
    assert QueryAnchoredTableRowClosureReceipt.from_dict(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witness=artifacts.witness,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=artifacts.canonical_text,
            packed_text=artifacts.packed_text,
            catalog_texts=(artifacts.packed_text,),
        )
        is receipt
    )
    with pytest.raises(ValidationError, match="frozen"):
        receipt.row_hash = sha256_digest("forbidden")  # type: ignore[misc]


def test_uses_second_identifier_cell_and_information_need_deterministically() -> None:
    artifacts = _artifacts(
        row="| 001 | Widget-Z |",
        source_query="请列出对应条目",
        information_need="Widget Z 的开放能力列表",
        header_row="| 序号 | 型号 |",
    )

    receipt = artifacts.build_receipt()

    assert receipt.key_cell_index == 1
    assert receipt.anchor_field == "information_need"
    assert receipt.key_hash == sha256_digest("widgetz")


def test_rejects_a_query_match_in_a_non_identifier_value_column() -> None:
    artifacts = _artifacts(
        row="| Gamma | Beta |",
        source_query="Beta 有哪些功能？",
        information_need="Beta 的功能列表",
        header_row="| 对象 | 功能 |",
    )

    with pytest.raises(ValueError, match="explicit identifier header"):
        artifacts.build_receipt()


def test_identifier_header_matching_uses_tokens_not_ascii_suffix_substrings() -> None:
    artifacts = _artifacts(
        row="| Beta | 功能甲 |",
        source_query="Beta 有哪些功能？",
        information_need="Beta 的功能列表",
        header_row="| invalid | description |",
    )

    with pytest.raises(ValueError, match="explicit identifier header"):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    ("closure", "atom_kind", "universe_status", "message"),
    [
        (
            FacetClosure.SINGLE,
            WitnessAtomKind.TABLE_DATA_ROW,
            WitnessUniverseStatus.COMPLETE,
            "open_enumeration",
        ),
        (
            FacetClosure.OPEN_ENUMERATION,
            WitnessAtomKind.TABLE_ROW,
            WitnessUniverseStatus.COMPLETE,
            "table_data_row",
        ),
        (
            FacetClosure.OPEN_ENUMERATION,
            WitnessAtomKind.TABLE_DATA_ROW,
            WitnessUniverseStatus.TRUNCATED,
            "complete witness catalog",
        ),
    ],
)
def test_rejects_wrong_closure_atom_or_catalog_state(
    closure: FacetClosure,
    atom_kind: WitnessAtomKind,
    universe_status: WitnessUniverseStatus,
    message: str,
) -> None:
    artifacts = _artifacts(
        closure=closure,
        atom_kind=atom_kind,
        universe_status=universe_status,
    )

    with pytest.raises(ValueError, match=message):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    "row",
    [
        "ACME-42 双面打印",
        "| ACME-42 \\| Alias | 双面打印 |",
        "| `ACME-42` | 双面打印 |",
        "| ACME-42 |\n| 双面打印 |",
        "| --- | :---: |",
        "| ACME-42 | |",
    ],
)
def test_rejects_non_simple_or_non_data_markdown_rows(row: str) -> None:
    artifacts = _artifacts(row=row)

    with pytest.raises(
        ValueError,
        match="complete simple Markdown data row|exactly one complete physical line",
    ):
        artifacts.build_receipt()


def test_rejects_a_table_fragment_that_does_not_cover_the_physical_line() -> None:
    artifacts = _artifacts(row_prefix="前缀 ", row_suffix=" 后缀")

    with pytest.raises(ValueError, match="complete physical line"):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    ("row", "source_query", "information_need"),
    [
        ("| 001.0 | 名称 |", "001.0 名称", "编号和名称"),
        ("| Name | 002 |", "Name 002", "名称与编号"),
        ("| Header | 参数 |", "Header 参数", "表头与参数"),
        ("| 序号 | 类型 | ACME-42 |", "ACME-42", "ACME-42 的条目"),
        ("| Other-7 | Widget-Z |", "ACME-42", "ACME-42 的条目"),
    ],
)
def test_rejects_numeric_generic_third_column_or_unanchored_identifiers(
    row: str,
    source_query: str,
    information_need: str,
) -> None:
    artifacts = _artifacts(
        row=row,
        source_query=source_query,
        information_need=information_need,
        header_row=("| 标识 | 类型 | 值 |" if row.count("|") == 4 else "| 标识 | 值 |"),
        separator_row=("| --- | --- | --- |" if row.count("|") == 4 else "| --- | --- |"),
    )

    with pytest.raises(ValueError, match="first two table cells"):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    ("row", "source_query", "information_need"),
    [
        ("| A | 功能 |", "Show A options", "A options"),
        ("| ON | 功能 |", "What are all options?", "available options"),
        ("| PRO | 模式 |", "Please provide all available modes", "available modes"),
    ],
)
def test_ascii_identifier_requires_a_specific_full_token_match(
    row: str,
    source_query: str,
    information_need: str,
) -> None:
    artifacts = _artifacts(
        row=row,
        source_query=source_query,
        information_need=information_need,
    )

    with pytest.raises(ValueError, match="specific identifier anchored"):
        artifacts.build_receipt()


def test_rejects_a_row_when_both_identifier_cells_anchor_the_query() -> None:
    artifacts = _artifacts(
        row="| Device-A | Module-X |",
        source_query="What can Device A and Module X do?",
        information_need="Device A Module X capabilities",
    )

    with pytest.raises(ValueError, match="exactly one of the first two table cells"):
        artifacts.build_receipt()


def test_validate_artifacts_rejects_changed_evidence_snapshot() -> None:
    artifacts = _artifacts()
    receipt = artifacts.build_receipt()

    with pytest.raises(ValueError, match="supplied artifacts"):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witness=artifacts.witness,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=artifacts.canonical_text + "篡改",
            packed_text=artifacts.packed_text,
            catalog_texts=(artifacts.packed_text,),
        )


def test_deserialization_rejects_a_forged_bound_hash() -> None:
    receipt = _artifacts().build_receipt()
    forged = receipt.to_dict()
    forged["row_hash"] = sha256_digest("different row")

    with pytest.raises(ValidationError, match="receipt_hash does not match"):
        QueryAnchoredTableRowClosureReceipt.from_dict(forged)


def test_rejects_duplicate_query_key_in_the_containing_table() -> None:
    artifacts = _artifacts(rows_after=("| ACME-42 | 彩色打印 |",))

    with pytest.raises(ValueError, match="unique explicit table primary keys"):
        artifacts.build_receipt()


@pytest.mark.parametrize("continuation_key", ["同上", "同前", "same as above", "ditto", "〃", "—"])
def test_rejects_implicit_continuation_rows(continuation_key: str) -> None:
    artifacts = _artifacts(rows_after=(f"| {continuation_key} | 装订 |",))

    with pytest.raises(ValueError, match="explicit primary key in every table row"):
        artifacts.build_receipt()


@pytest.mark.parametrize("visual_key", ["Be\u00adta", "B**eta**", "Bеta"])
def test_visual_or_formatted_duplicate_keys_cannot_bypass_catalog_identity_scan(
    visual_key: str,
) -> None:
    artifacts = _artifacts(
        row="| Beta | 功能甲 |",
        rows_after=(f"| Gamma | {visual_key} 还支持功能乙 |",),
        source_query="Beta 有哪些功能？",
        information_need="Beta 的功能列表",
    )

    with pytest.raises(ValueError, match="exactly once across the complete catalog"):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    ("header_row", "separator_row", "message"),
    [
        ("不是表头", "| --- | --- |", "missing a simple header"),
        ("| 标识 | 值 |", "不是分隔行", "missing an adjacent separator row"),
        ("| 标识 | 值 |", "| --- | --- | --- |", "different column count"),
    ],
)
def test_rejects_missing_or_inconsistent_table_authority(
    header_row: str,
    separator_row: str,
    message: str,
) -> None:
    artifacts = _artifacts(header_row=header_row, separator_row=separator_row)

    with pytest.raises(ValueError, match=message):
        artifacts.build_receipt()


def test_rejects_canonical_and_packed_table_structure_drift() -> None:
    artifacts = _artifacts(
        packed_header_row="| 标识 | 其他值 |",
    )

    with pytest.raises(ValueError, match="structures differ"):
        artifacts.build_receipt()


def test_rejects_adjacent_markdown_tables_without_a_non_table_boundary() -> None:
    artifacts = _artifacts(
        rows_after=(
            "| Other-1 | 单面打印 |",
            "| --- | --- |",
            "| Other-2 | 彩色打印 |",
        )
    )

    with pytest.raises(ValueError, match="require a non-table boundary"):
        artifacts.build_receipt()


def test_rejects_a_preceding_adjacent_table_without_a_non_table_boundary() -> None:
    artifacts = _artifacts(
        row="| Beta | 功能新 |",
        source_query="Beta 有哪些功能？",
        information_need="Beta 的功能列表",
        document_prefix=("| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能旧 |"),
    )

    with pytest.raises(ValueError, match="require a non-table boundary"):
        artifacts.build_receipt()


def test_rejects_non_table_content_in_the_selected_representation() -> None:
    artifacts = _artifacts(document_suffix="\n补充：ACME-42 还支持装订。")

    with pytest.raises(ValueError, match="cover the full representation"):
        artifacts.build_receipt()


def test_rejects_the_same_key_in_another_catalog_representation() -> None:
    artifacts = _artifacts()

    with pytest.raises(ValueError, match="exactly one complete catalog representation"):
        artifacts.build_receipt(
            catalog_texts=(
                artifacts.packed_text,
                "补充：ACME-42 还支持装订。",
            )
        )


def test_rejects_even_an_unrelated_second_catalog_representation_without_entity_binding() -> None:
    artifacts = _artifacts()

    with pytest.raises(ValueError, match="exactly one complete catalog representation"):
        artifacts.build_receipt(
            catalog_texts=(
                artifacts.packed_text,
                "Widget-Z 的维护周期为一年。",
            )
        )


def test_rejects_a_single_catalog_representation_not_equal_to_the_packed_table() -> None:
    artifacts = _artifacts()

    with pytest.raises(ValueError, match="exactly one complete catalog representation"):
        artifacts.build_receipt(catalog_texts=("Widget-Z 的维护周期为一年。",))


@dataclass(frozen=True)
class _ProcedureArtifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    canonical_text: str
    packed_text: str

    def build_receipt(self) -> QueryAnchoredProcedureListClosureReceipt:
        return QueryAnchoredProcedureListClosureReceipt.build(
            question=self.question,
            catalog=self.catalog,
            obligation=self.obligation,
            witnesses=self.witnesses,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=self.canonical_text,
            packed_text=self.packed_text,
            catalog_texts=(self.packed_text,),
        )


def _procedure_artifacts(
    text: str = ("打印产品报告步骤\n1. 打开控制面板。\n2. 选择打印信息报告。\n3. 按下确定键。"),
    *,
    selected_item_count: int | None = None,
) -> _ProcedureArtifacts:
    facet = QueryFacet(
        facet_id=_OBLIGATION_ID,
        information_need="打印产品报告步骤",
        retrieval_expression="打印产品报告步骤",
        closure=FacetClosure.OPEN_ENUMERATION,
    )
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "2" * 32,
        source_query="请说明全部打印产品报告步骤",
        facets=(facet,),
        builder_version="procedure-closure-test.v1",
        kind_by_obligation={_OBLIGATION_ID: ObligationKind.PROCEDURE},
        output_form_by_obligation={_OBLIGATION_ID: RequiredOutputForm.ORDERED_STEPS},
    )
    lines = text.splitlines()
    item_lines = [line for line in lines[1:] if line[:1].isdigit()]
    if selected_item_count is not None:
        item_lines = item_lines[:selected_item_count]
    witnesses: list[EvidenceWitness] = []
    search_start = len(lines[0]) + 1
    for ordinal, line in enumerate(item_lines):
        start = text.index(line, search_start)
        search_start = start + len(line)
        span_ref = EvidenceSpanRef.from_texts(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=text,
            packed_text=text,
            canonical_start=start,
            canonical_end=start + len(line),
            packed_start=start,
            packed_end=start + len(line),
            equivalence=SpanEquivalence.EXACT_UTF8,
        )
        witnesses.append(
            EvidenceWitness.build(
                context_pack_hash=_CONTEXT_PACK_HASH,
                authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                span_ref=span_ref,
                atom_kind=WitnessAtomKind.LIST_ITEM,
                ordinal=ordinal,
            )
        )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=tuple(witnesses),
        universe_status=WitnessUniverseStatus.COMPLETE,
        omissions=(),
        builder_version="procedure-closure-test.v1",
    )
    return _ProcedureArtifacts(
        question=question,
        obligation=question.obligations[0],
        catalog=catalog,
        witnesses=tuple(witnesses),
        canonical_text=text,
        packed_text=text,
    )


def test_builds_and_replays_complete_query_anchored_procedure_list() -> None:
    artifacts = _procedure_artifacts()

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
    assert receipt.witness_ids == tuple(item.witness_id for item in artifacts.witnesses)
    assert receipt.witness_hashes == tuple(item.witness_hash for item in artifacts.witnesses)
    assert receipt.witness_ordinals == (0, 1, 2)
    assert receipt.item_ordinals == (1, 2, 3)
    assert receipt.marker_style == "arabic:."
    assert receipt.item_count == 3
    assert receipt.title_anchor_hash == sha256_digest("打印产品报告步骤")
    assert QueryAnchoredProcedureListClosureReceipt.from_dict(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=artifacts.canonical_text,
            packed_text=artifacts.packed_text,
            catalog_texts=(artifacts.packed_text,),
        )
        is receipt
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "打印产品报告步骤\n1. 打开控制面板。\n3. 按下确定键。",
            "ordinals must start at one",
        ),
        (
            "打印产品报告步骤\n2. 选择打印信息报告。\n1. 打开控制面板。",
            "ordinals must start at one",
        ),
        (
            "打印产品报告步骤\n1. 打开控制面板。\n2. 按下确定键。\n补充说明。",
            "permits only ordered list items",
        ),
    ],
)
def test_procedure_list_rejects_gaps_disorder_and_trailing_content(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _procedure_artifacts(text).build_receipt()


def test_procedure_list_rejects_an_omitted_trailing_source_item() -> None:
    artifacts = _procedure_artifacts(selected_item_count=2)

    with pytest.raises(ValueError, match="bind every source item witness"):
        artifacts.build_receipt()


def test_procedure_list_rejects_receipt_hash_and_content_drift() -> None:
    artifacts = _procedure_artifacts()
    receipt = artifacts.build_receipt()
    forged = receipt.to_dict()
    forged["canonical_item_hashes"][1] = sha256_digest("tampered")

    with pytest.raises(ValidationError, match="receipt_hash does not match"):
        QueryAnchoredProcedureListClosureReceipt.from_dict(forged)
    with pytest.raises(ValueError):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=artifacts.canonical_text + "篡改",
            packed_text=artifacts.packed_text,
            catalog_texts=(artifacts.packed_text,),
        )


@dataclass(frozen=True)
class _ScopedProcedureArtifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    authorities: tuple[ScopedListAuthorityRepresentation, ...]

    def build_receipt(
        self,
        *,
        witnesses: tuple[EvidenceWitness, ...] | None = None,
        authorities: tuple[ScopedListAuthorityRepresentation, ...] | None = None,
    ) -> QueryAnchoredScopedProcedureListClosureReceipt:
        return QueryAnchoredScopedProcedureListClosureReceipt.build(
            question=self.question,
            catalog=self.catalog,
            obligation=self.obligation,
            witnesses=witnesses if witnesses is not None else self.witnesses,
            authority_representations=(
                authorities if authorities is not None else self.authorities
            ),
        )


def _scoped_procedure_artifacts(
    selected_text: str = (
        "本节介绍热点功能。\n\n"
        "标题路径：前言 > 4.2.2.2. Wi-Fi热点连接方法\n\n"
        "1.打开打印机电源。\n\n"
        "2.按下菜单键并开启Wi-Fi热点。\n\n"
        "3.开启计算机或移动设备。\n\n"
        "附录：术语说明。"
    ),
    *,
    other_text: str = "标题路径：前言 > 网络术语\n\nSSID 是无线网络名称。",
    selected_item_count: int | None = None,
    information_need: str = "设置打印机Wi-Fi热点的操作步骤",
    source_query: str = "设置打印机Wi-Fi热点时需要完成哪些操作步骤？",
    include_interstitial_witnesses: bool = False,
    packed_selected_text: str | None = None,
    additional_fact_need: str | None = None,
) -> _ScopedProcedureArtifacts:
    facet = QueryFacet(
        facet_id=_OBLIGATION_ID,
        information_need=information_need,
        retrieval_expression=information_need,
        closure=FacetClosure.OPEN_ENUMERATION,
    )
    additional_facet = (
        QueryFacet(
            facet_id="qf_" + "9" * 32,
            information_need=additional_fact_need,
            retrieval_expression=additional_fact_need,
            closure=FacetClosure.SINGLE,
        )
        if additional_fact_need is not None
        else None
    )
    facets = (facet,) if additional_facet is None else (additional_facet, facet)
    kind_by_obligation = {_OBLIGATION_ID: ObligationKind.PROCEDURE}
    output_form_by_obligation = {
        _OBLIGATION_ID: RequiredOutputForm.ORDERED_STEPS,
    }
    if additional_facet is not None:
        kind_by_obligation[additional_facet.facet_id] = ObligationKind.FACT
        output_form_by_obligation[additional_facet.facet_id] = RequiredOutputForm.FACT
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "3" * 32,
        source_query=source_query,
        facets=facets,
        builder_version="scoped-procedure-closure-test.v1",
        kind_by_obligation=kind_by_obligation,
        output_form_by_obligation=output_form_by_obligation,
    )
    if include_interstitial_witnesses:
        item_lines = [line for line in selected_text.splitlines() if line.strip()]
    else:
        item_lines = [
            line
            for line in selected_text.splitlines()
            if top_level_list_item_body_span(line) is not None
        ]
    if selected_item_count is not None:
        item_lines = item_lines[:selected_item_count]
    resolved_packed_text = packed_selected_text or selected_text
    witnesses: list[EvidenceWitness] = []
    canonical_search_start = 0
    packed_search_start = 0
    for ordinal, line in enumerate(item_lines):
        canonical_start = selected_text.index(line, canonical_search_start)
        canonical_search_start = canonical_start + len(line)
        packed_start = resolved_packed_text.index(line, packed_search_start)
        packed_search_start = packed_start + len(line)
        span_ref = EvidenceSpanRef.from_texts(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=selected_text,
            packed_text=resolved_packed_text,
            canonical_start=canonical_start,
            canonical_end=canonical_start + len(line),
            packed_start=packed_start,
            packed_end=packed_start + len(line),
            equivalence=SpanEquivalence.EXACT_UTF8,
        )
        witnesses.append(
            EvidenceWitness.build(
                context_pack_hash=_CONTEXT_PACK_HASH,
                authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                span_ref=span_ref,
                atom_kind=(
                    WitnessAtomKind.LIST_ITEM
                    if top_level_list_item_body_span(line) is not None
                    else WitnessAtomKind.MEDIA_DERIVATION
                    if line.lstrip().startswith("![")
                    else WitnessAtomKind.SENTENCE
                ),
                ordinal=ordinal,
            )
        )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=tuple(witnesses),
        universe_status=WitnessUniverseStatus.COMPLETE,
        omissions=(),
        builder_version="scoped-procedure-closure-test.v1",
    )
    return _ScopedProcedureArtifacts(
        question=question,
        obligation=next(
            item for item in question.obligations if item.obligation_id == _OBLIGATION_ID
        ),
        catalog=catalog,
        witnesses=tuple(witnesses),
        authorities=(
            ScopedListAuthorityRepresentation(
                evidence_id=_EVIDENCE_ID,
                representation_id=_REPRESENTATION_ID,
                canonical_text=selected_text,
                packed_text=resolved_packed_text,
            ),
            ScopedListAuthorityRepresentation(
                evidence_id=_OTHER_EVIDENCE_ID,
                representation_id=_OTHER_REPRESENTATION_ID,
                canonical_text=other_text,
                packed_text=other_text,
            ),
        ),
    )


def test_builds_and_replays_unique_catalog_wide_scoped_procedure() -> None:
    artifacts = _scoped_procedure_artifacts()

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
    assert receipt.catalog_representation_count == 2
    assert receipt.catalog_evidence_ids == (_EVIDENCE_ID, _OTHER_EVIDENCE_ID)
    assert receipt.catalog_representation_ids == (
        _REPRESENTATION_ID,
        _OTHER_REPRESENTATION_ID,
    )
    assert receipt.selected_authority_index == 0
    assert receipt.authority_candidate_count == 1
    assert receipt.canonical_block_start > 0
    assert receipt.canonical_non_block_character_count > 0
    assert receipt.item_ordinals == (1, 2, 3)
    assert receipt.marker_style == "arabic:."
    assert receipt.selection_kind == "complete_sequence"
    assert receipt.anchor_match_key_hashes
    assert receipt.witness_ids == tuple(item.witness_id for item in artifacts.witnesses)
    assert parse_source_enumeration_closure_receipt(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        is receipt
    )


def test_scoped_procedure_rejects_missing_and_ambiguous_authority() -> None:
    artifacts = _scoped_procedure_artifacts()
    with pytest.raises(ValueError, match="complete raw authority"):
        artifacts.build_receipt(authorities=())

    duplicate_text = "标题路径：附录 > Wi-Fi热点连接方法\n1.打开设备。\n2.启用Wi-Fi热点。"
    ambiguous_authorities = (
        artifacts.authorities[0],
        ScopedListAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text=duplicate_text,
            packed_text=duplicate_text,
        ),
    )
    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build_receipt(authorities=ambiguous_authorities)


@pytest.mark.parametrize(
    ("selected_text", "message"),
    [
        (
            "标题路径：前言 > Wi-Fi热点连接方法\n1.打开设备。\n3.启用热点。",
            "exactly one catalog-wide candidate",
        ),
        (
            "标题路径：前言 > Wi-Fi热点连接方法\n1.打开设备。\n1.启用热点。",
            "exactly one catalog-wide candidate",
        ),
        (
            "标题路径：前言 > Wi-Fi热点连接方法\n2.启用热点。\n1.打开设备。",
            "exactly one catalog-wide candidate",
        ),
        (
            "标题路径：前言 > Wi-Fi热点连接方法\n1.打开设备。\n2.启用热点等。",
            "open-ended enumeration cue",
        ),
        (
            "标题路径：前言 > Wi-Fi热点连接方法\n1.打开设备。\n2.查阅第4章继续设置。",
            "open cross-reference",
        ),
    ],
)
def test_scoped_procedure_rejects_non_closed_sequences(
    selected_text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _scoped_procedure_artifacts(selected_text).build_receipt()


def test_scoped_procedure_accepts_owned_media_note_and_parenthetical_reference() -> None:
    selected_text = (
        "标题路径：前言 > Wi-Fi热点连接方法\n"
        "1.打开打印机电源。\n"
        "![热点设置界面](images/hotspot.png)\n"
        "屏幕显示热点设置入口。\n"
        "2.登录Web服务器（如何登录，参阅网络设置章节）。\n"
        "注：确认无线指示灯保持点亮。\n"
        "3.保存设置并重新启动打印机。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        include_interstitial_witnesses=True,
    )

    receipt = artifacts.build_receipt()

    assert receipt.item_ordinals == (1, 2, 3)
    assert receipt.witness_ordinals == (1, 4, 6)
    assert receipt.selection_kind == "complete_sequence"


def test_scoped_procedure_pairs_replayed_image_omissions_by_source_order() -> None:
    canonical_text = (
        "标题路径：维护 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "![网络设置界面](images/network.png)\n"
        "界面中显示网络设置入口。\n"
        "2.选择网络连接设置。\n"
        "3.保存设置并重新启动设备。"
    )
    packed_text = derive_safe_representation(canonical_text).safe_representation
    artifacts = _scoped_procedure_artifacts(
        canonical_text,
        information_need="Acme Z9000网络连接的完整操作步骤",
        source_query="Acme Z9000网络连接需要执行哪些操作步骤？",
        packed_selected_text=packed_text,
    )

    receipt = artifacts.build_receipt()

    assert receipt.item_ordinals == (1, 2, 3)
    assert receipt.packed_item_starts[1] < receipt.canonical_item_starts[1]


def test_scoped_procedure_does_not_treat_unproved_placeholder_as_media() -> None:
    text = (
        "标题路径：维护 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "[image omitted]\n"
        "2.选择网络连接设置。\n"
        "3.保存设置并重新启动设备。"
    )
    artifacts = _scoped_procedure_artifacts(
        text,
        information_need="Acme Z9000网络连接的完整操作步骤",
        source_query="Acme Z9000网络连接需要执行哪些操作步骤？",
    )

    with pytest.raises(ValueError, match="replayed safe derivation"):
        artifacts.build_receipt()


def test_scoped_procedure_rejects_reordered_image_placeholder() -> None:
    canonical_text = (
        "标题路径：维护 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "![网络设置界面](images/network.png)\n"
        "2.选择网络连接设置。\n"
        "3.保存设置并重新启动设备。"
    )
    derived_text = derive_safe_representation(canonical_text).safe_representation
    packed_text = derived_text.replace(
        "[image omitted]\n2.选择网络连接设置。",
        "2.选择网络连接设置。\n[image omitted]",
    )
    artifacts = _scoped_procedure_artifacts(
        canonical_text,
        information_need="Acme Z9000网络连接的完整操作步骤",
        source_query="Acme Z9000网络连接需要执行哪些操作步骤？",
        packed_selected_text=packed_text,
    )

    with pytest.raises(ValueError, match="replayed safe derivation"):
        artifacts.build_receipt()


def test_scoped_procedure_ignores_only_a_less_specific_blocked_candidate() -> None:
    selected_text = (
        "标题路径：网络 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "2.选择Acme Z9000网络连接设置。\n"
        "3.保存设置并重新启动设备。"
    )
    lower_ranked_blocker = (
        "标题路径：附录 > 备用网络连接方法\n"
        "1.打开备用网络设置。\n"
        "2.检查网络连接状态。\n"
        "4.保存备用设置。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        other_text=lower_ranked_blocker,
        information_need="Acme Z9000网络连接的完整操作步骤",
        source_query="Acme Z9000网络连接需要执行哪些操作步骤？",
    )

    receipt = artifacts.build_receipt()

    assert receipt.selected_authority_index == 0
    assert receipt.item_ordinals == (1, 2, 3)


def test_scoped_procedure_rejects_an_equally_specific_blocked_candidate() -> None:
    selected_text = (
        "标题路径：网络 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "2.选择Acme Z9000网络连接设置。\n"
        "3.保存设置并重新启动设备。"
    )
    tied_blocker = (
        "标题路径：网络 > Acme Z9000网络连接方法\n"
        "1.打开Acme Z9000设备电源。\n"
        "2.选择Acme Z9000网络连接设置。\n"
        "4.保存设置并重新启动设备。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        other_text=tied_blocker,
        information_need="Acme Z9000网络连接的完整操作步骤",
        source_query="Acme Z9000网络连接需要执行哪些操作步骤？",
    )

    with pytest.raises(ValueError, match="equally or more specific blocked candidate"):
        artifacts.build_receipt()


def test_scoped_procedure_prefers_a_long_ascii_entity_over_generic_overlap() -> None:
    selected_text = (
        "标题路径：网络 > PantumP3500DWSeries网络部署\n"
        "1.打开打印机电源。\n"
        "2.选择网络部署选项。\n"
        "3.保存网络设置。"
    )
    generic_candidate = (
        "标题路径：网络 > 无线网络连接配置说明\n"
        "1.打开无线网络配置。\n"
        "2.选择网络连接选项。\n"
        "3.保存无线配置。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        other_text=generic_candidate,
        information_need="PantumP3500DWSeries无线网络连接配置步骤",
        source_query="PantumP3500DWSeries无线网络连接配置步骤是什么？",
    )

    receipt = artifacts.build_receipt()

    assert receipt.selected_authority_index == 0


def test_scoped_procedure_does_not_promote_a_two_key_generic_branch() -> None:
    selected_text = (
        "标题路径：维护 > 标准纸盒卡纸处理步骤\n"
        "1.完全抽出标准纸盒。\n"
        "2.取出标准纸盒中的卡纸。\n"
        "3.重新装入标准纸盒。"
    )
    optional_branch = (
        "标题路径：维护 > 选配纸盒卡纸处理步骤\n"
        "1.完全抽出选配纸盒。\n"
        "2.取出选配纸盒中的纸张。\n"
        "3.重新装入选配纸盒。\n"
        "如出现选配纸盒卡纸，请按以下步骤处理：\n"
        "1.打开设备后盖。\n"
        "2.缓慢取出后部纸张。\n"
        "3.关闭设备后盖。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        other_text=optional_branch,
        information_need="标准纸盒发生卡纸时应执行哪些操作？",
        source_query="标准纸盒发生卡纸时应执行哪些操作？",
    )

    receipt = artifacts.build_receipt()

    assert receipt.selected_authority_index == 0
    assert receipt.selection_kind == "complete_sequence"


def test_scoped_procedure_selects_an_explicit_query_bounded_prefix() -> None:
    selected_text = (
        "标题路径：安装 > USB安装\n"
        "1.使用USB线连接打印机和计算机并打开电源。\n"
        "2.插入安装光盘并运行Autorun.exe安装程序。\n"
        "3.阅读许可协议、选择安装语言并点击安装。\n"
        "4.系统自动安装驱动。\n"
        "5.安装完成后关闭安装界面。"
    )
    query = "使用USB线连接Acme Z9000打印机时，安装驱动前需完成哪些操作？"
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need=query,
        source_query=query,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selection_kind == "query_selected_prefix"
    assert receipt.item_ordinals == (1, 2, 3)
    assert receipt.prefix_trigger_hash == sha256_digest("安装驱动前")
    assert receipt.excluded_successor_ordinal == 4
    assert receipt.excluded_successor_hash == sha256_digest("系统自动安装驱动。")
    assert receipt.boundary_rule_version == "explicit-query-before-successor.v1"


def test_scoped_procedure_prefix_requires_one_numbered_phase_boundary() -> None:
    selected_text = (
        "标题路径：安装 > USB安装\n"
        "1.使用USB线连接打印机和计算机并打开电源。\n"
        "2.插入安装光盘并运行Autorun.exe安装程序。\n"
        "安装驱动将在稍后开始。\n"
        "3.阅读许可协议并选择安装语言。\n"
        "4.点击安装按钮。"
    )
    query = "使用USB线连接Acme Z9000打印机时，安装驱动前需完成哪些操作？"
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need=query,
        source_query=query,
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build_receipt()


def test_scoped_procedure_consumes_a_strict_owned_optional_note_block() -> None:
    selected_text = (
        "标题路径：网络 > Wi-Fi热点连接方法\n"
        "1.打开打印机电源。\n"
        "2.在网络设置中开启Wi-Fi热点。\n"
        "3.打印网络配置页并查看热点SSID和密码。\n"
        "注：• Wi-Fi热点默认为关闭。\n"
        "• 您还可以通过内嵌Web服务器打开Wi-Fi热点（参阅网络章节）。\n"
        "4.开启计算机或移动设备。"
    )
    artifacts = _scoped_procedure_artifacts(selected_text)

    receipt = artifacts.build_receipt()

    assert receipt.item_ordinals == (1, 2, 3, 4)
    assert receipt.selection_kind == "complete_sequence"


def test_scoped_procedure_rejects_a_required_action_hidden_in_a_note_block() -> None:
    selected_text = (
        "标题路径：网络 > Wi-Fi热点连接方法\n"
        "1.打开打印机电源。\n"
        "2.在网络设置中开启Wi-Fi热点。\n"
        "3.打印网络配置页并查看热点SSID和密码。\n"
        "注：• Wi-Fi热点默认为关闭。\n"
        "• 必须重新启动路由器并再次开启Wi-Fi热点。\n"
        "4.开启计算机或移动设备。"
    )
    artifacts = _scoped_procedure_artifacts(selected_text)

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build_receipt()


def test_scoped_procedure_accepts_owned_gui_echoes_with_one_to_one_media() -> None:
    canonical_text = (
        "标题路径：系统 > 省墨设置步骤\n"
        "1)按菜单键进入设置界面。\n"
        "2)按OK键选择“1.系统设置”选项。\n"
        "菜单设置1.系统设置\n"
        "![系统设置界面](images/system.png)\n"
        "3)按方向键选择“3.省墨”选项。\n"
        "系统设置3.省墨\n"
        "4)选择“1.关闭”或“2.开启”并按OK键。\n"
        "![省墨状态界面](images/toner-save.png)\n"
        "5)按菜单键退出设置界面。"
    )
    packed_text = derive_safe_representation(canonical_text).safe_representation
    query = "在省墨设置中如何开启或关闭省墨模式？请说明具体操作步骤。"
    artifacts = _scoped_procedure_artifacts(
        canonical_text,
        information_need=query,
        source_query=query,
        packed_selected_text=packed_text,
    )

    receipt = artifacts.build_receipt()

    assert receipt.item_ordinals == (1, 2, 3, 4, 5)
    assert receipt.packed_item_starts[-1] < receipt.canonical_item_starts[-1]


def test_scoped_procedure_gui_echo_requires_a_proved_image_and_matching_state() -> None:
    canonical_text = (
        "标题路径：系统 > 省墨设置步骤\n"
        "1)按菜单键进入设置界面。\n"
        "2)按OK键选择“1.系统设置”选项。\n"
        "菜单设置1.系统设置\n"
        "![系统设置界面](images/system.png)\n"
        "3)按方向键选择“3.省墨”选项。\n"
        "系统设置2.省墨\n"
        "4)选择“1.关闭”或“2.开启”并按OK键。\n"
        "![省墨状态界面](images/toner-save.png)\n"
        "5)按菜单键退出设置界面。"
    )
    packed_text = derive_safe_representation(canonical_text).safe_representation
    query = "在省墨设置中如何开启或关闭省墨模式？请说明具体操作步骤。"
    mismatch = _scoped_procedure_artifacts(
        canonical_text,
        information_need=query,
        source_query=query,
        packed_selected_text=packed_text,
    )
    unproved = _scoped_procedure_artifacts(
        canonical_text,
        information_need=query,
        source_query=query,
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        mismatch.build_receipt()
    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        unproved.build_receipt()


def test_scoped_procedure_does_not_cross_an_earlier_obligation_facet() -> None:
    selected_text = (
        "标题路径：维护 > 双面单元卡纸步骤\n"
        "1.取出双面打印单元。\n"
        "2.从双面打印单元中取出卡纸。\n"
        "3.如果纸张未随单元出来，请从底部取出卡纸。\n"
        "4.取出卡纸后，装回双面打印单元并检查其他部位。"
    )
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need="取出卡纸后还需进行哪些检查和操作",
        source_query=(
            "如果双面打印单元卡纸且纸张未随其一起出来，应如何操作？"
            "取出卡纸后还需进行哪些检查和操作？"
        ),
        additional_fact_need="如果纸张未随双面打印单元一起出来，应如何操作",
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build_receipt()


def test_scoped_procedure_selects_query_anchored_continuous_suffix() -> None:
    selected_text = (
        "标题路径：维护 > 清除卡纸步骤\n"
        "1.打开设备前盖。\n"
        "2.检查纸张所在位置。\n"
        "3.如果纸张可见，请取出卡纸。\n"
        "4.重新装入纸张并关闭前盖。"
    )
    query = "如果纸张可见，取出卡纸后还要执行哪些清除卡纸步骤？"
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need=query,
        source_query=query,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selection_kind == "query_selected_continuation"
    assert receipt.item_ordinals == (3, 4)
    assert receipt.anchor_match_key_hashes


def test_scoped_procedure_selects_explicit_conditional_branch_after_reset() -> None:
    selected_text = (
        "标题路径：维护 > 清除卡纸步骤\n"
        "1.打开设备前盖。\n"
        "2.取出可见纸张。\n"
        "3.关闭设备前盖。\n"
        "如果在该区域看不到纸张，请停止并转至下一步：\n"
        "1.打开设备后盖。\n"
        "2.缓慢取出后部卡纸。\n"
        "3.关闭设备后盖。"
    )
    query = "如果在前盖区域看不到纸张，后续应执行哪些清除卡纸步骤？"
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need=query,
        source_query=query,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selection_kind == "query_selected_branch"
    assert receipt.item_ordinals == (1, 2, 3)


def test_scoped_procedure_accepts_declared_finite_sublist_inside_suffix() -> None:
    selected_text = (
        "标题路径：网络 > 配置无线网络步骤\n"
        "4.输入无线网络地址。\n"
        "- 支持三种安全模式：\n"
        "1)选择WPA安全模式。\n"
        "2)选择WEP安全模式。\n"
        "3)选择开放安全模式。\n"
        "5.保存无线网络设置。\n"
        "6.重新启动网络设备。"
    )
    query = "输入无线网络地址后如何完成配置无线网络步骤？"
    artifacts = _scoped_procedure_artifacts(
        selected_text,
        information_need=query,
        source_query=query,
        include_interstitial_witnesses=True,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selection_kind == "query_selected_continuation"
    assert receipt.item_ordinals == (4, 5, 6)
    assert receipt.witness_ordinals == (1, 6, 7)


def test_scoped_procedure_rejects_truncation_and_unbound_tail_item() -> None:
    truncated = _scoped_procedure_artifacts(
        "标题路径：前言 > Wi-Fi热点连接方法\n"
        "1.打开设备。\n"
        "2.启用热点。\n"
        "…[证据正文因上下文预算被截断]"
    )
    with pytest.raises(ValueError, match="authority is truncated"):
        truncated.build_receipt()

    omitted = _scoped_procedure_artifacts(selected_item_count=2)
    with pytest.raises(ValueError, match="bind every source item witness once"):
        omitted.build_receipt()


def test_scoped_procedure_replay_rejects_unselected_authority_drift() -> None:
    artifacts = _scoped_procedure_artifacts()
    receipt = artifacts.build_receipt()
    drifted_authorities = (
        artifacts.authorities[0],
        ScopedListAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text="标题路径：前言 > 网络术语\nSSID 已被改写。",
            packed_text="标题路径：前言 > 网络术语\nSSID 已被改写。",
        ),
    )

    with pytest.raises(ValueError, match="does not match supplied artifacts"):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=drifted_authorities,
        )


def test_no_space_ordered_marker_is_supported_without_accepting_section_numbers() -> None:
    assert top_level_list_item_body_span("1.打开打印机。") == (2, 8)
    assert top_level_list_item_body_span("2)点击下一步。") == (2, 8)
    assert top_level_list_item_body_span("1.2 纸张规格") is None
    assert top_level_list_item_body_span("1.2.3 is a version") is None


@dataclass(frozen=True)
class _ScopedTableArtifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    authorities: tuple[ScopedTableAuthorityRepresentation, ...]

    def build_receipt(
        self,
        *,
        witnesses: tuple[EvidenceWitness, ...] | None = None,
        authorities: tuple[ScopedTableAuthorityRepresentation, ...] | None = None,
        catalog: WitnessCatalogReceipt | None = None,
    ) -> QueryAnchoredScopedTableSelectionClosureReceipt:
        return QueryAnchoredScopedTableSelectionClosureReceipt.build(
            question=self.question,
            catalog=catalog if catalog is not None else self.catalog,
            obligation=self.obligation,
            witnesses=witnesses if witnesses is not None else self.witnesses,
            authority_representations=(
                authorities if authorities is not None else self.authorities
            ),
        )


_HOTSPOT_TABLE = (
    "| 热点名称 | 信息组 | 加密方式 | 信号强度 |\n"
    "| --- | --- | --- | --- |\n"
    "| Test-AP1 | 常规信息 | WPA/WPA2 | 80% |\n"
    "| Test-AP2 | 无线网络信息 | WPA/WPA2 | 60% |\n"
    "| Test-AP3 | 无线网络信息 | WEP | 60% |\n"
    "| Test-AP4 | 访客信息 | 无加密 | 40% |"
)


def _scoped_table_artifacts(
    *,
    information_need: str,
    source_query: str | None = None,
    table: str = _HOTSPOT_TABLE,
    packed_table: str | None = None,
    other_text: str = "标题路径：附录 > 网络术语\nSSID 表示无线网络名称。",
    selected_suffix: str = "附注：数据快照。",
) -> _ScopedTableArtifacts:
    resolved_source_query = source_query or information_need
    facet = QueryFacet(
        facet_id=_OBLIGATION_ID,
        information_need=information_need,
        retrieval_expression=information_need,
        closure=FacetClosure.OPEN_ENUMERATION,
    )
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "4" * 32,
        source_query=resolved_source_query,
        facets=(facet,),
        builder_version="scoped-table-selection-test.v1",
        kind_by_obligation={_OBLIGATION_ID: ObligationKind.ENUMERATION},
        output_form_by_obligation={_OBLIGATION_ID: RequiredOutputForm.LIST},
    )
    canonical_text = "标题路径：网络信息 > 热点列表\n" + table + "\n" + selected_suffix
    resolved_packed_table = packed_table if packed_table is not None else table
    packed_text = "标题路径：网络信息 > 热点列表\n" + resolved_packed_table + "\n" + selected_suffix

    def authority_data_rows(value: str, raw_table: str) -> list[str]:
        try:
            return [
                row.line.text
                for block in scan_markdown_table_blocks(value)
                for row in block.data_rows
            ]
        except MarkdownTableParseError:
            # 无效表 fixture 仍需形成可传入 build 的 catalog；生产扫描会 fail-closed。
            fallback = raw_table.splitlines()[2:]
            return fallback or [raw_table.splitlines()[0]]

    canonical_rows = authority_data_rows(canonical_text, table)
    packed_rows = authority_data_rows(packed_text, resolved_packed_table)
    witnesses: list[EvidenceWitness] = []
    canonical_cursor = 0
    packed_cursor = 0
    for ordinal, (canonical_row, packed_row) in enumerate(
        zip(canonical_rows, packed_rows, strict=True)
    ):
        canonical_start = canonical_text.index(canonical_row, canonical_cursor)
        packed_start = packed_text.index(packed_row, packed_cursor)
        canonical_cursor = canonical_start + len(canonical_row)
        packed_cursor = packed_start + len(packed_row)
        span_ref = EvidenceSpanRef.from_texts(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=canonical_text,
            packed_text=packed_text,
            canonical_start=canonical_start,
            canonical_end=canonical_start + len(canonical_row),
            packed_start=packed_start,
            packed_end=packed_start + len(packed_row),
            equivalence=(
                SpanEquivalence.EXACT_UTF8
                if canonical_row == packed_row
                else SpanEquivalence.NFKC_EQUIVALENT
            ),
        )
        witnesses.append(
            EvidenceWitness.build(
                context_pack_hash=_CONTEXT_PACK_HASH,
                authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                span_ref=span_ref,
                atom_kind=WitnessAtomKind.TABLE_DATA_ROW,
                ordinal=ordinal,
            )
        )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=tuple(witnesses),
        universe_status=WitnessUniverseStatus.COMPLETE,
        omissions=(),
        builder_version="scoped-table-selection-test.v1",
    )
    return _ScopedTableArtifacts(
        question=question,
        obligation=question.obligations[0],
        catalog=catalog,
        witnesses=tuple(witnesses),
        authorities=(
            ScopedTableAuthorityRepresentation(
                evidence_id=_EVIDENCE_ID,
                representation_id=_REPRESENTATION_ID,
                canonical_text=canonical_text,
                packed_text=packed_text,
            ),
            ScopedTableAuthorityRepresentation(
                evidence_id=_OTHER_EVIDENCE_ID,
                representation_id=_OTHER_REPRESENTATION_ID,
                canonical_text=other_text,
                packed_text=other_text,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("information_need", "expected_column", "expected_rows", "selector_count"),
    [
        ("信号强度为60%的全部热点记录", 3, (1, 2), 1),
        ("Test-AP2和Test-AP4的完整记录", 0, (1, 3), 2),
        ("无线网络信息组的全部成员", 1, (1, 2), 1),
        ("Test-AP3的完整记录", 0, (2,), 1),
    ],
)
def test_scoped_table_selection_covers_exact_values_in_any_column(
    information_need: str,
    expected_column: int,
    expected_rows: tuple[int, ...],
    selector_count: int,
) -> None:
    artifacts = _scoped_table_artifacts(information_need=information_need)

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION
    assert receipt.anchor_field == "information_need"
    assert receipt.selector_column_index == expected_column
    assert receipt.selected_row_indices == expected_rows
    assert receipt.selector_count == selector_count
    assert receipt.selected_row_count == len(expected_rows)
    assert receipt.table_parse_mode == "standard"
    assert receipt.catalog_canonical_selector_cell_counts == receipt.selector_occurrence_counts
    assert receipt.catalog_packed_selector_cell_counts == receipt.selector_occurrence_counts
    assert receipt.witness_ids == tuple(
        artifacts.witnesses[index].witness_id for index in expected_rows
    )
    assert receipt.catalog_representation_count == 2
    assert receipt.selected_authority_index == 0
    assert parse_source_enumeration_closure_receipt(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=tuple(artifacts.witnesses[index] for index in expected_rows),
            authority_representations=artifacts.authorities,
        )
        is receipt
    )


def test_scoped_table_selector_accepts_an_ordinal_prefixed_menu_label() -> None:
    table = (
        "| 一级菜单 | 二级菜单 | 三级菜单 |\n"
        "| --- | --- | --- |\n"
        "| 4.网络信息 | 1.有线网络信息 | 1.IP 地址 |\n"
        "| 4.网络信息 | 2.无线网络信息 | 1.连接状态 |\n"
        "| 4.网络信息 | 2.无线网络信息 | 2.IP 地址 |"
    )
    artifacts = _scoped_table_artifacts(
        information_need="无线网络信息的完整记录",
        table=table,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selector_column_index == 1
    assert receipt.selected_row_indices == (1, 2)


def test_scoped_table_selection_prefers_information_need_before_source_query() -> None:
    artifacts = _scoped_table_artifacts(
        information_need="无线网络信息组的全部成员",
        source_query="无线网络信息组中加密方式为WEP的热点有哪些？",
    )

    receipt = artifacts.build_receipt()

    assert receipt.anchor_field == "information_need"
    assert receipt.selector_column_index == 1
    assert receipt.selected_row_indices == (1, 2)


def test_scoped_table_selection_falls_back_to_source_query() -> None:
    artifacts = _scoped_table_artifacts(
        information_need="相关表格记录",
        source_query="请列出Test-AP3的完整记录",
    )

    receipt = artifacts.build_receipt()

    assert receipt.anchor_field == "source_query"
    assert receipt.selector_column_index == 0
    assert receipt.selected_row_indices == (2,)


def test_scoped_table_selection_rejects_missing_or_ambiguous_authority() -> None:
    artifacts = _scoped_table_artifacts(information_need="信号强度为60%的全部热点记录")
    with pytest.raises(ValueError, match="complete raw authority"):
        artifacts.build_receipt(authorities=())

    duplicate_table_text = "标题路径：附录 > 热点副本\n" + _HOTSPOT_TABLE
    ambiguous = (
        artifacts.authorities[0],
        ScopedTableAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text=duplicate_table_text,
            packed_text=duplicate_table_text,
        ),
    )
    with pytest.raises(ValueError, match="exactly one catalog-wide selector candidate"):
        artifacts.build_receipt(authorities=ambiguous)


def test_scoped_table_selection_rejects_ambiguous_column_and_duplicate_headers() -> None:
    ambiguous_columns = (
        "| 主键 | 分组 | 状态 |\n"
        "| --- | --- | --- |\n"
        "| 组甲 | 组甲 | 开启 |\n"
        "| 项目乙 | 组甲 | 关闭 |"
    )
    with pytest.raises(ValueError, match="exactly one catalog-wide selector candidate"):
        _scoped_table_artifacts(
            information_need="组甲的全部记录",
            table=ambiguous_columns,
        ).build_receipt()

    duplicate_headers = (
        "| 名称 | 名称 | 状态 |\n"
        "| --- | --- | --- |\n"
        "| 项目甲 | 组甲 | 开启 |\n"
        "| 项目乙 | 组甲 | 关闭 |"
    )
    with pytest.raises(ValueError, match="columns are not uniquely identified"):
        _scoped_table_artifacts(
            information_need="组甲的全部记录",
            table=duplicate_headers,
        ).build_receipt()


def test_scoped_table_selection_rejects_empty_partial_and_unbound_selection() -> None:
    empty = _scoped_table_artifacts(information_need="不存在的热点记录")
    with pytest.raises(ValueError, match="exactly one catalog-wide selector candidate"):
        empty.build_receipt()

    partial = _scoped_table_artifacts(information_need="信号强度为60的热点记录")
    with pytest.raises(ValueError, match="exactly one catalog-wide selector candidate"):
        partial.build_receipt()

    selected = _scoped_table_artifacts(information_need="信号强度为60%的全部热点记录")
    with pytest.raises(ValueError, match="bind every selected row witness once"):
        selected.build_receipt(witnesses=(selected.witnesses[1],))


def test_scoped_table_selection_rejects_truncation_omission_and_structure_drift() -> None:
    artifacts = _scoped_table_artifacts(information_need="Test-AP3的完整记录")
    truncated_authorities = (
        artifacts.authorities[0],
        ScopedTableAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text="网络附录。",
            packed_text="网络附录。\n…[证据正文因上下文预算被截断]",
        ),
    )
    with pytest.raises(ValueError, match="truncation marker"):
        artifacts.build_receipt(authorities=truncated_authorities)

    truncated_catalog = WitnessCatalogReceipt.build(
        question_contract_id=artifacts.question.contract_id,
        question_contract_hash=artifacts.question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=artifacts.witnesses,
        universe_status=WitnessUniverseStatus.TRUNCATED,
        omissions=(
            WitnessCatalogOmission(
                code=WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED,
                count=1,
            ),
        ),
        builder_version="scoped-table-selection-test.v1",
    )
    with pytest.raises(ValueError, match="complete witness catalog"):
        artifacts.build_receipt(catalog=truncated_catalog)

    drifted_packed_authorities = (
        ScopedTableAuthorityRepresentation(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=artifacts.authorities[0].canonical_text,
            packed_text=artifacts.authorities[0].packed_text.replace(
                "| Test-AP3 |", "| Test-AP9 |"
            ),
        ),
        artifacts.authorities[1],
    )
    with pytest.raises(ValueError, match="structures differ"):
        artifacts.build_receipt(authorities=drifted_packed_authorities)


def test_scoped_table_selection_replay_and_identity_reject_authority_drift() -> None:
    artifacts = _scoped_table_artifacts(information_need="Test-AP3的完整记录")
    receipt = artifacts.build_receipt()
    forged = receipt.to_dict()
    forged["selector_hashes"][0] = sha256_digest("tampered")
    with pytest.raises(ValidationError, match="selector_values_hash"):
        QueryAnchoredScopedTableSelectionClosureReceipt.from_dict(forged)

    drifted_authorities = (
        artifacts.authorities[0],
        ScopedTableAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text="标题路径：附录 > 网络术语\nSSID 含义已改变。",
            packed_text="标题路径：附录 > 网络术语\nSSID 含义已改变。",
        ),
    )
    with pytest.raises(ValueError, match="does not match supplied artifacts"):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=(artifacts.witnesses[2],),
            authority_representations=drifted_authorities,
        )


@pytest.mark.parametrize(
    ("table", "other_text"),
    [
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| Gamma | B**eta** 功能乙 |",
            "无关附录。",
        ),
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| Gamma | Be\u00adta 功能乙 |",
            "无关附录。",
        ),
    ],
)
def test_scoped_table_selector_mentions_are_confined_to_selected_cells(
    table: str,
    other_text: str,
) -> None:
    artifacts = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=table,
        other_text=other_text,
    )

    with pytest.raises(ValueError, match="exists outside selected accepted-table cells"):
        artifacts.build_receipt()


def test_scoped_table_selector_rejects_same_representation_supplement() -> None:
    table = "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |"
    artifacts = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=table,
        selected_suffix="补充：Beta 还支持功能乙。",
    )

    with pytest.raises(ValueError, match="selector supplement"):
        artifacts.build_receipt()


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| 同上 | 功能乙 |",
            "ditto",
        ),
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n|  | 功能乙 |",
            "exactly one catalog-wide selector candidate",
        ),
    ],
)
def test_scoped_table_selection_rejects_unowned_continuations(
    table: str,
    message: str,
) -> None:
    artifacts = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=table,
    )

    with pytest.raises(ValueError, match=message):
        artifacts.build_receipt()


def test_scoped_table_selection_rejects_local_and_catalog_wide_supplements() -> None:
    local = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table="| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |",
        selected_suffix="该型号还支持功能乙。",
    )
    with pytest.raises(ValueError, match="selector supplement"):
        local.build_receipt()

    unrelated = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table="| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |",
        other_text="补充：Beta 还支持功能乙。",
    )
    with pytest.raises(ValueError, match="selector supplement"):
        unrelated.build_receipt()


def test_scoped_table_selection_accepts_strict_ocr_table() -> None:
    ocr_table = (
        "表头：热点名称 | 信息组 | 加密方式 | 信号强度\n"
        "| Test-AP1 | 常规信息 | WPA/WPA2 | 80% |\n"
        "| --- | --- | --- | --- |\n"
        "| Test-AP2 | 无线网络信息 | WPA/WPA2 | 60% |\n"
        "| Test-AP3 | 无线网络信息 | WEP | 60% |\n"
        "| Test-AP4 | 访客信息 | 无加密 | 40% |"
    )
    artifacts = _scoped_table_artifacts(
        information_need="信号强度为60%的全部热点记录",
        table=ocr_table,
    )

    receipt = artifacts.build_receipt()

    assert receipt.table_parse_mode == "ocr_header_data_before_separator"
    assert receipt.selected_row_indices == (1, 2)


def test_scoped_table_selection_skips_header_separator_pseudo_table() -> None:
    pseudo_then_real = (
        "| 临时名称 | 临时状态 |\n"
        "| --- | --- |\n\n"
        "| 对象 | 功能 |\n"
        "| --- | --- |\n"
        "| Beta | 功能甲 |"
    )
    artifacts = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=pseudo_then_real,
    )

    receipt = artifacts.build_receipt()

    assert receipt.selected_table_index == 0
    assert receipt.table_data_row_count == 1


def test_scoped_table_selection_rejects_parse_mode_drift_and_two_table_selector() -> None:
    standard = "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| Gamma | 功能乙 |"
    ocr = "表头：对象 | 功能\n| Beta | 功能甲 |\n| --- | --- |\n| Gamma | 功能乙 |"
    mode_drift = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=standard,
        packed_table=ocr,
    )
    with pytest.raises(ValueError, match="structures differ"):
        mode_drift.build_receipt()

    two_tables = standard + "\n\n" + standard
    ambiguous = _scoped_table_artifacts(
        information_need="Beta的全部功能",
        table=two_tables,
    )
    with pytest.raises(ValueError, match="exactly one catalog-wide selector candidate"):
        ambiguous.build_receipt()


@dataclass(frozen=True)
class _ScopedBulletArtifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    authorities: tuple[ScopedBulletAuthorityRepresentation, ...]

    def build_receipt(
        self,
        *,
        witnesses: tuple[EvidenceWitness, ...] | None = None,
        authorities: tuple[ScopedBulletAuthorityRepresentation, ...] | None = None,
        catalog: WitnessCatalogReceipt | None = None,
    ) -> QueryAnchoredScopedBulletListClosureReceipt:
        return QueryAnchoredScopedBulletListClosureReceipt.build(
            question=self.question,
            catalog=catalog if catalog is not None else self.catalog,
            obligation=self.obligation,
            witnesses=witnesses if witnesses is not None else self.witnesses,
            authority_representations=(
                authorities if authorities is not None else self.authorities
            ),
        )


def _scoped_bullet_artifacts(
    selected_text: str = (
        "标题路径：前言 > 10. 打印故障\n\n"
        "在处理打印故障之前，请检查以下情况：\n\n"
        "• 电源连接是否正常。\n\n"
        "• 接口电缆是否连接正确。"
    ),
    *,
    information_need: str = "在处理打印故障前应检查哪些关键项目，请列出全部内容",
    source_query: str | None = None,
    packed_text: str | None = None,
    other_text: str = "标题路径：附录 > 网络术语\nSSID 表示无线网络名称。",
) -> _ScopedBulletArtifacts:
    resolved_source_query = source_query or information_need
    facet = QueryFacet(
        facet_id=_OBLIGATION_ID,
        information_need=information_need,
        retrieval_expression=information_need,
        closure=FacetClosure.OPEN_ENUMERATION,
    )
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "5" * 32,
        source_query=resolved_source_query,
        facets=(facet,),
        builder_version="scoped-bullet-list-test.v1",
        kind_by_obligation={_OBLIGATION_ID: ObligationKind.ENUMERATION},
        output_form_by_obligation={_OBLIGATION_ID: RequiredOutputForm.LIST},
    )
    resolved_packed_text = packed_text if packed_text is not None else selected_text

    def bullet_lines(value: str) -> list[str]:
        return [
            line
            for line in value.splitlines()
            if line[:1] in {"-", "*", "•", "+"} and top_level_list_item_body_span(line) is not None
        ]

    canonical_lines = bullet_lines(selected_text)
    packed_lines = bullet_lines(resolved_packed_text)
    if len(canonical_lines) != len(packed_lines):
        raise ValueError("test fixture canonical and packed bullet counts differ")
    witnesses: list[EvidenceWitness] = []
    canonical_cursor = 0
    packed_cursor = 0
    for ordinal, (canonical_line, packed_line) in enumerate(
        zip(canonical_lines, packed_lines, strict=True)
    ):
        canonical_start = selected_text.index(canonical_line, canonical_cursor)
        packed_start = resolved_packed_text.index(packed_line, packed_cursor)
        canonical_cursor = canonical_start + len(canonical_line)
        packed_cursor = packed_start + len(packed_line)
        span_ref = EvidenceSpanRef.from_texts(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=selected_text,
            packed_text=resolved_packed_text,
            canonical_start=canonical_start,
            canonical_end=canonical_start + len(canonical_line),
            packed_start=packed_start,
            packed_end=packed_start + len(packed_line),
            equivalence=(
                SpanEquivalence.EXACT_UTF8
                if canonical_line == packed_line
                else SpanEquivalence.NFKC_EQUIVALENT
            ),
        )
        witnesses.append(
            EvidenceWitness.build(
                context_pack_hash=_CONTEXT_PACK_HASH,
                authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                span_ref=span_ref,
                atom_kind=WitnessAtomKind.LIST_ITEM,
                ordinal=ordinal,
            )
        )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=tuple(witnesses),
        universe_status=WitnessUniverseStatus.COMPLETE,
        omissions=(),
        builder_version="scoped-bullet-list-test.v1",
    )
    return _ScopedBulletArtifacts(
        question=question,
        obligation=question.obligations[0],
        catalog=catalog,
        witnesses=tuple(witnesses),
        authorities=(
            ScopedBulletAuthorityRepresentation(
                evidence_id=_EVIDENCE_ID,
                representation_id=_REPRESENTATION_ID,
                canonical_text=selected_text,
                packed_text=resolved_packed_text,
            ),
            ScopedBulletAuthorityRepresentation(
                evidence_id=_OTHER_EVIDENCE_ID,
                representation_id=_OTHER_REPRESENTATION_ID,
                canonical_text=other_text,
                packed_text=other_text,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("selected_text", "information_need", "expected_kind", "expected_count"),
    [
        (
            "标题路径：网络诊断 > 设备离线检查清单：\n• 检查设备电源。\n• 检查网络连接。",
            "设备离线时应检查哪些项目，请列出全部内容",
            "bounded_title",
            2,
        ),
        (
            "标题路径：前言 > 设备维护注意事项\n"
            "14. 设备进水时若出现以下危险，请立即断电并联系专业维修人员：\n"
            "• 设备内部进水。\n"
            "• 设备外壳破损。\n"
            "15. 后续维护由专业人员完成。",
            "设备进水后哪些危险需要断电并联系维修人员，请列出全部情况",
            "ordered_closed_intro",
            2,
        ),
        (
            "标题路径：前言 > 10. 打印故障\n"
            "在处理打印故障之前，请检查以下情况：\n"
            "• 电源连接是否正常。\n"
            "• 接口电缆是否连接正确。\n"
            "• 驱动程序是否安装正确。",
            "在处理打印故障前应检查哪些关键项目，请列出全部内容",
            "prose_closed_intro",
            3,
        ),
    ],
)
def test_scoped_bullet_list_builds_three_explicit_anchor_shapes(
    selected_text: str,
    information_need: str,
    expected_kind: str,
    expected_count: int,
) -> None:
    artifacts = _scoped_bullet_artifacts(
        selected_text,
        information_need=information_need,
    )

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION
    assert receipt.anchor_kind == expected_kind
    assert receipt.item_count == expected_count
    assert receipt.catalog_representation_count == 2
    assert receipt.selected_authority_index == 0
    assert receipt.authority_candidate_count == 1
    assert receipt.anchor_match_key_count >= 2
    assert receipt.witness_ids == tuple(item.witness_id for item in artifacts.witnesses)
    assert parse_source_enumeration_closure_receipt(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        is receipt
    )


@pytest.mark.parametrize(
    ("selected_text", "message"),
    [
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目，包括但不限于：\n"
            "• 检查电源。\n• 检查端口。",
            "open-ended enumeration cue",
        ),
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目：\n"
            "• 检查电源。\n• 检查端口等。",
            "open-ended enumeration cue",
        ),
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目：\n"
            "• 检查电源。\n• 请参阅网络设置指南检查端口。",
            "open cross-reference",
        ),
        (
            "标题路径：维护 > 打印故障\n处理打印故障时需检查以下项目：\n• 检查电源。\n• 同上。",
            "ambiguous continuation",
        ),
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目：\n"
            "• 检查电源。\n• 检查端口。\n补充：还应检查备用线缆。",
            "unowned continuation",
        ),
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目：\n"
            "• 检查电源。\n• 检查端口。\n•",
            "empty or nested item",
        ),
        (
            "标题路径：维护 > 打印故障\n"
            "处理打印故障时需检查以下项目：\n"
            "• 检查电源。\n• 检查端口。\n  • 检查子端口。",
            "empty or nested item",
        ),
        (
            "标题路径：维护 > 打印故障\n处理打印故障时需检查以下项目：\n• 检查电源。\n• 检查电源。",
            "non-empty and distinct",
        ),
    ],
)
def test_scoped_bullet_list_rejects_open_or_ambiguous_membership(
    selected_text: str,
    message: str,
) -> None:
    artifacts = _scoped_bullet_artifacts(selected_text)

    with pytest.raises(ValueError, match=message):
        artifacts.build_receipt()


def test_scoped_bullet_list_rejects_missing_ambiguous_and_truncated_authority() -> None:
    artifacts = _scoped_bullet_artifacts()
    with pytest.raises(ValueError, match="complete raw authority"):
        artifacts.build_receipt(authorities=())

    duplicate = ScopedBulletAuthorityRepresentation(
        evidence_id=_OTHER_EVIDENCE_ID,
        representation_id=_OTHER_REPRESENTATION_ID,
        canonical_text=artifacts.authorities[0].canonical_text,
        packed_text=artifacts.authorities[0].packed_text,
    )
    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build_receipt(authorities=(artifacts.authorities[0], duplicate))

    truncated = ScopedBulletAuthorityRepresentation(
        evidence_id=_OTHER_EVIDENCE_ID,
        representation_id=_OTHER_REPRESENTATION_ID,
        canonical_text="无关附录。",
        packed_text="无关附录。\n…[证据正文因上下文预算被截断]",
    )
    with pytest.raises(ValueError, match="authority is truncated"):
        artifacts.build_receipt(authorities=(artifacts.authorities[0], truncated))


def test_scoped_bullet_list_rejects_missing_witness_and_incomplete_catalog() -> None:
    artifacts = _scoped_bullet_artifacts()
    with pytest.raises(ValueError, match="bind every source item witness once"):
        artifacts.build_receipt(witnesses=(artifacts.witnesses[0],))

    truncated_catalog = WitnessCatalogReceipt.build(
        question_contract_id=artifacts.question.contract_id,
        question_contract_hash=artifacts.question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=artifacts.witnesses,
        universe_status=WitnessUniverseStatus.TRUNCATED,
        omissions=(
            WitnessCatalogOmission(
                code=WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED,
                count=1,
            ),
        ),
        builder_version="scoped-bullet-list-test.v1",
    )
    with pytest.raises(ValueError, match="complete witness catalog"):
        artifacts.build_receipt(catalog=truncated_catalog)


def test_scoped_bullet_list_rejects_structure_drift_and_same_topic_supplement() -> None:
    canonical = (
        "标题路径：前言 > 打印故障\n处理打印故障时需检查以下项目：\n• 检查电源。\n• 检查端口。"
    )
    drifted_packed = canonical.replace("检查端口", "检查纸盒")
    artifacts = _scoped_bullet_artifacts(canonical)
    drifted_authorities = (
        ScopedBulletAuthorityRepresentation(
            evidence_id=_EVIDENCE_ID,
            representation_id=_REPRESENTATION_ID,
            canonical_text=canonical,
            packed_text=drifted_packed,
        ),
        artifacts.authorities[1],
    )
    with pytest.raises(ValueError, match="structures differ"):
        artifacts.build_receipt(authorities=drifted_authorities)

    same_topic_other = "标题路径：附录 > 打印故障\n补充：还应检查备用端口。"
    with pytest.raises(ValueError, match="same-topic supplement"):
        _scoped_bullet_artifacts(other_text=same_topic_other).build_receipt()


def test_scoped_bullet_list_replay_and_identity_reject_authority_drift() -> None:
    artifacts = _scoped_bullet_artifacts()
    receipt = artifacts.build_receipt()
    forged = receipt.to_dict()
    forged["anchor_match_key_hashes"][0] = sha256_digest("tampered")
    with pytest.raises(ValidationError, match="anchor_match_hash"):
        QueryAnchoredScopedBulletListClosureReceipt.from_dict(forged)

    drifted_authorities = (
        artifacts.authorities[0],
        ScopedBulletAuthorityRepresentation(
            evidence_id=_OTHER_EVIDENCE_ID,
            representation_id=_OTHER_REPRESENTATION_ID,
            canonical_text="标题路径：附录 > 网络术语\nSSID 含义已改变。",
            packed_text="标题路径：附录 > 网络术语\nSSID 含义已改变。",
        ),
    )
    with pytest.raises(ValueError, match="does not match supplied artifacts"):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=drifted_authorities,
        )


@dataclass(frozen=True)
class _InlineActionArtifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    authorities: tuple[ScopedListAuthorityRepresentation, ...]

    def build_receipt(
        self,
        *,
        catalog: WitnessCatalogReceipt | None = None,
        witnesses: tuple[EvidenceWitness, ...] | None = None,
        authorities: tuple[ScopedListAuthorityRepresentation, ...] | None = None,
    ) -> QueryAnchoredInlineActionChainClosureReceipt:
        return QueryAnchoredInlineActionChainClosureReceipt.build(
            question=self.question,
            catalog=catalog or self.catalog,
            obligation=self.obligation,
            witnesses=witnesses or self.witnesses,
            authority_representations=authorities or self.authorities,
            catalog_witnesses=self.witnesses,
        )


def _inline_action_artifacts(
    selected_text: str = (
        "标题路径：设置 > 打开帮助文档\n\n"
        "您可以打开“打印首选项”，点击“帮助”按钮（仅适用于 Windows 系统）。"
    ),
    *,
    selected_witness_texts: tuple[tuple[str, WitnessAtomKind], ...] | None = None,
    packed_selected_text: str | None = None,
    other_text: str = "标题路径：附录 > 网络术语\n\nSSID 是无线网络名称。",
    other_witness_texts: tuple[tuple[str, WitnessAtomKind], ...] = (),
    information_need: str = "在Windows系统中如何通过打印首选项打开帮助文档",
    source_query: str = "在Windows系统中如何通过打印首选项打开帮助文档？",
    prior_fact_need: str | None = None,
    universe_status: WitnessUniverseStatus = WitnessUniverseStatus.COMPLETE,
) -> _InlineActionArtifacts:
    procedure_facet = QueryFacet(
        facet_id=_OBLIGATION_ID,
        information_need=information_need,
        retrieval_expression=information_need,
        closure=FacetClosure.OPEN_ENUMERATION,
    )
    prior_facet = (
        QueryFacet(
            facet_id="qf_" + "9" * 32,
            information_need=prior_fact_need,
            retrieval_expression=prior_fact_need,
            closure=FacetClosure.SINGLE,
        )
        if prior_fact_need is not None
        else None
    )
    facets = (procedure_facet,) if prior_facet is None else (prior_facet, procedure_facet)
    kind_by_obligation = {_OBLIGATION_ID: ObligationKind.PROCEDURE}
    output_by_obligation = {_OBLIGATION_ID: RequiredOutputForm.ORDERED_STEPS}
    if prior_facet is not None:
        kind_by_obligation[prior_facet.facet_id] = ObligationKind.FACT
        output_by_obligation[prior_facet.facet_id] = RequiredOutputForm.FACT
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "7" * 32,
        source_query=source_query,
        facets=facets,
        builder_version="inline-action-chain-test.v1",
        kind_by_obligation=kind_by_obligation,
        output_form_by_obligation=output_by_obligation,
    )
    resolved_selected_witnesses = selected_witness_texts or (
        (
            "您可以打开“打印首选项”，点击“帮助”按钮（仅适用于 Windows 系统）。",
            WitnessAtomKind.SENTENCE,
        ),
    )
    packed_selected = packed_selected_text or selected_text
    authority_inputs = (
        (
            _EVIDENCE_ID,
            _REPRESENTATION_ID,
            selected_text,
            packed_selected,
            resolved_selected_witnesses,
        ),
        (
            _OTHER_EVIDENCE_ID,
            _OTHER_REPRESENTATION_ID,
            other_text,
            other_text,
            other_witness_texts,
        ),
    )
    witnesses: list[EvidenceWitness] = []
    authorities: list[ScopedListAuthorityRepresentation] = []
    for (
        evidence_id,
        representation_id,
        canonical_text,
        packed_text,
        witness_specs,
    ) in authority_inputs:
        authorities.append(
            ScopedListAuthorityRepresentation(
                evidence_id=evidence_id,
                representation_id=representation_id,
                canonical_text=canonical_text,
                packed_text=packed_text,
            )
        )
        canonical_cursor = 0
        packed_cursor = 0
        for witness_text, atom_kind in witness_specs:
            canonical_start = canonical_text.index(witness_text, canonical_cursor)
            packed_start = packed_text.index(witness_text, packed_cursor)
            canonical_cursor = canonical_start + len(witness_text)
            packed_cursor = packed_start + len(witness_text)
            span_ref = EvidenceSpanRef.from_texts(
                evidence_id=evidence_id,
                representation_id=representation_id,
                canonical_text=canonical_text,
                packed_text=packed_text,
                canonical_start=canonical_start,
                canonical_end=canonical_start + len(witness_text),
                packed_start=packed_start,
                packed_end=packed_start + len(witness_text),
                equivalence=SpanEquivalence.EXACT_UTF8,
            )
            witnesses.append(
                EvidenceWitness.build(
                    context_pack_hash=_CONTEXT_PACK_HASH,
                    authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                    span_ref=span_ref,
                    atom_kind=atom_kind,
                    ordinal=len(witnesses),
                )
            )
    omissions = (
        ()
        if universe_status == WitnessUniverseStatus.COMPLETE
        else (
            WitnessCatalogOmission(
                code=WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED,
                count=1,
            ),
        )
    )
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=question.contract_id,
        question_contract_hash=question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=tuple(witnesses),
        universe_status=universe_status,
        omissions=omissions,
        builder_version="inline-action-chain-test.v1",
    )
    return _InlineActionArtifacts(
        question=question,
        obligation=next(
            item for item in question.obligations if item.obligation_id == _OBLIGATION_ID
        ),
        catalog=catalog,
        witnesses=tuple(witnesses),
        authorities=tuple(authorities),
    )


def test_inline_action_chain_builds_and_replays_one_sentence_witness() -> None:
    artifacts = _inline_action_artifacts()

    receipt = artifacts.build_receipt()

    assert receipt.schema_version == QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION
    assert receipt.witness_ids == (artifacts.witnesses[0].witness_id,)
    assert receipt.witness_atom_kind == "sentence"
    assert receipt.action_count == 2
    assert receipt.action_ordinals == (1, 2)
    authority = artifacts.authorities[receipt.selected_authority_index]
    actions = tuple(
        authority.canonical_text[start:end]
        for start, end in zip(
            receipt.canonical_action_starts,
            receipt.canonical_action_ends,
            strict=True,
        )
    )
    assert actions == (
        "您可以打开“打印首选项”",
        "点击“帮助”按钮（仅适用于 Windows 系统）",
    )
    assert receipt.canonical_terminal_result_hash is None
    assert parse_source_enumeration_closure_receipt(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
            catalog_witnesses=artifacts.witnesses,
        )
        is receipt
    )


def test_inline_action_chain_uses_only_the_current_obligation_temporal_trigger() -> None:
    text = (
        "标题路径：前言 > 中部卡纸\n\n"
        "2. 打开前盖，取出激光碳粉盒。\n\n"
        "4. 取出卡纸后，重新装入激光碳粉盒，合上前盖，打印机将自动恢复打印。"
    )
    artifacts = _inline_action_artifacts(
        text,
        selected_witness_texts=(
            ("2. 打开前盖，取出激光碳粉盒。", WitnessAtomKind.LIST_ITEM),
            (
                "4. 取出卡纸后，重新装入激光碳粉盒，合上前盖，打印机将自动恢复打印。",
                WitnessAtomKind.LIST_ITEM,
            ),
        ),
        information_need="取出卡纸后又需完成哪些操作才能恢复打印",
        source_query=("取出激光碳粉盒后如何避免曝光？取出卡纸后又需完成哪些操作才能恢复打印？"),
        prior_fact_need="取出激光碳粉盒后如何避免感光鼓曝光",
    )

    receipt = artifacts.build_receipt()

    assert receipt.witness_ids == (artifacts.witnesses[1].witness_id,)
    assert receipt.witness_atom_kind == "list_item"
    assert receipt.action_count == 2
    assert receipt.local_trigger_hash == sha256_digest("取出卡纸后")
    assert receipt.canonical_terminal_result_hash is not None


def test_inline_action_chain_owns_temporal_prefix_without_counting_the_result() -> None:
    text = (
        "标题路径：前言 > 双面打印单元卡纸\n\n"
        "4. 取出卡纸后，装回双面打印单元，检查产品其他部位，"
        "确保无卡纸后，合上后盖，打印机将自动恢复打印。"
    )
    witness_text = text.split("\n\n", maxsplit=1)[1]
    artifacts = _inline_action_artifacts(
        text,
        selected_witness_texts=((witness_text, WitnessAtomKind.LIST_ITEM),),
        information_need="取出卡纸后还需进行哪些检查和操作",
        source_query="如果纸张未出来怎么办？取出卡纸后还需进行哪些检查和操作？",
        prior_fact_need="如果纸张未随双面打印单元一起出来应如何操作",
    )

    receipt = artifacts.build_receipt()
    authority = artifacts.authorities[0]
    actions = tuple(
        authority.canonical_text[start:end]
        for start, end in zip(
            receipt.canonical_action_starts,
            receipt.canonical_action_ends,
            strict=True,
        )
    )

    assert receipt.action_count == 3
    assert actions == (
        "取出卡纸后，装回双面打印单元",
        "检查产品其他部位",
        "确保无卡纸后，合上后盖",
    )
    assert "自动恢复打印" not in "\n".join(actions)


@pytest.mark.parametrize(
    "witness_text",
    [
        "1. 打开前盖，参阅第 3 章完成后续操作。",
        "1. 如果指示灯闪烁，打开前盖，取出组件。",
        "1. 按照上述步骤打开前盖，取出组件。",
        "1. 打开前盖，取出组件等。",
        "1. 打开前盖，组件将自动取出。",
    ],
)
def test_inline_action_chain_rejects_open_or_dependent_action_sources(
    witness_text: str,
) -> None:
    text = f"标题路径：维护 > 组件处理操作\n\n{witness_text}"
    artifacts = _inline_action_artifacts(
        text,
        selected_witness_texts=((witness_text, WitnessAtomKind.LIST_ITEM),),
        information_need="组件处理操作需要完成哪些动作",
        source_query="组件处理操作需要完成哪些动作？",
    )

    with pytest.raises(ValueError, match="no eligible candidate"):
        artifacts.build_receipt()


def test_inline_action_chain_rejects_tied_witnesses_and_incomplete_catalog() -> None:
    witness_text = "取出卡纸后，重新装入组件，合上前盖。"
    selected = f"标题路径：维护 > 卡纸恢复操作\n\n{witness_text}"
    other = f"标题路径：维护 > 卡纸恢复操作\n\n{witness_text}"
    tied = _inline_action_artifacts(
        selected,
        selected_witness_texts=((witness_text, WitnessAtomKind.SENTENCE),),
        other_text=other,
        other_witness_texts=((witness_text, WitnessAtomKind.SENTENCE),),
        information_need="取出卡纸后要完成哪些卡纸恢复操作",
        source_query="取出卡纸后要完成哪些卡纸恢复操作？",
        prior_fact_need="卡纸发生在哪里",
    )
    with pytest.raises(ValueError, match="tied top candidate"):
        tied.build_receipt()

    incomplete = _inline_action_artifacts(
        universe_status=WitnessUniverseStatus.TRUNCATED,
    )
    with pytest.raises(ValueError, match="complete witness catalog"):
        incomplete.build_receipt()


def test_inline_action_chain_rejects_title_drift_and_receipt_tampering() -> None:
    canonical = "标题路径：设置 > 打开帮助文档\n\n您可以打开“打印首选项”，点击“帮助”按钮。"
    packed = canonical.replace("打开帮助文档", "启动帮助文档")
    drifted = _inline_action_artifacts(
        canonical,
        selected_witness_texts=(
            ("您可以打开“打印首选项”，点击“帮助”按钮。", WitnessAtomKind.SENTENCE),
        ),
        packed_selected_text=packed,
    )
    with pytest.raises(ValueError, match="no eligible candidate"):
        drifted.build_receipt()

    artifacts = _inline_action_artifacts()
    receipt = artifacts.build_receipt()
    forged = receipt.to_dict()
    forged["canonical_action_hashes"][0] = sha256_digest("tampered")
    with pytest.raises(ValidationError, match="structure hash|receipt_hash"):
        QueryAnchoredInlineActionChainClosureReceipt.from_dict(forged)
