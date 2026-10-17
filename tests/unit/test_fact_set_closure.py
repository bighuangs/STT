from __future__ import annotations

from dataclasses import dataclass, replace
import re

import pytest
from pydantic import ValidationError

from app.domain.answering._base import sha256_digest
from app.domain.answering.fact_set_closure import (
    LIST_ITEM_FACT_SEGMENTATION_METHOD,
    QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION,
    SENTENCE_FACT_SEGMENTATION_METHOD,
    FactSetAuthorityRepresentation,
    FactSetMode,
    FactSetRelation,
    QueryAnchoredScopedFactSetClosureReceipt,
    build_query_anchored_scoped_fact_set_closure_receipt,
)
from app.domain.answering.enumeration_closure import (
    parse_source_enumeration_closure_receipt,
    source_closure_witness_ids,
)
from app.domain.answering.question import QuestionContract, QuestionObligation
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


_OBLIGATION_ID = "qf_" + "a" * 32
_CONTEXT_PACK_HASH = sha256_digest("fact-set-test-context")
_AUTHORIZED_SCOPE_HASH = "scope_" + "d" * 64
_EVIDENCE_DIGITS = "bcdef0123456789a"
_REPRESENTATION_DIGITS = "123456789abcdef0"


@dataclass(frozen=True)
class _Artifacts:
    question: QuestionContract
    obligation: QuestionObligation
    catalog: WitnessCatalogReceipt
    witnesses: tuple[EvidenceWitness, ...]
    authorities: tuple[FactSetAuthorityRepresentation, ...]

    def build(self) -> QueryAnchoredScopedFactSetClosureReceipt:
        return build_query_anchored_scoped_fact_set_closure_receipt(
            question=self.question,
            catalog=self.catalog,
            obligation=self.obligation,
            witnesses=self.witnesses,
            authority_representations=self.authorities,
        )


def _artifacts(
    *,
    source_query: str,
    information_need: str | None = None,
    authority_texts: tuple[str, ...],
    packed_texts: tuple[str, ...] | None = None,
    witness_specs: tuple[tuple[int, str, WitnessAtomKind], ...],
    universe_status: WitnessUniverseStatus = WitnessUniverseStatus.COMPLETE,
) -> _Artifacts:
    resolved_need = information_need or source_query
    question = QuestionContract.from_facets(
        query_plan_id="qpl_" + "1" * 32,
        source_query=source_query,
        facets=(
            QueryFacet(
                facet_id=_OBLIGATION_ID,
                information_need=resolved_need,
                retrieval_expression=resolved_need,
                closure=FacetClosure.OPEN_ENUMERATION,
            ),
        ),
        builder_version="fact-set-closure-test.v1",
    )
    resolved_packed_texts = packed_texts or authority_texts
    if len(resolved_packed_texts) != len(authority_texts):
        raise AssertionError("canonical and packed authority counts must match")
    authorities = tuple(
        FactSetAuthorityRepresentation(
            evidence_id="ev_" + _EVIDENCE_DIGITS[index] * 40,
            representation_id="rep_" + _REPRESENTATION_DIGITS[index] * 40,
            canonical_text=text,
            packed_text=resolved_packed_texts[index],
        )
        for index, text in enumerate(authority_texts)
    )
    witnesses: list[EvidenceWitness] = []
    seen: dict[tuple[int, str], int] = {}
    for ordinal, (authority_index, member_text, atom_kind) in enumerate(witness_specs):
        authority = authorities[authority_index]
        occurrence = seen.get((authority_index, member_text), 0)
        starts = [
            match.start() for match in re.finditer(re.escape(member_text), authority.canonical_text)
        ]
        packed_starts = [
            match.start() for match in re.finditer(re.escape(member_text), authority.packed_text)
        ]
        if occurrence >= len(starts) or occurrence >= len(packed_starts):
            raise AssertionError(f"fixture member not found: {member_text}")
        start = starts[occurrence]
        packed_start = packed_starts[occurrence]
        seen[(authority_index, member_text)] = occurrence + 1
        span_ref = EvidenceSpanRef.from_texts(
            evidence_id=authority.evidence_id,
            representation_id=authority.representation_id,
            canonical_text=authority.canonical_text,
            packed_text=authority.packed_text,
            canonical_start=start,
            canonical_end=start + len(member_text),
            packed_start=packed_start,
            packed_end=packed_start + len(member_text),
            equivalence=SpanEquivalence.EXACT_UTF8,
        )
        witnesses.append(
            EvidenceWitness.build(
                context_pack_hash=_CONTEXT_PACK_HASH,
                authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
                span_ref=span_ref,
                atom_kind=atom_kind,
                ordinal=ordinal,
            )
        )
    omissions = (
        ()
        if universe_status == WitnessUniverseStatus.COMPLETE
        else (
            WitnessCatalogOmission(
                code=WitnessOmissionCode.CONTEXT_BUDGET,
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
        builder_version="fact-set-closure-test.v1",
    )
    return _Artifacts(
        question=question,
        obligation=question.obligations[0],
        catalog=catalog,
        witnesses=tuple(witnesses),
        authorities=authorities,
    )


@pytest.mark.parametrize(
    ("query", "sentence", "relation"),
    (
        (
            "远航设备有限公司注册的商标包括哪些？",
            "Orion、Orion Cloud 和 Orion Link 是远航设备有限公司注册的商标。",
            FactSetRelation.OWNERSHIP_REGISTRATION,
        ),
        (
            "多功能设备包含哪些模块？",
            "多功能设备包含扫描器、双面器和无线模块。",
            FactSetRelation.INCLUSION,
        ),
    ),
)
def test_builds_closed_inline_coordination_receipt(
    query: str,
    sentence: str,
    relation: FactSetRelation,
) -> None:
    artifacts = _artifacts(
        source_query=query,
        authority_texts=(f"标题路径：产品手册 > 法律说明\n\n{sentence}",),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )

    receipt = artifacts.build()

    assert receipt.schema_version == QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION
    assert receipt.mode == FactSetMode.CLOSED_INLINE_COORDINATION
    assert receipt.relation == relation
    assert receipt.member_count == 1
    assert receipt.semantic_count == 3
    assert receipt.member_semantic_counts == (3,)
    assert receipt.segmentation_methods == (SENTENCE_FACT_SEGMENTATION_METHOD,)
    assert QueryAnchoredScopedFactSetClosureReceipt.from_dict(receipt.to_dict()) == receipt
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_builds_eu_directive_coordination_from_ec_titled_authority() -> None:
    compliance = (
        "本产品符合欧共体理事会 2004/108/EC 和 2006/95/EC 指令的成员国近似和"
        "协调法规中涉及电磁兼容性和电气设备安全性的保护要求。"
    )
    rohs = "本产品完全符合 ROHS 指令 2009/95/EC 及重订指令 2011/65/EU 的管理要求。"
    artifacts = _artifacts(
        source_query="该打印机符合哪些欧盟指令的合规要求？",
        authority_texts=(
            f"标题路径：前言 > CE\n\n欧共体 (EC) 指令合规性\n\n{compliance}\n\n{rohs}",
        ),
        witness_specs=(
            (0, compliance, WitnessAtomKind.SENTENCE),
            (0, rohs, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.CLOSED_INLINE_COORDINATION
    assert receipt.relation == FactSetRelation.DIRECTIVE_COMPLIANCE
    assert receipt.semantic_count == 2
    assert receipt.witness_ids == (artifacts.witnesses[0].witness_id,)
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_builds_title_bounded_geographic_applicability_dimensions() -> None:
    compliance = (
        "本产品符合欧共体理事会 2004/108/EC 和 2006/95/EC 指令的成员国近似和"
        "协调法规中涉及电磁兼容性和电气设备安全性的保护要求。"
    )
    climate = "本产品仅适用于非热带地区安全使用。"
    altitude = "本产品仅适用于海拔 2000 米及以下地区安全使用。"
    artifacts = _artifacts(
        source_query=("该打印机符合哪些欧盟指令的合规要求？其适用的地理环境限制是什么？"),
        information_need="其适用的地理环境限制是什么",
        authority_texts=(
            "标题路径：前言 > CE\n\n欧共体 (EC) 指令合规性\n\n"
            f"{compliance}\n\n{climate}\n\n{altitude}",
        ),
        witness_specs=(
            (0, compliance, WitnessAtomKind.SENTENCE),
            (0, climate, WitnessAtomKind.SENTENCE),
            (0, altitude, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET
    assert receipt.relation == FactSetRelation.APPLICABILITY_RESTRICTION
    assert receipt.topic_group_member_counts == (1, 1)
    assert receipt.witness_ids == tuple(item.witness_id for item in artifacts.witnesses[1:])
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_geographic_applicability_rejects_a_missing_dimension() -> None:
    climate = "本产品仅适用于非热带地区安全使用。"
    artifacts = _artifacts(
        source_query=("该打印机符合哪些欧盟指令的合规要求？其适用的地理环境限制是什么？"),
        information_need="其适用的地理环境限制是什么",
        authority_texts=(f"标题路径：前言 > CE\n\n欧共体 (EC) 指令合规性\n\n{climate}",),
        witness_specs=((0, climate, WitnessAtomKind.SENTENCE),),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_fact_set_parser_round_trips_as_source_closure_receipt() -> None:
    sentence = "多功能设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="多功能设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )

    receipt = artifacts.build()
    parsed = parse_source_enumeration_closure_receipt(receipt.to_dict())

    assert parsed == receipt
    assert source_closure_witness_ids(parsed) == receipt.witness_ids


def test_fact_set_accepts_complete_catalog_with_non_fact_atom() -> None:
    heading_text = "标题路径：产品手册 > 法律说明"
    sentence = "Orion、Orion Cloud 和 Orion Link 是远航设备有限公司注册的商标。"
    artifacts = _artifacts(
        source_query="远航设备有限公司注册的商标包括哪些？",
        authority_texts=(f"{heading_text}\n\n{sentence}",),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )
    authority = artifacts.authorities[0]
    heading_span = EvidenceSpanRef.from_texts(
        evidence_id=authority.evidence_id,
        representation_id=authority.representation_id,
        canonical_text=authority.canonical_text,
        packed_text=authority.packed_text,
        canonical_start=0,
        canonical_end=len(heading_text),
        packed_start=0,
        packed_end=len(heading_text),
        equivalence=SpanEquivalence.EXACT_UTF8,
    )
    heading = EvidenceWitness.build(
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        span_ref=heading_span,
        atom_kind=WitnessAtomKind.MEDIA_DERIVATION,
        ordinal=0,
    )
    sentence_witness = artifacts.witnesses[0].model_copy(update={"ordinal": 1})
    catalog_witnesses = (heading, sentence_witness)
    catalog = WitnessCatalogReceipt.build(
        question_contract_id=artifacts.question.contract_id,
        question_contract_hash=artifacts.question.contract_hash,
        context_pack_hash=_CONTEXT_PACK_HASH,
        authorized_scope_hash=_AUTHORIZED_SCOPE_HASH,
        witnesses=catalog_witnesses,
        universe_status=WitnessUniverseStatus.COMPLETE,
        builder_version="fact-set-closure-test.v1",
    )

    receipt = build_query_anchored_scoped_fact_set_closure_receipt(
        question=artifacts.question,
        catalog=catalog,
        obligation=artifacts.obligation,
        witnesses=(sentence_witness,),
        authority_representations=artifacts.authorities,
        catalog_witnesses=catalog_witnesses,
    )

    assert receipt.witness_ids == (sentence_witness.witness_id,)


def test_fact_set_skips_sentence_atoms_that_partition_one_title_path_line() -> None:
    heading_prefix = "标题路径：法律说明 > 6.2.1.1."
    heading_tail = "商标"
    sentence = "Mopria、Mopria 标识和 Mopria Scan 是 Mopria Alliance, Inc. 注册的商标。"
    artifacts = _artifacts(
        source_query="Mopria Alliance, Inc. 注册的商标包括哪些？",
        authority_texts=(f"{heading_prefix} {heading_tail}\n\n{sentence}",),
        witness_specs=(
            (0, heading_prefix, WitnessAtomKind.SENTENCE),
            (0, heading_tail, WitnessAtomKind.SENTENCE),
            (0, sentence, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.witness_ids == (artifacts.witnesses[-1].witness_id,)
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_fact_set_skips_an_ocr_description_heading_without_skipping_body_fact() -> None:
    sentence = "Mopria、Mopria 标识和 Mopria Scan 是远航设备有限公司注册的商标。"
    ocr_heading = "一般图形内容"
    ocr_description = "图像中从左至右展示三个圆形品牌图标。"
    artifacts = _artifacts(
        source_query="远航设备有限公司注册的商标包括哪些？",
        authority_texts=(
            f"标题路径：法律说明 > 商标\n\n{sentence}\n\n{ocr_heading}\n\n{ocr_description}",
        ),
        witness_specs=(
            (0, sentence, WitnessAtomKind.SENTENCE),
            (0, ocr_heading, WitnessAtomKind.SENTENCE),
            (0, ocr_description, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.witness_ids == (artifacts.witnesses[0].witness_id,)


def test_fact_set_does_not_skip_a_real_fact_before_a_title_line() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(f"{sentence}\n\n标题路径：产品手册 > 功能\n\n{sentence}",),
        witness_specs=(
            (0, sentence, WitnessAtomKind.SENTENCE),
            (0, sentence, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_fact_set_ignores_packed_image_placeholder_as_a_section_heading() -> None:
    power = "电源插拔时请保持双手干燥，以免触电。"
    heat = "高温定影单元运行后请勿触摸，以免烫伤。"
    canonical = f"标题路径：前言 > 警告\n\n![警告图](images/warning.jpg)\n\n{power}\n\n{heat}"
    packed = f"标题路径：前言 > 警告\n\n[image omitted]\n\n{power}\n\n{heat}"
    artifacts = _artifacts(
        source_query="在打印机警告中，关于电源插拔和高温定影单元分别提出了哪些安全提示？",
        authority_texts=(canonical,),
        packed_texts=(packed,),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET
    assert receipt.member_count == 2
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_fact_set_rejects_omitted_candidate_witness_from_complete_catalog() -> None:
    power = "电源插拔时请保持双手干燥，以免触电。"
    heat = "高温定影组件运行后仍会发热，请勿触摸，以免烫伤。"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源插拔和高温定影组件分别有哪些安全提示？",
        authority_texts=(f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}",),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError, match="exactly match"):
        build_query_anchored_scoped_fact_set_closure_receipt(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses[:1],
            authority_representations=artifacts.authorities,
            catalog_witnesses=artifacts.witnesses,
        )


@pytest.mark.parametrize(
    "forged_update",
    (
        {"ordinal": 7},
        {"witness_hash": sha256_digest("forged-fact-witness")},
    ),
)
def test_fact_set_rejects_forged_candidate_witness(
    forged_update: dict[str, object],
) -> None:
    sentence = "多功能设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="多功能设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )
    forged = artifacts.witnesses[0].model_copy(update=forged_update)

    with pytest.raises(ValueError):
        build_query_anchored_scoped_fact_set_closure_receipt(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=(forged,),
            authority_representations=artifacts.authorities,
            catalog_witnesses=artifacts.witnesses,
        )


@pytest.mark.parametrize("atom_kind", (WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM))
def test_builds_title_bounded_topic_predicate_set(atom_kind: WitnessAtomKind) -> None:
    prefix = "- " if atom_kind == WitnessAtomKind.LIST_ITEM else ""
    power = f"{prefix}电源插拔时请保持双手干燥，以免触电。"
    heat = f"{prefix}高温定影组件运行后仍会发热，请勿触摸，以免烫伤。"
    text = f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源插拔和高温定影组件分别有哪些安全提示？",
        authority_texts=(text,),
        witness_specs=((0, power, atom_kind), (0, heat, atom_kind)),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET
    assert receipt.relation == FactSetRelation.SAFETY_INSTRUCTION
    assert receipt.topic_group_count == 2
    assert receipt.topic_group_member_counts == (1, 1)
    assert receipt.member_count == 2
    assert receipt.member_semantic_counts == (1, 1)
    expected_segmentation = (
        SENTENCE_FACT_SEGMENTATION_METHOD
        if atom_kind == WitnessAtomKind.SENTENCE
        else LIST_ITEM_FACT_SEGMENTATION_METHOD
    )
    assert receipt.segmentation_methods == (expected_segmentation, expected_segmentation)


def test_stateful_topic_does_not_absorb_a_warning_label_fact() -> None:
    power = "电源插拔时请保持双手干燥，以免触电。"
    heat = "加热组件运行后处于高温状态，请勿触摸加热单元，以免烫伤。"
    label = "加热单元有高温警示标签，请勿移动或损坏该标签。"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源插拔和高温加热单元分别有哪些安全提示？",
        authority_texts=(f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}\n\n{label}",),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
            (0, label, WitnessAtomKind.SENTENCE),
        ),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET
    assert receipt.member_count == 2
    assert receipt.witness_ids == tuple(item.witness_id for item in artifacts.witnesses[:2])


@pytest.mark.parametrize(
    "sentence",
    (
        "多功能设备包含扫描器、双面器等。",
        "多功能设备包含扫描器、双面器，包括但不限于无线模块。",
        "多功能设备包含扫描器、双面器和。",
        "多功能设备包含扫描器、双面器和无线模块，参阅第 3 章。",
        "多功能设备包含扫描器、双面器和无线模块<em>等</em>。",
    ),
)
def test_inline_coordination_rejects_open_reference_or_dangling_sources(
    sentence: str,
) -> None:
    artifacts = _artifacts(
        source_query="多功能设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )

    with pytest.raises(ValueError):
        artifacts.build()


def test_scoped_set_rejects_external_reference() -> None:
    power = "电源插拔时请保持双手干燥，其他要求参阅第 4 章。"
    heat = "高温定影组件运行后请勿触摸，以免烫伤。"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源插拔和高温定影组件分别有哪些安全提示？",
        authority_texts=(f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}",),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError):
        artifacts.build()


def test_scoped_set_rejects_same_topic_supplement() -> None:
    power = "电源插拔时请保持双手干燥，以免触电。"
    heat = "高温定影组件运行后请勿触摸，以免烫伤。"
    supplement = "此外，电源插拔前还必须检查插头是否破损。"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源插拔和高温定影组件分别有哪些安全提示？",
        authority_texts=(f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}\n\n{supplement}",),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
            (0, supplement, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError):
        artifacts.build()


def test_scoped_set_rejects_anaphor_without_local_antecedent() -> None:
    power = "该电源部件必须保持干燥，以免触电。"
    heat = "高温部件运行后请勿触摸，以免烫伤。"
    artifacts = _artifacts(
        source_query="在设备安全提示中，关于电源部件和高温部件分别有哪些安全提示？",
        authority_texts=(f"标题路径：设备手册 > 安全提示\n\n{power}\n\n{heat}",),
        witness_specs=(
            (0, power, WitnessAtomKind.SENTENCE),
            (0, heat, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError):
        artifacts.build()


def test_catalog_wide_duplicate_candidate_is_rejected() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(sentence, sentence),
        witness_specs=(
            (0, sentence, WitnessAtomKind.SENTENCE),
            (1, sentence, WitnessAtomKind.SENTENCE),
        ),
    )

    with pytest.raises(ValueError):
        artifacts.build()


def test_incomplete_catalog_is_rejected() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
        universe_status=WitnessUniverseStatus.TRUNCATED,
    )

    with pytest.raises(ValueError, match="complete witness catalog"):
        artifacts.build()


def test_validate_artifacts_rejects_raw_authority_drift() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )
    receipt = artifacts.build()
    drifted_authority = replace(
        artifacts.authorities[0],
        canonical_text=artifacts.authorities[0].canonical_text + "\n附录说明。",
        packed_text=artifacts.authorities[0].packed_text + "\n附录说明。",
    )

    with pytest.raises(ValueError):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=(drifted_authority,),
        )


def test_validate_artifacts_rejects_authority_reorder() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(sentence, "标题路径：附录 > 术语\n\n这里没有候选事实。"),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )
    receipt = artifacts.build()

    with pytest.raises(ValueError, match="does not match supplied artifacts"):
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=tuple(reversed(artifacts.authorities)),
        )


def test_from_dict_rejects_forged_fail_closed_counter() -> None:
    sentence = "设备包含扫描器、双面器和无线模块。"
    artifacts = _artifacts(
        source_query="设备包含哪些模块？",
        authority_texts=(sentence,),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )
    forged = artifacts.build().to_dict()
    forged["supplement_candidate_count"] = 1

    with pytest.raises(ValidationError):
        QueryAnchoredScopedFactSetClosureReceipt.from_dict(forged)


_NETWORK_CHECK_QUERY = "当打印机在网络中无法被搜索到时，应检查哪几项内容？请列出所有相关检查点。"
_NETWORK_CHECK_TITLE = "4.如果在网络中使用打印机时出现问题，请检查以下几方面："
_NETWORK_CHECK_ITEMS = (
    "• 关闭计算机、接入点（无线路由器）和打印机的电源，然后再重启。",
    "• 检查防火墙软件是否封锁通信。如果计算机和打印机连接在同一网络中却搜索不到，"
    "可能是因为防火墙软件封锁了通信。请参阅防火墙软件的用户指南，将防火墙关闭，"
    "然后重新尝试搜索打印机。",
    "• 检查打印机 IP 地址的分配是否正确。您可以打印打印机的网络配置信息页来检查 IP 地址。",
)


def _network_checklist_text(
    items: tuple[str, ...] = _NETWORK_CHECK_ITEMS,
    *,
    title: str = _NETWORK_CHECK_TITLE,
) -> str:
    return f"标题路径：前言 > {title}\n\n" + "\n\n".join(items)


def test_builds_scoped_bullet_lead_clause_set_with_one_narrow_tail_xref() -> None:
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(_network_checklist_text(),),
        witness_specs=tuple((0, item, WitnessAtomKind.LIST_ITEM) for item in _NETWORK_CHECK_ITEMS),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET
    assert receipt.relation == FactSetRelation.DIRECTIVE_COMPLIANCE
    assert receipt.member_count == 3
    assert receipt.topic_group_member_counts == (3,)
    assert receipt.member_semantic_counts == (1, 1, 1)
    assert receipt.canonical_member_hashes == tuple(
        sha256_digest(item) for item in _NETWORK_CHECK_ITEMS
    )
    assert (
        receipt.validate_artifacts(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses,
            authority_representations=artifacts.authorities,
        )
        == receipt
    )


def test_scoped_bullet_lead_rejects_a_tail_sibling_check() -> None:
    changed = (
        _NETWORK_CHECK_ITEMS[0],
        "• 检查防火墙软件是否封锁通信。另外，还应检查代理服务器设置。",
        _NETWORK_CHECK_ITEMS[2],
    )
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(_network_checklist_text(changed),),
        witness_specs=tuple((0, item, WitnessAtomKind.LIST_ITEM) for item in changed),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_scoped_bullet_lead_rejects_a_pure_xref_item() -> None:
    changed = (
        _NETWORK_CHECK_ITEMS[0],
        "• 请参阅防火墙软件的用户指南。",
        _NETWORK_CHECK_ITEMS[2],
    )
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(_network_checklist_text(changed),),
        witness_specs=tuple((0, item, WitnessAtomKind.LIST_ITEM) for item in changed),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_scoped_bullet_lead_rejects_two_title_matched_candidates() -> None:
    text = _network_checklist_text()
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(text, text),
        witness_specs=tuple(
            (authority_index, item, WitnessAtomKind.LIST_ITEM)
            for authority_index in range(2)
            for item in _NETWORK_CHECK_ITEMS
        ),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_scoped_bullet_lead_rejects_a_weakly_matched_title() -> None:
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(_network_checklist_text(title="网络故障排查"),),
        witness_specs=tuple((0, item, WitnessAtomKind.LIST_ITEM) for item in _NETWORK_CHECK_ITEMS),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()


def test_scoped_bullet_lead_rejects_a_missing_catalog_witness() -> None:
    artifacts = _artifacts(
        source_query=_NETWORK_CHECK_QUERY,
        authority_texts=(_network_checklist_text(),),
        witness_specs=tuple((0, item, WitnessAtomKind.LIST_ITEM) for item in _NETWORK_CHECK_ITEMS),
    )

    with pytest.raises(ValueError, match="exactly match"):
        build_query_anchored_scoped_fact_set_closure_receipt(
            question=artifacts.question,
            catalog=artifacts.catalog,
            obligation=artifacts.obligation,
            witnesses=artifacts.witnesses[:-1],
            authority_representations=artifacts.authorities,
            catalog_witnesses=artifacts.witnesses,
        )


def test_builds_network_installation_prerequisite_information_set() -> None:
    sentence = "驱动安装前，必须了解您的接入点（无线路由器）的“网络名（SSID）”和“密码”。"
    artifacts = _artifacts(
        source_query=(
            "在配置Nova X1打印机连接新无线网络时，必须满足什么前提条件，并在安装向导中如何操作？"
        ),
        information_need="在配置Nova X1打印机连接新无线网络时，必须满足什么前提条件",
        authority_texts=(
            f"标题路径：用户手册 > 基础结构连接安装\n\n{sentence}\n\n"
            "如果无法确定，请咨询网络管理员。",
        ),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )

    receipt = artifacts.build()

    assert receipt.mode == FactSetMode.CLOSED_INLINE_COORDINATION
    assert receipt.relation == FactSetRelation.REQUIRED_INFORMATION
    assert receipt.member_count == 1
    assert receipt.semantic_count == 2
    assert receipt.canonical_member_hashes == (sha256_digest(sentence),)


@pytest.mark.parametrize(
    ("title", "sentence"),
    (
        (
            "基础结构连接安装",
            "驱动安装前，必须了解管理员用户名和密码。",
        ),
        (
            "通用帮助",
            "驱动安装前，必须了解接入点的网络名（SSID）和密码。",
        ),
        (
            "基础结构连接安装",
            "驱动安装后，必须了解接入点的网络名（SSID）和密码。",
        ),
    ),
)
def test_network_installation_prerequisite_rejects_weak_semantic_bridges(
    title: str,
    sentence: str,
) -> None:
    artifacts = _artifacts(
        source_query="在配置Nova X1连接新无线网络时，必须满足什么前提条件？",
        authority_texts=(f"标题路径：用户手册 > {title}\n\n{sentence}",),
        witness_specs=((0, sentence, WitnessAtomKind.SENTENCE),),
    )

    with pytest.raises(ValueError, match="exactly one catalog-wide candidate"):
        artifacts.build()
