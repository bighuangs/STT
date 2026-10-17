"""QuestionContract 与 WitnessCatalog 的确定性构建和重解析。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.application.query_understanding_service import is_multi_step_procedure_request
from app.application.query_answering.atomizer import atomize_evidence_text
from app.application.query_answering.proposals import MAX_WITNESS_CATALOG_ITEMS
from app.domain.answering.enumeration_closure import (
    QueryAnchoredInlineActionChainClosureReceipt,
    QueryAnchoredProcedureListClosureReceipt,
    QueryAnchoredScopedBulletListClosureReceipt,
    QueryAnchoredScopedProcedureListClosureReceipt,
    QueryAnchoredScopedTableSelectionClosureReceipt,
    QueryAnchoredTableRowClosureReceipt,
    ScopedListAuthorityRepresentation,
    SourceEnumerationClosureReceipt,
    parse_source_enumeration_closure_receipt,
    source_closure_witness_ids,
)
from app.domain.answering.fact_set_closure import (
    FactSetAuthorityRepresentation,
    QueryAnchoredScopedFactSetClosureReceipt,
    build_query_anchored_scoped_fact_set_closure_receipt,
)
from app.domain.answering.question import (
    ObligationKind,
    QuestionContract,
    RequiredOutputForm,
)
from app.domain.answering.witness import (
    PACKED_TRUNCATION_MARKER,
    EvidenceSpanRef,
    EvidenceWitness,
    SpanEquivalence,
    WitnessAtomKind,
    WitnessCatalogOmission,
    WitnessCatalogReceipt,
    WitnessOmissionCode,
    WitnessUniverseStatus,
)
from app.domain.evidence import Evidence, sha256_digest
from app.domain.query_planning import FacetClosure, QueryPlan
from app.domain.safe_representation import (
    SAFE_REPRESENTATION_TRANSFORM_VERSION,
    SafeRepresentationDerivationReceipt,
    SafeRepresentationSpan,
    SafeSpanOffsetMapping,
    SafeSpanRelation,
)
from app.rag.query.context_pack_service import (
    ContextPack,
    replay_context_pack_content_hash,
)


QUESTION_CONTRACT_BUILDER_VERSION = "question-contract-builder.v6"
WITNESS_CATALOG_BUILDER_VERSION = "witness-catalog-builder.v13"
_EMPTY_SCOPE_HASH = "scope_" + "0" * 64


class _PackedBlockView(BaseModel):
    """Application 只读取消费见证构建所需的 ContextPack v2 字段。"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    representation_id: str = Field(pattern=r"^rep_[0-9a-f]{40}$")
    original_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=4_100_000)
    sub_query_ids: tuple[str, ...] = ()
    truncation: Mapping[str, Any] | None = None

    @field_validator("sub_query_ids")
    @classmethod
    def validate_sub_query_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("ContextPack sub_query_ids cannot contain blank identifiers")
        if len(normalized) != len(set(normalized)):
            raise ValueError("ContextPack sub_query_ids cannot contain duplicates")
        return normalized


class _OmittedEvidenceView(BaseModel):
    """只保留证明目录完整性判定需要的省略事实。"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    reason: str = Field(min_length=1, max_length=128)


class _CitationLabelView(BaseModel):
    """验证标签到 Evidence/representation 的关键身份映射。"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{40}$")
    representation_id: str = Field(pattern=r"^rep_[0-9a-f]{40}$")


class _ContextPackV2View(BaseModel):
    """Application 内部最小 ContextPack 契约，拒绝旧版本和身份漂移。"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: Literal["context-pack.v2"]
    packed_blocks: tuple[_PackedBlockView, ...]
    omitted: tuple[_OmittedEvidenceView, ...]
    label_mapping: tuple[_CitationLabelView, ...]
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_mapping(self) -> Self:
        expected_labels = tuple(f"E{index}" for index in range(1, len(self.packed_blocks) + 1))
        labels = tuple(item.label for item in self.packed_blocks)
        if labels != expected_labels:
            raise ValueError("ContextPack labels must be contiguous and ordered")
        expected_mapping = tuple(
            (item.label, item.evidence_id, item.representation_id) for item in self.packed_blocks
        )
        actual_mapping = tuple(
            (item.label, item.evidence_id, item.representation_id) for item in self.label_mapping
        )
        if actual_mapping != expected_mapping:
            raise ValueError("ContextPack label mapping must match packed block identities")
        evidence_ids = tuple(item.evidence_id for item in self.packed_blocks)
        representation_ids = tuple(item.representation_id for item in self.packed_blocks)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("ContextPack packed Evidence identities must be unique")
        if len(representation_ids) != len(set(representation_ids)):
            raise ValueError("ContextPack representation identities must be unique")
        return self


def _obligation_shape(
    text: str, closure: FacetClosure
) -> tuple[ObligationKind, RequiredOutputForm]:
    normalized = "".join(text.split())
    if closure != FacetClosure.SINGLE:
        lowered = normalized.lower()
        if is_multi_step_procedure_request(normalized) or "step" in lowered:
            return ObligationKind.PROCEDURE, RequiredOutputForm.ORDERED_STEPS
        return ObligationKind.ENUMERATION, RequiredOutputForm.LIST
    if any(
        token in normalized for token in ("是否", "能否", "可否", "是不是", "对不对", "需不需要")
    ):
        return ObligationKind.BINARY, RequiredOutputForm.DIRECT_VERDICT
    # 输出结构必须服从 facet 已冻结的集合闭包。仅凭“如何/怎么/分别”等
    # 词面不能把 SINGLE 义务升级成步骤或比较集合，否则会制造伪基数证明。
    return ObligationKind.FACT, RequiredOutputForm.FACT


def build_question_contract(state: Mapping[str, Any]) -> QuestionContract:
    """只从原始 QueryPlan facet ledger 构建问题契约。"""

    plan = QueryPlan.model_validate(state.get("query_plan") or {})
    facets = plan.facet_ledger.facets
    if plan.disposition.value == "retrieve" and not facets:
        raise ValueError("retrieval QueryPlan must publish at least one original facet")
    kind_by_id: dict[str, ObligationKind] = {}
    form_by_id: dict[str, RequiredOutputForm] = {}
    for facet in facets:
        shape_text = (
            plan.understanding.standalone_query if len(facets) == 1 else facet.information_need
        )
        kind, form = _obligation_shape(shape_text, facet.closure)
        kind_by_id[facet.facet_id] = kind
        form_by_id[facet.facet_id] = form
    return QuestionContract.from_facets(
        query_plan_id=plan.plan_id,
        source_query=plan.understanding.standalone_query,
        facets=facets,
        builder_version=QUESTION_CONTRACT_BUILDER_VERSION,
        kind_by_obligation=kind_by_id,
        output_form_by_obligation=form_by_id,
    )


def _scope_hash(state: Mapping[str, Any]) -> str:
    raw = state.get("authorized_scope")
    if isinstance(raw, Mapping):
        value = str(raw.get("scope_hash") or "")
        if value:
            return value
    return _EMPTY_SCOPE_HASH


def _alias_maps(
    aliases: list[Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_label: dict[str, dict[str, Any]] = {}
    by_representation: dict[str, dict[str, Any]] = {}
    for raw in aliases:
        if not isinstance(raw, Mapping):
            raise ValueError("ContextPack alias must be a mapping")
        item = dict(raw)
        label = str(item.get("label") or "")
        document = item.get("document")
        representation_id = (
            str(document.get("representation_id") or "") if isinstance(document, Mapping) else ""
        )
        if label in by_label:
            raise ValueError("ContextPack alias labels must be unique")
        by_label[label] = item
        if representation_id:
            by_representation[representation_id] = item
    return by_label, by_representation


@dataclass(frozen=True, slots=True)
class _MappedSafeAtom:
    canonical_start: int
    canonical_end: int
    equivalence: SpanEquivalence
    control_only: bool = False


@dataclass(slots=True)
class _SafeSpanCursor:
    """按递增 atom 坐标线性消费 receipt spans。"""

    spans: tuple[SafeRepresentationSpan, ...]
    index: int = 0

    def overlaps(self, *, safe_start: int, safe_end: int) -> tuple[SafeRepresentationSpan, ...]:
        while self.index < len(self.spans) and self.spans[self.index].safe_end <= safe_start:
            self.index += 1
        end_index = self.index
        while end_index < len(self.spans) and self.spans[end_index].safe_start < safe_end:
            end_index += 1
        return self.spans[self.index : end_index]


def _validated_safe_derivation(
    alias: Mapping[str, Any],
    *,
    evidence: Evidence,
    replay_cache: dict[tuple[str, str, str], SafeRepresentationDerivationReceipt] | None = None,
) -> tuple[SafeRepresentationDerivationReceipt, str]:
    """从安全证据别名读取并重放 canonical↔safe receipt。"""

    document = alias.get("document")
    if not isinstance(document, Mapping):
        raise ValueError("ContextPack alias is missing its secured source document")
    raw_receipt = document.get("safe_representation_derivation")
    raw_safe = document.get("security_safe_content")
    if raw_receipt is None:
        raise ValueError("secured Evidence is missing its derivation receipt")
    if raw_safe is None:
        raise ValueError("safe derivation receipt is missing its safe representation")
    safe_text = str(raw_safe)
    receipt = SafeRepresentationDerivationReceipt.from_dict(raw_receipt)
    cache_key = (
        receipt.canonical_content_hash,
        receipt.safe_representation_hash,
        receipt.receipt_hash,
    )
    if (
        receipt.canonical_content_hash != evidence.content_hash
        or receipt.safe_representation_hash != sha256_digest(safe_text)
    ):
        raise ValueError("safe derivation artifact hashes no longer match active Evidence")
    cached = replay_cache.get(cache_key) if replay_cache is not None else None
    if cached is None:
        receipt.validate_artifacts(
            canonical_text=evidence.content,
            safe_text=safe_text,
        )
        if replay_cache is not None:
            replay_cache[cache_key] = receipt
    elif cached != receipt:
        raise ValueError("safe derivation replay cache identity collision")
    return receipt, safe_text


def _packed_source_text(block: _PackedBlockView) -> tuple[str, int | None]:
    """只依据 typed truncation receipt 移除 ContextPack 注入的末尾控制串。"""

    if block.truncation is None:
        return block.text, None
    marker = str(block.truncation.get("marker") or "")
    if marker != PACKED_TRUNCATION_MARKER or not block.text.endswith(marker):
        raise ValueError("typed ContextPack truncation marker is missing or displaced")
    truncation_start = len(block.text) - len(marker)
    if truncation_start <= 0:
        raise ValueError("typed ContextPack truncation removed the entire Evidence body")
    return block.text[:truncation_start], truncation_start


def _validated_context_pack(raw_pack: Mapping[str, Any]) -> _ContextPackV2View:
    """重放完整 ContextPack 身份后再缩窄为 answering 所需视图。"""

    full_pack = ContextPack.model_validate(raw_pack)
    if full_pack.schema_version != "context-pack.v2":
        raise ValueError("answering requires ContextPack v2")
    if replay_context_pack_content_hash(full_pack) != full_pack.content_hash:
        raise ValueError("ContextPack content hash no longer matches its replayable fields")
    return _ContextPackV2View.model_validate(full_pack.model_dump(mode="json"))


def _validate_packed_safe_relation(
    *,
    block: _PackedBlockView,
    packed_source_text: str,
    truncation_start: int | None,
    full_safe_text: str,
) -> None:
    """无截断时要求全文相等；typed 截断时验证前缀及字符计数。"""

    if truncation_start is None:
        if packed_source_text != full_safe_text:
            raise ValueError(
                "untruncated ContextPack block must equal the full safe representation"
            )
        return
    truncation = block.truncation or {}
    if (
        not full_safe_text.startswith(packed_source_text)
        or len(packed_source_text) >= len(full_safe_text)
        or truncation.get("original_characters") != len(full_safe_text)
        or truncation.get("packed_characters") != len(block.text)
    ):
        raise ValueError("typed ContextPack truncation does not match the safe representation")


def _canonical_boundary(
    span: SafeRepresentationSpan,
    *,
    safe_position: int,
) -> int | None:
    if safe_position == span.safe_start:
        return span.canonical_start
    if safe_position == span.safe_end:
        return span.canonical_end
    if span.offset_mapping == SafeSpanOffsetMapping.LINEAR:
        return span.canonical_start + safe_position - span.safe_start
    if (
        span.offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP
        and span.canonical_step is not None
        and span.safe_step is not None
    ):
        safe_delta = safe_position - span.safe_start
        if safe_delta % span.safe_step == 0:
            return span.canonical_start + (safe_delta // span.safe_step) * span.canonical_step
    return None


def _map_safe_atom(
    *,
    receipt: SafeRepresentationDerivationReceipt,
    safe_start: int,
    safe_end: int,
    span_cursor: _SafeSpanCursor | None = None,
) -> _MappedSafeAtom | None:
    """把 safe 半开区间映回 canonical；不能证明的内部边界返回 ``None``。"""

    overlaps = (
        span_cursor.overlaps(safe_start=safe_start, safe_end=safe_end)
        if span_cursor is not None
        else tuple(
            span
            for span in receipt.spans
            if span.safe_end > safe_start and span.safe_start < safe_end
        )
    )
    if (
        not overlaps
        or overlaps[0].safe_start > safe_start
        or overlaps[-1].safe_end < safe_end
        or any(
            span.relation
            in {
                SafeSpanRelation.MAPPING_BUDGET_EXCEEDED,
                SafeSpanRelation.UNPROVEN,
            }
            for span in overlaps
        )
    ):
        return None
    canonical_start = _canonical_boundary(overlaps[0], safe_position=safe_start)
    canonical_end = _canonical_boundary(overlaps[-1], safe_position=safe_end)
    if canonical_start is None or canonical_end is None or canonical_start >= canonical_end:
        return None
    control_only = all(
        span.relation == SafeSpanRelation.CONTROL_MARKDOWN_IMAGE for span in overlaps
    )
    if control_only:
        return _MappedSafeAtom(
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            equivalence=SpanEquivalence.SECURITY_SAFE_DERIVED,
            control_only=True,
        )
    if any(
        span.relation
        in {
            SafeSpanRelation.SECURITY_SAFE_DERIVED,
            SafeSpanRelation.CONTROL_MARKDOWN_IMAGE,
        }
        for span in overlaps
    ):
        equivalence = SpanEquivalence.SECURITY_SAFE_DERIVED
    elif any(span.relation == SafeSpanRelation.NFKC_EQUIVALENT for span in overlaps):
        equivalence = SpanEquivalence.NFKC_EQUIVALENT
    else:
        equivalence = SpanEquivalence.EXACT_UTF8
    return _MappedSafeAtom(
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        equivalence=equivalence,
    )


def build_witness_catalog_bundle(
    state: Mapping[str, Any],
    question_contract: QuestionContract,
) -> dict[str, Any]:
    """校验 ContextPack 三方映射并建立无正文 Witness artifacts。"""

    plan = QueryPlan.model_validate(state.get("query_plan") or {})
    scope_hash = _scope_hash(state)
    raw_pack = state.get("context_pack")
    if not isinstance(raw_pack, Mapping) or not raw_pack:
        if plan.disposition.value == "retrieve":
            raise ValueError("retrieval plan requires ContextPack before witness construction")
        context_pack_hash = sha256_digest("non-retrieval-empty-context")
        receipt = WitnessCatalogReceipt.build(
            question_contract_id=question_contract.contract_id,
            question_contract_hash=question_contract.contract_hash,
            context_pack_hash=context_pack_hash,
            authorized_scope_hash=scope_hash,
            witnesses=(),
            universe_status=WitnessUniverseStatus.COMPLETE,
            builder_version=WITNESS_CATALOG_BUILDER_VERSION,
        )
        return {
            "schema_version": "witness-catalog-bundle.v3",
            "receipt": receipt.to_dict(),
            "witnesses": [],
            "lineage": [],
            "source_enumeration_closure_receipts": [],
        }

    pack = _validated_context_pack(raw_pack)
    aliases = list(state.get("context_aliases") or [])
    by_label, _ = _alias_maps(aliases)
    block_labels = [item.label for item in pack.packed_blocks]
    if set(by_label) != set(block_labels) or len(by_label) != len(block_labels):
        raise ValueError("ContextPack blocks and aliases must have the same labels")

    witnesses: list[EvidenceWitness] = []
    lineage: list[dict[str, Any]] = []
    catalog_safe_text_by_label: dict[str, str] = {}
    omissions: Counter[WitnessOmissionCode] = Counter()
    if pack.omitted:
        omissions[WitnessOmissionCode.CONTEXT_BUDGET] += len(pack.omitted)
    for block in pack.packed_blocks:
        alias = by_label[block.label]
        raw_evidence = alias.get("evidence")
        if not isinstance(raw_evidence, Mapping):
            raise ValueError("ContextPack alias is missing Evidence")
        evidence = Evidence.model_validate(raw_evidence)
        if (
            evidence.evidence_id != block.evidence_id
            or evidence.content_hash != block.original_content_hash
            or str(alias.get("packed_text") or "") != block.text
        ):
            raise ValueError("ContextPack block, label mapping and alias disagree")
        packed_source_text, truncation_start = _packed_source_text(block)
        if truncation_start is not None:
            omissions[WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED] += 1
        try:
            safe_receipt, full_safe_text = _validated_safe_derivation(
                alias,
                evidence=evidence,
            )
        except ValueError:
            atoms = atomize_evidence_text(packed_source_text)
            omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += max(1, len(atoms))
            continue
        try:
            _validate_packed_safe_relation(
                block=block,
                packed_source_text=packed_source_text,
                truncation_start=truncation_start,
                full_safe_text=full_safe_text,
            )
        except ValueError:
            omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += 1
            continue
        if safe_receipt.mapping_budget_exceeded:
            omissions[WitnessOmissionCode.SAFE_MAPPING_BUDGET_EXCEEDED] += 1
            continue
        if safe_receipt.has_unproven_transform:
            omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += max(
                1,
                safe_receipt.unproven_transform_count,
            )
            continue
        catalog_safe_text_by_label[block.label] = full_safe_text
        protected_spans: list[tuple[int, int] | tuple[int, int, int]] = []
        for span in safe_receipt.spans:
            if span.safe_start >= len(packed_source_text):
                break
            protected_end = span.safe_end
            if protected_end <= span.safe_start:
                continue
            if span.offset_mapping == SafeSpanOffsetMapping.ENDPOINTS_ONLY:
                protected_spans.append((span.safe_start, protected_end))
            elif (
                span.offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP
                and span.safe_step is not None
            ):
                protected_spans.append((span.safe_start, protected_end, span.safe_step))
        try:
            atoms = atomize_evidence_text(
                packed_source_text,
                protected_spans=protected_spans,
            )
        except ValueError:
            omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += 1
            continue
        span_cursor = _SafeSpanCursor(safe_receipt.spans)
        for atom_index, atom in enumerate(atoms):
            if len(witnesses) >= MAX_WITNESS_CATALOG_ITEMS:
                omitted_atoms = 0
                for remaining in atoms[atom_index:]:
                    remaining_mapping = _map_safe_atom(
                        receipt=safe_receipt,
                        safe_start=remaining.start,
                        safe_end=remaining.end,
                        span_cursor=span_cursor,
                    )
                    if remaining_mapping is None:
                        if truncation_start is not None and remaining.end == truncation_start:
                            continue
                        omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += 1
                    elif not remaining_mapping.control_only:
                        omitted_atoms += 1
                omissions[WitnessOmissionCode.ATOM_LIMIT] += omitted_atoms
                break
            mapped = _map_safe_atom(
                receipt=safe_receipt,
                safe_start=atom.start,
                safe_end=atom.end,
                span_cursor=span_cursor,
            )
            if mapped is None:
                if truncation_start is not None and atom.end == truncation_start:
                    # 末端变换只因 typed ContextPack 截断而不完整，已有确定性 truncation omission。
                    continue
                omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += 1
                continue
            if mapped.control_only:
                # 只由安全层生成的控制占位不属于模型可引用的事实宇宙。
                continue
            source_start = mapped.canonical_start
            source_end = mapped.canonical_end
            equivalence = mapped.equivalence
            try:
                span_ref = EvidenceSpanRef.from_texts(
                    evidence_id=block.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=block.text,
                    canonical_start=source_start,
                    canonical_end=source_end,
                    packed_start=atom.start,
                    packed_end=atom.end,
                    equivalence=equivalence,
                    safe_derivation_transform_version=(
                        SAFE_REPRESENTATION_TRANSFORM_VERSION
                        if equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED
                        else None
                    ),
                    safe_derivation_receipt_hash=(
                        safe_receipt.receipt_hash
                        if equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED
                        else None
                    ),
                )
            except ValueError:
                omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] += 1
                continue
            witness = EvidenceWitness.build(
                context_pack_hash=pack.content_hash,
                authorized_scope_hash=scope_hash,
                span_ref=span_ref,
                atom_kind=WitnessAtomKind(atom.kind),
                ordinal=len(witnesses),
            )
            witnesses.append(witness)
            lineage.append(
                {
                    "witness_id": witness.witness_id,
                    "label": block.label,
                    "source_sub_query_ids": list(block.sub_query_ids),
                }
            )

    if omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN]:
        universe = WitnessUniverseStatus.INDETERMINATE
    elif omissions:
        universe = WitnessUniverseStatus.TRUNCATED
    else:
        universe = WitnessUniverseStatus.COMPLETE
    omission_facts = tuple(
        WitnessCatalogOmission(code=code, count=count)
        for code, count in sorted(omissions.items(), key=lambda item: item[0].value)
        if count
    )
    receipt = WitnessCatalogReceipt.build(
        question_contract_id=question_contract.contract_id,
        question_contract_hash=question_contract.contract_hash,
        context_pack_hash=pack.content_hash,
        authorized_scope_hash=scope_hash,
        witnesses=tuple(witnesses),
        universe_status=universe,
        omissions=omission_facts,
        builder_version=WITNESS_CATALOG_BUILDER_VERSION,
    )
    source_closure_receipts: list[SourceEnumerationClosureReceipt] = []
    if receipt.universe_status == WitnessUniverseStatus.COMPLETE:
        if set(catalog_safe_text_by_label) != set(block_labels):
            raise ValueError("complete witness catalog is missing a validated representation")
        catalog_texts = tuple(
            catalog_safe_text_by_label[block.label] for block in pack.packed_blocks
        )
        scoped_authorities: list[ScopedListAuthorityRepresentation] = []
        fact_authorities: list[FactSetAuthorityRepresentation] = []
        for block in pack.packed_blocks:
            alias = by_label[block.label]
            raw_evidence = alias.get("evidence")
            if not isinstance(raw_evidence, Mapping):
                raise ValueError("scoped procedure authority is missing Evidence")
            evidence = Evidence.model_validate(raw_evidence)
            scoped_authorities.append(
                ScopedListAuthorityRepresentation(
                    evidence_id=evidence.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=block.text,
                )
            )
            fact_authorities.append(
                FactSetAuthorityRepresentation(
                    evidence_id=evidence.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=block.text,
                )
            )
        lineage_by_id = {str(item["witness_id"]): item for item in lineage}
        open_obligations = tuple(
            item
            for item in question_contract.obligations
            if item.closure == FacetClosure.OPEN_ENUMERATION
        )
        for obligation in open_obligations:
            obligation_receipt_start = len(source_closure_receipts)
            table_receipt_start = len(source_closure_receipts)
            for witness in witnesses:
                if witness.atom_kind != WitnessAtomKind.TABLE_DATA_ROW:
                    continue
                witness_lineage = lineage_by_id[witness.witness_id]
                label = str(witness_lineage["label"])
                alias = by_label[label]
                raw_evidence = alias.get("evidence")
                block = next(item for item in pack.packed_blocks if item.label == label)
                if not isinstance(raw_evidence, Mapping):
                    raise ValueError("source closure witness is missing Evidence")
                evidence = Evidence.model_validate(raw_evidence)
                try:
                    source_closure_receipts.append(
                        QueryAnchoredTableRowClosureReceipt.build(
                            question=question_contract,
                            catalog=receipt,
                            obligation=obligation,
                            witness=witness,
                            evidence_id=evidence.evidence_id,
                            representation_id=block.representation_id,
                            canonical_text=evidence.content,
                            packed_text=block.text,
                            catalog_texts=catalog_texts,
                        )
                    )
                except ValueError:
                    # 大多数表格行与当前开放枚举问题无显式标识绑定，不生成闭包即可。
                    continue
            legacy_table_receipts = source_closure_receipts[table_receipt_start:]
            if len(legacy_table_receipts) != 1:
                del source_closure_receipts[table_receipt_start:]
                table_witnesses = tuple(
                    witness
                    for witness in witnesses
                    if witness.atom_kind == WitnessAtomKind.TABLE_DATA_ROW
                )
                if table_witnesses and obligation.required_output_form == RequiredOutputForm.LIST:
                    try:
                        source_closure_receipts.append(
                            QueryAnchoredScopedTableSelectionClosureReceipt.build(
                                question=question_contract,
                                catalog=receipt,
                                obligation=obligation,
                                witnesses=table_witnesses,
                                authority_representations=tuple(scoped_authorities),
                            )
                        )
                    except ValueError:
                        # 表、selector、列或 authority 任一不唯一时保持开放枚举。
                        pass
            if (
                obligation.required_output_form == RequiredOutputForm.LIST
                and len(source_closure_receipts) == obligation_receipt_start
            ):
                bullet_witnesses = tuple(
                    witness
                    for witness in witnesses
                    if witness.atom_kind == WitnessAtomKind.LIST_ITEM
                )
                if bullet_witnesses:
                    try:
                        source_closure_receipts.append(
                            QueryAnchoredScopedBulletListClosureReceipt.build(
                                question=question_contract,
                                catalog=receipt,
                                obligation=obligation,
                                witnesses=bullet_witnesses,
                                authority_representations=tuple(scoped_authorities),
                            )
                        )
                    except ValueError:
                        # bullet 范围、锚点或完整 authority 不唯一时保持开放枚举。
                        pass
            if (
                obligation.required_output_form == RequiredOutputForm.LIST
                and len(source_closure_receipts) == obligation_receipt_start
            ):
                fact_witnesses = tuple(
                    witness
                    for witness in witnesses
                    if witness.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
                )
                if fact_witnesses:
                    try:
                        source_closure_receipts.append(
                            build_query_anchored_scoped_fact_set_closure_receipt(
                                question=question_contract,
                                catalog=receipt,
                                obligation=obligation,
                                witnesses=fact_witnesses,
                                authority_representations=tuple(fact_authorities),
                                catalog_witnesses=tuple(witnesses),
                            )
                        )
                    except ValueError:
                        # 事实关系、查询锚点或完整 authority 不唯一时保持开放枚举。
                        pass
            if obligation.required_output_form == RequiredOutputForm.ORDERED_STEPS:
                list_witnesses = tuple(
                    witness
                    for witness in witnesses
                    if witness.atom_kind == WitnessAtomKind.LIST_ITEM
                )
                issued_legacy_receipt = False
                if list_witnesses and len(pack.packed_blocks) == 1:
                    block = pack.packed_blocks[0]
                    alias = by_label[block.label]
                    raw_evidence = alias.get("evidence")
                    if not isinstance(raw_evidence, Mapping):
                        raise ValueError("procedure list closure is missing Evidence")
                    evidence = Evidence.model_validate(raw_evidence)
                    try:
                        source_closure_receipts.append(
                            QueryAnchoredProcedureListClosureReceipt.build(
                                question=question_contract,
                                catalog=receipt,
                                obligation=obligation,
                                witnesses=list_witnesses,
                                evidence_id=evidence.evidence_id,
                                representation_id=block.representation_id,
                                canonical_text=evidence.content,
                                packed_text=block.text,
                                catalog_texts=catalog_texts,
                            )
                        )
                        issued_legacy_receipt = True
                    except ValueError:
                        # 只有完整、标题锚定且连续编号的唯一 representation 才签发。
                        pass
                if list_witnesses and not issued_legacy_receipt:
                    try:
                        source_closure_receipts.append(
                            QueryAnchoredScopedProcedureListClosureReceipt.build(
                                question=question_contract,
                                catalog=receipt,
                                obligation=obligation,
                                witnesses=list_witnesses,
                                authority_representations=tuple(scoped_authorities),
                            )
                        )
                    except ValueError:
                        # 局部流程必须在全部 authority 中查询锚定且唯一。
                        pass
                if len(source_closure_receipts) == obligation_receipt_start:
                    inline_witnesses = tuple(
                        witness
                        for witness in witnesses
                        if witness.atom_kind
                        in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
                    )
                    if inline_witnesses:
                        try:
                            source_closure_receipts.append(
                                QueryAnchoredInlineActionChainClosureReceipt.build(
                                    question=question_contract,
                                    catalog=receipt,
                                    obligation=obligation,
                                    witnesses=inline_witnesses,
                                    authority_representations=tuple(scoped_authorities),
                                    catalog_witnesses=tuple(witnesses),
                                )
                            )
                        except ValueError:
                            # 单原子动作链必须由当前义务局部锚定且唯一。
                            pass
        candidate_count_by_obligation = Counter(
            item.obligation_id for item in source_closure_receipts
        )
        source_closure_receipts = [
            item
            for item in source_closure_receipts
            if candidate_count_by_obligation[item.obligation_id] == 1
        ]
    return {
        "schema_version": "witness-catalog-bundle.v3",
        "receipt": receipt.to_dict(),
        "witnesses": [item.to_dict() for item in witnesses],
        "lineage": lineage,
        "source_enumeration_closure_receipts": [item.to_dict() for item in source_closure_receipts],
    }


def resolve_witness_prompt_views(
    state: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    safe_replay_cache: dict[tuple[str, str, str], SafeRepresentationDerivationReceipt]
    | None = None,
) -> list[dict[str, Any]]:
    """在每次模型/渲染边界重算 span，并仅瞬时物化 Witness 正文。"""

    if bundle.get("schema_version") != "witness-catalog-bundle.v3":
        raise ValueError("witness catalog bundle schema is unsupported")

    aliases = list(state.get("context_aliases") or [])
    by_label, _ = _alias_maps(aliases)
    raw_pack = state.get("context_pack")
    if not isinstance(raw_pack, Mapping):
        raise ValueError("witness resolution requires its ContextPack")
    pack = _validated_context_pack(raw_pack)
    block_by_label = {item.label: item for item in pack.packed_blocks}
    lineage_by_id = {
        str(item.get("witness_id") or ""): dict(item)
        for item in bundle.get("lineage") or []
        if isinstance(item, Mapping)
    }
    result: list[dict[str, Any]] = []
    witnesses = tuple(
        EvidenceWitness.from_dict(item)
        for item in bundle.get("witnesses") or []
        if isinstance(item, Mapping)
    )
    witness_by_id = {item.witness_id: item for item in witnesses}
    if len(witness_by_id) != len(witnesses):
        raise ValueError("witness catalog contains duplicate witness IDs")
    receipt = WitnessCatalogReceipt.from_dict(bundle.get("receipt") or {})
    receipt.validate_witnesses(witnesses)
    question = QuestionContract.from_dict(state.get("question_contract") or {})
    obligations_by_id = {item.obligation_id: item for item in question.obligations}
    raw_source_receipts = bundle.get("source_enumeration_closure_receipts")
    if not isinstance(raw_source_receipts, list):
        raise ValueError("witness catalog bundle is missing source closure receipts")
    source_receipts = tuple(
        parse_source_enumeration_closure_receipt(item)
        for item in raw_source_receipts
        if isinstance(item, Mapping)
    )
    if len(source_receipts) != len(raw_source_receipts):
        raise ValueError("source closure receipts must be mappings")
    source_receipt_ids = tuple(item.receipt_id for item in source_receipts)
    if len(source_receipt_ids) != len(set(source_receipt_ids)):
        raise ValueError("source closure receipt IDs must be unique")
    source_receipt_pairs = tuple(
        (item.obligation_id, source_closure_witness_ids(item)) for item in source_receipts
    )
    if len(source_receipt_pairs) != len(set(source_receipt_pairs)):
        raise ValueError("source closure obligation/witness pairs must be unique")
    source_receipt_obligation_ids = tuple(item.obligation_id for item in source_receipts)
    if len(source_receipt_obligation_ids) != len(set(source_receipt_obligation_ids)):
        raise ValueError("each open obligation may carry at most one source closure receipt")
    source_receipts_by_witness: dict[str, list[SourceEnumerationClosureReceipt]] = {}
    source_closed_ids_by_witness: dict[str, list[str]] = {}
    for source_receipt in source_receipts:
        member_ids = source_closure_witness_ids(source_receipt)
        source_receipts_by_witness.setdefault(member_ids[0], []).append(source_receipt)
        for member_id in member_ids:
            source_closed_ids_by_witness.setdefault(member_id, []).append(
                source_receipt.obligation_id
            )
    validated_source_receipt_ids: set[str] = set()
    validated_derivations: dict[
        str,
        tuple[SafeRepresentationDerivationReceipt, str],
    ] = {}
    replay_cache = safe_replay_cache if safe_replay_cache is not None else {}
    catalog_texts: tuple[str, ...] = ()
    scoped_authorities: tuple[ScopedListAuthorityRepresentation, ...] = ()
    fact_authorities: tuple[FactSetAuthorityRepresentation, ...] = ()
    if source_receipts:
        validated_catalog_texts: list[str] = []
        validated_scoped_authorities: list[ScopedListAuthorityRepresentation] = []
        validated_fact_authorities: list[FactSetAuthorityRepresentation] = []
        for block in pack.packed_blocks:
            alias = by_label.get(block.label)
            if alias is None or not isinstance(alias.get("evidence"), Mapping):
                raise ValueError("source closure catalog representation is missing Evidence")
            evidence = Evidence.model_validate(alias["evidence"])
            packed = str(alias.get("packed_text") or "")
            if (
                evidence.evidence_id != block.evidence_id
                or evidence.content_hash != block.original_content_hash
                or packed != block.text
            ):
                raise ValueError("source closure catalog representation no longer resolves")
            derivation = _validated_safe_derivation(
                alias,
                evidence=evidence,
                replay_cache=replay_cache,
            )
            safe_receipt, full_safe_text = derivation
            packed_source, truncation_start = _packed_source_text(block)
            _validate_packed_safe_relation(
                block=block,
                packed_source_text=packed_source,
                truncation_start=truncation_start,
                full_safe_text=full_safe_text,
            )
            if safe_receipt.has_unproven_transform or safe_receipt.mapping_budget_exceeded:
                raise ValueError("source closure catalog representation is not fully mapped")
            validated_derivations[block.label] = derivation
            validated_catalog_texts.append(full_safe_text)
            validated_scoped_authorities.append(
                ScopedListAuthorityRepresentation(
                    evidence_id=evidence.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=block.text,
                )
            )
            validated_fact_authorities.append(
                FactSetAuthorityRepresentation(
                    evidence_id=evidence.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=block.text,
                )
            )
        catalog_texts = tuple(validated_catalog_texts)
        scoped_authorities = tuple(validated_scoped_authorities)
        fact_authorities = tuple(validated_fact_authorities)
    for witness in witnesses:
        lineage = lineage_by_id.get(witness.witness_id)
        if lineage is None:
            raise ValueError("witness lineage is missing")
        alias = by_label.get(str(lineage.get("label") or ""))
        if alias is None or not isinstance(alias.get("evidence"), Mapping):
            raise ValueError("witness lineage points outside ContextPack aliases")
        evidence = Evidence.model_validate(alias["evidence"])
        packed = str(alias.get("packed_text") or "")
        label = str(lineage.get("label") or "")
        block = block_by_label.get(label)
        if block is None or block.text != packed:
            raise ValueError("witness lineage no longer resolves to its ContextPack block")
        packed_source, _ = _packed_source_text(block)
        derivation = validated_derivations.get(label)
        if derivation is None:
            derivation = _validated_safe_derivation(
                alias,
                evidence=evidence,
                replay_cache=replay_cache,
            )
            validated_derivations[label] = derivation
        safe_receipt, full_safe_text = derivation
        _, truncation_start = _packed_source_text(block)
        try:
            _validate_packed_safe_relation(
                block=block,
                packed_source_text=packed_source,
                truncation_start=truncation_start,
                full_safe_text=full_safe_text,
            )
        except ValueError as exc:
            raise ValueError("witness safe derivation no longer resolves") from exc
        if safe_receipt.has_unproven_transform or safe_receipt.mapping_budget_exceeded:
            raise ValueError("witness safe derivation no longer resolves")
        if witness.span_ref.equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED:
            if (
                witness.span_ref.safe_derivation_transform_version != safe_receipt.transform_version
                or witness.span_ref.safe_derivation_receipt_hash != safe_receipt.receipt_hash
            ):
                raise ValueError("witness safe derivation identity mismatch")
        witness.span_ref.validate_texts(canonical_text=evidence.content, packed_text=packed)
        content = packed[witness.span_ref.packed_start : witness.span_ref.packed_end]
        validated_source_receipts: list[SourceEnumerationClosureReceipt] = []
        for source_receipt in source_receipts_by_witness.get(witness.witness_id, []):
            obligation = obligations_by_id.get(source_receipt.obligation_id)
            if obligation is None:
                raise ValueError("source closure receipt references an unknown obligation")
            if isinstance(source_receipt, QueryAnchoredInlineActionChainClosureReceipt):
                member_ids = source_closure_witness_ids(source_receipt)
                if any(member_id not in witness_by_id for member_id in member_ids):
                    raise ValueError("inline action-chain closure references an unknown witness")
                inline_witnesses = tuple(
                    item
                    for item in witnesses
                    if item.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
                )
                source_receipt.validate_artifacts(
                    question=question,
                    catalog=receipt,
                    obligation=obligation,
                    witnesses=inline_witnesses,
                    authority_representations=scoped_authorities,
                    catalog_witnesses=witnesses,
                )
            elif isinstance(source_receipt, QueryAnchoredScopedFactSetClosureReceipt):
                member_ids = source_closure_witness_ids(source_receipt)
                if any(member_id not in witness_by_id for member_id in member_ids):
                    raise ValueError("fact-set closure references an unknown witness")
                fact_witnesses = tuple(
                    item
                    for item in witnesses
                    if item.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
                )
                source_receipt.validate_artifacts(
                    question=question,
                    catalog=receipt,
                    obligation=obligation,
                    witnesses=fact_witnesses,
                    authority_representations=fact_authorities,
                    catalog_witnesses=witnesses,
                )
            elif isinstance(
                source_receipt,
                (
                    QueryAnchoredScopedTableSelectionClosureReceipt,
                    QueryAnchoredScopedBulletListClosureReceipt,
                ),
            ):
                member_witnesses: list[EvidenceWitness] = []
                for member_id in source_closure_witness_ids(source_receipt):
                    member = witness_by_id.get(member_id)
                    member_lineage = lineage_by_id.get(member_id)
                    if member is None or member_lineage is None:
                        raise ValueError("scoped source closure references an unknown witness")
                    if str(member_lineage.get("label") or "") != label:
                        raise ValueError(
                            "scoped source closure crosses ContextPack representations"
                        )
                    member_witnesses.append(member)
                source_receipt.validate_artifacts(
                    question=question,
                    catalog=receipt,
                    obligation=obligation,
                    witnesses=tuple(member_witnesses),
                    authority_representations=scoped_authorities,
                )
            elif isinstance(source_receipt, QueryAnchoredTableRowClosureReceipt):
                source_receipt.validate_artifacts(
                    question=question,
                    catalog=receipt,
                    obligation=obligation,
                    witness=witness,
                    evidence_id=evidence.evidence_id,
                    representation_id=block.representation_id,
                    canonical_text=evidence.content,
                    packed_text=packed,
                    catalog_texts=catalog_texts,
                )
            else:
                member_witnesses: list[EvidenceWitness] = []
                for member_id in source_closure_witness_ids(source_receipt):
                    member = witness_by_id.get(member_id)
                    member_lineage = lineage_by_id.get(member_id)
                    if member is None or member_lineage is None:
                        raise ValueError("procedure list closure references an unknown witness")
                    if str(member_lineage.get("label") or "") != label:
                        raise ValueError(
                            "procedure list closure crosses ContextPack representations"
                        )
                    member_witnesses.append(member)
                if isinstance(
                    source_receipt,
                    QueryAnchoredScopedProcedureListClosureReceipt,
                ):
                    source_receipt.validate_artifacts(
                        question=question,
                        catalog=receipt,
                        obligation=obligation,
                        witnesses=tuple(member_witnesses),
                        authority_representations=scoped_authorities,
                    )
                else:
                    source_receipt.validate_artifacts(
                        question=question,
                        catalog=receipt,
                        obligation=obligation,
                        witnesses=tuple(member_witnesses),
                        evidence_id=evidence.evidence_id,
                        representation_id=block.representation_id,
                        canonical_text=evidence.content,
                        packed_text=packed,
                        catalog_texts=catalog_texts,
                    )
            validated_source_receipts.append(source_receipt)
            validated_source_receipt_ids.add(source_receipt.receipt_id)
        result.append(
            {
                "witness_id": witness.witness_id,
                "content": content,
                "source_sub_query_ids": tuple(lineage.get("source_sub_query_ids") or ()),
                "label": label,
                "evidence_id": witness.span_ref.evidence_id,
                "witness": witness,
                "source_closed_open_obligation_ids": tuple(
                    source_closed_ids_by_witness.get(witness.witness_id, ())
                ),
                "source_enumeration_closure_receipts": tuple(validated_source_receipts),
            }
        )
    if validated_source_receipt_ids != set(source_receipt_ids):
        raise ValueError("source closure receipt does not resolve to the frozen witness graph")
    return result


def collect_source_enumeration_closure_receipts(
    views: list[Mapping[str, Any]],
) -> tuple[SourceEnumerationClosureReceipt, ...]:
    """从已重放的瞬时视图收集去重后的来源闭包回执。"""

    collected: list[SourceEnumerationClosureReceipt] = []
    seen: set[str] = set()
    for view in views:
        witness_id = str(view.get("witness_id") or "")
        for raw_receipt in view.get("source_enumeration_closure_receipts") or ():
            receipt = (
                raw_receipt
                if isinstance(
                    raw_receipt,
                    (
                        QueryAnchoredTableRowClosureReceipt,
                        QueryAnchoredInlineActionChainClosureReceipt,
                        QueryAnchoredProcedureListClosureReceipt,
                        QueryAnchoredScopedBulletListClosureReceipt,
                        QueryAnchoredScopedTableSelectionClosureReceipt,
                        QueryAnchoredScopedFactSetClosureReceipt,
                    ),
                )
                else parse_source_enumeration_closure_receipt(raw_receipt)
            )
            if source_closure_witness_ids(receipt)[0] != witness_id:
                raise ValueError("source closure receipt belongs to another witness view")
            if receipt.receipt_id in seen:
                raise ValueError("source closure receipt is duplicated across witness views")
            seen.add(receipt.receipt_id)
            collected.append(receipt)
    return tuple(collected)


__all__ = [
    "QUESTION_CONTRACT_BUILDER_VERSION",
    "WITNESS_CATALOG_BUILDER_VERSION",
    "build_question_contract",
    "build_witness_catalog_bundle",
    "collect_source_enumeration_closure_receipts",
    "resolve_witness_prompt_views",
]
