"""Claim、Obligation 与最终 Answer 的可重放证明图。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
import re
from typing import Any, Literal, Self
import unicodedata

from pydantic import Field, field_validator, model_validator

from app.domain.answering._base import (
    CLAIM_ID_PATTERN,
    FACET_ID_PATTERN,
    SHA256_PATTERN,
    WITNESS_ID_PATTERN,
    AnsweringContractModel,
    require_unique,
    sha256_digest,
    stable_content_hash,
    stable_identifier,
)
from app.domain.answering.failures import AnsweringFailure, AnsweringFailureCode
from app.domain.answering.fact_set_closure import QueryAnchoredScopedFactSetClosureReceipt
from app.domain.answering.enumeration_closure import (
    QueryAnchoredInlineActionChainClosureReceipt,
    QueryAnchoredScopedBulletListClosureReceipt,
    QueryAnchoredScopedTableSelectionClosureReceipt,
    QueryAnchoredTableRowClosureReceipt,
    SourceEnumerationClosureReceipt,
    source_closure_witness_ids,
)
from app.domain.answering.item_boundary import (
    ClosedCountTableCellProjectionStatus,
    closed_count_markdown_table_cell_item_spans,
    markdown_table_data_row_body_span,
    resolve_closed_count_markdown_table_cell_projection,
    top_level_list_item_body_span,
)
from app.domain.answering.plan import (
    AnswerDecision,
    AnswerPlan,
    RenderingRule,
)
from app.domain.answering.question import QuestionContract, RequiredOutputForm
from app.domain.answering.witness import (
    EvidenceWitness,
    WitnessAtomKind,
    WitnessCatalogReceipt,
    WitnessUniverseStatus,
)
from app.domain.evidence import AnswerSpan
from app.domain.query_planning import FacetClosure


CLAIM_PROOF_SCHEMA_VERSION = "claim-proof.v1"
PROVEN_ITEM_REF_SCHEMA_VERSION = "proven-item-ref.v1"
OBLIGATION_PROOF_SCHEMA_VERSION = "obligation-proof.v3"
SEMANTIC_REVIEW_CLOSURE_SCHEMA_VERSION = "semantic-review-closure.v2"
SEMANTIC_REVIEW_TERMINAL_SCHEMA_VERSION = "semantic-review-terminal.v1"
ANSWER_PROOF_SCHEMA_VERSION = "answer-proof.v3"


class ProofVerdict(StrEnum):
    """单项证明的封闭三态。"""

    PROVEN = "proven"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class ProofRelation(StrEnum):
    """Witness 与可见 claim 的关系。"""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"
    NOT_APPLICABLE = "not_applicable"


class ObligationOutcome(StrEnum):
    """一个问题义务最终获得的业务结论。"""

    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    EVIDENCE_ABSENT = "evidence_absent"
    CONFLICTING = "conflicting"
    INSUFFICIENT_RETRIEVED_EVIDENCE = "insufficient_retrieved_evidence"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class ObligationClosureStatus(StrEnum):
    """义务集合边界是否由当前 proof 显式闭合。"""

    PROVEN_COMPLETE = "proven_complete"
    INCOMPLETE = "incomplete"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class ItemSegmentationMethod(StrEnum):
    """服务端允许声明业务 item 的封闭句法边界。"""

    SINGLE_WITNESS_ATOM = "single_witness_atom.v1"
    OPEN_ENUMERATION_WITNESS_ATOM = "open_enumeration_witness_atom.v1"
    TOP_LEVEL_LIST_ENTRY = "top_level_list_entry.v1"
    MARKDOWN_TABLE_DATA_ROW = "markdown_table_data_row.v1"
    MARKDOWN_TABLE_PROJECTED_CELL_ITEM = "markdown_table_projected_cell_item.v1"
    INLINE_ACTION_SPAN = "inline_action_span.v1"


def _inline_action_relative_spans(
    receipt: QueryAnchoredInlineActionChainClosureReceipt,
) -> tuple[tuple[int, int], ...]:
    """按动作序号把 packed authority 坐标转换为 Witness 相对坐标。"""

    ordered = sorted(
        zip(
            receipt.action_ordinals,
            receipt.packed_action_starts,
            receipt.packed_action_ends,
            strict=True,
        ),
        key=lambda item: item[0],
    )
    return tuple(
        (
            start - receipt.packed_witness_start,
            end - receipt.packed_witness_start,
        )
        for _ordinal, start, end in ordered
    )


class OverallProofStatus(StrEnum):
    """AnswerProof 是否可提交及其提交形态。"""

    ACCEPTED = "accepted"
    ACCEPTED_PARTIAL = "accepted_partial"
    ACCEPTED_REFUSAL = "accepted_refusal"
    ACCEPTED_ABSTENTION = "accepted_abstention"
    NOT_APPLICABLE = "not_applicable"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class ProofDimensionStatus(StrEnum):
    """Grounding 与 relevance 的独立状态。"""

    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class CompletenessProofStatus(StrEnum):
    """问题义务完整性状态。"""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class VerifierRef(AnsweringContractModel):
    """参与证明的确定性 verifier 身份。"""

    verifier_id: str = Field(min_length=1, max_length=128)
    verifier_version: str = Field(min_length=1, max_length=128)

    @field_validator("verifier_id", "verifier_version")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("verifier identifiers must be trimmed")
        return value


def _failure_code_values(codes: tuple[AnsweringFailureCode, ...]) -> tuple[str, ...]:
    values = tuple(item.value for item in codes)
    return require_unique(values, field_name="failure_codes")


def _claim_proof_payload(
    *,
    claim_id: str,
    obligation_id: str | None,
    status: ProofVerdict,
    method: RenderingRule,
    witness_ids: tuple[str, ...],
    answer_span: AnswerSpan | None,
    rendered_text_hash: str | None,
    relation: ProofRelation,
    verifier: VerifierRef,
    failure_codes: tuple[AnsweringFailureCode, ...],
) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_PROOF_SCHEMA_VERSION,
        "claim_id": claim_id,
        "obligation_id": obligation_id,
        "status": status.value,
        "method": method.value,
        "witness_ids": list(witness_ids),
        "answer_span": answer_span.model_dump(mode="json") if answer_span else None,
        "rendered_text_hash": rendered_text_hash,
        "relation": relation.value,
        "verifier": verifier.to_dict(),
        "failure_codes": list(_failure_code_values(failure_codes)),
    }


class ClaimProof(AnsweringContractModel):
    """渲染 claim 的身份、证据、相关性和答案坐标证明。"""

    schema_version: Literal["claim-proof.v1"] = CLAIM_PROOF_SCHEMA_VERSION
    claim_proof_id: str = Field(pattern=r"^cpr_[0-9a-f]{40}$")
    claim_proof_hash: str = Field(pattern=SHA256_PATTERN)
    claim_id: str = Field(pattern=CLAIM_ID_PATTERN)
    obligation_id: str | None = Field(default=None, pattern=FACET_ID_PATTERN)
    status: ProofVerdict
    method: RenderingRule
    witness_ids: tuple[str, ...] = Field(default=(), max_length=16)
    answer_span: AnswerSpan | None = None
    rendered_text_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    relation: ProofRelation
    verifier: VerifierRef
    failure_codes: tuple[AnsweringFailureCode, ...] = Field(default=(), max_length=32)

    @field_validator("witness_ids")
    @classmethod
    def validate_witness_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(value, field_name="witness_ids")
        if any(re.fullmatch(WITNESS_ID_PATTERN, item) is None for item in value):
            raise ValueError("claim proof contains an invalid witness identity")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _claim_proof_payload(
            claim_id=self.claim_id,
            obligation_id=self.obligation_id,
            status=self.status,
            method=self.method,
            witness_ids=self.witness_ids,
            answer_span=self.answer_span,
            rendered_text_hash=self.rendered_text_hash,
            relation=self.relation,
            verifier=self.verifier,
            failure_codes=self.failure_codes,
        )

    @model_validator(mode="after")
    def validate_claim_proof(self) -> Self:
        _failure_code_values(self.failure_codes)
        if (self.answer_span is None) != (self.rendered_text_hash is None):
            raise ValueError("answer_span and rendered_text_hash must appear together")
        if self.status == ProofVerdict.PROVEN:
            if self.answer_span is None or self.failure_codes:
                raise ValueError("proven claim requires a rendered span and no failure codes")
        elif not self.failure_codes:
            raise ValueError("non-proven claim requires failure_codes")

        if self.method == RenderingRule.SYSTEM_NOTICE:
            if self.witness_ids or self.relation != ProofRelation.NOT_APPLICABLE:
                raise ValueError("system_notice proof cannot claim an Evidence relation")
        else:
            if not self.witness_ids or self.obligation_id is None:
                raise ValueError("factual claim proof requires witnesses and obligation_id")
            if self.relation == ProofRelation.NOT_APPLICABLE:
                raise ValueError("factual claim proof requires an Evidence relation")
            if self.method == RenderingRule.EXACT_WITNESS and len(self.witness_ids) != 1:
                raise ValueError("exact_witness proof requires exactly one witness")
            if self.status == ProofVerdict.PROVEN and self.relation != ProofRelation.SUPPORTS:
                raise ValueError("proven visible claim requires a supporting relation")
        payload = self._identity_payload()
        if self.claim_proof_hash != stable_content_hash(payload):
            raise ValueError("claim_proof_hash does not match ClaimProof")
        if self.claim_proof_id != stable_identifier("cpr_", payload):
            raise ValueError("claim_proof_id does not match ClaimProof")
        return self

    @classmethod
    def build(
        cls,
        *,
        claim_id: str,
        obligation_id: str | None,
        status: ProofVerdict,
        method: RenderingRule,
        witness_ids: tuple[str, ...],
        answer_span: AnswerSpan | None,
        rendered_text_hash: str | None,
        relation: ProofRelation,
        verifier: VerifierRef,
        failure_codes: tuple[AnsweringFailureCode, ...] = (),
    ) -> "ClaimProof":
        """建立可重放 claim proof 并生成稳定身份。"""

        payload = _claim_proof_payload(
            claim_id=claim_id,
            obligation_id=obligation_id,
            status=status,
            method=method,
            witness_ids=witness_ids,
            answer_span=answer_span,
            rendered_text_hash=rendered_text_hash,
            relation=relation,
            verifier=verifier,
            failure_codes=failure_codes,
        )
        return cls(
            **payload,
            claim_proof_id=stable_identifier("cpr_", payload),
            claim_proof_hash=stable_content_hash(payload),
        )


def _normalized_item_hash(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return sha256_digest(normalized)


def _proven_item_ref_payload(
    *,
    obligation_id: str,
    claim_id: str,
    witness_id: str,
    witness_hash: str,
    relative_start: int,
    relative_end: int,
    raw_span_hash: str,
    normalized_item_hash: str,
    segmentation_method: ItemSegmentationMethod,
) -> dict[str, Any]:
    return {
        "schema_version": PROVEN_ITEM_REF_SCHEMA_VERSION,
        "obligation_id": obligation_id,
        "claim_id": claim_id,
        "witness_id": witness_id,
        "witness_hash": witness_hash,
        "relative_start": relative_start,
        "relative_end": relative_end,
        "raw_span_hash": raw_span_hash,
        "normalized_item_hash": normalized_item_hash,
        "segmentation_method": segmentation_method.value,
    }


class ProvenItemRef(AnsweringContractModel):
    """绑定最终 claim、Witness 与可重放相对 span 的业务 item 回执。"""

    schema_version: Literal["proven-item-ref.v1"] = PROVEN_ITEM_REF_SCHEMA_VERSION
    item_ref_id: str = Field(pattern=r"^pit_[0-9a-f]{40}$")
    item_ref_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    claim_id: str = Field(pattern=CLAIM_ID_PATTERN)
    witness_id: str = Field(pattern=WITNESS_ID_PATTERN)
    witness_hash: str = Field(pattern=SHA256_PATTERN)
    relative_start: int = Field(ge=0)
    relative_end: int = Field(gt=0)
    raw_span_hash: str = Field(pattern=SHA256_PATTERN)
    normalized_item_hash: str = Field(pattern=SHA256_PATTERN)
    segmentation_method: ItemSegmentationMethod

    def _identity_payload(self) -> dict[str, Any]:
        return _proven_item_ref_payload(
            obligation_id=self.obligation_id,
            claim_id=self.claim_id,
            witness_id=self.witness_id,
            witness_hash=self.witness_hash,
            relative_start=self.relative_start,
            relative_end=self.relative_end,
            raw_span_hash=self.raw_span_hash,
            normalized_item_hash=self.normalized_item_hash,
            segmentation_method=self.segmentation_method,
        )

    @model_validator(mode="after")
    def validate_item_ref(self) -> Self:
        if self.relative_end <= self.relative_start:
            raise ValueError("item ref must satisfy relative_start < relative_end")
        payload = self._identity_payload()
        if self.item_ref_hash != stable_content_hash(payload):
            raise ValueError("item_ref_hash does not match ProvenItemRef")
        if self.item_ref_id != stable_identifier("pit_", payload):
            raise ValueError("item_ref_id does not match ProvenItemRef")
        return self

    @classmethod
    def build(
        cls,
        *,
        obligation_id: str,
        claim_id: str,
        witness: EvidenceWitness,
        witness_text: str,
        relative_start: int,
        relative_end: int,
        segmentation_method: ItemSegmentationMethod,
    ) -> "ProvenItemRef":
        """由已解析 Witness 正文建立无正文 item 回执。"""

        if not 0 <= relative_start < relative_end <= len(witness_text):
            raise ValueError("item ref lies outside its Witness text")
        raw_span = witness_text[relative_start:relative_end]
        if not raw_span.strip():
            raise ValueError("item ref cannot point to blank text")
        payload = _proven_item_ref_payload(
            obligation_id=obligation_id,
            claim_id=claim_id,
            witness_id=witness.witness_id,
            witness_hash=witness.witness_hash,
            relative_start=relative_start,
            relative_end=relative_end,
            raw_span_hash=sha256_digest(raw_span),
            normalized_item_hash=_normalized_item_hash(raw_span),
            segmentation_method=segmentation_method,
        )
        return cls(
            **payload,
            item_ref_id=stable_identifier("pit_", payload),
            item_ref_hash=stable_content_hash(payload),
        )

    def validate_witness_text(self, witness: EvidenceWitness, witness_text: str) -> Self:
        """重放 item 相对坐标、Witness 身份与正文哈希。"""

        if self.witness_id != witness.witness_id or self.witness_hash != witness.witness_hash:
            raise ValueError("item ref belongs to a different Witness")
        if not 0 <= self.relative_start < self.relative_end <= len(witness_text):
            raise ValueError("item ref lies outside its Witness text")
        raw_span = witness_text[self.relative_start : self.relative_end]
        if (
            sha256_digest(raw_span) != self.raw_span_hash
            or _normalized_item_hash(raw_span) != self.normalized_item_hash
        ):
            raise ValueError("item ref text hashes do not match the Witness span")
        return self


def _obligation_proof_payload(
    *,
    obligation_id: str,
    status: ProofVerdict,
    outcome: ObligationOutcome,
    supporting_claim_ids: tuple[str, ...],
    contradiction_witness_ids: tuple[str, ...],
    closure: FacetClosure,
    required_output_form: RequiredOutputForm,
    expected_item_count: int | None,
    proven_item_count: int,
    proven_items: tuple[ProvenItemRef, ...],
    closure_status: ObligationClosureStatus,
    source_enumeration_closure_receipt: SourceEnumerationClosureReceipt | None,
    catalog_id: str,
    catalog_hash: str,
    catalog_universe_status: WitnessUniverseStatus,
    verifier: VerifierRef,
    failure_codes: tuple[AnsweringFailureCode, ...],
) -> dict[str, Any]:
    return {
        "schema_version": OBLIGATION_PROOF_SCHEMA_VERSION,
        "obligation_id": obligation_id,
        "status": status.value,
        "outcome": outcome.value,
        "supporting_claim_ids": list(supporting_claim_ids),
        "contradiction_witness_ids": list(contradiction_witness_ids),
        "closure": closure.value,
        "required_output_form": required_output_form.value,
        "expected_item_count": expected_item_count,
        "proven_item_count": proven_item_count,
        "proven_items": [item.to_dict() for item in proven_items],
        "closure_status": closure_status.value,
        "source_enumeration_closure_receipt": (
            source_enumeration_closure_receipt.to_dict()
            if source_enumeration_closure_receipt is not None
            else None
        ),
        "catalog_id": catalog_id,
        "catalog_hash": catalog_hash,
        "catalog_universe_status": catalog_universe_status.value,
        "verifier": verifier.to_dict(),
        "failure_codes": list(_failure_code_values(failure_codes)),
    }


class ObligationProof(AnsweringContractModel):
    """回答、缺证、冲突或未知的逐义务证明。"""

    schema_version: Literal["obligation-proof.v3"] = OBLIGATION_PROOF_SCHEMA_VERSION
    obligation_proof_id: str = Field(pattern=r"^opr_[0-9a-f]{40}$")
    obligation_proof_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    status: ProofVerdict
    outcome: ObligationOutcome
    supporting_claim_ids: tuple[str, ...] = Field(default=(), max_length=64)
    contradiction_witness_ids: tuple[str, ...] = Field(default=(), max_length=32)
    closure: FacetClosure
    required_output_form: RequiredOutputForm
    expected_item_count: int | None = Field(default=None, ge=1, le=128)
    proven_item_count: int = Field(ge=0, le=64)
    proven_items: tuple[ProvenItemRef, ...] = Field(default=(), max_length=64)
    closure_status: ObligationClosureStatus
    source_enumeration_closure_receipt: SourceEnumerationClosureReceipt | None = None
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_universe_status: WitnessUniverseStatus
    verifier: VerifierRef
    failure_codes: tuple[AnsweringFailureCode, ...] = Field(default=(), max_length=32)

    @field_validator("supporting_claim_ids", "contradiction_witness_ids")
    @classmethod
    def validate_reference_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return require_unique(value, field_name="proof references")

    def _identity_payload(self) -> dict[str, Any]:
        return _obligation_proof_payload(
            obligation_id=self.obligation_id,
            status=self.status,
            outcome=self.outcome,
            supporting_claim_ids=self.supporting_claim_ids,
            contradiction_witness_ids=self.contradiction_witness_ids,
            closure=self.closure,
            required_output_form=self.required_output_form,
            expected_item_count=self.expected_item_count,
            proven_item_count=self.proven_item_count,
            proven_items=self.proven_items,
            closure_status=self.closure_status,
            source_enumeration_closure_receipt=self.source_enumeration_closure_receipt,
            catalog_id=self.catalog_id,
            catalog_hash=self.catalog_hash,
            catalog_universe_status=self.catalog_universe_status,
            verifier=self.verifier,
            failure_codes=self.failure_codes,
        )

    @model_validator(mode="after")
    def validate_obligation_proof(self) -> Self:
        _failure_code_values(self.failure_codes)
        if self.proven_item_count != len(self.proven_items):
            raise ValueError("proven_item_count must equal len(proven_items)")
        source_closure = self.source_enumeration_closure_receipt
        normalized_hashes = tuple(item.normalized_item_hash for item in self.proven_items)
        if isinstance(source_closure, QueryAnchoredInlineActionChainClosureReceipt):
            require_unique(
                tuple(item.item_ref_hash for item in self.proven_items),
                field_name="inline action item identities",
            )
        else:
            require_unique(normalized_hashes, field_name="proven item normalized identities")
        if self.closure == FacetClosure.CLOSED_COUNT:
            if self.expected_item_count is None:
                raise ValueError("closed_count proof requires expected_item_count")
        elif self.expected_item_count is not None:
            raise ValueError("only closed_count proof may carry expected_item_count")
        if self.status == ProofVerdict.PROVEN:
            if self.failure_codes:
                raise ValueError("proven obligation cannot carry failure_codes")
        elif not self.failure_codes:
            raise ValueError("non-proven obligation requires failure_codes")

        if (
            self.outcome == ObligationOutcome.ANSWERED
            and self.closure == FacetClosure.OPEN_ENUMERATION
        ):
            if source_closure is None:
                raise ValueError(
                    "answered open enumeration requires a bound source closure receipt"
                )
            witness_ids = tuple(item.witness_id for item in self.proven_items)
            witness_hashes = tuple(item.witness_hash for item in self.proven_items)
            common_binding_invalid = (
                source_closure.obligation_id != self.obligation_id
                or source_closure.catalog_id != self.catalog_id
                or source_closure.catalog_hash != self.catalog_hash
            )
            if isinstance(source_closure, QueryAnchoredInlineActionChainClosureReceipt):
                expected_spans = _inline_action_relative_spans(source_closure)
                expected_claim_id = (
                    self.supporting_claim_ids[0] if len(self.supporting_claim_ids) == 1 else None
                )
                invalid = (
                    common_binding_invalid
                    or self.required_output_form != RequiredOutputForm.ORDERED_STEPS
                    or source_closure.action_ordinals
                    != tuple(range(1, source_closure.action_count + 1))
                    or self.proven_item_count != source_closure.action_count
                    or expected_claim_id is None
                    or witness_ids != (source_closure.witness_ids[0],) * source_closure.action_count
                    or witness_hashes
                    != (source_closure.witness_hashes[0],) * source_closure.action_count
                    or tuple(item.claim_id for item in self.proven_items)
                    != (expected_claim_id,) * source_closure.action_count
                    or tuple((item.relative_start, item.relative_end) for item in self.proven_items)
                    != expected_spans
                    or tuple(item.raw_span_hash for item in self.proven_items)
                    != source_closure.packed_action_hashes
                    or any(
                        item.segmentation_method != ItemSegmentationMethod.INLINE_ACTION_SPAN
                        for item in self.proven_items
                    )
                )
            else:
                common_invalid = (
                    common_binding_invalid
                    or witness_ids != source_closure_witness_ids(source_closure)
                    or self.proven_item_count != len(self.supporting_claim_ids)
                    or tuple(item.claim_id for item in self.proven_items)
                    != self.supporting_claim_ids
                )
            if isinstance(source_closure, QueryAnchoredTableRowClosureReceipt):
                invalid = (
                    common_invalid
                    or self.proven_item_count != 1
                    or witness_hashes != (source_closure.witness_hash,)
                    or self.proven_items[0].segmentation_method
                    != ItemSegmentationMethod.MARKDOWN_TABLE_DATA_ROW
                )
            elif isinstance(source_closure, QueryAnchoredInlineActionChainClosureReceipt):
                pass
            elif isinstance(
                source_closure,
                QueryAnchoredScopedTableSelectionClosureReceipt,
            ):
                invalid = (
                    common_invalid
                    or self.required_output_form != RequiredOutputForm.LIST
                    or witness_hashes != source_closure.witness_hashes
                    or any(
                        item.segmentation_method != ItemSegmentationMethod.MARKDOWN_TABLE_DATA_ROW
                        for item in self.proven_items
                    )
                )
            elif isinstance(source_closure, QueryAnchoredScopedBulletListClosureReceipt):
                invalid = (
                    common_invalid
                    or self.required_output_form != RequiredOutputForm.LIST
                    or witness_hashes != source_closure.witness_hashes
                    or any(
                        item.segmentation_method != ItemSegmentationMethod.TOP_LEVEL_LIST_ENTRY
                        for item in self.proven_items
                    )
                )
            elif isinstance(source_closure, QueryAnchoredScopedFactSetClosureReceipt):
                invalid = (
                    common_invalid
                    or self.required_output_form != RequiredOutputForm.LIST
                    or witness_hashes != source_closure.witness_hashes
                    or tuple(item.segmentation_method.value for item in self.proven_items)
                    != source_closure.segmentation_methods
                )
            else:
                invalid = (
                    common_invalid
                    or self.required_output_form != RequiredOutputForm.ORDERED_STEPS
                    or witness_hashes != source_closure.witness_hashes
                    or any(
                        item.segmentation_method != ItemSegmentationMethod.TOP_LEVEL_LIST_ENTRY
                        for item in self.proven_items
                    )
                )
            if invalid:
                raise ValueError(
                    "answered open enumeration does not match its source closure receipt"
                )
        elif source_closure is not None:
            raise ValueError(
                "source enumeration closure receipt is reserved for answered open enumeration"
            )

        if self.outcome == ObligationOutcome.ANSWERED:
            if (
                self.status != ProofVerdict.PROVEN
                or not self.supporting_claim_ids
                or self.contradiction_witness_ids
                or self.proven_item_count < 1
                or self.closure_status != ObligationClosureStatus.PROVEN_COMPLETE
            ):
                raise ValueError("answered obligation requires only supporting claims")
            if self.catalog_universe_status != WitnessUniverseStatus.COMPLETE:
                raise ValueError("answered obligation requires a complete witness catalog")
            if self.closure == FacetClosure.SINGLE and (
                len(self.supporting_claim_ids) != 1 or self.proven_item_count != 1
            ):
                raise ValueError(
                    "answered single obligation requires exactly one claim and one item"
                )
            if (
                self.closure == FacetClosure.CLOSED_COUNT
                and self.proven_item_count != self.expected_item_count
            ):
                raise ValueError("answered closed_count obligation requires exact cardinality")
            if self.closure == FacetClosure.CLOSED_COUNT and (
                self.proven_item_count < len(self.supporting_claim_ids)
                or {item.claim_id for item in self.proven_items} != set(self.supporting_claim_ids)
            ):
                raise ValueError(
                    "answered closed_count obligation requires every claim to prove an item"
                )
        elif self.outcome == ObligationOutcome.PARTIALLY_ANSWERED:
            if (
                self.status != ProofVerdict.PROVEN
                or not self.supporting_claim_ids
                or self.contradiction_witness_ids
                or self.proven_item_count < 1
                or self.closure_status
                not in {
                    ObligationClosureStatus.INCOMPLETE,
                    ObligationClosureStatus.INDETERMINATE,
                }
            ):
                raise ValueError("partially_answered obligation requires proven supporting claims")
        elif self.outcome == ObligationOutcome.EVIDENCE_ABSENT:
            if (
                self.supporting_claim_ids
                or self.contradiction_witness_ids
                or self.proven_item_count
                or self.closure_status != ObligationClosureStatus.NOT_APPLICABLE
            ):
                raise ValueError("evidence_absent cannot carry claim/witness support")
            if self.status == ProofVerdict.PROVEN:
                raise ValueError(
                    "proven evidence absence requires an exhaustive retrieval closure receipt"
                )
        elif self.outcome == ObligationOutcome.INSUFFICIENT_RETRIEVED_EVIDENCE:
            if (
                self.status != ProofVerdict.PROVEN
                or self.supporting_claim_ids
                or self.contradiction_witness_ids
                or self.proven_item_count
                or self.closure_status != ObligationClosureStatus.NOT_APPLICABLE
            ):
                raise ValueError("local insufficiency cannot claim answer or closure proof")
        elif self.outcome == ObligationOutcome.CONFLICTING:
            if (
                self.supporting_claim_ids
                or len(self.contradiction_witness_ids) < 2
                or self.proven_item_count
                or self.closure_status != ObligationClosureStatus.NOT_APPLICABLE
            ):
                raise ValueError("conflicting obligation requires at least two witnesses")
        elif self.outcome == ObligationOutcome.INDETERMINATE:
            if self.status != ProofVerdict.INDETERMINATE:
                raise ValueError("indeterminate outcome requires indeterminate status")
            if self.supporting_claim_ids:
                raise ValueError("indeterminate obligation cannot claim answer support")
            if self.proven_item_count or self.closure_status not in {
                ObligationClosureStatus.INDETERMINATE,
                ObligationClosureStatus.NOT_APPLICABLE,
            }:
                raise ValueError("indeterminate obligation cannot claim closure or item proof")
        else:
            if self.status != ProofVerdict.PROVEN:
                raise ValueError("not_applicable obligation must be proven")
            if (
                self.supporting_claim_ids
                or self.contradiction_witness_ids
                or self.proven_item_count
                or self.closure_status != ObligationClosureStatus.NOT_APPLICABLE
            ):
                raise ValueError("not_applicable obligation cannot carry proof references")

        payload = self._identity_payload()
        if self.obligation_proof_hash != stable_content_hash(payload):
            raise ValueError("obligation_proof_hash does not match ObligationProof")
        if self.obligation_proof_id != stable_identifier("opr_", payload):
            raise ValueError("obligation_proof_id does not match ObligationProof")
        return self

    @classmethod
    def build(
        cls,
        *,
        obligation_id: str,
        status: ProofVerdict,
        outcome: ObligationOutcome,
        supporting_claim_ids: tuple[str, ...],
        contradiction_witness_ids: tuple[str, ...],
        closure: FacetClosure,
        required_output_form: RequiredOutputForm,
        expected_item_count: int | None,
        proven_items: tuple[ProvenItemRef, ...],
        closure_status: ObligationClosureStatus,
        source_enumeration_closure_receipt: SourceEnumerationClosureReceipt | None = None,
        catalog: WitnessCatalogReceipt,
        verifier: VerifierRef,
        failure_codes: tuple[AnsweringFailureCode, ...] = (),
    ) -> "ObligationProof":
        """建立绑定到唯一 WitnessCatalog 的义务证明。"""

        payload = _obligation_proof_payload(
            obligation_id=obligation_id,
            status=status,
            outcome=outcome,
            supporting_claim_ids=supporting_claim_ids,
            contradiction_witness_ids=contradiction_witness_ids,
            closure=closure,
            required_output_form=required_output_form,
            expected_item_count=expected_item_count,
            proven_item_count=len(proven_items),
            proven_items=proven_items,
            closure_status=closure_status,
            source_enumeration_closure_receipt=source_enumeration_closure_receipt,
            catalog_id=catalog.catalog_id,
            catalog_hash=catalog.catalog_hash,
            catalog_universe_status=catalog.universe_status,
            verifier=verifier,
            failure_codes=failure_codes,
        )
        return cls(
            **payload,
            obligation_proof_id=stable_identifier("opr_", payload),
            obligation_proof_hash=stable_content_hash(payload),
        )


def _semantic_review_closure_payload(
    *,
    answer_plan_id: str,
    answer_plan_hash: str,
    catalog_id: str,
    catalog_hash: str,
    closed_obligation_ids: tuple[str, ...],
    review_prompt_version: str,
    model_id: str,
    verifier_version: str,
    review_logical_call_id: str,
    review_prompt_hash: str,
    review_response_hash: str,
    review_proposal_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_REVIEW_CLOSURE_SCHEMA_VERSION,
        "answer_plan_id": answer_plan_id,
        "answer_plan_hash": answer_plan_hash,
        "catalog_id": catalog_id,
        "catalog_hash": catalog_hash,
        "decision": "accept",
        "closed_obligation_ids": list(closed_obligation_ids),
        "review_prompt_version": review_prompt_version,
        "model_id": model_id,
        "verifier_version": verifier_version,
        "review_logical_call_id": review_logical_call_id,
        "review_prompt_hash": review_prompt_hash,
        "review_response_hash": review_response_hash,
        "review_proposal_hash": review_proposal_hash,
    }


class SemanticReviewClosureReceipt(AnsweringContractModel):
    """绑定最终计划、目录与 reviewer 身份的显式闭包声明。"""

    schema_version: Literal["semantic-review-closure.v2"] = SEMANTIC_REVIEW_CLOSURE_SCHEMA_VERSION
    receipt_id: str = Field(pattern=r"^src_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    answer_plan_id: str = Field(pattern=r"^apn_[0-9a-f]{40}$")
    answer_plan_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    decision: Literal["accept"] = "accept"
    closed_obligation_ids: tuple[str, ...] = Field(default=(), max_length=32)
    review_prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=256)
    verifier_version: str = Field(min_length=1, max_length=128)
    review_logical_call_id: str = Field(min_length=1, max_length=128)
    review_prompt_hash: str = Field(pattern=SHA256_PATTERN)
    review_response_hash: str = Field(pattern=SHA256_PATTERN)
    review_proposal_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("closed_obligation_ids")
    @classmethod
    def validate_closed_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(value, field_name="closed_obligation_ids")
        if any(re.fullmatch(FACET_ID_PATTERN, item) is None for item in value):
            raise ValueError("closed_obligation_ids contain an invalid identity")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _semantic_review_closure_payload(
            answer_plan_id=self.answer_plan_id,
            answer_plan_hash=self.answer_plan_hash,
            catalog_id=self.catalog_id,
            catalog_hash=self.catalog_hash,
            closed_obligation_ids=self.closed_obligation_ids,
            review_prompt_version=self.review_prompt_version,
            model_id=self.model_id,
            verifier_version=self.verifier_version,
            review_logical_call_id=self.review_logical_call_id,
            review_prompt_hash=self.review_prompt_hash,
            review_response_hash=self.review_response_hash,
            review_proposal_hash=self.review_proposal_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match SemanticReviewClosureReceipt")
        if self.receipt_id != stable_identifier("src_", payload):
            raise ValueError("receipt_id does not match SemanticReviewClosureReceipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        answer_plan: AnswerPlan,
        catalog: WitnessCatalogReceipt,
        closed_obligation_ids: tuple[str, ...],
        review_prompt_version: str,
        model_id: str,
        verifier_version: str,
        review_logical_call_id: str,
        review_prompt_hash: str,
        review_response_hash: str,
        review_proposal_hash: str,
    ) -> "SemanticReviewClosureReceipt":
        payload = _semantic_review_closure_payload(
            answer_plan_id=answer_plan.plan_id,
            answer_plan_hash=answer_plan.plan_hash,
            catalog_id=catalog.catalog_id,
            catalog_hash=catalog.catalog_hash,
            closed_obligation_ids=closed_obligation_ids,
            review_prompt_version=review_prompt_version,
            model_id=model_id,
            verifier_version=verifier_version,
            review_logical_call_id=review_logical_call_id,
            review_prompt_hash=review_prompt_hash,
            review_response_hash=review_response_hash,
            review_proposal_hash=review_proposal_hash,
        )
        return cls(
            **payload,
            receipt_id=stable_identifier("src_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_against(
        self,
        *,
        answer_plan: AnswerPlan,
        catalog: WitnessCatalogReceipt,
    ) -> Self:
        if (
            self.answer_plan_id != answer_plan.plan_id
            or self.answer_plan_hash != answer_plan.plan_hash
            or self.catalog_id != catalog.catalog_id
            or self.catalog_hash != catalog.catalog_hash
        ):
            raise ValueError("semantic review closure references a different plan/catalog")
        return self


def _semantic_review_terminal_payload(
    *,
    answer_plan_id: str,
    answer_plan_hash: str,
    reviewed_answer_plan_id: str,
    reviewed_answer_plan_hash: str,
    catalog_id: str,
    catalog_hash: str,
    decision: Literal["accept", "reject", "repair"],
    reason_code: str,
    reviewed_closed_obligation_ids: tuple[str, ...],
    review_prompt_version: str,
    model_id: str,
    verifier_version: str,
    review_logical_call_id: str,
    review_prompt_hash: str,
    review_response_hash: str,
    review_proposal_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_REVIEW_TERMINAL_SCHEMA_VERSION,
        "answer_plan_id": answer_plan_id,
        "answer_plan_hash": answer_plan_hash,
        "reviewed_answer_plan_id": reviewed_answer_plan_id,
        "reviewed_answer_plan_hash": reviewed_answer_plan_hash,
        "catalog_id": catalog_id,
        "catalog_hash": catalog_hash,
        "decision": decision,
        "reason_code": reason_code,
        "closed_obligation_ids": [],
        "reviewed_closed_obligation_ids": list(reviewed_closed_obligation_ids),
        "review_prompt_version": review_prompt_version,
        "model_id": model_id,
        "verifier_version": verifier_version,
        "review_logical_call_id": review_logical_call_id,
        "review_prompt_hash": review_prompt_hash,
        "review_response_hash": review_response_hash,
        "review_proposal_hash": review_proposal_hash,
    }


class SemanticReviewTerminalReceipt(AnsweringContractModel):
    """把 reviewer 的拒绝或修复上限绑定为无事实正文的安全弃答。"""

    schema_version: Literal["semantic-review-terminal.v1"] = SEMANTIC_REVIEW_TERMINAL_SCHEMA_VERSION
    receipt_id: str = Field(pattern=r"^srt_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    answer_plan_id: str = Field(pattern=r"^apn_[0-9a-f]{40}$")
    answer_plan_hash: str = Field(pattern=SHA256_PATTERN)
    reviewed_answer_plan_id: str = Field(pattern=r"^apn_[0-9a-f]{40}$")
    reviewed_answer_plan_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    decision: Literal["accept", "reject", "repair"]
    reason_code: Literal[
        "SEMANTIC_REVIEW_REJECTED",
        "REPAIR_LIMIT_REACHED",
        "OBLIGATION_CLOSURE_UNPROVEN",
        "OUTPUT_SECURITY_TRANSFORMED",
    ]
    closed_obligation_ids: tuple[()] = ()
    reviewed_closed_obligation_ids: tuple[str, ...] = Field(default=(), max_length=32)
    review_prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=256)
    verifier_version: str = Field(min_length=1, max_length=128)
    review_logical_call_id: str = Field(min_length=1, max_length=128)
    review_prompt_hash: str = Field(pattern=SHA256_PATTERN)
    review_response_hash: str = Field(pattern=SHA256_PATTERN)
    review_proposal_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("reviewed_closed_obligation_ids")
    @classmethod
    def validate_reviewed_closed_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique(value, field_name="reviewed_closed_obligation_ids")
        if any(re.fullmatch(FACET_ID_PATTERN, item) is None for item in value):
            raise ValueError("reviewed_closed_obligation_ids contain an invalid identity")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _semantic_review_terminal_payload(
            answer_plan_id=self.answer_plan_id,
            answer_plan_hash=self.answer_plan_hash,
            reviewed_answer_plan_id=self.reviewed_answer_plan_id,
            reviewed_answer_plan_hash=self.reviewed_answer_plan_hash,
            catalog_id=self.catalog_id,
            catalog_hash=self.catalog_hash,
            decision=self.decision,
            reason_code=self.reason_code,
            reviewed_closed_obligation_ids=self.reviewed_closed_obligation_ids,
            review_prompt_version=self.review_prompt_version,
            model_id=self.model_id,
            verifier_version=self.verifier_version,
            review_logical_call_id=self.review_logical_call_id,
            review_prompt_hash=self.review_prompt_hash,
            review_response_hash=self.review_response_hash,
            review_proposal_hash=self.review_proposal_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected_reasons = {
            "accept": {
                "OBLIGATION_CLOSURE_UNPROVEN",
                "OUTPUT_SECURITY_TRANSFORMED",
            },
            "reject": {
                "SEMANTIC_REVIEW_REJECTED",
                "REPAIR_LIMIT_REACHED",
            },
            "repair": {"REPAIR_LIMIT_REACHED"},
        }[self.decision]
        if self.reason_code not in expected_reasons:
            raise ValueError("terminal reason does not match the reviewer decision")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match SemanticReviewTerminalReceipt")
        if self.receipt_id != stable_identifier("srt_", payload):
            raise ValueError("receipt_id does not match SemanticReviewTerminalReceipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        answer_plan: AnswerPlan,
        reviewed_answer_plan: AnswerPlan,
        catalog: WitnessCatalogReceipt,
        decision: Literal["accept", "reject", "repair"],
        reason_code: Literal[
            "SEMANTIC_REVIEW_REJECTED",
            "REPAIR_LIMIT_REACHED",
            "OBLIGATION_CLOSURE_UNPROVEN",
            "OUTPUT_SECURITY_TRANSFORMED",
        ],
        reviewed_closed_obligation_ids: tuple[str, ...],
        review_prompt_version: str,
        model_id: str,
        verifier_version: str,
        review_logical_call_id: str,
        review_prompt_hash: str,
        review_response_hash: str,
        review_proposal_hash: str,
    ) -> "SemanticReviewTerminalReceipt":
        payload = _semantic_review_terminal_payload(
            answer_plan_id=answer_plan.plan_id,
            answer_plan_hash=answer_plan.plan_hash,
            reviewed_answer_plan_id=reviewed_answer_plan.plan_id,
            reviewed_answer_plan_hash=reviewed_answer_plan.plan_hash,
            catalog_id=catalog.catalog_id,
            catalog_hash=catalog.catalog_hash,
            decision=decision,
            reason_code=reason_code,
            reviewed_closed_obligation_ids=reviewed_closed_obligation_ids,
            review_prompt_version=review_prompt_version,
            model_id=model_id,
            verifier_version=verifier_version,
            review_logical_call_id=review_logical_call_id,
            review_prompt_hash=review_prompt_hash,
            review_response_hash=review_response_hash,
            review_proposal_hash=review_proposal_hash,
        )
        return cls(
            **payload,
            receipt_id=stable_identifier("srt_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_against(
        self,
        *,
        answer_plan: AnswerPlan,
        reviewed_answer_plan: AnswerPlan,
        catalog: WitnessCatalogReceipt,
    ) -> Self:
        if (
            self.answer_plan_id != answer_plan.plan_id
            or self.answer_plan_hash != answer_plan.plan_hash
            or self.reviewed_answer_plan_id != reviewed_answer_plan.plan_id
            or self.reviewed_answer_plan_hash != reviewed_answer_plan.plan_hash
            or self.catalog_id != catalog.catalog_id
            or self.catalog_hash != catalog.catalog_hash
            or answer_plan.decision != AnswerDecision.REFUSE
            or answer_plan.substantive_claims
        ):
            raise ValueError("semantic review terminal receipt references another graph")
        return self


SemanticReviewReceipt = SemanticReviewClosureReceipt | SemanticReviewTerminalReceipt


def semantic_review_receipt_from_dict(value: Mapping[str, Any]) -> SemanticReviewReceipt:
    """按显式 schema 解析 ACCEPT 闭包或终态安全弃答回执。"""

    schema_version = value.get("schema_version")
    if schema_version == SEMANTIC_REVIEW_CLOSURE_SCHEMA_VERSION:
        return SemanticReviewClosureReceipt.from_dict(value)
    if schema_version == SEMANTIC_REVIEW_TERMINAL_SCHEMA_VERSION:
        return SemanticReviewTerminalReceipt.from_dict(value)
    raise ValueError("unsupported semantic review receipt schema")


def _answer_proof_payload(
    *,
    answer_plan_id: str,
    answer_plan_hash: str,
    question_contract_id: str,
    question_contract_hash: str,
    context_pack_hash: str | None,
    witness_catalog_id: str | None,
    witness_catalog_hash: str | None,
    semantic_review_closure: SemanticReviewReceipt | None,
    answer_id: str | None,
    answer_hash: str | None,
    decision: AnswerDecision,
    overall_status: OverallProofStatus,
    grounding_status: ProofDimensionStatus,
    relevance_status: ProofDimensionStatus,
    completeness_status: CompletenessProofStatus,
    strict_eligible: bool,
    claim_proofs: tuple[ClaimProof, ...],
    obligation_proofs: tuple[ObligationProof, ...],
    verifier_versions: tuple[VerifierRef, ...],
    failures: tuple[AnsweringFailure, ...],
) -> dict[str, Any]:
    return {
        "schema_version": ANSWER_PROOF_SCHEMA_VERSION,
        "answer_plan_id": answer_plan_id,
        "answer_plan_hash": answer_plan_hash,
        "question_contract_id": question_contract_id,
        "question_contract_hash": question_contract_hash,
        "context_pack_hash": context_pack_hash,
        "witness_catalog_id": witness_catalog_id,
        "witness_catalog_hash": witness_catalog_hash,
        "semantic_review_closure": (
            semantic_review_closure.to_dict() if semantic_review_closure else None
        ),
        "answer_id": answer_id,
        "answer_hash": answer_hash,
        "decision": decision.value,
        "overall_status": overall_status.value,
        "grounding_status": grounding_status.value,
        "relevance_status": relevance_status.value,
        "completeness_status": completeness_status.value,
        "strict_eligible": strict_eligible,
        "claim_proofs": [item.to_dict() for item in claim_proofs],
        "obligation_proofs": [item.to_dict() for item in obligation_proofs],
        "verifier_versions": [item.to_dict() for item in verifier_versions],
        "failures": [item.to_dict() for item in failures],
    }


class AnswerProof(AnsweringContractModel):
    """提交 AnswerEnvelope 前唯一权威、可重放的完整证明图。"""

    schema_version: Literal["answer-proof.v3"] = ANSWER_PROOF_SCHEMA_VERSION
    proof_id: str = Field(pattern=r"^apr_[0-9a-f]{40}$")
    proof_hash: str = Field(pattern=SHA256_PATTERN)
    answer_plan_id: str = Field(pattern=r"^apn_[0-9a-f]{40}$")
    answer_plan_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    context_pack_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    witness_catalog_id: str | None = Field(
        default=None,
        pattern=r"^wcat_[0-9a-f]{40}$",
    )
    witness_catalog_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    semantic_review_closure: SemanticReviewReceipt | None = None
    answer_id: str | None = Field(default=None, pattern=r"^ans_[0-9a-f]{32}$")
    answer_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decision: AnswerDecision
    overall_status: OverallProofStatus
    grounding_status: ProofDimensionStatus
    relevance_status: ProofDimensionStatus
    completeness_status: CompletenessProofStatus
    strict_eligible: bool = False
    claim_proofs: tuple[ClaimProof, ...] = Field(default=(), max_length=64)
    obligation_proofs: tuple[ObligationProof, ...] = Field(default=(), max_length=32)
    verifier_versions: tuple[VerifierRef, ...] = Field(default=(), max_length=16)
    failures: tuple[AnsweringFailure, ...] = Field(default=(), max_length=64)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value

    def _identity_payload(self) -> dict[str, Any]:
        return _answer_proof_payload(
            answer_plan_id=self.answer_plan_id,
            answer_plan_hash=self.answer_plan_hash,
            question_contract_id=self.question_contract_id,
            question_contract_hash=self.question_contract_hash,
            context_pack_hash=self.context_pack_hash,
            witness_catalog_id=self.witness_catalog_id,
            witness_catalog_hash=self.witness_catalog_hash,
            semantic_review_closure=self.semantic_review_closure,
            answer_id=self.answer_id,
            answer_hash=self.answer_hash,
            decision=self.decision,
            overall_status=self.overall_status,
            grounding_status=self.grounding_status,
            relevance_status=self.relevance_status,
            completeness_status=self.completeness_status,
            strict_eligible=self.strict_eligible,
            claim_proofs=self.claim_proofs,
            obligation_proofs=self.obligation_proofs,
            verifier_versions=self.verifier_versions,
            failures=self.failures,
        )

    @property
    def is_committable(self) -> bool:
        """只有领域明确允许的 proof 终态才能触发外部交付。"""

        return self.overall_status in {
            OverallProofStatus.ACCEPTED,
            OverallProofStatus.ACCEPTED_PARTIAL,
            OverallProofStatus.ACCEPTED_REFUSAL,
            OverallProofStatus.ACCEPTED_ABSTENTION,
            OverallProofStatus.NOT_APPLICABLE,
        }

    @model_validator(mode="after")
    def validate_local_proof(self) -> Self:
        claim_ids = tuple(item.claim_id for item in self.claim_proofs)
        obligation_ids = tuple(item.obligation_id for item in self.obligation_proofs)
        require_unique(claim_ids, field_name="claim proof claim_ids")
        require_unique(obligation_ids, field_name="obligation proof obligation_ids")
        verifier_ids = tuple(item.verifier_id for item in self.verifier_versions)
        require_unique(verifier_ids, field_name="verifier_ids")
        failure_keys = tuple(
            f"{item.stage.value}:{item.code.value}:{item.subject_id or ''}"
            for item in self.failures
        )
        require_unique(failure_keys, field_name="answer proof failures")

        if (self.answer_id is None) != (self.answer_hash is None):
            raise ValueError("answer_id and answer_hash must appear together")
        if self.is_committable:
            if self.answer_id is None or self.failures:
                raise ValueError(
                    "committable proof requires answer identity and no top-level failures"
                )
        elif not self.failures:
            raise ValueError("non-committable proof requires structured failures")

        expected_decision = {
            OverallProofStatus.ACCEPTED: AnswerDecision.ANSWER,
            OverallProofStatus.ACCEPTED_PARTIAL: AnswerDecision.PARTIAL,
            OverallProofStatus.ACCEPTED_REFUSAL: AnswerDecision.REFUSE,
            OverallProofStatus.ACCEPTED_ABSTENTION: AnswerDecision.REFUSE,
        }.get(self.overall_status)
        if expected_decision is not None and self.decision != expected_decision:
            raise ValueError("overall_status disagrees with answer decision")
        if self.overall_status == OverallProofStatus.NOT_APPLICABLE and self.decision not in {
            AnswerDecision.CLARIFY,
            AnswerDecision.DIRECT_ANSWER,
        }:
            raise ValueError("not_applicable proof is reserved for non-retrieval decisions")
        if self.strict_eligible and self.overall_status not in {
            OverallProofStatus.ACCEPTED,
            OverallProofStatus.ACCEPTED_REFUSAL,
        }:
            raise ValueError("only complete answers or proven refusals may be strict eligible")

        expected_dimensions = {
            OverallProofStatus.ACCEPTED: (
                ProofDimensionStatus.PASSED,
                ProofDimensionStatus.PASSED,
                CompletenessProofStatus.COMPLETE,
            ),
            OverallProofStatus.ACCEPTED_PARTIAL: (
                ProofDimensionStatus.PASSED,
                ProofDimensionStatus.PASSED,
                CompletenessProofStatus.INCOMPLETE,
            ),
            OverallProofStatus.ACCEPTED_REFUSAL: (
                ProofDimensionStatus.NOT_APPLICABLE,
                ProofDimensionStatus.NOT_APPLICABLE,
                CompletenessProofStatus.COMPLETE,
            ),
            OverallProofStatus.ACCEPTED_ABSTENTION: (
                ProofDimensionStatus.NOT_APPLICABLE,
                ProofDimensionStatus.NOT_APPLICABLE,
                CompletenessProofStatus.INDETERMINATE,
            ),
            OverallProofStatus.NOT_APPLICABLE: (
                ProofDimensionStatus.NOT_APPLICABLE,
                ProofDimensionStatus.NOT_APPLICABLE,
                CompletenessProofStatus.NOT_APPLICABLE,
            ),
        }.get(self.overall_status)
        if (
            expected_dimensions is not None
            and (
                self.grounding_status,
                self.relevance_status,
                self.completeness_status,
            )
            != expected_dimensions
        ):
            raise ValueError("proof dimensions disagree with overall_status")

        payload = self._identity_payload()
        if self.proof_hash != stable_content_hash(payload):
            raise ValueError("proof_hash does not match AnswerProof")
        if self.proof_id != stable_identifier("apr_", payload):
            raise ValueError("proof_id does not match AnswerProof")
        return self

    @classmethod
    def build(
        cls,
        *,
        answer_plan: AnswerPlan,
        question_contract: QuestionContract,
        catalog: WitnessCatalogReceipt | None,
        witnesses: tuple[EvidenceWitness, ...] = (),
        witness_texts: Mapping[str, str] | None = None,
        validated_source_enumeration_closure_receipts: tuple[
            SourceEnumerationClosureReceipt, ...
        ] = (),
        semantic_review_closure: SemanticReviewReceipt | None = None,
        answer_id: str | None,
        answer_text: str | None,
        overall_status: OverallProofStatus,
        grounding_status: ProofDimensionStatus,
        relevance_status: ProofDimensionStatus,
        completeness_status: CompletenessProofStatus,
        claim_proofs: tuple[ClaimProof, ...],
        obligation_proofs: tuple[ObligationProof, ...],
        verifier_versions: tuple[VerifierRef, ...],
        failures: tuple[AnsweringFailure, ...] = (),
        created_at: datetime | None = None,
    ) -> "AnswerProof":
        """生成 AnswerProof，并立即校验完整引用图与最终答案文本。"""

        answer_hash = sha256_digest(answer_text) if answer_text is not None else None
        proof_by_obligation = {item.obligation_id: item for item in obligation_proofs}
        required_ids = question_contract.required_obligation_ids
        all_claims_proven = all(item.status == ProofVerdict.PROVEN for item in claim_proofs)
        accepted_answer_strict = (
            overall_status == OverallProofStatus.ACCEPTED
            and isinstance(semantic_review_closure, SemanticReviewClosureReceipt)
            and catalog is not None
            and catalog.universe_status == WitnessUniverseStatus.COMPLETE
            and grounding_status == ProofDimensionStatus.PASSED
            and relevance_status == ProofDimensionStatus.PASSED
            and completeness_status == CompletenessProofStatus.COMPLETE
            and all_claims_proven
            and set(required_ids).issubset(set(semantic_review_closure.closed_obligation_ids))
            and all(
                (proof := proof_by_obligation.get(obligation_id)) is not None
                and proof.status == ProofVerdict.PROVEN
                and proof.outcome == ObligationOutcome.ANSWERED
                for obligation_id in required_ids
            )
        )
        accepted_refusal_strict = (
            overall_status == OverallProofStatus.ACCEPTED_REFUSAL
            and isinstance(semantic_review_closure, SemanticReviewClosureReceipt)
            and catalog is not None
            and catalog.universe_status == WitnessUniverseStatus.COMPLETE
            and bool(required_ids)
            and not answer_plan.substantive_claims
            and all_claims_proven
            and all(
                (proof := proof_by_obligation.get(obligation_id)) is not None
                and proof.status == ProofVerdict.PROVEN
                and proof.outcome
                in {ObligationOutcome.EVIDENCE_ABSENT, ObligationOutcome.CONFLICTING}
                for obligation_id in required_ids
            )
        )
        strict_eligible = accepted_answer_strict or accepted_refusal_strict
        payload = _answer_proof_payload(
            answer_plan_id=answer_plan.plan_id,
            answer_plan_hash=answer_plan.plan_hash,
            question_contract_id=question_contract.contract_id,
            question_contract_hash=question_contract.contract_hash,
            context_pack_hash=catalog.context_pack_hash if catalog else None,
            witness_catalog_id=catalog.catalog_id if catalog else None,
            witness_catalog_hash=catalog.catalog_hash if catalog else None,
            semantic_review_closure=semantic_review_closure,
            answer_id=answer_id,
            answer_hash=answer_hash,
            decision=answer_plan.decision,
            overall_status=overall_status,
            grounding_status=grounding_status,
            relevance_status=relevance_status,
            completeness_status=completeness_status,
            strict_eligible=strict_eligible,
            claim_proofs=claim_proofs,
            obligation_proofs=obligation_proofs,
            verifier_versions=verifier_versions,
            failures=failures,
        )
        proof = cls(
            **payload,
            proof_id=stable_identifier("apr_", payload),
            proof_hash=stable_content_hash(payload),
            created_at=created_at or datetime.now(UTC),
        )
        proof.validate_against(
            answer_plan=answer_plan,
            question_contract=question_contract,
            catalog=catalog,
            witnesses=witnesses,
            witness_texts=witness_texts,
            validated_source_enumeration_closure_receipts=(
                validated_source_enumeration_closure_receipts
            ),
        )
        if answer_text is not None:
            proof.validate_answer_text(answer_text)
        return proof

    def validate_against(
        self,
        *,
        answer_plan: AnswerPlan,
        question_contract: QuestionContract,
        catalog: WitnessCatalogReceipt | None,
        witnesses: tuple[EvidenceWitness, ...] = (),
        witness_texts: Mapping[str, str] | None = None,
        validated_source_enumeration_closure_receipts: tuple[
            SourceEnumerationClosureReceipt, ...
        ] = (),
    ) -> Self:
        """执行 plan、question、catalog、claim 与 obligation 的严格图校验。"""

        if (
            self.answer_plan_id != answer_plan.plan_id
            or self.answer_plan_hash != answer_plan.plan_hash
            or self.question_contract_id != question_contract.contract_id
            or self.question_contract_hash != question_contract.contract_hash
            or self.decision != answer_plan.decision
        ):
            raise ValueError("AnswerProof references a different plan/question graph")
        answer_plan.validate_against(question_contract=question_contract, catalog=catalog)
        validated_source_receipts = {
            item.receipt_id: item for item in validated_source_enumeration_closure_receipts
        }
        if len(validated_source_receipts) != len(validated_source_enumeration_closure_receipts):
            raise ValueError("validated source closure receipt IDs must be unique")
        known_verifiers = {
            (item.verifier_id, item.verifier_version) for item in self.verifier_versions
        }
        used_verifiers = {
            (item.verifier.verifier_id, item.verifier.verifier_version)
            for item in (*self.claim_proofs, *self.obligation_proofs)
        }
        if not used_verifiers.issubset(known_verifiers):
            raise ValueError("proof graph uses an undeclared verifier version")

        if catalog is None:
            if any(
                value is not None
                for value in (
                    self.context_pack_hash,
                    self.witness_catalog_id,
                    self.witness_catalog_hash,
                )
            ):
                raise ValueError("proof without catalog cannot carry catalog references")
            known_witnesses: set[str] = set()
            witness_by_id: dict[str, EvidenceWitness] = {}
            resolved_witness_texts: Mapping[str, str] = {}
            if self.semantic_review_closure is not None:
                raise ValueError("proof without catalog cannot carry semantic review closure")
            if validated_source_receipts:
                raise ValueError("proof without catalog cannot validate source closure receipts")
        else:
            if (
                self.context_pack_hash != catalog.context_pack_hash
                or self.witness_catalog_id != catalog.catalog_id
                or self.witness_catalog_hash != catalog.catalog_hash
            ):
                raise ValueError("AnswerProof catalog references are inconsistent")
            catalog.validate_witnesses(witnesses)
            known_witnesses = set(catalog.witness_ids)
            witness_by_id = {item.witness_id: item for item in witnesses}
            resolved_witness_texts = witness_texts or {}
            if set(resolved_witness_texts) != known_witnesses:
                raise ValueError("AnswerProof validation requires every resolved Witness text")
            if isinstance(self.semantic_review_closure, SemanticReviewClosureReceipt):
                self.semantic_review_closure.validate_against(
                    answer_plan=answer_plan,
                    catalog=catalog,
                )
            elif isinstance(self.semantic_review_closure, SemanticReviewTerminalReceipt):
                if (
                    self.semantic_review_closure.answer_plan_id != answer_plan.plan_id
                    or self.semantic_review_closure.answer_plan_hash != answer_plan.plan_hash
                    or self.semantic_review_closure.catalog_id != catalog.catalog_id
                    or self.semantic_review_closure.catalog_hash != catalog.catalog_hash
                    or answer_plan.decision != AnswerDecision.REFUSE
                ):
                    raise ValueError("terminal review receipt references another final graph")
            if self.is_committable and self.semantic_review_closure is None:
                raise ValueError("committable retrieval proof requires semantic review closure")

        if tuple(item.claim_id for item in self.claim_proofs) != tuple(
            item.claim_id for item in answer_plan.claims
        ):
            raise ValueError("AnswerProof must contain one ordered proof per planned claim")
        proof_by_claim = {item.claim_id: item for item in self.claim_proofs}
        plan_by_claim = {item.claim_id: item for item in answer_plan.claims}
        for claim in answer_plan.claims:
            proof = proof_by_claim[claim.claim_id]
            if (
                proof.obligation_id != claim.obligation_id
                or proof.method != claim.rendering_rule
                or proof.witness_ids != claim.witness_ids
            ):
                raise ValueError("ClaimProof does not match its ClaimPlan")
            if not set(proof.witness_ids).issubset(known_witnesses):
                raise ValueError("ClaimProof references an unknown witness")

        expected_obligations = tuple(item.obligation_id for item in question_contract.obligations)
        if tuple(item.obligation_id for item in self.obligation_proofs) != expected_obligations:
            raise ValueError("AnswerProof must contain one ordered proof per question obligation")

        obligation_contracts = {item.obligation_id: item for item in question_contract.obligations}
        for obligation_proof in self.obligation_proofs:
            if catalog is None:
                raise ValueError("obligation proof requires WitnessCatalogReceipt")
            if (
                obligation_proof.catalog_id != catalog.catalog_id
                or obligation_proof.catalog_hash != catalog.catalog_hash
                or obligation_proof.catalog_universe_status != catalog.universe_status
            ):
                raise ValueError("ObligationProof catalog references are inconsistent")
            if not set(obligation_proof.contradiction_witness_ids).issubset(known_witnesses):
                raise ValueError("ObligationProof references an unknown contradiction witness")
            obligation = obligation_contracts[obligation_proof.obligation_id]
            if (
                obligation_proof.closure != obligation.closure
                or obligation_proof.required_output_form != obligation.required_output_form
                or obligation_proof.expected_item_count != obligation.expected_count
            ):
                raise ValueError("ObligationProof closure contract does not match the question")
            table_cell_projection = (
                resolve_closed_count_markdown_table_cell_projection(
                    question=question_contract,
                    obligation=obligation,
                    witnesses=witnesses,
                    witness_texts=resolved_witness_texts,
                )
                if obligation.closure == FacetClosure.CLOSED_COUNT
                else None
            )
            source_closure = obligation_proof.source_enumeration_closure_receipt
            if source_closure is not None:
                validated_source = validated_source_receipts.get(source_closure.receipt_id)
                member_ids = source_closure_witness_ids(source_closure)
                member_witnesses = tuple(witness_by_id.get(item) for item in member_ids)
                common_invalid = (
                    validated_source != source_closure
                    or source_closure.question_contract_id != question_contract.contract_id
                    or source_closure.question_contract_hash != question_contract.contract_hash
                    or source_closure.obligation_id != obligation.obligation_id
                    or source_closure.obligation_hash != obligation.obligation_hash
                    or source_closure.catalog_id != catalog.catalog_id
                    or source_closure.catalog_hash != catalog.catalog_hash
                    or any(item is None for item in member_witnesses)
                )
                if common_invalid:
                    raise ValueError(
                        "source enumeration closure receipt is not bound to this proof graph"
                    )
                typed_members = tuple(item for item in member_witnesses if item is not None)
                if isinstance(source_closure, QueryAnchoredTableRowClosureReceipt):
                    witness = typed_members[0]
                    binding_invalid = (
                        source_closure.witness_hash != witness.witness_hash
                        or source_closure.evidence_id != witness.span_ref.evidence_id
                        or source_closure.evidence_content_hash
                        != witness.span_ref.evidence_content_hash
                        or source_closure.representation_id != witness.span_ref.representation_id
                        or source_closure.span_ref_hash != witness.span_ref.span_ref_hash
                    )
                else:
                    binding_invalid = (
                        source_closure.witness_hashes
                        != tuple(item.witness_hash for item in typed_members)
                        or source_closure.witness_ordinals
                        != tuple(item.ordinal for item in typed_members)
                        or source_closure.span_ref_hashes
                        != tuple(item.span_ref.span_ref_hash for item in typed_members)
                        or any(
                            source_closure.evidence_id != item.span_ref.evidence_id
                            or source_closure.evidence_content_hash
                            != item.span_ref.evidence_content_hash
                            or source_closure.representation_id != item.span_ref.representation_id
                            for item in typed_members
                        )
                    )
                if binding_invalid:
                    raise ValueError(
                        "source enumeration closure receipt witness bindings are invalid"
                    )
            for claim_id in obligation_proof.supporting_claim_ids:
                claim = plan_by_claim.get(claim_id)
                claim_proof = proof_by_claim.get(claim_id)
                if (
                    claim is None
                    or claim_proof is None
                    or claim.obligation_id != obligation_proof.obligation_id
                    or not claim.is_substantive
                    or claim_proof.status != ProofVerdict.PROVEN
                ):
                    raise ValueError("ObligationProof claim support is invalid")
            item_spans_by_witness: dict[str, list[tuple[int, int]]] = {}
            projected_cell_spans_by_witness: dict[str, tuple[tuple[int, int], ...]] = {}
            projected_cell_actual_by_witness: dict[str, list[tuple[int, int]]] = {}
            inline_action_spans_by_witness: dict[str, tuple[tuple[int, int], ...]] = {}
            inline_action_actual_by_witness: dict[str, list[tuple[int, int]]] = {}
            for item_ref in obligation_proof.proven_items:
                claim = plan_by_claim.get(item_ref.claim_id)
                if (
                    item_ref.obligation_id != obligation_proof.obligation_id
                    or item_ref.claim_id not in obligation_proof.supporting_claim_ids
                    or claim is None
                    or item_ref.witness_id not in claim.witness_ids
                ):
                    raise ValueError("proven item is outside its obligation supporting claims")
                witness = witness_by_id[item_ref.witness_id]
                witness_text = resolved_witness_texts[item_ref.witness_id]
                item_ref.validate_witness_text(witness, witness_text)
                if item_ref.segmentation_method == ItemSegmentationMethod.SINGLE_WITNESS_ATOM:
                    if obligation.closure != FacetClosure.SINGLE or (
                        item_ref.relative_start,
                        item_ref.relative_end,
                    ) != (0, len(witness_text)):
                        raise ValueError("single-witness item must cover a single obligation atom")
                elif item_ref.segmentation_method == (
                    ItemSegmentationMethod.OPEN_ENUMERATION_WITNESS_ATOM
                ):
                    if obligation.closure != FacetClosure.OPEN_ENUMERATION or (
                        item_ref.relative_start,
                        item_ref.relative_end,
                    ) != (0, len(witness_text)):
                        raise ValueError(
                            "open-enumeration witness item must cover its complete atom"
                        )
                elif item_ref.segmentation_method == (ItemSegmentationMethod.TOP_LEVEL_LIST_ENTRY):
                    if witness.atom_kind != WitnessAtomKind.LIST_ITEM or (
                        item_ref.relative_start,
                        item_ref.relative_end,
                    ) != top_level_list_item_body_span(witness_text):
                        raise ValueError(
                            "enumeration item proof does not match the list body boundary"
                        )
                elif item_ref.segmentation_method == (
                    ItemSegmentationMethod.MARKDOWN_TABLE_PROJECTED_CELL_ITEM
                ):
                    expected_spans = projected_cell_spans_by_witness.setdefault(
                        item_ref.witness_id,
                        closed_count_markdown_table_cell_item_spans(
                            question=question_contract,
                            obligation=obligation,
                            selected_witness=witness,
                            witnesses=witnesses,
                            witness_texts=resolved_witness_texts,
                        ),
                    )
                    item_span = (item_ref.relative_start, item_ref.relative_end)
                    if (
                        witness.atom_kind != WitnessAtomKind.TABLE_DATA_ROW
                        or obligation.closure != FacetClosure.CLOSED_COUNT
                        or not expected_spans
                        or item_span not in expected_spans
                    ):
                        raise ValueError(
                            "closed-count item proof does not match a query-projected table cell"
                        )
                    projected_cell_actual_by_witness.setdefault(item_ref.witness_id, []).append(
                        item_span
                    )
                elif item_ref.segmentation_method == ItemSegmentationMethod.INLINE_ACTION_SPAN:
                    if not isinstance(
                        source_closure,
                        QueryAnchoredInlineActionChainClosureReceipt,
                    ):
                        raise ValueError(
                            "inline action item requires its bound action-chain receipt"
                        )
                    expected_spans = inline_action_spans_by_witness.setdefault(
                        item_ref.witness_id,
                        _inline_action_relative_spans(source_closure),
                    )
                    item_span = (item_ref.relative_start, item_ref.relative_end)
                    if (
                        obligation.closure != FacetClosure.OPEN_ENUMERATION
                        or obligation.required_output_form != RequiredOutputForm.ORDERED_STEPS
                        or item_ref.witness_id != source_closure.witness_ids[0]
                        or witness.atom_kind
                        not in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
                        or item_span not in expected_spans
                    ):
                        raise ValueError(
                            "inline action item does not match its action-chain boundary"
                        )
                    inline_action_actual_by_witness.setdefault(item_ref.witness_id, []).append(
                        item_span
                    )
                elif witness.atom_kind != WitnessAtomKind.TABLE_DATA_ROW or (
                    item_ref.relative_start,
                    item_ref.relative_end,
                ) != markdown_table_data_row_body_span(witness_text):
                    raise ValueError("enumeration item proof does not match a Markdown data row")
                item_spans_by_witness.setdefault(item_ref.witness_id, []).append(
                    (item_ref.relative_start, item_ref.relative_end)
                )
            for spans in item_spans_by_witness.values():
                ordered_spans = sorted(spans)
                if any(
                    current[0] < previous[1]
                    for previous, current in zip(ordered_spans, ordered_spans[1:], strict=False)
                ):
                    raise ValueError("proven item spans must not overlap within a Witness")
            if (
                table_cell_projection is not None
                and table_cell_projection.status
                != ClosedCountTableCellProjectionStatus.NOT_APPLICABLE
                and obligation_proof.proven_items
            ):
                actual_projection = tuple(
                    (
                        item.witness_id,
                        item.relative_start,
                        item.relative_end,
                        item.segmentation_method,
                    )
                    for item in obligation_proof.proven_items
                )
                expected_projection = tuple(
                    (
                        table_cell_projection.witness_id,
                        start,
                        end,
                        ItemSegmentationMethod.MARKDOWN_TABLE_PROJECTED_CELL_ITEM,
                    )
                    for start, end in table_cell_projection.item_spans
                )
                if (
                    table_cell_projection.status != ClosedCountTableCellProjectionStatus.VALID
                    or actual_projection != expected_projection
                ):
                    raise ValueError(
                        "closed-count projected table-cell scope cannot use fallback items"
                    )
            for witness_id, expected_spans in projected_cell_spans_by_witness.items():
                if tuple(projected_cell_actual_by_witness.get(witness_id, ())) != expected_spans:
                    raise ValueError(
                        "closed-count table-cell proof must contain every projected item"
                    )
            for witness_id, expected_spans in inline_action_spans_by_witness.items():
                if tuple(inline_action_actual_by_witness.get(witness_id, ())) != expected_spans:
                    raise ValueError(
                        "inline action proof must contain every action exactly once and in order"
                    )
            if (
                obligation_proof.status == ProofVerdict.PROVEN
                and obligation_proof.outcome == ObligationOutcome.EVIDENCE_ABSENT
                and catalog.witness_count != 0
            ):
                raise ValueError(
                    "proven evidence absence requires an empty complete witness catalog"
                )
            if obligation_proof.outcome == ObligationOutcome.ANSWERED and (
                not isinstance(
                    self.semantic_review_closure,
                    SemanticReviewClosureReceipt,
                )
                or obligation_proof.obligation_id
                not in set(self.semantic_review_closure.closed_obligation_ids)
            ):
                raise ValueError("answered obligation lacks its bound semantic closure review")

        committable = self.overall_status in {
            OverallProofStatus.ACCEPTED,
            OverallProofStatus.ACCEPTED_PARTIAL,
            OverallProofStatus.ACCEPTED_REFUSAL,
            OverallProofStatus.ACCEPTED_ABSTENTION,
            OverallProofStatus.NOT_APPLICABLE,
        }
        if committable and any(item.status != ProofVerdict.PROVEN for item in self.claim_proofs):
            raise ValueError("committable answer cannot expose a non-proven planned claim")

        obligation_by_id = {item.obligation_id: item for item in self.obligation_proofs}
        required = set(question_contract.required_obligation_ids)
        required_proofs = tuple(obligation_by_id[item] for item in required)
        substantive_claims = answer_plan.substantive_claims

        if self.overall_status == OverallProofStatus.ACCEPTED:
            if any(
                item.status != ProofVerdict.PROVEN or item.outcome != ObligationOutcome.ANSWERED
                for item in required_proofs
            ):
                raise ValueError("accepted answer requires every required obligation answered")
        elif self.overall_status == OverallProofStatus.ACCEPTED_PARTIAL:
            supported = tuple(
                item
                for item in required_proofs
                if item.status == ProofVerdict.PROVEN
                and item.outcome
                in {
                    ObligationOutcome.ANSWERED,
                    ObligationOutcome.PARTIALLY_ANSWERED,
                }
            )
            unresolved = tuple(
                item for item in required_proofs if item.outcome != ObligationOutcome.ANSWERED
            )
            allowed_unresolved = {
                ObligationOutcome.PARTIALLY_ANSWERED,
                ObligationOutcome.EVIDENCE_ABSENT,
                ObligationOutcome.CONFLICTING,
                ObligationOutcome.INDETERMINATE,
            }
            if (
                not supported
                or not unresolved
                or any(item.outcome not in allowed_unresolved for item in unresolved)
                or any(
                    (
                        item.outcome == ObligationOutcome.PARTIALLY_ANSWERED
                        and item.status != ProofVerdict.PROVEN
                    )
                    or (
                        item.outcome
                        in {
                            ObligationOutcome.EVIDENCE_ABSENT,
                            ObligationOutcome.CONFLICTING,
                        }
                        and item.status != ProofVerdict.PROVEN
                    )
                    or (
                        item.outcome == ObligationOutcome.INDETERMINATE
                        and item.status != ProofVerdict.INDETERMINATE
                    )
                    for item in unresolved
                )
            ):
                raise ValueError("accepted partial answer has an invalid obligation partition")
            planned_unresolved = set(answer_plan.unresolved_obligation_ids)
            actual_unresolved = {
                item.obligation_id
                for item in required_proofs
                if item.outcome != ObligationOutcome.ANSWERED
            }
            if actual_unresolved != planned_unresolved:
                raise ValueError("partial proof disagrees with planned unresolved obligations")
        elif self.overall_status == OverallProofStatus.ACCEPTED_REFUSAL:
            if catalog is None or catalog.universe_status != WitnessUniverseStatus.COMPLETE:
                raise ValueError("accepted refusal requires a complete witness catalog")
            if substantive_claims or not required:
                raise ValueError("accepted refusal cannot contain factual claims or no obligations")
            if any(
                item.status != ProofVerdict.PROVEN
                or item.outcome
                not in {ObligationOutcome.EVIDENCE_ABSENT, ObligationOutcome.CONFLICTING}
                for item in required_proofs
            ):
                raise ValueError("refusal requires proven absence/conflict for every obligation")
        elif self.overall_status == OverallProofStatus.ACCEPTED_ABSTENTION:
            if substantive_claims or not required:
                raise ValueError("accepted abstention cannot contain factual claims")
            if any(
                item.status != ProofVerdict.PROVEN
                or item.outcome != ObligationOutcome.INSUFFICIENT_RETRIEVED_EVIDENCE
                for item in required_proofs
            ):
                raise ValueError(
                    "accepted abstention requires proven local insufficiency per obligation"
                )
        elif self.overall_status == OverallProofStatus.NOT_APPLICABLE:
            if question_contract.obligations or catalog is not None or self.obligation_proofs:
                raise ValueError("non-retrieval proof cannot carry evidence obligations")
        accepted_answer_strict = (
            self.overall_status == OverallProofStatus.ACCEPTED
            and isinstance(self.semantic_review_closure, SemanticReviewClosureReceipt)
            and catalog is not None
            and catalog.universe_status == WitnessUniverseStatus.COMPLETE
            and required.issubset(set(self.semantic_review_closure.closed_obligation_ids))
            and all(item.status == ProofVerdict.PROVEN for item in self.claim_proofs)
            and all(
                item.status == ProofVerdict.PROVEN and item.outcome == ObligationOutcome.ANSWERED
                for item in required_proofs
            )
        )
        accepted_refusal_strict = (
            self.overall_status == OverallProofStatus.ACCEPTED_REFUSAL
            and isinstance(self.semantic_review_closure, SemanticReviewClosureReceipt)
            and catalog is not None
            and catalog.universe_status == WitnessUniverseStatus.COMPLETE
            and bool(required)
            and not substantive_claims
            and all(item.status == ProofVerdict.PROVEN for item in self.claim_proofs)
            and all(
                item.status == ProofVerdict.PROVEN
                and item.outcome
                in {ObligationOutcome.EVIDENCE_ABSENT, ObligationOutcome.CONFLICTING}
                for item in required_proofs
            )
        )
        if self.strict_eligible != (accepted_answer_strict or accepted_refusal_strict):
            raise ValueError("strict_eligible disagrees with the validated proof graph")
        return self

    def validate_answer_text(self, answer_text: str) -> Self:
        """验证提交文本、每个可见 claim span 与证明哈希精确一致。"""

        if self.answer_hash != sha256_digest(answer_text):
            raise ValueError("answer_hash does not match final answer text")
        previous_end = 0
        uncovered_segments: list[str] = []
        for proof in self.claim_proofs:
            if proof.answer_span is None or proof.rendered_text_hash is None:
                continue
            if proof.answer_span.end > len(answer_text):
                raise ValueError("claim proof span exceeds final answer text")
            if proof.answer_span.start < previous_end:
                raise ValueError("claim proof spans must be ordered and non-overlapping")
            uncovered_segments.append(answer_text[previous_end : proof.answer_span.start])
            rendered = answer_text[proof.answer_span.start : proof.answer_span.end]
            if sha256_digest(rendered) != proof.rendered_text_hash:
                raise ValueError("rendered claim hash does not match final answer span")
            previous_end = proof.answer_span.end
        uncovered_segments.append(answer_text[previous_end:])
        if self.overall_status in {
            OverallProofStatus.ACCEPTED,
            OverallProofStatus.ACCEPTED_PARTIAL,
            OverallProofStatus.ACCEPTED_REFUSAL,
            OverallProofStatus.ACCEPTED_ABSTENTION,
            OverallProofStatus.NOT_APPLICABLE,
        } and any(segment.strip() for segment in uncovered_segments):
            raise ValueError("committable answer contains text outside proven claim spans")
        return self


__all__ = [
    "ANSWER_PROOF_SCHEMA_VERSION",
    "CLAIM_PROOF_SCHEMA_VERSION",
    "OBLIGATION_PROOF_SCHEMA_VERSION",
    "PROVEN_ITEM_REF_SCHEMA_VERSION",
    "SEMANTIC_REVIEW_CLOSURE_SCHEMA_VERSION",
    "AnswerProof",
    "ClaimProof",
    "CompletenessProofStatus",
    "ItemSegmentationMethod",
    "ObligationClosureStatus",
    "ObligationOutcome",
    "ObligationProof",
    "OverallProofStatus",
    "ProofDimensionStatus",
    "ProofRelation",
    "ProofVerdict",
    "ProvenItemRef",
    "SemanticReviewClosureReceipt",
    "VerifierRef",
]
