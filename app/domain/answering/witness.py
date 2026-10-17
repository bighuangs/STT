"""模型实际可见证据原子的不可变身份与完整目录回执。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self
import unicodedata

from pydantic import Field, model_validator

from app.domain.answering._base import (
    EVIDENCE_ID_PATTERN,
    REPRESENTATION_ID_PATTERN,
    SHA256_PATTERN,
    SCOPE_HASH_PATTERN,
    WITNESS_ID_PATTERN,
    AnsweringContractModel,
    require_unique,
    sha256_digest,
    stable_content_hash,
    stable_identifier,
)
from app.domain.safe_representation import (
    SAFE_REPRESENTATION_TRANSFORM_VERSION,
    derive_safe_representation,
)


EVIDENCE_SPAN_REF_SCHEMA_VERSION = "evidence-span-ref.v1"
EVIDENCE_WITNESS_SCHEMA_VERSION = "evidence-witness.v1"
WITNESS_CATALOG_SCHEMA_VERSION = "witness-catalog.v1"
PACKED_TRUNCATION_MARKER = "\n…[证据正文因上下文预算被截断]"


class SpanEquivalence(StrEnum):
    """canonical 与 packed 片段之间允许的最窄等价关系。"""

    EXACT_UTF8 = "exact_utf8"
    NFKC_EQUIVALENT = "nfkc_equivalent"
    SECURITY_SAFE_DERIVED = "security_safe_derived"


class WitnessAtomKind(StrEnum):
    """证据原子的结构类别，不承载问题义务。"""

    SENTENCE = "sentence"
    TABLE_ROW = "table_row"
    TABLE_DATA_ROW = "table_data_row"
    LIST_ITEM = "list_item"
    MEDIA_DERIVATION = "media_derivation"


class WitnessUniverseStatus(StrEnum):
    """WitnessCatalog 对模型可见上下文的闭合程度。"""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    INDETERMINATE = "indeterminate"


class WitnessOmissionCode(StrEnum):
    """目录不完整时的可判定遗漏原因。"""

    PACKED_REPRESENTATION_TRUNCATED = "packed_representation_truncated"
    CONTEXT_BUDGET = "context_budget"
    ATOM_LIMIT = "atom_limit"
    SAFE_MAPPING_BUDGET_EXCEEDED = "safe_mapping_budget_exceeded"
    SOURCE_MAPPING_UNPROVEN = "source_mapping_unproven"
    UNSUPPORTED_MODALITY = "unsupported_modality"
    BUILDER_INTERNAL_ERROR = "builder_internal_error"


class WitnessCatalogOmission(AnsweringContractModel):
    """不保存正文的聚合遗漏计数。"""

    code: WitnessOmissionCode
    count: int = Field(ge=1)


def _normalized_span_hash(value: str) -> str:
    return sha256_digest(unicodedata.normalize("NFKC", value))


def _span_ref_payload(
    *,
    evidence_id: str,
    evidence_content_hash: str,
    representation_id: str,
    representation_hash: str,
    canonical_start: int,
    canonical_end: int,
    canonical_span_hash: str,
    packed_start: int,
    packed_end: int,
    packed_span_hash: str,
    canonical_normalized_span_hash: str,
    packed_normalized_span_hash: str,
    equivalence: SpanEquivalence,
    safe_derivation_transform_version: str | None,
    safe_derivation_receipt_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SPAN_REF_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "evidence_content_hash": evidence_content_hash,
        "representation_id": representation_id,
        "representation_hash": representation_hash,
        "canonical_start": canonical_start,
        "canonical_end": canonical_end,
        "canonical_span_hash": canonical_span_hash,
        "packed_start": packed_start,
        "packed_end": packed_end,
        "packed_span_hash": packed_span_hash,
        "canonical_normalized_span_hash": canonical_normalized_span_hash,
        "packed_normalized_span_hash": packed_normalized_span_hash,
        "coordinate_semantics": "unicode-codepoint-half-open.v1",
        "equivalence": equivalence.value,
        "safe_derivation_transform_version": safe_derivation_transform_version,
        "safe_derivation_receipt_hash": safe_derivation_receipt_hash,
    }


class EvidenceSpanRef(AnsweringContractModel):
    """同一证据原子在 canonical Evidence 与 packed 表示中的精确坐标。"""

    schema_version: Literal["evidence-span-ref.v1"] = EVIDENCE_SPAN_REF_SCHEMA_VERSION
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    representation_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(gt=0)
    canonical_span_hash: str = Field(pattern=SHA256_PATTERN)
    packed_start: int = Field(ge=0)
    packed_end: int = Field(gt=0)
    packed_span_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_normalized_span_hash: str = Field(pattern=SHA256_PATTERN)
    packed_normalized_span_hash: str = Field(pattern=SHA256_PATTERN)
    coordinate_semantics: Literal["unicode-codepoint-half-open.v1"] = (
        "unicode-codepoint-half-open.v1"
    )
    equivalence: SpanEquivalence = SpanEquivalence.EXACT_UTF8
    safe_derivation_transform_version: Literal["rag-safe-representation.v2"] | None = None
    safe_derivation_receipt_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    span_ref_hash: str = Field(pattern=SHA256_PATTERN)

    def _identity_payload(self) -> dict[str, Any]:
        return _span_ref_payload(
            evidence_id=self.evidence_id,
            evidence_content_hash=self.evidence_content_hash,
            representation_id=self.representation_id,
            representation_hash=self.representation_hash,
            canonical_start=self.canonical_start,
            canonical_end=self.canonical_end,
            canonical_span_hash=self.canonical_span_hash,
            packed_start=self.packed_start,
            packed_end=self.packed_end,
            packed_span_hash=self.packed_span_hash,
            canonical_normalized_span_hash=self.canonical_normalized_span_hash,
            packed_normalized_span_hash=self.packed_normalized_span_hash,
            equivalence=self.equivalence,
            safe_derivation_transform_version=self.safe_derivation_transform_version,
            safe_derivation_receipt_hash=self.safe_derivation_receipt_hash,
        )

    @model_validator(mode="after")
    def validate_span_contract(self) -> Self:
        if self.canonical_end <= self.canonical_start:
            raise ValueError("canonical span must satisfy start < end")
        if self.packed_end <= self.packed_start:
            raise ValueError("packed span must satisfy start < end")
        is_safe_derivation = self.equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED
        has_safe_derivation_identity = (
            self.safe_derivation_transform_version is not None
            and self.safe_derivation_receipt_hash is not None
        )
        if is_safe_derivation != has_safe_derivation_identity:
            raise ValueError(
                "security-safe span mapping requires exactly one versioned derivation identity"
            )
        if not is_safe_derivation and (
            self.safe_derivation_transform_version is not None
            or self.safe_derivation_receipt_hash is not None
        ):
            raise ValueError("non-security-safe span cannot carry derivation identity")
        if (
            not is_safe_derivation
            and self.canonical_normalized_span_hash != self.packed_normalized_span_hash
        ):
            raise ValueError("canonical and packed spans are not NFKC-equivalent")
        if (
            self.equivalence == SpanEquivalence.EXACT_UTF8
            and self.canonical_span_hash != self.packed_span_hash
        ):
            raise ValueError("exact span mapping requires identical span hashes")
        if self.span_ref_hash != stable_content_hash(self._identity_payload()):
            raise ValueError("span_ref_hash does not match EvidenceSpanRef")
        return self

    @classmethod
    def from_texts(
        cls,
        *,
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
        canonical_start: int,
        canonical_end: int,
        packed_start: int,
        packed_end: int,
        equivalence: SpanEquivalence = SpanEquivalence.EXACT_UTF8,
        safe_derivation_transform_version: str | None = None,
        safe_derivation_receipt_hash: str | None = None,
    ) -> "EvidenceSpanRef":
        """从两个冻结文本快照建立并即时验证坐标映射。"""

        if not 0 <= canonical_start < canonical_end <= len(canonical_text):
            raise ValueError("canonical span is outside canonical_text")
        if not 0 <= packed_start < packed_end <= len(packed_text):
            raise ValueError("packed span is outside packed_text")
        canonical_span = canonical_text[canonical_start:canonical_end]
        packed_span = packed_text[packed_start:packed_end]
        if equivalence == SpanEquivalence.EXACT_UTF8:
            if canonical_span != packed_span:
                raise ValueError("exact span mapping requires identical text")
        elif equivalence == SpanEquivalence.NFKC_EQUIVALENT:
            if unicodedata.normalize("NFKC", canonical_span) != unicodedata.normalize(
                "NFKC", packed_span
            ):
                raise ValueError("span texts are not NFKC-equivalent")
        else:
            if safe_derivation_transform_version != SAFE_REPRESENTATION_TRANSFORM_VERSION:
                raise ValueError("security-safe span uses an unsupported transform version")
            if safe_derivation_receipt_hash is None:
                raise ValueError("security-safe span is missing its derivation receipt hash")
            derivation = derive_safe_representation(canonical_span)
            if derivation.receipt.has_unproven_transform:
                raise ValueError("security-safe span contains an unproven transform")
            if derivation.safe_representation != packed_span:
                raise ValueError("span texts do not match the frozen security derivation")

        if equivalence != SpanEquivalence.SECURITY_SAFE_DERIVED and (
            safe_derivation_transform_version is not None
            or safe_derivation_receipt_hash is not None
        ):
            raise ValueError("non-security-safe span cannot carry derivation identity")

        payload = _span_ref_payload(
            evidence_id=evidence_id,
            evidence_content_hash=sha256_digest(canonical_text),
            representation_id=representation_id,
            representation_hash=sha256_digest(packed_text),
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            canonical_span_hash=sha256_digest(canonical_span),
            packed_start=packed_start,
            packed_end=packed_end,
            packed_span_hash=sha256_digest(packed_span),
            canonical_normalized_span_hash=_normalized_span_hash(canonical_span),
            packed_normalized_span_hash=_normalized_span_hash(packed_span),
            equivalence=equivalence,
            safe_derivation_transform_version=safe_derivation_transform_version,
            safe_derivation_receipt_hash=safe_derivation_receipt_hash,
        )
        return cls(**payload, span_ref_hash=stable_content_hash(payload))

    def validate_texts(self, *, canonical_text: str, packed_text: str) -> Self:
        """在 artifact 边界重新解析坐标与所有文本哈希。"""

        rebuilt = self.from_texts(
            evidence_id=self.evidence_id,
            representation_id=self.representation_id,
            canonical_text=canonical_text,
            packed_text=packed_text,
            canonical_start=self.canonical_start,
            canonical_end=self.canonical_end,
            packed_start=self.packed_start,
            packed_end=self.packed_end,
            equivalence=self.equivalence,
            safe_derivation_transform_version=self.safe_derivation_transform_version,
            safe_derivation_receipt_hash=self.safe_derivation_receipt_hash,
        )
        if rebuilt != self:
            raise ValueError("EvidenceSpanRef does not resolve against supplied artifacts")
        return self

    def validate_artifacts(
        self,
        *,
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
    ) -> Self:
        """同时核验 artifact 身份和正文，避免只验证一组自洽但错误的坐标。"""

        if evidence_id != self.evidence_id or representation_id != self.representation_id:
            raise ValueError("EvidenceSpanRef artifact identity mismatch")
        return self.validate_texts(canonical_text=canonical_text, packed_text=packed_text)

    def resolve_canonical_span(self, canonical_text: str) -> str:
        """从冻结 canonical artifact 解析 source quote 并重验身份。"""

        if sha256_digest(canonical_text) != self.evidence_content_hash:
            raise ValueError("EvidenceSpanRef canonical artifact hash mismatch")
        if not 0 <= self.canonical_start < self.canonical_end <= len(canonical_text):
            raise ValueError("EvidenceSpanRef canonical coordinates are outside artifact")
        resolved = canonical_text[self.canonical_start : self.canonical_end]
        if sha256_digest(resolved) != self.canonical_span_hash:
            raise ValueError("EvidenceSpanRef canonical span hash mismatch")
        return resolved


def _witness_payload(
    *,
    context_pack_hash: str,
    authorized_scope_hash: str,
    span_ref: EvidenceSpanRef,
    atom_kind: WitnessAtomKind,
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_WITNESS_SCHEMA_VERSION,
        "context_pack_hash": context_pack_hash,
        "authorized_scope_hash": authorized_scope_hash,
        "span_ref": span_ref.to_dict(),
        "atom_kind": atom_kind.value,
    }


class EvidenceWitness(AnsweringContractModel):
    """问题无关的证据原子；不保存义务、标签、quote 或事实陈述。"""

    schema_version: Literal["evidence-witness.v1"] = EVIDENCE_WITNESS_SCHEMA_VERSION
    witness_id: str = Field(pattern=WITNESS_ID_PATTERN)
    witness_hash: str = Field(pattern=SHA256_PATTERN)
    context_pack_hash: str = Field(pattern=SHA256_PATTERN)
    authorized_scope_hash: str = Field(pattern=SCOPE_HASH_PATTERN)
    span_ref: EvidenceSpanRef
    atom_kind: WitnessAtomKind
    ordinal: int = Field(ge=0)

    def _identity_payload(self) -> dict[str, Any]:
        return _witness_payload(
            context_pack_hash=self.context_pack_hash,
            authorized_scope_hash=self.authorized_scope_hash,
            span_ref=self.span_ref,
            atom_kind=self.atom_kind,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        payload = self._identity_payload()
        if self.witness_hash != stable_content_hash(payload):
            raise ValueError("witness_hash does not match EvidenceWitness")
        if self.witness_id != stable_identifier("wit_", payload):
            raise ValueError("witness_id does not match EvidenceWitness")
        return self

    @classmethod
    def build(
        cls,
        *,
        context_pack_hash: str,
        authorized_scope_hash: str,
        span_ref: EvidenceSpanRef,
        atom_kind: WitnessAtomKind,
        ordinal: int,
    ) -> "EvidenceWitness":
        """由已验证 span ref 生成与问题无关的 Witness。"""

        payload = _witness_payload(
            context_pack_hash=context_pack_hash,
            authorized_scope_hash=authorized_scope_hash,
            span_ref=span_ref,
            atom_kind=atom_kind,
        )
        return cls(
            **payload,
            witness_id=stable_identifier("wit_", payload),
            witness_hash=stable_content_hash(payload),
            ordinal=ordinal,
        )


def _catalog_payload(
    *,
    question_contract_id: str,
    question_contract_hash: str,
    context_pack_hash: str,
    authorized_scope_hash: str,
    witness_ids: tuple[str, ...],
    witness_count: int,
    universe_status: WitnessUniverseStatus,
    omissions: tuple[WitnessCatalogOmission, ...],
    builder_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": WITNESS_CATALOG_SCHEMA_VERSION,
        "question_contract_id": question_contract_id,
        "question_contract_hash": question_contract_hash,
        "context_pack_hash": context_pack_hash,
        "authorized_scope_hash": authorized_scope_hash,
        "witness_ids": list(witness_ids),
        "witness_count": witness_count,
        "universe_status": universe_status.value,
        "omissions": [item.to_dict() for item in omissions],
        "builder_version": builder_version,
    }


class WitnessCatalogReceipt(AnsweringContractModel):
    """证明 Witness 全集是否闭合的回执；拒答不能绕过此边界。"""

    schema_version: Literal["witness-catalog.v1"] = WITNESS_CATALOG_SCHEMA_VERSION
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    context_pack_hash: str = Field(pattern=SHA256_PATTERN)
    authorized_scope_hash: str = Field(pattern=SCOPE_HASH_PATTERN)
    witness_ids: tuple[str, ...] = Field(default=(), max_length=4096)
    witness_count: int = Field(ge=0, le=4096)
    universe_status: WitnessUniverseStatus
    omissions: tuple[WitnessCatalogOmission, ...] = Field(default=(), max_length=16)
    builder_version: str = Field(min_length=1, max_length=128)

    def _identity_payload(self) -> dict[str, Any]:
        return _catalog_payload(
            question_contract_id=self.question_contract_id,
            question_contract_hash=self.question_contract_hash,
            context_pack_hash=self.context_pack_hash,
            authorized_scope_hash=self.authorized_scope_hash,
            witness_ids=self.witness_ids,
            witness_count=self.witness_count,
            universe_status=self.universe_status,
            omissions=self.omissions,
            builder_version=self.builder_version,
        )

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        require_unique(self.witness_ids, field_name="witness_ids")
        if self.witness_count != len(self.witness_ids):
            raise ValueError("witness_count must equal len(witness_ids)")
        omission_codes = tuple(item.code for item in self.omissions)
        if len(omission_codes) != len(set(omission_codes)):
            raise ValueError("catalog omission codes must be unique")
        if self.universe_status == WitnessUniverseStatus.COMPLETE:
            if self.omissions:
                raise ValueError("complete witness catalog cannot contain omissions")
        elif not self.omissions:
            raise ValueError("non-complete witness catalog requires omission facts")
        if self.universe_status == WitnessUniverseStatus.TRUNCATED and not any(
            item.code
            in {
                WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED,
                WitnessOmissionCode.CONTEXT_BUDGET,
                WitnessOmissionCode.ATOM_LIMIT,
                WitnessOmissionCode.SAFE_MAPPING_BUDGET_EXCEEDED,
            }
            for item in self.omissions
        ):
            raise ValueError("truncated catalog requires a truncation omission code")
        payload = self._identity_payload()
        if self.catalog_hash != stable_content_hash(payload):
            raise ValueError("catalog_hash does not match WitnessCatalogReceipt")
        if self.catalog_id != stable_identifier("wcat_", payload):
            raise ValueError("catalog_id does not match WitnessCatalogReceipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question_contract_id: str,
        question_contract_hash: str,
        context_pack_hash: str,
        authorized_scope_hash: str,
        witnesses: tuple[EvidenceWitness, ...],
        universe_status: WitnessUniverseStatus,
        omissions: tuple[WitnessCatalogOmission, ...] = (),
        builder_version: str,
    ) -> "WitnessCatalogReceipt":
        """从有序 Witness 集合建立目录回执并校验共同作用域。"""

        expected_ordinals = tuple(range(len(witnesses)))
        if tuple(item.ordinal for item in witnesses) != expected_ordinals:
            raise ValueError("witness ordinals must be contiguous and zero-based")
        if any(item.context_pack_hash != context_pack_hash for item in witnesses):
            raise ValueError("witness belongs to a different context pack")
        if any(item.authorized_scope_hash != authorized_scope_hash for item in witnesses):
            raise ValueError("witness belongs to a different authorized scope")
        witness_ids = tuple(item.witness_id for item in witnesses)
        payload = _catalog_payload(
            question_contract_id=question_contract_id,
            question_contract_hash=question_contract_hash,
            context_pack_hash=context_pack_hash,
            authorized_scope_hash=authorized_scope_hash,
            witness_ids=witness_ids,
            witness_count=len(witness_ids),
            universe_status=universe_status,
            omissions=omissions,
            builder_version=builder_version,
        )
        return cls(
            **payload,
            catalog_id=stable_identifier("wcat_", payload),
            catalog_hash=stable_content_hash(payload),
        )

    def validate_witnesses(self, witnesses: tuple[EvidenceWitness, ...]) -> Self:
        """验证目录身份与实际 Witness artifact 一一对应。"""

        if tuple(item.witness_id for item in witnesses) != self.witness_ids:
            raise ValueError("witness artifacts do not match catalog order")
        if tuple(item.ordinal for item in witnesses) != tuple(range(len(witnesses))):
            raise ValueError("witness artifact ordinals are not contiguous")
        if any(item.context_pack_hash != self.context_pack_hash for item in witnesses):
            raise ValueError("witness artifact context pack mismatch")
        if any(item.authorized_scope_hash != self.authorized_scope_hash for item in witnesses):
            raise ValueError("witness artifact authorized scope mismatch")
        return self


__all__ = [
    "EVIDENCE_SPAN_REF_SCHEMA_VERSION",
    "EVIDENCE_WITNESS_SCHEMA_VERSION",
    "PACKED_TRUNCATION_MARKER",
    "WITNESS_CATALOG_SCHEMA_VERSION",
    "EvidenceSpanRef",
    "EvidenceWitness",
    "SpanEquivalence",
    "WitnessAtomKind",
    "WitnessCatalogOmission",
    "WitnessCatalogReceipt",
    "WitnessOmissionCode",
    "WitnessUniverseStatus",
]
