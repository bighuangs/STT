"""把回答图的最终证明投影为无 Witness/Prompt 正文的 v3 领域评测回执。

公开观测只暴露 ``Mapping``；来源枚举 strict closure 另从瞬时 raw sidecar
重建领域对象进行可信重放。v1/v2 观测不在这里重写；调用者收到 ``None`` 后
必须走历史只读逻辑。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


OBSERVATION_SCHEMA_V3 = "stt.eval.observation.v3"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$")
_SAFE_HASH = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SAFE_SCOPE_HASH = re.compile(r"^scope_[0-9a-f]{64}$")
_PROVEN = frozenset({"valid", "verified", "proven", "supported", "complete", "passed"})
_TERMINAL_NOT_APPLICABLE = frozenset({"clarify", "direct_answer"})
_ROOT_FAILURE_STATUSES = frozenset({"rejected", "invalid", "indeterminate", "failed"})
_STRICT_ANSWER_OUTCOMES = frozenset({"answered"})
_STRICT_REFUSAL_OUTCOMES = frozenset({"evidence_absent", "conflicting"})


@dataclass(frozen=True, slots=True, repr=False)
class ProofEvidenceAuthority:
    """仅在单次评测调用内存活的原始 Evidence 权威正文。"""

    evidence_id: str
    canonical_text: str


@dataclass(frozen=True, slots=True, repr=False)
class ProofRepresentationAuthority:
    """仅在单次评测调用内存活的原始 ContextPack 表示正文。"""

    representation_id: str
    packed_text: str
    evidence_id: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ProofEvaluationAuthority:
    """不进入 observation 的单次评测 sidecar；字段均不可变且非 Mapping。"""

    question_contract_json: str
    witness_catalog_json: str
    witness_rows_json: tuple[str, ...]
    evidences: tuple[ProofEvidenceAuthority, ...]
    representations: tuple[ProofRepresentationAuthority, ...]
    source_ledger_complete: bool
    witness_ledger_complete: bool


_SOURCE_ENUMERATION_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "witness_id",
        "evidence_id",
        "representation_id",
        "anchor_field",
    }
)
_SOURCE_ENUMERATION_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "catalog_hash",
        "obligation_hash",
        "witness_hash",
        "evidence_content_hash",
        "span_ref_hash",
        "row_hash",
        "packed_row_hash",
        "table_hash",
        "packed_table_hash",
        "header_hash",
        "packed_header_hash",
        "separator_hash",
        "packed_separator_hash",
        "table_structure_hash",
        "key_hash",
    }
)
_SOURCE_ENUMERATION_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "table_start",
        "table_end",
        "packed_table_start",
        "packed_table_end",
        "table_line_index",
        "table_column_count",
        "table_data_row_count",
        "witness_table_row_index",
        "key_cell_index",
        "key_occurrence_count",
        "catalog_representation_count",
        "catalog_key_mention_count",
        "canonical_non_table_character_count",
        "packed_non_table_character_count",
    }
)
_SOURCE_ENUMERATION_CLOSURE_FIELDS = (
    _SOURCE_ENUMERATION_CLOSURE_TOKEN_FIELDS
    | _SOURCE_ENUMERATION_CLOSURE_HASH_FIELDS
    | _SOURCE_ENUMERATION_CLOSURE_INTEGER_FIELDS
)

_SOURCE_PROCEDURE_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "evidence_id",
        "representation_id",
        "marker_style",
        "anchor_field",
    }
)
_SOURCE_PROCEDURE_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "catalog_hash",
        "obligation_hash",
        "evidence_content_hash",
        "canonical_block_hash",
        "packed_block_hash",
        "canonical_title_hash",
        "packed_title_hash",
        "list_structure_hash",
        "title_anchor_hash",
    }
)
_SOURCE_PROCEDURE_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "canonical_block_start",
        "canonical_block_end",
        "packed_block_start",
        "packed_block_end",
        "title_line_index",
        "first_item_line_index",
        "item_count",
        "catalog_representation_count",
        "canonical_non_block_character_count",
        "packed_non_block_character_count",
    }
)
_SOURCE_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS = frozenset({"witness_ids"})
_SOURCE_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS = frozenset(
    {
        "witness_hashes",
        "span_ref_hashes",
        "canonical_item_hashes",
        "packed_item_hashes",
    }
)
_SOURCE_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS = frozenset({"witness_ordinals", "item_ordinals"})
_SOURCE_PROCEDURE_CLOSURE_FIELDS = (
    _SOURCE_PROCEDURE_CLOSURE_TOKEN_FIELDS
    | _SOURCE_PROCEDURE_CLOSURE_HASH_FIELDS
    | _SOURCE_PROCEDURE_CLOSURE_INTEGER_FIELDS
    | _SOURCE_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
)

_SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_FIELDS = _SOURCE_PROCEDURE_CLOSURE_TOKEN_FIELDS | frozenset(
    {"selection_kind"}
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_TOKEN_FIELDS = frozenset({"boundary_rule_version"})
_SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_FIELDS = _SOURCE_PROCEDURE_CLOSURE_HASH_FIELDS | frozenset(
    {
        "catalog_authority_hash",
        "canonical_outside_scope_hash",
        "packed_outside_scope_hash",
        "normalized_anchor_hash",
    }
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_HASH_FIELDS = frozenset(
    {"prefix_trigger_hash", "excluded_successor_hash"}
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_FIELDS = (
    _SOURCE_PROCEDURE_CLOSURE_INTEGER_FIELDS
    | frozenset(
        {
            "selected_authority_index",
            "authority_candidate_count",
            "canonical_anchor_start",
            "canonical_anchor_end",
            "packed_anchor_start",
            "packed_anchor_end",
        }
    )
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_INTEGER_FIELDS = frozenset({"excluded_successor_ordinal"})
_SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS = (
    _SOURCE_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
    | frozenset({"catalog_evidence_ids", "catalog_representation_ids"})
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS = (
    _SOURCE_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
    | frozenset(
        {
            "catalog_evidence_content_hashes",
            "catalog_packed_text_hashes",
            "anchor_match_key_hashes",
        }
    )
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS = (
    _SOURCE_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
    | frozenset(
        {
            "canonical_item_starts",
            "canonical_item_ends",
            "packed_item_starts",
            "packed_item_ends",
        }
    )
)
_SOURCE_SCOPED_PROCEDURE_CLOSURE_FIELDS = (
    _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_TOKEN_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_HASH_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_INTEGER_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
)

_SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "evidence_id",
        "representation_id",
        "anchor_field",
        "table_parse_mode",
    }
)
_SOURCE_SCOPED_TABLE_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "catalog_hash",
        "obligation_hash",
        "evidence_content_hash",
        "catalog_authority_hash",
        "selector_header_hash",
        "selector_values_hash",
        "canonical_table_hash",
        "packed_table_hash",
        "canonical_header_hash",
        "packed_header_hash",
        "canonical_separator_hash",
        "packed_separator_hash",
        "table_structure_hash",
        "canonical_outside_table_hash",
        "packed_outside_table_hash",
    }
)
_SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "catalog_representation_count",
        "selected_authority_index",
        "selected_table_index",
        "authority_candidate_count",
        "selector_column_index",
        "selector_count",
        "selected_row_count",
        "canonical_table_start",
        "canonical_table_end",
        "packed_table_start",
        "packed_table_end",
        "table_line_index",
        "table_column_count",
        "table_data_row_count",
        "canonical_non_table_character_count",
        "packed_non_table_character_count",
    }
)
_SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_ARRAY_FIELDS = frozenset(
    {"witness_ids", "catalog_evidence_ids", "catalog_representation_ids"}
)
_SOURCE_SCOPED_TABLE_CLOSURE_HASH_ARRAY_FIELDS = frozenset(
    {
        "witness_hashes",
        "span_ref_hashes",
        "catalog_evidence_content_hashes",
        "catalog_packed_text_hashes",
        "selector_hashes",
        "canonical_row_hashes",
        "packed_row_hashes",
    }
)
_SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_ARRAY_FIELDS = frozenset(
    {
        "witness_ordinals",
        "selector_occurrence_counts",
        "catalog_canonical_selector_cell_counts",
        "catalog_packed_selector_cell_counts",
        "selected_row_indices",
        "canonical_row_starts",
        "canonical_row_ends",
        "packed_row_starts",
        "packed_row_ends",
    }
)
_SOURCE_SCOPED_TABLE_CLOSURE_FIELDS = (
    _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_FIELDS
    | _SOURCE_SCOPED_TABLE_CLOSURE_HASH_FIELDS
    | _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_FIELDS
    | _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_SCOPED_TABLE_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_ARRAY_FIELDS
)

_SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "evidence_id",
        "representation_id",
        "anchor_field",
        "anchor_kind",
        "marker_style",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "catalog_hash",
        "obligation_hash",
        "evidence_content_hash",
        "catalog_authority_hash",
        "anchor_match_hash",
        "canonical_anchor_hash",
        "packed_anchor_hash",
        "canonical_scope_hash",
        "packed_scope_hash",
        "list_structure_hash",
        "canonical_local_section_hash",
        "packed_local_section_hash",
        "canonical_outside_scope_hash",
        "packed_outside_scope_hash",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_HASH_FIELDS = frozenset(
    {"canonical_title_hash", "packed_title_hash"}
)
_SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "catalog_representation_count",
        "selected_authority_index",
        "selected_block_index",
        "authority_candidate_count",
        "anchor_match_key_count",
        "canonical_anchor_start",
        "canonical_anchor_end",
        "packed_anchor_start",
        "packed_anchor_end",
        "canonical_anchor_line_index",
        "packed_anchor_line_index",
        "item_count",
        "canonical_scope_start",
        "canonical_scope_end",
        "packed_scope_start",
        "packed_scope_end",
        "canonical_local_section_start",
        "canonical_local_section_end",
        "packed_local_section_start",
        "packed_local_section_end",
        "canonical_non_scope_character_count",
        "packed_non_scope_character_count",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_INTEGER_FIELDS = frozenset(
    {
        "canonical_title_start",
        "canonical_title_end",
        "packed_title_start",
        "packed_title_end",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_ARRAY_FIELDS = frozenset(
    {"witness_ids", "catalog_evidence_ids", "catalog_representation_ids"}
)
_SOURCE_SCOPED_BULLET_CLOSURE_HASH_ARRAY_FIELDS = frozenset(
    {
        "witness_hashes",
        "span_ref_hashes",
        "catalog_evidence_content_hashes",
        "catalog_packed_text_hashes",
        "anchor_match_key_hashes",
        "canonical_item_hashes",
        "packed_item_hashes",
        "canonical_item_body_hashes",
        "packed_item_body_hashes",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_ARRAY_FIELDS = frozenset(
    {
        "witness_ordinals",
        "canonical_item_starts",
        "canonical_item_ends",
        "packed_item_starts",
        "packed_item_ends",
    }
)
_SOURCE_SCOPED_BULLET_CLOSURE_FIELDS = (
    _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_HASH_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_HASH_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_INTEGER_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_ARRAY_FIELDS
)

# 此 allowlist 与 v1 领域 receipt 的 61 个公共字段逐一对应；任何增删都必须
# 通过显式 schema 版本演进，避免 evaluator 静默接受额外自报权威。
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "witness_atom_kind",
        "evidence_id",
        "representation_id",
        "anchor_field",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "source_query_hash",
        "catalog_hash",
        "obligation_hash",
        "information_need_hash",
        "evidence_content_hash",
        "canonical_witness_hash",
        "packed_witness_hash",
        "canonical_title_hash",
        "packed_title_hash",
        "normalized_title_hash",
        "action_structure_hash",
        "catalog_authority_hash",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_HASH_FIELDS = frozenset(
    {
        "canonical_terminal_result_hash",
        "packed_terminal_result_hash",
        "local_trigger_hash",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "canonical_witness_start",
        "canonical_witness_end",
        "packed_witness_start",
        "packed_witness_end",
        "canonical_title_start",
        "canonical_title_end",
        "packed_title_start",
        "packed_title_end",
        "action_count",
        "eligible_candidate_count",
        "top_score_candidate_count",
        "catalog_representation_count",
        "selected_authority_index",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_INTEGER_FIELDS = frozenset(
    {
        "canonical_terminal_result_start",
        "canonical_terminal_result_end",
        "packed_terminal_result_start",
        "packed_terminal_result_end",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_ARRAY_FIELDS = frozenset(
    {"witness_ids", "catalog_evidence_ids", "catalog_representation_ids"}
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_ARRAY_FIELDS = frozenset(
    {
        "witness_hashes",
        "span_ref_hashes",
        "canonical_action_hashes",
        "packed_action_hashes",
        "anchor_match_key_hashes",
        "catalog_evidence_content_hashes",
        "catalog_packed_text_hashes",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_ARRAY_FIELDS = frozenset(
    {
        "witness_ordinals",
        "canonical_action_starts",
        "canonical_action_ends",
        "packed_action_starts",
        "packed_action_ends",
        "action_ordinals",
        "selection_score",
    }
)
_SOURCE_INLINE_ACTION_CHAIN_CLOSURE_FIELDS = (
    _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_HASH_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_INTEGER_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_ARRAY_FIELDS
)

_SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "receipt_id",
        "question_contract_id",
        "catalog_id",
        "obligation_id",
        "mode",
        "relation",
        "anchor_field",
        "evidence_id",
        "representation_id",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_FIELDS = frozenset(
    {
        "receipt_hash",
        "question_contract_hash",
        "catalog_hash",
        "obligation_hash",
        "anchor_match_hash",
        "predicate_signature_hash",
        "selector_hash",
        "catalog_authority_hash",
        "evidence_content_hash",
        "canonical_scope_hash",
        "packed_scope_hash",
        "canonical_outside_scope_hash",
        "packed_outside_scope_hash",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_HASH_FIELDS = frozenset(
    {"canonical_title_hash", "packed_title_hash"}
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_FIELDS = frozenset(
    {
        "anchor_match_key_count",
        "topic_group_count",
        "catalog_representation_count",
        "selected_authority_index",
        "authority_candidate_count",
        "canonical_scope_start",
        "canonical_scope_end",
        "packed_scope_start",
        "packed_scope_end",
        "canonical_non_scope_character_count",
        "packed_non_scope_character_count",
        "member_count",
        "semantic_count",
        "omission_marker_count",
        "open_marker_count",
        "reference_marker_count",
        "ambiguous_anaphor_count",
        "truncation_marker_count",
        "supplement_candidate_count",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_INTEGER_FIELDS = frozenset(
    {
        "canonical_title_start",
        "canonical_title_end",
        "packed_title_start",
        "packed_title_end",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_ARRAY_FIELDS = frozenset(
    {
        "catalog_evidence_ids",
        "catalog_representation_ids",
        "witness_ids",
        "member_atom_kinds",
        "segmentation_methods",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_ARRAY_FIELDS = frozenset(
    {
        "anchor_match_key_hashes",
        "topic_group_hashes",
        "catalog_evidence_content_hashes",
        "catalog_packed_text_hashes",
        "witness_hashes",
        "span_ref_hashes",
        "canonical_member_hashes",
        "packed_member_hashes",
        "canonical_member_normalized_hashes",
        "packed_member_normalized_hashes",
        "flattened_semantic_hashes",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_ARRAY_FIELDS = frozenset(
    {
        "topic_group_member_counts",
        "witness_ordinals",
        "canonical_member_starts",
        "canonical_member_ends",
        "packed_member_starts",
        "packed_member_ends",
        "member_semantic_counts",
    }
)
_SOURCE_SCOPED_FACT_SET_CLOSURE_FIELDS = (
    _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_HASH_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_INTEGER_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_ARRAY_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_ARRAY_FIELDS
    | _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_ARRAY_FIELDS
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _token(value: object) -> str | None:
    text = str(value or "").strip()
    return text if _SAFE_TOKEN.fullmatch(text) else None


def _hash(value: object) -> str | None:
    text = str(value or "").strip().lower()
    return text if _SAFE_HASH.fullmatch(text) else None


def _scope_hash(value: object) -> str | None:
    text = str(value or "").strip().lower()
    return text if _SAFE_SCOPE_HASH.fullmatch(text) else None


def _enum(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text and len(text) <= 128 and _SAFE_TOKEN.fullmatch(text) else None


def _integer(value: object, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _tokens(value: object, *, limit: int = 4096) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for raw in value:
        item = _token(raw)
        if item and item not in result:
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _exact_tokens(value: object, *, limit: int = 64) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if not 1 <= len(value) <= limit:
        return None
    result = [_token(item) for item in value]
    return None if any(item is None for item in result) else [str(item) for item in result]


def _exact_hashes(value: object, *, limit: int = 64) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if not 1 <= len(value) <= limit:
        return None
    result = [_hash(item) for item in value]
    return None if any(item is None for item in result) else [str(item) for item in result]


def _exact_integers(value: object, *, limit: int = 64) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if not 1 <= len(value) <= limit:
        return None
    result = [_integer(item) for item in value]
    return None if any(item is None for item in result) else [int(item) for item in result]


def _codes(value: object, *, limit: int = 64) -> list[str]:
    return _tokens(value, limit=limit)


def _set_if(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != []:
        target[key] = value


def _stable_mapping_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _stable_sequence_hash(value: Sequence[object]) -> str:
    encoded = json.dumps(
        list(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _physical_text_lines(value: str) -> list[tuple[int, int, int, str]]:
    lines: list[tuple[int, int, int, str]] = []
    start = 0
    for delimiter in re.finditer(r"\r\n|\r|\n", value):
        lines.append((len(lines), start, delimiter.start(), value[start : delimiter.start()]))
        start = delimiter.end()
    if start < len(value):
        lines.append((len(lines), start, len(value), value[start:]))
    return lines


def _simple_markdown_row(value: str) -> tuple[tuple[str, ...], bool] | None:
    if "|" not in value or "\\|" in value or "`" in value:
        return None
    body = value.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = tuple(item.strip() for item in body.split("|"))
    if len(cells) < 2 or any(not item for item in cells):
        return None
    separator = all(re.fullmatch(r":?-{3,}:?", item) is not None for item in cells)
    return cells, separator


def _normalized_table_cell(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _full_simple_table_summary(
    text: str,
    *,
    witness_start: int,
    witness_end: int,
) -> dict[str, Any] | None:
    """直接从权威正文生成无正文表结构摘要，不读取待验证 receipt 的计数字段。"""

    lines = _physical_text_lines(text)
    content_indices = [index for index, _, _, line in lines if line.strip()]
    if len(content_indices) < 3:
        return None
    first_index = content_indices[0]
    last_index = content_indices[-1]
    if content_indices != list(range(first_index, last_index + 1)):
        return None
    selected_lines = lines[first_index : last_index + 1]
    parsed = [_simple_markdown_row(line) for _, _, _, line in selected_lines]
    if any(item is None for item in parsed):
        return None
    rows = [item for item in parsed if item is not None]
    column_count = len(rows[0][0])
    if (
        rows[0][1]
        or not rows[1][1]
        or any(item[1] for item in rows[2:])
        or any(len(item[0]) != column_count for item in rows)
    ):
        return None
    witness_matches = [
        index
        for index, (_, start, end, _) in enumerate(selected_lines[2:])
        if start == witness_start and end == witness_end
    ]
    if len(witness_matches) != 1:
        return None
    header_line = selected_lines[0]
    separator_line = selected_lines[1]
    witness_line = selected_lines[2 + witness_matches[0]]
    structure = {
        "column_count": column_count,
        "header": [_normalized_table_cell(cell) for cell in rows[0][0]],
        "separator": [_normalized_table_cell(cell) for cell in rows[1][0]],
        "data_rows": [[_normalized_table_cell(cell) for cell in row[0]] for row in rows[2:]],
        "witness_row_index": witness_matches[0],
    }
    table_start = header_line[1]
    table_end = selected_lines[-1][2]
    return {
        "table_start": table_start,
        "table_end": table_end,
        "table_line_index": header_line[0],
        "table_column_count": column_count,
        "table_data_row_count": len(rows) - 2,
        "witness_table_row_index": witness_matches[0],
        "table_hash": _text_hash(text[table_start:table_end]),
        "header_hash": _text_hash(header_line[3]),
        "separator_hash": _text_hash(separator_line[3]),
        "row_hash": _text_hash(witness_line[3]),
        "table_structure_hash": _stable_mapping_hash(structure),
    }


def _source_table_authority_summaries(
    source_receipts: Sequence[Mapping[str, Any]],
    *,
    packed_blocks: Sequence[Mapping[str, Any]],
    context_aliases: object,
    witness_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks_by_representation = {
        str(item.get("representation_id") or ""): item for item in packed_blocks
    }
    evidence_by_id = {
        str(evidence.get("evidence_id") or ""): evidence
        for alias in _items(context_aliases)
        if (evidence := _mapping(alias.get("evidence")))
    }
    witnesses_by_id = {str(item.get("witness_id") or ""): item for item in witness_rows}
    summaries: list[dict[str, Any]] = []
    for receipt in source_receipts:
        receipt_id = _token(receipt.get("receipt_id"))
        receipt_hash = _hash(receipt.get("receipt_hash"))
        representation_id = _token(receipt.get("representation_id"))
        evidence_id = _token(receipt.get("evidence_id"))
        witness_id = _token(receipt.get("witness_id"))
        block = blocks_by_representation.get(representation_id or "", {})
        evidence = evidence_by_id.get(evidence_id or "", {})
        witness = witnesses_by_id.get(witness_id or "", {})
        span = _mapping(witness.get("span_ref"))
        packed_text = block.get("text")
        canonical_text = evidence.get("content")
        if (
            receipt_id is None
            or receipt_hash is None
            or representation_id is None
            or evidence_id is None
            or witness_id is None
            or not isinstance(packed_text, str)
            or not isinstance(canonical_text, str)
        ):
            continue
        canonical_start = span.get("canonical_start")
        canonical_end = span.get("canonical_end")
        packed_start = span.get("packed_start")
        packed_end = span.get("packed_end")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (canonical_start, canonical_end, packed_start, packed_end)
        ):
            continue
        canonical = _full_simple_table_summary(
            canonical_text,
            witness_start=int(canonical_start),
            witness_end=int(canonical_end),
        )
        packed = _full_simple_table_summary(
            packed_text,
            witness_start=int(packed_start),
            witness_end=int(packed_end),
        )
        if canonical is None or packed is None:
            continue
        summaries.append(
            {
                "schema_version": "source-table-authority-summary.v1",
                "receipt_id": receipt_id,
                "receipt_hash": receipt_hash,
                "witness_id": witness_id,
                "evidence_id": evidence_id,
                "representation_id": representation_id,
                "canonical": canonical,
                "packed": packed,
            }
        )
    return summaries


_PROCEDURE_ARABIC_ITEM_RE = re.compile(
    r"^(?P<ordinal>\d{1,3})(?:(?P<dot>[.)．])(?=\s|[^\d\s])|(?P<dunhao>、)(?!\d))"
    r"\s*(?P<body>\S(?:.*\S)?)$"
)
_PROCEDURE_CHINESE_ITEM_RE = re.compile(
    r"^(?P<ordinal>[一二三四五六七八九十]{1,3})(?P<marker>[、.)．])"
    r"\s*(?P<body>\S(?:.*\S)?)$"
)
_PROCEDURE_HEADING_RE = re.compile(r"^#{1,6}\s+")
_PROCEDURE_SECTION_NUMBER_RE = re.compile(
    r"^(?:(?:第\s*)?\d+(?:\s*[.．]\s*\d+){0,5}\s*(?:章|节|章节)?\s*)"
)


def _procedure_default_ignorable(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or 0x180B <= codepoint <= 0x180D
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _procedure_anchor_normalized(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
        and not _procedure_default_ignorable(character)
    )


_GENERIC_PROCEDURE_TITLE_HASH_INPUTS = frozenset(
    _procedure_anchor_normalized(value)
    for value in (
        "步骤",
        "操作步骤",
        "操作流程",
        "流程",
        "操作方法",
        "procedure",
        "steps",
        "instructions",
    )
)


def _procedure_title_normalized(value: str) -> str:
    title = _PROCEDURE_HEADING_RE.sub("", value.strip())
    if len(title) >= 4 and title[:2] == title[-2:] and title[:2] in {"**", "__"}:
        title = title[2:-2].strip()
    title = _PROCEDURE_SECTION_NUMBER_RE.sub("", title).strip(" \t:：-—")
    return _procedure_anchor_normalized(title)


def _procedure_chinese_ordinal(value: str) -> int | None:
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" not in value:
        return digits.get(value)
    left, _, right = value.partition("十")
    result = digits.get(left, 1) * 10 + (digits.get(right, 0) if right else 0)
    return result if 1 <= result <= 64 else None


def _procedure_ordered_item(value: str) -> tuple[int, str, str] | None:
    matched = _PROCEDURE_ARABIC_ITEM_RE.fullmatch(value)
    if matched is not None:
        ordinal = int(matched.group("ordinal"))
        marker_style = f"arabic:{matched.group('dot') or '、'}"
        body = matched.group("body")
    else:
        matched = _PROCEDURE_CHINESE_ITEM_RE.fullmatch(value)
        if matched is None:
            return None
        ordinal = _procedure_chinese_ordinal(matched.group("ordinal")) or 0
        marker_style = f"chinese:{matched.group('marker')}"
        body = matched.group("body")
    if not 1 <= ordinal <= 64:
        return None
    return ordinal, marker_style, body


def _full_procedure_list_summary(
    text: str,
    *,
    witness_spans: Sequence[tuple[int, int]],
) -> dict[str, Any] | None:
    """直接从权威正文重建完整标题+有序步骤块，不信任 receipt 的计数字段。"""

    nonempty: list[tuple[tuple[int, int, int, str], int, int, str]] = []
    for line in _physical_text_lines(text):
        raw = line[3]
        left = 0
        right = len(raw)
        while left < right and raw[left].isspace():
            left += 1
        while right > left and raw[right - 1].isspace():
            right -= 1
        if left < right:
            nonempty.append((line, line[1] + left, line[1] + right, raw[left:right]))
    if len(nonempty) < 3:
        return None
    title_line, title_start, title_end, title = nonempty[0]
    if _procedure_ordered_item(title) is not None or _simple_markdown_row(title) is not None:
        return None
    parsed_items = [_procedure_ordered_item(item[3]) for item in nonempty[1:]]
    if any(item is None for item in parsed_items):
        return None
    items = [item for item in parsed_items if item is not None]
    item_lines = nonempty[1:]
    if [item[0][0] for item in item_lines] != list(
        range(title_line[0] + 1, title_line[0] + 1 + len(items))
    ):
        return None
    ordinals = [item[0] for item in items]
    if ordinals != list(range(1, len(items) + 1)):
        return None
    marker_styles = {item[1] for item in items}
    if len(marker_styles) != 1:
        return None
    normalized_bodies = [_normalized_table_cell(item[2]) for item in items]
    if any(not item for item in normalized_bodies) or len(set(normalized_bodies)) != len(items):
        return None
    normalized_title = _procedure_title_normalized(title)
    if (
        len(normalized_title) < 2
        or normalized_title.isnumeric()
        or normalized_title in _GENERIC_PROCEDURE_TITLE_HASH_INPUTS
    ):
        return None
    item_spans = [(item[1], item[2]) for item in item_lines]
    if list(witness_spans) != item_spans:
        return None
    block_start = title_start
    block_end = item_lines[-1][2]
    non_block_count = sum(
        not character.isspace() for character in text[:block_start] + text[block_end:]
    )
    if non_block_count:
        return None
    structure = {
        "title": normalized_title,
        "title_line_index": title_line[0],
        "first_item_line_index": item_lines[0][0][0],
        "marker_style": next(iter(marker_styles)),
        "ordinals": ordinals,
        "items": normalized_bodies,
    }
    return {
        "block_start": block_start,
        "block_end": block_end,
        "title_line_index": title_line[0],
        "first_item_line_index": item_lines[0][0][0],
        "item_count": len(items),
        "item_ordinals": ordinals,
        "marker_style": next(iter(marker_styles)),
        "item_starts": [start for start, _ in item_spans],
        "item_ends": [end for _, end in item_spans],
        "item_hashes": [_text_hash(text[start:end]) for start, end in item_spans],
        "block_hash": _text_hash(text[block_start:block_end]),
        "title_hash": _text_hash(text[title_start:title_end]),
        "list_structure_hash": _stable_mapping_hash(structure),
        "title_anchor_hash": _text_hash(normalized_title),
        "non_block_character_count": 0,
    }


def _source_procedure_authority_summaries(
    source_receipts: Sequence[Mapping[str, Any]],
    *,
    packed_blocks: Sequence[Mapping[str, Any]],
    context_aliases: object,
    witness_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks_by_representation = {
        str(item.get("representation_id") or ""): item for item in packed_blocks
    }
    evidence_by_id = {
        str(evidence.get("evidence_id") or ""): evidence
        for alias in _items(context_aliases)
        if (evidence := _mapping(alias.get("evidence")))
    }
    witnesses_by_id = {str(item.get("witness_id") or ""): item for item in witness_rows}
    summaries: list[dict[str, Any]] = []
    for receipt in source_receipts:
        if receipt.get("schema_version") != "query-anchored-procedure-list-closure-receipt.v1":
            continue
        receipt_id = _token(receipt.get("receipt_id"))
        receipt_hash = _hash(receipt.get("receipt_hash"))
        representation_id = _token(receipt.get("representation_id"))
        evidence_id = _token(receipt.get("evidence_id"))
        witness_ids = _exact_tokens(receipt.get("witness_ids"))
        block = blocks_by_representation.get(representation_id or "", {})
        evidence = evidence_by_id.get(evidence_id or "", {})
        packed_text = block.get("text")
        canonical_text = evidence.get("content")
        if (
            receipt_id is None
            or receipt_hash is None
            or representation_id is None
            or evidence_id is None
            or witness_ids is None
            or len(witness_ids) < 2
            or not isinstance(packed_text, str)
            or not isinstance(canonical_text, str)
        ):
            continue
        canonical_spans: list[tuple[int, int]] = []
        packed_spans: list[tuple[int, int]] = []
        valid_spans = True
        for witness_id in witness_ids:
            span = _mapping(witnesses_by_id.get(witness_id, {}).get("span_ref"))
            values = (
                span.get("canonical_start"),
                span.get("canonical_end"),
                span.get("packed_start"),
                span.get("packed_end"),
            )
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                valid_spans = False
                break
            canonical_spans.append((int(values[0]), int(values[1])))
            packed_spans.append((int(values[2]), int(values[3])))
        if not valid_spans:
            continue
        canonical = _full_procedure_list_summary(
            canonical_text,
            witness_spans=canonical_spans,
        )
        packed = _full_procedure_list_summary(packed_text, witness_spans=packed_spans)
        if canonical is None or packed is None:
            continue
        summaries.append(
            {
                "schema_version": "source-procedure-list-authority-summary.v1",
                "receipt_id": receipt_id,
                "receipt_hash": receipt_hash,
                "witness_ids": witness_ids,
                "evidence_id": evidence_id,
                "representation_id": representation_id,
                "canonical": canonical,
                "packed": packed,
            }
        )
    return summaries


def build_proof_evaluation_authority(
    state: Mapping[str, Any],
) -> ProofEvaluationAuthority:
    """从 Query final_state 冻结不持久化的原始 proof 权威 sidecar。

    这里保留原始 canonical/packed 正文和完整 Witness ledger；结构重放延迟到
    ``answer_proof_evaluation``，避免把预计算摘要或无密钥哈希误当信任根。
    """

    raw_aliases = _items(state.get("context_aliases"))
    evidences: list[ProofEvidenceAuthority] = []
    for alias in raw_aliases:
        evidence = _mapping(alias.get("evidence"))
        evidence_id = _token(evidence.get("evidence_id"))
        canonical_text = evidence.get("content")
        if evidence_id is not None and isinstance(canonical_text, str):
            evidences.append(
                ProofEvidenceAuthority(
                    evidence_id=evidence_id,
                    canonical_text=canonical_text,
                )
            )

    representations: list[ProofRepresentationAuthority] = []
    context_pack = _mapping(state.get("context_pack"))
    raw_blocks = _items(context_pack.get("packed_blocks"))
    for block in raw_blocks:
        representation_id = _token(block.get("representation_id"))
        evidence_id = _token(block.get("evidence_id"))
        packed_text = block.get("text")
        if representation_id is not None and isinstance(packed_text, str):
            representations.append(
                ProofRepresentationAuthority(
                    representation_id=representation_id,
                    packed_text=packed_text,
                    evidence_id=evidence_id,
                )
            )

    question_contract = _mapping(state.get("question_contract"))
    catalog_container = _mapping(state.get("witness_catalog"))
    catalog_receipt = _mapping(
        catalog_container.get("receipt")
        or catalog_container.get("catalog_receipt")
        or catalog_container.get("catalog")
    )
    if not catalog_receipt:
        catalog_receipt = catalog_container
    witness_rows = _items(catalog_container.get("witnesses") or catalog_container.get("items"))

    def canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    question_contract_json = canonical_json(question_contract) if question_contract else ""
    witness_catalog_json = canonical_json(catalog_receipt) if catalog_receipt else ""
    witness_rows_json = tuple(canonical_json(row) for row in witness_rows)
    witness_ids = [_token(row.get("witness_id")) for row in witness_rows]
    ordinals = [_integer(row.get("ordinal")) for row in witness_rows]
    evidence_ids = [item.evidence_id for item in evidences]
    representation_ids = [item.representation_id for item in representations]
    source_ledger_complete = (
        bool(question_contract_json)
        and bool(witness_catalog_json)
        and len(evidences) == len(raw_aliases)
        and len(representations) == len(raw_blocks)
        and len(set(evidence_ids)) == len(evidence_ids)
        and len(set(representation_ids)) == len(representation_ids)
    )
    ledger_complete = (
        all(item is not None for item in witness_ids)
        and len(set(witness_ids)) == len(witness_ids)
        and ordinals == list(range(len(witness_rows)))
    )
    return ProofEvaluationAuthority(
        question_contract_json=question_contract_json,
        witness_catalog_json=witness_catalog_json,
        witness_rows_json=witness_rows_json,
        evidences=tuple(evidences),
        representations=tuple(representations),
        source_ledger_complete=source_ledger_complete,
        witness_ledger_complete=ledger_complete,
    )


def _stable_identity_matches(
    value: Mapping[str, Any],
    *,
    prefix: str,
    identity_key: str,
    hash_key: str,
    payload: Mapping[str, Any],
) -> bool:
    expected_hash = _stable_mapping_hash(payload)
    expected_id = prefix + expected_hash.removeprefix("sha256:")[:40]
    return value.get(hash_key) == expected_hash and value.get(identity_key) == expected_id


def _verifier_serialization(value: object) -> dict[str, Any]:
    verifier = _mapping(value)
    return {
        "verifier_id": verifier.get("verifier_id"),
        "verifier_version": verifier.get("verifier_version"),
    }


def _answer_span_serialization(value: object) -> dict[str, Any] | None:
    span = _mapping(value)
    if not span:
        return None
    return {"start": span.get("start"), "end": span.get("end")}


def _claim_proof_serialization(value: Mapping[str, Any]) -> dict[str, Any]:
    """还原 ClaimProof.to_dict() 的身份相关结构，排除评测派生 bindings。"""

    return {
        "schema_version": value.get("schema_version"),
        "claim_proof_id": value.get("claim_proof_id"),
        "claim_proof_hash": value.get("claim_proof_hash"),
        "claim_id": value.get("claim_id"),
        "obligation_id": value.get("obligation_id"),
        "status": value.get("status"),
        "method": value.get("method"),
        "witness_ids": _tokens(value.get("witness_ids")),
        "answer_span": _answer_span_serialization(value.get("answer_span")),
        "rendered_text_hash": value.get("rendered_text_hash"),
        "relation": value.get("relation"),
        "verifier": _verifier_serialization(value.get("verifier")),
        "failure_codes": _codes(value.get("failure_codes")),
    }


def _claim_proof_identity_is_valid(value: Mapping[str, Any]) -> bool:
    serialized = _claim_proof_serialization(value)
    payload = {
        key: item
        for key, item in serialized.items()
        if key not in {"claim_proof_id", "claim_proof_hash"}
    }
    return _stable_identity_matches(
        value,
        prefix="cpr_",
        identity_key="claim_proof_id",
        hash_key="claim_proof_hash",
        payload=payload,
    )


def _proven_item_serialization(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value.get("schema_version"),
        "item_ref_id": value.get("item_ref_id"),
        "item_ref_hash": value.get("item_ref_hash"),
        "obligation_id": value.get("obligation_id"),
        "claim_id": value.get("claim_id"),
        "witness_id": value.get("witness_id"),
        "witness_hash": value.get("witness_hash"),
        "relative_start": value.get("relative_start"),
        "relative_end": value.get("relative_end"),
        "raw_span_hash": value.get("raw_span_hash"),
        "normalized_item_hash": value.get("normalized_item_hash"),
        "segmentation_method": value.get("segmentation_method"),
    }


def _proven_item_identity_is_valid(value: Mapping[str, Any]) -> bool:
    serialized = _proven_item_serialization(value)
    payload = {
        key: item for key, item in serialized.items() if key not in {"item_ref_id", "item_ref_hash"}
    }
    return _stable_identity_matches(
        value,
        prefix="pit_",
        identity_key="item_ref_id",
        hash_key="item_ref_hash",
        payload=payload,
    )


def _obligation_proof_serialization(value: Mapping[str, Any]) -> dict[str, Any]:
    source_closure = _mapping(value.get("source_enumeration_closure_receipt"))
    return {
        "schema_version": value.get("schema_version"),
        "obligation_proof_id": value.get("obligation_proof_id"),
        "obligation_proof_hash": value.get("obligation_proof_hash"),
        "obligation_id": value.get("obligation_id"),
        "status": value.get("status"),
        "outcome": value.get("outcome"),
        "supporting_claim_ids": _tokens(value.get("supporting_claim_ids")),
        "contradiction_witness_ids": _tokens(value.get("contradiction_witness_ids")),
        "closure": value.get("closure"),
        "required_output_form": value.get("required_output_form"),
        "expected_item_count": value.get("expected_item_count"),
        "proven_item_count": value.get("proven_item_count"),
        "proven_items": [
            _proven_item_serialization(item) for item in _items(value.get("proven_items"))
        ],
        "closure_status": value.get("closure_status"),
        "source_enumeration_closure_receipt": dict(source_closure) if source_closure else None,
        "catalog_id": value.get("catalog_id"),
        "catalog_hash": value.get("catalog_hash"),
        "catalog_universe_status": value.get("catalog_universe_status"),
        "verifier": _verifier_serialization(value.get("verifier")),
        "failure_codes": _codes(value.get("failure_codes")),
    }


def _obligation_proof_identity_is_valid(value: Mapping[str, Any]) -> bool:
    serialized = _obligation_proof_serialization(value)
    payload = {
        key: item
        for key, item in serialized.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    return _stable_identity_matches(
        value,
        prefix="opr_",
        identity_key="obligation_proof_id",
        hash_key="obligation_proof_hash",
        payload=payload,
    )


def _semantic_review_serialization(value: Mapping[str, Any]) -> dict[str, Any] | None:
    schema_version = value.get("schema_version")
    common = {
        "schema_version": schema_version,
        "receipt_id": value.get("receipt_id"),
        "receipt_hash": value.get("receipt_hash"),
        "answer_plan_id": value.get("answer_plan_id"),
        "answer_plan_hash": value.get("answer_plan_hash"),
        "catalog_id": value.get("catalog_id"),
        "catalog_hash": value.get("catalog_hash"),
        "decision": value.get("decision"),
    }
    tail = {
        "review_prompt_version": value.get("review_prompt_version"),
        "model_id": value.get("model_id"),
        "verifier_version": value.get("verifier_version"),
        "review_logical_call_id": value.get("review_logical_call_id"),
        "review_prompt_hash": value.get("review_prompt_hash"),
        "review_response_hash": value.get("review_response_hash"),
        "review_proposal_hash": value.get("review_proposal_hash"),
    }
    if schema_version == "semantic-review-closure.v2":
        return {
            **common,
            "closed_obligation_ids": _tokens(value.get("closed_obligation_ids")),
            **tail,
        }
    if schema_version == "semantic-review-terminal.v1":
        return {
            **common,
            "reviewed_answer_plan_id": value.get("reviewed_answer_plan_id"),
            "reviewed_answer_plan_hash": value.get("reviewed_answer_plan_hash"),
            "reason_code": value.get("reason_code"),
            "closed_obligation_ids": [],
            "reviewed_closed_obligation_ids": _tokens(value.get("reviewed_closed_obligation_ids")),
            **tail,
        }
    return None


def _semantic_review_identity_is_valid(value: Mapping[str, Any]) -> bool:
    serialized = _semantic_review_serialization(value)
    if serialized is None:
        return False
    prefix = "src_" if value.get("schema_version") == "semantic-review-closure.v2" else "srt_"
    payload = {
        key: item for key, item in serialized.items() if key not in {"receipt_id", "receipt_hash"}
    }
    return _stable_identity_matches(
        value,
        prefix=prefix,
        identity_key="receipt_id",
        hash_key="receipt_hash",
        payload=payload,
    )


def _failure_serialization(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": value.get("stage"),
        "code": value.get("code"),
        "subject_id": value.get("subject_id"),
        "retryability": value.get("retryability"),
        "diagnostic_hash": value.get("diagnostic_hash"),
    }


def _answer_proof_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    semantic = _mapping(value.get("semantic_review_closure"))
    return {
        "schema_version": value.get("schema_version"),
        "answer_plan_id": value.get("answer_plan_id"),
        "answer_plan_hash": value.get("answer_plan_hash"),
        "question_contract_id": value.get("question_contract_id"),
        "question_contract_hash": value.get("question_contract_hash"),
        "context_pack_hash": value.get("context_pack_hash"),
        "witness_catalog_id": value.get("witness_catalog_id"),
        "witness_catalog_hash": value.get("witness_catalog_hash"),
        "semantic_review_closure": (_semantic_review_serialization(semantic) if semantic else None),
        "answer_id": value.get("answer_id"),
        "answer_hash": value.get("answer_hash"),
        "decision": value.get("decision"),
        "overall_status": value.get("overall_status"),
        "grounding_status": value.get("grounding_status"),
        "relevance_status": value.get("relevance_status"),
        "completeness_status": value.get("completeness_status"),
        "strict_eligible": value.get("strict_eligible"),
        "claim_proofs": [
            _claim_proof_serialization(item) for item in _items(value.get("claim_proofs"))
        ],
        "obligation_proofs": [
            _obligation_proof_serialization(item) for item in _items(value.get("obligation_proofs"))
        ],
        "verifier_versions": [
            _verifier_serialization(item) for item in _items(value.get("verifier_versions"))
        ],
        "failures": [_failure_serialization(item) for item in _items(value.get("failures"))],
    }


def _answer_proof_identity_is_valid(value: Mapping[str, Any]) -> bool:
    payload = _answer_proof_identity_payload(value)
    return _stable_identity_matches(
        value,
        prefix="apr_",
        identity_key="proof_id",
        hash_key="proof_hash",
        payload=payload,
    )


def _question_contract_receipt(value: object) -> dict[str, Any]:
    contract = _mapping(value)
    if not contract:
        return {}
    obligations = _items(contract.get("obligations"))
    obligation_ids = [
        item for raw in obligations if (item := _token(raw.get("obligation_id"))) is not None
    ]
    required_ids = [
        item
        for raw in obligations
        if str(raw.get("necessity") or "required").strip().lower() == "required"
        and (item := _token(raw.get("obligation_id"))) is not None
    ]
    obligation_receipts: list[dict[str, Any]] = []
    for raw in obligations:
        item: dict[str, Any] = {}
        for key in (
            "obligation_id",
            "necessity",
            "polarity",
            "closure",
            "kind",
            "required_output_form",
        ):
            _set_if(item, key, _token(raw.get(key)))
        _set_if(item, "expected_count", _integer(raw.get("expected_count"), minimum=1))
        _set_if(item, "obligation_hash", _hash(raw.get("obligation_hash")))
        obligation_receipts.append(item)
    receipt: dict[str, Any] = {
        "obligation_count": len(obligation_ids),
        "obligation_ids": obligation_ids,
        "required_obligation_ids": required_ids,
        "obligations": obligation_receipts,
    }
    for key in ("schema_version", "contract_id", "query_plan_id", "builder_version"):
        _set_if(receipt, key, _token(contract.get(key)))
    _set_if(
        receipt,
        "contract_hash",
        _hash(contract.get("contract_hash") or contract.get("question_contract_hash")),
    )
    _set_if(receipt, "source_query_hash", _hash(contract.get("source_query_hash")))
    return receipt


def _witness_catalog_receipt(
    value: object,
    *,
    context_pack: object = None,
    context_aliases: object = None,
) -> dict[str, Any]:
    catalog_container = _mapping(value)
    if not catalog_container:
        return {}
    receipt_source = _mapping(
        catalog_container.get("receipt")
        or catalog_container.get("catalog_receipt")
        or catalog_container.get("catalog")
    )
    if not receipt_source:
        receipt_source = catalog_container
    witness_rows = _items(catalog_container.get("witnesses") or catalog_container.get("items"))
    witness_ids = _tokens(receipt_source.get("witness_ids"))
    if not witness_ids:
        witness_ids = [
            item for row in witness_rows if (item := _token(row.get("witness_id"))) is not None
        ]
    omission_rows = _items(receipt_source.get("omissions"))
    omission_codes = [
        item
        for row in omission_rows
        if (item := _token(row.get("code") or row.get("reason_code"))) is not None
    ]
    receipt: dict[str, Any] = {
        "witness_count": _integer(receipt_source.get("witness_count"))
        if _integer(receipt_source.get("witness_count")) is not None
        else len(witness_ids),
        "witness_ids": witness_ids,
        "witness_atom_kinds": {
            witness_id: atom_kind
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (atom_kind := _token(row.get("atom_kind"))) is not None
        },
        "witness_hashes": {
            witness_id: witness_hash
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (witness_hash := _hash(row.get("witness_hash"))) is not None
        },
        "witness_evidence_ids": {
            witness_id: evidence_id
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (evidence_id := _token(_mapping(row.get("span_ref")).get("evidence_id")))
            is not None
        },
        "witness_evidence_content_hashes": {
            witness_id: content_hash
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (content_hash := _hash(_mapping(row.get("span_ref")).get("evidence_content_hash")))
            is not None
        },
        "witness_representation_ids": {
            witness_id: representation_id
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (
                representation_id := _token(_mapping(row.get("span_ref")).get("representation_id"))
            )
            is not None
        },
        "witness_span_ref_hashes": {
            witness_id: span_ref_hash
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (span_ref_hash := _hash(_mapping(row.get("span_ref")).get("span_ref_hash")))
            is not None
        },
        "witness_canonical_span_hashes": {
            witness_id: span_hash
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (span_hash := _hash(_mapping(row.get("span_ref")).get("canonical_span_hash")))
            is not None
        },
        "witness_packed_span_hashes": {
            witness_id: span_hash
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (span_hash := _hash(_mapping(row.get("span_ref")).get("packed_span_hash")))
            is not None
        },
        "witness_canonical_starts": {
            witness_id: start
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (start := _integer(_mapping(row.get("span_ref")).get("canonical_start")))
            is not None
        },
        "witness_canonical_ends": {
            witness_id: end
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (end := _integer(_mapping(row.get("span_ref")).get("canonical_end"), minimum=1))
            is not None
        },
        "witness_packed_starts": {
            witness_id: start
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (start := _integer(_mapping(row.get("span_ref")).get("packed_start"))) is not None
        },
        "witness_packed_ends": {
            witness_id: end
            for row in witness_rows
            if (witness_id := _token(row.get("witness_id"))) is not None
            and (end := _integer(_mapping(row.get("span_ref")).get("packed_end"), minimum=1))
            is not None
        },
        "omission_codes": omission_codes,
    }
    for key in (
        "schema_version",
        "catalog_id",
        "question_contract_id",
        "universe_status",
        "builder_version",
    ):
        _set_if(receipt, key, _token(receipt_source.get(key)))
    for key in (
        "catalog_hash",
        "question_contract_hash",
        "context_pack_hash",
    ):
        _set_if(receipt, key, _hash(receipt_source.get(key)))
    _set_if(
        receipt,
        "authorized_scope_hash",
        _scope_hash(receipt_source.get("authorized_scope_hash")),
    )
    source_closure_rows = _items(catalog_container.get("source_enumeration_closure_receipts"))
    receipt["source_enumeration_closure_receipts"] = [
        projected
        for row in source_closure_rows
        if (projected := _source_enumeration_closure_receipt(row))
    ]

    packed_blocks = _items(_mapping(context_pack).get("packed_blocks"))
    receipt["context_representation_count"] = len(packed_blocks)
    receipt["representation_packed_hashes"] = {
        representation_id: "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        for block in packed_blocks
        if (representation_id := _token(block.get("representation_id"))) is not None
        and isinstance((text := block.get("text")), str)
    }
    receipt["representation_packed_lengths"] = {
        representation_id: len(text)
        for block in packed_blocks
        if (representation_id := _token(block.get("representation_id"))) is not None
        and isinstance((text := block.get("text")), str)
    }

    evidence_rows = [_mapping(alias.get("evidence")) for alias in _items(context_aliases)]
    receipt["evidence_canonical_hashes"] = {
        evidence_id: "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        for evidence in evidence_rows
        if (evidence_id := _token(evidence.get("evidence_id"))) is not None
        and isinstance((content := evidence.get("content")), str)
    }
    receipt["evidence_canonical_lengths"] = {
        evidence_id: len(content)
        for evidence in evidence_rows
        if (evidence_id := _token(evidence.get("evidence_id"))) is not None
        and isinstance((content := evidence.get("content")), str)
    }
    receipt["source_table_authority_summaries"] = _source_table_authority_summaries(
        source_closure_rows,
        packed_blocks=packed_blocks,
        context_aliases=context_aliases,
        witness_rows=witness_rows,
    )
    receipt["source_procedure_authority_summaries"] = _source_procedure_authority_summaries(
        source_closure_rows,
        packed_blocks=packed_blocks,
        context_aliases=context_aliases,
        witness_rows=witness_rows,
    )
    return receipt


def _plan_claim_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for key in (
        "schema_version",
        "claim_id",
        "obligation_id",
        "witness_id",
        "claim_kind",
        "rendering_rule",
        "template_id",
    ):
        _set_if(receipt, key, _token(value.get(key)))
    _set_if(receipt, "claim_hash", _hash(value.get("claim_hash")))
    _set_if(receipt, "ordinal", _integer(value.get("ordinal")))
    witness_ids = _tokens(value.get("witness_ids"))
    evidence_ids = _tokens(value.get("evidence_ids"))
    if not evidence_ids:
        evidence_ids = [
            item
            for row in _items(value.get("evidence"))
            if (item := _token(row.get("evidence_id"))) is not None
        ]
    _set_if(receipt, "witness_ids", witness_ids)
    _set_if(receipt, "evidence_ids", evidence_ids)
    return receipt


def _proposal_receipt(value: object) -> dict[str, Any]:
    proposal = _mapping(value)
    if not proposal:
        return {}
    claims = [_plan_claim_receipt(item) for item in _items(proposal.get("claims"))]
    receipt: dict[str, Any] = {
        "claim_count": len(claims),
        "claims": claims,
        "reason_codes": _codes(proposal.get("reason_codes")),
    }
    _set_if(receipt, "schema_version", _token(proposal.get("schema_version")))
    _set_if(receipt, "decision", _enum(proposal.get("decision")))
    return receipt


def _answer_plan_receipt(plan_value: object, proposal_value: object) -> dict[str, Any]:
    plan = _mapping(plan_value)
    proposal_receipt = _proposal_receipt(proposal_value)
    if not plan and not proposal_receipt:
        return {}
    claims = [_plan_claim_receipt(item) for item in _items(plan.get("claims"))]
    receipt: dict[str, Any] = {
        "claim_count": len(claims),
        "claims": claims,
        "unresolved_obligation_ids": _tokens(
            plan.get("unresolved_obligation_ids") or plan.get("missing_obligation_ids")
        ),
    }
    for key in (
        "schema_version",
        "plan_id",
        "question_contract_id",
        "witness_catalog_id",
    ):
        _set_if(receipt, key, _token(plan.get(key)))
    for key in (
        "plan_hash",
        "answer_plan_hash",
        "question_contract_hash",
        "context_pack_hash",
        "witness_catalog_hash",
    ):
        _set_if(receipt, key, _hash(plan.get(key)))
    _set_if(receipt, "decision", _enum(plan.get("decision") or plan.get("disposition")))
    _set_if(receipt, "proposal_schema_version", _token(plan.get("proposal_schema_version")))
    _set_if(receipt, "prompt_contract_version", _token(plan.get("prompt_contract_version")))
    if proposal_receipt:
        receipt["proposal"] = proposal_receipt
    return receipt


def _claim_plan_serialization(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """从低正文 receipt 精确还原 ClaimPlan 的稳定身份载荷。"""

    ordinal = _integer(value.get("ordinal"))
    claim_id = _token(value.get("claim_id"))
    claim_hash = _hash(value.get("claim_hash"))
    claim_kind = _enum(value.get("claim_kind"))
    rendering_rule = _enum(value.get("rendering_rule"))
    raw_witness_ids = value.get("witness_ids", [])
    if not isinstance(raw_witness_ids, Sequence) or isinstance(
        raw_witness_ids, (str, bytes, bytearray)
    ):
        return None
    witness_ids = _tokens(raw_witness_ids, limit=16)
    if (
        value.get("schema_version") != "claim-plan.v1"
        or ordinal is None
        or claim_id != f"clm_{ordinal + 1}"
        or claim_hash is None
        or claim_kind is None
        or rendering_rule is None
        or len(witness_ids) != len(raw_witness_ids)
    ):
        return None
    obligation_id = value.get("obligation_id")
    template_id = value.get("template_id")
    if obligation_id is not None and _token(obligation_id) != obligation_id:
        return None
    if template_id is not None and _token(template_id) != template_id:
        return None
    serialized = {
        "schema_version": "claim-plan.v1",
        "claim_id": claim_id,
        "claim_hash": claim_hash,
        "obligation_id": obligation_id,
        "witness_ids": witness_ids,
        "claim_kind": claim_kind,
        "rendering_rule": rendering_rule,
        "template_id": template_id,
        "ordinal": ordinal,
    }
    payload = {key: item for key, item in serialized.items() if key != "claim_hash"}
    if claim_hash != _stable_mapping_hash(payload):
        return None
    return serialized


def _answer_plan_identity_is_valid(value: Mapping[str, Any]) -> bool:
    """不信任 plan_id/plan_hash，按投影 claims 重放 AnswerPlan 身份。"""

    claims = [_claim_plan_serialization(item) for item in _items(value.get("claims"))]
    claim_count = _integer(value.get("claim_count"))
    unresolved = _tokens(value.get("unresolved_obligation_ids"), limit=32)
    if (
        value.get("schema_version") != "answer-plan.v1"
        or claim_count is None
        or claim_count != len(claims)
        or any(item is None for item in claims)
        or [item.get("ordinal") for item in claims if item is not None] != list(range(len(claims)))
    ):
        return False
    identity_tokens = {
        key: _token(value.get(key)) for key in ("question_contract_id", "witness_catalog_id")
    }
    identity_hashes = {
        key: _hash(value.get(key))
        for key in (
            "question_contract_hash",
            "context_pack_hash",
            "witness_catalog_hash",
        )
    }
    decision = _enum(value.get("decision"))
    if (
        identity_tokens["question_contract_id"] is None
        or decision is None
        or any(
            raw is not None and parsed is None
            for raw, parsed in (
                (value.get("witness_catalog_id"), identity_tokens["witness_catalog_id"]),
                (value.get("context_pack_hash"), identity_hashes["context_pack_hash"]),
                (value.get("witness_catalog_hash"), identity_hashes["witness_catalog_hash"]),
            )
        )
        or identity_hashes["question_contract_hash"] is None
    ):
        return False
    payload = {
        "schema_version": "answer-plan.v1",
        "question_contract_id": identity_tokens["question_contract_id"],
        "question_contract_hash": identity_hashes["question_contract_hash"],
        "context_pack_hash": identity_hashes["context_pack_hash"],
        "witness_catalog_id": identity_tokens["witness_catalog_id"],
        "witness_catalog_hash": identity_hashes["witness_catalog_hash"],
        "decision": decision,
        "claims": [item for item in claims if item is not None],
        "unresolved_obligation_ids": unresolved,
        "proposal_schema_version": value.get("proposal_schema_version"),
        "prompt_contract_version": value.get("prompt_contract_version"),
    }
    return _stable_identity_matches(
        value,
        prefix="apn_",
        identity_key="plan_id",
        hash_key="plan_hash",
        payload=payload,
    )


def _complete_witness_catalog_identity_is_valid(value: Mapping[str, Any]) -> bool:
    """对 strict closure 所依赖的 complete catalog 独立重放稳定身份。"""

    raw_witness_ids = value.get("witness_ids")
    if not isinstance(raw_witness_ids, Sequence) or isinstance(
        raw_witness_ids, (str, bytes, bytearray)
    ):
        return False
    witness_ids = _tokens(raw_witness_ids)
    witness_count = _integer(value.get("witness_count"))
    if (
        value.get("schema_version") != "witness-catalog.v1"
        or value.get("universe_status") != "complete"
        or value.get("omission_codes") != []
        or witness_count != len(witness_ids)
        or len(witness_ids) != len(raw_witness_ids)
    ):
        return False
    payload = {
        "schema_version": "witness-catalog.v1",
        "question_contract_id": value.get("question_contract_id"),
        "question_contract_hash": value.get("question_contract_hash"),
        "context_pack_hash": value.get("context_pack_hash"),
        "authorized_scope_hash": value.get("authorized_scope_hash"),
        "witness_ids": witness_ids,
        "witness_count": witness_count,
        "universe_status": "complete",
        "omissions": [],
        "builder_version": value.get("builder_version"),
    }
    if (
        _token(payload["question_contract_id"]) != payload["question_contract_id"]
        or _hash(payload["question_contract_hash"]) != payload["question_contract_hash"]
        or _hash(payload["context_pack_hash"]) != payload["context_pack_hash"]
        or _scope_hash(payload["authorized_scope_hash"]) != payload["authorized_scope_hash"]
        or _token(payload["builder_version"]) != payload["builder_version"]
    ):
        return False
    return _stable_identity_matches(
        value,
        prefix="wcat_",
        identity_key="catalog_id",
        hash_key="catalog_hash",
        payload=payload,
    )


def _review_claim_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for key in ("claim_id", "obligation_id", "witness_id", "rendering_rule"):
        _set_if(receipt, key, _token(value.get(key)))
    for key in ("status", "support", "relevance", "decision"):
        _set_if(receipt, key, _enum(value.get(key)))
    _set_if(receipt, "witness_ids", _tokens(value.get("witness_ids")))
    _set_if(receipt, "reason_codes", _codes(value.get("reason_codes")))
    _set_if(receipt, "failure_codes", _codes(value.get("failure_codes")))
    return receipt


def _answer_review_receipt(value: object, verification_value: object) -> dict[str, Any]:
    review = _mapping(value)
    if not review:
        return {}
    claims = [_review_claim_receipt(item) for item in _items(review.get("claims"))]
    obligations = [_review_claim_receipt(item) for item in _items(review.get("obligations"))]
    receipt: dict[str, Any] = {
        "claim_count": len(claims),
        "obligation_count": len(obligations),
        "claims": claims,
        "obligations": obligations,
        "reason_codes": _codes(review.get("reason_codes")),
        "failure_codes": _codes(review.get("failure_codes")),
        "closed_obligation_ids": _tokens(review.get("closed_obligation_ids")),
        "proposal_hash": _stable_mapping_hash(review),
    }
    for key in ("schema_version", "review_id"):
        _set_if(receipt, key, _token(review.get(key)))
    for key in ("review_hash", "reviewer_response_hash"):
        _set_if(receipt, key, _hash(review.get(key)))
    _set_if(receipt, "decision", _enum(review.get("decision")))
    verification = _mapping(verification_value)
    if verification:
        verification_receipt: dict[str, Any] = {}
        for key in (
            "schema_version",
            "status",
            "semantic_review",
            "plan_id",
            "catalog_id",
        ):
            _set_if(verification_receipt, key, _token(verification.get(key)))
        _set_if(verification_receipt, "plan_hash", _hash(verification.get("plan_hash")))
        _set_if(
            verification_receipt,
            "reviewed_claim_count",
            _integer(verification.get("reviewed_claim_count")),
        )
        _set_if(
            verification_receipt,
            "reason_codes",
            _codes(verification.get("reason_codes")),
        )
        _set_if(
            verification_receipt,
            "closed_obligation_ids",
            _tokens(verification.get("closed_obligation_ids")),
        )
        receipt["verification"] = verification_receipt
    return receipt


def _answer_review_call_receipt(
    prompt_receipts_value: object,
    semantic_closure_value: object,
) -> dict[str, Any]:
    semantic = _mapping(semantic_closure_value)
    logical_call_id = _token(semantic.get("review_logical_call_id"))
    if logical_call_id is None or logical_call_id.startswith("lrv_"):
        return {}
    matching = [
        item
        for item in _items(prompt_receipts_value)
        if _token(item.get("logical_call_id")) == logical_call_id
    ]
    if len(matching) != 1:
        return {}
    source = matching[0]
    receipt: dict[str, Any] = {}
    for key in (
        "logical_call_id",
        "scene",
        "prompt_version",
        "model_id",
        "status",
    ):
        _set_if(receipt, key, _token(source.get(key)))
    _set_if(receipt, "prompt_hash", _hash(source.get("prompt_sha256")))
    _set_if(receipt, "response_hash", _hash(source.get("response_sha256")))
    parsed = _mapping(source.get("parsed_proposal"))
    if parsed:
        receipt["parsed_proposal_hash"] = _stable_mapping_hash(parsed)
    return receipt


def _binding_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for key in ("witness_id", "evidence_id"):
        _set_if(receipt, key, _token(value.get(key)))
    for key in (
        "witness_hash",
        "quote_hash",
        "source_content_hash",
        "span_hash",
    ):
        _set_if(receipt, key, _hash(value.get(key)))
    for key in ("quote_start", "quote_end", "start", "end"):
        _set_if(receipt, key, _integer(value.get(key)))
    _set_if(receipt, "validation", _enum(value.get("validation") or value.get("status")))
    _set_if(
        receipt,
        "reason_codes",
        _codes(value.get("reason_codes") or value.get("failure_codes")),
    )
    return receipt


def _verifier_receipt(value: object) -> dict[str, Any]:
    verifier = _mapping(value)
    if not verifier:
        return {}
    receipt: dict[str, Any] = {}
    _set_if(receipt, "verifier_id", _token(verifier.get("verifier_id")))
    _set_if(receipt, "verifier_version", _token(verifier.get("verifier_version")))
    return receipt


def _failure_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for key in ("stage", "code", "subject_id", "retryability"):
        _set_if(receipt, key, _token(value.get(key)))
    _set_if(receipt, "diagnostic_hash", _hash(value.get("diagnostic_hash")))
    return receipt


def _answer_span_receipt(value: object) -> dict[str, Any]:
    span = _mapping(value)
    if not span:
        return {}
    receipt: dict[str, Any] = {}
    for key in ("start", "end", "start_char", "end_char"):
        _set_if(receipt, key, _integer(span.get(key)))
    for key in ("span_hash", "text_hash", "rendered_text_hash"):
        _set_if(receipt, key, _hash(span.get(key)))
    return receipt


def _claim_proof_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    bindings = [
        _binding_receipt(item)
        for item in _items(
            value.get("bindings") or value.get("evidence_bindings") or value.get("witness_bindings")
        )
    ]
    receipt: dict[str, Any] = {"bindings": bindings}
    for key in (
        "schema_version",
        "claim_proof_id",
        "claim_id",
        "obligation_id",
        "method",
        "relation",
    ):
        _set_if(receipt, key, _token(value.get(key)))
    for key in (
        "claim_proof_hash",
        "claim_hash",
        "rendered_text_hash",
    ):
        _set_if(receipt, key, _hash(value.get(key)))
    for key in ("status", "support", "relevance"):
        _set_if(receipt, key, _enum(value.get(key)))
    _set_if(receipt, "witness_ids", _tokens(value.get("witness_ids")))
    _set_if(receipt, "answer_span", _answer_span_receipt(value.get("answer_span")))
    _set_if(receipt, "verifier", _verifier_receipt(value.get("verifier")))
    _set_if(
        receipt,
        "failure_codes",
        _codes(value.get("failure_codes") or value.get("reason_codes")),
    )
    return receipt


def _obligation_proof_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for key in (
        "schema_version",
        "obligation_proof_id",
        "obligation_id",
        "outcome",
        "catalog_id",
        "catalog_universe_status",
        "closure",
        "required_output_form",
        "closure_status",
    ):
        _set_if(receipt, key, _token(value.get(key)))
    for key in ("obligation_proof_hash", "catalog_hash"):
        _set_if(receipt, key, _hash(value.get(key)))
    _set_if(receipt, "status", _enum(value.get("status")))
    _set_if(receipt, "completeness", _enum(value.get("completeness")))
    _set_if(
        receipt,
        "expected_item_count",
        _integer(value.get("expected_item_count"), minimum=1),
    )
    _set_if(receipt, "proven_item_count", _integer(value.get("proven_item_count")))
    proven_items: list[dict[str, Any]] = []
    for raw in _items(value.get("proven_items")):
        item: dict[str, Any] = {}
        for key in (
            "schema_version",
            "item_ref_id",
            "obligation_id",
            "claim_id",
            "witness_id",
            "segmentation_method",
        ):
            _set_if(item, key, _token(raw.get(key)))
        for key in (
            "item_ref_hash",
            "witness_hash",
            "raw_span_hash",
            "normalized_item_hash",
        ):
            _set_if(item, key, _hash(raw.get(key)))
        _set_if(item, "relative_start", _integer(raw.get("relative_start")))
        _set_if(item, "relative_end", _integer(raw.get("relative_end"), minimum=1))
        proven_items.append(item)
    receipt["proven_items"] = proven_items
    _set_if(receipt, "verifier", _verifier_receipt(value.get("verifier")))
    _set_if(
        receipt,
        "supporting_claim_ids",
        _tokens(value.get("supporting_claim_ids") or value.get("claim_ids")),
    )
    _set_if(
        receipt,
        "contradiction_witness_ids",
        _tokens(value.get("contradiction_witness_ids")),
    )
    _set_if(
        receipt,
        "failure_codes",
        _codes(value.get("failure_codes") or value.get("reason_codes")),
    )
    source_closure_value = value.get("source_enumeration_closure_receipt")
    if source_closure_value is not None:
        source_closure = _source_enumeration_closure_receipt(source_closure_value)
        receipt["source_enumeration_closure_receipt"] = source_closure or {
            "schema_version": "invalid-source-enumeration-closure-receipt.v1"
        }
    return receipt


def _source_inline_action_chain_closure_receipt(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """以固定 61 字段投影 inline action-chain receipt。"""

    if set(source) != _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_FIELDS:
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
    token_arrays = {
        key: _exact_tokens(
            source.get(key),
            limit=(256 if key.startswith("catalog_") else 1),
        )
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_ARRAY_FIELDS
    }
    hash_arrays = {
        key: _exact_hashes(
            source.get(key),
            limit=(
                256
                if key.startswith("catalog_")
                else 128
                if key == "anchor_match_key_hashes"
                else 16
                if "action" in key
                else 1
            ),
        )
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_ARRAY_FIELDS
    }
    integer_arrays = {
        key: _exact_integers(
            source.get(key),
            limit=(5 if key == "selection_score" else 16 if "action" in key else 1),
        )
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_ARRAY_FIELDS
    }
    optional_hashes = {
        key: None if source.get(key) is None else _hash(source.get(key))
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_HASH_FIELDS
    }
    optional_integers = {
        key: None if source.get(key) is None else _integer(source.get(key))
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_OPTIONAL_INTEGER_FIELDS
    }
    required_tokens_valid = all(
        _token(source.get(key)) == source.get(key)
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_FIELDS
    )
    required_hashes_valid = all(
        _hash(source.get(key)) == source.get(key)
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_FIELDS
    )
    required_integers_valid = all(
        _integer(source.get(key)) == source.get(key)
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_FIELDS
    )
    terminal_fields = (
        optional_integers["canonical_terminal_result_start"],
        optional_integers["canonical_terminal_result_end"],
        optional_integers["packed_terminal_result_start"],
        optional_integers["packed_terminal_result_end"],
        optional_hashes["canonical_terminal_result_hash"],
        optional_hashes["packed_terminal_result_hash"],
    )
    if (
        source.get("schema_version") != "query-anchored-inline-action-chain-closure-receipt.v1"
        or source.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or source.get("anchor_field") != "information_need"
        or source.get("witness_atom_kind") not in {"sentence", "list_item"}
        or not required_tokens_valid
        or not required_hashes_valid
        or not required_integers_valid
        or any(item is None for item in token_arrays.values())
        or any(item is None for item in hash_arrays.values())
        or any(item is None for item in integer_arrays.values())
        or any(
            source.get(key) is not None and item is None for key, item in optional_hashes.items()
        )
        or any(
            source.get(key) is not None and item is None for key, item in optional_integers.items()
        )
        or (
            any(item is None for item in terminal_fields)
            and not all(item is None for item in terminal_fields)
        )
    ):
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}

    witness_ids = token_arrays["witness_ids"] or []
    witness_hashes = hash_arrays["witness_hashes"] or []
    span_ref_hashes = hash_arrays["span_ref_hashes"] or []
    witness_ordinals = integer_arrays["witness_ordinals"] or []
    canonical_starts = integer_arrays["canonical_action_starts"] or []
    canonical_ends = integer_arrays["canonical_action_ends"] or []
    packed_starts = integer_arrays["packed_action_starts"] or []
    packed_ends = integer_arrays["packed_action_ends"] or []
    action_ordinals = integer_arrays["action_ordinals"] or []
    selection_score = integer_arrays["selection_score"] or []
    canonical_action_hashes = hash_arrays["canonical_action_hashes"] or []
    packed_action_hashes = hash_arrays["packed_action_hashes"] or []
    anchor_hashes = hash_arrays["anchor_match_key_hashes"] or []
    catalog_evidence_ids = token_arrays["catalog_evidence_ids"] or []
    catalog_representation_ids = token_arrays["catalog_representation_ids"] or []
    catalog_evidence_hashes = hash_arrays["catalog_evidence_content_hashes"] or []
    catalog_packed_hashes = hash_arrays["catalog_packed_text_hashes"] or []
    action_count = int(source["action_count"])
    representation_count = int(source["catalog_representation_count"])
    selected_index = int(source["selected_authority_index"])
    action_arrays = (
        canonical_starts,
        canonical_ends,
        packed_starts,
        packed_ends,
        canonical_action_hashes,
        packed_action_hashes,
        action_ordinals,
    )
    authority_arrays = (
        catalog_evidence_ids,
        catalog_representation_ids,
        catalog_evidence_hashes,
        catalog_packed_hashes,
    )
    if (
        len(witness_ids) != 1
        or len(witness_hashes) != 1
        or len(span_ref_hashes) != 1
        or len(witness_ordinals) != 1
        or not 2 <= action_count <= 16
        or any(len(item) != action_count for item in action_arrays)
        or action_ordinals != list(range(1, action_count + 1))
        or len(selection_score) != 5
        or selection_score[0] not in {0, 1}
        or selection_score[1] not in {0, 1}
        or not anchor_hashes
        or len(anchor_hashes) != len(set(anchor_hashes))
        or not 1 <= int(source["eligible_candidate_count"]) <= 4096
        or source.get("top_score_candidate_count") != 1
        or not 1 <= representation_count <= 256
        or any(len(item) != representation_count for item in authority_arrays)
        or len(set(catalog_evidence_ids)) != representation_count
        or len(set(catalog_representation_ids)) != representation_count
        or not 0 <= selected_index < representation_count
        or catalog_evidence_ids[selected_index] != source.get("evidence_id")
        or catalog_representation_ids[selected_index] != source.get("representation_id")
        or catalog_evidence_hashes[selected_index] != source.get("evidence_content_hash")
    ):
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}

    canonical_witness_start = int(source["canonical_witness_start"])
    canonical_witness_end = int(source["canonical_witness_end"])
    packed_witness_start = int(source["packed_witness_start"])
    packed_witness_end = int(source["packed_witness_end"])
    if (
        not 0
        <= int(source["canonical_title_start"])
        < int(source["canonical_title_end"])
        <= canonical_witness_start
        < canonical_witness_end
        or not 0
        <= int(source["packed_title_start"])
        < int(source["packed_title_end"])
        <= packed_witness_start
        < packed_witness_end
        or any(
            not canonical_witness_start <= start < end <= canonical_witness_end
            for start, end in zip(canonical_starts, canonical_ends, strict=True)
        )
        or any(
            not packed_witness_start <= start < end <= packed_witness_end
            for start, end in zip(packed_starts, packed_ends, strict=True)
        )
        or any(
            left_end > right_start
            for left_end, right_start in zip(canonical_ends, canonical_starts[1:])
        )
        or any(
            left_end > right_start for left_end, right_start in zip(packed_ends, packed_starts[1:])
        )
    ):
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
    if terminal_fields[0] is not None and not (
        canonical_ends[-1]
        <= int(terminal_fields[0])
        < int(terminal_fields[1])
        <= canonical_witness_end
        and packed_ends[-1]
        <= int(terminal_fields[2])
        < int(terminal_fields[3])
        <= packed_witness_end
    ):
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}

    receipt: dict[str, Any] = {
        key: source[key]
        for key in (
            _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_TOKEN_FIELDS
            | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_HASH_FIELDS
            | _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_INTEGER_FIELDS
        )
    }
    receipt.update(optional_hashes)
    receipt.update(optional_integers)
    receipt.update({key: item for key, item in token_arrays.items() if item is not None})
    receipt.update({key: item for key, item in hash_arrays.items() if item is not None})
    receipt.update({key: item for key, item in integer_arrays.items() if item is not None})
    if set(receipt) != _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_FIELDS:
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
    return receipt


def _source_enumeration_closure_receipt(value: object) -> dict[str, Any]:
    """投影版本化来源闭包身份与边界，不透传查询或 Evidence 正文。"""

    source = _mapping(value)
    if not source:
        return {}
    if source.get("schema_version") == "query-anchored-inline-action-chain-closure-receipt.v1":
        return _source_inline_action_chain_closure_receipt(source)
    if source.get("schema_version") == "query-anchored-scoped-fact-set-closure-receipt.v1":
        if set(source) != _SOURCE_SCOPED_FACT_SET_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        token_arrays = {
            key: _exact_tokens(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_ARRAY_FIELDS
        }
        hash_arrays = {
            key: _exact_hashes(
                source.get(key),
                limit=(
                    256
                    if key.startswith("catalog_")
                    else 128
                    if key == "flattened_semantic_hashes"
                    else 64
                ),
            )
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_ARRAY_FIELDS
        }
        integer_arrays = {
            key: _exact_integers(source.get(key), limit=64)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_ARRAY_FIELDS
        }
        optional_hashes = {
            key: None if source.get(key) is None else _hash(source.get(key))
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_HASH_FIELDS
        }
        optional_integers = {
            key: None if source.get(key) is None else _integer(source.get(key))
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_INTEGER_FIELDS
        }
        atom_kinds = token_arrays.get("member_atom_kinds")
        segmentation_methods = token_arrays.get("segmentation_methods")
        if (
            source.get("mode")
            not in {
                "closed_inline_coordination",
                "scoped_topic_predicate_set",
                "scoped_bullet_lead_clause_set",
            }
            or source.get("relation")
            not in {
                "inclusion",
                "required_information",
                "directive_compliance",
                "applicability_restriction",
                "safety_instruction",
                "ownership_registration",
            }
            or source.get("anchor_field") not in {"source_query", "information_need"}
            or any(item is None for item in token_arrays.values())
            or any(item is None for item in hash_arrays.values())
            or any(item is None for item in integer_arrays.values())
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_hashes.items()
            )
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_integers.items()
            )
            or atom_kinds is None
            or any(item not in {"sentence", "list_item"} for item in atom_kinds)
            or segmentation_methods is None
            or any(
                item
                not in {
                    "open_enumeration_witness_atom.v1",
                    "top_level_list_entry.v1",
                }
                for item in segmentation_methods
            )
        ):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        receipt: dict[str, Any] = {}
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_FIELDS:
            _set_if(receipt, key, _token(source.get(key)))
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_FIELDS:
            _set_if(receipt, key, _hash(source.get(key)))
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_FIELDS:
            _set_if(receipt, key, _integer(source.get(key)))
        receipt.update(optional_hashes)
        receipt.update(optional_integers)
        receipt.update({key: item for key, item in token_arrays.items() if item is not None})
        receipt.update({key: item for key, item in hash_arrays.items() if item is not None})
        receipt.update({key: item for key, item in integer_arrays.items() if item is not None})
        if set(receipt) != _SOURCE_SCOPED_FACT_SET_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        return receipt
    if source.get("schema_version") == "query-anchored-scoped-bullet-list-closure-receipt.v1":
        if set(source) != _SOURCE_SCOPED_BULLET_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        marker_style = source.get("marker_style")
        token_arrays = {
            key: _exact_tokens(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_ARRAY_FIELDS
        }
        hash_arrays = {
            key: _exact_hashes(
                source.get(key),
                limit=(
                    256 if key.startswith("catalog_") or key == "anchor_match_key_hashes" else 64
                ),
            )
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_HASH_ARRAY_FIELDS
        }
        integer_arrays = {
            key: _exact_integers(source.get(key), limit=64)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_ARRAY_FIELDS
        }
        optional_hashes = {
            key: None if source.get(key) is None else _hash(source.get(key))
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_HASH_FIELDS
        }
        optional_integers = {
            key: None if source.get(key) is None else _integer(source.get(key))
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_INTEGER_FIELDS
        }
        if (
            not isinstance(marker_style, str)
            or re.fullmatch(r"bullet:[-*•+]", marker_style) is None
            or any(item is None for item in token_arrays.values())
            or any(item is None for item in hash_arrays.values())
            or any(item is None for item in integer_arrays.values())
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_hashes.items()
            )
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_integers.items()
            )
        ):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        receipt: dict[str, Any] = {"marker_style": marker_style}
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_FIELDS - {"marker_style"}:
            _set_if(receipt, key, _token(source.get(key)))
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_HASH_FIELDS:
            _set_if(receipt, key, _hash(source.get(key)))
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_FIELDS:
            _set_if(receipt, key, _integer(source.get(key)))
        receipt.update(optional_hashes)
        receipt.update(optional_integers)
        receipt.update({key: item for key, item in token_arrays.items() if item is not None})
        receipt.update({key: item for key, item in hash_arrays.items() if item is not None})
        receipt.update({key: item for key, item in integer_arrays.items() if item is not None})
        if set(receipt) != _SOURCE_SCOPED_BULLET_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        return receipt
    if source.get("schema_version") == "query-anchored-scoped-table-selection-closure-receipt.v2":
        if set(source) != _SOURCE_SCOPED_TABLE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        token_arrays = {
            key: _exact_tokens(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_ARRAY_FIELDS
        }
        hash_arrays = {
            key: _exact_hashes(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_HASH_ARRAY_FIELDS
        }
        integer_arrays = {
            key: _exact_integers(source.get(key))
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_ARRAY_FIELDS
        }
        if (
            any(item is None for item in token_arrays.values())
            or any(item is None for item in hash_arrays.values())
            or any(item is None for item in integer_arrays.values())
            or source.get("table_parse_mode")
            not in {"standard", "ocr_header_data_before_separator"}
        ):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        receipt: dict[str, Any] = {}
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_FIELDS:
            _set_if(receipt, key, _token(source.get(key)))
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_HASH_FIELDS:
            _set_if(receipt, key, _hash(source.get(key)))
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_FIELDS:
            _set_if(receipt, key, _integer(source.get(key)))
        receipt.update({key: item for key, item in token_arrays.items() if item is not None})
        receipt.update({key: item for key, item in hash_arrays.items() if item is not None})
        receipt.update({key: item for key, item in integer_arrays.items() if item is not None})
        if set(receipt) != _SOURCE_SCOPED_TABLE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        return receipt
    if source.get("schema_version") == "query-anchored-scoped-procedure-list-closure-receipt.v4":
        if set(source) != _SOURCE_SCOPED_PROCEDURE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        marker_style = source.get("marker_style")
        token_arrays = {
            key: _exact_tokens(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
        }
        hash_arrays = {
            key: _exact_hashes(
                source.get(key),
                limit=256 if key.startswith("catalog_") else 64,
            )
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
        }
        integer_arrays = {
            key: _exact_integers(source.get(key))
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
        }
        optional_tokens = {
            key: None if source.get(key) is None else _token(source.get(key))
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_TOKEN_FIELDS
        }
        optional_hashes = {
            key: None if source.get(key) is None else _hash(source.get(key))
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_HASH_FIELDS
        }
        optional_integers = {
            key: None if source.get(key) is None else _integer(source.get(key))
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_INTEGER_FIELDS
        }
        if (
            not isinstance(marker_style, str)
            or re.fullmatch(r"(?:arabic|chinese):[.)．、]", marker_style) is None
            or any(item is None for item in token_arrays.values())
            or any(item is None for item in hash_arrays.values())
            or any(item is None for item in integer_arrays.values())
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_tokens.items()
            )
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_hashes.items()
            )
            or any(
                source.get(key) is not None and item is None
                for key, item in optional_integers.items()
            )
        ):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        selection_kind = source.get("selection_kind")
        if selection_kind not in {
            "complete_sequence",
            "query_selected_continuation",
            "query_selected_branch",
            "query_selected_prefix",
        }:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        item_ordinals = integer_arrays.get("item_ordinals") or []
        prefix_fields = (
            optional_hashes.get("prefix_trigger_hash"),
            optional_integers.get("excluded_successor_ordinal"),
            optional_hashes.get("excluded_successor_hash"),
            optional_tokens.get("boundary_rule_version"),
        )
        if selection_kind == "query_selected_prefix":
            if (
                any(item is None for item in prefix_fields)
                or not item_ordinals
                or item_ordinals[0] != 1
                or optional_integers.get("excluded_successor_ordinal") != item_ordinals[-1] + 1
                or optional_tokens.get("boundary_rule_version")
                != "explicit-query-before-successor.v1"
            ):
                return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        elif any(item is not None for item in prefix_fields):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        receipt: dict[str, Any] = {
            "marker_style": marker_style,
            "selection_kind": selection_kind,
        }
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_FIELDS - {
            "marker_style",
            "selection_kind",
        }:
            _set_if(receipt, key, _token(source.get(key)))
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_FIELDS:
            _set_if(receipt, key, _hash(source.get(key)))
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_FIELDS:
            _set_if(receipt, key, _integer(source.get(key)))
        receipt.update(optional_tokens)
        receipt.update(optional_hashes)
        receipt.update(optional_integers)
        receipt.update({key: item for key, item in token_arrays.items() if item is not None})
        receipt.update({key: item for key, item in hash_arrays.items() if item is not None})
        receipt.update({key: item for key, item in integer_arrays.items() if item is not None})
        if set(receipt) != _SOURCE_SCOPED_PROCEDURE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        return receipt
    if source.get("schema_version") == "query-anchored-procedure-list-closure-receipt.v1":
        if set(source) != _SOURCE_PROCEDURE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        marker_style = source.get("marker_style")
        token_arrays = {
            key: _exact_tokens(source.get(key))
            for key in _SOURCE_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
        }
        hash_arrays = {
            key: _exact_hashes(source.get(key))
            for key in _SOURCE_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
        }
        integer_arrays = {
            key: _exact_integers(source.get(key))
            for key in _SOURCE_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
        }
        if (
            not isinstance(marker_style, str)
            or re.fullmatch(r"(?:arabic|chinese):[.)．、]", marker_style) is None
            or any(value is None for value in token_arrays.values())
            or any(value is None for value in hash_arrays.values())
            or any(value is None for value in integer_arrays.values())
        ):
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        receipt: dict[str, Any] = {"marker_style": marker_style}
        for key in _SOURCE_PROCEDURE_CLOSURE_TOKEN_FIELDS - {"marker_style"}:
            _set_if(receipt, key, _token(source.get(key)))
        for key in _SOURCE_PROCEDURE_CLOSURE_HASH_FIELDS:
            _set_if(receipt, key, _hash(source.get(key)))
        for key in _SOURCE_PROCEDURE_CLOSURE_INTEGER_FIELDS:
            _set_if(receipt, key, _integer(source.get(key)))
        receipt.update({key: value for key, value in token_arrays.items() if value is not None})
        receipt.update({key: value for key, value in hash_arrays.items() if value is not None})
        receipt.update({key: value for key, value in integer_arrays.items() if value is not None})
        if set(receipt) != _SOURCE_PROCEDURE_CLOSURE_FIELDS:
            return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
        return receipt
    if set(source) != _SOURCE_ENUMERATION_CLOSURE_FIELDS:
        return {"schema_version": "invalid-source-enumeration-closure-receipt.v1"}
    receipt: dict[str, Any] = {}
    for key in _SOURCE_ENUMERATION_CLOSURE_TOKEN_FIELDS:
        _set_if(receipt, key, _token(source.get(key)))
    for key in _SOURCE_ENUMERATION_CLOSURE_HASH_FIELDS:
        _set_if(receipt, key, _hash(source.get(key)))
    for key in _SOURCE_ENUMERATION_CLOSURE_INTEGER_FIELDS:
        _set_if(receipt, key, _integer(source.get(key)))
    return receipt


def _semantic_review_closure_receipt(value: object) -> dict[str, Any]:
    source = _mapping(value)
    if not source:
        return {}
    receipt: dict[str, Any] = {
        "closed_obligation_ids": _tokens(source.get("closed_obligation_ids")),
        "reviewed_closed_obligation_ids": _tokens(source.get("reviewed_closed_obligation_ids")),
    }
    for key in (
        "schema_version",
        "receipt_id",
        "answer_plan_id",
        "reviewed_answer_plan_id",
        "catalog_id",
        "decision",
        "reason_code",
        "review_prompt_version",
        "model_id",
        "verifier_version",
        "review_logical_call_id",
    ):
        _set_if(receipt, key, _token(source.get(key)))
    for key in (
        "receipt_hash",
        "answer_plan_hash",
        "reviewed_answer_plan_hash",
        "catalog_hash",
        "review_prompt_hash",
        "review_response_hash",
        "review_proposal_hash",
    ):
        _set_if(receipt, key, _hash(source.get(key)))
    return receipt


def _answer_proof_receipt(value: object) -> dict[str, Any]:
    proof = _mapping(value)
    if not proof:
        return {}
    claims = [
        _claim_proof_receipt(item)
        for item in _items(proof.get("claim_proofs") or proof.get("claims"))
    ]
    obligations = [
        _obligation_proof_receipt(item)
        for item in _items(proof.get("obligation_proofs") or proof.get("obligations"))
    ]
    receipt: dict[str, Any] = {
        "strict_eligible": proof.get("strict_eligible")
        if isinstance(proof.get("strict_eligible"), bool)
        else None,
        "claim_proofs": claims,
        "obligation_proofs": obligations,
        "semantic_review_closure": _semantic_review_closure_receipt(
            proof.get("semantic_review_closure")
        ),
    }
    for key in (
        "schema_version",
        "proof_id",
        "answer_plan_id",
        "question_contract_id",
        "witness_catalog_id",
        "answer_id",
        "decision",
        "disposition",
        "overall_status",
        "status",
        "grounding_status",
        "relevance_status",
        "completeness_status",
        "completeness",
        "verifier_version",
        "created_at",
    ):
        _set_if(receipt, key, _enum(proof.get(key)))
    for key in (
        "proof_hash",
        "answer_plan_hash",
        "question_contract_hash",
        "context_pack_hash",
        "witness_catalog_hash",
        "answer_hash",
        "reviewer_response_hash",
    ):
        _set_if(receipt, key, _hash(proof.get(key)))
    verifiers = [
        item for raw in _items(proof.get("verifier_versions")) if (item := _verifier_receipt(raw))
    ]
    _set_if(receipt, "verifier_versions", verifiers)
    failures = [item for raw in _items(proof.get("failures")) if (item := _failure_receipt(raw))]
    _set_if(receipt, "failures", failures)
    _set_if(
        receipt,
        "failure_codes",
        _codes(proof.get("failure_codes") or proof.get("reason_codes"))
        or [str(item["code"]) for item in failures if item.get("code")],
    )
    return receipt


def project_proof_observation(state: Mapping[str, Any]) -> dict[str, Any]:
    """从 Query state 建立 v3 安全回执，不透传 Witness 或 Prompt 正文。"""

    answer_proof = _answer_proof_receipt(state.get("answer_proof"))
    semantic_closure = _mapping(answer_proof.get("semantic_review_closure"))
    return {
        "answer_proof": answer_proof,
        "question_contract_receipt": _question_contract_receipt(state.get("question_contract")),
        "witness_catalog_receipt": _witness_catalog_receipt(
            state.get("witness_catalog"),
            context_pack=state.get("context_pack"),
            context_aliases=state.get("context_aliases"),
        ),
        "answer_plan_receipt": _answer_plan_receipt(
            state.get("answer_plan"),
            state.get("answer_plan_proposal"),
        ),
        "reviewed_answer_plan_receipt": _answer_plan_receipt(
            state.get("answer_reviewed_plan"),
            state.get("answer_plan_proposal"),
        ),
        "answer_plan_review_receipt": _answer_review_receipt(
            state.get("answer_plan_review"),
            state.get("answer_verification"),
        ),
        "answer_review_call_receipt": _answer_review_call_receipt(
            state.get("answer_prompt_receipts"),
            semantic_closure,
        ),
    }


def is_v3_observation(value: Mapping[str, Any] | None) -> bool:
    """仅显式 v3 才启用 proof-first；v1/v2 保持历史只读行为。"""

    return isinstance(value, Mapping) and value.get("observation_schema") == OBSERVATION_SCHEMA_V3


def _normalized_status(value: object) -> str:
    return str(value or "").strip().lower()


def _schema_matches(value: object, prefix: str) -> bool:
    schema = _token(value)
    return schema is not None and schema.startswith(prefix)


def _binding_is_proven(binding: Mapping[str, Any]) -> bool:
    return _normalized_status(binding.get("validation") or binding.get("status")) in _PROVEN


def _claim_is_system_notice(claim: Mapping[str, Any]) -> bool:
    return (
        _normalized_status(claim.get("method")) == "system_notice"
        or _normalized_status(claim.get("relation")) == "not_applicable"
        and not _tokens(claim.get("witness_ids"))
    )


def _claim_support_is_proven(claim: Mapping[str, Any]) -> bool:
    status = _normalized_status(claim.get("status"))
    if status:
        if status not in _PROVEN:
            return False
        if _claim_is_system_notice(claim):
            return True
        return _normalized_status(claim.get("relation")) in {"supports", "supported"} and bool(
            _tokens(claim.get("witness_ids"))
        )
    support = _normalized_status(claim.get("support"))
    bindings = _items(claim.get("bindings"))
    return support in _PROVEN and bool(bindings) and any(_binding_is_proven(x) for x in bindings)


def _claim_relevance_is_proven(
    claim: Mapping[str, Any],
    *,
    root_relevance: str,
) -> bool:
    if _claim_is_system_notice(claim):
        return _claim_support_is_proven(claim)
    relevance = _normalized_status(claim.get("relevance"))
    if relevance:
        return relevance in _PROVEN
    return _normalized_status(claim.get("status")) in _PROVEN and root_relevance in _PROVEN


def _obligation_is_resolved(obligation: Mapping[str, Any]) -> bool:
    completeness = _normalized_status(obligation.get("completeness"))
    if completeness:
        return completeness in _PROVEN
    status = _normalized_status(obligation.get("status"))
    outcome = _normalized_status(obligation.get("outcome"))
    return status in _PROVEN and outcome in {
        "answered",
        "evidence_absent",
        "conflicting",
        "not_applicable",
    }


def _proof_rows(
    proof: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    return (
        _items(proof.get("claim_proofs") or proof.get("claims")),
        _items(proof.get("obligation_proofs") or proof.get("obligations")),
    )


def _required_obligation_ids(
    observation: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    receipt = _mapping(observation.get("question_contract_receipt"))
    if "required_obligation_ids" in receipt:
        return tuple(_tokens(receipt.get("required_obligation_ids")))
    return tuple(
        item for row in obligations if (item := _token(row.get("obligation_id"))) is not None
    )


def _identity_mismatch(
    left: Mapping[str, Any],
    left_key: str,
    right: Mapping[str, Any],
    right_key: str,
) -> bool:
    left_value = left.get(left_key)
    right_value = right.get(right_key)
    return left_value is None or right_value is None or left_value != right_value


def _answer_hash_matches(observation: Mapping[str, Any], proof: Mapping[str, Any]) -> bool:
    expected = _hash(proof.get("answer_hash"))
    answer = observation.get("answer")
    if expected is None or not isinstance(answer, str):
        return False
    actual = "sha256:" + hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return actual == expected


def _authority_source_pair(
    authority: ProofEvaluationAuthority,
    *,
    evidence_id: str,
    representation_id: str,
) -> tuple[ProofEvidenceAuthority, ProofRepresentationAuthority] | None:
    evidences = [item for item in authority.evidences if item.evidence_id == evidence_id]
    representations = [
        item for item in authority.representations if item.representation_id == representation_id
    ]
    if len(evidences) != 1 or len(representations) != 1:
        return None
    return evidences[0], representations[0]


def _catalog_matches_proof_authority(
    catalog: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """以瞬时 raw sidecar 重放完整 catalog ledger，拒绝 public ledger 协调篡改。"""

    trusted = _trusted_proof_inputs(authority)
    if trusted is None:
        return False
    _question, trusted_catalog, witnesses = trusted
    trusted_projection = _witness_catalog_receipt(
        {
            "receipt": trusted_catalog.to_dict(),
            "witnesses": [item.to_dict() for item in witnesses],
        },
        context_pack={
            "packed_blocks": [
                {
                    "representation_id": item.representation_id,
                    "text": item.packed_text,
                }
                for item in authority.representations
            ]
        },
        context_aliases=[
            {
                "evidence": {
                    "evidence_id": item.evidence_id,
                    "content": item.canonical_text,
                }
            }
            for item in authority.evidences
        ],
    )
    trusted_fields = (
        "schema_version",
        "catalog_id",
        "question_contract_id",
        "universe_status",
        "builder_version",
        "catalog_hash",
        "question_contract_hash",
        "context_pack_hash",
        "authorized_scope_hash",
        "witness_count",
        "witness_ids",
        "witness_atom_kinds",
        "witness_hashes",
        "witness_evidence_ids",
        "witness_evidence_content_hashes",
        "witness_representation_ids",
        "witness_span_ref_hashes",
        "witness_canonical_span_hashes",
        "witness_packed_span_hashes",
        "witness_canonical_starts",
        "witness_canonical_ends",
        "witness_packed_starts",
        "witness_packed_ends",
        "context_representation_count",
        "representation_packed_hashes",
        "representation_packed_lengths",
        "evidence_canonical_hashes",
        "evidence_canonical_lengths",
    )
    return all(catalog.get(key) == trusted_projection.get(key) for key in trusted_fields)


def _trusted_proof_inputs(authority: ProofEvaluationAuthority | None):
    if (
        authority is None
        or not authority.source_ledger_complete
        or not authority.witness_ledger_complete
        or not authority.representations
    ):
        return None
    try:
        from app.domain.answering.question import QuestionContract
        from app.domain.answering.witness import EvidenceWitness, WitnessCatalogReceipt

        question_raw = json.loads(authority.question_contract_json)
        catalog_raw = json.loads(authority.witness_catalog_json)
        witness_raw = [json.loads(item) for item in authority.witness_rows_json]
        if (
            not isinstance(question_raw, Mapping)
            or not isinstance(catalog_raw, Mapping)
            or any(not isinstance(item, Mapping) for item in witness_raw)
        ):
            return None
        question = QuestionContract.from_dict(dict(question_raw))
        catalog = WitnessCatalogReceipt.from_dict(dict(catalog_raw))
        witnesses = tuple(EvidenceWitness.from_dict(dict(item)) for item in witness_raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if tuple(item.witness_id for item in witnesses) != tuple(catalog.witness_ids) or tuple(
        item.ordinal for item in witnesses
    ) != tuple(range(len(witnesses))):
        return None
    return question, catalog, witnesses


def _normalized_proven_item_hash(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return _text_hash(normalized)


def _closed_count_table_cell_items_match_proof_authority(
    obligation_proof: Mapping[str, Any],
    *,
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """主动识别 cell scope，并从 canonical/packed authority 双重重建 item。"""

    items = _items(obligation_proof.get("proven_items"))
    trusted = _trusted_proof_inputs(authority)
    if trusted is None or authority is None:
        return False
    question, _catalog, witnesses = trusted
    obligation_id = str(obligation_proof.get("obligation_id") or "")
    obligations = [item for item in question.obligations if item.obligation_id == obligation_id]
    if len(obligations) != 1:
        return False
    obligation = obligations[0]
    witness_by_id = {item.witness_id: item for item in witnesses}
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    representation_by_id = {item.representation_id: item for item in authority.representations}
    if len(evidence_by_id) != len(authority.evidences) or len(representation_by_id) != len(
        authority.representations
    ):
        return False

    canonical_texts: dict[str, str] = {}
    packed_texts: dict[str, str] = {}
    try:
        for witness in witnesses:
            span = witness.span_ref
            evidence = evidence_by_id[span.evidence_id]
            representation = representation_by_id[span.representation_id]
            if representation.evidence_id not in {None, span.evidence_id}:
                return False
            span.validate_artifacts(
                evidence_id=span.evidence_id,
                representation_id=span.representation_id,
                canonical_text=evidence.canonical_text,
                packed_text=representation.packed_text,
            )
            canonical_texts[witness.witness_id] = evidence.canonical_text[
                span.canonical_start : span.canonical_end
            ]
            packed_texts[witness.witness_id] = representation.packed_text[
                span.packed_start : span.packed_end
            ]
    except (KeyError, TypeError, ValueError):
        return False

    try:
        from app.domain.answering.item_boundary import (
            ClosedCountTableCellProjectionStatus,
            resolve_closed_count_markdown_table_cell_projection,
        )

        packed_projection = resolve_closed_count_markdown_table_cell_projection(
            question=question,
            obligation=obligation,
            witnesses=witnesses,
            witness_texts=packed_texts,
            coordinate="packed",
        )
        canonical_projection = resolve_closed_count_markdown_table_cell_projection(
            question=question,
            obligation=obligation,
            witnesses=witnesses,
            witness_texts=canonical_texts,
            coordinate="canonical",
        )
        if packed_projection.status != canonical_projection.status:
            return False
        if packed_projection.status == ClosedCountTableCellProjectionStatus.NOT_APPLICABLE:
            return all(
                item.get("segmentation_method") != "markdown_table_projected_cell_item.v1"
                for item in items
            )
        if (
            packed_projection.status != ClosedCountTableCellProjectionStatus.VALID
            or canonical_projection.status != ClosedCountTableCellProjectionStatus.VALID
            or packed_projection.witness_id != canonical_projection.witness_id
            or packed_projection.witness_id not in witness_by_id
            or len(packed_projection.item_spans) != len(canonical_projection.item_spans)
            or len(items) != len(packed_projection.item_spans)
        ):
            return False

        witness_id = packed_projection.witness_id
        if witness_id is None:
            return False
        witness = witness_by_id[witness_id]
        packed_witness_text = packed_texts[witness_id]
        canonical_witness_text = canonical_texts[witness_id]
        for item, packed_span, canonical_span in zip(
            items,
            packed_projection.item_spans,
            canonical_projection.item_spans,
            strict=True,
        ):
            if (
                item.get("segmentation_method") != "markdown_table_projected_cell_item.v1"
                or item.get("witness_id") != witness_id
                or item.get("relative_start") != packed_span[0]
                or item.get("relative_end") != packed_span[1]
            ):
                return False
            packed_item = packed_witness_text[packed_span[0] : packed_span[1]]
            canonical_item = canonical_witness_text[canonical_span[0] : canonical_span[1]]
            if (
                unicodedata.normalize("NFKC", packed_item)
                != unicodedata.normalize("NFKC", canonical_item)
                or item.get("witness_hash") != witness.witness_hash
                or item.get("raw_span_hash") != _text_hash(packed_item)
                or item.get("normalized_item_hash") != _normalized_proven_item_hash(packed_item)
            ):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _question_matches_proof_authority(
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    trusted = _trusted_proof_inputs(authority)
    if trusted is None:
        return False
    question, _catalog, _witnesses = trusted
    return dict(question_receipt) == _question_contract_receipt(question.to_dict())


def _table_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    evidence_id = str(receipt.get("evidence_id") or "")
    representation_id = str(receipt.get("representation_id") or "")
    witness_id = str(receipt.get("witness_id") or "")
    source_pair = _authority_source_pair(
        authority,
        evidence_id=evidence_id,
        representation_id=representation_id,
    )
    matching_witnesses = [item for item in witnesses if item.witness_id == witness_id]
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if source_pair is None or len(matching_witnesses) != 1 or len(obligations) != 1:
        return False
    evidence, representation = source_pair
    witness = matching_witnesses[0]
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredTableRowClosureReceipt,
        )

        typed_receipt = QueryAnchoredTableRowClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witness=witness,
            evidence_id=evidence_id,
            representation_id=representation_id,
            canonical_text=evidence.canonical_text,
            packed_text=representation.packed_text,
            catalog_texts=tuple(item.packed_text for item in authority.representations),
        )
    except (TypeError, ValueError):
        return False
    return True


def _procedure_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, all_witnesses = trusted
    evidence_id = str(receipt.get("evidence_id") or "")
    representation_id = str(receipt.get("representation_id") or "")
    source_pair = _authority_source_pair(
        authority,
        evidence_id=evidence_id,
        representation_id=representation_id,
    )
    witnesses = sorted(
        (
            item
            for item in all_witnesses
            if item.span_ref.evidence_id == evidence_id
            and item.span_ref.representation_id == representation_id
            and item.atom_kind.value == "list_item"
        ),
        key=lambda item: item.ordinal,
    )
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if source_pair is None or len(witnesses) < 2 or len(obligations) != 1:
        return False
    evidence, representation = source_pair
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredProcedureListClosureReceipt,
        )

        typed_receipt = QueryAnchoredProcedureListClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=tuple(witnesses),
            evidence_id=evidence_id,
            representation_id=representation_id,
            canonical_text=evidence.canonical_text,
            packed_text=representation.packed_text,
            catalog_texts=tuple(item.packed_text for item in authority.representations),
        )
    except (TypeError, ValueError):
        return False
    return True


def _scoped_procedure_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """从完整 raw authority 重建 scoped receipt，不信任公开摘要或自报范围。"""

    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    receipt_representation_ids = _exact_tokens(
        receipt.get("catalog_representation_ids"),
        limit=256,
    )
    receipt_evidence_ids = _exact_tokens(
        receipt.get("catalog_evidence_ids"),
        limit=256,
    )
    authority_representation_ids = [item.representation_id for item in authority.representations]
    authority_evidence_ids = [item.evidence_id for item in authority.representations]
    if (
        receipt_representation_ids != authority_representation_ids
        or receipt_evidence_ids != authority_evidence_ids
        or any(item is None for item in authority_evidence_ids)
    ):
        return False
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    if len(evidence_by_id) != len(authority.evidences):
        return False
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if len(obligations) != 1:
        return False
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredScopedProcedureListClosureReceipt,
            ScopedListAuthorityRepresentation,
        )

        raw_authorities = tuple(
            ScopedListAuthorityRepresentation(
                evidence_id=str(item.evidence_id),
                representation_id=item.representation_id,
                canonical_text=evidence_by_id[str(item.evidence_id)].canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority.representations
        )
        typed_receipt = QueryAnchoredScopedProcedureListClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=witnesses,
            authority_representations=raw_authorities,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _scoped_table_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """从完整 raw authority 重建多行表选择回执。"""

    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    receipt_representation_ids = _exact_tokens(
        receipt.get("catalog_representation_ids"),
        limit=256,
    )
    receipt_evidence_ids = _exact_tokens(
        receipt.get("catalog_evidence_ids"),
        limit=256,
    )
    authority_representation_ids = [item.representation_id for item in authority.representations]
    authority_evidence_ids = [item.evidence_id for item in authority.representations]
    if (
        receipt_representation_ids != authority_representation_ids
        or receipt_evidence_ids != authority_evidence_ids
        or any(item is None for item in authority_evidence_ids)
    ):
        return False
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    if len(evidence_by_id) != len(authority.evidences):
        return False
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if len(obligations) != 1:
        return False
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredScopedTableSelectionClosureReceipt,
            ScopedTableAuthorityRepresentation,
        )

        raw_authorities = tuple(
            ScopedTableAuthorityRepresentation(
                evidence_id=str(item.evidence_id),
                representation_id=item.representation_id,
                canonical_text=evidence_by_id[str(item.evidence_id)].canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority.representations
        )
        typed_receipt = QueryAnchoredScopedTableSelectionClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=witnesses,
            authority_representations=raw_authorities,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _scoped_bullet_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """从完整 raw authority 重建显式有界 bullet 列表回执。"""

    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    receipt_representation_ids = _exact_tokens(
        receipt.get("catalog_representation_ids"),
        limit=256,
    )
    receipt_evidence_ids = _exact_tokens(
        receipt.get("catalog_evidence_ids"),
        limit=256,
    )
    authority_representation_ids = [item.representation_id for item in authority.representations]
    authority_evidence_ids = [item.evidence_id for item in authority.representations]
    if (
        receipt_representation_ids != authority_representation_ids
        or receipt_evidence_ids != authority_evidence_ids
        or any(item is None for item in authority_evidence_ids)
    ):
        return False
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    if len(evidence_by_id) != len(authority.evidences):
        return False
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if len(obligations) != 1:
        return False
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredScopedBulletListClosureReceipt,
            ScopedBulletAuthorityRepresentation,
        )

        raw_authorities = tuple(
            ScopedBulletAuthorityRepresentation(
                evidence_id=str(item.evidence_id),
                representation_id=item.representation_id,
                canonical_text=evidence_by_id[str(item.evidence_id)].canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority.representations
        )
        typed_receipt = QueryAnchoredScopedBulletListClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=witnesses,
            authority_representations=raw_authorities,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _inline_action_chain_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """从 raw canonical/packed authority 独立重建内联动作链。"""

    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    receipt_representation_ids = _exact_tokens(
        receipt.get("catalog_representation_ids"),
        limit=256,
    )
    receipt_evidence_ids = _exact_tokens(
        receipt.get("catalog_evidence_ids"),
        limit=256,
    )
    authority_representation_ids = [item.representation_id for item in authority.representations]
    authority_evidence_ids = [item.evidence_id for item in authority.representations]
    if (
        receipt_representation_ids != authority_representation_ids
        or receipt_evidence_ids != authority_evidence_ids
        or any(item is None for item in authority_evidence_ids)
    ):
        return False
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    if len(evidence_by_id) != len(authority.evidences):
        return False
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if len(obligations) != 1:
        return False
    try:
        from app.domain.answering.enumeration_closure import (
            QueryAnchoredInlineActionChainClosureReceipt,
            ScopedListAuthorityRepresentation,
        )

        raw_authorities = tuple(
            ScopedListAuthorityRepresentation(
                evidence_id=str(item.evidence_id),
                representation_id=item.representation_id,
                canonical_text=evidence_by_id[str(item.evidence_id)].canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority.representations
        )
        typed_receipt = QueryAnchoredInlineActionChainClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=tuple(
                item for item in witnesses if item.atom_kind.value in {"sentence", "list_item"}
            ),
            authority_representations=raw_authorities,
            catalog_witnesses=witnesses,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _scoped_fact_set_receipt_matches_proof_authority(
    receipt: Mapping[str, Any],
    *,
    catalog_receipt: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    authority: ProofEvaluationAuthority | None,
) -> bool:
    """从完整 raw canonical/packed authority 重建事实集合回执。"""

    trusted = _trusted_proof_inputs(authority)
    if (
        trusted is None
        or not _catalog_matches_proof_authority(catalog_receipt, authority)
        or not _question_matches_proof_authority(question_receipt, authority)
    ):
        return False
    assert authority is not None
    question, catalog, witnesses = trusted
    receipt_representation_ids = _exact_tokens(
        receipt.get("catalog_representation_ids"),
        limit=256,
    )
    receipt_evidence_ids = _exact_tokens(
        receipt.get("catalog_evidence_ids"),
        limit=256,
    )
    authority_representation_ids = [item.representation_id for item in authority.representations]
    authority_evidence_ids = [item.evidence_id for item in authority.representations]
    if (
        receipt_representation_ids != authority_representation_ids
        or receipt_evidence_ids != authority_evidence_ids
        or any(item is None for item in authority_evidence_ids)
    ):
        return False
    evidence_by_id = {item.evidence_id: item for item in authority.evidences}
    if len(evidence_by_id) != len(authority.evidences):
        return False
    obligations = [
        item for item in question.obligations if item.obligation_id == receipt.get("obligation_id")
    ]
    if len(obligations) != 1:
        return False
    try:
        from app.domain.answering.fact_set_closure import (
            FactSetAuthorityRepresentation,
            QueryAnchoredScopedFactSetClosureReceipt,
        )

        raw_authorities = tuple(
            FactSetAuthorityRepresentation(
                evidence_id=str(item.evidence_id),
                representation_id=item.representation_id,
                canonical_text=evidence_by_id[str(item.evidence_id)].canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority.representations
        )
        typed_receipt = QueryAnchoredScopedFactSetClosureReceipt.from_dict(dict(receipt))
        typed_receipt.validate_artifacts(
            question=question,
            catalog=catalog,
            obligation=obligations[0],
            witnesses=tuple(
                item for item in witnesses if item.atom_kind.value in {"sentence", "list_item"}
            ),
            authority_representations=raw_authorities,
            catalog_witnesses=witnesses,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _source_table_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """只凭低基数回执重放表记录闭包的身份和 proof 图绑定。"""

    if set(receipt) != _SOURCE_ENUMERATION_CLOSURE_FIELDS:
        return False
    if (
        receipt.get("schema_version") != "query-anchored-table-row-closure-receipt.v4"
        or receipt.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
    ):
        return False
    if any(
        _token(receipt.get(key)) != receipt.get(key)
        for key in _SOURCE_ENUMERATION_CLOSURE_TOKEN_FIELDS
    ) or any(
        _hash(receipt.get(key)) != receipt.get(key)
        for key in _SOURCE_ENUMERATION_CLOSURE_HASH_FIELDS
    ):
        return False
    if any(
        _integer(receipt.get(key)) != receipt.get(key)
        for key in _SOURCE_ENUMERATION_CLOSURE_INTEGER_FIELDS
    ):
        return False

    table_start = int(receipt["table_start"])
    table_end = int(receipt["table_end"])
    packed_table_start = int(receipt["packed_table_start"])
    packed_table_end = int(receipt["packed_table_end"])
    table_data_row_count = int(receipt["table_data_row_count"])
    if (
        table_start >= table_end
        or packed_table_start >= packed_table_end
        or int(receipt["table_column_count"]) < 2
        or table_data_row_count < 1
        or int(receipt["witness_table_row_index"]) >= table_data_row_count
        or receipt.get("key_cell_index") not in {0, 1}
        or receipt.get("key_occurrence_count") != 1
        or receipt.get("catalog_representation_count") != 1
        or receipt.get("catalog_key_mention_count") != 1
        or receipt.get("canonical_non_table_character_count") != 0
        or receipt.get("packed_non_table_character_count") != 0
    ):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_ENUMERATION_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qtrc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _table_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False

    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    matching_authoritative_receipts = [
        item for item in authoritative_receipts if dict(item) == dict(receipt)
    ]
    if (
        len(matching_authoritative_receipts) != 1
        or catalog_receipt.get("context_representation_count") != 1
    ):
        return False

    authority_summaries = _items(catalog_receipt.get("source_table_authority_summaries"))
    matching_summaries = [
        item
        for item in authority_summaries
        if item.get("schema_version") == "source-table-authority-summary.v1"
        and item.get("receipt_id") == receipt.get("receipt_id")
        and item.get("receipt_hash") == receipt.get("receipt_hash")
        and item.get("witness_id") == receipt.get("witness_id")
        and item.get("evidence_id") == receipt.get("evidence_id")
        and item.get("representation_id") == receipt.get("representation_id")
    ]
    if len(matching_summaries) != 1:
        return False
    canonical_summary = _mapping(matching_summaries[0].get("canonical"))
    packed_summary = _mapping(matching_summaries[0].get("packed"))
    canonical_bindings = {
        "table_start": "table_start",
        "table_end": "table_end",
        "table_line_index": "table_line_index",
        "table_column_count": "table_column_count",
        "table_data_row_count": "table_data_row_count",
        "witness_table_row_index": "witness_table_row_index",
        "table_hash": "table_hash",
        "header_hash": "header_hash",
        "separator_hash": "separator_hash",
        "row_hash": "row_hash",
        "table_structure_hash": "table_structure_hash",
    }
    packed_bindings = {
        "table_start": "packed_table_start",
        "table_end": "packed_table_end",
        "table_line_index": "table_line_index",
        "table_column_count": "table_column_count",
        "table_data_row_count": "table_data_row_count",
        "witness_table_row_index": "witness_table_row_index",
        "table_hash": "packed_table_hash",
        "header_hash": "packed_header_hash",
        "separator_hash": "packed_separator_hash",
        "row_hash": "packed_row_hash",
        "table_structure_hash": "table_structure_hash",
    }
    if any(
        canonical_summary.get(summary_key) != receipt.get(receipt_key)
        for summary_key, receipt_key in canonical_bindings.items()
    ) or any(
        packed_summary.get(summary_key) != receipt.get(receipt_key)
        for summary_key, receipt_key in packed_bindings.items()
    ):
        return False

    representation_id = _token(receipt.get("representation_id")) or ""
    evidence_id = _token(receipt.get("evidence_id")) or ""
    packed_hash = _mapping(catalog_receipt.get("representation_packed_hashes")).get(
        representation_id
    )
    packed_length = _mapping(catalog_receipt.get("representation_packed_lengths")).get(
        representation_id
    )
    canonical_hash = _mapping(catalog_receipt.get("evidence_canonical_hashes")).get(evidence_id)
    canonical_length = _mapping(catalog_receipt.get("evidence_canonical_lengths")).get(evidence_id)
    if (
        not isinstance(canonical_length, int)
        or not isinstance(packed_length, int)
        or table_end > canonical_length
        or packed_table_end > packed_length
        or receipt.get("evidence_content_hash") != canonical_hash
        or packed_hash is None
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if len(supporting_claim_ids) != 1 or len(proven_items) != 1:
        return False
    claim_id = supporting_claim_ids[0]
    claim = claim_by_id.get(claim_id)
    item = proven_items[0]
    witness_id = _token(receipt.get("witness_id"))
    if (
        claim is None
        or obligation_id is None
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or claim.get("obligation_id") != obligation_id
        or _tokens(claim.get("witness_ids")) != [witness_id]
        or item.get("obligation_id") != obligation_id
        or item.get("claim_id") != claim_id
        or item.get("witness_id") != witness_id
        or item.get("witness_hash") != receipt.get("witness_hash")
        or item.get("segmentation_method") != "markdown_table_data_row.v1"
    ):
        return False

    witness_bindings = {
        "witness_hashes": "witness_hash",
        "witness_evidence_ids": "evidence_id",
        "witness_evidence_content_hashes": "evidence_content_hash",
        "witness_representation_ids": "representation_id",
        "witness_span_ref_hashes": "span_ref_hash",
        "witness_canonical_span_hashes": "row_hash",
        "witness_packed_span_hashes": "packed_row_hash",
    }
    if any(
        _mapping(catalog_receipt.get(mapping_key)).get(witness_id or "") != receipt.get(receipt_key)
        for mapping_key, receipt_key in witness_bindings.items()
    ):
        return False
    if _mapping(catalog_receipt.get("witness_atom_kinds")).get(
        witness_id or ""
    ) != "table_data_row" or witness_id not in _tokens(catalog_receipt.get("witness_ids")):
        return False

    canonical_start = _mapping(catalog_receipt.get("witness_canonical_starts")).get(
        witness_id or ""
    )
    canonical_end = _mapping(catalog_receipt.get("witness_canonical_ends")).get(witness_id or "")
    packed_start = _mapping(catalog_receipt.get("witness_packed_starts")).get(witness_id or "")
    packed_end = _mapping(catalog_receipt.get("witness_packed_ends")).get(witness_id or "")
    return (
        isinstance(canonical_start, int)
        and not isinstance(canonical_start, bool)
        and isinstance(canonical_end, int)
        and not isinstance(canonical_end, bool)
        and isinstance(packed_start, int)
        and not isinstance(packed_start, bool)
        and isinstance(packed_end, int)
        and not isinstance(packed_end, bool)
        and table_start <= canonical_start < canonical_end <= table_end
        and packed_table_start <= packed_start < packed_end <= packed_table_end
    )


def _source_procedure_list_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    answer: object,
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """从权威正文摘要、WitnessCatalog 和 proof 图独立重放有序步骤闭包。"""

    if not _complete_witness_catalog_identity_is_valid(catalog_receipt):
        return False
    if set(receipt) != _SOURCE_PROCEDURE_CLOSURE_FIELDS:
        return False
    marker_style = receipt.get("marker_style")
    if (
        receipt.get("schema_version") != "query-anchored-procedure-list-closure-receipt.v1"
        or receipt.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
        or not isinstance(marker_style, str)
        or re.fullmatch(r"(?:arabic|chinese):[.)．、]", marker_style) is None
    ):
        return False
    if any(
        _token(receipt.get(key)) != receipt.get(key)
        for key in _SOURCE_PROCEDURE_CLOSURE_TOKEN_FIELDS - {"marker_style"}
    ) or any(
        _hash(receipt.get(key)) != receipt.get(key) for key in _SOURCE_PROCEDURE_CLOSURE_HASH_FIELDS
    ):
        return False
    if any(
        _integer(receipt.get(key)) != receipt.get(key)
        for key in _SOURCE_PROCEDURE_CLOSURE_INTEGER_FIELDS
    ):
        return False
    witness_ids = _exact_tokens(receipt.get("witness_ids"))
    witness_hashes = _exact_hashes(receipt.get("witness_hashes"))
    span_ref_hashes = _exact_hashes(receipt.get("span_ref_hashes"))
    canonical_item_hashes = _exact_hashes(receipt.get("canonical_item_hashes"))
    packed_item_hashes = _exact_hashes(receipt.get("packed_item_hashes"))
    witness_ordinals = _exact_integers(receipt.get("witness_ordinals"))
    item_ordinals = _exact_integers(receipt.get("item_ordinals"))
    arrays = (
        witness_ids,
        witness_hashes,
        span_ref_hashes,
        canonical_item_hashes,
        packed_item_hashes,
        witness_ordinals,
        item_ordinals,
    )
    if any(item is None for item in arrays):
        return False
    item_count = int(receipt["item_count"])
    if (
        not 2 <= item_count <= 64
        or any(len(item or []) != item_count for item in arrays)
        or len(set(witness_ids or [])) != item_count
        or witness_ordinals
        != list(range((witness_ordinals or [0])[0], (witness_ordinals or [0])[0] + item_count))
        or item_ordinals != list(range(1, item_count + 1))
        or int(receipt["canonical_block_start"]) >= int(receipt["canonical_block_end"])
        or int(receipt["packed_block_start"]) >= int(receipt["packed_block_end"])
        or int(receipt["first_item_line_index"]) != int(receipt["title_line_index"]) + 1
        or receipt.get("catalog_representation_count") != 1
        or receipt.get("canonical_non_block_character_count") != 0
        or receipt.get("packed_non_block_character_count") != 0
    ):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_PROCEDURE_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qplc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _procedure_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False

    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if (
        sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1
        or catalog_receipt.get("context_representation_count") != 1
    ):
        return False
    summaries = [
        item
        for item in _items(catalog_receipt.get("source_procedure_authority_summaries"))
        if item.get("schema_version") == "source-procedure-list-authority-summary.v1"
        and item.get("receipt_id") == receipt.get("receipt_id")
        and item.get("receipt_hash") == receipt.get("receipt_hash")
        and item.get("witness_ids") == witness_ids
        and item.get("evidence_id") == receipt.get("evidence_id")
        and item.get("representation_id") == receipt.get("representation_id")
    ]
    if len(summaries) != 1:
        return False
    canonical_summary = _mapping(summaries[0].get("canonical"))
    packed_summary = _mapping(summaries[0].get("packed"))
    canonical_bindings = {
        "block_start": "canonical_block_start",
        "block_end": "canonical_block_end",
        "title_line_index": "title_line_index",
        "first_item_line_index": "first_item_line_index",
        "item_count": "item_count",
        "item_ordinals": "item_ordinals",
        "marker_style": "marker_style",
        "item_hashes": "canonical_item_hashes",
        "block_hash": "canonical_block_hash",
        "title_hash": "canonical_title_hash",
        "list_structure_hash": "list_structure_hash",
        "title_anchor_hash": "title_anchor_hash",
        "non_block_character_count": "canonical_non_block_character_count",
    }
    packed_bindings = {
        "block_start": "packed_block_start",
        "block_end": "packed_block_end",
        "title_line_index": "title_line_index",
        "first_item_line_index": "first_item_line_index",
        "item_count": "item_count",
        "item_ordinals": "item_ordinals",
        "marker_style": "marker_style",
        "item_hashes": "packed_item_hashes",
        "block_hash": "packed_block_hash",
        "title_hash": "packed_title_hash",
        "list_structure_hash": "list_structure_hash",
        "title_anchor_hash": "title_anchor_hash",
        "non_block_character_count": "packed_non_block_character_count",
    }
    if any(
        canonical_summary.get(summary_key) != receipt.get(receipt_key)
        for summary_key, receipt_key in canonical_bindings.items()
    ) or any(
        packed_summary.get(summary_key) != receipt.get(receipt_key)
        for summary_key, receipt_key in packed_bindings.items()
    ):
        return False

    representation_id = _token(receipt.get("representation_id")) or ""
    evidence_id = _token(receipt.get("evidence_id")) or ""
    packed_length = _mapping(catalog_receipt.get("representation_packed_lengths")).get(
        representation_id
    )
    canonical_length = _mapping(catalog_receipt.get("evidence_canonical_lengths")).get(evidence_id)
    if (
        not isinstance(canonical_length, int)
        or isinstance(canonical_length, bool)
        or not isinstance(packed_length, int)
        or isinstance(packed_length, bool)
        or int(receipt["canonical_block_end"]) > canonical_length
        or int(receipt["packed_block_end"]) > packed_length
        or receipt.get("evidence_content_hash")
        != _mapping(catalog_receipt.get("evidence_canonical_hashes")).get(evidence_id)
        or representation_id not in _mapping(catalog_receipt.get("representation_packed_hashes"))
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != item_count
        or len(proven_items) != item_count
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "ordered_steps"
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    atom_kinds = _mapping(catalog_receipt.get("witness_atom_kinds"))
    evidence_ids = _mapping(catalog_receipt.get("witness_evidence_ids"))
    representation_ids = _mapping(catalog_receipt.get("witness_representation_ids"))
    authoritative_witness_ids = [
        witness_id
        for witness_id in catalog_witness_ids
        if atom_kinds.get(witness_id) == "list_item"
        and evidence_ids.get(witness_id) == evidence_id
        and representation_ids.get(witness_id) == representation_id
    ]
    if (
        authoritative_witness_ids != witness_ids
        or [catalog_witness_ids.index(item) for item in authoritative_witness_ids]
        != witness_ordinals
    ):
        return False
    catalog_maps = {
        "witness_hashes": witness_hashes,
        "witness_span_ref_hashes": span_ref_hashes,
        "witness_canonical_span_hashes": canonical_item_hashes,
        "witness_packed_span_hashes": packed_item_hashes,
    }
    for mapping_key, expected_values in catalog_maps.items():
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if [mapping.get(item) for item in witness_ids or []] != expected_values:
            return False
    for mapping_key, expected_value in (
        ("witness_evidence_ids", evidence_id),
        ("witness_evidence_content_hashes", receipt.get("evidence_content_hash")),
        ("witness_representation_ids", representation_id),
        ("witness_atom_kinds", "list_item"),
    ):
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if any(mapping.get(item) != expected_value for item in witness_ids or []):
            return False
    canonical_starts = [
        _mapping(catalog_receipt.get("witness_canonical_starts")).get(item)
        for item in witness_ids or []
    ]
    canonical_ends = [
        _mapping(catalog_receipt.get("witness_canonical_ends")).get(item)
        for item in witness_ids or []
    ]
    packed_starts = [
        _mapping(catalog_receipt.get("witness_packed_starts")).get(item)
        for item in witness_ids or []
    ]
    packed_ends = [
        _mapping(catalog_receipt.get("witness_packed_ends")).get(item) for item in witness_ids or []
    ]
    if (
        canonical_summary.get("item_starts") != canonical_starts
        or canonical_summary.get("item_ends") != canonical_ends
        or packed_summary.get("item_starts") != packed_starts
        or packed_summary.get("item_ends") != packed_ends
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*canonical_starts, *canonical_ends, *packed_starts, *packed_ends)
        )
        or int(receipt["canonical_block_end"]) != canonical_ends[-1]
        or int(receipt["packed_block_end"]) != packed_ends[-1]
    ):
        return False

    if not isinstance(answer, str):
        return False
    for index, (claim_id, witness_id, witness_hash) in enumerate(
        zip(supporting_claim_ids, witness_ids or [], witness_hashes or [], strict=True)
    ):
        claim = claim_by_id.get(claim_id)
        item = proven_items[index]
        if (
            claim is None
            or claim.get("obligation_id") != obligation_id
            or _tokens(claim.get("witness_ids")) != [witness_id]
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hash
            or item.get("segmentation_method") != "top_level_list_entry.v1"
        ):
            return False
        answer_span = _mapping(claim.get("answer_span"))
        answer_start = _integer(answer_span.get("start"))
        answer_end = _integer(answer_span.get("end"), minimum=1)
        packed_start = packed_starts[index]
        packed_end = packed_ends[index]
        if (
            answer_start is None
            or answer_end is None
            or answer_start >= answer_end
            or answer_end > len(answer)
            or not isinstance(packed_start, int)
            or isinstance(packed_start, bool)
            or not isinstance(packed_end, int)
            or isinstance(packed_end, bool)
            or packed_start >= packed_end
        ):
            return False
        rendered = answer[answer_start:answer_end]
        content_length = packed_end - packed_start
        content = rendered[:content_length]
        suffix = rendered[content_length:]
        parsed = _procedure_ordered_item(content)
        if (
            _text_hash(rendered) != claim.get("rendered_text_hash")
            or _text_hash(content) != (packed_item_hashes or [])[index]
            or re.fullmatch(r" \[[^\]\r\n]+\]", suffix) is None
            or parsed is None
            or parsed[0] != index + 1
            or parsed[1] != marker_style
        ):
            return False
    return True


def _source_inline_action_chain_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """重放唯一 Witness 的动作边界，并逐项绑定 proof。"""

    if (
        not _complete_witness_catalog_identity_is_valid(catalog_receipt)
        or set(receipt) != _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_FIELDS
        or _source_inline_action_chain_closure_receipt(receipt) != dict(receipt)
    ):
        return False
    payload = {
        key: receipt[key]
        for key in _SOURCE_INLINE_ACTION_CHAIN_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qiac_" + expected_hash.removeprefix("sha256:")[:40]
    if (
        receipt.get("receipt_hash") != expected_hash
        or receipt.get("receipt_id") != expected_id
        or not _inline_action_chain_receipt_matches_proof_authority(
            receipt,
            catalog_receipt=catalog_receipt,
            question_receipt=question_receipt,
            authority=proof_authority,
        )
    ):
        return False
    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1:
        return False

    witness_ids = _exact_tokens(receipt.get("witness_ids"), limit=1) or []
    witness_hashes = _exact_hashes(receipt.get("witness_hashes"), limit=1) or []
    witness_ordinals = _exact_integers(receipt.get("witness_ordinals"), limit=1) or []
    span_ref_hashes = _exact_hashes(receipt.get("span_ref_hashes"), limit=1) or []
    canonical_starts = (
        _exact_integers(
            receipt.get("canonical_action_starts"),
            limit=16,
        )
        or []
    )
    canonical_ends = (
        _exact_integers(
            receipt.get("canonical_action_ends"),
            limit=16,
        )
        or []
    )
    packed_starts = _exact_integers(receipt.get("packed_action_starts"), limit=16) or []
    packed_ends = _exact_integers(receipt.get("packed_action_ends"), limit=16) or []
    action_ordinals = _exact_integers(receipt.get("action_ordinals"), limit=16) or []
    canonical_action_hashes = (
        _exact_hashes(
            receipt.get("canonical_action_hashes"),
            limit=16,
        )
        or []
    )
    packed_action_hashes = (
        _exact_hashes(
            receipt.get("packed_action_hashes"),
            limit=16,
        )
        or []
    )
    catalog_evidence_ids = (
        _exact_tokens(
            receipt.get("catalog_evidence_ids"),
            limit=256,
        )
        or []
    )
    catalog_representation_ids = (
        _exact_tokens(
            receipt.get("catalog_representation_ids"),
            limit=256,
        )
        or []
    )
    catalog_evidence_hashes = (
        _exact_hashes(
            receipt.get("catalog_evidence_content_hashes"),
            limit=256,
        )
        or []
    )
    catalog_packed_hashes = (
        _exact_hashes(
            receipt.get("catalog_packed_text_hashes"),
            limit=256,
        )
        or []
    )
    action_count = int(receipt["action_count"])
    selected_index = int(receipt["selected_authority_index"])
    witness_id = witness_ids[0]
    evidence_id = catalog_evidence_ids[selected_index]
    representation_id = catalog_representation_ids[selected_index]

    representation_hashes = _mapping(catalog_receipt.get("representation_packed_hashes"))
    evidence_hashes = _mapping(catalog_receipt.get("evidence_canonical_hashes"))
    if (
        catalog_receipt.get("context_representation_count")
        != receipt.get("catalog_representation_count")
        or any(
            representation_hashes.get(item_id) != item_hash
            for item_id, item_hash in zip(
                catalog_representation_ids,
                catalog_packed_hashes,
                strict=True,
            )
        )
        or any(
            evidence_hashes.get(item_id) != item_hash
            for item_id, item_hash in zip(
                catalog_evidence_ids,
                catalog_evidence_hashes,
                strict=True,
            )
        )
        or receipt.get("catalog_authority_hash")
        != _stable_mapping_hash(
            {
                "evidence_ids": catalog_evidence_ids,
                "representation_ids": catalog_representation_ids,
                "canonical_hashes": catalog_evidence_hashes,
                "packed_hashes": catalog_packed_hashes,
            }
        )
        or receipt.get("evidence_id") != evidence_id
        or receipt.get("representation_id") != representation_id
        or receipt.get("evidence_content_hash") != catalog_evidence_hashes[selected_index]
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    witness_maps = {
        "witness_hashes": witness_hashes[0],
        "witness_span_ref_hashes": span_ref_hashes[0],
        "witness_evidence_ids": evidence_id,
        "witness_evidence_content_hashes": receipt.get("evidence_content_hash"),
        "witness_representation_ids": representation_id,
        "witness_atom_kinds": receipt.get("witness_atom_kind"),
        "witness_canonical_span_hashes": receipt.get("canonical_witness_hash"),
        "witness_packed_span_hashes": receipt.get("packed_witness_hash"),
        "witness_canonical_starts": receipt.get("canonical_witness_start"),
        "witness_canonical_ends": receipt.get("canonical_witness_end"),
        "witness_packed_starts": receipt.get("packed_witness_start"),
        "witness_packed_ends": receipt.get("packed_witness_end"),
    }
    if (
        witness_id not in catalog_witness_ids
        or catalog_witness_ids.index(witness_id) != witness_ordinals[0]
        or any(
            _mapping(catalog_receipt.get(mapping_key)).get(witness_id) != expected
            for mapping_key, expected in witness_maps.items()
        )
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"), limit=64)
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != 1
        or len(proven_items) != action_count
        or receipt.get("source_query_hash") != question_receipt.get("source_query_hash")
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "ordered_steps"
    ):
        return False
    claim_id = supporting_claim_ids[0]
    claim = claim_by_id.get(claim_id)
    if (
        claim is None
        or claim.get("obligation_id") != obligation_id
        or _tokens(claim.get("witness_ids")) != [witness_id]
    ):
        return False

    if proof_authority is None:
        return False
    source_pair = _authority_source_pair(
        proof_authority,
        evidence_id=evidence_id,
        representation_id=representation_id,
    )
    if source_pair is None:
        return False
    evidence, representation = source_pair
    ordered_actions = sorted(
        zip(
            action_ordinals,
            canonical_starts,
            canonical_ends,
            packed_starts,
            packed_ends,
            canonical_action_hashes,
            packed_action_hashes,
            strict=True,
        ),
        key=lambda item: item[0],
    )
    packed_witness_start = int(receipt["packed_witness_start"])
    for item, action in zip(proven_items, ordered_actions, strict=True):
        (
            _ordinal,
            canonical_start,
            canonical_end,
            packed_start,
            packed_end,
            canonical_hash,
            packed_hash,
        ) = action
        canonical_text = evidence.canonical_text[canonical_start:canonical_end]
        packed_text = representation.packed_text[packed_start:packed_end]
        if (
            _text_hash(canonical_text) != canonical_hash
            or _text_hash(packed_text) != packed_hash
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hashes[0]
            or item.get("relative_start") != packed_start - packed_witness_start
            or item.get("relative_end") != packed_end - packed_witness_start
            or item.get("raw_span_hash") != packed_hash
            or item.get("normalized_item_hash") != _normalized_proven_item_hash(packed_text)
            or item.get("segmentation_method") != "inline_action_span.v1"
        ):
            return False
    return True


def _source_scoped_fact_set_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """重放查询锚定事实集合，并将每个来源成员绑定到 proof 图。"""

    if (
        not _complete_witness_catalog_identity_is_valid(catalog_receipt)
        or set(receipt) != _SOURCE_SCOPED_FACT_SET_CLOSURE_FIELDS
        or receipt.get("schema_version") != "query-anchored-scoped-fact-set-closure-receipt.v1"
        or receipt.get("normalization_version") != "nfkc-markup-aware-fact-selector.v1"
        or receipt.get("mode")
        not in {
            "closed_inline_coordination",
            "scoped_topic_predicate_set",
            "scoped_bullet_lead_clause_set",
        }
        or receipt.get("relation")
        not in {
            "inclusion",
            "required_information",
            "directive_compliance",
            "applicability_restriction",
            "safety_instruction",
            "ownership_registration",
        }
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
    ):
        return False
    if (
        any(
            _token(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_FIELDS
        )
        or any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_FIELDS
        )
    ):
        return False

    optional_title_values = tuple(
        receipt.get(key)
        for key in (
            "canonical_title_hash",
            "packed_title_hash",
            "canonical_title_start",
            "canonical_title_end",
            "packed_title_start",
            "packed_title_end",
        )
    )
    if any(item is None for item in optional_title_values) and not all(
        item is None for item in optional_title_values
    ):
        return False
    if optional_title_values[0] is not None and (
        any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_OPTIONAL_INTEGER_FIELDS
        )
    ):
        return False

    token_arrays = {
        key: _exact_tokens(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_TOKEN_ARRAY_FIELDS
    }
    hash_arrays = {
        key: _exact_hashes(
            receipt.get(key),
            limit=(
                256
                if key.startswith("catalog_")
                else 128
                if key == "flattened_semantic_hashes"
                else 64
            ),
        )
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_HASH_ARRAY_FIELDS
    }
    integer_arrays = {
        key: _exact_integers(receipt.get(key), limit=64)
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_INTEGER_ARRAY_FIELDS
    }
    if (
        any(item is None for item in token_arrays.values())
        or any(item is None for item in hash_arrays.values())
        or any(item is None for item in integer_arrays.values())
    ):
        return False

    catalog_evidence_ids = token_arrays["catalog_evidence_ids"] or []
    catalog_representation_ids = token_arrays["catalog_representation_ids"] or []
    witness_ids = token_arrays["witness_ids"] or []
    member_atom_kinds = token_arrays["member_atom_kinds"] or []
    segmentation_methods = token_arrays["segmentation_methods"] or []
    anchor_match_key_hashes = hash_arrays["anchor_match_key_hashes"] or []
    topic_group_hashes = hash_arrays["topic_group_hashes"] or []
    catalog_evidence_hashes = hash_arrays["catalog_evidence_content_hashes"] or []
    catalog_packed_hashes = hash_arrays["catalog_packed_text_hashes"] or []
    witness_hashes = hash_arrays["witness_hashes"] or []
    span_ref_hashes = hash_arrays["span_ref_hashes"] or []
    canonical_member_hashes = hash_arrays["canonical_member_hashes"] or []
    packed_member_hashes = hash_arrays["packed_member_hashes"] or []
    canonical_normalized_hashes = hash_arrays["canonical_member_normalized_hashes"] or []
    packed_normalized_hashes = hash_arrays["packed_member_normalized_hashes"] or []
    flattened_semantic_hashes = hash_arrays["flattened_semantic_hashes"] or []
    topic_group_member_counts = integer_arrays["topic_group_member_counts"] or []
    witness_ordinals = integer_arrays["witness_ordinals"] or []
    canonical_member_starts = integer_arrays["canonical_member_starts"] or []
    canonical_member_ends = integer_arrays["canonical_member_ends"] or []
    packed_member_starts = integer_arrays["packed_member_starts"] or []
    packed_member_ends = integer_arrays["packed_member_ends"] or []
    member_semantic_counts = integer_arrays["member_semantic_counts"] or []
    member_count = int(receipt["member_count"])
    semantic_count = int(receipt["semantic_count"])
    topic_group_count = int(receipt["topic_group_count"])
    anchor_match_key_count = int(receipt["anchor_match_key_count"])
    representation_count = int(receipt["catalog_representation_count"])
    selected_authority_index = int(receipt["selected_authority_index"])
    member_arrays = (
        witness_ids,
        witness_hashes,
        witness_ordinals,
        span_ref_hashes,
        member_atom_kinds,
        segmentation_methods,
        canonical_member_starts,
        canonical_member_ends,
        packed_member_starts,
        packed_member_ends,
        canonical_member_hashes,
        packed_member_hashes,
        canonical_normalized_hashes,
        packed_normalized_hashes,
        member_semantic_counts,
    )
    authority_arrays = (
        catalog_evidence_ids,
        catalog_representation_ids,
        catalog_evidence_hashes,
        catalog_packed_hashes,
    )
    expected_segmentation = {
        "sentence": "open_enumeration_witness_atom.v1",
        "list_item": "top_level_list_entry.v1",
    }
    if (
        not 1 <= member_count <= 64
        or any(len(item) != member_count for item in member_arrays)
        or len(set(witness_ids)) != member_count
        or witness_ordinals != sorted(set(witness_ordinals))
        or canonical_member_starts != sorted(canonical_member_starts)
        or packed_member_starts != sorted(packed_member_starts)
        or any(
            expected_segmentation.get(atom_kind) != segmentation_method
            for atom_kind, segmentation_method in zip(
                member_atom_kinds,
                segmentation_methods,
                strict=True,
            )
        )
        or not 2 <= anchor_match_key_count <= 64
        or len(anchor_match_key_hashes) != anchor_match_key_count
        or len(set(anchor_match_key_hashes)) != anchor_match_key_count
        or receipt.get("anchor_match_hash") != _stable_sequence_hash(anchor_match_key_hashes)
        or not 1 <= topic_group_count <= 32
        or len(topic_group_hashes) != topic_group_count
        or len(topic_group_member_counts) != topic_group_count
        or len(set(topic_group_hashes)) != topic_group_count
        or any(count < 1 for count in topic_group_member_counts)
        or semantic_count != sum(member_semantic_counts)
        or semantic_count != len(flattened_semantic_hashes)
        or semantic_count < 2
        or any(count < 1 for count in member_semantic_counts)
        or canonical_normalized_hashes != packed_normalized_hashes
        or not 1 <= representation_count <= 256
        or any(len(item) != representation_count for item in authority_arrays)
        or len(set(catalog_evidence_ids)) != representation_count
        or len(set(catalog_representation_ids)) != representation_count
        or not 0 <= selected_authority_index < representation_count
        or receipt.get("authority_candidate_count") != 1
        or catalog_receipt.get("context_representation_count") != representation_count
        or int(receipt["canonical_scope_start"]) >= int(receipt["canonical_scope_end"])
        or int(receipt["packed_scope_start"]) >= int(receipt["packed_scope_end"])
        or any(
            receipt.get(key) != 0
            for key in (
                "omission_marker_count",
                "open_marker_count",
                "reference_marker_count",
                "ambiguous_anaphor_count",
                "truncation_marker_count",
                "supplement_candidate_count",
            )
        )
    ):
        return False
    if receipt.get("mode") == "closed_inline_coordination":
        if member_count != 1 or member_semantic_counts[0] < 2:
            return False
    elif receipt.get("mode") == "scoped_topic_predicate_set":
        if (
            topic_group_count < 2
            or member_semantic_counts != [1] * member_count
            or topic_group_member_counts != [1] * topic_group_count
            or all(item is None for item in optional_title_values)
        ):
            return False
    elif (
        member_count < 2
        or topic_group_count != 1
        or topic_group_member_counts != [member_count]
        or member_semantic_counts != [1] * member_count
        or member_atom_kinds != ["list_item"] * member_count
        or segmentation_methods != ["top_level_list_entry.v1"] * member_count
        or any(item is None for item in optional_title_values)
    ):
        return False
    if any(
        start < int(receipt["canonical_scope_start"])
        or end > int(receipt["canonical_scope_end"])
        or start >= end
        for start, end in zip(canonical_member_starts, canonical_member_ends, strict=True)
    ) or any(
        start < int(receipt["packed_scope_start"])
        or end > int(receipt["packed_scope_end"])
        or start >= end
        for start, end in zip(packed_member_starts, packed_member_ends, strict=True)
    ):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_SCOPED_FACT_SET_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qfsc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _scoped_fact_set_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False
    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1:
        return False

    representation_hashes = _mapping(catalog_receipt.get("representation_packed_hashes"))
    evidence_hashes = _mapping(catalog_receipt.get("evidence_canonical_hashes"))
    if any(
        representation_hashes.get(representation_id) != packed_hash
        for representation_id, packed_hash in zip(
            catalog_representation_ids,
            catalog_packed_hashes,
            strict=True,
        )
    ) or any(
        evidence_hashes.get(evidence_id) != evidence_hash
        for evidence_id, evidence_hash in zip(
            catalog_evidence_ids,
            catalog_evidence_hashes,
            strict=True,
        )
    ):
        return False
    authority_payload = {
        "evidence_ids": catalog_evidence_ids,
        "representation_ids": catalog_representation_ids,
        "canonical_hashes": catalog_evidence_hashes,
        "packed_hashes": catalog_packed_hashes,
    }
    evidence_id = catalog_evidence_ids[selected_authority_index]
    representation_id = catalog_representation_ids[selected_authority_index]
    if (
        receipt.get("catalog_authority_hash") != _stable_mapping_hash(authority_payload)
        or receipt.get("evidence_id") != evidence_id
        or receipt.get("representation_id") != representation_id
        or receipt.get("evidence_content_hash") != catalog_evidence_hashes[selected_authority_index]
    ):
        return False
    canonical_length = _mapping(catalog_receipt.get("evidence_canonical_lengths")).get(evidence_id)
    packed_length = _mapping(catalog_receipt.get("representation_packed_lengths")).get(
        representation_id
    )
    if (
        not isinstance(canonical_length, int)
        or isinstance(canonical_length, bool)
        or not isinstance(packed_length, int)
        or isinstance(packed_length, bool)
        or int(receipt["canonical_scope_end"]) > canonical_length
        or int(receipt["packed_scope_end"]) > packed_length
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != member_count
        or len(proven_items) != member_count
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "list"
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    if (
        any(item not in catalog_witness_ids for item in witness_ids)
        or [catalog_witness_ids.index(item) for item in witness_ids] != witness_ordinals
    ):
        return False
    catalog_maps = {
        "witness_hashes": witness_hashes,
        "witness_span_ref_hashes": span_ref_hashes,
        "witness_canonical_span_hashes": canonical_member_hashes,
        "witness_packed_span_hashes": packed_member_hashes,
        "witness_canonical_starts": canonical_member_starts,
        "witness_canonical_ends": canonical_member_ends,
        "witness_packed_starts": packed_member_starts,
        "witness_packed_ends": packed_member_ends,
    }
    for mapping_key, expected_values in catalog_maps.items():
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if [mapping.get(item) for item in witness_ids] != expected_values:
            return False
    for mapping_key, expected_value in (
        ("witness_evidence_ids", evidence_id),
        ("witness_evidence_content_hashes", receipt.get("evidence_content_hash")),
        ("witness_representation_ids", representation_id),
    ):
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if any(mapping.get(item) != expected_value for item in witness_ids):
            return False
    atom_kind_map = _mapping(catalog_receipt.get("witness_atom_kinds"))
    if [atom_kind_map.get(item) for item in witness_ids] != member_atom_kinds:
        return False

    for index, (claim_id, witness_id, witness_hash) in enumerate(
        zip(supporting_claim_ids, witness_ids, witness_hashes, strict=True)
    ):
        claim = claim_by_id.get(claim_id)
        item = proven_items[index]
        if (
            claim is None
            or claim.get("obligation_id") != obligation_id
            or _tokens(claim.get("witness_ids")) != [witness_id]
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hash
            or item.get("segmentation_method") != segmentation_methods[index]
        ):
            return False
    return True


def _source_scoped_bullet_list_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """独立重放完整 authority 中唯一、显式有界的 bullet 列表。"""

    marker_style = receipt.get("marker_style")
    if (
        not _complete_witness_catalog_identity_is_valid(catalog_receipt)
        or set(receipt) != _SOURCE_SCOPED_BULLET_CLOSURE_FIELDS
        or receipt.get("schema_version") != "query-anchored-scoped-bullet-list-closure-receipt.v1"
        or receipt.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
        or receipt.get("anchor_kind")
        not in {
            "title_closed_intro",
            "ordered_closed_intro",
            "prose_closed_intro",
            "bounded_title",
        }
        or not isinstance(marker_style, str)
        or re.fullmatch(r"bullet:[-*•+]", marker_style) is None
    ):
        return False
    if (
        any(
            _token(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_FIELDS - {"marker_style"}
        )
        or any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_FIELDS
        )
    ):
        return False
    optional_title_values = tuple(
        receipt.get(key)
        for key in (
            "canonical_title_hash",
            "packed_title_hash",
            "canonical_title_start",
            "canonical_title_end",
            "packed_title_start",
            "packed_title_end",
        )
    )
    if any(item is None for item in optional_title_values) and not all(
        item is None for item in optional_title_values
    ):
        return False
    if optional_title_values[0] is not None and (
        any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_BULLET_CLOSURE_OPTIONAL_INTEGER_FIELDS
        )
    ):
        return False

    token_arrays = {
        key: _exact_tokens(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_TOKEN_ARRAY_FIELDS
    }
    hash_arrays = {
        key: _exact_hashes(
            receipt.get(key),
            limit=(256 if key.startswith("catalog_") or key == "anchor_match_key_hashes" else 64),
        )
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_HASH_ARRAY_FIELDS
    }
    integer_arrays = {
        key: _exact_integers(receipt.get(key), limit=64)
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_INTEGER_ARRAY_FIELDS
    }
    if (
        any(item is None for item in token_arrays.values())
        or any(item is None for item in hash_arrays.values())
        or any(item is None for item in integer_arrays.values())
    ):
        return False

    witness_ids = token_arrays["witness_ids"] or []
    catalog_evidence_ids = token_arrays["catalog_evidence_ids"] or []
    catalog_representation_ids = token_arrays["catalog_representation_ids"] or []
    witness_hashes = hash_arrays["witness_hashes"] or []
    span_ref_hashes = hash_arrays["span_ref_hashes"] or []
    catalog_evidence_hashes = hash_arrays["catalog_evidence_content_hashes"] or []
    catalog_packed_hashes = hash_arrays["catalog_packed_text_hashes"] or []
    anchor_match_key_hashes = hash_arrays["anchor_match_key_hashes"] or []
    canonical_item_hashes = hash_arrays["canonical_item_hashes"] or []
    packed_item_hashes = hash_arrays["packed_item_hashes"] or []
    canonical_item_body_hashes = hash_arrays["canonical_item_body_hashes"] or []
    packed_item_body_hashes = hash_arrays["packed_item_body_hashes"] or []
    witness_ordinals = integer_arrays["witness_ordinals"] or []
    canonical_item_starts = integer_arrays["canonical_item_starts"] or []
    canonical_item_ends = integer_arrays["canonical_item_ends"] or []
    packed_item_starts = integer_arrays["packed_item_starts"] or []
    packed_item_ends = integer_arrays["packed_item_ends"] or []
    item_count = int(receipt["item_count"])
    anchor_match_key_count = int(receipt["anchor_match_key_count"])
    representation_count = int(receipt["catalog_representation_count"])
    selected_authority_index = int(receipt["selected_authority_index"])
    member_arrays = (
        witness_ids,
        witness_hashes,
        witness_ordinals,
        span_ref_hashes,
        canonical_item_hashes,
        packed_item_hashes,
        canonical_item_body_hashes,
        packed_item_body_hashes,
        canonical_item_starts,
        canonical_item_ends,
        packed_item_starts,
        packed_item_ends,
    )
    authority_arrays = (
        catalog_evidence_ids,
        catalog_representation_ids,
        catalog_evidence_hashes,
        catalog_packed_hashes,
    )
    if (
        not 2 <= item_count <= 64
        or any(len(item) != item_count for item in member_arrays)
        or len(set(witness_ids)) != item_count
        or witness_ordinals != sorted(set(witness_ordinals))
        or canonical_item_starts != sorted(canonical_item_starts)
        or packed_item_starts != sorted(packed_item_starts)
        or not 2 <= anchor_match_key_count <= 256
        or len(anchor_match_key_hashes) != anchor_match_key_count
        or receipt.get("anchor_match_hash") != _stable_sequence_hash(anchor_match_key_hashes)
        or not 1 <= representation_count <= 256
        or any(len(item) != representation_count for item in authority_arrays)
        or len(set(catalog_evidence_ids)) != representation_count
        or len(set(catalog_representation_ids)) != representation_count
        or not 0 <= selected_authority_index < representation_count
        or receipt.get("authority_candidate_count") != 1
        or catalog_receipt.get("context_representation_count") != representation_count
        or int(receipt["canonical_scope_start"]) != int(receipt["canonical_anchor_start"])
        or int(receipt["packed_scope_start"]) != int(receipt["packed_anchor_start"])
        or int(receipt["canonical_anchor_start"]) >= int(receipt["canonical_anchor_end"])
        or int(receipt["packed_anchor_start"]) >= int(receipt["packed_anchor_end"])
        or int(receipt["canonical_scope_start"]) >= int(receipt["canonical_scope_end"])
        or int(receipt["packed_scope_start"]) >= int(receipt["packed_scope_end"])
    ):
        return False
    if any(
        start < int(receipt["canonical_anchor_end"])
        or end > int(receipt["canonical_scope_end"])
        or start >= end
        for start, end in zip(canonical_item_starts, canonical_item_ends, strict=True)
    ) or any(
        start < int(receipt["packed_anchor_end"])
        or end > int(receipt["packed_scope_end"])
        or start >= end
        for start, end in zip(packed_item_starts, packed_item_ends, strict=True)
    ):
        return False
    if not (
        int(receipt["canonical_local_section_start"])
        <= int(receipt["canonical_scope_start"])
        < int(receipt["canonical_scope_end"])
        <= int(receipt["canonical_local_section_end"])
    ) or not (
        int(receipt["packed_local_section_start"])
        <= int(receipt["packed_scope_start"])
        < int(receipt["packed_scope_end"])
        <= int(receipt["packed_local_section_end"])
    ):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_SCOPED_BULLET_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qblc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _scoped_bullet_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False
    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1:
        return False

    representation_hashes = _mapping(catalog_receipt.get("representation_packed_hashes"))
    evidence_hashes = _mapping(catalog_receipt.get("evidence_canonical_hashes"))
    if any(
        representation_hashes.get(representation_id) != packed_hash
        for representation_id, packed_hash in zip(
            catalog_representation_ids,
            catalog_packed_hashes,
            strict=True,
        )
    ) or any(
        evidence_hashes.get(evidence_id) != evidence_hash
        for evidence_id, evidence_hash in zip(
            catalog_evidence_ids,
            catalog_evidence_hashes,
            strict=True,
        )
    ):
        return False
    authority_payload = {
        "evidence_ids": catalog_evidence_ids,
        "representation_ids": catalog_representation_ids,
        "canonical_hashes": catalog_evidence_hashes,
        "packed_hashes": catalog_packed_hashes,
    }
    evidence_id = catalog_evidence_ids[selected_authority_index]
    representation_id = catalog_representation_ids[selected_authority_index]
    if (
        receipt.get("catalog_authority_hash") != _stable_mapping_hash(authority_payload)
        or receipt.get("evidence_id") != evidence_id
        or receipt.get("representation_id") != representation_id
        or receipt.get("evidence_content_hash") != catalog_evidence_hashes[selected_authority_index]
    ):
        return False
    canonical_length = _mapping(catalog_receipt.get("evidence_canonical_lengths")).get(evidence_id)
    packed_length = _mapping(catalog_receipt.get("representation_packed_lengths")).get(
        representation_id
    )
    if (
        not isinstance(canonical_length, int)
        or isinstance(canonical_length, bool)
        or not isinstance(packed_length, int)
        or isinstance(packed_length, bool)
        or int(receipt["canonical_local_section_end"]) > canonical_length
        or int(receipt["packed_local_section_end"]) > packed_length
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != item_count
        or len(proven_items) != item_count
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "list"
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    if (
        any(item not in catalog_witness_ids for item in witness_ids)
        or [catalog_witness_ids.index(item) for item in witness_ids] != witness_ordinals
    ):
        return False
    catalog_maps = {
        "witness_hashes": witness_hashes,
        "witness_span_ref_hashes": span_ref_hashes,
        "witness_canonical_span_hashes": canonical_item_hashes,
        "witness_packed_span_hashes": packed_item_hashes,
        "witness_canonical_starts": canonical_item_starts,
        "witness_canonical_ends": canonical_item_ends,
        "witness_packed_starts": packed_item_starts,
        "witness_packed_ends": packed_item_ends,
    }
    for mapping_key, expected_values in catalog_maps.items():
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if [mapping.get(item) for item in witness_ids] != expected_values:
            return False
    for mapping_key, expected_value in (
        ("witness_evidence_ids", evidence_id),
        ("witness_evidence_content_hashes", receipt.get("evidence_content_hash")),
        ("witness_representation_ids", representation_id),
        ("witness_atom_kinds", "list_item"),
    ):
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if any(mapping.get(item) != expected_value for item in witness_ids):
            return False

    for index, (claim_id, witness_id, witness_hash) in enumerate(
        zip(supporting_claim_ids, witness_ids, witness_hashes, strict=True)
    ):
        claim = claim_by_id.get(claim_id)
        item = proven_items[index]
        if (
            claim is None
            or claim.get("obligation_id") != obligation_id
            or _tokens(claim.get("witness_ids")) != [witness_id]
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hash
            or item.get("segmentation_method") != "top_level_list_entry.v1"
        ):
            return False
    return True


def _source_scoped_table_selection_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """独立重放完整 authority 中显式 selector 选出的全部表格行。"""

    if (
        not _complete_witness_catalog_identity_is_valid(catalog_receipt)
        or set(receipt) != _SOURCE_SCOPED_TABLE_CLOSURE_FIELDS
        or receipt.get("schema_version")
        != "query-anchored-scoped-table-selection-closure-receipt.v2"
        or receipt.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
        or receipt.get("table_parse_mode") not in {"standard", "ocr_header_data_before_separator"}
    ):
        return False
    if (
        any(
            _token(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_FIELDS
        )
        or any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_FIELDS
        )
    ):
        return False

    token_arrays = {
        key: _exact_tokens(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_TOKEN_ARRAY_FIELDS
    }
    hash_arrays = {
        key: _exact_hashes(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_HASH_ARRAY_FIELDS
    }
    integer_arrays = {
        key: _exact_integers(receipt.get(key), limit=64)
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_INTEGER_ARRAY_FIELDS
    }
    if (
        any(item is None for item in token_arrays.values())
        or any(item is None for item in hash_arrays.values())
        or any(item is None for item in integer_arrays.values())
    ):
        return False

    witness_ids = token_arrays["witness_ids"] or []
    catalog_evidence_ids = token_arrays["catalog_evidence_ids"] or []
    catalog_representation_ids = token_arrays["catalog_representation_ids"] or []
    witness_hashes = hash_arrays["witness_hashes"] or []
    span_ref_hashes = hash_arrays["span_ref_hashes"] or []
    catalog_evidence_hashes = hash_arrays["catalog_evidence_content_hashes"] or []
    catalog_packed_hashes = hash_arrays["catalog_packed_text_hashes"] or []
    selector_hashes = hash_arrays["selector_hashes"] or []
    canonical_row_hashes = hash_arrays["canonical_row_hashes"] or []
    packed_row_hashes = hash_arrays["packed_row_hashes"] or []
    witness_ordinals = integer_arrays["witness_ordinals"] or []
    selector_occurrence_counts = integer_arrays["selector_occurrence_counts"] or []
    canonical_cell_counts = integer_arrays["catalog_canonical_selector_cell_counts"] or []
    packed_cell_counts = integer_arrays["catalog_packed_selector_cell_counts"] or []
    selected_row_indices = integer_arrays["selected_row_indices"] or []
    canonical_row_starts = integer_arrays["canonical_row_starts"] or []
    canonical_row_ends = integer_arrays["canonical_row_ends"] or []
    packed_row_starts = integer_arrays["packed_row_starts"] or []
    packed_row_ends = integer_arrays["packed_row_ends"] or []
    selected_row_count = int(receipt["selected_row_count"])
    selector_count = int(receipt["selector_count"])
    representation_count = int(receipt["catalog_representation_count"])
    selected_authority_index = int(receipt["selected_authority_index"])
    table_data_row_count = int(receipt["table_data_row_count"])
    member_arrays = (
        witness_ids,
        witness_hashes,
        witness_ordinals,
        span_ref_hashes,
        selected_row_indices,
        canonical_row_hashes,
        packed_row_hashes,
        canonical_row_starts,
        canonical_row_ends,
        packed_row_starts,
        packed_row_ends,
    )
    selector_arrays = (
        selector_hashes,
        selector_occurrence_counts,
        canonical_cell_counts,
        packed_cell_counts,
    )
    authority_arrays = (
        catalog_evidence_ids,
        catalog_representation_ids,
        catalog_evidence_hashes,
        catalog_packed_hashes,
    )
    if (
        not 1 <= selected_row_count <= 64
        or any(len(item) != selected_row_count for item in member_arrays)
        or len(set(witness_ids)) != selected_row_count
        or witness_ordinals != sorted(set(witness_ordinals))
        or selected_row_indices != sorted(set(selected_row_indices))
        or not 1 <= selector_count <= 64
        or any(len(item) != selector_count for item in selector_arrays)
        or any(count < 1 for count in selector_occurrence_counts)
        or canonical_cell_counts != selector_occurrence_counts
        or packed_cell_counts != selector_occurrence_counts
        or not 1 <= representation_count <= 256
        or any(len(item) != representation_count for item in authority_arrays)
        or len(set(catalog_evidence_ids)) != representation_count
        or len(set(catalog_representation_ids)) != representation_count
        or not 0 <= selected_authority_index < representation_count
        or receipt.get("authority_candidate_count") != 1
        or catalog_receipt.get("context_representation_count") != representation_count
        or int(receipt["table_column_count"]) < 2
        or table_data_row_count < 1
        or int(receipt["selector_column_index"]) >= int(receipt["table_column_count"])
        or any(index >= table_data_row_count for index in selected_row_indices)
        or int(receipt["canonical_table_start"]) >= int(receipt["canonical_table_end"])
        or int(receipt["packed_table_start"]) >= int(receipt["packed_table_end"])
    ):
        return False
    if any(
        start < int(receipt["canonical_table_start"])
        or end > int(receipt["canonical_table_end"])
        or start >= end
        for start, end in zip(canonical_row_starts, canonical_row_ends, strict=True)
    ) or any(
        start < int(receipt["packed_table_start"])
        or end > int(receipt["packed_table_end"])
        or start >= end
        for start, end in zip(packed_row_starts, packed_row_ends, strict=True)
    ):
        return False
    if receipt.get("selector_values_hash") != _stable_sequence_hash(selector_hashes):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_SCOPED_TABLE_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qtsc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _scoped_table_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False
    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1:
        return False

    representation_hashes = _mapping(catalog_receipt.get("representation_packed_hashes"))
    evidence_hashes = _mapping(catalog_receipt.get("evidence_canonical_hashes"))
    if any(
        representation_hashes.get(representation_id) != packed_hash
        for representation_id, packed_hash in zip(
            catalog_representation_ids,
            catalog_packed_hashes,
            strict=True,
        )
    ) or any(
        evidence_hashes.get(evidence_id) != evidence_hash
        for evidence_id, evidence_hash in zip(
            catalog_evidence_ids,
            catalog_evidence_hashes,
            strict=True,
        )
    ):
        return False
    authority_payload = {
        "evidence_ids": catalog_evidence_ids,
        "representation_ids": catalog_representation_ids,
        "canonical_hashes": catalog_evidence_hashes,
        "packed_hashes": catalog_packed_hashes,
    }
    evidence_id = catalog_evidence_ids[selected_authority_index]
    representation_id = catalog_representation_ids[selected_authority_index]
    if (
        receipt.get("catalog_authority_hash") != _stable_mapping_hash(authority_payload)
        or receipt.get("evidence_id") != evidence_id
        or receipt.get("representation_id") != representation_id
        or receipt.get("evidence_content_hash") != catalog_evidence_hashes[selected_authority_index]
    ):
        return False
    canonical_length = _mapping(catalog_receipt.get("evidence_canonical_lengths")).get(evidence_id)
    packed_length = _mapping(catalog_receipt.get("representation_packed_lengths")).get(
        representation_id
    )
    if (
        not isinstance(canonical_length, int)
        or isinstance(canonical_length, bool)
        or not isinstance(packed_length, int)
        or isinstance(packed_length, bool)
        or int(receipt["canonical_table_end"]) > canonical_length
        or int(receipt["packed_table_end"]) > packed_length
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != selected_row_count
        or len(proven_items) != selected_row_count
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "list"
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    if (
        any(item not in catalog_witness_ids for item in witness_ids)
        or [catalog_witness_ids.index(item) for item in witness_ids] != witness_ordinals
    ):
        return False
    catalog_maps = {
        "witness_hashes": witness_hashes,
        "witness_span_ref_hashes": span_ref_hashes,
        "witness_canonical_span_hashes": canonical_row_hashes,
        "witness_packed_span_hashes": packed_row_hashes,
        "witness_canonical_starts": canonical_row_starts,
        "witness_canonical_ends": canonical_row_ends,
        "witness_packed_starts": packed_row_starts,
        "witness_packed_ends": packed_row_ends,
    }
    for mapping_key, expected_values in catalog_maps.items():
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if [mapping.get(item) for item in witness_ids] != expected_values:
            return False
    for mapping_key, expected_value in (
        ("witness_evidence_ids", evidence_id),
        ("witness_evidence_content_hashes", receipt.get("evidence_content_hash")),
        ("witness_representation_ids", representation_id),
        ("witness_atom_kinds", "table_data_row"),
    ):
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if any(mapping.get(item) != expected_value for item in witness_ids):
            return False

    for index, (claim_id, witness_id, witness_hash) in enumerate(
        zip(supporting_claim_ids, witness_ids, witness_hashes, strict=True)
    ):
        claim = claim_by_id.get(claim_id)
        item = proven_items[index]
        if (
            claim is None
            or claim.get("obligation_id") != obligation_id
            or _tokens(claim.get("witness_ids")) != [witness_id]
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hash
            or item.get("segmentation_method") != "markdown_table_data_row.v1"
        ):
            return False
    return True


def _source_scoped_procedure_list_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    answer: object,
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    """独立重放多 representation 中唯一查询锚定的局部有序流程。"""

    if (
        not _complete_witness_catalog_identity_is_valid(catalog_receipt)
        or set(receipt) != _SOURCE_SCOPED_PROCEDURE_CLOSURE_FIELDS
    ):
        return False
    marker_style = receipt.get("marker_style")
    if (
        receipt.get("schema_version") != "query-anchored-scoped-procedure-list-closure-receipt.v4"
        or receipt.get("normalization_version") != "nfkc-default-ignorable-markup-aware-token.v3"
        or receipt.get("anchor_field") not in {"source_query", "information_need"}
        or receipt.get("selection_kind")
        not in {
            "complete_sequence",
            "query_selected_continuation",
            "query_selected_branch",
            "query_selected_prefix",
        }
        or not isinstance(marker_style, str)
        or re.fullmatch(r"(?:arabic|chinese):[.)．、]", marker_style) is None
    ):
        return False
    if (
        any(
            _token(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_FIELDS
            - {"marker_style", "selection_kind"}
        )
        or any(
            _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_FIELDS
        )
        or any(
            _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_FIELDS
        )
        or any(
            receipt.get(key) is not None and _token(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_TOKEN_FIELDS
        )
        or any(
            receipt.get(key) is not None and _hash(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_HASH_FIELDS
        )
        or any(
            receipt.get(key) is not None and _integer(receipt.get(key)) != receipt.get(key)
            for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_OPTIONAL_INTEGER_FIELDS
        )
    ):
        return False

    token_arrays = {
        key: _exact_tokens(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_TOKEN_ARRAY_FIELDS
    }
    hash_arrays = {
        key: _exact_hashes(
            receipt.get(key),
            limit=256 if key.startswith("catalog_") else 64,
        )
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_HASH_ARRAY_FIELDS
    }
    integer_arrays = {
        key: _exact_integers(receipt.get(key))
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_INTEGER_ARRAY_FIELDS
    }
    if (
        any(item is None for item in token_arrays.values())
        or any(item is None for item in hash_arrays.values())
        or any(item is None for item in integer_arrays.values())
    ):
        return False
    witness_ids = token_arrays["witness_ids"] or []
    witness_hashes = hash_arrays["witness_hashes"] or []
    span_ref_hashes = hash_arrays["span_ref_hashes"] or []
    canonical_item_hashes = hash_arrays["canonical_item_hashes"] or []
    packed_item_hashes = hash_arrays["packed_item_hashes"] or []
    witness_ordinals = integer_arrays["witness_ordinals"] or []
    item_ordinals = integer_arrays["item_ordinals"] or []
    canonical_item_starts = integer_arrays["canonical_item_starts"] or []
    canonical_item_ends = integer_arrays["canonical_item_ends"] or []
    packed_item_starts = integer_arrays["packed_item_starts"] or []
    packed_item_ends = integer_arrays["packed_item_ends"] or []
    anchor_match_key_hashes = hash_arrays["anchor_match_key_hashes"] or []
    catalog_evidence_ids = token_arrays["catalog_evidence_ids"] or []
    catalog_representation_ids = token_arrays["catalog_representation_ids"] or []
    catalog_evidence_hashes = hash_arrays["catalog_evidence_content_hashes"] or []
    catalog_packed_hashes = hash_arrays["catalog_packed_text_hashes"] or []
    item_count = int(receipt["item_count"])
    representation_count = int(receipt["catalog_representation_count"])
    selected_index = int(receipt["selected_authority_index"])
    prefix_fields = (
        receipt.get("prefix_trigger_hash"),
        receipt.get("excluded_successor_ordinal"),
        receipt.get("excluded_successor_hash"),
        receipt.get("boundary_rule_version"),
    )
    item_arrays = (
        witness_ids,
        witness_hashes,
        span_ref_hashes,
        canonical_item_hashes,
        packed_item_hashes,
        witness_ordinals,
        item_ordinals,
        canonical_item_starts,
        canonical_item_ends,
        packed_item_starts,
        packed_item_ends,
    )
    authority_arrays = (
        catalog_evidence_ids,
        catalog_representation_ids,
        catalog_evidence_hashes,
        catalog_packed_hashes,
    )
    if (
        not 2 <= item_count <= 64
        or any(len(item) != item_count for item in item_arrays)
        or len(set(witness_ids)) != item_count
        or witness_ordinals != sorted(set(witness_ordinals))
        or item_ordinals != list(range(item_ordinals[0], item_ordinals[0] + item_count))
        or not anchor_match_key_hashes
        or len(anchor_match_key_hashes) != len(set(anchor_match_key_hashes))
        or (receipt.get("selection_kind") == "complete_sequence" and item_ordinals[0] != 1)
        or (
            receipt.get("selection_kind") == "query_selected_continuation" and item_ordinals[0] == 1
        )
        or (
            receipt.get("selection_kind") == "query_selected_prefix"
            and (
                any(item is None for item in prefix_fields)
                or item_ordinals[0] != 1
                or receipt.get("excluded_successor_ordinal") != item_ordinals[-1] + 1
                or receipt.get("boundary_rule_version") != "explicit-query-before-successor.v1"
            )
        )
        or (
            receipt.get("selection_kind") != "query_selected_prefix"
            and any(item is not None for item in prefix_fields)
        )
        or not 1 <= representation_count <= 256
        or any(len(item) != representation_count for item in authority_arrays)
        or len(set(catalog_evidence_ids)) != representation_count
        or len(set(catalog_representation_ids)) != representation_count
        or not 0 <= selected_index < representation_count
        or receipt.get("authority_candidate_count") != 1
        or catalog_receipt.get("context_representation_count") != representation_count
        or int(receipt["canonical_block_start"]) >= int(receipt["canonical_block_end"])
        or int(receipt["packed_block_start"]) >= int(receipt["packed_block_end"])
        or int(receipt["canonical_anchor_start"]) >= int(receipt["canonical_anchor_end"])
        or int(receipt["packed_anchor_start"]) >= int(receipt["packed_anchor_end"])
    ):
        return False
    if any(
        start < int(receipt["canonical_block_start"])
        or end > int(receipt["canonical_block_end"])
        or start >= end
        for start, end in zip(canonical_item_starts, canonical_item_ends, strict=True)
    ) or any(
        start < int(receipt["packed_block_start"])
        or end > int(receipt["packed_block_end"])
        or start >= end
        for start, end in zip(packed_item_starts, packed_item_ends, strict=True)
    ):
        return False

    payload = {
        key: receipt[key]
        for key in _SOURCE_SCOPED_PROCEDURE_CLOSURE_FIELDS - {"receipt_id", "receipt_hash"}
    }
    expected_hash = _stable_mapping_hash(payload)
    expected_id = "qplc_" + expected_hash.removeprefix("sha256:")[:40]
    if receipt.get("receipt_hash") != expected_hash or receipt.get("receipt_id") != expected_id:
        return False
    if not _scoped_procedure_receipt_matches_proof_authority(
        receipt,
        catalog_receipt=catalog_receipt,
        question_receipt=question_receipt,
        authority=proof_authority,
    ):
        return False
    authoritative_receipts = _items(catalog_receipt.get("source_enumeration_closure_receipts"))
    if sum(dict(item) == dict(receipt) for item in authoritative_receipts) != 1:
        return False

    representation_hashes = _mapping(catalog_receipt.get("representation_packed_hashes"))
    evidence_hashes = _mapping(catalog_receipt.get("evidence_canonical_hashes"))
    if any(
        representation_hashes.get(representation_id) != packed_hash
        for representation_id, packed_hash in zip(
            catalog_representation_ids,
            catalog_packed_hashes,
            strict=True,
        )
    ) or any(
        evidence_hashes.get(evidence_id) != evidence_hash
        for evidence_id, evidence_hash in zip(
            catalog_evidence_ids,
            catalog_evidence_hashes,
            strict=True,
        )
    ):
        return False
    authority_payload = {
        "evidence_ids": catalog_evidence_ids,
        "representation_ids": catalog_representation_ids,
        "canonical_hashes": catalog_evidence_hashes,
        "packed_hashes": catalog_packed_hashes,
    }
    evidence_id = catalog_evidence_ids[selected_index]
    representation_id = catalog_representation_ids[selected_index]
    if (
        receipt.get("catalog_authority_hash") != _stable_mapping_hash(authority_payload)
        or receipt.get("evidence_id") != evidence_id
        or receipt.get("representation_id") != representation_id
        or receipt.get("evidence_content_hash") != catalog_evidence_hashes[selected_index]
    ):
        return False

    obligation_id = _token(obligation.get("obligation_id"))
    supporting_claim_ids = _exact_tokens(obligation.get("supporting_claim_ids"))
    proven_items = _items(obligation.get("proven_items"))
    if (
        obligation_id is None
        or supporting_claim_ids is None
        or len(supporting_claim_ids) != item_count
        or len(proven_items) != item_count
        or receipt.get("question_contract_id") != question_receipt.get("contract_id")
        or receipt.get("question_contract_hash") != question_receipt.get("contract_hash")
        or receipt.get("obligation_id") != obligation_id
        or receipt.get("obligation_hash") != contract.get("obligation_hash")
        or receipt.get("catalog_id") != catalog_receipt.get("catalog_id")
        or receipt.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or _normalized_status(catalog_receipt.get("universe_status")) != "complete"
        or _normalized_status(contract.get("closure")) != "open_enumeration"
        or _normalized_status(contract.get("required_output_form")) != "ordered_steps"
    ):
        return False

    catalog_witness_ids = _tokens(catalog_receipt.get("witness_ids"))
    if (
        any(item not in catalog_witness_ids for item in witness_ids)
        or [catalog_witness_ids.index(item) for item in witness_ids] != witness_ordinals
    ):
        return False
    catalog_maps = {
        "witness_hashes": witness_hashes,
        "witness_span_ref_hashes": span_ref_hashes,
        "witness_canonical_span_hashes": canonical_item_hashes,
        "witness_packed_span_hashes": packed_item_hashes,
        "witness_canonical_starts": canonical_item_starts,
        "witness_canonical_ends": canonical_item_ends,
        "witness_packed_starts": packed_item_starts,
        "witness_packed_ends": packed_item_ends,
    }
    for mapping_key, expected_values in catalog_maps.items():
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if [mapping.get(item) for item in witness_ids] != expected_values:
            return False
    for mapping_key, expected_value in (
        ("witness_evidence_ids", evidence_id),
        ("witness_evidence_content_hashes", receipt.get("evidence_content_hash")),
        ("witness_representation_ids", representation_id),
        ("witness_atom_kinds", "list_item"),
    ):
        mapping = _mapping(catalog_receipt.get(mapping_key))
        if any(mapping.get(item) != expected_value for item in witness_ids):
            return False

    if not isinstance(answer, str):
        return False
    for index, (claim_id, witness_id, witness_hash) in enumerate(
        zip(supporting_claim_ids, witness_ids, witness_hashes, strict=True)
    ):
        claim = claim_by_id.get(claim_id)
        item = proven_items[index]
        if (
            claim is None
            or claim.get("obligation_id") != obligation_id
            or _tokens(claim.get("witness_ids")) != [witness_id]
            or item.get("obligation_id") != obligation_id
            or item.get("claim_id") != claim_id
            or item.get("witness_id") != witness_id
            or item.get("witness_hash") != witness_hash
            or item.get("segmentation_method") != "top_level_list_entry.v1"
        ):
            return False
        answer_span = _mapping(claim.get("answer_span"))
        answer_start = _integer(answer_span.get("start"))
        answer_end = _integer(answer_span.get("end"), minimum=1)
        if (
            answer_start is None
            or answer_end is None
            or answer_start >= answer_end
            or answer_end > len(answer)
        ):
            return False
        rendered = answer[answer_start:answer_end]
        content_length = packed_item_ends[index] - packed_item_starts[index]
        content = rendered[:content_length]
        suffix = rendered[content_length:]
        parsed = _procedure_ordered_item(content)
        if (
            _text_hash(rendered) != claim.get("rendered_text_hash")
            or _text_hash(content) != packed_item_hashes[index]
            or re.fullmatch(r" \[[^\]\r\n]+\]", suffix) is None
            or parsed is None
            or parsed[0] != item_ordinals[index]
            or parsed[1] != marker_style
        ):
            return False
    return True


def _source_enumeration_closure_is_valid(
    receipt: Mapping[str, Any],
    *,
    obligation: Mapping[str, Any],
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    answer: object,
    proof_authority: ProofEvaluationAuthority | None,
) -> bool:
    if receipt.get("schema_version") == "query-anchored-inline-action-chain-closure-receipt.v1":
        return _source_inline_action_chain_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-scoped-fact-set-closure-receipt.v1":
        return _source_scoped_fact_set_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-scoped-bullet-list-closure-receipt.v1":
        return _source_scoped_bullet_list_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-scoped-table-selection-closure-receipt.v2":
        return _source_scoped_table_selection_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-scoped-procedure-list-closure-receipt.v4":
        return _source_scoped_procedure_list_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            answer=answer,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-table-row-closure-receipt.v4":
        return _source_table_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            proof_authority=proof_authority,
        )
    if receipt.get("schema_version") == "query-anchored-procedure-list-closure-receipt.v1":
        return _source_procedure_list_closure_is_valid(
            receipt,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            answer=answer,
            proof_authority=proof_authority,
        )
    return False


def _obligation_graph_reasons(
    obligation: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    question_receipt: Mapping[str, Any],
    catalog_receipt: Mapping[str, Any],
    claim_by_id: Mapping[str, Mapping[str, Any]],
    known_witness_ids: set[str],
    witness_atom_kinds: Mapping[str, str],
    semantically_closed_ids: set[str],
    answer: object,
    proof_authority: ProofEvaluationAuthority | None,
) -> list[str]:
    """重放 v3 义务的输出形态、item、闭包与 claim 引用关系。"""

    reasons: list[str] = []
    obligation_id = _token(obligation.get("obligation_id"))
    closure = _normalized_status(obligation.get("closure"))
    output_form = _normalized_status(obligation.get("required_output_form"))
    expected_count = _integer(obligation.get("expected_item_count"), minimum=1)
    if (
        not contract
        or obligation_id != _token(contract.get("obligation_id"))
        or closure != _normalized_status(contract.get("closure"))
        or output_form != _normalized_status(contract.get("required_output_form"))
        or expected_count != _integer(contract.get("expected_count"), minimum=1)
    ):
        reasons.append("answer_proof_obligation_contract_mismatch")

    supporting = _tokens(obligation.get("supporting_claim_ids"))
    items = _items(obligation.get("proven_items"))
    source_closure_value = obligation.get("source_enumeration_closure_receipt")
    source_closure = _mapping(source_closure_value)
    inline_action_closure = (
        source_closure.get("schema_version")
        == "query-anchored-inline-action-chain-closure-receipt.v1"
    )
    declared_item_count = _integer(obligation.get("proven_item_count"))
    if declared_item_count is None or declared_item_count != len(items):
        reasons.append("answer_proof_item_count_invalid")
    item_ids = [_token(item.get("item_ref_id")) for item in items]
    item_hashes = [_hash(item.get("item_ref_hash")) for item in items]
    normalized_hashes = [_hash(item.get("normalized_item_hash")) for item in items]
    if (
        len({item for item in item_ids if item is not None}) != len(items)
        or len({item for item in item_hashes if item is not None}) != len(items)
        or (
            not inline_action_closure
            and len({item for item in normalized_hashes if item is not None}) != len(items)
        )
    ):
        reasons.append("answer_proof_item_identity_invalid")
    item_claim_ids: list[str] = []
    for item in items:
        claim_id = _token(item.get("claim_id"))
        witness_id = _token(item.get("witness_id"))
        claim = claim_by_id.get(claim_id or "")
        start = _integer(item.get("relative_start"))
        end = _integer(item.get("relative_end"), minimum=1)
        segmentation_method = _token(item.get("segmentation_method"))
        atom_kind = witness_atom_kinds.get(witness_id or "")
        structured_method_valid = (
            segmentation_method == "single_witness_atom.v1"
            or segmentation_method == "open_enumeration_witness_atom.v1"
            and closure == "open_enumeration"
            or segmentation_method == "top_level_list_entry.v1"
            and atom_kind == "list_item"
            or segmentation_method == "markdown_table_data_row.v1"
            and atom_kind == "table_data_row"
            or segmentation_method == "markdown_table_projected_cell_item.v1"
            and closure == "closed_count"
            and atom_kind == "table_data_row"
            or segmentation_method == "inline_action_span.v1"
            and closure == "open_enumeration"
            and atom_kind in {"sentence", "list_item"}
        )
        if (
            item.get("schema_version") != "proven-item-ref.v1"
            or _hash(item.get("witness_hash")) is None
            or _hash(item.get("raw_span_hash")) is None
            or not structured_method_valid
            or item.get("obligation_id") != obligation_id
            or claim_id not in supporting
            or claim is None
            or claim.get("obligation_id") != obligation_id
            or witness_id not in known_witness_ids
            or witness_id not in _tokens(claim.get("witness_ids"))
            or start is None
            or end is None
            or start >= end
        ):
            reasons.append("answer_proof_item_reference_invalid")
            break
        item_claim_ids.append(claim_id)

    if (
        closure == "closed_count"
        and any(
            item.get("segmentation_method")
            in {
                "markdown_table_data_row.v1",
                "markdown_table_projected_cell_item.v1",
            }
            for item in items
        )
        and not _closed_count_table_cell_items_match_proof_authority(
            obligation,
            authority=proof_authority,
        )
    ):
        reasons.append("answer_proof_closed_count_table_cell_invalid")

    status = _normalized_status(obligation.get("status"))
    outcome = _normalized_status(obligation.get("outcome"))
    closure_status = _normalized_status(obligation.get("closure_status"))
    if outcome == "answered" and closure == "open_enumeration":
        if source_closure_value is None:
            reasons.append("answer_proof_open_enumeration_source_closure_missing")
        elif not source_closure or not _source_enumeration_closure_is_valid(
            source_closure,
            obligation=obligation,
            contract=contract,
            question_receipt=question_receipt,
            catalog_receipt=catalog_receipt,
            claim_by_id=claim_by_id,
            answer=answer,
            proof_authority=proof_authority,
        ):
            reasons.append("answer_proof_open_enumeration_source_closure_invalid")
        if (
            source_closure.get("schema_version")
            in {
                "query-anchored-table-row-closure-receipt.v4",
                "query-anchored-procedure-list-closure-receipt.v1",
                "query-anchored-scoped-procedure-list-closure-receipt.v4",
                "query-anchored-scoped-table-selection-closure-receipt.v2",
                "query-anchored-scoped-bullet-list-closure-receipt.v1",
                "query-anchored-scoped-fact-set-closure-receipt.v1",
                "query-anchored-inline-action-chain-closure-receipt.v1",
            }
            and proof_authority is None
        ):
            reasons.append("answer_proof_trusted_authority_missing")
    elif source_closure_value is not None:
        reasons.append("answer_proof_open_enumeration_source_closure_unexpected")

    if outcome == "answered":
        if (
            status not in _PROVEN
            or closure_status != "proven_complete"
            or _normalized_status(obligation.get("catalog_universe_status")) != "complete"
            or not supporting
            or obligation_id not in semantically_closed_ids
        ):
            reasons.append("answer_proof_answered_closure_invalid")
        if closure == "single" and (len(supporting) != 1 or len(items) != 1):
            reasons.append("answer_proof_single_cardinality_invalid")
        elif closure == "closed_count" and (
            expected_count is None
            or len(items) != expected_count
            or len(supporting) > expected_count
            or set(item_claim_ids) != set(supporting)
        ):
            reasons.append("answer_proof_closed_cardinality_invalid")
    elif outcome == "partially_answered":
        if (
            status not in _PROVEN
            or not supporting
            or not items
            or closure_status not in {"incomplete", "indeterminate"}
        ):
            reasons.append("answer_proof_partial_obligation_invalid")
    elif outcome == "insufficient_retrieved_evidence":
        if status not in _PROVEN or supporting or items or closure_status != "not_applicable":
            reasons.append("answer_proof_local_insufficiency_invalid")
    elif outcome == "evidence_absent" and status in _PROVEN:
        reasons.append("answer_proof_exhaustive_absence_receipt_missing")
    return reasons


def answer_proof_evaluation(
    observation: Mapping[str, Any],
    *,
    proof_authority: ProofEvaluationAuthority | None = None,
) -> dict[str, Any] | None:
    """验证 v3 proof 图并返回低基数摘要；历史观测返回 ``None``。"""

    if not is_v3_observation(observation):
        return None
    proof = _mapping(observation.get("answer_proof"))
    reasons: list[str] = []
    if not proof:
        return {
            "available": False,
            "applicable": True,
            "contract_valid": False,
            "strict_eligible": False,
            "strict_graph_valid": False,
            "decision": "",
            "overall_status": "",
            "claim_count": 0,
            "supported_claim_count": 0,
            "fully_verified_claim_count": 0,
            "factual_claim_count": 0,
            "supported_factual_claim_count": 0,
            "fully_verified_factual_claim_count": 0,
            "binding_count": 0,
            "valid_binding_count": 0,
            "obligation_count": 0,
            "complete_obligation_count": 0,
            "required_obligation_count": 0,
            "reason_codes": ["answer_proof_missing"],
        }

    if proof.get("schema_version") != "answer-proof.v3":
        reasons.append("answer_proof_schema_invalid")
    if (
        _token(proof.get("proof_id")) is None
        or _hash(proof.get("proof_hash")) is None
        or not _answer_proof_identity_is_valid(proof)
    ):
        reasons.append("answer_proof_identity_invalid")
    strict_raw = proof.get("strict_eligible")
    if not isinstance(strict_raw, bool):
        reasons.append("answer_proof_strict_eligible_missing")
    strict_eligible = strict_raw is True
    decision = _normalized_status(proof.get("decision") or proof.get("disposition"))
    overall_status = _normalized_status(proof.get("overall_status") or proof.get("status"))
    applicable = not (decision in _TERMINAL_NOT_APPLICABLE or overall_status == "not_applicable")

    claims, obligations = _proof_rows(proof)
    root_relevance = _normalized_status(proof.get("relevance_status"))
    factual_claims = [item for item in claims if not _claim_is_system_notice(item)]
    supported_factual_claims = [item for item in factual_claims if _claim_support_is_proven(item)]
    fully_verified_factual_claims = [
        item
        for item in supported_factual_claims
        if _claim_relevance_is_proven(item, root_relevance=root_relevance)
    ]
    bindings = [binding for claim in claims for binding in _items(claim.get("bindings"))]
    valid_bindings = [item for item in bindings if _binding_is_proven(item)]
    resolved_obligations = [item for item in obligations if _obligation_is_resolved(item)]

    claim_id_list = [_token(item.get("claim_id")) for item in claims]
    obligation_id_list = [_token(item.get("obligation_id")) for item in obligations]
    claim_ids = {item for item in claim_id_list if item is not None}
    obligation_ids = {item for item in obligation_id_list if item is not None}
    if len(claim_ids) != len(claim_id_list):
        reasons.append("answer_proof_claim_identity_invalid")
    if len(obligation_ids) != len(obligation_id_list):
        reasons.append("answer_proof_obligation_identity_invalid")
    if any(
        not _schema_matches(item.get("schema_version"), "claim-proof.")
        or _token(item.get("claim_proof_id")) is None
        or _hash(item.get("claim_proof_hash")) is None
        or not _claim_proof_identity_is_valid(item)
        for item in claims
        if item.get("support") is None
    ):
        reasons.append("answer_proof_claim_schema_invalid")
    if any(
        not _schema_matches(item.get("schema_version"), "obligation-proof.")
        or _token(item.get("obligation_proof_id")) is None
        or _hash(item.get("obligation_proof_hash")) is None
        or not _obligation_proof_identity_is_valid(item)
        for item in obligations
        if item.get("completeness") is None
    ):
        reasons.append("answer_proof_obligation_schema_invalid")
    referenced_claims = {
        item
        for obligation in obligations
        for item in _tokens(obligation.get("supporting_claim_ids") or obligation.get("claim_ids"))
    }
    if not referenced_claims.issubset(claim_ids):
        reasons.append("answer_proof_claim_reference_invalid")
    if any(
        not _proven_item_identity_is_valid(item)
        for obligation in obligations
        for item in _items(obligation.get("proven_items"))
    ):
        reasons.append("answer_proof_item_identity_invalid")
    claim_obligations = {
        item for claim in claims if (item := _token(claim.get("obligation_id"))) is not None
    }
    if obligation_ids and not claim_obligations.issubset(obligation_ids):
        reasons.append("answer_proof_obligation_reference_invalid")

    question_receipt = _mapping(observation.get("question_contract_receipt"))
    plan_receipt = _mapping(observation.get("answer_plan_receipt"))
    reviewed_plan_receipt = _mapping(observation.get("reviewed_answer_plan_receipt"))
    catalog_receipt = _mapping(observation.get("witness_catalog_receipt"))
    review_receipt = _mapping(observation.get("answer_plan_review_receipt"))
    review_call_receipt = _mapping(observation.get("answer_review_call_receipt"))
    semantic_closure = _mapping(proof.get("semantic_review_closure"))
    if not question_receipt:
        reasons.append("answer_proof_question_contract_receipt_missing")
    elif _identity_mismatch(
        proof, "question_contract_id", question_receipt, "contract_id"
    ) or _identity_mismatch(proof, "question_contract_hash", question_receipt, "contract_hash"):
        reasons.append("answer_proof_question_contract_mismatch")
    if not plan_receipt:
        reasons.append("answer_proof_answer_plan_receipt_missing")
    elif (
        not _answer_plan_identity_is_valid(plan_receipt)
        or _identity_mismatch(proof, "answer_plan_id", plan_receipt, "plan_id")
        or _identity_mismatch(proof, "answer_plan_hash", plan_receipt, "plan_hash")
    ):
        reasons.append("answer_proof_answer_plan_mismatch")

    receipt_obligation_ids = _tokens(question_receipt.get("obligation_ids"))
    if question_receipt and receipt_obligation_ids != [
        item for item in obligation_id_list if item is not None
    ]:
        reasons.append("answer_proof_obligation_ledger_mismatch")
    plan_claims = _items(plan_receipt.get("claims"))
    if plan_receipt and [item.get("claim_id") for item in plan_claims] != [
        item for item in claim_id_list if item is not None
    ]:
        reasons.append("answer_proof_claim_ledger_mismatch")
    if plan_receipt and any(
        plan_claim.get("obligation_id") != proof_claim.get("obligation_id")
        or _tokens(plan_claim.get("witness_ids")) != _tokens(proof_claim.get("witness_ids"))
        or plan_claim.get("rendering_rule") != proof_claim.get("method")
        for plan_claim, proof_claim in zip(plan_claims, claims, strict=False)
    ):
        reasons.append("answer_proof_claim_plan_mismatch")
    if plan_receipt and _normalized_status(plan_receipt.get("decision")) != decision:
        reasons.append("answer_proof_plan_decision_mismatch")

    catalog_id = _token(proof.get("witness_catalog_id"))
    catalog_hash = _hash(proof.get("witness_catalog_hash"))
    if catalog_id is not None or catalog_hash is not None:
        if not catalog_receipt:
            reasons.append("answer_proof_witness_catalog_receipt_missing")
        elif _identity_mismatch(proof, "witness_catalog_id", catalog_receipt, "catalog_id") or (
            _identity_mismatch(proof, "witness_catalog_hash", catalog_receipt, "catalog_hash")
        ):
            reasons.append("answer_proof_witness_catalog_mismatch")
    elif catalog_receipt:
        reasons.append("answer_proof_unexpected_witness_catalog")
    known_witness_ids = set(_tokens(catalog_receipt.get("witness_ids")))
    witness_atom_kinds = {
        str(key): str(value)
        for key, value in _mapping(catalog_receipt.get("witness_atom_kinds")).items()
    }
    referenced_witness_ids = {
        witness_id for claim in claims for witness_id in _tokens(claim.get("witness_ids"))
    } | {
        witness_id
        for obligation in obligations
        for witness_id in _tokens(obligation.get("contradiction_witness_ids"))
    }
    if referenced_witness_ids and not referenced_witness_ids.issubset(known_witness_ids):
        reasons.append("answer_proof_witness_reference_invalid")
    if any(
        obligation.get("catalog_id") != catalog_receipt.get("catalog_id")
        or obligation.get("catalog_hash") != catalog_receipt.get("catalog_hash")
        or obligation.get("catalog_universe_status") != catalog_receipt.get("universe_status")
        for obligation in obligations
    ):
        reasons.append("answer_proof_obligation_catalog_mismatch")
    claim_by_id = {item: row for row in claims if (item := _token(row.get("claim_id"))) is not None}
    if any(
        (claim := claim_by_id.get(claim_id)) is None
        or claim.get("obligation_id") != obligation.get("obligation_id")
        or not _claim_support_is_proven(claim)
        for obligation in obligations
        for claim_id in _tokens(obligation.get("supporting_claim_ids"))
    ):
        reasons.append("answer_proof_obligation_support_invalid")

    retrieval_committable = overall_status in {
        "accepted",
        "accepted_partial",
        "accepted_refusal",
        "accepted_abstention",
    }
    semantic_closed_ids = set(_tokens(semantic_closure.get("closed_obligation_ids")))
    semantic_closure_valid = True
    if retrieval_committable:
        schema_version = semantic_closure.get("schema_version")
        common_receipt_valid = bool(semantic_closure) and (
            _token(semantic_closure.get("receipt_id")) is not None
            and _hash(semantic_closure.get("receipt_hash")) is not None
            and _semantic_review_identity_is_valid(semantic_closure)
            and semantic_closure.get("answer_plan_id") == proof.get("answer_plan_id")
            and semantic_closure.get("answer_plan_hash") == proof.get("answer_plan_hash")
            and semantic_closure.get("catalog_id") == proof.get("witness_catalog_id")
            and semantic_closure.get("catalog_hash") == proof.get("witness_catalog_hash")
            and _token(semantic_closure.get("review_prompt_version")) is not None
            and _token(semantic_closure.get("model_id")) is not None
            and _token(semantic_closure.get("verifier_version")) is not None
            and _token(semantic_closure.get("review_logical_call_id")) is not None
            and _hash(semantic_closure.get("review_prompt_hash")) is not None
            and _hash(semantic_closure.get("review_response_hash")) is not None
            and _hash(semantic_closure.get("review_proposal_hash")) is not None
            and semantic_closed_ids.issubset(obligation_ids)
            and review_receipt.get("proposal_hash") == semantic_closure.get("review_proposal_hash")
        )
        if schema_version == "semantic-review-closure.v2":
            semantic_closure_valid = (
                common_receipt_valid and semantic_closure.get("decision") == "accept"
            )
        elif schema_version == "semantic-review-terminal.v1":
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
            }.get(_normalized_status(semantic_closure.get("decision")))
            semantic_closure_valid = (
                common_receipt_valid
                and overall_status == "accepted_abstention"
                and decision == "refuse"
                and not semantic_closed_ids
                and expected_reasons is not None
                and semantic_closure.get("reason_code") in expected_reasons
                and reviewed_plan_receipt.get("plan_id")
                == semantic_closure.get("reviewed_answer_plan_id")
                and reviewed_plan_receipt.get("plan_hash")
                == semantic_closure.get("reviewed_answer_plan_hash")
                and _normalized_status(review_receipt.get("decision"))
                == _normalized_status(semantic_closure.get("decision"))
                and set(_tokens(review_receipt.get("closed_obligation_ids")))
                == set(_tokens(semantic_closure.get("reviewed_closed_obligation_ids")))
            )
        else:
            semantic_closure_valid = False
        logical_call_id = _token(semantic_closure.get("review_logical_call_id")) or ""
        if logical_call_id.startswith("lrv_"):
            semantic_closure_valid = (
                semantic_closure_valid and decision == "refuse" and not review_call_receipt
            )
        else:
            semantic_closure_valid = semantic_closure_valid and (
                review_call_receipt.get("logical_call_id") == logical_call_id
                and review_call_receipt.get("scene") == "answer_plan_semantic_review"
                and review_call_receipt.get("status") == "succeeded"
                and review_call_receipt.get("prompt_version")
                == semantic_closure.get("review_prompt_version")
                and review_call_receipt.get("model_id") == semantic_closure.get("model_id")
                and review_call_receipt.get("prompt_hash")
                == semantic_closure.get("review_prompt_hash")
                and review_call_receipt.get("response_hash")
                == semantic_closure.get("review_response_hash")
                and review_call_receipt.get("parsed_proposal_hash")
                == semantic_closure.get("review_proposal_hash")
            )
        if not semantic_closure_valid:
            reasons.append("answer_proof_semantic_closure_invalid")
        review_closed_ids = set(_tokens(review_receipt.get("closed_obligation_ids")))
        verification_closed_ids = set(
            _tokens(_mapping(review_receipt.get("verification")).get("closed_obligation_ids"))
        )
        expected_review_closed_ids = (
            set(_tokens(semantic_closure.get("reviewed_closed_obligation_ids")))
            if schema_version == "semantic-review-terminal.v1"
            else semantic_closed_ids
        )
        if review_closed_ids != expected_review_closed_ids or (
            verification_closed_ids and verification_closed_ids != expected_review_closed_ids
        ):
            reasons.append("answer_proof_semantic_review_ledger_mismatch")

    question_obligations = {
        item: row
        for row in _items(question_receipt.get("obligations"))
        if (item := _token(row.get("obligation_id"))) is not None
    }
    for obligation in obligations:
        obligation_id = _token(obligation.get("obligation_id"))
        if obligation.get("schema_version") != "obligation-proof.v3":
            reasons.append("answer_proof_obligation_schema_invalid")
            continue
        reasons.extend(
            _obligation_graph_reasons(
                obligation,
                contract=question_obligations.get(obligation_id or "", {}),
                question_receipt=question_receipt,
                catalog_receipt=catalog_receipt,
                claim_by_id=claim_by_id,
                known_witness_ids=known_witness_ids,
                witness_atom_kinds=witness_atom_kinds,
                semantically_closed_ids=semantic_closed_ids,
                answer=observation.get("answer"),
                proof_authority=proof_authority,
            )
        )

    if applicable and not _answer_hash_matches(observation, proof):
        reasons.append("answer_proof_answer_hash_mismatch")
    if applicable and decision not in {"refuse"} and not claims:
        reasons.append("answer_proof_claims_missing")
    if applicable and not obligations:
        reasons.append("answer_proof_obligations_missing")

    required_ids = _required_obligation_ids(observation, obligations)
    obligation_by_id = {
        item: row for row in obligations if (item := _token(row.get("obligation_id"))) is not None
    }
    required_proofs = [obligation_by_id.get(item) for item in required_ids]
    graph_contract_valid = not reasons
    strict_graph_valid = False
    if overall_status == "accepted" and decision == "answer":
        strict_graph_valid = (
            graph_contract_valid
            and semantic_closure_valid
            and semantic_closure.get("schema_version") == "semantic-review-closure.v2"
            and set(required_ids).issubset(semantic_closed_ids)
            and _normalized_status(proof.get("grounding_status")) in _PROVEN
            and root_relevance in _PROVEN
            and _normalized_status(proof.get("completeness_status")) in _PROVEN
            and len(fully_verified_factual_claims) == len(factual_claims)
            and bool(factual_claims)
            and all(
                row is not None
                and _normalized_status(row.get("status")) in _PROVEN
                and _normalized_status(row.get("outcome")) in _STRICT_ANSWER_OUTCOMES
                for row in required_proofs
            )
        )
    elif overall_status == "accepted_refusal" and decision == "refuse":
        strict_graph_valid = (
            graph_contract_valid
            and semantic_closure_valid
            and semantic_closure.get("schema_version") == "semantic-review-closure.v2"
            and not factual_claims
            and bool(required_proofs)
            and _normalized_status(catalog_receipt.get("universe_status")) == "complete"
            and all(
                row is not None
                and _normalized_status(row.get("status")) in _PROVEN
                and _normalized_status(row.get("outcome")) in _STRICT_REFUSAL_OUTCOMES
                for row in required_proofs
            )
        )

    if strict_eligible and not strict_graph_valid:
        reasons.append("answer_proof_strict_graph_invalid")
    if strict_eligible and overall_status in _ROOT_FAILURE_STATUSES:
        reasons.append("answer_proof_root_status_not_accepted")
    if strict_eligible and proof.get("failures"):
        reasons.append("answer_proof_strict_failures_present")

    contract_valid = not reasons
    return {
        "available": True,
        "applicable": applicable,
        "contract_valid": contract_valid,
        "strict_eligible": strict_eligible,
        "strict_graph_valid": strict_graph_valid,
        "decision": decision,
        "overall_status": overall_status,
        "claim_count": len(claims),
        "supported_claim_count": len(supported_factual_claims),
        "fully_verified_claim_count": len(fully_verified_factual_claims),
        "factual_claim_count": len(factual_claims),
        "supported_factual_claim_count": len(supported_factual_claims),
        "fully_verified_factual_claim_count": len(fully_verified_factual_claims),
        "binding_count": len(bindings),
        "valid_binding_count": len(valid_bindings),
        "obligation_count": len(obligations),
        "complete_obligation_count": len(resolved_obligations),
        "required_obligation_count": len(required_ids),
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def proof_generation_metrics(
    observation: Mapping[str, Any],
    *,
    proof_evaluation: Mapping[str, Any] | None = None,
    proof_authority: ProofEvaluationAuthority | None = None,
) -> dict[str, Any] | None:
    """返回 proof-first 事实支持指标；历史观测返回 ``None``。"""

    evaluation = (
        dict(proof_evaluation)
        if proof_evaluation is not None
        else answer_proof_evaluation(observation, proof_authority=proof_authority)
    )
    if evaluation is None:
        return None
    if not evaluation["applicable"]:
        return {
            "faithfulness": None,
            "claim_count": 0,
            "supported_claim_count": 0,
            "factual_claim_count": 0,
            "supported_factual_claim_count": 0,
            "unverified_factual_claim_count": 0,
            "unsupported_factual_claim_count": 0,
            "source": "answer_proof",
            "evaluation_status": "not_applicable",
        }
    claim_count = int(evaluation["factual_claim_count"])
    supported = int(evaluation["fully_verified_factual_claim_count"])
    if not evaluation["contract_valid"]:
        supported = 0
    unsupported = max(0, claim_count - supported)
    return {
        "faithfulness": supported / claim_count if claim_count else None,
        "claim_count": claim_count,
        "supported_claim_count": supported,
        "factual_claim_count": claim_count,
        "supported_factual_claim_count": supported,
        "unverified_factual_claim_count": unsupported,
        "unsupported_factual_claim_count": unsupported,
        "source": "answer_proof",
        "evaluation_status": (
            "evaluated" if evaluation["contract_valid"] else "invalid_answer_proof"
        ),
    }


def proof_citation_metrics(
    observation: Mapping[str, Any],
    *,
    proof_evaluation: Mapping[str, Any] | None = None,
    proof_authority: ProofEvaluationAuthority | None = None,
) -> dict[str, Any] | None:
    """从 proof binding/witness 回执计算引用指标，不再重解析最终文本。"""

    evaluation = (
        dict(proof_evaluation)
        if proof_evaluation is not None
        else answer_proof_evaluation(observation, proof_authority=proof_authority)
    )
    if evaluation is None:
        return None
    proof = _mapping(observation.get("answer_proof"))
    claims, _ = _proof_rows(proof)
    factual_claims = [item for item in claims if not _claim_is_system_notice(item)]
    if not evaluation["applicable"] or not factual_claims:
        return {
            "precision": None,
            "recall": None,
            "coverage": None,
            "citation_count": 0,
            "resolved_count": 0,
            "relevant_count": 0,
            "unresolved_count": 0,
            "cited_claim_count": 0,
            "evaluation_status": "not_applicable",
            "metric_source": "answer_proof",
        }
    bindings = [binding for claim in factual_claims for binding in _items(claim.get("bindings"))]
    witness_references = [
        witness_id for claim in factual_claims for witness_id in _tokens(claim.get("witness_ids"))
    ]
    known_witnesses = set(
        _tokens(_mapping(observation.get("witness_catalog_receipt")).get("witness_ids"))
    )
    citation_count = len(bindings) or len(witness_references)
    resolved_count = (
        len([item for item in bindings if _binding_is_proven(item)])
        if bindings
        else sum(1 for item in witness_references if item in known_witnesses)
    )
    root_relevance = _normalized_status(proof.get("relevance_status"))
    claim_has_citation = [
        bool(_tokens(claim.get("witness_ids")) or _items(claim.get("bindings")))
        for claim in factual_claims
    ]
    relevant_count = sum(
        len(_items(claim.get("bindings"))) or len(_tokens(claim.get("witness_ids")))
        for claim in factual_claims
        if _claim_support_is_proven(claim)
        and _claim_relevance_is_proven(claim, root_relevance=root_relevance)
    )
    if not evaluation["contract_valid"]:
        relevant_count = 0
    claim_count = len(factual_claims)
    cited_claim_count = sum(1 for value in claim_has_citation if value)
    return {
        "precision": relevant_count / citation_count if citation_count else 0.0,
        "recall": cited_claim_count / claim_count if claim_count else 0.0,
        "coverage": cited_claim_count / claim_count if claim_count else 0.0,
        "citation_count": citation_count,
        "resolved_count": resolved_count,
        "relevant_count": relevant_count,
        "unresolved_count": max(0, citation_count - resolved_count),
        "cited_claim_count": cited_claim_count,
        "evaluation_status": (
            "evaluated" if evaluation["contract_valid"] else "invalid_answer_proof"
        ),
        "metric_source": "answer_proof",
    }


__all__ = [
    "OBSERVATION_SCHEMA_V3",
    "ProofEvaluationAuthority",
    "answer_proof_evaluation",
    "build_proof_evaluation_authority",
    "is_v3_observation",
    "project_proof_observation",
    "proof_citation_metrics",
    "proof_generation_metrics",
]
