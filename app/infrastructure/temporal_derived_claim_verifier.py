"""在评测基础设施边界独立复核确定性时序 gate 的单个直接结论。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from app.domain.evaluation import TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION
from app.shared.utils.claim_binding import (
    local_temporal_claim_support_preserved,
    local_temporal_gating_verdict_preserved,
    temporal_actions_compatible,
    temporal_positive_immediate_permission_actions,
)


_OUTER_STATUS = "committed_deterministic_temporal_binary_verdict"
_INNER_STATUS = "candidate_ready"
_INNER_REASON = "UNIQUE_MANDATORY_TEMPORAL_GATE"
_RULE_VERSION = "direct-binary-temporal-gating/1.0.0"
_EXECUTION_CONTRACT = "deterministic-zero-call.v1"
_OUTER_AUDIT_SCHEMA = "completeness-self-review-audit.v6"
_RECEIPT_SCHEMA = "runtime-completeness-receipt.v1"
_SOLVER_SCHEMA = "runtime-completeness-solver.v1"

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "solver_schema_version",
        "status",
        "reason_code",
        "solution_cardinality",
        "contract_hash",
        "projection_hash",
        "candidate_count",
        "authorized_candidate_count",
        "authorized_universe_block_count",
        "universe_digest",
        "universe_complete",
        "facet_set_digest",
        "facet_contract_complete",
        "zero_bindable_facet_count",
        "nonzero_bindable_facet_count",
        "facet_count",
        "query_obligation_count",
        "constraint_count",
        "obligation_count",
        "option_count",
        "search_state_count",
        "solution_count",
        "selection_count",
        "selection_hash",
        "residual_selection_count",
        "materializer_validation_count",
        "materializer_rejection_count",
        "materializer_rejection_reason_counts",
        "materializer_rejection_reason_hash",
        "current_trusted_claim_count",
        "current_mapped_claim_count",
        "current_ambiguous_claim_count",
        "current_unmapped_claim_count",
        "current_mapping_complete",
        "current_projection_available",
        "current_candidate_count",
        "current_incomplete_facet_count",
        "current_uncovered_constraint_count",
        "current_uncovered_query_obligation_count",
        "current_uncovered_query_obligation_type_counts",
        "trusted_floor_trusted_claim_count",
        "trusted_floor_mapped_claim_count",
        "trusted_floor_ambiguous_claim_count",
        "trusted_floor_unmapped_claim_count",
        "trusted_floor_mapping_complete",
        "trusted_floor_projection_available",
        "trusted_floor_candidate_count",
        "trusted_floor_coverage_credit_count",
        "trusted_floor_source",
        "initial_floor_candidate_count",
        "model_calls_added",
        "semantic_absence_proven",
        "abstention_authorized",
        "canonical_selection_count",
        "canonical_selection_hash",
        "canonicalization_policy",
    }
)
# receipt.v1 的当前 projector 比历史 canary 多一个 floor 计数字段，但没有升 schema。
# 因而只允许这一个已知向前字段缺席；未知字段仍一律拒绝。
_RECEIPT_OPTIONAL_FIELDS = frozenset({"trusted_floor_coverage_credit_count"})
_RECEIPT_REQUIRED_FIELDS = _RECEIPT_FIELDS - _RECEIPT_OPTIONAL_FIELDS
_RECEIPT_INTEGER_FIELDS = frozenset(
    {
        "candidate_count",
        "authorized_candidate_count",
        "authorized_universe_block_count",
        "zero_bindable_facet_count",
        "nonzero_bindable_facet_count",
        "facet_count",
        "query_obligation_count",
        "constraint_count",
        "obligation_count",
        "option_count",
        "search_state_count",
        "solution_count",
        "selection_count",
        "residual_selection_count",
        "materializer_validation_count",
        "materializer_rejection_count",
        "current_trusted_claim_count",
        "current_mapped_claim_count",
        "current_ambiguous_claim_count",
        "current_unmapped_claim_count",
        "current_candidate_count",
        "current_incomplete_facet_count",
        "current_uncovered_constraint_count",
        "current_uncovered_query_obligation_count",
        "trusted_floor_trusted_claim_count",
        "trusted_floor_mapped_claim_count",
        "trusted_floor_ambiguous_claim_count",
        "trusted_floor_unmapped_claim_count",
        "trusted_floor_candidate_count",
        "trusted_floor_coverage_credit_count",
        "initial_floor_candidate_count",
        "model_calls_added",
        "canonical_selection_count",
    }
)
_RECEIPT_BOOLEAN_FIELDS = frozenset(
    {
        "universe_complete",
        "facet_contract_complete",
        "current_mapping_complete",
        "current_projection_available",
        "trusted_floor_mapping_complete",
        "trusted_floor_projection_available",
        "semantic_absence_proven",
        "abstention_authorized",
    }
)
_RECEIPT_TEXT_FIELDS = frozenset(
    {
        "schema_version",
        "solver_schema_version",
        "status",
        "reason_code",
        "solution_cardinality",
        "contract_hash",
        "projection_hash",
        "universe_digest",
        "facet_set_digest",
        "selection_hash",
        "materializer_rejection_reason_hash",
        "trusted_floor_source",
        "canonical_selection_hash",
        "canonicalization_policy",
    }
)
_INNER_FIELDS = frozenset(
    {
        "rule_version",
        "status",
        "reason_code",
        "query_sha256",
        "evidence_set_sha256",
        "authorized_evidence_count",
        "direct_binary_query",
        "immediate_semantics",
        "current_envelope_valid",
        "qualifying_gate_count",
        "conflict_count",
        "condition_already_satisfied",
        "condition_state_ambiguous",
        "unsatisfied_trigger_proven",
        "selected_evidence_sha256",
    }
)

_CLAIM_BOUNDARY_RE = re.compile(r"[。！？!?；;\n]+")
_INLINE_CITATION_RE = re.compile(r"\[([^\[\]\n]{1,80})\]|【([^【】\n]{1,80})】")
_LABEL_RE = re.compile(r"E[1-9]\d*")
_NEGATIVE_RE = re.compile(
    r"(?:不能|不可以|不可|不得|请勿|切勿|禁止|严禁|不应|"
    r"\b(?:cannot|can't|must\s+not|may\s+not|not\s+allowed)\b)",
    re.IGNORECASE,
)
_IMMEDIATE_RE = re.compile(
    r"(?:马上|立即|立刻|现在|当前|此时|当下|"
    r"\b(?:immediately|right\s+now|currently|at\s+once|now)\b)",
    re.IGNORECASE,
)
_CN_QUERY_ACTION_RE = re.compile(
    r"(?:能否|可否|是否|应否|能不能|可不可以|可以|能够|(?<!功)(?<!性)能|可)"
    r"\s*(?:马上|立即|立刻|现在|当前|此时|当下|就|直接)*\s*"
    r"(?P<action>[^。！？!?\n]{2,64}?)(?:吗|呢)?(?:[？?]|$)",
    re.IGNORECASE,
)
_EN_QUERY_ACTION_RE = re.compile(
    r"\b(?:can|may|could|should)\s+"
    r"(?:(?:i|we|you)|the\s+(?:operator|user))?\s*"
    r"(?P<action>[^.!?\n]{2,96}?)(?:[?]|$)",
    re.IGNORECASE,
)
_UNSATISFIED_RE = re.compile(
    r"(?:刚\s*(?:使用|运行|操作|工作)?\s*(?:完|结束)|"
    r"刚\s*(?:停机|关机|结束(?:使用|运行|操作|工作))|"
    r"\b(?:has\s+|have\s+)?just\s+(?:finished|stopped|shut\s+down|"
    r"completed|ended|used|run|operated)\b)",
    re.IGNORECASE,
)
_GATE_COMPLETED_RE = re.compile(
    r"(?:已经|已|完成|满足|经过|结束|冷却后|等待后|"
    r"\b(?:already|completed|satisfied|after|has\s+finished)\b)",
    re.IGNORECASE,
)
_STATE_TRANSITION_RE = re.compile(
    r"(?:现?已(?:经)?\s*(?:恢复|归零|降至|降到|达到|回到|变为|变成|冷却|完成|满足)|"
    r"刚\s*(?:恢复|归零|降至|降到|达到|回到|变为|变成|冷却完成)|"
    r"\b(?:has\s+)?(?:recovered|reset|returned|dropped|reached|cooled)\b)",
    re.IGNORECASE,
)
_ACTION_NOISE_RE = re.compile(
    r"(?:本设备|该设备|此设备|设备的|\b(?:the|a|an|i|we|you)\b|"
    r"\s+|[，,。！？!?；;：:、])",
    re.IGNORECASE,
)
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CLAIM_ID_RE = re.compile(r"clm_[0-9]+")
_CITATION_ID_RE = re.compile(r"cit_[0-9]+")
_GATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:(?:必须|务必|应当|应该|需要|需|请)\s*先|"
        r"(?:^|[。！？!?；;\n：:])\s*先)\s*"
        r"(?P<gate>[^。！？!?；;\n]{2,48}?)\s*[,，]?\s*再\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:必须|务必|应当|应该|需要|需)\s*"
        r"(?P<gate>[^。！？!?；;\n]{2,48}?)\s*(?:后|之后|以后|完成后)\s*"
        r"(?:再|才(?:能|可以|可)?|方可)\s*"
        r"(?P<action>[^。！？!?；;\n]{2,64})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmust\s+(?P<gate>[^.!?;\n]{2,80}?)\s+before\s+"
        r"(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bonly\s+after\s+(?P<gate>[^.!?;\n]{2,80}?)\s+"
        r"(?:may|can|should)\s+(?P<action>[^.!?;\n]{2,80})",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class TemporalDerivedClaimValidation:
    """不含正文的时序派生主张复核结果。"""

    status: str
    reason_code: str
    derived_claim_count: int
    validated_count: int
    claim_index: int | None = None
    answer_claim_count: int = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "temporal_derived_claim_count": self.derived_claim_count,
            "temporal_derived_validated_count": self.validated_count,
            "temporal_receipt_status": {
                "contract_version": TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
                "status": self.status,
                "reason_code": self.reason_code,
            },
        }


@dataclass(frozen=True, slots=True)
class _ClaimSegment:
    text: str
    start: int
    end: int
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ContextAlias:
    label: str
    evidence_id: str
    text: str


def _result(
    status: str,
    reason_code: str,
    *,
    derived_claim_count: int,
    claim_index: int | None = None,
    answer_claim_count: int = 0,
) -> TemporalDerivedClaimValidation:
    return TemporalDerivedClaimValidation(
        status=status,
        reason_code=reason_code,
        derived_claim_count=derived_claim_count,
        validated_count=1 if status == "validated" else 0,
        claim_index=claim_index if status == "validated" else None,
        answer_claim_count=answer_claim_count,
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _digest(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _inline_labels(value: str) -> list[str]:
    labels: list[str] = []
    for match in _INLINE_CITATION_RE.finditer(value):
        raw = (match.group(1) or match.group(2) or "").strip()
        for label in re.split(r"[,，;；]", raw):
            normalized = label.strip()
            if _LABEL_RE.fullmatch(normalized):
                labels.append(normalized)
    return labels


def _claim_segments(answer: str) -> list[_ClaimSegment]:
    """以与基础指标一致的坐标规则提取可见 claim。"""

    segments: list[_ClaimSegment] = []
    cursor = 0
    boundaries: list[re.Match[str] | None] = [*_CLAIM_BOUNDARY_RE.finditer(answer), None]
    for boundary in boundaries:
        end = boundary.start() if boundary is not None else len(answer)
        raw = answer[cursor:end]
        leading_labels: list[str] = []
        remaining = raw
        remaining_offset = 0
        while True:
            leading = re.match(
                r"^\s*(?:\[([^\[\]\n]{1,80})\]|【([^【】\n]{1,80})】)",
                remaining,
            )
            if leading is None:
                break
            leading_labels.extend(_inline_labels(leading.group(0)))
            remaining_offset += leading.end()
            remaining = remaining[leading.end() :]
        if leading_labels and segments:
            previous = segments[-1]
            segments[-1] = _ClaimSegment(
                previous.text,
                previous.start,
                previous.end,
                tuple(dict.fromkeys((*previous.labels, *leading_labels))),
            )
        labels = tuple(dict.fromkeys(_inline_labels(remaining)))
        marker_offsets: set[int] = set()
        for marker in _INLINE_CITATION_RE.finditer(remaining):
            marker_offsets.update(range(marker.start(), marker.end()))
        visible = [
            (character, offset)
            for offset, character in enumerate(remaining)
            if offset not in marker_offsets
        ]
        strip_characters = " \t\r,，:："
        while visible and visible[0][0] in strip_characters:
            visible.pop(0)
        while visible and visible[-1][0] in strip_characters:
            visible.pop()
        content = "".join(character for character, _ in visible)
        if content:
            content_start = cursor + remaining_offset + visible[0][1]
            content_end = cursor + remaining_offset + visible[-1][1] + 1
            segments.append(_ClaimSegment(content, content_start, content_end, labels))
        elif labels and segments:
            previous = segments[-1]
            segments[-1] = _ClaimSegment(
                previous.text,
                previous.start,
                previous.end,
                tuple(dict.fromkeys((*previous.labels, *labels))),
            )
        if boundary is not None:
            cursor = boundary.end()
    return segments


def _negative_immediate_action(value: str) -> str | None:
    clauses = [part.strip() for part in _CLAIM_BOUNDARY_RE.split(value) if part.strip()]
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    if _NEGATIVE_RE.search(clause) is None or _IMMEDIATE_RE.search(clause) is None:
        return None
    action = _NEGATIVE_RE.sub("", clause, count=1)
    action = _IMMEDIATE_RE.sub("", action, count=1).strip()
    return action if len(_action_key(action)) >= 4 else None


def _query_action(query: str) -> str | None:
    matches = list(_CN_QUERY_ACTION_RE.finditer(query))
    if len(matches) == 1:
        action = _IMMEDIATE_RE.sub("", str(matches[0].group("action") or ""), count=1)
        action = action.strip(" ，,。！？!?；;：:吗呢")
        return action if len(_action_key(action)) >= 4 else None
    english = list(_EN_QUERY_ACTION_RE.finditer(query))
    if len(english) != 1:
        return None
    action = _IMMEDIATE_RE.sub("", str(english[0].group("action") or ""))
    action = action.strip(" ,.!?;:")
    return action if len(_action_key(action)) >= 4 else None


def _action_key(value: str) -> str:
    value = _ACTION_NOISE_RE.sub("", unicodedata.normalize("NFKC", value).casefold())
    return "".join(character for character in value if character.isalnum())


def _cjk_action_key(value: str) -> str:
    """移除 ASCII token，保留混合动作中的 CJK 动词、对象与修饰词。"""

    normalized = unicodedata.normalize("NFKC", value)
    return _action_key(_ASCII_TOKEN_RE.sub("", normalized))


def _cjk_action_key_matches(left: str, right: str) -> bool:
    """CJK core 必须精确同核；两侧均空时退化为纯 ASCII 比较。"""

    return _cjk_action_key(left) == _cjk_action_key(right)


def _gate_matches(value: str, *, action: str) -> list[re.Match[str]]:
    return [
        match
        for pattern in _GATE_PATTERNS
        for match in pattern.finditer(value)
        if _cjk_action_key_matches(action, str(match.group("action") or ""))
        and temporal_actions_compatible(action, str(match.group("action") or ""))
    ]


def _gate_is_satisfied(query: str, gate: str) -> bool:
    if _GATE_COMPLETED_RE.search(query) is None:
        return False
    gate_key = _action_key(
        re.sub(
            r"(?:先|等待|等候|待|直到|完成|满足|下来|以后|之后|"
            r"\b(?:wait|until|for|to|turn)\b)",
            "",
            gate,
            flags=re.IGNORECASE,
        )
    )
    query_key = _action_key(query)
    if len(gate_key) < 2:
        return False
    if gate_key in query_key:
        return True
    gate_pairs = {gate_key[index : index + 2] for index in range(len(gate_key) - 1)}
    query_pairs = {query_key[index : index + 2] for index in range(len(query_key) - 1)}
    return bool(gate_pairs and len(gate_pairs & query_pairs) / len(gate_pairs) >= 0.6)


def _strict_int(value: Any, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def _strict_trimmed_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _span_matches(value: Any, *, start: int, end: int) -> bool:
    return bool(
        isinstance(value, Mapping)
        and type(value.get("start")) is int
        and type(value.get("end")) is int
        and value.get("start") == start
        and value.get("end") == end
    )


def _rewrite_equivalence_proven(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    fields = set(value)
    if fields not in (
        {"status", "reasons"},
        {"status", "semantic_preservation_score", "reasons"},
    ):
        return False
    reasons = value.get("reasons")
    if not (
        value.get("status") == "accepted"
        and isinstance(reasons, list)
        and reasons == ["rewrite_equivalent"]
    ):
        return False
    if "semantic_preservation_score" not in value:
        return True
    score = value.get("semantic_preservation_score")
    return bool(type(score) in {int, float} and math.isfinite(float(score)) and float(score) == 1.0)


def _context_aliases(observation: Mapping[str, Any]) -> list[_ContextAlias] | None:
    context_pack = observation.get("context_pack")
    if not isinstance(context_pack, Mapping):
        return None
    raw_blocks = context_pack.get("packed_blocks")
    raw_mapping = context_pack.get("label_mapping")
    raw_context = observation.get("context")
    if not all(
        isinstance(value, list) and value for value in (raw_blocks, raw_mapping, raw_context)
    ):
        return None

    blocks: dict[str, tuple[str, str]] = {}
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            return None
        label = raw.get("label")
        evidence_id = raw.get("evidence_id")
        packed_text = raw.get("text")
        if (
            not _strict_trimmed_text(label)
            or not _strict_trimmed_text(evidence_id)
            or not _strict_trimmed_text(packed_text)
            or _LABEL_RE.fullmatch(label) is None
            or label in blocks
        ):
            return None
        blocks[label] = (evidence_id, packed_text)

    mappings: dict[str, str] = {}
    for raw in raw_mapping:
        if not isinstance(raw, Mapping):
            return None
        label = raw.get("label")
        evidence_id = raw.get("evidence_id")
        if (
            not _strict_trimmed_text(label)
            or not _strict_trimmed_text(evidence_id)
            or _LABEL_RE.fullmatch(label) is None
            or label in mappings
        ):
            return None
        mappings[label] = evidence_id

    contexts: dict[str, tuple[str, str]] = {}
    for raw in raw_context:
        if not isinstance(raw, Mapping):
            return None
        label = raw.get("context_label") or raw.get("label")
        nested_evidence = raw.get("evidence")
        nested_evidence = nested_evidence if isinstance(nested_evidence, Mapping) else {}
        top_evidence_id = raw.get("evidence_id")
        nested_evidence_id = nested_evidence.get("evidence_id")
        if top_evidence_id is not None and not _strict_trimmed_text(top_evidence_id):
            return None
        if nested_evidence_id is not None and not _strict_trimmed_text(nested_evidence_id):
            return None
        if top_evidence_id and nested_evidence_id and top_evidence_id != nested_evidence_id:
            return None
        evidence_id = top_evidence_id or nested_evidence_id
        text_value = raw.get("text")
        content_value = raw.get("content")
        if not (
            _strict_trimmed_text(text_value)
            and _strict_trimmed_text(content_value)
            and text_value == content_value
        ):
            return None
        packed_text = text_value
        if (
            not _strict_trimmed_text(label)
            or not _strict_trimmed_text(evidence_id)
            or _LABEL_RE.fullmatch(label) is None
            or label in contexts
        ):
            return None
        contexts[label] = (evidence_id, packed_text)

    if set(blocks) != set(mappings) or set(blocks) != set(contexts):
        return None
    aliases: list[_ContextAlias] = []
    for label in sorted(blocks, key=lambda item: int(item[1:])):
        evidence_id, packed_text = blocks[label]
        if mappings[label] != evidence_id or contexts[label] != (evidence_id, packed_text):
            return None
        aliases.append(_ContextAlias(label, evidence_id, packed_text))
    if len({alias.evidence_id for alias in aliases}) != len(aliases):
        return None
    return aliases


def _zero_call_boundary_valid(audit: Mapping[str, Any]) -> bool:
    usage = audit.get("usage")
    return bool(
        audit.get("execution_contract") == _EXECUTION_CONTRACT
        and audit.get("provider_review_enabled") is False
        and audit.get("prompt_sha256") == ""
        and audit.get("provider_response_sha256") == ""
        and isinstance(usage, Mapping)
        and usage.get("status") == "not_applicable"
        and usage.get("reason") == "deterministic zero-call completeness contract"
        and all(
            _strict_int(audit.get(field)) and audit.get(field) == 0
            for field in (
                "calls_attempted",
                "provider_attempt_count",
                "retry_count",
                "output_token_cap",
            )
        )
    )


def _indeterminate_receipt_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    fields = set(value)
    if not _RECEIPT_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(_RECEIPT_FIELDS):
        return False
    if not all(
        _strict_int(value.get(field), minimum=0)
        for field in _RECEIPT_INTEGER_FIELDS
        if field in value
    ):
        return False
    if not all(type(value.get(field)) is bool for field in _RECEIPT_BOOLEAN_FIELDS):
        return False
    if not all(isinstance(value.get(field), str) for field in _RECEIPT_TEXT_FIELDS):
        return False

    rejection_counts = value.get("materializer_rejection_reason_counts")
    obligation_type_counts = value.get("current_uncovered_query_obligation_type_counts")
    if not (
        isinstance(rejection_counts, Mapping)
        and all(
            _strict_trimmed_text(reason_code) and _strict_int(count, minimum=1)
            for reason_code, count in rejection_counts.items()
        )
        and isinstance(obligation_type_counts, Mapping)
        and all(
            reason_type in {"query_clause", "direct_binary_verdict", "unknown"}
            and _strict_int(count, minimum=1)
            for reason_type, count in obligation_type_counts.items()
        )
    ):
        return False
    rejection_projection = [
        {"reason_code": reason_code, "count": rejection_counts[reason_code]}
        for reason_code in sorted(rejection_counts)
    ]
    expected_rejection_hash = (
        _digest(_canonical_json(rejection_projection)) if rejection_projection else ""
    )
    if not (
        value.get("materializer_rejection_count") == sum(rejection_counts.values())
        and value.get("materializer_rejection_reason_hash") == expected_rejection_hash
        and value.get("current_uncovered_query_obligation_count")
        == sum(obligation_type_counts.values())
    ):
        return False
    if (
        "trusted_floor_coverage_credit_count" in value
        and value.get("trusted_floor_coverage_credit_count") != 0
    ):
        return False

    exact_zero_fields = (
        "current_ambiguous_claim_count",
        "current_unmapped_claim_count",
        "current_incomplete_facet_count",
        "current_uncovered_constraint_count",
        "initial_floor_candidate_count",
        "residual_selection_count",
        "selection_count",
        "solution_count",
        "canonical_selection_count",
        "materializer_validation_count",
        "materializer_rejection_count",
        "search_state_count",
        "zero_bindable_facet_count",
        "trusted_floor_ambiguous_claim_count",
        "trusted_floor_unmapped_claim_count",
        "model_calls_added",
    )
    positive_fields = (
        "current_candidate_count",
        "current_trusted_claim_count",
        "current_mapped_claim_count",
    )
    if not all(
        _strict_int(value.get(field)) and value.get(field) == 0 for field in exact_zero_fields
    ):
        return False
    if not all(_strict_int(value.get(field), minimum=1) for field in positive_fields):
        return False
    if any(
        value.get(field) != ""
        for field in (
            "selection_hash",
            "canonical_selection_hash",
            "canonicalization_policy",
        )
    ):
        return False
    candidate_count = value.get("current_candidate_count")
    if (
        value.get("current_trusted_claim_count") != candidate_count
        or value.get("current_mapped_claim_count") != candidate_count
    ):
        return False
    floor_source = value.get("trusted_floor_source")
    floor_counts = (
        value.get("trusted_floor_trusted_claim_count"),
        value.get("trusted_floor_mapped_claim_count"),
        value.get("trusted_floor_candidate_count"),
    )
    floor_mapping_complete = value.get("trusted_floor_mapping_complete")
    floor_projection_available = value.get("trusted_floor_projection_available")
    if floor_source == "none":
        if not (
            floor_counts == (0, 0, 0)
            and floor_mapping_complete is False
            and floor_projection_available is False
        ):
            return False
    elif floor_source == "current":
        if not (
            floor_counts == (candidate_count, candidate_count, candidate_count)
            and value.get("trusted_floor_ambiguous_claim_count") == 0
            and value.get("trusted_floor_unmapped_claim_count") == 0
            and floor_mapping_complete is True
            and floor_projection_available is True
        ):
            return False
    else:
        return False
    return bool(
        value.get("schema_version") == _RECEIPT_SCHEMA
        and value.get("solver_schema_version") == _SOLVER_SCHEMA
        and value.get("status") == "INDETERMINATE"
        and value.get("reason_code") == "CURRENT_COVERAGE_UNPROVEN_NO_GAP"
        and value.get("solution_cardinality") == "indeterminate"
        and value.get("current_mapping_complete") is True
        and value.get("current_projection_available") is True
        and _strict_int(value.get("current_uncovered_query_obligation_count"))
        and value.get("current_uncovered_query_obligation_count") == 1
        and value.get("current_uncovered_query_obligation_type_counts")
        == {"direct_binary_verdict": 1}
        and value.get("semantic_absence_proven") is False
        and value.get("abstention_authorized") is False
    )


def _inner_receipt_shape_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != _INNER_FIELDS:
        return False
    string_fields = (
        "rule_version",
        "status",
        "reason_code",
        "query_sha256",
        "evidence_set_sha256",
        "selected_evidence_sha256",
    )
    integer_fields = (
        "authorized_evidence_count",
        "qualifying_gate_count",
        "conflict_count",
    )
    boolean_fields = (
        "direct_binary_query",
        "immediate_semantics",
        "current_envelope_valid",
        "condition_already_satisfied",
        "condition_state_ambiguous",
        "unsatisfied_trigger_proven",
    )
    return bool(
        all(isinstance(value.get(field), str) for field in string_fields)
        and all(_strict_int(value.get(field), minimum=0) for field in integer_fields)
        and all(type(value.get(field)) is bool for field in boolean_fields)
    )


def _spans_are_disjoint(spans: list[tuple[int, int]]) -> bool:
    ordered = sorted(spans)
    return all(
        right_start >= left_end for (_, left_end), (right_start, _) in zip(ordered, ordered[1:])
    )


def _envelope_graph(
    *,
    answer: str,
    raw_claims: list[Any],
    raw_citations: list[Any],
    raw_sources: list[Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]] | None:
    """独立复核 envelope 的 ID、span、引用所有权和 source 图。"""

    claims_by_id: dict[str, Mapping[str, Any]] = {}
    claim_spans: list[tuple[int, int]] = []
    claim_citation_ids: dict[str, tuple[str, ...]] = {}
    for item in raw_claims:
        if not isinstance(item, Mapping):
            return None
        claim_id = item.get("claim_id")
        start = item.get("answer_start")
        end = item.get("answer_end")
        text = item.get("text")
        citation_ids = item.get("citation_ids")
        if not (
            _strict_trimmed_text(claim_id)
            and _CLAIM_ID_RE.fullmatch(claim_id) is not None
            and claim_id not in claims_by_id
            and _strict_int(start, minimum=0)
            and _strict_int(end, minimum=1)
            and start < end <= len(answer)
            and isinstance(text, str)
            and text == answer[start:end]
            and item.get("claim_type") in {"factual", "caveat", "uncertainty"}
            and type(item.get("supported")) is bool
            and isinstance(citation_ids, (list, tuple))
            and all(
                _strict_trimmed_text(citation_id)
                and _CITATION_ID_RE.fullmatch(citation_id) is not None
                for citation_id in citation_ids
            )
            and len(citation_ids) == len(set(citation_ids))
        ):
            return None
        claims_by_id[claim_id] = item
        claim_spans.append((start, end))
        claim_citation_ids[claim_id] = tuple(citation_ids)
    if not _spans_are_disjoint(claim_spans):
        return None

    source_ids: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, Mapping):
            return None
        evidence_id = item.get("evidence_id")
        if not _strict_trimmed_text(evidence_id) or evidence_id in source_ids:
            return None
        source_ids.add(evidence_id)

    citations_by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_citations:
        if not isinstance(item, Mapping):
            return None
        citation_id = item.get("citation_id")
        claim_id = item.get("claim_id")
        evidence_id = item.get("evidence_id")
        start = item.get("answer_start")
        end = item.get("answer_end")
        if not (
            _strict_trimmed_text(citation_id)
            and _CITATION_ID_RE.fullmatch(citation_id) is not None
            and citation_id not in citations_by_id
            and _strict_trimmed_text(claim_id)
            and _CLAIM_ID_RE.fullmatch(claim_id) is not None
            and claim_id in claims_by_id
            and _strict_trimmed_text(evidence_id)
            and evidence_id in source_ids
            and _strict_int(start, minimum=0)
            and _strict_int(end, minimum=1)
            and start < end <= len(answer)
            and item.get("validation") in {"valid", "degraded", "invalid"}
            and item.get("relation") in {"SUPPORTS", "CONTRADICTS", "CONTEXT"}
        ):
            return None
        owner = claims_by_id[claim_id]
        if start != owner.get("answer_start") or end != owner.get("answer_end"):
            return None
        citations_by_id[citation_id] = item

    references: dict[str, list[str]] = {citation_id: [] for citation_id in citations_by_id}
    for claim_id, citation_ids in claim_citation_ids.items():
        for citation_id in citation_ids:
            citation = citations_by_id.get(citation_id)
            if citation is None or citation.get("claim_id") != claim_id:
                return None
            references[citation_id].append(claim_id)
        trusted = any(
            citations_by_id[citation_id].get("validation") in {"valid", "degraded"}
            for citation_id in citation_ids
        )
        if claims_by_id[claim_id].get("supported") is not trusted:
            return None
    if any(len(owners) != 1 for owners in references.values()):
        return None
    return claims_by_id, citations_by_id


def _citation_identity_valid(
    observation: Mapping[str, Any],
    *,
    answer: str,
    claim: _ClaimSegment,
    selected: _ContextAlias,
    aliases: list[_ContextAlias],
) -> bool:
    raw_citations = observation.get("citations")
    validation = observation.get("citation_validation")
    envelope = observation.get("answer_envelope")
    if (
        not isinstance(raw_citations, list)
        or not raw_citations
        or not isinstance(validation, Mapping)
        or not isinstance(envelope, Mapping)
    ):
        return False
    if not (
        validation.get("status") == "valid"
        and type(validation.get("coverage")) in {int, float}
        and not isinstance(validation.get("coverage"), bool)
        and float(validation.get("coverage")) == 1.0
        and _strict_int(validation.get("citation_count"))
        and validation.get("citation_count") == len(raw_citations)
        and _strict_int(validation.get("trusted_count"))
        and validation.get("trusted_count") == len(raw_citations)
        and isinstance(validation.get("issues"), list)
        and not validation.get("issues")
    ):
        return False
    if claim.labels != (selected.label,):
        return False

    envelope_claims = envelope.get("claims")
    envelope_citations = envelope.get("citations")
    envelope_sources = envelope.get("sources")
    projected_claims = observation.get("claims")
    if not (
        envelope.get("schema_version") == "answer.v1"
        and envelope.get("answer") == answer
        and isinstance(envelope_claims, list)
        and isinstance(envelope_citations, list)
        and isinstance(envelope_sources, list)
        and projected_claims == envelope_claims
    ):
        return False

    graph = _envelope_graph(
        answer=answer,
        raw_claims=envelope_claims,
        raw_citations=envelope_citations,
        raw_sources=envelope_sources,
    )
    if graph is None:
        return False
    claims_by_id, citations_by_id = graph

    aliases_by_label = {alias.label: alias for alias in aliases}
    runtime_spans: list[tuple[int, int]] = []
    for item in raw_citations:
        if not isinstance(item, Mapping):
            return False
        span = item.get("answer_span")
        if not isinstance(span, Mapping):
            return False
        start = span.get("start")
        end = span.get("end")
        label = item.get("label")
        evidence_id = item.get("evidence_id")
        claim_text = item.get("claim_text")
        if not (
            _strict_int(start, minimum=0)
            and _strict_int(end, minimum=1)
            and start < end <= len(answer)
            and _strict_trimmed_text(label)
            and _LABEL_RE.fullmatch(label) is not None
            and _strict_trimmed_text(evidence_id)
            and aliases_by_label.get(label) is not None
            and aliases_by_label[label].evidence_id == evidence_id
            and isinstance(claim_text, str)
            and claim_text == answer[start:end]
            and item.get("validation_status") == "valid"
        ):
            return False
        envelope_matches = [
            envelope_citation
            for envelope_citation in citations_by_id.values()
            if envelope_citation.get("answer_start") == start
            and envelope_citation.get("answer_end") == end
            and envelope_citation.get("evidence_id") == evidence_id
            and envelope_citation.get("validation") == "valid"
            and envelope_citation.get("relation") == "SUPPORTS"
        ]
        if len(envelope_matches) != 1:
            return False
        runtime_spans.append((start, end))
    if not _spans_are_disjoint(runtime_spans):
        return False

    matching_claims = [
        item
        for item in claims_by_id.values()
        if type(item.get("answer_start")) is int
        and type(item.get("answer_end")) is int
        and item.get("answer_start") == claim.start
        and item.get("answer_end") == claim.end
        and item.get("text") == answer[claim.start : claim.end] == claim.text
    ]
    if len(matching_claims) != 1:
        return False
    envelope_claim = matching_claims[0]
    claim_id = envelope_claim.get("claim_id")
    citation_ids = envelope_claim.get("citation_ids")
    if not (
        claim_id
        and envelope_claim.get("claim_type") == "factual"
        and envelope_claim.get("supported") is True
        and isinstance(citation_ids, (list, tuple))
        and len(citation_ids) == 1
    ):
        return False

    envelope_citation = citations_by_id.get(citation_ids[0])
    if envelope_citation is None:
        return False
    if not (
        envelope_citation.get("claim_id") == claim_id
        and envelope_citation.get("evidence_id") == selected.evidence_id
        and envelope_citation.get("answer_start") == claim.start
        and envelope_citation.get("answer_end") == claim.end
        and envelope_citation.get("validation") == "valid"
        and envelope_citation.get("relation") == "SUPPORTS"
    ):
        return False

    source_ids = [item.get("evidence_id") for item in envelope_sources]
    if source_ids.count(selected.evidence_id) != 1:
        return False

    same_span: list[Mapping[str, Any]] = []
    for item in raw_citations:
        span = item.get("answer_span")
        if _span_matches(span, start=claim.start, end=claim.end):
            same_span.append(item)
    if len(same_span) != 1:
        return False
    citation = same_span[0]
    visible = answer[claim.start : claim.end]
    return bool(
        citation.get("label") == selected.label
        and citation.get("evidence_id") == selected.evidence_id
        and citation.get("claim_text") == claim.text
        and visible == claim.text
    )


def validate_temporal_derived_claim(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> TemporalDerivedClaimValidation:
    """不信任运行态结论，按实际 query/answer/context/citation 重新验真。"""

    answer = _text(observation.get("answer"))
    segments = _claim_segments(answer)
    derived = [
        (index, segment, action)
        for index, segment in enumerate(segments)
        if (action := _negative_immediate_action(segment.text)) is not None
    ]
    derived_count = len(derived)
    audit = observation.get("completeness_self_review")
    if not isinstance(audit, Mapping) or not audit:
        return _result(
            "not_applicable",
            "TEMPORAL_RECEIPT_ABSENT",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if audit.get("schema_version") != _OUTER_AUDIT_SCHEMA or audit.get("status") != _OUTER_STATUS:
        return _result(
            "rejected",
            "TEMPORAL_OUTER_STATUS_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not _zero_call_boundary_valid(audit):
        return _result(
            "rejected",
            "TEMPORAL_ZERO_CALL_BOUNDARY_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not _indeterminate_receipt_valid(audit.get("runtime_completeness_receipt")):
        return _result(
            "rejected",
            "TEMPORAL_INDETERMINATE_RECEIPT_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if derived_count != 1:
        return _result(
            "rejected",
            "TEMPORAL_DIRECT_CLAIM_COUNT_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )

    claim_index, claim, claim_action = derived[0]
    original_query = observation.get("query")
    expected_query = case.get("query")
    rewritten_query = observation.get("rewritten_query")
    understanding = observation.get("query_understanding")
    query_plan = observation.get("query_plan")
    plan_understanding = (
        query_plan.get("understanding") if isinstance(query_plan, Mapping) else None
    )
    if not (
        isinstance(original_query, str)
        and isinstance(expected_query, str)
        and original_query == expected_query
        and isinstance(rewritten_query, str)
        and rewritten_query.strip()
        and rewritten_query == rewritten_query.strip()
        and isinstance(understanding, Mapping)
        and isinstance(plan_understanding, Mapping)
        and understanding == plan_understanding
        and understanding.get("standalone_query") == rewritten_query
        and plan_understanding.get("standalone_query") == rewritten_query
        and understanding.get("original_query") == rewritten_query
        and understanding.get("display_query") == rewritten_query
        and plan_understanding.get("original_query") == rewritten_query
        and plan_understanding.get("display_query") == rewritten_query
        and _rewrite_equivalence_proven(understanding.get("rewrite"))
    ):
        return _result(
            "rejected",
            "TEMPORAL_QUERY_BINDING_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    query = rewritten_query
    query_action = _query_action(query)
    if (
        not query
        or _IMMEDIATE_RE.search(query) is None
        or query_action is None
        or not _cjk_action_key_matches(claim_action, query_action)
        or not temporal_actions_compatible(claim_action, query_action)
    ):
        return _result(
            "rejected",
            "TEMPORAL_QUERY_ACTION_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )

    aliases = _context_aliases(observation)
    if aliases is None:
        return _result(
            "rejected",
            "TEMPORAL_CONTEXT_IDENTITY_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    evidence_ids = sorted({item.evidence_id for item in aliases})
    inner = audit.get("temporal_binary_verdict")
    if not _inner_receipt_shape_valid(inner):
        return _result(
            "rejected",
            "TEMPORAL_INNER_RECEIPT_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not (
        inner.get("status") == _INNER_STATUS
        and inner.get("reason_code") == _INNER_REASON
        and inner.get("rule_version") == _RULE_VERSION
        and inner.get("direct_binary_query") is True
        and inner.get("immediate_semantics") is True
        and inner.get("current_envelope_valid") is True
        and _strict_int(inner.get("authorized_evidence_count"))
        and inner.get("authorized_evidence_count") == len(evidence_ids)
    ):
        return _result(
            "rejected",
            "TEMPORAL_INNER_RECEIPT_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )

    if inner.get("query_sha256") != _digest(query):
        reason_code = "TEMPORAL_QUERY_HASH_MISMATCH"
    elif inner.get("evidence_set_sha256") != _digest(_canonical_json(evidence_ids)):
        reason_code = "TEMPORAL_EVIDENCE_SET_HASH_MISMATCH"
    elif any(
        audit.get(field) != _digest(answer)
        for field in (
            "materialized_candidate_sha256",
            "security_approved_candidate_sha256",
            "committed_answer_sha256",
        )
    ):
        reason_code = "TEMPORAL_ANSWER_HASH_MISMATCH"
    else:
        reason_code = ""
    if reason_code:
        return _result(
            "rejected",
            reason_code,
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )

    qualifying: list[tuple[_ContextAlias, re.Match[str]]] = []
    for alias in aliases:
        if not local_temporal_gating_verdict_preserved(claim.text, alias.text):
            continue
        matches = _gate_matches(alias.text, action=claim_action)
        if len(matches) == 1:
            qualifying.append((alias, matches[0]))
    conflict_count = sum(
        any(
            temporal_actions_compatible(claim_action, permission_action)
            for permission_action in temporal_positive_immediate_permission_actions(alias.text)
        )
        for alias in aliases
    )
    if not (
        _strict_int(inner.get("qualifying_gate_count"))
        and inner.get("qualifying_gate_count") == len(qualifying) == 1
    ):
        return _result(
            "rejected",
            "TEMPORAL_GATE_NOT_UNIQUE",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not (
        _strict_int(inner.get("conflict_count"))
        and inner.get("conflict_count") == conflict_count == 0
    ):
        return _result(
            "rejected",
            "TEMPORAL_CONFLICTING_PERMISSION",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )

    selected, gate_match = qualifying[0]
    gate = _text(gate_match.group("gate")).strip()
    gate_action = _text(gate_match.group("action")).strip()
    unsatisfied = _UNSATISFIED_RE.search(query) is not None
    ambiguous = _STATE_TRANSITION_RE.search(query) is not None
    satisfied = _gate_is_satisfied(query, gate)
    if not (
        inner.get("unsatisfied_trigger_proven") is unsatisfied is True
        and inner.get("condition_state_ambiguous") is ambiguous is False
        and inner.get("condition_already_satisfied") is satisfied is False
        and _cjk_action_key_matches(claim_action, gate_action)
        and temporal_actions_compatible(claim_action, gate_action)
    ):
        return _result(
            "rejected",
            "TEMPORAL_CONDITION_STATE_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if inner.get("selected_evidence_sha256") != _digest(selected.evidence_id):
        return _result(
            "rejected",
            "TEMPORAL_SELECTED_EVIDENCE_HASH_MISMATCH",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not _citation_identity_valid(
        observation,
        answer=answer,
        claim=claim,
        selected=selected,
        aliases=aliases,
    ):
        return _result(
            "rejected",
            "TEMPORAL_CITATION_IDENTITY_INVALID",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    if not (
        local_temporal_gating_verdict_preserved(claim.text, selected.text)
        and local_temporal_claim_support_preserved(claim.text, selected.text)
    ):
        return _result(
            "rejected",
            "TEMPORAL_SHARED_BINDING_REJECTED",
            derived_claim_count=derived_count,
            answer_claim_count=len(segments),
        )
    return _result(
        "validated",
        "TEMPORAL_DERIVED_CLAIM_VALIDATED",
        derived_claim_count=derived_count,
        claim_index=claim_index,
        answer_claim_count=len(segments),
    )


__all__ = [
    "TemporalDerivedClaimValidation",
    "validate_temporal_derived_claim",
]
