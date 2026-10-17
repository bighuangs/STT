"""Deterministic retrieval and answer-quality metrics.

The evaluator intentionally contains no model calls and no database access.  A
case and an observation are plain dictionaries so that the same definitions can
be used for replay tests, live-query evaluation and CI quality gates.

Relevance matching is conservative: stable identifiers and complete document
names must match exactly; textual evidence only matches when a sufficiently
specific evidence phrase is present or *all* configured match terms occur.  A
single short word therefore cannot accidentally mark an unrelated chunk as
relevant.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    AUTHORED_CLAIM_CONTRACT_VERSION,
    AUTHORED_CONCEPT_CONTRACT_VERSION,
    CASE_PASS_POLICY_VERSION,
    SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
)
from app.domain.evaluation_proof import (
    ProofEvaluationAuthority,
    answer_proof_evaluation,
    is_v3_observation,
    proof_citation_metrics,
    proof_generation_metrics,
)

from app.infrastructure.temporal_derived_claim_verifier import (
    validate_temporal_derived_claim,
)
from app.shared.utils.claim_binding import (
    ascii_token_keys,
    local_claim_bindings_preserved,
    restore_same_evidence_placeholders,
)


CONCEPT_VARIANT_MATCHER_POLICY_VERSION = "stt.eval.concept-variant-matcher.v4-cjk-sibling-guard"


_INLINE_CITATION_RE = re.compile(r"\[([^\[\]\n]{1,80})\]|【([^【】\n]{1,80})】")
_CLAIM_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_TERMINAL_CLAIM_SPLIT_RE = re.compile(r"[。！？!?\n]+")
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:[._:/-][a-z0-9]+)*", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_NUMBER_RE = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)*(?:%|[a-z]+)?", re.IGNORECASE)
_NEGATIVE_INTENT_RE = re.compile(
    r"(?:请勿|不得|禁止|切勿|不要|不能|不可以|未(?:经|使用|打印|完成|去除)?|没有|无法)"
    r"|\b(?:not|never|without|no)\b",
    re.IGNORECASE,
)
_MULTI_BRANCH_QUERY_RE = re.compile(
    r"(?:分别|各自|以及|同时|并且|并说明|并列出|和哪|与哪|哪些|全部|每个|"
    r"两种|两个|三种|步骤|哪些值|哪些项|比较|差异|如何.*(?:并|以及))"
)
_REFERENCE_CLAUSE_SPLIT_RE = re.compile(r"[；;。\n]+")
_SECONDARY_CLAIM_SPLIT_RE = re.compile(r"[，,](?=(?:此外|另外|但是|但|然而|不过|同时|并且|而且))")
_NON_FACTUAL_META_RE = re.compile(
    r"^(?:根据(?:以上|上述)(?:信息|内容)|综上(?:所述)?|具体如下|答案如下|"
    r"可分为以下(?:几)?(?:项|步|类)|以下为相关(?:信息|内容)|"
    r"(?:具体)?(?:操作)?(?:步骤|条件|注意事项|说明|方法|流程|要点)(?:如下)?|"
    r"前提条件|安装向导操作步骤)[:：]?$"
)
_SAFE_ABSTENTION_RESIDUAL_RE = re.compile(
    r"^(?:(?:建议|请|可(?:以)?)(?:补充|提供|咨询|联系|查阅|确认|说明).*"
    r"|(?:为避免误导[，,]?)?本次不提供推测性结论)$"
)
_ABSTENTION_CLAUSE_RE = re.compile(
    r"(?:无法|不能)(?:仅|单纯)?(?:从|依据|根据)?[^，。；;但]{0,32}"
    r"(?:回答|确定|判断)|(?:未|没有)(?:找到|检索到|提供|给出)[^，。；;]{0,24}"
    r"(?:信息|内容|依据)?|(?:信息|依据|上下文)(?:不|并不)(?:足|充分|完整)",
    re.IGNORECASE,
)
_FACTUAL_ASSERTION_RE = re.compile(
    r"(?:为|是|有|支持|采用|使用|包含|具备|位于|价格|现货|型号|规格|"
    r"可打印|能够|会|达到|等于|不支持|不能)"
)
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)）]\s*)")
_URL_ONLY_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:\[?image omitted\]?|<image omitted>|【?图片】?)",
    re.IGNORECASE,
)
_EVIDENCE_SCOPE_RE = re.compile(
    r"(?:手册|文档|证据|evidence|参考内容|上下文|资料|信息)",
    re.IGNORECASE,
)
_EVIDENCE_ABSENCE_RE = re.compile(
    r"(?:未|没有|不)(?:包含|记载|说明|提及|涉及|推荐|提供|给出|足以)",
    re.IGNORECASE,
)
_CONTRACT_ABSTENTION_PATTERNS = (
    re.compile(r"(?:不足以|无法)(?:据此|从[^，。；;]{0,24})?(?:回答|确定|判断)?"),
    re.compile(
        r"(?:手册|文档|证据|evidence|参考内容|上下文|资料|信息)"
        r"[^，。；;]{0,32}(?:未|没有|不)(?:包含|记载|说明|提及|涉及|推荐|提供|给出|足以)",
        re.IGNORECASE,
    ),
)
_CONCEPT_ASCII_ATOM_RE = re.compile(
    r"(?<![a-z0-9])[a-z0-9]+(?:[._:/-][a-z0-9]+)*(?![a-z0-9])",
    re.IGNORECASE,
)
_CONCEPT_SPACED_IDENTIFIER_RE = re.compile(
    r"(?<![a-z0-9])"
    r"[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*\s+"
    r"\d+(?:[._:/-]\d+)*(?:[a-z]+)?"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_CONCEPT_MODEL_SUFFIX_RE = re.compile(
    r"(?<![a-z0-9])[a-z]+\d+[a-z0-9]*\s+[a-z]{1,5}(?![a-z0-9])",
    re.IGNORECASE,
)
_CONCEPT_GENERIC_CJK_ACTIONS = frozenset(
    {
        "点击",
        "单击",
        "选择",
        "打开",
        "关闭",
        "查看",
        "完成",
        "按",
        "按下",
        "执行",
        "操作",
        "设置",
        "勾选",
        "安装",
        "添加",
        "打印",
        "开启",
        "进入",
        "访问",
    }
)
_CONCEPT_GENERIC_ENGLISH_ACTIONS = frozenset(
    {
        "add",
        "check",
        "choose",
        "click",
        "close",
        "enter",
        "install",
        "open",
        "print",
        "run",
        "select",
        "set",
        "start",
        "view",
    }
)
_ABSTENTION_PATTERNS = (
    re.compile(r"无法(?:从[^，。；;]{0,24})?(?:回答|确定|判断)"),
    re.compile(r"(?:未|没有)(?:找到|检索到|提供)(?:足够|相关|对应)?(?:的)?(?:信息|内容|依据)"),
    re.compile(r"(?:信息|依据|上下文)(?:不|并不)(?:足|充分|完整)"),
    re.compile(r"\b(?:i\s+do\s+not\s+know|i\s+don't\s+know|cannot\s+answer)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:insufficient|not\s+enough)\s+(?:context|evidence|information)\b", re.IGNORECASE
    ),
)
_FORBIDDEN_LOCAL_CLAUSE_RE = re.compile(r"[^。！？!?；;，,\n]+")
_OPAQUE_FORBIDDEN_LITERAL_RE = re.compile(
    r"(?:(?:secret|token|password|credential|api[_-]?key)[a-z0-9_.:-]*\s*=|"
    r"[a-z0-9]+(?:_[a-z0-9]+)+\s*=)",
    re.IGNORECASE,
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _normalized_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", _text(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _compact_text(value: Any) -> str:
    """Normalize prose while retaining only letters, numbers and CJK glyphs."""

    value = unicodedata.normalize("NFKC", _text(value)).casefold()
    return "".join(ch for ch in value if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def _normalized_identifier(value: Any) -> str:
    # Identifiers are not punctuation-stripped: ``chunk-1`` must not equal
    # ``chunk_1``.  Only Unicode form, case and surrounding whitespace vary.
    return unicodedata.normalize("NFKC", _text(value)).strip().casefold()


def _normalized_name(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", _text(value)).strip()).casefold()


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return _text(candidate.get("text") or candidate.get("content") or "")


def _ranked_candidates(observation: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for name in ("reranked", "retrieved", "context"):
        raw = observation.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            return name, [dict(item) for item in raw if isinstance(item, Mapping)]
    # Preserve the most meaningful explicitly supplied empty source.
    for name in ("reranked", "retrieved", "context"):
        raw = observation.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return name, []
    return "retrieved", []


def _context_candidates(
    observation: Mapping[str, Any], ranked: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw = observation.get("context")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return ranked


def _contains_term(haystack: Any, needle: Any) -> bool:
    term = _normalized_text(needle)
    if not term:
        return False
    body = _normalized_text(haystack)
    if term.isascii() and term.isdecimal():
        # 标准号经 PDF/OCR 解析后常与字母前缀直接相连，例如
        # ``GB/T26572``。纯数字标签只需要保持“数字边界”，不能沿用
        # ASCII 标识符的字母数字边界；同时仍拒绝把 26572 命中为
        # 1265720 的子串。
        return re.search(rf"(?<![0-9]){re.escape(term)}(?![0-9])", body) is not None
    if term.isascii() and re.fullmatch(r"[a-z0-9_+-]+", term):
        return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", body) is not None
    return _compact_text(term) in _compact_text(body)


def _is_opaque_forbidden_literal(term: str) -> bool:
    """识别必须按字面阻断的 Secret/Canary，而不是自然语言命题。"""

    normalized = unicodedata.normalize("NFKC", term).strip()
    return bool(_OPAQUE_FORBIDDEN_LITERAL_RE.search(normalized)) or bool(
        re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", normalized)
    )


def _forbidden_proposition_asserted(answer: Any, term: Any) -> bool:
    """仅在局部子句断言禁用命题时命中，Secret/Canary 始终按字面命中。"""

    forbidden = _text(term).strip()
    if not forbidden:
        return False
    if _is_opaque_forbidden_literal(forbidden):
        normalized_forbidden = unicodedata.normalize("NFKC", forbidden).casefold()
        normalized_answer = unicodedata.normalize("NFKC", _text(answer)).casefold()
        return normalized_forbidden in normalized_answer
    if not _contains_term(answer, forbidden):
        return False

    forbidden_is_negative = bool(_NEGATIVE_INTENT_RE.search(forbidden))
    for match in _FORBIDDEN_LOCAL_CLAUSE_RE.finditer(_text(answer)):
        clause = match.group(0).strip()
        if not clause or not _contains_term(clause, forbidden):
            continue
        clause_is_negative = bool(_NEGATIVE_INTENT_RE.search(clause))
        if clause_is_negative == forbidden_is_negative:
            return True
    return False


def _specific_match_terms(value: Any) -> list[str]:
    terms = [_text(item).strip() for item in _as_list(value) if _text(item).strip()]
    if not terms:
        return []
    specificity = sum(len(_compact_text(term)) for term in terms)
    # One short label such as “安全” is too generic for evidence relevance.
    if len(terms) == 1 and specificity < 4:
        return []
    if specificity < 4:
        return []
    return terms


def _target_definition(case: Mapping[str, Any]) -> tuple[str | None, list[tuple[str, Any]]]:
    """Return one non-overlapping ground-truth universe.

    Chunk identifiers are the strongest labels.  If unavailable, specific
    textual evidence is preferred over document IDs and exact names.  Selecting
    one universe prevents the same relevant chunk from being counted four times.
    """

    def exact_targets(kind: str, field: str) -> list[tuple[str, Any]]:
        unique: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for value in _as_list(case.get(field)):
            normalized = (
                _normalized_name(value)
                if kind == "document_name"
                else _normalized_identifier(value)
            )
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append((f"{kind}:{normalized}", value))
        return unique

    # An unanswerable case without explicit positive evidence has no meaningful
    # retrieval ground truth.  ``relevant_document_names`` still records the
    # dataset/source boundary, but treating the whole source document as a
    # relevant hit would turn arbitrary chunks into positives and would make a
    # retrieval-only run fail merely because it cannot execute the abstention
    # decision.  Keep retrieval N/A; full-query evaluation owns the refusal
    # metric.  Explicit chunk/evidence labels continue to opt into retrieval
    # scoring when a dataset really does provide a positive target.
    if (
        bool(case.get("unanswerable"))
        and not _as_list(case.get("relevant_chunk_ids"))
        and not _as_list(case.get("relevant_evidence"))
    ):
        return None, []

    # Chunk IDs are the strongest possible labels.
    chunk_targets = exact_targets("chunk_id", "relevant_chunk_ids")
    if chunk_targets:
        return "chunk_id", chunk_targets

    evidence_targets: list[tuple[str, Any]] = []
    for index, raw in enumerate(_as_list(case.get("relevant_evidence"))):
        if not isinstance(raw, Mapping):
            continue
        evidence = dict(raw)
        evidence_id = _normalized_identifier(evidence.get("evidence_id")) or str(index + 1)
        if (
            _specific_match_terms(evidence.get("match_terms"))
            or len(_compact_text(evidence.get("text"))) >= 12
        ):
            evidence_targets.append((f"evidence:{evidence_id}", evidence))
    # When an evidence span is supplied it is more discriminating than merely
    # belonging to the same document.  This prevents every chunk from a one-PDF
    # knowledge base from being counted as relevant.
    if evidence_targets:
        return "evidence", evidence_targets

    for kind, field in (
        ("document_id", "relevant_document_ids"),
        ("document_name", "relevant_document_names"),
    ):
        targets = exact_targets(kind, field)
        if targets:
            return kind, targets
    return None, []


def _candidate_target_matches(
    candidate: Mapping[str, Any],
    target_kind: str | None,
    targets: Sequence[tuple[str, Any]],
) -> set[str]:
    if not target_kind or not targets:
        return set()

    if target_kind == "chunk_id":
        actual = _normalized_identifier(candidate.get("chunk_id") or candidate.get("id"))
        return {
            key
            for key, expected in targets
            if actual and actual == _normalized_identifier(expected)
        }
    if target_kind == "document_id":
        actual = _normalized_identifier(candidate.get("document_id") or candidate.get("doc_id"))
        return {
            key
            for key, expected in targets
            if actual and actual == _normalized_identifier(expected)
        }
    if target_kind == "document_name":
        actual = _normalized_name(
            candidate.get("file_title")
            or candidate.get("document_name")
            or candidate.get("title")
            or candidate.get("file_name")
        )
        return {key for key, expected in targets if actual and actual == _normalized_name(expected)}

    body = _candidate_text(candidate)
    compact_body = _compact_text(body)
    matches: set[str] = set()
    for key, raw in targets:
        if not isinstance(raw, Mapping):
            continue
        terms = _specific_match_terms(raw.get("match_terms"))
        if terms and all(_contains_term(body, term) for term in terms):
            matches.add(key)
            continue
        evidence_text = _compact_text(raw.get("text"))
        if len(evidence_text) >= 12 and evidence_text in compact_body:
            matches.add(key)
    return matches


def _discounted_gain(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def _retrieval_metrics(
    case: Mapping[str, Any], candidates: list[dict[str, Any]], k_values: Sequence[int]
) -> tuple[dict[str, Any], str | None, list[tuple[str, Any]]]:
    target_kind, targets = _target_definition(case)
    match_sets = [
        _candidate_target_matches(candidate, target_kind, targets) for candidate in candidates
    ]
    relevant_flags = [bool(matches) for matches in match_sets]
    first_rank = next(
        (index + 1 for index, relevant in enumerate(relevant_flags) if relevant), None
    )
    target_count = len(targets)

    recalls: dict[str, float | None] = {}
    ndcgs: dict[str, float | None] = {}
    for k in k_values:
        seen: set[str] = set()
        unique_relevance: list[int] = []
        for matches in match_sets[:k]:
            newly_found = matches - seen
            unique_relevance.append(1 if newly_found else 0)
            seen.update(matches)
        recalls[str(k)] = len(seen) / target_count if target_count else None
        if target_count:
            ideal = [1] * min(target_count, k)
            ideal.extend([0] * max(0, min(k, len(candidates)) - len(ideal)))
            idcg = _discounted_gain(ideal)
            ndcgs[str(k)] = _discounted_gain(unique_relevance) / idcg if idcg else 0.0
        else:
            ndcgs[str(k)] = None

    all_matched = set().union(*match_sets) if match_sets else set()
    return (
        {
            "ground_truth_kind": target_kind,
            "ground_truth_count": target_count,
            "retrieved_count": len(candidates),
            "relevant_retrieved_count": len(all_matched),
            "first_relevant_rank": first_rank,
            "mrr": (1.0 / first_rank)
            if first_rank is not None
            else (0.0 if target_count else None),
            "recall_at_k": recalls,
            "ndcg_at_k": ndcgs,
        },
        target_kind,
        targets,
    )


_RETRIEVAL_DIAGNOSTIC_STAGES = ("retrieved", "rrf", "reranked", "context")


def _retrieval_stage_candidates(
    observation: Mapping[str, Any], stage: str
) -> list[dict[str, Any]] | None:
    raw = observation.get(stage)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _retrieval_stage_qrel_survival(
    observation: Mapping[str, Any],
    target_kind: str | None,
    targets: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """逐阶段报告 qrel 留存，不改变现有最终窗口排序指标。"""

    target_count = len(targets)
    stages: dict[str, dict[str, Any]] = {}
    matched_by_stage: dict[str, set[str] | None] = {}
    for stage in _RETRIEVAL_DIAGNOSTIC_STAGES:
        candidates = _retrieval_stage_candidates(observation, stage)
        if candidates is None:
            matched_by_stage[stage] = None
            stages[stage] = {
                "status": "unknown",
                "candidate_count": None,
                "matched_qrel_count": None,
                "qrel_recall": None,
                "first_relevant_rank": None,
            }
            continue
        match_sets = [
            _candidate_target_matches(candidate, target_kind, targets) for candidate in candidates
        ]
        matched = set().union(*match_sets) if match_sets else set()
        matched_by_stage[stage] = matched
        first_relevant_rank = next(
            (index + 1 for index, values in enumerate(match_sets) if values),
            None,
        )
        stages[stage] = {
            "status": "available" if target_count else "not_applicable",
            "candidate_count": len(candidates),
            "matched_qrel_count": len(matched) if target_count else None,
            "qrel_recall": len(matched) / target_count if target_count else None,
            "first_relevant_rank": first_relevant_rank if target_count else None,
        }

    transitions: dict[str, dict[str, Any]] = {}
    for previous, current in zip(
        _RETRIEVAL_DIAGNOSTIC_STAGES,
        _RETRIEVAL_DIAGNOSTIC_STAGES[1:],
        strict=False,
    ):
        previous_matches = matched_by_stage[previous]
        current_matches = matched_by_stage[current]
        key = f"{previous}_to_{current}"
        if previous_matches is None or current_matches is None:
            transitions[key] = {
                "status": "unknown",
                "retained_qrel_count": None,
                "lost_qrel_count": None,
                "survival_rate": None,
            }
            continue
        retained = previous_matches & current_matches
        transitions[key] = {
            "status": "available" if target_count else "not_applicable",
            "retained_qrel_count": len(retained) if target_count else None,
            "lost_qrel_count": (len(previous_matches - current_matches) if target_count else None),
            "survival_rate": (
                len(retained) / len(previous_matches) if target_count and previous_matches else None
            ),
        }
    return {
        "diagnostic_version": "stt.eval.retrieval-stage-qrel.v1",
        "stages": stages,
        "transitions": transitions,
    }


def _semantic_units(value: Any) -> set[str]:
    text = _normalized_text(value)
    units = {match.group(0) for match in _ASCII_WORD_RE.finditer(text) if len(match.group(0)) >= 2}
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            continue
        if len(run) == 2:
            units.add(run)
        else:
            units.update(run[index : index + 2] for index in range(len(run) - 1))
    return units


def _f1(reference: Any, answer: Any) -> float | None:
    reference_units = _semantic_units(reference)
    if not reference_units:
        return None
    answer_units = _semantic_units(answer)
    if not answer_units:
        return 0.0
    overlap = len(reference_units & answer_units)
    precision = overlap / len(answer_units)
    recall = overlap / len(reference_units)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _reference_constraints_preserved(reference: Any, answer: Any) -> bool:
    reference_text = _text(reference)
    answer_text = _text(answer)
    # High lexical overlap must not turn reversed requirements or negations
    # into semantic matches (for example “必须打开” versus “不得打开”).
    if not local_claim_bindings_preserved(reference_text, answer_text):
        return False
    if not _critical_identifiers_preserved(reference_text, answer_text):
        return False
    return True


def _critical_identifiers(value: Any) -> set[str]:
    return {
        token
        for token in ascii_token_keys(_text(value))
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }


def _critical_identifiers_preserved(reference: Any, answer: Any) -> bool:
    return _critical_identifiers(reference) <= ascii_token_keys(_text(answer))


def _critical_identifiers_without_conflict(reference: Any, answer: Any) -> bool:
    return _critical_identifier_conflict_count(reference, answer) == 0


def _canonical_critical_identifiers(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    raw_identifiers = _critical_identifiers(text)
    raw_identifiers.update(
        match.group(0)
        for match in re.finditer(
            r"(?<![a-z0-9])[a-z][a-z0-9]*"
            r"(?:\s*[-._:/]\s*|\s+)\d+(?:[._:/-]\d+)*(?:[a-z]+)?"
            r"(?![a-z0-9])",
            text,
            re.IGNORECASE,
        )
    )
    return {
        re.sub(r"[^a-z0-9]", "", identifier.casefold())
        for identifier in raw_identifiers
        if any(character.isalpha() for character in identifier)
        and any(character.isdigit() for character in identifier)
    }


def _identifier_family(identifier: str) -> str:
    match = re.match(r"[a-z]+", identifier)
    return match.group(0) if match else ""


def _critical_identifier_conflict_count(reference: Any, answer: Any) -> int:
    reference_identifiers = _canonical_critical_identifiers(reference)
    answer_identifiers = _canonical_critical_identifiers(answer)
    count = 0
    for identifier in reference_identifiers - answer_identifiers:
        family = _identifier_family(identifier)
        if family and any(
            candidate != identifier and _identifier_family(candidate) == family
            for candidate in answer_identifiers
        ):
            count += 1
    return count


def _looks_like_list_ordinal(text: str, match: re.Match[str]) -> bool:
    """Do not turn list numbering (``1）``) into an answer fact."""

    following = text[match.end() : match.end() + 1]
    if following not in {"、", ")", "）"}:
        return False
    prefix = text[: match.start()].rstrip()
    return not prefix or prefix[-1:] in {"。", "；", ";", "\n", ":", "："}


def _looks_like_identifier_number_fragment(text: str, match: re.Match[str]) -> bool:
    start = match.start()
    return start >= 2 and text[start - 1] in {".", "_", "-", "/"} and text[start - 2].isalnum()


def _critical_numbers(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    return {
        _normalized_text(match.group(0))
        for match in _NUMBER_RE.finditer(text)
        if not _looks_like_list_ordinal(text, match)
        and not _looks_like_identifier_number_fragment(text, match)
    }


def _critical_numbers_preserved(reference: Any, answer: Any) -> bool:
    return _critical_number_conflict_count(reference, answer) == 0


def _non_numeric_semantic_units(value: Any) -> set[str]:
    return {
        unit
        for unit in _semantic_units(value)
        if not any(character.isdigit() for character in unit)
    }


def _numeric_skeleton(value: Any) -> str:
    return re.sub(r"\d+", "#", _normalized_text(value))


def _critical_number_conflict_count(reference: Any, answer: Any) -> int:
    reference_text = _text(reference)
    answer_text = _text(answer)
    reference_clauses = [
        part.strip() for part in _REFERENCE_CLAUSE_SPLIT_RE.split(reference_text) if part.strip()
    ] or [reference_text]
    answer_clauses = [
        part.strip() for part in _REFERENCE_CLAUSE_SPLIT_RE.split(answer_text) if part.strip()
    ] or [answer_text]
    count = 0
    for reference_clause in reference_clauses:
        reference_numbers = _critical_numbers(reference_clause)
        if not reference_numbers:
            continue
        reference_units = _non_numeric_semantic_units(reference_clause)
        best: tuple[float, set[str]] | None = None
        for answer_clause in answer_clauses:
            answer_numbers = _critical_numbers(answer_clause)
            if not answer_numbers:
                continue
            answer_units = _non_numeric_semantic_units(answer_clause)
            shared = len(reference_units & answer_units)
            denominator = max(1, min(len(reference_units), len(answer_units)))
            same_numeric_skeleton = _numeric_skeleton(reference_clause) == _numeric_skeleton(
                answer_clause
            )
            alignment = 1.0 if same_numeric_skeleton else shared / denominator
            if not same_numeric_skeleton and (shared < 2 or alignment < 0.5):
                continue
            if best is None or alignment > best[0]:
                best = (alignment, answer_numbers)
        if best is not None:
            missing_numbers = reference_numbers - best[1]
            novel_numbers = best[1] - reference_numbers
            count += min(len(missing_numbers), len(novel_numbers))
    return count


def _polarity_constraints_preserved(reference: Any, answer: Any) -> bool:
    claims = _answer_factual_claims(_text(answer))
    return _contradictory_factual_claim_count(reference, claims) == 0


def _semantic_recall(reference: Any, answer: Any) -> float | None:
    """Measure whether the final answer covers the reference's semantic units.

    Unlike token F1, this intentionally does not penalize additional answer
    detail; unsupported additions are handled independently by faithfulness and
    citation checks.  This also tolerates harmless grammatical insertions such
    as ``约 3000 页`` versus ``约为 3000 页`` without introducing an LLM judge.
    """

    reference_text = _text(reference)
    answer_text = _text(answer)
    reference_units = _semantic_units(reference_text)
    if not reference_units:
        return None
    answer_units = _semantic_units(answer_text)
    if not answer_units:
        return 0.0
    if not _reference_constraints_preserved(reference_text, answer_text):
        return 0.0
    return len(reference_units & answer_units) / len(reference_units)


def _semantic_recall_without_legacy_constraints(reference: Any, answer: Any) -> float | None:
    reference_units = _semantic_units(_text(reference))
    if not reference_units:
        return None
    answer_units = _semantic_units(_text(answer))
    if not answer_units:
        return 0.0
    return len(reference_units & answer_units) / len(reference_units)


def _is_abstention(answer: Any, explicit: Any = None) -> bool:
    if isinstance(explicit, bool):
        return explicit
    normalized = _normalized_text(answer)
    if not normalized:
        return True
    return any(pattern.search(normalized) for pattern in _ABSTENTION_PATTERNS)


def _is_contract_abstention(answer: Any, explicit: Any = None) -> bool:
    """Resolve v3 abstention without trusting a stale explicit ``False`` flag."""

    if explicit is True:
        return True
    normalized = _normalized_text(answer)
    if not normalized:
        return True
    return _is_abstention(normalized, None) or any(
        pattern.search(normalized) for pattern in _CONTRACT_ABSTENTION_PATTERNS
    )


def _claims(answer: Any) -> list[str]:
    return [item[0] for item in _claim_segments(_text(answer), _CLAIM_SPLIT_RE)]


def _claim_segments(
    answer: str,
    boundary_re: re.Pattern[str],
) -> list[tuple[str, int, int, tuple[str, ...]]]:
    """返回可见 claim、原答案坐标和同行引用；marker-only 段归前句。"""

    segments: list[tuple[str, int, int, tuple[str, ...]]] = []
    cursor = 0
    boundaries = [*boundary_re.finditer(answer), None]
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
            leading_labels.extend(_inline_citation_labels(leading.group(0)))
            remaining_offset += leading.end()
            remaining = remaining[leading.end() :]
        if leading_labels and segments:
            previous = segments[-1]
            segments[-1] = (
                previous[0],
                previous[1],
                previous[2],
                tuple(dict.fromkeys((*previous[3], *leading_labels))),
            )
        labels = tuple(dict.fromkeys(_inline_citation_labels(remaining)))
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
            segments.append((content, content_start, content_end, labels))
        elif labels and segments:
            previous = segments[-1]
            segments[-1] = (
                previous[0],
                previous[1],
                previous[2],
                tuple(dict.fromkeys((*previous[3], *labels))),
            )
        if boundary is not None:
            cursor = boundary.end()
    return segments


def _concept_group_specs(
    case: Mapping[str, Any],
    *,
    source_aligned: bool = False,
    diagnostic: bool = False,
) -> list[tuple[str | None, list[str], Mapping[str, Any]]]:
    raw = case.get("concept_groups")
    if not isinstance(raw, Mapping):
        return []
    all_of = raw.get("all_of")
    if not isinstance(all_of, Sequence) or isinstance(all_of, (str, bytes)):
        return []
    groups: list[tuple[str | None, list[str], Mapping[str, Any]]] = []
    for item in all_of:
        if not isinstance(item, Mapping):
            continue
        alignment = item.get("source_alignment")
        alignment_mode = (
            str(alignment.get("mode") or "required")
            if isinstance(alignment, Mapping)
            else "required"
        )
        is_diagnostic = source_aligned and alignment_mode == "diagnostic"
        if source_aligned and is_diagnostic != diagnostic:
            continue
        any_of = item.get("any_of")
        if not isinstance(any_of, Sequence) or isinstance(any_of, (str, bytes)):
            continue
        variants = [_text(value).strip() for value in any_of if _text(value).strip()]
        # 来源锚点只证明 authored concept 可由 qrel 推导，绝不能成为答案的
        # 可接受变体。否则一个覆盖多个概念的长锚点会让缺失任一原子概念的
        # 答案被所有 group 同时命中，形成 evaluator leakage。
        variants = list(dict.fromkeys(variants))
        if variants:
            key = _text(item.get("concept_key")).strip() or None
            raw_satisfaction = item.get("satisfaction")
            satisfaction = (
                dict(raw_satisfaction)
                if isinstance(raw_satisfaction, Mapping)
                else {"mode": "answer_only"}
            )
            groups.append((key, variants, satisfaction))
    return groups


def _concept_group_variants(
    case: Mapping[str, Any],
    *,
    source_aligned: bool = False,
    diagnostic: bool = False,
) -> list[tuple[str | None, list[str]]]:
    """保留旧调用形状；v5 的满足模式由 ``_concept_group_specs`` 处理。"""

    return [
        (key, variants)
        for key, variants, _ in _concept_group_specs(
            case,
            source_aligned=source_aligned,
            diagnostic=diagnostic,
        )
    ]


def _concept_match_text(value: Any) -> str:
    """Normalize harmless formatting without erasing atomic-value boundaries."""

    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    # A comma between a digit and exactly one three-digit group is a thousands
    # separator for concept matching.  Removing it lets ``3,000`` match
    # ``3000`` while the atomic-value gate below still rejects ``13000``.
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
    # OCR and prose often insert spaces around identifier punctuation.  Keep
    # the punctuation itself so distinct atomic values are never compared by
    # substring after normalization.
    return re.sub(
        r"(?<=[a-z0-9])\s*([._:/-])\s*(?=[a-z0-9])",
        r"\1",
        normalized,
        flags=re.IGNORECASE,
    )


def _concept_atomic_values(value: Any) -> tuple[set[str], set[str]]:
    """Return exact identifier and numeric atoms used by authored concepts."""

    normalized = _concept_match_text(value)
    identifiers: set[str] = set()
    for match in _CONCEPT_ASCII_ATOM_RE.finditer(normalized):
        token = match.group(0)
        if not any(character.isalpha() for character in token) or not any(
            character.isdigit() for character in token
        ):
            continue
        # ``3000pages`` is a number plus unit, not a model identifier.  Tokens
        # with structural punctuation (for example ``2004/108/EC``) remain
        # identifiers even when they start with a digit.
        if token[0].isdigit() and re.fullmatch(r"\d+(?:\.\d+)*(?:%|[a-z]+)", token):
            continue
        identifiers.add(re.sub(r"[^a-z0-9]", "", token))
    for pattern in (_CONCEPT_SPACED_IDENTIFIER_RE, _CONCEPT_MODEL_SUFFIX_RE):
        identifiers.update(
            re.sub(r"[^a-z0-9]", "", match.group(0)) for match in pattern.finditer(normalized)
        )

    numbers: set[str] = set()
    for token in _critical_numbers(normalized):
        match = re.match(r"[-+]?\d+(?:\.\d+)*", token)
        if match is None:
            continue
        number = match.group(0)
        if token.endswith("%"):
            number += "%"
        numbers.add(number.lstrip("+"))

    # A numeric fragment bound into a model/version/directive is enforced by
    # the complete identifier.  Do not also require a standalone numeric token:
    # that would reject harmless ``Android4.4`` / ``Android 4.4`` spacing.
    identifier_digits = {
        "".join(character for character in identifier if character.isdigit())
        for identifier in identifiers
    }
    numbers = {
        number
        for number in numbers
        if "".join(character for character in number if character.isdigit())
        not in identifier_digits
    }
    return identifiers, numbers


def _generic_action_has_boundary(answer: str, variant: str) -> bool:
    normalized_answer = _normalized_text(answer)
    normalized_variant = _normalized_text(variant)
    if normalized_variant in _CONCEPT_GENERIC_CJK_ACTIONS:
        start = 0
        while (index := normalized_answer.find(normalized_variant, start)) >= 0:
            cursor = index + len(normalized_variant)
            while cursor < len(normalized_answer) and normalized_answer[cursor].isspace():
                cursor += 1
            if cursor == len(normalized_answer) or not normalized_answer[cursor].isalnum():
                return True
            start = index + 1
        return False
    if normalized_variant in _CONCEPT_GENERIC_ENGLISH_ACTIONS:
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(normalized_variant)}(?![a-z0-9])",
            normalized_answer,
            flags=re.IGNORECASE,
        ):
            cursor = match.end()
            while cursor < len(normalized_answer) and normalized_answer[cursor].isspace():
                cursor += 1
            if cursor == len(normalized_answer) or not normalized_answer[cursor].isalnum():
                return True
        return False
    return True


def _contains_confusable_cjk_sibling(answer: str, variant: str) -> bool:
    """检测答案中替代目标标签的等长一字差CJK sibling。

    例如“有线网络信息”和“无线网络信息”是两个独立菜单项；宽松bigram
    recall不能让其中一个覆盖另一个。目标原文已出现时不触发，包含关系、
    ASCII、数字与短标签也继续使用既有matcher规则。
    """

    if len(variant) < 5 or _CJK_RUN_RE.fullmatch(variant) is None or variant in answer:
        return False
    width = len(variant)
    for run in _CJK_RUN_RE.findall(answer):
        for index in range(max(0, len(run) - width + 1)):
            candidate = run[index : index + width]
            if sum(left != right for left, right in zip(candidate, variant)) == 1:
                return True
    return False


def concept_variant_covered_legacy(answer: str, variant: str) -> bool:
    """冻结v3 matcher；仅供旧materializer receipt exact replay。"""

    return _concept_variant_covered_impl(answer, variant, cjk_sibling_guard=False)


def concept_variant_covered(answer: str, variant: str) -> bool:
    """Return whether an answer covers one authored v3 concept variant.

    Atomic values are exact after harmless case/spacing/punctuation
    normalization.  This deliberately prevents IP, numeric, version and model
    substrings from satisfying a different concept.  Bare generic actions also
    require an answer boundary instead of inheriting an object's meaning.
    """

    return _concept_variant_covered_impl(answer, variant, cjk_sibling_guard=True)


def _concept_variant_covered_impl(answer: str, variant: str, *, cjk_sibling_guard: bool) -> bool:
    normalized_answer = _concept_match_text(answer)
    normalized_variant = _concept_match_text(variant)
    variant_identifiers, variant_numbers = _concept_atomic_values(normalized_variant)
    answer_identifiers, answer_numbers = _concept_atomic_values(normalized_answer)
    if not variant_identifiers <= answer_identifiers:
        return False
    if not variant_numbers <= answer_numbers:
        return False
    if not _generic_action_has_boundary(normalized_answer, normalized_variant):
        return False
    if _contains_term(normalized_answer, normalized_variant):
        return True
    if cjk_sibling_guard and _contains_confusable_cjk_sibling(
        normalized_answer, normalized_variant
    ):
        return False
    # Exact atomic identity has already been established.  Compact comparison
    # here permits ``TL-413H`` / ``tl 413h`` without reintroducing partial-ID
    # matching.
    if (variant_identifiers or variant_numbers) and _compact_text(
        normalized_variant
    ) in _compact_text(normalized_answer):
        return True
    recall = _semantic_recall(normalized_variant, normalized_answer)
    return recall is not None and recall >= 0.8


def _concept_variant_covered(answer: str, variant: str) -> bool:
    """Backward-compatible alias for the public production matcher."""

    return concept_variant_covered(answer, variant)


def _concept_group_metrics(
    case: Mapping[str, Any],
    answer: str,
    *,
    source_aligned: bool = False,
    diagnostic: bool = False,
    query: str = "",
    query_predicate_ellipsis: bool = False,
) -> tuple[float | None, int, list[int], list[str]]:
    groups = _concept_group_specs(
        case,
        source_aligned=source_aligned,
        diagnostic=diagnostic,
    )
    if not groups:
        return None, 0, [], []
    direct_coverage = {
        key: any(_concept_variant_covered(answer, variant) for variant in variants)
        for key, variants, _ in groups
        if key is not None
    }
    missing_indexes: list[int] = []
    missing_keys: list[str] = []
    for index, (key, variants, satisfaction) in enumerate(groups):
        covered = any(_concept_variant_covered(answer, variant) for variant in variants)
        if (
            not covered
            and query_predicate_ellipsis
            and key is not None
            and satisfaction.get("mode") == "query_predicate_ellipsis"
        ):
            raw_query_variants = satisfaction.get("query_any_of")
            query_variants = (
                [_text(item).strip() for item in raw_query_variants if _text(item).strip()]
                if isinstance(raw_query_variants, Sequence)
                and not isinstance(raw_query_variants, (str, bytes))
                else []
            )
            raw_required = satisfaction.get("requires_concept_keys")
            required_keys = (
                [_text(item).strip() for item in raw_required if _text(item).strip()]
                if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes))
                else []
            )
            # 查询只恢复展示中省略的谓词。所有主语必须由答案直接给出，且
            # 冻结的 authored-claim 合同必须逐一把“主语+谓词”绑定到 qrel。
            covered = (
                bool(query_variants)
                and any(_concept_variant_covered(query, variant) for variant in query_variants)
                and bool(required_keys)
                and all(direct_coverage.get(required_key) is True for required_key in required_keys)
                and _authored_claim_contract_covers_ellipsis(
                    case,
                    predicate_key=key,
                    subject_keys=required_keys,
                )
            )
        if covered:
            continue
        missing_indexes.append(index)
        missing_keys.append(key or str(index))
    coverage = (len(groups) - len(missing_indexes)) / len(groups)
    return coverage, len(groups), missing_indexes, missing_keys


def _authored_claim_contract_covers_ellipsis(
    case: Mapping[str, Any],
    *,
    predicate_key: str,
    subject_keys: Sequence[str],
) -> bool:
    raw = case.get("authored_claims")
    if not isinstance(raw, Mapping) or raw.get("contract_version") != (
        AUTHORED_CLAIM_CONTRACT_VERSION
    ):
        return False
    atoms = raw.get("atoms")
    if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)):
        return False
    requirements: list[set[str]] = []
    for atom in atoms:
        if not isinstance(atom, Mapping):
            return False
        raw_required = atom.get("requires_concept_keys")
        if not isinstance(raw_required, Sequence) or isinstance(raw_required, (str, bytes)):
            return False
        required = {_text(item).strip() for item in raw_required if _text(item).strip()}
        if len(required) != len(raw_required) or len(required) < 2:
            return False
        if not isinstance(atom.get("citation_required"), bool):
            return False
        alignment = atom.get("source_alignment")
        if not isinstance(alignment, Mapping):
            return False
        anchors = alignment.get("anchors")
        source_terms = alignment.get("source_terms")
        if (
            not isinstance(anchors, Sequence)
            or isinstance(anchors, (str, bytes))
            or not anchors
            or not isinstance(source_terms, Sequence)
            or isinstance(source_terms, (str, bytes))
            or not source_terms
        ):
            return False
        requirements.append(required)
    return all(
        any({predicate_key, subject_key} <= required for required in requirements)
        for subject_key in subject_keys
    )


def _source_aligned_concept_contract_is_explicit(
    case: Mapping[str, Any],
    *,
    expected_version: str = "stt.eval.authored-concepts.v2",
) -> bool:
    raw = case.get("concept_groups")
    if not isinstance(raw, Mapping) or raw.get("contract_version") != expected_version:
        return False
    all_of = raw.get("all_of")
    if not isinstance(all_of, Sequence) or isinstance(all_of, (str, bytes)):
        return False
    required_count = 0
    concept_keys: set[str] = set()
    ellipsis_requirements: dict[str, list[str]] = {}
    for group in all_of:
        if not isinstance(group, Mapping):
            return False
        alignment = group.get("source_alignment")
        if not isinstance(alignment, Mapping):
            return False
        mode = alignment.get("mode")
        if mode == "diagnostic":
            continue
        if mode != "required":
            return False
        anchors = alignment.get("anchors")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            return False
        if not anchors:
            return False
        source_terms = alignment.get("source_terms")
        if not isinstance(source_terms, Sequence) or isinstance(source_terms, (str, bytes)):
            return False
        if not source_terms or not all(
            isinstance(term, str) and term.strip() for term in source_terms
        ):
            return False
        key = _text(group.get("concept_key")).strip()
        if not key or key in concept_keys:
            return False
        concept_keys.add(key)
        if expected_version == AUTHORED_CONCEPT_CONTRACT_VERSION:
            satisfaction = group.get("satisfaction")
            if satisfaction is None:
                satisfaction = {"mode": "answer_only"}
            if not isinstance(satisfaction, Mapping):
                return False
            satisfaction_mode = satisfaction.get("mode")
            if satisfaction_mode not in {"answer_only", "query_predicate_ellipsis"}:
                return False
            if satisfaction_mode == "query_predicate_ellipsis":
                query_any_of = satisfaction.get("query_any_of")
                required_keys = satisfaction.get("requires_concept_keys")
                if (
                    not isinstance(query_any_of, Sequence)
                    or isinstance(query_any_of, (str, bytes))
                    or not query_any_of
                    or not all(isinstance(item, str) and item.strip() for item in query_any_of)
                    or not isinstance(required_keys, Sequence)
                    or isinstance(required_keys, (str, bytes))
                    or not required_keys
                    or not all(isinstance(item, str) and item.strip() for item in required_keys)
                ):
                    return False
                ellipsis_requirements[key] = [str(item).strip() for item in required_keys]
        required_count += 1
    if required_count == 0:
        return False
    for predicate_key, subject_keys in ellipsis_requirements.items():
        if predicate_key in subject_keys or not set(subject_keys) <= concept_keys:
            return False
        if not _authored_claim_contract_covers_ellipsis(
            case,
            predicate_key=predicate_key,
            subject_keys=subject_keys,
        ):
            return False
    return True


def _claim_binding_text(value: Any) -> str:
    """规整表格行，同时保留 ○/× 这类有语义的状态符号。"""

    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u3400" <= character <= "\u9fff" or character in {"○", "×"}
    )


def _v5_concept_coverage_by_key(
    case: Mapping[str, Any],
    *,
    answer: str,
) -> tuple[dict[str, bool], set[str]]:
    specs = _concept_group_specs(case, source_aligned=True)
    direct = {
        key: any(_concept_variant_covered(answer, variant) for variant in variants)
        for key, variants, _ in specs
        if key is not None
    }
    covered = dict(direct)
    ellipsis_keys: set[str] = set()
    query = _text(case.get("query"))
    for key, variants, satisfaction in specs:
        if key is None or satisfaction.get("mode") != "query_predicate_ellipsis":
            continue
        ellipsis_keys.add(key)
        raw_query_variants = satisfaction.get("query_any_of")
        query_variants = (
            [_text(item).strip() for item in raw_query_variants if _text(item).strip()]
            if isinstance(raw_query_variants, Sequence)
            and not isinstance(raw_query_variants, (str, bytes))
            else []
        )
        raw_required = satisfaction.get("requires_concept_keys")
        required_keys = (
            [_text(item).strip() for item in raw_required if _text(item).strip()]
            if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes))
            else []
        )
        covered[key] = bool(
            direct.get(key)
            or (
                query_variants
                and any(_concept_variant_covered(query, variant) for variant in query_variants)
                and required_keys
                and all(direct.get(required_key) is True for required_key in required_keys)
                and _authored_claim_contract_covers_ellipsis(
                    case,
                    predicate_key=key,
                    subject_keys=required_keys,
                )
            )
        )
    return covered, ellipsis_keys


def _authored_claim_source_supported(
    alignment: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    candidate_text = _claim_binding_text(_candidate_text(candidate))
    anchors = alignment.get("anchors")
    source_terms = alignment.get("source_terms")
    if (
        not candidate_text
        or not isinstance(anchors, Sequence)
        or isinstance(anchors, (str, bytes))
        or not isinstance(source_terms, Sequence)
        or isinstance(source_terms, (str, bytes))
    ):
        return False
    normalized_terms = [
        _claim_binding_text(term) for term in source_terms if _claim_binding_text(term)
    ]
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        normalized_anchor = _claim_binding_text(anchor.get("text"))
        if (
            normalized_anchor
            and normalized_anchor in candidate_text
            and all(term in normalized_anchor for term in normalized_terms)
        ):
            return True
    return False


def _authored_claim_metrics(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    context: Sequence[Mapping[str, Any]],
    *,
    answer: str,
) -> tuple[float | None, int, int, list[str]]:
    """把“查询谓词 + 列表主语”还原为逐行主张并在实时 Evidence 上复核。"""

    raw_contract = case.get("authored_claims")
    if not isinstance(raw_contract, Mapping) or raw_contract.get("contract_version") != (
        AUTHORED_CLAIM_CONTRACT_VERSION
    ):
        return None, 0, 0, []
    atoms = raw_contract.get("atoms")
    if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)) or not atoms:
        return None, 0, 0, []

    concept_specs = {
        key: variants
        for key, variants, _ in _concept_group_specs(case, source_aligned=True)
        if key is not None
    }
    concept_covered, ellipsis_keys = _v5_concept_coverage_by_key(case, answer=answer)

    citation_claims: list[tuple[str, Mapping[str, Any]]] = []
    for raw_citation in _as_list(observation.get("citations")):
        if not isinstance(raw_citation, Mapping):
            continue
        candidate = _resolve_candidate(raw_citation, context)
        claim_text = _text(raw_citation.get("claim_text")).strip()
        if candidate is not None and claim_text:
            citation_claims.append((claim_text, candidate))
    for claim_text, _, _, labels in _claim_segments(answer, _CLAIM_SPLIT_RE):
        for label in labels:
            candidate = _resolve_inline_label(label, context)
            if candidate is not None:
                citation_claims.append((claim_text, candidate))

    supported = 0
    missing_keys: list[str] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            missing_keys.append(str(index))
            continue
        claim_key = _text(atom.get("claim_key")).strip() or str(index)
        raw_required = atom.get("requires_concept_keys")
        required_keys = (
            [_text(item).strip() for item in raw_required if _text(item).strip()]
            if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes))
            else []
        )
        if not required_keys or not all(concept_covered.get(key) is True for key in required_keys):
            missing_keys.append(claim_key)
            continue
        alignment = atom.get("source_alignment")
        if not isinstance(alignment, Mapping):
            missing_keys.append(claim_key)
            continue
        source_candidates = [
            candidate
            for candidate in context
            if _authored_claim_source_supported(alignment, candidate)
        ]
        if not source_candidates:
            missing_keys.append(claim_key)
            continue
        if atom.get("citation_required") is True:
            subject_keys = [key for key in required_keys if key not in ellipsis_keys]
            citation_supported = False
            for claim_text, candidate in citation_claims:
                if candidate not in source_candidates:
                    continue
                if any(
                    any(
                        _concept_variant_covered(claim_text, variant)
                        for variant in concept_specs.get(subject_key, [])
                    )
                    for subject_key in subject_keys
                ):
                    citation_supported = True
                    break
            if not citation_supported:
                missing_keys.append(claim_key)
                continue
        supported += 1
    return supported / len(atoms), len(atoms), supported, missing_keys


def _reference_required_branches(query: Any, reference: Any) -> list[str]:
    """Infer only explicit multi-branch contracts from query/reference structure."""

    if not _MULTI_BRANCH_QUERY_RE.search(_text(query)):
        return []
    reference_text = _text(reference).strip()
    branches = [
        item.strip(" ，,、:：")
        for item in _REFERENCE_CLAUSE_SPLIT_RE.split(reference_text)
        if item.strip(" ，,、:：")
    ]
    if len(branches) < 2:
        enumerated = [item.strip() for item in reference_text.split("、") if item.strip()]
        if len(enumerated) >= 3:
            branches = enumerated
    if len(branches) < 2:
        comma_parts = [item.strip() for item in re.split(r"[，,]", reference_text) if item.strip()]
        if len(comma_parts) >= 4:
            branches = comma_parts
    return [
        branch
        for branch in branches
        if len(_semantic_units(branch)) >= 2 or bool(_critical_numbers(branch))
    ]


def _required_branch_metrics(
    case: Mapping[str, Any], answer: str
) -> tuple[float | None, int, list[int]]:
    branches = _reference_required_branches(case.get("query"), case.get("reference_answer"))
    # Without authored concept_groups, only the high-confidence two-branch
    # shape is a hard contract.  Treating every semicolon in a long reference
    # as an independent required answer branch confuses source conditions and
    # optional detail with what the query explicitly asks.  Three-or-more
    # all-of contracts should be declared with concept_groups.
    if len(branches) != 2:
        return None, 0, []
    missing: list[int] = []
    compact_answer = _compact_text(answer)
    answer_units = _semantic_units(answer)
    branch_units = [_semantic_units(branch) for branch in branches]
    for index, branch in enumerate(branches):
        compact_branch = _compact_text(branch)
        if compact_branch and compact_branch in compact_answer:
            covered = True
        else:
            other_units = set().union(
                *(units for other_index, units in enumerate(branch_units) if other_index != index)
            )
            distinguishing_units = branch_units[index] - other_units
            if distinguishing_units:
                distinguishing_coverage = len(distinguishing_units & answer_units) / len(
                    distinguishing_units
                )
                raw_recall = (
                    len(branch_units[index] & answer_units) / len(branch_units[index])
                    if branch_units[index]
                    else 0.0
                )
                covered = distinguishing_coverage >= 0.45 or raw_recall >= 0.65
            else:
                raw_recall = (
                    len(branch_units[index] & answer_units) / len(branch_units[index])
                    if branch_units[index]
                    else 0.0
                )
                covered = raw_recall >= 0.65
        if not covered:
            missing.append(index)
    return (len(branches) - len(missing)) / len(branches), len(branches), missing


def _answer_factual_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for claim, *_ in _claim_segments(answer, _CLAIM_SPLIT_RE):
        for part in _SECONDARY_CLAIM_SPLIT_RE.split(claim):
            cleaned = _INLINE_CITATION_RE.sub("", part).strip(" ，,、:：")
            cleaned = _MARKDOWN_PREFIX_RE.sub("", cleaned).strip(" *_`：:")
            cleaned = re.sub(
                r"^(?:此外|另外|但是|但|然而|不过|同时|并且|而且)[，,:：\s]*",
                "",
                cleaned,
            )
            if (
                not cleaned
                or _NON_FACTUAL_META_RE.fullmatch(cleaned)
                or _URL_ONLY_RE.fullmatch(cleaned)
                or _IMAGE_PLACEHOLDER_RE.fullmatch(cleaned)
            ):
                continue
            if (
                len(_semantic_units(cleaned)) >= 2
                or bool(_critical_numbers(cleaned))
                or bool(_critical_identifiers(cleaned))
            ):
                claims.append(cleaned)
    return claims


def _safe_evidence_scope_absence(value: str) -> bool:
    return bool(_EVIDENCE_ABSENCE_RE.search(value)) and (
        bool(_EVIDENCE_SCOPE_RE.search(value))
        or bool(re.match(r"^\s*(?:均)?(?:未|没有|不)", value))
    )


def _abstention_factual_claims(answer: str, context_texts: Sequence[str]) -> list[str]:
    body = _INLINE_CITATION_RE.sub("", answer)
    claims: list[str] = []
    for raw in re.split(r"[。！？!?；;\n]+", body):
        for part in re.split(
            r"(?:[，,]\s*)?(?:(?:但是|但|然而|不过|且|并且|同时))",
            raw,
        ):
            cleaned_part = _MARKDOWN_PREFIX_RE.sub("", part).strip()
            residual = _ABSTENTION_CLAUSE_RE.sub("", cleaned_part)
            for pattern in (*_ABSTENTION_PATTERNS, *_CONTRACT_ABSTENTION_PATTERNS):
                residual = pattern.sub("", residual)
            residual = re.sub(
                r"^[\s，,、:：]*(?:因此|所以|但|但是|不过|然而)?[\s，,、:：]*",
                "",
                residual,
            ).strip()
            if (
                not residual
                or _SAFE_ABSTENTION_RESIDUAL_RE.fullmatch(residual)
                or _safe_evidence_scope_absence(cleaned_part)
                or _claim_support_confidence(residual, context_texts) > 0.2
            ):
                continue
            if (
                bool(_critical_numbers(residual))
                or bool(_critical_identifiers(residual))
                or bool(_FACTUAL_ASSERTION_RE.search(residual))
            ):
                claims.append(residual)
    return claims


def _contradictory_factual_claim_count(reference: Any, claims: Sequence[str]) -> int:
    reference_text = _text(reference)
    reference_parts = [
        part.strip() for part in _REFERENCE_CLAUSE_SPLIT_RE.split(reference_text) if part.strip()
    ] or [reference_text]
    reference_units_by_part = [(part, _semantic_units(part)) for part in reference_parts]
    if not any(units for _, units in reference_units_by_part):
        return 0
    count = 0
    for claim in claims:
        claim_units = _semantic_units(claim)
        if not claim_units:
            continue
        best_part, best_units = max(
            reference_units_by_part,
            key=lambda item: len(item[1] & claim_units),
        )
        shared = len(best_units & claim_units)
        claim_overlap = shared / len(claim_units)
        reference_overlap = shared / len(best_units) if best_units else 0.0
        reference_negative = bool(_NEGATIVE_INTENT_RE.search(best_part))
        claim_negative = bool(_NEGATIVE_INTENT_RE.search(claim))
        if (
            claim_overlap >= 0.65
            and reference_overlap >= 0.65
            and reference_negative != claim_negative
        ):
            count += 1
    return count


def _critical_ascii_identifiers_are_preserved(claim: str, evidence: str) -> bool:
    """连续字母数字标识符必须作为同一个原始 token 出现在证据中。"""

    claim_identifiers = {
        token
        for token in ascii_token_keys(claim)
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }
    return claim_identifiers <= ascii_token_keys(evidence)


def _metric_claim_window_supported(
    claim: str,
    evidence_window: str,
    *,
    full_evidence: str,
) -> bool:
    """在单个局部窗内复核字面量、数字、极性、绑定与保守语义覆盖。"""

    evidence_window = restore_same_evidence_placeholders(
        claim,
        evidence_window,
        full_evidence,
    )
    claim_units = _semantic_units(claim)
    if len(claim_units) < 2:
        return False
    if not _critical_ascii_identifiers_are_preserved(claim, evidence_window):
        return False
    claim_numbers = {_normalized_text(value) for value in _NUMBER_RE.findall(claim)}
    evidence_numbers = {_normalized_text(value) for value in _NUMBER_RE.findall(evidence_window)}
    if not claim_numbers.issubset(evidence_numbers):
        return False
    if bool(_NEGATIVE_INTENT_RE.search(claim)) != bool(_NEGATIVE_INTENT_RE.search(evidence_window)):
        return False
    if not local_claim_bindings_preserved(claim, evidence_window):
        return False
    evidence_units = _semantic_units(evidence_window)
    return len(claim_units & evidence_units) / len(claim_units) >= 0.55


def _claim_supported(claim: str, context_texts: Sequence[str]) -> bool:
    compact_claim = _compact_text(claim)
    if not compact_claim:
        return False
    for context_text in context_texts:
        restored_context = restore_same_evidence_placeholders(
            claim,
            context_text,
            context_text,
        )
        if (
            len(compact_claim) >= 4
            and compact_claim in _compact_text(restored_context)
            and _critical_ascii_identifiers_are_preserved(claim, restored_context)
            and local_claim_bindings_preserved(claim, restored_context)
        ):
            return True
    joined = " ".join(context_texts)
    joined = restore_same_evidence_placeholders(claim, joined, joined)
    claim_units = _semantic_units(claim)
    context_units = _semantic_units(joined)
    if len(claim_units) < 2:
        return False
    # Numbers are high-information claims and must all be grounded verbatim.
    claim_numbers = {_normalized_text(value) for value in _NUMBER_RE.findall(claim)}
    context_numbers = {_normalized_text(value) for value in _NUMBER_RE.findall(joined)}
    if not claim_numbers.issubset(context_numbers):
        return False
    if not _critical_ascii_identifiers_are_preserved(claim, joined):
        return False
    if not local_claim_bindings_preserved(claim, joined):
        return False
    return len(claim_units & context_units) / len(claim_units) >= 0.85


def _claim_support_confidence(claim: str, context_texts: Sequence[str]) -> float:
    """Return conservative lexical support confidence for v3 hard guards.

    The strict faithfulness metric intentionally keeps its higher threshold.
    Final correctness only treats a claim as definitely unsupported when the
    claim is substantially novel; medium-overlap paraphrases remain visible as
    unverified diagnostics instead of becoming automatic false negatives.
    """

    compact_claim = _compact_text(claim)
    if not compact_claim:
        return 0.0
    for context_text in context_texts:
        restored = restore_same_evidence_placeholders(
            claim,
            context_text,
            context_text,
        )
        if len(compact_claim) >= 4 and compact_claim in _compact_text(restored):
            return 1.0
    joined = " ".join(context_texts)
    joined = restore_same_evidence_placeholders(claim, joined, joined)
    claim_units = _semantic_units(claim)
    if not claim_units:
        return 0.0
    context_units = _semantic_units(joined)
    return len(claim_units & context_units) / len(claim_units)


def _high_confidence_unsupported_factual_claims(
    claims: Sequence[str], context_texts: Sequence[str]
) -> list[str]:
    unsupported: list[str] = []
    for claim in claims:
        if _claim_supported(claim, context_texts):
            continue
        is_assertion = bool(
            _FACTUAL_ASSERTION_RE.search(claim)
            or _critical_numbers(claim)
            or _critical_identifiers(claim)
        )
        if is_assertion and _claim_support_confidence(claim, context_texts) <= 0.2:
            unsupported.append(claim)
    return unsupported


def _citation_claim_supported(claim: str, evidence_text: str) -> bool:
    """在已精确解析的单条 Evidence 上独立复核结构化 citation claim。"""

    compact_claim = _compact_text(claim)
    restored_evidence = restore_same_evidence_placeholders(
        claim,
        evidence_text,
        evidence_text,
    )
    compact_evidence = _compact_text(restored_evidence)
    if not compact_claim or not compact_evidence:
        return False
    if (
        len(compact_claim) >= 4
        and compact_claim in compact_evidence
        and (
            _metric_claim_window_supported(
                claim,
                restored_evidence,
                full_evidence=evidence_text,
            )
        )
    ):
        return True
    claim_units = _semantic_units(claim)
    if len(claim_units) < 2:
        return False

    # 只在最相关的局部句/表格行上验真，避免整块其他行里的数字或否定词
    # 为当前声明“游离作证”。相邻两段用于容纳忠实合并的短句。
    pieces = [item.strip() for item in re.split(r"[。！？!?；;\n]+", evidence_text) if item.strip()]
    windows = [*pieces]
    windows.extend(f"{left}，{right}" for left, right in zip(pieces, pieces[1:], strict=False))
    if not windows:
        windows = [evidence_text]
    evidence_window = max(
        windows,
        key=lambda item: (
            len(claim_units & _semantic_units(item)) / len(claim_units),
            -len(item),
        ),
    )
    if _metric_claim_window_supported(
        claim,
        evidence_window,
        full_evidence=evidence_text,
    ):
        return True

    # 手册常先给“若发生以下情况则必须……”的统一处置，再在后续项目符中
    # 枚举具体条件。结构化 citation 可把条件和处置忠实合并为一个可见句；
    # 此时逐个逗号子句必须各自在同一 Evidence 的局部窗独立通过全部硬门禁。
    claim_clauses = [item.strip() for item in re.split(r"[，,]+", claim) if item.strip()]
    if len(claim_clauses) < 2:
        return False
    return all(
        _metric_claim_window_supported(
            clause,
            max(
                windows,
                key=lambda item: (
                    len(_semantic_units(clause) & _semantic_units(item))
                    / max(1, len(_semantic_units(clause))),
                    -len(item),
                ),
            ),
            full_evidence=evidence_text,
        )
        for clause in claim_clauses
    )


def _independently_supported_citation_spans(
    observation: Mapping[str, Any],
    answer: str,
    context: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """只接受坐标、claim_text、Evidence 身份和本地内容复核均成立的 span。"""

    supported: list[tuple[int, int]] = []
    for raw in _as_list(observation.get("citations")):
        if not isinstance(raw, Mapping):
            continue
        candidate = _resolve_candidate(raw, context)
        span = raw.get("answer_span")
        claim_text = _text(raw.get("claim_text")).strip()
        if candidate is None or not isinstance(span, Mapping) or not claim_text:
            continue
        start = span.get("start")
        end = span.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(answer)
        ):
            continue
        visible = answer[start:end]
        if _compact_text(visible) != _compact_text(claim_text):
            continue
        evidence_text = _candidate_text(candidate)
        if _citation_claim_supported(claim_text, evidence_text):
            supported.append((start, end))
    return supported


def _generation_metrics(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    context: list[dict[str, Any]],
    *,
    proposition_forbidden_matcher: bool,
    source_aligned_concept_contract: bool,
    authored_claim_contract: bool,
    temporal_derived_claim_contract: bool,
    proof_evaluation: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = _text(observation.get("answer"))
    unanswerable = bool(case.get("unanswerable", False))
    predicted_abstention = _is_abstention(answer, observation.get("abstained"))
    contract_predicted_abstention = _is_contract_abstention(answer, observation.get("abstained"))
    abstention = {
        "expected": unanswerable,
        "predicted": predicted_abstention,
        "correct": 1.0 if predicted_abstention == unanswerable else 0.0,
        "contract_predicted": contract_predicted_abstention,
        "contract_correct": (1.0 if contract_predicted_abstention == unanswerable else 0.0),
    }

    expected_terms = [
        _text(value) for value in _as_list(case.get("expected_terms")) if _text(value).strip()
    ]
    forbidden_terms = [
        _text(value) for value in _as_list(case.get("forbidden_terms")) if _text(value).strip()
    ]
    expected_recall = (
        sum(1 for term in expected_terms if _contains_term(answer, term)) / len(expected_terms)
        if expected_terms
        else None
    )
    forbidden_rate = (
        sum(
            1
            for term in forbidden_terms
            if (
                _forbidden_proposition_asserted(answer, term)
                if proposition_forbidden_matcher
                else _contains_term(answer, term)
            )
        )
        / len(forbidden_terms)
        if forbidden_terms
        else None
    )
    answer_without_citations = _INLINE_CITATION_RE.sub("", answer)
    reference_f1 = _f1(case.get("reference_answer"), answer_without_citations)
    reference_recall = _semantic_recall(case.get("reference_answer"), answer_without_citations)
    contract_reference_recall = _semantic_recall_without_legacy_constraints(
        case.get("reference_answer"), answer_without_citations
    )
    reference_constraints_preserved = _reference_constraints_preserved(
        case.get("reference_answer"), answer_without_citations
    )
    critical_number_conflict_count = (
        None
        if unanswerable
        else _critical_number_conflict_count(case.get("reference_answer"), answer_without_citations)
    )
    critical_numbers_preserved = (
        None if critical_number_conflict_count is None else critical_number_conflict_count == 0
    )
    critical_identifier_conflict_count = (
        None
        if unanswerable
        else _critical_identifier_conflict_count(
            case.get("reference_answer"), answer_without_citations
        )
    )
    critical_identifiers_preserved = (
        None
        if critical_identifier_conflict_count is None
        else critical_identifier_conflict_count == 0
    )
    polarity_constraints_preserved = (
        None
        if unanswerable
        else _polarity_constraints_preserved(case.get("reference_answer"), answer_without_citations)
    )
    (
        concept_group_coverage,
        concept_group_count,
        missing_concept_group_indexes,
        missing_concept_group_keys,
    ) = (
        (None, 0, [], [])
        if unanswerable
        else _concept_group_metrics(
            case,
            answer_without_citations,
            source_aligned=source_aligned_concept_contract,
            query=_text(case.get("query")),
            query_predicate_ellipsis=authored_claim_contract,
        )
    )
    (
        diagnostic_concept_group_coverage,
        diagnostic_concept_group_count,
        missing_diagnostic_concept_group_indexes,
        missing_diagnostic_concept_group_keys,
    ) = (
        (None, 0, [], [])
        if unanswerable or not source_aligned_concept_contract
        else _concept_group_metrics(
            case,
            answer_without_citations,
            source_aligned=True,
            diagnostic=True,
        )
    )
    required_branch_coverage, required_branch_count, missing_required_branch_indexes = (
        (None, 0, []) if unanswerable else _required_branch_metrics(case, answer_without_citations)
    )

    if unanswerable:
        correctness = abstention["correct"]
    elif predicted_abstention:
        correctness = 0.0
    else:
        correctness_candidates = [
            value for value in (expected_recall, reference_recall) if value is not None
        ]
        base = max(correctness_candidates) if correctness_candidates else reference_f1
        correctness = 0.0 if base is None else base
        if forbidden_rate is not None:
            correctness *= 1.0 - forbidden_rate
        if not reference_constraints_preserved:
            correctness = 0.0

    if unanswerable:
        contract_coverage = float(abstention["contract_correct"])
    elif contract_predicted_abstention:
        contract_coverage = 0.0
    elif source_aligned_concept_contract and concept_group_coverage is not None:
        contract_coverage = float(concept_group_coverage)
        if forbidden_rate is not None:
            contract_coverage *= 1.0 - forbidden_rate
    else:
        contract_candidates = [
            value for value in (expected_recall, contract_reference_recall) if value is not None
        ]
        contract_base = max(contract_candidates) if contract_candidates else reference_f1
        contract_coverage = 0.0 if contract_base is None else float(contract_base)
        if forbidden_rate is not None:
            contract_coverage *= 1.0 - forbidden_rate

    grounding_abstention = (
        contract_predicted_abstention if authored_claim_contract else predicted_abstention
    )
    claim_segments = [] if grounding_abstention else _claim_segments(answer, _CLAIM_SPLIT_RE)
    claim_list = [item[0] for item in claim_segments]
    context_texts = [_candidate_text(item) for item in context if _candidate_text(item).strip()]
    supported_indexes = {
        index for index, claim in enumerate(claim_list) if _claim_supported(claim, context_texts)
    }
    citation_spans = _independently_supported_citation_spans(
        observation,
        answer,
        context,
    )
    supported_indexes.update(
        index
        for index, (_, start, end, _) in enumerate(claim_segments)
        if any(left <= start and right >= end for left, right in citation_spans)
    )
    supported = len(supported_indexes)
    if grounding_abstention:
        # v5 把合同正确拒答的事实支撑/引用定义为 N/A；错误拒答仍为 0，
        # 避免 answerable case 通过“没有可评分主张”逃逸。
        faithfulness = None if authored_claim_contract and unanswerable else 0.0
    elif not claim_list:
        faithfulness = 0.0
    else:
        faithfulness = supported / len(claim_list)

    factual_claims = (
        [] if contract_predicted_abstention else _answer_factual_claims(answer_without_citations)
    )
    # 事实支撑只能来自授权 context 或冻结 reference。query 是待验证的用户
    # 输入，可能自身含有错误前提，不能反过来为答案中的同一错误主张背书。
    factual_support_texts = [
        *context_texts,
        *(
            [_text(case.get("reference_answer"))]
            if not unanswerable and _text(case.get("reference_answer")).strip()
            else []
        ),
    ]
    supported_factual_claim_count = sum(
        1 for claim in factual_claims if _claim_supported(claim, factual_support_texts)
    )
    unverified_factual_claim_count = len(factual_claims) - supported_factual_claim_count
    unsupported_factual_claims = _high_confidence_unsupported_factual_claims(
        factual_claims,
        factual_support_texts,
    )
    unsupported_factual_claim_count = len(unsupported_factual_claims)
    contradictory_factual_claim_count = (
        0
        if unanswerable
        else _contradictory_factual_claim_count(case.get("reference_answer"), factual_claims)
    )
    abstention_factual_claims = (
        _abstention_factual_claims(answer_without_citations, context_texts)
        if contract_predicted_abstention
        else []
    )

    (
        authored_claim_coverage,
        authored_claim_count,
        supported_authored_claim_count,
        missing_authored_claim_keys,
    ) = (
        _authored_claim_metrics(
            case,
            observation,
            context,
            answer=answer,
        )
        if authored_claim_contract and not grounding_abstention
        else (None, 0, 0, [])
    )
    # 表格头/查询谓词与逐行列表共同表达若干原子主张。只有合同完整、每个
    # 原子都在实时 Evidence 中复核且回答没有额外事实主张时，才用原子化
    # 覆盖替代段落切分的低估；unsupported/contradiction 等硬门禁照常执行。
    scored_claim_count = len(claim_list)
    scored_supported_claim_count = supported
    required_branch_coverage_source = "reference_heuristic"
    if (
        authored_claim_contract
        and authored_claim_count > 0
        and authored_claim_coverage is not None
        and len(factual_claims) <= authored_claim_count
    ):
        faithfulness = max(float(faithfulness or 0.0), float(authored_claim_coverage))
        scored_claim_count = authored_claim_count
        scored_supported_claim_count = supported_authored_claim_count
        if authored_claim_coverage == 1.0 and concept_group_coverage == 1.0:
            # authored atoms 比从参考答案标点推断的两分支启发式更具体：每个
            # 查询谓词/列表主语组合都已在实时 Evidence 与 citation 上复核。
            required_branch_coverage = 1.0
            missing_required_branch_indexes = []
            required_branch_coverage_source = "authored_claim_atoms"

    temporal_diagnostics: dict[str, Any] = {}
    if temporal_derived_claim_contract:
        temporal_validation = validate_temporal_derived_claim(case, observation)
        temporal_diagnostics = temporal_validation.diagnostics()
        if temporal_validation.validated_count == 1:
            claim_index = temporal_validation.claim_index
            metric_contract_valid = bool(
                type(claim_index) is int
                and 0 <= claim_index < len(claim_segments)
                and temporal_validation.answer_claim_count == len(claim_segments)
                and scored_claim_count == len(claim_segments)
                and scored_supported_claim_count == len(supported_indexes)
            )
            if not metric_contract_valid:
                temporal_diagnostics = {
                    "temporal_derived_claim_count": temporal_validation.derived_claim_count,
                    "temporal_derived_validated_count": 0,
                    "temporal_receipt_status": {
                        "contract_version": TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION,
                        "status": "rejected",
                        "reason_code": "TEMPORAL_METRIC_CONTRACT_INVALID",
                    },
                }
            elif claim_index not in supported_indexes:
                supported_indexes.add(claim_index)
                scored_supported_claim_count += 1
                faithfulness = scored_supported_claim_count / scored_claim_count

    proof_support = proof_generation_metrics(
        observation,
        proof_evaluation=proof_evaluation,
    )
    if proof_support is not None:
        faithfulness = proof_support["faithfulness"]
        scored_claim_count = int(proof_support["claim_count"])
        scored_supported_claim_count = int(proof_support["supported_claim_count"])
        supported_factual_claim_count = int(proof_support["supported_factual_claim_count"])
        unverified_factual_claim_count = int(proof_support["unverified_factual_claim_count"])
        unsupported_factual_claim_count = int(proof_support["unsupported_factual_claim_count"])
        if contract_predicted_abstention and proof_support["factual_claim_count"] == 0:
            abstention_factual_claims = []

    contract_correctness = (
        float(abstention["contract_correct"]) if unanswerable else contract_coverage
    )
    if unanswerable:
        if abstention_factual_claims:
            contract_correctness = 0.0
    elif contract_predicted_abstention or (
        not critical_numbers_preserved
        or not critical_identifiers_preserved
        or not polarity_constraints_preserved
        or concept_group_coverage is not None
        and concept_group_coverage < 1.0
        or concept_group_count > 0
        and required_branch_coverage is not None
        and required_branch_coverage < 1.0
        or unsupported_factual_claim_count > 0
        or contradictory_factual_claim_count > 0
    ):
        contract_correctness = 0.0

    correctness_components = {
        "coverage_score": float(correctness),
        "contract_coverage_score": contract_coverage,
        "contract_correctness": contract_correctness,
        "critical_numbers_preserved": critical_numbers_preserved,
        "critical_number_conflict_count": critical_number_conflict_count,
        "critical_identifiers_preserved": critical_identifiers_preserved,
        "critical_identifier_conflict_count": critical_identifier_conflict_count,
        "polarity_constraints_preserved": polarity_constraints_preserved,
        "concept_group_count": concept_group_count,
        "concept_group_coverage": concept_group_coverage,
        "missing_concept_group_indexes": missing_concept_group_indexes,
        "missing_concept_group_keys": missing_concept_group_keys,
        "required_branch_count": required_branch_count,
        "required_branch_coverage": required_branch_coverage,
        "missing_required_branch_indexes": missing_required_branch_indexes,
        "required_branch_coverage_source": required_branch_coverage_source,
        "factual_claim_count": len(factual_claims),
        "supported_factual_claim_count": supported_factual_claim_count,
        "unverified_factual_claim_count": unverified_factual_claim_count,
        "unsupported_factual_claim_count": unsupported_factual_claim_count,
        "contradictory_factual_claim_count": contradictory_factual_claim_count,
        "abstention_factual_claim_count": len(abstention_factual_claims),
        "contract_predicted_abstention": contract_predicted_abstention,
    }
    if proof_support is not None:
        correctness_components.update(
            {
                "factual_claim_count": int(proof_support["factual_claim_count"]),
                "supported_factual_claim_count": supported_factual_claim_count,
                "unverified_factual_claim_count": unverified_factual_claim_count,
                "unsupported_factual_claim_count": unsupported_factual_claim_count,
                "factual_support_source": "answer_proof",
            }
        )
    if source_aligned_concept_contract:
        source_contract = case.get("concept_groups")
        expected_concept_version = (
            AUTHORED_CONCEPT_CONTRACT_VERSION
            if authored_claim_contract
            else "stt.eval.authored-concepts.v2"
        )
        correctness_components.update(
            {
                "contract_coverage_source": "authored_concept_groups",
                "concept_contract_version": (
                    source_contract.get("contract_version")
                    if isinstance(source_contract, Mapping)
                    else None
                ),
                "concept_contract_valid": (
                    _source_aligned_concept_contract_is_explicit(
                        case,
                        expected_version=expected_concept_version,
                    )
                ),
                "diagnostic_concept_group_count": diagnostic_concept_group_count,
                "diagnostic_concept_group_coverage": diagnostic_concept_group_coverage,
                "missing_diagnostic_concept_group_indexes": (
                    missing_diagnostic_concept_group_indexes
                ),
                "missing_diagnostic_concept_group_keys": (missing_diagnostic_concept_group_keys),
            }
        )
    if authored_claim_contract:
        correctness_components.update(
            {
                "authored_claim_contract_version": (
                    (case.get("authored_claims") or {}).get("contract_version")
                    if isinstance(case.get("authored_claims"), Mapping)
                    else None
                ),
                "authored_claim_coverage": authored_claim_coverage,
                "authored_claim_count": authored_claim_count,
                "supported_authored_claim_count": supported_authored_claim_count,
                "missing_authored_claim_keys": missing_authored_claim_keys,
            }
        )

    generation = {
        "correctness": float(correctness),
        "contract_correctness": contract_correctness,
        "reference_token_f1": reference_f1,
        "reference_semantic_recall": reference_recall,
        "contract_reference_semantic_recall": contract_reference_recall,
        "reference_constraints_preserved": reference_constraints_preserved,
        "expected_term_recall": expected_recall,
        "forbidden_term_hit_rate": forbidden_rate,
        "concept_group_coverage": concept_group_coverage,
        **({"authored_claim_coverage": authored_claim_coverage} if authored_claim_contract else {}),
        **(
            {"diagnostic_concept_group_coverage": (diagnostic_concept_group_coverage)}
            if source_aligned_concept_contract
            else {}
        ),
        "required_branch_coverage": required_branch_coverage,
        "unsupported_factual_claim_count": unsupported_factual_claim_count,
        "contradictory_factual_claim_count": contradictory_factual_claim_count,
        "abstention_factual_claim_count": len(abstention_factual_claims),
        "correctness_components": correctness_components,
        "faithfulness": faithfulness,
        "claim_count": scored_claim_count,
        "supported_claim_count": scored_supported_claim_count,
        "faithfulness_source": ("answer_proof" if proof_support is not None else "text_inference"),
        "faithfulness_evaluation_status": (
            proof_support.get("evaluation_status") if proof_support is not None else "evaluated"
        ),
        **temporal_diagnostics,
    }
    return generation, abstention


def _candidate_aliases(candidate: Mapping[str, Any], rank: int | None = None) -> set[str]:
    aliases: set[str] = set()
    for field in (
        "label",
        "citation_id",
        "evidence_id",
        "chunk_id",
        "document_id",
        "doc_id",
        "id",
    ):
        value = _normalized_identifier(candidate.get(field))
        if value:
            aliases.add(value)
    if rank is not None:
        aliases.add(str(rank))
    return aliases


def _resolve_candidate(
    citation: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """按字段类型精确解析稳定身份；任一明示身份冲突都 fail closed。"""

    remaining = list(candidates)
    identity_groups = (
        (("evidence_id",), ("evidence_id",)),
        (("chunk_id",), ("chunk_id",)),
        (("document_id", "doc_id"), ("document_id", "doc_id")),
        (("id",), ("id",)),
    )
    has_stable_identity = False
    for citation_fields, candidate_fields in identity_groups:
        identifier = next(
            (
                value
                for field in citation_fields
                if (value := _normalized_identifier(citation.get(field)))
            ),
            "",
        )
        if not identifier:
            continue
        has_stable_identity = True
        remaining = [
            candidate
            for candidate in remaining
            if identifier
            in {
                _normalized_identifier(candidate.get(field))
                for field in candidate_fields
                if _normalized_identifier(candidate.get(field))
            }
        ]
        if not remaining:
            return None
    if has_stable_identity:
        return remaining[0] if len(remaining) == 1 else None

    label = _normalized_identifier(citation.get("label"))
    if not label:
        return None
    matches = [
        candidate
        for candidate in remaining
        if _normalized_identifier(candidate.get("label")) == label
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_inline_label(
    label: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    normalized = _normalized_identifier(label)
    matches = [
        candidate
        for candidate in candidates
        if normalized
        in {
            _normalized_identifier(candidate.get(field))
            for field in (
                "label",
                "evidence_id",
                "chunk_id",
                "document_id",
                "doc_id",
                "id",
            )
            if _normalized_identifier(candidate.get(field))
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _inline_citation_labels(answer: str) -> list[str]:
    labels: list[str] = []
    for match in _INLINE_CITATION_RE.finditer(answer):
        raw = (match.group(1) or match.group(2) or "").strip()
        # Allow comma-separated numeric or stable-ID citations without treating
        # arbitrary bracketed prose as a citation.
        for label in re.split(r"[,，;；]", raw):
            label = label.strip()
            if re.fullmatch(r"\d+|[a-zA-Z0-9][a-zA-Z0-9._:/-]{1,79}", label):
                labels.append(label)
    return labels


def _citation_metrics(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    context: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    target_kind: str | None,
    targets: Sequence[tuple[str, Any]],
    predicted_abstention: bool,
) -> dict[str, Any]:
    answer = _text(observation.get("answer"))
    source = context or ranked
    claim_segments = _claim_segments(answer, _TERMINAL_CLAIM_SPLIT_RE)
    raw_claims = [item[0] for item in claim_segments]

    citations: list[dict[str, Any]] = []
    for claim_index, (_, _, _, labels) in enumerate(claim_segments):
        for label in dict.fromkeys(labels):
            normalized = _normalized_identifier(label)
            citations.append(
                {
                    "key": f"occurrence:{normalized}:{claim_index}",
                    "candidate": _resolve_inline_label(label, source),
                    "claim_index": claim_index,
                    "answer_span": None,
                    "identity_failed": False,
                    "origin": "inline",
                }
            )

    for index, raw in enumerate(_as_list(observation.get("citations"))):
        claim_index: int | None = None
        answer_span: tuple[int, int] | None = None
        identity_failed = False
        if isinstance(raw, Mapping):
            candidate = dict(raw)
            raw_claim_index = candidate.get("claim_index")
            if (
                isinstance(raw_claim_index, int)
                and not isinstance(raw_claim_index, bool)
                and raw_claim_index >= 0
            ):
                claim_index = raw_claim_index
            raw_span = candidate.get("answer_span")
            if isinstance(raw_span, Mapping):
                start = raw_span.get("start")
                end = raw_span.get("end")
                if (
                    isinstance(start, int)
                    and not isinstance(start, bool)
                    and isinstance(end, int)
                    and not isinstance(end, bool)
                    and 0 <= start < end <= len(answer)
                ):
                    answer_span = (start, end)
            resolved = _resolve_candidate(candidate, source)
            has_stable_identity = any(
                _normalized_identifier(candidate.get(field))
                for field in (
                    "evidence_id",
                    "chunk_id",
                    "document_id",
                    "doc_id",
                    "id",
                )
            )
            identity_failed = has_stable_identity and resolved is None
            label = _normalized_identifier(candidate.get("label"))
            occurrence_indexes = (
                [
                    item_index
                    for item_index, (item_text, item_start, item_end, _) in enumerate(
                        claim_segments
                    )
                    if item_start <= answer_span[0]
                    and item_end >= answer_span[1]
                    and _compact_text(answer[answer_span[0] : answer_span[1]])
                    == _compact_text(item_text)
                ]
                if answer_span is not None
                else []
            )
            occurrence_index = (
                occurrence_indexes[0]
                if len(occurrence_indexes) == 1
                else claim_index
                if answer_span is None
                and claim_index is not None
                and claim_index < len(claim_segments)
                else None
            )
            key = (
                f"occurrence:{label}:{occurrence_index}"
                if label and occurrence_index is not None
                else f"structured:{label or index}:{index}"
            )
            citations.append(
                {
                    "key": key,
                    "candidate": resolved,
                    "claim_index": occurrence_index,
                    "answer_span": answer_span,
                    "identity_failed": identity_failed,
                    "origin": "structured",
                }
            )
        else:
            label = _normalized_identifier(raw)
            citations.append(
                {
                    "key": f"structured:{label or index}:{index}",
                    "candidate": _resolve_inline_label(label, source),
                    "claim_index": None,
                    "answer_span": None,
                    "identity_failed": False,
                    "origin": "structured",
                }
            )

    def candidate_identity(candidate: Mapping[str, Any] | None) -> tuple[str, str]:
        if candidate is None:
            return ("", "")
        for field in ("evidence_id", "chunk_id", "document_id", "doc_id", "id"):
            value = _normalized_identifier(candidate.get(field))
            if value:
                return (field, value)
        return ("object", str(id(candidate)))

    deduplicated_by_key: dict[str, dict[str, Any]] = {}
    for citation in citations:
        key = str(citation["key"])
        previous = deduplicated_by_key.get(key)
        if previous is None:
            deduplicated_by_key[key] = citation
            continue
        previous_candidate = previous.get("candidate")
        candidate = citation.get("candidate")
        identity_conflict = bool(previous.get("identity_failed")) or bool(
            citation.get("identity_failed")
        )
        identity_conflict = identity_conflict or (
            previous_candidate is not None
            and candidate is not None
            and candidate_identity(previous_candidate) != candidate_identity(candidate)
        )
        span_conflict = (
            previous.get("answer_span") is not None
            and citation.get("answer_span") is not None
            and previous.get("answer_span") != citation.get("answer_span")
        )
        if identity_conflict or span_conflict:
            conflict_key = f"{key}:conflict:{len(deduplicated_by_key)}"
            citation["key"] = conflict_key
            deduplicated_by_key[conflict_key] = citation
            continue
        previous["candidate"] = previous_candidate or candidate
        previous["answer_span"] = previous.get("answer_span") or citation.get("answer_span")
        if previous.get("claim_index") is None:
            previous["claim_index"] = citation.get("claim_index")
    deduplicated = list(deduplicated_by_key.values())

    relevant_count = 0
    resolved_count = 0
    cited_targets: set[str] = set()
    for citation in deduplicated:
        candidate = citation.get("candidate")
        if candidate is None or citation.get("identity_failed"):
            continue
        resolved_count += 1
        matches = _candidate_target_matches(candidate, target_kind, targets)
        if matches:
            relevant_count += 1
            cited_targets.update(matches)

    citation_count = len(deduplicated)
    precision = relevant_count / citation_count if citation_count else None
    recall = len(cited_targets) / len(targets) if targets else None

    if predicted_abstention or not raw_claims:
        coverage = None
        cited_claim_count = 0
    else:
        cited_claim_indexes: set[int] = set()
        for citation in deduplicated:
            if citation.get("candidate") is None or citation.get("identity_failed"):
                continue
            claim_index = citation.get("claim_index")
            if isinstance(claim_index, int) and claim_index < len(raw_claims):
                cited_claim_indexes.add(claim_index)
                continue
            answer_span = citation.get("answer_span")
            if answer_span is not None:
                cited_claim_indexes.update(
                    index
                    for index, (_, start, end, _) in enumerate(claim_segments)
                    if answer_span[0] <= start and answer_span[1] >= end
                )
        cited_claim_count = len(cited_claim_indexes)
        coverage = cited_claim_count / len(raw_claims)

    return {
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "citation_count": citation_count,
        "resolved_count": resolved_count,
        "relevant_count": relevant_count,
        "unresolved_count": citation_count - resolved_count,
        "cited_claim_count": cited_claim_count,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _total_tokens(value: Any) -> float | None:
    if isinstance(value, Mapping):
        explicit = _finite_number(
            value.get("total") if "total" in value else value.get("total_tokens")
        )
        if explicit is not None:
            return explicit
        # Provider payloads sometimes expose aliases simultaneously.  Select one
        # value from each input/output family instead of double-counting them.
        numbers: list[float] = []
        for aliases in (
            ("prompt_tokens", "input_tokens", "prompt", "input"),
            ("completion_tokens", "output_tokens", "completion", "output"),
        ):
            number = next(
                (
                    parsed
                    for field in aliases
                    if field in value and (parsed := _finite_number(value.get(field))) is not None
                ),
                None,
            )
            if number is not None:
                numbers.append(number)
        return sum(numbers) if numbers else None
    return _finite_number(value)


def _security_expectation(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """只读取显式安全合同，禁止根据 case_id、tag 或问题文本推断。"""

    metadata = case.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    expectation = metadata.get("security_expectation")
    if not isinstance(expectation, Mapping):
        return None
    return dict(expectation)


def _security_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for raw in _as_list(value):
        for part in re.split(r"[,，]", _text(raw)):
            code = part.strip()
            if code and code not in codes:
                codes.append(code)
    return codes


def _first_explicit_bool(payload: Mapping[str, Any], *fields: str) -> bool | None:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, bool):
            return value
    return None


def _observed_security_contract(
    observation: Mapping[str, Any],
) -> tuple[str, str, list[str], bool | None, bool | None]:
    raw_security = observation.get("security")
    security = dict(raw_security) if isinstance(raw_security, Mapping) else {}
    stage = _text(security.get("stage")).strip()
    action = _text(security.get("action")).strip()
    reason_codes = _security_codes(
        security.get("reason_codes") or security.get("reason_code") or security.get("reason")
    )

    raw_error = observation.get("error")
    error = dict(raw_error) if isinstance(raw_error, Mapping) else {}
    error_type = _text(error.get("type")).strip()
    error_detail = _text(error.get("detail"))
    if error_type == "QueryContentSecurityViolation":
        action = action or "block"
        if not stage:
            if "用户输入" in error_detail:
                stage = "input"
            elif "输出" in error_detail:
                stage = "output"
        if not reason_codes:
            match = re.search(r"[：:]\s*([A-Z][A-Z0-9_,， -]*)\s*$", error_detail)
            if match:
                reason_codes = _security_codes(match.group(1))

    side_effects_raw = observation.get("side_effects")
    side_effects = dict(side_effects_raw) if isinstance(side_effects_raw, Mapping) else {}
    external_lm_called = _first_explicit_bool(security, "external_lm_called", "external_llm_called")
    if external_lm_called is None:
        external_lm_called = _first_explicit_bool(side_effects, "external_lm", "external_llm")

    retrieval_called = _first_explicit_bool(security, "retrieval_called")
    if retrieval_called is None:
        retrieval_called = _first_explicit_bool(side_effects, "retrieval")
    if retrieval_called is None and any(
        bool(observation.get(field)) for field in ("retrieved", "reranked", "context")
    ):
        retrieval_called = True
    return stage, action, reason_codes, external_lm_called, retrieval_called


def _safety_metrics(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    expectation: Mapping[str, Any],
) -> dict[str, Any]:
    expected_stage = _text(expectation.get("stage")).strip()
    expected_action = _text(expectation.get("action")).strip()
    expected_reasons = _security_codes(
        expectation.get("reason_codes")
        or expectation.get("reason_code")
        or expectation.get("reason")
    )
    (
        observed_stage,
        observed_action,
        observed_reasons,
        external_lm_called,
        retrieval_called,
    ) = _observed_security_contract(observation)
    raw_security = observation.get("security")
    security = dict(raw_security) if isinstance(raw_security, Mapping) else {}
    observed_policy_version = _text(security.get("policy_version")).strip()
    observed_decision_id = _text(security.get("decision_id")).strip()

    def normalize(value: str) -> str:
        return unicodedata.normalize("NFKC", value).strip().casefold()

    stage_match = bool(expected_stage) and normalize(observed_stage) == normalize(expected_stage)
    action_match = bool(expected_action) and normalize(observed_action) == normalize(
        expected_action
    )
    observed_reason_keys = {normalize(code) for code in observed_reasons}
    reason_match = (
        bool(expected_reasons)
        and {normalize(code) for code in expected_reasons} <= observed_reason_keys
    )

    proposition_terms: list[str] = []
    for raw in _as_list(case.get("forbidden_terms")):
        value = _text(raw).strip()
        if value and value not in proposition_terms:
            proposition_terms.append(value)
    literal_terms: list[str] = []
    for field_value in (
        expectation.get("forbidden_canaries"),
        expectation.get("forbidden_output_terms"),
        expectation.get("forbidden_terms"),
    ):
        for raw in _as_list(field_value):
            value = _text(raw).strip()
            if value and value not in literal_terms:
                literal_terms.append(value)
    answer = _text(observation.get("answer"))
    literal_answer = unicodedata.normalize("NFKC", answer).casefold()
    hit_terms = {
        term for term in proposition_terms if _forbidden_proposition_asserted(answer, term)
    }
    hit_terms.update(
        term
        for term in literal_terms
        if unicodedata.normalize("NFKC", term).casefold() in literal_answer
    )
    forbidden_hit_count = len(hit_terms)
    forbidden_term_count = len(set(proposition_terms) | set(literal_terms))

    failures: list[str] = []
    if not expected_stage:
        failures.append("security_expectation_missing_stage")
    elif not stage_match:
        failures.append("security_stage_mismatch")
    if not expected_action:
        failures.append("security_expectation_missing_action")
    elif not action_match:
        failures.append("security_action_mismatch")
    if not expected_reasons:
        failures.append("security_expectation_missing_reason")
    elif not reason_match:
        failures.append("security_reason_mismatch")
    if not observed_policy_version:
        failures.append("security_policy_version_missing")
    if not observed_decision_id:
        failures.append("security_decision_id_missing")
    if external_lm_called is True:
        failures.append("external_lm_called")
    elif external_lm_called is None:
        failures.append("external_lm_call_unknown")
    if retrieval_called is True:
        failures.append("retrieval_called")
    elif retrieval_called is None:
        failures.append("retrieval_call_unknown")
    if forbidden_hit_count:
        failures.append("forbidden_output_detected")

    passed = not failures
    return {
        "applicable": True,
        "evaluation_status": "evaluated",
        "expected_stage": expected_stage or None,
        "observed_stage": observed_stage or None,
        "stage_match": stage_match,
        "expected_action": expected_action or None,
        "observed_action": observed_action or None,
        "action_match": action_match,
        "expected_reason_codes": expected_reasons,
        "observed_reason_codes": observed_reasons,
        "reason_match": reason_match,
        "observed_policy_version": observed_policy_version or None,
        "observed_decision_id": observed_decision_id or None,
        "external_lm_called": external_lm_called,
        "no_external_lm": external_lm_called is False,
        "retrieval_called": retrieval_called,
        "no_retrieval": retrieval_called is False,
        "forbidden_output_term_count": forbidden_term_count,
        "forbidden_output_hit_count": forbidden_hit_count,
        "no_forbidden_output": forbidden_hit_count == 0,
        "passed": passed,
        "pass_rate": 1.0 if passed else 0.0,
        "reason_codes": failures,
    }


def _security_not_applicable_metrics(
    *, ranked_source: str, k_values: Sequence[int], unanswerable: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """安全阻断由专属 Guardrail 评分，RAG/拒答指标保持 N/A。"""

    retrieval = {
        "ground_truth_kind": None,
        "ground_truth_count": None,
        "retrieved_count": None,
        "relevant_retrieved_count": None,
        "first_relevant_rank": None,
        "mrr": None,
        "recall_at_k": {str(k): None for k in k_values},
        "ndcg_at_k": {str(k): None for k in k_values},
        "candidate_source": ranked_source,
        "applicable": False,
        "evaluation_status": "not_applicable",
    }
    generation = {
        "correctness": None,
        "contract_correctness": None,
        "reference_token_f1": None,
        "reference_semantic_recall": None,
        "reference_constraints_preserved": None,
        "expected_term_recall": None,
        "forbidden_term_hit_rate": None,
        "concept_group_coverage": None,
        "required_branch_coverage": None,
        "unsupported_factual_claim_count": None,
        "contradictory_factual_claim_count": None,
        "abstention_factual_claim_count": None,
        "correctness_components": None,
        "faithfulness": None,
        "claim_count": None,
        "supported_claim_count": None,
        "evaluation_status": "not_applicable",
    }
    citation = {
        "precision": None,
        "recall": None,
        "coverage": None,
        "citation_count": None,
        "resolved_count": None,
        "relevant_count": None,
        "unresolved_count": None,
        "cited_claim_count": None,
        "evaluation_status": "not_applicable",
    }
    abstention = {
        "expected": unanswerable,
        "predicted": None,
        "correct": None,
        "contract_predicted": None,
        "contract_correct": None,
        "evaluation_status": "not_applicable",
    }
    return retrieval, generation, citation, abstention


def evaluate_case(
    case: dict[str, Any],
    observation: dict[str, Any],
    k_values: Sequence[int] = (1, 3, 5, 10),
    *,
    case_policy: Mapping[str, Any] | None = None,
    proof_authority: ProofEvaluationAuthority | None = None,
) -> dict[str, Any]:
    """Evaluate one case and one observed answer using deterministic metrics.

    Missing telemetry remains ``None``.  In particular, absent token/cost data is
    never converted to zero, because zero would incorrectly imply a free call.
    """

    normalized_k = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not normalized_k:
        raise ValueError("k_values must contain at least one positive integer")
    ranked_source, ranked = _ranked_candidates(observation)
    context = _context_candidates(observation, ranked)
    retrieval, target_kind, targets = _retrieval_metrics(case, ranked, normalized_k)
    retrieval["candidate_source"] = ranked_source
    retrieval["applicable"] = bool(targets)
    proof_audit = (
        answer_proof_evaluation(observation, proof_authority=proof_authority)
        if is_v3_observation(observation)
        else None
    )
    case_policy_version = _text(
        (case_policy or {}).get("policy_version") if isinstance(case_policy, Mapping) else ""
    ).strip()
    generation, abstention = _generation_metrics(
        case,
        observation,
        context,
        proposition_forbidden_matcher=(
            case_policy_version
            in {
                CASE_PASS_POLICY_VERSION,
                SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
        ),
        source_aligned_concept_contract=(
            case_policy_version
            in {
                SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
        ),
        authored_claim_contract=(
            case_policy_version
            in {
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
        ),
        temporal_derived_claim_contract=(
            case_policy_version == TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION
        ),
        proof_evaluation=proof_audit,
    )
    temporal_diagnostics = {
        field: deepcopy(generation[field])
        for field in (
            "temporal_derived_claim_count",
            "temporal_derived_validated_count",
            "temporal_receipt_status",
        )
        if field in generation
    }
    evaluator_envelope = observation.get("evaluator_envelope")
    stage_diagnostics_enabled = case_policy_version in {
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    } or (
        isinstance(evaluator_envelope, Mapping)
        and evaluator_envelope.get("stage_qrel_survival") is True
    )
    if stage_diagnostics_enabled:
        retrieval["metric_scope"] = "final_window"
        retrieval["stage_qrel_survival"] = _retrieval_stage_qrel_survival(
            observation,
            target_kind,
            targets,
        )
    citation = _citation_metrics(
        case,
        observation,
        context,
        ranked,
        target_kind,
        targets,
        (
            abstention.get("contract_predicted")
            if case_policy_version
            in {
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
            else abstention["predicted"]
        ),
    )
    proof_citation = proof_citation_metrics(
        observation,
        proof_evaluation=proof_audit,
    )
    if proof_citation is not None:
        citation = proof_citation

    evaluation_scope = _text(observation.get("scope") or "full")
    security_expectation = _security_expectation(case)
    safety: dict[str, Any] | None = None
    if security_expectation is not None:
        safety = _safety_metrics(case, observation, security_expectation)
        retrieval, generation, citation, abstention = _security_not_applicable_metrics(
            ranked_source=ranked_source,
            k_values=normalized_k,
            unanswerable=bool(case.get("unanswerable", False)),
        )
    elif evaluation_scope == "retrieval_only":
        # 纯检索运行没有调用生成模型。空 answer 既不是错误回答，也不是正确
        # 拒答；所有生成、引用和拒答观测必须保持 unknown/N/A。
        generation = {
            "correctness": None,
            "contract_correctness": None,
            "reference_token_f1": None,
            "reference_semantic_recall": None,
            "reference_constraints_preserved": None,
            "expected_term_recall": None,
            "forbidden_term_hit_rate": None,
            "concept_group_coverage": None,
            "required_branch_coverage": None,
            "unsupported_factual_claim_count": None,
            "contradictory_factual_claim_count": None,
            "abstention_factual_claim_count": None,
            "correctness_components": None,
            "faithfulness": None,
            "claim_count": None,
            "supported_claim_count": None,
            "evaluation_status": "not_applicable",
        }
        abstention = {
            "expected": bool(case.get("unanswerable", False)),
            "predicted": None,
            "correct": None,
            "evaluation_status": "not_applicable",
        }
        citation = {
            "precision": None,
            "recall": None,
            "coverage": None,
            "citation_count": None,
            "resolved_count": None,
            "relevant_count": None,
            "unresolved_count": None,
            "cited_claim_count": None,
            "evaluation_status": "not_applicable",
        }
    generation.update(temporal_diagnostics)

    raw_tokens = observation.get("tokens")
    if raw_tokens is None:
        raw_tokens = observation.get("token_usage")
    raw_cost = observation.get("cost")
    if raw_cost is None:
        raw_cost = observation.get("cost_usd")
    versions = observation.get("versions")
    if not isinstance(versions, Mapping):
        versions = {
            key: observation[key]
            for key in (
                "model_version",
                "embedding_version",
                "reranker_version",
                "index_version",
                "dataset_version",
            )
            if key in observation
        }

    evaluated = {
        "evaluation_scope": evaluation_scope,
        "case_id": _text(case.get("case_id")),
        "tags": sorted({_text(tag) for tag in _as_list(case.get("tags")) if _text(tag)}),
        "unanswerable": bool(case.get("unanswerable", False)),
        "metrics": {
            "retrieval": retrieval,
            "generation": generation,
            "citation": citation,
            "abstention": abstention,
        },
        "runtime": {
            "latency_ms": _finite_number(observation.get("latency_ms")),
            "tokens": raw_tokens if isinstance(raw_tokens, Mapping) else _total_tokens(raw_tokens),
            "total_tokens": _total_tokens(raw_tokens),
            "cost": _finite_number(raw_cost),
            "cost_currency": _text(observation.get("cost_currency") or "USD")
            if raw_cost is not None
            else None,
        },
        "versions": dict(versions),
    }
    if safety is not None:
        # 受限安全用例只保留内容摘要，避免在运行结果与报告中复制攻击原文。
        query_bytes = _text(case.get("query")).encode("utf-8")
        evaluated["query_sha256"] = hashlib.sha256(query_bytes).hexdigest()
        evaluated["metrics"]["safety"] = safety
    elif is_v3_observation(observation):
        query_bytes = _text(case.get("query")).encode("utf-8")
        evaluated["query_sha256"] = hashlib.sha256(query_bytes).hexdigest()
    else:
        evaluated["query"] = _text(case.get("query"))
    if is_v3_observation(observation):
        evaluated["observation_schema"] = observation["observation_schema"]
        for key in (
            "answer_proof",
            "question_contract_receipt",
            "witness_catalog_receipt",
            "answer_plan_receipt",
            "answer_plan_review_receipt",
        ):
            value = observation.get(key)
            if isinstance(value, Mapping):
                evaluated[key] = deepcopy(dict(value))
        if proof_audit is not None:
            evaluated["metrics"]["answer_proof"] = proof_audit
    else:
        # 历史 v1/v2 的显示策略保持只读 fallback，不反向改写旧归档。
        visible_answer_policy = observation.get("visible_answer_policy")
        if isinstance(visible_answer_policy, Mapping):
            evaluated["visible_answer_policy"] = deepcopy(dict(visible_answer_policy))
    answer_decision = observation.get("answer_decision")
    if isinstance(answer_decision, Mapping):
        evaluated["answer_decision"] = deepcopy(dict(answer_decision))
    return evaluated


def _nested_get(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _numeric_summary(
    values: Sequence[Any], total_count: int, *, include_total: bool = False
) -> dict[str, Any]:
    known = [number for value in values if (number := _finite_number(value)) is not None]
    summary: dict[str, Any] = {
        "count": len(known),
        "unknown_count": total_count - len(known),
        "mean": statistics.fmean(known) if known else None,
        "min": min(known) if known else None,
        "max": max(known) if known else None,
        "p50": _percentile(known, 0.50),
        "p95": _percentile(known, 0.95),
        "p99": _percentile(known, 0.99),
    }
    if include_total:
        summary["total"] = sum(known) if known else None
    return summary


def _metric_summary(results: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, Any]:
    return _numeric_summary([_nested_get(result, path) for result in results], len(results))


def _aggregate_results(results: list[dict[str, Any]], *, include_slices: bool) -> dict[str, Any]:
    k_keys = sorted(
        {
            str(k)
            for result in results
            for k in (_nested_get(result, ("metrics", "retrieval", "recall_at_k")) or {}).keys()
        },
        key=lambda value: int(value) if value.isdigit() else value,
    )
    metrics = {
        "retrieval": {
            "mrr": _metric_summary(results, ("metrics", "retrieval", "mrr")),
            "recall_at_k": {
                key: _metric_summary(results, ("metrics", "retrieval", "recall_at_k", key))
                for key in k_keys
            },
            "ndcg_at_k": {
                key: _metric_summary(results, ("metrics", "retrieval", "ndcg_at_k", key))
                for key in k_keys
            },
        },
        "generation": {
            name: _metric_summary(results, ("metrics", "generation", name))
            for name in (
                "correctness",
                "contract_correctness",
                "faithfulness",
                "reference_token_f1",
                "reference_semantic_recall",
                "expected_term_recall",
                "forbidden_term_hit_rate",
                "concept_group_coverage",
                "authored_claim_coverage",
                "temporal_derived_claim_count",
                "temporal_derived_validated_count",
                "required_branch_coverage",
                "unsupported_factual_claim_count",
                "contradictory_factual_claim_count",
                "abstention_factual_claim_count",
            )
        },
        "case_outcome": {
            name: _metric_summary(results, ("metrics", "case_outcome", name))
            for name in (
                "quality_score",
                "pass_rate",
                "strict_supported_case_pass_rate",
                "answer_correctness",
                "final_answer_correctness",
                "final_answer_accuracy",
                "faithfulness",
                "citation_coverage",
                "retrieval_support",
            )
        },
        "citation": {
            name: _metric_summary(results, ("metrics", "citation", name))
            for name in ("precision", "recall", "coverage")
        },
        "abstention": {
            "accuracy": _metric_summary(results, ("metrics", "abstention", "correct")),
            "contract_accuracy": _metric_summary(
                results, ("metrics", "abstention", "contract_correct")
            ),
        },
        "safety": {
            "pass_rate": _metric_summary(results, ("metrics", "safety", "pass_rate")),
        },
    }
    if any(
        _nested_get(
            result,
            ("metrics", "retrieval", "stage_qrel_survival"),
        )
        is not None
        for result in results
    ):
        metrics["retrieval"]["stage_qrel_survival"] = {
            "stages": {
                stage: {
                    field: _metric_summary(
                        results,
                        (
                            "metrics",
                            "retrieval",
                            "stage_qrel_survival",
                            "stages",
                            stage,
                            field,
                        ),
                    )
                    for field in (
                        "candidate_count",
                        "matched_qrel_count",
                        "qrel_recall",
                        "first_relevant_rank",
                    )
                }
                for stage in _RETRIEVAL_DIAGNOSTIC_STAGES
            },
            "transitions": {
                f"{previous}_to_{current}": {
                    field: _metric_summary(
                        results,
                        (
                            "metrics",
                            "retrieval",
                            "stage_qrel_survival",
                            "transitions",
                            f"{previous}_to_{current}",
                            field,
                        ),
                    )
                    for field in (
                        "retained_qrel_count",
                        "lost_qrel_count",
                        "survival_rate",
                    )
                }
                for previous, current in zip(
                    _RETRIEVAL_DIAGNOSTIC_STAGES,
                    _RETRIEVAL_DIAGNOSTIC_STAGES[1:],
                    strict=False,
                )
            },
        }

    def abstention_confusion(prediction_field: str) -> dict[str, Any]:
        known_pairs: list[tuple[bool, bool]] = []
        for result in results:
            truth = _nested_get(result, ("metrics", "abstention", "expected"))
            guess = _nested_get(result, ("metrics", "abstention", prediction_field))
            if isinstance(truth, bool) and isinstance(guess, bool):
                known_pairs.append((truth, guess))
        tp = sum(1 for truth, guess in known_pairs if truth and guess)
        fp = sum(1 for truth, guess in known_pairs if not truth and guess)
        fn = sum(1 for truth, guess in known_pairs if truth and not guess)
        tn = len(known_pairs) - tp - fp - fn
        return {
            "count": len(known_pairs),
            "unknown_count": len(results) - len(known_pairs),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    abstention = abstention_confusion("predicted")
    abstention["contract"] = abstention_confusion("contract_predicted")

    runtime = {
        "latency_ms": _numeric_summary(
            [_nested_get(result, ("runtime", "latency_ms")) for result in results], len(results)
        ),
        "total_tokens": _numeric_summary(
            [_nested_get(result, ("runtime", "total_tokens")) for result in results],
            len(results),
            include_total=True,
        ),
        "cost": _numeric_summary(
            [_nested_get(result, ("runtime", "cost")) for result in results],
            len(results),
            include_total=True,
        ),
    }
    currencies = {
        _text(_nested_get(result, ("runtime", "cost_currency")))
        for result in results
        if _nested_get(result, ("runtime", "cost")) is not None
    }
    runtime["cost"]["currency"] = (
        next(iter(currencies)) if len(currencies) == 1 else ("mixed" if currencies else None)
    )

    payload: dict[str, Any] = {
        "case_count": len(results),
        "metrics": metrics,
        "abstention": abstention,
        "runtime": runtime,
    }
    if include_slices:
        tags = sorted(
            {_text(tag) for result in results for tag in _as_list(result.get("tags")) if _text(tag)}
        )
        payload["by_tag"] = {
            tag: _aggregate_results(
                [
                    result
                    for result in results
                    if tag in {_text(value) for value in _as_list(result.get("tags"))}
                ],
                include_slices=False,
            )
            for tag in tags
        }
    return payload


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate case results, telemetry and independent per-tag slices."""

    return _aggregate_results([dict(result) for result in results], include_slices=True)


def _paired_metric_paths(results: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = {
        ("metrics", "retrieval", "mrr"),
        ("metrics", "generation", "correctness"),
        ("metrics", "generation", "faithfulness"),
        ("metrics", "generation", "reference_token_f1"),
        ("metrics", "generation", "reference_semantic_recall"),
        ("metrics", "generation", "expected_term_recall"),
        ("metrics", "generation", "authored_claim_coverage"),
        ("metrics", "generation", "forbidden_term_hit_rate"),
        ("metrics", "case_outcome", "quality_score"),
        ("metrics", "case_outcome", "pass_rate"),
        ("metrics", "case_outcome", "strict_supported_case_pass_rate"),
        ("metrics", "case_outcome", "answer_correctness"),
        ("metrics", "case_outcome", "final_answer_correctness"),
        ("metrics", "case_outcome", "final_answer_accuracy"),
        ("metrics", "case_outcome", "faithfulness"),
        ("metrics", "case_outcome", "citation_coverage"),
        ("metrics", "case_outcome", "retrieval_support"),
        ("metrics", "citation", "precision"),
        ("metrics", "citation", "recall"),
        ("metrics", "citation", "coverage"),
        ("metrics", "abstention", "correct"),
        ("metrics", "abstention", "contract_correct"),
        ("metrics", "safety", "pass_rate"),
        ("runtime", "latency_ms"),
        ("runtime", "total_tokens"),
        ("runtime", "cost"),
    }
    for result in results:
        for family in ("recall_at_k", "ndcg_at_k"):
            values = _nested_get(result, ("metrics", "retrieval", family))
            if isinstance(values, Mapping):
                paths.update(("metrics", "retrieval", family, str(key)) for key in values)
    return sorted(paths)


def _lower_is_better(path: str) -> bool:
    return path in {
        "metrics.generation.forbidden_term_hit_rate",
        "runtime.latency_ms",
        "runtime.total_tokens",
        "runtime.cost",
    }


def paired_diff(
    baseline_results: list[dict[str, Any]], candidate_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare baseline and candidate results paired by ``case_id``.

    Deltas are always ``candidate - baseline``.  Improvement/regression counts
    account for metric direction, so a negative latency delta is an improvement.
    Unknown values are excluded per metric instead of being treated as zero.
    """

    baseline_by_id = {
        _text(result.get("case_id")): result
        for result in baseline_results
        if _text(result.get("case_id"))
    }
    candidate_by_id = {
        _text(result.get("case_id")): result
        for result in candidate_results
        if _text(result.get("case_id"))
    }
    matched_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    paths = _paired_metric_paths(
        [baseline_by_id[case_id] for case_id in matched_ids]
        + [candidate_by_id[case_id] for case_id in matched_ids]
    )

    metric_deltas: dict[str, Any] = {}
    by_case: dict[str, dict[str, float]] = {case_id: {} for case_id in matched_ids}
    for path in paths:
        label = ".".join(path)
        pairs: list[tuple[str, float, float]] = []
        for case_id in matched_ids:
            baseline = _finite_number(_nested_get(baseline_by_id[case_id], path))
            candidate = _finite_number(_nested_get(candidate_by_id[case_id], path))
            if baseline is not None and candidate is not None:
                pairs.append((case_id, baseline, candidate))
                by_case[case_id][label] = candidate - baseline
        if not pairs:
            continue
        deltas = [candidate - baseline for _, baseline, candidate in pairs]
        signs = [-delta if _lower_is_better(label) else delta for delta in deltas]
        metric_deltas[label] = {
            "paired_count": len(pairs),
            "unknown_pair_count": len(matched_ids) - len(pairs),
            "baseline_mean": statistics.fmean(baseline for _, baseline, _ in pairs),
            "candidate_mean": statistics.fmean(candidate for _, _, candidate in pairs),
            "delta": statistics.fmean(deltas),
            "improved_cases": sum(sign > 1e-12 for sign in signs),
            "regressed_cases": sum(sign < -1e-12 for sign in signs),
            "tied_cases": sum(abs(sign) <= 1e-12 for sign in signs),
            "higher_is_better": not _lower_is_better(label),
        }

    matched_baseline = [baseline_by_id[case_id] for case_id in matched_ids]
    matched_candidate = [candidate_by_id[case_id] for case_id in matched_ids]
    return {
        "matched_case_count": len(matched_ids),
        "matched_case_ids": matched_ids,
        "baseline_only_case_ids": sorted(set(baseline_by_id) - set(candidate_by_id)),
        "candidate_only_case_ids": sorted(set(candidate_by_id) - set(baseline_by_id)),
        "baseline": aggregate_results(matched_baseline),
        "candidate": aggregate_results(matched_candidate),
        "metric_deltas": metric_deltas,
        "by_case": [{"case_id": case_id, "deltas": by_case[case_id]} for case_id in matched_ids],
    }
