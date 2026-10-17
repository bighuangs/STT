"""Build a versioned, reviewable suite-v2 concept-contract candidate.

The builder is deliberately side-effect free outside this directory.  It reads
the frozen v1 manifests, their source PDF, and the independent source/evaluator
QA artifact; it never edits or imports the source datasets.  No extraction rule
depends on a case id, a production answer, or a production pass/fail outcome.

The generated contracts are *candidates*, not gold labels.  Every derived field
retains provenance and review flags so heuristic parsing cannot silently become
an evaluation baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from pypdf import PdfReader


OUTPUT_ROOT = Path(__file__).resolve().parent
REMEDIATION_ROOT = OUTPUT_ROOT.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
PROJECT_ROOT = SUITE_ROOT.parents[2]
SOURCE_DATASET_ROOT = SUITE_ROOT / "datasets"
SOURCE_QA_ROOT = REMEDIATION_ROOT / "suite-source-evaluator-qa"
SOURCE_SUITE_PATH = SUITE_ROOT / "suite.json"
SOURCE_VISUAL_AUDIT_PATH = OUTPUT_ROOT / "source-visual-audit.json"

DIFFICULTIES = ("easy", "medium", "hard")
PARENT_SUITE_KEY = "three-difficulty-live-2026-08-10"
SUITE_KEY = "three-difficulty-live-2026-08-10-concept-v2-candidate"
DATASET_VERSION = "2.0.0-candidate.1"
SUITE_SCHEMA = "stt.eval.suite.v2-concept-contract-candidate"
DATASET_SCHEMA = "stt.eval.dataset.v2-concept-contract-candidate"
CONTRACT_SCHEMA = "stt.eval.answer-contract.v2-candidate"
BUILD_SCHEMA = "stt.eval.suite-v2-concept-build.v1"

_PRIMARY_CLAUSE_SPLIT_RE = re.compile(r"[；;。\n]+")
_COMMA_SPLIT_RE = re.compile(r"(?<!\d)[，,]|[，,](?!\d)")
_LIST_SPLIT_RE = re.compile(r"[、]")
_QUERY_BOUNDARY_RE = re.compile(r"[？?；;。\n]+")
_QUERY_CONNECTOR_RE = re.compile(
    r"(?:^|(?<=[，,]))(?=(?:同时|此外|另外|并请说明|并说明|并列出|还请说明))"
)
_MULTI_QUERY_RE = re.compile(
    r"(?:分别|各自|以及|同时|并且|并说明|并列出|比较|差异|全部|每个|两种|"
    r"两个|三种|步骤|哪些值|哪些项|哪些条件|为何|为什么)"
)
_TABLE_QUERY_RE = re.compile(
    r"(?:表格|(?<!列)表中|字段(?:值|内容)|参数表|规格表|根据.*规格|哪些值|列出.*值)"
)
_NEGATIVE_CUE_RE = re.compile(
    r"(?:请勿|不得|禁止|切勿|不要|不能|不适合|不支持|无需|未|没有|无法|无介质|"
    r"不超过|不可|并非|仅限|只允许)"
)
_POSITIVE_CUE_RE = re.compile(r"(?:可以|能够|支持|适合|需要|应当|应|必须|允许|用于|符合)")
_YES_NO_RE = re.compile(r"(?:是否|能否|可否|有没有|需不需要)")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_MIXED_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._/+-]*[A-Za-z])(?=[A-Za-z0-9._/+-]*\d)"
    r"[A-Za-z0-9]+(?:[._/+-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
_ACRONYM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,10}(?:[._/+\-][A-Za-z0-9]+)*|Wi-?Fi|dpi)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"^(?:\s*)(页|张|米|毫米|厘米|克|千克|分钟|秒|小时|天|个月|月|年|%|％|"
    r"dpi|ppm|MB|GB|V|W|Hz|mA|A|份|次|个|位|级|种|项|步|键|端口|字符)",
    re.IGNORECASE,
)
_ACTION_OR_REASON_RE = re.compile(
    r"(?:应|请勿|需要|用于|联系|以免|否则|可能|只能|仅|必须|如果|若|区别|相比|"
    r"导致|从.*拔下|调整|保存|选择|打开|关闭|安装|拆除|连接)"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

_FLAG_SEVERITY: dict[str, str] = {
    "SOURCE_PAGE_TEXT_TERM_GAP": "high",
    "TABLE_PAGE_VISUAL_REVIEW_REQUIRED": "high",
    "REFERENCE_NUMBER_NOT_EVIDENCE_BACKED": "high",
    "REFERENCE_IDENTIFIER_NOT_EVIDENCE_BACKED": "high",
    "CONCEPT_WITHOUT_EVIDENCE_SUPPORT": "high",
    "EVIDENCE_WITHOUT_CONCEPT_COVERAGE": "high",
    "QUERY_BRANCH_WITHOUT_CONCEPT": "high",
    "QUERY_REFERENCE_POLARITY_AMBIGUOUS": "high",
    "VARIANT_POLARITY_CONFLICT": "high",
    "UNANSWERABLE_BOUNDARY_REVIEW_REQUIRED": "high",
    "BRANCH_CONCEPT_ASSIGNMENT_ORDINAL": "medium",
    "REFERENCE_RESIDUAL_CONCEPT_ADDED": "medium",
    "EXPECTED_VARIANTS_MERGED": "medium",
    "SHORT_EXPECTED_VARIANT": "medium",
    "MULTI_QUERY_SINGLE_BRANCH_PARSE": "medium",
    "MIXED_REFERENCE_POLARITY": "medium",
    "CONCEPT_SUPPORTED_BY_MULTIPLE_EVIDENCE": "low",
    "SOURCE_VISUAL_INTERPRETATION_UNRESOLVED": "high",
    "ADJACENT_PAGE_ATTRIBUTION_REVIEW": "medium",
}


@dataclass(frozen=True)
class Inputs:
    suite: dict[str, Any]
    manifests: dict[str, dict[str, Any]]
    source_qa_summary: dict[str, Any]
    source_qa_cases: dict[str, dict[str, Any]]
    source_path: Path
    source_pages: tuple[str, ...]
    source_manifest_file_hashes: dict[str, str]
    visual_audit: dict[str, dict[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalized(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", _normalized(value))


def _ngrams(value: Any, size: int = 2) -> set[str]:
    compact = _compact(value)
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _contains(haystack: Any, needle: Any) -> bool:
    compact_needle = _compact(needle)
    return bool(compact_needle) and compact_needle in _compact(haystack)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = str(value or "").strip()
        key = _compact(stripped)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(stripped)
    return result


def _load_inputs() -> Inputs:
    suite = json.loads(SOURCE_SUITE_PATH.read_text(encoding="utf-8"))
    if suite.get("suite_key") != PARENT_SUITE_KEY:
        raise RuntimeError(f"unexpected parent suite: {suite.get('suite_key')!r}")

    manifests: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    for difficulty in DIFFICULTIES:
        path = SOURCE_DATASET_ROOT / f"{difficulty}.json"
        manifests[difficulty] = json.loads(path.read_text(encoding="utf-8"))
        manifest_hashes[difficulty] = _sha256_file(path)

    source_qa_summary = json.loads(
        (SOURCE_QA_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    source_qa_rows = json.loads(
        (SOURCE_QA_ROOT / "cases.json").read_text(encoding="utf-8")
    )
    source_qa_cases = {str(row["case_id"]): row for row in source_qa_rows}
    if len(source_qa_cases) != 150:
        raise RuntimeError("source QA must contain exactly 150 unique case rows")

    source_rel = str(manifests["easy"]["source"]["relative_path"])
    source_path = PROJECT_ROOT / source_rel
    reader = PdfReader(str(source_path))
    source_pages = tuple(page.extract_text() or "" for page in reader.pages)

    declared_hashes = {
        str(manifest["source"]["content_sha256"]) for manifest in manifests.values()
    }
    actual_source_hash = _sha256_file(source_path)
    if declared_hashes != {actual_source_hash}:
        raise RuntimeError(
            f"source PDF hash mismatch: declared={sorted(declared_hashes)} actual={actual_source_hash}"
        )
    declared_pages = {
        int(manifest["source"]["page_count"]) for manifest in manifests.values()
    }
    if declared_pages != {len(source_pages)}:
        raise RuntimeError(
            f"source PDF page mismatch: declared={sorted(declared_pages)} actual={len(source_pages)}"
        )

    visual_audit_document = json.loads(
        SOURCE_VISUAL_AUDIT_PATH.read_text(encoding="utf-8")
    )
    if str(visual_audit_document.get("source_pdf_sha256") or "") != actual_source_hash:
        raise RuntimeError("source visual audit is not locked to the active source PDF")
    visual_audit = {
        str(record["parent_case_id"]): dict(record)
        for record in visual_audit_document.get("records") or []
    }
    if len(visual_audit) != len(visual_audit_document.get("records") or []):
        raise RuntimeError("source visual audit parent_case_id values must be unique")
    manifest_case_ids = {
        str(case["case_id"])
        for manifest in manifests.values()
        for case in manifest.get("cases") or []
    }
    unknown_visual_cases = sorted(set(visual_audit) - manifest_case_ids)
    if unknown_visual_cases:
        raise RuntimeError(
            f"source visual audit refers to unknown parent cases: {unknown_visual_cases}"
        )

    return Inputs(
        suite=suite,
        manifests=manifests,
        source_qa_summary=source_qa_summary,
        source_qa_cases=source_qa_cases,
        source_path=source_path,
        source_pages=source_pages,
        source_manifest_file_hashes=manifest_hashes,
        visual_audit=visual_audit,
    )


def _group_expected_terms(terms: Sequence[str]) -> list[list[str]]:
    """Collapse spelling/substring variants into one any-of group."""

    if not terms:
        return []
    parent = list(range(len(terms)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    compact = [_compact(term) for term in terms]
    for left in range(len(terms)):
        for right in range(left + 1, len(terms)):
            a, b = compact[left], compact[right]
            if not a or not b:
                continue
            containment = (a in b or b in a) and min(len(a), len(b)) / max(len(a), len(b)) >= 0.55
            similar = SequenceMatcher(None, a, b).ratio() >= 0.86
            if containment or similar:
                union(left, right)

    grouped: dict[int, list[str]] = defaultdict(list)
    for index, term in enumerate(terms):
        grouped[find(index)].append(str(term))
    return [_dedupe(grouped[key]) for key in sorted(grouped)]


def _reference_fragments(reference: str) -> list[str]:
    fragments: list[str] = []
    for primary in _PRIMARY_CLAUSE_SPLIT_RE.split(reference):
        primary = primary.strip(" ：:，,、\t")
        if not primary:
            continue
        comma_parts = [part.strip(" ：:，,、\t") for part in _COMMA_SPLIT_RE.split(primary)]
        should_split_commas = len(comma_parts) >= 3 or len(_compact(primary)) >= 64
        candidates = comma_parts if should_split_commas else [primary]
        for candidate in candidates:
            list_parts = [part.strip(" ：:，,、\t") for part in _LIST_SPLIT_RE.split(candidate)]
            if len(list_parts) >= 3:
                fragments.extend(part for part in list_parts if len(_compact(part)) >= 2)
            elif len(_compact(candidate)) >= 2:
                fragments.append(candidate)
    return _dedupe(fragments)


def _query_branches(query: str) -> tuple[list[str], bool]:
    expanded = _QUERY_CONNECTOR_RE.sub("|||", query)
    pieces: list[str] = []
    for part in expanded.split("|||"):
        pieces.extend(_QUERY_BOUNDARY_RE.split(part))
    branches = _dedupe(piece.strip(" ：:，,\t") for piece in pieces)
    multi_signal = bool(_MULTI_QUERY_RE.search(query)) or query.count("？") + query.count("?") >= 2
    return branches or [query.strip()], multi_signal


def _variant_polarities(values: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.add("negative" if _NEGATIVE_CUE_RE.search(value) else "non_negative")
    return result


def _concept_support_score(concept: dict[str, Any], evidence: dict[str, Any]) -> float:
    evidence_text = str(evidence.get("text") or "")
    match_terms = evidence.get("match_terms") or []
    if isinstance(match_terms, str):
        match_terms = [match_terms]
    candidates = [
        str(concept.get("canonical_text") or ""),
        *[str(value) for value in concept.get("any_of") or []],
    ]
    if any(_contains(evidence_text, candidate) for candidate in candidates if _compact(candidate)):
        return 1.0
    if any(
        _contains(term, candidate) or _contains(candidate, term)
        for candidate in candidates
        for term in match_terms
        if _compact(candidate) and _compact(term)
    ):
        return 1.0
    concept_text = max(candidates, key=lambda value: len(_compact(value)), default="")
    concept_grams = _ngrams(concept_text)
    if not concept_grams:
        return 0.0
    evidence_grams = _ngrams(evidence_text)
    return len(concept_grams & evidence_grams) / len(concept_grams)


def _branch_similarity(branch: str, concept: dict[str, Any]) -> float:
    branch_grams = _ngrams(branch)
    if not branch_grams:
        return 0.0
    candidates = [
        str(concept.get("canonical_text") or ""),
        *[str(value) for value in concept.get("any_of") or []],
    ]
    best = 0.0
    for candidate in candidates:
        candidate_grams = _ngrams(candidate)
        if not candidate_grams:
            continue
        overlap = len(branch_grams & candidate_grams)
        score = overlap / min(len(branch_grams), len(candidate_grams))
        if _contains(branch, candidate) or _contains(candidate, branch):
            score = max(score, 0.9)
        best = max(best, score)
    return best


def _identifier_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in (_MIXED_IDENTIFIER_RE, _ACRONYM_RE):
        for match in pattern.finditer(text):
            raw = match.group(0)
            if len(raw) < 2:
                continue
            spans.append((match.start(), match.end(), raw))
    deduped: dict[str, tuple[int, int, str]] = {}
    for span in sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]))):
        deduped.setdefault(_compact(span[2]), span)
    return list(deduped.values())


def _critical_values(
    *,
    query: str,
    reference: str,
    expected_terms: Sequence[str],
    evidence: Sequence[dict[str, Any]],
    concepts: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    evidence_texts = {str(item.get("evidence_id") or ""): str(item.get("text") or "") for item in evidence}
    identifier_spans = _identifier_spans(reference)
    identifiers: list[dict[str, Any]] = []
    for start, end, raw in identifier_spans:
        evidence_ids = [eid for eid, text in evidence_texts.items() if _contains(text, raw)]
        concept_ids = [
            str(concept["concept_id"])
            for concept in concepts
            if _contains(concept.get("canonical_text"), raw)
            or any(_contains(value, raw) for value in concept.get("any_of") or [])
        ]
        identifiers.append(
            {
                "value": raw,
                "normalized": _compact(raw),
                "reference_context": reference[max(0, start - 14) : min(len(reference), end + 14)],
                "appears_in_query": _contains(query, raw),
                "appears_in_expected_terms": any(_contains(term, raw) for term in expected_terms),
                "evidence_ids": evidence_ids,
                "concept_ids": concept_ids,
                "required_exact": True,
            }
        )

    identifier_ranges = [(start, end) for start, end, _ in identifier_spans]
    numerics: list[dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(reference):
        if any(start <= match.start() and match.end() <= end for start, end in identifier_ranges):
            continue
        raw = match.group(0)
        unit_match = _UNIT_RE.match(reference[match.end() : match.end() + 12])
        unit = unit_match.group(1) if unit_match else None
        evidence_ids = [eid for eid, text in evidence_texts.items() if _contains(text, raw)]
        concept_ids = [
            str(concept["concept_id"])
            for concept in concepts
            if _contains(concept.get("canonical_text"), raw)
            or any(_contains(value, raw) for value in concept.get("any_of") or [])
        ]
        numerics.append(
            {
                "value": raw,
                "normalized": raw.replace(",", ""),
                "unit": unit,
                "reference_context": reference[max(0, match.start() - 14) : min(len(reference), match.end() + 14)],
                "appears_in_query": _contains(query, raw),
                "appears_in_expected_terms": any(_contains(term, raw) for term in expected_terms),
                "evidence_ids": evidence_ids,
                "concept_ids": concept_ids,
                "required_exact": True,
            }
        )

    flags: list[str] = []
    if any(not item["evidence_ids"] for item in numerics):
        flags.append("REFERENCE_NUMBER_NOT_EVIDENCE_BACKED")
    if any(not item["evidence_ids"] for item in identifiers):
        flags.append("REFERENCE_IDENTIFIER_NOT_EVIDENCE_BACKED")
    return numerics, identifiers, flags


def _polarity_contract(
    *, query: str, reference: str, evidence_text: str, unanswerable: bool
) -> tuple[dict[str, Any], list[str]]:
    if unanswerable:
        return (
            {
                "required": "abstain_without_factual_claims",
                "query_negative_cues": _dedupe(_NEGATIVE_CUE_RE.findall(query)),
                "reference_negative_cues": [],
                "evidence_negative_cues": [],
            },
            ["UNANSWERABLE_BOUNDARY_REVIEW_REQUIRED"],
        )
    query_negative = _dedupe(_NEGATIVE_CUE_RE.findall(query))
    reference_negative = _dedupe(_NEGATIVE_CUE_RE.findall(reference))
    evidence_negative = _dedupe(_NEGATIVE_CUE_RE.findall(evidence_text))
    reference_positive = _dedupe(_POSITIVE_CUE_RE.findall(reference))
    if reference_negative and reference_positive:
        required = "mixed_preserve_each_claim"
    elif reference_negative:
        required = "negative"
    elif _YES_NO_RE.search(query):
        required = "affirmative_or_explicit_yes_no"
    else:
        required = "not_applicable"
    flags: list[str] = []
    if query_negative and not reference_negative:
        flags.append("QUERY_REFERENCE_POLARITY_AMBIGUOUS")
    if reference_negative and reference_positive:
        flags.append("MIXED_REFERENCE_POLARITY")
    return (
        {
            "required": required,
            "query_negative_cues": query_negative,
            "reference_negative_cues": reference_negative,
            "reference_positive_cues": reference_positive,
            "evidence_negative_cues": evidence_negative,
        },
        flags,
    )


def _table_data_row_count(evidence_text: str) -> int:
    rows = [line.strip() for line in evidence_text.splitlines() if _TABLE_ROW_RE.match(line)]
    substantive: list[str] = []
    for index, row in enumerate(rows):
        if index == 0 or re.fullmatch(r"[| :\-]+", row):
            continue
        first_cell = row.strip("|").split("|", maxsplit=1)[0].strip()
        if re.match(r"^(?:备注|注[:：]?|说明[:：]?)", first_cell):
            continue
        substantive.append(row)
    return len(substantive)


def _source_page_flags(
    evidence: Sequence[dict[str, Any]], source_pages: Sequence[str]
) -> tuple[list[str], int, int]:
    matched = 0
    total = 0
    for item in evidence:
        page_number = int(item.get("page") or 0)
        page_text = source_pages[page_number - 1] if 1 <= page_number <= len(source_pages) else ""
        match_terms = item.get("match_terms") or []
        if isinstance(match_terms, str):
            match_terms = [match_terms]
        for term in match_terms:
            total += 1
            if _contains(page_text, term):
                matched += 1
    return (["SOURCE_PAGE_TEXT_TERM_GAP"] if total and matched < total else []), matched, total


def _make_concepts(
    *, expected_terms: Sequence[str], reference: str, evidence: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    term_groups = _group_expected_terms(expected_terms)
    fragments = _reference_fragments(reference)
    flags: list[str] = []
    if len(term_groups) < len(expected_terms):
        flags.append("EXPECTED_VARIANTS_MERGED")
    if any(len(_compact(term)) < 4 for term in expected_terms):
        flags.append("SHORT_EXPECTED_VARIANT")

    concepts: list[dict[str, Any]] = []
    fragment_group_hits: dict[int, list[int]] = defaultdict(list)
    for group_index, group in enumerate(term_groups):
        matching_fragments = [
            index
            for index, fragment in enumerate(fragments)
            if any(_contains(fragment, variant) or _contains(variant, fragment) for variant in group)
        ]
        for index in matching_fragments:
            fragment_group_hits[index].append(group_index)
        canonical = (
            min(
                (fragments[index] for index in matching_fragments),
                key=lambda value: len(_compact(value)),
            )
            if matching_fragments
            else max(group, key=lambda value: len(_compact(value)))
        )
        concept = {
            "concept_id": f"c{len(concepts) + 1:02d}",
            "canonical_text": canonical,
            "any_of": list(group),
            "derivation": {
                "method": "expected_term_variant_group",
                "expected_term_indexes": [expected_terms.index(term) for term in group],
                "reference_fragment_indexes": matching_fragments,
            },
            "evidence_support": [],
        }
        if len(_variant_polarities(group)) > 1:
            flags.append("VARIANT_POLARITY_CONFLICT")
        concepts.append(concept)

    residual_count = 0
    for fragment_index, fragment in enumerate(fragments):
        hit_groups = fragment_group_hits.get(fragment_index, [])
        add_residual = not hit_groups or (
            bool(_ACTION_OR_REASON_RE.search(fragment))
            and len(_compact(fragment)) >= 12
            and not any(_compact(fragment) == _compact(concepts[index]["canonical_text"]) for index in hit_groups)
        )
        if not add_residual:
            continue
        if any(_compact(fragment) == _compact(concept["canonical_text"]) for concept in concepts):
            continue
        concepts.append(
            {
                "concept_id": f"c{len(concepts) + 1:02d}",
                "canonical_text": fragment,
                "any_of": [fragment],
                "derivation": {
                    "method": "reference_residual_fragment",
                    "reference_fragment_indexes": [fragment_index],
                    "overlapping_expected_concept_ids": [
                        concepts[index]["concept_id"] for index in hit_groups
                    ],
                },
                "evidence_support": [],
            }
        )
        residual_count += 1
    if residual_count:
        flags.append("REFERENCE_RESIDUAL_CONCEPT_ADDED")

    for concept in concepts:
        support: list[dict[str, Any]] = []
        for item in evidence:
            score = _concept_support_score(concept, item)
            if score >= 0.45:
                support.append(
                    {
                        "evidence_id": str(item.get("evidence_id") or ""),
                        "score": round(score, 6),
                        "method": "exact_or_bigram_recall",
                    }
                )
        concept["evidence_support"] = support
        if not support:
            flags.append("CONCEPT_WITHOUT_EVIDENCE_SUPPORT")
        elif len(support) > 1:
            flags.append("CONCEPT_SUPPORTED_BY_MULTIPLE_EVIDENCE")
    return concepts, flags


def _make_branches(
    query: str, concepts: Sequence[dict[str, Any]], *, unanswerable: bool
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    branch_texts, multi_signal = _query_branches(query)
    flags: list[str] = []
    if multi_signal and len(branch_texts) == 1:
        flags.append("MULTI_QUERY_SINGLE_BRANCH_PARSE")
    branches = [
        {
            "branch_id": f"b{index + 1:02d}",
            "query_span": text,
            "required_concept_ids": [],
            "assignment_method": "not_applicable" if unanswerable else "lexical_similarity",
        }
        for index, text in enumerate(branch_texts)
    ]
    if unanswerable:
        for branch in branches:
            branch["answerability"] = "must_abstain"
        return branches, flags, []
    if len(branches) == 1:
        branches[0]["required_concept_ids"] = [concept["concept_id"] for concept in concepts]
        return branches, flags, []

    unassigned: list[str] = []
    ordinal_fallback = False
    for concept_index, concept in enumerate(concepts):
        scores = [_branch_similarity(branch["query_span"], concept) for branch in branches]
        best_score = max(scores, default=0.0)
        if best_score >= 0.12:
            branch_index = scores.index(best_score)
        else:
            branch_index = min(
                len(branches) - 1,
                (concept_index * len(branches)) // max(1, len(concepts)),
            )
            ordinal_fallback = True
        branches[branch_index]["required_concept_ids"].append(concept["concept_id"])
    if ordinal_fallback:
        flags.append("BRANCH_CONCEPT_ASSIGNMENT_ORDINAL")
    if any(not branch["required_concept_ids"] for branch in branches):
        flags.append("QUERY_BRANCH_WITHOUT_CONCEPT")
    assigned = {
        concept_id
        for branch in branches
        for concept_id in branch["required_concept_ids"]
    }
    unassigned = [concept["concept_id"] for concept in concepts if concept["concept_id"] not in assigned]
    return branches, flags, unassigned


def _flag_objects(codes: Iterable[str], details: dict[str, str] | None = None) -> list[dict[str, str]]:
    details = details or {}
    return [
        {
            "code": code,
            "severity": _FLAG_SEVERITY.get(code, "medium"),
            "detail": details.get(code, ""),
        }
        for code in sorted(set(codes))
    ]


def _v2_case_id(parent_case_id: str) -> str:
    match = re.fullmatch(r"(.+)-(\d{3})", parent_case_id)
    if not match:
        return f"{parent_case_id}-V2"
    return f"{match.group(1)}-V2-{match.group(2)}"


def _build_case(
    *, difficulty: str, source_case: dict[str, Any], inputs: Inputs
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_case_id = str(source_case["case_id"])
    query = str(source_case.get("query") or "")
    reference = str(source_case.get("reference_answer") or "")
    unanswerable = bool(source_case.get("unanswerable"))
    expected_terms = [str(value) for value in source_case.get("expected_terms") or []]
    evidence = [dict(item) for item in source_case.get("relevant_evidence") or []]
    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)

    concepts: list[dict[str, Any]] = []
    concept_flags: list[str] = []
    if not unanswerable:
        concepts, concept_flags = _make_concepts(
            expected_terms=expected_terms,
            reference=reference,
            evidence=evidence,
        )
    branches, branch_flags, unassigned_concepts = _make_branches(
        query, concepts, unanswerable=unanswerable
    )
    numerics, identifiers, value_flags = _critical_values(
        query=query,
        reference=reference,
        expected_terms=expected_terms,
        evidence=evidence,
        concepts=concepts,
    )
    polarity, polarity_flags = _polarity_contract(
        query=query,
        reference=reference,
        evidence_text=evidence_text,
        unanswerable=unanswerable,
    )
    source_flags, page_term_hits, page_term_total = _source_page_flags(
        evidence, inputs.source_pages
    )
    if (
        not unanswerable
        and (_TABLE_QUERY_RE.search(query) or (source_case.get("metadata") or {}).get("type") == "table")
        and _table_data_row_count(evidence_text) == 0
    ):
        source_flags.append("TABLE_PAGE_VISUAL_REVIEW_REQUIRED")

    evidence_coverage = []
    evidence_without_concepts: list[str] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        supporting = [
            concept["concept_id"]
            for concept in concepts
            if any(
                support["evidence_id"] == evidence_id
                for support in concept.get("evidence_support") or []
            )
        ]
        evidence_coverage.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": str(item.get("chunk_id") or ""),
                "page": item.get("page"),
                "required_concept_ids": supporting,
                "coverage_status": "covered" if supporting else "uncovered",
            }
        )
        if not supporting and not unanswerable:
            evidence_without_concepts.append(evidence_id)

    flags = [
        *concept_flags,
        *branch_flags,
        *value_flags,
        *polarity_flags,
        *source_flags,
    ]
    if evidence_without_concepts:
        flags.append("EVIDENCE_WITHOUT_CONCEPT_COVERAGE")
    if not unanswerable and not concepts:
        hard_blockers = ["ANSWERABLE_WITHOUT_REQUIRED_CONCEPT"]
    else:
        hard_blockers = []
    if not unanswerable and not evidence:
        hard_blockers.append("ANSWERABLE_WITHOUT_EVIDENCE")

    flag_objects = _flag_objects(flags)
    if hard_blockers:
        readiness = "blocked"
    elif flag_objects:
        readiness = "manual_or_llm_review_required"
    else:
        readiness = "deterministic_candidate_ready"

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "contract_status": "candidate_pending_full_review",
        "answerability": {
            "required": "abstain" if unanswerable else "answer",
            "unsupported_factual_additions_forbidden": True,
            "contradictory_claims_forbidden": True,
        },
        "required_concepts": {"all_of": concepts},
        "critical_constraints": {
            "numerics": numerics,
            "identifiers": identifiers,
            "polarity": polarity,
        },
        "required_branches": branches,
        "unassigned_concept_ids": unassigned_concepts,
        "evidence_concept_coverage": evidence_coverage,
        "review": {
            "status": "pending",
            "readiness": readiness,
            "flags": flag_objects,
            "hard_blockers": hard_blockers,
            "review_policy": "same_prompt_full_150_case_review_before_import",
        },
    }

    v2_case = deepcopy(source_case)
    v2_case["case_id"] = _v2_case_id(parent_case_id)
    v2_case["answer_contract"] = contract
    v2_case["tags"] = _dedupe(
        [*(source_case.get("tags") or []), "suite-v2", "concept-contract-candidate"]
    )
    v2_case["metadata"] = {
        **dict(source_case.get("metadata") or {}),
        "suite_key": SUITE_KEY,
        "parent_suite_key": PARENT_SUITE_KEY,
        "parent_case_id": parent_case_id,
        "contract_version": DATASET_VERSION,
        "contract_candidate_hash": _sha256_json(contract),
    }

    source_qa = inputs.source_qa_cases[parent_case_id]
    legacy_source_flags = sorted(
        code
        for code in source_qa.get("issue_codes") or []
        if code
        in {
            "PDF_TEXT_EXTRACTION_TERM_GAP",
            "TABLE_QUERY_WITHOUT_PARSED_DATA_ROW",
            "EXPECTED_TERM_VARIANTS_OVERLAP",
            "EXPECTED_TERM_TOO_SHORT",
            "QUERY_REFERENCE_POLARITY_CANDIDATE",
            "QUERY_SINGULAR_REFERENCE_ENUMERATION",
            "HARD_EVIDENCE_WITHOUT_SCORING_CONCEPT",
        }
    )
    qa_row = {
        "case_id": v2_case["case_id"],
        "parent_case_id": parent_case_id,
        "difficulty": difficulty,
        "unanswerable": unanswerable,
        "case_type": str((source_case.get("metadata") or {}).get("type") or ""),
        "concept_count": len(concepts),
        "expected_group_concept_count": sum(
            concept["derivation"]["method"] == "expected_term_variant_group"
            for concept in concepts
        ),
        "reference_residual_concept_count": sum(
            concept["derivation"]["method"] == "reference_residual_fragment"
            for concept in concepts
        ),
        "concept_without_evidence_count": sum(
            not concept.get("evidence_support") for concept in concepts
        ),
        "branch_count": len(branches),
        "branch_without_concept_count": sum(
            not branch.get("required_concept_ids") and not unanswerable for branch in branches
        ),
        "unassigned_concept_count": len(unassigned_concepts),
        "evidence_count": len(evidence),
        "evidence_without_concept_count": len(evidence_without_concepts),
        "critical_numeric_count": len(numerics),
        "critical_numeric_without_evidence_count": sum(
            not item["evidence_ids"] for item in numerics
        ),
        "critical_identifier_count": len(identifiers),
        "critical_identifier_without_evidence_count": sum(
            not item["evidence_ids"] for item in identifiers
        ),
        "polarity_required": polarity["required"],
        "source_page_match_term_hits": page_term_hits,
        "source_page_match_term_total": page_term_total,
        "review_readiness": readiness,
        "review_flag_count": len(flag_objects),
        "review_flags": [item["code"] for item in flag_objects],
        "hard_blocker_count": len(hard_blockers),
        "hard_blockers": hard_blockers,
        "legacy_source_qa_flags": legacy_source_flags,
        "contract_sha256": _sha256_json(contract),
    }

    review_payload = {
        "query": query,
        "reference_answer": reference,
        "unanswerable": unanswerable,
        "legacy_expected_terms": expected_terms,
        "forbidden_terms": [str(value) for value in source_case.get("forbidden_terms") or []],
        "branches": [
            {"branch_label": f"B{index + 1}", "query_span": branch["query_span"]}
            for index, branch in enumerate(branches)
        ],
        "evidence": [
            {
                "evidence_label": f"E{index + 1}",
                "page": item.get("page"),
                "text": str(item.get("text") or ""),
            }
            for index, item in enumerate(evidence)
        ],
        "deterministic_candidate": {
            "answerability": contract["answerability"],
            "required_concepts": contract["required_concepts"],
            "critical_constraints": contract["critical_constraints"],
            "required_branches": contract["required_branches"],
            "evidence_concept_coverage": contract["evidence_concept_coverage"],
            "review_flags": [item["code"] for item in flag_objects],
        },
    }

    # Apply independent visual findings only after the blind semantic payload is
    # frozen.  The text-only reviewer must not be asked to overrule page-level
    # evidence it cannot see, and the request pack therefore remains identical.
    visual_record = inputs.visual_audit.get(parent_case_id)
    qa_row.update(
        {
            "visual_audit_status": None,
            "visual_audit_severity": None,
            "visual_audit_declared_pages": [],
            "visual_audit_pdf_pages": [],
            "visual_audit_finding": None,
        }
    )
    if visual_record is not None:
        visual_status = str(visual_record.get("status") or "")
        contract["source_visual_audit"] = deepcopy(visual_record)
        qa_row.update(
            {
                "visual_audit_status": visual_status,
                "visual_audit_severity": visual_record.get("severity"),
                "visual_audit_declared_pages": visual_record.get("declared_evidence_pages") or [],
                "visual_audit_pdf_pages": visual_record.get("visual_pdf_pages") or [],
                "visual_audit_finding": visual_record.get("finding"),
            }
        )
        if visual_status in {
            "confirmed_reference_source_conflict",
            "confirmed_reference_incomplete",
        }:
            blockers = contract["review"]["hard_blockers"]
            blockers.append("CONFIRMED_VISUAL_SOURCE_CONTRACT_CONFLICT")
            contract["review"]["hard_blockers"] = sorted(set(blockers))
            contract["review"]["readiness"] = "blocked"
            contract["contract_status"] = "candidate_blocked_by_source_visual_audit"
        elif visual_status == "manual_visual_review_required":
            existing_flags = {
                item["code"]: item for item in contract["review"]["flags"]
            }
            existing_flags["SOURCE_VISUAL_INTERPRETATION_UNRESOLVED"] = _flag_objects(
                ["SOURCE_VISUAL_INTERPRETATION_UNRESOLVED"]
            )[0]
            contract["review"]["flags"] = [
                existing_flags[key] for key in sorted(existing_flags)
            ]
            contract["review"]["readiness"] = "manual_or_llm_review_required"
        elif visual_status == "adjacent_page_attribution":
            existing_flags = {
                item["code"]: item
                for item in contract["review"]["flags"]
                if item["code"] != "SOURCE_PAGE_TEXT_TERM_GAP"
            }
            existing_flags["ADJACENT_PAGE_ATTRIBUTION_REVIEW"] = _flag_objects(
                ["ADJACENT_PAGE_ATTRIBUTION_REVIEW"]
            )[0]
            contract["review"]["flags"] = [
                existing_flags[key] for key in sorted(existing_flags)
            ]
            contract["review"]["readiness"] = "manual_or_llm_review_required"

        qa_row["review_readiness"] = contract["review"]["readiness"]
        qa_row["review_flags"] = [
            item["code"] for item in contract["review"]["flags"]
        ]
        qa_row["review_flag_count"] = len(qa_row["review_flags"])
        qa_row["hard_blockers"] = contract["review"]["hard_blockers"]
        qa_row["hard_blocker_count"] = len(qa_row["hard_blockers"])
        qa_row["contract_sha256"] = _sha256_json(contract)
        v2_case["metadata"]["contract_candidate_hash"] = qa_row["contract_sha256"]
    return v2_case, qa_row, review_payload


def _build_manifests(inputs: Inputs) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []
    review_payloads: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        source_manifest = inputs.manifests[difficulty]
        cases: list[dict[str, Any]] = []
        for source_case in source_manifest["cases"]:
            v2_case, qa_row, review_payload = _build_case(
                difficulty=difficulty,
                source_case=source_case,
                inputs=inputs,
            )
            cases.append(v2_case)
            qa_rows.append(qa_row)
            review_payloads.append(review_payload)
        manifest = deepcopy(source_manifest)
        manifest["dataset_key"] = f"{source_manifest['dataset_key']}-concept-v2-candidate"
        manifest["name"] = f"{source_manifest['name']}（suite v2 概念契约候选）"
        manifest["description"] = (
            f"{source_manifest.get('description', '')} 旁路生成的 concept-contract v2 候选；"
            "未经全量同 prompt 审阅，不得导入、运行或登记为基线。"
        ).strip()
        manifest["version"] = DATASET_VERSION
        manifest["schema_version"] = DATASET_SCHEMA
        manifest["suite_key"] = SUITE_KEY
        manifest["tags"] = _dedupe(
            [*(source_manifest.get("tags") or []), "suite-v2", "concept-contract-candidate"]
        )
        manifest["source"] = {
            **dict(source_manifest.get("source") or {}),
            "parent_suite_key": PARENT_SUITE_KEY,
            "parent_dataset_key": source_manifest["dataset_key"],
            "parent_dataset_version": source_manifest["version"],
            "parent_manifest_file_sha256": inputs.source_manifest_file_hashes[difficulty],
        }
        manifest["generation"] = {
            **dict(source_manifest.get("generation") or {}),
            "contract_builder": Path(__file__).name,
            "contract_builder_schema": BUILD_SCHEMA,
            "contract_build_method": "deterministic-uniform-full-suite",
            "llm_review_status": "not_executed",
        }
        manifest["cases"] = cases
        manifests[difficulty] = manifest
    if len(qa_rows) != 150 or len({row["parent_case_id"] for row in qa_rows}) != 150:
        raise RuntimeError("builder must produce exactly 150 unique parent-grain QA rows")
    return manifests, qa_rows, review_payloads


def _summary(inputs: Inputs, manifests: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    readiness = Counter(row["review_readiness"] for row in rows)
    flag_counts = Counter(code for row in rows for code in row["review_flags"])
    hard_counts = Counter(code for row in rows for code in row["hard_blockers"])
    difficulty: dict[str, Any] = {}
    for level in DIFFICULTIES:
        subset = [row for row in rows if row["difficulty"] == level]
        difficulty[level] = {
            "case_count": len(subset),
            "deterministic_candidate_ready": sum(
                row["review_readiness"] == "deterministic_candidate_ready" for row in subset
            ),
            "review_required": sum(
                row["review_readiness"] == "manual_or_llm_review_required" for row in subset
            ),
            "blocked": sum(row["review_readiness"] == "blocked" for row in subset),
            "source_page_text_gap_candidates": sum(
                "SOURCE_PAGE_TEXT_TERM_GAP" in row["review_flags"] for row in subset
            ),
            "concept_without_evidence_cases": sum(
                row["concept_without_evidence_count"] > 0 for row in subset
            ),
            "evidence_without_concept_cases": sum(
                row["evidence_without_concept_count"] > 0 for row in subset
            ),
            "branch_ambiguity_cases": sum(
                any(
                    code in row["review_flags"]
                    for code in (
                        "BRANCH_CONCEPT_ASSIGNMENT_ORDINAL",
                        "QUERY_BRANCH_WITHOUT_CONCEPT",
                        "MULTI_QUERY_SINGLE_BRANCH_PARSE",
                    )
                )
                for row in subset
            ),
            "confirmed_visual_contract_blockers": sum(
                row.get("visual_audit_status")
                in {"confirmed_reference_source_conflict", "confirmed_reference_incomplete"}
                for row in subset
            ),
            "unresolved_visual_review_cases": sum(
                row.get("visual_audit_status") == "manual_visual_review_required"
                for row in subset
            ),
            "adjacent_page_attribution_cases": sum(
                row.get("visual_audit_status") == "adjacent_page_attribution"
                for row in subset
            ),
        }
    return {
        "schema_version": BUILD_SCHEMA,
        "suite_key": SUITE_KEY,
        "parent_suite_key": PARENT_SUITE_KEY,
        "intended_grain": "one versioned candidate contract per parent case; exactly 150",
        "uniformity_guards": {
            "case_id_rules": False,
            "production_answer_input": False,
            "production_pass_fail_input": False,
            "difficulty_specific_extraction_rules": False,
            "all_cases_included": len(rows) == 150,
            "same_llm_review_prompt_required": True,
        },
        "source_lock": {
            "source_pdf_relative_path": str(inputs.manifests["easy"]["source"]["relative_path"]),
            "source_pdf_sha256": _sha256_file(inputs.source_path),
            "source_pdf_pages": len(inputs.source_pages),
            "index_version_id": inputs.suite["index_version_id"],
            "parent_manifest_file_sha256": inputs.source_manifest_file_hashes,
            "parent_canonical_manifest_sha256": inputs.source_qa_summary["input_checks"][
                "canonical_manifest_hashes"
            ],
            "source_visual_audit_sha256": _sha256_file(SOURCE_VISUAL_AUDIT_PATH),
            "source_visual_audit_record_count": len(inputs.visual_audit),
        },
        "counts": {
            "case_count": len(rows),
            "review_readiness": dict(sorted(readiness.items())),
            "review_required_or_blocked": sum(
                row["review_readiness"] != "deterministic_candidate_ready" for row in rows
            ),
            "hard_blocked_cases": sum(row["hard_blocker_count"] > 0 for row in rows),
            "source_ambiguity_candidates": sum(
                any(
                    code in row["review_flags"]
                    for code in (
                        "SOURCE_PAGE_TEXT_TERM_GAP",
                        "TABLE_PAGE_VISUAL_REVIEW_REQUIRED",
                    )
                )
                for row in rows
            ),
            "confirmed_visual_contract_blockers": sum(
                row.get("visual_audit_status")
                in {"confirmed_reference_source_conflict", "confirmed_reference_incomplete"}
                for row in rows
            ),
            "unresolved_visual_review_cases": sum(
                row.get("visual_audit_status") == "manual_visual_review_required"
                for row in rows
            ),
            "adjacent_page_attribution_cases": sum(
                row.get("visual_audit_status") == "adjacent_page_attribution"
                for row in rows
            ),
        },
        "difficulty": difficulty,
        "review_flag_case_counts": dict(sorted(flag_counts.items())),
        "hard_blocker_case_counts": dict(sorted(hard_counts.items())),
        "candidate_manifest_sha256": {
            level: _sha256_json(manifest) for level, manifest in manifests.items()
        },
        "unambiguous_build_assessment": {
            "can_promote_without_review": all(
                row["review_readiness"] == "deterministic_candidate_ready" for row in rows
            ),
            "decision": "review_required"
            if any(row["review_readiness"] != "deterministic_candidate_ready" for row in rows)
            else "deterministic_candidate_ready",
            "reason": (
                "Heuristic concept, branch, polarity, or source-page mappings remain explicitly flagged; "
                "the generated manifests are not gold labels and are not importable baselines."
            ),
        },
        "llm_review": {
            "status": "dry_run_requests_prepared_not_executed",
            "planned_case_calls": 150,
            "planned_calibration_calls": 4,
            "planned_total_calls": 154,
            "semantic_payload_excludes": [
                "case_id",
                "difficulty",
                "production answer",
                "production pass/fail",
                "production metrics",
            ],
        },
    }


_REVIEW_SYSTEM_PROMPT = """You are auditing answer contracts for a grounded RAG evaluation suite. Apply exactly the same rules to every supplied item. Use only the query, reference answer, legacy labels, and evidence in the payload. Do not infer from case identity, difficulty, model output, pass/fail history, or evaluation metrics; those fields are intentionally absent.

Return one JSON object only. The contract must require every independently necessary answer concept (all_of), while each concept's semantically equivalent surface forms belong in any_of. Preserve every critical number, unit, identifier, condition, comparison side, procedure step, and polarity. Map each required concept to the evidence labels that directly support it and to the query branches it answers. Do not count an evidence item as covered merely because it is topically related. If the source/reference is ambiguous or unsupported, mark verdict=block instead of inventing a label. For unanswerable items, require abstention without additional factual claims. Unsupported or contradictory factual additions are always forbidden.

Required output keys: verdict (approve|revise|block), answerability (answer|abstain), required_concepts {all_of:[{concept_key,canonical_text,any_of,evidence_labels,branch_labels}]}, critical_constraints {numerics:[{value,unit,evidence_labels}],identifiers:[{value,evidence_labels}],polarity:{required,cues}}, required_branches [{branch_label,query_span,required_concept_keys}], evidence_concept_coverage [{evidence_label,required_concept_keys,status}], review_notes [string]."""


def _review_requests(
    payloads: Sequence[dict[str, Any]], *, expected_count: int = 150
) -> tuple[list[dict[str, Any]], str]:
    prompt_hash = _sha256_bytes(_REVIEW_SYSTEM_PROMPT.encode("utf-8"))
    requests: list[dict[str, Any]] = []
    for payload in payloads:
        payload_hash = _sha256_json(payload)
        requests.append(
            {
                "custom_id": f"concept-review-{payload_hash[:20]}",
                "prompt_sha256": prompt_hash,
                "payload_sha256": payload_hash,
                "request": {
                    "model": "qwen3.7-max",
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        },
                    ],
                },
            }
        )
    if len(requests) != expected_count or len({item["custom_id"] for item in requests}) != expected_count:
        raise RuntimeError(f"review request pack must contain {expected_count} unique payloads")
    if len({item["prompt_sha256"] for item in requests}) != 1:
        raise RuntimeError("review request pack must use one identical prompt")
    forbidden_payload_keys = {
        "case_id",
        "parent_case_id",
        "difficulty",
        "production_final_correct",
        "production_final_accuracy",
        "final_answer",
        "pre_grounding_answer",
        "production_metrics",
    }
    for item in requests:
        content = json.loads(item["request"]["messages"][1]["content"])
        stack: list[Any] = [content]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                leaked = forbidden_payload_keys.intersection(current)
                if leaked:
                    raise RuntimeError(f"semantic review payload leaked forbidden keys: {sorted(leaked)}")
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
    return requests, prompt_hash


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _suite_manifest(
    inputs: Inputs,
    manifests: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUITE_SCHEMA,
        "suite_key": SUITE_KEY,
        "version": DATASET_VERSION,
        "status": "candidate_pending_full_review",
        "parent_suite_key": PARENT_SUITE_KEY,
        "evaluation_mode": "live",
        "knowledge_base_id": inputs.suite["knowledge_base_id"],
        "source_sha256": summary["source_lock"]["source_pdf_sha256"],
        "index_version_id": inputs.suite["index_version_id"],
        "import_allowed": False,
        "baseline_registration_allowed": False,
        "datasets": {
            difficulty: {
                "path": f"datasets/{difficulty}.v2.json",
                "dataset_key": manifest["dataset_key"],
                "version": manifest["version"],
                "case_count": len(manifest["cases"]),
                "checksum": _sha256_json(manifest),
                "parent_path": f"../datasets/{difficulty}.json",
                "parent_file_sha256": inputs.source_manifest_file_hashes[difficulty],
            }
            for difficulty, manifest in manifests.items()
        },
        "review_gate": {
            "full_150_same_prompt_review_required": True,
            "independent_review_merge_required": True,
            "hard_blockers_must_equal_zero": True,
            "all_review_flags_must_be_resolved_or_accepted_with_rationale": True,
        },
    }


def _notebook(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a small executable notebook without hand-maintained outputs."""

    root_expr = "Path.cwd() if (Path.cwd() / 'qa' / 'summary.json').exists() else Path.cwd() / 'docs' / 'evaluation-baselines' / '2026-08-10-three-difficulty' / 'remediation-2026-08-10' / 'suite-v2-concept-contract'"
    summary_text = summary["unambiguous_build_assessment"]["decision"]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## tl;dr\n",
                f"确定性构建结论：`{summary_text}`。本 Notebook 复核 150 行候选合同、已保存的全量审阅结果（若存在）与审核门；不发起模型调用，也不导入基线。\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Context & Methods\n",
                "输入为冻结的 v1 manifests、绑定 evidence、源 PDF 和 source/evaluator QA。所有题使用同一规则；case_id、难度与生产成败不进入语义审核 payload。\n",
                "\n### Key Assumptions\n",
                "字符串分组、分支切分和页面可提取文本核对只能生成候选，任何审核标记都不能被视为已确认金标。\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                f"ROOT = {root_expr}\n",
                "summary = json.loads((ROOT / 'qa' / 'summary.json').read_text(encoding='utf-8'))\n",
                "cases = json.loads((ROOT / 'qa' / 'cases.json').read_text(encoding='utf-8'))\n",
                "reviewed_summary_path = ROOT / 'reviewed' / 'summary.json'\n",
                "reviewed_summary = json.loads(reviewed_summary_path.read_text(encoding='utf-8')) if reviewed_summary_path.exists() else None\n",
                "len(cases), len({row['parent_case_id'] for row in cases})\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data\n", "150 行 QA 的 grain 是 parent_case_id；以下检查覆盖、唯一性和脱敏审阅包。\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "assert len(cases) == 150\n",
                "assert len({row['case_id'] for row in cases}) == 150\n",
                "assert len({row['parent_case_id'] for row in cases}) == 150\n",
                "assert sum(summary['difficulty'][d]['case_count'] for d in ('easy','medium','hard')) == 150\n",
                "summary['counts']\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results\n", "按难度与审核状态检查候选覆盖；精确计数来自已保存 summary，不依赖隐藏状态。\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "{difficulty: summary['difficulty'][difficulty] for difficulty in ('easy','medium','hard')}\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "requests = [json.loads(line) for line in (ROOT / 'review' / 'requests.jsonl').read_text(encoding='utf-8').splitlines() if line]\n",
                "assert len(requests) == 150\n",
                "assert len({row['prompt_sha256'] for row in requests}) == 1\n",
                "assert all('case_id' not in row['request']['messages'][1]['content'] for row in requests)\n",
                "len(requests), requests[0]['prompt_sha256']\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if reviewed_summary is not None:\n",
                "    assert reviewed_summary['case_count'] == 150\n",
                "    assert reviewed_summary['import_allowed'] is False\n",
                "    assert reviewed_summary['baseline_registration_allowed'] is False\n",
                "    display({key: reviewed_summary[key] for key in ('status', 'verdict_counts', 'llm_verdict_counts', 'blocked_case_count', 'usage')})\n",
                "else:\n",
                "    print('reviewed artifacts not present; deterministic candidate only')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Takeaways\n",
                "- v2 候选与旧 manifests 并存，且 suite 明确禁止导入/登记。\n",
                "- 自动提取能覆盖全部 150 题，但审核标记代表真实语义不确定性。\n",
                "- reviewed 版本只有在 150/150 同 prompt 响应完整时生成；它仍需解决 block 并完成独立验证，不能直接接入 UI、导入或登记基线。\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "stt.eval.answer-contract.v2-candidate.schema.json",
        "title": "STT evaluation answer contract v2 candidate",
        "type": "object",
        "required": [
            "schema_version",
            "contract_status",
            "answerability",
            "required_concepts",
            "critical_constraints",
            "required_branches",
            "evidence_concept_coverage",
            "review",
        ],
        "properties": {
            "schema_version": {"const": CONTRACT_SCHEMA},
            "contract_status": {"const": "candidate_pending_full_review"},
            "answerability": {
                "type": "object",
                "required": ["required", "unsupported_factual_additions_forbidden", "contradictory_claims_forbidden"],
            },
            "required_concepts": {
                "type": "object",
                "required": ["all_of"],
                "properties": {
                    "all_of": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["concept_id", "canonical_text", "any_of", "derivation", "evidence_support"],
                        },
                    }
                },
            },
            "critical_constraints": {"type": "object"},
            "required_branches": {"type": "array"},
            "evidence_concept_coverage": {"type": "array"},
            "review": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _checksums() -> None:
    excluded = {OUTPUT_ROOT / "SHA256SUMS"}
    paths = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
        and path not in excluded
        and "__pycache__" not in path.parts
        and not path.name.endswith(".tmp")
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}" for path in paths]
    (OUTPUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(*, write: bool) -> dict[str, Any]:
    inputs = _load_inputs()
    manifests, qa_rows, review_payloads = _build_manifests(inputs)
    summary = _summary(inputs, manifests, qa_rows)
    suite = _suite_manifest(inputs, manifests, summary)
    review_requests, prompt_hash = _review_requests(review_payloads)
    summary["llm_review"]["prompt_sha256"] = prompt_hash
    summary["llm_review"]["request_pack_sha256"] = _sha256_json(review_requests)

    if write:
        for difficulty, manifest in manifests.items():
            _write_json(OUTPUT_ROOT / "datasets" / f"{difficulty}.v2.json", manifest)
        _write_json(OUTPUT_ROOT / "suite-v2.json", suite)
        _write_json(OUTPUT_ROOT / "qa" / "cases.json", qa_rows)
        _write_csv(OUTPUT_ROOT / "qa" / "cases.csv", qa_rows)
        _write_json(OUTPUT_ROOT / "qa" / "summary.json", summary)
        _write_jsonl(OUTPUT_ROOT / "review" / "requests.jsonl", review_requests)
        (OUTPUT_ROOT / "review" / "PROMPT.md").write_text(
            "# Full-suite concept-contract review prompt\n\n"
            f"Prompt SHA-256: `{prompt_hash}`\n\n"
            "The following system prompt is identical for all 150 semantic review requests.\n\n"
            "```text\n"
            + _REVIEW_SYSTEM_PROMPT
            + "\n```\n",
            encoding="utf-8",
        )
        _write_json(
            OUTPUT_ROOT / "review" / "request-map.json",
            [
                {
                    "custom_id": request["custom_id"],
                    "case_id": qa_row["case_id"],
                    "parent_case_id": qa_row["parent_case_id"],
                    "difficulty": qa_row["difficulty"],
                }
                for request, qa_row in zip(review_requests, qa_rows, strict=True)
            ],
        )
        _write_json(
            OUTPUT_ROOT / "review" / "dry-run.json",
            {
                "status": "not_executed",
                "request_count": len(review_requests),
                "calibration_call_count": 4,
                "estimated_total_call_count": len(review_requests) + 4,
                "prompt_sha256": prompt_hash,
                "payload_excludes": summary["llm_review"]["semantic_payload_excludes"],
                "execution_gate": [
                    "--execute",
                    "--acknowledge-cost",
                    "--confirm-full-review",
                    "successful synthetic calibration",
                    "DASHSCOPE_API_KEY in environment",
                ],
            },
        )
        _write_json(OUTPUT_ROOT / "schema" / "answer-contract-v2.schema.json", _schema())
        _write_json(
            OUTPUT_ROOT / "suite_v2_concept_contract.ipynb",
            _notebook(summary),
        )
        original_after = {
            difficulty: _sha256_file(SOURCE_DATASET_ROOT / f"{difficulty}.json")
            for difficulty in DIFFICULTIES
        }
        if original_after != inputs.source_manifest_file_hashes:
            raise RuntimeError("a frozen source manifest changed during the build")
        _checksums()
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the deterministic summary without writing outputs",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build(write=not args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
