"""Systematic case-grain QA for the frozen three-difficulty evaluation suite.

The audit is intentionally read-only with respect to the application, source
index, archived runs, and immutable dataset manifests.  It writes only the
derived artifacts in this directory.  No rule depends on a case id.

Intended grain
--------------
One row per immutable ``case_id``.  The primary join is:

    dataset case (150) 1:1 archived live-cand-max-mono-01 result (150)

The script validates uniqueness, completeness, canonical manifest hashes,
archived dataset hashes, source-PDF identity/page count, and the frozen indexed
evidence records before calculating any diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from pypdf import PdfReader


QA_ROOT = Path(__file__).resolve().parent
REMEDIATION_ROOT = QA_ROOT.parent
SUITE_ROOT = REMEDIATION_ROOT.parent
PROJECT_ROOT = SUITE_ROOT.parents[2]
DATASET_ROOT = SUITE_ROOT / "datasets"
RUN_ROOT = REMEDIATION_ROOT / "runs" / "live-cand-max-mono-01"
CONTROL_DB = PROJECT_ROOT / "volumes" / "control" / "stt-control.db"
DIFFICULTIES = ("easy", "medium", "hard")
TARGETS = {
    "easy": {"operator": ">", "threshold": 0.95, "minimum_correct": 48},
    "medium": {"operator": ">", "threshold": 0.85, "minimum_correct": 43},
    "hard": {"operator": None, "threshold": None, "minimum_correct": None},
}
RUN_LABEL = "live-cand-max-mono-01"

sys.path.insert(0, str(PROJECT_ROOT))
from app.evaluation.metrics import evaluate_case  # noqa: E402


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?")
_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._/+-]*[A-Za-z])(?=[A-Za-z0-9._/+-]*\d)"
    r"[A-Za-z0-9]+(?:[._/+-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
_NEGATIVE_RE = re.compile(
    r"(?:请勿|不得|禁止|切勿|不要|不能|不适合|不支持|未|没有|无法|无介质|不超过)"
)
_MULTI_QUERY_RE = re.compile(
    r"(?:分别|各自|以及|同时|并且|并说明|并列出|和哪|与哪|哪些|哪些操作|哪些条件|"
    r"如何.*(?:并|以及)|比较|差异|全部|每个|两种|两个|三种|步骤|哪些值|哪些项)"
)
_TABLE_QUERY_RE = re.compile(
    r"(?:表格|(?<!列)表中|字段(?:值|内容)|参数表|规格表|根据.*规格|哪一种有毒有害物质|哪些值|列出.*值)"
)
_SINGULAR_CHOICE_RE = re.compile(r"(?:哪一种|哪个|哪一项|何种)")
_INFERENCE_HEAVY_RE = re.compile(
    r"(?:如何判断|为何|为什么|影响|原因|若.*(?:应|如何)|推断|综合|优先选择|是否由)"
)
_CLAUSE_SPLIT_RE = re.compile(r"[；;。\n]+")


@dataclass(frozen=True)
class Inputs:
    suite: dict[str, Any]
    manifests: dict[str, dict[str, Any]]
    results: dict[str, dict[str, dict[str, Any]]]
    run_manifests: dict[str, dict[str, Any]]
    source_path: Path
    source_pages: list[str]
    indexed_evidence: dict[str, dict[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _normalized(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold(),
    ).strip()


def _compact(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    return "".join(ch for ch in normalized if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def _contains(haystack: Any, needle: Any) -> bool:
    term = _compact(needle)
    return bool(term) and term in _compact(haystack)


def _ngrams(value: Any, size: int = 3) -> set[str]:
    compact = _compact(value)
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nested(value: Any, *path: str) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: result must be an object")
        rows.append(value)
    return rows


def _load_indexed_evidence(index_version_id: str) -> dict[str, dict[str, Any]]:
    uri = f"file:{CONTROL_DB.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, chunk_id, document_version_id, index_version_id,
                   primary_page_no, content, content_hash, source_label
            FROM evidence_records
            WHERE index_version_id = ?
            ORDER BY id
            """,
            (index_version_id,),
        ).fetchall()
    finally:
        connection.close()
    return {str(row["id"]): dict(row) for row in rows}


def load_inputs() -> Inputs:
    suite = json.loads((SUITE_ROOT / "suite.json").read_text(encoding="utf-8"))
    manifests = {
        difficulty: json.loads((DATASET_ROOT / f"{difficulty}.json").read_text(encoding="utf-8"))
        for difficulty in DIFFICULTIES
    }
    source_relative = manifests["easy"]["source"]["relative_path"]
    source_path = (PROJECT_ROOT / source_relative).resolve()
    pdf = PdfReader(str(source_path))
    source_pages = [page.extract_text() or "" for page in pdf.pages]

    results: dict[str, dict[str, dict[str, Any]]] = {}
    run_manifests: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        artifact_dirs = sorted((RUN_ROOT / "artifacts" / difficulty).glob("erun_*"))
        if len(artifact_dirs) != 1:
            raise ValueError(
                f"expected one archived {difficulty} artifact directory, found {len(artifact_dirs)}"
            )
        artifact_dir = artifact_dirs[0]
        rows = _load_jsonl(artifact_dir / "results.jsonl")
        counts = Counter(str(row.get("case_id") or "") for row in rows)
        duplicates = sorted(case_id for case_id, count in counts.items() if count != 1)
        if duplicates:
            raise ValueError(f"duplicate archived result case ids in {difficulty}: {duplicates}")
        results[difficulty] = {str(row["case_id"]): row for row in rows}
        run_manifests[difficulty] = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )

    index_version_id = str(suite["index_version_id"])
    indexed_evidence = _load_indexed_evidence(index_version_id)
    return Inputs(
        suite=suite,
        manifests=manifests,
        results=results,
        run_manifests=run_manifests,
        source_path=source_path,
        source_pages=source_pages,
        indexed_evidence=indexed_evidence,
    )


def _term_groups(terms: list[str]) -> list[list[str]]:
    """Group overlapping term variants without using case-specific rules."""

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
            overlap = a in b or b in a
            similarity = SequenceMatcher(None, a, b).ratio()
            if overlap or similarity >= 0.86:
                union(left, right)
    grouped: dict[int, list[str]] = defaultdict(list)
    for index, term in enumerate(terms):
        grouped[find(index)].append(term)
    return list(grouped.values())


def _score_answer(case: dict[str, Any], answer: str, *, abstained: bool | None = None) -> dict[str, Any]:
    observation: dict[str, Any] = {"answer": answer, "context": []}
    if abstained is not None:
        observation["abstained"] = abstained
    evaluated = evaluate_case(case, observation)
    generation = evaluated["metrics"]["generation"]
    abstention = evaluated["metrics"]["abstention"]
    if case.get("unanswerable"):
        final_correct = bool(abstention.get("correct"))
    else:
        correctness = _safe_float(generation.get("correctness")) or 0.0
        forbidden_rate = _safe_float(generation.get("forbidden_term_hit_rate"))
        final_correct = correctness >= 0.75 and (forbidden_rate is None or forbidden_rate == 0)
    return {
        "correctness": generation.get("correctness"),
        "reference_semantic_recall": generation.get("reference_semantic_recall"),
        "expected_term_recall": generation.get("expected_term_recall"),
        "reference_constraints_preserved": generation.get("reference_constraints_preserved"),
        "abstention_correct": abstention.get("correct"),
        "final_correct": final_correct,
    }


def _truncated_reference(reference: str) -> str | None:
    clauses = [item.strip(" ，,、") for item in _CLAUSE_SPLIT_RE.split(reference) if item.strip()]
    if len(clauses) >= 2:
        return "；".join(clauses[:-1]) + "。"
    comma_parts = [item.strip() for item in re.split(r"[，,]", reference) if item.strip()]
    if len(comma_parts) >= 4:
        return "，".join(comma_parts[:-1]) + "。"
    list_parts = [item.strip() for item in reference.split("、") if item.strip()]
    if len(list_parts) >= 3:
        return "、".join(list_parts[:-1]) + "。"
    return None


def _mutate_first_number(reference: str, expected_terms: list[str]) -> str | None:
    """Mutate the first number explicitly represented in the scoring labels."""

    expected_numbers = {
        number for term in expected_terms for number in _NUMBER_RE.findall(term)
    }
    match = next(
        (
            candidate
            for candidate in _NUMBER_RE.finditer(reference)
            if candidate.group(0) in expected_numbers
        ),
        None,
    )
    if match is None:
        return None
    raw = match.group(0)
    if "." in raw:
        replacement = str(float(raw) + 1.0).rstrip("0").rstrip(".")
    else:
        replacement = str(int(raw) + 1)
    return reference[: match.start()] + replacement + reference[match.end() :]


def _table_data_row_count(evidence_text: str) -> int:
    rows = []
    for line in evidence_text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        if re.fullmatch(r"[| :\-]+", stripped):
            continue
        rows.append(stripped)
    # The first non-separator row is normally the header.
    data_rows = rows[1:]
    substantive = []
    for row in data_rows:
        first_cell = row.strip("|").split("|", maxsplit=1)[0].strip()
        if re.match(r"^(?:备注|注[:：]?|说明[:：]?)", first_cell):
            continue
        substantive.append(row)
    return len(substantive)


def _case_row(
    *,
    difficulty: str,
    case: dict[str, Any],
    result: dict[str, Any],
    inputs: Inputs,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    reference = str(case.get("reference_answer") or "")
    query = str(case.get("query") or "")
    expected_terms = [str(term) for term in case.get("expected_terms") or []]
    forbidden_terms = [str(term) for term in case.get("forbidden_terms") or []]
    evidence = [dict(item) for item in case.get("relevant_evidence") or []]
    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)
    term_groups = _term_groups(expected_terms)

    expected_in_reference = [term for term in expected_terms if _contains(reference, term)]
    expected_in_evidence = [term for term in expected_terms if _contains(evidence_text, term)]
    redundant_term_count = len(expected_terms) - len(term_groups)
    largest_group_weight = (
        max((len(group) for group in term_groups), default=0) / len(expected_terms)
        if expected_terms
        else 0.0
    )

    evidence_store_matches = 0
    evidence_hash_matches = 0
    evidence_chunk_matches = 0
    evidence_page_matches = 0
    pdf_term_hits = 0
    pdf_terms_total = 0
    pdf_evidence_overlap: list[float] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "")
        stored = inputs.indexed_evidence.get(evidence_id)
        if stored is not None:
            evidence_store_matches += 1
            if str(stored.get("content_hash") or "") == str(item.get("content_hash") or ""):
                evidence_hash_matches += 1
            if str(stored.get("chunk_id") or "") == str(item.get("chunk_id") or ""):
                evidence_chunk_matches += 1
            if int(stored.get("primary_page_no") or 0) == int(item.get("page") or 0):
                evidence_page_matches += 1
        page_number = int(item.get("page") or 0)
        page_text = (
            inputs.source_pages[page_number - 1]
            if 1 <= page_number <= len(inputs.source_pages)
            else ""
        )
        match_terms = item.get("match_terms")
        if isinstance(match_terms, str):
            match_terms = [match_terms]
        for term in match_terms or []:
            pdf_terms_total += 1
            if _contains(page_text, term):
                pdf_term_hits += 1
        evidence_ngrams = _ngrams(item.get("text") or "", size=3)
        page_ngrams = _ngrams(page_text, size=3)
        pdf_evidence_overlap.append(_jaccard(evidence_ngrams, page_ngrams))

    reference_numbers = set(_NUMBER_RE.findall(reference))
    evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
    reference_identifiers = {value.casefold() for value in _IDENTIFIER_RE.findall(reference)}
    evidence_identifiers = {value.casefold() for value in _IDENTIFIER_RE.findall(evidence_text)}

    result_metrics = _nested(result, "metrics", "metrics") or {}
    generation = result_metrics.get("generation") or {}
    outcome = result_metrics.get("case_outcome") or {}
    retrieval = result_metrics.get("retrieval") or {}
    citation = result_metrics.get("citation") or {}
    observation = result.get("observation") or {}
    final_answer = str(observation.get("answer") or "")
    pre_answer = str(observation.get("pre_grounding_answer") or "")
    recomputed = evaluate_case(case, observation)
    recomputed_generation = recomputed["metrics"]["generation"]
    stored_correctness = _safe_float(generation.get("correctness"))
    recomputed_correctness = _safe_float(recomputed_generation.get("correctness"))
    evaluator_drift = not (
        stored_correctness is None
        and recomputed_correctness is None
        or stored_correctness is not None
        and recomputed_correctness is not None
        and abs(stored_correctness - recomputed_correctness) <= 1e-12
    )

    reference_self_score = _score_answer(case, reference, abstained=bool(case.get("unanswerable")))
    expected_only_answer = "；".join(
        expected_terms[: max(1, math.ceil(len(expected_terms) * 0.75))]
    )
    expected_only_score = (
        _score_answer(case, expected_only_answer, abstained=False)
        if expected_terms and not case.get("unanswerable")
        else None
    )
    truncated_answer = _truncated_reference(reference)
    truncation_score = (
        _score_answer(case, truncated_answer, abstained=False)
        if truncated_answer and not case.get("unanswerable")
        else None
    )
    numeric_mutation = _mutate_first_number(reference, expected_terms)
    numeric_mutation_score = (
        _score_answer(case, numeric_mutation, abstained=False)
        if numeric_mutation and not case.get("unanswerable")
        else None
    )
    false_addition_answer = (
        reference + " 此外，这款打印机支持手册未记载的卫星通信功能。"
    )
    false_addition_score = (
        _score_answer(case, false_addition_answer, abstained=False)
        if not case.get("unanswerable")
        else None
    )
    fabricated_refusal_score = (
        _score_answer(
            case,
            "无法仅依据手册确定，但上海现货价为 9999 元。",
            abstained=True,
        )
        if case.get("unanswerable")
        else None
    )

    is_multi_query = bool(_MULTI_QUERY_RE.search(query))
    is_table_query = bool(_TABLE_QUERY_RE.search(query))
    is_inference_heavy = bool(_INFERENCE_HEAVY_RE.search(query))
    singular_multi_answer = bool(
        _SINGULAR_CHOICE_RE.search(query)
        and (reference.count("、") >= 2 or reference.count("；") >= 2)
    )
    table_data_rows = _table_data_row_count(evidence_text)
    negative_query = bool(_NEGATIVE_RE.search(query))
    negative_reference = bool(_NEGATIVE_RE.search(reference))
    polarity_mismatch = negative_query and not negative_reference and not case.get("unanswerable")

    issue_codes: list[str] = []
    classifications: set[str] = set()
    if len(term_groups) < len(expected_terms):
        issue_codes.append("EXPECTED_TERM_VARIANTS_OVERLAP")
        classifications.add("dataset_reference_candidate")
    if len(term_groups) > 1 and largest_group_weight >= 0.75:
        issue_codes.append("EXPECTED_TERM_DUPLICATE_WEIGHT_CAN_DOMINATE")
        classifications.update(("dataset_reference_candidate", "evaluator_error"))
    if any(len(_compact(term)) < 4 for term in expected_terms):
        issue_codes.append("EXPECTED_TERM_TOO_SHORT")
        classifications.add("dataset_reference_candidate")
    if len(expected_in_reference) != len(expected_terms):
        issue_codes.append("EXPECTED_TERM_ABSENT_REFERENCE")
        classifications.add("dataset_reference_candidate")
    if not case.get("unanswerable") and len(expected_in_evidence) != len(expected_terms):
        issue_codes.append("EXPECTED_TERM_ABSENT_EVIDENCE")
        classifications.add("dataset_reference_candidate")
    if reference_numbers - evidence_numbers and not case.get("unanswerable"):
        issue_codes.append("REFERENCE_NUMBER_ABSENT_EVIDENCE")
        classifications.add("dataset_reference_candidate")
    if reference_identifiers - evidence_identifiers and not case.get("unanswerable"):
        issue_codes.append("REFERENCE_IDENTIFIER_ABSENT_EVIDENCE")
        classifications.add("dataset_reference_candidate")
    if pdf_terms_total and pdf_term_hits < pdf_terms_total:
        issue_codes.append("PDF_TEXT_EXTRACTION_TERM_GAP")
        classifications.add("source_ambiguity")
    if singular_multi_answer:
        issue_codes.append("QUERY_SINGULAR_REFERENCE_ENUMERATION")
        classifications.add("dataset_reference_candidate")
    if is_table_query and table_data_rows == 0 and not case.get("unanswerable"):
        issue_codes.append("TABLE_QUERY_WITHOUT_PARSED_DATA_ROW")
        classifications.add("source_ambiguity")
    if polarity_mismatch:
        issue_codes.append("QUERY_REFERENCE_POLARITY_CANDIDATE")
        classifications.add("dataset_reference_candidate")
    if difficulty == "hard" and not case.get("unanswerable"):
        empty_match_evidence = sum(
            1 for item in evidence if not (item.get("match_terms") or [])
        )
        if empty_match_evidence:
            issue_codes.append("HARD_EVIDENCE_WITHOUT_SCORING_CONCEPT")
            classifications.add("dataset_reference_candidate")
    else:
        empty_match_evidence = 0
    if evaluator_drift:
        issue_codes.append("ARCHIVED_EVALUATOR_DRIFT")
        classifications.add("evaluator_error")
    if reference_self_score["final_correct"] is not True:
        issue_codes.append("REFERENCE_FAILS_OWN_EVALUATOR")
        classifications.add("evaluator_error")
    expected_only_vulnerability = bool(
        expected_only_score
        and expected_only_score["final_correct"]
        and (
            _safe_float(expected_only_score["reference_semantic_recall"]) is None
            or float(expected_only_score["reference_semantic_recall"]) < 0.75
        )
    )
    if expected_only_vulnerability:
        issue_codes.append("EXPECTED_TERMS_ONLY_CAN_PASS")
        classifications.add("evaluator_error")
    if is_multi_query and truncation_score and truncation_score["final_correct"]:
        issue_codes.append("MULTI_BRANCH_TRUNCATION_CAN_PASS")
        classifications.add("evaluator_error")
    if numeric_mutation_score and numeric_mutation_score["final_correct"]:
        issue_codes.append("NUMERIC_MUTATION_CAN_PASS")
        classifications.add("evaluator_error")
    if false_addition_score and false_addition_score["final_correct"]:
        issue_codes.append("UNSUPPORTED_FALSE_ADDITION_CAN_PASS")
        classifications.add("evaluator_error")
    if fabricated_refusal_score and fabricated_refusal_score["final_correct"]:
        issue_codes.append("FABRICATED_FACT_WITH_REFUSAL_CAN_PASS")
        classifications.add("evaluator_error")

    final_correct = outcome.get("final_answer_correct")
    faithfulness = _safe_float(generation.get("faithfulness"))
    citation_coverage = _safe_float(citation.get("coverage"))
    expected_recall = _safe_float(generation.get("expected_term_recall"))
    reference_recall = _safe_float(generation.get("reference_semantic_recall"))
    if final_correct is True and faithfulness is not None and faithfulness < 0.8:
        issue_codes.append("ACTUAL_FINAL_CORRECT_WITH_LOW_FAITHFULNESS")
        classifications.add("evaluator_error")
    if final_correct is True and citation_coverage is not None and citation_coverage < 0.8:
        issue_codes.append("ACTUAL_FINAL_CORRECT_WITH_LOW_CITATION_COVERAGE")
        classifications.add("evaluator_error")
    if (
        final_correct is True
        and expected_recall is not None
        and expected_recall >= 0.75
        and (reference_recall is None or reference_recall < 0.75)
    ):
        issue_codes.append("ACTUAL_PASS_DEPENDS_ON_EXPECTED_TERMS")
        classifications.add("evaluator_error")
    if (
        final_correct is False
        and reference_recall is not None
        and 0.65 <= reference_recall < 0.75
        and bool(generation.get("reference_constraints_preserved"))
    ):
        issue_codes.append("ACTUAL_NEAR_THRESHOLD_FALSE_NEGATIVE_CANDIDATE")
        classifications.add("evaluator_error_candidate")

    source_integrity_ok = (
        len(evidence) == evidence_store_matches
        == evidence_hash_matches
        == evidence_chunk_matches
        == evidence_page_matches
    )
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "query": query,
        "reference_answer": reference,
        "unanswerable": bool(case.get("unanswerable")),
        "case_type": str((case.get("metadata") or {}).get("type") or ""),
        "critical": bool((case.get("metadata") or {}).get("critical")),
        "evidence_count": len(evidence),
        "expected_term_count": len(expected_terms),
        "expected_concept_group_count": len(term_groups),
        "expected_term_redundant_count": redundant_term_count,
        "largest_expected_group_weight": round(largest_group_weight, 6),
        "expected_terms": expected_terms,
        "expected_term_groups": term_groups,
        "forbidden_terms": forbidden_terms,
        "expected_reference_coverage": (
            len(expected_in_reference) / len(expected_terms) if expected_terms else None
        ),
        "expected_evidence_coverage": (
            len(expected_in_evidence) / len(expected_terms) if expected_terms else None
        ),
        "reference_numbers_absent_evidence": sorted(reference_numbers - evidence_numbers),
        "reference_identifiers_absent_evidence": sorted(reference_identifiers - evidence_identifiers),
        "source_integrity_ok": source_integrity_ok,
        "indexed_evidence_joined": evidence_store_matches,
        "pdf_match_term_coverage": (
            pdf_term_hits / pdf_terms_total if pdf_terms_total else None
        ),
        "pdf_evidence_trigram_jaccard_mean": (
            mean(pdf_evidence_overlap) if pdf_evidence_overlap else None
        ),
        "is_multi_query": is_multi_query,
        "is_table_query": is_table_query,
        "is_inference_heavy": is_inference_heavy,
        "table_data_row_count": table_data_rows,
        "singular_multi_answer_candidate": singular_multi_answer,
        "negative_query": negative_query,
        "negative_reference": negative_reference,
        "hard_empty_match_evidence_count": empty_match_evidence,
        "archived_run_id": str(result.get("run_id") or ""),
        "archived_result_id": str(result.get("id") or ""),
        "archived_query_matches": str(_nested(result, "metrics", "query") or "") == query,
        "final_answer": final_answer,
        "pre_grounding_answer": pre_answer,
        "production_final_correct": final_correct,
        "production_final_accuracy": outcome.get("final_answer_accuracy"),
        "production_correctness": generation.get("correctness"),
        "production_expected_term_recall": generation.get("expected_term_recall"),
        "production_reference_semantic_recall": generation.get("reference_semantic_recall"),
        "production_reference_constraints_preserved": generation.get(
            "reference_constraints_preserved"
        ),
        "production_faithfulness": generation.get("faithfulness"),
        "production_citation_coverage": citation.get("coverage"),
        "production_strict_pass": outcome.get("strict_supported_case_pass"),
        "production_recall_at_5": (retrieval.get("recall_at_k") or {}).get("5"),
        "recomputed_correctness": recomputed_generation.get("correctness"),
        "archived_evaluator_drift": evaluator_drift,
        "reference_self_pass": reference_self_score["final_correct"],
        "expected_terms_only_pass": (
            expected_only_vulnerability if expected_only_score else None
        ),
        "multi_branch_truncation_pass": (
            truncation_score["final_correct"]
            if truncation_score and is_multi_query
            else None
        ),
        "numeric_mutation_pass": (
            numeric_mutation_score["final_correct"] if numeric_mutation_score else None
        ),
        "unsupported_false_addition_pass": (
            false_addition_score["final_correct"] if false_addition_score else None
        ),
        "fabricated_fact_with_refusal_pass": (
            fabricated_refusal_score["final_correct"]
            if fabricated_refusal_score
            else None
        ),
        "issue_codes": sorted(set(issue_codes)),
        "qa_classifications": sorted(classifications),
    }


def _input_checks(inputs: Inputs) -> dict[str, Any]:
    suite = inputs.suite
    all_case_ids = [
        str(case["case_id"])
        for difficulty in DIFFICULTIES
        for case in inputs.manifests[difficulty]["cases"]
    ]
    all_result_ids = [
        case_id
        for difficulty in DIFFICULTIES
        for case_id in inputs.results[difficulty]
    ]
    canonical_hashes = {
        difficulty: _sha256_json(inputs.manifests[difficulty])
        for difficulty in DIFFICULTIES
    }
    suite_hashes = {
        difficulty: str(suite["datasets"][difficulty]["checksum"])
        for difficulty in DIFFICULTIES
    }
    archived_hashes = {
        difficulty: str(inputs.run_manifests[difficulty]["dataset_checksum"])
        for difficulty in DIFFICULTIES
    }
    file_hashes = {
        difficulty: _sha256_file(DATASET_ROOT / f"{difficulty}.json")
        for difficulty in DIFFICULTIES
    }
    source_hash = _sha256_file(inputs.source_path)
    source_hashes_declared = {
        str(inputs.manifests[difficulty]["source"]["content_sha256"])
        for difficulty in DIFFICULTIES
    }
    source_page_counts_declared = {
        int(inputs.manifests[difficulty]["source"]["page_count"])
        for difficulty in DIFFICULTIES
    }
    return {
        "intended_grain": "one immutable evaluation case per case_id",
        "expected_case_count": 150,
        "dataset_case_count": len(all_case_ids),
        "dataset_case_id_unique_count": len(set(all_case_ids)),
        "result_case_count": len(all_result_ids),
        "result_case_id_unique_count": len(set(all_result_ids)),
        "joined_case_count": len(set(all_case_ids) & set(all_result_ids)),
        "dataset_without_result": sorted(set(all_case_ids) - set(all_result_ids)),
        "result_without_dataset": sorted(set(all_result_ids) - set(all_case_ids)),
        "canonical_manifest_hashes": canonical_hashes,
        "suite_manifest_hashes": suite_hashes,
        "archived_run_dataset_hashes": archived_hashes,
        "dataset_file_sha256": file_hashes,
        "manifest_hashes_match_suite": canonical_hashes == suite_hashes,
        "manifest_hashes_match_archived_run": canonical_hashes == archived_hashes,
        "source_path": str(inputs.source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": source_hash,
        "source_declared_sha256_values": sorted(source_hashes_declared),
        "source_hash_matches": source_hashes_declared == {source_hash}
        and source_hash == str(suite["source_sha256"]),
        "source_pdf_page_count": len(inputs.source_pages),
        "source_declared_page_counts": sorted(source_page_counts_declared),
        "source_page_count_matches": source_page_counts_declared == {len(inputs.source_pages)},
        "indexed_evidence_count": len(inputs.indexed_evidence),
        "index_version_id": str(suite["index_version_id"]),
        "run_label": RUN_LABEL,
    }


def _difficulty_summary(difficulty: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_correct = sum(row["production_final_correct"] is True for row in rows)
    target = TARGETS[difficulty]
    minimum = target["minimum_correct"]
    shortfall = max(0, minimum - current_correct) if minimum is not None else None
    type_counts = Counter(row["case_type"] for row in rows)
    issue_counts = Counter(code for row in rows for code in row["issue_codes"])
    return {
        "case_count": len(rows),
        "answerable_count": sum(not row["unanswerable"] for row in rows),
        "unanswerable_count": sum(row["unanswerable"] for row in rows),
        "critical_count": sum(row["critical"] for row in rows),
        "unique_query_count": len({_compact(row["query"]) for row in rows}),
        "type_distribution": dict(sorted(type_counts.items())),
        "mean_query_chars": round(mean(len(row["query"]) for row in rows), 3),
        "median_query_chars": median(len(row["query"]) for row in rows),
        "mean_reference_chars": round(mean(len(row["reference_answer"]) for row in rows), 3),
        "median_reference_chars": median(len(row["reference_answer"]) for row in rows),
        "mean_evidence_count": round(mean(row["evidence_count"] for row in rows), 3),
        "production_correct": current_correct,
        "production_final_accuracy": current_correct / len(rows),
        "production_strict_pass": sum(row["production_strict_pass"] is True for row in rows),
        "target": target,
        "target_shortfall_cases": shortfall,
        "source_integrity_pass": sum(row["source_integrity_ok"] for row in rows),
        "archived_query_match": sum(row["archived_query_matches"] for row in rows),
        "archived_evaluator_drift": sum(row["archived_evaluator_drift"] for row in rows),
        "reference_self_pass": sum(row["reference_self_pass"] is True for row in rows),
        "expected_term_overlap_cases": issue_counts["EXPECTED_TERM_VARIANTS_OVERLAP"],
        "duplicate_weight_dominance_cases": issue_counts[
            "EXPECTED_TERM_DUPLICATE_WEIGHT_CAN_DOMINATE"
        ],
        "expected_terms_only_can_pass": issue_counts["EXPECTED_TERMS_ONLY_CAN_PASS"],
        "multi_branch_truncation_can_pass": issue_counts[
            "MULTI_BRANCH_TRUNCATION_CAN_PASS"
        ],
        "numeric_mutation_can_pass": issue_counts["NUMERIC_MUTATION_CAN_PASS"],
        "unsupported_false_addition_can_pass": issue_counts[
            "UNSUPPORTED_FALSE_ADDITION_CAN_PASS"
        ],
        "fabricated_fact_with_refusal_can_pass": issue_counts[
            "FABRICATED_FACT_WITH_REFUSAL_CAN_PASS"
        ],
        "actual_final_correct_low_faithfulness": issue_counts[
            "ACTUAL_FINAL_CORRECT_WITH_LOW_FAITHFULNESS"
        ],
        "actual_final_correct_low_citation": issue_counts[
            "ACTUAL_FINAL_CORRECT_WITH_LOW_CITATION_COVERAGE"
        ],
        "actual_pass_expected_terms_only": issue_counts[
            "ACTUAL_PASS_DEPENDS_ON_EXPECTED_TERMS"
        ],
        "actual_near_threshold_false_negative_candidates": issue_counts[
            "ACTUAL_NEAR_THRESHOLD_FALSE_NEGATIVE_CANDIDATE"
        ],
        "source_ambiguity_candidates": sum(
            "source_ambiguity" in row["qa_classifications"] for row in rows
        ),
        "dataset_reference_candidates": sum(
            "dataset_reference_candidate" in row["qa_classifications"] for row in rows
        ),
    }


def _finding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = {
        "UNSUPPORTED_FALSE_ADDITION_CAN_PASS": (
            "evaluator_error",
            "critical",
            "在完整参考答案后追加明确无源的虚假功能，final correctness 仍通过；最终正确率没有惩罚额外错误断言。",
        ),
        "FABRICATED_FACT_WITH_REFUSAL_CAN_PASS": (
            "evaluator_error",
            "critical",
            "unanswerable 只核对 abstention 标志；拒答文本同时给出伪造事实仍通过。",
        ),
        "MULTI_BRANCH_TRUNCATION_CAN_PASS": (
            "evaluator_error",
            "high",
            "多分支题删除参考答案末分支后仍达到 0.75 阈值，允许不完整最终答案计为正确。",
        ),
        "NUMERIC_MUTATION_CAN_PASS": (
            "evaluator_error",
            "high",
            "参考答案首个独立数值被改错后仍通过，说明召回阈值未形成关键数值全约束。",
        ),
        "EXPECTED_TERMS_ONLY_CAN_PASS": (
            "evaluator_error",
            "high",
            "仅拼接达到 75% 数量的 expected terms 即可通过，未要求形成完整、连贯且无矛盾的答案。",
        ),
        "EXPECTED_TERM_DUPLICATE_WEIGHT_CAN_DOMINATE": (
            "dataset_reference_candidate+evaluator_error",
            "high",
            "同一概念的重叠 expected-term 变体占到至少 75% 权重，可挤掉其他必答概念。",
        ),
        "ACTUAL_FINAL_CORRECT_WITH_LOW_FAITHFULNESS": (
            "evaluator_error",
            "high",
            "实际归档输出被记为答对，但项目自身 faithfulness 低于严格阈值。",
        ),
        "ACTUAL_FINAL_CORRECT_WITH_LOW_CITATION_COVERAGE": (
            "evaluator_error",
            "medium",
            "实际归档输出被记为答对，但项目自身 citation coverage 低于严格阈值。",
        ),
        "ACTUAL_PASS_DEPENDS_ON_EXPECTED_TERMS": (
            "evaluator_error",
            "high",
            "实际通过依赖 expected terms；reference semantic recall 本身不足 0.75。",
        ),
        "ACTUAL_NEAR_THRESHOLD_FALSE_NEGATIVE_CANDIDATE": (
            "evaluator_error_candidate",
            "medium",
            "实际答案保持参考约束且语义召回在 0.65–0.75，需盲审确认是否为阈值型误杀。",
        ),
        "EXPECTED_TERM_VARIANTS_OVERLAP": (
            "dataset_reference_candidate",
            "medium",
            "expected_terms 存在包含/高度相似变体，当前逐字符串平均会重复加权同一概念。",
        ),
        "HARD_EVIDENCE_WITHOUT_SCORING_CONCEPT": (
            "dataset_reference_candidate",
            "medium",
            "困难题至少一条绑定证据没有任何 expected/match concept，无法证明两条证据都被最终答案实质使用。",
        ),
        "QUERY_SINGULAR_REFERENCE_ENUMERATION": (
            "dataset_reference_candidate",
            "medium",
            "问题使用单数选择表达，参考答案却列出多个并列项，存在口径歧义候选。",
        ),
        "TABLE_QUERY_WITHOUT_PARSED_DATA_ROW": (
            "source_ambiguity",
            "high",
            "表格型问题的绑定证据没有可解析数据行，参考答案可能来自表头/备注推断而非明确单元格。",
        ),
        "PDF_TEXT_EXTRACTION_TERM_GAP": (
            "source_ambiguity",
            "medium",
            "索引证据中的 match term 未在声明 PDF 页的可提取文本中复现；可能是表格/图像提取差异，需页面视觉复核。",
        ),
        "REFERENCE_NUMBER_ABSENT_EVIDENCE": (
            "dataset_reference_candidate",
            "high",
            "参考答案引入绑定证据中不存在的数值。",
        ),
        "REFERENCE_IDENTIFIER_ABSENT_EVIDENCE": (
            "dataset_reference_candidate",
            "high",
            "参考答案引入绑定证据中不存在的字母数字标识符。",
        ),
        "QUERY_REFERENCE_POLARITY_CANDIDATE": (
            "dataset_reference_candidate",
            "high",
            "问题包含否定约束，但参考答案未保留可识别的否定极性。",
        ),
    }
    values: list[dict[str, Any]] = []
    for code, (category, severity, description) in definitions.items():
        members = [row for row in rows if code in row["issue_codes"]]
        if not members:
            continue
        by_difficulty = {
            difficulty: sum(row["difficulty"] == difficulty for row in members)
            for difficulty in DIFFICULTIES
        }
        values.append(
            {
                "issue_code": code,
                "category": category,
                "severity": severity,
                "case_count": len(members),
                "case_rate": len(members) / len(rows),
                "easy": by_difficulty["easy"],
                "medium": by_difficulty["medium"],
                "hard": by_difficulty["hard"],
                "description": description,
                "examples": [row["case_id"] for row in members[:8]],
            }
        )
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(
        values,
        key=lambda item: (
            severity_order[item["severity"]],
            -item["case_count"],
            item["issue_code"],
        ),
    )


def build_audit(inputs: Inputs | None = None) -> dict[str, Any]:
    inputs = inputs or load_inputs()
    checks = _input_checks(inputs)
    dataset_ids = [
        str(case["case_id"])
        for difficulty in DIFFICULTIES
        for case in inputs.manifests[difficulty]["cases"]
    ]
    if not (
        checks["dataset_case_count"]
        == checks["dataset_case_id_unique_count"]
        == checks["result_case_count"]
        == checks["result_case_id_unique_count"]
        == checks["joined_case_count"]
        == 150
    ):
        raise ValueError(f"case-grain 150/150 join failed: {checks}")
    if checks["dataset_without_result"] or checks["result_without_dataset"]:
        raise ValueError("dataset/result join is incomplete")
    if not checks["manifest_hashes_match_suite"]:
        raise ValueError("canonical manifest hashes do not match suite.json")
    if not checks["manifest_hashes_match_archived_run"]:
        raise ValueError("archived run dataset hashes do not match current immutable manifests")
    if not checks["source_hash_matches"] or not checks["source_page_count_matches"]:
        raise ValueError("source PDF identity/page count mismatch")

    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        case_map = {
            str(case["case_id"]): case for case in inputs.manifests[difficulty]["cases"]
        }
        for case_id in sorted(case_map):
            rows.append(
                _case_row(
                    difficulty=difficulty,
                    case=case_map[case_id],
                    result=inputs.results[difficulty][case_id],
                    inputs=inputs,
                )
            )
    if len(rows) != len(dataset_ids) or {row["case_id"] for row in rows} != set(dataset_ids):
        raise ValueError("case row coverage mismatch")
    if sum(row["source_integrity_ok"] for row in rows) != 150:
        raise ValueError("one or more cases failed indexed evidence integrity")
    if sum(row["archived_query_matches"] for row in rows) != 150:
        raise ValueError("one or more archived result queries do not match the manifest")

    difficulty_summary = {
        difficulty: _difficulty_summary(
            difficulty, [row for row in rows if row["difficulty"] == difficulty]
        )
        for difficulty in DIFFICULTIES
    }
    findings = _finding_rows(rows)
    issue_counts = Counter(code for row in rows for code in row["issue_codes"])
    classification_counts = Counter(
        category for row in rows for category in row["qa_classifications"]
    )

    query_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            score = _jaccard(_ngrams(left["query"]), _ngrams(right["query"]))
            if score >= 0.88:
                query_pairs.append(
                    {
                        "left": left["case_id"],
                        "right": right["case_id"],
                        "trigram_jaccard": round(score, 6),
                    }
                )

    blocking = {
        "status": "blocked_for_new_baseline_registration",
        "reasons": [
            "easy and medium stored production accuracy remain below their strict targets",
            "final correctness admits explicit unsupported false additions for answerable cases",
            "unanswerable correctness admits a refusal that also states a fabricated fact",
            "multi-branch and numeric counterfactuals expose completeness/critical-constraint blind spots",
            "expected-term strings lack concept-level all-of/any-of semantics and can duplicate-weight one concept",
        ],
        "not_claimed": [
            "counterfactual susceptibility does not prove every current production pass is factually wrong",
            "PDF text-extraction gaps are source-ambiguity candidates, not confirmed source errors",
            "near-threshold actual failures require an independent blind semantic review before relabeling",
        ],
    }
    return {
        "schema_version": "stt.eval.suite-source-evaluator-qa.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "suite_key": str(inputs.suite["suite_key"]),
        "run_label": RUN_LABEL,
        "intended_grain": "case_id",
        "input_checks": checks,
        "difficulty_summary": difficulty_summary,
        "difficulty_ordering_checks": {
            "mean_query_chars_strictly_increases": (
                difficulty_summary["easy"]["mean_query_chars"]
                < difficulty_summary["medium"]["mean_query_chars"]
                < difficulty_summary["hard"]["mean_query_chars"]
            ),
            "mean_reference_chars_easy_below_medium": (
                difficulty_summary["easy"]["mean_reference_chars"]
                < difficulty_summary["medium"]["mean_reference_chars"]
            ),
            "evidence_cardinality_contract": {
                "easy": sorted({row["evidence_count"] for row in rows if row["difficulty"] == "easy"}),
                "medium": sorted({row["evidence_count"] for row in rows if row["difficulty"] == "medium"}),
                "hard_answerable": sorted(
                    {
                        row["evidence_count"]
                        for row in rows
                        if row["difficulty"] == "hard" and not row["unanswerable"]
                    }
                ),
                "hard_unanswerable": sorted(
                    {
                        row["evidence_count"]
                        for row in rows
                        if row["difficulty"] == "hard" and row["unanswerable"]
                    }
                ),
            },
        },
        "query_near_duplicates": query_pairs,
        "issue_counts": dict(sorted(issue_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "findings": findings,
        "blocking_conclusion": blocking,
        "cases": rows,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_cases_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _report_markdown(audit: dict[str, Any]) -> str:
    checks = audit["input_checks"]
    by_difficulty = audit["difficulty_summary"]
    findings = audit["findings"]
    confirmed_source_candidates = sum(
        row["source_ambiguity_candidates"] for row in by_difficulty.values()
    )
    table_lines = [
        "| 难度 | 题数 | 当前答对 | 最终正确率 | 目标 | 差额 | 严格通过 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    for difficulty in DIFFICULTIES:
        item = by_difficulty[difficulty]
        target = item["target"]
        target_text = (
            f">{target['threshold']:.0%}（至少 {target['minimum_correct']}/50）"
            if target["threshold"] is not None
            else "无强制目标"
        )
        shortfall = item["target_shortfall_cases"]
        table_lines.append(
            f"| {labels[difficulty]} | {item['case_count']} | {item['production_correct']} | "
            f"{item['production_final_accuracy']:.1%} | {target_text} | "
            f"{shortfall if shortfall is not None else 'N/A'} | {item['production_strict_pass']} |"
        )
    finding_lines = [
        "| 严重度 | 类别 | 规则 | 覆盖 | easy / medium / hard |",
        "|---|---|---|---:|---:|",
    ]
    for item in findings:
        finding_lines.append(
            f"| {item['severity']} | {item['category']} | `{item['issue_code']}` | "
            f"{item['case_count']}/150 | {item['easy']} / {item['medium']} / {item['hard']} |"
        )

    return "\n".join(
        [
            "# 固定 150 题：数据集、源证据与评测器一致性 QA",
            "",
            "## 技术摘要",
            "",
            "**结论：当前套件不能作为新的达标基线登记。** 这不是因为 150 题连接或源索引损坏："
            f"manifest→归档结果为 {checks['joined_case_count']}/150 一对一完整连接，case_id 唯一，"
            "三份 canonical manifest hash 与 suite.json、归档运行一致；源 PDF SHA-256、95 页页数及"
            " 145 个可回答题绑定的索引证据均通过完整性核对。阻塞来自两层：当前 easy=40/50、"
            "medium=35/50，分别距离严格目标 8 题；同时 final correctness 的确定性合同存在可复现"
            "的通用盲区，不能把当前二元分数直接当作最终事实正确性的充分证明。",
            "",
            "最严重的反例不依赖题号：对全部可回答题，在完整参考答案后追加明确无源的虚假功能，"
            "final correctness 仍可能通过；对全部拒答题，`abstained=true` 的文本即使同时给出伪造"
            "价格也通过。多分支删尾、关键数值改写与 expected-terms-only 反例进一步表明，0.75 召回"
            "阈值不能表达“所有必答分支正确且没有错误附加断言”。因此后续修复必须版本化调整全套"
            "答案合同/评测合同，而不能改几道失败题。",
            "",
            "## 当前结果与目标差额",
            "",
            *table_lines,
            "",
            "## 全量 QA 覆盖与可信输入",
            "",
            f"- intended grain：`case_id`；dataset={checks['dataset_case_count']}、"
            f"result={checks['result_case_count']}、join={checks['joined_case_count']}，"
            "未连接集合均为空。",
            f"- 唯一性：dataset unique={checks['dataset_case_id_unique_count']}；"
            f"result unique={checks['result_case_id_unique_count']}；近重复问题对="
            f"{len(audit['query_near_duplicates'])}（三元组 Jaccard ≥0.88）。",
            "- 哈希：三份 canonical manifest 与 suite/归档 run manifest 全部一致；数据集文件本身的"
            " SHA-256 另行记录，避免混淆 canonical JSON hash 与文件字节 hash。",
            f"- 源：`{checks['source_path']}` SHA-256 一致，PDF 页数="
            f"{checks['source_pdf_page_count']}，当前索引版本 `{checks['index_version_id']}` 含"
            f" {checks['indexed_evidence_count']} 条 evidence；逐题 evidence id/chunk/hash/page 连接"
            " 150/150 通过。",
            "- 题型/难度：easy 与 medium 均为 50 个可回答题、单证据；hard 为 45 个双证据综合题"
            "加 5 个知识边界拒答题。问题平均长度按 easy→medium→hard 递增；这只能证明结构分层，"
            "不等价于语义难度已完全校准。",
            "",
            "## 阻塞性发现按类别分开",
            "",
            *finding_lines,
            "",
            "### 1. Evaluator error：二元“答对”没有覆盖最终答案的全部事实",
            "",
            "当前 answerable 合同取 `max(expected_term_recall, reference_semantic_recall)`，达到 0.75"
            " 且未命中少量手写 forbidden terms 即判 final correct；faithfulness 与 citation 只影响严格"
            "有据通过率，不影响目标用的 final accuracy。该结构会接受完整参考答案后的虚假附加断言；"
            "unanswerable 则只看拒答判断，不审计同一文本中是否夹带事实。它直接违背“任何指标均服务"
            "于最终结果正确”的目标，因此是确认的评测器阻塞，而非模型性能问题。",
            "",
            "### 2. Dataset/reference candidate：expected terms 没有概念组语义",
            "",
            "全套 `expected_terms` 是扁平字符串数组，评测器把同义/包含变体当成独立等权必答项。"
            "重叠变体会重复加权同一概念；当一个概念的变体占到 75% 时，只回答该概念便可越过门槛，"
            "即使遗漏其他必答概念。困难题还存在绑定证据没有任何 match/scoring concept 的系统性候选，"
            "这使“两条证据均被实质使用”无法由现有标签验证。",
            "",
            "### 3. Source ambiguity：候选必须视觉复核，不能自动当作源错误",
            "",
            f"共有 {confirmed_source_candidates}/150 题触发至少一个 source-ambiguity 候选，主要是"
            " PDF 可提取文本与索引化表格/图像证据的词项差异，或表格问题没有可解析数据行。PDF"
            "提取失败可能只是布局/OCR 限制，不能据此直接删题或改答案。任何变更都应先按同一规则"
            "对全 150 题做页面级视觉复核并发布 suite v2，而不是只修改当前错题。",
            "",
            "## 方法与稳健性",
            "",
            "脚本先锁定 manifest 题粒度与归档 run hash，再对每题连接当前索引 evidence 和声明 PDF 页；"
            "检查 query/reference/expected/forbidden、数字与字母数字标识符、否定极性、表格/多分支/"
            "双证据标签。随后用项目当前原生 evaluator 对归档答案重算，要求与存储 correctness 完全一致，"
            "并运行四类不依赖题号的反事实：删末分支、首数字改值、仅拼接 75% expected terms、追加"
            "明确虚假断言；拒答题另测“拒答+伪造事实”。所有计数均以 150 为固定分母或注明适用子集。",
            "",
            "反事实用于证明评测合同的可接受集合过宽，不声称每个当前通过答案都实际包含错误；"
            "`ACTUAL_NEAR_THRESHOLD_FALSE_NEGATIVE_CANDIDATE` 也只是候选，必须结合独立盲审才能判定"
            "评测误杀。PDF 文本核对只覆盖机器可提取层，页面视觉信息属于保留限制。报告未绘制趋势图，"
            "因为这是固定快照的一次性 case-grain 审计，精确覆盖表比趋势/分布图更能支持修复决策。",
            "",
            "## 全套、版本化修复建议",
            "",
            "1. 发布 evaluator v3：最终正确性要求“所有 query-derived 必答分支/关键数值/标识符均满足”"
            "且“无不受证据支持或相互矛盾的事实”；answerable 与 unanswerable 使用同一事实完整性检查。"
            "final accuracy 仍为目标指标，但其判定必须吸收 faithfulness/contradiction，而不是只调权重。",
            "2. 发布 dataset suite v2：把扁平 `expected_terms` 迁移为 concept groups，例如 `all_of`"
            " 必答概念与每概念内 `any_of` 表达变体；所有 150 题走同一迁移、源证据复核和双人/盲审，"
            "新 case/version/hash 与旧版并存，禁止原地修改 manifest。",
            "3. 对困难题建立 evidence→concept 覆盖矩阵，要求每条绑定 evidence 至少支撑一个独立必答"
            "概念；无法满足者按全套规则降级难度、重写或标记 source ambiguity，不按运行成败挑题。",
            "4. 先用归档 observation 做 evaluator-v3 shadow replay，并同时报告旧/新二元结果与独立语义"
            "盲审冲突矩阵；只有 easy≥48、medium≥43 且确认无系统性假阳性后，才登记新 baseline。",
            "",
            "## 限制、未决问题与登记决定",
            "",
            "旧的 qwen-flash 150/150 作者质量审计通过，只证明该提示词下的 grounded/complete 判断没有"
            "报错，不能覆盖本次反事实发现。当前 QA 没有把 PDF 图像/表格逐页人工视觉签核，也没有把"
            "近阈值失败擅自改判为正确。结论强度因此是：输入完整性 ready；当前 final-accuracy 解释"
            " needs revision；新 baseline 登记 blocked。后续关键问题是 evaluator v3 与独立盲审的一致率、"
            "concept-group 迁移后旧结果的重算变化，以及页面视觉复核后真正的 source ambiguity 数量。",
            "",
            "完整 150 行判据与答案文本见 `cases.json` / `cases.csv`；聚合见 `summary.json`，可复算入口"
            "为 `audit_suite.py` 与已执行的 `suite_source_evaluator_qa.ipynb`。",
            "",
        ]
    )


def _artifact(audit: dict[str, Any]) -> dict[str, Any]:
    by_difficulty = audit["difficulty_summary"]
    findings = audit["findings"]
    coverage_rows = []
    labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    for order, difficulty in enumerate(DIFFICULTIES, start=1):
        item = by_difficulty[difficulty]
        coverage_rows.append(
            {
                "order": order,
                "difficulty": labels[difficulty],
                "cases": item["case_count"],
                "correct": item["production_correct"],
                "accuracy": item["production_final_accuracy"],
                "minimum_correct": item["target"]["minimum_correct"],
                "shortfall": item["target_shortfall_cases"],
                "strict_pass": item["production_strict_pass"],
            }
        )
    finding_rows = [
        {
            "severity_order": {"critical": 1, "high": 2, "medium": 3, "low": 4}[
                item["severity"]
            ],
            "severity": item["severity"],
            "category": item["category"],
            "issue": item["issue_code"],
            "cases": item["case_count"],
            "rate": item["case_rate"],
            "easy": item["easy"],
            "medium": item["medium"],
            "hard": item["hard"],
        }
        for item in findings
    ]
    source = {
        "id": "suite_qa",
        "label": "Frozen suite + max-mono archive + indexed source evidence",
        "path": "docs/evaluation-baselines/2026-08-10-three-difficulty/remediation-2026-08-10/suite-source-evaluator-qa/summary.json",
        "query": {
            "description": "The deterministic Python audit performs a 150/150 case-id join, verifies hashes and indexed evidence, recomputes the native evaluator, and applies suite-wide counterfactual rules without case-id branches.",
            "engine": "Python 3 over immutable JSON/JSONL, read-only SQLite evidence, and source PDF",
            "language": "python",
            "filters": [
                "suite_key = three-difficulty-live-2026-08-10",
                "run_label = live-cand-max-mono-01",
                "grain = case_id",
                "150 manifest cases joined 1:1 to 150 archived results",
            ],
            "metric_definitions": [
                "Final accuracy: stored metrics.case_outcome.final_answer_accuracy.",
                "Target: easy >95% means at least 48/50; medium >85% means at least 43/50.",
                "Counterfactual pass: the current native evaluator still returns final_correct after a deterministic answer mutation.",
                "Source ambiguity flags are candidates until PDF page visual review; they are not automatic source errors.",
            ],
            "tables_used": [
                "datasets/easy.json",
                "datasets/medium.json",
                "datasets/hard.json",
                "runs/live-cand-max-mono-01/artifacts/*/*/results.jsonl",
                "volumes/control/stt-control.db:evidence_records",
                "doc/PantumP3500用户手册zh_CNV1.2_1644316283788.pdf",
            ],
        },
    }
    checks = audit["input_checks"]
    total_false_addition = sum(
        item["unsupported_false_addition_can_pass"] for item in by_difficulty.values()
    )
    total_truncation = sum(
        item["multi_branch_truncation_can_pass"] for item in by_difficulty.values()
    )
    report_sections = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# 固定 150 题：数据集、源证据与评测器一致性 QA",
        },
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": "## 技术摘要\n\n**新基线登记应保持 blocked。** 150/150 case_id 连接、manifest/run hash、源 PDF 和索引证据完整性均通过；但 easy=40/50、medium=35/50，各距严格目标 8 题。更关键的是，当前 final correctness 会接受完整参考答案后的明确无源虚假断言，也会接受夹带伪造事实的拒答，因此二元正确率还不是最终事实正确性的充分度量。",
        },
        {
            "id": "headline",
            "type": "metric-strip",
            "cardIds": ["join", "easy_accuracy", "medium_accuracy", "false_addition"],
        },
        {
            "id": "targets_text",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": "## 当前结果离目标各差 8 题\n\n简单和中等的 stored production binary 分别是 40/50 与 35/50；目标严格使用 `>`，所以最低门槛是 48/50 与 43/50。困难集无强制准确率要求，但仍纳入相同的数据/源/评测合同 QA。",
        },
        {"id": "targets_table_block", "type": "table", "tableId": "targets", "layout": "full"},
        {
            "id": "findings_text",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": f"## 阻塞来自评测合同而非输入连接\n\n反事实规则不使用题号：明确虚假附加断言可通过 {total_false_addition} 个可回答题；多分支删尾仍可通过 {total_truncation} 个题。扁平 expected terms 还会重复加权同一概念。下面按 evaluator error、dataset/reference candidate 与 source ambiguity 分列覆盖量；source ambiguity 仅是待页面视觉复核候选。",
        },
        {"id": "findings_table_block", "type": "table", "tableId": "findings", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": f"## 范围、数据与定义\n\n粒度为一行一个 immutable `case_id`：dataset={checks['dataset_case_count']}、archive={checks['result_case_count']}、join={checks['joined_case_count']}，两侧孤儿均为 0。三份 canonical manifest hash 与 suite 和归档 run manifest 一致；源 PDF SHA-256 与 95 页页数一致；当前 index `{checks['index_version_id']}` 的 evidence id/chunk/hash/page 逐题连接通过。",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": "## 方法与可复现性\n\n脚本全量核对唯一性、难度结构、reference/expected/forbidden、否定极性、标识符/数值、表格/多分支与 evidence→PDF/索引一致性；再用项目原生 evaluator 重算归档答案，并运行删末分支、改首数值、expected-only、虚假附加断言与拒答夹带事实反例。完整 150 行在 cases.csv/json；companion notebook 只调用同一模块。",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": "## 限制与稳健性\n\n反事实证明评测器可接受集合过宽，不证明每个当前通过答案实际错误。PDF 可提取文本无法完全覆盖图像与复杂表格，相关 gap 只标 source ambiguity candidate。near-threshold false-negative candidate 必须经独立盲审后才能改判。固定快照没有趋势维度，因此使用精确表而不画趋势图。",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "suite_qa",
            "body": "## 全套、版本化修复建议\n\n1. evaluator v3 将所有 query-derived 必答分支、关键数值/标识符和无矛盾有据性纳入 final-correct 判定；拒答也检查夹带事实。\n2. suite v2 把 expected terms 迁移为 `all_of` concept groups，每组内部 `any_of` 表达变体；全 150 题统一迁移、双审和换 hash，禁止原地改单题。\n3. 困难题建立 evidence→concept 覆盖矩阵；每条证据至少支持一个独立必答概念。\n4. 用归档 observation shadow replay evaluator v3，并与独立盲审做冲突矩阵；仅在 easy≥48、medium≥43 且假阳性风险解除后登记新 baseline。",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": "## 未决问题\n\nevaluator v3 与盲审的一致率是多少？concept-group 迁移会让多少旧结果换判？页面视觉复核后真正的 source ambiguity 还有多少？在回答这些问题前，当前输入完整性可以信任，但 final-accuracy 解释需要修订。",
        },
    ]
    cards = [
        {
            "id": "join",
            "dataset": "headline",
            "description": "manifest 与归档结果一对一",
            "sourceId": "suite_qa",
            "metrics": [{"field": "join_rate", "label": "Case join", "format": "percent"}],
        },
        {
            "id": "easy_accuracy",
            "dataset": "headline",
            "description": "目标 >95%，至少 48/50",
            "sourceId": "suite_qa",
            "metrics": [{"field": "easy_accuracy", "label": "Easy final", "format": "percent"}],
        },
        {
            "id": "medium_accuracy",
            "dataset": "headline",
            "description": "目标 >85%，至少 43/50",
            "sourceId": "suite_qa",
            "metrics": [{"field": "medium_accuracy", "label": "Medium final", "format": "percent"}],
        },
        {
            "id": "false_addition",
            "dataset": "headline",
            "description": "可回答题追加明确虚假断言仍通过",
            "sourceId": "suite_qa",
            "metrics": [{"field": "false_addition_rate", "label": "Evaluator blind spot", "format": "percent"}],
        },
    ]
    tables = [
        {
            "id": "targets",
            "title": "三档当前正确率与门槛",
            "subtitle": "每档固定 50 题；hard 无强制准确率目标。",
            "dataset": "coverage",
            "density": "spacious",
            "layout": "full",
            "sourceId": "suite_qa",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "顺序", "type": "number"},
                {"field": "difficulty", "label": "难度", "type": "text"},
                {"field": "cases", "label": "题数", "type": "number"},
                {"field": "correct", "label": "答对", "type": "number"},
                {"field": "accuracy", "label": "最终正确率", "format": "percent"},
                {"field": "minimum_correct", "label": "最低答对", "type": "number"},
                {"field": "shortfall", "label": "差额", "type": "number"},
                {"field": "strict_pass", "label": "严格通过", "type": "number"},
            ],
        },
        {
            "id": "findings",
            "title": "阻塞性规则覆盖",
            "subtitle": "全 150 题统一规则；source ambiguity 是候选，非自动定错。",
            "dataset": "findings",
            "density": "dense",
            "layout": "full",
            "sourceId": "suite_qa",
            "defaultSort": {"field": "severity_order", "direction": "asc"},
            "columns": [
                {"field": "severity_order", "label": "级别序", "type": "number"},
                {"field": "severity", "label": "严重度", "type": "text"},
                {"field": "category", "label": "类别", "type": "text"},
                {"field": "issue", "label": "规则", "type": "text"},
                {"field": "cases", "label": "覆盖题数", "type": "number"},
                {"field": "rate", "label": "覆盖率", "format": "percent"},
                {"field": "easy", "label": "Easy", "type": "number"},
                {"field": "medium", "label": "Medium", "type": "number"},
                {"field": "hard", "label": "Hard", "type": "number"},
            ],
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "surface": "report",
            "version": "1.0",
            "title": "固定 150 题：数据集、源证据与评测器一致性 QA",
            "description": "Case-grain QA of dataset references, indexed/PDF evidence, archived production outcomes, and evaluator behavior.",
            "generatedAt": audit["generated_at"],
            "blocks": report_sections,
            "cards": cards,
            "charts": [],
            "tables": tables,
            "sources": [source],
        },
        "snapshot": {
            "version": "1.0",
            "status": "ready",
            "generatedAt": audit["generated_at"],
            "accessIssues": [],
            "datasets": {
                "headline": [
                    {
                        "join_rate": checks["joined_case_count"] / checks["expected_case_count"],
                        "easy_accuracy": by_difficulty["easy"]["production_final_accuracy"],
                        "medium_accuracy": by_difficulty["medium"]["production_final_accuracy"],
                        "false_addition_rate": total_false_addition
                        / sum(item["answerable_count"] for item in by_difficulty.values()),
                    }
                ],
                "coverage": coverage_rows,
                "findings": finding_rows,
            },
        },
        "sources": [source],
    }


def write_outputs(audit: dict[str, Any]) -> None:
    QA_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in audit.items() if key != "cases"}
    (QA_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (QA_ROOT / "cases.json").write_text(
        json.dumps(audit["cases"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_cases_csv(audit["cases"], QA_ROOT / "cases.csv")
    (QA_ROOT / "findings.json").write_text(
        json.dumps(audit["findings"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (QA_ROOT / "REPORT.md").write_text(_report_markdown(audit), encoding="utf-8")
    (QA_ROOT / "artifact.json").write_text(
        json.dumps(_artifact(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def finalize_checksums(extra_paths: Iterable[Path] = ()) -> None:
    names = [
        "audit_suite.py",
        "summary.json",
        "cases.json",
        "cases.csv",
        "findings.json",
        "REPORT.md",
        "artifact.json",
    ]
    for path in extra_paths:
        if path.parent == QA_ROOT and path.name not in names:
            names.append(path.name)
    lines = []
    for name in sorted(names):
        path = QA_ROOT / name
        if path.is_file():
            lines.append(f"{_sha256_file(path)}  {name}")
    (QA_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the full audit and print the summary without writing outputs",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    audit = build_audit()
    if not args.check_only:
        write_outputs(audit)
        finalize_checksums()
    compact = {
        "joined": audit["input_checks"]["joined_case_count"],
        "difficulty_summary": audit["difficulty_summary"],
        "classification_counts": audit["classification_counts"],
        "blocking": audit["blocking_conclusion"]["status"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
