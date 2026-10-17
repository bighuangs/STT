"""Validate the three-difficulty evaluation suite before it is imported.

The deterministic pass checks the immutable manifest contract and joins every
answerable case back to the currently published evidence rows.  The optional
online pass asks the configured LLM to independently review grounding,
completeness, ambiguity, and difficulty fit.  All audit material is written
under this result directory and no credential is logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = RESULT_ROOT / "datasets"
QUALITY_ROOT = RESULT_ROOT / "quality"
RAW_ROOT = QUALITY_ROOT / "raw"
CONTROL_DB = PROJECT_ROOT / "volumes" / "control" / "stt-control.db"
SUITE_KEY = "three-difficulty-live-2026-08-10"
DIFFICULTIES = ("easy", "medium", "hard")
MINIMUM_CASE_COUNT = 50


AUDIT_SYSTEM_PROMPT = """你是企业 RAG 测评集的独立质量审计员。只依据输入中的问题、参考答案和冻结证据判断，不补充外部知识。
逐题检查：
1. grounded：参考答案的每个事实、数值、步骤和因果/比较关系均能由证据直接支持；
2. complete：参考答案覆盖问题的每个分支；
3. unambiguous：结合证据后问题指向明确，不存在多个相互冲突的正确答案；
4. difficulty_fit：easy 为单个直接事实；medium 为同一证据中的多个约束/步骤/值；hard 为两条证据的实质综合，或明确要求手册之外实时/第三方信息的拒答题；
5. uses_all_evidence：hard 可回答题的答案必须实质使用每条绑定证据，不只是把无关事实硬拼在一起。其他题填 null。
unanswerable=true 时没有正向证据是设计行为：只检查问题是否明确要求手册外信息，reference_answer 是否给出清晰拒答；不要因为缺少正向证据判失败。
pass 只有在所有适用项均通过时才为 true。severity 只能是 none/low/medium/high。输出严格 JSON，不要 Markdown。"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_file_checksums() -> dict[str, str]:
    return {
        difficulty: _digest_file(DATASET_ROOT / f"{difficulty}.json")
        for difficulty in DIFFICULTIES
    }


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).lower())


def _ngrams(value: str, size: int = 3) -> set[str]:
    compact = _compact(value)
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    check: str,
    message: str,
    *,
    difficulty: str | None = None,
    case_id: str | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "difficulty": difficulty,
            "case_id": case_id,
            "message": message,
        }
    )


def _load_evidence(index_version_id: str) -> tuple[dict[str, dict[str, Any]], str]:
    connection = sqlite3.connect(f"file:{CONTROL_DB.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT er.id, er.chunk_id, er.document_version_id, er.primary_page_no,
                   er.content, er.content_hash, er.source_label
            FROM evidence_records er
            WHERE er.index_version_id = ?
            ORDER BY er.id
            """,
            (index_version_id,),
        ).fetchall()
        active = connection.execute(
            """
            SELECT iv.id, iv.kb_id, iv.status, iv.active_slot,
                   COUNT(er.id) AS evidence_count
            FROM index_versions iv
            LEFT JOIN evidence_records er ON er.index_version_id = iv.id
            WHERE iv.id = ?
            GROUP BY iv.id
            """,
            (index_version_id,),
        ).fetchone()
    finally:
        connection.close()
    if active is None:
        raise RuntimeError(f"frozen index does not exist: {index_version_id}")
    if active["status"] != "active" or active["active_slot"] != "active":
        raise RuntimeError(f"frozen index is no longer active: {dict(active)}")
    values = {str(row["id"]): dict(row) for row in rows}
    corpus = "\n".join(str(row["content"] or "") for row in rows)
    return values, corpus


def deterministic_validation() -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.domain.evaluation import validate_dataset_manifest

    suite_path = RESULT_ROOT / "suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    all_case_ids: list[tuple[str, str]] = []
    all_queries: list[tuple[str, str, str]] = []

    if suite.get("suite_key") != SUITE_KEY:
        _issue(issues, "error", "suite_identity", "suite.json has an unexpected suite_key")
    if suite.get("evaluation_mode") != "live":
        _issue(issues, "error", "suite_mode", "suite evaluation_mode must be live")

    source_sha = str(suite.get("source_sha256") or "").lower()
    source_path: Path | None = None
    index_version_id = str(suite.get("index_version_id") or "")
    evidence_by_id, corpus_text = _load_evidence(index_version_id)

    for difficulty in DIFFICULTIES:
        path = DATASET_ROOT / f"{difficulty}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests[difficulty] = manifest
        try:
            validate_dataset_manifest(manifest)
        except Exception as exc:  # noqa: BLE001 - audit must retain the exact contract failure
            _issue(issues, "error", "manifest_contract", str(exc), difficulty=difficulty)

        if manifest.get("suite_key") != SUITE_KEY:
            _issue(issues, "error", "suite_identity", "manifest suite_key mismatch", difficulty=difficulty)
        if manifest.get("difficulty") != difficulty:
            _issue(issues, "error", "difficulty", "manifest difficulty mismatch", difficulty=difficulty)
        if manifest.get("evaluation_mode_default") != "live":
            _issue(issues, "error", "mode", "evaluation_mode_default must be live", difficulty=difficulty)
        if manifest.get("minimum_case_count") != MINIMUM_CASE_COUNT:
            _issue(issues, "error", "case_count", "minimum_case_count must be 50", difficulty=difficulty)
        cases = list(manifest.get("cases") or [])
        if len(cases) != MINIMUM_CASE_COUNT:
            _issue(issues, "error", "case_count", f"expected 50 cases, found {len(cases)}", difficulty=difficulty)

        suite_entry = dict((suite.get("datasets") or {}).get(difficulty) or {})
        if suite_entry.get("checksum") != _digest_json(manifest):
            _issue(issues, "error", "manifest_checksum", "suite checksum does not match canonical manifest", difficulty=difficulty)
        if suite_entry.get("case_count") != len(cases):
            _issue(issues, "error", "suite_case_count", "suite case_count does not match manifest", difficulty=difficulty)

        source = dict(manifest.get("source") or {})
        manifest_source_path = (PROJECT_ROOT / str(source.get("relative_path") or "")).resolve()
        if source_path is None:
            source_path = manifest_source_path
        if manifest_source_path != source_path:
            _issue(issues, "error", "source_identity", "datasets do not share one source path", difficulty=difficulty)
        if not manifest_source_path.is_file():
            _issue(issues, "error", "source_file", "source file is missing", difficulty=difficulty)
        else:
            actual_sha = _digest_file(manifest_source_path)
            if actual_sha.lower() != str(source.get("content_sha256") or "").lower():
                _issue(issues, "error", "source_hash", "manifest source SHA-256 mismatch", difficulty=difficulty)
            if actual_sha.lower() != source_sha:
                _issue(issues, "error", "suite_source_hash", "suite source SHA-256 mismatch", difficulty=difficulty)

        evidence_ids: list[str] = []
        pages: list[int] = []
        expected_counts: list[int] = []
        answerable = 0
        unanswerable = 0
        critical = 0
        for case in cases:
            case_id = str(case.get("case_id") or "")
            all_case_ids.append((difficulty, case_id))
            query = str(case.get("query") or "")
            all_queries.append((difficulty, case_id, query))
            metadata = dict(case.get("metadata") or {})
            tags = [str(value) for value in case.get("tags") or []]
            if metadata.get("difficulty") != difficulty or metadata.get("suite_key") != SUITE_KEY:
                _issue(issues, "error", "case_metadata", "case difficulty/suite metadata mismatch", difficulty=difficulty, case_id=case_id)
            if f"difficulty-{difficulty}" not in tags:
                _issue(issues, "error", "case_tags", "case is missing its difficulty tag", difficulty=difficulty, case_id=case_id)
            if metadata.get("critical") is True:
                critical += 1

            is_unanswerable = bool(case.get("unanswerable"))
            relevant = list(case.get("relevant_evidence") or [])
            chunk_ids = [str(value) for value in case.get("relevant_chunk_ids") or []]
            terms = [str(value).strip() for value in case.get("expected_terms") or [] if str(value).strip()]
            reference = str(case.get("reference_answer") or "")
            expected_counts.append(len(terms))
            if is_unanswerable:
                unanswerable += 1
                if difficulty != "hard":
                    _issue(issues, "error", "unanswerable_distribution", "only hard may contain unanswerable cases", difficulty=difficulty, case_id=case_id)
                if relevant or chunk_ids:
                    _issue(issues, "error", "unanswerable_contract", "unanswerable case must not carry positive evidence", difficulty=difficulty, case_id=case_id)
                if not reference.strip() or not terms:
                    _issue(issues, "error", "unanswerable_contract", "unanswerable case needs a refusal reference and expected refusal terms", difficulty=difficulty, case_id=case_id)
                for term in terms:
                    if _compact(term) not in _compact(reference):
                        _issue(issues, "error", "expected_terms", f"refusal term is absent from reference_answer: {term}", difficulty=difficulty, case_id=case_id)
                forbidden = [str(value) for value in case.get("forbidden_terms") or [] if str(value).strip()]
                present = [term for term in forbidden if _compact(term) and _compact(term) in _compact(corpus_text)]
                if present:
                    _issue(issues, "error", "unanswerable_boundary", f"forbidden terms unexpectedly occur in the frozen corpus: {present}", difficulty=difficulty, case_id=case_id)
                continue

            answerable += 1
            required_evidence = 2 if difficulty == "hard" else 1
            if len(relevant) != required_evidence or len(chunk_ids) != required_evidence:
                _issue(issues, "error", "evidence_cardinality", f"expected {required_evidence} evidence targets", difficulty=difficulty, case_id=case_id)
            if not terms:
                _issue(issues, "error", "expected_terms", "answerable case has no expected terms", difficulty=difficulty, case_id=case_id)
            joined_evidence = "\n".join(str(item.get("text") or "") for item in relevant)
            for term in terms:
                if _compact(term) not in _compact(reference):
                    _issue(issues, "error", "expected_terms", f"term is absent from reference_answer: {term}", difficulty=difficulty, case_id=case_id)
                if _compact(term) not in _compact(joined_evidence):
                    _issue(issues, "error", "expected_terms", f"term is absent from bound evidence: {term}", difficulty=difficulty, case_id=case_id)
            reference_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", reference))
            evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", joined_evidence))
            if not reference_numbers.issubset(evidence_numbers):
                _issue(issues, "error", "numeric_grounding", f"reference introduces numbers absent from evidence: {sorted(reference_numbers - evidence_numbers)}", difficulty=difficulty, case_id=case_id)

            for position, target in enumerate(relevant):
                evidence_id = str(target.get("evidence_id") or "")
                evidence_ids.append(evidence_id)
                stored = evidence_by_id.get(evidence_id)
                if stored is None:
                    _issue(issues, "error", "evidence_integrity", f"evidence does not exist in frozen index: {evidence_id}", difficulty=difficulty, case_id=case_id)
                    continue
                pages.append(int(target.get("page") or 0))
                declared_chunk = str(target.get("chunk_id") or "")
                if declared_chunk != str(stored["chunk_id"]) or declared_chunk not in chunk_ids:
                    _issue(issues, "error", "chunk_integrity", "evidence/chunk target mismatch", difficulty=difficulty, case_id=case_id)
                if int(target.get("page") or 0) != int(stored["primary_page_no"] or 0):
                    _issue(issues, "error", "page_integrity", "evidence page mismatch", difficulty=difficulty, case_id=case_id)
                if str(target.get("content_hash") or "") != str(stored["content_hash"] or ""):
                    _issue(issues, "error", "evidence_hash", "evidence content hash mismatch", difficulty=difficulty, case_id=case_id)
                if str(target.get("text") or "") != str(stored["content"] or "")[:1800]:
                    _issue(issues, "error", "evidence_text", "embedded evidence text is not the frozen prefix", difficulty=difficulty, case_id=case_id)
                declared_matches = [str(value) for value in target.get("match_terms") or []]
                invalid_matches = [value for value in declared_matches if _compact(value) not in _compact(stored["content"])]
                if invalid_matches:
                    _issue(issues, "error", "match_terms", f"declared match terms are absent from evidence: {invalid_matches}", difficulty=difficulty, case_id=case_id)
                if difficulty == "hard" and not declared_matches:
                    _issue(issues, "warning", "hard_evidence_coverage", f"hard evidence #{position + 1} has no expected-term label", difficulty=difficulty, case_id=case_id)

        profiles[difficulty] = {
            "case_count": len(cases),
            "answerable_count": answerable,
            "unanswerable_count": unanswerable,
            "critical_count": critical,
            "unique_evidence_count": len(set(evidence_ids)),
            "evidence_reuse_max": max(Counter(evidence_ids).values(), default=0),
            "page_min": min(pages, default=None),
            "page_max": max(pages, default=None),
            "unique_page_count": len(set(pages)),
            "mean_query_chars": round(mean(len(str(case.get("query") or "")) for case in cases), 2),
            "median_query_chars": median(len(str(case.get("query") or "")) for case in cases),
            "mean_reference_chars": round(mean(len(str(case.get("reference_answer") or "")) for case in cases), 2),
            "median_reference_chars": median(len(str(case.get("reference_answer") or "")) for case in cases),
            "mean_expected_terms": round(mean(expected_counts), 2),
        }

    case_counter = Counter(case_id for _, case_id in all_case_ids)
    for case_id, count in case_counter.items():
        if count > 1:
            _issue(issues, "error", "suite_case_id", f"case_id appears {count} times: {case_id}")
    query_counter = Counter(_compact(query) for _, _, query in all_queries)
    for normalized, count in query_counter.items():
        if normalized and count > 1:
            members = [f"{difficulty}/{case_id}" for difficulty, case_id, query in all_queries if _compact(query) == normalized]
            _issue(issues, "error", "duplicate_query", f"exact duplicate query: {members}")
    query_ngrams = [(difficulty, case_id, query, _ngrams(query)) for difficulty, case_id, query in all_queries]
    near_duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(query_ngrams):
        for right in query_ngrams[left_index + 1 :]:
            similarity = _jaccard(left[3], right[3])
            if similarity >= 0.88:
                near_duplicates.append(
                    {
                        "left": f"{left[0]}/{left[1]}",
                        "right": f"{right[0]}/{right[1]}",
                        "similarity": round(similarity, 4),
                    }
                )
    if near_duplicates:
        _issue(issues, "warning", "near_duplicate_query", f"found {len(near_duplicates)} query pairs with trigram Jaccard >= 0.88")

    hard_profile = profiles.get("hard") or {}
    if hard_profile.get("answerable_count") != 45 or hard_profile.get("unanswerable_count") != 5:
        _issue(issues, "error", "hard_composition", "hard dataset must contain 45 answerable and 5 unanswerable cases", difficulty="hard")

    severity_counts = Counter(item["severity"] for item in issues)
    return {
        "schema_version": "stt.eval.quality.v1",
        "validated_at": datetime.now(UTC).isoformat(),
        "suite_key": SUITE_KEY,
        "status": "passed" if not severity_counts["error"] else "failed",
        "checks": {
            "case_grain": "one immutable evaluation case",
            "total_cases": sum(profile["case_count"] for profile in profiles.values()),
            "active_index_version_id": index_version_id,
            "active_evidence_count": len(evidence_by_id),
            "source_sha256": source_sha,
            "near_duplicate_threshold": 0.88,
        },
        "profiles": profiles,
        "near_duplicates": near_duplicates,
        "issue_counts": dict(severity_counts),
        "issues": issues,
    }


def _extract_json(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("audit response must be an object")
    return parsed


def _audit_payload(case: dict[str, Any], difficulty: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "difficulty": difficulty,
        "query": case["query"],
        "conversation": case.get("conversation") or [],
        "reference_answer": case.get("reference_answer") or "",
        "unanswerable": bool(case.get("unanswerable")),
        "expected_terms": case.get("expected_terms") or [],
        "forbidden_terms": case.get("forbidden_terms") or [],
        "evidence": [
            {
                "evidence_id": item.get("evidence_id"),
                "page": item.get("page"),
                "text": item.get("text"),
            }
            for item in case.get("relevant_evidence") or []
        ],
    }


def online_audit(*, batch_size: int, model_override: str | None) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.shared.config.settings_config import settings

    model = model_override or settings.llm_default_model
    if not settings.openai_base_url or not settings.llm.api_key or not model:
        raise RuntimeError("online LLM configuration is incomplete")
    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
        max_retries=2,
    )
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    reviews: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        manifest = json.loads((DATASET_ROOT / f"{difficulty}.json").read_text(encoding="utf-8"))
        cases = list(manifest["cases"])
        for offset in range(0, len(cases), batch_size):
            batch = cases[offset : offset + batch_size]
            expected_ids = {str(case["case_id"]) for case in batch}
            batch_no = offset // batch_size + 1
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": _canonical_json(
                                    {
                                        "required_output": {
                                            "evaluations": [
                                                {
                                                    "case_id": "string",
                                                    "pass": "boolean",
                                                    "grounded": "boolean",
                                                    "complete": "boolean",
                                                    "unambiguous": "boolean",
                                                    "difficulty_fit": "boolean",
                                                    "uses_all_evidence": "boolean|null",
                                                    "severity": "none|low|medium|high",
                                                    "issues": ["concise Chinese strings"],
                                                }
                                            ]
                                        },
                                        "cases": [_audit_payload(case, difficulty) for case in batch],
                                    }
                                ),
                            },
                        ],
                    )
                    raw_text = str(completion.choices[0].message.content or "")
                    raw_path = RAW_ROOT / f"{difficulty}-batch-{batch_no:02d}-attempt-{attempt}.json"
                    raw_path.write_text(raw_text, encoding="utf-8")
                    parsed = _extract_json(raw_text)
                    values = parsed.get("evaluations")
                    if not isinstance(values, list):
                        raise ValueError("audit response is missing evaluations")
                    ids = [str(value.get("case_id") or "") for value in values if isinstance(value, dict)]
                    if len(values) != len(batch) or set(ids) != expected_ids:
                        raise ValueError(f"audit case IDs mismatch: {ids}")
                    normalized: list[dict[str, Any]] = []
                    for value in values:
                        if not isinstance(value, dict):
                            raise ValueError("audit evaluation must be an object")
                        severity = str(value.get("severity") or "").lower()
                        if severity not in {"none", "low", "medium", "high"}:
                            raise ValueError(f"invalid audit severity: {severity}")
                        normalized.append(
                            {
                                "case_id": str(value["case_id"]),
                                "difficulty": difficulty,
                                "pass": bool(value.get("pass")),
                                "grounded": bool(value.get("grounded")),
                                "complete": bool(value.get("complete")),
                                "unambiguous": bool(value.get("unambiguous")),
                                "difficulty_fit": bool(value.get("difficulty_fit")),
                                "uses_all_evidence": value.get("uses_all_evidence"),
                                "severity": severity,
                                "issues": [str(item) for item in value.get("issues") or []],
                            }
                        )
                    reviews.extend(normalized)
                    usage = completion.usage
                    usage_records.append(
                        {
                            "difficulty": difficulty,
                            "batch": batch_no,
                            "attempt": attempt,
                            "model": str(completion.model or model),
                            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                        }
                    )
                    print(f"[{difficulty}] audit batch {batch_no}: {len(batch)} reviewed", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - retry provider/shape failures
                    last_error = exc
                    if attempt == 3:
                        raise RuntimeError(f"{difficulty} audit batch {batch_no} failed") from exc
                    time.sleep(float(attempt))
            if last_error is not None and not reviews:
                raise last_error

    failed = [row for row in reviews if not row["pass"]]
    by_difficulty = {
        difficulty: {
            "reviewed": sum(row["difficulty"] == difficulty for row in reviews),
            "passed": sum(row["difficulty"] == difficulty and row["pass"] for row in reviews),
            "failed": sum(row["difficulty"] == difficulty and not row["pass"] for row in reviews),
            "pass_rate": round(
                sum(row["difficulty"] == difficulty and row["pass"] for row in reviews)
                / max(1, sum(row["difficulty"] == difficulty for row in reviews)),
                4,
            ),
        }
        for difficulty in DIFFICULTIES
    }
    return {
        "schema_version": "stt.eval.llm-quality-audit.v1",
        "audited_at": datetime.now(UTC).isoformat(),
        "suite_key": SUITE_KEY,
        "model": model,
        "dataset_checksums": _dataset_file_checksums(),
        "status": "passed" if not failed else "failed",
        "review_count": len(reviews),
        "failed_count": len(failed),
        "by_difficulty": by_difficulty,
        "usage": {
            "calls": len(usage_records),
            "prompt_tokens": sum(row["prompt_tokens"] for row in usage_records),
            "completion_tokens": sum(row["completion_tokens"] for row in usage_records),
            "total_tokens": sum(row["total_tokens"] for row in usage_records),
            "records": usage_records,
        },
        "reviews": reviews,
    }


def adjudicate_failed_reviews(*, model_override: str | None, rounds: int = 3) -> dict[str, Any]:
    """Re-review failed batched judgments one case at a time.

    A batch judge can occasionally quote text from a neighboring case.  The
    original audit remains immutable; this artifact records an explicit
    majority adjudication and all provider usage.
    """

    audit_path = QUALITY_ROOT / "llm-audit.json"
    if not audit_path.is_file():
        raise RuntimeError("quality/llm-audit.json does not exist")
    original = json.loads(audit_path.read_text(encoding="utf-8"))
    failures = [dict(row) for row in original.get("reviews") or [] if not row.get("pass")]
    if not failures:
        return {
            "schema_version": "stt.eval.llm-quality-adjudication.v1",
            "adjudicated_at": datetime.now(UTC).isoformat(),
            "suite_key": SUITE_KEY,
            "dataset_checksums": _dataset_file_checksums(),
            "status": "not_needed",
            "original_failed_count": 0,
            "resolved_count": 0,
            "unresolved_count": 0,
            "cases": [],
            "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "records": []},
        }

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.shared.config.settings_config import settings

    model = model_override or settings.llm_default_model
    if not settings.openai_base_url or not settings.llm.api_key or not model:
        raise RuntimeError("online LLM configuration is incomplete")
    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
        max_retries=2,
    )
    manifests = {
        difficulty: json.loads((DATASET_ROOT / f"{difficulty}.json").read_text(encoding="utf-8"))
        for difficulty in DIFFICULTIES
    }
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []
    for failure in failures:
        difficulty = str(failure["difficulty"])
        case_id = str(failure["case_id"])
        case = next(row for row in manifests[difficulty]["cases"] if row["case_id"] == case_id)
        judgments: list[dict[str, Any]] = []
        for round_no in range(1, rounds + 1):
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": AUDIT_SYSTEM_PROMPT
                        + "\n这是单题盲复核。必须把 actual_case.reference_answer 逐字复制到 reference_seen；禁止引用输入中不存在的句子。",
                    },
                    {
                        "role": "user",
                        "content": _canonical_json(
                            {
                                "instruction": "独立审核 actual_case。严格按 required_output 输出；reference_seen 必须逐字复制 actual_case.reference_answer。",
                                "required_output": {
                                    "evaluation": {
                                        "case_id": case_id,
                                        "reference_seen": "逐字复制 actual_case.reference_answer",
                                        "pass": "boolean",
                                        "grounded": "boolean",
                                        "complete": "boolean",
                                        "unambiguous": "boolean",
                                        "difficulty_fit": "boolean",
                                        "uses_all_evidence": "boolean|null",
                                        "severity": "none|low|medium|high",
                                        "issues": ["concise Chinese strings"],
                                    }
                                },
                                "actual_case": _audit_payload(case, difficulty),
                            }
                        ),
                    },
                ],
            )
            raw_text = str(completion.choices[0].message.content or "")
            (RAW_ROOT / f"adjudication-{case_id}-round-{round_no}.json").write_text(
                raw_text, encoding="utf-8"
            )
            parsed = _extract_json(raw_text)
            value = parsed.get("evaluation", parsed)
            if not isinstance(value, dict) or str(value.get("case_id") or "") != case_id:
                raise ValueError(f"invalid adjudication payload for {case_id}")
            if _compact(value.get("reference_seen")) != _compact(case.get("reference_answer")):
                raise ValueError(f"adjudicator did not read the actual reference for {case_id}")
            severity = str(value.get("severity") or "").lower()
            if severity not in {"none", "low", "medium", "high"}:
                raise ValueError(f"invalid adjudication severity for {case_id}: {severity}")
            judgment = {
                "round": round_no,
                "reference_seen": str(value.get("reference_seen") or ""),
                "pass": bool(value.get("pass")),
                "grounded": bool(value.get("grounded")),
                "complete": bool(value.get("complete")),
                "unambiguous": bool(value.get("unambiguous")),
                "difficulty_fit": bool(value.get("difficulty_fit")),
                "uses_all_evidence": value.get("uses_all_evidence"),
                "severity": severity,
                "issues": [str(item) for item in value.get("issues") or []],
            }
            judgments.append(judgment)
            usage = completion.usage
            usage_records.append(
                {
                    "case_id": case_id,
                    "round": round_no,
                    "model": str(completion.model or model),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            )
        pass_votes = sum(bool(row["pass"]) for row in judgments)
        resolved = pass_votes > rounds // 2
        decisions.append(
            {
                "case_id": case_id,
                "difficulty": difficulty,
                "original_judgment": failure,
                "rounds": judgments,
                "pass_votes": pass_votes,
                "fail_votes": rounds - pass_votes,
                "resolved_as_pass": resolved,
            }
        )
        print(f"[{difficulty}] adjudicated {case_id}: pass votes {pass_votes}/{rounds}", flush=True)

    unresolved = [row for row in decisions if not row["resolved_as_pass"]]
    return {
        "schema_version": "stt.eval.llm-quality-adjudication.v1",
        "adjudicated_at": datetime.now(UTC).isoformat(),
        "suite_key": SUITE_KEY,
        "model": model,
        "dataset_checksums": _dataset_file_checksums(),
        "status": "passed" if not unresolved else "failed",
        "original_failed_count": len(failures),
        "resolved_count": len(decisions) - len(unresolved),
        "unresolved_count": len(unresolved),
        "cases": decisions,
        "usage": {
            "calls": len(usage_records),
            "prompt_tokens": sum(row["prompt_tokens"] for row in usage_records),
            "completion_tokens": sum(row["completion_tokens"] for row in usage_records),
            "total_tokens": sum(row["total_tokens"] for row in usage_records),
            "records": usage_records,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-audit", action="store_true")
    parser.add_argument("--adjudicate-failures", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.batch_size <= 10:
        parser.error("--batch-size must be in 1..10")

    deterministic = deterministic_validation()
    _write_json(QUALITY_ROOT / "deterministic-validation.json", deterministic)
    audit: dict[str, Any] | None = None
    adjudication: dict[str, Any] | None = None
    if args.online_audit and deterministic["status"] == "passed":
        audit = online_audit(batch_size=args.batch_size, model_override=args.model)
        _write_json(QUALITY_ROOT / "llm-audit.json", audit)
    elif (QUALITY_ROOT / "llm-audit.json").is_file():
        existing_audit = json.loads(
            (QUALITY_ROOT / "llm-audit.json").read_text(encoding="utf-8")
        )
        if existing_audit.get("dataset_checksums") == _dataset_file_checksums():
            audit = existing_audit
    if args.adjudicate_failures and deterministic["status"] == "passed":
        adjudication = adjudicate_failed_reviews(model_override=args.model)
        _write_json(QUALITY_ROOT / "llm-adjudication.json", adjudication)
    elif (
        audit is not None
        and audit.get("status") != "passed"
        and (QUALITY_ROOT / "llm-adjudication.json").is_file()
    ):
        existing_adjudication = json.loads(
            (QUALITY_ROOT / "llm-adjudication.json").read_text(encoding="utf-8")
        )
        if existing_adjudication.get("dataset_checksums") == _dataset_file_checksums():
            adjudication = existing_adjudication
    online_ready = bool(
        audit is not None
        and (
            audit["status"] == "passed"
            or (adjudication is not None and adjudication["status"] in {"passed", "not_needed"})
        )
    )
    online_status = (
        "passed_after_adjudication"
        if audit is not None
        and audit["status"] != "passed"
        and adjudication is not None
        and adjudication["status"] == "passed"
        else audit["status"] if audit is not None else "not_run"
    )
    summary = {
        "schema_version": "stt.eval.quality-summary.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "suite_key": SUITE_KEY,
        "deterministic_status": deterministic["status"],
        "online_audit_status": online_status,
        "ready_for_import": deterministic["status"] == "passed" and online_ready,
        "dataset_checksums": _dataset_file_checksums(),
        "deterministic_report": "quality/deterministic-validation.json",
        "online_audit_report": "quality/llm-audit.json" if audit is not None else None,
        "adjudication_report": (
            "quality/llm-adjudication.json" if adjudication is not None else None
        ),
    }
    _write_json(QUALITY_ROOT / "quality-summary.json", summary)
    print(
        f"deterministic={summary['deterministic_status']} "
        f"online={summary['online_audit_status']} ready={summary['ready_for_import']}",
        flush=True,
    )
    return 0 if summary["ready_for_import"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
