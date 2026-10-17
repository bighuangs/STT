"""Optional, cost-bearing semantic review for ambiguous failure diagnostics.

Dry-run is the default.  Network calls require both --execute and
--acknowledge-cost.  No API key, Authorization header, or credential-bearing
URL is written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
QUEUE_PATH = HERE / "llm-review-queue.jsonl"
OUTPUT_PATH = HERE / "llm-review-responses.jsonl"
SUMMARY_PATH = HERE / "llm-review-summary.json"
MODEL_DEFAULT = "qwen-plus"
CALIBRATION_CASES = ("P3500-EASY-001", "P3500-EASY-006")
CALIBRATION_EXPECTED = {
    "P3500-EASY-001": False,
    "P3500-EASY-006": True,
}
ALLOWED_LABELS = {
    "retrieval_insufficient",
    "retrieval_unclear_or_noisy",
    "context_loss",
    "intermediate_pipeline_loss",
    "evaluator_error",
    "generation_or_reasoning",
    "dataset_reference_issue",
    "other",
}


def _read_queue() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        raise RuntimeError(f"review queue is missing: {QUEUE_PATH}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(QUEUE_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("case_id"):
            raise RuntimeError(f"invalid queue row {line_number}")
        rows.append(value)
    return rows


def _sanitize_endpoint(base_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("OPENAI_BASE_URL must be an http(s) URL")
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    clean_base = urllib.parse.urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    endpoint = clean_base if clean_base.endswith("/chat/completions") else clean_base + "/chat/completions"
    safe_display = urllib.parse.urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    return endpoint, safe_display


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
    return {
        "query": evidence.get("query"),
        "reference_answer": evidence.get("reference_answer"),
        "authorized_evidence_excerpt": evidence.get("relevant_evidence_excerpt"),
        "actual_answer": evidence.get("answer"),
        "expected_terms_secondary_hint_only": evidence.get("expected_terms"),
        "metrics": evidence.get("metrics"),
        "stage_trace": evidence.get("stage_trace"),
        "decision": evidence.get("decision"),
        "assessment": evidence.get("assessment"),
    }


def _messages(row: Mapping[str, Any], prior: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    evidence = _compact_evidence(row)
    system = """You are auditing a saved RAG evaluation result. Return one JSON object only.

HARD ANSWER-CLAIMS ISOLATION RULE:
- ONLY text inside ACTUAL_ANSWER_BEGIN/ACTUAL_ANSWER_END counts as the model's visible answer.
- AUTHORIZED_EVIDENCE is grounding material only. Unstated source text NEVER counts as answered.
- A citation marker such as [E1] can ground an adjacent visible claim, but it cannot import other source sentences.
- First enumerate answer_claims using only ACTUAL_ANSWER. Then compare those claims with the question, reference answer, and authorized evidence.
- expected_terms are secondary evaluator hints, never the sole truth source.

Choose exactly one primary root-cause label from the supplied taxonomy and zero or more secondary labels. Distinguish semantic correctness from the historical passed flag. Do not create case-id-specific rules.

Required JSON keys: semantic_correct (boolean or null), semantic_status (correct|partial|incorrect|unresolved), answer_claims (array), missing_reference_aspects (array), primary (string), secondary (array), decisive_evidence (array), confidence_score (0..1), reasoning_summary (string)."""
    user = {
        "case_id_for_join_only": row.get("case_id"),
        "taxonomy": sorted(ALLOWED_LABELS),
        "query": evidence.get("query"),
        "reference_answer": evidence.get("reference_answer"),
        "AUTHORIZED_EVIDENCE": evidence.get("authorized_evidence_excerpt"),
        "ACTUAL_ANSWER_BEGIN": evidence.get("actual_answer"),
        "ACTUAL_ANSWER_END": True,
        "expected_terms_secondary_hint_only": evidence.get("expected_terms_secondary_hint_only"),
        "pipeline_signals": {
            "metrics": evidence.get("metrics"),
            "stage_trace": evidence.get("stage_trace"),
            "decision": evidence.get("decision"),
            "assessment": evidence.get("assessment"),
        },
    }
    if prior:
        user["prior_judgment_to_recheck"] = prior
        user["recheck_instruction"] = (
            "Independently re-derive answer_claims under the isolation rule. Resolve any conflict "
            "between the prior judgment and the saved evidence; do not defer to the prior result."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _request(
    endpoint: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float,
) -> tuple[dict[str, Any], str, str]:
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    request = urllib.request.Request(
        endpoint,
        data=encoded.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(raw)
    return parsed, raw, _sha256_text(encoded)


def _judgment(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("judge response has no choices")
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("judge response content is not text")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise RuntimeError("judge response content is not an object")
    if value.get("primary") not in ALLOWED_LABELS:
        raise RuntimeError(f"judge returned invalid primary label: {value.get('primary')}")
    secondary = value.get("secondary") or []
    if not isinstance(secondary, list) or any(label not in ALLOWED_LABELS for label in secondary):
        raise RuntimeError("judge returned invalid secondary labels")
    score = value.get("confidence_score")
    if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        raise RuntimeError("judge returned invalid confidence_score")
    if value.get("semantic_correct") not in {True, False, None}:
        raise RuntimeError("judge returned invalid semantic_correct")
    return value


def _deterministic_semantic(row: Mapping[str, Any]) -> bool | None:
    diagnosis = row.get("deterministic_diagnosis")
    if not isinstance(diagnosis, Mapping):
        return None
    status = diagnosis.get("semantic_correctness")
    if not isinstance(status, Mapping):
        evidence = row.get("evidence")
        status = evidence.get("semantic_correctness") if isinstance(evidence, Mapping) else None
    if not isinstance(status, Mapping):
        return None
    value = status.get("status")
    if value == "likely_correct":
        return True
    if value == "likely_incorrect_or_incomplete":
        return False
    return None


def _needs_second_review(row: Mapping[str, Any], judgment: Mapping[str, Any]) -> bool:
    score = float(judgment.get("confidence_score") or 0)
    deterministic = _deterministic_semantic(row)
    conflict = deterministic is not None and judgment.get("semantic_correct") is not deterministic
    return 0.4 <= score <= 0.8 or conflict


def _select(rows: Sequence[dict[str, Any]], sample_size: int, full_run: bool) -> list[dict[str, Any]]:
    by_id = {str(row["case_id"]): row for row in rows}
    selected: list[dict[str, Any]] = [by_id[case_id] for case_id in CALIBRATION_CASES if case_id in by_id]
    if full_run:
        return list(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        diagnosis = row.get("deterministic_diagnosis")
        primary = diagnosis.get("primary") if isinstance(diagnosis, Mapping) else "unknown"
        groups.setdefault((str(row.get("difficulty")), str(primary)), []).append(row)
    candidates = []
    for key in sorted(groups):
        candidates.extend(sorted(groups[key], key=lambda item: str(item["case_id"])))
    seen = {str(row["case_id"]) for row in selected}
    for row in candidates:
        if str(row["case_id"]) in seen:
            continue
        selected.append(row)
        seen.add(str(row["case_id"]))
        if len(selected) >= sample_size:
            break
    return selected[:sample_size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Review the entire queue after calibration")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rows = _read_queue()
    if args.sample_size < len(CALIBRATION_CASES):
        raise RuntimeError(f"--sample-size must be at least {len(CALIBRATION_CASES)}")
    if args.all and not args.confirm_full_run:
        raise RuntimeError("--all requires --confirm-full-run after reviewing calibration output")
    selected = _select(rows, args.sample_size, args.all)
    plan = {
        "mode": "full" if args.all else "calibration_sample",
        "model": args.model,
        "selected_case_ids": [row["case_id"] for row in selected],
        "calibration_cases": list(CALIBRATION_CASES),
        "execute": args.execute,
        "cost_acknowledged": args.acknowledge_cost,
        "second_review_rule": "confidence 0.4..0.8 inclusive or deterministic/LLM semantic conflict",
    }
    if not args.execute:
        print(json.dumps({"status": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return 0
    if not args.acknowledge_cost:
        raise RuntimeError("--execute requires --acknowledge-cost")
    if not args.base_url:
        raise RuntimeError("OPENAI_BASE_URL or --base-url is required")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key environment variable is missing: {args.api_key_env}")
    endpoint, safe_base_url = _sanitize_endpoint(args.base_url)

    records: list[dict[str, Any]] = []
    for row in selected:
        first_messages = _messages(row)
        response, raw, payload_hash = _request(endpoint, api_key, args.model, first_messages, args.timeout)
        first = _judgment(response)
        records.append(
            {
                "case_id": row["case_id"],
                "difficulty": row.get("difficulty"),
                "review_index": 1,
                "calibration": row["case_id"] in CALIBRATION_CASES,
                "calibration_expected_semantic_correct": CALIBRATION_EXPECTED.get(row["case_id"]),
                "request_summary": {
                    "model": args.model,
                    "base_url": safe_base_url,
                    "temperature": 0,
                    "payload_sha256": payload_hash,
                    "prompt_sha256": _sha256_text(json.dumps(first_messages, ensure_ascii=False)),
                    "answer_claims_isolation": True,
                },
                "judgment": first,
                "usage": response.get("usage"),
                "response_model": response.get("model"),
                "response_id": response.get("id"),
                "raw_response": raw,
            }
        )
        if _needs_second_review(row, first):
            second_messages = _messages(row, first)
            response2, raw2, payload_hash2 = _request(
                endpoint, api_key, args.model, second_messages, args.timeout
            )
            second = _judgment(response2)
            records.append(
                {
                    "case_id": row["case_id"],
                    "difficulty": row.get("difficulty"),
                    "review_index": 2,
                    "calibration": row["case_id"] in CALIBRATION_CASES,
                    "calibration_expected_semantic_correct": CALIBRATION_EXPECTED.get(row["case_id"]),
                    "request_summary": {
                        "model": args.model,
                        "base_url": safe_base_url,
                        "temperature": 0,
                        "payload_sha256": payload_hash2,
                        "prompt_sha256": _sha256_text(json.dumps(second_messages, ensure_ascii=False)),
                        "answer_claims_isolation": True,
                        "recheck": True,
                    },
                    "judgment": second,
                    "usage": response2.get("usage"),
                    "response_model": response2.get("model"),
                    "response_id": response2.get("id"),
                    "raw_response": raw2,
                }
            )
        time.sleep(0.1)

    OUTPUT_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    calibration = {}
    for case_id, expected in CALIBRATION_EXPECTED.items():
        case_records = [record for record in records if record["case_id"] == case_id]
        final = case_records[-1]["judgment"] if case_records else {}
        calibration[case_id] = {
            "expected_semantic_correct": expected,
            "actual_semantic_correct": final.get("semantic_correct"),
            "matched": final.get("semantic_correct") is expected,
            "review_count": len(case_records),
        }
    summary = {
        "status": "succeeded",
        "plan": plan,
        "record_count": len(records),
        "unique_case_count": len({record["case_id"] for record in records}),
        "calibration": calibration,
        "calibration_passed": all(item["matched"] for item in calibration.values()),
        "credentials_recorded": False,
        "output_sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
