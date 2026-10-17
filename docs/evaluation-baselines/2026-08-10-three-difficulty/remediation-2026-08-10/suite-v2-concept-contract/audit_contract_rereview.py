"""Validate same-semantic weak-contract rereviews before reviewed-manifest merge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from merge_reviewed_contracts import (
    CONTRACT_REREVIEW_ROOT,
    REVIEW_ROOT,
    _load_contract_rereviews,
    _materialize_concept_groups,
    _merge_contract,
    _sha256_json,
    _validate_contract_coverage,
)
from review_contracts import _read_jsonl


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = CONTRACT_REREVIEW_ROOT / "audit.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_by_parent() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        manifest = json.loads(
            (ROOT / "datasets" / f"{difficulty}.v2.json").read_text(encoding="utf-8")
        )
        for case in manifest.get("cases") or []:
            parent_case_id = str((case.get("metadata") or {}).get("parent_case_id") or "")
            if not parent_case_id or parent_case_id in result:
                raise RuntimeError("deterministic candidate lineage must be unique")
            result[parent_case_id] = case
    return result


def audit() -> dict[str, Any]:
    original_requests = {
        str(row["custom_id"]): row for row in _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    }
    original_request_map = json.loads(
        (REVIEW_ROOT / "request-map.json").read_text(encoding="utf-8")
    )
    original_parent_by_custom_id = {
        str(row.get("custom_id") or ""): str(row.get("parent_case_id") or "")
        for row in original_request_map
    }
    prompt_hashes = {
        str(row.get("prompt_sha256") or "") for row in original_requests.values()
    }
    if len(prompt_hashes) != 1:
        raise RuntimeError("original request prompt invariant failed")
    rereviews = _load_contract_rereviews(
        expected_prompt_hash=next(iter(prompt_hashes)),
        original_requests=original_requests,
        original_parent_by_custom_id=original_parent_by_custom_id,
    )
    candidates = _candidate_by_parent()
    resolved: list[str] = []
    failures: dict[str, dict[str, str]] = {}
    filter_audit: dict[str, Any] = {}
    for parent_case_id, response in sorted(rereviews.items()):
        candidate = candidates.get(parent_case_id)
        if candidate is None:
            raise RuntimeError(f"missing unchanged candidate for {parent_case_id}")
        review = dict(response.get("review") or {})
        if str(review.get("verdict") or "") == "block":
            failures[parent_case_id] = {
                "failure_type": "reviewer_block_after_same_semantic_rereview",
                "reason": "; ".join(str(value) for value in review.get("review_notes") or []),
            }
            continue
        try:
            contract, _qa = _merge_contract(candidate_case=candidate, response=response)
            _validate_contract_coverage(
                contract,
                candidate.get("relevant_evidence") or [],
                case_id=str(candidate.get("case_id") or parent_case_id),
                expected_branch_spans=[
                    str(branch.get("query_span") or "")
                    for branch in (candidate.get("answer_contract") or {}).get(
                        "required_branches"
                    )
                    or []
                ],
            )
            _groups, variant_audit = _materialize_concept_groups(
                contract,
                case_id=str(candidate.get("case_id") or parent_case_id),
                evidence=candidate.get("relevant_evidence") or [],
            )
        except RuntimeError as exc:
            failures[parent_case_id] = {
                "failure_type": "contract_or_discriminative_variant_gate_failure",
                "reason": str(exc),
            }
            continue
        resolved.append(parent_case_id)
        filter_audit[parent_case_id] = variant_audit

    status = "passed" if rereviews and not failures else "blocked"
    return {
        "schema_version": "stt.eval.suite-v2-contract-rereview-audit.v1",
        "status": status,
        "policy": {
            "semantic_payload_unchanged": True,
            "dataset_cases_changed": False,
            "same_discriminative_variant_gate_applied": True,
            "reviewer_block_is_not_auto_reclassified_as_data_block": True,
        },
        "case_count": len(rereviews),
        "resolved_case_count": len(resolved),
        "resolved_case_ids": resolved,
        "reviewer_contract_failure_count": len(failures),
        "reviewer_contract_failure_case_ids": sorted(failures),
        "reviewer_contract_failures": failures,
        "variant_filter_audit": filter_audit,
        "inputs": {
            "prepared_summary_sha256": _sha256_file(
                CONTRACT_REREVIEW_ROOT / "summary.json"
            ),
            "requests_sha256": _sha256_file(CONTRACT_REREVIEW_ROOT / "requests.jsonl"),
            "request_map_sha256": _sha256_file(
                CONTRACT_REREVIEW_ROOT / "request-map.json"
            ),
            "responses_sha256": _sha256_file(
                CONTRACT_REREVIEW_ROOT / "responses.jsonl"
            ),
            "execution_summary_sha256": _sha256_file(
                CONTRACT_REREVIEW_ROOT / "execution-summary.json"
            ),
            "audit_identity_sha256": _sha256_json(
                {
                    "targets": sorted(rereviews),
                    "response_hashes": {
                        key: _sha256_json(value.get("review"))
                        for key, value in sorted(rereviews.items())
                    },
                }
            ),
        },
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = audit()
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    if args.write:
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
