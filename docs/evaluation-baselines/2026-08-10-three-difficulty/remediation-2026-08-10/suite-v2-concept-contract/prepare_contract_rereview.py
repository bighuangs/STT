"""Prepare same-semantic rereview requests for weak reviewer contracts.

The query/reference/evidence payload and frozen prompt are copied byte-for-byte
from the original request.  Only the transport ``custom_id`` changes, so these
cases receive an independent response without turning a reviewer-contract
failure into an unsupported dataset replacement.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from build_suite_v2 import _canonical_json, _sha256_json
import review_contracts as runner
from review_contracts import _read_jsonl, _walk_forbidden
from review_release_integrity import (
    EXPECTED_REVIEW_PROMPT_SHA256,
    contract_rereview_custom_id,
    validate_block_union_freshness,
    validate_contract_rereview_prepared_receipt,
    validate_request_pack_identity,
    validate_rereview_request_policy,
)


ROOT = Path(__file__).resolve().parent
AUDIT_PATH = ROOT / "review-block-union-audit.json"
OUTPUT_ROOT = ROOT / "contract-rereview"
EXPECTED_CASES = 150
EXPECTED_PROMPT_SHA256 = EXPECTED_REVIEW_PROMPT_SHA256


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_canonical_json(row) for row in rows) + "\n", encoding="utf-8")


def _new_custom_id(original_custom_id: str, payload_sha256: str) -> str:
    return contract_rereview_custom_id(original_custom_id, payload_sha256)


def prepare(*, write: bool) -> dict[str, Any]:
    audit = _read_json(AUDIT_PATH)
    missing_case_count = (audit.get("review") or {}).get("missing_case_count")
    if (
        audit.get("status") != "passed"
        or int((audit.get("review") or {}).get("successful_case_count") or 0)
        != EXPECTED_CASES
        or type(missing_case_count) is not int
        or missing_case_count != 0
    ):
        raise RuntimeError("final 150-case review/block-union audit has not passed")
    validate_block_union_freshness(
        audit,
        requests_path=ROOT / "review" / "requests.jsonl",
        responses_path=ROOT / "review" / "responses.jsonl",
        visual_audit_path=ROOT / "source-visual-audit.json",
        base_proposals_path=ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json",
        supplemental_proposals_path=ROOT / "reviewer-block-supplemental-proposals.json",
        label="contract-rereview block-union audit",
    )
    target_parent_ids = [str(value) for value in audit.get("contract_rereview_case_ids") or []]
    if len(target_parent_ids) != len(set(target_parent_ids)):
        raise RuntimeError("contract-rereview parent ids must be unique")
    if audit.get("contract_rereview_case_ids_sha256") != _sha256_json(
        target_parent_ids
    ):
        raise RuntimeError("contract-rereview queue hash is stale or malformed")
    if set(target_parent_ids) & set(audit.get("required_data_remediation_case_ids") or []):
        raise RuntimeError("data-remediation and same-semantic contract-rereview queues overlap")

    requests = _read_jsonl(ROOT / "review" / "requests.jsonl")
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_count=EXPECTED_CASES,
        pack_name="original full-suite review pack",
    )
    request_by_id = {str(row.get("custom_id") or ""): row for row in requests}
    request_map = _read_json(ROOT / "review" / "request-map.json")
    mapping_by_parent = {
        str(row.get("parent_case_id") or ""): row for row in request_map
    }
    response_rows = _read_jsonl(ROOT / "review" / "responses.jsonl")
    successful: dict[str, dict[str, Any]] = {}
    for row in response_rows:
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        prior = successful.get(custom_id)
        if prior is not None and _sha256_json(prior.get("review")) != _sha256_json(
            row.get("review")
        ):
            raise RuntimeError(f"conflicting successful source reviews: {custom_id}")
        successful.setdefault(custom_id, row)

    prepared_requests: list[dict[str, Any]] = []
    prepared_map: list[dict[str, Any]] = []
    original_ids = set(request_by_id)
    for parent_case_id in target_parent_ids:
        lineage = mapping_by_parent.get(parent_case_id)
        if lineage is None:
            raise RuntimeError(f"missing original lineage for {parent_case_id}")
        original_custom_id = str(lineage.get("custom_id") or "")
        original_request = request_by_id.get(original_custom_id)
        original_response = successful.get(original_custom_id)
        if original_request is None or original_response is None:
            raise RuntimeError(f"missing original request/response for {parent_case_id}")
        if not isinstance(original_response.get("review"), dict):
            raise RuntimeError(f"original review is not an object for {parent_case_id}")
        if str(original_request.get("prompt_sha256") or "") != EXPECTED_PROMPT_SHA256:
            raise RuntimeError(f"frozen prompt drift for {parent_case_id}")
        messages = deepcopy((original_request.get("request") or {}).get("messages") or [])
        if len(messages) != 2 or _walk_forbidden(json.loads(messages[1]["content"])):
            raise RuntimeError(f"invalid or leaking semantic payload for {parent_case_id}")
        payload_sha256 = str(original_request.get("payload_sha256") or "")
        rereview_request = deepcopy(original_request)
        rereview_request["custom_id"] = _new_custom_id(original_custom_id, payload_sha256)
        if rereview_request["custom_id"] in original_ids:
            raise RuntimeError(f"rereview custom id collides with original: {parent_case_id}")
        if rereview_request["request"]["messages"] != original_request["request"]["messages"]:
            raise RuntimeError(f"semantic messages changed for {parent_case_id}")
        prepared_requests.append(rereview_request)
        prepared_map.append(
            {
                "custom_id": rereview_request["custom_id"],
                "case_id": lineage.get("case_id"),
                "parent_case_id": parent_case_id,
                "difficulty": lineage.get("difficulty"),
                "original_custom_id": original_custom_id,
                "original_response_sha256": _sha256_json(original_response.get("review")),
                "payload_sha256": payload_sha256,
                "semantic_messages_sha256": _sha256_json(messages),
                "materialization_block_reason": (
                    audit.get("materialization_block_reasons") or {}
                ).get(parent_case_id),
                "semantic_payload_unchanged": True,
                "dataset_case_unchanged": True,
            }
        )

    if len({str(row["custom_id"]) for row in prepared_requests}) != len(prepared_requests):
        raise RuntimeError("contract-rereview custom ids are not unique")
    request_identity = validate_request_pack_identity(
        prepared_requests,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_count=len(target_parent_ids),
        expected_pack_sha256=None,
        custom_id_mode="contract_rereview",
        original_custom_ids_by_rereview_id={
            str(row["custom_id"]): str(row["original_custom_id"])
            for row in prepared_map
        },
        pack_name="same-semantic contract-rereview pack",
    )
    execution_contract = validate_rereview_request_policy(
        prepared_requests,
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        sse_completion_policy=runner.SSE_COMPLETION_POLICY,
    )
    summary = {
        "schema_version": "stt.eval.suite-v2-contract-rereview-pack.v1",
        "status": "prepared_pending_independent_same-semantic_review",
        "case_count": len(prepared_requests),
        "parent_case_ids": target_parent_ids,
        "parent_case_ids_sha256": _sha256_json(target_parent_ids),
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "request_pack_sha256": request_identity["request_pack_sha256"],
        "execution_contract_sha256": execution_contract,
        "semantic_payload_unchanged": True,
        "dataset_cases_changed": False,
        "new_custom_ids": True,
        "same_discriminative_variant_gate_required": True,
        "block_union_audit_sha256": _sha256_json(audit),
        "write_performed": write,
    }
    if write:
        validate_contract_rereview_prepared_receipt(summary)
        _write_jsonl(OUTPUT_ROOT / "requests.jsonl", prepared_requests)
        _write_json(OUTPUT_ROOT / "request-map.json", prepared_map)
        _write_json(OUTPUT_ROOT / "summary.json", summary)
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare(write=args.write)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
