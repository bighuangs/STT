"""Prepare every audited data-blocker remediation for independent review.

The sibling remediation proposal is treated as an immutable, hash-checked
authoring input.  This script does not reuse any review of the superseded case.
It emits a separate request pack whose semantic payload uses the exact system
prompt of the original 150-case review.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from build_suite_v2 import (
    _build_case,
    _canonical_json,
    _load_inputs,
    _review_requests,
    _sha256_json,
)
from review_release_integrity import (
    EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS,
    EXPECTED_REVIEW_PROMPT_SHA256,
    validate_blocker_rereview_prepared_receipt,
    validate_block_union_freshness,
    validate_proposal_validation_binding,
    validate_request_pack_identity,
    validate_rereview_request_policy,
)
import review_contracts as runner


ROOT = Path(__file__).resolve().parent
PROPOSAL_ROOT = ROOT.parent / "suite-v2-blocker-remediation"
OUTPUT_ROOT = ROOT / "blocker-rereview"
EXPECTED_BASE_PROPOSALS = EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
EXPECTED_SUPPLEMENTAL_PROPOSALS = EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
EXPECTED_DATA_BLOCKERS = EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
EXPECTED_PROMPT_SHA256 = EXPECTED_REVIEW_PROMPT_SHA256


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_canonical_json(row) for row in rows) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(*, write: bool) -> dict[str, Any]:
    proposals_doc = _read_json(PROPOSAL_ROOT / "proposals.json")
    validation = _read_json(PROPOSAL_ROOT / "validation.json")
    supplemental_doc = _read_json(ROOT / "reviewer-block-supplemental-proposals.json")
    block_union_audit = _read_json(ROOT / "review-block-union-audit.json")
    if (
        block_union_audit.get("status") != "passed"
        or int((block_union_audit.get("review") or {}).get("successful_case_count") or 0)
        != 150
        or (block_union_audit.get("review") or {}).get("missing_case_count") != 0
    ):
        raise RuntimeError("final 150-case reviewer/visual block-union audit has not passed")
    validate_block_union_freshness(
        block_union_audit,
        requests_path=ROOT / "review" / "requests.jsonl",
        responses_path=ROOT / "review" / "responses.jsonl",
        visual_audit_path=ROOT / "source-visual-audit.json",
        base_proposals_path=PROPOSAL_ROOT / "proposals.json",
        supplemental_proposals_path=ROOT / "reviewer-block-supplemental-proposals.json",
        label="blocker-rereview block-union audit",
    )
    if (
        proposals_doc.get("schema_version")
        != "stt.eval.suite-v2-blocker-remediation-proposal.v2-no-index-refresh"
        or validation.get("status") != "passed"
        or validation.get("schema_version")
        != "stt.eval.suite-v2-blocker-remediation-validation.v2-no-index-refresh"
    ):
        raise RuntimeError("blocker remediation proposal/validation gate is not satisfied")
    validate_proposal_validation_binding(
        validation,
        proposals_path=PROPOSAL_ROOT / "proposals.json",
        label="base blocker-remediation validation",
    )
    if (
        validation.get("full_suite_total_preserved") != 150
        or (validation.get("active_index") or {}).get("index_rebuild_required") is not False
        or (validation.get("active_index") or {}).get("evidence_refresh_required") is not False
    ):
        raise RuntimeError("proposal is not the validated no-index-refresh variant")

    base_proposals = [
        {**deepcopy(item), "_proposal_schema_version": proposals_doc["schema_version"]}
        for item in proposals_doc.get("proposals") or []
    ]
    supplemental_proposals = [
        {**deepcopy(item), "_proposal_schema_version": supplemental_doc["schema_version"]}
        for item in supplemental_doc.get("proposals") or []
    ]
    if len(base_proposals) != EXPECTED_BASE_PROPOSALS:
        raise RuntimeError(
            f"expected the validated {EXPECTED_BASE_PROPOSALS}-case no-index proposal"
        )
    if len(supplemental_proposals) != EXPECTED_SUPPLEMENTAL_PROPOSALS:
        raise RuntimeError(
            f"expected {EXPECTED_SUPPLEMENTAL_PROPOSALS} supplemental reviewer-block "
            "proposals"
        )
    proposals = [*base_proposals, *supplemental_proposals]
    if len(proposals) != EXPECTED_DATA_BLOCKERS:
        raise RuntimeError(
            f"expected exactly {EXPECTED_DATA_BLOCKERS} data-blocker proposals"
        )
    parent_ids = [str(item.get("case_id") or "") for item in proposals]
    if len(set(parent_ids)) != len(proposals):
        raise RuntimeError("blocker proposals must use unique parent ids")
    if {str(item["case_id"]) for item in base_proposals} != set(
        validation.get("target_case_ids") or []
    ):
        raise RuntimeError("proposal ids do not match validation target ids")
    required_data_ids = set(
        str(value)
        for value in block_union_audit.get("required_data_remediation_case_ids") or []
    )
    if len(required_data_ids) != EXPECTED_DATA_BLOCKERS or set(parent_ids) != required_data_ids:
        raise RuntimeError(
            "the 15 data-remediation proposals do not exactly cover the final "
            "reviewer/visual block union"
        )
    audit_inputs = block_union_audit.get("inputs") or {}
    if (
        audit_inputs.get("base_proposals_sha256")
        != _sha256_file(PROPOSAL_ROOT / "proposals.json")
        or audit_inputs.get("supplemental_proposals_sha256")
        != _sha256_file(ROOT / "reviewer-block-supplemental-proposals.json")
        or (block_union_audit.get("review") or {}).get("responses_sha256")
        != _sha256_file(ROOT / "review" / "responses.jsonl")
        or (block_union_audit.get("review") or {}).get("request_pack_sha256")
        != _sha256_file(ROOT / "review" / "requests.jsonl")
    ):
        raise RuntimeError("block-union audit input hashes no longer match current inputs")

    inputs = _load_inputs()
    source_case_by_id = {
        str(case["case_id"]): case
        for manifest in inputs.manifests.values()
        for case in manifest.get("cases") or []
    }
    original_request_map = json.loads(
        (ROOT / "review" / "request-map.json").read_text(encoding="utf-8")
    )
    original_success = {
        str(row.get("custom_id") or ""): row
        for row in (
            json.loads(line)
            for line in (ROOT / "review" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
        if row.get("status") == "succeeded"
    }
    original_custom_by_parent = {
        str(item["parent_case_id"]): str(item["custom_id"])
        for item in original_request_map
    }
    for proposal in supplemental_proposals:
        parent_case_id = str(proposal["case_id"])
        source_case = source_case_by_id.get(parent_case_id)
        if source_case is None or _sha256_json(source_case) != str(
            (proposal.get("audit") or {}).get("original_case_sha256") or ""
        ):
            raise RuntimeError(f"supplemental original-case hash mismatch: {parent_case_id}")
        proposed_case = proposal.get("proposed_case") or {}
        if (
            proposal.get("disposition") != "semantic_replacement"
            or str(proposed_case.get("query") or "") == str(source_case.get("query") or "")
            or str(proposed_case.get("reference_answer") or "")
            == str(source_case.get("reference_answer") or "")
            or (proposed_case.get("metadata") or {}).get("lineage_semantic_replacement")
            is not True
        ):
            raise RuntimeError(
                f"supplemental reviewer block must replace query+reference semantics: "
                f"{parent_case_id}"
            )
        review = original_success.get(original_custom_by_parent.get(parent_case_id, ""))
        if review is None or (review.get("review") or {}).get("verdict") != "block":
            raise RuntimeError(
                f"supplemental remediation lacks a successful uniform-review block: {parent_case_id}"
            )
    stripped_visual_audit = {
        key: value for key, value in inputs.visual_audit.items() if key not in set(parent_ids)
    }
    remediated_inputs = replace(inputs, visual_audit=stripped_visual_audit)
    candidates: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    proposal_by_parent: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        parent_case_id = str(proposal["case_id"])
        proposed_case = deepcopy(proposal["proposed_case"])
        expected_hash = str((proposal.get("audit") or {}).get("proposed_case_sha256") or "")
        if _sha256_json(proposed_case) != expected_hash:
            raise RuntimeError(f"proposal hash mismatch: {parent_case_id}")
        if str(proposed_case.get("case_id") or "") != parent_case_id:
            raise RuntimeError(f"proposal case id drift: {parent_case_id}")
        difficulty = str(proposal.get("difficulty") or "")
        if difficulty not in {"easy", "medium", "hard"}:
            raise RuntimeError(f"invalid proposal difficulty: {parent_case_id}")
        proposal_groups = proposed_case.pop("concept_groups", None)
        v2_case, qa_row, review_payload = _build_case(
            difficulty=difficulty,
            source_case=proposed_case,
            inputs=remediated_inputs,
        )
        v2_case["metadata"] = {
            **dict(v2_case.get("metadata") or {}),
            "blocker_remediation_status": "proposed_case_built_pending_independent_rereview",
            "blocker_remediation_disposition": proposal.get("disposition"),
            "blocker_remediation_proposal_sha256": expected_hash,
            "blocker_remediation_proposal_schema": proposal[
                "_proposal_schema_version"
            ],
            "proposal_concept_groups_sha256": _sha256_json(proposal_groups),
            "superseded_review_must_not_be_reused": True,
        }
        qa_row.update(
            {
                "blocker_remediation_disposition": proposal.get("disposition"),
                "blocker_remediation_proposal_sha256": expected_hash,
                "superseded_review_reused": False,
            }
        )
        candidates.append(v2_case)
        qa_rows.append(qa_row)
        payloads.append(review_payload)
        proposal_by_parent[parent_case_id] = proposal

    requests, prompt_hash = _review_requests(payloads, expected_count=len(proposals))
    if prompt_hash != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(f"same-prompt invariant failed: {prompt_hash}")
    if len({row["custom_id"] for row in requests}) != len(proposals):
        raise RuntimeError("blocker rereview request ids are not unique")
    request_identity = validate_request_pack_identity(
        requests,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_count=len(proposals),
        pack_name="changed blocker-rereview pack",
    )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        sse_completion_policy=runner.SSE_COMPLETION_POLICY,
    )
    original_request_ids = {
        json.loads(line)["custom_id"]
        for line in (ROOT / "review" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }
    if original_request_ids & {row["custom_id"] for row in requests}:
        raise RuntimeError("changed-case request ids collide with superseded review ids")

    request_map = []
    for case, request in zip(candidates, requests, strict=True):
        metadata = case.get("metadata") or {}
        parent_case_id = str(metadata["parent_case_id"])
        proposal = proposal_by_parent[parent_case_id]
        request_map.append(
            {
                "custom_id": request["custom_id"],
                "case_id": case["case_id"],
                "parent_case_id": parent_case_id,
                "difficulty": metadata["difficulty"],
                "disposition": proposal["disposition"],
                "proposed_case_sha256": (proposal.get("audit") or {})[
                    "proposed_case_sha256"
                ],
                "candidate_case_sha256": _sha256_json(case),
                "payload_sha256": request["payload_sha256"],
            }
        )
    summary = {
        "schema_version": "stt.eval.suite-v2-blocker-rereview-pack.v1",
        "status": "prepared_pending_independent_same_prompt_review",
        "case_count": len(candidates),
        "difficulty_counts": {
            difficulty: sum(
                str((case.get("metadata") or {}).get("difficulty")) == difficulty
                for case in candidates
            )
            for difficulty in ("easy", "medium", "hard")
        },
        "disposition_counts": {
            disposition: sum(item.get("disposition") == disposition for item in proposals)
            for disposition in sorted({str(item.get("disposition") or "") for item in proposals})
        },
        "prompt_sha256": prompt_hash,
        "request_pack_sha256": request_identity["request_pack_sha256"],
        "execution_contract_sha256": execution_contract,
        "proposal_document_sha256": _sha256_json(proposals_doc),
        "supplemental_proposal_document_sha256": _sha256_json(supplemental_doc),
        "proposal_validation_sha256": _sha256_json(validation),
        "validated_no_index_proposal_count": len(base_proposals),
        "supplemental_reviewer_block_proposal_count": len(supplemental_proposals),
        "reviewer_block_union_visual_block": block_union_audit.get(
            "reviewer_block_union_visual_block"
        ),
        "data_blocker_case_count": len(required_data_ids),
        "review_block_union_audit_sha256": _sha256_json(block_union_audit),
        "index_rebuild_required": False,
        "evidence_refresh_required": False,
        "superseded_review_reused": False,
        "write_performed": write,
    }
    if write:
        validate_blocker_rereview_prepared_receipt(summary)
        _write_json(OUTPUT_ROOT / "candidates.json", candidates)
        _write_json(OUTPUT_ROOT / "qa.json", qa_rows)
        _write_jsonl(OUTPUT_ROOT / "requests.jsonl", requests)
        _write_json(OUTPUT_ROOT / "request-map.json", request_map)
        _write_json(OUTPUT_ROOT / "summary.json", summary)
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the audited data-blocker pack")
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
