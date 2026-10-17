"""Audit the complete reviewer/visual/materialization remediation union.

This is a data-driven, fail-closed gate.  It does not call a model and it does
not change any candidate.  A final audit can be written only after all 150
frozen requests have one valid successful response and every case that needs a
semantic replacement/correction is covered by a versioned proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from merge_reviewed_contracts import (
    EXPECTED_CASES,
    _materialize_concept_groups,
    _merge_contract,
    _sha256_json,
    _validate_contract_coverage,
)
from review_contracts import (
    _validate_abstain_factual_constraint_recovery_audit,
    _validate_coverage_status_canonicalization_audit,
    _validate_empty_required_branch_block_audit,
    _validate_exact_payload_branch_span_quote_prefix_recovery_audit,
    _validate_inverse_edge_canonicalization_audit,
    _validate_nonsemantic_canonicalization_composition_audit,
    _validate_quote_fragment_inverse_composition_audit,
    _validate_redundant_field_coverage_composition_audit,
    _validate_exact_variant_deduplication_audit,
    _validate_saved_strict_revalidation_audit,
    _validate_unsupported_required_concept_block_audit,
    _read_jsonl,
    _review_schema_extra_paths,
    _validate_against_payload,
    _validate_review,
)
from review_release_integrity import (
    EXPECTED_REVIEW_PROMPT_SHA256,
    validate_block_union_freshness,
    validate_request_pack_identity,
)


ROOT = Path(__file__).resolve().parent
REVIEW_ROOT = ROOT / "review"
BASE_PROPOSALS_PATH = ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json"
SUPPLEMENTAL_PROPOSALS_PATH = ROOT / "reviewer-block-supplemental-proposals.json"
VISUAL_AUDIT_PATH = ROOT / "source-visual-audit.json"
OUTPUT_PATH = ROOT / "review-block-union-audit.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _proposal_ids(document: dict[str, Any], *, source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for proposal in document.get("proposals") or []:
        case_id = str(proposal.get("case_id") or "").strip()
        if not case_id or case_id in result:
            raise RuntimeError(f"{source}: proposal case ids must be non-empty and unique")
        result[case_id] = str(proposal.get("disposition") or "")
    return result


def _successful_reviews() -> tuple[dict[str, dict[str, Any]], list[str]]:
    requests = _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=EXPECTED_CASES,
        pack_name="frozen full-suite review pack",
    )
    request_ids = {str(row.get("custom_id") or "") for row in requests}
    if len(requests) != EXPECTED_CASES or len(request_ids) != EXPECTED_CASES or "" in request_ids:
        raise RuntimeError("frozen request pack must contain exactly 150 unique ids")
    prompt_hashes = {str(row.get("prompt_sha256") or "") for row in requests}
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise RuntimeError("frozen request pack must use exactly one non-empty prompt hash")

    successful: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for row in _read_jsonl(REVIEW_ROOT / "responses.jsonl"):
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        if custom_id not in request_ids:
            raise RuntimeError(f"successful response id is outside request pack: {custom_id}")
        if str(row.get("prompt_sha256") or "") not in prompt_hashes:
            raise RuntimeError(f"successful response prompt hash drift: {custom_id}")
        if not isinstance(row.get("review"), dict):
            raise RuntimeError(f"successful response has no review object: {custom_id}")
        prior = successful.get(custom_id)
        if prior is not None:
            if _sha256_json(prior.get("review")) != _sha256_json(row.get("review")):
                conflicts.append(custom_id)
            continue
        successful[custom_id] = row
    if conflicts:
        raise RuntimeError(f"conflicting duplicate successful responses: {sorted(conflicts)}")
    return successful, sorted(request_ids - set(successful))


def _candidate_by_parent() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        manifest = _load_json(ROOT / "datasets" / f"{difficulty}.v2.json")
        for case in manifest.get("cases") or []:
            parent_case_id = str((case.get("metadata") or {}).get("parent_case_id") or "")
            if not parent_case_id or parent_case_id in result:
                raise RuntimeError("candidate parent case ids must be non-empty and unique")
            result[parent_case_id] = case
    if len(result) != EXPECTED_CASES:
        raise RuntimeError(f"expected 150 deterministic candidates, found {len(result)}")
    return result


def audit() -> dict[str, Any]:
    successful, missing = _successful_reviews()
    source_response_rows = _read_jsonl(REVIEW_ROOT / "responses.jsonl")
    request_map = _load_json(REVIEW_ROOT / "request-map.json")
    if len(request_map) != EXPECTED_CASES:
        raise RuntimeError("request map must contain exactly 150 rows")
    lineage_by_custom_id = {str(row.get("custom_id") or ""): row for row in request_map}
    if len(lineage_by_custom_id) != EXPECTED_CASES:
        raise RuntimeError("request map custom ids must be unique")
    candidates = _candidate_by_parent()
    request_by_custom_id = {
        str(row.get("custom_id") or ""): row
        for row in _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    }
    candidate_lineage = {
        (
            str(case.get("case_id") or ""),
            parent_case_id,
            str((case.get("metadata") or {}).get("difficulty") or ""),
        )
        for parent_case_id, case in candidates.items()
    }
    request_lineage = {
        (
            str(row.get("case_id") or ""),
            str(row.get("parent_case_id") or ""),
            str(row.get("difficulty") or ""),
        )
        for row in request_map
    }
    if (
        set(lineage_by_custom_id) != set(request_by_custom_id)
        or len(request_lineage) != EXPECTED_CASES
        or request_lineage != candidate_lineage
    ):
        raise RuntimeError("request map does not exactly cover the 150 candidate parents")

    reviewer_blocks: set[str] = set()
    materialization_blocks: dict[str, str] = {}
    review_schema_extra_paths: dict[str, list[str]] = {}
    review_contract_validation_failures: dict[str, list[str]] = {}
    structural_normalization_actions: dict[str, list[str]] = {}
    inverse_edge_canonicalization_audits: dict[str, dict[str, Any]] = {}
    coverage_status_canonicalization_audits: dict[str, dict[str, Any]] = {}
    nonsemantic_composition_audits: dict[str, dict[str, Any]] = {}
    quote_fragment_composition_audits: dict[str, dict[str, Any]] = {}
    exact_span_quote_prefix_audits: dict[str, dict[str, Any]] = {}
    abstain_factual_constraint_audits: dict[str, dict[str, Any]] = {}
    redundant_field_composition_audits: dict[str, dict[str, Any]] = {}
    unsupported_required_concept_block_audits: dict[str, dict[str, Any]] = {}
    empty_required_branch_block_audits: dict[str, dict[str, Any]] = {}
    exact_variant_deduplication_audits: dict[str, dict[str, Any]] = {}
    saved_strict_revalidation_audits: dict[str, dict[str, Any]] = {}
    for custom_id, response in successful.items():
        lineage = lineage_by_custom_id.get(custom_id)
        if lineage is None:
            raise RuntimeError(f"missing request-map lineage for {custom_id}")
        parent_case_id = str(lineage.get("parent_case_id") or "")
        candidate = candidates.get(parent_case_id)
        if candidate is None:
            raise RuntimeError(f"missing candidate for lineage parent {parent_case_id}")
        review = dict(response.get("review") or {})
        if str(review.get("verdict") or "") == "block":
            reviewer_blocks.add(parent_case_id)
        schema_extra_paths = _review_schema_extra_paths(review)
        if schema_extra_paths:
            review_schema_extra_paths[parent_case_id] = schema_extra_paths
        request = request_by_custom_id.get(custom_id)
        if request is None:
            raise RuntimeError(f"missing frozen semantic request for {custom_id}")
        semantic_payload = json.loads(request["request"]["messages"][1]["content"])
        validation_errors = _validate_review(review)
        validation_errors.extend(_validate_against_payload(review, semantic_payload))
        inverse_edge_errors = _validate_inverse_edge_canonicalization_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(inverse_edge_errors)
        exact_dedupe_errors = _validate_exact_variant_deduplication_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(exact_dedupe_errors)
        coverage_status_errors = _validate_coverage_status_canonicalization_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(coverage_status_errors)
        composition_errors = (
            _validate_nonsemantic_canonicalization_composition_audit(
                response,
                semantic_payload,
                source_response_rows=source_response_rows,
            )
        )
        validation_errors.extend(composition_errors)
        quote_fragment_composition_errors = (
            _validate_quote_fragment_inverse_composition_audit(
                response,
                semantic_payload,
                source_response_rows=source_response_rows,
            )
        )
        validation_errors.extend(quote_fragment_composition_errors)
        abstain_factual_constraint_errors = (
            _validate_abstain_factual_constraint_recovery_audit(
                response,
                semantic_payload,
                source_response_rows=source_response_rows,
            )
        )
        validation_errors.extend(abstain_factual_constraint_errors)
        exact_span_quote_prefix_errors = (
            _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                response,
                semantic_payload,
                source_response_rows=source_response_rows,
            )
        )
        validation_errors.extend(exact_span_quote_prefix_errors)
        redundant_field_composition_errors = (
            _validate_redundant_field_coverage_composition_audit(
                response,
                semantic_payload,
                source_response_rows=source_response_rows,
            )
        )
        validation_errors.extend(redundant_field_composition_errors)
        unsupported_block_errors = _validate_unsupported_required_concept_block_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(unsupported_block_errors)
        empty_branch_block_errors = _validate_empty_required_branch_block_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(empty_branch_block_errors)
        strict_revalidation_errors = _validate_saved_strict_revalidation_audit(
            response,
            semantic_payload,
            source_response_rows=source_response_rows,
        )
        validation_errors.extend(strict_revalidation_errors)
        inverse_edge_audit = response.get(
            "same_response_inverse_edge_canonicalization"
        )
        if isinstance(inverse_edge_audit, dict) and not inverse_edge_errors:
            inverse_edge_canonicalization_audits[parent_case_id] = {
                "policy": inverse_edge_audit.get("policy"),
                "api_call_performed": inverse_edge_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": inverse_edge_audit.get("raw_review_sha256"),
                "projected_review_sha256": inverse_edge_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    inverse_edge_audit.get("source") or {}
                ).get("source_attempt"),
                "action_count": len(inverse_edge_audit.get("actions") or []),
            }
        exact_dedupe_audit = response.get(
            "same_response_exact_variant_deduplication"
        )
        if isinstance(exact_dedupe_audit, dict) and not exact_dedupe_errors:
            exact_variant_deduplication_audits[parent_case_id] = {
                "policy": exact_dedupe_audit.get("policy"),
                "api_call_performed": exact_dedupe_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": exact_dedupe_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": exact_dedupe_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    exact_dedupe_audit.get("source") or {}
                ).get("source_attempt"),
                "action_count": len(exact_dedupe_audit.get("actions") or []),
            }
        coverage_status_audit = response.get(
            "same_response_coverage_status_canonicalization"
        )
        if isinstance(coverage_status_audit, dict) and not coverage_status_errors:
            coverage_status_canonicalization_audits[parent_case_id] = {
                "policy": coverage_status_audit.get("policy"),
                "api_call_performed": coverage_status_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": coverage_status_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": coverage_status_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    coverage_status_audit.get("source") or {}
                ).get("source_attempt"),
                "action_count": len(coverage_status_audit.get("actions") or []),
            }
        composition_audit = response.get(
            "same_response_nonsemantic_canonicalization_composition"
        )
        if isinstance(composition_audit, dict) and not composition_errors:
            nonsemantic_composition_audits[parent_case_id] = {
                "policy": composition_audit.get("policy"),
                "api_call_performed": composition_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": composition_audit.get(
                    "raw_review_sha256"
                ),
                "intermediate_review_sha256": composition_audit.get(
                    "intermediate_review_sha256"
                ),
                "projected_review_sha256": composition_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (composition_audit.get("source") or {}).get(
                    "source_attempt"
                ),
                "stage_count": len(composition_audit.get("stages") or []),
            }
        quote_fragment_composition_audit = response.get(
            "same_response_quote_fragment_inverse_canonicalization_composition"
        )
        if (
            isinstance(quote_fragment_composition_audit, dict)
            and not quote_fragment_composition_errors
        ):
            quote_fragment_composition_audits[parent_case_id] = {
                "policy": quote_fragment_composition_audit.get("policy"),
                "api_call_performed": quote_fragment_composition_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": quote_fragment_composition_audit.get(
                    "raw_review_sha256"
                ),
                "intermediate_review_sha256": (
                    quote_fragment_composition_audit.get(
                        "intermediate_review_sha256"
                    )
                ),
                "projected_review_sha256": quote_fragment_composition_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    quote_fragment_composition_audit.get("source") or {}
                ).get("source_attempt"),
                "stage_count": len(
                    quote_fragment_composition_audit.get("stages") or []
                ),
            }
        exact_span_quote_prefix_audit = response.get(
            "saved_response_exact_payload_branch_span_quote_prefix_recovery"
        )
        if (
            isinstance(exact_span_quote_prefix_audit, dict)
            and not exact_span_quote_prefix_errors
        ):
            exact_span_quote_prefix_audits[parent_case_id] = {
                "policy": exact_span_quote_prefix_audit.get("policy"),
                "api_call_performed": exact_span_quote_prefix_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": exact_span_quote_prefix_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": exact_span_quote_prefix_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    exact_span_quote_prefix_audit.get("source") or {}
                ).get("source_attempt"),
                "stage_count": len(
                    exact_span_quote_prefix_audit.get("stages") or []
                ),
            }
        abstain_factual_constraint_audit = response.get(
            "saved_response_abstain_factual_constraint_recovery"
        )
        if (
            isinstance(abstain_factual_constraint_audit, dict)
            and not abstain_factual_constraint_errors
        ):
            abstain_factual_constraint_audits[parent_case_id] = {
                "policy": abstain_factual_constraint_audit.get("policy"),
                "api_call_performed": abstain_factual_constraint_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": abstain_factual_constraint_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": abstain_factual_constraint_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    abstain_factual_constraint_audit.get("source") or {}
                ).get("source_attempt"),
                "stage_count": len(
                    abstain_factual_constraint_audit.get("stages") or []
                ),
                "attempt_shape_count": len(
                    abstain_factual_constraint_audit.get(
                        "attempt_shape_receipts"
                    )
                    or []
                ),
                "semantic_consensus_claimed": (
                    abstain_factual_constraint_audit.get("selection") or {}
                ).get("semantic_consensus_claimed"),
                "cross_attempt_merge": (
                    abstain_factual_constraint_audit.get("selection") or {}
                ).get("cross_attempt_merge"),
            }
        redundant_field_composition_audit = response.get(
            "saved_response_redundant_field_coverage_composition"
        )
        if (
            isinstance(redundant_field_composition_audit, dict)
            and not redundant_field_composition_errors
        ):
            redundant_field_composition_audits[parent_case_id] = {
                "policy": redundant_field_composition_audit.get("policy"),
                "api_call_performed": redundant_field_composition_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": redundant_field_composition_audit.get(
                    "raw_review_sha256"
                ),
                "intermediate_review_sha256": (
                    redundant_field_composition_audit.get(
                        "intermediate_review_sha256"
                    )
                ),
                "projected_review_sha256": redundant_field_composition_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    redundant_field_composition_audit.get("source") or {}
                ).get("source_attempt"),
                "stage_count": len(
                    redundant_field_composition_audit.get("stages") or []
                ),
                "attempt_shape_count": len(
                    redundant_field_composition_audit.get(
                        "attempt_shape_receipts"
                    )
                    or []
                ),
                "semantic_consensus_claimed": (
                    redundant_field_composition_audit.get("selection") or {}
                ).get("semantic_consensus_claimed"),
                "cross_attempt_merge": (
                    redundant_field_composition_audit.get("selection") or {}
                ).get("cross_attempt_merge"),
            }
        unsupported_block_audit = response.get(
            "same_response_unsupported_required_concept_block"
        )
        if isinstance(unsupported_block_audit, dict) and not unsupported_block_errors:
            unsupported_required_concept_block_audits[parent_case_id] = {
                "policy": unsupported_block_audit.get("policy"),
                "api_call_performed": unsupported_block_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": unsupported_block_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": unsupported_block_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    unsupported_block_audit.get("source") or {}
                ).get("source_attempt"),
                "action_count": len(unsupported_block_audit.get("actions") or []),
            }
        empty_branch_block_audit = response.get(
            "multi_response_empty_required_branch_block"
        )
        if (
            isinstance(empty_branch_block_audit, dict)
            and not empty_branch_block_errors
        ):
            empty_required_branch_block_audits[parent_case_id] = {
                "policy": empty_branch_block_audit.get("policy"),
                "api_call_performed": empty_branch_block_audit.get(
                    "api_call_performed"
                ),
                "raw_review_sha256": empty_branch_block_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": empty_branch_block_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    empty_branch_block_audit.get("source") or {}
                ).get("source_attempt"),
                "consensus_response_count": (
                    empty_branch_block_audit.get("consensus") or {}
                ).get("response_count"),
                "action_count": len(empty_branch_block_audit.get("actions") or []),
            }
        strict_revalidation_audit = response.get(
            "saved_response_strict_revalidation"
        )
        if (
            isinstance(strict_revalidation_audit, dict)
            and not strict_revalidation_errors
        ):
            saved_strict_revalidation_audits[parent_case_id] = {
                "policy": strict_revalidation_audit.get("policy"),
                "api_call_performed": strict_revalidation_audit.get(
                    "api_call_performed"
                ),
                "validator_runner_version": strict_revalidation_audit.get(
                    "validator_runner_version"
                ),
                "raw_review_sha256": strict_revalidation_audit.get(
                    "raw_review_sha256"
                ),
                "projected_review_sha256": strict_revalidation_audit.get(
                    "projected_review_sha256"
                ),
                "source_attempt": (
                    strict_revalidation_audit.get("source") or {}
                ).get("source_attempt"),
            }
        normalization_actions = [
            str(item)
            for item in response.get("structural_normalization") or []
            if str(item).strip()
        ]
        if normalization_actions:
            structural_normalization_actions[parent_case_id] = normalization_actions
            validation_errors.append(
                "persisted success required structural normalization; strict independent "
                "same-semantic rereview is required"
            )
        if validation_errors:
            review_contract_validation_failures[parent_case_id] = validation_errors
            materialization_blocks[parent_case_id] = (
                "REVIEW_CONTRACT_VALIDATION_FAILURE requires same-semantic rereview; "
                f"errors={validation_errors}"
            )
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
            _materialize_concept_groups(
                contract,
                case_id=str(candidate.get("case_id") or parent_case_id),
                evidence=candidate.get("relevant_evidence") or [],
            )
        except RuntimeError as exc:
            materialization_blocks[parent_case_id] = str(exc)

    visual_audit = _load_json(VISUAL_AUDIT_PATH)
    visual_confirmed_blocks = {
        str(record.get("parent_case_id") or "")
        for record in visual_audit.get("records") or []
        if str(record.get("severity") or "") == "block"
    }
    visual_manual_remediation = {
        str(record.get("parent_case_id") or "")
        for record in visual_audit.get("records") or []
        if str(record.get("status") or "") == "manual_visual_review_required"
    }
    visual_confirmed_blocks.discard("")
    visual_manual_remediation.discard("")

    base_doc = _load_json(BASE_PROPOSALS_PATH)
    supplemental_doc = _load_json(SUPPLEMENTAL_PROPOSALS_PATH)
    base_proposals = _proposal_ids(base_doc, source="base proposals")
    supplemental_proposals = _proposal_ids(
        supplemental_doc, source="supplemental proposals"
    )
    duplicate_proposals = sorted(set(base_proposals) & set(supplemental_proposals))
    if duplicate_proposals:
        raise RuntimeError(f"duplicate base/supplemental proposals: {duplicate_proposals}")
    proposals = {**base_proposals, **supplemental_proposals}
    candidate_parent_ids = set(candidates)
    unknown_gate_ids = sorted(
        (
            reviewer_blocks
            | visual_confirmed_blocks
            | visual_manual_remediation
            | set(materialization_blocks)
            | set(proposals)
        )
        - candidate_parent_ids
    )
    if unknown_gate_ids:
        raise RuntimeError(f"review/block/proposal ids outside candidate universe: {unknown_gate_ids}")

    reviewer_block_union_visual_block = (
        reviewer_blocks | visual_confirmed_blocks | visual_manual_remediation
    )
    required_data_remediation = reviewer_block_union_visual_block
    contract_rereview = set(materialization_blocks) - required_data_remediation
    uncovered = sorted(required_data_remediation - set(proposals))
    proposal_without_trigger = sorted(set(proposals) - required_data_remediation)
    complete = not missing
    status = (
        "passed"
        if complete and not uncovered and not proposal_without_trigger
        else "blocked"
    )
    prompt_hashes = {
        str(row.get("prompt_sha256") or "")
        for row in _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    }
    result = {
        "schema_version": "stt.eval.suite-v2-review-block-union-audit.v1",
        "status": status,
        "policy": {
            "case_rules_uniform": True,
            "production_answers_read": False,
            "production_pass_fail_read": False,
            "visual_manual_review_requires_remediation": True,
            "weak_only_concept_requires_remediation": True,
            "proposal_coverage_required": True,
        },
        "review": {
            "expected_case_count": EXPECTED_CASES,
            "successful_case_count": len(successful),
            "missing_case_count": len(missing),
            "missing_custom_ids": missing,
            "prompt_sha256": next(iter(prompt_hashes)),
            "request_pack_sha256": _sha256_file(REVIEW_ROOT / "requests.jsonl"),
            "responses_sha256": _sha256_file(REVIEW_ROOT / "responses.jsonl"),
        },
        "reviewer_block_case_ids": sorted(reviewer_blocks),
        "visual_confirmed_block_case_ids": sorted(visual_confirmed_blocks),
        "visual_manual_remediation_case_ids": sorted(visual_manual_remediation),
        "reviewer_block_union_visual_block": sorted(reviewer_block_union_visual_block),
        "materialization_block_case_ids": sorted(materialization_blocks),
        "materialization_block_reasons": dict(sorted(materialization_blocks.items())),
        "review_schema_extra_field_case_ids": sorted(review_schema_extra_paths),
        "review_schema_extra_paths_by_case": dict(sorted(review_schema_extra_paths.items())),
        "review_schema_extra_disposition": (
            "same_semantic_rereview_required_no_projection_or_semantic_reuse"
        ),
        "review_contract_validation_failure_case_ids": sorted(
            review_contract_validation_failures
        ),
        "review_contract_validation_failure_reasons": dict(
            sorted(review_contract_validation_failures.items())
        ),
        "structural_normalization_case_ids": sorted(structural_normalization_actions),
        "structural_normalization_actions_by_case": dict(
            sorted(structural_normalization_actions.items())
        ),
        "structural_normalization_disposition": (
            "same_semantic_rereview_required_original_normalized_semantics_not_reused"
        ),
        "inverse_edge_canonicalization_case_ids": sorted(
            inverse_edge_canonicalization_audits
        ),
        "inverse_edge_canonicalization_audits_by_case": dict(
            sorted(inverse_edge_canonicalization_audits.items())
        ),
        "inverse_edge_canonicalization_disposition": (
            "adoptable_only_when_same_response_projection_is_strictly_reproducible"
        ),
        "exact_variant_deduplication_case_ids": sorted(
            exact_variant_deduplication_audits
        ),
        "exact_variant_deduplication_audits_by_case": dict(
            sorted(exact_variant_deduplication_audits.items())
        ),
        "exact_variant_deduplication_disposition": (
            "adoptable_only_for_reproducible_normalized_exact_duplicates"
        ),
        "coverage_status_canonicalization_case_ids": sorted(
            coverage_status_canonicalization_audits
        ),
        "coverage_status_canonicalization_audits_by_case": dict(
            sorted(coverage_status_canonicalization_audits.items())
        ),
        "coverage_status_canonicalization_disposition": (
            "adoptable_only_for_reproducible_finite_zero-support_status_aliases"
        ),
        "nonsemantic_canonicalization_composition_case_ids": sorted(
            nonsemantic_composition_audits
        ),
        "nonsemantic_canonicalization_composition_audits_by_case": dict(
            sorted(nonsemantic_composition_audits.items())
        ),
        "nonsemantic_canonicalization_composition_disposition": (
            "adoptable_only_for_fixed_order_disjoint_saved_response_projection"
        ),
        "quote_fragment_inverse_composition_case_ids": sorted(
            quote_fragment_composition_audits
        ),
        "quote_fragment_inverse_composition_audits_by_case": dict(
            sorted(quote_fragment_composition_audits.items())
        ),
        "quote_fragment_inverse_composition_disposition": (
            "adoptable_only_for_unique_exact_payload_quote_reconstruction_and_"
            "inverse_branch_replay"
        ),
        "exact_payload_branch_span_quote_prefix_recovery_case_ids": sorted(
            exact_span_quote_prefix_audits
        ),
        "exact_payload_branch_span_quote_prefix_recovery_audits_by_case": dict(
            sorted(exact_span_quote_prefix_audits.items())
        ),
        "exact_payload_branch_span_quote_prefix_recovery_disposition": (
            "format_only_final_complete_attempt_exact_payload_span_replacement_"
            "audited_without_semantic_consensus_or_cross_attempt_merge"
        ),
        "abstain_factual_constraint_recovery_case_ids": sorted(
            abstain_factual_constraint_audits
        ),
        "abstain_factual_constraint_recovery_audits_by_case": dict(
            sorted(abstain_factual_constraint_audits.items())
        ),
        "abstain_factual_constraint_recovery_disposition": (
            "format_only_final_complete_attempt_abstention_constraint_removal_"
            "audited_without_semantic_consensus_or_cross_attempt_merge"
        ),
        "redundant_field_coverage_composition_case_ids": sorted(
            redundant_field_composition_audits
        ),
        "redundant_field_coverage_composition_audits_by_case": dict(
            sorted(redundant_field_composition_audits.items())
        ),
        "redundant_field_coverage_composition_disposition": (
            "format_only_final_complete_attempt_recovery_audited_no_semantic_"
            "consensus_or_cross_attempt_merge"
        ),
        "unsupported_required_concept_block_case_ids": sorted(
            unsupported_required_concept_block_audits
        ),
        "unsupported_required_concept_block_audits_by_case": dict(
            sorted(unsupported_required_concept_block_audits.items())
        ),
        "unsupported_required_concept_block_disposition": (
            "data_remediation_required_never_adopt_as_runnable_contract"
        ),
        "empty_required_branch_block_case_ids": sorted(
            empty_required_branch_block_audits
        ),
        "empty_required_branch_block_audits_by_case": dict(
            sorted(empty_required_branch_block_audits.items())
        ),
        "empty_required_branch_block_disposition": (
            "branch_segmentation_data_remediation_and_new_payload_rereview_required_"
            "never_adopt_as_runnable_contract"
        ),
        "saved_strict_revalidation_case_ids": sorted(
            saved_strict_revalidation_audits
        ),
        "saved_strict_revalidation_audits_by_case": dict(
            sorted(saved_strict_revalidation_audits.items())
        ),
        "saved_strict_revalidation_disposition": (
            "adoptable_only_as_unchanged_single_saved_response_after_strict_policy_fix"
        ),
        "reviewer_contract_failure_case_ids": sorted(materialization_blocks),
        "reviewer_contract_failure_reasons": dict(sorted(materialization_blocks.items())),
        "required_data_remediation_case_ids": sorted(required_data_remediation),
        "contract_rereview_case_ids": sorted(contract_rereview),
        "contract_rereview_case_ids_sha256": _sha256_json(
            sorted(contract_rereview)
        ),
        "contract_rereview_policy": {
            "semantic_payload_must_remain_identical": True,
            "new_custom_id_required": True,
            "same_prompt_required": True,
            "dataset_replacement_before_rereview_forbidden": True,
            "same_discriminative_variant_gate_required": True,
        },
        "proposal_case_ids": sorted(proposals),
        "proposal_dispositions": dict(sorted(proposals.items())),
        "uncovered_data_remediation_case_ids": uncovered,
        "proposal_without_current_trigger_case_ids": proposal_without_trigger,
        "inputs": {
            "visual_audit_sha256": _sha256_file(VISUAL_AUDIT_PATH),
            "base_proposals_sha256": _sha256_file(BASE_PROPOSALS_PATH),
            "supplemental_proposals_sha256": _sha256_file(
                SUPPLEMENTAL_PROPOSALS_PATH
            ),
        },
    }
    validate_block_union_freshness(
        result,
        requests_path=REVIEW_ROOT / "requests.jsonl",
        responses_path=REVIEW_ROOT / "responses.jsonl",
        visual_audit_path=VISUAL_AUDIT_PATH,
        base_proposals_path=BASE_PROPOSALS_PATH,
        supplemental_proposals_path=SUPPLEMENTAL_PROPOSALS_PATH,
        label="new block-union audit",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write only a complete, fully covered final audit",
    )
    args = parser.parse_args()
    try:
        result = audit()
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write:
        if result["status"] != "passed":
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reason": "final audit is incomplete or proposal coverage has gaps",
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
