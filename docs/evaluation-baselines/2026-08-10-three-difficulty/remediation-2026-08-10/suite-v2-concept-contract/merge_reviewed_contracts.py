"""Merge only a complete, validated 150-case LLM review into reviewed candidates.

This script never mutates the frozen v1 manifests or the deterministic v2
candidate files.  It refuses partial review packs, unknown evidence/branch
labels, duplicate response ids, failed responses, and prompt-hash drift.  Even a
successful merge remains non-importable until independent validation resolves
all reviewer ``block`` verdicts and the suite owner explicitly promotes it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.evaluation import freeze_case_pass_policy  # noqa: E402
from concept_materialization import (  # noqa: E402
    ABSTENTION_SENTINEL_CANONICAL_TEXT,
    ABSTENTION_SENTINEL_POLARITY,
    ABSTENTION_SENTINEL_VARIANTS,
    LEGACY_MATERIALIZATION_POLICY_VERSION,
    MATERIALIZATION_POLICY_VERSION,
    abstention_variant_is_pure_refusal as _abstention_variant_is_pure_refusal,
    materialize_concept_groups as _shared_materialize_concept_groups,
    policy_normalized as _policy_normalized,
    variant_weak_reason as _shared_variant_weak_reason,
)
from review_contracts import (  # noqa: E402
    POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
    POST150_SAVED_RECOVERY_RUNNER_VERSION,
    RUNNER_VERSION,
    SAVED_RESPONSE_TRANSPORT,
    SSE_COMPLETION_POLICY,
    TRANSPORT,
    _read_jsonl,
    _review_materialization_errors,
    _validate_against_payload,
    _validate_abstain_factual_constraint_recovery_audit,
    _validate_coverage_status_canonicalization_audit,
    _validate_empty_required_branch_block_audit,
    _validate_exact_payload_branch_span_quote_prefix_recovery_audit,
    _validate_exact_variant_deduplication_audit,
    _validate_inverse_edge_canonicalization_audit,
    _validate_invalid_attempt_raw_normalization_provenance,
    _validate_nonsemantic_canonicalization_composition_audit,
    _validate_quote_fragment_inverse_composition_audit,
    _validate_redundant_field_coverage_composition_audit,
    _validate_review,
    _validate_saved_strict_revalidation_audit,
    _validate_saved_post150_recovery,
    _validate_unsupported_required_concept_block_audit,
    _walk_forbidden,
)
from review_release_integrity import (  # noqa: E402
    EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS,
    EXPECTED_REREVIEW_MODEL,
    EXPECTED_REVIEW_PROMPT_SHA256,
    FORMAL_STREAM_DIAGNOSTIC_POLICY,
    MATERIALIZATION_ADOPTION_POLICY,
    MATERIALIZATION_RETRY_QUOTA,
    formal_rereview_success_ledger,
    formal_rereview_failed_result_errors,
    materialization_adoption_for_authoritative_success,
    sha256_json,
    validate_block_union_freshness,
    validate_blocker_rereview_prepared_receipt,
    validate_contract_rereview_prepared_receipt,
    validate_proposal_validation_binding,
    validate_request_pack_identity,
    validate_rereview_request_policy,
)


ROOT = Path(__file__).resolve().parent
REVIEW_ROOT = ROOT / "review"
BLOCKER_REREVIEW_ROOT = ROOT / "blocker-rereview"
CONTRACT_REREVIEW_ROOT = ROOT / "contract-rereview"
REVIEWED_ROOT = ROOT / "reviewed"
BASE_PROPOSALS_PATH = ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json"
BASE_PROPOSAL_VALIDATION_PATH = (
    ROOT.parent / "suite-v2-blocker-remediation" / "validation.json"
)
SUPPLEMENTAL_PROPOSALS_PATH = ROOT / "reviewer-block-supplemental-proposals.json"
VISUAL_AUDIT_PATH = ROOT / "source-visual-audit.json"
DIFFICULTIES = ("easy", "medium", "hard")
EXPECTED_CASES = 150
REVIEWED_CONTRACT_SCHEMA = "stt.eval.answer-contract.v2-llm-reviewed-candidate"
REVIEWED_SUITE_SCHEMA = "stt.eval.suite.v2-llm-reviewed-candidate"
REVIEWED_SUITE_KEY = "three-difficulty-live-2026-08-11-reviewed-v2"
REVIEWED_VERSION = "2.0.0-reviewed-v2.20260811"
REVIEWED_DATASET_KEYS = {
    difficulty: f"pantum-p3500-{difficulty}-live-reviewed-v2-2026-08-11"
    for difficulty in DIFFICULTIES
}
FROZEN_CASE_PASS_POLICY = freeze_case_pass_policy({"policy_version": "stt.eval.case.v3"})
_ABSTENTION_CUE_RE = re.compile(
    r"(?:(?:未|没有)(?:提供|给出|说明|记载|包含)|无相关|无.*信息|没有.*信息|"
    r"无法|不能确定|不可确定|不足以|缺少|缺乏|不包含|"
    r"not provided|cannot determine|insufficient)",
    re.IGNORECASE,
)


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = str(value or "").strip()
        key = stripped.casefold()
        if not stripped or key in seen:
            continue
        seen.add(key)
        result.append(stripped)
    return result


def _formal_post150_success_ledger(
    responses: list[dict[str, Any]],
    request_by_id: dict[str, dict[str, Any]],
    *,
    execution_contract_sha256: str,
    expected_prompt_hash: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    def validate_saved(
        result: dict[str, Any],
        request: dict[str, Any],
        source_rows: Sequence[dict[str, Any]],
    ) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        return _validate_saved_post150_recovery(
            result,
            request=request,
            semantic_payload=payload,
            source_response_rows=source_rows,
            execution_contract_sha256=execution_contract_sha256,
            materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        )

    def validate_live(
        result: dict[str, Any], request: dict[str, Any]
    ) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        review = result.get("review")
        if not isinstance(review, dict):
            return ["live formal review must be an object"]
        errors = _validate_review(review)
        errors.extend(_validate_against_payload(review, payload))
        errors.extend(_validate_inverse_edge_canonicalization_audit(result, payload))
        errors.extend(_validate_exact_variant_deduplication_audit(result, payload))
        errors.extend(_validate_coverage_status_canonicalization_audit(result, payload))
        for item in result.get("invalid_attempts_before_success") or []:
            errors.extend(
                _validate_invalid_attempt_raw_normalization_provenance(
                    item, result, payload
                )
            )
        return errors

    def validate_failed(
        result: dict[str, Any], request: dict[str, Any]
    ) -> list[str]:
        return formal_rereview_failed_result_errors(
            result,
            request,
            expected_sequence=(
                list(request_by_id).index(str(request["custom_id"])) + 1
            ),
            execution_contract_sha256=execution_contract_sha256,
            live_runner_version=RUNNER_VERSION,
            live_transport=TRANSPORT,
            validate_invalid_attempt=(
                _validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    return formal_rereview_success_ledger(
        responses,
        requests_by_id=request_by_id,
        execution_contract_sha256=execution_contract_sha256,
        expected_prompt_sha256=expected_prompt_hash,
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=RUNNER_VERSION,
        live_transport=TRANSPORT,
        saved_runner_version=POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=SAVED_RESPONSE_TRANSPORT,
        additional_saved_profiles=(
            (
                POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
                SAVED_RESPONSE_TRANSPORT,
            ),
        ),
        validate_live_result=validate_live,
        validate_saved_result=validate_saved,
        label=label,
        materializer_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        validate_materialization_result=lambda result, request: (
            _review_materialization_errors(
                result["review"],
                json.loads(request["request"]["messages"][1]["content"]),
                materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
            )
        ),
        validate_failed_result=validate_failed,
    )


def _expected_review_payload(candidate_case: dict[str, Any]) -> dict[str, Any]:
    contract = candidate_case.get("answer_contract") or {}
    branches = contract.get("required_branches") or []
    evidence = candidate_case.get("relevant_evidence") or []
    review_flags = [
        str(item.get("code") or "")
        for item in (contract.get("review") or {}).get("flags") or []
    ]
    return {
        "query": str(candidate_case.get("query") or ""),
        "reference_answer": str(candidate_case.get("reference_answer") or ""),
        "unanswerable": bool(candidate_case.get("unanswerable")),
        "legacy_expected_terms": [
            str(value) for value in candidate_case.get("expected_terms") or []
        ],
        "forbidden_terms": [
            str(value) for value in candidate_case.get("forbidden_terms") or []
        ],
        "branches": [
            {"branch_label": f"B{index + 1}", "query_span": branch.get("query_span")}
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
            "answerability": contract.get("answerability"),
            "required_concepts": contract.get("required_concepts"),
            "critical_constraints": contract.get("critical_constraints"),
            "required_branches": branches,
            "evidence_concept_coverage": contract.get("evidence_concept_coverage"),
            "review_flags": review_flags,
        },
    }


def _variant_weak_reason(
    variant: str,
    *,
    critical_values: set[str],
    evidence_tokens: set[str],
) -> str | None:
    return _shared_variant_weak_reason(
        variant,
        critical_values=critical_values,
        evidence_tokens=evidence_tokens,
    )


def _materialize_concept_groups(
    contract: dict[str, Any],
    *,
    case_id: str,
    evidence: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _shared_materialize_concept_groups(contract, evidence)
    except ValueError as exc:
        raise RuntimeError(
            f"{case_id}: DISCRIMINATIVE_VARIANT_GATE {exc}"
        ) from exc


def _validate_contract_coverage(
    contract: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    case_id: str,
    expected_branch_spans: Sequence[str],
) -> None:
    concepts = (contract.get("required_concepts") or {}).get("all_of") or []
    concept_id_list = [str(item.get("concept_id") or "") for item in concepts]
    concept_ids = set(concept_id_list)
    if (
        not concept_ids
        or "" in concept_ids
        or len(concept_ids) != len(concept_id_list)
    ):
        raise RuntimeError(f"{case_id}: contract must have unique authored concepts")
    branches = contract.get("required_branches") or []
    coverage = contract.get("evidence_concept_coverage") or []
    evidence_id_list = [str(item.get("evidence_id") or "") for item in evidence]
    coverage_id_list = [str(item.get("evidence_id") or "") for item in coverage]
    expected_branch_counter = Counter(
        _policy_normalized(value) for value in expected_branch_spans
    )
    actual_branch_counter = Counter(
        _policy_normalized(branch.get("query_span")) for branch in branches
    )
    if (
        "" in evidence_id_list
        or len(set(evidence_id_list)) != len(evidence_id_list)
        or Counter(coverage_id_list) != Counter(evidence_id_list)
        or actual_branch_counter != expected_branch_counter
        or "" in actual_branch_counter
    ):
        raise RuntimeError(
            f"{case_id}: evidence/branch grain does not match the authored candidate"
        )
    answerability = str((contract.get("answerability") or {}).get("required") or "")
    constraints = contract.get("critical_constraints") or {}
    if answerability == "abstain":
        concept = concepts[0] if len(concepts) == 1 else {}
        concept_id = str(concept.get("concept_id") or "")
        variants = [
            str(concept.get("canonical_text") or ""),
            *(str(value or "") for value in concept.get("any_of") or []),
        ]
        if (
            len(concepts) != 1
            or concept.get("canonical_text") != ABSTENTION_SENTINEL_CANONICAL_TEXT
            or concept.get("any_of") != list(ABSTENTION_SENTINEL_VARIANTS)
            or constraints.get("polarity") != ABSTENTION_SENTINEL_POLARITY
            or not variants
            or any(not _abstention_variant_is_pure_refusal(value) for value in variants)
            or concept.get("evidence_support")
            or Counter(
                _policy_normalized(value)
                for value in concept.get("query_branch_spans") or []
            )
            != expected_branch_counter
            or any(
                branch.get("required_concept_ids") != [concept_id]
                or branch.get("answerability") != "must_abstain"
                for branch in branches
            )
            or any(
                item.get("required_concept_ids")
                or item.get("coverage_status") != "not_applicable_to_abstention"
                for item in coverage
            )
            or constraints.get("numerics")
            or constraints.get("identifiers")
        ):
            raise RuntimeError(
                f"{case_id}: abstain contract must express only an evidence-free "
                "absence/uncertainty boundary"
            )
        return
    if answerability != "answer":
        raise RuntimeError(f"{case_id}: unknown answerability contract {answerability!r}")

    unsupported = [
        str(item.get("concept_id") or "")
        for item in concepts
        if not item.get("evidence_support") or not item.get("query_branch_spans")
    ]
    concept_evidence_edges = {
        (str(concept.get("concept_id") or ""), str(support.get("evidence_id") or ""))
        for concept in concepts
        for support in concept.get("evidence_support") or []
    }
    coverage_edges = {
        (str(concept_id), str(item.get("evidence_id") or ""))
        for item in coverage
        if item.get("coverage_status") == "covered"
        for concept_id in item.get("required_concept_ids") or []
    }
    concept_branch_edges = {
        (str(concept.get("concept_id") or ""), _policy_normalized(span))
        for concept in concepts
        for span in concept.get("query_branch_spans") or []
    }
    branch_edges = {
        (str(concept_id), _policy_normalized(branch.get("query_span")))
        for branch in branches
        for concept_id in branch.get("required_concept_ids") or []
    }
    covered_evidence_ids = {
        str(item.get("evidence_id") or "")
        for item in coverage
        if item.get("coverage_status") == "covered"
    }
    constraint_evidence_ids = {
        str(evidence_id)
        for constraint in [
            *(constraints.get("numerics") or []),
            *(constraints.get("identifiers") or []),
        ]
        for evidence_id in constraint.get("evidence_ids") or []
    }
    invalid_coverage_rows = [
        str(item.get("evidence_id") or "")
        for item in coverage
        if (
            item.get("coverage_status") == "covered"
            and not item.get("required_concept_ids")
        )
        or (
            item.get("coverage_status") in {"uncovered", "insufficient"}
            and item.get("required_concept_ids")
        )
        or item.get("coverage_status")
        not in {"covered", "uncovered", "insufficient"}
    ]
    if (
        unsupported
        or concept_evidence_edges != coverage_edges
        or concept_branch_edges != branch_edges
        or invalid_coverage_rows
        or not covered_evidence_ids
        or not constraint_evidence_ids.issubset(set(evidence_id_list))
        or any(not branch.get("required_concept_ids") for branch in branches)
    ):
        raise RuntimeError(
            f"{case_id}: evidence/concept/branch edge coverage is incomplete or asymmetric; "
            f"unsupported={unsupported}; invalid_coverage={invalid_coverage_rows}; "
            f"unknown_constraint_evidence="
            f"{sorted(constraint_evidence_ids - set(evidence_id_list))}"
        )


def _load_complete_reviews() -> tuple[dict[str, dict[str, Any]], str]:
    requests = _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=EXPECTED_CASES,
        pack_name="full-suite merge review pack",
    )
    if len(requests) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} requests, found {len(requests)}")
    request_ids = {str(row["custom_id"]) for row in requests}
    prompt_hashes = {str(row["prompt_sha256"]) for row in requests}
    if len(request_ids) != EXPECTED_CASES or len(prompt_hashes) != 1:
        raise RuntimeError("request pack uniqueness/prompt invariant failed")

    response_rows = _read_jsonl(REVIEW_ROOT / "responses.jsonl")
    successful: dict[str, dict[str, Any]] = {}
    for row in response_rows:
        custom_id = str(row.get("custom_id") or "")
        if row.get("status") != "succeeded":
            continue
        if custom_id in successful:
            if _sha256_json(successful[custom_id]["review"]) != _sha256_json(row["review"]):
                raise RuntimeError(f"conflicting successful resume rows for {custom_id}")
            continue
        if custom_id not in request_ids:
            raise RuntimeError(f"response id not present in request pack: {custom_id}")
        if str(row.get("prompt_sha256") or "") not in prompt_hashes:
            raise RuntimeError(f"response prompt hash drift for {custom_id}")
        if not isinstance(row.get("review"), dict):
            raise RuntimeError(f"successful response has no review object: {custom_id}")
        successful[custom_id] = row
    missing = sorted(request_ids - set(successful))
    if missing:
        raise RuntimeError(
            f"review pack is incomplete: {len(successful)}/{EXPECTED_CASES} succeeded; "
            f"first missing={missing[:3]}"
        )
    request_by_id = {str(row["custom_id"]): row for row in requests}
    for custom_id, response in successful.items():
        payload = json.loads(
            request_by_id[custom_id]["request"]["messages"][1]["content"]
        )
        audit_errors = _validate_saved_strict_revalidation_audit(
            response, payload, source_response_rows=response_rows
        )
        audit_errors.extend(
            _validate_coverage_status_canonicalization_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_nonsemantic_canonicalization_composition_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_quote_fragment_inverse_composition_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_abstain_factual_constraint_recovery_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_redundant_field_coverage_composition_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_unsupported_required_concept_block_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        audit_errors.extend(
            _validate_empty_required_branch_block_audit(
                response,
                payload,
                source_response_rows=response_rows,
            )
        )
        if audit_errors:
            raise RuntimeError(
                f"invalid saved-response strict revalidation {custom_id}: "
                + "; ".join(audit_errors)
            )
    return successful, next(iter(prompt_hashes))


def _load_blocker_rereviews(
    *, expected_prompt_hash: str, original_request_ids: set[str]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    candidates = json.loads(
        (BLOCKER_REREVIEW_ROOT / "candidates.json").read_text(encoding="utf-8")
    )
    request_map = json.loads(
        (BLOCKER_REREVIEW_ROOT / "request-map.json").read_text(encoding="utf-8")
    )
    requests = _read_jsonl(BLOCKER_REREVIEW_ROOT / "requests.jsonl")
    responses = _read_jsonl(BLOCKER_REREVIEW_ROOT / "responses.jsonl")
    execution_summary = json.loads(
        (BLOCKER_REREVIEW_ROOT / "execution-summary.json").read_text(encoding="utf-8")
    )
    prepared_summary = json.loads(
        (BLOCKER_REREVIEW_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    prepared_receipt = validate_blocker_rereview_prepared_receipt(prepared_summary)
    current_block_union_audit = json.loads(
        (ROOT / "review-block-union-audit.json").read_text(encoding="utf-8")
    )
    current_base_proposals = json.loads(BASE_PROPOSALS_PATH.read_text(encoding="utf-8"))
    current_supplemental_proposals = json.loads(
        SUPPLEMENTAL_PROPOSALS_PATH.read_text(encoding="utf-8")
    )
    current_proposal_validation = json.loads(
        BASE_PROPOSAL_VALIDATION_PATH.read_text(encoding="utf-8")
    )
    current_receipts = {
        "review_block_union_audit_sha256": sha256_json(current_block_union_audit),
        "proposal_document_sha256": sha256_json(current_base_proposals),
        "supplemental_proposal_document_sha256": sha256_json(
            current_supplemental_proposals
        ),
        "proposal_validation_sha256": sha256_json(current_proposal_validation),
    }
    if any(
        prepared_summary.get(key) != value for key, value in current_receipts.items()
    ):
        raise RuntimeError("changed blocker-case prepared upstream receipt is stale")
    base_proposal_rows = current_base_proposals.get("proposals") or []
    supplemental_proposal_rows = current_supplemental_proposals.get("proposals") or []
    current_proposal_parent_ids = {
        str(row.get("case_id") or "")
        for row in [*base_proposal_rows, *supplemental_proposal_rows]
    }
    required_proposal_parent_ids = {
        str(value)
        for value in current_block_union_audit.get(
            "required_data_remediation_case_ids"
        )
        or []
    }
    if (
        len(base_proposal_rows) != EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
        or len(supplemental_proposal_rows)
        != EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        or len(current_proposal_parent_ids)
        != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
        or current_proposal_parent_ids != required_proposal_parent_ids
    ):
        raise RuntimeError("changed blocker-case current upstream is not exact 13+2")
    expected_changed_cases = int(prepared_summary.get("case_count") or 0)
    if execution_summary.get("status") != "succeeded":
        raise RuntimeError("changed blocker-case rereview is not complete")
    if expected_changed_cases != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS or not all(
        len(items) == expected_changed_cases for items in (candidates, request_map, requests)
    ):
        raise RuntimeError("changed blocker-case pack count mismatch")
    request_by_id = {str(row["custom_id"]): row for row in requests}
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=expected_prompt_hash,
        expected_count=expected_changed_cases,
        expected_pack_sha256=str(prepared_receipt["request_pack_sha256"]),
        pack_name="changed blocker-rereview merge pack",
    )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=RUNNER_VERSION,
        transport=TRANSPORT,
        sse_completion_policy=SSE_COMPLETION_POLICY,
    )
    if (
        prepared_summary.get("execution_contract_sha256") != execution_contract
        or execution_summary.get("execution_contract_sha256") != execution_contract
    ):
        raise RuntimeError("changed blocker-case execution-contract receipt drift")
    if len(request_by_id) != expected_changed_cases or set(request_by_id) & original_request_ids:
        raise RuntimeError("changed blocker request ids are not unique/independent")
    if {str(row.get("prompt_sha256") or "") for row in requests} != {
        expected_prompt_hash
    }:
        raise RuntimeError("changed blocker cases were not reviewed with the frozen prompt")
    successful = _formal_post150_success_ledger(
        responses,
        request_by_id,
        execution_contract_sha256=execution_contract,
        expected_prompt_hash=expected_prompt_hash,
        label="changed blocker-rereview merge ledger",
    )
    authoritative_successful = successful
    successful = {}
    for row in authoritative_successful.values():
        custom_id = str(row.get("custom_id") or "")
        if str(row.get("prompt_sha256") or "") != expected_prompt_hash:
            raise RuntimeError(f"changed-case response prompt hash drift: {custom_id}")
        review = dict(row.get("review") or {})
        payload = json.loads(request_by_id[custom_id]["request"]["messages"][1]["content"])
        errors = _validate_review(review)
        errors.extend(_validate_against_payload(review, payload))
        errors.extend(
            _validate_inverse_edge_canonicalization_audit(
                row, payload, source_response_rows=responses
            )
        )
        errors.extend(
            _validate_exact_variant_deduplication_audit(
                row, payload, source_response_rows=responses
            )
        )
        errors.extend(
            _validate_coverage_status_canonicalization_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_nonsemantic_canonicalization_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_quote_fragment_inverse_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_abstain_factual_constraint_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_redundant_field_coverage_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_unsupported_required_concept_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_empty_required_branch_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_saved_strict_revalidation_audit(
                row, payload, source_response_rows=responses
            )
        )
        if row.get("transport") == SAVED_RESPONSE_TRANSPORT:
            errors.extend(
                _validate_saved_post150_recovery(
                    row,
                    request=request_by_id[custom_id],
                    semantic_payload=payload,
                    source_response_rows=responses,
                    execution_contract_sha256=execution_contract,
                )
            )
        if any(
            str(item).strip()
            for item in row.get("structural_normalization") or []
        ):
            errors.append(
                "successful changed-case rereview required structural normalization"
            )
        if errors:
            raise RuntimeError(f"invalid changed-case review {custom_id}: {'; '.join(errors)}")
        successful[custom_id] = row
    if set(successful) != set(request_by_id):
        raise RuntimeError(
            f"changed blocker rereview incomplete: {len(successful)}/{expected_changed_cases}"
        )

    current_proposal_sha_by_parent: dict[str, str] = {}
    for document in (current_base_proposals, current_supplemental_proposals):
        for proposal in document.get("proposals") or []:
            parent_case_id = str(proposal.get("case_id") or "")
            proposed_case_sha256 = str(
                (proposal.get("audit") or {}).get("proposed_case_sha256") or ""
            )
            if (
                not parent_case_id
                or parent_case_id in current_proposal_sha_by_parent
                or _sha256_json(proposal.get("proposed_case") or {})
                != proposed_case_sha256
            ):
                raise RuntimeError("current blocker proposal identity is invalid")
            current_proposal_sha_by_parent[parent_case_id] = proposed_case_sha256
    candidate_by_id = {str(case["case_id"]): case for case in candidates}
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for mapping in request_map:
        custom_id = str(mapping["custom_id"])
        case_id = str(mapping["case_id"])
        parent_case_id = str(mapping["parent_case_id"])
        candidate = candidate_by_id.get(case_id)
        if candidate is None or _sha256_json(candidate) != mapping.get("candidate_case_sha256"):
            raise RuntimeError(f"changed candidate hash/mapping mismatch: {parent_case_id}")
        if (candidate.get("metadata") or {}).get("superseded_review_must_not_be_reused") is not True:
            raise RuntimeError(f"changed candidate lacks superseded-review guard: {parent_case_id}")
        metadata = candidate.get("metadata") or {}
        current_proposal_sha256 = current_proposal_sha_by_parent.get(parent_case_id)
        if (
            not current_proposal_sha256
            or mapping.get("proposed_case_sha256") != current_proposal_sha256
            or metadata.get("blocker_remediation_proposal_sha256")
            != current_proposal_sha256
        ):
            raise RuntimeError(
                f"changed candidate is not bound to current proposal: {parent_case_id}"
            )
        request = request_by_id.get(custom_id)
        if request is None:
            raise RuntimeError(f"changed request mapping has no request: {parent_case_id}")
        payload = json.loads(request["request"]["messages"][1]["content"])
        if (
            str(metadata.get("parent_case_id") or "") != parent_case_id
            or str(metadata.get("difficulty") or "") != str(mapping.get("difficulty") or "")
            or str(mapping.get("payload_sha256") or "")
            != str(request.get("payload_sha256") or "")
            or _sha256_json(payload) != str(request.get("payload_sha256") or "")
            or payload != _expected_review_payload(candidate)
            or _walk_forbidden(payload)
        ):
            raise RuntimeError(f"changed candidate/request/payload lineage mismatch: {parent_case_id}")
        if parent_case_id in result:
            raise RuntimeError(f"duplicate changed parent id: {parent_case_id}")
        result[parent_case_id] = (candidate, successful[custom_id])
    if len(result) != expected_changed_cases:
        raise RuntimeError("changed blocker rereview mapping count mismatch")
    return result


def _load_contract_rereviews(
    *,
    expected_prompt_hash: str,
    original_requests: dict[str, dict[str, Any]],
    original_parent_by_custom_id: dict[str, str],
) -> dict[str, dict[str, Any]]:
    audit = json.loads((ROOT / "review-block-union-audit.json").read_text(encoding="utf-8"))
    target_parent_ids = {
        str(value) for value in audit.get("contract_rereview_case_ids") or []
    }
    if not target_parent_ids:
        return {}
    prepared_summary = json.loads(
        (CONTRACT_REREVIEW_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    validate_contract_rereview_prepared_receipt(prepared_summary)
    execution_summary = json.loads(
        (CONTRACT_REREVIEW_ROOT / "execution-summary.json").read_text(encoding="utf-8")
    )
    request_map = json.loads(
        (CONTRACT_REREVIEW_ROOT / "request-map.json").read_text(encoding="utf-8")
    )
    requests = _read_jsonl(CONTRACT_REREVIEW_ROOT / "requests.jsonl")
    responses = _read_jsonl(CONTRACT_REREVIEW_ROOT / "responses.jsonl")
    expected = len(target_parent_ids)
    ordered_target_parent_ids = [
        str(value) for value in audit.get("contract_rereview_case_ids") or []
    ]
    if (
        prepared_summary.get("block_union_audit_sha256") != sha256_json(audit)
        or prepared_summary.get("parent_case_ids") != ordered_target_parent_ids
        or prepared_summary.get("parent_case_ids_sha256")
        != sha256_json(ordered_target_parent_ids)
    ):
        raise RuntimeError("same-semantic contract-rereview prepared receipt is stale")
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=expected_prompt_hash,
        expected_count=expected,
        expected_pack_sha256=str(prepared_summary.get("request_pack_sha256") or ""),
        custom_id_mode="contract_rereview",
        original_custom_ids_by_rereview_id={
            str(row.get("custom_id") or ""): str(row.get("original_custom_id") or "")
            for row in request_map
        },
        pack_name="same-semantic contract-rereview merge pack",
    )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=RUNNER_VERSION,
        transport=TRANSPORT,
        sse_completion_policy=SSE_COMPLETION_POLICY,
    )
    if (
        prepared_summary.get("execution_contract_sha256") != execution_contract
        or execution_summary.get("execution_contract_sha256") != execution_contract
    ):
        raise RuntimeError("same-semantic contract-rereview execution-contract drift")
    if (
        prepared_summary.get("semantic_payload_unchanged") is not True
        or prepared_summary.get("dataset_cases_changed") is not False
        or execution_summary.get("status") != "succeeded"
        or execution_summary.get("semantic_payload_unchanged") is not True
        or execution_summary.get("dataset_cases_changed") is not False
        or any(len(rows) != expected for rows in (request_map, requests))
    ):
        raise RuntimeError("same-semantic contract rereview is not complete/immutable")
    request_by_id = {str(row.get("custom_id") or ""): row for row in requests}
    if len(request_by_id) != expected or set(request_by_id) & set(original_requests):
        raise RuntimeError("contract-rereview request id invariant failed")
    mapping_by_id = {str(row.get("custom_id") or ""): row for row in request_map}
    if set(mapping_by_id) != set(request_by_id):
        raise RuntimeError("contract-rereview request mapping mismatch")

    successful = _formal_post150_success_ledger(
        responses,
        request_by_id,
        execution_contract_sha256=execution_contract,
        expected_prompt_hash=expected_prompt_hash,
        label="same-semantic contract-rereview merge ledger",
    )
    authoritative_successful = successful
    successful = {}
    for row in authoritative_successful.values():
        custom_id = str(row.get("custom_id") or "")
        if str(row.get("prompt_sha256") or "") != expected_prompt_hash:
            raise RuntimeError(f"contract-rereview response prompt hash drift: {custom_id}")
        review = dict(row.get("review") or {})
        payload = json.loads(request_by_id[custom_id]["request"]["messages"][1]["content"])
        errors = _validate_review(review)
        errors.extend(_validate_against_payload(review, payload))
        errors.extend(
            _validate_inverse_edge_canonicalization_audit(
                row, payload, source_response_rows=responses
            )
        )
        errors.extend(
            _validate_exact_variant_deduplication_audit(
                row, payload, source_response_rows=responses
            )
        )
        errors.extend(
            _validate_coverage_status_canonicalization_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_nonsemantic_canonicalization_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_quote_fragment_inverse_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_abstain_factual_constraint_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_redundant_field_coverage_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_unsupported_required_concept_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_empty_required_branch_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        errors.extend(
            _validate_saved_strict_revalidation_audit(
                row, payload, source_response_rows=responses
            )
        )
        if row.get("transport") == SAVED_RESPONSE_TRANSPORT:
            errors.extend(
                _validate_saved_post150_recovery(
                    row,
                    request=request_by_id[custom_id],
                    semantic_payload=payload,
                    source_response_rows=responses,
                    execution_contract_sha256=execution_contract,
                )
            )
        if any(
            str(item).strip()
            for item in row.get("structural_normalization") or []
        ):
            errors.append(
                "successful contract rereview required structural normalization"
            )
        if errors:
            raise RuntimeError(f"invalid contract-rereview {custom_id}: {'; '.join(errors)}")
        successful[custom_id] = row
    if set(successful) != set(request_by_id):
        raise RuntimeError(f"contract rereview incomplete: {len(successful)}/{expected}")

    result: dict[str, dict[str, Any]] = {}
    if set(original_parent_by_custom_id) != set(original_requests):
        raise RuntimeError("original request/parent lineage ids do not match")
    for custom_id, mapping in mapping_by_id.items():
        parent_case_id = str(mapping.get("parent_case_id") or "")
        original_custom_id = str(mapping.get("original_custom_id") or "")
        original = original_requests.get(original_custom_id)
        rereview = request_by_id[custom_id]
        if parent_case_id not in target_parent_ids or parent_case_id in result:
            raise RuntimeError("contract-rereview target lineage mismatch")
        if original is None:
            raise RuntimeError(f"missing original semantic request: {parent_case_id}")
        if (
            original_parent_by_custom_id.get(original_custom_id) != parent_case_id
            or str(rereview.get("prompt_sha256") or "") != expected_prompt_hash
            or rereview.get("request", {}).get("messages")
            != original.get("request", {}).get("messages")
            or str(rereview.get("payload_sha256") or "")
            != str(original.get("payload_sha256") or "")
            or mapping.get("semantic_payload_unchanged") is not True
            or mapping.get("dataset_case_unchanged") is not True
        ):
            raise RuntimeError(f"contract-rereview semantic payload drift: {parent_case_id}")
        result[parent_case_id] = successful[custom_id]
    if set(result) != target_parent_ids:
        raise RuntimeError("contract-rereview parent coverage mismatch")
    return result


def _label_map(items: Sequence[dict[str, Any]], prefix: str, value_key: str) -> dict[str, str]:
    return {f"{prefix}{index + 1}": str(item[value_key]) for index, item in enumerate(items)}


def _resolve_labels(
    labels: Sequence[str], mapping: dict[str, str], *, kind: str, custom_id: str
) -> list[str]:
    unknown = [str(label) for label in labels if str(label) not in mapping]
    if unknown:
        raise RuntimeError(f"{custom_id}: unknown {kind} label(s): {unknown}")
    return _dedupe(mapping[str(label)] for label in labels)


def _merge_contract(
    *, candidate_case: dict[str, Any], response: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    custom_id = str(response["custom_id"])
    review = dict(response["review"])
    evidence = [dict(item) for item in candidate_case.get("relevant_evidence") or []]
    evidence_map = _label_map(evidence, "E", "evidence_id")
    candidate_branches = list((candidate_case.get("answer_contract") or {}).get("required_branches") or [])
    branch_map = {
        f"B{index + 1}": str(branch.get("query_span") or "")
        for index, branch in enumerate(candidate_branches)
    }

    reviewer_concepts = list((review.get("required_concepts") or {}).get("all_of") or [])
    key_to_id: dict[str, str] = {}
    concepts: list[dict[str, Any]] = []
    for index, reviewer_concept in enumerate(reviewer_concepts):
        reviewer_key = str(reviewer_concept["concept_key"])
        if reviewer_key in key_to_id:
            raise RuntimeError(f"{custom_id}: duplicate concept_key {reviewer_key!r}")
        concept_id = f"c{index + 1:02d}"
        key_to_id[reviewer_key] = concept_id
        evidence_ids = _resolve_labels(
            reviewer_concept.get("evidence_labels") or [],
            evidence_map,
            kind="evidence",
            custom_id=custom_id,
        )
        branch_spans = _resolve_labels(
            reviewer_concept.get("branch_labels") or [],
            branch_map,
            kind="branch",
            custom_id=custom_id,
        )
        concepts.append(
            {
                "concept_id": concept_id,
                "canonical_text": str(reviewer_concept["canonical_text"]),
                "any_of": _dedupe(str(value) for value in reviewer_concept.get("any_of") or []),
                "derivation": {
                    "method": "full_suite_same_prompt_llm_review",
                    "reviewer_concept_key": reviewer_key,
                    "custom_id": custom_id,
                },
                "evidence_support": [
                    {"evidence_id": evidence_id, "method": "llm_reviewed_direct_support"}
                    for evidence_id in evidence_ids
                ],
                "query_branch_spans": branch_spans,
            }
        )

    is_abstain = str(review.get("answerability") or "") == "abstain"
    if is_abstain:
        if not concepts:
            abstain_key = "abstain_boundary"
            key_to_id[abstain_key] = "c01"
            concepts.append(
                {
                    "concept_id": "c01",
                    "canonical_text": "授权资料未提供回答所需信息，无法确定",
                    "any_of": [
                        "现有授权资料不足以回答",
                        "无法根据提供的资料确定",
                        "资料中没有提供相关信息，不能确定",
                    ],
                    "derivation": {
                        "method": "reviewed_answerability_abstention_boundary_v1",
                        "reviewer_concept_key": None,
                        "custom_id": custom_id,
                    },
                    "evidence_support": [],
                    "query_branch_spans": list(branch_map.values()),
                }
            )
        elif len(concepts) == 1:
            concepts[0]["evidence_support"] = []
            concepts[0]["query_branch_spans"] = list(branch_map.values())
        else:
            raise RuntimeError(f"{custom_id}: abstain contract may contain only one concept")

    constraints = dict(review.get("critical_constraints") or {})
    numerics = []
    for item in constraints.get("numerics") or []:
        numerics.append(
            {
                "value": str(item.get("value") or ""),
                "unit": item.get("unit"),
                "evidence_ids": _resolve_labels(
                    item.get("evidence_labels") or [],
                    evidence_map,
                    kind="evidence",
                    custom_id=custom_id,
                ),
                "required_exact": True,
            }
        )
    identifiers = []
    for item in constraints.get("identifiers") or []:
        identifiers.append(
            {
                "value": str(item.get("value") or ""),
                "evidence_ids": _resolve_labels(
                    item.get("evidence_labels") or [],
                    evidence_map,
                    kind="evidence",
                    custom_id=custom_id,
                ),
                "required_exact": True,
            }
        )
    if is_abstain and (numerics or identifiers):
        raise RuntimeError(f"{custom_id}: abstain contract cannot require factual constraints")

    required_branches = []
    for item in review.get("required_branches") or []:
        branch_label = str(item.get("branch_label") or "")
        if branch_label not in branch_map:
            raise RuntimeError(f"{custom_id}: unknown branch label {branch_label!r}")
        reviewer_keys = [str(value) for value in item.get("required_concept_keys") or []]
        unknown_keys = [value for value in reviewer_keys if value not in key_to_id]
        if unknown_keys:
            raise RuntimeError(f"{custom_id}: branch references unknown concept keys {unknown_keys}")
        required_branches.append(
            {
                "branch_id": f"b{len(required_branches) + 1:02d}",
                "query_span": branch_map[branch_label],
                "required_concept_ids": _dedupe(key_to_id[value] for value in reviewer_keys),
                "assignment_method": "full_suite_same_prompt_llm_review",
            }
        )
    if is_abstain:
        abstain_concept_id = str(concepts[0]["concept_id"])
        required_branches = [
            {
                "branch_id": f"b{index + 1:02d}",
                "query_span": query_span,
                "required_concept_ids": [abstain_concept_id],
                "assignment_method": "reviewed_answerability_abstention_boundary_v1",
                "answerability": "must_abstain",
            }
            for index, query_span in enumerate(branch_map.values())
        ]

    evidence_coverage = []
    for item in review.get("evidence_concept_coverage") or []:
        evidence_label = str(item.get("evidence_label") or "")
        if evidence_label not in evidence_map:
            raise RuntimeError(f"{custom_id}: unknown coverage evidence label {evidence_label!r}")
        reviewer_keys = [str(value) for value in item.get("required_concept_keys") or []]
        unknown_keys = [value for value in reviewer_keys if value not in key_to_id]
        if unknown_keys:
            raise RuntimeError(f"{custom_id}: coverage references unknown concept keys {unknown_keys}")
        source_evidence = evidence[list(evidence_map).index(evidence_label)]
        evidence_coverage.append(
            {
                "evidence_id": evidence_map[evidence_label],
                "chunk_id": str(source_evidence.get("chunk_id") or ""),
                "page": source_evidence.get("page"),
                "required_concept_ids": _dedupe(key_to_id[value] for value in reviewer_keys),
                "coverage_status": str(item.get("status") or ""),
            }
        )
    if is_abstain:
        evidence_coverage = [
            {
                "evidence_id": str(source_evidence.get("evidence_id") or ""),
                "chunk_id": str(source_evidence.get("chunk_id") or ""),
                "page": source_evidence.get("page"),
                "required_concept_ids": [],
                "coverage_status": "not_applicable_to_abstention",
            }
            for source_evidence in evidence
        ]

    llm_verdict = str(review["verdict"])
    visual_audit = deepcopy(
        (candidate_case.get("answer_contract") or {}).get("source_visual_audit")
    )
    deterministic_review = deepcopy(
        (candidate_case.get("answer_contract") or {}).get("review") or {}
    )
    visual_status = str((visual_audit or {}).get("status") or "")
    visual_block = visual_status in {
        "confirmed_reference_source_conflict",
        "confirmed_reference_incomplete",
    }
    verdict = "block" if visual_block else llm_verdict
    independent_status = (
        "blocked_by_confirmed_source_visual_audit"
        if visual_block
        else "required_visual_review"
        if visual_status == "manual_visual_review_required"
        else "pending"
    )
    contract = {
        "schema_version": REVIEWED_CONTRACT_SCHEMA,
        "contract_status": "llm_reviewed_blocked" if verdict == "block" else "llm_reviewed_candidate",
        "answerability": {
            "required": str(review["answerability"]),
            "unsupported_factual_additions_forbidden": True,
            "contradictory_claims_forbidden": True,
        },
        "required_concepts": {"all_of": concepts},
        "critical_constraints": {
            "numerics": numerics,
            "identifiers": identifiers,
            "polarity": dict(constraints.get("polarity") or {}),
        },
        "required_branches": required_branches,
        "evidence_concept_coverage": evidence_coverage,
        "review": {
            "status": "llm_review_complete",
            "verdict": verdict,
            "llm_verdict": llm_verdict,
            "source_visual_override": visual_block,
            "model": response.get("model"),
            "runner_version": response.get("runner_version") or "legacy-unversioned-runner",
            "prompt_sha256": response.get("prompt_sha256"),
            "custom_id": custom_id,
            "response_sha256": _sha256_json(review),
            "review_notes": [str(value) for value in review.get("review_notes") or []],
            "independent_validation_status": independent_status,
            "deterministic_candidate_review": deterministic_review,
        },
    }
    if visual_audit is not None:
        contract["source_visual_audit"] = visual_audit
    qa = {
        "case_id": candidate_case["case_id"],
        "parent_case_id": (candidate_case.get("metadata") or {}).get("parent_case_id"),
        "difficulty": (candidate_case.get("metadata") or {}).get("difficulty"),
        "custom_id": custom_id,
        "verdict": verdict,
        "llm_verdict": llm_verdict,
        "source_visual_override": visual_block,
        "visual_audit_status": visual_status or None,
        "independent_validation_status": independent_status,
        "deterministic_candidate_readiness": deterministic_review.get("readiness"),
        "deterministic_review_flags": [
            str(item.get("code") or "") for item in deterministic_review.get("flags") or []
        ],
        "deterministic_hard_blockers": [
            str(item.get("code") or "")
            for item in deterministic_review.get("hard_blockers") or []
        ],
        "answerability": review["answerability"],
        "concept_count": len(concepts),
        "branch_count": len(required_branches),
        "evidence_count": len(evidence),
        "evidence_covered_count": sum(bool(item["required_concept_ids"]) for item in evidence_coverage),
        "numeric_count": len(numerics),
        "identifier_count": len(identifiers),
        "polarity_required": (constraints.get("polarity") or {}).get("required"),
        "prompt_sha256": response.get("prompt_sha256"),
        "runner_version": response.get("runner_version") or "legacy-unversioned-runner",
        "response_sha256": _sha256_json(review),
        "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
        "total_tokens": (response.get("usage") or {}).get("total_tokens"),
        "review_notes": [str(value) for value in review.get("review_notes") or []],
    }
    return contract, qa


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _checksums() -> None:
    checksum_path = ROOT / "SHA256SUMS"
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != checksum_path
        and "__pycache__" not in path.parts
        and not path.name.endswith(".tmp")
    )
    checksum_path.write_text(
        "\n".join(f"{_sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths)
        + "\n",
        encoding="utf-8",
    )


def _reviewed_contract_schema() -> dict[str, Any]:
    schema = json.loads(
        (ROOT / "schema" / "answer-contract-v2.schema.json").read_text(encoding="utf-8")
    )
    schema["$id"] = "stt.eval.answer-contract.v2-llm-reviewed-candidate.schema.json"
    schema["title"] = "STT evaluation answer contract v2 LLM-reviewed candidate"
    schema["properties"]["schema_version"] = {"const": REVIEWED_CONTRACT_SCHEMA}
    schema["properties"]["contract_status"] = {
        "enum": ["llm_reviewed_candidate", "llm_reviewed_blocked"]
    }
    return schema


def merge() -> dict[str, Any]:
    block_union_audit = json.loads(
        (ROOT / "review-block-union-audit.json").read_text(encoding="utf-8")
    )
    if block_union_audit.get("status") != "passed":
        raise RuntimeError("final reviewer/visual/materialization audit has not passed")
    validate_block_union_freshness(
        block_union_audit,
        requests_path=REVIEW_ROOT / "requests.jsonl",
        responses_path=REVIEW_ROOT / "responses.jsonl",
        visual_audit_path=VISUAL_AUDIT_PATH,
        base_proposals_path=BASE_PROPOSALS_PATH,
        supplemental_proposals_path=SUPPLEMENTAL_PROPOSALS_PATH,
        label="merge block-union audit",
    )
    proposal_validation = json.loads(
        BASE_PROPOSAL_VALIDATION_PATH.read_text(encoding="utf-8")
    )
    validate_proposal_validation_binding(
        proposal_validation,
        proposals_path=BASE_PROPOSALS_PATH,
        label="merge base proposal validation",
    )
    reviews, prompt_hash = _load_complete_reviews()
    original_requests = {
        str(row["custom_id"]): row for row in _read_jsonl(REVIEW_ROOT / "requests.jsonl")
    }
    request_map = json.loads((REVIEW_ROOT / "request-map.json").read_text(encoding="utf-8"))
    custom_to_lineage = {str(item["custom_id"]): item for item in request_map}
    if set(custom_to_lineage) != set(reviews):
        raise RuntimeError("request map and successful review ids do not match exactly")
    original_parent_by_custom_id = {
        custom_id: str(lineage.get("parent_case_id") or "")
        for custom_id, lineage in custom_to_lineage.items()
    }
    blocker_rereviews = _load_blocker_rereviews(
        expected_prompt_hash=prompt_hash,
        original_request_ids=set(reviews),
    )
    contract_rereviews = _load_contract_rereviews(
        expected_prompt_hash=prompt_hash,
        original_requests=original_requests,
        original_parent_by_custom_id=original_parent_by_custom_id,
    )
    contract_rereview_audit: dict[str, Any] = {
        "status": "passed",
        "case_count": 0,
        "reviewer_contract_failure_count": 0,
    }
    if contract_rereviews:
        contract_rereview_audit = json.loads(
            (CONTRACT_REREVIEW_ROOT / "audit.json").read_text(encoding="utf-8")
        )
        audit_inputs = contract_rereview_audit.get("inputs") or {}
        if (
            contract_rereview_audit.get("status") != "passed"
            or contract_rereview_audit.get("case_count") != len(contract_rereviews)
            or contract_rereview_audit.get("reviewer_contract_failure_count") != 0
            or audit_inputs.get("requests_sha256")
            != _sha256_file(CONTRACT_REREVIEW_ROOT / "requests.jsonl")
            or audit_inputs.get("request_map_sha256")
            != _sha256_file(CONTRACT_REREVIEW_ROOT / "request-map.json")
            or audit_inputs.get("responses_sha256")
            != _sha256_file(CONTRACT_REREVIEW_ROOT / "responses.jsonl")
            or audit_inputs.get("execution_summary_sha256")
            != _sha256_file(CONTRACT_REREVIEW_ROOT / "execution-summary.json")
        ):
            raise RuntimeError("same-semantic contract-rereview audit is stale or failed")
    if set(blocker_rereviews) & set(contract_rereviews):
        raise RuntimeError("data-remediation and same-semantic contract-rereview queues overlap")
    if set(blocker_rereviews) != set(
        str(value)
        for value in block_union_audit.get("required_data_remediation_case_ids") or []
    ):
        raise RuntimeError("data-remediation rereviews do not match final block-union audit")
    if set(contract_rereviews) != set(
        str(value) for value in block_union_audit.get("contract_rereview_case_ids") or []
    ):
        raise RuntimeError("same-semantic contract rereviews do not match final audit")
    review_schema_extra_parent_ids = {
        str(value)
        for value in block_union_audit.get("review_schema_extra_field_case_ids") or []
    }
    if not review_schema_extra_parent_ids.issubset(set(contract_rereviews)):
        raise RuntimeError(
            "schema-external original reviews must be superseded by strict same-semantic rereview"
        )
    structural_normalization_parent_ids = {
        str(value)
        for value in block_union_audit.get("structural_normalization_case_ids") or []
    }
    if not structural_normalization_parent_ids.issubset(set(contract_rereviews)):
        raise RuntimeError(
            "structurally normalized original reviews must be superseded by strict rereview"
        )
    manifests: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []
    used_blocker_rereviews: set[str] = set()
    used_contract_rereviews: set[str] = set()
    for difficulty in DIFFICULTIES:
        candidate_path = ROOT / "datasets" / f"{difficulty}.v2.json"
        candidate_manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
        reviewed_manifest = deepcopy(candidate_manifest)
        reviewed_cases = []
        for candidate_case in candidate_manifest["cases"]:
            case_id = str(candidate_case["case_id"])
            parent_case_id = str((candidate_case.get("metadata") or {}).get("parent_case_id") or "")
            matching = [
                custom_id
                for custom_id, lineage in custom_to_lineage.items()
                if str(lineage["case_id"]) == case_id
            ]
            if len(matching) != 1:
                raise RuntimeError(f"expected one review mapping for {case_id}, found {matching}")
            superseded_original_review_id = matching[0]
            if parent_case_id in blocker_rereviews:
                active_candidate_case, response = blocker_rereviews[parent_case_id]
                selected_response_log = _read_jsonl(
                    BLOCKER_REREVIEW_ROOT / "responses.jsonl"
                )
                used_blocker_rereviews.add(parent_case_id)
            elif parent_case_id in contract_rereviews:
                active_candidate_case = candidate_case
                response = contract_rereviews[parent_case_id]
                selected_response_log = _read_jsonl(
                    CONTRACT_REREVIEW_ROOT / "responses.jsonl"
                )
                used_contract_rereviews.add(parent_case_id)
            else:
                active_candidate_case = candidate_case
                response = reviews[superseded_original_review_id]
                selected_response_log = _read_jsonl(
                    REVIEW_ROOT / "responses.jsonl"
                )
            active_payload = _expected_review_payload(active_candidate_case)
            selected_review = dict(response.get("review") or {})
            selected_errors = _validate_review(selected_review)
            selected_errors.extend(
                _validate_against_payload(selected_review, active_payload)
            )
            selected_errors.extend(
                _validate_inverse_edge_canonicalization_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_exact_variant_deduplication_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_coverage_status_canonicalization_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_nonsemantic_canonicalization_composition_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_quote_fragment_inverse_composition_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_abstain_factual_constraint_recovery_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_redundant_field_coverage_composition_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_unsupported_required_concept_block_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_empty_required_branch_block_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            selected_errors.extend(
                _validate_saved_strict_revalidation_audit(
                    response,
                    active_payload,
                    source_response_rows=selected_response_log,
                )
            )
            if selected_errors:
                raise RuntimeError(
                    f"selected response is not strictly adoptable for {case_id}: "
                    + "; ".join(selected_errors)
                )
            contract, qa = _merge_contract(
                candidate_case=active_candidate_case,
                response=response,
            )
            _validate_contract_coverage(
                contract,
                active_candidate_case.get("relevant_evidence") or [],
                case_id=case_id,
                expected_branch_spans=[
                    str(branch.get("query_span") or "")
                    for branch in (
                        active_candidate_case.get("answer_contract") or {}
                    ).get("required_branches")
                    or []
                ],
            )
            contract_hash = _sha256_json(contract)
            concept_groups, variant_filter_audit = _materialize_concept_groups(
                contract,
                case_id=case_id,
                evidence=active_candidate_case.get("relevant_evidence") or [],
            )
            concept_groups_hash = _sha256_json(concept_groups)
            reviewed_case = deepcopy(active_candidate_case)
            reviewed_case["answer_contract"] = contract
            reviewed_case["concept_groups"] = concept_groups
            reviewed_case["metadata"] = {
                **dict(reviewed_case.get("metadata") or {}),
                "reviewed_contract_hash": contract_hash,
                "concept_groups_hash": concept_groups_hash,
                "concept_groups_source_contract_hash": contract_hash,
                "concept_groups_materializer": "merge_reviewed_contracts.py:v1",
                "concept_groups_variant_filter": variant_filter_audit,
            }
            reviewed_cases.append(reviewed_case)
            qa["answer_contract_hash"] = contract_hash
            qa["concept_groups_hash"] = concept_groups_hash
            qa["concept_groups_source_contract_hash"] = contract_hash
            qa["concept_group_count"] = len(concept_groups["all_of"])
            qa["filtered_weak_variant_count"] = variant_filter_audit[
                "removed_variant_count"
            ]
            qa["filtered_weak_variants"] = variant_filter_audit["removed_variants"]
            qa["blocker_remediation_applied"] = parent_case_id in blocker_rereviews
            qa["contract_rereview_applied"] = parent_case_id in contract_rereviews
            qa["contract_rereview_semantic_payload_unchanged"] = (
                True if parent_case_id in contract_rereviews else None
            )
            qa["contract_rereview_dataset_case_unchanged"] = (
                True if parent_case_id in contract_rereviews else None
            )
            qa["superseded_original_review_custom_id"] = (
                superseded_original_review_id
                if parent_case_id in blocker_rereviews or parent_case_id in contract_rereviews
                else None
            )
            qa["superseded_original_review_reused"] = (
                False
                if parent_case_id in blocker_rereviews or parent_case_id in contract_rereviews
                else None
            )
            qa["original_review_schema_extra_paths"] = (
                (block_union_audit.get("review_schema_extra_paths_by_case") or {}).get(
                    parent_case_id, []
                )
            )
            qa["original_schema_extra_semantics_reused"] = (
                False if parent_case_id in review_schema_extra_parent_ids else None
            )
            qa["original_review_structural_normalization_actions"] = (
                (block_union_audit.get("structural_normalization_actions_by_case") or {}).get(
                    parent_case_id, []
                )
            )
            qa["original_normalized_semantics_reused"] = (
                False if parent_case_id in structural_normalization_parent_ids else None
            )
            qa["same_response_inverse_edge_canonicalization"] = deepcopy(
                response.get("same_response_inverse_edge_canonicalization")
            )
            qa["same_response_exact_variant_deduplication"] = deepcopy(
                response.get("same_response_exact_variant_deduplication")
            )
            qa["post150_materialization_adoption"] = (
                materialization_adoption_for_authoritative_success(
                    selected_response_log, response
                )
                if parent_case_id in blocker_rereviews
                or parent_case_id in contract_rereviews
                else None
            )
            qa["same_response_coverage_status_canonicalization"] = deepcopy(
                response.get("same_response_coverage_status_canonicalization")
            )
            qa["same_response_nonsemantic_canonicalization_composition"] = deepcopy(
                response.get(
                    "same_response_nonsemantic_canonicalization_composition"
                )
            )
            qa[
                "same_response_quote_fragment_inverse_canonicalization_composition"
            ] = deepcopy(
                response.get(
                    "same_response_quote_fragment_inverse_canonicalization_composition"
                )
            )
            qa[
                "saved_response_exact_payload_branch_span_quote_prefix_recovery"
            ] = deepcopy(
                response.get(
                    "saved_response_exact_payload_branch_span_quote_prefix_recovery"
                )
            )
            qa["saved_response_abstain_factual_constraint_recovery"] = deepcopy(
                response.get(
                    "saved_response_abstain_factual_constraint_recovery"
                )
            )
            qa["saved_response_redundant_field_coverage_composition"] = deepcopy(
                response.get(
                    "saved_response_redundant_field_coverage_composition"
                )
            )
            qa["same_response_unsupported_required_concept_block"] = deepcopy(
                response.get("same_response_unsupported_required_concept_block")
            )
            qa["multi_response_empty_required_branch_block"] = deepcopy(
                response.get("multi_response_empty_required_branch_block")
            )
            qa["saved_response_strict_revalidation"] = deepcopy(
                response.get("saved_response_strict_revalidation")
            )
            qa["saved_response_same_concept_variant_subset_recovery"] = deepcopy(
                response.get(
                    "saved_response_same_concept_variant_subset_recovery"
                )
            )
            qa["saved_response_quote_key_branch_span_recovery"] = deepcopy(
                response.get(
                    "saved_response_quote_key_branch_span_recovery"
                )
            )
            qa["difficulty"] = difficulty
            qa_rows.append(qa)
        reviewed_manifest["schema_version"] = "stt.eval.dataset.v2-llm-reviewed-candidate"
        reviewed_manifest["version"] = REVIEWED_VERSION
        reviewed_manifest["suite_key"] = REVIEWED_SUITE_KEY
        reviewed_manifest["dataset_key"] = REVIEWED_DATASET_KEYS[difficulty]
        reviewed_manifest["import_allowed"] = False
        reviewed_manifest["baseline_registration_allowed"] = False
        reviewed_manifest["generation"] = {
            **dict(reviewed_manifest.get("generation") or {}),
            "llm_review_status": "complete",
            "llm_review_prompt_sha256": prompt_hash,
            "independent_validation_status": "pending",
            "answer_contract_to_concept_groups_materialization_status": "complete",
        }
        reviewed_manifest["gate_policy"] = {
            **dict(reviewed_manifest.get("gate_policy") or {}),
            "case_pass_policy": deepcopy(FROZEN_CASE_PASS_POLICY),
            "case_pass_policy_activation_status": "reserved_not_active",
            "case_pass_policy_authored_contract_field": "concept_groups",
            "case_pass_policy_materialization_required": True,
        }
        reviewed_manifest["cases"] = reviewed_cases
        manifests[difficulty] = reviewed_manifest

    if used_blocker_rereviews != set(blocker_rereviews):
        raise RuntimeError("not every changed blocker rereview was applied exactly once")
    if used_contract_rereviews != set(contract_rereviews):
        raise RuntimeError("not every same-semantic contract rereview was applied exactly once")

    if len(qa_rows) != EXPECTED_CASES or len({row["case_id"] for row in qa_rows}) != EXPECTED_CASES:
        raise RuntimeError("reviewed QA must contain exactly 150 unique cases")
    verdict_counts = Counter(str(row["verdict"]) for row in qa_rows)
    difficulty_counts = {
        difficulty: dict(
            Counter(str(row["verdict"]) for row in qa_rows if row["difficulty"] == difficulty)
        )
        for difficulty in DIFFICULTIES
    }
    usage = {
        key: sum(int(row.get(key) or 0) for row in qa_rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    inverse_edge_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("same_response_inverse_edge_canonicalization"), dict
        )
    ]
    exact_dedupe_rows = [
        row
        for row in qa_rows
        if isinstance(row.get("same_response_exact_variant_deduplication"), dict)
    ]
    coverage_status_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("same_response_coverage_status_canonicalization"), dict
        )
    ]
    nonsemantic_composition_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("same_response_nonsemantic_canonicalization_composition"),
            dict,
        )
    ]
    quote_fragment_composition_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get(
                "same_response_quote_fragment_inverse_canonicalization_composition"
            ),
            dict,
        )
    ]
    exact_span_quote_prefix_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get(
                "saved_response_exact_payload_branch_span_quote_prefix_recovery"
            ),
            dict,
        )
    ]
    abstain_factual_constraint_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("saved_response_abstain_factual_constraint_recovery"),
            dict,
        )
    ]
    redundant_field_composition_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("saved_response_redundant_field_coverage_composition"),
            dict,
        )
    ]
    unsupported_block_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("same_response_unsupported_required_concept_block"), dict
        )
    ]
    empty_branch_block_rows = [
        row
        for row in qa_rows
        if isinstance(row.get("multi_response_empty_required_branch_block"), dict)
    ]
    saved_strict_rows = [
        row
        for row in qa_rows
        if isinstance(row.get("saved_response_strict_revalidation"), dict)
    ]
    saved_variant_subset_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("saved_response_same_concept_variant_subset_recovery"),
            dict,
        )
    ]
    saved_quote_key_rows = [
        row
        for row in qa_rows
        if isinstance(
            row.get("saved_response_quote_key_branch_span_recovery"),
            dict,
        )
    ]
    materialization_adoption_rows = [
        row
        for row in qa_rows
        if isinstance(row.get("post150_materialization_adoption"), dict)
    ]
    summary = {
        "schema_version": "stt.eval.suite-v2-concept-review-summary.v1",
        "status": "reviewed_candidate_pending_independent_validation",
        "case_count": len(qa_rows),
        "prompt_sha256": prompt_hash,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "llm_verdict_counts": dict(
            sorted(Counter(str(row["llm_verdict"]) for row in qa_rows).items())
        ),
        "difficulty_verdict_counts": difficulty_counts,
        "blocked_case_count": verdict_counts.get("block", 0),
        "revised_case_count": verdict_counts.get("revise", 0),
        "independent_validation_pending_count": len(qa_rows),
        "concept_groups_materialized_case_count": sum(
            int(row.get("concept_group_count") or 0) > 0 for row in qa_rows
        ),
        "concept_groups_materialization_status": "complete",
        "filtered_weak_variant_count": sum(
            int(row.get("filtered_weak_variant_count") or 0) for row in qa_rows
        ),
        "filtered_weak_variant_case_ids": [
            str(row["case_id"])
            for row in qa_rows
            if int(row.get("filtered_weak_variant_count") or 0) > 0
        ],
        "discriminative_variant_policy_version": (
            MATERIALIZATION_POLICY_VERSION
        ),
        "post150_materialization_adoption_policy": MATERIALIZATION_ADOPTION_POLICY,
        "post150_materialization_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "post150_materialization_global_fresh_retry_quota": (
            MATERIALIZATION_RETRY_QUOTA
        ),
        "post150_materialization_global_fresh_retry_consumed_count": len(
            materialization_adoption_rows
        ),
        "post150_formal_stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "post150_formal_stream_content_sha256_authoritative": False,
        "post150_formal_stream_content_chars_authoritative": False,
        "post150_formal_stream_event_counts_authoritative": False,
        "post150_materialization_adopted_case_count": len(
            materialization_adoption_rows
        ),
        "post150_materialization_adopted_case_ids": sorted(
            str(row["case_id"]) for row in materialization_adoption_rows
        ),
        "post150_materialization_adopted_episode_ids": sorted(
            str(row["post150_materialization_adoption"]["episode_id"])
            for row in materialization_adoption_rows
        ),
        "post150_materialization_adoption_by_case_id": {
            str(row["case_id"]): row["post150_materialization_adoption"]
            for row in sorted(
                materialization_adoption_rows,
                key=lambda item: str(item["case_id"]),
            )
        },
        "reserved_case_pass_policy_version": "stt.eval.case.v3",
        "blocker_remediation_applied_case_count": sum(
            bool(row.get("blocker_remediation_applied")) for row in qa_rows
        ),
        "blocker_rereview_case_count": len(blocker_rereviews),
        "blocker_rereview_prompt_sha256": prompt_hash,
        "reviewer_block_case_ids": block_union_audit.get("reviewer_block_case_ids"),
        "reviewer_block_union_visual_block": block_union_audit.get(
            "reviewer_block_union_visual_block"
        ),
        "data_blocker_case_count": len(blocker_rereviews),
        "contract_rereview_case_count": len(contract_rereviews),
        "contract_rereview_case_ids": block_union_audit.get(
            "contract_rereview_case_ids"
        ),
        "contract_rereview_prompt_sha256": prompt_hash,
        "review_schema_extra_field_case_ids": sorted(
            review_schema_extra_parent_ids
        ),
        "review_schema_extra_disposition": (
            "same_semantic_rereview_completed_no_projection_or_semantic_reuse"
        ),
        "structural_normalization_case_ids": sorted(
            structural_normalization_parent_ids
        ),
        "structural_normalization_disposition": (
            "same_semantic_rereview_completed_original_normalized_semantics_not_reused"
        ),
        "same_response_inverse_edge_canonicalization_case_count": len(
            inverse_edge_rows
        ),
        "same_response_inverse_edge_canonicalization_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in inverse_edge_rows
        ),
        "same_response_inverse_edge_canonicalization_disposition": (
            "adopted_only_after_reproducible_same_response_inverse_edge_validation"
        ),
        "same_response_exact_variant_deduplication_case_count": len(
            exact_dedupe_rows
        ),
        "same_response_exact_variant_deduplication_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in exact_dedupe_rows
        ),
        "same_response_exact_variant_deduplication_disposition": (
            "adopted_only_after_reproducible_normalized_exact_deduplication"
        ),
        "same_response_coverage_status_canonicalization_case_count": len(
            coverage_status_rows
        ),
        "same_response_coverage_status_canonicalization_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in coverage_status_rows
        ),
        "same_response_coverage_status_canonicalization_disposition": (
            "adopted_only_after_reproducible_finite_zero-support_status_alias"
        ),
        "same_response_nonsemantic_canonicalization_composition_case_count": len(
            nonsemantic_composition_rows
        ),
        "same_response_nonsemantic_canonicalization_composition_parent_case_ids": sorted(
            str(row["parent_case_id"])
            for row in nonsemantic_composition_rows
        ),
        "same_response_nonsemantic_canonicalization_composition_disposition": (
            "adopted_only_after_fixed_order_disjoint_saved_response_replay"
        ),
        "same_response_quote_fragment_inverse_canonicalization_composition_case_count": len(
            quote_fragment_composition_rows
        ),
        "same_response_quote_fragment_inverse_canonicalization_composition_parent_case_ids": sorted(
            str(row["parent_case_id"])
            for row in quote_fragment_composition_rows
        ),
        "same_response_quote_fragment_inverse_canonicalization_composition_disposition": (
            "adopted_only_after_unique_exact_payload_quote_reconstruction_and_"
            "inverse_branch_replay"
        ),
        "saved_response_exact_payload_branch_span_quote_prefix_recovery_case_count": len(
            exact_span_quote_prefix_rows
        ),
        "saved_response_exact_payload_branch_span_quote_prefix_recovery_parent_case_ids": sorted(
            str(row["parent_case_id"])
            for row in exact_span_quote_prefix_rows
        ),
        "saved_response_exact_payload_branch_span_quote_prefix_recovery_disposition": (
            "format_only_final_complete_attempt_exact_payload_span_replacement_"
            "without_semantic_consensus_or_cross_attempt_merge"
        ),
        "saved_response_abstain_factual_constraint_recovery_case_count": len(
            abstain_factual_constraint_rows
        ),
        "saved_response_abstain_factual_constraint_recovery_parent_case_ids": sorted(
            str(row["parent_case_id"])
            for row in abstain_factual_constraint_rows
        ),
        "saved_response_abstain_factual_constraint_recovery_disposition": (
            "format_only_final_complete_attempt_abstention_constraint_removal_"
            "without_semantic_consensus_or_cross_attempt_merge"
        ),
        "saved_response_redundant_field_coverage_composition_case_count": len(
            redundant_field_composition_rows
        ),
        "saved_response_redundant_field_coverage_composition_parent_case_ids": sorted(
            str(row["parent_case_id"])
            for row in redundant_field_composition_rows
        ),
        "saved_response_redundant_field_coverage_composition_disposition": (
            "format_only_final_complete_attempt_adopted_without_semantic_consensus_"
            "or_cross_attempt_merge"
        ),
        "same_response_unsupported_required_concept_block_case_count": len(
            unsupported_block_rows
        ),
        "same_response_unsupported_required_concept_block_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in unsupported_block_rows
        ),
        "same_response_unsupported_required_concept_block_disposition": (
            "data_remediation_required_original_block_never_runnable"
        ),
        "multi_response_empty_required_branch_block_case_count": len(
            empty_branch_block_rows
        ),
        "multi_response_empty_required_branch_block_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in empty_branch_block_rows
        ),
        "multi_response_empty_required_branch_block_disposition": (
            "branch_segmentation_data_remediation_and_new_payload_rereview_required_"
            "original_block_never_runnable"
        ),
        "saved_response_strict_revalidation_case_count": len(saved_strict_rows),
        "saved_response_strict_revalidation_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in saved_strict_rows
        ),
        "saved_response_strict_revalidation_disposition": (
            "adopted_unchanged_after_strict_policy_fix_and_hash_revalidation"
        ),
        "saved_response_same_concept_variant_subset_recovery_case_count": len(
            saved_variant_subset_rows
        ),
        "saved_response_same_concept_variant_subset_recovery_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in saved_variant_subset_rows
        ),
        "saved_response_same_concept_variant_subset_recovery_disposition": (
            "adopted_only_after_exact_latest_failed_row_same_concept_existing_"
            "any_of_subset_replay_without_api_consensus_or_cross_attempt_merge"
        ),
        "saved_response_quote_key_branch_span_recovery_case_count": len(
            saved_quote_key_rows
        ),
        "saved_response_quote_key_branch_span_recovery_parent_case_ids": sorted(
            str(row["parent_case_id"]) for row in saved_quote_key_rows
        ),
        "saved_response_quote_key_branch_span_recovery_disposition": (
            "adopted_only_after_exact_latest_failed_row_quoted_schema_key_"
            "removal_and_same_request_exact_span_replay_without_api_consensus_"
            "or_cross_attempt_merge"
        ),
        "contract_rereview_semantic_payload_unchanged": True,
        "contract_rereview_dataset_cases_changed": False,
        "contract_rereview_failure_count": contract_rereview_audit.get(
            "reviewer_contract_failure_count"
        ),
        "superseded_original_review_reused": False,
        "superseded_original_review_reused_for_rereviewed_cases": False,
        "confirmed_source_visual_override_count": sum(
            bool(row.get("source_visual_override")) for row in qa_rows
        ),
        "usage": usage,
        "price_amount": None,
        "import_allowed": False,
        "baseline_registration_allowed": False,
        "remaining_gate": (
            "independent source/contract validation must resolve every effective block and "
            "validate every revised or approved contract across the complete 150-case manifests"
        ),
    }
    suite = json.loads((ROOT / "suite-v2.json").read_text(encoding="utf-8"))
    reviewed_suite = {
        **suite,
        "schema_version": REVIEWED_SUITE_SCHEMA,
        "suite_key": REVIEWED_SUITE_KEY,
        "version": REVIEWED_VERSION,
        "status": summary["status"],
        "import_allowed": False,
        "baseline_registration_allowed": False,
        "datasets": {
            difficulty: {
                "path": f"datasets/{difficulty}.reviewed.json",
                "dataset_key": manifest["dataset_key"],
                "version": manifest["version"],
                "case_count": len(manifest["cases"]),
                "checksum": _sha256_json(manifest),
                "deterministic_candidate_path": f"../datasets/{difficulty}.v2.json",
                "blocker_remediation_candidate_path": "../blocker-rereview/candidates.json",
            }
            for difficulty, manifest in manifests.items()
        },
        "review_gate": {
            **dict(suite.get("review_gate") or {}),
            "full_150_same_prompt_review_complete": True,
            "changed_blocker_same_prompt_rereview_complete": True,
            "changed_blocker_superseded_review_reused": False,
            "weak_contract_same_semantic_rereview_complete": True,
            "weak_contract_dataset_cases_changed": False,
            "weak_contract_superseded_review_reused": False,
            "review_prompt_sha256": prompt_hash,
            "independent_validation_complete": False,
        },
        "activation_gate": {
            "frontend_binding_allowed": False,
            "frontend_binding_changed_by_this_build": False,
            "evaluation_import_allowed": False,
            "baseline_registration_allowed": False,
            "full_150_reviewer_responses_parsed": True,
            "blocker_remediation_applied_case_count": len(blocker_rereviews),
            "changed_blocker_cases_independently_rereviewed": True,
            "weak_contract_cases_independently_rereviewed": len(contract_rereviews),
            "weak_contract_semantic_payload_unchanged": True,
            "answer_contract_to_concept_groups_materialization_complete": True,
            "effective_blocker_count": verdict_counts.get("block", 0),
            "all_effective_blockers_resolved": verdict_counts.get("block", 0) == 0,
            "independent_schema_validation_required": True,
            "reason": "reviewed candidate still contains unresolved blocker/independent-validation gates",
        },
    }

    for difficulty, manifest in manifests.items():
        _write_json(REVIEWED_ROOT / "datasets" / f"{difficulty}.reviewed.json", manifest)
    _write_json(REVIEWED_ROOT / "suite-v2-reviewed.json", reviewed_suite)
    _write_json(REVIEWED_ROOT / "qa-reviewed.json", qa_rows)
    _write_csv(REVIEWED_ROOT / "qa-reviewed.csv", qa_rows)
    _write_json(REVIEWED_ROOT / "summary.json", summary)
    _write_json(ROOT / "schema" / "answer-contract-v2-reviewed.schema.json", _reviewed_contract_schema())
    _checksums()
    return summary


def main() -> int:
    try:
        result = merge()
    except RuntimeError as exc:
        print(
            json.dumps(
                {"status": "refused", "reason": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    formal_rereview_success_ledger,
