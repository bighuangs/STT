"""Validate deterministic and, when present, reviewed suite-v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
from jsonschema import Draft202012Validator  # noqa: E402
from review_contracts import (  # noqa: E402
    POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
    POST150_SAVED_RECOVERY_RUNNER_VERSION,
    RUNNER_VERSION,
    SAVED_RESPONSE_TRANSPORT,
    SSE_COMPLETION_POLICY,
    TRANSPORT,
    _review_materialization_errors,
    _review_schema_extra_paths,
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
    formal_rereview_failed_result_errors,
    formal_rereview_success_ledger,
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
SOURCE_ROOT = ROOT.parent.parent / "datasets"
BASE_PROPOSALS_PATH = ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json"
BASE_PROPOSAL_VALIDATION_PATH = (
    ROOT.parent / "suite-v2-blocker-remediation" / "validation.json"
)
SUPPLEMENTAL_PROPOSALS_PATH = ROOT / "reviewer-block-supplemental-proposals.json"
VISUAL_AUDIT_PATH = ROOT / "source-visual-audit.json"
DIFFICULTIES = ("easy", "medium", "hard")
EXPECTED_PARENT_HASHES = {
    "easy": "1a3cacbb59b3aa7f354a569f828ad6214351f07cd3d0923a798e98cc33c41b2a",
    "medium": "89c570d10282342fd7facec9bd7d20af61337b5ecaef0038157ed824c25c0369",
    "hard": "e5be4d2495618e819b9758fa753bb10d553037ca10b26dc6ccd8521404600a04",
}
_CASE_ID_RE = re.compile(r"P3500-(?:EASY|MEDIUM|HARD)(?:-V2)?-\d{3}", re.IGNORECASE)
_FORBIDDEN_SEMANTIC_KEYS = {
    "case_id",
    "parent_case_id",
    "difficulty",
    "final_answer",
    "pre_grounding_answer",
    "production_final_correct",
    "production_final_accuracy",
    "production_metrics",
}
EXPECTED_CASE_PASS_POLICY = freeze_case_pass_policy({"policy_version": "stt.eval.case.v3"})
_ABSTENTION_CUE_RE = re.compile(
    r"(?:(?:未|没有)(?:提供|给出|说明|记载|包含)|无相关|无.*信息|没有.*信息|"
    r"无法|不能确定|不可确定|不足以|缺少|缺乏|不包含|"
    r"not provided|cannot determine|insufficient)",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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
        source_rows: list[dict[str, Any]],
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


def _strict_success_response_errors(
    responses: list[dict[str, Any]],
    request_by_id: dict[str, dict[str, Any]],
    *,
    expected_prompt_hash: str,
    pack_name: str,
) -> tuple[list[str], set[str]]:
    """Independently reject any successful row that is not directly adoptable.

    Rereview results must already satisfy the recursive response schema and the
    exact blind semantic payload.  In particular, a runner-side structural
    normalization is audit evidence for another rereview, never an adoptable
    response.
    """

    errors: list[str] = []
    succeeded: dict[str, dict[str, Any]] = {}
    for row in responses:
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        request = request_by_id.get(custom_id)
        if request is None:
            errors.append(f"{pack_name} success has no request: {custom_id}")
            continue
        if str(row.get("prompt_sha256") or "") != expected_prompt_hash:
            errors.append(f"{pack_name} success prompt hash drift: {custom_id}")
        review = row.get("review")
        if not isinstance(review, dict):
            errors.append(f"{pack_name} success has no review object: {custom_id}")
            continue
        prior = succeeded.get(custom_id)
        if prior is not None and _sha256_json(prior.get("review")) != _sha256_json(
            review
        ):
            errors.append(f"{pack_name} has conflicting successes: {custom_id}")
            continue
        succeeded.setdefault(custom_id, row)
        payload = json.loads(request["request"]["messages"][1]["content"])
        validation_errors = _validate_review(review)
        validation_errors.extend(_validate_against_payload(review, payload))
        validation_errors.extend(
            _validate_inverse_edge_canonicalization_audit(
                row, payload, source_response_rows=responses
            )
        )
        validation_errors.extend(
            _validate_exact_variant_deduplication_audit(
                row, payload, source_response_rows=responses
            )
        )
        validation_errors.extend(
            _validate_coverage_status_canonicalization_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_nonsemantic_canonicalization_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_quote_fragment_inverse_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_abstain_factual_constraint_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_redundant_field_coverage_composition_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_unsupported_required_concept_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_empty_required_branch_block_audit(
                row,
                payload,
                source_response_rows=responses,
            )
        )
        validation_errors.extend(
            _validate_saved_strict_revalidation_audit(
                row, payload, source_response_rows=responses
            )
        )
        if isinstance(
            row.get("saved_response_same_concept_variant_subset_recovery"), dict
        ):
            validation_errors.extend(
                _validate_saved_post150_recovery(
                    row,
                    request=request,
                    semantic_payload=payload,
                    source_response_rows=responses,
                    execution_contract_sha256=str(
                        row.get("execution_contract_sha256") or ""
                    ),
                )
            )
        normalization_actions = [
            str(item)
            for item in row.get("structural_normalization") or []
            if str(item).strip()
        ]
        if normalization_actions:
            validation_errors.append(
                "successful rereview required structural normalization and cannot be adopted"
            )
        if validation_errors:
            errors.append(
                f"{pack_name} success is not strictly adoptable {custom_id}: "
                + "; ".join(validation_errors)
            )
    return errors, set(succeeded)


def _expected_concept_groups(
    contract: dict[str, Any], evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _shared_materialize_concept_groups(contract, evidence)


def _variant_weak_reason(
    variant: str, *, critical_values: set[str], evidence_tokens: set[str]
) -> str | None:
    return _shared_variant_weak_reason(
        variant,
        critical_values=critical_values,
        evidence_tokens=evidence_tokens,
    )


def _contract_coverage_errors(
    contract: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    expected_branch_spans: list[str],
    expected_unanswerable: bool,
) -> list[str]:
    errors: list[str] = []
    concepts = (contract.get("required_concepts") or {}).get("all_of") or []
    concept_id_list = [str(item.get("concept_id") or "") for item in concepts]
    concept_ids = set(concept_id_list)
    if (
        not concept_ids
        or "" in concept_ids
        or len(concept_ids) != len(concept_id_list)
    ):
        errors.append("contract must have unique authored concepts")
        return errors
    branches = contract.get("required_branches") or []
    coverage = contract.get("evidence_concept_coverage") or []
    evidence_ids = [str(item.get("evidence_id") or "") for item in evidence]
    coverage_ids = [str(item.get("evidence_id") or "") for item in coverage]
    expected_branch_counter = Counter(
        _policy_normalized(value) for value in expected_branch_spans
    )
    actual_branch_counter = Counter(
        _policy_normalized(branch.get("query_span")) for branch in branches
    )
    if (
        "" in evidence_ids
        or len(set(evidence_ids)) != len(evidence_ids)
        or Counter(coverage_ids) != Counter(evidence_ids)
        or actual_branch_counter != expected_branch_counter
        or "" in actual_branch_counter
    ):
        errors.append("evidence/branch grain mismatch")

    answerability = str((contract.get("answerability") or {}).get("required") or "")
    if (answerability == "abstain") != expected_unanswerable:
        errors.append("answerability conflicts with dataset unanswerable boundary")
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
            errors.append("abstain contract contains factual or evidence-supported requirements")
        return errors
    if answerability != "answer":
        errors.append("unknown answerability contract")
        return errors

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
        or not constraint_evidence_ids.issubset(set(evidence_ids))
        or any(not branch.get("required_concept_ids") for branch in branches)
    ):
        errors.append("concept/evidence/branch edges are incomplete or asymmetric")
    return errors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _semantic_leaks(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _FORBIDDEN_SEMANTIC_KEYS:
                errors.append(f"{path}.{key}")
            errors.extend(_semantic_leaks(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_semantic_leaks(nested, f"{path}[{index}]"))
    elif isinstance(value, str) and _CASE_ID_RE.search(value):
        errors.append(path)
    return errors


def _reviewed_parent_coverage_errors(
    *,
    expected_case_ids_by_difficulty: dict[str, set[str]],
    expected_parent_by_case_id: dict[str, str],
    reviewed_cases_by_difficulty: dict[str, list[dict[str, Any]]],
    reviewed_qa: list[dict[str, Any]],
    active_payload_parent_ids: set[str],
) -> list[str]:
    """Prove reviewed cases, QA rows and active payloads cover the same 150 parents."""

    errors: list[str] = []
    expected_case_ids = set(expected_parent_by_case_id)
    expected_parent_ids = set(expected_parent_by_case_id.values())
    reviewed_parent_ids: list[str] = []
    reviewed_case_ids: list[str] = []
    for difficulty, expected_ids in expected_case_ids_by_difficulty.items():
        manifest_cases = reviewed_cases_by_difficulty.get(difficulty) or []
        actual_ids = {str(case.get("case_id") or "") for case in manifest_cases}
        if actual_ids != expected_ids or len(manifest_cases) != len(expected_ids):
            errors.append(
                f"{difficulty} reviewed case-id set does not exactly cover candidate set"
            )
        for case in manifest_cases:
            case_id = str(case.get("case_id") or "")
            parent_id = str(
                ((case.get("metadata") or {}).get("parent_case_id")) or ""
            )
            reviewed_case_ids.append(case_id)
            reviewed_parent_ids.append(parent_id)
            if expected_parent_by_case_id.get(case_id) != parent_id:
                errors.append(
                    f"reviewed case parent lineage mismatch: {difficulty}/{case_id}"
                )
    if (
        len(reviewed_case_ids) != len(expected_case_ids)
        or set(reviewed_case_ids) != expected_case_ids
        or len(set(reviewed_case_ids)) != len(reviewed_case_ids)
    ):
        errors.append("reviewed cases do not exactly cover all candidate case ids")
    if (
        len(reviewed_parent_ids) != len(expected_parent_ids)
        or set(reviewed_parent_ids) != expected_parent_ids
        or len(set(reviewed_parent_ids)) != len(reviewed_parent_ids)
    ):
        errors.append("reviewed cases do not exactly cover 150 unique candidate parents")

    qa_case_ids: list[str] = []
    qa_parent_ids: list[str] = []
    for row in reviewed_qa:
        case_id = str(row.get("case_id") or "")
        parent_id = str(row.get("parent_case_id") or "")
        difficulty = str(row.get("difficulty") or "")
        qa_case_ids.append(case_id)
        qa_parent_ids.append(parent_id)
        if (
            expected_parent_by_case_id.get(case_id) != parent_id
            or case_id not in expected_case_ids_by_difficulty.get(difficulty, set())
        ):
            errors.append(f"reviewed QA parent lineage mismatch: {difficulty}/{case_id}")
    if (
        len(qa_case_ids) != len(expected_case_ids)
        or set(qa_case_ids) != expected_case_ids
        or len(set(qa_case_ids)) != len(qa_case_ids)
        or len(qa_parent_ids) != len(expected_parent_ids)
        or set(qa_parent_ids) != expected_parent_ids
        or len(set(qa_parent_ids)) != len(qa_parent_ids)
    ):
        errors.append("reviewed QA does not exactly cover candidate case/parent sets")
    if active_payload_parent_ids != expected_parent_ids:
        errors.append("active blind payloads do not exactly cover candidate parent set")
    return errors


def _checksum_coverage_errors(
    checksum_paths: list[str], expected_paths: set[str]
) -> list[str]:
    errors: list[str] = []
    if len(checksum_paths) != len(set(checksum_paths)):
        errors.append("SHA256SUMS contains duplicate paths")
    missing = sorted(expected_paths - set(checksum_paths))
    unexpected = sorted(set(checksum_paths) - expected_paths)
    if missing:
        errors.append(f"SHA256SUMS omits final artifacts: {missing}")
    if unexpected:
        errors.append(f"SHA256SUMS contains unexpected paths: {unexpected}")
    return errors


def _materialization_adoption_reporting_errors(
    *,
    reviewed_qa: list[dict[str, Any]],
    reviewed_summary: dict[str, Any],
    expected_by_parent_case_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Bind reviewed adoption claims to independently replayed formal episodes."""

    errors: list[str] = []
    expected_by_case_id: dict[str, dict[str, Any]] = {}
    for row in reviewed_qa:
        case_id = str(row.get("case_id") or "")
        parent_case_id = str(row.get("parent_case_id") or "")
        expected = expected_by_parent_case_id.get(parent_case_id)
        if row.get("post150_materialization_adoption") != expected:
            errors.append(
                f"reviewed QA materialization adoption receipt mismatch: {case_id}"
            )
        if expected is not None:
            if not case_id or case_id in expected_by_case_id:
                errors.append("reviewed QA materialization adoption case identity is invalid")
            else:
                expected_by_case_id[case_id] = expected

    expected_case_ids = sorted(expected_by_case_id)
    expected_episode_ids = sorted(
        str(receipt.get("episode_id") or "")
        for receipt in expected_by_case_id.values()
    )
    expected_summary = {
        "post150_materialization_adoption_policy": MATERIALIZATION_ADOPTION_POLICY,
        "post150_materialization_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "post150_materialization_global_fresh_retry_quota": (
            MATERIALIZATION_RETRY_QUOTA
        ),
        "post150_materialization_global_fresh_retry_consumed_count": len(
            expected_by_case_id
        ),
        "post150_formal_stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "post150_formal_stream_content_sha256_authoritative": False,
        "post150_formal_stream_content_chars_authoritative": False,
        "post150_formal_stream_event_counts_authoritative": False,
        "post150_materialization_adopted_case_count": len(expected_by_case_id),
        "post150_materialization_adopted_case_ids": expected_case_ids,
        "post150_materialization_adopted_episode_ids": expected_episode_ids,
        "post150_materialization_adoption_by_case_id": {
            case_id: expected_by_case_id[case_id] for case_id in expected_case_ids
        },
    }
    if any(
        reviewed_summary.get(key) != value
        for key, value in expected_summary.items()
    ):
        errors.append(
            "reviewed summary materialization adoption receipts/counts are stale"
        )
    return errors


def validate(*, require_reviewed: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    schema_validated_contract_count = 0
    for difficulty, expected in EXPECTED_PARENT_HASHES.items():
        actual = _sha256_file(SOURCE_ROOT / f"{difficulty}.json")
        if actual != expected:
            errors.append(f"frozen parent hash drift for {difficulty}: {actual}")

    suite = json.loads((ROOT / "suite-v2.json").read_text(encoding="utf-8"))
    if suite.get("suite_key") != "three-difficulty-live-2026-08-10-concept-v2-candidate":
        errors.append("unexpected deterministic candidate suite_key")
    if suite.get("import_allowed") is not False or suite.get("baseline_registration_allowed") is not False:
        errors.append("candidate suite must explicitly prohibit import and baseline registration")
    candidate_contract_schema = json.loads(
        (ROOT / "schema" / "answer-contract-v2.schema.json").read_text(encoding="utf-8")
    )
    candidate_contract_validator = Draft202012Validator(candidate_contract_schema)
    cases: list[dict[str, Any]] = []
    parent_ids: list[str] = []
    candidate_case_ids_by_difficulty: dict[str, set[str]] = {
        difficulty: set() for difficulty in DIFFICULTIES
    }
    candidate_parent_by_case_id: dict[str, str] = {}
    for difficulty in DIFFICULTIES:
        manifest = json.loads(
            (ROOT / "datasets" / f"{difficulty}.v2.json").read_text(encoding="utf-8")
        )
        if len(manifest.get("cases") or []) != 50:
            errors.append(f"{difficulty} candidate must contain 50 cases")
        for case in manifest.get("cases") or []:
            cases.append(case)
            metadata = case.get("metadata") or {}
            case_id = str(case.get("case_id") or "")
            parent_case_id = str(metadata.get("parent_case_id") or "")
            candidate_case_ids_by_difficulty[difficulty].add(case_id)
            candidate_parent_by_case_id[case_id] = parent_case_id
            parent_ids.append(parent_case_id)
            if str(case.get("case_id") or "") == str(metadata.get("parent_case_id") or ""):
                errors.append(f"candidate reused parent case id: {case.get('case_id')}")
            contract = case.get("answer_contract") or {}
            contract_schema_errors = list(candidate_contract_validator.iter_errors(contract))
            if contract_schema_errors:
                errors.append(
                    f"candidate contract schema failure {case.get('case_id')}: "
                    f"{contract_schema_errors[0].message}"
                )
            else:
                schema_validated_contract_count += 1
            answerability = (contract.get("answerability") or {}).get("required")
            concepts = (contract.get("required_concepts") or {}).get("all_of") or []
            if case.get("unanswerable"):
                if answerability != "abstain":
                    errors.append(f"unanswerable contract not abstain: {case.get('case_id')}")
            elif answerability != "answer" or not concepts:
                errors.append(f"answerable contract lacks concepts: {case.get('case_id')}")
            evidence_ids = {
                str(item.get("evidence_id") or "")
                for item in case.get("relevant_evidence") or []
            }
            coverage_ids = {
                str(item.get("evidence_id") or "")
                for item in contract.get("evidence_concept_coverage") or []
            }
            if evidence_ids != coverage_ids:
                errors.append(f"evidence coverage grain mismatch: {case.get('case_id')}")
    if len(cases) != 150 or len({case["case_id"] for case in cases}) != 150:
        errors.append("candidate manifests must contain 150 unique v2 case ids")
    if len(parent_ids) != 150 or len(set(parent_ids)) != 150 or "" in parent_ids:
        errors.append("candidate manifests must preserve 150 unique parent ids")

    qa_rows = json.loads((ROOT / "qa" / "cases.json").read_text(encoding="utf-8"))
    if len(qa_rows) != 150 or len({row["parent_case_id"] for row in qa_rows}) != 150:
        errors.append("QA artifact must contain 150 unique parent-grain rows")
    if {
        (
            str(row.get("difficulty") or ""),
            str(row.get("case_id") or ""),
            str(row.get("parent_case_id") or ""),
        )
        for row in qa_rows
    } != {
        (difficulty, case_id, candidate_parent_by_case_id[case_id])
        for difficulty, case_ids in candidate_case_ids_by_difficulty.items()
        for case_id in case_ids
    }:
        errors.append("QA artifact does not exactly match candidate case/parent lineage")
    requests = _read_jsonl(ROOT / "review" / "requests.jsonl")
    try:
        validate_request_pack_identity(
            requests,
            expected_prompt_sha256=EXPECTED_REVIEW_PROMPT_SHA256,
            expected_count=150,
            pack_name="validator full-suite review pack",
        )
    except RuntimeError as exc:
        errors.append(str(exc))
    if len(requests) != 150 or len({row["custom_id"] for row in requests}) != 150:
        errors.append("review pack must contain 150 unique requests")
    if len({row["prompt_sha256"] for row in requests}) != 1:
        errors.append("review pack prompt hash drift")
    for row in requests:
        payload = json.loads(row["request"]["messages"][1]["content"])
        leaks = _semantic_leaks(payload)
        if leaks:
            errors.append(f"semantic payload leak for {row['custom_id']}: {leaks[:3]}")
    request_map = json.loads((ROOT / "review" / "request-map.json").read_text(encoding="utf-8"))
    if (
        len(request_map) != 150
        or len({row["custom_id"] for row in request_map}) != 150
        or {row["custom_id"] for row in request_map}
        != {row["custom_id"] for row in requests}
    ):
        errors.append("request map ids do not match review request ids")
    if {
        (
            str(row.get("difficulty") or ""),
            str(row.get("case_id") or ""),
            str(row.get("parent_case_id") or ""),
        )
        for row in request_map
    } != {
        (difficulty, case_id, candidate_parent_by_case_id[case_id])
        for difficulty, case_ids in candidate_case_ids_by_difficulty.items()
        for case_id in case_ids
    }:
        errors.append("request map does not exactly match candidate case/parent lineage")

    checksum_lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checksum_checked = 0
    checksum_paths: list[str] = []
    for line in checksum_lines:
        if not line.strip():
            continue
        expected, relative = line.split("  ", maxsplit=1)
        checksum_paths.append(relative)
        path = ROOT / relative
        if not path.exists():
            errors.append(f"checksum path missing: {relative}")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")
        checksum_checked += 1
    expected_checksum_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != ROOT / "SHA256SUMS"
        and "__pycache__" not in path.parts
        and not path.name.endswith(".tmp")
    }
    errors.extend(_checksum_coverage_errors(checksum_paths, expected_checksum_paths))

    reviewed_status = "not_present"
    reviewed_summary_path = ROOT / "reviewed" / "summary.json"
    if reviewed_summary_path.exists():
        reviewed_status = "present"
        reviewed_summary = json.loads(reviewed_summary_path.read_text(encoding="utf-8"))
        if reviewed_summary.get("case_count") != 150:
            errors.append("reviewed summary must contain 150 cases")
        if (
            reviewed_summary.get("import_allowed") is not False
            or reviewed_summary.get("baseline_registration_allowed") is not False
        ):
            errors.append("reviewed candidate must prohibit import and baseline registration")
        block_union_audit = json.loads(
            (ROOT / "review-block-union-audit.json").read_text(encoding="utf-8")
        )
        audit_review = block_union_audit.get("review") or {}
        audit_inputs = block_union_audit.get("inputs") or {}
        try:
            validate_block_union_freshness(
                block_union_audit,
                requests_path=ROOT / "review" / "requests.jsonl",
                responses_path=ROOT / "review" / "responses.jsonl",
                visual_audit_path=VISUAL_AUDIT_PATH,
                base_proposals_path=BASE_PROPOSALS_PATH,
                supplemental_proposals_path=SUPPLEMENTAL_PROPOSALS_PATH,
                label="validator block-union audit",
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        proposal_validation = json.loads(
            BASE_PROPOSAL_VALIDATION_PATH.read_text(encoding="utf-8")
        )
        try:
            validate_proposal_validation_binding(
                proposal_validation,
                proposals_path=BASE_PROPOSALS_PATH,
                label="validator base proposal validation",
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        if (
            block_union_audit.get("status") != "passed"
            or audit_review.get("successful_case_count") != 150
            or audit_review.get("missing_case_count") != 0
            or audit_review.get("responses_sha256")
            != _sha256_file(ROOT / "review" / "responses.jsonl")
            or audit_review.get("request_pack_sha256")
            != _sha256_file(ROOT / "review" / "requests.jsonl")
            or audit_inputs.get("base_proposals_sha256")
            != _sha256_file(ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json")
            or audit_inputs.get("supplemental_proposals_sha256")
            != _sha256_file(ROOT / "reviewer-block-supplemental-proposals.json")
            or audit_inputs.get("visual_audit_sha256")
            != _sha256_file(VISUAL_AUDIT_PATH)
        ):
            errors.append("final reviewer/visual/materialization audit is stale or incomplete")
        if (
            reviewed_summary.get("reviewer_block_case_ids")
            != block_union_audit.get("reviewer_block_case_ids")
            or reviewed_summary.get("reviewer_block_union_visual_block")
            != block_union_audit.get("reviewer_block_union_visual_block")
            or reviewed_summary.get("contract_rereview_case_ids")
            != block_union_audit.get("contract_rereview_case_ids")
        ):
            errors.append("reviewed summary block/rereview union does not match final audit")
        schema_extra_parent_ids = {
            str(value)
            for value in block_union_audit.get("review_schema_extra_field_case_ids") or []
        }
        if (
            reviewed_summary.get("review_schema_extra_field_case_ids")
            != sorted(schema_extra_parent_ids)
            or reviewed_summary.get("review_schema_extra_disposition")
            != "same_semantic_rereview_completed_no_projection_or_semantic_reuse"
        ):
            errors.append("reviewed summary schema-extra supersession audit mismatch")
        normalized_parent_ids = {
            str(value)
            for value in block_union_audit.get("structural_normalization_case_ids") or []
        }
        if (
            reviewed_summary.get("structural_normalization_case_ids")
            != sorted(normalized_parent_ids)
            or reviewed_summary.get("structural_normalization_disposition")
            != "same_semantic_rereview_completed_original_normalized_semantics_not_reused"
        ):
            errors.append("reviewed summary structural-normalization supersession mismatch")

        original_request_by_id = {str(row["custom_id"]): row for row in requests}
        original_parent_by_id = {
            str(row.get("custom_id") or ""): str(row.get("parent_case_id") or "")
            for row in request_map
        }
        original_success_by_id: dict[str, dict[str, Any]] = {}
        original_response_rows = _read_jsonl(ROOT / "review" / "responses.jsonl")
        for row in original_response_rows:
            if row.get("status") != "succeeded":
                continue
            custom_id = str(row.get("custom_id") or "")
            prior = original_success_by_id.get(custom_id)
            if prior is not None and _sha256_json(prior.get("review")) != _sha256_json(
                row.get("review")
            ):
                errors.append(f"conflicting original successful reviews: {custom_id}")
                continue
            original_success_by_id.setdefault(custom_id, row)
        independently_invalid_originals: dict[str, list[str]] = {}
        independent_schema_extra_paths: dict[str, list[str]] = {}
        independent_normalization_actions: dict[str, list[str]] = {}
        independent_inverse_edge_audits: dict[str, dict[str, Any]] = {}
        independent_exact_dedupe_audits: dict[str, dict[str, Any]] = {}
        independent_coverage_status_audits: dict[str, dict[str, Any]] = {}
        independent_nonsemantic_composition_audits: dict[
            str, dict[str, Any]
        ] = {}
        independent_quote_fragment_composition_audits: dict[
            str, dict[str, Any]
        ] = {}
        independent_exact_span_quote_prefix_audits: dict[
            str, dict[str, Any]
        ] = {}
        independent_abstain_factual_constraint_audits: dict[
            str, dict[str, Any]
        ] = {}
        independent_redundant_field_composition_audits: dict[
            str, dict[str, Any]
        ] = {}
        independent_unsupported_block_audits: dict[str, dict[str, Any]] = {}
        independent_empty_branch_block_audits: dict[str, dict[str, Any]] = {}
        independent_saved_strict_audits: dict[str, dict[str, Any]] = {}
        for custom_id, row in original_success_by_id.items():
            request = original_request_by_id.get(custom_id)
            parent_case_id = original_parent_by_id.get(custom_id, "")
            if request is None or not parent_case_id:
                errors.append(f"original success has no request/parent lineage: {custom_id}")
                continue
            review = dict(row.get("review") or {})
            payload = json.loads(request["request"]["messages"][1]["content"])
            validation_errors = _validate_review(review)
            validation_errors.extend(_validate_against_payload(review, payload))
            inverse_edge_errors = _validate_inverse_edge_canonicalization_audit(
                row, payload, source_response_rows=original_response_rows
            )
            validation_errors.extend(inverse_edge_errors)
            exact_dedupe_errors = _validate_exact_variant_deduplication_audit(
                row, payload, source_response_rows=original_response_rows
            )
            validation_errors.extend(exact_dedupe_errors)
            coverage_status_errors = _validate_coverage_status_canonicalization_audit(
                row,
                payload,
                source_response_rows=original_response_rows,
            )
            validation_errors.extend(coverage_status_errors)
            composition_errors = (
                _validate_nonsemantic_canonicalization_composition_audit(
                    row,
                    payload,
                    source_response_rows=original_response_rows,
                )
            )
            validation_errors.extend(composition_errors)
            quote_fragment_composition_errors = (
                _validate_quote_fragment_inverse_composition_audit(
                    row,
                    payload,
                    source_response_rows=original_response_rows,
                )
            )
            validation_errors.extend(quote_fragment_composition_errors)
            abstain_factual_constraint_errors = (
                _validate_abstain_factual_constraint_recovery_audit(
                    row,
                    payload,
                    source_response_rows=original_response_rows,
                )
            )
            validation_errors.extend(abstain_factual_constraint_errors)
            exact_span_quote_prefix_errors = (
                _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
                    row,
                    payload,
                    source_response_rows=original_response_rows,
                )
            )
            validation_errors.extend(exact_span_quote_prefix_errors)
            redundant_field_composition_errors = (
                _validate_redundant_field_coverage_composition_audit(
                    row,
                    payload,
                    source_response_rows=original_response_rows,
                )
            )
            validation_errors.extend(redundant_field_composition_errors)
            unsupported_block_errors = _validate_unsupported_required_concept_block_audit(
                row,
                payload,
                source_response_rows=original_response_rows,
            )
            validation_errors.extend(unsupported_block_errors)
            empty_branch_block_errors = _validate_empty_required_branch_block_audit(
                row,
                payload,
                source_response_rows=original_response_rows,
            )
            validation_errors.extend(empty_branch_block_errors)
            strict_revalidation_errors = _validate_saved_strict_revalidation_audit(
                row, payload, source_response_rows=original_response_rows
            )
            validation_errors.extend(strict_revalidation_errors)
            inverse_edge_audit = row.get(
                "same_response_inverse_edge_canonicalization"
            )
            if isinstance(inverse_edge_audit, dict) and not inverse_edge_errors:
                independent_inverse_edge_audits[parent_case_id] = {
                    "policy": inverse_edge_audit.get("policy"),
                    "api_call_performed": inverse_edge_audit.get(
                        "api_call_performed"
                    ),
                    "raw_review_sha256": inverse_edge_audit.get(
                        "raw_review_sha256"
                    ),
                    "projected_review_sha256": inverse_edge_audit.get(
                        "projected_review_sha256"
                    ),
                    "source_attempt": (
                        inverse_edge_audit.get("source") or {}
                    ).get("source_attempt"),
                    "action_count": len(inverse_edge_audit.get("actions") or []),
                }
            exact_dedupe_audit = row.get(
                "same_response_exact_variant_deduplication"
            )
            if isinstance(exact_dedupe_audit, dict) and not exact_dedupe_errors:
                independent_exact_dedupe_audits[parent_case_id] = {
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
            coverage_status_audit = row.get(
                "same_response_coverage_status_canonicalization"
            )
            if (
                isinstance(coverage_status_audit, dict)
                and not coverage_status_errors
            ):
                independent_coverage_status_audits[parent_case_id] = {
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
                    "action_count": len(
                        coverage_status_audit.get("actions") or []
                    ),
                }
            composition_audit = row.get(
                "same_response_nonsemantic_canonicalization_composition"
            )
            if isinstance(composition_audit, dict) and not composition_errors:
                independent_nonsemantic_composition_audits[parent_case_id] = {
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
            quote_fragment_composition_audit = row.get(
                "same_response_quote_fragment_inverse_canonicalization_composition"
            )
            if (
                isinstance(quote_fragment_composition_audit, dict)
                and not quote_fragment_composition_errors
            ):
                independent_quote_fragment_composition_audits[parent_case_id] = {
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
            exact_span_quote_prefix_audit = row.get(
                "saved_response_exact_payload_branch_span_quote_prefix_recovery"
            )
            if (
                isinstance(exact_span_quote_prefix_audit, dict)
                and not exact_span_quote_prefix_errors
            ):
                independent_exact_span_quote_prefix_audits[parent_case_id] = {
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
            abstain_factual_constraint_audit = row.get(
                "saved_response_abstain_factual_constraint_recovery"
            )
            if (
                isinstance(abstain_factual_constraint_audit, dict)
                and not abstain_factual_constraint_errors
            ):
                independent_abstain_factual_constraint_audits[parent_case_id] = {
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
            redundant_field_composition_audit = row.get(
                "saved_response_redundant_field_coverage_composition"
            )
            if (
                isinstance(redundant_field_composition_audit, dict)
                and not redundant_field_composition_errors
            ):
                independent_redundant_field_composition_audits[parent_case_id] = {
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
                    "projected_review_sha256": (
                        redundant_field_composition_audit.get(
                            "projected_review_sha256"
                        )
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
            unsupported_block_audit = row.get(
                "same_response_unsupported_required_concept_block"
            )
            if (
                isinstance(unsupported_block_audit, dict)
                and not unsupported_block_errors
            ):
                independent_unsupported_block_audits[parent_case_id] = {
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
            empty_branch_block_audit = row.get(
                "multi_response_empty_required_branch_block"
            )
            if (
                isinstance(empty_branch_block_audit, dict)
                and not empty_branch_block_errors
            ):
                independent_empty_branch_block_audits[parent_case_id] = {
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
                    "action_count": len(
                        empty_branch_block_audit.get("actions") or []
                    ),
                }
            strict_revalidation_audit = row.get(
                "saved_response_strict_revalidation"
            )
            if (
                isinstance(strict_revalidation_audit, dict)
                and not strict_revalidation_errors
            ):
                independent_saved_strict_audits[parent_case_id] = {
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
            extra_paths = _review_schema_extra_paths(review)
            if extra_paths:
                independent_schema_extra_paths[parent_case_id] = extra_paths
            normalization_actions = [
                str(item)
                for item in row.get("structural_normalization") or []
                if str(item).strip()
            ]
            if normalization_actions:
                independent_normalization_actions[parent_case_id] = normalization_actions
                validation_errors.append(
                    "persisted success required structural normalization; strict independent "
                    "same-semantic rereview is required"
                )
            if validation_errors:
                independently_invalid_originals[parent_case_id] = validation_errors
        if independently_invalid_originals != (
            block_union_audit.get("review_contract_validation_failure_reasons") or {}
        ):
            errors.append("independent original-review strict validation differs from audit")
        if independent_schema_extra_paths != (
            block_union_audit.get("review_schema_extra_paths_by_case") or {}
        ):
            errors.append("independent schema-extra scan differs from block-union audit")
        if independent_normalization_actions != (
            block_union_audit.get("structural_normalization_actions_by_case") or {}
        ):
            errors.append("independent structural-normalization scan differs from audit")
        if independent_inverse_edge_audits != (
            block_union_audit.get("inverse_edge_canonicalization_audits_by_case")
            or {}
        ):
            errors.append("independent inverse-edge canonicalization scan differs from audit")
        if independent_exact_dedupe_audits != (
            block_union_audit.get("exact_variant_deduplication_audits_by_case")
            or {}
        ):
            errors.append("independent exact-variant deduplication scan differs from audit")
        if independent_coverage_status_audits != (
            block_union_audit.get("coverage_status_canonicalization_audits_by_case")
            or {}
        ):
            errors.append(
                "independent coverage-status canonicalization scan differs from audit"
            )
        if independent_nonsemantic_composition_audits != (
            block_union_audit.get(
                "nonsemantic_canonicalization_composition_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent nonsemantic composition scan differs from audit"
            )
        if independent_quote_fragment_composition_audits != (
            block_union_audit.get(
                "quote_fragment_inverse_composition_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent quote-fragment inverse composition scan differs from audit"
            )
        if independent_exact_span_quote_prefix_audits != (
            block_union_audit.get(
                "exact_payload_branch_span_quote_prefix_recovery_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent exact payload branch-span recovery scan differs from audit"
            )
        if independent_abstain_factual_constraint_audits != (
            block_union_audit.get(
                "abstain_factual_constraint_recovery_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent abstain factual-constraint recovery scan differs from audit"
            )
        if independent_redundant_field_composition_audits != (
            block_union_audit.get(
                "redundant_field_coverage_composition_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent redundant-field coverage composition scan differs from audit"
            )
        if independent_unsupported_block_audits != (
            block_union_audit.get(
                "unsupported_required_concept_block_audits_by_case"
            )
            or {}
        ):
            errors.append(
                "independent unsupported-required-concept block scan differs from audit"
            )
        if independent_empty_branch_block_audits != (
            block_union_audit.get("empty_required_branch_block_audits_by_case")
            or {}
        ):
            errors.append(
                "independent empty-required-branch block scan differs from audit"
            )
        if independent_saved_strict_audits != (
            block_union_audit.get("saved_strict_revalidation_audits_by_case")
            or {}
        ):
            errors.append("independent saved strict-revalidation scan differs from audit")
        expected_contract_queue = sorted(
            str(value)
            for value in block_union_audit.get("contract_rereview_case_ids") or []
        )
        if block_union_audit.get("contract_rereview_case_ids_sha256") != _sha256_json(
            expected_contract_queue
        ):
            errors.append("contract-rereview queue hash mismatch")
        active_payload_by_parent: dict[str, dict[str, Any]] = {}
        for mapping in request_map:
            request = original_request_by_id.get(str(mapping.get("custom_id") or ""))
            if request is None:
                continue
            case_id = str(mapping.get("case_id") or "")
            parent_case_id = str(mapping.get("parent_case_id") or "")
            difficulty = str(mapping.get("difficulty") or "")
            if (
                case_id not in candidate_case_ids_by_difficulty.get(difficulty, set())
                or candidate_parent_by_case_id.get(case_id) != parent_case_id
            ):
                errors.append(f"original request-map candidate lineage mismatch: {case_id}")
            if parent_case_id in active_payload_by_parent:
                errors.append(f"duplicate original active-payload parent: {parent_case_id}")
                continue
            active_payload_by_parent[parent_case_id] = json.loads(
                request["request"]["messages"][1]["content"]
            )
        blocker_requests = _read_jsonl(ROOT / "blocker-rereview" / "requests.jsonl")
        blocker_request_map = json.loads(
            (ROOT / "blocker-rereview" / "request-map.json").read_text(encoding="utf-8")
        )
        blocker_request_by_id = {
            str(row.get("custom_id") or ""): row for row in blocker_requests
        }
        blocker_execution_contract = ""
        base_proposals_document = json.loads(
            BASE_PROPOSALS_PATH.read_text(encoding="utf-8")
        )
        supplemental_proposals_document = json.loads(
            SUPPLEMENTAL_PROPOSALS_PATH.read_text(encoding="utf-8")
        )
        try:
            blocker_prepared_summary = json.loads(
                (ROOT / "blocker-rereview" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_blocker_rereview_prepared_receipt(blocker_prepared_summary)
            validate_request_pack_identity(
                blocker_requests,
                expected_prompt_sha256=str(reviewed_summary.get("prompt_sha256") or ""),
                expected_count=int(
                    reviewed_summary.get("blocker_rereview_case_count") or 0
                ),
                expected_pack_sha256=str(
                    blocker_prepared_summary.get("request_pack_sha256") or ""
                ),
                pack_name="validator changed blocker-rereview pack",
            )
            blocker_execution_contract = validate_rereview_request_policy(
                blocker_requests,
                runner_version=RUNNER_VERSION,
                transport=TRANSPORT,
                sse_completion_policy=SSE_COMPLETION_POLICY,
            )
            blocker_execution_summary = json.loads(
                (ROOT / "blocker-rereview" / "execution-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            blocker_current_receipts = {
                "review_block_union_audit_sha256": sha256_json(block_union_audit),
                "proposal_document_sha256": sha256_json(base_proposals_document),
                "supplemental_proposal_document_sha256": sha256_json(
                    supplemental_proposals_document
                ),
                "proposal_validation_sha256": sha256_json(proposal_validation),
            }
            if (
                blocker_prepared_summary.get("execution_contract_sha256")
                != blocker_execution_contract
                or blocker_execution_summary.get("execution_contract_sha256")
                != blocker_execution_contract
                or any(
                    blocker_prepared_summary.get(key) != value
                    for key, value in blocker_current_receipts.items()
                )
            ):
                errors.append("changed blocker rereview receipt/contract is stale")
            base_proposal_rows = base_proposals_document.get("proposals") or []
            supplemental_proposal_rows = (
                supplemental_proposals_document.get("proposals") or []
            )
            current_proposal_parent_ids = {
                str(row.get("case_id") or "")
                for row in [*base_proposal_rows, *supplemental_proposal_rows]
            }
            if (
                len(base_proposal_rows)
                != EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
                or len(supplemental_proposal_rows)
                != EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
                or len(current_proposal_parent_ids)
                != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
                or current_proposal_parent_ids
                != {
                    str(value)
                    for value in block_union_audit.get(
                        "required_data_remediation_case_ids"
                    )
                    or []
                }
            ):
                errors.append("changed blocker current upstream is not exact 13+2")
        except RuntimeError as exc:
            errors.append(str(exc))
        changed_payload_parent_ids: set[str] = set()
        for mapping in blocker_request_map:
            request = blocker_request_by_id.get(str(mapping.get("custom_id") or ""))
            if request is None:
                continue
            parent_case_id = str(mapping.get("parent_case_id") or "")
            if parent_case_id in changed_payload_parent_ids:
                errors.append(f"duplicate changed active-payload parent: {parent_case_id}")
                continue
            changed_payload_parent_ids.add(parent_case_id)
            if parent_case_id not in set(candidate_parent_by_case_id.values()):
                errors.append(f"unknown changed active-payload parent: {parent_case_id}")
            payload = json.loads(request["request"]["messages"][1]["content"])
            leaks = _semantic_leaks(payload)
            if leaks:
                errors.append(
                    f"changed blocker semantic payload leak for {mapping.get('parent_case_id')}: "
                    f"{leaks[:3]}"
                )
            active_payload_by_parent[parent_case_id] = payload
        if set(active_payload_by_parent) != set(candidate_parent_by_case_id.values()):
            errors.append("active blind payloads do not exactly cover 150 candidate parents")

        reviewed_suite = json.loads(
            (ROOT / "reviewed" / "suite-v2-reviewed.json").read_text(encoding="utf-8")
        )
        expected_reviewed_suite_key = (
            "three-difficulty-live-2026-08-11-reviewed-v2"
        )
        if reviewed_suite.get("suite_key") != expected_reviewed_suite_key:
            errors.append("unexpected reviewed candidate suite_key")
        if reviewed_suite.get("version") != "2.0.0-reviewed-v2.20260811":
            errors.append("unexpected reviewed candidate suite version")
        if (
            reviewed_suite.get("import_allowed") is not False
            or reviewed_suite.get("baseline_registration_allowed") is not False
        ):
            errors.append("reviewed suite must prohibit import and baseline registration")
        review_gate = reviewed_suite.get("review_gate") or {}
        if review_gate.get("full_150_same_prompt_review_complete") is not True:
            errors.append("reviewed suite must record full 150-case same-prompt review")
        if (
            review_gate.get("changed_blocker_same_prompt_rereview_complete") is not True
            or review_gate.get("changed_blocker_superseded_review_reused") is not False
        ):
            errors.append("reviewed suite must record independent same-prompt review of changed blockers")
        if (
            review_gate.get("weak_contract_same_semantic_rereview_complete") is not True
            or review_gate.get("weak_contract_dataset_cases_changed") is not False
            or review_gate.get("weak_contract_superseded_review_reused") is not False
        ):
            errors.append("reviewed suite must record independent same-semantic weak-contract review")
        if review_gate.get("independent_validation_complete") is not False:
            errors.append("reviewed suite must retain pending independent-validation gate")
        activation_gate = reviewed_suite.get("activation_gate") or {}
        if any(
            activation_gate.get(key) is not False
            for key in (
                "frontend_binding_allowed",
                "frontend_binding_changed_by_this_build",
                "evaluation_import_allowed",
                "baseline_registration_allowed",
            )
        ):
            errors.append("reviewed suite activation gate must prohibit UI/import/baseline changes")
        if (
            activation_gate.get("full_150_reviewer_responses_parsed") is not True
            or activation_gate.get("blocker_remediation_applied_case_count")
            != reviewed_summary.get("blocker_rereview_case_count")
            or activation_gate.get("changed_blocker_cases_independently_rereviewed")
            is not True
            or activation_gate.get("weak_contract_cases_independently_rereviewed")
            != reviewed_summary.get("contract_rereview_case_count")
            or activation_gate.get("weak_contract_semantic_payload_unchanged") is not True
            or activation_gate.get("answer_contract_to_concept_groups_materialization_complete")
            is not True
        ):
            errors.append("reviewed suite must record complete review/materialization")
        if activation_gate.get("effective_blocker_count") != reviewed_summary.get(
            "blocked_case_count"
        ):
            errors.append("reviewed suite activation blocker count mismatch")
        if reviewed_summary.get("blocked_case_count", 0) > 0 and activation_gate.get(
            "all_effective_blockers_resolved"
        ) is not False:
            errors.append("reviewed suite cannot mark effective blockers resolved")

        reviewed_contract_schema_path = (
            ROOT / "schema" / "answer-contract-v2-reviewed.schema.json"
        )
        reviewed_contract_schema = json.loads(
            reviewed_contract_schema_path.read_text(encoding="utf-8")
        )
        reviewed_contract_validator = Draft202012Validator(reviewed_contract_schema)
        reviewed_cases: list[dict[str, Any]] = []
        reviewed_cases_by_difficulty: dict[str, list[dict[str, Any]]] = {}
        for difficulty in DIFFICULTIES:
            reviewed_manifest_path = ROOT / "reviewed" / "datasets" / f"{difficulty}.reviewed.json"
            reviewed_manifest = json.loads(reviewed_manifest_path.read_text(encoding="utf-8"))
            expected_dataset_key = (
                f"pantum-p3500-{difficulty}-live-reviewed-v2-2026-08-11"
            )
            if reviewed_manifest.get("version") != "2.0.0-reviewed-v2.20260811":
                errors.append(f"{difficulty} reviewed manifest version mismatch")
            if reviewed_manifest.get("suite_key") != expected_reviewed_suite_key:
                errors.append(f"{difficulty} reviewed manifest suite_key mismatch")
            if reviewed_manifest.get("dataset_key") != expected_dataset_key:
                errors.append(f"{difficulty} reviewed manifest dataset_key mismatch")
            if (
                reviewed_manifest.get("import_allowed") is not False
                or reviewed_manifest.get("baseline_registration_allowed") is not False
            ):
                errors.append(f"{difficulty} reviewed manifest must prohibit import/baseline")
            case_policy = (reviewed_manifest.get("gate_policy") or {}).get(
                "case_pass_policy"
            ) or {}
            if case_policy != EXPECTED_CASE_PASS_POLICY:
                errors.append(f"{difficulty} reviewed manifest case.v3 policy is not frozen")
            if (
                (reviewed_manifest.get("gate_policy") or {}).get(
                    "case_pass_policy_activation_status"
                )
                != "reserved_not_active"
            ):
                errors.append(f"{difficulty} reviewed case.v3 must remain inactive")
            generation = reviewed_manifest.get("generation") or {}
            if (
                generation.get("answer_contract_to_concept_groups_materialization_status")
                != "complete"
            ):
                errors.append(f"{difficulty} concept_groups materialization is not complete")
            manifest_cases = reviewed_manifest.get("cases") or []
            reviewed_cases_by_difficulty[difficulty] = list(manifest_cases)
            if len(manifest_cases) != 50:
                errors.append(f"{difficulty} reviewed manifest must contain 50 cases")
            suite_dataset = (reviewed_suite.get("datasets") or {}).get(difficulty) or {}
            if suite_dataset.get("dataset_key") != expected_dataset_key:
                errors.append(f"{difficulty} reviewed suite dataset_key mismatch")
            if suite_dataset.get("checksum") != _sha256_json(reviewed_manifest):
                errors.append(f"{difficulty} reviewed manifest checksum mismatch in suite")
            for case in manifest_cases:
                reviewed_cases.append(case)
                contract = case.get("answer_contract") or {}
                parent_case_id = str((case.get("metadata") or {}).get("parent_case_id") or "")
                active_payload = active_payload_by_parent.get(parent_case_id)
                if active_payload is None:
                    active_payload = {}
                    errors.append(f"reviewed case has no active blind payload: {case.get('case_id')}")
                expected_evidence_payload = [
                    {
                        "evidence_label": f"E{index + 1}",
                        "page": item.get("page"),
                        "text": str(item.get("text") or ""),
                    }
                    for index, item in enumerate(case.get("relevant_evidence") or [])
                ]
                if (
                    str(case.get("query") or "") != str(active_payload.get("query") or "")
                    or str(case.get("reference_answer") or "")
                    != str(active_payload.get("reference_answer") or "")
                    or bool(case.get("unanswerable"))
                    != bool(active_payload.get("unanswerable"))
                    or expected_evidence_payload != (active_payload.get("evidence") or [])
                ):
                    errors.append(
                        f"reviewed case/query/reference/evidence drift from blind payload: "
                        f"{case.get('case_id')}"
                    )
                if bool(active_payload.get("unanswerable")) and not _abstention_variant_is_pure_refusal(
                    str(active_payload.get("reference_answer") or "")
                ):
                    errors.append(
                        f"unanswerable expected answer contains an affirmative-fact contract: "
                        f"{case.get('case_id')}"
                    )
                contract_schema_errors = list(reviewed_contract_validator.iter_errors(contract))
                if contract_schema_errors:
                    errors.append(
                        f"reviewed contract schema failure {case.get('case_id')}: "
                        f"{contract_schema_errors[0].message}"
                    )
                else:
                    schema_validated_contract_count += 1
                verdict = ((contract.get("review") or {}).get("verdict"))
                expected_status = (
                    "llm_reviewed_blocked" if verdict == "block" else "llm_reviewed_candidate"
                )
                if contract.get("contract_status") != expected_status:
                    errors.append(f"reviewed contract status/verdict mismatch: {case.get('case_id')}")
                concept_groups = case.get("concept_groups") or {}
                try:
                    expected_concept_groups, expected_filter_audit = _expected_concept_groups(
                        contract, case.get("relevant_evidence") or []
                    )
                except ValueError as exc:
                    expected_concept_groups = {"all_of": []}
                    expected_filter_audit = {}
                    errors.append(
                        f"reviewed discriminative-variant gate failure {case.get('case_id')}: {exc}"
                    )
                all_of = concept_groups.get("all_of")
                if not isinstance(all_of, list) or not all_of:
                    errors.append(f"reviewed concept_groups empty: {case.get('case_id')}")
                elif concept_groups != expected_concept_groups:
                    errors.append(
                        f"reviewed concept_groups do not exactly materialize answer_contract: "
                        f"{case.get('case_id')}"
                    )
                else:
                    concept_keys = [_policy_normalized(item.get("concept_key")) for item in all_of]
                    signatures = [
                        tuple(sorted(_policy_normalized(value) for value in item.get("any_of") or []))
                        for item in all_of
                    ]
                    if (
                        "" in concept_keys
                        or len(concept_keys) != len(set(concept_keys))
                        or len(signatures) != len(set(signatures))
                        or any(not signature or "" in signature for signature in signatures)
                    ):
                        errors.append(
                            f"reviewed concept_groups uniqueness failure: {case.get('case_id')}"
                        )
                metadata = case.get("metadata") or {}
                if metadata.get("concept_groups_variant_filter") != expected_filter_audit:
                    errors.append(
                        f"reviewed concept_groups variant-filter audit mismatch: "
                        f"{case.get('case_id')}"
                    )
                contract_hash = _sha256_json(contract)
                concept_groups_hash = _sha256_json(concept_groups)
                if metadata.get("reviewed_contract_hash") != contract_hash:
                    errors.append(f"reviewed contract hash mismatch: {case.get('case_id')}")
                if metadata.get("concept_groups_hash") != concept_groups_hash:
                    errors.append(f"reviewed concept_groups hash mismatch: {case.get('case_id')}")
                if metadata.get("concept_groups_source_contract_hash") != contract_hash:
                    errors.append(
                        f"reviewed concept_groups source trace mismatch: {case.get('case_id')}"
                    )
                coverage_errors = _contract_coverage_errors(
                    contract,
                    list(case.get("relevant_evidence") or []),
                    expected_branch_spans=[
                        str(item.get("query_span") or "")
                        for item in active_payload.get("branches") or []
                    ],
                    expected_unanswerable=bool(active_payload.get("unanswerable")),
                )
                if coverage_errors:
                    errors.append(
                        f"reviewed contract coverage failure {case.get('case_id')}: "
                        f"{'; '.join(coverage_errors)}"
                    )
        if len(reviewed_cases) != 150 or len({case["case_id"] for case in reviewed_cases}) != 150:
            errors.append("reviewed manifests must contain 150 unique v2 case ids")

        reviewed_qa = json.loads(
            (ROOT / "reviewed" / "qa-reviewed.json").read_text(encoding="utf-8")
        )
        errors.extend(
            _reviewed_parent_coverage_errors(
                expected_case_ids_by_difficulty=candidate_case_ids_by_difficulty,
                expected_parent_by_case_id=candidate_parent_by_case_id,
                reviewed_cases_by_difficulty=reviewed_cases_by_difficulty,
                reviewed_qa=reviewed_qa,
                active_payload_parent_ids=set(active_payload_by_parent),
            )
        )
        if len(reviewed_qa) != 150 or len({row["case_id"] for row in reviewed_qa}) != 150:
            errors.append("reviewed QA must contain 150 unique case rows")
        reviewed_case_by_id = {str(case["case_id"]): case for case in reviewed_cases}
        for row in reviewed_qa:
            case = reviewed_case_by_id.get(str(row.get("case_id") or ""))
            if case is None:
                errors.append(f"reviewed QA has unknown case: {row.get('case_id')}")
                continue
            contract_hash = _sha256_json(case.get("answer_contract") or {})
            concept_groups_hash = _sha256_json(case.get("concept_groups") or {})
            if (
                row.get("answer_contract_hash") != contract_hash
                or row.get("concept_groups_source_contract_hash") != contract_hash
                or row.get("concept_groups_hash") != concept_groups_hash
                or int(row.get("concept_group_count") or 0) <= 0
            ):
                errors.append(f"reviewed QA materialization trace mismatch: {row.get('case_id')}")
            parent_case_id = str(row.get("parent_case_id") or "")
            expected_extra_paths = (
                block_union_audit.get("review_schema_extra_paths_by_case") or {}
            ).get(parent_case_id, [])
            if parent_case_id in schema_extra_parent_ids:
                if (
                    row.get("contract_rereview_applied") is not True
                    or row.get("original_review_schema_extra_paths")
                    != expected_extra_paths
                    or row.get("original_schema_extra_semantics_reused") is not False
                ):
                    errors.append(
                        f"schema-extra original review was not strictly superseded: "
                        f"{row.get('case_id')}"
                    )
            if parent_case_id in normalized_parent_ids:
                expected_actions = (
                    block_union_audit.get("structural_normalization_actions_by_case") or {}
                ).get(parent_case_id, [])
                if (
                    row.get("contract_rereview_applied") is not True
                    or row.get("original_review_structural_normalization_actions")
                    != expected_actions
                    or row.get("original_normalized_semantics_reused") is not False
                ):
                    errors.append(
                        f"structurally normalized original review was not superseded: "
                        f"{row.get('case_id')}"
                    )
        verdict_counts: dict[str, int] = {}
        for row in reviewed_qa:
            verdict = str(row.get("verdict") or "")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if dict(sorted(verdict_counts.items())) != reviewed_summary.get("verdict_counts"):
            errors.append("reviewed summary verdict counts do not match reviewed QA")
        if reviewed_summary.get("blocked_case_count") != verdict_counts.get("block", 0):
            errors.append("reviewed summary blocked count does not match reviewed QA")
        if reviewed_summary.get("concept_groups_materialized_case_count") != 150:
            errors.append("reviewed summary must record 150 materialized concept_groups")
        filtered_count = sum(
            int(row.get("filtered_weak_variant_count") or 0) for row in reviewed_qa
        )
        filtered_case_ids = [
            str(row["case_id"])
            for row in reviewed_qa
            if int(row.get("filtered_weak_variant_count") or 0) > 0
        ]
        if (
            reviewed_summary.get("filtered_weak_variant_count") != filtered_count
            or reviewed_summary.get("filtered_weak_variant_case_ids") != filtered_case_ids
            or reviewed_summary.get("discriminative_variant_policy_version")
            != MATERIALIZATION_POLICY_VERSION
        ):
            errors.append("reviewed discriminative-variant filter summary mismatch")
        coverage_status_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get("same_response_coverage_status_canonicalization"), dict
            )
        )
        if (
            reviewed_summary.get(
                "same_response_coverage_status_canonicalization_case_count"
            )
            != len(coverage_status_parent_ids)
            or reviewed_summary.get(
                "same_response_coverage_status_canonicalization_parent_case_ids"
            )
            != coverage_status_parent_ids
            or coverage_status_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "coverage_status_canonicalization_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed coverage-status canonicalization summary mismatch")
        composition_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get(
                    "same_response_nonsemantic_canonicalization_composition"
                ),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "same_response_nonsemantic_canonicalization_composition_case_count"
            )
            != len(composition_parent_ids)
            or reviewed_summary.get(
                "same_response_nonsemantic_canonicalization_composition_parent_case_ids"
            )
            != composition_parent_ids
            or composition_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "nonsemantic_canonicalization_composition_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed nonsemantic composition summary mismatch")
        quote_fragment_composition_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get(
                    "same_response_quote_fragment_inverse_canonicalization_composition"
                ),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "same_response_quote_fragment_inverse_canonicalization_composition_case_count"
            )
            != len(quote_fragment_composition_parent_ids)
            or reviewed_summary.get(
                "same_response_quote_fragment_inverse_canonicalization_composition_parent_case_ids"
            )
            != quote_fragment_composition_parent_ids
            or quote_fragment_composition_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "quote_fragment_inverse_composition_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed quote-fragment inverse composition summary mismatch")
        exact_span_quote_prefix_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get(
                    "saved_response_exact_payload_branch_span_quote_prefix_recovery"
                ),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "saved_response_exact_payload_branch_span_quote_prefix_recovery_case_count"
            )
            != len(exact_span_quote_prefix_parent_ids)
            or reviewed_summary.get(
                "saved_response_exact_payload_branch_span_quote_prefix_recovery_parent_case_ids"
            )
            != exact_span_quote_prefix_parent_ids
            or exact_span_quote_prefix_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "exact_payload_branch_span_quote_prefix_recovery_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed exact payload branch-span recovery summary mismatch")
        abstain_factual_constraint_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get("saved_response_abstain_factual_constraint_recovery"),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "saved_response_abstain_factual_constraint_recovery_case_count"
            )
            != len(abstain_factual_constraint_parent_ids)
            or reviewed_summary.get(
                "saved_response_abstain_factual_constraint_recovery_parent_case_ids"
            )
            != abstain_factual_constraint_parent_ids
            or abstain_factual_constraint_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "abstain_factual_constraint_recovery_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed abstain factual-constraint recovery summary mismatch")
        redundant_field_composition_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get(
                    "saved_response_redundant_field_coverage_composition"
                ),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "saved_response_redundant_field_coverage_composition_case_count"
            )
            != len(redundant_field_composition_parent_ids)
            or reviewed_summary.get(
                "saved_response_redundant_field_coverage_composition_parent_case_ids"
            )
            != redundant_field_composition_parent_ids
            or redundant_field_composition_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "redundant_field_coverage_composition_case_ids"
                )
                or []
            )
        ):
            errors.append(
                "reviewed redundant-field coverage composition summary mismatch"
            )
        saved_variant_subset_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get(
                    "saved_response_same_concept_variant_subset_recovery"
                ),
                dict,
            )
        )
        if (
            reviewed_summary.get(
                "saved_response_same_concept_variant_subset_recovery_case_count"
            )
            != len(saved_variant_subset_parent_ids)
            or reviewed_summary.get(
                "saved_response_same_concept_variant_subset_recovery_parent_case_ids"
            )
            != saved_variant_subset_parent_ids
            or reviewed_summary.get(
                "saved_response_same_concept_variant_subset_recovery_disposition"
            )
            != (
                "adopted_only_after_exact_latest_failed_row_same_concept_existing_"
                "any_of_subset_replay_without_api_consensus_or_cross_attempt_merge"
            )
        ):
            errors.append(
                "reviewed saved same-concept variant-subset recovery summary mismatch"
            )
        saved_quote_key_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get("saved_response_quote_key_branch_span_recovery"),
                dict,
            )
        )
        quote_key_summary_fields = {
            "saved_response_quote_key_branch_span_recovery_case_count",
            "saved_response_quote_key_branch_span_recovery_parent_case_ids",
            "saved_response_quote_key_branch_span_recovery_disposition",
        }
        if (saved_quote_key_parent_ids or quote_key_summary_fields & set(reviewed_summary)) and (
            reviewed_summary.get(
                "saved_response_quote_key_branch_span_recovery_case_count"
            )
            != len(saved_quote_key_parent_ids)
            or reviewed_summary.get(
                "saved_response_quote_key_branch_span_recovery_parent_case_ids"
            )
            != saved_quote_key_parent_ids
            or reviewed_summary.get(
                "saved_response_quote_key_branch_span_recovery_disposition"
            )
            != (
                "adopted_only_after_exact_latest_failed_row_quoted_schema_key_"
                "removal_and_same_request_exact_span_replay_without_api_consensus_"
                "or_cross_attempt_merge"
            )
        ):
            errors.append(
                "reviewed saved quote-key branch-span recovery summary mismatch"
            )
        unsupported_block_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get("same_response_unsupported_required_concept_block"), dict
            )
        )
        if (
            reviewed_summary.get(
                "same_response_unsupported_required_concept_block_case_count"
            )
            != len(unsupported_block_parent_ids)
            or reviewed_summary.get(
                "same_response_unsupported_required_concept_block_parent_case_ids"
            )
            != unsupported_block_parent_ids
            or unsupported_block_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "unsupported_required_concept_block_case_ids"
                )
                or []
            )
        ):
            errors.append(
                "reviewed unsupported-required-concept block summary mismatch"
            )
        empty_branch_block_parent_ids = sorted(
            str(row.get("parent_case_id") or "")
            for row in reviewed_qa
            if isinstance(
                row.get("multi_response_empty_required_branch_block"), dict
            )
        )
        if (
            reviewed_summary.get(
                "multi_response_empty_required_branch_block_case_count"
            )
            != len(empty_branch_block_parent_ids)
            or reviewed_summary.get(
                "multi_response_empty_required_branch_block_parent_case_ids"
            )
            != empty_branch_block_parent_ids
            or empty_branch_block_parent_ids
            != sorted(
                str(value)
                for value in block_union_audit.get(
                    "empty_required_branch_block_case_ids"
                )
                or []
            )
        ):
            errors.append("reviewed empty-required-branch block summary mismatch")
        if (
            int(reviewed_summary.get("blocker_rereview_case_count") or 0)
            != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
            or reviewed_summary.get("blocker_remediation_applied_case_count")
            != reviewed_summary.get("blocker_rereview_case_count")
            or reviewed_summary.get("superseded_original_review_reused") is not False
        ):
            errors.append("reviewed summary must record every independent blocker remediation")
        expected_changed_cases = int(reviewed_summary.get("blocker_rereview_case_count") or 0)
        remediation_rows = [
            row for row in reviewed_qa if row.get("blocker_remediation_applied") is True
        ]
        if (
            len(remediation_rows) != expected_changed_cases
            or any(row.get("superseded_original_review_reused") is not False for row in remediation_rows)
            or any(not row.get("superseded_original_review_custom_id") for row in remediation_rows)
        ):
            errors.append("reviewed QA blocker-remediation lineage is incomplete")

        expected_contract_rereviews = int(
            reviewed_summary.get("contract_rereview_case_count") or 0
        )
        contract_rereview_rows = [
            row for row in reviewed_qa if row.get("contract_rereview_applied") is True
        ]
        if (
            len(contract_rereview_rows) != expected_contract_rereviews
            or any(
                row.get("contract_rereview_semantic_payload_unchanged") is not True
                or row.get("contract_rereview_dataset_case_unchanged") is not True
                or row.get("superseded_original_review_reused") is not False
                or not row.get("superseded_original_review_custom_id")
                for row in contract_rereview_rows
            )
            or reviewed_summary.get("contract_rereview_semantic_payload_unchanged")
            is not True
            or reviewed_summary.get("contract_rereview_dataset_cases_changed") is not False
        ):
            errors.append("reviewed QA same-semantic contract-rereview lineage is incomplete")

        blocker_responses = _read_jsonl(ROOT / "blocker-rereview" / "responses.jsonl")
        blocker_candidates = json.loads(
            (ROOT / "blocker-rereview" / "candidates.json").read_text(encoding="utf-8")
        )
        blocker_candidate_by_id = {
            str(case.get("case_id") or ""): case for case in blocker_candidates
        }
        current_proposal_sha_by_parent: dict[str, str] = {}
        for document in (base_proposals_document, supplemental_proposals_document):
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
                    errors.append("current blocker proposal identity is invalid")
                    continue
                current_proposal_sha_by_parent[parent_case_id] = proposed_case_sha256
        if (
            len(blocker_requests) != expected_changed_cases
            or len({row["custom_id"] for row in blocker_requests}) != expected_changed_cases
            or {row["prompt_sha256"] for row in blocker_requests}
            != {reviewed_summary.get("prompt_sha256")}
        ):
            errors.append("changed blocker-case request pack invariant failed")
        mapped_blocker_parent_ids: set[str] = set()
        for mapping in blocker_request_map:
            custom_id = str(mapping.get("custom_id") or "")
            request = blocker_request_by_id.get(custom_id)
            candidate = blocker_candidate_by_id.get(str(mapping.get("case_id") or ""))
            parent_case_id = str(mapping.get("parent_case_id") or "")
            mapped_blocker_parent_ids.add(parent_case_id)
            payload = (
                json.loads(request["request"]["messages"][1]["content"])
                if request is not None
                else {}
            )
            candidate_metadata = (candidate or {}).get("metadata") or {}
            if (
                request is None
                or candidate is None
                or _sha256_json(candidate) != mapping.get("candidate_case_sha256")
                or _sha256_json(payload) != request.get("payload_sha256")
                or request.get("payload_sha256") != mapping.get("payload_sha256")
                or str(candidate_metadata.get("parent_case_id") or "") != parent_case_id
                or str(candidate_metadata.get("difficulty") or "")
                != str(mapping.get("difficulty") or "")
                or mapping.get("proposed_case_sha256")
                != current_proposal_sha_by_parent.get(parent_case_id)
                or candidate_metadata.get("blocker_remediation_proposal_sha256")
                != current_proposal_sha_by_parent.get(parent_case_id)
                or _semantic_leaks(payload)
            ):
                errors.append(f"changed blocker candidate/request lineage mismatch: {parent_case_id}")
        if mapped_blocker_parent_ids != set(
            block_union_audit.get("required_data_remediation_case_ids") or []
        ):
            errors.append("changed blocker targets do not match reviewer/visual data union")
        blocker_formal_success: dict[str, dict[str, Any]] = {}
        try:
            blocker_formal_success = _formal_post150_success_ledger(
                blocker_responses,
                blocker_request_by_id,
                execution_contract_sha256=blocker_execution_contract,
                expected_prompt_hash=str(reviewed_summary.get("prompt_sha256") or ""),
                label="validator changed blocker-rereview ledger",
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        blocker_strict_errors, blocker_success = _strict_success_response_errors(
            list(blocker_formal_success.values()),
            blocker_request_by_id,
            expected_prompt_hash=str(reviewed_summary.get("prompt_sha256") or ""),
            pack_name="changed blocker-case rereview",
        )
        errors.extend(blocker_strict_errors)
        if blocker_success != {str(row["custom_id"]) for row in blocker_requests}:
            errors.append("changed blocker-case review is not fully complete")
        if any(
            row.get("status") == "succeeded"
            and str(row.get("prompt_sha256") or "")
            != str(reviewed_summary.get("prompt_sha256") or "")
            for row in blocker_responses
        ):
            errors.append("changed blocker-case response prompt hash drift")

        contract_requests = _read_jsonl(ROOT / "contract-rereview" / "requests.jsonl")
        contract_request_map = json.loads(
            (ROOT / "contract-rereview" / "request-map.json").read_text(encoding="utf-8")
        )
        contract_responses = _read_jsonl(ROOT / "contract-rereview" / "responses.jsonl")
        original_request_by_id = {str(row["custom_id"]): row for row in requests}
        contract_request_by_id = {
            str(row.get("custom_id") or ""): row for row in contract_requests
        }
        contract_mapping_by_id = {
            str(row.get("custom_id") or ""): row for row in contract_request_map
        }
        contract_execution_contract = ""
        try:
            contract_prepared_summary = json.loads(
                (ROOT / "contract-rereview" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_contract_rereview_prepared_receipt(contract_prepared_summary)
            validate_request_pack_identity(
                contract_requests,
                expected_prompt_sha256=str(reviewed_summary.get("prompt_sha256") or ""),
                expected_count=expected_contract_rereviews,
                expected_pack_sha256=str(
                    contract_prepared_summary.get("request_pack_sha256") or ""
                ),
                custom_id_mode="contract_rereview",
                original_custom_ids_by_rereview_id={
                    custom_id: str(mapping.get("original_custom_id") or "")
                    for custom_id, mapping in contract_mapping_by_id.items()
                },
                pack_name="validator same-semantic contract-rereview pack",
            )
            contract_execution_contract = validate_rereview_request_policy(
                contract_requests,
                runner_version=RUNNER_VERSION,
                transport=TRANSPORT,
                sse_completion_policy=SSE_COMPLETION_POLICY,
            )
            contract_execution_summary = json.loads(
                (ROOT / "contract-rereview" / "execution-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            ordered_contract_queue = [
                str(value)
                for value in block_union_audit.get("contract_rereview_case_ids") or []
            ]
            if (
                contract_prepared_summary.get("execution_contract_sha256")
                != contract_execution_contract
                or contract_execution_summary.get("execution_contract_sha256")
                != contract_execution_contract
                or contract_prepared_summary.get("block_union_audit_sha256")
                != sha256_json(block_union_audit)
                or contract_prepared_summary.get("parent_case_ids")
                != ordered_contract_queue
                or contract_prepared_summary.get("parent_case_ids_sha256")
                != sha256_json(ordered_contract_queue)
            ):
                errors.append("same-semantic contract-rereview receipt is stale")
        except RuntimeError as exc:
            errors.append(str(exc))
        if (
            len(contract_requests) != expected_contract_rereviews
            or len(contract_request_by_id) != expected_contract_rereviews
            or set(contract_request_by_id) != set(contract_mapping_by_id)
            or set(contract_request_by_id) & set(original_request_by_id)
            or {str(row.get("prompt_sha256") or "") for row in contract_requests}
            != {reviewed_summary.get("prompt_sha256")}
        ):
            errors.append("same-semantic contract-rereview request pack invariant failed")
        mapped_contract_parent_ids: set[str] = set()
        for custom_id, mapping in contract_mapping_by_id.items():
            original = original_request_by_id.get(str(mapping.get("original_custom_id") or ""))
            current = contract_request_by_id.get(custom_id)
            parent_case_id = str(mapping.get("parent_case_id") or "")
            mapped_contract_parent_ids.add(parent_case_id)
            if (
                original is None
                or current is None
                or current.get("request", {}).get("messages")
                != original.get("request", {}).get("messages")
                or current.get("payload_sha256") != original.get("payload_sha256")
                or original_parent_by_id.get(str(mapping.get("original_custom_id") or ""))
                != parent_case_id
                or mapping.get("semantic_payload_unchanged") is not True
                or mapping.get("dataset_case_unchanged") is not True
                or _semantic_leaks(
                    json.loads(current["request"]["messages"][1]["content"])
                    if current is not None
                    else {}
                )
            ):
                errors.append(f"contract-rereview semantic drift: {parent_case_id}")
        if mapped_contract_parent_ids != set(
            block_union_audit.get("contract_rereview_case_ids") or []
        ):
            errors.append("contract-rereview targets do not match materialization audit")
        contract_formal_success: dict[str, dict[str, Any]] = {}
        try:
            contract_formal_success = _formal_post150_success_ledger(
                contract_responses,
                contract_request_by_id,
                execution_contract_sha256=contract_execution_contract,
                expected_prompt_hash=str(reviewed_summary.get("prompt_sha256") or ""),
                label="validator same-semantic contract-rereview ledger",
            )
        except RuntimeError as exc:
            errors.append(str(exc))
        contract_strict_errors, contract_success = _strict_success_response_errors(
            list(contract_formal_success.values()),
            contract_request_by_id,
            expected_prompt_hash=str(reviewed_summary.get("prompt_sha256") or ""),
            pack_name="same-semantic contract rereview",
        )
        errors.extend(contract_strict_errors)
        if contract_success != set(contract_request_by_id):
            errors.append("same-semantic contract rereview is not fully complete")
        if any(
            row.get("status") == "succeeded"
            and str(row.get("prompt_sha256") or "")
            != str(reviewed_summary.get("prompt_sha256") or "")
            for row in contract_responses
        ):
            errors.append("same-semantic contract response prompt hash drift")

        expected_adoption_by_parent: dict[str, dict[str, Any]] = {}
        blocker_parent_by_id = {
            str(mapping.get("custom_id") or ""): str(
                mapping.get("parent_case_id") or ""
            )
            for mapping in blocker_request_map
        }
        contract_parent_by_id = {
            custom_id: str(mapping.get("parent_case_id") or "")
            for custom_id, mapping in contract_mapping_by_id.items()
        }
        for response_rows, formal_success, parent_by_id in (
            (blocker_responses, blocker_formal_success, blocker_parent_by_id),
            (contract_responses, contract_formal_success, contract_parent_by_id),
        ):
            for custom_id, authoritative_success in formal_success.items():
                try:
                    adoption = materialization_adoption_for_authoritative_success(
                        response_rows, authoritative_success
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
                    continue
                if adoption is None:
                    continue
                parent_case_id = parent_by_id.get(custom_id, "")
                if (
                    not parent_case_id
                    or parent_case_id in expected_adoption_by_parent
                ):
                    errors.append(
                        "formal materialization adoption parent lineage is invalid"
                    )
                    continue
                expected_adoption_by_parent[parent_case_id] = adoption
        errors.extend(
            _materialization_adoption_reporting_errors(
                reviewed_qa=reviewed_qa,
                reviewed_summary=reviewed_summary,
                expected_by_parent_case_id=expected_adoption_by_parent,
            )
        )
        unsuperseded_invalid_originals = set(independently_invalid_originals) - (
            mapped_blocker_parent_ids | mapped_contract_parent_ids
        )
        if unsuperseded_invalid_originals:
            errors.append(
                "strict-invalid original reviews were not superseded: "
                f"{sorted(unsuperseded_invalid_originals)}"
            )
        contract_audit = json.loads(
            (ROOT / "contract-rereview" / "audit.json").read_text(encoding="utf-8")
        )
        contract_audit_inputs = contract_audit.get("inputs") or {}
        if (
            contract_audit.get("status") != "passed"
            or contract_audit.get("case_count") != expected_contract_rereviews
            or contract_audit.get("reviewer_contract_failure_count") != 0
            or contract_audit_inputs.get("requests_sha256")
            != _sha256_file(ROOT / "contract-rereview" / "requests.jsonl")
            or contract_audit_inputs.get("request_map_sha256")
            != _sha256_file(ROOT / "contract-rereview" / "request-map.json")
            or contract_audit_inputs.get("responses_sha256")
            != _sha256_file(ROOT / "contract-rereview" / "responses.jsonl")
            or contract_audit_inputs.get("execution_summary_sha256")
            != _sha256_file(ROOT / "contract-rereview" / "execution-summary.json")
        ):
            errors.append("same-semantic contract-rereview audit is stale or failed")
    elif require_reviewed:
        errors.append("reviewed artifacts are required for final validation")

    result = {
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "candidate_case_count": len(cases),
        "qa_row_count": len(qa_rows),
        "review_request_count": len(requests),
        "checksum_file_count": checksum_checked,
        "schema_validated_contract_count": schema_validated_contract_count,
        "reviewed_artifacts": reviewed_status,
    }
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-reviewed",
        action="store_true",
        help="validate only deterministic candidates before the reviewed merge",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(require_reviewed=not args.allow_missing_reviewed),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
