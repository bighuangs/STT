#!/usr/bin/env python3
"""合并已完成的 easy/medium 权威复审，生成独立 100 题 reviewed 产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import merge_reviewed_contracts as full
import run_easy_medium_nonthinking as nonthinking
from easy_medium_scope import (
    SCOPE_BASELINE_REGISTRATION_ALLOWED,
    SCOPE_CANDIDATE_RUN_ALLOWED,
    SCOPE_CANONICAL_PROMOTION_ALLOWED,
    SCOPE_DIFFICULTIES,
    SCOPE_HARD_REFERENCE_ONLY,
    SCOPE_ID,
    SCOPE_NAME,
    SCOPE_PROMOTABLE,
    ScopeSelection,
    build_blocker_scope_authority,
    build_contract_scope_authority,
    load_frozen_candidate_manifest_scope_authority,
    scope_profile,
    select_scope_requests,
)
from review_release_integrity import (
    EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS,
    materialization_adoption_for_authoritative_success,
    validate_block_union_freshness,
    validate_blocker_rereview_prepared_receipt,
    validate_contract_rereview_prepared_receipt,
    validate_proposal_validation_binding,
    validate_request_pack_identity,
    validate_rereview_request_policy,
)
from validate_easy_medium_reviewed import (
    CHECKSUM_FILENAME,
    DATASET_KEYS,
    EXPECTED_BLOCKER_REREVIEW_COUNTS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNT_PER_DIFFICULTY,
    EXPECTED_CONTRACT_REREVIEW_COUNTS,
    QA_FILENAME,
    REVIEWED_ROOT,
    SUITE_FILENAME,
    SUITE_KEY,
    SUITE_SCHEMA,
    SUMMARY_SCHEMA,
    SUMMARY_FILENAME,
    VERSION,
)


ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = ROOT.parents[4]
REVIEW_ROOT = ROOT / "review"
BLOCKER_REREVIEW_ROOT = ROOT / "blocker-rereview"
CONTRACT_REREVIEW_ROOT = ROOT / "contract-rereview"
SOURCE_QA_ROOT = ROOT.parent / "suite-source-evaluator-qa"
BLOCKER_ROOT = ROOT.parent / "suite-v2-blocker-remediation"
BASE_PROPOSALS_PATH = BLOCKER_ROOT / "proposals.json"
BASE_PROPOSAL_VALIDATION_PATH = BLOCKER_ROOT / "validation.json"
SUPPLEMENTAL_PROPOSALS_PATH = ROOT / "reviewer-block-supplemental-proposals.json"
VISUAL_AUDIT_PATH = ROOT / "source-visual-audit.json"
BLOCK_UNION_AUDIT_PATH = ROOT / "review-block-union-audit.json"
QA_SCHEMA = "stt.eval.easy-medium-reviewed-qa.v1"


class ScopedMergeError(RuntimeError):
    """表示 E/M scoped merge 缺少权威输入或出现谱系漂移。"""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScopedMergeError(f"invalid JSON: {path}") from exc


def _read_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    return full._read_jsonl(path) if path.is_file() else []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    return full._sha256_json(value)


def _selected_response_errors(
    response: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_rows: Sequence[dict[str, Any]],
) -> list[str]:
    """重放 full merge 已定义的全部严格 response 校验。"""

    errors = full._validate_review(dict(response.get("review") or {}))
    errors.extend(full._validate_against_payload(dict(response.get("review") or {}), payload))
    validators = (
        full._validate_inverse_edge_canonicalization_audit,
        full._validate_exact_variant_deduplication_audit,
        full._validate_coverage_status_canonicalization_audit,
        full._validate_nonsemantic_canonicalization_composition_audit,
        full._validate_quote_fragment_inverse_composition_audit,
        full._validate_abstain_factual_constraint_recovery_audit,
        full._validate_exact_payload_branch_span_quote_prefix_recovery_audit,
        full._validate_redundant_field_coverage_composition_audit,
        full._validate_unsupported_required_concept_block_audit,
        full._validate_empty_required_branch_block_audit,
        full._validate_saved_strict_revalidation_audit,
    )
    if response.get("execution_profile_id") in {
        nonthinking.profile.PROFILE_ID,
        nonthinking.profile.V4_PROFILE_ID,
        nonthinking.profile.V5_PROFILE_ID,
    }:
        if response.get("execution_profile_id") == nonthinking.profile.V4_PROFILE_ID:
            adoption = response.get("v5_materializer_adoption")
            if (
                not isinstance(adoption, Mapping)
                or adoption.get("policy") != nonthinking.profile.V4_ADOPTION_POLICY
                or adoption.get("api_call_performed") is not False
                or adoption.get("materializer_policy_version")
                != nonthinking.profile.MATERIALIZATION_POLICY_VERSION
                or adoption.get("production_matcher_policy_version")
                != nonthinking.profile.PRODUCTION_MATCHER_POLICY_VERSION
            ):
                errors.append("v4 adopted response lineage is invalid")
        if response.get("execution_profile_id") == nonthinking.profile.V5_PROFILE_ID:
            adoption = response.get("v6_materializer_adoption")
            if (
                not isinstance(adoption, Mapping)
                or set(adoption)
                != {
                    "policy",
                    "api_call_performed",
                    "source_started_event_sha256",
                    "source_completed_event_sha256",
                    "source_completion_state",
                    "source_materializer_policy_version",
                    "materializer_policy_version",
                    "production_matcher_policy_version",
                }
                or adoption.get("policy") != nonthinking.profile.V5_ADOPTION_POLICY
                or adoption.get("api_call_performed") is not False
                or adoption.get("source_completion_state")
                != "result_without_completed"
                or adoption.get("source_completed_event_sha256") is not None
                or adoption.get("source_materializer_policy_version")
                != nonthinking.profile.V5_MATERIALIZATION_POLICY_VERSION
                or adoption.get("materializer_policy_version")
                != nonthinking.profile.MATERIALIZATION_POLICY_VERSION
                or adoption.get("production_matcher_policy_version")
                != nonthinking.profile.PRODUCTION_MATCHER_POLICY_VERSION
                or not isinstance(adoption.get("source_started_event_sha256"), str)
                or len(adoption["source_started_event_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in adoption["source_started_event_sha256"]
                )
            ):
                errors.append("v5 adopted response lineage is invalid")
        errors.extend(
            full._review_materialization_errors(
                dict(response.get("review") or {}),
                payload,
                materialization_policy_version=(
                    nonthinking.profile.MATERIALIZATION_POLICY_VERSION
                ),
            )
        )
    else:
        for validator in validators:
            errors.extend(
                validator(response, payload, source_response_rows=source_rows)
            )
    if response.get("transport") == full.SAVED_RESPONSE_TRANSPORT:
        # saved profile 已在 formal reducer 中按原请求完整重放；这里不重新猜测请求。
        pass
    if any(str(item).strip() for item in response.get("structural_normalization") or []):
        errors.append("successful scoped rereview required structural normalization")
    return errors


def _load_validated_original_scope_inputs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    ScopeSelection,
]:
    """用冻结候选题绑定原始150请求及其lineage映射。"""

    requests = full._read_jsonl(REVIEW_ROOT / "requests.jsonl")
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=full.EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=full.EXPECTED_CASES,
        pack_name="scoped original full upstream pack",
    )
    request_map = _read_json(REVIEW_ROOT / "request-map.json")
    if not isinstance(request_map, list):
        raise ScopedMergeError("original review request-map is invalid")
    authority = load_frozen_candidate_manifest_scope_authority(ROOT)
    selection = select_scope_requests(
        requests,
        request_map,
        SCOPE_NAME,
        authority_rows=authority,
    )
    return requests, request_map, selection


def _load_scope_original_reviews() -> tuple[
    dict[str, dict[str, Any]],
    str,
    dict[str, Any],
]:
    """只采用 E/M 原始 review，并用完整物理日志重放其 saved-audit 谱系。"""

    requests_path = REVIEW_ROOT / "requests.jsonl"
    request_map_path = REVIEW_ROOT / "request-map.json"
    responses_path = REVIEW_ROOT / "responses.jsonl"
    requests, request_map, selection = _load_validated_original_scope_inputs()
    if (
        selection.target_case_count != EXPECTED_CASE_COUNT
        or selection.target_difficulty_counts
        != {
            difficulty: EXPECTED_CASE_COUNT_PER_DIFFICULTY
            for difficulty in SCOPE_DIFFICULTIES
        }
    ):
        raise ScopedMergeError("original E/M review selection is not exact 2x50")
    target_ids = set(selection.target_custom_ids)
    full_ids = {str(row.get("custom_id") or "") for row in requests}
    response_rows = full._read_jsonl(responses_path)
    successful: dict[str, dict[str, Any]] = {}
    full_successful: dict[str, dict[str, Any]] = {}
    for row in response_rows:
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        if custom_id not in full_ids:
            raise ScopedMergeError(
                f"original successful response is outside full request pack: {custom_id}"
            )
        if row.get("prompt_sha256") != full.EXPECTED_REVIEW_PROMPT_SHA256:
            raise ScopedMergeError(f"original full response prompt hash drift: {custom_id}")
        if not isinstance(row.get("review"), dict):
            raise ScopedMergeError(f"original full success has no review: {custom_id}")
        prior = full_successful.get(custom_id)
        if prior is not None:
            if _sha256_json(prior["review"]) != _sha256_json(row["review"]):
                raise ScopedMergeError(f"conflicting original full successes: {custom_id}")
            continue
        full_successful[custom_id] = row
        if custom_id in target_ids:
            successful[custom_id] = row
    if set(full_successful) != full_ids:
        raise ScopedMergeError(
            "original full 150 response selection is incomplete or contains foreign ids"
        )
    missing = [
        custom_id for custom_id in selection.target_custom_ids if custom_id not in successful
    ]
    if missing:
        raise ScopedMergeError(
            f"original E/M review pack is incomplete: {len(successful)}/100"
        )
    request_by_id = {str(row["custom_id"]): row for row in requests}
    for custom_id, response in successful.items():
        payload = json.loads(
            request_by_id[custom_id]["request"]["messages"][1]["content"]
        )
        errors: list[str] = []
        validators = (
            full._validate_saved_strict_revalidation_audit,
            full._validate_coverage_status_canonicalization_audit,
            full._validate_nonsemantic_canonicalization_composition_audit,
            full._validate_quote_fragment_inverse_composition_audit,
            full._validate_abstain_factual_constraint_recovery_audit,
            full._validate_exact_payload_branch_span_quote_prefix_recovery_audit,
            full._validate_redundant_field_coverage_composition_audit,
            full._validate_unsupported_required_concept_block_audit,
            full._validate_empty_required_branch_block_audit,
        )
        for validator in validators:
            errors.extend(
                validator(
                    response,
                    payload,
                    source_response_rows=response_rows,
                )
            )
        if errors:
            raise ScopedMergeError(
                f"invalid original E/M saved-audit {custom_id}: " + "; ".join(errors)
            )
    selected_physical = [
        row
        for row in response_rows
        if str(row.get("custom_id") or "") in target_ids
    ]
    receipt = {
        **selection.receipt(),
        "receipt_kind": "original",
        "selected_case_count": EXPECTED_CASE_COUNT,
        "authoritative_success_count": len(successful),
        "missing_case_count": 0,
        "selected_response_row_sha256_by_id": {
            custom_id: _sha256_json(row) for custom_id, row in successful.items()
        },
        "selected_physical_row_count": len(selected_physical),
        "selected_physical_row_sha256": [
            _sha256_json(row) for row in selected_physical
        ],
        "hard_rows_counted": 0,
        "hard_usage_counted": 0,
        "hard_materialization_episodes_counted": 0,
        "requests_sha256": _sha256_file(requests_path),
        "request_map_sha256_file": _sha256_file(request_map_path),
        "responses_sha256": _sha256_file(responses_path),
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    return successful, full.EXPECTED_REVIEW_PROMPT_SHA256, receipt


def _usage_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {key: 0 for key in keys}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in keys:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key] += value
    return totals


def _authoritative_scope_receipt(
    *,
    kind: str,
    selection: ScopeSelection,
    successful: Mapping[str, dict[str, Any]],
    physical_rows: Sequence[dict[str, Any]],
    requests_path: Path,
    request_map_path: Path,
    responses_path: Path,
    execution_contract_sha256: str,
    nonthinking_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_ids = set(selection.target_custom_ids)
    selected_success = {
        custom_id: successful[custom_id]
        for custom_id in selection.target_custom_ids
        if custom_id in successful
    }
    missing = [
        custom_id
        for custom_id in selection.target_custom_ids
        if custom_id not in selected_success
    ]
    selected_physical = [
        row for row in physical_rows if str(row.get("custom_id") or "") in target_ids
    ]
    physical_hashes: dict[str, list[str]] = {
        custom_id: [] for custom_id in selection.target_custom_ids
    }
    for row in selected_physical:
        physical_hashes[str(row["custom_id"])].append(_sha256_json(row))
    selected_rows = list(selected_success.values())
    adoptions: dict[str, dict[str, Any]] = {}
    for custom_id, row in selected_success.items():
        if row.get("execution_profile_id") in {
            nonthinking.profile.PROFILE_ID,
            nonthinking.profile.PREDECESSOR_PROFILE_ID,
            nonthinking.profile.V4_PROFILE_ID,
            nonthinking.profile.V5_PROFILE_ID,
        }:
            # v2 live与exact-replayed v1 predecessor均已在独立validator中
            # 重放direct materialization，不属于旧正式ledger的retry episode。
            if row.get("execution_profile_id") == nonthinking.profile.V4_PROFILE_ID:
                if (
                    nonthinking_receipt is None
                    or custom_id
                    not in set(
                        nonthinking_receipt.get("v4_adopted_success_custom_ids") or []
                    )
                    or (nonthinking_receipt.get("selected_response_sha256_by_id") or {}).get(
                        custom_id
                    )
                    != _sha256_json(row)
                ):
                    raise ScopedMergeError(
                        f"v4 adopted success lacks combined receipt lineage: {custom_id}"
                    )
            if row.get("execution_profile_id") == nonthinking.profile.V5_PROFILE_ID:
                if (
                    nonthinking_receipt is None
                    or custom_id
                    not in set(
                        nonthinking_receipt.get("v5_adopted_success_custom_ids") or []
                    )
                    or (nonthinking_receipt.get("selected_response_sha256_by_id") or {}).get(
                        custom_id
                    )
                    != _sha256_json(row)
                ):
                    raise ScopedMergeError(
                        f"v5 adopted success lacks combined receipt lineage: {custom_id}"
                    )
            continue
        adoption = materialization_adoption_for_authoritative_success(
            physical_rows, row
        )
        if adoption is not None:
            adoptions[custom_id] = adoption
    receipt = {
        **selection.receipt(),
        "receipt_kind": kind,
        "selected_case_count": selection.target_case_count,
        "authoritative_success_count": len(selected_success),
        "missing_case_count": len(missing),
        "missing_custom_ids": missing,
        "selected_response_row_sha256_by_id": {
            custom_id: _sha256_json(row)
            for custom_id, row in selected_success.items()
        },
        "selected_physical_row_sha256_by_id": physical_hashes,
        "selected_physical_row_count": len(selected_physical),
        "selected_usage": _usage_from_rows(selected_rows),
        "selected_materialization_episode_count": len(adoptions),
        "selected_materialization_adoption_by_id": adoptions,
        "hard_rows_counted": 0,
        "hard_usage_counted": 0,
        "hard_materialization_episodes_counted": 0,
        "execution_contract_sha256": execution_contract_sha256,
        "requests_path": requests_path.relative_to(PROJECT_ROOT).as_posix(),
        "requests_sha256": _sha256_file(requests_path),
        "request_map_path": request_map_path.relative_to(PROJECT_ROOT).as_posix(),
        "request_map_sha256_file": _sha256_file(request_map_path),
        "responses_path": responses_path.relative_to(PROJECT_ROOT).as_posix(),
        "responses_sha256": _sha256_file(responses_path) if responses_path.is_file() else None,
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    return receipt


def _validate_full_pack_and_replay(
    *,
    kind: str,
    rereview_root: Path,
) -> tuple[
    ScopeSelection,
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    requests_path = rereview_root / "requests.jsonl"
    request_map_path = rereview_root / "request-map.json"
    responses_path = rereview_root / "responses.jsonl"
    prepared_path = rereview_root / "summary.json"
    requests = full._read_jsonl(requests_path)
    request_map = _read_json(request_map_path)
    prepared = _read_json(prepared_path)
    if not isinstance(request_map, list) or not isinstance(prepared, Mapping):
        raise ScopedMergeError(f"{kind} prepared artifacts are invalid")
    if kind == "blocker":
        prepared_receipt = validate_blocker_rereview_prepared_receipt(dict(prepared))
        expected_count = 15
        custom_id_mode = "payload"
        original_ids = None
        candidates = _read_json(rereview_root / "candidates.json")
        if not isinstance(candidates, list):
            raise ScopedMergeError("blocker candidate authority is invalid")
        proposals: list[dict[str, Any]] = []
        for proposal_path in (BASE_PROPOSALS_PATH, SUPPLEMENTAL_PROPOSALS_PATH):
            document = _read_json(proposal_path)
            rows = document.get("proposals") if isinstance(document, Mapping) else None
            if not isinstance(rows, list):
                raise ScopedMergeError("blocker proposal authority is invalid")
            proposals.extend(rows)
        authority = build_blocker_scope_authority(
            candidates,
            request_map=request_map,
            proposals=proposals,
        )
    else:
        validate_contract_rereview_prepared_receipt(dict(prepared))
        prepared_receipt = prepared
        expected_count = 32
        custom_id_mode = "contract_rereview"
        original_ids = {
            str(row.get("custom_id") or ""): str(row.get("original_custom_id") or "")
            for row in request_map
        }
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=full.EXPECTED_REVIEW_PROMPT_SHA256,
        expected_count=expected_count,
        expected_pack_sha256=str(prepared_receipt.get("request_pack_sha256") or ""),
        custom_id_mode=custom_id_mode,
        original_custom_ids_by_rereview_id=original_ids,
        pack_name=f"scoped {kind} full upstream pack",
    )
    if kind == "contract":
        original_requests, original_request_map, _ = (
            _load_validated_original_scope_inputs()
        )
        authority = build_contract_scope_authority(
            requests,
            original_requests,
            original_request_map,
        )
    selection = select_scope_requests(
        requests,
        request_map,
        SCOPE_NAME,
        authority_rows=authority,
    )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=full.RUNNER_VERSION,
        transport=full.TRANSPORT,
        sse_completion_policy=full.SSE_COMPLETION_POLICY,
    )
    if prepared.get("execution_contract_sha256") != execution_contract:
        raise ScopedMergeError(f"{kind} prepared execution contract drifted")
    rows = _read_jsonl_optional(responses_path)
    legacy_successful = full._formal_post150_success_ledger(
        rows,
        {str(row["custom_id"]): row for row in requests},
        execution_contract_sha256=execution_contract,
        expected_prompt_hash=full.EXPECTED_REVIEW_PROMPT_SHA256,
        label=f"scoped {kind} full-ledger replay",
    )
    nonthinking_snapshot = nonthinking.load_release_snapshot()
    if nonthinking_snapshot is None:
        successful = legacy_successful
        response_source_rows = rows
        nonthinking_receipt = None
    else:
        successful = nonthinking_snapshot["combined_authoritative_by_kind"][kind]
        legacy_target_success = {
            custom_id: legacy_successful[custom_id]
            for custom_id in selection.target_custom_ids
            if custom_id in legacy_successful
        }
        for custom_id, legacy_row in legacy_target_success.items():
            if successful.get(custom_id) != legacy_row:
                raise ScopedMergeError(
                    f"nonthinking combined replay did not preserve old success: {custom_id}"
                )
        response_source_rows = [
            *rows,
            *nonthinking._read_jsonl(nonthinking.RESPONSES_PATH),
        ]
        nonthinking_receipt = nonthinking_snapshot["combined_receipt_by_kind"][kind]
    receipt = _authoritative_scope_receipt(
        kind=kind,
        selection=selection,
        successful=successful,
        physical_rows=response_source_rows,
        requests_path=requests_path,
        request_map_path=request_map_path,
        responses_path=responses_path,
        execution_contract_sha256=execution_contract,
        nonthinking_receipt=nonthinking_receipt,
    )
    if nonthinking_receipt is not None:
        receipt["nonthinking_combined_receipt"] = nonthinking_receipt
        receipt["nonthinking_sidecar_sha256"] = nonthinking_snapshot[
            "sidecar_sha256"
        ]
        receipt["receipt_sha256"] = _sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    return selection, successful, response_source_rows, receipt


def scope_readiness() -> dict[str, Any]:
    """返回 E/M 复审缺口；不把 full pack 的 hard 完成度计入范围。"""

    blocker_selection, _, _, blocker = _validate_full_pack_and_replay(
        kind="blocker",
        rereview_root=BLOCKER_REREVIEW_ROOT,
    )
    contract_selection, _, _, contract = _validate_full_pack_and_replay(
        kind="contract",
        rereview_root=CONTRACT_REREVIEW_ROOT,
    )
    if blocker_selection.target_difficulty_counts != EXPECTED_BLOCKER_REREVIEW_COUNTS:
        raise ScopedMergeError("scoped blocker request counts drifted")
    if contract_selection.target_difficulty_counts != EXPECTED_CONTRACT_REREVIEW_COUNTS:
        raise ScopedMergeError("scoped contract request counts drifted")
    missing = blocker["missing_case_count"] + contract["missing_case_count"]
    return {
        "schema_version": "stt.eval.easy-medium-review-readiness.v1",
        "status": "ready" if missing == 0 else "incomplete",
        "scope": scope_profile(),
        "missing_case_count": missing,
        "blocker": blocker,
        "contract": contract,
        "hard_reference_only": SCOPE_HARD_REFERENCE_ONLY,
        "hard_rows_counted": 0,
        "hard_usage_counted": 0,
        "hard_materialization_episodes_counted": 0,
    }


def _load_scope_rereviews() -> tuple[
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    readiness = scope_readiness()
    if readiness["missing_case_count"]:
        raise ScopedMergeError(
            "easy/medium authoritative rereview is incomplete: "
            f"{readiness['missing_case_count']} case(s) missing"
        )
    block_union = _read_json(BLOCK_UNION_AUDIT_PATH)
    validate_block_union_freshness(
        block_union,
        requests_path=REVIEW_ROOT / "requests.jsonl",
        responses_path=REVIEW_ROOT / "responses.jsonl",
        visual_audit_path=VISUAL_AUDIT_PATH,
        base_proposals_path=BASE_PROPOSALS_PATH,
        supplemental_proposals_path=SUPPLEMENTAL_PROPOSALS_PATH,
        label="scoped merge block-union audit",
    )
    proposal_validation = _read_json(BASE_PROPOSAL_VALIDATION_PATH)
    validate_proposal_validation_binding(
        proposal_validation,
        proposals_path=BASE_PROPOSALS_PATH,
        label="scoped merge proposal validation",
    )
    blocker_selection, blocker_success, blocker_rows, _ = _validate_full_pack_and_replay(
        kind="blocker", rereview_root=BLOCKER_REREVIEW_ROOT
    )
    contract_selection, contract_success, contract_rows, _ = _validate_full_pack_and_replay(
        kind="contract", rereview_root=CONTRACT_REREVIEW_ROOT
    )

    base_proposals = _read_json(BASE_PROPOSALS_PATH)
    supplemental = _read_json(SUPPLEMENTAL_PROPOSALS_PATH)
    base_rows = base_proposals.get("proposals") or []
    supplemental_rows = supplemental.get("proposals") or []
    required_ids = {
        str(value) for value in block_union.get("required_data_remediation_case_ids") or []
    }
    all_proposal_rows = [*base_rows, *supplemental_rows]
    if (
        len(base_rows) != EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
        or len(supplemental_rows) != EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        or len(required_ids) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
    ):
        raise ScopedMergeError("full blocker proposal union drifted")
    proposal_sha_by_parent: dict[str, str] = {}
    for proposal in all_proposal_rows:
        parent_id = str(proposal.get("case_id") or "")
        proposed_case = proposal.get("proposed_case") or {}
        proposed_sha = str((proposal.get("audit") or {}).get("proposed_case_sha256") or "")
        if (
            not parent_id
            or parent_id in proposal_sha_by_parent
            or _sha256_json(proposed_case) != proposed_sha
        ):
            raise ScopedMergeError("blocker proposal identity is invalid")
        proposal_sha_by_parent[parent_id] = proposed_sha

    blocker_map = _read_json(BLOCKER_REREVIEW_ROOT / "request-map.json")
    blocker_candidates = _read_json(BLOCKER_REREVIEW_ROOT / "candidates.json")
    candidate_by_id = {
        str(case.get("case_id") or ""): case for case in blocker_candidates
    }
    blocker_request_by_id = {
        str(row["custom_id"]): row
        for row in full._read_jsonl(BLOCKER_REREVIEW_ROOT / "requests.jsonl")
    }
    target_blocker_ids = set(blocker_selection.target_custom_ids)
    blockers: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for mapping in blocker_map:
        custom_id = str(mapping.get("custom_id") or "")
        if custom_id not in target_blocker_ids:
            continue
        parent_id = str(mapping.get("parent_case_id") or "")
        case_id = str(mapping.get("case_id") or "")
        candidate = candidate_by_id.get(case_id)
        response = blocker_success.get(custom_id)
        request = blocker_request_by_id.get(custom_id)
        if candidate is None or response is None or request is None:
            raise ScopedMergeError(f"scoped blocker lineage is incomplete: {parent_id}")
        metadata = candidate.get("metadata") or {}
        payload = json.loads(request["request"]["messages"][1]["content"])
        expected_proposal_sha = proposal_sha_by_parent.get(parent_id)
        if (
            _sha256_json(candidate) != mapping.get("candidate_case_sha256")
            or mapping.get("proposed_case_sha256") != expected_proposal_sha
            or metadata.get("blocker_remediation_proposal_sha256") != expected_proposal_sha
            or metadata.get("superseded_review_must_not_be_reused") is not True
            or payload != full._expected_review_payload(candidate)
            or mapping.get("payload_sha256") != request.get("payload_sha256")
        ):
            raise ScopedMergeError(f"scoped blocker candidate binding drifted: {parent_id}")
        errors = _selected_response_errors(response, payload, source_rows=blocker_rows)
        if errors:
            raise ScopedMergeError(
                f"scoped blocker response is invalid for {parent_id}: {'; '.join(errors)}"
            )
        blockers[parent_id] = (candidate, response)

    original_requests = {
        str(row["custom_id"]): row for row in full._read_jsonl(REVIEW_ROOT / "requests.jsonl")
    }
    original_map = _read_json(REVIEW_ROOT / "request-map.json")
    original_by_parent = {
        str(row.get("parent_case_id") or ""): original_requests[str(row["custom_id"])]
        for row in original_map
    }
    contract_map = _read_json(CONTRACT_REREVIEW_ROOT / "request-map.json")
    contract_request_by_id = {
        str(row["custom_id"]): row
        for row in full._read_jsonl(CONTRACT_REREVIEW_ROOT / "requests.jsonl")
    }
    target_contract_ids = set(contract_selection.target_custom_ids)
    contracts: dict[str, dict[str, Any]] = {}
    for mapping in contract_map:
        custom_id = str(mapping.get("custom_id") or "")
        if custom_id not in target_contract_ids:
            continue
        parent_id = str(mapping.get("parent_case_id") or "")
        original = original_by_parent.get(parent_id)
        request = contract_request_by_id.get(custom_id)
        response = contract_success.get(custom_id)
        if original is None or request is None or response is None:
            raise ScopedMergeError(f"scoped contract lineage is incomplete: {parent_id}")
        if (
            request.get("request", {}).get("messages")
            != original.get("request", {}).get("messages")
            or request.get("payload_sha256") != original.get("payload_sha256")
            or mapping.get("semantic_payload_unchanged") is not True
            or mapping.get("dataset_case_unchanged") is not True
        ):
            raise ScopedMergeError(f"scoped contract semantic payload drifted: {parent_id}")
        payload = json.loads(request["request"]["messages"][1]["content"])
        errors = _selected_response_errors(response, payload, source_rows=contract_rows)
        if errors:
            raise ScopedMergeError(
                f"scoped contract response is invalid for {parent_id}: {'; '.join(errors)}"
            )
        contracts[parent_id] = response
    if len(blockers) != 4 or len(contracts) != 26:
        raise ScopedMergeError("scoped rereview mapping counts are not exact 4+26")
    return blockers, contracts, readiness


def _source_index_integrity(
    readiness: Mapping[str, Any],
    *,
    original_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source_summary_path = SOURCE_QA_ROOT / "summary.json"
    blocker_validation_path = BLOCKER_ROOT / "validation.json"
    source_summary = _read_json(source_summary_path)
    blocker_validation = _read_json(blocker_validation_path)
    checks = source_summary.get("input_checks") or {}
    active_index = blocker_validation.get("active_index") or {}
    candidate_sources = [
        (_read_json(ROOT / "datasets" / f"{difficulty}.v2.json").get("source") or {})
        for difficulty in SCOPE_DIFFICULTIES
    ]
    frozen_source = candidate_sources[0] if candidate_sources else {}
    frozen_source_identity_fields = (
        "relative_path",
        "content_sha256",
        "page_count",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
    )
    if (
        checks.get("source_hash_matches") is not True
        or checks.get("source_page_count_matches") is not True
        or checks.get("joined_case_count") != 150
        or checks.get("indexed_evidence_count") != 218
        or blocker_validation.get("status") != "passed"
        or blocker_validation.get("selected_chunks_resolve_to_frozen_active_index") is not True
        or active_index.get("frozen_chunk_count") != 218
        or active_index.get("index_rebuild_required") is not False
        or active_index.get("evidence_refresh_required") is not False
        or any(
            any(
                source.get(field) != frozen_source.get(field)
                for field in frozen_source_identity_fields
            )
            for source in candidate_sources
        )
        or frozen_source.get("relative_path") != checks.get("source_path")
        or frozen_source.get("content_sha256") != checks.get("source_sha256")
        or frozen_source.get("page_count") != checks.get("source_pdf_page_count")
        or frozen_source.get("index_version_id") != active_index.get("index_version_id")
    ):
        raise ScopedMergeError("source/index integrity receipts are not frozen")
    return {
        "source": {
            "path": checks.get("source_path"),
            "sha256": checks.get("source_sha256"),
            "page_count": checks.get("source_pdf_page_count"),
        },
        "active_index": {
            "kb_id": frozen_source.get("kb_id"),
            "document_id": frozen_source.get("document_id"),
            "document_version_id": frozen_source.get("document_version_id"),
            "index_version_id": active_index.get("index_version_id"),
            "frozen_chunk_count": active_index.get("frozen_chunk_count"),
            "source_qa_summary_path": source_summary_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_qa_summary_sha256": _sha256_file(source_summary_path),
            "blocker_validation_path": blocker_validation_path.relative_to(PROJECT_ROOT).as_posix(),
            "blocker_validation_sha256": _sha256_file(blocker_validation_path),
        },
        "review": {
            "scope_id": SCOPE_ID,
            "scoped_case_count": EXPECTED_CASE_COUNT,
            "hard_reference_only": SCOPE_HARD_REFERENCE_ONLY,
            "hard_imported_case_count": 0,
            "full_source_joined_case_count_reference_only": 150,
            "block_union_audit_path": BLOCK_UNION_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "block_union_audit_sha256": _sha256_file(BLOCK_UNION_AUDIT_PATH),
            "original_scope_receipt_sha256": original_receipt["receipt_sha256"],
            "blocker_scope_receipt_sha256": readiness["blocker"]["receipt_sha256"],
            "contract_scope_receipt_sha256": readiness["contract"]["receipt_sha256"],
        },
    }


def build() -> dict[str, Any]:
    """在内存中构造完整 100 题产物；任何缺口均在写盘前失败。"""

    # 先报告当前范围的真实缺口，避免 reference-only hard 历史遮蔽 E/M 门禁。
    blockers, contracts, readiness = _load_scope_rereviews()
    original_reviews, prompt_hash, original_receipt = _load_scope_original_reviews()
    original_map = _read_json(REVIEW_ROOT / "request-map.json")
    original_custom_by_case = {
        str(row.get("case_id") or ""): str(row.get("custom_id") or "")
        for row in original_map
        if str(row.get("difficulty") or "") in SCOPE_DIFFICULTIES
    }
    block_union = _read_json(BLOCK_UNION_AUDIT_PATH)
    schema_extra_ids = {
        str(value) for value in block_union.get("review_schema_extra_field_case_ids") or []
    }
    normalized_ids = {
        str(value) for value in block_union.get("structural_normalization_case_ids") or []
    }
    manifests: dict[str, dict[str, Any]] = {}
    qa_rows: list[dict[str, Any]] = []
    used_blockers: set[str] = set()
    used_contracts: set[str] = set()
    for difficulty in SCOPE_DIFFICULTIES:
        candidate_path = ROOT / "datasets" / f"{difficulty}.v2.json"
        candidate_manifest = _read_json(candidate_path)
        reviewed_cases: list[dict[str, Any]] = []
        for candidate_case in candidate_manifest.get("cases") or []:
            case_id = str(candidate_case.get("case_id") or "")
            parent_id = str((candidate_case.get("metadata") or {}).get("parent_case_id") or "")
            original_custom_id = original_custom_by_case.get(case_id)
            if not original_custom_id or original_custom_id not in original_reviews:
                raise ScopedMergeError(f"original review lineage missing: {case_id}")
            if parent_id in blockers:
                active_case, response = blockers[parent_id]
                source_rows = full._read_jsonl(BLOCKER_REREVIEW_ROOT / "responses.jsonl")
                source_kind = "blocker"
                used_blockers.add(parent_id)
            elif parent_id in contracts:
                active_case = candidate_case
                response = contracts[parent_id]
                source_rows = full._read_jsonl(CONTRACT_REREVIEW_ROOT / "responses.jsonl")
                source_kind = "contract"
                used_contracts.add(parent_id)
            else:
                active_case = candidate_case
                response = original_reviews[original_custom_id]
                source_rows = full._read_jsonl(REVIEW_ROOT / "responses.jsonl")
                source_kind = "original"
            active_metadata = active_case.get("metadata")
            if (
                not isinstance(active_metadata, Mapping)
                or str(active_case.get("case_id") or "") != case_id
                or str(active_metadata.get("parent_case_id") or "") != parent_id
                or str(active_metadata.get("difficulty") or "") != difficulty
            ):
                raise ScopedMergeError(
                    f"selected case metadata difficulty/lineage drifted: {case_id}"
                )
            payload = full._expected_review_payload(active_case)
            errors = _selected_response_errors(response, payload, source_rows=source_rows)
            if errors:
                raise ScopedMergeError(
                    f"selected response is not adoptable for {case_id}: {'; '.join(errors)}"
                )
            contract, qa = full._merge_contract(
                candidate_case=active_case,
                response=response,
            )
            full._validate_contract_coverage(
                contract,
                active_case.get("relevant_evidence") or [],
                case_id=case_id,
                expected_branch_spans=[
                    str(branch.get("query_span") or "")
                    for branch in (active_case.get("answer_contract") or {}).get(
                        "required_branches"
                    )
                    or []
                ],
            )
            concept_groups, variant_audit = full._materialize_concept_groups(
                contract,
                case_id=case_id,
                evidence=active_case.get("relevant_evidence") or [],
            )
            contract_hash = _sha256_json(contract)
            groups_hash = _sha256_json(concept_groups)
            reviewed_case = deepcopy(active_case)
            reviewed_case["answer_contract"] = contract
            reviewed_case["concept_groups"] = concept_groups
            reviewed_case["metadata"] = {
                **dict(reviewed_case.get("metadata") or {}),
                "reviewed_contract_hash": contract_hash,
                "concept_groups_hash": groups_hash,
                "concept_groups_source_contract_hash": contract_hash,
                "concept_groups_materializer": "merge_reviewed_contracts.py:v1",
                "concept_groups_variant_filter": variant_audit,
                "easy_medium_scope_id": SCOPE_ID,
            }
            reviewed_cases.append(reviewed_case)
            qa.update(
                {
                    "parent_case_id": parent_id,
                    "difficulty": difficulty,
                    "answer_contract_hash": contract_hash,
                    "concept_groups_hash": groups_hash,
                    "concept_groups_source_contract_hash": contract_hash,
                    "concept_group_count": len(concept_groups["all_of"]),
                    "selected_response_source": source_kind,
                    "selected_response_sha256": _sha256_json(response),
                    "saved_response_quote_key_branch_span_recovery": deepcopy(
                        response.get(
                            "saved_response_quote_key_branch_span_recovery"
                        )
                    ),
                    "blocker_remediation_applied": parent_id in blockers,
                    "contract_rereview_applied": parent_id in contracts,
                    "superseded_original_review_custom_id": (
                        original_custom_id if source_kind != "original" else None
                    ),
                    "superseded_original_review_reused": (
                        False if source_kind != "original" else None
                    ),
                    "original_schema_extra_semantics_reused": (
                        False if parent_id in schema_extra_ids else None
                    ),
                    "original_normalized_semantics_reused": (
                        False if parent_id in normalized_ids else None
                    ),
                    "scope_id": SCOPE_ID,
                }
            )
            qa_rows.append(qa)
        reviewed_manifest = deepcopy(candidate_manifest)
        reviewed_manifest.update(
            {
                "schema_version": "stt.eval.dataset.v2-llm-reviewed-candidate",
                "suite_key": SUITE_KEY,
                "dataset_key": DATASET_KEYS[difficulty],
                "version": VERSION,
                "difficulty": difficulty,
                "scope": scope_profile(),
                "status": "reviewed_candidate",
                "import_allowed": False,
                "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
                "promotable": SCOPE_PROMOTABLE,
                "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
                "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
                "cases": reviewed_cases,
            }
        )
        reviewed_manifest["generation"] = {
            **dict(reviewed_manifest.get("generation") or {}),
            "llm_review_status": "complete",
            "llm_review_prompt_sha256": prompt_hash,
            "independent_validation_status": "pending",
            "answer_contract_to_concept_groups_materialization_status": "complete",
        }
        reviewed_manifest["gate_policy"] = {
            **dict(reviewed_manifest.get("gate_policy") or {}),
            "case_pass_policy": deepcopy(full.FROZEN_CASE_PASS_POLICY),
            "case_pass_policy_activation_status": "reserved_not_active",
            "case_pass_policy_authored_contract_field": "concept_groups",
            "case_pass_policy_materialization_required": True,
        }
        manifests[difficulty] = reviewed_manifest
    if used_blockers != set(blockers) or used_contracts != set(contracts):
        raise ScopedMergeError("not every scoped rereview was applied exactly once")
    if (
        len(qa_rows) != EXPECTED_CASE_COUNT
        or len({str(row.get("case_id") or "") for row in qa_rows}) != EXPECTED_CASE_COUNT
        or len({str(row.get("parent_case_id") or "") for row in qa_rows})
        != EXPECTED_CASE_COUNT
    ):
        raise ScopedMergeError("scoped QA must cover 100 unique case/parent lineages")
    verdict_counts = Counter(str(row.get("verdict") or "") for row in qa_rows)
    if verdict_counts.get("block", 0):
        raise ScopedMergeError("scoped reviewed output still contains blockers")
    source_integrity = _source_index_integrity(
        readiness,
        original_receipt=original_receipt,
    )
    dataset_entries: dict[str, dict[str, Any]] = {}
    for difficulty, manifest in manifests.items():
        relative = f"datasets/{difficulty}.reviewed.json"
        dataset_entries[difficulty] = {
            "path": relative,
            "sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
            "manifest_sha256": _sha256_json(manifest),
            "dataset_key": DATASET_KEYS[difficulty],
            "version": VERSION,
            "case_count": EXPECTED_CASE_COUNT_PER_DIFFICULTY,
        }
    suite = {
        "schema_version": SUITE_SCHEMA,
        "suite_key": SUITE_KEY,
        "version": VERSION,
        "status": "reviewed_candidate_pending_independent_validation",
        "scope": scope_profile(),
        "import_allowed": False,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "datasets": dataset_entries,
        "source_and_index_integrity": source_integrity,
    }
    difficulty_counts = {
        difficulty: sum(row.get("difficulty") == difficulty for row in qa_rows)
        for difficulty in SCOPE_DIFFICULTIES
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "reviewed_candidate_pending_independent_validation",
        "suite_key": SUITE_KEY,
        "scope_id": SCOPE_ID,
        "scope": scope_profile(),
        "case_count": EXPECTED_CASE_COUNT,
        "blocked_case_count": 0,
        "difficulty_case_counts": difficulty_counts,
        "blocker_rereview_counts": EXPECTED_BLOCKER_REREVIEW_COUNTS,
        "contract_rereview_counts": EXPECTED_CONTRACT_REREVIEW_COUNTS,
        "authoritative_scope_receipts": {
            "original": original_receipt,
            "blocker": readiness["blocker"],
            "contract": readiness["contract"],
        },
        "hard_reference_only": SCOPE_HARD_REFERENCE_ONLY,
        "hard_imported_case_count": 0,
        "candidate_run_allowed": SCOPE_CANDIDATE_RUN_ALLOWED,
        "promotable": SCOPE_PROMOTABLE,
        "baseline_registration_allowed": SCOPE_BASELINE_REGISTRATION_ALLOWED,
        "canonical_promotion_allowed": SCOPE_CANONICAL_PROMOTION_ALLOWED,
        "review_prompt_sha256": prompt_hash,
        "verdict_counts": dict(verdict_counts),
        "source_and_index_integrity_sha256": _sha256_json(source_integrity),
    }
    qa_document = qa_rows
    return {
        "suite": suite,
        "summary": summary,
        "qa": qa_document,
        "manifests": manifests,
    }


def build_current_case_review_authority() -> dict[str, dict[str, Any]]:
    """从当前物理审阅账本确定性投影逐题最终 contract 与响应谱系。"""

    built = build()
    manifests = built.get("manifests")
    qa_rows = built.get("qa")
    summary = built.get("summary")
    receipts = (
        summary.get("authoritative_scope_receipts")
        if isinstance(summary, Mapping)
        else None
    )
    if (
        not isinstance(manifests, Mapping)
        or not isinstance(qa_rows, list)
        or not isinstance(receipts, Mapping)
    ):
        raise ScopedMergeError("current case review authority projection is malformed")
    qa_by_case: dict[str, Mapping[str, Any]] = {}
    for row in qa_rows:
        case_id = str(row.get("case_id") or "") if isinstance(row, Mapping) else ""
        if not case_id or case_id in qa_by_case:
            raise ScopedMergeError("current projected QA case coverage is invalid")
        qa_by_case[case_id] = row

    authority: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for difficulty in SCOPE_DIFFICULTIES:
        manifest = manifests.get(difficulty)
        cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
        if not isinstance(cases, list):
            raise ScopedMergeError(
                f"current {difficulty} review authority projection is incomplete"
            )
        for case in cases:
            if not isinstance(case, Mapping):
                raise ScopedMergeError("current projected reviewed case is invalid")
            case_id = str(case.get("case_id") or "")
            metadata = case.get("metadata")
            parent_case_id = (
                str(metadata.get("parent_case_id") or "")
                if isinstance(metadata, Mapping)
                else ""
            )
            contract = case.get("answer_contract")
            contract_review = (
                contract.get("review") if isinstance(contract, Mapping) else None
            )
            qa = qa_by_case.get(case_id)
            if (
                not case_id
                or case_id in authority
                or not parent_case_id
                or not isinstance(contract, Mapping)
                or not isinstance(contract_review, Mapping)
                or not isinstance(qa, Mapping)
            ):
                raise ScopedMergeError(
                    f"current review authority lineage is invalid: {case_id or '<missing>'}"
                )
            source = str(qa.get("selected_response_source") or "")
            custom_id = str(qa.get("custom_id") or "")
            row_sha256 = str(qa.get("selected_response_sha256") or "")
            review_sha256 = str(qa.get("response_sha256") or "")
            prompt_sha256 = str(qa.get("prompt_sha256") or "")
            runner_version = str(qa.get("runner_version") or "")
            receipt = receipts.get(source)
            selected_rows = (
                receipt.get("selected_response_row_sha256_by_id")
                if isinstance(receipt, Mapping)
                else None
            )
            if (
                source not in {"original", "blocker", "contract"}
                or not custom_id
                or len(row_sha256) != 64
                or any(character not in "0123456789abcdef" for character in row_sha256)
                or not isinstance(selected_rows, Mapping)
                or selected_rows.get(custom_id) != row_sha256
                or contract_review.get("custom_id") != custom_id
                or contract_review.get("response_sha256") != review_sha256
                or contract_review.get("prompt_sha256") != prompt_sha256
                or contract_review.get("runner_version") != runner_version
            ):
                raise ScopedMergeError(
                    f"current selected response provenance is invalid: {case_id}"
                )
            authority[case_id] = {
                "parent_case_id": parent_case_id,
                "difficulty": difficulty,
                "answer_contract": deepcopy(dict(contract)),
                "selected_response": {
                    "source": source,
                    "custom_id": custom_id,
                    "row_sha256": row_sha256,
                    "review_sha256": review_sha256,
                    "prompt_sha256": prompt_sha256,
                    "runner_version": runner_version,
                    "quote_key_recovery": deepcopy(
                        qa.get(
                            "saved_response_quote_key_branch_span_recovery"
                        )
                    ),
                },
            }
            source_counts[source] += 1
    if (
        len(authority) != EXPECTED_CASE_COUNT
        or source_counts
        != Counter(
            {
                "original": EXPECTED_CASE_COUNT
                - sum(EXPECTED_BLOCKER_REREVIEW_COUNTS.values())
                - sum(EXPECTED_CONTRACT_REREVIEW_COUNTS.values()),
                "blocker": sum(EXPECTED_BLOCKER_REREVIEW_COUNTS.values()),
                "contract": sum(EXPECTED_CONTRACT_REREVIEW_COUNTS.values()),
            }
        )
    ):
        raise ScopedMergeError(
            "current case review authority coverage/source counts drifted"
        )
    return authority


def _write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_bytes(artifacts: Mapping[str, Any]) -> dict[str, bytes]:
    outputs = {
        SUITE_FILENAME: _json_bytes(artifacts["suite"]),
        SUMMARY_FILENAME: _json_bytes(artifacts["summary"]),
        QA_FILENAME: _json_bytes(artifacts["qa"]),
        **{
            f"datasets/{difficulty}.reviewed.json": _json_bytes(
                artifacts["manifests"][difficulty]
            )
            for difficulty in SCOPE_DIFFICULTIES
        },
    }
    outputs[CHECKSUM_FILENAME] = "".join(
        f"{hashlib.sha256(outputs[relative]).hexdigest()}  {relative}\n"
        for relative in sorted(outputs)
    ).encode("utf-8")
    return outputs


def merge(*, write: bool) -> dict[str, Any]:
    """构造 scoped reviewed 产物，并在显式授权时一次性写入专属目录。"""

    artifacts = build()
    outputs = _output_bytes(artifacts)
    result = {
        "status": "written" if write else "ready_to_write",
        "write_performed": write,
        "scope": scope_profile(),
        "case_count": EXPECTED_CASE_COUNT,
        "hard_reference_only": True,
        "hard_imported_case_count": 0,
        "output_sha256": {
            relative: hashlib.sha256(content).hexdigest()
            for relative, content in outputs.items()
        },
    }
    if not write:
        return result
    existing = {
        path.relative_to(REVIEWED_ROOT).as_posix(): path
        for path in REVIEWED_ROOT.rglob("*")
        if path.is_file()
    } if REVIEWED_ROOT.exists() else {}
    if set(existing) - set(outputs):
        raise ScopedMergeError("scoped reviewed root contains unexpected files")
    for relative, content in outputs.items():
        path = REVIEWED_ROOT / relative
        if path.exists() and path.read_bytes() != content:
            raise ScopedMergeError(f"refusing conflicting scoped reviewed output: {path}")
    for relative, content in outputs.items():
        path = REVIEWED_ROOT / relative
        if not path.exists():
            _write_exact(path, content)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--readiness",
        action="store_true",
        help="只重放 scoped 账本并报告剩余复审，不构造 reviewed 产物",
    )
    args = parser.parse_args()
    try:
        result = scope_readiness() if args.readiness else merge(write=args.write)
    except (OSError, ValueError, RuntimeError, ScopedMergeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
