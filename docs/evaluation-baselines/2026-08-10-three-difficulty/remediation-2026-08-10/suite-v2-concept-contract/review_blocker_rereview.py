"""Review the 15 changed blocker-remediation cases with the frozen prompt.

Execution is refused until the original 150-case review is complete.  Responses
are stored separately so a superseded-case response can never satisfy a changed
case request id.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import review_contracts as runner
from concept_materialization import (
    LEGACY_MATERIALIZATION_POLICY_VERSION,
)
from easy_medium_scope import (
    FULL_SCOPE_NAME,
    SCOPE_NAME,
    ScopeSelection,
    build_blocker_scope_authority,
    scope_completion_errors,
    scope_progress,
    scope_summary_path,
    select_scope_requests,
)
from review_release_integrity import (
    EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
    EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
    EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS,
    EXPECTED_REREVIEW_BASE_URL,
    EXPECTED_REREVIEW_MODEL,
    EXPECTED_REVIEW_PROMPT_SHA256,
    bind_rereview_calibration_receipt,
    execute_materialization_retry_episodes,
    formal_rereview_completion_errors,
    formal_rereview_success_ledger,
    formal_rereview_state_reducer,
    formal_rereview_failed_result_errors,
    materialization_state_summary,
    rereview_calibration_ledger,
    rereview_calibration_custom_id,
    sha256_json,
    validate_block_union_freshness,
    validate_blocker_rereview_prepared_receipt,
    validate_proposal_validation_binding,
    validate_request_pack_identity,
    validate_rereview_request_policy,
    validate_rereview_runtime,
)


ROOT = Path(__file__).resolve().parent
REREVIEW_ROOT = ROOT / "blocker-rereview"
REQUESTS_PATH = REREVIEW_ROOT / "requests.jsonl"
REQUEST_MAP_PATH = REREVIEW_ROOT / "request-map.json"
CANDIDATES_PATH = REREVIEW_ROOT / "candidates.json"
RESPONSES_PATH = REREVIEW_ROOT / "responses.jsonl"
SUMMARY_PATH = REREVIEW_ROOT / "execution-summary.json"
EXPECTED_PROMPT_SHA256 = EXPECTED_REVIEW_PROMPT_SHA256
PROPOSAL_ROOT = ROOT.parent / "suite-v2-blocker-remediation"


def _scope_authority(request_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从已验签proposal与候选题构造独立scope权威行。"""

    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS:
        raise RuntimeError("blocker-rereview candidates must contain all 15 blockers")
    base = _read_json(PROPOSAL_ROOT / "proposals.json").get("proposals") or []
    supplemental = _read_json(ROOT / "reviewer-block-supplemental-proposals.json").get(
        "proposals"
    ) or []
    proposal_rows = [*base, *supplemental]
    proposals = {str(row.get("case_id") or ""): row for row in proposal_rows}
    if len(proposals) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS:
        raise RuntimeError("blocker-rereview trusted proposal lineage is incomplete")
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        parent_case_id = str(metadata.get("parent_case_id") or "")
        proposal = proposals.get(parent_case_id)
        expected_proposal_sha = str((proposal or {}).get("audit", {}).get("proposed_case_sha256") or "")
        prefix, separator, ordinal = parent_case_id.rpartition("-")
        expected_case_id = f"{prefix}-V2-{ordinal}" if separator else ""
        if (
            proposal is None
            or metadata.get("difficulty") != proposal.get("difficulty")
            or metadata.get("blocker_remediation_proposal_sha256") != expected_proposal_sha
            or candidate.get("case_id") != expected_case_id
        ):
            raise RuntimeError(f"blocker-rereview candidate authority drift: {parent_case_id}")
    return build_blocker_scope_authority(
        candidates,
        request_map=request_map,
        proposals=proposal_rows,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_upstream_freshness(prepared_summary: dict[str, Any]) -> None:
    block_union_audit = _read_json(ROOT / "review-block-union-audit.json")
    base_proposals = _read_json(PROPOSAL_ROOT / "proposals.json")
    supplemental = _read_json(ROOT / "reviewer-block-supplemental-proposals.json")
    proposal_validation = _read_json(PROPOSAL_ROOT / "validation.json")
    if block_union_audit.get("status") != "passed" or proposal_validation.get("status") != "passed":
        raise RuntimeError("blocker-rereview upstream audit/validation has not passed")
    base_rows = base_proposals.get("proposals") or []
    supplemental_rows = supplemental.get("proposals") or []
    proposal_parent_ids = {
        str(row.get("case_id") or "") for row in [*base_rows, *supplemental_rows]
    }
    required_parent_ids = {
        str(value) for value in block_union_audit.get("required_data_remediation_case_ids") or []
    }
    if (
        len(base_rows) != EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
        or len(supplemental_rows) != EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        or len(proposal_parent_ids) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
        or proposal_parent_ids != required_parent_ids
    ):
        raise RuntimeError("blocker-rereview current upstream is not the exact 13+2 union")
    validate_block_union_freshness(
        block_union_audit,
        requests_path=ROOT / "review" / "requests.jsonl",
        responses_path=ROOT / "review" / "responses.jsonl",
        visual_audit_path=ROOT / "source-visual-audit.json",
        base_proposals_path=PROPOSAL_ROOT / "proposals.json",
        supplemental_proposals_path=ROOT / "reviewer-block-supplemental-proposals.json",
        label="blocker-rereview paid entry",
    )
    validate_proposal_validation_binding(
        proposal_validation,
        proposals_path=PROPOSAL_ROOT / "proposals.json",
        label="blocker-rereview paid entry proposal validation",
    )
    expected = {
        "review_block_union_audit_sha256": sha256_json(block_union_audit),
        "proposal_document_sha256": sha256_json(base_proposals),
        "supplemental_proposal_document_sha256": sha256_json(supplemental),
        "proposal_validation_sha256": sha256_json(proposal_validation),
    }
    drifted = sorted(key for key, value in expected.items() if prepared_summary.get(key) != value)
    if drifted:
        raise RuntimeError(f"blocker-rereview prepared upstream receipt drift: {drifted}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_requests() -> tuple[list[dict[str, Any]], int]:
    prepared_summary = json.loads((REREVIEW_ROOT / "summary.json").read_text(encoding="utf-8"))
    _validate_upstream_freshness(prepared_summary)
    prepared_receipt = validate_blocker_rereview_prepared_receipt(prepared_summary)
    expected_cases = int(prepared_summary.get("case_count") or 0)
    if expected_cases != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS:
        raise RuntimeError("changed-case prepared summary must contain all 15 data blockers")
    requests = runner._read_jsonl(REQUESTS_PATH)
    validate_request_pack_identity(
        requests,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_count=expected_cases,
        expected_pack_sha256=str(prepared_receipt["request_pack_sha256"]),
        pack_name="changed blocker-rereview paid pack",
    )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        sse_completion_policy=runner.SSE_COMPLETION_POLICY,
    )
    if prepared_summary.get("execution_contract_sha256") != execution_contract:
        raise RuntimeError("blocker-rereview prepared execution-contract SHA drift")
    if len(requests) != expected_cases:
        raise RuntimeError(f"expected {expected_cases} changed-case requests")
    if len({str(row.get("custom_id") or "") for row in requests}) != expected_cases:
        raise RuntimeError("changed-case request ids must be unique")
    original_ids = {
        str(row.get("custom_id") or "") for row in runner._read_jsonl(runner.REQUESTS_PATH)
    }
    if original_ids & {str(row["custom_id"]) for row in requests}:
        raise RuntimeError("changed-case request ids collide with superseded review ids")
    if {str(row.get("prompt_sha256") or "") for row in requests} != {EXPECTED_PROMPT_SHA256}:
        raise RuntimeError("changed-case requests do not use the frozen full-suite prompt")
    for row in requests:
        messages = (row.get("request") or {}).get("messages") or []
        if len(messages) != 2:
            raise RuntimeError("changed-case request message structure is invalid")
        payload = json.loads(str(messages[1].get("content") or ""))
        leaks = runner._walk_forbidden(payload)
        if leaks:
            raise RuntimeError(f"changed-case semantic payload leak: {leaks[:3]}")
    return requests, expected_cases


def _original_review_complete() -> bool:
    requests = runner._read_jsonl(runner.REQUESTS_PATH)
    request_ids = {str(row.get("custom_id") or "") for row in requests}
    if (
        len(requests) != runner.EXPECTED_CASE_CALLS
        or len(request_ids) != runner.EXPECTED_CASE_CALLS
    ):
        return False
    try:
        validate_request_pack_identity(
            requests,
            expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
            expected_count=runner.EXPECTED_CASE_CALLS,
            pack_name="original full-suite paid prerequisite",
        )
    except RuntimeError:
        return False
    successful: dict[str, dict[str, Any]] = {}
    for row in runner._read_jsonl(runner.RESPONSES_PATH):
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        if custom_id not in request_ids or custom_id in successful:
            return False
        if row.get("prompt_sha256") != EXPECTED_PROMPT_SHA256 or not isinstance(
            row.get("review"), dict
        ):
            return False
        successful[custom_id] = row
    return set(successful) == request_ids


def _formal_success_ledger(
    rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    execution_contract_sha256: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    requests_by_id = {str(row["custom_id"]): row for row in requests}

    def validate_saved(
        result: dict[str, Any],
        request: dict[str, Any],
        source_rows: list[dict[str, Any]],
    ) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        return runner._validate_saved_post150_recovery(
            result,
            request=request,
            semantic_payload=payload,
            source_response_rows=source_rows,
            execution_contract_sha256=execution_contract_sha256,
            materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        )

    def validate_live(result: dict[str, Any], request: dict[str, Any]) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        review = result.get("review")
        if not isinstance(review, dict):
            return ["live formal review must be an object"]
        errors = runner._validate_review(review)
        errors.extend(runner._validate_against_payload(review, payload))
        errors.extend(runner._validate_inverse_edge_canonicalization_audit(result, payload))
        errors.extend(runner._validate_exact_variant_deduplication_audit(result, payload))
        errors.extend(runner._validate_coverage_status_canonicalization_audit(result, payload))
        for item in result.get("invalid_attempts_before_success") or []:
            errors.extend(
                runner._validate_invalid_attempt_raw_normalization_provenance(item, result, payload)
            )
        return errors

    def validate_failed(result: dict[str, Any], request: dict[str, Any]) -> list[str]:
        return formal_rereview_failed_result_errors(
            result,
            request,
            expected_sequence=requests.index(request) + 1,
            execution_contract_sha256=execution_contract_sha256,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    return formal_rereview_success_ledger(
        rows,
        requests_by_id=requests_by_id,
        execution_contract_sha256=execution_contract_sha256,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_model=EXPECTED_REREVIEW_MODEL,
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        additional_saved_profiles=(
            (
                runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
                runner.SAVED_RESPONSE_TRANSPORT,
            ),
        ),
        validate_live_result=validate_live,
        validate_saved_result=validate_saved,
        label=label,
        materializer_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        validate_materialization_result=lambda result, request: (
            runner._review_materialization_errors(
                result["review"],
                json.loads(request["request"]["messages"][1]["content"]),
                materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
            )
        ),
        validate_failed_result=validate_failed,
    )


def _formal_state(
    rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    execution_contract_sha256: str,
    label: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    requests_by_id = {str(row["custom_id"]): row for row in requests}

    def validate_saved(
        result: dict[str, Any],
        request: dict[str, Any],
        source_rows: list[dict[str, Any]],
    ) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        return runner._validate_saved_post150_recovery(
            result,
            request=request,
            semantic_payload=payload,
            source_response_rows=source_rows,
            execution_contract_sha256=execution_contract_sha256,
            materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        )

    def validate_live(result: dict[str, Any], request: dict[str, Any]) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        review = result.get("review")
        if not isinstance(review, dict):
            return ["live formal review must be an object"]
        errors = runner._validate_review(review)
        errors.extend(runner._validate_against_payload(review, payload))
        errors.extend(runner._validate_inverse_edge_canonicalization_audit(result, payload))
        errors.extend(runner._validate_exact_variant_deduplication_audit(result, payload))
        errors.extend(runner._validate_coverage_status_canonicalization_audit(result, payload))
        for item in result.get("invalid_attempts_before_success") or []:
            errors.extend(
                runner._validate_invalid_attempt_raw_normalization_provenance(item, result, payload)
            )
        return errors

    def validate_materialization(result: dict[str, Any], request: dict[str, Any]) -> list[str]:
        payload = json.loads(request["request"]["messages"][1]["content"])
        review = result.get("review")
        if not isinstance(review, dict):
            return ["live formal materialization review must be an object"]
        return runner._review_materialization_errors(
            review,
            payload,
            materialization_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        )

    def validate_failed(result: dict[str, Any], request: dict[str, Any]) -> list[str]:
        return formal_rereview_failed_result_errors(
            result,
            request,
            expected_sequence=requests.index(request) + 1,
            execution_contract_sha256=execution_contract_sha256,
            live_runner_version=runner.RUNNER_VERSION,
            live_transport=runner.TRANSPORT,
            validate_invalid_attempt=(
                runner._validate_invalid_attempt_raw_normalization_provenance
            ),
        )

    return formal_rereview_state_reducer(
        rows,
        requests_by_id=requests_by_id,
        execution_contract_sha256=execution_contract_sha256,
        expected_prompt_sha256=EXPECTED_PROMPT_SHA256,
        expected_model=EXPECTED_REREVIEW_MODEL,
        expected_endpoint=runner._endpoint(EXPECTED_REREVIEW_BASE_URL),
        live_runner_version=runner.RUNNER_VERSION,
        live_transport=runner.TRANSPORT,
        saved_runner_version=runner.POST150_SAVED_RECOVERY_RUNNER_VERSION,
        saved_transport=runner.SAVED_RESPONSE_TRANSPORT,
        additional_saved_profiles=(
            (
                runner.POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
                runner.SAVED_RESPONSE_TRANSPORT,
            ),
        ),
        materializer_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
        validate_live_result=validate_live,
        validate_saved_result=validate_saved,
        validate_materialization_result=validate_materialization,
        validate_failed_result=validate_failed,
        label=label,
    )


def _authoritative_summary_snapshot(
    requests: list[dict[str, Any]],
    *,
    expected_cases: int,
    execution_contract_sha256: str,
    label: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    """写出执行摘要前，先重放持久化账本并取得权威进度。"""
    rows = runner._read_jsonl(RESPONSES_PATH)
    state = _formal_state(
        rows,
        requests,
        execution_contract_sha256=execution_contract_sha256,
        label=label,
    )
    completed = len(state["successful_by_id"])
    progress = {
        "already_completed_case_calls": completed,
        "remaining_case_calls": expected_cases - completed,
        "usage": runner._usage_totals(rows),
        "api_attempts_recorded": runner._attempt_count(rows),
        "usage_is_lower_bound": any(row.get("usage_is_lower_bound") is True for row in rows),
    }
    return rows, state, progress


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(FULL_SCOPE_NAME, SCOPE_NAME),
        default=FULL_SCOPE_NAME,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--confirm-full-review", action="store_true")
    parser.add_argument("--confirm-no-concurrent-review", action="store_true")
    parser.add_argument("--api-key-env", default=runner.DEFAULT_KEY_ENV)
    parser.add_argument("--base-url", default=EXPECTED_REREVIEW_BASE_URL)
    parser.add_argument("--model", default=EXPECTED_REREVIEW_MODEL)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    requests, expected_cases = _validate_requests()
    selection: ScopeSelection | None = None
    if args.scope == SCOPE_NAME:
        request_map = json.loads(REQUEST_MAP_PATH.read_text(encoding="utf-8"))
        if not isinstance(request_map, list):
            raise RuntimeError("blocker-rereview request-map 必须是数组")
        selection = select_scope_requests(
            requests,
            request_map,
            args.scope,
            authority_rows=_scope_authority(request_map),
        )
        if (
            selection.target_case_count != 4
            or selection.target_difficulty_counts != {"easy": 4, "medium": 0}
        ):
            raise RuntimeError("blocker-rereview easy/medium 范围必须精确包含 E4/M0")
    target_custom_ids = (
        set(selection.target_custom_ids)
        if selection is not None
        else {str(row["custom_id"]) for row in requests}
    )
    summary_path = (
        SUMMARY_PATH
        if selection is None
        else scope_summary_path(REREVIEW_ROOT, args.scope)
    )

    def scope_snapshot(
        state: dict[str, dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        if selection is None:
            return {}
        full_rows = runner._read_jsonl(RESPONSES_PATH)
        scope_rows = [
            row
            for row in full_rows
            if str(row.get("custom_id") or "") in target_custom_ids
        ]
        return {
            **selection.receipt(),
            **scope_progress(state, selection),
            "scope_physical_response_row_count": len(scope_rows),
            "scope_usage": runner._usage_totals(scope_rows),
            "scope_api_attempts_recorded": runner._attempt_count(scope_rows),
            "scope_usage_is_lower_bound": any(
                row.get("usage_is_lower_bound") is True for row in scope_rows
            ),
            "full_physical_response_row_count": len(full_rows),
            "full_usage": runner._usage_totals(full_rows),
            "full_api_attempts_recorded": runner._attempt_count(full_rows),
            "full_usage_is_lower_bound": any(
                row.get("usage_is_lower_bound") is True for row in full_rows
            ),
        }

    def completion_errors(
        state: dict[str, dict[str, dict[str, Any]]],
    ) -> list[str]:
        if selection is not None:
            return scope_completion_errors(state, selection)
        return formal_rereview_completion_errors(
            state,
            expected_custom_ids=[str(row["custom_id"]) for row in requests],
        )
    execution_contract = validate_rereview_request_policy(
        requests,
        runner_version=runner.RUNNER_VERSION,
        transport=runner.TRANSPORT,
        sse_completion_policy=runner.SSE_COMPLETION_POLICY,
    )
    prepared_summary = _read_json(REREVIEW_ROOT / "summary.json")
    validate_rereview_runtime(
        model=args.model,
        base_url=args.base_url,
        execution_contract_sha256=execution_contract,
        expected_execution_contract_sha256=str(
            prepared_summary.get("execution_contract_sha256") or ""
        ),
    )
    original_review_complete = _original_review_complete()
    if args.execute:
        missing = [
            name
            for enabled, name in (
                (args.acknowledge_cost, "--acknowledge-cost"),
                (args.confirm_full_review, "--confirm-full-review"),
                (args.confirm_no_concurrent_review, "--confirm-no-concurrent-review"),
            )
            if not enabled
        ]
        if missing:
            raise SystemExit("paid execution refused; missing gate(s): " + ", ".join(missing))
        if not original_review_complete:
            raise SystemExit(
                "changed-case review refused until original 150/150 review is complete"
            )
    response_rows = runner._read_jsonl(RESPONSES_PATH)
    completed = _formal_success_ledger(
        response_rows,
        requests,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview resume",
    )
    initial_episode_state = _formal_state(
        response_rows,
        requests,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview dry materialization state",
    )
    dry_run = {
        "status": "dry_run_only",
        "runner_version": runner.RUNNER_VERSION,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "prepared_case_calls": expected_cases,
        "already_completed_case_calls": len(completed),
        "remaining_case_calls": expected_cases - len(completed),
        "original_150_review_complete": original_review_complete,
        "superseded_review_reused": False,
        "execution_contract_sha256": execution_contract,
        "materialization_adoption": materialization_state_summary(initial_episode_state),
        **scope_snapshot(initial_episode_state),
    }
    if not args.execute:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0
    runner._acquire_execution_lock(
        recover_stale=False,
        purpose="suite-v2 15 changed blocker-case independent rereview",
    )
    api_key: str | None = None

    def require_api_key() -> str:
        nonlocal api_key
        if api_key is None:
            api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                f"paid execution refused; environment variable {args.api_key_env} is empty"
            )
        return api_key

    # Saved-only recovery is deterministic and zero-cost.  Replay and append
    # every eligible latest failed event while holding the execution lock,
    # before calibration can read a key or call the provider.
    response_rows = runner._read_jsonl(RESPONSES_PATH)
    completed = _formal_success_ledger(
        response_rows,
        requests,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview pre-calibration saved resume",
    )
    for row in requests:
        custom_id = str(row["custom_id"])
        if custom_id not in target_custom_ids or custom_id in completed:
            continue
        saved_result = runner._recover_latest_saved_same_concept_variant_subset(
            row=row,
            response_rows=response_rows,
            execution_contract_sha256=execution_contract,
        )
        if saved_result is None:
            saved_result = runner._recover_latest_saved_quote_key_branch_span(
                row=row,
                response_rows=response_rows,
                execution_contract_sha256=execution_contract,
            )
        if saved_result is None:
            continue
        payload = json.loads(row["request"]["messages"][1]["content"])
        replay_errors = runner._validate_saved_post150_recovery(
            saved_result,
            request=row,
            semantic_payload=payload,
            source_response_rows=response_rows,
            execution_contract_sha256=execution_contract,
        )
        if replay_errors:
            raise RuntimeError(
                "saved blocker-rereview recovery replay failed: " + "; ".join(replay_errors)
            )
        runner._append_jsonl(RESPONSES_PATH, saved_result)
        response_rows.append(saved_result)
        completed = _formal_success_ledger(
            response_rows,
            requests,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview post-saved resume",
        )

    if not completion_errors(
        _formal_state(
            response_rows,
            requests,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview scoped completion check",
        )
    ):
        _, completed_episode_state, progress = _authoritative_summary_snapshot(
            requests,
            expected_cases=expected_cases,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview completed summary state",
        )
        final_completion_errors = completion_errors(completed_episode_state)
        if final_completion_errors:
            summary = {
                **dry_run,
                "status": "incomplete_final_formal_state",
                **progress,
                **scope_snapshot(completed_episode_state),
                "formal_completion_errors": final_completion_errors,
                "materialization_adoption": materialization_state_summary(completed_episode_state),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            _write_json(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        final_scope_progress = scope_snapshot(completed_episode_state)
        summary = {
            "status": "succeeded",
            "runner_version": runner.RUNNER_VERSION,
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "case_success_count": (
                final_scope_progress.get("scope_case_success_count")
                if selection is not None
                else progress["already_completed_case_calls"]
            ),
            "case_expected_count": (
                selection.target_case_count if selection is not None else expected_cases
            ),
            "original_150_review_complete": True,
            "superseded_review_reused": False,
            "execution_contract_sha256": execution_contract,
            **progress,
            **final_scope_progress,
            "materialization_adoption": materialization_state_summary(completed_episode_state),
            "price_amount": None,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    episode_state = _formal_state(
        response_rows,
        requests,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview pre-calibration materialization state",
    )
    blocking_retry_started_ids = set(episode_state["retry_started_by_id"])
    blocking_terminal_ids = set(episode_state["terminal_by_id"])
    if selection is not None:
        blocking_retry_started_ids &= target_custom_ids
        blocking_terminal_ids &= target_custom_ids
    if blocking_retry_started_ids or blocking_terminal_ids:
        _, episode_state, progress = _authoritative_summary_snapshot(
            requests,
            expected_cases=expected_cases,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview pre-calibration fail-stop summary state",
        )
        summary = {
            **dry_run,
            "status": "incomplete_materialization_retry_fail_stop",
            **progress,
            **scope_snapshot(episode_state),
            "materialization_retry_started_custom_ids": sorted(
                blocking_retry_started_ids
            ),
            "materialization_terminal_custom_ids": sorted(blocking_terminal_ids),
            "materialization_adoption": materialization_state_summary(episode_state),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    endpoint = runner._endpoint(args.base_url)

    calibration_requests = runner._calibration_requests(requests[0])
    for row in calibration_requests:
        row["custom_id"] = rereview_calibration_custom_id(str(row["custom_id"]), execution_contract)
    calibration_ids = {str(row["custom_id"]) for row in calibration_requests}
    if len(calibration_requests) != 4 or len(calibration_ids) != 4:
        raise SystemExit("blocker-rereview calibration pack must contain exactly 4 ids")

    def load_calibration_ledger() -> dict[str, dict[str, Any]]:
        return rereview_calibration_ledger(
            runner._read_jsonl(runner.CALIBRATION_RESPONSES_PATH),
            requests=calibration_requests,
            execution_contract_sha256=execution_contract,
            validate_core_result=runner._validate_calibration_result,
        )

    calibration_done = load_calibration_ledger()
    if episode_state["invalidated_by_id"] and set(calibration_done) != calibration_ids:
        _, episode_state, progress = _authoritative_summary_snapshot(
            requests,
            expected_cases=expected_cases,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview invalidation summary state",
        )
        summary = {
            **dry_run,
            "status": "incomplete_materialization_invalidation_without_calibration",
            **progress,
            **scope_snapshot(episode_state),
            "materialization_invalidated_custom_ids": sorted(episode_state["invalidated_by_id"]),
            "materialization_adoption": materialization_state_summary(episode_state),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    for row in calibration_requests:
        if row["custom_id"] in calibration_done:
            continue
        result = runner._execute_one(
            row=row,
            endpoint=endpoint,
            api_key=require_api_key(),
            model=args.model,
            timeout=args.request_timeout,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            calibration=True,
        )
        bound_result = bind_rereview_calibration_receipt(
            result,
            execution_contract_sha256=execution_contract,
        )
        runner._append_jsonl(runner.CALIBRATION_RESPONSES_PATH, bound_result)
        calibration_done = load_calibration_ledger()
        if row["custom_id"] not in calibration_done:
            raise SystemExit(f"calibration failed: {row['custom_id']}")
    calibration_done = load_calibration_ledger()
    if set(calibration_done) != calibration_ids:
        raise SystemExit("blocker-rereview calibration set is not exactly 4/4 current ids")

    response_rows = runner._read_jsonl(RESPONSES_PATH)

    def load_episode_state(
        current_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        return _formal_state(
            current_rows,
            requests,
            execution_contract_sha256=execution_contract,
            label="blocker-rereview materialization retry replay",
        )

    def validate_episode_materialization(
        result: dict[str, Any], request: dict[str, Any]
    ) -> list[str]:
        review = result.get("review")
        if not isinstance(review, dict):
            return ["live formal materialization review must be an object"]
        payload = json.loads(request["request"]["messages"][1]["content"])
        return runner._review_materialization_errors(review, payload)

    def run_materialization_retry(
        target_custom_ids: list[str] | None = None,
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], int]:
        nonlocal response_rows
        retry_api_key = require_api_key()

        def execute_episode(row: dict[str, Any]) -> dict[str, Any]:
            return runner._execute_one(
                row=row,
                endpoint=endpoint,
                api_key=retry_api_key,
                model=args.model,
                timeout=args.request_timeout,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                calibration=False,
            )

        response_rows, state, code = execute_materialization_retry_episodes(
            rows=response_rows,
            requests=requests,
            state_loader=load_episode_state,
            append_event=lambda event: runner._append_jsonl(RESPONSES_PATH, event),
            execute_one=execute_episode,
            validate_materialization_result=validate_episode_materialization,
            execution_contract_sha256=execution_contract,
            materializer_policy_version=LEGACY_MATERIALIZATION_POLICY_VERSION,
            endpoint=endpoint,
            request_timeout=args.request_timeout,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            target_custom_ids=target_custom_ids,
        )
        return state, code

    def stop_on_materialization_state(
        _state: dict[str, dict[str, dict[str, Any]]], *, status: str
    ) -> int:
        _, state, progress = _authoritative_summary_snapshot(
            requests,
            expected_cases=expected_cases,
            execution_contract_sha256=execution_contract,
            label=f"blocker-rereview {status} summary state",
        )
        summary = {
            **dry_run,
            "status": status,
            **progress,
            **scope_snapshot(state),
            "materialization_retry_eligible_custom_ids": sorted(state["retry_eligible_by_id"]),
            "materialization_invalidated_custom_ids": sorted(state["invalidated_by_id"]),
            "materialization_retry_started_custom_ids": sorted(state["retry_started_by_id"]),
            "materialization_terminal_custom_ids": sorted(state["terminal_by_id"]),
            "materialization_adoption": materialization_state_summary(state),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    episode_state = load_episode_state(response_rows)
    scoped_retry_ids = target_custom_ids & (
        set(episode_state["retry_eligible_by_id"])
        | set(episode_state["invalidated_by_id"])
    )
    if scoped_retry_ids:
        episode_state, episode_code = run_materialization_retry(
            sorted(target_custom_ids)
        )
        if episode_code != 0:
            return stop_on_materialization_state(
                episode_state,
                status="incomplete_materialization_retry_terminal",
            )

    completed = _formal_success_ledger(
        response_rows,
        requests,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview resume",
    )
    for sequence, row in enumerate(requests, start=1):
        custom_id = str(row["custom_id"])
        if custom_id not in target_custom_ids or custom_id in completed:
            continue
        result = runner._recover_latest_saved_same_concept_variant_subset(
            row=row,
            response_rows=response_rows,
            execution_contract_sha256=execution_contract,
        )
        if result is None:
            result = runner._recover_latest_saved_quote_key_branch_span(
                row=row,
                response_rows=response_rows,
                execution_contract_sha256=execution_contract,
            )
        if result is not None:
            payload = json.loads(row["request"]["messages"][1]["content"])
            replay_errors = runner._validate_saved_post150_recovery(
                result,
                request=row,
                semantic_payload=payload,
                source_response_rows=response_rows,
                execution_contract_sha256=execution_contract,
            )
            if replay_errors:
                raise RuntimeError(
                    "saved blocker-rereview recovery replay failed: " + "; ".join(replay_errors)
                )
        else:
            result = runner._execute_one(
                row=row,
                endpoint=endpoint,
                api_key=require_api_key(),
                model=args.model,
                timeout=args.request_timeout,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                calibration=False,
            )
            result["execution_contract_sha256"] = execution_contract
            result["sequence"] = sequence
        runner._append_jsonl(RESPONSES_PATH, result)
        response_rows.append(result)
        if result["status"] != "succeeded":
            _, failed_state, progress = _authoritative_summary_snapshot(
                requests,
                expected_cases=expected_cases,
                execution_contract_sha256=execution_contract,
                label="blocker-rereview stopped-on-failure summary state",
            )
            summary = {
                **dry_run,
                "status": "incomplete_stopped_on_failure",
                **progress,
                **scope_snapshot(failed_state),
                "failed_custom_ids": [custom_id],
                "materialization_adoption": materialization_state_summary(failed_state),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            _write_json(summary_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        post_append_state = load_episode_state(response_rows)
        if custom_id in post_append_state["retry_eligible_by_id"]:
            post_append_state, episode_code = run_materialization_retry([custom_id])
            if episode_code != 0:
                return stop_on_materialization_state(
                    post_append_state,
                    status="incomplete_materialization_retry_terminal",
                )
        if custom_id not in post_append_state["successful_by_id"]:
            return stop_on_materialization_state(
                post_append_state,
                status="incomplete_materialization_authority_fail_stop",
            )
        completed = dict(post_append_state["successful_by_id"])

    _, final_episode_state, progress = _authoritative_summary_snapshot(
        requests,
        expected_cases=expected_cases,
        execution_contract_sha256=execution_contract,
        label="blocker-rereview final summary state",
    )
    final_errors = completion_errors(final_episode_state)
    if final_errors:
        return stop_on_materialization_state(
            final_episode_state,
            status="incomplete_final_formal_state",
        )
    final_scope_progress = scope_snapshot(final_episode_state)
    summary = {
        "status": "succeeded",
        "runner_version": runner.RUNNER_VERSION,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "case_success_count": (
            final_scope_progress.get("scope_case_success_count")
            if selection is not None
            else progress["already_completed_case_calls"]
        ),
        "case_expected_count": (
            selection.target_case_count if selection is not None else expected_cases
        ),
        "original_150_review_complete": True,
        "superseded_review_reused": False,
        "execution_contract_sha256": execution_contract,
        **progress,
        **final_scope_progress,
        "materialization_adoption": materialization_state_summary(final_episode_state),
        "price_amount": None,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
