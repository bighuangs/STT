"""Deterministic release-integrity checks for suite-v2 review artifacts.

The LLM runner is intentionally outside this module.  These helpers validate
the immutable request and audit envelopes that surround model execution so the
release workflow never trusts a declared hash without recomputing it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal


RequestIdMode = Literal["payload", "contract_rereview"]
EXPECTED_REVIEW_PROMPT_SHA256 = (
    "d5af42cbcf5385c408583c4ed36b110ac89740be24d83248fafb58681d04a78b"
)
EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS = 13
EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS = 2
EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS = 15
EXPECTED_REREVIEW_MODEL = "qwen3.7-max"
EXPECTED_REREVIEW_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EXPECTED_REREVIEW_TEMPERATURE = 0
EXPECTED_REREVIEW_RESPONSE_FORMAT = {"type": "json_object"}
MATERIALIZATION_ADOPTION_POLICY = "post150-formal-materialization-adoption-v1"
MATERIALIZATION_RETRY_QUOTA = 1
FORMAL_STREAM_DIAGNOSTIC_POLICY = (
    "post150-formal-stream-content-diagnostics-non-authoritative-v1"
)
MATERIALIZATION_RETRY_REQUEST_CONFIG = {
    "request_timeout_seconds": 600.0,
    "max_attempts": 3,
    "retry_delay_seconds": 2.0,
    "temperature": EXPECTED_REREVIEW_TEMPERATURE,
    "response_format": EXPECTED_REREVIEW_RESPONSE_FORMAT,
    "max_tokens": "forbidden",
}
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_rereview_custom_id(original_custom_id: str, payload_sha256: str) -> str:
    identity = f"contract-rereview-v1:{original_custom_id}:{payload_sha256}"
    return f"concept-contract-rereview-{sha256_bytes(identity.encode())[:20]}"


def rereview_execution_contract_sha256(
    *,
    runner_version: str,
    transport: str,
    sse_completion_policy: str,
) -> str:
    """Hash the frozen paid-rereview runtime and request policy."""

    return sha256_json(
        {
            "runner_version": runner_version,
            "model": EXPECTED_REREVIEW_MODEL,
            "base_url": EXPECTED_REREVIEW_BASE_URL,
            "temperature": EXPECTED_REREVIEW_TEMPERATURE,
            "response_format": EXPECTED_REREVIEW_RESPONSE_FORMAT,
            "max_tokens": "forbidden",
            "transport": transport,
            "sse_completion_policy": sse_completion_policy,
            "prompt_sha256": EXPECTED_REVIEW_PROMPT_SHA256,
        }
    )


def validate_rereview_request_policy(
    rows: Sequence[dict[str, Any]],
    *,
    runner_version: str,
    transport: str,
    sse_completion_policy: str,
) -> str:
    """Require every post-150 request to use the frozen JSON-object policy."""

    for row in rows:
        request = row.get("request")
        if not isinstance(request, Mapping) or (
            request.get("model") != EXPECTED_REREVIEW_MODEL
            or request.get("temperature") != EXPECTED_REREVIEW_TEMPERATURE
            or isinstance(request.get("temperature"), bool)
            or request.get("response_format") != EXPECTED_REREVIEW_RESPONSE_FORMAT
            or "max_tokens" in request
        ):
            raise RuntimeError("frozen rereview request policy drift")
    return rereview_execution_contract_sha256(
        runner_version=runner_version,
        transport=transport,
        sse_completion_policy=sse_completion_policy,
    )


def validate_rereview_runtime(
    *,
    model: str,
    base_url: str,
    execution_contract_sha256: str,
    expected_execution_contract_sha256: str,
) -> None:
    """Refuse runtime overrides and a stale prepared execution contract."""

    if (
        model != EXPECTED_REREVIEW_MODEL
        or base_url.rstrip("/") != EXPECTED_REREVIEW_BASE_URL
    ):
        raise RuntimeError("rereview runtime model/base URL drift")
    if execution_contract_sha256 != expected_execution_contract_sha256:
        raise RuntimeError("prepared rereview execution-contract SHA drift")


def rereview_calibration_custom_id(custom_id: str, execution_contract_sha256: str) -> str:
    """Namespace calibration receipts by the complete execution contract."""

    return f"{custom_id}-{execution_contract_sha256[:16]}"


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")


def bind_rereview_calibration_receipt(
    result: Mapping[str, Any],
    *,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    """Bind one untouched core calibration envelope to the post-150 runtime.

    The outer field is deliberately not part of the runner's exact calibration
    schema.  Consumers must remove it before replaying the core validator.
    """

    _require_sha256(
        execution_contract_sha256,
        label="rereview calibration execution contract",
    )
    if "execution_contract_sha256" in result:
        raise RuntimeError("core calibration result already contains the outer binding field")
    return {**dict(result), "execution_contract_sha256": execution_contract_sha256}


def rereview_calibration_ledger(
    rows: Sequence[dict[str, Any]],
    *,
    requests: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
    validate_core_result: Callable[[dict[str, Any], dict[str, Any]], Sequence[str]],
) -> dict[str, dict[str, Any]]:
    """Replay the latest bound calibration event for every current request.

    The append-only ledger uses latest-event semantics.  A later malformed or
    drifted event therefore makes that id incomplete and forces a fresh call;
    an older valid event cannot silently mask it.  Foreign calibration ids are
    ignored because the file is shared by versioned execution contracts.
    """

    _require_sha256(
        execution_contract_sha256,
        label="rereview calibration execution contract",
    )
    request_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        custom_id = str(request.get("custom_id") or "")
        if not custom_id or custom_id in request_by_id:
            raise RuntimeError("rereview calibration request ids must be non-empty and unique")
        request_by_id[custom_id] = request

    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = str(row.get("custom_id") or "")
        if custom_id in request_by_id:
            latest_by_id[custom_id] = row

    completed: dict[str, dict[str, Any]] = {}
    for custom_id, bound_result in latest_by_id.items():
        core_result = dict(bound_result)
        if core_result.pop("execution_contract_sha256", None) != execution_contract_sha256:
            continue
        errors = list(validate_core_result(core_result, request_by_id[custom_id]))
        if errors:
            continue
        completed[custom_id] = dict(bound_result)
    return completed


def strict_success_ledger(
    rows: Sequence[dict[str, Any]],
    *,
    expected_custom_ids: set[str],
    execution_contract_sha256: str,
    expected_prompt_sha256: str,
    expected_model: str,
    expected_runner_version: str,
    expected_transport: str,
    label: str,
    allow_foreign_successes: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load successful rows without silently accepting duplicates or drift."""

    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "succeeded":
            continue
        custom_id = str(row.get("custom_id") or "")
        if custom_id not in expected_custom_ids:
            if allow_foreign_successes:
                continue
            raise RuntimeError(f"{label}: successful response is outside current pack")
        if custom_id in successful:
            raise RuntimeError(f"{label}: duplicate successful response: {custom_id}")
        if (
            row.get("execution_contract_sha256") != execution_contract_sha256
            or row.get("prompt_sha256") != expected_prompt_sha256
            or row.get("model") != expected_model
            or row.get("runner_version") != expected_runner_version
            or row.get("transport") != expected_transport
            or not isinstance(row.get("review"), Mapping)
        ):
            raise RuntimeError(f"{label}: successful response execution contract drift")
        successful[custom_id] = dict(row)
    return successful


_FORMAL_LIVE_BASE_KEYS = frozenset(
    {
        "custom_id",
        "status",
        "attempt",
        "model",
        "runner_version",
        "transport",
        "sse_completion_policy",
        "prompt_sha256",
        "review",
        "response_sha256",
        "usage_is_lower_bound",
        "usage",
        "successful_response_usage",
        "attempt_usage",
        "sse_integrity",
        "sse_integrity_attempts",
        "completed_at",
        "retry_accounting",
        "execution_contract_sha256",
        "sequence",
    }
)
_FORMAL_LIVE_CANONICALIZATION_KEYS = frozenset(
    {
        "same_response_inverse_edge_canonicalization",
        "same_response_exact_variant_deduplication",
        "same_response_coverage_status_canonicalization",
    }
)
_FORMAL_LIVE_LINEAGE_KEY_SETS = (
    frozenset(),
    frozenset({"failures_before_success"}),
    frozenset({"failures_before_success", "invalid_attempts_before_success"}),
)
_FORMAL_LIVE_TOP_PROFILES = frozenset(
    _FORMAL_LIVE_BASE_KEYS | lineage | frozenset(canonical_subset)
    for lineage in _FORMAL_LIVE_LINEAGE_KEY_SETS
    for subset_size in range(len(_FORMAL_LIVE_CANONICALIZATION_KEYS) + 1)
    for canonical_subset in combinations(_FORMAL_LIVE_CANONICALIZATION_KEYS, subset_size)
)
_FORMAL_SSE_BASE_KEYS = frozenset(
    {
        "completion_policy",
        "termination",
        "done_seen",
        "finish_reason",
        "finish_reasons",
        "finish_reason_stop_seen",
        "event_count",
        "content_event_count",
        "content_chars",
        "content_sha256",
        "event_integrity",
        "usage_integrity",
        "review_validation",
        "attempt",
    }
)
_FORMAL_TOKEN_KEYS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens"}
)
_FORMAL_USAGE_INTEGRITY_PASSED = {
    "status": "passed",
    "usage_event_count": 1,
    "required_token_fields_valid": True,
    "total_matches_prompt_plus_completion": True,
    "at_or_after_finish_reason_stop": True,
}
_FORMAL_USAGE_INTEGRITY_KEYS = frozenset(_FORMAL_USAGE_INTEGRITY_PASSED)
_FORMAL_PROVIDER_ERROR_BASE_KEYS = frozenset({"policy", "shape"})


def _formal_usage_valid(
    value: Any,
    *,
    has_attempt: bool = False,
    allow_response_status: bool = False,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    base = _FORMAL_TOKEN_KEYS | ({"attempt"} if has_attempt else set())
    allowed_profiles = {base}
    if allow_response_status:
        allowed_profiles.add(base | {"response_status"})
    return bool(
        frozenset(value) in allowed_profiles
        and all(type(value.get(key)) is int and value[key] >= 0 for key in _FORMAL_TOKEN_KEYS)
        and value["total_tokens"]
        == value["prompt_tokens"] + value["completion_tokens"]
        and (
            "response_status" not in value
            or value.get("response_status") == "rejected_sse_integrity"
        )
    )


def _formal_terminal_sse_errors(
    integrity: Any,
    *,
    attempt: int,
    completion_policy: str,
    review_validation: str,
    content_object: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(integrity, Mapping) or frozenset(integrity) != _FORMAL_SSE_BASE_KEYS:
        return ["live formal terminal SSE schema is invalid"]
    minimum_chars = (
        len(json.dumps(content_object, ensure_ascii=False, separators=(",", ":")))
        if content_object is not None
        else 1
    )
    termination = integrity.get("termination")
    event_count = integrity.get("event_count")
    content_event_count = integrity.get("content_event_count")
    if (
        type(integrity.get("attempt")) is not int
        or integrity.get("attempt") != attempt
        or integrity.get("completion_policy") != completion_policy
        or termination not in {"done_sentinel", "validated_stop_eof"}
        or integrity.get("done_seen") is not (termination == "done_sentinel")
        or integrity.get("finish_reason") != "stop"
        or integrity.get("finish_reasons") != ["stop"]
        or integrity.get("finish_reason_stop_seen") is not True
        or integrity.get("event_integrity") != "passed"
        or integrity.get("review_validation") != review_validation
        or type(event_count) is not int
        or event_count < 2
        or type(content_event_count) is not int
        or not 1 <= content_event_count <= event_count - 1
        or type(integrity.get("content_chars")) is not int
        or integrity["content_chars"] < minimum_chars
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(integrity.get("content_sha256") or "")
        )
        or integrity.get("usage_integrity") != _FORMAL_USAGE_INTEGRITY_PASSED
    ):
        return ["live formal terminal SSE ledger is invalid"]
    return []


def _formal_provider_error_receipt_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    optional_profiles = {
        "code": {"code"},
        "code_hash": {"code_redacted", "code_sha256", "code_byte_length"},
        "type": {"type"},
        "type_hash": {"type_redacted", "type_sha256", "type_byte_length"},
        "status": {"status"},
        "status_hash": {"status_redacted", "status_sha256", "status_byte_length"},
        "message_hash": {
            "message_redacted",
            "message_sha256",
            "message_byte_length",
        },
        "unknown": {"unknown_field_count"},
    }
    possible = {
        _FORMAL_PROVIDER_ERROR_BASE_KEYS | frozenset().union(*selected)
        for size in range(len(optional_profiles) + 1)
        for selected in combinations(optional_profiles.values(), size)
    }
    if frozenset(value) not in possible:
        return False
    if (
        value.get("policy") != "allowlisted-provider-error-receipt-v2"
        or value.get("shape")
        not in {"object", "string", "array", "null", "boolean", "number", "unknown"}
    ):
        return False
    for field in ("code", "type", "status"):
        if field in value and f"{field}_redacted" in value:
            return False
        if field in value and (
            not isinstance(value[field], str) or not value[field]
        ):
            return False
        if f"{field}_redacted" in value and (
            value.get(f"{field}_redacted") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get(f"{field}_sha256") or ""))
            or type(value.get(f"{field}_byte_length")) is not int
            or value[f"{field}_byte_length"] < 0
        ):
            return False
    if "message_redacted" in value and (
        value.get("message_redacted") is not True
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("message_sha256") or ""))
        or type(value.get("message_byte_length")) is not int
        or value["message_byte_length"] < 0
    ):
        return False
    if "unknown_field_count" in value and (
        type(value.get("unknown_field_count")) is not int
        or value["unknown_field_count"] < 0
    ):
        return False
    if value.get("shape") == "object":
        if "unknown_field_count" not in value:
            return False
    elif (
        "message_redacted" not in value
        or any(
            field in value or f"{field}_redacted" in value
            for field in ("code", "type", "status")
        )
        or "unknown_field_count" in value
    ):
        return False
    return True


def _formal_failed_sse_errors(
    integrity: Any,
    *,
    attempt: int,
    completion_policy: str,
) -> list[str]:
    if not isinstance(integrity, Mapping):
        return ["live formal failed SSE receipt must be an object"]
    allowed_profiles = {
        _FORMAL_SSE_BASE_KEYS,
        _FORMAL_SSE_BASE_KEYS | {"response_sha256"},
        _FORMAL_SSE_BASE_KEYS | {"provider_error_receipt"},
        _FORMAL_SSE_BASE_KEYS | {"response_sha256", "provider_error_receipt"},
    }
    if frozenset(integrity) not in allowed_profiles:
        return ["live formal failed SSE exact schema is invalid"]
    termination = integrity.get("termination")
    finish_reason = integrity.get("finish_reason")
    finish_reasons = integrity.get("finish_reasons")
    stop_seen = integrity.get("finish_reason_stop_seen")
    event_count = integrity.get("event_count")
    content_event_count = integrity.get("content_event_count")
    usage_integrity = integrity.get("usage_integrity")
    if (
        type(integrity.get("attempt")) is not int
        or integrity.get("attempt") != attempt
        or integrity.get("completion_policy") != completion_policy
        or termination
        not in {
            "done_sentinel",
            "normal_eof",
            "protocol_error",
            "transport_exception",
        }
        or integrity.get("done_seen") is not (termination == "done_sentinel")
        or finish_reason not in {None, "stop"}
        or (finish_reason == "stop")
        is not (finish_reasons == ["stop"] and stop_seen is True)
        or (finish_reason is None)
        is not (finish_reasons == [] and stop_seen is False)
        or integrity.get("event_integrity") not in {"passed", "failed"}
        or integrity.get("review_validation") not in {"not_run", "failed"}
        or type(event_count) is not int
        or event_count < 2
        or type(content_event_count) is not int
        or not 1 <= content_event_count <= event_count - 1
        or type(integrity.get("content_chars")) is not int
        or integrity["content_chars"] < 1
        or not re.fullmatch(r"[0-9a-f]{64}", str(integrity.get("content_sha256") or ""))
        or not isinstance(usage_integrity, Mapping)
        or frozenset(usage_integrity) != _FORMAL_USAGE_INTEGRITY_KEYS
        or usage_integrity.get("status") not in {"passed", "missing", "failed"}
        or type(usage_integrity.get("usage_event_count")) is not int
        or usage_integrity["usage_event_count"] < 0
        or any(
            type(usage_integrity.get(field)) is not bool
            for field in (
                "required_token_fields_valid",
                "total_matches_prompt_plus_completion",
                "at_or_after_finish_reason_stop",
            )
        )
        or (
            usage_integrity.get("status") == "passed"
            and dict(usage_integrity) != _FORMAL_USAGE_INTEGRITY_PASSED
        )
        or (
            usage_integrity.get("status") != "passed"
            and usage_integrity.get("usage_event_count") == 1
            and usage_integrity.get("required_token_fields_valid") is True
            and usage_integrity.get("total_matches_prompt_plus_completion") is True
            and usage_integrity.get("at_or_after_finish_reason_stop") is True
        )
        or (
            "response_sha256" in integrity
            and not re.fullmatch(
                r"[0-9a-f]{64}", str(integrity.get("response_sha256") or "")
            )
        )
        or (
            "provider_error_receipt" in integrity
            and not _formal_provider_error_receipt_valid(
                integrity.get("provider_error_receipt")
            )
        )
    ):
        return ["live formal failed SSE receipt is invalid"]
    return []


def _formal_canonicalization_errors(row: Mapping[str, Any]) -> list[str]:
    expected_policies = {
        "same_response_inverse_edge_canonicalization": (
            "same-response-authoritative-inverse-edge-canonicalization-v1"
        ),
        "same_response_exact_variant_deduplication": (
            "same-response-normalized-exact-any-of-deduplication-v1"
        ),
        "same_response_coverage_status_canonicalization": (
            "same-response-zero-direct-concept-support-status-alias-to-uncovered-v4"
        ),
    }
    expected_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "actions",
        "source",
    }
    errors: list[str] = []
    for field, policy in expected_policies.items():
        if field not in row:
            continue
        audit = row[field]
        source = audit.get("source") if isinstance(audit, Mapping) else None
        if (
            not isinstance(audit, Mapping)
            or set(audit) != expected_keys
            or audit.get("policy") != policy
            or audit.get("api_call_performed") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("raw_review_sha256") or ""))
            or audit.get("projected_review_sha256") != row.get("response_sha256")
            or not isinstance(audit.get("raw_validation_errors"), list)
            or not isinstance(audit.get("actions"), list)
            or not audit["actions"]
            or not isinstance(source, Mapping)
            or set(source) != {"source_attempt", "raw_review_sha256"}
            or source.get("source_attempt") != row.get("attempt")
            or source.get("raw_review_sha256") != audit.get("raw_review_sha256")
        ):
            errors.append(f"live formal canonicalization audit is invalid: {field}")
    return errors


def _formal_live_receipt_errors(
    row: Mapping[str, Any], request: Mapping[str, Any], *, expected_sequence: int
) -> list[str]:
    """Replay the exact frozen live-2.32 success envelope and its lineage."""

    errors: list[str] = []
    if frozenset(row) not in _FORMAL_LIVE_TOP_PROFILES:
        errors.append("live formal receipt exact schema is invalid")
    request_body = request.get("request") or {}
    if (
        row.get("custom_id") != request.get("custom_id")
        or row.get("status") != "succeeded"
        or row.get("sse_completion_policy")
        != "strict-terminal-usage-complete-json-v2"
        or row.get("prompt_sha256") != request.get("prompt_sha256")
        or row.get("model") != request_body.get("model")
        or not isinstance(row.get("review"), Mapping)
        or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("response_sha256") or ""))
        or row.get("response_sha256") != sha256_json(row.get("review"))
        or type(row.get("attempt")) is not int
        or int(row["attempt"]) < 1
        or type(row.get("sequence")) is not int
        or row.get("sequence") != expected_sequence
        or type(row.get("usage_is_lower_bound")) is not bool
    ):
        errors.append("live formal identity/response binding is invalid")

    try:
        completed_at = datetime.fromisoformat(str(row.get("completed_at") or ""))
    except ValueError:
        completed_at = None
    if (
        completed_at is None
        or completed_at.tzinfo is None
        or completed_at.utcoffset() != UTC.utcoffset(None)
    ):
        errors.append("live formal completion timestamp is not UTC")

    usage = row.get("usage")
    successful_usage = row.get("successful_response_usage")
    attempts = row.get("attempt_usage")
    if (
        not _formal_usage_valid(usage)
        or not _formal_usage_valid(successful_usage)
        or not isinstance(attempts, list)
        or not attempts
        or any(
            not _formal_usage_valid(
                item, has_attempt=True, allow_response_status=True
            )
            or type(item.get("attempt")) is not int
            for item in attempts
        )
        or [item["attempt"] for item in attempts]
        != sorted({item["attempt"] for item in attempts})
        or attempts[-1]["attempt"] != row.get("attempt")
        or {
            key: sum(int(item[key]) for item in attempts)
            for key in _FORMAL_TOKEN_KEYS
        }
        != usage
        or {key: attempts[-1][key] for key in _FORMAL_TOKEN_KEYS}
        != successful_usage
        or "response_status" in attempts[-1]
    ):
        errors.append("live formal usage ledger is invalid")

    retry = row.get("retry_accounting") or {}
    if (
        not isinstance(retry, Mapping)
        or set(retry)
        != {
            "policy",
            "transport_failure_quota",
            "semantic_failure_quota",
            "hard_total_attempt_quota",
            "total_attempts",
            "transport_failure_count",
            "semantic_failure_count",
            "quota_exhausted",
        }
        or retry.get("policy") != "bounded-transport-semantic-failure-quotas-v1"
        or any(
            type(retry.get(key)) is not int
            for key in (
                "transport_failure_quota",
                "semantic_failure_quota",
                "hard_total_attempt_quota",
                "total_attempts",
                "transport_failure_count",
                "semantic_failure_count",
            )
        )
        or retry.get("transport_failure_quota") != 3
        or retry.get("semantic_failure_quota") != 3
        or retry.get("hard_total_attempt_quota") != 6
        or retry.get("total_attempts") != row.get("attempt")
        or retry.get("quota_exhausted") is not None
        or retry.get("transport_failure_count")
        + retry.get("semantic_failure_count")
        != row.get("attempt") - 1
    ):
        errors.append("live formal retry ledger is invalid")

    failure_attempts = row.get("failures_before_success")
    invalid_attempts = row.get("invalid_attempts_before_success")
    expected_prior_attempts = list(range(1, int(row.get("attempt") or 0)))
    if expected_prior_attempts:
        if (
            not isinstance(failure_attempts, list)
            or len(failure_attempts) != len(expected_prior_attempts)
            or [item.get("attempt") for item in failure_attempts]
            != expected_prior_attempts
        ):
            errors.append("live formal failure ledger is not contiguous")
            failure_attempts = []
    elif failure_attempts is not None:
        errors.append("live formal attempt-one receipt contains failure lineage")
        failure_attempts = []
    else:
        failure_attempts = []

    failure_base_keys = {
        "attempt",
        "termination",
        "error_type",
        "failure_category",
        "response_sha256",
        "response_hash_status",
    }
    failure_profiles = {
        frozenset(failure_base_keys),
        frozenset(failure_base_keys | {"http_status", "http_reason"}),
    }
    for failure in failure_attempts:
        if (
            not isinstance(failure, Mapping)
            or frozenset(failure) not in failure_profiles
            or failure.get("failure_category") not in {"transport", "semantic"}
            or not isinstance(failure.get("termination"), str)
            or not failure["termination"]
            or not isinstance(failure.get("error_type"), str)
            or not failure["error_type"]
            or failure.get("response_hash_status")
            not in {
                "complete_json_sha256",
                "terminal_stream_content_sha256",
                "partial_stream_content_sha256",
                "not_available_incomplete_transport_or_parser_failure",
            }
            or (
                failure.get("response_sha256") is not None
                and not re.fullmatch(
                    r"[0-9a-f]{64}", str(failure.get("response_sha256") or "")
                )
            )
            or ("http_status" in failure)
            is not (failure.get("error_type") == "HTTPError")
            or (
                "http_status" in failure
                and (
                    type(failure.get("http_status")) is not int
                    or not 100 <= failure["http_status"] <= 599
                    or not isinstance(failure.get("http_reason"), str)
                    or not failure["http_reason"]
                )
            )
            or (
                failure.get("response_hash_status")
                == "not_available_incomplete_transport_or_parser_failure"
            )
            is not (failure.get("response_sha256") is None)
        ):
            errors.append("live formal failure attempt schema is invalid")

    semantic_failures = [
        item for item in failure_attempts if item.get("failure_category") == "semantic"
    ]
    transport_failures = [
        item for item in failure_attempts if item.get("failure_category") == "transport"
    ]
    if (
        retry.get("semantic_failure_count") != len(semantic_failures)
        or retry.get("transport_failure_count") != len(transport_failures)
        or row.get("usage_is_lower_bound") is not bool(transport_failures)
    ):
        errors.append("live formal retry/failure lineage does not mirror")

    if invalid_attempts is None:
        invalid_attempts = []
    if not isinstance(invalid_attempts, list):
        errors.append("live formal invalid-attempt ledger is invalid")
        invalid_attempts = []
    invalid_by_attempt: dict[int, Mapping[str, Any]] = {}
    invalid_keys = {
        "attempt",
        "raw_review",
        "raw_review_sha256",
        "pre_normalization_review_sha256",
        "review_sha256",
        "review",
        "normalization_provenance",
        "validation_errors",
        "sse_integrity",
    }
    provenance_keys = {
        "policy",
        "raw_review_sha256",
        "pre_normalization_review_sha256",
        "normalized_review_sha256",
        "actions",
    }
    for item in invalid_attempts:
        raw_review = item.get("raw_review") if isinstance(item, Mapping) else None
        review = item.get("review") if isinstance(item, Mapping) else None
        provenance = (
            item.get("normalization_provenance")
            if isinstance(item, Mapping)
            else None
        )
        item_attempt = item.get("attempt") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item) != invalid_keys
            or type(item_attempt) is not int
            or item_attempt not in expected_prior_attempts
            or item_attempt in invalid_by_attempt
            or not isinstance(raw_review, Mapping)
            or not isinstance(review, Mapping)
            or item.get("raw_review_sha256") != sha256_json(raw_review)
            or item.get("review_sha256") != sha256_json(review)
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("pre_normalization_review_sha256") or ""),
            )
            or not isinstance(provenance, Mapping)
            or set(provenance) != provenance_keys
            or provenance.get("policy")
            != "strict-rejected-raw-normalization-provenance-v1"
            or provenance.get("raw_review_sha256")
            != item.get("raw_review_sha256")
            or provenance.get("pre_normalization_review_sha256")
            != item.get("pre_normalization_review_sha256")
            or provenance.get("normalized_review_sha256")
            != item.get("review_sha256")
            or not isinstance(provenance.get("actions"), list)
            or not isinstance(item.get("validation_errors"), list)
            or not item["validation_errors"]
            or any(
                not isinstance(error, str) or not error
                for error in item["validation_errors"]
            )
        ):
            errors.append("live formal invalid-attempt receipt is invalid")
            continue
        invalid_by_attempt[item_attempt] = item

    integrity = row.get("sse_integrity")
    integrity_attempts = row.get("sse_integrity_attempts")
    if (
        not isinstance(integrity_attempts, list)
        or not integrity_attempts
        or any(not isinstance(item, Mapping) for item in integrity_attempts)
        or [item.get("attempt") for item in integrity_attempts]
        != sorted({item.get("attempt") for item in integrity_attempts})
        or integrity_attempts[-1] != integrity
        or integrity_attempts[-1].get("attempt") != row.get("attempt")
    ):
        errors.append("live formal SSE attempt ledger is invalid")
        integrity_attempts = []
    sse_by_attempt = {
        item["attempt"]: item
        for item in integrity_attempts
        if type(item.get("attempt")) is int
    }
    errors.extend(
        _formal_terminal_sse_errors(
            integrity,
            attempt=int(row.get("attempt") or 0),
            completion_policy=str(row.get("sse_completion_policy") or ""),
            review_validation="passed",
            content_object=row.get("review")
            if isinstance(row.get("review"), Mapping)
            else None,
        )
    )
    for attempt_number, invalid in invalid_by_attempt.items():
        attempt_integrity = sse_by_attempt.get(attempt_number)
        if invalid.get("sse_integrity") != attempt_integrity:
            errors.append("live formal invalid-attempt/SSE mirror is invalid")
            continue
        errors.extend(
            _formal_terminal_sse_errors(
                attempt_integrity,
                attempt=attempt_number,
                completion_policy=str(row.get("sse_completion_policy") or ""),
                review_validation="failed",
                content_object=invalid.get("raw_review"),
            )
        )
        matching_failure = next(
            (
                failure
                for failure in semantic_failures
                if failure.get("attempt") == attempt_number
            ),
            None,
        )
        if (
            matching_failure is None
            or matching_failure.get("termination") != "review_validation_error"
            or matching_failure.get("error_type") != "ValueError"
            or matching_failure.get("response_hash_status")
            != "complete_json_sha256"
            or matching_failure.get("response_sha256")
            != invalid.get("raw_review_sha256")
        ):
            errors.append("live formal invalid-attempt/failure mirror is invalid")

    if set(invalid_by_attempt) != {
        item.get("attempt")
        for item in semantic_failures
        if item.get("termination") == "review_validation_error"
    }:
        errors.append("live formal semantic invalid-attempt ledger is incomplete")

    sse_failure_by_attempt = {
        int(failure["attempt"]): failure
        for failure in failure_attempts
        if failure.get("error_type") == "SSEIntegrityError"
        and type(failure.get("attempt")) is int
    }
    for attempt_number, failure in sse_failure_by_attempt.items():
        attempt_integrity = sse_by_attempt.get(attempt_number)
        if attempt_integrity is None:
            errors.append("live formal SSE failure receipt is missing")
            continue
        errors.extend(
            _formal_failed_sse_errors(
                attempt_integrity,
                attempt=attempt_number,
                completion_policy=str(row.get("sse_completion_policy") or ""),
            )
        )
        usage_status = (attempt_integrity.get("usage_integrity") or {}).get("status")
        response_sha = (
            attempt_integrity.get("response_sha256")
            or attempt_integrity.get("content_sha256")
        )
        expected_hash_status = (
            "complete_json_sha256"
            if attempt_integrity.get("response_sha256")
            else "terminal_stream_content_sha256"
            if attempt_integrity.get("finish_reason_stop_seen") is True
            and usage_status == "passed"
            and response_sha is not None
            else "partial_stream_content_sha256"
            if response_sha is not None
            else "not_available_incomplete_transport_or_parser_failure"
        )
        if (
            failure.get("termination") != attempt_integrity.get("termination")
            or failure.get("response_sha256") != response_sha
            or failure.get("response_hash_status") != expected_hash_status
        ):
            errors.append("live formal SSE failure/hash mirror is invalid")

    expected_sse_attempts = (
        {int(row.get("attempt") or 0)}
        | set(invalid_by_attempt)
        | set(sse_failure_by_attempt)
    )
    if set(sse_by_attempt) != expected_sse_attempts:
        errors.append("live formal SSE/failure-attempt ledger is incomplete")

    expected_usage_attempts = (
        {int(row.get("attempt") or 0)}
        | set(invalid_by_attempt)
        | {
            attempt_number
            for attempt_number in sse_failure_by_attempt
            if (
                sse_by_attempt.get(attempt_number, {}).get("usage_integrity") or {}
            ).get("status")
            == "passed"
        }
    )
    if {item.get("attempt") for item in attempts} != expected_usage_attempts:
        errors.append("live formal usage/semantic-attempt ledger is incomplete")
    for usage_item in attempts:
        usage_attempt = usage_item.get("attempt")
        if usage_attempt in sse_failure_by_attempt:
            if usage_item.get("response_status") != "rejected_sse_integrity":
                errors.append("live formal rejected-SSE usage marker is missing")
        elif "response_status" in usage_item:
            errors.append("live formal non-SSE usage has a rejected-SSE marker")
    errors.extend(_formal_canonicalization_errors(row))
    return errors


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(None):
        raise RuntimeError(f"{label} timestamp must be UTC")
    return parsed


def _materialization_episode_id(
    *,
    custom_id: str,
    source_success_sha256: str,
    request_sha256: str,
    execution_contract_sha256: str,
    materializer_policy_version: str,
    materialization_errors_sha256: str,
) -> str:
    digest = sha256_json(
        {
            "policy": MATERIALIZATION_ADOPTION_POLICY,
            "custom_id": custom_id,
            "source_success_sha256": source_success_sha256,
            "request_sha256": request_sha256,
            "execution_contract_sha256": execution_contract_sha256,
            "materializer_policy_version": materializer_policy_version,
            "materialization_errors_sha256": materialization_errors_sha256,
        }
    )
    return f"post150-materialization-{digest[:24]}"


def _materialization_errors(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str or not item for item in value)
    ):
        raise RuntimeError(f"{label} materialization errors are invalid")
    return list(value)


def _materialization_invalidation_value(
    *,
    source_success: Mapping[str, Any],
    request: Mapping[str, Any],
    materialization_errors: Sequence[str],
    execution_contract_sha256: str,
    materializer_policy_version: str,
    invalidated_at: str,
) -> dict[str, Any]:
    errors = _materialization_errors(
        list(materialization_errors), label="materialization invalidation"
    )
    source_sha256 = sha256_json(source_success)
    request_sha256 = sha256_json(request)
    errors_sha256 = sha256_json(errors)
    return {
        "custom_id": source_success.get("custom_id"),
        "status": "materialization_invalidated",
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "episode_id": _materialization_episode_id(
            custom_id=str(source_success.get("custom_id") or ""),
            source_success_sha256=source_sha256,
            request_sha256=request_sha256,
            execution_contract_sha256=execution_contract_sha256,
            materializer_policy_version=materializer_policy_version,
            materialization_errors_sha256=errors_sha256,
        ),
        "runner_version": source_success.get("runner_version"),
        "transport": source_success.get("transport"),
        "sse_completion_policy": source_success.get("sse_completion_policy"),
        "prompt_sha256": source_success.get("prompt_sha256"),
        "model": source_success.get("model"),
        "execution_contract_sha256": execution_contract_sha256,
        "sequence": source_success.get("sequence"),
        "request_sha256": request_sha256,
        "source_success_sha256": source_sha256,
        "previous_same_id_sha256": source_sha256,
        "materializer_policy_version": materializer_policy_version,
        "materialization_errors": errors,
        "materialization_errors_sha256": errors_sha256,
        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "attempt": 0,
        "api_call_performed": False,
        "usage": dict(_ZERO_USAGE),
        "invalidated_at": invalidated_at,
    }


def build_materialization_invalidation(
    *,
    source_success: Mapping[str, Any],
    request: Mapping[str, Any],
    materialization_errors: Sequence[str],
    execution_contract_sha256: str,
    materializer_policy_version: str,
) -> dict[str, Any]:
    """Build the zero-side-effect fence before a single fresh formal retry."""

    _require_sha256(
        execution_contract_sha256,
        label="materialization invalidation execution contract",
    )
    return _materialization_invalidation_value(
        source_success=source_success,
        request=request,
        materialization_errors=materialization_errors,
        execution_contract_sha256=execution_contract_sha256,
        materializer_policy_version=materializer_policy_version,
        invalidated_at=datetime.now(UTC).isoformat(),
    )


def _materialization_retry_started_value(
    *,
    invalidation: Mapping[str, Any],
    request: Mapping[str, Any],
    endpoint: str,
    started_at: str,
    request_config: Mapping[str, Any],
) -> dict[str, Any]:
    invalidation_sha256 = sha256_json(invalidation)
    return {
        "custom_id": invalidation.get("custom_id"),
        "status": "materialization_retry_started",
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "episode_id": invalidation.get("episode_id"),
        "runner_version": invalidation.get("runner_version"),
        "transport": invalidation.get("transport"),
        "sse_completion_policy": invalidation.get("sse_completion_policy"),
        "prompt_sha256": invalidation.get("prompt_sha256"),
        "model": invalidation.get("model"),
        "execution_contract_sha256": invalidation.get(
            "execution_contract_sha256"
        ),
        "sequence": invalidation.get("sequence"),
        "request_sha256": sha256_json(request),
        "source_success_sha256": invalidation.get("source_success_sha256"),
        "invalidation_sha256": invalidation_sha256,
        "previous_same_id_sha256": invalidation_sha256,
        "materializer_policy_version": invalidation.get(
            "materializer_policy_version"
        ),
        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "retry_ordinal": 1,
        "endpoint": endpoint,
        "request_config": dict(request_config),
        "side_effect_status": "unknown/in_flight",
        "started_at": started_at,
    }


def build_materialization_retry_started(
    *,
    invalidation: Mapping[str, Any],
    request: Mapping[str, Any],
    endpoint: str,
    request_timeout: float,
    max_attempts: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Consume the sole quota before the provider call can begin."""

    request_config = {
        "request_timeout_seconds": request_timeout,
        "max_attempts": max_attempts,
        "retry_delay_seconds": retry_delay,
        "temperature": EXPECTED_REREVIEW_TEMPERATURE,
        "response_format": EXPECTED_REREVIEW_RESPONSE_FORMAT,
        "max_tokens": "forbidden",
    }
    return _materialization_retry_started_value(
        invalidation=invalidation,
        request=request,
        endpoint=endpoint,
        started_at=datetime.now(UTC).isoformat(),
        request_config=request_config,
    )


def _bound_retry_usage(bound_result: Mapping[str, Any]) -> dict[str, int]:
    usage = bound_result.get("usage")
    if usage is None:
        return dict(_ZERO_USAGE)
    if not _formal_usage_valid(usage):
        raise RuntimeError("materialization retry bound-result usage is invalid")
    return {key: int(usage[key]) for key in _FORMAL_TOKEN_KEYS}


def _materialization_retry_outcome_value(
    *,
    started: Mapping[str, Any],
    request: Mapping[str, Any],
    bound_result: Mapping[str, Any],
    status: str,
    outcome_at: str,
    materialization_errors: Sequence[str] | None,
) -> dict[str, Any]:
    started_sha256 = sha256_json(started)
    value: dict[str, Any] = {
        "custom_id": started.get("custom_id"),
        "status": status,
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "episode_id": started.get("episode_id"),
        "runner_version": started.get("runner_version"),
        "transport": started.get("transport"),
        "sse_completion_policy": started.get("sse_completion_policy"),
        "prompt_sha256": started.get("prompt_sha256"),
        "model": started.get("model"),
        "execution_contract_sha256": started.get("execution_contract_sha256"),
        "sequence": started.get("sequence"),
        "request_sha256": sha256_json(request),
        "source_success_sha256": started.get("source_success_sha256"),
        "invalidation_sha256": started.get("invalidation_sha256"),
        "retry_started_sha256": started_sha256,
        "previous_same_id_sha256": started_sha256,
        "materializer_policy_version": started.get("materializer_policy_version"),
        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "retry_ordinal": 1,
        "bound_result_status": bound_result.get("status"),
        "bound_result": dict(bound_result),
        "bound_result_sha256": sha256_json(bound_result),
        "attempt": bound_result.get("attempt"),
        "usage_is_lower_bound": bound_result.get("usage_is_lower_bound"),
        "usage": _bound_retry_usage(bound_result),
        "outcome_at": outcome_at,
    }
    if status == "materialization_rejected":
        errors = _materialization_errors(
            list(materialization_errors or []), label="materialization rejection"
        )
        value["materialization_errors"] = errors
        value["materialization_errors_sha256"] = sha256_json(errors)
    elif materialization_errors is not None:
        raise RuntimeError("materialization call-failed outcome cannot carry errors")
    return value


def build_materialization_retry_outcome(
    *,
    started: Mapping[str, Any],
    request: Mapping[str, Any],
    bound_result: Mapping[str, Any],
    materialization_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Wrap a non-adoptable fresh result without appending a naked result row."""

    if bound_result.get("status") == "succeeded":
        status = "materialization_rejected"
        if not materialization_errors:
            raise RuntimeError("materialization rejection requires exact errors")
    elif bound_result.get("status") == "failed":
        status = "materialization_call_failed"
        if materialization_errors is not None:
            raise RuntimeError("materialization call failure cannot carry errors")
    else:
        raise RuntimeError("materialization retry result status is invalid")
    return _materialization_retry_outcome_value(
        started=started,
        request=request,
        bound_result=bound_result,
        status=status,
        outcome_at=datetime.now(UTC).isoformat(),
        materialization_errors=materialization_errors,
    )


def _validate_materialization_invalidation(
    row: Mapping[str, Any],
    *,
    source_success: Mapping[str, Any],
    request: Mapping[str, Any],
    exact_errors: Sequence[str],
    execution_contract_sha256: str,
    materializer_policy_version: str,
) -> None:
    invalidated_at = _utc_timestamp(
        row.get("invalidated_at"), label="materialization invalidation"
    )
    source_at = _utc_timestamp(
        source_success.get("completed_at"), label="materialization source"
    )
    if invalidated_at < source_at:
        raise RuntimeError("materialization invalidation predates its source")
    expected = _materialization_invalidation_value(
        source_success=source_success,
        request=request,
        materialization_errors=exact_errors,
        execution_contract_sha256=execution_contract_sha256,
        materializer_policy_version=materializer_policy_version,
        invalidated_at=str(row.get("invalidated_at") or ""),
    )
    if dict(row) != expected:
        raise RuntimeError("materialization invalidation exact receipt is invalid")


def _validate_materialization_retry_started(
    row: Mapping[str, Any],
    *,
    invalidation: Mapping[str, Any],
    request: Mapping[str, Any],
    expected_endpoint: str,
) -> None:
    started_at = _utc_timestamp(
        row.get("started_at"), label="materialization retry start"
    )
    invalidated_at = _utc_timestamp(
        invalidation.get("invalidated_at"), label="materialization invalidation"
    )
    if started_at < invalidated_at:
        raise RuntimeError("materialization retry start predates invalidation")
    config = row.get("request_config")
    if (
        not isinstance(config, Mapping)
        or dict(config) != MATERIALIZATION_RETRY_REQUEST_CONFIG
        or type(config.get("request_timeout_seconds")) is not float
        or type(config.get("max_attempts")) is not int
        or type(config.get("retry_delay_seconds")) is not float
        or type(config.get("temperature")) is not int
    ):
        raise RuntimeError("materialization retry request config is invalid")
    expected = _materialization_retry_started_value(
        invalidation=invalidation,
        request=request,
        endpoint=expected_endpoint,
        started_at=str(row.get("started_at") or ""),
        request_config=MATERIALIZATION_RETRY_REQUEST_CONFIG,
    )
    if dict(row) != expected:
        raise RuntimeError("materialization retry-start exact receipt is invalid")


def _formal_failed_call_errors(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    expected_sequence: int,
    live_runner_version: str,
    live_transport: str,
) -> list[str]:
    errors: list[str] = []
    required = {
        "custom_id",
        "status",
        "attempt",
        "model",
        "runner_version",
        "transport",
        "sse_completion_policy",
        "prompt_sha256",
        "error",
        "completed_at",
        "usage_is_lower_bound",
        "retry_accounting",
        "execution_contract_sha256",
        "sequence",
    }
    optional = {
        "invalid_review",
        "invalid_attempts",
        "failure_termination",
        "failure_attempts",
        "sse_integrity",
        "sse_integrity_attempts",
        "attempt_usage",
        "usage",
    }
    request_body = request.get("request") or {}
    if (
        not required.issubset(result)
        or not set(result).issubset(required | optional)
        or result.get("custom_id") != request.get("custom_id")
        or result.get("status") != "failed"
        or result.get("model") != request_body.get("model")
        or result.get("runner_version") != live_runner_version
        or result.get("transport") != live_transport
        or result.get("sse_completion_policy")
        != "strict-terminal-usage-complete-json-v2"
        or result.get("prompt_sha256") != request.get("prompt_sha256")
        or type(result.get("attempt")) is not int
        or int(result["attempt"]) < 1
        or type(result.get("sequence")) is not int
        or result.get("sequence") != expected_sequence
        or type(result.get("error")) is not str
        or not result.get("error")
        or type(result.get("usage_is_lower_bound")) is not bool
    ):
        errors.append("materialization failed-call identity/schema is invalid")
    try:
        _utc_timestamp(result.get("completed_at"), label="materialization failed call")
    except RuntimeError as exc:
        errors.append(str(exc))
    retry = result.get("retry_accounting")
    attempt = result.get("attempt")
    if (
        not isinstance(retry, Mapping)
        or set(retry)
        != {
            "policy",
            "transport_failure_quota",
            "semantic_failure_quota",
            "hard_total_attempt_quota",
            "total_attempts",
            "transport_failure_count",
            "semantic_failure_count",
            "quota_exhausted",
        }
        or retry.get("policy") != "bounded-transport-semantic-failure-quotas-v1"
        or any(
            type(retry.get(key)) is not int
            for key in (
                "transport_failure_quota",
                "semantic_failure_quota",
                "hard_total_attempt_quota",
                "total_attempts",
                "transport_failure_count",
                "semantic_failure_count",
            )
        )
        or retry.get("transport_failure_quota") != 3
        or retry.get("semantic_failure_quota") != 3
        or retry.get("hard_total_attempt_quota") != 6
        or retry.get("total_attempts") != attempt
        or retry.get("transport_failure_count", -1)
        + retry.get("semantic_failure_count", -1)
        != attempt
        or retry.get("quota_exhausted") not in {"transport", "semantic", "hard_total"}
    ):
        errors.append("materialization failed-call retry ledger is invalid")
    usage = result.get("usage")
    attempt_usage = result.get("attempt_usage")
    if usage is None:
        if attempt_usage is not None:
            errors.append("materialization failed-call usage mirror is invalid")
    elif (
        not _formal_usage_valid(usage)
        or not isinstance(attempt_usage, list)
        or not attempt_usage
        or any(
            not _formal_usage_valid(item, has_attempt=True, allow_response_status=True)
            for item in attempt_usage
        )
        or [item.get("attempt") for item in attempt_usage]
        != sorted({item.get("attempt") for item in attempt_usage})
        or {
            key: sum(int(item[key]) for item in attempt_usage)
            for key in _FORMAL_TOKEN_KEYS
        }
        != usage
    ):
        errors.append("materialization failed-call usage ledger is invalid")
    return errors


def formal_rereview_failed_result_errors(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    expected_sequence: int,
    execution_contract_sha256: str,
    live_runner_version: str,
    live_transport: str,
    validate_invalid_attempt: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], Sequence[str]
    ],
) -> list[str]:
    """Replay every legal terminal 2.32 failure profile exactly.

    This is the failure-side counterpart of ``_formal_live_receipt_errors``.
    It covers transport-only, semantic-only and interleaved hard-total quota
    exhaustion without trusting nested mirrors or a declared status.
    """

    errors = _formal_failed_call_errors(
        result,
        request,
        expected_sequence=expected_sequence,
        live_runner_version=live_runner_version,
        live_transport=live_transport,
    )
    common = {
        "custom_id",
        "status",
        "attempt",
        "model",
        "runner_version",
        "transport",
        "sse_completion_policy",
        "prompt_sha256",
        "error",
        "completed_at",
        "usage_is_lower_bound",
        "retry_accounting",
        "failure_termination",
        "failure_attempts",
        "execution_contract_sha256",
        "sequence",
    }
    sse_fields = {"sse_integrity", "sse_integrity_attempts"}
    invalid_fields = {"invalid_review", "invalid_attempts"}
    usage_fields = {"attempt_usage", "usage"}
    allowed_top_profiles = {
        frozenset(common | sse | invalid | usage)
        for sse in (set(), sse_fields)
        for invalid in (set(), invalid_fields)
        for usage in (set(), usage_fields)
        if not invalid or sse == sse_fields and usage == usage_fields
    }
    if frozenset(result) not in allowed_top_profiles:
        errors.append("materialization failed-call exact top schema is invalid")
    if result.get("execution_contract_sha256") != execution_contract_sha256:
        errors.append("materialization failed-call execution contract drift")
    attempt = result.get("attempt")
    retry = result.get("retry_accounting") or {}
    failure_attempts = result.get("failure_attempts")
    if (
        type(attempt) is not int
        or not isinstance(failure_attempts, list)
        or len(failure_attempts) != attempt
        or [item.get("attempt") for item in failure_attempts]
        != list(range(1, attempt + 1))
    ):
        errors.append("materialization failed-call failure ledger is not contiguous")
        failure_attempts = []
    failure_base_keys = {
        "attempt",
        "termination",
        "error_type",
        "failure_category",
        "response_sha256",
        "response_hash_status",
    }
    failure_profiles = {
        frozenset(failure_base_keys),
        frozenset(failure_base_keys | {"http_status", "http_reason"}),
    }
    for failure in failure_attempts:
        if (
            not isinstance(failure, Mapping)
            or frozenset(failure) not in failure_profiles
            or failure.get("failure_category") not in {"transport", "semantic"}
            or type(failure.get("attempt")) is not int
            or not isinstance(failure.get("termination"), str)
            or not failure["termination"]
            or not isinstance(failure.get("error_type"), str)
            or not failure["error_type"]
            or failure.get("response_hash_status")
            not in {
                "complete_json_sha256",
                "terminal_stream_content_sha256",
                "partial_stream_content_sha256",
                "not_available_incomplete_transport_or_parser_failure",
            }
            or (
                failure.get("response_sha256") is not None
                and not re.fullmatch(
                    r"[0-9a-f]{64}", str(failure.get("response_sha256") or "")
                )
            )
            or ("http_status" in failure)
            is not (failure.get("error_type") == "HTTPError")
            or (
                "http_status" in failure
                and (
                    type(failure.get("http_status")) is not int
                    or not 100 <= failure["http_status"] <= 599
                    or not isinstance(failure.get("http_reason"), str)
                    or not failure["http_reason"]
                )
            )
            or (
                failure.get("response_hash_status")
                == "not_available_incomplete_transport_or_parser_failure"
            )
            is not (failure.get("response_sha256") is None)
        ):
            errors.append("materialization failed-call failure attempt is invalid")
    semantic_failures = [
        item for item in failure_attempts if item.get("failure_category") == "semantic"
    ]
    transport_failures = [
        item for item in failure_attempts if item.get("failure_category") == "transport"
    ]
    expected_quota_exhausted = (
        "hard_total"
        if type(attempt) is int and attempt >= 6
        else "transport"
        if len(transport_failures) >= 3
        else "semantic"
        if len(semantic_failures) >= 3
        else "hard_total"
    )
    if (
        retry.get("semantic_failure_count") != len(semantic_failures)
        or retry.get("transport_failure_count") != len(transport_failures)
        or result.get("usage_is_lower_bound") is not bool(transport_failures)
        or result.get("failure_termination")
        != (failure_attempts[-1].get("termination") if failure_attempts else None)
        or retry.get("quota_exhausted") != expected_quota_exhausted
    ):
        errors.append("materialization failed-call retry/failure mirror is invalid")

    integrity = result.get("sse_integrity")
    integrity_attempts = result.get("sse_integrity_attempts")
    if integrity_attempts is None:
        if integrity is not None:
            errors.append("materialization failed-call SSE mirror is incomplete")
        integrity_attempts = []
    elif (
        not isinstance(integrity_attempts, list)
        or not integrity_attempts
        or any(not isinstance(item, Mapping) for item in integrity_attempts)
        or [item.get("attempt") for item in integrity_attempts]
        != sorted({item.get("attempt") for item in integrity_attempts})
        or integrity_attempts[-1] != integrity
        or any(
            type(item.get("attempt")) is not int
            or not 1 <= int(item["attempt"]) <= int(attempt or 0)
            for item in integrity_attempts
        )
    ):
        errors.append("materialization failed-call SSE ledger is invalid")
        integrity_attempts = []
    sse_by_attempt = {
        int(item["attempt"]): item
        for item in integrity_attempts
        if type(item.get("attempt")) is int
    }

    invalid_attempts = result.get("invalid_attempts")
    if invalid_attempts is None:
        invalid_attempts = []
    if not isinstance(invalid_attempts, list):
        errors.append("materialization failed-call invalid-attempt ledger is invalid")
        invalid_attempts = []
    invalid_by_attempt: dict[int, Mapping[str, Any]] = {}
    semantic_payload = json.loads(
        str((request.get("request") or {}).get("messages", [{}, {}])[1].get("content") or "{}")
    )
    for item in invalid_attempts:
        item_attempt = item.get("attempt") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or type(item_attempt) is not int
            or item_attempt in invalid_by_attempt
            or item_attempt not in {failure.get("attempt") for failure in semantic_failures}
            or list(validate_invalid_attempt(dict(item), dict(result), semantic_payload))
        ):
            errors.append("materialization failed-call invalid-attempt receipt is invalid")
            continue
        invalid_by_attempt[item_attempt] = item
        attempt_integrity = sse_by_attempt.get(item_attempt)
        if item.get("sse_integrity") != attempt_integrity:
            errors.append("materialization failed-call invalid-attempt/SSE mirror is invalid")
            continue
        errors.extend(
            _formal_terminal_sse_errors(
                attempt_integrity,
                attempt=item_attempt,
                completion_policy=str(result.get("sse_completion_policy") or ""),
                review_validation="failed",
                content_object=item.get("raw_review"),
            )
        )
        failure = next(
            item
            for item in semantic_failures
            if item.get("attempt") == item_attempt
        )
        if (
            failure.get("termination") != "review_validation_error"
            or failure.get("error_type") != "ValueError"
            or failure.get("response_hash_status") != "complete_json_sha256"
            or failure.get("response_sha256") != item.get("raw_review_sha256")
        ):
            errors.append("materialization failed-call semantic mirror is invalid")
    if set(invalid_by_attempt) != {
        item.get("attempt")
        for item in semantic_failures
        if item.get("termination") == "review_validation_error"
    }:
        errors.append("materialization failed-call semantic ledger is incomplete")
    if bool(invalid_attempts) is not (result.get("invalid_review") is not None):
        errors.append("materialization failed-call invalid-review mirror is invalid")
    elif invalid_attempts and result.get("invalid_review") != invalid_attempts[-1].get("review"):
        errors.append("materialization failed-call latest invalid review is not mirrored")

    if failure_attempts:
        final_failure = failure_attempts[-1]
        final_type = str(final_failure.get("error_type") or "")
        expected_error: str | None = None
        if final_type == "HTTPError":
            expected_error = (
                f"HTTPError: status={final_failure.get('http_status')} "
                f"reason={final_failure.get('http_reason')}"
            )
            if final_failure.get("termination") != "http_error":
                errors.append("materialization failed-call HTTP termination is invalid")
        elif final_type in {"URLError", "IncompleteRead", "TimeoutError"}:
            expected_error = f"{final_type}: transport request failed"
            if final_failure.get("termination") != "transport_error":
                errors.append("materialization failed-call transport termination is invalid")
        elif final_failure.get("termination") == "review_validation_error":
            final_invalid = invalid_by_attempt.get(int(final_failure.get("attempt") or 0))
            if final_type != "ValueError" or final_invalid is None:
                errors.append("materialization failed-call review termination is invalid")
            else:
                expected_error = "ValueError: " + "; ".join(
                    final_invalid.get("validation_errors") or []
                )
        elif not str(result.get("error") or "").startswith(final_type + ": "):
            errors.append("materialization failed-call final error type is not mirrored")
        if expected_error is not None and result.get("error") != expected_error:
            errors.append("materialization failed-call final error is not mirrored")

    sse_failure_by_attempt = {
        int(failure["attempt"]): failure
        for failure in failure_attempts
        if failure.get("error_type") == "SSEIntegrityError"
        and type(failure.get("attempt")) is int
    }
    for attempt_number, failure in sse_failure_by_attempt.items():
        attempt_integrity = sse_by_attempt.get(attempt_number)
        if attempt_integrity is None:
            errors.append("materialization failed-call SSE failure receipt is missing")
            continue
        errors.extend(
            _formal_failed_sse_errors(
                attempt_integrity,
                attempt=attempt_number,
                completion_policy=str(result.get("sse_completion_policy") or ""),
            )
        )
        usage_status = (attempt_integrity.get("usage_integrity") or {}).get("status")
        response_sha = attempt_integrity.get("response_sha256") or attempt_integrity.get(
            "content_sha256"
        )
        expected_hash_status = (
            "complete_json_sha256"
            if attempt_integrity.get("response_sha256")
            else "terminal_stream_content_sha256"
            if attempt_integrity.get("finish_reason_stop_seen") is True
            and usage_status == "passed"
            and response_sha is not None
            else "partial_stream_content_sha256"
            if response_sha is not None
            else "not_available_incomplete_transport_or_parser_failure"
        )
        if (
            failure.get("termination") != attempt_integrity.get("termination")
            or failure.get("response_sha256") != response_sha
            or failure.get("response_hash_status") != expected_hash_status
        ):
            errors.append("materialization failed-call SSE/hash mirror is invalid")
    generic_terminal_by_attempt = {
        int(failure["attempt"]): failure
        for failure in semantic_failures
        if failure.get("attempt") not in invalid_by_attempt
        and failure.get("attempt") not in sse_failure_by_attempt
        and failure.get("attempt") in sse_by_attempt
    }
    for attempt_number, failure in generic_terminal_by_attempt.items():
        attempt_integrity = sse_by_attempt[attempt_number]
        if failure.get("termination") != "review_or_parser_error":
            errors.append(
                "materialization failed-call generic semantic termination is invalid"
            )
        errors.extend(
            _formal_terminal_sse_errors(
                attempt_integrity,
                attempt=attempt_number,
                completion_policy=str(result.get("sse_completion_policy") or ""),
                review_validation=str(
                    attempt_integrity.get("review_validation") or ""
                ),
                content_object=None,
            )
        )
    if set(sse_by_attempt) != (
        set(sse_failure_by_attempt)
        | set(invalid_by_attempt)
        | set(generic_terminal_by_attempt)
    ):
        errors.append("materialization failed-call SSE attempt coverage is incomplete")

    attempt_usage = result.get("attempt_usage")
    usage = result.get("usage")
    expected_usage_attempts = {
        attempt_number
        for attempt_number, attempt_integrity in sse_by_attempt.items()
        if (attempt_integrity.get("usage_integrity") or {}).get("status")
        == "passed"
    }
    if expected_usage_attempts:
        if (
            not isinstance(attempt_usage, list)
            or {item.get("attempt") for item in attempt_usage}
            != expected_usage_attempts
            or not _formal_usage_valid(usage)
            or any(
                not _formal_usage_valid(
                    item, has_attempt=True, allow_response_status=True
                )
                for item in attempt_usage
            )
            or {
                key: sum(int(item[key]) for item in attempt_usage)
                for key in _FORMAL_TOKEN_KEYS
            }
            != usage
            or any(
                ("response_status" in item)
                is not (
                    next(
                        failure
                        for failure in failure_attempts
                        if failure.get("attempt") == item.get("attempt")
                    ).get("error_type")
                    == "SSEIntegrityError"
                )
                for item in attempt_usage
            )
        ):
            errors.append("materialization failed-call usage ledger is invalid")
    elif attempt_usage is not None or usage is not None:
        errors.append("materialization failed-call unexpected usage ledger")
    return errors


def _validate_materialization_retry_outcome(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    invalidation: Mapping[str, Any],
    started: Mapping[str, Any],
    request: Mapping[str, Any],
    expected_sequence: int,
    live_runner_version: str,
    live_transport: str,
    validate_live_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    validate_materialization_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    validate_failed_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
) -> None:
    outcome_at = _utc_timestamp(row.get("outcome_at"), label="materialization outcome")
    started_at = _utc_timestamp(started.get("started_at"), label="materialization retry start")
    if outcome_at < started_at:
        raise RuntimeError("materialization outcome predates retry start")
    bound_result = row.get("bound_result")
    if not isinstance(bound_result, Mapping):
        raise RuntimeError("materialization outcome bound result is invalid")
    status = str(row.get("status") or "")
    if status == "materialization_rejected":
        exact_errors = list(validate_materialization_result(dict(bound_result), request))
        if not exact_errors:
            raise RuntimeError("materialization rejected result actually passes")
        replay_errors = _formal_live_receipt_errors(
            bound_result, request, expected_sequence=expected_sequence
        )
        replay_errors.extend(validate_live_result(dict(bound_result), request))
        if replay_errors:
            raise RuntimeError(
                "materialization rejected bound live result is invalid: "
                + "; ".join(replay_errors)
            )
        materialization_errors: Sequence[str] | None = exact_errors
    elif status == "materialization_call_failed":
        failure_errors = list(validate_failed_result(dict(bound_result), dict(request)))
        if failure_errors:
            raise RuntimeError(
                "materialization call-failed bound result is invalid: "
                + "; ".join(failure_errors)
            )
        materialization_errors = None
    else:
        raise RuntimeError("materialization outcome status is invalid")
    expected = _materialization_retry_outcome_value(
        started=started,
        request=request,
        bound_result=bound_result,
        status=status,
        outcome_at=str(row.get("outcome_at") or ""),
        materialization_errors=materialization_errors,
    )
    if dict(row) != expected:
        raise RuntimeError("materialization outcome exact receipt is invalid")
    if (
        started.get("source_success_sha256") != sha256_json(source)
        or started.get("invalidation_sha256") != sha256_json(invalidation)
    ):
        raise RuntimeError("materialization outcome episode lineage is invalid")


def formal_rereview_state_reducer(
    rows: Sequence[dict[str, Any]],
    *,
    requests_by_id: Mapping[str, dict[str, Any]],
    execution_contract_sha256: str,
    expected_prompt_sha256: str,
    expected_model: str,
    expected_endpoint: str,
    live_runner_version: str,
    live_transport: str,
    saved_runner_version: str,
    saved_transport: str,
    materializer_policy_version: str,
    validate_live_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    validate_saved_result: Callable[
        [dict[str, Any], dict[str, Any], Sequence[dict[str, Any]]],
        Sequence[str],
    ],
    validate_materialization_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    validate_failed_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ]
    | None = None,
    label: str,
    additional_saved_profiles: Sequence[tuple[str, str]] = (),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Replay each current id's physical append chain as a fail-closed state machine."""

    _require_sha256(execution_contract_sha256, label=f"{label} execution contract")
    primary_saved_profile = (saved_runner_version, saved_transport)
    extra_saved_profiles = tuple(additional_saved_profiles)
    if (
        any(
            not isinstance(profile, tuple)
            or len(profile) != 2
            or any(not isinstance(value, str) or not value for value in profile)
            for profile in extra_saved_profiles
        )
    ):
        raise RuntimeError(f"{label}: saved response profiles are invalid")
    saved_profiles = {primary_saved_profile, *extra_saved_profiles}
    if (
        len(saved_profiles) != 1 + len(extra_saved_profiles)
        or (live_runner_version, live_transport) in saved_profiles
    ):
        raise RuntimeError(f"{label}: saved response profiles are invalid")
    failed_result_validator = validate_failed_result or (
        lambda _result, _request: [
            "exact live failed-result validator was not configured"
        ]
    )
    expected_sequence_by_id = {
        custom_id: sequence
        for sequence, custom_id in enumerate(requests_by_id, start=1)
    }
    rows_by_id: dict[str, list[dict[str, Any]]] = {
        custom_id: [] for custom_id in requests_by_id
    }
    allowed_statuses = {
        "succeeded",
        "failed",
        "materialization_invalidated",
        "materialization_retry_started",
        "materialization_rejected",
        "materialization_call_failed",
    }
    for row in rows:
        custom_id = str(row.get("custom_id") or "")
        status = row.get("status")
        if custom_id not in requests_by_id:
            raise RuntimeError(f"{label}: response event is outside current pack")
        if status not in allowed_statuses:
            raise RuntimeError(f"{label}: unknown response status: {status!r}")
        rows_by_id[custom_id].append(row)

    state: dict[str, dict[str, dict[str, Any]]] = {
        "successful_by_id": {},
        "retry_eligible_by_id": {},
        "invalidated_by_id": {},
        "retry_started_by_id": {},
        "terminal_by_id": {},
        "adopted_by_id": {},
        "episode_by_id": {},
    }
    for custom_id, request in requests_by_id.items():
        chain = rows_by_id[custom_id]
        success_count = 0
        current_mode = "incomplete"
        current_row: dict[str, Any] | None = None
        current_errors: list[str] = []
        adopted: dict[str, Any] | None = None
        for index, row in enumerate(chain):
            status = row.get("status")
            profile = (row.get("runner_version"), row.get("transport"))
            if current_mode == "invalidated" and status != (
                "materialization_retry_started"
            ):
                raise RuntimeError(
                    f"{label}: materialization invalidation must be followed by retry start"
                )
            if current_mode == "started" and status not in {
                "succeeded",
                "materialization_rejected",
                "materialization_call_failed",
            }:
                raise RuntimeError(
                    f"{label}: materialization retry start must be followed by a fresh result"
                )
            if current_mode in {
                "adopted",
                "terminal_materialization_rejected",
                "terminal_materialization_call_failed",
            }:
                raise RuntimeError(
                    f"{label}: materialization terminal state rejects any later event"
                )
            if status == "succeeded":
                success_count += 1
                if (
                    row.get("execution_contract_sha256")
                    != execution_contract_sha256
                    or row.get("prompt_sha256") != expected_prompt_sha256
                    or row.get("model") != expected_model
                    or not isinstance(row.get("review"), Mapping)
                ):
                    raise RuntimeError(
                        f"{label}: successful response execution contract drift"
                    )
                if profile in saved_profiles:
                    if success_count > 1:
                        raise RuntimeError(
                            f"{label}: duplicate successful response: {custom_id}"
                        )
                    replay_errors = list(validate_saved_result(dict(row), request, rows))
                    if replay_errors:
                        raise RuntimeError(
                            f"{label}: saved response replay failed: "
                            + "; ".join(str(error) for error in replay_errors)
                        )
                    current_mode = "saved"
                    current_row = row
                    current_errors = []
                    continue
                if profile != (live_runner_version, live_transport):
                    raise RuntimeError(
                        f"{label}: successful response formal profile drift"
                    )
                replay_errors = _formal_live_receipt_errors(
                    row,
                    request,
                    expected_sequence=expected_sequence_by_id[custom_id],
                )
                replay_errors.extend(validate_live_result(dict(row), request))
                if replay_errors:
                    raise RuntimeError(
                        f"{label}: live response replay failed: "
                        + "; ".join(replay_errors)
                    )
                materialization_errors = list(
                    validate_materialization_result(dict(row), request)
                )
                if index > 0 and chain[index - 1].get("status") == (
                    "materialization_retry_started"
                ):
                    if success_count != 2 or index < 3:
                        raise RuntimeError(
                            f"{label}: duplicate successful response lacks complete materialization episode"
                        )
                    source, invalidation, started = chain[index - 3 : index]
                    if (
                        source.get("status") != "succeeded"
                        or invalidation.get("status") != "materialization_invalidated"
                        or started.get("status") != "materialization_retry_started"
                    ):
                        raise RuntimeError(
                            f"{label}: materialization adoption chain is not adjacent"
                        )
                    if materialization_errors:
                        raise RuntimeError(
                            f"{label}: adopted fresh success still fails materialization"
                        )
                    started_at = _utc_timestamp(
                        started.get("started_at"), label="materialization retry start"
                    )
                    completed_at = _utc_timestamp(
                        row.get("completed_at"), label="materialization fresh success"
                    )
                    if completed_at < started_at:
                        raise RuntimeError(
                            f"{label}: materialization fresh success predates retry start"
                        )
                    current_mode = "adopted"
                    current_row = row
                    current_errors = []
                    adopted = {
                        "episode_id": str(started.get("episode_id") or ""),
                        "source_success_sha256": sha256_json(source),
                        "invalidation_sha256": sha256_json(invalidation),
                        "retry_started_sha256": sha256_json(started),
                        "fresh_success_sha256": sha256_json(row),
                        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
                        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
                        "stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
                        "stream_content_sha256_authoritative": False,
                        "policy": MATERIALIZATION_ADOPTION_POLICY,
                    }
                    continue
                if success_count > 1:
                    raise RuntimeError(
                        f"{label}: duplicate successful response: {custom_id}"
                    )
                current_mode = "eligible" if materialization_errors else "direct"
                current_row = row
                current_errors = materialization_errors
                continue

            if status == "materialization_invalidated":
                if index == 0 or chain[index - 1].get("status") != "succeeded":
                    raise RuntimeError(
                        f"{label}: materialization invalidation is not adjacent to source"
                    )
                source = chain[index - 1]
                if (source.get("runner_version"), source.get("transport")) != (
                    live_runner_version,
                    live_transport,
                ):
                    raise RuntimeError(
                        f"{label}: materialization invalidation source profile is invalid"
                    )
                exact_errors = list(validate_materialization_result(dict(source), request))
                if not exact_errors:
                    raise RuntimeError(
                        f"{label}: materializer-pass success cannot be invalidated"
                    )
                _validate_materialization_invalidation(
                    row,
                    source_success=source,
                    request=request,
                    exact_errors=exact_errors,
                    execution_contract_sha256=execution_contract_sha256,
                    materializer_policy_version=materializer_policy_version,
                )
                state["episode_by_id"][custom_id] = {
                    "episode_id": str(row.get("episode_id") or ""),
                    "invalidation_sha256": sha256_json(row),
                }
                current_mode = "invalidated"
                current_row = row
                current_errors = exact_errors
                continue

            if status == "materialization_retry_started":
                if (
                    index == 0
                    or chain[index - 1].get("status")
                    != "materialization_invalidated"
                ):
                    raise RuntimeError(
                        f"{label}: materialization retry start is not adjacent to invalidation"
                    )
                invalidation = chain[index - 1]
                _validate_materialization_retry_started(
                    row,
                    invalidation=invalidation,
                    request=request,
                    expected_endpoint=expected_endpoint,
                )
                current_mode = "started"
                current_row = row
                current_errors = []
                continue

            if status in {
                "materialization_rejected",
                "materialization_call_failed",
            }:
                if (
                    index < 3
                    or chain[index - 1].get("status")
                    != "materialization_retry_started"
                    or chain[index - 2].get("status")
                    != "materialization_invalidated"
                    or chain[index - 3].get("status") != "succeeded"
                ):
                    raise RuntimeError(
                        f"{label}: materialization outcome chain is not adjacent"
                    )
                source, invalidation, started = chain[index - 3 : index]
                _validate_materialization_retry_outcome(
                    row,
                    source=source,
                    invalidation=invalidation,
                    started=started,
                    request=request,
                    expected_sequence=expected_sequence_by_id[custom_id],
                    live_runner_version=live_runner_version,
                    live_transport=live_transport,
                    validate_live_result=validate_live_result,
                    validate_materialization_result=validate_materialization_result,
                    validate_failed_result=failed_result_validator,
                )
                current_mode = "terminal_" + str(status)
                current_row = row
                current_errors = []
                continue

            if status == "failed":
                failed_errors = list(failed_result_validator(dict(row), request))
                if failed_errors:
                    raise RuntimeError(
                        f"{label}: live failed response replay failed: "
                        + "; ".join(str(error) for error in failed_errors)
                    )
                current_mode = "incomplete"
                current_row = row
                current_errors = []
                adopted = None
                continue

            if str(status or "").startswith("materialization_"):
                raise RuntimeError(f"{label}: unknown materialization episode event")
            if index > 0 and chain[index - 1].get("status") == (
                "materialization_retry_started"
            ):
                raise RuntimeError(
                    f"{label}: naked result follows materialization retry start"
                )
            current_mode = "incomplete"
            current_row = row
            current_errors = []
            adopted = None

        if current_row is None:
            continue
        if current_mode in {"direct", "saved", "adopted"}:
            state["successful_by_id"][custom_id] = dict(current_row)
            if current_mode == "adopted" and adopted is not None:
                state["adopted_by_id"][custom_id] = adopted
        elif current_mode == "eligible":
            state["retry_eligible_by_id"][custom_id] = {
                "source_success": dict(current_row),
                "source_success_sha256": sha256_json(current_row),
                "materialization_errors": current_errors,
                "materialization_errors_sha256": sha256_json(current_errors),
                "sequence": expected_sequence_by_id[custom_id],
            }
        elif current_mode == "invalidated":
            state["invalidated_by_id"][custom_id] = dict(current_row)
        elif current_mode == "started":
            state["retry_started_by_id"][custom_id] = dict(current_row)
        elif current_mode.startswith("terminal"):
            state["terminal_by_id"][custom_id] = dict(current_row)
    if len(state["episode_by_id"]) > MATERIALIZATION_RETRY_QUOTA:
        raise RuntimeError(
            f"{label}: global materialization fresh-retry quota exceeded"
        )
    return state


def execute_materialization_retry_episodes(
    *,
    rows: list[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    state_loader: Callable[
        [list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]
    ],
    append_event: Callable[[dict[str, Any]], None],
    execute_one: Callable[[dict[str, Any]], dict[str, Any]],
    validate_materialization_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    execution_contract_sha256: str,
    materializer_policy_version: str,
    endpoint: str,
    request_timeout: float,
    max_attempts: int,
    retry_delay: float,
    target_custom_ids: Sequence[str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    int,
]:
    """Execute at most one fenced fresh retry per eligible current id.

    The caller must validate/cache its API credential and complete calibration
    before entering this function.  Every event is replayed in memory before
    append.  A persisted ``started`` event is an unknown provider side-effect,
    so restart is fail-stop and never calls ``execute_one`` again.
    """

    if (
        type(request_timeout) is not float
        or type(max_attempts) is not int
        or type(retry_delay) is not float
        or {
            "request_timeout_seconds": request_timeout,
            "max_attempts": max_attempts,
            "retry_delay_seconds": retry_delay,
            "temperature": EXPECTED_REREVIEW_TEMPERATURE,
            "response_format": EXPECTED_REREVIEW_RESPONSE_FORMAT,
            "max_tokens": "forbidden",
        }
        != MATERIALIZATION_RETRY_REQUEST_CONFIG
    ):
        raise RuntimeError("materialization retry runtime config is not frozen")
    current_rows = list(rows)
    sequence_by_id = {
        str(request["custom_id"]): sequence
        for sequence, request in enumerate(requests, start=1)
    }
    request_by_id = {str(request["custom_id"]): request for request in requests}
    target_ids = (
        set(request_by_id)
        if target_custom_ids is None
        else {str(custom_id) for custom_id in target_custom_ids}
    )
    if not target_ids or target_ids - set(request_by_id):
        raise RuntimeError("materialization retry target set is invalid")
    state = state_loader(current_rows)
    # scoped执行只被目标集内的未知副作用/终态阻断；episode配额仍从完整账本计算。
    blocking_ids = set(state["retry_started_by_id"]) | set(state["terminal_by_id"])
    if blocking_ids & target_ids:
        return current_rows, state, 2
    for custom_id in request_by_id:
        if custom_id not in target_ids:
            continue
        state = state_loader(current_rows)
        if custom_id in state["successful_by_id"]:
            continue
        if custom_id in state["retry_started_by_id"] or custom_id in state[
            "terminal_by_id"
        ]:
            return current_rows, state, 2
        request = request_by_id[custom_id]
        invalidation = state["invalidated_by_id"].get(custom_id)
        if invalidation is None:
            eligible = state["retry_eligible_by_id"].get(custom_id)
            if eligible is None:
                continue
            if len(state["episode_by_id"]) >= MATERIALIZATION_RETRY_QUOTA:
                return current_rows, state, 2
            invalidation = build_materialization_invalidation(
                source_success=eligible["source_success"],
                request=request,
                materialization_errors=eligible["materialization_errors"],
                execution_contract_sha256=execution_contract_sha256,
                materializer_policy_version=materializer_policy_version,
            )
            candidate_rows = [*current_rows, invalidation]
            candidate_state = state_loader(candidate_rows)
            if custom_id not in candidate_state["invalidated_by_id"]:
                raise RuntimeError("materialization invalidation replay did not persist")
            append_event(invalidation)
            current_rows = candidate_rows
            state = candidate_state

        started = build_materialization_retry_started(
            invalidation=invalidation,
            request=request,
            endpoint=endpoint,
            request_timeout=request_timeout,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
        candidate_rows = [*current_rows, started]
        candidate_state = state_loader(candidate_rows)
        if custom_id not in candidate_state["retry_started_by_id"]:
            raise RuntimeError("materialization retry-start replay did not persist")
        append_event(started)
        current_rows = candidate_rows

        raw_result = execute_one(request)
        if "execution_contract_sha256" in raw_result or "sequence" in raw_result:
            raise RuntimeError("raw materialization retry result is already outer-bound")
        bound_result = {
            **raw_result,
            "execution_contract_sha256": execution_contract_sha256,
            "sequence": sequence_by_id[custom_id],
        }
        result_status = bound_result.get("status")
        if result_status == "succeeded":
            exact_errors = list(validate_materialization_result(bound_result, request))
            if exact_errors:
                outcome = build_materialization_retry_outcome(
                    started=started,
                    request=request,
                    bound_result=bound_result,
                    materialization_errors=exact_errors,
                )
                candidate_rows = [*current_rows, outcome]
                candidate_state = state_loader(candidate_rows)
                if custom_id not in candidate_state["terminal_by_id"]:
                    raise RuntimeError("materialization rejection replay did not persist")
                append_event(outcome)
                return candidate_rows, candidate_state, 2
            candidate_rows = [*current_rows, bound_result]
            candidate_state = state_loader(candidate_rows)
            if (
                custom_id not in candidate_state["successful_by_id"]
                or custom_id not in candidate_state["adopted_by_id"]
            ):
                raise RuntimeError("materialization fresh success replay did not adopt")
            append_event(bound_result)
            current_rows = candidate_rows
            state = candidate_state
            continue
        if result_status == "failed":
            outcome = build_materialization_retry_outcome(
                started=started,
                request=request,
                bound_result=bound_result,
            )
            candidate_rows = [*current_rows, outcome]
            candidate_state = state_loader(candidate_rows)
            if custom_id not in candidate_state["terminal_by_id"]:
                raise RuntimeError("materialization call failure replay did not persist")
            append_event(outcome)
            return candidate_rows, candidate_state, 2
        raise RuntimeError("materialization retry returned an unknown status")
    return current_rows, state_loader(current_rows), 0


def materialization_state_summary(
    state: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return a compact hash-bound audit of the already-replayed state."""

    adopted = state.get("adopted_by_id") or {}
    invalidated = state.get("invalidated_by_id") or {}
    started = state.get("retry_started_by_id") or {}
    terminal = state.get("terminal_by_id") or {}
    eligible = state.get("retry_eligible_by_id") or {}
    episodes = state.get("episode_by_id") or {}
    return {
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "materializer_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_consumed_count": len(episodes),
        "global_fresh_retry_episode_ids": sorted(
            str(episode.get("episode_id") or "") for episode in episodes.values()
        ),
        "stream_receipt_authority": {
            "policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
            "content_sha256_authoritative": False,
            "content_chars_authoritative": False,
            "event_counts_authoritative": False,
            "semantic_authority": (
                "response_sha256_recomputed_from_review_plus_exact_core_payload_"
                "canonicalization_materializer_and_reducer_replay"
            ),
        },
        "adopted_episode_count": len(adopted),
        "adopted_custom_ids": sorted(adopted),
        "adopted_episodes": {
            custom_id: dict(adopted[custom_id]) for custom_id in sorted(adopted)
        },
        "retry_eligible_count": len(eligible),
        "retry_eligible_custom_ids": sorted(eligible),
        "invalidated_pending_count": len(invalidated),
        "invalidated_pending_receipt_sha256_by_id": {
            custom_id: sha256_json(invalidated[custom_id])
            for custom_id in sorted(invalidated)
        },
        "retry_started_in_doubt_count": len(started),
        "retry_started_receipt_sha256_by_id": {
            custom_id: sha256_json(started[custom_id])
            for custom_id in sorted(started)
        },
        "terminal_episode_count": len(terminal),
        "terminal_outcome_sha256_by_id": {
            custom_id: sha256_json(terminal[custom_id])
            for custom_id in sorted(terminal)
        },
    }


def formal_rereview_completion_errors(
    state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_custom_ids: Sequence[str],
) -> list[str]:
    """Require one authoritative success per current id and no pending episode."""

    expected = {str(custom_id) for custom_id in expected_custom_ids}
    successful = set(state.get("successful_by_id") or {})
    errors: list[str] = []
    if successful != expected:
        errors.append("formal successful ids do not exactly cover current request ids")
    for key in (
        "retry_eligible_by_id",
        "invalidated_by_id",
        "retry_started_by_id",
        "terminal_by_id",
    ):
        if state.get(key):
            errors.append(f"formal completion retains nonterminal state: {key}")
    return errors


def materialization_adoption_for_authoritative_success(
    rows: Sequence[dict[str, Any]], authoritative_success: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the complete adjacent episode receipt for one adopted success."""

    custom_id = str(authoritative_success.get("custom_id") or "")
    authoritative_sha256 = sha256_json(authoritative_success)
    chain = [row for row in rows if str(row.get("custom_id") or "") == custom_id]
    matches = [
        index for index, row in enumerate(chain) if sha256_json(row) == authoritative_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError("authoritative materialization success is not unique in ledger")
    index = matches[0]
    if index < 3 or chain[index - 1].get("status") != "materialization_retry_started":
        return None
    source, invalidation, started = chain[index - 3 : index]
    if (
        source.get("status") != "succeeded"
        or invalidation.get("status") != "materialization_invalidated"
        or started.get("status") != "materialization_retry_started"
        or invalidation.get("source_success_sha256") != sha256_json(source)
        or started.get("invalidation_sha256") != sha256_json(invalidation)
        or started.get("previous_same_id_sha256") != sha256_json(invalidation)
    ):
        raise RuntimeError("authoritative materialization adoption lineage is invalid")
    return {
        "policy": MATERIALIZATION_ADOPTION_POLICY,
        "fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "global_fresh_retry_quota": MATERIALIZATION_RETRY_QUOTA,
        "stream_receipt_policy": FORMAL_STREAM_DIAGNOSTIC_POLICY,
        "stream_content_sha256_authoritative": False,
        "episode_id": started.get("episode_id"),
        "source_success_sha256": sha256_json(source),
        "invalidation_sha256": sha256_json(invalidation),
        "retry_started_sha256": sha256_json(started),
        "fresh_success_sha256": authoritative_sha256,
    }


def formal_rereview_success_ledger(
    rows: Sequence[dict[str, Any]],
    *,
    requests_by_id: Mapping[str, dict[str, Any]],
    execution_contract_sha256: str,
    expected_prompt_sha256: str,
    expected_model: str,
    live_runner_version: str,
    live_transport: str,
    saved_runner_version: str,
    saved_transport: str,
    validate_live_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
    validate_saved_result: Callable[
        [dict[str, Any], dict[str, Any], Sequence[dict[str, Any]]],
        Sequence[str],
    ],
    label: str,
    expected_endpoint: str = (
        EXPECTED_REREVIEW_BASE_URL + "/chat/completions"
    ),
    materializer_policy_version: str = "compatibility-no-materializer-policy",
    validate_materialization_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ]
    | None = None,
    validate_failed_result: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ]
    | None = None,
    additional_saved_profiles: Sequence[tuple[str, str]] = (),
) -> dict[str, dict[str, Any]]:
    """Load one exact live or independently replayed saved-success profile.

    Live receipts retain the frozen paid execution profile.  A saved receipt
    is a separate offline profile under the same outer request/execution
    contract and is accepted only after its append-only source audit replays.
    This keeps existing paid successes byte-for-byte authoritative instead of
    weakening their transport or runner checks.
    """

    materialization_validator = validate_materialization_result or (
        lambda _result, _request: []
    )
    state = formal_rereview_state_reducer(
        rows,
        requests_by_id=requests_by_id,
        execution_contract_sha256=execution_contract_sha256,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_model=expected_model,
        expected_endpoint=expected_endpoint,
        live_runner_version=live_runner_version,
        live_transport=live_transport,
        saved_runner_version=saved_runner_version,
        saved_transport=saved_transport,
        materializer_policy_version=materializer_policy_version,
        validate_live_result=validate_live_result,
        validate_saved_result=validate_saved_result,
        validate_materialization_result=materialization_validator,
        validate_failed_result=validate_failed_result,
        label=label,
        additional_saved_profiles=additional_saved_profiles,
    )
    return state["successful_by_id"]


def validate_request_pack_identity(
    rows: Sequence[dict[str, Any]],
    *,
    expected_prompt_sha256: str,
    pack_name: str,
    expected_count: int | None = None,
    expected_pack_sha256: str | None = None,
    custom_id_mode: RequestIdMode = "payload",
    original_custom_ids_by_rereview_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate actual prompt, payload, request ids, and the complete pack hash."""

    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(
            f"{pack_name}: expected {expected_count} requests, found {len(rows)}"
        )
    actual_pack_sha256 = sha256_json(rows)
    if expected_pack_sha256 is not None and actual_pack_sha256 != expected_pack_sha256:
        raise RuntimeError(f"{pack_name}: prepared request-pack SHA drift")

    custom_ids: list[str] = []
    payload_hashes: list[str] = []
    for position, row in enumerate(rows, start=1):
        custom_id = str(row.get("custom_id") or "")
        if not custom_id:
            raise RuntimeError(f"{pack_name}: request {position} has no custom_id")
        custom_ids.append(custom_id)

        declared_prompt_sha256 = str(row.get("prompt_sha256") or "")
        request = row.get("request")
        if not isinstance(request, dict):
            raise RuntimeError(f"{pack_name}: {custom_id} has no request object")
        messages = request.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or not isinstance(messages[0], dict)
            or not isinstance(messages[1], dict)
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise RuntimeError(f"{pack_name}: {custom_id} message envelope is invalid")
        system_content = messages[0].get("content")
        user_content = messages[1].get("content")
        if not isinstance(system_content, str) or not isinstance(user_content, str):
            raise RuntimeError(f"{pack_name}: {custom_id} message content must be text")
        actual_prompt_sha256 = sha256_bytes(system_content.encode("utf-8"))
        if (
            declared_prompt_sha256 != expected_prompt_sha256
            or actual_prompt_sha256 != expected_prompt_sha256
        ):
            raise RuntimeError(f"{pack_name}: {custom_id} actual prompt SHA drift")

        try:
            payload = json.loads(user_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{pack_name}: {custom_id} payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{pack_name}: {custom_id} payload must be a JSON object")
        payload_sha256 = sha256_json(payload)
        payload_hashes.append(payload_sha256)
        if str(row.get("payload_sha256") or "") != payload_sha256:
            raise RuntimeError(f"{pack_name}: {custom_id} payload SHA drift")

        if custom_id_mode == "payload":
            expected_custom_id = f"concept-review-{payload_sha256[:20]}"
        else:
            original_custom_id = str(
                (original_custom_ids_by_rereview_id or {}).get(custom_id) or ""
            )
            if not original_custom_id:
                raise RuntimeError(
                    f"{pack_name}: {custom_id} has no original request identity"
                )
            expected_custom_id = contract_rereview_custom_id(
                original_custom_id, payload_sha256
            )
        if custom_id != expected_custom_id:
            raise RuntimeError(f"{pack_name}: {custom_id} is not payload-bound")

    if len(set(custom_ids)) != len(custom_ids):
        raise RuntimeError(f"{pack_name}: request custom_ids are not unique")
    if len(set(payload_hashes)) != len(payload_hashes):
        raise RuntimeError(f"{pack_name}: semantic payloads are not unique")
    return {
        "case_count": len(rows),
        "request_pack_sha256": actual_pack_sha256,
        "prompt_sha256": expected_prompt_sha256,
    }


def validate_blocker_rereview_prepared_receipt(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact locally prepared changed-case rereview boundary."""

    exact_counts = (
        summary.get("validated_no_index_proposal_count")
        == EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS
        and summary.get("supplemental_reviewer_block_proposal_count")
        == EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        and summary.get("data_blocker_case_count")
        == EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
        and summary.get("case_count") == EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
    )
    if not exact_counts:
        raise RuntimeError(
            "changed-case prepared receipt must cover exactly 13 base + 2 supplemental "
            "data blockers"
        )
    difficulty_counts = summary.get("difficulty_counts")
    if (
        not isinstance(difficulty_counts, Mapping)
        or set(difficulty_counts) != {"easy", "medium", "hard"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in difficulty_counts.values()
        )
        or sum(difficulty_counts.values()) != EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS
    ):
        raise RuntimeError("changed-case prepared receipt difficulty counts do not total 15")
    if (
        summary.get("schema_version")
        != "stt.eval.suite-v2-blocker-rereview-pack.v1"
        or summary.get("status")
        != "prepared_pending_independent_same_prompt_review"
        or summary.get("write_performed") is not True
    ):
        raise RuntimeError("changed-case prepared receipt is not a written review pack")
    if summary.get("prompt_sha256") != EXPECTED_REVIEW_PROMPT_SHA256:
        raise RuntimeError("changed-case prepared receipt does not use the frozen full-suite prompt")
    if summary.get("superseded_review_reused") is not False:
        raise RuntimeError("changed-case prepared receipt permits superseded review reuse")
    if (
        summary.get("index_rebuild_required") is not False
        or summary.get("evidence_refresh_required") is not False
    ):
        raise RuntimeError("changed-case prepared receipt is not the no-index-refresh variant")
    request_pack_sha256 = str(summary.get("request_pack_sha256") or "")
    if len(request_pack_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in request_pack_sha256
    ):
        raise RuntimeError("changed-case prepared receipt has an invalid request-pack SHA")
    execution_contract_sha256 = str(
        summary.get("execution_contract_sha256") or ""
    )
    if len(execution_contract_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in execution_contract_sha256
    ):
        raise RuntimeError(
            "changed-case prepared receipt has an invalid execution-contract SHA"
        )
    return {
        "base_proposal_count": EXPECTED_BLOCKER_REREVIEW_BASE_PROPOSALS,
        "supplemental_proposal_count": (
            EXPECTED_BLOCKER_REREVIEW_SUPPLEMENTAL_PROPOSALS
        ),
        "data_blocker_case_count": EXPECTED_BLOCKER_REREVIEW_DATA_BLOCKERS,
        "request_pack_sha256": request_pack_sha256,
    }


def validate_contract_rereview_prepared_receipt(
    summary: Mapping[str, Any],
) -> None:
    """Require a written, independent, same-semantic contract-rereview pack."""

    if (
        summary.get("schema_version")
        != "stt.eval.suite-v2-contract-rereview-pack.v1"
        or summary.get("status")
        != "prepared_pending_independent_same-semantic_review"
        or summary.get("write_performed") is not True
        or summary.get("semantic_payload_unchanged") is not True
        or summary.get("dataset_cases_changed") is not False
        or summary.get("new_custom_ids") is not True
        or summary.get("same_discriminative_variant_gate_required") is not True
    ):
        raise RuntimeError("contract-rereview prepared receipt is not immutable/written")
    case_count = summary.get("case_count")
    parent_case_ids = summary.get("parent_case_ids")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < 1
        or not isinstance(parent_case_ids, list)
        or len(parent_case_ids) != case_count
        or len({str(value) for value in parent_case_ids}) != case_count
    ):
        raise RuntimeError("contract-rereview prepared receipt case identity is invalid")


def validate_block_union_freshness(
    audit: Mapping[str, Any],
    *,
    requests_path: Path,
    responses_path: Path,
    visual_audit_path: Path,
    base_proposals_path: Path,
    supplemental_proposals_path: Path,
    label: str = "block-union audit",
) -> dict[str, str]:
    """Recompute every immutable input hash recorded by a block-union audit."""

    review = audit.get("review")
    inputs = audit.get("inputs")
    if not isinstance(review, Mapping) or not isinstance(inputs, Mapping):
        raise RuntimeError(f"{label}: missing review/input hash receipts")
    actual = {
        "request_pack_sha256": sha256_file(requests_path),
        "responses_sha256": sha256_file(responses_path),
        "visual_audit_sha256": sha256_file(visual_audit_path),
        "base_proposals_sha256": sha256_file(base_proposals_path),
        "supplemental_proposals_sha256": sha256_file(supplemental_proposals_path),
    }
    expected = {
        "request_pack_sha256": str(review.get("request_pack_sha256") or ""),
        "responses_sha256": str(review.get("responses_sha256") or ""),
        "visual_audit_sha256": str(inputs.get("visual_audit_sha256") or ""),
        "base_proposals_sha256": str(inputs.get("base_proposals_sha256") or ""),
        "supplemental_proposals_sha256": str(
            inputs.get("supplemental_proposals_sha256") or ""
        ),
    }
    drifted = sorted(key for key, value in actual.items() if expected[key] != value)
    if drifted:
        raise RuntimeError(f"{label}: stale input hash receipt(s): {drifted}")
    return actual


def validate_proposal_validation_binding(
    validation: Mapping[str, Any],
    *,
    proposals_path: Path,
    label: str = "proposal validation",
) -> str:
    """Require a validation receipt to bind the exact proposal file it approved."""

    actual = sha256_file(proposals_path)
    if str(validation.get("proposal_file_sha256") or "") != actual:
        raise RuntimeError(f"{label}: proposal file SHA drift")
    return actual
