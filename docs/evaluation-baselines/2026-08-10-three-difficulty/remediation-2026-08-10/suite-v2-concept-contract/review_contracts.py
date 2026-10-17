"""Run a calibrated, resumable full-suite LLM review of concept contracts.

Dry-run is the default.  Paid execution requires four explicit conditions:
``--execute``, ``--acknowledge-cost``, ``--confirm-full-review`` and
``--confirm-no-concurrent-review``.  The API key is read only from the named
environment variable and is never printed or written.

The semantic payload intentionally excludes case ids, difficulty, production
answers, pass/fail outcomes, and metrics.  A separate local request map restores
lineage after review.  Every one of the 150 source cases uses the exact same
system prompt hash.
"""

from __future__ import annotations

import argparse
import atexit
from copy import deepcopy
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.metrics import concept_variant_covered  # noqa: E402
from concept_materialization import (  # noqa: E402
    ABSTENTION_SENTINEL_CANONICAL_TEXT,
    ABSTENTION_SENTINEL_POLARITY,
    ABSTENTION_SENTINEL_VARIANTS,
    abstention_variant_is_refusal as _abstention_variant_is_refusal,
    materialize_concept_groups as _shared_materialize_concept_groups,
)


ROOT = Path(__file__).resolve().parent
REQUESTS_PATH = ROOT / "review" / "requests.jsonl"
RESPONSES_PATH = ROOT / "review" / "responses.jsonl"
CALIBRATION_RESPONSES_PATH = ROOT / "review" / "calibration-responses.jsonl"
EXECUTION_SUMMARY_PATH = ROOT / "review" / "execution-summary.json"
EXECUTION_LOCK_PATH = ROOT / "review" / "execution.lock"

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max"
DEFAULT_KEY_ENV = "DASHSCOPE_API_KEY"
EXPECTED_CASE_CALLS = 150
RUNNER_VERSION = "2.32.0-v229-formal-ledger-strict-scalars"
TRANSPORT = "openai-compatible-sse-v4-strict-terminal-schema"
SSE_COMPLETION_POLICY = "strict-terminal-usage-complete-json-v2"
# This is an offline post-150 wrapper profile, not a live runner revision.
# Keeping it separate preserves every paid 2.32 receipt and the frozen
# post-150 execution contract while making saved provenance explicit.
POST150_SAVED_RECOVERY_RUNNER_VERSION = (
    "post150-saved-recovery.1-same-concept-variant-subset"
)
POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION = (
    "post150-saved-recovery.2-quoted-branch-key-span"
)
SAVED_RESPONSE_TRANSPORT = "saved-response-recovery-no-api"
SAVED_SAME_CONCEPT_VARIANT_SUBSET_POLICY = (
    "saved-latest-final-terminal-same-concept-existing-any-of-subset-v1"
)
SAVED_QUOTE_KEY_BRANCH_SPAN_RECOVERY_POLICY = (
    "saved-latest-final-terminal-quoted-branch-key-exact-payload-span-v1"
)
SAVED_QUOTE_KEY_RECOVERY_EVENT_TIME_POLICY = (
    "source-completed-at-plus-one-microsecond-v1"
)
POST150_SAVED_RECOVERY_SOURCE_RUNNER_VERSION = (
    "2.32.0-v229-formal-ledger-strict-scalars"
)
INVERSE_EDGE_CANONICALIZATION_POLICY = (
    "same-response-authoritative-inverse-edge-canonicalization-v1"
)
EXACT_VARIANT_DEDUPLICATION_POLICY = (
    "same-response-normalized-exact-any-of-deduplication-v1"
)
COVERAGE_STATUS_CANONICALIZATION_POLICY = (
    "same-response-zero-direct-concept-support-status-alias-to-uncovered-v4"
)
LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY = (
    "same-response-zero-support-not-required-to-uncovered-v1"
)
LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY_V2 = (
    "same-response-zero-support-status-alias-to-uncovered-v2"
)
LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY_V3 = (
    "same-response-zero-support-status-alias-to-uncovered-v3"
)
COVERAGE_STATUS_ZERO_SUPPORT_ALIASES = frozenset(
    {"not_required", "not_applicable", "not_needed", "not_covered"}
)
NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY = (
    "saved-response-coverage-status-then-inverse-branch-labels-v2"
)
REDUNDANT_FIELD_RECONSTRUCTION_POLICY = (
    "saved-final-complete-attempt-same-response-redundant-field-reconstruction-v1"
)
REDUNDANT_FIELD_COVERAGE_COMPOSITION_POLICY = (
    "saved-final-complete-attempt-redundant-fields-then-coverage-alias-v1"
)
EXACT_PAYLOAD_BRANCH_SPAN_QUOTE_PREFIX_RECOVERY_POLICY = (
    "saved-final-complete-attempt-exact-payload-branch-span-quote-prefix-v1"
)
LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY = (
    "saved-final-complete-attempt-abstain-factual-constraint-removal-v1"
)
ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY = (
    "saved-final-complete-attempt-abstain-sentinel-and-constraint-projection-v2"
)
CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY = (
    "fresh-api-calibration-only-abstain-sentinel-projection-v1"
)
LEGACY_NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY = (
    "saved-response-coverage-status-then-inverse-branch-labels-v1"
)
LEGACY_COVERAGE_STATUS_POLICY_BY_RUNNER = {
    "2.15.0-not-required-coverage-canonicalization": (
        LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY
    ),
    "2.16.0-unsupported-required-concept-block": (
        LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY
    ),
    "2.18.0-zero-support-coverage-status-aliases": (
        LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY_V2
    ),
    "2.19.0-native-block-zero-support-status-alias": (
        LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY_V3
    ),
}
LEGACY_NONSEMANTIC_COMPOSITION_POLICY_BY_RUNNER = {
    "2.21.0-saved-nonsemantic-canonicalizer-composition": (
        LEGACY_NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY
    ),
    "2.22.0-saved-quote-fragment-inverse-composition": (
        LEGACY_NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY
    ),
}
QUOTE_FRAGMENT_BRANCH_RECONSTRUCTION_POLICY = (
    "saved-response-unique-quoted-branch-fragment-reconstruction-v1"
)
QUOTE_FRAGMENT_INVERSE_COMPOSITION_POLICY = (
    "saved-response-quote-fragment-then-inverse-branch-labels-v1"
)
QUOTE_FRAGMENT_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)
UNSUPPORTED_REQUIRED_CONCEPT_BLOCK_POLICY = (
    "multi-response-all-of-unsupported-required-concept-revise-to-block-v2"
)
LEGACY_UNSUPPORTED_REQUIRED_CONCEPT_BLOCK_POLICY = (
    "same-response-unsupported-required-concept-revise-to-block-v1"
)
EMPTY_REQUIRED_BRANCH_BLOCK_POLICY = (
    "multi-response-context-preamble-empty-required-branch-revise-to-block-v1"
)
VERDICT_BLOCK_CONSENSUS_POLICY = (
    "semantic-quota-exhausted-multi-response-verdict-block-consensus-v1"
)
SAVED_STRICT_REVALIDATION_POLICY = (
    "single-saved-response-strict-revalidation-after-validator-policy-v1"
)
RAW_NORMALIZATION_PROVENANCE_POLICY = (
    "strict-rejected-raw-normalization-provenance-v1"
)
RAW_PROVENANCE_REQUIRED_RUNNER_VERSIONS = frozenset(
    {
        "2.23.0-constraint-evidence-separation-raw-provenance",
        "2.24.0-saved-redundant-field-format-recovery",
        "2.25.0-saved-exact-payload-branch-span-quote-prefix-recovery",
        "2.26.0-saved-abstain-factual-constraint-recovery",
        "2.27.0-saved-abstain-sentinel-recovery",
        "2.28.0-calibration-only-abstain-sentinel-projection",
        "2.29.0-real-sse-calibration-ledger",
        "2.30.0-saved-abstain-sentinel-mismatch-recovery",
        "2.31.0-v229-formal-failure-envelope-replay",
        "2.32.0-v229-formal-ledger-strict-scalars",
    }
)
EXACT_PAYLOAD_BRANCH_SPAN_SOURCE_RUNNER_VERSIONS = frozenset(
    {
        "2.24.0-saved-redundant-field-format-recovery",
        "2.25.0-saved-exact-payload-branch-span-quote-prefix-recovery",
    }
)
ABSTAIN_FACTUAL_CONSTRAINT_SOURCE_RUNNER_VERSIONS = frozenset(
    {
        "2.25.0-saved-exact-payload-branch-span-quote-prefix-recovery",
        "2.26.0-saved-abstain-factual-constraint-recovery",
    }
)
ABSTAIN_SENTINEL_SOURCE_RUNNER_BY_TARGET = {
    "2.29.0-real-sse-calibration-ledger": (
        "2.26.0-saved-abstain-factual-constraint-recovery"
    ),
    "2.32.0-v229-formal-ledger-strict-scalars": (
        "2.29.0-real-sse-calibration-ledger"
    ),
}
PRE_RAW_PROVENANCE_RUNNER_VERSIONS = frozenset(
    {
        "2.1.0-internal-crossref-repair",
        "2.2.0-prior-response-recovery",
        "2.3.0-cross-attempt-union",
        "2.5.0-sse-abstention-factual-guard",
        "2.6.0-sse-eof-integrity",
        "2.7.0-sse-null-usage-integrity",
        "2.8.0-sse-strict-terminal-schema",
        "2.9.0-same-response-inverse-edge",
        "2.10.0-same-response-exact-dedupe",
        "2.11.0-partial-evidence-coverage",
        "2.12.0-bounded-transport-semantic-quotas",
        "2.13.0-redacted-provider-error-receipt",
        "2.14.0-allowlisted-provider-error-receipt",
        "2.15.0-not-required-coverage-canonicalization",
        "2.16.0-unsupported-required-concept-block",
        "2.17.0-multi-response-verdict-block-consensus",
        "2.18.0-zero-support-coverage-status-aliases",
        "2.19.0-native-block-zero-support-status-alias",
        "2.20.0-all-of-unsupported-required-consensus-block",
        "2.21.0-saved-nonsemantic-canonicalizer-composition",
        "2.22.0-saved-quote-fragment-inverse-composition",
    }
)
RETRY_ACCOUNTING_POLICY = "bounded-transport-semantic-failure-quotas-v1"
PROVIDER_ERROR_RECEIPT_POLICY = "allowlisted-provider-error-receipt-v2"
PROVIDER_ERROR_SAFE_CODES = frozenset(
    {
        # Alibaba Cloud Model Studio / DashScope documented identifiers.
        "accessdenied.unpurchased",
        "arrearage",
        "conflict",
        "data_inspection_failed",
        "datainspectionfailed",
        "insufficient_quota",
        "internalerror",
        "internalerror.algo.invalidparameter",
        "invalidparameter",
        "limit_burst_rate",
        "limit_requests",
        "limitrequests",
        "model.accessdenied",
        "model_not_found",
        "modelnotfound",
        "notfound",
        "responsetimeout",
        "throttling",
        "throttling.allocationquota",
        "throttling.burstrate",
        "throttling.ratequota",
        "workspacenotfound",
        # OpenAI-compatible stable categories used by this endpoint family.
        "authentication_error",
        "content_filter",
        "content_filter_error",
        "context_length_exceeded",
        "internal_error",
        "invalid_api_key",
        "invalid_request_error",
        "model_not_supported",
        "output_length_exceeded",
        "rate_limit_exceeded",
        "request_timeout",
        "server_error",
        "service_unavailable",
    }
)
PROVIDER_ERROR_SAFE_TYPES = frozenset(
    {
        "api_error",
        "authentication_error",
        "content_filter_error",
        "data_inspection_failed",
        "invalid_request_error",
        "not_found_error",
        "permission_error",
        "rate_limit_error",
        "server_error",
        "service_unavailable_error",
        "timeout_error",
    }
)
PROVIDER_ERROR_SAFE_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "failed",
        "rejected",
    }
)
MAX_FAILURE_QUOTA_PER_CATEGORY = 3

_CASE_ID_RE = re.compile(r"P3500-(?:EASY|MEDIUM|HARD)(?:-V2)?-\d{3}", re.IGNORECASE)
_FORBIDDEN_KEYS = {
    "case_id",
    "parent_case_id",
    "difficulty",
    "production_answer",
    "final_answer",
    "pre_grounding_answer",
    "production_final_correct",
    "production_final_accuracy",
    "production_metrics",
    "strict_pass",
}

_REVIEW_ALLOWED_FIELDS = {
    "verdict",
    "answerability",
    "required_concepts",
    "critical_constraints",
    "required_branches",
    "evidence_concept_coverage",
    "review_notes",
}
_REQUIRED_CONCEPTS_ALLOWED_FIELDS = {"all_of"}
_CONCEPT_ALLOWED_FIELDS = {
    "concept_key",
    "canonical_text",
    "any_of",
    "evidence_labels",
    "branch_labels",
}
_CRITICAL_CONSTRAINTS_ALLOWED_FIELDS = {"numerics", "identifiers", "polarity"}
_NUMERIC_ALLOWED_FIELDS = {"value", "unit", "evidence_labels"}
_IDENTIFIER_ALLOWED_FIELDS = {"value", "evidence_labels"}
_POLARITY_ALLOWED_FIELDS = {"required", "cues"}
# The original blind-review prompt required polarity preservation but did not
# prescribe a vocabulary.  Freeze the finite vocabulary actually needed by the
# deterministic candidates and semantically valid reviewer responses.  Keeping
# the mixed/conditional/abstention categories distinct avoids a lossy mapping,
# while rejecting arbitrary free-form labels remains fail-closed.
_POLARITY_REQUIRED_VALUES = frozenset(
    {
        "positive",
        "negative",
        "not_applicable",
        "mixed_preserve_each_claim",
        "affirmative",
        "positive_with_conditional",
        "affirmative_or_explicit_yes_no",
        "abstain_without_factual_claims",
    }
)
_BRANCH_ALLOWED_FIELDS = {"branch_label", "query_span", "required_concept_keys"}
_COVERAGE_ALLOWED_FIELDS = {
    "evidence_label",
    "required_concept_keys",
    "status",
}


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    payload: dict[str, Any]
    expected: dict[str, Any]


class SSEIntegrityError(ValueError):
    """Fail-closed SSE error carrying content-free transport audit metadata."""

    def __init__(
        self,
        message: str,
        *,
        audit: dict[str, Any],
        validated_usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.audit = audit
        self.validated_usage = dict(validated_usage or {})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _acquire_execution_lock(
    *, recover_stale: bool, purpose: str = "suite-v2 full-150 concept-contract review"
) -> str:
    """Acquire a process-unique paid-review lock using atomic file creation."""

    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": purpose,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(EXECUTION_LOCK_PATH, flags)
    except FileExistsError:
        if not recover_stale:
            detail = EXECUTION_LOCK_PATH.read_text(encoding="utf-8", errors="replace")[:500]
            raise SystemExit(
                "paid execution refused; another review lock exists. Verify the recorded process, "
                "then pass --recover-stale-lock only if it is no longer running. lock=" + detail
            )
        EXECUTION_LOCK_PATH.unlink()
        descriptor = os.open(EXECUTION_LOCK_PATH, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def release() -> None:
        try:
            current = json.loads(EXECUTION_LOCK_PATH.read_text(encoding="utf-8"))
            if current.get("token") == token:
                EXECUTION_LOCK_PATH.unlink()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

    atexit.register(release)
    return token


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                errors.append(f"{path}.{key}: forbidden semantic field")
            errors.extend(_walk_forbidden(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_walk_forbidden(nested, f"{path}[{index}]"))
    elif isinstance(value, str) and _CASE_ID_RE.search(value):
        errors.append(f"{path}: case identifier leaked into semantic payload")
    return errors


def _load_requests() -> tuple[list[dict[str, Any]], str, str]:
    rows = _read_jsonl(REQUESTS_PATH)
    if len(rows) != EXPECTED_CASE_CALLS:
        raise RuntimeError(f"expected {EXPECTED_CASE_CALLS} review requests, found {len(rows)}")
    custom_ids = [str(row.get("custom_id") or "") for row in rows]
    if len(set(custom_ids)) != EXPECTED_CASE_CALLS or any(not value for value in custom_ids):
        raise RuntimeError("review custom_id values must be non-empty and unique")
    prompt_hashes = {str(row.get("prompt_sha256") or "") for row in rows}
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise RuntimeError("all 150 cases must use exactly one non-empty prompt hash")
    models = {str((row.get("request") or {}).get("model") or "") for row in rows}
    if len(models) != 1 or "" in models:
        raise RuntimeError("prepared review requests must use exactly one model placeholder")
    for row in rows:
        messages = (row.get("request") or {}).get("messages") or []
        if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
            raise RuntimeError("each request must contain one system and one user message")
        try:
            semantic_payload = json.loads(str(messages[1].get("content") or ""))
        except json.JSONDecodeError as exc:
            raise RuntimeError("review user content must be a JSON payload") from exc
        errors = _walk_forbidden(semantic_payload)
        if errors:
            raise RuntimeError("semantic payload redaction failed: " + "; ".join(errors[:5]))
    return rows, next(iter(prompt_hashes)), next(iter(models))


def _calibration_cases() -> list[CalibrationCase]:
    """Synthetic cases test invariants without using suite questions or outcomes."""

    return [
        CalibrationCase(
            name="negative_numeric_identifier",
            payload={
                "query": "设备 ZX-2 是否允许单次输出超过 10 页？",
                "reference_answer": "ZX-2 不允许单次输出超过 10 页。",
                "unanswerable": False,
                "legacy_expected_terms": ["ZX-2", "不允许", "10 页"],
                "forbidden_terms": [],
                "branches": [{"branch_label": "B1", "query_span": "设备 ZX-2 是否允许单次输出超过 10 页"}],
                "evidence": [{"evidence_label": "E1", "page": 1, "text": "ZX-2 单次输出不得超过 10 页。"}],
                "deterministic_candidate": {},
            },
            expected={
                "answerability": "answer",
                "concept_terms": ["zx-2", "10"],
                "numeric": "10",
                "identifier": "zx-2",
                "polarity": "negative",
                "evidence_labels": ["E1"],
                "branch_labels": ["B1"],
            },
        ),
        CalibrationCase(
            name="two_branch_two_evidence",
            payload={
                "query": "默认接口是什么？最大容量是多少？",
                "reference_answer": "默认接口为 USB；最大容量为 500 页。",
                "unanswerable": False,
                "legacy_expected_terms": ["USB", "500 页"],
                "forbidden_terms": [],
                "branches": [
                    {"branch_label": "B1", "query_span": "默认接口是什么"},
                    {"branch_label": "B2", "query_span": "最大容量是多少"},
                ],
                "evidence": [
                    {"evidence_label": "E1", "page": 1, "text": "默认接口：USB。"},
                    {"evidence_label": "E2", "page": 2, "text": "最大容量：500 页。"},
                ],
                "deterministic_candidate": {},
            },
            expected={
                "answerability": "answer",
                "concept_terms": ["usb", "500"],
                "numeric": "500",
                "identifier": "usb",
                "polarity": None,
                "evidence_labels": ["E1", "E2"],
                "branch_labels": ["B1", "B2"],
            },
        ),
        CalibrationCase(
            name="unanswerable_no_fact",
            payload={
                "query": "这台设备今天在上海的现货价是多少？",
                "reference_answer": "手册未提供实时价格，无法依据当前知识库确定。",
                "unanswerable": True,
                "legacy_expected_terms": [],
                "forbidden_terms": [],
                "branches": [{"branch_label": "B1", "query_span": "这台设备今天在上海的现货价是多少"}],
                "evidence": [],
                "deterministic_candidate": {},
            },
            expected={
                "answerability": "abstain",
                "concept_terms": [],
                "numeric": None,
                "identifier": None,
                "polarity": None,
                "evidence_labels": [],
                "branch_labels": ["B1"],
                "max_concepts": 1,
                "calibration_projectable_abstention": True,
            },
        ),
        CalibrationCase(
            name="variant_group",
            payload={
                "query": "复位键会产生什么结果？",
                "reference_answer": "复位键用于恢复出厂设置。",
                "unanswerable": False,
                "legacy_expected_terms": ["恢复出厂设置", "恢复默认设置"],
                "forbidden_terms": [],
                "branches": [{"branch_label": "B1", "query_span": "复位键会产生什么结果"}],
                "evidence": [{"evidence_label": "E1", "page": 3, "text": "按复位键可恢复出厂设置（也称恢复默认设置）。"}],
                "deterministic_candidate": {},
            },
            expected={
                "answerability": "answer",
                "concept_terms": ["恢复出厂设置", "恢复默认设置"],
                "numeric": None,
                "identifier": None,
                "polarity": None,
                "evidence_labels": ["E1"],
                "branch_labels": ["B1"],
                "max_concepts": 1,
            },
        ),
    ]


def _calibration_execution_contract_sha256(
    template: dict[str, Any], *, model: str, endpoint: str
) -> str:
    request = template["request"]
    effective_request_config = {
        key: deepcopy(value)
        for key, value in request.items()
        if key not in {"messages", "model"}
    }
    return _sha256_json(
        {
            "runner_version": RUNNER_VERSION,
            "model": model,
            "endpoint": endpoint,
            "transport": TRANSPORT,
            "sse_completion_policy": SSE_COMPLETION_POLICY,
            "prompt_sha256": template["prompt_sha256"],
            "effective_request_config": effective_request_config,
            "calibration_cases_manifest_sha256": _sha256_json(
                [
                    {"name": case.name, "payload": case.payload, "expected": case.expected}
                    for case in _calibration_cases()
                ]
            ),
            "calibration_projection_policy": (
                CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY
            ),
        }
    )


def _calibration_requests(
    template: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str | None = None,
) -> list[dict[str, Any]]:
    system_message = template["request"]["messages"][0]
    canonical_endpoint = endpoint or _endpoint(DEFAULT_BASE_URL)
    execution_contract_sha256 = _calibration_execution_contract_sha256(
        template, model=model, endpoint=canonical_endpoint
    )
    requests: list[dict[str, Any]] = []
    for case in _calibration_cases():
        requests.append(
            {
                "custom_id": (
                    f"calibration-{RUNNER_VERSION}-{execution_contract_sha256[:16]}-"
                    f"{case.name}"
                ),
                "prompt_sha256": template["prompt_sha256"],
                "request": {
                    **dict(template["request"]),
                    "messages": [
                        dict(system_message),
                        {
                            "role": "user",
                            "content": json.dumps(case.payload, ensure_ascii=False, separators=(",", ":")),
                        },
                    ],
                },
                "expected": case.expected,
                "calibration_execution_contract_sha256": execution_contract_sha256,
                "calibration_model": model,
                "calibration_endpoint": canonical_endpoint,
            }
        )
    return requests


def _extract_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("review response must be one JSON object")
    return value


def _extract_complete_json(content: str) -> dict[str, Any]:
    """Parse one complete JSON object without the legacy substring recovery.

    A normal EOF without the SSE ``[DONE]`` sentinel is weaker transport proof,
    so trailing prose, a partial markdown fence, or any other unconsumed bytes
    must remain a hard failure. A complete fenced JSON object is still accepted
    because the review prompt permits JSON markdown formatting in practice.
    """

    stripped = content.strip()
    if stripped.startswith("```"):
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is None:
            raise ValueError("normal EOF response has an incomplete JSON code fence")
        stripped = fenced.group(1).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("review response must be one complete JSON object")
    return value


def _concept_key_normalized(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    match = re.fullmatch(r"c0*(\d+)", raw)
    return f"c{int(match.group(1))}" if match else raw


def _hydrate_approved_review(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Repair structure only when an approval explicitly affirms the candidate.

    Some JSON-mode responses state an approval and correctly reference every
    candidate concept from branches/coverage, but serialize only a prefix of the
    concept objects.  For an ``approve`` verdict only, missing structural fields
    can be restored deterministically from the reviewed candidate already in the
    same payload.  ``revise`` and ``block`` responses are never hydrated because
    their semantic changes must be fully emitted by the reviewer.
    """

    if value.get("verdict") != "approve":
        return value, []
    candidate = semantic_payload.get("deterministic_candidate") or {}
    candidate_concepts = (candidate.get("required_concepts") or {}).get("all_of") or []
    if not candidate_concepts:
        return value, []
    hydrated = deepcopy(value)
    concepts = (hydrated.setdefault("required_concepts", {})).setdefault("all_of", [])
    if not isinstance(concepts, list):
        return value, []

    candidate_by_key = {
        _concept_key_normalized(item.get("concept_id")): item
        for item in candidate_concepts
        if isinstance(item, dict) and item.get("concept_id")
    }
    evidence_coverage = candidate.get("evidence_concept_coverage") or []
    evidence_labels = [
        str(item.get("evidence_label") or "")
        for item in semantic_payload.get("evidence") or []
    ]
    raw_evidence_to_label = {
        str(item.get("evidence_id") or ""): evidence_labels[index]
        for index, item in enumerate(evidence_coverage)
        if index < len(evidence_labels) and item.get("evidence_id")
    }
    candidate_branches = candidate.get("required_branches") or []
    raw_branch_to_label = {
        str(item.get("branch_id") or ""): f"B{index + 1}"
        for index, item in enumerate(candidate_branches)
        if item.get("branch_id")
    }
    reviewed_by_key = {
        _concept_key_normalized(item.get("concept_key")): item
        for item in concepts
        if isinstance(item, dict) and item.get("concept_key")
    }
    referenced_keys = {
        _concept_key_normalized(key)
        for branch in hydrated.get("required_branches") or []
        for key in branch.get("required_concept_keys") or []
    }
    referenced_keys.update(
        _concept_key_normalized(key)
        for coverage in hydrated.get("evidence_concept_coverage") or []
        for key in coverage.get("required_concept_keys") or []
    )
    if not referenced_keys:
        referenced_keys = set(candidate_by_key)

    actions: list[str] = []
    for normalized_key in sorted(referenced_keys):
        candidate_concept = candidate_by_key.get(normalized_key)
        if candidate_concept is None:
            continue
        reviewer_concept = reviewed_by_key.get(normalized_key)
        if reviewer_concept is None:
            reviewer_concept = {"concept_key": str(candidate_concept["concept_id"])}
            concepts.append(reviewer_concept)
            reviewed_by_key[normalized_key] = reviewer_concept
            actions.append(f"added_missing_concept:{candidate_concept['concept_id']}")
        canonical = str(reviewer_concept.get("canonical_text") or "").strip()
        if len(re.sub(r"\s+", "", canonical)) < 2:
            reviewer_concept["canonical_text"] = str(candidate_concept.get("canonical_text") or "")
            actions.append(f"restored_canonical_text:{candidate_concept['concept_id']}")
        if not reviewer_concept.get("any_of"):
            reviewer_concept["any_of"] = [
                str(item) for item in candidate_concept.get("any_of") or []
            ]
            actions.append(f"restored_any_of:{candidate_concept['concept_id']}")
        if "evidence_labels" not in reviewer_concept:
            reviewer_concept["evidence_labels"] = sorted(
                {
                    raw_evidence_to_label[str(support.get("evidence_id") or "")]
                    for support in candidate_concept.get("evidence_support") or []
                    if str(support.get("evidence_id") or "") in raw_evidence_to_label
                }
            )
            actions.append(f"restored_evidence_labels:{candidate_concept['concept_id']}")
        if "branch_labels" not in reviewer_concept:
            reviewer_concept["branch_labels"] = [
                raw_branch_to_label[str(branch.get("branch_id") or "")]
                for branch in candidate_branches
                if str(branch.get("branch_id") or "") in raw_branch_to_label
                and candidate_concept["concept_id"] in (branch.get("required_concept_ids") or [])
            ]
            actions.append(f"restored_branch_labels:{candidate_concept['concept_id']}")
    concepts.sort(
        key=lambda item: _concept_key_normalized(item.get("concept_key"))
        if isinstance(item, dict)
        else ""
    )
    return hydrated, actions


def _normalize_review_structure(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Repair only fields already implied by the same reviewer response.

    An approval may reuse the deterministic candidate because the verdict
    explicitly affirms it.  A revise/block response never receives candidate
    semantics: a missing ``any_of`` may only reuse its own canonical text, and
    missing evidence/branch labels may only be reversed from that response's
    coverage and branch cross-references.
    """

    if value.get("verdict") == "approve":
        return _hydrate_approved_review(value, semantic_payload)
    if value.get("verdict") not in {"revise", "block"}:
        return value, []

    normalized = deepcopy(value)
    concepts = (normalized.get("required_concepts") or {}).get("all_of")
    if not isinstance(concepts, list):
        return value, []
    evidence_by_concept: dict[str, list[str]] = {}
    for coverage in normalized.get("evidence_concept_coverage") or []:
        if not isinstance(coverage, dict):
            continue
        if coverage.get("status") != "covered":
            continue
        label = str(coverage.get("evidence_label") or "")
        for raw_key in coverage.get("required_concept_keys") or []:
            key = _concept_key_normalized(raw_key)
            if key and label:
                evidence_by_concept.setdefault(key, []).append(label)
    branches_by_concept: dict[str, list[str]] = {}
    for branch in normalized.get("required_branches") or []:
        if not isinstance(branch, dict):
            continue
        label = str(branch.get("branch_label") or "")
        for raw_key in branch.get("required_concept_keys") or []:
            key = _concept_key_normalized(raw_key)
            if key and label:
                branches_by_concept.setdefault(key, []).append(label)

    actions: list[str] = []
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        key = _concept_key_normalized(concept.get("concept_key"))
        canonical = str(concept.get("canonical_text") or "").strip()
        if not key or not canonical:
            continue
        if not concept.get("any_of"):
            concept["any_of"] = [canonical]
            actions.append(f"self_filled_any_of:{key}")
        if "evidence_labels" not in concept:
            concept["evidence_labels"] = sorted(set(evidence_by_concept.get(key, [])))
            actions.append(f"crossref_filled_evidence_labels:{key}")
        if "branch_labels" not in concept:
            concept["branch_labels"] = sorted(set(branches_by_concept.get(key, [])))
            actions.append(f"crossref_filled_branch_labels:{key}")
    return normalized, actions


def _validate_against_payload(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected_answerability = "abstain" if semantic_payload.get("unanswerable") else "answer"
    if (
        value.get("answerability") != expected_answerability
        and value.get("verdict") != "block"
    ):
        errors.append(f"answerability conflicts with payload: expected {expected_answerability}")
    expected_branch_spans = {
        str(item.get("branch_label") or ""): str(item.get("query_span") or "")
        for item in semantic_payload.get("branches") or []
    }
    allowed_branches = set(expected_branch_spans)
    allowed_evidence = {
        str(item.get("evidence_label") or "")
        for item in semantic_payload.get("evidence") or []
    }
    concepts = (value.get("required_concepts") or {}).get("all_of") or []
    concept_keys = [
        _concept_key_normalized(item.get("concept_key"))
        for item in concepts
        if isinstance(item, dict)
    ]
    if len(concept_keys) != len(set(concept_keys)):
        errors.append("concept_key values must be unique")

    used_evidence: list[str] = []
    used_branches: list[str] = []
    concept_evidence_edges: set[tuple[str, str]] = set()
    concept_branch_edges: set[tuple[str, str]] = set()
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        concept_key = _concept_key_normalized(concept.get("concept_key"))
        for item in concept.get("evidence_labels") or []:
            label = str(item)
            used_evidence.append(label)
            concept_evidence_edges.add((concept_key, label))
        for item in concept.get("branch_labels") or []:
            label = str(item)
            used_branches.append(label)
            concept_branch_edges.add((concept_key, label))
    constraints = value.get("critical_constraints") or {}
    for item in [*(constraints.get("numerics") or []), *(constraints.get("identifiers") or [])]:
        if isinstance(item, dict):
            labels = [str(label) for label in item.get("evidence_labels") or []]
            used_evidence.extend(labels)
    actual_branch_spans: dict[str, str] = {}
    branch_edges: set[tuple[str, str]] = set()
    for branch in value.get("required_branches") or []:
        if not isinstance(branch, dict):
            continue
        branch_label = str(branch.get("branch_label") or "")
        used_branches.append(branch_label)
        actual_branch_spans[branch_label] = str(branch.get("query_span") or "")
        referenced_keys = {
            _concept_key_normalized(key)
            for key in branch.get("required_concept_keys") or []
        }
        unknown = referenced_keys - set(concept_keys)
        if unknown:
            errors.append(f"branch references unknown concept keys: {sorted(unknown)}")
        branch_edges.update((key, branch_label) for key in referenced_keys)
    actual_coverage_labels: set[str] = set()
    covered_evidence_labels: set[str] = set()
    coverage_edges: set[tuple[str, str]] = set()
    for coverage in value.get("evidence_concept_coverage") or []:
        if not isinstance(coverage, dict):
            continue
        evidence_label = str(coverage.get("evidence_label") or "")
        used_evidence.append(evidence_label)
        actual_coverage_labels.add(evidence_label)
        referenced_keys = {
            _concept_key_normalized(key)
            for key in coverage.get("required_concept_keys") or []
        }
        unknown = referenced_keys - set(concept_keys)
        if unknown:
            errors.append(f"coverage references unknown concept keys: {sorted(unknown)}")
        if coverage.get("status") == "covered":
            covered_evidence_labels.add(evidence_label)
            coverage_edges.update((key, evidence_label) for key in referenced_keys)
    unknown_evidence = sorted(set(used_evidence) - allowed_evidence)
    unknown_branches = sorted(set(used_branches) - allowed_branches)
    if unknown_evidence:
        errors.append(f"unknown evidence labels: {unknown_evidence}")
    if unknown_branches:
        errors.append(f"unknown branch labels: {unknown_branches}")
    for label, span in actual_branch_spans.items():
        if label in expected_branch_spans and span != expected_branch_spans[label]:
            errors.append(f"branch query_span differs from payload for {label}")
    if value.get("verdict") != "block":
        if actual_branch_spans != expected_branch_spans:
            errors.append("required branch labels/query spans must exactly match payload")
        if actual_coverage_labels != allowed_evidence:
            errors.append("evidence coverage labels must exactly match payload evidence")
        if value.get("answerability") == "answer" and not covered_evidence_labels:
            errors.append("answerable response must have at least one covered evidence item")
    if concept_branch_edges != branch_edges:
        errors.append("concept.branch_labels and required_branches edges must be symmetric")
    if concept_evidence_edges != coverage_edges:
        errors.append(
            "concept.evidence_labels and evidence_concept_coverage edges must be symmetric"
        )
    return errors


def _review_schema_extra_paths(value: dict[str, Any]) -> list[str]:
    """Return every schema-external object key without projecting its meaning."""

    paths: list[str] = []

    def record(obj: Any, allowed: set[str], path: str) -> None:
        if not isinstance(obj, dict):
            return
        paths.extend(f"{path}.{key}" for key in sorted(set(obj) - allowed))

    record(value, _REVIEW_ALLOWED_FIELDS, "$")
    required_concepts = value.get("required_concepts")
    record(required_concepts, _REQUIRED_CONCEPTS_ALLOWED_FIELDS, "$.required_concepts")
    if isinstance(required_concepts, dict):
        for index, concept in enumerate(required_concepts.get("all_of") or []):
            record(concept, _CONCEPT_ALLOWED_FIELDS, f"$.required_concepts.all_of[{index}]")

    constraints = value.get("critical_constraints")
    record(constraints, _CRITICAL_CONSTRAINTS_ALLOWED_FIELDS, "$.critical_constraints")
    if isinstance(constraints, dict):
        for index, numeric in enumerate(constraints.get("numerics") or []):
            record(numeric, _NUMERIC_ALLOWED_FIELDS, f"$.critical_constraints.numerics[{index}]")
        for index, identifier in enumerate(constraints.get("identifiers") or []):
            record(
                identifier,
                _IDENTIFIER_ALLOWED_FIELDS,
                f"$.critical_constraints.identifiers[{index}]",
            )
        record(
            constraints.get("polarity"),
            _POLARITY_ALLOWED_FIELDS,
            "$.critical_constraints.polarity",
        )
    for index, branch in enumerate(value.get("required_branches") or []):
        record(branch, _BRANCH_ALLOWED_FIELDS, f"$.required_branches[{index}]")
    for index, coverage in enumerate(value.get("evidence_concept_coverage") or []):
        record(
            coverage,
            _COVERAGE_ALLOWED_FIELDS,
            f"$.evidence_concept_coverage[{index}]",
        )
    return sorted(set(paths))


def _validate_review(
    value: dict[str, Any],
    *,
    allow_schema_extras: bool = False,
    abstain_cue_policy: str = "v2_pure_clause",
) -> list[str]:
    errors: list[str] = []
    extra_paths = _review_schema_extra_paths(value)
    if extra_paths and not allow_schema_extras:
        errors.append(f"review contains schema-external fields: {extra_paths}")

    def string_list_errors(
        raw: Any, path: str, *, require_nonempty: bool = False
    ) -> list[str]:
        nested_errors: list[str] = []
        if not isinstance(raw, list):
            return [f"{path} must be an array"]
        if require_nonempty and not raw:
            nested_errors.append(f"{path} must be non-empty")
        if any(not isinstance(item, str) or not item.strip() for item in raw):
            nested_errors.append(f"{path} entries must be non-blank strings")
        normalized = [str(item).strip().casefold() for item in raw]
        if len(normalized) != len(set(normalized)):
            nested_errors.append(f"{path} entries must be unique")
        return nested_errors

    if value.get("verdict") not in {"approve", "revise", "block"}:
        errors.append("verdict must be approve|revise|block")
    if value.get("answerability") not in {"answer", "abstain"}:
        errors.append("answerability must be answer|abstain")
    required_concepts = value.get("required_concepts")
    if not isinstance(required_concepts, dict):
        errors.append("required_concepts must be an object")
        required_concepts = {}
    concepts = required_concepts.get("all_of")
    if not isinstance(concepts, list):
        errors.append("required_concepts.all_of must be an array")
        concepts = []
    concept_keys: list[str] = []
    for index, concept in enumerate(concepts):
        if not isinstance(concept, dict):
            errors.append(f"concept[{index}] must be an object")
            continue
        for key in ("concept_key", "canonical_text", "any_of", "evidence_labels", "branch_labels"):
            if key not in concept:
                errors.append(f"concept[{index}] missing {key}")
        concept_key = concept.get("concept_key")
        canonical_text = concept.get("canonical_text")
        if not isinstance(concept_key, str) or not concept_key.strip():
            errors.append(f"concept[{index}].concept_key must be a non-blank string")
        else:
            concept_keys.append(_concept_key_normalized(concept_key))
        if not isinstance(canonical_text, str) or not canonical_text.strip():
            errors.append(f"concept[{index}].canonical_text must be a non-blank string")
        errors.extend(
            string_list_errors(
                concept.get("any_of"), f"concept[{index}].any_of", require_nonempty=True
            )
        )
        errors.extend(
            string_list_errors(
                concept.get("evidence_labels"), f"concept[{index}].evidence_labels"
                , require_nonempty=(
                    value.get("verdict") != "block"
                    and value.get("answerability") == "answer"
                )
            )
        )
        errors.extend(
            string_list_errors(
                concept.get("branch_labels"),
                f"concept[{index}].branch_labels",
                require_nonempty=value.get("verdict") != "block",
            )
        )
    if len(concept_keys) != len(set(concept_keys)):
        errors.append("concept_key values must be non-empty and unique")

    constraints = value.get("critical_constraints")
    if not isinstance(constraints, dict):
        errors.append("critical_constraints must be an object")
        constraints = {}
    else:
        for key in ("numerics", "identifiers", "polarity"):
            if key not in constraints:
                errors.append(f"critical_constraints missing {key}")
    numerics = constraints.get("numerics")
    identifiers = constraints.get("identifiers")
    if not isinstance(numerics, list):
        errors.append("critical_constraints.numerics must be an array")
        numerics = []
    if not isinstance(identifiers, list):
        errors.append("critical_constraints.identifiers must be an array")
        identifiers = []
    for kind, items in (("numerics", numerics), ("identifiers", identifiers)):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"critical_constraints.{kind}[{index}] must be an object")
                continue
            if not str(item.get("value") or "").strip():
                errors.append(
                    f"critical_constraints.{kind}[{index}].value must be non-blank"
                )
            unit = item.get("unit")
            if kind == "numerics" and "unit" not in item:
                errors.append(f"critical_constraints.numerics[{index}] missing unit")
            if kind == "numerics" and not (
                unit is None or isinstance(unit, str) and unit.strip()
            ):
                errors.append(
                    f"critical_constraints.numerics[{index}].unit must be null or a "
                    "non-blank string"
                )
            errors.extend(
                string_list_errors(
                    item.get("evidence_labels"),
                    f"critical_constraints.{kind}[{index}].evidence_labels",
                    require_nonempty=value.get("answerability") == "answer",
                )
            )
    polarity = constraints.get("polarity")
    if not isinstance(polarity, dict):
        errors.append("critical_constraints.polarity must be an object")
    else:
        if polarity.get("required") not in _POLARITY_REQUIRED_VALUES:
            errors.append(
                "critical_constraints.polarity.required must be one of the frozen "
                f"polarity values: {sorted(_POLARITY_REQUIRED_VALUES)}"
            )
        errors.extend(
            string_list_errors(
                polarity.get("cues"), "critical_constraints.polarity.cues"
            )
        )

    branches = value.get("required_branches")
    if not isinstance(branches, list):
        errors.append("required_branches must be an array")
        branches = []
    branch_labels: list[str] = []
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            errors.append(f"required_branches[{index}] must be an object")
            continue
        label = branch.get("branch_label")
        query_span = branch.get("query_span")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"required_branches[{index}].branch_label must be non-blank")
        else:
            branch_labels.append(label.strip().casefold())
        if not isinstance(query_span, str) or not query_span.strip():
            errors.append(f"required_branches[{index}].query_span must be non-blank")
        errors.extend(
            string_list_errors(
                branch.get("required_concept_keys"),
                f"required_branches[{index}].required_concept_keys",
                require_nonempty=value.get("verdict") != "block",
            )
        )
    if len(branch_labels) != len(set(branch_labels)):
        errors.append("required_branches.branch_label values must be unique")

    coverage = value.get("evidence_concept_coverage")
    if not isinstance(coverage, list):
        errors.append("evidence_concept_coverage must be an array")
        coverage = []
    evidence_labels: list[str] = []
    block_statuses = {"covered", "uncovered", "insufficient", "ambiguous"}
    for index, item in enumerate(coverage):
        if not isinstance(item, dict):
            errors.append(f"evidence_concept_coverage[{index}] must be an object")
            continue
        label = item.get("evidence_label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"evidence_concept_coverage[{index}].evidence_label must be non-blank")
        else:
            evidence_labels.append(label.strip().casefold())
        status = item.get("status")
        allowed_statuses = (
            block_statuses
            if value.get("verdict") == "block"
            else {"uncovered", "insufficient"}
            if value.get("answerability") == "abstain"
            else {"covered", "uncovered", "insufficient"}
        )
        errors.extend(
            string_list_errors(
                item.get("required_concept_keys"),
                f"evidence_concept_coverage[{index}].required_concept_keys",
                require_nonempty=(
                    value.get("verdict") != "block" and status == "covered"
                ),
            )
        )
        if status not in allowed_statuses:
            errors.append(
                f"evidence_concept_coverage[{index}].status must be one of "
                f"{sorted(allowed_statuses)}"
            )
        if (
            value.get("verdict") != "block"
            and status in {"uncovered", "insufficient"}
            and item.get("required_concept_keys")
        ):
            errors.append(
                f"evidence_concept_coverage[{index}].required_concept_keys must be empty "
                f"when status={status}"
            )
    if len(evidence_labels) != len(set(evidence_labels)):
        errors.append("evidence_concept_coverage.evidence_label values must be unique")
    if (
        value.get("verdict") != "block"
        and value.get("answerability") == "answer"
        and not any(
            isinstance(item, dict) and item.get("status") == "covered"
            for item in coverage
        )
    ):
        errors.append("answerable response must have at least one covered evidence item")

    errors.extend(
        string_list_errors(value.get("review_notes"), "review_notes", require_nonempty=True)
    )
    if value.get("answerability") == "answer" and not concepts and value.get("verdict") != "block":
        errors.append("answerable non-blocked response must have at least one concept")
    if value.get("answerability") == "abstain" and concepts:
        # A reviewer may encode the refusal rationale (for example, "the manual
        # does not provide a real-time price") as one contract concept. That is
        # not an answered fact. Accept only an evidence-free absence/uncertainty
        # constraint; any affirmative factual concept remains a hard failure.
        if len(concepts) > 1:
            errors.append("abstain response may contain at most one refusal-rationale concept")
        for index, concept in enumerate(concepts):
            refusal_variants = [
                str(concept.get("canonical_text") or ""),
                *(str(item or "") for item in concept.get("any_of") or []),
            ]
            def cue_valid(text: str) -> bool:
                return _abstention_variant_is_refusal(
                    text, policy=abstain_cue_policy
                )

            exact_v2_sentinel = (
                concept.get("canonical_text") == ABSTENTION_SENTINEL_CANONICAL_TEXT
                and concept.get("any_of") == list(ABSTENTION_SENTINEL_VARIANTS)
                and (constraints or {}).get("polarity")
                == ABSTENTION_SENTINEL_POLARITY
            )
            if (
                abstain_cue_policy == "v2_pure_clause"
                and not exact_v2_sentinel
            ) or any(not cue_valid(text) for text in refusal_variants):
                errors.append(
                    f"abstain concept[{index}] variants must each express only an explicit "
                    "absence/uncertainty rationale"
                )
            if concept.get("evidence_labels"):
                errors.append(f"abstain concept[{index}] must not claim direct evidence support")
    if value.get("answerability") == "abstain" and isinstance(constraints, dict):
        if constraints.get("numerics") or constraints.get("identifiers"):
            errors.append("abstain response must not require factual numerics or identifiers")
    return errors


def _flatten_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()


def _validate_calibration(value: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors = _validate_review(value)
    if value.get("answerability") != expected["answerability"]:
        errors.append(f"expected answerability={expected['answerability']}")
    concepts = (value.get("required_concepts") or {}).get("all_of") or []
    concept_text = _flatten_text(concepts)
    for term in expected.get("concept_terms") or []:
        if str(term).casefold() not in concept_text:
            errors.append(f"missing calibration concept term: {term}")
    max_concepts = expected.get("max_concepts")
    if max_concepts is not None and len(concepts) > int(max_concepts):
        errors.append(f"expected at most {max_concepts} concept(s), got {len(concepts)}")
    constraints = value.get("critical_constraints") or {}
    if expected.get("numeric") is not None and str(expected["numeric"]) not in _flatten_text(constraints.get("numerics")):
        errors.append(f"missing calibration numeric: {expected['numeric']}")
    if expected.get("identifier") is not None and str(expected["identifier"]).casefold() not in _flatten_text(constraints.get("identifiers")):
        errors.append(f"missing calibration identifier: {expected['identifier']}")
    if expected.get("polarity") is not None and str(expected["polarity"]).casefold() not in _flatten_text(constraints.get("polarity")):
        errors.append(f"missing calibration polarity: {expected['polarity']}")
    result_text = _flatten_text(value)
    for label in expected.get("evidence_labels") or []:
        if str(label).casefold() not in result_text:
            errors.append(f"missing calibration evidence label: {label}")
    for label in expected.get("branch_labels") or []:
        if str(label).casefold() not in result_text:
            errors.append(f"missing calibration branch label: {label}")
    return errors


_ABSTAIN_SENTINEL_VARIANT_ERROR = (
    "abstain concept[0] variants must each express only an explicit "
    "absence/uncertainty rationale"
)


def _calibration_manifest_sha256(
    row: dict[str, Any], semantic_payload: dict[str, Any]
) -> str:
    return _sha256_json(
        {
            "custom_id": row.get("custom_id"),
            "prompt_sha256": row.get("prompt_sha256"),
            "expected": row.get("expected"),
            "calibration_execution_contract_sha256": row.get(
                "calibration_execution_contract_sha256"
            ),
            "calibration_model": row.get("calibration_model"),
            "calibration_endpoint": row.get("calibration_endpoint"),
            "semantic_payload": semantic_payload,
        }
    )


def _project_fresh_calibration_abstention(
    value: dict[str, Any],
    semantic_payload: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Project only the synthetic unanswerable calibration response.

    This is not a formal-response recovery API.  It accepts one fresh model
    response whose sole defect is the frozen sentinel representation, then
    proves that a copy-on-write projection is strict and payload aligned.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    concepts = (raw.get("required_concepts") or {}).get("all_of")
    constraints = raw.get("critical_constraints")
    branches = raw.get("required_branches")
    if (
        expected.get("calibration_projectable_abstention") is not True
        or expected.get("answerability") != "abstain"
        or semantic_payload.get("unanswerable") is not True
        or semantic_payload.get("evidence") != []
        or raw_errors != [_ABSTAIN_SENTINEL_VARIANT_ERROR]
        or raw.get("answerability") != "abstain"
        or raw.get("verdict") not in {"approve", "revise", "block"}
        or _review_schema_extra_paths(raw)
        or not isinstance(concepts, list)
        or len(concepts) != 1
        or not isinstance(concepts[0], dict)
        or concepts[0].get("evidence_labels") != []
        or raw.get("evidence_concept_coverage") != []
        or not isinstance(branches, list)
        or not branches
        or any(
            not isinstance(branch, dict)
            or not branch.get("required_concept_keys")
            for branch in branches
        )
        or not isinstance(constraints, dict)
        or constraints.get("numerics") != []
        or constraints.get("identifiers") != []
        or not isinstance(constraints.get("polarity"), dict)
    ):
        return raw, [], raw_errors

    projected = deepcopy(raw)
    concept = projected["required_concepts"]["all_of"][0]
    before = {
        "canonical_text": concept.get("canonical_text"),
        "any_of": deepcopy(concept.get("any_of")),
        "polarity": deepcopy(constraints.get("polarity")),
    }
    concept["canonical_text"] = ABSTENTION_SENTINEL_CANONICAL_TEXT
    concept["any_of"] = list(ABSTENTION_SENTINEL_VARIANTS)
    projected["critical_constraints"]["polarity"] = deepcopy(
        ABSTENTION_SENTINEL_POLARITY
    )
    final_errors = _validate_calibration(projected, expected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(
        _review_materialization_errors(projected, semantic_payload)
    )
    if final_errors:
        return raw, [], raw_errors
    unchanged = deepcopy(projected)
    unchanged_concept = unchanged["required_concepts"]["all_of"][0]
    unchanged_concept["canonical_text"] = before["canonical_text"]
    unchanged_concept["any_of"] = deepcopy(before["any_of"])
    unchanged["critical_constraints"]["polarity"] = deepcopy(before["polarity"])
    if unchanged != raw:
        return raw, [], raw_errors
    action = {
        "operation": "replace",
        "json_paths": [
            "/required_concepts/all_of/0/canonical_text",
            "/required_concepts/all_of/0/any_of",
            "/critical_constraints/polarity",
        ],
        "before_sha256": _sha256_json(before),
        "after": {
            "canonical_text": ABSTENTION_SENTINEL_CANONICAL_TEXT,
            "any_of": list(ABSTENTION_SENTINEL_VARIANTS),
            "polarity": deepcopy(ABSTENTION_SENTINEL_POLARITY),
        },
        "authority": "synthetic_unanswerable_calibration_representation_only",
    }
    return projected, [action], raw_errors


def _dedupe_same_response_exact_variants(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Remove only normalized-exact duplicate ``any_of`` strings.

    The first occurrence is retained.  Case or surrounding whitespace may
    differ, but punctuation and all other content remain significant.  The raw
    response must have no defect other than the strict duplicate error.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    duplicate_error = re.compile(r"^concept\[\d+\]\.any_of entries must be unique$")
    if not raw_errors or any(not duplicate_error.fullmatch(error) for error in raw_errors):
        return raw, [], raw_errors

    projected = deepcopy(raw)
    actions: list[dict[str, Any]] = []
    for concept in (projected.get("required_concepts") or {}).get("all_of") or []:
        before = list(concept["any_of"])
        after: list[str] = []
        first_index_by_normalized: dict[str, int] = {}
        removed: list[dict[str, Any]] = []
        for index, variant in enumerate(before):
            normalized = str(variant).strip().casefold()
            retained_index = first_index_by_normalized.get(normalized)
            if retained_index is not None:
                removed.append(
                    {
                        "index": index,
                        "value": variant,
                        "normalized": normalized,
                        "retained_index": retained_index,
                        "retained_value": before[retained_index],
                    }
                )
                continue
            first_index_by_normalized[normalized] = index
            after.append(variant)
        if not removed:
            continue
        concept["any_of"] = after
        actions.append(
            {
                "concept_key": str(concept["concept_key"]),
                "field": "any_of",
                "before": before,
                "after": after,
                "removed": removed,
                "authority": "normalized_exact_duplicate_first_occurrence",
            }
        )
    projected_errors = _validate_review(projected)
    projected_errors.extend(_validate_against_payload(projected, semantic_payload))
    if not actions or projected_errors:
        return deepcopy(value), [], raw_errors
    return projected, actions, raw_errors


def _canonicalize_same_response_coverage_statuses(
    value: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    permitted_additional_errors: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Map a finite set of zero-support status aliases to ``uncovered``.

    The frozen blind-review prompt required a status field but did not prescribe
    its enum.  Only the frozen exact alias allowlist is accepted, and only for
    an evidence row with no direct full-concept support edge.  Critical
    constraints carry their own evidence labels, so a constraint-only edge is
    intentionally independent.  No labels, keys, concepts, constraints, or
    semantic text are changed.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    status_error = re.compile(
        r"^evidence_concept_coverage\[\d+\]\.status must be one of \[.*\]$"
    )
    residual_raw_errors = [
        error for error in raw_errors if not status_error.fullmatch(error)
    ]
    if (
        not raw_errors
        or not any(status_error.fullmatch(error) for error in raw_errors)
        or any(error not in permitted_additional_errors for error in residual_raw_errors)
        or raw.get("answerability") != "answer"
    ):
        return raw, [], raw_errors

    concepts = (raw.get("required_concepts") or {}).get("all_of") or []
    concept_keys = {
        _concept_key_normalized(item.get("concept_key"))
        for item in concepts
        if isinstance(item, dict)
    }
    if not concept_keys or len(concept_keys) != len(concepts) or "" in concept_keys:
        return raw, [], raw_errors

    concept_evidence_edges = {
        str(label)
        for concept in concepts
        for label in concept.get("evidence_labels") or []
    }
    coverage = raw.get("evidence_concept_coverage") or []
    covered_support_by_key = {key: set() for key in concept_keys}
    canonicalizable_indexes: list[int] = []
    for index, item in enumerate(coverage):
        if not isinstance(item, dict):
            return raw, [], raw_errors
        label = str(item.get("evidence_label") or "")
        status = item.get("status")
        keys = [
            _concept_key_normalized(key)
            for key in item.get("required_concept_keys") or []
        ]
        if any(key not in concept_keys for key in keys):
            return raw, [], raw_errors
        if status == "covered":
            if not keys:
                return raw, [], raw_errors
            for key in keys:
                covered_support_by_key[key].add(label)
        elif status in {"uncovered", "insufficient"}:
            if keys:
                return raw, [], raw_errors
        elif status in COVERAGE_STATUS_ZERO_SUPPORT_ALIASES:
            if keys or label in concept_evidence_edges:
                return raw, [], raw_errors
            canonicalizable_indexes.append(index)
        else:
            return raw, [], raw_errors
    if (
        not canonicalizable_indexes
        or (
            raw.get("verdict") != "block"
            and any(not labels for labels in covered_support_by_key.values())
        )
    ):
        return raw, [], raw_errors

    projected = deepcopy(raw)
    actions: list[dict[str, Any]] = []
    for index in canonicalizable_indexes:
        item = projected["evidence_concept_coverage"][index]
        before = str(item["status"])
        item["status"] = "uncovered"
        actions.append(
            {
                "coverage_index": index,
                "evidence_label": str(item["evidence_label"]),
                "field": "status",
                "before": before,
                "after": "uncovered",
                "required_concept_keys": [],
                "authority": "zero_direct_concept_support_edges",
            }
        )
    projected_errors = _validate_review(projected)
    projected_errors.extend(_validate_against_payload(projected, semantic_payload))
    if projected_errors != residual_raw_errors:
        return deepcopy(value), [], raw_errors
    return projected, actions, raw_errors


def _canonicalize_same_response_unsupported_required_concept_block(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Promote a structurally proven unsupported required branch to data block.

    This projection changes only ``verdict=revise`` to ``verdict=block``.  The
    unsupported concept, branch, variants, notes, and empty evidence edge are
    retained verbatim so the result can only serve as a remediation gate, never
    as a runnable answer contract.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    missing_support_error = re.compile(
        r"^concept\[(\d+)\]\.evidence_labels must be non-empty$"
    )
    matches = [missing_support_error.fullmatch(error) for error in raw_errors]
    if (
        not raw_errors
        or any(match is None for match in matches)
        or raw.get("verdict") != "revise"
        or raw.get("answerability") != "answer"
    ):
        return raw, [], raw_errors

    concepts = (raw.get("required_concepts") or {}).get("all_of") or []
    unsupported_indexes = {int(match.group(1)) for match in matches if match is not None}
    if not unsupported_indexes or any(index >= len(concepts) for index in unsupported_indexes):
        return raw, [], raw_errors
    actual_empty_indexes = {
        index
        for index, concept in enumerate(concepts)
        if isinstance(concept, dict) and concept.get("evidence_labels") == []
    }
    if unsupported_indexes != actual_empty_indexes:
        return raw, [], raw_errors

    concept_raw_keys = [
        str(concept.get("concept_key") or "")
        for concept in concepts
        if isinstance(concept, dict)
    ]
    concept_keys = [_concept_key_normalized(key) for key in concept_raw_keys]
    if len(concept_keys) != len(concepts) or len(concept_keys) != len(set(concept_keys)):
        return raw, [], raw_errors
    unsupported_keys = {concept_keys[index] for index in unsupported_indexes}
    if not unsupported_keys or "" in unsupported_keys:
        return raw, [], raw_errors

    constraints = raw.get("critical_constraints") or {}
    supported_constraint_facts = [
        " ".join(
            part
            for part in (
                str(item.get("value") or "").strip(),
                str(item.get("unit") or "").strip(),
            )
            if part
        )
        for item in constraints.get("numerics") or []
        if item.get("evidence_labels") and str(item.get("value") or "").strip()
    ]
    supported_constraint_facts.extend(
        str(item.get("value") or "").strip()
        for item in constraints.get("identifiers") or []
        if item.get("evidence_labels") and str(item.get("value") or "").strip()
    )
    for index in unsupported_indexes:
        concept = concepts[index]
        variants = [
            str(concept.get("canonical_text") or "").strip(),
            *(str(value or "").strip() for value in concept.get("any_of") or []),
        ]
        if any(
            concept_variant_covered(fact, variant)
            for fact in supported_constraint_facts
            for variant in variants
            if variant
        ):
            return raw, [], raw_errors

    referenced_unsupported: set[str] = set()
    unsupported_branch_labels: list[str] = []
    for branch in raw.get("required_branches") or []:
        keys = {
            _concept_key_normalized(key)
            for key in branch.get("required_concept_keys") or []
        }
        intersection = keys & unsupported_keys
        if not intersection:
            continue
        referenced_unsupported.update(intersection)
        unsupported_branch_labels.append(str(branch.get("branch_label") or ""))
    if referenced_unsupported != unsupported_keys or not unsupported_branch_labels:
        return raw, [], raw_errors

    for coverage in raw.get("evidence_concept_coverage") or []:
        keys = {
            _concept_key_normalized(key)
            for key in coverage.get("required_concept_keys") or []
        }
        if keys & unsupported_keys:
            return raw, [], raw_errors

    projected = deepcopy(raw)
    projected["verdict"] = "block"
    actions = [
        {
            "field": "verdict",
            "before": "revise",
            "after": "block",
            "unsupported_concept_keys": [
                concept_raw_keys[index] for index in sorted(unsupported_indexes)
            ],
            "unsupported_branch_labels": unsupported_branch_labels,
            "authority": "required_branch_has_no_directly_supported_concept",
        }
    ]
    projected_errors = _validate_review(projected)
    projected_errors.extend(_validate_against_payload(projected, semantic_payload))
    if projected_errors:
        return deepcopy(value), [], raw_errors
    return projected, actions, raw_errors


_ANSWER_SEEKING_BRANCH_RE = re.compile(
    r"[?？]|(?:如何|怎样|怎么|为什么|为何|何时|哪里|哪些|什么|是否|能否|可否|"
    r"多少|有何|何种|哪一|哪个)|(?:^|[，,：:；;])\s*(?:请|请问|说明|列出|给出|"
    r"判断|比较|解释|描述|指出|回答|确认|分析|计算)|"
    r"\b(?:how|why|when|where|which|what|whether|can|could|should)\b|"
    r"^\s*(?:please|explain|list|state|identify|compare|determine|describe|"
    r"calculate|answer|provide)\b",
    re.IGNORECASE,
)
_CONTEXT_PREAMBLE_BRANCH_RE = re.compile(
    r"(?:中|里|内)(?:提到|已说明|说明了|指出|显示|记载)|"
    r"^(?:已知|假设|背景|前提)(?:是|为|：|:|，|,|\s)|"
    r"\b(?:given that|assuming that|as context|as background|"
    r"the (?:question|prompt|manual|document) (?:states|mentions|notes|shows))\b",
    re.IGNORECASE,
)
_CONTEXT_PREAMBLE_NOTE_RE = re.compile(
    r"context(?:ual)?\s+(?:preamble|premise|background)|"
    r"not\s+(?:a\s+)?question|requires?\s+no\s+(?:answer|concept)|"
    r"无需(?:回答|作答)|不(?:是|属于)(?:问题|问句)|"
    r"(?:上下文|背景)(?:前提|陈述|信息)|陈述性前提",
    re.IGNORECASE,
)


def _is_context_preamble_branch(
    query_span: str, review_notes: Sequence[Any]
) -> bool:
    """Conservatively identify a declarative query preamble, never a request.

    Notes are corroborating audit evidence only.  The branch text must itself
    contain a context marker and must not contain a question or imperative cue.
    """

    span = str(query_span or "").strip()
    if (
        not span
        or _ANSWER_SEEKING_BRANCH_RE.search(span)
        or not _CONTEXT_PREAMBLE_BRANCH_RE.search(span)
    ):
        return False
    return any(
        _CONTEXT_PREAMBLE_NOTE_RE.search(str(note or ""))
        for note in review_notes
    )


def _canonicalize_same_response_empty_required_branch_block(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Project a proven context-preamble branch defect to a data-only block.

    This pure projector never authorizes live single-response adoption.  Its
    caller must additionally prove semantic-quota exhaustion and multi-response
    structural consensus before persisting the projected block.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    empty_branch_error = re.compile(
        r"^required_branches\[(\d+)\]\.required_concept_keys must be non-empty$"
    )
    matches = [empty_branch_error.fullmatch(error) for error in raw_errors]
    if (
        not raw_errors
        or any(match is None for match in matches)
        or raw.get("verdict") != "revise"
        or raw.get("answerability") != "answer"
    ):
        return raw, [], raw_errors

    branches = raw.get("required_branches") or []
    empty_indexes = {int(match.group(1)) for match in matches if match is not None}
    actual_empty_indexes = {
        index
        for index, branch in enumerate(branches)
        if isinstance(branch, dict) and branch.get("required_concept_keys") == []
    }
    if (
        not empty_indexes
        or empty_indexes != actual_empty_indexes
        or any(index >= len(branches) for index in empty_indexes)
        or not any(
            isinstance(branch, dict) and branch.get("required_concept_keys")
            for branch in branches
        )
    ):
        return raw, [], raw_errors

    notes = raw.get("review_notes") or []
    empty_labels: list[str] = []
    empty_spans: list[str] = []
    for index in sorted(empty_indexes):
        branch = branches[index]
        label = str(branch.get("branch_label") or "")
        span = str(branch.get("query_span") or "")
        if not label or not _is_context_preamble_branch(span, notes):
            return raw, [], raw_errors
        empty_labels.append(label)
        empty_spans.append(span)

    if any(
        set(str(label) for label in concept.get("branch_labels") or [])
        & set(empty_labels)
        for concept in (raw.get("required_concepts") or {}).get("all_of") or []
        if isinstance(concept, dict)
    ):
        return raw, [], raw_errors

    # Constraint rows do not carry branch labels.  Requiring factual constraint
    # arrays to be empty is the only fail-closed proof that an empty branch does
    # not hide a numeric or identifier obligation.
    constraints = raw.get("critical_constraints") or {}
    if constraints.get("numerics") or constraints.get("identifiers"):
        return raw, [], raw_errors

    projected = deepcopy(raw)
    projected["verdict"] = "block"
    actions = [
        {
            "field": "verdict",
            "before": "revise",
            "after": "block",
            "empty_branch_labels": empty_labels,
            "empty_branch_query_spans": empty_spans,
            "authority": "context_preamble_has_no_required_answer_concept",
        }
    ]
    projected_errors = _validate_review(projected)
    projected_errors.extend(_validate_against_payload(projected, semantic_payload))
    if projected_errors:
        return deepcopy(value), [], raw_errors
    return projected, actions, raw_errors


def _verdict_block_structural_signature(review: dict[str, Any]) -> dict[str, Any]:
    """Return only topology needed to prove cross-response blocker consensus."""

    concepts = (review.get("required_concepts") or {}).get("all_of") or []
    constraints = review.get("critical_constraints") or {}
    return {
        "verdict": review.get("verdict"),
        "answerability": review.get("answerability"),
        "concept_edges": [
            {
                "concept_key": _concept_key_normalized(concept.get("concept_key")),
                "evidence_labels": sorted(
                    str(label) for label in concept.get("evidence_labels") or []
                ),
                "branch_labels": sorted(
                    str(label) for label in concept.get("branch_labels") or []
                ),
            }
            for concept in concepts
            if isinstance(concept, dict)
        ],
        "required_branches": [
            {
                "branch_label": str(branch.get("branch_label") or ""),
                "query_span": str(branch.get("query_span") or ""),
                "required_concept_keys": sorted(
                    _concept_key_normalized(key)
                    for key in branch.get("required_concept_keys") or []
                ),
            }
            for branch in review.get("required_branches") or []
            if isinstance(branch, dict)
        ],
        "evidence_concept_coverage": [
            {
                "evidence_label": str(coverage.get("evidence_label") or ""),
                "status": str(coverage.get("status") or ""),
                "required_concept_keys": sorted(
                    _concept_key_normalized(key)
                    for key in coverage.get("required_concept_keys") or []
                ),
            }
            for coverage in review.get("evidence_concept_coverage") or []
            if isinstance(coverage, dict)
        ],
        "critical_constraints": {
            "numerics": _numeric_constraint_consensus_signature(
                constraints.get("numerics") or []
            ),
            "identifiers": constraints.get("identifiers") or [],
            "polarity_required": (constraints.get("polarity") or {}).get("required"),
        },
    }


_CLOSED_NUMERIC_RANGE_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*(?:~|～|至)\s*"
    r"([+-]?\d+(?:[.,]\d+)?)\s*$"
)


def _numeric_constraint_consensus_signature(
    constraints: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize only explicit closed numeric ranges for consensus comparison.

    ``a~b``/``a至b`` and two separately listed endpoints are equivalent only
    when unit and evidence-label bindings are identical.  Open ranges and all
    other value syntax remain verbatim, so they cannot gain equivalence.
    """

    atoms: list[dict[str, Any]] = []
    for item in constraints:
        value = str(item.get("value") or "").strip()
        unit = re.sub(r"\s+", " ", str(item.get("unit") or "").strip()).casefold()
        evidence_labels = sorted(
            str(label) for label in item.get("evidence_labels") or []
        )
        match = _CLOSED_NUMERIC_RANGE_RE.fullmatch(value)
        values = sorted(match.groups()) if match is not None else [value]
        atoms.extend(
            {
                "value": endpoint,
                "unit": unit,
                "evidence_labels": evidence_labels,
            }
            for endpoint in values
        )
    return sorted(
        atoms,
        key=lambda item: (
            item["unit"],
            tuple(item["evidence_labels"]),
            item["value"],
        ),
    )


def _unsupported_target_semantic_signature(
    review: dict[str, Any], actions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Bind unsupported target text/topology across every consensus response."""

    if len(actions) != 1 or "unsupported_concept_keys" not in actions[0]:
        return None
    target_keys = {
        _concept_key_normalized(value)
        for value in actions[0].get("unsupported_concept_keys") or []
    }
    concepts = (review.get("required_concepts") or {}).get("all_of") or []
    targets = [
        {
            "concept_key": _concept_key_normalized(concept.get("concept_key")),
            "canonical_text": str(concept.get("canonical_text") or ""),
            "any_of": [str(value) for value in concept.get("any_of") or []],
            "branch_labels": sorted(
                str(value) for value in concept.get("branch_labels") or []
            ),
        }
        for concept in concepts
        if isinstance(concept, dict)
        and _concept_key_normalized(concept.get("concept_key")) in target_keys
    ]
    if len(targets) != len(target_keys):
        return []
    return sorted(targets, key=lambda item: item["concept_key"])


def _constraint_evidence_labels(review: dict[str, Any]) -> set[str]:
    constraints = review.get("critical_constraints") or {}
    return {
        str(label)
        for item in [
            *(constraints.get("numerics") or []),
            *(constraints.get("identifiers") or []),
        ]
        if isinstance(item, dict)
        for label in item.get("evidence_labels") or []
    }


def _invalid_attempt_has_structural_normalization(item: dict[str, Any]) -> bool:
    provenance = item.get("normalization_provenance")
    if isinstance(provenance, dict) and provenance.get("actions"):
        return True
    return any(
        str(error).startswith("strict runner rejects structural normalization:")
        for error in item.get("validation_errors") or []
    )


def _validate_invalid_attempt_raw_normalization_provenance(
    item: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    extra_review_validator: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ]
    | None = None,
) -> list[str]:
    """Replay the raw-to-normalized receipt for current-runner invalid attempts.

    Pre-2.23 append-only rows did not preserve the raw response, so their
    terminal integrity remains verifiable under the legacy hash rule.  They are
    never made projection-eligible when their validation receipt records a
    structural normalization.
    """

    runner_version = str(source_row.get("runner_version") or "")
    provenance_fields = {
        "raw_review",
        "raw_review_sha256",
        "terminal_content_sha256",
        "pre_normalization_review_sha256",
        "normalization_provenance",
    }
    has_provenance = any(field in item for field in provenance_fields)
    if runner_version in PRE_RAW_PROVENANCE_RUNNER_VERSIONS and not has_provenance:
        return []
    if (
        runner_version not in RAW_PROVENANCE_REQUIRED_RUNNER_VERSIONS
        and not has_provenance
    ):
        return [
            "invalid attempt runner is not authorized to omit raw normalization "
            "provenance"
        ]

    errors: list[str] = []
    raw_review = item.get("raw_review")
    normalized_review = item.get("review")
    provenance = item.get("normalization_provenance")
    if not isinstance(raw_review, dict):
        errors.append("invalid attempt raw review must be an object")
        return errors
    if not isinstance(normalized_review, dict):
        errors.append("invalid attempt normalized review must be an object")
        return errors
    if not isinstance(provenance, dict):
        errors.append("invalid attempt normalization provenance must be an object")
        return errors

    expected_provenance_keys = {
        "policy",
        "raw_review_sha256",
        "pre_normalization_review_sha256",
        "normalized_review_sha256",
        "actions",
    }
    if set(provenance) != expected_provenance_keys:
        errors.append("invalid attempt normalization provenance schema is invalid")

    raw_hash = _sha256_json(raw_review)
    if item.get("raw_review_sha256") != raw_hash:
        errors.append("invalid attempt raw review hash mismatch")
    if provenance.get("policy") != RAW_NORMALIZATION_PROVENANCE_POLICY:
        errors.append("invalid attempt normalization provenance policy mismatch")
    if provenance.get("raw_review_sha256") != raw_hash:
        errors.append("invalid attempt provenance raw review hash mismatch")

    replayed, _dedupe_actions, _dedupe_errors = (
        _dedupe_same_response_exact_variants(raw_review, semantic_payload)
    )
    replayed, _coverage_actions, _coverage_errors = (
        _canonicalize_same_response_coverage_statuses(replayed, semantic_payload)
    )
    replayed, _inverse_actions, _inverse_errors = (
        _canonicalize_same_response_inverse_edges(replayed, semantic_payload)
    )
    pre_normalization_hash = _sha256_json(replayed)
    if item.get("pre_normalization_review_sha256") != pre_normalization_hash:
        errors.append("invalid attempt pre-normalization review hash mismatch")
    if provenance.get("pre_normalization_review_sha256") != pre_normalization_hash:
        errors.append("invalid attempt provenance pre-normalization hash mismatch")

    replayed_normalized, normalization_actions = _normalize_review_structure(
        replayed, semantic_payload
    )
    normalized_hash = _sha256_json(replayed_normalized)
    if normalized_review != replayed_normalized:
        errors.append("invalid attempt normalized review is not reproducible")
    if item.get("review_sha256") != normalized_hash:
        errors.append("invalid attempt normalized review hash mismatch")
    if provenance.get("normalized_review_sha256") != normalized_hash:
        errors.append("invalid attempt provenance normalized review hash mismatch")
    if provenance.get("actions") != normalization_actions:
        errors.append("invalid attempt normalization action audit mismatch")

    legacy_v1_receipt = (
        runner_version
        in {
            "2.25.0-saved-exact-payload-branch-span-quote-prefix-recovery",
            "2.26.0-saved-abstain-factual-constraint-recovery",
        }
        and item.get("validation_errors")
        == [_ABSTAIN_FACTUAL_CONSTRAINT_ERROR]
    )
    expected_validation_errors = _validate_review(
        replayed_normalized,
        abstain_cue_policy=(
            "v1_substring" if legacy_v1_receipt else "v2_pure_clause"
        ),
    )
    if normalization_actions:
        expected_validation_errors.append(
            "strict runner rejects structural normalization: "
            f"{len(normalization_actions)} action(s)"
        )
    expected_validation_errors.extend(
        _validate_against_payload(replayed_normalized, semantic_payload)
    )
    if extra_review_validator is not None:
        expected_validation_errors.extend(
            str(error)
            for error in extra_review_validator(
                replayed_normalized, semantic_payload
            )
            if str(error)
        )
    if (
        (item.get("sse_integrity") or {}).get("termination")
        == "validated_stop_eof"
        and (item.get("sse_integrity") or {}).get("review_validation") == "failed"
        and not expected_validation_errors
    ):
        expected_validation_errors.append(
            "validated EOF prevalidation disagreed with authoritative validation"
        )
    if item.get("validation_errors") != expected_validation_errors:
        errors.append("invalid attempt validation-error receipt mismatch")
    return errors


def _complete_terminal_semantic_attempt(
    item: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any] | None = None,
) -> bool:
    attempt = item.get("attempt")
    integrity = item.get("sse_integrity") or {}
    usage_integrity = integrity.get("usage_integrity") or {}
    if (
        not isinstance(attempt, int)
        or attempt < 1
        or integrity.get("termination")
        not in {"done_sentinel", "validated_stop_eof"}
        or integrity.get("finish_reason") != "stop"
        or integrity.get("finish_reason_stop_seen") is not True
        or integrity.get("event_integrity") != "passed"
        or usage_integrity.get("status") != "passed"
        or integrity.get("review_validation") != "failed"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(integrity.get("content_sha256") or "")
        )
    ):
        return False
    if semantic_payload is None:
        return False
    if _validate_invalid_attempt_raw_normalization_provenance(
        item, source_row, semantic_payload
    ):
        return False
    response_hash = (
        item.get("raw_review_sha256")
        if "raw_review_sha256" in item
        else item.get("review_sha256")
    )
    failure_matches = [
        failure
        for failure in source_row.get("failure_attempts") or []
        if isinstance(failure, dict)
        and failure.get("attempt") == attempt
        and failure.get("failure_category") == "semantic"
        and failure.get("termination") == "review_validation_error"
        and failure.get("response_hash_status") == "complete_json_sha256"
        and failure.get("response_sha256") == response_hash
    ]
    usage_matches = [
        usage
        for usage in source_row.get("attempt_usage") or []
        if isinstance(usage, dict)
        and usage.get("attempt") == attempt
        and all(
            isinstance(usage.get(key), int) and int(usage[key]) >= 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        and int(usage["total_tokens"])
        == int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
    ]
    return len(failure_matches) == 1 and len(usage_matches) == 1


def _quota_exhausted_verdict_block_consensus(
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    projector: Callable[
        [dict[str, Any], dict[str, Any]],
        tuple[dict[str, Any], list[dict[str, Any]], list[str]],
    ],
    require_target_semantic_consensus: bool = True,
) -> dict[str, Any] | None:
    """Prove a verdict-changing projection from a persisted failure consensus."""

    retry = source_row.get("retry_accounting") or {}
    semantic_quota = retry.get("semantic_failure_quota")
    semantic_failures = retry.get("semantic_failure_count")
    invalid_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict) and isinstance(item.get("review"), dict)
    ]
    if (
        source_row.get("status") != "failed"
        or source_row.get("failure_termination") != "review_validation_error"
        or retry.get("quota_exhausted") != "semantic"
        or not isinstance(semantic_quota, int)
        or semantic_quota < 2
        or not isinstance(semantic_failures, int)
        or semantic_failures < semantic_quota
        or len(invalid_attempts) != semantic_failures
        or len(invalid_attempts) < 2
    ):
        return None

    members: list[dict[str, Any]] = []
    for item in invalid_attempts:
        review = deepcopy(item["review"])
        if (
            item.get("review_sha256") != _sha256_json(review)
            or _invalid_attempt_has_structural_normalization(item)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
        ):
            return None
        projected, actions, raw_errors = projector(review, semantic_payload)
        if (
            not actions
            or item.get("validation_errors") != raw_errors
            or projected.get("verdict") != "block"
        ):
            return None
        target_semantic_signature = (
            _unsupported_target_semantic_signature(review, actions)
            if require_target_semantic_consensus
            else None
        )
        if target_semantic_signature == []:
            return None
        members.append(
            {
                "attempt": int(item["attempt"]),
                "raw_review": review,
                "raw_review_sha256": _sha256_json(review),
                "projected_review": projected,
                "projected_review_sha256": _sha256_json(projected),
                "raw_validation_errors": raw_errors,
                "actions": actions,
                "structural_signature": _verdict_block_structural_signature(review),
                "target_semantic_signature": target_semantic_signature,
                "termination": (item.get("sse_integrity") or {}).get("termination"),
                "content_sha256": (item.get("sse_integrity") or {}).get(
                    "content_sha256"
                ),
            }
        )

    structural_hashes = {
        _sha256_json(member["structural_signature"]) for member in members
    }
    action_hashes = {_sha256_json(member["actions"]) for member in members}
    target_error_shapes = {
        tuple(member["raw_validation_errors"]) for member in members
    }
    target_semantic_hashes = {
        _sha256_json(member["target_semantic_signature"])
        for member in members
        if member["target_semantic_signature"] is not None
    }
    target_semantic_presence = {
        member["target_semantic_signature"] is not None for member in members
    }
    attempts = [int(member["attempt"]) for member in members]
    if (
        len(structural_hashes) != 1
        or len(action_hashes) != 1
        or len(target_error_shapes) != 1
        or len(target_semantic_presence) != 1
        or target_semantic_hashes
        and len(target_semantic_hashes) != 1
        or len(attempts) != len(set(attempts))
    ):
        return None
    members.sort(key=lambda member: int(member["attempt"]))
    final = members[-1]
    consensus = {
        "policy": VERDICT_BLOCK_CONSENSUS_POLICY,
        "semantic_failure_quota": semantic_quota,
        "semantic_failure_count": semantic_failures,
        "response_count": len(members),
        "source_response_row_sha256": _sha256_json(source_row),
        "structural_signature_sha256": next(iter(structural_hashes)),
        "target_action_sha256": next(iter(action_hashes)),
        "sources": [
            {
                "source_attempt": member["attempt"],
                "raw_review_sha256": member["raw_review_sha256"],
                "validation_errors": member["raw_validation_errors"],
                "termination": member["termination"],
                "content_sha256": member["content_sha256"],
            }
            for member in members
        ],
    }
    if target_semantic_hashes:
        consensus["target_semantic_signature_sha256"] = next(
            iter(target_semantic_hashes)
        )
    return {"final": final, "members": members, "consensus": consensus}


def _canonicalize_same_response_inverse_edges(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Canonicalize only redundant concept cross-references from one response.

    ``required_branches`` and ``evidence_concept_coverage`` are the authoritative
    edge declarations.  Projection is allowed only when the raw response is
    otherwise fully strict and aligned with the blind payload.  No concept,
    variant, branch, evidence item, or semantic text is created here.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    permitted_errors = {
        "concept.branch_labels and required_branches edges must be symmetric",
        "concept.evidence_labels and evidence_concept_coverage edges must be symmetric",
    }
    if (
        not raw_errors
        or set(raw_errors) - permitted_errors
        or raw.get("verdict") == "block"
        or raw.get("answerability") != "answer"
    ):
        return raw, [], raw_errors

    concepts = (raw.get("required_concepts") or {}).get("all_of") or []
    concept_by_key = {
        _concept_key_normalized(item.get("concept_key")): item
        for item in concepts
        if isinstance(item, dict)
    }
    if len(concept_by_key) != len(concepts) or not concept_by_key:
        return raw, [], raw_errors

    branch_labels_by_key = {key: [] for key in concept_by_key}
    for branch in raw.get("required_branches") or []:
        label = str(branch["branch_label"])
        keys = [
            _concept_key_normalized(key)
            for key in branch["required_concept_keys"]
        ]
        if not keys or any(key not in concept_by_key for key in keys):
            return deepcopy(value), [], raw_errors
        for key in keys:
            if label not in branch_labels_by_key[key]:
                branch_labels_by_key[key].append(label)

    evidence_labels_by_key = {key: [] for key in concept_by_key}
    for coverage in raw.get("evidence_concept_coverage") or []:
        label = str(coverage["evidence_label"])
        keys = [
            _concept_key_normalized(key)
            for key in coverage["required_concept_keys"]
        ]
        status = coverage.get("status")
        if (
            status == "covered"
            and not keys
            or status in {"uncovered", "insufficient"}
            and keys
            or status not in {"covered", "uncovered", "insufficient"}
            or any(key not in concept_by_key for key in keys)
        ):
            return deepcopy(value), [], raw_errors
        if status != "covered":
            continue
        for key in keys:
            if label not in evidence_labels_by_key[key]:
                evidence_labels_by_key[key].append(label)

    if any(
        not branch_labels_by_key[key] or not evidence_labels_by_key[key]
        for key in concept_by_key
    ):
        return deepcopy(value), [], raw_errors

    projected = deepcopy(raw)
    actions: list[dict[str, Any]] = []
    for concept in (projected.get("required_concepts") or {}).get("all_of") or []:
        key = _concept_key_normalized(concept["concept_key"])
        for field, replacement in (
            ("branch_labels", branch_labels_by_key[key]),
            ("evidence_labels", evidence_labels_by_key[key]),
        ):
            before = list(concept[field])
            if before == replacement:
                continue
            concept[field] = list(replacement)
            actions.append(
                {
                    "concept_key": str(concept["concept_key"]),
                    "field": field,
                    "before": before,
                    "after": list(replacement),
                    "authority": (
                        "required_branches"
                        if field == "branch_labels"
                        else "evidence_concept_coverage"
                    ),
                }
            )
    projected_errors = _validate_review(projected)
    projected_errors.extend(_validate_against_payload(projected, semantic_payload))
    if not actions or projected_errors:
        return deepcopy(value), [], raw_errors
    return projected, actions, raw_errors


_BRANCH_EDGE_SYMMETRY_ERROR = (
    "concept.branch_labels and required_branches edges must be symmetric"
)


def _review_materialization_errors(
    review: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    abstention_cue_policy: str = "v2_pure_clause",
    materialization_policy_version: str | None = None,
) -> list[str]:
    contract = {
        "answerability": {"required": review.get("answerability")},
        "required_concepts": {
            "all_of": [
                {
                    "concept_id": concept.get("concept_key"),
                    "canonical_text": concept.get("canonical_text"),
                    "any_of": concept.get("any_of") or [],
                }
                for concept in (review.get("required_concepts") or {}).get("all_of")
                or []
            ]
        },
        "critical_constraints": review.get("critical_constraints") or {},
    }
    evidence = [
        {"text": str(item.get("text") or "")}
        for item in semantic_payload.get("evidence") or []
        if isinstance(item, dict)
    ]
    try:
        _shared_materialize_concept_groups(
            contract,
            evidence,
            abstention_cue_policy=abstention_cue_policy,
            **(
                {"policy_version": materialization_policy_version}
                if materialization_policy_version is not None
                else {}
            ),
        )
    except ValueError as exc:
        return [f"concept materialization failed: {exc}"]
    return []


def _review_materialization_result(
    review: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    materialization_policy_version: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return the production materialization and audit for one strict review."""

    contract = {
        "answerability": {"required": review.get("answerability")},
        "required_concepts": {
            "all_of": [
                {
                    "concept_id": concept.get("concept_key"),
                    "canonical_text": concept.get("canonical_text"),
                    "any_of": concept.get("any_of") or [],
                }
                for concept in (review.get("required_concepts") or {}).get("all_of")
                or []
            ]
        },
        "critical_constraints": review.get("critical_constraints") or {},
    }
    evidence = [
        {"text": str(item.get("text") or "")}
        for item in semantic_payload.get("evidence") or []
        if isinstance(item, dict)
    ]
    try:
        return _shared_materialize_concept_groups(
            contract,
            evidence,
            **(
                {"policy_version": materialization_policy_version}
                if materialization_policy_version is not None
                else {}
            ),
        )
    except ValueError:
        return None


def _project_saved_same_concept_variant_subset(
    value: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    materialization_policy_version: str | None = None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[str], dict[str, Any]
] | None:
    """Project only model-authored same-concept variants onto canonical slots.

    This intentionally does *not* call the structural normalizer, inspect the
    case/reference/candidate, append text, or combine attempts.  Each schema
    extra must be an exact duplicate of that concept's first existing ``any_of``
    value.  Replacing canonical text with that already-authored variant makes
    the result a semantic subset of one response; deleting the inert delimiter
    then makes the envelope schema-valid.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    extra_paths = _review_schema_extra_paths(raw)
    expected_schema_error = f"review contains schema-external fields: {extra_paths}"
    if not extra_paths or raw_errors != [expected_schema_error]:
        return None

    path_re = re.compile(
        r"^\$\.required_concepts\.all_of\[(0|[1-9][0-9]*)\]\.([^\.\[\]]+)$"
    )
    concepts = (raw.get("required_concepts") or {}).get("all_of") or []
    selected: list[tuple[int, str, str]] = []
    for path in extra_paths:
        match = path_re.fullmatch(path)
        if match is None:
            return None
        index = int(match.group(1))
        key = match.group(2)
        if (
            index >= len(concepts)
            or not isinstance(concepts[index], dict)
            or not key
            or key in _CONCEPT_ALLOWED_FIELDS
        ):
            return None
        concept = concepts[index]
        any_of = concept.get("any_of")
        if (
            not isinstance(any_of, list)
            or not any_of
            or type(any_of[0]) is not str
            or key != any_of[0]
            or key not in any_of
        ):
            return None
        extra_value = concept.get(key)
        if type(extra_value) is not str or extra_value not in {"", ","}:
            return None
        selected.append((index, key, path))

    projected = deepcopy(raw)
    actions: list[dict[str, Any]] = []
    for index, key, path in selected:
        concept = projected["required_concepts"]["all_of"][index]
        before_canonical = concept["canonical_text"]
        before_extra = concept[key]
        actions.extend(
            (
                {
                    "operation": "replace_canonical_with_existing_any_of",
                    "json_path": (
                        f"$.required_concepts.all_of[{index}].canonical_text"
                    ),
                    "before_sha256": _sha256_json(before_canonical),
                    "after_sha256": _sha256_json(key),
                    "after": key,
                    "authority": (
                        f"$.required_concepts.all_of[{index}].any_of[0]"
                    ),
                },
                {
                    "operation": "remove_schema_extra",
                    "json_path": path,
                    "before_sha256": _sha256_json(before_extra),
                    "after_state": "absent",
                    "authority": "same_concept_existing_any_of_exact_key",
                },
            )
        )
        concept["canonical_text"] = key
        del concept[key]

    expected = deepcopy(projected)
    for index, key, _path in selected:
        raw_concept = raw["required_concepts"]["all_of"][index]
        projected_concept = expected["required_concepts"]["all_of"][index]
        projected_concept["canonical_text"] = raw_concept["canonical_text"]
        projected_concept[key] = raw_concept[key]
    if expected != raw:
        return None
    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    if final_errors:
        return None
    materialization = _review_materialization_result(
        projected,
        semantic_payload,
        materialization_policy_version=materialization_policy_version,
    )
    if materialization is None:
        return None
    _groups, materialization_audit = materialization
    if (
        materialization_audit.get("removed_variant_count") != 0
        or materialization_audit.get("removed_variants") != []
        or materialization_audit.get("weak_canonical_count") != 0
        or materialization_audit.get("production_match_collision_count") != 0
    ):
        return None
    return projected, actions, raw_errors, materialization_audit


def _project_saved_quote_key_branch_span(
    value: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    materialization_policy_version: str | None = None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[str], dict[str, Any]
] | None:
    """仅恢复被唯一引号内容泄漏成空值键的一条 branch span。

    权威文本完全来自同一冻结请求的精确 ``query_span``。投影只允许删除
    一个空字符串 schema-extra 键并替换一个前缀截断的 span；不拼接模型
    字段、不读取其他 attempt，也不改变任何答案语义字段。
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    branches = raw.get("required_branches")
    payload_branches = semantic_payload.get("branches")
    extra_paths = _review_schema_extra_paths(raw)
    if (
        not isinstance(branches, list)
        or not branches
        or not isinstance(payload_branches, list)
        or len(branches) != len(payload_branches)
        or any(not isinstance(item, dict) for item in branches)
        or any(not isinstance(item, dict) for item in payload_branches)
        or len(extra_paths) != 1
    ):
        return None

    raw_labels = [str(item.get("branch_label") or "") for item in branches]
    payload_labels = [
        str(item.get("branch_label") or "") for item in payload_branches
    ]
    if (
        raw_labels != payload_labels
        or any(not label for label in raw_labels)
        or len(raw_labels) != len(set(raw_labels))
    ):
        return None

    mismatches: list[tuple[int, str, str, str]] = []
    for index, (branch, payload_branch) in enumerate(
        zip(branches, payload_branches, strict=True)
    ):
        raw_span = branch.get("query_span")
        expected_span = payload_branch.get("query_span")
        if not isinstance(raw_span, str) or not isinstance(expected_span, str):
            return None
        extras = sorted(set(branch) - _BRANCH_ALLOWED_FIELDS)
        if raw_span == expected_span:
            if extras or set(branch) != _BRANCH_ALLOWED_FIELDS:
                return None
            continue
        if len(extras) != 1:
            return None
        mismatches.append((index, extras[0], raw_span, expected_span))
    if len(mismatches) != 1:
        return None

    branch_index, extra_key, raw_span, expected_span = mismatches[0]
    branch_label = raw_labels[branch_index]
    branch = branches[branch_index]
    expected_path = f"$.required_branches[{branch_index}].{extra_key}"
    expected_raw_errors = [
        f"review contains schema-external fields: {[expected_path]}",
        f"branch query_span differs from payload for {branch_label}",
        "required branch labels/query spans must exactly match payload",
    ]
    if (
        extra_paths != [expected_path]
        or raw_errors != expected_raw_errors
        or type(branch.get(extra_key)) is not str
        or branch[extra_key] != ""
        or not raw_span
        or len(raw_span) >= len(expected_span)
        or not expected_span.startswith(raw_span)
    ):
        return None

    suffix = expected_span[len(raw_span) :]
    quote_matches: list[tuple[str, str, str]] = []
    for opener, closer in QUOTE_FRAGMENT_PAIRS:
        if not suffix.startswith(opener):
            continue
        if opener == closer:
            if suffix.count(opener) != 2:
                continue
            closer_index = suffix.find(closer, len(opener))
        else:
            if suffix.count(opener) != 1 or suffix.count(closer) != 1:
                continue
            closer_index = suffix.find(closer, len(opener))
        if closer_index < len(opener):
            continue
        quoted_content = suffix[len(opener) : closer_index]
        tail = suffix[closer_index + len(closer) :]
        if quoted_content and tail.strip():
            quote_matches.append((opener, closer, quoted_content))
    if len(quote_matches) != 1:
        return None
    opener, closer, quoted_content = quote_matches[0]
    if extra_key != quoted_content:
        return None

    projected = deepcopy(raw)
    projected_branch = projected["required_branches"][branch_index]
    before_extra = projected_branch.pop(extra_key)
    projected_branch["query_span"] = expected_span
    inverse = deepcopy(projected)
    inverse_branch = inverse["required_branches"][branch_index]
    inverse_branch["query_span"] = raw_span
    inverse_branch[extra_key] = before_extra
    if inverse != raw:
        return None

    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    if final_errors:
        return None
    materialization = _review_materialization_result(
        projected,
        semantic_payload,
        materialization_policy_version=materialization_policy_version,
    )
    if materialization is None:
        return None
    _groups, materialization_audit = materialization
    actions = [
        {
            "operation": "remove_quoted_schema_extra_key",
            "json_path": expected_path,
            "before_sha256": _sha256_json(before_extra),
            "after_state": "absent",
            "authority": "unique_exact_request_suffix_quote_content",
            "quote_pair": [opener, closer],
        },
        {
            "operation": "replace_branch_span_with_exact_payload_span",
            "json_path": (
                f"$.required_branches[{branch_index}].query_span"
            ),
            "before_sha256": _sha256_json(raw_span),
            "after_sha256": _sha256_json(expected_span),
            "after": expected_span,
            "authority": "same_request_payload_exact_branch_span",
            "quote_pair": [opener, closer],
        },
    ]
    return projected, actions, raw_errors, materialization_audit


def _canonicalize_saved_nonsemantic_composition(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Compose exactly status-alias then inverse branch-label projection.

    Both stages are verdict-preserving and touch disjoint JSON paths.  This
    helper deliberately excludes evidence-edge inversion, variant changes, and
    every verdict/answerability projector.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    status_error = re.compile(
        r"^evidence_concept_coverage\[\d+\]\.status must be one of \[.*\]$"
    )
    status_errors = [error for error in raw_errors if status_error.fullmatch(error)]
    branch_errors = [error for error in raw_errors if error == _BRANCH_EDGE_SYMMETRY_ERROR]
    if (
        not status_errors
        or branch_errors != [_BRANCH_EDGE_SYMMETRY_ERROR]
        or len(raw_errors) != len(status_errors) + 1
    ):
        return raw, [], raw_errors

    intermediate, coverage_actions, coverage_raw_errors = (
        _canonicalize_same_response_coverage_statuses(
            raw,
            semantic_payload,
            permitted_additional_errors=frozenset({_BRANCH_EDGE_SYMMETRY_ERROR}),
        )
    )
    if not coverage_actions or coverage_raw_errors != raw_errors:
        return raw, [], raw_errors
    intermediate_errors = _validate_review(intermediate)
    intermediate_errors.extend(
        _validate_against_payload(intermediate, semantic_payload)
    )
    if intermediate_errors != [_BRANCH_EDGE_SYMMETRY_ERROR]:
        return raw, [], raw_errors

    projected, inverse_actions, inverse_raw_errors = (
        _canonicalize_same_response_inverse_edges(intermediate, semantic_payload)
    )
    if (
        not inverse_actions
        or inverse_raw_errors != intermediate_errors
        or any(action.get("field") != "branch_labels" for action in inverse_actions)
        or projected.get("verdict") != raw.get("verdict")
        or projected.get("answerability") != raw.get("answerability")
    ):
        return raw, [], raw_errors
    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(_review_materialization_errors(projected, semantic_payload))
    if final_errors:
        return raw, [], raw_errors

    coverage_stage_actions = [
        {
            "json_path": (
                f"/evidence_concept_coverage/{action['coverage_index']}/status"
            ),
            "before": action["before"],
            "after": action["after"],
        }
        for action in coverage_actions
    ]
    concepts = (intermediate.get("required_concepts") or {}).get("all_of") or []
    concept_index_by_key = {
        _concept_key_normalized(concept.get("concept_key")): index
        for index, concept in enumerate(concepts)
        if isinstance(concept, dict)
    }
    inverse_stage_actions: list[dict[str, Any]] = []
    for action in inverse_actions:
        index = concept_index_by_key.get(
            _concept_key_normalized(action.get("concept_key"))
        )
        if index is None:
            return raw, [], raw_errors
        inverse_stage_actions.append(
            {
                "json_path": f"/required_concepts/all_of/{index}/branch_labels",
                "before": action["before"],
                "after": action["after"],
            }
        )
    touched_paths = {
        action["json_path"]
        for action in [*coverage_stage_actions, *inverse_stage_actions]
    }
    if len(touched_paths) != len(coverage_stage_actions) + len(inverse_stage_actions):
        return raw, [], raw_errors

    raw_hash = _sha256_json(raw)
    intermediate_hash = _sha256_json(intermediate)
    final_hash = _sha256_json(projected)
    stages = [
        {
            "stage": "coverage_status_alias",
            "policy": COVERAGE_STATUS_CANONICALIZATION_POLICY,
            "input_review_sha256": raw_hash,
            "output_review_sha256": intermediate_hash,
            "actions": coverage_stage_actions,
        },
        {
            "stage": "inverse_branch_labels",
            "policy": INVERSE_EDGE_CANONICALIZATION_POLICY,
            "input_review_sha256": intermediate_hash,
            "output_review_sha256": final_hash,
            "actions": inverse_stage_actions,
        },
    ]
    return projected, stages, raw_errors


def _json_pointer_segment(value: str) -> str:
    """Escape one RFC 6901 JSON Pointer segment for append-only audit paths."""

    return value.replace("~", "~0").replace("/", "~1")


def _canonicalize_saved_redundant_field_coverage_composition(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Reconstruct only same-response redundant fields, then map coverage aliases.

    The reconstruction stage never reads candidate semantics.  A missing
    ``any_of`` repeats the same concept's canonical text; missing evidence and
    branch labels reverse explicit edges already present in this raw response.
    Explicit empty/null/wrong-type fields are not repaired.  The second stage
    is the frozen coverage-v4 finite-alias projector.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    if raw.get("answerability") != "answer" or raw.get("verdict") not in {
        "revise",
        "block",
    }:
        return raw, [], raw_errors

    concepts = (raw.get("required_concepts") or {}).get("all_of")
    branches = raw.get("required_branches")
    coverage = raw.get("evidence_concept_coverage")
    if (
        not isinstance(concepts, list)
        or not concepts
        or any(not isinstance(item, dict) for item in concepts)
        or not isinstance(branches, list)
        or any(not isinstance(item, dict) for item in branches)
        or not isinstance(coverage, list)
        or any(not isinstance(item, dict) for item in coverage)
    ):
        return raw, [], raw_errors

    concept_keys = [
        _concept_key_normalized(item.get("concept_key")) for item in concepts
    ]
    if (
        "" in concept_keys
        or len(concept_keys) != len(set(concept_keys))
    ):
        return raw, [], raw_errors

    evidence_by_concept: dict[str, list[str]] = {key: [] for key in concept_keys}
    for item in coverage:
        if item.get("status") != "covered":
            continue
        label = str(item.get("evidence_label") or "")
        keys = item.get("required_concept_keys")
        if not label or not isinstance(keys, list):
            return raw, [], raw_errors
        normalized_keys = [_concept_key_normalized(key) for key in keys]
        if (
            not normalized_keys
            or "" in normalized_keys
            or len(normalized_keys) != len(set(normalized_keys))
            or any(key not in evidence_by_concept for key in normalized_keys)
        ):
            return raw, [], raw_errors
        for key in normalized_keys:
            evidence_by_concept[key].append(label)

    branches_by_concept: dict[str, list[str]] = {key: [] for key in concept_keys}
    for item in branches:
        label = str(item.get("branch_label") or "")
        keys = item.get("required_concept_keys")
        if not label or not isinstance(keys, list):
            return raw, [], raw_errors
        normalized_keys = [_concept_key_normalized(key) for key in keys]
        if (
            not normalized_keys
            or "" in normalized_keys
            or len(normalized_keys) != len(set(normalized_keys))
            or any(key not in branches_by_concept for key in normalized_keys)
        ):
            return raw, [], raw_errors
        for key in normalized_keys:
            branches_by_concept[key].append(label)

    intermediate = deepcopy(raw)
    intermediate_concepts = intermediate["required_concepts"]["all_of"]
    reconstruction_actions: list[dict[str, Any]] = []
    for index, concept in enumerate(intermediate_concepts):
        key = concept_keys[index]
        canonical = str(concept.get("canonical_text") or "").strip()
        base_path = f"/required_concepts/all_of/{index}"
        if "any_of" not in concept:
            if not canonical:
                return raw, [], raw_errors
            after = [canonical]
            concept["any_of"] = after
            reconstruction_actions.append(
                {
                    "operation": "add",
                    "json_path": f"{base_path}/any_of",
                    "concept_key": str(concept.get("concept_key") or ""),
                    "field": "any_of",
                    "before_state": "absent",
                    "after": after,
                    "authority": "same_concept_canonical_text",
                }
            )
        if "evidence_labels" not in concept:
            after = sorted(set(evidence_by_concept[key]))
            if not after:
                return raw, [], raw_errors
            concept["evidence_labels"] = after
            reconstruction_actions.append(
                {
                    "operation": "add",
                    "json_path": f"{base_path}/evidence_labels",
                    "concept_key": str(concept.get("concept_key") or ""),
                    "field": "evidence_labels",
                    "before_state": "absent",
                    "after": after,
                    "authority": "same_response_covered_coverage_inverse",
                }
            )
        if "branch_labels" not in concept:
            after = sorted(set(branches_by_concept[key]))
            if not after:
                return raw, [], raw_errors
            concept["branch_labels"] = after
            reconstruction_actions.append(
                {
                    "operation": "add",
                    "json_path": f"{base_path}/branch_labels",
                    "concept_key": str(concept.get("concept_key") or ""),
                    "field": "branch_labels",
                    "before_state": "absent",
                    "after": after,
                    "authority": "same_response_required_branches_inverse",
                }
            )
    if not reconstruction_actions:
        return raw, [], raw_errors
    touched_paths = [str(action["json_path"]) for action in reconstruction_actions]
    if len(touched_paths) != len(set(touched_paths)):
        return raw, [], raw_errors

    intermediate_errors = _validate_review(intermediate)
    intermediate_errors.extend(
        _validate_against_payload(intermediate, semantic_payload)
    )
    projected, coverage_actions, coverage_raw_errors = (
        _canonicalize_same_response_coverage_statuses(
            intermediate, semantic_payload
        )
    )
    if (
        not coverage_actions
        or coverage_raw_errors != intermediate_errors
    ):
        return raw, [], raw_errors

    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(_review_materialization_errors(projected, semantic_payload))
    if final_errors:
        return raw, [], raw_errors

    coverage_stage_actions = [
        {
            "operation": "replace",
            "json_path": (
                f"/evidence_concept_coverage/{action['coverage_index']}/status"
            ),
            "before": action["before"],
            "after": action["after"],
            "authority": action["authority"],
        }
        for action in coverage_actions
    ]
    all_paths = [
        str(action["json_path"])
        for action in [*reconstruction_actions, *coverage_stage_actions]
    ]
    if len(all_paths) != len(set(all_paths)):
        return raw, [], raw_errors

    raw_hash = _sha256_json(raw)
    intermediate_hash = _sha256_json(intermediate)
    final_hash = _sha256_json(projected)
    return (
        projected,
        [
            {
                "stage": "same_response_redundant_field_reconstruction",
                "policy": REDUNDANT_FIELD_RECONSTRUCTION_POLICY,
                "input_review_sha256": raw_hash,
                "output_review_sha256": intermediate_hash,
                "actions": reconstruction_actions,
            },
            {
                "stage": "coverage_status_alias",
                "policy": COVERAGE_STATUS_CANONICALIZATION_POLICY,
                "input_review_sha256": intermediate_hash,
                "output_review_sha256": final_hash,
                "actions": coverage_stage_actions,
            },
        ],
        raw_errors,
    )


_ABSTAIN_FACTUAL_CONSTRAINT_ERROR = (
    "abstain response must not require factual numerics or identifiers"
)


def _abstain_constraint_action_signature(
    stages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the invariant shape of the one permitted projection."""

    return [
        {
            "stage": stage.get("stage"),
            "policy": stage.get("policy"),
            "actions": [
                {
                    key: action.get(key)
                    for key in ("operation", "json_paths", "authority")
                }
                for action in stage.get("actions") or []
                if isinstance(action, dict)
            ],
        }
        for stage in stages
    ]


def _canonicalize_saved_abstain_factual_constraints(
    value: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    recovery_policy: str = ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Project one structurally unsupported abstention to a frozen sentinel.

    V1 is retained byte-for-byte for historical audit replay.  V2 does not
    interpret model-authored refusal prose: it replaces the sole unsupported
    concept with a fixed sentinel and removes evidence-free factual constraints.
    No case identity, reference answer, candidate, or other attempt is read.
    """

    raw = deepcopy(value)
    abstain_cue_policy = (
        "v1_substring"
        if recovery_policy == LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        else "v2_pure_clause"
    )
    validation_raw = deepcopy(raw)
    if recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY:
        validation_constraints = validation_raw.get("critical_constraints")
        if isinstance(validation_constraints, dict):
            validation_constraints["polarity"] = deepcopy(
                ABSTENTION_SENTINEL_POLARITY
            )
    raw_errors = _validate_review(
        validation_raw, abstain_cue_policy=abstain_cue_policy
    )
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    concepts = (raw.get("required_concepts") or {}).get("all_of")
    constraints = raw.get("critical_constraints")
    branches = raw.get("required_branches")
    coverage = raw.get("evidence_concept_coverage")
    variant_error = (
        "abstain concept[0] variants must each express only an explicit "
        "absence/uncertainty rationale"
    )
    permitted_raw_errors = (
        {(_ABSTAIN_FACTUAL_CONSTRAINT_ERROR,)}
        if recovery_policy == LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        else {
            (variant_error,),
            (_ABSTAIN_FACTUAL_CONSTRAINT_ERROR,),
            (variant_error, _ABSTAIN_FACTUAL_CONSTRAINT_ERROR),
        }
    )
    if (
        tuple(raw_errors) not in permitted_raw_errors
        or raw.get("answerability") != "abstain"
        or semantic_payload.get("unanswerable") is not True
        or raw.get("verdict") not in {"approve", "revise", "block"}
        or _review_schema_extra_paths(raw)
        or not isinstance(concepts, list)
        or len(concepts) != 1
        or not isinstance(concepts[0], dict)
        or concepts[0].get("evidence_labels") != []
        or coverage != []
        or not isinstance(branches, list)
        or not branches
        or any(
            not isinstance(branch, dict)
            or not branch.get("required_concept_keys")
            for branch in branches
        )
        or not isinstance(constraints, dict)
        or (constraints.get("polarity") or {}).get("required")
        != "abstain_without_factual_claims"
    ):
        return raw, [], raw_errors

    numerics = constraints.get("numerics")
    identifiers = constraints.get("identifiers")
    if (
        not isinstance(numerics, list)
        or not isinstance(identifiers, list)
        or recovery_policy == LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        and not numerics
        and not identifiers
    ):
        return raw, [], raw_errors
    for kind, items, allowed_keys in (
        ("numerics", numerics, {"value", "unit", "evidence_labels"}),
        ("identifiers", identifiers, {"value", "evidence_labels"}),
    ):
        for item in items:
            if (
                not isinstance(item, dict)
                or set(item) != allowed_keys
                or not isinstance(item.get("value"), str)
                or not item["value"].strip()
                or item.get("evidence_labels") != []
            ):
                return raw, [], raw_errors
            if kind == "numerics" and not (
                item.get("unit") is None
                or isinstance(item.get("unit"), str)
                and item["unit"].strip()
            ):
                return raw, [], raw_errors

    projected = deepcopy(raw)
    raw_concept_text = {
        "canonical_text": concepts[0].get("canonical_text"),
        "any_of": deepcopy(concepts[0].get("any_of")),
    }
    raw_polarity = deepcopy(constraints.get("polarity"))
    if recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY:
        projected_concept = projected["required_concepts"]["all_of"][0]
        projected_concept["canonical_text"] = ABSTENTION_SENTINEL_CANONICAL_TEXT
        projected_concept["any_of"] = list(ABSTENTION_SENTINEL_VARIANTS)
        projected["critical_constraints"]["polarity"] = deepcopy(
            ABSTENTION_SENTINEL_POLARITY
        )
    projected["critical_constraints"]["numerics"] = []
    projected["critical_constraints"]["identifiers"] = []
    final_errors = _validate_review(
        projected, abstain_cue_policy=abstain_cue_policy
    )
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(
        _review_materialization_errors(
            projected,
            semantic_payload,
            abstention_cue_policy=abstain_cue_policy,
        )
    )
    if final_errors:
        return raw, [], raw_errors

    unchanged = deepcopy(projected)
    if recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY:
        unchanged_concept = unchanged["required_concepts"]["all_of"][0]
        unchanged_concept["canonical_text"] = raw_concept_text["canonical_text"]
        unchanged_concept["any_of"] = deepcopy(raw_concept_text["any_of"])
        unchanged["critical_constraints"]["polarity"] = raw_polarity
    unchanged["critical_constraints"]["numerics"] = deepcopy(numerics)
    unchanged["critical_constraints"]["identifiers"] = deepcopy(identifiers)
    if unchanged != raw:
        return raw, [], raw_errors

    before = {"numerics": numerics, "identifiers": identifiers}
    json_paths = [
        "/critical_constraints/numerics",
        "/critical_constraints/identifiers",
    ]
    after: dict[str, Any] = {"numerics": [], "identifiers": []}
    authority = "same_response_abstention_forbids_factual_constraints"
    if recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY:
        before = {
            "canonical_text": raw_concept_text["canonical_text"],
            "any_of": raw_concept_text["any_of"],
            "polarity": raw_polarity,
            **before,
        }
        json_paths = [
            "/required_concepts/all_of/0/canonical_text",
            "/required_concepts/all_of/0/any_of",
            "/critical_constraints/polarity",
            *json_paths,
        ]
        after = {
            "canonical_text": ABSTENTION_SENTINEL_CANONICAL_TEXT,
            "any_of": list(ABSTENTION_SENTINEL_VARIANTS),
            "polarity": deepcopy(ABSTENTION_SENTINEL_POLARITY),
            **after,
        }
        authority = "unsupported_abstention_structural_projection_to_frozen_sentinel"
    action = {
        "operation": "replace",
        "json_paths": json_paths,
        "before_sha256": _sha256_json(before),
        "after": after,
        "authority": authority,
    }
    return (
        projected,
        [
            {
                "stage": (
                    "abstain_factual_constraint_removal"
                    if recovery_policy
                    == LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
                    else "abstain_sentinel_and_factual_constraint_projection"
                ),
                "policy": recovery_policy,
                "input_review_sha256": _sha256_json(raw),
                "output_review_sha256": _sha256_json(projected),
                "actions": [action],
            }
        ],
        raw_errors,
    )


def _canonicalize_saved_quote_fragment_inverse_composition(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Rejoin one uniquely quoted branch fragment, then repair inverse labels.

    This projector is deliberately saved-response-only.  It does not infer a
    span: a finite quote-pair reconstruction must match the same request branch
    byte-for-byte.  The second stage reuses required branches as the sole
    authority for concept ``branch_labels``.  No answer semantics are added.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    extra_paths = _review_schema_extra_paths(raw)
    branches = raw.get("required_branches")
    payload_branches = semantic_payload.get("branches")
    if (
        len(extra_paths) != 1
        or not isinstance(branches, list)
        or not branches
        or not isinstance(payload_branches, list)
        or not payload_branches
        or any(not isinstance(item, dict) for item in branches)
        or any(not isinstance(item, dict) for item in payload_branches)
    ):
        return raw, [], raw_errors

    payload_labels = [str(item.get("branch_label") or "") for item in payload_branches]
    raw_labels = [str(item.get("branch_label") or "") for item in branches]
    if (
        any(not label for label in payload_labels)
        or len(payload_labels) != len(set(payload_labels))
        or len(raw_labels) != len(set(raw_labels))
        or raw_labels != payload_labels
    ):
        return raw, [], raw_errors
    expected_span_by_label = {
        str(item["branch_label"]): str(item.get("query_span") or "")
        for item in payload_branches
    }

    unexpected: list[tuple[int, str, str]] = []
    for index, branch in enumerate(branches):
        for key in sorted(set(branch) - _BRANCH_ALLOWED_FIELDS):
            fragment = branch.get(key)
            if not isinstance(key, str) or not isinstance(fragment, str):
                return raw, [], raw_errors
            unexpected.append((index, key, fragment))
    if len(unexpected) != 1:
        return raw, [], raw_errors
    branch_index, fragment_key, fragment_value = unexpected[0]
    branch = branches[branch_index]
    branch_label = str(branch.get("branch_label") or "")
    prefix = branch.get("query_span")
    if (
        not fragment_key.strip()
        or not fragment_value.strip()
        or not isinstance(prefix, str)
        or not prefix
        or branch_label not in expected_span_by_label
    ):
        return raw, [], raw_errors
    expected_span = expected_span_by_label[branch_label]
    matches = [
        (opener, closer, f"{prefix}{opener}{fragment_key}{closer}{fragment_value}")
        for opener, closer in QUOTE_FRAGMENT_PAIRS
        if f"{prefix}{opener}{fragment_key}{closer}{fragment_value}"
        == expected_span
    ]
    if len(matches) != 1:
        return raw, [], raw_errors

    schema_error = f"review contains schema-external fields: {extra_paths}"
    expected_raw_errors = [
        schema_error,
        f"branch query_span differs from payload for {branch_label}",
        "required branch labels/query spans must exactly match payload",
        _BRANCH_EDGE_SYMMETRY_ERROR,
    ]
    if raw_errors != expected_raw_errors:
        return raw, [], raw_errors

    opener, closer, reconstructed_span = matches[0]
    intermediate = deepcopy(raw)
    intermediate_branch = intermediate["required_branches"][branch_index]
    before_prefix = intermediate_branch["query_span"]
    before_fragment = intermediate_branch.pop(fragment_key)
    intermediate_branch["query_span"] = reconstructed_span
    intermediate_errors = _validate_review(intermediate)
    intermediate_errors.extend(
        _validate_against_payload(intermediate, semantic_payload)
    )
    if intermediate_errors != [_BRANCH_EDGE_SYMMETRY_ERROR]:
        return raw, [], raw_errors

    projected, inverse_actions, inverse_raw_errors = (
        _canonicalize_same_response_inverse_edges(intermediate, semantic_payload)
    )
    if (
        not inverse_actions
        or inverse_raw_errors != intermediate_errors
        or any(action.get("field") != "branch_labels" for action in inverse_actions)
        or projected.get("verdict") != raw.get("verdict")
        or projected.get("answerability") != raw.get("answerability")
        or projected.get("required_branches") != intermediate.get("required_branches")
        or projected.get("evidence_concept_coverage")
        != raw.get("evidence_concept_coverage")
        or projected.get("critical_constraints") != raw.get("critical_constraints")
    ):
        return raw, [], raw_errors
    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(_review_materialization_errors(projected, semantic_payload))
    if final_errors:
        return raw, [], raw_errors

    escaped_fragment_key = _json_pointer_segment(fragment_key)
    quote_actions = [
        {
            "operation": "replace",
            "json_path": f"/required_branches/{branch_index}/query_span",
            "before": before_prefix,
            "after": reconstructed_span,
            "authority": "unique_exact_payload_branch_span",
            "quote_pair": [opener, closer],
        },
        {
            "operation": "remove",
            "json_path": (
                f"/required_branches/{branch_index}/{escaped_fragment_key}"
            ),
            "before": before_fragment,
            "after": None,
            "authority": "schema_external_quote_fragment",
            "quote_pair": [opener, closer],
        },
    ]
    concepts = (intermediate.get("required_concepts") or {}).get("all_of") or []
    concept_index_by_key = {
        _concept_key_normalized(concept.get("concept_key")): index
        for index, concept in enumerate(concepts)
        if isinstance(concept, dict)
    }
    inverse_stage_actions: list[dict[str, Any]] = []
    for action in inverse_actions:
        index = concept_index_by_key.get(
            _concept_key_normalized(action.get("concept_key"))
        )
        if index is None:
            return raw, [], raw_errors
        inverse_stage_actions.append(
            {
                "operation": "replace",
                "json_path": f"/required_concepts/all_of/{index}/branch_labels",
                "before": action["before"],
                "after": action["after"],
                "authority": "required_branches",
            }
        )
    touched_paths = {
        action["json_path"]
        for action in [*quote_actions, *inverse_stage_actions]
    }
    if len(touched_paths) != len(quote_actions) + len(inverse_stage_actions):
        return raw, [], raw_errors

    raw_hash = _sha256_json(raw)
    intermediate_hash = _sha256_json(intermediate)
    final_hash = _sha256_json(projected)
    stages = [
        {
            "stage": "quote_fragment_branch_reconstruction",
            "policy": QUOTE_FRAGMENT_BRANCH_RECONSTRUCTION_POLICY,
            "input_review_sha256": raw_hash,
            "output_review_sha256": intermediate_hash,
            "actions": quote_actions,
        },
        {
            "stage": "inverse_branch_labels",
            "policy": INVERSE_EDGE_CANONICALIZATION_POLICY,
            "input_review_sha256": intermediate_hash,
            "output_review_sha256": final_hash,
            "actions": inverse_stage_actions,
        },
    ]
    return projected, stages, raw_errors


def _canonicalize_saved_exact_payload_branch_span_quote_prefix(
    value: dict[str, Any], semantic_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Replace one quote-boundary-truncated span with its exact payload value.

    This saved-only projector does not infer or compose text.  The raw value
    must end immediately before one finite opening quote, while the immutable
    request payload supplies the byte-for-byte replacement.  Every other
    review field, including branch concept edges, remains untouched.
    """

    raw = deepcopy(value)
    raw_errors = _validate_review(raw)
    raw_errors.extend(_validate_against_payload(raw, semantic_payload))
    branches = raw.get("required_branches")
    payload_branches = semantic_payload.get("branches")
    if (
        raw.get("verdict") not in {"revise", "block"}
        or raw.get("answerability") != "answer"
        or _review_schema_extra_paths(raw)
        or not isinstance(branches, list)
        or not branches
        or not isinstance(payload_branches, list)
        or len(branches) != len(payload_branches)
        or any(not isinstance(item, dict) for item in branches)
        or any(not isinstance(item, dict) for item in payload_branches)
        or any(set(item) != _BRANCH_ALLOWED_FIELDS for item in branches)
    ):
        return raw, [], raw_errors

    raw_labels = [str(item.get("branch_label") or "") for item in branches]
    payload_labels = [
        str(item.get("branch_label") or "") for item in payload_branches
    ]
    if (
        raw_labels != payload_labels
        or any(not label for label in payload_labels)
        or len(set(payload_labels)) != len(payload_labels)
        or any(
            not isinstance(item.get("required_concept_keys"), list)
            or not item["required_concept_keys"]
            or any(
                not isinstance(key, str) or not key.strip()
                for key in item["required_concept_keys"]
            )
            for item in branches
        )
    ):
        return raw, [], raw_errors

    mismatches: list[tuple[int, str, str, str]] = []
    for index, (branch, payload_branch) in enumerate(
        zip(branches, payload_branches, strict=True)
    ):
        raw_span = branch.get("query_span")
        expected_span = payload_branch.get("query_span")
        if not isinstance(raw_span, str) or not isinstance(expected_span, str):
            return raw, [], raw_errors
        if raw_span != expected_span:
            mismatches.append((index, raw_labels[index], raw_span, expected_span))
    if len(mismatches) != 1:
        return raw, [], raw_errors
    branch_index, branch_label, raw_span, expected_span = mismatches[0]
    expected_raw_errors = [
        f"branch query_span differs from payload for {branch_label}",
        "required branch labels/query spans must exactly match payload",
    ]
    if (
        raw_errors != expected_raw_errors
        or not raw_span
        or len(raw_span) >= len(expected_span)
        or not expected_span.startswith(raw_span)
    ):
        return raw, [], raw_errors

    suffix = expected_span[len(raw_span) :]
    matches: list[tuple[str, str]] = []
    for opener, closer in QUOTE_FRAGMENT_PAIRS:
        if not suffix.startswith(opener):
            continue
        after_opener = suffix[len(opener) :]
        if after_opener.count(closer) != 1:
            continue
        closer_index = after_opener.index(closer)
        quoted_content = after_opener[:closer_index]
        query_tail = after_opener[closer_index + len(closer) :]
        if quoted_content and query_tail.strip():
            matches.append((opener, closer))
    if len(matches) != 1:
        return raw, [], raw_errors

    opener, closer = matches[0]
    projected = deepcopy(raw)
    projected["required_branches"][branch_index]["query_span"] = expected_span
    final_errors = _validate_review(projected)
    final_errors.extend(_validate_against_payload(projected, semantic_payload))
    final_errors.extend(_review_materialization_errors(projected, semantic_payload))
    if final_errors:
        return raw, [], raw_errors

    action = {
        "operation": "replace",
        "json_path": f"/required_branches/{branch_index}/query_span",
        "before": raw_span,
        "after": expected_span,
        "authority": "same_request_payload_exact_branch_span",
        "quote_pair": [opener, closer],
    }
    return (
        projected,
        [
            {
                "stage": "exact_payload_branch_span_quote_prefix_recovery",
                "policy": EXACT_PAYLOAD_BRANCH_SPAN_QUOTE_PREFIX_RECOVERY_POLICY,
                "input_review_sha256": _sha256_json(raw),
                "output_review_sha256": _sha256_json(projected),
                "actions": [action],
            }
        ],
        raw_errors,
    )


def _validate_exact_payload_branch_span_quote_prefix_recovery_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Independently replay the final-attempt exact-span saved recovery."""

    audit = response.get(
        "saved_response_exact_payload_branch_span_quote_prefix_recovery"
    )
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["exact payload branch-span recovery audit must be an object"]
    expected_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "stages",
        "selection",
        "source",
    }
    errors: list[str] = []
    if set(audit) != expected_keys:
        errors.append("exact payload branch-span recovery audit schema is invalid")
    if (
        response.get("runner_version") != RUNNER_VERSION
        or audit.get("policy")
        != EXACT_PAYLOAD_BRANCH_SPAN_QUOTE_PREFIX_RECOVERY_POLICY
    ):
        errors.append("exact payload branch-span recovery policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("exact payload branch-span recovery must be saved-only")
    review = response.get("review")
    if not isinstance(review, dict):
        errors.append("exact payload branch-span response must contain review")
        return errors
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("exact payload branch-span projected review hash mismatch")
    source = audit.get("source")
    if not isinstance(source, dict):
        errors.append("exact payload branch-span source must be an object")
        return errors
    if source_response_rows is None:
        errors.append("exact payload branch-span recovery requires append-only response log")
        return errors
    source_row_hash = str(source.get("response_row_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_row_hash):
        errors.append("exact payload branch-span source-row hash is invalid")
        return errors
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append("exact payload branch-span source row is not present exactly once")
        return errors
    source_row = matching_rows[0]
    replay = _recover_saved_exact_payload_branch_span_quote_prefix(
        row={
            "custom_id": response.get("custom_id"),
            "prompt_sha256": response.get("prompt_sha256"),
        },
        source_row=source_row,
        semantic_payload=semantic_payload,
    )
    if replay is None:
        errors.append("exact payload branch-span recovery source is not replayable")
        return errors
    expected_audit = replay[
        "saved_response_exact_payload_branch_span_quote_prefix_recovery"
    ]
    if audit != expected_audit:
        errors.append("exact payload branch-span audit/hash/action replay mismatch")
    if review != replay.get("review"):
        errors.append("exact payload branch-span selected final review mismatch")
    if response.get("response_sha256") != replay.get("response_sha256"):
        errors.append("exact payload branch-span response hash mismatch")
    if (
        response.get("attempt") != 0
        or response.get("transport") != "saved-response-recovery-no-api"
        or response.get("usage_is_lower_bound") is not False
        or response.get("usage")
        != {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        or response.get("successful_response_usage") != {}
        or response.get("attempt_usage") != []
    ):
        errors.append("exact payload branch-span zero-API receipt is invalid")
    if response.get("prior_response_recovery") != replay.get(
        "prior_response_recovery"
    ):
        errors.append("exact payload branch-span recovery lineage mismatch")
    return errors


def _validate_abstain_factual_constraint_recovery_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Independently replay the final-attempt abstention projection."""

    audit = response.get("saved_response_abstain_factual_constraint_recovery")
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["abstain factual-constraint recovery audit must be an object"]
    expected_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "stages",
        "attempt_shape_receipts",
        "selection",
        "source",
    }
    errors: list[str] = []
    if set(audit) != expected_keys:
        errors.append("abstain factual-constraint recovery audit schema is invalid")
    expected_policy_by_runner = {
        "2.26.0-saved-abstain-factual-constraint-recovery": (
            LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        ),
        "2.29.0-real-sse-calibration-ledger": (
            ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        ),
        "2.32.0-v229-formal-ledger-strict-scalars": (
            ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
        ),
    }
    expected_policy = expected_policy_by_runner.get(
        str(response.get("runner_version") or "")
    )
    if expected_policy is None or audit.get("policy") != expected_policy:
        errors.append("abstain factual-constraint recovery policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("abstain factual-constraint recovery must be saved-only")
    review = response.get("review")
    if not isinstance(review, dict):
        errors.append("abstain factual-constraint response must contain review")
        return errors
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("abstain factual-constraint projected review hash mismatch")
    source = audit.get("source")
    if not isinstance(source, dict):
        errors.append("abstain factual-constraint source must be an object")
        return errors
    if source_response_rows is None:
        errors.append(
            "abstain factual-constraint recovery requires append-only response log"
        )
        return errors
    source_row_hash = str(source.get("response_row_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_row_hash):
        errors.append("abstain factual-constraint source-row hash is invalid")
        return errors
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append(
            "abstain factual-constraint source row is not present exactly once"
        )
        return errors
    latest_failed_rows = [
        row
        for row in source_response_rows
        if row.get("status") == "failed"
        and row.get("custom_id") == response.get("custom_id")
        and row.get("prompt_sha256") == response.get("prompt_sha256")
    ]
    if not latest_failed_rows or matching_rows[0] is not latest_failed_rows[-1]:
        errors.append("abstain factual-constraint source row is not the latest failed row")
        return errors
    replay = _recover_saved_abstain_factual_constraints(
        row={
            "custom_id": response.get("custom_id"),
            "prompt_sha256": response.get("prompt_sha256"),
        },
        source_row=matching_rows[0],
        semantic_payload=semantic_payload,
        recovery_policy=expected_policy,
        target_runner_version=str(response.get("runner_version") or ""),
    )
    if replay is None:
        errors.append("abstain factual-constraint recovery source is not replayable")
        return errors
    if audit != replay.get("saved_response_abstain_factual_constraint_recovery"):
        errors.append("abstain factual-constraint audit/hash/action replay mismatch")
    if review != replay.get("review"):
        errors.append("abstain factual-constraint selected final review mismatch")
    if response.get("response_sha256") != replay.get("response_sha256"):
        errors.append("abstain factual-constraint response hash mismatch")
    if (
        response.get("attempt") != 0
        or response.get("transport") != "saved-response-recovery-no-api"
        or response.get("usage_is_lower_bound") is not False
        or response.get("usage")
        != {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        or response.get("successful_response_usage") != {}
        or response.get("attempt_usage") != []
    ):
        errors.append("abstain factual-constraint zero-API receipt is invalid")
    if response.get("prior_response_recovery") != replay.get(
        "prior_response_recovery"
    ):
        errors.append("abstain factual-constraint recovery lineage mismatch")
    return errors


def _validate_redundant_field_coverage_composition_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Independently replay the saved final-attempt format recovery."""

    audit = response.get(
        "saved_response_redundant_field_coverage_composition"
    )
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["redundant-field coverage composition audit must be an object"]
    expected_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "intermediate_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "stages",
        "attempt_shape_receipts",
        "selection",
        "source",
    }
    errors: list[str] = []
    if set(audit) != expected_keys:
        errors.append("redundant-field coverage composition audit schema is invalid")
    if (
        response.get("runner_version") != RUNNER_VERSION
        or audit.get("policy") != REDUNDANT_FIELD_COVERAGE_COMPOSITION_POLICY
    ):
        errors.append("redundant-field coverage composition policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("redundant-field coverage composition must be saved-only")
    review = response.get("review")
    if not isinstance(review, dict):
        errors.append("redundant-field coverage response must contain review")
        return errors
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("redundant-field coverage projected review hash mismatch")
    source = audit.get("source")
    if not isinstance(source, dict):
        errors.append("redundant-field coverage source must be an object")
        return errors
    if source_response_rows is None:
        errors.append("redundant-field coverage recovery requires append-only response log")
        return errors
    source_row_hash = str(source.get("response_row_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_row_hash):
        errors.append("redundant-field coverage source-row hash is invalid")
        return errors
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append(
            "redundant-field coverage source row is not present exactly once"
        )
        return errors
    source_row = matching_rows[0]
    replay = _recover_saved_redundant_field_coverage_composition(
        row={
            "custom_id": response.get("custom_id"),
            "prompt_sha256": response.get("prompt_sha256"),
        },
        source_row=source_row,
        semantic_payload=semantic_payload,
    )
    if replay is None:
        errors.append("redundant-field coverage recovery source is not replayable")
        return errors
    expected_audit = replay[
        "saved_response_redundant_field_coverage_composition"
    ]
    if audit != expected_audit:
        errors.append("redundant-field coverage audit/hash/action replay mismatch")
    if review != replay.get("review"):
        errors.append("redundant-field coverage selected final review mismatch")
    if response.get("response_sha256") != replay.get("response_sha256"):
        errors.append("redundant-field coverage response hash mismatch")
    if (
        response.get("attempt") != 0
        or response.get("transport") != "saved-response-recovery-no-api"
        or response.get("usage_is_lower_bound") is not False
        or response.get("usage")
        != {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        or response.get("successful_response_usage") != {}
        or response.get("attempt_usage") != []
    ):
        errors.append("redundant-field coverage zero-API receipt is invalid")
    if response.get("prior_response_recovery") != replay.get(
        "prior_response_recovery"
    ):
        errors.append("redundant-field coverage recovery lineage mismatch")
    return errors


def _validate_quote_fragment_inverse_composition_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    audit = response.get(
        "same_response_quote_fragment_inverse_canonicalization_composition"
    )
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["quote-fragment inverse composition audit must be an object"]
    expected_audit_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "intermediate_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "stages",
        "source",
    }
    errors: list[str] = []
    if set(audit) != expected_audit_keys:
        errors.append("quote-fragment inverse composition audit schema is invalid")
    if audit.get("policy") != QUOTE_FRAGMENT_INVERSE_COMPOSITION_POLICY:
        errors.append("quote-fragment inverse composition policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("quote-fragment inverse composition must be zero-API saved recovery")
    review = response.get("review")
    if not isinstance(review, dict):
        errors.append("quote-fragment inverse response must contain review")
        return errors
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("quote-fragment inverse projected review hash mismatch")
    source = audit.get("source")
    if not isinstance(source, dict):
        errors.append("quote-fragment inverse source must be an object")
        return errors
    if (
        source.get("custom_id") != response.get("custom_id")
        or source.get("prompt_sha256") != response.get("prompt_sha256")
        or source.get("raw_review_sha256") != audit.get("raw_review_sha256")
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
        )
    ):
        errors.append("quote-fragment inverse source lineage/hash is invalid")
        return errors
    if source_response_rows is None:
        errors.append("quote-fragment inverse requires append-only response log")
        return errors
    source_row_hash = str(source["response_row_sha256"])
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append("quote-fragment inverse source row is not present exactly once")
        return errors
    source_row = matching_rows[0]
    if (
        source_row.get("custom_id") != response.get("custom_id")
        or source_row.get("prompt_sha256") != response.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
    ):
        errors.append("quote-fragment inverse source row/quota lineage is invalid")
        return errors
    matching_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict)
        and item.get("attempt") == source.get("source_attempt")
        and isinstance(item.get("review"), dict)
        and _sha256_json(item["review"]) == audit.get("raw_review_sha256")
    ]
    if len(matching_attempts) != 1:
        errors.append("quote-fragment inverse source attempt is not present exactly once")
        return errors
    source_attempt = matching_attempts[0]
    if _invalid_attempt_has_structural_normalization(
        source_attempt
    ) or not _complete_terminal_semantic_attempt(
        source_attempt, source_row, semantic_payload
    ):
        errors.append("quote-fragment inverse source attempt is not complete terminal semantic")
        return errors
    latest_eligible_attempt: dict[str, Any] | None = None
    for item in reversed(source_row.get("invalid_attempts") or []):
        if not isinstance(item, dict) or not isinstance(item.get("review"), dict):
            continue
        candidate_review = deepcopy(item["review"])
        candidate_hash = _sha256_json(candidate_review)
        if (
            item.get("review_sha256") != candidate_hash
            or _invalid_attempt_has_structural_normalization(item)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
        ):
            continue
        _candidate_projected, candidate_stages, candidate_errors = (
            _canonicalize_saved_quote_fragment_inverse_composition(
                candidate_review, semantic_payload
            )
        )
        if candidate_stages and item.get("validation_errors") == candidate_errors:
            latest_eligible_attempt = item
            break
    if (
        latest_eligible_attempt is None
        or latest_eligible_attempt.get("attempt") != source.get("source_attempt")
        or _sha256_json(latest_eligible_attempt["review"])
        != audit.get("raw_review_sha256")
    ):
        errors.append("quote-fragment inverse source is not the latest eligible attempt")
        return errors
    raw_review = deepcopy(source_attempt["review"])
    projected, stages, raw_errors = (
        _canonicalize_saved_quote_fragment_inverse_composition(
            raw_review, semantic_payload
        )
    )
    if not stages or projected != review:
        errors.append("quote-fragment inverse composition is not reproducible")
    if source_attempt.get("validation_errors") != raw_errors:
        errors.append("quote-fragment inverse source validation errors mismatch")
    if audit.get("raw_validation_errors") != raw_errors:
        errors.append("quote-fragment inverse raw validation errors mismatch")
    if audit.get("stages") != stages:
        errors.append("quote-fragment inverse stage/order/hash audit mismatch")
    if stages and audit.get("intermediate_review_sha256") != stages[0].get(
        "output_review_sha256"
    ):
        errors.append("quote-fragment inverse intermediate hash mismatch")
    if source.get("validation_errors") != raw_errors:
        errors.append("quote-fragment inverse source error receipt mismatch")
    return errors


def _validate_nonsemantic_canonicalization_composition_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    audit = response.get(
        "same_response_nonsemantic_canonicalization_composition"
    )
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["nonsemantic canonicalization composition audit must be an object"]
    expected_audit_keys = {
        "policy",
        "api_call_performed",
        "raw_review_sha256",
        "intermediate_review_sha256",
        "projected_review_sha256",
        "raw_validation_errors",
        "stages",
        "source",
    }
    errors: list[str] = []
    if set(audit) != expected_audit_keys:
        errors.append("nonsemantic composition audit schema is invalid")
    runner_version = str(response.get("runner_version") or "")
    legacy_policy_allowed = (
        LEGACY_NONSEMANTIC_COMPOSITION_POLICY_BY_RUNNER.get(runner_version)
        == audit.get("policy")
    )
    current_policy_allowed = (
        runner_version == RUNNER_VERSION
        and audit.get("policy")
        == NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY
    )
    if not current_policy_allowed and not legacy_policy_allowed:
        errors.append("nonsemantic composition policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("nonsemantic composition must be zero-API saved recovery")
    review = response.get("review")
    if not isinstance(review, dict):
        errors.append("nonsemantic composition response must contain review")
        return errors
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("nonsemantic composition projected review hash mismatch")
    source = audit.get("source")
    if not isinstance(source, dict):
        errors.append("nonsemantic composition source must be an object")
        return errors
    if (
        source.get("custom_id") != response.get("custom_id")
        or source.get("prompt_sha256") != response.get("prompt_sha256")
        or source.get("raw_review_sha256") != audit.get("raw_review_sha256")
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
        )
    ):
        errors.append("nonsemantic composition source lineage/hash is invalid")
        return errors
    if source_response_rows is None:
        errors.append("nonsemantic composition requires append-only response log")
        return errors
    source_row_hash = str(source["response_row_sha256"])
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append("nonsemantic composition source row is not present exactly once")
        return errors
    source_row = matching_rows[0]
    if (
        source_row.get("custom_id") != response.get("custom_id")
        or source_row.get("prompt_sha256") != response.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
    ):
        errors.append("nonsemantic composition source row/quota lineage is invalid")
        return errors
    matching_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict)
        and item.get("attempt") == source.get("source_attempt")
        and isinstance(item.get("review"), dict)
        and _sha256_json(item["review"]) == audit.get("raw_review_sha256")
    ]
    if len(matching_attempts) != 1:
        errors.append("nonsemantic composition source attempt is not present exactly once")
        return errors
    source_attempt = matching_attempts[0]
    if _invalid_attempt_has_structural_normalization(
        source_attempt
    ) or not _complete_terminal_semantic_attempt(
        source_attempt, source_row, semantic_payload
    ):
        errors.append("nonsemantic composition source attempt is not complete terminal semantic")
        return errors
    raw_review = deepcopy(source_attempt["review"])
    projected, stages, raw_errors = _canonicalize_saved_nonsemantic_composition(
        raw_review, semantic_payload
    )
    expected_stages = deepcopy(stages)
    if legacy_policy_allowed and expected_stages:
        coverage_items = raw_review.get("evidence_concept_coverage") or []
        canonicalized_labels: set[str] = set()
        for action in expected_stages[0].get("actions") or []:
            path_match = re.fullmatch(
                r"/evidence_concept_coverage/(\d+)/status",
                str(action.get("json_path") or ""),
            )
            if path_match is None:
                continue
            coverage_index = int(path_match.group(1))
            if coverage_index >= len(coverage_items):
                continue
            coverage_item = coverage_items[coverage_index]
            if isinstance(coverage_item, dict):
                canonicalized_labels.add(
                    str(coverage_item.get("evidence_label") or "")
                )
        if canonicalized_labels & _constraint_evidence_labels(raw_review):
            errors.append(
                "legacy nonsemantic composition cannot claim constraint-only "
                "evidence semantics"
            )
        expected_stages[0]["policy"] = (
            LEGACY_COVERAGE_STATUS_CANONICALIZATION_POLICY_V3
        )
    if not stages or projected != review:
        errors.append("nonsemantic composition is not reproducible")
    if source_attempt.get("validation_errors") != raw_errors:
        errors.append("nonsemantic composition source validation errors mismatch")
    if audit.get("raw_validation_errors") != raw_errors:
        errors.append("nonsemantic composition raw validation errors mismatch")
    if audit.get("stages") != expected_stages:
        errors.append("nonsemantic composition stage/order/hash audit mismatch")
    if stages and audit.get("intermediate_review_sha256") != stages[0].get(
        "output_review_sha256"
    ):
        errors.append("nonsemantic composition intermediate hash mismatch")
    if source.get("validation_errors") != raw_errors:
        errors.append("nonsemantic composition source error receipt mismatch")
    return errors


def _validate_coverage_status_canonicalization_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    audit = response.get("same_response_coverage_status_canonicalization")
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["same-response coverage-status canonicalization audit must be an object"]
    review = response.get("review")
    if not isinstance(review, dict):
        return ["coverage-status canonicalized response must contain a review object"]
    errors: list[str] = []
    runner_version = str(response.get("runner_version") or "")
    legacy_policy_allowed = (
        LEGACY_COVERAGE_STATUS_POLICY_BY_RUNNER.get(runner_version)
        == audit.get("policy")
    )
    current_policy_allowed = (
        runner_version == RUNNER_VERSION
        and audit.get("policy") == COVERAGE_STATUS_CANONICALIZATION_POLICY
    )
    if not current_policy_allowed and not legacy_policy_allowed:
        errors.append("coverage-status canonicalization policy mismatch")
    api_call_performed = audit.get("api_call_performed")
    if api_call_performed not in {True, False}:
        errors.append("coverage-status canonicalization API-call flag must be boolean")
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("coverage-status projected review hash mismatch")
    raw_hash = str(audit.get("raw_review_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        errors.append("coverage-status raw review hash is invalid")
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("raw_review_sha256") != raw_hash
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
    ):
        errors.append("coverage-status source attempt/hash audit is invalid")
    actions = audit.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("coverage-status canonicalization actions must be non-empty")
        return errors

    reconstructed = deepcopy(review)
    coverage = reconstructed.get("evidence_concept_coverage") or []
    seen_indexes: set[int] = set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {
            "coverage_index",
            "evidence_label",
            "field",
            "before",
            "after",
            "required_concept_keys",
            "authority",
        }:
            errors.append("coverage-status action schema is invalid")
            continue
        index = action.get("coverage_index")
        if (
            type(index) is not int
            or index < 0
            or index >= len(coverage)
            or index in seen_indexes
        ):
            errors.append("coverage-status action index is invalid")
            continue
        seen_indexes.add(index)
        item = coverage[index]
        if (
            not isinstance(item, dict)
            or action.get("field") != "status"
            or action.get("before") not in COVERAGE_STATUS_ZERO_SUPPORT_ALIASES
            or action.get("after") != "uncovered"
            or action.get("required_concept_keys") != []
            or action.get("authority")
            != (
                "zero_direct_support_edges"
                if legacy_policy_allowed
                else "zero_direct_concept_support_edges"
            )
            or item.get("evidence_label") != action.get("evidence_label")
            or item.get("required_concept_keys") != []
            or item.get("status") != "uncovered"
        ):
            errors.append("coverage-status action cannot be reversed exactly")
            continue
        item["status"] = action["before"]

    if errors:
        return errors
    if _sha256_json(reconstructed) != raw_hash:
        errors.append("coverage-status reconstructed raw review hash mismatch")
        return errors
    expected_review, expected_actions, expected_raw_errors = (
        _canonicalize_same_response_coverage_statuses(
            reconstructed, semantic_payload
        )
    )
    if legacy_policy_allowed:
        canonicalized_labels = {
            str(action.get("evidence_label") or "") for action in actions
        }
        if canonicalized_labels & _constraint_evidence_labels(reconstructed):
            errors.append(
                "legacy coverage-status audit cannot claim constraint-only "
                "evidence semantics"
            )
        expected_actions = deepcopy(expected_actions)
        for action in expected_actions:
            action["authority"] = "zero_direct_support_edges"
    if expected_review != review or expected_actions != actions:
        errors.append("coverage-status canonicalization is not reproducible")
    if audit.get("raw_validation_errors") != expected_raw_errors:
        errors.append("coverage-status raw validation-error audit mismatch")

    if api_call_performed is False:
        if not isinstance(source, dict):
            return errors
        if (
            source.get("custom_id") != response.get("custom_id")
            or source.get("prompt_sha256") != response.get("prompt_sha256")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
            )
            or source.get("validation_errors") != expected_raw_errors
        ):
            errors.append(
                "saved coverage-status source row/prompt/prior-errors audit is invalid"
            )
        elif source_response_rows is None:
            errors.append(
                "saved coverage-status canonicalization requires the append-only "
                "source response log"
            )
        else:
            source_row_hash = str(source["response_row_sha256"])
            matching_rows = [
                row
                for row in source_response_rows
                if _sha256_json(row) == source_row_hash
            ]
            source_attempt = int(source["source_attempt"])
            matching_attempts = [
                item
                for row in matching_rows
                if row.get("status") == "failed"
                and row.get("custom_id") == response.get("custom_id")
                and row.get("prompt_sha256") == response.get("prompt_sha256")
                for item in row.get("invalid_attempts") or []
                if isinstance(item, dict)
                and item.get("attempt") == source_attempt
                and isinstance(item.get("review"), dict)
                and _sha256_json(item["review"]) == raw_hash
                and item.get("validation_errors") == expected_raw_errors
                and not _invalid_attempt_has_structural_normalization(item)
                and not _validate_invalid_attempt_raw_normalization_provenance(
                    item, row, semantic_payload
                )
            ]
            if len(matching_rows) != 1 or len(matching_attempts) != 1:
                errors.append(
                    "saved coverage-status source row/attempt is not present exactly "
                    "once in the append-only response log"
                )
    return errors


def _validate_quota_exhausted_verdict_block_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    audit_key: str,
    audit_policy: str,
    projector: Callable[
        [dict[str, Any], dict[str, Any]],
        tuple[dict[str, Any], list[dict[str, Any]], list[str]],
    ],
    action_keys: set[str],
    action_authority: str,
    label: str,
    source_response_rows: Sequence[dict[str, Any]] | None,
    legacy_consensus_runner_versions: frozenset[str] = frozenset(),
) -> list[str]:
    audit = response.get(audit_key)
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return [f"{label} audit must be an object"]
    review = response.get("review")
    if not isinstance(review, dict):
        return [f"{label} response must contain review"]
    errors: list[str] = []
    if audit.get("policy") != audit_policy:
        errors.append(f"{label} policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append(f"{label} must be a zero-API saved-response projection")
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append(f"{label} projected review hash mismatch")
    raw_hash = str(audit.get("raw_review_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        errors.append(f"{label} raw review hash is invalid")
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("raw_review_sha256") != raw_hash
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
    ):
        errors.append(f"{label} source attempt/hash is invalid")
    actions = audit.get("actions")
    if not isinstance(actions, list) or len(actions) != 1:
        errors.append(f"{label} requires exactly one action")
        return errors
    action = actions[0]
    if (
        not isinstance(action, dict)
        or set(action) != action_keys
        or action.get("field") != "verdict"
        or action.get("before") != "revise"
        or action.get("after") != "block"
        or action.get("authority") != action_authority
        or review.get("verdict") != "block"
    ):
        errors.append(f"{label} action schema is invalid")
        return errors

    reconstructed = deepcopy(review)
    reconstructed["verdict"] = "revise"
    if _sha256_json(reconstructed) != raw_hash:
        errors.append(f"{label} reconstructed raw hash mismatch")
        return errors
    expected_review, expected_actions, expected_raw_errors = projector(
        reconstructed, semantic_payload
    )
    if expected_review != review or expected_actions != actions:
        errors.append(f"{label} is not reproducible")
    if audit.get("raw_validation_errors") != expected_raw_errors:
        errors.append(f"{label} raw validation errors mismatch")

    if not isinstance(source, dict):
        return errors
    if (
        source.get("custom_id") != response.get("custom_id")
        or source.get("prompt_sha256") != response.get("prompt_sha256")
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
        )
        or source.get("validation_errors") != expected_raw_errors
    ):
        errors.append(f"saved {label} source row/prompt/errors invalid")
        return errors
    if source_response_rows is None:
        errors.append(f"saved {label} requires the append-only source response log")
        return errors
    source_row_hash = str(source["response_row_sha256"])
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append(f"saved {label} source row is not present exactly once")
        return errors
    source_row = matching_rows[0]
    if (
        source_row.get("custom_id") != response.get("custom_id")
        or source_row.get("prompt_sha256") != response.get("prompt_sha256")
    ):
        errors.append(f"saved {label} source row lineage mismatch")
        return errors
    legacy_allowed = response.get("runner_version") in legacy_consensus_runner_versions
    consensus_result = _quota_exhausted_verdict_block_consensus(
        source_row,
        semantic_payload,
        projector=projector,
        require_target_semantic_consensus=not legacy_allowed,
    )
    if consensus_result is None:
        errors.append(f"saved {label} lacks quota-exhausted multi-response consensus")
        return errors
    matching_members = [
        member
        for member in consensus_result["members"]
        if member["attempt"] == source.get("source_attempt")
        and member["raw_review_sha256"] == raw_hash
        and member["projected_review"] == review
        and member["raw_validation_errors"] == expected_raw_errors
    ]
    if len(matching_members) != 1:
        errors.append(f"saved {label} selected source attempt is outside consensus")
    consensus = audit.get("consensus")
    if consensus is None and legacy_allowed:
        # The legacy row binds the entire failed response row by SHA.  Replaying
        # that row above proves all attempts even though the derived member list
        # was not duplicated inside the earlier success row.
        return errors
    if consensus != consensus_result["consensus"]:
        errors.append(f"saved {label} consensus audit mismatch")
    final = consensus_result["final"]
    if (
        source.get("source_attempt") != final["attempt"]
        or raw_hash != final["raw_review_sha256"]
        or review != final["projected_review"]
    ):
        errors.append(f"saved {label} must project the final consensus response")
    return errors


def _validate_unsupported_required_concept_block_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    legacy_runner = (
        response.get("runner_version")
        == "2.16.0-unsupported-required-concept-block"
    )
    return _validate_quota_exhausted_verdict_block_audit(
        response,
        semantic_payload,
        audit_key="same_response_unsupported_required_concept_block",
        audit_policy=(
            LEGACY_UNSUPPORTED_REQUIRED_CONCEPT_BLOCK_POLICY
            if legacy_runner
            else UNSUPPORTED_REQUIRED_CONCEPT_BLOCK_POLICY
        ),
        projector=_canonicalize_same_response_unsupported_required_concept_block,
        action_keys={
            "field",
            "before",
            "after",
            "unsupported_concept_keys",
            "unsupported_branch_labels",
            "authority",
        },
        action_authority="required_branch_has_no_directly_supported_concept",
        label="unsupported-required-concept block",
        source_response_rows=source_response_rows,
        legacy_consensus_runner_versions=frozenset(
            {"2.16.0-unsupported-required-concept-block"}
        ),
    )


def _validate_empty_required_branch_block_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    return _validate_quota_exhausted_verdict_block_audit(
        response,
        semantic_payload,
        audit_key="multi_response_empty_required_branch_block",
        audit_policy=EMPTY_REQUIRED_BRANCH_BLOCK_POLICY,
        projector=_canonicalize_same_response_empty_required_branch_block,
        action_keys={
            "field",
            "before",
            "after",
            "empty_branch_labels",
            "empty_branch_query_spans",
            "authority",
        },
        action_authority="context_preamble_has_no_required_answer_concept",
        label="empty-required-branch block",
        source_response_rows=source_response_rows,
    )


def _validate_saved_single_response_canonicalization_source(
    response: dict[str, Any],
    audit: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    expected_raw_errors: list[str],
    source_response_rows: Sequence[dict[str, Any]] | None,
    label: str,
    legacy_minimal_response_runner: str,
    legacy_minimal_source_runner: str,
) -> list[str]:
    """Bind a zero-API single-response projector to one immutable failed row."""

    if audit.get("api_call_performed") is not False:
        return []
    errors: list[str] = []
    source = audit.get("source")
    if not isinstance(source, dict):
        return [f"saved {label} source must be an object"]

    common_source_keys = {
        "response_row_sha256",
        "completed_at",
        "failure_max_attempt",
        "source_attempt",
        "raw_review_sha256",
        "validation_errors",
    }
    full_source_keys = {
        *common_source_keys,
        "custom_id",
        "prompt_sha256",
        "runner_version",
    }
    legacy_minimal_source_allowed = (
        response.get("runner_version") == legacy_minimal_response_runner
        and set(source) == common_source_keys
    )
    if set(source) != full_source_keys and not legacy_minimal_source_allowed:
        errors.append(f"saved {label} source schema is invalid")
    if (
        source.get("raw_review_sha256") != audit.get("raw_review_sha256")
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
        )
        or source.get("validation_errors") != expected_raw_errors
    ):
        errors.append(f"saved {label} source lineage/error receipt is invalid")
    if source_response_rows is None:
        errors.append(f"saved {label} requires the append-only response log")
        return errors

    source_row_hash = str(source.get("response_row_sha256") or "")
    matching_rows = [
        row for row in source_response_rows if _sha256_json(row) == source_row_hash
    ]
    if len(matching_rows) != 1:
        errors.append(f"saved {label} source row is not present exactly once")
        return errors
    source_row = matching_rows[0]
    if (
        source_row.get("status") != "failed"
        or source_row.get("custom_id") != response.get("custom_id")
        or source_row.get("prompt_sha256") != response.get("prompt_sha256")
        or source.get("completed_at") != source_row.get("completed_at")
        or source.get("failure_max_attempt") != source_row.get("attempt")
        or (
            legacy_minimal_source_allowed
            and source_row.get("runner_version") != legacy_minimal_source_runner
        )
        or (
            not legacy_minimal_source_allowed
            and (
                source.get("custom_id") != response.get("custom_id")
                or source.get("prompt_sha256") != response.get("prompt_sha256")
                or source.get("runner_version") != source_row.get("runner_version")
            )
        )
    ):
        errors.append(f"saved {label} source row lineage is invalid")

    source_attempt = source.get("source_attempt")
    raw_hash = str(audit.get("raw_review_sha256") or "")
    matching_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict)
        and item.get("attempt") == source_attempt
        and isinstance(item.get("review"), dict)
        and _sha256_json(item["review"]) == raw_hash
        and item.get("review_sha256") == raw_hash
    ]
    if len(matching_attempts) != 1:
        errors.append(f"saved {label} source attempt is not present exactly once")
        return errors
    source_item = matching_attempts[0]
    if _invalid_attempt_has_structural_normalization(source_item):
        errors.append(f"saved {label} source used structural normalization")
    if source_item.get("validation_errors") != expected_raw_errors:
        errors.append(f"saved {label} source validation-error receipt mismatch")
    provenance_errors = _validate_invalid_attempt_raw_normalization_provenance(
        source_item, source_row, semantic_payload
    )
    if provenance_errors:
        errors.append(
            f"saved {label} source raw-normalization provenance is invalid: "
            + "; ".join(provenance_errors)
        )
    return errors


def _validate_inverse_edge_canonicalization_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    audit = response.get("same_response_inverse_edge_canonicalization")
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["same-response inverse-edge canonicalization audit must be an object"]
    review = response.get("review")
    if not isinstance(review, dict):
        return ["canonicalized response must contain a review object"]
    errors: list[str] = []
    if audit.get("policy") != INVERSE_EDGE_CANONICALIZATION_POLICY:
        errors.append("inverse-edge canonicalization policy mismatch")
    if audit.get("api_call_performed") not in {True, False}:
        errors.append("inverse-edge canonicalization API-call flag must be boolean")
    projected_hash = _sha256_json(review)
    if audit.get("projected_review_sha256") != projected_hash:
        errors.append("inverse-edge projected review hash mismatch")
    raw_hash = str(audit.get("raw_review_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        errors.append("inverse-edge raw review hash is invalid")
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("raw_review_sha256") != raw_hash
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
    ):
        errors.append("inverse-edge source attempt/hash audit is invalid")
    actions = audit.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("inverse-edge canonicalization actions must be non-empty")
        return errors

    reconstructed = deepcopy(review)
    concepts = (reconstructed.get("required_concepts") or {}).get("all_of") or []
    by_key = {
        _concept_key_normalized(item.get("concept_key")): item
        for item in concepts
        if isinstance(item, dict)
    }
    seen_fields: set[tuple[str, str]] = set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {
            "concept_key",
            "field",
            "before",
            "after",
            "authority",
        }:
            errors.append("inverse-edge action schema is invalid")
            continue
        key = _concept_key_normalized(action.get("concept_key"))
        field = str(action.get("field") or "")
        authority = str(action.get("authority") or "")
        expected_authority = {
            "branch_labels": "required_branches",
            "evidence_labels": "evidence_concept_coverage",
        }.get(field)
        concept = by_key.get(key)
        before = action.get("before")
        after = action.get("after")
        if (
            concept is None
            or expected_authority is None
            or authority != expected_authority
            or not isinstance(before, list)
            or not isinstance(after, list)
            or before == after
            or concept.get(field) != after
            or (key, field) in seen_fields
        ):
            errors.append("inverse-edge action cannot be reversed exactly")
            continue
        seen_fields.add((key, field))
        concept[field] = deepcopy(before)

    if errors:
        return errors
    if _sha256_json(reconstructed) != raw_hash:
        errors.append("inverse-edge reconstructed raw review hash mismatch")
        return errors
    expected_review, expected_actions, expected_raw_errors = (
        _canonicalize_same_response_inverse_edges(reconstructed, semantic_payload)
    )
    if expected_review != review or expected_actions != actions:
        errors.append("inverse-edge canonicalization is not reproducible")
    if audit.get("raw_validation_errors") != expected_raw_errors:
        errors.append("inverse-edge raw validation-error audit mismatch")
    errors.extend(
        _validate_saved_single_response_canonicalization_source(
            response,
            audit,
            semantic_payload,
            expected_raw_errors=expected_raw_errors,
            source_response_rows=source_response_rows,
            label="inverse-edge canonicalization",
            legacy_minimal_response_runner="2.9.0-same-response-inverse-edge",
            legacy_minimal_source_runner="2.8.0-sse-strict-terminal-schema",
        )
    )
    return errors


def _validate_exact_variant_deduplication_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    audit = response.get("same_response_exact_variant_deduplication")
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["same-response exact-variant deduplication audit must be an object"]
    review = response.get("review")
    if not isinstance(review, dict):
        return ["deduplicated response must contain a review object"]
    errors: list[str] = []
    if audit.get("policy") != EXACT_VARIANT_DEDUPLICATION_POLICY:
        errors.append("exact-variant deduplication policy mismatch")
    if audit.get("api_call_performed") not in {True, False}:
        errors.append("exact-variant deduplication API-call flag must be boolean")
    if audit.get("projected_review_sha256") != _sha256_json(review):
        errors.append("exact-variant projected review hash mismatch")
    raw_hash = str(audit.get("raw_review_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        errors.append("exact-variant raw review hash is invalid")
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("raw_review_sha256") != raw_hash
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
    ):
        errors.append("exact-variant source attempt/hash audit is invalid")
    actions = audit.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("exact-variant deduplication actions must be non-empty")
        return errors

    reconstructed = deepcopy(review)
    concepts = (reconstructed.get("required_concepts") or {}).get("all_of") or []
    by_key = {
        _concept_key_normalized(item.get("concept_key")): item
        for item in concepts
        if isinstance(item, dict)
    }
    seen_keys: set[str] = set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {
            "concept_key",
            "field",
            "before",
            "after",
            "removed",
            "authority",
        }:
            errors.append("exact-variant action schema is invalid")
            continue
        key = _concept_key_normalized(action.get("concept_key"))
        concept = by_key.get(key)
        before = action.get("before")
        after = action.get("after")
        if (
            concept is None
            or key in seen_keys
            or action.get("field") != "any_of"
            or action.get("authority")
            != "normalized_exact_duplicate_first_occurrence"
            or not isinstance(before, list)
            or not isinstance(after, list)
            or before == after
            or concept.get("any_of") != after
            or not isinstance(action.get("removed"), list)
            or not action["removed"]
        ):
            errors.append("exact-variant action cannot be reversed exactly")
            continue
        seen_keys.add(key)
        concept["any_of"] = deepcopy(before)

    if errors:
        return errors
    if _sha256_json(reconstructed) != raw_hash:
        errors.append("exact-variant reconstructed raw review hash mismatch")
        return errors
    expected_review, expected_actions, expected_raw_errors = (
        _dedupe_same_response_exact_variants(reconstructed, semantic_payload)
    )
    if expected_review != review or expected_actions != actions:
        errors.append("exact-variant deduplication is not reproducible")
    if audit.get("raw_validation_errors") != expected_raw_errors:
        errors.append("exact-variant raw validation-error audit mismatch")
    errors.extend(
        _validate_saved_single_response_canonicalization_source(
            response,
            audit,
            semantic_payload,
            expected_raw_errors=expected_raw_errors,
            source_response_rows=source_response_rows,
            label="exact-variant deduplication",
            legacy_minimal_response_runner="2.10.0-same-response-exact-dedupe",
            legacy_minimal_source_runner="2.9.0-same-response-inverse-edge",
        )
    )
    return errors


def _validate_saved_strict_revalidation_audit(
    response: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    source_response_rows: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate zero-API adoption of one unchanged, previously saved response."""

    audit = response.get("saved_response_strict_revalidation")
    if audit is None:
        return []
    if not isinstance(audit, dict):
        return ["saved-response strict revalidation audit must be an object"]
    review = response.get("review")
    if not isinstance(review, dict):
        return ["saved-response strict revalidation must contain a review object"]
    errors: list[str] = []
    review_hash = _sha256_json(review)
    if audit.get("policy") != SAVED_STRICT_REVALIDATION_POLICY:
        errors.append("saved-response strict revalidation policy mismatch")
    if audit.get("api_call_performed") is not False:
        errors.append("saved-response strict revalidation must not perform an API call")
    if audit.get("raw_review_sha256") != review_hash:
        errors.append("saved-response raw review hash mismatch")
    if audit.get("projected_review_sha256") != review_hash:
        errors.append("saved-response projected review hash mismatch")
    if audit.get("validator_runner_version") != response.get("runner_version"):
        errors.append("saved-response validator runner version mismatch")
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("custom_id") != response.get("custom_id")
        or source.get("prompt_sha256") != response.get("prompt_sha256")
        or source.get("raw_review_sha256") != review_hash
        or not isinstance(source.get("source_attempt"), int)
        or int(source["source_attempt"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("response_row_sha256") or "")
        )
        or not isinstance(source.get("validation_errors"), list)
        or not source.get("validation_errors")
    ):
        errors.append("saved-response source attempt/hash/prior-errors audit is invalid")
    elif source_response_rows is None:
        errors.append(
            "saved-response strict revalidation requires the append-only source response log"
        )
    else:
        source_row_hash = str(source["response_row_sha256"])
        matching_rows = [
            row
            for row in source_response_rows
            if _sha256_json(row) == source_row_hash
        ]
        source_attempt = int(source["source_attempt"])
        matching_attempts = [
            item
            for row in matching_rows
            if row.get("status") == "failed"
            and row.get("custom_id") == response.get("custom_id")
            and row.get("prompt_sha256") == response.get("prompt_sha256")
            for item in row.get("invalid_attempts") or []
            if isinstance(item, dict)
            and item.get("attempt") == source_attempt
            and isinstance(item.get("review"), dict)
            and _sha256_json(item["review"]) == review_hash
            and item.get("validation_errors") == source.get("validation_errors")
            and not _invalid_attempt_has_structural_normalization(item)
            and not _validate_invalid_attempt_raw_normalization_provenance(
                item, row, semantic_payload
            )
        ]
        if len(matching_rows) != 1 or len(matching_attempts) != 1:
            errors.append(
                "saved-response source row/attempt is not present exactly once in the "
                "append-only response log"
            )
    current_errors = _validate_review(review)
    current_errors.extend(_validate_against_payload(review, semantic_payload))
    if current_errors:
        errors.append(
            "saved-response review no longer passes current strict validation: "
            + "; ".join(current_errors)
        )
    return errors


def _recover_quota_exhausted_verdict_block(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    projector: Callable[
        [dict[str, Any], dict[str, Any]],
        tuple[dict[str, Any], list[dict[str, Any]], list[str]],
    ],
    audit_key: str,
    audit_policy: str,
    recovery_mode: str,
) -> dict[str, Any] | None:
    consensus_result = _quota_exhausted_verdict_block_consensus(
        source_row, semantic_payload, projector=projector
    )
    if consensus_result is None:
        return None
    final = consensus_result["final"]
    projected = final["projected_review"]
    custom_id = str(row["custom_id"])
    source = {
        "custom_id": custom_id,
        "prompt_sha256": source_row.get("prompt_sha256"),
        "runner_version": source_row.get("runner_version"),
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": final["attempt"],
        "raw_review_sha256": final["raw_review_sha256"],
        "validation_errors": final["raw_validation_errors"],
    }
    audit = {
        "policy": audit_policy,
        "api_call_performed": False,
        "raw_review_sha256": final["raw_review_sha256"],
        "projected_review_sha256": final["projected_review_sha256"],
        "raw_validation_errors": final["raw_validation_errors"],
        "actions": final["actions"],
        "source": source,
        "consensus": consensus_result["consensus"],
    }
    return {
        "custom_id": custom_id,
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": RUNNER_VERSION,
        "transport": "saved-response-recovery-no-api",
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": final["projected_review_sha256"],
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "successful_response_usage": {},
        "attempt_usage": [],
        audit_key: audit,
        "prior_response_recovery": {
            "mode": recovery_mode,
            "api_call_performed": False,
            "sources": consensus_result["consensus"]["sources"],
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _saved_semantic_quota_exhausted(source_row: dict[str, Any]) -> bool:
    retry = source_row.get("retry_accounting") or {}
    quota = retry.get("semantic_failure_quota")
    failures = retry.get("semantic_failure_count")
    invalid_attempts = source_row.get("invalid_attempts") or []
    return bool(
        source_row.get("status") == "failed"
        and source_row.get("failure_termination") == "review_validation_error"
        and retry.get("quota_exhausted") == "semantic"
        and isinstance(quota, int)
        and quota >= 1
        and isinstance(failures, int)
        and failures >= quota
        and len(invalid_attempts) == failures
    )


def _redundant_field_stage_action_signature(
    stages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the content-free failure-shape signature for two fixed stages."""

    return [
        {
            "stage": str(stage.get("stage") or ""),
            "policy": str(stage.get("policy") or ""),
            "actions": [
                {
                    key: action.get(key)
                    for key in (
                        "operation",
                        "json_path",
                        "field",
                        "before_state",
                        "authority",
                    )
                    if key in action
                }
                for action in stage.get("actions") or []
                if isinstance(action, dict)
            ],
        }
        for stage in stages
    ]


def _expected_redundant_normalization_actions(
    reconstruction_actions: Sequence[dict[str, Any]],
) -> list[str]:
    prefixes = {
        "any_of": "self_filled_any_of",
        "evidence_labels": "crossref_filled_evidence_labels",
        "branch_labels": "crossref_filled_branch_labels",
    }
    result: list[str] = []
    for action in reconstruction_actions:
        field = str(action.get("field") or "")
        prefix = prefixes.get(field)
        key = _concept_key_normalized(action.get("concept_key"))
        if prefix is None or not key:
            return []
        result.append(f"{prefix}:{key}")
    return result


def _saved_redundant_attempt_receipts_valid(source_row: dict[str, Any]) -> bool:
    """Bind a saved format recovery to one complete bounded-retry ledger.

    Transport attempts may be interleaved with semantic attempts, but every
    failed attempt must occupy exactly one consecutive attempt number.  Only
    complete semantic attempts carry terminal usage and an invalid-review
    receipt; the final overall attempt must be the selected semantic attempt.
    """

    retry = source_row.get("retry_accounting") or {}
    semantic_quota = retry.get("semantic_failure_quota")
    transport_quota = retry.get("transport_failure_quota")
    hard_total_quota = retry.get("hard_total_attempt_quota")
    total_attempts = retry.get("total_attempts")
    transport_failures = retry.get("transport_failure_count")
    semantic_failures = retry.get("semantic_failure_count")
    if (
        retry.get("policy") != RETRY_ACCOUNTING_POLICY
        or not isinstance(semantic_quota, int)
        or semantic_quota < 2
        or not isinstance(transport_quota, int)
        or transport_quota != semantic_quota
        or not isinstance(hard_total_quota, int)
        or hard_total_quota != semantic_quota * 2
        or not isinstance(total_attempts, int)
        or total_attempts < 2
        or not isinstance(transport_failures, int)
        or transport_failures < 0
        or not isinstance(semantic_failures, int)
        or semantic_failures != semantic_quota
        or transport_failures >= transport_quota
        or transport_failures + semantic_failures != total_attempts
        or total_attempts > hard_total_quota
        or source_row.get("attempt") != total_attempts
    ):
        return False

    failure_attempts = source_row.get("failure_attempts") or []
    if (
        len(failure_attempts) != total_attempts
        or any(not isinstance(item, dict) for item in failure_attempts)
    ):
        return False
    expected_attempts = list(range(1, total_attempts + 1))
    if [item.get("attempt") for item in failure_attempts] != expected_attempts:
        return False
    semantic_attempts = [
        int(item["attempt"])
        for item in failure_attempts
        if item.get("failure_category") == "semantic"
    ]
    transport_attempts = [
        int(item["attempt"])
        for item in failure_attempts
        if item.get("failure_category") == "transport"
    ]
    if (
        len(semantic_attempts) != semantic_failures
        or len(transport_attempts) != transport_failures
        or len(semantic_attempts) + len(transport_attempts) != total_attempts
        or semantic_attempts[-1] != total_attempts
        or source_row.get("failure_termination")
        != failure_attempts[-1].get("termination")
    ):
        return False

    invalid_attempts = source_row.get("invalid_attempts") or []
    if (
        len(invalid_attempts) != semantic_failures
        or any(not isinstance(item, dict) for item in invalid_attempts)
        or [item.get("attempt") for item in invalid_attempts] != semantic_attempts
        or source_row.get("invalid_review")
        != invalid_attempts[-1].get("review")
    ):
        return False

    sse_integrity_attempts = source_row.get("sse_integrity_attempts")
    if (
        not isinstance(sse_integrity_attempts, list)
        or not sse_integrity_attempts
        or any(not isinstance(item, dict) for item in sse_integrity_attempts)
    ):
        return False
    expected_sse_attempts = [
        int(item["attempt"])
        for item in failure_attempts
        if item.get("failure_category") == "semantic"
        or item.get("error_type") == "SSEIntegrityError"
    ]
    if (
        [item.get("attempt") for item in sse_integrity_attempts]
        != expected_sse_attempts
        or source_row.get("sse_integrity") != sse_integrity_attempts[-1]
        or sse_integrity_attempts[-1].get("attempt") != total_attempts
        or invalid_attempts[-1].get("sse_integrity")
        != sse_integrity_attempts[-1]
    ):
        return False
    sse_by_attempt = {
        int(item["attempt"]): item for item in sse_integrity_attempts
    }
    if len(sse_by_attempt) != len(sse_integrity_attempts) or any(
        item.get("sse_integrity") != sse_by_attempt.get(int(item["attempt"]))
        for item in invalid_attempts
    ):
        return False

    attempt_usage = source_row.get("attempt_usage") or []
    if (
        len(attempt_usage) != semantic_failures
        or any(not isinstance(item, dict) for item in attempt_usage)
        or [item.get("attempt") for item in attempt_usage] != semantic_attempts
    ):
        return False
    for item in attempt_usage:
        if not all(
            isinstance(item.get(key), int) and int(item[key]) >= 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) or int(item["total_tokens"]) != int(item["prompt_tokens"]) + int(
            item["completion_tokens"]
        ):
            return False
    aggregate_usage = source_row.get("usage") or {}
    if any(
        aggregate_usage.get(key)
        != sum(int(item[key]) for item in attempt_usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return False
    if source_row.get("usage_is_lower_bound") is not (transport_failures > 0):
        return False
    return True


def _saved_v229_formal_failure_envelope_valid(
    source_row: dict[str, Any], semantic_payload: dict[str, Any]
) -> bool:
    """Replay the exact 2.29 formal-failure envelope used by 2.31 recovery."""

    if source_row.get("runner_version") != "2.29.0-real-sse-calibration-ledger":
        return False
    if set(source_row) != {
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
        "invalid_review",
        "invalid_attempts",
        "sse_integrity",
        "sse_integrity_attempts",
        "failure_termination",
        "failure_attempts",
        "attempt_usage",
        "usage",
        "sequence",
    }:
        return False
    try:
        completed_at = datetime.fromisoformat(str(source_row.get("completed_at") or ""))
    except ValueError:
        return False
    if (
        completed_at.tzinfo is None
        or completed_at.utcoffset() != UTC.utcoffset(None)
        or source_row.get("status") != "failed"
        or not isinstance(source_row.get("custom_id"), str)
        or not source_row["custom_id"]
        or not isinstance(source_row.get("model"), str)
        or not source_row["model"]
        or source_row.get("transport") != TRANSPORT
        or source_row.get("sse_completion_policy") != SSE_COMPLETION_POLICY
        or source_row.get("failure_termination") != "review_validation_error"
        or source_row.get("usage_is_lower_bound") is not False
        or type(source_row.get("attempt")) is not int
        or source_row["attempt"] < 1
        or type(source_row.get("sequence")) is not int
        or source_row["sequence"] < 1
        or not _saved_semantic_quota_exhausted(source_row)
        or not _saved_redundant_attempt_receipts_valid(source_row)
    ):
        return False

    retry = source_row["retry_accounting"]
    if set(retry) != {
        "policy",
        "transport_failure_quota",
        "semantic_failure_quota",
        "hard_total_attempt_quota",
        "total_attempts",
        "transport_failure_count",
        "semantic_failure_count",
        "quota_exhausted",
    } or any(
        type(retry.get(key)) is not int
        for key in (
            "transport_failure_quota",
            "semantic_failure_quota",
            "hard_total_attempt_quota",
            "total_attempts",
            "transport_failure_count",
            "semantic_failure_count",
        )
    ):
        return False
    aggregate_usage = source_row.get("usage") or {}
    if set(aggregate_usage) != {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    } or any(
        type(aggregate_usage.get(key)) is not int
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return False

    invalid_attempts = source_row["invalid_attempts"]
    failures = source_row["failure_attempts"]
    usages = source_row["attempt_usage"]
    sse_attempts = source_row["sse_integrity_attempts"]
    if not (
        len(invalid_attempts)
        == len(failures)
        == len(usages)
        == len(sse_attempts)
        == source_row["attempt"]
    ):
        return False
    expected_sse_keys = {
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
    expected_usage_integrity = {
        "status": "passed",
        "usage_event_count": 1,
        "required_token_fields_valid": True,
        "total_matches_prompt_plus_completion": True,
        "at_or_after_finish_reason_stop": True,
    }
    for attempt, (item, failure, usage, integrity) in enumerate(
        zip(invalid_attempts, failures, usages, sse_attempts, strict=True), start=1
    ):
        raw_review = item.get("raw_review") if isinstance(item, dict) else None
        if (
            not isinstance(raw_review, dict)
            or set(item)
            != {
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
            or type(item.get("attempt")) is not int
            or item.get("attempt") != attempt
            or item.get("raw_review_sha256") != _sha256_json(raw_review)
            or item.get("sse_integrity") != integrity
            or _validate_invalid_attempt_raw_normalization_provenance(
                item, source_row, semantic_payload
            )
            or not isinstance(failure, dict)
            or set(failure)
            != {
                "attempt",
                "termination",
                "error_type",
                "failure_category",
                "response_sha256",
                "response_hash_status",
            }
            or type(failure.get("attempt")) is not int
            or failure.get("attempt") != attempt
            or failure.get("termination") != "review_validation_error"
            or failure.get("error_type") != "ValueError"
            or failure.get("failure_category") != "semantic"
            or failure.get("response_sha256") != item["raw_review_sha256"]
            or failure.get("response_hash_status") != "complete_json_sha256"
            or not isinstance(usage, dict)
            or set(usage)
            != {"attempt", "prompt_tokens", "completion_tokens", "total_tokens"}
            or type(usage.get("attempt")) is not int
            or usage.get("attempt") != attempt
            or any(
                type(usage.get(key)) is not int or usage[key] < 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            or usage.get("total_tokens")
            != usage.get("prompt_tokens") + usage.get("completion_tokens")
            or not isinstance(integrity, dict)
            or set(integrity) != expected_sse_keys
            or type(integrity.get("attempt")) is not int
            or integrity.get("attempt") != attempt
            or integrity.get("completion_policy") != SSE_COMPLETION_POLICY
            or integrity.get("termination")
            not in {"done_sentinel", "validated_stop_eof"}
            or integrity.get("done_seen")
            is not (integrity.get("termination") == "done_sentinel")
            or integrity.get("finish_reason") != "stop"
            or integrity.get("finish_reasons") != ["stop"]
            or integrity.get("finish_reason_stop_seen") is not True
            or integrity.get("event_integrity") != "passed"
            or integrity.get("review_validation") != "failed"
            or type(integrity.get("event_count")) is not int
            or integrity["event_count"] < 2
            or type(integrity.get("content_event_count")) is not int
            or not 1
            <= integrity["content_event_count"]
            <= integrity["event_count"] - 1
            or type(integrity.get("content_chars")) is not int
            or integrity["content_chars"]
            < len(json.dumps(raw_review, ensure_ascii=False, separators=(",", ":")))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(integrity.get("content_sha256") or "")
            )
            or not isinstance(integrity.get("usage_integrity"), dict)
            or set(integrity["usage_integrity"]) != set(expected_usage_integrity)
            or integrity["usage_integrity"].get("status") != "passed"
            or type(integrity["usage_integrity"].get("usage_event_count"))
            is not int
            or integrity["usage_integrity"].get("usage_event_count") != 1
            or integrity["usage_integrity"].get("required_token_fields_valid")
            is not True
            or integrity["usage_integrity"].get(
                "total_matches_prompt_plus_completion"
            )
            is not True
            or integrity["usage_integrity"].get(
                "at_or_after_finish_reason_stop"
            )
            is not True
        ):
            return False
    if (
        source_row.get("sse_integrity") != sse_attempts[-1]
        or source_row.get("invalid_review") != invalid_attempts[-1].get("review")
        or source_row.get("error")
        != "ValueError: " + "; ".join(invalid_attempts[-1]["validation_errors"])
    ):
        return False
    return True


def _saved_post150_formal_failure_envelope_valid(
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    *,
    execution_contract_sha256: str,
) -> bool:
    """Replay the exact live 2.32 post-150 failed-row envelope.

    The core 2.32 envelope is byte-for-byte the frozen 2.29 formal envelope
    plus the wrapper's outer execution-contract binding.  Reusing the mature
    exact validator on a copied compatibility view avoids weakening either
    profile and does not alter the live runner.
    """

    if (
        source_row.get("runner_version")
        != POST150_SAVED_RECOVERY_SOURCE_RUNNER_VERSION
        or source_row.get("execution_contract_sha256")
        != execution_contract_sha256
        or set(source_row)
        != {
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
            "invalid_review",
            "invalid_attempts",
            "sse_integrity",
            "sse_integrity_attempts",
            "failure_termination",
            "failure_attempts",
            "attempt_usage",
            "usage",
            "execution_contract_sha256",
            "sequence",
        }
    ):
        return False
    compatibility_view = deepcopy(source_row)
    del compatibility_view["execution_contract_sha256"]
    compatibility_view["runner_version"] = (
        "2.29.0-real-sse-calibration-ledger"
    )
    return _saved_v229_formal_failure_envelope_valid(
        compatibility_view, semantic_payload
    )


def _saved_recovery_source_adjacency_errors(
    response: dict[str, Any],
    source_response_rows: Sequence[dict[str, Any]],
    *,
    source_index: int,
    label: str,
) -> list[str]:
    """验证恢复行生成时的相邻前缀，不让后续合法事件追溯性失效。"""

    same_id_indices = [
        index
        for index, row in enumerate(source_response_rows)
        if row.get("custom_id") == response.get("custom_id")
    ]
    response_hash = _sha256_json(response)
    persisted_response_indices = [
        index
        for index, row in enumerate(source_response_rows)
        if _sha256_json(row) == response_hash
    ]
    if persisted_response_indices:
        if len(persisted_response_indices) != 1:
            return [f"{label} is not present exactly once"]
        response_index = persisted_response_indices[0]
        prior_same_id_indices = [
            index for index in same_id_indices if index < response_index
        ]
        if not prior_same_id_indices or prior_same_id_indices[-1] != source_index:
            return [f"{label} is not adjacent to its latest source prefix"]
        return []
    if not same_id_indices or same_id_indices[-1] != source_index:
        return [f"{label} source is not the physical latest same-id event"]
    return []


def _json_tree_type_exact_equal(actual: Any, expected: Any) -> bool:
    """递归比较JSON树，同时区分bool、int与float等Python相等值。"""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_tree_type_exact_equal(actual[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_tree_type_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _deterministic_saved_quote_recovery_completed_at(
    source_row: dict[str, Any],
) -> str | None:
    """从已验真的源时间确定性派生离线恢复事件时间。"""

    try:
        source_completed_at = datetime.fromisoformat(
            str(source_row.get("completed_at") or "")
        )
    except ValueError:
        return None
    if (
        source_completed_at.tzinfo is None
        or source_completed_at.utcoffset() != UTC.utcoffset(None)
    ):
        return None
    return (
        source_completed_at.astimezone(UTC) + timedelta(microseconds=1)
    ).isoformat()


def _recover_saved_same_concept_variant_subset_from_source(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    execution_contract_sha256: str,
    completed_at: str | None = None,
    materialization_policy_version: str | None = None,
) -> dict[str, Any] | None:
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or source_row.get("model") != (row.get("request") or {}).get("model")
        or not _saved_post150_formal_failure_envelope_valid(
            source_row,
            semantic_payload,
            execution_contract_sha256=execution_contract_sha256,
        )
        or not _saved_semantic_quota_exhausted(source_row)
        or not _saved_redundant_attempt_receipts_valid(source_row)
    ):
        return None
    invalid_attempts = source_row.get("invalid_attempts") or []
    if len(invalid_attempts) < 2:
        return None
    final = invalid_attempts[-1]
    raw_review = final.get("raw_review") if isinstance(final, dict) else None
    provenance = final.get("normalization_provenance") or {}
    if (
        not isinstance(raw_review, dict)
        or final.get("attempt") != source_row.get("attempt")
        or final.get("review") != raw_review
        or final.get("raw_review_sha256") != _sha256_json(raw_review)
        or final.get("pre_normalization_review_sha256")
        != final.get("raw_review_sha256")
        or final.get("review_sha256") != final.get("raw_review_sha256")
        or provenance.get("actions") != []
        or _invalid_attempt_has_structural_normalization(final)
        or not _complete_terminal_semantic_attempt(
            final, source_row, semantic_payload
        )
    ):
        return None
    projection = _project_saved_same_concept_variant_subset(
        raw_review,
        semantic_payload,
        materialization_policy_version=materialization_policy_version,
    )
    if projection is None:
        return None
    projected, actions, raw_errors, materialization_audit = projection
    if final.get("validation_errors") != raw_errors or len(raw_errors) != 1:
        return None
    raw_hash = _sha256_json(raw_review)
    projected_hash = _sha256_json(projected)
    source = {
        "custom_id": str(row["custom_id"]),
        "prompt_sha256": source_row.get("prompt_sha256"),
        "model": source_row.get("model"),
        "runner_version": source_row.get("runner_version"),
        "transport": source_row.get("transport"),
        "execution_contract_sha256": execution_contract_sha256,
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "sequence": source_row.get("sequence"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": final.get("attempt"),
        "raw_review_sha256": raw_hash,
        "terminal_content_sha256": (final.get("sse_integrity") or {}).get(
            "content_sha256"
        ),
        "normalization_actions_sha256": _sha256_json([]),
        "validation_errors": deepcopy(raw_errors),
    }
    audit = {
        "policy": SAVED_SAME_CONCEPT_VARIANT_SUBSET_POLICY,
        "api_call_performed": False,
        "raw_review_sha256": raw_hash,
        "projected_review_sha256": projected_hash,
        "raw_validation_errors": deepcopy(raw_errors),
        "actions": deepcopy(actions),
        "selection": {
            "policy": "latest_failed_row_final_complete_terminal_attempt",
            "selected_attempt": final.get("attempt"),
            "final_complete_terminal_attempt": True,
            "semantic_consensus": False,
            "cross_attempt_merge": False,
        },
        "materialization_audit": deepcopy(materialization_audit),
        "source": source,
    }
    return {
        "custom_id": str(row["custom_id"]),
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": POST150_SAVED_RECOVERY_RUNNER_VERSION,
        "transport": SAVED_RESPONSE_TRANSPORT,
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": projected_hash,
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "successful_response_usage": {},
        "attempt_usage": [],
        "saved_response_same_concept_variant_subset_recovery": audit,
        "prior_response_recovery": {
            "mode": "saved_same_concept_existing_any_of_subset_recovery",
            "api_call_performed": False,
            "sources": [deepcopy(source)],
        },
        "completed_at": completed_at or datetime.now(UTC).isoformat(),
        "execution_contract_sha256": execution_contract_sha256,
        "sequence": source_row.get("sequence"),
    }


def _recover_latest_saved_same_concept_variant_subset(
    *,
    row: dict[str, Any],
    response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
) -> dict[str, Any] | None:
    """Recover only when the physical latest same-id event is eligible."""

    matching_events = [
        event
        for event in response_rows
        if event.get("custom_id") == row.get("custom_id")
    ]
    if not matching_events or matching_events[-1].get("status") != "failed":
        return None
    source_row = matching_events[-1]
    try:
        semantic_payload = json.loads(
            str(row["request"]["messages"][1]["content"])
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return _recover_saved_same_concept_variant_subset_from_source(
        row=row,
        source_row=source_row,
        semantic_payload=semantic_payload,
        execution_contract_sha256=execution_contract_sha256,
    )


def _validate_saved_same_concept_variant_subset_recovery(
    response: dict[str, Any],
    *,
    request: dict[str, Any],
    semantic_payload: dict[str, Any],
    source_response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
    materialization_policy_version: str | None = None,
) -> list[str]:
    """Independently replay one post-150 saved subset projection receipt."""

    errors: list[str] = []
    audit = response.get(
        "saved_response_same_concept_variant_subset_recovery"
    )
    source = audit.get("source") if isinstance(audit, dict) else None
    if not isinstance(source, dict):
        return ["saved same-concept variant-subset audit is missing"]
    source_hash = str(source.get("response_row_sha256") or "")
    matching_sources = [
        (index, row)
        for index, row in enumerate(source_response_rows)
        if _sha256_json(row) == source_hash
    ]
    if len(matching_sources) != 1:
        return ["saved recovery source row is not present exactly once"]
    source_index, source_row = matching_sources[0]
    errors.extend(
        _saved_recovery_source_adjacency_errors(
            response,
            source_response_rows,
            source_index=source_index,
            label="saved recovery",
        )
    )

    expected = _recover_saved_same_concept_variant_subset_from_source(
        row=request,
        source_row=source_row,
        semantic_payload=semantic_payload,
        execution_contract_sha256=execution_contract_sha256,
        completed_at=str(response.get("completed_at") or ""),
        materialization_policy_version=materialization_policy_version,
    )
    if expected is None or not _json_tree_type_exact_equal(response, expected):
        errors.append("saved same-concept variant-subset receipt replay mismatch")
    return errors


def _recover_saved_quote_key_branch_span_from_source(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    execution_contract_sha256: str,
    materialization_policy_version: str | None = None,
) -> dict[str, Any] | None:
    """从一个严格的 2.32 final failed row 恢复引号键泄漏。"""

    retry = source_row.get("retry_accounting") or {}
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or source_row.get("model") != (row.get("request") or {}).get("model")
        or not _saved_post150_formal_failure_envelope_valid(
            source_row,
            semantic_payload,
            execution_contract_sha256=execution_contract_sha256,
        )
        or not _saved_semantic_quota_exhausted(source_row)
        or not _saved_redundant_attempt_receipts_valid(source_row)
        or retry.get("semantic_failure_quota") != 3
        or retry.get("semantic_failure_count") != 3
        or retry.get("transport_failure_count") != 0
        or retry.get("total_attempts") != 3
        or source_row.get("attempt") != 3
    ):
        return None
    invalid_attempts = source_row.get("invalid_attempts") or []
    if len(invalid_attempts) != 3:
        return None
    final = invalid_attempts[-1]
    raw_review = final.get("raw_review") if isinstance(final, dict) else None
    provenance = final.get("normalization_provenance") or {}
    if (
        not isinstance(raw_review, dict)
        or final.get("attempt") != 3
        or final.get("attempt") != source_row.get("attempt")
        or final.get("review") != raw_review
        or final.get("raw_review_sha256") != _sha256_json(raw_review)
        or final.get("pre_normalization_review_sha256")
        != final.get("raw_review_sha256")
        or final.get("review_sha256") != final.get("raw_review_sha256")
        or provenance.get("actions") != []
        or _invalid_attempt_has_structural_normalization(final)
        or not _complete_terminal_semantic_attempt(
            final, source_row, semantic_payload
        )
    ):
        return None
    projection = _project_saved_quote_key_branch_span(
        raw_review,
        semantic_payload,
        materialization_policy_version=materialization_policy_version,
    )
    if projection is None:
        return None
    projected, actions, raw_errors, materialization_audit = projection
    if final.get("validation_errors") != raw_errors or len(raw_errors) != 3:
        return None
    deterministic_completed_at = _deterministic_saved_quote_recovery_completed_at(
        source_row
    )
    if deterministic_completed_at is None:
        return None

    raw_hash = _sha256_json(raw_review)
    projected_hash = _sha256_json(projected)
    source = {
        "custom_id": str(row["custom_id"]),
        "prompt_sha256": source_row.get("prompt_sha256"),
        "model": source_row.get("model"),
        "runner_version": source_row.get("runner_version"),
        "transport": source_row.get("transport"),
        "execution_contract_sha256": execution_contract_sha256,
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "sequence": source_row.get("sequence"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": final.get("attempt"),
        "raw_review_sha256": raw_hash,
        "terminal_content_sha256": (final.get("sse_integrity") or {}).get(
            "content_sha256"
        ),
        "normalization_actions_sha256": _sha256_json([]),
        "validation_errors": deepcopy(raw_errors),
    }
    audit = {
        "policy": SAVED_QUOTE_KEY_BRANCH_SPAN_RECOVERY_POLICY,
        "completed_at_policy": SAVED_QUOTE_KEY_RECOVERY_EVENT_TIME_POLICY,
        "api_call_performed": False,
        "raw_review_sha256": raw_hash,
        "projected_review_sha256": projected_hash,
        "raw_validation_errors": deepcopy(raw_errors),
        "actions": deepcopy(actions),
        "selection": {
            "policy": "latest_failed_row_final_complete_terminal_attempt",
            "selected_attempt": 3,
            "final_complete_terminal_attempt": True,
            "semantic_consensus": False,
            "cross_attempt_merge": False,
        },
        "materialization_audit": deepcopy(materialization_audit),
        "source": source,
    }
    return {
        "custom_id": str(row["custom_id"]),
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
        "transport": SAVED_RESPONSE_TRANSPORT,
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": projected_hash,
        "usage_is_lower_bound": False,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "successful_response_usage": {},
        "attempt_usage": [],
        "saved_response_quote_key_branch_span_recovery": audit,
        "prior_response_recovery": {
            "mode": "saved_quote_key_exact_payload_branch_span_recovery",
            "api_call_performed": False,
            "sources": [deepcopy(source)],
        },
        "completed_at": deterministic_completed_at,
        "execution_contract_sha256": execution_contract_sha256,
        "sequence": source_row.get("sequence"),
    }


def _recover_latest_saved_quote_key_branch_span(
    *,
    row: dict[str, Any],
    response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
) -> dict[str, Any] | None:
    """仅当物理 latest same-id event 为合格 failed row 时恢复。"""

    matching_events = [
        event
        for event in response_rows
        if event.get("custom_id") == row.get("custom_id")
    ]
    if not matching_events or matching_events[-1].get("status") != "failed":
        return None
    try:
        semantic_payload = json.loads(
            str(row["request"]["messages"][1]["content"])
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    return _recover_saved_quote_key_branch_span_from_source(
        row=row,
        source_row=matching_events[-1],
        semantic_payload=semantic_payload,
        execution_contract_sha256=execution_contract_sha256,
    )


def _validate_saved_quote_key_branch_span_recovery(
    response: dict[str, Any],
    *,
    request: dict[str, Any],
    semantic_payload: dict[str, Any],
    source_response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
    materialization_policy_version: str | None = None,
) -> list[str]:
    """独立重放一个引号键泄漏 saved receipt。"""

    audit = response.get("saved_response_quote_key_branch_span_recovery")
    source = audit.get("source") if isinstance(audit, dict) else None
    if not isinstance(source, dict):
        return ["saved quote-key branch-span audit is missing"]
    source_hash = str(source.get("response_row_sha256") or "")
    matching_sources = [
        (index, row)
        for index, row in enumerate(source_response_rows)
        if _sha256_json(row) == source_hash
    ]
    if len(matching_sources) != 1:
        return ["saved quote-key recovery source row is not present exactly once"]
    source_index, source_row = matching_sources[0]
    errors: list[str] = []
    errors.extend(
        _saved_recovery_source_adjacency_errors(
            response,
            source_response_rows,
            source_index=source_index,
            label="saved quote-key recovery",
        )
    )

    expected = _recover_saved_quote_key_branch_span_from_source(
        row=request,
        source_row=source_row,
        semantic_payload=semantic_payload,
        execution_contract_sha256=execution_contract_sha256,
        materialization_policy_version=materialization_policy_version,
    )
    if expected is None or not _json_tree_type_exact_equal(response, expected):
        errors.append("saved quote-key branch-span receipt replay mismatch")
    return errors


def _validate_saved_post150_recovery(
    response: dict[str, Any],
    *,
    request: dict[str, Any],
    semantic_payload: dict[str, Any],
    source_response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
    materialization_policy_version: str | None = None,
) -> list[str]:
    """按独立 runner profile 分派并重放 post150 saved receipt。"""

    profile = (response.get("runner_version"), response.get("transport"))
    if profile == (
        POST150_SAVED_RECOVERY_RUNNER_VERSION,
        SAVED_RESPONSE_TRANSPORT,
    ):
        return _validate_saved_same_concept_variant_subset_recovery(
            response,
            request=request,
            semantic_payload=semantic_payload,
            source_response_rows=source_response_rows,
            execution_contract_sha256=execution_contract_sha256,
            materialization_policy_version=materialization_policy_version,
        )
    if profile == (
        POST150_SAVED_QUOTE_KEY_RECOVERY_RUNNER_VERSION,
        SAVED_RESPONSE_TRANSPORT,
    ):
        return _validate_saved_quote_key_branch_span_recovery(
            response,
            request=request,
            semantic_payload=semantic_payload,
            source_response_rows=source_response_rows,
            execution_contract_sha256=execution_contract_sha256,
            materialization_policy_version=materialization_policy_version,
        )
    return ["post150 saved response profile is not authorized"]


def _recover_latest_saved_post150_recovery(
    *,
    row: dict[str, Any],
    response_rows: Sequence[dict[str, Any]],
    execution_contract_sha256: str,
) -> dict[str, Any] | None:
    """按互斥投影规则选择一个零费用 post150 saved profile。"""

    legacy = _recover_latest_saved_same_concept_variant_subset(
        row=row,
        response_rows=response_rows,
        execution_contract_sha256=execution_contract_sha256,
    )
    quote_key = _recover_latest_saved_quote_key_branch_span(
        row=row,
        response_rows=response_rows,
        execution_contract_sha256=execution_contract_sha256,
    )
    if legacy is not None and quote_key is not None:
        raise RuntimeError("post150 saved recovery profiles overlap")
    return legacy if legacy is not None else quote_key


def _recover_saved_redundant_field_coverage_composition(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover the final persisted attempt after format-only quota exhaustion.

    Every semantic attempt must independently pass the same deterministic
    two-stage projector and expose the same missing-field action shape.  Their
    semantic contracts may differ and are never combined: the failed row's
    final complete attempt is selected deterministically.
    """

    retry = source_row.get("retry_accounting") or {}
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
        or not isinstance(retry.get("semantic_failure_count"), int)
        or int(retry["semantic_failure_count"]) < 2
        or not isinstance(source_row.get("sequence"), int)
        or int(source_row["sequence"]) < 1
        or source_row.get("runner_version")
        not in RAW_PROVENANCE_REQUIRED_RUNNER_VERSIONS
        or not _saved_redundant_attempt_receipts_valid(source_row)
    ):
        return None

    invalid_attempts = source_row.get("invalid_attempts") or []
    if (
        len(invalid_attempts) < 2
        or any(not isinstance(item, dict) for item in invalid_attempts)
    ):
        return None

    members: list[dict[str, Any]] = []
    expected_shape: list[dict[str, Any]] | None = None
    for item in invalid_attempts:
        raw_review = item.get("raw_review")
        if (
            not isinstance(raw_review, dict)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
        ):
            return None
        raw_hash = _sha256_json(raw_review)
        if item.get("raw_review_sha256") != raw_hash:
            return None
        projected, stages, raw_errors = (
            _canonicalize_saved_redundant_field_coverage_composition(
                raw_review, semantic_payload
            )
        )
        if len(stages) != 2:
            return None
        reconstruction_actions = stages[0].get("actions") or []
        expected_normalization_actions = _expected_redundant_normalization_actions(
            reconstruction_actions
        )
        provenance = item.get("normalization_provenance") or {}
        if (
            not expected_normalization_actions
            or provenance.get("actions") != expected_normalization_actions
            or not isinstance(item.get("validation_errors"), list)
        ):
            return None
        shape = _redundant_field_stage_action_signature(stages)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            return None
        members.append(
            {
                "attempt": item.get("attempt"),
                "raw_review": deepcopy(raw_review),
                "raw_review_sha256": raw_hash,
                "raw_validation_errors": raw_errors,
                "persisted_validation_errors": deepcopy(
                    item["validation_errors"]
                ),
                "normalization_actions": expected_normalization_actions,
                "action_signature": shape,
                "action_signature_sha256": _sha256_json(shape),
                "projected_review": projected,
                "projected_review_sha256": _sha256_json(projected),
                "stages": stages,
                "terminal_content_sha256": (
                    item.get("sse_integrity") or {}
                ).get("content_sha256"),
            }
        )

    attempts = [member.get("attempt") for member in members]
    if (
        any(not isinstance(attempt, int) for attempt in attempts)
        or len(set(attempts)) != len(attempts)
    ):
        return None
    selected = max(members, key=lambda member: int(member["attempt"]))
    if (
        selected["attempt"] != source_row.get("attempt")
        or selected["attempt"] != retry.get("total_attempts")
    ):
        return None

    selected_attempt_items = [
        item
        for item in invalid_attempts
        if item.get("attempt") == selected["attempt"]
    ]
    if len(selected_attempt_items) != 1:
        return None
    selected_item = selected_attempt_items[0]
    source = {
        "custom_id": str(row["custom_id"]),
        "prompt_sha256": source_row.get("prompt_sha256"),
        "runner_version": source_row.get("runner_version"),
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": selected["attempt"],
        "raw_review_sha256": selected["raw_review_sha256"],
        "validation_errors": selected_item.get("validation_errors") or [],
    }
    receipts = [
        {
            "attempt": member["attempt"],
            "raw_review_sha256": member["raw_review_sha256"],
            "raw_validation_errors_sha256": _sha256_json(
                member["raw_validation_errors"]
            ),
            "persisted_validation_errors_sha256": _sha256_json(
                member["persisted_validation_errors"]
            ),
            "normalization_actions": member["normalization_actions"],
            "action_signature_sha256": member["action_signature_sha256"],
            "projected_review_sha256": member["projected_review_sha256"],
            "terminal_content_sha256": member["terminal_content_sha256"],
        }
        for member in members
    ]
    projected = selected["projected_review"]
    projected_hash = selected["projected_review_sha256"]
    stages = selected["stages"]
    audit = {
        "policy": REDUNDANT_FIELD_COVERAGE_COMPOSITION_POLICY,
        "api_call_performed": False,
        "raw_review_sha256": selected["raw_review_sha256"],
        "intermediate_review_sha256": stages[0]["output_review_sha256"],
        "projected_review_sha256": projected_hash,
        "raw_validation_errors": selected["raw_validation_errors"],
        "stages": stages,
        "attempt_shape_receipts": receipts,
        "selection": {
            "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
            "selected_attempt": selected["attempt"],
            "semantic_consensus_claimed": False,
            "cross_attempt_merge": False,
        },
        "source": source,
    }
    return {
        "custom_id": str(row["custom_id"]),
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": RUNNER_VERSION,
        "transport": "saved-response-recovery-no-api",
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": projected_hash,
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "successful_response_usage": {},
        "attempt_usage": [],
        "saved_response_redundant_field_coverage_composition": audit,
        "prior_response_recovery": {
            "mode": "saved_final_attempt_redundant_field_coverage_format_recovery",
            "api_call_performed": False,
            "sources": [source],
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _recover_saved_abstain_factual_constraints(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
    recovery_policy: str = ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY,
    target_runner_version: str = RUNNER_VERSION,
) -> dict[str, Any] | None:
    """Recover only the final independently valid abstention projection."""

    retry = source_row.get("retry_accounting") or {}
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
        or not isinstance(retry.get("semantic_failure_count"), int)
        or int(retry["semantic_failure_count"]) < 2
        or not isinstance(source_row.get("sequence"), int)
        or int(source_row["sequence"]) < 1
        or (
            recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
            and source_row.get("runner_version")
            != ABSTAIN_SENTINEL_SOURCE_RUNNER_BY_TARGET.get(
                target_runner_version
            )
        )
        or (
            recovery_policy == LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
            and source_row.get("runner_version")
            not in ABSTAIN_FACTUAL_CONSTRAINT_SOURCE_RUNNER_VERSIONS
        )
        or recovery_policy
        not in {
            ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY,
            LEGACY_ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY,
        }
        or source_row.get("runner_version")
        == "2.29.0-real-sse-calibration-ledger"
        and not _saved_v229_formal_failure_envelope_valid(
            source_row, semantic_payload
        )
        or not _saved_redundant_attempt_receipts_valid(source_row)
    ):
        return None

    invalid_attempts = source_row.get("invalid_attempts") or []
    if (
        len(invalid_attempts) < 2
        or any(not isinstance(item, dict) for item in invalid_attempts)
    ):
        return None

    members: list[dict[str, Any]] = []
    expected_shape: list[dict[str, Any]] | None = None
    for item in invalid_attempts:
        raw_review = item.get("raw_review")
        persisted_errors = item.get("validation_errors")
        variant_error = (
            "abstain concept[0] variants must each express only an explicit "
            "absence/uncertainty rationale"
        )
        persisted_errors_valid = persisted_errors == [
            _ABSTAIN_FACTUAL_CONSTRAINT_ERROR
        ] or (
            recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY
            and persisted_errors
            in (
                [variant_error],
                [variant_error, _ABSTAIN_FACTUAL_CONSTRAINT_ERROR],
            )
        )
        if (
            not isinstance(raw_review, dict)
            or _invalid_attempt_has_structural_normalization(item)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
            or item.get("raw_review_sha256") != _sha256_json(raw_review)
            or not persisted_errors_valid
        ):
            return None
        projected, stages, raw_errors = (
            _canonicalize_saved_abstain_factual_constraints(
                raw_review,
                semantic_payload,
                recovery_policy=recovery_policy,
            )
        )
        if (
            tuple(raw_errors)
            not in {
                (variant_error,),
                (_ABSTAIN_FACTUAL_CONSTRAINT_ERROR,),
                (variant_error, _ABSTAIN_FACTUAL_CONSTRAINT_ERROR),
            }
            or len(stages) != 1
            or len(stages[0].get("actions") or []) != 1
        ):
            return None
        shape = _abstain_constraint_action_signature(stages)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            return None
        members.append(
            {
                "attempt": item.get("attempt"),
                "raw_review": deepcopy(raw_review),
                "raw_review_sha256": _sha256_json(raw_review),
                "raw_validation_errors": raw_errors,
                "persisted_validation_errors": deepcopy(persisted_errors),
                "action_signature_sha256": _sha256_json(shape),
                "projected_review": projected,
                "projected_review_sha256": _sha256_json(projected),
                "stages": stages,
                "terminal_content_sha256": (
                    item.get("sse_integrity") or {}
                ).get("content_sha256"),
            }
        )

    attempts = [member.get("attempt") for member in members]
    if (
        any(not isinstance(attempt, int) for attempt in attempts)
        or len(set(attempts)) != len(attempts)
    ):
        return None
    selected = max(members, key=lambda member: int(member["attempt"]))
    if (
        selected["attempt"] != source_row.get("attempt")
        or selected["attempt"] != retry.get("total_attempts")
    ):
        return None

    source = {
        "custom_id": str(row["custom_id"]),
        "prompt_sha256": source_row.get("prompt_sha256"),
        "runner_version": source_row.get("runner_version"),
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": selected["attempt"],
        "raw_review_sha256": selected["raw_review_sha256"],
        "validation_errors": selected["persisted_validation_errors"],
        "terminal_content_sha256": selected["terminal_content_sha256"],
    }
    receipts = [
        {
            "attempt": member["attempt"],
            "raw_review_sha256": member["raw_review_sha256"],
            "raw_validation_errors_sha256": _sha256_json(
                member["raw_validation_errors"]
            ),
            "action_signature_sha256": member["action_signature_sha256"],
            "projected_review_sha256": member["projected_review_sha256"],
            "terminal_content_sha256": member["terminal_content_sha256"],
        }
        for member in members
    ]
    if recovery_policy == ABSTAIN_FACTUAL_CONSTRAINT_RECOVERY_POLICY:
        for receipt, member in zip(receipts, members, strict=True):
            receipt["persisted_validation_errors_sha256"] = _sha256_json(
                member["persisted_validation_errors"]
            )
    projected = selected["projected_review"]
    audit = {
        "policy": recovery_policy,
        "api_call_performed": False,
        "raw_review_sha256": selected["raw_review_sha256"],
        "projected_review_sha256": selected["projected_review_sha256"],
        "raw_validation_errors": selected["raw_validation_errors"],
        "stages": selected["stages"],
        "attempt_shape_receipts": receipts,
        "selection": {
            "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
            "selected_attempt": selected["attempt"],
            "semantic_consensus_claimed": False,
            "cross_attempt_merge": False,
        },
        "source": source,
    }
    return {
        "custom_id": str(row["custom_id"]),
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": RUNNER_VERSION,
        "transport": "saved-response-recovery-no-api",
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": selected["projected_review_sha256"],
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "successful_response_usage": {},
        "attempt_usage": [],
        "saved_response_abstain_factual_constraint_recovery": audit,
        "prior_response_recovery": {
            "mode": "saved_final_attempt_abstain_factual_constraint_recovery",
            "api_call_performed": False,
            "sources": [source],
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _recover_saved_exact_payload_branch_span_quote_prefix(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover only the failed row's final complete, nonstructural attempt."""

    retry = source_row.get("retry_accounting") or {}
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
        or not isinstance(retry.get("semantic_failure_count"), int)
        or int(retry["semantic_failure_count"]) < 2
        or not isinstance(source_row.get("sequence"), int)
        or int(source_row["sequence"]) < 1
        or source_row.get("runner_version")
        not in EXACT_PAYLOAD_BRANCH_SPAN_SOURCE_RUNNER_VERSIONS
        or not _saved_redundant_attempt_receipts_valid(source_row)
    ):
        return None
    invalid_attempts = source_row.get("invalid_attempts") or []
    final_items = [
        item
        for item in invalid_attempts
        if isinstance(item, dict)
        and item.get("attempt") == source_row.get("attempt")
    ]
    if len(final_items) != 1:
        return None
    item = final_items[0]
    raw_review = item.get("raw_review")
    if (
        not isinstance(raw_review, dict)
        or _invalid_attempt_has_structural_normalization(item)
        or not _complete_terminal_semantic_attempt(
            item, source_row, semantic_payload
        )
        or item.get("raw_review_sha256") != _sha256_json(raw_review)
    ):
        return None
    projected, stages, raw_errors = (
        _canonicalize_saved_exact_payload_branch_span_quote_prefix(
            raw_review, semantic_payload
        )
    )
    if (
        len(stages) != 1
        or item.get("validation_errors") != raw_errors
        or stages[0].get("input_review_sha256") != _sha256_json(raw_review)
        or stages[0].get("output_review_sha256") != _sha256_json(projected)
    ):
        return None

    source = {
        "custom_id": str(row["custom_id"]),
        "prompt_sha256": source_row.get("prompt_sha256"),
        "runner_version": source_row.get("runner_version"),
        "response_row_sha256": _sha256_json(source_row),
        "completed_at": source_row.get("completed_at"),
        "failure_max_attempt": source_row.get("attempt"),
        "source_attempt": item.get("attempt"),
        "raw_review_sha256": _sha256_json(raw_review),
        "validation_errors": deepcopy(item.get("validation_errors") or []),
        "terminal_content_sha256": (item.get("sse_integrity") or {}).get(
            "content_sha256"
        ),
    }
    audit = {
        "policy": EXACT_PAYLOAD_BRANCH_SPAN_QUOTE_PREFIX_RECOVERY_POLICY,
        "api_call_performed": False,
        "raw_review_sha256": _sha256_json(raw_review),
        "projected_review_sha256": _sha256_json(projected),
        "raw_validation_errors": raw_errors,
        "stages": stages,
        "selection": {
            "policy": "final_complete_terminal_attempt_after_semantic_quota_exhaustion",
            "selected_attempt": item.get("attempt"),
            "semantic_consensus_claimed": False,
            "cross_attempt_merge": False,
        },
        "source": source,
    }
    return {
        "custom_id": str(row["custom_id"]),
        "status": "succeeded",
        "attempt": 0,
        "model": source_row.get("model"),
        "runner_version": RUNNER_VERSION,
        "transport": "saved-response-recovery-no-api",
        "prompt_sha256": row["prompt_sha256"],
        "review": projected,
        "response_sha256": _sha256_json(projected),
        "usage_is_lower_bound": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "successful_response_usage": {},
        "attempt_usage": [],
        "saved_response_exact_payload_branch_span_quote_prefix_recovery": audit,
        "prior_response_recovery": {
            "mode": "saved_final_attempt_exact_payload_branch_span_quote_prefix_recovery",
            "api_call_performed": False,
            "sources": [source],
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _recover_saved_quote_fragment_inverse_composition(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
    ):
        return None
    invalid_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict) and isinstance(item.get("review"), dict)
    ]
    for item in reversed(invalid_attempts):
        raw_review = deepcopy(item["review"])
        raw_hash = _sha256_json(raw_review)
        if (
            item.get("review_sha256") != raw_hash
            or _invalid_attempt_has_structural_normalization(item)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
        ):
            continue
        projected, stages, raw_errors = (
            _canonicalize_saved_quote_fragment_inverse_composition(
                raw_review, semantic_payload
            )
        )
        if not stages or item.get("validation_errors") != raw_errors:
            continue
        projected_hash = _sha256_json(projected)
        source = {
            "custom_id": str(row["custom_id"]),
            "prompt_sha256": source_row.get("prompt_sha256"),
            "runner_version": source_row.get("runner_version"),
            "response_row_sha256": _sha256_json(source_row),
            "completed_at": source_row.get("completed_at"),
            "failure_max_attempt": source_row.get("attempt"),
            "source_attempt": item.get("attempt"),
            "raw_review_sha256": raw_hash,
            "validation_errors": raw_errors,
        }
        return {
            "custom_id": str(row["custom_id"]),
            "status": "succeeded",
            "attempt": 0,
            "model": source_row.get("model"),
            "runner_version": RUNNER_VERSION,
            "transport": "saved-response-recovery-no-api",
            "prompt_sha256": row["prompt_sha256"],
            "review": projected,
            "response_sha256": projected_hash,
            "usage_is_lower_bound": False,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "successful_response_usage": {},
            "attempt_usage": [],
            "same_response_quote_fragment_inverse_canonicalization_composition": {
                "policy": QUOTE_FRAGMENT_INVERSE_COMPOSITION_POLICY,
                "api_call_performed": False,
                "raw_review_sha256": raw_hash,
                "intermediate_review_sha256": stages[0][
                    "output_review_sha256"
                ],
                "projected_review_sha256": projected_hash,
                "raw_validation_errors": raw_errors,
                "stages": stages,
                "source": source,
            },
            "prior_response_recovery": {
                "mode": (
                    "saved_quote_fragment_inverse_canonicalization_composition"
                ),
                "api_call_performed": False,
                "sources": [source],
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
    return None


def _recover_saved_nonsemantic_composition(
    *,
    row: dict[str, Any],
    source_row: dict[str, Any],
    semantic_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if (
        source_row.get("custom_id") != row.get("custom_id")
        or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        or not _saved_semantic_quota_exhausted(source_row)
    ):
        return None
    invalid_attempts = [
        item
        for item in source_row.get("invalid_attempts") or []
        if isinstance(item, dict) and isinstance(item.get("review"), dict)
    ]
    for item in reversed(invalid_attempts):
        raw_review = deepcopy(item["review"])
        raw_hash = _sha256_json(raw_review)
        if (
            item.get("review_sha256") != raw_hash
            or _invalid_attempt_has_structural_normalization(item)
            or not _complete_terminal_semantic_attempt(
                item, source_row, semantic_payload
            )
        ):
            continue
        projected, stages, raw_errors = (
            _canonicalize_saved_nonsemantic_composition(
                raw_review, semantic_payload
            )
        )
        if not stages or item.get("validation_errors") != raw_errors:
            continue
        projected_hash = _sha256_json(projected)
        source = {
            "custom_id": str(row["custom_id"]),
            "prompt_sha256": source_row.get("prompt_sha256"),
            "runner_version": source_row.get("runner_version"),
            "response_row_sha256": _sha256_json(source_row),
            "completed_at": source_row.get("completed_at"),
            "failure_max_attempt": source_row.get("attempt"),
            "source_attempt": item.get("attempt"),
            "raw_review_sha256": raw_hash,
            "validation_errors": raw_errors,
        }
        return {
            "custom_id": str(row["custom_id"]),
            "status": "succeeded",
            "attempt": 0,
            "model": source_row.get("model"),
            "runner_version": RUNNER_VERSION,
            "transport": "saved-response-recovery-no-api",
            "prompt_sha256": row["prompt_sha256"],
            "review": projected,
            "response_sha256": projected_hash,
            "usage_is_lower_bound": False,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "successful_response_usage": {},
            "attempt_usage": [],
            "same_response_nonsemantic_canonicalization_composition": {
                "policy": NONSEMANTIC_CANONICALIZATION_COMPOSITION_POLICY,
                "api_call_performed": False,
                "raw_review_sha256": raw_hash,
                "intermediate_review_sha256": stages[0][
                    "output_review_sha256"
                ],
                "projected_review_sha256": projected_hash,
                "raw_validation_errors": raw_errors,
                "stages": stages,
                "source": source,
            },
            "prior_response_recovery": {
                "mode": "saved_nonsemantic_canonicalizer_composition",
                "api_call_performed": False,
                "sources": [source],
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
    return None


def _recover_prior_invalid_review(
    *, row: dict[str, Any], response_rows: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Recover a strict saved response without adding or combining semantics."""

    custom_id = str(row["custom_id"])
    semantic_payload = json.loads(str(row["request"]["messages"][1]["content"]))
    matching_failed_rows = [
        source_row
        for source_row in response_rows
        if source_row.get("status") == "failed"
        and str(source_row.get("custom_id") or "") == custom_id
        and source_row.get("prompt_sha256") == row.get("prompt_sha256")
    ]
    latest_failed_row = matching_failed_rows[-1] if matching_failed_rows else None
    for source_row in response_rows:
        if (
            source_row.get("status") != "failed"
            or str(source_row.get("custom_id") or "") != custom_id
            or source_row.get("prompt_sha256") != row.get("prompt_sha256")
        ):
            continue
        unsupported_block = _recover_quota_exhausted_verdict_block(
            row=row,
            source_row=source_row,
            semantic_payload=semantic_payload,
            projector=_canonicalize_same_response_unsupported_required_concept_block,
            audit_key="same_response_unsupported_required_concept_block",
            audit_policy=UNSUPPORTED_REQUIRED_CONCEPT_BLOCK_POLICY,
            recovery_mode=(
                "quota_exhausted_multi_response_unsupported_required_concept_block"
            ),
        )
        if unsupported_block is not None:
            return unsupported_block
        empty_branch_block = _recover_quota_exhausted_verdict_block(
            row=row,
            source_row=source_row,
            semantic_payload=semantic_payload,
            projector=_canonicalize_same_response_empty_required_branch_block,
            audit_key="multi_response_empty_required_branch_block",
            audit_policy=EMPTY_REQUIRED_BRANCH_BLOCK_POLICY,
            recovery_mode="quota_exhausted_multi_response_empty_required_branch_block",
        )
        if empty_branch_block is not None:
            return empty_branch_block
        abstain_factual_constraints = (
            _recover_saved_abstain_factual_constraints(
                row=row,
                source_row=source_row,
                semantic_payload=semantic_payload,
            )
            if source_row is latest_failed_row
            and source_row.get("runner_version")
            == ABSTAIN_SENTINEL_SOURCE_RUNNER_BY_TARGET.get(RUNNER_VERSION)
            else None
        )
        if abstain_factual_constraints is not None:
            return abstain_factual_constraints
        exact_span_quote_prefix = (
            _recover_saved_exact_payload_branch_span_quote_prefix(
                row=row,
                source_row=source_row,
                semantic_payload=semantic_payload,
            )
        )
        if exact_span_quote_prefix is not None:
            return exact_span_quote_prefix
        quote_fragment_composition = (
            _recover_saved_quote_fragment_inverse_composition(
                row=row,
                source_row=source_row,
                semantic_payload=semantic_payload,
            )
        )
        if quote_fragment_composition is not None:
            return quote_fragment_composition
        composition = _recover_saved_nonsemantic_composition(
            row=row,
            source_row=source_row,
            semantic_payload=semantic_payload,
        )
        if composition is not None:
            return composition
        redundant_field_composition = (
            _recover_saved_redundant_field_coverage_composition(
                row=row,
                source_row=source_row,
                semantic_payload=semantic_payload,
            )
        )
        if redundant_field_composition is not None:
            return redundant_field_composition
        source_reviews = [
            item
            for item in source_row.get("invalid_attempts") or []
            if isinstance(item, dict) and isinstance(item.get("review"), dict)
        ]
        if not source_reviews and isinstance(source_row.get("invalid_review"), dict):
            source_reviews.append(
                {
                    "attempt": None,
                    "review": source_row["invalid_review"],
                    "validation_errors": [],
                }
            )
        seen_source_hashes: set[str] = set()
        for source_item in source_reviews:
            if (
                _invalid_attempt_has_structural_normalization(source_item)
                or _validate_invalid_attempt_raw_normalization_provenance(
                    source_item, source_row, semantic_payload
                )
            ):
                continue
            raw_review = deepcopy(source_item["review"])
            review_hash = _sha256_json(raw_review)
            if review_hash in seen_source_hashes:
                continue
            seen_source_hashes.add(review_hash)
            source = {
                "custom_id": custom_id,
                "prompt_sha256": source_row.get("prompt_sha256"),
                "runner_version": source_row.get("runner_version"),
                "response_row_sha256": _sha256_json(source_row),
                "completed_at": source_row.get("completed_at"),
                "failure_max_attempt": source_row.get("attempt"),
                "source_attempt": source_item.get("attempt"),
                "raw_review_sha256": review_hash,
                "validation_errors": source_item.get("validation_errors") or [],
            }
            current_errors = _validate_review(raw_review)
            current_errors.extend(
                _validate_against_payload(raw_review, semantic_payload)
            )
            if (
                not current_errors
                and source["validation_errors"]
                and isinstance(source["source_attempt"], int)
                and source["source_attempt"] >= 1
            ):
                return {
                    "custom_id": custom_id,
                    "status": "succeeded",
                    "attempt": 0,
                    "model": source_row.get("model"),
                    "runner_version": RUNNER_VERSION,
                    "transport": "saved-response-recovery-no-api",
                    "prompt_sha256": row["prompt_sha256"],
                    "review": raw_review,
                    "response_sha256": review_hash,
                    "usage_is_lower_bound": False,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "successful_response_usage": {},
                    "attempt_usage": [],
                    "saved_response_strict_revalidation": {
                        "policy": SAVED_STRICT_REVALIDATION_POLICY,
                        "api_call_performed": False,
                        "validator_runner_version": RUNNER_VERSION,
                        "raw_review_sha256": review_hash,
                        "projected_review_sha256": review_hash,
                        "source": source,
                    },
                    "prior_response_recovery": {
                        "mode": "single_saved_strict_response_revalidated_after_validator_policy",
                        "api_call_performed": False,
                        "sources": [source],
                    },
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            deduplicated, dedupe_actions, raw_errors = (
                _dedupe_same_response_exact_variants(raw_review, semantic_payload)
            )
            if dedupe_actions:
                return {
                    "custom_id": custom_id,
                    "status": "succeeded",
                    "attempt": 0,
                    "model": source_row.get("model"),
                    "runner_version": RUNNER_VERSION,
                    "transport": "saved-response-recovery-no-api",
                    "prompt_sha256": row["prompt_sha256"],
                    "review": deduplicated,
                    "response_sha256": _sha256_json(deduplicated),
                    "usage_is_lower_bound": False,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "successful_response_usage": {},
                    "attempt_usage": [],
                    "same_response_exact_variant_deduplication": {
                        "policy": EXACT_VARIANT_DEDUPLICATION_POLICY,
                        "api_call_performed": False,
                        "raw_review_sha256": review_hash,
                        "projected_review_sha256": _sha256_json(deduplicated),
                        "raw_validation_errors": raw_errors,
                        "actions": dedupe_actions,
                        "source": source,
                    },
                    "prior_response_recovery": {
                        "mode": "same_response_exact_variant_deduplication",
                        "api_call_performed": False,
                        "sources": [source],
                    },
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            projected, coverage_actions, raw_errors = (
                _canonicalize_same_response_coverage_statuses(
                    raw_review, semantic_payload
                )
            )
            if coverage_actions:
                return {
                    "custom_id": custom_id,
                    "status": "succeeded",
                    "attempt": 0,
                    "model": source_row.get("model"),
                    "runner_version": RUNNER_VERSION,
                    "transport": "saved-response-recovery-no-api",
                    "prompt_sha256": row["prompt_sha256"],
                    "review": projected,
                    "response_sha256": _sha256_json(projected),
                    "usage_is_lower_bound": False,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "successful_response_usage": {},
                    "attempt_usage": [],
                    "same_response_coverage_status_canonicalization": {
                        "policy": COVERAGE_STATUS_CANONICALIZATION_POLICY,
                        "api_call_performed": False,
                        "raw_review_sha256": review_hash,
                        "projected_review_sha256": _sha256_json(projected),
                        "raw_validation_errors": raw_errors,
                        "actions": coverage_actions,
                        "source": source,
                    },
                    "prior_response_recovery": {
                        "mode": "same_response_coverage_status_canonicalization",
                        "api_call_performed": False,
                        "sources": [source],
                    },
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            projected, edge_actions, raw_errors = (
                _canonicalize_same_response_inverse_edges(raw_review, semantic_payload)
            )
            if edge_actions:
                return {
                    "custom_id": custom_id,
                    "status": "succeeded",
                    "attempt": 0,
                    "model": source_row.get("model"),
                    "runner_version": RUNNER_VERSION,
                    "transport": "saved-response-recovery-no-api",
                    "prompt_sha256": row["prompt_sha256"],
                    "review": projected,
                    "response_sha256": _sha256_json(projected),
                    "usage_is_lower_bound": False,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "successful_response_usage": {},
                    "attempt_usage": [],
                    "same_response_inverse_edge_canonicalization": {
                        "policy": INVERSE_EDGE_CANONICALIZATION_POLICY,
                        "api_call_performed": False,
                        "raw_review_sha256": review_hash,
                        "projected_review_sha256": _sha256_json(projected),
                        "raw_validation_errors": raw_errors,
                        "actions": edge_actions,
                        "source": source,
                    },
                    "prior_response_recovery": {
                        "mode": "same_response_inverse_edge_canonicalization",
                        "api_call_performed": False,
                        "sources": [source],
                    },
                    "completed_at": datetime.now(UTC).isoformat(),
                }
    return None


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _sse_usage_integrity(
    usage_events: Sequence[tuple[int, dict[str, Any]]],
    *,
    finish_stop_event_index: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the single terminal usage event required for sentinel-free EOF."""

    audit: dict[str, Any] = {
        "status": "missing" if not usage_events else "invalid",
        "usage_event_count": len(usage_events),
        "required_token_fields_valid": False,
        "total_matches_prompt_plus_completion": False,
        "at_or_after_finish_reason_stop": False,
    }
    if len(usage_events) != 1:
        return audit, {}
    event_index, usage = usage_events[0]
    required_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    fields_valid = all(type(usage.get(key)) is int and usage[key] >= 0 for key in required_fields)
    audit["required_token_fields_valid"] = fields_valid
    if fields_valid:
        audit["total_matches_prompt_plus_completion"] = usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )
    audit["at_or_after_finish_reason_stop"] = (
        finish_stop_event_index is not None and event_index >= finish_stop_event_index
    )
    if (
        audit["required_token_fields_valid"]
        and audit["total_matches_prompt_plus_completion"]
        and audit["at_or_after_finish_reason_stop"]
    ):
        audit["status"] = "passed"
        return audit, {key: int(usage[key]) for key in required_fields}
    return audit, {}


def _provider_error_bytes(value: Any) -> bytes:
    """Serialize an already-decoded provider error value for hashing only."""

    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _provider_error_hash_receipt(field: str, value: Any) -> dict[str, Any]:
    encoded = _provider_error_bytes(value)
    return {
        f"{field}_redacted": True,
        f"{field}_sha256": hashlib.sha256(encoded).hexdigest(),
        f"{field}_byte_length": len(encoded),
    }


def _safe_provider_error_token(field: str, value: Any) -> dict[str, Any]:
    """Retain only field-specific documented identifiers, never arbitrary tokens."""

    if field == "status" and type(value) is int and 100 <= value <= 599:
        return {field: str(value)}
    if not isinstance(value, str):
        return _provider_error_hash_receipt(field, value)
    raw = value
    encoded = raw.encode("utf-8")
    normalized = raw.strip().casefold()
    if len(encoded) > 96 or not raw.isascii() or any(
        ord(char) < 32 or ord(char) == 127 for char in raw
    ):
        return _provider_error_hash_receipt(field, value)
    allowed = (
        PROVIDER_ERROR_SAFE_CODES
        if field == "code"
        else PROVIDER_ERROR_SAFE_TYPES
        if field == "type"
        else PROVIDER_ERROR_SAFE_STATUSES
    )
    if normalized in allowed:
        return {field: normalized}
    return _provider_error_hash_receipt(field, value)


def _redacted_provider_error_receipt(raw_error: Any) -> dict[str, Any]:
    """Create a content-free diagnostic receipt for an SSE provider error frame.

    Provider messages and unknown fields can echo request text or credentials, so
    they are never persisted.  Only bounded identifier-like fields may remain in
    clear text; all rejected fields are represented by a digest and byte length.
    """

    shape = (
        "object"
        if isinstance(raw_error, dict)
        else "string"
        if isinstance(raw_error, str)
        else "array"
        if isinstance(raw_error, list)
        else "null"
        if raw_error is None
        else "boolean"
        if isinstance(raw_error, bool)
        else "number"
        if isinstance(raw_error, (int, float))
        else "unknown"
    )
    receipt: dict[str, Any] = {
        "policy": PROVIDER_ERROR_RECEIPT_POLICY,
        "shape": shape,
    }
    if isinstance(raw_error, dict):
        for field in ("code", "type", "status"):
            if field in raw_error:
                receipt.update(_safe_provider_error_token(field, raw_error[field]))
        if "message" in raw_error:
            receipt.update(_provider_error_hash_receipt("message", raw_error["message"]))
        receipt["unknown_field_count"] = sum(
            key not in {"code", "type", "status", "message"}
            for key in raw_error
        )
    else:
        # A non-object error frame is semantically message-like and is therefore
        # represented only by a digest, never by its raw value.
        receipt.update(_provider_error_hash_receipt("message", raw_error))
    return receipt


def _call_api(
    *,
    endpoint: str,
    api_key: str,
    request_body: dict[str, Any],
    timeout: float,
    eof_review_validator: Callable[[dict[str, Any]], Sequence[str]] | None = None,
    integrity_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    streaming_body = {
        **request_body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(streaming_body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    usage_events: list[tuple[int, dict[str, Any]]] = []
    event_count = 0
    content_event_count = 0
    done_seen = False
    finish_reasons: list[str] = []
    finish_stop_event_index: int | None = None
    stream_eof_reached = False

    def audit_snapshot(
        *,
        event_integrity: str,
        review_validation: str = "not_run",
        termination: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        usage_audit, validated_usage = _sse_usage_integrity(
            usage_events,
            finish_stop_event_index=finish_stop_event_index,
        )
        return (
            {
                "completion_policy": SSE_COMPLETION_POLICY,
                "termination": termination
                or (
                    "done_sentinel"
                    if done_seen
                    else "normal_eof"
                    if stream_eof_reached
                    else "protocol_error"
                ),
                "done_seen": done_seen,
                "finish_reason": finish_reasons[-1] if finish_reasons else None,
                "finish_reasons": list(finish_reasons),
                "finish_reason_stop_seen": finish_stop_event_index is not None,
                "event_count": event_count,
                "content_event_count": content_event_count,
                "content_chars": sum(len(part) for part in content_parts),
                "content_sha256": (
                    hashlib.sha256("".join(content_parts).encode("utf-8")).hexdigest()
                    if content_parts
                    else None
                ),
                "event_integrity": event_integrity,
                "usage_integrity": usage_audit,
                "review_validation": review_validation,
            },
            validated_usage,
        )

    def fail(
        reason: str,
        *,
        event_integrity: str = "passed",
        review_validation: str = "not_run",
        termination: str | None = None,
        response_sha256: str | None = None,
        provider_error_receipt: dict[str, Any] | None = None,
    ) -> None:
        audit, validated_usage = audit_snapshot(
            event_integrity=event_integrity,
            review_validation=review_validation,
            termination=termination,
        )
        if response_sha256 is not None:
            audit["response_sha256"] = response_sha256
        if provider_error_receipt is not None:
            audit["provider_error_receipt"] = provider_error_receipt
        raise SSEIntegrityError(
            f"{reason}: done_seen={audit['done_seen']}, "
            f"finish_reason={audit['finish_reason']}, event_count={audit['event_count']}, "
            f"content_chars={audit['content_chars']}, "
            f"usage_integrity={audit['usage_integrity']['status']}",
            audit=audit,
            validated_usage=validated_usage,
        )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = str(response.headers.get("Content-Type") or "").casefold()
            if "text/event-stream" not in content_type:
                fail(
                    f"expected SSE response, got content-type={content_type!r}",
                    event_integrity="failed",
                )
            for raw_line in response:
                try:
                    line = raw_line.decode("utf-8", errors="strict").strip()
                except (AttributeError, UnicodeDecodeError):
                    fail("SSE line is not valid UTF-8 bytes", event_integrity="failed")
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    fail("unexpected non-data SSE field", event_integrity="failed")
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    done_seen = True
                    break
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    fail("malformed SSE data event", event_integrity="failed")
                if not isinstance(event, dict):
                    fail("SSE data event must be a JSON object", event_integrity="failed")
                event_count += 1
                if usage_events:
                    fail(
                        "SSE JSON event observed after terminal usage object",
                        event_integrity="failed",
                    )
                if "error" in event and event["error"] is not None:
                    fail(
                        "SSE data event reported an error",
                        event_integrity="failed",
                        provider_error_receipt=_redacted_provider_error_receipt(
                            event["error"]
                        ),
                    )

                raw_usage = event.get("usage")
                if raw_usage is not None:
                    if not isinstance(raw_usage, dict):
                        fail("SSE usage must be a JSON object", event_integrity="failed")
                    usage = dict(raw_usage)
                    usage_events.append((event_count, usage))

                if "choices" not in event or not isinstance(event["choices"], list):
                    fail("SSE choices must be an array", event_integrity="failed")
                choices = event["choices"]
                if len(choices) > 1:
                    fail("SSE review stream must contain at most one choice", event_integrity="failed")
                if not choices:
                    if raw_usage is None:
                        fail(
                            "empty SSE choices event has no terminal usage object",
                            event_integrity="failed",
                        )
                    continue
                if raw_usage is not None:
                    fail(
                        "terminal SSE usage object must have empty choices",
                        event_integrity="failed",
                    )
                if finish_stop_event_index is not None:
                    fail("choice event observed after finish_reason=stop", event_integrity="failed")
                choice = choices[0]
                if not isinstance(choice, dict):
                    fail("SSE choice must be a JSON object", event_integrity="failed")
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    fail("SSE choice delta must be a JSON object", event_integrity="failed")
                fragment = delta.get("content")
                fragments: list[str] = []
                if isinstance(fragment, str):
                    fragments.append(fragment)
                elif isinstance(fragment, list):
                    for item in fragment:
                        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                            fail("SSE content part must contain text", event_integrity="failed")
                        fragments.append(item["text"])
                elif fragment is not None:
                    fail("SSE delta content has an unsupported type", event_integrity="failed")
                if fragments:
                    content_parts.extend(fragments)
                    content_event_count += 1

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    if not isinstance(finish_reason, str) or not finish_reason:
                        fail("SSE finish_reason must be a non-empty string", event_integrity="failed")
                    finish_reasons.append(finish_reason)
                    if finish_reason != "stop":
                        fail(
                            f"non-success SSE finish_reason={finish_reason}",
                            event_integrity="failed",
                        )
                    finish_stop_event_index = event_count
        stream_eof_reached = True
    except SSEIntegrityError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError):
        # Preserve the existing status-only HTTP audit and URL transport
        # classification in the bounded retry layer.  In particular, never
        # read an HTTPError response body here.
        raise
    except (http.client.IncompleteRead, OSError) as exc:
        fail(
            f"SSE transport did not reach a normal EOF ({type(exc).__name__})",
            event_integrity="failed",
            termination="transport_exception",
        )

    content = "".join(content_parts)
    if event_count < 1 or not content.strip():
        fail("incomplete SSE response has no review content")

    if done_seen:
        if finish_stop_event_index is None:
            fail("[DONE] stream is missing finish_reason=stop")
        usage_audit, validated_usage = _sse_usage_integrity(
            usage_events,
            finish_stop_event_index=finish_stop_event_index,
        )
        if usage_audit["status"] != "passed":
            fail("[DONE] stream failed terminal usage/event integrity")
        try:
            review = _extract_complete_json(content)
        except (json.JSONDecodeError, ValueError):
            fail("[DONE] SSE response is not one complete parseable JSON object")
        final_audit, validated_usage = audit_snapshot(
            event_integrity="passed",
            review_validation="deferred_to_existing_validation",
        )
        if integrity_audit is not None:
            integrity_audit.clear()
            integrity_audit.update(final_audit)
        return review, validated_usage

    if finish_stop_event_index is None:
        fail("normal EOF without [DONE] is missing finish_reason=stop")
    usage_audit, validated_usage = _sse_usage_integrity(
        usage_events,
        finish_stop_event_index=finish_stop_event_index,
    )
    if usage_audit["status"] != "passed":
        fail("normal EOF without [DONE] failed usage/event integrity")
    try:
        review = _extract_complete_json(content)
    except (json.JSONDecodeError, ValueError):
        fail("normal EOF without [DONE] is not one complete parseable JSON object")
    if eof_review_validator is None:
        fail("normal EOF without [DONE] has no existing review/schema/payload validator")
    try:
        validation_errors = list(eof_review_validator(review))
    except Exception as exc:  # pragma: no cover - validator failures are fail-closed
        fail(
            f"existing review/schema/payload validator raised {type(exc).__name__}",
            review_validation="failed",
        )
    if validation_errors:
        final_audit, _validated_usage = audit_snapshot(
            event_integrity="passed",
            review_validation="failed",
            termination="validated_stop_eof",
        )
        if integrity_audit is not None:
            integrity_audit.clear()
            integrity_audit.update(final_audit)
        # The transport is complete and the JSON object is safe to retain.  The
        # single authoritative validation path in ``_execute_one`` must persist
        # its raw/normalized provenance and reject it as a semantic attempt.
        return review, validated_usage
    final_audit, _validated_usage = audit_snapshot(
        event_integrity="passed",
        review_validation="passed",
    )
    if integrity_audit is not None:
        integrity_audit.clear()
        integrity_audit.update(final_audit)
    return review, validated_usage


def _execute_one(
    *,
    row: dict[str, Any],
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float,
    max_attempts: int,
    retry_delay: float,
    calibration: bool,
    extra_review_validator: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ]
    | None = None,
) -> dict[str, Any]:
    if calibration and (
        row.get("calibration_model") != model
        or row.get("calibration_endpoint") != endpoint
    ):
        raise ValueError("calibration execution model/endpoint contract mismatch")
    if not 1 <= max_attempts <= MAX_FAILURE_QUOTA_PER_CATEGORY:
        raise ValueError(
            f"max_attempts must be in [1, {MAX_FAILURE_QUOTA_PER_CATEGORY}]"
        )
    last_error = ""
    last_invalid_review: dict[str, Any] | None = None
    normalization_actions: list[str] = []
    attempt_usages: list[dict[str, Any]] = []
    invalid_attempts: list[dict[str, Any]] = []
    sse_integrity_attempts: list[dict[str, Any]] = []
    failure_attempts: list[dict[str, Any]] = []
    transport_failure_count = 0
    semantic_failure_count = 0
    hard_total_attempts = max_attempts * 2
    attempt = 0
    semantic_payload = json.loads(str(row["request"]["messages"][1]["content"]))

    def validate_eof_review(review: dict[str, Any]) -> list[str]:
        review, _dedupe_actions, _dedupe_raw_errors = (
            _dedupe_same_response_exact_variants(review, semantic_payload)
        )
        review, _coverage_actions, _coverage_raw_errors = (
            _canonicalize_same_response_coverage_statuses(
                review, semantic_payload
            )
        )
        projected, _edge_actions, _raw_errors = (
            _canonicalize_same_response_inverse_edges(review, semantic_payload)
        )
        normalized, actions = _normalize_review_structure(projected, semantic_payload)
        errors = (
            _validate_calibration(normalized, row["expected"])
            if calibration
            else _validate_review(normalized)
        )
        if actions:
            errors.append(
                "strict runner rejects structural normalization: "
                f"{len(actions)} action(s)"
            )
        errors.extend(_validate_against_payload(normalized, semantic_payload))
        if extra_review_validator is not None and not calibration:
            errors.extend(
                str(error)
                for error in extra_review_validator(normalized, semantic_payload)
                if str(error)
            )
        if calibration and errors == [_ABSTAIN_SENTINEL_VARIANT_ERROR]:
            projected, actions, raw_errors = _project_fresh_calibration_abstention(
                normalized, semantic_payload, row["expected"]
            )
            if actions and raw_errors == [_ABSTAIN_SENTINEL_VARIANT_ERROR]:
                return []
        return errors

    while (
        attempt < hard_total_attempts
        and transport_failure_count < max_attempts
        and semantic_failure_count < max_attempts
    ):
        attempt += 1
        current_sse_integrity: dict[str, Any] = {}
        try:
            request_body = dict(row["request"])
            request_body["model"] = model
            raw_review, usage = _call_api(
                endpoint=endpoint,
                api_key=api_key,
                request_body=request_body,
                timeout=timeout,
                eof_review_validator=validate_eof_review,
                integrity_audit=current_sse_integrity,
            )
            current_sse_integrity["attempt"] = attempt
            if calibration:
                current_sse_integrity["parsed_review_sha256"] = _sha256_json(
                    raw_review
                )
            sse_integrity_attempts.append(current_sse_integrity)
            attempt_usages.append({"attempt": attempt, **usage})
            review, dedupe_actions, dedupe_raw_validation_errors = (
                _dedupe_same_response_exact_variants(raw_review, semantic_payload)
            )
            review, coverage_actions, coverage_raw_validation_errors = (
                _canonicalize_same_response_coverage_statuses(
                    review, semantic_payload
                )
            )
            review, edge_actions, raw_validation_errors = (
                _canonicalize_same_response_inverse_edges(review, semantic_payload)
            )
            pre_normalization_review = deepcopy(review)
            review, normalization_actions = _normalize_review_structure(
                review, semantic_payload
            )
            validation_errors = (
                _validate_calibration(review, row["expected"])
                if calibration
                else _validate_review(review)
            )
            if normalization_actions:
                validation_errors.append(
                    "strict runner rejects structural normalization: "
                    f"{len(normalization_actions)} action(s)"
                )
            validation_errors.extend(_validate_against_payload(review, semantic_payload))
            if extra_review_validator is not None and not calibration:
                validation_errors.extend(
                    str(error)
                    for error in extra_review_validator(review, semantic_payload)
                    if str(error)
                )
            if (
                current_sse_integrity.get("review_validation") == "failed"
                and not validation_errors
            ):
                validation_errors.append(
                    "validated EOF prevalidation disagreed with authoritative validation"
                )
            if validation_errors:
                current_sse_integrity["review_validation"] = "failed"
                if calibration:
                    projected, projection_actions, raw_errors = (
                        _project_fresh_calibration_abstention(
                            review, semantic_payload, row["expected"]
                        )
                    )
                    integrity_usage = current_sse_integrity.get(
                        "usage_integrity"
                    ) or {}
                    if (
                        projection_actions
                        and raw_errors == [_ABSTAIN_SENTINEL_VARIANT_ERROR]
                        and transport_failure_count == 0
                        and current_sse_integrity.get("termination")
                        in {"done_sentinel", "validated_stop_eof"}
                        and current_sse_integrity.get("finish_reason") == "stop"
                        and integrity_usage.get("status") == "passed"
                    ):
                        current_sse_integrity["review_validation"] = (
                            "calibration_projectable_pass"
                        )
                        raw_hash = _sha256_json(review)
                        projected_hash = _sha256_json(projected)
                        observed_usage = {
                            key: sum(
                                int(item.get(key) or 0)
                                for item in attempt_usages
                                if isinstance(item.get(key), (int, float))
                            )
                            for key in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            )
                        }
                        audit = {
                            "policy": CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY,
                            "api_call_performed": True,
                            "formal_response_written": False,
                            "raw_model_contract_pass": False,
                            "raw_semantics_not_authoritative": True,
                            "projected_contract_pass": True,
                            "calibration_manifest_sha256": (
                                _calibration_manifest_sha256(row, semantic_payload)
                            ),
                            "calibration_execution_contract_sha256": row.get(
                                "calibration_execution_contract_sha256"
                            ),
                            "raw_review_sha256": raw_hash,
                            "terminal_content_sha256": current_sse_integrity.get(
                                "content_sha256"
                            ),
                            "stream_content_sha256_authoritative": False,
                            "projected_review_sha256": projected_hash,
                            "raw_validation_errors": raw_errors,
                            "actions": projection_actions,
                            "source_attempt": attempt,
                            "usage_sha256": _sha256_json(observed_usage),
                        }
                        return {
                            "custom_id": row["custom_id"],
                            "status": "calibration_projectable_pass",
                            "attempt": attempt,
                            "model": model,
                            "runner_version": RUNNER_VERSION,
                            "transport": TRANSPORT,
                            "sse_completion_policy": SSE_COMPLETION_POLICY,
                            "prompt_sha256": row["prompt_sha256"],
                            "calibration_execution_contract_sha256": row.get(
                                "calibration_execution_contract_sha256"
                            ),
                            "calibration_manifest_sha256": (
                                _calibration_manifest_sha256(row, semantic_payload)
                            ),
                            "calibration_endpoint": endpoint,
                            "raw_review": review,
                            "projected_review": projected,
                            "response_sha256": projected_hash,
                            "usage_is_lower_bound": transport_failure_count > 0,
                            "usage": observed_usage,
                            "successful_response_usage": usage,
                            "attempt_usage": attempt_usages,
                            "sse_integrity": dict(current_sse_integrity),
                            "sse_integrity_attempts": sse_integrity_attempts,
                            "calibration_abstain_sentinel_projection": audit,
                            "completed_at": datetime.now(UTC).isoformat(),
                        }
                last_invalid_review = review
                raw_review_sha256 = _sha256_json(raw_review)
                pre_normalization_review_sha256 = _sha256_json(
                    pre_normalization_review
                )
                review_sha256 = _sha256_json(review)
                invalid_attempts.append(
                    {
                        "attempt": attempt,
                        "raw_review": raw_review,
                        "raw_review_sha256": raw_review_sha256,
                        "pre_normalization_review_sha256": (
                            pre_normalization_review_sha256
                        ),
                        "review_sha256": review_sha256,
                        "review": review,
                        "normalization_provenance": {
                            "policy": RAW_NORMALIZATION_PROVENANCE_POLICY,
                            "raw_review_sha256": raw_review_sha256,
                            "pre_normalization_review_sha256": (
                                pre_normalization_review_sha256
                            ),
                            "normalized_review_sha256": review_sha256,
                            "actions": normalization_actions,
                        },
                        "validation_errors": validation_errors,
                        "sse_integrity": dict(current_sse_integrity),
                    }
                )
                raise ValueError("; ".join(validation_errors))
            current_sse_integrity["review_validation"] = "passed"
            observed_usage = {
                key: sum(
                    int(item.get(key) or 0)
                    for item in attempt_usages
                    if isinstance(item.get(key), (int, float))
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            succeeded = {
                "custom_id": row["custom_id"],
                "status": "succeeded",
                "attempt": attempt,
                "model": model,
                "runner_version": RUNNER_VERSION,
                "transport": TRANSPORT,
                "sse_completion_policy": SSE_COMPLETION_POLICY,
                "prompt_sha256": row["prompt_sha256"],
                "review": review,
                "response_sha256": _sha256_json(review),
                "usage_is_lower_bound": transport_failure_count > 0,
                "usage": observed_usage,
                "successful_response_usage": usage,
                "attempt_usage": attempt_usages,
                "sse_integrity": dict(current_sse_integrity),
                "sse_integrity_attempts": sse_integrity_attempts,
                "completed_at": datetime.now(UTC).isoformat(),
                "retry_accounting": {
                    "policy": RETRY_ACCOUNTING_POLICY,
                    "transport_failure_quota": max_attempts,
                    "semantic_failure_quota": max_attempts,
                    "hard_total_attempt_quota": hard_total_attempts,
                    "total_attempts": attempt,
                    "transport_failure_count": transport_failure_count,
                    "semantic_failure_count": semantic_failure_count,
                    "quota_exhausted": None,
                },
            }
            if calibration:
                succeeded.update(
                    {
                        "calibration_execution_contract_sha256": row.get(
                            "calibration_execution_contract_sha256"
                        ),
                        "calibration_manifest_sha256": _calibration_manifest_sha256(
                            row, semantic_payload
                        ),
                        "calibration_endpoint": endpoint,
                    }
                )
            if edge_actions:
                raw_review_sha256 = _sha256_json(raw_review)
                succeeded["same_response_inverse_edge_canonicalization"] = {
                    "policy": INVERSE_EDGE_CANONICALIZATION_POLICY,
                    "api_call_performed": True,
                    "raw_review_sha256": raw_review_sha256,
                    "projected_review_sha256": _sha256_json(review),
                    "raw_validation_errors": raw_validation_errors,
                    "actions": edge_actions,
                    "source": {
                        "source_attempt": attempt,
                        "raw_review_sha256": raw_review_sha256,
                    },
                }
            if dedupe_actions:
                raw_review_sha256 = _sha256_json(raw_review)
                succeeded["same_response_exact_variant_deduplication"] = {
                    "policy": EXACT_VARIANT_DEDUPLICATION_POLICY,
                    "api_call_performed": True,
                    "raw_review_sha256": raw_review_sha256,
                    "projected_review_sha256": _sha256_json(review),
                    "raw_validation_errors": dedupe_raw_validation_errors,
                    "actions": dedupe_actions,
                    "source": {
                        "source_attempt": attempt,
                        "raw_review_sha256": raw_review_sha256,
                    },
                }
            if coverage_actions:
                raw_review_sha256 = _sha256_json(raw_review)
                succeeded["same_response_coverage_status_canonicalization"] = {
                    "policy": COVERAGE_STATUS_CANONICALIZATION_POLICY,
                    "api_call_performed": True,
                    "raw_review_sha256": raw_review_sha256,
                    "projected_review_sha256": _sha256_json(review),
                    "raw_validation_errors": coverage_raw_validation_errors,
                    "actions": coverage_actions,
                    "source": {
                        "source_attempt": attempt,
                        "raw_review_sha256": raw_review_sha256,
                    },
                }
            if normalization_actions:
                succeeded["structural_normalization"] = normalization_actions
            if invalid_attempts:
                succeeded["invalid_attempts_before_success"] = invalid_attempts
            if failure_attempts:
                succeeded["failures_before_success"] = failure_attempts
            return succeeded
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            failure_termination = "review_or_parser_error"
            if isinstance(exc, SSEIntegrityError):
                sse_integrity_attempts.append({"attempt": attempt, **exc.audit})
                failure_termination = str(exc.audit.get("termination") or "sse_integrity_error")
                if exc.validated_usage:
                    attempt_usages.append(
                        {
                            "attempt": attempt,
                            **exc.validated_usage,
                            "response_status": "rejected_sse_integrity",
                        }
                    )
            if isinstance(exc, urllib.error.HTTPError):
                failure_termination = "http_error"
                status_phrase = http.client.responses.get(int(exc.code), "HTTP error")
                last_error = f"HTTPError: status={exc.code} reason={status_phrase}"
            elif isinstance(
                exc,
                (
                    urllib.error.URLError,
                    http.client.IncompleteRead,
                    TimeoutError,
                ),
            ):
                failure_termination = "transport_error"
                last_error = f"{type(exc).__name__}: transport request failed"
            else:
                last_error = f"{type(exc).__name__}: {exc}"
                if invalid_attempts and invalid_attempts[-1].get("attempt") == attempt:
                    failure_termination = "review_validation_error"
            terminal_usage_complete = False
            response_sha256: str | None = None
            if isinstance(exc, SSEIntegrityError):
                usage_integrity = exc.audit.get("usage_integrity") or {}
                terminal_usage_complete = bool(
                    exc.audit.get("finish_reason_stop_seen")
                    and usage_integrity.get("status") == "passed"
                )
                response_sha256 = (
                    str(exc.audit.get("response_sha256"))
                    if exc.audit.get("response_sha256")
                    else str(exc.audit.get("content_sha256"))
                    if exc.audit.get("content_sha256")
                    else None
                )
            elif invalid_attempts and invalid_attempts[-1].get("attempt") == attempt:
                terminal_usage_complete = True
                response_sha256 = str(invalid_attempts[-1]["raw_review_sha256"])
            elif current_sse_integrity:
                usage_integrity = current_sse_integrity.get("usage_integrity") or {}
                terminal_usage_complete = bool(
                    current_sse_integrity.get("finish_reason_stop_seen")
                    and usage_integrity.get("status") == "passed"
                )
            transport_exception = isinstance(
                exc,
                (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    http.client.IncompleteRead,
                    TimeoutError,
                ),
            )
            failure_category = (
                "semantic"
                if terminal_usage_complete or failure_termination == "review_validation_error"
                else "transport"
                if isinstance(exc, SSEIntegrityError) or transport_exception
                else "semantic"
            )
            if failure_category == "transport":
                transport_failure_count += 1
            else:
                semantic_failure_count += 1
            complete_json_hashed = bool(
                isinstance(exc, SSEIntegrityError)
                and exc.audit.get("response_sha256")
                or invalid_attempts
                and invalid_attempts[-1].get("attempt") == attempt
            )
            response_hash_status = (
                "complete_json_sha256"
                if complete_json_hashed
                else "terminal_stream_content_sha256"
                if terminal_usage_complete and response_sha256 is not None
                else "partial_stream_content_sha256"
                if response_sha256 is not None
                else "not_available_incomplete_transport_or_parser_failure"
            )
            failure_record: dict[str, Any] = {
                "attempt": attempt,
                "termination": failure_termination,
                "error_type": type(exc).__name__,
                "failure_category": failure_category,
                "response_sha256": response_sha256,
                "response_hash_status": response_hash_status,
            }
            if isinstance(exc, urllib.error.HTTPError):
                failure_record["http_status"] = int(exc.code)
                failure_record["http_reason"] = http.client.responses.get(
                    int(exc.code), "HTTP error"
                )
            failure_attempts.append(failure_record)
            if (
                attempt < hard_total_attempts
                and transport_failure_count < max_attempts
                and semantic_failure_count < max_attempts
            ):
                time.sleep(retry_delay)
    quota_exhausted = (
        "hard_total"
        if attempt >= hard_total_attempts
        else "transport"
        if transport_failure_count >= max_attempts
        else "semantic"
        if semantic_failure_count >= max_attempts
        else "hard_total"
    )
    failed = {
        "custom_id": row["custom_id"],
        "status": "failed",
        "attempt": attempt,
        "model": model,
        "runner_version": RUNNER_VERSION,
        "transport": TRANSPORT,
        "sse_completion_policy": SSE_COMPLETION_POLICY,
        "prompt_sha256": row["prompt_sha256"],
        "error": last_error,
        "completed_at": datetime.now(UTC).isoformat(),
        "usage_is_lower_bound": transport_failure_count > 0,
        "retry_accounting": {
            "policy": RETRY_ACCOUNTING_POLICY,
            "transport_failure_quota": max_attempts,
            "semantic_failure_quota": max_attempts,
            "hard_total_attempt_quota": hard_total_attempts,
            "total_attempts": attempt,
            "transport_failure_count": transport_failure_count,
            "semantic_failure_count": semantic_failure_count,
            "quota_exhausted": quota_exhausted,
        },
    }
    if last_invalid_review is not None:
        failed["invalid_review"] = last_invalid_review
    if invalid_attempts:
        failed["invalid_attempts"] = invalid_attempts
    if sse_integrity_attempts:
        failed["sse_integrity"] = dict(sse_integrity_attempts[-1])
        failed["sse_integrity_attempts"] = sse_integrity_attempts
    if failure_attempts:
        failed["failure_termination"] = failure_attempts[-1]["termination"]
        failed["failure_attempts"] = failure_attempts
    if attempt_usages:
        failed["attempt_usage"] = attempt_usages
        failed["usage"] = {
            key: sum(
                int(item.get(key) or 0)
                for item in attempt_usages
                if isinstance(item.get(key), (int, float))
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    return failed


def _successful_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("custom_id") or ""): row
        for row in _read_jsonl(path)
        if row.get("status") == "succeeded" and row.get("custom_id")
    }


def _validate_fresh_calibration_terminal_ledger(
    result: dict[str, Any], *, terminal_review: dict[str, Any], projectable: bool
) -> list[str]:
    # The terminal stream digest is diagnostic because raw stream bytes are not
    # persisted. Authorization instead binds to the independently reproducible
    # parsed review hash below; storing raw provider output would expand the
    # sensitive-data surface without strengthening semantic validation.
    errors: list[str] = []
    attempt = result.get("attempt")
    attempt_usage = result.get("attempt_usage") or []
    sse_attempts = result.get("sse_integrity_attempts") or []
    integrity = result.get("sse_integrity") or {}
    retry = result.get("retry_accounting") or {}
    expected_integrity_keys = {
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
        "parsed_review_sha256",
        "event_integrity",
        "usage_integrity",
        "review_validation",
        "attempt",
    }
    if (
        type(attempt) is not int
        or attempt < 1
        or len(attempt_usage) != attempt
        or len(sse_attempts) != attempt
        or [item.get("attempt") for item in attempt_usage]
        != list(range(1, attempt + 1))
        or [item.get("attempt") for item in sse_attempts]
        != list(range(1, attempt + 1))
        or sse_attempts[-1] != integrity
    ):
        errors.append("calibration attempt/SSE ledger is not contiguous")
        return errors
    for usage in attempt_usage:
        if (
            set(usage)
            != {"attempt", "prompt_tokens", "completion_tokens", "total_tokens"}
            or any(
                type(usage.get(key)) is not int or usage[key] < 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            or usage.get("total_tokens")
            != usage.get("prompt_tokens") + usage.get("completion_tokens")
        ):
            errors.append("calibration attempt usage is invalid")
    expected_usage = {
        key: sum(item[key] for item in attempt_usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    final_usage = {
        key: attempt_usage[-1][key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    usage_integrity = integrity.get("usage_integrity") or {}
    if (
        result.get("usage") != expected_usage
        or result.get("successful_response_usage") != final_usage
        or result.get("usage_is_lower_bound") is not False
        or integrity.get("event_integrity") != "passed"
        or set(integrity) != expected_integrity_keys
        or integrity.get("completion_policy") != SSE_COMPLETION_POLICY
        or integrity.get("termination")
        not in {"done_sentinel", "validated_stop_eof"}
        or integrity.get("done_seen")
        is not (integrity.get("termination") == "done_sentinel")
        or integrity.get("finish_reason") != "stop"
        or integrity.get("finish_reasons") != ["stop"]
        or integrity.get("finish_reason_stop_seen") is not True
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(integrity.get("content_sha256") or "")
        )
        or type(integrity.get("event_count")) is not int
        or integrity["event_count"] < 2
        or type(integrity.get("content_event_count")) is not int
        # ``event_count`` counts parsed JSON SSE frames, not the out-of-band
        # ``[DONE]`` sentinel.  A valid terminal stream therefore needs at
        # least one non-content JSON frame, but not two: DashScope commonly
        # emits N content frames plus one terminal usage/finish frame.
        or not 1 <= integrity["content_event_count"] <= integrity["event_count"] - 1
        or type(integrity.get("content_chars")) is not int
        or integrity["content_chars"]
        < len(json.dumps(terminal_review, ensure_ascii=False, separators=(",", ":")))
        or integrity.get("parsed_review_sha256") != _sha256_json(terminal_review)
        or usage_integrity
        != {
            "status": "passed",
            "usage_event_count": 1,
            "required_token_fields_valid": True,
            "total_matches_prompt_plus_completion": True,
            "at_or_after_finish_reason_stop": True,
        }
    ):
        errors.append("calibration terminal/usage integrity is invalid")
    if projectable:
        if attempt != 1 or retry:
            errors.append("projectable calibration must be one fresh API attempt")
    else:
        semantic_failures = len(result.get("invalid_attempts_before_success") or [])
        transport_failures = len(result.get("failures_before_success") or [])
        if (
            transport_failures != 0
            or retry.get("policy") != RETRY_ACCOUNTING_POLICY
            or retry.get("total_attempts") != attempt
            or retry.get("transport_failure_count") != 0
            or retry.get("semantic_failure_count") != semantic_failures
            or retry.get("quota_exhausted") is not None
            or semantic_failures + 1 != attempt
        ):
            errors.append("raw calibration retry ledger is invalid")
    return errors


def _validate_calibration_result(
    result: dict[str, Any], request_row: dict[str, Any]
) -> list[str]:
    """Replay one current-runner calibration result without trusting status."""

    errors: list[str] = []
    semantic_payload = json.loads(
        str(request_row["request"]["messages"][1]["content"])
    )
    try:
        completed_at = datetime.fromisoformat(str(result.get("completed_at") or ""))
    except ValueError:
        return ["calibration completion timestamp is invalid"]
    if completed_at.tzinfo is None or completed_at.utcoffset() != UTC.utcoffset(None):
        return ["calibration completion timestamp must be UTC"]
    if (
        result.get("custom_id") != request_row.get("custom_id")
        or result.get("prompt_sha256") != request_row.get("prompt_sha256")
        or result.get("runner_version") != RUNNER_VERSION
        or result.get("model") != request_row.get("calibration_model")
    ):
        return ["calibration result identity/runner mismatch"]
    status = result.get("status")
    integrity = result.get("sse_integrity") or {}
    usage = result.get("usage")
    if (
        result.get("transport") != TRANSPORT
        or result.get("sse_completion_policy") != SSE_COMPLETION_POLICY
        or not isinstance(result.get("attempt"), int)
        or int(result["attempt"]) < 1
        or not isinstance(usage, dict)
        or result.get("usage_is_lower_bound") is not False
        or result.get("attempt_usage") in (None, [])
        or integrity.get("termination")
        not in {"done_sentinel", "validated_stop_eof"}
        or integrity.get("finish_reason") != "stop"
        or (integrity.get("usage_integrity") or {}).get("status") != "passed"
    ):
        errors.append("calibration terminal/usage envelope is invalid")
    common_keys = {
        "custom_id",
        "status",
        "attempt",
        "model",
        "runner_version",
        "transport",
        "sse_completion_policy",
        "prompt_sha256",
        "response_sha256",
        "usage_is_lower_bound",
        "usage",
        "successful_response_usage",
        "attempt_usage",
        "sse_integrity",
        "sse_integrity_attempts",
        "completed_at",
        "calibration_execution_contract_sha256",
        "calibration_manifest_sha256",
        "calibration_endpoint",
    }
    raw_success_required = common_keys | {"review", "retry_accounting"}
    raw_success_optional = {
        "same_response_inverse_edge_canonicalization",
        "same_response_exact_variant_deduplication",
        "same_response_coverage_status_canonicalization",
        "invalid_attempts_before_success",
        "failures_before_success",
    }
    projectable_required = common_keys | {
        "raw_review",
        "projected_review",
        "calibration_abstain_sentinel_projection",
    }
    if status == "succeeded":
        forbidden_projection_keys = {
            "raw_review",
            "projected_review",
            "calibration_abstain_sentinel_projection",
        }
        if forbidden_projection_keys.intersection(result):
            errors.append("raw calibration success contains projection fields")
        if not raw_success_required.issubset(result) or not set(result).issubset(
            raw_success_required | raw_success_optional
        ):
            errors.append("raw calibration success schema is invalid")
        review = result.get("review")
        if not isinstance(review, dict):
            return ["raw calibration success must contain review"]
        errors.extend(_validate_calibration(review, request_row["expected"]))
        errors.extend(_validate_against_payload(review, semantic_payload))
        if result.get("response_sha256") != _sha256_json(review):
            errors.append("raw calibration success response hash mismatch")
        if (
            result.get("calibration_execution_contract_sha256")
            != request_row.get("calibration_execution_contract_sha256")
            or result.get("calibration_manifest_sha256")
            != _calibration_manifest_sha256(request_row, semantic_payload)
            or result.get("calibration_endpoint")
            != request_row.get("calibration_endpoint")
        ):
            errors.append("raw calibration execution contract mismatch")
        if integrity.get("review_validation") != "passed":
            errors.append("raw calibration success validation receipt is invalid")
        errors.extend(
            _validate_fresh_calibration_terminal_ledger(
                result, terminal_review=review, projectable=False
            )
        )
        attempt_usage = result.get("attempt_usage") or []
        expected_usage = {
            key: sum(
                int(item.get(key) or 0)
                for item in attempt_usage
                if isinstance(item, dict)
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        final_attempt_usage = attempt_usage[-1] if attempt_usage else {}
        expected_success_usage = {
            key: final_attempt_usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        if (
            result.get("usage") != expected_usage
            or result.get("successful_response_usage") != expected_success_usage
            or len(result.get("sse_integrity_attempts") or [])
            != result.get("attempt")
            or result.get("sse_integrity_attempts")[-1] != integrity
        ):
            errors.append("raw calibration success usage/attempt ledger is inconsistent")
        return errors
    if status != "calibration_projectable_pass":
        return ["calibration result status is not accepted"]
    if request_row["expected"].get("calibration_projectable_abstention") is not True:
        errors.append("raw calibration case cannot use projectable status")
    if set(result) != projectable_required:
        errors.append("projectable calibration result schema is invalid")
    if "review" in result:
        errors.append("projectable calibration must not contain production review")
    expected_keys = {
        "policy",
        "api_call_performed",
        "formal_response_written",
        "raw_model_contract_pass",
        "raw_semantics_not_authoritative",
        "projected_contract_pass",
        "calibration_manifest_sha256",
        "calibration_execution_contract_sha256",
        "raw_review_sha256",
        "terminal_content_sha256",
        "stream_content_sha256_authoritative",
        "projected_review_sha256",
        "raw_validation_errors",
        "actions",
        "source_attempt",
        "usage_sha256",
    }
    audit = result.get("calibration_abstain_sentinel_projection")
    raw = result.get("raw_review")
    projected = result.get("projected_review")
    if not isinstance(audit, dict) or set(audit) != expected_keys:
        return ["calibration projection receipt schema is invalid"]
    if not isinstance(raw, dict) or not isinstance(projected, dict):
        return ["calibration projection reviews must be objects"]
    replay, actions, raw_errors = _project_fresh_calibration_abstention(
        raw, semantic_payload, request_row["expected"]
    )
    if (
        audit.get("policy") != CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY
        or audit.get("api_call_performed") is not True
        or audit.get("formal_response_written") is not False
        or audit.get("raw_model_contract_pass") is not False
        or audit.get("raw_semantics_not_authoritative") is not True
        or audit.get("projected_contract_pass") is not True
    ):
        errors.append("calibration projection authority flags are invalid")
    if audit.get("calibration_manifest_sha256") != _calibration_manifest_sha256(
        request_row, semantic_payload
    ):
        errors.append("calibration projection manifest hash mismatch")
    if audit.get("calibration_execution_contract_sha256") != request_row.get(
        "calibration_execution_contract_sha256"
    ):
        errors.append("calibration execution-contract hash mismatch")
    if (
        result.get("calibration_execution_contract_sha256")
        != request_row.get("calibration_execution_contract_sha256")
        or result.get("calibration_manifest_sha256")
        != _calibration_manifest_sha256(request_row, semantic_payload)
        or result.get("calibration_endpoint")
        != request_row.get("calibration_endpoint")
    ):
        errors.append("projectable calibration execution envelope mismatch")
    if (
        audit.get("raw_review_sha256") != _sha256_json(raw)
        or audit.get("projected_review_sha256") != _sha256_json(projected)
        or result.get("response_sha256") != _sha256_json(projected)
        or projected != replay
        or audit.get("actions") != actions
        or audit.get("raw_validation_errors") != raw_errors
    ):
        errors.append("calibration projection action/hash replay mismatch")
    if audit.get("stream_content_sha256_authoritative") is not False:
        errors.append("stream content hash must remain diagnostic-only")
    errors.extend(
        _validate_fresh_calibration_terminal_ledger(
            result, terminal_review=raw, projectable=True
        )
    )
    if (
        audit.get("usage_sha256") != _sha256_json(usage)
        or audit.get("source_attempt") != result.get("attempt")
        or result.get("attempt") != 1
        or len(result.get("attempt_usage") or []) != 1
        or len(result.get("sse_integrity_attempts") or []) != 1
        or result.get("sse_integrity_attempts")[-1] != integrity
        or integrity.get("attempt") != 1
        or integrity.get("termination")
        not in {"done_sentinel", "validated_stop_eof"}
        or integrity.get("done_seen")
        is not (integrity.get("termination") == "done_sentinel")
        or integrity.get("finish_reason") != "stop"
        or integrity.get("finish_reasons") != ["stop"]
        or integrity.get("finish_reason_stop_seen") is not True
        or integrity.get("event_integrity") != "passed"
        or not isinstance(integrity.get("event_count"), int)
        or int(integrity["event_count"]) < 1
        or not isinstance(integrity.get("content_event_count"), int)
        or int(integrity["content_event_count"]) < 1
        or not isinstance(integrity.get("content_chars"), int)
        or int(integrity["content_chars"]) < 1
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(integrity.get("content_sha256") or "")
        )
        or integrity.get("parsed_review_sha256") != _sha256_json(raw)
        or (integrity.get("usage_integrity") or {}).get("status") != "passed"
        or integrity.get("review_validation") != "calibration_projectable_pass"
    ):
        errors.append("calibration projection terminal/usage receipt is invalid")
    attempt_usage = (result.get("attempt_usage") or [{}])[0]
    usage_integrity = integrity.get("usage_integrity") or {}
    expected_success_usage = {
        key: attempt_usage.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if (
        set(attempt_usage)
        != {"attempt", "prompt_tokens", "completion_tokens", "total_tokens"}
        or attempt_usage.get("attempt") != 1
        or any(
            type(attempt_usage.get(key)) is not int or int(attempt_usage[key]) < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        or attempt_usage.get("total_tokens")
        != attempt_usage.get("prompt_tokens")
        + attempt_usage.get("completion_tokens")
        or usage_integrity.get("usage_event_count") != 1
        or usage_integrity.get("required_token_fields_valid") is not True
        or usage_integrity.get("total_matches_prompt_plus_completion") is not True
        or usage_integrity.get("at_or_after_finish_reason_stop") is not True
        or
        result.get("successful_response_usage") != expected_success_usage
        or usage != expected_success_usage
    ):
        errors.append("calibration projection usage totals are inconsistent")
    return errors


def _completed_calibrations_by_id(
    path: Path, requests: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    request_by_id = {str(row["custom_id"]): row for row in requests}
    latest_by_id: dict[str, dict[str, Any]] = {}
    for result in _read_jsonl(path):
        custom_id = str(result.get("custom_id") or "")
        if custom_id in request_by_id:
            latest_by_id[custom_id] = result
    completed: dict[str, dict[str, Any]] = {}
    for custom_id, result in latest_by_id.items():
        request_row = request_by_id[custom_id]
        if not _validate_calibration_result(result, request_row):
            completed[custom_id] = result
    return completed


def _usage_totals(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage") or {}
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def _attempt_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(int(row.get("attempt") or 0) for row in rows)


def _dry_run_summary(
    requests: Sequence[dict[str, Any]], prompt_hash: str, prepared_model: str
) -> dict[str, Any]:
    completed = _successful_by_id(RESPONSES_PATH)
    calibration_requests = _calibration_requests(
        requests[0], model=DEFAULT_MODEL, endpoint=_endpoint(DEFAULT_BASE_URL)
    )
    calibration_completed = _completed_calibrations_by_id(
        CALIBRATION_RESPONSES_PATH, calibration_requests
    )
    response_rows = _read_jsonl(RESPONSES_PATH)
    calibration_rows = _read_jsonl(CALIBRATION_RESPONSES_PATH)
    return {
        "status": "dry_run_only",
        "prepared_case_calls": len(requests),
        "calibration_calls": len(_calibration_cases()),
        "planned_total_calls_without_resume": len(requests) + len(_calibration_cases()),
        "already_completed_case_calls": len(completed),
        "already_completed_calibration_calls": len(calibration_completed),
        "raw_contract_calibration_passes": sum(
            row.get("status") == "succeeded"
            for row in calibration_completed.values()
        ),
        "projectable_calibration_passes": sum(
            row.get("status") == "calibration_projectable_pass"
            for row in calibration_completed.values()
        ),
        "remaining_case_calls": len(requests) - len(completed),
        "formal_api_attempts_recorded": _attempt_count(response_rows),
        "calibration_api_attempts_recorded": _attempt_count(calibration_rows),
        "observed_usage_lower_bound": _usage_totals([*response_rows, *calibration_rows]),
        "prompt_sha256": prompt_hash,
        "runner_version": RUNNER_VERSION,
        "transport": TRANSPORT,
        "sse_completion_policy": SSE_COMPLETION_POLICY,
        "prepared_model": prepared_model,
        "default_execution_model": DEFAULT_MODEL,
        "price_amount": None,
        "price_note": "The API response may provide token usage but not a monetary amount; no price is guessed.",
        "semantic_payload_redaction": "passed",
        "semantic_payload_excludes": sorted(_FORBIDDEN_KEYS),
        "execution_gates": [
            "--execute",
            "--acknowledge-cost",
            "--confirm-full-review",
            "--confirm-no-concurrent-review",
            "four independently replayed synthetic calibration results",
            f"non-empty {DEFAULT_KEY_ENV}",
        ],
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform paid API calls")
    parser.add_argument("--acknowledge-cost", action="store_true")
    parser.add_argument("--confirm-full-review", action="store_true")
    parser.add_argument("--confirm-no-concurrent-review", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="remove an execution lock only after externally verifying its process is gone",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default=DEFAULT_KEY_ENV)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "per-category transport and semantic failure quota (1..3); hard total "
            "attempt limit is twice this value"
        ),
    )
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    requests, prompt_hash, prepared_model = _load_requests()
    dry_run = _dry_run_summary(requests, prompt_hash, prepared_model)
    if not args.execute:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0

    missing_gates = [
        name
        for enabled, name in (
            (args.acknowledge_cost, "--acknowledge-cost"),
            (args.confirm_full_review, "--confirm-full-review"),
            (args.confirm_no_concurrent_review, "--confirm-no-concurrent-review"),
        )
        if not enabled
    ]
    if missing_gates:
        raise SystemExit("paid execution refused; missing gate(s): " + ", ".join(missing_gates))
    _acquire_execution_lock(recover_stale=args.recover_stale_lock)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"paid execution refused; environment variable {args.api_key_env} is empty")
    if not 1 <= args.max_attempts <= MAX_FAILURE_QUOTA_PER_CATEGORY:
        raise SystemExit(
            f"--max-attempts must be in [1, {MAX_FAILURE_QUOTA_PER_CATEGORY}]"
        )

    endpoint = _endpoint(args.base_url)
    calibration_requests = _calibration_requests(
        requests[0], model=args.model, endpoint=endpoint
    )
    calibration_done = _completed_calibrations_by_id(
        CALIBRATION_RESPONSES_PATH, calibration_requests
    )
    for row in calibration_requests:
        if row["custom_id"] in calibration_done:
            continue
        result = _execute_one(
            row=row,
            endpoint=endpoint,
            api_key=api_key,
            model=args.model,
            timeout=args.request_timeout,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            calibration=True,
        )
        _append_jsonl(CALIBRATION_RESPONSES_PATH, result)
        if _validate_calibration_result(result, row):
            raise SystemExit(
                f"calibration failed for {row['custom_id']}; full review was not started"
            )
        calibration_done[row["custom_id"]] = result
    if len(calibration_done) != len(calibration_requests):
        raise SystemExit("calibration set is incomplete; full review was not started")
    if args.calibration_only:
        print(
            json.dumps(
                {
                    "status": "calibration_succeeded",
                    "runner_version": RUNNER_VERSION,
                    "calibration_calls": len(calibration_done),
                    "raw_contract_calibration_passes": sum(
                        row.get("status") == "succeeded"
                        for row in calibration_done.values()
                    ),
                    "projectable_calibration_passes": sum(
                        row.get("status") == "calibration_projectable_pass"
                        for row in calibration_done.values()
                    ),
                    "prompt_sha256": prompt_hash,
                    "usage": _usage_totals(list(calibration_done.values())),
                    "all_runner_calibration_usage_lower_bound": _usage_totals(
                        _read_jsonl(CALIBRATION_RESPONSES_PATH)
                    ),
                    "api_attempts_recorded": _attempt_count(
                        _read_jsonl(CALIBRATION_RESPONSES_PATH)
                    ),
                    "full_review_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    completed = _successful_by_id(RESPONSES_PATH)
    for index, row in enumerate(requests, start=1):
        custom_id = str(row["custom_id"])
        if custom_id in completed:
            continue
        recovered = _recover_prior_invalid_review(
            row=row, response_rows=_read_jsonl(RESPONSES_PATH)
        )
        if recovered is not None:
            recovered["sequence"] = index
            _append_jsonl(RESPONSES_PATH, recovered)
            completed[custom_id] = recovered
            continue
        result = _execute_one(
            row=row,
            endpoint=endpoint,
            api_key=api_key,
            model=args.model,
            timeout=args.request_timeout,
            max_attempts=args.max_attempts,
            retry_delay=args.retry_delay,
            calibration=False,
        )
        result["sequence"] = index
        _append_jsonl(RESPONSES_PATH, result)
        if result["status"] == "succeeded":
            completed[custom_id] = result
        else:
            summary = {
                "status": "incomplete_stopped_on_failure",
                "model": args.model,
                "runner_version": RUNNER_VERSION,
                "transport": TRANSPORT,
                "sse_completion_policy": SSE_COMPLETION_POLICY,
                "prompt_sha256": prompt_hash,
                "calibration_succeeded": True,
                "raw_contract_calibration_passes": sum(
                    row.get("status") == "succeeded"
                    for row in calibration_done.values()
                ),
                "projectable_calibration_passes": sum(
                    row.get("status") == "calibration_projectable_pass"
                    for row in calibration_done.values()
                ),
                "current_runner_calibration_usage": _usage_totals(
                    list(calibration_done.values())
                ),
                "case_success_count": len(completed),
                "case_expected_count": EXPECTED_CASE_CALLS,
                "failed_custom_ids": [custom_id],
                "usage": _usage_totals(_read_jsonl(RESPONSES_PATH)),
                "formal_api_attempts_recorded": _attempt_count(
                    _read_jsonl(RESPONSES_PATH)
                ),
                "price_amount": None,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            _write_json(EXECUTION_SUMMARY_PATH, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

    all_rows = _read_jsonl(RESPONSES_PATH)
    summary = {
        "status": "succeeded" if len(completed) == EXPECTED_CASE_CALLS else "incomplete",
        "model": args.model,
        "runner_version": RUNNER_VERSION,
        "transport": TRANSPORT,
        "sse_completion_policy": SSE_COMPLETION_POLICY,
        "prompt_sha256": prompt_hash,
        "calibration_succeeded": len(calibration_done) == len(calibration_requests),
        "raw_contract_calibration_passes": sum(
            row.get("status") == "succeeded" for row in calibration_done.values()
        ),
        "projectable_calibration_passes": sum(
            row.get("status") == "calibration_projectable_pass"
            for row in calibration_done.values()
        ),
        "current_runner_calibration_usage": _usage_totals(list(calibration_done.values())),
        "current_runner_calibration_api_attempts_recorded": _attempt_count(
            list(calibration_done.values())
        ),
        "case_success_count": len(completed),
        "case_expected_count": EXPECTED_CASE_CALLS,
        "failed_custom_ids": [],
        "usage": _usage_totals(all_rows),
        "formal_api_attempts_recorded": _attempt_count(all_rows),
        "calibration_api_attempts_recorded": _attempt_count(
            _read_jsonl(CALIBRATION_RESPONSES_PATH)
        ),
        "usage_is_lower_bound": any(
            bool(row.get("usage_is_lower_bound"))
            or int(row.get("attempt") or 0) > 0
            and not row.get("usage")
            for row in [*all_rows, *_read_jsonl(CALIBRATION_RESPONSES_PATH)]
        ),
        "price_amount": None,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(EXECUTION_SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
