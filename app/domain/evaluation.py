"""TODO-18 evaluation domain invariants and stable state transitions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Any, Mapping


class EvaluationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeedbackStatus(StrEnum):
    PENDING = "pending"
    TRIAGED = "triaged"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AnnotationStatus(StrEnum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    CONFLICT = "conflict"
    APPROVED = "approved"
    REJECTED = "rejected"


RUN_TRANSITIONS = {
    EvaluationRunStatus.QUEUED: {
        EvaluationRunStatus.RUNNING,
        EvaluationRunStatus.CANCELLED,
        EvaluationRunStatus.FAILED,
    },
    EvaluationRunStatus.RUNNING: {
        EvaluationRunStatus.SUCCEEDED,
        EvaluationRunStatus.FAILED,
        EvaluationRunStatus.CANCELLED,
    },
}

FEEDBACK_TRANSITIONS = {
    FeedbackStatus.PENDING: {FeedbackStatus.TRIAGED, FeedbackStatus.REJECTED},
    FeedbackStatus.TRIAGED: {
        FeedbackStatus.IN_REVIEW,
        FeedbackStatus.REJECTED,
    },
    FeedbackStatus.IN_REVIEW: {
        FeedbackStatus.ACCEPTED,
        FeedbackStatus.REJECTED,
    },
}


def ensure_run_transition(current: str, target: str) -> None:
    source = EvaluationRunStatus(current)
    destination = EvaluationRunStatus(target)
    if source == destination:
        return
    if destination not in RUN_TRANSITIONS.get(source, set()):
        raise ValueError(f"invalid evaluation run transition: {current} -> {target}")


def ensure_feedback_transition(current: str, target: str) -> None:
    source = FeedbackStatus(current)
    destination = FeedbackStatus(target)
    if source == destination:
        return
    if destination not in FEEDBACK_TRANSITIONS.get(source, set()):
        raise ValueError(f"invalid feedback transition: {current} -> {target}")


_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_SUITE_MANIFEST_FIELDS = frozenset({"suite_key", "difficulty", "minimum_case_count"})
_SUITE_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


LEGACY_EVALUATION_GATE_POLICY_VERSION = "stt.eval.gate.v1"
EVALUATION_GATE_POLICY_VERSION = "stt.eval.gate.v2"
LEGACY_CASE_PASS_POLICY_VERSION = "stt.eval.case.v1"
PREVIOUS_CASE_PASS_POLICY_VERSION = "stt.eval.case.v2"
CASE_PASS_POLICY_VERSION = "stt.eval.case.v3"
SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION = "stt.eval.case.v4"
SOURCE_ALIGNED_CONCEPT_CONTRACT_VERSION = "stt.eval.authored-concepts.v2"
AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION = "stt.eval.case.v5"
AUTHORED_CONCEPT_CONTRACT_VERSION = "stt.eval.authored-concepts.v3"
AUTHORED_CLAIM_CONTRACT_VERSION = "stt.eval.authored-claims.v1"
TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION = "stt.eval.case.v6"
TEMPORAL_DERIVED_CLAIM_CONTRACT_VERSION = "stt.eval.temporal-derived-claim.v1"

# A live case is primarily judged by whether its final answer is correct.  The
# remaining components cannot manufacture a pass: faithfulness and citation
# coverage are independent hard grounding guards, while retrieval is retained
# as a diagnostic signal because a stale/incomplete relevance label must not
# overrule an answer that is independently supported by the returned evidence.
_V3_CASE_PASS_POLICY: dict[str, Any] = {
    "policy_version": CASE_PASS_POLICY_VERSION,
    "quality_score_minimum": 0.75,
    "answer_correctness_minimum": 0.75,
    "faithfulness_minimum": 0.8,
    "citation_coverage_minimum": 0.8,
    "answer_correctness_weight": 0.7,
    "faithfulness_weight": 0.15,
    "citation_coverage_weight": 0.1,
    "retrieval_support_weight": 0.05,
    # v3 makes the final-answer correctness contract conjunctive.  Coverage
    # alone cannot offset an omitted required branch or an unsupported factual
    # assertion.  These values are materialized only after an immutable
    # manifest explicitly opts into v3; they are never inferred during replay.
    "required_branch_coverage_minimum": 1.0,
    "concept_group_coverage_minimum": 1.0,
    "unsupported_factual_claim_maximum": 0,
    "contradictory_factual_claim_maximum": 0,
    "abstention_factual_claim_maximum": 0,
}

_V4_CASE_PASS_POLICY: dict[str, Any] = {
    **_V3_CASE_PASS_POLICY,
    "policy_version": SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
}

_V5_CASE_PASS_POLICY: dict[str, Any] = {
    **_V3_CASE_PASS_POLICY,
    "policy_version": AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
}

_V6_CASE_PASS_POLICY: dict[str, Any] = {
    **_V5_CASE_PASS_POLICY,
    "policy_version": TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
}

DEFAULT_CASE_PASS_POLICY: dict[str, Any] = {
    key: value
    for key, value in _V3_CASE_PASS_POLICY.items()
    if key
    not in {
        "required_branch_coverage_minimum",
        "concept_group_coverage_minimum",
        "unsupported_factual_claim_maximum",
        "contradictory_factual_claim_maximum",
        "abstention_factual_claim_maximum",
    }
}
DEFAULT_CASE_PASS_POLICY["policy_version"] = PREVIOUS_CASE_PASS_POLICY_VERSION

# These are server defaults, not mutations of the immutable dataset manifest.
# Every newly accepted run freezes the fully materialized policy in its config
# snapshot, so later deployments cannot reinterpret that run.
DEFAULT_EVALUATION_GATE_POLICY: dict[str, Any] = {
    "policy_version": EVALUATION_GATE_POLICY_VERSION,
    "recall_at_1_max_drop": 0.02,
    "recall_at_3_max_drop": 0.02,
    "recall_at_10_max_drop": 0.02,
    "mrr_max_drop": 0.02,
    "ndcg_at_10_max_drop": 0.02,
    "grounding_coverage_max_drop": 0.0,
    "unanswerable_false_answer_max_increase": 0.0,
    "p95_latency_max_increase_ratio": 0.2,
    "critical_failure_blocks": True,
    "minimum_metric_coverage": 0.8,
    "answer_correctness_max_drop": 0.0,
    "case_quality_max_drop": 0.0,
    "faithfulness_max_drop": 0.0,
    "case_pass_policy": DEFAULT_CASE_PASS_POLICY,
}

_UNIT_INTERVAL_GATE_FIELDS = (
    "recall_at_1_max_drop",
    "recall_at_3_max_drop",
    "recall_at_10_max_drop",
    "mrr_max_drop",
    "ndcg_at_10_max_drop",
    "grounding_coverage_max_drop",
    "unanswerable_false_answer_max_increase",
    "minimum_metric_coverage",
    "answer_correctness_max_drop",
    "case_quality_max_drop",
    "faithfulness_max_drop",
)

_CASE_PASS_THRESHOLD_FIELDS = (
    "quality_score_minimum",
    "answer_correctness_minimum",
    "faithfulness_minimum",
    "citation_coverage_minimum",
)
_V3_CASE_PASS_COVERAGE_FIELDS = (
    "required_branch_coverage_minimum",
    "concept_group_coverage_minimum",
)
_V3_CASE_PASS_ZERO_MAXIMUM_FIELDS = (
    "unsupported_factual_claim_maximum",
    "contradictory_factual_claim_maximum",
    "abstention_factual_claim_maximum",
)
_CASE_PASS_WEIGHT_FIELDS = (
    "answer_correctness_weight",
    "faithfulness_weight",
    "citation_coverage_weight",
    "retrieval_support_weight",
)


def _finite_gate_number(
    policy: Mapping[str, Any],
    field: str,
    *,
    default: float,
    maximum: float,
) -> float:
    raw = policy.get(field, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"evaluation gate field {field} must be a finite number")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0 or value > maximum:
        raise ValueError(f"evaluation gate field {field} must be within 0..{maximum:g}")
    return value


def freeze_case_pass_policy(raw_policy: Any) -> dict[str, Any]:
    """Materialize the safe, answer-first live-case decision contract.

    The policy remains configurable for stricter deployments, but manifests
    cannot make unsupported answers pass or make a support metric outweigh the
    final-answer correctness component.
    """

    if raw_policy is None:
        source: dict[str, Any] = {}
    elif isinstance(raw_policy, Mapping):
        source = dict(raw_policy)
    else:
        raise ValueError("evaluation case_pass_policy must be an object")

    requested_version = source.get("policy_version")
    if requested_version not in {
        None,
        PREVIOUS_CASE_PASS_POLICY_VERSION,
        CASE_PASS_POLICY_VERSION,
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    }:
        raise ValueError(
            "new evaluation runs only support case pass policies "
            f"{PREVIOUS_CASE_PASS_POLICY_VERSION}, {CASE_PASS_POLICY_VERSION}, "
            f"{SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION} and "
            f"{AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION}, and "
            f"{TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION}"
        )

    # v3 is a candidate concept contract and must be explicitly selected by the
    # immutable manifest.  An omitted (or unversioned) policy keeps the stable
    # v2 decision contract instead of silently reinterpreting the dataset.
    policy_version = requested_version or PREVIOUS_CASE_PASS_POLICY_VERSION
    defaults = {
        PREVIOUS_CASE_PASS_POLICY_VERSION: DEFAULT_CASE_PASS_POLICY,
        CASE_PASS_POLICY_VERSION: _V3_CASE_PASS_POLICY,
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION: _V4_CASE_PASS_POLICY,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION: _V5_CASE_PASS_POLICY,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION: _V6_CASE_PASS_POLICY,
    }[policy_version]
    frozen = {**defaults, **source}
    frozen["policy_version"] = policy_version
    for field in (*_CASE_PASS_THRESHOLD_FIELDS, *_CASE_PASS_WEIGHT_FIELDS):
        frozen[field] = _finite_gate_number(
            source,
            field,
            default=float(defaults[field]),
            maximum=1.0,
        )

    if policy_version in {
        CASE_PASS_POLICY_VERSION,
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    }:
        for field in _V3_CASE_PASS_COVERAGE_FIELDS:
            frozen[field] = _finite_gate_number(
                source,
                field,
                default=float(_V3_CASE_PASS_POLICY[field]),
                maximum=1.0,
            )
            if float(frozen[field]) < 1.0:
                raise ValueError(
                    f"evaluation case pass field {field} must be 1 for strict policies"
                )
        for field in _V3_CASE_PASS_ZERO_MAXIMUM_FIELDS:
            raw_value = source.get(field, _V3_CASE_PASS_POLICY[field])
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(f"evaluation case pass field {field} must be the integer 0")
            if raw_value != 0:
                raise ValueError(
                    f"evaluation case pass field {field} must be 0 for strict policies"
                )
            frozen[field] = 0

    weights = [float(frozen[field]) for field in _CASE_PASS_WEIGHT_FIELDS]
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("evaluation case pass weights must sum to 1")
    answer_weight = float(frozen["answer_correctness_weight"])
    if answer_weight <= 0.5 or answer_weight <= max(weights[1:]):
        raise ValueError(
            "evaluation case pass answer_correctness_weight must be the unique majority weight"
        )
    # A caller may tighten these guards, never switch them off.  Correctness is
    # deliberately allowed a wider configurable range than the grounding
    # metrics, but still cannot be reduced below a meaningful majority score.
    for field in (
        "answer_correctness_minimum",
        "faithfulness_minimum",
        "citation_coverage_minimum",
    ):
        if float(frozen[field]) < 0.5:
            raise ValueError(f"evaluation case pass field {field} must be at least 0.5")
    return frozen


def freeze_evaluation_gate_policy(raw_policy: Any) -> dict[str, Any]:
    """Validate and materialize the current server-side gate contract.

    Dataset manifests may omit fields for backward compatibility. Every newly
    accepted run nevertheless stores all v2 defaults explicitly. Historical
    snapshots without ``policy_version`` remain v1 and are never rewritten.
    Unknown extension keys are preserved so multimodal or deployment-specific
    policies can coexist with the core paired quality gate.
    """

    if raw_policy is None:
        source: dict[str, Any] = {}
    elif isinstance(raw_policy, Mapping):
        source = dict(raw_policy)
    else:
        raise ValueError("evaluation gate_policy must be an object")

    requested_version = source.get("policy_version")
    if requested_version not in {None, EVALUATION_GATE_POLICY_VERSION}:
        raise ValueError(
            f"new evaluation runs only support gate policy {EVALUATION_GATE_POLICY_VERSION}"
        )

    frozen = {**DEFAULT_EVALUATION_GATE_POLICY, **source}
    frozen["policy_version"] = EVALUATION_GATE_POLICY_VERSION
    for field in _UNIT_INTERVAL_GATE_FIELDS:
        frozen[field] = _finite_gate_number(
            source,
            field,
            default=float(DEFAULT_EVALUATION_GATE_POLICY[field]),
            maximum=1.0,
        )
    frozen["p95_latency_max_increase_ratio"] = _finite_gate_number(
        source,
        "p95_latency_max_increase_ratio",
        default=float(DEFAULT_EVALUATION_GATE_POLICY["p95_latency_max_increase_ratio"]),
        maximum=10.0,
    )
    critical_failure_blocks = source.get(
        "critical_failure_blocks",
        DEFAULT_EVALUATION_GATE_POLICY["critical_failure_blocks"],
    )
    if not isinstance(critical_failure_blocks, bool):
        raise ValueError("evaluation gate field critical_failure_blocks must be a boolean")
    frozen["critical_failure_blocks"] = critical_failure_blocks
    frozen["case_pass_policy"] = freeze_case_pass_policy(source.get("case_pass_policy"))
    return frozen


def validate_dataset_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "dataset_key",
        "name",
        "version",
        "schema_version",
        "source",
        "cases",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"dataset manifest missing fields: {', '.join(missing)}")
    if not _VERSION.fullmatch(str(manifest["version"])):
        raise ValueError("dataset version must be immutable semantic version X.Y.Z")
    # Validate without rewriting the manifest; its checksum continues to
    # describe the exact immutable user-published content.
    frozen_gate_policy = freeze_evaluation_gate_policy(manifest.get("gate_policy"))
    case_policy_version = str(
        (frozen_gate_policy.get("case_pass_policy") or {}).get("policy_version") or ""
    )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain at least one case")
    declared_suite_fields = _SUITE_MANIFEST_FIELDS.intersection(manifest)
    if declared_suite_fields:
        missing_suite_fields = sorted(_SUITE_MANIFEST_FIELDS - manifest.keys())
        if missing_suite_fields:
            raise ValueError(
                "evaluation suite manifest missing fields: " + ", ".join(missing_suite_fields)
            )
        suite_key = manifest["suite_key"]
        if not isinstance(suite_key, str) or not suite_key.strip():
            raise ValueError("evaluation suite_key must be a non-empty string")
        difficulty = manifest["difficulty"]
        if difficulty not in _SUITE_DIFFICULTIES:
            raise ValueError("evaluation difficulty must be one of: easy, medium, hard")
        minimum_case_count = manifest["minimum_case_count"]
        if (
            isinstance(minimum_case_count, bool)
            or not isinstance(minimum_case_count, int)
            or minimum_case_count <= 0
        ):
            raise ValueError("evaluation minimum_case_count must be a positive integer")
        if len(cases) < minimum_case_count:
            raise ValueError(
                f"evaluation suite requires at least {minimum_case_count} cases, got {len(cases)}"
            )
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if any(not case_id for case_id in case_ids):
        raise ValueError("every evaluation case must have case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case_id must be unique within a dataset version")
    for case in cases:
        if not str(case.get("query", "")).strip():
            raise ValueError(f"case {case.get('case_id')} has empty query")
        if not case.get("unanswerable") and not str(case.get("reference_answer", "")).strip():
            raise ValueError(f"answerable case {case.get('case_id')} needs reference_answer")
        metadata = case.get("metadata")
        has_security_contract = isinstance(metadata, Mapping) and (
            "security_expectation" in metadata
        )
        if (
            case_policy_version
            in {
                SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
            and not case.get("unanswerable")
            and not has_security_contract
            and "concept_groups" not in case
        ):
            raise ValueError(
                f"case {case.get('case_id')} source-aligned answerable case needs concept_groups"
            )
        if "concept_groups" in case:
            _validate_concept_groups(
                case.get("concept_groups"),
                case_id=case.get("case_id"),
                relevant_evidence=case.get("relevant_evidence"),
                required_contract_version=(
                    AUTHORED_CONCEPT_CONTRACT_VERSION
                    if case_policy_version
                    in {
                        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
                    }
                    else SOURCE_ALIGNED_CONCEPT_CONTRACT_VERSION
                    if case_policy_version == SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION
                    else None
                ),
            )
        if case_policy_version in {
            AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
            TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
        }:
            _validate_authored_claims(
                case.get("authored_claims"),
                case_id=case.get("case_id"),
                concept_groups=case.get("concept_groups"),
                relevant_evidence=case.get("relevant_evidence"),
            )
        if isinstance(metadata, Mapping) and "security_expectation" in metadata:
            _validate_security_expectation(
                metadata.get("security_expectation"),
                case_id=case.get("case_id"),
            )


def _validate_security_expectation(value: Any, *, case_id: Any) -> None:
    """校验显式安全评分合同，避免从标签或输入正文反推阻断语义。"""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"case {case_id} security_expectation must be a non-empty object")
    if value.get("contract_version") != "stt.eval.security.v1":
        raise ValueError(f"case {case_id} security_expectation contract_version is unsupported")
    # v1 只声明当前评测适配器能够稳定投影的输入阻断；其他阶段必须在
    # canonical observation 支持落地后发布新合同版本，不能提前接受却无法执行。
    if value.get("stage") != "input":
        raise ValueError(f"case {case_id} security_expectation stage is unsupported")
    if value.get("action") != "block":
        raise ValueError(f"case {case_id} security_expectation action is unsupported")
    reason_codes = value.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        raise ValueError(f"case {case_id} security_expectation reason_codes must be non-empty")
    if any(
        not isinstance(code, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", code) is None
        for code in reason_codes
    ):
        raise ValueError(f"case {case_id} security_expectation reason_codes are invalid")
    for field in ("forbidden_canaries", "forbidden_output_terms"):
        terms = value.get(field)
        if terms is None:
            continue
        if (
            not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError(
                f"case {case_id} security_expectation {field} must contain non-empty strings"
            )


def _validate_concept_groups(
    value: Any,
    *,
    case_id: Any,
    relevant_evidence: Any = None,
    required_contract_version: str | None = None,
) -> None:
    """Validate the optional all-of / per-group any-of scoring contract.

    The schema deliberately models concepts rather than weighted strings:
    every item in ``all_of`` is required, while spelling or terminology
    variants live inside that item's ``any_of`` list.  It is optional so old
    immutable manifests remain valid and byte-for-byte unchanged.
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"case {case_id} concept_groups must be a non-empty object")
    require_source_alignment = required_contract_version is not None
    if required_contract_version is not None and (
        value.get("contract_version") != required_contract_version
    ):
        raise ValueError(
            f"case {case_id} concept_groups contract_version must be "
            f"{required_contract_version} for source-aligned scoring"
        )
    all_of = value.get("all_of")
    if not isinstance(all_of, list) or not all_of:
        raise ValueError(f"case {case_id} concept_groups.all_of must be a non-empty list")

    concept_keys: set[str] = set()
    group_signatures: set[tuple[str, ...]] = set()
    evidence_by_id: dict[str, str] = {}
    if isinstance(relevant_evidence, list):
        for raw_evidence in relevant_evidence:
            if not isinstance(raw_evidence, Mapping):
                continue
            evidence_id = str(raw_evidence.get("evidence_id") or "").strip()
            evidence_text = str(raw_evidence.get("text") or "").strip()
            if evidence_id and evidence_text:
                evidence_by_id[evidence_id] = evidence_text

    required_group_count = 0
    ellipsis_requirements: dict[str, tuple[str, ...]] = {}
    for index, raw_group in enumerate(all_of):
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"case {case_id} concept_groups.all_of[{index}] must be an object")
        any_of = raw_group.get("any_of")
        if not isinstance(any_of, list) or not any_of:
            raise ValueError(
                f"case {case_id} concept_groups.all_of[{index}].any_of must be a non-empty list"
            )
        normalized_variants: list[str] = []
        for variant in any_of:
            if not isinstance(variant, str) or not variant.strip():
                raise ValueError(
                    f"case {case_id} concept_groups.all_of[{index}].any_of "
                    "must contain non-empty strings"
                )
            normalized_variants.append(_normalized_policy_text(variant))
        if len(normalized_variants) != len(set(normalized_variants)):
            raise ValueError(
                f"case {case_id} concept_groups.all_of[{index}].any_of "
                "must not contain duplicate variants"
            )
        signature = tuple(sorted(normalized_variants))
        if signature in group_signatures:
            raise ValueError(f"case {case_id} concept_groups.all_of contains duplicate groups")
        group_signatures.add(signature)

        concept_key = raw_group.get("concept_key")
        if concept_key is not None:
            if not isinstance(concept_key, str) or not concept_key.strip():
                raise ValueError(f"case {case_id} concept_groups concept_key must be non-empty")
            normalized_key = _normalized_policy_text(concept_key)
            if normalized_key in concept_keys:
                raise ValueError(f"case {case_id} concept_groups concept_key must be unique")
            concept_keys.add(normalized_key)

        if required_contract_version == AUTHORED_CONCEPT_CONTRACT_VERSION:
            satisfaction = raw_group.get("satisfaction")
            if satisfaction is None:
                satisfaction = {"mode": "answer_only"}
            if not isinstance(satisfaction, Mapping):
                raise ValueError(f"case {case_id} concept satisfaction must be an object")
            satisfaction_mode = satisfaction.get("mode")
            if satisfaction_mode not in {"answer_only", "query_predicate_ellipsis"}:
                raise ValueError(f"case {case_id} concept satisfaction mode is invalid")
            if satisfaction_mode == "query_predicate_ellipsis":
                if not isinstance(concept_key, str) or not concept_key.strip():
                    raise ValueError(f"case {case_id} query predicate concept needs concept_key")
                query_any_of = satisfaction.get("query_any_of")
                if (
                    not isinstance(query_any_of, list)
                    or not query_any_of
                    or any(not isinstance(item, str) or not item.strip() for item in query_any_of)
                ):
                    raise ValueError(f"case {case_id} query predicate concept needs query_any_of")
                normalized_queries = [_normalized_policy_text(item) for item in query_any_of]
                if len(normalized_queries) != len(set(normalized_queries)):
                    raise ValueError(f"case {case_id} query predicate variants must be unique")
                required_keys = satisfaction.get("requires_concept_keys")
                if (
                    not isinstance(required_keys, list)
                    or not required_keys
                    or any(not isinstance(item, str) or not item.strip() for item in required_keys)
                ):
                    raise ValueError(f"case {case_id} query predicate needs requires_concept_keys")
                normalized_required = tuple(_normalized_policy_text(item) for item in required_keys)
                if len(normalized_required) != len(set(normalized_required)):
                    raise ValueError(f"case {case_id} query predicate requirements must be unique")
                ellipsis_requirements[_normalized_policy_text(concept_key)] = normalized_required

        alignment = raw_group.get("source_alignment")
        if alignment is None and not require_source_alignment:
            continue
        if not isinstance(alignment, Mapping):
            raise ValueError(f"case {case_id} concept_groups source_alignment must be an object")
        mode = alignment.get("mode")
        if mode not in {"required", "diagnostic"}:
            raise ValueError(f"case {case_id} concept_groups source_alignment mode is invalid")
        if mode == "diagnostic":
            reason_code = alignment.get("reason_code")
            if (
                not isinstance(reason_code, str)
                or re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", reason_code) is None
            ):
                raise ValueError(f"case {case_id} diagnostic concept group needs reason_code")
            continue

        required_group_count += 1
        source_terms = alignment.get("source_terms")
        if not isinstance(source_terms, list) or not source_terms:
            raise ValueError(f"case {case_id} required concept group needs atomic source_terms")
        normalized_source_terms: list[str] = []
        for source_term in source_terms:
            if not isinstance(source_term, str) or not source_term.strip():
                raise ValueError(f"case {case_id} source_terms must contain non-empty strings")
            normalized_term = _source_binding_text(source_term)
            # 单字符状态符号是表格中的原子语义；其余绑定至少需要两个可见
            # 字符，避免用“可”“不”等泛化片段制造错误自证。
            if len(normalized_term) < 2 and normalized_term not in {"○", "×"}:
                raise ValueError(f"case {case_id} source term is too weak to bind a concept")
            normalized_source_terms.append(normalized_term)
        if len(normalized_source_terms) != len(set(normalized_source_terms)):
            raise ValueError(f"case {case_id} source_terms must not contain duplicates")

        anchors = alignment.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"case {case_id} required concept group needs source anchors")
        anchor_signatures: set[tuple[str, str]] = set()
        anchor_binding_texts: list[str] = []
        for anchor_index, anchor in enumerate(anchors):
            if not isinstance(anchor, Mapping):
                raise ValueError(f"case {case_id} source anchor {anchor_index} must be an object")
            evidence_id = anchor.get("evidence_id")
            anchor_text = anchor.get("text")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError(f"case {case_id} source anchor evidence_id must be non-empty")
            if not isinstance(anchor_text, str) or not anchor_text.strip():
                raise ValueError(f"case {case_id} source anchor text must be non-empty")
            evidence_text = evidence_by_id.get(evidence_id.strip())
            if evidence_text is None:
                raise ValueError(f"case {case_id} source anchor references unknown evidence_id")
            normalized_anchor = _compact_policy_text(anchor_text)
            if not normalized_anchor or normalized_anchor not in _compact_policy_text(
                evidence_text
            ):
                raise ValueError(f"case {case_id} source anchor must occur in referenced evidence")
            signature = (evidence_id.strip(), normalized_anchor)
            if signature in anchor_signatures:
                raise ValueError(f"case {case_id} source anchors must not contain duplicates")
            anchor_signatures.add(signature)
            anchor_binding_texts.append(_source_binding_text(anchor_text))

        semantic_variants = [_source_binding_text(variant) for variant in any_of]
        for source_term, normalized_term in zip(source_terms, normalized_source_terms, strict=True):
            if not any(normalized_term in variant for variant in semantic_variants):
                raise ValueError(
                    f"case {case_id} source term must occur in concept any_of semantics"
                )
            if not any(normalized_term in anchor_text for anchor_text in anchor_binding_texts):
                raise ValueError(
                    f"case {case_id} source term must occur in a referenced qrel anchor"
                )

    if require_source_alignment and required_group_count == 0:
        raise ValueError(
            f"case {case_id} source-aligned concept_groups need at least one required group"
        )
    for predicate_key, required_keys in ellipsis_requirements.items():
        if predicate_key in required_keys:
            raise ValueError(f"case {case_id} query predicate cannot require itself")
        if not set(required_keys) <= concept_keys:
            raise ValueError(f"case {case_id} query predicate references unknown concept_key")


def _validate_authored_claims(
    value: Any,
    *,
    case_id: Any,
    concept_groups: Any,
    relevant_evidence: Any,
) -> None:
    """校验 v5 表格原子主张合同；查询只提供省略谓词，不提供事实支撑。"""

    raw_groups = concept_groups.get("all_of") if isinstance(concept_groups, Mapping) else None
    groups = raw_groups if isinstance(raw_groups, list) else []
    concept_source_terms: dict[str, set[str]] = {}
    ellipsis_requirements: dict[str, set[str]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        raw_key = group.get("concept_key")
        if not isinstance(raw_key, str) or not raw_key.strip():
            continue
        key = _normalized_policy_text(raw_key)
        alignment = group.get("source_alignment")
        raw_terms = alignment.get("source_terms") if isinstance(alignment, Mapping) else None
        if isinstance(raw_terms, list):
            concept_source_terms[key] = {
                _source_binding_text(term)
                for term in raw_terms
                if isinstance(term, str) and term.strip()
            }
        satisfaction = group.get("satisfaction")
        if not isinstance(satisfaction, Mapping):
            continue
        if satisfaction.get("mode") != "query_predicate_ellipsis":
            continue
        raw_required = satisfaction.get("requires_concept_keys")
        if isinstance(raw_required, list):
            ellipsis_requirements[key] = {
                _normalized_policy_text(item)
                for item in raw_required
                if isinstance(item, str) and item.strip()
            }

    if value is None:
        if ellipsis_requirements:
            raise ValueError(f"case {case_id} query predicate ellipsis needs authored_claims")
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"case {case_id} authored_claims must be a non-empty object")
    if value.get("contract_version") != AUTHORED_CLAIM_CONTRACT_VERSION:
        raise ValueError(
            f"case {case_id} authored_claims contract_version must be "
            f"{AUTHORED_CLAIM_CONTRACT_VERSION}"
        )
    atoms = value.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError(f"case {case_id} authored_claims.atoms must be non-empty")

    evidence_by_id: dict[str, str] = {}
    if isinstance(relevant_evidence, list):
        for evidence in relevant_evidence:
            if not isinstance(evidence, Mapping):
                continue
            evidence_id = str(evidence.get("evidence_id") or "").strip()
            evidence_text = str(evidence.get("text") or "").strip()
            if evidence_id and evidence_text:
                evidence_by_id[evidence_id] = evidence_text

    claim_keys: set[str] = set()
    atom_requirements: list[set[str]] = []
    known_concept_keys = set(concept_source_terms)
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            raise ValueError(f"case {case_id} authored claim {index} must be an object")
        claim_key = atom.get("claim_key")
        if not isinstance(claim_key, str) or not claim_key.strip():
            raise ValueError(f"case {case_id} authored claim needs claim_key")
        normalized_claim_key = _normalized_policy_text(claim_key)
        if normalized_claim_key in claim_keys:
            raise ValueError(f"case {case_id} authored claim_key must be unique")
        claim_keys.add(normalized_claim_key)

        raw_required = atom.get("requires_concept_keys")
        if (
            not isinstance(raw_required, list)
            or len(raw_required) < 2
            or any(not isinstance(item, str) or not item.strip() for item in raw_required)
        ):
            raise ValueError(f"case {case_id} authored claim needs at least two concept keys")
        required = {_normalized_policy_text(item) for item in raw_required}
        if len(required) != len(raw_required):
            raise ValueError(f"case {case_id} authored claim concept keys must be unique")
        if not required <= known_concept_keys:
            raise ValueError(f"case {case_id} authored claim references unknown concept_key")
        atom_requirements.append(required)

        if not isinstance(atom.get("citation_required"), bool):
            raise ValueError(f"case {case_id} authored claim citation_required must be boolean")
        alignment = atom.get("source_alignment")
        if not isinstance(alignment, Mapping):
            raise ValueError(f"case {case_id} authored claim needs source_alignment")
        raw_terms = alignment.get("source_terms")
        if (
            not isinstance(raw_terms, list)
            or not raw_terms
            or any(not isinstance(term, str) or not term.strip() for term in raw_terms)
        ):
            raise ValueError(f"case {case_id} authored claim needs source_terms")
        normalized_terms = [_source_binding_text(term) for term in raw_terms]
        if len(normalized_terms) != len(set(normalized_terms)):
            raise ValueError(f"case {case_id} authored claim source_terms must be unique")
        for required_key in required:
            if not (concept_source_terms.get(required_key, set()) & set(normalized_terms)):
                raise ValueError(f"case {case_id} authored claim must bind every required concept")

        anchors = alignment.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"case {case_id} authored claim needs source anchors")
        has_joint_anchor = False
        for anchor_index, anchor in enumerate(anchors):
            if not isinstance(anchor, Mapping):
                raise ValueError(
                    f"case {case_id} authored claim anchor {anchor_index} must be an object"
                )
            evidence_id = anchor.get("evidence_id")
            anchor_text = anchor.get("text")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError(
                    f"case {case_id} authored claim anchor evidence_id must be non-empty"
                )
            if not isinstance(anchor_text, str) or not anchor_text.strip():
                raise ValueError(f"case {case_id} authored claim anchor text must be non-empty")
            evidence_text = evidence_by_id.get(evidence_id.strip())
            normalized_anchor = _source_binding_text(anchor_text)
            if evidence_text is None or normalized_anchor not in _source_binding_text(
                evidence_text
            ):
                raise ValueError(
                    f"case {case_id} authored claim anchor must occur in referenced evidence"
                )
            if all(term in normalized_anchor for term in normalized_terms):
                has_joint_anchor = True
        if not has_joint_anchor:
            raise ValueError(f"case {case_id} authored claim source terms need one joint anchor")

    for predicate_key, subject_keys in ellipsis_requirements.items():
        for subject_key in subject_keys:
            if not any({predicate_key, subject_key} <= required for required in atom_requirements):
                raise ValueError(
                    f"case {case_id} query predicate needs a source-bound claim atom "
                    f"for every required concept"
                )


def _normalized_policy_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _compact_policy_text(value: str) -> str:
    """仅保留可见字母数字与中日韩字符，用于验证锚点来源。"""

    normalized = value.casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )


def _source_binding_text(value: str) -> str:
    """保留表格状态符号的紧凑文本，用于原子来源绑定。"""

    return "".join(character for character in value.casefold() if not character.isspace())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


LEGACY_FORMAL_EVALUATION_CONTRACT_VERSION = "stt.eval.formal-run.v1"
FORMAL_EVALUATION_CONTRACT_VERSION = "stt.eval.formal-run.v2"
FORMAL_EVALUATION_INTENT_VERSION = "stt.eval.formal-intent.v1"
FORMAL_EVALUATION_SNAPSHOT_KEY = "formal_evaluation"
_FORMAL_CONTRACT_FIELDS = frozenset(
    {
        "contract_version",
        "contract_sha256",
        "config_schema_version",
        "config_id",
        "config_sha256",
        "target_id",
        "dataset",
        "scenario",
        "graders",
        "failure_policy",
        "formal_evaluation",
        "execution_mode",
        "budgets",
    }
)
_FORMAL_DATASET_FIELDS = frozenset(
    {
        "registry_id",
        "asset_sha256",
        "record_id",
        "record_checksum",
        "dataset_key",
        "dataset_version",
    }
)
_FORMAL_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "scenario_sha256",
        "observation_scope",
        "allowed_side_effects",
        "external_model_allowed",
        "formal_evaluation",
    }
)
_FORMAL_BUDGET_FIELDS = frozenset(
    {
        "scope",
        "token_budget",
        "tool_budget",
        "cost_budget",
        "cost_currency",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SHA256_DIGEST = re.compile(r"^(?:sha256:)?(?P<digest>[0-9a-f]{64})$")
_EXPECTATION_UNSET = object()


def _formal_exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"formal evaluation {name} fields are invalid: " + "; ".join(details))


def _formal_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"formal evaluation {field} must be a non-empty canonical string")
    return value


def _formal_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"formal evaluation {field} must be a lowercase sha256")
    return value


def _formal_canonical_sha256_digest(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"formal evaluation {field} must be a lowercase sha256 digest")
    matched = _CANONICAL_SHA256_DIGEST.fullmatch(value)
    if matched is None:
        raise ValueError(f"formal evaluation {field} must be a lowercase sha256 digest")
    return matched.group("digest")


def _freeze_formal_budgets(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("formal evaluation budgets must be an object")
    _formal_exact_fields(value, _FORMAL_BUDGET_FIELDS, "budget")
    if value.get("scope") != "per_case_query_plan":
        raise ValueError("formal evaluation budget scope must be per_case_query_plan")
    token_budget = value.get("token_budget")
    tool_budget = value.get("tool_budget")
    cost_budget = value.get("cost_budget")
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or not 128 <= token_budget <= 131_072
    ):
        raise ValueError("formal evaluation token_budget must be within 128..131072")
    if (
        isinstance(tool_budget, bool)
        or not isinstance(tool_budget, int)
        or not 1 <= tool_budget <= 8
    ):
        raise ValueError("formal evaluation tool_budget must be within 1..8")
    if isinstance(cost_budget, bool) or not isinstance(cost_budget, (int, float)):
        raise ValueError("formal evaluation cost_budget must be a finite number")
    normalized_cost = float(cost_budget)
    if not math.isfinite(normalized_cost) or not 0.05 <= normalized_cost <= 1_000.0:
        raise ValueError("formal evaluation cost_budget must be within 0.05..1000")
    if value.get("cost_currency") != "USD":
        raise ValueError("formal evaluation cost_currency must be USD")
    return {
        "scope": "per_case_query_plan",
        "token_budget": token_budget,
        "tool_budget": tool_budget,
        "cost_budget": normalized_cost,
        "cost_currency": "USD",
    }


def _freeze_legacy_formal_evaluation_contract(
    raw_contract: Any,
    *,
    expected_mode: str | None = None,
    expected_dataset_record_id: str | None = None,
    expected_dataset_checksum: str | None = None,
    expected_dataset_key: str | None = None,
    expected_dataset_version: str | None = None,
) -> dict[str, Any]:
    """校验并规范化耐久正式评测权威，任何漂移均失败关闭。"""

    if not isinstance(raw_contract, Mapping):
        raise ValueError("formal evaluation contract must be an object")
    _formal_exact_fields(raw_contract, _FORMAL_CONTRACT_FIELDS, "contract")
    if raw_contract.get("contract_version") != LEGACY_FORMAL_EVALUATION_CONTRACT_VERSION:
        raise ValueError("unsupported formal evaluation contract version")
    if raw_contract.get("formal_evaluation") is not True:
        raise ValueError("formal evaluation contract must remain formal")

    raw_dataset = raw_contract.get("dataset")
    if not isinstance(raw_dataset, Mapping):
        raise ValueError("formal evaluation dataset snapshot must be an object")
    _formal_exact_fields(raw_dataset, _FORMAL_DATASET_FIELDS, "dataset")
    dataset = {
        "registry_id": _formal_identifier(raw_dataset.get("registry_id"), "dataset.registry_id"),
        "asset_sha256": _formal_sha256(raw_dataset.get("asset_sha256"), "dataset.asset_sha256"),
        "record_id": _formal_identifier(raw_dataset.get("record_id"), "dataset.record_id"),
        "record_checksum": _formal_sha256(
            raw_dataset.get("record_checksum"), "dataset.record_checksum"
        ),
        "dataset_key": _formal_identifier(raw_dataset.get("dataset_key"), "dataset.dataset_key"),
        "dataset_version": _formal_identifier(
            raw_dataset.get("dataset_version"), "dataset.dataset_version"
        ),
    }

    raw_scenario = raw_contract.get("scenario")
    if not isinstance(raw_scenario, Mapping):
        raise ValueError("formal evaluation scenario snapshot must be an object")
    _formal_exact_fields(raw_scenario, _FORMAL_SCENARIO_FIELDS, "scenario")
    side_effects = raw_scenario.get("allowed_side_effects")
    if (
        not isinstance(side_effects, list)
        or not all(isinstance(item, str) and item for item in side_effects)
        or len(side_effects) != len(set(side_effects))
    ):
        raise ValueError("formal evaluation scenario side effects must be unique strings")
    if not isinstance(raw_scenario.get("external_model_allowed"), bool):
        raise ValueError("formal evaluation scenario external_model_allowed must be a boolean")
    if raw_scenario.get("formal_evaluation") is not True:
        raise ValueError("formal evaluation scenario must be approved for formal execution")
    scenario = {
        "scenario_id": _formal_identifier(raw_scenario.get("scenario_id"), "scenario.scenario_id"),
        "scenario_sha256": _formal_sha256(
            raw_scenario.get("scenario_sha256"), "scenario.scenario_sha256"
        ),
        "observation_scope": raw_scenario.get("observation_scope"),
        "allowed_side_effects": list(side_effects),
        "external_model_allowed": raw_scenario.get("external_model_allowed"),
        "formal_evaluation": True,
    }
    scenario_content = {key: value for key, value in scenario.items() if key != "scenario_sha256"}
    if scenario["scenario_sha256"] != content_digest(scenario_content):
        raise ValueError("formal evaluation scenario sha256 does not match its content")
    execution_mode = raw_contract.get("execution_mode")
    expected_scope_by_mode = {"live": "full", "retrieval_only": "retrieval_only"}
    if execution_mode not in expected_scope_by_mode:
        raise ValueError("formal evaluation execution_mode is unsupported")
    if scenario["observation_scope"] != expected_scope_by_mode[execution_mode]:
        raise ValueError("formal evaluation scenario scope does not match execution mode")
    expected_effects = (
        {"vector_read", "external_llm"} if execution_mode == "live" else {"vector_read"}
    )
    if set(scenario["allowed_side_effects"]) != expected_effects:
        raise ValueError("formal evaluation scenario side effects do not match execution mode")
    if scenario["external_model_allowed"] is not (execution_mode == "live"):
        raise ValueError("formal evaluation model permission does not match execution mode")

    graders = raw_contract.get("graders")
    if (
        not isinstance(graders, list)
        or not graders
        or not all(isinstance(item, str) and item for item in graders)
        or len(graders) != len(set(graders))
    ):
        raise ValueError("formal evaluation graders must be unique non-empty strings")
    failure_policy = raw_contract.get("failure_policy")
    if failure_policy not in {"fail_fast", "continue"}:
        raise ValueError("formal evaluation failure policy is unsupported")
    config_schema_version = raw_contract.get("config_schema_version")
    if config_schema_version != "stt.eval.harness-config.v2":
        raise ValueError("formal evaluation config schema must be v2")
    config_id = _formal_identifier(raw_contract.get("config_id"), "config_id")
    config_sha256 = _formal_sha256(raw_contract.get("config_sha256"), "config_sha256")
    target_id = _formal_identifier(raw_contract.get("target_id"), "target_id")
    budgets = _freeze_formal_budgets(raw_contract.get("budgets"))
    normalized_config = {
        "schema_version": config_schema_version,
        "config_id": config_id,
        "target_id": target_id,
        "dataset_id": dataset["registry_id"],
        "scenario_id": scenario["scenario_id"],
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "budgets": budgets,
    }
    if config_sha256 != content_digest(normalized_config):
        raise ValueError("formal evaluation config sha256 does not match its content")
    normalized = {
        "contract_version": LEGACY_FORMAL_EVALUATION_CONTRACT_VERSION,
        "config_schema_version": config_schema_version,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "target_id": target_id,
        "dataset": dataset,
        "scenario": scenario,
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "execution_mode": execution_mode,
        "budgets": budgets,
    }
    declared_digest = _formal_sha256(raw_contract.get("contract_sha256"), "contract_sha256")
    actual_digest = content_digest(normalized)
    if declared_digest != actual_digest:
        raise ValueError("formal evaluation contract sha256 does not match its content")
    normalized["contract_sha256"] = actual_digest

    expectations = {
        "execution mode": (expected_mode, execution_mode),
        "dataset record id": (expected_dataset_record_id, dataset["record_id"]),
        "dataset checksum": (expected_dataset_checksum, dataset["record_checksum"]),
        "dataset key": (expected_dataset_key, dataset["dataset_key"]),
        "dataset version": (expected_dataset_version, dataset["dataset_version"]),
    }
    for label, (expected, actual) in expectations.items():
        if expected is not None and expected != actual:
            raise ValueError(f"formal evaluation {label} does not match the durable run")
    return json.loads(canonical_json(normalized))


def build_legacy_formal_evaluation_contract(
    *,
    config_schema_version: str,
    config_id: str,
    config_sha256: str,
    target_id: str,
    dataset: Mapping[str, Any],
    scenario: Mapping[str, Any],
    graders: list[str] | tuple[str, ...],
    failure_policy: str,
    execution_mode: str,
    budgets: Mapping[str, Any],
) -> dict[str, Any]:
    """构造带内容哈希的正式评测权威快照。"""

    contract: dict[str, Any] = {
        "contract_version": LEGACY_FORMAL_EVALUATION_CONTRACT_VERSION,
        "config_schema_version": config_schema_version,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "target_id": target_id,
        "dataset": dict(dataset),
        "scenario": dict(scenario),
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "execution_mode": execution_mode,
        "budgets": dict(budgets),
    }
    contract["contract_sha256"] = content_digest(contract)
    return _freeze_legacy_formal_evaluation_contract(contract)


_FORMAL_INTENT_FIELDS = frozenset(
    {
        "intent_version",
        "intent_sha256",
        "config_schema_version",
        "config_id",
        "config_sha256",
        "target_id",
        "dataset",
        "scenario",
        "graders",
        "failure_policy",
        "formal_evaluation",
        "execution_mode",
        "planning_admission_budgets",
        "usage_accounting",
        "allowed_intervention_dimensions",
    }
)
_FORMAL_PLANNING_ADMISSION_FIELDS = frozenset(
    {
        "scope",
        "max_plan_tokens",
        "max_routes",
        "max_planned_cost_units",
    }
)
_FORMAL_USAGE_ACCOUNTING_FIELDS = frozenset(
    {
        "provider_tokens",
        "provider_cost_usd",
        "retrieval_calls",
        "enforcement",
    }
)
_FORMAL_V2_CONTRACT_FIELDS = frozenset(
    {"contract_version", "contract_sha256", "intent", "run_authority"}
)
_FORMAL_RUN_AUTHORITY_FIELDS = frozenset(
    {
        "run_id",
        "tenant_id",
        "kb_id",
        "baseline",
        "selected_cases",
        "retrieval_profile",
        "vector_read_set",
        "evaluation_security",
        "evaluation_security_sha256",
        "code_revision",
        "config_version",
        "model_ids",
        "runtime_flags",
        "runtime_flags_sha256",
        "enforced_execution_controls",
        "preflight_sha256",
        "runtime_snapshot_sha256",
        "comparison_snapshot_sha256",
    }
)
_FORMAL_BASELINE_FIELDS = frozenset({"baseline_run_id", "baseline_formal_contract_sha256"})
_FORMAL_SELECTED_CASE_FIELDS = frozenset(
    {"selected_case_ids", "selected_case_count", "selected_case_ids_sha256"}
)
_FORMAL_RETRIEVAL_PROFILE_FIELDS = frozenset({"profile_id", "version", "config_hash"})
_FORMAL_VECTOR_READ_SET_FIELDS = frozenset({"digest", "scopes", "scopes_sha256"})
_FORMAL_MODEL_ID_FIELDS = frozenset({"embedding", "reranker", "llm"})
_FORMAL_EXECUTION_CONTROL_FIELDS = frozenset(
    {
        "query_plan_route_timeout_ms",
        "generation_output_token_cap",
        "provider_request_timeout_seconds",
        "provider_max_attempts_per_request",
        "provider_retry_base_delay_seconds",
        "provider_retry_max_delay_seconds",
    }
)
_FORMAL_INTERVENTION_DIMENSIONS = frozenset(
    {
        "code_revision",
        "config_version",
        "retrieval_profile",
        "vector_read_set",
        "evaluation_security",
        "model_ids",
        "runtime_flags",
    }
)


def _formal_json_object(value: Any, field: str, *, require_nonempty: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"formal evaluation {field} must be an object")
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"formal evaluation {field} must contain canonical JSON") from exc
    if require_nonempty and not normalized:
        raise ValueError(f"formal evaluation {field} cannot be empty")
    return normalized


def _freeze_planning_admission_budgets(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("formal evaluation planning_admission_budgets must be an object")
    _formal_exact_fields(
        value,
        _FORMAL_PLANNING_ADMISSION_FIELDS,
        "planning admission budget",
    )
    if value.get("scope") != "per_case_query_plan_admission":
        raise ValueError(
            "formal evaluation planning budget scope must be per_case_query_plan_admission"
        )
    max_plan_tokens = value.get("max_plan_tokens")
    max_routes = value.get("max_routes")
    max_planned_cost_units = value.get("max_planned_cost_units")
    if (
        isinstance(max_plan_tokens, bool)
        or not isinstance(max_plan_tokens, int)
        or not 128 <= max_plan_tokens <= 131_072
    ):
        raise ValueError("formal evaluation max_plan_tokens must be within 128..131072")
    if isinstance(max_routes, bool) or not isinstance(max_routes, int) or not 1 <= max_routes <= 8:
        raise ValueError("formal evaluation max_routes must be within 1..8")
    if isinstance(max_planned_cost_units, bool) or not isinstance(
        max_planned_cost_units, (int, float)
    ):
        raise ValueError("formal evaluation max_planned_cost_units must be finite")
    normalized_cost = float(max_planned_cost_units)
    if not math.isfinite(normalized_cost) or not 0.05 <= normalized_cost <= 1_000.0:
        raise ValueError("formal evaluation max_planned_cost_units must be within 0.05..1000")
    return {
        "scope": "per_case_query_plan_admission",
        "max_plan_tokens": max_plan_tokens,
        "max_routes": max_routes,
        "max_planned_cost_units": normalized_cost,
    }


def _freeze_usage_accounting(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("formal evaluation usage_accounting must be an object")
    _formal_exact_fields(value, _FORMAL_USAGE_ACCOUNTING_FIELDS, "usage accounting")
    normalized = {
        "provider_tokens": value.get("provider_tokens"),
        "provider_cost_usd": value.get("provider_cost_usd"),
        "retrieval_calls": value.get("retrieval_calls"),
        "enforcement": value.get("enforcement"),
    }
    if normalized != {
        "provider_tokens": "unavailable",
        "provider_cost_usd": "unavailable",
        "retrieval_calls": "unavailable",
        "enforcement": "not_enforced",
    }:
        raise ValueError(
            "formal evaluation actual usage must be explicitly unavailable/not_enforced"
        )
    return normalized


def freeze_formal_evaluation_intent(raw_intent: Any) -> dict[str, Any]:
    """校验 CLI 可声明的意图；运行事实只能由 repository 完成。"""

    if not isinstance(raw_intent, Mapping):
        raise ValueError("formal evaluation intent must be an object")
    _formal_exact_fields(raw_intent, _FORMAL_INTENT_FIELDS, "intent")
    if raw_intent.get("intent_version") != FORMAL_EVALUATION_INTENT_VERSION:
        raise ValueError("unsupported formal evaluation intent version")
    if raw_intent.get("formal_evaluation") is not True:
        raise ValueError("formal evaluation intent must remain formal")

    raw_dataset = raw_intent.get("dataset")
    if not isinstance(raw_dataset, Mapping):
        raise ValueError("formal evaluation dataset snapshot must be an object")
    _formal_exact_fields(raw_dataset, _FORMAL_DATASET_FIELDS, "dataset")
    dataset = {
        "registry_id": _formal_identifier(raw_dataset.get("registry_id"), "dataset.registry_id"),
        "asset_sha256": _formal_sha256(raw_dataset.get("asset_sha256"), "dataset.asset_sha256"),
        "record_id": _formal_identifier(raw_dataset.get("record_id"), "dataset.record_id"),
        "record_checksum": _formal_sha256(
            raw_dataset.get("record_checksum"), "dataset.record_checksum"
        ),
        "dataset_key": _formal_identifier(raw_dataset.get("dataset_key"), "dataset.dataset_key"),
        "dataset_version": _formal_identifier(
            raw_dataset.get("dataset_version"), "dataset.dataset_version"
        ),
    }

    raw_scenario = raw_intent.get("scenario")
    if not isinstance(raw_scenario, Mapping):
        raise ValueError("formal evaluation scenario snapshot must be an object")
    _formal_exact_fields(raw_scenario, _FORMAL_SCENARIO_FIELDS, "scenario")
    side_effects = raw_scenario.get("allowed_side_effects")
    if (
        not isinstance(side_effects, list)
        or not all(isinstance(item, str) and item for item in side_effects)
        or len(side_effects) != len(set(side_effects))
    ):
        raise ValueError("formal evaluation scenario side effects must be unique strings")
    if not isinstance(raw_scenario.get("external_model_allowed"), bool):
        raise ValueError("formal evaluation scenario external_model_allowed must be a boolean")
    if raw_scenario.get("formal_evaluation") is not True:
        raise ValueError("formal evaluation scenario must be approved for formal execution")
    scenario = {
        "scenario_id": _formal_identifier(raw_scenario.get("scenario_id"), "scenario.scenario_id"),
        "scenario_sha256": _formal_sha256(
            raw_scenario.get("scenario_sha256"), "scenario.scenario_sha256"
        ),
        "observation_scope": raw_scenario.get("observation_scope"),
        "allowed_side_effects": list(side_effects),
        "external_model_allowed": raw_scenario.get("external_model_allowed"),
        "formal_evaluation": True,
    }
    scenario_content = {key: value for key, value in scenario.items() if key != "scenario_sha256"}
    if scenario["scenario_sha256"] != content_digest(scenario_content):
        raise ValueError("formal evaluation scenario sha256 does not match its content")

    execution_mode = raw_intent.get("execution_mode")
    expected_scope_by_mode = {"live": "full", "retrieval_only": "retrieval_only"}
    if execution_mode not in expected_scope_by_mode:
        raise ValueError("formal evaluation execution_mode is unsupported")
    if scenario["observation_scope"] != expected_scope_by_mode[execution_mode]:
        raise ValueError("formal evaluation scenario scope does not match execution mode")
    expected_effects = (
        {"vector_read", "external_llm"} if execution_mode == "live" else {"vector_read"}
    )
    if set(scenario["allowed_side_effects"]) != expected_effects:
        raise ValueError("formal evaluation scenario side effects do not match execution mode")
    if scenario["external_model_allowed"] is not (execution_mode == "live"):
        raise ValueError("formal evaluation model permission does not match execution mode")

    graders = raw_intent.get("graders")
    if (
        not isinstance(graders, list)
        or not graders
        or not all(isinstance(item, str) and item for item in graders)
        or len(graders) != len(set(graders))
    ):
        raise ValueError("formal evaluation graders must be unique non-empty strings")
    failure_policy = raw_intent.get("failure_policy")
    if failure_policy not in {"fail_fast", "continue"}:
        raise ValueError("formal evaluation failure policy is unsupported")
    config_schema_version = raw_intent.get("config_schema_version")
    if config_schema_version != "stt.eval.harness-config.v3":
        raise ValueError("formal evaluation config schema must be v3")
    config_id = _formal_identifier(raw_intent.get("config_id"), "config_id")
    config_sha256 = _formal_sha256(raw_intent.get("config_sha256"), "config_sha256")
    target_id = _formal_identifier(raw_intent.get("target_id"), "target_id")
    planning_admission_budgets = _freeze_planning_admission_budgets(
        raw_intent.get("planning_admission_budgets")
    )
    usage_accounting = _freeze_usage_accounting(raw_intent.get("usage_accounting"))
    raw_interventions = raw_intent.get("allowed_intervention_dimensions")
    if (
        not isinstance(raw_interventions, list)
        or not all(isinstance(item, str) and item for item in raw_interventions)
        or len(raw_interventions) != len(set(raw_interventions))
    ):
        raise ValueError("formal evaluation intervention dimensions must be unique strings")
    unknown_interventions = sorted(set(raw_interventions) - _FORMAL_INTERVENTION_DIMENSIONS)
    if unknown_interventions:
        raise ValueError(
            "formal evaluation intervention dimensions are unsupported: "
            + ", ".join(unknown_interventions)
        )
    allowed_interventions = sorted(raw_interventions)
    normalized_config = {
        "schema_version": config_schema_version,
        "config_id": config_id,
        "target_id": target_id,
        "dataset_id": dataset["registry_id"],
        "scenario_id": scenario["scenario_id"],
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "planning_admission_budgets": planning_admission_budgets,
        "usage_accounting": usage_accounting,
        "allowed_intervention_dimensions": allowed_interventions,
    }
    if config_sha256 != content_digest(normalized_config):
        raise ValueError("formal evaluation config sha256 does not match its content")

    normalized = {
        "intent_version": FORMAL_EVALUATION_INTENT_VERSION,
        "config_schema_version": config_schema_version,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "target_id": target_id,
        "dataset": dataset,
        "scenario": scenario,
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "execution_mode": execution_mode,
        "planning_admission_budgets": planning_admission_budgets,
        "usage_accounting": usage_accounting,
        "allowed_intervention_dimensions": allowed_interventions,
    }
    declared_digest = _formal_sha256(raw_intent.get("intent_sha256"), "intent_sha256")
    actual_digest = content_digest(normalized)
    if declared_digest != actual_digest:
        raise ValueError("formal evaluation intent sha256 does not match its content")
    normalized["intent_sha256"] = actual_digest
    return json.loads(canonical_json(normalized))


def build_formal_evaluation_intent(
    *,
    config_schema_version: str,
    config_id: str,
    config_sha256: str,
    target_id: str,
    dataset: Mapping[str, Any],
    scenario: Mapping[str, Any],
    graders: list[str] | tuple[str, ...],
    failure_policy: str,
    execution_mode: str,
    planning_admission_budgets: Mapping[str, Any],
    usage_accounting: Mapping[str, Any],
    allowed_intervention_dimensions: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """构造不含服务端运行事实的正式评测意图。"""

    intent: dict[str, Any] = {
        "intent_version": FORMAL_EVALUATION_INTENT_VERSION,
        "config_schema_version": config_schema_version,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "target_id": target_id,
        "dataset": dict(dataset),
        "scenario": dict(scenario),
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": True,
        "execution_mode": execution_mode,
        "planning_admission_budgets": dict(planning_admission_budgets),
        "usage_accounting": dict(usage_accounting),
        "allowed_intervention_dimensions": list(allowed_intervention_dimensions),
    }
    intent["intent_sha256"] = content_digest(intent)
    return freeze_formal_evaluation_intent(intent)


def formal_runtime_snapshot_projection(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """返回 contract 外的完整运行快照，避免自引用并绑定所有服务端事实。"""

    return json.loads(
        canonical_json(
            {
                key: value
                for key, value in config_snapshot.items()
                if key not in {FORMAL_EVALUATION_SNAPSHOT_KEY, "selected_case_ids"}
            }
        )
    )


def _freeze_selected_cases(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("formal evaluation selected_cases must be an object")
    _formal_exact_fields(value, _FORMAL_SELECTED_CASE_FIELDS, "selected cases")
    case_ids = value.get("selected_case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(item, str) and item and item == item.strip() for item in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        raise ValueError("formal evaluation selected case ids must be unique canonical strings")
    count = value.get("selected_case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(case_ids):
        raise ValueError("formal evaluation selected case count is invalid")
    digest = _formal_sha256(
        value.get("selected_case_ids_sha256"), "selected_cases.selected_case_ids_sha256"
    )
    if digest != content_digest({"selected_case_ids": case_ids}):
        raise ValueError("formal evaluation selected case digest does not match its content")
    return {
        "selected_case_ids": list(case_ids),
        "selected_case_count": count,
        "selected_case_ids_sha256": digest,
    }


def _freeze_formal_run_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("formal evaluation run_authority must be an object")
    _formal_exact_fields(value, _FORMAL_RUN_AUTHORITY_FIELDS, "run authority")
    baseline = value.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("formal evaluation baseline authority must be an object")
    _formal_exact_fields(baseline, _FORMAL_BASELINE_FIELDS, "baseline authority")
    baseline_run_id = baseline.get("baseline_run_id")
    baseline_contract_sha = baseline.get("baseline_formal_contract_sha256")
    if (baseline_run_id is None) is not (baseline_contract_sha is None):
        raise ValueError("formal evaluation baseline id and contract sha must be paired")
    if baseline_run_id is not None:
        baseline_run_id = _formal_identifier(baseline_run_id, "baseline.baseline_run_id")
        baseline_contract_sha = _formal_sha256(
            baseline_contract_sha,
            "baseline.baseline_formal_contract_sha256",
        )

    raw_profile = value.get("retrieval_profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError("formal evaluation retrieval profile must be an object")
    _formal_exact_fields(raw_profile, _FORMAL_RETRIEVAL_PROFILE_FIELDS, "retrieval profile")
    profile_version = raw_profile.get("version")
    if (
        isinstance(profile_version, bool)
        or not isinstance(profile_version, int)
        or profile_version < 1
    ):
        raise ValueError("formal evaluation retrieval profile version must be positive")
    retrieval_profile = {
        "profile_id": _formal_identifier(
            raw_profile.get("profile_id"), "retrieval_profile.profile_id"
        ),
        "version": profile_version,
        "config_hash": _formal_sha256(
            raw_profile.get("config_hash"), "retrieval_profile.config_hash"
        ),
    }

    raw_vector = value.get("vector_read_set")
    if not isinstance(raw_vector, Mapping):
        raise ValueError("formal evaluation vector read set must be an object")
    _formal_exact_fields(raw_vector, _FORMAL_VECTOR_READ_SET_FIELDS, "vector read set")
    scopes = raw_vector.get("scopes")
    if (
        not isinstance(scopes, list)
        or not scopes
        or not all(isinstance(item, Mapping) for item in scopes)
    ):
        raise ValueError("formal evaluation vector read scopes must be non-empty objects")
    normalized_scopes = json.loads(canonical_json(scopes))
    scopes_sha = _formal_sha256(raw_vector.get("scopes_sha256"), "vector_read_set.scopes_sha256")
    if scopes_sha != content_digest(normalized_scopes):
        raise ValueError("formal evaluation vector read scopes sha does not match")
    vector_read_set = {
        "digest": _formal_canonical_sha256_digest(
            raw_vector.get("digest"), "vector_read_set.digest"
        ),
        "scopes": normalized_scopes,
        "scopes_sha256": scopes_sha,
    }

    raw_models = value.get("model_ids")
    if not isinstance(raw_models, Mapping):
        raise ValueError("formal evaluation model_ids must be an object")
    _formal_exact_fields(raw_models, _FORMAL_MODEL_ID_FIELDS, "model ids")
    model_ids = {
        field: _formal_identifier(raw_models.get(field), f"model_ids.{field}")
        for field in sorted(_FORMAL_MODEL_ID_FIELDS)
    }
    evaluation_security = _formal_json_object(
        value.get("evaluation_security"), "evaluation_security", require_nonempty=True
    )
    evaluation_security_sha = _formal_sha256(
        value.get("evaluation_security_sha256"), "evaluation_security_sha256"
    )
    if evaluation_security_sha != content_digest(evaluation_security):
        raise ValueError("formal evaluation security sha does not match its content")
    runtime_flags = _formal_json_object(
        value.get("runtime_flags"), "runtime_flags", require_nonempty=True
    )
    runtime_flags_sha = _formal_sha256(value.get("runtime_flags_sha256"), "runtime_flags_sha256")
    if runtime_flags_sha != content_digest(runtime_flags):
        raise ValueError("formal evaluation runtime flags sha does not match their content")
    raw_controls = value.get("enforced_execution_controls")
    if not isinstance(raw_controls, Mapping):
        raise ValueError("formal evaluation enforced execution controls must be an object")
    _formal_exact_fields(
        raw_controls,
        _FORMAL_EXECUTION_CONTROL_FIELDS,
        "enforced execution controls",
    )
    route_timeout = raw_controls.get("query_plan_route_timeout_ms")
    if (
        isinstance(route_timeout, bool)
        or not isinstance(route_timeout, int)
        or not 250 <= route_timeout <= 120_000
    ):
        raise ValueError("formal evaluation query plan route timeout is invalid")
    output_cap = raw_controls.get("generation_output_token_cap")
    if output_cap is not None and (
        isinstance(output_cap, bool)
        or not isinstance(output_cap, int)
        or not 1 <= output_cap <= 131_072
    ):
        raise ValueError("formal evaluation generation output token cap is invalid")
    provider_timeout = raw_controls.get("provider_request_timeout_seconds")
    provider_attempts = raw_controls.get("provider_max_attempts_per_request")
    provider_retry_base = raw_controls.get("provider_retry_base_delay_seconds")
    provider_retry_max = raw_controls.get("provider_retry_max_delay_seconds")
    provider_controls = (
        provider_timeout,
        provider_attempts,
        provider_retry_base,
        provider_retry_max,
    )
    if any(item is None for item in provider_controls) and not all(
        item is None for item in provider_controls
    ):
        raise ValueError("formal evaluation provider controls must be all present or all absent")
    if provider_timeout is not None:
        if (
            isinstance(provider_timeout, bool)
            or not isinstance(provider_timeout, (int, float))
            or not math.isfinite(float(provider_timeout))
            or float(provider_timeout) <= 0
        ):
            raise ValueError("formal evaluation provider request timeout is invalid")
        if (
            isinstance(provider_attempts, bool)
            or not isinstance(provider_attempts, int)
            or provider_attempts < 1
        ):
            raise ValueError("formal evaluation provider max attempts is invalid")
        for delay in (provider_retry_base, provider_retry_max):
            if (
                isinstance(delay, bool)
                or not isinstance(delay, (int, float))
                or not math.isfinite(float(delay))
                or float(delay) < 0
            ):
                raise ValueError("formal evaluation provider retry delay is invalid")
        if float(provider_retry_base) > float(provider_retry_max):
            raise ValueError("formal evaluation provider retry delays are inconsistent")
    enforced_execution_controls = {
        "query_plan_route_timeout_ms": route_timeout,
        "generation_output_token_cap": output_cap,
        "provider_request_timeout_seconds": (
            float(provider_timeout) if provider_timeout is not None else None
        ),
        "provider_max_attempts_per_request": provider_attempts,
        "provider_retry_base_delay_seconds": (
            float(provider_retry_base) if provider_retry_base is not None else None
        ),
        "provider_retry_max_delay_seconds": (
            float(provider_retry_max) if provider_retry_max is not None else None
        ),
    }
    normalized = {
        "run_id": _formal_identifier(value.get("run_id"), "run_id"),
        "tenant_id": _formal_identifier(value.get("tenant_id"), "tenant_id"),
        "kb_id": _formal_identifier(value.get("kb_id"), "kb_id"),
        "baseline": {
            "baseline_run_id": baseline_run_id,
            "baseline_formal_contract_sha256": baseline_contract_sha,
        },
        "selected_cases": _freeze_selected_cases(value.get("selected_cases")),
        "retrieval_profile": retrieval_profile,
        "vector_read_set": vector_read_set,
        "evaluation_security": evaluation_security,
        "evaluation_security_sha256": evaluation_security_sha,
        "code_revision": _formal_identifier(value.get("code_revision"), "code_revision"),
        "config_version": _formal_identifier(value.get("config_version"), "config_version"),
        "model_ids": model_ids,
        "runtime_flags": runtime_flags,
        "runtime_flags_sha256": runtime_flags_sha,
        "enforced_execution_controls": enforced_execution_controls,
        "preflight_sha256": _formal_sha256(value.get("preflight_sha256"), "preflight_sha256"),
        "runtime_snapshot_sha256": _formal_sha256(
            value.get("runtime_snapshot_sha256"), "runtime_snapshot_sha256"
        ),
        "comparison_snapshot_sha256": _formal_sha256(
            value.get("comparison_snapshot_sha256"), "comparison_snapshot_sha256"
        ),
    }
    return json.loads(canonical_json(normalized))


def formal_comparison_snapshot_projection(
    runtime_snapshot: Mapping[str, Any],
    allowed_intervention_dimensions: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """仅遮蔽显式干预维度；其他运行事实变化仍破坏 paired 可比性。"""

    comparison = json.loads(canonical_json(dict(runtime_snapshot)))
    allowed = set(allowed_intervention_dimensions)
    field_groups = {
        "code_revision": ("code_revision", "answering_contract"),
        "config_version": ("config_version",),
        "retrieval_profile": ("retrieval_profile",),
        "vector_read_set": ("vector_read_scopes", "vector_read_set_digest"),
        "evaluation_security": ("evaluation_security",),
        "model_ids": ("embedding_model", "reranker_model", "llm_model"),
        "runtime_flags": ("runtime_flags",),
    }
    for dimension in allowed:
        for field in field_groups[dimension]:
            comparison.pop(field, None)
    preflight = comparison.get("preflight")
    if isinstance(preflight, dict):
        if "retrieval_profile" in allowed:
            preflight.pop("retrieval_profile", None)
            preflight.pop("retrieval_profile_selection", None)
        if "vector_read_set" in allowed:
            preflight.pop("vector_read_scopes", None)
            preflight.pop("vector_read_set_digest", None)
        if "evaluation_security" in allowed:
            preflight.pop("evaluation_security", None)
    return comparison


def _runtime_authority_from_snapshot(
    *,
    config_snapshot: Mapping[str, Any],
    run_id: str,
    tenant_id: str,
    kb_id: str,
    baseline_run_id: str | None,
    baseline_formal_contract_sha256: str | None,
    selected_case_ids: list[str] | tuple[str, ...],
    allowed_intervention_dimensions: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    runtime_snapshot = formal_runtime_snapshot_projection(config_snapshot)
    retrieval_profile = _formal_json_object(
        runtime_snapshot.get("retrieval_profile"),
        "retrieval_profile",
        require_nonempty=True,
    )
    scopes = runtime_snapshot.get("vector_read_scopes")
    if not isinstance(scopes, list):
        scopes = []
    evaluation_security = _formal_json_object(
        runtime_snapshot.get("evaluation_security"),
        "evaluation_security",
        require_nonempty=True,
    )
    runtime_flags = _formal_json_object(
        runtime_snapshot.get("runtime_flags"),
        "runtime_flags",
        require_nonempty=True,
    )
    preflight = _formal_json_object(
        runtime_snapshot.get("preflight"),
        "preflight",
        require_nonempty=True,
    )
    case_ids = list(selected_case_ids)
    profile_config = _formal_json_object(
        retrieval_profile.get("config"),
        "retrieval_profile.config",
    )
    context_pack = profile_config.get("context_pack")
    context_pack = context_pack if isinstance(context_pack, dict) else {}
    answering_contract = runtime_snapshot.get("answering_contract")
    if answering_contract is None:
        # 历史快照尚未单独冻结回答输出契约，只能保留原有 profile 投影。
        generation_output_token_cap = max(
            1, int(context_pack.get("reserved_output_tokens") or 1_024)
        )
    else:
        answering_contract = _formal_json_object(
            answering_contract,
            "answering_contract",
            require_nonempty=True,
        )
        generation_output_token_cap = answering_contract.get("generation_output_token_cap")
        if (
            isinstance(generation_output_token_cap, bool)
            or not isinstance(generation_output_token_cap, int)
            or generation_output_token_cap <= 0
        ):
            raise ValueError("formal answering generation output token cap is invalid")
    route_timeout = max(250, min(120_000, int(profile_config.get("timeout_ms") or 8_000)))
    live_mode = runtime_snapshot.get("evaluation_mode") == "live"
    enforced_execution_controls = {
        "query_plan_route_timeout_ms": route_timeout,
        "generation_output_token_cap": generation_output_token_cap if live_mode else None,
        "provider_request_timeout_seconds": (
            runtime_flags.get("llm_request_timeout_seconds") if live_mode else None
        ),
        "provider_max_attempts_per_request": (
            runtime_flags.get("llm_max_attempts") if live_mode else None
        ),
        "provider_retry_base_delay_seconds": (
            runtime_flags.get("llm_retry_base_delay_seconds") if live_mode else None
        ),
        "provider_retry_max_delay_seconds": (
            runtime_flags.get("llm_retry_max_delay_seconds") if live_mode else None
        ),
    }
    authority = {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "baseline": {
            "baseline_run_id": baseline_run_id,
            "baseline_formal_contract_sha256": baseline_formal_contract_sha256,
        },
        "selected_cases": {
            "selected_case_ids": case_ids,
            "selected_case_count": len(case_ids),
            "selected_case_ids_sha256": content_digest({"selected_case_ids": case_ids}),
        },
        "retrieval_profile": {
            "profile_id": retrieval_profile.get("profile_id"),
            "version": retrieval_profile.get("version"),
            "config_hash": retrieval_profile.get("config_hash"),
        },
        "vector_read_set": {
            "digest": runtime_snapshot.get("vector_read_set_digest"),
            "scopes": scopes,
            "scopes_sha256": content_digest(scopes),
        },
        "evaluation_security": evaluation_security,
        "evaluation_security_sha256": content_digest(evaluation_security),
        "code_revision": runtime_snapshot.get("code_revision"),
        "config_version": runtime_snapshot.get("config_version"),
        "model_ids": {
            "embedding": runtime_snapshot.get("embedding_model"),
            "reranker": runtime_snapshot.get("reranker_model"),
            "llm": runtime_snapshot.get("llm_model"),
        },
        "runtime_flags": runtime_flags,
        "runtime_flags_sha256": content_digest(runtime_flags),
        "enforced_execution_controls": enforced_execution_controls,
        "preflight_sha256": content_digest(preflight),
        "runtime_snapshot_sha256": content_digest(runtime_snapshot),
        "comparison_snapshot_sha256": content_digest(
            formal_comparison_snapshot_projection(
                runtime_snapshot,
                allowed_intervention_dimensions,
            )
        ),
    }
    return _freeze_formal_run_authority(authority)


def finalize_formal_evaluation_contract(
    raw_intent: Any,
    *,
    config_snapshot: Mapping[str, Any],
    run_id: str,
    tenant_id: str,
    kb_id: str,
    baseline_run_id: str | None,
    baseline_formal_contract_sha256: str | None,
    selected_case_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """由 repository 使用实际运行事实完成不可移植的正式 contract。"""

    intent = freeze_formal_evaluation_intent(raw_intent)
    contract: dict[str, Any] = {
        "contract_version": FORMAL_EVALUATION_CONTRACT_VERSION,
        "intent": intent,
        "run_authority": _runtime_authority_from_snapshot(
            config_snapshot=config_snapshot,
            run_id=run_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            baseline_run_id=baseline_run_id,
            baseline_formal_contract_sha256=baseline_formal_contract_sha256,
            selected_case_ids=selected_case_ids,
            allowed_intervention_dimensions=intent["allowed_intervention_dimensions"],
        ),
    }
    contract["contract_sha256"] = content_digest(contract)
    return freeze_formal_evaluation_contract(
        contract,
        expected_mode=intent["execution_mode"],
        expected_dataset_record_id=intent["dataset"]["record_id"],
        expected_dataset_checksum=intent["dataset"]["record_checksum"],
        expected_dataset_key=intent["dataset"]["dataset_key"],
        expected_dataset_version=intent["dataset"]["dataset_version"],
        expected_run_id=run_id,
        expected_tenant_id=tenant_id,
        expected_kb_id=kb_id,
        expected_baseline_run_id=baseline_run_id,
        expected_baseline_formal_contract_sha256=baseline_formal_contract_sha256,
        expected_selected_case_ids=selected_case_ids,
        expected_runtime_snapshot=config_snapshot,
    )


def freeze_formal_evaluation_contract(
    raw_contract: Any,
    *,
    expected_mode: str | None = None,
    expected_dataset_record_id: str | None = None,
    expected_dataset_checksum: str | None = None,
    expected_dataset_key: str | None = None,
    expected_dataset_version: str | None = None,
    expected_run_id: str | None = None,
    expected_tenant_id: str | None = None,
    expected_kb_id: str | None = None,
    expected_baseline_run_id: Any = _EXPECTATION_UNSET,
    expected_baseline_formal_contract_sha256: Any = _EXPECTATION_UNSET,
    expected_selected_case_ids: list[str] | tuple[str, ...] | None = None,
    expected_total_cases: int | None = None,
    expected_runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """验证历史 v1 或当前 v2 contract；当前 contract 绑定实际运行行。"""

    if not isinstance(raw_contract, Mapping):
        raise ValueError("formal evaluation contract must be an object")
    if raw_contract.get("contract_version") == LEGACY_FORMAL_EVALUATION_CONTRACT_VERSION:
        return _freeze_legacy_formal_evaluation_contract(
            raw_contract,
            expected_mode=expected_mode,
            expected_dataset_record_id=expected_dataset_record_id,
            expected_dataset_checksum=expected_dataset_checksum,
            expected_dataset_key=expected_dataset_key,
            expected_dataset_version=expected_dataset_version,
        )
    _formal_exact_fields(raw_contract, _FORMAL_V2_CONTRACT_FIELDS, "contract")
    if raw_contract.get("contract_version") != FORMAL_EVALUATION_CONTRACT_VERSION:
        raise ValueError("unsupported formal evaluation contract version")
    intent = freeze_formal_evaluation_intent(raw_contract.get("intent"))
    authority = _freeze_formal_run_authority(raw_contract.get("run_authority"))
    normalized = {
        "contract_version": FORMAL_EVALUATION_CONTRACT_VERSION,
        "intent": intent,
        "run_authority": authority,
    }
    declared_digest = _formal_sha256(raw_contract.get("contract_sha256"), "contract_sha256")
    actual_digest = content_digest(normalized)
    if declared_digest != actual_digest:
        raise ValueError("formal evaluation contract sha256 does not match its content")
    normalized["contract_sha256"] = actual_digest

    dataset = intent["dataset"]
    expectations: dict[str, tuple[Any, Any]] = {
        "execution mode": (expected_mode, intent["execution_mode"]),
        "dataset record id": (expected_dataset_record_id, dataset["record_id"]),
        "dataset checksum": (expected_dataset_checksum, dataset["record_checksum"]),
        "dataset key": (expected_dataset_key, dataset["dataset_key"]),
        "dataset version": (expected_dataset_version, dataset["dataset_version"]),
        "run id": (expected_run_id, authority["run_id"]),
        "tenant id": (expected_tenant_id, authority["tenant_id"]),
        "knowledge base id": (expected_kb_id, authority["kb_id"]),
    }
    for label, (expected, actual) in expectations.items():
        if expected is not None and expected != actual:
            raise ValueError(f"formal evaluation {label} does not match the durable run")
    baseline_expectations = {
        "baseline run id": (
            expected_baseline_run_id,
            authority["baseline"]["baseline_run_id"],
        ),
        "baseline formal contract sha256": (
            expected_baseline_formal_contract_sha256,
            authority["baseline"]["baseline_formal_contract_sha256"],
        ),
    }
    for label, (expected, actual) in baseline_expectations.items():
        if expected is not _EXPECTATION_UNSET and expected != actual:
            raise ValueError(f"formal evaluation {label} does not match the durable run")
    if expected_selected_case_ids is not None:
        selected = authority["selected_cases"]
        if list(expected_selected_case_ids) != selected["selected_case_ids"]:
            raise ValueError("formal evaluation selected cases do not match the durable run")
    if (
        expected_total_cases is not None
        and expected_total_cases != authority["selected_cases"]["selected_case_count"]
    ):
        raise ValueError("formal evaluation selected case count does not match the durable run")

    if expected_runtime_snapshot is not None:
        runtime_snapshot = formal_runtime_snapshot_projection(expected_runtime_snapshot)
        expected_authority = _runtime_authority_from_snapshot(
            config_snapshot=expected_runtime_snapshot,
            run_id=authority["run_id"],
            tenant_id=authority["tenant_id"],
            kb_id=authority["kb_id"],
            baseline_run_id=authority["baseline"]["baseline_run_id"],
            baseline_formal_contract_sha256=authority["baseline"][
                "baseline_formal_contract_sha256"
            ],
            selected_case_ids=authority["selected_cases"]["selected_case_ids"],
            allowed_intervention_dimensions=intent["allowed_intervention_dimensions"],
        )
        if authority != expected_authority:
            raise ValueError("formal evaluation runtime authority does not match the snapshot")
        if authority["runtime_snapshot_sha256"] != content_digest(runtime_snapshot):
            raise ValueError("formal evaluation runtime snapshot sha does not match")
    return json.loads(canonical_json(normalized))
