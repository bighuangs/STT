"""独立的 Easy/Medium non-thinking 复审执行合同与幂等账本。

本模块不修改冻结的 2.32 request、execution contract、calibration 或响应账本。
它从冻结 request 确定性叠加 ``enable_thinking=false``，把新调用写入独立的
append-only journal/response 视图，并将新结果和旧正式账本一起严格重放。

外部调用必须遵循 ``started -> result -> completed``：``started`` 在调用前
fsync；若重启时只有 ``started``，副作用状态未知并 fail-stop；若已有 result
但缺 completed，则只补确定性的 completed 事件，绝不再次调用供应商。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Literal


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import review_contracts as legacy  # noqa: E402
import easy_medium_scope as scope  # noqa: E402
from concept_materialization import (  # noqa: E402
    LEGACY_MATERIALIZATION_POLICY_VERSION,
    MATERIALIZATION_POLICY_VERSION,
    V5_MATERIALIZATION_POLICY_VERSION,
    PRODUCTION_MATCHER,
    PRODUCTION_MATCHER_POLICY_VERSION,
    SUBSET_SOLVER_POLICY,
)


PROFILE_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-profile.v6"
EVENT_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-journal.v6"
RESULT_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-result.v6"
SUMMARY_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-summary.v6"
PROFILE_ID = "easy-medium-qwen-json-nonthinking-2026-08-13-v6"
PROFILE_RUNNER_VERSION = "2.38.0-easy-medium-nonthinking-json-v6"
V5_PROFILE_ID = "easy-medium-qwen-json-nonthinking-2026-08-13-v5"
V5_PROFILE_RUNNER_VERSION = "2.37.0-easy-medium-nonthinking-json-v5"
V5_PROFILE_SHA256 = (
    "d16edccf3d4406b5018da5f0468b170bcf299dcfdd8d097392cfee6d8c1746e1"
)
V5_ADOPTION_POLICY = (
    "stt.eval.easy-medium-nonthinking.v6-adopt-v5-result-only-materializer-v1"
)
V5_ADOPTION_RESULT_ONLY_ID = "concept-contract-rereview-fac9cb70ec1293521f33"
V5_ADOPTION_OLD_ERRORS_BY_ID = {
    V5_ADOPTION_RESULT_ONLY_ID: [
        "concept materialization failed: concept 'c02' has no "
        "production-discriminative variant"
    ]
}
V5_PHYSICAL_SIDECAR_SHA256 = {
    "execution-profile.json": "468a974b81aa5bef16c86edf97e031fcb4c15e39ce744aade3c4f6caa9363c0d",
    "journal.jsonl": "5332ef98ba2e1f8a0332101d627a47e0b7771d80478e7451ae85fa14f1d603ca",
    "responses.jsonl": "b31012b47e29facd6873885618e2560cbda3d1eee10c8b68f18a947a9ad4ef5a",
    "calibration-responses.jsonl": "ca53fcd08ac49fe8c479991315f90bcc273f653066e5ec24830d83cafec24113",
}
V5_STARTED_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "event_type",
        "kind",
        "logical_custom_id",
        "sequence",
        "operation_id",
        "execution_profile_sha256",
        "base_execution_contract_sha256",
        "base_request_sha256",
        "effective_request_sha256",
        "started_at",
        "side_effect_status",
    }
)
V5_COMPLETED_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "event_type",
        "kind",
        "logical_custom_id",
        "sequence",
        "operation_id",
        "execution_profile_sha256",
        "started_event_sha256",
        "result_envelope_sha256",
        "result_status",
        "completed_at",
        "side_effect_status",
    }
)
V5_RESULT_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "event_type",
        "kind",
        "logical_custom_id",
        "sequence",
        "operation_id",
        "execution_profile_sha256",
        "profile_runner_version",
        "model",
        "endpoint",
        "transport",
        "sse_completion_policy",
        "request_overlay",
        "base_execution_contract_sha256",
        "base_request_sha256",
        "effective_request_sha256",
        "started_event_sha256",
        "core_result_sha256",
        "core_result",
    }
)
V5_ADOPTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "policy",
        "source_profile_id",
        "source_profile_sha256",
        "source_materializer_policy_version",
        "source_production_matcher_policy_version",
        "target_materializer_policy_version",
        "target_production_matcher_policy_version",
        "physical_sidecar_sha256",
        "execution_summary_state",
        "physical_row_counts",
        "adopted_result_envelope_sha256_by_id",
        "adopted_started_event_sha256_by_id",
        "adopted_completed_event_sha256_by_id",
        "adopted_core_result_sha256_by_id",
        "adopted_old_materialization_errors_by_id",
        "adopted_new_materialization_errors_by_id",
        "calibration_base_request_sha256_by_id",
        "calibration_request_sha256_by_id",
        "calibration_result_envelope_sha256_by_id",
        "calibration_started_event_sha256_by_id",
        "calibration_completed_event_sha256_by_id",
        "calibration_core_result_sha256_by_id",
        "calibration_evidence_role",
        "calibration_authorizes_v6_calls",
        "receipt_sha256",
    }
)
V4_PROFILE_ID = "easy-medium-qwen-json-nonthinking-2026-08-13-v4"
LEGACY_PRODUCTION_MATCHER_POLICY_VERSION = "stt.eval.concept-variant-matcher.v3"
V4_PROFILE_RUNNER_VERSION = "2.36.0-easy-medium-nonthinking-json-v4"
V4_PROFILE_SHA256 = (
    "0531602a109549045cd737bb3da93ad3ef148a63fc876e0a77d1c0c6611db2c9"
)
V4_ADOPTION_POLICY = (
    "stt.eval.easy-medium-nonthinking.v6-adopt-v4-materializer-only-v1"
)
V4_ADOPTION_COMPLETED_ID = "concept-contract-rereview-52d20717a579d2f585a6"
V4_ADOPTION_RESULT_ONLY_ID = "concept-contract-rereview-056e9d3e6c713db68790"
V4_ADOPTION_OLD_ERRORS_BY_ID = {
    V4_ADOPTION_COMPLETED_ID: [],
    V4_ADOPTION_RESULT_ONLY_ID: [
        "concept materialization failed: concept 'c01' has no "
        "production-discriminative variant"
    ],
}
PREDECESSOR_PROFILE_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-profile.v1"
PREDECESSOR_EVENT_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-journal.v1"
PREDECESSOR_RESULT_SCHEMA_VERSION = "stt.eval.easy-medium-nonthinking-result.v1"
PREDECESSOR_PROFILE_ID = "easy-medium-qwen-json-nonthinking-2026-08-13-v1"
PREDECESSOR_RUNNER_VERSION = "2.33.0-easy-medium-nonthinking-json-v1"
PREDECESSOR_PROFILE_SHA256 = (
    "8cba95ca13fdaf93904cdefe9be2a238f8df8ab403d55102fe60ecd324ade7e0"
)
LEGACY_RUNNER_VERSION = legacy.RUNNER_VERSION
TRANSPORT = legacy.TRANSPORT
SSE_COMPLETION_POLICY = legacy.SSE_COMPLETION_POLICY
REQUEST_OVERLAY = {"enable_thinking": False}
POLARITY_REQUIRED_VALUES = (
    "abstain_without_factual_claims",
    "affirmative",
    "affirmative_or_explicit_yes_no",
    "mixed_preserve_each_claim",
    "negative",
    "not_applicable",
    "positive",
    "positive_with_conditional",
)
NEUTRAL_POLARITY_POLICY = (
    "Neutral descriptive, identifier, or procedural action-to-outcome facts that "
    "express no negation, prohibition, permission, obligation or requirement, "
    "recommendation, comparison or contrast, or condition MUST use "
    'required="not_applicable" and cues=[]. A grammatically positive statement '
    "or a neutral fact being true MUST NOT by itself be classified as "
    '"affirmative".'
)
PROMPT_SCHEMA_APPENDIX = (
    "\n\nSchema invariant: critical_constraints.polarity.required MUST be exactly "
    "one frozen JSON string enum value: "
    + json.dumps(POLARITY_REQUIRED_VALUES, ensure_ascii=False, separators=(",", ":"))
    + ". Boolean, number, null, object, and array values are invalid. Use "
    '"not_applicable" when polarity is not semantically applicable. '
    "critical_constraints.polarity MUST be an object containing exactly these "
    'two keys: {"required":<one frozen enum string>,"cues":<JSON string array>}. '
    "critical_constraints.polarity.cues MUST be a JSON array of strings and is "
    "REQUIRED. Do not omit the cues key; use [] when no polarity cue applies. "
    + NEUTRAL_POLARITY_POLICY
)
KINDS = ("blocker", "contract")
ALL_EVENT_KINDS = (*KINDS, "calibration")
SUCCESS_STATUSES = frozenset({"succeeded", "calibration_projectable_pass"})
RESULT_STATUSES = frozenset({*SUCCESS_STATUSES, "failed"})


class ProfileContractError(RuntimeError):
    """执行 profile、journal 或 result 违反确定性合同。"""


@dataclass(frozen=True)
class PackInput:
    """冻结 full pack、lineage 映射与独立 scope authority。"""

    requests: tuple[dict[str, Any], ...]
    request_map: tuple[dict[str, Any], ...]
    authority_rows: tuple[dict[str, Any], ...]
    base_execution_contract_sha256: str


ReplayLegacyState = Callable[
    [list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]
]


@dataclass(frozen=True)
class PredecessorBridge:
    """经 v1 profile/journal/result/calibration exact replay 后的只读桥接证据。"""

    receipt: dict[str, Any]
    success_rows: tuple[dict[str, Any], ...]
    success_envelopes: tuple[dict[str, Any], ...]
    failed_envelopes: tuple[dict[str, Any], ...]
    source_core_rows: tuple[dict[str, Any], ...]
    calibration_core_rows: tuple[dict[str, Any], ...]
    completed_events: tuple[dict[str, Any], ...]
    validate_predecessor_core: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ]


@dataclass(frozen=True)
class V4AdoptionBridge:
    """对v4已付费结果与4/4 calibration做物理pin及零API重放。"""

    receipt: dict[str, Any]
    source_profile: dict[str, Any]
    success_rows: tuple[dict[str, Any], ...]
    source_envelopes: tuple[dict[str, Any], ...]
    source_started_events: tuple[dict[str, Any], ...]
    source_completed_events: tuple[dict[str, Any], ...]
    source_core_rows: tuple[dict[str, Any], ...]
    calibration_envelopes: tuple[dict[str, Any], ...]
    calibration_base_requests: tuple[dict[str, Any], ...]
    calibration_requests: tuple[dict[str, Any], ...]
    calibration_started_events: tuple[dict[str, Any], ...]
    calibration_completed_events: tuple[dict[str, Any], ...]
    validate_v4_source_core: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ]
    validate_v5_adoption_core: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ]


@dataclass(frozen=True)
class V5AdoptionBridge:
    """对v5 result-only及4/4 calibration做物理pin与零API采用。"""

    receipt: dict[str, Any]
    source_profile: dict[str, Any]
    success_rows: tuple[dict[str, Any], ...]
    source_envelopes: tuple[dict[str, Any], ...]
    source_started_events: tuple[dict[str, Any], ...]
    source_core_rows: tuple[dict[str, Any], ...]
    calibration_envelopes: tuple[dict[str, Any], ...]
    calibration_base_requests: tuple[dict[str, Any], ...]
    calibration_requests: tuple[dict[str, Any], ...]
    calibration_started_events: tuple[dict[str, Any], ...]
    calibration_completed_events: tuple[dict[str, Any], ...]
    validate_v5_source_core: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ]
    validate_v6_adoption_core: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ]


@dataclass(frozen=True)
class PackRuntime:
    """一个 rereview pack 的旧账本与共享重放边界。"""

    kind: Literal["blocker", "contract"]
    pack: PackInput
    legacy_rows: tuple[dict[str, Any], ...]
    replay_legacy: ReplayLegacyState
    predecessor_bridge: PredecessorBridge | None = None
    v4_adoption_bridge: V4AdoptionBridge | None = None
    v5_adoption_bridge: V5AdoptionBridge | None = None


def _selection(pack: PackInput) -> scope.ScopeSelection:
    """用正式 scope selector 重算 E/M，调用方不能直接声明 target ID。"""

    try:
        selection = scope.select_scope_requests(
            list(pack.requests),
            list(pack.request_map),
            scope.SCOPE_NAME,
            authority_rows=list(pack.authority_rows),
        )
    except RuntimeError as exc:
        raise ProfileContractError(f"scope authority validation failed: {exc}") from exc
    difficulties = dict(selection.target_difficulty_by_custom_id)
    if (
        not selection.target_custom_ids
        or set(difficulties) != set(selection.target_custom_ids)
        or set(difficulties.values()) - set(scope.SCOPE_DIFFICULTIES)
        or set(selection.target_difficulty_counts) - set(scope.SCOPE_DIFFICULTIES)
    ):
        raise ProfileContractError("scope selection contains a Hard/non-E/M target")
    return selection


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProfileContractError(f"{label} must be a lowercase SHA-256")
    return text


def _validate_runtime_knobs(
    *, request_timeout: float, max_attempts: int, retry_delay: float
) -> None:
    if isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float)):
        raise ProfileContractError("request_timeout must be numeric")
    if float(request_timeout) <= 0:
        raise ProfileContractError("request_timeout must be positive")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ProfileContractError("max_attempts must be an integer in [1, 3]")
    if isinstance(retry_delay, bool) or not isinstance(retry_delay, (int, float)):
        raise ProfileContractError("retry_delay must be numeric")
    if float(retry_delay) < 0:
        raise ProfileContractError("retry_delay must be non-negative")


def _base_request_errors(
    row: Mapping[str, Any], *, expected_model: str | None = None
) -> list[str]:
    errors: list[str] = []
    if set(row) != {"custom_id", "payload_sha256", "prompt_sha256", "request"}:
        errors.append("base request row schema is not exact")
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        errors.append("base request custom_id is invalid")
    request = row.get("request")
    if not isinstance(request, Mapping):
        return [*errors, "base request body is not an object"]
    if "enable_thinking" in request:
        errors.append("frozen base request must not already contain enable_thinking")
    if set(request) != {"messages", "model", "response_format", "temperature"}:
        errors.append("base request config schema is not exact")
    if request.get("temperature") != 0 or isinstance(request.get("temperature"), bool):
        errors.append("base request temperature must be exact numeric zero")
    if request.get("response_format") != {"type": "json_object"}:
        errors.append("base request response_format must be json_object")
    if not isinstance(request.get("model"), str) or not request.get("model"):
        errors.append("base request model is invalid")
    elif expected_model is not None and request.get("model") != expected_model:
        errors.append("base request model differs from execution profile model")
    messages = request.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not all(isinstance(item, Mapping) for item in messages)
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        errors.append("base request messages are invalid")
    else:
        system_content = messages[0].get("content")
        if not isinstance(system_content, str):
            errors.append("base request system prompt is invalid")
        else:
            actual_prompt_sha256 = hashlib.sha256(
                system_content.encode("utf-8")
            ).hexdigest()
            if row.get("prompt_sha256") != actual_prompt_sha256:
                errors.append("base request prompt content/hash binding drifted")
    return errors


def _effective_from_base(row: Mapping[str, Any]) -> dict[str, Any]:
    errors = _base_request_errors(row)
    if errors:
        raise ProfileContractError("; ".join(errors))
    effective = deepcopy(dict(row))
    base_prompt = str(effective["request"]["messages"][0]["content"])
    effective_prompt = base_prompt + PROMPT_SCHEMA_APPENDIX
    effective["request"]["messages"][0]["content"] = effective_prompt
    effective["prompt_sha256"] = hashlib.sha256(
        effective_prompt.encode("utf-8")
    ).hexdigest()
    effective["request"]["enable_thinking"] = False
    return effective


def _base_from_effective(row: Mapping[str, Any]) -> dict[str, Any]:
    """仅逆转 v2 deterministic overlay；任何非exact prompt bytes均拒绝。"""

    effective = deepcopy(dict(row))
    request = effective.get("request")
    messages = request.get("messages") if isinstance(request, Mapping) else None
    if (
        not isinstance(request, dict)
        or request.get("enable_thinking") is not False
        or not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"].endswith(PROMPT_SCHEMA_APPENDIX)
    ):
        raise ProfileContractError("effective request prompt/overlay is not exact v2")
    prompt = messages[0]["content"]
    expected_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if effective.get("prompt_sha256") != expected_prompt_sha:
        raise ProfileContractError("effective request prompt bytes/hash drifted")
    base_prompt = prompt[: -len(PROMPT_SCHEMA_APPENDIX)]
    messages[0]["content"] = base_prompt
    effective["prompt_sha256"] = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()
    del request["enable_thinking"]
    return effective


def _scope_contract(scope_profile: Mapping[str, Any]) -> dict[str, Any]:
    expected = scope.scope_profile()
    if dict(scope_profile) != expected:
        raise ProfileContractError("scope profile is not exact non-promotable E/M")
    return deepcopy(expected)


def _v4_calibration_cases() -> list[legacy.CalibrationCase]:
    """冻结 v4 四题校准；双重覆盖中性事实的 ``not_applicable`` 语义。"""

    cases = deepcopy(legacy._calibration_cases()[:3])
    cases[1].expected["polarity"] = "not_applicable"
    cases.append(
        legacy.CalibrationCase(
            name="not_applicable_polarity",
            payload={
                "query": "如何打开设置菜单？",
                "reference_answer": "按菜单键打开设置菜单。",
                "unanswerable": False,
                "legacy_expected_terms": ["按菜单键", "打开设置菜单"],
                "forbidden_terms": [],
                "branches": [
                    {"branch_label": "B1", "query_span": "如何打开设置菜单"}
                ],
                "evidence": [
                    {
                        "evidence_label": "E1",
                        "page": 1,
                        "text": "按菜单键可打开设置菜单。",
                    }
                ],
                "deterministic_candidate": {},
            },
            expected={
                "answerability": "answer",
                "concept_terms": ["菜单", "设置"],
                "numeric": None,
                "identifier": None,
                "polarity": "not_applicable",
                "evidence_labels": ["E1"],
                "branch_labels": ["B1"],
            },
        )
    )
    return cases


def _calibration_manifest() -> list[dict[str, Any]]:
    return [
        {"name": case.name, "payload": case.payload, "expected": case.expected}
        for case in _v4_calibration_cases()
    ]


def _expected_v5_calibration_ids() -> set[str]:
    """由冻结v5 profile digest与校准清单重建exact四个物理ID。"""

    return {
        f"em-nonthinking-{V5_PROFILE_SHA256[:16]}-{index}-{case.name}"
        for index, case in enumerate(_v4_calibration_cases(), start=1)
    }


def _default_predecessor_policy() -> dict[str, Any]:
    return {
        "profile_id": PREDECESSOR_PROFILE_ID,
        "profile_sha256": PREDECESSOR_PROFILE_SHA256,
        "completed_success_bridge_allowed": True,
        "completed_fresh_calibration_bridge_required": True,
        "failed_result_bridge_allowed": False,
        "bridge_receipt_sha256": None,
    }


def _pack_receipt(
    *, kind: str, pack: PackInput, expected_model: str
) -> tuple[dict[str, Any], set[str]]:
    """从实际冻结pack重建完整收据，避免信任持久化manifest自述。"""

    requests = list(pack.requests)
    selection = _selection(pack)
    if not requests:
        raise ProfileContractError(f"{kind} base pack is empty")
    custom_ids = [str(row.get("custom_id") or "") for row in requests]
    if len(set(custom_ids)) != len(custom_ids) or any(not value for value in custom_ids):
        raise ProfileContractError(f"{kind} base request ids are invalid")
    base_prompt_hashes: set[str] = set()
    for row in requests:
        errors = _base_request_errors(row, expected_model=expected_model)
        if errors:
            raise ProfileContractError(
                f"{kind} frozen base request invalid: {'; '.join(errors)}"
            )
        base_prompt_hashes.add(str(row.get("prompt_sha256") or ""))
    targets = list(selection.target_custom_ids)
    target_set = set(targets)
    base_targets = [deepcopy(row) for row in requests if row["custom_id"] in target_set]
    if [row["custom_id"] for row in base_targets] != targets:
        raise ProfileContractError(f"{kind} target order must match frozen request order")
    effective_targets = [_effective_from_base(row) for row in base_targets]
    effective_prompt_hashes = {
        str(row.get("prompt_sha256") or "") for row in effective_targets
    }
    if len(base_prompt_hashes) != 1 or len(effective_prompt_hashes) != 1:
        raise ProfileContractError(f"{kind} prompt hashes are not uniform")
    _require_sha256(
        pack.base_execution_contract_sha256,
        label=f"{kind} base execution contract",
    )
    receipt = {
        "full_case_count": len(requests),
        "target_case_count": len(targets),
        "target_custom_ids": targets,
        "target_sequence_by_custom_id": {
            custom_id: custom_ids.index(custom_id) + 1 for custom_id in targets
        },
        "target_difficulty_by_custom_id": dict(
            selection.target_difficulty_by_custom_id
        ),
        "target_difficulty_counts": dict(selection.target_difficulty_counts),
        "base_execution_contract_sha256": pack.base_execution_contract_sha256,
        "base_pack_sha256": sha256_json(requests),
        "request_map_sha256": sha256_json(list(pack.request_map)),
        "authority_rows_sha256": sha256_json(list(pack.authority_rows)),
        "scope_selection_receipt": selection.receipt(),
        "scope_selection_receipt_sha256": sha256_json(selection.receipt()),
        "base_target_pack_sha256": sha256_json(base_targets),
        "effective_target_pack_sha256": sha256_json(effective_targets),
        "base_prompt_sha256": next(iter(base_prompt_hashes)),
        "effective_prompt_sha256": next(iter(effective_prompt_hashes)),
        "base_request_sha256_by_id": {
            str(row["custom_id"]): sha256_json(row) for row in base_targets
        },
        "effective_request_sha256_by_id": {
            str(row["custom_id"]): sha256_json(effective)
            for row, effective in zip(base_targets, effective_targets, strict=True)
        },
    }
    return receipt, effective_prompt_hashes


def build_execution_profile(
    *,
    packs: Mapping[str, PackInput],
    scope_profile: Mapping[str, Any],
    model: str,
    endpoint: str,
    request_timeout: float,
    max_attempts: int,
    retry_delay: float,
    request_overlay: Mapping[str, Any] | None = None,
    predecessor_bridge_receipt: Mapping[str, Any] | None = None,
    v4_adoption_bridge_receipt: Mapping[str, Any] | None = None,
    v5_adoption_bridge_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从两个冻结 full pack 构造一个共享、独立版本的执行快照。"""

    overlay = deepcopy(dict(REQUEST_OVERLAY if request_overlay is None else request_overlay))
    if overlay != REQUEST_OVERLAY or set(overlay) != {"enable_thinking"}:
        raise ProfileContractError("request overlay must be exact enable_thinking=false")
    if set(packs) != set(KINDS):
        raise ProfileContractError("profile packs must be exact blocker+contract")
    if not isinstance(model, str) or not model:
        raise ProfileContractError("model must be non-empty")
    if not isinstance(endpoint, str) or not endpoint:
        raise ProfileContractError("endpoint must be non-empty")
    _validate_runtime_knobs(
        request_timeout=request_timeout,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )

    pack_receipts: dict[str, Any] = {}
    all_prompt_hashes: set[str] = set()
    for kind in KINDS:
        receipt, prompt_hashes = _pack_receipt(
            kind=kind,
            pack=packs[kind],
            expected_model=model,
        )
        pack_receipts[kind] = receipt
        all_prompt_hashes.update(prompt_hashes)
    if len(all_prompt_hashes) != 1:
        raise ProfileContractError("both packs must share one frozen prompt SHA")
    prompt_sha256 = _require_sha256(next(iter(all_prompt_hashes)), label="prompt")
    base_prompt_hashes = {
        str(pack_receipts[kind]["base_prompt_sha256"]) for kind in KINDS
    }
    if len(base_prompt_hashes) != 1:
        raise ProfileContractError("both packs must share one frozen base prompt SHA")
    base_prompt_sha256 = _require_sha256(
        next(iter(base_prompt_hashes)), label="base prompt"
    )
    scope = _scope_contract(scope_profile)
    predecessor_policy = _default_predecessor_policy()
    if predecessor_bridge_receipt is not None:
        receipt = deepcopy(dict(predecessor_bridge_receipt))
        claimed = receipt.get("receipt_sha256")
        if claimed != sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        ):
            raise ProfileContractError("predecessor bridge receipt self-hash drifted")
        predecessor_policy["bridge_receipt_sha256"] = claimed
    v4_adoption_policy = {
        "policy": V4_ADOPTION_POLICY,
        "source_profile_id": V4_PROFILE_ID,
        "source_profile_sha256": V4_PROFILE_SHA256,
        "source_materializer_policy_version": LEGACY_MATERIALIZATION_POLICY_VERSION,
        "source_production_matcher_policy_version": (
            LEGACY_PRODUCTION_MATCHER_POLICY_VERSION
        ),
        "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
        "production_matcher": PRODUCTION_MATCHER,
        "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "bridge_receipt_sha256": None,
        "source_calibration_evidence_only": True,
        "v4_calibration_authorizes_v5_calls": False,
    }
    if v4_adoption_bridge_receipt is not None:
        receipt = deepcopy(dict(v4_adoption_bridge_receipt))
        claimed = receipt.get("receipt_sha256")
        if claimed != sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        ):
            raise ProfileContractError("v4 adoption bridge receipt self-hash drifted")
        v4_adoption_policy["bridge_receipt_sha256"] = claimed
    v5_adoption_policy = {
        "policy": V5_ADOPTION_POLICY,
        "source_profile_id": V5_PROFILE_ID,
        "source_profile_sha256": V5_PROFILE_SHA256,
        "source_materializer_policy_version": V5_MATERIALIZATION_POLICY_VERSION,
        "source_production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
        "production_matcher": PRODUCTION_MATCHER,
        "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "bridge_receipt_sha256": None,
        "source_calibration_evidence_only": True,
        "v5_calibration_authorizes_v6_calls": False,
    }
    if v5_adoption_bridge_receipt is not None:
        receipt = deepcopy(dict(v5_adoption_bridge_receipt))
        claimed = receipt.get("receipt_sha256")
        if claimed != sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        ):
            raise ProfileContractError("v5 adoption bridge receipt self-hash drifted")
        v5_adoption_policy["bridge_receipt_sha256"] = claimed
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "runner_version": PROFILE_RUNNER_VERSION,
        "legacy_runner_version": LEGACY_RUNNER_VERSION,
        "legacy_contracts_preserved": True,
        "scope_profile": scope,
        "scope_profile_sha256": sha256_json(scope),
        "model": model,
        "endpoint": endpoint,
        "transport": TRANSPORT,
        "sse_completion_policy": SSE_COMPLETION_POLICY,
        "prompt_sha256": prompt_sha256,
        "base_prompt_sha256": base_prompt_sha256,
        "prompt_schema_policy": {
            "appendix": PROMPT_SCHEMA_APPENDIX,
            "polarity_required_values": list(POLARITY_REQUIRED_VALUES),
            "required_json_type": "string",
            "polarity_object_exact_keys": ["required", "cues"],
            "cues_json_type": "array[string]",
            "empty_cues_required_when_not_applicable": True,
            "neutral_fact_policy": NEUTRAL_POLARITY_POLICY,
        },
        "request_overlay": overlay,
        "stream_request_config": {
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        "runtime_knobs": {
            "request_timeout_seconds": float(request_timeout),
            "max_attempts_per_failure_category": max_attempts,
            "retry_delay_seconds": float(retry_delay),
        },
        "base_packs": pack_receipts,
        "combined_base_pack_sha256": sha256_json(
            {kind: pack_receipts[kind]["base_pack_sha256"] for kind in KINDS}
        ),
        "combined_effective_target_pack_sha256": sha256_json(
            {
                kind: pack_receipts[kind]["effective_target_pack_sha256"]
                for kind in KINDS
            }
        ),
        "calibration_policy": {
            "required_case_count": 4,
            "legacy_calibration_reuse_allowed": False,
            "fresh_profile_ids_required": True,
            "v4_calibration_reuse_allowed": False,
            "v5_calibration_reuse_allowed": False,
            "calibration_cases_manifest_sha256": sha256_json(_calibration_manifest()),
            "not_applicable_case_required": True,
            "calibration_projection_policy": (
                legacy.CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY
            ),
        },
        "side_effect_policy": {
            "started_fence_required": True,
            "started_without_result": "fail_stop_unknown_side_effect",
            "result_without_completed": "deterministic_reconcile_no_api",
            "duplicate_provider_operation_allowed": False,
        },
        "core_result_policy": {
            "production_execution_contract_sha256": "execution_profile_sha256",
            "sequence_basis": "full_frozen_pack_1_based",
            "legacy_core_runner_version_preserved": True,
        },
        "materialization_policy": {
            "policy_version": MATERIALIZATION_POLICY_VERSION,
            "production_matcher": PRODUCTION_MATCHER,
            "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
            "subset_solver_policy": dict(SUBSET_SOLVER_POLICY),
        },
        "predecessor_policy": predecessor_policy,
        "v4_adoption_policy": v4_adoption_policy,
        "v5_adoption_policy": v5_adoption_policy,
        "promotion_policy": {
            "promotable": False,
            "baseline_registration_allowed": False,
            "canonical_promotion_allowed": False,
        },
    }
    manifest["execution_profile_sha256"] = sha256_json(manifest)
    return manifest


def validate_execution_profile(profile: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "profile_id",
        "runner_version",
        "legacy_runner_version",
        "legacy_contracts_preserved",
        "scope_profile",
        "scope_profile_sha256",
        "model",
        "endpoint",
        "transport",
        "sse_completion_policy",
        "prompt_sha256",
        "base_prompt_sha256",
        "prompt_schema_policy",
        "request_overlay",
        "stream_request_config",
        "runtime_knobs",
        "base_packs",
        "combined_base_pack_sha256",
        "combined_effective_target_pack_sha256",
        "calibration_policy",
        "side_effect_policy",
        "core_result_policy",
        "materialization_policy",
        "predecessor_policy",
        "v4_adoption_policy",
        "v5_adoption_policy",
        "promotion_policy",
        "execution_profile_sha256",
    }
    if set(profile) != expected_keys:
        raise ProfileContractError("execution profile schema is not exact")
    if (
        profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or profile.get("profile_id") != PROFILE_ID
        or profile.get("runner_version") != PROFILE_RUNNER_VERSION
        or profile.get("legacy_runner_version") != LEGACY_RUNNER_VERSION
        or profile.get("legacy_contracts_preserved") is not True
        or profile.get("transport") != TRANSPORT
        or profile.get("sse_completion_policy") != SSE_COMPLETION_POLICY
        or profile.get("request_overlay") != REQUEST_OVERLAY
        or profile.get("prompt_schema_policy")
        != {
            "appendix": PROMPT_SCHEMA_APPENDIX,
            "polarity_required_values": list(POLARITY_REQUIRED_VALUES),
            "required_json_type": "string",
            "polarity_object_exact_keys": ["required", "cues"],
            "cues_json_type": "array[string]",
            "empty_cues_required_when_not_applicable": True,
            "neutral_fact_policy": NEUTRAL_POLARITY_POLICY,
        }
        or profile.get("stream_request_config")
        != {"stream": True, "stream_options": {"include_usage": True}}
        or profile.get("calibration_policy")
        != {
            "required_case_count": 4,
            "legacy_calibration_reuse_allowed": False,
            "fresh_profile_ids_required": True,
            "v4_calibration_reuse_allowed": False,
            "v5_calibration_reuse_allowed": False,
            "calibration_cases_manifest_sha256": sha256_json(_calibration_manifest()),
            "not_applicable_case_required": True,
            "calibration_projection_policy": (
                legacy.CALIBRATION_ABSTAIN_SENTINEL_PROJECTION_POLICY
            ),
        }
        or profile.get("side_effect_policy")
        != {
            "started_fence_required": True,
            "started_without_result": "fail_stop_unknown_side_effect",
            "result_without_completed": "deterministic_reconcile_no_api",
            "duplicate_provider_operation_allowed": False,
        }
        or profile.get("core_result_policy")
        != {
            "production_execution_contract_sha256": "execution_profile_sha256",
            "sequence_basis": "full_frozen_pack_1_based",
            "legacy_core_runner_version_preserved": True,
        }
        or profile.get("materialization_policy")
        != {
            "policy_version": MATERIALIZATION_POLICY_VERSION,
            "production_matcher": PRODUCTION_MATCHER,
            "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
            "subset_solver_policy": dict(SUBSET_SOLVER_POLICY),
        }
        or not isinstance(profile.get("v4_adoption_policy"), Mapping)
        or {
            key: value
            for key, value in profile["v4_adoption_policy"].items()
            if key != "bridge_receipt_sha256"
        }
        != {
            "policy": V4_ADOPTION_POLICY,
            "source_profile_id": V4_PROFILE_ID,
            "source_profile_sha256": V4_PROFILE_SHA256,
            "source_materializer_policy_version": (
                LEGACY_MATERIALIZATION_POLICY_VERSION
            ),
            "source_production_matcher_policy_version": (
                LEGACY_PRODUCTION_MATCHER_POLICY_VERSION
            ),
            "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
            "production_matcher": PRODUCTION_MATCHER,
            "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
            "source_calibration_evidence_only": True,
            "v4_calibration_authorizes_v5_calls": False,
        }
        or not isinstance(profile.get("predecessor_policy"), Mapping)
        or not isinstance(profile.get("v5_adoption_policy"), Mapping)
        or {
            key: value
            for key, value in profile["v5_adoption_policy"].items()
            if key != "bridge_receipt_sha256"
        }
        != {
            "policy": V5_ADOPTION_POLICY,
            "source_profile_id": V5_PROFILE_ID,
            "source_profile_sha256": V5_PROFILE_SHA256,
            "source_materializer_policy_version": V5_MATERIALIZATION_POLICY_VERSION,
            "source_production_matcher_policy_version": (
                PRODUCTION_MATCHER_POLICY_VERSION
            ),
            "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
            "production_matcher": PRODUCTION_MATCHER,
            "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
            "source_calibration_evidence_only": True,
            "v5_calibration_authorizes_v6_calls": False,
        }
        or not isinstance(profile.get("predecessor_policy"), Mapping)
        or {
            key: value
            for key, value in profile["predecessor_policy"].items()
            if key != "bridge_receipt_sha256"
        }
        != {
            key: value
            for key, value in _default_predecessor_policy().items()
            if key != "bridge_receipt_sha256"
        }
        or profile.get("promotion_policy")
        != {
            "promotable": False,
            "baseline_registration_allowed": False,
            "canonical_promotion_allowed": False,
        }
    ):
        raise ProfileContractError("execution profile frozen fields drifted")
    _scope_contract(profile.get("scope_profile") or {})
    if profile.get("scope_profile_sha256") != sha256_json(profile["scope_profile"]):
        raise ProfileContractError("scope profile hash drifted")
    if not isinstance(profile.get("model"), str) or not profile.get("model"):
        raise ProfileContractError("execution profile model is invalid")
    if not isinstance(profile.get("endpoint"), str) or not profile.get("endpoint"):
        raise ProfileContractError("execution profile endpoint is invalid")
    _require_sha256(profile.get("prompt_sha256"), label="execution profile prompt")
    _require_sha256(profile.get("base_prompt_sha256"), label="execution profile base prompt")
    bridge_receipt_sha = profile["predecessor_policy"].get("bridge_receipt_sha256")
    if bridge_receipt_sha is not None:
        _require_sha256(bridge_receipt_sha, label="predecessor bridge receipt")
    v4_bridge_receipt_sha = profile["v4_adoption_policy"].get(
        "bridge_receipt_sha256"
    )
    if v4_bridge_receipt_sha is not None:
        _require_sha256(v4_bridge_receipt_sha, label="v4 adoption bridge receipt")
    v5_bridge_receipt_sha = profile["v5_adoption_policy"].get(
        "bridge_receipt_sha256"
    )
    if v5_bridge_receipt_sha is not None:
        _require_sha256(v5_bridge_receipt_sha, label="v5 adoption bridge receipt")
    knobs = profile.get("runtime_knobs")
    if not isinstance(knobs, Mapping) or set(knobs) != {
        "request_timeout_seconds",
        "max_attempts_per_failure_category",
        "retry_delay_seconds",
    }:
        raise ProfileContractError("execution profile runtime knobs schema is not exact")
    _validate_runtime_knobs(
        request_timeout=knobs["request_timeout_seconds"],
        max_attempts=knobs["max_attempts_per_failure_category"],
        retry_delay=knobs["retry_delay_seconds"],
    )
    pack_receipts = profile.get("base_packs")
    receipt_keys = {
        "full_case_count",
        "target_case_count",
        "target_custom_ids",
        "target_sequence_by_custom_id",
        "target_difficulty_by_custom_id",
        "target_difficulty_counts",
        "base_execution_contract_sha256",
        "base_pack_sha256",
        "request_map_sha256",
        "authority_rows_sha256",
        "scope_selection_receipt",
        "scope_selection_receipt_sha256",
        "base_target_pack_sha256",
        "effective_target_pack_sha256",
        "base_prompt_sha256",
        "effective_prompt_sha256",
        "base_request_sha256_by_id",
        "effective_request_sha256_by_id",
    }
    if not isinstance(pack_receipts, Mapping) or set(pack_receipts) != set(KINDS):
        raise ProfileContractError("execution profile base packs schema is not exact")
    for kind in KINDS:
        receipt = pack_receipts[kind]
        if not isinstance(receipt, Mapping) or set(receipt) != receipt_keys:
            raise ProfileContractError(f"{kind} execution profile receipt schema is not exact")
        targets = receipt.get("target_custom_ids")
        sequences = receipt.get("target_sequence_by_custom_id")
        difficulties = receipt.get("target_difficulty_by_custom_id")
        base_hashes = receipt.get("base_request_sha256_by_id")
        effective_hashes = receipt.get("effective_request_sha256_by_id")
        if (
            not isinstance(targets, list)
            or not targets
            or len(set(targets)) != len(targets)
            or not all(isinstance(value, str) and value for value in targets)
            or not all(isinstance(value, Mapping) for value in [sequences, difficulties, base_hashes, effective_hashes])
            or set(sequences) != set(targets)
            or set(difficulties) != set(targets)
            or set(base_hashes) != set(targets)
            or set(effective_hashes) != set(targets)
            or set(difficulties.values()) - set(scope.SCOPE_DIFFICULTIES)
            or receipt.get("target_case_count") != len(targets)
            or type(receipt.get("full_case_count")) is not int
            or receipt["full_case_count"] < len(targets)
        ):
            raise ProfileContractError(f"{kind} execution profile target receipt is invalid")
        for label in (
            "base_execution_contract_sha256",
            "base_pack_sha256",
            "request_map_sha256",
            "authority_rows_sha256",
            "scope_selection_receipt_sha256",
            "base_target_pack_sha256",
            "effective_target_pack_sha256",
            "base_prompt_sha256",
            "effective_prompt_sha256",
        ):
            _require_sha256(receipt.get(label), label=f"{kind} {label}")
        if receipt.get("scope_selection_receipt_sha256") != sha256_json(
            receipt.get("scope_selection_receipt")
        ):
            raise ProfileContractError(f"{kind} scope selection receipt hash drifted")
        selection_receipt = receipt["scope_selection_receipt"]
        if (
            not isinstance(selection_receipt, Mapping)
            or selection_receipt.get("scope_target_custom_ids") != targets
            or selection_receipt.get("scope_case_expected_count") != len(targets)
            or selection_receipt.get("full_case_expected_count")
            != receipt.get("full_case_count")
            or selection_receipt.get("scope_target_difficulty_counts")
            != receipt.get("target_difficulty_counts")
            or selection_receipt.get("full_request_pack_sha256")
            != receipt.get("base_pack_sha256")
            or selection_receipt.get("scope_target_request_pack_sha256")
            != receipt.get("base_target_pack_sha256")
            or selection_receipt.get("request_map_sha256")
            != receipt.get("request_map_sha256")
        ):
            raise ProfileContractError(f"{kind} target receipt cross-binding drifted")
        if (
            receipt.get("base_prompt_sha256") != profile.get("base_prompt_sha256")
            or receipt.get("effective_prompt_sha256") != profile.get("prompt_sha256")
        ):
            raise ProfileContractError(f"{kind} prompt receipt cross-binding drifted")
        if any(type(value) is not int or value < 1 for value in sequences.values()):
            raise ProfileContractError(f"{kind} target sequence receipt is invalid")
        for label, values in (("base request", base_hashes), ("effective request", effective_hashes)):
            for value in values.values():
                _require_sha256(value, label=f"{kind} {label}")
    if profile.get("combined_base_pack_sha256") != sha256_json(
        {kind: pack_receipts[kind]["base_pack_sha256"] for kind in KINDS}
    ):
        raise ProfileContractError("combined base pack hash drifted")
    if profile.get("combined_effective_target_pack_sha256") != sha256_json(
        {
            kind: pack_receipts[kind]["effective_target_pack_sha256"]
            for kind in KINDS
        }
    ):
        raise ProfileContractError("combined effective target pack hash drifted")
    digest_view = deepcopy(dict(profile))
    observed = digest_view.pop("execution_profile_sha256", None)
    if observed != sha256_json(digest_view):
        raise ProfileContractError("execution profile hash drifted")


def validate_profile_against_pack(
    execution_profile: Mapping[str, Any], *, kind: str, pack: PackInput
) -> None:
    """用实际pack重建named receipt，拒绝co-rehash后的伪造manifest。"""

    validate_execution_profile(execution_profile)
    if kind not in KINDS:
        raise ProfileContractError("runtime pack kind is invalid")
    actual, prompt_hashes = _pack_receipt(
        kind=kind,
        pack=pack,
        expected_model=str(execution_profile["model"]),
    )
    if actual != execution_profile["base_packs"][kind]:
        raise ProfileContractError(f"{kind} actual pack/profile receipt mismatch")
    if prompt_hashes != {execution_profile["prompt_sha256"]}:
        raise ProfileContractError(f"{kind} actual pack prompt/profile mismatch")


def _base_request_by_id(
    pack: PackInput, custom_id: str
) -> dict[str, Any]:
    rows = [row for row in pack.requests if row.get("custom_id") == custom_id]
    if len(rows) != 1:
        raise ProfileContractError("base request is not present exactly once")
    return rows[0]


def effective_request_row(
    row: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    """校验 base row 身份后返回不修改原对象的 effective row。"""

    validate_execution_profile(execution_profile)
    if kind not in KINDS:
        raise ProfileContractError("effective request kind is invalid")
    base_errors = _base_request_errors(
        row, expected_model=str(execution_profile["model"])
    )
    if base_errors:
        raise ProfileContractError("; ".join(base_errors))
    custom_id = str(row.get("custom_id") or "")
    receipt = execution_profile["base_packs"][kind]
    expected_base_hash = (receipt.get("base_request_sha256_by_id") or {}).get(custom_id)
    if expected_base_hash is None or sha256_json(row) != expected_base_hash:
        raise ProfileContractError("base request identity drifted")
    effective = _effective_from_base(row)
    expected_effective_hash = (receipt.get("effective_request_sha256_by_id") or {}).get(
        custom_id
    )
    if sha256_json(effective) != expected_effective_hash:
        raise ProfileContractError("effective request identity drifted")
    return effective


def build_calibration_requests(
    template: Mapping[str, Any], execution_profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """生成与新 profile 绑定的四个 fresh calibration request。"""

    validate_execution_profile(execution_profile)
    template_effective = _effective_from_base(template)
    digest = str(execution_profile["execution_profile_sha256"])
    calibration_contract_sha256 = sha256_json(
        {
            "profile_sha256": digest,
            "model": execution_profile["model"],
            "endpoint": execution_profile["endpoint"],
            "transport": TRANSPORT,
            "sse_completion_policy": SSE_COMPLETION_POLICY,
            "prompt_sha256": execution_profile["prompt_sha256"],
            "effective_request_config": {
                key: deepcopy(value)
                for key, value in template_effective["request"].items()
                if key not in {"messages", "model"}
            },
            "calibration_cases_manifest_sha256": sha256_json(
                _calibration_manifest()
            ),
        }
    )
    result: list[dict[str, Any]] = []
    system_message = deepcopy(template_effective["request"]["messages"][0])
    for index, case in enumerate(_v4_calibration_cases(), start=1):
        fresh = {
            "custom_id": f"em-nonthinking-{digest[:16]}-{index}-{case.name}",
            "prompt_sha256": execution_profile["prompt_sha256"],
            "request": {
                **deepcopy(template_effective["request"]),
                "messages": [
                    system_message,
                    {
                        "role": "user",
                        "content": json.dumps(
                            case.payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
            "expected": deepcopy(case.expected),
            "calibration_execution_contract_sha256": calibration_contract_sha256,
            "calibration_model": execution_profile["model"],
            "calibration_endpoint": execution_profile["endpoint"],
            "easy_medium_execution_profile_sha256": digest,
        }
        result.append(fresh)
    if len(result) != 4 or len({row["custom_id"] for row in result}) != 4:
        raise ProfileContractError("fresh calibration pack is not exact 4/4")
    return result


def bind_production_core_result(
    core_result: Mapping[str, Any],
    *,
    kind: str,
    logical_custom_id: str,
    execution_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """给新profile的production core补独立合同与冻结序号，不写旧账本。"""

    validate_execution_profile(execution_profile)
    if kind not in KINDS:
        raise ProfileContractError("production core result kind is invalid")
    if "execution_contract_sha256" in core_result or "sequence" in core_result:
        raise ProfileContractError("raw production core result already declares a contract")
    bound = deepcopy(dict(core_result))
    bound["execution_contract_sha256"] = execution_profile[
        "execution_profile_sha256"
    ]
    bound["sequence"] = _expected_sequence(
        kind=kind,
        logical_custom_id=logical_custom_id,
        execution_profile=execution_profile,
    )
    return bound


def execute_profile_request(
    effective_row: Mapping[str, Any],
    *,
    kind: str,
    calibration: bool,
    execution_profile: Mapping[str, Any],
    api_key: str,
    execute_one: Callable[..., dict[str, Any]] = legacy._execute_one,
) -> dict[str, Any]:
    """production adapter：把exact effective body传入现有SSE执行器。"""

    validate_execution_profile(execution_profile)
    row = deepcopy(dict(effective_row))
    request = row.get("request")
    if not isinstance(request, Mapping) or request.get("enable_thinking") is not False:
        raise ProfileContractError("provider request must contain top-level enable_thinking=false")
    if request.get("model") != execution_profile.get("model"):
        raise ProfileContractError("provider request model/profile mismatch")
    if calibration:
        if (
            kind != "calibration"
            or row.get("easy_medium_execution_profile_sha256")
            != execution_profile.get("execution_profile_sha256")
            or row.get("calibration_model") != execution_profile.get("model")
            or row.get("calibration_endpoint") != execution_profile.get("endpoint")
        ):
            raise ProfileContractError("calibration request/profile binding mismatch")
    else:
        if kind not in KINDS:
            raise ProfileContractError("provider production kind is invalid")
        base = _base_from_effective(row)
        expected = effective_request_row(base, execution_profile, kind=kind)
        if row != expected:
            raise ProfileContractError("provider effective request replay mismatch")
    knobs = execution_profile["runtime_knobs"]
    core = execute_one(
        row=row,
        endpoint=str(execution_profile["endpoint"]),
        api_key=api_key,
        model=str(execution_profile["model"]),
        timeout=float(knobs["request_timeout_seconds"]),
        max_attempts=int(knobs["max_attempts_per_failure_category"]),
        retry_delay=float(knobs["retry_delay_seconds"]),
        calibration=calibration,
        extra_review_validator=(
            None
            if calibration
            else lambda review, payload: legacy._review_materialization_errors(
                review,
                payload,
                materialization_policy_version=MATERIALIZATION_POLICY_VERSION,
            )
        ),
    )
    if calibration:
        return core
    return bind_production_core_result(
        core,
        kind=kind,
        logical_custom_id=str(row["custom_id"]),
        execution_profile=execution_profile,
    )


def _request_hashes(effective_row: Mapping[str, Any]) -> tuple[str, str]:
    base = _base_from_effective(effective_row)
    return sha256_json(base), sha256_json(effective_row)


def _operation_id(
    *, kind: str, logical_custom_id: str, effective_request_sha256: str, profile_sha256: str
) -> str:
    return "em-review-" + sha256_json(
        {
            "kind": kind,
            "logical_custom_id": logical_custom_id,
            "effective_request_sha256": effective_request_sha256,
            "execution_profile_sha256": profile_sha256,
        }
    )[:32]


def _parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise ProfileContractError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(None):
        raise ProfileContractError(f"{label} timestamp must be UTC")
    return parsed


def _expected_sequence(
    *, kind: str, logical_custom_id: str, execution_profile: Mapping[str, Any]
) -> int:
    if kind in KINDS:
        value = execution_profile["base_packs"][kind][
            "target_sequence_by_custom_id"
        ].get(logical_custom_id)
        if type(value) is not int:
            raise ProfileContractError("target sequence is not bound by the frozen pack")
        return value
    prefix = f"em-nonthinking-{execution_profile['execution_profile_sha256'][:16]}-"
    if not logical_custom_id.startswith(prefix):
        raise ProfileContractError("calibration id is not bound to this profile")
    remainder = logical_custom_id[len(prefix) :]
    sequence_text, separator, _suffix = remainder.partition("-")
    if not separator or not sequence_text.isdigit() or not 1 <= int(sequence_text) <= 4:
        raise ProfileContractError("calibration sequence is invalid")
    return int(sequence_text)


def build_started_event(
    *,
    kind: str,
    logical_custom_id: str,
    effective_row: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    started_at: str,
    allow_non_target_for_test: bool = False,
) -> dict[str, Any]:
    validate_execution_profile(execution_profile)
    _parse_utc_timestamp(started_at, label="started")
    if kind not in ALL_EVENT_KINDS:
        raise ProfileContractError("started event kind is invalid")
    if effective_row.get("custom_id") != logical_custom_id:
        raise ProfileContractError("started event logical id/request mismatch")
    if kind in KINDS:
        targets = execution_profile["base_packs"][kind]["target_custom_ids"]
        if logical_custom_id not in targets and not allow_non_target_for_test:
            raise ProfileContractError("started event targets a non-target E/M id")
    base_sha, effective_sha = _request_hashes(effective_row)
    base_contract = (
        execution_profile["base_packs"][kind]["base_execution_contract_sha256"]
        if kind in KINDS
        else None
    )
    profile_sha = str(execution_profile["execution_profile_sha256"])
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_type": "started",
        "kind": kind,
        "logical_custom_id": logical_custom_id,
        "sequence": _expected_sequence(
            kind=kind,
            logical_custom_id=logical_custom_id,
            execution_profile=execution_profile,
        ),
        "operation_id": _operation_id(
            kind=kind,
            logical_custom_id=logical_custom_id,
            effective_request_sha256=effective_sha,
            profile_sha256=profile_sha,
        ),
        "execution_profile_sha256": profile_sha,
        "base_execution_contract_sha256": base_contract,
        "base_request_sha256": base_sha,
        "effective_request_sha256": effective_sha,
        "started_at": started_at,
        "side_effect_status": "unknown_until_terminal_result",
    }


def wrap_result(
    *,
    kind: str,
    logical_custom_id: str,
    effective_row: Mapping[str, Any],
    core_result: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    started_event: Mapping[str, Any],
) -> dict[str, Any]:
    """把未修改的 legacy core result 封装进独立 profile receipt。"""

    validate_execution_profile(execution_profile)
    base_sha, effective_sha = _request_hashes(effective_row)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "event_type": "result",
        "kind": kind,
        "logical_custom_id": logical_custom_id,
        "sequence": started_event.get("sequence"),
        "operation_id": started_event.get("operation_id"),
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "profile_runner_version": PROFILE_RUNNER_VERSION,
        "model": execution_profile["model"],
        "endpoint": execution_profile["endpoint"],
        "transport": TRANSPORT,
        "sse_completion_policy": SSE_COMPLETION_POLICY,
        "request_overlay": deepcopy(REQUEST_OVERLAY),
        "base_execution_contract_sha256": started_event.get(
            "base_execution_contract_sha256"
        ),
        "base_request_sha256": base_sha,
        "effective_request_sha256": effective_sha,
        "started_event_sha256": sha256_json(started_event),
        "core_result_sha256": sha256_json(core_result),
        "core_result": deepcopy(dict(core_result)),
    }


def validate_result_envelope(
    envelope: Mapping[str, Any],
    *,
    kind: str,
    request_row: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    started_event: Mapping[str, Any],
    validate_core_result: Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]],
) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema_version",
        "event_type",
        "kind",
        "logical_custom_id",
        "sequence",
        "operation_id",
        "execution_profile_sha256",
        "profile_runner_version",
        "model",
        "endpoint",
        "transport",
        "sse_completion_policy",
        "request_overlay",
        "base_execution_contract_sha256",
        "base_request_sha256",
        "effective_request_sha256",
        "started_event_sha256",
        "core_result_sha256",
        "core_result",
    }
    if set(envelope) != expected_keys:
        return ["result envelope schema is not exact"]
    core_result = envelope.get("core_result")
    if not isinstance(core_result, Mapping):
        return ["result envelope core result is invalid"]
    expected = wrap_result(
        kind=kind,
        logical_custom_id=str(envelope.get("logical_custom_id") or ""),
        effective_row=request_row,
        core_result=core_result,
        execution_profile=execution_profile,
        started_event=started_event,
    )
    if dict(envelope) != expected:
        errors.append("result envelope replay mismatch")
    calibration = kind == "calibration"
    status = core_result.get("status")
    allowed_statuses = RESULT_STATUSES if calibration else {"succeeded", "failed"}
    if (
        core_result.get("custom_id") != envelope.get("logical_custom_id")
        or core_result.get("runner_version") != LEGACY_RUNNER_VERSION
        or core_result.get("transport") != TRANSPORT
        or core_result.get("sse_completion_policy") != SSE_COMPLETION_POLICY
        or core_result.get("model") != execution_profile.get("model")
        or core_result.get("prompt_sha256") != execution_profile.get("prompt_sha256")
        or status not in allowed_statuses
    ):
        errors.append("core result execution envelope mismatch")
    if not calibration and (
        core_result.get("execution_contract_sha256")
        != execution_profile.get("execution_profile_sha256")
        or core_result.get("sequence") != envelope.get("sequence")
    ):
        errors.append("production core result profile contract/sequence mismatch")
    try:
        errors.extend(
            str(error)
            for error in validate_core_result(
                deepcopy(dict(core_result)), deepcopy(dict(request_row)), calibration
            )
            if str(error)
        )
    except Exception as exc:  # pragma: no cover - fail-closed adapter boundary
        errors.append(f"core result validator raised {type(exc).__name__}")
    return errors


def build_completed_event(
    *, started_event: Mapping[str, Any], result_envelope: Mapping[str, Any]
) -> dict[str, Any]:
    core = result_envelope.get("core_result") or {}
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_type": "completed",
        "kind": started_event.get("kind"),
        "logical_custom_id": started_event.get("logical_custom_id"),
        "sequence": started_event.get("sequence"),
        "operation_id": started_event.get("operation_id"),
        "execution_profile_sha256": started_event.get("execution_profile_sha256"),
        "started_event_sha256": sha256_json(started_event),
        "result_envelope_sha256": sha256_json(result_envelope),
        "result_status": core.get("status"),
        "completed_at": core.get("completed_at"),
        "side_effect_status": "complete",
    }


def _validate_started(
    event: Mapping[str, Any],
    *,
    kind: str,
    effective_row: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
) -> list[str]:
    expected = build_started_event(
        kind=kind,
        logical_custom_id=str(event.get("logical_custom_id") or ""),
        effective_row=effective_row,
        execution_profile=execution_profile,
        started_at=str(event.get("started_at") or ""),
        allow_non_target_for_test=True,
    )
    return [] if dict(event) == expected else ["started event replay mismatch"]


def _validate_completed(
    event: Mapping[str, Any],
    *,
    started: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    expected = build_completed_event(started_event=started, result_envelope=result)
    if dict(event) != expected:
        return ["completed event replay mismatch"]
    try:
        started_at = _parse_utc_timestamp(started.get("started_at"), label="started")
        completed_at = _parse_utc_timestamp(event.get("completed_at"), label="completed")
    except ProfileContractError as exc:
        return [str(exc)]
    if completed_at < started_at:
        return ["completed timestamp precedes started fence"]
    return []


def _project_authoritative_result(
    envelope: Mapping[str, Any], execution_profile: Mapping[str, Any]
) -> dict[str, Any]:
    projected = deepcopy(dict(envelope["core_result"]))
    projected["runner_version"] = PROFILE_RUNNER_VERSION
    projected["execution_profile_id"] = PROFILE_ID
    projected["execution_profile_sha256"] = execution_profile[
        "execution_profile_sha256"
    ]
    projected["base_execution_contract_sha256"] = envelope[
        "base_execution_contract_sha256"
    ]
    projected["effective_request_sha256"] = envelope["effective_request_sha256"]
    projected["result_envelope_sha256"] = sha256_json(envelope)
    return projected


def _project_predecessor_success(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """从已验证的v1 result envelope确定性重建bridge authoritative row。"""

    core = envelope.get("core_result")
    if not isinstance(core, Mapping) or core.get("status") != "succeeded":
        raise ProfileContractError("predecessor source is not a succeeded core")
    projected = deepcopy(dict(core))
    projected["runner_version"] = PREDECESSOR_RUNNER_VERSION
    projected["execution_profile_id"] = PREDECESSOR_PROFILE_ID
    projected["execution_profile_sha256"] = PREDECESSOR_PROFILE_SHA256
    projected["base_execution_contract_sha256"] = envelope.get(
        "base_execution_contract_sha256"
    )
    projected["effective_request_sha256"] = envelope.get(
        "effective_request_sha256"
    )
    projected["result_envelope_sha256"] = sha256_json(envelope)
    return projected


def _project_v4_adopted_success(
    envelope: Mapping[str, Any],
    *,
    started_event: Mapping[str, Any],
    completed_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把v4已付费core原样投影为带明确零API lineage的v5权威行。"""

    core = envelope.get("core_result")
    if not isinstance(core, Mapping) or core.get("status") != "succeeded":
        raise ProfileContractError("v4 adoption source is not a succeeded core")
    projected = deepcopy(dict(core))
    projected["execution_profile_id"] = V4_PROFILE_ID
    projected["execution_profile_sha256"] = V4_PROFILE_SHA256
    projected["base_execution_contract_sha256"] = envelope.get(
        "base_execution_contract_sha256"
    )
    projected["effective_request_sha256"] = envelope.get(
        "effective_request_sha256"
    )
    projected["result_envelope_sha256"] = sha256_json(envelope)
    projected["v5_materializer_adoption"] = {
        "policy": V4_ADOPTION_POLICY,
        "api_call_performed": False,
        "source_started_event_sha256": sha256_json(started_event),
        "source_completed_event_sha256": (
            sha256_json(completed_event) if completed_event is not None else None
        ),
        "source_completion_state": (
            "completed" if completed_event is not None else "result_without_completed"
        ),
        "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
        "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
    }
    return projected


def _project_v5_adopted_success(
    envelope: Mapping[str, Any], *, started_event: Mapping[str, Any]
) -> dict[str, Any]:
    """把v5 result-only确定性投影为v6零API权威行。"""

    core = envelope.get("core_result")
    if not isinstance(core, Mapping) or core.get("status") != "succeeded":
        raise ProfileContractError("v5 adoption source is not a succeeded core")
    projected = deepcopy(dict(core))
    projected.update(
        {
            "execution_profile_id": V5_PROFILE_ID,
            "execution_profile_sha256": V5_PROFILE_SHA256,
            "base_execution_contract_sha256": envelope.get(
                "base_execution_contract_sha256"
            ),
            "effective_request_sha256": envelope.get("effective_request_sha256"),
            "result_envelope_sha256": sha256_json(envelope),
            "v6_materializer_adoption": {
                "policy": V5_ADOPTION_POLICY,
                "api_call_performed": False,
                "source_started_event_sha256": sha256_json(started_event),
                "source_completed_event_sha256": None,
                "source_completion_state": "result_without_completed",
                "source_materializer_policy_version": (
                    V5_MATERIALIZATION_POLICY_VERSION
                ),
                "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
                "production_matcher_policy_version": (
                    PRODUCTION_MATCHER_POLICY_VERSION
                ),
            },
        }
    )
    return projected


def v5_adoption_bridge_receipt(
    *,
    source_profile: Mapping[str, Any],
    source_envelope: Mapping[str, Any],
    source_started_event: Mapping[str, Any],
    calibration_envelopes: Sequence[Mapping[str, Any]],
    calibration_base_requests: Sequence[Mapping[str, Any]],
    calibration_requests: Sequence[Mapping[str, Any]],
    calibration_started_events: Sequence[Mapping[str, Any]],
    calibration_completed_events: Sequence[Mapping[str, Any]],
    physical_sidecar_sha256: Mapping[str, str],
    old_materialization_errors: Sequence[str],
    new_materialization_errors: Sequence[str],
) -> dict[str, Any]:
    """构造绑定v5完整物理sidecar及result-only采用的自散列收据。"""

    source_contract_profile = (
        (source_profile.get("base_packs") or {}).get("contract") or {}
    )
    if (
        source_profile.get("profile_id") != V5_PROFILE_ID
        or source_profile.get("runner_version") != V5_PROFILE_RUNNER_VERSION
        or source_profile.get("execution_profile_sha256") != V5_PROFILE_SHA256
        or sha256_json(
            {
                key: value
                for key, value in source_profile.items()
                if key != "execution_profile_sha256"
            }
        )
        != V5_PROFILE_SHA256
        or source_envelope.get("logical_custom_id") != V5_ADOPTION_RESULT_ONLY_ID
    ):
        raise ProfileContractError("v5 adoption source identity drifted")
    if (
        set(source_envelope) != V5_RESULT_ENVELOPE_KEYS
        or set(source_started_event) != V5_STARTED_EVENT_KEYS
        or source_envelope.get("logical_custom_id") != V5_ADOPTION_RESULT_ONLY_ID
        or source_started_event.get("logical_custom_id")
        != V5_ADOPTION_RESULT_ONLY_ID
        or source_envelope.get("schema_version")
        != "stt.eval.easy-medium-nonthinking-result.v5"
        or source_started_event.get("schema_version")
        != "stt.eval.easy-medium-nonthinking-journal.v5"
        or source_envelope.get("event_type") != "result"
        or source_started_event.get("event_type") != "started"
        or source_envelope.get("kind") != "contract"
        or source_started_event.get("kind") != "contract"
        or source_envelope.get("execution_profile_sha256") != V5_PROFILE_SHA256
        or source_started_event.get("execution_profile_sha256")
        != V5_PROFILE_SHA256
        or source_envelope.get("profile_runner_version")
        != V5_PROFILE_RUNNER_VERSION
        or source_envelope.get("model") != source_profile.get("model")
        or source_envelope.get("endpoint") != source_profile.get("endpoint")
        or source_envelope.get("transport") != source_profile.get("transport")
        or source_envelope.get("sse_completion_policy")
        != source_profile.get("sse_completion_policy")
        or source_envelope.get("request_overlay")
        != source_profile.get("request_overlay")
        or source_started_event.get("base_execution_contract_sha256")
        != source_contract_profile.get("base_execution_contract_sha256")
        or source_envelope.get("base_execution_contract_sha256")
        != source_contract_profile.get("base_execution_contract_sha256")
        or source_started_event.get("base_request_sha256")
        != source_contract_profile.get("base_request_sha256_by_id", {}).get(
            V5_ADOPTION_RESULT_ONLY_ID
        )
        or source_envelope.get("base_request_sha256")
        != source_contract_profile.get("base_request_sha256_by_id", {}).get(
            V5_ADOPTION_RESULT_ONLY_ID
        )
        or source_started_event.get("effective_request_sha256")
        != source_contract_profile.get("effective_request_sha256_by_id", {}).get(
            V5_ADOPTION_RESULT_ONLY_ID
        )
        or source_envelope.get("effective_request_sha256")
        != source_contract_profile.get("effective_request_sha256_by_id", {}).get(
            V5_ADOPTION_RESULT_ONLY_ID
        )
        or source_envelope.get("operation_id")
        != source_started_event.get("operation_id")
        or not str(source_envelope.get("operation_id") or "")
        or source_envelope.get("sequence") != source_started_event.get("sequence")
        or source_envelope.get("sequence")
        != (source_profile.get("base_packs") or {})
        .get("contract", {})
        .get("target_sequence_by_custom_id", {})
        .get(V5_ADOPTION_RESULT_ONLY_ID)
        or source_envelope.get("started_event_sha256")
        != sha256_json(source_started_event)
    ):
        raise ProfileContractError("v5 adopted result/start identity is not exact")
    source_core = source_envelope.get("core_result")
    if (
        not isinstance(source_core, Mapping)
        or source_envelope.get("core_result_sha256") != sha256_json(source_core)
    ):
        raise ProfileContractError("v5 adopted core/result binding is invalid")
    cal_by_id = {str(row["logical_custom_id"]): row for row in calibration_envelopes}
    base_by_id = {str(row["custom_id"]): row for row in calibration_base_requests}
    request_by_id = {str(row["custom_id"]): row for row in calibration_requests}
    started_by_id = {
        str(row["logical_custom_id"]): row for row in calibration_started_events
    }
    completed_by_id = {
        str(row["logical_custom_id"]): row for row in calibration_completed_events
    }
    if not (
        len(calibration_envelopes)
        == len(calibration_base_requests)
        == len(calibration_requests)
        == len(calibration_started_events)
        == len(calibration_completed_events)
        == len(cal_by_id)
        == len(base_by_id)
        == len(request_by_id)
        == len(started_by_id)
        == len(completed_by_id)
        == 4
        and set(cal_by_id)
        == set(base_by_id)
        == set(request_by_id)
        == set(started_by_id)
        == set(completed_by_id)
        == _expected_v5_calibration_ids()
    ):
        raise ProfileContractError("v5 calibration source is not exact 4/4")
    for custom_id, envelope in cal_by_id.items():
        started = started_by_id[custom_id]
        completed = completed_by_id[custom_id]
        core = envelope.get("core_result")
        expected_sequence = int(custom_id.split("-")[3])
        operation_id = str(envelope.get("operation_id") or "")
        if (
            set(started) != V5_STARTED_EVENT_KEYS
            or set(envelope) != V5_RESULT_ENVELOPE_KEYS
            or set(completed) != V5_COMPLETED_EVENT_KEYS
            or not operation_id
            or base_by_id[custom_id].get("custom_id") != custom_id
            or request_by_id[custom_id].get("custom_id") != custom_id
            or any(
                row.get("kind") != "calibration"
                or row.get("logical_custom_id") != custom_id
                or row.get("operation_id") != operation_id
                or row.get("sequence") != expected_sequence
                for row in (started, envelope, completed)
            )
            or started.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or completed.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v5"
            or started.get("event_type") != "started"
            or envelope.get("event_type") != "result"
            or completed.get("event_type") != "completed"
            or any(
                row.get("execution_profile_sha256") != V5_PROFILE_SHA256
                for row in (started, envelope, completed)
            )
            or envelope.get("profile_runner_version")
            != V5_PROFILE_RUNNER_VERSION
            or envelope.get("model") != source_profile.get("model")
            or envelope.get("endpoint") != source_profile.get("endpoint")
            or envelope.get("transport") != source_profile.get("transport")
            or envelope.get("sse_completion_policy")
            != source_profile.get("sse_completion_policy")
            or envelope.get("request_overlay")
            != source_profile.get("request_overlay")
            or started.get("base_execution_contract_sha256") is not None
            or envelope.get("base_execution_contract_sha256") is not None
            or envelope.get("started_event_sha256") != sha256_json(started)
            or completed.get("started_event_sha256") != sha256_json(started)
            or completed.get("result_envelope_sha256") != sha256_json(envelope)
            or not isinstance(core, Mapping)
            or envelope.get("core_result_sha256") != sha256_json(core)
            or started.get("base_request_sha256")
            != sha256_json(base_by_id[custom_id])
            or envelope.get("base_request_sha256")
            != sha256_json(base_by_id[custom_id])
            or started.get("effective_request_sha256")
            != sha256_json(request_by_id[custom_id])
            or envelope.get("effective_request_sha256")
            != sha256_json(request_by_id[custom_id])
            or completed.get("result_status") != core.get("status")
        ):
            raise ProfileContractError("v5 calibration causal chain is not exact")
    old_errors = list(old_materialization_errors)
    new_errors = list(new_materialization_errors)
    if old_errors != V5_ADOPTION_OLD_ERRORS_BY_ID[V5_ADOPTION_RESULT_ONLY_ID] or new_errors:
        raise ProfileContractError("v5 adoption old/new outcomes drifted")
    if dict(physical_sidecar_sha256) != V5_PHYSICAL_SIDECAR_SHA256:
        raise ProfileContractError("v5 physical sidecar identity is not exact")
    receipt = {
        "schema_version": "stt.eval.easy-medium-nonthinking-v5-adoption.v1",
        "policy": V5_ADOPTION_POLICY,
        "source_profile_id": V5_PROFILE_ID,
        "source_profile_sha256": V5_PROFILE_SHA256,
        "source_materializer_policy_version": V5_MATERIALIZATION_POLICY_VERSION,
        "source_production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "target_materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
        "target_production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "physical_sidecar_sha256": dict(physical_sidecar_sha256),
        "execution_summary_state": "absent",
        "physical_row_counts": {
            "journal.jsonl": 9,
            "responses.jsonl": 1,
            "calibration-responses.jsonl": 4,
        },
        "adopted_result_envelope_sha256_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: sha256_json(source_envelope)
        },
        "adopted_started_event_sha256_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: sha256_json(source_started_event)
        },
        "adopted_completed_event_sha256_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: None
        },
        "adopted_core_result_sha256_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: sha256_json(source_core)
        },
        "adopted_old_materialization_errors_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: old_errors
        },
        "adopted_new_materialization_errors_by_id": {
            V5_ADOPTION_RESULT_ONLY_ID: new_errors
        },
        "calibration_base_request_sha256_by_id": {
            custom_id: sha256_json(row) for custom_id, row in base_by_id.items()
        },
        "calibration_request_sha256_by_id": {
            custom_id: sha256_json(row) for custom_id, row in request_by_id.items()
        },
        "calibration_result_envelope_sha256_by_id": {
            custom_id: sha256_json(row) for custom_id, row in cal_by_id.items()
        },
        "calibration_started_event_sha256_by_id": {
            custom_id: sha256_json(row) for custom_id, row in started_by_id.items()
        },
        "calibration_completed_event_sha256_by_id": {
            custom_id: sha256_json(row) for custom_id, row in completed_by_id.items()
        },
        "calibration_core_result_sha256_by_id": {
            custom_id: sha256_json(row["core_result"])
            for custom_id, row in cal_by_id.items()
        },
        "calibration_evidence_role": "source_execution_precondition_only",
        "calibration_authorizes_v6_calls": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    if set(receipt) != V5_ADOPTION_RECEIPT_KEYS:
        raise ProfileContractError("v5 adoption receipt schema is not exact")
    return receipt


def v4_adoption_bridge_receipt(
    *,
    source_profile: Mapping[str, Any],
    source_envelopes: Sequence[Mapping[str, Any]],
    source_started_events: Sequence[Mapping[str, Any]],
    source_completed_events: Sequence[Mapping[str, Any]],
    calibration_envelopes: Sequence[Mapping[str, Any]],
    calibration_base_requests: Sequence[Mapping[str, Any]],
    calibration_requests: Sequence[Mapping[str, Any]],
    calibration_started_events: Sequence[Mapping[str, Any]],
    calibration_completed_events: Sequence[Mapping[str, Any]],
    physical_sidecar_sha256: Mapping[str, str],
    old_materialization_errors_by_id: Mapping[str, Sequence[str]],
    new_materialization_errors_by_id: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """构造绑定v4完整物理sidecar及零API采用范围的自散列收据。"""

    if source_profile.get("execution_profile_sha256") != V4_PROFILE_SHA256:
        raise ProfileContractError("v4 adoption source profile SHA drifted")
    started_by_operation = {
        str(row.get("operation_id") or ""): row for row in source_started_events
    }
    completed_by_operation = {
        str(row.get("operation_id") or ""): row for row in source_completed_events
    }
    adopted_ids = [str(row.get("logical_custom_id") or "") for row in source_envelopes]
    if (
        len(adopted_ids) != 2
        or len(set(adopted_ids)) != 2
        or any(not value for value in adopted_ids)
    ):
        raise ProfileContractError("v4 adoption must bind exact two production results")
    calibration_request_by_id = {
        str(row.get("custom_id") or ""): row for row in calibration_requests
    }
    if (
        len(calibration_request_by_id) != 4
        or len(calibration_request_by_id) != len(calibration_requests)
    ):
        raise ProfileContractError("v4 adoption calibration requests are not exact 4/4")
    calibration_base_request_by_id = {
        str(row.get("custom_id") or ""): row for row in calibration_base_requests
    }
    if (
        set(calibration_base_request_by_id) != set(calibration_request_by_id)
        or len(calibration_base_request_by_id) != len(calibration_base_requests)
    ):
        raise ProfileContractError("v4 adoption calibration base requests drifted")
    normalized_old_errors = {
        str(custom_id): list(errors)
        for custom_id, errors in old_materialization_errors_by_id.items()
    }
    normalized_new_errors = {
        str(custom_id): list(errors)
        for custom_id, errors in new_materialization_errors_by_id.items()
    }
    if (
        normalized_old_errors != V4_ADOPTION_OLD_ERRORS_BY_ID
        or normalized_new_errors != {custom_id: [] for custom_id in adopted_ids}
    ):
        raise ProfileContractError("v4 adoption old/new validator outcomes drifted")
    receipt = {
        "schema_version": "stt.eval.easy-medium-nonthinking-v4-adoption.v1",
        "policy": V4_ADOPTION_POLICY,
        "source_profile_id": V4_PROFILE_ID,
        "source_profile_sha256": V4_PROFILE_SHA256,
        "source_prompt_sha256": source_profile.get("prompt_sha256"),
        "source_materializer_policy_version": LEGACY_MATERIALIZATION_POLICY_VERSION,
        "source_production_matcher_policy_version": (
            LEGACY_PRODUCTION_MATCHER_POLICY_VERSION
        ),
        "materializer_policy_version": MATERIALIZATION_POLICY_VERSION,
        "production_matcher": PRODUCTION_MATCHER,
        "production_matcher_policy_version": PRODUCTION_MATCHER_POLICY_VERSION,
        "physical_sidecar_sha256": dict(physical_sidecar_sha256),
        "adopted_result_envelope_sha256_by_id": {
            str(row["logical_custom_id"]): sha256_json(row)
            for row in source_envelopes
        },
        "adopted_started_event_sha256_by_id": {
            str(row["logical_custom_id"]): sha256_json(
                started_by_operation[str(row["operation_id"])]
            )
            for row in source_envelopes
        },
        "adopted_completed_event_sha256_by_id": {
            str(row["logical_custom_id"]): (
                sha256_json(completed_by_operation[str(row["operation_id"])])
                if str(row["operation_id"]) in completed_by_operation
                else None
            )
            for row in source_envelopes
        },
        "adopted_old_materialization_errors_by_id": normalized_old_errors,
        "adopted_new_materialization_errors_by_id": normalized_new_errors,
        "adopted_old_materialization_errors_sha256_by_id": {
            custom_id: sha256_json(errors)
            for custom_id, errors in normalized_old_errors.items()
        },
        "adopted_new_materialization_errors_sha256_by_id": {
            custom_id: sha256_json(errors)
            for custom_id, errors in normalized_new_errors.items()
        },
        "calibration_base_request_sha256_by_id": {
            custom_id: sha256_json(row)
            for custom_id, row in calibration_base_request_by_id.items()
        },
        "calibration_request_sha256_by_id": {
            custom_id: sha256_json(row)
            for custom_id, row in calibration_request_by_id.items()
        },
        "calibration_result_envelope_sha256_by_id": {
            str(row["logical_custom_id"]): sha256_json(row)
            for row in calibration_envelopes
        },
        "calibration_started_event_sha256_by_id": {
            str(row["logical_custom_id"]): sha256_json(row)
            for row in calibration_started_events
        },
        "calibration_completed_event_sha256_by_id": {
            str(row["logical_custom_id"]): sha256_json(row)
            for row in calibration_completed_events
        },
        "calibration_evidence_role": "source_execution_precondition_only",
        "calibration_authorizes_v5_calls": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def predecessor_bridge_receipt(
    *,
    predecessor_profile: Mapping[str, Any],
    predecessor_journal_rows: Sequence[dict[str, Any]],
    predecessor_response_rows: Sequence[dict[str, Any]],
    predecessor_calibration_rows: Sequence[dict[str, Any]],
    successful_envelope_by_id: Mapping[str, Mapping[str, Any]],
    failed_envelope_by_id: Mapping[str, Mapping[str, Any]],
    physical_sidecar_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """构造 v1 exact replay bridge；failed 只保留证据，永不变成成功。"""

    if (
        predecessor_profile.get("schema_version") != PREDECESSOR_PROFILE_SCHEMA_VERSION
        or predecessor_profile.get("profile_id") != PREDECESSOR_PROFILE_ID
        or predecessor_profile.get("runner_version") != PREDECESSOR_RUNNER_VERSION
        or predecessor_profile.get("execution_profile_sha256")
        != PREDECESSOR_PROFILE_SHA256
        or sha256_json(
            {
                key: value
                for key, value in predecessor_profile.items()
                if key != "execution_profile_sha256"
            }
        )
        != PREDECESSOR_PROFILE_SHA256
    ):
        raise ProfileContractError("predecessor v1 profile identity/hash drifted")
    expected_sidecar_names = {
        "execution-profile.json",
        "journal.jsonl",
        "responses.jsonl",
        "calibration-responses.jsonl",
        "execution-summary.json",
    }
    if set(physical_sidecar_sha256) != expected_sidecar_names or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in physical_sidecar_sha256.values()
    ):
        raise ProfileContractError("predecessor physical sidecar hashes are invalid")
    started_by_operation = {
        str(row.get("operation_id") or ""): row
        for row in predecessor_journal_rows
        if row.get("event_type") == "started"
    }
    completed_by_operation = {
        str(row.get("operation_id") or ""): row
        for row in predecessor_journal_rows
        if row.get("event_type") == "completed"
        and row.get("execution_profile_sha256") == PREDECESSOR_PROFILE_SHA256
    }
    if (
        len(started_by_operation) != len(completed_by_operation)
        or len(predecessor_journal_rows) != 2 * len(started_by_operation)
    ):
        raise ProfileContractError("predecessor journal is not exact started/completed pairs")
    all_envelopes = [*predecessor_response_rows, *predecessor_calibration_rows]
    if len(all_envelopes) != len(started_by_operation):
        raise ProfileContractError("predecessor result/journal operation counts drifted")
    for envelope in all_envelopes:
        operation_id = str(envelope.get("operation_id") or "")
        started = started_by_operation.get(operation_id)
        completed = completed_by_operation.get(operation_id)
        core = envelope.get("core_result")
        if (
            not isinstance(started, Mapping)
            or not isinstance(completed, Mapping)
            or not isinstance(core, Mapping)
            or envelope.get("schema_version") != PREDECESSOR_RESULT_SCHEMA_VERSION
            or envelope.get("event_type") != "result"
            or envelope.get("execution_profile_sha256") != PREDECESSOR_PROFILE_SHA256
            or envelope.get("profile_runner_version") != PREDECESSOR_RUNNER_VERSION
            or envelope.get("started_event_sha256") != sha256_json(started)
            or envelope.get("core_result_sha256") != sha256_json(core)
            or started.get("schema_version") != PREDECESSOR_EVENT_SCHEMA_VERSION
            or started.get("execution_profile_sha256") != PREDECESSOR_PROFILE_SHA256
            or completed.get("schema_version") != PREDECESSOR_EVENT_SCHEMA_VERSION
            or completed.get("execution_profile_sha256") != PREDECESSOR_PROFILE_SHA256
            or completed.get("started_event_sha256") != sha256_json(started)
            or completed.get("result_envelope_sha256") != sha256_json(envelope)
            or completed.get("result_status") != core.get("status")
            or any(
                envelope.get(key) != started.get(key)
                or envelope.get(key) != completed.get(key)
                for key in ("kind", "logical_custom_id", "sequence", "operation_id")
            )
        ):
            raise ProfileContractError("predecessor operation exact replay failed")
    result_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in predecessor_response_rows
        if row.get("event_type") == "result"
    }
    if len(result_by_id) != len(predecessor_response_rows):
        raise ProfileContractError("predecessor response ids are duplicate/foreign")
    for custom_id, envelope in result_by_id.items():
        completed = completed_by_operation.get(str(envelope.get("operation_id") or ""))
        if (
            envelope.get("schema_version") != PREDECESSOR_RESULT_SCHEMA_VERSION
            or envelope.get("execution_profile_sha256") != PREDECESSOR_PROFILE_SHA256
            or envelope.get("profile_runner_version") != PREDECESSOR_RUNNER_VERSION
            or completed is None
            or completed.get("schema_version") != PREDECESSOR_EVENT_SCHEMA_VERSION
            or completed.get("logical_custom_id") != custom_id
            or completed.get("result_envelope_sha256") != sha256_json(envelope)
        ):
            raise ProfileContractError("predecessor result is not exact completed v1")
    if set(successful_envelope_by_id) & set(failed_envelope_by_id):
        raise ProfileContractError("predecessor success/failed bridge overlaps")
    for custom_id, envelope in successful_envelope_by_id.items():
        if (
            dict(envelope) != result_by_id.get(custom_id)
            or envelope.get("core_result", {}).get("status") != "succeeded"
        ):
            raise ProfileContractError("predecessor success bridge is not exact v1 success")
    for custom_id, envelope in failed_envelope_by_id.items():
        if (
            dict(envelope) != result_by_id.get(custom_id)
            or envelope.get("core_result", {}).get("status") != "failed"
        ):
            raise ProfileContractError("predecessor failure evidence is not exact v1 failed")
    calibration_success_ids = sorted(
        str(row.get("logical_custom_id") or "")
        for row in predecessor_calibration_rows
        if row.get("core_result", {}).get("status") in SUCCESS_STATUSES
        and str(row.get("operation_id") or "") in completed_by_operation
    )
    if len(calibration_success_ids) != 4:
        raise ProfileContractError("predecessor bridge requires exact completed v1 4/4")
    receipt = {
        "predecessor_profile_id": PREDECESSOR_PROFILE_ID,
        "predecessor_profile_sha256": PREDECESSOR_PROFILE_SHA256,
        "physical_sidecar_sha256": dict(sorted(physical_sidecar_sha256.items())),
        "journal_pack_sha256": sha256_json(list(predecessor_journal_rows)),
        "response_pack_sha256": sha256_json(list(predecessor_response_rows)),
        "calibration_response_pack_sha256": sha256_json(
            list(predecessor_calibration_rows)
        ),
        "calibration_success_custom_ids": calibration_success_ids,
        "bridged_success_envelope_sha256_by_id": {
            custom_id: sha256_json(envelope)
            for custom_id, envelope in sorted(successful_envelope_by_id.items())
        },
        "failed_evidence_envelope_sha256_by_id": {
            custom_id: sha256_json(envelope)
            for custom_id, envelope in sorted(failed_envelope_by_id.items())
        },
        "failed_reuse_allowed": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def _json_tree_differences(
    before: Any, after: Any, *, pointer: str = ""
) -> list[dict[str, Any]]:
    """按JSON类型精确计算叶节点差异，避免 ``False == 0`` 混淆。"""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return [{"json_pointer": pointer or "/", "kind": "object_keys"}]
        differences: list[dict[str, Any]] = []
        for key in before:
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            differences.extend(
                _json_tree_differences(
                    before[key], after[key], pointer=f"{pointer}/{escaped}"
                )
            )
        return differences
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [{"json_pointer": pointer or "/", "kind": "array_length"}]
        differences = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            differences.extend(
                _json_tree_differences(left, right, pointer=f"{pointer}/{index}")
            )
        return differences
    if type(before) is type(after) and before == after:
        return []
    return [
        {
            "json_pointer": pointer or "/",
            "kind": "replace",
            "before_type": type(before).__name__,
            "before_sha256": sha256_json(before),
            "after_type": type(after).__name__,
            "after_sha256": sha256_json(after),
        }
    ]


def project_terminal_polarity_failure(
    *,
    source_envelope: Mapping[str, Any],
    source_completed_event: Mapping[str, Any],
    base_request: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    kind: str,
    validate_projected_review: Callable[
        [dict[str, Any], dict[str, Any]], Sequence[str]
    ],
) -> dict[str, Any]:
    """已禁用：v2不得把任何predecessor failed投影为success。"""

    raise ProfileContractError(
        "saved projector is disabled; predecessor failed must remain v2 pending"
    )

    validate_execution_profile(execution_profile)
    custom_id = str(source_envelope.get("logical_custom_id") or "")
    core = source_envelope.get("core_result")
    error = (
        "critical_constraints.polarity.required must be one of the frozen polarity values: "
        f"{list(POLARITY_REQUIRED_VALUES)}"
    )
    if (
        kind not in KINDS
        or not isinstance(core, Mapping)
        or source_envelope.get("schema_version") != PREDECESSOR_RESULT_SCHEMA_VERSION
        or source_envelope.get("execution_profile_sha256")
        != PREDECESSOR_PROFILE_SHA256
        or core.get("status") != "failed"
        or source_completed_event.get("result_envelope_sha256")
        != sha256_json(source_envelope)
        or source_completed_event.get("result_status") != "failed"
        or core.get("custom_id") != custom_id
        or core.get("failure_termination") != "review_validation_error"
        or core.get("retry_accounting")
        != {
            "policy": "bounded-transport-semantic-failure-quotas-v1",
            "transport_failure_quota": 3,
            "semantic_failure_quota": 3,
            "hard_total_attempt_quota": 6,
            "total_attempts": 3,
            "transport_failure_count": 0,
            "semantic_failure_count": 3,
            "quota_exhausted": "semantic",
        }
        or core.get("error") != f"ValueError: {error}"
    ):
        raise ProfileContractError("source is not the exact terminal polarity failure")
    invalid_attempts = core.get("invalid_attempts")
    if not isinstance(invalid_attempts, list) or len(invalid_attempts) != 3:
        raise ProfileContractError("source polarity failure attempts are not exact three")
    reviews: list[dict[str, Any]] = []
    for index, attempt in enumerate(invalid_attempts, start=1):
        raw = attempt.get("raw_review")
        polarity = (
            (raw.get("critical_constraints") or {}).get("polarity")
            if isinstance(raw, Mapping)
            else None
        )
        if (
            attempt.get("attempt") != index
            or attempt.get("validation_errors") != [error]
            or not isinstance(raw, Mapping)
            or raw.get("verdict") != "approve"
            or not isinstance(polarity, Mapping)
            or polarity.get("required") is not False
            or polarity.get("cues") != []
            or attempt.get("review") != raw
        ):
            raise ProfileContractError("source polarity failure attempt is not projectable")
        reviews.append(deepcopy(dict(raw)))
    raw = deepcopy(dict(core.get("invalid_review") or {}))
    if raw != reviews[-1]:
        raise ProfileContractError("terminal invalid review differs from final attempt")
    payload = json.loads(str(base_request["request"]["messages"][1]["content"]))
    candidate_polarity = (
        ((payload.get("deterministic_candidate") or {}).get("critical_constraints") or {})
        .get("polarity")
    )
    if (
        not isinstance(candidate_polarity, Mapping)
        or candidate_polarity.get("required") != "not_applicable"
        or any(
            candidate_polarity.get(key) != []
            for key in (
                "query_negative_cues",
                "reference_negative_cues",
                "reference_positive_cues",
                "evidence_negative_cues",
            )
        )
    ):
        raise ProfileContractError("payload candidate does not prove not_applicable")
    projected = deepcopy(raw)
    projected["critical_constraints"]["polarity"]["required"] = "not_applicable"
    tree_diff = _json_tree_differences(raw, projected)
    expected_diff = [{
        "json_pointer": "/critical_constraints/polarity/required",
        "kind": "replace",
        "before_type": "bool",
        "before_sha256": sha256_json(False),
        "after_type": "str",
        "after_sha256": sha256_json("not_applicable"),
    }]
    if tree_diff != expected_diff:
        raise ProfileContractError("saved projection tree diff is not the one allowed leaf")
    errors = [str(value) for value in validate_projected_review(projected, payload) if str(value)]
    if errors:
        raise ProfileContractError("saved projection strict replay failed: " + "; ".join(errors))
    effective = effective_request_row(base_request, execution_profile, kind=kind)
    receipt = {
        "schema_version": "disabled-saved-result",
        "event_type": "saved_result",
        "kind": kind,
        "custom_id": custom_id,
        "logical_custom_id": custom_id,
        "sequence": execution_profile["base_packs"][kind][
            "target_sequence_by_custom_id"
        ][custom_id],
        "execution_profile_id": PROFILE_ID,
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "runner_version": "disabled-saved-runner",
        "transport": "disabled-saved-transport",
        "status": "succeeded",
        "model": execution_profile["model"],
        "prompt_sha256": execution_profile["prompt_sha256"],
        "completed_at": source_completed_event["completed_at"],
        "api_call_performed": False,
        "attempt": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "usage_is_lower_bound": False,
        "successful_response_usage": {},
        "attempt_usage": [],
        "review": projected,
        "response_sha256": sha256_json(projected),
        "base_request_sha256": sha256_json(base_request),
        "effective_request_sha256": sha256_json(effective),
        "payload_sha256": base_request["payload_sha256"],
        "projection": {
            "operation": "replace",
            "json_pointer": "/critical_constraints/polarity/required",
            "before_type": "boolean",
            "before_sha256": sha256_json(False),
            "after": "not_applicable",
            "after_sha256": sha256_json("not_applicable"),
            "tree_diff_sha256": sha256_json(tree_diff),
            "source_profile_sha256": PREDECESSOR_PROFILE_SHA256,
            "source_result_envelope_sha256": sha256_json(source_envelope),
            "source_completed_event_sha256": sha256_json(source_completed_event),
            "source_core_result_sha256": sha256_json(core),
            "source_terminal_attempt": 3,
            "source_usage": deepcopy(dict(core.get("usage") or {})),
            "source_attempt_usage_sha256": sha256_json(core.get("attempt_usage")),
        },
    }
    receipt["saved_result_sha256"] = sha256_json(receipt)
    return receipt


def _validate_runtime(runtime: PackRuntime) -> dict[str, Any]:
    if runtime.kind not in KINDS:
        raise ProfileContractError("runtime kind is invalid")
    try:
        replay_state = runtime.replay_legacy(
            [deepcopy(row) for row in runtime.legacy_rows]
        )
    except Exception as exc:
        raise ProfileContractError(
            f"legacy formal replay failed: {type(exc).__name__}: {exc}"
        ) from exc
    expected_state_keys = {
        "successful_by_id",
        "retry_eligible_by_id",
        "invalidated_by_id",
        "retry_started_by_id",
        "terminal_by_id",
        "adopted_by_id",
        "episode_by_id",
    }
    if not isinstance(replay_state, Mapping) or set(replay_state) != expected_state_keys:
        raise ProfileContractError("legacy formal replay state schema is not exact")
    if any(not isinstance(replay_state[key], Mapping) for key in expected_state_keys):
        raise ProfileContractError("legacy formal replay state bucket is invalid")
    legacy_success = replay_state["successful_by_id"]
    physical_by_id: dict[str, list[dict[str, Any]]] = {}
    pack_ids = {str(row.get("custom_id") or "") for row in runtime.pack.requests}
    for row in runtime.legacy_rows:
        physical_by_id.setdefault(str(row.get("custom_id") or ""), []).append(row)
    trusted: dict[str, dict[str, Any]] = {}
    for key, value in legacy_success.items():
        custom_id = str(key or "")
        if (
            custom_id not in pack_ids
            or not isinstance(value, Mapping)
            or value.get("custom_id") != custom_id
            or value.get("status") != "succeeded"
            or not any(dict(value) == physical for physical in physical_by_id.get(custom_id, []))
        ):
            raise ProfileContractError(
                "legacy formal replay returned a row not byte-exact in physical legacy rows"
            )
        trusted[custom_id] = deepcopy(dict(value))
    state_ids = {
        str(custom_id)
        for key in expected_state_keys
        for custom_id in replay_state[key]
    }
    if state_ids - pack_ids:
        raise ProfileContractError("legacy formal replay state contains a foreign id")
    materialization_blocked = set()
    for key in (
        "retry_eligible_by_id",
        "invalidated_by_id",
        "retry_started_by_id",
        "terminal_by_id",
    ):
        materialization_blocked.update(str(value) for value in replay_state[key])
    if materialization_blocked & set(trusted):
        raise ProfileContractError("legacy replay success/materialization state overlaps")
    return {
        "successful_by_id": trusted,
        "materialization_blocked_custom_ids": sorted(materialization_blocked),
        "formal_state": deepcopy(dict(replay_state)),
    }


def _validated_v4_adoption(
    runtime: PackRuntime, execution_profile: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """从runtime携带的物理v4源重建exact adoption与calibration迁移收据。"""

    bridge = runtime.v4_adoption_bridge
    if bridge is None:
        if execution_profile["v4_adoption_policy"].get("bridge_receipt_sha256") is not None:
            raise ProfileContractError("v5 profile/runtime v4 adoption bridge mismatch")
        return {}, None
    receipt = deepcopy(bridge.receipt)
    claimed = receipt.pop("receipt_sha256", None)
    if (
        claimed != sha256_json(receipt)
        or claimed
        != execution_profile["v4_adoption_policy"].get("bridge_receipt_sha256")
        or receipt.get("source_profile_sha256") != V4_PROFILE_SHA256
        or receipt.get("source_materializer_policy_version")
        != LEGACY_MATERIALIZATION_POLICY_VERSION
        or receipt.get("source_production_matcher_policy_version")
        != LEGACY_PRODUCTION_MATCHER_POLICY_VERSION
        or receipt.get("materializer_policy_version")
        != MATERIALIZATION_POLICY_VERSION
        or receipt.get("production_matcher_policy_version")
        != PRODUCTION_MATCHER_POLICY_VERSION
    ):
        raise ProfileContractError("v4 adoption bridge/profile receipt mismatch")
    target_ids = set(_selection(runtime.pack).target_custom_ids)
    expected_results = {
        custom_id: digest
        for custom_id, digest in receipt["adopted_result_envelope_sha256_by_id"].items()
        if custom_id in target_ids
    }
    source_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.source_envelopes
    }
    projected_by_id = {
        str(row.get("custom_id") or ""): row for row in bridge.success_rows
    }
    started_by_operation = {
        str(row.get("operation_id") or ""): row
        for row in bridge.source_started_events
    }
    completed_by_operation = {
        str(row.get("operation_id") or ""): row
        for row in bridge.source_completed_events
    }
    if (
        len(source_by_id) != len(bridge.source_envelopes)
        or len(projected_by_id) != len(bridge.success_rows)
        or set(source_by_id) != set(expected_results)
        or set(projected_by_id) != set(expected_results)
    ):
        raise ProfileContractError("v4 adoption source IDs are missing/duplicate")
    adopted: dict[str, dict[str, Any]] = {}
    for custom_id, envelope in source_by_id.items():
        operation_id = str(envelope.get("operation_id") or "")
        started = started_by_operation.get(operation_id)
        completed = completed_by_operation.get(operation_id)
        core = envelope.get("core_result")
        base = _base_request_by_id(runtime.pack, custom_id)
        effective = _effective_from_base(base)
        old_errors = (
            ["v4 adopted core is invalid"]
            if not isinstance(core, dict)
            else list(
                bridge.validate_v4_source_core(deepcopy(core), effective, False)
            )
        )
        new_errors = (
            ["v4 adopted core is invalid"]
            if not isinstance(core, dict)
            else list(
                bridge.validate_v5_adoption_core(deepcopy(core), effective, False)
            )
        )
        expected_completed_sha = receipt[
            "adopted_completed_event_sha256_by_id"
        ][custom_id]
        if (
            sha256_json(envelope) != expected_results[custom_id]
            or not isinstance(started, dict)
            or sha256_json(started)
            != receipt["adopted_started_event_sha256_by_id"][custom_id]
            or (
                sha256_json(completed) if completed is not None else None
            )
            != expected_completed_sha
            or old_errors
            != receipt["adopted_old_materialization_errors_by_id"][custom_id]
            or new_errors
            != receipt["adopted_new_materialization_errors_by_id"][custom_id]
            or sha256_json(old_errors)
            != receipt["adopted_old_materialization_errors_sha256_by_id"][custom_id]
            or sha256_json(new_errors)
            != receipt["adopted_new_materialization_errors_sha256_by_id"][custom_id]
        ):
            raise ProfileContractError(
                "v4 adoption source exact replay failed"
                + (": " + "; ".join(new_errors) if new_errors else "")
            )
        expected_projection = _project_v4_adopted_success(
            envelope, started_event=started, completed_event=completed
        )
        if projected_by_id[custom_id] != expected_projection:
            raise ProfileContractError("v4 adoption projection exact replay drifted")
        adopted[custom_id] = deepcopy(expected_projection)
    if list(bridge.source_core_rows) != [
        deepcopy(row["core_result"]) for row in bridge.source_envelopes
    ]:
        raise ProfileContractError("v4 adoption source core/usage rows drifted")

    calibration_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_envelopes
    }
    calibration_started_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_started_events
    }
    calibration_completed_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_completed_events
    }
    calibration_requests = {
        str(row["custom_id"]): row for row in bridge.calibration_requests
    }
    calibration_base_requests = {
        str(row["custom_id"]): row for row in bridge.calibration_base_requests
    }
    expected_calibration = receipt["calibration_result_envelope_sha256_by_id"]
    if not (
        len(calibration_by_id) == 4
        and set(calibration_by_id)
        == set(calibration_started_by_id)
        == set(calibration_completed_by_id)
        == set(calibration_requests)
        == set(expected_calibration)
        == set(receipt["calibration_request_sha256_by_id"])
        == set(calibration_base_requests)
        == set(receipt["calibration_base_request_sha256_by_id"])
    ):
        raise ProfileContractError("v4 calibration migration source is not exact 4/4")
    for custom_id, envelope in calibration_by_id.items():
        core = envelope.get("core_result")
        errors = (
            ["v4 calibration core is invalid"]
            if not isinstance(core, dict)
            else list(
                bridge.validate_v4_source_core(
                    deepcopy(core), calibration_requests[custom_id], True
                )
            )
        )
        request = calibration_requests[custom_id]
        base_request = calibration_base_requests[custom_id]
        started = calibration_started_by_id[custom_id]
        if (
            sha256_json(envelope) != expected_calibration[custom_id]
            or sha256_json(request)
            != receipt["calibration_request_sha256_by_id"][custom_id]
            or sha256_json(base_request)
            != receipt["calibration_base_request_sha256_by_id"][custom_id]
            or sha256_json(calibration_started_by_id[custom_id])
            != receipt["calibration_started_event_sha256_by_id"][custom_id]
            or sha256_json(calibration_completed_by_id[custom_id])
            != receipt["calibration_completed_event_sha256_by_id"][custom_id]
            or core.get("status") not in SUCCESS_STATUSES
            or started.get("base_request_sha256") != sha256_json(base_request)
            or started.get("effective_request_sha256") != sha256_json(request)
            or envelope.get("base_request_sha256") != sha256_json(base_request)
            or envelope.get("effective_request_sha256") != sha256_json(request)
            or errors
        ):
            raise ProfileContractError("v4 calibration migration exact replay failed")
    migration = {
        "evidence_role": "source_execution_precondition_only",
        "source_profile_sha256": V4_PROFILE_SHA256,
        "source_result_envelope_sha256_by_id": deepcopy(expected_calibration),
        "authorizes_v5_calls": False,
        "bridge_receipt_sha256": claimed,
    }
    migration["receipt_sha256"] = sha256_json(migration)
    return adopted, migration


def _validated_v5_adoption(
    runtime: PackRuntime, execution_profile: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """从pin住的v5物理source重建result-only adoption与cal evidence。"""

    bridge = runtime.v5_adoption_bridge
    profile_receipt_sha = execution_profile["v5_adoption_policy"].get(
        "bridge_receipt_sha256"
    )
    if bridge is None:
        if profile_receipt_sha is not None:
            raise ProfileContractError("v6 profile/runtime v5 adoption bridge mismatch")
        return {}, None
    source_profile = bridge.source_profile
    if (
        source_profile.get("profile_id") != V5_PROFILE_ID
        or source_profile.get("runner_version") != V5_PROFILE_RUNNER_VERSION
        or source_profile.get("execution_profile_sha256") != V5_PROFILE_SHA256
        or sha256_json(
            {
                key: value
                for key, value in source_profile.items()
                if key != "execution_profile_sha256"
            }
        )
        != V5_PROFILE_SHA256
    ):
        raise ProfileContractError("v5 adoption source profile is not exact")
    receipt = deepcopy(bridge.receipt)
    if set(receipt) != V5_ADOPTION_RECEIPT_KEYS:
        raise ProfileContractError("v5 adoption receipt schema is not exact")
    claimed = receipt.pop("receipt_sha256", None)
    adopted_map_keys = (
        "adopted_result_envelope_sha256_by_id",
        "adopted_started_event_sha256_by_id",
        "adopted_completed_event_sha256_by_id",
        "adopted_core_result_sha256_by_id",
        "adopted_old_materialization_errors_by_id",
        "adopted_new_materialization_errors_by_id",
    )
    calibration_map_keys = (
        "calibration_base_request_sha256_by_id",
        "calibration_request_sha256_by_id",
        "calibration_result_envelope_sha256_by_id",
        "calibration_started_event_sha256_by_id",
        "calibration_completed_event_sha256_by_id",
        "calibration_core_result_sha256_by_id",
    )
    if any(
        not isinstance(receipt.get(key), Mapping)
        or set(receipt[key]) != {V5_ADOPTION_RESULT_ONLY_ID}
        for key in adopted_map_keys
    ) or any(
        not isinstance(receipt.get(key), Mapping)
        or set(receipt[key]) != _expected_v5_calibration_ids()
        for key in calibration_map_keys
    ):
        raise ProfileContractError("v5 adoption receipt source maps are not exact")
    digest_map_keys = (
        "adopted_result_envelope_sha256_by_id",
        "adopted_started_event_sha256_by_id",
        "adopted_core_result_sha256_by_id",
        *calibration_map_keys,
    )
    if (
        receipt["adopted_completed_event_sha256_by_id"]
        != {V5_ADOPTION_RESULT_ONLY_ID: None}
        or receipt["adopted_old_materialization_errors_by_id"]
        != V5_ADOPTION_OLD_ERRORS_BY_ID
        or receipt["adopted_new_materialization_errors_by_id"]
        != {V5_ADOPTION_RESULT_ONLY_ID: []}
        or any(
            any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in receipt[key].values()
            )
            for key in digest_map_keys
        )
    ):
        raise ProfileContractError("v5 adoption receipt source values are not exact")
    if (
        claimed != sha256_json(receipt)
        or claimed != profile_receipt_sha
        or receipt.get("schema_version")
        != "stt.eval.easy-medium-nonthinking-v5-adoption.v1"
        or receipt.get("policy") != V5_ADOPTION_POLICY
        or receipt.get("source_profile_id") != V5_PROFILE_ID
        or receipt.get("source_profile_sha256") != V5_PROFILE_SHA256
        or receipt.get("source_materializer_policy_version")
        != V5_MATERIALIZATION_POLICY_VERSION
        or receipt.get("source_production_matcher_policy_version")
        != PRODUCTION_MATCHER_POLICY_VERSION
        or receipt.get("target_materializer_policy_version")
        != MATERIALIZATION_POLICY_VERSION
        or receipt.get("target_production_matcher_policy_version")
        != PRODUCTION_MATCHER_POLICY_VERSION
        or receipt.get("physical_sidecar_sha256")
        != V5_PHYSICAL_SIDECAR_SHA256
        or receipt.get("execution_summary_state") != "absent"
        or receipt.get("physical_row_counts")
        != {
            "journal.jsonl": 9,
            "responses.jsonl": 1,
            "calibration-responses.jsonl": 4,
        }
        or receipt.get("calibration_evidence_role")
        != "source_execution_precondition_only"
        or receipt.get("calibration_authorizes_v6_calls") is not False
    ):
        raise ProfileContractError("v5 adoption bridge/profile receipt mismatch")
    target_ids = set(_selection(runtime.pack).target_custom_ids)
    expected_global = dict(receipt["adopted_result_envelope_sha256_by_id"])
    expected = {
        custom_id: digest
        for custom_id, digest in expected_global.items()
        if custom_id in target_ids
    }
    source_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.source_envelopes
    }
    projected_by_id = {
        str(row.get("custom_id") or ""): row for row in bridge.success_rows
    }
    started_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.source_started_events
    }
    if (
        len(source_by_id) != len(bridge.source_envelopes)
        or set(source_by_id) != set(expected)
        or set(projected_by_id) != set(expected)
        or set(started_by_id) != set(expected)
    ):
        raise ProfileContractError("v5 adoption source IDs are missing/duplicate")
    adopted: dict[str, dict[str, Any]] = {}
    for custom_id, envelope in source_by_id.items():
        started = started_by_id[custom_id]
        core = envelope.get("core_result")
        base = _base_request_by_id(runtime.pack, custom_id)
        effective = _effective_from_base(base)
        old_errors = (
            ["v5 adopted core is invalid"]
            if not isinstance(core, dict)
            else list(bridge.validate_v5_source_core(deepcopy(core), effective, False))
        )
        new_errors = (
            ["v5 adopted core is invalid"]
            if not isinstance(core, dict)
            else list(bridge.validate_v6_adoption_core(deepcopy(core), effective, False))
        )
        source_contract_profile = source_profile["base_packs"][runtime.kind]
        expected_production_sequence = _expected_sequence(
            kind=runtime.kind,
            logical_custom_id=custom_id,
            execution_profile=source_profile,
        )
        if (
            set(envelope) != V5_RESULT_ENVELOPE_KEYS
            or set(started) != V5_STARTED_EVENT_KEYS
            or sha256_json(envelope) != expected[custom_id]
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v5"
            or started.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or envelope.get("event_type") != "result"
            or started.get("event_type") != "started"
            or envelope.get("kind") != runtime.kind
            or started.get("kind") != runtime.kind
            or envelope.get("logical_custom_id") != custom_id
            or started.get("logical_custom_id") != custom_id
            or envelope.get("operation_id") != started.get("operation_id")
            or not str(envelope.get("operation_id") or "")
            or envelope.get("execution_profile_sha256") != V5_PROFILE_SHA256
            or started.get("execution_profile_sha256") != V5_PROFILE_SHA256
            or envelope.get("profile_runner_version")
            != V5_PROFILE_RUNNER_VERSION
            or envelope.get("model") != source_profile.get("model")
            or envelope.get("endpoint") != source_profile.get("endpoint")
            or envelope.get("transport") != source_profile.get("transport")
            or envelope.get("sse_completion_policy")
            != source_profile.get("sse_completion_policy")
            or envelope.get("request_overlay")
            != source_profile.get("request_overlay")
            or envelope.get("sequence") != expected_production_sequence
            or started.get("sequence") != expected_production_sequence
            or sha256_json(started)
            != receipt["adopted_started_event_sha256_by_id"][custom_id]
            or receipt["adopted_completed_event_sha256_by_id"][custom_id] is not None
            or sha256_json(core)
            != receipt["adopted_core_result_sha256_by_id"][custom_id]
            or old_errors
            != receipt["adopted_old_materialization_errors_by_id"][custom_id]
            or new_errors
            != receipt["adopted_new_materialization_errors_by_id"][custom_id]
            or envelope.get("started_event_sha256") != sha256_json(started)
            or envelope.get("core_result_sha256") != sha256_json(core)
            or started.get("base_request_sha256") != sha256_json(base)
            or started.get("effective_request_sha256") != sha256_json(effective)
            or envelope.get("base_request_sha256") != sha256_json(base)
            or envelope.get("effective_request_sha256") != sha256_json(effective)
            or started.get("base_execution_contract_sha256")
            != source_contract_profile.get("base_execution_contract_sha256")
            or envelope.get("base_execution_contract_sha256")
            != source_contract_profile.get("base_execution_contract_sha256")
        ):
            raise ProfileContractError("v5 adoption source exact replay failed")
        projected = _project_v5_adopted_success(envelope, started_event=started)
        if projected_by_id[custom_id] != projected:
            raise ProfileContractError("v5 adoption projection exact replay drifted")
        adopted[custom_id] = projected
    if list(bridge.source_core_rows) != [
        deepcopy(row["core_result"]) for row in bridge.source_envelopes
    ]:
        raise ProfileContractError("v5 adoption source usage rows drifted")

    cal_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_envelopes
    }
    base_by_id = {str(row["custom_id"]): row for row in bridge.calibration_base_requests}
    request_by_id = {str(row["custom_id"]): row for row in bridge.calibration_requests}
    cal_started_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_started_events
    }
    cal_completed_by_id = {
        str(row.get("logical_custom_id") or ""): row
        for row in bridge.calibration_completed_events
    }
    if not (
        len(bridge.calibration_envelopes)
        == len(bridge.calibration_base_requests)
        == len(bridge.calibration_requests)
        == len(bridge.calibration_started_events)
        == len(bridge.calibration_completed_events)
        == len(cal_by_id)
        == len(base_by_id)
        == len(request_by_id)
        == len(cal_started_by_id)
        == len(cal_completed_by_id)
        == 4
        and set(cal_by_id)
        == set(base_by_id)
        == set(request_by_id)
        == set(cal_started_by_id)
        == set(cal_completed_by_id)
        == _expected_v5_calibration_ids()
    ):
        raise ProfileContractError("v5 calibration evidence is not exact 4/4")
    for custom_id, envelope in cal_by_id.items():
        core = envelope.get("core_result")
        errors = (
            ["v5 calibration core is invalid"]
            if not isinstance(core, dict)
            else list(
                bridge.validate_v5_source_core(
                    deepcopy(core), request_by_id[custom_id], True
                )
            )
        )
        started = cal_started_by_id[custom_id]
        completed = cal_completed_by_id[custom_id]
        expected_sequence = int(custom_id.split("-")[3])
        if (
            set(started) != V5_STARTED_EVENT_KEYS
            or set(envelope) != V5_RESULT_ENVELOPE_KEYS
            or set(completed) != V5_COMPLETED_EVENT_KEYS
            or base_by_id[custom_id].get("custom_id") != custom_id
            or request_by_id[custom_id].get("custom_id") != custom_id
            or any(
                row.get("logical_custom_id") != custom_id
                or row.get("kind") != "calibration"
                or row.get("operation_id") != envelope.get("operation_id")
                or row.get("sequence") != expected_sequence
                or row.get("execution_profile_sha256") != V5_PROFILE_SHA256
                for row in (started, envelope, completed)
            )
            or started.get("event_type") != "started"
            or envelope.get("event_type") != "result"
            or completed.get("event_type") != "completed"
            or started.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or completed.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-journal.v5"
            or envelope.get("schema_version")
            != "stt.eval.easy-medium-nonthinking-result.v5"
            or envelope.get("profile_runner_version")
            != V5_PROFILE_RUNNER_VERSION
            or envelope.get("model") != source_profile.get("model")
            or envelope.get("endpoint") != source_profile.get("endpoint")
            or envelope.get("transport") != source_profile.get("transport")
            or envelope.get("sse_completion_policy")
            != source_profile.get("sse_completion_policy")
            or envelope.get("request_overlay")
            != source_profile.get("request_overlay")
            or started.get("base_execution_contract_sha256") is not None
            or envelope.get("base_execution_contract_sha256") is not None
            or envelope.get("started_event_sha256") != sha256_json(started)
            or completed.get("started_event_sha256") != sha256_json(started)
            or completed.get("result_envelope_sha256") != sha256_json(envelope)
            or sha256_json(base_by_id[custom_id])
            != receipt["calibration_base_request_sha256_by_id"][custom_id]
            or sha256_json(request_by_id[custom_id])
            != receipt["calibration_request_sha256_by_id"][custom_id]
            or sha256_json(envelope)
            != receipt["calibration_result_envelope_sha256_by_id"][custom_id]
            or sha256_json(started)
            != receipt["calibration_started_event_sha256_by_id"][custom_id]
            or sha256_json(completed)
            != receipt["calibration_completed_event_sha256_by_id"][custom_id]
            or sha256_json(core)
            != receipt["calibration_core_result_sha256_by_id"][custom_id]
            or envelope.get("core_result_sha256") != sha256_json(core)
            or started.get("base_request_sha256")
            != sha256_json(base_by_id[custom_id])
            or started.get("effective_request_sha256")
            != sha256_json(request_by_id[custom_id])
            or envelope.get("base_request_sha256")
            != sha256_json(base_by_id[custom_id])
            or envelope.get("effective_request_sha256")
            != sha256_json(request_by_id[custom_id])
            or completed.get("result_status") != core.get("status")
            or core.get("status") not in SUCCESS_STATUSES
            or errors
        ):
            raise ProfileContractError("v5 calibration evidence exact replay failed")
    evidence = {
        "evidence_role": "source_execution_precondition_only",
        "source_profile_sha256": V5_PROFILE_SHA256,
        "source_result_envelope_sha256_by_id": deepcopy(
            receipt["calibration_result_envelope_sha256_by_id"]
        ),
        "authorizes_v6_calls": False,
        "bridge_receipt_sha256": claimed,
    }
    evidence["receipt_sha256"] = sha256_json(evidence)
    return adopted, evidence


def _reduce_production_journal(
    *,
    runtime: PackRuntime,
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    validate_core_result: Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]],
) -> dict[str, Any]:
    """重放一个 production pack 的独立 journal 与 response 视图。"""

    validate_profile_against_pack(
        execution_profile,
        kind=runtime.kind,
        pack=runtime.pack,
    )
    legacy_replay = _validate_runtime(runtime)
    legacy_success = legacy_replay["successful_by_id"]
    selection = _selection(runtime.pack)
    target_custom_ids = selection.target_custom_ids
    kind = runtime.kind
    target_ids = set(target_custom_ids)
    predecessor_success: dict[str, dict[str, Any]] = {}
    predecessor_failed: list[dict[str, Any]] = []
    if runtime.predecessor_bridge is not None:
        bridge = runtime.predecessor_bridge
        receipt_view = deepcopy(bridge.receipt)
        receipt_sha = receipt_view.pop("receipt_sha256", None)
        if (
            execution_profile["predecessor_policy"]["bridge_receipt_sha256"]
            != receipt_sha
            or receipt_sha != sha256_json(receipt_view)
            or bridge.receipt.get("predecessor_profile_sha256")
            != PREDECESSOR_PROFILE_SHA256
        ):
            raise ProfileContractError("runtime predecessor bridge/profile mismatch")
        receipt_success = bridge.receipt.get(
            "bridged_success_envelope_sha256_by_id"
        )
        receipt_failed = bridge.receipt.get(
            "failed_evidence_envelope_sha256_by_id"
        )
        if not isinstance(receipt_success, Mapping) or not isinstance(
            receipt_failed, Mapping
        ):
            raise ProfileContractError("predecessor receipt source maps are invalid")
        expected_success = {
            str(custom_id): str(digest)
            for custom_id, digest in receipt_success.items()
            if str(custom_id) in target_ids
        }
        expected_failed = {
            str(custom_id): str(digest)
            for custom_id, digest in receipt_failed.items()
            if str(custom_id) in target_ids
        }
        source_success_ids = [
            str(row.get("logical_custom_id") or "")
            for row in bridge.success_envelopes
        ]
        projected_success_ids = [
            str(row.get("custom_id") or "") for row in bridge.success_rows
        ]
        failed_ids = [
            str(row.get("logical_custom_id") or "")
            for row in bridge.failed_envelopes
        ]
        if (
            len(source_success_ids) != len(set(source_success_ids))
            or len(projected_success_ids) != len(set(projected_success_ids))
            or len(failed_ids) != len(set(failed_ids))
            or set(source_success_ids) != set(expected_success)
            or set(projected_success_ids) != set(expected_success)
            or set(failed_ids) != set(expected_failed)
        ):
            raise ProfileContractError("predecessor bridge source IDs are missing/duplicate")
        source_success_by_id = {
            str(row["logical_custom_id"]): row for row in bridge.success_envelopes
        }
        projected_success_by_id = {
            str(row["custom_id"]): row for row in bridge.success_rows
        }
        completed_by_operation = {
            str(row.get("operation_id") or ""): row
            for row in bridge.completed_events
            if row.get("event_type") == "completed"
        }
        for custom_id, source in source_success_by_id.items():
            core = source.get("core_result")
            completed = completed_by_operation.get(
                str(source.get("operation_id") or "")
            )
            base = _base_request_by_id(runtime.pack, custom_id)
            predecessor_effective = deepcopy(base)
            predecessor_effective["request"]["enable_thinking"] = False
            validation_errors = (
                ["predecessor source core is invalid"]
                if not isinstance(core, Mapping)
                else [
                    str(error)
                    for error in bridge.validate_predecessor_core(
                        deepcopy(dict(core)), predecessor_effective, False
                    )
                    if str(error)
                ]
            )
            if (
                sha256_json(source) != expected_success[custom_id]
                or source.get("kind") != kind
                or not isinstance(core, Mapping)
                or source.get("core_result_sha256") != sha256_json(core)
                or not isinstance(completed, Mapping)
                or completed.get("result_envelope_sha256") != sha256_json(source)
                or validation_errors
            ):
                raise ProfileContractError("predecessor success source replay is invalid")
            expected_projection = _project_predecessor_success(source)
            if projected_success_by_id[custom_id] != expected_projection:
                raise ProfileContractError("predecessor success projection exact replay drifted")
            predecessor_success[custom_id] = deepcopy(expected_projection)
        for custom_id, source in zip(failed_ids, bridge.failed_envelopes, strict=True):
            if sha256_json(source) != expected_failed[custom_id]:
                raise ProfileContractError("predecessor failed evidence source drifted")
        predecessor_failed = [deepcopy(row) for row in bridge.failed_envelopes]
        expected_source_cores = [
            deepcopy(row["core_result"])
            for row in (*bridge.success_envelopes, *bridge.failed_envelopes)
        ]
        if list(bridge.source_core_rows) != expected_source_cores:
            raise ProfileContractError("predecessor source usage/core rows drifted")
    v4_adopted_success, v4_source_calibration_evidence_receipt = _validated_v4_adoption(
        runtime, execution_profile
    )
    if set(v4_adopted_success) & (set(legacy_success) | set(predecessor_success)):
        raise ProfileContractError("v4 adoption overlaps an earlier authority")
    v5_adopted_success, v5_source_calibration_evidence_receipt = _validated_v5_adoption(
        runtime, execution_profile
    )
    if set(v5_adopted_success) & (
        set(legacy_success) | set(predecessor_success) | set(v4_adopted_success)
    ):
        raise ProfileContractError("v5 adoption overlaps an earlier authority")
    for row in journal_rows:
        if row.get("kind") not in ALL_EVENT_KINDS or row.get("event_type") not in {
            "started",
            "completed",
        }:
            raise ProfileContractError("unknown/foreign journal event")
        if row.get("execution_profile_sha256") != execution_profile.get(
            "execution_profile_sha256"
        ):
            raise ProfileContractError("foreign execution profile journal event")
    for row in response_rows:
        if row.get("kind") not in ALL_EVENT_KINDS or row.get("event_type") != "result":
            raise ProfileContractError("unknown/foreign response event")
        if row.get("execution_profile_sha256") != execution_profile.get(
            "execution_profile_sha256"
        ):
            raise ProfileContractError("foreign execution profile result")

    starts: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    started_operation_by_custom_id: dict[str, str] = {}
    for event in journal_rows:
        if event.get("kind") != kind:
            continue
        custom_id = str(event.get("logical_custom_id") or "")
        if custom_id not in target_ids:
            raise ProfileContractError("journal event targets a non-target id")
        if (
            custom_id in legacy_success
            or custom_id in predecessor_success
            or custom_id in v4_adopted_success
            or custom_id in v5_adopted_success
        ):
            raise ProfileContractError("journal event targets an id that already succeeded")
        operation_id = str(event.get("operation_id") or "")
        if event.get("event_type") == "started":
            if operation_id in starts:
                raise ProfileContractError("duplicate journal operation event")
            prior_operation = started_operation_by_custom_id.get(custom_id)
            if prior_operation is not None and prior_operation != operation_id:
                raise ProfileContractError("duplicate provider operation for one target id")
            started_operation_by_custom_id[custom_id] = operation_id
            starts[operation_id] = deepcopy(event)
        else:
            if operation_id in completions:
                raise ProfileContractError("duplicate journal operation event")
            if operation_id not in starts:
                raise ProfileContractError("completed event precedes/misses started fence")
            completions[operation_id] = deepcopy(event)
    result_operation_by_custom_id: dict[str, str] = {}
    for envelope in response_rows:
        if envelope.get("kind") != kind:
            continue
        custom_id = str(envelope.get("logical_custom_id") or "")
        if custom_id not in target_ids:
            raise ProfileContractError("result targets a non-target id")
        if (
            custom_id in legacy_success
            or custom_id in predecessor_success
            or custom_id in v4_adopted_success
            or custom_id in v5_adopted_success
        ):
            raise ProfileContractError("result targets an id that already succeeded")
        operation_id = str(envelope.get("operation_id") or "")
        if operation_id in results:
            raise ProfileContractError("duplicate result for one provider operation")
        prior_operation = result_operation_by_custom_id.get(custom_id)
        if prior_operation is not None and prior_operation != operation_id:
            raise ProfileContractError("duplicate provider result for one target id")
        result_operation_by_custom_id[custom_id] = operation_id
        results[operation_id] = deepcopy(envelope)

    for operation_id, started in starts.items():
        custom_id = str(started["logical_custom_id"])
        base = _base_request_by_id(runtime.pack, custom_id)
        effective = effective_request_row(base, execution_profile, kind=kind)
        errors = _validate_started(
            started,
            kind=kind,
            effective_row=effective,
            execution_profile=execution_profile,
        )
        if errors:
            raise ProfileContractError("; ".join(errors))
        envelope = results.get(operation_id)
        completed = completions.get(operation_id)
        if envelope is None and completed is not None:
            raise ProfileContractError("completed event has no result")
        if envelope is not None:
            errors = validate_result_envelope(
                envelope,
                kind=kind,
                request_row=effective,
                execution_profile=execution_profile,
                started_event=started,
                validate_core_result=validate_core_result,
            )
            if errors:
                raise ProfileContractError("; ".join(errors))
        if completed is not None:
            errors = _validate_completed(completed, started=started, result=envelope or {})
            if errors:
                raise ProfileContractError("; ".join(errors))
    if set(results) - set(starts):
        raise ProfileContractError("result has no started fence")
    if set(completions) - set(starts):
        raise ProfileContractError("completed event has no started fence")

    new_envelope_by_id = {
        str(envelope["logical_custom_id"]): envelope
        for operation_id, envelope in results.items()
        if operation_id in completions
        and envelope["core_result"].get("status") == "succeeded"
    }
    authoritative: dict[str, dict[str, Any]] = {}
    for custom_id in target_custom_ids:
        if custom_id in legacy_success:
            authoritative[custom_id] = deepcopy(legacy_success[custom_id])
        elif custom_id in predecessor_success:
            authoritative[custom_id] = deepcopy(predecessor_success[custom_id])
        elif custom_id in v4_adopted_success:
            authoritative[custom_id] = deepcopy(v4_adopted_success[custom_id])
        elif custom_id in v5_adopted_success:
            authoritative[custom_id] = deepcopy(v5_adopted_success[custom_id])
        elif custom_id in new_envelope_by_id:
            envelope = new_envelope_by_id[custom_id]
            authoritative[custom_id] = _project_authoritative_result(
                envelope, execution_profile
            )

    uncertain = sorted(
        str(started["logical_custom_id"])
        for operation_id, started in starts.items()
        if operation_id not in results
    )
    reconcile = sorted(
        str(starts[operation_id]["logical_custom_id"])
        for operation_id in results
        if operation_id not in completions
    )
    failed = sorted(
        str(envelope["logical_custom_id"])
        for operation_id, envelope in results.items()
        if operation_id in completions
        and envelope["core_result"].get("status") == "failed"
    )
    pending = [
        custom_id
        for custom_id in target_custom_ids
        if custom_id not in authoritative
        and custom_id not in uncertain
        and custom_id not in reconcile
        and custom_id not in failed
        and custom_id not in legacy_replay["materialization_blocked_custom_ids"]
    ]
    return {
        "kind": kind,
        "authoritative_success_by_id": authoritative,
        "legacy_success_by_id": {
            custom_id: legacy_success[custom_id]
            for custom_id in target_custom_ids
            if custom_id in legacy_success
        },
        "new_success_custom_ids": sorted(
            set(authoritative)
            - set(legacy_success)
            - set(predecessor_success)
            - set(v4_adopted_success)
            - set(v5_adopted_success)
        ),
        "predecessor_success_custom_ids": sorted(predecessor_success),
        "v4_adopted_success_custom_ids": sorted(v4_adopted_success),
        "v4_source_calibration_evidence_receipt": (
            v4_source_calibration_evidence_receipt
        ),
        "v5_adopted_success_custom_ids": sorted(v5_adopted_success),
        "v5_source_calibration_evidence_receipt": (
            v5_source_calibration_evidence_receipt
        ),
        "predecessor_failed_envelopes": predecessor_failed,
        "pending_custom_ids": pending,
        "failed_custom_ids": failed,
        "uncertain_side_effect_custom_ids": uncertain,
        "reconcile_custom_ids": reconcile,
        "legacy_materialization_blocked_custom_ids": legacy_replay[
            "materialization_blocked_custom_ids"
        ],
        "started_by_operation_id": starts,
        "result_by_operation_id": results,
        "completed_by_operation_id": completions,
    }


def _calibration_state(
    *,
    template: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    validate_core_result: Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]],
) -> dict[str, Any]:
    requests = build_calibration_requests(template, execution_profile)
    by_id = {str(row["custom_id"]): row for row in requests}
    starts: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    operation_by_custom_id: dict[str, str] = {}
    for event in journal_rows:
        if event.get("kind") != "calibration":
            continue
        if event.get("execution_profile_sha256") != execution_profile.get(
            "execution_profile_sha256"
        ):
            raise ProfileContractError("foreign calibration journal event")
        custom_id = str(event.get("logical_custom_id") or "")
        if custom_id not in by_id:
            raise ProfileContractError("foreign/legacy calibration id")
        operation_id = str(event.get("operation_id") or "")
        if event.get("event_type") not in {"started", "completed"}:
            raise ProfileContractError("invalid/duplicate calibration journal event")
        if event.get("event_type") == "started":
            if operation_id in starts:
                raise ProfileContractError("invalid/duplicate calibration journal event")
            prior = operation_by_custom_id.get(custom_id)
            if prior is not None and prior != operation_id:
                raise ProfileContractError("duplicate calibration provider operation")
            operation_by_custom_id[custom_id] = operation_id
            starts[operation_id] = deepcopy(event)
        else:
            if operation_id in completions:
                raise ProfileContractError("invalid/duplicate calibration journal event")
            if operation_id not in starts:
                raise ProfileContractError(
                    "calibration completed precedes/misses started fence"
                )
            completions[operation_id] = deepcopy(event)
    result_operation_by_custom_id: dict[str, str] = {}
    for envelope in response_rows:
        if envelope.get("kind") != "calibration":
            raise ProfileContractError("calibration response file contains a foreign kind")
        custom_id = str(envelope.get("logical_custom_id") or "")
        if custom_id not in by_id:
            raise ProfileContractError("foreign/legacy calibration result id")
        operation_id = str(envelope.get("operation_id") or "")
        if operation_id in results:
            raise ProfileContractError("duplicate calibration result")
        prior = result_operation_by_custom_id.get(custom_id)
        if prior is not None and prior != operation_id:
            raise ProfileContractError("duplicate calibration provider result")
        result_operation_by_custom_id[custom_id] = operation_id
        results[operation_id] = deepcopy(envelope)
    for operation_id, started in starts.items():
        row = by_id[str(started["logical_custom_id"])]
        errors = _validate_started(
            started,
            kind="calibration",
            effective_row=row,
            execution_profile=execution_profile,
        )
        if errors:
            raise ProfileContractError("; ".join(errors))
        envelope = results.get(operation_id)
        completed = completions.get(operation_id)
        if envelope is None and completed is not None:
            raise ProfileContractError("calibration completed has no result")
        if envelope is not None:
            errors = validate_result_envelope(
                envelope,
                kind="calibration",
                request_row=row,
                execution_profile=execution_profile,
                started_event=started,
                validate_core_result=validate_core_result,
            )
            if errors:
                raise ProfileContractError("; ".join(errors))
        if completed is not None:
            errors = _validate_completed(completed, started=started, result=envelope or {})
            if errors:
                raise ProfileContractError("; ".join(errors))
    if set(results) - set(starts) or set(completions) - set(starts):
        raise ProfileContractError("calibration result/completion has no started fence")
    succeeded = {
        str(envelope["logical_custom_id"])
        for operation_id, envelope in results.items()
        if operation_id in completions
        and envelope["core_result"].get("status") in SUCCESS_STATUSES
    }
    failed = {
        str(envelope["logical_custom_id"])
        for operation_id, envelope in results.items()
        if operation_id in completions
        and envelope["core_result"].get("status") == "failed"
    }
    uncertain = {
        str(event["logical_custom_id"])
        for operation_id, event in starts.items()
        if operation_id not in results
    }
    reconcile = {
        str(starts[operation_id]["logical_custom_id"])
        for operation_id in results
        if operation_id not in completions
    }
    return {
        "requests": requests,
        "succeeded_custom_ids": sorted(succeeded),
        "failed_custom_ids": sorted(failed),
        "uncertain_side_effect_custom_ids": sorted(uncertain),
        "reconcile_custom_ids": sorted(reconcile),
        "started_by_operation_id": starts,
        "result_by_operation_id": results,
        "completed_by_operation_id": completions,
    }


def _fresh_calibration_receipt(
    *,
    template: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]],
    validate_core_result: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ],
) -> dict[str, Any]:
    state = _calibration_state(
        template=template,
        execution_profile=execution_profile,
        journal_rows=journal_rows,
        response_rows=calibration_response_rows,
        validate_core_result=validate_core_result,
    )
    expected_ids = [str(row["custom_id"]) for row in state["requests"]]
    if (
        state["succeeded_custom_ids"] != sorted(expected_ids)
        or state["failed_custom_ids"]
        or state["uncertain_side_effect_custom_ids"]
        or state["reconcile_custom_ids"]
        or len(state["completed_by_operation_id"]) != 4
    ):
        raise ProfileContractError(
            "new execution profile requires exact fresh completed calibration 4/4"
        )
    envelope_sha_by_id: dict[str, str] = {}
    for operation_id, envelope in state["result_by_operation_id"].items():
        if operation_id not in state["completed_by_operation_id"]:
            continue
        envelope_sha_by_id[str(envelope["logical_custom_id"])] = sha256_json(envelope)
    if set(envelope_sha_by_id) != set(expected_ids):
        raise ProfileContractError("fresh calibration result receipt is incomplete")
    receipt = {
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "required_case_count": 4,
        "succeeded_custom_ids": expected_ids,
        "result_envelope_sha256_by_id": {
            custom_id: envelope_sha_by_id[custom_id] for custom_id in expected_ids
        },
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def reduce_journal(
    *,
    runtime: PackRuntime,
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]] = (),
    validate_core_result: Callable[
        [dict[str, Any], dict[str, Any], bool], Sequence[str]
    ],
) -> dict[str, Any]:
    """发布侧权威重放；v5 provider success必须由fresh v5 4/4门控。"""

    state = _reduce_production_journal(
        runtime=runtime,
        execution_profile=execution_profile,
        journal_rows=journal_rows,
        response_rows=response_rows,
        validate_core_result=validate_core_result,
    )
    calibration_receipt = None
    if state["new_success_custom_ids"]:
        calibration_receipt = _fresh_calibration_receipt(
            template=runtime.pack.requests[0],
            execution_profile=execution_profile,
            journal_rows=journal_rows,
            calibration_response_rows=calibration_response_rows,
            validate_core_result=validate_core_result,
        )
    return {**state, "fresh_calibration_receipt": calibration_receipt}


def _append_and_track(
    rows: list[dict[str, Any]], value: dict[str, Any], append: Callable[[dict[str, Any]], None]
) -> None:
    append(value)
    rows.append(value)


def preflight_execution(
    *,
    runtimes: Mapping[str, PackRuntime],
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]],
    validate_core_result_by_kind: Mapping[
        str, Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]]
    ],
) -> dict[str, Any]:
    """双pack全局只读预检；任何provider调用前必须完整通过。"""

    if set(runtimes) != set(KINDS) or set(validate_core_result_by_kind) != set(KINDS):
        raise ProfileContractError("global preflight requires exact blocker+contract runtimes")
    # 从两个physical pack重建完整profile并比较，封闭跨pack co-rehash。
    for kind in KINDS:
        runtime = runtimes[kind]
        if runtime.kind != kind:
            raise ProfileContractError("global preflight runtime kind mismatch")
        validate_profile_against_pack(
            execution_profile,
            kind=kind,
            pack=runtime.pack,
        )
    states = {
        kind: _reduce_production_journal(
            runtime=runtimes[kind],
            execution_profile=execution_profile,
            journal_rows=journal_rows,
            response_rows=response_rows,
            validate_core_result=validate_core_result_by_kind[kind],
        )
        for kind in KINDS
    }
    calibration = _calibration_state(
        template=runtimes["blocker"].pack.requests[0],
        execution_profile=execution_profile,
        journal_rows=journal_rows,
        response_rows=calibration_response_rows,
        validate_core_result=validate_core_result_by_kind["blocker"],
    )
    unknown = {
        kind: states[kind]["uncertain_side_effect_custom_ids"] for kind in KINDS
    }
    unknown["calibration"] = calibration["uncertain_side_effect_custom_ids"]
    reconcile = {kind: states[kind]["reconcile_custom_ids"] for kind in KINDS}
    reconcile["calibration"] = calibration["reconcile_custom_ids"]
    return {
        "states": states,
        "calibration": calibration,
        "unknown_side_effect_by_kind": unknown,
        "reconcile_by_kind": reconcile,
        "api_safe": not any(unknown.values()) and not any(reconcile.values()),
    }


def drive_pack(
    *,
    runtime: PackRuntime,
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]],
    append_journal: Callable[[dict[str, Any]], None],
    append_response: Callable[[dict[str, Any]], None],
    append_calibration_response: Callable[[dict[str, Any]], None],
    execute_one: Callable[[dict[str, Any], bool], dict[str, Any]],
    now: Callable[[], str] | None,
    validate_core_result: Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]],
) -> dict[str, Any]:
    """先完成fresh v5 4/4，再执行一个pack的未完成E/M ID。"""

    clock = now or (lambda: datetime.now(UTC).isoformat())
    journals = [deepcopy(row) for row in journal_rows]
    responses = [deepcopy(row) for row in response_rows]
    calibrations = [deepcopy(row) for row in calibration_response_rows]
    template = runtime.pack.requests[0]

    # 全部只读重放必须发生在任何外部调用前。production 的未知副作用优先级
    # 高于calibration，避免一个旧started fence触发新的付费校准调用。
    state = _reduce_production_journal(
        runtime=runtime,
        execution_profile=execution_profile,
        journal_rows=journals,
        response_rows=responses,
        validate_core_result=validate_core_result,
    )
    calibration = _calibration_state(
        template=template,
        execution_profile=execution_profile,
        journal_rows=journals,
        response_rows=calibrations,
        validate_core_result=validate_core_result,
    )
    if state["uncertain_side_effect_custom_ids"]:
        return {
            "status": "fail_stop_unknown_side_effect",
            "phase": state["kind"],
            "custom_ids": state["uncertain_side_effect_custom_ids"],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if calibration["uncertain_side_effect_custom_ids"]:
        return {
            "status": "fail_stop_unknown_side_effect",
            "phase": "calibration",
            "custom_ids": calibration["uncertain_side_effect_custom_ids"],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if state["reconcile_custom_ids"]:
        custom_id = state["reconcile_custom_ids"][0]
        operation_id = next(
            operation
            for operation, event in state["started_by_operation_id"].items()
            if event["logical_custom_id"] == custom_id
        )
        completed = build_completed_event(
            started_event=state["started_by_operation_id"][operation_id],
            result_envelope=state["result_by_operation_id"][operation_id],
        )
        _append_and_track(journals, completed, append_journal)
        return {
            "status": "reconciled_result_without_api",
            "phase": state["kind"],
            "custom_ids": [custom_id],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if calibration["reconcile_custom_ids"]:
        custom_id = calibration["reconcile_custom_ids"][0]
        operation_id = next(
            operation
            for operation, event in calibration["started_by_operation_id"].items()
            if event["logical_custom_id"] == custom_id
        )
        completed = build_completed_event(
            started_event=calibration["started_by_operation_id"][operation_id],
            result_envelope=calibration["result_by_operation_id"][operation_id],
        )
        _append_and_track(journals, completed, append_journal)
        return {
            "status": "reconciled_result_without_api",
            "phase": "calibration",
            "custom_ids": [custom_id],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if calibration["failed_custom_ids"]:
        return {
            "status": "calibration_failed",
            "custom_ids": calibration["failed_custom_ids"],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if state["failed_custom_ids"]:
        return {
            "status": "stopped_on_failure",
            "phase": state["kind"],
            "custom_ids": state["failed_custom_ids"],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if state["legacy_materialization_blocked_custom_ids"]:
        return {
            "status": "blocked_by_legacy_materialization_state",
            "phase": state["kind"],
            "custom_ids": state["legacy_materialization_blocked_custom_ids"],
            "api_calls": 0,
            "hard_calls": 0,
        }
    if not state["pending_custom_ids"]:
        return {
            "status": "succeeded",
            "phase": state["kind"],
            "case_success_count": len(state["authoritative_success_by_id"]),
            "case_expected_count": len(_selection(runtime.pack).target_custom_ids),
            "pending_custom_ids": [],
            "api_calls": 0,
            "hard_calls": 0,
        }

    api_calls = 0
    completed_calibration = set(calibration["succeeded_custom_ids"])
    for row in calibration["requests"]:
        custom_id = str(row["custom_id"])
        if custom_id in completed_calibration:
            continue
        started = build_started_event(
            kind="calibration",
            logical_custom_id=custom_id,
            effective_row=row,
            execution_profile=execution_profile,
            started_at=clock(),
        )
        _append_and_track(journals, started, append_journal)
        core_result = execute_one(deepcopy(row), True)
        api_calls += 1
        envelope = wrap_result(
            kind="calibration",
            logical_custom_id=custom_id,
            effective_row=row,
            core_result=core_result,
            execution_profile=execution_profile,
            started_event=started,
        )
        _append_and_track(calibrations, envelope, append_calibration_response)
        errors = validate_result_envelope(
            envelope,
            kind="calibration",
            request_row=row,
            execution_profile=execution_profile,
            started_event=started,
            validate_core_result=validate_core_result,
        )
        if errors:
            raise ProfileContractError("; ".join(errors))
        completed = build_completed_event(started_event=started, result_envelope=envelope)
        _append_and_track(journals, completed, append_journal)
        if core_result.get("status") not in SUCCESS_STATUSES:
            return {
                "status": "calibration_failed",
                "custom_ids": [custom_id],
                "api_calls": api_calls,
                "hard_calls": 0,
            }

    calibration = _calibration_state(
        template=template,
        execution_profile=execution_profile,
        journal_rows=journals,
        response_rows=calibrations,
        validate_core_result=validate_core_result,
    )
    if len(calibration["succeeded_custom_ids"]) != 4:
        raise ProfileContractError("v5 fresh calibration is not exact 4/4")

    state = _reduce_production_journal(
        runtime=runtime,
        execution_profile=execution_profile,
        journal_rows=journals,
        response_rows=responses,
        validate_core_result=validate_core_result,
    )
    if (
        state["uncertain_side_effect_custom_ids"]
        or state["reconcile_custom_ids"]
        or state["failed_custom_ids"]
    ):
        raise ProfileContractError("production state changed during calibration replay")

    kind = str(state["kind"])
    for custom_id in list(state["pending_custom_ids"]):
        base = _base_request_by_id(runtime.pack, custom_id)
        effective = effective_request_row(base, execution_profile, kind=kind)
        started = build_started_event(
            kind=kind,
            logical_custom_id=custom_id,
            effective_row=effective,
            execution_profile=execution_profile,
            started_at=clock(),
        )
        _append_and_track(journals, started, append_journal)
        core_result = execute_one(deepcopy(effective), False)
        api_calls += 1
        envelope = wrap_result(
            kind=kind,
            logical_custom_id=custom_id,
            effective_row=effective,
            core_result=core_result,
            execution_profile=execution_profile,
            started_event=started,
        )
        _append_and_track(responses, envelope, append_response)
        errors = validate_result_envelope(
            envelope,
            kind=kind,
            request_row=effective,
            execution_profile=execution_profile,
            started_event=started,
            validate_core_result=validate_core_result,
        )
        if errors:
            raise ProfileContractError("; ".join(errors))
        completed = build_completed_event(started_event=started, result_envelope=envelope)
        _append_and_track(journals, completed, append_journal)
        if core_result.get("status") != "succeeded":
            return {
                "status": "stopped_on_failure",
                "phase": kind,
                "custom_ids": [custom_id],
                "api_calls": api_calls,
                "hard_calls": 0,
            }
        state = _reduce_production_journal(
            runtime=runtime,
            execution_profile=execution_profile,
            journal_rows=journals,
            response_rows=responses,
            validate_core_result=validate_core_result,
        )
        if custom_id not in state["authoritative_success_by_id"]:
            raise ProfileContractError("new result did not become authoritative after replay")

    final = _reduce_production_journal(
        runtime=runtime,
        execution_profile=execution_profile,
        journal_rows=journals,
        response_rows=responses,
        validate_core_result=validate_core_result,
    )
    return {
        "status": "succeeded" if not final["pending_custom_ids"] else "incomplete",
        "phase": final["kind"],
        "case_success_count": len(final["authoritative_success_by_id"]),
        "case_expected_count": len(_selection(runtime.pack).target_custom_ids),
        "pending_custom_ids": final["pending_custom_ids"],
        "api_calls": api_calls,
        "hard_calls": 0,
    }


def _usage_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in totals:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key] += value
    return totals


def _add_usage(*values: Mapping[str, int]) -> dict[str, int]:
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _attempt_totals(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(row["attempt"])
        for row in rows
        if type(row.get("attempt")) is int and int(row["attempt"]) >= 0
    )


def build_scope_summary(
    *,
    runtimes: Mapping[str, PackRuntime],
    execution_profile: Mapping[str, Any],
    response_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]],
    journal_rows: Sequence[dict[str, Any]],
    validate_core_result_by_kind: Mapping[
        str, Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]]
    ],
) -> dict[str, Any]:
    if set(runtimes) != set(KINDS) or set(validate_core_result_by_kind) != set(KINDS):
        raise ProfileContractError(
            "summary requires exact blocker+contract runtimes/validators"
        )
    states = {
        kind: reduce_journal(
            runtime=runtimes[kind],
            execution_profile=execution_profile,
            journal_rows=journal_rows,
            response_rows=response_rows,
            calibration_response_rows=calibration_response_rows,
            validate_core_result=validate_core_result_by_kind[kind],
        )
        for kind in KINDS
    }
    calibration_state = _calibration_state(
        template=runtimes["blocker"].pack.requests[0],
        execution_profile=execution_profile,
        journal_rows=journal_rows,
        response_rows=calibration_response_rows,
        validate_core_result=validate_core_result_by_kind["blocker"],
    )
    target_legacy_rows: list[dict[str, Any]] = []
    for runtime in runtimes.values():
        targets = set(_selection(runtime.pack).target_custom_ids)
        target_legacy_rows.extend(
            row
            for row in runtime.legacy_rows
            if str(row.get("custom_id") or "") in targets
        )
    new_core_rows = [
        deepcopy(dict(row["core_result"]))
        for row in response_rows
        if row.get("kind") in KINDS
    ]
    calibration_core_rows = [
        deepcopy(dict(row["core_result"])) for row in calibration_response_rows
    ]
    predecessor_core_by_sha: dict[str, dict[str, Any]] = {}
    predecessor_calibration_by_sha: dict[str, dict[str, Any]] = {}
    v4_core_by_sha: dict[str, dict[str, Any]] = {}
    v4_calibration_by_sha: dict[str, dict[str, Any]] = {}
    v5_core_by_sha: dict[str, dict[str, Any]] = {}
    v5_calibration_by_sha: dict[str, dict[str, Any]] = {}
    for runtime in runtimes.values():
        bridge = runtime.predecessor_bridge
        if bridge is not None:
            for row in bridge.source_core_rows:
                predecessor_core_by_sha.setdefault(sha256_json(row), deepcopy(row))
            for row in bridge.calibration_core_rows:
                predecessor_calibration_by_sha.setdefault(
                    sha256_json(row), deepcopy(row)
                )
        v4_bridge = runtime.v4_adoption_bridge
        if v4_bridge is not None:
            for row in v4_bridge.source_core_rows:
                v4_core_by_sha.setdefault(sha256_json(row), deepcopy(row))
            for envelope in v4_bridge.calibration_envelopes:
                core = deepcopy(dict(envelope["core_result"]))
                v4_calibration_by_sha.setdefault(sha256_json(core), core)
        v5_bridge = runtime.v5_adoption_bridge
        if v5_bridge is not None:
            for row in v5_bridge.source_core_rows:
                v5_core_by_sha.setdefault(sha256_json(row), deepcopy(row))
            for envelope in v5_bridge.calibration_envelopes:
                core = deepcopy(dict(envelope["core_result"]))
                v5_calibration_by_sha.setdefault(sha256_json(core), core)
    predecessor_core_rows = list(predecessor_core_by_sha.values())
    predecessor_calibration_rows = list(predecessor_calibration_by_sha.values())
    v4_core_rows = list(v4_core_by_sha.values())
    v4_calibration_rows = list(v4_calibration_by_sha.values())
    v5_core_rows = list(v5_core_by_sha.values())
    v5_calibration_rows = list(v5_calibration_by_sha.values())
    legacy_usage = _usage_totals(target_legacy_rows)
    new_usage = _usage_totals(new_core_rows)
    calibration_usage = _usage_totals(calibration_core_rows)
    predecessor_usage = _usage_totals(predecessor_core_rows)
    predecessor_calibration_usage = _usage_totals(predecessor_calibration_rows)
    v4_usage = _usage_totals(v4_core_rows)
    v4_calibration_usage = _usage_totals(v4_calibration_rows)
    v5_usage = _usage_totals(v5_core_rows)
    v5_calibration_usage = _usage_totals(v5_calibration_rows)
    legacy_profile_counts: dict[str, int] = {}
    for state in states.values():
        for row in state["legacy_success_by_id"].values():
            runner_version = str(row.get("runner_version") or "unknown")
            legacy_profile_counts[runner_version] = (
                legacy_profile_counts.get(runner_version, 0) + 1
            )
    profile_counts = {
        "legacy_by_runner_version": legacy_profile_counts,
        "new_execution_profile": sum(
            len(state["new_success_custom_ids"]) for state in states.values()
        ),
        "predecessor_v1_success": sum(
            len(state["predecessor_success_custom_ids"])
            for state in states.values()
        ),
        "v4_adopted_success": sum(
            len(state["v4_adopted_success_custom_ids"])
            for state in states.values()
        ),
        "v5_adopted_success": sum(
            len(state["v5_adopted_success_custom_ids"])
            for state in states.values()
        ),
        "fresh_calibration": len(calibration_state["succeeded_custom_ids"]),
    }
    total_expected = sum(
        len(_selection(runtime.pack).target_custom_ids)
        for runtime in runtimes.values()
    )
    total_success = sum(len(state["authoritative_success_by_id"]) for state in states.values())
    any_new_success = any(state["new_success_custom_ids"] for state in states.values())
    calibration_ready = len(calibration_state["succeeded_custom_ids"]) == 4
    unknown_started_count = sum(
        len(state["uncertain_side_effect_custom_ids"]) for state in states.values()
    ) + len(calibration_state["uncertain_side_effect_custom_ids"])
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": (
            "ready"
            if total_success == total_expected
            and (not any_new_success or calibration_ready)
            else "incomplete"
        ),
        "execution_profile_id": PROFILE_ID,
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "scope_id": execution_profile["scope_profile"]["scope_id"],
        "hard_reference_only": True,
        "promotable": False,
        "baseline_registration_allowed": False,
        "canonical_promotion_allowed": False,
        "case_success_count": total_success,
        "case_expected_count": total_expected,
        "profile_counts": profile_counts,
        "legacy_usage": legacy_usage,
        "predecessor_v1_case_usage": predecessor_usage,
        "predecessor_v1_calibration_usage": predecessor_calibration_usage,
        "v4_adopted_case_usage": v4_usage,
        "v4_source_calibration_usage": v4_calibration_usage,
        "v5_adopted_case_usage": v5_usage,
        "v5_source_calibration_usage": v5_calibration_usage,
        "new_profile_case_usage": new_usage,
        "new_profile_calibration_usage": calibration_usage,
        "usage": _add_usage(
            legacy_usage,
            predecessor_usage,
            predecessor_calibration_usage,
            v4_usage,
            v4_calibration_usage,
            v5_usage,
            v5_calibration_usage,
            new_usage,
            calibration_usage,
        ),
        "usage_is_lower_bound": unknown_started_count > 0 or any(
            row.get("usage_is_lower_bound") is True
            for row in [
                *target_legacy_rows,
                *predecessor_core_rows,
                *predecessor_calibration_rows,
                *v4_core_rows,
                *v4_calibration_rows,
                *v5_core_rows,
                *v5_calibration_rows,
                *new_core_rows,
                *calibration_core_rows,
            ]
        ),
        "recorded_api_attempts": _attempt_totals(
            [
                *target_legacy_rows,
                *predecessor_core_rows,
                *predecessor_calibration_rows,
                *v4_core_rows,
                *v4_calibration_rows,
                *v5_core_rows,
                *v5_calibration_rows,
                *new_core_rows,
                *calibration_core_rows,
            ]
        )
        + unknown_started_count,
        "unknown_started_operation_count": unknown_started_count,
        "hard_calls": 0,
        "hard_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "blocker": {
            "success_count": len(states["blocker"]["authoritative_success_by_id"]),
            "expected_count": len(
                _selection(runtimes["blocker"].pack).target_custom_ids
            ),
            "pending_custom_ids": states["blocker"]["pending_custom_ids"],
        },
        "contract": {
            "success_count": len(states["contract"]["authoritative_success_by_id"]),
            "expected_count": len(
                _selection(runtimes["contract"].pack).target_custom_ids
            ),
            "pending_custom_ids": states["contract"]["pending_custom_ids"],
        },
    }


def combined_authoritative_view(
    *,
    runtime: PackRuntime,
    execution_profile: Mapping[str, Any],
    journal_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    calibration_response_rows: Sequence[dict[str, Any]] = (),
    validate_core_result: Callable[[dict[str, Any], dict[str, Any], bool], Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """供 merge、validate 与 QA 共享的唯一 combined replay 入口。"""

    state = reduce_journal(
        runtime=runtime,
        execution_profile=execution_profile,
        journal_rows=journal_rows,
        response_rows=response_rows,
        calibration_response_rows=calibration_response_rows,
        validate_core_result=validate_core_result,
    )
    receipt = {
        "execution_profile_id": PROFILE_ID,
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "legacy_execution_contract_sha256": runtime.pack.base_execution_contract_sha256,
        "legacy_success_count": len(state["legacy_success_by_id"]),
        "predecessor_success_count": len(
            state["predecessor_success_custom_ids"]
        ),
        "predecessor_bridge_receipt_sha256": execution_profile[
            "predecessor_policy"
        ].get("bridge_receipt_sha256"),
        "v4_adoption_success_count": len(
            state["v4_adopted_success_custom_ids"]
        ),
        "v4_adopted_success_custom_ids": state[
            "v4_adopted_success_custom_ids"
        ],
        "v4_adoption_bridge_receipt_sha256": execution_profile[
            "v4_adoption_policy"
        ].get("bridge_receipt_sha256"),
        "v4_source_calibration_evidence_receipt": state[
            "v4_source_calibration_evidence_receipt"
        ],
        "v5_adoption_success_count": len(
            state["v5_adopted_success_custom_ids"]
        ),
        "v5_adopted_success_custom_ids": state[
            "v5_adopted_success_custom_ids"
        ],
        "v5_adoption_bridge_receipt_sha256": execution_profile[
            "v5_adoption_policy"
        ].get("bridge_receipt_sha256"),
        "v5_source_calibration_evidence_receipt": state[
            "v5_source_calibration_evidence_receipt"
        ],
        "materialization_policy": deepcopy(
            dict(execution_profile["materialization_policy"])
        ),
        "new_profile_success_count": len(state["new_success_custom_ids"]),
        "fresh_calibration_receipt": state["fresh_calibration_receipt"],
        "authoritative_success_count": len(state["authoritative_success_by_id"]),
        "selected_response_sha256_by_id": {
            custom_id: sha256_json(row)
            for custom_id, row in state["authoritative_success_by_id"].items()
        },
        "hard_rows_counted": 0,
        "hard_usage_counted": 0,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return state["authoritative_success_by_id"], receipt
