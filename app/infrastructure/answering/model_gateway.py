"""回答计划与复核的结构化模型调用边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
from typing import Any, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel
import tiktoken

from app.application.model_usage_service import record_model_call, usage_numbers
from app.application.query_answering.proposals import (
    ProviderWireContractError,
    ProviderWireFailureReason,
)
from app.application.query_security_pipeline import authorize_external_call
from app.infra.llm.providers import llm_provider
from app.infrastructure.answering.prompts import (
    ANSWERING_MAX_LOGICAL_CALLS,
    ANSWERING_OUTPUT_TOKEN_CAP,
    ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION,
    resolve_answering_model_limits,
)
from app.infrastructure.model_calls.query_retry import (
    configured_query_model_retry_policy,
    invoke_query_model_call,
)
from app.shared.config.settings_config import settings


_T = TypeVar("_T", bound=BaseModel)


class StructuredAnswerModelError(ValueError):
    """Provider 响应不是目标 JSON Contract。"""

    code = "STRUCTURED_ANSWER_MODEL_INVALID"

    def __init__(
        self,
        message: str,
        *,
        scene: str | None = None,
        failure_stage: str | None = None,
        failure_reason_code: str | None = None,
        provider_attempt_count: int | None = None,
        logical_call_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_component = "structured_answer_model"
        self._bind_safe_diagnostics(
            scene=scene,
            failure_stage=failure_stage,
            failure_reason_code=failure_reason_code,
            provider_attempt_count=provider_attempt_count,
            logical_call_id=logical_call_id,
        )

    def _bind_safe_diagnostics(
        self,
        *,
        scene: str | None = None,
        failure_stage: str | None = None,
        failure_reason_code: str | None = None,
        provider_attempt_count: int | None = None,
        logical_call_id: str | None = None,
    ) -> None:
        """仅附加 evaluation_service 白名单允许的低基数字段。"""

        if scene is not None:
            self.safe_scene = scene
        if failure_stage is not None:
            self.safe_failure_stage = failure_stage
        if failure_reason_code is not None:
            self.safe_failure_reason_code = failure_reason_code
        if provider_attempt_count is not None:
            self.safe_provider_attempt_count = provider_attempt_count
        if logical_call_id is not None:
            self.safe_logical_call_id = logical_call_id


class _RetryableStructuredAnswerModelError(StructuredAnswerModelError):
    """只由专用 ProviderWireContractError 映射出的内部重试标记。"""


class AnsweringPromptBudgetError(RuntimeError):
    """回答 Prompt 没有可验证的完整路径窗口预算。"""

    code = "ANSWERING_PROMPT_BUDGET_BLOCKED"


def _url_host(value: object, *, field_name: str) -> str:
    """从 provider URL 提取规范化 host；不接受凭据或非 HTTP(S) URL。"""

    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AnsweringPromptBudgetError(f"{field_name} is not a safe provider URL")
    return host


def _frozen_llm_target_domain(state: Mapping[str, Any]) -> str:
    """把当前 provider endpoint 精确绑定到 QueryRun 接受时的域名白名单。"""

    target_domain = _url_host(
        settings.openai_base_url,
        field_name="configured LLM base URL",
    )
    policies = state.get("egress_policies")
    policies = policies if isinstance(policies, Mapping) else {}
    policy = policies.get("llm_generation")
    policy = policy if isinstance(policy, Mapping) else {}
    raw_domains = policy.get("allowed_domains")
    if not isinstance(raw_domains, Sequence) or isinstance(raw_domains, (str, bytes, bytearray)):
        raise AnsweringPromptBudgetError("frozen LLM domain allowlist is missing")
    frozen_domains = tuple(str(item).rstrip(".").lower() for item in raw_domains)
    if frozen_domains != (target_domain,):
        raise AnsweringPromptBudgetError(
            "configured LLM host differs from the frozen domain allowlist"
        )
    frozen_host = str(state.get("llm_host") or "").rstrip(".").lower()
    if frozen_host and frozen_host != target_domain:
        raise AnsweringPromptBudgetError("configured LLM host differs from frozen llm_host")
    return target_domain


def _client_target_domain(client: object) -> str:
    """从已创建但尚未调用的 SDK client 重取实际 endpoint host。"""

    candidates = (
        getattr(client, "openai_api_base", None),
        getattr(client, "base_url", None),
        getattr(getattr(client, "root_client", None), "base_url", None),
        getattr(getattr(client, "client", None), "base_url", None),
    )
    for candidate in candidates:
        if candidate:
            return _url_host(candidate, field_name="LLM client base URL")
    raise AnsweringPromptBudgetError("LLM client does not expose its target host")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


def _prompt_hash(prompt: Any) -> str:
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def _digest_response(response: object) -> str:
    """仅保存 provider payload 摘要，不把潜在正文写入审计。"""

    content = _content_text(getattr(response, "content", ""))
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def _prompt_contents(prompt: Any) -> tuple[str, ...]:
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes, bytearray)):
        raise AnsweringPromptBudgetError("structured answer prompt must be a message sequence")
    contents: list[str] = []
    for item in prompt:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            raise AnsweringPromptBudgetError("structured answer prompt message is invalid")
        contents.append(item["content"])
    if not contents:
        raise AnsweringPromptBudgetError("structured answer prompt cannot be empty")
    return tuple(contents)


def _count_prompt_tokens(prompt: Any, *, tokenizer_id: str) -> int:
    contents = _prompt_contents(prompt)
    if tokenizer_id.startswith("tiktoken:"):
        encoding_name = tokenizer_id.removeprefix("tiktoken:")
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise AnsweringPromptBudgetError("frozen answering tokenizer is unavailable") from exc
        return sum(len(encoding.encode(content)) for content in contents)
    if tokenizer_id == "utf8-byte-upper-bound":
        return sum(len(content.encode("utf-8")) for content in contents)
    raise AnsweringPromptBudgetError("frozen answering tokenizer is unsupported")


def _expected_tokenizer_binding(model_id: str) -> tuple[str, str, str]:
    """独立解析模型/tokenizer 绑定；未知模型只能使用 byte 上界。"""

    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except Exception:
        return "utf8-byte-upper-bound", "v1", "utf8_byte_upper_bound"
    return (
        f"tiktoken:{encoding.name}",
        str(getattr(tiktoken, "__version__", "unknown")),
        "model_exact",
    )


def _ledger(state: dict[str, Any]) -> dict[str, Any]:
    ledger = state.get("answer_model_ledger")
    if not isinstance(ledger, dict):
        raise AnsweringPromptBudgetError("answer model attempt ledger is missing")
    return ledger


def _logical_call_id(
    state: Mapping[str, Any],
    *,
    scene: str,
    prompt_version: str,
    prompt_sha256: str,
) -> str:
    context_pack = state.get("context_pack")
    context_pack = context_pack if isinstance(context_pack, Mapping) else {}
    payload = {
        "context_pack_hash": context_pack.get("content_hash"),
        "scene": scene,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"qmc_{digest[:32]}"


def _existing_prompt_receipt(
    state: Mapping[str, Any], logical_call_id: str
) -> dict[str, Any] | None:
    receipts = state.get("answer_prompt_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        return None
    for item in reversed(receipts):
        if isinstance(item, dict) and item.get("logical_call_id") == logical_call_id:
            return item
    return None


def _register_logical_call(
    state: dict[str, Any],
    *,
    logical_call_id: str,
    scene: str,
    prompt_sha256: str,
    actual_input_tokens: int,
) -> dict[str, Any]:
    ledger = _ledger(state)
    logical_calls = ledger.get("logical_calls")
    if not isinstance(logical_calls, list):
        raise AnsweringPromptBudgetError("answer model logical call ledger is malformed")
    for item in logical_calls:
        if isinstance(item, dict) and item.get("logical_call_id") == logical_call_id:
            return item
    count = ledger.get("logical_call_count")
    limit = ledger.get("max_logical_calls")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or count >= limit
    ):
        raise AnsweringPromptBudgetError("answer model logical call budget is exhausted")
    entry = {
        "logical_call_id": logical_call_id,
        "scene": scene,
        "prompt_sha256": prompt_sha256,
        "input_tokens": actual_input_tokens,
        "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "status": "running",
    }
    logical_calls.append(entry)
    ledger["logical_call_count"] = count + 1
    return entry


def _reserve_provider_attempt(
    state: dict[str, Any],
    *,
    logical_call_id: str,
    scene: str,
    actual_input_tokens: int,
) -> dict[str, Any]:
    """在每次真实 SDK 调用前保守预留一次完整 provider 上界。"""

    ledger = _ledger(state)
    attempts = ledger.get("provider_attempts")
    if not isinstance(attempts, list):
        raise AnsweringPromptBudgetError("answer model provider attempt ledger is malformed")
    input_cap = ledger.get("provider_attempt_input_token_cap")
    output_cap = ledger.get("provider_attempt_output_token_cap")
    attempt_limit = ledger.get("max_provider_attempts")
    per_call_limit = ledger.get("max_provider_attempts_per_logical_call")
    count = ledger.get("provider_attempt_count")
    integers = (input_cap, output_cap, attempt_limit, per_call_limit, count)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
        raise AnsweringPromptBudgetError("answer model provider attempt limits are malformed")
    assert isinstance(input_cap, int)
    assert isinstance(output_cap, int)
    assert isinstance(attempt_limit, int)
    assert isinstance(per_call_limit, int)
    assert isinstance(count, int)
    per_call_count = sum(
        item.get("logical_call_id") == logical_call_id
        for item in attempts
        if isinstance(item, Mapping)
    )
    if count >= attempt_limit or per_call_count >= per_call_limit:
        raise AnsweringPromptBudgetError("answer model provider attempt budget is exhausted")
    if actual_input_tokens > input_cap:
        raise AnsweringPromptBudgetError("answer model attempt input exceeds its frozen cap")
    next_reserved_input = int(ledger.get("reserved_input_tokens") or 0) + input_cap
    next_reserved_output = int(ledger.get("reserved_output_tokens") or 0) + output_cap
    if next_reserved_input + next_reserved_output > int(ledger.get("max_total_tokens") or 0):
        raise AnsweringPromptBudgetError("answer model token reservation is exhausted")
    attempt = {
        "attempt_id": f"{logical_call_id}:pa{per_call_count + 1}",
        "logical_call_id": logical_call_id,
        "scene": scene,
        "attempt": per_call_count + 1,
        "actual_input_tokens": actual_input_tokens,
        "reserved_input_tokens": input_cap,
        "reserved_output_tokens": output_cap,
        "status": "running",
    }
    attempts.append(attempt)
    ledger["provider_attempt_count"] = count + 1
    ledger["reserved_input_tokens"] = next_reserved_input
    ledger["reserved_output_tokens"] = next_reserved_output
    return attempt


def _reconcile_provider_attempt(
    state: dict[str, Any],
    attempt: dict[str, Any],
    *,
    response: object | None,
    status: str,
) -> None:
    """以 provider usage 核销；缺失元数据时按冻结上界保守计费。"""

    ledger = _ledger(state)
    input_cap = int(attempt["reserved_input_tokens"])
    output_cap = int(attempt["reserved_output_tokens"])
    observed_input, observed_output = usage_numbers(response) if response is not None else (0, 0)
    metadata_available = bool(observed_input or observed_output)
    if metadata_available and (observed_input > input_cap or observed_output > output_cap):
        attempt.update(
            {
                "status": "budget_violation",
                "usage_status": "provider_exceeded_cap",
                "observed_input_tokens": observed_input,
                "observed_output_tokens": observed_output,
            }
        )
        raise AnsweringPromptBudgetError("provider usage exceeded the frozen attempt cap")
    accounted_input = observed_input if metadata_available else input_cap
    accounted_output = observed_output if metadata_available else output_cap
    attempt.update(
        {
            "status": status,
            "usage_status": "available" if metadata_available else "estimated_upper_bound",
            "observed_input_tokens": observed_input,
            "observed_output_tokens": observed_output,
            "accounted_input_tokens": accounted_input,
            "accounted_output_tokens": accounted_output,
        }
    )
    ledger["observed_input_tokens"] = int(ledger.get("observed_input_tokens") or 0) + (
        observed_input
    )
    ledger["observed_output_tokens"] = (
        int(ledger.get("observed_output_tokens") or 0) + observed_output
    )
    ledger["accounted_input_tokens"] = (
        int(ledger.get("accounted_input_tokens") or 0) + accounted_input
    )
    ledger["accounted_output_tokens"] = (
        int(ledger.get("accounted_output_tokens") or 0) + accounted_output
    )
    if not metadata_available:
        ledger["usage_metadata_complete"] = False


def _terminalize_running_provider_attempts(
    state: dict[str, Any],
    *,
    logical_call_id: str,
    receipt: Mapping[str, Any],
    response: object | None,
    failure_stage: str,
) -> None:
    """异常收口时保守核销本 logical call 中尚未落终态的 attempt。"""

    ledger = _ledger(state)
    attempts = ledger.get("provider_attempts")
    if not isinstance(attempts, list):
        raise AnsweringPromptBudgetError("answer model provider attempt ledger is malformed")
    response_available = response is not None
    for attempt in attempts:
        if (
            not isinstance(attempt, dict)
            or attempt.get("logical_call_id") != logical_call_id
            or attempt.get("status") != "running"
        ):
            continue
        try:
            _reconcile_provider_attempt(
                state,
                attempt,
                response=response if response_available else None,
                status="failed",
            )
        except Exception:
            if attempt.get("status") == "running":
                input_cap = int(attempt["reserved_input_tokens"])
                output_cap = int(attempt["reserved_output_tokens"])
                attempt.update(
                    {
                        "status": "failed",
                        "usage_status": "estimated_upper_bound",
                        "observed_input_tokens": 0,
                        "observed_output_tokens": 0,
                        "accounted_input_tokens": input_cap,
                        "accounted_output_tokens": output_cap,
                    }
                )
                ledger["accounted_input_tokens"] = (
                    int(ledger.get("accounted_input_tokens") or 0) + input_cap
                )
                ledger["accounted_output_tokens"] = (
                    int(ledger.get("accounted_output_tokens") or 0) + output_cap
                )
                ledger["usage_metadata_complete"] = False
        response_available = False

    response_attempts = receipt.get("response_attempts")
    if not isinstance(response_attempts, list):
        return
    for response_attempt in response_attempts:
        if not isinstance(response_attempt, dict):
            continue
        if response_attempt.get("status") not in {"running", "parsing"}:
            continue
        response_attempt.update(
            {
                "status": "failed",
                "failure_stage": failure_stage,
            }
        )


def enforce_answer_prompt_budget(
    state: dict[str, Any],
    *,
    scene: str,
    prompt: Any,
    output_token_cap: int,
) -> int:
    """在授权和 client 创建前重放冻结预算并校验当前消息。"""

    if scene not in {"answer_plan_selection", "answer_plan_semantic_review"}:
        raise AnsweringPromptBudgetError("unknown structured answering scene")
    if output_token_cap != ANSWERING_OUTPUT_TOKEN_CAP:
        raise AnsweringPromptBudgetError("structured answering output cap drifted")
    if bool(state.get("context_budget_blocked")):
        raise AnsweringPromptBudgetError("worst answering path exceeds the model window")

    receipt = state.get("answer_prompt_budget")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    context_pack = state.get("context_pack")
    context_pack = context_pack if isinstance(context_pack, Mapping) else {}
    context_budget = context_pack.get("budget")
    context_budget = context_budget if isinstance(context_budget, Mapping) else {}
    model_id = str(state.get("llm_model_id") or "").strip()
    if (
        not model_id
        or receipt.get("schema_version") != ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("context_pack_hash") != context_pack.get("content_hash")
        or receipt.get("llm_model_id") != model_id
    ):
        raise AnsweringPromptBudgetError("answering prompt budget receipt is missing or stale")

    integer_fields = (
        "model_context_limit",
        "configured_model_context_limit",
        "provider_max_input_tokens",
        "reserved_output_tokens",
        "output_token_cap",
        "safety_margin_tokens",
        "max_input_tokens",
        "worst_input_tokens",
    )
    if any(
        not isinstance(receipt.get(field), int) or isinstance(receipt.get(field), bool)
        for field in integer_fields
    ):
        raise AnsweringPromptBudgetError("answering prompt budget receipt is malformed")
    model_limit = int(receipt["model_context_limit"])
    configured_model_limit = int(receipt["configured_model_context_limit"])
    provider_max_input_tokens = int(receipt["provider_max_input_tokens"])
    reserved_output = int(receipt["reserved_output_tokens"])
    safety_margin = int(receipt["safety_margin_tokens"])
    max_input = int(receipt["max_input_tokens"])
    worst_input = int(receipt["worst_input_tokens"])
    (
        expected_context_window,
        expected_max_input,
        expected_limits_version,
        expected_limits_mode,
    ) = resolve_answering_model_limits(
        model_id,
        configured_context_limit=configured_model_limit,
    )
    if (
        int(receipt["output_token_cap"]) != ANSWERING_OUTPUT_TOKEN_CAP
        or reserved_output < ANSWERING_OUTPUT_TOKEN_CAP
        or context_budget.get("model_context_limit") != model_limit
        or context_budget.get("reserved_output") != reserved_output
        or context_budget.get("safety_margin") != safety_margin
        or model_limit != expected_context_window
        or provider_max_input_tokens != expected_max_input
        or receipt.get("model_limits_version") != expected_limits_version
        or receipt.get("model_limits_binding_mode") != expected_limits_mode
        or max_input
        != min(
            model_limit - reserved_output - safety_margin,
            provider_max_input_tokens - safety_margin,
        )
        or worst_input > max_input
    ):
        raise AnsweringPromptBudgetError("answering prompt budget invariants are invalid")

    tokenizer_id = receipt.get("tokenizer_id")
    tokenizer_version = receipt.get("tokenizer_version")
    binding_mode = receipt.get("tokenizer_binding_mode")
    expected_id, expected_version, expected_mode = _expected_tokenizer_binding(model_id)
    if (
        not isinstance(tokenizer_id, str)
        or tokenizer_id != context_pack.get("tokenizer_id")
        or tokenizer_version != context_pack.get("tokenizer_version")
        or (tokenizer_id, tokenizer_version, binding_mode)
        != (expected_id, expected_version, expected_mode)
    ):
        raise AnsweringPromptBudgetError("answering prompt tokenizer receipt is invalid")
    actual_input = _count_prompt_tokens(prompt, tokenizer_id=tokenizer_id)
    if actual_input > worst_input or actual_input > max_input:
        raise AnsweringPromptBudgetError("current answering prompt exceeds its frozen budget")
    retry_attempts = configured_query_model_retry_policy().max_attempts
    run_integer_fields = (
        "max_logical_calls",
        "max_provider_attempts_per_logical_call",
        "max_provider_attempts",
        "provider_attempt_input_token_cap",
        "provider_attempt_output_token_cap",
        "max_total_input_tokens",
        "max_total_output_tokens",
        "max_total_tokens",
        "answer_run_token_budget",
        "answer_run_tokens_available",
    )
    if any(
        not isinstance(receipt.get(field), int) or isinstance(receipt.get(field), bool)
        for field in run_integer_fields
    ):
        raise AnsweringPromptBudgetError("answering run budget receipt is malformed")
    max_provider_attempts = ANSWERING_MAX_LOGICAL_CALLS * retry_attempts
    provider_attempt_input_cap = worst_input + safety_margin
    max_total_input = max_provider_attempts * provider_attempt_input_cap
    max_total_output = max_provider_attempts * ANSWERING_OUTPUT_TOKEN_CAP
    if (
        receipt["max_logical_calls"] != ANSWERING_MAX_LOGICAL_CALLS
        or receipt["max_provider_attempts_per_logical_call"] != retry_attempts
        or receipt["max_provider_attempts"] != max_provider_attempts
        or receipt["provider_attempt_input_token_cap"] != provider_attempt_input_cap
        or receipt["provider_attempt_output_token_cap"] != ANSWERING_OUTPUT_TOKEN_CAP
        or receipt["max_total_input_tokens"] != max_total_input
        or receipt["max_total_output_tokens"] != max_total_output
        or receipt["max_total_tokens"] != max_total_input + max_total_output
        or receipt["answer_run_tokens_available"] != receipt["answer_run_token_budget"]
        or receipt["max_total_tokens"] > receipt["answer_run_tokens_available"]
        or receipt.get("cost_budget_status") != "unavailable"
        or receipt.get("cost_limit") is not None
        or receipt.get("cost_currency") is not None
    ):
        raise AnsweringPromptBudgetError("answering run token budget invariants are invalid")
    ledger = _ledger(state)
    if (
        ledger.get("schema_version") != "answer-model-budget-ledger.v1"
        or ledger.get("status") != "ready"
        or ledger.get("context_pack_hash") != context_pack.get("content_hash")
        or ledger.get("llm_model_id") != model_id
        or ledger.get("cost_budget_status") != "unavailable"
        or ledger.get("cost_limit") is not None
        or ledger.get("cost_currency") is not None
        or any(
            ledger.get(field) != receipt.get(field)
            for field in (
                "max_logical_calls",
                "max_provider_attempts_per_logical_call",
                "max_provider_attempts",
                "provider_attempt_input_token_cap",
                "provider_attempt_output_token_cap",
                "max_total_tokens",
            )
        )
    ):
        raise AnsweringPromptBudgetError("answer model attempt ledger is missing or stale")
    state.setdefault("answer_prompt_preflights", []).append(
        {
            "schema_version": "answer-prompt-preflight.v1",
            "scene": scene,
            "prompt_sha256": _prompt_hash(prompt),
            "input_tokens": actual_input,
            "worst_path_input_tokens": worst_input,
            "max_input_tokens": max_input,
            "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
            "status": "ready",
        }
    )
    return actual_input


def invoke_structured_answer_model(
    state: dict[str, Any],
    *,
    scene: str,
    prompt: Any,
    response_model: type[_T],
    response_parser: Callable[[str], BaseModel],
    prompt_version: str,
    output_token_cap: int,
    wire_response_model: type[BaseModel] | None = None,
    response_projector: Callable[[BaseModel], _T] | None = None,
) -> _T:
    """只重试 provider wire 错误，并统一收口调用后的所有终态。"""

    actual_input_tokens = enforce_answer_prompt_budget(
        state,
        scene=scene,
        prompt=prompt,
        output_token_cap=output_token_cap,
    )
    prompt_sha256 = _prompt_hash(prompt)
    logical_call_id = _logical_call_id(
        state,
        scene=scene,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
    )
    existing_receipt = _existing_prompt_receipt(state, logical_call_id)
    if existing_receipt is not None:
        if existing_receipt.get("status") == "succeeded" and isinstance(
            existing_receipt.get("parsed_proposal"), Mapping
        ):
            return response_model.model_validate(existing_receipt["parsed_proposal"])
        if existing_receipt.get("failure_stage") == "wire_contract":
            attempts = existing_receipt.get("response_attempts")
            attempt_count = len(attempts) if isinstance(attempts, list) else 0
            reason_code = str(
                existing_receipt.get("failure_reason_code")
                or ProviderWireFailureReason.WIRE_CONTRACT_INVALID.value
            )
            raise StructuredAnswerModelError(
                "cached provider response violated the JSON contract",
                scene=scene,
                failure_stage="wire_contract",
                failure_reason_code=reason_code,
                provider_attempt_count=attempt_count,
                logical_call_id=logical_call_id,
            )
        raise AnsweringPromptBudgetError(
            "answer logical call has an indeterminate prior provider outcome"
        )
    receipts = state.get("answer_prompt_receipts")
    if receipts is None:
        receipts = []
        state["answer_prompt_receipts"] = receipts
    if not isinstance(receipts, list):
        raise AnsweringPromptBudgetError("answer prompt receipt collection is malformed")
    logical_entry = _register_logical_call(
        state,
        logical_call_id=logical_call_id,
        scene=scene,
        prompt_sha256=prompt_sha256,
        actual_input_tokens=actual_input_tokens,
    )
    receipt = {
        "schema_version": "answer-prompt-receipt.v6",
        "logical_call_id": logical_call_id,
        "scene": scene,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "response_schema": (wire_response_model or response_model).__name__,
        "parsed_schema": response_model.__name__,
        "model_id": str(state.get("llm_model_id") or ""),
        "input_tokens": actual_input_tokens,
        "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "provider_status": "not_started",
        "parse_status": "not_started",
        "status": "running",
        "target_domain": None,
        "response_attempts": [],
    }
    receipts.append(receipt)
    model_id = str(state.get("llm_model_id") or settings.llm_default_model or "unknown")
    state["llm_model_id"] = model_id
    receipt["model_id"] = model_id
    response: object | None = None
    failure_stage = "endpoint_validation"

    try:
        target_domain = _frozen_llm_target_domain(state)
        receipt["target_domain"] = target_domain
        failure_stage = "authorization"
        authorize_external_call(
            state,
            purpose="llm_generation",
            provider="configured-llm",
            target_domain=target_domain,
        )
        failure_stage = "client_init"
        client = llm_provider.chat(model=model_id, json_mode=True)
        failure_stage = "endpoint_validation"
        if _client_target_domain(client) != target_domain:
            raise AnsweringPromptBudgetError(
                "LLM client target differs from the authorized provider host"
            )
        failure_stage = "bind"
        bind = getattr(client, "bind", None)
        if callable(bind):
            client = bind(max_tokens=ANSWERING_OUTPUT_TOKEN_CAP)
        failure_stage = "endpoint_validation"
        if _client_target_domain(client) != target_domain:
            raise AnsweringPromptBudgetError(
                "bound LLM client target differs from the authorized provider host"
            )

        def call(remaining_seconds: float) -> tuple[object, BaseModel]:
            nonlocal response
            response = None
            timeout = min(
                max(0.001, float(settings.llm_request_timeout)),
                remaining_seconds,
            )
            attempt = _reserve_provider_attempt(
                state,
                logical_call_id=logical_call_id,
                scene=scene,
                actual_input_tokens=actual_input_tokens,
            )
            response_attempt = {
                "attempt_id": attempt["attempt_id"],
                "attempt_number": attempt["attempt"],
                "stage": "transport",
                "status": "running",
                "failure_stage": None,
                "response_sha256": None,
            }
            receipt["response_attempts"].append(response_attempt)
            receipt["provider_status"] = "started"
            try:
                attempt_response = client.invoke(prompt, timeout=timeout)
            except Exception:
                response_attempt.update(
                    {
                        "status": "failed",
                        "failure_stage": "transport",
                    }
                )
                _reconcile_provider_attempt(
                    state,
                    attempt,
                    response=None,
                    status="failed",
                )
                raise
            response = attempt_response
            response_sha256 = _digest_response(attempt_response)
            receipt["provider_status"] = "returned"
            receipt["response_sha256"] = response_sha256
            response_attempt.update(
                {
                    "stage": "wire_parse",
                    "status": "parsing",
                    "response_sha256": response_sha256,
                }
            )
            receipt["parse_status"] = "started"
            raw = _content_text(getattr(attempt_response, "content", ""))
            try:
                parsed_wire = response_parser(raw)
            except ProviderWireContractError as exc:
                reason_code = exc.reason_code.value
                receipt["parse_status"] = "failed_wire_contract"
                response_attempt.update(
                    {
                        "status": "failed",
                        "failure_stage": "wire_contract",
                        "failure_reason_code": reason_code,
                    }
                )
                receipt["failure_reason_code"] = reason_code
                _reconcile_provider_attempt(
                    state,
                    attempt,
                    response=attempt_response,
                    status="failed_parse",
                )
                raise _RetryableStructuredAnswerModelError(
                    "provider response violated JSON contract",
                    scene=scene,
                    failure_stage="wire_contract",
                    failure_reason_code=reason_code,
                    provider_attempt_count=int(attempt["attempt"]),
                    logical_call_id=logical_call_id,
                ) from exc
            except Exception:
                receipt["parse_status"] = "failed_parser_invariant"
                response_attempt.update(
                    {
                        "status": "failed",
                        "failure_stage": "parser_invariant",
                    }
                )
                _reconcile_provider_attempt(
                    state,
                    attempt,
                    response=attempt_response,
                    status="failed_parse",
                )
                raise
            try:
                _reconcile_provider_attempt(
                    state,
                    attempt,
                    response=attempt_response,
                    status="succeeded",
                )
            except Exception:
                response_attempt.update(
                    {
                        "status": "failed",
                        "failure_stage": "budget_reconciliation",
                    }
                )
                raise
            receipt["parse_status"] = "succeeded"
            response_attempt["status"] = "wire_valid"
            return attempt_response, parsed_wire

        failure_stage = "provider_call"
        retry_audit = {"logical_call_id": logical_call_id}
        response, parsed_wire = invoke_query_model_call(
            state,
            scene=scene,
            model_id=model_id,
            invoke=call,
            retryable_error_types=(_RetryableStructuredAnswerModelError,),
            audit=retry_audit,
        )
        failure_stage = "server_projection"
        if response_projector is None:
            result = response_model.model_validate(parsed_wire.model_dump(mode="json"))
        else:
            result = response_projector(parsed_wire)
        receipt["parsed_proposal"] = result.model_dump(mode="json")
    except Exception as exc:
        terminal_stage = failure_stage
        attempts = receipt["response_attempts"]
        if failure_stage == "provider_call" and attempts:
            last_failure = attempts[-1].get("failure_stage")
            terminal_stage = str(last_failure or "retry_control")
        _terminalize_running_provider_attempts(
            state,
            logical_call_id=logical_call_id,
            receipt=receipt,
            response=response,
            failure_stage=terminal_stage,
        )
        receipt["failure_stage"] = terminal_stage
        receipt["status"] = "failed"
        logical_entry["status"] = "failed"
        if isinstance(exc, StructuredAnswerModelError):
            exc._bind_safe_diagnostics(
                scene=scene,
                failure_stage=terminal_stage,
                failure_reason_code=str(
                    receipt.get("failure_reason_code")
                    or ProviderWireFailureReason.WIRE_CONTRACT_INVALID.value
                ),
                provider_attempt_count=len(attempts),
                logical_call_id=logical_call_id,
            )
        state["usage"] = record_model_call(
            state,
            (response,) if response is not None else (),
            scene=scene,
            call_status="failed",
            model_id=model_id,
        )
        raise
    receipt["status"] = "succeeded"
    receipt["failure_stage"] = None
    logical_entry["status"] = "succeeded"
    state["usage"] = record_model_call(
        state,
        (response,),
        scene=scene,
        model_id=model_id,
    )
    return result


__all__ = [
    "AnsweringPromptBudgetError",
    "StructuredAnswerModelError",
    "enforce_answer_prompt_budget",
    "invoke_structured_answer_model",
]
