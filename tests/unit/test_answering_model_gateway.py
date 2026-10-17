from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

from app.application.query_answering.proposals import (
    MAX_PROPOSAL_CLAIMS,
    AnswerDecision,
    AnswerPlanProposal,
    AnswerPlanWireProposal,
    ClaimProposal,
    PlanReasonCode,
    ProposalValidationError,
    ProviderWireFailureReason,
    RenderingRule,
    ReviewDecision,
    parse_answer_plan_wire_proposal_json,
    project_answer_plan_wire_proposal,
    validate_claim_allowlists,
)
from app.domain.answering.question import QuestionContract
from app.domain.query_planning import QueryFacet
from app.infrastructure.answering import model_gateway, provider as answering_provider
from app.infrastructure.answering.model_gateway import (
    AnsweringPromptBudgetError,
    StructuredAnswerModelError,
    invoke_structured_answer_model,
)
from app.infrastructure.answering.provider import (
    ConfiguredAnsweringModelAdapter,
    _output_cap,
)
from app.infrastructure.answering.prompts import ANSWERING_OUTPUT_TOKEN_CAP
from app.infrastructure.model_calls.query_retry import (
    QueryModelCallDeadlineExceeded,
    QueryModelRetryPolicy,
    invoke_query_model_call,
)


_VALID_REFUSAL = '{"decision":"refuse","claims":[]}'
_OBLIGATION_ID = "qf_" + "a" * 32


def _question_contract() -> QuestionContract:
    return QuestionContract.from_facets(
        query_plan_id="qpl_" + "1" * 32,
        source_query="合成问题",
        facets=(
            QueryFacet(
                facet_id=_OBLIGATION_ID,
                information_need="合成义务",
                retrieval_expression="合成义务",
            ),
        ),
        builder_version="model-gateway-test.v1",
    )


@dataclass
class _Response:
    content: Any
    usage_metadata: dict[str, int]


class _FakeChatClient:
    def __init__(
        self,
        outcomes: list[object],
        *,
        base_url: str = "https://llm.example/v1",
        bind_error: Exception | None = None,
        bound_base_url: str | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self.openai_api_base = base_url
        self._bind_error = bind_error
        self._bound_base_url = bound_base_url
        self.bind_calls: list[dict[str, Any]] = []
        self.invoke_calls: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> "_FakeChatClient":
        self.bind_calls.append(dict(kwargs))
        if self._bind_error is not None:
            raise self._bind_error
        if self._bound_base_url is not None:
            self.openai_api_base = self._bound_base_url
        return self

    def invoke(self, prompt: Any, *, timeout: float) -> object:
        self.invoke_calls.append({"prompt": prompt, "timeout": timeout})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _mock_usage(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """隔离 SQL/指标后端，只验证网关提交的逻辑调用回执。"""

    calls: list[dict[str, Any]] = []

    def record(
        state: dict[str, Any],
        responses: tuple[object, ...],
        *,
        scene: str,
        call_status: str = "succeeded",
        provider: str = "configured-llm",
        model_id: str | None = None,
    ) -> dict[str, Any]:
        del provider
        response = responses[0] if responses else None
        metadata = getattr(response, "usage_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        input_tokens = int(metadata.get("input_tokens") or 0)
        output_tokens = int(metadata.get("output_tokens") or 0)
        available = call_status == "succeeded" and bool(input_tokens or output_tokens)
        summary = (
            {
                "status": "available",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if available
            else {"status": "unavailable", "reason": "synthetic usage unavailable"}
        )
        call = {
            "scene": scene,
            "call_status": call_status,
            "usage_status": summary["status"],
            "model_id": model_id or "unknown",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        calls.append({**call, "response_count": len(responses)})
        state.setdefault("model_usage_summaries", []).append(call)
        return summary

    monkeypatch.setattr(model_gateway, "record_model_call", record)
    return calls


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeChatClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chat_calls: list[dict[str, Any]] = []
    authorization_calls: list[dict[str, Any]] = []

    def authorize(_state: dict[str, Any], **kwargs: Any) -> None:
        authorization_calls.append(dict(kwargs))

    def chat(*, model: str, json_mode: bool) -> _FakeChatClient:
        chat_calls.append({"model": model, "json_mode": json_mode})
        return client

    monkeypatch.setattr(model_gateway, "authorize_external_call", authorize)
    monkeypatch.setattr(model_gateway.llm_provider, "chat", chat)
    monkeypatch.setattr(
        model_gateway,
        "settings",
        SimpleNamespace(
            openai_base_url="https://llm.example/v1",
            llm_default_model="synthetic-model",
            llm_request_timeout=60.0,
        ),
    )
    return chat_calls, authorization_calls


def _install_fast_retry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeout: float = 5.0,
    max_attempts: int = 2,
) -> None:
    def bounded_retry(
        state,
        *,
        scene,
        model_id,
        invoke,
        retryable_error_types=(),
        audit=None,
    ):
        return invoke_query_model_call(
            state,
            scene=scene,
            model_id=model_id,
            invoke=invoke,
            retryable_error_types=retryable_error_types,
            audit=audit,
            policy=QueryModelRetryPolicy(
                max_attempts=max_attempts,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                total_timeout_seconds=timeout,
            ),
            sleep=lambda _seconds: None,
        )

    monkeypatch.setattr(model_gateway, "invoke_query_model_call", bounded_retry)


def _budget_state(**extra: Any) -> dict[str, Any]:
    context_pack_hash = "sha256:" + "1" * 64
    max_total_input_tokens = 13_200
    max_total_output_tokens = 12 * ANSWERING_OUTPUT_TOKEN_CAP
    max_total_tokens = max_total_input_tokens + max_total_output_tokens
    ledger_limits = {
        "max_logical_calls": 4,
        "max_provider_attempts_per_logical_call": 3,
        "max_provider_attempts": 12,
        "provider_attempt_input_token_cap": 1_100,
        "provider_attempt_output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "max_total_tokens": max_total_tokens,
    }
    state: dict[str, Any] = {
        "llm_model_id": "synthetic-model",
        "egress_policies": {
            "llm_generation": {"allowed_domains": ["llm.example"]},
        },
        "context_pack": {
            "content_hash": context_pack_hash,
            "tokenizer_id": "utf8-byte-upper-bound",
            "tokenizer_version": "v1",
            "budget": {
                "model_context_limit": 10_000,
                "reserved_output": ANSWERING_OUTPUT_TOKEN_CAP,
                "safety_margin": 100,
            },
        },
        "answer_prompt_budget": {
            "schema_version": "answer-prompt-budget.v4",
            "status": "ready",
            "context_pack_hash": context_pack_hash,
            "llm_model_id": "synthetic-model",
            "tokenizer_id": "utf8-byte-upper-bound",
            "tokenizer_version": "v1",
            "tokenizer_binding_mode": "utf8_byte_upper_bound",
            "model_context_limit": 10_000,
            "configured_model_context_limit": 10_000,
            "provider_max_input_tokens": 10_000,
            "model_limits_version": "retrieval-profile.context-pack.v1",
            "model_limits_binding_mode": "profile_configured",
            "reserved_output_tokens": ANSWERING_OUTPUT_TOKEN_CAP,
            "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
            "safety_margin_tokens": 100,
            "max_input_tokens": 10_000 - ANSWERING_OUTPUT_TOKEN_CAP - 100,
            "worst_input_tokens": 1_000,
            **ledger_limits,
            "max_total_input_tokens": max_total_input_tokens,
            "max_total_output_tokens": max_total_output_tokens,
            "answer_run_token_budget": max_total_tokens,
            "answer_run_tokens_available": max_total_tokens,
            "cost_budget_status": "unavailable",
            "cost_limit": None,
            "cost_currency": None,
        },
        "answer_model_ledger": {
            "schema_version": "answer-model-budget-ledger.v1",
            "status": "ready",
            "context_pack_hash": context_pack_hash,
            "llm_model_id": "synthetic-model",
            **ledger_limits,
            "logical_call_count": 0,
            "provider_attempt_count": 0,
            "reserved_input_tokens": 0,
            "reserved_output_tokens": 0,
            "accounted_input_tokens": 0,
            "accounted_output_tokens": 0,
            "observed_input_tokens": 0,
            "observed_output_tokens": 0,
            "usage_metadata_complete": True,
            "logical_calls": [],
            "provider_attempts": [],
            "cost_budget_status": "unavailable",
            "cost_limit": None,
            "cost_currency": None,
        },
    }
    state.update(extra)
    return state


def _assert_failed_terminalized(
    state: dict[str, Any],
    *,
    failure_stage: str,
) -> None:
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["status"] == "failed"
    assert receipt["failure_stage"] == failure_stage
    assert state["answer_model_ledger"]["logical_calls"][0]["status"] == "failed"
    assert all(
        item["status"] != "running" for item in state["answer_model_ledger"]["provider_attempts"]
    )
    assert all(
        item["status"] not in {"running", "parsing"} for item in receipt["response_attempts"]
    )


def _invoke(
    state: dict[str, Any],
    *,
    output_token_cap: int = ANSWERING_OUTPUT_TOKEN_CAP,
    response_parser: Callable[[str], AnswerPlanWireProposal] | None = None,
    response_projector: Callable[[AnswerPlanWireProposal], AnswerPlanProposal] | None = None,
) -> AnswerPlanProposal:
    def parse_wire(raw: str) -> AnswerPlanWireProposal:
        if response_parser is not None:
            return response_parser(raw)
        wire = parse_answer_plan_wire_proposal_json(raw)
        validate_claim_allowlists(
            wire.claims,
            allowed_obligation_ids=("qf_" + "a" * 32,),
            allowed_witness_ids=("wit_" + "0" * 40,),
        )
        return wire

    def project_wire(wire: AnswerPlanWireProposal) -> AnswerPlanProposal:
        if response_projector is not None:
            return response_projector(wire)
        return project_answer_plan_wire_proposal(wire)

    return invoke_structured_answer_model(
        state,
        scene="answer_plan_selection",
        prompt=[{"role": "user", "content": "synthetic opaque task"}],
        response_model=AnswerPlanProposal,
        response_parser=parse_wire,
        prompt_version="test-answer-plan.v1",
        output_token_cap=output_token_cap,
        wire_response_model=AnswerPlanWireProposal,
        response_projector=project_wire,
    )


def test_success_uses_json_mode_hard_token_cap_and_records_receipts(
    monkeypatch: pytest.MonkeyPatch,
    _mock_usage: list[dict[str, Any]],
) -> None:
    response = _Response(
        content=_VALID_REFUSAL,
        usage_metadata={"input_tokens": 17, "output_tokens": 9},
    )
    client = _FakeChatClient([response])
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, timeout=0.25)
    state = _budget_state()

    result = _invoke(state)

    assert result.decision == AnswerDecision.REFUSE
    assert result.reason_codes == (PlanReasonCode.NO_SUPPORTING_WITNESS,)
    assert authorization_calls == [
        {
            "purpose": "llm_generation",
            "provider": "configured-llm",
            "target_domain": "llm.example",
        }
    ]
    assert chat_calls == [{"model": "synthetic-model", "json_mode": True}]
    assert client.bind_calls == [{"max_tokens": ANSWERING_OUTPUT_TOKEN_CAP}]
    assert len(client.invoke_calls) == 1
    assert 0 < client.invoke_calls[0]["timeout"] <= 0.25
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["schema_version"] == "answer-prompt-receipt.v6"
    assert receipt["logical_call_id"].startswith("qmc_")
    assert receipt["scene"] == "answer_plan_selection"
    assert receipt["prompt_version"] == "test-answer-plan.v1"
    assert receipt["response_schema"] == "AnswerPlanWireProposal"
    assert receipt["parsed_schema"] == "AnswerPlanProposal"
    assert receipt["model_id"] == "synthetic-model"
    assert receipt["target_domain"] == "llm.example"
    assert receipt["output_token_cap"] == ANSWERING_OUTPUT_TOKEN_CAP
    assert receipt["provider_status"] == "returned"
    assert receipt["parse_status"] == "succeeded"
    assert receipt["response_attempts"] == [
        {
            "attempt_id": receipt["logical_call_id"] + ":pa1",
            "attempt_number": 1,
            "stage": "wire_parse",
            "status": "wire_valid",
            "failure_stage": None,
            "response_sha256": receipt["response_sha256"],
        }
    ]
    assert receipt["parsed_proposal"]["decision"] == "refuse"
    assert receipt["status"] == "succeeded"
    assert state["usage"]["status"] == "available"
    assert state["usage"]["total_tokens"] == 26
    assert state["model_usage_summaries"][-1]["call_status"] == "succeeded"
    assert state["query_model_call_audits"][0]["provider_attempt_count"] == 1
    ledger = state["answer_model_ledger"]
    assert ledger["logical_call_count"] == 1
    assert ledger["provider_attempt_count"] == 1
    assert ledger["accounted_input_tokens"] == 17
    assert ledger["accounted_output_tokens"] == 9
    assert _mock_usage == [
        {
            "scene": "answer_plan_selection",
            "call_status": "succeeded",
            "usage_status": "available",
            "model_id": "synthetic-model",
            "input_tokens": 17,
            "output_tokens": 9,
            "response_count": 1,
        }
    ]


def test_succeeded_logical_call_is_replayed_without_second_paid_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    _mock_usage: list[dict[str, Any]],
) -> None:
    client = _FakeChatClient(
        [
            _Response(
                content=_VALID_REFUSAL,
                usage_metadata={"input_tokens": 7, "output_tokens": 4},
            )
        ]
    )
    _, authorization_calls = _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    first = _invoke(state)
    second = _invoke(state)

    assert first == second
    assert len(client.invoke_calls) == 1
    assert len(authorization_calls) == 1
    assert len(_mock_usage) == 1
    assert state["answer_model_ledger"]["logical_call_count"] == 1
    assert state["answer_model_ledger"]["provider_attempt_count"] == 1


@pytest.mark.parametrize("reserved_output", [ANSWERING_OUTPUT_TOKEN_CAP, 100_000])
def test_configured_adapter_uses_the_fixed_output_cap(reserved_output: int) -> None:
    state = {"context_pack": {"budget": {"reserved_output": reserved_output}}}

    assert _output_cap(state) == ANSWERING_OUTPUT_TOKEN_CAP


@pytest.mark.parametrize("reserved_output", [32, 640, 2_048, "invalid", True, None])
def test_configured_adapter_rejects_insufficient_or_invalid_output_reservation(
    reserved_output: object,
) -> None:
    state = {"context_pack": {"budget": {"reserved_output": reserved_output}}}

    with pytest.raises(AnsweringPromptBudgetError):
        _output_cap(state)


@pytest.mark.parametrize("claim_count", [0, 1, MAX_PROPOSAL_CLAIMS])
def test_configured_adapter_and_gateway_accept_every_schema_valid_claim_cardinality(
    monkeypatch: pytest.MonkeyPatch,
    claim_count: int,
) -> None:
    witnesses = tuple(
        {
            "witness_id": "wit_" + f"{index:040x}",
            "content": f"合成证据 {index}",
        }
        for index in range(max(1, claim_count))
    )
    claims = [
        {
            "obligation_id": _OBLIGATION_ID,
            "witness_id": item["witness_id"],
            "rendering_rule": "exact_witness",
        }
        for item in witnesses[:claim_count]
    ]
    content = json.dumps(
        {
            "decision": "refuse" if claim_count == 0 else "answer",
            "claims": claims,
        },
        separators=(",", ":"),
    )
    client = _FakeChatClient(
        [
            _Response(
                content=content,
                usage_metadata={"input_tokens": 100, "output_tokens": 20},
            )
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    monkeypatch.setattr(
        model_gateway,
        "enforce_answer_prompt_budget",
        lambda *_args, **_kwargs: 100,
    )
    state = _budget_state()
    state["witness_catalog"] = {
        "receipt": {
            "witness_ids": [item["witness_id"] for item in witnesses],
            "witness_count": len(witnesses),
            "universe_status": "complete",
            "omissions": [],
        }
    }

    result = ConfiguredAnsweringModelAdapter().select_plan(
        state,
        question_contract=_question_contract(),
        witness_catalog=witnesses,
    )

    assert len(result.claims) == claim_count
    assert result.decision == (AnswerDecision.REFUSE if claim_count == 0 else AnswerDecision.ANSWER)
    assert len(client.invoke_calls) == 1
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["response_schema"] == "AnswerPlanWireProposal"
    assert receipt["parsed_schema"] == "AnswerPlanProposal"
    assert receipt["status"] == "succeeded"


@pytest.mark.parametrize("claim_count", [0, 1, MAX_PROPOSAL_CLAIMS])
def test_configured_reviewer_and_gateway_accept_every_valid_claim_cardinality(
    monkeypatch: pytest.MonkeyPatch,
    claim_count: int,
) -> None:
    witnesses = tuple(
        {
            "witness_id": "wit_" + f"{index:040x}",
            "content": f"合成证据 {index}",
        }
        for index in range(max(1, claim_count))
    )
    claims = [
        {
            "obligation_id": _OBLIGATION_ID,
            "witness_id": item["witness_id"],
            "rendering_rule": "exact_witness",
        }
        for item in witnesses[:claim_count]
    ]
    content = json.dumps(
        {
            "decision": "accept" if claim_count == 0 else "repair",
            "claims": claims,
        },
        separators=(",", ":"),
    )
    client = _FakeChatClient(
        [
            _Response(
                content=content,
                usage_metadata={"input_tokens": 100, "output_tokens": 20},
            )
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    monkeypatch.setattr(
        model_gateway,
        "enforce_answer_prompt_budget",
        lambda *_args, **_kwargs: 100,
    )
    receipt_payload = {
        "witness_ids": [item["witness_id"] for item in witnesses],
        "witness_count": len(witnesses),
        "universe_status": "complete",
        "omissions": [],
    }
    monkeypatch.setattr(
        answering_provider,
        "_review_accept_contract",
        lambda *_args, **_kwargs: (receipt_payload, (_OBLIGATION_ID,)),
    )
    state = _budget_state()
    state["witness_catalog"] = {"receipt": receipt_payload}
    plan = AnswerPlanProposal(
        decision=AnswerDecision.ANSWER,
        claims=(
            ClaimProposal(
                obligation_id=_OBLIGATION_ID,
                witness_id=witnesses[0]["witness_id"],
                rendering_rule=RenderingRule.EXACT_WITNESS,
            ),
        ),
        reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
    )

    result = ConfiguredAnsweringModelAdapter().review_plan(
        state,
        question_contract=_question_contract(),
        witness_catalog=witnesses,
        plan_proposal=plan,
        repair_attempt=0,
    )

    assert len(result.claims) == claim_count
    assert result.decision == (ReviewDecision.ACCEPT if claim_count == 0 else ReviewDecision.REPAIR)
    assert result.closed_obligation_ids == ((_OBLIGATION_ID,) if claim_count == 0 else ())
    assert len(client.invoke_calls) == 1
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["response_schema"] == "ReviewWireProposal"
    assert receipt["parsed_schema"] == "ReviewProposal"
    assert receipt["status"] == "succeeded"


@pytest.mark.parametrize(
    ("invalid_content", "expected_reason"),
    [
        (
            '{"decision":"refuse","decision":"answer","claims":[]}',
            ProviderWireFailureReason.DUPLICATE_JSON_KEY,
        ),
        (
            '{"decision":"refuse","claims":[],"unexpected":true}',
            ProviderWireFailureReason.SCHEMA_INVALID,
        ),
        (
            "```json\n" + _VALID_REFUSAL + "\n```",
            ProviderWireFailureReason.INVALID_JSON,
        ),
    ],
)
def test_invalid_json_contract_uses_bounded_parse_retry_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    invalid_content: str,
    expected_reason: ProviderWireFailureReason,
) -> None:
    response = _Response(
        content=invalid_content,
        usage_metadata={"input_tokens": 11, "output_tokens": 7},
    )
    client = _FakeChatClient([response, response])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    with pytest.raises(StructuredAnswerModelError) as captured:
        _invoke(state)

    assert isinstance(captured.value.__cause__, ProposalValidationError)
    assert len(client.invoke_calls) == 2
    assert state["answer_prompt_receipts"][0]["status"] == "failed"
    assert state["answer_prompt_receipts"][0]["parse_status"] == ("failed_wire_contract")
    assert state["answer_prompt_receipts"][0]["failure_stage"] == "wire_contract"
    assert state["answer_prompt_receipts"][0]["failure_reason_code"] == (expected_reason.value)
    assert [item["status"] for item in state["answer_prompt_receipts"][0]["response_attempts"]] == [
        "failed",
        "failed",
    ]
    assert [
        item["failure_stage"] for item in state["answer_prompt_receipts"][0]["response_attempts"]
    ] == ["wire_contract", "wire_contract"]
    assert [
        item["failure_reason_code"]
        for item in state["answer_prompt_receipts"][0]["response_attempts"]
    ] == [expected_reason.value, expected_reason.value]
    assert state["usage"]["status"] == "unavailable"
    assert state["model_usage_summaries"][-1] == {
        "scene": "answer_plan_selection",
        "call_status": "failed",
        "usage_status": "unavailable",
        "model_id": "synthetic-model",
        "input_tokens": 11,
        "output_tokens": 7,
    }
    audit = state["query_model_call_audits"][0]
    assert audit["status"] == "failed_attempt_limit"
    assert audit["provider_attempt_count"] == 2
    assert state["answer_model_ledger"]["provider_attempt_count"] == 2
    assert [item["status"] for item in state["answer_model_ledger"]["provider_attempts"]] == [
        "failed_parse",
        "failed_parse",
    ]


def test_three_wire_failures_expose_only_bounded_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_content = '{"decision":"refuse","decision":"answer","claims":[]}'
    response = _Response(
        content=invalid_content,
        usage_metadata={"input_tokens": 11, "output_tokens": 7},
    )
    client = _FakeChatClient([response, response, response])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    state = _budget_state()

    with pytest.raises(StructuredAnswerModelError) as captured:
        _invoke(state)

    error = captured.value
    receipt = state["answer_prompt_receipts"][0]
    assert error.safe_component == "structured_answer_model"
    assert error.safe_scene == "answer_plan_selection"
    assert error.safe_failure_stage == "wire_contract"
    assert error.safe_failure_reason_code == (ProviderWireFailureReason.DUPLICATE_JSON_KEY.value)
    assert error.safe_provider_attempt_count == 3
    assert error.safe_logical_call_id == receipt["logical_call_id"]
    assert receipt["failure_reason_code"] == (ProviderWireFailureReason.DUPLICATE_JSON_KEY.value)
    assert [item["failure_reason_code"] for item in receipt["response_attempts"]] == [
        ProviderWireFailureReason.DUPLICATE_JSON_KEY.value,
        ProviderWireFailureReason.DUPLICATE_JSON_KEY.value,
        ProviderWireFailureReason.DUPLICATE_JSON_KEY.value,
    ]
    assert len(client.invoke_calls) == 3
    assert invalid_content not in json.dumps(receipt, sort_keys=True)


def test_invalid_first_response_retries_and_binds_the_final_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _Response(
        content='{"decision":"answer","claims":[]}',
        usage_metadata={"input_tokens": 13, "output_tokens": 5},
    )
    valid = _Response(
        content=_VALID_REFUSAL,
        usage_metadata={"input_tokens": 17, "output_tokens": 9},
    )
    client = _FakeChatClient([invalid, valid])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    result = _invoke(state)

    assert result.decision == AnswerDecision.REFUSE
    assert len(client.invoke_calls) == 2
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["status"] == "succeeded"
    assert receipt["parse_status"] == "succeeded"
    assert [item["status"] for item in receipt["response_attempts"]] == [
        "failed",
        "wire_valid",
    ]
    assert [item["failure_stage"] for item in receipt["response_attempts"]] == [
        "wire_contract",
        None,
    ]
    assert receipt["response_sha256"] == receipt["response_attempts"][-1]["response_sha256"]
    assert receipt["response_attempts"][0]["response_sha256"] != receipt["response_sha256"]
    assert [item["status"] for item in state["answer_model_ledger"]["provider_attempts"]] == [
        "failed_parse",
        "succeeded",
    ]
    audit = state["query_model_call_audits"][0]
    assert audit["status"] == "succeeded"
    assert audit["provider_attempt_count"] == 2


def test_three_wire_failures_and_success_share_one_logical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = (
        '{"decision":',
        '{"decision":"answer","claims":[]}',
        _VALID_REFUSAL,
    )
    client = _FakeChatClient(
        [
            _Response(
                content=content,
                usage_metadata={"input_tokens": 9, "output_tokens": 4},
            )
            for content in contents
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    state = _budget_state()

    result = _invoke(state)

    assert result.decision == AnswerDecision.REFUSE
    assert len(client.invoke_calls) == 3
    receipt = state["answer_prompt_receipts"][0]
    logical_call_id = receipt["logical_call_id"]
    assert state["answer_model_ledger"]["logical_call_count"] == 1
    assert state["answer_model_ledger"]["logical_calls"] == [
        {
            "logical_call_id": logical_call_id,
            "scene": "answer_plan_selection",
            "prompt_sha256": receipt["prompt_sha256"],
            "input_tokens": receipt["input_tokens"],
            "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
            "status": "succeeded",
        }
    ]
    attempts = receipt["response_attempts"]
    assert [item["attempt_number"] for item in attempts] == [1, 2, 3]
    assert [item["attempt_id"] for item in attempts] == [
        f"{logical_call_id}:pa1",
        f"{logical_call_id}:pa2",
        f"{logical_call_id}:pa3",
    ]
    assert [item["stage"] for item in attempts] == ["wire_parse"] * 3
    assert [item["status"] for item in attempts] == [
        "failed",
        "failed",
        "wire_valid",
    ]
    assert [item["failure_stage"] for item in attempts] == [
        "wire_contract",
        "wire_contract",
        None,
    ]
    assert [item["response_sha256"] for item in attempts] == [
        f"sha256:{sha256(content.encode('utf-8')).hexdigest()}" for content in contents
    ]
    assert state["query_model_call_audits"][0]["logical_call_id"] == logical_call_id


def test_server_projection_value_error_is_original_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_error = ValueError("sensitive server projection invariant")
    client = _FakeChatClient(
        [
            _Response(
                content=_VALID_REFUSAL,
                usage_metadata={"input_tokens": 9, "output_tokens": 4},
            )
            for _ in range(3)
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    state = _budget_state()

    def fail_projection(_wire: AnswerPlanWireProposal) -> AnswerPlanProposal:
        raise server_error

    with pytest.raises(ValueError) as captured:
        _invoke(state, response_projector=fail_projection)

    assert captured.value is server_error
    assert len(client.invoke_calls) == 1
    _assert_failed_terminalized(state, failure_stage="server_projection")
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["response_attempts"][0]["status"] == "wire_valid"
    assert receipt["response_attempts"][0]["failure_stage"] is None
    assert state["answer_model_ledger"]["provider_attempts"][0]["status"] == ("succeeded")
    assert "sensitive server projection invariant" not in json.dumps(receipt)


def test_parser_invariant_value_error_is_original_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_error = ValueError("sensitive parser invariant")
    client = _FakeChatClient(
        [
            _Response(
                content=_VALID_REFUSAL,
                usage_metadata={"input_tokens": 9, "output_tokens": 4},
            )
            for _ in range(3)
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch, max_attempts=3)
    state = _budget_state()

    def fail_parser(_raw: str) -> AnswerPlanWireProposal:
        raise parser_error

    with pytest.raises(ValueError) as captured:
        _invoke(state, response_parser=fail_parser)

    assert captured.value is parser_error
    assert len(client.invoke_calls) == 1
    _assert_failed_terminalized(state, failure_stage="parser_invariant")
    receipt = state["answer_prompt_receipts"][0]
    assert receipt["response_attempts"][0]["failure_stage"] == "parser_invariant"
    assert "sensitive parser invariant" not in json.dumps(receipt)


def test_provider_allowlist_failure_retries_within_the_same_logical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside_allowlist = _Response(
        content=json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "obligation_id": "qf_" + "b" * 32,
                        "witness_id": "wit_" + "0" * 40,
                        "rendering_rule": "exact_witness",
                    }
                ],
            },
            separators=(",", ":"),
        ),
        usage_metadata={"input_tokens": 13, "output_tokens": 7},
    )
    valid = _Response(
        content=_VALID_REFUSAL,
        usage_metadata={"input_tokens": 17, "output_tokens": 5},
    )
    client = _FakeChatClient([outside_allowlist, valid])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    result = _invoke(state)

    assert result.decision == AnswerDecision.REFUSE
    assert len(client.invoke_calls) == 2
    receipt = state["answer_prompt_receipts"][0]
    assert (
        receipt["logical_call_id"]
        == state["answer_model_ledger"]["logical_calls"][0]["logical_call_id"]
    )
    assert [item["status"] for item in receipt["response_attempts"]] == [
        "failed",
        "wire_valid",
    ]
    assert set(receipt["response_attempts"][0]) == {
        "attempt_id",
        "attempt_number",
        "stage",
        "status",
        "failure_stage",
        "failure_reason_code",
        "response_sha256",
    }
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert "qf_" + "b" * 32 not in serialized_receipt
    assert "synthetic opaque task" not in serialized_receipt


def test_provider_invoke_value_error_is_not_misclassified_as_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_error = ValueError("synthetic client configuration defect")
    client = _FakeChatClient([provider_error])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    with pytest.raises(ValueError) as captured:
        _invoke(state)

    assert captured.value is provider_error
    assert not isinstance(captured.value, StructuredAnswerModelError)
    assert len(client.invoke_calls) == 1
    _assert_failed_terminalized(state, failure_stage="transport")
    assert state["usage"]["status"] == "unavailable"
    assert "synthetic client configuration defect" not in json.dumps(
        state["answer_prompt_receipts"][0]
    )
    audit = state["query_model_call_audits"][0]
    assert audit["status"] == "failed_non_retryable"
    assert audit["provider_attempt_count"] == 1


def test_transport_retry_is_audited_without_prompt_or_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    client = _FakeChatClient(
        [
            httpx.ConnectTimeout("secret transport body", request=request),
            _Response(
                content=_VALID_REFUSAL,
                usage_metadata={"input_tokens": 5, "output_tokens": 3},
            ),
        ]
    )
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state()

    _invoke(state)

    assert len(client.invoke_calls) == 2
    audit = state["query_model_call_audits"][0]
    assert audit["status"] == "succeeded"
    assert audit["provider_attempt_count"] == 2
    assert audit["retry_count"] == 1
    ledger = state["answer_model_ledger"]
    assert ledger["provider_attempt_count"] == 2
    assert ledger["reserved_input_tokens"] == 2_200
    assert ledger["reserved_output_tokens"] == 2 * ANSWERING_OUTPUT_TOKEN_CAP
    assert ledger["accounted_input_tokens"] == 1_105
    assert ledger["accounted_output_tokens"] == ANSWERING_OUTPUT_TOKEN_CAP + 3
    assert ledger["usage_metadata_complete"] is False
    serialized = json.dumps(audit)
    assert "secret transport body" not in serialized
    assert "synthetic opaque task" not in serialized


def test_exhausted_query_deadline_never_invokes_provider_and_records_failure_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([])
    _install_provider(monkeypatch, client)
    _install_fast_retry(monkeypatch)
    state = _budget_state(deadline_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat())

    with pytest.raises(QueryModelCallDeadlineExceeded):
        _invoke(state)

    assert client.invoke_calls == []
    assert state["answer_prompt_receipts"][0]["status"] == "failed"
    assert state["usage"]["status"] == "unavailable"
    assert state["model_usage_summaries"][-1]["call_status"] == "failed"
    audit = state["query_model_call_audits"][0]
    assert audit["status"] == "deadline_exhausted"
    assert audit["provider_attempt_count"] == 0


def test_output_cap_drift_is_blocked_before_authorization_or_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([])
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)
    state = _budget_state()

    with pytest.raises(AnsweringPromptBudgetError):
        _invoke(state, output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP + 1)

    assert authorization_calls == []
    assert chat_calls == []
    assert client.bind_calls == []
    assert "answer_prompt_receipts" not in state
    assert "model_usage_summaries" not in state


def test_run_token_budget_drift_is_blocked_before_authorization_or_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([])
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)
    state = _budget_state()
    state["answer_prompt_budget"]["answer_run_tokens_available"] -= 1

    with pytest.raises(AnsweringPromptBudgetError, match="run token budget"):
        _invoke(state)

    assert authorization_calls == []
    assert chat_calls == []
    assert client.invoke_calls == []


def test_missing_provider_host_is_blocked_before_authorization_or_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([])
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)
    model_gateway.settings.openai_base_url = ""
    state = _budget_state()

    with pytest.raises(AnsweringPromptBudgetError, match="safe provider URL"):
        _invoke(state)

    assert authorization_calls == []
    assert chat_calls == []
    assert client.invoke_calls == []
    _assert_failed_terminalized(state, failure_stage="endpoint_validation")


def test_configured_host_drift_is_blocked_before_authorization_or_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([])
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)
    state = _budget_state()
    state["egress_policies"]["llm_generation"]["allowed_domains"] = ["frozen.example"]

    with pytest.raises(AnsweringPromptBudgetError, match="frozen domain allowlist"):
        _invoke(state)

    assert authorization_calls == []
    assert chat_calls == []
    assert client.invoke_calls == []
    _assert_failed_terminalized(state, failure_stage="endpoint_validation")


@pytest.mark.parametrize(
    ("injected_stage", "expected_failure_stage"),
    [
        ("authorization", "authorization"),
        ("client_init", "client_init"),
        ("bind", "bind"),
        ("bound_endpoint", "endpoint_validation"),
    ],
)
def test_setup_failures_after_receipt_registration_reach_one_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    injected_stage: str,
    expected_failure_stage: str,
) -> None:
    setup_error = ValueError("sensitive setup failure")
    client = _FakeChatClient(
        [],
        bind_error=setup_error if injected_stage == "bind" else None,
        bound_base_url=(
            "https://bound-drift.example/v1" if injected_stage == "bound_endpoint" else None
        ),
    )
    _install_provider(monkeypatch, client)
    if injected_stage == "authorization":
        monkeypatch.setattr(
            model_gateway,
            "authorize_external_call",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(setup_error),
        )
    elif injected_stage == "client_init":
        monkeypatch.setattr(
            model_gateway.llm_provider,
            "chat",
            lambda **_kwargs: (_ for _ in ()).throw(setup_error),
        )
    state = _budget_state()

    expected_error = (
        AnsweringPromptBudgetError if injected_stage == "bound_endpoint" else ValueError
    )
    with pytest.raises(expected_error) as captured:
        _invoke(state)

    if injected_stage != "bound_endpoint":
        assert captured.value is setup_error
    assert client.invoke_calls == []
    _assert_failed_terminalized(state, failure_stage=expected_failure_stage)
    assert "sensitive setup failure" not in json.dumps(state["answer_prompt_receipts"][0])


def test_client_endpoint_drift_is_blocked_before_sdk_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeChatClient([], base_url="https://drift.example/v1")
    chat_calls, authorization_calls = _install_provider(monkeypatch, client)

    state = _budget_state()
    with pytest.raises(AnsweringPromptBudgetError, match="client target"):
        _invoke(state)

    assert authorization_calls == [
        {
            "purpose": "llm_generation",
            "provider": "configured-llm",
            "target_domain": "llm.example",
        }
    ]
    assert chat_calls == [{"model": "synthetic-model", "json_mode": True}]
    assert client.invoke_calls == []
    _assert_failed_terminalized(state, failure_stage="endpoint_validation")
