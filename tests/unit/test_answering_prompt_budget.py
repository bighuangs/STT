from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from app.application.query_answering.proposals import (
    MAX_PROPOSAL_BYTES,
    MAX_PROPOSAL_CLAIMS,
    AnswerDecision,
    AnswerPlanWireProposal,
    AnswerPlanProposal,
    ClaimProposal,
    PlanReasonCode,
    RenderingRule,
    ReviewDecision,
    ReviewWireProposal,
    parse_answer_plan_proposal_json,
)
from app.infrastructure.answering import model_gateway
from app.infrastructure.answering.model_gateway import (
    AnsweringPromptBudgetError,
    enforce_answer_prompt_budget,
    invoke_structured_answer_model,
)
from app.infrastructure.answering.prompts import (
    ANSWERING_MAX_LOGICAL_CALLS,
    ANSWERING_OUTPUT_TOKEN_CAP,
    ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION,
    MAX_PLAN_PROPOSAL_UTF8_BYTES,
    MAX_REVIEW_PROPOSAL_UTF8_BYTES,
    PLAN_PROMPT_VERSION,
    PromptAssemblyError,
    REVIEW_PROMPT_VERSION,
    build_plan_prompt_payload,
    build_review_prompt_payload,
    estimate_answering_prompt_path_budget,
    estimate_plan_prompt,
    estimate_review_prompt,
    resolve_answering_model_limits,
)
from app.rag.query.context_assembly_service import _answer_run_token_budget, _limits
from app.rag.query.context_pack_service import (
    TiktokenTokenCounter,
    Utf8ByteUpperBoundCounter,
)
from app.domain.retrieval import ContextPackProfile
from app.process.query.agent.state import QueryGraphState, create_query_default_state


def _question() -> dict[str, Any]:
    return {
        "source_query": "合成设备启动前需要核对什么？",
        "obligations": [
            {
                "obligation_id": "qf_" + "a" * 32,
                "information_need": "合成启动核对项",
                "kind": "fact",
                "necessity": "required",
                "polarity": "affirmative",
                "closure": "single",
                "expected_count": None,
                "required_output_form": "fact",
            }
        ],
    }


def _witnesses(count: int = MAX_PROPOSAL_CLAIMS) -> list[dict[str, str]]:
    return [
        {
            "witness_id": "wit_" + f"{index:040x}",
            "content": f"合成见证原子 {index}。",
        }
        for index in range(count)
    ]


def _catalog_receipt(
    witnesses: list[dict[str, str]],
    *,
    universe_status: str = "complete",
) -> dict[str, Any]:
    omissions = [] if universe_status == "complete" else [{"code": "atom_limit", "count": 1}]
    return {
        "witness_ids": [item["witness_id"] for item in witnesses],
        "witness_count": len(witnesses),
        "universe_status": universe_status,
        "omissions": omissions,
    }


def _claims(count: int = MAX_PROPOSAL_CLAIMS) -> tuple[ClaimProposal, ...]:
    return tuple(
        ClaimProposal(
            obligation_id="qf_" + "a" * 32,
            witness_id="wit_" + f"{index:040x}",
            rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
        )
        for index in range(count)
    )


def _maximum_plan() -> AnswerPlanProposal:
    return AnswerPlanProposal(
        decision=AnswerDecision.PARTIAL,
        claims=_claims(),
        reason_codes=(
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
    )


def _ready_state(
    *,
    worst_input_tokens: int,
    model_id: str = "synthetic-model",
    model_limit: int = 20_000,
    configured_model_limit: int | None = None,
    provider_max_input_tokens: int | None = None,
    model_limits_version: str = "retrieval-profile.context-pack.v1",
    model_limits_binding_mode: str = "profile_configured",
    answer_run_token_budget: int | None = None,
) -> dict[str, Any]:
    content_hash = "sha256:" + "b" * 64
    configured_model_limit = configured_model_limit or model_limit
    provider_max_input_tokens = provider_max_input_tokens or model_limit
    reserved_output = ANSWERING_OUTPUT_TOKEN_CAP
    safety_margin = 256
    max_input = min(
        model_limit - reserved_output - safety_margin,
        provider_max_input_tokens - safety_margin,
    )
    attempts_per_call = 3
    max_provider_attempts = ANSWERING_MAX_LOGICAL_CALLS * attempts_per_call
    provider_attempt_input_cap = worst_input_tokens + safety_margin
    max_total_input = max_provider_attempts * provider_attempt_input_cap
    max_total_output = max_provider_attempts * ANSWERING_OUTPUT_TOKEN_CAP
    max_total = max_total_input + max_total_output
    answer_run_token_budget = answer_run_token_budget or max_total
    ledger_limits = {
        "max_logical_calls": ANSWERING_MAX_LOGICAL_CALLS,
        "max_provider_attempts_per_logical_call": attempts_per_call,
        "max_provider_attempts": max_provider_attempts,
        "provider_attempt_input_token_cap": provider_attempt_input_cap,
        "provider_attempt_output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "max_total_tokens": max_total,
    }
    return {
        "llm_model_id": model_id,
        "context_budget_blocked": False,
        "context_pack": {
            "content_hash": content_hash,
            "tokenizer_id": "utf8-byte-upper-bound",
            "tokenizer_version": "v1",
            "budget": {
                "model_context_limit": model_limit,
                "reserved_output": reserved_output,
                "safety_margin": safety_margin,
            },
        },
        "answer_prompt_budget": {
            "schema_version": ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION,
            "status": "ready",
            "context_pack_hash": content_hash,
            "llm_model_id": model_id,
            "tokenizer_id": "utf8-byte-upper-bound",
            "tokenizer_version": "v1",
            "tokenizer_binding_mode": "utf8_byte_upper_bound",
            "model_context_limit": model_limit,
            "configured_model_context_limit": configured_model_limit,
            "provider_max_input_tokens": provider_max_input_tokens,
            "model_limits_version": model_limits_version,
            "model_limits_binding_mode": model_limits_binding_mode,
            "reserved_output_tokens": reserved_output,
            "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
            "safety_margin_tokens": safety_margin,
            "max_input_tokens": max_input,
            "worst_input_tokens": worst_input_tokens,
            **ledger_limits,
            "max_total_input_tokens": max_total_input,
            "max_total_output_tokens": max_total_output,
            "answer_run_token_budget": answer_run_token_budget,
            "answer_run_tokens_available": answer_run_token_budget,
            "cost_budget_status": "unavailable",
            "cost_limit": None,
            "cost_currency": None,
        },
        "answer_model_ledger": {
            "schema_version": "answer-model-budget-ledger.v1",
            "status": "ready",
            "context_pack_hash": content_hash,
            "llm_model_id": model_id,
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


def test_sixteen_claim_outputs_fit_the_minimum_provider_agnostic_utf8_bound() -> None:
    assert MAX_PROPOSAL_CLAIMS == 16
    assert MAX_PROPOSAL_BYTES == ANSWERING_OUTPUT_TOKEN_CAP == 2_561
    assert MAX_PLAN_PROPOSAL_UTF8_BYTES == ANSWERING_OUTPUT_TOKEN_CAP == 2_561
    assert MAX_REVIEW_PROPOSAL_UTF8_BYTES == 2_560
    maximum_accept = ReviewWireProposal(
        decision=ReviewDecision.ACCEPT,
        claims=(),
    )
    accept_bytes = len(
        json.dumps(
            maximum_accept.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert accept_bytes == 33
    assert accept_bytes < ANSWERING_OUTPUT_TOKEN_CAP
    maximum_repair = ReviewWireProposal(
        decision=ReviewDecision.REPAIR,
        claims=_claims(MAX_PROPOSAL_CLAIMS),
    )
    repair_bytes = len(
        json.dumps(
            maximum_repair.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert repair_bytes == MAX_REVIEW_PROPOSAL_UTF8_BYTES == 2_560
    with pytest.raises(ValidationError):
        ReviewWireProposal(
            decision=ReviewDecision.ACCEPT,
            claims=_claims(MAX_PROPOSAL_CLAIMS),
        )


def test_versioned_prompt_schemas_expose_only_semantic_wire_fields() -> None:
    witnesses = _witnesses(1)
    receipt = _catalog_receipt(witnesses)
    plan_payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )
    review_payload = build_review_prompt_payload(
        _question(),
        witnesses,
        AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=_claims(1),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        ),
        witness_catalog_receipt=receipt,
    )
    repair_review_payload = build_review_prompt_payload(
        _question(),
        witnesses,
        AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=_claims(1),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        ),
        witness_catalog_receipt=receipt,
        repair_attempt=1,
    )
    final_review_payload = build_review_prompt_payload(
        _question(),
        witnesses,
        AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=_claims(1),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        ),
        witness_catalog_receipt=receipt,
        repair_attempt=2,
    )

    assert PLAN_PROMPT_VERSION == "1.9.0-draft"
    assert REVIEW_PROMPT_VERSION == "1.9.0-draft"
    assert ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION == "answer-prompt-budget.v4"
    assert ANSWERING_MAX_LOGICAL_CALLS == 4
    assert set(plan_payload["input_schema"]["properties"]) == {
        "question_contract",
        "witness_catalog",
        "witness_catalog_summary",
    }
    assert set(plan_payload["output_schema"]["properties"]) == {"decision", "claims"}
    assert set(review_payload["output_schema"]["properties"]) == {"decision", "claims"}
    assert plan_payload["output_schema"]["properties"]["claims"]["maxItems"] == 16
    assert review_payload["output_schema"]["properties"]["claims"]["maxItems"] == 16
    assert "reason_codes" not in json.dumps(
        plan_payload["output_schema"],
        sort_keys=True,
    )
    assert "reason_codes" not in json.dumps(
        review_payload["output_schema"],
        sort_keys=True,
    )
    assert '"reason_codes"' not in review_payload["messages"][1]["content"]
    assert '"reason_codes"' not in repair_review_payload["messages"][1]["content"]
    assert '"reason_codes"' not in final_review_payload["messages"][1]["content"]
    assert '"repair_attempt"' not in plan_payload["messages"][1]["content"]
    assert '"repair_feedback"' not in plan_payload["messages"][1]["content"]
    assert '"repair_attempt":1' in repair_review_payload["messages"][1]["content"]
    assert '"repair_attempt":2' in final_review_payload["messages"][1]["content"]
    assert "closed_obligation_ids" not in json.dumps(
        review_payload["output_schema"],
        sort_keys=True,
    )
    decision_branches = review_payload["output_schema"]["oneOf"]
    assert [item["properties"]["decision"]["const"] for item in decision_branches] == [
        "accept",
        "repair",
        "reject",
    ]
    assert decision_branches[0]["properties"]["claims"]["maxItems"] == 0
    assert decision_branches[1]["properties"]["claims"]["minItems"] == 1


def test_seventeenth_claim_is_rejected_by_both_proposal_contracts() -> None:
    with pytest.raises(ValidationError):
        AnswerPlanWireProposal(
            decision=AnswerDecision.PARTIAL,
            claims=_claims(MAX_PROPOSAL_CLAIMS + 1),
        )
    with pytest.raises(ValidationError):
        ReviewWireProposal(
            decision=ReviewDecision.REPAIR,
            claims=_claims(MAX_PROPOSAL_CLAIMS + 1),
        )


def test_four_path_budget_upper_bounds_every_maximum_actual_prompt() -> None:
    question = _question()
    witnesses = _witnesses()
    counter = TiktokenTokenCounter()
    receipt = _catalog_receipt(witnesses)
    path = estimate_answering_prompt_path_budget(
        question,
        witnesses,
        witness_catalog_receipt=receipt,
        token_counter=counter.count,
    )
    plan = _maximum_plan()

    assert path.initial_plan_tokens >= estimate_plan_prompt(
        question,
        witnesses,
        witness_catalog_receipt=receipt,
        token_counter=counter.count,
    )
    assert path.initial_review_tokens >= estimate_review_prompt(
        question,
        witnesses,
        plan,
        witness_catalog_receipt=receipt,
        token_counter=counter.count,
    )
    assert path.repair_review_tokens >= estimate_review_prompt(
        question,
        witnesses,
        plan,
        witness_catalog_receipt=receipt,
        repair_attempt=1,
        token_counter=counter.count,
    )
    assert path.final_review_tokens >= estimate_review_prompt(
        question,
        witnesses,
        plan,
        witness_catalog_receipt=receipt,
        repair_attempt=2,
        token_counter=counter.count,
    )
    assert set(path.path_tokens) == {
        "initial_plan",
        "initial_review",
        "repair_review",
        "final_review",
    }


def test_safe_mapping_budget_omission_is_a_valid_truncation_prompt_fact() -> None:
    witnesses = _witnesses(1)
    receipt = {
        **_catalog_receipt(witnesses, universe_status="truncated"),
        "omissions": [{"code": "safe_mapping_budget_exceeded", "count": 2}],
    }

    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )

    assert '"safe_mapping_budget_exceeded"' in payload["messages"][1]["content"]


def test_legacy_profile_output_reservation_is_safely_raised() -> None:
    _, reserved_output, _, _ = _limits(
        {
            "llm_model_id": "synthetic-model",
            "retrieval_profile": {"context_pack": {"reserved_output_tokens": 1_024}},
        }
    )
    assert reserved_output == ANSWERING_OUTPUT_TOKEN_CAP


def test_default_answer_run_budget_is_independent_from_legacy_planning_budget() -> None:
    state = {
        "query_plan": {"token_budget": 4_096},
        "retrieval_profile": {"context_pack": {}},
    }

    assert _answer_run_token_budget(state) == 12_000_000
    assert _answer_run_token_budget(state) > state["query_plan"]["token_budget"]
    assert ContextPackProfile().answer_run_token_budget == 12_000_000


def test_qwen37_max_uses_the_frozen_official_context_limits() -> None:
    assert resolve_answering_model_limits(
        "qwen3.7-max",
        configured_context_limit=8_192,
    ) == (
        1_000_000,
        991_808,
        "aliyun-model-studio.qwen3.7-max.2026-07-24",
        "provider_exact",
    )
    assert resolve_answering_model_limits(
        "qwen3.7-max-unverified-alias",
        configured_context_limit=8_192,
    ) == (
        8_192,
        8_192,
        "retrieval-profile.context-pack.v1",
        "profile_configured",
    )


@pytest.mark.parametrize("witness_count", [1, MAX_PROPOSAL_CLAIMS])
def test_qwen37_minimum_and_normal_byte_prompts_pass_preflight(
    witness_count: int,
) -> None:
    witnesses = _witnesses(witness_count)
    receipt = _catalog_receipt(witnesses)
    counter = Utf8ByteUpperBoundCounter()
    path = estimate_answering_prompt_path_budget(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
        token_counter=counter.count,
    )
    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )
    state = _ready_state(
        worst_input_tokens=path.worst_input_tokens,
        model_id="qwen3.7-max",
        model_limit=1_000_000,
        configured_model_limit=8_192,
        provider_max_input_tokens=991_808,
        model_limits_version="aliyun-model-studio.qwen3.7-max.2026-07-24",
        model_limits_binding_mode="provider_exact",
        answer_run_token_budget=12_000_000,
    )

    actual = enforce_answer_prompt_budget(
        state,
        scene="answer_plan_selection",
        prompt=payload["messages"],
        output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP,
    )

    assert path.worst_input_tokens < 991_808 - 256
    assert 0 < actual <= path.worst_input_tokens


def test_qwen37_stale_profile_window_binding_is_rejected() -> None:
    witnesses = _witnesses(1)
    receipt = _catalog_receipt(witnesses)
    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )
    state = _ready_state(
        worst_input_tokens=10_000,
        model_id="qwen3.7-max",
        model_limit=1_000_000,
        configured_model_limit=8_192,
        provider_max_input_tokens=991_808,
        model_limits_version="aliyun-model-studio.qwen3.7-max.2026-07-24",
        model_limits_binding_mode="provider_exact",
        answer_run_token_budget=12_000_000,
    )
    state["answer_prompt_budget"]["model_context_limit"] = 8_192

    with pytest.raises(AnsweringPromptBudgetError, match="invariants"):
        enforce_answer_prompt_budget(
            state,
            scene="answer_plan_selection",
            prompt=payload["messages"],
            output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP,
        )


def test_compiled_graph_preserves_budget_receipt_and_preflights_across_nodes() -> None:
    budget = {"schema_version": ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION, "status": "ready"}
    ledger = {"schema_version": "answer-model-budget-ledger.v1", "status": "ready"}
    first_preflight = {"scene": "answer_plan_selection", "status": "ready"}
    second_preflight = {"scene": "answer_plan_semantic_review", "status": "ready"}

    def write_budget(_state: QueryGraphState) -> dict[str, Any]:
        return {
            "answer_prompt_budget": budget,
            "answer_prompt_preflights": [first_preflight],
            "answer_model_ledger": ledger,
        }

    def consume_budget(state: QueryGraphState) -> dict[str, Any]:
        assert state["answer_prompt_budget"] == budget
        assert state["answer_prompt_preflights"] == [first_preflight]
        assert state["answer_model_ledger"] == ledger
        return {
            "answer_prompt_preflights": [
                *state["answer_prompt_preflights"],
                second_preflight,
            ]
        }

    graph = StateGraph(QueryGraphState)
    graph.add_node("write_budget", write_budget)
    graph.add_node("consume_budget", consume_budget)
    graph.set_entry_point("write_budget")
    graph.add_edge("write_budget", "consume_budget")
    graph.add_edge("consume_budget", END)

    result = graph.compile().invoke(create_query_default_state())

    assert result["answer_prompt_budget"] == budget
    assert result["answer_prompt_preflights"] == [first_preflight, second_preflight]
    assert result["answer_model_ledger"] == ledger


def test_single_fact_prompt_requires_one_strongest_exactly_relevant_witness() -> None:
    witnesses = _witnesses(1)
    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=_catalog_receipt(witnesses),
    )
    user_message = payload["messages"][1]["content"]

    assert '"kind":"fact"' in user_message
    assert '"closure":"single"' in user_message
    assert '"required_output_form":"fact"' in user_message
    system_message = payload["messages"][0]["content"]
    assert "exact source_query" in system_message
    assert "exact information_need" in system_message
    assert "exactly one strongest witness atom" in system_message
    assert "Do not select redundant corroborating atoms" in system_message


def test_non_complete_catalog_is_explicit_in_plan_and_review_prompts() -> None:
    witnesses = _witnesses(1)
    receipt = _catalog_receipt(witnesses, universe_status="truncated")
    plan = AnswerPlanProposal(
        decision=AnswerDecision.PARTIAL,
        claims=_claims(1),
        reason_codes=(
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
    )

    plan_payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )
    review_payload = build_review_prompt_payload(
        _question(),
        witnesses,
        plan,
        witness_catalog_receipt=receipt,
    )

    for payload in (plan_payload, review_payload):
        assert '"universe_status":"truncated"' in payload["messages"][1]["content"]
        assert '"code":"atom_limit"' in payload["messages"][1]["content"]
    assert "If universe_status is not complete" in plan_payload["messages"][0]["content"]
    normalized_review_prompt = " ".join(review_payload["messages"][0]["content"].split())
    assert "complete answer proposal cannot be accepted" in normalized_review_prompt
    assert "semantically valid partial proposal may be" in normalized_review_prompt


def test_open_enumeration_prompt_defaults_to_partial_with_exact_source_exception() -> None:
    question = _question()
    question["obligations"][0].update(
        {
            "kind": "enumeration",
            "closure": "open_enumeration",
            "required_output_form": "list",
        }
    )
    witnesses = _witnesses(1)
    receipt = _catalog_receipt(witnesses)
    plan_payload = build_plan_prompt_payload(
        question,
        witnesses,
        witness_catalog_receipt=receipt,
    )
    plan = AnswerPlanProposal(
        decision=AnswerDecision.PARTIAL,
        claims=_claims(1),
        reason_codes=(
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
    )
    review_payload = build_review_prompt_payload(
        question,
        witnesses,
        plan,
        witness_catalog_receipt=receipt,
    )

    assert (
        "normally select safe distinct supplied items only as partial"
        in (plan_payload["messages"][0]["content"])
    )
    assert "selected witnesses exactly" in plan_payload["messages"][0]["content"]
    assert "equal, in supplied catalog order" in plan_payload["messages"][0]["content"]
    assert "typed procedure/list" in review_payload["messages"][0]["content"]
    assert "answer decision is normally partial" in review_payload["messages"][0]["content"]
    assert '"source_closed_open_obligation_ids":[]' in (plan_payload["messages"][1]["content"])

    source_closed_witnesses = _witnesses(1)
    source_closed_witnesses[0]["source_closed_open_obligation_ids"] = [
        question["obligations"][0]["obligation_id"]
    ]
    source_closed_payload = build_plan_prompt_payload(
        question,
        source_closed_witnesses,
        witness_catalog_receipt=_catalog_receipt(source_closed_witnesses),
    )
    assert (
        '"source_closed_open_obligation_ids":["qf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]'
        in source_closed_payload["messages"][1]["content"]
    )


def test_review_prompt_freezes_closure_and_full_replacement_policy() -> None:
    witnesses = _witnesses(1)
    payload = build_review_prompt_payload(
        _question(),
        witnesses,
        AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=_claims(1),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        ),
        witness_catalog_receipt=_catalog_receipt(witnesses),
    )
    system_message = payload["messages"][0]["content"]

    assert "exact source_query and exact information_need" in system_message
    assert "exactly one strongest witness atom" in system_message
    assert "exactly expected_count distinct" in system_message
    assert "complete replacement proposal" in system_message
    assert "including every safe retained claim" in system_message
    assert "runtime, not the model" in system_message
    assert "Never output closed_obligation_ids" in system_message
    assert "Never reject solely because a semantically valid proposal is partial" in system_message
    assert '{"decision":"accept","claims":[]}' in system_message


def test_procedure_prompt_projection_accepts_ordered_multi_item_contract() -> None:
    question = _question()
    question["obligations"][0].update(
        {
            "kind": "procedure",
            "closure": "open_enumeration",
            "required_output_form": "ordered_steps",
        }
    )
    witnesses = _witnesses(2)

    payload = build_plan_prompt_payload(
        question,
        witnesses,
        witness_catalog_receipt=_catalog_receipt(witnesses),
    )

    user_message = payload["messages"][1]["content"]
    system_message = payload["messages"][0]["content"]
    assert '"kind":"procedure"' in user_message
    assert '"required_output_form":"ordered_steps"' in user_message
    assert "keep their supplied catalog order" in system_message


def test_procedure_prompt_projection_rejects_single_ordered_steps_drift() -> None:
    question = _question()
    question["obligations"][0].update(
        {
            "kind": "procedure",
            "closure": "single",
            "required_output_form": "ordered_steps",
        }
    )
    witnesses = _witnesses(1)

    with pytest.raises(PromptAssemblyError, match="question contract prompt projection"):
        build_plan_prompt_payload(
            question,
            witnesses,
            witness_catalog_receipt=_catalog_receipt(witnesses),
        )


def test_ready_receipt_allows_only_a_prompt_within_the_frozen_worst_path() -> None:
    witnesses = _witnesses()
    receipt = _catalog_receipt(witnesses)
    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
    )
    path = estimate_answering_prompt_path_budget(
        _question(),
        witnesses,
        witness_catalog_receipt=receipt,
        token_counter=Utf8ByteUpperBoundCounter().count,
    )
    state = _ready_state(worst_input_tokens=path.worst_input_tokens)

    actual = enforce_answer_prompt_budget(
        state,
        scene="answer_plan_selection",
        prompt=payload["messages"],
        output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP,
    )

    assert 0 < actual <= path.worst_input_tokens
    assert state["answer_prompt_preflights"][0]["status"] == "ready"


def test_unknown_model_cannot_claim_an_unverified_tiktoken_binding() -> None:
    witnesses = _witnesses(1)
    payload = build_plan_prompt_payload(
        _question(),
        witnesses,
        witness_catalog_receipt=_catalog_receipt(witnesses),
    )
    state = _ready_state(worst_input_tokens=10_000)
    state["context_pack"]["tokenizer_id"] = "tiktoken:cl100k_base"
    state["context_pack"]["tokenizer_version"] = "unverified"
    state["answer_prompt_budget"]["tokenizer_id"] = "tiktoken:cl100k_base"
    state["answer_prompt_budget"]["tokenizer_version"] = "unverified"
    state["answer_prompt_budget"]["tokenizer_binding_mode"] = "model_exact"

    with pytest.raises(AnsweringPromptBudgetError, match="tokenizer"):
        enforce_answer_prompt_budget(
            state,
            scene="answer_plan_selection",
            prompt=payload["messages"],
            output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP,
        )


def test_blocked_budget_stops_before_authorization_or_client_creation(monkeypatch) -> None:
    calls = {"authorization": 0, "client": 0}

    def authorization(*_args, **_kwargs) -> None:
        calls["authorization"] += 1

    def client(*_args, **_kwargs):
        calls["client"] += 1
        raise AssertionError("provider client must not be created")

    monkeypatch.setattr(model_gateway, "authorize_external_call", authorization)
    monkeypatch.setattr(model_gateway.llm_provider, "chat", client)
    state = {"context_budget_blocked": True}

    with pytest.raises(AnsweringPromptBudgetError):
        witnesses = _witnesses()
        invoke_structured_answer_model(
            state,
            scene="answer_plan_selection",
            prompt=build_plan_prompt_payload(
                _question(),
                witnesses,
                witness_catalog_receipt=_catalog_receipt(witnesses),
            )["messages"],
            response_model=AnswerPlanProposal,
            response_parser=parse_answer_plan_proposal_json,
            prompt_version="synthetic-budget-test.v1",
            output_token_cap=ANSWERING_OUTPUT_TOKEN_CAP,
        )

    assert calls == {"authorization": 0, "client": 0}
