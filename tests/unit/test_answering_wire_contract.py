from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.application.query_answering.proposals import (
    ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION,
    MAX_PROPOSAL_BYTES,
    MAX_PROPOSAL_CLAIMS,
    REVIEW_PROPOSAL_SCHEMA_VERSION,
    AnswerDecision,
    AnswerPlanProposal,
    AnswerPlanWireProposal,
    ClaimProposal,
    PlanReasonCode,
    ProposalValidationError,
    ProviderWireFailureReason,
    RenderingRule,
    ReviewDecision,
    ReviewProposal,
    ReviewReasonCode,
    ReviewWireProposal,
    parse_answer_plan_proposal_json,
    parse_answer_plan_wire_proposal_json,
    parse_review_wire_proposal_json,
    project_answer_plan_wire_proposal,
    project_review_wire_proposal,
    validate_claim_allowlists,
    validate_review_wire_allowlists,
)
from app.domain.answering.question import QuestionContract
from app.domain.answering.witness import WitnessUniverseStatus
from app.domain.query_planning import FacetClosure, QueryFacet
from app.infrastructure.answering import provider
from app.infrastructure.answering.prompts import ANSWERING_OUTPUT_TOKEN_CAP


_OBLIGATION_ID = "qf_" + "a" * 32
_OTHER_OBLIGATION_ID = "qf_" + "b" * 32
_WITNESS_ID = "wit_" + "1" * 40
_OTHER_WITNESS_ID = "wit_" + "2" * 40


def _claim(
    *,
    obligation_id: str = _OBLIGATION_ID,
    witness_id: str = _WITNESS_ID,
) -> ClaimProposal:
    return ClaimProposal(
        obligation_id=obligation_id,
        witness_id=witness_id,
        rendering_rule=RenderingRule.EXACT_WITNESS,
    )


def _claims(count: int) -> tuple[ClaimProposal, ...]:
    return tuple(_claim(witness_id="wit_" + f"{index:040x}") for index in range(count))


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
        builder_version="wire-contract-test.v1",
    )


def _adapter_state() -> dict[str, object]:
    return {
        "context_pack": {"budget": {"reserved_output": ANSWERING_OUTPUT_TOKEN_CAP}},
        "witness_catalog": {
            "receipt": {
                "witness_ids": [_WITNESS_ID],
                "witness_count": 1,
                "universe_status": "complete",
                "omissions": [],
            }
        },
    }


@pytest.mark.parametrize(
    ("decision", "claims", "expected_reasons"),
    [
        (
            AnswerDecision.ANSWER,
            (_claim(),),
            (PlanReasonCode.ANSWER_SUPPORTED,),
        ),
        (
            AnswerDecision.PARTIAL,
            (_claim(),),
            (
                PlanReasonCode.PARTIAL_SUPPORT,
                PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
            ),
        ),
        (
            AnswerDecision.REFUSE,
            (),
            (PlanReasonCode.NO_SUPPORTING_WITNESS,),
        ),
    ],
)
def test_plan_wire_projects_only_server_owned_reason_codes(
    decision: AnswerDecision,
    claims: tuple[ClaimProposal, ...],
    expected_reasons: tuple[PlanReasonCode, ...],
) -> None:
    canonical = project_answer_plan_wire_proposal(
        AnswerPlanWireProposal(decision=decision, claims=claims)
    )

    assert canonical.decision == decision
    assert canonical.claims == claims
    assert canonical.reason_codes == expected_reasons


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"refuse","decision":"answer","claims":[]}',
        '{"decision":"refuse","claims":[],"reason_codes":["NO_SUPPORTING_WITNESS"]}',
        '{"decision":"unknown","claims":[]}',
        '{"decision":"answer","claims":[]}',
        json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "obligation_id": "invalid",
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "free_text",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "decision": "partial",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": "wit_" + f"{index:040x}",
                        "rendering_rule": "exact_witness",
                    }
                    for index in range(MAX_PROPOSAL_CLAIMS + 1)
                ],
            }
        ),
        '{"decision":"refuse","claims":[],"padding":"' + "x" * MAX_PROPOSAL_BYTES + '"}',
    ],
)
def test_plan_wire_parser_remains_fail_closed(raw: str) -> None:
    with pytest.raises(ProposalValidationError):
        parse_answer_plan_wire_proposal_json(raw)


def test_sixteen_claim_plan_and_repair_wires_round_trip_at_the_shared_limit() -> None:
    claims = _claims(MAX_PROPOSAL_CLAIMS)
    plan_raw = json.dumps(
        AnswerPlanWireProposal(decision=AnswerDecision.PARTIAL, claims=claims).model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    review_raw = json.dumps(
        ReviewWireProposal(decision=ReviewDecision.REPAIR, claims=claims).model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert MAX_PROPOSAL_CLAIMS == 16
    assert ANSWER_PLAN_PROPOSAL_SCHEMA_VERSION == "answer-plan-proposal.v3"
    assert REVIEW_PROPOSAL_SCHEMA_VERSION == "answer-plan-review-proposal.v3"
    assert AnswerPlanWireProposal.model_json_schema()["properties"]["claims"]["maxItems"] == 16
    assert ReviewWireProposal.model_json_schema()["properties"]["claims"]["maxItems"] == 16
    assert len(plan_raw.encode("utf-8")) <= MAX_PROPOSAL_BYTES
    assert len(review_raw.encode("utf-8")) <= MAX_PROPOSAL_BYTES
    assert len(parse_answer_plan_wire_proposal_json(plan_raw).claims) == 16
    projected_review = project_review_wire_proposal(
        parse_review_wire_proposal_json(review_raw),
        repair_attempt=1,
        required_closed_obligation_ids=(),
    )
    serialized_review = json.loads(json.dumps(projected_review.model_dump(mode="json")))
    assert len(ReviewProposal.model_validate(serialized_review).claims) == 16


def test_seventeenth_claim_wires_fail_closed_before_projection() -> None:
    for decision, parser in (
        ("partial", parse_answer_plan_wire_proposal_json),
        ("repair", parse_review_wire_proposal_json),
    ):
        raw = json.dumps(
            {
                "decision": decision,
                "claims": [item.model_dump(mode="json") for item in _claims(17)],
            },
            separators=(",", ":"),
        )

        with pytest.raises(ProposalValidationError) as captured:
            parser(raw)

        assert captured.value.reason_code == ProviderWireFailureReason.PAYLOAD_TOO_LARGE


def test_twelve_claim_repair_survives_projection_and_state_serialization() -> None:
    claims = _claims(12)
    wire_raw = json.dumps(
        ReviewWireProposal(decision=ReviewDecision.REPAIR, claims=claims).model_dump(mode="json"),
        separators=(",", ":"),
    )
    projected = project_review_wire_proposal(
        parse_review_wire_proposal_json(wire_raw),
        repair_attempt=1,
        required_closed_obligation_ids=(),
    )
    state_payload = json.loads(json.dumps(projected.model_dump(mode="json")))
    restored = ReviewProposal.model_validate(state_payload)
    replacement_raw = AnswerPlanProposal(
        decision=AnswerDecision.PARTIAL,
        claims=restored.claims,
        reason_codes=(
            PlanReasonCode.PARTIAL_SUPPORT,
            PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
        ),
    ).model_dump_json()

    replacement = parse_answer_plan_proposal_json(replacement_raw)

    assert restored.decision == ReviewDecision.REPAIR
    assert restored.claims == claims
    assert replacement.claims == claims
    assert len(replacement.claims) == 12


def test_plan_wire_allowlist_is_checked_before_projection() -> None:
    wire = AnswerPlanWireProposal(
        decision=AnswerDecision.ANSWER,
        claims=(_claim(obligation_id=_OTHER_OBLIGATION_ID),),
    )

    with pytest.raises(ProposalValidationError, match="allowlists"):
        validate_claim_allowlists(
            wire.claims,
            allowed_obligation_ids=(_OBLIGATION_ID,),
            allowed_witness_ids=(_WITNESS_ID,),
        )


@pytest.mark.parametrize(
    (
        "decision",
        "claims",
        "repair_attempt",
        "required_closed_ids",
        "expected_decision",
        "expected_reason",
        "expected_claim_count",
        "expected_closed_ids",
    ),
    [
        (
            ReviewDecision.ACCEPT,
            (),
            0,
            (_OBLIGATION_ID,),
            ReviewDecision.ACCEPT,
            ReviewReasonCode.PLAN_SEMANTICALLY_VALID,
            0,
            (_OBLIGATION_ID,),
        ),
        (
            ReviewDecision.REPAIR,
            (_claim(),),
            0,
            (_OBLIGATION_ID,),
            ReviewDecision.REPAIR,
            ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,
            1,
            (),
        ),
        (
            ReviewDecision.REPAIR,
            (_claim(),),
            1,
            (_OBLIGATION_ID,),
            ReviewDecision.REPAIR,
            ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,
            1,
            (),
        ),
        (
            ReviewDecision.REPAIR,
            (_claim(),),
            2,
            (_OBLIGATION_ID,),
            ReviewDecision.REJECT,
            ReviewReasonCode.REPAIR_LIMIT_REACHED,
            0,
            (),
        ),
        (
            ReviewDecision.REJECT,
            (),
            0,
            (_OBLIGATION_ID,),
            ReviewDecision.REJECT,
            ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,
            0,
            (),
        ),
        (
            ReviewDecision.REJECT,
            (),
            1,
            (_OBLIGATION_ID,),
            ReviewDecision.REJECT,
            ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,
            0,
            (),
        ),
        (
            ReviewDecision.REJECT,
            (),
            2,
            (_OBLIGATION_ID,),
            ReviewDecision.REJECT,
            ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,
            0,
            (),
        ),
    ],
)
def test_review_wire_uses_a_non_escalating_projection_matrix(
    decision: ReviewDecision,
    claims: tuple[ClaimProposal, ...],
    repair_attempt: int,
    required_closed_ids: tuple[str, ...],
    expected_decision: ReviewDecision,
    expected_reason: ReviewReasonCode,
    expected_claim_count: int,
    expected_closed_ids: tuple[str, ...],
) -> None:
    wire = ReviewWireProposal(decision=decision, claims=claims)

    canonical = project_review_wire_proposal(
        wire,
        repair_attempt=repair_attempt,  # type: ignore[arg-type]
        required_closed_obligation_ids=required_closed_ids,
    )

    assert canonical.decision == expected_decision
    assert canonical.reason_codes == (expected_reason,)
    assert len(canonical.claims) == expected_claim_count
    assert canonical.closed_obligation_ids == expected_closed_ids


@pytest.mark.parametrize(
    "wire",
    [
        ReviewWireProposal(
            decision=ReviewDecision.REPAIR,
            claims=(_claim(witness_id=_OTHER_WITNESS_ID),),
        ),
        ReviewWireProposal(
            decision=ReviewDecision.REPAIR,
            claims=(_claim(obligation_id=_OTHER_OBLIGATION_ID),),
        ),
    ],
)
def test_review_checks_repair_ids_before_repair_limit_projection_discards_claims(
    wire: ReviewWireProposal,
) -> None:
    with pytest.raises(ProposalValidationError):
        validate_review_wire_allowlists(
            wire,
            allowed_obligation_ids=(_OBLIGATION_ID,),
            allowed_witness_ids=(_WITNESS_ID,),
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"reject","decision":"accept","claims":[]}',
        '{"decision":"reject","claims":[],"reason_codes":["NO_SEMANTICALLY_VALID_PLAN"]}',
        '{"decision":"unknown","claims":[]}',
        '{"decision":"accept","claims":[],"closed_obligation_ids":[]}',
        json.dumps(
            {
                "decision": "accept",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            }
        ),
        '{"decision":"repair","claims":[]}',
        json.dumps(
            {
                "decision": "repair",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
                "closed_obligation_ids": [_OBLIGATION_ID],
            }
        ),
        json.dumps(
            {
                "decision": "reject",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "decision": "repair",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": "wit_" + f"{index:040x}",
                        "rendering_rule": "exact_witness",
                    }
                    for index in range(MAX_PROPOSAL_CLAIMS + 1)
                ],
            }
        ),
    ],
)
def test_review_wire_parser_remains_fail_closed(raw: str) -> None:
    with pytest.raises(ProposalValidationError):
        parse_review_wire_proposal_json(raw)


def test_review_server_invariant_is_rejected_outside_wire_parsing() -> None:
    wire = parse_review_wire_proposal_json('{"decision":"reject","claims":[]}')

    with pytest.raises(ValueError, match="repair_attempt") as captured:
        project_review_wire_proposal(
            wire,
            repair_attempt=3,  # type: ignore[arg-type]
            required_closed_obligation_ids=(),
        )
    assert type(captured.value) is ValueError


@pytest.mark.parametrize(
    (
        "required_ids",
        "obligations",
        "supported_ids",
        "unresolved_ids",
        "universe_status",
        "expected_ids",
    ),
    [
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.SINGLE, None),),
            (_OBLIGATION_ID,),
            (),
            WitnessUniverseStatus.COMPLETE,
            (_OBLIGATION_ID,),
        ),
        (
            (),
            ((_OTHER_OBLIGATION_ID, FacetClosure.SINGLE, None),),
            (_OTHER_OBLIGATION_ID,),
            (),
            WitnessUniverseStatus.COMPLETE,
            (),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.SINGLE, None),),
            (_OBLIGATION_ID,),
            (_OBLIGATION_ID,),
            WitnessUniverseStatus.COMPLETE,
            (),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.SINGLE, None),),
            (_OBLIGATION_ID,),
            (),
            WitnessUniverseStatus.TRUNCATED,
            (),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.OPEN_ENUMERATION, None),),
            (_OBLIGATION_ID,),
            (),
            WitnessUniverseStatus.COMPLETE,
            (),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.SINGLE, None),),
            (_OBLIGATION_ID, _OBLIGATION_ID),
            (),
            WitnessUniverseStatus.COMPLETE,
            (),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.CLOSED_COUNT, 2),),
            (_OBLIGATION_ID, _OBLIGATION_ID),
            (),
            WitnessUniverseStatus.COMPLETE,
            (_OBLIGATION_ID,),
        ),
        (
            (_OBLIGATION_ID,),
            ((_OBLIGATION_ID, FacetClosure.CLOSED_COUNT, 2),),
            (_OBLIGATION_ID,),
            (),
            WitnessUniverseStatus.COMPLETE,
            (),
        ),
    ],
    ids=(
        "eligible-exact-closure",
        "conditional-cannot-close",
        "unresolved-cannot-close",
        "noncomplete-cannot-close",
        "open-enum-without-source-receipt-cannot-close",
        "single-requires-exactly-one-claim",
        "closed-count-exactly-matches",
        "closed-count-underfilled",
    ),
)
def test_review_accept_closure_is_derived_and_injected_by_the_server(
    required_ids: tuple[str, ...],
    obligations: tuple[tuple[str, FacetClosure, int | None], ...],
    supported_ids: tuple[str, ...],
    unresolved_ids: tuple[str, ...],
    universe_status: WitnessUniverseStatus,
    expected_ids: tuple[str, ...],
) -> None:
    question = SimpleNamespace(
        required_obligation_ids=required_ids,
        obligations=tuple(
            SimpleNamespace(
                obligation_id=obligation_id,
                closure=closure,
                expected_count=expected_count,
            )
            for obligation_id, closure, expected_count in obligations
        ),
    )
    plan = SimpleNamespace(
        substantive_claims=tuple(
            SimpleNamespace(obligation_id=obligation_id) for obligation_id in supported_ids
        ),
        unresolved_obligation_ids=unresolved_ids,
    )
    catalog = SimpleNamespace(universe_status=universe_status)
    derived = provider._required_closed_obligation_ids(
        question_contract=question,
        answer_plan=plan,
        catalog=catalog,
    )
    assert derived == expected_ids

    canonical = project_review_wire_proposal(
        ReviewWireProposal(decision=ReviewDecision.ACCEPT, claims=()),
        repair_attempt=0,
        required_closed_obligation_ids=derived,
    )
    assert canonical.closed_obligation_ids == expected_ids


def test_review_accept_closes_open_enumeration_only_for_exact_source_witness_set() -> None:
    question = SimpleNamespace(
        required_obligation_ids=(_OBLIGATION_ID,),
        obligations=(
            SimpleNamespace(
                obligation_id=_OBLIGATION_ID,
                closure=FacetClosure.OPEN_ENUMERATION,
                expected_count=None,
            ),
        ),
    )
    plan = SimpleNamespace(
        substantive_claims=(
            SimpleNamespace(
                obligation_id=_OBLIGATION_ID,
                witness_ids=(_WITNESS_ID,),
            ),
        ),
        unresolved_obligation_ids=(),
    )
    catalog = SimpleNamespace(universe_status=WitnessUniverseStatus.COMPLETE)

    assert provider._required_closed_obligation_ids(
        question_contract=question,
        answer_plan=plan,
        catalog=catalog,
        source_closed_witness_ids={_OBLIGATION_ID: (_WITNESS_ID,)},
    ) == (_OBLIGATION_ID,)
    assert (
        provider._required_closed_obligation_ids(
            question_contract=question,
            answer_plan=plan,
            catalog=catalog,
            source_closed_witness_ids={_OBLIGATION_ID: (_OTHER_WITNESS_ID,)},
        )
        == ()
    )


def test_configured_adapter_binds_wire_schema_but_receipts_canonical_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def invoke(_state, **kwargs):
        observed.update(kwargs)
        raw = json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            },
            separators=(",", ":"),
        )
        wire = kwargs["response_parser"](raw)
        return kwargs["response_projector"](wire)

    monkeypatch.setattr(provider, "invoke_structured_answer_model", invoke)
    adapter = provider.ConfiguredAnsweringModelAdapter()

    proposal = adapter.select_plan(
        _adapter_state(),
        question_contract=_question_contract(),
        witness_catalog=({"witness_id": _WITNESS_ID, "content": "合成证据"},),
    )

    assert observed["wire_response_model"] is AnswerPlanWireProposal
    assert observed["response_model"].__name__ == "AnswerPlanProposal"
    assert proposal.reason_codes == (PlanReasonCode.ANSWER_SUPPORTED,)


def test_configured_reviewer_never_accepts_contradictory_replacement_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def invoke(_state, **kwargs):
        observed.update(kwargs)
        raw = json.dumps(
            {
                "decision": "accept",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            },
            separators=(",", ":"),
        )
        wire = kwargs["response_parser"](raw)
        return kwargs["response_projector"](wire)

    monkeypatch.setattr(provider, "invoke_structured_answer_model", invoke)
    monkeypatch.setattr(
        provider,
        "_review_accept_contract",
        lambda *_args, **_kwargs: (
            _adapter_state()["witness_catalog"]["receipt"],
            (_OBLIGATION_ID,),
        ),
    )
    adapter = provider.ConfiguredAnsweringModelAdapter()
    plan = project_answer_plan_wire_proposal(
        AnswerPlanWireProposal(
            decision=AnswerDecision.ANSWER,
            claims=(_claim(),),
        )
    )

    with pytest.raises(ProposalValidationError):
        adapter.review_plan(
            _adapter_state(),
            question_contract=_question_contract(),
            witness_catalog=({"witness_id": _WITNESS_ID, "content": "合成证据"},),
            plan_proposal=plan,
            repair_attempt=0,
        )

    assert observed["wire_response_model"] is ReviewWireProposal
    assert observed["response_model"].__name__ == "ReviewProposal"


def test_configured_reviewer_projects_terminal_repair_to_bounded_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke(_state, **kwargs):
        raw = json.dumps(
            {
                "decision": "repair",
                "claims": [
                    {
                        "obligation_id": _OBLIGATION_ID,
                        "witness_id": _WITNESS_ID,
                        "rendering_rule": "exact_witness",
                    }
                ],
            },
            separators=(",", ":"),
        )
        wire = kwargs["response_parser"](raw)
        return kwargs["response_projector"](wire)

    monkeypatch.setattr(provider, "invoke_structured_answer_model", invoke)
    monkeypatch.setattr(
        provider,
        "_review_accept_contract",
        lambda *_args, **_kwargs: (
            _adapter_state()["witness_catalog"]["receipt"],
            (_OBLIGATION_ID,),
        ),
    )
    adapter = provider.ConfiguredAnsweringModelAdapter()
    plan = project_answer_plan_wire_proposal(
        AnswerPlanWireProposal(
            decision=AnswerDecision.ANSWER,
            claims=(_claim(),),
        )
    )

    review = adapter.review_plan(
        _adapter_state(),
        question_contract=_question_contract(),
        witness_catalog=({"witness_id": _WITNESS_ID, "content": "合成证据"},),
        plan_proposal=plan,
        repair_attempt=2,
    )

    assert review.decision == ReviewDecision.REJECT
    assert review.reason_codes == (ReviewReasonCode.REPAIR_LIMIT_REACHED,)
    assert review.claims == ()
