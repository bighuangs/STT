from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from typing import Any

import pytest

import app.application.query_answering.builders as answer_builders
import app.application.query_answering.rendering as answer_rendering
import app.domain.evaluation_proof as evaluation_proof_module
import app.domain.safe_representation as safe_representation
import app.infrastructure.answering.provider as answering_provider
from app.application.query_answering.pipeline import (
    AnswerProofNotCommittableError,
    commit_proven_answer,
    decide_answer_coverage,
    freeze_question_contract,
    freeze_witness_catalog,
    plan_answer,
    render_proven_answer,
    validate_committable_answer,
    verify_answer_plan,
)
from app.application.query_answering.atomizer import atomize_evidence_text
from app.application.query_answering.builders import resolve_witness_prompt_views
from app.application.query_answering.builders import _obligation_shape
from app.application.query_security_pipeline import secure_evidence_documents
from app.agent_system.agents.query.workflow import (
    node_answer_coverage_after_router,
    node_answer_render_after_router,
)
from app.application.query_answering.proposals import (
    MAX_WITNESS_CATALOG_ITEMS,
    AnswerDecision,
    AnswerPlanProposal,
    ClaimProposal,
    PlanReasonCode,
    ProposalValidationError,
    RenderingRule,
    ReviewDecision,
    ReviewProposal,
    ReviewReasonCode,
    parse_answer_plan_proposal_json,
    parse_review_proposal_json,
)
from app.application.query_understanding_service import (
    DeterministicQueryPlanner,
    QueryPlanningInput,
)
from app.domain.answering.item_boundary import (
    markdown_table_data_row_body_span,
    top_level_list_item_body_span,
)
from app.domain.answering.plan import AnswerPlan
from app.domain.answering.proof import AnswerProof, ClaimProof
from app.domain.answering.question import QuestionContract
from app.domain.answering.witness import (
    PACKED_TRUNCATION_MARKER,
    EvidenceSpanRef,
    EvidenceWitness,
    SpanEquivalence,
    WitnessCatalogReceipt,
    WitnessOmissionCode,
)
from app.domain.answering._base import stable_content_hash
from app.domain.evidence import (
    AnswerSpan,
    CitationFact,
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
    citation_fact_bundle_hash,
    sha256_digest,
)
from app.domain.evaluation_proof import (
    OBSERVATION_SCHEMA_V3,
    answer_proof_evaluation,
    build_proof_evaluation_authority,
    project_proof_observation,
)
from app.domain.query_planning import AuthorizedScopeRef, PlanningPolicy
from app.domain.query_planning import FacetClosure
from app.infrastructure.answering.render_adapter import configured_answer_render_adapter
from app.process.query.agent.state import create_query_default_state
from app.rag.query.context_pack_service import (
    ContextPack,
    ContextBudget,
    ContextPackPolicy,
    build_context_pack,
    replay_context_pack_content_hash,
)


class _CharacterCounter:
    @property
    def tokenizer_id(self) -> str:
        return "test-character-counter"

    @property
    def tokenizer_version(self) -> str:
        return "v1"

    def count(self, text: str) -> int:
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: max(0, max_tokens)]


def _query_plan(original_query: str = "设备启动前需要检查什么？"):
    return DeterministicQueryPlanner().plan(
        QueryPlanningInput(
            original_query=original_query,
            authorized_scope_ref=AuthorizedScopeRef(
                scope_id="scope-proof-v2-test",
                tenant_id="tenant-test",
                kb_ids=("kb-test",),
            ),
            profile_version=1,
            policy=PlanningPolicy(
                allow_web=False,
                total_timeout_ms=1_000,
                token_budget=2_000,
                cost_budget=1.0,
            ),
            deadline_at=datetime.now(UTC) + timedelta(seconds=30),
        )
    )


def _evidence(content: str, *, identity_suffix: str = "test") -> Evidence:
    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id="tenant-test",
            kb_id="kb-test",
            document_id=f"doc-{identity_suffix}",
            document_version_id=f"dv-{identity_suffix}-v1",
            index_version_id="idx-test-v1",
            chunk_id=f"chunk-{identity_suffix}",
            source_block_ids=[f"block-{identity_suffix}"],
        ),
        content=content,
        provenance_spans=[
            EvidenceLocationSpan(
                block_id=f"block-{identity_suffix}",
                availability="exact",
                coordinate_space="normalized_page",
                page_no=1,
                bbox=(0.1, 0.1, 0.9, 0.2),
            )
        ],
        source_label="synthetic-manual.pdf",
    )


def _state(
    content: str | None,
    *,
    original_query: str = "设备启动前需要检查什么？",
    through_security: bool = True,
    tampered_safe_content: str | None = None,
    additional_contents: tuple[str, ...] = (),
    max_block_characters: int | None = None,
) -> dict[str, Any]:
    plan = _query_plan(original_query)
    documents: list[dict[str, Any]] = []
    contents = (() if content is None else (content,)) + additional_contents
    for index, item_content in enumerate(contents):
        identity_suffix = "test" if index == 0 else f"test-{index + 1}"
        evidence = _evidence(item_content, identity_suffix=identity_suffix)
        documents.append(
            {
                "evidence_id": evidence.evidence_id,
                "evidence": evidence.model_dump(mode="json", exclude_none=True),
                "title": evidence.source_label,
                "text": evidence.content,
                "content": evidence.content,
                "type": "milvus",
                "rerank_score": 1.0,
                "sub_query_ids": ["sq_primary"],
            }
        )
    security_state: dict[str, Any] = {"tenant_id": "tenant-test"}
    if through_security:
        documents = secure_evidence_documents(security_state, documents)
    if tampered_safe_content is not None:
        if not documents:
            raise ValueError("safe content tamper requires an Evidence document")
        documents[0] = {
            **documents[0],
            "security_safe_content": tampered_safe_content,
        }
    result = build_context_pack(
        documents,
        query_run_id="qry_00000000000000000000000000000000",
        budget=ContextBudget.from_window(
            model_context_limit=20_000,
            reserved_system=1_000,
            reserved_output=1_000,
            safety_margin=100,
            evidence_budget=15_000,
            evidence_character_budget=100_000,
        ),
        counter=_CharacterCounter(),
        policy=(
            ContextPackPolicy(max_block_characters=max_block_characters)
            if max_block_characters is not None
            else None
        ),
        required_sub_query_ids=("sq_primary",),
        facet_ledger=plan.facet_ledger,
    )
    packed_by_label = {block.label: block.text for block in result.pack.packed_blocks}
    aliases = [
        {**item, "packed_text": packed_by_label[item["label"]]} for item in result.citation_aliases
    ]
    return create_query_default_state(
        session_id="session-proof-v2",
        conversation_id="conversation-proof-v2",
        turn_id="turn-proof-v2",
        tenant_id="tenant-test",
        user_id="user-test",
        kb_id="kb-test",
        original_query=plan.understanding.original_query,
        rewritten_query=plan.understanding.standalone_query,
        query_plan=plan.model_dump(mode="json"),
        query_facet_ledger=plan.facet_ledger.model_dump(mode="json"),
        query_run_id="qry_00000000000000000000000000000000",
        answer_id="ans_00000000000000000000000000000000",
        authorized_scope={"scope_hash": "scope_" + "0" * 64},
        context_pack=result.pack.model_dump(mode="json"),
        context_aliases=aliases,
        evidences=[item["evidence"] for item in aliases],
        evaluation_mode=True,
        llm_model_id="test-model",
        is_stream=False,
        content_security=security_state.get("content_security") or {},
    )


class _AcceptingModel:
    plan_prompt_version = "test-plan-prompt.v1"
    review_prompt_version = "test-review-prompt.v1"

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse

    def select_plan(self, _state, *, question_contract, witness_catalog):
        if self.refuse:
            return AnswerPlanProposal(
                decision=AnswerDecision.REFUSE,
                claims=(),
                reason_codes=(PlanReasonCode.NO_SUPPORTING_WITNESS,),
            )
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(witness_catalog[0]["witness_id"]),
                    rendering_rule=RenderingRule.EXACT_WITNESS,
                ),
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )

    def review_plan(
        self,
        _state,
        *,
        question_contract,
        witness_catalog,
        plan_proposal,
        repair_attempt,
    ):
        assert question_contract is not None
        assert repair_attempt == 0
        if self.refuse:
            assert plan_proposal.decision == AnswerDecision.REFUSE
        else:
            assert plan_proposal.claims
        review = ReviewProposal(
            decision=ReviewDecision.ACCEPT,
            claims=(),
            reason_codes=(ReviewReasonCode.PLAN_SEMANTICALLY_VALID,),
            closed_obligation_ids=(
                question_contract.required_obligation_ids
                if plan_proposal.decision == AnswerDecision.ANSWER
                else ()
            ),
        )
        proposal_payload = review.model_dump(mode="json")
        _state.setdefault("answer_prompt_receipts", []).append(
            {
                "logical_call_id": "answer-review-test-call",
                "scene": "answer_plan_semantic_review",
                "prompt_version": self.review_prompt_version,
                "prompt_sha256": stable_content_hash(
                    {
                        "plan": plan_proposal.model_dump(mode="json"),
                        "repair_attempt": repair_attempt,
                    }
                ),
                "response_sha256": stable_content_hash(proposal_payload),
                "model_id": "test-model",
                "status": "succeeded",
                "parsed_proposal": proposal_payload,
            }
        )
        return review


class _IncompleteClosureAcceptingModel(_AcceptingModel):
    """模拟端口漂移：ACCEPT 收据有效，但未返回权威闭包全集。"""

    def review_plan(self, state, **kwargs):
        accepted = super().review_plan(state, **kwargs)
        incomplete = accepted.model_copy(update={"closed_obligation_ids": ()})
        payload = incomplete.model_dump(mode="json")
        state["answer_prompt_receipts"][-1]["parsed_proposal"] = payload
        state["answer_prompt_receipts"][-1]["response_sha256"] = stable_content_hash(payload)
        return incomplete


class _PartialModel(_AcceptingModel):
    def select_plan(self, _state, *, question_contract, witness_catalog):
        return AnswerPlanProposal(
            decision=AnswerDecision.PARTIAL,
            claims=(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(witness_catalog[0]["witness_id"]),
                    rendering_rule=RenderingRule.EXACT_WITNESS,
                ),
            ),
            reason_codes=(PlanReasonCode.PARTIAL_SUPPORT,),
        )


class _RejectingModel(_AcceptingModel):
    def review_plan(
        self,
        _state,
        *,
        question_contract,
        witness_catalog,
        plan_proposal,
        repair_attempt,
    ):
        assert question_contract is not None
        assert witness_catalog
        assert plan_proposal.decision == AnswerDecision.REFUSE
        assert repair_attempt == 0
        review = ReviewProposal(
            decision=ReviewDecision.REJECT,
            claims=(),
            reason_codes=(ReviewReasonCode.NO_SEMANTICALLY_VALID_PLAN,),
        )
        proposal_payload = review.model_dump(mode="json")
        _state.setdefault("answer_prompt_receipts", []).append(
            {
                "logical_call_id": "answer-review-reject-test-call",
                "scene": "answer_plan_semantic_review",
                "prompt_version": self.review_prompt_version,
                "prompt_sha256": stable_content_hash(
                    {
                        "plan": plan_proposal.model_dump(mode="json"),
                        "repair_attempt": repair_attempt,
                    }
                ),
                "response_sha256": stable_content_hash(proposal_payload),
                "model_id": "test-model",
                "status": "succeeded",
                "parsed_proposal": proposal_payload,
            }
        )
        return review


class _MultiSingleModel(_AcceptingModel):
    def select_plan(self, _state, *, question_contract, witness_catalog):
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.EXACT_WITNESS,
                )
                for item in witness_catalog[:2]
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _ReverseStepsModel(_AcceptingModel):
    def select_plan(self, _state, *, question_contract, witness_catalog):
        assert len(witness_catalog) >= 3
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for item in reversed(witness_catalog[:3])
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _PartialStepsModel(_AcceptingModel):
    def select_plan(self, _state, *, question_contract, witness_catalog):
        assert len(witness_catalog) >= 3
        return AnswerPlanProposal(
            decision=AnswerDecision.PARTIAL,
            claims=tuple(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for item in witness_catalog[:3]
            ),
            reason_codes=(
                PlanReasonCode.PARTIAL_SUPPORT,
                PlanReasonCode.REQUIRED_OBLIGATION_UNSUPPORTED,
            ),
        )


class _OpenEnumerationModel(_AcceptingModel):
    def select_plan(self, _state, *, question_contract, witness_catalog):
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for item in witness_catalog[:4]
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _SourceClosedOpenEnumerationModel(_AcceptingModel):
    """只选择已由运行时验证为来源闭合的开放枚举表格记录。"""

    def select_plan(self, _state, *, question_contract, witness_catalog):
        obligation_id = question_contract.required_obligation_ids[0]
        eligible = [
            item
            for item in witness_catalog
            if obligation_id in item.get("source_closed_open_obligation_ids", ())
        ]
        assert len(eligible) == 1
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=(
                ClaimProposal(
                    obligation_id=obligation_id,
                    witness_id=str(eligible[0]["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                ),
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _SourceClosedProcedureModel(_AcceptingModel):
    """选择 typed procedure/list receipt 标记的完整步骤集合。"""

    def __init__(self, *, omit_last: bool = False) -> None:
        super().__init__()
        self.omit_last = omit_last

    def select_plan(self, _state, *, question_contract, witness_catalog):
        obligation_id = question_contract.required_obligation_ids[0]
        eligible = [
            item
            for item in witness_catalog
            if obligation_id in item.get("source_closed_open_obligation_ids", ())
        ]
        assert len(eligible) >= 2
        selected = eligible[:-1] if self.omit_last else eligible
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=obligation_id,
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for item in selected
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _SourceClosedMembersModel(_AcceptingModel):
    """选择某个开放枚举回执声明的全部成员，不假设成员数量。"""

    def select_plan(self, _state, *, question_contract, witness_catalog):
        obligation_id = question_contract.required_obligation_ids[0]
        eligible = [
            item
            for item in witness_catalog
            if obligation_id in item.get("source_closed_open_obligation_ids", ())
        ]
        assert eligible
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=obligation_id,
                    witness_id=str(item["witness_id"]),
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for item in eligible
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _ProjectedTableCellModel(_AcceptingModel):
    """让两个独立表列义务引用同一条已选择 data row。"""

    def select_plan(self, _state, *, question_contract, witness_catalog):
        prompt_ids = {str(item["witness_id"]) for item in witness_catalog}
        table_row_ids = [
            str(item["witness_id"])
            for item in _state["witness_catalog"]["witnesses"]
            if item.get("atom_kind") == "table_data_row" and str(item["witness_id"]) in prompt_ids
        ]
        assert table_row_ids
        witness_id = table_row_ids[0]
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=tuple(
                ClaimProposal(
                    obligation_id=obligation_id,
                    witness_id=witness_id,
                    rendering_rule=RenderingRule.DETERMINISTIC_TEMPLATE,
                )
                for obligation_id in question_contract.required_obligation_ids
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )


class _RepairThenAcceptModel(_AcceptingModel):
    """记录唯一 selector 调用，并验证 reviewer 回边只接收当前计划。"""

    def __init__(self) -> None:
        super().__init__()
        self.select_calls = 0
        self.review_inputs: list[tuple[str, ...]] = []
        self.initial_witness_id = ""
        self.replacement_witness_id = ""

    def select_plan(self, _state, *, question_contract, witness_catalog):
        self.select_calls += 1
        assert self.select_calls == 1
        assert len(witness_catalog) >= 2
        self.initial_witness_id = str(witness_catalog[0]["witness_id"])
        self.replacement_witness_id = str(witness_catalog[1]["witness_id"])
        return AnswerPlanProposal(
            decision=AnswerDecision.ANSWER,
            claims=(
                ClaimProposal(
                    obligation_id=question_contract.required_obligation_ids[0],
                    witness_id=self.initial_witness_id,
                    rendering_rule=RenderingRule.EXACT_WITNESS,
                ),
            ),
            reason_codes=(PlanReasonCode.ANSWER_SUPPORTED,),
        )

    def review_plan(
        self,
        _state,
        *,
        question_contract,
        witness_catalog,
        plan_proposal,
        repair_attempt,
    ):
        assert witness_catalog
        reviewed_witness_ids = tuple(item.witness_id for item in plan_proposal.claims)
        self.review_inputs.append(reviewed_witness_ids)
        if repair_attempt == 0:
            assert reviewed_witness_ids == (self.initial_witness_id,)
            review = ReviewProposal(
                decision=ReviewDecision.REPAIR,
                claims=(
                    ClaimProposal(
                        obligation_id=question_contract.required_obligation_ids[0],
                        witness_id=self.replacement_witness_id,
                        rendering_rule=RenderingRule.EXACT_WITNESS,
                    ),
                ),
                reason_codes=(ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,),
            )
        else:
            assert repair_attempt == 1
            assert reviewed_witness_ids == (self.replacement_witness_id,)
            review = ReviewProposal(
                decision=ReviewDecision.ACCEPT,
                claims=(),
                reason_codes=(ReviewReasonCode.PLAN_SEMANTICALLY_VALID,),
                closed_obligation_ids=question_contract.required_obligation_ids,
            )
        proposal_payload = review.model_dump(mode="json")
        call_index = len(self.review_inputs)
        _state.setdefault("answer_prompt_receipts", []).append(
            {
                "logical_call_id": "qmc_" + f"{call_index:032x}",
                "scene": "answer_plan_semantic_review",
                "prompt_version": self.review_prompt_version,
                "prompt_sha256": stable_content_hash(
                    {
                        "plan": plan_proposal.model_dump(mode="json"),
                        "repair_attempt": repair_attempt,
                    }
                ),
                "response_sha256": stable_content_hash(proposal_payload),
                "model_id": "test-model",
                "status": "succeeded",
                "parsed_proposal": proposal_payload,
            }
        )
        return review


class _RepairTwiceThenAcceptModel(_RepairThenAcceptModel):
    """连续提出两份 replacement，并仅在第二份经过终审后接受。"""

    def __init__(self) -> None:
        super().__init__()
        self.final_witness_id = ""

    def select_plan(self, state, *, question_contract, witness_catalog):
        proposal = super().select_plan(
            state,
            question_contract=question_contract,
            witness_catalog=witness_catalog,
        )
        assert len(witness_catalog) >= 3
        self.final_witness_id = str(witness_catalog[2]["witness_id"])
        return proposal

    def review_plan(
        self,
        state,
        *,
        question_contract,
        witness_catalog,
        plan_proposal,
        repair_attempt,
    ):
        assert witness_catalog
        reviewed_witness_ids = tuple(item.witness_id for item in plan_proposal.claims)
        self.review_inputs.append(reviewed_witness_ids)
        expected_witness_id = (
            self.initial_witness_id
            if repair_attempt == 0
            else self.replacement_witness_id
            if repair_attempt == 1
            else self.final_witness_id
        )
        assert reviewed_witness_ids == (expected_witness_id,)
        if repair_attempt < 2:
            replacement_witness_id = (
                self.replacement_witness_id if repair_attempt == 0 else self.final_witness_id
            )
            review = ReviewProposal(
                decision=ReviewDecision.REPAIR,
                claims=(
                    ClaimProposal(
                        obligation_id=question_contract.required_obligation_ids[0],
                        witness_id=replacement_witness_id,
                        rendering_rule=RenderingRule.EXACT_WITNESS,
                    ),
                ),
                reason_codes=(ReviewReasonCode.REPLACEMENT_SELECTION_PROPOSED,),
            )
        else:
            review = ReviewProposal(
                decision=ReviewDecision.ACCEPT,
                claims=(),
                reason_codes=(ReviewReasonCode.PLAN_SEMANTICALLY_VALID,),
                closed_obligation_ids=question_contract.required_obligation_ids,
            )
        proposal_payload = review.model_dump(mode="json")
        call_index = len(self.review_inputs)
        state.setdefault("answer_prompt_receipts", []).append(
            {
                "logical_call_id": "qmc_" + f"{call_index:032x}",
                "scene": "answer_plan_semantic_review",
                "prompt_version": self.review_prompt_version,
                "prompt_sha256": stable_content_hash(
                    {
                        "plan": plan_proposal.model_dump(mode="json"),
                        "repair_attempt": repair_attempt,
                    }
                ),
                "response_sha256": stable_content_hash(proposal_payload),
                "model_id": "test-model",
                "status": "succeeded",
                "parsed_proposal": proposal_payload,
            }
        )
        return review


class _RepairTwiceThenRejectModel(_RepairTwiceThenAcceptModel):
    """模拟 provider 在终审继续 repair 后的 canonical bounded reject。"""

    def review_plan(
        self,
        state,
        *,
        question_contract,
        witness_catalog,
        plan_proposal,
        repair_attempt,
    ):
        if repair_attempt < 2:
            return super().review_plan(
                state,
                question_contract=question_contract,
                witness_catalog=witness_catalog,
                plan_proposal=plan_proposal,
                repair_attempt=repair_attempt,
            )
        reviewed_witness_ids = tuple(item.witness_id for item in plan_proposal.claims)
        assert reviewed_witness_ids == (self.final_witness_id,)
        self.review_inputs.append(reviewed_witness_ids)
        review = ReviewProposal(
            decision=ReviewDecision.REJECT,
            claims=(),
            reason_codes=(ReviewReasonCode.REPAIR_LIMIT_REACHED,),
        )
        proposal_payload = review.model_dump(mode="json")
        state.setdefault("answer_prompt_receipts", []).append(
            {
                "logical_call_id": "qmc_" + f"{len(self.review_inputs):032x}",
                "scene": "answer_plan_semantic_review",
                "prompt_version": self.review_prompt_version,
                "prompt_sha256": stable_content_hash(
                    {
                        "plan": plan_proposal.model_dump(mode="json"),
                        "repair_attempt": repair_attempt,
                    }
                ),
                "response_sha256": stable_content_hash(proposal_payload),
                "model_id": "test-model",
                "status": "succeeded",
                "parsed_proposal": proposal_payload,
            }
        )
        return review


def _run_to_render(state: dict[str, Any], model: _AcceptingModel) -> dict[str, Any]:
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    return render_proven_answer(state, render_port=configured_answer_render_adapter)


def test_supported_answer_builds_strict_proof_and_exact_citation() -> None:
    content = "启动设备前必须检查保护接地。"
    state = _run_to_render(_state(content), _AcceptingModel())

    assert content in state["answer"]
    assert "[E1]" in state["answer"]
    assert state["answer_proof"]["overall_status"] == "accepted"
    assert state["answer_proof"]["strict_eligible"] is True
    assert state["citation_validation"]["status"] == "valid"
    assert state["citations"][0]["source_quote"] == content
    assert state["answer_envelope"]["decision"] == "ANSWER"


def test_pipeline_replaces_non_authoritative_plan_reason_codes() -> None:
    state = _state("启动设备前必须检查保护接地。")
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=_PartialModel())

    assert state["answer_plan_raw_proposal"]["reason_codes"] == ["PARTIAL_SUPPORT"]
    assert state["answer_plan_proposal"]["reason_codes"] == [
        "PARTIAL_SUPPORT",
        "REQUIRED_OBLIGATION_UNSUPPORTED",
    ]


@pytest.mark.parametrize(
    ("canonical", "safe_text", "equivalence", "support_method"),
    (
        (
            "状态为ＮＦＫＣ：正常。",
            "状态为NFKC:正常。",
            SpanEquivalence.NFKC_EQUIVALENT,
            "ANSWER_PROOF_NFKC_WITNESS",
        ),
        (
            "<b>状态正常</b>。",
            "&lt;b&gt;状态正常&lt;/b&gt;。",
            SpanEquivalence.SECURITY_SAFE_DERIVED,
            "ANSWER_PROOF_SECURITY_SAFE_WITNESS",
        ),
        (
            "A &amp; B 均正常。",
            "A & B 均正常。",
            SpanEquivalence.SECURITY_SAFE_DERIVED,
            "ANSWER_PROOF_SECURITY_SAFE_WITNESS",
        ),
        (
            "A &amp B 均正常。",
            "A & B 均正常。",
            SpanEquivalence.SECURITY_SAFE_DERIVED,
            "ANSWER_PROOF_SECURITY_SAFE_WITNESS",
        ),
        (
            "Cafe\u0301 状态正常。",
            "Café 状态正常。",
            SpanEquivalence.NFKC_EQUIVALENT,
            "ANSWER_PROOF_NFKC_WITNESS",
        ),
        (
            "\u1100\u1161 상태 정상。",
            "가 상태 정상。",
            SpanEquivalence.NFKC_EQUIVALENT,
            "ANSWER_PROOF_NFKC_WITNESS",
        ),
    ),
)
def test_secured_evidence_maps_safe_witness_back_to_canonical_citation(
    canonical: str,
    safe_text: str,
    equivalence: SpanEquivalence,
    support_method: str,
) -> None:
    state = _run_to_render(
        _state(canonical, through_security=True),
        _AcceptingModel(),
    )

    witness = EvidenceWitness.from_dict(state["witness_catalog"]["witnesses"][0])
    derivation = state["context_aliases"][0]["document"]["safe_representation_derivation"]
    assert state["witness_catalog"]["receipt"]["universe_status"] == "complete"
    assert derivation["schema_version"] == "safe-representation-derivation.v2"
    assert derivation["transform_version"] == "rag-safe-representation.v2"
    assert witness.span_ref.equivalence == equivalence
    assert safe_text in state["answer"]
    assert state["citations"][0]["claim_text"] == safe_text
    assert state["citations"][0]["source_quote"] == canonical
    assert state["citations"][0]["quote_hash"] == sha256_digest(canonical)
    assert state["citations"][0]["support_method"] == support_method
    validate_committable_answer(state)


def test_image_control_span_is_omitted_without_poisoning_witness_universe() -> None:
    canonical = "![示意图](https://example.test/device.png)\n启动设备前必须检查保护接地。"
    state = _run_to_render(
        _state(canonical, through_security=True),
        _AcceptingModel(),
    )

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    derivation_spans = state["context_aliases"][0]["document"]["safe_representation_derivation"][
        "spans"
    ]
    assert receipt.universe_status.value == "complete"
    assert receipt.omissions == ()
    assert receipt.witness_count == 1
    assert any(item["relation"] == "control_markdown_image" for item in derivation_spans)
    assert "[image omitted]" not in state["answer"]
    assert state["citations"][0]["source_quote"] == "启动设备前必须检查保护接地。"
    validate_committable_answer(state)


def test_mixed_image_control_and_fact_remains_citable_via_typed_derivation() -> None:
    canonical = "![示意图](https://example.test/device.png) 启动设备前必须检查保护接地。"
    safe_text = "[image omitted] 启动设备前必须检查保护接地。"
    state = _run_to_render(
        _state(canonical, through_security=True),
        _AcceptingModel(),
    )

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    witness = EvidenceWitness.from_dict(state["witness_catalog"]["witnesses"][0])
    assert receipt.universe_status.value == "complete"
    assert receipt.omissions == ()
    assert receipt.witness_count == 1
    assert witness.span_ref.equivalence == SpanEquivalence.SECURITY_SAFE_DERIVED
    assert state["citations"][0]["claim_text"] == safe_text
    assert state["citations"][0]["source_quote"] == canonical
    validate_committable_answer(state)


def test_literal_image_control_text_remains_an_exact_citable_witness() -> None:
    canonical = "[image omitted]。"
    state = _run_to_render(
        _state(canonical, through_security=True),
        _AcceptingModel(),
    )

    witness = EvidenceWitness.from_dict(state["witness_catalog"]["witnesses"][0])
    derivation_spans = state["context_aliases"][0]["document"]["safe_representation_derivation"][
        "spans"
    ]
    assert witness.span_ref.equivalence == SpanEquivalence.EXACT_UTF8
    assert all(item["relation"] != "control_markdown_image" for item in derivation_spans)
    assert state["citations"][0]["source_quote"] == canonical
    validate_committable_answer(state)


@pytest.mark.parametrize(
    ("canonical", "safe_claim", "canonical_quote"),
    (
        ("A&amp正常。后文。", "A&正常。", "A&amp正常。"),
        ("A&nbsp中文。后文。", "A 中文。", "A&nbsp中文。"),
        ("A&copy2026。后文。", "A©2026。", "A&copy2026。"),
    ),
)
def test_legacy_html_entity_adjacent_suffix_maps_without_swallowing_following_text(
    canonical: str,
    safe_claim: str,
    canonical_quote: str,
) -> None:
    state = _run_to_render(_state(canonical), _AcceptingModel())

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    assert receipt.universe_status.value == "complete"
    assert receipt.witness_count == 2
    assert state["citations"][0]["claim_text"] == safe_claim
    assert state["citations"][0]["source_quote"] == canonical_quote
    validate_committable_answer(state)


@pytest.mark.parametrize(
    ("canonical", "expected_equivalence"),
    (
        ("A" * 1_799 + "ﬁ。", SpanEquivalence.NFKC_EQUIVALENT),
        ("A" * 1_799 + "<b>尾部</b>。", SpanEquivalence.SECURITY_SAFE_DERIVED),
        (
            "A" * 1_799 + "![示意](https://example.test/image.png) 尾部事实。",
            SpanEquivalence.SECURITY_SAFE_DERIVED,
        ),
    ),
)
def test_receipt_aware_atomization_snaps_around_non_linear_transform(
    canonical: str,
    expected_equivalence: SpanEquivalence,
) -> None:
    state = _state(canonical)
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    witnesses = tuple(
        EvidenceWitness.from_dict(item) for item in state["witness_catalog"]["witnesses"]
    )
    assert receipt.universe_status.value == "complete"
    assert receipt.omissions == ()
    assert receipt.witness_count >= 2
    assert any(item.span_ref.equivalence == expected_equivalence for item in witnesses)
    assert all(item.span_ref.packed_end - item.span_ref.packed_start <= 1_800 for item in witnesses)


def test_typed_truncation_inside_transform_is_known_omission_not_mapping_failure() -> None:
    state = _state(
        "A" * 1_799 + "<b>尾部</b>。",
        max_block_characters=1_800,
    )
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    assert receipt.universe_status.value == "truncated"
    assert receipt.witness_count == 1
    assert omissions[WitnessOmissionCode.PACKED_REPRESENTATION_TRUNCATED] == 1
    assert WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN not in omissions

    rendered = _run_to_render(state, _AcceptingModel())
    assert rendered["answer_proof"]["strict_eligible"] is False


def test_silent_prefix_without_typed_truncation_is_indeterminate() -> None:
    state = _state(
        "A" * 1_799 + "<b>尾部</b>。",
        max_block_characters=1_800,
    )
    block = state["context_pack"]["packed_blocks"][0]
    assert block["truncation"] is not None
    silent_prefix = block["text"].removesuffix(PACKED_TRUNCATION_MARKER)
    block["text"] = silent_prefix
    block["truncation"] = None
    state["context_aliases"][0]["packed_text"] = silent_prefix
    mutated_pack = ContextPack.model_validate(state["context_pack"])
    state["context_pack"]["content_hash"] = replay_context_pack_content_hash(mutated_pack)

    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    assert receipt.universe_status.value == "indeterminate"
    assert receipt.witness_count == 0
    assert omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] == 1
    assert state["witness_catalog"]["source_enumeration_closure_receipts"] == []


def test_witness_freeze_rejects_a_stale_context_pack_hash_after_block_deletion() -> None:
    state = _state(
        "| 对象 | 功能 |\n| --- | --- |\n| Alpha | 功能甲 |",
        original_query="Alpha 和 Beta 分别有哪些功能？",
        additional_contents=("| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能乙 |",),
    )
    assert len(state["context_pack"]["packed_blocks"]) == 2
    state["context_pack"]["packed_blocks"].pop()
    state["context_pack"]["label_mapping"].pop()
    state["context_aliases"].pop()

    freeze_question_contract(state)
    with pytest.raises(ValueError, match="ContextPack content hash"):
        freeze_witness_catalog(state)


def test_literal_truncation_marker_is_normal_citable_text() -> None:
    canonical = f"前文事实。{PACKED_TRUNCATION_MARKER}\n后文事实。"
    state = _state(canonical)
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    assert receipt.universe_status.value == "complete"
    assert receipt.omissions == ()
    assert any("[证据正文因上下文预算被截断]" in item["content"] for item in views)


@pytest.mark.parametrize("keep_safe_only", (False, True))
def test_retrieval_witness_requires_versioned_safe_derivation_receipt(
    keep_safe_only: bool,
) -> None:
    state = _state("启动设备前必须检查保护接地。")
    document = state["context_aliases"][0]["document"]
    document.pop("safe_representation_derivation", None)
    if not keep_safe_only:
        document.pop("security_safe_content", None)
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    assert receipt.universe_status.value == "indeterminate"
    assert receipt.witness_count == 0
    assert omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] == 1


def test_safe_mapping_resource_limit_is_explicit_truncation_not_unknown_mapping() -> None:
    state = _state(("ﬃe\u0301" * 16_385) + "。")
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    assert receipt.universe_status.value == "truncated"
    assert receipt.witness_count == 0
    assert omissions[WitnessOmissionCode.SAFE_MAPPING_BUDGET_EXCEEDED] == 1
    assert WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN not in omissions


def test_empty_safe_evidence_is_quarantined_and_remaining_evidence_continues() -> None:
    state = _state("\u200b", additional_contents=("有效证据。",))

    assert len(state["context_aliases"]) == 1
    evidence_security = state["content_security"]["evidence"]
    assert evidence_security["approved_count"] == 1
    assert evidence_security["quarantined_count"] == 1
    assert evidence_security["quarantined"][0]["reason_codes"] == ["SAFE_REPRESENTATION_EMPTY"]

    rendered = _run_to_render(state, _AcceptingModel())
    assert rendered["answer_proof"]["strict_eligible"] is True
    assert rendered["citations"][0]["source_quote"] == "有效证据。"


@pytest.mark.parametrize("contents", (("\u200b",), ("\u200b", "<!-- removed -->")))
def test_all_empty_safe_evidence_produces_deterministic_abstention(
    contents: tuple[str, ...],
) -> None:
    state = _state(contents[0], additional_contents=contents[1:])

    assert state["context_aliases"] == []
    assert state["content_security"]["evidence"]["approved_count"] == 0
    rendered = _run_to_render(state, _AcceptingModel(refuse=True))
    assert rendered["answer_proof"]["overall_status"] == "accepted_abstention"
    assert rendered["citations"] == []


def test_repeated_identical_atoms_keep_distinct_canonical_coordinates() -> None:
    state = _state("重复事实。重复事实。")
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    witnesses = tuple(
        EvidenceWitness.from_dict(item) for item in state["witness_catalog"]["witnesses"]
    )
    assert len(witnesses) == 2
    assert witnesses[0].span_ref.canonical_start == 0
    assert witnesses[1].span_ref.canonical_start == len("重复事实。")


def test_safe_derivation_replays_once_per_artifact_during_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state("<b>第一项</b>。<i>第二项</i>。")
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    plan_answer(state, model_port=_MultiSingleModel())
    plan = AnswerPlan.from_dict(state["answer_plan"])
    original = safe_representation.derive_safe_representation
    calls = 0

    def counted(content: str):
        nonlocal calls
        calls += 1
        return original(content)

    monkeypatch.setattr(safe_representation, "derive_safe_representation", counted)
    answer_rendering._render_claims(
        state,
        plan,
        state["witness_catalog"],
    )

    assert calls == 1


def test_renderer_rejects_active_canonical_evidence_drift_outside_evaluation_mode() -> None:
    state = _state("<b>有效事实</b>。")
    state["evaluation_mode"] = False
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    plan_answer(state, model_port=_AcceptingModel())
    plan = AnswerPlan.from_dict(state["answer_plan"])
    state["context_aliases"][0]["evidence"]["content"] = "<b>漂移事实</b>。"

    with pytest.raises(ValueError):
        answer_rendering._render_claims(
            state,
            plan,
            state["witness_catalog"],
        )


def test_unknown_safe_transform_fails_closed_without_citation() -> None:
    state = _run_to_render(
        _state(
            "启动设备前必须检查保护接地。",
            through_security=True,
            tampered_safe_content="未经登记的改写。",
        ),
        _AcceptingModel(refuse=True),
    )

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    assert receipt.universe_status.value == "indeterminate"
    assert receipt.witness_count == 0
    assert omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] >= 1
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["citations"] == []
    validate_committable_answer(state)


def test_unproven_deletion_derivation_fails_closed_without_citation() -> None:
    state = _run_to_render(
        _state("启动\u200b前检查保护接地。", through_security=True),
        _AcceptingModel(refuse=True),
    )

    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])
    omissions = {item.code: item.count for item in receipt.omissions}
    derivation = state["context_aliases"][0]["document"]["safe_representation_derivation"]
    assert derivation["unproven_transform_count"] == 1
    assert receipt.universe_status.value == "indeterminate"
    assert receipt.witness_count == 0
    assert omissions[WitnessOmissionCode.SOURCE_MAPPING_UNPROVEN] == 1
    assert state["citations"] == []
    validate_committable_answer(state)


def test_empty_catalog_builds_safe_local_abstention_not_global_absence() -> None:
    state = _run_to_render(_state(None), _AcceptingModel(refuse=True))

    assert state["answer_proof"]["overall_status"] == "accepted_abstention"
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["answer_proof"]["obligation_proofs"][0]["outcome"] == (
        "insufficient_retrieved_evidence"
    )
    assert len(state["answer_proof"]["claim_proofs"]) == 1
    assert state["answer_envelope"]["decision"] == "REFUSE"
    assert state["citations"] == []


def test_conditional_primary_query_builds_a_valid_refusal_plan() -> None:
    state = _state(
        None,
        original_query="如果状态灯持续闪烁，设备应如何处理？",
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    plan_answer(state, model_port=_AcceptingModel(refuse=True))

    required_ids = state["question_contract"]["obligations"][0]["obligation_id"]
    assert state["question_contract"]["obligations"][0]["necessity"] == "required"
    assert state["answer_plan"]["decision"] == "refuse"
    assert state["answer_plan"]["unresolved_obligation_ids"] == [required_ids]


def test_single_obligation_partial_is_never_upgraded_to_strict_answer() -> None:
    state = _run_to_render(_state("启动前先核对设备状态。"), _PartialModel())

    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer_proof"]["overall_status"] == "accepted_partial"
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["answer_proof"]["obligation_proofs"][0]["outcome"] == ("partially_answered")


def test_single_obligation_with_multiple_claims_is_downgraded_before_review() -> None:
    state = _run_to_render(
        _state("启动前检查保护接地。还要核对急停开关。"),
        _MultiSingleModel(),
    )

    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer_plan_normalizations"][-1]["code"] == ("LOCAL_CLOSURE_DOWNGRADE")
    assert state["answer_proof"]["strict_eligible"] is False


def test_open_enumeration_with_multiple_items_renders_as_committable_partial() -> None:
    state = _run_to_render(
        _state(
            "1.可查看打印机的基本信息。\n"
            "2.可配置打印机支持的网络协议参数。\n"
            "3.可配置打印机的无线网络连接。\n"
            "4.可配置邮件服务器客户端。",
            original_query="通过内嵌Web服务器可以配置哪些功能？",
        ),
        _OpenEnumerationModel(),
    )

    assert state["question_contract"]["obligations"][0]["closure"] == "open_enumeration"
    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer_proof"]["overall_status"] == "accepted_partial"
    assert state["answer_proof"]["strict_eligible"] is False
    assert len(state["citations"]) == 4
    proven_items = state["answer_proof"]["obligation_proofs"][0]["proven_items"]
    assert len(proven_items) == 4
    assert {item["segmentation_method"] for item in proven_items} == {"top_level_list_entry.v1"}
    assert state["answer_envelope"]["decision"] == "PARTIAL_ANSWER"
    validate_committable_answer(state)
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    audit = answer_proof_evaluation(observation)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is False


def test_query_anchored_table_row_closes_open_enumeration_with_replayable_proof() -> None:
    table = (
        "| 序号 | 按键 | 操作 |\n"
        "| --- | --- | --- |\n"
        "| 7 | 返回键 | 使用此按键执行以下操作：退出菜单；返回上一级。 |"
    )
    state = _state(table, original_query="返回键可以执行哪些操作？")

    freeze_question_contract(state)
    assert state["question_contract"]["obligations"][0]["closure"] == "open_enumeration"
    freeze_witness_catalog(state)
    bundle = state["witness_catalog"]
    receipts = bundle["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    assert receipts[0]["schema_version"] == ("query-anchored-table-row-closure-receipt.v4")

    views = resolve_witness_prompt_views(state, bundle)
    flagged = [item for item in views if item["source_closed_open_obligation_ids"]]
    assert len(flagged) == 1
    assert flagged[0]["source_closed_open_obligation_ids"] == (
        state["question_contract"]["obligations"][0]["obligation_id"],
    )

    model = _SourceClosedOpenEnumerationModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert state["answer_plan"]["decision"] == "answer"
    assert state["answer_proof"]["overall_status"] == "accepted"
    assert state["answer_proof"]["strict_eligible"] is True
    obligation_proof = state["answer_proof"]["obligation_proofs"][0]
    assert obligation_proof["outcome"] == "answered"
    assert obligation_proof["schema_version"] == "obligation-proof.v3"
    assert obligation_proof["source_enumeration_closure_receipt"] == receipts[0]
    validate_committable_answer(state)

    proof_observation = project_proof_observation(state)
    serialized_observation = json.dumps(proof_observation, ensure_ascii=False)
    assert "返回键" not in serialized_observation
    assert "退出菜单" not in serialized_observation
    assert "| 序号 |" not in serialized_observation
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **proof_observation,
    }
    proof_authority = build_proof_evaluation_authority(state)
    replay_without_authority = answer_proof_evaluation(observation)
    assert replay_without_authority is not None
    assert replay_without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in replay_without_authority["reason_codes"]
    audit = answer_proof_evaluation(observation, proof_authority=proof_authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    missing_receipt = copy.deepcopy(observation)
    missing_receipt["answer_proof"]["obligation_proofs"][0].pop(
        "source_enumeration_closure_receipt"
    )
    missing_audit = answer_proof_evaluation(
        missing_receipt,
        proof_authority=proof_authority,
    )
    assert missing_audit is not None
    assert missing_audit["contract_valid"] is False
    assert missing_audit["strict_graph_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_missing" in (missing_audit["reason_codes"])

    forged_receipt = copy.deepcopy(observation)
    forged_receipt["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "receipt_hash"
    ] = "sha256:" + "0" * 64
    forged_audit = answer_proof_evaluation(
        forged_receipt,
        proof_authority=proof_authority,
    )
    assert forged_audit is not None
    assert forged_audit["contract_valid"] is False
    assert forged_audit["strict_graph_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in (forged_audit["reason_codes"])

    locally_rehashed = copy.deepcopy(observation)
    nested = locally_rehashed["answer_proof"]["obligation_proofs"][0][
        "source_enumeration_closure_receipt"
    ]
    nested["table_start"] = 1
    nested_payload = {
        key: value for key, value in nested.items() if key not in {"receipt_id", "receipt_hash"}
    }
    nested["receipt_hash"] = stable_content_hash(nested_payload)
    nested["receipt_id"] = "qtrc_" + nested["receipt_hash"].removeprefix("sha256:")[:40]
    locally_rehashed_audit = answer_proof_evaluation(
        locally_rehashed,
        proof_authority=proof_authority,
    )
    assert locally_rehashed_audit is not None
    assert locally_rehashed_audit["contract_valid"] is False
    assert locally_rehashed_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" in locally_rehashed_audit["reason_codes"]
    assert (
        "answer_proof_open_enumeration_source_closure_invalid"
        in (locally_rehashed_audit["reason_codes"])
    )

    catalog_receipt_drift = copy.deepcopy(observation)
    catalog_receipt_drift["witness_catalog_receipt"]["source_enumeration_closure_receipts"] = []
    catalog_receipt_drift_audit = answer_proof_evaluation(
        catalog_receipt_drift,
        proof_authority=proof_authority,
    )
    assert catalog_receipt_drift_audit is not None
    assert catalog_receipt_drift_audit["contract_valid"] is False
    assert catalog_receipt_drift_audit["strict_graph_valid"] is False
    assert (
        "answer_proof_open_enumeration_source_closure_invalid"
        in (catalog_receipt_drift_audit["reason_codes"])
    )

    coordinated_row_count_drift = copy.deepcopy(observation)
    coordinated_proof = coordinated_row_count_drift["answer_proof"]
    coordinated_obligation = coordinated_proof["obligation_proofs"][0]
    coordinated_nested = coordinated_obligation["source_enumeration_closure_receipt"]
    coordinated_nested["table_data_row_count"] = 2
    coordinated_nested_payload = {
        key: value
        for key, value in coordinated_nested.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    coordinated_nested["receipt_hash"] = stable_content_hash(coordinated_nested_payload)
    coordinated_nested["receipt_id"] = (
        "qtrc_" + coordinated_nested["receipt_hash"].removeprefix("sha256:")[:40]
    )
    coordinated_row_count_drift["witness_catalog_receipt"]["source_enumeration_closure_receipts"][
        0
    ] = copy.deepcopy(coordinated_nested)
    coordinated_obligation_serialized = evaluation_proof_module._obligation_proof_serialization(
        coordinated_obligation
    )
    coordinated_obligation_payload = {
        key: value
        for key, value in coordinated_obligation_serialized.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    coordinated_obligation["obligation_proof_hash"] = stable_content_hash(
        coordinated_obligation_payload
    )
    coordinated_obligation["obligation_proof_id"] = (
        "opr_" + coordinated_obligation["obligation_proof_hash"].removeprefix("sha256:")[:40]
    )
    coordinated_proof_payload = evaluation_proof_module._answer_proof_identity_payload(
        coordinated_proof
    )
    coordinated_proof["proof_hash"] = stable_content_hash(coordinated_proof_payload)
    coordinated_proof["proof_id"] = (
        "apr_" + coordinated_proof["proof_hash"].removeprefix("sha256:")[:40]
    )
    coordinated_row_count_drift_audit = answer_proof_evaluation(
        coordinated_row_count_drift,
        proof_authority=proof_authority,
    )
    assert coordinated_row_count_drift_audit is not None
    assert coordinated_row_count_drift_audit["contract_valid"] is False
    assert coordinated_row_count_drift_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in coordinated_row_count_drift_audit["reason_codes"]
    assert (
        "answer_proof_obligation_schema_invalid"
        not in coordinated_row_count_drift_audit["reason_codes"]
    )
    assert (
        "answer_proof_open_enumeration_source_closure_invalid"
        in coordinated_row_count_drift_audit["reason_codes"]
    )

    coordinated_key_drift = copy.deepcopy(observation)
    key_proof = coordinated_key_drift["answer_proof"]
    key_obligation = key_proof["obligation_proofs"][0]
    key_receipt = key_obligation["source_enumeration_closure_receipt"]
    key_receipt["key_cell_index"] = 0
    key_receipt["key_hash"] = sha256_digest("7")
    key_receipt["anchor_field"] = "source_query"
    key_payload = {
        key: value
        for key, value in key_receipt.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    key_receipt["receipt_hash"] = stable_content_hash(key_payload)
    key_receipt["receipt_id"] = "qtrc_" + key_receipt["receipt_hash"].removeprefix("sha256:")[:40]
    key_catalog = coordinated_key_drift["witness_catalog_receipt"]
    key_catalog["source_enumeration_closure_receipts"][0] = copy.deepcopy(key_receipt)
    key_summary = key_catalog["source_table_authority_summaries"][0]
    key_summary["receipt_id"] = key_receipt["receipt_id"]
    key_summary["receipt_hash"] = key_receipt["receipt_hash"]
    key_obligation_serialized = evaluation_proof_module._obligation_proof_serialization(
        key_obligation
    )
    key_obligation_payload = {
        key: value
        for key, value in key_obligation_serialized.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    key_obligation["obligation_proof_hash"] = stable_content_hash(key_obligation_payload)
    key_obligation["obligation_proof_id"] = (
        "opr_" + key_obligation["obligation_proof_hash"].removeprefix("sha256:")[:40]
    )
    key_proof_payload = evaluation_proof_module._answer_proof_identity_payload(key_proof)
    key_proof["proof_hash"] = stable_content_hash(key_proof_payload)
    key_proof["proof_id"] = "apr_" + key_proof["proof_hash"].removeprefix("sha256:")[:40]
    key_drift_audit = answer_proof_evaluation(
        coordinated_key_drift,
        proof_authority=proof_authority,
    )
    assert key_drift_audit is not None
    assert key_drift_audit["contract_valid"] is False
    assert key_drift_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in key_drift_audit["reason_codes"]
    assert "answer_proof_obligation_schema_invalid" not in key_drift_audit["reason_codes"]
    assert "answer_proof_open_enumeration_source_closure_invalid" in key_drift_audit["reason_codes"]

    tampered = copy.deepcopy(state)
    tampered["context_aliases"][0]["evidence"]["content"] = table.replace(
        "返回上一级", "恢复出厂设置"
    )
    with pytest.raises(ValueError):
        validate_committable_answer(tampered)


def test_query_anchored_procedure_list_closes_only_with_full_replayable_sequence() -> None:
    procedure = "打印产品报告步骤\n1. 打开控制面板。\n2. 选择打印信息报告。\n3. 按下确定键。"
    state = _state(procedure, original_query="请说明全部打印产品报告步骤")

    freeze_question_contract(state)
    obligation = state["question_contract"]["obligations"][0]
    assert obligation["closure"] == "open_enumeration"
    assert obligation["kind"] == "procedure"
    assert obligation["required_output_form"] == "ordered_steps"
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-procedure-list-closure-receipt.v1")
    assert receipt["item_count"] == 3

    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    flagged = [item for item in views if item["source_closed_open_obligation_ids"]]
    assert [item["witness_id"] for item in flagged] == receipt["witness_ids"]
    assert len(flagged) == 3

    model = _SourceClosedProcedureModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert state["answer_plan"]["decision"] == "answer"
    assert state["answer_proof"]["overall_status"] == "accepted"
    assert state["answer_proof"]["strict_eligible"] is True
    assert "1. 1." not in state["answer"]
    obligation_proof = state["answer_proof"]["obligation_proofs"][0]
    assert [item["witness_id"] for item in obligation_proof["proven_items"]] == receipt[
        "witness_ids"
    ]
    assert {item["segmentation_method"] for item in obligation_proof["proven_items"]} == {
        "top_level_list_entry.v1"
    }
    assert obligation_proof["source_enumeration_closure_receipt"] == receipt
    validate_committable_answer(state)

    proof_observation = project_proof_observation(state)
    serialized = json.dumps(proof_observation, ensure_ascii=False)
    assert "打开控制面板" not in serialized
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **proof_observation,
    }
    proof_authority = build_proof_evaluation_authority(state)
    replay_without_authority = answer_proof_evaluation(observation)
    assert replay_without_authority is not None
    assert replay_without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in replay_without_authority["reason_codes"]
    audit = answer_proof_evaluation(observation, proof_authority=proof_authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    receipt_drift = copy.deepcopy(observation)
    receipt_drift["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "witness_ids"
    ] = list(reversed(receipt["witness_ids"]))
    drift_audit = answer_proof_evaluation(
        receipt_drift,
        proof_authority=proof_authority,
    )
    assert drift_audit is not None
    assert drift_audit["contract_valid"] is False
    assert drift_audit["strict_graph_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in drift_audit["reason_codes"]

    hash_drift = copy.deepcopy(observation)
    hash_drift["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "receipt_hash"
    ] = "sha256:" + "0" * 64
    hash_audit = answer_proof_evaluation(
        hash_drift,
        proof_authority=proof_authority,
    )
    assert hash_audit is not None
    assert hash_audit["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in hash_audit["reason_codes"]

    def rehash_proof_graph(value: dict[str, Any]) -> None:
        proof = value["answer_proof"]
        obligation_row = proof["obligation_proofs"][0]
        obligation_serialized = evaluation_proof_module._obligation_proof_serialization(
            obligation_row
        )
        obligation_payload = {
            key: item
            for key, item in obligation_serialized.items()
            if key not in {"obligation_proof_id", "obligation_proof_hash"}
        }
        obligation_row["obligation_proof_hash"] = stable_content_hash(obligation_payload)
        obligation_row["obligation_proof_id"] = (
            "opr_" + obligation_row["obligation_proof_hash"].removeprefix("sha256:")[:40]
        )
        proof_payload = evaluation_proof_module._answer_proof_identity_payload(proof)
        proof["proof_hash"] = stable_content_hash(proof_payload)
        proof["proof_id"] = "apr_" + proof["proof_hash"].removeprefix("sha256:")[:40]

    coordinated_marker = copy.deepcopy(observation)
    marker_proof = coordinated_marker["answer_proof"]
    marker_obligation = marker_proof["obligation_proofs"][0]
    marker_receipt = marker_obligation["source_enumeration_closure_receipt"]
    marker_receipt["marker_style"] = "arabic:)"
    marker_payload = {
        key: value
        for key, value in marker_receipt.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    marker_receipt["receipt_hash"] = stable_content_hash(marker_payload)
    marker_receipt["receipt_id"] = (
        "qplc_" + marker_receipt["receipt_hash"].removeprefix("sha256:")[:40]
    )
    marker_catalog = coordinated_marker["witness_catalog_receipt"]
    marker_catalog["source_enumeration_closure_receipts"][0] = copy.deepcopy(marker_receipt)
    marker_summary = marker_catalog["source_procedure_authority_summaries"][0]
    marker_summary["receipt_id"] = marker_receipt["receipt_id"]
    marker_summary["receipt_hash"] = marker_receipt["receipt_hash"]
    marker_summary["canonical"]["marker_style"] = "arabic:)"
    marker_summary["packed"]["marker_style"] = "arabic:)"
    rehash_proof_graph(coordinated_marker)

    marker_audit = answer_proof_evaluation(
        coordinated_marker,
        proof_authority=proof_authority,
    )
    assert marker_audit is not None
    assert marker_audit["contract_valid"] is False
    assert marker_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in marker_audit["reason_codes"]
    assert "answer_proof_open_enumeration_source_closure_invalid" in marker_audit["reason_codes"]

    coordinated_trim = copy.deepcopy(observation)
    coordinated_trim["answer"] = "\n".join(coordinated_trim["answer"].splitlines()[:2])
    trim_plan = coordinated_trim["answer_plan_receipt"]
    trim_plan["claims"] = trim_plan["claims"][:2]
    trim_plan["claim_count"] = 2
    trim_proof = coordinated_trim["answer_proof"]
    trim_proof["answer_hash"] = sha256_digest(coordinated_trim["answer"])
    trim_proof["claim_proofs"] = trim_proof["claim_proofs"][:2]
    trim_obligation = trim_proof["obligation_proofs"][0]
    trim_obligation["supporting_claim_ids"] = trim_obligation["supporting_claim_ids"][:2]
    trim_obligation["proven_items"] = trim_obligation["proven_items"][:2]
    trim_obligation["proven_item_count"] = 2
    trim_receipt = trim_obligation["source_enumeration_closure_receipt"]
    for field in (
        "witness_ids",
        "witness_hashes",
        "witness_ordinals",
        "span_ref_hashes",
        "canonical_item_hashes",
        "packed_item_hashes",
        "item_ordinals",
    ):
        trim_receipt[field] = trim_receipt[field][:2]
    trim_receipt["item_count"] = 2
    partial_source = "\n".join(procedure.splitlines()[:3])
    trim_receipt["canonical_block_end"] = len(partial_source)
    trim_receipt["packed_block_end"] = len(partial_source)
    trim_receipt["canonical_block_hash"] = sha256_digest(partial_source)
    trim_receipt["packed_block_hash"] = sha256_digest(partial_source)
    trimmed_structure = {
        "title": evaluation_proof_module._procedure_title_normalized(procedure.splitlines()[0]),
        "title_line_index": 0,
        "first_item_line_index": 1,
        "marker_style": "arabic:.",
        "ordinals": [1, 2],
        "items": [
            evaluation_proof_module._normalized_table_cell(
                evaluation_proof_module._procedure_ordered_item(line)[2]
            )
            for line in procedure.splitlines()[1:3]
        ],
    }
    trim_receipt["list_structure_hash"] = stable_content_hash(trimmed_structure)
    trim_payload = {
        key: value
        for key, value in trim_receipt.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    trim_receipt["receipt_hash"] = stable_content_hash(trim_payload)
    trim_receipt["receipt_id"] = "qplc_" + trim_receipt["receipt_hash"].removeprefix("sha256:")[:40]
    trim_catalog = coordinated_trim["witness_catalog_receipt"]
    trim_catalog["source_enumeration_closure_receipts"][0] = copy.deepcopy(trim_receipt)
    trim_summary = trim_catalog["source_procedure_authority_summaries"][0]
    trim_summary["receipt_id"] = trim_receipt["receipt_id"]
    trim_summary["receipt_hash"] = trim_receipt["receipt_hash"]
    trim_summary["witness_ids"] = trim_receipt["witness_ids"]
    for side, hash_field in (
        ("canonical", "canonical_block_hash"),
        ("packed", "packed_block_hash"),
    ):
        summary_side = trim_summary[side]
        summary_side["block_end"] = len(partial_source)
        summary_side["item_count"] = 2
        summary_side["item_ordinals"] = [1, 2]
        summary_side["item_starts"] = summary_side["item_starts"][:2]
        summary_side["item_ends"] = summary_side["item_ends"][:2]
        summary_side["item_hashes"] = summary_side["item_hashes"][:2]
        summary_side["block_hash"] = trim_receipt[hash_field]
        summary_side["list_structure_hash"] = trim_receipt["list_structure_hash"]
    rehash_proof_graph(coordinated_trim)

    trim_audit = answer_proof_evaluation(
        coordinated_trim,
        proof_authority=proof_authority,
    )
    assert trim_audit is not None
    assert trim_audit["contract_valid"] is False
    assert trim_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in trim_audit["reason_codes"]
    assert "answer_proof_answer_plan_mismatch" in trim_audit["reason_codes"]
    assert "answer_proof_open_enumeration_source_closure_invalid" in trim_audit["reason_codes"]

    fully_rehashed_trim = copy.deepcopy(coordinated_trim)
    rehashed_plan = fully_rehashed_trim["answer_plan_receipt"]
    serialized_plan_claims = [
        evaluation_proof_module._claim_plan_serialization(item) for item in rehashed_plan["claims"]
    ]
    assert all(item is not None for item in serialized_plan_claims)
    rehashed_plan_payload = {
        "schema_version": "answer-plan.v1",
        "question_contract_id": rehashed_plan["question_contract_id"],
        "question_contract_hash": rehashed_plan["question_contract_hash"],
        "context_pack_hash": rehashed_plan["context_pack_hash"],
        "witness_catalog_id": rehashed_plan["witness_catalog_id"],
        "witness_catalog_hash": rehashed_plan["witness_catalog_hash"],
        "decision": rehashed_plan["decision"],
        "claims": serialized_plan_claims,
        "unresolved_obligation_ids": rehashed_plan["unresolved_obligation_ids"],
        "proposal_schema_version": rehashed_plan["proposal_schema_version"],
        "prompt_contract_version": rehashed_plan["prompt_contract_version"],
    }
    rehashed_plan["plan_hash"] = stable_content_hash(rehashed_plan_payload)
    rehashed_plan["plan_id"] = "apn_" + rehashed_plan["plan_hash"].removeprefix("sha256:")[:40]
    fully_rehashed_proof = fully_rehashed_trim["answer_proof"]
    fully_rehashed_proof["answer_plan_id"] = rehashed_plan["plan_id"]
    fully_rehashed_proof["answer_plan_hash"] = rehashed_plan["plan_hash"]
    semantic_receipt = fully_rehashed_proof["semantic_review_closure"]
    semantic_receipt["answer_plan_id"] = rehashed_plan["plan_id"]
    semantic_receipt["answer_plan_hash"] = rehashed_plan["plan_hash"]
    semantic_serialized = evaluation_proof_module._semantic_review_serialization(semantic_receipt)
    assert semantic_serialized is not None
    semantic_payload = {
        key: value
        for key, value in semantic_serialized.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    semantic_receipt["receipt_hash"] = stable_content_hash(semantic_payload)
    semantic_receipt["receipt_id"] = (
        "src_" + semantic_receipt["receipt_hash"].removeprefix("sha256:")[:40]
    )
    verification_receipt = fully_rehashed_trim["answer_plan_review_receipt"]["verification"]
    verification_receipt["plan_id"] = rehashed_plan["plan_id"]
    verification_receipt["plan_hash"] = rehashed_plan["plan_hash"]
    rehash_proof_graph(fully_rehashed_trim)

    fully_rehashed_trim_audit = answer_proof_evaluation(
        fully_rehashed_trim,
        proof_authority=proof_authority,
    )
    assert fully_rehashed_trim_audit is not None
    assert fully_rehashed_trim_audit["contract_valid"] is False
    assert fully_rehashed_trim_audit["strict_graph_valid"] is False
    assert "answer_proof_answer_plan_mismatch" not in fully_rehashed_trim_audit["reason_codes"]
    assert (
        "answer_proof_open_enumeration_source_closure_invalid"
        in (fully_rehashed_trim_audit["reason_codes"])
    )

    coordinated_catalog_trim = copy.deepcopy(fully_rehashed_trim)
    catalog_trim_proof = coordinated_catalog_trim["answer_proof"]
    catalog_trim_obligation = catalog_trim_proof["obligation_proofs"][0]
    catalog_trim_receipt = catalog_trim_obligation["source_enumeration_closure_receipt"]
    public_catalog = coordinated_catalog_trim["witness_catalog_receipt"]
    removed_witness_id = receipt["witness_ids"][2]
    remaining_catalog_ids = [
        item for item in public_catalog["witness_ids"] if item != removed_witness_id
    ]
    catalog_payload = {
        "schema_version": public_catalog["schema_version"],
        "question_contract_id": public_catalog["question_contract_id"],
        "question_contract_hash": public_catalog["question_contract_hash"],
        "context_pack_hash": public_catalog["context_pack_hash"],
        "authorized_scope_hash": public_catalog["authorized_scope_hash"],
        "witness_ids": remaining_catalog_ids,
        "witness_count": len(remaining_catalog_ids),
        "universe_status": public_catalog["universe_status"],
        "omissions": [],
        "builder_version": public_catalog["builder_version"],
    }
    new_catalog_hash = stable_content_hash(catalog_payload)
    new_catalog_id = "wcat_" + new_catalog_hash.removeprefix("sha256:")[:40]
    public_catalog["catalog_id"] = new_catalog_id
    public_catalog["catalog_hash"] = new_catalog_hash
    public_catalog["witness_ids"] = remaining_catalog_ids
    public_catalog["witness_count"] = len(remaining_catalog_ids)
    for field in (
        "witness_atom_kinds",
        "witness_hashes",
        "witness_evidence_ids",
        "witness_evidence_content_hashes",
        "witness_representation_ids",
        "witness_span_ref_hashes",
        "witness_canonical_span_hashes",
        "witness_packed_span_hashes",
        "witness_canonical_starts",
        "witness_canonical_ends",
        "witness_packed_starts",
        "witness_packed_ends",
    ):
        public_catalog[field].pop(removed_witness_id)

    catalog_trim_receipt["catalog_id"] = new_catalog_id
    catalog_trim_receipt["catalog_hash"] = new_catalog_hash
    catalog_trim_payload = {
        key: value
        for key, value in catalog_trim_receipt.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    catalog_trim_receipt["receipt_hash"] = stable_content_hash(catalog_trim_payload)
    catalog_trim_receipt["receipt_id"] = (
        "qplc_" + catalog_trim_receipt["receipt_hash"].removeprefix("sha256:")[:40]
    )
    public_catalog["source_enumeration_closure_receipts"][0] = copy.deepcopy(catalog_trim_receipt)
    catalog_trim_summary = public_catalog["source_procedure_authority_summaries"][0]
    catalog_trim_summary["receipt_id"] = catalog_trim_receipt["receipt_id"]
    catalog_trim_summary["receipt_hash"] = catalog_trim_receipt["receipt_hash"]

    catalog_trim_plan = coordinated_catalog_trim["answer_plan_receipt"]
    catalog_trim_plan["witness_catalog_id"] = new_catalog_id
    catalog_trim_plan["witness_catalog_hash"] = new_catalog_hash
    catalog_trim_plan_claims = [
        evaluation_proof_module._claim_plan_serialization(item)
        for item in catalog_trim_plan["claims"]
    ]
    assert all(item is not None for item in catalog_trim_plan_claims)
    catalog_trim_plan_payload = {
        "schema_version": "answer-plan.v1",
        "question_contract_id": catalog_trim_plan["question_contract_id"],
        "question_contract_hash": catalog_trim_plan["question_contract_hash"],
        "context_pack_hash": catalog_trim_plan["context_pack_hash"],
        "witness_catalog_id": new_catalog_id,
        "witness_catalog_hash": new_catalog_hash,
        "decision": catalog_trim_plan["decision"],
        "claims": catalog_trim_plan_claims,
        "unresolved_obligation_ids": catalog_trim_plan["unresolved_obligation_ids"],
        "proposal_schema_version": catalog_trim_plan["proposal_schema_version"],
        "prompt_contract_version": catalog_trim_plan["prompt_contract_version"],
    }
    catalog_trim_plan["plan_hash"] = stable_content_hash(catalog_trim_plan_payload)
    catalog_trim_plan["plan_id"] = (
        "apn_" + catalog_trim_plan["plan_hash"].removeprefix("sha256:")[:40]
    )

    catalog_trim_obligation["catalog_id"] = new_catalog_id
    catalog_trim_obligation["catalog_hash"] = new_catalog_hash
    catalog_trim_obligation_serialized = evaluation_proof_module._obligation_proof_serialization(
        catalog_trim_obligation
    )
    catalog_trim_obligation_payload = {
        key: value
        for key, value in catalog_trim_obligation_serialized.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    catalog_trim_obligation["obligation_proof_hash"] = stable_content_hash(
        catalog_trim_obligation_payload
    )
    catalog_trim_obligation["obligation_proof_id"] = (
        "opr_" + catalog_trim_obligation["obligation_proof_hash"].removeprefix("sha256:")[:40]
    )

    catalog_trim_semantic = catalog_trim_proof["semantic_review_closure"]
    catalog_trim_semantic["answer_plan_id"] = catalog_trim_plan["plan_id"]
    catalog_trim_semantic["answer_plan_hash"] = catalog_trim_plan["plan_hash"]
    catalog_trim_semantic["catalog_id"] = new_catalog_id
    catalog_trim_semantic["catalog_hash"] = new_catalog_hash
    catalog_trim_semantic_serialized = evaluation_proof_module._semantic_review_serialization(
        catalog_trim_semantic
    )
    assert catalog_trim_semantic_serialized is not None
    catalog_trim_semantic_payload = {
        key: value
        for key, value in catalog_trim_semantic_serialized.items()
        if key not in {"receipt_id", "receipt_hash"}
    }
    catalog_trim_semantic["receipt_hash"] = stable_content_hash(catalog_trim_semantic_payload)
    catalog_trim_semantic["receipt_id"] = (
        "src_" + catalog_trim_semantic["receipt_hash"].removeprefix("sha256:")[:40]
    )
    catalog_trim_proof["answer_plan_id"] = catalog_trim_plan["plan_id"]
    catalog_trim_proof["answer_plan_hash"] = catalog_trim_plan["plan_hash"]
    catalog_trim_proof["witness_catalog_id"] = new_catalog_id
    catalog_trim_proof["witness_catalog_hash"] = new_catalog_hash
    catalog_trim_verification = coordinated_catalog_trim["answer_plan_review_receipt"][
        "verification"
    ]
    catalog_trim_verification["plan_id"] = catalog_trim_plan["plan_id"]
    catalog_trim_verification["plan_hash"] = catalog_trim_plan["plan_hash"]
    rehash_proof_graph(coordinated_catalog_trim)

    coordinated_catalog_trim_audit = answer_proof_evaluation(
        coordinated_catalog_trim,
        proof_authority=proof_authority,
    )
    assert coordinated_catalog_trim_audit is not None
    assert coordinated_catalog_trim_audit["contract_valid"] is False
    assert coordinated_catalog_trim_audit["strict_graph_valid"] is False
    assert "answer_proof_identity_invalid" not in coordinated_catalog_trim_audit["reason_codes"]
    assert "answer_proof_answer_plan_mismatch" not in coordinated_catalog_trim_audit["reason_codes"]
    assert (
        "answer_proof_obligation_schema_invalid"
        not in coordinated_catalog_trim_audit["reason_codes"]
    )
    assert (
        "answer_proof_open_enumeration_source_closure_invalid"
        in (coordinated_catalog_trim_audit["reason_codes"])
    )

    content_drift = copy.deepcopy(state)
    content_drift["context_aliases"][0]["evidence"]["content"] = procedure.replace(
        "选择打印信息报告",
        "删除打印信息报告",
    )
    with pytest.raises(ValueError):
        validate_committable_answer(content_drift)

    omitted = _run_to_render(
        _state(procedure, original_query="请说明全部打印产品报告步骤"),
        _SourceClosedProcedureModel(omit_last=True),
    )
    assert omitted["answer_plan"]["decision"] == "partial"
    assert omitted["answer_proof"]["strict_eligible"] is False


def test_closed_count_projects_exact_items_from_one_query_anchored_table_cell() -> None:
    table = (
        "| 故障现象 | 故障原因 | 解决办法 |\n"
        "| --- | --- | --- |\n"
        "| 打印全黑版 | 未正确安装激光碳粉盒。激光碳粉盒内部损坏。"
        "机器内部充电异常,未给激光碳粉盒充电。 | "
        "确保正确安装激光碳粉盒。建议更换原装激光碳粉盒。 |\n"
        "| 碳粉晕开 | 机器内部脏污。 | 清洁机器内部。 |"
    )
    query = "打印全黑版的故障原因是什么？解决办法中建议采取哪两项措施？"
    state = _run_to_render(
        _state(table, original_query=query),
        _ProjectedTableCellModel(),
    )

    assert [item["closure"] for item in state["question_contract"]["obligations"]] == [
        "single",
        "closed_count",
    ]
    closed = state["answer_proof"]["obligation_proofs"][1]
    assert closed["outcome"] == "answered"
    assert closed["proven_item_count"] == 2
    assert {item["claim_id"] for item in closed["proven_items"]} == {
        closed["supporting_claim_ids"][0]
    }
    assert {item["segmentation_method"] for item in closed["proven_items"]} == {
        "markdown_table_projected_cell_item.v1"
    }
    witness_id = closed["proven_items"][0]["witness_id"]
    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    witness_text = next(str(item["content"]) for item in views if item["witness_id"] == witness_id)
    assert [
        witness_text[item["relative_start"] : item["relative_end"]]
        for item in closed["proven_items"]
    ] == [
        "确保正确安装激光碳粉盒。",
        "建议更换原装激光碳粉盒。",
    ]
    assert state["answer_proof"]["strict_eligible"] is True
    validate_committable_answer(state)

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    without_authority = answer_proof_evaluation(observation)
    assert without_authority is not None
    assert without_authority["contract_valid"] is False
    assert "answer_proof_closed_count_table_cell_invalid" in without_authority["reason_codes"]

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(
            authority,
            evidences=tuple(
                replace(
                    item,
                    canonical_text=item.canonical_text.replace(
                        "建议更换原装激光碳粉盒。",
                        "建议联系维护人员。",
                    ),
                )
                for item in authority.evidences
            ),
        ),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_closed_count_table_cell_invalid" in raw_drift["reason_codes"]


@pytest.mark.parametrize(
    ("query", "solution"),
    (
        (
            "打印全黑版的故障原因是什么？故障原因和解决办法中建议采取哪两项措施？",
            "确保正确安装激光碳粉盒。建议更换原装激光碳粉盒。",
        ),
        (
            "打印全黑版的故障原因是什么？解决办法中建议采取哪两项措施？",
            "确保正确安装激光碳粉盒。",
        ),
        (
            "打印全黑版的故障原因是什么？解决办法中建议采取哪两项措施？",
            "例如检查安装。建议更换原装激光碳粉盒。",
        ),
    ),
)
def test_closed_count_table_cell_fails_closed_on_column_count_or_open_scope(
    query: str,
    solution: str,
) -> None:
    table = (
        "| 故障现象 | 故障原因 | 解决办法 |\n"
        "| --- | --- | --- |\n"
        f"| 打印全黑版 | 安装异常。部件损坏。充电异常。 | {solution} |"
    )
    state = _run_to_render(
        _state(table, original_query=query),
        _ProjectedTableCellModel(),
    )

    assert state["answer_plan"]["decision"] in {"partial", "refuse"}
    assert state["answer_proof"]["strict_eligible"] is False
    closed = next(
        item
        for item in state["answer_proof"]["obligation_proofs"]
        if item["closure"] == "closed_count"
    )
    assert closed["outcome"] in {
        "partially_answered",
        "insufficient_retrieved_evidence",
    }
    assert all(
        item["segmentation_method"] != "markdown_table_projected_cell_item.v1"
        for item in closed["proven_items"]
    )


def test_scoped_table_selection_closes_all_matching_rows_with_raw_replay() -> None:
    table = (
        "| 热点 | 分组 | 加密方式 | 信号强度 |\n"
        "| --- | --- | --- | --- |\n"
        "| Test-AP1 | 常规信息 | WPA/WPA2 | 80% |\n"
        "| Test-AP2 | 无线网络信息 | WPA/WPA2 | 60% |\n"
        "| Test-AP3 | 无线网络信息 | WEP | 60% |\n"
        "| Test-AP4 | 访客信息 | 无加密 | 40% |"
    )
    state = _state(
        table,
        original_query="请列出信号强度为60%的全部热点记录",
        additional_contents=("标题路径：附录 > 网络术语\nSSID 是无线网络名称。",),
    )

    freeze_question_contract(state)
    assert state["question_contract"]["obligations"][0]["closure"] == "open_enumeration"
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-scoped-table-selection-closure-receipt.v2")
    assert receipt["selected_row_indices"] == [1, 2]
    assert receipt["selected_row_count"] == 2
    assert receipt["catalog_representation_count"] == 2
    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    assert (
        sum(receipt["obligation_id"] in item["source_closed_open_obligation_ids"] for item in views)
        == 2
    )
    assert answering_provider._validated_source_closed_witness_ids(views) == {
        receipt["obligation_id"]: tuple(receipt["witness_ids"])
    }

    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert state["answer_plan"]["decision"] == "answer"
    assert state["answer_proof"]["overall_status"] == "accepted"
    assert state["answer_proof"]["strict_eligible"] is True
    validate_committable_answer(state)

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    without_authority = answer_proof_evaluation(observation)
    assert without_authority is not None
    assert without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in without_authority["reason_codes"]

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    order_drift = copy.deepcopy(state)
    order_proof = order_drift["answer_proof"]
    order_obligation = order_proof["obligation_proofs"][0]
    order_obligation["supporting_claim_ids"] = list(
        reversed(order_obligation["supporting_claim_ids"])
    )
    serialized_obligation = evaluation_proof_module._obligation_proof_serialization(
        order_obligation
    )
    obligation_payload = {
        key: value
        for key, value in serialized_obligation.items()
        if key not in {"obligation_proof_id", "obligation_proof_hash"}
    }
    order_obligation["obligation_proof_hash"] = stable_content_hash(obligation_payload)
    order_obligation["obligation_proof_id"] = (
        "opr_" + order_obligation["obligation_proof_hash"].removeprefix("sha256:")[:40]
    )
    order_proof["proof_hash"] = stable_content_hash(
        evaluation_proof_module._answer_proof_identity_payload(order_proof)
    )
    order_proof["proof_id"] = "apr_" + order_proof["proof_hash"].removeprefix("sha256:")[:40]
    order_drift["answer_render_receipt"]["proof_id"] = order_proof["proof_id"]
    order_drift["answer_render_receipt"]["proof_hash"] = order_proof["proof_hash"]
    with pytest.raises(ValueError, match="does not match its source closure receipt"):
        validate_committable_answer(order_drift)
    order_drift_observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": order_drift["answer"],
        **project_proof_observation(order_drift),
    }
    order_drift_audit = answer_proof_evaluation(
        order_drift_observation,
        proof_authority=authority,
    )
    assert order_drift_audit is not None
    assert order_drift_audit["contract_valid"] is False
    assert "answer_proof_identity_invalid" not in order_drift_audit["reason_codes"]
    assert (
        "answer_proof_open_enumeration_source_closure_invalid" in order_drift_audit["reason_codes"]
    )

    selected_evidence_id = receipt["evidence_id"]
    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(
            authority,
            evidences=tuple(
                replace(item, canonical_text=item.canonical_text.replace("60%", "61%", 1))
                if item.evidence_id == selected_evidence_id
                else item
                for item in authority.evidences
            ),
        ),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in raw_drift["reason_codes"]


@pytest.mark.parametrize(
    "additional_contents",
    (
        (),
        ("补充：Beta 还支持功能乙。",),
    ),
)
def test_table_row_with_same_key_supplement_cannot_close_open_enumeration(
    additional_contents: tuple[str, ...],
) -> None:
    suffix = "\n补充：Beta 还支持功能乙。" if not additional_contents else ""
    state = _state(
        "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |" + suffix,
        original_query="Beta 有哪些功能？",
        additional_contents=additional_contents,
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    assert state["witness_catalog"]["source_enumeration_closure_receipts"] == []

    model = _AcceptingModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer_proof"]["overall_status"] == "accepted_partial"
    assert state["answer_proof"]["strict_eligible"] is False
    validate_committable_answer(state)


@pytest.mark.parametrize(
    ("content", "additional_contents"),
    (
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| 同上 | 功能乙 |",
            (),
        ),
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |",
            ("该型号（即用户所问型号）还支持功能乙。",),
        ),
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| Gamma | B**eta** 还支持功能乙 |",
            (),
        ),
        (
            "| 对象 | 功能 |\n| --- | --- |\n| Beta | 功能甲 |\n| Gamma | Be\u00adta 还支持功能乙 |",
            (),
        ),
    ),
)
def test_ambiguous_record_continuations_never_upgrade_open_enumeration_to_strict(
    content: str,
    additional_contents: tuple[str, ...],
) -> None:
    state = _state(
        content,
        original_query="Beta 有哪些功能？",
        additional_contents=additional_contents,
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    assert state["witness_catalog"]["source_enumeration_closure_receipts"] == []

    model = _AcceptingModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert state["answer_plan"]["decision"] in {"partial", "refuse"}
    expected_status = (
        "accepted_partial"
        if state["answer_plan"]["decision"] == "partial"
        else "accepted_abstention"
    )
    assert state["answer_proof"]["overall_status"] == expected_status
    assert state["answer_proof"]["strict_eligible"] is False
    validate_committable_answer(state)


def test_semantic_repair_back_edge_projects_replacement_without_reselecting() -> None:
    state = _state("初始候选证据。替代候选证据。")
    model = _RepairThenAcceptModel()
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=model)
    source_plan = copy.deepcopy(state["answer_plan"])
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)

    assert state["answer_repair_requested"] is True
    assert state["answer_repair_count"] == 0
    plan_answer(state, model_port=model)

    transition = state["answer_repair_transition"]
    assert model.select_calls == 1
    assert state["answer_repair_count"] == 1
    assert transition["schema_version"] == "answer-repair-transition.v1"
    assert transition["status"] == "applied"
    assert transition["repair_count"] == 1
    assert state["answer_repair_transitions"] == [transition]
    assert transition["source_plan_id"] == source_plan["plan_id"]
    assert transition["source_plan_hash"] == source_plan["plan_hash"]
    assert transition["resulting_plan_id"] == state["answer_plan"]["plan_id"]
    assert transition["resulting_plan_hash"] == state["answer_plan"]["plan_hash"]
    assert transition["source_review_logical_call_id"] == "qmc_" + f"{1:032x}"
    assert state["answer_plan"]["claims"][0]["witness_ids"] == [model.replacement_witness_id]

    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)

    assert model.review_inputs == [
        (model.initial_witness_id,),
        (model.replacement_witness_id,),
    ]
    assert model.initial_witness_id not in model.review_inputs[1]
    assert state["answer_plan_review"]["decision"] == "accept"
    assert state["answer_coverage_decision"]["status"] == "accepted"
    assert state["answer_repair_requested"] is False


def test_second_repair_is_reviewed_and_preserves_append_only_lineage() -> None:
    state = _state("初始候选证据。第一替代证据。第二替代证据。")
    model = _RepairTwiceThenAcceptModel()
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    assert state["answer_repair_requested"] is True

    plan_answer(state, model_port=model)
    assert state["answer_repair_count"] == 1
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    assert state["answer_repair_requested"] is True

    plan_answer(state, model_port=model)
    assert state["answer_repair_count"] == 2
    assert [item["repair_count"] for item in state["answer_repair_transitions"]] == [1, 2]
    assert state["answer_repair_transition"] == state["answer_repair_transitions"][-1]
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)

    assert model.select_calls == 1
    assert model.review_inputs == [
        (model.initial_witness_id,),
        (model.replacement_witness_id,),
        (model.final_witness_id,),
    ]
    assert state["answer_plan_review"]["decision"] == "accept"
    assert state["answer_coverage_decision"]["status"] == "accepted"
    assert state["answer_repair_requested"] is False


def test_second_repair_rejects_tampered_transition_lineage() -> None:
    state = _state("初始候选证据。第一替代证据。第二替代证据。")
    model = _RepairTwiceThenAcceptModel()
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    state["answer_repair_transitions"][0]["catalog_hash"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="transition history binding"):
        plan_answer(state, model_port=model)


def test_terminal_repair_limit_reason_survives_render_and_evaluator_replay() -> None:
    state = _state("初始候选证据。第一替代证据。第二替代证据。")
    model = _RepairTwiceThenRejectModel()
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=model)
    for _ in range(2):
        verify_answer_plan(state, model_port=model)
        decide_answer_coverage(state)
        assert state["answer_repair_requested"] is True
        plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)

    assert state["answer_coverage_decision"]["reason_code"] == "REPAIR_LIMIT_REACHED"
    assert state["answer_review_closure_receipt"]["decision"] == "reject"
    assert state["answer_review_closure_receipt"]["reason_code"] == "REPAIR_LIMIT_REACHED"
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    assert state["answer_proof"]["overall_status"] == "accepted_abstention"
    assert state["answer_proof"]["strict_eligible"] is False
    validate_committable_answer(state)
    audit = answer_proof_evaluation(
        {
            "observation_schema": OBSERVATION_SCHEMA_V3,
            "answer": state["answer"],
            **project_proof_observation(state),
        }
    )
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is False


def test_incomplete_accept_closure_converges_without_synthetic_repair() -> None:
    state = _state("启动设备前必须检查保护接地。")
    model = _IncompleteClosureAcceptingModel()
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)

    assert state["answer_coverage_decision"]["status"] == "accepted"
    assert state["answer_coverage_decision"]["reason_code"] == ("OBLIGATION_CLOSURE_UNPROVEN")
    assert state["answer_repair_requested"] is False
    assert state["answer_repair_feedback"] == {}
    assert state["answer_plan"]["decision"] == "refuse"
    assert state["answer_review_closure_receipt"]["schema_version"] == (
        "semantic-review-terminal.v1"
    )
    assert node_answer_coverage_after_router(state) == "node_answer_render"

    render_proven_answer(state, render_port=configured_answer_render_adapter)
    assert state["answer_proof"]["overall_status"] == "accepted_abstention"
    validate_committable_answer(state)


def test_ordered_steps_are_canonicalized_to_frozen_witness_order() -> None:
    state = _run_to_render(
        _state(
            "1、切断电源。\n2、检查接地。\n3、确认急停开关。",
            original_query="请按顺序列出三个操作步骤",
        ),
        _ReverseStepsModel(),
    )

    expected_witness_ids = state["witness_catalog"]["receipt"]["witness_ids"][:3]
    actual_witness_ids = [item["witness_ids"][0] for item in state["answer_plan"]["claims"]]
    assert actual_witness_ids == expected_witness_ids
    assert state["answer"].index("1、切断电源") < state["answer"].index("2、检查接地")
    assert state["answer"].index("2、检查接地") < state["answer"].index("3、确认急停")
    assert state["answer_proof"]["strict_eligible"] is True


def test_unknown_count_scoped_procedure_commits_ordered_partial_answer() -> None:
    state = _run_to_render(
        _state(
            "1、打开菜单。\n2、选择报告。\n3、确认打印。",
            original_query="如何通过控制面板打印状态报告？",
        ),
        _PartialStepsModel(),
    )

    obligation = state["question_contract"]["obligations"][0]
    assert obligation["closure"] == "open_enumeration"
    assert obligation["kind"] == "procedure"
    assert obligation["required_output_form"] == "ordered_steps"
    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer"].index("1、打开菜单") < state["answer"].index("2、选择报告")
    assert state["answer"].index("2、选择报告") < state["answer"].index("3、确认打印")
    assert state["answer_proof"]["overall_status"] == "accepted_partial"
    assert state["answer_proof"]["strict_eligible"] is False


def test_reviewer_reject_converges_to_committable_safe_abstention() -> None:
    content = "当前证据不能支持所选拒答。"
    state = _run_to_render(_state(content), _RejectingModel(refuse=True))
    called: list[bool] = []

    assert state["answer_proof"]["overall_status"] == "accepted_abstention"
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["answer_review_closure_receipt"]["schema_version"] == (
        "semantic-review-terminal.v1"
    )
    assert node_answer_render_after_router(state) == "node_answer_commit"
    commit_proven_answer(state, commit_port=lambda _state: called.append(True) or {})
    assert called == [True]
    audit = answer_proof_evaluation(
        {
            "observation_schema": OBSERVATION_SCHEMA_V3,
            "answer": state["answer"],
            **project_proof_observation(state),
        }
    )
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is False


def test_output_security_rewrite_converges_to_proven_safe_abstention(monkeypatch) -> None:
    state = _state("启动设备前必须检查保护接地。")
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    plan_answer(state, model_port=_AcceptingModel())
    verify_answer_plan(state, model_port=_AcceptingModel())
    decide_answer_coverage(state)

    def rewrite_output(target: dict[str, Any]) -> None:
        target["answer"] = "模拟的安全替换文本，不得直接进入 proof。"
        target["content_security"] = {
            "output": {"action": "redact", "policy_version": "test-policy.v1"}
        }

    monkeypatch.setattr(answer_rendering, "secure_answer_state", rewrite_output)
    render_proven_answer(state, render_port=configured_answer_render_adapter)

    assert "模拟的安全替换文本" not in state["answer"]
    assert state["answer_plan"]["decision"] == "refuse"
    assert state["answer_proof"]["overall_status"] == "accepted_abstention"
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["answer_review_closure_receipt"]["reason_code"] == ("OUTPUT_SECURITY_TRANSFORMED")
    assert state["citations"] == []
    validate_committable_answer(state)
    audit = answer_proof_evaluation(
        {
            "observation_schema": OBSERVATION_SCHEMA_V3,
            "answer": state["answer"],
            **project_proof_observation(state),
        }
    )
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is False


def test_committable_gate_rejects_missing_pending_citation_fact() -> None:
    state = _run_to_render(_state("启动设备前必须检查保护接地。"), _AcceptingModel())

    validate_committable_answer(state)
    state["pending_citation_facts"] = []

    with pytest.raises(AnswerProofNotCommittableError, match="citation identity graph"):
        validate_committable_answer(state)


def test_commit_gate_replays_text_from_witness_instead_of_trusting_self_reported_hashes() -> None:
    state = _run_to_render(_state("启动设备前必须检查保护接地。"), _AcceptingModel())
    question = QuestionContract.from_dict(state["question_contract"])
    plan = AnswerPlan.from_dict(state["answer_plan"])
    catalog_bundle = state["witness_catalog"]
    catalog = WitnessCatalogReceipt.from_dict(catalog_bundle["receipt"])
    witnesses = tuple(EvidenceWitness.from_dict(item) for item in catalog_bundle["witnesses"])
    witness_texts = {
        str(item["witness_id"]): str(item["content"])
        for item in resolve_witness_prompt_views(state, catalog_bundle)
    }
    original = AnswerProof.from_dict(state["answer_proof"])
    original_claim = original.claim_proofs[0]
    forged_quote = "这是完全伪造且不在证据中的内容。"
    forged_answer = f"{forged_quote} [E1]"
    forged_claim = ClaimProof.build(
        claim_id=original_claim.claim_id,
        obligation_id=original_claim.obligation_id,
        status=original_claim.status,
        method=original_claim.method,
        witness_ids=original_claim.witness_ids,
        answer_span=AnswerSpan(start=0, end=len(forged_answer)),
        rendered_text_hash=sha256_digest(forged_answer),
        relation=original_claim.relation,
        verifier=original_claim.verifier,
    )
    forged_proof = AnswerProof.build(
        answer_plan=plan,
        question_contract=question,
        catalog=catalog,
        witnesses=witnesses,
        witness_texts=witness_texts,
        semantic_review_closure=original.semantic_review_closure,
        answer_id=original.answer_id,
        answer_text=forged_answer,
        overall_status=original.overall_status,
        grounding_status=original.grounding_status,
        relevance_status=original.relevance_status,
        completeness_status=original.completeness_status,
        claim_proofs=(forged_claim,),
        obligation_proofs=original.obligation_proofs,
        verifier_versions=original.verifier_versions,
    )
    prior_citation = state["citations"][0]
    forged_fact = CitationFact(
        answer_id=state["answer_id"],
        evidence_id=prior_citation["evidence_id"],
        answer_span=AnswerSpan(start=0, end=len(forged_quote)),
        source_quote=forged_quote,
        quote_hash=sha256_digest(forged_quote),
        source_label=prior_citation["source_label"],
        validation_status="valid",
        ordinal=1,
    )
    forged_public_citation = forged_fact.model_dump(mode="json", exclude_none=True)
    forged_public_citation.update(
        {
            "claim_id": plan.substantive_claims[0].claim_id,
            "obligation_id": plan.substantive_claims[0].obligation_id,
            "witness_id": plan.substantive_claims[0].witness_ids[0],
            "claim_text": forged_quote,
        }
    )
    state["answer"] = forged_answer
    state["answer_proof"] = forged_proof.to_dict()
    state["citations"] = [forged_public_citation]
    state["pending_citation_facts"] = [forged_fact.model_dump(mode="json", exclude_none=True)]
    state["answer_envelope"] = configured_answer_render_adapter.build_answer_envelope(
        state
    ).model_dump(mode="json")
    state["answer_render_receipt"] = {
        "schema_version": "answer-render-receipt.v3",
        "status": "rendered",
        "answer_id": state["answer_id"],
        "query_run_id": state["query_run_id"],
        "answer_sha256": sha256_digest(forged_answer),
        "proof_id": forged_proof.proof_id,
        "proof_hash": forged_proof.proof_hash,
        "claim_count": 1,
        "citation_count": 1,
        "citation_fact_bundle_hash": citation_fact_bundle_hash((forged_fact,)),
    }

    with pytest.raises(AnswerProofNotCommittableError, match="not derived"):
        validate_committable_answer(state)


def test_commit_gate_rejects_semantic_closure_without_matching_review_call() -> None:
    state = _run_to_render(_state("启动设备前必须检查保护接地。"), _AcceptingModel())
    state["answer_prompt_receipts"][-1]["parsed_proposal"]["closed_obligation_ids"] = []

    with pytest.raises(AnswerProofNotCommittableError, match="semantic review closure"):
        validate_committable_answer(state)


def test_v3_evaluation_replays_semantic_closure_and_item_graph() -> None:
    state = _run_to_render(_state("启动设备前必须检查保护接地。"), _AcceptingModel())
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }

    audit = answer_proof_evaluation(observation)

    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    tampered = copy.deepcopy(observation)
    tampered["answer_proof"]["obligation_proofs"][0]["proven_item_count"] = 2
    tampered_audit = answer_proof_evaluation(tampered)
    assert tampered_audit is not None
    assert tampered_audit["contract_valid"] is False
    assert tampered_audit["strict_graph_valid"] is False
    assert "answer_proof_item_count_invalid" in tampered_audit["reason_codes"]


@pytest.mark.parametrize(
    "text",
    (
        "3.14 is the documented threshold.",
        "1.2.3 is the supported version.",
        "2026.08 is the release tag.",
        "3．14 是阈值。",
        "１．２ 是版本号。",
        "1、2号设备均支持该功能。",
        "1、2两项必须同时满足。",
    ),
)
def test_decimal_and_version_lines_are_not_list_items(text: str) -> None:
    assert top_level_list_item_body_span(text) is None
    atoms = atomize_evidence_text(text)
    assert atoms
    assert all(item.kind == "sentence" for item in atoms)


def test_english_sentences_are_atomized_independently() -> None:
    text = "First supported fact. Second supported fact."

    atoms = atomize_evidence_text(text)

    assert [item.resolve(text) for item in atoms] == [
        "First supported fact.",
        "Second supported fact.",
    ]


def test_explicit_multi_step_question_freezes_ordered_steps_contract() -> None:
    kind, output_form = _obligation_shape(
        "请按顺序列出三个操作步骤",
        FacetClosure.CLOSED_COUNT,
    )

    assert kind.value == "procedure"
    assert output_form.value == "ordered_steps"


def test_markdown_table_only_counts_verified_data_rows() -> None:
    text = "| 名称 | 值 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"

    atoms = atomize_evidence_text(text)

    assert [item.kind for item in atoms] == [
        "table_row",
        "table_row",
        "table_data_row",
        "table_data_row",
    ]
    assert [item.resolve(text) for item in atoms] == text.splitlines()
    data_span = markdown_table_data_row_body_span(atoms[2].resolve(text))
    assert data_span is not None
    assert atoms[2].resolve(text)[data_span[0] : data_span[1]] == "A | 1"


@pytest.mark.parametrize(
    "text",
    [
        "value | other",
        "`value | other`",
        r"value \| other",
        "| 名称 | 值 |\n| A | 1 |",
    ],
)
def test_pipe_text_without_complete_markdown_table_is_not_countable(text: str) -> None:
    atoms = atomize_evidence_text(text)

    assert all(item.kind != "table_data_row" for item in atoms)


def test_witness_limit_is_explicitly_recorded_as_truncated() -> None:
    content = "".join(f"合成检查项{i}。" for i in range(MAX_WITNESS_CATALOG_ITEMS + 12))
    state = _state(content)
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipt = WitnessCatalogReceipt.from_dict(state["witness_catalog"]["receipt"])

    assert receipt.witness_count == MAX_WITNESS_CATALOG_ITEMS
    assert receipt.universe_status.value == "truncated"
    omissions = {item.code: item.count for item in receipt.omissions}
    assert omissions[WitnessOmissionCode.ATOM_LIMIT] == 12


def test_witness_mapping_uses_monotonic_receipt_span_cursor() -> None:
    canonical = "&amp; 项。" * 1_000
    derivation = safe_representation.derive_safe_representation(canonical)
    atoms = atomize_evidence_text(derivation.safe_representation)

    class CountingSpans:
        def __init__(self, values) -> None:
            self.values = tuple(values)
            self.accesses = 0

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, item):
            if isinstance(item, slice):
                start, stop, step = item.indices(len(self.values))
                self.accesses += len(range(start, stop, step))
            else:
                self.accesses += 1
            return self.values[item]

    spans = CountingSpans(derivation.receipt.spans)
    cursor = answer_builders._SafeSpanCursor(spans)  # type: ignore[arg-type]
    for atom in atoms:
        mapped = answer_builders._map_safe_atom(
            receipt=derivation.receipt,
            safe_start=atom.start,
            safe_end=atom.end,
            span_cursor=cursor,
        )
        assert mapped is not None

    assert spans.accesses <= 12 * (len(spans) + len(atoms))


def test_witness_limit_short_circuits_expensive_span_ref_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_from_texts = EvidenceSpanRef.from_texts.__func__
    calls = 0

    def counted_from_texts(cls, **kwargs):
        nonlocal calls
        calls += 1
        return original_from_texts(cls, **kwargs)

    monkeypatch.setattr(
        EvidenceSpanRef,
        "from_texts",
        classmethod(counted_from_texts),
    )
    state = _state("".join(f"合成检查项{i}。" for i in range(MAX_WITNESS_CATALOG_ITEMS + 12)))
    freeze_question_contract(state)
    freeze_witness_catalog(state)

    assert calls == MAX_WITNESS_CATALOG_ITEMS


def test_truncated_catalog_cannot_produce_answered_or_strict_proof() -> None:
    content = "".join(f"合成检查项{i}。" for i in range(MAX_WITNESS_CATALOG_ITEMS + 12))

    state = _run_to_render(_state(content), _AcceptingModel())

    assert state["witness_catalog"]["receipt"]["universe_status"] == "truncated"
    assert state["answer_plan"]["decision"] == "partial"
    assert state["answer_proof"]["overall_status"] == "accepted_partial"
    assert state["answer_proof"]["strict_eligible"] is False
    assert state["answer_proof"]["obligation_proofs"][0]["outcome"] == ("partially_answered")


def test_provider_json_parser_rejects_duplicate_keys_and_terminal_repair() -> None:
    with pytest.raises(ProposalValidationError):
        parse_answer_plan_proposal_json(
            '{"decision":"refuse","decision":"answer",'
            '"claims":[],"reason_codes":["NO_SUPPORTING_WITNESS"]}'
        )

    repair = (
        '{"decision":"repair","claims":[{'
        '"obligation_id":"qf_11111111111111111111111111111111",'
        '"witness_id":"wit_2222222222222222222222222222222222222222",'
        '"rendering_rule":"exact_witness"}],'
        '"reason_codes":["WITNESS_NOT_SUPPORTIVE"]}'
    )
    intermediate = parse_review_proposal_json(repair, repair_attempt=1)
    assert intermediate.decision == ReviewDecision.REPAIR

    terminal = parse_review_proposal_json(repair, repair_attempt=2)
    assert terminal.decision == ReviewDecision.REJECT
    assert terminal.reason_codes == (ReviewReasonCode.REPAIR_LIMIT_REACHED,)


def test_runtime_replay_restores_and_revalidates_final_proof_checkpoint(monkeypatch) -> None:
    from types import SimpleNamespace

    import app.api.http.query_runtime_routes as runtime_routes
    from app.application.query_cache_service import QueryFingerprint, QueryReplayManifest
    from app.domain.query_checkpoint import QueryExecutionStatus
    from app.foundation.query_checkpoint_crypto import (
        QueryCheckpointCipher,
        QueryCheckpointEncryptionContext,
    )
    from app.security.principal import Principal

    state = _run_to_render(
        _state("启动设备前必须检查保护接地。"),
        _AcceptingModel(),
    )
    sequence = 9
    metadata = {
        "checkpoint_namespace": "query-runtime-proof-v2",
        "workflow_version": "query-workflow.test.v2",
        "state_schema_version": "query-state.v3",
        "checkpoint_schema_version": "query-checkpoint.v2",
    }
    context = QueryCheckpointEncryptionContext(
        tenant_id="tenant-test",
        user_id="user-test",
        query_run_id=state["query_run_id"],
        sequence=sequence,
        completed_node="answer_commit",
        next_node=None,
        **metadata,
    )
    cipher = QueryCheckpointCipher(b"r" * 32)
    encrypted = cipher.encrypt_state(state, context=context)
    checkpoint = SimpleNamespace(
        query_run_id=state["query_run_id"],
        sequence=sequence,
        completed_node="answer_commit",
        next_node=None,
        nonce=encrypted.nonce,
        ciphertext=encrypted.ciphertext,
        ciphertext_sha256=encrypted.ciphertext_sha256,
        size_bytes=encrypted.size_bytes,
        **metadata,
    )
    execution = SimpleNamespace(
        status=QueryExecutionStatus.COMPLETED,
        next_node=None,
        checkpoint_sequence=sequence,
        **metadata,
    )

    class _CheckpointRepository:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_execution(self, _query_run_id):
            return execution

        def latest_checkpoint(self, _query_run_id):
            return checkpoint

    principal = Principal(
        user_id="user-test",
        tenant_id="tenant-test",
        roles=("TenantAdmin",),
        scopes=(),
        auth_type="password",
        credential_id="credential-test",
        request_id="request-test",
    )
    monkeypatch.setattr(runtime_routes, "QueryCheckpointRepository", _CheckpointRepository)
    monkeypatch.setattr(runtime_routes, "_checkpoint_cipher", lambda: cipher)
    restored, binding = runtime_routes._load_final_proof_checkpoint(
        session=object(),
        principal=principal,
        query_run_id=state["query_run_id"],
    )
    assert restored["answer_proof"] == state["answer_proof"]
    assert restored["question_contract"] == state["question_contract"]
    assert restored["answer_plan"] == state["answer_plan"]
    assert restored["witness_catalog"] == state["witness_catalog"]
    assert restored["context_aliases"] == state["context_aliases"]
    assert restored["answer_review_closure_receipt"] == (state["answer_review_closure_receipt"])
    assert restored["answer_prompt_receipts"] == state["answer_prompt_receipts"]
    assert restored["answer_render_receipt"] == state["answer_render_receipt"]
    assert restored["citations"] == state["citations"]

    fingerprint = QueryFingerprint.from_request(
        tenant_id="tenant-test",
        principal_scope=["TenantAdmin"],
        normalized_query="设备启动前需要检查什么？",
        conversation_history=[],
        authorized_scope={"kb_ids": ["kb-test"]},
        filters={},
        index_versions=["idx-test-v1"],
        query_profile_version="rpf-test@1",
        planner_version="planner-test.v1",
        embedding_model_version="embedding-test.v1",
        rerank_model_version="rerank-test.v1",
        generator_model_version="generator-test.v1",
        prompt_version="prompt-test.v1",
        policy_version="policy-test.v1",
        web_snapshot_policy="disabled",
    )
    public_state = runtime_routes._public_result(state)
    saved_manifests: list[QueryReplayManifest] = []

    class _ReliabilityRepository:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def save_replay_manifest(self, manifest, *, ttl) -> None:
            assert ttl.total_seconds() > 0
            saved_manifests.append(manifest)

    monkeypatch.setattr(runtime_routes, "QueryReliabilityRepository", _ReliabilityRepository)
    monkeypatch.setattr(runtime_routes, "_runtime_repository", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime_routes,
        "_event_stream_hash",
        lambda *_args, **_kwargs: runtime_routes.canonical_hash([]),
    )
    runtime_routes._save_replay_manifest(
        session=object(),
        principal=principal,
        run=SimpleNamespace(query_run_id=state["query_run_id"], turn_id="turn-test"),
        history=SimpleNamespace(content_hash=runtime_routes.canonical_hash([])),
        fingerprint=fingerprint,
        result=public_state,
    )
    assert len(saved_manifests) == 1
    manifest = saved_manifests[0]
    assert runtime_routes._ReplayProofCheckpointBinding.from_manifest(manifest) == binding
    replayed = runtime_routes._restore_replay_proof_state(
        session=object(),
        principal=principal,
        manifest=manifest,
    )
    assert replayed["answer_envelope"] == state["answer_envelope"]

    tampered = manifest.model_copy(
        update={"answer_envelope": {**manifest.answer_envelope, "answer": "篡改"}}
    )
    with pytest.raises(ValueError, match="answer envelope differs"):
        runtime_routes._restore_replay_proof_state(
            session=object(),
            principal=principal,
            manifest=tampered,
        )

    tampered_binding = runtime_routes._ReplayProofCheckpointBinding(
        query_run_id=binding.query_run_id,
        sequence=binding.sequence,
        ciphertext_sha256=runtime_routes.canonical_hash("tampered-checkpoint"),
        proof_projection_sha256=binding.proof_projection_sha256,
    )
    tampered_refs = tuple(
        tampered_binding.to_ref()
        if item.startswith(runtime_routes._REPLAY_PROOF_CHECKPOINT_REF_PREFIX)
        else item
        for item in manifest.route_result_refs
    )
    with pytest.raises(ValueError, match="binding is stale or tampered"):
        runtime_routes._restore_replay_proof_state(
            session=object(),
            principal=principal,
            manifest=manifest.model_copy(update={"route_result_refs": tampered_refs}),
        )


def test_runtime_replay_rejects_missing_or_tampered_proof_binding() -> None:
    import app.api.http.query_runtime_routes as runtime_routes
    from app.application.query_cache_service import QueryFingerprint, QueryReplayManifest

    state = _run_to_render(
        _state("启动设备前必须检查保护接地。"),
        _AcceptingModel(),
    )
    missing_proof = dict(state)
    missing_proof.pop("answer_proof")
    with pytest.raises(AnswerProofNotCommittableError, match="missing proof state"):
        runtime_routes._replay_proof_projection(missing_proof)

    tampered_proof = dict(state)
    tampered_proof["answer_proof"] = {
        **state["answer_proof"],
        "answer_id": "ans_tampered",
    }
    with pytest.raises((AnswerProofNotCommittableError, ValueError)):
        runtime_routes._replay_proof_projection(tampered_proof)

    fingerprint = QueryFingerprint.from_request(
        tenant_id="tenant-test",
        principal_scope=[],
        normalized_query="test",
        conversation_history=[],
        authorized_scope={},
        filters={},
        index_versions=[],
        query_profile_version="test",
        planner_version="test",
        embedding_model_version="test",
        rerank_model_version="test",
        generator_model_version="test",
        prompt_version="test",
        policy_version="test",
        web_snapshot_policy="disabled",
    )
    manifest = QueryReplayManifest(
        schema_version=runtime_routes._REPLAY_MANIFEST_SCHEMA_VERSION,
        query_run_id=state["query_run_id"],
        fingerprint=fingerprint,
        request_hash=runtime_routes.canonical_hash({}),
        query_plan=state["query_plan"],
        route_result_refs=(runtime_routes.canonical_hash({"route": "only"}),),
        answer_envelope=state["answer_envelope"],
        event_stream_hash=runtime_routes.canonical_hash([]),
    )
    with pytest.raises(ValueError, match="exactly one proof checkpoint"):
        runtime_routes._ReplayProofCheckpointBinding.from_manifest(manifest)
    with pytest.raises(ValueError, match="schema version is unsupported"):
        runtime_routes._ReplayProofCheckpointBinding.from_manifest(
            manifest.model_copy(update={"schema_version": "query-replay/1"})
        )


def test_scoped_procedure_closure_replays_from_complete_multi_representation_authority() -> None:
    procedure = (
        "标题路径：设置 > 打印产品报告步骤\n\n"
        "1. 打开控制面板。\n"
        "2. 选择打印信息报告。\n"
        "3. 按下确定键。\n\n"
        "完成后返回主页。"
    )
    state = _state(
        procedure,
        original_query="请按顺序说明打印产品报告步骤",
        additional_contents=("标题路径：附录 > 网络术语\n\nSSID 是无线网络名称。",),
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-scoped-procedure-list-closure-receipt.v4")
    assert receipt["catalog_representation_count"] == 2
    assert receipt["authority_candidate_count"] == 1

    model = _SourceClosedProcedureModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    assert state["answer_proof"]["strict_eligible"] is True

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    assert all(item.evidence_id is not None for item in authority.representations)
    without_authority = answer_proof_evaluation(observation)
    assert without_authority is not None
    assert without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in without_authority["reason_codes"]

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    selected_evidence_id = receipt["evidence_id"]
    drifted_evidences = tuple(
        replace(item, canonical_text=item.canonical_text.replace("打开", "关闭", 1))
        if item.evidence_id == selected_evidence_id
        else item
        for item in authority.evidences
    )
    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(authority, evidences=drifted_evidences),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in raw_drift["reason_codes"]

    missing_representation = answer_proof_evaluation(
        observation,
        proof_authority=replace(authority, representations=authority.representations[:-1]),
    )
    assert missing_representation is not None
    assert missing_representation["contract_valid"] is False


def test_scoped_procedure_suffix_with_interstitial_note_replays_strictly() -> None:
    procedure = (
        "标题路径：维护 > 清除卡纸步骤\n"
        "1. 打开设备前盖。\n"
        "2. 检查纸张所在位置。\n"
        "3. 如果纸张可见，请取出卡纸。\n"
        "注：确认取出的纸张保持完整。\n"
        "4. 重新装入纸张并关闭前盖。"
    )
    query = "如果纸张可见，取出卡纸后还要执行哪些清除卡纸步骤？"
    state = _state(
        procedure,
        original_query=query,
        additional_contents=("标题路径：附录 > 网络术语\nSSID 是无线网络名称。",),
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-scoped-procedure-list-closure-receipt.v4")
    assert receipt["selection_kind"] == "query_selected_continuation"
    assert receipt["item_ordinals"] == [3, 4]
    assert receipt["witness_ordinals"] == sorted(set(receipt["witness_ordinals"]))
    assert receipt["witness_ordinals"][1] > receipt["witness_ordinals"][0] + 1
    assert receipt["anchor_match_key_hashes"]

    model = _SourceClosedProcedureModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    audit = answer_proof_evaluation(
        observation,
        proof_authority=build_proof_evaluation_authority(state),
    )

    assert state["answer_proof"]["strict_eligible"] is True
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True


def test_scoped_procedure_prefix_v4_replays_and_rejects_boundary_tamper() -> None:
    procedure = (
        "标题路径：安装 > USB安装\n"
        "1.使用USB线连接打印机和计算机并打开电源。\n"
        "2.插入安装光盘并运行Autorun.exe安装程序。\n"
        "3.阅读许可协议、选择安装语言并点击安装。\n"
        "4.系统自动安装驱动。\n"
        "5.安装完成后关闭安装界面。"
    )
    query = "使用USB线连接Acme Z9000打印机时，安装驱动前需完成哪些操作？"
    state = _state(procedure, original_query=query)

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipt = state["witness_catalog"]["source_enumeration_closure_receipts"][0]
    assert receipt["schema_version"] == ("query-anchored-scoped-procedure-list-closure-receipt.v4")
    assert receipt["selection_kind"] == "query_selected_prefix"
    assert receipt["item_ordinals"] == [1, 2, 3]
    assert receipt["excluded_successor_ordinal"] == 4
    assert receipt["boundary_rule_version"] == "explicit-query-before-successor.v1"

    model = _SourceClosedProcedureModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True, audit["reason_codes"]
    assert audit["strict_graph_valid"] is True

    tampered = copy.deepcopy(observation)
    tampered["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "excluded_successor_hash"
    ] = "sha256:" + "0" * 64
    tampered_audit = answer_proof_evaluation(tampered, proof_authority=authority)
    assert tampered_audit is not None
    assert tampered_audit["contract_valid"] is False
    assert (
        "answer_proof_open_enumeration_source_closure_invalid" in (tampered_audit["reason_codes"])
    )


def test_inline_action_chain_builder_replays_typed_receipt_through_prompt_views() -> None:
    from app.domain.answering.enumeration_closure import (
        QueryAnchoredInlineActionChainClosureReceipt,
    )
    from app.infrastructure.answering.provider import _validated_source_closed_witness_ids

    state = _state(
        "标题路径：设置 > 打开帮助文档\n\n"
        "您可以打开“打印首选项”，点击“帮助”按钮（仅适用于 Windows 系统）。",
        original_query="在Windows系统中如何通过打印首选项打开帮助文档？请说明具体操作路径。",
    )

    freeze_question_contract(state)
    freeze_witness_catalog(state)

    raw_receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(raw_receipts) == 1
    assert raw_receipts[0]["schema_version"] == (
        "query-anchored-inline-action-chain-closure-receipt.v1"
    )
    assert raw_receipts[0]["action_count"] == 2
    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    owner_views = [view for view in views if view["source_enumeration_closure_receipts"]]
    assert len(owner_views) == 1
    typed_receipt = owner_views[0]["source_enumeration_closure_receipts"][0]
    assert isinstance(typed_receipt, QueryAnchoredInlineActionChainClosureReceipt)
    assert _validated_source_closed_witness_ids(views) == {
        typed_receipt.obligation_id: typed_receipt.witness_ids
    }


def test_scoped_bullet_list_closes_all_members_with_raw_replay() -> None:
    bullet_list = (
        "标题路径：前言 > 10. 打印故障\n"
        "在处理打印故障之前，请检查以下情况：\n"
        "• 电源连接是否正常。\n"
        "• 接口电缆是否连接正确。\n"
        "• 驱动程序是否安装正确。"
    )
    state = _state(
        bullet_list,
        original_query="在处理打印故障前应检查哪些关键项目，请列出全部内容",
        additional_contents=("标题路径：附录 > 网络术语\nSSID 表示无线网络名称。",),
    )

    freeze_question_contract(state)
    assert state["question_contract"]["obligations"][0]["closure"] == "open_enumeration"
    assert state["question_contract"]["obligations"][0]["required_output_form"] == "list"
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-scoped-bullet-list-closure-receipt.v1")
    assert receipt["item_count"] == 3
    assert receipt["catalog_representation_count"] == 2

    views = resolve_witness_prompt_views(state, state["witness_catalog"])
    assert (
        sum(receipt["obligation_id"] in item["source_closed_open_obligation_ids"] for item in views)
        == 3
    )
    assert answering_provider._validated_source_closed_witness_ids(views) == {
        receipt["obligation_id"]: tuple(receipt["witness_ids"])
    }

    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    assert state["answer_plan"]["decision"] == "answer"
    assert state["answer_proof"]["overall_status"] == "accepted"
    assert state["answer_proof"]["strict_eligible"] is True

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    without_authority = answer_proof_evaluation(observation)
    assert without_authority is not None
    assert without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in without_authority["reason_codes"]

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    selected_evidence_id = receipt["evidence_id"]
    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(
            authority,
            evidences=tuple(
                replace(
                    item,
                    canonical_text=item.canonical_text.replace("电源连接", "电池连接", 1),
                )
                if item.evidence_id == selected_evidence_id
                else item
                for item in authority.evidences
            ),
        ),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in raw_drift["reason_codes"]


def test_scoped_fact_set_closes_inline_members_with_raw_replay() -> None:
    sentence = "Orion、Orion Cloud 和 Orion Link 是远航设备有限公司注册的商标。"
    state = _state(
        f"标题路径：产品手册 > 法律说明\n\n{sentence}",
        original_query="远航设备有限公司注册的商标包括哪些？",
        additional_contents=("标题路径：附录 > 网络术语\nSSID 表示无线网络名称。",),
    )

    freeze_question_contract(state)
    assert state["question_contract"]["obligations"][0]["closure"] == "open_enumeration"
    assert state["question_contract"]["obligations"][0]["required_output_form"] == "list"
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == ("query-anchored-scoped-fact-set-closure-receipt.v1")
    assert receipt["mode"] == "closed_inline_coordination"
    assert receipt["member_count"] == 1
    assert receipt["semantic_count"] == 3

    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    assert state["answer_proof"]["strict_eligible"] is True

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    selected_evidence_id = receipt["evidence_id"]
    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(
            authority,
            evidences=tuple(
                replace(
                    item,
                    canonical_text=item.canonical_text.replace("Orion Link", "Orion Sync", 1),
                )
                if item.evidence_id == selected_evidence_id
                else item
                for item in authority.evidences
            ),
        ),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in raw_drift["reason_codes"]

    field_tamper = copy.deepcopy(observation)
    field_tamper["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "supplement_candidate_count"
    ] = 1
    field_tamper_audit = answer_proof_evaluation(
        field_tamper,
        proof_authority=authority,
    )
    assert field_tamper_audit is not None
    assert field_tamper_audit["contract_valid"] is False
    assert (
        "answer_proof_open_enumeration_source_closure_invalid" in field_tamper_audit["reason_codes"]
    )

    extra_field_tamper = copy.deepcopy(observation)
    extra_field_tamper["answer_proof"]["obligation_proofs"][0][
        "source_enumeration_closure_receipt"
    ]["raw_authority_text"] = sentence
    extra_field_audit = answer_proof_evaluation(
        extra_field_tamper,
        proof_authority=authority,
    )
    assert extra_field_audit is not None
    assert extra_field_audit["contract_valid"] is False
    assert (
        "answer_proof_open_enumeration_source_closure_invalid" in extra_field_audit["reason_codes"]
    )


def test_scoped_bullet_lead_fact_set_replays_and_requires_complete_title_boundary() -> None:
    checklist = (
        "标题路径：前言 > 4.如果在网络中使用打印机时出现问题，请检查以下几方面：\n\n"
        "• 关闭计算机、接入点（无线路由器）和打印机的电源，然后再重启。\n\n"
        "• 检查防火墙软件是否封锁通信。如果计算机和打印机连接在同一网络中却搜索不到，"
        "可能是因为防火墙软件封锁了通信。请参阅防火墙软件的用户指南，将防火墙关闭，"
        "然后重新尝试搜索打印机。\n\n"
        "• 检查打印机 IP 地址的分配是否正确。您可以打印打印机的网络配置信息页来检查 IP 地址。"
    )
    query = "当打印机在网络中无法被搜索到时，应检查哪几项内容？请列出所有相关检查点。"
    state = _state(checklist, original_query=query)

    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipt = state["witness_catalog"]["source_enumeration_closure_receipts"][0]
    assert receipt["mode"] == "scoped_bullet_lead_clause_set"
    assert receipt["member_count"] == 3
    assert receipt["topic_group_member_counts"] == [3]
    assert receipt["member_atom_kinds"] == ["list_item"] * 3
    assert receipt["segmentation_methods"] == ["top_level_list_entry.v1"] * 3

    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)

    audit = answer_proof_evaluation(observation, proof_authority=authority)
    assert audit is not None
    assert audit["contract_valid"] is True
    assert audit["strict_graph_valid"] is True

    tampered = copy.deepcopy(observation)
    tampered["answer_proof"]["obligation_proofs"][0]["source_enumeration_closure_receipt"][
        "canonical_title_hash"
    ] = None
    tampered_audit = answer_proof_evaluation(tampered, proof_authority=authority)
    assert tampered_audit is not None
    assert tampered_audit["contract_valid"] is False
    assert (
        "answer_proof_open_enumeration_source_closure_invalid" in (tampered_audit["reason_codes"])
    )


@pytest.mark.parametrize(
    ("content", "query", "expected_action_count", "has_terminal_result"),
    [
        (
            "标题路径：设置 > 打开帮助文档\n\n"
            "您可以打开“打印首选项”，点击“帮助”按钮（仅适用于 Windows 系统）。",
            "在Windows系统中如何通过打印首选项打开帮助文档？请说明具体操作路径。",
            2,
            False,
        ),
        (
            "标题路径：前言 > 中部卡纸\n\n"
            "4. 取出卡纸后，重新装入激光碳粉盒，合上前盖，打印机将自动恢复打印。",
            "取出卡纸后又需完成哪些操作才能恢复打印？",
            2,
            True,
        ),
        (
            "标题路径：前言 > 双面打印单元卡纸\n\n"
            "4. 取出卡纸后，装回双面打印单元，检查产品其他部位，"
            "确保无卡纸后，合上后盖，打印机将自动恢复打印。",
            "取出卡纸后还需进行哪些检查和操作？",
            3,
            True,
        ),
    ],
)
def test_inline_action_chain_proof_replays_each_action_and_excludes_terminal_result(
    content: str,
    query: str,
    expected_action_count: int,
    has_terminal_result: bool,
) -> None:
    state = _state(content, original_query=query)
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    receipts = state["witness_catalog"]["source_enumeration_closure_receipts"]
    inline_receipts = [
        item
        for item in receipts
        if item.get("schema_version") == "query-anchored-inline-action-chain-closure-receipt.v1"
    ]
    assert len(inline_receipts) == 1
    receipt = inline_receipts[0]
    assert len(receipt) == 61
    assert receipt["action_count"] == expected_action_count
    assert (receipt["canonical_terminal_result_hash"] is not None) is has_terminal_result

    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)

    obligation_proof = next(
        item
        for item in state["answer_proof"]["obligation_proofs"]
        if item["obligation_id"] == receipt["obligation_id"]
    )
    proven_items = obligation_proof["proven_items"]
    expected_spans = [
        (
            start - receipt["packed_witness_start"],
            end - receipt["packed_witness_start"],
        )
        for _ordinal, start, end in sorted(
            zip(
                receipt["action_ordinals"],
                receipt["packed_action_starts"],
                receipt["packed_action_ends"],
                strict=True,
            )
        )
    ]
    assert obligation_proof["proven_item_count"] == expected_action_count
    assert [item["segmentation_method"] for item in proven_items] == [
        "inline_action_span.v1"
    ] * expected_action_count
    assert [
        (item["relative_start"], item["relative_end"]) for item in proven_items
    ] == expected_spans
    assert [item["raw_span_hash"] for item in proven_items] == receipt["packed_action_hashes"]
    if has_terminal_result:
        terminal_span = (
            receipt["packed_terminal_result_start"] - receipt["packed_witness_start"],
            receipt["packed_terminal_result_end"] - receipt["packed_witness_start"],
        )
        assert terminal_span not in expected_spans
        assert all(end <= terminal_span[0] for _start, end in expected_spans)

    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    without_authority = answer_proof_evaluation(observation)
    assert without_authority is not None
    assert without_authority["strict_graph_valid"] is False
    assert "answer_proof_trusted_authority_missing" in without_authority["reason_codes"]

    audit = answer_proof_evaluation(
        observation,
        proof_authority=build_proof_evaluation_authority(state),
    )
    assert audit is not None
    assert audit["contract_valid"] is True, audit["reason_codes"]
    assert audit["strict_graph_valid"] is True


def test_inline_action_chain_evaluator_rejects_item_receipt_and_raw_authority_drift() -> None:
    state = _state(
        "标题路径：前言 > 双面打印单元卡纸\n\n"
        "4. 取出卡纸后，装回双面打印单元，检查产品其他部位，"
        "确保无卡纸后，合上后盖，打印机将自动恢复打印。",
        original_query="取出卡纸后还需进行哪些检查和操作？",
    )
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    model = _SourceClosedMembersModel()
    plan_answer(state, model_port=model)
    verify_answer_plan(state, model_port=model)
    decide_answer_coverage(state)
    render_proven_answer(state, render_port=configured_answer_render_adapter)
    validate_committable_answer(state)
    observation = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer": state["answer"],
        **project_proof_observation(state),
    }
    authority = build_proof_evaluation_authority(state)
    obligation = observation["answer_proof"]["obligation_proofs"][0]

    for mutate in (
        lambda value: value["proven_items"].reverse(),
        lambda value: value["proven_items"][0].__setitem__(
            "relative_end",
            value["proven_items"][0]["relative_end"] + 1,
        ),
        lambda value: value["source_enumeration_closure_receipt"].__setitem__(
            "packed_action_hashes",
            ["sha256:" + "0" * 64]
            + value["source_enumeration_closure_receipt"]["packed_action_hashes"][1:],
        ),
        lambda value: value["source_enumeration_closure_receipt"].__setitem__(
            "unexpected_authority_text",
            "not allowed",
        ),
    ):
        tampered = copy.deepcopy(observation)
        mutate(tampered["answer_proof"]["obligation_proofs"][0])
        audit = answer_proof_evaluation(tampered, proof_authority=authority)
        assert audit is not None
        assert audit["contract_valid"] is False
        assert "answer_proof_open_enumeration_source_closure_invalid" in audit["reason_codes"]

    receipt = obligation["source_enumeration_closure_receipt"]
    selected_evidence_id = receipt["evidence_id"]
    raw_drift = answer_proof_evaluation(
        observation,
        proof_authority=replace(
            authority,
            evidences=tuple(
                replace(
                    item,
                    canonical_text=item.canonical_text.replace("装回", "取下", 1),
                )
                if item.evidence_id == selected_evidence_id
                else item
                for item in authority.evidences
            ),
        ),
    )
    assert raw_drift is not None
    assert raw_drift["contract_valid"] is False
    assert "answer_proof_open_enumeration_source_closure_invalid" in raw_drift["reason_codes"]


def test_inline_action_chain_tied_candidates_never_issue_a_source_closure() -> None:
    candidate = "标题路径：维护 > 卡纸恢复操作\n\n取出卡纸后，重新装入组件，合上前盖。"
    state = _state(
        candidate,
        original_query="取出卡纸后要完成哪些卡纸恢复操作？",
        additional_contents=(candidate,),
    )
    freeze_question_contract(state)
    freeze_witness_catalog(state)
    assert all(
        item.get("schema_version") != "query-anchored-inline-action-chain-closure-receipt.v1"
        for item in state["witness_catalog"]["source_enumeration_closure_receipts"]
    )
