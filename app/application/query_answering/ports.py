"""Proof-Carrying Answer 应用层依赖端口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.application.query_answering.proposals import (
    AnswerPlanProposal,
    ReviewProposal,
)
from app.domain.answering.question import QuestionContract
from app.domain.evidence import CitationFact


class AnsweringModelPort(Protocol):
    """受限 proposal 与语义复核模型端口。"""

    plan_prompt_version: str
    review_prompt_version: str

    def select_plan(
        self,
        state: dict[str, Any],
        *,
        question_contract: QuestionContract,
        witness_catalog: Sequence[Mapping[str, Any]],
    ) -> AnswerPlanProposal: ...

    def review_plan(
        self,
        state: dict[str, Any],
        *,
        question_contract: QuestionContract,
        witness_catalog: Sequence[Mapping[str, Any]],
        plan_proposal: AnswerPlanProposal,
        repair_attempt: int,
    ) -> ReviewProposal: ...


class AnswerRenderPort(Protocol):
    """证据复核、引用持久化与公开投影端口。"""

    def new_answer_id(self) -> str: ...

    def new_query_run_id(self) -> str: ...

    def revalidate_cited_evidence(
        self,
        state: Mapping[str, Any],
        evidence_ids: set[str],
    ) -> dict[str, dict[str, Any]]: ...

    def persist_verified_citations(
        self,
        state: Mapping[str, Any],
        *,
        evidences: Mapping[str, dict[str, Any]],
        facts: list[CitationFact],
    ) -> None: ...

    def build_answer_sources(
        self,
        aliases: list[dict[str, Any]],
        evidence_ids: set[str],
    ) -> list[dict[str, Any]]: ...

    def cited_image_urls(self, state: Mapping[str, Any]) -> list[str]: ...

    def build_answer_envelope(self, state: dict[str, Any]) -> Any: ...


class AnswerCommitPort(Protocol):
    """最终不可安全重放副作用的提交端口。"""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]: ...


__all__ = ["AnswerCommitPort", "AnswerRenderPort", "AnsweringModelPort"]
