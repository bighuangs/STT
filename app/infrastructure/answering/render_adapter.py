"""配置化 AnswerRenderPort 实现。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.evidence import CitationFact
from app.infrastructure.answering.delivery import cited_image_urls
from app.infrastructure.answering.envelope import build_proof_answer_envelope
from app.infrastructure.answering.persistence import (
    new_answer_id,
    new_query_run_id,
    persist_verified_citations,
    revalidate_cited_evidence,
)
from app.infrastructure.answering.sources import build_answer_sources


class ConfiguredAnswerRenderAdapter:
    """把应用层渲染端口映射到现有授权、存储和公开投影实现。"""

    def new_answer_id(self) -> str:
        return new_answer_id()

    def new_query_run_id(self) -> str:
        return new_query_run_id()

    def revalidate_cited_evidence(
        self,
        state: Mapping[str, Any],
        evidence_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        return revalidate_cited_evidence(state, evidence_ids)

    def persist_verified_citations(
        self,
        state: Mapping[str, Any],
        *,
        evidences: Mapping[str, dict[str, Any]],
        facts: list[CitationFact],
    ) -> None:
        persist_verified_citations(state, evidences=evidences, facts=facts)

    def build_answer_sources(
        self,
        aliases: list[dict[str, Any]],
        evidence_ids: set[str],
    ) -> list[dict[str, Any]]:
        return build_answer_sources(aliases, evidence_ids)

    def cited_image_urls(self, state: Mapping[str, Any]) -> list[str]:
        return cited_image_urls(state)

    def build_answer_envelope(self, state: dict[str, Any]) -> Any:
        return build_proof_answer_envelope(state)


configured_answer_render_adapter = ConfiguredAnswerRenderAdapter()


__all__ = [
    "ConfiguredAnswerRenderAdapter",
    "configured_answer_render_adapter",
]
