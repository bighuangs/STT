"""最终答案的可信引用与正式终态提交适配器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.evidence import CitationFact, citation_fact_bundle_hash
from app.infrastructure.answering.persistence import (
    persist_verified_citations,
    revalidate_cited_evidence,
)
from app.rag.query.evidence_asset_service import manifested_evidence_image_assets


class LegacyAnswerDeliveryRetiredError(RuntimeError):
    """旧回答发布路径已退役，不允许绕过 QueryRun 终态存储。"""

    code = "LEGACY_ANSWER_DELIVERY_RETIRED"


def cited_image_urls(state: Mapping[str, Any]) -> list[str]:
    """只投影最终证明实际引用 Evidence 的同源图片。"""

    cited_ids = {
        str(item.get("evidence_id") or "")
        for item in state.get("citations") or []
        if isinstance(item, Mapping) and item.get("validation_status") == "valid"
    }
    urls: list[str] = []
    for raw in state.get("context_aliases") or []:
        if not isinstance(raw, Mapping):
            continue
        evidence = raw.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id not in cited_ids:
            continue
        for asset in manifested_evidence_image_assets(dict(evidence)):
            url = asset.preview_url(evidence_id)
            if url and url not in urls:
                urls.append(url)
    return urls


def commit_answer_delivery(state: dict[str, Any]) -> dict[str, Any]:
    """在 proof/render 已冻结后提交可信引用，终态交由 QueryRun 持久化。"""

    evaluation_mode = bool(state.get("evaluation_mode", False))
    runtime_managed = bool(state.get("runtime_managed_history", False))
    if not evaluation_mode and not runtime_managed:
        # 旧 Mongo/SSE/task-result 通道无法原子持久化 AnswerProof，
        # 因此必须在任何引用或发布副作用前失败。
        raise LegacyAnswerDeliveryRetiredError(
            "legacy answer delivery is retired; use the durable QueryRun runtime"
        )

    raw_facts = state.get("pending_citation_facts") or []
    if not isinstance(raw_facts, list) or any(
        not isinstance(item, Mapping) for item in raw_facts
    ):
        raise ValueError("pending CitationFact bundle must be an ordered mapping list")
    citation_facts = [CitationFact.model_validate(item) for item in raw_facts]
    bundle_hash = citation_fact_bundle_hash(citation_facts)
    render_receipt = state.get("answer_render_receipt")
    if not isinstance(render_receipt, Mapping) or (
        render_receipt.get("schema_version") != "answer-render-receipt.v3"
        or render_receipt.get("citation_count") != len(citation_facts)
        or render_receipt.get("citation_fact_bundle_hash") != bundle_hash
    ):
        raise ValueError("CitationFact bundle differs from the frozen render receipt")
    evidence_ids = {item.evidence_id for item in citation_facts}
    evidences = revalidate_cited_evidence(state, evidence_ids)
    persist_verified_citations(state, evidences=evidences, facts=citation_facts)
    return {
        "schema_version": "answer-commit-receipt.v2",
        "status": "evaluated" if evaluation_mode else "delegated",
        "stream_chunk_count": 0,
        "task_result_written": False,
        "history_written": False,
        "citation_fact_count": len(citation_facts),
        "citation_fact_bundle_hash": bundle_hash,
        "runtime_terminal_write_delegated": runtime_managed,
    }


__all__ = [
    "LegacyAnswerDeliveryRetiredError",
    "cited_image_urls",
    "commit_answer_delivery",
]
