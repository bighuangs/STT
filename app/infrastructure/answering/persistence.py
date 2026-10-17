"""可信引用的授权复核与原子持久化适配器。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4

from app.application.control_plane_service import record_citation_bundle
from app.domain.evidence import CitationFact
from app.rag.query.evidence_service import ensure_active_evidence
from app.security.principal import current_principal


def new_answer_id() -> str:
    """生成公开回答身份。"""

    return f"ans_{uuid4().hex}"


def new_query_run_id() -> str:
    """为没有可靠运行身份的本地入口生成查询身份。"""

    return f"qry_{uuid4().hex}"


def revalidate_cited_evidence(
    state: Mapping[str, Any],
    evidence_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """在持久化前再次校验 Evidence 仍属于当前授权发布范围。"""

    requested = set(evidence_ids)
    candidates: dict[str, dict[str, Any]] = {}
    for raw in state.get("context_aliases") or []:
        if not isinstance(raw, Mapping):
            continue
        evidence = raw.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id in requested:
            candidates[evidence_id] = dict(evidence)
    if set(candidates) != requested:
        raise ValueError("answer plan references Evidence outside ContextPack")

    if state.get("evaluation_mode", False):
        return candidates
    return {
        evidence_id: ensure_active_evidence(
            evidence,
            kb_id=str(state.get("kb_id") or "") or None,
        )
        for evidence_id, evidence in candidates.items()
    }


def persist_verified_citations(
    state: Mapping[str, Any],
    *,
    evidences: Mapping[str, dict[str, Any]],
    facts: list[CitationFact],
) -> None:
    """引用落库失败时抛错，禁止把未持久化关系伪装为可信引用。"""

    if state.get("evaluation_mode", False) or not facts:
        return
    principal = current_principal()
    record_citation_bundle(
        list(evidences.values()),
        facts,
        tenant_id=principal.tenant_id if principal is not None else None,
    )
