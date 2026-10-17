"""TODO-08 查询期 Evidence 恢复、完整性校验与授权收敛。

Milvus 只承担召回，不是来源事实库。每个命中都必须用 ``evidence_id`` 回到
控制库恢复正文与完整 provenance，并再次核对当前 active binding。任何缺失、
越权、旧版本或正文 hash 不一致的命中都 fail closed，不进入提示词。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
    RetrievalScore,
    sha256_digest,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.repositories import ControlRepository, ResourceNotFoundError
from app.rag.query.active_scope_filter import load_active_scopes
from app.security.principal import current_principal
from app.shared.runtime.logger import logger


class EvidenceIntegrityError(ValueError):
    """向量命中无法闭合到控制库事实。"""


def _evidence_domain_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    """从控制库 read model 投影出严格的 Evidence 领域载荷。

    ``ControlRepository.evidence_dict`` 还包含 ``created_at``、``updated_at``
    和便于前端展示的 ``primary_location``。这些字段不是 Evidence 身份的一部分，
    不能直接喂给 ``extra="forbid"`` 的领域模型；投影后，内容 hash、稳定 ID、
    来源身份和 locator 仍由 Evidence 完整校验。
    """

    return {
        field_name: evidence[field_name]
        for field_name in Evidence.model_fields
        if field_name in evidence
    }


def _active_scope_keys(kb_id: str | None) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            str(scope["tenant_id"]),
            str(scope["kb_id"]),
            str(scope["document_id"]),
            str(scope["document_version_id"]),
            str(scope["index_version_id"]),
        )
        for scope in load_active_scopes(kb_id)
    }


def _source_scope_key(evidence: dict[str, Any]) -> tuple[str, str, str, str, str] | None:
    source = evidence.get("source") or {}
    if source.get("source_type") != "local":
        return None
    fields = (
        source.get("tenant_id"),
        source.get("kb_id"),
        source.get("document_id"),
        source.get("document_version_id"),
        source.get("index_version_id"),
    )
    if not all(fields):
        return None
    return tuple(str(value) for value in fields)  # type: ignore[return-value]


def _load_evidence_rows(evidence_ids: list[str]) -> dict[str, dict[str, Any]]:
    principal = current_principal()
    tenant_id = principal.tenant_id if principal else None
    rows: dict[str, dict[str, Any]] = {}
    with session_scope() as session:
        repo = ControlRepository(session, tenant_id=tenant_id)
        for row in repo.get_evidence_models(evidence_ids):
            rows[row.id] = repo.evidence_dict(row)
    return rows


def hydrate_retrieved_hits(
    hits: list[dict[str, Any]],
    *,
    kb_id: str | None,
    route: str,
) -> list[dict[str, Any]]:
    """将 Milvus 紧凑命中恢复为经过授权与 hash 校验的 Evidence。"""

    if not hits:
        return []
    active_keys = _active_scope_keys(kb_id)
    by_id = _load_evidence_rows([str(hit.get("evidence_id") or "") for hit in hits])
    hydrated: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        evidence_id = str(hit.get("evidence_id") or "")
        evidence = by_id.get(evidence_id)
        reason = ""
        if not evidence:
            reason = "EVIDENCE_NOT_FOUND"
        elif _source_scope_key(evidence) not in active_keys:
            reason = "EVIDENCE_NOT_ACTIVE"
        else:
            source = evidence["source"]
            expected = {
                "document_id": source.get("document_id"),
                "document_version_id": source.get("document_version_id"),
                "index_version_id": source.get("index_version_id"),
                "chunk_id": source.get("chunk_id"),
                "content_hash": evidence.get("content_hash"),
            }
            observed = {
                "document_id": hit.get("document_id"),
                "document_version_id": hit.get("document_version_id"),
                "index_version_id": hit.get("index_version_id"),
                "chunk_id": hit.get("source_chunk_id") or hit.get("chunk_id"),
                "content_hash": hit.get("content_hash"),
            }
            if any(str(observed[key] or "") != str(value or "") for key, value in expected.items()):
                reason = "EVIDENCE_IDENTITY_MISMATCH"
            elif sha256_digest(str(hit.get("content") or "")) != evidence.get("content_hash"):
                reason = "EVIDENCE_CONTENT_HASH_MISMATCH"

        if reason:
            logger.warning(
                "丢弃无法校验的向量命中: code={} evidence_id={} route={}",
                reason,
                evidence_id[:48],
                route,
            )
            continue

        assert evidence is not None
        score = hit.get("retrieval_score", hit.get("score"))
        retrieval = RetrievalScore(
            route=route,
            raw_score=float(score) if score is not None else None,
            rank=rank,
        ).model_dump(mode="json", exclude_none=True)
        evidence = {**evidence, "retrieval": retrieval}
        source = evidence["source"]
        hydrated.append(
            {
                **hit,
                "chunk_id": source["chunk_id"],
                "source_chunk_id": source["chunk_id"],
                "evidence_id": evidence_id,
                "content": evidence["content"],
                "text": evidence["content"],
                "file_title": evidence.get("source_label") or hit.get("file_title"),
                "evidence": evidence,
                "retrieval_route": route,
                "retrieval_observations": [retrieval],
            }
        )
    return hydrated


def ensure_active_evidence(evidence: dict[str, Any], *, kb_id: str | None) -> dict[str, Any]:
    """生成结束前重新校验 active scope，阻断查询期间发布/删除竞态。"""

    value = Evidence.model_validate(_evidence_domain_payload(evidence)).model_dump(
        mode="json", exclude_none=True
    )
    if value["source"]["source_type"] == "web":
        principal = current_principal()
        if principal and value["source"].get("tenant_id") != principal.tenant_id:
            raise EvidenceIntegrityError("EVIDENCE_NOT_AUTHORIZED")
        return value
    if _source_scope_key(value) not in _active_scope_keys(kb_id):
        raise EvidenceIntegrityError("EVIDENCE_NOT_ACTIVE")
    current = _load_evidence_rows([value["evidence_id"]]).get(value["evidence_id"])
    if current is None:
        raise EvidenceIntegrityError("EVIDENCE_NOT_FOUND")
    # 控制库返回的是包含 created_at / primary_location 等展示字段的 read model。
    # 回源后再次还原为严格领域载荷，既校验库内事实本身，也保证下游
    # Context Pack / Citation 不会再次因 read-model 扩展字段而拒绝该证据。
    current_value = Evidence.model_validate(_evidence_domain_payload(current)).model_dump(
        mode="json", exclude_none=True
    )
    if current_value.get("content_hash") != value.get("content_hash"):
        raise EvidenceIntegrityError("EVIDENCE_CONTENT_HASH_MISMATCH")
    return current_value


def create_web_evidence(document: dict[str, Any]) -> dict[str, Any] | None:
    url = str(document.get("url") or "").strip()
    content = str(document.get("text") or document.get("snippet") or "").strip()
    if not url.startswith(("http://", "https://")) or not content:
        return None
    fetched_at = document.get("fetched_at")
    if isinstance(fetched_at, str):
        try:
            fetched_at = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        except ValueError:
            fetched_at = None
    if not isinstance(fetched_at, datetime) or fetched_at.tzinfo is None:
        # Provider 没有给抓取时刻时，以本服务接收并固定正文的时刻作为 captured-at。
        fetched_at = datetime.now(UTC)
    principal = current_principal()
    if principal is None:
        logger.warning("缺少认证主体，联网结果不会提升为可引用 Evidence")
        return None
    content_hash = sha256_digest(content)
    value = Evidence(
        source=EvidenceSourceIdentity(
            source_type="web",
            tenant_id=principal.tenant_id,
            source_url=url,
            content_hash=content_hash,
            fetched_at=fetched_at,
        ),
        content=content,
        content_hash=content_hash,
        provenance_spans=[
            EvidenceLocationSpan(
                block_id="web",
                availability="unavailable",
                reason_code="WEB_LOCATION_UNAVAILABLE",
            )
        ],
        source_label=str(document.get("title") or url)[:1024],
        retrieval=RetrievalScore(
            route="web",
            raw_score=(
                float(document["retrieval_score"])
                if document.get("retrieval_score") is not None
                else None
            ),
            rerank_score=(
                float(document["rerank_score"])
                if document.get("rerank_score") is not None
                else None
            ),
        ),
    )
    return value.model_dump(mode="json", exclude_none=True)


def prepare_reranked_evidence(
    documents: list[dict[str, Any]], *, kb_id: str | None
) -> list[dict[str, Any]]:
    """为 Prompt 准备最终 Evidence；本地事实再次校验，Web 固定内容快照。"""

    prepared: list[dict[str, Any]] = []
    for document in documents:
        raw_evidence = document.get("evidence")
        query_retrieval = (
            dict(raw_evidence.get("retrieval") or {})
            if isinstance(raw_evidence, dict)
            else {}
        )
        try:
            if document.get("type") == "web":
                evidence = create_web_evidence(document)
            else:
                evidence = (
                    ensure_active_evidence(raw_evidence, kb_id=kb_id)
                    if raw_evidence
                    else None
                )
        except (EvidenceIntegrityError, ValueError) as exc:
            logger.warning(
                "最终证据校验失败，已从生成上下文移除: evidence_id={} code={}",
                str(document.get("evidence_id") or "")[:48],
                str(exc),
            )
            continue
        if evidence is None:
            continue
        retrieval = query_retrieval or dict(evidence.get("retrieval") or {})
        observed_routes = document.get("retrieval_observations")
        if isinstance(observed_routes, list) and observed_routes:
            metadata = dict(retrieval.get("metadata") or {})
            metadata["route_observations"] = observed_routes
            retrieval["metadata"] = metadata
        if retrieval.get("raw_score") is None and document.get("retrieval_score") is not None:
            retrieval["raw_score"] = float(document["retrieval_score"])
        if retrieval.get("rank") is None and document.get("rank") is not None:
            retrieval["rank"] = int(document["rank"])
        if document.get("rrf_score") is not None:
            retrieval["rrf_score"] = float(document["rrf_score"])
        if document.get("rerank_score") is not None:
            retrieval["rerank_score"] = float(document["rerank_score"])
        retrieval.setdefault(
            "route",
            str(document.get("retrieval_route") or document.get("type") or "unknown"),
        )
        evidence = {**evidence, "retrieval": retrieval}
        prepared.append(
            {
                **document,
                "evidence_id": evidence["evidence_id"],
                "chunk_id": evidence["source"].get("chunk_id") or evidence["evidence_id"],
                "text": evidence["content"],
                "content": evidence["content"],
                "evidence": evidence,
            }
        )
    return prepared


def get_authorized_evidence(evidence_id: str) -> dict[str, Any]:
    """读取当前租户可访问的证据；保留的旧版本也可回看，不偷换 active 版本。"""

    row = _load_evidence_rows([evidence_id]).get(evidence_id)
    if row is None:
        raise ResourceNotFoundError(f"evidence not found: {evidence_id}")
    return row


__all__ = [
    "EvidenceIntegrityError",
    "create_web_evidence",
    "ensure_active_evidence",
    "get_authorized_evidence",
    "hydrate_retrieved_hits",
    "prepare_reranked_evidence",
]
