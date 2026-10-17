"""TODO-34 查询流水线的内容安全应用服务。

该服务把 legacy 字典候选收敛成只含安全表示的对象；调用方若需保留原文，应写入
独立隔离存储，不能再把原字典直接送入 ContextPack 或 prompt。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.content_security import (
    ContentSecurityPolicy,
    ContentSource,
    SecuredContent,
    SecurityAction,
    SecurityStage,
    TrustLevel,
    make_security_decision,
)
from app.domain.query_security import DataClassification
from app.security.rag_content_security import (
    ClassifierPort,
    OutputSecurityResult,
    inspect_content,
    secure_model_output,
    validate_external_url,
)


class SecurityDecisionAuditPort(Protocol):
    def record(self, *, fields: dict[str, Any]) -> None: ...


class NullSecurityDecisionAudit:
    def record(self, *, fields: dict[str, Any]) -> None:
        del fields


class SecureEvidenceCandidate(BaseModel):
    """可进入 ContextPack 的证据候选；不包含原始未审查正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=512)
    evidence_id: str | None = Field(default=None, max_length=128)
    source_url: str | None = Field(default=None, max_length=4096)
    content: SecuredContent
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuarantinedEvidence(BaseModel):
    """隔离结果只保留定位字段/hash/判定，不复制恶意正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    evidence_id: str | None = None
    content_hash: str
    decision_id: str
    reason_codes: tuple[str, ...]


class EvidenceSecurityBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: tuple[SecureEvidenceCandidate, ...]
    quarantined: tuple[QuarantinedEvidence, ...]


class RagSecurityService:
    def __init__(
        self,
        *,
        policy: ContentSecurityPolicy,
        classifier: ClassifierPort | None = None,
        audit: SecurityDecisionAuditPort | None = None,
    ) -> None:
        self.policy = policy
        self._classifier = classifier
        self._audit = audit or NullSecurityDecisionAudit()

    def inspect_query(
        self,
        *,
        text: str,
        message_id: str,
        tenant_id: str,
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> SecuredContent:
        result = inspect_content(
            text,
            source=ContentSource.USER,
            trust_level=TrustLevel.AUTHENTICATED_USER,
            source_ref=message_id,
            classification=classification,
            owner_tenant=tenant_id,
            policy=self.policy,
            stage=SecurityStage.INPUT,
            classifier=self._classifier,
        )
        self._audit.record(fields=result.decision.audit_fields())
        return result

    def inspect_evidence(
        self,
        candidates: list[dict[str, Any]],
        *,
        tenant_id: str,
        default_classification: DataClassification = DataClassification.INTERNAL,
    ) -> EvidenceSecurityBatch:
        approved: list[SecureEvidenceCandidate] = []
        quarantined: list[QuarantinedEvidence] = []
        for position, candidate in enumerate(candidates, start=1):
            candidate_id = str(
                candidate.get("chunk_id")
                or candidate.get("evidence_id")
                or candidate.get("url")
                or f"candidate-{position}"
            )
            source = (
                ContentSource.KNOWLEDGE_BASE
                if str(candidate.get("type") or "").lower() in {"milvus", "knowledge_base", "local"}
                else ContentSource.WEB
            )
            classification_raw = str(
                candidate.get("data_classification") or default_classification.value
            ).lower()
            try:
                classification = DataClassification(classification_raw)
            except ValueError:
                classification = DataClassification.RESTRICTED
            secured = inspect_content(
                str(candidate.get("text") or candidate.get("content") or ""),
                source=source,
                trust_level=(
                    TrustLevel.TENANT_MANAGED
                    if source == ContentSource.KNOWLEDGE_BASE
                    else TrustLevel.EXTERNAL_UNTRUSTED
                ),
                source_ref=candidate_id,
                classification=classification,
                owner_tenant=(tenant_id if source == ContentSource.KNOWLEDGE_BASE else None),
                policy=self.policy,
                stage=SecurityStage.EVIDENCE,
                classifier=self._classifier,
            )
            source_url = str(candidate.get("url") or "").strip() or None
            if source == ContentSource.WEB and source_url:
                url_decision = validate_external_url(
                    source_url,
                    allowed_domains=self.policy.allowed_fetch_domains,
                )
                if not url_decision.allowed:
                    decision = make_security_decision(
                        stage=SecurityStage.EVIDENCE,
                        action=SecurityAction.QUARANTINE,
                        reason_codes=(url_decision.reason_code,),
                        policy_version=self.policy.policy_version,
                        detector_versions=secured.decision.detector_versions,
                        content_hash=secured.label.content_hash,
                        source_ref=candidate_id,
                    )
                    secured = secured.model_copy(
                        update={
                            "safe_representation": "",
                            "decision": decision,
                            "derivation_receipt": None,
                        }
                    )
                else:
                    source_url = url_decision.normalized_url
            if (
                secured.decision.action in {SecurityAction.ALLOW, SecurityAction.REDACT}
                and not secured.safe_representation.strip()
            ):
                decision = make_security_decision(
                    stage=SecurityStage.EVIDENCE,
                    action=SecurityAction.QUARANTINE,
                    reason_codes=("SAFE_REPRESENTATION_EMPTY",),
                    policy_version=self.policy.policy_version,
                    detector_versions=secured.decision.detector_versions,
                    content_hash=secured.label.content_hash,
                    source_ref=candidate_id,
                )
                secured = secured.model_copy(
                    update={
                        "safe_representation": "",
                        "decision": decision,
                        "derivation_receipt": None,
                    }
                )
            self._audit.record(fields=secured.decision.audit_fields())
            if secured.decision.action in {
                SecurityAction.ALLOW,
                SecurityAction.REDACT,
            }:
                # 只保留展示/检索诊断所需非正文元数据。
                safe_metadata = {
                    key: candidate[key]
                    for key in (
                        "title",
                        "document_id",
                        "document_version_id",
                        "index_version_id",
                        "kb_id",
                        "score",
                    )
                    if key in candidate
                }
                approved.append(
                    SecureEvidenceCandidate(
                        candidate_id=candidate_id,
                        evidence_id=candidate.get("evidence_id"),
                        source_url=source_url,
                        content=secured,
                        metadata=safe_metadata,
                    )
                )
            else:
                quarantined.append(
                    QuarantinedEvidence(
                        candidate_id=candidate_id,
                        evidence_id=candidate.get("evidence_id"),
                        content_hash=secured.label.content_hash,
                        decision_id=secured.decision.decision_id,
                        reason_codes=secured.decision.reason_codes,
                    )
                )
        return EvidenceSecurityBatch(
            approved=tuple(approved),
            quarantined=tuple(quarantined),
        )

    def inspect_answer(
        self,
        *,
        output: str,
        answer_id: str,
        allowed_citation_ids: set[str],
    ) -> OutputSecurityResult:
        result = secure_model_output(
            output,
            source_ref=answer_id,
            allowed_citation_ids=allowed_citation_ids,
            policy=self.policy,
        )
        self._audit.record(fields=result.content.decision.audit_fields())
        return result


__all__ = [
    "EvidenceSecurityBatch",
    "NullSecurityDecisionAudit",
    "QuarantinedEvidence",
    "RagSecurityService",
    "SecureEvidenceCandidate",
    "SecurityDecisionAuditPort",
]
