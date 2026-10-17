"""P3 RAG 内容安全、信任标签与决策领域契约。"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.query_security import DataClassification
from app.domain.safe_representation import SafeRepresentationDerivationReceipt


class ContentSource(StrEnum):
    USER = "user"
    KNOWLEDGE_BASE = "knowledge_base"
    WEB = "web"
    HISTORY = "history"
    TOOL = "tool"
    APPLICATION = "application"
    MODEL_OUTPUT = "model_output"


class TrustLevel(StrEnum):
    TRUSTED_APPLICATION = "trusted_application"
    AUTHENTICATED_USER = "authenticated_user"
    TENANT_MANAGED = "tenant_managed"
    EXTERNAL_UNTRUSTED = "external_untrusted"
    TOOL_UNTRUSTED = "tool_untrusted"


class SecurityAction(StrEnum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class SecurityStage(StrEnum):
    INPUT = "input"
    EVIDENCE = "evidence"
    CONTEXT = "context"
    OUTPUT = "output"
    FETCH = "fetch"
    ACTION = "action"


class ContentSecurityLabel(BaseModel):
    """内容的来源、信任级别和安全处理标签，不包含原文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ContentSource
    trust_level: TrustLevel
    owner_tenant: str | None = Field(default=None, max_length=64)
    source_ref: str = Field(min_length=1, max_length=512)
    data_classification: DataClassification
    injection_signals: tuple[str, ...] = ()
    sanitization_actions: tuple[str, ...] = ()
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    detector_versions: tuple[str, ...] = ()

    @field_validator("owner_tenant")
    @classmethod
    def tenant_required_for_managed_sources(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("owner_tenant cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_owner(self) -> "ContentSecurityLabel":
        if (
            self.source in {ContentSource.KNOWLEDGE_BASE, ContentSource.HISTORY}
            and self.owner_tenant is None
        ):
            raise ValueError("tenant-managed content requires owner_tenant")
        return self


class SecurityDecision(BaseModel):
    """版本化安全判定，可持久化审计但不包含敏感正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: SecurityStage
    action: SecurityAction
    reason_codes: tuple[str, ...] = Field(min_length=1)
    policy_version: str = Field(min_length=1, max_length=128)
    detector_versions: tuple[str, ...]
    decision_id: str = Field(pattern=r"^sec_[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_ref: str = Field(min_length=1, max_length=512)

    def audit_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SecuredContent(BaseModel):
    """允许进入下一处理阶段的安全表示。

    原始正文不放进该对象，调用方应在受控隔离存储中用 ``content_hash`` 关联保存。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    safe_representation: str = Field(max_length=4_000_000)
    label: ContentSecurityLabel
    decision: SecurityDecision
    derivation_receipt: SafeRepresentationDerivationReceipt | None = None

    @model_validator(mode="after")
    def consistent_decision(self) -> "SecuredContent":
        if self.label.content_hash != self.decision.content_hash:
            raise ValueError("content label and decision hashes must match")
        if self.label.source_ref != self.decision.source_ref:
            raise ValueError("content label and decision source_ref must match")
        if (
            self.decision.action
            in {
                SecurityAction.BLOCK,
                SecurityAction.QUARANTINE,
            }
            and self.safe_representation
        ):
            raise ValueError("blocked or quarantined content cannot have a safe representation")
        if self.derivation_receipt is not None:
            if self.derivation_receipt.canonical_content_hash != self.label.content_hash:
                raise ValueError("safe derivation and content label hashes must match")
            if self.derivation_receipt.safe_representation_hash != content_digest(
                self.safe_representation
            ):
                raise ValueError("safe derivation and safe representation hashes must match")
            if self.decision.action in {
                SecurityAction.BLOCK,
                SecurityAction.QUARANTINE,
            }:
                raise ValueError("blocked or quarantined content cannot carry a derivation")
        return self


class ContentSecurityPolicy(BaseModel):
    """查询内容安全策略；默认禁止 Web/工具并对外部内容 fail-closed。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="rag-security.v1", min_length=1, max_length=128)
    web_enabled: bool = False
    tools_enabled: bool = False
    high_risk_sources_enabled: bool = False
    classifier_required_for_external: bool = False
    block_user_injection: bool = True
    max_input_chars: int = Field(default=32_000, ge=1, le=1_000_000)
    max_evidence_chars: int = Field(default=200_000, ge=1, le=4_000_000)
    max_output_chars: int = Field(default=128_000, ge=1, le=1_000_000)
    max_context_items: int = Field(default=32, ge=1, le=10_000)
    max_items_per_source: int = Field(default=5, ge=1, le=1000)
    allowed_fetch_domains: tuple[str, ...] = ()
    allowed_output_link_domains: tuple[str, ...] = ()
    allowed_fetch_mime_types: tuple[str, ...] = (
        "text/plain",
        "text/html",
        "application/json",
        "application/pdf",
    )
    max_fetch_bytes: int = Field(default=5_000_000, ge=1, le=100_000_000)
    max_redirects: int = Field(default=3, ge=0, le=10)


def content_digest(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def make_security_decision(
    *,
    stage: SecurityStage,
    action: SecurityAction,
    reason_codes: tuple[str, ...],
    policy_version: str,
    detector_versions: tuple[str, ...],
    content_hash: str,
    source_ref: str,
) -> SecurityDecision:
    payload = {
        "stage": stage.value,
        "action": action.value,
        "reason_codes": sorted(set(reason_codes)),
        "policy_version": policy_version,
        "detector_versions": sorted(set(detector_versions)),
        "content_hash": content_hash,
        "source_ref": source_ref,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return SecurityDecision(
        **payload,
        decision_id=f"sec_{sha256(serialized.encode('utf-8')).hexdigest()}",
    )


__all__ = [
    "ContentSecurityLabel",
    "ContentSecurityPolicy",
    "ContentSource",
    "SecuredContent",
    "SecurityAction",
    "SecurityDecision",
    "SecurityStage",
    "TrustLevel",
    "content_digest",
    "make_security_decision",
]
