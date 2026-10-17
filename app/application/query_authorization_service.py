"""TODO-26 查询授权范围的应用层门禁与传播守卫。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Protocol, runtime_checkable

from app.domain.query_security import (
    ActiveIndexBinding,
    AuthorizedRetrievalScope,
    EgressPolicySnapshot,
    EntitlementSnapshot,
    ExternalDataDecision,
    ExternalDataRequest,
    RequestedRetrievalScope,
    ScopeAuthorizationError,
    authorize_retrieval_scope,
    decide_external_data,
)
from app.security.principal import Principal


class EntitlementUnavailableError(RuntimeError):
    """权限事实源不可用；调用方必须 fail-closed，不能回退到全租户。"""


@runtime_checkable
class EntitlementResolverPort(Protocol):
    def resolve(
        self,
        *,
        principal: Principal,
        requested: RequestedRetrievalScope,
        workspace_id: str | None = None,
    ) -> EntitlementSnapshot: ...


@runtime_checkable
class ActiveIndexBindingResolverPort(Protocol):
    def resolve(
        self,
        *,
        tenant_id: str,
        kb_ids: tuple[str, ...],
    ) -> tuple[ActiveIndexBinding, ...]: ...


@runtime_checkable
class ScopeDecisionAuditPort(Protocol):
    def record_scope_decision(
        self,
        *,
        allowed: bool,
        reason_code: str,
        fields: dict[str, Any],
    ) -> None: ...

    def record_egress_decision(self, decision: ExternalDataDecision) -> None: ...


class NullScopeDecisionAudit:
    """测试/离线调用的无状态审计端口。生产组合根应替换为持久化适配器。"""

    def record_scope_decision(
        self,
        *,
        allowed: bool,
        reason_code: str,
        fields: dict[str, Any],
    ) -> None:
        del allowed, reason_code, fields

    def record_egress_decision(self, decision: ExternalDataDecision) -> None:
        del decision


class QueryAuthorizationService:
    """把 principal、客户端范围和控制面事实冻结成 QueryRun 范围。"""

    def __init__(
        self,
        *,
        entitlements: EntitlementResolverPort,
        active_indexes: ActiveIndexBindingResolverPort,
        audit: ScopeDecisionAuditPort | None = None,
    ) -> None:
        self._entitlements = entitlements
        self._active_indexes = active_indexes
        self._audit = audit or NullScopeDecisionAudit()

    def authorize(
        self,
        *,
        principal: Principal,
        requested: RequestedRetrievalScope,
        workspace_id: str | None = None,
    ) -> AuthorizedRetrievalScope:
        safe_attempt = {
            "tenant_id": principal.tenant_id,
            "principal_id": principal.user_id,
            "requested_kb_count": len(requested.kb_ids),
            "requested_document_count": len(requested.documents),
            "business_filter_present": requested.business_filter is not None,
        }
        try:
            entitlement = self._entitlements.resolve(
                principal=principal,
                requested=requested,
                workspace_id=workspace_id,
            )
            if (
                entitlement.tenant_id != principal.tenant_id
                or entitlement.principal_id != principal.user_id
            ):
                raise ScopeAuthorizationError(
                    "entitlement identity does not match authenticated principal"
                )
            indexes = self._active_indexes.resolve(
                tenant_id=principal.tenant_id,
                kb_ids=tuple(item.kb_id for item in entitlement.grants),
            )
            scope = authorize_retrieval_scope(
                requested,
                entitlement,
                active_indexes=indexes,
            )
        except Exception as exc:
            self._audit.record_scope_decision(
                allowed=False,
                reason_code=(
                    exc.reason_code
                    if isinstance(exc, ScopeAuthorizationError)
                    else "ENTITLEMENT_UNAVAILABLE"
                ),
                fields=safe_attempt,
            )
            if isinstance(exc, ScopeAuthorizationError):
                raise
            raise EntitlementUnavailableError("authorization facts are unavailable") from exc

        self._audit.record_scope_decision(
            allowed=True,
            reason_code="RETRIEVAL_SCOPE_AUTHORIZED",
            fields=scope.audit_fields(),
        )
        return scope

    def decide_egress(
        self,
        *,
        request: ExternalDataRequest,
        policy: EgressPolicySnapshot | None,
    ) -> ExternalDataDecision:
        decision = decide_external_data(request, policy)
        self._audit.record_egress_decision(decision)
        return decision


@dataclass(slots=True)
class ScopePropagationGuard:
    """确保整个 QueryRun 的每个敏感阶段都消费同一个 scope。

    应在 dense、sparse、HyDE、rerank、parent、neighbor、citation、web 和 answer
    入口调用 :meth:`consume`。缺失、换 hash、换租户或扩大 KB 都立即失败。
    """

    run_id: str
    authorized_scope: AuthorizedRetrievalScope
    consumed_stages: dict[str, str] = field(default_factory=dict)

    REQUIRED_RETRIEVAL_STAGES = frozenset(
        {
            "dense",
            "sparse",
            "hyde",
            "rerank",
            "parent",
            "neighbor",
            "citation",
        }
    )

    def consume(
        self,
        stage: str,
        scope: AuthorizedRetrievalScope | None,
    ) -> AuthorizedRetrievalScope:
        normalized = stage.strip().lower()
        if not normalized:
            raise ValueError("scope consumer stage cannot be blank")
        if scope is None:
            raise ScopeAuthorizationError(f"{normalized} requires authorized scope")
        expected = self.authorized_scope
        if (
            scope.scope_hash != expected.scope_hash
            or scope.tenant_id != expected.tenant_id
            or scope.principal_id != expected.principal_id
            or not set(scope.allowed_kb_ids).issubset(expected.allowed_kb_ids)
        ):
            raise ScopeAuthorizationError(
                f"{normalized} attempted to replace or broaden authorized scope"
            )
        self.consumed_stages[normalized] = scope.scope_hash
        return scope

    def assert_required_stages(self, *, stages: set[str] | None = None) -> None:
        expected = stages or set(self.REQUIRED_RETRIEVAL_STAGES)
        missing = expected.difference(self.consumed_stages)
        if missing:
            raise ScopeAuthorizationError(
                "retrieval scope was not consumed by stages: " + ", ".join(sorted(missing))
            )


def scope_cache_fragment(scope: AuthorizedRetrievalScope) -> str:
    """缓存键必须同时包含策略与授权 hash；严禁只用 query/kb。"""

    payload = {
        "scope_hash": scope.scope_hash,
        "policy_version": scope.policy_version,
        "entitlement_version": scope.entitlement_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"authz_{sha256(raw.encode('utf-8')).hexdigest()}"


def compile_milvus_authorized_scope(
    scope: AuthorizedRetrievalScope,
    *,
    resolved_business_document_ids: tuple[str, ...] | None = None,
) -> str:
    """把冻结发布事实编译成 Milvus 表达式，所有值都经 JSON 转义。

    business filter 必须先由 SQL metadata catalog 解析；缺失或空交集不能被忽略。
    """

    if not scope.active_indexes:
        raise ScopeAuthorizationError("authorized scope has no frozen active index")
    if scope.business_filter is not None and resolved_business_document_ids is None:
        raise ScopeAuthorizationError("business filter has not been resolved")
    selected = (
        set(resolved_business_document_ids) if resolved_business_document_ids is not None else None
    )
    rows = [
        item for item in scope.active_indexes if selected is None or item.document_id in selected
    ]
    if not rows:
        raise ScopeAuthorizationError("authorized filter intersection is empty")

    clauses = [
        (
            "("
            f"tenant_id == {json.dumps(scope.tenant_id, ensure_ascii=False)} and "
            f"kb_id == {json.dumps(item.kb_id, ensure_ascii=False)} and "
            f"doc_id == {json.dumps(item.document_id, ensure_ascii=False)} and "
            "document_version_id == "
            f"{json.dumps(item.document_version_id, ensure_ascii=False)} and "
            f"index_version_id == {json.dumps(item.index_version_id, ensure_ascii=False)} and "
            "available_int == 1"
            ")"
        )
        for item in rows
    ]
    return "(" + " or ".join(clauses) + ")"


_DIRECT_IDENTIFIER_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b"),
    re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|secret)"
    r"\s*[:=]\s*([^\s,;]{4,})"
)


def redact_external_payload(
    payload: str,
    decision: ExternalDataDecision,
) -> str:
    """在实际外发前执行决策中的确定性脱敏；拒绝决策不能取得载荷。"""

    if not decision.allowed:
        raise PermissionError(f"external data denied: {decision.reason_code}")
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", payload)
    if "direct_identifier" in decision.redactions:
        for pattern in _DIRECT_IDENTIFIER_PATTERNS:
            sanitized = pattern.sub("[REDACTED_IDENTIFIER]", sanitized)
    return sanitized


__all__ = [
    "ActiveIndexBindingResolverPort",
    "EntitlementResolverPort",
    "EntitlementUnavailableError",
    "NullScopeDecisionAudit",
    "QueryAuthorizationService",
    "compile_milvus_authorized_scope",
    "redact_external_payload",
    "ScopeDecisionAuditPort",
    "ScopePropagationGuard",
    "scope_cache_fragment",
]
