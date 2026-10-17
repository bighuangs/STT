"""TODO-26 基于 SQL 控制面的授权、活动索引和外发策略适配器。"""

from __future__ import annotations

from hashlib import sha256
import json

from sqlalchemy import or_, select

from app.application.index_lifecycle_service import active_index_scopes
from app.domain.query_security import (
    ActiveIndexBinding,
    DataClassification,
    EgressPolicySnapshot,
    EntitledKnowledgeBase,
    EntitlementSnapshot,
    ExternalDataDecision,
    RequestedRetrievalScope,
    ScopeAuthorizationError,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import (
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.security.principal import Principal
from app.infra.security.audit_repository import audit


_QUERY_ACL_ROLES = frozenset({"owner", "admin", "editor", "reader", "query"})


def _version(prefix: str, payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}.{sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _service_scope_allows(principal: Principal, kb_id: str) -> bool:
    values = set(principal.scopes)
    return (
        "*" in values
        or "kb:*:read" in values
        or f"kb:{kb_id}:read" in values
        or f"kb:{kb_id}:query" in values
    )


class SqlEntitlementResolver:
    """从租户、workspace、owner 与 KB ACL 解析授权。

    非 TenantAdmin 不因拥有 ``query:execute`` 自动获得整个租户；需要 owner、
    user/role ACL 或 API key 的资源 scope。查询只筛选客户端请求的 ID，未授权和
    不存在资源得到同一空交集，防止枚举。
    """

    policy_version = "query-authz.sql.v1"

    def resolve(
        self,
        *,
        principal: Principal,
        requested: RequestedRetrievalScope,
        workspace_id: str | None = None,
    ) -> EntitlementSnapshot:
        with session_scope() as session:
            rows = session.scalars(
                select(KnowledgeBaseModel)
                .join(
                    WorkspaceModel,
                    WorkspaceModel.id == KnowledgeBaseModel.workspace_id,
                )
                .join(TenantModel, TenantModel.id == KnowledgeBaseModel.tenant_id)
                .where(
                    KnowledgeBaseModel.tenant_id == principal.tenant_id,
                    KnowledgeBaseModel.status == "active",
                    WorkspaceModel.status == "active",
                    WorkspaceModel.tenant_id == principal.tenant_id,
                    TenantModel.status == "active",
                    KnowledgeBaseModel.id.in_(requested.kb_ids),
                )
            ).all()
            if workspace_id is not None:
                rows = [item for item in rows if item.workspace_id == workspace_id]
            workspaces = {item.workspace_id for item in rows}
            if len(workspaces) > 1:
                raise ScopeAuthorizationError("one query run cannot span multiple workspaces")
            effective_workspace = workspace_id or (
                next(iter(workspaces)) if workspaces else "workspace_denied"
            )

            acl_rows = (
                session.scalars(
                    select(KnowledgeBaseAclModel).where(
                        KnowledgeBaseAclModel.kb_id.in_([item.id for item in rows]),
                        or_(
                            (KnowledgeBaseAclModel.principal_type == "user")
                            & (KnowledgeBaseAclModel.principal_id == principal.user_id),
                            (KnowledgeBaseAclModel.principal_type == "role")
                            & (KnowledgeBaseAclModel.principal_id.in_(principal.roles)),
                        ),
                    )
                ).all()
                if rows
                else []
            )
            acl_by_kb: dict[str, list[KnowledgeBaseAclModel]] = {}
            for item in acl_rows:
                acl_by_kb.setdefault(item.kb_id, []).append(item)

            grants: list[EntitledKnowledgeBase] = []
            facts: list[dict[str, str]] = []
            for kb in rows:
                source: str | None = None
                role = ""
                if principal.auth_type == "api_key":
                    if _service_scope_allows(principal, kb.id):
                        source, role = "service_scope", "query"
                elif "TenantAdmin" in principal.roles:
                    source, role = "tenant_admin", "admin"
                elif kb.created_by == principal.user_id:
                    source, role = "owner", "owner"
                else:
                    matching = next(
                        (
                            item
                            for item in acl_by_kb.get(kb.id, ())
                            if item.role.lower() in _QUERY_ACL_ROLES
                        ),
                        None,
                    )
                    if matching is not None:
                        source = "user_acl" if matching.principal_type == "user" else "role_acl"
                        role = matching.role.lower()
                if source is None:
                    continue
                grants.append(
                    EntitledKnowledgeBase(
                        kb_id=kb.id,
                        document_ids=None,
                        grant_source=source,
                        grant_role=role,
                    )
                )
                facts.append(
                    {
                        "kb_id": kb.id,
                        "workspace_id": kb.workspace_id,
                        "source": source,
                        "role": role,
                        "updated_at": kb.updated_at.isoformat(),
                    }
                )
            facts.append(
                {
                    "principal_id": principal.user_id,
                    "roles": ",".join(sorted(principal.roles)),
                    "scopes": ",".join(sorted(principal.scopes)),
                    "matching_acl": ",".join(
                        sorted(
                            f"{item.id}:{item.principal_type}:{item.principal_id}:"
                            f"{item.role}:{item.updated_at.isoformat()}"
                            for item in acl_rows
                        )
                    ),
                }
            )

        return EntitlementSnapshot(
            tenant_id=principal.tenant_id,
            principal_id=principal.user_id,
            workspace_id=effective_workspace,
            grants=tuple(sorted(grants, key=lambda item: item.kb_id)),
            policy_version=self.policy_version,
            entitlement_version=_version("ent", facts),
        )


class SqlActiveIndexBindingResolver:
    """把当前发布指针冻结到 QueryRun scope。"""

    def resolve(
        self,
        *,
        tenant_id: str,
        kb_ids: tuple[str, ...],
    ) -> tuple[ActiveIndexBinding, ...]:
        selected = set(kb_ids)
        output: list[ActiveIndexBinding] = []
        for row in active_index_scopes():
            if (
                str(row.get("tenant_id") or "") == tenant_id
                and str(row.get("kb_id") or "") in selected
            ):
                output.append(
                    ActiveIndexBinding(
                        kb_id=str(row["kb_id"]),
                        document_id=str(row["document_id"]),
                        document_version_id=str(row["document_version_id"]),
                        index_version_id=str(row["index_version_id"]),
                    )
                )
        return tuple(
            sorted(
                output,
                key=lambda item: (
                    item.kb_id,
                    item.document_id,
                    item.document_version_id,
                    item.index_version_id,
                ),
            )
        )


class SqlScopeDecisionAudit:
    """复用 TODO-04 审计表，只持久化 hash/ID/策略版本，不写查询正文。"""

    def __init__(self, principal: Principal) -> None:
        self._principal = principal

    def record_scope_decision(
        self,
        *,
        allowed: bool,
        reason_code: str,
        fields: dict,
    ) -> None:
        audit(
            self._principal,
            "query.scope.authorize",
            "allowed" if allowed else "denied",
            self._principal.request_id,
            None,
            {"reason_code": reason_code, **fields},
        )

    def record_egress_decision(self, decision: ExternalDataDecision) -> None:
        audit(
            self._principal,
            "query.egress.decide",
            "allowed" if decision.allowed else "denied",
            self._principal.request_id,
            None,
            decision.audit_fields(),
        )


def resolve_egress_policy(
    *,
    tenant_id: str,
    kb_ids: tuple[str, ...],
    purpose: str,
    allowed_providers: tuple[str, ...] = (),
    allowed_domains: tuple[str, ...] = (),
) -> EgressPolicySnapshot:
    """从所有授权 KB 取最严格外发策略。

    任一 KB 禁止外部 LLM/Web、资源缺失或跨租户都会令 ``enabled=False``。
    机密/受限 KB 即使布尔开关开启也不会放宽到超过 ``internal``。
    """

    selected = set(kb_ids)
    with session_scope() as session:
        rows = session.scalars(
            select(KnowledgeBaseModel).where(
                KnowledgeBaseModel.tenant_id == tenant_id,
                KnowledgeBaseModel.id.in_(selected),
                KnowledgeBaseModel.status == "active",
            )
        ).all()
    complete = {item.id for item in rows} == selected and bool(selected)
    if purpose == "web_search":
        enabled = complete and all(item.web_search_allowed for item in rows)
    else:
        enabled = complete and all(item.external_llm_allowed for item in rows)
    classifications = {str(item.data_classification or "internal").lower() for item in rows}
    max_classification = (
        DataClassification.PUBLIC
        if classifications and classifications <= {"public"}
        else DataClassification.INTERNAL
    )
    facts = [
        {
            "kb_id": item.id,
            "classification": item.data_classification,
            "external_llm_allowed": item.external_llm_allowed,
            "web_search_allowed": item.web_search_allowed,
            "updated_at": item.updated_at.isoformat(),
        }
        for item in rows
    ]
    return EgressPolicySnapshot(
        policy_version=_version("egress", facts),
        enabled=enabled,
        allowed_providers=allowed_providers,
        allowed_purposes=(purpose,) if enabled else (),
        allowed_domains=allowed_domains,
        max_classification=max_classification,
        redact_direct_identifiers=True,
        retention_policy="provider-zero-retention-required",
    )


__all__ = [
    "SqlActiveIndexBindingResolver",
    "SqlEntitlementResolver",
    "SqlScopeDecisionAudit",
    "resolve_egress_policy",
]
