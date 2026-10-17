"""TODO-26 SQL entitlement/ACL 与外发策略的隔离数据库验证。"""

from __future__ import annotations

from app.domain.query_security import DataClassification, RequestedRetrievalScope
from app.infra.persistence.sql.database import (
    configure_database,
    initialize_control_database,
    session_scope,
)
from app.infra.persistence.sql.models import (
    KnowledgeBaseAclModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.repositories import new_id
from app.infra.security.sql_query_authorization import (
    SqlEntitlementResolver,
    resolve_egress_policy,
)
from app.security.principal import Principal


def _principal(
    *,
    roles: tuple[str, ...] = ("Viewer",),
    scopes: tuple[str, ...] = (),
    auth_type: str = "test",
) -> Principal:
    return Principal(
        user_id="user_a",
        tenant_id="tenant_a",
        roles=roles,
        scopes=scopes,
        auth_type=auth_type,
        credential_id="credential_a",
    )


def _seed() -> None:
    with session_scope() as session:
        session.add_all(
            [
                TenantModel(id="tenant_a", name="Tenant A", status="active"),
                TenantModel(id="tenant_b", name="Tenant B", status="active"),
                WorkspaceModel(
                    id="workspace_a",
                    tenant_id="tenant_a",
                    name="Workspace A",
                    status="active",
                ),
                WorkspaceModel(
                    id="workspace_disabled",
                    tenant_id="tenant_a",
                    name="Disabled",
                    status="disabled",
                ),
                WorkspaceModel(
                    id="workspace_b",
                    tenant_id="tenant_b",
                    name="Workspace B",
                    status="active",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                KnowledgeBaseModel(
                    id="kb_owner",
                    tenant_id="tenant_a",
                    workspace_id="workspace_a",
                    name="Owner KB",
                    status="active",
                    created_by="user_a",
                    data_classification="internal",
                    external_llm_allowed=True,
                    web_search_allowed=False,
                ),
                KnowledgeBaseModel(
                    id="kb_acl",
                    tenant_id="tenant_a",
                    workspace_id="workspace_a",
                    name="ACL KB",
                    status="active",
                    created_by="other",
                    data_classification="public",
                    external_llm_allowed=True,
                    web_search_allowed=True,
                ),
                KnowledgeBaseModel(
                    id="kb_denied",
                    tenant_id="tenant_a",
                    workspace_id="workspace_a",
                    name="Denied KB",
                    status="active",
                    created_by="other",
                    data_classification="restricted",
                    external_llm_allowed=False,
                    web_search_allowed=False,
                ),
                KnowledgeBaseModel(
                    id="kb_disabled_workspace",
                    tenant_id="tenant_a",
                    workspace_id="workspace_disabled",
                    name="Disabled Workspace KB",
                    status="active",
                    created_by="user_a",
                ),
                KnowledgeBaseModel(
                    id="kb_other_tenant",
                    tenant_id="tenant_b",
                    workspace_id="workspace_b",
                    name="Other Tenant KB",
                    status="active",
                    created_by="user_a",
                ),
            ]
        )
        session.flush()
        session.add(
            KnowledgeBaseAclModel(
                id=new_id("acl"),
                kb_id="kb_acl",
                principal_type="user",
                principal_id="user_a",
                role="reader",
            )
        )


def test_sql_entitlement_resolves_owner_acl_and_tenant_boundary(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo26.db').as_posix()}")
    initialize_control_database()
    _seed()
    requested = RequestedRetrievalScope(
        kb_ids=(
            "kb_owner",
            "kb_acl",
            "kb_denied",
            "kb_disabled_workspace",
            "kb_other_tenant",
        )
    )
    entitlement = SqlEntitlementResolver().resolve(
        principal=_principal(),
        requested=requested,
        workspace_id="workspace_a",
    )
    assert [item.kb_id for item in entitlement.grants] == ["kb_acl", "kb_owner"]
    assert {item.grant_source for item in entitlement.grants} == {
        "owner",
        "user_acl",
    }
    assert entitlement.workspace_id == "workspace_a"
    assert entitlement.entitlement_version.startswith("ent.")

    admin = SqlEntitlementResolver().resolve(
        principal=_principal(roles=("TenantAdmin",)),
        requested=requested,
        workspace_id="workspace_a",
    )
    assert [item.kb_id for item in admin.grants] == [
        "kb_acl",
        "kb_denied",
        "kb_owner",
    ]
    assert all(item.grant_source == "tenant_admin" for item in admin.grants)


def test_service_resource_scope_and_egress_policy_are_fail_closed(tmp_path) -> None:
    configure_database(f"sqlite:///{(tmp_path / 'todo26-service.db').as_posix()}")
    initialize_control_database()
    _seed()
    requested = RequestedRetrievalScope(kb_ids=("kb_denied", "kb_acl"))
    service = SqlEntitlementResolver().resolve(
        principal=_principal(
            roles=("ServiceAccount",),
            scopes=("kb:kb_denied:read",),
            auth_type="api_key",
        ),
        requested=requested,
        workspace_id="workspace_a",
    )
    assert [item.kb_id for item in service.grants] == ["kb_denied"]
    assert service.grants[0].grant_source == "service_scope"

    mixed = resolve_egress_policy(
        tenant_id="tenant_a",
        kb_ids=("kb_acl", "kb_denied"),
        purpose="llm_generation",
        allowed_providers=("approved-llm",),
    )
    assert mixed.enabled is False
    assert mixed.max_classification == DataClassification.INTERNAL

    web = resolve_egress_policy(
        tenant_id="tenant_a",
        kb_ids=("kb_acl",),
        purpose="web_search",
        allowed_providers=("approved-search",),
        allowed_domains=("docs.example.com",),
    )
    assert web.enabled is True
    assert web.allowed_purposes == ("web_search",)
    assert web.max_classification == DataClassification.PUBLIC
