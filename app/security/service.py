from datetime import UTC, datetime
import hmac
from secrets import token_urlsafe

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
try:
    # argon2-cffi 23.1+ 使用 InvalidHashError；旧版仍导出 InvalidHash。
    from argon2.exceptions import InvalidHashError
except ImportError:  # pragma: no cover - 仅用于兼容旧运行环境
    from argon2.exceptions import InvalidHash as InvalidHashError
from sqlalchemy import select

from app.infra.persistence.sql.database import initialize_control_database, session_scope
from app.infra.security.audit_repository import audit
from app.infra.persistence.sql.models import (
    ApiCredentialModel, AuditEventModel, QuerySessionModel, RefreshCredentialModel,
    TenantMembershipModel, TenantModel, UserModel, WorkspaceModel,
)
from app.infra.persistence.sql.repositories import DEFAULT_TENANT_ID, DEFAULT_WORKSPACE_ID, new_id
from app.security.permissions import is_allowed
from app.security.principal import Principal
from app.security.tokens import InvalidTokenError, decode_token, issue_token, token_digest, token_digests
from app.shared.config.settings_config import settings

_password_hasher = PasswordHasher()


class AuthenticationError(ValueError):
    pass


class AuthorizationError(PermissionError):
    pass


class SessionKnowledgeBaseConflict(AuthorizationError):
    """查询会话已绑定其他知识库，不能原地改写其安全与检索快照。"""


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def bootstrap_security() -> None:
    if settings.app_env.lower() in {"prod", "production"}:
        if settings.auth_token_secret == "dev-only-change-this-stt-token-secret-2026":
            raise RuntimeError("生产环境必须显式设置 AUTH_TOKEN_SECRET")
        if settings.auth_bootstrap_admin_password == "Stt-Admin-ChangeMe!":
            raise RuntimeError("生产环境必须显式设置 AUTH_BOOTSTRAP_ADMIN_PASSWORD")
    initialize_control_database()
    with session_scope() as session:
        tenant = session.get(TenantModel, DEFAULT_TENANT_ID)
        if tenant is None:
            tenant = TenantModel(id=DEFAULT_TENANT_ID, name="Default Tenant", status="active")
            session.add(tenant)
        if session.get(WorkspaceModel, DEFAULT_WORKSPACE_ID) is None:
            session.add(WorkspaceModel(id=DEFAULT_WORKSPACE_ID, tenant_id=DEFAULT_TENANT_ID, name="Default Workspace", status="active"))
        email = settings.auth_bootstrap_admin_email.strip().lower()
        user = session.scalar(select(UserModel).where(UserModel.email == email))
        if user is None:
            user = UserModel(id=new_id("usr"), email=email, display_name="STT Bootstrap Admin", password_hash=hash_password(settings.auth_bootstrap_admin_password), status="active", token_version=1)
            session.add(user)
            session.flush()
        membership = session.scalar(select(TenantMembershipModel).where(TenantMembershipModel.tenant_id == DEFAULT_TENANT_ID, TenantMembershipModel.user_id == user.id))
        if membership is None:
            session.add(TenantMembershipModel(id=new_id("mem"), tenant_id=DEFAULT_TENANT_ID, user_id=user.id, role="TenantAdmin", status="active"))


def _membership(session, user_id: str, tenant_id: str | None):
    query = select(TenantMembershipModel).where(TenantMembershipModel.user_id == user_id, TenantMembershipModel.status == "active")
    if tenant_id:
        query = query.where(TenantMembershipModel.tenant_id == tenant_id)
    return session.scalar(query.order_by(TenantMembershipModel.created_at))


def _principal(session, user: UserModel, membership: TenantMembershipModel, auth_type: str, credential_id: str, scopes=()) -> Principal:
    if user.status != "active" or membership.status != "active":
        raise AuthenticationError("account or tenant membership is disabled")
    return Principal(user.id, membership.tenant_id, (membership.role,), tuple(scopes), auth_type, credential_id)


def login(email: str, password: str, tenant_id: str | None = None) -> tuple[dict, str]:
    with session_scope() as session:
        user = session.scalar(select(UserModel).where(UserModel.email == email.strip().lower()))
        if user is None or not verify_password(user.password_hash, password):
            raise AuthenticationError("invalid email or password")
        membership = _membership(session, user.id, tenant_id)
        if membership is None:
            raise AuthenticationError("no active membership for the selected tenant")
        access, _, access_exp = issue_token(user_id=user.id, tenant_id=membership.tenant_id, token_version=user.token_version, token_type="access")
        refresh, _, refresh_exp = issue_token(user_id=user.id, tenant_id=membership.tenant_id, token_version=user.token_version, token_type="refresh")
        session.add(RefreshCredentialModel(id=new_id("rft"), tenant_id=membership.tenant_id, user_id=user.id, token_hash=token_digest(refresh), expires_at=refresh_exp))
        principal = _principal(session, user, membership, "password", user.id)
        return {"access_token": access, "refresh_token": refresh, "token_type": "bearer", "expires_at": access_exp.isoformat(), "principal": principal_dict(principal)}, access


def refresh_tokens(refresh_token: str) -> tuple[dict, str]:
    try:
        claims = decode_token(refresh_token, "refresh")
    except InvalidTokenError as exc:
        raise AuthenticationError(str(exc)) from exc
    with session_scope() as session:
        credential = session.scalar(select(RefreshCredentialModel).where(RefreshCredentialModel.token_hash.in_(token_digests(refresh_token))))
        if credential is None or credential.revoked_at is not None or _aware(credential.expires_at) <= datetime.now(UTC):
            raise AuthenticationError("refresh token is revoked or expired")
        user = session.get(UserModel, claims["sub"])
        membership = _membership(session, claims["sub"], claims["tid"])
        if user is None or membership is None or user.token_version != int(claims["ver"]):
            raise AuthenticationError("refresh token is no longer valid")
        credential.revoked_at = datetime.now(UTC)
        access, _, access_exp = issue_token(user_id=user.id, tenant_id=membership.tenant_id, token_version=user.token_version, token_type="access")
        replacement, _, refresh_exp = issue_token(user_id=user.id, tenant_id=membership.tenant_id, token_version=user.token_version, token_type="refresh")
        session.add(RefreshCredentialModel(id=new_id("rft"), tenant_id=membership.tenant_id, user_id=user.id, token_hash=token_digest(replacement), expires_at=refresh_exp))
        principal = _principal(session, user, membership, "refresh", credential.id)
        return {"access_token": access, "refresh_token": replacement, "token_type": "bearer", "expires_at": access_exp.isoformat(), "principal": principal_dict(principal)}, access


def authenticate_access(token: str, request_id: str = "") -> Principal:
    try:
        claims = decode_token(token, "access")
    except InvalidTokenError as exc:
        raise AuthenticationError(str(exc)) from exc
    with session_scope() as session:
        user = session.get(UserModel, claims["sub"])
        membership = _membership(session, claims["sub"], claims["tid"])
        if user is None or membership is None or user.token_version != int(claims["ver"]):
            raise AuthenticationError("access token is no longer valid")
        principal = _principal(session, user, membership, "bearer", claims["jti"])
        return Principal(**{**principal.__dict__, "request_id": request_id})


def authenticate_api_key(value: str, request_id: str = "") -> Principal:
    prefix = value.split(".", 1)[0]
    with session_scope() as session:
        credential = session.scalar(select(ApiCredentialModel).where(ApiCredentialModel.prefix == prefix, ApiCredentialModel.status == "active"))
        if credential is None or not any(
            hmac.compare_digest(credential.secret_hash, digest) for digest in token_digests(value)
        ):
            raise AuthenticationError("invalid API key")
        if credential.expires_at and _aware(credential.expires_at) <= datetime.now(UTC):
            raise AuthenticationError("API key expired")
        user = session.get(UserModel, credential.user_id)
        membership = _membership(session, credential.user_id, credential.tenant_id)
        if user is None or membership is None:
            raise AuthenticationError("API key owner is disabled")
        credential.last_used_at = datetime.now(UTC)
        if user.status != "active" or membership.status != "active":
            raise AuthenticationError("API key owner is disabled")
        return Principal(
            user.id, membership.tenant_id, ("ServiceAccount",), tuple(credential.scopes),
            "api_key", credential.id, request_id,
        )


def require(principal: Principal, action: str) -> None:
    if not is_allowed(principal.roles, principal.scopes, action):
        raise AuthorizationError(f"permission denied: {action}")


def principal_dict(principal: Principal) -> dict:
    return {"user_id": principal.user_id, "tenant_id": principal.tenant_id, "roles": list(principal.roles), "scopes": list(principal.scopes), "auth_type": principal.auth_type, "credential_id": principal.credential_id, "request_id": principal.request_id}


def create_tenant_user(principal: Principal, email: str, password: str, display_name: str, role: str) -> dict:
    require(principal, "member:manage")
    if role not in {"TenantAdmin", "KBAdmin", "Editor", "Viewer"}:
        raise ValueError("unsupported role")
    with session_scope() as session:
        normalized = email.strip().lower()
        user = session.scalar(select(UserModel).where(UserModel.email == normalized))
        if user is None:
            user = UserModel(id=new_id("usr"), email=normalized, display_name=display_name.strip() or normalized, password_hash=hash_password(password), status="active", token_version=1)
            session.add(user)
            session.flush()
        membership = _membership(session, user.id, principal.tenant_id)
        if membership is None:
            membership = TenantMembershipModel(id=new_id("mem"), tenant_id=principal.tenant_id, user_id=user.id, role=role, status="active")
            session.add(membership)
        else:
            membership.role = role
            membership.status = "active"
        return {"user_id": user.id, "email": user.email, "display_name": user.display_name, "tenant_id": principal.tenant_id, "role": role}


def create_tenant(principal: Principal, name: str) -> dict:
    require(principal, "tenant:create")
    with session_scope() as session:
        tenant = TenantModel(id=new_id("tenant"), name=name.strip(), status="active")
        session.add(tenant)
        session.flush()
        workspace = WorkspaceModel(id=new_id("ws"), tenant_id=tenant.id, name="Default Workspace", status="active")
        membership = TenantMembershipModel(id=new_id("mem"), tenant_id=tenant.id, user_id=principal.user_id, role="TenantAdmin", status="active")
        session.add_all([workspace, membership])
        return {"id": tenant.id, "name": tenant.name, "status": tenant.status, "admin_user_id": principal.user_id}


def create_api_key(principal: Principal, name: str, scopes: list[str]) -> dict:
    require(principal, "credential:manage")
    prefix = f"stt_{token_urlsafe(6)}"
    value = f"{prefix}.{token_urlsafe(32)}"
    with session_scope() as session:
        row = ApiCredentialModel(id=new_id("key"), tenant_id=principal.tenant_id, user_id=principal.user_id, name=name, prefix=prefix, secret_hash=token_digest(value), scopes=scopes, status="active")
        session.add(row)
        return {"id": row.id, "name": name, "prefix": prefix, "api_key": value, "scopes": scopes}


def revoke_api_key(principal: Principal, credential_id: str) -> dict:
    require(principal, "credential:manage")
    with session_scope() as session:
        row = session.get(ApiCredentialModel, credential_id)
        if row is None or row.tenant_id != principal.tenant_id:
            from app.infra.persistence.sql.repositories import ResourceNotFoundError
            raise ResourceNotFoundError(f"API credential not found: {credential_id}")
        row.status = "revoked"
        return {"id": row.id, "status": row.status}


def revoke_refresh(refresh_token: str | None) -> None:
    if not refresh_token:
        return
    with session_scope() as session:
        row = session.scalar(select(RefreshCredentialModel).where(RefreshCredentialModel.token_hash.in_(token_digests(refresh_token))))
        if row and row.revoked_at is None:
            row.revoked_at = datetime.now(UTC)


def list_audit(principal: Principal, limit: int = 100) -> list[dict]:
    require(principal, "audit:read")
    with session_scope() as session:
        rows = session.scalars(select(AuditEventModel).where(AuditEventModel.tenant_id == principal.tenant_id).order_by(AuditEventModel.created_at.desc()).limit(min(max(limit, 1), 500))).all()
        return [{
            "id": r.id, "actor_id": r.actor_id, "action": r.action,
            "resource_type": r.resource_type, "resource_id": r.resource_id,
            "result": r.result, "request_id": r.request_id,
            "ip_address": r.ip_address, "detail": r.detail,
            "created_at": r.created_at.isoformat(),
        } for r in rows]


def register_query_session(principal: Principal, session_id: str, kb_id: str) -> dict:
    from app.infra.persistence.sql.repositories import ControlRepository, ResourceNotFoundError
    from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository
    from app.domain.query_security import RequestedRetrievalScope, ScopeAuthorizationError
    from app.application.query_authorization_service import QueryAuthorizationService
    from app.infra.security.sql_query_authorization import (
        SqlActiveIndexBindingResolver,
        SqlEntitlementResolver,
        SqlScopeDecisionAudit,
    )

    requested = RequestedRetrievalScope(kb_ids=(kb_id,))
    try:
        authorized_scope = QueryAuthorizationService(
            entitlements=SqlEntitlementResolver(),
            active_indexes=SqlActiveIndexBindingResolver(),
            audit=SqlScopeDecisionAudit(principal),
        ).authorize(
            principal=principal,
            requested=requested,
        )
    except ScopeAuthorizationError as exc:
        # 未授权与不存在使用同一响应，避免枚举其他租户的 KB。
        raise AuthorizationError("knowledge base is not authorized") from exc
    with session_scope() as session:
        kb = ControlRepository(session, tenant_id=principal.tenant_id).get_knowledge_base_model(
            kb_id
        )
        row = session.get(QuerySessionModel, session_id)
        if row is not None and (row.tenant_id != principal.tenant_id or row.user_id != principal.user_id):
            raise ResourceNotFoundError(f"query session not found: {session_id}")
        if row is None:
            row = QuerySessionModel(
                id=session_id, tenant_id=principal.tenant_id, user_id=principal.user_id,
                kb_id=kb_id, config_version=settings.config_version,
            )
            session.add(row)
            # sessionmaker 关闭了 autoflush；先落入 identity map，避免 profile resolver
            # 把同一 session 当作不存在而重复创建。
            session.flush()
        elif row.kb_id != kb_id:
            raise SessionKnowledgeBaseConflict("同一会话不能切换到其他知识库")
        retrieval_repository = RetrievalProfileRepository(
            session, tenant_id=principal.tenant_id, actor_id=principal.user_id
        )
        retrieval_profile = retrieval_repository.resolve_for_session(
            session_id,
            kb_id=kb_id,
            user_id=principal.user_id,
            config_version=row.config_version,
        )
        return {
            "session_id": session_id, "kb_id": kb_id,
            "web_search_allowed": kb.web_search_allowed,
            # 不在会话创建阶段阻断本地检索。生成/Embedding/Rerank/Web 的每一次
            # 实际外发都必须再消费 TODO-26 的 EgressPolicySnapshot。
            "external_llm_allowed": kb.external_llm_allowed,
            "data_classification": kb.data_classification,
            "authorization_policy_version": authorized_scope.policy_version,
            "entitlement_version": authorized_scope.entitlement_version,
            "authorized_scope": authorized_scope.model_dump(mode="json"),
            "config_version": row.config_version,
            "retrieval_profile": retrieval_repository.profile_dict(retrieval_profile),
        }


def assert_query_session_owner(principal: Principal, session_id: str) -> QuerySessionModel:
    from app.infra.persistence.sql.repositories import ResourceNotFoundError
    with session_scope() as session:
        row = session.get(QuerySessionModel, session_id)
        if row is None or row.tenant_id != principal.tenant_id or row.user_id != principal.user_id:
            raise ResourceNotFoundError(f"query session not found: {session_id}")
        session.expunge(row)
        return row
