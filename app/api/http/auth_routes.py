from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.security.service import (
    AuthenticationError, AuthorizationError, create_api_key, create_tenant, create_tenant_user,
    list_audit, login, principal_dict, refresh_tokens, revoke_api_key, revoke_refresh,
)


class LoginRequest(BaseModel):
    email: str
    password: str
    tenant_id: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class UserCreateRequest(BaseModel):
    email: str
    password: str = Field(min_length=12)
    display_name: str = ""
    role: str = "Viewer"


class ApiKeyCreateRequest(BaseModel):
    name: str = "web-validation"
    scopes: list[str] = Field(default_factory=list)


class TenantCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


def _cookies(response: Response, access_token: str, refresh_token: str) -> None:
    from app.shared.config.settings_config import settings
    secure = settings.app_env.lower() in {"prod", "production"}
    response.set_cookie("stt_access_token", access_token, httponly=True, secure=secure, samesite="strict", max_age=settings.auth_access_token_minutes * 60, path="/")
    response.set_cookie("stt_refresh_token", refresh_token, httponly=True, secure=secure, samesite="strict", max_age=settings.auth_refresh_token_days * 86400, path="/api/v1/auth")


def register_auth_routes(app: FastAPI) -> None:
    @app.post("/api/v1/auth/login", tags=["认证"])
    def login_api(payload: LoginRequest, response: Response):
        try:
            result, cookie = login(payload.email, payload.password, payload.tenant_id)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _cookies(response, cookie, result["refresh_token"])
        return result

    @app.post("/api/v1/auth/refresh", tags=["认证"])
    def refresh_api(payload: RefreshRequest, request: Request, response: Response):
        token = payload.refresh_token or request.cookies.get("stt_refresh_token", "")
        if not token:
            raise HTTPException(status_code=401, detail="缺少刷新令牌")
        try:
            result, cookie = refresh_tokens(token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _cookies(response, cookie, result["refresh_token"])
        return result

    @app.get("/api/v1/auth/me", tags=["认证"])
    def me(request: Request):
        return principal_dict(request.state.principal)

    @app.post("/api/v1/auth/logout", tags=["认证"])
    def logout(request: Request, response: Response):
        revoke_refresh(request.cookies.get("stt_refresh_token"))
        response.delete_cookie("stt_access_token", path="/")
        response.delete_cookie("stt_refresh_token", path="/api/v1/auth")
        return {"message": "已退出当前浏览器会话"}

    @app.post("/api/v1/admin/users", tags=["租户管理"])
    def create_user(payload: UserCreateRequest, request: Request):
        try:
            return create_tenant_user(request.state.principal, payload.email, payload.password, payload.display_name, payload.role)
        except (AuthorizationError, ValueError) as exc:
            raise HTTPException(status_code=403 if isinstance(exc, AuthorizationError) else 422, detail=str(exc)) from exc

    @app.post("/api/v1/admin/tenants", tags=["租户管理"])
    def create_tenant_api(payload: TenantCreateRequest, request: Request):
        return create_tenant(request.state.principal, payload.name)

    @app.post("/api/v1/api-keys", tags=["认证"])
    def create_key(payload: ApiKeyCreateRequest, request: Request):
        return create_api_key(request.state.principal, payload.name, payload.scopes)

    @app.delete("/api/v1/api-keys/{credential_id}", tags=["认证"])
    def revoke_key(credential_id: str, request: Request):
        return revoke_api_key(request.state.principal, credential_id)

    @app.get("/api/v1/audit-events", tags=["审计"])
    def audit_events(request: Request, limit: int = 100):
        return {"items": list_audit(request.state.principal, limit)}
