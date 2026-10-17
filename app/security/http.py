import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.security.principal import reset_current_principal, set_current_principal
from app.security.service import (
    AuthenticationError, AuthorizationError, audit, authenticate_access,
    authenticate_api_key, require,
)

PUBLIC_PATHS = {
    "/", "/html", "/workbench/chat", "/workbench/retrieval",
    "/eval/html", "/p2/html", "/p3/html",
    "/health", "/health/live", "/health/ready",
    # logout 必须是幂等公开端点：access cookie 已过期时，浏览器仍需用
    # refresh cookie 撤销会话并清除两个 HttpOnly cookie。
    "/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/logout",
}


def _apply_browser_security_headers(response):
    """为内置验收页设置最低限度的浏览器执行与嵌入边界。"""

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'; connect-src 'self'; "
        "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' http: https:; font-src 'self';",
    )
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
    )
    return response


def action_for(method: str, path: str) -> str:
    if path == "/api/v1/ui/bootstrap":
        return "authenticated"
    if path.startswith("/api/v1/workbench"):
        # Workbench routes expose heterogeneous reads and body-selected batch
        # actions.  Their application service enforces the concrete permission
        # plus the current KB owner/ACL for every resource before pagination or
        # mutation; this outer gate only establishes an authenticated principal.
        return "authenticated"
    if path == "/health/dependencies":
        return "health:read"
    if path in {"/api/v1/auth/me", "/api/v1/auth/logout"}:
        return "authenticated"
    if path.startswith("/api/v1/admin/users"):
        return "member:manage"
    if path.startswith("/api/v1/admin/tenants"):
        return "tenant:create"
    if path.startswith("/api/v1/api-keys"):
        return "credential:manage"
    if path.startswith("/api/v1/audit-events"):
        return "audit:read"
    if path.startswith("/api/v1/p2"):
        preview_suffixes = (
            ":preview",
            ":dry-run",
            ":compile",
            ":execute",
            ":rank",
            ":inspect",
        )
        if method == "GET":
            return "p2:read"
        if (
            path == "/api/v1/p2/self-check"
            or path.endswith(preview_suffixes)
            or "/multimodal/images:inspect" in path
        ):
            return "p2:run"
        return "p2:manage"
    if path.startswith("/api/v1/p3"):
        if path.endswith("/observability"):
            return "p3:manage"
        if method == "GET":
            return "p3:read"
        if (
            path.endswith("/self-check")
            or path.endswith(":inspect")
            or path.endswith("/security/inspect")
        ):
            return "p3:run"
        return "p3:manage"
    if path.startswith("/api/v1/query-runs"):
        return "query:execute"
    if path.startswith("/api/v1/conversations"):
        if method == "DELETE" or path.endswith(":archive") or path.endswith(":restore"):
            return "history:delete"
        return "history:read"
    if path in {"/query", "/api/v1/query"}:
        return "query:execute"
    if path.startswith("/api/v1/evidence/"):
        return "document:read"
    if path.startswith("/api/v1/retrieval-profiles/"):
        return "kb:read" if method == "GET" else "kb:update"
    if re.match(
        r"^/api/v1/knowledge-bases/[^/]+/retrieval-profiles$", path
    ):
        return "kb:read" if method == "GET" else "kb:update"
    if path.endswith(":rollback-retrieval-profile"):
        return "kb:update"
    if path == "/api/v1/parsing/capabilities":
        return "document:read"
    if path == "/api/v1/parsing/preview":
        return "document:upload"
    if path.startswith("/api/v1/evaluation/datasets"):
        return "eval:manage" if method == "POST" else "eval:read"
    if path == "/api/v1/evaluation/preflight":
        return "eval:run"
    if path.startswith("/api/v1/evaluation/runs"):
        return "eval:run" if path == "/api/v1/evaluation/runs" and method == "POST" else "eval:read"
    if path.startswith("/api/v1/feedback"):
        return "feedback:create" if path == "/api/v1/feedback" and method == "POST" else "feedback:review"
    if path.startswith("/api/v1/annotation-tasks"):
        return "feedback:review"
    if path.startswith("/history/") or path.startswith("/api/v1/history/"):
        return "history:delete" if method == "DELETE" else "history:read"
    if path.startswith("/stream/") or path.startswith("/api/v1/stream/"):
        return "query:execute"
    if path == "/api/v1/knowledge-bases":
        return "kb:create" if method == "POST" else "kb:read"
    if re.match(r"^/api/v1/knowledge-bases/[^/:]+$", path):
        return {"GET": "kb:read", "PATCH": "kb:update", "DELETE": "kb:delete"}.get(method, "authenticated")
    if path.endswith(":rollback-index"):
        return "index:rollback"
    if re.match(r"^/api/v1/knowledge-bases/[^/]+/documents$", path) and method == "POST":
        return "document:upload"
    if re.match(r"^/api/v1/documents/[^/]+/versions$", path):
        return "document:upload"
    if path.startswith("/api/v1/document-versions/") and path.endswith("/ingestion-jobs"):
        return "job:manage"
    if path.startswith("/api/v1/documents/") and (path.endswith(":enable") or path.endswith(":disable")):
        return "document:delete"
    if "/documents/" in path and method == "DELETE":
        return "document:delete"
    if "/documents" in path:
        return "document:read"
    if re.match(r"^/api/v1/knowledge-bases/[^/]+/index-versions$", path):
        return "index:rebuild" if method == "POST" else "index:read"
    if path.startswith("/api/v1/index-versions/") and path.endswith(":publish"):
        return "index:publish"
    if path.startswith("/api/v1/index-versions/"):
        return "index:read"
    if "/indexes/rebuild" in path:
        return "index:rebuild"
    if "/publish" in path:
        return "index:publish"
    if "/rollback" in path:
        return "index:rollback"
    if "/indexes" in path:
        return "index:read"
    if path.startswith("/api/v1/jobs"):
        return "job:manage" if method in {"POST", "DELETE"} else "job:read"
    if path.startswith("/api/v1/workers"):
        return "job:read"
    if path.startswith("/api/v1/ingestion-jobs"):
        return "job:manage" if method == "POST" else "job:read"
    if path.startswith("/api/v1/ingestion-workers"):
        return "job:read"
    if path.startswith("/api/v1/cleanup-jobs"):
        return "cleanup:run" if method == "POST" else "cleanup:read"
    if path.startswith("/api/v1/cleanup") or path.endswith("/cleanup"):
        return "cleanup:run" if method == "POST" else "cleanup:read"
    if path in {"/upload"}:
        return "document:upload"
    if path.startswith("/status/"):
        return "job:read"
    return "__unmapped_business_action__"


def install_security(app: FastAPI) -> None:
    @app.middleware("http")
    async def authentication_boundary(request: Request, call_next):
        request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex}"
        request.state.request_id = request_id
        path = request.url.path.rstrip("/") or "/"
        if (
            path in PUBLIC_PATHS
            or path.startswith("/ui-assets/")
            or request.method == "OPTIONS"
            or path.startswith("/docs")
            or path.startswith("/openapi")
        ):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return _apply_browser_security_headers(response)

        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        cookie_token = request.cookies.get("stt_access_token", "")
        try:
            if api_key:
                principal = authenticate_api_key(api_key, request_id)
            else:
                token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else cookie_token
                if not token:
                    raise AuthenticationError("缺少访问凭证")
                principal = authenticate_access(token, request_id)
            action = action_for(request.method, path)
            require(principal, action)
        except AuthenticationError as exc:
            audit(None, "authentication", "denied", request_id, request.client.host if request.client else None, {"reason": str(exc), "path": path})
            return JSONResponse(status_code=401, content={"request_id": request_id, "error": {"code": "UNAUTHENTICATED", "message": str(exc)}}, headers={"WWW-Authenticate": "Bearer"})
        except AuthorizationError as exc:
            audit(principal, action, "denied", request_id, request.client.host if request.client else None, {"reason": str(exc), "path": path})
            return JSONResponse(status_code=403, content={"request_id": request_id, "error": {"code": "FORBIDDEN", "message": str(exc)}})

        request.state.principal = principal
        token = set_current_principal(principal)
        try:
            response = await call_next(request)
            audit(principal, action, "allowed" if response.status_code < 400 else "failed", request_id, request.client.host if request.client else None, {"path": path, "status_code": response.status_code})
            response.headers["X-Request-ID"] = request_id
            return _apply_browser_security_headers(response)
        finally:
            reset_current_principal(token)
