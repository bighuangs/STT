"""统一前端工作台的只读启动上下文与静态资源路由。

前端不再要求操作者手工输入原始知识库 ID。该模块从认证主体、租户事实和
知识库 ACL 计算可见知识库，并只返回页面完成业务选择所需的脱敏摘要。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.application.control_plane_service import (
    list_documents,
    list_knowledge_bases,
)
from app.domain.query_security import RequestedRetrievalScope, ScopeAuthorizationError
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import TenantModel, UserModel
from app.infra.security.sql_query_authorization import SqlEntitlementResolver
from app.security.permissions import effective_permissions
from app.security.principal import Principal
from app.shared.runtime.logger import PROJECT_ROOT


UI_ASSET_ROOT = PROJECT_ROOT / "app" / "process" / "shared" / "page"
UI_ASSETS = {
    "stt-console.css": ("text/css; charset=utf-8", UI_ASSET_ROOT / "stt-console.css"),
    "stt-console.js": (
        "application/javascript; charset=utf-8",
        UI_ASSET_ROOT / "stt-console.js",
    ),
    "stt-diagnostics.js": (
        "application/javascript; charset=utf-8",
        UI_ASSET_ROOT / "stt-diagnostics.js",
    ),
    "lucide-nav.svg": (
        "image/svg+xml",
        UI_ASSET_ROOT / "vendor" / "lucide" / "lucide-nav.svg",
    ),
}


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise HTTPException(status_code=401, detail="缺少有效的登录会话")
    return principal


def _identity(principal: Principal) -> dict:
    with session_scope() as session:
        user = session.get(UserModel, principal.user_id)
        tenant = session.get(TenantModel, principal.tenant_id)
    return {
        "user_id": principal.user_id,
        "email": user.email if user is not None else "",
        "display_name": user.display_name if user is not None else "",
        "tenant_id": principal.tenant_id,
        "tenant_name": tenant.name if tenant is not None else principal.tenant_id,
        "roles": list(principal.roles),
        "auth_type": principal.auth_type,
    }


def _authorized_kb_grant(principal: Principal, kb_id: str) -> dict | None:
    try:
        snapshot = SqlEntitlementResolver().resolve(
            principal=principal,
            requested=RequestedRetrievalScope(kb_ids=(kb_id,)),
        )
    except ScopeAuthorizationError:
        return None
    if not snapshot.grants:
        return None
    grant = snapshot.grants[0]
    return {
        "source": grant.grant_source,
        "role": grant.grant_role,
        "entitlement_version": snapshot.entitlement_version,
    }


def _knowledge_bases(principal: Principal) -> list[dict]:
    output: list[dict] = []
    for kb in list_knowledge_bases():
        if kb.get("tenant_id") != principal.tenant_id or kb.get("status") != "active":
            continue
        grant = _authorized_kb_grant(principal, str(kb["id"]))
        if grant is None:
            continue
        documents = list_documents(str(kb["id"]))
        queryable_count = sum(
            1
            for document in documents
            if document.get("status") in {"ready", "processing"}
            and document.get("active_version_id")
            and document.get("active_index_version_id")
        )
        processing_count = sum(
            1
            for document in documents
            if document.get("status") in {"draft", "uploaded", "processing", "indexing"}
            or (document.get("latest_job") or {}).get("status")
            in {"queued", "leased", "running", "retry_wait", "processing"}
        )
        active_index = kb.get("active_index_version_id")
        if queryable_count and active_index:
            readiness = "ready"
            readiness_message = "可查询（有文档正在更新）" if processing_count else "可查询"
        elif processing_count:
            readiness = "processing"
            readiness_message = "文档处理中"
        elif not documents:
            readiness = "empty"
            readiness_message = "尚未导入文档"
        elif not active_index:
            readiness = "blocked"
            readiness_message = "缺少已发布索引"
        else:
            readiness = "blocked"
            readiness_message = "暂无可查询文档"
        output.append(
            {
                "id": kb["id"],
                "name": kb["name"],
                "description": kb.get("description") or "",
                "status": kb["status"],
                "data_classification": kb.get("data_classification") or "internal",
                "document_count": len(documents),
                # v1 bootstrap 字段保留兼容，但其产品含义一直是“可查询文档数”；
                # 另给出显式字段，避免调用方再用 Document.status 猜测。
                "ready_document_count": queryable_count,
                "queryable_document_count": queryable_count,
                "processing_document_count": processing_count,
                "active_index_version_id": active_index,
                "external_llm_allowed": bool(kb.get("external_llm_allowed")),
                "web_search_allowed": bool(kb.get("web_search_allowed")),
                "readiness": readiness,
                "readiness_message": readiness_message,
                "access": grant,
            }
        )
    return output


def register_ui_routes(
    app: FastAPI,
    service: Literal["import", "query"],
) -> None:
    """为导入和查询进程注册同一套 UI 资源与认证后启动上下文。"""

    @app.get("/ui-assets/{asset_name}", include_in_schema=False)
    def ui_asset(asset_name: str):
        asset = UI_ASSETS.get(asset_name)
        if asset is None:
            raise HTTPException(status_code=404, detail="UI 资源不存在")
        media_type, path = asset
        if not Path(path).is_file():
            raise HTTPException(status_code=404, detail="UI 资源尚未构建")
        return FileResponse(path=path, media_type=media_type)

    @app.get("/api/v1/ui/bootstrap", tags=["前端工作台"])
    def ui_bootstrap(request: Request):
        principal = _principal(request)
        permissions = effective_permissions(principal.roles, principal.scopes)
        return {
            "request_id": getattr(request.state, "request_id", ""),
            "schema_version": "stt.ui.bootstrap.v1",
            "service": service,
            "identity": _identity(principal),
            "permissions": sorted(permissions),
            "knowledge_bases": _knowledge_bases(principal),
        }
