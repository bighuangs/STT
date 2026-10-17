"""安全审计事件的 SQL 写入适配器。"""

from __future__ import annotations

from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import AuditEventModel
from app.infra.persistence.sql.primitives import new_id
from app.security.principal import Principal
from app.shared.runtime.logger import logger


def audit(
    principal: Principal | None,
    action: str,
    result: str,
    request_id: str,
    ip_address: str | None,
    detail: dict | None = None,
) -> None:
    """以 fail-open 方式写入审计事件，但保留可诊断的失败信号。"""

    try:
        safe_detail = detail or {}
        path = str(safe_detail.get("path", ""))
        parts = [part for part in path.split("/") if part]
        resource_type = (
            parts[2]
            if len(parts) > 2 and parts[:2] == ["api", "v1"]
            else (parts[0] if parts else None)
        )
        resource_id = parts[-1].split(":", 1)[0] if len(parts) > 3 else None
        with session_scope() as session:
            session.add(
                AuditEventModel(
                    id=new_id("aud"),
                    tenant_id=principal.tenant_id if principal else None,
                    actor_id=principal.user_id if principal else None,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    result=result,
                    request_id=request_id,
                    ip_address=ip_address,
                    detail=safe_detail,
                )
            )
    except Exception as exc:  # 审计不可阻断请求，但不能静默吞掉基础设施故障。
        logger.warning("安全审计写入失败：{}", type(exc).__name__)


__all__ = ["audit"]
