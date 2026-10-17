"""Vector GC / migration 的窄 Worker dispatch。

本文件刻意不提供 ``main()``，也不从环境变量自动打开破坏性权限。部署必须显式
构造 ``VectorMaintenanceWorkerHandler`` 并注入经过审批的 gateway。
"""

from __future__ import annotations

from typing import Any, Literal

from app.application.vector_maintenance_service import (
    VectorMaintenanceWorkerHandler,
)


def handle_vector_maintenance_job(
    handler: VectorMaintenanceWorkerHandler,
    payload: dict[str, Any],
    *,
    owner_id: str,
) -> dict[str, str]:
    kind: (
        Literal[
            "vector_gc",
            "vector_migration_backfill",
            "vector_migration_target_rollback",
        ]
        | str
    ) = str(payload.get("kind") or "")
    run_id = str(payload.get("run_id") or "").strip()
    if not owner_id.strip() or not run_id:
        raise ValueError("vector maintenance job requires owner_id and run_id")
    if kind == "vector_gc":
        status = handler.run_gc(run_id, owner_id=owner_id)
    elif kind == "vector_migration_backfill":
        status = handler.run_migration_backfill(run_id, owner_id=owner_id)
    elif kind == "vector_migration_target_rollback":
        status = handler.rollback_unpublished_target(run_id, owner_id=owner_id)
    else:
        raise ValueError("unsupported vector maintenance job kind")
    return {"kind": kind, "run_id": run_id, "status": status.value}


__all__ = ["handle_vector_maintenance_job"]
