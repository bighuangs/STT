"""运行时健康、启动校验与资源关闭入口。"""
from app.infra.health.service import (
    HealthManager,
    close_runtime_resources,
    health_manager,
    validate_startup,
)

__all__ = ["HealthManager", "health_manager", "validate_startup", "close_runtime_resources"]
