"""统一 liveness/readiness/dependency health HTTP 协议。"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.infra.health import health_manager


def register_health_routes(app: FastAPI, role: str) -> None:
    @app.get("/health/live", tags=["health"])
    @app.get("/health", include_in_schema=False)
    def health_live():
        return health_manager.liveness(role)

    @app.get("/health/ready", tags=["health"])
    def health_ready():
        # 公共编排探针始终使用 TTL 缓存；只有管理员依赖接口允许 force。
        result = health_manager.check(role, force=False)
        public = {
            "status": result["status"], "ready": result["ready"], "role": role,
            "cached": result["cached"], "config_version": result["config_version"],
            "checked_at": result["checked_at"], "latency_ms": result["latency_ms"],
            "required_failed": result["required_failed"],
            "dependencies": {
                name: {
                    "status": item["status"], "required": item["required"],
                    "latency_ms": item["latency_ms"], "error_code": item.get("error_code"),
                }
                for name, item in result["dependencies"].items()
            },
        }
        return JSONResponse(status_code=200 if result["ready"] else 503, content=public)

    @app.get("/health/dependencies", tags=["health"])
    def health_dependencies(force: bool = False):
        result = health_manager.check(role, force=force, include_config=True)
        return JSONResponse(status_code=200 if result["ready"] else 503, content=result)


__all__ = ["register_health_routes"]
