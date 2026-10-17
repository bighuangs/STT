"""SQL 持久化适配器共享的稳定原语。

该模块只保存跨仓储使用的资源错误、默认资源 ID 与 ID 生成函数，避免大型仓储
门面互相导入。旧的 ``repositories`` 模块继续重导出这些符号，现有调用方无需迁移。
"""

from __future__ import annotations

from uuid import uuid4


DEFAULT_TENANT_ID = "tenant_default"
DEFAULT_WORKSPACE_ID = "workspace_default"
DEFAULT_KB_ID = "kb_default"


class ResourceNotFoundError(LookupError):
    """租户作用域内不存在目标资源。"""


class ResourceConflictError(ValueError):
    """目标资源当前状态与请求操作冲突。"""


def new_id(prefix: str) -> str:
    """生成带稳定类型前缀的随机资源 ID。"""

    return f"{prefix}_{uuid4().hex}"


__all__ = [
    "DEFAULT_KB_ID",
    "DEFAULT_TENANT_ID",
    "DEFAULT_WORKSPACE_ID",
    "ResourceConflictError",
    "ResourceNotFoundError",
    "new_id",
]
