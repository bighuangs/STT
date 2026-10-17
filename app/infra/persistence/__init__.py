"""
应用主包 / 基础设施层 / 持久化子模块的初始化文件，用于声明包边界与导出约定。
"""
__all__ = ["history_repository"]


def __getattr__(name: str):
    """避免加载 SQL 控制库时意外初始化 MongoDB 历史连接。"""
    if name == "history_repository":
        from app.infra.persistence.history_repository import history_repository

        return history_repository
    raise AttributeError(name)
