"""进程入口的组合根；集中装配外层适配器与生命周期。"""

from app.bootstrap.observability import observability_lifespan

__all__ = ["observability_lifespan"]
