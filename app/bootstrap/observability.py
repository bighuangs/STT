"""Phoenix 观测适配器的进程生命周期组合根。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from app.config.observability import PhoenixObservabilitySettings
from app.infrastructure.observability.phoenix import initialize, shutdown
from app.shared.runtime.logger import logger


@contextmanager
def observability_lifespan(role: str) -> Iterator[None]:
    """为 API/Worker 启停同一个 fail-open exporter，且不记录敏感配置。"""

    result = initialize(PhoenixObservabilitySettings())
    logger.info(
        "观测生命周期已进入：role={} state={} active={}",
        role,
        result.state.value,
        result.active,
    )
    try:
        yield
    finally:
        stopped = shutdown()
        logger.info(
            "观测生命周期已退出：role={} state={}",
            role,
            stopped.state.value,
        )


__all__ = ["observability_lifespan"]
