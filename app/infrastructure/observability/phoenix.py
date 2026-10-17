"""默认关闭且 fail-open 的 Phoenix/OpenTelemetry 生命周期适配器。"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable, Iterator, Protocol, cast

from app.config.observability import PhoenixObservabilitySettings


logger = logging.getLogger(__name__)


class _TracerProvider(Protocol):
    def shutdown(self) -> object:
        """刷新队列并停止 exporter。"""


_Register = Callable[..., _TracerProvider]


class PhoenixLifecycleState(str, Enum):
    """生命周期操作结果；不会携带 endpoint、header 或凭据。"""

    DISABLED = "disabled"
    INITIALIZED = "initialized"
    ALREADY_INITIALIZED = "already_initialized"
    DEPENDENCY_MISSING = "dependency_missing"
    INITIALIZATION_FAILED = "initialization_failed"
    SHUTDOWN = "shutdown"
    ALREADY_SHUTDOWN = "already_shutdown"
    SHUTDOWN_FAILED = "shutdown_failed"


@dataclass(frozen=True, slots=True)
class PhoenixLifecycleResult:
    """供 bootstrap 判断状态的脱敏结果。"""

    state: PhoenixLifecycleState
    active: bool
    error_type: str | None = None


_REDACTION_ENVIRONMENT = (
    "OPENINFERENCE_HIDE_INPUTS",
    "OPENINFERENCE_HIDE_OUTPUTS",
    "OPENINFERENCE_HIDE_INPUT_IMAGES",
    "OPENINFERENCE_HIDE_EMBEDDING_VECTORS",
    "OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS",
    "OPENINFERENCE_HIDE_LLM_PROMPTS",
    "OPENINFERENCE_HIDE_LLM_TOOLS",
)

_lock = RLock()
_provider: _TracerProvider | None = None


def _load_register() -> _Register:
    """延迟导入可选依赖，使禁用状态不要求安装 Phoenix SDK。"""
    from phoenix.otel import register

    return cast(_Register, register)


@contextmanager
def _redacted_instrumentation_environment() -> Iterator[None]:
    """只在 instrumentor 初始化时强制内容脱敏，并恢复进程原环境。"""
    missing = object()
    previous: dict[str, str | object] = {
        name: os.environ.get(name, missing) for name in _REDACTION_ENVIRONMENT
    }
    try:
        for name in _REDACTION_ENVIRONMENT:
            os.environ[name] = "true"
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = cast(str, value)


def initialize(
    settings: PhoenixObservabilitySettings | None = None,
) -> PhoenixLifecycleResult:
    """幂等初始化 Phoenix；任何配置、依赖或 SDK 错误都不会中断业务。"""
    global _provider

    with _lock:
        if _provider is not None:
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.ALREADY_INITIALIZED,
                active=True,
            )

        try:
            resolved = settings or PhoenixObservabilitySettings()
        except Exception as exc:
            logger.warning(
                "Phoenix 配置无效，已保持业务链路运行；error_type=%s",
                type(exc).__name__,
            )
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.INITIALIZATION_FAILED,
                active=False,
                error_type=type(exc).__name__,
            )

        if not resolved.enabled:
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.DISABLED,
                active=False,
            )

        try:
            register = _load_register()
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "Phoenix 可选依赖未安装，已跳过 trace 导出；error_type=%s",
                type(exc).__name__,
            )
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.DEPENDENCY_MISSING,
                active=False,
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            logger.warning(
                "Phoenix SDK 加载失败，已保持业务链路运行；error_type=%s",
                type(exc).__name__,
            )
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.INITIALIZATION_FAILED,
                active=False,
                error_type=type(exc).__name__,
            )

        api_key = resolved.api_key_value.get_secret_value() or None
        try:
            with _redacted_instrumentation_environment():
                provider = register(
                    endpoint=resolved.export_endpoint,
                    project_name=resolved.project_name,
                    protocol=resolved.protocol,
                    batch=resolved.batch,
                    auto_instrument=resolved.auto_instrument,
                    api_key=api_key,
                    set_global_tracer_provider=True,
                    verbose=False,
                )
        except Exception as exc:
            logger.warning(
                "Phoenix 初始化失败，已保持业务链路运行；error_type=%s",
                type(exc).__name__,
            )
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.INITIALIZATION_FAILED,
                active=False,
                error_type=type(exc).__name__,
            )

        _provider = provider
        logger.info("Phoenix trace 导出已启用")
        return PhoenixLifecycleResult(
            state=PhoenixLifecycleState.INITIALIZED,
            active=True,
        )


def shutdown() -> PhoenixLifecycleResult:
    """幂等停止 exporter；刷新失败只降级观测，不影响应用退出。"""
    global _provider

    with _lock:
        provider = _provider
        _provider = None
        if provider is None:
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.ALREADY_SHUTDOWN,
                active=False,
            )
        try:
            provider.shutdown()
        except Exception as exc:
            logger.warning(
                "Phoenix trace 刷新或关闭失败；error_type=%s",
                type(exc).__name__,
            )
            return PhoenixLifecycleResult(
                state=PhoenixLifecycleState.SHUTDOWN_FAILED,
                active=False,
                error_type=type(exc).__name__,
            )
        logger.info("Phoenix trace 导出已停止")
        return PhoenixLifecycleResult(
            state=PhoenixLifecycleState.SHUTDOWN,
            active=False,
        )


def is_initialized() -> bool:
    """返回当前进程是否持有已初始化 provider。"""
    with _lock:
        return _provider is not None
