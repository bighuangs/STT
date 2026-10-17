"""线程安全的惰性对象代理。"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Generic, TypeVar, cast


T = TypeVar("T")
_MISSING = object()


class LazyObject(Generic[T]):
    """首次使用时创建昂贵对象，并把后续属性访问转发给同一实例。"""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._value: T | object = _MISSING
        self._lock = RLock()

    def get(self) -> T:
        value = self._value
        if value is not _MISSING:
            return cast(T, value)
        with self._lock:
            if self._value is _MISSING:
                self._value = self._factory()
            return cast(T, self._value)

    def reset(self) -> None:
        """清除缓存；仅供测试或受控的进程内重装配使用。"""

        with self._lock:
            self._value = _MISSING

    def __getattr__(self, name: str):
        return getattr(self.get(), name)

    def __repr__(self) -> str:
        state = "ready" if self._value is not _MISSING else "pending"
        return f"{type(self).__name__}(state={state})"
