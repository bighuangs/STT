"""TODO-13 的线程安全内存 registry。

这是无持久化副作用的参考实现，供 dry-run、契约测试和后续 SQL repository
适配。注册信息不含任何密钥值。
"""

from __future__ import annotations

from threading import RLock

from app.domain.model_routing import (
    ModelHealth,
    ModelInstance,
    ProviderDefinition,
)


class InMemoryModelRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._providers: dict[str, ProviderDefinition] = {}
        self._models: dict[str, ModelInstance] = {}
        self._health: dict[str, ModelHealth] = {}

    def register_provider(self, provider: ProviderDefinition) -> None:
        with self._lock:
            self._providers[provider.provider_id] = provider

    def register_model(
        self,
        model: ModelInstance,
        *,
        health: ModelHealth | None = None,
    ) -> None:
        with self._lock:
            if model.provider_id not in self._providers:
                raise ValueError("model provider must be registered first")
            self._models[model.model_id] = model
            self._health[model.model_id] = health or ModelHealth(
                model_id=model.model_id,
                status="unknown",
            )

    def set_health(self, health: ModelHealth) -> None:
        with self._lock:
            if health.model_id not in self._models:
                raise KeyError(health.model_id)
            self._health[health.model_id] = health

    def get_provider(self, provider_id: str) -> ProviderDefinition | None:
        with self._lock:
            return self._providers.get(provider_id)

    def get_model(self, model_id: str) -> ModelInstance | None:
        with self._lock:
            return self._models.get(model_id)

    def get_health(self, model_id: str) -> ModelHealth:
        with self._lock:
            return self._health.get(model_id, ModelHealth(model_id=model_id))

    def list_models(self) -> tuple[ModelInstance, ...]:
        with self._lock:
            return tuple(self._models[key] for key in sorted(self._models))


__all__ = ["InMemoryModelRegistry"]
