"""TODO-21 的纯内存目录、幂等存储、审计汇与确定性 executor。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from app.domain.agent_runtime import (
    PluginPackage,
    ToolAuditRecord,
    ToolDefinition,
    ToolReference,
)


ToolHandler = Callable[[Mapping[str, Any]], dict[str, Any]]


class InMemoryToolCatalog:
    def __init__(self) -> None:
        self._lock = RLock()
        self._tools: dict[str, ToolDefinition] = {}
        self._plugins: dict[str, PluginPackage] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        with self._lock:
            self._tools[tool.reference.key] = tool

    def register_plugin(self, plugin: PluginPackage) -> None:
        with self._lock:
            self._plugins[plugin.plugin_id] = plugin

    def get_tool(self, reference: ToolReference) -> ToolDefinition | None:
        with self._lock:
            return self._tools.get(reference.key)

    def get_plugin(self, plugin_id: str) -> PluginPackage | None:
        with self._lock:
            return self._plugins.get(plugin_id)

    def set_tool_enabled(self, reference: ToolReference, enabled: bool) -> None:
        with self._lock:
            current = self._tools.get(reference.key)
            if current is None:
                raise KeyError(reference.key)
            self._tools[reference.key] = current.model_copy(update={"enabled": enabled})

    def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> None:
        with self._lock:
            current = self._plugins.get(plugin_id)
            if current is None:
                raise KeyError(plugin_id)
            self._plugins[plugin_id] = current.model_copy(update={"enabled": enabled})


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._values: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    def get(self, scope: str, key: str) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            return self._values.get((scope, key))

    def put(
        self,
        scope: str,
        key: str,
        argument_hash: str,
        output: dict[str, Any],
    ) -> None:
        with self._lock:
            identity = (scope, key)
            current = self._values.get(identity)
            if current is not None and current[0] != argument_hash:
                raise ValueError("idempotency key already belongs to different arguments")
            self._values[identity] = (argument_hash, output)


class InMemoryToolAuditSink:
    def __init__(self) -> None:
        self._lock = RLock()
        self.records: list[ToolAuditRecord] = []

    def record(self, audit: ToolAuditRecord) -> None:
        with self._lock:
            self.records.append(audit)


class DeterministicToolExecutor:
    """只执行精确注册的 Python handler；HTTP/MCP 永远不会被隐式调用。"""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self.call_count: dict[str, int] = {}

    def register(self, reference: ToolReference, handler: ToolHandler) -> None:
        self._handlers[reference.key] = handler

    def execute(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> dict[str, Any]:
        del trace_id
        handler = self._handlers.get(tool.reference.key)
        if handler is None:
            raise RuntimeError("tool has no explicitly registered deterministic handler")
        self.call_count[tool.reference.key] = self.call_count.get(tool.reference.key, 0) + 1
        return handler(arguments)


__all__ = [
    "DeterministicToolExecutor",
    "InMemoryIdempotencyStore",
    "InMemoryToolAuditSink",
    "InMemoryToolCatalog",
]
