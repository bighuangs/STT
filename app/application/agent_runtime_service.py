"""TODO-21 Agent DSL 静态编译器与集中式 Tool Gateway。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import ipaddress
import json
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from app.domain.agent_runtime import (
    AgentNode,
    AgentVersion,
    CompiledAgentPlan,
    PluginPackage,
    ToolAuditRecord,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolLock,
    ToolReference,
)


class AgentCompileError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class ToolPolicyError(RuntimeError):
    def __init__(self, code: str, message: str, *, audit: ToolAuditRecord) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.audit = audit


class ToolCatalogPort(Protocol):
    def get_tool(self, reference: ToolReference) -> ToolDefinition | None: ...

    def get_plugin(self, plugin_id: str) -> PluginPackage | None: ...


class ToolExecutorPort(Protocol):
    def execute(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> dict[str, Any]: ...


class ToolAuditSinkPort(Protocol):
    def record(self, audit: ToolAuditRecord) -> None: ...


class IdempotencyStorePort(Protocol):
    def get(self, scope: str, key: str) -> tuple[str, dict[str, Any]] | None: ...

    def put(
        self,
        scope: str,
        key: str,
        argument_hash: str,
        output: dict[str, Any],
    ) -> None: ...


_ALLOWED_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "description",
}
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|token|api[_-]?key|credential)(?:$|[_-])",
    re.IGNORECASE,
)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON data") from exc


def _digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()}"


def validate_schema_definition(schema: dict[str, Any], *, path: str = "$") -> None:
    """校验无 ``$ref``/组合器/正则的安全 JSON Schema 子集。"""

    if not isinstance(schema, dict):
        raise ValueError(f"{path} schema must be an object")
    unknown = set(schema) - _ALLOWED_SCHEMA_KEYS
    if unknown:
        raise ValueError(f"{path} schema contains unsupported keywords: {sorted(unknown)}")
    schema_type = schema.get("type")
    if schema_type not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError(f"{path} schema type is unsupported")
    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"{path} object schema requires properties")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{path} object schema must set additionalProperties=false")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ValueError(f"{path} required must be a string array")
        if not set(required).issubset(properties):
            raise ValueError(f"{path} required references an unknown property")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path} property names must be non-empty strings")
            validate_schema_definition(child, path=f"{path}.{name}")
    elif schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"{path} array schema requires items")
        validate_schema_definition(schema["items"], path=f"{path}[]")
    elif "properties" in schema or "items" in schema:
        raise ValueError(f"{path} non-container schema cannot declare properties/items")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise ValueError(f"{path} enum must be a non-empty array")


def validate_json_value(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    validate_schema_definition(schema, path=f"{path}#schema")
    expected = schema["type"]
    valid = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }[expected](value)
    if not valid:
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is outside enum")
    if expected == "object":
        properties = schema["properties"]
        extra = set(value) - set(properties)
        missing = set(schema.get("required", [])) - set(value)
        if extra:
            raise ValueError(f"{path} contains unknown properties: {sorted(extra)}")
        if missing:
            raise ValueError(f"{path} misses required properties: {sorted(missing)}")
        for name, child in value.items():
            validate_json_value(child, properties[name], path=f"{path}.{name}")
    elif expected == "array":
        minimum = int(schema.get("minItems", 0))
        maximum = int(schema.get("maxItems", 1_000_000))
        if not minimum <= len(value) <= maximum:
            raise ValueError(f"{path} array length is out of bounds")
        for index, child in enumerate(value):
            validate_json_value(child, schema["items"], path=f"{path}[{index}]")
    elif expected == "string":
        minimum = int(schema.get("minLength", 0))
        maximum = int(schema.get("maxLength", 1_000_000))
        if not minimum <= len(value) <= maximum:
            raise ValueError(f"{path} string length is out of bounds")
    elif expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")
    _canonical(value)


def _binding_source(binding: str) -> str | None:
    if not binding.startswith("$nodes."):
        return None
    return binding[len("$nodes.") :].split(".", 1)[0]


class AgentCompiler:
    """执行前完成类型、拓扑、预算、权限、审批和工具锁定检查。"""

    def __init__(self, catalog: ToolCatalogPort) -> None:
        self._catalog = catalog

    def compile(self, version: AgentVersion) -> CompiledAgentPlan:
        validate_schema_definition(version.input_schema, path="$input")
        validate_schema_definition(version.output_schema, path="$output")
        nodes = {node.node_id: node for node in version.nodes}
        if len(nodes) != len(version.nodes):
            raise AgentCompileError("DUPLICATE_NODE", "agent node_id must be unique")
        begins = [node for node in version.nodes if node.kind == "begin"]
        answers = [node for node in version.nodes if node.kind == "answer"]
        if len(begins) != 1 or len(answers) != 1:
            raise AgentCompileError(
                "TERMINAL_SHAPE_INVALID",
                "agent requires exactly one begin and one answer",
            )
        if len(nodes) > version.budget.max_nodes:
            raise AgentCompileError("NODE_BUDGET_EXCEEDED", "agent exceeds max_nodes")

        indegree = {node_id: 0 for node_id in nodes}
        children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for node in version.nodes:
            for dependency in node.depends_on:
                if dependency not in nodes:
                    raise AgentCompileError(
                        "UNKNOWN_DEPENDENCY",
                        f"node {node.node_id} depends on unknown node {dependency}",
                    )
                if dependency == node.node_id:
                    raise AgentCompileError("CYCLE", "node cannot depend on itself")
                indegree[node.node_id] += 1
                children[dependency].append(node.node_id)

        queue = deque(node.node_id for node in version.nodes if indegree[node.node_id] == 0)
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for child in children[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(nodes):
            raise AgentCompileError("CYCLE", "agent graph must be acyclic")
        if order[0] != begins[0].node_id:
            raise AgentCompileError("MULTIPLE_ROOTS", "begin must be the only graph root")

        answer_ancestors = self._ancestors(answers[0].node_id, nodes)
        if answer_ancestors | {answers[0].node_id} != set(nodes):
            raise AgentCompileError(
                "UNREACHABLE_NODE",
                "every node must contribute to the answer terminal",
            )

        allowed_permissions = set(version.allowed_permissions)
        tool_locks: list[ToolLock] = []
        tool_count = sum(node.kind == "tool" for node in version.nodes)
        llm_count = sum(node.kind == "llm" for node in version.nodes)
        if tool_count > version.budget.max_tool_calls:
            raise AgentCompileError("TOOL_BUDGET_EXCEEDED", "agent exceeds max_tool_calls")
        if llm_count > version.budget.max_llm_calls:
            raise AgentCompileError("LLM_BUDGET_EXCEEDED", "agent exceeds max_llm_calls")

        for node in version.nodes:
            ancestors = self._ancestors(node.node_id, nodes)
            bindings = [*node.input_bindings.values()]
            if node.condition is not None:
                bindings.append(node.condition.source)
            if node.idempotency_key_binding is not None:
                bindings.append(node.idempotency_key_binding)
            for binding in bindings:
                source = _binding_source(binding)
                if source is not None and source not in ancestors:
                    raise AgentCompileError(
                        "INVALID_BINDING_SCOPE",
                        f"node {node.node_id} reads non-ancestor node {source}",
                    )
            if node.tool is None:
                continue
            tool = self._catalog.get_tool(node.tool)
            if tool is None:
                raise AgentCompileError("TOOL_NOT_FOUND", f"tool {node.tool.key} not found")
            validate_schema_definition(tool.input_schema, path=f"${node.node_id}.input")
            validate_schema_definition(tool.output_schema, path=f"${node.node_id}.output")
            if not tool.enabled:
                raise AgentCompileError("TOOL_DISABLED", f"tool {node.tool.key} is disabled")
            if tool.tenant_id != version.tenant_id:
                raise AgentCompileError("TOOL_TENANT_DENIED", "tool belongs to another tenant")
            if not set(tool.required_permissions).issubset(allowed_permissions):
                raise AgentCompileError(
                    "TOOL_PERMISSION_UNDECLARED",
                    f"agent does not declare all permissions for {node.tool.key}",
                )
            if tool.plugin_id:
                plugin = self._catalog.get_plugin(tool.plugin_id)
                if plugin is None or not plugin.enabled:
                    raise AgentCompileError(
                        "PLUGIN_DISABLED",
                        f"plugin for {node.tool.key} is disabled",
                    )
            if tool.side_effect in {"write", "external_write"}:
                if node.idempotency_key_binding is None:
                    raise AgentCompileError(
                        "IDEMPOTENCY_REQUIRED",
                        f"mutating tool node {node.node_id} needs idempotency binding",
                    )
                if not any(nodes[item].kind == "approval" for item in ancestors):
                    raise AgentCompileError(
                        "APPROVAL_REQUIRED",
                        f"mutating tool node {node.node_id} needs an approval ancestor",
                    )
            contract_hash = _digest(tool.model_dump(mode="json"))
            tool_locks.append(ToolLock(reference=tool.reference, contract_hash=contract_hash))

        payload = {
            "version": version.model_dump(mode="json"),
            "order": order,
            "tool_locks": [item.model_dump(mode="json") for item in tool_locks],
        }
        return CompiledAgentPlan(
            agent_id=version.agent_id,
            agent_version=version.version,
            tenant_id=version.tenant_id,
            topological_order=tuple(order),
            tool_locks=tuple(tool_locks),
            plan_hash=_digest(payload),
        )

    @staticmethod
    def _ancestors(node_id: str, nodes: dict[str, AgentNode]) -> set[str]:
        found: set[str] = set()
        stack = list(nodes[node_id].depends_on)
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(nodes[current].depends_on)
        return found


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _SECRET_KEY.search(str(key)) or _contains_secret_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(child) for child in value)
    return False


def _validate_remote_endpoint(tool: ToolDefinition) -> None:
    if tool.kind == "deterministic":
        return
    parts = urlsplit(tool.endpoint or "")
    hostname = (parts.hostname or "").lower()
    allowlist = {item.lower() for item in tool.allowed_hosts}
    if parts.scheme != "https" or not hostname or hostname not in allowlist:
        raise ValueError("remote endpoint must use HTTPS and an exact allowlisted host")
    if hostname == "localhost":
        raise ValueError("localhost is forbidden for remote tools")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    ):
        raise ValueError("private or special IP endpoints are forbidden")


class ToolGateway:
    """权限、审批、幂等、schema、SSRF、大小、审计的单一执行边界。"""

    def __init__(
        self,
        *,
        catalog: ToolCatalogPort,
        executor: ToolExecutorPort,
        audit_sink: ToolAuditSinkPort,
        idempotency_store: IdempotencyStorePort,
    ) -> None:
        self._catalog = catalog
        self._executor = executor
        self._audit_sink = audit_sink
        self._idempotency_store = idempotency_store

    def execute(self, call: ToolCall) -> ToolExecutionResult:
        try:
            argument_hash = _digest(call.arguments)
        except ValueError:
            argument_hash = _digest({"invalid": True})
            self._deny(call, argument_hash, "ARGUMENTS_NOT_JSON")
        tool = self._catalog.get_tool(call.tool)
        if tool is None:
            self._deny(call, argument_hash, "TOOL_NOT_FOUND")
        assert tool is not None
        if not tool.enabled:
            self._deny(call, argument_hash, "TOOL_DISABLED")
        if tool.tenant_id != call.principal.tenant_id:
            self._deny(call, argument_hash, "TENANT_SCOPE_DENIED")
        if not set(tool.required_permissions).issubset(call.principal.permissions):
            self._deny(call, argument_hash, "PERMISSION_DENIED")
        if call.kb_id is not None:
            if call.kb_id not in call.principal.kb_ids:
                self._deny(call, argument_hash, "KB_SCOPE_DENIED")
            if tool.allowed_kb_ids and call.kb_id not in tool.allowed_kb_ids:
                self._deny(call, argument_hash, "TOOL_KB_SCOPE_DENIED")
        if tool.plugin_id:
            plugin = self._catalog.get_plugin(tool.plugin_id)
            if plugin is None or not plugin.enabled:
                self._deny(call, argument_hash, "PLUGIN_DISABLED")
        try:
            _validate_remote_endpoint(tool)
            validate_json_value(call.arguments, tool.input_schema)
        except ValueError as exc:
            self._deny(call, argument_hash, f"INPUT_REJECTED:{type(exc).__name__}")
        if _contains_secret_key(call.arguments):
            self._deny(call, argument_hash, "SECRET_ARGUMENT_DENIED")
        if tool.side_effect in {"write", "external_write"}:
            if not call.idempotency_key:
                self._deny(call, argument_hash, "IDEMPOTENCY_KEY_REQUIRED")
            self._validate_approval(call, tool, argument_hash)

        scope = f"{call.principal.tenant_id}:{call.tool.key}:{call.principal.principal_id}"
        if call.idempotency_key:
            cached = self._idempotency_store.get(scope, call.idempotency_key)
            if cached is not None:
                previous_hash, output = cached
                if previous_hash != argument_hash:
                    self._deny(call, argument_hash, "IDEMPOTENCY_KEY_CONFLICT")
                audit = self._audit(
                    call,
                    argument_hash,
                    status="replayed",
                    output=output,
                    approval_id=call.approval.approval_id if call.approval else None,
                )
                return ToolExecutionResult(
                    status="replayed",
                    output=output,
                    replayed=True,
                    audit=audit,
                )

        try:
            output = self._executor.execute(
                tool,
                call.arguments,
                trace_id=call.trace_id,
            )
            validate_json_value(output, tool.output_schema)
            if _contains_secret_key(output):
                raise ValueError("tool output contains secret-shaped fields")
            encoded = _canonical(output).encode("utf-8")
            if len(encoded) > tool.max_output_bytes:
                raise ValueError("tool output exceeds max_output_bytes")
        except Exception as exc:
            audit = self._audit(
                call,
                argument_hash,
                status="failed",
                reason_code=f"EXECUTION_FAILED:{type(exc).__name__}",
                approval_id=call.approval.approval_id if call.approval else None,
            )
            return ToolExecutionResult(status="failed", audit=audit)

        if call.idempotency_key:
            self._idempotency_store.put(
                scope,
                call.idempotency_key,
                argument_hash,
                output,
            )
        audit = self._audit(
            call,
            argument_hash,
            status="succeeded",
            output=output,
            approval_id=call.approval.approval_id if call.approval else None,
        )
        return ToolExecutionResult(status="succeeded", output=output, audit=audit)

    def _validate_approval(
        self,
        call: ToolCall,
        tool: ToolDefinition,
        argument_hash: str,
    ) -> None:
        approval = call.approval
        if approval is None:
            self._deny(call, argument_hash, "APPROVAL_REQUIRED")
        assert approval is not None
        now = datetime.now(UTC)
        if approval.expires_at <= now:
            self._deny(call, argument_hash, "APPROVAL_EXPIRED")
        expected = (
            approval.run_id == call.run_id
            and approval.node_id == call.node_id
            and approval.tool == tool.reference
            and approval.principal_id == call.principal.principal_id
            and approval.argument_hash == argument_hash
        )
        if not expected:
            self._deny(call, argument_hash, "APPROVAL_SCOPE_MISMATCH")

    def _deny(self, call: ToolCall, argument_hash: str, reason_code: str) -> None:
        audit = self._audit(
            call,
            argument_hash,
            status="denied",
            reason_code=reason_code,
            approval_id=call.approval.approval_id if call.approval else None,
        )
        raise ToolPolicyError(reason_code, "tool call denied by runtime policy", audit=audit)

    def _audit(
        self,
        call: ToolCall,
        argument_hash: str,
        *,
        status: str,
        output: dict[str, Any] | None = None,
        reason_code: str | None = None,
        approval_id: str | None = None,
    ) -> ToolAuditRecord:
        output_hash = _digest(output) if output is not None else None
        output_bytes = len(_canonical(output).encode("utf-8")) if output is not None else 0
        key_hash = _digest(call.idempotency_key) if call.idempotency_key else None
        identity = {
            "run_id": call.run_id,
            "node_id": call.node_id,
            "trace_id": call.trace_id,
            "tool": call.tool.model_dump(mode="json"),
            "status": status,
            "argument_hash": argument_hash,
            "output_hash": output_hash,
            "reason": reason_code,
        }
        audit = ToolAuditRecord(
            audit_id=f"ta_{_digest(identity).removeprefix('sha256:')[:32]}",
            run_id=call.run_id,
            node_id=call.node_id,
            trace_id=call.trace_id,
            principal_id=call.principal.principal_id,
            tenant_id=call.principal.tenant_id,
            tool=call.tool,
            status=status,
            argument_hash=argument_hash,
            output_hash=output_hash,
            idempotency_key_hash=key_hash,
            approval_id=approval_id,
            reason_code=reason_code,
            output_bytes=output_bytes,
        )
        self._audit_sink.record(audit)
        return audit


def argument_hash(arguments: dict[str, Any]) -> str:
    """审批服务生成 grant 时使用，避免复制网关的规范化算法。"""

    return _digest(arguments)


__all__ = [
    "AgentCompileError",
    "AgentCompiler",
    "IdempotencyStorePort",
    "ToolAuditSinkPort",
    "ToolCatalogPort",
    "ToolExecutorPort",
    "ToolGateway",
    "ToolPolicyError",
    "argument_hash",
    "validate_json_value",
    "validate_schema_definition",
]
