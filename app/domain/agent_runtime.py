"""TODO-21 Agent DSL、工具目录、审批与审计的稳定领域契约。

工具描述和工具输出均视为不可信数据。本模块不执行表达式、HTTP、MCP 或插件代码；
运行时只能使用编译后的拓扑顺序和组合根显式注入的 allowlisted executor。
"""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AGENT_DSL_SCHEMA_VERSION = "agent-dsl.v1"
TOOL_CONTRACT_SCHEMA_VERSION = "tool-contract.v1"
TOOL_AUDIT_SCHEMA_VERSION = "tool-audit.v1"

NodeKind = Literal[
    "begin",
    "retrieval",
    "llm",
    "condition",
    "approval",
    "tool",
    "answer",
]
ToolKind = Literal["deterministic", "http", "mcp"]
SideEffectLevel = Literal["none", "read", "write", "external_write"]
AgentVersionStatus = Literal["draft", "published", "retired"]
ToolExecutionStatus = Literal["succeeded", "replayed", "failed", "denied"]

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_BINDING = re.compile(
    r"^\$(?:input\.[A-Za-z][A-Za-z0-9_.-]{0,255}|"
    r"nodes\.[A-Za-z][A-Za-z0-9_.:-]{0,127}\.[A-Za-z][A-Za-z0-9_.-]{0,255})$"
)


def _safe_identifier(value: str, *, name: str) -> str:
    value = value.strip()
    if not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a safe identifier")
    return value


def _timezone(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value


class ToolReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)

    @field_validator("tool_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_identifier(value, name="tool_id")

    @property
    def key(self) -> str:
        return f"{self.tool_id}@{self.version}"


class AgentBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_nodes: int = Field(default=16, ge=2, le=256)
    max_tool_calls: int = Field(default=4, ge=0, le=64)
    max_llm_calls: int = Field(default=4, ge=0, le=64)
    max_total_tokens: int = Field(default=16_000, ge=0, le=10_000_000)
    timeout_ms: int = Field(default=120_000, ge=1, le=3_600_000)
    max_cost: float = Field(default=0.0, ge=0.0)


class ConditionRule(BaseModel):
    """声明式条件，不接受 Python/Jinja/JavaScript 表达式。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(pattern=_BINDING.pattern)
    operator: Literal["exists", "equals", "not_equals", "in"]
    value: str | int | float | bool | tuple[str | int | float | bool, ...] | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "ConditionRule":
        if self.operator == "exists" and self.value is not None:
            raise ValueError("exists condition cannot have a comparison value")
        if self.operator != "exists" and self.value is None:
            raise ValueError(f"{self.operator} condition requires value")
        if self.operator == "in" and not isinstance(self.value, tuple):
            raise ValueError("in condition requires an array value")
        return self


class AgentNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1, max_length=128)
    kind: NodeKind
    depends_on: tuple[str, ...] = Field(default_factory=tuple)
    input_bindings: dict[str, str] = Field(default_factory=dict)
    tool: ToolReference | None = None
    condition: ConditionRule | None = None
    idempotency_key_binding: str | None = Field(
        default=None,
        pattern=_BINDING.pattern,
    )

    @field_validator("node_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_identifier(value, name="node_id")

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on cannot contain duplicates")
        return value

    @field_validator("input_bindings")
    @classmethod
    def validate_bindings(cls, value: dict[str, str]) -> dict[str, str]:
        for key, binding in value.items():
            _safe_identifier(key, name="input binding name")
            if not _BINDING.fullmatch(binding):
                raise ValueError("input binding must use $input.* or $nodes.<id>.*")
        return value

    @model_validator(mode="after")
    def validate_kind_config(self) -> "AgentNode":
        if (self.kind == "tool") != (self.tool is not None):
            raise ValueError("only tool node must declare tool")
        if (self.kind == "condition") != (self.condition is not None):
            raise ValueError("only condition node must declare condition")
        if self.kind != "tool" and self.idempotency_key_binding is not None:
            raise ValueError("idempotency key binding only applies to tool nodes")
        if self.kind == "begin" and self.depends_on:
            raise ValueError("begin node cannot have dependencies")
        return self


class AgentVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent-dsl.v1"] = AGENT_DSL_SCHEMA_VERSION
    agent_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)
    tenant_id: str = Field(min_length=1, max_length=64)
    owner_id: str = Field(min_length=1, max_length=128)
    status: AgentVersionStatus = "draft"
    enabled: bool = False
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_permissions: tuple[str, ...] = Field(default_factory=tuple)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    nodes: tuple[AgentNode, ...] = Field(min_length=2, max_length=256)

    @field_validator("agent_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_identifier(value, name="agent_id")

    @field_validator("allowed_permissions")
    @classmethod
    def unique_permissions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_permissions cannot contain duplicates")
        return value


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=128)
    active_version: str | None = Field(default=None, pattern=_VERSION.pattern)
    enabled: bool = False


class ComponentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)
    kind: NodeKind
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_permissions: tuple[str, ...] = Field(default_factory=tuple)
    side_effect: SideEffectLevel = "none"


class ToolDefinition(BaseModel):
    """版本化工具契约。remote 类型也只描述边界，不赋予执行能力。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tool-contract.v1"] = TOOL_CONTRACT_SCHEMA_VERSION
    tool_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)
    tenant_id: str = Field(min_length=1, max_length=64)
    kind: ToolKind
    display_name: str = Field(min_length=1, max_length=256)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_permissions: tuple[str, ...] = Field(default_factory=tuple)
    side_effect: SideEffectLevel = "none"
    requires_approval: bool = False
    requires_idempotency_key: bool = False
    allowed_kb_ids: tuple[str, ...] = Field(default_factory=tuple)
    endpoint: str | None = Field(default=None, max_length=2048)
    allowed_hosts: tuple[str, ...] = Field(default_factory=tuple)
    credential_refs: tuple[str, ...] = Field(default_factory=tuple)
    timeout_ms: int = Field(default=10_000, ge=1, le=120_000)
    max_output_bytes: int = Field(default=64 * 1024, ge=1, le=4 * 1024 * 1024)
    plugin_id: str | None = Field(default=None, max_length=128)
    enabled: bool = False

    @field_validator("tool_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_identifier(value, name="tool_id")

    @field_validator(
        "required_permissions",
        "allowed_kb_ids",
        "allowed_hosts",
        "credential_refs",
    )
    @classmethod
    def unique_values(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} cannot contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_security_boundary(self) -> "ToolDefinition":
        mutating = self.side_effect in {"write", "external_write"}
        if mutating and not self.requires_approval:
            raise ValueError("mutating tool requires explicit approval")
        if mutating and not self.requires_idempotency_key:
            raise ValueError("mutating tool requires idempotency key")
        if self.kind == "deterministic":
            if self.endpoint or self.allowed_hosts or self.credential_refs:
                raise ValueError("deterministic tool cannot declare remote transport")
        else:
            if not self.endpoint:
                raise ValueError("HTTP/MCP tool requires endpoint")
            if not self.allowed_hosts:
                raise ValueError("HTTP/MCP tool requires a network allowlist")
        return self

    @property
    def reference(self) -> ToolReference:
        return ToolReference(tool_id=self.tool_id, version=self.version)


class McpServerDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)
    tenant_id: str = Field(min_length=1, max_length=64)
    transport: Literal["stdio", "streamable_http", "sse"]
    endpoint: str | None = Field(default=None, max_length=2048)
    allowed_hosts: tuple[str, ...] = Field(default_factory=tuple)
    credential_refs: tuple[str, ...] = Field(default_factory=tuple)
    pinned_tool_contracts: tuple[ToolReference, ...] = Field(default_factory=tuple)
    enabled: bool = False

    @model_validator(mode="after")
    def validate_transport(self) -> "McpServerDefinition":
        if self.transport == "stdio" and self.endpoint is not None:
            raise ValueError("stdio MCP server cannot declare a network endpoint")
        if self.transport != "stdio" and (not self.endpoint or not self.allowed_hosts):
            raise ValueError("network MCP server requires endpoint and allowed_hosts")
        return self


class PluginPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_VERSION.pattern)
    publisher: str = Field(min_length=1, max_length=256)
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    signature: str = Field(pattern=r"^[A-Za-z0-9+/=_-]{16,4096}$")
    permissions: tuple[str, ...] = Field(default_factory=tuple)
    compatible_agent_dsl: Literal["agent-dsl.v1"] = AGENT_DSL_SCHEMA_VERSION
    enabled: bool = False


class ToolLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: ToolReference
    contract_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CompiledAgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent-dsl.v1"] = AGENT_DSL_SCHEMA_VERSION
    agent_id: str
    agent_version: str
    tenant_id: str
    topological_order: tuple[str, ...] = Field(min_length=2)
    tool_locks: tuple[ToolLock, ...] = Field(default_factory=tuple)
    plan_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RuntimePrincipal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=64)
    permissions: tuple[str, ...] = Field(default_factory=tuple)
    kb_ids: tuple[str, ...] = Field(default_factory=tuple)


class ApprovalGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    tool: ToolReference
    principal_id: str = Field(min_length=1, max_length=128)
    argument_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    approved_at: datetime
    expires_at: datetime

    @field_validator("approved_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info) -> datetime:
        return _timezone(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_window(self) -> "ApprovalGrant":
        if self.expires_at <= self.approved_at:
            raise ValueError("approval expires_at must follow approved_at")
        return self


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    tool: ToolReference
    principal: RuntimePrincipal
    kb_id: str | None = Field(default=None, max_length=64)
    arguments: dict[str, Any]
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)
    approval: ApprovalGrant | None = None


class ToolAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tool-audit.v1"] = TOOL_AUDIT_SCHEMA_VERSION
    audit_id: str = Field(pattern=r"^ta_[0-9a-f]{32}$")
    run_id: str
    node_id: str
    trace_id: str
    principal_id: str
    tenant_id: str
    tool: ToolReference
    status: ToolExecutionStatus
    argument_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    idempotency_key_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    approval_id: str | None = None
    reason_code: str | None = Field(default=None, max_length=128)
    output_bytes: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _timezone(value, name="created_at")


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolExecutionStatus
    output: dict[str, Any] | None = None
    replayed: bool = False
    audit: ToolAuditRecord

    @model_validator(mode="after")
    def validate_replay(self) -> "ToolExecutionResult":
        if self.replayed != (self.status == "replayed"):
            raise ValueError("replayed flag must match replayed status")
        if self.status in {"succeeded", "replayed"} and self.output is None:
            raise ValueError("successful tool execution requires output")
        return self


__all__ = [
    "AGENT_DSL_SCHEMA_VERSION",
    "AgentBudget",
    "AgentDefinition",
    "AgentNode",
    "AgentVersion",
    "AgentVersionStatus",
    "ApprovalGrant",
    "CompiledAgentPlan",
    "ComponentDefinition",
    "ConditionRule",
    "McpServerDefinition",
    "NodeKind",
    "PluginPackage",
    "RuntimePrincipal",
    "SideEffectLevel",
    "ToolAuditRecord",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "ToolKind",
    "ToolLock",
    "ToolReference",
]
