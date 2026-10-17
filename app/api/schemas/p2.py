"""P2 高级能力控制面与验收 API schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.p2_control_service import validate_capability_resource_scope
from app.domain.advanced_retrieval import (
    AdvancedIndexSnapshot,
    AdvancedRetrievalRequest,
)
from app.domain.agent_runtime import (
    AgentVersion,
    PluginPackage,
    ToolCall,
    ToolDefinition,
)
from app.domain.connectors import ChangePage, DataSource, SourceItemBinding, SyncCursor
from app.domain.evidence import Evidence
from app.domain.knowledge_governance import MetadataTarget
from app.domain.model_routing import (
    ModelHealth,
    ModelInstance,
    ModelRoutePolicy,
    ModelRouteRequest,
    ProviderDefinition,
)
from app.domain.multimodal import (
    MultimodalCandidate,
    MultimodalRetrievalProfile,
)


P2ResourceType = Literal[
    "model_route",
    "metadata_schema",
    "tag_taxonomy",
    "advanced_index",
    "connector_source",
    "agent_definition",
    "memory_policy",
    "multimodal_profile",
]


class P2CapabilityResourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_type: P2ResourceType
    logical_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)
    kb_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "KB-scoped resource types require this field; tenant-scoped types must omit it"
        ),
    )
    payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_container_scope(self) -> "P2CapabilityResourceCreate":
        validate_capability_resource_scope(
            resource_type=self.resource_type,
            kb_id=self.kb_id,
        )
        return self


class P2CapabilityPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = False


class P2CapabilityEnableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class P2CapabilityRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_type: str = Field(min_length=1, max_length=64)
    resource_id: str | None = Field(default=None, min_length=1, max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    input_summary: dict[str, Any] = Field(default_factory=dict)


class P2CapabilityRunTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "running",
        "waiting_approval",
        "succeeded",
        "partial",
        "failed",
        "cancelled",
    ]
    output: dict[str, Any] | None = None
    error_code: str | None = Field(default=None, max_length=128)
    error_detail: str | None = Field(default=None, max_length=4000)


class MetadataValueProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_resource_id: str = Field(min_length=1, max_length=64)
    target: MetadataTarget
    field_key: str = Field(min_length=1, max_length=128)
    value: Any
    source: Literal["auto", "manual", "import"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    enhancement: dict[str, Any] | None = None
    schema_revision: int | None = Field(default=None, ge=1)


class ModelRoutingPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: tuple[ProviderDefinition, ...] = Field(min_length=1, max_length=32)
    models: tuple[ModelInstance, ...] = Field(min_length=1, max_length=128)
    health: tuple[ModelHealth, ...] = Field(default_factory=tuple, max_length=128)
    policy: ModelRoutePolicy
    request: ModelRouteRequest


class ConnectorDryRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source: DataSource
    page: ChangePage
    cursor: SyncCursor
    bindings: tuple[SourceItemBinding, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_scope(self) -> "ConnectorDryRunRequest":
        expected = self.data_source.scope
        if self.cursor.scope != expected:
            raise ValueError("connector cursor scope does not match data source")
        if self.cursor.data_source_id != self.data_source.data_source_id:
            raise ValueError("connector cursor data_source_id does not match")
        for binding in self.bindings:
            if binding.scope != expected:
                raise ValueError("connector binding scope does not match data source")
            if binding.data_source_id != self.data_source.data_source_id:
                raise ValueError("connector binding data_source_id does not match")
        return self


class AdvancedRetrievalPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: AdvancedRetrievalRequest
    snapshots: tuple[AdvancedIndexSnapshot, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple, max_length=5000)

    @model_validator(mode="after")
    def unique_evidence(self) -> "AdvancedRetrievalPreviewRequest":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if any(item is None for item in evidence_ids):
            raise ValueError("advanced preview Evidence must have stable IDs")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("advanced preview Evidence IDs must be unique")
        return self


class AgentCompilePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentVersion
    tools: tuple[ToolDefinition, ...] = Field(default_factory=tuple, max_length=128)
    plugins: tuple[PluginPackage, ...] = Field(default_factory=tuple, max_length=64)


class ToolExecutionPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolDefinition
    call: ToolCall
    deterministic_output: dict[str, Any] = Field(default_factory=dict)


class MemoryProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4096)
    session_id: str = Field(min_length=1, max_length=128)
    source_message_id: str = Field(min_length=1, max_length=128)
    explicit_request: bool = True


class MemoryConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)


class MemoryRecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=4096)
    token_budget: int = Field(default=256, ge=16, le=8192)


class MultimodalRankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: MultimodalRetrievalProfile
    candidates: tuple[MultimodalCandidate, ...] = Field(max_length=1000)


class P2SelfCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    persist_demo_resources: bool = False


__all__ = [
    "AdvancedRetrievalPreviewRequest",
    "AgentCompilePreviewRequest",
    "ConnectorDryRunRequest",
    "MemoryConfirmRequest",
    "MemoryProposalRequest",
    "MemoryRecallRequest",
    "MetadataValueProposalRequest",
    "ModelRoutingPreviewRequest",
    "MultimodalRankRequest",
    "P2CapabilityEnableRequest",
    "P2CapabilityPublishRequest",
    "P2CapabilityResourceCreate",
    "P2CapabilityRunCreate",
    "P2CapabilityRunTransition",
    "P2ResourceType",
    "P2SelfCheckRequest",
    "ToolExecutionPreviewRequest",
]
