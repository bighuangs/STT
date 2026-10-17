"""TODO-13 多模型注册、策略路由与 embedding 迁移的领域契约。

本模块刻意不读取环境变量、不持有密钥，也不调用任何模型供应商。所有对象均为
不可变 Pydantic 模型，便于写入 query/job trace 并在之后复现一次路由决策。
"""

from __future__ import annotations

from datetime import UTC, datetime
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MODEL_ROUTING_SCHEMA_VERSION = "model-routing.v1"
MODEL_SNAPSHOT_SCHEMA_VERSION = "model-snapshot.v1"
EMBEDDING_MIGRATION_SCHEMA_VERSION = "embedding-migration.v1"

ModelCapability = Literal["chat", "vision", "embedding", "rerank"]
ModelScenario = Literal[
    "rewrite",
    "generation",
    "vision",
    "enhance",
    "embedding",
    "rerank",
]
DataClassification = Literal["public", "internal", "confidential", "restricted"]
ProviderKind = Literal["local", "openai_compatible", "managed"]
ModelStatus = Literal["enabled", "draining", "disabled"]
HealthStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]
ModelErrorCode = Literal[
    "timeout",
    "rate_limited",
    "provider_5xx",
    "invalid_response",
    "authentication",
    "policy_denied",
    "quota_exhausted",
    "cancelled",
]
MigrationStatus = Literal[
    "draft",
    "running",
    "evaluated",
    "approved",
    "published",
    "rolled_back",
    "failed",
]

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_CLASSIFICATION_RANK = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}
_SCENARIO_CAPABILITY: dict[ModelScenario, ModelCapability] = {
    "rewrite": "chat",
    "generation": "chat",
    "vision": "vision",
    "enhance": "chat",
    "embedding": "embedding",
    "rerank": "rerank",
}


def _identifier(value: str, *, field_name: str) -> str:
    value = value.strip()
    if not _ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe identifier")
    return value


def _timezone(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


class CredentialReference(BaseModel):
    """只能保存 SecretProvider 定位信息，绝不包含密钥值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_id: str = Field(min_length=1, max_length=128)
    secret_ref: str = Field(pattern=r"^(?:secret|vault|env)://[A-Za-z0-9_./:-]+$")
    secret_version: str = Field(min_length=1, max_length=128)
    owner_tenant_id: str = Field(min_length=1, max_length=64)
    status: Literal["active", "rotating", "revoked"] = "active"

    @field_validator("credential_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, field_name="credential_id")


class ProviderDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1, max_length=128)
    provider_version: str = Field(pattern=_VERSION.pattern)
    kind: ProviderKind
    display_name: str = Field(min_length=1, max_length=256)
    base_url: str | None = Field(default=None, max_length=2048)
    credential_id: str | None = Field(default=None, max_length=128)
    network_regions: tuple[str, ...] = Field(min_length=1)
    external: bool = True
    enabled: bool = False

    @field_validator("provider_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, field_name="provider_id")

    @field_validator("network_regions")
    @classmethod
    def unique_regions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(region.strip().lower() for region in value)
        if any(not region for region in normalized):
            raise ValueError("network_regions cannot contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("network_regions cannot contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_transport(self) -> "ProviderDefinition":
        if self.kind == "local":
            if self.external:
                raise ValueError("local provider cannot be marked external")
            if self.credential_id is not None:
                raise ValueError("local provider cannot reference a remote credential")
        elif not self.base_url:
            raise ValueError("remote provider requires base_url")
        return self


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_million: float = Field(default=0.0, ge=0.0)
    output_per_million: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="CNY", min_length=3, max_length=8)

    @field_validator("input_per_million", "output_per_million")
    @classmethod
    def finite_price(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("model prices must be finite")
        return value


class ModelInstance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(pattern=_VERSION.pattern)
    provider_id: str = Field(min_length=1, max_length=128)
    remote_model_name: str = Field(min_length=1, max_length=512)
    capability: ModelCapability
    features: tuple[str, ...] = Field(default_factory=tuple)
    context_window: int | None = Field(default=None, ge=1)
    embedding_dimension: int | None = Field(default=None, ge=1, le=65_536)
    max_batch_size: int = Field(default=1, ge=1, le=100_000)
    price: ModelPrice = Field(default_factory=ModelPrice)
    allowed_tenant_ids: tuple[str, ...] = ("*",)
    status: ModelStatus = "disabled"

    @field_validator("model_id", "provider_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, field_name=info.field_name)

    @field_validator("features", "allowed_tenant_ids")
    @classmethod
    def unique_values(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError(f"{info.field_name} cannot contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} cannot contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_capability_shape(self) -> "ModelInstance":
        if self.capability == "embedding":
            if self.embedding_dimension is None:
                raise ValueError("embedding model requires embedding_dimension")
        elif self.embedding_dimension is not None:
            raise ValueError("only embedding model can declare embedding_dimension")
        if self.capability in {"chat", "vision"} and self.context_window is None:
            raise ValueError("chat and vision models require context_window")
        return self


class ModelHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    status: HealthStatus = "unknown"
    circuit_open: bool = False
    quota_remaining: int | None = Field(default=None, ge=0)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason_code: str | None = Field(default=None, max_length=128)

    @field_validator("observed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _timezone(value, field_name="observed_at")


class ModelRoutePolicy(BaseModel):
    """租户场景的不可变路由策略；enabled 默认为 False，确保新增能力默认关闭。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(pattern=_VERSION.pattern)
    tenant_id: str = Field(min_length=1, max_length=64)
    scenario: ModelScenario
    capability: ModelCapability
    ordered_model_ids: tuple[str, ...] = Field(min_length=1)
    enabled: bool = False
    allow_external: bool = False
    allowed_regions: tuple[str, ...] = Field(default_factory=tuple)
    max_data_classification: DataClassification = "internal"
    max_estimated_cost: float = Field(default=0.0, ge=0.0)
    required_features: tuple[str, ...] = Field(default_factory=tuple)
    timeout_ms: int = Field(default=30_000, ge=1, le=600_000)
    fallback_on: tuple[ModelErrorCode, ...] = (
        "timeout",
        "rate_limited",
        "provider_5xx",
    )

    @field_validator("route_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, field_name="route_id")

    @field_validator(
        "ordered_model_ids",
        "allowed_regions",
        "required_features",
        "fallback_on",
    )
    @classmethod
    def unique_values(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} cannot contain duplicates")
        return value

    @field_validator("max_estimated_cost")
    @classmethod
    def finite_budget(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("max_estimated_cost must be finite")
        return value

    @model_validator(mode="after")
    def validate_scenario_capability(self) -> "ModelRoutePolicy":
        if _SCENARIO_CAPABILITY[self.scenario] != self.capability:
            raise ValueError("scenario and capability do not match")
        unsafe = {"authentication", "policy_denied", "invalid_response", "cancelled"}
        if unsafe.intersection(self.fallback_on):
            raise ValueError("fallback_on can only contain retryable provider failures")
        return self


class ModelRouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str | None = Field(default=None, max_length=64)
    scenario: ModelScenario
    data_classification: DataClassification = "internal"
    required_features: tuple[str, ...] = Field(default_factory=tuple)
    estimated_input_tokens: int = Field(default=0, ge=0)
    estimated_output_tokens: int = Field(default=0, ge=0)
    request_budget: float | None = Field(default=None, ge=0.0)
    approved_model_ids: tuple[str, ...] = Field(default_factory=tuple)
    active_index_embedding_model_id: str | None = Field(default=None, max_length=128)
    active_index_dimension: int | None = Field(default=None, ge=1, le=65_536)

    @field_validator("request_budget")
    @classmethod
    def finite_budget(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("request_budget must be finite")
        return value

    @model_validator(mode="after")
    def validate_embedding_pin(self) -> "ModelRouteRequest":
        fields = (
            self.active_index_embedding_model_id,
            self.active_index_dimension,
        )
        if any(value is not None for value in fields) and self.scenario != "embedding":
            raise ValueError("active index embedding constraints only apply to embedding")
        if (fields[0] is None) != (fields[1] is None):
            raise ValueError("active index model and dimension must be supplied together")
        return self


class RejectedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    reason_codes: tuple[str, ...] = Field(min_length=1)


class ModelSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["model-snapshot.v1"] = MODEL_SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str = Field(pattern=r"^ms_[0-9a-f]{32}$")
    route_id: str
    policy_version: str
    model_id: str
    model_version: str
    provider_id: str
    provider_version: str
    remote_model_name: str
    capability: ModelCapability
    embedding_dimension: int | None = None
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _timezone(value, field_name="created_at")


class ModelRoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["model-routing.v1"] = MODEL_ROUTING_SCHEMA_VERSION
    trace_id: str
    route_id: str
    policy_version: str
    selected: ModelSnapshot
    fallback_snapshots: tuple[ModelSnapshot, ...] = Field(default_factory=tuple)
    rejected: tuple[RejectedModel, ...] = Field(default_factory=tuple)
    estimated_costs: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> "ModelRoutePlan":
        ids = [self.selected.model_id, *(item.model_id for item in self.fallback_snapshots)]
        if len(ids) != len(set(ids)):
            raise ValueError("route plan candidates must be unique")
        return self


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="CNY", min_length=3, max_length=8)


class ModelAttempt(BaseModel):
    """一次调用或 fallback 的可审计记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=1)
    snapshot: ModelSnapshot
    status: Literal["succeeded", "failed"]
    usage: ModelUsage = Field(default_factory=ModelUsage)
    error_code: ModelErrorCode | None = None
    fallback_from_model_id: str | None = None
    fallback_marked: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> "ModelAttempt":
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed model attempt requires error_code")
        if self.status == "succeeded" and self.error_code is not None:
            raise ValueError("successful model attempt cannot contain error_code")
        if self.fallback_from_model_id and not self.fallback_marked:
            raise ValueError("fallback attempt must be explicitly marked")
        return self


class ShadowSampleMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=128)
    old_recall: float = Field(ge=0.0, le=1.0)
    new_recall: float = Field(ge=0.0, le=1.0)
    old_latency_ms: float = Field(ge=0.0)
    new_latency_ms: float = Field(ge=0.0)

    @field_validator("old_recall", "new_recall", "old_latency_ms", "new_latency_ms")
    @classmethod
    def finite_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("shadow metrics must be finite")
        return value


class ShadowEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(pattern=r"^shadow_[0-9a-f]{32}$")
    sample_count: int = Field(ge=1)
    old_mean_recall: float = Field(ge=0.0, le=1.0)
    new_mean_recall: float = Field(ge=0.0, le=1.0)
    recall_delta: float = Field(ge=-1.0, le=1.0)
    old_p95_latency_ms: float = Field(ge=0.0)
    new_p95_latency_ms: float = Field(ge=0.0)
    passed: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    evaluator_version: str = "embedding-shadow.v1"


class EmbeddingMigrationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["embedding-migration.v1"] = EMBEDDING_MIGRATION_SCHEMA_VERSION
    migration_id: str = Field(pattern=r"^mig_[0-9a-f]{32}$")
    tenant_id: str = Field(min_length=1, max_length=64)
    kb_id: str = Field(min_length=1, max_length=64)
    old_index_version_id: str = Field(min_length=1, max_length=128)
    new_index_version_id: str = Field(min_length=1, max_length=128)
    old_model_id: str = Field(min_length=1, max_length=128)
    new_model_id: str = Field(min_length=1, max_length=128)
    old_dimension: int = Field(ge=1, le=65_536)
    new_dimension: int = Field(ge=1, le=65_536)
    status: MigrationStatus = "draft"
    shadow_report: ShadowEvaluationReport | None = None
    published_at: datetime | None = None

    @field_validator("published_at")
    @classmethod
    def validate_time(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            return _timezone(value, field_name="published_at")
        return value

    @model_validator(mode="after")
    def validate_isolation(self) -> "EmbeddingMigrationPlan":
        if self.old_index_version_id == self.new_index_version_id:
            raise ValueError("embedding migration requires a new index version")
        if self.old_model_id == self.new_model_id:
            raise ValueError("embedding migration requires a different model")
        if self.status in {"evaluated", "approved", "published"} and self.shadow_report is None:
            raise ValueError(f"{self.status} migration requires shadow_report")
        if self.status in {"approved", "published"} and not self.shadow_report.passed:
            raise ValueError("failed shadow evaluation cannot be approved or published")
        if (self.status == "published") != (self.published_at is not None):
            raise ValueError("published_at must exist exactly for published migration")
        return self


__all__ = [
    "CredentialReference",
    "DataClassification",
    "EmbeddingMigrationPlan",
    "HealthStatus",
    "MigrationStatus",
    "ModelAttempt",
    "ModelCapability",
    "ModelErrorCode",
    "ModelHealth",
    "ModelInstance",
    "ModelPrice",
    "ModelRoutePlan",
    "ModelRoutePolicy",
    "ModelRouteRequest",
    "ModelScenario",
    "ModelSnapshot",
    "ModelStatus",
    "ModelUsage",
    "ProviderDefinition",
    "ProviderKind",
    "RejectedModel",
    "ShadowEvaluationReport",
    "ShadowSampleMetric",
]
