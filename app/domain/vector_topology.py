"""Milvus 向量拓扑、兼容指纹、放置、迁移与 GC 的纯领域合同。

本模块不连接 Milvus，也不接受调用方传入物理 collection/partition 名。它把
TODO-37 的安全边界固化为可测试的不变式，供导入、检索、运维 API 和持久任务共同
复用。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_PHYSICAL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@/-]{1,255}$")


class VectorTopologyError(ValueError):
    """向量拓扑领域约束被违反。"""


class CompatibilityDriftError(VectorTopologyError):
    """期望兼容指纹与实际物理结构不一致。"""

    def __init__(self, differences: tuple[str, ...]) -> None:
        self.differences = differences
        super().__init__("vector compatibility drift: " + ", ".join(differences))


class PlacementMode(StrEnum):
    SHARED_COMPATIBLE = "shared_compatible"
    TENANT_DEDICATED = "tenant_dedicated"
    KB_DEDICATED = "kb_dedicated"
    CLUSTER_DEDICATED = "cluster_dedicated"


class FamilyStatus(StrEnum):
    CREATING = "creating"
    VALIDATING = "validating"
    ACTIVE = "active"
    DRAINING = "draining"
    RETIRED = "retired"
    FAILED = "failed"


class PlacementStatus(StrEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    SOURCE = "source"
    TARGET = "target"
    RETIRED = "retired"


class GcRunStatus(StrEnum):
    PLANNED = "planned"
    DRY_RUN_VALIDATED = "dry_run_validated"
    RUNNING = "running"
    PARTIAL = "partial"
    AWAITING_COMPACTION = "awaiting_compaction"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    BLOCKED_MANUAL = "blocked_manual"
    CANCELLED = "cancelled"


class MigrationStatus(StrEnum):
    PLANNED = "planned"
    TARGET_CREATED = "target_created"
    BACKFILLING = "backfilling"
    VALIDATING = "validating"
    READY_TO_SWITCH = "ready_to_switch"
    SWITCHED = "switched"
    ROLLBACK_WINDOW = "rollback_window"
    SOURCE_GC_ELIGIBLE = "source_gc_eligible"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED_RETRYABLE = "failed_retryable"
    BLOCKED_MANUAL = "blocked_manual"
    CANCELLED = "cancelled"


class ScalarFieldContract(BaseModel):
    """影响 collection 兼容性的静态标量字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    data_type: str = Field(min_length=1, max_length=64)
    primary_key: bool = False
    nullable: bool = False
    max_length: int | None = Field(default=None, ge=1)
    default_value: Any = None

    @field_validator("data_type")
    @classmethod
    def normalize_data_type(cls, value: str) -> str:
        return value.strip().upper()


class DenseVectorContract(BaseModel):
    """稠密向量语义空间与物理索引合同。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    dimension: int = Field(ge=1, le=65536)
    provider: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=255)
    revision: str = Field(min_length=1, max_length=255)
    tokenizer: str = Field(min_length=1, max_length=255)
    normalization: str = Field(min_length=1, max_length=64)
    query_instruction: str = Field(default="", max_length=4096)
    document_instruction: str = Field(default="", max_length=4096)
    metric: str = Field(min_length=1, max_length=32)
    index_type: str = Field(min_length=1, max_length=64)
    index_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metric", "index_type")
    @classmethod
    def normalize_uppercase(cls, value: str) -> str:
        return value.strip().upper()


class SparseVectorContract(BaseModel):
    """稀疏向量算法与物理索引合同。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    algorithm: str = Field(min_length=1, max_length=128)
    analyzer: str = Field(min_length=1, max_length=255)
    metric: str = Field(min_length=1, max_length=32)
    index_type: str = Field(min_length=1, max_length=64)
    index_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metric", "index_type")
    @classmethod
    def normalize_uppercase(cls, value: str) -> str:
        return value.strip().upper()


class PartitionContract(BaseModel):
    """有界 partition-key 合同。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["none", "bounded_hash", "tenant_key"]
    key_field: str | None = Field(default=None, max_length=128)
    algorithm_version: str = Field(default="v1", min_length=1, max_length=64)
    num_partitions: int = Field(default=1, ge=1, le=4096)

    @model_validator(mode="after")
    def validate_strategy(self) -> PartitionContract:
        if self.strategy == "none":
            if self.key_field is not None or self.num_partitions != 1:
                raise ValueError("partition strategy none requires one partition and no key")
        elif not self.key_field:
            raise ValueError("partition key strategy requires key_field")
        if self.strategy == "bounded_hash" and self.num_partitions & (self.num_partitions - 1):
            raise ValueError("bounded_hash num_partitions must be a power of two")
        return self


class CompatibilityProfile(BaseModel):
    """决定向量能否物理共享的完整 canonical payload。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1, max_length=64)
    chunk_identity_schema_version: str = Field(min_length=1, max_length=64)
    evidence_identity_schema_version: str = Field(min_length=1, max_length=64)
    dynamic_fields_enabled: bool = False
    scalar_fields: tuple[ScalarFieldContract, ...] = Field(min_length=1)
    dense: DenseVectorContract
    sparse: SparseVectorContract | None = None
    partition: PartitionContract
    preprocessing_version: str = Field(min_length=1, max_length=128)
    quantization_version: str = Field(default="none", min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_unique_fields(self) -> CompatibilityProfile:
        names = [field.name for field in self.scalar_fields]
        names.extend([self.dense.field_name])
        if self.sparse is not None:
            names.append(self.sparse.field_name)
        if len(names) != len(set(names)):
            raise ValueError("compatibility profile field names must be unique")
        if self.partition.key_field and self.partition.key_field not in {
            field.name for field in self.scalar_fields
        }:
            raise ValueError("partition key must be a declared static scalar field")
        return self


class CompatibilityFingerprint(BaseModel):
    """完整指纹以及可独立审计的 schema/index 子指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_payload_digest(self) -> CompatibilityFingerprint:
        if _digest(self.canonical_payload) != self.compatibility:
            raise ValueError("compatibility fingerprint does not match canonical payload")
        return self


def canonical_json(value: Any) -> str:
    """跨进程、跨字段顺序稳定的 JSON。"""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def compatibility_fingerprint(profile: CompatibilityProfile) -> CompatibilityFingerprint:
    """计算完整、schema 和 index 三层指纹，禁止仅以 dimension 判兼容。"""

    payload = profile.model_dump(mode="json")
    # schema field declaration order does not change Milvus compatibility；canonical
    # payload 先按字段名排序，避免不同语言/构造顺序产生假漂移。
    payload["scalar_fields"] = sorted(payload["scalar_fields"], key=lambda item: item["name"])
    schema_payload = {
        "schema_version": payload["schema_version"],
        "chunk_identity_schema_version": payload["chunk_identity_schema_version"],
        "evidence_identity_schema_version": payload["evidence_identity_schema_version"],
        "dynamic_fields_enabled": payload["dynamic_fields_enabled"],
        "scalar_fields": payload["scalar_fields"],
        "dense": {
            "field_name": payload["dense"]["field_name"],
            "dimension": payload["dense"]["dimension"],
        },
        "sparse": (
            {"field_name": payload["sparse"]["field_name"]}
            if payload["sparse"] is not None
            else None
        ),
        "partition": payload["partition"],
    }
    index_payload = {
        "dense": {
            "metric": payload["dense"]["metric"],
            "index_type": payload["dense"]["index_type"],
            "index_params": payload["dense"]["index_params"],
        },
        "sparse": (
            {
                "metric": payload["sparse"]["metric"],
                "index_type": payload["sparse"]["index_type"],
                "index_params": payload["sparse"]["index_params"],
                "algorithm": payload["sparse"]["algorithm"],
                "analyzer": payload["sparse"]["analyzer"],
            }
            if payload["sparse"] is not None
            else None
        ),
    }
    return CompatibilityFingerprint(
        compatibility=_digest(payload),
        schema_fingerprint=_digest(schema_payload),
        index_fingerprint=_digest(index_payload),
        canonical_payload=payload,
    )


def _diff_values(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if type(expected) is not type(actual):
        return [f"{path}:type"]
    if isinstance(expected, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child}:unexpected")
            elif key not in actual:
                differences.append(f"{child}:missing")
            else:
                differences.extend(_diff_values(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}:length"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_diff_values(left, right, f"{path}[{index}]"))
        return differences
    return [] if expected == actual else [f"{path}:value"]


def require_compatible(
    expected: CompatibilityFingerprint,
    actual: CompatibilityFingerprint,
) -> None:
    """远端无法证明完全兼容时 fail-closed，并给出不含密钥/正文的差异路径。"""

    if (
        expected.compatibility == actual.compatibility
        and expected.schema_fingerprint == actual.schema_fingerprint
        and expected.index_fingerprint == actual.index_fingerprint
    ):
        return
    differences = list(_diff_values(expected.canonical_payload, actual.canonical_payload))
    if expected.schema_fingerprint != actual.schema_fingerprint:
        differences.append("$.schema_fingerprint:value")
    if expected.index_fingerprint != actual.index_fingerprint:
        differences.append("$.index_fingerprint:value")
    raise CompatibilityDriftError(tuple(differences or ("$.compatibility:value",)))


class PlacementPolicy(BaseModel):
    """版本化、服务端所有的 placement 策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    default_database: str = Field(default="default", min_length=1, max_length=63)
    shared_partition_count: int = Field(default=64, ge=1, le=4096)
    tenant_entity_threshold: int = Field(default=5_000_000, ge=1)
    kb_entity_threshold: int = Field(default=2_000_000, ge=1)
    hot_qps_threshold: float = Field(default=50.0, ge=0)
    regulated_tenants: frozenset[str] = Field(default_factory=frozenset)
    cluster_dedicated_tenants: frozenset[str] = Field(default_factory=frozenset)
    tenant_dedicated_tenants: frozenset[str] = Field(default_factory=frozenset)
    kb_dedicated_ids: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("default_database")
    @classmethod
    def validate_database_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("default_database is not a valid server physical name")
        return value

    @field_validator("shared_partition_count")
    @classmethod
    def validate_partition_count(cls, value: int) -> int:
        if value & (value - 1):
            raise ValueError("shared_partition_count must be a power of two")
        return value


class PlacementRequest(BaseModel):
    """仅包含逻辑/容量事实；特意不提供物理名称字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=255)
    kb_id: str = Field(min_length=1, max_length=255)
    index_version_id: str = Field(min_length=1, max_length=255)
    compliance_tier: Literal["standard", "regulated", "isolated"] = "standard"
    estimated_tenant_entities: int = Field(default=0, ge=0)
    estimated_kb_entities: int = Field(default=0, ge=0)
    estimated_qps: float = Field(default=0, ge=0)
    special_vector_contract: bool = False

    @field_validator("tenant_id", "kb_id", "index_version_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("logical identity contains unsupported characters")
        return value


class PlacementDecision(BaseModel):
    """服务端解析出的物理位置，可完整写入 trace/audit。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: PlacementMode
    physical_database: str
    physical_collection: str
    partition_bucket: str | None
    compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str
    policy_version: int = Field(ge=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)

    @field_validator("physical_database", "physical_collection")
    @classmethod
    def validate_physical_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("invalid server physical name")
        return value


def _scope_token(*parts: str, length: int = 12) -> str:
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def resolve_placement(
    policy: PlacementPolicy,
    request: PlacementRequest,
    fingerprint: CompatibilityFingerprint,
) -> PlacementDecision:
    """只依据服务端策略和控制事实选择 placement。"""

    reasons: list[str] = []
    if (
        request.tenant_id in policy.cluster_dedicated_tenants
        or request.compliance_tier == "isolated"
    ):
        mode = PlacementMode.CLUSTER_DEDICATED
        reasons.append("cluster_isolation_policy")
    elif (
        request.kb_id in policy.kb_dedicated_ids
        or request.special_vector_contract
        or request.estimated_kb_entities >= policy.kb_entity_threshold
    ):
        mode = PlacementMode.KB_DEDICATED
        reasons.append(
            "special_vector_contract"
            if request.special_vector_contract
            else "kb_capacity_or_policy"
        )
    elif (
        request.tenant_id in policy.tenant_dedicated_tenants
        or request.tenant_id in policy.regulated_tenants
        or request.compliance_tier == "regulated"
        or request.estimated_tenant_entities >= policy.tenant_entity_threshold
        or request.estimated_qps >= policy.hot_qps_threshold
    ):
        mode = PlacementMode.TENANT_DEDICATED
        reasons.append("tenant_risk_capacity_or_policy")
    else:
        mode = PlacementMode.SHARED_COMPATIBLE
        reasons.append("compatible_shared_default")

    compatibility_token = fingerprint.compatibility[:12]
    if mode == PlacementMode.SHARED_COMPATIBLE:
        database = policy.default_database
        collection = f"stt_chunks_c_{compatibility_token}"
        bucket = int(_scope_token(request.tenant_id, request.kb_id, length=16), 16)
        partition = f"p_{bucket % policy.shared_partition_count:04d}"
    elif mode == PlacementMode.TENANT_DEDICATED:
        database = policy.default_database
        collection = f"stt_chunks_t_{_scope_token(request.tenant_id)}_{compatibility_token[:8]}"
        partition = None
    elif mode == PlacementMode.KB_DEDICATED:
        database = policy.default_database
        collection = f"stt_chunks_k_{_scope_token(request.tenant_id, request.kb_id)}_{compatibility_token[:8]}"
        partition = None
    else:
        database = f"stt_t_{_scope_token(request.tenant_id)}"
        collection = f"stt_chunks_c_{compatibility_token}"
        partition = None

    return PlacementDecision(
        mode=mode,
        physical_database=database,
        physical_collection=collection,
        partition_bucket=partition,
        compatibility_fingerprint=fingerprint.compatibility,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        reason_codes=tuple(reasons),
    )


class VectorCollectionFamily(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: str = Field(min_length=1, max_length=255)
    physical_database: str
    physical_collection: str
    placement_mode: PlacementMode
    compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_payload: dict[str, Any]
    partition_strategy: str = Field(min_length=1, max_length=64)
    num_partitions: int = Field(ge=1, le=4096)
    status: FamilyStatus
    epoch: int = Field(ge=1)

    @field_validator("physical_database", "physical_collection")
    @classmethod
    def validate_physical_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("invalid server physical name")
        return value

    @model_validator(mode="after")
    def validate_saved_fingerprints(self) -> VectorCollectionFamily:
        profile = CompatibilityProfile.model_validate(self.compatibility_payload)
        generated = compatibility_fingerprint(profile)
        if (
            generated.compatibility != self.compatibility_fingerprint
            or generated.schema_fingerprint != self.schema_fingerprint
            or generated.index_fingerprint != self.index_fingerprint
        ):
            raise ValueError("vector family fingerprints do not match saved payload")
        return self


class VectorPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    placement_id: str = Field(min_length=1, max_length=255)
    tenant_id: str = Field(min_length=1, max_length=255)
    kb_id: str = Field(min_length=1, max_length=255)
    index_version_id: str = Field(min_length=1, max_length=255)
    family_id: str = Field(min_length=1, max_length=255)
    physical_database: str
    physical_collection: str
    partition_bucket: str | None = None
    policy_id: str = Field(min_length=1, max_length=128)
    policy_version: int = Field(ge=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)
    status: PlacementStatus
    epoch: int = Field(ge=1)
    readable: bool = False
    writable: bool = False

    @field_validator("physical_database", "physical_collection")
    @classmethod
    def validate_physical_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("invalid server physical name")
        return value


class QueryPhysicalRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: str
    physical_database: str
    physical_collection: str
    partition_bucket: str | None
    placement_epoch: int
    index_version_ids: tuple[str, ...] = Field(min_length=1)


class QueryPhysicalReadSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    routes: tuple[QueryPhysicalRoute, ...] = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def freeze_query_read_set(
    placements: tuple[VectorPlacement, ...],
) -> QueryPhysicalReadSet:
    """固定 QueryRun 的物理读集合；同一 index 不得出现歧义 active read。"""

    readable = [item for item in placements if item.readable]
    if not readable:
        raise VectorTopologyError("query read placement is empty")
    seen: dict[str, VectorPlacement] = {}
    for placement in readable:
        if placement.status not in {
            PlacementStatus.ACTIVE,
            PlacementStatus.SOURCE,
            PlacementStatus.TARGET,
        }:
            raise VectorTopologyError(
                f"placement is readable in an invalid status: {placement.status}"
            )
        prior = seen.get(placement.index_version_id)
        if prior is not None and prior.placement_id != placement.placement_id:
            raise VectorTopologyError(
                f"index version has ambiguous read placements: {placement.index_version_id}"
            )
        seen[placement.index_version_id] = placement

    grouped: dict[tuple[str, str, str, str | None, int], list[str]] = {}
    for placement in seen.values():
        key = (
            placement.family_id,
            placement.physical_database,
            placement.physical_collection,
            placement.partition_bucket,
            placement.epoch,
        )
        grouped.setdefault(key, []).append(placement.index_version_id)
    routes = tuple(
        QueryPhysicalRoute(
            family_id=key[0],
            physical_database=key[1],
            physical_collection=key[2],
            partition_bucket=key[3],
            placement_epoch=key[4],
            index_version_ids=tuple(sorted(index_ids)),
        )
        for key, index_ids in sorted(grouped.items(), key=lambda item: str(item[0]))
    )
    return QueryPhysicalReadSet(
        routes=routes,
        digest=_digest([route.model_dump(mode="json") for route in routes]),
    )


class VectorIdentity(BaseModel):
    """删除、迁移和 inventory 使用的完整逻辑身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=255)
    kb_id: str = Field(min_length=1, max_length=255)
    document_id: str = Field(min_length=1, max_length=255)
    document_version_id: str = Field(min_length=1, max_length=255)
    index_version_id: str = Field(min_length=1, max_length=255)


class VectorGcScope(VectorIdentity):
    family_id: str = Field(min_length=1, max_length=255)
    physical_database: str
    physical_collection: str

    @field_validator("physical_database", "physical_collection")
    @classmethod
    def validate_physical_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("invalid server physical name")
        return value

    def milvus_filter(self) -> str:
        """从结构化身份生成完整删除表达式，不接受任意 filter 字符串。"""

        def quoted(value: str) -> str:
            return json.dumps(value, ensure_ascii=False)

        return " and ".join(
            (
                f"tenant_id == {quoted(self.tenant_id)}",
                f"kb_id == {quoted(self.kb_id)}",
                f"doc_id == {quoted(self.document_id)}",
                f"document_version_id == {quoted(self.document_version_id)}",
                f"index_version_id == {quoted(self.index_version_id)}",
            )
        )


class GcProtectionFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_binding: bool = False
    rollback_pinned: bool = False
    legal_hold: bool = False
    rollback_expires_at: datetime | None = None
    family_reference_count: int = Field(default=1, ge=0)
    placement_mode: PlacementMode


class GcEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    drop_family_allowed: bool
    reason_codes: tuple[str, ...]


def evaluate_gc_eligibility(
    facts: GcProtectionFacts,
    *,
    now: datetime | None = None,
) -> GcEligibility:
    """active、pin、hold 或回滚窗口内的向量永远不能进入 GC。"""

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    reasons: list[str] = []
    if facts.active_binding:
        reasons.append("active_binding")
    if facts.rollback_pinned:
        reasons.append("rollback_pinned")
    if facts.legal_hold:
        reasons.append("legal_hold")
    if facts.rollback_expires_at is not None:
        expires_at = facts.rollback_expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if current < expires_at:
            reasons.append("rollback_window_open")
    eligible = not reasons
    drop_family_allowed = (
        eligible
        and facts.placement_mode != PlacementMode.SHARED_COMPATIBLE
        and facts.family_reference_count == 0
    )
    if eligible and facts.family_reference_count > 0:
        reasons.append("family_still_referenced")
    if eligible and facts.placement_mode == PlacementMode.SHARED_COMPATIBLE:
        reasons.append("shared_family_never_dropped_for_single_scope")
    return GcEligibility(
        eligible=eligible,
        drop_family_allowed=drop_family_allowed,
        reason_codes=tuple(reasons or ("eligible",)),
    )


class VectorGcPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gc_run_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=255)
    scope: VectorGcScope
    expected_count: int = Field(ge=0)
    drop_family_requested: bool = False
    status: GcRunStatus = GcRunStatus.PLANNED
    dry_run: bool = True
    attempt: int = Field(default=0, ge=0)
    fencing_token: int = Field(default=0, ge=0)


class GcDryRunObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_count: int = Field(ge=0)
    sampled_identities: tuple[VectorIdentity, ...] = ()


class GcDryRunDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GcRunStatus
    safe_to_execute: bool
    reason_codes: tuple[str, ...]


def validate_gc_dry_run(
    plan: VectorGcPlan,
    observation: GcDryRunObservation,
    eligibility: GcEligibility,
) -> GcDryRunDecision:
    """数量或身份无法与控制事实精确对账时阻断 delete。"""

    reasons: list[str] = []
    if not eligibility.eligible:
        reasons.extend(eligibility.reason_codes)
    expected_identity = VectorIdentity.model_validate(
        plan.scope.model_dump(
            include={
                "tenant_id",
                "kb_id",
                "document_id",
                "document_version_id",
                "index_version_id",
            }
        )
    )
    if any(item != expected_identity for item in observation.sampled_identities):
        reasons.append("cross_scope_identity")
    if observation.matched_count != plan.expected_count:
        reasons.append("matched_count_mismatch")
    if plan.drop_family_requested and not eligibility.drop_family_allowed:
        reasons.append("family_drop_not_allowed")
    if reasons:
        return GcDryRunDecision(
            status=GcRunStatus.BLOCKED_MANUAL,
            safe_to_execute=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
    return GcDryRunDecision(
        status=GcRunStatus.DRY_RUN_VALIDATED,
        safe_to_execute=True,
        reason_codes=("scope_and_count_reconciled",),
    )


class GcReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_count: int = Field(ge=0)
    matched_before: int = Field(ge=0)
    delete_acknowledged: bool
    remaining_after: int = Field(ge=0)
    compaction_status: Literal["not_requested", "pending", "running", "completed", "failed"] = (
        "not_requested"
    )


class GcOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GcRunStatus
    logical_delete_verified: bool
    physical_reclaimed: bool
    reason_codes: tuple[str, ...]


def reconcile_gc(result: GcReconciliation) -> GcOutcome:
    """RPC ack、逻辑不可见和物理空间回收是三个不同事实。"""

    if result.matched_before != result.expected_count:
        return GcOutcome(
            status=GcRunStatus.BLOCKED_MANUAL,
            logical_delete_verified=False,
            physical_reclaimed=False,
            reason_codes=("matched_count_mismatch",),
        )
    if not result.delete_acknowledged:
        return GcOutcome(
            status=GcRunStatus.FAILED_RETRYABLE,
            logical_delete_verified=False,
            physical_reclaimed=False,
            reason_codes=("delete_not_acknowledged",),
        )
    if result.remaining_after:
        return GcOutcome(
            status=GcRunStatus.PARTIAL,
            logical_delete_verified=False,
            physical_reclaimed=False,
            reason_codes=("delete_has_remaining_entities",),
        )
    if result.compaction_status == "failed":
        return GcOutcome(
            status=GcRunStatus.FAILED_RETRYABLE,
            logical_delete_verified=True,
            physical_reclaimed=False,
            reason_codes=("compaction_failed",),
        )
    if result.compaction_status != "completed":
        return GcOutcome(
            status=GcRunStatus.AWAITING_COMPACTION,
            logical_delete_verified=True,
            physical_reclaimed=False,
            reason_codes=("logical_delete_verified_compaction_pending",),
        )
    return GcOutcome(
        status=GcRunStatus.SUCCEEDED,
        logical_delete_verified=True,
        physical_reclaimed=True,
        reason_codes=("delete_and_compaction_reconciled",),
    )


_GC_TRANSITIONS: dict[GcRunStatus, set[GcRunStatus]] = {
    GcRunStatus.PLANNED: {
        GcRunStatus.DRY_RUN_VALIDATED,
        GcRunStatus.BLOCKED_MANUAL,
        GcRunStatus.CANCELLED,
    },
    GcRunStatus.DRY_RUN_VALIDATED: {
        GcRunStatus.RUNNING,
        GcRunStatus.BLOCKED_MANUAL,
        GcRunStatus.CANCELLED,
    },
    GcRunStatus.RUNNING: {
        GcRunStatus.PARTIAL,
        GcRunStatus.AWAITING_COMPACTION,
        GcRunStatus.SUCCEEDED,
        GcRunStatus.FAILED_RETRYABLE,
        GcRunStatus.BLOCKED_MANUAL,
    },
    GcRunStatus.PARTIAL: {
        GcRunStatus.RUNNING,
        GcRunStatus.BLOCKED_MANUAL,
        GcRunStatus.CANCELLED,
    },
    GcRunStatus.AWAITING_COMPACTION: {
        GcRunStatus.SUCCEEDED,
        GcRunStatus.FAILED_RETRYABLE,
    },
    GcRunStatus.FAILED_RETRYABLE: {
        GcRunStatus.RUNNING,
        GcRunStatus.CANCELLED,
    },
}


def ensure_gc_transition(current: GcRunStatus, target: GcRunStatus) -> None:
    if current == target:
        return
    if target not in _GC_TRANSITIONS.get(current, set()):
        raise VectorTopologyError(f"invalid vector GC transition: {current} -> {target}")


class VectorMigrationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    migration_run_id: str = Field(min_length=1, max_length=255)
    tenant_id: str = Field(min_length=1, max_length=255)
    kb_id: str = Field(min_length=1, max_length=255)
    index_version_id: str = Field(min_length=1, max_length=255)
    source_family_id: str = Field(min_length=1, max_length=255)
    target_family_id: str = Field(min_length=1, max_length=255)
    expected_count: int = Field(ge=0)
    chunk_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollback_deadline: datetime

    @model_validator(mode="after")
    def validate_distinct_families(self) -> VectorMigrationManifest:
        if self.source_family_id == self.target_family_id:
            raise ValueError("migration source and target families must differ")
        return self


class MigrationValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_count: int = Field(ge=0)
    actual_count: int = Field(ge=0)
    expected_chunk_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_chunk_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_compatibility_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampled_payloads_match: bool
    sampled_search_passed: bool
    cross_tenant_hit_count: int = Field(ge=0)

    @property
    def passed(self) -> bool:
        return (
            self.expected_count == self.actual_count
            and self.expected_chunk_ids_sha256 == self.actual_chunk_ids_sha256
            and self.expected_compatibility_fingerprint == self.actual_compatibility_fingerprint
            and self.sampled_payloads_match
            and self.sampled_search_passed
            and self.cross_tenant_hit_count == 0
        )

    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.expected_count != self.actual_count:
            reasons.append("entity_count_mismatch")
        if self.expected_chunk_ids_sha256 != self.actual_chunk_ids_sha256:
            reasons.append("chunk_identity_hash_mismatch")
        if self.expected_compatibility_fingerprint != self.actual_compatibility_fingerprint:
            reasons.append("compatibility_fingerprint_mismatch")
        if not self.sampled_payloads_match:
            reasons.append("sampled_payload_mismatch")
        if not self.sampled_search_passed:
            reasons.append("sampled_search_failed")
        if self.cross_tenant_hit_count:
            reasons.append("cross_tenant_hit")
        return tuple(reasons)


_MIGRATION_TRANSITIONS: dict[MigrationStatus, set[MigrationStatus]] = {
    MigrationStatus.PLANNED: {
        MigrationStatus.TARGET_CREATED,
        MigrationStatus.CANCELLED,
        MigrationStatus.BLOCKED_MANUAL,
    },
    MigrationStatus.TARGET_CREATED: {
        MigrationStatus.BACKFILLING,
        MigrationStatus.FAILED_RETRYABLE,
        MigrationStatus.CANCELLED,
    },
    MigrationStatus.BACKFILLING: {
        MigrationStatus.VALIDATING,
        MigrationStatus.FAILED_RETRYABLE,
        MigrationStatus.BLOCKED_MANUAL,
    },
    MigrationStatus.VALIDATING: {
        MigrationStatus.READY_TO_SWITCH,
        MigrationStatus.FAILED_RETRYABLE,
        MigrationStatus.BLOCKED_MANUAL,
    },
    MigrationStatus.READY_TO_SWITCH: {
        MigrationStatus.SWITCHED,
        MigrationStatus.CANCELLED,
        MigrationStatus.BLOCKED_MANUAL,
    },
    MigrationStatus.SWITCHED: {
        MigrationStatus.ROLLBACK_WINDOW,
        MigrationStatus.ROLLED_BACK,
        MigrationStatus.FAILED_RETRYABLE,
    },
    MigrationStatus.ROLLBACK_WINDOW: {
        MigrationStatus.SOURCE_GC_ELIGIBLE,
        MigrationStatus.ROLLED_BACK,
        MigrationStatus.BLOCKED_MANUAL,
    },
    MigrationStatus.SOURCE_GC_ELIGIBLE: {
        MigrationStatus.COMPLETED,
        MigrationStatus.ROLLED_BACK,
        MigrationStatus.FAILED_RETRYABLE,
    },
    MigrationStatus.FAILED_RETRYABLE: {
        MigrationStatus.TARGET_CREATED,
        MigrationStatus.BACKFILLING,
        MigrationStatus.VALIDATING,
        MigrationStatus.ROLLBACK_WINDOW,
        MigrationStatus.CANCELLED,
        MigrationStatus.BLOCKED_MANUAL,
    },
}


def ensure_migration_transition(
    current: MigrationStatus,
    target: MigrationStatus,
    *,
    validation: MigrationValidation | None = None,
    rollback_deadline: datetime | None = None,
    rollback_pinned: bool = False,
    now: datetime | None = None,
) -> None:
    """禁止跳过验证切读，也禁止在回滚窗口/保护期内清 source。"""

    if current == target:
        return
    if target not in _MIGRATION_TRANSITIONS.get(current, set()):
        raise VectorTopologyError(f"invalid vector migration transition: {current} -> {target}")
    if target == MigrationStatus.READY_TO_SWITCH:
        if validation is None or not validation.passed:
            reasons = validation.failure_reasons() if validation else ("missing_validation",)
            raise VectorTopologyError("migration validation failed: " + ", ".join(reasons))
    if target == MigrationStatus.SOURCE_GC_ELIGIBLE:
        if rollback_pinned:
            raise VectorTopologyError("migration source is rollback-pinned")
        if rollback_deadline is None:
            raise VectorTopologyError("migration rollback deadline is required")
        current_time = now or datetime.now(UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        deadline = rollback_deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        if current_time < deadline:
            raise VectorTopologyError("migration rollback window is still open")


class InventoryEntity(BaseModel):
    """只读 inventory 样本；identity 允许为空以便报告 legacy 缺失。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str | None = None
    tenant_id: str | None = None
    kb_id: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    index_version_id: str | None = None
    available: bool | None = None


class CollectionInventorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    physical_database: str
    physical_collection: str
    entity_count: int = Field(ge=0)
    partitions: tuple[str, ...] = ()
    indexes: tuple[dict[str, Any], ...] = ()
    schema_snapshot: dict[str, Any] = Field(default_factory=dict)
    sampled_entities: tuple[InventoryEntity, ...] = ()
    actual_fingerprint: CompatibilityFingerprint | None = None

    @field_validator("physical_database", "physical_collection")
    @classmethod
    def validate_physical_name(cls, value: str) -> str:
        if not _PHYSICAL_NAME.fullmatch(value):
            raise ValueError("invalid server physical name")
        return value


class TopologyInventoryReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    physical_database: str
    physical_collection: str
    entity_count: int
    partition_count: int
    index_count: int
    sampled_count: int
    missing_identity_rows: tuple[int, ...]
    duplicate_chunk_ids: tuple[str, ...]
    drift_paths: tuple[str, ...]
    warnings: tuple[str, ...]
    dry_run: Literal[True] = True
    destructive_actions: tuple[Any, ...] = ()


def audit_inventory(
    snapshot: CollectionInventorySnapshot,
    *,
    expected_fingerprint: CompatibilityFingerprint | None = None,
) -> TopologyInventoryReport:
    """生成可重复、零写入的 collection inventory 报告。"""

    required = (
        "chunk_id",
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
        "available",
    )
    missing_rows: list[int] = []
    chunk_counts: dict[str, int] = {}
    for index, row in enumerate(snapshot.sampled_entities):
        payload = row.model_dump()
        if any(payload.get(field) in (None, "") for field in required):
            missing_rows.append(index)
        if row.chunk_id:
            chunk_counts[row.chunk_id] = chunk_counts.get(row.chunk_id, 0) + 1
    duplicates = tuple(sorted(key for key, count in chunk_counts.items() if count > 1))
    drift: tuple[str, ...] = ()
    if expected_fingerprint is not None:
        if snapshot.actual_fingerprint is None:
            drift = ("$.actual_fingerprint:missing",)
        elif snapshot.actual_fingerprint.compatibility != expected_fingerprint.compatibility:
            drift = tuple(
                _diff_values(
                    expected_fingerprint.canonical_payload,
                    snapshot.actual_fingerprint.canonical_payload,
                )
            ) or ("$.compatibility:value",)
    warnings: list[str] = []
    if missing_rows:
        warnings.append("sample_contains_missing_identity")
    if duplicates:
        warnings.append("sample_contains_duplicate_chunk_id")
    if drift:
        warnings.append("collection_compatibility_drift")
    if snapshot.entity_count > len(snapshot.sampled_entities):
        warnings.append("identity_audit_is_sampled")
    return TopologyInventoryReport(
        physical_database=snapshot.physical_database,
        physical_collection=snapshot.physical_collection,
        entity_count=snapshot.entity_count,
        partition_count=len(snapshot.partitions),
        index_count=len(snapshot.indexes),
        sampled_count=len(snapshot.sampled_entities),
        missing_identity_rows=tuple(missing_rows),
        duplicate_chunk_ids=duplicates,
        drift_paths=drift,
        warnings=tuple(warnings),
    )


__all__ = [
    "CollectionInventorySnapshot",
    "CompatibilityDriftError",
    "CompatibilityFingerprint",
    "CompatibilityProfile",
    "DenseVectorContract",
    "FamilyStatus",
    "GcDryRunDecision",
    "GcDryRunObservation",
    "GcEligibility",
    "GcOutcome",
    "GcProtectionFacts",
    "GcReconciliation",
    "GcRunStatus",
    "InventoryEntity",
    "MigrationStatus",
    "MigrationValidation",
    "PartitionContract",
    "PlacementDecision",
    "PlacementMode",
    "PlacementPolicy",
    "PlacementRequest",
    "PlacementStatus",
    "QueryPhysicalReadSet",
    "ScalarFieldContract",
    "SparseVectorContract",
    "TopologyInventoryReport",
    "VectorCollectionFamily",
    "VectorGcPlan",
    "VectorGcScope",
    "VectorIdentity",
    "VectorMigrationManifest",
    "VectorPlacement",
    "VectorTopologyError",
    "audit_inventory",
    "canonical_json",
    "compatibility_fingerprint",
    "ensure_gc_transition",
    "ensure_migration_transition",
    "evaluate_gc_eligibility",
    "freeze_query_read_set",
    "reconcile_gc",
    "require_compatible",
    "resolve_placement",
    "validate_gc_dry_run",
]
