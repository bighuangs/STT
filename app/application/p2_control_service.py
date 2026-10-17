"""P2 能力资源的严格 payload 校验与作用域绑定。

控制库使用统一版本表，但这里仍逐类解析为领域模型；因此 API 不能把任意 JSON
标成已发布能力。运行时同时要求“领域对象允许执行”和“控制面资源已发布且启用”。
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Literal

from app.application.memory_service import MemoryPolicy
from app.domain.advanced_retrieval import AdvancedIndexSnapshot
from app.domain.agent_runtime import AgentVersion
from app.domain.connectors import DataSource
from app.domain.knowledge_governance import (
    MetadataSchemaVersion,
    TagTaxonomyVersion,
)
from app.domain.model_routing import ModelRoutePolicy
from app.domain.multimodal import MultimodalRetrievalProfile


class P2CapabilityValidationError(ValueError):
    """能力资源 payload 或其 tenant/KB/owner 绑定不合法。"""


P2ResourceScope = Literal["tenant", "kb"]

_CAPABILITY_RESOURCE_RULES: dict[str, tuple[str, P2ResourceScope]] = {
    "model_route": ("model-routing.v1", "tenant"),
    "metadata_schema": ("metadata-schema.v1", "kb"),
    "tag_taxonomy": ("tag-taxonomy.v1", "kb"),
    "advanced_index": ("advanced-snapshot.v1", "kb"),
    "connector_source": ("connector-source.v1", "kb"),
    "agent_definition": ("agent-dsl.v1", "tenant"),
    "memory_policy": ("memory-policy.v1", "kb"),
    "multimodal_profile": ("multimodal-profile.v1", "kb"),
}


def expected_schema_version(resource_type: str) -> str:
    try:
        return _CAPABILITY_RESOURCE_RULES[resource_type][0]
    except KeyError as exc:
        raise P2CapabilityValidationError(f"unsupported P2 resource type: {resource_type}") from exc


def expected_resource_scope(resource_type: str) -> P2ResourceScope:
    """返回统一资源类型的容器作用域，不从可选 ``kb_id`` 猜测作用域。"""

    try:
        return _CAPABILITY_RESOURCE_RULES[resource_type][1]
    except KeyError as exc:
        raise P2CapabilityValidationError(f"unsupported P2 resource type: {resource_type}") from exc


def validate_capability_resource_scope(
    *,
    resource_type: str,
    kb_id: str | None,
) -> P2ResourceScope:
    """校验统一资源外层作用域。

    tenant-scoped payload 没有可核对的 KB 身份，因此不能用外层 ``kb_id`` 把它伪装成
    KB 资源；KB-scoped payload 则必须显式携带外层 ``kb_id``，不能借省略该字段落到
    tenant scope。
    """

    scope = expected_resource_scope(resource_type)
    if scope == "tenant" and kb_id is not None:
        raise P2CapabilityValidationError(
            f"{resource_type} is tenant-scoped and does not accept resource kb_id"
        )
    if scope == "kb" and kb_id is None:
        raise P2CapabilityValidationError(
            f"{resource_type} is KB-scoped and requires resource kb_id"
        )
    return scope


def _assert_scope(
    *,
    actual_tenant_id: str,
    expected_tenant_id: str,
    actual_kb_id: str | None = None,
    expected_kb_id: str | None = None,
    compare_kb: bool = False,
) -> None:
    if actual_tenant_id != expected_tenant_id:
        raise P2CapabilityValidationError(
            "capability payload tenant scope does not match authenticated tenant"
        )
    if compare_kb and actual_kb_id != expected_kb_id:
        raise P2CapabilityValidationError("capability payload KB scope does not match resource KB")


def _memory_policy(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(MemoryPolicy)}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise P2CapabilityValidationError(
            "memory policy contains unsupported fields: " + ", ".join(unexpected)
        )
    policy = MemoryPolicy(**payload)
    # 触发三个长期类型的 TTL 和预算边界。dataclass 本身是运行时策略而不是 API schema。
    if policy.max_context_tokens <= 0 or policy.max_hits <= 0:
        raise P2CapabilityValidationError("memory context token and hit budgets must be positive")
    if policy.max_context_tokens > 8192 or policy.max_hits > 100:
        raise P2CapabilityValidationError("memory policy exceeds service hard limits")
    if any(
        value <= 0
        for value in (
            policy.preference_ttl_days,
            policy.profile_fact_ttl_days,
            policy.episodic_ttl_days,
        )
    ):
        raise P2CapabilityValidationError("memory TTL must be positive")
    return asdict(policy)


def validate_capability_payload(
    *,
    resource_type: str,
    schema_version: str,
    payload: dict[str, Any],
    tenant_id: str,
    kb_id: str | None,
    actor_id: str,
) -> dict[str, Any]:
    """解析、绑定并返回 canonical JSON-safe 领域快照。"""

    expected = expected_schema_version(resource_type)
    if schema_version != expected:
        raise P2CapabilityValidationError(f"{resource_type} requires schema_version={expected}")
    validate_capability_resource_scope(resource_type=resource_type, kb_id=kb_id)

    if resource_type == "model_route":
        value = ModelRoutePolicy.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.tenant_id,
            expected_tenant_id=tenant_id,
        )
    elif resource_type == "metadata_schema":
        value = MetadataSchemaVersion.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.tenant_id,
            expected_tenant_id=tenant_id,
            actual_kb_id=value.kb_id,
            expected_kb_id=kb_id,
            compare_kb=True,
        )
    elif resource_type == "tag_taxonomy":
        value = TagTaxonomyVersion.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.tenant_id,
            expected_tenant_id=tenant_id,
            actual_kb_id=value.kb_id,
            expected_kb_id=kb_id,
            compare_kb=True,
        )
    elif resource_type == "advanced_index":
        value = AdvancedIndexSnapshot.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.index.tenant_id,
            expected_tenant_id=tenant_id,
            actual_kb_id=value.index.kb_id,
            expected_kb_id=kb_id,
            compare_kb=True,
        )
    elif resource_type == "connector_source":
        value = DataSource.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.scope.tenant_id,
            expected_tenant_id=tenant_id,
            actual_kb_id=value.scope.kb_id,
            expected_kb_id=kb_id,
            compare_kb=True,
        )
    elif resource_type == "agent_definition":
        value = AgentVersion.model_validate(payload)
        _assert_scope(
            actual_tenant_id=value.tenant_id,
            expected_tenant_id=tenant_id,
        )
        if value.owner_id != actor_id:
            raise P2CapabilityValidationError(
                "agent owner_id must match the authenticated resource creator"
            )
    elif resource_type == "memory_policy":
        return _memory_policy(payload)
    elif resource_type == "multimodal_profile":
        value = MultimodalRetrievalProfile.model_validate(payload)
    else:  # pragma: no cover - expected_schema_version already guards this
        raise P2CapabilityValidationError(f"unsupported P2 resource type: {resource_type}")

    return value.model_dump(mode="json")


__all__ = [
    "P2CapabilityValidationError",
    "P2ResourceScope",
    "expected_resource_scope",
    "expected_schema_version",
    "validate_capability_resource_scope",
    "validate_capability_payload",
]
