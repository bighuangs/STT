"""TODO-13 的纯应用层路由与 embedding shadow/migration 服务。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import hashlib
import json
import math
from typing import Protocol

from app.domain.model_routing import (
    EmbeddingMigrationPlan,
    ModelErrorCode,
    ModelHealth,
    ModelInstance,
    ModelRoutePlan,
    ModelRoutePolicy,
    ModelRouteRequest,
    ModelSnapshot,
    ProviderDefinition,
    RejectedModel,
    ShadowEvaluationReport,
    ShadowSampleMetric,
)


class ModelRoutingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class ModelRegistryPort(Protocol):
    def get_provider(self, provider_id: str) -> ProviderDefinition | None: ...

    def get_model(self, model_id: str) -> ModelInstance | None: ...

    def get_health(self, model_id: str) -> ModelHealth: ...


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _estimated_cost(
    model: ModelInstance,
    request: ModelRouteRequest,
) -> float:
    return (
        request.estimated_input_tokens * model.price.input_per_million
        + request.estimated_output_tokens * model.price.output_per_million
    ) / 1_000_000


def _snapshot(
    policy: ModelRoutePolicy,
    model: ModelInstance,
    provider: ProviderDefinition,
) -> ModelSnapshot:
    config = {
        "route_id": policy.route_id,
        "policy_version": policy.policy_version,
        "model": model.model_dump(mode="json"),
        "provider": provider.model_dump(mode="json"),
    }
    digest = _hash(config)
    return ModelSnapshot(
        snapshot_id=f"ms_{digest[:32]}",
        route_id=policy.route_id,
        policy_version=policy.policy_version,
        model_id=model.model_id,
        model_version=model.model_version,
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        remote_model_name=model.remote_model_name,
        capability=model.capability,
        embedding_dimension=model.embedding_dimension,
        config_hash=f"sha256:{digest}",
    )


class ModelRouter:
    """只选择已经通过同一策略过滤的模型；fallback 不会重新放宽边界。"""

    def __init__(self, registry: ModelRegistryPort) -> None:
        self._registry = registry

    def route(
        self,
        policy: ModelRoutePolicy,
        request: ModelRouteRequest,
    ) -> ModelRoutePlan:
        if not policy.enabled:
            raise ModelRoutingError("ROUTE_DISABLED", "model route is disabled")
        if policy.tenant_id != request.tenant_id:
            raise ModelRoutingError("TENANT_SCOPE_DENIED", "route belongs to another tenant")
        if policy.scenario != request.scenario:
            raise ModelRoutingError("SCENARIO_MISMATCH", "route scenario does not match request")

        accepted: list[tuple[ModelInstance, ProviderDefinition, float]] = []
        rejected: list[RejectedModel] = []
        approved = set(request.approved_model_ids)
        required_features = set(policy.required_features) | set(request.required_features)

        for model_id in policy.ordered_model_ids:
            reasons: list[str] = []
            model = self._registry.get_model(model_id)
            if model is None:
                rejected.append(RejectedModel(model_id=model_id, reason_codes=("MODEL_MISSING",)))
                continue
            provider = self._registry.get_provider(model.provider_id)
            if provider is None:
                rejected.append(
                    RejectedModel(model_id=model_id, reason_codes=("PROVIDER_MISSING",))
                )
                continue
            health = self._registry.get_health(model_id)
            if model.capability != policy.capability:
                reasons.append("CAPABILITY_MISMATCH")
            if model.status != "enabled":
                reasons.append("MODEL_NOT_ENABLED")
            if not provider.enabled:
                reasons.append("PROVIDER_NOT_ENABLED")
            if request.tenant_id not in model.allowed_tenant_ids and (
                "*" not in model.allowed_tenant_ids
            ):
                reasons.append("TENANT_NOT_ALLOWED")
            if approved and model_id not in approved:
                reasons.append("KB_MODEL_NOT_APPROVED")
            if not required_features.issubset(model.features):
                reasons.append("FEATURE_MISMATCH")
            if health.status in {"unhealthy", "unknown"}:
                reasons.append("MODEL_UNHEALTHY")
            if health.circuit_open:
                reasons.append("CIRCUIT_OPEN")
            if health.quota_remaining == 0:
                reasons.append("QUOTA_EXHAUSTED")
            if provider.external and not policy.allow_external:
                reasons.append("EXTERNAL_PROVIDER_DENIED")
            if provider.external and request.data_classification == "restricted":
                reasons.append("RESTRICTED_DATA_EXTERNAL_DENIED")
            if _classification_rank(request.data_classification) > _classification_rank(
                policy.max_data_classification
            ):
                reasons.append("DATA_CLASSIFICATION_DENIED")
            if policy.allowed_regions and not (
                set(provider.network_regions) & set(policy.allowed_regions)
            ):
                reasons.append("REGION_DENIED")
            if request.scenario == "embedding":
                if request.active_index_embedding_model_id is not None:
                    if model.model_id != request.active_index_embedding_model_id:
                        reasons.append("ACTIVE_INDEX_MODEL_MISMATCH")
                    if model.embedding_dimension != request.active_index_dimension:
                        reasons.append("ACTIVE_INDEX_DIMENSION_MISMATCH")

            cost = _estimated_cost(model, request)
            budget = policy.max_estimated_cost
            if request.request_budget is not None:
                budget = min(budget, request.request_budget)
            if cost > budget:
                reasons.append("BUDGET_EXCEEDED")
            if reasons:
                rejected.append(
                    RejectedModel(model_id=model_id, reason_codes=tuple(dict.fromkeys(reasons)))
                )
            else:
                accepted.append((model, provider, cost))

        if not accepted:
            raise ModelRoutingError("NO_COMPLIANT_MODEL", "no model satisfies route policy")

        snapshots = tuple(_snapshot(policy, model, provider) for model, provider, _ in accepted)
        return ModelRoutePlan(
            trace_id=request.trace_id,
            route_id=policy.route_id,
            policy_version=policy.policy_version,
            selected=snapshots[0],
            fallback_snapshots=snapshots[1:],
            rejected=tuple(rejected),
            estimated_costs={model.model_id: cost for model, _, cost in accepted},
        )

    @staticmethod
    def fallback_after(
        policy: ModelRoutePolicy,
        plan: ModelRoutePlan,
        failed_model_id: str,
        error_code: ModelErrorCode,
    ) -> ModelSnapshot:
        if error_code not in policy.fallback_on:
            raise ModelRoutingError(
                "FALLBACK_NOT_ALLOWED",
                "failure is not eligible for fallback",
            )
        candidates = (plan.selected, *plan.fallback_snapshots)
        current = next(
            (index for index, item in enumerate(candidates) if item.model_id == failed_model_id),
            None,
        )
        if current is None:
            raise ModelRoutingError("MODEL_NOT_IN_PLAN", "failed model is not in route plan")
        if current + 1 >= len(candidates):
            raise ModelRoutingError("FALLBACK_EXHAUSTED", "no compliant fallback remains")
        return candidates[current + 1]


def _classification_rank(value: str) -> int:
    ranks = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    return ranks[value]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


class EmbeddingMigrationService:
    """新旧索引并存的状态机；只有通过 shadow 门槛才能批准和发布。"""

    @staticmethod
    def start(plan: EmbeddingMigrationPlan) -> EmbeddingMigrationPlan:
        if plan.status != "draft":
            raise ModelRoutingError("INVALID_MIGRATION_STATE", "only draft can start")
        return plan.model_copy(update={"status": "running"})

    @staticmethod
    def evaluate(
        plan: EmbeddingMigrationPlan,
        samples: Iterable[ShadowSampleMetric],
        *,
        min_recall_delta: float = -0.01,
        max_latency_ratio: float = 1.5,
    ) -> EmbeddingMigrationPlan:
        if plan.status != "running":
            raise ModelRoutingError(
                "INVALID_MIGRATION_STATE",
                "only running migration can be evaluated",
            )
        values = tuple(samples)
        if not values:
            raise ModelRoutingError("EMPTY_SHADOW_SET", "shadow evaluation requires samples")
        old_recall = sum(item.old_recall for item in values) / len(values)
        new_recall = sum(item.new_recall for item in values) / len(values)
        old_p95 = _p95([item.old_latency_ms for item in values])
        new_p95 = _p95([item.new_latency_ms for item in values])
        reasons: list[str] = []
        if new_recall - old_recall < min_recall_delta:
            reasons.append("RECALL_REGRESSION")
        latency_limit = old_p95 * max_latency_ratio
        if new_p95 > latency_limit:
            reasons.append("LATENCY_REGRESSION")
        payload = {
            "migration_id": plan.migration_id,
            "samples": [item.model_dump(mode="json") for item in values],
            "thresholds": [min_recall_delta, max_latency_ratio],
        }
        report = ShadowEvaluationReport(
            report_id=f"shadow_{_hash(payload)[:32]}",
            sample_count=len(values),
            old_mean_recall=old_recall,
            new_mean_recall=new_recall,
            recall_delta=new_recall - old_recall,
            old_p95_latency_ms=old_p95,
            new_p95_latency_ms=new_p95,
            passed=not reasons,
            reasons=tuple(reasons),
        )
        return plan.model_copy(update={"status": "evaluated", "shadow_report": report})

    @staticmethod
    def approve(plan: EmbeddingMigrationPlan) -> EmbeddingMigrationPlan:
        if plan.status != "evaluated" or plan.shadow_report is None:
            raise ModelRoutingError(
                "INVALID_MIGRATION_STATE",
                "only evaluated migration can be approved",
            )
        if not plan.shadow_report.passed:
            raise ModelRoutingError(
                "SHADOW_EVALUATION_FAILED",
                "failed shadow evaluation cannot be approved",
            )
        return plan.model_copy(update={"status": "approved"})

    @staticmethod
    def publish(
        plan: EmbeddingMigrationPlan,
        *,
        published_at: datetime | None = None,
    ) -> EmbeddingMigrationPlan:
        if plan.status != "approved":
            raise ModelRoutingError(
                "INVALID_MIGRATION_STATE",
                "only approved migration can be published",
            )
        return plan.model_copy(
            update={
                "status": "published",
                "published_at": published_at or datetime.now(UTC),
            }
        )

    @staticmethod
    def rollback(plan: EmbeddingMigrationPlan) -> EmbeddingMigrationPlan:
        if plan.status != "published":
            raise ModelRoutingError(
                "INVALID_MIGRATION_STATE",
                "only published migration can be rolled back",
            )
        return plan.model_copy(update={"status": "rolled_back", "published_at": None})


__all__ = [
    "EmbeddingMigrationService",
    "ModelRegistryPort",
    "ModelRouter",
    "ModelRoutingError",
]
