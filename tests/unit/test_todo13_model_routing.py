from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.application.model_ports import (
    ChatModelPort,
    EmbeddingModelPort,
    EmbeddingModelRequest,
    ProviderCallError,
    RerankerModelPort,
    TextModelRequest,
    VisionModelPort,
)
from app.application.model_routing_service import (
    EmbeddingMigrationService,
    ModelRouter,
    ModelRoutingError,
)
from app.domain.model_routing import (
    CredentialReference,
    EmbeddingMigrationPlan,
    ModelHealth,
    ModelInstance,
    ModelPrice,
    ModelRoutePolicy,
    ModelRouteRequest,
    ProviderDefinition,
    ShadowSampleMetric,
)
from app.infra.llm.model_registry import InMemoryModelRegistry
from app.infra.llm.safe_model_adapters import (
    DeterministicLocalModelAdapter,
    DisabledRemoteModelAdapter,
)


def _local_provider(*, enabled: bool = True) -> ProviderDefinition:
    return ProviderDefinition(
        provider_id="local",
        provider_version="1.0.0",
        kind="local",
        display_name="Local",
        network_regions=("local",),
        external=False,
        enabled=enabled,
    )


def _remote_provider(*, enabled: bool = True) -> ProviderDefinition:
    return ProviderDefinition(
        provider_id="remote",
        provider_version="1.0.0",
        kind="openai_compatible",
        display_name="Remote",
        base_url="https://models.example.com/v1",
        credential_id="cred.remote",
        network_regions=("cn",),
        external=True,
        enabled=enabled,
    )


def _chat_model(
    model_id: str,
    provider_id: str,
    *,
    status: str = "enabled",
    price: float = 0.0,
) -> ModelInstance:
    return ModelInstance(
        model_id=model_id,
        model_version="1.0.0",
        provider_id=provider_id,
        remote_model_name=model_id,
        capability="chat",
        features=("json_schema", "stream"),
        context_window=32_000,
        price=ModelPrice(input_per_million=price, output_per_million=price),
        status=status,
    )


def _policy(**updates) -> ModelRoutePolicy:
    values = {
        "route_id": "generation.default",
        "policy_version": "1.0.0",
        "tenant_id": "tenant_a",
        "scenario": "generation",
        "capability": "chat",
        "ordered_model_ids": ("primary", "fallback"),
        "enabled": True,
        "allow_external": False,
        "allowed_regions": ("local",),
        "max_data_classification": "confidential",
        "max_estimated_cost": 1.0,
        "required_features": ("json_schema",),
    }
    values.update(updates)
    return ModelRoutePolicy(**values)


def _request(**updates) -> ModelRouteRequest:
    values = {
        "trace_id": "trace-1",
        "tenant_id": "tenant_a",
        "scenario": "generation",
        "data_classification": "internal",
        "estimated_input_tokens": 100,
        "estimated_output_tokens": 20,
    }
    values.update(updates)
    return ModelRouteRequest(**values)


def test_registry_and_route_are_fail_closed_by_default():
    assert _local_provider(enabled=False).enabled is False
    assert _chat_model("model", "local", status="disabled").status == "disabled"
    policy = _policy(enabled=False)
    registry = InMemoryModelRegistry()
    with pytest.raises(ModelRoutingError, match="disabled") as caught:
        ModelRouter(registry).route(policy, _request())
    assert caught.value.code == "ROUTE_DISABLED"


def test_route_filters_unhealthy_primary_and_keeps_only_compliant_fallback():
    registry = InMemoryModelRegistry()
    registry.register_provider(_local_provider())
    registry.register_model(
        _chat_model("primary", "local"),
        health=ModelHealth(model_id="primary", status="unhealthy"),
    )
    registry.register_model(
        _chat_model("fallback", "local"),
        health=ModelHealth(model_id="fallback", status="healthy"),
    )

    plan = ModelRouter(registry).route(_policy(), _request())

    assert plan.selected.model_id == "fallback"
    assert plan.fallback_snapshots == ()
    assert plan.rejected[0].model_id == "primary"
    assert "MODEL_UNHEALTHY" in plan.rejected[0].reason_codes
    assert plan.selected.config_hash.startswith("sha256:")


def test_external_provider_cannot_bypass_classification_or_region_policy():
    registry = InMemoryModelRegistry()
    registry.register_provider(_remote_provider())
    registry.register_model(
        _chat_model("primary", "remote"),
        health=ModelHealth(model_id="primary", status="healthy"),
    )
    policy = _policy(
        ordered_model_ids=("primary",),
        allow_external=True,
        allowed_regions=("cn",),
        max_data_classification="restricted",
    )
    with pytest.raises(ModelRoutingError) as caught:
        ModelRouter(registry).route(
            policy,
            _request(data_classification="restricted"),
        )
    assert caught.value.code == "NO_COMPLIANT_MODEL"


def test_tenant_pin_feature_and_budget_are_all_enforced_before_selection():
    registry = InMemoryModelRegistry()
    registry.register_provider(_local_provider())
    expensive = _chat_model("primary", "local", price=10_000)
    registry.register_model(
        expensive,
        health=ModelHealth(model_id="primary", status="healthy"),
    )
    with pytest.raises(ModelRoutingError):
        ModelRouter(registry).route(
            _policy(ordered_model_ids=("primary",)),
            _request(
                request_budget=0.00001,
                approved_model_ids=("other-model",),
                required_features=("tool_calling",),
            ),
        )


def test_fallback_is_only_allowed_for_retryable_error_and_stays_in_plan():
    registry = InMemoryModelRegistry()
    registry.register_provider(_local_provider())
    for model_id in ("primary", "fallback"):
        registry.register_model(
            _chat_model(model_id, "local"),
            health=ModelHealth(model_id=model_id, status="healthy"),
        )
    policy = _policy()
    plan = ModelRouter(registry).route(policy, _request())
    fallback = ModelRouter.fallback_after(policy, plan, "primary", "timeout")
    assert fallback.model_id == "fallback"
    with pytest.raises(ModelRoutingError) as caught:
        ModelRouter.fallback_after(policy, plan, "primary", "authentication")
    assert caught.value.code == "FALLBACK_NOT_ALLOWED"


def test_embedding_query_must_match_active_index_model_and_dimension():
    registry = InMemoryModelRegistry()
    registry.register_provider(_local_provider())
    registry.register_model(
        ModelInstance(
            model_id="embed-v2",
            model_version="2.0.0",
            provider_id="local",
            remote_model_name="embed-v2",
            capability="embedding",
            embedding_dimension=768,
            status="enabled",
        ),
        health=ModelHealth(model_id="embed-v2", status="healthy"),
    )
    policy = ModelRoutePolicy(
        route_id="embedding.default",
        policy_version="1.0.0",
        tenant_id="tenant_a",
        scenario="embedding",
        capability="embedding",
        ordered_model_ids=("embed-v2",),
        enabled=True,
        allowed_regions=("local",),
    )
    with pytest.raises(ModelRoutingError):
        ModelRouter(registry).route(
            policy,
            ModelRouteRequest(
                trace_id="trace-embed",
                tenant_id="tenant_a",
                scenario="embedding",
                active_index_embedding_model_id="embed-v1",
                active_index_dimension=1024,
            ),
        )


def test_credential_contract_accepts_reference_but_rejects_plain_secret():
    reference = CredentialReference(
        credential_id="cred.remote",
        secret_ref="vault://tenant-a/model-provider",
        secret_version="v3",
        owner_tenant_id="tenant_a",
    )
    assert "actual-secret" not in reference.model_dump_json()
    with pytest.raises(ValidationError):
        CredentialReference(
            credential_id="cred.remote",
            secret_ref="actual-secret",
            secret_version="v3",
            owner_tenant_id="tenant_a",
        )


def _migration() -> EmbeddingMigrationPlan:
    return EmbeddingMigrationPlan(
        migration_id="mig_1234567890abcdef1234567890abcdef",
        tenant_id="tenant_a",
        kb_id="kb_a",
        old_index_version_id="idx_v1",
        new_index_version_id="idx_v2",
        old_model_id="embed-v1",
        new_model_id="embed-v2",
        old_dimension=1024,
        new_dimension=768,
    )


def test_embedding_migration_requires_shadow_before_publish_and_can_rollback():
    service = EmbeddingMigrationService()
    running = service.start(_migration())
    evaluated = service.evaluate(
        running,
        (
            ShadowSampleMetric(
                case_id="c1",
                old_recall=0.8,
                new_recall=0.85,
                old_latency_ms=100,
                new_latency_ms=110,
            ),
            ShadowSampleMetric(
                case_id="c2",
                old_recall=0.7,
                new_recall=0.75,
                old_latency_ms=120,
                new_latency_ms=125,
            ),
        ),
    )
    assert evaluated.shadow_report is not None
    assert evaluated.shadow_report.passed is True
    published = service.publish(
        service.approve(evaluated),
        published_at=datetime(2026, 7, 24, tzinfo=UTC),
    )
    assert published.status == "published"
    assert service.rollback(published).status == "rolled_back"


def test_failed_shadow_report_cannot_be_approved():
    service = EmbeddingMigrationService()
    evaluated = service.evaluate(
        service.start(_migration()),
        (
            ShadowSampleMetric(
                case_id="regression",
                old_recall=0.9,
                new_recall=0.4,
                old_latency_ms=100,
                new_latency_ms=500,
            ),
        ),
    )
    assert evaluated.shadow_report is not None
    assert evaluated.shadow_report.reasons == (
        "RECALL_REGRESSION",
        "LATENCY_REGRESSION",
    )
    with pytest.raises(ModelRoutingError) as caught:
        service.approve(evaluated)
    assert caught.value.code == "SHADOW_EVALUATION_FAILED"


def test_safe_adapters_satisfy_ports_and_remote_adapter_never_calls_network():
    local = DeterministicLocalModelAdapter()
    remote = DisabledRemoteModelAdapter()
    assert isinstance(local, ChatModelPort)
    assert isinstance(local, VisionModelPort)
    assert isinstance(local, EmbeddingModelPort)
    assert isinstance(local, RerankerModelPort)
    snapshot = (
        ModelRouter(_single_embedding_registry())
        .route(
            ModelRoutePolicy(
                route_id="embedding.default",
                policy_version="1.0.0",
                tenant_id="tenant_a",
                scenario="embedding",
                capability="embedding",
                ordered_model_ids=("embed-v2",),
                enabled=True,
                allowed_regions=("local",),
            ),
            ModelRouteRequest(
                trace_id="trace",
                tenant_id="tenant_a",
                scenario="embedding",
            ),
        )
        .selected
    )
    response = local.embed(EmbeddingModelRequest(snapshot=snapshot, texts=("same", "same")))
    assert response.dimension == 768
    assert response.vectors[0] == response.vectors[1]
    with pytest.raises(ProviderCallError) as caught:
        remote.invoke_chat(
            TextModelRequest(
                snapshot=snapshot.model_copy(
                    update={"capability": "chat", "embedding_dimension": None}
                ),
                messages=({"role": "user", "content": "hello"},),
            )
        )
    assert caught.value.code == "policy_denied"
    assert caught.value.retryable is False


def _single_embedding_registry() -> InMemoryModelRegistry:
    registry = InMemoryModelRegistry()
    registry.register_provider(_local_provider())
    registry.register_model(
        ModelInstance(
            model_id="embed-v2",
            model_version="2.0.0",
            provider_id="local",
            remote_model_name="embed-v2",
            capability="embedding",
            embedding_dimension=768,
            status="enabled",
        ),
        health=ModelHealth(model_id="embed-v2", status="healthy"),
    )
    return registry
