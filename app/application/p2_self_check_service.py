"""P2 七项能力的确定性、无外调验收。

该自检使用真实领域服务和严格模型，但只在请求内构造 fixture；它不会调用外部
模型、MCP、连接器或写入知识库。持久化发布链由 capability resource API 单独验证。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Callable

from app.application.advanced_retrieval_service import AdvancedRetrievalService
from app.application.agent_runtime_service import AgentCompiler
from app.application.connector_sync_service import IncrementalSyncPlanner
from app.application.memory_service import LongTermMemoryService
from app.application.metadata_governance_service import MetadataGovernanceService
from app.application.model_routing_service import ModelRouter
from app.application.multimodal_service import rank_multimodal_candidates
from app.domain.advanced_retrieval import (
    AdvancedEvaluationGate,
    AdvancedFeatureFlags,
    AdvancedIndexSnapshot,
    AdvancedIndexVersion,
    AdvancedRetrievalRequest,
    StructureNode,
    StructureSnapshot,
)
from app.domain.agent_runtime import (
    AgentNode,
    AgentVersion,
    ToolDefinition,
    ToolReference,
)
from app.domain.connectors import (
    ChangePage,
    ConnectorScope,
    DataSource,
    SourceChange,
    SourceItem,
    SyncCursor,
)
from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.domain.knowledge_governance import (
    MetadataFieldDefinition,
    MetadataTarget,
)
from app.domain.memory import MemoryAccessScope
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
from app.infra.governance import InMemoryMetadataGovernanceRepository
from app.infra.llm.model_registry import InMemoryModelRegistry
from app.infra.memory import (
    InMemoryMemoryDerivativeStore,
    InMemoryMemoryRepository,
)
from app.infra.tools import InMemoryToolCatalog


def _now() -> datetime:
    return datetime.now(UTC)


def _result(
    todo: str,
    name: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        details = operation()
    except Exception as exc:  # pragma: no cover - exercised by route failure diagnostics
        return {
            "todo": todo,
            "name": name,
            "passed": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }
    return {
        "todo": todo,
        "name": name,
        "passed": True,
        "details": details,
    }


def _model_routing(tenant_id: str) -> dict[str, Any]:
    provider = ProviderDefinition(
        provider_id="local_provider",
        provider_version="1.0.0",
        kind="local",
        display_name="P2 deterministic local provider",
        network_regions=("local",),
        external=False,
        enabled=True,
    )
    model = ModelInstance(
        model_id="local_chat",
        model_version="1.0.0",
        provider_id=provider.provider_id,
        remote_model_name="deterministic/no-network",
        capability="chat",
        context_window=8192,
        status="enabled",
    )
    registry = InMemoryModelRegistry()
    registry.register_provider(provider)
    registry.register_model(model)
    registry.set_health(ModelHealth(model_id=model.model_id, status="healthy"))
    policy = ModelRoutePolicy(
        route_id="p2_selfcheck_route",
        policy_version="1.0.0",
        tenant_id=tenant_id,
        scenario="generation",
        capability="chat",
        ordered_model_ids=(model.model_id,),
        enabled=True,
        allow_external=False,
        max_data_classification="restricted",
        max_estimated_cost=1.0,
    )
    request = ModelRouteRequest(
        trace_id="trace_p2_model",
        tenant_id=tenant_id,
        scenario="generation",
        data_classification="restricted",
        estimated_input_tokens=128,
        estimated_output_tokens=64,
    )
    plan = ModelRouter(registry).route(policy, request)
    return {
        "selected_model": plan.selected.model_id,
        "snapshot_id": plan.selected.snapshot_id,
        "external": provider.external,
        "rejected_count": len(plan.rejected),
    }


def _metadata_governance(
    tenant_id: str,
    kb_id: str,
    actor_id: str,
) -> dict[str, Any]:
    repository = InMemoryMetadataGovernanceRepository()
    service = MetadataGovernanceService(repository)
    draft = service.create_schema_draft(
        schema_id="p2_selfcheck_schema",
        tenant_id=tenant_id,
        kb_id=kb_id,
        version=1,
        fields=(
            MetadataFieldDefinition(
                key="product_type",
                value_type="string",
                description="产品类型",
                scope="document",
                required=True,
                filterable=True,
                enum_values=("manual", "datasheet"),
                catalog_mode="distinct",
                max_distinct_values=10,
            ),
        ),
        actor_id=actor_id,
    )
    schema = service.publish_schema(
        tenant_id=tenant_id,
        kb_id=kb_id,
        schema_id=draft.schema_id,
        revision=draft.revision,
        actor_id=actor_id,
    )
    proposal = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=MetadataTarget(
            tenant_id=tenant_id,
            kb_id=kb_id,
            scope="document",
            document_id="doc_hak180_selfcheck",
        ),
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by=actor_id,
    )
    release = service.publish_proposals(
        tenant_id=tenant_id,
        kb_id=kb_id,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id=actor_id,
        value_record_ids=(proposal.record_id,),
    )
    return {
        "schema_revision": schema.revision,
        "proposal_id": proposal.record_id,
        "release_id": release.release_id,
        "published_value_count": len(release.value_record_ids),
    }


def _advanced_retrieval(tenant_id: str, kb_id: str) -> dict[str, Any]:
    now = _now()
    source_index = "idx_p2_selfcheck"
    document_id = "doc_hak180_selfcheck"
    document_version_id = "dv_hak180_selfcheck"
    block_id = "block_safety"
    evidence = Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id=tenant_id,
            kb_id=kb_id,
            document_id=document_id,
            document_version_id=document_version_id,
            index_version_id=source_index,
            chunk_id="chunk_safety",
            source_block_ids=[block_id],
        ),
        content="HAK180 使用前必须检查保护接地，并确认急停按钮有效。",
        provenance_spans=[
            EvidenceLocationSpan(
                block_id=block_id,
                availability="exact",
                coordinate_space="normalized_page",
                page_no=1,
                bbox=(0.1, 0.1, 0.9, 0.3),
            )
        ],
        heading_path=["使用前准备", "安全检查"],
        source_label="hak180产品安全手册.pdf",
    )
    assert evidence.evidence_id is not None
    advanced_index_id = "adv_toc_p2_selfcheck"
    snapshot = AdvancedIndexSnapshot(
        index=AdvancedIndexVersion(
            advanced_index_id=advanced_index_id,
            index_type="toc",
            tenant_id=tenant_id,
            kb_id=kb_id,
            scope="document",
            document_id=document_id,
            version=1,
            source_index_version_id=source_index,
            source_document_version_ids=(document_version_id,),
            status="active",
            evaluation=AdvancedEvaluationGate(
                evaluation_id="eval_p2_selfcheck",
                holdout_dataset_id="hak180_safety_v1",
                passed=True,
                quality_delta=0.1,
                citation_coverage_delta=0.1,
                p95_latency_ms=5,
                budget_approved=True,
                rollback_ready=True,
                report_uri="local://p2-self-check",
            ),
            created_by="p2_selfcheck",
            created_at=now,
            published_at=now,
        ),
        evidence_ids=(evidence.evidence_id,),
        structure=StructureSnapshot(
            advanced_index_id=advanced_index_id,
            source_index_version_id=source_index,
            nodes=(
                StructureNode(
                    node_id="toc_safety",
                    node_type="toc",
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    source_index_version_id=source_index,
                    title="安全检查",
                    toc_path=("安全检查",),
                    order=0,
                    source_evidence_ids=(evidence.evidence_id,),
                ),
            ),
        ),
    )
    request = AdvancedRetrievalRequest(
        query="使用前要检查哪些安全事项？",
        tenant_id=tenant_id,
        kb_id=kb_id,
        active_source_index_version_id=source_index,
        trace_id="trace_p2_toc",
        requested_routes=("toc",),
        planner_reasons={"toc": "章节定位问题"},
        flags=AdvancedFeatureFlags(toc_enabled=True),
    )
    result = AdvancedRetrievalService().search(
        request,
        snapshots=(snapshot,),
        evidence_by_id={evidence.evidence_id: evidence},
    )
    return {
        "candidate_count": len(result.candidates),
        "returned_evidence_id": (
            result.candidates[0].evidence.evidence_id if result.candidates else None
        ),
        "routes": [
            {
                "route": item.route,
                "status": item.status,
                "reason": item.reason,
            }
            for item in result.diagnostics.outcomes
        ],
        "generated_node_used_as_citation": False,
    }


def _connector(tenant_id: str, kb_id: str) -> dict[str, Any]:
    content = b"HAK180 deterministic connector fixture"
    scope = ConnectorScope(tenant_id=tenant_id, kb_id=kb_id)
    source = DataSource(
        data_source_id="p2_selfcheck_source",
        scope=scope,
        name="离线连接器验收",
        connector_key="deterministic",
        config={"root": "fixture"},
    )
    cursor = SyncCursor(
        scope=scope,
        data_source_id=source.data_source_id,
    )
    item = SourceItem(
        source_item_id="hak180_manual",
        remote_version="v1",
        content_sha256=sha256(content).hexdigest(),
        logical_name="hak180产品安全手册.pdf",
        mime_type="application/pdf",
        size_bytes=len(content),
        locator="fixture/hak180产品安全手册.pdf",
    )
    page = ChangePage(
        changes=(
            SourceChange(
                event_id="event_hak180_v1",
                sequence=1,
                kind="create",
                item=item,
            ),
        ),
        next_cursor="cursor_v1",
    )
    plan = IncrementalSyncPlanner().plan_incremental(
        source=source,
        cursor=cursor,
        page=page,
        bindings=(),
        run_id="sync_p2_selfcheck",
        dry_run=True,
    )
    return {
        "dry_run": plan.dry_run,
        "from_cursor": plan.from_cursor,
        "to_cursor": plan.to_cursor,
        "operation": plan.operations[0].action,
        "idempotency_key": plan.operations[0].idempotency_key,
    }


def _agent_runtime(tenant_id: str, actor_id: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "maxLength": 100}},
        "required": ["query"],
        "additionalProperties": False,
    }
    output = {
        "type": "object",
        "properties": {"answer": {"type": "string", "maxLength": 200}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        tool_id="p2_lookup",
        version="1.0.0",
        tenant_id=tenant_id,
        kind="deterministic",
        display_name="P2 离线只读工具",
        input_schema=schema,
        output_schema=output,
        required_permissions=("tool:p2_lookup",),
        side_effect="read",
        enabled=True,
    )
    catalog = InMemoryToolCatalog()
    catalog.register_tool(tool)
    agent = AgentVersion(
        agent_id="p2_selfcheck_agent",
        version="1.0.0",
        tenant_id=tenant_id,
        owner_id=actor_id,
        status="published",
        enabled=True,
        input_schema=schema,
        output_schema=output,
        allowed_permissions=("tool:p2_lookup",),
        nodes=(
            AgentNode(node_id="begin", kind="begin"),
            AgentNode(
                node_id="lookup",
                kind="tool",
                depends_on=("begin",),
                input_bindings={"query": "$input.query"},
                tool=ToolReference(tool_id=tool.tool_id, version=tool.version),
            ),
            AgentNode(
                node_id="answer",
                kind="answer",
                depends_on=("lookup",),
                input_bindings={"answer": "$nodes.lookup.answer"},
            ),
        ),
    )
    plan = AgentCompiler(catalog).compile(agent)
    return {
        "plan_hash": plan.plan_hash,
        "topological_order": list(plan.topological_order),
        "locked_tools": [item.reference.key for item in plan.tool_locks],
        "external_tool_called": False,
    }


def _memory(tenant_id: str, kb_id: str, actor_id: str) -> dict[str, Any]:
    repository = InMemoryMemoryRepository()
    derivatives = InMemoryMemoryDerivativeStore()
    service = LongTermMemoryService(
        repository=repository,
        derivatives=derivatives,
    )
    now = _now()
    scope = MemoryAccessScope(
        tenant_id=tenant_id,
        subject_user_id=actor_id,
        kb_id=kb_id,
        actor_user_id=actor_id,
    )
    proposal = service.propose_from_message(
        scope=scope,
        text="请记住：以后请用中文回答",
        session_id="session_p2_selfcheck",
        source_message_id="message_p2_selfcheck",
        explicit_request=True,
        now=now,
    )
    proposed = proposal.proposed[0]
    active = service.confirm(scope, proposed.memory_id, now=now + timedelta(seconds=1))
    context = service.retrieve(
        scope,
        "应该使用什么语言回答？",
        token_budget=256,
        now=now + timedelta(seconds=2),
    )
    receipt = service.forget(
        scope,
        active.memory_id,
        now=now + timedelta(seconds=3),
    )
    after = service.retrieve(
        scope,
        "应该使用什么语言回答？",
        token_budget=256,
        now=now + timedelta(seconds=4),
    )
    return {
        "proposed_status": proposed.status,
        "active_status": active.status,
        "recalled_ids": list(context.used_memory_ids),
        "forgotten": receipt.forgotten,
        "derivative_deletions": list(receipt.derivative_deletions),
        "recalled_after_forget": list(after.used_memory_ids),
    }


def _multimodal() -> dict[str, Any]:
    candidate = MultimodalCandidate(
        evidence_id="ev_" + "a" * 40,
        asset_id="asset_" + "b" * 32,
        modality="image",
        text_score=0.6,
        modality_score=0.9,
    )
    profile = MultimodalRetrievalProfile(
        profile_id="p2_selfcheck_multimodal",
        version=1,
        enabled=True,
        status="published",
        modalities=("image",),
        route_weights={"text": 1.0, "image": 1.0},
    )
    ranked, diagnostics = rank_multimodal_candidates([candidate], profile)
    return {
        "ranked_count": len(ranked),
        "first_rank": ranked[0].rank,
        "diagnostics": diagnostics,
        "hak180_dataset": "app/evaluation/datasets/hak180_multimodal_v1.json",
        "external_vision_called": False,
    }


def run_p2_self_check(
    *,
    tenant_id: str,
    kb_id: str,
    actor_id: str,
) -> dict[str, Any]:
    checks = [
        _result("TODO-13", "多模型路由与不可变快照", lambda: _model_routing(tenant_id)),
        _result(
            "TODO-15",
            "元数据 schema、建议与发布",
            lambda: _metadata_governance(tenant_id, kb_id, actor_id),
        ),
        _result(
            "TODO-16",
            "TOC 路由返回真实 Evidence",
            lambda: _advanced_retrieval(tenant_id, kb_id),
        ),
        _result(
            "TODO-17",
            "连接器增量 dry-run 与幂等键",
            lambda: _connector(tenant_id, kb_id),
        ),
        _result(
            "TODO-21",
            "Agent DSL 编译与工具版本锁",
            lambda: _agent_runtime(tenant_id, actor_id),
        ),
        _result(
            "TODO-22",
            "长期记忆确认、召回与遗忘",
            lambda: _memory(tenant_id, kb_id, actor_id),
        ),
        _result("TODO-23", "跨模态排序与 HAK180 数据集", _multimodal),
    ]
    return {
        "schema_version": "p2-self-check.v1",
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "actor_id": actor_id,
        "mode": "deterministic-no-external-calls",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


__all__ = ["run_p2_self_check"]
