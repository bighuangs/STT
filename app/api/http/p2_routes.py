"""P2 高级能力版本控制、dry-run 与网页验收 API。"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Query, Request, UploadFile, status
from sqlalchemy import func, select

from app.api.schemas.control_plane import ResourceEnvelope
from app.api.schemas.p2 import (
    AdvancedRetrievalPreviewRequest,
    AgentCompilePreviewRequest,
    ConnectorDryRunRequest,
    MetadataValueProposalRequest,
    ModelRoutingPreviewRequest,
    MultimodalRankRequest,
    P2CapabilityEnableRequest,
    P2CapabilityPublishRequest,
    P2CapabilityResourceCreate,
    P2CapabilityRunCreate,
    P2CapabilityRunTransition,
    P2SelfCheckRequest,
    ToolExecutionPreviewRequest,
)
from app.application.advanced_retrieval_service import AdvancedRetrievalService
from app.application.agent_runtime_service import (
    AgentCompiler,
    ToolGateway,
    ToolPolicyError,
)
from app.application.connector_sync_service import (
    ConnectorSyncError,
    IncrementalSyncPlanner,
)
from app.application.model_routing_service import ModelRouter, ModelRoutingError
from app.application.multimodal_service import (
    MediaPolicyError,
    MediaValidationError,
    MultimodalService,
    rank_multimodal_candidates,
)
from app.application.p2_control_service import validate_capability_payload
from app.application.p2_self_check_service import run_p2_self_check
from app.infra.llm.model_registry import InMemoryModelRegistry
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import (
    CapabilityResourceModel,
    DocumentModel,
    DocumentVersionModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.p2_repository import (
    CapabilityResourceRepository,
    CapabilityRunRepository,
    MetadataValueRepository,
)
from app.infra.persistence.sql.repositories import ResourceNotFoundError
from app.infra.tools import (
    DeterministicToolExecutor,
    InMemoryIdempotencyStore,
    InMemoryToolAuditSink,
    InMemoryToolCatalog,
)
from app.security.service import audit, require
from app.security.permissions import is_allowed
from app.shared.runtime.logger import PROJECT_ROOT


_CAPABILITIES = (
    {
        "todo": "TODO-13",
        "resource_type": "model_route",
        "name": "多模型供应商、路由与模型迁移",
        "default_enabled": False,
        "verification": "model-routing:preview",
    },
    {
        "todo": "TODO-15",
        "resource_type": "metadata_schema",
        "name": "元数据、标签与知识运营闭环",
        "default_enabled": False,
        "verification": "resource validation + metadata proposal",
    },
    {
        "todo": "TODO-16",
        "resource_type": "advanced_index",
        "name": "TOC、RAPTOR 与 GraphRAG",
        "default_enabled": False,
        "verification": "self-check TOC route",
    },
    {
        "todo": "TODO-17",
        "resource_type": "connector_source",
        "name": "数据源连接器与增量同步",
        "default_enabled": False,
        "verification": "connectors:dry-run",
    },
    {
        "todo": "TODO-21",
        "resource_type": "agent_definition",
        "name": "Agent 工具运行时、MCP 与插件能力",
        "default_enabled": False,
        "verification": "agents:compile + tools:execute",
    },
    {
        "todo": "TODO-22",
        "resource_type": "memory_policy",
        "name": "长期记忆与可控遗忘",
        "default_enabled": False,
        "verification": "self-check confirm/recall/forget",
    },
    {
        "todo": "TODO-23",
        "resource_type": "multimodal_profile",
        "name": "多模态文档理解与跨模态检索",
        "default_enabled": False,
        "verification": "HAK180 dataset + image inspect + ranking",
    },
)


def _envelope(request: Request, data: object) -> ResourceEnvelope:
    return ResourceEnvelope(request_id=request.state.request_id, data=data)


def _resource_repository(request: Request, session) -> CapabilityResourceRepository:
    principal = request.state.principal
    return CapabilityResourceRepository(
        session,
        tenant_id=principal.tenant_id,
        actor_id=principal.user_id,
    )


def _assert_kb_scope(session, request: Request, kb_id: str) -> None:
    principal = request.state.principal
    row = session.scalar(
        select(KnowledgeBaseModel.id).where(
            KnowledgeBaseModel.id == kb_id,
            KnowledgeBaseModel.tenant_id == principal.tenant_id,
            KnowledgeBaseModel.deleted_at.is_(None),
        )
    )
    if row is None:
        raise ResourceNotFoundError("knowledge base not found in tenant scope")


def _assert_document_version_scope(
    session,
    request: Request,
    *,
    kb_id: str,
    document_id: str,
    document_version_id: str,
) -> None:
    """验证 document/version 确实属于当前租户的指定知识库。

    图片检查接口当前不落盘，但这里仍提前冻结完整作用域，避免未来把同一路由接到
    持久资产仓储时误信客户端提交的 document/version 标识。
    """

    principal = request.state.principal
    row = session.execute(
        select(DocumentModel.id, DocumentVersionModel.id)
        .join(
            DocumentVersionModel,
            DocumentVersionModel.document_id == DocumentModel.id,
        )
        .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
        .where(
            KnowledgeBaseModel.tenant_id == principal.tenant_id,
            KnowledgeBaseModel.deleted_at.is_(None),
            DocumentModel.kb_id == kb_id,
            DocumentModel.id == document_id,
            DocumentModel.deleted_at.is_(None),
            DocumentVersionModel.id == document_version_id,
        )
    ).first()
    if row is None:
        raise ResourceNotFoundError(
            "document version not found in tenant/knowledge-base/document scope"
        )


def _scope_source(request: Request, tenant_id: str, kb_id: str) -> None:
    principal = request.state.principal
    if tenant_id != principal.tenant_id:
        raise ValueError("请求 tenant 与登录 tenant 不一致")
    with session_scope() as session:
        _assert_kb_scope(session, request, kb_id)


def _load_hak180_dataset() -> dict:
    path = PROJECT_ROOT / "app" / "evaluation" / "datasets" / "hak180_multimodal_v1.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    pdf_path = PROJECT_ROOT / dataset["source"]["relative_path"]
    current_hash = sha256(pdf_path.read_bytes()).hexdigest()
    dataset["runtime_verification"] = {
        "pdf_exists": pdf_path.is_file(),
        "pdf_hash_matches": current_hash == dataset["source"]["content_sha256"],
        "current_pdf_sha256": current_hash,
        "dataset_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    render_path = PROJECT_ROOT / "tmp" / "pdfs" / "hak180-page1.png"
    if render_path.is_file():
        render_hash = sha256(render_path.read_bytes()).hexdigest()
        dataset["runtime_verification"].update(
            {
                "review_render_exists": True,
                "review_render_hash_matches": (
                    render_hash == dataset["source"]["render"]["content_sha256"]
                ),
                "review_render_sha256": render_hash,
            }
        )
    else:
        dataset["runtime_verification"]["review_render_exists"] = False
    return dataset


def register_p2_routes(app: FastAPI) -> None:
    @app.get(
        "/api/v1/p2/capabilities",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def capability_catalog(request: Request):
        principal = request.state.principal
        with session_scope() as session:
            rows = session.execute(
                select(
                    CapabilityResourceModel.resource_type,
                    CapabilityResourceModel.status,
                    CapabilityResourceModel.enabled,
                    func.count(CapabilityResourceModel.id),
                )
                .where(CapabilityResourceModel.tenant_id == principal.tenant_id)
                .group_by(
                    CapabilityResourceModel.resource_type,
                    CapabilityResourceModel.status,
                    CapabilityResourceModel.enabled,
                )
            ).all()
        counts: dict[str, dict[str, int]] = {}
        for resource_type, resource_status, enabled, count in rows:
            item = counts.setdefault(
                resource_type,
                {"draft": 0, "published": 0, "retired": 0, "enabled": 0},
            )
            item[resource_status] = item.get(resource_status, 0) + int(count)
            if enabled:
                item["enabled"] += int(count)
        return _envelope(
            request,
            {
                "schema_version": "p2-capability-catalog.v1",
                "tenant_id": principal.tenant_id,
                "capabilities": [
                    {**item, "resource_counts": counts.get(item["resource_type"], {})}
                    for item in _CAPABILITIES
                ],
                "execution_policy": {
                    "all_new_routes_default_disabled": True,
                    "external_calls_in_self_check": False,
                    "runtime_requires_published_and_enabled_resource": True,
                },
            },
        )

    @app.get(
        "/api/v1/p2/resources",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def list_resources(
        request: Request,
        resource_type: str | None = None,
        kb_id: str | None = None,
        resource_status: str | None = Query(default=None, alias="status"),
    ):
        with session_scope() as session:
            repository = _resource_repository(request, session)
            if kb_id is not None:
                _assert_kb_scope(session, request, kb_id)
            return _envelope(
                request,
                {
                    "items": repository.list(
                        resource_type=resource_type,
                        kb_id=kb_id,
                        status=resource_status,
                    )
                },
            )

    @app.get(
        "/api/v1/p2/resources/{resource_id}",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def get_resource(resource_id: str, request: Request):
        with session_scope() as session:
            repository = _resource_repository(request, session)
            return _envelope(
                request,
                repository.as_dict(repository.get(resource_id)),
            )

    @app.post(
        "/api/v1/p2/resources",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["P2 能力实验台"],
    )
    def create_resource(
        payload: P2CapabilityResourceCreate,
        request: Request,
    ):
        principal = request.state.principal
        normalized = validate_capability_payload(
            resource_type=payload.resource_type,
            schema_version=payload.schema_version,
            payload=payload.payload,
            tenant_id=principal.tenant_id,
            kb_id=payload.kb_id,
            actor_id=principal.user_id,
        )
        with session_scope() as session:
            repository = _resource_repository(request, session)
            row = repository.create(
                resource_type=payload.resource_type,
                logical_id=payload.logical_id,
                version=payload.version,
                schema_version=payload.schema_version,
                payload=normalized,
                kb_id=payload.kb_id,
            )
            return _envelope(request, row)

    @app.post(
        "/api/v1/p2/resources/{resource_id}:publish",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def publish_resource(
        resource_id: str,
        payload: P2CapabilityPublishRequest,
        request: Request,
    ):
        with session_scope() as session:
            repository = _resource_repository(request, session)
            return _envelope(
                request,
                repository.publish(resource_id, enable=payload.enable),
            )

    @app.post(
        "/api/v1/p2/resources/{resource_id}:enable",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def enable_resource(
        resource_id: str,
        payload: P2CapabilityEnableRequest,
        request: Request,
    ):
        with session_scope() as session:
            repository = _resource_repository(request, session)
            return _envelope(
                request,
                repository.set_enabled(resource_id, payload.enabled),
            )

    @app.post(
        "/api/v1/p2/resources/{resource_id}:retire",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def retire_resource(resource_id: str, request: Request):
        with session_scope() as session:
            repository = _resource_repository(request, session)
            return _envelope(request, repository.retire(resource_id))

    @app.post(
        "/api/v1/p2/runs",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["P2 能力实验台"],
    )
    def create_run(payload: P2CapabilityRunCreate, request: Request):
        principal = request.state.principal
        with session_scope() as session:
            repository = CapabilityRunRepository(
                session,
                tenant_id=principal.tenant_id,
            )
            row = repository.create(
                run_type=payload.run_type,
                trace_id=request.state.request_id,
                input_payload=payload.input_payload,
                input_summary=payload.input_summary,
                resource_id=payload.resource_id,
                idempotency_key=payload.idempotency_key,
            )
            return _envelope(request, row)

    @app.post(
        "/api/v1/p2/runs/{run_id}:transition",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def transition_run(
        run_id: str,
        payload: P2CapabilityRunTransition,
        request: Request,
    ):
        principal = request.state.principal
        with session_scope() as session:
            repository = CapabilityRunRepository(
                session,
                tenant_id=principal.tenant_id,
            )
            return _envelope(
                request,
                repository.set_status(
                    run_id,
                    status=payload.status,
                    output=payload.output,
                    error_code=payload.error_code,
                    error_detail=payload.error_detail,
                ),
            )

    @app.post(
        "/api/v1/p2/metadata-values",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["P2 能力实验台"],
    )
    def propose_metadata(
        payload: MetadataValueProposalRequest,
        request: Request,
    ):
        principal = request.state.principal
        if payload.target.tenant_id != principal.tenant_id:
            raise ValueError("元数据 target tenant 与登录 tenant 不一致")
        with session_scope() as session:
            _assert_kb_scope(session, request, payload.target.kb_id)
            repository = MetadataValueRepository(
                session,
                tenant_id=principal.tenant_id,
                actor_id=principal.user_id,
            )
            row = repository.propose(
                schema_resource_id=payload.schema_resource_id,
                target=payload.target,
                field_key=payload.field_key,
                value=payload.value,
                source=payload.source,
                confidence=payload.confidence,
                provenance=payload.provenance,
                enhancement=payload.enhancement,
                schema_revision=payload.schema_revision,
            )
            return _envelope(request, row)

    @app.post(
        "/api/v1/p2/metadata-values/{value_id}:publish",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def publish_metadata(value_id: str, request: Request):
        principal = request.state.principal
        with session_scope() as session:
            repository = MetadataValueRepository(
                session,
                tenant_id=principal.tenant_id,
                actor_id=principal.user_id,
            )
            requires_compliance = repository.requires_compliance_permission(value_id)
            allow_compliance = is_allowed(
                principal.roles,
                principal.scopes,
                "metadata:compliance",
            )
            if requires_compliance:
                audit(
                    principal,
                    "metadata:compliance",
                    "allowed" if allow_compliance else "denied",
                    request.state.request_id,
                    request.client.host if request.client else None,
                    {
                        "path": request.url.path,
                        "metadata_value_id": value_id,
                    },
                )
                require(principal, "metadata:compliance")
            return _envelope(
                request,
                repository.publish(
                    value_id,
                    allow_compliance_controlled=allow_compliance,
                ),
            )

    @app.post(
        "/api/v1/p2/metadata-values/{value_id}:reject",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def reject_metadata(value_id: str, request: Request):
        principal = request.state.principal
        with session_scope() as session:
            repository = MetadataValueRepository(
                session,
                tenant_id=principal.tenant_id,
                actor_id=principal.user_id,
            )
            return _envelope(request, repository.reject(value_id))

    @app.post(
        "/api/v1/p2/model-routing:preview",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def preview_model_routing(
        payload: ModelRoutingPreviewRequest,
        request: Request,
    ):
        principal = request.state.principal
        if (
            payload.policy.tenant_id != principal.tenant_id
            or payload.request.tenant_id != principal.tenant_id
        ):
            raise ValueError("模型路由 tenant 与登录 tenant 不一致")
        if payload.request.kb_id:
            with session_scope() as session:
                _assert_kb_scope(session, request, payload.request.kb_id)
        registry = InMemoryModelRegistry()
        for provider in payload.providers:
            registry.register_provider(provider)
        health = {item.model_id: item for item in payload.health}
        for model in payload.models:
            registry.register_model(model, health=health.get(model.model_id))
        try:
            plan = ModelRouter(registry).route(payload.policy, payload.request)
        except ModelRoutingError as exc:
            raise ValueError(f"{exc.code}: {exc}") from exc
        return _envelope(request, plan.model_dump(mode="json"))

    @app.post(
        "/api/v1/p2/connectors:dry-run",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def connector_dry_run(
        payload: ConnectorDryRunRequest,
        request: Request,
    ):
        source = payload.data_source
        _scope_source(
            request,
            source.scope.tenant_id,
            source.scope.kb_id,
        )
        try:
            plan = IncrementalSyncPlanner().plan_incremental(
                source=source,
                cursor=payload.cursor,
                page=payload.page,
                bindings=payload.bindings,
                run_id=f"dry_{uuid4().hex}",
                dry_run=True,
            )
        except ConnectorSyncError as exc:
            raise ValueError(str(exc)) from exc
        return _envelope(request, plan.model_dump(mode="json"))

    @app.post(
        "/api/v1/p2/advanced-retrieval:preview",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def preview_advanced_retrieval(
        payload: AdvancedRetrievalPreviewRequest,
        request: Request,
    ):
        retrieval_request = payload.request
        _scope_source(
            request,
            retrieval_request.tenant_id,
            retrieval_request.kb_id,
        )
        evidence_by_id = {str(item.evidence_id): item for item in payload.evidence}
        result = AdvancedRetrievalService().search(
            retrieval_request,
            snapshots=payload.snapshots,
            evidence_by_id=evidence_by_id,
        )
        return _envelope(request, result.model_dump(mode="json"))

    @app.post(
        "/api/v1/p2/agents:compile",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def compile_agent(
        payload: AgentCompilePreviewRequest,
        request: Request,
    ):
        principal = request.state.principal
        if payload.agent.tenant_id != principal.tenant_id:
            raise ValueError("Agent tenant 与登录 tenant 不一致")
        catalog = InMemoryToolCatalog()
        for plugin in payload.plugins:
            catalog.register_plugin(plugin)
        for tool in payload.tools:
            if tool.tenant_id != principal.tenant_id:
                raise ValueError("Tool tenant 与登录 tenant 不一致")
            catalog.register_tool(tool)
        plan = AgentCompiler(catalog).compile(payload.agent)
        return _envelope(request, plan.model_dump(mode="json"))

    @app.post(
        "/api/v1/p2/tools:execute",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def execute_deterministic_tool(
        payload: ToolExecutionPreviewRequest,
        request: Request,
    ):
        principal = request.state.principal
        if payload.tool.kind != "deterministic":
            raise ValueError("验收 API 只执行显式注册的 deterministic 工具")
        if (
            payload.tool.tenant_id != principal.tenant_id
            or payload.call.principal.tenant_id != principal.tenant_id
            or payload.call.principal.principal_id != principal.user_id
        ):
            raise ValueError("工具调用身份与登录主体不一致")
        catalog = InMemoryToolCatalog()
        catalog.register_tool(payload.tool)
        executor = DeterministicToolExecutor()
        executor.register(
            payload.tool.reference,
            lambda _: dict(payload.deterministic_output),
        )
        gateway = ToolGateway(
            catalog=catalog,
            executor=executor,
            audit_sink=InMemoryToolAuditSink(),
            idempotency_store=InMemoryIdempotencyStore(),
        )
        try:
            result = gateway.execute(payload.call)
        except ToolPolicyError as exc:
            raise ValueError(f"{exc.code}: {exc}") from exc
        return _envelope(request, result.model_dump(mode="json"))

    @app.post(
        "/api/v1/p2/multimodal:rank",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def rank_multimodal(
        payload: MultimodalRankRequest,
        request: Request,
    ):
        ranked, diagnostics = rank_multimodal_candidates(
            list(payload.candidates),
            payload.profile,
        )
        return _envelope(
            request,
            {
                "items": [item.model_dump(mode="json") for item in ranked],
                "diagnostics": diagnostics,
            },
        )

    @app.post(
        "/api/v1/p2/multimodal/images:inspect",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    async def inspect_image(
        request: Request,
        file: UploadFile = File(...),
        source_kind: Literal["query_attachment", "document_asset"] = Query(
            default="query_attachment"
        ),
        kb_id: str | None = None,
        document_id: str | None = None,
        document_version_id: str | None = None,
        page_no: int | None = Query(default=None, ge=1),
        x0: float | None = Query(default=None, ge=0, le=1),
        y0: float | None = Query(default=None, ge=0, le=1),
        x1: float | None = Query(default=None, ge=0, le=1),
        y1: float | None = Query(default=None, ge=0, le=1),
    ):
        principal = request.state.principal
        content = await file.read(25 * 1024 * 1024 + 1)
        if source_kind == "document_asset":
            if not all((kb_id, document_id, document_version_id, page_no)):
                raise ValueError("document_asset requires KB/document/version/page and bbox")
            if any(value is None for value in (x0, y0, x1, y1)):
                raise ValueError("document_asset requires a normalized bbox")
            with session_scope() as session:
                _assert_kb_scope(session, request, str(kb_id))
                _assert_document_version_scope(
                    session,
                    request,
                    kb_id=str(kb_id),
                    document_id=str(document_id),
                    document_version_id=str(document_version_id),
                )
        service = MultimodalService()
        try:
            asset = service.ingest_image(
                content,
                claimed_media_type=file.content_type or "application/octet-stream",
                tenant_id=principal.tenant_id,
                source_kind=source_kind,
                kb_id=kb_id if source_kind == "document_asset" else None,
                document_id=document_id if source_kind == "document_asset" else None,
                document_version_id=(
                    document_version_id if source_kind == "document_asset" else None
                ),
                owner_user_id=(principal.user_id if source_kind == "query_attachment" else None),
                page_no=page_no if source_kind == "document_asset" else None,
                bbox=(
                    (float(x0), float(y0), float(x1), float(y1))
                    if source_kind == "document_asset"
                    and all(value is not None for value in (x0, y0, x1, y1))
                    else None
                ),
            )
        except (MediaPolicyError, MediaValidationError) as exc:
            raise ValueError(str(exc)) from exc
        return _envelope(
            request,
            {
                "asset": asset.model_dump(mode="json"),
                "bytes_persisted": False,
                "note": "inspect 使用请求内存储；正式资产需通过已发布导入制品接入对象存储。",
            },
        )

    @app.get(
        "/api/v1/p2/hak180-multimodal-dataset",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def hak180_multimodal_dataset(request: Request):
        return _envelope(request, _load_hak180_dataset())

    @app.post(
        "/api/v1/p2/self-check",
        response_model=ResourceEnvelope,
        tags=["P2 能力实验台"],
    )
    def p2_self_check(payload: P2SelfCheckRequest, request: Request):
        principal = request.state.principal
        if payload.persist_demo_resources:
            require(principal, "p2:manage")
        with session_scope() as session:
            _assert_kb_scope(session, request, payload.kb_id)
        result = run_p2_self_check(
            tenant_id=principal.tenant_id,
            kb_id=payload.kb_id,
            actor_id=principal.user_id,
        )
        if payload.persist_demo_resources:
            demo_payload = {
                "auto_candidate_extraction_enabled": False,
                "auto_activation_enabled": False,
                "preference_ttl_days": 365,
                "profile_fact_ttl_days": 180,
                "episodic_ttl_days": 30,
                "max_context_tokens": 256,
                "max_hits": 8,
                "embedding_version": "memory-lexical/1.0.0",
            }
            with session_scope() as session:
                repository = _resource_repository(request, session)
                row = repository.create(
                    resource_type="memory_policy",
                    logical_id="p2_selfcheck_memory_policy",
                    version="1.0.0",
                    schema_version="memory-policy.v1",
                    payload=demo_payload,
                    kb_id=payload.kb_id,
                )
                if row["status"] == "draft":
                    row = repository.publish(row["id"], enable=False)
            result["persisted_demo_resource"] = row
        return _envelope(request, result)


__all__ = ["register_p2_routes"]
