"""P3 查询可靠性、安全和向量拓扑验收路由。

路由以 ``register_p3_routes`` 显式注册，避免导入时修改全局 FastAPI 应用。查询
运行接口由独立的 ``query_runtime_routes`` 提供；本模块只提供能力目录、自检、
内容安全判定、Milvus 只读 inventory 和验证页面。
"""

from __future__ import annotations

from collections import Counter

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.api.schemas.control_plane import ResourceEnvelope
from app.api.schemas.p3 import (
    P3SecurityInspectRequest,
    P3SelfCheckRequest,
    P3TopologyInventoryRequest,
)
from app.application.p3_self_check_service import run_p3_self_check
from app.application.rag_security_service import RagSecurityService
from app.application.vector_topology_service import VectorTopologyService
from app.domain.content_security import (
    ContentSecurityPolicy,
    SecurityAction,
)
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.models import KnowledgeBaseModel
from app.infra.persistence.sql.p3_models import ModelUsageRecordModel
from app.infra.persistence.sql.query_runtime_repository import (
    EventCursorExpired,
    QueryRuntimeRepository,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.vectorstore.milvus_topology_inspector import (
    MilvusReadOnlyTopologyInspector,
)
from app.application.observability_service import metric_registry, trace_recorder
from app.shared.runtime.logger import PROJECT_ROOT


_CAPABILITIES = (
    {
        "todo": "TODO-10",
        "name": "查询策略路由、预算与故障降级",
        "control": "required/optional route + deadline/budget/bulkhead/circuit breaker",
    },
    {
        "todo": "TODO-14",
        "name": "可观测性、审计、Token 与成本",
        "control": "trace parentage + redaction + low-cardinality metric + usage",
    },
    {
        "todo": "TODO-24",
        "name": "查询运行、会话与消息身份模型",
        "control": "Conversation/Turn/QueryRun/Message + immutable history snapshot",
    },
    {
        "todo": "TODO-25",
        "name": "多轮会话一致性与并发控制",
        "control": "turn sequence + optimistic CAS + ordered commit",
    },
    {
        "todo": "TODO-26",
        "name": "查询范围、租户授权与外发策略",
        "control": "requested ∩ entitlement + frozen scope hash + default-deny egress",
    },
    {
        "todo": "TODO-31",
        "name": "可靠流式、取消、续传、背压与事件协议",
        "control": "durable event seq + Last-Event-ID + bounded SSE + explicit cancel",
    },
    {
        "todo": "TODO-32",
        "name": "查询幂等、缓存、Singleflight 与可重放",
        "control": "full-version fingerprint + authorize-before-cache + replay manifest",
    },
    {
        "todo": "TODO-34",
        "name": "RAG 查询安全与间接提示注入防护",
        "control": "trust label + quarantine + SSRF/output guard + kill switch",
    },
    {
        "todo": "TODO-37",
        "name": "Milvus 拓扑、分区隔离与生命周期治理",
        "control": "compatibility fingerprint + placement + dry-run GC/migration",
    },
)


def _envelope(request: Request, data: object) -> ResourceEnvelope:
    return ResourceEnvelope(request_id=request.state.request_id, data=data)


def _assert_kb_scope(request: Request, kb_id: str) -> None:
    principal = request.state.principal
    with session_scope() as session:
        row = session.scalar(
            select(KnowledgeBaseModel.id).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == principal.tenant_id,
                KnowledgeBaseModel.deleted_at.is_(None),
            )
        )
    if row is None:
        # 不区分不存在与跨租户，避免资源探测。
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在或当前身份无权访问",
        )


def _security_policy() -> ContentSecurityPolicy:
    return ContentSecurityPolicy(
        policy_version="rag-security.p3-console.v1",
        web_enabled=True,
        tools_enabled=False,
        high_risk_sources_enabled=True,
        classifier_required_for_external=False,
        allowed_fetch_domains=(),
        allowed_output_link_domains=(),
    )


def _security_summary(
    *,
    action: SecurityAction,
    reason_codes: tuple[str, ...],
    decision_id: str,
    content_hash: str,
    source: str,
) -> dict[str, object]:
    return {
        "schema_version": "p3-security-inspection.v1",
        "source": source,
        "action": action.value,
        "reason_codes": list(reason_codes),
        "decision_id": decision_id,
        "content_hash": content_hash,
        "raw_content_returned": False,
        "safe_representation_returned": False,
    }


def register_p3_routes(app: FastAPI) -> None:
    @app.get(
        "/p3/html",
        include_in_schema=False,
    )
    def p3_html():
        html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "p3.html"
        return FileResponse(path=html_path, media_type="text/html; charset=utf-8")

    @app.get(
        "/api/v1/p3/capabilities",
        response_model=ResourceEnvelope,
        tags=["P3 查询可靠性验收"],
    )
    def p3_capabilities(request: Request):
        principal = request.state.principal
        return _envelope(
            request,
            {
                "schema_version": "p3-capability-catalog.v1",
                "tenant_id": principal.tenant_id,
                "capabilities": list(_CAPABILITIES),
                "verification_surfaces": {
                    "deterministic_self_check": "/api/v1/p3/self-check",
                    "content_security": "/api/v1/p3/security/inspect",
                    "milvus_read_only_inventory": "/api/v1/p3/topology/inventory",
                    "query_runs": "/api/v1/query-runs",
                    "query_observability": (
                        "/api/v1/p3/query-runs/{query_run_id}/observability"
                    ),
                },
                "safety_contract": {
                    "self_check_external_calls": False,
                    "self_check_persistence_writes": False,
                    "topology_inventory_read_only": True,
                    "topology_destructive_actions": False,
                },
            },
        )

    @app.post(
        "/api/v1/p3/self-check",
        response_model=ResourceEnvelope,
        tags=["P3 查询可靠性验收"],
    )
    async def p3_self_check(payload: P3SelfCheckRequest, request: Request):
        principal = request.state.principal
        _assert_kb_scope(request, payload.kb_id)
        result = await run_p3_self_check(
            tenant_id=principal.tenant_id,
            kb_id=payload.kb_id,
            actor_id=principal.user_id,
            question=payload.question,
        )
        return _envelope(request, result)

    @app.get(
        "/api/v1/p3/query-runs/{query_run_id}/observability",
        response_model=ResourceEnvelope,
        tags=["P3 查询可靠性验收"],
    )
    def p3_query_observability(query_run_id: str, request: Request):
        """返回当前用户运行的安全观测投影，不回传问题、Prompt 或证据正文。"""

        principal = request.state.principal
        with session_scope() as session:
            repository = QueryRuntimeRepository(
                session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
            run = repository.get_run(query_run_id)
            try:
                page = repository.read_event_page(
                    query_run_id,
                    after_seq=0,
                    limit=1000,
                )
                event_counts = Counter(item.type.value for item in page.events)
                stages = list(
                    dict.fromkeys(
                        item.stage for item in page.events if item.stage is not None
                    )
                )
                event_window = {
                    "available": True,
                    "first_available_seq": page.first_available_seq,
                    "last_available_seq": page.last_available_seq,
                    "event_counts": dict(sorted(event_counts.items())),
                    "stages": stages,
                }
            except EventCursorExpired:
                event_window = {
                    "available": False,
                    "reason_code": "EVENT_RETENTION_EXPIRED",
                    "event_counts": {},
                    "stages": [],
                }
            usage_rows = session.scalars(
                select(ModelUsageRecordModel)
                .where(
                    ModelUsageRecordModel.tenant_id == principal.tenant_id,
                    ModelUsageRecordModel.query_run_id == query_run_id,
                )
                .order_by(
                    ModelUsageRecordModel.fallback_sequence,
                    ModelUsageRecordModel.created_at,
                )
            ).all()

        spans = trace_recorder.records(trace_id=run.trace_id)
        result = dict(run.result or {})
        profile = dict(
            (run.config_snapshot or {}).get("retrieval_profile") or {}
        )
        scope = dict((run.config_snapshot or {}).get("authorized_scope") or {})
        usage = {
            "calls": sum(item.calls for item in usage_rows),
            "input_tokens": sum(item.input_tokens for item in usage_rows),
            "output_tokens": sum(item.output_tokens for item in usage_rows),
            "cache_read_tokens": sum(item.cache_read_tokens for item in usage_rows),
            "cost": round(sum(item.cost for item in usage_rows), 8),
            "currency": (
                next((item.currency for item in usage_rows), None)
                if usage_rows
                else None
            ),
            "estimated": any(item.estimated for item in usage_rows),
            "records": [
                {
                    "provider": item.provider,
                    "model_id": item.model_id,
                    "scene": item.scene,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "cache_read_tokens": item.cache_read_tokens,
                    "calls": item.calls,
                    "cost": item.cost,
                    "currency": item.currency,
                    "unit_price_version": item.unit_price_version,
                    "estimated": item.estimated,
                    "fallback_sequence": item.fallback_sequence,
                }
                for item in usage_rows
            ],
        }
        return _envelope(
            request,
            {
                "schema_version": "p3-query-observability.v1",
                "query_run_id": run.query_run_id,
                "trace_id": run.trace_id,
                "status": run.status.value,
                "current_stage": run.current_stage,
                "attempt": run.attempt,
                "retryable": run.retryable,
                "error": (
                    {
                        "code": run.error_code,
                        "message": run.safe_error_message,
                    }
                    if run.error_code
                    else None
                ),
                "scope": {
                    "scope_hash": scope.get("scope_hash"),
                    "policy_version": scope.get("policy_version"),
                    "entitlement_version": scope.get("entitlement_version"),
                    "active_index_count": len(scope.get("active_indexes") or ()),
                },
                "retrieval_profile": {
                    "profile_id": profile.get("profile_id"),
                    "version": profile.get("version"),
                    "config_hash": profile.get("config_hash"),
                },
                "answer_facts": {
                    "answer_id": result.get("answer_id"),
                    "citation_count": len(result.get("citations") or ()),
                    "source_count": len(result.get("sources") or ()),
                    "has_answer_envelope": bool(result.get("answer_envelope")),
                    "answer_body_returned": False,
                },
                "usage": usage,
                "event_window": event_window,
                "trace": {
                    "available_in_current_process": bool(spans),
                    "span_count": len(spans),
                    "spans": [
                        {
                            "span_id": item.span_id,
                            "parent_span_id": item.parent_span_id,
                            "name": item.name,
                            "started_at": item.started_at.isoformat(),
                            "ended_at": item.ended_at.isoformat(),
                            "duration_ms": item.duration_ms,
                            "status": item.status,
                            "attributes": item.attributes,
                            "error_code": item.error_code,
                        }
                        for item in spans
                    ],
                },
                "metrics": metric_registry.snapshot(),
                "safety": {
                    "query_returned": False,
                    "prompt_returned": False,
                    "evidence_content_returned": False,
                    "vectors_returned": False,
                },
            },
        )

    @app.post(
        "/api/v1/p3/security/inspect",
        response_model=ResourceEnvelope,
        tags=["P3 查询可靠性验收"],
    )
    def p3_security_inspect(
        payload: P3SecurityInspectRequest,
        request: Request,
    ):
        principal = request.state.principal
        _assert_kb_scope(request, payload.kb_id)
        service = RagSecurityService(policy=_security_policy())
        if payload.source == "user":
            secured = service.inspect_query(
                text=payload.text,
                message_id="p3-console-user-input",
                tenant_id=principal.tenant_id,
                classification=payload.data_classification,
            )
            result = _security_summary(
                action=secured.decision.action,
                reason_codes=secured.decision.reason_codes,
                decision_id=secured.decision.decision_id,
                content_hash=secured.decision.content_hash,
                source=payload.source,
            )
        else:
            candidate = {
                "chunk_id": "p3-console-content",
                "type": ("milvus" if payload.source == "knowledge_base" else "web"),
                "content": payload.text,
                "data_classification": payload.data_classification.value,
            }
            batch = service.inspect_evidence(
                [candidate],
                tenant_id=principal.tenant_id,
                default_classification=payload.data_classification,
            )
            if batch.approved:
                secured = batch.approved[0].content
                result = _security_summary(
                    action=secured.decision.action,
                    reason_codes=secured.decision.reason_codes,
                    decision_id=secured.decision.decision_id,
                    content_hash=secured.decision.content_hash,
                    source=payload.source,
                )
            else:
                quarantined = batch.quarantined[0]
                result = {
                    "schema_version": "p3-security-inspection.v1",
                    "source": payload.source,
                    "action": SecurityAction.QUARANTINE.value,
                    "reason_codes": list(quarantined.reason_codes),
                    "decision_id": quarantined.decision_id,
                    "content_hash": quarantined.content_hash,
                    "raw_content_returned": False,
                    "safe_representation_returned": False,
                }
        return _envelope(request, result)

    @app.post(
        "/api/v1/p3/topology/inventory",
        response_model=ResourceEnvelope,
        tags=["P3 查询可靠性验收"],
    )
    def p3_topology_inventory(
        payload: P3TopologyInventoryRequest,
        request: Request,
    ):
        configured_collection = milvus_gateway.chunks_collection
        physical_collection = payload.physical_collection or configured_collection
        if payload.physical_database != "default":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只读检查器仅允许当前服务绑定的 default 数据库",
            )
        if physical_collection != configured_collection:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只读检查器仅允许当前服务配置的知识块 collection",
            )
        client = milvus_gateway.client()
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Milvus 客户端当前不可用",
            )
        inspector = MilvusReadOnlyTopologyInspector(
            client,
            configured_database="default",
        )
        try:
            report = VectorTopologyService.audit_collection(
                inspector,
                physical_database=payload.physical_database,
                physical_collection=physical_collection,
                sample_limit=payload.sample_limit,
            )
        except Exception:
            # 不把连接 URI、驱动错误或远端响应细节回传浏览器。
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Milvus 只读拓扑检查失败，请结合 request_id 查看服务端日志",
            ) from None
        principal = request.state.principal
        return _envelope(
            request,
            {
                **report.model_dump(mode="json"),
                "tenant_id": principal.tenant_id,
                "sample_contains_content": False,
                "sample_contains_vectors": False,
                "write_operations_exposed": False,
            },
        )


__all__ = ["register_p3_routes"]
