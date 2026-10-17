"""查询服务 HTTP 组合根；回答发布只通过可持久 QueryRun 路由。"""

import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from mimetypes import guess_type
from pathlib import Path

# 兼容直接以 `python query_server.py` 方式启动，提前把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.query import HistoryItem, HistoryResponse, QueryRequest, QueryResponse
from app.bootstrap.observability import observability_lifespan
from app.domain.answer import AnswerEnvelope
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config import settings
from app.infra.persistence.history_repository import history_repository
from app.rag.query.answer_envelope_service import legacy_envelope
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_FAILED,
    get_done_task_list,
    get_task_result,
    update_task_status,
)
from app.api.http.auth_routes import register_auth_routes
from app.api.http.evaluation_routes import register_evaluation_routes
from app.api.http.p2_routes import register_p2_routes
from app.api.http.p3_routes import register_p3_routes
from app.api.http.query_runtime_routes import _execute_query_run, register_query_runtime_routes
from app.api.http.retrieval_routes import register_retrieval_routes
from app.application.evaluation_service import bootstrap_evaluation
from app.application.evaluation_runtime_maintenance import (
    evaluation_execution_owner,
    interrupt_owned_evaluation_runs,
    recover_stale_evaluation_runs,
    run_evaluation_recovery_loop,
)
from app.application.query_runtime_maintenance import (
    recover_expired_query_runs,
    run_query_runtime_recovery_loop,
)
from app.infrastructure.query_runtime_durability import (
    DurableQueryRecoveryController,
    run_durable_query_recovery_loop,
)
from app.security.http import install_security
from app.security.service import bootstrap_security
from app.security.service import (
    AuthorizationError,
    SessionKnowledgeBaseConflict,
    assert_query_session_owner,
    register_query_session,
)
from app.security.principal import Principal, reset_current_principal, set_current_principal
from app.infra.persistence.sql.repositories import (
    DEFAULT_KB_ID,
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.api.http.health_routes import register_health_routes
from app.api.http.ui_routes import register_ui_routes
from app.infra.health import close_runtime_resources, validate_startup
from app.rag.query.evidence_service import get_authorized_evidence
from app.rag.query.evidence_asset_service import (
    evidence_image_assets,
    evidence_image_projection,
    image_asset_is_manifested,
)
from app.application.control_plane_service import get_document_version
from app.infra.security.sql_query_authorization import resolve_egress_policy
from app.infra.security.sql_query_authorization import SqlEntitlementResolver
from app.domain.query_security import RequestedRetrievalScope, ScopeAuthorizationError
from app.infra.object_storage.minio_gateway import minio_gateway
from minio.error import S3Error


_DURABLE_RECOVERY_SHUTDOWN_GRACE_SECONDS = 2.0


class LegacyQueryPathRetiredError(RuntimeError):
    """旧 /query 执行路径已退役，回答必须通过可持久 QueryRun 提交。"""

    code = "LEGACY_QUERY_PATH_RETIRED"


@asynccontextmanager
async def lifespan(_: FastAPI):
    with observability_lifespan("query-api"):
        bootstrap_security()
        bootstrap_evaluation()
        validate_startup("query")
        evaluation_recovery = recover_stale_evaluation_runs()
        if evaluation_recovery["scanned"]:
            logger.info("评测运行启动恢复完成：{}", evaluation_recovery)
        recovery = recover_expired_query_runs()
        if recovery["scanned"]:
            logger.info("查询运行启动恢复完成：{}", recovery)
        recovery_stop = asyncio.Event()
        recovery_task = asyncio.create_task(
            run_query_runtime_recovery_loop(
                recovery_stop,
                interval_seconds=settings.query_recovery_interval_seconds,
            ),
            name="query-runtime-recovery",
        )
        durable_recovery_stop = asyncio.Event()
        durable_recovery_controller = DurableQueryRecoveryController()

        def execute_recovered_query(query_run_id, principal, execution_query):
            _execute_query_run(
                query_run_id,
                principal,
                execution_query,
                recovery_controller=durable_recovery_controller,
            )

        durable_recovery_task = asyncio.create_task(
            run_durable_query_recovery_loop(
                durable_recovery_stop,
                execute=execute_recovered_query,
                interval_seconds=settings.query_recovery_interval_seconds,
            ),
            name="query-checkpoint-recovery",
        )
        evaluation_recovery_stop = asyncio.Event()
        evaluation_recovery_task = asyncio.create_task(
            run_evaluation_recovery_loop(
                evaluation_recovery_stop,
                interval_seconds=settings.query_recovery_interval_seconds,
            ),
            name="evaluation-runtime-recovery",
        )
        try:
            yield
        finally:
            recovery_stop.set()
            durable_recovery_stop.set()
            evaluation_recovery_stop.set()
            shutdown_checkpoint_recovery = durable_recovery_controller.stop_and_interrupt_bounded(
                timeout_seconds=_DURABLE_RECOVERY_SHUTDOWN_GRACE_SECONDS,
            )
            if shutdown_checkpoint_recovery["scanned"]:
                logger.info(
                    "可靠查询 checkpoint 关闭收敛完成：{}",
                    shutdown_checkpoint_recovery,
                )
            if shutdown_checkpoint_recovery.get("timed_out"):
                logger.warning(
                    "query checkpoint shutdown fence exceeded grace: code={} residual_workers={}",
                    "QUERY_RECOVERY_SHUTDOWN_FENCE_TIMEOUT",
                    1,
                )
            await asyncio.gather(
                recovery_task,
                evaluation_recovery_task,
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(durable_recovery_task),
                    timeout=_DURABLE_RECOVERY_SHUTDOWN_GRACE_SECONDS,
                )
            except TimeoutError:
                # 同步 provider SDK 无法由 asyncio 强制中止。超过 grace 后停止
                # 等待并交给进程管理器 hard-stop；started attempt 会在下次恢复时
                # 诚实收敛为 interrupted，绝不自动重放。
                durable_recovery_task.cancel()
                await asyncio.gather(durable_recovery_task, return_exceptions=True)
                logger.warning(
                    "query checkpoint recovery shutdown exceeded grace: "
                    "code={} residual_workers={}",
                    "QUERY_RECOVERY_SHUTDOWN_GRACE_EXCEEDED",
                    1,
                )
            shutdown_recovery = interrupt_owned_evaluation_runs(evaluation_execution_owner())
            if shutdown_recovery["scanned"]:
                logger.info("评测运行关闭收敛完成：{}", shutdown_recovery)
            close_runtime_resources()


app = FastAPI(
    title=settings.query_app_name,
    description="企业化 RAG 查询服务，负责问答、SSE 输出与历史记录查询。",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
register_auth_routes(app)
register_health_routes(app, "query")
register_ui_routes(app, "query")
register_evaluation_routes(app)
register_retrieval_routes(app)
register_p2_routes(app)
register_query_runtime_routes(app)
register_p3_routes(app)
install_security(app)


@app.exception_handler(ResourceNotFoundError)
async def handle_query_not_found(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "request_id": request.state.request_id,
            "error": {"code": "NOT_FOUND", "message": str(exc)},
        },
    )


@app.exception_handler(AuthorizationError)
async def handle_query_forbidden(request: Request, exc: AuthorizationError):
    error_code = (
        "SESSION_KB_CONFLICT" if isinstance(exc, SessionKnowledgeBaseConflict) else "FORBIDDEN"
    )
    return JSONResponse(
        status_code=403,
        content={
            "request_id": request.state.request_id,
            "error": {"code": error_code, "message": str(exc)},
        },
    )


@app.exception_handler(ResourceConflictError)
async def handle_evaluation_conflict(request: Request, exc: ResourceConflictError):
    return JSONResponse(
        status_code=409,
        content={
            "request_id": request.state.request_id,
            "error": {"code": "STATE_CONFLICT", "message": str(exc)},
        },
    )


@app.exception_handler(ValueError)
async def handle_evaluation_value_error(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request.state.request_id,
            "error": {"code": "VALIDATION_ERROR", "message": str(exc)},
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation(request: Request, exc: RequestValidationError):
    # 不回显 input 字段，避免密码或大段用户内容进入错误响应/日志。
    details = [
        {
            "location": [str(item) for item in error.get("loc", ())],
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request.state.request_id,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "请求参数校验失败",
                "details": details,
            },
        },
    )


def new_session_id() -> str:
    """
    生成新的会话 ID。

    Returns:
        str: 基于 UUID 生成的唯一会话标识。
    """
    return str(uuid.uuid4())


def build_query_security_context(
    principal: Principal,
    kb_id: str,
    session_policy: dict,
) -> dict:
    """冻结旧/新查询入口共用的授权、外发与内容安全策略。"""

    llm_policy = resolve_egress_policy(
        tenant_id=principal.tenant_id,
        kb_ids=(kb_id,),
        purpose="llm_generation",
        allowed_providers=("configured-llm",),
    )
    web_policy = resolve_egress_policy(
        tenant_id=principal.tenant_id,
        kb_ids=(kb_id,),
        purpose="web_search",
        allowed_providers=("dashscope-web",),
        allowed_domains=settings.query_web_allowed_domains,
    )
    return {
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
        "authorized_scope": session_policy.get("authorized_scope") or {},
        "data_classification": session_policy.get("data_classification") or "internal",
        "external_llm_allowed": bool(session_policy.get("external_llm_allowed")),
        "egress_policies": {
            "llm_generation": llm_policy.model_dump(mode="json"),
            "web_search": web_policy.model_dump(mode="json"),
        },
        "content_security_policy": {
            "policy_version": "rag-security.v1",
            "web_enabled": bool(web_policy.enabled),
            "high_risk_sources_enabled": bool(web_policy.enabled),
            "allowed_fetch_domains": list(settings.query_web_allowed_domains),
            "allowed_output_link_domains": list(settings.query_output_link_allowed_domains),
        },
    }


def invoke_query(
    session_id: str,
    query: str,
    is_stream: bool,
    kb_id: str | None = None,
    web_search_allowed: bool = False,
    retrieval_policy: dict | None = None,
    retrieval_filter: dict | None = None,
    security_context: dict | None = None,
) -> dict:
    """阻断未持久化 AnswerProof 的旧图直调路径。"""

    del (
        session_id,
        query,
        is_stream,
        kb_id,
        web_search_allowed,
        retrieval_policy,
        retrieval_filter,
        security_context,
    )
    raise LegacyQueryPathRetiredError(
        "legacy query execution is retired; use POST /api/v1/query-runs"
    )


def run_stream_query_background(
    session_id: str,
    query: str,
    kb_id: str | None = None,
    principal: Principal | None = None,
    web_search_allowed: bool = False,
    retrieval_policy: dict | None = None,
    retrieval_filter: dict | None = None,
    security_context: dict | None = None,
) -> None:
    """
    在后台执行一次流式查询任务。

    Args:
        session_id: 当前流式查询对应的会话 ID。
        query: 用户原始问题。
    """
    context_token = set_current_principal(principal) if principal else None
    try:
        state = invoke_query(
            session_id=session_id,
            query=query,
            is_stream=True,
            kb_id=kb_id,
            web_search_allowed=web_search_allowed,
            retrieval_policy=retrieval_policy,
            retrieval_filter=retrieval_filter,
            security_context=security_context,
        )
        raw_answer_envelope = state.get("answer_envelope")
        answer_envelope = (
            AnswerEnvelope.model_validate(raw_answer_envelope).model_dump(mode="json")
            if raw_answer_envelope
            else None
        )
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": get_task_result(session_id, "answer") or state.get("answer", ""),
                "status": "completed",
                "image_urls": state.get("image_urls", []),
                "answer_id": state.get("answer_id", ""),
                "citations": state.get("citations", []),
                "sources": state.get("sources", []),
                "citation_validation": state.get("citation_validation", {}),
                "retrieval_profile": {
                    "profile_id": state.get("retrieval_profile_id", ""),
                    "version": state.get("retrieval_profile_version", 0),
                    "config_hash": state.get("retrieval_profile_hash", ""),
                },
                "retrieval_diagnostics": state.get("retrieval_diagnostics", {}),
                "query_plan": state.get("query_plan", {}),
                "fusion_diagnostics": state.get("fusion_diagnostics", {}),
                "context_pack_summary": state.get("context_pack_summary", {}),
                "evidence_assessment": state.get("evidence_assessment", {}),
                "answer_decision": state.get("answer_decision", {}),
                "answer_envelope": answer_envelope,
            },
        )
    except Exception as exc:
        logger.exception(f"流式查询执行失败，session_id={session_id}, error={exc}")
        update_task_status(session_id, TASK_STATUS_FAILED, True)
        push_to_session(
            session_id,
            SSEEvent.ERROR,
            {"message": "查询执行失败，请检查日志或稍后重试。"},
        )
    finally:
        if context_token is not None:
            reset_current_principal(context_token)


def start_stream_query(
    background_tasks: BackgroundTasks,
    query: str,
    session_id: str | None = None,
    kb_id: str | None = None,
    principal: Principal | None = None,
    retrieval_filter: dict | None = None,
) -> QueryResponse:
    """
    启动一条流式查询任务。

    Args:
        background_tasks: FastAPI 后台任务对象。
        query: 用户原始问题。
        session_id: 可选会话 ID；为空时自动生成。

    Returns:
        QueryResponse: 返回处理中提示与当前会话 ID。
    """
    final_session_id = session_id or new_session_id()
    assert principal is not None
    resolved_kb_id = kb_id or DEFAULT_KB_ID
    policy = register_query_session(principal, final_session_id, resolved_kb_id)
    security_context = build_query_security_context(principal, resolved_kb_id, policy)
    create_sse_queue(final_session_id)
    background_tasks.add_task(
        run_stream_query_background,
        final_session_id,
        query,
        resolved_kb_id,
        principal,
        policy["web_search_allowed"],
        policy.get("retrieval_profile"),
        retrieval_filter,
        security_context,
    )
    retrieval_identity = {
        key: policy.get("retrieval_profile", {}).get(key)
        for key in ("profile_id", "version", "config_hash")
    }
    return QueryResponse(
        message="结果正在处理中...",
        session_id=final_session_id,
        retrieval_profile=retrieval_identity,
    )


def execute_query(
    query: str,
    session_id: str | None = None,
    kb_id: str | None = None,
    principal: Principal | None = None,
    retrieval_filter: dict | None = None,
) -> QueryResponse:
    """
    以非流式方式执行查询。

    Args:
        query: 用户原始问题。
        session_id: 可选会话 ID；为空时自动生成。

    Returns:
        QueryResponse: 包含最终答案和已完成节点的响应对象。
    """
    final_session_id = session_id or new_session_id()
    assert principal is not None
    resolved_kb_id = kb_id or DEFAULT_KB_ID
    policy = register_query_session(principal, final_session_id, resolved_kb_id)
    security_context = build_query_security_context(principal, resolved_kb_id, policy)
    state = invoke_query(
        session_id=final_session_id,
        query=query,
        is_stream=False,
        kb_id=resolved_kb_id,
        web_search_allowed=policy["web_search_allowed"],
        retrieval_policy=policy.get("retrieval_profile"),
        retrieval_filter=retrieval_filter,
        security_context=security_context,
    )
    answer = get_task_result(final_session_id, "answer") or state.get("answer", "")
    return QueryResponse(
        message="处理完成！",
        session_id=final_session_id,
        answer=answer,
        done_list=get_done_task_list(final_session_id),
        image_urls=state.get("image_urls", []),
        answer_id=state.get("answer_id", ""),
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        citation_validation=state.get("citation_validation", {}),
        retrieval_profile={
            "profile_id": state.get("retrieval_profile_id", ""),
            "version": state.get("retrieval_profile_version", 0),
            "config_hash": state.get("retrieval_profile_hash", ""),
        },
        retrieval_diagnostics=state.get("retrieval_diagnostics", {}),
        query_plan=state.get("query_plan", {}),
        fusion_diagnostics=state.get("fusion_diagnostics", {}),
        context_pack_summary=state.get("context_pack_summary", {}),
        evidence_assessment=state.get("evidence_assessment", {}),
        answer_decision=state.get("answer_decision", {}),
        answer_envelope=state.get("answer_envelope") or None,
    )


def build_history_response(session_id: str, limit: int = 10) -> HistoryResponse:
    """
    查询并组装指定会话的历史记录。

    Args:
        session_id: 目标会话 ID。
        limit: 返回记录条数上限。

    Returns:
        HistoryResponse: 组装后的历史消息集合。
    """
    records = history_repository.list_recent(session_id, limit=limit)

    def envelope_for(record: dict) -> AnswerEnvelope | None:
        raw = record.get("answer_envelope")
        if raw:
            try:
                return AnswerEnvelope.model_validate(raw)
            except (TypeError, ValueError):
                logger.warning("历史 answer_envelope 无法通过 v1 校验，回退 text-only 兼容映射")
        if record.get("role") != "assistant":
            return None
        timestamp = record.get("ts")
        try:
            created_at = datetime.fromtimestamp(float(timestamp), UTC)
        except (TypeError, ValueError, OSError):
            created_at = None
        return legacy_envelope(
            session_id=str(record.get("session_id") or session_id),
            answer=str(record.get("text") or ""),
            answer_id=record.get("answer_id"),
            created_at=created_at,
            seed=str(record.get("_id") or timestamp or ""),
        )

    items = [
        HistoryItem(
            # mongo中的_id是一个Mongo对象，需要转成字符串
            _id=str(record.get("_id")) if record.get("_id") is not None else "",
            session_id=record.get("session_id", ""),
            role=record.get("role", ""),
            text=record.get("text", ""),
            rewritten_query=record.get("rewritten_query", ""),
            image_urls=record.get("image_urls") or [],
            answer_id=record.get("answer_id", ""),
            citations=record.get("citations") or [],
            sources=record.get("sources") or [],
            citation_validation=record.get("citation_validation") or {},
            retrieval_profile=record.get("retrieval_profile") or {},
            retrieval_diagnostics=record.get("retrieval_diagnostics") or {},
            answer_envelope=envelope_for(record),
            ts=record.get("ts"),
        )
        for record in records
    ]
    return HistoryResponse(session_id=session_id, items=items)


def clear_query_history(session_id: str) -> dict:
    """
    清空指定会话的历史记录。

    Args:
        session_id: 目标会话 ID。

    Returns:
        dict: 删除结果说明。
    """
    delete_count = history_repository.clear_session(session_id)
    return {
        "message": f"删除:{session_id}会话对应的聊天记录成功!!",
        "deleted_count": delete_count,
    }


@app.get("/")
def index():
    """
    返回查询服务首页导航信息。

    Returns:
        dict: 包含页面地址、查询接口、流式接口与历史接口导航信息。
    """
    return {
        "message": "Enterprise RAG Query Service",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "routes": {
            "html": "/html",
            "evaluation": "/eval/html",
            "evaluation_api": "/api/v1/evaluation/runs",
            "p2": "/p2/html",
            "p2_api": "/api/v1/p2/capabilities",
            "p3": "/p3/html",
            "p3_api": "/api/v1/p3/capabilities",
            "query_runs": "/api/v1/query-runs",
        },
    }


@app.get("/html")
def query_html():
    """
    返回查询演示页面。

    Returns:
        FileResponse: 本地聊天演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/p2/html")
def p2_html():
    """返回 P2 高级能力验收实验台。"""

    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "p2.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


def _authorized_evidence_for_request(request: Request, evidence_id: str) -> dict:
    """按 tenant 事实和当前 KB entitlement 双重闭合 Evidence。"""

    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise ResourceNotFoundError("evidence not found")
    evidence = get_authorized_evidence(evidence_id)
    source = evidence.get("source") or {}
    if source.get("source_type") == "web" and source.get("tenant_id") == principal.tenant_id:
        return evidence
    kb_id = str(source.get("kb_id") or "")
    if (
        source.get("source_type") != "local"
        or source.get("tenant_id") != principal.tenant_id
        or not kb_id
    ):
        raise ResourceNotFoundError("evidence not found")
    try:
        entitlement = SqlEntitlementResolver().resolve(
            principal=principal,
            requested=RequestedRetrievalScope(kb_ids=(kb_id,)),
        )
    except ScopeAuthorizationError as exc:
        raise ResourceNotFoundError("evidence not found") from exc
    if not any(grant.kb_id == kb_id for grant in entitlement.grants):
        raise ResourceNotFoundError("evidence not found")
    return evidence


@app.get("/api/v1/evidence/{evidence_id}")
def evidence_preview(request: Request, evidence_id: str):
    """返回授权 Evidence 的原始版本快照，不替换成新 active version。"""

    evidence = _authorized_evidence_for_request(request, evidence_id)
    images = evidence_image_projection(evidence)
    evidence["images"] = images
    evidence["image_urls"] = [item["url"] for item in images]
    if evidence.get("source", {}).get("source_type") == "local":
        evidence["source_download_url"] = f"/api/v1/evidence/{evidence_id}/source"
    return evidence


@app.get("/api/v1/evidence/{evidence_id}/source")
def evidence_source_download(request: Request, evidence_id: str):
    """下载本地原件；来源已清除或不可用时明确失败，绝不偷换版本。"""

    evidence = _authorized_evidence_for_request(request, evidence_id)
    source = evidence.get("source") or {}
    if source.get("source_type") != "local":
        raise ResourceNotFoundError("Web Evidence 没有可由本服务下载的本地原件")
    version = get_document_version(str(source["document_version_id"]))
    candidate = Path(str(version.get("object_key") or "")).resolve()
    project_root = PROJECT_ROOT.resolve()
    if not candidate.is_file() or (
        candidate != project_root and project_root not in candidate.parents
    ):
        raise ResourceNotFoundError("source object was purged or is not locally available")
    return FileResponse(
        path=candidate,
        media_type=version.get("mime_type") or guess_type(candidate.name)[0],
        filename=candidate.name,
    )


def _stream_minio_image(response):
    """分块转发 MinIO 响应，并在完成/中断时归还连接。"""

    try:
        yield from response.stream(amt=64 * 1024)
    finally:
        response.close()
        response.release_conn()


@app.get("/api/v1/evidence/{evidence_id}/images/{image_id}")
def evidence_image_preview(request: Request, evidence_id: str, image_id: str):
    """读取当前用户仍获授权、且属于固定 manifest 的 Evidence 图片。"""

    evidence = _authorized_evidence_for_request(request, evidence_id)
    asset = next(
        (item for item in evidence_image_assets(evidence) if item.image_id == image_id),
        None,
    )
    if asset is None or not image_asset_is_manifested(evidence, asset):
        raise ResourceNotFoundError("evidence image not found")
    try:
        client = minio_gateway.client()
        stat = client.stat_object(minio_gateway.bucket_name, asset.object_key)
        response = client.get_object(minio_gateway.bucket_name, asset.object_key)
    except S3Error as exc:
        raise ResourceNotFoundError("evidence image is no longer available") from exc
    headers = {
        "Cache-Control": "private, max-age=300",
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; sandbox",
    }
    size = getattr(stat, "size", None)
    if isinstance(size, int) and size >= 0:
        headers["Content-Length"] = str(size)
    etag = str(getattr(stat, "etag", "") or "").strip('"')
    if etag:
        headers["ETag"] = f'"{etag}"'
    return StreamingResponse(
        _stream_minio_image(response),
        media_type=asset.media_type,
        headers=headers,
    )


async def query(payload: QueryRequest, background_tasks: BackgroundTasks, request: Request):
    """仅保留给旧单元测试的非 HTTP 适配器；生产路由 QueryRun 承载。

    Args:
        request: 查询请求体，包含问题文本、是否流式输出和可选会话 ID。
        background_tasks: FastAPI 后台任务对象，流式模式下用于异步启动查询任务。

    Returns:
        QueryResponse: 非流式时返回完整答案；流式时返回会话 ID 与处理中提示。
    """
    if payload.is_stream:
        return start_stream_query(
            background_tasks=background_tasks,
            query=payload.query,
            session_id=payload.session_id,
            kb_id=payload.kb_id,
            principal=request.state.principal,
            retrieval_filter=payload.retrieval_filter,
        )
    # 查询图仍是同步执行契约，且其同步节点会在边界处运行短生命周期的异步
    # 适配器。ASGI 事件循环内直接调用会导致嵌套 asyncio.run；线程池同时由
    # AnyIO 复制当前 Context，确保安全 principal 等 ContextVar 不丢失。
    return await run_in_threadpool(
        execute_query,
        query=payload.query,
        session_id=payload.session_id,
        kb_id=payload.kb_id,
        principal=request.state.principal,
        retrieval_filter=payload.retrieval_filter,
    )


async def stream_query_result(session_id: str, request: Request):
    """仅保留给旧单元测试的非 HTTP SSE 适配器。

    Args:
        session_id: 目标会话 ID。
        request: 当前 HTTP 请求对象，用于感知客户端断开连接。

    Returns:
        StreamingResponse: 基于 SSE 协议的流式响应对象。
    """
    assert_query_session_owner(request.state.principal, session_id)
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def history(session_id: str, request: Request, limit: int = 10):
    """仅保留给旧单元测试的非 HTTP Mongo 历史适配器。

    Args:
        session_id: 目标会话 ID。
        limit: 返回记录条数上限。

    Returns:
        HistoryResponse: 当前会话最近的历史消息集合。
    """
    assert_query_session_owner(request.state.principal, session_id)
    return build_history_response(session_id=session_id, limit=limit)


def clear_history(session_id: str, request: Request):
    """仅保留给旧单元测试的非 HTTP Mongo 清理适配器。

    Args:
        session_id: 目标会话 ID。

    Returns:
        dict: 删除结果说明。
    """
    assert_query_session_owner(request.state.principal, session_id)
    return clear_query_history(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
