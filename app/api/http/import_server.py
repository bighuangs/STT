"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import hashlib
import asyncio
import json
import sys
import unicodedata
import uuid
from contextlib import asynccontextmanager
from mimetypes import guess_type
from pathlib import Path, PurePosixPath
from typing import List, Literal
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
# 兼容直接以 `python import_server.py` 方式启动，提前把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.import_ import ImportStatusResponse, ParsePreviewResponse, UploadResponse
from app.bootstrap.observability import observability_lifespan
from app.application.document_parsing_service import (
    parsing_capabilities,
    preview_document,
    resolved_parser_config_snapshot,
)
from app.domain.parsing import DocumentParseError
from app.api.schemas.control_plane import (
    DocumentUploadResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    IndexRebuildRequest,
    ResourceEnvelope,
)
from app.application.control_plane_service import (
    bootstrap_control_plane,
    create_ingestion_job,
    create_knowledge_base,
    get_document,
    get_ingestion_job,
    get_knowledge_base,
    list_documents,
    list_knowledge_bases,
    register_uploaded_document,
    update_knowledge_base,
)
from app.application.ingestion_job_service import (
    cancel_job,
    list_job_events,
    list_jobs,
    list_workers,
)
from app.application.index_lifecycle_service import (
    create_index_rebuild,
    delete_document_with_cleanup,
    delete_knowledge_base_with_cleanup,
    execute_cleanup,
    get_cleanup_job,
    get_index_lifecycle,
    list_cleanup_jobs,
    list_index_lifecycle,
    publish_index,
    rollback_index,
    scan_orphan_objects,
    set_document_enabled,
)
# 仅保留旧扩展/测试的导入符号；HTTP 请求不再直接调用执行器。
from app.application.import_runner import run_graph_task  # noqa: F401
from app.domain._state_machine import InvalidStateTransition
from app.infra.persistence.sql.repositories import (
    DEFAULT_KB_ID,
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config import settings
from app.api.http.auth_routes import register_auth_routes
from app.security.http import install_security
from app.security.service import bootstrap_security
from app.api.http.health_routes import register_health_routes
from app.api.http.knowledge_workbench_routes import register_knowledge_workbench_routes
from app.api.http.ui_routes import register_ui_routes
from app.infra.health import close_runtime_resources, validate_startup


@asynccontextmanager
async def lifespan(_: FastAPI):
    with observability_lifespan("import-api"):
        # 避免模块导入即创建默认数据库；服务启动时才迁移并初始化默认资源。
        bootstrap_control_plane()
        bootstrap_security()
        validate_startup("import")
        try:
            yield
        finally:
            close_runtime_resources()


app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.3.0",
    lifespan=lifespan,
)

# 配置 CORS（跨域资源共享）以允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

register_auth_routes(app)
register_health_routes(app, "import")
register_ui_routes(app, "import")
register_knowledge_workbench_routes(app)
install_security(app)

ORIGINALS_ROOT = PROJECT_ROOT / "output" / "originals"

_WINDOWS_FORBIDDEN_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAME_STEMS = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}


def _safe_upload_filename(raw_name: str | None, *, logical_name: bool = False) -> str:
    """保留 Unicode 文件名，同时把文档身份与不可信传输路径安全解耦。"""

    if raw_name is None:
        if logical_name:
            raise HTTPException(status_code=422, detail="invalid upload logical_name")
        raw_name = "upload.bin"
    normalized = unicodedata.normalize("NFC", raw_name)
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in normalized):
        raise HTTPException(status_code=422, detail="invalid upload filename")
    normalized_path = normalized.replace("\\", "/")
    if logical_name and "/" in normalized_path:
        raise HTTPException(status_code=422, detail="invalid upload logical_name")
    filename = PurePosixPath(normalized_path).name.strip().rstrip(". ")
    if (
        filename in {"", ".", ".."}
        or len(filename) > (512 if logical_name else 240)
        or any(character in _WINDOWS_FORBIDDEN_FILENAME_CHARS for character in filename)
        or filename.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_FILENAME_STEMS
    ):
        detail = "invalid upload logical_name" if logical_name else "invalid upload filename"
        raise HTTPException(status_code=422, detail=detail)
    return filename


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ResourceNotFoundError)
async def handle_not_found(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"request_id": request.state.request_id, "error": {"code": "NOT_FOUND", "message": str(exc)}},
    )


@app.exception_handler(ResourceConflictError)
@app.exception_handler(InvalidStateTransition)
async def handle_conflict(request: Request, exc: Exception):
    return JSONResponse(
        status_code=409,
        content={"request_id": request.state.request_id, "error": {"code": "STATE_CONFLICT", "message": str(exc)}},
    )


@app.exception_handler(DocumentParseError)
async def handle_document_parse_error(request: Request, exc: DocumentParseError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request.state.request_id,
            "error": {
                "code": exc.code,
                "message": exc.safe_message,
                "retryable": exc.retryable,
            },
        },
    )

@app.get("/html")
def import_html():
    """
    返回导入演示页面。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "process" / "import_" / "page" / "import.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/api/v1/parsing/capabilities", response_model=ResourceEnvelope)
async def parsing_capabilities_api(request: Request):
    return _envelope(request, parsing_capabilities())


@app.post("/api/v1/parsing/preview", response_model=ParsePreviewResponse)
async def parsing_preview_api(
    request: Request,
    file: UploadFile = File(...),
    parser_config: str = Form("{}"),
    max_blocks: int = Form(20),
):
    config = _parse_parser_config(parser_config)
    path, _, _, _ = await _save_upload(file, "__preview__")
    try:
        result = preview_document(
            path,
            declared_mime=file.content_type,
            parser_config=config,
            max_blocks=max_blocks,
        )
        return ParsePreviewResponse(request_id=request.state.request_id, **result)
    finally:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass

async def _save_upload(file: UploadFile, kb_id: str) -> tuple[Path, str, int, str]:
    safe_name = _safe_upload_filename(file.filename)
    upload_dir = ORIGINALS_ROOT / kb_id / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=False)
    target = upload_dir / safe_name
    digest = hashlib.sha256()
    size_bytes = 0
    with target.open("wb") as output:
        while data := await file.read(1024 * 1024):
            size_bytes += len(data)
            if size_bytes > settings.max_upload_bytes:
                output.close()
                target.unlink(missing_ok=True)
                upload_dir.rmdir()
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds limit: {settings.max_upload_bytes} bytes",
                )
            digest.update(data)
            output.write(data)
    return target, digest.hexdigest(), size_bytes, safe_name


async def _register_upload(
    file: UploadFile,
    kb_id: str,
    parser_config: dict | None = None,
    document_id: str | None = None,
    idempotency_key: str | None = None,
    logical_name: str | None = None,
) -> dict:
    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()
        if not idempotency_key or len(idempotency_key) > 255:
            raise HTTPException(
                status_code=422,
                detail="Idempotency-Key must contain 1..255 characters",
            )
    if document_id is not None:
        # 版本上传以持久化文档身份为准；临时文件名和客户端字段都不能重命名文档。
        requested_logical_name = get_document(document_id)["logical_name"]
    else:
        requested_logical_name = (
            _safe_upload_filename(logical_name, logical_name=True)
            if logical_name is not None
            else None
        )
    declared_mime = file.content_type or guess_type(file.filename or "")[0]
    path, digest, size_bytes, safe_name = await _save_upload(file, kb_id)
    requested_logical_name = requested_logical_name or safe_name
    try:
        resolved_config, parser_profile = resolved_parser_config_snapshot(
            path,
            declared_mime=declared_mime,
            parser_config=parser_config or {},
        )
        registered = register_uploaded_document(
            kb_id=kb_id,
            logical_name=requested_logical_name,
            content_sha256=digest,
            size_bytes=size_bytes,
            mime_type=parser_profile.detected_mime_type,
            object_key=str(path),
            parser_config_snapshot=resolved_config,
            document_id=document_id,
            idempotency_key=idempotency_key,
        )
    except Exception:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        raise
    if registered["deduplicated"] and registered["document_version"]["object_key"] != str(path):
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
    registered["parser_profile"] = parser_profile.model_dump(mode="json")
    return registered


def _upload_response_profile(registered: dict) -> dict:
    snapshot = registered["document_version"].get("parser_config_snapshot") or {}
    profile = registered.get("parser_profile") or {}
    return {
        "detected_format": profile.get("detected_format") or snapshot.get("detected_format", ""),
        "detected_mime_type": profile.get("detected_mime_type") or snapshot.get("detected_mime_type", "application/octet-stream"),
        "parser_id": profile.get("parser_id") or snapshot.get("parser_id", ""),
        "parser_version": profile.get("parser_version") or snapshot.get("parser_version", ""),
        "chunking_template_id": profile.get("template_id") or snapshot.get("chunking_template", "general"),
        "chunking_template_version": profile.get("template_version") or snapshot.get("chunking_template_version", ""),
        "suggested_chunking_template_id": profile.get("suggested_template_id") or snapshot.get("suggested_chunking_template_id", "general"),
        "parse_warnings": profile.get("warnings") or snapshot.get("parse_profile_warnings", []),
    }


def _envelope(request: Request, data) -> ResourceEnvelope:
    return ResourceEnvelope(request_id=request.state.request_id, data=data)


@app.post("/api/v1/knowledge-bases", response_model=ResourceEnvelope, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base_api(payload: KnowledgeBaseCreate, request: Request):
    return _envelope(request, create_knowledge_base(**payload.model_dump()))


@app.get("/api/v1/knowledge-bases", response_model=ResourceEnvelope)
async def list_knowledge_bases_api(request: Request):
    return _envelope(request, list_knowledge_bases())


@app.get("/api/v1/knowledge-bases/{kb_id}", response_model=ResourceEnvelope)
async def get_knowledge_base_api(kb_id: str, request: Request):
    return _envelope(request, get_knowledge_base(kb_id))


@app.patch("/api/v1/knowledge-bases/{kb_id}", response_model=ResourceEnvelope)
async def update_knowledge_base_api(kb_id: str, payload: KnowledgeBaseUpdate, request: Request):
    return _envelope(
        request,
        update_knowledge_base(kb_id, **payload.model_dump(exclude_unset=True)),
    )


@app.delete("/api/v1/knowledge-bases/{kb_id}", response_model=ResourceEnvelope)
async def delete_knowledge_base_api(
    kb_id: str,
    request: Request,
    mode: Literal["soft", "purge"] = "soft",
):
    return _envelope(request, delete_knowledge_base_with_cleanup(kb_id, mode))


@app.get("/api/v1/knowledge-bases/{kb_id}/index-versions", response_model=ResourceEnvelope)
async def list_index_versions_api(kb_id: str, request: Request):
    return _envelope(request, list_index_lifecycle(kb_id))


@app.post(
    "/api/v1/knowledge-bases/{kb_id}/index-versions",
    response_model=ResourceEnvelope,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_index_rebuild_api(
    kb_id: str, payload: IndexRebuildRequest, request: Request
):
    return _envelope(
        request,
        create_index_rebuild(kb_id, **payload.model_dump(exclude_none=True)),
    )


@app.get("/api/v1/index-versions/{index_version_id}", response_model=ResourceEnvelope)
async def get_index_version_api(index_version_id: str, request: Request):
    return _envelope(request, get_index_lifecycle(index_version_id))


@app.post("/api/v1/index-versions/{index_version_id}:publish", response_model=ResourceEnvelope)
async def publish_index_version_api(index_version_id: str, request: Request):
    return _envelope(request, publish_index(index_version_id))


@app.post("/api/v1/knowledge-bases/{kb_id}:rollback-index", response_model=ResourceEnvelope)
async def rollback_index_api(kb_id: str, request: Request):
    return _envelope(request, rollback_index(kb_id))


@app.get("/api/v1/knowledge-bases/{kb_id}/documents", response_model=ResourceEnvelope)
async def list_documents_api(kb_id: str, request: Request):
    return _envelope(request, list_documents(kb_id))


def _parse_parser_config(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"parser_config must be valid JSON: {exc.msg}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="parser_config must be a JSON object")
    return parsed


@app.post(
    "/api/v1/knowledge-bases/{kb_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_api(
    kb_id: str,
    request: Request,
    file: UploadFile = File(...),
    parser_config: str = Form("{}"),
    logical_name: str | None = Form(None),
):
    registered = await _register_upload(
        file,
        kb_id,
        _parse_parser_config(parser_config),
        idempotency_key=request.headers.get("Idempotency-Key"),
        logical_name=logical_name,
    )
    document = registered["document"]
    version = registered["document_version"]
    job = registered["ingestion_job"]
    index = registered["index_version"]
    return DocumentUploadResponse(
        request_id=request.state.request_id,
        document_id=document["id"],
        document_version_id=version["id"],
        ingestion_job_id=job["id"],
        index_version_id=index["id"],
        version_no=version["version_no"],
        deduplicated=registered["deduplicated"],
        reused_job=registered["reused_job"],
        idempotency_replayed=registered.get("idempotency_replayed", False),
        status=job["status"],
        **_upload_response_profile(registered),
    )


@app.post(
    "/api/v1/documents/{document_id}/versions",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_version_api(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
    parser_config: str = Form("{}"),
):
    document = get_document(document_id)
    registered = await _register_upload(
        file,
        document["kb_id"],
        _parse_parser_config(parser_config),
        document_id=document_id,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )
    version = registered["document_version"]
    job = registered["ingestion_job"]
    index = registered["index_version"]
    return DocumentUploadResponse(
        request_id=request.state.request_id,
        document_id=document_id,
        document_version_id=version["id"],
        ingestion_job_id=job["id"],
        index_version_id=index["id"],
        version_no=version["version_no"],
        deduplicated=registered["deduplicated"],
        reused_job=registered["reused_job"],
        idempotency_replayed=registered.get("idempotency_replayed", False),
        status=job["status"],
        **_upload_response_profile(registered),
    )


@app.get("/api/v1/documents/{document_id}", response_model=ResourceEnvelope)
async def get_document_api(document_id: str, request: Request):
    return _envelope(request, get_document(document_id))


@app.post("/api/v1/documents/{document_id}:disable", response_model=ResourceEnvelope)
async def disable_document_api(document_id: str, request: Request):
    return _envelope(request, set_document_enabled(document_id, False))


@app.post("/api/v1/documents/{document_id}:enable", response_model=ResourceEnvelope)
async def enable_document_api(document_id: str, request: Request):
    return _envelope(request, set_document_enabled(document_id, True))


@app.delete("/api/v1/documents/{document_id}", response_model=ResourceEnvelope)
async def delete_document_api(
    document_id: str,
    request: Request,
    mode: Literal["soft", "purge"] = "soft",
):
    return _envelope(request, delete_document_with_cleanup(document_id, mode))


@app.get("/api/v1/cleanup-jobs", response_model=ResourceEnvelope)
async def list_cleanup_jobs_api(
    request: Request, limit: int = 100, kb_id: str | None = None
):
    return _envelope(request, list_cleanup_jobs(limit, kb_id=kb_id))


@app.get("/api/v1/cleanup-jobs/{cleanup_job_id}", response_model=ResourceEnvelope)
async def get_cleanup_job_api(cleanup_job_id: str, request: Request):
    return _envelope(request, get_cleanup_job(cleanup_job_id))


@app.post("/api/v1/cleanup-jobs/{cleanup_job_id}:run", response_model=ResourceEnvelope)
async def run_cleanup_job_api(cleanup_job_id: str, request: Request):
    return _envelope(request, execute_cleanup(cleanup_job_id))


@app.get("/api/v1/cleanup/orphans", response_model=ResourceEnvelope)
async def scan_orphan_objects_api(request: Request, retention_seconds: int = 86400):
    return _envelope(request, scan_orphan_objects(retention_seconds))


@app.post(
    "/api/v1/document-versions/{version_id}/ingestion-jobs",
    response_model=ResourceEnvelope,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job_api(
    version_id: str, request: Request
):
    job = create_ingestion_job(version_id)
    return _envelope(request, job)


@app.get("/api/v1/ingestion-jobs/{job_id}", response_model=ResourceEnvelope)
async def get_ingestion_job_api(job_id: str, request: Request):
    return _envelope(request, get_ingestion_job(job_id))


@app.get("/api/v1/ingestion-jobs", response_model=ResourceEnvelope)
async def list_ingestion_jobs_api(
    request: Request,
    kb_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
):
    return _envelope(request, list_jobs(kb_id=kb_id, status=status, limit=limit))


@app.post("/api/v1/ingestion-jobs/{job_id}/cancel", response_model=ResourceEnvelope)
async def cancel_ingestion_job_api(job_id: str, request: Request):
    return _envelope(request, cancel_job(job_id))


@app.post(
    "/api/v1/ingestion-jobs/{job_id}/retry",
    response_model=ResourceEnvelope,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ingestion_job_api(job_id: str, request: Request):
    """为失败/取消的版本创建一次可审计的新导入尝试。"""
    previous = get_ingestion_job(job_id)
    if previous["status"] not in {"failed", "cancelled"}:
        raise ResourceConflictError(
            f"only failed or cancelled jobs can be retried: {job_id} is {previous['status']}"
        )
    return _envelope(request, create_ingestion_job(previous["document_version_id"]))


@app.get("/api/v1/ingestion-jobs/{job_id}/events", response_model=ResourceEnvelope)
async def list_ingestion_job_events_api(
    job_id: str,
    request: Request,
    after_sequence: int = 0,
    limit: int = 500,
):
    return _envelope(request, list_job_events(job_id, after_sequence, limit))


@app.get("/api/v1/ingestion-jobs/{job_id}/events/stream")
async def stream_ingestion_job_events_api(job_id: str, request: Request):
    header = request.headers.get("Last-Event-ID", "0")
    try:
        cursor = max(0, int(header))
    except ValueError:
        cursor = 0

    async def generate():
        nonlocal cursor
        while not await request.is_disconnected():
            events = list_job_events(job_id, cursor, 200)
            for event in events:
                cursor = event["sequence"]
                yield (
                    f"id: {cursor}\n"
                    f"event: {event['event_type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
            job = get_ingestion_job(job_id)
            if job["status"] in {"succeeded", "completed", "failed", "cancelled"} and cursor >= job["last_event_sequence"]:
                return
            if not events:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/ingestion-workers", response_model=ResourceEnvelope)
async def list_ingestion_workers_api(request: Request):
    return _envelope(request, list_workers())




# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(
    files: List[UploadFile] = File(...),
    kb_id: str = Form(DEFAULT_KB_ID),
    parser_config: str = Form("{}"),
):
    """
    文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    task_ids: list[str] = []
    document_ids: list[str] = []
    version_ids: list[str] = []
    index_ids: list[str] = []
    parser_profiles: list[dict] = []
    config = _parse_parser_config(parser_config)
    for file in files:
        registered = await _register_upload(file, kb_id, config)
        job_id = registered["ingestion_job"]["id"]
        task_ids.append(job_id)
        document_ids.append(registered["document"]["id"])
        version_ids.append(registered["document_version"]["id"])
        index_ids.append(registered["index_version"]["id"])
        parser_profiles.append(_upload_response_profile(registered))

    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return UploadResponse(
        code=200,
        message=f"Files uploaded successfully, total: {len(files)}",
        task_ids=task_ids,
        document_ids=document_ids,
        document_version_ids=version_ids,
        index_version_ids=index_ids,
        parser_profiles=parser_profiles,
    )


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
# 2. 改造接口
@app.get("/status/{task_id}",
         summary="任务状态查询",
         description="根据TaskID查询单个文件的处理进度和全局状态",
         response_model=ImportStatusResponse)  # 绑定模型
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: ImportStatusResponse 格式响应
    """
    persisted = get_ingestion_job(task_id)
    events = list_job_events(task_id)
    done_list = [
        event["stage"] for event in events if event["event_type"] == "stage_completed"
    ]
    running_list = (
        [persisted["stage"]]
        if persisted["status"] in {"leased", "running", "processing"}
        else []
    )
    logger.info(
        f"[{task_id}] 持久任务状态查询：{persisted['status']} / {persisted['stage']}"
    )

    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=persisted["status"],
        done_list=done_list,
        running_list=running_list,
        document_id=persisted.get("document_id"),
        document_version_id=persisted.get("document_version_id"),
        index_version_id=persisted.get("index_version_id"),
        stage=persisted.get("stage"),
        progress=persisted.get("progress"),
        error_code=persisted.get("error_code"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
