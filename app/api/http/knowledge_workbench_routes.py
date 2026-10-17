"""HTTP boundary for the TODO-12 knowledge/document/chunk workbench.

The boundary intentionally returns public projections from the application service;
database rows and storage locators never cross it.  All mutable resources use
``If-Match`` and the source endpoint implements bounded single-range streaming.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
from urllib.parse import quote

from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas.knowledge_workbench import (
    BlockView,
    ChunkDraftCreateRequest,
    ChunkDraftPublishRequest,
    ChunkDraftView,
    ChunkMergeRequest,
    ChunkReplaceRequest,
    ChunkSplitRequest,
    ChunkView,
    DocumentBatchResultView,
    DocumentBatchActionRequest,
    DocumentDetailView,
    DocumentSummaryView,
    DocumentVersionView,
    JobDetailView,
    JobListItemView,
    KnowledgeBaseView,
    KnowledgeBaseWorkbenchUpdate,
    VersionDiffView,
    WorkbenchEnvelope,
    WorkbenchPage,
)
from app.application.knowledge_workbench_service import (
    WorkbenchCursorError,
    WorkbenchPreconditionError,
    batch_document_action,
    create_chunk_draft,
    delete_chunk,
    get_chunk_draft,
    get_document,
    get_job_detail,
    get_knowledge_base,
    get_version,
    list_blocks,
    list_chunks,
    list_chunk_draft_chunks,
    list_documents,
    list_jobs,
    list_knowledge_bases,
    list_versions,
    merge_chunks,
    publish_chunk_draft,
    replace_chunk,
    source_file,
    split_chunk,
    update_knowledge_base,
    version_diff,
)
from app.security.principal import Principal


_SINGLE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        # The authentication middleware should make this unreachable.  Keeping the
        # boundary fail-closed also protects direct route calls in tests/extensions.
        raise WorkbenchPreconditionError(
            "authenticated principal is required", required=True
        )
    return principal


def _envelope(request: Request, data: object) -> dict:
    return {"request_id": request.state.request_id, "data": data}


def _page(value: dict) -> dict:
    if "schema_version" not in value:
        return {"schema_version": "workbench-page.v1", **value}
    return value


def _set_etag(response: Response, value: dict) -> None:
    etag = value.get("etag")
    if etag:
        response.headers["ETag"] = str(etag)


def _read_slice(path: Path, start: int, length: int) -> Iterator[bytes]:
    remaining = length
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining:
            payload = stream.read(min(64 * 1024, remaining))
            if not payload:
                break
            remaining -= len(payload)
            yield payload


def _range_bounds(value: str, size: int) -> tuple[int, int]:
    if "," in value:
        raise ValueError("multiple byte ranges are not supported")
    match = _SINGLE_RANGE.fullmatch(value.strip())
    if match is None or (not match.group(1) and not match.group(2)):
        raise ValueError("invalid byte range")
    start_text, end_text = match.groups()
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0 or size <= 0:
            raise ValueError("unsatisfiable byte range")
        start = max(0, size - suffix)
        return start, size - 1
    start = int(start_text)
    if start >= size:
        raise ValueError("unsatisfiable byte range")
    end = int(end_text) if end_text else size - 1
    if end < start:
        raise ValueError("invalid byte range")
    return start, min(end, size - 1)


def register_knowledge_workbench_routes(app: FastAPI) -> None:
    @app.exception_handler(WorkbenchCursorError)
    async def workbench_cursor_error(request: Request, exc: WorkbenchCursorError):
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request.state.request_id,
                "error": {"code": "INVALID_CURSOR", "message": str(exc)},
            },
        )

    @app.exception_handler(WorkbenchPreconditionError)
    async def workbench_precondition_error(
        request: Request, exc: WorkbenchPreconditionError
    ):
        code = "PRECONDITION_REQUIRED" if exc.required else "ETAG_MISMATCH"
        headers = {"ETag": exc.current_etag} if exc.current_etag else None
        return JSONResponse(
            status_code=428 if exc.required else 412,
            content={
                "request_id": request.state.request_id,
                "error": {
                    "code": code,
                    "message": str(exc),
                    "current_etag": exc.current_etag,
                },
            },
            headers=headers,
        )

    @app.get(
        "/api/v1/workbench/knowledge-bases",
        response_model=WorkbenchEnvelope[WorkbenchPage[KnowledgeBaseView]],
        tags=["知识工作台"],
    )
    def knowledge_base_page(
        request: Request,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        q: str | None = Query(default=None, max_length=1000),
        status: str | None = Query(default=None, max_length=32),
        sort: str = Query(default="updated_desc", max_length=32),
    ):
        data = list_knowledge_bases(
            _principal(request),
            cursor=cursor,
            limit=limit,
            q=q,
            status=status,
            sort=sort,
        )
        return _envelope(request, _page(data))

    @app.get(
        "/api/v1/workbench/knowledge-bases/{kb_id}",
        response_model=WorkbenchEnvelope[KnowledgeBaseView],
        tags=["知识工作台"],
    )
    def knowledge_base_detail(kb_id: str, request: Request, response: Response):
        data = get_knowledge_base(_principal(request), kb_id)
        _set_etag(response, data)
        return _envelope(request, data)

    @app.patch(
        "/api/v1/workbench/knowledge-bases/{kb_id}",
        response_model=WorkbenchEnvelope[KnowledgeBaseView],
        tags=["知识工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def knowledge_base_update(
        kb_id: str,
        payload: KnowledgeBaseWorkbenchUpdate,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = update_knowledge_base(
            _principal(request),
            kb_id,
            payload.model_dump(exclude_unset=True),
            if_match,
        )
        _set_etag(response, data)
        return _envelope(request, data)

    @app.get(
        "/api/v1/workbench/knowledge-bases/{kb_id}/documents",
        response_model=WorkbenchEnvelope[WorkbenchPage[DocumentSummaryView]],
        tags=["文档工作台"],
    )
    def document_page(
        kb_id: str,
        request: Request,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        q: str | None = Query(default=None, max_length=1000),
        status: str | None = Query(default=None, max_length=32),
        mime_type: str | None = Query(default=None, max_length=255),
        sort: str = Query(default="updated_desc", max_length=32),
    ):
        data = list_documents(
            _principal(request),
            kb_id,
            cursor=cursor,
            limit=limit,
            q=q,
            status=status,
            mime_type=mime_type,
            sort=sort,
        )
        return _envelope(request, _page(data))

    @app.post(
        "/api/v1/workbench/documents:batch-action",
        response_model=WorkbenchEnvelope[DocumentBatchResultView],
        tags=["文档工作台"],
    )
    def document_batch(payload: DocumentBatchActionRequest, request: Request):
        data = batch_document_action(
            _principal(request),
            payload.action,
            [item.model_dump() for item in payload.items],
        )
        return _envelope(request, data)

    @app.get(
        "/api/v1/workbench/documents/{document_id}",
        response_model=WorkbenchEnvelope[DocumentDetailView],
        tags=["文档工作台"],
    )
    def document_detail(document_id: str, request: Request, response: Response):
        data = get_document(_principal(request), document_id)
        _set_etag(response, data)
        return _envelope(request, data)

    @app.get(
        "/api/v1/workbench/documents/{document_id}/versions",
        response_model=WorkbenchEnvelope[WorkbenchPage[DocumentVersionView]],
        tags=["文档工作台"],
    )
    def document_versions(document_id: str, request: Request):
        return _envelope(request, _page(list_versions(_principal(request), document_id)))

    @app.get(
        "/api/v1/workbench/document-versions/{version_id}",
        response_model=WorkbenchEnvelope[DocumentVersionView],
        tags=["文档工作台"],
    )
    def document_version(version_id: str, request: Request):
        return _envelope(request, get_version(_principal(request), version_id))

    @app.get(
        "/api/v1/workbench/document-versions/{version_id}/diff",
        response_model=WorkbenchEnvelope[VersionDiffView],
        tags=["文档工作台"],
    )
    def document_version_diff(
        version_id: str,
        request: Request,
        other_version_id: str = Query(min_length=1, max_length=64),
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return _envelope(
            request,
            version_diff(
                _principal(request),
                version_id,
                other_version_id,
                cursor=cursor,
                limit=limit,
            ),
        )

    @app.get(
        "/api/v1/workbench/document-versions/{version_id}/source",
        tags=["文档工作台"],
    )
    def document_source(version_id: str, request: Request):
        path, metadata = source_file(_principal(request), version_id)
        size = path.stat().st_size
        # Use the verified local size as the transport truth.  A metadata mismatch
        # cannot make the range generator read beyond the object.
        etag = str(metadata["etag"])
        range_header = request.headers.get("Range")
        if_range = request.headers.get("If-Range")
        if range_header and if_range and if_range.strip() != etag:
            range_header = None
        start, end, response_status = 0, size - 1, 200
        if range_header:
            try:
                start, end = _range_bounds(range_header, size)
            except (TypeError, ValueError):
                return JSONResponse(
                    status_code=416,
                    content={
                        "request_id": request.state.request_id,
                        "error": {
                            "code": "RANGE_NOT_SATISFIABLE",
                            "message": "requested byte range cannot be served",
                        },
                    },
                    headers={"Content-Range": f"bytes */{size}", "ETag": etag},
                )
            response_status = 206
        length = max(0, end - start + 1)
        filename = str(metadata["filename"]).replace("\r", "").replace("\n", "")
        fallback = "document" + path.suffix.lower()
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "ETag": etag,
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "Content-Security-Policy": (
                "default-src 'none'; frame-ancestors 'self'; sandbox"
            ),
            "Content-Disposition": (
                f"inline; filename={fallback}; filename*=UTF-8''{quote(filename)}"
            ),
        }
        if response_status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(
            _read_slice(path, start, length),
            status_code=response_status,
            media_type=str(metadata["media_type"]),
            headers=headers,
        )

    @app.get(
        "/api/v1/workbench/document-versions/{version_id}/blocks",
        response_model=WorkbenchEnvelope[WorkbenchPage[BlockView]],
        tags=["切片工作台"],
    )
    def block_page(
        version_id: str,
        request: Request,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        q: str | None = Query(default=None, max_length=1000),
        block_type: str | None = Query(default=None, max_length=64),
        page_no: int | None = Query(default=None, ge=1),
    ):
        return _envelope(
            request,
            list_blocks(
                _principal(request),
                version_id,
                cursor=cursor,
                limit=limit,
                q=q,
                block_type=block_type,
                page_no=page_no,
            ),
        )

    @app.get(
        "/api/v1/workbench/document-versions/{version_id}/chunks",
        response_model=WorkbenchEnvelope[WorkbenchPage[ChunkView]],
        tags=["切片工作台"],
    )
    def chunk_page(
        version_id: str,
        request: Request,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        q: str | None = Query(default=None, max_length=1000),
        block_type: str | None = Query(default=None, max_length=64),
        page_no: int | None = Query(default=None, ge=1),
    ):
        return _envelope(
            request,
            list_chunks(
                _principal(request),
                version_id,
                cursor=cursor,
                limit=limit,
                q=q,
                block_type=block_type,
                page_no=page_no,
            ),
        )

    @app.get(
        "/api/v1/workbench/jobs",
        response_model=WorkbenchEnvelope[WorkbenchPage[JobListItemView]],
        tags=["任务工作台"],
    )
    def job_page(
        request: Request,
        kb_id: str | None = Query(default=None, max_length=64),
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        status: str | None = Query(default=None, max_length=32),
    ):
        return _envelope(
            request,
            _page(
                list_jobs(
                    _principal(request),
                    kb_id=kb_id,
                    cursor=cursor,
                    limit=limit,
                    status=status,
                )
            ),
        )

    @app.get(
        "/api/v1/workbench/jobs/{job_id}",
        response_model=WorkbenchEnvelope[JobDetailView],
        tags=["任务工作台"],
    )
    def job_detail(job_id: str, request: Request):
        return _envelope(request, get_job_detail(_principal(request), job_id))

    @app.post(
        "/api/v1/workbench/document-versions/{version_id}/chunk-drafts",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        status_code=201,
        tags=["切片工作台"],
    )
    def chunk_draft_create(
        version_id: str,
        payload: ChunkDraftCreateRequest,
        request: Request,
        response: Response,
    ):
        data = create_chunk_draft(
            _principal(request), version_id, payload.base_chunking_run_id
        )
        _set_etag(response, data)
        response.headers["Location"] = f"/api/v1/workbench/chunk-drafts/{data['id']}"
        return _envelope(request, data)

    @app.get(
        "/api/v1/workbench/chunk-drafts/{draft_id}",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        tags=["切片工作台"],
    )
    def chunk_draft_detail(draft_id: str, request: Request, response: Response):
        data = get_chunk_draft(_principal(request), draft_id)
        _set_etag(response, data)
        return _envelope(request, data)

    @app.get(
        "/api/v1/workbench/chunk-drafts/{draft_id}/chunks",
        response_model=WorkbenchEnvelope[WorkbenchPage[ChunkView]],
        tags=["切片工作台"],
    )
    def chunk_draft_page(
        draft_id: str,
        request: Request,
        cursor: str | None = Query(default=None, max_length=4096),
        limit: int = Query(default=50, ge=1, le=200),
        q: str | None = Query(default=None, max_length=1000),
        block_type: str | None = Query(default=None, max_length=64),
        page_no: int | None = Query(default=None, ge=1),
    ):
        return _envelope(
            request,
            list_chunk_draft_chunks(
                _principal(request),
                draft_id,
                cursor=cursor,
                limit=limit,
                q=q,
                block_type=block_type,
                page_no=page_no,
            ),
        )

    @app.patch(
        "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        tags=["切片工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def chunk_replace(
        draft_id: str,
        chunk_id: str,
        payload: ChunkReplaceRequest,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = replace_chunk(
            _principal(request), draft_id, chunk_id, payload.chunk, if_match
        )
        _set_etag(response, data)
        return _envelope(request, data)

    @app.post(
        "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}:split",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        tags=["切片工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def chunk_split(
        draft_id: str,
        chunk_id: str,
        payload: ChunkSplitRequest,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = split_chunk(
            _principal(request), draft_id, chunk_id, payload.chunks, if_match
        )
        _set_etag(response, data)
        return _envelope(request, data)

    @app.post(
        "/api/v1/workbench/chunk-drafts/{draft_id}/chunks:merge",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        tags=["切片工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def chunk_merge(
        draft_id: str,
        payload: ChunkMergeRequest,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = merge_chunks(
            _principal(request),
            draft_id,
            payload.chunk_ids,
            payload.chunk,
            if_match,
        )
        _set_etag(response, data)
        return _envelope(request, data)

    @app.delete(
        "/api/v1/workbench/chunk-drafts/{draft_id}/chunks/{chunk_id}",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        tags=["切片工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def chunk_delete(
        draft_id: str,
        chunk_id: str,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = delete_chunk(_principal(request), draft_id, chunk_id, if_match)
        _set_etag(response, data)
        return _envelope(request, data)

    @app.post(
        "/api/v1/workbench/chunk-drafts/{draft_id}:publish",
        response_model=WorkbenchEnvelope[ChunkDraftView],
        status_code=202,
        tags=["切片工作台"],
        openapi_extra={"x-stt-required-headers": ["If-Match"]},
    )
    def chunk_draft_publish(
        draft_id: str,
        payload: ChunkDraftPublishRequest,
        request: Request,
        response: Response,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        data = publish_chunk_draft(
            _principal(request), draft_id, if_match=if_match, note=payload.note
        )
        _set_etag(response, data)
        if data.get("published_job_id"):
            response.headers["Location"] = (
                f"/api/v1/workbench/jobs/{data['published_job_id']}"
            )
        return _envelope(request, data)


__all__ = ["register_knowledge_workbench_routes"]
