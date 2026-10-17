"""TODO-09 检索配置版本、发布、回滚与真实引擎预览 API。"""

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas.control_plane import ResourceEnvelope
from app.api.schemas.retrieval import (
    RetrievalProfileDraftCreate,
    RetrievalProfilePreviewRequest,
    RetrievalProfilePublishRequest,
    RetrievalProfileRollbackRequest,
)
from app.application.retrieval_service import preview_retrieval_profile
from app.domain.retrieval import RetrievalEngineError
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository


def _envelope(request: Request, data: object) -> ResourceEnvelope:
    return ResourceEnvelope(request_id=request.state.request_id, data=data)


def _repository(request: Request, session) -> RetrievalProfileRepository:
    principal = request.state.principal
    return RetrievalProfileRepository(
        session,
        tenant_id=principal.tenant_id,
        actor_id=principal.user_id,
    )


def _index_not_ready_response(request: Request) -> JSONResponse:
    """把未发布索引收敛为稳定的业务冲突，不暴露检索拓扑细节。"""

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "request_id": request.state.request_id,
            "error": {
                "code": "INDEX_NOT_READY",
                "message": (
                    "当前知识库还没有可查询的已发布索引。请等待文档处理完成，确认索引已发布后再试。"
                ),
                "retryable": False,
            },
        },
    )


def register_retrieval_routes(app: FastAPI) -> None:
    @app.get(
        "/api/v1/knowledge-bases/{kb_id}/retrieval-profiles",
        response_model=ResourceEnvelope,
        tags=["检索配置"],
    )
    def list_profiles(
        kb_id: str,
        request: Request,
        profile_status: str | None = Query(default=None, alias="status"),
    ):
        with session_scope() as session:
            repository = _repository(request, session)
            # A GET must never repair or publish control-plane state.  Missing
            # pointers are surfaced to the caller instead of activating the
            # highest draft as the legacy compatibility path does.
            current = repository.resolve_for_kb(kb_id, allow_default=False)
            rows = repository.list(kb_id=kb_id, status=profile_status)
            return _envelope(
                request,
                {
                    "control_plane_revision": repository.control_plane_revision(kb_id),
                    "current": repository.profile_dict(current),
                    "items": [repository.profile_dict(row) for row in rows],
                },
            )

    @app.post(
        "/api/v1/knowledge-bases/{kb_id}/retrieval-profiles",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["检索配置"],
    )
    def create_profile(
        kb_id: str,
        payload: RetrievalProfileDraftCreate,
        request: Request,
    ):
        with session_scope() as session:
            repository = _repository(request, session)
            row = repository.create_draft(
                kb_id,
                payload.name,
                payload.config,
                created_by=request.state.principal.user_id,
                description=payload.description,
                profile_id=payload.profile_id,
                expected_current_profile_id=payload.expected_current_profile_id,
                expected_current_version=payload.expected_current_version,
                expected_current_config_hash=payload.expected_current_config_hash,
                expected_control_plane_revision=payload.expected_control_plane_revision,
                expected_max_version=payload.expected_max_version,
            )
            return _envelope(request, repository.profile_dict(row))

    @app.get(
        "/api/v1/retrieval-profiles/{profile_id}/versions/{version}",
        response_model=ResourceEnvelope,
        tags=["检索配置"],
    )
    def get_profile(
        profile_id: str,
        version: int,
        request: Request,
        kb_id: str = Query(min_length=1, max_length=64),
    ):
        with session_scope() as session:
            repository = _repository(request, session)
            return _envelope(
                request,
                repository.profile_dict(repository.get(profile_id, version, kb_id=kb_id)),
            )

    @app.post(
        "/api/v1/retrieval-profiles/{profile_id}/versions/{version}:publish",
        response_model=ResourceEnvelope,
        tags=["检索配置"],
    )
    def publish_profile(
        profile_id: str,
        version: int,
        payload: RetrievalProfilePublishRequest,
        request: Request,
    ):
        with session_scope() as session:
            repository = _repository(request, session)
            row = repository.publish(
                payload.kb_id,
                profile_id,
                version,
                actor_id=request.state.principal.user_id,
                expected_current_profile_id=payload.expected_current_profile_id,
                expected_current_version=payload.expected_current_version,
                expected_current_config_hash=payload.expected_current_config_hash,
                expected_control_plane_revision=payload.expected_control_plane_revision,
                expected_max_version=payload.expected_max_version,
            )
            return _envelope(request, repository.profile_dict(row))

    @app.post(
        "/api/v1/knowledge-bases/{kb_id}:rollback-retrieval-profile",
        response_model=ResourceEnvelope,
        tags=["检索配置"],
    )
    def rollback_profile(
        kb_id: str,
        payload: RetrievalProfileRollbackRequest,
        request: Request,
    ):
        with session_scope() as session:
            repository = _repository(request, session)
            row = repository.rollback(
                kb_id,
                payload.profile_id,
                payload.version,
                actor_id=request.state.principal.user_id,
                expected_current_profile_id=payload.expected_current_profile_id,
                expected_current_version=payload.expected_current_version,
                expected_current_config_hash=payload.expected_current_config_hash,
                expected_control_plane_revision=payload.expected_control_plane_revision,
                expected_max_version=payload.expected_max_version,
            )
            return _envelope(request, repository.profile_dict(row))

    @app.post(
        "/api/v1/retrieval-profiles/{profile_id}/versions/{version}:preview",
        response_model=ResourceEnvelope,
        tags=["检索配置"],
    )
    def preview_profile(
        profile_id: str,
        version: int,
        payload: RetrievalProfilePreviewRequest,
        request: Request,
    ):
        # DB 会话在真实 Milvus 调用前关闭，避免长查询占用控制库事务。
        with session_scope() as session:
            repository = _repository(request, session)
            profile = repository.profile_dict(
                repository.get(profile_id, version, kb_id=payload.kb_id)
            )
        try:
            preview = preview_retrieval_profile(
                kb_id=payload.kb_id,
                query=payload.query,
                profile_record=profile,
                raw_filter=payload.retrieval_filter,
            )
        except RetrievalEngineError as exc:
            # 空活动读集以及已经变化/失效的索引作用域都不能继续检索。对客户端
            # 统一表达为“索引未就绪”，不返回 collection、placement 等内部事实。
            if exc.code == "RETRIEVAL_STALE_SCOPE":
                return _index_not_ready_response(request)
            raise
        return _envelope(request, preview)


__all__ = ["register_retrieval_routes"]
