"""TODO-18 评测、反馈和人工标注 HTTP 路由。"""

from __future__ import annotations

from mimetypes import guess_type

from fastapi import BackgroundTasks, FastAPI, Query, Request, status
from fastapi.responses import FileResponse

from app.api.schemas.control_plane import ResourceEnvelope
from app.api.schemas.evaluation import (
    AnnotationArbitrateRequest,
    AnnotationReviewRequest,
    AnnotationTaskCreateRequest,
    DatasetImportRequest,
    EvaluationArtifact,
    EvaluationPreflightRequest,
    EvaluationRunCreateRequest,
    FeedbackCreateRequest,
    FeedbackTriageRequest,
)
from app.application.evaluation_service import (
    arbitrate_annotation,
    create_annotation,
    create_feedback,
    create_run,
    evaluation_preflight,
    execute_run,
    get_dataset,
    get_run,
    import_dataset,
    list_annotations,
    list_datasets,
    list_feedback,
    list_results,
    list_runs,
    report_path,
    review_annotation,
    triage_feedback,
)
from app.security.principal import Principal
from app.shared.runtime.logger import PROJECT_ROOT


def _envelope(request: Request, data: object) -> ResourceEnvelope:
    return ResourceEnvelope(request_id=request.state.request_id, data=data)


def _copy_principal(request: Request) -> Principal:
    """复制请求身份，避免后台任务依赖请求结束后已重置的 ContextVar。"""

    source: Principal = request.state.principal
    return Principal(
        user_id=source.user_id,
        tenant_id=source.tenant_id,
        roles=tuple(source.roles),
        scopes=tuple(source.scopes),
        auth_type=source.auth_type,
        credential_id=source.credential_id,
        request_id=source.request_id or request.state.request_id,
    )


def register_evaluation_routes(app: FastAPI) -> None:
    @app.get("/eval/html", include_in_schema=False)
    def evaluation_html():
        html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "evaluation.html"
        return FileResponse(path=html_path, media_type="text/html; charset=utf-8")

    @app.get(
        "/api/v1/evaluation/datasets",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def list_datasets_api(request: Request):
        return _envelope(request, list_datasets())

    @app.post(
        "/api/v1/evaluation/datasets",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["离线评测"],
    )
    def import_dataset_api(payload: DatasetImportRequest, request: Request):
        return _envelope(
            request,
            import_dataset(
                payload.manifest,
                parent_dataset_id=payload.parent_dataset_id,
            ),
        )

    @app.get(
        "/api/v1/evaluation/datasets/{dataset_id}",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def get_dataset_api(dataset_id: str, request: Request):
        return _envelope(request, get_dataset(dataset_id))

    @app.post(
        "/api/v1/evaluation/preflight",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def evaluation_preflight_api(payload: EvaluationPreflightRequest, request: Request):
        profile_pin = (
            {
                "retrieval_profile_id": payload.retrieval_profile_id,
                "retrieval_profile_version": payload.retrieval_profile_version,
            }
            if payload.retrieval_profile_id is not None
            else {}
        )
        return _envelope(
            request,
            evaluation_preflight(
                payload.dataset_id,
                payload.kb_id,
                mode=payload.mode,
                **profile_pin,
            ),
        )

    @app.get(
        "/api/v1/evaluation/runs",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def list_runs_api(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return _envelope(request, list_runs(limit))

    @app.post(
        "/api/v1/evaluation/runs",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["离线评测"],
    )
    def create_run_api(
        payload: EvaluationRunCreateRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ):
        run = create_run(
            dataset_id=payload.dataset_id,
            kb_id=payload.kb_id,
            mode=payload.mode,
            name=payload.name,
            baseline_run_id=payload.baseline_run_id,
            case_ids=payload.case_ids,
            retrieval_profile_id=payload.retrieval_profile_id,
            retrieval_profile_version=payload.retrieval_profile_version,
        )
        # FastAPI BackgroundTasks 在响应返回后运行；这里必须携带身份快照，
        # execute_run 会在自己的上下文中恢复它，从而保持租户隔离。
        background_tasks.add_task(execute_run, run["id"], _copy_principal(request))
        return _envelope(request, run)

    @app.get(
        "/api/v1/evaluation/runs/{run_id}",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def get_run_api(run_id: str, request: Request):
        return _envelope(request, get_run(run_id))

    @app.get(
        "/api/v1/evaluation/runs/{run_id}/results",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def list_results_api(
        run_id: str,
        request: Request,
        only_failed: bool = False,
    ):
        return _envelope(request, list_results(run_id, only_failed=only_failed))

    @app.get(
        "/api/v1/evaluation/runs/{run_id}/comparison",
        response_model=ResourceEnvelope,
        tags=["离线评测"],
    )
    def get_comparison_api(run_id: str, request: Request):
        run = get_run(run_id)
        return _envelope(
            request,
            {
                "run_id": run["id"],
                "baseline_run_id": run.get("baseline_run_id"),
                "comparison_summary": run.get("comparison_summary") or {},
                "gate_result": run.get("gate_result") or {},
            },
        )

    @app.get(
        "/api/v1/evaluation/runs/{run_id}/artifacts/{artifact}",
        tags=["离线评测"],
    )
    def download_artifact_api(run_id: str, artifact: EvaluationArtifact):
        path = report_path(run_id, artifact)
        media_type = {
            "json": "application/json",
            "summary": "application/json",
            "manifest": "application/json",
            "comparison": "application/json",
            "results": "application/x-ndjson",
            "markdown": "text/markdown; charset=utf-8",
            "html": "text/html; charset=utf-8",
        }.get(artifact) or guess_type(path.name)[0] or "application/octet-stream"
        kwargs = {
            "path": path,
            "media_type": media_type,
            "headers": {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        }
        if artifact == "html":
            return FileResponse(**kwargs)
        return FileResponse(**kwargs, filename=path.name)

    @app.post(
        "/api/v1/feedback",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["反馈闭环"],
    )
    def create_feedback_api(payload: FeedbackCreateRequest, request: Request):
        return _envelope(request, create_feedback(payload.model_dump()))

    @app.get(
        "/api/v1/feedback",
        response_model=ResourceEnvelope,
        tags=["反馈闭环"],
    )
    def list_feedback_api(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status", max_length=32),
        mine: bool = False,
    ):
        return _envelope(
            request,
            list_feedback(status=status_filter, mine=mine),
        )

    @app.post(
        "/api/v1/feedback/{feedback_id}:triage",
        response_model=ResourceEnvelope,
        tags=["反馈闭环"],
    )
    def triage_feedback_api(
        feedback_id: str,
        payload: FeedbackTriageRequest,
        request: Request,
    ):
        return _envelope(
            request,
            triage_feedback(
                feedback_id,
                status=payload.status,
                review_note=payload.review_note,
            ),
        )

    @app.get(
        "/api/v1/annotation-tasks",
        response_model=ResourceEnvelope,
        tags=["反馈闭环"],
    )
    def list_annotation_tasks_api(request: Request):
        return _envelope(request, list_annotations())

    @app.post(
        "/api/v1/annotation-tasks",
        response_model=ResourceEnvelope,
        status_code=status.HTTP_201_CREATED,
        tags=["反馈闭环"],
    )
    def create_annotation_task_api(
        payload: AnnotationTaskCreateRequest,
        request: Request,
    ):
        return _envelope(
            request,
            create_annotation(
                payload.feedback_id,
                target_dataset_key=payload.target_dataset_key,
                proposed_case=payload.proposed_case,
            ),
        )

    @app.post(
        "/api/v1/annotation-tasks/{task_id}:review",
        response_model=ResourceEnvelope,
        tags=["反馈闭环"],
    )
    def review_annotation_task_api(
        task_id: str,
        payload: AnnotationReviewRequest,
        request: Request,
    ):
        return _envelope(
            request,
            review_annotation(task_id, decision=payload.decision, note=payload.note),
        )

    @app.post(
        "/api/v1/annotation-tasks/{task_id}:arbitrate",
        response_model=ResourceEnvelope,
        tags=["反馈闭环"],
    )
    def arbitrate_annotation_task_api(
        task_id: str,
        payload: AnnotationArbitrateRequest,
        request: Request,
    ):
        return _envelope(
            request,
            arbitrate_annotation(
                task_id,
                decision=payload.decision,
                note=payload.note,
                resolved_case=payload.resolved_case,
            ),
        )
