"""TODO-02 SQL 持久任务队列。

任务行本身是队列事实；所有 Worker 写入均校验 lease_token（fencing token），
避免租约过期后的旧 Worker 覆盖新 Worker 的结果。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from app.infra.persistence.sql.models import (
    ChunkEditDraftModel,
    DocumentIndexBindingModel,
    DocumentModel,
    DocumentVersionModel,
    IngestionJobCheckpointModel,
    IngestionJobEventModel,
    IngestionJobModel,
    IngestionWorkerModel,
    IndexVersionModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.index_lifecycle_repository import IndexLifecycleRepository
from app.infra.persistence.sql.repositories import (
    ControlRepository,
    ResourceConflictError,
    ResourceNotFoundError,
    new_id,
)
from app.security.principal import current_principal


ACTIVE_STATUSES = {"queued", "leased", "running", "retry_wait", "processing"}
TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled"}


def utcnow() -> datetime:
    return datetime.now(UTC)


class LeaseLostError(ResourceConflictError):
    """Worker 的租约已丢失，当前执行结果必须丢弃。"""


class JobCancelledError(RuntimeError):
    pass


class DurableJobRepository:
    def __init__(self, session: Session):
        self.session = session

    def _job(self, job_id: str) -> IngestionJobModel:
        row = self.session.get(IngestionJobModel, job_id)
        if row is None:
            raise ResourceNotFoundError(f"ingestion job not found: {job_id}")
        version = self.session.get(DocumentVersionModel, row.document_version_id)
        document = self.session.get(DocumentModel, version.document_id) if version else None
        if document is None:
            raise ResourceNotFoundError(f"ingestion job not found: {job_id}")
        ControlRepository(self.session).get_knowledge_base_model(document.kb_id, include_deleted=True)
        return row

    def _leased_job(self, job_id: str, lease_token: str) -> IngestionJobModel:
        row = self._job(job_id)
        if row.lease_token != lease_token or row.status not in {"leased", "running"}:
            raise LeaseLostError(f"lease lost for ingestion job: {job_id}")
        return row

    def append_event(
        self,
        job: IngestionJobModel,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> IngestionJobEventModel:
        # cancel API 与 Worker 可能同时写事件；数据库原子自增避免读取同一旧序号。
        self.session.flush()
        event_status = job.status
        event_stage = job.stage
        event_progress = job.progress
        sequence = self.session.execute(
            update(IngestionJobModel)
            .where(IngestionJobModel.id == job.id)
            .values(last_event_sequence=IngestionJobModel.last_event_sequence + 1)
            .returning(IngestionJobModel.last_event_sequence)
            .execution_options(synchronize_session=False)
        ).scalar_one()
        set_committed_value(job, "last_event_sequence", sequence)
        event = IngestionJobEventModel(
            job_id=job.id,
            sequence=sequence,
            event_type=event_type,
            status=event_status,
            stage=event_stage,
            progress=event_progress,
            payload=payload or {},
        )
        self.session.add(event)
        self.session.flush()
        return event

    def ensure_created_event(self, job_id: str, payload: dict | None = None) -> None:
        job = self._job(job_id)
        if not job.last_event_sequence:
            self.append_event(job, "queued", payload)

    def claim_next(self, worker_id: str, lease_seconds: int) -> dict | None:
        """原子领取一个可运行任务；支持 SQLite 条件 UPDATE 和数据库行锁。"""
        now = utcnow()
        ready = or_(
            IngestionJobModel.status == "queued",
            and_(
                IngestionJobModel.status == "retry_wait",
                or_(IngestionJobModel.next_retry_at.is_(None), IngestionJobModel.next_retry_at <= now),
            ),
            and_(
                IngestionJobModel.status.in_({"leased", "running"}),
                IngestionJobModel.lease_expires_at <= now,
            ),
        )
        query = (
            select(IngestionJobModel)
            .where(ready)
            .order_by(IngestionJobModel.priority.desc(), IngestionJobModel.created_at.asc())
            .limit(1)
        )
        if self.session.bind is not None and self.session.bind.dialect.name != "sqlite":
            query = query.with_for_update(skip_locked=True)

        for _ in range(8):
            candidate = self.session.scalar(query)
            if candidate is None:
                return None
            old_status = candidate.status
            old_token = candidate.lease_token
            recovered = old_status in {"leased", "running"}
            token = uuid4().hex
            conditions = [IngestionJobModel.id == candidate.id, IngestionJobModel.status == old_status]
            if recovered:
                conditions.extend(
                    [
                        IngestionJobModel.lease_token == old_token,
                        IngestionJobModel.lease_expires_at <= now,
                    ]
                )
            elif old_status == "retry_wait":
                conditions.append(
                    or_(IngestionJobModel.next_retry_at.is_(None), IngestionJobModel.next_retry_at <= now)
                )
            result = self.session.execute(
                update(IngestionJobModel)
                .where(*conditions)
                .values(
                    status="leased",
                    stage="leased",
                    lease_owner=worker_id,
                    lease_token=token,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    heartbeat_at=now,
                    next_retry_at=None,
                    run_attempt=IngestionJobModel.run_attempt + 1,
                    recovered_count=IngestionJobModel.recovered_count + (1 if recovered else 0),
                    started_at=IngestionJobModel.started_at if candidate.started_at else now,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                self.session.expire_all()
                continue
            self.session.flush()
            # 条件 UPDATE 关闭了 ORM 会话同步；重新加载数据库中的 run_attempt/status/lease。
            self.session.expire_all()
            job = self._job(candidate.id)
            self.append_event(
                job,
                "lease_recovered" if recovered else "leased",
                {
                    "worker_id": worker_id,
                    "run_attempt": job.run_attempt,
                    "previous_status": old_status,
                },
            )
            claimed = self.detail(job)
            claimed["lease_token"] = token
            return claimed
        return None

    def mark_running(self, job_id: str, lease_token: str) -> dict:
        job = self._leased_job(job_id, lease_token)
        if job.cancel_requested_at:
            return self._cancel_leased(job, "cancelled before execution")
        job.status = "running"
        job.stage = job.checkpoint_stage or "node_entry"
        job.progress = max(job.progress, 0.02)
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        if version.status in {"uploaded", "failed"}:
            version.status = "parsing"
        self.append_event(job, "running", {"run_attempt": job.run_attempt})
        return self.detail(job)

    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: int) -> bool:
        now = utcnow()
        result = self.session.execute(
            update(IngestionJobModel)
            .where(
                IngestionJobModel.id == job_id,
                IngestionJobModel.lease_token == lease_token,
                IngestionJobModel.status.in_({"leased", "running"}),
            )
            .values(heartbeat_at=now, lease_expires_at=now + timedelta(seconds=lease_seconds))
            .execution_options(synchronize_session=False)
        )
        self.session.flush()
        return result.rowcount == 1

    def cancellation_requested(self, job_id: str, lease_token: str) -> bool:
        job = self._leased_job(job_id, lease_token)
        return job.cancel_requested_at is not None

    def update_stage(
        self,
        job_id: str,
        lease_token: str,
        stage: str,
        progress: float,
        detail: dict | None = None,
    ) -> dict:
        job = self._leased_job(job_id, lease_token)
        if job.cancel_requested_at:
            raise JobCancelledError(f"ingestion job cancelled: {job_id}")
        job.status = "running"
        job.stage = stage
        job.progress = max(job.progress, min(progress, 0.99))
        job.progress_detail = detail or {}
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        target = None
        if stage in {"node_metadata_extract", "node_link_and_table_chunk", "node_chunk_merge"}:
            target = "chunking"
        elif stage == "node_bge_embedding":
            target = "embedding"
        elif stage == "node_import_milvus":
            target = "staged"
        if target:
            self._advance_version(version, target)
        self.append_event(job, "stage_completed", detail or {"node": stage})
        return self.detail(job)

    def start_stage(
        self,
        job_id: str,
        lease_token: str,
        stage: str,
        progress: float,
        detail: dict | None = None,
    ) -> dict:
        """投影当前正在执行的节点，并保持 ``stage_completed`` 的完成语义。"""
        job = self._leased_job(job_id, lease_token)
        if job.cancel_requested_at:
            raise JobCancelledError(f"ingestion job cancelled: {job_id}")
        payload = {"node": stage, "phase": "started"}
        if detail:
            payload.update(detail)
        job.status = "running"
        job.stage = stage
        job.progress = max(job.progress, min(progress, 0.99))
        job.progress_detail = payload
        self.append_event(job, "stage_started", payload)
        return self.detail(job)

    @staticmethod
    def _advance_version(version: DocumentVersionModel, target: str) -> None:
        order = ["uploaded", "parsing", "chunking", "embedding", "staged", "published"]
        if version.status == "failed":
            version.status = "parsing"
        if version.status not in order:
            return
        for status in order[order.index(version.status) + 1 : order.index(target) + 1]:
            version.status = status

    def record_checkpoint(
        self,
        job_id: str,
        lease_token: str,
        *,
        stage: str,
        resume_node: str,
        artifact_uri: str,
        artifact_sha256: str,
        artifact_size: int,
        input_sha256: str,
        config_sha256: str,
    ) -> dict:
        job = self._leased_job(job_id, lease_token)
        sequence = (job.checkpoint_sequence or 0) + 1
        checkpoint = IngestionJobCheckpointModel(
            id=new_id("cp"),
            job_id=job_id,
            sequence=sequence,
            stage=stage,
            resume_node=resume_node,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            artifact_size=artifact_size,
            input_sha256=input_sha256,
            config_sha256=config_sha256,
        )
        self.session.add(checkpoint)
        job.checkpoint_stage = stage
        job.checkpoint_sequence = sequence
        self.append_event(
            job,
            "checkpoint_saved",
            {"checkpoint_sequence": sequence, "stage": stage, "resume_node": resume_node},
        )
        return self.checkpoint_dict(checkpoint)

    def record_manifest(self, job_id: str, lease_token: str, manifest: dict) -> dict:
        job = self._leased_job(job_id, lease_token)
        if manifest.get("document_version_id") != job.document_version_id:
            raise ResourceConflictError("manifest document version does not match leased job")
        if manifest.get("index_version_id") != job.index_version_id:
            raise ResourceConflictError("manifest index version does not match leased job")
        result = IndexLifecycleRepository(self.session).record_manifest(job, manifest)
        self.append_event(
            job,
            "manifest_validated",
            {
                "manifest_id": result["id"],
                "expected_chunk_count": result["expected_chunk_count"],
                "written_chunk_count": result["written_chunk_count"],
                "searchable_chunk_count": result["searchable_chunk_count"],
            },
        )
        return result

    def latest_checkpoint(self, job_id: str) -> dict | None:
        self._job(job_id)
        row = self.session.scalar(
            select(IngestionJobCheckpointModel)
            .where(IngestionJobCheckpointModel.job_id == job_id)
            .order_by(IngestionJobCheckpointModel.sequence.desc())
            .limit(1)
        )
        return self.checkpoint_dict(row) if row else None

    def schedule_retry(
        self,
        job_id: str,
        lease_token: str,
        *,
        error_category: str,
        error_code: str,
        safe_message: str,
        delay_seconds: float,
    ) -> dict:
        job = self._leased_job(job_id, lease_token)
        if job.cancel_requested_at:
            return self._cancel_leased(job, safe_message)
        if job.run_attempt >= job.max_attempts:
            return self.fail(
                job_id,
                lease_token,
                error_category=error_category,
                error_code=error_code,
                safe_message=safe_message,
            )
        job.status = "retry_wait"
        job.stage = "retry_wait"
        job.error_category = error_category
        job.error_code = error_code
        job.error_detail = safe_message[:1000]
        job.next_retry_at = utcnow() + timedelta(seconds=max(0, delay_seconds))
        job.lease_owner = None
        job.lease_token = None
        job.lease_expires_at = None
        self.append_event(
            job,
            "retry_scheduled",
            {
                "error_category": error_category,
                "error_code": error_code,
                "delay_seconds": delay_seconds,
                "next_retry_at": job.next_retry_at.isoformat(),
            },
        )
        return self.detail(job)

    def fail(
        self,
        job_id: str,
        lease_token: str,
        *,
        error_category: str,
        error_code: str,
        safe_message: str,
    ) -> dict:
        job = self._leased_job(job_id, lease_token)
        job.status = "failed"
        job.stage = "failed"
        job.error_category = error_category
        job.error_code = error_code
        job.error_detail = safe_message[:1000]
        job.finished_at = utcnow()
        self._clear_lease(job)
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        if job.operation != "rebuild" and version.status != "published":
            version.status = "failed"
        document = self.session.get(DocumentModel, version.document_id)
        assert document is not None
        if document.status != "deleted":
            document.status = "ready" if document.active_version_id else "failed"
        index = self.session.get(IndexVersionModel, job.index_version_id)
        if index is not None and index.status == "staging":
            self.session.flush()
            IndexLifecycleRepository(self.session)._refresh_index_validation(index.id)
        self.append_event(
            job,
            "failed",
            {"error_category": error_category, "error_code": error_code, "message": safe_message[:1000]},
        )
        return self.detail(job)

    def complete(
        self,
        job_id: str,
        lease_token: str,
        *,
        chunk_count: int,
        token_count: int = 0,
    ) -> dict:
        job = self._leased_job(job_id, lease_token)
        if job.cancel_requested_at:
            return self._cancel_leased(job, "cancelled before publish")
        lifecycle = IndexLifecycleRepository(self.session)
        manifest = lifecycle.manifest_for_job(job.id)
        if manifest is None or manifest.validation_status != "validated":
            raise ResourceConflictError(f"validated index manifest is required before publish: {job.id}")
        if not (
            manifest.expected_chunk_count
            == manifest.written_chunk_count
            == manifest.searchable_chunk_count
            == chunk_count
        ):
            raise ResourceConflictError("manifest count mismatch blocks publication")
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        document = self.session.get(DocumentModel, version.document_id)
        assert document is not None
        index = self.session.get(IndexVersionModel, job.index_version_id)
        if index is None:
            raise ResourceConflictError("ingestion target index is missing")
        binding = self.session.get(DocumentIndexBindingModel, (version.id, job.index_version_id))
        if binding is None or binding.status != "staged":
            raise ResourceConflictError("staged document-index binding is required before publish")
        binding.chunk_count = chunk_count
        binding.token_count = token_count
        if index.status == "staging":
            # Rebuild and document-update COW jobs only establish a validated staging
            # binding.  A separate publish transaction switches index + document read
            # pointers together; completing an individual job must not mutate online data.
            final_stage = "validated"
        else:
            self._advance_version(version, "published")
            version.published_at = utcnow()
            document.status = "ready"
            document.active_version_id = version.id
            version_ids = self.session.scalars(
                select(DocumentVersionModel.id).where(
                    DocumentVersionModel.document_id == document.id,
                    DocumentVersionModel.id != version.id,
                )
            ).all()
            if version_ids:
                self.session.query(DocumentIndexBindingModel).filter(
                    DocumentIndexBindingModel.document_version_id.in_(version_ids),
                    DocumentIndexBindingModel.status == "published",
                ).update(
                    {"status": "retired", "retired_at": utcnow()},
                    synchronize_session=False,
                )
            binding.status = "published"
            binding.published_at = utcnow()
            binding.retired_at = None
            final_stage = "published"
        job.status = "succeeded"
        job.stage = final_stage
        job.progress = 1.0
        job.finished_at = utcnow()
        job.error_category = None
        job.error_code = None
        job.error_detail = None
        self._clear_lease(job)
        chunk_draft = self.session.scalar(
            select(ChunkEditDraftModel).where(
                ChunkEditDraftModel.published_job_id == job.id
            )
        )
        if chunk_draft is not None:
            if (
                chunk_draft.published_document_version_id != version.id
                or chunk_draft.published_index_version_id != index.id
                or chunk_draft.status not in {"publishing", "indexed"}
            ):
                raise ResourceConflictError(
                    "chunk draft publication pointers do not match the completed job"
                )
            if index.status == "staging":
                chunk_draft.status = "indexed"
            else:
                chunk_draft.status = "published"
                chunk_draft.published_at = utcnow()
        if index.status == "staging":
            self.session.flush()
            lifecycle._refresh_index_validation(job.index_version_id)
        self.append_event(
            job,
            "succeeded",
            {
                "chunk_count": chunk_count,
                "token_count": token_count,
                "operation": job.operation,
                "binding_status": binding.status,
            },
        )
        return self.detail(job)

    def request_cancel(self, job_id: str, reason: str = "requested by user") -> dict:
        job = self._job(job_id)
        if job.status in TERMINAL_STATUSES:
            return self.detail(job)
        if job.cancel_requested_at is None:
            job.cancel_requested_at = utcnow()
        if job.status in {"queued", "retry_wait"}:
            return self._cancel_unleased(job, reason)
        self.append_event(job, "cancel_requested", {"reason": reason[:500]})
        return self.detail(job)

    def cancel_claimed(self, job_id: str, lease_token: str, reason: str) -> dict:
        return self._cancel_leased(self._leased_job(job_id, lease_token), reason)

    def _cancel_leased(self, job: IngestionJobModel, reason: str) -> dict:
        self._clear_lease(job)
        return self._cancel_unleased(job, reason)

    def _cancel_unleased(self, job: IngestionJobModel, reason: str) -> dict:
        job.status = "cancelled"
        job.stage = "cancelled"
        job.finished_at = utcnow()
        job.error_category = "cancelled"
        job.error_code = "CANCELLED"
        job.error_detail = reason[:1000]
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        if job.operation != "rebuild" and version.status != "published":
            version.status = "failed"
        document = self.session.get(DocumentModel, version.document_id)
        assert document is not None
        if document.status != "deleted":
            document.status = "ready" if document.active_version_id else "failed"
        index = self.session.get(IndexVersionModel, job.index_version_id)
        if index is not None and index.status == "staging":
            self.session.flush()
            IndexLifecycleRepository(self.session)._refresh_index_validation(index.id)
        self.append_event(job, "cancelled", {"reason": reason[:500]})
        return self.detail(job)

    @staticmethod
    def _clear_lease(job: IngestionJobModel) -> None:
        job.lease_owner = None
        job.lease_token = None
        job.lease_expires_at = None

    def list_jobs(self, *, kb_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        query = select(IngestionJobModel)
        principal = current_principal()
        if kb_id or principal:
            query = (
                query.join(DocumentVersionModel, IngestionJobModel.document_version_id == DocumentVersionModel.id)
                .join(DocumentModel, DocumentVersionModel.document_id == DocumentModel.id)
                .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            )
        if kb_id:
            query = query.where(DocumentModel.kb_id == kb_id)
        if principal:
            query = query.where(KnowledgeBaseModel.tenant_id == principal.tenant_id)
        if status:
            query = query.where(IngestionJobModel.status == status)
        rows = self.session.scalars(
            query.order_by(IngestionJobModel.created_at.desc()).limit(max(1, min(limit, 500)))
        ).all()
        return [self.detail(row) for row in rows]

    def events(self, job_id: str, after_sequence: int = 0, limit: int = 500) -> list[dict]:
        self._job(job_id)
        rows = self.session.scalars(
            select(IngestionJobEventModel)
            .where(
                IngestionJobEventModel.job_id == job_id,
                IngestionJobEventModel.sequence > after_sequence,
            )
            .order_by(IngestionJobEventModel.sequence.asc())
            .limit(max(1, min(limit, 1000)))
        ).all()
        return [self.event_dict(row) for row in rows]

    def payload(self, job_id: str) -> dict:
        job = self._job(job_id)
        version = self.session.get(DocumentVersionModel, job.document_version_id)
        assert version is not None
        document = self.session.get(DocumentModel, version.document_id)
        assert document is not None
        index = self.session.get(IndexVersionModel, job.index_version_id)
        assert index is not None
        kb = self.session.get(KnowledgeBaseModel, document.kb_id)
        assert kb is not None
        return {
            **self.detail(job),
            "object_key": version.object_key,
            "content_sha256": version.content_sha256,
            "parser_config": version.parser_config_snapshot or {},
            "document_id": document.id,
            "kb_id": document.kb_id,
            "tenant_id": kb.tenant_id,
            "logical_name": document.logical_name,
            "operation": job.operation,
            "collection_name": index.collection_alias,
            "vector_dim": index.vector_dim,
            "schema_version": index.schema_version,
            "embedding_model_id": index.embedding_model_id,
        }

    def worker_heartbeat(self, worker_id: str, *, status: str, current_job_id: str | None, metadata: dict | None = None) -> None:
        now = utcnow()
        row = self.session.get(IngestionWorkerModel, worker_id)
        if row is None:
            row = IngestionWorkerModel(
                id=worker_id,
                status=status,
                current_job_id=current_job_id,
                started_at=now,
                heartbeat_at=now,
                metadata_json=metadata or {},
            )
            self.session.add(row)
        else:
            row.status = status
            row.current_job_id = current_job_id
            row.heartbeat_at = now
            if metadata is not None:
                row.metadata_json = metadata
        self.session.flush()

    def list_workers(self) -> list[dict]:
        query = select(IngestionWorkerModel)
        principal = current_principal()
        if principal:
            # Worker 池本身是平台级基础设施，idle worker 没有租户归属。租户 API
            # 只能看到正在处理本租户任务的 worker，不能枚举其他租户或全局 idle
            # worker 的主机标识、心跳与 metadata。
            tenant_job_ids = (
                select(IngestionJobModel.id)
                .join(
                    DocumentVersionModel,
                    IngestionJobModel.document_version_id == DocumentVersionModel.id,
                )
                .join(
                    DocumentModel,
                    DocumentVersionModel.document_id == DocumentModel.id,
                )
                .join(
                    KnowledgeBaseModel,
                    KnowledgeBaseModel.id == DocumentModel.kb_id,
                )
                .where(KnowledgeBaseModel.tenant_id == principal.tenant_id)
            )
            query = query.where(
                IngestionWorkerModel.current_job_id.in_(tenant_job_ids)
            )
        rows = self.session.scalars(
            query.order_by(IngestionWorkerModel.id)
        ).all()
        return [
            {
                "id": row.id,
                "status": row.status,
                "current_job_id": row.current_job_id,
                "started_at": row.started_at.isoformat(),
                "heartbeat_at": row.heartbeat_at.isoformat(),
                "metadata": row.metadata_json,
            }
            for row in rows
        ]

    def detail(self, row: IngestionJobModel) -> dict:
        base = ControlRepository.ingestion_job_dict(row)
        version = self.session.get(DocumentVersionModel, row.document_version_id)
        document = self.session.get(DocumentModel, version.document_id) if version else None
        base.update(
            {
                "document_id": document.id if document else None,
                "kb_id": document.kb_id if document else None,
                "logical_name": document.logical_name if document else None,
                "terminal": row.status in TERMINAL_STATUSES,
            }
        )
        return base

    @staticmethod
    def event_dict(row: IngestionJobEventModel) -> dict:
        return {
            "id": row.id,
            "job_id": row.job_id,
            "sequence": row.sequence,
            "event_type": row.event_type,
            "status": row.status,
            "stage": row.stage,
            "progress": row.progress,
            "payload": row.payload,
            "created_at": row.created_at.isoformat(),
        }

    @staticmethod
    def checkpoint_dict(row: IngestionJobCheckpointModel) -> dict:
        return {
            "id": row.id,
            "job_id": row.job_id,
            "sequence": row.sequence,
            "stage": row.stage,
            "resume_node": row.resume_node,
            "artifact_uri": row.artifact_uri,
            "artifact_sha256": row.artifact_sha256,
            "artifact_size": row.artifact_size,
            "input_sha256": row.input_sha256,
            "config_sha256": row.config_sha256,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
