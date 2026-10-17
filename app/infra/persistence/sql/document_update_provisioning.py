"""文档更新索引的写时复制规划。

该协作器位于控制仓储与索引生命周期仓储之间，只依赖窄化后的控制面读取协议。
这样上传事务可以复用同一个 SQLAlchemy Session，同时不再让两个大型仓储门面
形成双向导入。
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.infra.persistence.sql.models import (
    DocumentModel,
    DocumentVersionModel,
    IndexVersionModel,
    IngestionJobEventModel,
    IngestionJobModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.primitives import ResourceConflictError, new_id
from app.shared.config.settings_config import settings


class ControlPlaneReadPort(Protocol):
    """写时复制规划所需的最小控制面读取能力。"""

    def get_knowledge_base_model(self, kb_id: str) -> KnowledgeBaseModel: ...

    def get_active_index(self, kb_id: str) -> IndexVersionModel: ...

    def get_document_model(self, document_id: str) -> DocumentModel: ...

    def get_document_version_model(self, version_id: str) -> DocumentVersionModel: ...


def _target_versions(index: IndexVersionModel) -> dict[str, str]:
    summary = index.manifest_summary if isinstance(index.manifest_summary, dict) else {}
    raw = summary.get("target_document_versions")
    if not isinstance(raw, dict):
        return {}
    return {
        str(document_id): str(version_id)
        for document_id, version_id in raw.items()
        if str(document_id).strip() and str(version_id).strip()
    }


def _queue_rebuild_job(
    session: Session,
    *,
    index: IndexVersionModel,
    document_version_id: str,
) -> IngestionJobModel:
    existing = session.scalar(
        select(IngestionJobModel)
        .where(
            IngestionJobModel.index_version_id == index.id,
            IngestionJobModel.document_version_id == document_version_id,
            IngestionJobModel.status.in_(
                {
                    "queued",
                    "leased",
                    "running",
                    "retry_wait",
                    "processing",
                    "succeeded",
                    "completed",
                }
            ),
        )
        .order_by(IngestionJobModel.attempt.desc())
        .limit(1)
    )
    if existing is not None:
        return existing
    latest_attempt = (
        session.scalar(
            select(func.max(IngestionJobModel.attempt)).where(
                IngestionJobModel.document_version_id == document_version_id
            )
        )
        or 0
    )
    job = IngestionJobModel(
        id=new_id("job"),
        config_version=settings.config_version,
        operation="rebuild",
        document_version_id=document_version_id,
        index_version_id=index.id,
        status="queued",
        stage="queued",
        progress=0,
        attempt=latest_attempt + 1,
        max_attempts=settings.ingestion_max_attempts,
        last_event_sequence=1,
    )
    session.add(job)
    session.flush()
    session.add(
        IngestionJobEventModel(
            job_id=job.id,
            sequence=1,
            event_type="queued",
            status="queued",
            stage="queued",
            progress=0,
            payload={
                "operation": "rebuild",
                "index_version_id": index.id,
                "config_version": job.config_version,
            },
        )
    )
    return job


def prepare_document_update(
    session: Session,
    *,
    control: ControlPlaneReadPort,
    kb_id: str,
    document_id: str,
    document_version_id: str,
) -> IndexVersionModel:
    """创建或复用文档更新用的 COW staging index。"""

    knowledge_base = control.get_knowledge_base_model(kb_id)
    active = control.get_active_index(kb_id)
    document = control.get_document_model(document_id)
    version = control.get_document_version_model(document_version_id)
    if document.kb_id != kb_id or version.document_id != document_id:
        raise ResourceConflictError("document update identity does not belong to knowledge base")

    staging = session.scalar(
        select(IndexVersionModel).where(
            IndexVersionModel.kb_id == kb_id,
            IndexVersionModel.status.in_({"creating", "staging"}),
        )
    )
    if staging is not None:
        if (
            staging.status != "staging"
            or staging.build_kind != "document_update"
            or staging.previous_version_id != active.id
        ):
            raise ResourceConflictError(f"knowledge base already has a staging index: {staging.id}")
        index = staging
        targets = _target_versions(index)
    else:
        version_no = (
            session.scalar(
                select(func.max(IndexVersionModel.version_no)).where(
                    IndexVersionModel.kb_id == kb_id
                )
            )
            or 0
        ) + 1
        index = IndexVersionModel(
            id=new_id("idx"),
            kb_id=kb_id,
            version_no=version_no,
            embedding_model_id=active.embedding_model_id,
            vector_dim=active.vector_dim,
            schema_version=active.schema_version,
            status="staging",
            active_slot=None,
            collection_alias=active.collection_alias,
            vector_governance_mode="managed",
            previous_version_id=active.id,
            build_kind="document_update",
            validation_status="pending",
            manifest_summary={},
        )
        session.add(index)
        session.flush()

        from app.infra.persistence.sql.vector_placement_provisioning import (
            provision_managed_index_placement,
        )

        provision_managed_index_placement(
            session,
            knowledge_base=knowledge_base,
            index=index,
            readable=False,
        )
        published_documents = session.scalars(
            select(DocumentModel).where(
                DocumentModel.kb_id == kb_id,
                DocumentModel.active_version_id.is_not(None),
                DocumentModel.status.in_({"ready", "processing", "disabled"}),
            )
        ).all()
        targets = {
            item.id: str(item.active_version_id)
            for item in published_documents
            if item.active_version_id
        }

    targets[document_id] = document_version_id
    index.manifest_summary = {
        **(index.manifest_summary or {}),
        "expected_documents": len(targets),
        "validated_documents": 0,
        "succeeded_documents": 0,
        "target_document_versions": dict(sorted(targets.items())),
    }
    index.validation_status = "pending"

    # 新索引不能跨版本读取旧索引，未变更文档也必须重建到新 index_version_id。
    for target_document_id, target_version_id in targets.items():
        if target_document_id == document_id and target_version_id == document_version_id:
            continue
        _queue_rebuild_job(
            session,
            index=index,
            document_version_id=target_version_id,
        )
    session.flush()
    return index


__all__ = ["ControlPlaneReadPort", "prepare_document_update"]
