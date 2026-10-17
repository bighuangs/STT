"""TODO-03 索引发布、重建、查询可见性与清理应用服务。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from app.infra.object_storage import minio_gateway
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.index_lifecycle_repository import IndexLifecycleRepository
from app.infra.persistence.sql.models import (
    DocumentIndexBindingModel,
    IndexBuildManifestModel,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.runtime.logger import PROJECT_ROOT


def create_index_rebuild(kb_id: str, **config) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).create_rebuild(kb_id, **config)


def list_index_lifecycle(kb_id: str) -> list[dict]:
    with session_scope() as session:
        return IndexLifecycleRepository(session).list_indexes(kb_id)


def get_index_lifecycle(index_version_id: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).get_index(index_version_id)


def publish_index(index_version_id: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).publish(index_version_id)


def rollback_index(kb_id: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).rollback(kb_id)


def set_document_enabled(document_id: str, enabled: bool) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).set_document_enabled(document_id, enabled)


def delete_document_with_cleanup(document_id: str, mode: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).soft_delete_document(document_id, mode)


def delete_knowledge_base_with_cleanup(kb_id: str, mode: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).soft_delete_knowledge_base(kb_id, mode)


def list_cleanup_jobs(
    limit: int = 100, kb_id: str | None = None
) -> list[dict]:
    with session_scope() as session:
        return IndexLifecycleRepository(session).list_cleanup_jobs(
            limit, kb_id=kb_id
        )


def get_cleanup_job(cleanup_job_id: str) -> dict:
    with session_scope() as session:
        return IndexLifecycleRepository(session).get_cleanup_job(cleanup_job_id)


def active_index_scopes(kb_id: str | None = None) -> list[dict]:
    with session_scope() as session:
        return IndexLifecycleRepository(session).active_scopes(kb_id)


def _expected_cleanup_expression(identity: dict[str, object]) -> str:
    return " and ".join(
        f"{field} == {json.dumps(str(identity[source]), ensure_ascii=False)}"
        for field, source in (
            ("tenant_id", "tenant_id"),
            ("kb_id", "kb_id"),
            ("doc_id", "document_id"),
            ("document_version_id", "document_version_id"),
            ("index_version_id", "index_version_id"),
        )
    )


def execute_cleanup(cleanup_job_id: str) -> dict:
    """执行可重放的物理清理；失败项保留为 partial，下次只需再次调用。"""
    with session_scope() as session:
        job = IndexLifecycleRepository(session).start_cleanup(cleanup_job_id)
    if job["status"] == "succeeded":
        return job

    plan = job.get("plan") or {}
    result = {"milvus_deleted": [], "objects_deleted": [], "originals_deleted": []}
    errors: list[str] = []
    seen_filters: set[tuple[str, str | None, str]] = set()
    client = milvus_gateway.client()
    for item in plan.get("milvus_filters") or []:
        collection = str(item.get("collection") or "")
        expression = str(item.get("filter") or "")
        partition_name = item.get("partition_name")
        partition_name = str(partition_name) if partition_name is not None else None
        identity = item.get("identity")
        required_identity = {
            "tenant_id",
            "kb_id",
            "document_id",
            "document_version_id",
            "index_version_id",
        }
        if not (
            collection
            and expression
            and isinstance(identity, dict)
            and required_identity <= set(identity)
            and all(str(identity[field]).strip() for field in required_identity)
        ):
            errors.append("milvus cleanup target lacks the five-dimensional identity")
            continue
        expected_expression = _expected_cleanup_expression(identity)
        target_matches = (
            job["target_kind"] == "document"
            and str(identity["document_id"]) == str(job["target_id"])
        ) or (
            job["target_kind"] == "knowledge_base"
            and str(identity["kb_id"]) == str(job["target_id"])
        )
        if expression != expected_expression or not target_matches:
            errors.append(
                "milvus cleanup target identity/filter does not match the persisted cleanup scope"
            )
            continue
        key = (collection, partition_name, expression)
        if key in seen_filters:
            continue
        seen_filters.add(key)
        try:
            if client is not None and client.has_collection(collection):
                delete_kwargs = {
                    "collection_name": collection,
                    "filter": expression,
                }
                if partition_name is not None:
                    delete_kwargs["partition_name"] = partition_name
                client.delete(**delete_kwargs)
            result["milvus_deleted"].append(
                {
                    "collection": collection,
                    "partition_name": partition_name,
                    "filter": expression,
                    "identity": dict(identity),
                }
            )
        except Exception as exc:
            errors.append(f"milvus {collection}: {exc}")

    for object_key in sorted(set(plan.get("object_keys") or [])):
        try:
            minio_gateway.client().remove_object(minio_gateway.bucket_name, object_key)
            result["objects_deleted"].append(object_key)
        except Exception as exc:
            errors.append(f"object {object_key}: {exc}")

    originals_root = (PROJECT_ROOT / "output" / "originals").resolve()
    for raw_path in sorted(set(plan.get("original_paths") or [])):
        try:
            target = Path(raw_path).resolve()
            if target != originals_root and originals_root not in target.parents:
                raise ValueError("path is outside output/originals")
            target.unlink(missing_ok=True)
            result["originals_deleted"].append(str(target))
        except Exception as exc:
            errors.append(f"original {raw_path}: {exc}")

    with session_scope() as session:
        return IndexLifecycleRepository(session).finish_cleanup(cleanup_job_id, result, errors)


def scan_orphan_objects(retention_seconds: int = 86400) -> dict:
    """只报告超过保留期且不被 staged/published binding 引用的图片对象。"""
    from datetime import UTC, datetime, timedelta

    cutoff = datetime.now(UTC) - timedelta(seconds=max(0, retention_seconds))
    with session_scope() as session:
        rows = (
            session.execute(
                select(IndexBuildManifestModel)
                .join(
                    DocumentIndexBindingModel,
                    DocumentIndexBindingModel.manifest_id == IndexBuildManifestModel.id,
                )
                .where(DocumentIndexBindingModel.status.in_({"staged", "published"}))
            )
            .scalars()
            .all()
        )
        referenced = {key for row in rows for key in (row.object_keys or []) if key}
    prefix = minio_gateway.image_dir.lstrip("/")
    orphans: list[dict] = []
    for obj in minio_gateway.client().list_objects(
        minio_gateway.bucket_name, prefix=prefix, recursive=True
    ):
        modified = getattr(obj, "last_modified", None)
        if obj.object_name not in referenced and (modified is None or modified <= cutoff):
            orphans.append(
                {
                    "object_key": obj.object_name,
                    "size": getattr(obj, "size", None),
                    "last_modified": modified.isoformat() if modified else None,
                }
            )
    return {
        "retention_seconds": retention_seconds,
        "referenced_count": len(referenced),
        "orphan_count": len(orphans),
        "orphans": orphans,
        "dry_run": True,
    }
