"""TODO-03 索引版本、manifest、发布/回滚与清理计划仓储。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.infra.persistence.sql.models import (
    ChunkEditDraftModel,
    CleanupJobModel,
    DocumentIndexBindingModel,
    DocumentModel,
    DocumentVersionModel,
    IndexBuildManifestModel,
    IndexVersionModel,
    IngestionJobEventModel,
    IngestionJobModel,
    KnowledgeBaseModel,
)
from app.infra.persistence.sql.p3_models import (
    VectorCollectionFamilyModel,
    VectorPlacementModel,
)
from app.infra.persistence.sql.repositories import (
    ControlRepository,
    DEFAULT_KB_ID,
    ResourceConflictError,
    ResourceNotFoundError,
    new_id,
)
from app.shared.config.settings_config import settings
from app.security.principal import current_principal


def utcnow() -> datetime:
    return datetime.now(UTC)


class IndexLifecycleRepository:
    ACTIVE_JOB_STATUSES = {"queued", "leased", "running", "retry_wait", "processing"}

    def __init__(self, session: Session):
        self.session = session

    def _index(self, index_version_id: str) -> IndexVersionModel:
        row = self.session.get(IndexVersionModel, index_version_id)
        if row is None:
            raise ResourceNotFoundError(f"index version not found: {index_version_id}")
        ControlRepository(self.session).get_knowledge_base_model(row.kb_id, include_deleted=True)
        return row

    def _cleanup(self, cleanup_job_id: str) -> CleanupJobModel:
        row = self.session.get(CleanupJobModel, cleanup_job_id)
        if row is None:
            raise ResourceNotFoundError(f"cleanup job not found: {cleanup_job_id}")
        control = ControlRepository(self.session)
        if row.target_kind == "knowledge_base":
            control.get_knowledge_base_model(row.target_id, include_deleted=True)
        elif row.target_kind == "document":
            control.get_document_model(row.target_id, include_deleted=True)
        elif row.target_kind == "index_version":
            self._index(row.target_id)
        return row

    @staticmethod
    def _collection_name(base: str, schema_version: str, vector_dim: int, version_no: int) -> str:
        suffix = re.sub(r"[^a-zA-Z0-9_]", "_", schema_version).strip("_") or "schema"
        return f"{base}_{suffix}_{vector_dim}_v{version_no}"[:255]

    def _vector_cleanup_targets(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        document_ids: list[str],
        document_version_ids: list[str],
    ) -> list[dict]:
        """生成不可扩大为其他租户/版本的物理清理目标。

        一个文档可能被多个索引版本引用，而一个索引版本在迁移窗口内又可能有
        source/target 两个 placement。因此这里按每条 binding 展开所有已登记
        placement；只有整个 index 从未进入 P3 placement 控制面时才使用 SQL
        ``collection_alias`` 作为显式 legacy 目标。
        """

        if not document_ids or not document_version_ids:
            return []
        binding_rows = self.session.execute(
            select(
                DocumentVersionModel.document_id,
                DocumentIndexBindingModel.document_version_id,
                DocumentIndexBindingModel.index_version_id,
                IndexVersionModel.collection_alias,
            )
            .join(
                DocumentIndexBindingModel,
                DocumentIndexBindingModel.document_version_id == DocumentVersionModel.id,
            )
            .join(
                IndexVersionModel,
                IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
            )
            .where(
                DocumentVersionModel.document_id.in_(document_ids),
                DocumentVersionModel.id.in_(document_version_ids),
                IndexVersionModel.kb_id == kb_id,
            )
        ).all()
        index_ids = {str(row.index_version_id) for row in binding_rows}
        placements: dict[str, list[tuple[str, str | None]]] = {}
        if index_ids:
            placement_rows = self.session.execute(
                select(
                    VectorPlacementModel.index_version_id,
                    VectorPlacementModel.partition_bucket,
                    VectorCollectionFamilyModel.physical_collection,
                )
                .join(
                    VectorCollectionFamilyModel,
                    VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
                )
                .where(
                    VectorPlacementModel.tenant_id == tenant_id,
                    VectorPlacementModel.kb_id == kb_id,
                    VectorPlacementModel.index_version_id.in_(index_ids),
                )
            ).all()
            for row in placement_rows:
                placements.setdefault(
                    str(row.index_version_id),
                    [],
                ).append(
                    (
                        str(row.physical_collection),
                        (str(row.partition_bucket) if row.partition_bucket is not None else None),
                    )
                )

        targets: list[dict] = []
        seen: set[tuple[str, str | None, str, str, str]] = set()
        for row in binding_rows:
            index_version_id = str(row.index_version_id)
            physical_targets = placements.get(index_version_id)
            if not physical_targets:
                physical_targets = [(str(row.collection_alias), None)]
            identity = {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "document_id": str(row.document_id),
                "document_version_id": str(row.document_version_id),
                "index_version_id": index_version_id,
            }
            expression = " and ".join(
                f"{field} == {json.dumps(value, ensure_ascii=False)}"
                for field, value in (
                    ("tenant_id", identity["tenant_id"]),
                    ("kb_id", identity["kb_id"]),
                    ("doc_id", identity["document_id"]),
                    (
                        "document_version_id",
                        identity["document_version_id"],
                    ),
                    ("index_version_id", identity["index_version_id"]),
                )
            )
            for collection, partition_name in physical_targets:
                key = (
                    collection,
                    partition_name,
                    identity["document_version_id"],
                    identity["index_version_id"],
                    identity["document_id"],
                )
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    {
                        "collection": collection,
                        "partition_name": partition_name,
                        "filter": expression,
                        "identity": identity,
                    }
                )
        return targets

    @staticmethod
    def _target_versions(index: IndexVersionModel) -> dict[str, str]:
        """Return the frozen logical-document snapshot owned by a staging index.

        ``manifest_summary`` is deliberately used as the immutable build snapshot instead
        of re-reading ``documents.active_version_id`` while workers are running.  The
        latter is the *currently published* read set and may legitimately differ from a
        document-update candidate until the atomic publish transaction commits.
        """

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
        self,
        *,
        index: IndexVersionModel,
        document_version_id: str,
    ) -> IngestionJobModel:
        existing = self.session.scalar(
            select(IngestionJobModel)
            .where(
                IngestionJobModel.index_version_id == index.id,
                IngestionJobModel.document_version_id == document_version_id,
                IngestionJobModel.status.in_(
                    {"queued", "leased", "running", "retry_wait", "processing", "succeeded", "completed"}
                ),
            )
            .order_by(IngestionJobModel.attempt.desc())
            .limit(1)
        )
        if existing is not None:
            return existing
        latest_attempt = (
            self.session.scalar(
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
        self.session.add(job)
        self.session.flush()
        self.session.add(
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
        self,
        *,
        kb_id: str,
        document_id: str,
        document_version_id: str,
    ) -> IndexVersionModel:
        """Create/reuse a COW staging index for a document import.

        A published index version is an immutable read set.  Replacing one document (or
        adding a document to a non-empty KB) therefore creates a new index version and
        rebuilds the other published document versions under the new index identity.
        The caller queues the uploaded target version itself; this method only queues the
        unchanged members of the frozen snapshot.
        """

        control = ControlRepository(self.session)
        knowledge_base = control.get_knowledge_base_model(kb_id)
        active = control.get_active_index(kb_id)
        document = control.get_document_model(document_id)
        version = control.get_document_version_model(document_version_id)
        if document.kb_id != kb_id or version.document_id != document_id:
            raise ResourceConflictError("document update identity does not belong to knowledge base")

        staging = self.session.scalar(
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
                raise ResourceConflictError(
                    f"knowledge base already has a staging index: {staging.id}"
                )
            index = staging
            targets = self._target_versions(index)
        else:
            version_no = (
                self.session.scalar(
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
            self.session.add(index)
            self.session.flush()
            from app.infra.persistence.sql.vector_placement_provisioning import (
                provision_managed_index_placement,
            )

            provision_managed_index_placement(
                self.session,
                knowledge_base=knowledge_base,
                index=index,
                readable=False,
            )
            published_documents = self.session.scalars(
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

        # All unchanged members must be rewritten with the new index_version_id because
        # the query read set is five-dimensional and never widens to the previous index.
        for target_document_id, target_version_id in targets.items():
            if target_document_id == document_id and target_version_id == document_version_id:
                continue
            self._queue_rebuild_job(
                index=index,
                document_version_id=target_version_id,
            )
        self.session.flush()
        return index

    def create_rebuild(
        self,
        kb_id: str,
        *,
        embedding_model_id: str | None = None,
        vector_dim: int | None = None,
        schema_version: str | None = None,
    ) -> dict:
        control = ControlRepository(self.session)
        control.get_knowledge_base_model(kb_id)
        active = control.get_active_index(kb_id)
        active_job = self.session.scalar(
            select(IngestionJobModel.id)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .where(
                DocumentModel.kb_id == kb_id,
                IngestionJobModel.status.in_(self.ACTIVE_JOB_STATUSES),
            )
            .limit(1)
        )
        if active_job is not None:
            raise ResourceConflictError(
                f"knowledge base has an unfinished ingestion job: {active_job}"
            )
        existing = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.kb_id == kb_id,
                IndexVersionModel.status.in_({"creating", "staging"}),
            )
        )
        if existing is not None:
            raise ResourceConflictError(
                f"knowledge base already has a staging index: {existing.id}"
            )

        version_no = (
            self.session.scalar(
                select(func.max(IndexVersionModel.version_no)).where(
                    IndexVersionModel.kb_id == kb_id
                )
            )
            or 0
        ) + 1
        model_id = embedding_model_id or active.embedding_model_id
        dim = vector_dim or active.vector_dim
        schema = schema_version or active.schema_version
        compatible = (
            model_id == active.embedding_model_id
            and dim == active.vector_dim
            and schema == active.schema_version
        )
        collection = (
            active.collection_alias
            if compatible
            else self._collection_name(active.collection_alias, schema, dim, version_no)
        )
        index = IndexVersionModel(
            id=new_id("idx"),
            kb_id=kb_id,
            version_no=version_no,
            embedding_model_id=model_id,
            vector_dim=dim,
            schema_version=schema,
            status="staging",
            active_slot=None,
            collection_alias=collection,
            vector_governance_mode="managed",
            previous_version_id=active.id,
            build_kind="rebuild",
            validation_status="pending",
            manifest_summary={},
        )
        self.session.add(index)
        self.session.flush()
        from app.infra.persistence.sql.vector_placement_provisioning import (
            provision_managed_index_placement,
        )

        knowledge_base = self.session.get(KnowledgeBaseModel, kb_id)
        if knowledge_base is None:
            raise ResourceConflictError("rebuild knowledge base is missing")
        provision_managed_index_placement(
            self.session,
            knowledge_base=knowledge_base,
            index=index,
            readable=False,
        )

        documents = self.session.scalars(
            select(DocumentModel).where(
                DocumentModel.kb_id == kb_id,
                DocumentModel.active_version_id.is_not(None),
                DocumentModel.status.in_({"ready", "disabled"}),
            )
        ).all()
        targets = {
            document.id: str(document.active_version_id)
            for document in documents
            if document.active_version_id
        }
        index.manifest_summary = {
            "expected_documents": len(targets),
            "validated_documents": 0,
            "succeeded_documents": 0,
            "target_document_versions": dict(sorted(targets.items())),
        }
        jobs = [
            self._queue_rebuild_job(
                index=index,
                document_version_id=document_version_id,
            )
            for document_version_id in targets.values()
        ]

        if not jobs:
            index.validation_status = "validated"
            index.manifest_summary = {
                **index.manifest_summary,
                "expected_documents": 0,
                "validated_documents": 0,
                "succeeded_documents": 0,
            }
        self.session.flush()
        return {
            "index_version": self.index_detail(index),
            "jobs": [ControlRepository.ingestion_job_dict(job) for job in jobs],
        }

    def list_indexes(self, kb_id: str) -> list[dict]:
        ControlRepository(self.session).get_knowledge_base_model(kb_id)
        rows = self.session.scalars(
            select(IndexVersionModel)
            .where(IndexVersionModel.kb_id == kb_id)
            .order_by(IndexVersionModel.version_no.desc())
        ).all()
        return [self.index_detail(row) for row in rows]

    def get_index(self, index_version_id: str) -> dict:
        return self.index_detail(self._index(index_version_id))

    def record_manifest(self, job: IngestionJobModel, manifest: dict) -> dict:
        index = self._index(job.index_version_id)
        if index.status == "active":
            summary = index.manifest_summary if isinstance(index.manifest_summary, dict) else {}
            bootstrap_job_ids = {
                str(value) for value in summary.get("bootstrap_job_ids", []) if str(value)
            }
            if job.id not in bootstrap_job_ids:
                raise ResourceConflictError(
                    "published index version is immutable; import into a staging index"
                )
        elif index.status != "staging":
            raise ResourceConflictError(
                f"index version cannot accept manifests from status: {index.status}"
            )
        knowledge_base = self.session.get(KnowledgeBaseModel, index.kb_id)
        if knowledge_base is None:
            raise ResourceConflictError("manifest knowledge base is missing")
        placement_rows = self.session.execute(
            select(VectorPlacementModel, VectorCollectionFamilyModel)
            .join(
                VectorCollectionFamilyModel,
                VectorCollectionFamilyModel.id == VectorPlacementModel.family_id,
            )
            .where(
                VectorPlacementModel.tenant_id == knowledge_base.tenant_id,
                VectorPlacementModel.kb_id == index.kb_id,
                VectorPlacementModel.index_version_id == job.index_version_id,
            )
        ).all()
        if placement_rows:
            writable = [
                (placement, family)
                for placement, family in placement_rows
                if placement.writable and placement.write_slot == "active"
            ]
            if len(writable) != 1:
                raise ResourceConflictError(
                    "manifest index lacks a unique writable vector placement"
                )
            placement, family = writable[0]
            if (
                manifest.get("vector_placement_id") != placement.id
                or int(manifest.get("vector_placement_epoch") or 0) != placement.epoch
                or manifest.get("vector_compatibility_fingerprint")
                != family.compatibility_fingerprint
            ):
                raise ResourceConflictError("manifest vector placement/fencing identity is stale")
            expected_collection = family.physical_collection
        else:
            if index.vector_governance_mode != "legacy":
                raise ResourceConflictError("managed manifest index lacks a vector placement")
            if manifest.get("vector_legacy_policy") not in {
                None,
                "",
                "control_plane_legacy",
            }:
                raise ResourceConflictError("manifest legacy vector policy is not authorized")
            expected_collection = index.collection_alias
        if manifest.get("collection_name") != expected_collection:
            raise ResourceConflictError("manifest collection does not match target index version")
        if int(manifest.get("vector_dim") or 0) != index.vector_dim:
            raise ResourceConflictError(
                "manifest vector dimension does not match target index version"
            )
        errors = list(manifest.get("validation_errors") or [])
        status = manifest.get("validation_status") or ("validated" if not errors else "failed")
        if status != "validated" or errors:
            raise ResourceConflictError(f"index manifest validation failed: {errors or status}")
        existing = self.session.scalar(
            select(IndexBuildManifestModel).where(IndexBuildManifestModel.job_id == job.id)
        )
        values = {
            "document_version_id": job.document_version_id,
            "index_version_id": job.index_version_id,
            "input_sha256": manifest["input_sha256"],
            "config_sha256": manifest["config_sha256"],
            "expected_chunk_count": int(manifest["expected_chunk_count"]),
            "written_chunk_count": int(manifest["written_chunk_count"]),
            "searchable_chunk_count": int(manifest["searchable_chunk_count"]),
            "chunk_ids_sha256": manifest["chunk_ids_sha256"],
            "chunk_ids": list(manifest.get("chunk_ids") or []),
            "object_keys": list(manifest.get("object_keys") or []),
            "vector_dim": int(manifest["vector_dim"]),
            "collection_name": manifest["collection_name"],
            "validation_status": "validated",
            "validation_errors": [],
            "validated_at": utcnow(),
        }
        if existing is None:
            existing = IndexBuildManifestModel(id=new_id("mf"), job_id=job.id, **values)
            self.session.add(existing)
        else:
            for key, value in values.items():
                setattr(existing, key, value)
        binding = self.session.get(
            DocumentIndexBindingModel, (job.document_version_id, job.index_version_id)
        )
        if binding is None:
            binding = DocumentIndexBindingModel(
                document_version_id=job.document_version_id,
                index_version_id=job.index_version_id,
                status="staged",
                job_id=job.id,
                manifest_id=existing.id,
                staged_at=utcnow(),
                published_at=None,
                chunk_count=existing.written_chunk_count,
                token_count=0,
            )
            self.session.add(binding)
        else:
            binding.status = "staged"
            binding.job_id = job.id
            binding.manifest_id = existing.id
            binding.staged_at = utcnow()
            binding.chunk_count = existing.written_chunk_count
        self.session.flush()
        self._refresh_index_validation(job.index_version_id)
        return self.manifest_dict(existing)

    def manifest_for_job(self, job_id: str) -> IndexBuildManifestModel | None:
        return self.session.scalar(
            select(IndexBuildManifestModel).where(IndexBuildManifestModel.job_id == job_id)
        )

    def _refresh_index_validation(self, index_version_id: str) -> None:
        index = self._index(index_version_id)
        # Validation is a property of a not-yet-published snapshot.  Never rewrite a
        # published/retired index from a later ingestion manifest.
        if index.status != "staging" or index.build_kind not in {
            "rebuild",
            "document_update",
        }:
            return
        targets = self._target_versions(index)
        if not targets and index.build_kind == "rebuild":
            # Compatibility for staging rebuilds created before the frozen target map was
            # introduced.  New builds always persist the map at creation time.
            documents = self.session.scalars(
                select(DocumentModel).where(
                    DocumentModel.kb_id == index.kb_id,
                    DocumentModel.active_version_id.is_not(None),
                    DocumentModel.status.in_({"ready", "processing", "disabled"}),
                )
            ).all()
            targets = {
                item.id: str(item.active_version_id)
                for item in documents
                if item.active_version_id
            }

        target_versions = set(targets.values())
        validated_versions = set(
            self.session.scalars(
                select(IndexBuildManifestModel.document_version_id).where(
                    IndexBuildManifestModel.index_version_id == index.id,
                    IndexBuildManifestModel.document_version_id.in_(target_versions),
                    IndexBuildManifestModel.validation_status == "validated",
                )
            ).all()
            if target_versions
            else []
        )
        succeeded_versions = set(
            self.session.scalars(
                select(IngestionJobModel.document_version_id).where(
                    IngestionJobModel.index_version_id == index.id,
                    IngestionJobModel.document_version_id.in_(target_versions),
                    IngestionJobModel.status.in_({"succeeded", "completed"}),
                )
            ).all()
            if target_versions
            else []
        )
        failed_versions = set(
            self.session.scalars(
                select(IngestionJobModel.document_version_id).where(
                    IngestionJobModel.index_version_id == index.id,
                    IngestionJobModel.document_version_id.in_(target_versions),
                    IngestionJobModel.status.in_({"failed", "cancelled"}),
                )
            ).all()
            if target_versions
            else []
        )
        expected = len(target_versions)
        validated = len(validated_versions)
        succeeded = len(succeeded_versions)
        if failed_versions:
            index.validation_status = "failed"
        elif validated_versions == target_versions and succeeded_versions == target_versions:
            index.validation_status = "validated"
        else:
            index.validation_status = "pending"
        index.manifest_summary = {
            **(index.manifest_summary or {}),
            "expected_documents": expected,
            "validated_documents": validated,
            "succeeded_documents": succeeded,
            "failed_documents": len(failed_versions),
            "target_document_versions": dict(sorted(targets.items())),
        }

    def _activate_vector_read_slot(
        self,
        *,
        target: IndexVersionModel,
        retiring: IndexVersionModel | None,
    ) -> None:
        """与逻辑 index 槽位在同一事务内切换物理 placement 槽位。"""

        target_rows = self.session.scalars(
            select(VectorPlacementModel).where(VectorPlacementModel.index_version_id == target.id)
        ).all()
        if target.vector_governance_mode == "managed":
            writable = [
                row
                for row in target_rows
                if row.writable and row.write_slot == "active" and row.status == "active"
            ]
            if len(writable) == 1:
                target_placement = writable[0]
            elif (
                target.status == "retired"
                and len(target_rows) == 1
                and target_rows[0].status == "retired"
                and not target_rows[0].readable
                and not target_rows[0].writable
            ):
                target_placement = target_rows[0]
            else:
                raise ResourceConflictError(
                    "managed publish target lacks one active writable placement"
                )
            family = self.session.get(VectorCollectionFamilyModel, target_placement.family_id)
            if family is None or family.status != "active":
                raise ResourceConflictError("managed publish target family is not active")
        elif target_rows:
            raise ResourceConflictError("legacy index unexpectedly owns vector placements")
        else:
            target_placement = None

        if retiring is not None:
            for row in self.session.scalars(
                select(VectorPlacementModel).where(
                    VectorPlacementModel.index_version_id == retiring.id
                )
            ).all():
                row.readable = False
                row.writable = False
                row.read_slot = None
                row.write_slot = None
                row.status = "retired"
                row.valid_to = utcnow()

        if target_placement is not None:
            target_placement.readable = True
            target_placement.writable = True
            target_placement.read_slot = "active"
            target_placement.write_slot = "active"
            target_placement.status = "active"
            target_placement.valid_to = None

    def publish(self, index_version_id: str) -> dict:
        index = self._index(index_version_id)
        if index.status != "staging" or index.validation_status != "validated":
            raise ResourceConflictError(
                f"index is not publishable: status={index.status}, validation={index.validation_status}"
            )
        unfinished = self.session.scalar(
            select(IngestionJobModel.id)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .where(
                DocumentModel.kb_id == index.kb_id,
                IngestionJobModel.status.in_(self.ACTIVE_JOB_STATUSES),
            )
            .limit(1)
        )
        if unfinished is not None:
            raise ResourceConflictError(
                f"index publish is blocked by unfinished ingestion job: {unfinished}"
            )
        targets = self._target_versions(index)
        if not targets:
            # Compatibility for pre-COW staging rows: derive the frozen set from their
            # staged bindings.  New rebuilds always carry target_document_versions.
            staged_rows = self.session.execute(
                select(DocumentVersionModel.document_id, DocumentIndexBindingModel.document_version_id)
                .join(
                    DocumentIndexBindingModel,
                    DocumentIndexBindingModel.document_version_id == DocumentVersionModel.id,
                )
                .where(
                    DocumentIndexBindingModel.index_version_id == index.id,
                    DocumentIndexBindingModel.status == "staged",
                )
            ).all()
            targets = {
                str(document_id): str(document_version_id)
                for document_id, document_version_id in staged_rows
            }
        target_version_ids = set(targets.values())
        staged_target_ids = set(
            self.session.scalars(
                select(DocumentIndexBindingModel.document_version_id).where(
                    DocumentIndexBindingModel.index_version_id == index.id,
                    DocumentIndexBindingModel.document_version_id.in_(target_version_ids),
                    DocumentIndexBindingModel.status == "staged",
                )
            ).all()
            if target_version_ids
            else []
        )
        if staged_target_ids != target_version_ids:
            missing = sorted(target_version_ids - staged_target_ids)
            raise ResourceConflictError(
                f"index publish target bindings are incomplete: {missing}"
            )
        current = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.kb_id == index.kb_id,
                IndexVersionModel.active_slot == "active",
                IndexVersionModel.status == "active",
            )
        )
        now = utcnow()
        self._activate_vector_read_slot(target=index, retiring=current)
        if current is not None:
            current.active_slot = None
            current.status = "retired"
            current.retired_at = now
            # 唯一键 (kb_id, active_slot) 要求先释放 active 槽；仍在同一事务内，
            # 其他连接不会观察到中间态。
            self.session.flush()
            self.session.query(DocumentIndexBindingModel).filter(
                DocumentIndexBindingModel.index_version_id == current.id,
                DocumentIndexBindingModel.status == "published",
            ).update({"status": "retired", "retired_at": now}, synchronize_session=False)
        index.status = "active"
        index.active_slot = "active"
        index.activated_at = now
        index.retired_at = None
        if target_version_ids:
            self.session.query(DocumentIndexBindingModel).filter(
                DocumentIndexBindingModel.index_version_id == index.id,
                DocumentIndexBindingModel.document_version_id.in_(target_version_ids),
                DocumentIndexBindingModel.status == "staged",
            ).update(
                {"status": "published", "published_at": now, "retired_at": None},
                synchronize_session=False,
            )
            # A batch may supersede an earlier target while the same document-update
            # staging index is still building.  Such validated rows stay physically
            # harmless but must never become part of the published read set.
            self.session.query(DocumentIndexBindingModel).filter(
                DocumentIndexBindingModel.index_version_id == index.id,
                DocumentIndexBindingModel.document_version_id.not_in(target_version_ids),
                DocumentIndexBindingModel.status == "staged",
            ).update(
                {"status": "retired", "retired_at": now},
                synchronize_session=False,
            )
        for document_id, document_version_id in targets.items():
            document = self.session.get(DocumentModel, document_id)
            version = self.session.get(DocumentVersionModel, document_version_id)
            if (
                document is None
                or version is None
                or document.kb_id != index.kb_id
                or version.document_id != document.id
            ):
                raise ResourceConflictError(
                    "index publish target document/version identity is stale"
                )
            if version.status == "staged":
                version.status = "published"
                version.published_at = now
            elif version.status != "published":
                raise ResourceConflictError(
                    f"index publish target version is not staged: {document_version_id}/{version.status}"
                )
            document.active_version_id = version.id
            if document.status != "disabled":
                document.status = "ready"
        published_drafts = self.session.scalars(
            select(ChunkEditDraftModel).where(
                ChunkEditDraftModel.published_index_version_id == index.id,
                ChunkEditDraftModel.published_document_version_id.in_(
                    target_version_ids
                ),
                ChunkEditDraftModel.status.in_({"publishing", "indexed"}),
            )
        ).all()
        for draft in published_drafts:
            draft.status = "published"
            draft.published_at = now
        self.session.flush()
        return self.index_detail(index)

    def rollback(self, kb_id: str) -> dict:
        control = ControlRepository(self.session)
        current = control.get_active_index(kb_id)
        if not current.previous_version_id:
            raise ResourceConflictError("active index has no previous version to roll back to")
        previous = self._index(current.previous_version_id)
        if previous.kb_id != kb_id or previous.status != "retired":
            raise ResourceConflictError("previous index is not retained and rollback-ready")
        if previous.validation_status != "validated":
            raise ResourceConflictError("previous index is not validated and rollback-ready")
        now = utcnow()
        self._activate_vector_read_slot(target=previous, retiring=current)
        current.active_slot = None
        current.status = "retired"
        current.retired_at = now
        self.session.flush()
        previous.active_slot = "active"
        previous.status = "active"
        previous.activated_at = now
        previous.retired_at = None
        self.session.query(DocumentIndexBindingModel).filter(
            DocumentIndexBindingModel.index_version_id == current.id,
            DocumentIndexBindingModel.status == "published",
        ).update({"status": "retired", "retired_at": now}, synchronize_session=False)
        self.session.query(DocumentIndexBindingModel).filter(
            DocumentIndexBindingModel.index_version_id == previous.id,
            DocumentIndexBindingModel.status == "retired",
        ).update(
            {"status": "published", "published_at": now, "retired_at": None},
            synchronize_session=False,
        )
        previous_rows = self.session.execute(
            select(DocumentVersionModel.document_id, DocumentIndexBindingModel.document_version_id)
            .join(
                DocumentIndexBindingModel,
                DocumentIndexBindingModel.document_version_id == DocumentVersionModel.id,
            )
            .where(
                DocumentIndexBindingModel.index_version_id == previous.id,
                DocumentIndexBindingModel.status == "published",
            )
        ).all()
        previous_targets = {
            str(document_id): str(document_version_id)
            for document_id, document_version_id in previous_rows
        }
        current_targets = self._target_versions(current)
        for document_id in set(current_targets) | set(previous_targets):
            document = self.session.get(DocumentModel, document_id)
            if document is None or document.kb_id != kb_id:
                raise ResourceConflictError("rollback document identity is stale")
            restored_version_id = previous_targets.get(document_id)
            document.active_version_id = restored_version_id
            if restored_version_id is None:
                # The document did not exist in the previous read set.  Keep its immutable
                # version fact, but close visibility until a later explicit publication.
                if document.status != "deleted":
                    document.status = "disabled"
            elif document.status not in {"disabled", "deleted"}:
                document.status = "ready"
        self.session.flush()
        return {"active": self.index_detail(previous), "rolled_back_from": current.id}

    def set_document_enabled(self, document_id: str, enabled: bool) -> dict:
        control = ControlRepository(self.session)
        document = control.get_document_model(document_id)
        target = "ready" if enabled else "disabled"
        if document.status == target:
            return control.document_detail(document)
        if enabled and document.status != "disabled":
            raise ResourceConflictError(f"document cannot be enabled from {document.status}")
        if not enabled and document.status != "ready":
            raise ResourceConflictError(f"document cannot be disabled from {document.status}")
        document.status = target
        self.session.flush()
        return control.document_detail(document)

    def soft_delete_document(self, document_id: str, mode: str) -> dict:
        if mode not in {"soft", "purge"}:
            raise ResourceConflictError(f"unsupported delete mode: {mode}")
        control = ControlRepository(self.session)
        document = control.get_document_model(document_id)
        versions = self.session.scalars(
            select(DocumentVersionModel).where(DocumentVersionModel.document_id == document.id)
        ).all()
        version_ids = [row.id for row in versions]
        manifests = (
            self.session.scalars(
                select(IndexBuildManifestModel).where(
                    IndexBuildManifestModel.document_version_id.in_(version_ids)
                )
            ).all()
            if version_ids
            else []
        )
        object_keys = sorted({key for row in manifests for key in (row.object_keys or []) if key})
        originals = sorted({row.object_key for row in versions if row.object_key})
        kb = control.get_knowledge_base_model(document.kb_id)
        vector_targets = self._vector_cleanup_targets(
            tenant_id=kb.tenant_id,
            kb_id=kb.id,
            document_ids=[document.id],
            document_version_ids=version_ids,
        )
        now = utcnow()
        document.status = "deleted"
        document.deleted_at = now
        document.active_version_id = None
        if version_ids:
            self.session.query(DocumentIndexBindingModel).filter(
                DocumentIndexBindingModel.document_version_id.in_(version_ids),
                DocumentIndexBindingModel.status.in_({"published", "staged"}),
            ).update({"status": "retired", "retired_at": now}, synchronize_session=False)
        cleanup = CleanupJobModel(
            id=new_id("cleanup"),
            target_kind="document",
            target_id=document.id,
            mode=mode,
            status="queued",
            attempts=0,
            plan_json={
                "milvus_filters": vector_targets,
                "object_keys": object_keys,
                "original_paths": originals if mode == "purge" else [],
            },
            result_json={},
        )
        self.session.add(cleanup)
        self.session.flush()
        return {
            "document": ControlRepository.document_dict(document),
            "cleanup_job": self.cleanup_dict(cleanup),
        }

    def soft_delete_knowledge_base(self, kb_id: str, mode: str) -> dict:
        """先在同一 SQL 事务关闭整个 KB 可见性，再生成外部物理清理计划。"""
        if mode not in {"soft", "purge"}:
            raise ResourceConflictError(f"unsupported delete mode: {mode}")
        if kb_id == DEFAULT_KB_ID:
            raise ResourceConflictError("default knowledge base cannot be deleted")
        control = ControlRepository(self.session)
        control.get_knowledge_base_model(kb_id)
        unfinished = self.session.scalar(
            select(IngestionJobModel.id)
            .join(
                DocumentVersionModel,
                DocumentVersionModel.id == IngestionJobModel.document_version_id,
            )
            .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
            .where(
                DocumentModel.kb_id == kb_id,
                IngestionJobModel.status.in_(self.ACTIVE_JOB_STATUSES),
            )
            .limit(1)
        )
        if unfinished is not None:
            raise ResourceConflictError(
                f"knowledge base deletion is blocked by unfinished ingestion job: {unfinished}"
            )

        documents = self.session.scalars(
            select(DocumentModel).where(DocumentModel.kb_id == kb_id)
        ).all()
        document_ids = [row.id for row in documents]
        versions = (
            self.session.scalars(
                select(DocumentVersionModel).where(
                    DocumentVersionModel.document_id.in_(document_ids)
                )
            ).all()
            if document_ids
            else []
        )
        version_ids = [row.id for row in versions]
        indexes = self.session.scalars(
            select(IndexVersionModel).where(IndexVersionModel.kb_id == kb_id)
        ).all()
        index_ids = [row.id for row in indexes]
        manifests = (
            self.session.scalars(
                select(IndexBuildManifestModel).where(
                    IndexBuildManifestModel.index_version_id.in_(index_ids)
                )
            ).all()
            if index_ids
            else []
        )
        object_keys = sorted({key for row in manifests for key in (row.object_keys or []) if key})
        original_paths = sorted({row.object_key for row in versions if row.object_key})
        kb = control.get_knowledge_base_model(kb_id)
        vector_targets = self._vector_cleanup_targets(
            tenant_id=kb.tenant_id,
            kb_id=kb_id,
            document_ids=document_ids,
            document_version_ids=version_ids,
        )
        now = utcnow()
        for document in documents:
            document.status = "deleted"
            document.deleted_at = now
            document.active_version_id = None
        if version_ids:
            self.session.query(DocumentIndexBindingModel).filter(
                DocumentIndexBindingModel.document_version_id.in_(version_ids),
                DocumentIndexBindingModel.status.in_({"published", "staged"}),
            ).update(
                {"status": "retired", "retired_at": now},
                synchronize_session=False,
            )
        kb_result = control.delete_knowledge_base(kb_id)
        cleanup = CleanupJobModel(
            id=new_id("cleanup"),
            target_kind="knowledge_base",
            target_id=kb_id,
            mode=mode,
            status="queued",
            attempts=0,
            plan_json={
                "milvus_filters": vector_targets,
                "object_keys": object_keys,
                "original_paths": original_paths if mode == "purge" else [],
            },
            result_json={},
        )
        self.session.add(cleanup)
        self.session.flush()
        return {"knowledge_base": kb_result, "cleanup_job": self.cleanup_dict(cleanup)}

    def list_cleanup_jobs(
        self, limit: int = 100, kb_id: str | None = None
    ) -> list[dict]:
        if kb_id:
            ControlRepository(self.session).get_knowledge_base_model(
                kb_id, include_deleted=True
            )
        effective_limit = max(1, min(limit, 500))
        # 先按授权/KB 作用域过滤，再应用响应 limit；否则大量其他租户或其他
        # KB 的新任务会把目标任务挤出窗口，形成“软删后清理任务消失”。
        rows = self.session.scalars(
            select(CleanupJobModel).order_by(CleanupJobModel.created_at.desc())
        ).all()
        visible = []
        for row in rows:
            try:
                self._cleanup(row.id)
            except ResourceNotFoundError:
                continue
            scope = self._cleanup_scope(row)
            if kb_id and scope.get("kb_id") != kb_id:
                continue
            visible.append(self.cleanup_dict(row, scope=scope))
            if len(visible) >= effective_limit:
                break
        return visible

    def get_cleanup_job(self, cleanup_job_id: str) -> dict:
        return self.cleanup_dict(self._cleanup(cleanup_job_id))

    def start_cleanup(self, cleanup_job_id: str) -> dict:
        row = self._cleanup(cleanup_job_id)
        if row.status == "succeeded":
            return self.cleanup_dict(row)
        row.status = "running"
        row.attempts += 1
        row.error_detail = None
        self.session.flush()
        return self.cleanup_dict(row)

    def finish_cleanup(self, cleanup_job_id: str, result: dict, errors: list[str]) -> dict:
        row = self._cleanup(cleanup_job_id)
        row.result_json = result
        row.error_detail = "\n".join(errors)[:4000] if errors else None
        row.status = "partial" if errors else "succeeded"
        row.completed_at = None if errors else utcnow()
        self.session.flush()
        return self.cleanup_dict(row)

    def active_scopes(self, kb_id: str | None = None) -> list[dict]:
        query = (
            select(DocumentModel, DocumentIndexBindingModel, IndexVersionModel)
            .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
            .join(DocumentVersionModel, DocumentVersionModel.id == DocumentModel.active_version_id)
            .join(
                DocumentIndexBindingModel,
                DocumentIndexBindingModel.document_version_id == DocumentVersionModel.id,
            )
            .join(
                IndexVersionModel,
                IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
            )
            .where(
                KnowledgeBaseModel.status == "active",
                # 兼容修复前已将“有 active v1、v2 正在处理”的文档写成
                # processing 的在途数据。active_version + published binding 才是
                # 查询可见性的版本事实；disabled/deleted 仍会被排除。
                DocumentModel.status.in_({"ready", "processing"}),
                DocumentIndexBindingModel.status == "published",
                IndexVersionModel.status == "active",
                IndexVersionModel.active_slot == "active",
                # Fail closed if control-plane validation is degraded/corrupted.  An
                # active slot alone is insufficient proof that its frozen read set was
                # completely manifested and atomically published.
                IndexVersionModel.validation_status == "validated",
            )
        )
        if kb_id:
            query = query.where(DocumentModel.kb_id == kb_id)
        principal = current_principal()
        if principal:
            query = query.where(KnowledgeBaseModel.tenant_id == principal.tenant_id)
        rows = self.session.execute(query).all()
        return [
            {
                "tenant_id": self.session.get(KnowledgeBaseModel, document.kb_id).tenant_id,
                "kb_id": document.kb_id,
                "document_id": document.id,
                "document_version_id": binding.document_version_id,
                "index_version_id": binding.index_version_id,
                "collection_name": index.collection_alias,
            }
            for document, binding, index in rows
        ]

    def index_detail(self, row: IndexVersionModel) -> dict:
        result = ControlRepository.index_version_dict(row)
        summary = row.manifest_summary if isinstance(row.manifest_summary, dict) else {}
        result.update(
            {
                "previous_version_id": row.previous_version_id,
                "build_kind": row.build_kind,
                "validation_status": row.validation_status,
                "manifest_summary": summary,
            }
        )
        manifests = self.session.scalars(
            select(IndexBuildManifestModel).where(
                IndexBuildManifestModel.index_version_id == row.id
            )
        ).all()
        result["manifests"] = [self.manifest_dict(item) for item in manifests]

        # 覆盖率是 IndexVersion 自身冻结快照的属性，不能读取当前 KB 的
        # documents.active_version_id 作为历史版本分母。发布 v2 后，v1 的 binding
        # 会转为 retired，但它仍是可回滚的完整 v1 快照，必须计入冻结覆盖。
        binding_rows = self.session.execute(
            select(
                DocumentVersionModel.document_id,
                DocumentIndexBindingModel.document_version_id,
                DocumentIndexBindingModel.status,
            )
            .join(
                DocumentIndexBindingModel,
                DocumentIndexBindingModel.document_version_id == DocumentVersionModel.id,
            )
            .where(DocumentIndexBindingModel.index_version_id == row.id)
        ).all()
        manifest_version_ids = {
            str(item.document_version_id) for item in manifests if item.document_version_id
        }
        manifest_document_by_version = {
            str(version_id): str(document_id)
            for version_id, document_id in self.session.execute(
                select(DocumentVersionModel.id, DocumentVersionModel.document_id).where(
                    DocumentVersionModel.id.in_(manifest_version_ids)
                )
            ).all()
        } if manifest_version_ids else {}

        raw_targets = summary.get("target_document_versions")
        has_frozen_target_map = isinstance(raw_targets, dict)
        frozen_targets = self._target_versions(row) if has_frozen_target_map else {}
        if has_frozen_target_map:
            target_pairs = {
                (str(document_id), str(version_id))
                for document_id, version_id in frozen_targets.items()
            }
            coverage_basis = "target_document_versions"
        elif manifest_document_by_version:
            # 初始 bootstrap 索引早于 target map；其 index-scoped manifest 是
            # 次优但仍冻结于该版本的兼容事实，不能回退到当前 KB 文档集合。
            target_pairs = {
                (manifest_document_by_version[version_id], version_id)
                for version_id in manifest_version_ids
                if version_id in manifest_document_by_version
            }
            coverage_basis = "index_manifests"
        else:
            # 更早的存量/单元测试数据可能没有 manifest，只保留 binding。按该
            # IndexVersion 自身 binding 恢复历史快照，并明确标记为 legacy 口径。
            target_pairs = {
                (str(document_id), str(document_version_id))
                for document_id, document_version_id, status in binding_rows
                if status in {"staged", "published", "retired"}
            }
            coverage_basis = "index_bindings_legacy"

        target_documents = {document_id for document_id, _ in target_pairs}
        if target_pairs or has_frozen_target_map:
            expected_documents = len(target_documents)
        else:
            # 极老 staging 行可能只有冻结数量而没有版本明细；保持只读兼容，且
            # 仍不借用当前 KB 集合扩大分母。
            expected_documents = max(int(summary.get("expected_documents") or 0), 0)
            coverage_basis = "manifest_summary_legacy"

        bound_documents = {
            str(document_id)
            for document_id, document_version_id, status in binding_rows
            if status in {"staged", "published", "retired"}
            and (
                not target_pairs
                or (str(document_id), str(document_version_id)) in target_pairs
            )
        }
        validated_documents_set = {
            manifest_document_by_version[str(item.document_version_id)]
            for item in manifests
            if item.validation_status == "validated"
            and str(item.document_version_id) in manifest_document_by_version
            and (
                not target_pairs
                or (
                    manifest_document_by_version[str(item.document_version_id)],
                    str(item.document_version_id),
                ) in target_pairs
            )
        }
        bound_count = min(len(bound_documents), expected_documents)
        validated_count = min(len(validated_documents_set), expected_documents)
        result["document_coverage"] = {
            "expected_documents": expected_documents,
            # covered_documents 是既有 API 字段，保留为 bound_documents 的兼容别名。
            "covered_documents": bound_count,
            "bound_documents": bound_count,
            "validated_documents": validated_count,
            "coverage_basis": coverage_basis,
        }
        return result

    @staticmethod
    def manifest_dict(row: IndexBuildManifestModel) -> dict:
        return {
            "id": row.id,
            "job_id": row.job_id,
            "document_version_id": row.document_version_id,
            "index_version_id": row.index_version_id,
            "expected_chunk_count": row.expected_chunk_count,
            "written_chunk_count": row.written_chunk_count,
            "searchable_chunk_count": row.searchable_chunk_count,
            "chunk_ids_sha256": row.chunk_ids_sha256,
            "chunk_ids": row.chunk_ids,
            "object_keys": row.object_keys,
            "vector_dim": row.vector_dim,
            "collection_name": row.collection_name,
            "validation_status": row.validation_status,
            "validation_errors": row.validation_errors,
            "validated_at": row.validated_at.isoformat() if row.validated_at else None,
        }

    def _cleanup_scope(self, row: CleanupJobModel) -> dict:
        """返回清理任务的稳定租户/KB 归属，即使资源已经软删除。"""

        kb: KnowledgeBaseModel | None = None
        document: DocumentModel | None = None
        index: IndexVersionModel | None = None
        if row.target_kind == "knowledge_base":
            kb = self.session.get(KnowledgeBaseModel, row.target_id)
        elif row.target_kind == "document":
            document = self.session.get(DocumentModel, row.target_id)
            kb = (
                self.session.get(KnowledgeBaseModel, document.kb_id)
                if document is not None
                else None
            )
        elif row.target_kind == "index_version":
            index = self.session.get(IndexVersionModel, row.target_id)
            kb = (
                self.session.get(KnowledgeBaseModel, index.kb_id)
                if index is not None
                else None
            )
        return {
            "tenant_id": kb.tenant_id if kb is not None else None,
            "kb_id": kb.id if kb is not None else None,
            "kb_name": kb.name if kb is not None else None,
            "document_id": document.id if document is not None else None,
            "document_name": document.logical_name if document is not None else None,
            "index_version_id": index.id if index is not None else None,
            "display_name": (
                document.logical_name
                if document is not None
                else kb.name
                if kb is not None
                else row.target_id
            ),
        }

    def cleanup_dict(
        self, row: CleanupJobModel, *, scope: dict | None = None
    ) -> dict:
        return {
            "id": row.id,
            "target_kind": row.target_kind,
            "target_id": row.target_id,
            "scope": scope or self._cleanup_scope(row),
            "mode": row.mode,
            "status": row.status,
            "attempts": row.attempts,
            "plan": row.plan_json or {},
            "result": row.result_json or {},
            "error_detail": row.error_detail,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
