"""TODO-01 控制面 repository；API 不直接操作 ORM。"""

from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
import json
import re
import unicodedata

from sqlalchemy import and_, func, null, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.evidence import CitationFact, Evidence
from app.domain.document import (
    ensure_document_transition,
    ensure_document_version_transition,
)
from app.domain.ingestion_job import ensure_ingestion_job_transition
from app.domain.knowledge_base import KnowledgeBaseStatus, ensure_knowledge_base_transition
from app.infra.persistence.sql.models import (
    CitationFactModel,
    DocumentChunkingRunModel,
    DocumentIndexBindingModel,
    DocumentModel,
    DocumentParseArtifactModel,
    DocumentVersionModel,
    EvidenceRecordModel,
    IndexVersionModel,
    IngestionJobModel,
    IngestionJobEventModel,
    IngestionIdempotencyKeyModel,
    KnowledgeBaseModel,
    TenantModel,
    WorkspaceModel,
)
from app.infra.persistence.sql.primitives import (
    DEFAULT_KB_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_WORKSPACE_ID,
    ResourceConflictError,
    ResourceNotFoundError,
    new_id,
)
from app.shared.config.settings_config import settings
from app.security.principal import current_principal


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def canonical_json_sha256(value: dict | list | None) -> str:
    """返回与键顺序、空白和 Unicode 转义方式无关的规范 JSON SHA-256。"""
    payload = json.dumps(
        value if value is not None else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


_DOTTED_TERMINAL_HEADING = re.compile(r"^\s*(\d+(?:\s*\.\s*\d+)+)")
MAX_EXACT_HEADING_SCAN_ROWS = 10_000


def _terminal_heading_path(value: object) -> str | None:
    """读取标题最前端的完整 dotted path；子章节不会折叠到父章节。"""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    matched = _DOTTED_TERMINAL_HEADING.match(normalized)
    if matched is None:
        return None
    return ".".join(part.strip() for part in matched.group(1).split("."))


class ControlRepository:
    """单个 SQLAlchemy Session 上的领域仓储。"""

    def __init__(self, session: Session, tenant_id: str | None = None):
        self.session = session
        self._tenant_id_override = tenant_id

    @property
    def tenant_id(self) -> str | None:
        principal = current_principal()
        return principal.tenant_id if principal else self._tenant_id_override

    @property
    def actor_id(self) -> str:
        principal = current_principal()
        return principal.user_id if principal else "system"

    def _workspace_for_tenant(self, tenant_id: str) -> WorkspaceModel:
        workspace = self.session.scalar(
            select(WorkspaceModel)
            .where(WorkspaceModel.tenant_id == tenant_id)
            .order_by(WorkspaceModel.created_at)
        )
        if workspace is None:
            workspace = WorkspaceModel(
                id=new_id("ws"), tenant_id=tenant_id, name="Default Workspace", status="active"
            )
            self.session.add(workspace)
            self.session.flush()
        return workspace

    def ensure_defaults(self) -> KnowledgeBaseModel:
        tenant = self.session.get(TenantModel, DEFAULT_TENANT_ID)
        if tenant is None:
            tenant = TenantModel(id=DEFAULT_TENANT_ID, name="Default Tenant", status="active")
            self.session.add(tenant)
            self.session.flush()
        workspace = self.session.get(WorkspaceModel, DEFAULT_WORKSPACE_ID)
        if workspace is None:
            workspace = WorkspaceModel(
                id=DEFAULT_WORKSPACE_ID,
                tenant_id=DEFAULT_TENANT_ID,
                name="Default Workspace",
                status="active",
            )
            self.session.add(workspace)
            self.session.flush()
        kb = self.session.get(KnowledgeBaseModel, DEFAULT_KB_ID)
        if kb is None:
            kb = KnowledgeBaseModel(
                id=DEFAULT_KB_ID,
                tenant_id=DEFAULT_TENANT_ID,
                workspace_id=DEFAULT_WORKSPACE_ID,
                name="默认知识库",
                description="旧 /upload 接口和本地验证使用的默认知识库",
                status=KnowledgeBaseStatus.ACTIVE.value,
                created_by="system",
            )
            self.session.add(kb)
            self.session.flush()
            self._create_initial_index(kb.id)
        self.session.flush()
        return kb

    def _create_initial_index(self, kb_id: str) -> IndexVersionModel:
        knowledge_base = self.session.get(KnowledgeBaseModel, kb_id)
        if knowledge_base is None:
            raise ResourceNotFoundError(f"knowledge base not found: {kb_id}")
        index = IndexVersionModel(
            id=new_id("idx"),
            kb_id=kb_id,
            version_no=1,
            embedding_model_id="BAAI/bge-m3",
            vector_dim=1024,
            schema_version="stt-chunks-v2",
            status="active",
            active_slot="active",
            # 物理 alias 由服务端 placement policy 在同一事务内覆盖。
            collection_alias="pending_managed_placement",
            vector_governance_mode="managed",
            activated_at=utcnow(),
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
            readable=True,
        )
        return index

    def get_knowledge_base_model(
        self, kb_id: str, *, include_deleted: bool = False
    ) -> KnowledgeBaseModel:
        query = select(KnowledgeBaseModel).where(KnowledgeBaseModel.id == kb_id)
        if self.tenant_id:
            query = query.where(KnowledgeBaseModel.tenant_id == self.tenant_id)
        kb = self.session.scalar(query)
        if kb is None or (not include_deleted and kb.status == "deleted"):
            raise ResourceNotFoundError(f"knowledge base not found: {kb_id}")
        return kb

    def create_knowledge_base(
        self,
        name: str,
        description: str,
        created_by: str = "system",
        data_classification: str = "internal",
        external_llm_allowed: bool = True,
        web_search_allowed: bool = False,
    ) -> dict:
        self.ensure_defaults()
        tenant_id = self.tenant_id or DEFAULT_TENANT_ID
        workspace = self._workspace_for_tenant(tenant_id)
        kb = KnowledgeBaseModel(
            id=new_id("kb"),
            tenant_id=tenant_id,
            workspace_id=workspace.id,
            name=name.strip(),
            description=description.strip(),
            status="active",
            created_by=self.actor_id if created_by == "system" else created_by,
            data_classification=data_classification,
            external_llm_allowed=external_llm_allowed,
            web_search_allowed=web_search_allowed,
        )
        self.session.add(kb)
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise ResourceConflictError(f"knowledge base name already exists: {name}") from exc
        self._create_initial_index(kb.id)
        return self.knowledge_base_dict(kb)

    def list_knowledge_bases(self) -> list[dict]:
        self.ensure_defaults()
        query = select(KnowledgeBaseModel).where(KnowledgeBaseModel.status != "deleted")
        if self.tenant_id:
            query = query.where(KnowledgeBaseModel.tenant_id == self.tenant_id)
        rows = self.session.scalars(query.order_by(KnowledgeBaseModel.created_at.asc())).all()
        return [self.knowledge_base_dict(row) for row in rows]

    def update_knowledge_base(
        self,
        kb_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        status: str | None = None,
        data_classification: str | None = None,
        external_llm_allowed: bool | None = None,
        web_search_allowed: bool | None = None,
    ) -> dict:
        kb = self.get_knowledge_base_model(kb_id)
        if name is not None:
            kb.name = name.strip()
        if description is not None:
            kb.description = description.strip()
        if status is not None:
            ensure_knowledge_base_transition(kb.status, status)
            kb.status = status
        if data_classification is not None:
            kb.data_classification = data_classification
        if external_llm_allowed is not None:
            kb.external_llm_allowed = external_llm_allowed
        if web_search_allowed is not None:
            kb.web_search_allowed = web_search_allowed
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise ResourceConflictError(f"knowledge base name already exists: {name}") from exc
        return self.knowledge_base_dict(kb)

    def delete_knowledge_base(self, kb_id: str) -> dict:
        if kb_id == DEFAULT_KB_ID:
            raise ResourceConflictError("default knowledge base cannot be deleted")
        kb = self.get_knowledge_base_model(kb_id)
        ensure_knowledge_base_transition(kb.status, "deleted")
        kb.status = "deleted"
        kb.deleted_at = utcnow()
        self.session.flush()
        return self.knowledge_base_dict(kb)

    def get_active_index(self, kb_id: str) -> IndexVersionModel:
        self.get_knowledge_base_model(kb_id)
        row = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.kb_id == kb_id,
                IndexVersionModel.active_slot == "active",
                IndexVersionModel.status == "active",
            )
        )
        if row is None:
            raise ResourceConflictError(f"knowledge base has no active index: {kb_id}")
        return row

    def list_index_versions(self, kb_id: str) -> list[dict]:
        self.get_knowledge_base_model(kb_id)
        rows = self.session.scalars(
            select(IndexVersionModel)
            .where(IndexVersionModel.kb_id == kb_id)
            .order_by(IndexVersionModel.version_no.desc())
        ).all()
        return [self.index_version_dict(row) for row in rows]

    def register_upload_and_job(
        self,
        *,
        kb_id: str,
        logical_name: str,
        content_sha256: str,
        size_bytes: int,
        mime_type: str,
        object_key: str,
        parser_config_snapshot: dict | None,
        source_uri: str | None = None,
        document_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> dict:
        kb = self.get_knowledge_base_model(kb_id)
        if kb.status != "active":
            raise ResourceConflictError(f"knowledge base is not active: {kb_id}")

        if idempotency_key:
            key_binding = self.session.get(IngestionIdempotencyKeyModel, idempotency_key)
            existing_job = None
            existing_fingerprint = None
            if key_binding is not None:
                existing_job = self.session.get(IngestionJobModel, key_binding.job_id)
                existing_fingerprint = key_binding.request_fingerprint
            else:
                # 兼容迁移早期已经直接写在 job 行上的键。
                existing_job = self.session.scalar(
                    select(IngestionJobModel).where(
                        IngestionJobModel.idempotency_key == idempotency_key
                    )
                )
                existing_fingerprint = existing_job.request_fingerprint if existing_job else None
            if existing_job is not None:
                if existing_fingerprint != request_fingerprint:
                    raise ResourceConflictError(
                        "Idempotency-Key was already used with a different upload request"
                    )
                existing_version = self.get_document_version_model(existing_job.document_version_id)
                existing_document = self.get_document_model(existing_version.document_id)
                existing_index = self.session.get(IndexVersionModel, existing_job.index_version_id)
                assert existing_index is not None
                return {
                    "document": self.document_dict(existing_document),
                    "document_version": self.document_version_dict(existing_version),
                    "ingestion_job": self.ingestion_job_dict(existing_job),
                    "index_version": self.index_version_dict(existing_index),
                    "deduplicated": True,
                    "created_document": False,
                    "should_process": False,
                    "reused_job": True,
                    "idempotency_replayed": True,
                }

        document = None
        if document_id:
            document = self.get_document_model(document_id)
            if document.kb_id != kb_id:
                raise ResourceConflictError("document does not belong to knowledge base")
        else:
            document = self.session.scalar(
                select(DocumentModel).where(
                    DocumentModel.kb_id == kb_id,
                    DocumentModel.logical_name == logical_name,
                )
            )
        created_document = False
        if document is None:
            document = DocumentModel(
                id=document_id or new_id("doc"),
                kb_id=kb_id,
                logical_name=logical_name,
                source_type="upload",
                source_uri=source_uri,
                status="draft",
            )
            self.session.add(document)
            self.session.flush()
            created_document = True
        elif document.status == "deleted":
            raise ResourceConflictError(f"document was deleted: {logical_name}")

        config_snapshot = parser_config_snapshot or {}
        pipeline_fingerprint = canonical_json_sha256(config_snapshot)
        version_query = select(DocumentVersionModel).where(
            DocumentVersionModel.document_id == document.id,
            DocumentVersionModel.content_sha256 == content_sha256,
        )
        if parser_config_snapshot is not None:
            # 显式配置（包括显式空对象）按完整 pipeline 指纹区分版本。
            version_query = version_query.where(
                DocumentVersionModel.pipeline_fingerprint == pipeline_fingerprint
            )
        else:
            # 兼容旧应用调用：未声明 pipeline 表示复用相同内容的最近版本，
            # 而不是把“未声明”误解释成一套新的空配置 pipeline。
            version_query = version_query.order_by(DocumentVersionModel.version_no.desc())
        version = self.session.scalar(version_query.limit(1))
        deduplicated = version is not None
        if version is None:
            latest_no = (
                self.session.scalar(
                    select(func.max(DocumentVersionModel.version_no)).where(
                        DocumentVersionModel.document_id == document.id
                    )
                )
                or 0
            )
            version = DocumentVersionModel(
                id=new_id("dv"),
                document_id=document.id,
                version_no=latest_no + 1,
                content_sha256=content_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                size_bytes=size_bytes,
                mime_type=mime_type,
                object_key=object_key,
                parser_config_snapshot=config_snapshot,
                status="uploaded",
            )
            self.session.add(version)
            self.session.flush()

        active_index = self.get_active_index(kb_id)
        if deduplicated:
            reusable_job = self.session.scalar(
                select(IngestionJobModel)
                .where(
                    IngestionJobModel.document_version_id == version.id,
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
            if reusable_job is not None:
                reusable_index = self.session.get(
                    IndexVersionModel, reusable_job.index_version_id
                )
                if reusable_index is None:
                    raise ResourceConflictError(
                        f"ingestion job target index is missing: {reusable_job.id}"
                    )
                self._bind_idempotency_key(idempotency_key, request_fingerprint, reusable_job)
                return {
                    "document": self.document_dict(document),
                    "document_version": self.document_version_dict(version),
                    "ingestion_job": self.ingestion_job_dict(reusable_job),
                    "index_version": self.index_version_dict(reusable_index),
                    "deduplicated": True,
                    "created_document": created_document,
                    # 原请求负责投递；重复上传只复用资源，避免重复 BackgroundTask。
                    "should_process": False,
                    "reused_job": True,
                    "idempotency_replayed": False,
                }
        # An index that already exposes any published binding is an immutable read set.
        # A subsequent document import must target a copy-on-write staging index; otherwise
        # its manifest/binding can mutate an already published version in place.
        has_published_binding = self.session.scalar(
            select(DocumentIndexBindingModel.document_version_id)
            .where(
                DocumentIndexBindingModel.index_version_id == active_index.id,
                DocumentIndexBindingModel.status == "published",
            )
            .limit(1)
        )
        existing_staging = self.session.scalar(
            select(IndexVersionModel.id)
            .where(
                IndexVersionModel.kb_id == kb_id,
                IndexVersionModel.status.in_({"creating", "staging"}),
            )
            .limit(1)
        )
        if has_published_binding is not None or existing_staging is not None:
            from app.infra.persistence.sql.document_update_provisioning import (
                prepare_document_update,
            )

            index = prepare_document_update(
                self.session,
                control=self,
                kb_id=kb_id,
                document_id=document.id,
                document_version_id=version.id,
            )
        else:
            # The bootstrap index has no published data yet and may accept its first import.
            index = active_index

        latest_attempt = (
            self.session.scalar(
                select(func.max(IngestionJobModel.attempt)).where(
                    IngestionJobModel.document_version_id == version.id
                )
            )
            or 0
        )
        job = IngestionJobModel(
            id=new_id("job"),
            config_version=settings.config_version,
            document_version_id=version.id,
            index_version_id=index.id,
            status="queued",
            stage="queued",
            progress=0.0,
            attempt=latest_attempt + 1,
            max_attempts=settings.ingestion_max_attempts,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
        self.session.add(job)
        if index.status == "active":
            # The initial empty index is the sole compatibility exception to COW.  Freeze
            # every job admitted before its first binding is published so a concurrent
            # multi-file bootstrap remains valid, while later jobs cannot claim that
            # exception after the read set has gone live.
            summary = dict(index.manifest_summary or {})
            bootstrap_job_ids = {
                str(value) for value in summary.get("bootstrap_job_ids", []) if str(value)
            }
            bootstrap_job_ids.add(job.id)
            index.manifest_summary = {
                **summary,
                "bootstrap_job_ids": sorted(bootstrap_job_ids),
            }
        # Document.status 表达当前线上版本的可用性；新版本的处理中状态由
        # DocumentVersion/IngestionJob 承担。已经有 active_version 的文档在 v2
        # staging 期间必须继续保持 ready，否则 SQL active scope 会把仍 published
        # 的 v1 提前下线，破坏 TODO-03 的原子发布语义。
        if document.active_version_id is None:
            if document.status in {"ready", "failed"}:
                ensure_document_transition(document.status, "processing")
            elif document.status == "draft":
                ensure_document_transition("draft", "processing")
            document.status = "processing"
        self.session.flush()
        self._bind_idempotency_key(idempotency_key, request_fingerprint, job)
        self._append_initial_job_event(job)
        return {
            "document": self.document_dict(document),
            "document_version": self.document_version_dict(version),
            "ingestion_job": self.ingestion_job_dict(job),
            "index_version": self.index_version_dict(index),
            "deduplicated": deduplicated,
            "created_document": created_document,
            "should_process": True,
            "reused_job": False,
            "idempotency_replayed": False,
        }

    def create_job_for_version(self, version_id: str) -> dict:
        version = self.get_document_version_model(version_id)
        document = self.get_document_model(version.document_id)
        if version.status == "published":
            raise ResourceConflictError(
                "published document version is immutable; upload a new version instead"
            )
        if version.status not in {"uploaded", "failed"}:
            raise ResourceConflictError(
                f"document version cannot be queued from status: {version.status}"
            )
        active_job = self.session.scalar(
            select(IngestionJobModel)
            .where(
                IngestionJobModel.document_version_id == version_id,
                IngestionJobModel.status.in_(
                    {"queued", "leased", "running", "retry_wait", "processing"}
                ),
            )
            .order_by(IngestionJobModel.attempt.desc())
            .limit(1)
        )
        if active_job is not None:
            raise ResourceConflictError(
                f"document version already has an active ingestion job: {active_job.id}"
            )
        index = self.get_active_index(document.kb_id)
        latest_attempt = (
            self.session.scalar(
                select(func.max(IngestionJobModel.attempt)).where(
                    IngestionJobModel.document_version_id == version_id
                )
            )
            or 0
        )
        job = IngestionJobModel(
            id=new_id("job"),
            config_version=settings.config_version,
            document_version_id=version_id,
            index_version_id=index.id,
            status="queued",
            stage="queued",
            progress=0,
            attempt=latest_attempt + 1,
            max_attempts=settings.ingestion_max_attempts,
        )
        self.session.add(job)
        if document.active_version_id is None and document.status != "processing":
            ensure_document_transition(document.status, "processing")
            document.status = "processing"
        self.session.flush()
        self._append_initial_job_event(job)
        return self.ingestion_job_detail(job)

    def _append_initial_job_event(self, job: IngestionJobModel) -> None:
        if job.last_event_sequence:
            return
        job.last_event_sequence = 1
        self.session.add(
            IngestionJobEventModel(
                job_id=job.id,
                sequence=1,
                event_type="queued",
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                payload={"submission_attempt": job.attempt, "config_version": job.config_version},
            )
        )
        self.session.flush()

    def _bind_idempotency_key(
        self,
        idempotency_key: str | None,
        request_fingerprint: str | None,
        job: IngestionJobModel,
    ) -> None:
        if not idempotency_key or not request_fingerprint:
            return
        existing = self.session.get(IngestionIdempotencyKeyModel, idempotency_key)
        if existing is not None:
            if existing.request_fingerprint != request_fingerprint or existing.job_id != job.id:
                raise ResourceConflictError(
                    "Idempotency-Key was already used with a different upload request"
                )
            return
        self.session.add(
            IngestionIdempotencyKeyModel(
                key=idempotency_key,
                request_fingerprint=request_fingerprint,
                job_id=job.id,
            )
        )
        self.session.flush()

    def get_document_model(
        self, document_id: str, *, include_deleted: bool = False
    ) -> DocumentModel:
        row = self.session.get(DocumentModel, document_id)
        if row is None or (not include_deleted and row.status == "deleted"):
            raise ResourceNotFoundError(f"document not found: {document_id}")
        self.get_knowledge_base_model(row.kb_id, include_deleted=include_deleted)
        return row

    def get_document_version_model(self, version_id: str) -> DocumentVersionModel:
        row = self.session.get(DocumentVersionModel, version_id)
        if row is None:
            raise ResourceNotFoundError(f"document version not found: {version_id}")
        self.get_document_model(row.document_id, include_deleted=True)
        return row

    def get_parse_artifact_model(self, artifact_id: str) -> DocumentParseArtifactModel:
        row = self.session.get(DocumentParseArtifactModel, artifact_id)
        if row is None:
            raise ResourceNotFoundError(f"document parse artifact not found: {artifact_id}")
        # 复用 document -> knowledge base 的租户过滤；跨租户资源表现为 not found。
        self.get_document_version_model(row.document_version_id)
        return row

    def get_chunking_run_model(self, run_id: str) -> DocumentChunkingRunModel:
        row = self.session.get(DocumentChunkingRunModel, run_id)
        if row is None:
            raise ResourceNotFoundError(f"document chunking run not found: {run_id}")
        self.get_parse_artifact_model(row.parse_artifact_id)
        return row

    def get_evidence_model(self, evidence_id: str) -> EvidenceRecordModel:
        row = self.session.get(EvidenceRecordModel, evidence_id)
        if row is None or (self.tenant_id is not None and row.tenant_id != self.tenant_id):
            raise ResourceNotFoundError(f"evidence not found: {evidence_id}")
        # 本地证据通过 KB 再做一次租户收敛；Web 快照由显式 tenant_id 隔离。
        if row.source_type == "local":
            assert row.kb_id is not None
            self.get_knowledge_base_model(row.kb_id, include_deleted=True)
        return row

    def get_evidence_models(self, evidence_ids: Iterable[str]) -> list[EvidenceRecordModel]:
        """批量读取当前租户 Evidence，避免一次召回产生逐 ID SQL 查询。"""

        unique_ids = list(dict.fromkeys(str(value) for value in evidence_ids if value))
        if not unique_ids:
            return []
        query = select(EvidenceRecordModel).where(EvidenceRecordModel.id.in_(unique_ids))
        if self.tenant_id is not None:
            query = query.where(EvidenceRecordModel.tenant_id == self.tenant_id)
        rows = self.session.scalars(query).all()
        return list(rows)

    def find_exact_terminal_heading_evidence(
        self,
        *,
        chapter_paths: Iterable[str],
        scope_identities: Iterable[tuple[str, str, str, str, str]],
        document_ids: Iterable[str] = (),
        scan_limit: int = MAX_EXACT_HEADING_SCAN_ROWS,
    ) -> tuple[list[EvidenceRecordModel], int, bool]:
        """在冻结的本地索引身份内精确匹配 terminal heading 编号。"""

        wanted_paths = tuple(
            dict.fromkeys(str(value).strip() for value in chapter_paths if str(value).strip())
        )
        if not wanted_paths:
            return [], 0, False
        if any(re.fullmatch(r"\d+(?:\.\d+)+", value) is None for value in wanted_paths):
            raise ValueError("chapter paths must be normalized dotted paths")
        if not 1 <= scan_limit <= MAX_EXACT_HEADING_SCAN_ROWS:
            raise ValueError("exact heading scan limit is outside the supported range")

        identities = tuple(dict.fromkeys(scope_identities))
        if not identities:
            return [], 0, False
        if self.tenant_id is not None and any(
            tenant_id != self.tenant_id for tenant_id, *_ in identities
        ):
            raise ResourceConflictError("frozen heading scope tenant mismatch")
        clauses = [
            and_(
                EvidenceRecordModel.tenant_id == tenant_id,
                EvidenceRecordModel.kb_id == kb_id,
                EvidenceRecordModel.document_id == document_id,
                EvidenceRecordModel.document_version_id == document_version_id,
                EvidenceRecordModel.index_version_id == index_version_id,
            )
            for tenant_id, kb_id, document_id, document_version_id, index_version_id in identities
        ]
        query = select(EvidenceRecordModel).where(
            EvidenceRecordModel.source_type == "local",
            or_(*clauses),
        )
        allowed_documents = tuple(
            dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip())
        )
        if allowed_documents:
            query = query.where(EvidenceRecordModel.document_id.in_(allowed_documents))
        rows = list(
            self.session.scalars(
                query.order_by(EvidenceRecordModel.document_id, EvidenceRecordModel.id).limit(
                    scan_limit + 1
                )
            ).all()
        )
        if len(rows) > scan_limit:
            return [], scan_limit, True
        wanted = set(wanted_paths)
        matches = [
            row
            for row in rows
            if row.heading_path_json
            and _terminal_heading_path(row.heading_path_json[-1]) in wanted
        ]
        return matches, len(rows), False

    def get_citation_fact_model(self, citation_id: str) -> CitationFactModel:
        row = self.session.get(CitationFactModel, citation_id)
        if row is None:
            raise ResourceNotFoundError(f"citation fact not found: {citation_id}")
        self.get_evidence_model(row.evidence_id)
        return row

    def _validate_evidence_source(self, evidence: Evidence) -> None:
        source = evidence.source
        assert source.tenant_id is not None
        if self.tenant_id is not None and source.tenant_id != self.tenant_id:
            raise ResourceConflictError("evidence tenant does not match current tenant")
        tenant = self.session.get(TenantModel, source.tenant_id)
        if tenant is None:
            raise ResourceConflictError("evidence tenant does not exist")
        if source.source_type == "web":
            return
        # EvidenceSourceIdentity 已保证 local 字段均非空；assert 同时收窄静态类型。
        assert source.kb_id is not None
        assert source.document_id is not None
        assert source.document_version_id is not None
        assert source.index_version_id is not None
        assert source.chunk_id is not None

        kb = self.get_knowledge_base_model(source.kb_id, include_deleted=True)
        if kb.tenant_id != source.tenant_id:
            raise ResourceConflictError("evidence tenant does not own knowledge base")
        document = self.get_document_model(source.document_id, include_deleted=True)
        if document.kb_id != source.kb_id:
            raise ResourceConflictError("evidence document does not belong to knowledge base")
        version = self.get_document_version_model(source.document_version_id)
        if version.document_id != source.document_id:
            raise ResourceConflictError("evidence version does not belong to document")
        index = self.session.get(IndexVersionModel, source.index_version_id)
        if index is None or index.kb_id != source.kb_id:
            raise ResourceConflictError("evidence index does not belong to knowledge base")
        binding = self.session.get(
            DocumentIndexBindingModel,
            (source.document_version_id, source.index_version_id),
        )
        # 当前发布协议在 complete_job 时才创建 binding。索引节点必须先保存 Evidence，
        # 因而在 binding 尚未出现的短暂窗口，只接受与同一 version/index 严格匹配的
        # 非终态导入任务；任意孤立 version/index 组合仍然 fail closed。
        staging_job = None
        if binding is None:
            staging_job = self.session.scalar(
                select(IngestionJobModel).where(
                    IngestionJobModel.document_version_id == source.document_version_id,
                    IngestionJobModel.index_version_id == source.index_version_id,
                    IngestionJobModel.status.in_(
                        {"queued", "leased", "running", "retry_wait", "processing"}
                    ),
                )
            )
        if (binding is not None and binding.status not in {"staged", "published", "retired"}) or (
            binding is None and staging_job is None
        ):
            raise ResourceConflictError(
                "evidence requires a matching active ingestion job or retained binding"
            )

    def record_evidence(self, evidence: Evidence | dict) -> dict:
        """幂等保存不可变 Evidence；retrieval score/rank 不进入控制库。"""

        value = Evidence.model_validate(evidence).persistent_copy()
        self._validate_evidence_source(value)
        source = value.source
        assert value.evidence_id is not None
        assert value.content_hash is not None
        if source.source_type == "local":
            assert source.tenant_id is not None
            assert source.kb_id is not None
            assert source.document_id is not None
            assert source.document_version_id is not None
            assert source.index_version_id is not None
            assert source.chunk_id is not None
            natural_row = self.session.scalar(
                select(EvidenceRecordModel).where(
                    EvidenceRecordModel.document_version_id == source.document_version_id,
                    EvidenceRecordModel.index_version_id == source.index_version_id,
                    EvidenceRecordModel.chunk_id == source.chunk_id,
                )
            )
        else:
            natural_row = self.session.scalar(
                select(EvidenceRecordModel).where(
                    EvidenceRecordModel.source_type == "web",
                    EvidenceRecordModel.tenant_id == source.tenant_id,
                    EvidenceRecordModel.source_url == source.source_url,
                    EvidenceRecordModel.content_hash == source.content_hash,
                )
            )
        id_row = self.session.get(EvidenceRecordModel, value.evidence_id)
        row = id_row or natural_row
        if id_row is not None and natural_row is not None and id_row.id != natural_row.id:
            raise ResourceConflictError("evidence identity conflicts with existing chunk fact")
        if row is not None:
            persisted = self._evidence_domain(row)
            expected = value
            if row.source_type == "web":
                # fetched_at 是首次固化该 URL+正文快照的观察时间，不参与 Evidence ID；
                # 相同租户/URL/hash 的后续命中复用首次快照，不因新观察时刻制造冲突。
                expected = value.model_copy(
                    update={
                        "source": value.source.model_copy(
                            update={"fetched_at": persisted.source.fetched_at}
                        )
                    }
                )
            if persisted.model_dump(mode="json") != expected.model_dump(mode="json"):
                raise ResourceConflictError("evidence is immutable for the same source identity")
            return self.evidence_dict(row)

        primary = value.primary_location_span()
        row = EvidenceRecordModel(
            id=value.evidence_id,
            schema_version=value.schema_version,
            source_type=source.source_type,
            tenant_id=source.tenant_id,
            kb_id=source.kb_id,
            document_id=source.document_id,
            document_version_id=source.document_version_id,
            index_version_id=source.index_version_id,
            chunk_id=source.chunk_id,
            source_url=source.source_url,
            source_fetched_at=source.fetched_at,
            content=value.content,
            content_hash=value.content_hash,
            source_block_ids_json=list(source.source_block_ids),
            provenance_spans_json=[
                span.model_dump(mode="json", exclude_none=True) for span in value.provenance_spans
            ],
            heading_path_json=list(value.heading_path),
            asset_refs_json=list(value.asset_refs),
            source_label=value.source_label,
            primary_block_id=primary.block_id,
            primary_availability=primary.availability,
            primary_coordinate_space=primary.coordinate_space,
            primary_page_no=primary.page_no,
            # SQL CHECK 使用 ``IS NULL`` 判断坐标缺失；JSON(None) 会编码成
            # JSON 文本 ``null``，因此必须显式写 SQL NULL。
            primary_bbox_json=list(primary.bbox) if primary.bbox else null(),
            primary_slide_no=primary.slide_no,
            primary_sheet=primary.sheet,
            primary_cell_range=primary.cell_range,
            primary_line_start=primary.line_start,
            primary_line_end=primary.line_end,
            primary_reason_code=primary.reason_code,
        )
        self.session.add(row)
        self.session.flush()
        return self.evidence_dict(row)

    def record_evidence_batch(self, evidences: Iterable[Evidence | dict]) -> list[dict]:
        """在调用方的同一个 Session/事务中幂等保存整批 chunks。"""

        return [self.record_evidence(evidence) for evidence in evidences]

    def record_citation_bundle(
        self,
        evidences: Iterable[Evidence | dict],
        citations: Iterable[CitationFact | dict],
    ) -> dict[str, list[dict]]:
        """原子保存一次回答引用到的 Evidence 与 CitationFact。

        本方法不创建嵌套事务，也不吞掉异常。每个单条方法仍负责领域校验、
        来源授权和不可变幂等检查；调用方只需在一个 ``session_scope`` 中调用
        本方法，即可保证任一 Evidence/Citation 失败时由外层回滚整批。
        """

        saved_evidences = self.record_evidence_batch(evidences)
        saved_citations = [self.record_citation_fact(citation) for citation in citations]
        return {
            "evidences": saved_evidences,
            "citations": saved_citations,
        }

    def find_evidence(
        self,
        *,
        document_version_id: str,
        index_version_id: str,
        chunk_id: str,
    ) -> dict | None:
        self.get_document_version_model(document_version_id)
        row = self.session.scalar(
            select(EvidenceRecordModel).where(
                EvidenceRecordModel.document_version_id == document_version_id,
                EvidenceRecordModel.index_version_id == index_version_id,
                EvidenceRecordModel.chunk_id == chunk_id,
            )
        )
        return self.evidence_dict(row) if row else None

    def list_evidence(
        self,
        document_version_id: str,
        *,
        index_version_id: str | None = None,
    ) -> list[dict]:
        self.get_document_version_model(document_version_id)
        query = select(EvidenceRecordModel).where(
            EvidenceRecordModel.document_version_id == document_version_id
        )
        if index_version_id is not None:
            query = query.where(EvidenceRecordModel.index_version_id == index_version_id)
        rows = self.session.scalars(
            query.order_by(EvidenceRecordModel.created_at.asc(), EvidenceRecordModel.id.asc())
        ).all()
        return [self.evidence_dict(row) for row in rows]

    def record_citation_fact(self, citation: CitationFact | dict) -> dict:
        """幂等保存服务端校验结果；相同自然键的事实不可被覆盖。"""

        value = CitationFact.model_validate(citation)
        assert value.citation_id is not None
        evidence_row = self.get_evidence_model(value.evidence_id)
        if evidence_row.source_type == "local":
            active_binding = self.session.scalar(
                select(DocumentIndexBindingModel)
                .join(
                    DocumentVersionModel,
                    DocumentVersionModel.id == DocumentIndexBindingModel.document_version_id,
                )
                .join(DocumentModel, DocumentModel.id == DocumentVersionModel.document_id)
                .join(KnowledgeBaseModel, KnowledgeBaseModel.id == DocumentModel.kb_id)
                .join(
                    IndexVersionModel,
                    IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
                )
                .where(
                    DocumentIndexBindingModel.document_version_id
                    == evidence_row.document_version_id,
                    DocumentIndexBindingModel.index_version_id == evidence_row.index_version_id,
                    DocumentIndexBindingModel.status == "published",
                    DocumentModel.active_version_id == evidence_row.document_version_id,
                    # 与 active_scopes 保持同一可见性语义；processing 只用于兼容
                    # 修复前的在途 v2，active_version 仍精确锁定 published v1。
                    DocumentModel.status.in_({"ready", "processing"}),
                    KnowledgeBaseModel.status == "active",
                    IndexVersionModel.status == "active",
                    IndexVersionModel.active_slot == "active",
                )
            )
            if active_binding is None:
                raise ResourceConflictError(
                    "citation evidence is no longer in the active published scope"
                )
        if value.source_quote not in evidence_row.content:
            raise ResourceConflictError("citation source_quote is not present in evidence")
        natural_row = self.session.scalar(
            select(CitationFactModel).where(
                CitationFactModel.answer_id == value.answer_id,
                CitationFactModel.evidence_id == value.evidence_id,
                CitationFactModel.answer_span_start == value.answer_span.start,
                CitationFactModel.answer_span_end == value.answer_span.end,
                CitationFactModel.quote_hash == value.quote_hash,
            )
        )
        id_row = self.session.get(CitationFactModel, value.citation_id)
        row = id_row or natural_row
        if id_row is not None and natural_row is not None and id_row.id != natural_row.id:
            raise ResourceConflictError("citation identity conflicts with existing fact")
        if row is not None:
            if self._citation_domain(row).model_dump(mode="json") != value.model_dump(mode="json"):
                raise ResourceConflictError(
                    "citation fact is immutable for the same answer/evidence/span"
                )
            return self.citation_fact_dict(row)

        row = CitationFactModel(
            id=value.citation_id,
            schema_version=value.schema_version,
            answer_id=value.answer_id,
            evidence_id=value.evidence_id,
            answer_span_start=value.answer_span.start,
            answer_span_end=value.answer_span.end,
            source_quote=value.source_quote,
            quote_hash=value.quote_hash,
            source_label=value.source_label,
            validation_status=value.validation_status,
            reason_codes_json=list(value.reason_codes),
            ordinal=value.ordinal,
        )
        self.session.add(row)
        self.session.flush()
        return self.citation_fact_dict(row)

    def list_citation_facts(self, answer_id: str) -> list[dict]:
        query = (
            select(CitationFactModel)
            .join(
                EvidenceRecordModel,
                EvidenceRecordModel.id == CitationFactModel.evidence_id,
            )
            .where(CitationFactModel.answer_id == answer_id)
        )
        if self.tenant_id:
            query = query.where(EvidenceRecordModel.tenant_id == self.tenant_id)
        rows = self.session.scalars(
            query.order_by(CitationFactModel.ordinal.asc(), CitationFactModel.id.asc())
        ).all()
        return [self.citation_fact_dict(row) for row in rows]

    def record_parse_artifact(
        self,
        *,
        document_version_id: str,
        input_sha256: str,
        parser_id: str,
        parser_version: str,
        schema_version: str,
        status: str = "succeeded",
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        artifact_size: int = 0,
        warnings: list | None = None,
        metrics: dict | None = None,
    ) -> dict:
        version = self.get_document_version_model(document_version_id)
        if input_sha256 != version.content_sha256:
            raise ResourceConflictError(
                "parse artifact input SHA-256 does not match document version"
            )
        if status not in {"pending", "running", "succeeded", "failed"}:
            raise ResourceConflictError(f"unsupported parse artifact status: {status}")
        if artifact_size < 0:
            raise ResourceConflictError("parse artifact size must be non-negative")
        if status == "succeeded" and (not artifact_uri or not artifact_sha256):
            raise ResourceConflictError(
                "succeeded parse artifact requires artifact URI and SHA-256"
            )

        row = self.session.scalar(
            select(DocumentParseArtifactModel).where(
                DocumentParseArtifactModel.document_version_id == document_version_id,
                DocumentParseArtifactModel.parser_id == parser_id,
                DocumentParseArtifactModel.parser_version == parser_version,
                DocumentParseArtifactModel.schema_version == schema_version,
            )
        )
        if row is None:
            row = DocumentParseArtifactModel(
                id=new_id("pa"),
                document_version_id=document_version_id,
                input_sha256=input_sha256,
                parser_id=parser_id,
                parser_version=parser_version,
                schema_version=schema_version,
                status=status,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256,
                artifact_size=artifact_size,
                warnings_json=list(warnings or []),
                metrics_json=dict(metrics or {}),
            )
            self.session.add(row)
        else:
            if row.input_sha256 != input_sha256:
                raise ResourceConflictError(
                    "parse artifact cache key has a different input SHA-256"
                )
            if row.status == "succeeded" and (
                row.artifact_uri != artifact_uri
                or row.artifact_sha256 != artifact_sha256
                or row.artifact_size != artifact_size
            ):
                raise ResourceConflictError(
                    "succeeded parse artifact is immutable for the same parser cache key"
                )
            row.status = status
            row.artifact_uri = artifact_uri
            row.artifact_sha256 = artifact_sha256
            row.artifact_size = artifact_size
            row.warnings_json = list(warnings or [])
            row.metrics_json = dict(metrics or {})
        self.session.flush()
        return self.parse_artifact_dict(row)

    def find_parse_artifact(
        self,
        *,
        document_version_id: str,
        parser_id: str,
        parser_version: str,
        schema_version: str,
    ) -> dict | None:
        self.get_document_version_model(document_version_id)
        row = self.session.scalar(
            select(DocumentParseArtifactModel).where(
                DocumentParseArtifactModel.document_version_id == document_version_id,
                DocumentParseArtifactModel.parser_id == parser_id,
                DocumentParseArtifactModel.parser_version == parser_version,
                DocumentParseArtifactModel.schema_version == schema_version,
            )
        )
        return self.parse_artifact_dict(row) if row else None

    def list_parse_artifacts(self, document_version_id: str) -> list[dict]:
        self.get_document_version_model(document_version_id)
        rows = self.session.scalars(
            select(DocumentParseArtifactModel)
            .where(DocumentParseArtifactModel.document_version_id == document_version_id)
            .order_by(DocumentParseArtifactModel.created_at.asc())
        ).all()
        return [self.parse_artifact_detail(row) for row in rows]

    def record_chunking_run(
        self,
        *,
        parse_artifact_id: str,
        template_id: str,
        template_version: str,
        config_hash: str,
        status: str = "succeeded",
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        artifact_size: int = 0,
        chunk_count: int = 0,
    ) -> dict:
        parse_artifact = self.get_parse_artifact_model(parse_artifact_id)
        if parse_artifact.status != "succeeded":
            raise ResourceConflictError("chunking requires a succeeded parse artifact")
        if status not in {"pending", "running", "succeeded", "failed"}:
            raise ResourceConflictError(f"unsupported chunking run status: {status}")
        if artifact_size < 0 or chunk_count < 0:
            raise ResourceConflictError("chunking artifact size/count must be non-negative")
        if status == "succeeded" and (not artifact_uri or not artifact_sha256):
            raise ResourceConflictError("succeeded chunking run requires artifact URI and SHA-256")

        row = self.session.scalar(
            select(DocumentChunkingRunModel).where(
                DocumentChunkingRunModel.parse_artifact_id == parse_artifact_id,
                DocumentChunkingRunModel.template_id == template_id,
                DocumentChunkingRunModel.template_version == template_version,
                DocumentChunkingRunModel.config_hash == config_hash,
            )
        )
        if row is None:
            row = DocumentChunkingRunModel(
                id=new_id("crun"),
                parse_artifact_id=parse_artifact_id,
                template_id=template_id,
                template_version=template_version,
                config_hash=config_hash,
                status=status,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256,
                artifact_size=artifact_size,
                chunk_count=chunk_count,
            )
            self.session.add(row)
        else:
            if row.status == "succeeded" and (
                row.artifact_uri != artifact_uri
                or row.artifact_sha256 != artifact_sha256
                or row.artifact_size != artifact_size
                or row.chunk_count != chunk_count
            ):
                raise ResourceConflictError(
                    "succeeded chunking run is immutable for the same template cache key"
                )
            row.status = status
            row.artifact_uri = artifact_uri
            row.artifact_sha256 = artifact_sha256
            row.artifact_size = artifact_size
            row.chunk_count = chunk_count
        self.session.flush()
        return self.chunking_run_dict(row)

    def find_chunking_run(
        self,
        *,
        parse_artifact_id: str,
        template_id: str,
        template_version: str,
        config_hash: str,
    ) -> dict | None:
        self.get_parse_artifact_model(parse_artifact_id)
        row = self.session.scalar(
            select(DocumentChunkingRunModel).where(
                DocumentChunkingRunModel.parse_artifact_id == parse_artifact_id,
                DocumentChunkingRunModel.template_id == template_id,
                DocumentChunkingRunModel.template_version == template_version,
                DocumentChunkingRunModel.config_hash == config_hash,
            )
        )
        return self.chunking_run_dict(row) if row else None

    def list_chunking_runs(self, document_version_id: str) -> list[dict]:
        self.get_document_version_model(document_version_id)
        rows = self.session.scalars(
            select(DocumentChunkingRunModel)
            .join(
                DocumentParseArtifactModel,
                DocumentParseArtifactModel.id == DocumentChunkingRunModel.parse_artifact_id,
            )
            .where(DocumentParseArtifactModel.document_version_id == document_version_id)
            .order_by(DocumentChunkingRunModel.created_at.asc())
        ).all()
        return [self.chunking_run_dict(row) for row in rows]

    def get_ingestion_job_model(self, job_id: str) -> IngestionJobModel:
        row = self.session.get(IngestionJobModel, job_id)
        if row is None:
            raise ResourceNotFoundError(f"ingestion job not found: {job_id}")
        self.get_document_version_model(row.document_version_id)
        return row

    def list_documents(self, kb_id: str) -> list[dict]:
        self.get_knowledge_base_model(kb_id)
        rows = self.session.scalars(
            select(DocumentModel)
            .where(DocumentModel.kb_id == kb_id, DocumentModel.status != "deleted")
            .order_by(DocumentModel.created_at.desc())
        ).all()
        return [self.document_detail(row) for row in rows]

    def document_detail(self, document: DocumentModel) -> dict:
        versions = self.session.scalars(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document.id)
            .order_by(DocumentVersionModel.version_no.desc())
        ).all()
        latest_job = self.session.scalar(
            select(IngestionJobModel)
            .join(
                DocumentVersionModel,
                IngestionJobModel.document_version_id == DocumentVersionModel.id,
            )
            .where(DocumentVersionModel.document_id == document.id)
            # SQLite 的秒级 created_at 可能让紧接着创建的 v1/v2 job 同时刻；
            # 先按文档版本号、attempt 排序，确保 latest_job 真正属于最新版本。
            .order_by(
                DocumentVersionModel.version_no.desc(),
                IngestionJobModel.attempt.desc(),
                IngestionJobModel.created_at.desc(),
                IngestionJobModel.id.desc(),
            )
            .limit(1)
        )
        active_binding = None
        if document.active_version_id:
            active_binding = self.session.scalar(
                select(DocumentIndexBindingModel)
                .join(
                    IndexVersionModel,
                    IndexVersionModel.id == DocumentIndexBindingModel.index_version_id,
                )
                .where(
                    DocumentIndexBindingModel.document_version_id == document.active_version_id,
                    DocumentIndexBindingModel.status == "published",
                    IndexVersionModel.status == "active",
                    IndexVersionModel.active_slot == "active",
                )
            )
        result = self.document_dict(document)
        result.update(
            {
                "versions": [self.document_version_detail(row) for row in versions],
                "latest_job": self.ingestion_job_dict(latest_job) if latest_job else None,
                "chunk_count": active_binding.chunk_count if active_binding else 0,
                "active_index_version_id": active_binding.index_version_id
                if active_binding
                else None,
            }
        )
        return result

    def get_document_detail(self, document_id: str) -> dict:
        return self.document_detail(self.get_document_model(document_id))

    def get_document_version_detail(self, version_id: str) -> dict:
        return self.document_version_detail(self.get_document_version_model(version_id))

    def document_version_detail(self, version: DocumentVersionModel) -> dict:
        artifacts = self.session.scalars(
            select(DocumentParseArtifactModel)
            .where(DocumentParseArtifactModel.document_version_id == version.id)
            .order_by(DocumentParseArtifactModel.created_at.asc())
        ).all()
        runs = self.session.scalars(
            select(DocumentChunkingRunModel)
            .join(
                DocumentParseArtifactModel,
                DocumentParseArtifactModel.id == DocumentChunkingRunModel.parse_artifact_id,
            )
            .where(DocumentParseArtifactModel.document_version_id == version.id)
            .order_by(DocumentChunkingRunModel.created_at.asc())
        ).all()
        runs_by_artifact: dict[str, list[dict]] = {}
        for run in runs:
            runs_by_artifact.setdefault(run.parse_artifact_id, []).append(
                self.chunking_run_dict(run)
            )
        artifact_details = []
        for artifact in artifacts:
            detail = self.parse_artifact_dict(artifact)
            detail["chunking_runs"] = runs_by_artifact.get(artifact.id, [])
            artifact_details.append(detail)
        result = self.document_version_dict(version)
        result.update(
            {
                "parse_artifacts": artifact_details,
                "chunking_runs": [self.chunking_run_dict(row) for row in runs],
            }
        )
        return result

    def parse_artifact_detail(self, artifact: DocumentParseArtifactModel) -> dict:
        self.get_document_version_model(artifact.document_version_id)
        runs = self.session.scalars(
            select(DocumentChunkingRunModel)
            .where(DocumentChunkingRunModel.parse_artifact_id == artifact.id)
            .order_by(DocumentChunkingRunModel.created_at.asc())
        ).all()
        result = self.parse_artifact_dict(artifact)
        result["chunking_runs"] = [self.chunking_run_dict(row) for row in runs]
        return result

    def delete_document(self, document_id: str) -> dict:
        document = self.get_document_model(document_id)
        ensure_document_transition(document.status, "deleted")
        document.status = "deleted"
        document.deleted_at = utcnow()
        self.session.flush()
        return self.document_dict(document)

    def start_job(self, job_id: str) -> dict:
        job = self.get_ingestion_job_model(job_id)
        ensure_ingestion_job_transition(job.status, "processing")
        version = self.get_document_version_model(job.document_version_id)
        if version.status not in {"uploaded", "failed"}:
            raise ResourceConflictError(
                f"document version cannot start from status: {version.status}"
            )
        job.status = "processing"
        job.stage = "node_entry"
        job.progress = 0.02
        job.started_at = utcnow()
        if version.status == "failed":
            ensure_document_version_transition("failed", "parsing")
        elif version.status == "uploaded":
            ensure_document_version_transition("uploaded", "parsing")
        version.status = "parsing"
        self.session.flush()
        return self.ingestion_job_detail(job)

    def update_job_stage(self, job_id: str, stage: str, progress: float | None = None) -> dict:
        job = self.get_ingestion_job_model(job_id)
        if job.status != "processing":
            return self.ingestion_job_detail(job)
        job.stage = stage
        if progress is not None:
            job.progress = max(job.progress, min(progress, 0.99))
        version = self.get_document_version_model(job.document_version_id)
        target = None
        if stage in {"node_metadata_extract", "node_link_and_table_chunk", "node_chunk_merge"}:
            target = "chunking"
        elif stage == "node_bge_embedding":
            target = "embedding"
        elif stage == "node_import_milvus":
            target = "staged"
        if target:
            self._advance_version(version, target)
        self.session.flush()
        return self.ingestion_job_detail(job)

    def _advance_version(self, version: DocumentVersionModel, target: str) -> None:
        order = ["uploaded", "parsing", "chunking", "embedding", "staged", "published"]
        if version.status == "failed":
            ensure_document_version_transition("failed", "parsing")
            version.status = "parsing"
        current_index = order.index(version.status)
        target_index = order.index(target)
        for status in order[current_index + 1 : target_index + 1]:
            ensure_document_version_transition(version.status, status)
            version.status = status

    def complete_job(self, job_id: str, *, chunk_count: int, token_count: int = 0) -> dict:
        job = self.get_ingestion_job_model(job_id)
        index = self.session.get(IndexVersionModel, job.index_version_id)
        if index is None:
            raise ResourceConflictError("ingestion target index is missing")
        if index.status == "staging":
            raise ResourceConflictError(
                "staging index completion requires a validated manifest and durable lease"
            )
        bootstrap_job_ids = {
            str(value)
            for value in (index.manifest_summary or {}).get("bootstrap_job_ids", [])
            if str(value)
        }
        if job.id not in bootstrap_job_ids:
            raise ResourceConflictError(
                "published index version is immutable; import into a staging index"
            )
        ensure_ingestion_job_transition(job.status, "completed")
        version = self.get_document_version_model(job.document_version_id)
        self._advance_version(version, "published")
        document = self.get_document_model(version.document_id)
        if document.status != "ready":
            ensure_document_transition(document.status, "ready")
        document.status = "ready"
        document.active_version_id = version.id
        version.published_at = utcnow()
        binding = self.session.get(DocumentIndexBindingModel, (version.id, job.index_version_id))
        if binding is None:
            binding = DocumentIndexBindingModel(
                document_version_id=version.id,
                index_version_id=job.index_version_id,
                chunk_count=chunk_count,
                token_count=token_count,
            )
            self.session.add(binding)
        else:
            binding.chunk_count = chunk_count
            binding.token_count = token_count
        job.status = "completed"
        job.stage = "published"
        job.progress = 1.0
        job.finished_at = utcnow()
        self.session.flush()
        return self.ingestion_job_detail(job)

    def fail_job(self, job_id: str, *, error_code: str, error_detail: str) -> dict:
        job = self.get_ingestion_job_model(job_id)
        if job.status in {"completed", "cancelled"}:
            return self.ingestion_job_detail(job)
        if job.status == "failed":
            return self.ingestion_job_detail(job)
        ensure_ingestion_job_transition(job.status, "failed")
        job.status = "failed"
        job.stage = "failed"
        job.error_code = error_code
        job.error_detail = error_detail[:4000]
        job.finished_at = utcnow()
        version = self.get_document_version_model(job.document_version_id)
        if version.status != "published":
            ensure_document_version_transition(version.status, "failed")
            version.status = "failed"
        document = self.get_document_model(version.document_id)
        if document.status != "deleted":
            target_status = "ready" if document.active_version_id else "failed"
            if document.status != target_status:
                ensure_document_transition(document.status, target_status)
                document.status = target_status
        self.session.flush()
        return self.ingestion_job_detail(job)

    def get_ingestion_job_detail(self, job_id: str) -> dict:
        return self.ingestion_job_detail(self.get_ingestion_job_model(job_id))

    def import_legacy_document(
        self,
        *,
        kb_id: str,
        document_id: str,
        logical_name: str,
        chunk_count: int,
        collection_name: str,
        batch_id: str,
    ) -> dict:
        """把可识别的旧 Milvus doc_id 映射为控制面事实；不会改写向量数据。"""
        self.ensure_defaults()
        kb = self.session.get(KnowledgeBaseModel, kb_id)
        if kb is None:
            kb = KnowledgeBaseModel(
                id=kb_id,
                tenant_id=DEFAULT_TENANT_ID,
                workspace_id=DEFAULT_WORKSPACE_ID,
                name=f"Legacy {kb_id}"[:255],
                description="由 TODO-01 legacy migration 创建，请人工确认归属",
                status="disabled",
                created_by="legacy-migration",
            )
            self.session.add(kb)
            self.session.flush()
            self._create_initial_index(kb.id)
        existing = self.session.get(DocumentModel, document_id)
        if existing is not None:
            return {"action": "skipped", "document_id": document_id, "reason": "already_exists"}

        source_uri = f"legacy://milvus/{collection_name}/{document_id}?batch={batch_id}"
        document = DocumentModel(
            id=document_id,
            kb_id=kb_id,
            logical_name=logical_name[:512] or document_id,
            source_type="legacy_milvus",
            source_uri=source_uri,
            status="ready",
        )
        version = DocumentVersionModel(
            id=new_id("dv"),
            document_id=document_id,
            version_no=1,
            content_sha256=sha256(f"legacy:{collection_name}:{document_id}".encode()).hexdigest(),
            size_bytes=0,
            mime_type="application/x-stt-legacy",
            object_key=source_uri,
            parser_config_snapshot={"legacy": True, "original_object_missing": True},
            status="published",
            published_at=utcnow(),
        )
        self.session.add(document)
        self.session.flush()
        self.session.add(version)
        self.session.flush()
        index = self.get_active_index(kb_id)
        document.active_version_id = version.id
        job = IngestionJobModel(
            id=new_id("job"),
            config_version=settings.config_version,
            document_version_id=version.id,
            index_version_id=index.id,
            status="completed",
            stage="legacy_imported",
            progress=1.0,
            attempt=1,
            started_at=utcnow(),
            finished_at=utcnow(),
        )
        binding = DocumentIndexBindingModel(
            document_version_id=version.id,
            index_version_id=index.id,
            chunk_count=chunk_count,
            token_count=0,
        )
        self.session.add_all([job, binding])
        self.session.flush()
        return {
            "action": "created",
            "document_id": document.id,
            "document_version_id": version.id,
            "job_id": job.id,
            "index_version_id": index.id,
        }

    def rollback_legacy_batch(self, batch_id: str) -> int:
        documents = self.session.scalars(
            select(DocumentModel).where(
                DocumentModel.source_type == "legacy_milvus",
                DocumentModel.source_uri.like(f"%?batch={batch_id}"),
            )
        ).all()
        removed = 0
        for document in documents:
            version_ids = self.session.scalars(
                select(DocumentVersionModel.id).where(
                    DocumentVersionModel.document_id == document.id
                )
            ).all()
            evidence_ids = (
                self.session.scalars(
                    select(EvidenceRecordModel.id).where(
                        EvidenceRecordModel.document_version_id.in_(version_ids)
                    )
                ).all()
                if version_ids
                else []
            )
            if evidence_ids:
                # Citation -> Evidence 使用 RESTRICT，合规 purge/legacy rollback 必须显式
                # 先移除引用事实，避免静默留下悬空引用。
                self.session.query(CitationFactModel).filter(
                    CitationFactModel.evidence_id.in_(evidence_ids)
                ).delete(synchronize_session=False)
                self.session.query(EvidenceRecordModel).filter(
                    EvidenceRecordModel.id.in_(evidence_ids)
                ).delete(synchronize_session=False)
            artifact_ids = (
                self.session.scalars(
                    select(DocumentParseArtifactModel.id).where(
                        DocumentParseArtifactModel.document_version_id.in_(version_ids)
                    )
                ).all()
                if version_ids
                else []
            )
            if artifact_ids:
                self.session.query(DocumentChunkingRunModel).filter(
                    DocumentChunkingRunModel.parse_artifact_id.in_(artifact_ids)
                ).delete(synchronize_session=False)
                self.session.query(DocumentParseArtifactModel).filter(
                    DocumentParseArtifactModel.id.in_(artifact_ids)
                ).delete(synchronize_session=False)
            for version_id in version_ids:
                self.session.query(DocumentIndexBindingModel).filter_by(
                    document_version_id=version_id
                ).delete()
                self.session.query(IngestionJobModel).filter_by(
                    document_version_id=version_id
                ).delete()
            self.session.query(DocumentVersionModel).filter(
                DocumentVersionModel.id.in_(version_ids)
            ).delete(synchronize_session=False)
            self.session.flush()
            self.session.query(DocumentModel).filter_by(id=document.id).delete()
            removed += 1
        self.session.flush()
        return removed

    def knowledge_base_dict(self, row: KnowledgeBaseModel) -> dict:
        document_count = (
            self.session.scalar(
                select(func.count(DocumentModel.id)).where(
                    DocumentModel.kb_id == row.id,
                    DocumentModel.status != "deleted",
                )
            )
            or 0
        )
        active_index = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.kb_id == row.id,
                IndexVersionModel.active_slot == "active",
            )
        )
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "workspace_id": row.workspace_id,
            "name": row.name,
            "description": row.description,
            "status": row.status,
            "retrieval_config_id": row.retrieval_config_id,
            "created_by": row.created_by,
            "document_count": document_count,
            "active_index_version_id": active_index.id if active_index else None,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            "deleted_at": _iso(row.deleted_at),
            "data_classification": row.data_classification,
            "external_llm_allowed": row.external_llm_allowed,
            "web_search_allowed": row.web_search_allowed,
            "retention_policy_id": row.retention_policy_id,
        }

    @staticmethod
    def document_dict(row: DocumentModel) -> dict:
        return {
            "id": row.id,
            "kb_id": row.kb_id,
            "logical_name": row.logical_name,
            "source_type": row.source_type,
            "source_uri": row.source_uri,
            "status": row.status,
            "active_version_id": row.active_version_id,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            "deleted_at": _iso(row.deleted_at),
        }

    @staticmethod
    def document_version_dict(row: DocumentVersionModel) -> dict:
        return {
            "id": row.id,
            "document_id": row.document_id,
            "version_no": row.version_no,
            "content_sha256": row.content_sha256,
            "pipeline_fingerprint": row.pipeline_fingerprint,
            "size_bytes": row.size_bytes,
            "mime_type": row.mime_type,
            "object_key": row.object_key,
            "parser_config_snapshot": row.parser_config_snapshot,
            "status": row.status,
            "published_at": _iso(row.published_at),
            "created_at": _iso(row.created_at),
        }

    @staticmethod
    def parse_artifact_dict(row: DocumentParseArtifactModel) -> dict:
        return {
            "id": row.id,
            "document_version_id": row.document_version_id,
            "input_sha256": row.input_sha256,
            "parser_id": row.parser_id,
            "parser_version": row.parser_version,
            "schema_version": row.schema_version,
            "status": row.status,
            "artifact_uri": row.artifact_uri,
            "artifact_sha256": row.artifact_sha256,
            "artifact_size": row.artifact_size,
            "warnings": row.warnings_json or [],
            "metrics": row.metrics_json or {},
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    @staticmethod
    def chunking_run_dict(row: DocumentChunkingRunModel) -> dict:
        return {
            "id": row.id,
            "parse_artifact_id": row.parse_artifact_id,
            "template_id": row.template_id,
            "template_version": row.template_version,
            "config_hash": row.config_hash,
            "status": row.status,
            "artifact_uri": row.artifact_uri,
            "artifact_sha256": row.artifact_sha256,
            "artifact_size": row.artifact_size,
            "chunk_count": row.chunk_count,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    @staticmethod
    def _evidence_domain(row: EvidenceRecordModel) -> Evidence:
        if row.source_type == "local":
            source = {
                "source_type": "local",
                "tenant_id": row.tenant_id,
                "kb_id": row.kb_id,
                "document_id": row.document_id,
                "document_version_id": row.document_version_id,
                "index_version_id": row.index_version_id,
                "chunk_id": row.chunk_id,
                "source_block_ids": row.source_block_ids_json or [],
            }
        else:
            fetched_at = row.source_fetched_at
            # SQLite DateTime(timezone=True) 读取时可能丢失 tzinfo；数据库中的时间
            # 统一按 UTC 写入，恢复领域对象时显式补回，避免把快照误判为无时区。
            if fetched_at is not None and fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=UTC)
            source = {
                "source_type": "web",
                "tenant_id": row.tenant_id,
                "source_url": row.source_url,
                "content_hash": row.content_hash,
                "fetched_at": fetched_at,
                "source_block_ids": row.source_block_ids_json or [],
            }
        return Evidence.model_validate(
            {
                "schema_version": row.schema_version,
                "evidence_id": row.id,
                "source": source,
                "content": row.content,
                "content_hash": row.content_hash,
                "provenance_spans": row.provenance_spans_json or [],
                "heading_path": row.heading_path_json or [],
                "asset_refs": row.asset_refs_json or [],
                "source_label": row.source_label,
            }
        )

    @classmethod
    def evidence_dict(cls, row: EvidenceRecordModel) -> dict:
        result = cls._evidence_domain(row).model_dump(mode="json", exclude_none=True)
        result.update(
            {
                "created_at": _iso(row.created_at),
                "updated_at": _iso(row.updated_at),
                "primary_location": {
                    "block_id": row.primary_block_id,
                    "availability": row.primary_availability,
                    "coordinate_space": row.primary_coordinate_space,
                    **({"page_no": row.primary_page_no} if row.primary_page_no is not None else {}),
                    **(
                        {"bbox": row.primary_bbox_json} if row.primary_bbox_json is not None else {}
                    ),
                    **(
                        {"slide_no": row.primary_slide_no}
                        if row.primary_slide_no is not None
                        else {}
                    ),
                    **({"sheet": row.primary_sheet} if row.primary_sheet else {}),
                    **({"cell_range": row.primary_cell_range} if row.primary_cell_range else {}),
                    **(
                        {"line_start": row.primary_line_start}
                        if row.primary_line_start is not None
                        else {}
                    ),
                    **(
                        {"line_end": row.primary_line_end}
                        if row.primary_line_end is not None
                        else {}
                    ),
                    **({"reason_code": row.primary_reason_code} if row.primary_reason_code else {}),
                },
            }
        )
        return result

    @staticmethod
    def _citation_domain(row: CitationFactModel) -> CitationFact:
        return CitationFact.model_validate(
            {
                "schema_version": row.schema_version,
                "citation_id": row.id,
                "answer_id": row.answer_id,
                "evidence_id": row.evidence_id,
                "answer_span": {
                    "start": row.answer_span_start,
                    "end": row.answer_span_end,
                },
                "source_quote": row.source_quote,
                "quote_hash": row.quote_hash,
                "source_label": row.source_label,
                "validation_status": row.validation_status,
                "reason_codes": row.reason_codes_json or [],
                "ordinal": row.ordinal,
            }
        )

    @classmethod
    def citation_fact_dict(cls, row: CitationFactModel) -> dict:
        result = cls._citation_domain(row).model_dump(mode="json", exclude_none=True)
        result.update({"created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at)})
        return result

    @staticmethod
    def ingestion_job_dict(row: IngestionJobModel) -> dict:
        return {
            "id": row.id,
            "config_version": getattr(row, "config_version", "cfg_legacy"),
            "operation": getattr(row, "operation", "import"),
            "document_version_id": row.document_version_id,
            "index_version_id": row.index_version_id,
            "status": row.status,
            "stage": row.stage,
            "progress": row.progress,
            "attempt": row.attempt,
            "run_attempt": row.run_attempt,
            "max_attempts": row.max_attempts,
            "priority": row.priority,
            "idempotency_key": row.idempotency_key,
            "lease_owner": row.lease_owner,
            "lease_expires_at": _iso(row.lease_expires_at),
            "heartbeat_at": _iso(row.heartbeat_at),
            "next_retry_at": _iso(row.next_retry_at),
            "cancel_requested_at": _iso(row.cancel_requested_at),
            "error_category": row.error_category,
            "error_code": row.error_code,
            "error_detail": row.error_detail,
            "progress_detail": row.progress_detail or {},
            "checkpoint_stage": row.checkpoint_stage,
            "checkpoint_sequence": row.checkpoint_sequence,
            "last_event_sequence": row.last_event_sequence,
            "recovered_count": row.recovered_count,
            "started_at": _iso(row.started_at),
            "finished_at": _iso(row.finished_at),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    def ingestion_job_detail(self, row: IngestionJobModel) -> dict:
        version = self.get_document_version_model(row.document_version_id)
        document = self.get_document_model(version.document_id, include_deleted=True)
        result = self.ingestion_job_dict(row)
        result.update(
            {
                "document_id": document.id,
                "kb_id": document.kb_id,
                "object_key": version.object_key,
                "version_no": version.version_no,
                "parser_config_snapshot": version.parser_config_snapshot,
            }
        )
        return result

    @staticmethod
    def index_version_dict(row: IndexVersionModel) -> dict:
        return {
            "id": row.id,
            "kb_id": row.kb_id,
            "version_no": row.version_no,
            "embedding_model_id": row.embedding_model_id,
            "vector_dim": row.vector_dim,
            "schema_version": row.schema_version,
            "status": row.status,
            "collection_alias": row.collection_alias,
            "activated_at": _iso(row.activated_at),
            "retired_at": _iso(row.retired_at),
            "created_at": _iso(row.created_at),
        }
