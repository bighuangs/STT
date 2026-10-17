"""P2 高级能力的 tenant-scoped SQL 仓储。

严格领域模型在写入前由各能力服务完成校验；仓储只接受 JSON-safe snapshot，
并统一维护不可变版本、发布状态、运行幂等、元数据事实、连接器状态、长期记忆
和媒体派生链。所有读取都显式包含 tenant，禁止用客户端 payload 推断作用域。
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domain.connectors import SourceItemBinding, SyncCursor
from app.domain.knowledge_governance import (
    EnhancementReference,
    MetadataSchemaVersion,
    MetadataTarget,
    MetadataValueRecord,
    stable_governance_id,
)
from app.domain.memory import LongTermMemory
from app.domain.multimodal import MediaAsset, MediaDerivation
from app.infra.persistence.sql.models import (
    CapabilityResourceModel,
    CapabilityRunModel,
    ConnectorBindingModel,
    ConnectorCursorModel,
    DocumentModel,
    DocumentVersionModel,
    EvidenceRecordModel,
    KnowledgeBaseModel,
    LongTermMemoryModel,
    MediaAssetModel,
    MetadataValueRecordModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
    canonical_json_sha256,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(value: dict | list | None) -> str:
    return f"sha256:{canonical_json_sha256(value)}"


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _same_instant(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return left is right
    normalized_left = left.replace(tzinfo=UTC) if left.tzinfo is None else left.astimezone(UTC)
    normalized_right = right.replace(tzinfo=UTC) if right.tzinfo is None else right.astimezone(UTC)
    return normalized_left == normalized_right


def _scope_key(kb_id: str | None) -> str:
    return f"kb:{kb_id}" if kb_id is not None else "tenant"


class CapabilityResourceRepository:
    """模型路由、schema、advanced index、datasource、Agent 和 profile 注册表。"""

    def __init__(self, session: Session, *, tenant_id: str, actor_id: str) -> None:
        self.session = session
        self.tenant_id = tenant_id
        self.actor_id = actor_id

    def _assert_kb(self, kb_id: str | None) -> None:
        if kb_id is None:
            return
        row = self.session.scalar(
            select(KnowledgeBaseModel).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == self.tenant_id,
                KnowledgeBaseModel.deleted_at.is_(None),
            )
        )
        if row is None:
            raise ResourceNotFoundError("knowledge base not found in tenant scope")

    def create(
        self,
        *,
        resource_type: str,
        logical_id: str,
        version: str,
        schema_version: str,
        payload: dict[str, Any],
        kb_id: str | None = None,
        enabled: bool = False,
    ) -> dict[str, Any]:
        self._assert_kb(kb_id)
        if enabled:
            raise ResourceConflictError("draft capability resource cannot be enabled")
        scope_key = _scope_key(kb_id)
        existing = self.session.scalar(
            select(CapabilityResourceModel).where(
                CapabilityResourceModel.tenant_id == self.tenant_id,
                CapabilityResourceModel.resource_type == resource_type,
                CapabilityResourceModel.logical_id == logical_id,
                CapabilityResourceModel.version == version,
                CapabilityResourceModel.scope_key == scope_key,
            )
        )
        payload_hash = _hash(payload)
        if existing is not None:
            if existing.payload_hash == payload_hash:
                return self.as_dict(existing)
            raise ResourceConflictError(
                "capability resource version already exists with another payload"
            )
        row = CapabilityResourceModel(
            id=f"cap_{uuid4().hex}",
            tenant_id=self.tenant_id,
            kb_id=kb_id,
            scope_key=scope_key,
            resource_type=resource_type.strip(),
            logical_id=logical_id.strip(),
            version=version.strip(),
            status="draft",
            enabled=False,
            owner_id=self.actor_id,
            schema_version=schema_version.strip(),
            payload_json=payload,
            payload_hash=payload_hash,
        )
        self.session.add(row)
        self.session.flush()
        return self.as_dict(row)

    def get(
        self,
        resource_id: str,
        *,
        resource_type: str | None = None,
    ) -> CapabilityResourceModel:
        criteria = [
            CapabilityResourceModel.id == resource_id,
            CapabilityResourceModel.tenant_id == self.tenant_id,
        ]
        if resource_type is not None:
            criteria.append(CapabilityResourceModel.resource_type == resource_type)
        row = self.session.scalar(select(CapabilityResourceModel).where(*criteria))
        if row is None:
            raise ResourceNotFoundError("capability resource not found")
        return row

    def list(
        self,
        *,
        resource_type: str | None = None,
        kb_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        criteria = [CapabilityResourceModel.tenant_id == self.tenant_id]
        if resource_type:
            criteria.append(CapabilityResourceModel.resource_type == resource_type)
        if kb_id:
            criteria.append(CapabilityResourceModel.kb_id == kb_id)
        if status:
            criteria.append(CapabilityResourceModel.status == status)
        rows = self.session.scalars(
            select(CapabilityResourceModel)
            .where(*criteria)
            .order_by(
                CapabilityResourceModel.resource_type,
                CapabilityResourceModel.logical_id,
                CapabilityResourceModel.created_at.desc(),
            )
        ).all()
        return [self.as_dict(row) for row in rows]

    def publish(self, resource_id: str, *, enable: bool = False) -> dict[str, Any]:
        row = self.get(resource_id)
        if row.status != "draft":
            raise ResourceConflictError("only draft capability resource can be published")
        previous = self.session.scalars(
            select(CapabilityResourceModel).where(
                CapabilityResourceModel.tenant_id == self.tenant_id,
                CapabilityResourceModel.resource_type == row.resource_type,
                CapabilityResourceModel.logical_id == row.logical_id,
                CapabilityResourceModel.kb_id == row.kb_id,
                CapabilityResourceModel.status == "published",
            )
        ).all()
        now = _now()
        for item in previous:
            item.status = "retired"
            item.enabled = False
            item.retired_at = now
        row.status = "published"
        row.enabled = enable
        row.published_at = now
        row.retired_at = None
        self.session.flush()
        return self.as_dict(row)

    def set_enabled(self, resource_id: str, enabled: bool) -> dict[str, Any]:
        row = self.get(resource_id)
        if row.status != "published" and enabled:
            raise ResourceConflictError("only published resource can be enabled")
        row.enabled = enabled
        self.session.flush()
        return self.as_dict(row)

    def retire(self, resource_id: str) -> dict[str, Any]:
        row = self.get(resource_id)
        if row.status == "retired":
            return self.as_dict(row)
        row.status = "retired"
        row.enabled = False
        row.retired_at = _now()
        self.session.flush()
        return self.as_dict(row)

    @staticmethod
    def as_dict(row: CapabilityResourceModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "scope_key": row.scope_key,
            "resource_type": row.resource_type,
            "logical_id": row.logical_id,
            "version": row.version,
            "status": row.status,
            "enabled": row.enabled,
            "owner_id": row.owner_id,
            "schema_version": row.schema_version,
            "payload": row.payload_json,
            "payload_hash": row.payload_hash,
            "published_at": _iso(row.published_at),
            "retired_at": _iso(row.retired_at),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }


class CapabilityRunRepository:
    def __init__(self, session: Session, *, tenant_id: str) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def create(
        self,
        *,
        run_type: str,
        trace_id: str,
        input_payload: dict[str, Any],
        input_summary: dict[str, Any],
        resource_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key:
            existing = self.session.scalar(
                select(CapabilityRunModel).where(
                    CapabilityRunModel.tenant_id == self.tenant_id,
                    CapabilityRunModel.run_type == run_type,
                    CapabilityRunModel.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if (
                    existing.input_hash != _hash(input_payload)
                    or existing.resource_id != resource_id
                ):
                    raise ResourceConflictError(
                        "capability idempotency key belongs to another input"
                    )
                return self.as_dict(existing)
        if resource_id is not None:
            resource = self.session.scalar(
                select(CapabilityResourceModel).where(
                    CapabilityResourceModel.id == resource_id,
                    CapabilityResourceModel.tenant_id == self.tenant_id,
                )
            )
            if resource is None:
                raise ResourceNotFoundError("capability resource not found")
        row = CapabilityRunModel(
            id=f"crun_{uuid4().hex}",
            tenant_id=self.tenant_id,
            resource_id=resource_id,
            run_type=run_type.strip(),
            status="planned",
            trace_id=trace_id.strip(),
            idempotency_key=idempotency_key,
            input_hash=_hash(input_payload),
            input_summary=input_summary,
            output_json={},
        )
        self.session.add(row)
        self.session.flush()
        return self.as_dict(row)

    def get(self, run_id: str) -> CapabilityRunModel:
        row = self.session.scalar(
            select(CapabilityRunModel).where(
                CapabilityRunModel.id == run_id,
                CapabilityRunModel.tenant_id == self.tenant_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError("capability run not found")
        return row

    def set_status(
        self,
        run_id: str,
        *,
        status: str,
        output: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> dict[str, Any]:
        row = self.get(run_id)
        allowed = {
            "planned": {"running", "cancelled"},
            "running": {
                "waiting_approval",
                "succeeded",
                "partial",
                "failed",
                "cancelled",
            },
            "waiting_approval": {"running", "failed", "cancelled"},
            "succeeded": set(),
            "partial": set(),
            "failed": set(),
            "cancelled": set(),
        }
        if status not in allowed.get(row.status, set()):
            raise ResourceConflictError(f"invalid capability run transition {row.status}->{status}")
        row.status = status
        if status == "running" and row.started_at is None:
            row.started_at = _now()
        if status in {"succeeded", "partial", "failed", "cancelled"}:
            row.completed_at = _now()
        if output is not None:
            row.output_json = output
        row.error_code = error_code
        row.error_detail = error_detail
        self.session.flush()
        return self.as_dict(row)

    @staticmethod
    def as_dict(row: CapabilityRunModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "resource_id": row.resource_id,
            "run_type": row.run_type,
            "status": row.status,
            "trace_id": row.trace_id,
            "idempotency_key": row.idempotency_key,
            "input_hash": row.input_hash,
            "input_summary": row.input_summary,
            "output": row.output_json,
            "error_code": row.error_code,
            "error_detail": row.error_detail,
            "started_at": _iso(row.started_at),
            "completed_at": _iso(row.completed_at),
            "created_at": _iso(row.created_at),
        }


class MetadataValueRepository:
    def __init__(self, session: Session, *, tenant_id: str, actor_id: str) -> None:
        self.session = session
        self.tenant_id = tenant_id
        self.actor_id = actor_id

    def _load_published_schema(
        self,
        schema_resource_id: str,
        *,
        kb_id: str,
    ) -> tuple[CapabilityResourceModel, MetadataSchemaVersion]:
        schema = self.session.scalar(
            select(CapabilityResourceModel).where(
                CapabilityResourceModel.id == schema_resource_id,
                CapabilityResourceModel.tenant_id == self.tenant_id,
                CapabilityResourceModel.kb_id == kb_id,
                CapabilityResourceModel.resource_type == "metadata_schema",
                CapabilityResourceModel.status == "published",
            )
        )
        if schema is None:
            raise ResourceNotFoundError("published metadata schema not found")
        raw_schema = schema.payload_json.get("schema", schema.payload_json)
        try:
            published_schema = MetadataSchemaVersion.model_validate(raw_schema)
        except ValueError as exc:
            raise ResourceConflictError("published metadata schema payload is invalid") from exc
        if published_schema.status != "published":
            raise ResourceConflictError(
                "metadata capability payload must contain a published schema"
            )
        if (
            published_schema.tenant_id != self.tenant_id
            or published_schema.kb_id != kb_id
            or published_schema.schema_id != schema.logical_id
            or str(published_schema.version) != schema.version
            or published_schema.schema_version != schema.schema_version
        ):
            raise ResourceConflictError(
                "metadata schema payload does not match capability scope/version"
            )
        return schema, published_schema

    def _validate_target_exists(self, target: MetadataTarget) -> None:
        if target.tenant_id != self.tenant_id:
            raise ResourceConflictError("metadata target tenant scope mismatch")
        document = self.session.scalar(
            select(DocumentModel).where(
                DocumentModel.id == target.document_id,
                DocumentModel.kb_id == target.kb_id,
                DocumentModel.deleted_at.is_(None),
            )
        )
        if document is None:
            raise ResourceNotFoundError("metadata target document not found in KB scope")
        if target.document_version_id is not None:
            version = self.session.scalar(
                select(DocumentVersionModel).where(
                    DocumentVersionModel.id == target.document_version_id,
                    DocumentVersionModel.document_id == target.document_id,
                )
            )
            if version is None:
                raise ResourceNotFoundError(
                    "metadata target document version not found in document scope"
                )
        if target.scope == "chunk":
            evidence_id = self.session.scalar(
                select(EvidenceRecordModel.id).where(
                    EvidenceRecordModel.source_type == "local",
                    EvidenceRecordModel.tenant_id == self.tenant_id,
                    EvidenceRecordModel.kb_id == target.kb_id,
                    EvidenceRecordModel.document_id == target.document_id,
                    EvidenceRecordModel.document_version_id == target.document_version_id,
                    EvidenceRecordModel.chunk_id == target.chunk_id,
                )
            )
            if evidence_id is None:
                raise ResourceNotFoundError(
                    "metadata target chunk evidence not found in tenant/KB/document/version scope"
                )

    def propose(
        self,
        *,
        schema_resource_id: str,
        target: MetadataTarget,
        field_key: str,
        value: object,
        source: str,
        confidence: float | None,
        provenance: dict[str, Any] | None = None,
        enhancement: dict[str, Any] | None = None,
        schema_revision: int | None = None,
    ) -> dict[str, Any]:
        if target.tenant_id != self.tenant_id:
            raise ResourceConflictError("metadata target tenant scope mismatch")
        _, published_schema = self._load_published_schema(
            schema_resource_id,
            kb_id=target.kb_id,
        )
        if schema_revision is not None and schema_revision != published_schema.revision:
            raise ResourceConflictError("metadata schema revision does not match published payload")
        try:
            definition = published_schema.field(field_key)
        except KeyError as exc:
            raise ResourceConflictError("metadata field was not found") from exc
        if definition.scope != target.scope:
            raise ResourceConflictError("metadata target scope does not match its schema field")
        try:
            normalized_value = definition.normalize_value(value)
        except ValueError as exc:
            raise ResourceConflictError(str(exc)) from exc
        # ``string_list`` 的领域规范值是 tuple；控制库只保存 JSON-safe 规范形态。
        normalized_json_value = json.loads(
            json.dumps(
                normalized_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        self._validate_target_exists(target)
        if source not in {"auto", "manual", "import"}:
            raise ResourceConflictError("unsupported metadata value source")
        if source == "auto" and confidence is None:
            raise ResourceConflictError("automatic metadata requires confidence")
        if source != "auto" and enhancement is not None:
            raise ResourceConflictError(
                "only automatic metadata can reference enhancement provenance"
            )
        resolved_schema_revision = published_schema.revision
        scope_id = target.target_key
        enhancement_reference: EnhancementReference | None = None
        if enhancement is not None:
            try:
                enhancement_reference = EnhancementReference.model_validate(enhancement)
            except ValueError as exc:
                raise ResourceConflictError("metadata enhancement provenance is invalid") from exc
        proposal_payload = {
            "schema_id": published_schema.schema_id,
            "schema_revision": resolved_schema_revision,
            "target": target.model_dump(mode="json"),
            "field_key": definition.key,
            "value": normalized_json_value,
            "source": source,
            "proposed_by": self.actor_id,
            "enhancement_run_id": (enhancement_reference.run_id if enhancement_reference else None),
        }
        record_id = stable_governance_id("mv", proposal_payload)
        existing_proposal = self.session.scalar(
            select(MetadataValueRecordModel).where(
                MetadataValueRecordModel.id == record_id,
                MetadataValueRecordModel.tenant_id == self.tenant_id,
            )
        )
        if existing_proposal is not None:
            return self.as_dict(existing_proposal)
        latest = self.session.scalar(
            select(MetadataValueRecordModel)
            .where(
                MetadataValueRecordModel.tenant_id == self.tenant_id,
                MetadataValueRecordModel.kb_id == target.kb_id,
                MetadataValueRecordModel.scope_type == target.scope,
                MetadataValueRecordModel.scope_id == scope_id,
                MetadataValueRecordModel.field_key == field_key,
            )
            .order_by(MetadataValueRecordModel.revision.desc())
        )
        revision = (latest.revision + 1) if latest else 1
        now = _now()
        try:
            domain_record = MetadataValueRecord(
                record_id=record_id,
                schema_id=published_schema.schema_id,
                schema_revision=resolved_schema_revision,
                field_key=definition.key,
                target=target,
                value=normalized_json_value,
                source=source,
                confidence=confidence,
                proposed_by=self.actor_id,
                proposed_at=now,
                enhancement=enhancement_reference,
            )
        except ValueError as exc:
            raise ResourceConflictError("metadata proposal is invalid") from exc
        row = MetadataValueRecordModel(
            id=domain_record.record_id,
            tenant_id=self.tenant_id,
            kb_id=target.kb_id,
            schema_resource_id=schema_resource_id,
            schema_id=domain_record.schema_id,
            schema_revision=domain_record.schema_revision,
            scope_type=target.scope,
            scope_id=scope_id,
            document_id=target.document_id,
            document_version_id=target.document_version_id,
            chunk_id=target.chunk_id,
            target_json=target.model_dump(mode="json"),
            field_key=domain_record.field_key,
            value_json=domain_record.value,
            value_hash=_hash({"value": domain_record.value}),
            source=domain_record.source,
            confidence=domain_record.confidence,
            status="proposed",
            revision=revision,
            created_by=domain_record.proposed_by,
            proposed_at=domain_record.proposed_at,
            enhancement_json=(
                domain_record.enhancement.model_dump(mode="json")
                if domain_record.enhancement
                else None
            ),
            provenance_json=provenance or {},
            supersedes_id=latest.id if latest else None,
        )
        self.session.add(row)
        self.session.flush()
        return self.as_dict(row)

    def publish(
        self,
        value_id: str,
        *,
        review_reason: str = "approved",
        allow_compliance_controlled: bool = False,
    ) -> dict[str, Any]:
        """发布 proposal；合规字段必须由授权边界传入显式高权限结果。

        HTTP/任务调用方只能在通过 ``metadata:compliance`` 授权后把
        ``allow_compliance_controlled`` 设为 true。默认值保持 fail closed。
        """

        row = self._get(value_id)
        if row.status != "proposed":
            raise ResourceConflictError("only proposed metadata value can be published")
        try:
            target = MetadataTarget.model_validate(row.target_json)
        except ValueError as exc:
            raise ResourceConflictError("persisted metadata target is invalid") from exc
        if (
            target.tenant_id != self.tenant_id
            or target.tenant_id != row.tenant_id
            or target.kb_id != row.kb_id
            or target.scope != row.scope_type
            or target.document_id != row.document_id
            or target.document_version_id != row.document_version_id
            or target.chunk_id != row.chunk_id
        ):
            raise ResourceConflictError(
                "persisted metadata target does not match its scoped columns"
            )
        _, published_schema = self._load_published_schema(
            row.schema_resource_id,
            kb_id=row.kb_id,
        )
        if (
            row.schema_id != published_schema.schema_id
            or row.schema_revision != published_schema.revision
        ):
            raise ResourceConflictError(
                "metadata proposal schema no longer matches the published schema"
            )
        try:
            definition = published_schema.field(row.field_key)
        except KeyError as exc:
            raise ResourceConflictError("metadata field was not found") from exc
        if definition.scope != target.scope:
            raise ResourceConflictError("metadata target scope does not match its schema field")
        try:
            definition.normalize_value(row.value_json)
        except ValueError as exc:
            raise ResourceConflictError(str(exc)) from exc
        self._validate_target_exists(target)
        if definition.compliance_controlled and not allow_compliance_controlled:
            raise ResourceConflictError(
                "COMPLIANCE_PERMISSION_REQUIRED: publishing "
                f"{row.field_key} requires compliance permission"
            )
        current = self.session.scalars(
            select(MetadataValueRecordModel).where(
                MetadataValueRecordModel.tenant_id == self.tenant_id,
                MetadataValueRecordModel.kb_id == row.kb_id,
                MetadataValueRecordModel.scope_type == row.scope_type,
                MetadataValueRecordModel.scope_id == row.scope_id,
                MetadataValueRecordModel.field_key == row.field_key,
                MetadataValueRecordModel.status == "published",
            )
        ).all()
        if row.source == "auto" and any(item.source == "manual" for item in current):
            raise ResourceConflictError(
                "automatic metadata cannot overwrite a published manual value"
            )
        for item in current:
            item.status = "superseded"
        row.status = "published"
        row.reviewed_by = self.actor_id
        row.reviewed_at = _now()
        row.review_reason = review_reason
        self.session.flush()
        return self.as_dict(row)

    def requires_compliance_permission(self, value_id: str) -> bool:
        """在 HTTP 授权边界执行发布前，解析字段是否需要额外合规权限。

        这里只读取同 tenant 的 proposal 和其已发布 schema；真正发布仍会完整重验
        target、schema revision、规范值和 chunk Evidence，避免把授权探测当成状态校验。
        """

        row = self._get(value_id)
        if row.status != "proposed":
            raise ResourceConflictError("only proposed metadata value can be published")
        try:
            target = MetadataTarget.model_validate(row.target_json)
        except ValueError as exc:
            raise ResourceConflictError("persisted metadata target is invalid") from exc
        _, published_schema = self._load_published_schema(
            row.schema_resource_id,
            kb_id=target.kb_id,
        )
        try:
            definition = published_schema.field(row.field_key)
        except KeyError as exc:
            raise ResourceConflictError("metadata field was not found") from exc
        return bool(definition.compliance_controlled)

    def reject(
        self,
        value_id: str,
        *,
        review_reason: str = "rejected",
    ) -> dict[str, Any]:
        row = self._get(value_id)
        if row.status != "proposed":
            raise ResourceConflictError("only proposed metadata value can be rejected")
        row.status = "rejected"
        row.reviewed_by = self.actor_id
        row.reviewed_at = _now()
        row.review_reason = review_reason
        self.session.flush()
        return self.as_dict(row)

    def _get(self, value_id: str) -> MetadataValueRecordModel:
        row = self.session.scalar(
            select(MetadataValueRecordModel).where(
                MetadataValueRecordModel.id == value_id,
                MetadataValueRecordModel.tenant_id == self.tenant_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError("metadata value not found")
        return row

    @staticmethod
    def as_dict(row: MetadataValueRecordModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "schema_resource_id": row.schema_resource_id,
            "schema_id": row.schema_id,
            "schema_revision": row.schema_revision,
            "scope_type": row.scope_type,
            "scope_id": row.scope_id,
            "target": row.target_json,
            "field_key": row.field_key,
            "value": row.value_json,
            "value_hash": row.value_hash,
            "source": row.source,
            "confidence": row.confidence,
            "status": row.status,
            "revision": row.revision,
            "created_by": row.created_by,
            "proposed_at": row.proposed_at,
            "reviewed_by": row.reviewed_by,
            "reviewed_at": row.reviewed_at,
            "review_reason": row.review_reason,
            "enhancement": row.enhancement_json,
            "provenance": row.provenance_json,
            "supersedes_id": row.supersedes_id,
        }


class ConnectorStateRepository:
    """固定 tenant/KB 的连接器映射与游标仓储。"""

    def __init__(self, session: Session, *, tenant_id: str, kb_id: str) -> None:
        self.session = session
        self.tenant_id = tenant_id
        self.kb_id = kb_id
        kb = self.session.scalar(
            select(KnowledgeBaseModel.id).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == tenant_id,
                KnowledgeBaseModel.deleted_at.is_(None),
            )
        )
        if kb is None:
            raise ResourceNotFoundError("knowledge base not found in connector scope")

    def _assert_run(self, run_id: str | None) -> None:
        if run_id is None:
            return
        run = self.session.scalar(
            select(CapabilityRunModel.id).where(
                CapabilityRunModel.id == run_id,
                CapabilityRunModel.tenant_id == self.tenant_id,
            )
        )
        if run is None:
            raise ResourceNotFoundError("connector capability run not found")

    def _assert_document_scope(
        self,
        *,
        document_id: str,
        active_document_version_id: str | None,
        pending_document_version_id: str | None,
    ) -> None:
        document = self.session.scalar(
            select(DocumentModel.id).where(
                DocumentModel.id == document_id,
                DocumentModel.kb_id == self.kb_id,
            )
        )
        if document is None:
            raise ResourceNotFoundError("connector document not found in KB scope")
        for version_id in (active_document_version_id, pending_document_version_id):
            if version_id is None:
                continue
            version = self.session.scalar(
                select(DocumentVersionModel.id).where(
                    DocumentVersionModel.id == version_id,
                    DocumentVersionModel.document_id == document_id,
                )
            )
            if version is None:
                raise ResourceNotFoundError(
                    "connector document version not found in document scope"
                )

    def get_binding(self, data_source_id: str, source_item_id: str) -> ConnectorBindingModel | None:
        return self.session.scalar(
            select(ConnectorBindingModel).where(
                ConnectorBindingModel.tenant_id == self.tenant_id,
                ConnectorBindingModel.kb_id == self.kb_id,
                ConnectorBindingModel.data_source_id == data_source_id,
                ConnectorBindingModel.source_item_id == source_item_id,
            )
        )

    def upsert_binding(
        self,
        *,
        data_source_id: str,
        source_item_id: str,
        remote_version: str,
        content_sha256: str | None,
        logical_name: str,
        locator: str,
        status: str,
        document_id: str,
        active_document_version_id: str | None = None,
        pending_document_version_id: str | None = None,
        last_sync_run_id: str | None = None,
        last_seen_at: datetime | None = None,
        disabled_at: datetime | None = None,
        purge_after: datetime | None = None,
    ) -> dict[str, Any]:
        if status not in {
            "pending",
            "active",
            "disabled",
            "soft_deleted",
            "purged",
            "failed",
        }:
            raise ResourceConflictError("unsupported connector binding status")
        if content_sha256 is not None and (
            len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
        ):
            raise ResourceConflictError("connector content_sha256 must be lowercase hex")
        self._assert_document_scope(
            document_id=document_id,
            active_document_version_id=active_document_version_id,
            pending_document_version_id=pending_document_version_id,
        )
        self._assert_run(last_sync_run_id)
        row = self.get_binding(data_source_id, source_item_id)
        if row is None:
            row = ConnectorBindingModel(
                id=f"bind_{uuid4().hex}",
                tenant_id=self.tenant_id,
                kb_id=self.kb_id,
                data_source_id=data_source_id,
                source_item_id=source_item_id,
                document_id=document_id,
                remote_version=remote_version,
                content_sha256=content_sha256,
                logical_name=logical_name,
                locator=locator,
                status=status,
            )
            self.session.add(row)
        row.remote_version = remote_version
        row.content_sha256 = content_sha256
        row.logical_name = logical_name
        row.locator = locator
        row.status = status
        row.document_id = document_id
        row.active_document_version_id = active_document_version_id
        row.pending_document_version_id = pending_document_version_id
        row.last_sync_run_id = last_sync_run_id
        row.last_seen_at = last_seen_at or _now()
        row.disabled_at = disabled_at
        row.purge_after = purge_after
        self.session.flush()
        return self.binding_dict(row)

    def save_binding(self, binding: SourceItemBinding) -> dict[str, Any]:
        if binding.scope.tenant_id != self.tenant_id or binding.scope.kb_id != self.kb_id:
            raise ResourceConflictError("connector binding scope mismatch")
        return self.upsert_binding(
            data_source_id=binding.data_source_id,
            source_item_id=binding.source_item_id,
            remote_version=binding.remote_version,
            content_sha256=binding.content_sha256,
            logical_name=binding.logical_name,
            locator=binding.locator,
            status=binding.status.value,
            document_id=binding.document_id,
            active_document_version_id=binding.active_document_version_id,
            pending_document_version_id=binding.pending_document_version_id,
            last_seen_at=binding.last_seen_at,
            disabled_at=binding.disabled_at,
            purge_after=binding.purge_after,
        )

    def get_cursor(self, data_source_id: str) -> dict[str, Any] | None:
        row = self.session.scalar(
            select(ConnectorCursorModel).where(
                ConnectorCursorModel.tenant_id == self.tenant_id,
                ConnectorCursorModel.kb_id == self.kb_id,
                ConnectorCursorModel.data_source_id == data_source_id,
            )
        )
        return self.cursor_dict(row) if row else None

    def acknowledge_cursor(
        self,
        *,
        data_source_id: str,
        value: str,
        expected_generation: int,
        last_run_id: str | None = None,
        committed_at: datetime | None = None,
    ) -> dict[str, Any]:
        if expected_generation < 0:
            raise ResourceConflictError("connector cursor generation cannot be negative")
        self._assert_run(last_run_id)
        observed_at = committed_at or _now()
        row = self.session.scalar(
            select(ConnectorCursorModel).where(
                ConnectorCursorModel.tenant_id == self.tenant_id,
                ConnectorCursorModel.kb_id == self.kb_id,
                ConnectorCursorModel.data_source_id == data_source_id,
            )
        )
        if row is None:
            if expected_generation != 0:
                raise ResourceConflictError("connector cursor generation mismatch")
            row = ConnectorCursorModel(
                id=f"cur_{uuid4().hex}",
                tenant_id=self.tenant_id,
                kb_id=self.kb_id,
                data_source_id=data_source_id,
                cursor_value=value,
                cursor_hash=_text_hash(value),
                generation=1,
                last_run_id=last_run_id,
                committed_at=observed_at,
            )
            self.session.add(row)
            self.session.flush()
        else:
            result = self.session.execute(
                update(ConnectorCursorModel)
                .where(
                    ConnectorCursorModel.id == row.id,
                    ConnectorCursorModel.tenant_id == self.tenant_id,
                    ConnectorCursorModel.kb_id == self.kb_id,
                    ConnectorCursorModel.generation == expected_generation,
                )
                .values(
                    cursor_value=value,
                    cursor_hash=_text_hash(value),
                    generation=expected_generation + 1,
                    last_run_id=last_run_id,
                    committed_at=observed_at,
                )
            )
            if result.rowcount != 1:
                raise ResourceConflictError("connector cursor generation mismatch")
            self.session.flush()
            self.session.expire(row)
        return self.cursor_dict(row)

    def save_cursor(
        self,
        cursor: SyncCursor,
        *,
        expected_generation: int,
        last_run_id: str | None = None,
    ) -> dict[str, Any]:
        if cursor.scope.tenant_id != self.tenant_id or cursor.scope.kb_id != self.kb_id:
            raise ResourceConflictError("connector cursor scope mismatch")
        if cursor.generation != expected_generation + 1:
            raise ResourceConflictError(
                "cursor generation must advance exactly once from expected generation"
            )
        return self.acknowledge_cursor(
            data_source_id=cursor.data_source_id,
            value=cursor.value,
            expected_generation=expected_generation,
            last_run_id=last_run_id,
            committed_at=cursor.committed_at,
        )

    @staticmethod
    def binding_dict(row: ConnectorBindingModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "data_source_id": row.data_source_id,
            "source_item_id": row.source_item_id,
            "document_id": row.document_id,
            "remote_version": row.remote_version,
            "content_sha256": row.content_sha256,
            "logical_name": row.logical_name,
            "locator": row.locator,
            "status": row.status,
            "active_document_version_id": row.active_document_version_id,
            "pending_document_version_id": row.pending_document_version_id,
            "last_sync_run_id": row.last_sync_run_id,
            "last_seen_at": _iso(row.last_seen_at),
            "disabled_at": _iso(row.disabled_at),
            "purge_after": _iso(row.purge_after),
        }

    @staticmethod
    def cursor_dict(row: ConnectorCursorModel) -> dict[str, Any]:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "data_source_id": row.data_source_id,
            "value": row.cursor_value,
            "cursor_hash": row.cursor_hash,
            "generation": row.generation,
            "last_run_id": row.last_run_id,
            "committed_at": _iso(row.committed_at),
        }


class LongTermMemoryRepository:
    """构造时固定 tenant/subject/KB，管理员身份不能自动扩大正文读取范围。"""

    def __init__(
        self,
        session: Session,
        *,
        tenant_id: str,
        subject_user_id: str,
        kb_id: str,
    ) -> None:
        self.session = session
        self.tenant_id = tenant_id
        self.subject_user_id = subject_user_id
        self.kb_id = kb_id
        kb = self.session.scalar(
            select(KnowledgeBaseModel.id).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == tenant_id,
                KnowledgeBaseModel.deleted_at.is_(None),
            )
        )
        if kb is None:
            raise ResourceNotFoundError("knowledge base not found in memory scope")

    def _validate_scope(self, memory: LongTermMemory) -> None:
        if memory.namespace.tenant_id != self.tenant_id:
            raise ResourceConflictError("memory tenant scope mismatch")
        if memory.namespace.subject_user_id != self.subject_user_id:
            raise ResourceConflictError("memory subject scope mismatch")
        if memory.namespace.kb_id != self.kb_id:
            raise ResourceConflictError("memory knowledge-base scope mismatch")

    @staticmethod
    def _content_hash(content: str) -> str:
        return f"sha256:{_text_hash(content)}"

    def save(self, memory: LongTermMemory | dict[str, Any]) -> dict[str, Any]:
        domain = (
            memory if isinstance(memory, LongTermMemory) else LongTermMemory.model_validate(memory)
        )
        self._validate_scope(domain)
        row = self.session.get(LongTermMemoryModel, domain.memory_id)
        if row is not None:
            if (
                row.tenant_id != self.tenant_id
                or row.subject_user_id != self.subject_user_id
                or row.kb_id != self.kb_id
            ):
                raise ResourceNotFoundError("memory not found")
            if not self._same_snapshot(row, domain):
                raise ResourceConflictError("memory identity already has another snapshot")
            return self.as_dict(row)
        if domain.supersedes_memory_id is not None:
            self.get(domain.supersedes_memory_id)
        row = LongTermMemoryModel(
            id=domain.memory_id,
            tenant_id=self.tenant_id,
            subject_user_id=self.subject_user_id,
            kb_id=self.kb_id,
            candidate_id=domain.candidate_id,
            memory_type=domain.type.value,
            content=domain.content,
            content_hash=self._content_hash(domain.content),
            structured_value=domain.structured_value,
            conflict_key=domain.conflict_key,
            provenance_json=domain.provenance.model_dump(mode="json"),
            confidence=domain.confidence,
            sensitivity=domain.sensitivity.value,
            consent=domain.consent.value,
            status=domain.status.value,
            valid_from=domain.valid_from,
            valid_to=domain.valid_to,
            expires_at=domain.expires_at,
            created_by=domain.created_by,
            embedding_version=domain.embedding_version,
            supersedes_id=domain.supersedes_memory_id,
            forgotten_at=domain.forgotten_at,
            created_at=domain.created_at,
            updated_at=domain.updated_at,
        )
        self.session.add(row)
        self.session.flush()
        return self.as_dict(row)

    @classmethod
    def _same_snapshot(
        cls,
        row: LongTermMemoryModel,
        memory: LongTermMemory,
    ) -> bool:
        return (
            row.candidate_id == memory.candidate_id
            and row.memory_type == memory.type.value
            and row.content == memory.content
            and row.structured_value == memory.structured_value
            and row.conflict_key == memory.conflict_key
            and row.provenance_json == memory.provenance.model_dump(mode="json")
            and row.confidence == memory.confidence
            and row.sensitivity == memory.sensitivity.value
            and row.consent == memory.consent.value
            and row.status == memory.status.value
            and _same_instant(row.valid_from, memory.valid_from)
            and _same_instant(row.valid_to, memory.valid_to)
            and _same_instant(row.expires_at, memory.expires_at)
            and row.created_by == memory.created_by
            and row.embedding_version == memory.embedding_version
            and row.supersedes_id == memory.supersedes_memory_id
            and _same_instant(row.forgotten_at, memory.forgotten_at)
            and _same_instant(row.created_at, memory.created_at)
            and _same_instant(row.updated_at, memory.updated_at)
        )

    def get(self, memory_id: str) -> LongTermMemoryModel:
        row = self.session.scalar(
            select(LongTermMemoryModel).where(
                LongTermMemoryModel.id == memory_id,
                LongTermMemoryModel.tenant_id == self.tenant_id,
                LongTermMemoryModel.subject_user_id == self.subject_user_id,
                LongTermMemoryModel.kb_id == self.kb_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError("memory not found")
        return row

    def replace(self, memory: LongTermMemory | dict[str, Any]) -> dict[str, Any]:
        domain = (
            memory if isinstance(memory, LongTermMemory) else LongTermMemory.model_validate(memory)
        )
        self._validate_scope(domain)
        row = self.get(domain.memory_id)
        if (
            row.candidate_id != domain.candidate_id
            or row.memory_type != domain.type.value
            or row.conflict_key != domain.conflict_key
        ):
            raise ResourceConflictError("memory identity fields are immutable")
        allowed = {
            "proposed": {"proposed", "confirmed", "active", "rejected", "forgotten", "expired"},
            "confirmed": {"confirmed", "active", "rejected", "forgotten", "expired"},
            "active": {"active", "superseded", "forgotten", "expired"},
            "superseded": {"superseded", "forgotten"},
            "rejected": {"rejected", "forgotten"},
            "expired": {"expired", "forgotten"},
            "forgotten": {"forgotten"},
        }
        if domain.status.value not in allowed.get(row.status, set()):
            raise ResourceConflictError(
                f"invalid memory transition {row.status}->{domain.status.value}"
            )
        if domain.supersedes_memory_id is not None:
            self.get(domain.supersedes_memory_id)
        redacted = domain.status.value == "forgotten"
        content_hash = (
            row.content_hash
            if redacted or not domain.content
            else self._content_hash(domain.content)
        )
        for field, value in {
            "content": "" if redacted else domain.content,
            "content_hash": content_hash,
            "structured_value": {} if redacted else domain.structured_value,
            "provenance_json": {} if redacted else domain.provenance.model_dump(mode="json"),
            "confidence": domain.confidence,
            "sensitivity": domain.sensitivity.value,
            "consent": domain.consent.value,
            "status": domain.status.value,
            "valid_from": domain.valid_from,
            "valid_to": domain.valid_to,
            "expires_at": domain.expires_at,
            "embedding_version": None if redacted else domain.embedding_version,
            "supersedes_id": domain.supersedes_memory_id,
            "forgotten_at": domain.forgotten_at,
            "updated_at": domain.updated_at,
        }.items():
            setattr(row, field, value)
        self.session.flush()
        return self.as_dict(row)

    def list_recallable(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or _now()
        rows = self.session.scalars(
            select(LongTermMemoryModel)
            .where(
                LongTermMemoryModel.tenant_id == self.tenant_id,
                LongTermMemoryModel.subject_user_id == self.subject_user_id,
                LongTermMemoryModel.kb_id == self.kb_id,
                LongTermMemoryModel.status == "active",
                LongTermMemoryModel.consent == "confirmed",
                LongTermMemoryModel.expires_at > current,
            )
            .order_by(
                LongTermMemoryModel.confidence.desc(),
                LongTermMemoryModel.updated_at.desc(),
            )
        ).all()
        return [self.as_dict(row) for row in rows]

    def forget(self, memory_id: str, *, at: datetime | None = None) -> dict[str, Any]:
        row = self.get(memory_id)
        if row.status == "forgotten":
            return self.as_dict(row)
        row.status = "forgotten"
        row.consent = "withdrawn"
        row.forgotten_at = at or _now()
        row.valid_to = row.forgotten_at
        # 保留最小审计身份/hash，不保留正文、结构值、消息引用或向量版本。
        row.content = ""
        row.structured_value = {}
        row.provenance_json = {}
        row.embedding_version = None
        self.session.flush()
        return self.as_dict(row)

    @staticmethod
    def as_dict(row: LongTermMemoryModel) -> dict[str, Any]:
        return {
            "memory_id": row.id,
            "tenant_id": row.tenant_id,
            "subject_user_id": row.subject_user_id,
            "kb_id": row.kb_id,
            "namespace": {
                "tenant_id": row.tenant_id,
                "subject_user_id": row.subject_user_id,
                "kb_id": row.kb_id,
            },
            "candidate_id": row.candidate_id,
            "type": row.memory_type,
            "memory_type": row.memory_type,
            "content": row.content,
            "content_hash": row.content_hash,
            "structured_value": row.structured_value,
            "conflict_key": row.conflict_key,
            "provenance": row.provenance_json,
            "confidence": row.confidence,
            "sensitivity": row.sensitivity,
            "consent": row.consent,
            "status": row.status,
            "valid_from": row.valid_from,
            "valid_to": row.valid_to,
            "expires_at": row.expires_at,
            "created_by": row.created_by,
            "embedding_version": row.embedding_version,
            "supersedes_memory_id": row.supersedes_id,
            "supersedes_id": row.supersedes_id,
            "forgotten_at": row.forgotten_at,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


class MediaAssetRepository:
    def __init__(
        self,
        session: Session,
        *,
        tenant_id: str,
        kb_id: str | None = None,
        subject_user_id: str | None = None,
    ) -> None:
        self.session = session
        self.tenant_id = tenant_id
        self.kb_id = kb_id
        self.subject_user_id = subject_user_id
        if kb_id is not None:
            kb = self.session.scalar(
                select(KnowledgeBaseModel.id).where(
                    KnowledgeBaseModel.id == kb_id,
                    KnowledgeBaseModel.tenant_id == tenant_id,
                    KnowledgeBaseModel.deleted_at.is_(None),
                )
            )
            if kb is None:
                raise ResourceNotFoundError("knowledge base not found in media scope")

    def _assert_document_scope(self, asset: MediaAsset) -> None:
        if asset.source_kind != "document_asset":
            return
        kb = self.session.scalar(
            select(KnowledgeBaseModel.id).where(
                KnowledgeBaseModel.id == asset.kb_id,
                KnowledgeBaseModel.tenant_id == self.tenant_id,
                KnowledgeBaseModel.deleted_at.is_(None),
            )
        )
        if kb is None:
            raise ResourceNotFoundError("media knowledge base not found in tenant scope")
        document = self.session.scalar(
            select(DocumentModel.id).where(
                DocumentModel.id == asset.document_id,
                DocumentModel.kb_id == asset.kb_id,
                DocumentModel.deleted_at.is_(None),
            )
        )
        if document is None:
            raise ResourceNotFoundError("media document not found in KB scope")
        version = self.session.scalar(
            select(DocumentVersionModel.id).where(
                DocumentVersionModel.id == asset.document_version_id,
                DocumentVersionModel.document_id == asset.document_id,
            )
        )
        if version is None:
            raise ResourceNotFoundError("media document version not found in document scope")

    def save(self, asset: MediaAsset) -> dict[str, Any]:
        if asset.tenant_id != self.tenant_id:
            raise ResourceConflictError("media asset tenant scope mismatch")
        if asset.source_kind == "document_asset" and asset.kb_id != self.kb_id:
            raise ResourceConflictError("document media knowledge-base scope mismatch")
        if asset.source_kind == "query_attachment" and asset.owner_user_id != self.subject_user_id:
            raise ResourceConflictError("query attachment owner scope mismatch")
        self._assert_document_scope(asset)
        existing = self.session.get(MediaAssetModel, asset.asset_id)
        payload = asset.model_dump(mode="json")
        if existing is not None:
            if existing.tenant_id != self.tenant_id:
                raise ResourceNotFoundError("media asset not found")
            if existing.source_kind == "document_asset" and existing.kb_id != self.kb_id:
                raise ResourceNotFoundError("media asset not found")
            if (
                existing.source_kind == "query_attachment"
                and existing.owner_user_id != self.subject_user_id
            ):
                raise ResourceNotFoundError("media asset not found")
            if existing.asset_json != payload:
                raise ResourceConflictError("media asset identity collision")
            return self.as_dict(existing)
        row = MediaAssetModel(
            id=asset.asset_id,
            tenant_id=asset.tenant_id,
            kb_id=asset.kb_id,
            document_id=asset.document_id,
            document_version_id=asset.document_version_id,
            owner_user_id=asset.owner_user_id,
            source_kind=asset.source_kind,
            modality=asset.modality,
            media_type=asset.media_type,
            content_hash=asset.content_hash,
            size_bytes=asset.size_bytes,
            object_key=asset.object_key,
            classification=asset.classification,
            status=asset.status,
            location_json=asset.location.model_dump(mode="json"),
            asset_json=payload,
            derivations_json=[],
        )
        self.session.add(row)
        self.session.flush()
        return self.as_dict(row)

    def append_derivation(self, asset_id: str, derivation: MediaDerivation) -> dict[str, Any]:
        row = self._get(asset_id)
        if row.status != "active":
            raise ResourceConflictError("derivation can only be appended to an active media asset")
        if derivation.asset_id != asset_id:
            raise ResourceConflictError("derivation asset scope mismatch")
        payload = derivation.model_dump(mode="json")
        existing = {item.get("derivation_id"): item for item in row.derivations_json or []}
        current = existing.get(derivation.derivation_id)
        if current is not None and current != payload:
            raise ResourceConflictError("derivation identity collision")
        existing[derivation.derivation_id] = payload
        row.derivations_json = [existing[key] for key in sorted(existing)]
        self.session.flush()
        return self.as_dict(row)

    def _get(self, asset_id: str) -> MediaAssetModel:
        criteria = [
            MediaAssetModel.id == asset_id,
            MediaAssetModel.tenant_id == self.tenant_id,
        ]
        scope_criteria = []
        if self.kb_id is not None:
            scope_criteria.append(
                (MediaAssetModel.source_kind == "document_asset")
                & (MediaAssetModel.kb_id == self.kb_id)
            )
        if self.subject_user_id is not None:
            scope_criteria.append(
                (MediaAssetModel.source_kind == "query_attachment")
                & (MediaAssetModel.owner_user_id == self.subject_user_id)
            )
        if not scope_criteria:
            raise ResourceNotFoundError("media asset not found")
        allowed_scope = scope_criteria[0]
        for criterion in scope_criteria[1:]:
            allowed_scope = allowed_scope | criterion
        criteria.append(allowed_scope)
        row = self.session.scalar(select(MediaAssetModel).where(*criteria))
        if row is None:
            raise ResourceNotFoundError("media asset not found")
        return row

    def retire(self, asset_id: str) -> dict[str, Any]:
        row = self._get(asset_id)
        row.status = "retired"
        row.retired_at = _now()
        row.asset_json = {**row.asset_json, "status": "retired"}
        self.session.flush()
        return self.as_dict(row)

    @staticmethod
    def as_dict(row: MediaAssetModel) -> dict[str, Any]:
        return {
            "asset": row.asset_json,
            "derivations": row.derivations_json,
            "status": row.status,
            "retired_at": _iso(row.retired_at),
        }


__all__ = [
    "CapabilityResourceRepository",
    "CapabilityRunRepository",
    "ConnectorStateRepository",
    "LongTermMemoryRepository",
    "MediaAssetRepository",
    "MetadataValueRepository",
]
