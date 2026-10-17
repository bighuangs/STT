"""TODO-15 元数据/标签治理用例。

服务只依赖 :class:`MetadataGovernanceRepository`，因此 SQL 控制库可以在后续接入；
本轮的纯内存实现用于离线契约、回填预演和管理命令测试。任何查询过滤和 Milvus
投影都固定到一个 ``MetadataRelease``，不会扫描 chunk 来猜测目录。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from app.domain.knowledge_governance import (
    EnhancementReference,
    EnhancementRun,
    EnhancementRunStatus,
    MappedMetadataFilter,
    MappedMetadataGroup,
    MappedMetadataPredicate,
    MetadataAuditEvent,
    MetadataCatalog,
    MetadataCatalogEntry,
    MetadataFieldDefinition,
    MetadataFieldQuality,
    MetadataFilterPlan,
    MetadataMigrationIssue,
    MetadataMigrationPreview,
    MetadataProjectionBatch,
    MetadataQualityReport,
    MetadataRelease,
    MetadataSchemaVersion,
    MetadataTarget,
    MetadataValueRecord,
    ProjectionState,
    ProjectionTag,
    ProjectionValue,
    TagAssignmentRecord,
    TagDefinition,
    TagTaxonomyVersion,
    UnknownTagCandidate,
    ValueSource,
    stable_governance_id,
)
from app.domain.retrieval import (
    FilterGroup,
    FilterNode,
    FilterPredicate,
    validate_filter_ast,
)


class GovernanceError(RuntimeError):
    """管理 API/命令可稳定映射的业务错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


@runtime_checkable
class MetadataGovernanceRepository(Protocol):
    """控制库存储端口；实现必须保证 ``next_revision`` 原子递增。"""

    def next_revision(self, tenant_id: str, kb_id: str) -> int: ...

    def save_schema(self, schema: MetadataSchemaVersion) -> None: ...

    def get_schema(
        self,
        tenant_id: str,
        kb_id: str,
        schema_id: str,
        revision: int,
    ) -> MetadataSchemaVersion | None: ...

    def list_schemas(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataSchemaVersion, ...]: ...

    def save_taxonomy(self, taxonomy: TagTaxonomyVersion) -> None: ...

    def get_taxonomy(
        self,
        tenant_id: str,
        kb_id: str,
        taxonomy_id: str,
        revision: int,
    ) -> TagTaxonomyVersion | None: ...

    def list_taxonomies(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[TagTaxonomyVersion, ...]: ...

    def save_value(self, value: MetadataValueRecord) -> None: ...

    def get_value(self, record_id: str) -> MetadataValueRecord | None: ...

    def list_values(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataValueRecord, ...]: ...

    def save_tag_assignment(self, assignment: TagAssignmentRecord) -> None: ...

    def get_tag_assignment(self, assignment_id: str) -> TagAssignmentRecord | None: ...

    def list_tag_assignments(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[TagAssignmentRecord, ...]: ...

    def save_unknown_tag(self, candidate: UnknownTagCandidate) -> None: ...

    def list_unknown_tags(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[UnknownTagCandidate, ...]: ...

    def save_release(self, release: MetadataRelease) -> None: ...

    def current_release(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> MetadataRelease | None: ...

    def get_release(
        self,
        tenant_id: str,
        kb_id: str,
        revision: int,
    ) -> MetadataRelease | None: ...

    def list_releases(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataRelease, ...]: ...

    def save_projection_state(self, state: ProjectionState) -> None: ...

    def get_projection_state(
        self,
        tenant_id: str,
        kb_id: str,
        projection_name: str,
    ) -> ProjectionState | None: ...

    def save_enhancement_run(self, run: EnhancementRun) -> None: ...

    def get_enhancement_run(self, run_id: str) -> EnhancementRun | None: ...

    def append_audit(self, event: MetadataAuditEvent) -> None: ...

    def list_audit(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataAuditEvent, ...]: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _validated_copy(model, **updates):
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return type(model).model_validate(payload)


class MetadataGovernanceService:
    """版本、审核、发布、回滚、过滤与质量统计的事务边界。"""

    def __init__(
        self,
        repository: MetadataGovernanceRepository,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.repository = repository
        self.clock = clock

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise GovernanceError("INVALID_CLOCK", "governance clock must include timezone")
        return value.astimezone(timezone.utc)

    def _audit(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        action: str,
        actor_id: str,
        object_type: str,
        object_id: str,
        revision: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        occurred_at = self._now()
        payload = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "action": action,
            "actor_id": actor_id,
            "object_type": object_type,
            "object_id": object_id,
            "revision": revision,
            "details": dict(details or {}),
            "occurred_at": occurred_at.isoformat(),
        }
        self.repository.append_audit(
            MetadataAuditEvent(
                event_id=stable_governance_id("ma", payload),
                occurred_at=occurred_at,
                **{key: value for key, value in payload.items() if key != "occurred_at"},
            )
        )

    def create_schema_draft(
        self,
        *,
        schema_id: str,
        tenant_id: str,
        kb_id: str,
        version: int,
        fields: Sequence[MetadataFieldDefinition],
        actor_id: str,
    ) -> MetadataSchemaVersion:
        existing = [
            item
            for item in self.repository.list_schemas(tenant_id, kb_id)
            if item.schema_id == schema_id and item.version == version
        ]
        if existing:
            raise GovernanceError(
                "SCHEMA_VERSION_EXISTS",
                "metadata schema version already exists",
            )
        revision = self.repository.next_revision(tenant_id, kb_id)
        schema = MetadataSchemaVersion(
            schema_id=schema_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            version=version,
            revision=revision,
            status="draft",
            fields=tuple(fields),
            created_by=actor_id,
            created_at=self._now(),
        )
        self.repository.save_schema(schema)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="schema_created",
            actor_id=actor_id,
            object_type="metadata_schema",
            object_id=f"{schema.schema_id}:{schema.revision}",
            revision=revision,
        )
        return schema

    def publish_schema(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        schema_id: str,
        revision: int,
        actor_id: str,
    ) -> MetadataSchemaVersion:
        schema = self.repository.get_schema(tenant_id, kb_id, schema_id, revision)
        if schema is None:
            raise GovernanceError("SCHEMA_NOT_FOUND", "metadata schema was not found")
        if schema.status != "draft":
            raise GovernanceError("SCHEMA_NOT_DRAFT", "only a draft schema can be published")
        for other in self.repository.list_schemas(tenant_id, kb_id):
            if other.status == "published" and other.schema_id == schema_id:
                self.repository.save_schema(_validated_copy(other, status="retired"))
        published = _validated_copy(
            schema,
            status="published",
            published_at=self._now(),
        )
        self.repository.save_schema(published)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="schema_published",
            actor_id=actor_id,
            object_type="metadata_schema",
            object_id=f"{schema_id}:{revision}",
            revision=revision,
        )
        return published

    def create_taxonomy_draft(
        self,
        *,
        taxonomy_id: str,
        tenant_id: str,
        kb_id: str,
        version: int,
        tags: Sequence[TagDefinition],
        actor_id: str,
    ) -> TagTaxonomyVersion:
        if any(
            item.taxonomy_id == taxonomy_id and item.version == version
            for item in self.repository.list_taxonomies(tenant_id, kb_id)
        ):
            raise GovernanceError(
                "TAXONOMY_VERSION_EXISTS",
                "tag taxonomy version already exists",
            )
        revision = self.repository.next_revision(tenant_id, kb_id)
        taxonomy = TagTaxonomyVersion(
            taxonomy_id=taxonomy_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            version=version,
            revision=revision,
            status="draft",
            tags=tuple(tags),
            created_by=actor_id,
            created_at=self._now(),
        )
        self.repository.save_taxonomy(taxonomy)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="taxonomy_created",
            actor_id=actor_id,
            object_type="tag_taxonomy",
            object_id=f"{taxonomy_id}:{revision}",
            revision=revision,
        )
        return taxonomy

    def publish_taxonomy(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        taxonomy_id: str,
        revision: int,
        actor_id: str,
    ) -> TagTaxonomyVersion:
        taxonomy = self.repository.get_taxonomy(
            tenant_id,
            kb_id,
            taxonomy_id,
            revision,
        )
        if taxonomy is None:
            raise GovernanceError("TAXONOMY_NOT_FOUND", "tag taxonomy was not found")
        if taxonomy.status != "draft":
            raise GovernanceError(
                "TAXONOMY_NOT_DRAFT",
                "only a draft taxonomy can be published",
            )
        for other in self.repository.list_taxonomies(tenant_id, kb_id):
            if other.status == "published" and other.taxonomy_id == taxonomy_id:
                self.repository.save_taxonomy(_validated_copy(other, status="retired"))
        published = _validated_copy(
            taxonomy,
            status="published",
            published_at=self._now(),
        )
        self.repository.save_taxonomy(published)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="taxonomy_published",
            actor_id=actor_id,
            object_type="tag_taxonomy",
            object_id=f"{taxonomy_id}:{revision}",
            revision=revision,
        )
        return published

    def _published_schema(
        self,
        tenant_id: str,
        kb_id: str,
        schema_id: str,
        revision: int,
    ) -> MetadataSchemaVersion:
        schema = self.repository.get_schema(
            tenant_id,
            kb_id,
            schema_id,
            revision,
        )
        if schema is None or schema.status != "published":
            raise GovernanceError(
                "SCHEMA_NOT_PUBLISHED",
                "metadata proposals require a published schema",
            )
        return schema

    def _published_taxonomy(
        self,
        tenant_id: str,
        kb_id: str,
        taxonomy_id: str,
        revision: int,
    ) -> TagTaxonomyVersion:
        taxonomy = self.repository.get_taxonomy(
            tenant_id,
            kb_id,
            taxonomy_id,
            revision,
        )
        if taxonomy is None or taxonomy.status != "published":
            raise GovernanceError(
                "TAXONOMY_NOT_PUBLISHED",
                "tag proposals require a published taxonomy",
            )
        return taxonomy

    def propose_value(
        self,
        *,
        schema_id: str,
        schema_revision: int,
        target: MetadataTarget,
        field_key: str,
        value: Any,
        source: ValueSource,
        proposed_by: str,
        confidence: float | None = None,
        enhancement: EnhancementReference | None = None,
    ) -> MetadataValueRecord:
        schema = self._published_schema(
            target.tenant_id,
            target.kb_id,
            schema_id,
            schema_revision,
        )
        try:
            definition = schema.field(field_key)
        except KeyError as exc:
            raise GovernanceError("FIELD_NOT_FOUND", "metadata field was not found") from exc
        if definition.scope != target.scope:
            raise GovernanceError(
                "FIELD_SCOPE_MISMATCH",
                "metadata target scope does not match its schema field",
            )
        try:
            normalized = definition.normalize_value(value)
        except ValueError as exc:
            raise GovernanceError("INVALID_METADATA_VALUE", str(exc)) from exc
        payload = {
            "schema_id": schema_id,
            "schema_revision": schema_revision,
            "target": target.model_dump(mode="json"),
            "field_key": field_key,
            "value": normalized,
            "source": source,
            "proposed_by": proposed_by,
            "enhancement_run_id": enhancement.run_id if enhancement else None,
        }
        record_id = stable_governance_id("mv", payload)
        existing = self.repository.get_value(record_id)
        if existing is not None:
            return existing
        record = MetadataValueRecord(
            record_id=record_id,
            schema_id=schema_id,
            schema_revision=schema_revision,
            field_key=field_key,
            target=target,
            value=normalized,
            source=source,
            confidence=confidence,
            proposed_by=proposed_by,
            proposed_at=self._now(),
            enhancement=enhancement,
        )
        self.repository.save_value(record)
        self._audit(
            tenant_id=target.tenant_id,
            kb_id=target.kb_id,
            action="value_proposed",
            actor_id=proposed_by,
            object_type="metadata_value",
            object_id=record.record_id,
            details={"field_key": field_key, "source": source},
        )
        return record

    def propose_tag(
        self,
        *,
        taxonomy_id: str,
        taxonomy_revision: int,
        target: MetadataTarget,
        label: str,
        source: Literal["auto", "import", "manual"],
        proposed_by: str,
        confidence: float | None = None,
        enhancement: EnhancementReference | None = None,
    ) -> TagAssignmentRecord | UnknownTagCandidate:
        taxonomy = self._published_taxonomy(
            target.tenant_id,
            target.kb_id,
            taxonomy_id,
            taxonomy_revision,
        )
        tag = taxonomy.resolve(label)
        if tag is None:
            if source == "manual":
                raise GovernanceError(
                    "UNKNOWN_MANUAL_TAG",
                    "manual assignments must select a published tag definition",
                )
            payload = {
                "taxonomy_id": taxonomy_id,
                "taxonomy_revision": taxonomy_revision,
                "target": target.model_dump(mode="json"),
                "raw_label": label,
                "source": source,
                "enhancement_run_id": enhancement.run_id if enhancement else None,
            }
            candidate = UnknownTagCandidate(
                candidate_id=stable_governance_id("tc", payload),
                taxonomy_id=taxonomy_id,
                taxonomy_revision=taxonomy_revision,
                target=target,
                raw_label=label,
                normalized_label=label.strip().casefold(),
                source=source,
                confidence=confidence,
                proposed_at=self._now(),
                enhancement=enhancement,
            )
            self.repository.save_unknown_tag(candidate)
            return candidate

        payload = {
            "taxonomy_id": taxonomy_id,
            "taxonomy_revision": taxonomy_revision,
            "target": target.model_dump(mode="json"),
            "tag_id": tag.tag_id,
            "source": source,
            "proposed_by": proposed_by,
            "enhancement_run_id": enhancement.run_id if enhancement else None,
        }
        assignment_id = stable_governance_id("ta", payload)
        existing = self.repository.get_tag_assignment(assignment_id)
        if existing is not None:
            return existing
        assignment = TagAssignmentRecord(
            assignment_id=assignment_id,
            taxonomy_id=taxonomy_id,
            taxonomy_revision=taxonomy_revision,
            tag_id=tag.tag_id,
            target=target,
            source=source,
            confidence=confidence,
            proposed_by=proposed_by,
            proposed_at=self._now(),
            enhancement=enhancement,
        )
        self.repository.save_tag_assignment(assignment)
        self._audit(
            tenant_id=target.tenant_id,
            kb_id=target.kb_id,
            action="tag_proposed",
            actor_id=proposed_by,
            object_type="tag_assignment",
            object_id=assignment.assignment_id,
            details={"tag_id": tag.tag_id, "source": source},
        )
        return assignment

    @staticmethod
    def _value_key(record: MetadataValueRecord) -> tuple[str, str]:
        return record.target.target_key, record.field_key

    @staticmethod
    def _tag_key(record: TagAssignmentRecord) -> tuple[str, str]:
        return record.target.target_key, record.tag_id

    def reject_values(
        self,
        record_ids: Sequence[str],
        *,
        reviewer_id: str,
        reason: str,
    ) -> tuple[MetadataValueRecord, ...]:
        if not record_ids:
            return ()
        records = self._load_value_proposals(record_ids)
        now = self._now()
        rejected = tuple(
            _validated_copy(
                record,
                status="rejected",
                reviewed_by=reviewer_id,
                reviewed_at=now,
                review_reason=reason,
            )
            for record in records
        )
        for record in rejected:
            self.repository.save_value(record)
            self._audit(
                tenant_id=record.target.tenant_id,
                kb_id=record.target.kb_id,
                action="value_rejected",
                actor_id=reviewer_id,
                object_type="metadata_value",
                object_id=record.record_id,
                details={"reason": reason},
            )
        return rejected

    def _load_value_proposals(
        self,
        record_ids: Sequence[str],
    ) -> tuple[MetadataValueRecord, ...]:
        if len(record_ids) != len(set(record_ids)):
            raise GovernanceError("DUPLICATE_RECORD_ID", "record IDs must be unique")
        result: list[MetadataValueRecord] = []
        for record_id in record_ids:
            record = self.repository.get_value(record_id)
            if record is None:
                raise GovernanceError("VALUE_NOT_FOUND", f"metadata value {record_id} not found")
            if record.status != "proposed":
                raise GovernanceError(
                    "VALUE_ALREADY_REVIEWED",
                    f"metadata value {record_id} is not pending review",
                )
            result.append(record)
        return tuple(result)

    def _load_tag_proposals(
        self,
        assignment_ids: Sequence[str],
    ) -> tuple[TagAssignmentRecord, ...]:
        if len(assignment_ids) != len(set(assignment_ids)):
            raise GovernanceError("DUPLICATE_ASSIGNMENT_ID", "assignment IDs must be unique")
        result: list[TagAssignmentRecord] = []
        for assignment_id in assignment_ids:
            assignment = self.repository.get_tag_assignment(assignment_id)
            if assignment is None:
                raise GovernanceError(
                    "TAG_ASSIGNMENT_NOT_FOUND",
                    f"tag assignment {assignment_id} not found",
                )
            if assignment.status != "proposed":
                raise GovernanceError(
                    "TAG_ALREADY_REVIEWED",
                    f"tag assignment {assignment_id} is not pending review",
                )
            result.append(assignment)
        return tuple(result)

    def reject_tags(
        self,
        assignment_ids: Sequence[str],
        *,
        reviewer_id: str,
        reason: str,
    ) -> tuple[TagAssignmentRecord, ...]:
        if not assignment_ids:
            return ()
        assignments = self._load_tag_proposals(assignment_ids)
        now = self._now()
        rejected = tuple(
            _validated_copy(
                assignment,
                status="rejected",
                reviewed_by=reviewer_id,
                reviewed_at=now,
                review_reason=reason,
            )
            for assignment in assignments
        )
        for assignment in rejected:
            self.repository.save_tag_assignment(assignment)
        return rejected

    def publish_proposals(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        schema_id: str,
        schema_revision: int,
        reviewer_id: str,
        value_record_ids: Sequence[str] = (),
        tag_assignment_ids: Sequence[str] = (),
        taxonomy_id: str | None = None,
        taxonomy_revision: int | None = None,
        allow_compliance_controlled: bool = False,
        migration_from_release_id: str | None = None,
    ) -> MetadataRelease:
        """原子审核并发布一个新 release。

        自动建议无法覆盖当前 release 的人工值；这种建议会被明确标为 rejected，
        而不是悄悄覆盖或丢弃。合规字段还要求调用方声明独立的高权限校验结果。
        """

        schema = self._published_schema(
            tenant_id,
            kb_id,
            schema_id,
            schema_revision,
        )
        current = self.repository.current_release(tenant_id, kb_id)
        schema_changed = current is not None and (
            current.metadata_schema_id != schema_id
            or current.metadata_schema_revision != schema_revision
        )
        if schema_changed and migration_from_release_id != current.release_id:
            raise GovernanceError(
                "SCHEMA_MIGRATION_REQUIRED",
                "changing a release schema requires an explicit metadata migration",
            )
        if (taxonomy_id is None) != (taxonomy_revision is None):
            raise GovernanceError(
                "INVALID_TAXONOMY_REFERENCE",
                "taxonomy ID and revision must be supplied together",
            )
        taxonomy = (
            self._published_taxonomy(
                tenant_id,
                kb_id,
                taxonomy_id,
                taxonomy_revision,
            )
            if taxonomy_id is not None and taxonomy_revision is not None
            else None
        )
        if taxonomy is None and current is not None:
            taxonomy = self._release_taxonomy(current)
        if (
            current is not None
            and current.tag_assignment_ids
            and taxonomy is not None
            and (
                taxonomy.taxonomy_id != current.taxonomy_id
                or taxonomy.revision != current.taxonomy_revision
            )
        ):
            raise GovernanceError(
                "TAXONOMY_MIGRATION_REQUIRED",
                "changing a release taxonomy requires an explicit tag migration",
            )
        values = self._load_value_proposals(value_record_ids)
        tags = self._load_tag_proposals(tag_assignment_ids)
        scoped = [*values, *tags]
        if any(item.target.tenant_id != tenant_id or item.target.kb_id != kb_id for item in scoped):
            raise GovernanceError(
                "CROSS_SCOPE_PUBLICATION",
                "all proposals must belong to the release tenant and KB",
            )
        if any(
            item.schema_id != schema_id or item.schema_revision != schema_revision
            for item in values
        ):
            raise GovernanceError(
                "SCHEMA_REVISION_MISMATCH",
                "all metadata values must use the release schema revision",
            )
        if taxonomy is None and tags:
            raise GovernanceError(
                "TAXONOMY_REQUIRED",
                "tag assignments require a taxonomy on the release",
            )
        if taxonomy is not None and any(
            item.taxonomy_id != taxonomy.taxonomy_id or item.taxonomy_revision != taxonomy.revision
            for item in tags
        ):
            raise GovernanceError(
                "TAXONOMY_REVISION_MISMATCH",
                "all tag assignments must use the release taxonomy revision",
            )
        for record in values:
            definition = schema.field(record.field_key)
            if definition.compliance_controlled and not allow_compliance_controlled:
                raise GovernanceError(
                    "COMPLIANCE_PERMISSION_REQUIRED",
                    f"publishing {record.field_key} requires compliance permission",
                )

        current_values = (
            {}
            if schema_changed
            else {self._value_key(record): record for record in self._release_values(current)}
        )
        current_tags = {self._tag_key(record): record for record in self._release_tags(current)}
        now = self._now()
        protected: list[str] = []
        accepted_values: list[MetadataValueRecord] = []
        for proposal in values:
            previous = current_values.get(self._value_key(proposal))
            if previous is not None and previous.source == "manual" and proposal.source == "auto":
                rejected = _validated_copy(
                    proposal,
                    status="rejected",
                    reviewed_by=reviewer_id,
                    reviewed_at=now,
                    review_reason="MANUAL_VALUE_PROTECTED",
                )
                self.repository.save_value(rejected)
                protected.append(proposal.record_id)
                continue
            accepted = _validated_copy(
                proposal,
                status="published",
                reviewed_by=reviewer_id,
                reviewed_at=now,
                review_reason="APPROVED",
            )
            self.repository.save_value(accepted)
            current_values[self._value_key(accepted)] = accepted
            accepted_values.append(accepted)

        accepted_tags: list[TagAssignmentRecord] = []
        for proposal in tags:
            previous = current_tags.get(self._tag_key(proposal))
            if previous is not None and previous.source == "manual" and proposal.source == "auto":
                rejected = _validated_copy(
                    proposal,
                    status="rejected",
                    reviewed_by=reviewer_id,
                    reviewed_at=now,
                    review_reason="MANUAL_TAG_PROTECTED",
                )
                self.repository.save_tag_assignment(rejected)
                protected.append(proposal.assignment_id)
                continue
            accepted = _validated_copy(
                proposal,
                status="published",
                reviewed_by=reviewer_id,
                reviewed_at=now,
                review_reason="APPROVED",
            )
            self.repository.save_tag_assignment(accepted)
            current_tags[self._tag_key(accepted)] = accepted
            accepted_tags.append(accepted)

        revision = self.repository.next_revision(tenant_id, kb_id)
        release_payload = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "revision": revision,
            "schema_id": schema_id,
            "schema_revision": schema_revision,
            "taxonomy_id": taxonomy.taxonomy_id if taxonomy else None,
            "taxonomy_revision": taxonomy.revision if taxonomy else None,
            "values": sorted(item.record_id for item in current_values.values()),
            "tags": sorted(item.assignment_id for item in current_tags.values()),
        }
        release = MetadataRelease(
            release_id=stable_governance_id("mr", release_payload),
            tenant_id=tenant_id,
            kb_id=kb_id,
            revision=revision,
            metadata_schema_id=schema_id,
            metadata_schema_revision=schema_revision,
            taxonomy_id=taxonomy.taxonomy_id if taxonomy else None,
            taxonomy_revision=taxonomy.revision if taxonomy else None,
            value_record_ids=tuple(release_payload["values"]),
            tag_assignment_ids=tuple(release_payload["tags"]),
            published_by=reviewer_id,
            published_at=now,
        )
        self.repository.save_release(release)
        self.repository.save_projection_state(
            ProjectionState(
                tenant_id=tenant_id,
                kb_id=kb_id,
                projection_name="retrieval_metadata",
                control_revision=revision,
                projected_revision=(
                    self._projection_revision(tenant_id, kb_id, "retrieval_metadata")
                ),
                status="pending",
            )
        )
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="release_published",
            actor_id=reviewer_id,
            object_type="metadata_release",
            object_id=release.release_id,
            revision=revision,
            details={
                "accepted_values": [item.record_id for item in accepted_values],
                "accepted_tags": [item.assignment_id for item in accepted_tags],
                "manual_protected": protected,
            },
        )
        return release

    def preview_schema_migration(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        target_schema_id: str,
        target_schema_revision: int,
    ) -> MetadataMigrationPreview:
        """检查当前 release 到目标 schema 的兼容性，不写任何事实。"""

        current = self._current_release(tenant_id, kb_id)
        target = self._published_schema(
            tenant_id,
            kb_id,
            target_schema_id,
            target_schema_revision,
        )
        fields = {item.key: item for item in target.fields}
        compatible: list[str] = []
        issues: list[MetadataMigrationIssue] = []
        populated: set[str] = set()
        for record in self._release_values(current):
            definition = fields.get(record.field_key)
            if definition is None:
                issues.append(
                    MetadataMigrationIssue(
                        record_id=record.record_id,
                        field_key=record.field_key,
                        code="FIELD_REMOVED",
                        message="field does not exist in target schema",
                    )
                )
                continue
            if definition.scope != record.target.scope:
                issues.append(
                    MetadataMigrationIssue(
                        record_id=record.record_id,
                        field_key=record.field_key,
                        code="SCOPE_CHANGED",
                        message="field scope changed and requires an explicit target mapping",
                    )
                )
                continue
            try:
                definition.normalize_value(record.value)
            except ValueError as exc:
                issues.append(
                    MetadataMigrationIssue(
                        record_id=record.record_id,
                        field_key=record.field_key,
                        code="VALUE_INCOMPATIBLE",
                        message=str(exc),
                    )
                )
                continue
            compatible.append(record.record_id)
            populated.add(record.field_key)
        for definition in target.fields:
            if definition.required and definition.key not in populated:
                issues.append(
                    MetadataMigrationIssue(
                        field_key=definition.key,
                        code="REQUIRED_FIELD_UNPOPULATED",
                        message="required target field has no compatible published value",
                    )
                )
        blocking = {"SCOPE_CHANGED", "VALUE_INCOMPATIBLE", "REQUIRED_FIELD_UNPOPULATED"}
        return MetadataMigrationPreview(
            tenant_id=tenant_id,
            kb_id=kb_id,
            source_release_id=current.release_id,
            source_schema_revision=current.metadata_schema_revision,
            target_schema_revision=target.revision,
            compatible_record_ids=tuple(sorted(compatible)),
            issues=tuple(issues),
            publishable=not any(item.code in blocking for item in issues),
        )

    def propose_schema_migration(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        target_schema_id: str,
        target_schema_revision: int,
        actor_id: str,
    ) -> tuple[MetadataMigrationPreview, tuple[MetadataValueRecord, ...]]:
        """把兼容旧值复制为目标 schema 的 import proposal。

        返回 proposal 后仍需显式审核发布；不兼容、改 scope 或新增 required 字段
        不会被猜测填充。
        """

        preview = self.preview_schema_migration(
            tenant_id=tenant_id,
            kb_id=kb_id,
            target_schema_id=target_schema_id,
            target_schema_revision=target_schema_revision,
        )
        current = self._current_release(tenant_id, kb_id)
        by_id = {item.record_id: item for item in self._release_values(current)}
        proposals = tuple(
            self.propose_value(
                schema_id=target_schema_id,
                schema_revision=target_schema_revision,
                target=by_id[record_id].target,
                field_key=by_id[record_id].field_key,
                value=by_id[record_id].value,
                source="import",
                proposed_by=actor_id,
            )
            for record_id in preview.compatible_record_ids
        )
        return preview, proposals

    def rollback_release(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        target_revision: int,
        actor_id: str,
    ) -> MetadataRelease:
        target = self.repository.get_release(tenant_id, kb_id, target_revision)
        if target is None:
            raise GovernanceError("RELEASE_NOT_FOUND", "rollback target release was not found")
        revision = self.repository.next_revision(tenant_id, kb_id)
        payload = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "revision": revision,
            "rollback_of": target.release_id,
        }
        release = MetadataRelease(
            release_id=stable_governance_id("mr", payload),
            tenant_id=tenant_id,
            kb_id=kb_id,
            revision=revision,
            metadata_schema_id=target.metadata_schema_id,
            metadata_schema_revision=target.metadata_schema_revision,
            taxonomy_id=target.taxonomy_id,
            taxonomy_revision=target.taxonomy_revision,
            value_record_ids=target.value_record_ids,
            tag_assignment_ids=target.tag_assignment_ids,
            source_release_id=target.release_id,
            action="rollback",
            published_by=actor_id,
            published_at=self._now(),
        )
        self.repository.save_release(release)
        self.repository.save_projection_state(
            ProjectionState(
                tenant_id=tenant_id,
                kb_id=kb_id,
                projection_name="retrieval_metadata",
                control_revision=revision,
                projected_revision=self._projection_revision(
                    tenant_id,
                    kb_id,
                    "retrieval_metadata",
                ),
                status="pending",
            )
        )
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="release_rolled_back",
            actor_id=actor_id,
            object_type="metadata_release",
            object_id=release.release_id,
            revision=revision,
            details={"target_revision": target_revision},
        )
        return release

    def _release_values(
        self,
        release: MetadataRelease | None,
    ) -> tuple[MetadataValueRecord, ...]:
        if release is None:
            return ()
        result: list[MetadataValueRecord] = []
        for record_id in release.value_record_ids:
            record = self.repository.get_value(record_id)
            if record is None or record.status != "published":
                raise GovernanceError(
                    "RELEASE_INTEGRITY_ERROR",
                    f"release references unavailable metadata value {record_id}",
                )
            result.append(record)
        return tuple(result)

    def _release_tags(
        self,
        release: MetadataRelease | None,
    ) -> tuple[TagAssignmentRecord, ...]:
        if release is None:
            return ()
        result: list[TagAssignmentRecord] = []
        for assignment_id in release.tag_assignment_ids:
            assignment = self.repository.get_tag_assignment(assignment_id)
            if assignment is None or assignment.status != "published":
                raise GovernanceError(
                    "RELEASE_INTEGRITY_ERROR",
                    f"release references unavailable tag assignment {assignment_id}",
                )
            result.append(assignment)
        return tuple(result)

    def _current_release(self, tenant_id: str, kb_id: str) -> MetadataRelease:
        release = self.repository.current_release(tenant_id, kb_id)
        if release is None:
            raise GovernanceError("NO_PUBLISHED_RELEASE", "KB has no metadata release")
        return release

    def _release_schema(self, release: MetadataRelease) -> MetadataSchemaVersion:
        schema = self.repository.get_schema(
            release.tenant_id,
            release.kb_id,
            release.metadata_schema_id,
            release.metadata_schema_revision,
        )
        if schema is None:
            raise GovernanceError(
                "RELEASE_INTEGRITY_ERROR",
                "release metadata schema is missing",
            )
        return schema

    def _release_taxonomy(self, release: MetadataRelease) -> TagTaxonomyVersion | None:
        if release.taxonomy_id is None or release.taxonomy_revision is None:
            return None
        taxonomy = self.repository.get_taxonomy(
            release.tenant_id,
            release.kb_id,
            release.taxonomy_id,
            release.taxonomy_revision,
        )
        if taxonomy is None:
            raise GovernanceError(
                "RELEASE_INTEGRITY_ERROR",
                "release tag taxonomy is missing",
            )
        return taxonomy

    def build_catalog(self, *, tenant_id: str, kb_id: str) -> MetadataCatalog:
        release = self._current_release(tenant_id, kb_id)
        schema = self._release_schema(release)
        values = self._release_values(release)
        by_field: dict[str, list[Any]] = defaultdict(list)
        for record in values:
            by_field[record.field_key].append(record.value)
        entries: list[MetadataCatalogEntry] = []
        for definition in schema.fields:
            raw_values = by_field.get(definition.key, [])
            canonical = {
                str(value): value
                for value in sorted(
                    raw_values,
                    key=lambda item: str(item),
                )
            }
            distinct = tuple(canonical.values())
            shown: tuple[Any, ...] = ()
            truncated = False
            minimum = None
            maximum = None
            if definition.catalog_mode == "distinct":
                limit = definition.max_distinct_values or 0
                shown = distinct[:limit]
                truncated = len(distinct) > limit
            elif definition.catalog_mode == "range" and distinct:
                minimum = min(distinct)
                maximum = max(distinct)
            entries.append(
                MetadataCatalogEntry(
                    field_key=definition.key,
                    value_type=definition.value_type,
                    scope=definition.scope,
                    catalog_mode=definition.catalog_mode,
                    distinct_values=shown,
                    minimum=minimum,
                    maximum=maximum,
                    distinct_count=len(distinct),
                    truncated=truncated,
                )
            )
        return MetadataCatalog(
            tenant_id=tenant_id,
            kb_id=kb_id,
            release_id=release.release_id,
            revision=release.revision,
            entries=tuple(entries),
        )

    @staticmethod
    def _normalize_filter_value(
        definition: MetadataFieldDefinition,
        predicate: FilterPredicate,
    ) -> Any:
        if predicate.operator == "exists":
            return predicate.value
        raw = predicate.value
        if isinstance(raw, tuple):
            return tuple(definition.normalize_value(item) for item in raw)
        return definition.normalize_value(raw)

    def map_filter(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        filter_node: FilterNode | None,
    ) -> MetadataFilterPlan:
        release = self._current_release(tenant_id, kb_id)
        schema = self._release_schema(release)
        by_key = {field.key: field for field in schema.fields}

        def visit(node: FilterNode) -> MappedMetadataFilter:
            if isinstance(node, FilterGroup):
                return MappedMetadataGroup(
                    operator=node.operator,
                    conditions=tuple(visit(item) for item in node.conditions),
                )
            assert isinstance(node, FilterPredicate)
            if node.field == "document_id":
                if node.operator not in {"eq", "ne", "in", "not_in", "exists"}:
                    raise GovernanceError(
                        "FILTER_OPERATOR_UNSUPPORTED",
                        "document_id only supports identity operators",
                    )
                raw = node.value
                values = raw if isinstance(raw, tuple) else (raw,)
                if node.operator != "exists" and any(
                    not isinstance(item, str) or not item.strip() for item in values
                ):
                    raise GovernanceError(
                        "INVALID_FILTER_VALUE",
                        "document_id filters require non-empty strings",
                    )
                return MappedMetadataPredicate(
                    field_key="document_id",
                    projection_path="document_id",
                    operator=node.operator,
                    value=raw,
                    value_type="identifier",
                )
            key = node.field.removeprefix("metadata.")
            definition = by_key.get(key)
            if definition is None:
                raise GovernanceError(
                    "FILTER_FIELD_UNKNOWN",
                    f"metadata field {key} is not in the published schema",
                )
            if not definition.filterable:
                raise GovernanceError(
                    "FILTER_FIELD_NOT_FILTERABLE",
                    f"metadata field {key} is not filterable",
                )
            if not definition.supports_operator(node.operator):
                raise GovernanceError(
                    "FILTER_OPERATOR_UNSUPPORTED",
                    f"{node.operator} is incompatible with {definition.value_type}",
                )
            try:
                value = self._normalize_filter_value(definition, node)
            except ValueError as exc:
                raise GovernanceError("INVALID_FILTER_VALUE", str(exc)) from exc
            return MappedMetadataPredicate(
                field_key=key,
                projection_path=f"metadata.{definition.scope}.{key}",
                operator=node.operator,
                value=value,
                value_type=definition.value_type,
            )

        mapped = None
        if filter_node is not None:
            validate_filter_ast(filter_node)
            mapped = visit(filter_node)
        return MetadataFilterPlan(
            tenant_id=tenant_id,
            kb_id=kb_id,
            release_id=release.release_id,
            control_revision=release.revision,
            mandatory_tenant_id=tenant_id,
            mandatory_kb_id=kb_id,
            filter=mapped,
        )

    def quality_report(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        expected_target_keys: Mapping[str, Iterable[str]],
    ) -> MetadataQualityReport:
        """从控制库事实统计覆盖率；``expected_target_keys`` 由文档控制面提供。"""

        release = self._current_release(tenant_id, kb_id)
        schema = self._release_schema(release)
        published = self._release_values(release)
        all_records = self.repository.list_values(tenant_id, kb_id)
        active_ids = set(release.value_record_ids)
        quality: list[MetadataFieldQuality] = []
        required_hits = 0
        required_total = 0
        for definition in schema.fields:
            expected = set(expected_target_keys.get(definition.scope, ()))
            active = [
                item
                for item in published
                if item.field_key == definition.key and item.target.target_key in expected
            ]
            published_targets = {item.target.target_key for item in active}
            automatic = [item for item in active if item.source == "auto"]
            confidences = [
                float(item.confidence) for item in automatic if item.confidence is not None
            ]
            expected_count = len(expected)
            coverage = min(1.0, len(published_targets) / expected_count) if expected_count else 1.0
            if definition.required:
                required_hits += len(published_targets)
                required_total += expected_count
            quality.append(
                MetadataFieldQuality(
                    field_key=definition.key,
                    scope=definition.scope,
                    expected_target_count=expected_count,
                    published_target_count=len(published_targets),
                    proposed_count=sum(
                        1
                        for item in all_records
                        if item.field_key == definition.key and item.status == "proposed"
                    ),
                    rejected_count=sum(
                        1
                        for item in all_records
                        if item.field_key == definition.key and item.status == "rejected"
                    ),
                    manual_published_count=sum(
                        1
                        for item in active
                        if item.source == "manual" and item.record_id in active_ids
                    ),
                    automatic_published_count=len(automatic),
                    average_automatic_confidence=(
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                    coverage=coverage,
                )
            )
        state = self.repository.get_projection_state(
            tenant_id,
            kb_id,
            "retrieval_metadata",
        )
        projected = state.projected_revision if state else 0
        return MetadataQualityReport(
            tenant_id=tenant_id,
            kb_id=kb_id,
            release_id=release.release_id,
            revision=release.revision,
            fields=tuple(quality),
            required_field_coverage=(
                min(1.0, required_hits / required_total) if required_total else 1.0
            ),
            pending_review_count=sum(1 for item in all_records if item.status == "proposed"),
            rejected_count=sum(1 for item in all_records if item.status == "rejected"),
            projection_lag=max(0, release.revision - projected),
        )

    def build_projection_batch(
        self,
        *,
        tenant_id: str,
        kb_id: str,
    ) -> MetadataProjectionBatch:
        release = self._current_release(tenant_id, kb_id)
        schema = self._release_schema(release)
        field_by_key = {item.key: item for item in schema.fields}
        values = tuple(
            ProjectionValue(
                record_id=record.record_id,
                target=record.target,
                field_key=record.field_key,
                value=record.value,
                value_type=field_by_key[record.field_key].value_type,
            )
            for record in sorted(
                self._release_values(release),
                key=lambda item: (item.target.target_key, item.field_key),
            )
        )
        taxonomy = self._release_taxonomy(release)
        tag_by_id = {item.tag_id: item for item in taxonomy.tags} if taxonomy else {}

        def tag_path(tag: TagDefinition) -> tuple[str, ...]:
            result = [tag.name]
            cursor = tag
            while cursor.parent_tag_id is not None:
                cursor = tag_by_id[cursor.parent_tag_id]
                result.append(cursor.name)
            return tuple(reversed(result))

        tags = tuple(
            ProjectionTag(
                assignment_id=assignment.assignment_id,
                target=assignment.target,
                tag_id=assignment.tag_id,
                tag_name=tag_by_id[assignment.tag_id].name,
                tag_path=tag_path(tag_by_id[assignment.tag_id]),
            )
            for assignment in sorted(
                self._release_tags(release),
                key=lambda item: (item.target.target_key, item.tag_id),
            )
        )
        payload = {
            "release_id": release.release_id,
            "revision": release.revision,
            "value_ids": [item.record_id for item in values],
            "tag_ids": [item.assignment_id for item in tags],
        }
        return MetadataProjectionBatch(
            batch_id=stable_governance_id("mp", payload),
            tenant_id=tenant_id,
            kb_id=kb_id,
            release_id=release.release_id,
            control_revision=release.revision,
            values=values,
            tags=tags,
        )

    def projection_started(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        projection_name: str,
        actor_id: str,
    ) -> ProjectionState:
        release = self._current_release(tenant_id, kb_id)
        current = self.repository.get_projection_state(
            tenant_id,
            kb_id,
            projection_name,
        )
        state = ProjectionState(
            tenant_id=tenant_id,
            kb_id=kb_id,
            projection_name=projection_name,
            control_revision=release.revision,
            projected_revision=current.projected_revision if current else 0,
            status="syncing",
            attempt=(current.attempt if current else 0) + 1,
            last_started_at=self._now(),
            last_succeeded_at=current.last_succeeded_at if current else None,
        )
        self.repository.save_projection_state(state)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="projection_started",
            actor_id=actor_id,
            object_type="projection",
            object_id=projection_name,
            revision=release.revision,
        )
        return state

    def projection_finished(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        projection_name: str,
        actor_id: str,
        projected_revision: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ProjectionState:
        release = self._current_release(tenant_id, kb_id)
        current = self.repository.get_projection_state(
            tenant_id,
            kb_id,
            projection_name,
        )
        if current is None or current.status != "syncing":
            raise GovernanceError(
                "PROJECTION_NOT_RUNNING",
                "projection must be started before it can finish",
            )
        if error_code is not None:
            state = ProjectionState(
                tenant_id=tenant_id,
                kb_id=kb_id,
                projection_name=projection_name,
                control_revision=release.revision,
                projected_revision=current.projected_revision,
                status="failed",
                attempt=current.attempt,
                last_started_at=current.last_started_at,
                last_succeeded_at=current.last_succeeded_at,
                error_code=error_code,
                error_message=error_message or "projection failed",
            )
            action = "projection_failed"
        else:
            resolved = release.revision if projected_revision is None else projected_revision
            if resolved != release.revision:
                raise GovernanceError(
                    "PROJECTION_REVISION_MISMATCH",
                    "successful projection must acknowledge the current release revision",
                )
            state = ProjectionState(
                tenant_id=tenant_id,
                kb_id=kb_id,
                projection_name=projection_name,
                control_revision=release.revision,
                projected_revision=resolved,
                status="in_sync",
                attempt=current.attempt,
                last_started_at=current.last_started_at,
                last_succeeded_at=self._now(),
            )
            action = "projection_succeeded"
        self.repository.save_projection_state(state)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action=action,
            actor_id=actor_id,
            object_type="projection",
            object_id=projection_name,
            revision=release.revision,
            details={"error_code": error_code} if error_code else {},
        )
        return state

    def _projection_revision(
        self,
        tenant_id: str,
        kb_id: str,
        projection_name: str,
    ) -> int:
        state = self.repository.get_projection_state(
            tenant_id,
            kb_id,
            projection_name,
        )
        return state.projected_revision if state else 0

    def start_enhancement(
        self,
        *,
        run_id: str,
        tenant_id: str,
        kb_id: str,
        job_id: str,
        prompt_id: str,
        prompt_version: str,
        model_id: str,
        target_scope: Literal["document", "document_version", "chunk"],
        target_count: int,
        actor_id: str,
    ) -> EnhancementRun:
        existing = self.repository.get_enhancement_run(run_id)
        if existing is not None:
            return existing
        run = EnhancementRun(
            run_id=run_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            job_id=job_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            model_id=model_id,
            target_scope=target_scope,
            target_count=target_count,
            status="running",
            started_at=self._now(),
        )
        self.repository.save_enhancement_run(run)
        self._audit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            action="enhancement_started",
            actor_id=actor_id,
            object_type="enhancement_run",
            object_id=run_id,
        )
        return run

    def finish_enhancement(
        self,
        run_id: str,
        *,
        actor_id: str,
        status: EnhancementRunStatus,
        succeeded_count: int,
        failed_count: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        error_summary: str | None = None,
    ) -> EnhancementRun:
        run = self.repository.get_enhancement_run(run_id)
        if run is None:
            raise GovernanceError("ENHANCEMENT_NOT_FOUND", "enhancement run was not found")
        if run.status != "running":
            raise GovernanceError(
                "ENHANCEMENT_NOT_RUNNING",
                "only a running enhancement can finish",
            )
        if status not in {"succeeded", "partially_succeeded", "failed", "cancelled"}:
            raise GovernanceError(
                "INVALID_ENHANCEMENT_STATUS",
                "finish requires a terminal enhancement status",
            )
        finished = _validated_copy(
            run,
            status=status,
            succeeded_count=succeeded_count,
            failed_count=failed_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            finished_at=self._now(),
            error_summary=error_summary,
        )
        self.repository.save_enhancement_run(finished)
        self._audit(
            tenant_id=run.tenant_id,
            kb_id=run.kb_id,
            action="enhancement_finished",
            actor_id=actor_id,
            object_type="enhancement_run",
            object_id=run_id,
            details={"status": status},
        )
        return finished


__all__ = [
    "GovernanceError",
    "MetadataGovernanceRepository",
    "MetadataGovernanceService",
]
