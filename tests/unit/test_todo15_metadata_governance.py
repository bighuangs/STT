from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.application.metadata_governance_service import (
    GovernanceError,
    MetadataGovernanceService,
)
from app.domain.knowledge_governance import (
    EnhancementReference,
    MetadataFieldDefinition,
    MetadataSchemaVersion,
    MetadataTarget,
    TagDefinition,
    TagTaxonomyVersion,
    UnknownTagCandidate,
    normalize_metadata_value,
)
from app.domain.retrieval import FilterGroup, FilterPredicate
from app.infra.governance import InMemoryMetadataGovernanceRepository


NOW = datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc)
TENANT = "tenant_default"
KB = "kb_manual"


def _service() -> tuple[MetadataGovernanceService, InMemoryMetadataGovernanceRepository]:
    repository = InMemoryMetadataGovernanceRepository()
    return MetadataGovernanceService(repository, clock=lambda: NOW), repository


def _fields(*, compliance: bool = False) -> tuple[MetadataFieldDefinition, ...]:
    return (
        MetadataFieldDefinition(
            key="product_type",
            value_type="string",
            description="产品类型",
            scope="document",
            required=True,
            filterable=True,
            enum_values=("manual", "datasheet"),
            catalog_mode="distinct",
            max_distinct_values=10,
        ),
        MetadataFieldDefinition(
            key="risk_level",
            value_type="integer",
            description="风险等级",
            scope="chunk",
            filterable=True,
            catalog_mode="range",
            compliance_controlled=compliance,
        ),
        MetadataFieldDefinition(
            key="notes",
            value_type="string_list",
            description="备注",
            scope="document_version",
        ),
    )


def _published_schema(
    service: MetadataGovernanceService,
    *,
    fields: tuple[MetadataFieldDefinition, ...] | None = None,
    schema_id: str = "schema_manual",
    version: int = 1,
):
    draft = service.create_schema_draft(
        schema_id=schema_id,
        tenant_id=TENANT,
        kb_id=KB,
        version=version,
        fields=fields or _fields(),
        actor_id="admin",
    )
    return service.publish_schema(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=draft.schema_id,
        revision=draft.revision,
        actor_id="admin",
    )


def _published_taxonomy(service: MetadataGovernanceService):
    draft = service.create_taxonomy_draft(
        taxonomy_id="tags_manual",
        tenant_id=TENANT,
        kb_id=KB,
        version=1,
        tags=(
            TagDefinition(tag_id="safety", name="安全", aliases=("safety",)),
            TagDefinition(
                tag_id="electrical",
                name="电气安全",
                parent_tag_id="safety",
                color="#12abEF",
            ),
        ),
        actor_id="admin",
    )
    return service.publish_taxonomy(
        tenant_id=TENANT,
        kb_id=KB,
        taxonomy_id=draft.taxonomy_id,
        revision=draft.revision,
        actor_id="admin",
    )


def _document_target(document: str = "doc_hak180") -> MetadataTarget:
    return MetadataTarget(
        tenant_id=TENANT,
        kb_id=KB,
        scope="document",
        document_id=document,
    )


def _chunk_target(chunk: str = "chunk_1") -> MetadataTarget:
    return MetadataTarget(
        tenant_id=TENANT,
        kb_id=KB,
        scope="chunk",
        document_id="doc_hak180",
        document_version_id="dv_hak180_v1",
        chunk_id=chunk,
    )


def test_metadata_types_are_strict_and_normalized() -> None:
    assert normalize_metadata_value("integer", 1) == 1
    assert normalize_metadata_value("number", 1) == 1.0
    assert normalize_metadata_value("date", "2026-07-24") == "2026-07-24"
    assert (
        normalize_metadata_value("datetime", "2026-07-24T16:00:00+08:00") == "2026-07-24T08:00:00Z"
    )
    assert normalize_metadata_value("string_list", ["A", "B"]) == ("A", "B")
    with pytest.raises(ValueError, match="integer"):
        normalize_metadata_value("integer", "1")
    with pytest.raises(ValueError, match="boolean"):
        normalize_metadata_value("boolean", 1)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_metadata_value("string_list", ["A", "A"])
    with pytest.raises(ValueError, match="finite"):
        normalize_metadata_value("number", float("nan"))


def test_schema_rejects_unsafe_or_high_cardinality_filter_fields() -> None:
    with pytest.raises(ValidationError, match="unsafe"):
        MetadataFieldDefinition(
            key="_tenant_id",
            value_type="string",
            description="unsafe",
            scope="document",
        )
    with pytest.raises(ValidationError, match="cannot exceed 500"):
        MetadataFieldDefinition(
            key="serial",
            value_type="string",
            description="高基数字段",
            scope="document",
            filterable=True,
            catalog_mode="distinct",
            max_distinct_values=1000,
        )
    with pytest.raises(ValidationError, match="scalar"):
        MetadataFieldDefinition(
            key="labels",
            value_type="string_list",
            description="labels",
            scope="chunk",
            filterable=True,
        )


def test_schema_and_target_scope_invariants() -> None:
    with pytest.raises(ValidationError, match="unique"):
        MetadataSchemaVersion(
            schema_id="schema",
            tenant_id=TENANT,
            kb_id=KB,
            version=1,
            revision=1,
            fields=(_fields()[0], _fields()[0]),
            created_by="admin",
            created_at=NOW,
        )
    with pytest.raises(ValidationError, match="document metadata"):
        MetadataTarget(
            tenant_id=TENANT,
            kb_id=KB,
            scope="document",
            document_id="doc",
            document_version_id="dv",
        )
    with pytest.raises(ValidationError, match="requires document_version_id"):
        MetadataTarget(
            tenant_id=TENANT,
            kb_id=KB,
            scope="chunk",
            document_id="doc",
            chunk_id="chunk",
        )


def test_taxonomy_rejects_dangling_cycles_and_alias_collisions() -> None:
    with pytest.raises(ValidationError, match="unknown parent"):
        TagTaxonomyVersion(
            taxonomy_id="tags",
            tenant_id=TENANT,
            kb_id=KB,
            version=1,
            revision=1,
            tags=(TagDefinition(tag_id="child", name="子", parent_tag_id="missing"),),
            created_by="admin",
            created_at=NOW,
        )
    with pytest.raises(ValidationError, match="cycle"):
        TagTaxonomyVersion(
            taxonomy_id="tags",
            tenant_id=TENANT,
            kb_id=KB,
            version=1,
            revision=1,
            tags=(
                TagDefinition(tag_id="a", name="A", parent_tag_id="b"),
                TagDefinition(tag_id="b", name="B", parent_tag_id="a"),
            ),
            created_by="admin",
            created_at=NOW,
        )
    with pytest.raises(ValidationError, match="collision"):
        TagTaxonomyVersion(
            taxonomy_id="tags",
            tenant_id=TENANT,
            kb_id=KB,
            version=1,
            revision=1,
            tags=(
                TagDefinition(tag_id="a", name="A", aliases=("same",)),
                TagDefinition(tag_id="b", name="B", aliases=("SAME",)),
            ),
            created_by="admin",
            created_at=NOW,
        )


def test_proposal_is_typed_audited_and_idempotent() -> None:
    service, repository = _service()
    schema = _published_schema(service)
    target = _document_target()
    one = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=target,
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    two = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=target,
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    assert one.record_id == two.record_id
    assert len(repository.list_values(TENANT, KB)) == 1
    assert {event.action for event in repository.list_audit(TENANT, KB)} >= {
        "schema_created",
        "schema_published",
        "value_proposed",
    }
    with pytest.raises(GovernanceError) as error:
        service.propose_value(
            schema_id=schema.schema_id,
            schema_revision=schema.revision,
            target=target,
            field_key="product_type",
            value="unknown",
            source="manual",
            proposed_by="editor",
        )
    assert error.value.code == "INVALID_METADATA_VALUE"
    with pytest.raises(GovernanceError) as error:
        service.propose_value(
            schema_id=schema.schema_id,
            schema_revision=schema.revision,
            target=_chunk_target(),
            field_key="product_type",
            value="manual",
            source="manual",
            proposed_by="editor",
        )
    assert error.value.code == "FIELD_SCOPE_MISMATCH"


def test_automatic_proposals_require_traceable_model_run() -> None:
    service, _ = _service()
    schema = _published_schema(service)
    with pytest.raises(ValidationError, match="confidence"):
        service.propose_value(
            schema_id=schema.schema_id,
            schema_revision=schema.revision,
            target=_document_target(),
            field_key="product_type",
            value="manual",
            source="auto",
            proposed_by="worker",
        )
    reference = EnhancementReference(
        run_id="run_1",
        prompt_id="metadata",
        prompt_version="v2",
        model_id="qwen",
    )
    record = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="manual",
        source="auto",
        proposed_by="worker",
        confidence=0.91,
        enhancement=reference,
    )
    assert record.enhancement == reference


def test_unknown_generated_tag_enters_candidate_area_not_projection() -> None:
    service, repository = _service()
    _published_schema(service)
    taxonomy = _published_taxonomy(service)
    result = service.propose_tag(
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.revision,
        target=_document_target(),
        label="模型随便生成的新标签",
        source="auto",
        proposed_by="worker",
        confidence=0.7,
    )
    assert isinstance(result, UnknownTagCandidate)
    assert repository.list_unknown_tags(TENANT, KB) == (result,)
    assert repository.list_tag_assignments(TENANT, KB) == ()
    with pytest.raises(GovernanceError) as error:
        service.propose_tag(
            taxonomy_id=taxonomy.taxonomy_id,
            taxonomy_revision=taxonomy.revision,
            target=_document_target(),
            label="人工任意标签",
            source="manual",
            proposed_by="editor",
        )
    assert error.value.code == "UNKNOWN_MANUAL_TAG"


def test_batch_publish_protects_manual_value_from_automatic_regeneration() -> None:
    service, repository = _service()
    schema = _published_schema(service)
    manual = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    first = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(manual.record_id,),
    )
    automatic = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="datasheet",
        source="auto",
        proposed_by="worker",
        confidence=0.99,
        enhancement=EnhancementReference(
            run_id="run_2",
            prompt_id="metadata",
            prompt_version="v3",
            model_id="qwen",
        ),
    )
    second = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(automatic.record_id,),
    )
    assert second.revision > first.revision
    assert second.value_record_ids == (manual.record_id,)
    protected = repository.get_value(automatic.record_id)
    assert protected is not None
    assert protected.status == "rejected"
    assert protected.review_reason == "MANUAL_VALUE_PROTECTED"


def test_compliance_field_needs_explicit_privileged_review() -> None:
    service, _ = _service()
    schema = _published_schema(service, fields=_fields(compliance=True))
    proposal = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_chunk_target(),
        field_key="risk_level",
        value=3,
        source="manual",
        proposed_by="editor",
    )
    with pytest.raises(GovernanceError) as error:
        service.publish_proposals(
            tenant_id=TENANT,
            kb_id=KB,
            schema_id=schema.schema_id,
            schema_revision=schema.revision,
            reviewer_id="reviewer",
            value_record_ids=(proposal.record_id,),
        )
    assert error.value.code == "COMPLIANCE_PERMISSION_REQUIRED"
    release = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="compliance_admin",
        value_record_ids=(proposal.record_id,),
        allow_compliance_controlled=True,
    )
    assert release.value_record_ids == (proposal.record_id,)


def test_tag_alias_resolves_and_projection_has_hierarchical_path() -> None:
    service, _ = _service()
    schema = _published_schema(service)
    taxonomy = _published_taxonomy(service)
    assignment = service.propose_tag(
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.revision,
        target=_document_target(),
        label="电气安全",
        source="manual",
        proposed_by="editor",
    )
    assert not isinstance(assignment, UnknownTagCandidate)
    service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.revision,
        reviewer_id="reviewer",
        tag_assignment_ids=(assignment.assignment_id,),
    )
    batch = service.build_projection_batch(tenant_id=TENANT, kb_id=KB)
    assert batch.tags[0].tag_path == ("安全", "电气安全")
    assert batch.tags[0].target == _document_target()


def test_release_rollback_creates_new_revision_and_projection_lag() -> None:
    service, repository = _service()
    schema = _published_schema(service)
    first_value = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    first = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(first_value.record_id,),
    )
    second_value = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="datasheet",
        source="manual",
        proposed_by="editor",
    )
    second = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(second_value.record_id,),
    )
    rollback = service.rollback_release(
        tenant_id=TENANT,
        kb_id=KB,
        target_revision=first.revision,
        actor_id="admin",
    )
    assert rollback.action == "rollback"
    assert rollback.revision > second.revision
    assert rollback.value_record_ids == first.value_record_ids
    assert repository.current_release(TENANT, KB) == rollback
    assert (
        service.build_projection_batch(
            tenant_id=TENANT,
            kb_id=KB,
        ).control_revision
        == rollback.revision
    )


def test_filter_mapping_uses_published_schema_and_server_scope() -> None:
    service, _ = _service()
    schema = _published_schema(service)
    proposal = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_chunk_target(),
        field_key="risk_level",
        value=3,
        source="manual",
        proposed_by="editor",
    )
    service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(proposal.record_id,),
    )
    plan = service.map_filter(
        tenant_id=TENANT,
        kb_id=KB,
        filter_node=FilterGroup(
            operator="and",
            conditions=(
                FilterPredicate(
                    field="metadata.risk_level",
                    operator="gte",
                    value=2,
                ),
                FilterPredicate(
                    field="document_id",
                    operator="eq",
                    value="doc_hak180",
                ),
            ),
        ),
    )
    assert plan.mandatory_tenant_id == TENANT
    assert plan.mandatory_kb_id == KB
    assert plan.filter is not None
    dumped = plan.model_dump(mode="json")
    assert dumped["filter"]["conditions"][0]["projection_path"] == ("metadata.chunk.risk_level")
    with pytest.raises(GovernanceError) as error:
        service.map_filter(
            tenant_id=TENANT,
            kb_id=KB,
            filter_node=FilterPredicate(
                field="metadata.notes",
                operator="eq",
                value="x",
            ),
        )
    assert error.value.code == "FILTER_FIELD_NOT_FILTERABLE"
    with pytest.raises(GovernanceError) as error:
        service.map_filter(
            tenant_id=TENANT,
            kb_id=KB,
            filter_node=FilterPredicate(
                field="metadata.risk_level",
                operator="contains",
                value="3",
            ),
        )
    assert error.value.code == "FILTER_OPERATOR_UNSUPPORTED"


def test_catalog_quality_and_projection_are_built_only_from_release_facts() -> None:
    service, _ = _service()
    schema = _published_schema(service)
    published = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    pending = service.propose_value(
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        target=_chunk_target(),
        field_key="risk_level",
        value=5,
        source="auto",
        proposed_by="worker",
        confidence=0.4,
        enhancement=EnhancementReference(
            run_id="run_3",
            prompt_id="metadata",
            prompt_version="v1",
            model_id="qwen",
        ),
    )
    release = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(published.record_id,),
    )
    catalog = service.build_catalog(tenant_id=TENANT, kb_id=KB)
    product = next(item for item in catalog.entries if item.field_key == "product_type")
    risk = next(item for item in catalog.entries if item.field_key == "risk_level")
    assert product.distinct_values == ("manual",)
    assert risk.distinct_count == 0
    assert pending.record_id not in {
        value.record_id
        for value in service.build_projection_batch(tenant_id=TENANT, kb_id=KB).values
    }
    report = service.quality_report(
        tenant_id=TENANT,
        kb_id=KB,
        expected_target_keys={
            "document": {_document_target().target_key},
            "chunk": {_chunk_target().target_key},
            "document_version": set(),
        },
    )
    assert report.release_id == release.release_id
    assert report.required_field_coverage == 1.0
    assert report.pending_review_count == 1
    assert report.projection_lag == release.revision


def test_projection_failure_is_visible_and_retry_can_reconcile_revision() -> None:
    service, _ = _service()
    schema = _published_schema(service)
    release = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=schema.schema_id,
        schema_revision=schema.revision,
        reviewer_id="reviewer",
    )
    first = service.projection_started(
        tenant_id=TENANT,
        kb_id=KB,
        projection_name="retrieval_metadata",
        actor_id="worker",
    )
    failed = service.projection_finished(
        tenant_id=TENANT,
        kb_id=KB,
        projection_name="retrieval_metadata",
        actor_id="worker",
        error_code="MILVUS_TIMEOUT",
        error_message="timeout",
    )
    assert failed.status == "failed"
    assert failed.projected_revision == 0
    retry = service.projection_started(
        tenant_id=TENANT,
        kb_id=KB,
        projection_name="retrieval_metadata",
        actor_id="worker",
    )
    assert retry.attempt == first.attempt + 1
    done = service.projection_finished(
        tenant_id=TENANT,
        kb_id=KB,
        projection_name="retrieval_metadata",
        actor_id="worker",
        projected_revision=release.revision,
    )
    assert done.status == "in_sync"
    assert done.control_revision == done.projected_revision


def test_schema_migration_requires_preview_and_explicit_release_token() -> None:
    service, _ = _service()
    first_schema = _published_schema(service)
    current = service.propose_value(
        schema_id=first_schema.schema_id,
        schema_revision=first_schema.revision,
        target=_document_target(),
        field_key="product_type",
        value="manual",
        source="manual",
        proposed_by="editor",
    )
    first_release = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=first_schema.schema_id,
        schema_revision=first_schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(current.record_id,),
    )
    second_schema = _published_schema(
        service,
        schema_id="schema_manual_v2",
        version=2,
        fields=(
            MetadataFieldDefinition(
                key="product_type",
                value_type="string",
                description="产品类型",
                scope="document",
                required=True,
                filterable=True,
                enum_values=("manual", "datasheet"),
            ),
        ),
    )
    preview, proposals = service.propose_schema_migration(
        tenant_id=TENANT,
        kb_id=KB,
        target_schema_id=second_schema.schema_id,
        target_schema_revision=second_schema.revision,
        actor_id="migration_worker",
    )
    assert preview.publishable
    assert len(proposals) == 1
    with pytest.raises(GovernanceError) as error:
        service.publish_proposals(
            tenant_id=TENANT,
            kb_id=KB,
            schema_id=second_schema.schema_id,
            schema_revision=second_schema.revision,
            reviewer_id="reviewer",
            value_record_ids=(proposals[0].record_id,),
        )
    assert error.value.code == "SCHEMA_MIGRATION_REQUIRED"
    migrated = service.publish_proposals(
        tenant_id=TENANT,
        kb_id=KB,
        schema_id=second_schema.schema_id,
        schema_revision=second_schema.revision,
        reviewer_id="reviewer",
        value_record_ids=(proposals[0].record_id,),
        migration_from_release_id=first_release.release_id,
    )
    assert migrated.metadata_schema_revision == second_schema.revision
    assert migrated.value_record_ids == (proposals[0].record_id,)


def test_enhancement_run_records_model_prompt_cost_and_terminal_state() -> None:
    service, repository = _service()
    run = service.start_enhancement(
        run_id="run_metadata_1",
        tenant_id=TENANT,
        kb_id=KB,
        job_id="job_1",
        prompt_id="metadata",
        prompt_version="v3",
        model_id="qwen-plus",
        target_scope="chunk",
        target_count=10,
        actor_id="worker",
    )
    assert run.status == "running"
    finished = service.finish_enhancement(
        run.run_id,
        actor_id="worker",
        status="partially_succeeded",
        succeeded_count=8,
        failed_count=2,
        input_tokens=1000,
        output_tokens=200,
        cost=0.12,
        error_summary="2 chunks failed validation",
    )
    assert finished.finished_at == NOW
    assert finished.cost == 0.12
    assert repository.get_enhancement_run(run.run_id) == finished
