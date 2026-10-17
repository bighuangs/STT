from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.schemas.p2 import P2CapabilityResourceCreate
from app.application.p2_control_service import (
    P2CapabilityValidationError,
    expected_resource_scope,
    expected_schema_version,
    validate_capability_payload,
)
from app.domain.advanced_retrieval import (
    AdvancedIndexSnapshot,
    AdvancedIndexVersion,
    StructureNode,
    StructureSnapshot,
)
from app.domain.connectors import ConnectorScope, DataSource
from app.domain.knowledge_governance import (
    MetadataFieldDefinition,
    MetadataSchemaVersion,
    TagDefinition,
    TagTaxonomyVersion,
)


TENANT = "tenant_scope"
PAYLOAD_KB = "kb_payload"
RESOURCE_KB = "kb_resource"
NOW = datetime(2026, 7, 24, tzinfo=UTC)

RESOURCE_SCOPES = (
    ("model_route", "tenant"),
    ("metadata_schema", "kb"),
    ("tag_taxonomy", "kb"),
    ("advanced_index", "kb"),
    ("connector_source", "kb"),
    ("agent_definition", "tenant"),
    ("memory_policy", "kb"),
    ("multimodal_profile", "kb"),
)


def _resource_request(resource_type: str, *, kb_id: str | None) -> dict:
    return {
        "resource_type": resource_type,
        "logical_id": f"{resource_type}_scope_test",
        "version": "1.0.0",
        "schema_version": expected_schema_version(resource_type),
        "kb_id": kb_id,
        "payload": {},
    }


@pytest.mark.parametrize(("resource_type", "scope"), RESOURCE_SCOPES)
def test_all_capability_resource_types_have_an_explicit_container_scope(
    resource_type: str,
    scope: str,
) -> None:
    assert expected_resource_scope(resource_type) == scope
    kb_id = None if scope == "tenant" else RESOURCE_KB
    parsed = P2CapabilityResourceCreate.model_validate(
        _resource_request(resource_type, kb_id=kb_id)
    )
    assert parsed.kb_id == kb_id


@pytest.mark.parametrize(("resource_type", "scope"), RESOURCE_SCOPES)
def test_resource_schema_rejects_the_wrong_outer_scope_for_all_types(
    resource_type: str,
    scope: str,
) -> None:
    wrong_kb_id = RESOURCE_KB if scope == "tenant" else None
    expected_message = (
        "tenant-scoped and does not accept resource kb_id"
        if scope == "tenant"
        else "KB-scoped and requires resource kb_id"
    )
    with pytest.raises(ValidationError, match=expected_message):
        P2CapabilityResourceCreate.model_validate(
            _resource_request(resource_type, kb_id=wrong_kb_id)
        )


@pytest.mark.parametrize(("resource_type", "scope"), RESOURCE_SCOPES)
def test_control_service_rechecks_outer_scope_when_api_schema_is_bypassed(
    resource_type: str,
    scope: str,
) -> None:
    wrong_kb_id = RESOURCE_KB if scope == "tenant" else None
    expected_message = (
        "tenant-scoped and does not accept resource kb_id"
        if scope == "tenant"
        else "KB-scoped and requires resource kb_id"
    )
    with pytest.raises(P2CapabilityValidationError, match=expected_message):
        validate_capability_payload(
            resource_type=resource_type,
            schema_version=expected_schema_version(resource_type),
            payload={},
            tenant_id=TENANT,
            kb_id=wrong_kb_id,
            actor_id="user_scope",
        )


def _metadata_schema_payload() -> dict:
    return MetadataSchemaVersion(
        schema_id="schema_scope",
        tenant_id=TENANT,
        kb_id=PAYLOAD_KB,
        version=1,
        revision=1,
        fields=(
            MetadataFieldDefinition(
                key="product_type",
                value_type="string",
                description="产品类型",
                scope="document",
            ),
        ),
        created_by="user_scope",
        created_at=NOW,
    ).model_dump(mode="json")


def _tag_taxonomy_payload() -> dict:
    return TagTaxonomyVersion(
        taxonomy_id="taxonomy_scope",
        tenant_id=TENANT,
        kb_id=PAYLOAD_KB,
        version=1,
        revision=1,
        tags=(TagDefinition(tag_id="safety", name="安全"),),
        created_by="user_scope",
        created_at=NOW,
    ).model_dump(mode="json")


def _advanced_index_payload() -> dict:
    evidence_id = "ev_" + "a" * 40
    advanced_index_id = "advanced_scope"
    source_index_version_id = "index_source"
    structure = StructureSnapshot(
        advanced_index_id=advanced_index_id,
        source_index_version_id=source_index_version_id,
        nodes=(
            StructureNode(
                node_id="toc_root",
                node_type="toc",
                tenant_id=TENANT,
                kb_id=PAYLOAD_KB,
                document_id="doc_scope",
                document_version_id="document_version_scope",
                source_index_version_id=source_index_version_id,
                title="安全",
                toc_path=("安全",),
                order=0,
                source_evidence_ids=(evidence_id,),
            ),
        ),
    )
    return AdvancedIndexSnapshot(
        index=AdvancedIndexVersion(
            advanced_index_id=advanced_index_id,
            index_type="toc",
            tenant_id=TENANT,
            kb_id=PAYLOAD_KB,
            scope="document",
            document_id="doc_scope",
            version=1,
            source_index_version_id=source_index_version_id,
            source_document_version_ids=("document_version_scope",),
            created_by="user_scope",
            created_at=NOW,
        ),
        evidence_ids=(evidence_id,),
        structure=structure,
    ).model_dump(mode="json")


def _connector_source_payload() -> dict:
    return DataSource(
        data_source_id="source_scope",
        scope=ConnectorScope(tenant_id=TENANT, kb_id=PAYLOAD_KB),
        name="scope test",
        connector_key="deterministic",
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("resource_type", "payload_factory"),
    (
        ("metadata_schema", _metadata_schema_payload),
        ("tag_taxonomy", _tag_taxonomy_payload),
        ("advanced_index", _advanced_index_payload),
        ("connector_source", _connector_source_payload),
    ),
)
def test_kb_scoped_payload_identity_must_equal_outer_resource_kb(
    resource_type: str,
    payload_factory,
) -> None:
    with pytest.raises(
        P2CapabilityValidationError,
        match="payload KB scope does not match resource KB",
    ):
        validate_capability_payload(
            resource_type=resource_type,
            schema_version=expected_schema_version(resource_type),
            payload=payload_factory(),
            tenant_id=TENANT,
            kb_id=RESOURCE_KB,
            actor_id="user_scope",
        )
