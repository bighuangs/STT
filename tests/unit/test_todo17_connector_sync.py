"""TODO-17 connector contracts, planning, idempotency and reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib

import pytest
from pydantic import ValidationError

from app.application.connector_sync_service import (
    ConnectorConfigurationError,
    ConnectorRegistry,
    ConnectorRateLimitedError,
    ConnectorScopeError,
    ConnectorSyncService,
    ContentIntegrityError,
    ExternalConnectorDisabledError,
    IncrementalSyncPlanner,
    ReconciliationSafetyError,
)
from app.domain.connectors import (
    ConnectorCapabilities,
    ConnectorScope,
    DataSource,
    DeletePropagationPolicy,
    SourceBindingStatus,
    SourceChange,
    SourceChangeKind,
    SourceItem,
    SourceItemBinding,
    SourceSnapshot,
    SourceTombstone,
    SyncAction,
    SyncCursor,
    SyncPolicy,
)
from app.infra.connectors import (
    DeterministicFakeConnector,
    InMemoryConnectorSyncRepository,
    InMemoryDocumentVersionBoundary,
)


NOW = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)


def _scope(tenant: str = "tenant-a", kb: str = "kb-a") -> ConnectorScope:
    return ConnectorScope(tenant_id=tenant, kb_id=kb)


def _source(
    *,
    scope: ConnectorScope | None = None,
    page_size: int = 500,
    delete_policy: DeletePropagationPolicy = DeletePropagationPolicy.DISABLE,
) -> DataSource:
    return DataSource(
        data_source_id="source-a",
        scope=scope or _scope(),
        name="offline fixture",
        connector_key="fake",
        config={"root": "fixture-a"},
        sync_policy=SyncPolicy(
            max_changes_per_run=page_size,
            delete_policy=delete_policy,
            deletion_grace_seconds=3600,
            allowed_mime_types=("text/plain",),
        ),
    )


def _item(
    item_id: str,
    version: str,
    content: bytes,
    *,
    logical_name: str | None = None,
    locator: str | None = None,
    declared_hash: str | None = None,
) -> SourceItem:
    return SourceItem(
        source_item_id=item_id,
        remote_version=version,
        content_sha256=declared_hash or hashlib.sha256(content).hexdigest(),
        logical_name=logical_name or f"{item_id}.txt",
        mime_type="text/plain",
        size_bytes=len(content),
        locator=locator or f"folder/{item_id}.txt",
    )


def _change(
    event_id: str,
    sequence: int,
    item: SourceItem,
    *,
    kind: SourceChangeKind = SourceChangeKind.CREATE,
) -> SourceChange:
    return SourceChange(
        event_id=event_id,
        sequence=sequence,
        kind=kind,
        item=item,
        occurred_at=NOW,
    )


def _delete(
    event_id: str,
    sequence: int,
    item_id: str,
    version: str,
) -> SourceChange:
    return SourceChange(
        event_id=event_id,
        sequence=sequence,
        kind=SourceChangeKind.DELETE,
        tombstone=SourceTombstone(
            source_item_id=item_id,
            remote_version=version,
        ),
        occurred_at=NOW,
    )


def _service(
    connector: DeterministicFakeConnector,
    *,
    external_enabled: bool = False,
) -> tuple[
    ConnectorSyncService,
    InMemoryConnectorSyncRepository,
    InMemoryDocumentVersionBoundary,
]:
    registry = ConnectorRegistry()
    registry.register("fake", connector)
    repository = InMemoryConnectorSyncRepository()
    versions = InMemoryDocumentVersionBoundary()
    return (
        ConnectorSyncService(
            registry=registry,
            repository=repository,
            document_versions=versions,
            external_connectors_enabled=external_enabled,
        ),
        repository,
        versions,
    )


def test_domain_rejects_inline_credentials_and_metadata_scope_override():
    with pytest.raises(ValidationError, match="credential-like"):
        DataSource(
            data_source_id="source-a",
            scope=_scope(),
            name="unsafe",
            connector_key="fake",
            config={"api_token": "plaintext"},
        )
    with pytest.raises(ValidationError, match=r"config\.auth\.password"):
        DataSource(
            data_source_id="source-a",
            scope=_scope(),
            name="unsafe nested",
            connector_key="fake",
            config={"auth": {"password": "plaintext"}},
        )
    with pytest.raises(ValidationError, match="url_query"):
        DataSource(
            data_source_id="source-a",
            scope=_scope(),
            name="unsafe URL",
            connector_key="fake",
            config={"endpoint": "https://example.invalid/files?access_token=plaintext"},
        )
    with pytest.raises(ValidationError, match="scope fields"):
        _item("item-a", "v1", b"a").model_copy(
            update={"metadata": {"tenant_id": "tenant-b"}}
        ).model_validate(
            {
                **_item("item-a", "v1", b"a").model_dump(),
                "metadata": {"tenant_id": "tenant-b"},
            }
        )


def test_external_connectors_are_disabled_by_default():
    class _ExternalFake(DeterministicFakeConnector):
        @property
        def capabilities(self) -> ConnectorCapabilities:
            return super().capabilities.model_copy(update={"external_network": True})

    connector = _ExternalFake()
    service, _, _ = _service(connector)
    with pytest.raises(ExternalConnectorDisabledError):
        service.dry_run(_source())


def test_acl_and_partial_cursor_policies_fail_closed_when_adapter_cannot_honor_them():
    connector = DeterministicFakeConnector()
    service, _, _ = _service(connector)
    source = _source().model_copy(
        update={"sync_policy": _source().sync_policy.model_copy(update={"map_remote_acl": True})}
    )
    with pytest.raises(ConnectorConfigurationError, match="ACL"):
        service.dry_run(source)

    partial = _source().model_copy(
        update={
            "sync_policy": _source().sync_policy.model_copy(
                update={"allow_partial_cursor_advance": True}
            )
        }
    )
    with pytest.raises(ConnectorConfigurationError, match="retry ledger"):
        service.dry_run(partial)


def test_dry_run_has_no_fetch_ack_cursor_or_version_side_effects():
    content = b"alpha"
    item = _item("item-a", "v1", content)
    connector = DeterministicFakeConnector(
        (_change("event-a", 0, item),),
        objects={("item-a", "v1"): content},
    )
    service, repository, versions = _service(connector)

    plan = service.dry_run(_source(), run_id="dry-fixed")

    assert [operation.action for operation in plan.operations] == [SyncAction.CREATE_VERSION]
    assert plan.dry_run is True
    assert connector.fetch_calls == []
    assert connector.acked_cursors == []
    assert versions.create_calls == 0
    assert repository.get_cursor(_scope(), "source-a").value == "0"
    assert repository.list_bindings(_scope(), "source-a") == ()


def test_paginated_sync_uses_pending_version_until_publish_boundary():
    first_content, second_content = b"alpha", b"beta"
    first = _item("item-a", "v1", first_content)
    second = _item("item-b", "v1", second_content)
    connector = DeterministicFakeConnector(
        (
            _change("event-a", 0, first),
            _change("event-b", 1, second),
        ),
        objects={
            ("item-a", "v1"): first_content,
            ("item-b", "v1"): second_content,
        },
    )
    service, repository, versions = _service(connector)
    source = _source(page_size=1)

    run_one, plan_one = service.run_once(source, run_id="run-one")
    binding = repository.list_bindings(_scope(), "source-a")[0]

    assert run_one.status.value == "succeeded"
    assert plan_one.has_more is True
    assert repository.get_cursor(_scope(), "source-a").value == "1"
    assert binding.status == SourceBindingStatus.PENDING
    assert binding.active_document_version_id is None
    assert binding.pending_document_version_id is not None

    published = service.mark_version_published(
        scope=_scope(),
        data_source_id="source-a",
        source_item_id="item-a",
        document_version_id=binding.pending_document_version_id,
    )
    assert published.status == SourceBindingStatus.ACTIVE
    assert published.pending_document_version_id is None

    run_two, plan_two = service.run_once(source, run_id="run-two")
    assert plan_two.has_more is False
    assert run_two.applied_count == 1
    assert repository.get_cursor(_scope(), "source-a").value == "2"
    assert versions.create_calls == 2
    assert connector.acked_cursors == ["1", "2"]


def test_identical_event_replay_does_not_create_duplicate_version():
    content = b"same"
    item = _item("item-a", "v1", content)
    event = _change("stable-event", 0, item)
    connector = DeterministicFakeConnector(
        (event, event),
        objects={("item-a", "v1"): content},
    )
    service, repository, versions = _service(connector)
    source = _source(page_size=1)

    service.run_once(source, run_id="first")
    _, replay_plan = service.run_once(source, run_id="replay")

    assert replay_plan.operations[0].action == SyncAction.ALREADY_APPLIED
    assert versions.create_calls == 1
    assert connector.fetch_calls == [("item-a", "v1")]
    assert repository.get_cursor(_scope(), "source-a").value == "2"


def test_mid_page_failure_keeps_cursor_and_retry_reuses_version_idempotently():
    first_content, second_content = b"alpha", b"beta"
    first = _item("item-a", "v1", first_content)
    second = _item("item-b", "v1", second_content)
    connector = DeterministicFakeConnector(
        (
            _change("event-a", 0, first),
            _change("event-b", 1, second),
        ),
        objects={
            ("item-a", "v1"): first_content,
            ("item-b", "v1"): second_content,
        },
        fail_fetch_once_for=("item-b",),
    )
    service, repository, versions = _service(connector)

    with pytest.raises(OSError, match="injected fetch failure"):
        service.run_once(_source(), run_id="failed-run")

    assert repository.get_cursor(_scope(), "source-a").value == "0"
    assert repository.list_bindings(_scope(), "source-a") == ()
    assert versions.create_calls == 1

    completed, _ = service.run_once(_source(), run_id="retry-run")
    assert completed.applied_count == 2
    assert repository.get_cursor(_scope(), "source-a").value == "2"
    assert versions.create_calls == 2


def test_rate_limit_records_failed_run_and_retry_after_without_advancing_cursor():
    connector = DeterministicFakeConnector(retry_after_once=12.5)
    service, repository, _ = _service(connector)

    with pytest.raises(ConnectorRateLimitedError) as captured:
        service.run_once(_source(), run_id="rate-limited")

    assert captured.value.retry_after_seconds == 12.5
    assert repository.get_cursor(_scope(), "source-a").value == "0"
    run = repository.get_run(_scope(), "source-a", "rate-limited")
    assert run is not None
    assert run.status.value == "failed"
    assert run.error_code == "CONNECTOR_RATE_LIMITED"


def test_ack_failure_happens_after_atomic_commit_and_is_recorded_as_partial():
    content = b"alpha"
    item = _item("item-a", "v1", content)
    connector = DeterministicFakeConnector(
        (_change("event-a", 0, item),),
        objects={("item-a", "v1"): content},
        fail_ack_once=True,
    )
    service, repository, versions = _service(connector)

    with pytest.raises(OSError, match="ack failure"):
        service.run_once(_source(), run_id="ack-failed")

    assert repository.get_cursor(_scope(), "source-a").value == "1"
    assert len(repository.list_bindings(_scope(), "source-a")) == 1
    assert versions.create_calls == 1
    run = repository.get_run(_scope(), "source-a", "ack-failed")
    assert run is not None
    assert run.status.value == "partial"
    assert run.applied_count == 1


def test_remote_delete_defaults_to_disable_and_preserves_grace_period():
    content = b"alpha"
    item = _item("item-a", "v1", content)
    connector = DeterministicFakeConnector(
        (
            _change("created", 0, item),
            _delete("deleted", 1, "item-a", "v2"),
        ),
        objects={("item-a", "v1"): content},
    )
    service, repository, versions = _service(connector)
    source = _source(page_size=1)
    service.run_once(source, run_id="create")
    service.run_once(source, run_id="delete")

    binding = repository.list_bindings(_scope(), "source-a")[0]
    assert binding.status == SourceBindingStatus.DISABLED
    assert binding.disabled_at is not None
    assert binding.purge_after is not None
    assert versions.delete_calls == 1


def test_complete_snapshot_detects_missing_item_but_empty_snapshot_is_blocked():
    planner = IncrementalSyncPlanner()
    source = _source()
    cursor = SyncCursor(scope=_scope(), data_source_id="source-a", value="7")
    binding = SourceItemBinding(
        scope=_scope(),
        data_source_id="source-a",
        source_item_id="item-a",
        document_id="doc-a",
        remote_version="v1",
        content_sha256=hashlib.sha256(b"a").hexdigest(),
        logical_name="a.txt",
        locator="a.txt",
        status=SourceBindingStatus.ACTIVE,
    )
    with pytest.raises(ReconciliationSafetyError, match="empty snapshot"):
        planner.plan_reconciliation(
            source=source,
            cursor=cursor,
            snapshot=SourceSnapshot(
                items=(),
                cursor="8",
                complete=True,
                observed_at=NOW,
            ),
            bindings=(binding,),
            run_id="reconcile",
        )

    kept = _item("item-b", "v1", b"b")
    plan = planner.plan_reconciliation(
        source=source,
        cursor=cursor,
        snapshot=SourceSnapshot(
            items=(kept,),
            cursor="8",
            complete=True,
            observed_at=NOW,
        ),
        bindings=(binding,),
        run_id="reconcile-safe",
    )
    assert {operation.action for operation in plan.operations} == {
        SyncAction.CREATE_VERSION,
        SyncAction.DISABLE_DOCUMENT,
    }
    assert plan.reconciliation is True


def test_scope_mismatch_is_rejected_and_same_ids_remain_isolated():
    planner = IncrementalSyncPlanner()
    source = _source()
    foreign_cursor = SyncCursor(
        scope=_scope("tenant-b", "kb-a"),
        data_source_id="source-a",
    )
    with pytest.raises(ConnectorScopeError):
        planner.plan_incremental(
            source=source,
            cursor=foreign_cursor,
            page=DeterministicFakeConnector().list_changes(
                source,
                foreign_cursor,
                10,
            ),
            bindings=(),
            run_id="foreign",
        )

    repository = InMemoryConnectorSyncRepository()
    assert (
        repository.get_cursor(_scope("tenant-a", "kb-a"), "source-a").scope
        != repository.get_cursor(_scope("tenant-a", "kb-b"), "source-a").scope
    )


def test_hash_mismatch_fails_before_document_version_registration():
    item = _item(
        "item-a",
        "v1",
        b"declared",
        declared_hash=hashlib.sha256(b"declared").hexdigest(),
    )
    connector = DeterministicFakeConnector(
        (_change("event-a", 0, item),),
        objects={("item-a", "v1"): b"tampered"},
    )
    service, repository, versions = _service(connector)

    with pytest.raises(ContentIntegrityError, match="hash"):
        service.run_once(_source(), run_id="tampered")

    assert versions.create_calls == 0
    assert repository.get_cursor(_scope(), "source-a").value == "0"


def test_rename_with_same_hash_only_updates_metadata():
    content = b"same"
    original = _item("item-a", "v1", content, logical_name="old.txt")
    renamed = _item(
        "item-a",
        "v2",
        content,
        logical_name="new.txt",
        locator="moved/new.txt",
    )
    connector = DeterministicFakeConnector(
        (
            _change("create", 0, original),
            _change("rename", 1, renamed, kind=SourceChangeKind.RENAME),
        ),
        objects={("item-a", "v1"): content},
    )
    service, repository, versions = _service(connector)
    source = _source(page_size=1)
    service.run_once(source, run_id="create")
    _, plan = service.run_once(source, run_id="rename")

    assert plan.operations[0].action == SyncAction.UPDATE_METADATA
    assert versions.create_calls == 1
    assert versions.metadata_calls == 1
    binding = repository.list_bindings(_scope(), "source-a")[0]
    assert binding.logical_name == "new.txt"
    assert binding.remote_version == "v2"
