"""TODO-15 的确定性纯内存控制库。

该实现不是生产持久化替代品；它完整实现应用端口，适用于离线测试、管理命令预演
和 SQL repository 的契约测试基线。所有返回值都按稳定键排序，避免测试依赖 dict
插入顺序。
"""

from __future__ import annotations

from threading import RLock

from app.domain.knowledge_governance import (
    EnhancementRun,
    MetadataAuditEvent,
    MetadataRelease,
    MetadataSchemaVersion,
    MetadataValueRecord,
    ProjectionState,
    TagAssignmentRecord,
    TagTaxonomyVersion,
    UnknownTagCandidate,
)


class InMemoryMetadataGovernanceRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._revisions: dict[tuple[str, str], int] = {}
        self._schemas: dict[tuple[str, str, str, int], MetadataSchemaVersion] = {}
        self._taxonomies: dict[tuple[str, str, str, int], TagTaxonomyVersion] = {}
        self._values: dict[str, MetadataValueRecord] = {}
        self._tag_assignments: dict[str, TagAssignmentRecord] = {}
        self._unknown_tags: dict[str, UnknownTagCandidate] = {}
        self._releases: dict[tuple[str, str, int], MetadataRelease] = {}
        self._current_releases: dict[tuple[str, str], int] = {}
        self._projections: dict[tuple[str, str, str], ProjectionState] = {}
        self._enhancement_runs: dict[str, EnhancementRun] = {}
        self._audit: dict[str, MetadataAuditEvent] = {}

    def next_revision(self, tenant_id: str, kb_id: str) -> int:
        with self._lock:
            key = (tenant_id, kb_id)
            revision = self._revisions.get(key, 0) + 1
            self._revisions[key] = revision
            return revision

    def save_schema(self, schema: MetadataSchemaVersion) -> None:
        with self._lock:
            self._schemas[(schema.tenant_id, schema.kb_id, schema.schema_id, schema.revision)] = (
                schema
            )

    def get_schema(
        self,
        tenant_id: str,
        kb_id: str,
        schema_id: str,
        revision: int,
    ) -> MetadataSchemaVersion | None:
        with self._lock:
            return self._schemas.get((tenant_id, kb_id, schema_id, revision))

    def list_schemas(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataSchemaVersion, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for (tenant, kb, _, _), value in self._schemas.items()
                        if tenant == tenant_id and kb == kb_id
                    ),
                    key=lambda item: (item.schema_id, item.version, item.revision),
                )
            )

    def save_taxonomy(self, taxonomy: TagTaxonomyVersion) -> None:
        with self._lock:
            self._taxonomies[
                (
                    taxonomy.tenant_id,
                    taxonomy.kb_id,
                    taxonomy.taxonomy_id,
                    taxonomy.revision,
                )
            ] = taxonomy

    def get_taxonomy(
        self,
        tenant_id: str,
        kb_id: str,
        taxonomy_id: str,
        revision: int,
    ) -> TagTaxonomyVersion | None:
        with self._lock:
            return self._taxonomies.get((tenant_id, kb_id, taxonomy_id, revision))

    def list_taxonomies(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[TagTaxonomyVersion, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for (tenant, kb, _, _), value in self._taxonomies.items()
                        if tenant == tenant_id and kb == kb_id
                    ),
                    key=lambda item: (item.taxonomy_id, item.version, item.revision),
                )
            )

    def save_value(self, value: MetadataValueRecord) -> None:
        with self._lock:
            self._values[value.record_id] = value

    def get_value(self, record_id: str) -> MetadataValueRecord | None:
        with self._lock:
            return self._values.get(record_id)

    def list_values(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataValueRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for value in self._values.values()
                        if value.target.tenant_id == tenant_id and value.target.kb_id == kb_id
                    ),
                    key=lambda item: (item.proposed_at, item.record_id),
                )
            )

    def save_tag_assignment(self, assignment: TagAssignmentRecord) -> None:
        with self._lock:
            self._tag_assignments[assignment.assignment_id] = assignment

    def get_tag_assignment(self, assignment_id: str) -> TagAssignmentRecord | None:
        with self._lock:
            return self._tag_assignments.get(assignment_id)

    def list_tag_assignments(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[TagAssignmentRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for value in self._tag_assignments.values()
                        if value.target.tenant_id == tenant_id and value.target.kb_id == kb_id
                    ),
                    key=lambda item: (item.proposed_at, item.assignment_id),
                )
            )

    def save_unknown_tag(self, candidate: UnknownTagCandidate) -> None:
        with self._lock:
            self._unknown_tags[candidate.candidate_id] = candidate

    def list_unknown_tags(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[UnknownTagCandidate, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        value
                        for value in self._unknown_tags.values()
                        if value.target.tenant_id == tenant_id and value.target.kb_id == kb_id
                    ),
                    key=lambda item: (item.proposed_at, item.candidate_id),
                )
            )

    def save_release(self, release: MetadataRelease) -> None:
        with self._lock:
            key = (release.tenant_id, release.kb_id, release.revision)
            self._releases[key] = release
            self._current_releases[(release.tenant_id, release.kb_id)] = release.revision

    def current_release(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> MetadataRelease | None:
        with self._lock:
            revision = self._current_releases.get((tenant_id, kb_id))
            return None if revision is None else self._releases.get((tenant_id, kb_id, revision))

    def get_release(
        self,
        tenant_id: str,
        kb_id: str,
        revision: int,
    ) -> MetadataRelease | None:
        with self._lock:
            return self._releases.get((tenant_id, kb_id, revision))

    def list_releases(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataRelease, ...]:
        with self._lock:
            return tuple(
                value
                for (_, _, _), value in sorted(
                    (
                        (key, value)
                        for key, value in self._releases.items()
                        if key[0] == tenant_id and key[1] == kb_id
                    ),
                    key=lambda item: item[0][2],
                )
            )

    def save_projection_state(self, state: ProjectionState) -> None:
        with self._lock:
            self._projections[(state.tenant_id, state.kb_id, state.projection_name)] = state

    def get_projection_state(
        self,
        tenant_id: str,
        kb_id: str,
        projection_name: str,
    ) -> ProjectionState | None:
        with self._lock:
            return self._projections.get((tenant_id, kb_id, projection_name))

    def save_enhancement_run(self, run: EnhancementRun) -> None:
        with self._lock:
            self._enhancement_runs[run.run_id] = run

    def get_enhancement_run(self, run_id: str) -> EnhancementRun | None:
        with self._lock:
            return self._enhancement_runs.get(run_id)

    def append_audit(self, event: MetadataAuditEvent) -> None:
        with self._lock:
            self._audit[event.event_id] = event

    def list_audit(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[MetadataAuditEvent, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        event
                        for event in self._audit.values()
                        if event.tenant_id == tenant_id and event.kb_id == kb_id
                    ),
                    key=lambda item: (item.occurred_at, item.event_id),
                )
            )


__all__ = ["InMemoryMetadataGovernanceRepository"]
