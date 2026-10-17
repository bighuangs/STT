"""Typed HTTP contracts for the TODO-12 import workbench.

Storage locators deliberately do not appear in these models.  Source files and
assets are exposed only through authenticated streaming endpoints.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType


T = TypeVar("T")
JsonValue = TypeAliasType(
    "WorkbenchJsonValue",
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]


class WorkbenchPublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkbenchPage(WorkbenchPublicModel, Generic[T]):
    schema_version: Literal["workbench-page.v1"] = "workbench-page.v1"
    items: list[T]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    next_cursor: str | None = None


class WorkbenchEnvelope(WorkbenchPublicModel, Generic[T]):
    request_id: str
    data: T


class WorkbenchAccess(WorkbenchPublicModel):
    grant_role: str


class KnowledgeBaseAggregates(WorkbenchPublicModel):
    document_count: int = Field(ge=0)
    active_chunk_count: int = Field(ge=0)
    active_source_bytes: int = Field(ge=0)
    processing_job_count: int = Field(ge=0)
    latest_failure_code: str | None = None


class KnowledgeBaseView(WorkbenchPublicModel):
    id: str
    name: str
    description: str
    status: str
    data_classification: str
    external_llm_allowed: bool
    web_search_allowed: bool
    retrieval_profile_id: str | None = None
    active_index_version_id: str | None = None
    readiness: str
    aggregates: KnowledgeBaseAggregates
    access: WorkbenchAccess
    actions: list[str]
    etag: str
    created_at: str | None = None
    updated_at: str | None = None


class JobSummaryView(WorkbenchPublicModel):
    id: str
    document_version_id: str
    index_version_id: str
    operation: str
    status: str
    stage: str
    progress: float
    attempt: int
    run_attempt: int
    max_attempts: int
    next_retry_at: str | None = None
    error_category: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    checkpoint_stage: str | None = None
    checkpoint_sequence: int
    last_event_sequence: int
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class JobListItemView(JobSummaryView):
    document_id: str
    logical_name: str
    kb_id: str
    actions: list[str]


class ParseArtifactView(WorkbenchPublicModel):
    id: str
    parser_id: str
    parser_version: str
    schema_version: str
    status: str
    artifact_sha256: str | None = None
    artifact_size: int = Field(ge=0)
    warnings: list[JsonValue]
    metrics: JsonObject


class ChunkingRunView(WorkbenchPublicModel):
    id: str
    parse_artifact_id: str
    template_id: str
    template_version: str
    config_hash: str
    status: str
    artifact_sha256: str | None = None
    artifact_size: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class DocumentVersionView(WorkbenchPublicModel):
    id: str
    document_id: str
    version_no: int = Field(ge=1)
    content_sha256: str
    pipeline_fingerprint: str
    size_bytes: int = Field(ge=0)
    mime_type: str
    parser_config: JsonObject
    status: str
    published_at: str | None = None
    created_at: str | None = None
    parse_artifacts: list[ParseArtifactView]
    chunking_runs: list[ChunkingRunView]
    jobs: list[JobSummaryView]
    source_url: str
    blocks_url: str
    chunks_url: str


class DocumentSummaryView(WorkbenchPublicModel):
    id: str
    kb_id: str
    logical_name: str
    source_type: str
    status: str
    active_version_id: str | None = None
    version_count: int = Field(ge=0)
    latest_version_no: int = Field(ge=0)
    active_chunk_count: int = Field(ge=0)
    latest_job: JobSummaryView | None = None
    actions: list[str]
    etag: str
    created_at: str | None = None
    updated_at: str | None = None


class DocumentDetailView(DocumentSummaryView):
    versions: list[DocumentVersionView]


class BlockAssetView(WorkbenchPublicModel):
    asset_id: str
    media_type: str
    width: int | None = None
    height: int | None = None
    size_bytes: int = Field(ge=0)


class BlockView(WorkbenchPublicModel):
    block_id: str
    type: str
    text: str
    structured_data: JsonValue | None = None
    location: JsonObject
    parent_heading_path: list[str]
    order: int = Field(ge=0)
    assets: list[BlockAssetView]


class ChunkView(WorkbenchPublicModel):
    chunk_id: str
    content: str
    type: str
    source_block_ids: list[str]
    parent_heading_path: list[str]
    location: JsonValue
    metadata: JsonObject
    has_images: bool


class DiffChunkView(ChunkView):
    content_hash: str


class VersionDiffHunkView(WorkbenchPublicModel):
    alignment_key: str
    kind: Literal["added", "removed", "modified"]
    before: DiffChunkView | None = None
    after: DiffChunkView | None = None


class VersionDiffSummaryView(WorkbenchPublicModel):
    left_chunk_count: int = Field(ge=0)
    right_chunk_count: int = Field(ge=0)
    changed_hunk_count: int = Field(ge=0)
    parser_config_changed: bool
    content_changed: bool


class VersionDiffView(WorkbenchPublicModel):
    schema_version: Literal["document-version-diff.v1"]
    document_id: str
    left: DocumentVersionView
    right: DocumentVersionView
    summary: VersionDiffSummaryView
    hunks: WorkbenchPage[VersionDiffHunkView]


class JobDocumentView(WorkbenchPublicModel):
    id: str
    kb_id: str
    logical_name: str
    version_id: str
    version_no: int = Field(ge=1)


class JobEventView(WorkbenchPublicModel):
    sequence: int = Field(ge=1)
    event_type: str
    status: str
    stage: str
    progress: float
    payload: JsonObject
    created_at: str | None = None


class JobCheckpointView(WorkbenchPublicModel):
    id: str
    sequence: int = Field(ge=1)
    stage: str
    resume_node: str
    artifact_sha256: str
    artifact_size: int = Field(ge=0)
    input_sha256: str
    config_sha256: str
    created_at: str | None = None


class JobManifestView(WorkbenchPublicModel):
    id: str
    document_version_id: str
    index_version_id: str
    expected_chunk_count: int = Field(ge=0)
    written_chunk_count: int = Field(ge=0)
    searchable_chunk_count: int = Field(ge=0)
    chunk_ids_sha256: str | None = None
    vector_dim: int = Field(ge=0)
    validation_status: str
    validation_errors: list[JsonValue]
    validated_at: str | None = None


class JobArtifactView(WorkbenchPublicModel):
    id: str
    kind: str
    schema_version: str
    sha256: str | None = None
    size: int = Field(ge=0)
    status: str


class JobDetailView(JobSummaryView):
    document: JobDocumentView
    actions: list[str]
    events: list[JobEventView]
    checkpoints: list[JobCheckpointView]
    manifest: JobManifestView | None = None
    warnings: list[JsonValue]
    artifacts: list[JobArtifactView]


class ChunkReplacementView(WorkbenchPublicModel):
    content: str
    type: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
    source_block_ids: list[str] | None = None
    parent_heading_path: list[str] | None = None


class ChunkEditOperationView(WorkbenchPublicModel):
    id: str
    sequence: int = Field(ge=1)
    operation: Literal["replace", "split", "merge", "delete"]
    input_chunk_ids: list[str]
    output_chunks: list[ChunkReplacementView]
    created_at: str | None = None


class ChunkOperationResultView(WorkbenchPublicModel):
    input_chunk_ids: list[str]
    output_chunk_ids: list[str]


class PublishedVersionPointerView(WorkbenchPublicModel):
    id: str
    version_no: int = Field(ge=1)
    status: str


class PublishedIndexPointerView(WorkbenchPublicModel):
    id: str
    version_no: int = Field(ge=1)
    status: str


class ChunkDraftView(WorkbenchPublicModel):
    id: str
    kb_id: str
    document_id: str
    document_version_id: str
    base_chunking_run_id: str
    status: str
    revision: int = Field(ge=1)
    frozen_revision: int | None = Field(default=None, ge=1)
    etag: str
    operation_count: int = Field(ge=0)
    operations: list[ChunkEditOperationView]
    materialized_chunk_count: int | None = Field(default=None, ge=1)
    operation_result: ChunkOperationResultView | None = None
    published_document_version_id: str | None = None
    published_index_version_id: str | None = None
    published_job_id: str | None = None
    document_version: PublishedVersionPointerView | None = None
    index_version: PublishedIndexPointerView | None = None
    job: JobSummaryView | None = None
    published_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class BatchDocumentPointerView(WorkbenchPublicModel):
    id: str
    status: str


class BatchCleanupJobView(WorkbenchPublicModel):
    id: str
    status: str
    mode: str


class BatchMutationResultView(WorkbenchPublicModel):
    document: BatchDocumentPointerView | None = None
    cleanup_job: BatchCleanupJobView | None = None
    job: JobSummaryView | None = None


class BatchErrorView(WorkbenchPublicModel):
    code: str
    message: str
    current_etag: str | None = None


class BatchItemResultView(WorkbenchPublicModel):
    document_id: str
    ok: bool
    status: int
    etag: str | None = None
    result: BatchMutationResultView | None = None
    error: BatchErrorView | None = None


class DocumentBatchResultView(WorkbenchPublicModel):
    schema_version: Literal["document-batch-result.v1"]
    action: Literal["enable", "disable", "soft_delete", "purge", "retry"]
    items: list[BatchItemResultView]
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)


class KnowledgeBaseWorkbenchUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "disabled"] | None = None
    data_classification: Literal[
        "public", "internal", "confidential", "restricted"
    ] | None = None
    external_llm_allowed: bool | None = None
    web_search_allowed: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> "KnowledgeBaseWorkbenchUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one knowledge-base field is required")
        return self


class DocumentBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=64)
    etag: str = Field(min_length=3, max_length=255)


class DocumentBatchActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["enable", "disable", "soft_delete", "purge", "retry"]
    items: list[DocumentBatchItem] = Field(min_length=1, max_length=100)

    @field_validator("items")
    @classmethod
    def unique_documents(cls, value: list[DocumentBatchItem]) -> list[DocumentBatchItem]:
        identifiers = [item.document_id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("batch document ids must be unique")
        return value


class ChunkDraftCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_chunking_run_id: str | None = Field(default=None, max_length=64)


class ChunkReplacement(BaseModel):
    """A safe editable projection; vectors and storage identities are never accepted."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1_000_000)
    type: Literal["text", "heading", "table", "image", "list", "code", "hyperlinks"] | None = None
    metadata: JsonObject = Field(default_factory=dict)
    source_block_ids: list[str] | None = Field(default=None, max_length=2048)
    parent_heading_path: list[str] | None = Field(default=None, max_length=128)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk content cannot be blank")
        return value

    @field_validator("source_block_ids", "parent_heading_path")
    @classmethod
    def safe_string_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [str(item).strip() for item in value]
        if any(not item or len(item) > 128 for item in normalized):
            raise ValueError("chunk provenance identifiers must contain 1..128 characters")
        if len(normalized) != len(set(normalized)):
            raise ValueError("chunk provenance identifiers must be unique")
        return normalized


class ChunkReplaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: ChunkReplacement


class ChunkSplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[ChunkReplacement] = Field(min_length=2, max_length=20)


class ChunkMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[str] = Field(min_length=2, max_length=100)
    chunk: ChunkReplacement

    @field_validator("chunk_ids")
    @classmethod
    def unique_chunk_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or len(item) > 128 for item in normalized):
            raise ValueError("chunk ids must contain 1..128 characters")
        if len(normalized) != len(set(normalized)):
            raise ValueError("chunk ids must be unique")
        return normalized


class ChunkDraftPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str = Field(default="", max_length=2000)


class BlockPageQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cursor: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)
    q: str | None = Field(default=None, max_length=1000)
    block_type: str | None = Field(default=None, max_length=64)
    page_no: int | None = Field(default=None, ge=1)


__all__ = [
    "BatchCleanupJobView",
    "BatchDocumentPointerView",
    "BatchErrorView",
    "BatchItemResultView",
    "BatchMutationResultView",
    "BlockAssetView",
    "BlockPageQuery",
    "BlockView",
    "ChunkDraftCreateRequest",
    "ChunkDraftPublishRequest",
    "ChunkDraftView",
    "ChunkEditOperationView",
    "ChunkMergeRequest",
    "ChunkOperationResultView",
    "ChunkReplaceRequest",
    "ChunkReplacement",
    "ChunkReplacementView",
    "ChunkSplitRequest",
    "ChunkView",
    "ChunkingRunView",
    "DiffChunkView",
    "DocumentBatchActionRequest",
    "DocumentBatchResultView",
    "DocumentDetailView",
    "DocumentSummaryView",
    "DocumentVersionView",
    "JobArtifactView",
    "JobCheckpointView",
    "JobDetailView",
    "JobDocumentView",
    "JobEventView",
    "JobListItemView",
    "JobManifestView",
    "JobSummaryView",
    "KnowledgeBaseAggregates",
    "KnowledgeBaseView",
    "KnowledgeBaseWorkbenchUpdate",
    "ParseArtifactView",
    "PublishedIndexPointerView",
    "PublishedVersionPointerView",
    "VersionDiffHunkView",
    "VersionDiffSummaryView",
    "VersionDiffView",
    "WorkbenchAccess",
    "WorkbenchEnvelope",
    "WorkbenchPage",
    "WorkbenchPublicModel",
]
