"""TODO-01 控制面 API schema。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)
    data_classification: Literal["public", "internal", "confidential", "restricted"] = "internal"
    external_llm_allowed: bool = True
    web_search_allowed: bool = False


class KnowledgeBaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "disabled"] | None = None
    data_classification: Literal["public", "internal", "confidential", "restricted"] | None = None
    external_llm_allowed: bool | None = None
    web_search_allowed: bool | None = None


class ResourceEnvelope(BaseModel):
    request_id: str
    data: Any


class IndexRebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model_id: str | None = Field(default=None, max_length=255)
    vector_dim: int | None = Field(default=None, ge=1, le=65536)
    schema_version: str | None = Field(default=None, max_length=64)


class DocumentUploadResponse(BaseModel):
    request_id: str
    document_id: str
    document_version_id: str
    ingestion_job_id: str
    index_version_id: str
    version_no: int
    deduplicated: bool
    reused_job: bool
    idempotency_replayed: bool = False
    status: str
    detected_format: str = ""
    detected_mime_type: str = "application/octet-stream"
    parser_id: str = ""
    parser_version: str = ""
    chunking_template_id: str = "general"
    chunking_template_version: str = ""
    suggested_chunking_template_id: str = "general"
    parse_warnings: list[dict[str, Any]] = Field(default_factory=list)
