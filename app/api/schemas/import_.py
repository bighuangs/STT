"""
应用主包 / 接口层 / 数据模型层中的 import_ 模块，负责承载对应场景的具体实现逻辑。
"""
from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    code: int = 200
    message: str
    task_ids: list[str]
    document_ids: list[str] = Field(default_factory=list)
    document_version_ids: list[str] = Field(default_factory=list)
    index_version_ids: list[str] = Field(default_factory=list)
    parser_profiles: list[dict[str, Any]] = Field(default_factory=list)


class ImportStatusResponse(BaseModel):
    code: int = 200
    task_id: str
    status: str | None = None
    done_list: list[str]
    running_list: list[str]
    document_id: str | None = None
    document_version_id: str | None = None
    index_version_id: str | None = None
    stage: str | None = None
    progress: float | None = None
    error_code: str | None = None


class ParsePreviewResponse(BaseModel):
    request_id: str
    detected_format: str
    detected_mime_type: str
    parser_id: str
    parser_version: str
    chunking_template_id: str
    chunking_template_version: str
    suggested_chunking_template_id: str
    parsed_schema_version: str
    artifact: dict[str, Any]
    metadata: dict[str, Any]
    warnings: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    total_blocks: int
    total_chunks: int
