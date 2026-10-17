"""TODO-09 检索配置管理与预览请求模型。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalProfileDraftCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)
    profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    config: dict[str, Any]
    expected_current_profile_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    expected_current_version: int | None = Field(default=None, ge=1)
    expected_current_config_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expected_control_plane_revision: int | None = Field(default=None, ge=1)
    expected_max_version: int | None = Field(default=None, ge=0)


class RetrievalProfilePublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    expected_current_profile_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    expected_current_version: int | None = Field(default=None, ge=1)
    expected_current_config_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expected_control_plane_revision: int | None = Field(default=None, ge=1)
    expected_max_version: int | None = Field(default=None, ge=0)


class RetrievalProfileRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1)
    expected_current_profile_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    expected_current_version: int | None = Field(default=None, ge=1)
    expected_current_config_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expected_control_plane_revision: int | None = Field(default=None, ge=1)
    expected_max_version: int | None = Field(default=None, ge=0)


class RetrievalProfilePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=32_000)
    retrieval_filter: dict[str, Any] | None = None


class RetrievalProfileListFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["draft", "published", "retired"] | None = None
