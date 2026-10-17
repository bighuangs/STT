"""Phoenix/OpenTelemetry 观测配置。"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PhoenixObservabilitySettings(BaseSettings):
    """从环境变量读取 Phoenix 配置，默认不启用任何外部导出。"""

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    enabled: bool = Field(default=False, validation_alias="PHOENIX_ENABLED")
    collector_endpoint: str = Field(
        default="http://localhost:6006",
        validation_alias="PHOENIX_COLLECTOR_ENDPOINT",
    )
    project_name: str = Field(
        default="stt",
        validation_alias=AliasChoices("PHOENIX_PROJECT", "PHOENIX_PROJECT_NAME"),
    )
    protocol: Literal["http/protobuf", "grpc"] = Field(
        default="http/protobuf",
        validation_alias="PHOENIX_PROTOCOL",
    )
    batch: bool = Field(default=True, validation_alias="PHOENIX_BATCH")
    auto_instrument: bool = Field(
        default=True,
        validation_alias="PHOENIX_AUTO_INSTRUMENT",
    )
    api_key_value: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="PHOENIX_API_KEY",
        repr=False,
    )

    @field_validator("collector_endpoint")
    @classmethod
    def validate_collector_endpoint(cls, value: str) -> str:
        """拒绝会把凭据或查询参数混入日志和 exporter 配置的 URL。"""
        endpoint = value.strip().rstrip("/")
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Phoenix collector endpoint 必须是 http(s) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Phoenix collector endpoint 不得包含凭据、query 或 fragment")
        return endpoint

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, value: str) -> str:
        """项目名必须可用于稳定聚合同一应用的 trace。"""
        project_name = value.strip()
        if not project_name:
            raise ValueError("Phoenix project name 不得为空")
        return project_name

    @property
    def export_endpoint(self) -> str:
        """返回 SDK `endpoint` 参数所需的完整 OTLP trace 地址。"""
        if self.protocol == "grpc":
            return self.collector_endpoint
        parts = urlsplit(self.collector_endpoint)
        path = parts.path.rstrip("/")
        if not path.endswith("/v1/traces"):
            path = f"{path}/v1/traces"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
