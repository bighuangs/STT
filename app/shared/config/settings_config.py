"""TODO-05 统一、强类型且默认脱敏的应用配置入口。"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from threading import RLock
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEV_TOKEN_SECRET = "dev-only-change-this-stt-token-secret-2026"
DEV_ADMIN_PASSWORD = "Stt-Admin-ChangeMe!"


def _secret(value: SecretStr) -> str:
    return value.get_secret_value()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _safe_url(value: str) -> str:
    """保留定位所需信息，但删除 URL userinfo、query 和 fragment。"""
    if value.startswith("sqlite"):
        return value
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)


class ServerSettings(FrozenConfig):
    import_app_name: str
    query_app_name: str
    app_env: str
    host: str
    import_port: int
    query_port: int
    cors_origins: tuple[str, ...]


class DatabaseSettings(FrozenConfig):
    control_url: SecretStr


class QueueSettings(FrozenConfig):
    worker_id: str
    poll_seconds: float
    lease_seconds: int
    heartbeat_seconds: int
    max_attempts: int
    retry_base_seconds: float
    retry_max_seconds: float
    checkpoint_root: str


class LLMSettings(FrozenConfig):
    base_url: str
    api_key_value: SecretStr
    lv_model: str
    llm_model: str
    llm_temperature: float
    secret_version: str

    @property
    def api_key(self) -> str:
        return _secret(self.api_key_value)


class EmbeddingSettings(FrozenConfig):
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool


class RerankerSettings(FrozenConfig):
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool
    batch_size: int
    max_input_tokens: int
    query_max_tokens: int
    document_max_tokens: int
    failure_mode: str
    min_available_physical_mib: int
    min_available_commit_mib: int


class MilvusSettings(FrozenConfig):
    milvus_url: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str


class MongoSettings(FrozenConfig):
    url_value: SecretStr
    database: str
    secret_version: str

    @property
    def url(self) -> str:
        return _secret(self.url_value)


class MinIOSettings(FrozenConfig):
    endpoint: str
    access_key_value: SecretStr
    secret_key_value: SecretStr
    bucket_name: str
    minio_img_dir: str
    minio_secure: bool
    secret_version: str

    @property
    def access_key(self) -> str:
        return _secret(self.access_key_value)

    @property
    def secret_key(self) -> str:
        return _secret(self.secret_key_value)


class MinerUSettings(FrozenConfig):
    base_url: str
    api_key_value: SecretStr
    secret_version: str

    @property
    def api_key(self) -> str:
        return _secret(self.api_key_value)


class MCPSettings(FrozenConfig):
    mcp_base_url: str
    api_key_value: SecretStr

    @property
    def api_key(self) -> str:
        return _secret(self.api_key_value)


class SecuritySettings(FrozenConfig):
    token_secret_value: SecretStr
    token_secret_version: str
    previous_token_secret_value: SecretStr
    previous_token_secret_version: str
    token_issuer: str
    access_token_minutes: int
    refresh_token_days: int
    bootstrap_admin_email: str
    bootstrap_admin_password_value: SecretStr

    @property
    def token_secret(self) -> str:
        return _secret(self.token_secret_value)

    @property
    def token_secrets(self) -> tuple[str, ...]:
        values = (self.token_secret, _secret(self.previous_token_secret_value))
        return tuple(value for value in values if value)

    @property
    def bootstrap_admin_password(self) -> str:
        return _secret(self.bootstrap_admin_password_value)


class HealthSettings(FrozenConfig):
    timeout_seconds: float
    cache_ttl_seconds: float
    startup_require_ready: bool
    import_required: tuple[str, ...]
    query_required: tuple[str, ...]
    worker_required: tuple[str, ...]


class LoggingSettings(FrozenConfig):
    console_enabled: bool
    console_level: str
    file_enabled: bool
    file_level: str
    file_retention: str


class AppSettings(BaseSettings):
    """所有部署环境变量的唯一读取入口；旧字段通过只读属性兼容一版。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
        frozen=True,
    )

    import_app_name: str = Field(
        "Enterprise RAG Import Service", validation_alias="IMPORT_APP_NAME"
    )
    query_app_name: str = Field("Enterprise RAG Query Service", validation_alias="QUERY_APP_NAME")
    app_env: str = Field("dev", validation_alias="APP_ENV")
    app_host: str = Field("0.0.0.0", validation_alias="APP_HOST")
    import_app_port: int = Field(8000, validation_alias="IMPORT_APP_PORT")
    query_app_port: int = Field(8001, validation_alias="QUERY_APP_PORT")
    project_root: str = Field("", validation_alias="PROJECT_ROOT")
    cors_origins_csv: str = Field(
        "http://127.0.0.1:8000,http://127.0.0.1:8001", validation_alias="CORS_ORIGINS"
    )

    control_database_url_value: SecretStr = Field(
        SecretStr("sqlite:///./volumes/control/stt-control.db"),
        validation_alias="CONTROL_DATABASE_URL",
    )
    max_upload_bytes: int = Field(100 * 1024 * 1024, validation_alias="MAX_UPLOAD_BYTES")

    ingestion_worker_id: str = Field("", validation_alias="INGESTION_WORKER_ID")
    ingestion_worker_poll_seconds: float = Field(
        1, validation_alias="INGESTION_WORKER_POLL_SECONDS"
    )
    ingestion_lease_seconds: int = Field(60, validation_alias="INGESTION_LEASE_SECONDS")
    ingestion_heartbeat_seconds: int = Field(15, validation_alias="INGESTION_HEARTBEAT_SECONDS")
    ingestion_max_attempts: int = Field(5, validation_alias="INGESTION_MAX_ATTEMPTS")
    ingestion_retry_base_seconds: float = Field(2, validation_alias="INGESTION_RETRY_BASE_SECONDS")
    ingestion_retry_max_seconds: float = Field(300, validation_alias="INGESTION_RETRY_MAX_SECONDS")
    ingestion_checkpoint_root: str = Field(
        "./volumes/control/checkpoints", validation_alias="INGESTION_CHECKPOINT_ROOT"
    )

    auth_token_secret_value: SecretStr = Field(
        SecretStr(DEV_TOKEN_SECRET), validation_alias="AUTH_TOKEN_SECRET"
    )
    auth_token_secret_version: str = Field("env-v1", validation_alias="AUTH_TOKEN_SECRET_VERSION")
    auth_previous_token_secret_value: SecretStr = Field(
        SecretStr(""), validation_alias="AUTH_PREVIOUS_TOKEN_SECRET"
    )
    auth_previous_token_secret_version: str = Field(
        "", validation_alias="AUTH_PREVIOUS_TOKEN_SECRET_VERSION"
    )
    auth_token_issuer: str = Field("stt", validation_alias="AUTH_TOKEN_ISSUER")
    auth_access_token_minutes: int = Field(15, validation_alias="AUTH_ACCESS_TOKEN_MINUTES")
    auth_refresh_token_days: int = Field(7, validation_alias="AUTH_REFRESH_TOKEN_DAYS")
    auth_bootstrap_admin_email: str = Field(
        "admin@stt.local", validation_alias="AUTH_BOOTSTRAP_ADMIN_EMAIL"
    )
    auth_bootstrap_admin_password_value: SecretStr = Field(
        SecretStr(DEV_ADMIN_PASSWORD), validation_alias="AUTH_BOOTSTRAP_ADMIN_PASSWORD"
    )

    openai_base_url: str = Field("", validation_alias="OPENAI_BASE_URL")
    openai_api_key_value: SecretStr = Field(SecretStr(""), validation_alias="OPENAI_API_KEY")
    openai_api_key_version: str = Field("env-v1", validation_alias="OPENAI_API_KEY_VERSION")
    llm_default_model: str = Field("", validation_alias="LLM_DEFAULT_MODEL")
    llm_default_temperature: float = Field(0.1, validation_alias="LLM_DEFAULT_TEMPERATURE")
    vl_model: str = Field("", validation_alias="VL_MODEL")
    llm_request_timeout: float = Field(60, validation_alias="LLM_REQUEST_TIMEOUT")
    llm_max_connections: int = Field(20, validation_alias="LLM_MAX_CONNECTIONS")
    max_concurrent_chats: int = Field(10, validation_alias="MAX_CONCURRENT_CHATS")
    llm_max_attempts: int = Field(3, validation_alias="LLM_MAX_ATTEMPTS")
    llm_retry_base_delay: float = Field(0.6, validation_alias="LLM_RETRY_BASE_DELAY")
    llm_retry_max_delay: float = Field(8, validation_alias="LLM_RETRY_MAX_DELAY")

    # P3 查询运行、可靠事件、安全缓存与外发白名单。
    # QueryRun 覆盖规划、检索、重排和生成；应大于内建本地检索 profile 的
    # 90 秒预算，同时保持明确的全链路绝对截止时间。
    query_default_deadline_ms: int = Field(120_000, validation_alias="QUERY_DEFAULT_DEADLINE_MS")
    query_recovery_interval_seconds: int = Field(
        5, validation_alias="QUERY_RECOVERY_INTERVAL_SECONDS"
    )
    query_checkpoint_encryption_key: SecretStr | None = Field(
        None,
        validation_alias="QUERY_CHECKPOINT_ENCRYPTION_KEY",
    )
    query_event_retention_seconds: int = Field(
        604_800, validation_alias="QUERY_EVENT_RETENTION_SECONDS"
    )
    query_event_batch_size: int = Field(100, validation_alias="QUERY_EVENT_BATCH_SIZE")
    query_stream_max_lag_events: int = Field(1_000, validation_alias="QUERY_STREAM_MAX_LAG_EVENTS")
    query_stream_max_batch_bytes: int = Field(
        1_048_576, validation_alias="QUERY_STREAM_MAX_BATCH_BYTES"
    )
    query_cache_ttl_seconds: int = Field(300, validation_alias="QUERY_CACHE_TTL_SECONDS")
    query_idempotency_ttl_seconds: int = Field(
        86_400, validation_alias="QUERY_IDEMPOTENCY_TTL_SECONDS"
    )
    query_singleflight_lease_seconds: int = Field(
        30, validation_alias="QUERY_SINGLEFLIGHT_LEASE_SECONDS"
    )
    query_replay_retention_days: int = Field(30, validation_alias="QUERY_REPLAY_RETENTION_DAYS")
    query_web_allowed_domains_csv: str = Field("", validation_alias="QUERY_WEB_ALLOWED_DOMAINS")
    query_output_link_allowed_domains_csv: str = Field(
        "", validation_alias="QUERY_OUTPUT_LINK_ALLOWED_DOMAINS"
    )
    vector_placement_mode: str = Field(
        "shared_compatible", validation_alias="VECTOR_PLACEMENT_MODE"
    )
    vector_placement_policy_version: str = Field(
        "placement.v1", validation_alias="VECTOR_PLACEMENT_POLICY_VERSION"
    )

    bge_m3_path: str = Field("./models/bge-m3", validation_alias="BGE_M3_PATH")
    bge_m3: str = Field("BAAI/bge-m3", validation_alias="BGE_M3")
    bge_device: str = Field("cpu", validation_alias="BGE_DEVICE")
    bge_fp16: bool = Field(False, validation_alias="BGE_FP16")
    bge_reranker_large: str = Field(
        "./models/bge-reranker-v2-m3", validation_alias="BGE_RERANKER_LARGE"
    )
    bge_reranker_device: str = Field("cpu", validation_alias="BGE_RERANKER_DEVICE")
    bge_reranker_fp16: bool = Field(False, validation_alias="BGE_RERANKER_FP16")
    bge_reranker_batch_size: int = Field(2, ge=1, le=64, validation_alias="BGE_RERANKER_BATCH_SIZE")
    bge_reranker_max_input_tokens: int = Field(
        512, ge=32, le=8192, validation_alias="BGE_RERANKER_MAX_INPUT_TOKENS"
    )
    bge_reranker_query_max_tokens: int = Field(
        128, ge=8, le=4096, validation_alias="BGE_RERANKER_QUERY_MAX_TOKENS"
    )
    bge_reranker_document_max_tokens: int = Field(
        376, ge=8, le=8192, validation_alias="BGE_RERANKER_DOCUMENT_MAX_TOKENS"
    )
    bge_reranker_failure_mode: str = Field(
        "preserve_order", validation_alias="BGE_RERANKER_FAILURE_MODE"
    )
    bge_reranker_min_available_physical_mib: int = Field(
        1536,
        ge=0,
        validation_alias="BGE_RERANKER_MIN_AVAILABLE_PHYSICAL_MIB",
    )
    bge_reranker_min_available_commit_mib: int = Field(
        4096,
        ge=0,
        validation_alias="BGE_RERANKER_MIN_AVAILABLE_COMMIT_MIB",
    )

    milvus_url: str = Field("http://127.0.0.1:19530", validation_alias="MILVUS_URL")
    chunks_collection: str = Field("kb_chunks", validation_alias="CHUNKS_COLLECTION")
    entity_name_collection: str = Field("kb_entities", validation_alias="ENTITY_NAME_COLLECTION")
    item_name_collection: str = Field("kb_item_names", validation_alias="ITEM_NAME_COLLECTION")

    mongo_url_value: SecretStr = Field(
        SecretStr("mongodb://127.0.0.1:27017"), validation_alias="MONGO_URL"
    )
    mongo_db_name: str = Field("enterprise_rag", validation_alias="MONGO_DB_NAME")
    mongo_secret_version: str = Field("env-v1", validation_alias="MONGO_SECRET_VERSION")

    minio_endpoint: str = Field("127.0.0.1:9000", validation_alias="MINIO_ENDPOINT")
    minio_access_key_value: SecretStr = Field(
        SecretStr("minioadmin"), validation_alias="MINIO_ACCESS_KEY"
    )
    minio_secret_key_value: SecretStr = Field(
        SecretStr("minioadmin"), validation_alias="MINIO_SECRET_KEY"
    )
    minio_credential_version: str = Field("env-v1", validation_alias="MINIO_CREDENTIAL_VERSION")
    minio_bucket_name: str = Field("enterprise-rag", validation_alias="MINIO_BUCKET_NAME")
    minio_img_dir: str = Field("/kb-images", validation_alias="MINIO_IMG_DIR")
    minio_secure: bool = Field(False, validation_alias="MINIO_SECURE")

    mineru_base_url: str = Field("", validation_alias="MINERU_BASE_URL")
    mineru_api_token_value: SecretStr = Field(SecretStr(""), validation_alias="MINERU_API_TOKEN")
    mineru_token_version: str = Field("env-v1", validation_alias="MINERU_TOKEN_VERSION")
    mcp_dashscope_base_url: str = Field(
        "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
        validation_alias="MCP_DASHSCOPE_BASE_URL",
    )

    health_probe_timeout_seconds: float = Field(2, validation_alias="HEALTH_PROBE_TIMEOUT_SECONDS")
    health_cache_ttl_seconds: float = Field(5, validation_alias="HEALTH_CACHE_TTL_SECONDS")
    startup_require_ready: bool = Field(False, validation_alias="STARTUP_REQUIRE_READY")
    import_required_dependencies_csv: str = Field(
        "control_database,queue,milvus,minio", validation_alias="IMPORT_REQUIRED_DEPENDENCIES"
    )
    query_required_dependencies_csv: str = Field(
        "control_database,milvus,mongo,llm", validation_alias="QUERY_REQUIRED_DEPENDENCIES"
    )
    worker_required_dependencies_csv: str = Field(
        "control_database,queue,milvus,minio", validation_alias="WORKER_REQUIRED_DEPENDENCIES"
    )

    log_console_enable: bool = Field(True, validation_alias="LOG_CONSOLE_ENABLE")
    log_console_level: str = Field("INFO", validation_alias="LOG_CONSOLE_LEVEL")
    log_file_enable: bool = Field(True, validation_alias="LOG_FILE_ENABLE")
    log_file_level: str = Field("INFO", validation_alias="LOG_FILE_LEVEL")
    log_file_retention: str = Field("7 days", validation_alias="LOG_FILE_RETENTION")

    @model_validator(mode="after")
    def validate_deployment(self) -> "AppSettings":
        errors: list[str] = []
        if not 1 <= self.import_app_port <= 65535:
            errors.append("IMPORT_APP_PORT 必须在 1..65535")
        if not 1 <= self.query_app_port <= 65535:
            errors.append("QUERY_APP_PORT 必须在 1..65535")
        positive = {
            "MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "INGESTION_WORKER_POLL_SECONDS": self.ingestion_worker_poll_seconds,
            "INGESTION_LEASE_SECONDS": self.ingestion_lease_seconds,
            "INGESTION_HEARTBEAT_SECONDS": self.ingestion_heartbeat_seconds,
            "INGESTION_MAX_ATTEMPTS": self.ingestion_max_attempts,
            "HEALTH_PROBE_TIMEOUT_SECONDS": self.health_probe_timeout_seconds,
            "HEALTH_CACHE_TTL_SECONDS": self.health_cache_ttl_seconds,
            "LLM_REQUEST_TIMEOUT": self.llm_request_timeout,
            "LLM_MAX_CONNECTIONS": self.llm_max_connections,
            "MAX_CONCURRENT_CHATS": self.max_concurrent_chats,
            "LLM_MAX_ATTEMPTS": self.llm_max_attempts,
            "QUERY_DEFAULT_DEADLINE_MS": self.query_default_deadline_ms,
            "QUERY_RECOVERY_INTERVAL_SECONDS": self.query_recovery_interval_seconds,
            "QUERY_EVENT_RETENTION_SECONDS": self.query_event_retention_seconds,
            "QUERY_EVENT_BATCH_SIZE": self.query_event_batch_size,
            "QUERY_STREAM_MAX_LAG_EVENTS": self.query_stream_max_lag_events,
            "QUERY_STREAM_MAX_BATCH_BYTES": self.query_stream_max_batch_bytes,
            "QUERY_CACHE_TTL_SECONDS": self.query_cache_ttl_seconds,
            "QUERY_IDEMPOTENCY_TTL_SECONDS": self.query_idempotency_ttl_seconds,
            "QUERY_SINGLEFLIGHT_LEASE_SECONDS": self.query_singleflight_lease_seconds,
            "QUERY_REPLAY_RETENTION_DAYS": self.query_replay_retention_days,
        }
        errors.extend(f"{name} 必须大于 0" for name, value in positive.items() if value <= 0)
        if self.ingestion_heartbeat_seconds >= self.ingestion_lease_seconds:
            errors.append("INGESTION_HEARTBEAT_SECONDS 必须小于 INGESTION_LEASE_SECONDS")
        if self.ingestion_retry_base_seconds > self.ingestion_retry_max_seconds:
            errors.append("INGESTION_RETRY_BASE_SECONDS 不能大于 INGESTION_RETRY_MAX_SECONDS")
        if self.llm_retry_base_delay > self.llm_retry_max_delay:
            errors.append("LLM_RETRY_BASE_DELAY 不能大于 LLM_RETRY_MAX_DELAY")
        if (
            self.bge_reranker_query_max_tokens + self.bge_reranker_document_max_tokens + 8
            > self.bge_reranker_max_input_tokens
        ):
            errors.append(
                "BGE_RERANKER_QUERY_MAX_TOKENS + "
                "BGE_RERANKER_DOCUMENT_MAX_TOKENS + 8 "
                "不能大于 BGE_RERANKER_MAX_INPUT_TOKENS"
            )
        if self.bge_reranker_failure_mode not in {"error", "preserve_order"}:
            errors.append("BGE_RERANKER_FAILURE_MODE 仅允许 error/preserve_order")
        if self.query_stream_max_lag_events < self.query_event_batch_size:
            errors.append("QUERY_STREAM_MAX_LAG_EVENTS 不能小于 QUERY_EVENT_BATCH_SIZE")
        if self.vector_placement_mode not in {
            "shared_compatible",
            "tenant_dedicated",
            "kb_dedicated",
            "cluster_dedicated",
        }:
            errors.append("VECTOR_PLACEMENT_MODE 不是受支持的放置策略")
        if not self.cors_origins:
            errors.append("CORS_ORIGINS 至少配置一个明确来源")
        if (
            _secret(self.auth_token_secret_value)
            and len(_secret(self.auth_token_secret_value)) < 32
        ):
            errors.append("AUTH_TOKEN_SECRET 至少 32 个字符")
        if (
            _secret(self.auth_previous_token_secret_value)
            and len(_secret(self.auth_previous_token_secret_value)) < 32
        ):
            errors.append("AUTH_PREVIOUS_TOKEN_SECRET 非空时至少 32 个字符")
        if len(_secret(self.auth_bootstrap_admin_password_value)) < 12:
            errors.append("AUTH_BOOTSTRAP_ADMIN_PASSWORD 至少 12 个字符")
        if not self.control_database_url.startswith(("sqlite:", "postgresql:", "mysql:")):
            errors.append("CONTROL_DATABASE_URL 仅允许 sqlite/postgresql/mysql SQLAlchemy URL")
        for name, value, schemes in (
            ("MILVUS_URL", self.milvus_url, ("http", "https")),
            ("MONGO_URL", self.mongo_url, ("mongodb", "mongodb+srv")),
        ):
            if urlsplit(value).scheme not in schemes:
                errors.append(f"{name} URL scheme 非法")
        production = self.app_env.lower() in {"prod", "production"}
        if production:
            if "*" in self.cors_origins:
                errors.append("生产环境禁止 CORS_ORIGINS=*")
            if _secret(self.auth_token_secret_value) in {
                DEV_TOKEN_SECRET,
                "replace-with-at-least-32-random-characters",
            }:
                errors.append("生产环境必须注入独立 AUTH_TOKEN_SECRET")
            if _secret(self.auth_bootstrap_admin_password_value) in {
                DEV_ADMIN_PASSWORD,
                "replace-with-a-strong-bootstrap-password",
            }:
                errors.append("生产环境必须注入独立管理员密码")
            if self.minio_access_key == "minioadmin" or self.minio_secret_key == "minioadmin":
                errors.append("生产环境禁止默认 MinIO 凭据")
            for name, value in (
                ("OPENAI_BASE_URL", self.openai_base_url),
                ("MINERU_BASE_URL", self.mineru_base_url),
            ):
                if value and urlsplit(value).scheme != "https":
                    errors.append(f"生产环境 {name} 必须使用 HTTPS")
        if errors:
            raise ValueError("配置校验失败：" + "；".join(errors))
        return self

    @property
    def cors_origins(self) -> tuple[str, ...]:
        return _csv(self.cors_origins_csv)

    @property
    def query_web_allowed_domains(self) -> tuple[str, ...]:
        return _csv(self.query_web_allowed_domains_csv)

    @property
    def query_output_link_allowed_domains(self) -> tuple[str, ...]:
        return _csv(self.query_output_link_allowed_domains_csv)

    @property
    def control_database_url(self) -> str:
        return _secret(self.control_database_url_value)

    @property
    def auth_token_secret(self) -> str:
        return _secret(self.auth_token_secret_value)

    @property
    def auth_bootstrap_admin_password(self) -> str:
        return _secret(self.auth_bootstrap_admin_password_value)

    @property
    def mongo_url(self) -> str:
        return _secret(self.mongo_url_value)

    @property
    def minio_access_key(self) -> str:
        return _secret(self.minio_access_key_value)

    @property
    def minio_secret_key(self) -> str:
        return _secret(self.minio_secret_key_value)

    @property
    def server(self) -> ServerSettings:
        return ServerSettings(
            import_app_name=self.import_app_name,
            query_app_name=self.query_app_name,
            app_env=self.app_env,
            host=self.app_host,
            import_port=self.import_app_port,
            query_port=self.query_app_port,
            cors_origins=self.cors_origins,
        )

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(control_url=self.control_database_url_value)

    @property
    def queue(self) -> QueueSettings:
        return QueueSettings(
            worker_id=self.ingestion_worker_id,
            poll_seconds=self.ingestion_worker_poll_seconds,
            lease_seconds=self.ingestion_lease_seconds,
            heartbeat_seconds=self.ingestion_heartbeat_seconds,
            max_attempts=self.ingestion_max_attempts,
            retry_base_seconds=self.ingestion_retry_base_seconds,
            retry_max_seconds=self.ingestion_retry_max_seconds,
            checkpoint_root=self.ingestion_checkpoint_root,
        )

    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings(
            token_secret_value=self.auth_token_secret_value,
            token_secret_version=self.auth_token_secret_version,
            previous_token_secret_value=self.auth_previous_token_secret_value,
            previous_token_secret_version=self.auth_previous_token_secret_version,
            token_issuer=self.auth_token_issuer,
            access_token_minutes=self.auth_access_token_minutes,
            refresh_token_days=self.auth_refresh_token_days,
            bootstrap_admin_email=self.auth_bootstrap_admin_email,
            bootstrap_admin_password_value=self.auth_bootstrap_admin_password_value,
        )

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            base_url=self.openai_base_url,
            api_key_value=self.openai_api_key_value,
            lv_model=self.vl_model,
            llm_model=self.llm_default_model,
            llm_temperature=self.llm_default_temperature,
            secret_version=self.openai_api_key_version,
        )

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(
            bge_m3_path=self.bge_m3_path,
            bge_m3=self.bge_m3,
            bge_device=self.bge_device,
            bge_fp16=self.bge_fp16,
        )

    @property
    def reranker(self) -> RerankerSettings:
        return RerankerSettings(
            bge_reranker_large=self.bge_reranker_large,
            bge_reranker_device=self.bge_reranker_device,
            bge_reranker_fp16=self.bge_reranker_fp16,
            batch_size=self.bge_reranker_batch_size,
            max_input_tokens=self.bge_reranker_max_input_tokens,
            query_max_tokens=self.bge_reranker_query_max_tokens,
            document_max_tokens=self.bge_reranker_document_max_tokens,
            failure_mode=self.bge_reranker_failure_mode,
            min_available_physical_mib=self.bge_reranker_min_available_physical_mib,
            min_available_commit_mib=self.bge_reranker_min_available_commit_mib,
        )

    @property
    def milvus(self) -> MilvusSettings:
        return MilvusSettings(
            milvus_url=self.milvus_url,
            chunks_collection=self.chunks_collection,
            entity_name_collection=self.entity_name_collection,
            item_name_collection=self.item_name_collection,
        )

    @property
    def mongo(self) -> MongoSettings:
        return MongoSettings(
            url_value=self.mongo_url_value,
            database=self.mongo_db_name,
            secret_version=self.mongo_secret_version,
        )

    @property
    def minio(self) -> MinIOSettings:
        return MinIOSettings(
            endpoint=self.minio_endpoint,
            access_key_value=self.minio_access_key_value,
            secret_key_value=self.minio_secret_key_value,
            bucket_name=self.minio_bucket_name,
            minio_img_dir=self.minio_img_dir,
            minio_secure=self.minio_secure,
            secret_version=self.minio_credential_version,
        )

    @property
    def mineru(self) -> MinerUSettings:
        return MinerUSettings(
            base_url=self.mineru_base_url,
            api_key_value=self.mineru_api_token_value,
            secret_version=self.mineru_token_version,
        )

    @property
    def mcp(self) -> MCPSettings:
        return MCPSettings(
            mcp_base_url=self.mcp_dashscope_base_url, api_key_value=self.openai_api_key_value
        )

    @property
    def health(self) -> HealthSettings:
        return HealthSettings(
            timeout_seconds=self.health_probe_timeout_seconds,
            cache_ttl_seconds=self.health_cache_ttl_seconds,
            startup_require_ready=self.startup_require_ready,
            import_required=_csv(self.import_required_dependencies_csv),
            query_required=_csv(self.query_required_dependencies_csv),
            worker_required=_csv(self.worker_required_dependencies_csv),
        )

    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings(
            console_enabled=self.log_console_enable,
            console_level=self.log_console_level,
            file_enabled=self.log_file_enable,
            file_level=self.log_file_level,
            file_retention=self.log_file_retention,
        )

    def redacted_snapshot(self) -> dict:
        """只返回允许进入健康接口、审计和 trace 的脱敏配置。"""
        return {
            "server": self.server.model_dump(mode="json"),
            "database": {"control_url": _safe_url(self.control_database_url)},
            "queue": self.queue.model_dump(mode="json"),
            "milvus": self.milvus.model_dump(mode="json"),
            "mongo": {
                "url": _safe_url(self.mongo_url),
                "database": self.mongo_db_name,
                "secret_version": self.mongo_secret_version,
            },
            "minio": {
                "endpoint": self.minio_endpoint,
                "bucket_name": self.minio_bucket_name,
                "secure": self.minio_secure,
                "secret_version": self.minio_credential_version,
            },
            "llm": {
                "base_url": _safe_url(self.openai_base_url),
                "model": self.llm_default_model,
                "temperature": self.llm_default_temperature,
                "secret_version": self.openai_api_key_version,
                "request_timeout": self.llm_request_timeout,
                "max_connections": self.llm_max_connections,
                "max_concurrent_chats": self.max_concurrent_chats,
                "max_attempts": self.llm_max_attempts,
                "retry_base_delay": self.llm_retry_base_delay,
                "retry_max_delay": self.llm_retry_max_delay,
            },
            "embedding": self.embedding.model_dump(mode="json"),
            "reranker": self.reranker.model_dump(mode="json"),
            "mineru": {
                "base_url": _safe_url(self.mineru_base_url),
                "secret_version": self.mineru_token_version,
            },
            "security": {
                "issuer": self.auth_token_issuer,
                "access_token_minutes": self.auth_access_token_minutes,
                "refresh_token_days": self.auth_refresh_token_days,
                "token_secret_version": self.auth_token_secret_version,
                "previous_token_secret_version": self.auth_previous_token_secret_version,
            },
            "query_reliability": {
                "default_deadline_ms": self.query_default_deadline_ms,
                "recovery_interval_seconds": self.query_recovery_interval_seconds,
                "checkpoint_encryption_configured": bool(
                    self.query_checkpoint_encryption_key
                    and _secret(self.query_checkpoint_encryption_key)
                ),
                "event_retention_seconds": self.query_event_retention_seconds,
                "event_batch_size": self.query_event_batch_size,
                "stream_max_lag_events": self.query_stream_max_lag_events,
                "stream_max_batch_bytes": self.query_stream_max_batch_bytes,
                "cache_ttl_seconds": self.query_cache_ttl_seconds,
                "idempotency_ttl_seconds": self.query_idempotency_ttl_seconds,
                "singleflight_lease_seconds": self.query_singleflight_lease_seconds,
                "replay_retention_days": self.query_replay_retention_days,
                "web_allowed_domains": self.query_web_allowed_domains,
                "output_link_allowed_domains": self.query_output_link_allowed_domains,
            },
            "vector_topology": {
                "placement_mode": self.vector_placement_mode,
                "policy_version": self.vector_placement_policy_version,
            },
            "health": self.health.model_dump(mode="json"),
        }

    @property
    def config_version(self) -> str:
        encoded = json.dumps(
            self.redacted_snapshot(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return f"cfg_{hashlib.sha256(encoded).hexdigest()[:16]}"

    def legacy_env(self, name: str, default: object = "") -> object:
        mapping = {
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_API_KEY": _secret(self.openai_api_key_value),
            "LLM_DEFAULT_MODEL": self.llm_default_model,
            "LLM_DEFAULT_TEMPERATURE": self.llm_default_temperature,
            "VL_MODEL": self.vl_model,
            "BGE_M3_PATH": self.bge_m3_path,
            "BGE_M3": self.bge_m3,
            "BGE_DEVICE": self.bge_device,
            "BGE_FP16": self.bge_fp16,
            "BGE_RERANKER_LARGE": self.bge_reranker_large,
            "BGE_RERANKER_DEVICE": self.bge_reranker_device,
            "BGE_RERANKER_FP16": self.bge_reranker_fp16,
            "BGE_RERANKER_MIN_AVAILABLE_PHYSICAL_MIB": (
                self.bge_reranker_min_available_physical_mib
            ),
            "BGE_RERANKER_MIN_AVAILABLE_COMMIT_MIB": (self.bge_reranker_min_available_commit_mib),
            "MILVUS_URL": self.milvus_url,
            "CHUNKS_COLLECTION": self.chunks_collection,
            "ENTITY_NAME_COLLECTION": self.entity_name_collection,
            "ITEM_NAME_COLLECTION": self.item_name_collection,
            "MINIO_ENDPOINT": self.minio_endpoint,
            "MINIO_ACCESS_KEY": self.minio_access_key,
            "MINIO_SECRET_KEY": self.minio_secret_key,
            "MINIO_BUCKET_NAME": self.minio_bucket_name,
            "MINIO_IMG_DIR": self.minio_img_dir,
            "MINIO_SECURE": self.minio_secure,
            "MINERU_BASE_URL": self.mineru_base_url,
            "MINERU_API_TOKEN": _secret(self.mineru_api_token_value),
            "MCP_DASHSCOPE_BASE_URL": self.mcp_dashscope_base_url,
        }
        return mapping.get(name, default)


_NEW_ENV_OVERRIDES = {
    "app_env": "APP_ENV",
    "control_database_url_value": "CONTROL_DATABASE_URL",
    "auth_token_secret_value": "AUTH_TOKEN_SECRET",
    "auth_bootstrap_admin_password_value": "AUTH_BOOTSTRAP_ADMIN_PASSWORD",
    "auth_previous_token_secret_value": "AUTH_PREVIOUS_TOKEN_SECRET",
    "query_checkpoint_encryption_key": "QUERY_CHECKPOINT_ENCRYPTION_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_api_key_value": "OPENAI_API_KEY",
    "mongo_url_value": "MONGO_URL",
    "minio_access_key_value": "MINIO_ACCESS_KEY",
    "minio_secret_key_value": "MINIO_SECRET_KEY",
    "mineru_api_token_value": "MINERU_API_TOKEN",
}


class SettingsProvider:
    """可替换配置提供器；测试无需修改全局环境或重新 import 业务模块。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._settings: AppSettings | None = None

    def _build(self) -> AppSettings:
        overrides: dict[str, str] = {}
        for field_name, legacy_name in _NEW_ENV_OVERRIDES.items():
            new_name = f"STT_{legacy_name}"
            if new_name in os.environ:
                overrides[field_name] = os.environ[new_name]
                if legacy_name in os.environ:
                    warnings.warn(
                        f"{new_name} 与兼容变量 {legacy_name} 同时存在；已使用 {new_name}（不会记录变量值）",
                        DeprecationWarning,
                        stacklevel=2,
                    )
        return AppSettings(**overrides)

    def get(self) -> AppSettings:
        with self._lock:
            if self._settings is None:
                self._settings = self._build()
            return self._settings

    @contextmanager
    def override(self, value: AppSettings) -> Iterator[AppSettings]:
        with self._lock:
            previous = self._settings
            self._settings = value
        try:
            yield value
        finally:
            with self._lock:
                self._settings = previous

    def reset(self) -> AppSettings:
        with self._lock:
            self._settings = self._build()
            return self._settings


settings_provider = SettingsProvider()


def get_settings() -> AppSettings:
    return settings_provider.get()


settings = get_settings()


__all__ = [
    "AppSettings",
    "SettingsProvider",
    "get_settings",
    "settings",
    "settings_provider",
    "ServerSettings",
    "DatabaseSettings",
    "QueueSettings",
    "LLMSettings",
    "EmbeddingSettings",
    "RerankerSettings",
    "MilvusSettings",
    "MongoSettings",
    "MinIOSettings",
    "MinerUSettings",
    "MCPSettings",
    "SecuritySettings",
    "HealthSettings",
    "LoggingSettings",
]
