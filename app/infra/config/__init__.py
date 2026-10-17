"""
基础设施配置统一出口。

对外层代码仅暴露稳定配置入口，避免直接依赖 `settings.py` 或 `providers.py` 内部实现。
"""
from app.infra.config.providers import infra_config
from app.shared.config.settings_config import settings

app_settings = settings
llm_config = infra_config.llm
embedding_config = infra_config.embedding
reranker_config = infra_config.reranker
mcp_config = infra_config.mcp
milvus_config = infra_config.milvus
mineru_config = infra_config.mineru
minio_config = infra_config.minio

__all__ = [
    "infra_config",
    "settings",
    "app_settings",
    "llm_config",
    "embedding_config",
    "reranker_config",
    "mcp_config",
    "milvus_config",
    "mineru_config",
    "minio_config",
]
