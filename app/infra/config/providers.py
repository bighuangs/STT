"""统一配置提供器；保留 `infra_config.*` 旧访问方式，不再产生 import 副作用。"""
from app.shared.config.settings_config import AppSettings, get_settings, settings_provider


class InfraConfig:
    @property
    def app(self) -> AppSettings:
        return get_settings()

    @property
    def llm(self):
        return get_settings().llm

    @property
    def embedding(self):
        return get_settings().embedding

    @property
    def reranker(self):
        return get_settings().reranker

    @property
    def mcp(self):
        return get_settings().mcp

    @property
    def milvus(self):
        return get_settings().milvus

    @property
    def mongo(self):
        return get_settings().mongo

    @property
    def mineru(self):
        return get_settings().mineru

    @property
    def minio(self):
        return get_settings().minio


infra_config = InfraConfig()

__all__ = ["InfraConfig", "infra_config", "settings_provider"]
