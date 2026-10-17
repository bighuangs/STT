"""Embedding 旧导入路径适配器。"""
from app.shared.config.settings_config import EmbeddingSettings as EmbeddingConfig, get_settings

embedding_config = get_settings().embedding

__all__ = ["EmbeddingConfig", "embedding_config"]
