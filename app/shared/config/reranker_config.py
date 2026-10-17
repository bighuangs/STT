"""Reranker 旧导入路径适配器。"""
from app.shared.config.settings_config import RerankerSettings as RerankerConfig, get_settings

reranker_config = get_settings().reranker

__all__ = ["RerankerConfig", "reranker_config"]
