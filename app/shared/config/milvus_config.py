"""Milvus 旧导入路径适配器。"""
from app.shared.config.settings_config import MilvusSettings as MilvusConfig, get_settings

milvus_config = get_settings().milvus

__all__ = ["MilvusConfig", "milvus_config"]
