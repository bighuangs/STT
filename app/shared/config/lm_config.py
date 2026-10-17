"""LLM 旧导入路径适配器。"""
from app.shared.config.settings_config import LLMSettings as LLMConfig, get_settings

lm_config = get_settings().llm

__all__ = ["LLMConfig", "lm_config"]
