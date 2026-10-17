"""
应用主包 / 基础设施层 / 模型能力子模块的初始化文件，用于声明包边界与导出约定。
"""
from app.infra.llm.providers import llm_provider

__all__ = ["llm_provider"]
