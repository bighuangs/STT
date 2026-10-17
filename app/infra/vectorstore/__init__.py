"""
应用主包 / 基础设施层 / 向量库子模块的初始化文件，用于声明包边界与导出约定。
"""
from app.infra.vectorstore.milvus_gateway import milvus_gateway

__all__ = ["milvus_gateway"]
