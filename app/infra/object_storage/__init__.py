"""
应用主包 / 基础设施层 / 对象存储子模块的初始化文件，用于声明包边界与导出约定。
"""
from app.infra.object_storage.minio_gateway import minio_gateway

__all__ = ["minio_gateway"]
