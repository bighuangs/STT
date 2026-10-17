"""MinIO 旧导入路径适配器。"""
from app.shared.config.settings_config import MinIOSettings as MinIOConfig, get_settings

minio_config = get_settings().minio

__all__ = ["MinIOConfig", "minio_config"]
