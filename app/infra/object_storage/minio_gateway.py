"""
MinIO 门面模块，统一封装对象存储客户端与桶配置访问。
"""
from minio import Minio

from app.shared.clients.minio_utils import get_minio_client
from app.infra.config import infra_config


class MinIOGateway:
    # 1. 暴露桶名 `bucket_name`
    # 2. 暴露图片目录前缀 `image_dir`
    # 3. 提供客户端对象 `client()`
    # 4. 提供图片公开地址拼接方法 `build_image_url()`

    @property
    def bucket_name(self) -> str:
        """获取 MinIO 存储桶名称（从全局配置读取）"""
        return infra_config.minio.bucket_name

    @property
    def image_dir(self) -> str:
        """获取 MinIO 中存放图片的目录路径（从全局配置读取）"""
        return infra_config.minio.minio_img_dir

    def client(self) -> Minio:
        """获取 MinIO 客户端实例，用于上传、下载、查询文件等操作"""
        return get_minio_client()

    def build_image_url(self, stem: str, image_name: str) -> str:
        """
        构建图片对象的公开访问地址。拼接生成 MinIO 图片的可访问URL（HTTP/HTTPS）

        Args:
            stem: 当前文档或任务的目录名。
            image_name: 图片文件名。

        Returns:
            str: 对应图片在 MinIO 中的访问 URL。
        """
        # 根据配置决定使用 http 还是 https
        protocol = "https" if infra_config.minio.minio_secure else "http"
        return (
            f"{protocol}://{infra_config.minio.endpoint}/"
            f"{self.bucket_name}{self.image_dir}/{stem}/{image_name}"
        )

# 单例模式实例化 MinIOGateway
minio_gateway = MinIOGateway()
