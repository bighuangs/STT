"""
工具模块，负责提供 minio 相关的辅助能力。
"""
# 导入MinIO官方Python SDK核心类（用于MinIO对象存储的客户端操作）
from minio import Minio

# 导入项目内部配置与日志工具
from app.shared.config.minio_config import minio_config  # MinIO相关配置（端点、密钥、桶名等）
from app.shared.runtime.logger import logger            # 项目统一日志工具

# 全局MinIO客户端实例（单例模式，避免重复创建连接，提升性能）
_minio_client: Minio | None = None


# 1. 定义MinIO客户端连接创建函数（私有函数，仅内部调用）
def _create_minio_client() -> Minio:
    """
    创建并返回MinIO客户端连接
    核心作用：读取配置文件中的MinIO参数，初始化客户端连接
    :return: 初始化完成的MinIO客户端对象
    """
    return Minio(
        endpoint=minio_config.endpoint,        # MinIO服务端点（IP:端口）
        access_key=minio_config.access_key,    # MinIO访问密钥
        secret_key=minio_config.secret_key,    # MinIO秘密密钥
        secure=minio_config.minio_secure       # 是否启用HTTPS（True/False）
    )


# 2. 定义桶初始化函数（私有函数，仅内部调用）
def _create_bucket_ready(client: Minio):
    """
    检查MinIO桶是否存在，不存在则创建，并确保使用私有读取策略
    核心作用：确保图片上传所需的桶已就绪，避免上传失败
    :param client: 已初始化的MinIO客户端对象
    """
    bucket_name = minio_config.bucket_name  # 从配置中获取目标桶名
    # 检查桶是否存在
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)  # 不存在则创建桶
        logger.info(f"MinIO桶 {bucket_name} 已创建（默认私有）")
    else:
        logger.info(f"MinIO桶 {bucket_name} 已存在")

    # 旧版本曾为 Markdown 直链设置匿名 s3:GetObject。图片现由 Evidence
    # 绑定的同源鉴权接口提供，撤销旧策略可避免绕过租户/用户/KB 隔离。
    delete_policy = getattr(client, "delete_bucket_policy", None)
    if callable(delete_policy):
        try:
            delete_policy(bucket_name)
            logger.info(f"MinIO桶 {bucket_name} 已确认使用私有读取策略")
        except Exception as exc:
            # 未配置策略时 MinIO 也会返回 NoSuchBucketPolicy；默认私有已经满足
            # 目标，因此不让这一正常状态阻断服务启动。
            logger.debug(f"MinIO桶 {bucket_name} 无需移除匿名策略: {exc}")


def get_minio_client() -> Minio:
    """
    获取全局MinIO客户端（懒加载模式，无锁版本，适配单线程场景）
    核心逻辑：
    1. 首次调用：初始化客户端 + 检查/创建私有桶，将客户端实例赋值给全局变量
    2. 后续调用：直接复用全局客户端实例，避免重复创建连接（提升性能、节省资源）
    :return: 全局唯一的MinIO客户端对象
    """
    # 声明使用全局变量（修改全局变量需显式声明）
    global _minio_client

    # 懒加载：仅在客户端未初始化时执行创建逻辑
    if _minio_client is None:
        logger.info("开始初始化MinIO客户端（首次调用，执行懒加载）")
        client = _create_minio_client()          # 创建客户端连接
        _create_bucket_ready(client)             # 检查并初始化桶
        _minio_client = client                   # 赋值给全局变量，供后续复用
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    # 复用全局客户端实例，直接返回
    return _minio_client


def close_minio_client() -> None:
    """清理 MinIO SDK 的底层 HTTP 连接池。"""
    global _minio_client
    if _minio_client is not None:
        http = getattr(_minio_client, "_http", None)
        clear = getattr(http, "clear", None)
        if callable(clear):
            clear()
        _minio_client = None
