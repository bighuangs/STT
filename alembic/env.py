"""Alembic 环境：数据库 URL 由 CONTROL_DATABASE_URL 覆盖。"""

from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.infra.persistence.sql.models import Base
# 独立模型模块仍注册到同一个 Base；显式导入保证 autogenerate 能看到 P3 表。
from app.infra.persistence.sql import p3_models as _p3_models
from app.infra.persistence.sql import query_runtime_models as _query_runtime_models
from app.shared.config.settings_config import settings


config = context.config
if config.config_file_name is not None:
    # API/Worker 会在进程内调用 Alembic；不能让 fileConfig 禁用 Uvicorn/Loguru
    # 已注册 logger，否则迁移后服务虽然已监听，终端却看不到 startup/access 日志。
    fileConfig(config.config_file_name, disable_existing_loggers=False)
config.set_main_option(
    "sqlalchemy.url",
    os.getenv("CONTROL_DATABASE_URL")
    or config.get_main_option("sqlalchemy.url")
    or settings.control_database_url,
)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        if connection.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            # SQLAlchemy 2 的 exec_driver_sql 会 autobegin；先提交，避免 Alembic
            # 在连接关闭时回滚 version_num，造成 schema 已建但无法 downgrade。
            connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
