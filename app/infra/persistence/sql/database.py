"""SQLAlchemy 控制库连接、会话与初始化。"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.shared.config.settings_config import settings


_lock = RLock()
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_database_url: str | None = None

# Python sqlite3 默认只等待 5 秒。查询运行、导入 Worker 心跳和前端轮询会由
# 不同进程同时写控制库，短暂写锁不应立即演变成 ``database is locked``。
# connect timeout 与 PRAGMA 使用同一预算，避免连接池中新连接行为不一致。
SQLITE_BUSY_TIMEOUT_MS = 30_000
SQLITE_CONNECT_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1_000


def _prepare_sqlite_path(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return
    Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _is_sqlite_url(database_url: str) -> bool:
    return make_url(database_url).get_backend_name() == "sqlite"


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    """为连接池中的每条 SQLite 连接统一并发与完整性策略。"""

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        # WAL 允许读请求与单个写入者并行；对 :memory: SQLite 会安全地保持
        # ``memory`` journal mode，不影响现有单元测试语义。
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def configure_database(database_url: str | None = None) -> Engine:
    """配置控制库；测试可传入独立 SQLite URL 重新绑定。"""
    global _engine, _session_factory, _database_url
    with _lock:
        if database_url is None and _engine is not None:
            return _engine
        # dispose 后保留显式注入目标，确保测试/同进程重启不会悄然切回默认库。
        target_url = database_url or _database_url or settings.control_database_url
        if _engine is not None and _database_url == target_url:
            return _engine
        if _engine is not None:
            _engine.dispose()

        _prepare_sqlite_path(target_url)
        is_sqlite = _is_sqlite_url(target_url)
        connect_args = (
            {
                "check_same_thread": False,
                "timeout": SQLITE_CONNECT_TIMEOUT_SECONDS,
            }
            if is_sqlite
            else {}
        )
        _engine = create_engine(
            target_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        if is_sqlite:
            event.listen(
                _engine,
                "connect",
                _configure_sqlite_connection,
            )
        _session_factory = sessionmaker(
            bind=_engine,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )
        _database_url = target_url
        return _engine


def get_engine() -> Engine:
    return configure_database()


def dispose_database() -> None:
    """释放控制库连接池；应用关闭后下次访问会按当前配置重新创建。"""
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None


def initialize_control_database() -> None:
    """将当前控制库升级到 Alembic head；重复调用是幂等的。"""
    from alembic import command
    from alembic.config import Config

    engine = get_engine()
    database_url = engine.url.render_as_string(hide_password=False).replace("%", "%%")
    project_root = Path(__file__).resolve().parents[4]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


@contextmanager
def session_scope() -> Iterator[Session]:
    global _session_factory
    if _session_factory is None:
        configure_database()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
