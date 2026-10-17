"""
项目日志工具类
基于loguru实现，支持.env配置控制台/文件双输出，自动生成logs/app_年月日.log
特性：
1. 配置驱动：通过.env开关输出、修改日志级别
2. 自动路径：文件日志默认输出到 项目根/logs/app_YYYYMMDD.log
3. 自动清理：按配置保留日志，自动删除过期文件
4. 中文友好：utf-8编码，彻底解决中文乱码
5. 异步安全：开启异步入队，支持多线程/异步场景，避免日志错乱
6. 开箱即用：项目所有模块直接导入logger即可使用
7. 位置终极精准：穿透loguru内部+工具类自身，完美显示业务模块实际调用位置
"""

import hashlib
import inspect
import json
import re
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Mapping

from loguru import logger as _loguru_logger

from app.shared.config.settings_config import get_settings


# -------------------------- 第一步：从 TODO-05 统一配置入口读取日志参数 --------------------------
_logging_settings = get_settings().logging
LOG_CONSOLE_ENABLE = _logging_settings.console_enabled
LOG_CONSOLE_LEVEL = _logging_settings.console_level.upper()
LOG_FILE_ENABLE = _logging_settings.file_enabled
LOG_FILE_LEVEL = _logging_settings.file_level.upper()
LOG_FILE_RETENTION = _logging_settings.file_retention

# -------------------------- 第三步：定义日志路径（自动推导项目根） --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- 第四步：定义日志格式（彩色、结构化、易读） --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)


# -------------------------- 第五步：初始化日志配置（核心方法） --------------------------
def init_logger():
    """
    初始化全局日志配置
    1. 移除loguru默认控制台输出（避免重复打印）
    2. 根据.env配置开启/关闭控制台输出
    3. 根据.env配置开启/关闭文件输出（自动创建logs文件夹）
    4. 配置日志格式、级别、分割、保留策略
    :return: 配置完成的loguru logger实例
    """
    # 1. 移除loguru默认的控制台输出
    _loguru_logger.remove()

    # 2. 配置控制台输出（若.env开启）
    if LOG_CONSOLE_ENABLE:
        _loguru_logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True,
            # 控制台与文件 sink 都必须关闭 Loguru 的局部变量诊断。
            # 否则异常堆栈诊断会把函数参数中的 API key、预签名 URL 等
            # 凭据展开到 traceback，即使业务日志正文从未主动打印这些值。
            backtrace=False,
            diagnose=False,
        )

    # 3. 配置文件输出（若.env开启）
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            # diagnose/backtrace 会展开局部变量，可能把 token/密码写入日志。
            backtrace=False,
            diagnose=False,
        )

    return _loguru_logger


# -------------------------- 第六步：初始化并终极修正全局logger --------------------------
base_logger = init_logger()

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:x-api-key|api[_-]?key|secret|password|token)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(mongodb(?:\+srv)?://)[^@/\s]+@"),
)
_SAFE_LOG_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/-]{1,160}$")
_BOUND_SOURCE_NAME = "_stt_safe_source_name"
_BOUND_SOURCE_FUNCTION = "_stt_safe_source_function"
_BOUND_SOURCE_LINE = "_stt_safe_source_line"
_NODE_FAILURE_METADATA_FIELDS = (
    "origin_node",
    "origin_function",
    "origin_exception_type",
    "origin_fingerprint",
)


def redact_log_text(value: object) -> str:
    """删除常见凭据形态及当前进程已知秘密；永不返回秘密原值。"""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    current = get_settings()
    known = (
        current.auth_token_secret,
        *current.security.token_secrets,
        current.auth_bootstrap_admin_password,
        current.minio_access_key,
        current.minio_secret_key,
        current.llm.api_key,
        current.mineru.api_key,
    )
    for secret in known:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    return text


def fix_log_position(record):
    """遍历调用栈，跳过loguru内部帧+工具类自身帧，提取业务代码实际调用位置"""
    record["message"] = redact_log_text(record["message"])
    extra = record.get("extra", {})
    source_name = extra.get(_BOUND_SOURCE_NAME)
    source_function = extra.get(_BOUND_SOURCE_FUNCTION)
    source_line = extra.get(_BOUND_SOURCE_LINE)
    if (
        isinstance(source_name, str)
        and isinstance(source_function, str)
        and isinstance(source_line, int)
        and _SAFE_LOG_IDENTIFIER.fullmatch(source_name)
        and _SAFE_LOG_IDENTIFIER.fullmatch(source_function)
        and 0 <= source_line <= 10_000_000
    ):
        # 装饰器显式绑定被装饰函数的代码位置，避免 helper/contextlib 栈帧污染来源。
        record.update(name=source_name, function=source_function, line=source_line)
        return
    for frame in inspect.stack():
        # 终极过滤：排除loguru内部 + 排除工具类logger.py自身，直接定位业务模块
        if (
            "_logger.py" in frame.filename or frame.function == "_log"
        ) or "logger.py" in frame.filename:
            continue
        # 更新日志字段为业务代码实际位置
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno,
        )
        break


# 应用终极修复，导出全局可用的logger
logger = base_logger.patch(fix_log_position)


def _safe_log_identifier(value: object, *, fallback: str, max_length: int = 160) -> str:
    """仅接受低风险标识符，拒绝把自由文本带入运行日志。"""

    if not isinstance(value, (str, int)):
        return fallback
    normalized = str(value)
    if len(normalized) > max_length or not _SAFE_LOG_IDENTIFIER.fullmatch(normalized):
        return fallback
    return normalized


def _function_name(func) -> str:
    return _safe_log_identifier(getattr(func, "__name__", None), fallback="unknown_function")


def _source_logger(func, *, function_name: str):
    """把装饰器日志绑定到可信代码对象，不从运行输入推导源码位置。"""

    code = getattr(func, "__code__", None)
    raw_filename = getattr(code, "co_filename", "")
    source_name = _safe_log_identifier(
        Path(raw_filename).name if isinstance(raw_filename, str) else None,
        fallback="unknown_source",
    )
    raw_line = getattr(code, "co_firstlineno", 0)
    source_line = raw_line if isinstance(raw_line, int) and raw_line >= 0 else 0
    return logger.bind(
        **{
            _BOUND_SOURCE_NAME: source_name,
            _BOUND_SOURCE_FUNCTION: function_name,
            _BOUND_SOURCE_LINE: source_line,
        }
    )


def _trace_id(state) -> str:
    if isinstance(state, Mapping):
        return _safe_log_identifier(
            state.get("trace_id")
            or state.get("query_run_id")
            or state.get("task_id")
            or state.get("session_id")
            or "-",
            fallback="-",
        )
    return "-"


def _correlation_id(args: tuple, kwargs: dict) -> str:
    for key in ("trace_id", "query_run_id", "task_id", "session_id", "correlation_id"):
        if key in kwargs:
            return _safe_log_identifier(kwargs[key], fallback="-")
    state = kwargs.get("state")
    if isinstance(state, Mapping):
        return _trace_id(state)
    for candidate in args[:2]:
        if isinstance(candidate, Mapping):
            selected = _trace_id(candidate)
            if selected != "-":
                return selected
    return "-"


def _failure_fingerprint(
    *,
    stage: str,
    operation_name: str,
    function_name: str,
    exception_type: str,
) -> str:
    """仅用稳定分类字段生成指纹，不读取异常正文或 cause chain。"""

    payload = json.dumps(
        {
            "exception_type": exception_type,
            "function": function_name,
            "operation": operation_name,
            "stage": stage,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attach_node_failure_metadata(
    exc: Exception,
    *,
    node_name: str,
    function_name: str,
) -> tuple[str, str]:
    """在原异常上附加经过约束的节点来源，不读取或复制异常正文。"""

    exception_type = _safe_log_identifier(type(exc).__name__, fallback="Exception")
    fingerprint = _failure_fingerprint(
        stage="node",
        operation_name=node_name,
        function_name=function_name,
        exception_type=exception_type,
    )
    metadata = dict(
        zip(
            _NODE_FAILURE_METADATA_FIELDS,
            (node_name, function_name, exception_type, fingerprint),
            strict=True,
        )
    )
    for field_name, value in metadata.items():
        try:
            # 绕过异常子类可能自定义的 __setattr__，失败时仍保持原异常传播。
            object.__setattr__(exc, field_name, value)
        except Exception:
            continue
    return exception_type, fingerprint


def _log_safe_failure(
    *,
    stage: str,
    operation_name: str,
    function_name: str,
    correlation_id: str,
    exc: Exception,
    event_logger,
) -> None:
    """记录可关联的低基数错误元数据，不附加 exception 或 traceback。"""

    if stage == "node":
        exception_type, fingerprint = _attach_node_failure_metadata(
            exc,
            node_name=operation_name,
            function_name=function_name,
        )
        event_logger.error(
            "graph_execution_failed stage={} node={} function={} exception_type={} "
            "trace_id={} fingerprint={}",
            stage,
            operation_name,
            function_name,
            exception_type,
            correlation_id,
            fingerprint,
        )
        return
    exception_type = _safe_log_identifier(type(exc).__name__, fallback="Exception")
    fingerprint = _failure_fingerprint(
        stage=stage,
        operation_name=operation_name,
        function_name=function_name,
        exception_type=exception_type,
    )
    event_logger.error(
        "graph_execution_failed stage={} step={} function={} exception_type={} "
        "correlation_id={} fingerprint={}",
        stage,
        operation_name,
        function_name,
        exception_type,
        correlation_id,
        fingerprint,
    )


@contextmanager
def _node_log_scope(*, node_name: str, function_name: str, state, event_logger):
    from app.application.observability_service import metric_registry, trace_recorder

    trace_id = _trace_id(state)
    start_ts = time.time()
    status = "succeeded"
    event_logger.info(f"[{node_name}] 节点开始，追踪ID={trace_id}")
    try:
        # Trace 属性刻意不含问题、正文或资源 ID。
        with trace_recorder.span(
            node_name,
            trace_id=trace_id,
            attributes={"component": "workflow", "node": node_name},
        ):
            yield
    except Exception as exc:
        status = "failed"
        _log_safe_failure(
            stage="node",
            operation_name=node_name,
            function_name=function_name,
            correlation_id=trace_id,
            exc=exc,
            event_logger=event_logger,
        )
        raise
    else:
        cost_ms = int((time.time() - start_ts) * 1000)
        event_logger.info(f"[{node_name}] 节点完成，追踪ID={trace_id}，耗时={cost_ms}ms")
    finally:
        elapsed_ms = (time.time() - start_ts) * 1000
        metric_registry.increment(
            "workflow_node_total",
            labels={"node": node_name, "status": status},
        )
        metric_registry.observe(
            "workflow_node_duration_ms",
            elapsed_ms,
            labels={"node": node_name, "status": status},
        )


@contextmanager
def _step_log_scope(*, step_name: str, function_name: str, correlation_id: str, event_logger):
    start_ts = time.time()
    event_logger.info(f"[{step_name}] 步骤开始")
    try:
        yield
    except Exception as exc:
        _log_safe_failure(
            stage="step",
            operation_name=step_name,
            function_name=function_name,
            correlation_id=correlation_id,
            exc=exc,
            event_logger=event_logger,
        )
        raise
    else:
        cost_ms = int((time.time() - start_ts) * 1000)
        event_logger.info(f"[{step_name}] 步骤完成，耗时={cost_ms}ms")


# 定义node的装饰器（用于状态机/工作流图中的节点）
def node_log(node_name: str):
    """
    装饰器：为节点自动注入开始/完成/异常日志，并携带 trace_id 链路追踪。
    适用场景：LangGraph / 状态机 / 多节点流程编排。
    被装饰函数签名要求：第一个参数为 state（字典，需包含 session_id 或 task_id）。
    """

    def deco(func):
        # Node 名同时用作 Metric label，必须满足观测层 64 字符上限。
        safe_node_name = _safe_log_identifier(node_name, fallback="unknown_node", max_length=64)
        function_name = _function_name(func)
        event_logger = _source_logger(func, function_name=function_name)
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(state, *args, **kwargs):
                with _node_log_scope(
                    node_name=safe_node_name,
                    function_name=function_name,
                    state=state,
                    event_logger=event_logger,
                ):
                    return await func(state, *args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(state, *args, **kwargs):
            with _node_log_scope(
                node_name=safe_node_name,
                function_name=function_name,
                state=state,
                event_logger=event_logger,
            ):
                return func(state, *args, **kwargs)

        return wrapper

    return deco


# 定义业务的装饰器（用于一般业务函数/方法的通用日志切面）
def step_log(step_name: str):
    """
    步骤日志装饰器：
    - 自动打印步骤开始、完成及脱敏异常分类
    - 不吞异常，保持原有业务语义
    适用场景：普通业务方法、工具函数、脚本流水线等无状态追踪需求的场景。
    """

    def deco(func):
        safe_step_name = _safe_log_identifier(step_name, fallback="unknown_step", max_length=64)
        function_name = _function_name(func)
        event_logger = _source_logger(func, function_name=function_name)
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with _step_log_scope(
                    step_name=safe_step_name,
                    function_name=function_name,
                    correlation_id=_correlation_id(args, kwargs),
                    event_logger=event_logger,
                ):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            with _step_log_scope(
                step_name=safe_step_name,
                function_name=function_name,
                correlation_id=_correlation_id(args, kwargs),
                event_logger=event_logger,
            ):
                return func(*args, **kwargs)

        return wrapper

    return deco


# -------------------------- 测试代码（验证修复效果） --------------------------
if __name__ == "__main__":
    # 【debug】开发调试用，记录细节、变量，上线一般关闭
    # 目的是逐步输出，在整个日志中输出内容最多
    logger.debug("【调试】进入主程序入口，开始初始化日志")

    # 【info】正常流程日志，记录程序运行状态
    # 记录核心点，输出入口和出口的核心参数
    logger.info("【信息】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")

    # 打印日志文件路径，方便人工确认文件是否正确生成到 logs/ 目录下
    print(f"日志文件输出路径：{LOG_FILE_PATH}")

    # 【warning】警告，不影响运行，但需要关注
    # 对于某些该有值却没有输入导致采用默认值的函数使用，其不会影响业务正常处理
    logger.warning("【警告】未读取到自定义配置，使用默认配置")

    # 【error】当前功能出错，程序不会崩溃，但业务失败
    # 对于已知的异常，某些核心参数不存在，导致业务无法继续进行
    # 只包含纯错误信息
    logger.error("【错误】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")

    # 【异常】只记录稳定分类字段，避免调试入口形成异常正文旁路
    try:
        result = 10 / 0
        logger.info(f"【信息】业务计算结果：{result}")
    except Exception as exc:
        _log_safe_failure(
            stage="step",
            operation_name="logger_smoke_test",
            function_name="main",
            correlation_id="-",
            exc=exc,
            event_logger=logger,
        )
