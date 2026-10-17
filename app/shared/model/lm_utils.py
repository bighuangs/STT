"""
工具模块，负责提供 lm 相关的辅助能力。
"""
import asyncio
import weakref

import httpx
from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

from app.shared.config.lm_config import lm_config
from app.shared.config.settings_config import get_settings
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen3-32b"
_DEFAULT_TEMPERATURE = 0.1

# 单次 LLM 请求超时（秒）。
# 关键：代理/网关可能「接受 TCP 连接后不再回包」，此时既不报错也不返回，
# 若不设超时，openai 客户端会按其默认（约 600s）死等，导致并行增强节点静默卡死，
# reducer 永远等不到该节点完成 → 整个导入流程挂起。
# 设一个较短的超时，让 stall 快速抛出，交由 ainvoke_with_retry 退避重试。
_LLM_REQUEST_TIMEOUT = get_settings().llm_request_timeout

# httpx 连接池上限（无 keep-alive，故连接随用随建随关）。
MAX_CONNECTIONS = get_settings().llm_max_connections


def _build_sync_http_client() -> httpx.Client:
    """构造带 connect/read/write/pool 显式超时的同步 HTTP transport。"""

    limits = httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=0)
    timeout = httpx.Timeout(
        _LLM_REQUEST_TIMEOUT,
        connect=min(_LLM_REQUEST_TIMEOUT, 10.0),
        pool=_LLM_REQUEST_TIMEOUT,
    )
    return httpx.Client(limits=limits, timeout=timeout, trust_env=True)


def _build_async_http_client() -> httpx.AsyncClient:
    """
    构造一个**禁用 keep-alive 连接复用**、**全维度超时**的 httpx 异步客户端。

    为什么需要它（导入流水线并行增强节点静默卡死的真正根因）：
      - 5 个 LLM 节点经 LangGraph 线程池并行执行，每个节点 asyncio.run(...) 各起一个
        独立事件循环；httpx 默认会把连接 keep-alive 缓存在连接池里。
      - 代理/网关抖动会留下「半死」连接（被对端悄悄关闭，或其 asyncio 原语绑定到已
        关闭的循环）滞留在池中。
      - openai/langchain 传入的 `timeout` 只覆盖 connect/read/write，**不覆盖
        “从连接池取空闲槽位”的 pool 超时**（httpx 默认 pool 超时为 None=无限）。
        于是某个 chunk 的请求会卡在「等连接池槽位」上永不返回，asyncio.gather 永远
        等不到它 → 节点静默挂起、无任何日志、连超时都不触发。

    解决：
      - `max_keepalive_connections=0`：每个请求都新建连接、用完即关，连接池里不再
        滞留可被跨循环误用/半死的连接。
      - `httpx.Timeout(..., pool=...)`：给 pool 取槽位也设上限，杜绝无限等待。
      - `trust_env=True`：仍尊重 HTTP(S)_PROXY / NO_PROXY 环境变量。
    """
    limits = httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=0)
    timeout = httpx.Timeout(
        _LLM_REQUEST_TIMEOUT,  # 读/写/连接默认超时
        connect=min(_LLM_REQUEST_TIMEOUT, 10.0),  # 建连超时单独收紧
        pool=_LLM_REQUEST_TIMEOUT,  # 关键：等待连接池槽位的超时（默认 None=无限）
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=True)

# ---------------------------------------------------------------------------
# 客户端缓存：**按事件循环隔离**
#
# ChatOpenAI 底层的 openai.AsyncOpenAI 持有一个 httpx.AsyncClient，其连接池/锁
# （asyncio.Event 等）会绑定到「第一个使用它的事件循环」。导入流程的并行 LLM
# 增强节点经 LangGraph Send 扇出到各自的线程，每个线程独立 asyncio.run(...) 创建
# 新的事件循环；若多个循环复用同一个缓存客户端，另一个循环再调用就会抛
# RuntimeError: <asyncio.Event> is bound to a different event loop（被 openai
# 包装成 APIConnectionError）。
#
# 因此缓存按「当前运行的事件循环」分桶：
#   - 有运行中的 loop：用 WeakKeyDictionary 以 loop 为键存子缓存，loop 被回收后
#     其子缓存自动随之释放，避免泄漏。
#   - 无运行中的 loop（同步调用）：落到 _sync_client_cache 全局复用。
# ---------------------------------------------------------------------------
_sync_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}
_loop_client_caches: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, bool], ChatOpenAI]]" = (
    weakref.WeakKeyDictionary()
)


def _current_client_cache() -> dict[tuple[str, bool], ChatOpenAI]:
    """
    返回当前上下文应使用的客户端缓存字典。

    有运行中的事件循环时，返回该 loop 专属的子缓存（不存在则新建）；否则返回同步
    全局缓存。以此保证异步客户端不会被跨事件循环复用。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _sync_client_cache
    cache = _loop_client_caches.get(loop)
    if cache is None:
        cache = {}
        _loop_client_caches[loop] = cache
    return cache


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带缓存的LangChain ChatOpenAI客户端实例
    适配OpenAI/千问/即梦AI等**OpenAI兼容API**，支持自定义模型和JSON标准化输出
    核心特性：按事件循环隔离的缓存机制+配置统一加载+异常精准捕获+国产模型参数适配

    缓存按当前运行的事件循环隔离：并行的异步 LLM 节点各跑在独立 loop 上，互不复用
    同一个 httpx 异步客户端，避免「bound to a different event loop」连接错误。

    :param model: 模型名称，优先级：传入参数 > 配置文件 lm_config.llm_model > 内置默认模型
    :param json_mode: 是否开启JSON输出模式，开启后返回标准json_object格式（适配结构化数据解析）
    :return: 初始化完成的ChatOpenAI实例（优先从当前 loop 缓存获取，未命中则新建并缓存）
    :raise ValueError: 缺失API密钥/基础地址等核心配置
    :raise Exception: 模型初始化失败（LangChain封装层异常）
    """
    # 1. 确定目标模型（优先级递减，保证模型名非空）
    target_model = model or lm_config.llm_model or _DEFAULT_LLM_MODEL
    # 缓存键：模型名+JSON模式，唯一标识不同配置的客户端
    cache_key = (target_model, json_mode)
    # 选取与当前事件循环对应的缓存桶（同步上下文落全局缓存）
    client_cache = _current_client_cache()

    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    if cache_key in client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_BASE_URL（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    extra_body = {"enable_thinking": False}  # 千问专属：关闭思考链输出，减少冗余内容
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 5. 客户端初始化：捕获LangChain封装层异常，抛出更友好的提示
    try:
        llm_client = ChatOpenAI(
            model=target_model,  # 目标模型名
            temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,  # 低温度保证输出确定性（0~1）
            api_key=lm_config.api_key,  # API密钥
            base_url=lm_config.base_url,  # API基础地址（适配国产模型代理地址）
            extra_body=extra_body,  # 国产模型私有参数透传
            model_kwargs=model_kwargs,  # OpenAI通用参数
            timeout=_LLM_REQUEST_TIMEOUT,  # 单次请求超时，避免代理 stall 导致无限等待
            # 关闭 SDK 内置重试，避免与 Query/Import 调用侧的有界重试发生叠加。
            max_retries=0,
            http_client=_build_sync_http_client(),
            # 关键：禁用 keep-alive 连接复用 + pool 超时，杜绝并行节点等连接池槽位静默卡死
            http_async_client=_build_async_http_client(),
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入当前 loop 的缓存桶，供后续同循环调用复用
    client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client

