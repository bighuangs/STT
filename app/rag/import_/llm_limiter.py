"""
Service: 导入流水线 LLM/资源并发限流器 (llm_limiter)

参考 ragflow：
    - common/asyncio_utils.py:LoopLocalSemaphore（绑定事件循环的异步信号量）
    - rag/svr/task_executor_limiter.py（模块级 limiter 实例：chat_limiter / embed_limiter / minio_limiter 等）
    - rag/graphrag/utils.py:chat_limiter（MAX_CONCURRENT_CHATS）
    - rag/svr/task_executor.py 中各 doc_* 任务用 `async with chat_limiter:` 包裹 LLM 调用

核心功能：
    为 import_ 流水线中"扇出并发调用 LLM"的 service 提供统一的并发上限控制，
    防止一次性把几十路 LLM 请求打到后端导致限流/超时/被封。
    用法与 ragflow 完全一致：在 LLM 网络调用处 `async with chat_limiter:`。

设计要点 / 与 ragflow 的差异（重要）：
    ragflow 整个任务跑在【单一事件循环】里，LoopLocalSemaphore 等价于全局信号量。
    而 STT 的 5 个 LLM 增强节点是 LangGraph 线程池【并行】执行的，每个节点内部各自
    调用 asyncio.run() → 拥有【独立事件循环】。LoopLocalSemaphore 按事件循环维度
    各持一份信号量，因此本限流器的语义是：
        - 【节点内有效】：单个 service 扇出 N 个 chunk 时，本节点内并发被限到上限值
          （这是主要风险点——一个节点一次性 20+ 路 LLM 调用——已被有效控制）；
        - 【跨并行节点为各自独立上限】：4 个 LLM 节点各自独立计数，全局峰值最坏为
          节点数 × 上限值。若需要严格的"跨节点全局上限"，需改用线程安全原语
          （如 threading.BoundedSemaphore + run_in_executor）或降低 LangGraph 并行度，
          属于更大的改动，本文件暂按 ragflow 原语实现。
"""
import asyncio
import random
import weakref

from app.shared.runtime.logger import logger


class LoopLocalSemaphore:
    """
    绑定到事件循环的异步信号量，为每个运行中的事件循环维护独立的 Semaphore 实例。

    核心功能：解决 asyncio 中"信号量不能跨事件循环复用"的经典问题——
              通过弱引用字典按事件循环缓存 Semaphore，运行时自动识别当前循环并返回对应实例，
              支持模块级并发限制器在多线程/多协程环境下安全复用。

    业务逻辑：
        1. 实例化时记录并发上限 `_value`，初始化空的弱引用字典按循环缓存信号量
        2. `_get()` 获取当前运行循环，先清理已关闭循环的缓存条目防止内存泄漏
        3. 当前循环已有缓存则复用，否则新建 Semaphore 缓存后返回
        4. acquire/release 委托给当前循环对应的 Semaphore
        5. 支持 `async with` 协议，自动获取/释放

    用法示例：
        limiter = LoopLocalSemaphore(10)

        async def worker():
            async with limiter:
                await call_llm()
    """

    def __init__(self, value: int):
        """
        Args:
            value: 允许的最大并发数量（信号量计数器初始值）
        """
        self._value = int(value)
        # 弱引用字典：key=事件循环对象，value=该循环对应的 Semaphore；
        # 循环不再被引用时由 GC 自动回收，无需手动管理生命周期
        self._semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary()
        )

    def _get(self) -> asyncio.Semaphore:
        """
        获取当前事件循环对应的 Semaphore 实例（无则新建并缓存，已关闭循环则先剔除）。

        Returns:
            当前事件循环对应的 asyncio.Semaphore 实例
        """
        loop = asyncio.get_running_loop()

        # 清理已关闭的事件循环，避免内存泄漏和错误引用
        for cached_loop in list(self._semaphores):
            if cached_loop.is_closed():
                self._semaphores.pop(cached_loop, None)

        sem = self._semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self._value)
            self._semaphores[loop] = sem
        return sem

    async def acquire(self) -> bool:
        """获取信号量（阻塞等待直到成功），返回 True。"""
        return await self._get().acquire()

    def release(self) -> None:
        """释放信号量，允许其他协程继续执行。"""
        self._get().release()

    async def __aenter__(self):
        """进入 async with 上下文时自动获取信号量。"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """退出 async with 上下文时自动释放信号量（不吞没异常）。"""
        self.release()
        return False


# ---------------------------------------------------------------------------
# 模块级 limiter 实例（对齐 ragflow task_executor_limiter.py）
# 通过环境变量可调；缺省值与 ragflow 一致（MAX_CONCURRENT_CHATS=10）。
# ---------------------------------------------------------------------------

# LLM 对话/抽取类调用的并发上限，被以下 service 共享：
#   keyword_extraction / question_proposal / metadata_generate /
#   content_tagging / toc（阶段 1 标题抽取 + 阶段 2 层级分配）
from app.shared.config.settings_config import get_settings

MAX_CONCURRENT_CHATS = get_settings().max_concurrent_chats

chat_limiter = LoopLocalSemaphore(MAX_CONCURRENT_CHATS)


# LLM 调用失败重试参数（可经环境变量调整）
# 主要用于吸收代理/网络抖动导致的 APIConnectionError 等瞬时错误。
LLM_MAX_ATTEMPTS = get_settings().llm_max_attempts
LLM_RETRY_BASE_DELAY = get_settings().llm_retry_base_delay
LLM_RETRY_MAX_DELAY = get_settings().llm_retry_max_delay


async def ainvoke_with_retry(
    llm,
    messages,
    *,
    max_attempts: int = LLM_MAX_ATTEMPTS,
    base_delay: float = LLM_RETRY_BASE_DELAY,
    max_delay: float = LLM_RETRY_MAX_DELAY,
):
    """
    在 chat_limiter 并发约束下调用 llm.ainvoke，失败按指数退避（带抖动）重试。

    用于导入流水线的并行 LLM 增强节点：代理/网络抖动会偶发抛出 APIConnectionError
    等瞬时错误，单次调用失败率较高。本函数对每次调用做最多 max_attempts 次尝试，
    第 n 次失败后等待 min(max_delay, base_delay * 2**(n-1)) + 随机抖动 再重试；
    全部失败则抛出最后一次异常，交由调用方按原有逻辑兜底（返回空串 / "{}"）。

    Args:
        llm: ChatOpenAI 实例（由 llm_provider.chat 提供）
        messages: 传给 ainvoke 的消息列表
        max_attempts: 最大尝试次数（含首次）
        base_delay: 退避基准秒数
        max_delay: 单次退避上限秒数

    Returns:
        llm.ainvoke 的返回值（成功时）

    Raises:
        最后一次尝试的异常（全部失败时）
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            # chat_limiter 限制本事件循环内并发 LLM 调用数，避免一次性扇出过多请求
            async with chat_limiter:
                return await llm.ainvoke(messages)
        except Exception as e:  # 网络/连接/限流等瞬时错误统一重试
            last_exc = e
            if attempt >= max_attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, base_delay)  # 抖动：错开并发请求的重试时刻
            logger.warning(
                f"[ainvoke_with_retry] 第 {attempt}/{max_attempts} 次调用失败："
                f"{type(e).__name__}: {e}，{delay:.2f}s 后重试"
            )
            await asyncio.sleep(delay)
    # 全部尝试失败：抛出最后一次异常，调用方按既有 try/except 兜底
    raise last_exc
