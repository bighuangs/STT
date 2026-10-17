"""
工具模块，负责提供 reranker 相关的辅助能力。
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

from FlagEmbedding import FlagReranker

from app.shared.config.reranker_config import reranker_config
from app.shared.runtime.logger import logger

_reranker_model: FlagReranker | None = None

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryCapacitySnapshot:
    """加载守卫需要的非敏感容量快照，单位为 byte。"""

    available_physical: int | None
    available_commit: int | None
    source: str


class RerankerCapacityError(RuntimeError):
    """模型加载前可用容量不足；允许上层按既有策略安全降级。"""


def _windows_memory_capacity() -> MemoryCapacitySnapshot:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return MemoryCapacitySnapshot(
        available_physical=int(status.ullAvailPhys),
        # Windows 在此字段返回进程可继续提交的容量（commit availability）。
        available_commit=int(status.ullAvailPageFile),
        source="windows_global_memory_status_ex",
    )


def _linux_memory_capacity() -> MemoryCapacitySnapshot:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.strip().split()
        if parts:
            values[name] = int(parts[0]) * 1024
    commit_available = None
    if "CommitLimit" in values and "Committed_AS" in values:
        commit_available = max(0, values["CommitLimit"] - values["Committed_AS"])
    return MemoryCapacitySnapshot(
        available_physical=values.get("MemAvailable"),
        available_commit=commit_available,
        source="linux_proc_meminfo",
    )


def _portable_memory_capacity() -> MemoryCapacitySnapshot:
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return MemoryCapacitySnapshot(
        available_physical=int(page_size * available_pages),
        available_commit=None,
        source="posix_sysconf",
    )


def _memory_capacity_snapshot() -> MemoryCapacitySnapshot:
    """优先读取 OS 原生容量；探测失败时保留可用的跨平台信息。"""

    try:
        if os.name == "nt":
            return _windows_memory_capacity()
        if Path("/proc/meminfo").is_file():
            return _linux_memory_capacity()
        return _portable_memory_capacity()
    except (OSError, ValueError, AttributeError):
        logger.warning("无法读取重排模型加载前容量，将继续由运行时降级策略兜底")
        return MemoryCapacitySnapshot(None, None, "unavailable")


def _ensure_reranker_load_capacity() -> None:
    snapshot = _memory_capacity_snapshot()
    required_physical = reranker_config.min_available_physical_mib * _MIB
    required_commit = reranker_config.min_available_commit_mib * _MIB
    physical_low = (
        required_physical > 0
        and snapshot.available_physical is not None
        and snapshot.available_physical < required_physical
    )
    commit_low = (
        required_commit > 0
        and snapshot.available_commit is not None
        and snapshot.available_commit < required_commit
    )
    if not physical_low and not commit_low:
        return
    available_physical_mib = (
        None
        if snapshot.available_physical is None
        else snapshot.available_physical // _MIB
    )
    available_commit_mib = (
        None if snapshot.available_commit is None else snapshot.available_commit // _MIB
    )
    raise RerankerCapacityError(
        "重排模型加载前可用容量不足"
        f"（source={snapshot.source}, "
        f"available_physical_mib={available_physical_mib}, "
        f"required_physical_mib={reranker_config.min_available_physical_mib}, "
        f"available_commit_mib={available_commit_mib}, "
        f"required_commit_mib={reranker_config.min_available_commit_mib}）"
    )


def get_reranker_model() -> FlagReranker:
    """
    获取重排模型单例对象。

    Returns:
        FlagReranker: 初始化完成的重排模型实例。
    """
    global _reranker_model
    if _reranker_model is None:
        # 必须先于 FlagReranker/native loader；后者在极低 commit 下可能直接终止进程。
        _ensure_reranker_load_capacity()
        logger.info("开始初始化重排模型")
        _reranker_model = FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16,
            batch_size=reranker_config.batch_size,
            query_max_length=reranker_config.query_max_tokens,
            max_length=reranker_config.max_input_tokens,
        )
        logger.success("重排模型初始化成功")
    return _reranker_model
