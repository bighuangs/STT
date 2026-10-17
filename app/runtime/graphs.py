"""LangGraph 编译与进程内缓存的唯一生产运行时入口。"""

from __future__ import annotations

from threading import RLock
from typing import Any


_lock = RLock()
_query_graph: Any | None = None
_ingestion_graph: Any | None = None


def get_query_graph():
    """首次执行查询时编译图，避免模块导入产生编译副作用。"""

    global _query_graph
    if _query_graph is not None:
        return _query_graph
    with _lock:
        if _query_graph is None:
            from app.process.query.agent.main_graph import compile_query_graph

            _query_graph = compile_query_graph()
        return _query_graph


def get_ingestion_graph():
    """首次执行导入任务时编译图，并在当前 Worker 进程内复用。"""

    global _ingestion_graph
    if _ingestion_graph is not None:
        return _ingestion_graph
    with _lock:
        if _ingestion_graph is None:
            from app.process.import_.agent.main_graph import compile_ingestion_graph

            _ingestion_graph = compile_ingestion_graph()
        return _ingestion_graph


def reset_graph_registry() -> None:
    """清除图缓存；只应用于测试隔离与受控重装配。"""

    global _query_graph, _ingestion_graph
    with _lock:
        _query_graph = None
        _ingestion_graph = None


__all__ = ["get_ingestion_graph", "get_query_graph", "reset_graph_registry"]
