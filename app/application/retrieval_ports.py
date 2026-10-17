"""查询编排与检索基础设施之间的端口。

业务节点只依赖 Protocol；生产环境可以注入 Milvus/BGE，契约测试可以注入纯内存
Fake。这里不能导入 pymilvus 或 FlagEmbedding。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.retrieval import (
    RerankRequest,
    RerankResult,
    RetrievalRequest,
    RetrievalResult,
)


@runtime_checkable
class RetrievalPort(Protocol):
    """候选召回、路线分数与检索配置版本的统一边界。"""

    def search(self, request: RetrievalRequest) -> RetrievalResult: ...


@runtime_checkable
class RerankerPort(Protocol):
    """对标准检索命中重排，不接收 Milvus 原始对象。"""

    def rerank(self, request: RerankRequest) -> RerankResult: ...
