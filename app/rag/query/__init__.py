"""查询域能力的惰性导出，避免导入轻量过滤器时初始化 Mongo/LLM 客户端。"""

from importlib import import_module


_EXPORTS = {
    "rewrite_query": ("app.rag.query.query_rewrite_service", "rewrite_query"),
    "rerank_documents": ("app.rag.query.rerank_service", "rerank_documents"),
    "fuse_by_rrf": ("app.rag.query.rrf_service", "fuse_by_rrf"),
    "search_embedding_hyde": ("app.rag.query.hyde_search_service", "search_embedding_hyde"),
    "search_by_embedding": ("app.rag.query.embedding_search_service", "search_by_embedding"),
    "search_by_web": ("app.rag.query.web_search_service", "search_by_web"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "rewrite_query",
    "fuse_by_rrf",
    "rerank_documents",
    "search_by_embedding",
    "search_embedding_hyde",
    "search_by_web",
    # "validate_retrieval_state",
]
