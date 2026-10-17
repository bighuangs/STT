import sys

from app.shared.runtime.logger import logger, node_log
from app.rag.query.hyde_search_service import search_embedding_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        mivlus_result = search_embedding_hyde(state)
    except Exception as exc:
        # HyDE 是增强召回支路，不应因其失败中断普通向量检索和最终回答。
        # 保留完整堆栈便于运维排查，并以空结果进入 RRF 单路融合。
        logger.exception(f"HyDE检索失败，已降级为普通向量检索：{exc}")
        mivlus_result = []
        state["hyde_retrieval_diagnostics"] = {
            "query_variant": "hyde",
            "routes": [
                {
                    "route": getattr(exc, "route", None) or "hyde",
                    "status": "failed",
                    "error_code": getattr(exc, "code", "HYDE_FAILED"),
                    "message": str(exc),
                }
            ],
        }
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = {"hyde_embedding_chunks": mivlus_result}
    if state.get("retrieval_profile_id"):
        result["hyde_retrieval_diagnostics"] = state.get(
            "hyde_retrieval_diagnostics", {}
        )
    return result

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)
