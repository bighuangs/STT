import sys

from app.domain.query_resilience import RequiredRouteError
from app.shared.runtime.logger import logger, node_log
from app.rag.query.embedding_search_service import search_by_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id


def _local_route_is_required(state: dict) -> bool:
    """仅信任已经通过领域校验并写入状态的 QueryPlan。

    兼容旧入口：没有 QueryPlan 时仍维持原来的空结果降级行为；可靠 QueryRun
    一定携带 QueryPlan，因此必选本地路线的依赖失败会在生成答案前直接终止图。
    """

    return any(
        route.get("route_type") == "local_hybrid" and route.get("required") is True
        for route in ((state.get("query_plan") or {}).get("routes") or [])
        if isinstance(route, dict)
    )


@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    # 先记录节点开始，和 SSE 进度展示保持一致。
    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    # 这里只保留节点调度职责，参数校验和检索执行统一交给 service 入口。
    try:
        mivlus_result = search_by_embedding(state)
    except Exception as exc:
        if _local_route_is_required(state):
            # required local 是内部知识回答的信任根。此处必须在 RRF/生成前失败，
            # 禁止 optional Web/HyDE 的结果把本地依赖故障掩盖成成功回答。
            raise RequiredRouteError(
                "route_local_hybrid",
                str(getattr(exc, "code", "LOCAL_RETRIEVAL_FAILED")),
            ) from exc
        # 无 QueryPlan 的旧入口保持兼容：以空候选进入证据门控并拒答。
        logger.exception("本地知识库检索失败，将由证据门控收敛：%s", exc)
        mivlus_result = []
        state["embedding_retrieval_diagnostics"] = {
            "query_variant": "base",
            "status": "failed",
            "error_code": getattr(exc, "code", "LOCAL_RETRIEVAL_FAILED"),
            "message": str(exc)[:500],
        }
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = {"embedding_chunks": mivlus_result}
    if state.get("retrieval_profile_id"):
        result["embedding_retrieval_diagnostics"] = state.get("embedding_retrieval_diagnostics", {})
    return result


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "is_stream": False,
    }
    result = node_search_embedding(test_state)
    print(result)
