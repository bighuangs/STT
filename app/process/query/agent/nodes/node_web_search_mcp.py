import sys

from app.application.query_security_pipeline import authorize_external_call
from app.shared.runtime.logger import logger, node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息
    """
    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state["is_stream"])
    try:
        authorize_external_call(
            state,
            purpose="web_search",
            provider="dashscope-web",
        )
        plan = state.get("query_plan") or {}
        subqueries = {
            str(item.get("sub_query_id")): str(item.get("query") or "")
            for item in (plan.get("sub_queries") or [])
            if item.get("purpose") == "retrieval"
            and item.get("sub_query_id")
            and item.get("query")
        }
        web_route = next(
            (
                route
                for route in (plan.get("routes") or [])
                if route.get("route_type") == "web"
            ),
            None,
        )
        approved = [
            (sub_query_id, subqueries[sub_query_id])
            for sub_query_id in ((web_route or {}).get("sub_query_ids") or [])
            if sub_query_id in subqueries
        ]
        if not approved:
            approved = [(None, str(state.get("rewritten_query") or ""))]
        pages = []
        diagnostics = []
        for sub_query_id, query in approved:
            scoped_state = {**state, "rewritten_query": query}
            current = search_by_web(scoped_state, count=10)
            for document in current:
                pages.append({**document, "sub_query_id": sub_query_id})
            diagnostics.append(
                {
                    "sub_query_id": sub_query_id,
                    "query": query,
                    "status": "succeeded",
                    "candidate_count": len(current),
                }
            )
        state["web_retrieval_diagnostics"] = {
            "query_variant": "web",
            "status": "succeeded",
            "subqueries": diagnostics,
            "candidate_count": len(pages),
        }
    except Exception as exc:
        # Web 在 QueryPlan 中始终为可选路线；失败时保留诊断并继续本地回答。
        logger.exception("联网检索失败，已按计划降级为本地证据：%s", exc)
        pages = []
        state["web_retrieval_diagnostics"] = {
            "query_variant": "web",
            "status": "failed",
            "error_code": getattr(exc, "code", "WEB_RETRIEVAL_FAILED"),
            "message": str(exc)[:500],
        }
    add_done_task(task_id, sys._getframe().f_code.co_name, state["is_stream"])
    return {"web_search_docs": pages}

if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)
