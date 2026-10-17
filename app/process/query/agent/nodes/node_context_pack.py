import sys

from app.rag.query.context_assembly_service import prepare_query_context
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id


@node_log("node_context_pack")
def node_context_pack(state):
    """构造硬预算 ContextPack；只保留真正进入生成上下文的 Evidence。"""

    task_id = query_task_id(state)
    add_running_task(
        task_id,
        sys._getframe().f_code.co_name,
        state.get("is_stream"),
    )
    state = prepare_query_context(state)
    add_done_task(
        task_id,
        sys._getframe().f_code.co_name,
        state.get("is_stream"),
    )
    return state
