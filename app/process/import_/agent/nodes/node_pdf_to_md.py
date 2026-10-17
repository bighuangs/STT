import os

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState , create_default_state
from app.application.document_parsing_service import apply_parsing_to_graph_state
from app.shared.runtime.logger import logger , PROJECT_ROOT

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    TODO-07 通用解析兼容节点。

    节点名保留 ``node_pdf_to_md`` 是为了兼容 TODO-02 已持久化的 resume_node；
    实际语义已升级为 ParserRegistry → ParsedDocument artifact → ChunkTemplate，新增格式
    只注册 parser，不再修改 LangGraph 拓扑。
    """
    add_running_task(state["task_id"], "node_pdf_to_md")
    state = apply_parsing_to_graph_state(state)
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state

if __name__ == "__main__":
    logger.info("===== 开始 node_pdf_to_md 节点联调测试 =====")

    test_pdf_path = os.path.join(PROJECT_ROOT, "doc", "hak180使用说明书.pdf")
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )

    result = node_pdf_to_md(test_state)
    logger.info(f"md_path: {result['md_path']}")
    logger.info(f"md_content长度: {len(result['md_content'])}")
    logger.info("===== 结束 node_pdf_to_md 节点联调测试 =====")
