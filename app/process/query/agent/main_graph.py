from langgraph.graph import END, StateGraph

from app.agent_system.agents.query.workflow import (
    QUERY_ENTRY_NODE,
    QUERY_NODE_REGISTRY,
    QUERY_STATIC_SUCCESSORS,
    node_answer_coverage_after_router,
    node_answer_render_after_router,
    node_question_contract_after_router,
    node_query_rewrite_after_router,
    validate_query_workflow,
)
from app.infrastructure.answering.query_nodes import (
    node_answer_commit,
    node_answer_coverage,
    node_answer_plan,
    node_answer_render,
    node_answer_verify,
    node_evidence_witness,
    node_question_contract,
)
from app.process.query.agent.nodes.node_context_pack import node_context_pack
from app.process.query.agent.nodes.node_query_rewrite import node_query_rewrite
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_retrieval_policy import node_retrieval_policy
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.state import QueryGraphState


_QUERY_NODE_HANDLERS = {
    "node_query_rewrite": node_query_rewrite,
    "node_question_contract": node_question_contract,
    "node_retrieval_policy": node_retrieval_policy,
    "node_rrf": node_rrf,
    "node_rerank": node_rerank,
    "node_context_pack": node_context_pack,
    "node_evidence_witness": node_evidence_witness,
    "node_answer_plan": node_answer_plan,
    "node_answer_verify": node_answer_verify,
    "node_answer_coverage": node_answer_coverage,
    "node_answer_render": node_answer_render,
    "node_answer_commit": node_answer_commit,
}


def query_node_handlers():
    """返回耐久执行器可注入的 handler 副本。"""

    validate_query_workflow(set(_QUERY_NODE_HANDLERS))
    return dict(_QUERY_NODE_HANDLERS)


def build_query_graph() -> StateGraph:
    """构造未编译的查询图定义；Runtime 决定编译与生命周期。"""

    handlers = query_node_handlers()
    graph = StateGraph(QueryGraphState)
    for node_name in QUERY_NODE_REGISTRY:
        graph.add_node(node_name, handlers[node_name])
    graph.set_entry_point(QUERY_ENTRY_NODE)
    graph.add_conditional_edges(
        "node_query_rewrite",
        node_query_rewrite_after_router,
        {
            "node_question_contract": "node_question_contract",
        },
    )
    graph.add_conditional_edges(
        "node_question_contract",
        node_question_contract_after_router,
        {
            "node_evidence_witness": "node_evidence_witness",
            "node_retrieval_policy": "node_retrieval_policy",
        },
    )
    graph.add_conditional_edges(
        "node_answer_coverage",
        node_answer_coverage_after_router,
        {
            "node_answer_plan": "node_answer_plan",
            "node_answer_render": "node_answer_render",
        },
    )
    graph.add_conditional_edges(
        "node_answer_render",
        node_answer_render_after_router,
        {
            "node_answer_commit": "node_answer_commit",
        },
    )
    for source, target in QUERY_STATIC_SUCCESSORS.items():
        graph.add_edge(source, END if target is None else target)
    return graph


def compile_query_graph():
    """编译查询图；生产入口应通过 ``app.runtime.graphs`` 调用。"""

    return build_query_graph().compile()


__all__ = [
    "build_query_graph",
    "compile_query_graph",
    "node_question_contract_after_router",
    "node_query_rewrite_after_router",
    "query_node_handlers",
]
