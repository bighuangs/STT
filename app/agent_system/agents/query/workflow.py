"""Query StateGraph 与耐久执行器共享的纯工作流契约。

这里仅声明节点身份、路由和副作用类别，不导入 LangGraph、SQLAlchemy 或 legacy
节点实现。外层适配器负责注入 handler，避免 strict inner package 反向依赖实现层。
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from app.application.query_answering.pipeline import AnswerProofNotCommittableError
from app.application.query_answering.proposals import MAX_REPAIR_ATTEMPTS
from app.domain.answering.proof import AnswerProof


QUERY_WORKFLOW_VERSION = "query-workflow.v4"
QUERY_ENTRY_NODE = "node_query_rewrite"


@dataclass(frozen=True, slots=True)
class QueryNodeDefinition:
    """稳定节点身份与副作用类别；类别不构成 exactly-once 承诺。"""

    effect_class: str


QUERY_NODE_REGISTRY: Mapping[str, QueryNodeDefinition] = MappingProxyType(
    {
        "node_query_rewrite": QueryNodeDefinition("provider_and_history"),
        "node_question_contract": QueryNodeDefinition("deterministic_transform"),
        "node_retrieval_policy": QueryNodeDefinition("provider_and_retrieval"),
        "node_rrf": QueryNodeDefinition("deterministic_transform"),
        "node_rerank": QueryNodeDefinition("model_or_local_transform"),
        "node_context_pack": QueryNodeDefinition("deterministic_transform"),
        "node_evidence_witness": QueryNodeDefinition("deterministic_transform"),
        "node_answer_plan": QueryNodeDefinition("provider_or_deterministic_repair_transform"),
        "node_answer_verify": QueryNodeDefinition("provider_transform"),
        "node_answer_coverage": QueryNodeDefinition("deterministic_transform"),
        "node_answer_render": QueryNodeDefinition("citation_persistence"),
        "node_answer_commit": QueryNodeDefinition("history_and_delivery"),
    }
)

# 问题契约决定是否检索，覆盖节点只允许两次显式修复回边。
QUERY_STATIC_SUCCESSORS: Mapping[str, str | None] = MappingProxyType(
    {
        "node_retrieval_policy": "node_rrf",
        "node_rrf": "node_rerank",
        "node_rerank": "node_context_pack",
        "node_context_pack": "node_evidence_witness",
        "node_evidence_witness": "node_answer_plan",
        "node_answer_plan": "node_answer_verify",
        "node_answer_verify": "node_answer_coverage",
        "node_answer_commit": None,
    }
)


def node_query_rewrite_after_router(state: Mapping[str, Any]) -> str:
    """改写后总是先冻结问题契约，禁止在 ContextPack 后反推问题义务。"""

    return "node_question_contract"


def node_question_contract_after_router(state: Mapping[str, Any]) -> str:
    """只按冻结的 QueryPlan 决定检索或构造合法空见证目录。"""

    plan = state.get("query_plan") or {}
    if not isinstance(plan, Mapping):
        return "node_evidence_witness"
    disposition = plan.get("disposition")
    if disposition in {"direct_answer", "need_clarification"}:
        return "node_evidence_witness"
    allowed_route_types = {"local_hybrid", "hyde", "web"}
    routes = plan.get("routes") or []
    selected = {
        str(route.get("route_type") or "") for route in routes if isinstance(route, Mapping)
    }
    if not selected or not selected.issubset(allowed_route_types):
        return "node_evidence_witness"
    return "node_retrieval_policy"


def node_answer_coverage_after_router(state: Mapping[str, Any]) -> str:
    """覆盖失败最多回到计划节点两次，随后必须确定性收敛。"""

    requested = bool(state.get("answer_repair_requested", False))
    try:
        repair_count = int(state.get("answer_repair_count") or 0)
    except (TypeError, ValueError):
        repair_count = MAX_REPAIR_ATTEMPTS
    if requested and 0 <= repair_count < MAX_REPAIR_ATTEMPTS:
        return "node_answer_plan"
    return "node_answer_render"


def node_answer_render_after_router(state: Mapping[str, Any]) -> str:
    """只有领域可提交 proof 才能进入带外部副作用的 commit 节点。"""

    try:
        proof = AnswerProof.from_dict(state.get("answer_proof") or {})
    except (TypeError, ValueError) as exc:
        raise AnswerProofNotCommittableError("answer proof is missing or invalid") from exc
    if not proof.is_committable:
        raise AnswerProofNotCommittableError(
            f"answer proof is not committable: {proof.overall_status.value}"
        )
    return "node_answer_commit"


def next_query_node(
    completed_node: str | None,
    state: Mapping[str, Any],
) -> str | None:
    """返回已提交节点后的后继；``None`` 输入表示尚未执行入口节点。"""

    if completed_node is None:
        return QUERY_ENTRY_NODE
    if completed_node == "node_query_rewrite":
        return node_query_rewrite_after_router(state)
    if completed_node == "node_question_contract":
        return node_question_contract_after_router(state)
    if completed_node == "node_answer_coverage":
        return node_answer_coverage_after_router(state)
    if completed_node == "node_answer_render":
        return node_answer_render_after_router(state)
    if completed_node not in QUERY_STATIC_SUCCESSORS:
        raise ValueError("query workflow completed node is unknown")
    return QUERY_STATIC_SUCCESSORS[completed_node]


def validate_query_workflow(registered_handlers: Set[str] | None = None) -> None:
    """验证 registry、路由目标以及外层注入的 handler 集合闭合。"""

    if QUERY_ENTRY_NODE not in QUERY_NODE_REGISTRY:
        raise RuntimeError("query workflow entry node is not registered")
    static_sources = set(QUERY_STATIC_SUCCESSORS)
    conditional_sources = {
        "node_query_rewrite",
        "node_question_contract",
        "node_answer_coverage",
        "node_answer_render",
    }
    expected_sources = set(QUERY_NODE_REGISTRY) - conditional_sources
    if static_sources != expected_sources:
        raise RuntimeError("query workflow static successors do not cover registered nodes")
    targets = {target for target in QUERY_STATIC_SUCCESSORS.values() if target is not None}
    targets.update(
        {
            "node_question_contract",
            "node_evidence_witness",
            "node_retrieval_policy",
            "node_answer_plan",
            "node_answer_render",
            "node_answer_commit",
        }
    )
    if not targets.issubset(QUERY_NODE_REGISTRY):
        raise RuntimeError("query workflow routes to an unregistered node")
    if registered_handlers is not None and set(registered_handlers) != set(QUERY_NODE_REGISTRY):
        raise RuntimeError("query workflow handlers do not match the node registry")


__all__ = [
    "QUERY_ENTRY_NODE",
    "QUERY_NODE_REGISTRY",
    "QUERY_STATIC_SUCCESSORS",
    "QUERY_WORKFLOW_VERSION",
    "QueryNodeDefinition",
    "next_query_node",
    "node_answer_coverage_after_router",
    "node_answer_render_after_router",
    "node_question_contract_after_router",
    "node_query_rewrite_after_router",
    "validate_query_workflow",
]
