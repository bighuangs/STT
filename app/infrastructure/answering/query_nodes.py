"""Proof-carrying answer v2 的查询图节点适配器。

节点只负责把 Graph state 交给 Application 用例，并在组合边界注入模型、
渲染与提交适配器；逻辑 node ID 仍由查询工作流契约拥有。
"""

from __future__ import annotations

import sys

from app.application.query_answering.pipeline import (
    commit_proven_answer,
    decide_answer_coverage,
    freeze_question_contract,
    freeze_witness_catalog,
    plan_answer,
    render_proven_answer,
    verify_answer_plan,
)
from app.infrastructure.answering.delivery import commit_answer_delivery
from app.infrastructure.answering.provider import configured_answering_model_adapter
from app.infrastructure.answering.render_adapter import configured_answer_render_adapter
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id


@node_log("node_question_contract")
def node_question_contract(state):
    """在检索前冻结用户问题义务。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = freeze_question_contract(state)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_evidence_witness")
def node_evidence_witness(state):
    """把 ContextPack 转换为精确、可验证的证据见证目录。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = freeze_witness_catalog(state)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_answer_plan")
def node_answer_plan(state):
    """让模型只选择义务与见证 ID，并由服务端固化计划。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = plan_answer(state, model_port=configured_answering_model_adapter)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_answer_verify")
def node_answer_verify(state):
    """复核计划的语义相关性，不接收模型生成的事实正文。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = verify_answer_plan(state, model_port=configured_answering_model_adapter)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_answer_coverage")
def node_answer_coverage(state):
    """验证必需义务覆盖，并决定是否使用唯一一次修复回边。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = decide_answer_coverage(state)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_answer_render")
def node_answer_render(state):
    """仅从已验证见证确定性渲染答案、引用与最终证明。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = render_proven_answer(state, render_port=configured_answer_render_adapter)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


@node_log("node_answer_commit")
def node_answer_commit(state):
    """在证明已 checkpoint 后执行历史与交付副作用。"""

    task_id = query_task_id(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    result = commit_proven_answer(state, commit_port=commit_answer_delivery)
    add_done_task(task_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return result


__all__ = [
    "node_answer_commit",
    "node_answer_coverage",
    "node_answer_plan",
    "node_answer_render",
    "node_answer_verify",
    "node_evidence_witness",
    "node_question_contract",
]
