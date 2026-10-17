from typing_extensions import TypedDict
from typing import List
import copy


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["session_id"]、state.get("answer")）。
    """

    session_id: str  # 会话唯一标识
    conversation_id: str  # TODO-24 长期会话身份；兼容入口可与 session_id 相同
    turn_id: str  # TODO-24 单个用户轮身份
    tenant_id: str  # 服务端认证事实，不接受客户端覆盖
    user_id: str  # 服务端认证事实，不接受客户端覆盖
    original_query: str  # 用户原始问题
    kb_id: str  # 可选知识库范围；空时查询所有 active binding
    web_search_allowed: bool  # 仅由服务端 KB 策略注入，客户端不可覆盖
    config_version: str  # TODO-05 脱敏配置版本，用于查询 trace 复现
    retrieval_profile: dict  # TODO-09 会话固定的已发布检索配置快照
    retrieval_profile_id: str  # 检索配置稳定 ID
    retrieval_profile_version: int  # 检索配置不可变版本
    retrieval_profile_hash: str  # 检索配置规范化哈希
    retrieval_filter: dict | None  # 客户端结构化 Filter AST；不得包含系统隔离字段
    retrieval_diagnostics: dict  # 实际执行路由、逐路分数、错误与耗时
    embedding_retrieval_diagnostics: dict  # 普通 query variant 的检索诊断
    hyde_retrieval_diagnostics: dict  # HyDE query variant 的检索诊断
    web_retrieval_diagnostics: dict  # Web 可选路线检索诊断
    evaluation_mode: bool  # 离线评测模式：禁历史写入与联网副作用
    answer_id: str  # 助手消息/评测观察的稳定身份
    query_run_id: str  # 单次查询运行稳定身份
    trace_id: str  # 跨模块诊断身份；默认等于 query_run_id
    usage: dict  # token/cost；不可获得时显式标记 unavailable
    model_usage_summaries: list  # 每次改写/HyDE/生成调用的无正文安全摘要
    usage_telemetry_errors: list  # telemetry fail-open 的低基数错误回执
    llm_model_id: str  # 服务端在运行接受时冻结的回答模型；空值兼容旧入口默认模型
    query_understanding: dict  # TODO-27 结构化意图/实体/时效/歧义
    query_facet_ledger: dict  # 版本化信息需求账本；旧 checkpoint 缺失时按空账本读取
    query_plan: dict  # TODO-27 经授权与预算校验的不可变执行计划
    query_execution_summary: dict  # TODO-10 实际路线状态、降级和保守预算用量
    deadline_at: str | None  # QueryRun 的绝对 UTC 截止时间，供执行器与计划取交集

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档
    prepack_reranked_docs: list  # ContextPack 前的真实 reranker 输出，供评测/诊断区分装包损失
    fusion_result: dict  # TODO-28 融合版本、候选与淘汰诊断
    fusion_diagnostics: dict  # 可返回给 API 的脱敏融合摘要

    # Proof-Carrying Answer 过程中的数据
    answer: str  # 最终生成的答案
    question_contract: dict  # 原始 QueryPlan facet ledger 冻结出的回答义务
    witness_catalog: dict  # ContextPack 的精确、可寻址证据原子目录
    answer_plan_proposal: dict  # provider 仅返回义务/见证 ID 的结构化提案
    answer_plan_raw_proposal: dict  # provider 原始提案；仅供无正文审计
    answer_plan_normalizations: list  # required_output_form 的服务端规范化回执
    answer_plan: dict  # 服务端验证后的权威计划，不含自由事实正文
    answer_reviewed_plan: dict  # reviewer 实际审查的不可变 AnswerPlan 快照
    answer_plan_review: dict  # 语义复核回执，只含 ID、判定与原因码
    answer_review_closure_receipt: dict  # 复核对最终 plan/catalog 的闭合绑定回执
    answer_verification: dict  # claim/witness/obligation 逐项验证结果
    answer_coverage_decision: dict  # 覆盖结论与两次有界修复边的控制状态
    answer_repair_feedback: dict  # reviewer 给出的完整 ID-only replacement
    answer_repair_requested: bool
    answer_repair_count: int
    answer_repair_transition: dict  # replacement 到确定性新计划的来源绑定回执
    answer_repair_transitions: list  # 两次有界 replacement 的 append-only 谱系
    answer_proof: dict  # 最终逐 claim/obligation 证明与 strict 资格
    answer_prompt_budget: dict  # 四逻辑调用最坏路径 token 与输出 cap 的冻结回执
    answer_prompt_preflights: list  # 每次 provider 前的预算重放摘要，不含正文
    answer_model_ledger: dict  # 四逻辑调用及 provider attempts 的预留/核销账本
    answer_prompt_receipts: list  # Prompt 版本/hash/token 审计，不保存正文
    answer_render_receipt: dict  # 确定性渲染 hash/span 回执
    pending_citation_facts: list  # proof 完成后等待 commit 的精确 CitationFact
    answer_commit_receipt: dict  # 历史/SSE/任务结果提交状态
    query_model_call_audits: list  # provider 重试审计，不含 Prompt/response
    context_pack: dict  # TODO-29 可重放、硬预算的 Evidence ContextPack
    context_pack_summary: dict  # 不含正文的公开预算/覆盖摘要
    context_aliases: list  # ContextPack E<n> 到 Evidence 的服务端映射
    context_budget_blocked: bool  # 固定 Prompt 已超过模型窗口
    context_budget_reason: str
    evidence_assessment: dict  # TODO-30 生成前证据充分性
    answer_decision: dict  # TODO-30 结构化回答动作
    grounding: dict  # TODO-30 claim 级生成后校验
    answer_envelope: dict  # TODO-11 权威 answer.v1 终态

    # 查询改写阶段产物（node_query_rewrite 写入，对齐 ragflow dialog_service.py:643-663）
    rewritten_query: (
        str  # full_question 多轮改写后的「独立完整问题」（单轮=原问题，用于历史/展示/联网检索）
    )
    questions: List[
        str
    ]  # 检索变体列表（rewritten_query + cross_languages 译文，末位追加 keywords，向量/HyDE 检索用 " ".join）
    keywords: (
        str  # keyword_extraction 抽取的关键词（逗号拼接，仅记录；检索追加在 questions[-1] 中）
    )
    doc_ids: List[str]  # apply_meta_data_filter 元数据过滤命中的候选文档ID（空=不按文档过滤）
    meta_data_filter: dict  # 元数据过滤配置（method=auto/semi_auto/manual；None/空=关闭）

    # 辅助信息
    history: list  # 历史对话记录
    history_snapshot_loaded: bool  # 本次 query run 已冻结历史，后续禁止重读
    history_snapshot_hash: str  # TODO-25 冻结历史的可重放 hash
    history_base_turn_seq: int  # 本次运行可见的最后已决轮次
    history_version: int  # 会话历史提交版本
    runtime_managed_history: bool  # True 时 SQL Turn/Message 是唯一历史事实源
    authorized_scope: dict  # TODO-26 受权范围快照，所有敏感阶段消费同一 hash
    vector_read_scopes: list  # TODO-37 QueryRun 接受时冻结的物理 collection/partition
    vector_read_set_digest: str  # 冻结物理读集的规范化摘要，执行时 fail-closed 校验
    scope_consumed_stages: dict  # stage -> scope_hash，证明未在下游换 scope
    data_classification: str  # 控制面 KB 数据分类
    external_llm_allowed: bool  # 仅控制面注入；实际外发仍逐调用决策
    egress_policies: dict  # purpose -> EgressPolicySnapshot
    egress_decisions: list  # 仅含决策元数据，不含外发正文
    content_security_policy: dict  # 冻结的 ContentSecurityPolicy
    content_security: dict  # 输入/证据/输出判定摘要，不含被隔离正文
    is_stream: bool  # 是否流式输出标记
    image_urls: List[str]  # 答案中引用的图片链接
    evidences: list  # 经控制库恢复并二次授权的 Evidence
    evidence_aliases: dict  # 本次回答短标签 E1 -> 稳定 evidence_id
    citations: list  # 服务端校验后的 Citation
    sources: list  # 前端可展示/下钻的来源摘要
    citation_validation: dict  # 引用覆盖率、可信状态与问题码


# 默认状态
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "conversation_id": "",
    "turn_id": "",
    "tenant_id": "",
    "user_id": "",
    "original_query": "",
    "kb_id": "",
    "web_search_allowed": False,
    "config_version": "cfg_unknown",
    "retrieval_profile": {},
    "retrieval_profile_id": "",
    "retrieval_profile_version": 0,
    "retrieval_profile_hash": "",
    "retrieval_filter": None,
    "retrieval_diagnostics": {},
    "embedding_retrieval_diagnostics": {},
    "hyde_retrieval_diagnostics": {},
    "web_retrieval_diagnostics": {},
    "evaluation_mode": False,
    "answer_id": "",
    "query_run_id": "",
    "trace_id": "",
    "usage": {"status": "unavailable", "reason": "usage collector is not installed"},
    "model_usage_summaries": [],
    "usage_telemetry_errors": [],
    "llm_model_id": "",
    "query_understanding": {},
    "query_facet_ledger": {},
    "query_plan": {},
    "query_execution_summary": {},
    "deadline_at": None,
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prepack_reranked_docs": [],
    "fusion_result": {},
    "fusion_diagnostics": {},
    "answer": "",
    "question_contract": {},
    "witness_catalog": {},
    "answer_plan_proposal": {},
    "answer_plan_raw_proposal": {},
    "answer_plan_normalizations": [],
    "answer_plan": {},
    "answer_reviewed_plan": {},
    "answer_plan_review": {},
    "answer_review_closure_receipt": {},
    "answer_verification": {},
    "answer_coverage_decision": {},
    "answer_repair_feedback": {},
    "answer_repair_requested": False,
    "answer_repair_count": 0,
    "answer_repair_transition": {},
    "answer_repair_transitions": [],
    "answer_proof": {},
    "answer_prompt_budget": {},
    "answer_prompt_preflights": [],
    "answer_model_ledger": {},
    "answer_prompt_receipts": [],
    "answer_render_receipt": {},
    "pending_citation_facts": [],
    "answer_commit_receipt": {},
    "query_model_call_audits": [],
    "context_pack": {},
    "context_pack_summary": {},
    "context_aliases": [],
    "context_budget_blocked": False,
    "context_budget_reason": "",
    "evidence_assessment": {},
    "answer_decision": {},
    "grounding": {},
    "answer_envelope": {},
    "rewritten_query": "",
    "questions": [],
    "keywords": "",
    "doc_ids": [],
    "meta_data_filter": None,
    "history": [],
    "history_snapshot_loaded": False,
    "history_snapshot_hash": "",
    "history_base_turn_seq": 0,
    "history_version": 0,
    "runtime_managed_history": False,
    "authorized_scope": {},
    "vector_read_scopes": [],
    "vector_read_set_digest": "",
    "scope_consumed_stages": {},
    "data_classification": "internal",
    "external_llm_allowed": False,
    "egress_policies": {},
    "egress_decisions": [],
    "content_security_policy": {},
    "content_security": {},
    "is_stream": False,
    "image_urls": [],
    "evidences": [],
    "evidence_aliases": {},
    "citations": [],
    "sources": [],
    "citation_validation": {
        "status": "unverified",
        "coverage": 0.0,
        "citation_count": 0,
        "trusted_count": 0,
        "issues": [],
    },
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001", original_query="华为P60怎么样?", is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(state, original_query="修改后的问题")
    print("复制后的状态：", new_state)
