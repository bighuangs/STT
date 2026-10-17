import sys
from datetime import UTC, datetime
from hashlib import sha256

from app.application.query_security_pipeline import (
    QueryEgressDenied,
    authorize_external_call,
    inspect_user_query_state,
)
from app.domain.query_planning import PlanningPolicy
from app.rag.query.query_planning_service import (
    build_query_plan_from_state,
    query_plan_state_patch,
)
from app.security.principal import current_principal
from app.shared.runtime.logger import node_log
from app.rag.query.query_rewrite_service import (
    load_history,
    rewrite_query,
    save_user_message,
    validate_query_identity,
)
from app.shared.utils.task_utils import add_done_task, add_running_task, query_task_id


def _runtime_deadline(state: dict) -> datetime | None:
    raw = state.get("deadline_at")
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        value = raw
    else:
        value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("deadline_at must include timezone")
    return value.astimezone(UTC)


@node_log("node_query_rewrite")
def node_query_rewrite(state):
    """
    节点功能：查询改写（检索前的问题精炼）。

    对齐 ragflow dialog_service.py:643-663：
        full_question（多轮 → 独立问题）/ cross_languages（跨语言扩展）/
        apply_meta_data_filter（元数据 → doc_ids）/ keyword_extraction（关键词追加）。

    输入：state['original_query'] / state['session_id']（可选 state['meta_data_filter']）
    输出：更新 state['rewritten_query'] / state['questions'] / state['keywords'] / state['doc_ids']
    """
    task_id = query_task_id(state)
    # 输入安全检查不属于查询改写；通过后才登记节点，阻断时不会遗留 running。
    inspect_user_query_state(state)
    add_running_task(task_id, sys._getframe().f_code.co_name, state["is_stream"])
    _original_query, session_id = validate_query_identity(state)
    # Planner 先消费冻结的历史快照做确定性硬决策。问候、UI 帮助和必须澄清
    # 的问题不会先调用关键词/改写 LLM。
    history = (
        list(state.get("history") or [])
        if state.get("evaluation_mode", False) or state.get("runtime_managed_history", False)
        else load_history(session_id)
    )
    state["history"] = history
    state["history_snapshot_loaded"] = True
    state["rewritten_query"] = ""
    profile = state.get("retrieval_profile") or {}
    planning = profile.get("query_planning") or {}
    principal = current_principal()
    tenant_id = principal.tenant_id if principal else "tenant_default"
    scope_seed = (
        f"{tenant_id}|{state.get('kb_id')}|{state.get('session_id')}|{state.get('query_run_id')}"
    )
    policy = PlanningPolicy(
        allow_hyde=bool(planning.get("allow_hyde", True)),
        allow_web=bool(state.get("web_search_allowed", False)),
        max_subqueries=int(planning.get("max_subqueries", 4)),
        max_dependency_depth=int(planning.get("max_dependency_depth", 2)),
        max_routes=int(planning.get("max_routes", 3)),
        total_timeout_ms=max(
            250,
            min(120_000, int(profile.get("timeout_ms") or 8_000)),
        ),
        default_top_k=max(
            1,
            min(100, int(profile.get("top_k") or 8)),
        ),
        max_top_k=max(
            1,
            min(1_000, int(profile.get("candidate_k") or 40)),
        ),
        token_budget=int(planning.get("token_budget", 4096)),
        cost_budget=float(planning.get("cost_budget", 1.0)),
    )
    # candidate_k 可能小于默认 top_k 的旧配置在 TODO-09 已被 schema 禁止；
    # 这里仍做边界收敛，避免历史配置让 planner 自身报 500。
    if policy.max_top_k < policy.default_top_k:
        policy = policy.model_copy(update={"max_top_k": policy.default_top_k})

    def build_plan():
        return build_query_plan_from_state(
            state,
            tenant_id=tenant_id,
            scope_id=f"qscope_{sha256(scope_seed.encode('utf-8')).hexdigest()[:32]}",
            document_ids=tuple(
                item
                for item in (state.get("doc_ids") or [])
                if item and not str(item).startswith("__stt_")
            ),
            policy=policy,
            deadline_at=_runtime_deadline(state),
        )

    plan = build_plan()
    if plan.disposition.value == "retrieve":
        # 只有经过硬规则批准的检索计划才调用多轮改写、跨语言/关键词和元数据
        # 服务；rewrite_query 会复用上面的同一历史快照。
        try:
            authorize_external_call(
                state,
                purpose="llm_generation",
                provider="configured-llm",
            )
        except QueryEgressDenied as exc:
            # 外发被禁时仍保留本地检索能力；使用原问题，不调用任何改写 LLM。
            state["rewritten_query"] = _original_query
            state["questions"] = [_original_query]
            state["keywords"] = ""
            state["doc_ids"] = []
            state["query_rewrite_degradation"] = {
                "reason_code": exc.reason_code,
                "fallback": "original_query",
            }
        else:
            state = rewrite_query(state)
        plan = build_plan()
    else:
        state["rewritten_query"] = plan.understanding.standalone_query
        state["questions"] = [plan.understanding.standalone_query]
        state["keywords"] = ""
        state["doc_ids"] = []
        if not state.get("evaluation_mode", False) and not state.get(
            "runtime_managed_history", False
        ):
            save_user_message(state)
    state.update(query_plan_state_patch(plan))
    # Planner 的 standalone_query 是经过实体/数字/否定语义保真校验后的权威
    # 检索问题；展示仍始终使用 original_query。
    state["rewritten_query"] = plan.understanding.standalone_query
    # 改写完成后写入完成列表，方便前端展示当前节点已结束。
    add_done_task(task_id, sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == "__main__":
    # ============================================================
    # 查询改写节点自测（依赖 Mongo / LLM / Milvus，需在配好 .env 的环境运行）
    #
    # 一次性覆盖 rewrite_query 的四个能力（对齐 ragflow dialog_service.py:643-663）：
    #   1) full_question         多轮追问 → 独立完整问题
    #   2) cross_languages       跨语言扩展（CROSS_LANGUAGE_TARGETS=[] 时默认关闭，questions 仅含原问题）
    #   3) apply_meta_data_filter 元数据过滤 → doc_ids（auto / manual 两种）
    #   4) keyword_extraction    关键词抽取并追加到 questions[-1]
    #
    # 元数据用例直接命中【导入流水线真实写入 Milvus 的 meta_fields】，知识库 = KB_ID
    # （与 import_/agent/main_graph.py 全流程测试同一个库），不再打桩。
    # 每个用例使用独立 session 且先清空历史，保证可复现、互不串扰。
    # ============================================================
    import re

    from app.process.query.agent.state import create_query_default_state
    from app.infra.persistence.history_repository import history_repository
    from app.rag.query.query_rewrite_service import load_meta_catalog

    # 导入流水线写入的知识库 ID（见 import_/agent/main_graph.py 全流程测试）
    KB_ID = "test_kb_001"

    def _seed_multiturn(session_id: str) -> None:
        """播种一轮前序对话，让 full_question 有上下文可做指代消解。"""
        history_repository.save_message(
            session_id=session_id,
            role="user",
            text="哎哟我HAK 180不知道咋自动关了，现在啷个开啊？",
            rewritten_query="HAK 180 关机后怎么开机？",
        )
        history_repository.save_message(
            session_id=session_id,
            role="assistant",
            text="先打开总电源，再按面板上的启动键即可开机。",
        )

    def run_case(
        title: str,
        *,
        original_query: str,
        multiturn: bool = False,
        meta_data_filter: dict | None = None,
        kb_id: str | None = None,
    ) -> dict:
        """跑单个用例：独立 session（先清空保证可复现）→ 执行节点 → 打印产物。"""
        session_id = "test_qr_" + re.sub(r"\W+", "_", title)
        history_repository.clear_session(session_id)
        if multiturn:
            _seed_multiturn(session_id)

        state = create_query_default_state(
            session_id=session_id,
            original_query=original_query,
            is_stream=False,
            meta_data_filter=meta_data_filter,
            kb_id=kb_id,
        )
        result = node_query_rewrite(state)

        print("\n" + "=" * 64)
        print(f"用例：{title}（{'多轮' if multiturn else '单轮'}）")
        print("-" * 64)
        print(f"原始问题   original_query : {result.get('original_query')}")
        print(f"独立问题   rewritten_query: {result.get('rewritten_query')}")
        print(f"检索变体   questions      : {result.get('questions')}")
        print(f"抽取关键词 keywords       : {result.get('keywords')}")
        print(f"元数据配置 meta_filter    : {meta_data_filter}")
        print(f"候选文档   doc_ids        : {result.get('doc_ids')}")
        return result

    # ---- 步骤 0：打印 Milvus 真实元数据目录，便于核对下方过滤用例是否命中 ----
    print("=" * 64)
    print(f"Milvus 元数据目录（kb_id={KB_ID}）")
    print("-" * 64)
    catalog = load_meta_catalog(KB_ID)
    for key, values in catalog.items():
        detail = ", ".join(f"{v}×{len(docs)}篇" for v, docs in values.items())
        print(f"  {key}: {{ {detail} }}")
    if not catalog:
        print("  （空——请确认导入流水线已写入 meta_fields，且 KB_ID 正确）")

    # ---- 用例 1：多轮追问 → full_question 补全为独立问题（meta 过滤关闭）----
    run_case(
        "多轮改写 full_question",
        original_query="那它之后啷个关机勒？",  # 带代词「它」、省略主语，依赖上下文
        multiturn=True,
    )

    # ---- 用例 2：单轮对照 —— 无历史，full_question 不触发，rewritten==原问题 ----
    run_case(
        "单轮对照 不改写",
        original_query="HAK 180 咋用？",
    )

    # ---- 用例 3：元数据过滤 auto —— LLM 依据问题 + 真实目录自动生成条件命中 doc_ids ----
    run_case(
        "元数据过滤 auto",
        original_query="HAK 180 的产品安全怎么样",
        meta_data_filter={"method": "auto"},
        kb_id=KB_ID,
    )

    # ---- 用例 4：元数据过滤 manual —— 直接给定条件（文档类型=安全手册）----
    run_case(
        "元数据过滤 manual",
        original_query="关机注意事项",
        meta_data_filter={
            "method": "manual",
            "logic": "and",
            "manual": [{"key": "设备型号", "value": "Lenovo CS407", "op": "="}],
        },
        kb_id=KB_ID,
    )
