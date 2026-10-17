"""把已授权 Evidence 装入可复现 ContextPack，并做完整 Prompt 二次预算校验。"""

from __future__ import annotations

from typing import Any

import tiktoken

from app.application.query_answering.builders import (
    build_witness_catalog_bundle,
    resolve_witness_prompt_views,
)
from app.application.query_security_pipeline import (
    consume_authorized_scope,
    secure_evidence_documents,
)
from app.rag.query.context_pack_service import (
    ContextBudget,
    ContextPackPolicy,
    TiktokenTokenCounter,
    Utf8ByteUpperBoundCounter,
    build_context_pack,
)
from app.domain.answering.question import QuestionContract
from app.infrastructure.answering.prompts import (
    ANSWERING_MAX_LOGICAL_CALLS,
    ANSWERING_OUTPUT_TOKEN_CAP,
    ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION,
    MAX_PLAN_PROPOSAL_UTF8_BYTES,
    MAX_REVIEW_PROPOSAL_UTF8_BYTES,
    AnsweringPromptPathBudget,
    estimate_answering_prompt_path_budget,
    estimate_plan_prompt,
    resolve_answering_model_limits,
)
from app.infrastructure.model_calls.query_retry import configured_query_model_retry_policy
from app.rag.query.evidence_service import prepare_reranked_evidence
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger, step_log


DEFAULT_MODEL_CONTEXT_LIMIT = 8_192
DEFAULT_RESERVED_OUTPUT_TOKENS = ANSWERING_OUTPUT_TOKEN_CAP
DEFAULT_SAFETY_MARGIN_TOKENS = 256
DEFAULT_EVIDENCE_CHARACTER_BUDGET = 80_000
DEFAULT_ANSWER_RUN_TOKEN_BUDGET = 12_000_000


def _frozen_llm_model_id(state: dict[str, Any]) -> str:
    model_id = str(state.get("llm_model_id") or settings.llm_default_model or "").strip()
    if not model_id:
        raise ValueError("answering budget requires a frozen llm_model_id")
    state["llm_model_id"] = model_id
    return model_id


def _counter(state: dict[str, Any]):
    """解析与冻结回答模型绑定的 tokenizer；未知模型使用 byte 上界。"""

    model_id = _frozen_llm_model_id(state)
    try:
        encoding = tiktoken.encoding_for_model(model_id)
        return TiktokenTokenCounter(encoding_name=encoding.name)
    except Exception:
        logger.warning(
            "回答模型没有可验证的本地 tokenizer 绑定，ContextPack 使用 UTF-8 字节上界: %s",
            model_id,
        )
        return Utf8ByteUpperBoundCounter()


def _required_subquery_ids(state: dict[str, Any]) -> tuple[str, ...]:
    plan = state.get("query_plan") if isinstance(state.get("query_plan"), dict) else {}
    raw = plan.get("subqueries") or plan.get("sub_queries") or []
    result: list[str] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            if str(item.get("purpose") or "retrieval") != "retrieval":
                continue
            identifier = item.get("sub_query_id") or item.get("subquery_id") or item.get("id")
        else:
            identifier = None
        result.append(str(identifier or f"sq_{index}"))
    return tuple(dict.fromkeys(result))


def _facet_ledger(state: dict[str, Any]) -> dict[str, Any] | None:
    """兼容读取新旧 checkpoint 中的 Facet Ledger。"""

    raw = state.get("query_facet_ledger")
    if isinstance(raw, dict) and raw:
        return raw
    plan = state.get("query_plan") if isinstance(state.get("query_plan"), dict) else {}
    nested = plan.get("facet_ledger")
    return nested if isinstance(nested, dict) and nested else None


def _limits(state: dict[str, Any]) -> tuple[int, int, int, int]:
    profile = state.get("retrieval_profile")
    profile = profile if isinstance(profile, dict) else {}
    config = profile.get("context_pack")
    config = config if isinstance(config, dict) else {}
    configured_model_limit = int(config.get("model_context_limit") or DEFAULT_MODEL_CONTEXT_LIMIT)
    model_limit, _, _, _ = resolve_answering_model_limits(
        _frozen_llm_model_id(state),
        configured_context_limit=configured_model_limit,
    )
    configured_output = int(config.get("reserved_output_tokens") or DEFAULT_RESERVED_OUTPUT_TOKENS)
    # 旧 retrieval profile 的 1,024 不能承载 8-claim strict JSON。冻结预算时
    # 只允许安全抬升，绝不让 profile 把 provider cap 压低。
    reserved_output = max(ANSWERING_OUTPUT_TOKEN_CAP, configured_output)
    safety_margin = int(config.get("safety_margin_tokens") or DEFAULT_SAFETY_MARGIN_TOKENS)
    character_budget = int(
        config.get("evidence_character_budget") or DEFAULT_EVIDENCE_CHARACTER_BUDGET
    )
    return model_limit, reserved_output, safety_margin, character_budget


def _configured_model_context_limit(state: dict[str, Any]) -> int:
    profile = state.get("retrieval_profile")
    profile = profile if isinstance(profile, dict) else {}
    config = profile.get("context_pack")
    config = config if isinstance(config, dict) else {}
    return int(config.get("model_context_limit") or DEFAULT_MODEL_CONTEXT_LIMIT)


def _model_limit_binding(state: dict[str, Any]) -> tuple[int, int, str, str, int]:
    configured_limit = _configured_model_context_limit(state)
    resolved = resolve_answering_model_limits(
        _frozen_llm_model_id(state),
        configured_context_limit=configured_limit,
    )
    return (*resolved, configured_limit)


def _answer_run_token_budget(state: dict[str, Any]) -> int:
    """读取独立回答运行预算；不得复用旧 QueryPlan 检索准入预算。"""

    profile = state.get("retrieval_profile")
    profile = profile if isinstance(profile, dict) else {}
    config = profile.get("context_pack")
    config = config if isinstance(config, dict) else {}
    raw = config.get("answer_run_token_budget", DEFAULT_ANSWER_RUN_TOKEN_BUDGET)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        return 0
    return raw


def frozen_context_output_token_cap(state: dict[str, Any]) -> int:
    """返回与 ContextPack 输入预算同源的冻结输出上限。"""

    context_pack = state.get("context_pack")
    context_pack = context_pack if isinstance(context_pack, dict) else {}
    budget = context_pack.get("budget")
    budget = budget if isinstance(budget, dict) else {}
    raw = budget.get("reserved_output")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= ANSWERING_OUTPUT_TOKEN_CAP:
        return ANSWERING_OUTPUT_TOKEN_CAP
    raise ValueError("ContextPack did not freeze the required answering output reservation")


def _fixed_plan_prompt_tokens(state: dict[str, Any], counter: Any) -> int:
    """用真实 v2 Prompt 模板计算不含 Evidence 的固定输入预算。"""

    question_contract = state.get("question_contract")
    if not isinstance(question_contract, dict) or not question_contract:
        raise ValueError("ContextPack requires a frozen QuestionContract")
    return estimate_plan_prompt(
        question_contract,
        [],
        witness_catalog_receipt={
            "witness_ids": [],
            "witness_count": 0,
            "universe_status": "indeterminate",
            "omissions": [{"code": "builder_internal_error", "count": 1}],
        },
        token_counter=counter.count,
    )


def _result_prompt_path_budget(
    state: dict[str, Any],
    result: Any,
    counter: Any,
) -> AnsweringPromptPathBudget:
    """从真实 ContextPack 重建 Witness 视图，计算完整最坏调用路径。"""

    pack_dump = result.pack.model_dump(mode="json")
    packed_text_by_label = {block.label: block.text for block in result.pack.packed_blocks}
    aliases = [
        {**item, "packed_text": packed_text_by_label.get(item["label"], "")}
        for item in result.citation_aliases
    ]
    snapshot = {
        **state,
        "context_pack": pack_dump,
        "context_aliases": aliases,
    }
    question_contract = dict(state.get("question_contract") or {})
    bundle = build_witness_catalog_bundle(
        snapshot,
        QuestionContract.from_dict(question_contract),
    )
    views = resolve_witness_prompt_views(snapshot, bundle)
    return estimate_answering_prompt_path_budget(
        question_contract,
        views,
        witness_catalog_receipt=bundle["receipt"],
        token_counter=counter.count,
    )


def _pack_summary(pack: dict[str, Any], *, prompt_tokens: int) -> dict[str, Any]:
    budget = dict(pack.get("budget") or {})
    diagnostics = dict(pack.get("diagnostics") or {})
    coverage = dict(pack.get("coverage") or {})
    facet_coverage = dict(coverage.get("facet_coverage") or {})
    return {
        "schema_version": pack.get("schema_version"),
        "content_hash": pack.get("content_hash"),
        "tokenizer_id": pack.get("tokenizer_id"),
        "tokenizer_version": pack.get("tokenizer_version"),
        "strategy_version": pack.get("strategy_version"),
        "budget": budget,
        "used_tokens": int(pack.get("used_tokens") or 0),
        "used_characters": int(pack.get("used_characters") or 0),
        "whole_prompt_tokens": prompt_tokens,
        "packed_count": diagnostics.get("packed_count", 0),
        "omitted_count": diagnostics.get("omitted_count", 0),
        "truncated_count": diagnostics.get("truncated_count", 0),
        "token_utilization": diagnostics.get("token_utilization", 0.0),
        "expansion_status": diagnostics.get("expansion_status"),
        "facet_count": diagnostics.get("facet_count", 0),
        "required_facet_count": diagnostics.get("required_facet_count", 0),
        "supported_facet_count": diagnostics.get("supported_facet_count", 0),
        "branch_bound_facet_count": diagnostics.get("branch_bound_facet_count", 0),
        "uncovered_required_facet_count": diagnostics.get("uncovered_required_facet_count", 0),
        "coverage": coverage,
        "facet_coverage": facet_coverage,
        "omitted": [
            {
                "evidence_id": item.get("evidence_id"),
                "reason": item.get("reason"),
                "detail": item.get("detail", ""),
            }
            for item in (pack.get("omitted") or [])
            if isinstance(item, dict)
        ],
    }


def _prompt_budget_receipt(
    *,
    pack: dict[str, Any],
    llm_model_id: str,
    path_budget: AnsweringPromptPathBudget,
    model_limit: int,
    configured_model_limit: int,
    provider_max_input_tokens: int,
    model_limits_version: str,
    model_limits_binding_mode: str,
    reserved_output: int,
    safety_margin: int,
    provider_attempts_per_logical_call: int,
    answer_run_token_budget: int,
    planning_token_budget_diagnostic: int,
    planning_execution_tokens_diagnostic: int,
    blocked: bool,
) -> dict[str, Any]:
    """生成不含问题/Evidence 正文的可重放预算回执。"""

    max_input_tokens = min(
        model_limit - reserved_output - safety_margin,
        provider_max_input_tokens - safety_margin,
    )
    if blocked:
        status = "blocked"
    elif path_budget.model_calls_required:
        status = "ready"
    else:
        status = "not_applicable"
    max_provider_attempts = (
        ANSWERING_MAX_LOGICAL_CALLS * provider_attempts_per_logical_call
        if path_budget.model_calls_required
        else 0
    )
    provider_attempt_input_cap = path_budget.worst_input_tokens + safety_margin
    max_total_input_tokens = max_provider_attempts * provider_attempt_input_cap
    max_total_output_tokens = max_provider_attempts * ANSWERING_OUTPUT_TOKEN_CAP
    max_total_tokens = max_total_input_tokens + max_total_output_tokens
    return {
        "schema_version": ANSWERING_PROMPT_BUDGET_SCHEMA_VERSION,
        "status": status,
        "context_pack_hash": pack.get("content_hash"),
        "llm_model_id": llm_model_id,
        "tokenizer_id": pack.get("tokenizer_id"),
        "tokenizer_version": pack.get("tokenizer_version"),
        "tokenizer_binding_mode": (
            "model_exact"
            if str(pack.get("tokenizer_id") or "").startswith("tiktoken:")
            else "utf8_byte_upper_bound"
        ),
        "model_context_limit": model_limit,
        "configured_model_context_limit": configured_model_limit,
        "provider_max_input_tokens": provider_max_input_tokens,
        "model_limits_version": model_limits_version,
        "model_limits_binding_mode": model_limits_binding_mode,
        "reserved_output_tokens": reserved_output,
        "output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "safety_margin_tokens": safety_margin,
        "max_input_tokens": max_input_tokens,
        "model_calls_required": path_budget.model_calls_required,
        "path_input_tokens": path_budget.path_tokens,
        "worst_path": path_budget.worst_path,
        "worst_input_tokens": path_budget.worst_input_tokens,
        "max_logical_calls": ANSWERING_MAX_LOGICAL_CALLS,
        "max_provider_attempts_per_logical_call": provider_attempts_per_logical_call,
        "max_provider_attempts": max_provider_attempts,
        "provider_attempt_input_token_cap": provider_attempt_input_cap,
        "provider_attempt_output_token_cap": ANSWERING_OUTPUT_TOKEN_CAP,
        "max_total_input_tokens": max_total_input_tokens,
        "max_total_output_tokens": max_total_output_tokens,
        "max_total_tokens": max_total_tokens,
        "answer_run_token_budget": answer_run_token_budget,
        "answer_run_tokens_available": answer_run_token_budget,
        "planning_token_budget_diagnostic": planning_token_budget_diagnostic,
        "planning_execution_tokens_diagnostic": planning_execution_tokens_diagnostic,
        "cost_budget_status": "unavailable",
        "cost_limit": None,
        "cost_currency": None,
        "cost_budget_reason": (
            "no versioned model price snapshot and no explicit monetary cost limit"
        ),
        "max_plan_proposal_utf8_bytes": MAX_PLAN_PROPOSAL_UTF8_BYTES,
        "max_review_proposal_utf8_bytes": MAX_REVIEW_PROPOSAL_UTF8_BYTES,
    }


def _query_token_budget(state: dict[str, Any]) -> tuple[int, int]:
    """读取服务端 QueryPlan 总 token 预算及检索阶段已记用量。"""

    plan = state.get("query_plan")
    plan = plan if isinstance(plan, dict) else {}
    raw_budget = plan.get("token_budget")
    if not isinstance(raw_budget, int) or isinstance(raw_budget, bool) or raw_budget < 0:
        return 0, 0
    summary = state.get("query_execution_summary")
    summary = summary if isinstance(summary, dict) else {}
    usage = summary.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (input_tokens, output_tokens)
    ):
        return raw_budget, raw_budget
    return raw_budget, input_tokens + output_tokens


def _answer_model_ledger(receipt: dict[str, Any]) -> dict[str, Any]:
    """建立跨四次逻辑调用持久化的 provider attempt 账本。"""

    return {
        "schema_version": "answer-model-budget-ledger.v1",
        "status": receipt["status"],
        "context_pack_hash": receipt["context_pack_hash"],
        "llm_model_id": receipt["llm_model_id"],
        "max_logical_calls": receipt["max_logical_calls"],
        "max_provider_attempts_per_logical_call": receipt["max_provider_attempts_per_logical_call"],
        "max_provider_attempts": receipt["max_provider_attempts"],
        "provider_attempt_input_token_cap": receipt["provider_attempt_input_token_cap"],
        "provider_attempt_output_token_cap": receipt["provider_attempt_output_token_cap"],
        "max_total_tokens": receipt["max_total_tokens"],
        "logical_call_count": 0,
        "provider_attempt_count": 0,
        "reserved_input_tokens": 0,
        "reserved_output_tokens": 0,
        "accounted_input_tokens": 0,
        "accounted_output_tokens": 0,
        "observed_input_tokens": 0,
        "observed_output_tokens": 0,
        "usage_metadata_complete": True,
        "logical_calls": [],
        "provider_attempts": [],
        "cost_budget_status": receipt["cost_budget_status"],
        "cost_limit": receipt["cost_limit"],
        "cost_currency": receipt["cost_currency"],
    }


@step_log("prepare_query_context")
def prepare_query_context(state: dict[str, Any]) -> dict[str, Any]:
    """Evidence 二次授权 → ContextPack → 完整 Prompt 硬预算复核。"""

    consume_authorized_scope(state, "context")
    raw_documents = list(state.get("reranked_docs") or [])
    # ``reranked_docs`` 会在本函数末尾收窄为真正进入 ContextPack 的别名文档。
    # 在覆盖前保存 reranker 的原始 Top-K，才能区分“重排未召回”和“装包时省略/
    # 截断”两类完全不同的问题。只复制映射容器，不改动 Evidence 正文或身份。
    state["prepack_reranked_docs"] = [dict(item) for item in raw_documents]
    state["prepack_candidate_count"] = len(raw_documents)
    documents = prepare_reranked_evidence(
        raw_documents,
        kb_id=state.get("kb_id"),
    )
    state["authorized_candidate_count"] = len(documents)
    documents = secure_evidence_documents(state, documents)
    state["security_approved_candidate_count"] = len(documents)

    counter = _counter(state)
    model_limit, reserved_output, safety_margin, character_budget = _limits(state)
    (
        bound_context_window,
        provider_max_input_tokens,
        model_limits_version,
        model_limits_binding_mode,
        configured_model_limit,
    ) = _model_limit_binding(state)
    if bound_context_window != model_limit:
        raise ValueError("answering model context binding drifted during assembly")
    fixed_tokens = _fixed_plan_prompt_tokens(state, counter)
    max_input_tokens = min(
        model_limit - reserved_output - safety_margin,
        provider_max_input_tokens - safety_margin,
    )
    available = max_input_tokens - fixed_tokens
    # 固定 Prompt 已占满窗口时先构造空 ContextPack；后续预算回执会在任何
    # provider 调用前给出唯一、可审计的 fail-closed 结论。
    evidence_budget = max(0, available)
    accounted_fixed_tokens = min(
        fixed_tokens,
        max(0, max_input_tokens),
    )
    budget = ContextBudget.from_window(
        model_context_limit=model_limit,
        reserved_system=accounted_fixed_tokens,
        reserved_output=reserved_output,
        safety_margin=safety_margin,
        evidence_budget=evidence_budget,
        evidence_character_budget=character_budget,
    )
    profile = state.get("retrieval_profile")
    profile = profile if isinstance(profile, dict) else {}
    pack_profile = profile.get("context_pack")
    pack_profile = pack_profile if isinstance(pack_profile, dict) else {}
    policy = ContextPackPolicy(
        max_blocks=int(pack_profile.get("max_blocks") or 20),
        max_blocks_per_document=int(pack_profile.get("max_blocks_per_document") or 5),
        max_blocks_per_source=int(pack_profile.get("max_blocks_per_source") or 8),
        expansion_mode=str(pack_profile.get("expansion_mode") or "none"),
        neighbor_window=int(pack_profile.get("neighbor_window") or 1),
    )
    if policy.expansion_mode in {"parent", "parent_and_neighbors"}:
        consume_authorized_scope(state, "parent")
    if policy.expansion_mode in {"neighbors", "parent_and_neighbors"}:
        consume_authorized_scope(state, "neighbor")
    result = build_context_pack(
        documents,
        query_run_id=str(state.get("query_run_id") or "qry_legacy"),
        budget=budget,
        counter=counter,
        policy=policy,
        required_sub_query_ids=_required_subquery_ids(state),
        facet_ledger=_facet_ledger(state),
    )
    path_budget = _result_prompt_path_budget(state, result, counter)
    prompt_tokens = path_budget.worst_input_tokens

    # 首次装包只知道 initial plan 的固定开销。这里按四条最坏路径只允许一次
    # 确定性重包；不循环、不摘要、不让 LLM 改写 Evidence。
    if (
        path_budget.model_calls_required
        and prompt_tokens > max_input_tokens
        and result.pack.used_tokens
    ):
        overflow = prompt_tokens - max_input_tokens
        second_budget = max(0, result.pack.budget.evidence_budget - overflow - 8)
        budget = ContextBudget.from_window(
            model_context_limit=model_limit,
            reserved_system=accounted_fixed_tokens,
            reserved_output=reserved_output,
            safety_margin=safety_margin,
            evidence_budget=second_budget,
            evidence_character_budget=character_budget,
        )
        result = build_context_pack(
            documents,
            query_run_id=str(state.get("query_run_id") or "qry_legacy"),
            budget=budget,
            counter=counter,
            policy=policy,
            required_sub_query_ids=_required_subquery_ids(state),
            facet_ledger=_facet_ledger(state),
        )
        path_budget = _result_prompt_path_budget(state, result, counter)
        prompt_tokens = path_budget.worst_input_tokens

    retry_policy = configured_query_model_retry_policy()
    planning_token_budget, planning_execution_tokens = _query_token_budget(state)
    answer_run_token_budget = _answer_run_token_budget(state)
    max_provider_attempts = ANSWERING_MAX_LOGICAL_CALLS * retry_policy.max_attempts
    max_answering_tokens = max_provider_attempts * (
        path_budget.worst_input_tokens + safety_margin + ANSWERING_OUTPUT_TOKEN_CAP
    )
    path_blocked = path_budget.model_calls_required and prompt_tokens > max_input_tokens
    run_token_blocked = (
        path_budget.model_calls_required and max_answering_tokens > answer_run_token_budget
    )
    blocked = path_blocked or run_token_blocked
    if path_blocked:
        # 即使重包后仍越界，initial plan 也不得调用，避免付费后才发现 review
        # 或任一有界 repair 路径必然超窗。
        state["context_budget_blocked"] = True
        state["context_budget_reason"] = "WORST_ANSWERING_PATH_EXCEEDS_MODEL_WINDOW"
    elif run_token_blocked:
        state["context_budget_blocked"] = True
        state["context_budget_reason"] = "ANSWERING_RUN_TOKEN_BUDGET_EXCEEDED"
    else:
        state["context_budget_blocked"] = False
        state["context_budget_reason"] = ""

    pack_dump = result.pack.model_dump(mode="json")
    state["context_pack"] = pack_dump
    prompt_budget_receipt = _prompt_budget_receipt(
        pack=pack_dump,
        llm_model_id=_frozen_llm_model_id(state),
        path_budget=path_budget,
        model_limit=model_limit,
        configured_model_limit=configured_model_limit,
        provider_max_input_tokens=provider_max_input_tokens,
        model_limits_version=model_limits_version,
        model_limits_binding_mode=model_limits_binding_mode,
        reserved_output=reserved_output,
        safety_margin=safety_margin,
        provider_attempts_per_logical_call=retry_policy.max_attempts,
        answer_run_token_budget=answer_run_token_budget,
        planning_token_budget_diagnostic=planning_token_budget,
        planning_execution_tokens_diagnostic=planning_execution_tokens,
        blocked=blocked,
    )
    state["answer_prompt_budget"] = prompt_budget_receipt
    state["answer_model_ledger"] = _answer_model_ledger(prompt_budget_receipt)
    state["query_facet_ledger"] = dict(pack_dump.get("facet_ledger") or {})
    state["context_pack_summary"] = _pack_summary(
        pack_dump,
        prompt_tokens=prompt_tokens,
    )
    packed_text_by_label = {block.label: block.text for block in result.pack.packed_blocks}
    state["context_aliases"] = [
        {
            **item,
            "packed_text": packed_text_by_label.get(item["label"], ""),
        }
        for item in result.citation_aliases
    ]
    state["evidences"] = [item["evidence"] for item in result.citation_aliases]
    state["evidence_aliases"] = {
        item["label"]: item["evidence"]["evidence_id"] for item in result.citation_aliases
    }
    # 后续图片、citation 与 AnswerEnvelope 只能看到真正进入 ContextPack 的块。
    state["reranked_docs"] = [dict(item["document"]) for item in result.citation_aliases]
    return state


__all__ = [
    "DEFAULT_MODEL_CONTEXT_LIMIT",
    "frozen_context_output_token_cap",
    "prepare_query_context",
]
