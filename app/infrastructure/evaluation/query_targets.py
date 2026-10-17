"""把生产查询能力投影为稳定评测观测的基础设施适配器。"""

from __future__ import annotations

import hashlib
import re
import time
from copy import deepcopy

from app.application.evaluation_target import EvaluationRequest, EvaluationTargetResult
from app.domain.evaluation import (
    FORMAL_EVALUATION_SNAPSHOT_KEY,
    freeze_formal_evaluation_contract,
)
from app.domain.evaluation_proof import (
    OBSERVATION_SCHEMA_V3,
    build_proof_evaluation_authority,
    project_proof_observation,
)


_DATA_CLASSIFICATION_RANK = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}
_SAFE_RECEIPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,511}$")


def _normalized_classification(value: object, *, default: str) -> str:
    raw = str(value or default).strip().lower()
    return raw if raw in _DATA_CLASSIFICATION_RANK else "restricted"


def _effective_classification(kb_value: object, case_value: object) -> str:
    """返回知识库分类与用例敏感度中限制更严格的一项。"""

    candidates = (
        _normalized_classification(kb_value, default="restricted"),
        _normalized_classification(case_value, default="internal"),
    )
    return max(candidates, key=_DATA_CLASSIFICATION_RANK.__getitem__)


def _security_refusal_observation(
    *,
    violation,
    answer_id: str,
    session_id: str,
    latency_ms: float,
    data_classification: str,
    config: dict,
) -> dict:
    """把输入安全阻断投影为不含原始问题的 ``security`` 观测。"""

    security = {
        "status": "blocked",
        **violation.audit_fields(),
        "data_classification": data_classification,
    }
    return {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "answer_id": answer_id,
        "session_id": session_id,
        "query": "",
        "rewritten_query": None,
        "questions": [],
        "keywords": "",
        "answer": violation.safe_message,
        "answer_envelope": {},
        "claims": [],
        "citations": [],
        "sources": [],
        "citation_validation": {},
        "evidence_assessment": {},
        "answer_decision": {
            "action": "REFUSE",
            "reason_code": violation.code,
            "policy_version": violation.policy_version,
        },
        "grounding": {},
        **project_proof_observation({}),
        "abstained": True,
        "security": security,
        "retrieved": [],
        "routes": {"embedding": [], "hyde": [], "web": []},
        "rrf": [],
        "reranked": [],
        "context": [],
        "query_understanding": {},
        "query_plan": {},
        "retrieval_diagnostics": {
            "status": "not_attempted",
            "reason_code": violation.code,
        },
        "context_pack": {},
        "context_pack_summary": {},
        "latency_ms": round(latency_ms, 3),
        "token_usage": {
            "status": "not_attempted",
            "reason": "input_security_blocked",
        },
        "cost": None,
        "cost_currency": None,
        "versions": {
            "code_revision": config.get("code_revision"),
            "config_version": config.get("config_version"),
            "dataset_version": config.get("dataset_version"),
            "dataset_checksum": config.get("dataset_checksum"),
            "embedding_model": config.get("embedding_model"),
            "reranker_model": config.get("reranker_model"),
            "llm_model": config.get("llm_model"),
            "index_version_ids": [],
            "document_version_ids": [],
            "prompt_sha256": None,
        },
        "side_effects": {
            "history_read": False,
            "history_write": False,
            "retrieval": False,
            "web_search": False,
            "hyde": False,
            "external_llm": False,
            "security_audit": True,
        },
    }


def _route(items: list[dict] | None, name: str) -> list[dict]:
    return [{**dict(item), "route": name} for item in (items or [])]


def _safe_candidate_receipt(value: dict) -> dict:
    """只保留评测匹配所需身份/分数，禁止归档 Evidence 正文。"""

    receipt: dict = {}
    for key in (
        "id",
        "evidence_id",
        "witness_id",
        "representation_id",
        "chunk_id",
        "document_id",
        "doc_id",
        "index_version_id",
        "document_version_id",
        "route",
    ):
        raw = value.get(key)
        if isinstance(raw, str) and _SAFE_RECEIPT_ID.fullmatch(raw):
            receipt[key] = raw
    for key in (
        "content_hash",
        "text_hash",
        "source_content_hash",
        "context_pack_hash",
    ):
        raw = value.get(key)
        if isinstance(raw, str) and raw.startswith("sha256:") and len(raw) == 71:
            receipt[key] = raw
    for key in ("rank", "score", "rerank_score", "rrf_score", "relevance_score"):
        raw = value.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            receipt[key] = raw
    return receipt


def _safe_candidate_receipts(values: list[dict]) -> list[dict]:
    return [_safe_candidate_receipt(dict(item)) for item in values]


def _frozen_vector_scope(config: dict) -> dict:
    """恢复评测运行接受时冻结的物理读集，禁止执行期漂移。"""

    rows = config.get("vector_read_scopes")
    digest = config.get("vector_read_set_digest")
    if not isinstance(rows, list) or not rows or not isinstance(digest, str) or not digest:
        raise ValueError("evaluation config is missing a frozen vector read set")
    return {
        "vector_read_scopes": deepcopy(rows),
        "vector_read_set_digest": digest,
    }


def _apply_formal_planning_admission_budgets(
    profile_config: dict,
    config: dict,
    *,
    expected_mode: str,
    expected_run_id: str,
    expected_kb_id: str,
) -> dict:
    """只投影 QueryPlan 准入上限；不声称实际 provider 用量硬限制。"""

    raw_contract = config.get(FORMAL_EVALUATION_SNAPSHOT_KEY)
    if raw_contract is None:
        return profile_config
    contract = freeze_formal_evaluation_contract(
        raw_contract,
        expected_mode=expected_mode,
        expected_run_id=expected_run_id,
        expected_kb_id=expected_kb_id,
        expected_runtime_snapshot=config,
    )
    if contract.get("contract_version") == "stt.eval.formal-run.v2":
        from app.shared.config.settings_config import settings

        authority = contract["run_authority"]
        current_runtime_flags = {
            "query_default_deadline_ms": settings.query_default_deadline_ms,
            "llm_request_timeout_seconds": settings.llm_request_timeout,
            "llm_max_attempts": settings.llm_max_attempts,
            "llm_retry_base_delay_seconds": settings.llm_retry_base_delay,
            "llm_retry_max_delay_seconds": settings.llm_retry_max_delay,
            "reranker_failure_mode": settings.bge_reranker_failure_mode,
            "vector_placement_mode": settings.vector_placement_mode,
            "vector_placement_policy_version": settings.vector_placement_policy_version,
        }
        current_model_ids = {
            "embedding": settings.bge_m3_path or settings.bge_m3,
            "reranker": settings.bge_reranker_large,
            "llm": settings.llm_default_model,
        }
        if authority["runtime_flags"] != current_runtime_flags:
            raise ValueError("formal evaluation runtime flags drifted after run acceptance")
        if authority["model_ids"] != current_model_ids:
            raise ValueError("formal evaluation model ids drifted after run acceptance")
        budgets = contract["intent"]["planning_admission_budgets"]
    else:
        # 历史 v1 run 保持可恢复；字段只解释为同一 PlanningPolicy 准入值。
        legacy = contract["budgets"]
        budgets = {
            "max_plan_tokens": legacy["token_budget"],
            "max_routes": legacy["tool_budget"],
            "max_planned_cost_units": legacy["cost_budget"],
        }
    bounded = deepcopy(profile_config)
    planning = dict(bounded.get("query_planning") or {})
    planning["token_budget"] = min(
        int(planning.get("token_budget", 4096)),
        int(budgets["max_plan_tokens"]),
    )
    # max_routes 只限制 QueryPlan route 数，不等价于实际 RetrievalPort 调用数。
    planning["max_routes"] = min(
        int(planning.get("max_routes", 3)),
        int(budgets["max_routes"]),
    )
    planning["cost_budget"] = min(
        float(planning.get("cost_budget", 1.0)),
        float(budgets["max_planned_cost_units"]),
    )
    bounded["query_planning"] = planning
    return bounded


def _packed_context(final_state: dict) -> list[dict]:
    """投影模型真正看见的 ContextPack 文本，并保留可评测的来源身份。"""

    pack = final_state.get("context_pack")
    pack = pack if isinstance(pack, dict) else {}
    blocks = {
        str(item.get("label") or ""): dict(item)
        for item in (pack.get("packed_blocks") or [])
        if isinstance(item, dict) and item.get("label")
    }
    context: list[dict] = []
    for raw_alias in final_state.get("context_aliases") or []:
        if not isinstance(raw_alias, dict):
            continue
        document = dict(raw_alias.get("document") or {})
        label = str(raw_alias.get("label") or "")
        block = blocks.get(label, {})
        packed_text = str(raw_alias.get("packed_text") or block.get("text") or "")
        if not document or not packed_text:
            continue
        # ``document`` 可能仍带完整 Evidence 正文；评测 context 必须改为真正送入
        # prompt 的截断后表示，否则会虚构模型看见了被预算裁掉的内容。
        document["content"] = packed_text
        document["text"] = packed_text
        document["context_label"] = label
        document["representation_id"] = block.get("representation_id")
        document["context_truncation"] = block.get("truncation")
        document["context_inclusion_reason"] = block.get("inclusion_reason")
        context.append(document)
    if context:
        return context
    # 兼容该投影引入前的测试/旧图；生产图有 ContextPack 时不会走此回退。
    return [dict(item) for item in (final_state.get("reranked_docs") or [])]


class QueryGraphEvalAdapter:
    """复用生产 Query Graph，并显式禁止历史写入与 Web 副作用。"""

    def evaluate(self, request: EvaluationRequest) -> EvaluationTargetResult:
        """实现稳定 EvaluationTarget 契约。"""

        return self._evaluate_result(
            request.target_input.as_mapping(),
            kb_id=request.kb_id,
            run_id=request.run_id,
            config=dict(request.config),
        )

    def observe(self, case: dict, *, kb_id: str, run_id: str, config: dict) -> dict:
        """兼容旧调用方；可信 sidecar 只能由 ``evaluate`` 交给评测 service。"""

        return dict(
            self._evaluate_result(
                case,
                kb_id=kb_id,
                run_id=run_id,
                config=config,
            ).observation
        )

    def _evaluate_result(
        self,
        case: dict,
        *,
        kb_id: str,
        run_id: str,
        config: dict,
    ) -> EvaluationTargetResult:
        # 惰性导入使指标与单元测试收集阶段不依赖模型和 Milvus。
        from app.application.query_security_pipeline import (
            QueryContentSecurityViolation,
        )
        from app.domain.query_security import (
            AuthorizedRetrievalScope,
            EgressPolicySnapshot,
        )
        from app.runtime.graphs import get_query_graph
        from app.process.query.agent.state import create_query_default_state
        from app.security.principal import current_principal

        correlation_id = str(
            case.get("correlation_id")
            or hashlib.sha256(f"{run_id}|{case.get('query', '')}".encode("utf-8")).hexdigest()[:32]
        )
        session_id = f"eval-{run_id}-{correlation_id}"
        identity_digest = hashlib.sha256(
            f"{run_id}|{correlation_id}|{case['query']}".encode("utf-8")
        ).hexdigest()
        retrieval_profile = dict(config.get("retrieval_profile") or {})
        profile_config = deepcopy(retrieval_profile.get("config") or {})
        planning = dict(profile_config.get("query_planning") or {})
        planning["allow_hyde"] = False
        profile_config["query_planning"] = planning
        profile_config = _apply_formal_planning_admission_budgets(
            profile_config,
            config,
            expected_mode="live",
            expected_run_id=run_id,
            expected_kb_id=kb_id,
        )
        vector_scope = _frozen_vector_scope(config)
        security = deepcopy(config.get("evaluation_security") or {})
        principal = current_principal()
        if principal is None:
            raise ValueError("live evaluation requires an authenticated principal snapshot")
        try:
            authorized_scope = AuthorizedRetrievalScope.model_validate(
                security.get("authorized_scope") or {}
            )
            llm_policy = EgressPolicySnapshot.model_validate(
                (security.get("egress_policies") or {}).get("llm_generation") or {}
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("live evaluation security snapshot is invalid") from exc
        if (
            not security.get("scope_authorized")
            or authorized_scope.tenant_id != principal.tenant_id
            or authorized_scope.principal_id != principal.user_id
            or authorized_scope.allowed_kb_ids != (kb_id,)
        ):
            raise ValueError("live evaluation security scope does not match the executor")
        authorized_indexes = {
            (
                item.kb_id,
                item.document_id,
                item.document_version_id,
                item.index_version_id,
            )
            for item in authorized_scope.active_indexes
        }
        frozen_indexes = {
            (
                str(row.get("kb_id") or ""),
                str(row.get("document_id") or ""),
                str(row.get("document_version_id") or ""),
                str(row.get("index_version_id") or ""),
            )
            for row in vector_scope["vector_read_scopes"]
            if str(row.get("tenant_id") or "") == principal.tenant_id
        }
        if not authorized_indexes or frozen_indexes != authorized_indexes:
            raise ValueError("live evaluation vector read set is outside the authorized scope")
        if not security.get("external_llm_allowed") or not llm_policy.enabled:
            raise ValueError("live evaluation answer generation is not authorized")
        data_classification = _effective_classification(
            security.get("data_classification"),
            case.get("sensitivity"),
        )
        state = create_query_default_state(
            session_id=session_id,
            original_query=str(case["query"]),
            kb_id=kb_id,
            web_search_allowed=False,
            is_stream=False,
            evaluation_mode=True,
            answer_id=f"ans_{identity_digest[:32]}",
            query_run_id=f"qry_{identity_digest[32:64]}",
            trace_id=f"qry_{identity_digest[32:64]}",
            history=deepcopy(case.get("conversation") or []),
            config_version=str(config.get("config_version") or "cfg_unknown"),
            llm_model_id=str(config.get("llm_model") or ""),
            retrieval_profile=profile_config,
            retrieval_profile_id=retrieval_profile.get("profile_id") or "",
            retrieval_profile_version=int(retrieval_profile.get("version") or 0),
            retrieval_profile_hash=retrieval_profile.get("config_hash") or "",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            authorized_scope=authorized_scope.model_dump(mode="json"),
            data_classification=data_classification,
            external_llm_allowed=True,
            egress_policies=deepcopy(security.get("egress_policies") or {}),
            content_security_policy=deepcopy(security.get("content_security_policy") or {}),
            history_snapshot_loaded=True,
            **vector_scope,
        )
        started = time.perf_counter()
        try:
            final_state = get_query_graph().invoke(state)
        except QueryContentSecurityViolation as exc:
            # LangGraph 可能复制输入 state；本地引用也立即去除原始问题，避免
            # 后续异常处理或调试代码意外保留受限正文。
            state["original_query"] = ""
            return EvaluationTargetResult(
                observation=_security_refusal_observation(
                    violation=exc,
                    answer_id=f"ans_{identity_digest[:32]}",
                    session_id=session_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    data_classification=data_classification,
                    config=config,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1000

        embedding = _route(final_state.get("embedding_chunks"), "embedding")
        hyde = _route(final_state.get("hyde_embedding_chunks"), "hyde")
        web = _route(final_state.get("web_search_docs"), "web")
        retrieved = embedding + hyde + web
        rrf = [dict(item) for item in (final_state.get("rrf_chunks") or [])]
        reranked = [
            dict(item)
            for item in (
                final_state.get("prepack_reranked_docs") or final_state.get("reranked_docs") or []
            )
        ]
        context = _packed_context(final_state)
        prompt = str(final_state.get("prompt") or "")
        provenance = reranked or rrf or retrieved
        versions = {
            "code_revision": config.get("code_revision"),
            "config_version": final_state.get("config_version"),
            "dataset_version": config.get("dataset_version"),
            "dataset_checksum": config.get("dataset_checksum"),
            "embedding_model": config.get("embedding_model"),
            "reranker_model": config.get("reranker_model"),
            "llm_model": config.get("llm_model"),
            "index_version_ids": sorted(
                {
                    str(item.get("index_version_id"))
                    for item in provenance
                    if item.get("index_version_id")
                }
            ),
            "document_version_ids": sorted(
                {
                    str(item.get("document_version_id"))
                    for item in provenance
                    if item.get("document_version_id")
                }
            ),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None,
        }
        decision = dict(final_state.get("answer_decision") or {})
        proof_projection = project_proof_observation(final_state)
        safe_embedding = _safe_candidate_receipts(embedding)
        safe_hyde = _safe_candidate_receipts(hyde)
        safe_web = _safe_candidate_receipts(web)
        observation = {
            "observation_schema": OBSERVATION_SCHEMA_V3,
            "answer_id": final_state.get("answer_id") or f"ans_{identity_digest[:32]}",
            "session_id": session_id,
            "query": "",
            "query_sha256": hashlib.sha256(str(case["query"]).encode("utf-8")).hexdigest(),
            "rewritten_query": None,
            "questions": [],
            "keywords": "",
            "answer": final_state.get("answer") or "",
            "answer_envelope": {},
            "claims": [],
            "citations": [],
            "sources": [],
            "citation_validation": {},
            "evidence_assessment": {},
            "answer_decision": decision,
            "grounding": {},
            **proof_projection,
            "abstained": decision.get("action") in {"REFUSE", "CLARIFY"},
            "retrieved": _safe_candidate_receipts(retrieved),
            "routes": {
                "embedding": safe_embedding,
                "hyde": safe_hyde,
                "web": safe_web,
            },
            "rrf": _safe_candidate_receipts(rrf),
            "reranked": _safe_candidate_receipts(reranked),
            "context": _safe_candidate_receipts(context),
            "query_understanding": {},
            "query_plan": {},
            "retrieval_diagnostics": {},
            "context_pack": {},
            "context_pack_summary": {},
            "latency_ms": round(latency_ms, 3),
            "token_usage": final_state.get("usage")
            or {
                "status": "unavailable",
                "reason": "query graph usage collector is not installed",
            },
            "cost": None,
            "cost_currency": None,
            "versions": versions,
            "side_effects": {
                "history_read": False,
                "history_write": False,
                "web_search": False,
                "hyde": False,
                "external_llm": True,
            },
        }
        return EvaluationTargetResult(
            observation=observation,
            proof_authority=build_proof_evaluation_authority(final_state),
        )


class RetrievalOnlyEvalAdapter:
    """运行真实本地 embedding/Milvus 路径，但不调用外部 LLM。

    该目标只生成可信的检索与延迟观测；生成、引用、token 与成本指标保持未知。
    它适用于受限环境，不得被表述为完整查询质量基线。
    """

    def evaluate(self, request: EvaluationRequest) -> dict:
        """实现稳定 EvaluationTarget 契约。"""

        return self.observe(
            request.target_input.as_mapping(),
            kb_id=request.kb_id,
            run_id=request.run_id,
            config=dict(request.config),
        )

    def observe(self, case: dict, *, kb_id: str, run_id: str, config: dict) -> dict:
        from app.domain.query_planning import PlanningPolicy
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.embedding_search_service import search_by_embedding
        from app.rag.query.query_planning_service import (
            build_query_plan_from_state,
            query_plan_state_patch,
        )
        from app.rag.query.rerank_service import rerank_documents
        from app.rag.query.rrf_service import fuse_by_rrf

        correlation_id = str(
            case.get("correlation_id")
            or hashlib.sha256(f"{run_id}|{case.get('query', '')}".encode("utf-8")).hexdigest()[:32]
        )
        query = str(case["query"])
        retrieval_profile = dict(config.get("retrieval_profile") or {})
        vector_scope = _frozen_vector_scope(config)
        scope_rows = vector_scope["vector_read_scopes"]
        tenant_ids = {str(row.get("tenant_id") or "") for row in scope_rows}
        kb_ids = {str(row.get("kb_id") or "") for row in scope_rows}
        if len(tenant_ids) != 1 or "" in tenant_ids:
            raise ValueError("evaluation vector read set must belong to exactly one tenant")
        if kb_ids != {kb_id}:
            raise ValueError(
                "evaluation vector read set does not match the requested knowledge base"
            )
        profile_config = dict(retrieval_profile.get("config") or {})
        profile_config = _apply_formal_planning_admission_budgets(
            profile_config,
            config,
            expected_mode="retrieval_only",
            expected_run_id=run_id,
            expected_kb_id=kb_id,
        )
        planning = dict(profile_config.get("query_planning") or {})
        policy = PlanningPolicy(
            # retrieval_only 的边界是“无外部 LLM”：即使发布 Profile 允许 HyDE，
            # 离线检索评测也只执行确定性 planner 批准的本地混合检索子问题。
            allow_hyde=False,
            allow_web=False,
            max_subqueries=int(planning.get("max_subqueries", 4)),
            max_dependency_depth=int(planning.get("max_dependency_depth", 2)),
            max_routes=int(planning.get("max_routes", 3)),
            total_timeout_ms=max(
                250,
                min(120_000, int(profile_config.get("timeout_ms") or 8_000)),
            ),
            default_top_k=max(
                1,
                min(100, int(profile_config.get("top_k") or 8)),
            ),
            max_top_k=max(
                1,
                min(1_000, int(profile_config.get("candidate_k") or 40)),
            ),
            token_budget=int(planning.get("token_budget", 4096)),
            cost_budget=float(planning.get("cost_budget", 1.0)),
        )
        if policy.max_top_k < policy.default_top_k:
            policy = policy.model_copy(update={"max_top_k": policy.default_top_k})
        state = create_query_default_state(
            session_id=f"eval-retrieval-{run_id}-{correlation_id}",
            original_query=query,
            rewritten_query=query,
            questions=[query],
            kb_id=kb_id,
            web_search_allowed=False,
            evaluation_mode=True,
            is_stream=False,
            history=deepcopy(case.get("conversation") or []),
            retrieval_profile=profile_config,
            retrieval_profile_id=retrieval_profile.get("profile_id") or "",
            retrieval_profile_version=int(retrieval_profile.get("version") or 0),
            retrieval_profile_hash=retrieval_profile.get("config_hash") or "",
            **vector_scope,
        )
        started = time.perf_counter()
        scope_seed = f"{next(iter(tenant_ids))}|{kb_id}|{vector_scope['vector_read_set_digest']}"
        plan = build_query_plan_from_state(
            state,
            tenant_id=next(iter(tenant_ids)),
            scope_id=("evalscope_" + hashlib.sha256(scope_seed.encode("utf-8")).hexdigest()[:32]),
            document_ids=tuple(
                dict.fromkeys(
                    str(row.get("document_id")) for row in scope_rows if row.get("document_id")
                )
            ),
            policy=policy,
        )
        state.update(query_plan_state_patch(plan))
        state["rewritten_query"] = plan.understanding.standalone_query
        state["questions"] = [plan.understanding.standalone_query]
        # search_by_embedding 会按 QueryPlan 的 local_hybrid.sub_query_ids 逐个
        # 调用真实 RetrievalPort；无检索 disposition 时不得回退为整句搜索。
        retrieved = search_by_embedding(state) if plan.disposition.value == "retrieve" else []
        # 与生产查询图保持同一条本地检索路径：基础混合召回之后仍需经过
        # 统一融合策略与本地 reranker。禁止把未经重排的 search rows 同时
        # 填进 retrieved/reranked/context，否则评测报告会虚构模型已生效，
        # profile 的 top_k、去重和 max_per_document 也不会真正接受验证。
        state["embedding_chunks"] = retrieved
        state["hyde_embedding_chunks"] = []
        state["web_search_docs"] = []
        fused = fuse_by_rrf(state)
        state["rrf_chunks"] = fused
        reranked = rerank_documents(state)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "observation_schema": "stt.eval.observation.v1",
            "answer_id": f"ans-{run_id}-{correlation_id}",
            "session_id": state["session_id"],
            "query": query,
            "rewritten_query": plan.understanding.standalone_query,
            "query_understanding": state["query_understanding"],
            "query_plan": state["query_plan"],
            "answer": "",
            "visible_answer_policy": {},
            "retrieved": retrieved,
            "routes": {"embedding": retrieved, "hyde": [], "web": []},
            "rrf": fused,
            "reranked": reranked,
            "context": reranked,
            "retrieval_diagnostics": state.get("retrieval_diagnostics") or {},
            "latency_ms": round(latency_ms, 3),
            "token_usage": None,
            "cost": None,
            "cost_currency": None,
            "versions": config,
            "side_effects": {
                "history_read": False,
                "history_write": False,
                "web_search": False,
                "external_llm": False,
            },
            "scope": "retrieval_only",
        }


# 新名称表达架构角色；旧名称暂时保留，避免改变 application service 行为。
QueryGraphEvaluationTarget = QueryGraphEvalAdapter
RetrievalOnlyEvaluationTarget = RetrievalOnlyEvalAdapter
