import math

from app.application.retrieval_fusion import (
    CalibrationConfig,
    FusionConfig,
    fuse_route_results,
    fusion_result_to_mappings,
    normalized_query_fingerprint,
)
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import step_log

# RRF 融合默认配置
# 最终返回的融合结果数量
top = 5
# RRF 公式平滑系数，避免排名过低导致分数趋近于 0
k = 60


@step_log("validate_rrf_inputs")
def validate_rrf_inputs(
    state: dict,
    *,
    allow_empty: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    校验 RRF 融合所需的两路本地检索结果是否合法
    允许单路有结果、另一路为空，但不允许两路都为空
    """
    # 获取普通向量检索结果、HyDE 增强检索结果
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_chunks = state.get("hyde_embedding_chunks", [])

    if not allow_empty and not embedding_chunks and not hyde_chunks:
        raise ValueError("普通检索与 HyDE 检索结果均为空")

    return embedding_chunks, hyde_chunks


@step_log("rrf")
def rrf(
    param_list: list[tuple[list[dict], float]],
    *,
    k: int = 60,
    top: int = 5,
) -> list[dict]:
    """
    加权RRF（倒数排名融合）核心算法实现
    不依赖原始分数，只根据排名位置计算融合得分，支持多路加权融合

    Args:
        param_list: 列表，每一项是 (检索结果列表, 权重)
        k: 平滑系数，默认60
        top: 最终返回Top N条结果

    Returns:
        按RRF融合分数倒序排列后的统一结果列表
    """
    if k <= 0:
        raise ValueError("RRF k must be greater than zero")
    if top < 0:
        raise ValueError("RRF top must be non-negative")
    for _, weight in param_list:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("RRF route weights must be finite and non-negative")

    # 存储每个 chunk_id 的总融合分数
    score_dict: dict[str, float] = {}
    # 存储每个 chunk_id 对应的原始片段信息
    entity_dict: dict[str, dict] = {}

    # 遍历每一路检索结果，计算融合分数
    for chunks_list, weight in param_list:
        if weight == 0:
            continue
        route_chunks: list[dict] = []
        route_seen: set[str] = set()
        for chunk in chunks_list:
            chunk_id = chunk.get("chunk_id")
            # 同一路同一块只计一次，避免重复返回把该路权重放大。
            if not chunk_id or chunk_id in route_seen:
                continue
            route_seen.add(chunk_id)
            route_chunks.append(chunk)
        for rank, chunk in enumerate(route_chunks, start=1):
            chunk_id = chunk["chunk_id"]
            # RRF 核心公式：1/(k + 排名) * 权重，累加到总分
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (k + rank)) * weight
            # 稳定 identity 只保留一份正文，同时合并 dense/HyDE 各路观察。
            if chunk_id not in entity_dict:
                entity_dict[chunk_id] = chunk.copy()
                entity_dict[chunk_id]["retrieval_observations"] = list(
                    chunk.get("retrieval_observations") or []
                )
            else:
                observations = entity_dict[chunk_id].setdefault("retrieval_observations", [])
                for observation in chunk.get("retrieval_observations") or []:
                    if observation not in observations:
                        observations.append(observation)

    # 组装分数与原文信息
    document_list = []
    for chunck_id, score in score_dict.items():
        document = entity_dict.get(chunck_id, {}).copy()
        document["rrf_score"] = score
        document["score"] = score
        document_list.append(document)

    # 按分数倒序排序
    document_list.sort(key=lambda x: x["score"], reverse=True)

    # 截取 Top N 条结果
    document_list = document_list[:top]

    return document_list


def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    负责将本地两路检索结果（普通向量 + HyDE）进行加权融合
    两路默认权重均为 1.0
    """
    # 1. 所有召回路线只进入一次统一融合；全空是合法业务状态。
    embedding_chunks, hyde_embedding_chunks = validate_rrf_inputs(
        state,
        allow_empty=True,
    )

    profile_config = state.get("retrieval_profile") or {}
    rrf_config = profile_config.get("rrf") or {}
    route_weights = {
        str(item.get("route")): float(item.get("weight", 1.0))
        for item in (rrf_config.get("route_weights") or [])
        if isinstance(item, dict) and item.get("route")
    }
    rrf_k = int(rrf_config.get("k", k))
    candidate_top = int(profile_config.get("candidate_k", top))

    web_documents = [
        {**document, "type": "web", "source_type": "web"}
        for document in (state.get("web_search_docs") or [])
    ]
    fusion_profile = profile_config.get("fusion") or {}
    method = str(fusion_profile.get("method") or "rrf")
    if method not in {"rrf", "weighted"}:
        method = "rrf"
    configured_weights = {
        "base": route_weights.get("base", 1.0),
        "hyde": route_weights.get("hyde", 1.0),
        "web": route_weights.get("web", 1.0),
        **{
            str(key): float(value)
            for key, value in (fusion_profile.get("route_weights") or {}).items()
        },
    }
    near_duplicate_threshold = fusion_profile.get("near_duplicate_threshold")
    # 旧 profile 没有 fusion 字段时维持精确去重；新版本可显式开启近重复。
    if "near_duplicate_threshold" not in fusion_profile:
        near_duplicate_threshold = None
    config = FusionConfig(
        method=method,  # type: ignore[arg-type]
        output_k=candidate_top,
        rrf_k=rrf_k,
        route_weights=configured_weights,
        default_calibration=CalibrationConfig.model_validate(
            fusion_profile.get("default_calibration") or {}
        ),
        route_calibrations={
            str(key): CalibrationConfig.model_validate(value)
            for key, value in (fusion_profile.get("route_calibrations") or {}).items()
        },
        near_duplicate_threshold=near_duplicate_threshold,
        max_per_document=int(fusion_profile.get("max_per_document") or candidate_top),
        source_quotas={
            str(key): int(value)
            for key, value in (fusion_profile.get("source_quotas") or {}).items()
        },
        mmr_lambda=float(fusion_profile.get("mmr_lambda", 0.8)),
        subquery_coverage_bonus_weight=float(
            fusion_profile.get("subquery_coverage_bonus_weight", 0.0)
        ),
        subquery_coverage_bonus_cap=float(
            fusion_profile.get("subquery_coverage_bonus_cap", 0.25)
        ),
    )
    plan = state.get("query_plan") or {}
    planned_subqueries = {
        str(item.get("sub_query_id")): str(item.get("query") or "")
        for item in (plan.get("sub_queries") or [])
        if item.get("purpose") == "retrieval"
        and item.get("sub_query_id")
        and item.get("query")
    }
    approved_subquery_ids = {
        str(sub_query_id)
        for route in (plan.get("routes") or [])
        for sub_query_id in (route.get("sub_query_ids") or [])
        if str(sub_query_id) in planned_subqueries
    }
    approved_subquery_fingerprints = {
        sub_query_id: normalized_query_fingerprint(planned_subqueries[sub_query_id])
        for sub_query_id in approved_subquery_ids
    }
    approved_subquery_fingerprints = {
        key: value for key, value in approved_subquery_fingerprints.items() if value
    }
    fusion = fuse_route_results(
        {
            "base": embedding_chunks,
            "hyde": hyde_embedding_chunks,
            "web": web_documents,
        },
        config,
        approved_subquery_fingerprints=approved_subquery_fingerprints,
    )
    fused = fusion_result_to_mappings(fusion)

    # 把 QueryPlan 的子问题覆盖关系透传给 ContextPack。当前检索适配器可能把
    # 多个子问题合成一次查询，因此这里只声明该路线实际负责的批准子问题，
    # 不伪造新的候选或 Evidence。
    subqueries_by_route: dict[str, list[str]] = {}
    route_aliases = {
        "local_hybrid": "base",
        "hyde": "hyde",
        "web": "web",
    }
    for route in plan.get("routes") or []:
        alias = route_aliases.get(str(route.get("route_type") or ""))
        if alias:
            subqueries_by_route[alias] = [str(item) for item in (route.get("sub_query_ids") or [])]
    for document in fused:
        observations = document.get("retrieval_observations") or []
        explicit = [
            str(item.get("sub_query_id")) for item in observations if item.get("sub_query_id")
        ]
        # 当前检索适配器会在每个 observation 上记录精确 sub_query_id；此时
        # 必须只保留真实命中的分支，不能再把该路线批准的所有子问题都贴到每个
        # 候选上。仅对没有 observation 归属的旧适配器使用路线级兼容回退。
        if explicit:
            covered = explicit
        else:
            observed = {str(item.get("route") or "") for item in observations}
            covered = [
                subquery_id
                for route in observed
                for alias, approved in subqueries_by_route.items()
                if route == alias or route.startswith(f"{alias}.")
                for subquery_id in approved
            ]
        document["sub_query_ids"] = list(dict.fromkeys(covered))

    diagnostics = fusion.diagnostics.model_dump(mode="json")
    state["fusion_result"] = {
        "schema_version": "retrieval-fusion.v1",
        "method": config.method,
        "config": config.model_dump(mode="json"),
        "diagnostics": diagnostics,
    }
    state["fusion_diagnostics"] = diagnostics
    state["retrieval_diagnostics"] = {
        "profile_id": state.get("retrieval_profile_id", ""),
        "profile_version": state.get("retrieval_profile_version", 0),
        "profile_hash": state.get("retrieval_profile_hash", ""),
        "candidate_k": candidate_top,
        "top_k": int(profile_config.get("top_k", top)),
        "fusion": {
            "method": config.method,
            "rrf_k": rrf_k,
            "route_weights": configured_weights,
            "candidate_count": len(fused),
            "diagnostics": diagnostics,
        },
        "variants": [
            state.get("embedding_retrieval_diagnostics") or {},
            state.get("hyde_retrieval_diagnostics") or {},
            state.get("web_retrieval_diagnostics") or {},
        ],
    }
    return fused
