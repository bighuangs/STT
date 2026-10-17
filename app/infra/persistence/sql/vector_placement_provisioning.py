"""把新建 IndexVersion 接入 TODO-37 向量拓扑控制面。

这里只写 SQL 规划事实，不连接 Milvus。当前 chunks schema 的 partition contract 为
``none``，因此默认 shared 配置会安全提升为 KB dedicated，避免生成一个永远无法由
物理 schema 满足的共享 partition 路由。
"""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.application.vector_topology_service import VectorTopologyService
from app.domain.vector_topology import PlacementPolicy, PlacementRequest
from app.infra.persistence.sql.models import IndexVersionModel, KnowledgeBaseModel
from app.infra.persistence.sql.p3_repository import SqlVectorTopologyRepository
from app.infra.vectorstore.write_compatibility import current_chunks_profile
from app.shared.config.settings_config import settings


def _policy_version(value: str) -> int:
    match = re.search(r"(\d+)$", value.strip())
    return int(match.group(1)) if match else 1


def provision_managed_index_placement(
    session: Session,
    *,
    knowledge_base: KnowledgeBaseModel,
    index: IndexVersionModel,
    readable: bool,
) -> None:
    """在创建索引的同一事务内规划并激活唯一 managed placement。

    任一步骤异常都会让调用方事务回滚，因而不会留下“managed 但可走 legacy”的
    半成品索引。目标 Collection 是否存在、是否完全兼容仍由真实写入门禁验证。
    """

    if index.kb_id != knowledge_base.id:
        raise ValueError("index/knowledge-base identity mismatch")
    profile = current_chunks_profile(
        vector_dim=index.vector_dim,
        schema_version=index.schema_version,
        embedding_model_id=index.embedding_model_id,
    )
    configured_mode = settings.vector_placement_mode
    policy = PlacementPolicy(
        policy_id=settings.vector_placement_policy_version,
        version=_policy_version(settings.vector_placement_policy_version),
        default_database="default",
        tenant_dedicated_tenants=(
            frozenset({knowledge_base.tenant_id})
            if configured_mode == "tenant_dedicated"
            else frozenset()
        ),
        cluster_dedicated_tenants=(
            frozenset({knowledge_base.tenant_id})
            if configured_mode == "cluster_dedicated"
            else frozenset()
        ),
        kb_dedicated_ids=(
            frozenset({knowledge_base.id})
            if configured_mode in {"shared_compatible", "kb_dedicated"}
            else frozenset()
        ),
    )
    # 当前 schema 不创建 hash partitions；shared 配置必须提升为独占目标，而不能
    # 伪造一个 p_NNNN placement 后在首次写入时才失败。
    special_vector_contract = configured_mode == "shared_compatible"
    repository = SqlVectorTopologyRepository(session)
    service = VectorTopologyService(repository)
    family, placement, decision = service.plan_placement(
        policy=policy,
        request=PlacementRequest(
            tenant_id=knowledge_base.tenant_id,
            kb_id=knowledge_base.id,
            index_version_id=index.id,
            special_vector_contract=special_vector_contract,
        ),
        profile=profile,
    )
    if decision.partition_bucket is not None:
        raise ValueError("unpartitioned chunks schema received a partition placement")
    index.vector_governance_mode = "managed"
    index.collection_alias = decision.physical_collection
    service.activate_planned_placement(
        family_id=family.family_id,
        placement_id=placement.placement_id,
        readable=readable,
        writable=True,
    )


__all__ = ["provision_managed_index_placement"]
