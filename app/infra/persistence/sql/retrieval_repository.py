"""TODO-09 检索配置版本与元数据目录持久层。

该模块只管理控制面事实：

* 检索配置只能创建新 draft 版本，发布后不提供原地修改入口；
* 知识库指针与查询会话快照共同保证新会话切换、旧会话可重放；
* 元数据目录在入库时写入 SQL，查询改写不再扫描向量库中的大量 chunk。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.infra.persistence.sql.models import (
    DocumentModel,
    DocumentVersionModel,
    IndexVersionModel,
    KnowledgeBaseModel,
    MetadataCatalogEntryModel,
    QuerySessionModel,
    RetrievalProfileModel,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.security.principal import current_principal


SUPPORTED_VECTOR_DIMENSION = 1024
SUPPORTED_VECTOR_METRIC = "IP"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResourceConflictError(
            "retrieval profile and metadata values must be finite JSON values"
        ) from exc


def _json_hash(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: object) -> object:
    # JSON round-trip both validates the value and prevents callers from mutating the
    # SQLAlchemy JSON object after create_draft/replace_metadata_entries returns.
    return json.loads(_canonical_json(value))


def default_retrieval_profile_config() -> dict:
    """返回兼容 TODO-09 前既有行为的内建配置快照。

    配置结构直接由领域模型生成，避免数据库种子与在线解析器发生字段漂移。
    ``candidate_k`` 和最终 ``top_k`` 是两个独立字段，兼容基线中两者都为 5；
    dense/sparse 权重、RRF k 和长文本摘要行为也与旧实现一致。
    """

    from app.domain.retrieval import default_retrieval_profile

    profile = default_retrieval_profile("tenant_default")
    return profile.model_dump(
        mode="json",
        exclude={"profile_id", "version", "tenant_id", "status"},
    )


class RetrievalProfileRepository:
    """Tenant-scoped retrieval profile and metadata catalog repository."""

    def __init__(
        self,
        session: Session,
        tenant_id: str | None = None,
        actor_id: str | None = None,
    ):
        self.session = session
        principal = current_principal()
        self.tenant_id = (
            principal.tenant_id
            if principal is not None
            else (tenant_id.strip() if tenant_id else None)
        )
        self.actor_id = (
            principal.user_id
            if principal is not None
            else (actor_id.strip() if actor_id else "system")
        )

    def _required_tenant(self) -> str:
        if not self.tenant_id:
            raise ResourceConflictError("tenant_id is required for retrieval profile persistence")
        return self.tenant_id

    def _knowledge_base(self, kb_id: str) -> KnowledgeBaseModel:
        tenant_id = self._required_tenant()
        row = self.session.scalar(
            select(KnowledgeBaseModel).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == tenant_id,
                KnowledgeBaseModel.status != "deleted",
            )
        )
        if row is None:
            raise ResourceNotFoundError(f"knowledge base not found: {kb_id}")
        return row

    def control_plane_revision(self, kb_id: str) -> int:
        """Return the KB-wide optimistic concurrency token used by profile writes."""

        return int(self._knowledge_base(kb_id).revision)

    def _claim_control_plane(
        self,
        kb: KnowledgeBaseModel,
        *,
        expected_revision: int | None,
    ) -> int:
        """Serialize every retrieval-profile mutation on the KnowledgeBase row.

        The conditional revision increment is both an optimistic compare-and-swap
        and the database row lock shared by draft creation, publish, and rollback.
        Callers that omit the new token remain source-compatible: their observed
        revision is claimed, and a concurrent winner still makes the claim fail.
        """

        observed = int(kb.revision)
        claim_from = observed if expected_revision is None else int(expected_revision)
        if expected_revision is not None and observed != claim_from:
            raise ResourceConflictError(
                "knowledge base control-plane revision changed; refusing stale mutation"
            )
        changed = self.session.execute(
            update(KnowledgeBaseModel)
            .where(
                KnowledgeBaseModel.id == kb.id,
                KnowledgeBaseModel.tenant_id == kb.tenant_id,
                KnowledgeBaseModel.status != "deleted",
                KnowledgeBaseModel.revision == claim_from,
            )
            .values(revision=KnowledgeBaseModel.revision + 1)
            .execution_options(synchronize_session=False)
        )
        if changed.rowcount != 1:
            raise ResourceConflictError(
                "knowledge base control-plane revision changed during mutation claim"
            )
        self.session.refresh(kb)
        claimed = int(kb.revision)
        if claimed != claim_from + 1:  # pragma: no cover - database contract guard
            raise ResourceConflictError("knowledge base control-plane claim was not durable")
        return claimed

    def _max_profile_version(self, kb: KnowledgeBaseModel, profile_id: str) -> int:
        return int(
            self.session.scalar(
                select(func.max(RetrievalProfileModel.version)).where(
                    RetrievalProfileModel.tenant_id == kb.tenant_id,
                    RetrievalProfileModel.kb_id == kb.id,
                    RetrievalProfileModel.profile_id == profile_id,
                )
            )
            or 0
        )

    def _assert_expected_max_version(
        self,
        kb: KnowledgeBaseModel,
        profile_id: str,
        expected_max_version: int | None,
    ) -> int:
        actual = self._max_profile_version(kb, profile_id)
        if expected_max_version is not None and actual != int(expected_max_version):
            raise ResourceConflictError(
                "retrieval profile version lineage changed; refusing stale mutation"
            )
        return actual

    @staticmethod
    def _assert_profile_integrity(row: RetrievalProfileModel) -> None:
        actual_hash = _json_hash(row.config_json)
        if actual_hash != row.config_hash:
            raise ResourceConflictError(
                "retrieval profile config hash mismatch; refusing non-reproducible version"
            )

    def _expected_current(
        self,
        kb: KnowledgeBaseModel,
        *,
        profile_id: str | None,
        version: int | None,
        config_hash: str | None,
    ) -> RetrievalProfileModel | None:
        expectations = (profile_id, version, config_hash)
        if all(value is None for value in expectations):
            return None
        if any(value is None for value in expectations):
            raise ResourceConflictError(
                "expected current profile id, version, and config hash must be supplied together"
            )
        if not kb.retrieval_config_id:
            raise ResourceConflictError("knowledge base has no current retrieval profile")
        current = self.session.scalar(
            select(RetrievalProfileModel).where(
                RetrievalProfileModel.id == kb.retrieval_config_id,
                RetrievalProfileModel.tenant_id == kb.tenant_id,
                RetrievalProfileModel.kb_id == kb.id,
                RetrievalProfileModel.status == "published",
            )
        )
        if current is None:
            raise ResourceConflictError("knowledge base retrieval profile pointer is stale")
        self._assert_profile_integrity(current)
        if (
            current.profile_id != profile_id
            or current.version != int(version)
            or current.config_hash != config_hash
        ):
            raise ResourceConflictError(
                "current retrieval profile changed; refusing stale control-plane mutation"
            )
        return current

    def create_draft(
        self,
        kb_id: str,
        name: str,
        config: Mapping[str, object],
        created_by: str | None = None,
        description: str = "",
        profile_id: str | None = None,
        expected_current_profile_id: str | None = None,
        expected_current_version: int | None = None,
        expected_current_config_hash: str | None = None,
        expected_control_plane_revision: int | None = None,
        expected_max_version: int | None = None,
    ) -> RetrievalProfileModel:
        """创建新 draft；即便复用 profile_id 也只会递增版本，不会覆盖旧行。"""

        kb = self._knowledge_base(kb_id)
        self._claim_control_plane(
            kb,
            expected_revision=expected_control_plane_revision,
        )
        self._expected_current(
            kb,
            profile_id=expected_current_profile_id,
            version=expected_current_version,
            config_hash=expected_current_config_hash,
        )
        clean_name = name.strip()
        if not clean_name:
            raise ResourceConflictError("retrieval profile name must not be empty")
        if not isinstance(config, Mapping):
            raise ResourceConflictError("retrieval profile config must be a JSON object")
        snapshot = _json_copy(dict(config))
        assert isinstance(snapshot, dict)
        logical_id = (profile_id or _new_id("rpf")).strip()
        if not logical_id or len(logical_id) > 64:
            raise ResourceConflictError("retrieval profile_id must be 1..64 characters")
        previous_version = self._assert_expected_max_version(
            kb,
            logical_id,
            expected_max_version,
        )
        next_version = previous_version + 1
        # 仓储层也执行完整领域校验，避免绕过 HTTP/Pydantic 入口写入一个虽然是
        # JSON、却永远无法被在线查询解析的 draft。
        try:
            from app.domain.retrieval import RetrievalProfile

            RetrievalProfile(
                profile_id=logical_id,
                version=next_version,
                tenant_id=kb.tenant_id,
                status="draft",
                **snapshot,
            )
        except (TypeError, ValueError) as exc:
            raise ResourceConflictError(f"invalid retrieval profile config: {exc}") from exc
        row = RetrievalProfileModel(
            id=_new_id("rpv"),
            tenant_id=kb.tenant_id,
            kb_id=kb.id,
            profile_id=logical_id,
            version=next_version,
            name=clean_name[:255],
            description=description.strip(),
            status="draft",
            config_json=snapshot,
            config_hash=_json_hash(snapshot),
            created_by=(created_by or self.actor_id).strip() or self.actor_id,
        )
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise ResourceConflictError(
                f"retrieval profile version already exists: {logical_id} v{row.version}"
            ) from exc
        return row

    def list(
        self,
        kb_id: str | None = None,
        status: str | None = None,
    ) -> list[RetrievalProfileModel]:
        tenant_id = self._required_tenant()
        query = select(RetrievalProfileModel).where(RetrievalProfileModel.tenant_id == tenant_id)
        if kb_id is not None:
            self._knowledge_base(kb_id)
            query = query.where(RetrievalProfileModel.kb_id == kb_id)
        if status is not None:
            if status not in {"draft", "published", "retired"}:
                raise ResourceConflictError(f"invalid retrieval profile status: {status}")
            query = query.where(RetrievalProfileModel.status == status)
        rows = self.session.scalars(
            query.order_by(
                RetrievalProfileModel.created_at.desc(),
                RetrievalProfileModel.version.desc(),
            )
        ).all()
        for row in rows:
            self._assert_profile_integrity(row)
        return list(rows)

    def get(
        self,
        profile_id: str,
        version: int | None = None,
        status: str | None = None,
        *,
        kb_id: str | None = None,
    ) -> RetrievalProfileModel:
        """按逻辑 profile_id（或控制库行 id）读取租户内精确版本。"""

        tenant_id = self._required_tenant()
        query = select(RetrievalProfileModel).where(
            RetrievalProfileModel.tenant_id == tenant_id,
            (
                (RetrievalProfileModel.profile_id == profile_id)
                | (RetrievalProfileModel.id == profile_id)
            ),
        )
        if version is not None:
            query = query.where(RetrievalProfileModel.version == int(version))
        if status is not None:
            query = query.where(RetrievalProfileModel.status == status)
        if kb_id is not None:
            self._knowledge_base(kb_id)
            query = query.where(RetrievalProfileModel.kb_id == kb_id)
        ordered = query.order_by(RetrievalProfileModel.version.desc())
        if kb_id is None:
            # 旧调用方在租户内唯一匹配时仍可读取；一旦 profile identity 跨 KB
            # 或跨版本命中多行，就必须显式补全 KB 作用域，不能依赖数据库排序
            # 静默选中任意配置。
            rows = list(self.session.scalars(ordered.limit(2)).all())
            if len(rows) > 1:
                raise ResourceConflictError(
                    "retrieval profile identity is ambiguous; kb_id is required"
                )
            row = rows[0] if rows else None
        else:
            row = self.session.scalar(ordered)
        if row is None:
            suffix = f" v{version}" if version is not None else ""
            raise ResourceNotFoundError(f"retrieval profile not found: {profile_id}{suffix}")
        self._assert_profile_integrity(row)
        return row

    @staticmethod
    def _config_dimension(config: Mapping[str, object]) -> int:
        embedding = config.get("embedding")
        if isinstance(embedding, Mapping) and embedding.get("dimension") is not None:
            return int(embedding["dimension"])
        dense = config.get("dense")
        if isinstance(dense, Mapping) and dense.get("dimension") is not None:
            return int(dense["dimension"])
        for key in ("embedding_dimension", "vector_dim"):
            if config.get(key) is not None:
                return int(config[key])
        return SUPPORTED_VECTOR_DIMENSION

    @staticmethod
    def _config_metrics(config: Mapping[str, object]) -> set[str]:
        metrics: set[str] = set()
        for route in ("dense", "sparse"):
            section = config.get(route)
            if isinstance(section, Mapping):
                enabled = bool(section.get("enabled", True))
                if enabled:
                    metric = section.get("metric_type") or section.get("metric")
                    if metric:
                        metrics.add(str(metric).upper())
        if not metrics and config.get("metric_type"):
            metrics.add(str(config["metric_type"]).upper())
        return metrics or {SUPPORTED_VECTOR_METRIC}

    @staticmethod
    def _manifest_metric(index: IndexVersionModel) -> str:
        summary = index.manifest_summary or {}
        candidates: list[object] = [
            summary.get("metric_type"),
            summary.get("metric"),
            summary.get("vector_metric"),
            summary.get("dense_metric"),
        ]
        for section_name in ("dense", "dense_index", "vector", "index"):
            section = summary.get(section_name)
            if isinstance(section, Mapping):
                candidates.extend([section.get("metric_type"), section.get("metric")])
        metric = next((str(value).upper() for value in candidates if value), None)
        # Existing index schema before TODO-09 did not persist metric in the manifest;
        # the only index implementation is explicitly created with IP.
        return metric or SUPPORTED_VECTOR_METRIC

    def _validate_publish_compatibility(
        self,
        kb: KnowledgeBaseModel,
        profile: RetrievalProfileModel,
    ) -> IndexVersionModel:
        index = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.kb_id == kb.id,
                IndexVersionModel.status == "active",
                IndexVersionModel.active_slot == "active",
            )
        )
        if index is None:
            raise ResourceConflictError(f"knowledge base has no active index: {kb.id}")
        expected_dimension = self._config_dimension(profile.config_json)
        if index.vector_dim != SUPPORTED_VECTOR_DIMENSION or expected_dimension != index.vector_dim:
            raise ResourceConflictError(
                "retrieval profile is incompatible with active index dimension: "
                f"profile={expected_dimension}, index={index.vector_dim}, "
                f"supported={SUPPORTED_VECTOR_DIMENSION}"
            )
        profile_metrics = self._config_metrics(profile.config_json)
        index_metric = self._manifest_metric(index)
        if profile_metrics != {SUPPORTED_VECTOR_METRIC} or index_metric != SUPPORTED_VECTOR_METRIC:
            raise ResourceConflictError(
                "retrieval profile is incompatible with active index metric: "
                f"profile={sorted(profile_metrics)}, index={index_metric}, "
                f"supported={SUPPORTED_VECTOR_METRIC}"
            )
        return index

    def _activate(
        self,
        kb: KnowledgeBaseModel,
        target: RetrievalProfileModel,
        *,
        expected_current: RetrievalProfileModel | None = None,
        control_plane_revision: int | None = None,
    ) -> RetrievalProfileModel:
        if control_plane_revision is None:
            control_plane_revision = self._claim_control_plane(
                kb,
                expected_revision=None,
            )
        elif int(kb.revision) != int(control_plane_revision):
            raise ResourceConflictError("retrieval profile activation lost its KB claim")
        self._assert_profile_integrity(target)
        self._validate_publish_compatibility(kb, target)
        if target.status == "published" and kb.retrieval_config_id == target.id:
            return target
        now = _utcnow()
        current_rows = self.session.scalars(
            select(RetrievalProfileModel).where(
                RetrievalProfileModel.tenant_id == kb.tenant_id,
                RetrievalProfileModel.kb_id == kb.id,
                RetrievalProfileModel.status == "published",
                RetrievalProfileModel.id != target.id,
            )
        ).all()
        for current in current_rows:
            self._assert_profile_integrity(current)
            current.status = "retired"
            current.retired_at = now
        target.status = "published"
        target.published_at = now
        target.retired_at = None
        pointer_conditions = [
            KnowledgeBaseModel.id == kb.id,
            KnowledgeBaseModel.tenant_id == kb.tenant_id,
            KnowledgeBaseModel.revision == int(control_plane_revision),
        ]
        if expected_current is not None:
            pointer_conditions.append(KnowledgeBaseModel.retrieval_config_id == expected_current.id)
        changed = self.session.execute(
            update(KnowledgeBaseModel)
            .where(*pointer_conditions)
            .values(retrieval_config_id=target.id)
            .execution_options(synchronize_session=False)
        )
        if changed.rowcount != 1:
            raise ResourceConflictError("current retrieval profile changed during activation")
        self.session.expire(kb, ["retrieval_config_id"])
        self.session.flush()
        return target

    def publish(
        self,
        kb_id: str,
        profile_id: str,
        version: int,
        actor_id: str | None = None,
        expected_current_profile_id: str | None = None,
        expected_current_version: int | None = None,
        expected_current_config_hash: str | None = None,
        expected_control_plane_revision: int | None = None,
        expected_max_version: int | None = None,
    ) -> RetrievalProfileModel:
        """发布 draft；对同一目标重复调用幂等。"""

        del actor_id  # 生命周期事实由 created_by/published_at 表达，审计由 API 层记录。
        kb = self._knowledge_base(kb_id)
        claimed_revision = self._claim_control_plane(
            kb,
            expected_revision=expected_control_plane_revision,
        )
        expected_current = self._expected_current(
            kb,
            profile_id=expected_current_profile_id,
            version=expected_current_version,
            config_hash=expected_current_config_hash,
        )
        self._assert_expected_max_version(kb, profile_id, expected_max_version)
        target = self.get(profile_id, version, kb_id=kb_id)
        if target.status not in {"draft", "published"}:
            raise ResourceConflictError(
                "retired retrieval profile can only be reactivated through rollback"
            )
        return self._activate(
            kb,
            target,
            expected_current=expected_current,
            control_plane_revision=claimed_revision,
        )

    def rollback(
        self,
        kb_id: str,
        target_profile_id: str,
        target_version: int,
        actor_id: str | None = None,
        expected_current_profile_id: str | None = None,
        expected_current_version: int | None = None,
        expected_current_config_hash: str | None = None,
        expected_control_plane_revision: int | None = None,
        expected_max_version: int | None = None,
    ) -> RetrievalProfileModel:
        """将 KB 指针回滚到一个已发布过的精确版本。"""

        del actor_id
        kb = self._knowledge_base(kb_id)
        claimed_revision = self._claim_control_plane(
            kb,
            expected_revision=expected_control_plane_revision,
        )
        expected_current = self._expected_current(
            kb,
            profile_id=expected_current_profile_id,
            version=expected_current_version,
            config_hash=expected_current_config_hash,
        )
        self._assert_expected_max_version(
            kb,
            target_profile_id,
            expected_max_version,
        )
        target = self.get(
            target_profile_id,
            target_version,
            kb_id=kb_id,
        )
        if target.status not in {"published", "retired"} or target.published_at is None:
            raise ResourceConflictError(
                "rollback target must be a version that was published before"
            )
        return self._activate(
            kb,
            target,
            expected_current=expected_current,
            control_plane_revision=claimed_revision,
        )

    def resolve_for_kb(
        self,
        kb_id: str,
        allow_default: bool = True,
    ) -> RetrievalProfileModel:
        """解析 KB 当前发布指针；无指针时可原子创建并发布兼容默认版本。"""

        kb = self._knowledge_base(kb_id)
        if kb.retrieval_config_id:
            row = self.session.scalar(
                select(RetrievalProfileModel).where(
                    RetrievalProfileModel.id == kb.retrieval_config_id,
                    RetrievalProfileModel.tenant_id == kb.tenant_id,
                    RetrievalProfileModel.kb_id == kb.id,
                    RetrievalProfileModel.status == "published",
                )
            )
            if row is None:
                raise ResourceConflictError(
                    "knowledge base retrieval profile pointer is stale or cross-tenant"
                )
            self._assert_profile_integrity(row)
            return row
        if not allow_default:
            raise ResourceNotFoundError(
                f"knowledge base has no published retrieval profile: {kb_id}"
            )
        existing_default = self.session.scalar(
            select(RetrievalProfileModel)
            .where(
                RetrievalProfileModel.tenant_id == kb.tenant_id,
                RetrievalProfileModel.kb_id == kb.id,
                RetrievalProfileModel.profile_id == "rpf_default",
                RetrievalProfileModel.status.in_({"draft", "published"}),
            )
            .order_by(RetrievalProfileModel.version.desc())
        )
        if existing_default is not None:
            return self._activate(kb, existing_default)
        default = self.create_draft(
            kb_id,
            name="内建兼容检索配置",
            description="TODO-09 自动创建；保留既有 dense/sparse、RRF 与重排默认值。",
            config=default_retrieval_profile_config(),
            created_by="system",
            profile_id="rpf_default",
        )
        return self._activate(kb, default)

    def resolve_for_evaluation(
        self,
        kb_id: str,
        *,
        profile_id: str | None = None,
        version: int | None = None,
    ) -> RetrievalProfileModel:
        """解析评测使用的不可变配置，但绝不改变 KB 的发布指针。

        未指定版本时复用线上已发布配置；显式指定时允许对同一 tenant/KB 下的
        draft、published 或 retired 版本做离线 A/B。无论哪条路径都校验配置哈希
        和当前 active index 的维度/metric 兼容性，防止客户端伪造配置快照。
        """

        if (profile_id is None) != (version is None):
            raise ResourceConflictError(
                "evaluation retrieval profile id and version must be provided together"
            )
        kb = self._knowledge_base(kb_id)
        target = (
            self.get(profile_id, version, kb_id=kb_id)
            if profile_id is not None and version is not None
            else self.resolve_for_kb(kb_id)
        )
        self._assert_profile_integrity(target)
        self._validate_publish_compatibility(kb, target)
        return target

    def resolve_for_session(
        self,
        session_id: str,
        kb_id: str | None = None,
        user_id: str | None = None,
        *,
        config_version: str = "cfg_legacy",
    ) -> RetrievalProfileModel:
        """读取或写入会话 pin，保证配置发布后旧会话仍使用原版本。"""

        tenant_id = self._required_tenant()
        row = self.session.scalar(
            select(QuerySessionModel).where(
                QuerySessionModel.id == session_id,
                QuerySessionModel.tenant_id == tenant_id,
            )
        )
        if row is None:
            if not kb_id or not user_id:
                raise ResourceNotFoundError(f"query session not found: {session_id}")
            self._knowledge_base(kb_id)
            row = QuerySessionModel(
                id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                kb_id=kb_id,
                config_version=config_version,
            )
            self.session.add(row)
            self.session.flush()
        else:
            if kb_id is not None and row.kb_id != kb_id:
                raise ResourceConflictError("query session is pinned to a different knowledge base")
            if user_id is not None and row.user_id != user_id:
                raise ResourceNotFoundError(f"query session not found: {session_id}")

        pin_values = (
            row.retrieval_profile_id,
            row.retrieval_profile_version,
            row.retrieval_profile_hash,
        )
        if any(value is not None for value in pin_values) and not all(
            value is not None for value in pin_values
        ):
            raise ResourceConflictError("query session has an incomplete retrieval profile pin")
        if all(value is not None for value in pin_values):
            profile = self.get(
                str(row.retrieval_profile_id),
                int(row.retrieval_profile_version),
                kb_id=row.kb_id,
            )
            if profile.config_hash != row.retrieval_profile_hash:
                raise ResourceConflictError("query session retrieval profile hash mismatch")
            return profile

        profile = self.resolve_for_kb(row.kb_id)
        row.retrieval_profile_id = profile.profile_id
        row.retrieval_profile_version = profile.version
        row.retrieval_profile_hash = profile.config_hash
        self.session.flush()
        return profile

    @staticmethod
    def profile_dict(row: RetrievalProfileModel) -> dict:
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "kb_id": row.kb_id,
            "profile_id": row.profile_id,
            "version": row.version,
            "name": row.name,
            "description": row.description,
            "status": row.status,
            "config": deepcopy(row.config_json),
            "config_hash": row.config_hash,
            "created_by": row.created_by,
            "published_at": (row.published_at.isoformat() if row.published_at else None),
            "retired_at": row.retired_at.isoformat() if row.retired_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def _validate_catalog_scope(
        self,
        kb_id: str,
        document_id: str,
        document_version_id: str,
        index_version_id: str,
    ) -> KnowledgeBaseModel:
        kb = self._knowledge_base(kb_id)
        document = self.session.scalar(
            select(DocumentModel).where(
                DocumentModel.id == document_id,
                DocumentModel.kb_id == kb.id,
            )
        )
        version = self.session.scalar(
            select(DocumentVersionModel).where(
                DocumentVersionModel.id == document_version_id,
                DocumentVersionModel.document_id == document_id,
            )
        )
        index = self.session.scalar(
            select(IndexVersionModel).where(
                IndexVersionModel.id == index_version_id,
                IndexVersionModel.kb_id == kb.id,
            )
        )
        if document is None or version is None or index is None:
            raise ResourceNotFoundError(
                "metadata catalog scope does not belong to the requested tenant/KB"
            )
        return kb

    @staticmethod
    def _catalog_values(value: object) -> Iterable[object]:
        if value is None:
            return ()
        if isinstance(value, (list, tuple, set)):
            values: list[object] = []
            for item in value:
                if item is not None:
                    values.append(item)
            return values
        return (value,)

    @staticmethod
    def _catalog_value(value: object) -> tuple[str, object, str, str]:
        snapshot = _json_copy(value)
        if isinstance(snapshot, bool):
            value_type = "boolean"
            value_text = "true" if snapshot else "false"
        elif isinstance(snapshot, int):
            value_type = "integer"
            value_text = str(snapshot)
        elif isinstance(snapshot, float):
            if not math.isfinite(snapshot):
                raise ResourceConflictError("metadata catalog numbers must be finite")
            value_type = "number"
            value_text = _canonical_json(snapshot)
        elif isinstance(snapshot, dict) or isinstance(snapshot, list):
            value_type = "object"
            value_text = _canonical_json(snapshot)
        else:
            value_type = "string"
            value_text = str(snapshot)
            snapshot = value_text
        return value_type, snapshot, value_text, _json_hash(snapshot)

    def replace_metadata_entries(
        self,
        kb_id: str,
        document_id: str,
        document_version_id: str,
        index_version_id: str,
        metadata: Mapping[str, object],
    ) -> int:
        """幂等替换一个文档版本/索引版本的元数据目录。"""

        kb = self._validate_catalog_scope(
            kb_id,
            document_id,
            document_version_id,
            index_version_id,
        )
        if not isinstance(metadata, Mapping):
            raise ResourceConflictError("metadata catalog payload must be an object")
        self.session.execute(
            delete(MetadataCatalogEntryModel).where(
                MetadataCatalogEntryModel.tenant_id == kb.tenant_id,
                MetadataCatalogEntryModel.kb_id == kb.id,
                MetadataCatalogEntryModel.document_version_id == document_version_id,
                MetadataCatalogEntryModel.index_version_id == index_version_id,
            )
        )
        seen: set[tuple[str, str]] = set()
        created = 0
        for raw_key, raw_value in sorted(metadata.items(), key=lambda item: str(item[0])):
            key = str(raw_key).strip()
            if (
                not key
                or len(key) > 255
                or key.startswith("_")
                or any(char in key for char in ("\x00", "\r", "\n"))
            ):
                continue
            for value in self._catalog_values(raw_value):
                value_type, value_json, value_text, value_hash = self._catalog_value(value)
                identity = (key, value_hash)
                if identity in seen:
                    continue
                seen.add(identity)
                self.session.add(
                    MetadataCatalogEntryModel(
                        id=_new_id("mce"),
                        tenant_id=kb.tenant_id,
                        kb_id=kb.id,
                        document_id=document_id,
                        document_version_id=document_version_id,
                        index_version_id=index_version_id,
                        metadata_key=key,
                        value_type=value_type,
                        value_json=value_json,
                        value_text=value_text,
                        value_hash=value_hash,
                    )
                )
                created += 1
        self.session.flush()
        return created

    def load_metadata_catalog(
        self,
        kb_id: str,
        index_version_ids: Iterable[str] | None = None,
        document_ids: Iterable[str] | None = None,
        *,
        max_values_per_key: int = 200,
    ) -> dict[str, list[object]]:
        """从 SQL 控制目录读取有限、稳定排序的 ``key -> values`` 集合。"""

        kb = self._knowledge_base(kb_id)
        if max_values_per_key <= 0 or max_values_per_key > 1000:
            raise ResourceConflictError("max_values_per_key must be between 1 and 1000")
        requested_indexes = sorted({str(value) for value in (index_version_ids or []) if value})
        if not requested_indexes:
            active_index = self.session.scalar(
                select(IndexVersionModel.id).where(
                    IndexVersionModel.kb_id == kb.id,
                    IndexVersionModel.status == "active",
                    IndexVersionModel.active_slot == "active",
                )
            )
            if active_index is None:
                return {}
            requested_indexes = [active_index]
        owned_index_count = self.session.scalar(
            select(func.count(IndexVersionModel.id)).where(
                IndexVersionModel.kb_id == kb.id,
                IndexVersionModel.id.in_(requested_indexes),
            )
        )
        if int(owned_index_count or 0) != len(requested_indexes):
            raise ResourceNotFoundError(
                "one or more metadata catalog index versions are outside the KB"
            )
        query = select(MetadataCatalogEntryModel).where(
            MetadataCatalogEntryModel.tenant_id == kb.tenant_id,
            MetadataCatalogEntryModel.kb_id == kb.id,
            MetadataCatalogEntryModel.index_version_id.in_(requested_indexes),
        )
        requested_documents = sorted({str(value) for value in (document_ids or []) if value})
        if requested_documents:
            query = query.where(MetadataCatalogEntryModel.document_id.in_(requested_documents))
        rows = self.session.scalars(
            query.order_by(
                MetadataCatalogEntryModel.metadata_key.asc(),
                MetadataCatalogEntryModel.value_text.asc(),
                MetadataCatalogEntryModel.value_hash.asc(),
            )
        ).all()
        result: dict[str, list[object]] = {}
        seen: dict[str, set[str]] = {}
        for row in rows:
            values = result.setdefault(row.metadata_key, [])
            hashes = seen.setdefault(row.metadata_key, set())
            if row.value_hash in hashes or len(values) >= max_values_per_key:
                continue
            hashes.add(row.value_hash)
            values.append(deepcopy(row.value_json))
        return result

    def load_metadata_entries(
        self,
        kb_id: str,
        index_version_ids: Iterable[str] | None = None,
        document_ids: Iterable[str] | None = None,
    ) -> list[MetadataCatalogEntryModel]:
        """返回结构化目录行，供安全 Filter AST 解析器做类型感知匹配。"""

        kb = self._knowledge_base(kb_id)
        requested_indexes = sorted({str(value) for value in (index_version_ids or []) if value})
        if not requested_indexes:
            active_index = self.session.scalar(
                select(IndexVersionModel.id).where(
                    IndexVersionModel.kb_id == kb.id,
                    IndexVersionModel.status == "active",
                    IndexVersionModel.active_slot == "active",
                )
            )
            if active_index is None:
                return []
            requested_indexes = [active_index]
        query = select(MetadataCatalogEntryModel).where(
            MetadataCatalogEntryModel.tenant_id == kb.tenant_id,
            MetadataCatalogEntryModel.kb_id == kb.id,
            MetadataCatalogEntryModel.index_version_id.in_(requested_indexes),
        )
        requested_documents = sorted({str(value) for value in (document_ids or []) if value})
        if requested_documents:
            query = query.where(MetadataCatalogEntryModel.document_id.in_(requested_documents))
        return list(
            self.session.scalars(
                query.order_by(
                    MetadataCatalogEntryModel.document_id.asc(),
                    MetadataCatalogEntryModel.metadata_key.asc(),
                    MetadataCatalogEntryModel.value_text.asc(),
                )
            ).all()
        )


__all__ = [
    "RetrievalProfileRepository",
    "SUPPORTED_VECTOR_DIMENSION",
    "SUPPORTED_VECTOR_METRIC",
    "default_retrieval_profile_config",
]
