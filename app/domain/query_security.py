"""P3 查询授权范围与外发决策的稳定领域契约。

本模块刻意不依赖 FastAPI、SQLAlchemy、Milvus 或具体鉴权实现。查询运行只能把
客户端请求交给 :func:`authorize_retrieval_scope` 与服务端 entitlement 求交，得到
不可变的 :class:`AuthorizedRetrievalScope`。后续 dense/sparse/HyDE、父邻块扩展、
重排和引用回查应传递同一个 ``scope_hash``，不能重新根据客户端参数构造范围。
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.retrieval import FilterNode, validate_filter_ast


class ScopeAuthorizationError(PermissionError):
    """授权范围为空、过期或不一致。

    对外应统一映射成同一个 403/404 语义，避免通过错误差异探测资源是否存在。
    """

    reason_code = "RETRIEVAL_SCOPE_DENIED"


def _clean_id(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


class RequestedDocument(BaseModel):
    """客户端希望限制到的文档；必须显式携带 KB，避免 ID 猜测跨库。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kb_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)

    @field_validator("kb_id", "document_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _clean_id(value, field_name=info.field_name)


class RequestedRetrievalScope(BaseModel):
    """客户端表达的期望范围。

    它不是授权结果。空 KB、重复 ID、受保护字段过滤都由领域模型直接拒绝。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kb_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    documents: tuple[RequestedDocument, ...] = Field(
        default_factory=tuple,
        max_length=10_000,
    )
    business_filter: FilterNode | None = None

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_clean_id(item, field_name="kb_id") for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("kb_ids must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_documents(self) -> "RequestedRetrievalScope":
        if self.business_filter is not None:
            validate_filter_ast(self.business_filter)
        selected = set(self.kb_ids)
        identities = [(item.kb_id, item.document_id) for item in self.documents]
        if any(kb_id not in selected for kb_id, _ in identities):
            raise ValueError("requested documents must belong to requested kb_ids")
        if len(identities) != len(set(identities)):
            raise ValueError("requested documents must not contain duplicates")
        return self


class EntitledKnowledgeBase(BaseModel):
    """服务端权限事实中的单个 KB grant。

    ``document_ids=None`` 表示整个 KB；空元组表示没有任何文档权限，二者不能混淆。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kb_id: str = Field(min_length=1, max_length=64)
    document_ids: tuple[str, ...] | None = None
    grant_source: Literal["tenant_admin", "owner", "user_acl", "role_acl", "service_scope"]
    grant_role: str = Field(min_length=1, max_length=32)

    @field_validator("kb_id", "grant_role")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _clean_id(value, field_name=info.field_name)

    @field_validator("document_ids")
    @classmethod
    def normalize_document_ids(
        cls,
        value: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(_clean_id(item, field_name="document_id") for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("entitled document_ids must not contain duplicates")
        return normalized


class EntitlementSnapshot(BaseModel):
    """来自控制面的不可变授权事实快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=64)
    principal_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=64)
    grants: tuple[EntitledKnowledgeBase, ...]
    policy_version: str = Field(min_length=1, max_length=128)
    entitlement_version: str = Field(min_length=1, max_length=128)

    @field_validator(
        "tenant_id",
        "principal_id",
        "workspace_id",
        "policy_version",
        "entitlement_version",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _clean_id(value, field_name=info.field_name)

    @model_validator(mode="after")
    def unique_grants(self) -> "EntitlementSnapshot":
        kb_ids = [item.kb_id for item in self.grants]
        if len(kb_ids) != len(set(kb_ids)):
            raise ValueError("entitlement grants must contain at most one row per kb")
        return self


class AuthorizedKnowledgeBase(BaseModel):
    """请求与授权求交后的单个 KB 范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kb_id: str = Field(min_length=1, max_length=64)
    # None 表示该 KB 的全部已发布文档；空元组永远不会进入授权结果。
    document_ids: tuple[str, ...] | None = None
    grant_source: str = Field(min_length=1, max_length=32)
    grant_role: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def reject_empty_explicit_documents(self) -> "AuthorizedKnowledgeBase":
        if self.document_ids == ():
            raise ValueError("authorized KB cannot contain an empty explicit document set")
        return self


class ActiveIndexBinding(BaseModel):
    """查询运行冻结的活动发布事实，用于命中后的逐条 identity 复核。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kb_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    document_version_id: str = Field(min_length=1, max_length=64)
    index_version_id: str = Field(min_length=1, max_length=64)

    @field_validator(
        "kb_id",
        "document_id",
        "document_version_id",
        "index_version_id",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _clean_id(value, field_name=info.field_name)


class AuthorizedRetrievalScope(BaseModel):
    """任何检索、扩展、重排和引用回查都必须消费的不可变范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=64)
    principal_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=64)
    resources: tuple[AuthorizedKnowledgeBase, ...] = Field(min_length=1)
    business_filter: FilterNode | None = None
    active_indexes: tuple[ActiveIndexBinding, ...] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    entitlement_version: str = Field(min_length=1, max_length=128)
    scope_hash: str = Field(pattern=r"^scope_[0-9a-f]{64}$")

    @property
    def allowed_kb_ids(self) -> tuple[str, ...]:
        return tuple(item.kb_id for item in self.resources)

    @property
    def allowed_document_ids(self) -> tuple[str, ...] | None:
        if any(item.document_ids is None for item in self.resources):
            return None
        return tuple(
            document_id for item in self.resources for document_id in (item.document_ids or ())
        )

    @property
    def mandatory_filter_ast(self) -> dict[str, Any]:
        """只由服务端事实生成的强制过滤树，供 trace/受控编译器消费。"""

        clauses: list[dict[str, Any]] = []
        if self.active_indexes:
            for item in self.active_indexes:
                clauses.append(
                    {
                        "operator": "and",
                        "predicates": [
                            {"field": "tenant_id", "operator": "eq", "value": self.tenant_id},
                            {"field": "kb_id", "operator": "eq", "value": item.kb_id},
                            {
                                "field": "document_id",
                                "operator": "eq",
                                "value": item.document_id,
                            },
                            {
                                "field": "document_version_id",
                                "operator": "eq",
                                "value": item.document_version_id,
                            },
                            {
                                "field": "index_version_id",
                                "operator": "eq",
                                "value": item.index_version_id,
                            },
                            {"field": "available", "operator": "eq", "value": True},
                        ],
                    }
                )
        else:
            for resource in self.resources:
                predicates: list[dict[str, Any]] = [
                    {"field": "tenant_id", "operator": "eq", "value": self.tenant_id},
                    {"field": "kb_id", "operator": "eq", "value": resource.kb_id},
                ]
                if resource.document_ids is not None:
                    predicates.append(
                        {
                            "field": "document_id",
                            "operator": "in",
                            "value": list(resource.document_ids),
                        }
                    )
                clauses.append({"operator": "and", "predicates": predicates})
        return {"operator": "or", "conditions": clauses}

    def permits(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        document_id: str,
        document_version_id: str | None = None,
        index_version_id: str | None = None,
    ) -> bool:
        """逐条复核命中；存在冻结 index 时版本也必须精确匹配。"""

        if tenant_id != self.tenant_id:
            return False
        resource = next((item for item in self.resources if item.kb_id == kb_id), None)
        if resource is None:
            return False
        if resource.document_ids is not None and document_id not in resource.document_ids:
            return False
        if not self.active_indexes:
            return document_version_id is None and index_version_id is None
        return any(
            item.kb_id == kb_id
            and item.document_id == document_id
            and (document_version_id is None or item.document_version_id == document_version_id)
            and (index_version_id is None or item.index_version_id == index_version_id)
            for item in self.active_indexes
        )

    def audit_fields(self) -> dict[str, Any]:
        """最小化审计载荷；绝不包含查询正文或 metadata filter 的值。"""

        return {
            "scope_hash": self.scope_hash,
            "policy_version": self.policy_version,
            "entitlement_version": self.entitlement_version,
            "tenant_id": self.tenant_id,
            "principal_id": self.principal_id,
            "workspace_id": self.workspace_id,
            "kb_ids": list(self.allowed_kb_ids),
            "document_restricted": self.allowed_document_ids is not None,
            "active_index_count": len(self.active_indexes),
        }

    @model_validator(mode="after")
    def validate_snapshot(self) -> "AuthorizedRetrievalScope":
        kb_ids = [item.kb_id for item in self.resources]
        if len(kb_ids) != len(set(kb_ids)):
            raise ValueError("authorized resources must contain at most one row per kb")
        allowed = set(kb_ids)
        for item in self.active_indexes:
            if item.kb_id not in allowed:
                raise ValueError("active index is outside authorized KB scope")
            resource = next(entry for entry in self.resources if entry.kb_id == item.kb_id)
            if resource.document_ids is not None and item.document_id not in resource.document_ids:
                raise ValueError("active index is outside authorized document scope")
        expected = compute_scope_hash(
            tenant_id=self.tenant_id,
            principal_id=self.principal_id,
            workspace_id=self.workspace_id,
            resources=self.resources,
            business_filter=self.business_filter,
            active_indexes=self.active_indexes,
            policy_version=self.policy_version,
            entitlement_version=self.entitlement_version,
        )
        if self.scope_hash != expected:
            raise ValueError("scope_hash does not match authorized scope contents")
        return self


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def compute_scope_hash(
    *,
    tenant_id: str,
    principal_id: str,
    workspace_id: str,
    resources: tuple[AuthorizedKnowledgeBase, ...],
    business_filter: FilterNode | None,
    active_indexes: tuple[ActiveIndexBinding, ...],
    policy_version: str,
    entitlement_version: str,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "principal_id": principal_id,
        "workspace_id": workspace_id,
        "resources": sorted(
            (_canonical(item) for item in resources),
            key=lambda item: item["kb_id"],
        ),
        "business_filter": _canonical(business_filter),
        "active_indexes": sorted(
            (_canonical(item) for item in active_indexes),
            key=lambda item: (
                item["kb_id"],
                item["document_id"],
                item["document_version_id"],
                item["index_version_id"],
            ),
        ),
        "policy_version": policy_version,
        "entitlement_version": entitlement_version,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"scope_{sha256(serialized.encode('utf-8')).hexdigest()}"


def authorize_retrieval_scope(
    requested: RequestedRetrievalScope,
    entitlement: EntitlementSnapshot,
    *,
    active_indexes: tuple[ActiveIndexBinding, ...] = (),
) -> AuthorizedRetrievalScope:
    """计算 ``entitlement ∩ requested``，空交集 fail-closed。

    该函数只会丢弃无权资源，从不增加客户端未请求的 KB/文档。所有排序均规范化，
    因此相同授权事实会得到稳定 ``scope_hash``。
    """

    grants = {item.kb_id: item for item in entitlement.grants}
    requested_documents: dict[str, set[str]] = {}
    for item in requested.documents:
        requested_documents.setdefault(item.kb_id, set()).add(item.document_id)

    resources: list[AuthorizedKnowledgeBase] = []
    for kb_id in requested.kb_ids:
        grant = grants.get(kb_id)
        if grant is None:
            continue
        selected = requested_documents.get(kb_id)
        if selected is None:
            allowed_documents = grant.document_ids
        elif grant.document_ids is None:
            allowed_documents = tuple(sorted(selected))
        else:
            intersection = selected.intersection(grant.document_ids)
            if not intersection:
                continue
            allowed_documents = tuple(sorted(intersection))
        if allowed_documents == ():
            continue
        resources.append(
            AuthorizedKnowledgeBase(
                kb_id=kb_id,
                document_ids=allowed_documents,
                grant_source=grant.grant_source,
                grant_role=grant.grant_role,
            )
        )

    if not resources:
        raise ScopeAuthorizationError("requested retrieval scope is not authorized")

    normalized_resources = tuple(sorted(resources, key=lambda item: item.kb_id))
    allowed_kbs = {item.kb_id for item in normalized_resources}
    allowed_index_rows = tuple(
        sorted(
            (
                item
                for item in active_indexes
                if item.kb_id in allowed_kbs
                and any(
                    resource.kb_id == item.kb_id
                    and (resource.document_ids is None or item.document_id in resource.document_ids)
                    for resource in normalized_resources
                )
            ),
            key=lambda item: (
                item.kb_id,
                item.document_id,
                item.document_version_id,
                item.index_version_id,
            ),
        )
    )
    if active_indexes and not allowed_index_rows:
        raise ScopeAuthorizationError("authorized scope has no active index")

    kwargs = {
        "tenant_id": entitlement.tenant_id,
        "principal_id": entitlement.principal_id,
        "workspace_id": entitlement.workspace_id,
        "resources": normalized_resources,
        "business_filter": requested.business_filter,
        "active_indexes": allowed_index_rows,
        "policy_version": entitlement.policy_version,
        "entitlement_version": entitlement.entitlement_version,
    }
    return AuthorizedRetrievalScope(
        **kwargs,
        scope_hash=compute_scope_hash(**kwargs),
    )


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


_CLASSIFICATION_RANK = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


class ExternalDataRequest(BaseModel):
    """一次 Web/外部 LLM/reranker 外发请求，不携带原始正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: bool = False
    provider: str = Field(min_length=1, max_length=128)
    purpose: Literal["web_search", "llm_generation", "rerank", "embedding"]
    scope_hash: str = Field(pattern=r"^scope_[0-9a-f]{64}$")
    data_classes: tuple[DataClassification, ...] = Field(min_length=1)
    target_domain: str | None = Field(default=None, max_length=253)
    contains_direct_identifier: bool = False
    contains_secret: bool = False


class EgressPolicySnapshot(BaseModel):
    """租户/KB 级外发策略；默认构造即为拒绝。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="egress.v1-deny", min_length=1, max_length=128)
    enabled: bool = False
    allowed_providers: tuple[str, ...] = ()
    allowed_purposes: tuple[
        Literal["web_search", "llm_generation", "rerank", "embedding"], ...
    ] = ()
    allowed_domains: tuple[str, ...] = ()
    max_classification: DataClassification = DataClassification.PUBLIC
    redact_direct_identifiers: bool = True
    retention_policy: str = Field(default="no-retention", min_length=1, max_length=128)


class ExternalDataDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    provider: str
    purpose: str
    data_classes: tuple[DataClassification, ...]
    redactions: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    retention_policy: str
    reason_code: str = Field(min_length=1, max_length=128)
    decision_id: str = Field(pattern=r"^egd_[0-9a-f]{64}$")
    policy_version: str
    scope_hash: str

    def audit_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def decide_external_data(
    request: ExternalDataRequest,
    policy: EgressPolicySnapshot | None,
) -> ExternalDataDecision:
    """确定性、默认拒绝的外发判定。

    ``requested=True`` 仅代表客户端意图；只有开关、provider、purpose、域名、分类
    与秘密检查全部通过才允许。决策 ID 只基于元数据，不包含外发正文。
    """

    effective = policy or EgressPolicySnapshot()
    reason = "EGRESS_ALLOWED"
    allowed = True
    redactions: list[str] = []
    if not request.requested:
        allowed, reason = False, "EGRESS_NOT_REQUESTED"
    elif not effective.enabled:
        allowed, reason = False, "EGRESS_DISABLED"
    elif request.provider not in effective.allowed_providers:
        allowed, reason = False, "EGRESS_PROVIDER_DENIED"
    elif request.purpose not in effective.allowed_purposes:
        allowed, reason = False, "EGRESS_PURPOSE_DENIED"
    elif request.contains_secret:
        allowed, reason = False, "EGRESS_SECRET_BLOCKED"
    elif any(
        _CLASSIFICATION_RANK[item] > _CLASSIFICATION_RANK[effective.max_classification]
        for item in request.data_classes
    ):
        allowed, reason = False, "EGRESS_CLASSIFICATION_DENIED"
    elif request.target_domain and (
        not effective.allowed_domains
        or request.target_domain.lower() not in {item.lower() for item in effective.allowed_domains}
    ):
        allowed, reason = False, "EGRESS_DOMAIN_DENIED"
    elif request.contains_direct_identifier:
        if effective.redact_direct_identifiers:
            redactions.append("direct_identifier")
        else:
            allowed, reason = False, "EGRESS_IDENTIFIER_DENIED"

    decision_payload = {
        "allowed": allowed,
        "provider": request.provider,
        "purpose": request.purpose,
        "data_classes": sorted(item.value for item in request.data_classes),
        "redactions": sorted(redactions),
        "allowed_domains": sorted(effective.allowed_domains),
        "retention_policy": effective.retention_policy,
        "reason_code": reason,
        "policy_version": effective.policy_version,
        "scope_hash": request.scope_hash,
    }
    serialized = json.dumps(
        decision_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ExternalDataDecision(
        **decision_payload,
        decision_id=f"egd_{sha256(serialized.encode('utf-8')).hexdigest()}",
    )


__all__ = [
    "ActiveIndexBinding",
    "AuthorizedKnowledgeBase",
    "AuthorizedRetrievalScope",
    "DataClassification",
    "EgressPolicySnapshot",
    "EntitledKnowledgeBase",
    "EntitlementSnapshot",
    "ExternalDataDecision",
    "ExternalDataRequest",
    "RequestedDocument",
    "RequestedRetrievalScope",
    "ScopeAuthorizationError",
    "authorize_retrieval_scope",
    "compute_scope_hash",
    "decide_external_data",
]
