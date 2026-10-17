"""TODO-24/25/26/31 可靠查询运行 HTTP 路由。

该模块只负责注册独立的 ``/api/v1/query-runs`` 与 ``/api/v1/conversations``
接口，不持有进程内查询事实。QueryRun、历史快照和事件均以控制库为事实源；
FastAPI ``BackgroundTasks`` 仅负责触发当前进程中的真实查询图。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from mimetypes import guess_type
from pathlib import Path
import re
import time
from typing import Any, Literal
from urllib.parse import quote, urlsplit
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import select

from app.agent_system.agents.query.durable_executor import (
    DurableQueryPreflight,
    QueryBoundaryCancellationRequested,
    QueryCheckpointCompatibilityError,
    QueryCheckpointLoadError,
    QueryExecutionCancelled,
    QueryExecutionDeadlineExceeded,
    QueryInterruptedDuringNode,
)
from app.api.schemas.query_runtime import (
    AuthorizedScopeSummary,
    ConversationDeleteResponse,
    ConversationListResponse,
    ConversationMessageListResponse,
    ConversationMessageResponse,
    ConversationMutationRequest,
    ConversationMutationResponse,
    ConversationRunListResponse,
    ConversationTurnResponse,
    QueryEventPageResponse,
    QueryRunAcceptRequest,
    QueryRunAcceptedResponse,
    QueryRunCancelRequest,
    QueryRunCancelResponse,
    QueryReplayManifestResponse,
    QueryRunResponse,
    QueryRunReplayRequest,
    QueryRunReplayResponse,
    QueryRunResultResponse,
    QueryRunRetryRequest,
    RetrievalProfileIdentity,
)
from app.application.query_cache_service import (
    QueryFingerprint,
    QueryReplayManifest,
    ReplayService,
)
from app.application.control_plane_service import get_document_version
from app.application.immutable_source_service import (
    ImmutableSourceIntegrityError,
    ImmutableSourceUnavailableError,
    resolve_immutable_local_source,
)
from app.application.query_authorization_service import (
    EntitlementUnavailableError,
    QueryAuthorizationService,
    redact_external_payload,
)
from app.application.query_answering.pipeline import (
    AnswerProofNotCommittableError,
    validate_committable_answer,
)
from app.application.query_runtime_hooks import (
    QueryRuntimeProgressHook,
    bind_query_runtime_hook,
)
from app.application.query_security_pipeline import (
    QueryContentSecurityViolation,
    QueryEgressDenied,
)
from app.application.query_runtime_service import (
    QueryCancellationRequested,
    QueryRuntimeService,
)
from app.domain.content_security import ContentSecurityPolicy
from app.domain._state_machine import InvalidStateTransition
from app.domain.query_runtime import (
    ConversationHistorySnapshot,
    QueryEventType,
    QueryRun,
    QueryRunStatus,
    RUN_TERMINAL_STATUSES,
    TurnStatus,
    canonical_hash,
)
from app.domain.query_checkpoint import QueryExecutionLease, QueryExecutionStatus
from app.foundation.query_checkpoint_crypto import (
    EncryptedQueryCheckpoint,
    QueryCheckpointCipher,
    QueryCheckpointEncryptionContext,
)
from app.domain.answer import AnswerEnvelope
from app.domain.query_resilience import RequiredRouteError
from app.domain.query_security import (
    AuthorizedRetrievalScope,
    DataClassification,
    EgressPolicySnapshot,
    ExternalDataDecision,
    ExternalDataRequest,
    RequestedDocument,
    RequestedRetrievalScope,
    ScopeAuthorizationError,
)
from app.domain.retrieval import (
    FILTER_MAX_VALUES,
    FilterGroup,
    FilterNode,
    FilterPredicate,
    RetrievalEngineError,
    RetrievalScope,
)
from app.infra.config import settings
from app.infra.persistence.sql.database import session_scope
from app.infra.persistence.sql.p3_models import QueryIdempotencyKeyModel
from app.infra.persistence.sql.p3_repository import QueryReliabilityRepository
from app.infra.persistence.sql.query_runtime_repository import (
    EventCursorExpired,
    EventCursorInvalid,
    QueryCheckpointRepository,
    QueryExecutionFenceLost,
    QueryExecutionLeaseBusy,
    QueryExecutionNotInitialized,
    QueryRuntimeRepository,
    StaleQueryRevision,
)
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)
from app.infra.persistence.sql.models import KnowledgeBaseModel
from app.infra.persistence.sql.retrieval_repository import (
    RetrievalProfileRepository,
)
from app.infra.retrieval.control_plane_resolvers import (
    FrozenActiveScopeResolver,
    freeze_vector_read_set,
)
from app.infra.security.sql_query_authorization import (
    SqlActiveIndexBindingResolver,
    SqlEntitlementResolver,
    SqlScopeDecisionAudit,
    resolve_egress_policy,
)
from app.infrastructure.query_runtime_durability import (
    DurableQueryRecoveryController,
    QueryDurabilityConfigurationError,
    QueryRecoveryStopped,
    QueryResumeIdentityError,
    bind_query_execution_identity,
    build_durable_query_executor,
    build_query_checkpoint_cipher,
    initialize_query_execution,
    recover_query_principal,
)
from app.security.principal import (
    Principal,
    reset_current_principal,
    set_current_principal,
)
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.rag.query.evidence_service import get_authorized_evidence
from app.shared.utils.query_event_stream import (
    DurableStreamPolicy,
    durable_sse_generator,
)


_DIRECT_IDENTIFIER = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b"
    r"|(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|authorization|password|secret)"
    r"\s*[:=]\s*[^\s,;]{4,}"
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ORIGINALS_ROOT = PROJECT_ROOT / "output" / "originals"
_PUBLIC_RESULT_FIELDS = frozenset(
    {
        "answer",
        "answer_id",
        "answer_envelope",
        "answer_proof",
        "citations",
        "sources",
        "citation_validation",
        "image_urls",
        "retrieval_profile_id",
        "retrieval_profile_version",
        "retrieval_profile_hash",
        "retrieval_diagnostics",
        "query_understanding",
        "query_plan",
        "query_execution_summary",
        "fusion_diagnostics",
        "context_pack_summary",
        "evidence_assessment",
        "answer_decision",
        "cache",
        "grounding",
        "usage",
    }
)
_REPLAY_PROOF_CHECKPOINT_REF_PREFIX = "query-proof-checkpoint.v2|"
_REPLAY_MANIFEST_SCHEMA_VERSION = "query-replay/2"
_REPLAY_PROOF_STATE_FIELDS = (
    "query_run_id",
    "answer_id",
    "answer",
    "query_plan",
    "question_contract",
    "witness_catalog",
    "context_aliases",
    "answer_plan",
    "answer_reviewed_plan",
    "answer_plan_review",
    "answer_review_closure_receipt",
    "answer_prompt_receipts",
    "answer_coverage_decision",
    "answer_proof",
    "answer_envelope",
    "answer_render_receipt",
    "citations",
    "pending_citation_facts",
)


class QueryRuntimeApiError(RuntimeError):
    """带稳定错误码的安全 HTTP 失败；details 不得包含查询或上下文正文。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details or {})


class _DeduplicatedAcceptance(RuntimeError):
    def __init__(self, query_run_id: str) -> None:
        super().__init__("acceptance was deduplicated")
        self.query_run_id = query_run_id


@dataclass(frozen=True)
class _FrozenExecution:
    scope: AuthorizedRetrievalScope
    profile: dict[str, Any]
    index_snapshot: dict[str, Any]
    config_snapshot: dict[str, Any]
    execution_query: str

    @property
    def profile_version(self) -> str:
        return (
            f"{self.profile['profile_id']}@{self.profile['version']}:{self.profile['config_hash']}"
        )


@dataclass(frozen=True, slots=True)
class _ReplayProofCheckpointBinding:
    """Manifest 对最终 proof checkpoint 的不可变绑定。"""

    query_run_id: str
    sequence: int
    ciphertext_sha256: str
    proof_projection_sha256: str

    def to_ref(self) -> str:
        return "|".join(
            (
                "query-proof-checkpoint.v2",
                self.query_run_id,
                str(self.sequence),
                self.ciphertext_sha256,
                self.proof_projection_sha256,
            )
        )

    @classmethod
    def from_manifest(cls, manifest: QueryReplayManifest) -> "_ReplayProofCheckpointBinding":
        if manifest.schema_version != _REPLAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError("replay manifest schema version is unsupported")
        refs = [
            value
            for value in manifest.route_result_refs
            if value.startswith(_REPLAY_PROOF_CHECKPOINT_REF_PREFIX)
        ]
        if len(refs) != 1:
            raise ValueError("replay manifest must bind exactly one proof checkpoint")
        parts = refs[0].split("|")
        if len(parts) != 5:
            raise ValueError("replay proof checkpoint reference is malformed")
        try:
            sequence = int(parts[2])
        except ValueError as exc:
            raise ValueError("replay proof checkpoint sequence is malformed") from exc
        binding = cls(
            query_run_id=parts[1],
            sequence=sequence,
            ciphertext_sha256=parts[3],
            proof_projection_sha256=parts[4],
        )
        if binding.query_run_id != manifest.query_run_id or binding.sequence < 1:
            raise ValueError("replay proof checkpoint identity is inconsistent")
        for value in (binding.ciphertext_sha256, binding.proof_projection_sha256):
            if len(value) != 71 or not value.startswith("sha256:"):
                raise ValueError("replay proof checkpoint hash is malformed")
        return binding


def _checkpoint_cipher() -> QueryCheckpointCipher:
    """可靠 runtime 缺少独立 key 时 fail-closed；不允许复用其他 secret。"""

    try:
        return build_query_checkpoint_cipher(settings.query_checkpoint_encryption_key)
    except QueryDurabilityConfigurationError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "QUERY_CHECKPOINT_KEY_UNAVAILABLE",
            "可靠查询的持久恢复密钥未配置或配置无效。",
        ) from exc


def _bind_execution_identity(session, principal: Principal):
    """把基础设施身份拒绝映射为稳定 HTTP 错误。"""

    try:
        return bind_query_execution_identity(session, principal)
    except QueryResumeIdentityError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_403_FORBIDDEN,
            exc.code,
            exc.safe_message,
        ) from exc


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or f"req_{uuid4().hex}"


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise QueryRuntimeApiError(
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHENTICATED",
            "缺少有效的认证主体。",
        )
    return principal


def _runtime_repository(session, principal: Principal) -> QueryRuntimeRepository:
    return QueryRuntimeRepository(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
    )


def _authorization_service(principal: Principal) -> QueryAuthorizationService:
    return QueryAuthorizationService(
        entitlements=SqlEntitlementResolver(),
        active_indexes=SqlActiveIndexBindingResolver(),
        audit=SqlScopeDecisionAudit(principal),
    )


def _conversation_kb_is_currently_entitled(
    *,
    principal: Principal,
    kb_id: str,
) -> bool:
    """按当前控制面事实复核历史会话 KB；历史拥有权本身不是资源授权。"""

    try:
        entitlement = SqlEntitlementResolver().resolve(
            principal=principal,
            requested=RequestedRetrievalScope(kb_ids=(kb_id,)),
        )
    except ScopeAuthorizationError:
        return False
    except Exception as exc:
        # 授权依赖故障时不能退化成“仅凭历史 owner 可读”。统一 503 且不返回
        # 候选 KB/会话信息，避免将依赖故障误当成放行。
        logger.exception("历史会话 entitlement 复核失败")
        raise QueryRuntimeApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ENTITLEMENT_UNAVAILABLE",
            "历史会话授权事实暂时不可用，请稍后重试。",
        ) from exc
    return any(item.kb_id == kb_id for item in entitlement.grants)


def _authorized_conversation_kb_ids(
    *,
    principal: Principal,
    candidate_kb_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """把 tenant/user 历史候选集与实时 KB entitlement 求交。"""

    return tuple(
        kb_id
        for kb_id in candidate_kb_ids
        if _conversation_kb_is_currently_entitled(principal=principal, kb_id=kb_id)
    )


def _conversation_not_found() -> QueryRuntimeApiError:
    # 不区分不存在、其他 tenant/user 与已撤权，防止资源枚举。
    return QueryRuntimeApiError(
        status.HTTP_404_NOT_FOUND,
        "CONVERSATION_NOT_FOUND",
        "会话不存在或当前无权访问。",
    )


def _require_authorized_conversation(
    *,
    repository: QueryRuntimeRepository,
    principal: Principal,
    conversation_id: str,
):
    try:
        conversation = repository.get_conversation(conversation_id)
    except ResourceNotFoundError as exc:
        raise _conversation_not_found() from exc
    if not _conversation_kb_is_currently_entitled(
        principal=principal,
        kb_id=conversation.kb_id,
    ):
        raise _conversation_not_found()
    return conversation


def _conversation_conflict(exc: ResourceConflictError) -> QueryRuntimeApiError:
    message = str(exc)
    if isinstance(exc, StaleQueryRevision):
        return QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "CONVERSATION_REVISION_CONFLICT",
            "会话已被其他操作更新，请刷新后重试。",
        )
    if "active query runs" in message:
        return QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "CONVERSATION_BUSY",
            "会话仍有进行中的查询，结束或取消后才能执行此操作。",
        )
    if "archived before deletion" in message:
        return QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "CONVERSATION_ARCHIVE_REQUIRED",
            "永久删除前必须先归档会话。",
        )
    return QueryRuntimeApiError(
        status.HTTP_409_CONFLICT,
        "CONVERSATION_CONFLICT",
        "会话当前状态不允许执行此操作，请刷新后重试。",
    )


def _authorized_inline_source(
    *,
    principal: Principal,
    evidence_id: str,
) -> tuple[Path, str]:
    """解析授权 Evidence 原件；只把受控文件响应交给浏览器，不返回存储键。"""

    try:
        evidence = get_authorized_evidence(evidence_id)
    except Exception as exc:
        raise QueryRuntimeApiError(
            status.HTTP_404_NOT_FOUND,
            "EVIDENCE_NOT_FOUND",
            "引用来源不存在或当前无权访问。",
        ) from exc
    source = evidence.get("source") or {}
    kb_id = str(source.get("kb_id") or "")
    if (
        source.get("source_type") != "local"
        or source.get("tenant_id") != principal.tenant_id
        or not kb_id
        or not _conversation_kb_is_currently_entitled(
            principal=principal,
            kb_id=kb_id,
        )
    ):
        raise QueryRuntimeApiError(
            status.HTTP_404_NOT_FOUND,
            "EVIDENCE_NOT_FOUND",
            "引用来源不存在或当前无权访问。",
        )
    try:
        version = get_document_version(str(source["document_version_id"]))
    except Exception as exc:
        raise QueryRuntimeApiError(
            status.HTTP_404_NOT_FOUND,
            "EVIDENCE_SOURCE_UNAVAILABLE",
            "引用原件已不可用。",
        ) from exc
    try:
        immutable_source = resolve_immutable_local_source(
            object_key=version.get("object_key"),
            expected_size=version.get("size_bytes"),
            expected_sha256=version.get("content_sha256"),
            originals_root=ORIGINALS_ROOT,
        )
    except ImmutableSourceUnavailableError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_404_NOT_FOUND,
            "EVIDENCE_SOURCE_UNAVAILABLE",
            "引用原件已不可用。",
        ) from exc
    except ImmutableSourceIntegrityError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "SOURCE_OBJECT_INTEGRITY_ERROR",
            "引用原件与不可变文档版本不一致。",
        ) from exc
    media_type = str(
        version.get("mime_type")
        or guess_type(immutable_source.path.name)[0]
        or "application/octet-stream"
    )
    return immutable_source.path, media_type


def _public_message_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """只回放终态公开字段，防止异常旧记录夹带 prompt/context 正文。"""

    return {
        key: item for key, item in value.items() if key in _PUBLIC_RESULT_FIELDS and key != "answer"
    }


def _provider_identity(url: str) -> tuple[str, str]:
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    provider = host or "configured-openai-compatible"
    return provider, host


def _local_generator(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _knowledge_base_classification(
    *,
    tenant_id: str,
    kb_id: str,
) -> DataClassification:
    with session_scope() as session:
        raw = session.scalar(
            select(KnowledgeBaseModel.data_classification).where(
                KnowledgeBaseModel.id == kb_id,
                KnowledgeBaseModel.tenant_id == tenant_id,
                KnowledgeBaseModel.status == "active",
            )
        )
    if raw is None:
        raise QueryRuntimeApiError(
            status.HTTP_403_FORBIDDEN,
            "RETRIEVAL_SCOPE_DENIED",
            "请求的知识库或文档范围不可用。",
        )
    try:
        return DataClassification(str(raw).lower())
    except ValueError:
        # 未识别的新分级必须按最高敏感度收敛，不能默认为 internal。
        return DataClassification.RESTRICTED


def _external_request(
    *,
    requested: bool,
    provider: str,
    purpose: str,
    scope: AuthorizedRetrievalScope,
    classification,
    payload: str,
    target_domain: str | None,
) -> ExternalDataRequest:
    return ExternalDataRequest(
        requested=requested,
        provider=provider,
        purpose=purpose,
        scope_hash=scope.scope_hash,
        data_classes=(classification,),
        target_domain=target_domain or None,
        contains_direct_identifier=bool(_DIRECT_IDENTIFIER.search(payload)),
        contains_secret=bool(_SECRET_ASSIGNMENT.search(payload)),
    )


def _safe_decision(decision: ExternalDataDecision) -> dict[str, Any]:
    return decision.model_dump(mode="json")


def _freeze_vector_read_set(
    scope: AuthorizedRetrievalScope,
) -> tuple[list[dict[str, Any]], str]:
    """把逻辑 active binding 精确解析成 QueryRun 不可变物理读集。"""

    resolved, rows, digest = freeze_vector_read_set(
        RetrievalScope(
            tenant_id=scope.tenant_id,
            user_id=scope.principal_id,
            kb_ids=scope.allowed_kb_ids,
        )
    )
    expected = {
        (
            scope.tenant_id,
            item.kb_id,
            item.document_id,
            item.document_version_id,
            item.index_version_id,
        )
        for item in scope.active_indexes
    }
    actual = {
        (
            item.tenant_id,
            item.kb_id,
            item.document_id,
            item.document_version_id,
            item.index_version_id,
        )
        for item in resolved
    }
    if actual != expected or len(resolved) != len(expected):
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "物理 vector read set 与冻结的活动索引不一致",
            retryable=False,
        )
    return rows, digest


def _validate_frozen_vector_read_set(
    *,
    scope: AuthorizedRetrievalScope,
    rows: list[dict[str, Any]],
    digest: str,
) -> None:
    resolver = FrozenActiveScopeResolver(rows, expected_digest=digest)
    resolved = resolver.resolve(
        RetrievalScope(
            tenant_id=scope.tenant_id,
            user_id=scope.principal_id,
            kb_ids=scope.allowed_kb_ids,
        )
    )
    expected = {
        (
            scope.tenant_id,
            item.kb_id,
            item.document_id,
            item.document_version_id,
            item.index_version_id,
        )
        for item in scope.active_indexes
    }
    actual = {
        (
            item.tenant_id,
            item.kb_id,
            item.document_id,
            item.document_version_id,
            item.index_version_id,
        )
        for item in resolved
    }
    if actual != expected or len(resolved) != len(expected):
        raise RetrievalEngineError(
            "RETRIEVAL_STALE_SCOPE",
            "冻结的物理 vector read set 与授权索引身份不一致",
            retryable=False,
        )


def _freeze_execution(
    *,
    principal: Principal,
    payload: QueryRunAcceptRequest,
) -> _FrozenExecution:
    requested = RequestedRetrievalScope(
        kb_ids=payload.kb_ids,
        documents=payload.documents,
        business_filter=payload.business_filter,
    )
    authorization = _authorization_service(principal)
    try:
        scope = authorization.authorize(
            principal=principal,
            requested=requested,
            workspace_id=payload.workspace_id,
        )
    except ScopeAuthorizationError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_403_FORBIDDEN,
            "RETRIEVAL_SCOPE_DENIED",
            "请求的知识库或文档范围不可用。",
        ) from exc
    except EntitlementUnavailableError as exc:
        raise QueryRuntimeApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ENTITLEMENT_UNAVAILABLE",
            "查询授权事实暂时不可用，请稍后重试。",
        ) from exc

    # 当前 LangGraph 和检索节点仍以单个 kb_id 执行。显式拒绝比静默只取第一个
    # KB 更安全；领域模型和持久模型已经为后续多 KB 执行保留了扩展空间。
    if len(scope.allowed_kb_ids) != 1:
        raise QueryRuntimeApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "QUERY_MULTI_KB_RUNTIME_UNSUPPORTED",
            "当前在线查询运行一次仅支持一个知识库。",
        )
    if not scope.active_indexes:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_NO_ACTIVE_INDEX",
            "授权范围内没有可查询的已发布索引。",
        )
    if (
        scope.allowed_document_ids is not None
        and len(scope.allowed_document_ids) > FILTER_MAX_VALUES
    ):
        raise QueryRuntimeApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "QUERY_DOCUMENT_SCOPE_TOO_LARGE",
            f"单次显式文档范围最多支持 {FILTER_MAX_VALUES} 个文档。",
        )

    try:
        vector_read_scopes, vector_read_set_digest = _freeze_vector_read_set(scope)
    except RetrievalEngineError as exc:
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if exc.retryable or exc.code == "RETRIEVAL_UNAVAILABLE"
            else status.HTTP_409_CONFLICT
        )
        raise QueryRuntimeApiError(
            response_status,
            exc.code,
            "无法冻结本次查询的物理索引读集，请刷新后重试。",
        ) from exc

    kb_id = scope.allowed_kb_ids[0]
    with session_scope() as session:
        profile_repository = RetrievalProfileRepository(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
        )
        profile = profile_repository.profile_dict(profile_repository.resolve_for_kb(kb_id))

    data_classification = _knowledge_base_classification(
        tenant_id=principal.tenant_id,
        kb_id=kb_id,
    )
    _configured_provider, llm_host = _provider_identity(settings.openai_base_url)
    llm_provider = "configured-llm"
    if not llm_host or not settings.llm_default_model:
        raise QueryRuntimeApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "QUERY_GENERATOR_UNAVAILABLE",
            "没有配置可用的回答生成器。",
        )

    llm_is_local = _local_generator(llm_host)
    if llm_is_local:
        llm_policy = EgressPolicySnapshot(
            policy_version="egress.local-generator.v1",
            enabled=True,
            allowed_providers=(llm_provider,),
            allowed_purposes=("llm_generation",),
            allowed_domains=(llm_host,),
            max_classification=data_classification,
            redact_direct_identifiers=False,
            retention_policy="local-no-egress",
        )
        decision = authorization.decide_egress(
            request=_external_request(
                requested=True,
                provider=llm_provider,
                purpose="llm_generation",
                scope=scope,
                classification=data_classification,
                # 本地生成器不发生数据外发，外发专用的 identifier/secret
                # 拦截不应阻断本地处理；内容安全管线仍会独立检查原文。
                payload="",
                target_domain=llm_host,
            ),
            policy=llm_policy,
        )
        execution_query = payload.query
    else:
        llm_policy = resolve_egress_policy(
            tenant_id=principal.tenant_id,
            kb_ids=scope.allowed_kb_ids,
            purpose="llm_generation",
            allowed_providers=(llm_provider,),
            allowed_domains=(llm_host,),
        )
        decision = authorization.decide_egress(
            request=_external_request(
                requested=True,
                provider=llm_provider,
                purpose="llm_generation",
                scope=scope,
                classification=data_classification,
                payload=payload.query,
                target_domain=llm_host,
            ),
            policy=llm_policy,
        )
        if not decision.allowed:
            raise QueryRuntimeApiError(
                status.HTTP_403_FORBIDDEN,
                "EXTERNAL_LLM_POLICY_DENIED",
                "该知识库当前不允许向已配置的外部模型发送数据。",
                details={"reason_code": decision.reason_code},
            )
        execution_query = redact_external_payload(payload.query, decision)
    llm_decision = _safe_decision(decision)

    _configured_web_provider, web_host = _provider_identity(settings.mcp_dashscope_base_url)
    web_provider = "dashscope-web"
    web_policy = resolve_egress_policy(
        tenant_id=principal.tenant_id,
        kb_ids=scope.allowed_kb_ids,
        purpose="web_search",
        allowed_providers=(web_provider,),
        allowed_domains=settings.query_web_allowed_domains,
    )
    web_decision = authorization.decide_egress(
        request=_external_request(
            requested=payload.web_search_requested,
            provider=web_provider,
            purpose="web_search",
            scope=scope,
            classification=data_classification,
            payload=payload.query,
            # MCP 是受控 provider；搜索结果 URL 在 Evidence 安全阶段按
            # QUERY_WEB_ALLOWED_DOMAINS 逐条验证。
            target_domain=None,
        ),
        policy=web_policy,
    )
    web_search_allowed = bool(payload.web_search_requested and web_decision.allowed)
    content_security_policy = ContentSecurityPolicy(
        policy_version="rag-security.query-runtime.v1",
        web_enabled=web_search_allowed,
        high_risk_sources_enabled=web_search_allowed,
        allowed_fetch_domains=settings.query_web_allowed_domains,
        allowed_output_link_domains=settings.query_output_link_allowed_domains,
    )

    index_snapshot = {
        "schema_version": "query-index-snapshot.v2",
        "scope_hash": scope.scope_hash,
        "active_indexes": [item.model_dump(mode="json") for item in scope.active_indexes],
        "vector_read_scopes": vector_read_scopes,
        "vector_read_set_digest": vector_read_set_digest,
    }
    config_snapshot = {
        "schema_version": "query-execution-config.v1",
        "config_version": settings.config_version,
        # 与索引/Profile 一样在接受运行时冻结，执行线程不得重新解释进程默认值。
        "llm_model_id": settings.llm_default_model,
        "authorized_scope": scope.model_dump(mode="json"),
        "retrieval_profile": profile,
        "data_classification": data_classification.value,
        "external_llm_allowed": bool(decision.allowed),
        "egress_policies": {
            "llm_generation": llm_policy.model_dump(mode="json"),
            "web_search": web_policy.model_dump(mode="json"),
        },
        "content_security_policy": content_security_policy.model_dump(mode="json"),
        "vector_read_set_digest": vector_read_set_digest,
        "egress": {
            "llm": llm_decision,
            "web": _safe_decision(web_decision),
            "decisions": {
                "llm_generation": llm_decision,
                "web_search": _safe_decision(web_decision),
            },
            "llm_local": llm_is_local,
            "web_search_allowed": web_search_allowed,
        },
    }
    return _FrozenExecution(
        scope=scope,
        profile=profile,
        index_snapshot=index_snapshot,
        config_snapshot=config_snapshot,
        execution_query=execution_query,
    )


def _scope_summary(scope: AuthorizedRetrievalScope) -> AuthorizedScopeSummary:
    return AuthorizedScopeSummary(
        scope_hash=scope.scope_hash,
        workspace_id=scope.workspace_id,
        kb_ids=scope.allowed_kb_ids,
        document_restricted=scope.allowed_document_ids is not None,
        active_index_count=len(scope.active_indexes),
        policy_version=scope.policy_version,
        entitlement_version=scope.entitlement_version,
    )


def _profile_identity(profile: dict[str, Any]) -> RetrievalProfileIdentity:
    return RetrievalProfileIdentity(
        profile_id=str(profile["profile_id"]),
        version=int(profile["version"]),
        config_hash=str(profile["config_hash"]),
    )


def _accepted_response(run: QueryRun, *, created: bool) -> QueryRunAcceptedResponse:
    snapshot = dict(run.config_snapshot or {})
    try:
        scope = AuthorizedRetrievalScope.model_validate(snapshot["authorized_scope"])
        profile = dict(snapshot["retrieval_profile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_SNAPSHOT_INVALID",
            "查询运行的冻结配置不可重放。",
        ) from exc
    base = f"/api/v1/query-runs/{run.query_run_id}"
    return QueryRunAcceptedResponse(
        conversation_id=run.conversation_id,
        turn_id=run.turn_id,
        query_run_id=run.query_run_id,
        attempt=run.attempt,
        status=run.status,
        revision=run.revision,
        created=created,
        events_url=f"{base}/events",
        stream_url=f"{base}/events/stream",
        result_url=f"{base}/result",
        authorized_scope=_scope_summary(scope),
        retrieval_profile=_profile_identity(profile),
    )


def _effective_idempotency(principal: Principal, raw_key: str) -> str:
    return canonical_hash(
        {
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "key": raw_key,
        }
    )


def _idempotency_payload_hash(
    *,
    operation: str,
    principal: Principal,
    payload: dict[str, Any],
    frozen: _FrozenExecution,
    source_query_run_id: str | None = None,
) -> str:
    return canonical_hash(
        {
            "operation": operation,
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "source_query_run_id": source_query_run_id,
            "request": payload,
            "scope_hash": frozen.scope.scope_hash,
            "config_version": frozen.config_snapshot["config_version"],
            "retrieval_profile": {
                "profile_id": frozen.profile["profile_id"],
                "version": frozen.profile["version"],
                "config_hash": frozen.profile["config_hash"],
            },
            "index_snapshot": frozen.index_snapshot,
        }
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _existing_idempotent_run(
    session,
    *,
    principal: Principal,
    persisted_key: str,
    payload_hash: str,
) -> QueryRun | None:
    key_hash = canonical_hash(persisted_key)
    row = session.scalar(
        select(QueryIdempotencyKeyModel).where(
            QueryIdempotencyKeyModel.tenant_id == principal.tenant_id,
            QueryIdempotencyKeyModel.key_hash == key_hash,
        )
    )
    if row is None or _aware(row.expires_at) <= datetime.now(UTC):
        return None
    if row.payload_hash != payload_hash:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "IDEMPOTENCY_KEY_CONFLICT",
            "同一 Idempotency-Key 不能用于不同的查询请求。",
        )
    return _runtime_repository(session, principal).get_run(row.query_run_id)


def _request_filter(scope: AuthorizedRetrievalScope) -> FilterNode | None:
    resource = scope.resources[0]
    document_filter: FilterPredicate | None = None
    if resource.document_ids is not None:
        document_filter = FilterPredicate(
            field="document_id",
            operator="in",
            value=tuple(resource.document_ids),
        )
    if scope.business_filter is not None and document_filter is not None:
        return FilterGroup(
            operator="and",
            conditions=(scope.business_filter, document_filter),
        )
    return scope.business_filter or document_filter


def _sanitize_history(
    messages,
    *,
    llm_decision: dict[str, Any],
    local_generator: bool,
) -> list[dict[str, Any]]:
    decision: ExternalDataDecision | None = None
    if not local_generator:
        decision = ExternalDataDecision.model_validate(llm_decision)
    output: list[dict[str, Any]] = []
    for message in messages:
        content = message.content
        if decision is not None:
            content = redact_external_payload(content, decision)
        item = {
            "message_id": message.message_id,
            "turn_id": message.turn_id,
            "role": message.role,
            "text": content,
            "committed_at": message.committed_at.isoformat(),
        }
        if message.role == "user":
            # 旧查询改写服务读取 rewritten_query；SQL 历史没有必要另存一个未经
            # 验证的改写结果，因此以已提交用户正文作为稳定兼容值。
            item["rewritten_query"] = content
        output.append(item)
    return output


def _profile_config(profile: dict[str, Any]) -> dict[str, Any]:
    config = profile.get("config")
    if not isinstance(config, dict):
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_PROFILE_INVALID",
            "冻结的检索配置不是有效对象。",
        )
    serialized = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual_hash = sha256(serialized.encode("utf-8")).hexdigest()
    if actual_hash != str(profile.get("config_hash") or ""):
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_PROFILE_HASH_MISMATCH",
            "冻结的检索配置完整性校验失败。",
        )
    return config


def _prompt_bundle_version() -> str:
    prompt_root = Path(__file__).resolve().parents[3] / "app" / "resources" / "prompts"
    payload: list[tuple[str, str]] = []
    for path in sorted(prompt_root.glob("*.prompt")):
        payload.append((path.name, sha256(path.read_bytes()).hexdigest()))
    return canonical_hash(payload)


def _query_fingerprint(
    *,
    principal: Principal,
    run: QueryRun,
    history: ConversationHistorySnapshot,
    scope: AuthorizedRetrievalScope,
    profile: dict[str, Any],
    original_query: str,
) -> QueryFingerprint:
    config = dict(run.config_snapshot or {})
    egress = dict(config.get("egress") or {})
    frozen_llm_model = str(config.get("llm_model_id") or settings.llm_default_model or "")
    return QueryFingerprint.from_request(
        tenant_id=principal.tenant_id,
        principal_scope={
            "user_id": principal.user_id,
            "roles": sorted(principal.roles),
            "scopes": sorted(principal.scopes),
            "auth_type": principal.auth_type,
        },
        normalized_query=original_query,
        conversation_history=history.model_dump(mode="json"),
        authorized_scope=scope.model_dump(mode="json"),
        filters=scope.business_filter.model_dump(mode="json")
        if scope.business_filter is not None
        else {},
        index_versions=run.index_snapshot,
        query_profile_version=run.profile_version,
        planner_version="query-planner.v1",
        embedding_model_version=f"{settings.bge_m3}:{settings.config_version}",
        rerank_model_version=(f"{settings.bge_reranker_large}:{settings.config_version}"),
        generator_model_version=(
            f"configured-llm:{frozen_llm_model}:{settings.openai_api_key_version}"
        ),
        prompt_version=_prompt_bundle_version(),
        policy_version=canonical_hash(
            {
                "scope_policy": scope.policy_version,
                "entitlement_version": scope.entitlement_version,
                "egress_policies": config.get("egress_policies"),
                "content_security_policy": config.get("content_security_policy"),
            }
        ),
        web_snapshot_policy=(
            "disabled"
            if not egress.get("web_search_allowed")
            else canonical_hash(config.get("egress_policies", {}).get("web_search"))
        ),
    )


def _cacheable_query(run: QueryRun) -> bool:
    """Proof v2 尚不支持原子重签名，因此回答缓存必须整体旁路。"""

    del run
    return False


def _cache_value(public_result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "answer",
        "answer_envelope",
        "citations",
        "sources",
        "citation_validation",
        "image_urls",
        "evidence_assessment",
        "answer_decision",
        "grounding",
        "query_understanding",
        "query_plan",
        "context_pack_summary",
    }
    value = {key: public_result[key] for key in allowed if key in public_result}
    return {
        "schema_version": "query-answer-cache.v1",
        "result": value,
    }


def _rebind_cached_result(
    *,
    cached: dict[str, Any],
    run: QueryRun,
) -> dict[str, Any]:
    if cached.get("schema_version") != "query-answer-cache.v1":
        raise ResourceConflictError("query cache schema is unsupported")
    raw_result = cached.get("result")
    if not isinstance(raw_result, dict):
        raise ResourceConflictError("query cache result is invalid")
    result = dict(raw_result)
    answer_id = f"ans_{uuid4().hex}"
    raw_envelope = result.get("answer_envelope")
    if isinstance(raw_envelope, dict) and raw_envelope:
        envelope = {
            **raw_envelope,
            "answer_id": answer_id,
            "query_run_id": run.query_run_id,
            "session_id": run.conversation_id,
            "message_id": answer_id,
            "trace_id": run.trace_id,
            "usage": {
                "status": "unavailable",
                "reason": "answer cache hit; provider was not called",
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
        result["answer_envelope"] = AnswerEnvelope.model_validate(envelope).model_dump(mode="json")
    result["answer_id"] = answer_id
    result["usage"] = {
        "status": "unavailable",
        "reason": "answer cache hit; provider was not called",
    }
    return result


_EXECUTION_STAGE_ORDER = (
    QueryRunStatus.ACCEPTED,
    QueryRunStatus.PLANNING,
    QueryRunStatus.RETRIEVING,
    QueryRunStatus.PACKING,
    QueryRunStatus.GENERATING,
    QueryRunStatus.VALIDATING,
)


def _advance_to_validating(
    repository: QueryRuntimeRepository,
    run: QueryRun,
) -> QueryRun:
    if run.status in RUN_TERMINAL_STATUSES:
        return run
    if run.status == QueryRunStatus.CANCEL_REQUESTED:
        raise QueryCancellationRequested(run.query_run_id)
    try:
        current_index = _EXECUTION_STAGE_ORDER.index(run.status)
    except ValueError as exc:
        raise ResourceConflictError("query run cannot enter cached completion") from exc
    for target in _EXECUTION_STAGE_ORDER[current_index + 1 :]:
        run = repository.transition_run(
            run.query_run_id,
            expected_revision=run.revision,
            target=target,
            stage=target.value,
        )
    return run


def _event_stream_hash(
    repository: QueryRuntimeRepository,
    query_run_id: str,
) -> str:
    after_seq = 0
    events: list[dict[str, Any]] = []
    while True:
        page = repository.read_event_page(
            query_run_id,
            after_seq=after_seq,
            limit=1000,
        )
        events.extend(
            {
                "event_id": item.event_id,
                "seq": item.seq,
                "type": item.type.value,
                "stage": item.stage,
                "size_bytes": item.size_bytes,
            }
            for item in page.events
        )
        if not page.events or page.terminal:
            break
        after_seq = page.events[-1].seq
    return canonical_hash(events)


def _replay_proof_projection(state: dict[str, Any]) -> dict[str, Any]:
    """重验并提取恢复交付所需的完整 proof 状态。"""

    missing = [key for key in _REPLAY_PROOF_STATE_FIELDS if key not in state]
    if missing:
        raise AnswerProofNotCommittableError(
            "replay checkpoint is missing proof state fields: " + ",".join(missing)
        )
    validate_committable_answer(state)
    raw_proof = state.get("answer_proof")
    raw_closure = state.get("answer_review_closure_receipt")
    if not isinstance(raw_proof, dict) or not isinstance(raw_closure, dict):
        raise AnswerProofNotCommittableError(
            "replay checkpoint proof or semantic review closure is malformed"
        )
    proof_closure = raw_proof.get("semantic_review_closure")
    if raw_closure:
        if canonical_hash(proof_closure) != canonical_hash(raw_closure):
            raise AnswerProofNotCommittableError(
                "replay checkpoint semantic review closure is not bound to proof"
            )
    elif proof_closure not in (None, {}):
        raise AnswerProofNotCommittableError(
            "replay checkpoint proof has an untracked semantic review closure"
        )
    encoded = jsonable_encoder(
        {key: state[key] for key in _REPLAY_PROOF_STATE_FIELDS},
        exclude_none=False,
    )
    if not isinstance(encoded, dict):
        raise AnswerProofNotCommittableError("replay proof state is not portable JSON")
    return encoded


def _checkpoint_encryption_context(
    principal: Principal,
    checkpoint,
) -> QueryCheckpointEncryptionContext:
    return QueryCheckpointEncryptionContext(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        query_run_id=checkpoint.query_run_id,
        sequence=checkpoint.sequence,
        completed_node=checkpoint.completed_node,
        next_node=checkpoint.next_node,
        checkpoint_namespace=checkpoint.checkpoint_namespace,
        workflow_version=checkpoint.workflow_version,
        state_schema_version=checkpoint.state_schema_version,
        checkpoint_schema_version=checkpoint.checkpoint_schema_version,
    )


def _load_final_proof_checkpoint(
    *,
    session,
    principal: Principal,
    query_run_id: str,
) -> tuple[dict[str, Any], _ReplayProofCheckpointBinding]:
    repository = QueryCheckpointRepository(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
    )
    execution = repository.get_execution(query_run_id)
    checkpoint = repository.latest_checkpoint(query_run_id)
    if (
        execution.status != QueryExecutionStatus.COMPLETED
        or execution.next_node is not None
        or checkpoint.next_node is not None
        or checkpoint.sequence < 1
    ):
        raise ValueError("replay requires a completed final proof checkpoint")
    execution_versions = (
        execution.checkpoint_namespace,
        execution.workflow_version,
        execution.state_schema_version,
        execution.checkpoint_schema_version,
        execution.checkpoint_sequence,
    )
    checkpoint_versions = (
        checkpoint.checkpoint_namespace,
        checkpoint.workflow_version,
        checkpoint.state_schema_version,
        checkpoint.checkpoint_schema_version,
        checkpoint.sequence,
    )
    if execution_versions != checkpoint_versions:
        raise ValueError("replay execution and checkpoint versions are inconsistent")
    state = _checkpoint_cipher().decrypt_state(
        EncryptedQueryCheckpoint(
            nonce=checkpoint.nonce,
            ciphertext=checkpoint.ciphertext,
            ciphertext_sha256=checkpoint.ciphertext_sha256,
            size_bytes=checkpoint.size_bytes,
        ),
        context=_checkpoint_encryption_context(principal, checkpoint),
    )
    projection = _replay_proof_projection(state)
    return state, _ReplayProofCheckpointBinding(
        query_run_id=query_run_id,
        sequence=checkpoint.sequence,
        ciphertext_sha256=checkpoint.ciphertext_sha256,
        proof_projection_sha256=canonical_hash(projection),
    )


def _restore_replay_proof_state(
    *,
    session,
    principal: Principal,
    manifest: QueryReplayManifest,
) -> dict[str, Any]:
    """从 manifest 锁定的密文 checkpoint 恢复并重验 proof 图。"""

    expected = _ReplayProofCheckpointBinding.from_manifest(manifest)
    state, actual = _load_final_proof_checkpoint(
        session=session,
        principal=principal,
        query_run_id=manifest.query_run_id,
    )
    if actual != expected:
        raise ValueError("replay manifest proof checkpoint binding is stale or tampered")
    public_state = _public_result(state)
    if canonical_hash(public_state.get("answer_envelope")) != canonical_hash(
        manifest.answer_envelope
    ):
        raise ValueError("replay manifest answer envelope differs from proof checkpoint")
    if canonical_hash(public_state.get("query_plan")) != canonical_hash(manifest.query_plan):
        raise ValueError("replay manifest query plan differs from proof checkpoint")
    route_summary_hash = canonical_hash(
        {
            "query_plan": public_state.get("query_plan"),
            "query_understanding": public_state.get("query_understanding"),
            "sources": public_state.get("sources"),
        }
    )
    if route_summary_hash not in manifest.route_result_refs:
        raise ValueError("replay manifest route summary differs from proof checkpoint")
    return state


def _restore_replay_proof_state_or_error(
    *,
    session,
    principal: Principal,
    manifest: QueryReplayManifest,
) -> dict[str, Any]:
    try:
        return _restore_replay_proof_state(
            session=session,
            principal=principal,
            manifest=manifest,
        )
    except (
        AnswerProofNotCommittableError,
        KeyError,
        QueryExecutionNotInitialized,
        ResourceConflictError,
        TypeError,
        ValueError,
    ) as exc:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_REPLAY_PROOF_INVALID",
            "查询重放证明缺失、损坏或与最终检查点不一致。",
        ) from exc


def _save_replay_manifest(
    *,
    session,
    principal: Principal,
    run: QueryRun,
    history: ConversationHistorySnapshot,
    fingerprint: QueryFingerprint,
    result: dict[str, Any],
) -> None:
    checkpoint_state, checkpoint_binding = _load_final_proof_checkpoint(
        session=session,
        principal=principal,
        query_run_id=run.query_run_id,
    )
    checkpoint_public_result = _public_result(checkpoint_state)
    for key in ("answer_id", "answer_proof", "answer_envelope", "citations"):
        if canonical_hash(result.get(key)) != canonical_hash(checkpoint_public_result.get(key)):
            raise ValueError(f"public result {key} differs from final proof checkpoint")
    context_summary = checkpoint_public_result.get("context_pack_summary")
    context_summary = context_summary if isinstance(context_summary, dict) else {}
    context_hash = str(context_summary.get("content_hash") or "")
    route_summary = {
        "query_plan": checkpoint_public_result.get("query_plan"),
        "query_understanding": checkpoint_public_result.get("query_understanding"),
        "sources": checkpoint_public_result.get("sources"),
    }
    repository = _runtime_repository(session, principal)
    manifest = QueryReplayManifest(
        schema_version=_REPLAY_MANIFEST_SCHEMA_VERSION,
        query_run_id=run.query_run_id,
        fingerprint=fingerprint,
        request_hash=canonical_hash(
            {
                "turn_id": run.turn_id,
                "history_snapshot_hash": history.content_hash,
                "normalized_query_hash": fingerprint.normalized_query_hash,
            }
        ),
        history_snapshot_ref=(f"query-run://{run.query_run_id}/history/{history.content_hash}"),
        query_plan=dict(checkpoint_public_result.get("query_plan") or {}),
        route_result_refs=(canonical_hash(route_summary), checkpoint_binding.to_ref()),
        context_pack_ref=(
            f"query-run://{run.query_run_id}/context/{context_hash}" if context_hash else None
        ),
        answer_envelope=dict(checkpoint_public_result.get("answer_envelope") or {}),
        event_stream_hash=_event_stream_hash(repository, run.query_run_id),
        external_snapshot_refs=(),
    )
    QueryReliabilityRepository(
        session,
        tenant_id=principal.tenant_id,
    ).save_replay_manifest(
        manifest,
        ttl=timedelta(days=settings.query_replay_retention_days),
    )


def _cached_completion(
    *,
    principal: Principal,
    query_run_id: str,
    cached: dict[str, Any],
    fingerprint: QueryFingerprint,
    cache_role: str,
    execution_lease: QueryExecutionLease,
    execution_preflight: DurableQueryPreflight,
) -> bool:
    if (
        execution_preflight.lease != execution_lease
        or execution_preflight.execution.query_run_id != query_run_id
    ):
        raise QueryCheckpointCompatibilityError(
            "cache completion requires the current durable preflight receipt"
        )

    completed: QueryRun
    history: ConversationHistorySnapshot
    result: dict[str, Any]
    with session_scope() as session:
        repository = _runtime_repository(session, principal)
        QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).assert_lease(execution_lease)
        run = repository.get_run(query_run_id)
        if run.status in RUN_TERMINAL_STATUSES:
            _reconcile_terminal_execution(
                session,
                principal=principal,
                query_run_id=query_run_id,
            )
            return True
        if run.status == QueryRunStatus.CANCEL_REQUESTED:
            repository.finalize_cancel(
                query_run_id,
                expected_revision=run.revision,
            )
            _mark_idempotency_complete(
                session,
                principal=principal,
                query_run_id=query_run_id,
                response_status=status.HTTP_409_CONFLICT,
            )
            _reconcile_terminal_execution(
                session,
                principal=principal,
                query_run_id=query_run_id,
            )
            return True
        if _deadline_expired(run):
            repository.fail_run(
                query_run_id,
                expected_revision=run.revision,
                error_code="QUERY_TIMED_OUT",
                safe_message="查询在 cache 终结前超过了服务端截止时间。",
                retryable=True,
                error_detail_ref=f"trace:{run.trace_id}",
                timed_out=True,
            )
            _mark_idempotency_complete(
                session,
                principal=principal,
                query_run_id=query_run_id,
                response_status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
            _reconcile_terminal_execution(
                session,
                principal=principal,
                query_run_id=query_run_id,
            )
            return True
        history = repository.get_history_snapshot(query_run_id)
        result = _rebind_cached_result(cached=cached, run=run)
        result["cache"] = {
            "status": "hit",
            "role": cache_role,
            "fingerprint": fingerprint.key,
        }
        run = _advance_to_validating(repository, run)
        repository.append_event(
            query_run_id,
            event_type=QueryEventType.RETRIEVAL_SUMMARY,
            stage=QueryRunStatus.VALIDATING.value,
            data={
                "cache": "hit",
                "singleflight_role": cache_role,
                "fingerprint": fingerprint.key,
            },
        )
        answer = str(result.get("answer") or "")
        _emit_answer_events(repository, query_run_id, answer)
        completed = repository.complete_run(
            query_run_id,
            expected_revision=run.revision,
            answer=answer,
            result=result,
        )
        _reconcile_terminal_execution(
            session,
            principal=principal,
            query_run_id=query_run_id,
        )
        _mark_idempotency_complete(
            session,
            principal=principal,
            query_run_id=query_run_id,
            response_status=status.HTTP_200_OK,
        )
    try:
        with session_scope() as session:
            _save_replay_manifest(
                session=session,
                principal=principal,
                run=completed,
                history=history,
                fingerprint=fingerprint,
                result=result,
            )
    except Exception:
        # 重放快照是诊断增强，持久化异常不能回滚已经提交的正确查询终态。
        logger.error(
            "query durability auxiliary write failed: code={} stage={} query_run_id={}",
            "QUERY_REPLAY_MANIFEST_WRITE_FAILED",
            "cache_hit",
            query_run_id,
        )
    return True


def _abandon_singleflight(
    *,
    principal: Principal,
    query_run_id: str,
    fingerprint: QueryFingerprint | None,
    lease: dict[str, Any] | None,
) -> None:
    if fingerprint is None or not lease or not lease.get("lease_token"):
        return
    try:
        with session_scope() as session:
            QueryReliabilityRepository(
                session,
                tenant_id=principal.tenant_id,
            ).abandon_singleflight(
                fingerprint=fingerprint.key,
                owner_id=query_run_id,
                lease_token=str(lease["lease_token"]),
                fencing_token=int(lease["fencing_token"]),
            )
    except Exception:
        logger.error(
            "query durability auxiliary write failed: code={} stage={} query_run_id={}",
            "QUERY_SINGLEFLIGHT_RELEASE_FAILED",
            "release",
            query_run_id,
        )


def _public_result(state: dict[str, Any]) -> dict[str, Any]:
    encoded = jsonable_encoder(
        {key: state[key] for key in _PUBLIC_RESULT_FIELDS if key in state},
        exclude_none=True,
    )
    if not isinstance(encoded, dict):
        return {}
    # 防御式删除：即使未来 allowlist 被误扩展，也不允许 Prompt/ContextPack 正文
    # 进入终态结果或 Message metadata。
    encoded.pop("prompt", None)
    encoded.pop("context_pack", None)
    encoded.pop("embedding_chunks", None)
    encoded.pop("hyde_embedding_chunks", None)
    encoded.pop("reranked_docs", None)
    return encoded


def _emit_answer_events(
    repository: QueryRuntimeRepository,
    query_run_id: str,
    answer: str,
) -> None:
    if not answer:
        return
    chunk_size = 8_000
    for index in range(0, len(answer), chunk_size):
        repository.append_event(
            query_run_id,
            event_type=QueryEventType.ANSWER_DELTA,
            stage=QueryRunStatus.VALIDATING.value,
            operation_id=f"final-answer:{index // chunk_size:08d}",
            persistence_class="compactable",
            data={
                "operation": "replace" if index == 0 else "append",
                "text": answer[index : index + chunk_size],
            },
        )


def _deadline_expired(run: QueryRun) -> bool:
    return bool(run.deadline_at and _aware(run.deadline_at) <= datetime.now(UTC))


def _default_deadline_seconds() -> int:
    return max(1, (int(settings.query_default_deadline_ms) + 999) // 1000)


def _mark_idempotency_complete(
    session,
    *,
    principal: Principal,
    query_run_id: str,
    response_status: int,
) -> None:
    QueryReliabilityRepository(
        session,
        tenant_id=principal.tenant_id,
    ).complete_idempotency(query_run_id, response_status)


def _reconcile_terminal_execution(
    session,
    *,
    principal: Principal,
    query_run_id: str,
) -> None:
    """在 QueryRun 终态事务内 fence 掉迟到 worker；兼容迁移前旧运行。"""

    try:
        QueryCheckpointRepository(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        ).reconcile_terminal_run(query_run_id)
    except QueryExecutionNotInitialized:
        logger.info(
            "迁移前 QueryRun 没有 durable execution：query_run_id={}",
            query_run_id,
        )


def _cache_or_singleflight(
    *,
    principal: Principal,
    run: QueryRun,
    fingerprint: QueryFingerprint,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    """返回 ``(cached, role, owned_lease)``。

    waiter 仅轮询 SQL cache，取消自己不会触碰 producer；producer 失败时由
    fenced abandon 立即释放。超过 SQL lease 支持的 300 秒截止时间会明确
    bypass singleflight，但仍可安全读取/写入版本化 cache。
    """

    # 必须在 lookup 前判定；否则历史 cache 命中会绕过 proof 身份闭包。
    if not _cacheable_query(run):
        return None, "cache_bypass_answer_proof_identity", None

    scope_hash = fingerprint.authorized_scope_hash

    def cache_lookup() -> tuple[dict[str, Any] | None, str]:
        with session_scope() as session:
            return QueryReliabilityRepository(
                session,
                tenant_id=principal.tenant_id,
            ).get_cache(
                cache_key=fingerprint.key,
                authorized_scope_hash=scope_hash,
            )

    cached, cache_status = cache_lookup()
    if cached is not None:
        return cached, "cache_hit", None
    now = datetime.now(UTC)
    remaining_seconds = (
        max(1, int((_aware(run.deadline_at) - now).total_seconds()))
        if run.deadline_at is not None
        else 120
    )
    if remaining_seconds > 295:
        return None, "singleflight_bypass_long_deadline", None
    lease_seconds = min(
        300,
        max(
            settings.query_singleflight_lease_seconds,
            remaining_seconds + 5,
        ),
    )

    while True:
        with session_scope() as session:
            runtime = _runtime_repository(session, principal)
            current = runtime.get_run(run.query_run_id)
            if current.status in RUN_TERMINAL_STATUSES:
                return None, "run_already_terminal", None
            if current.status == QueryRunStatus.CANCEL_REQUESTED:
                raise QueryCancellationRequested(run.query_run_id)
            if _deadline_expired(current):
                raise TimeoutError("singleflight waiter exceeded query deadline")
            reliability = QueryReliabilityRepository(
                session,
                tenant_id=principal.tenant_id,
            )
            lease = reliability.acquire_singleflight(
                fingerprint=fingerprint.key,
                owner_id=run.query_run_id,
                lease_seconds=lease_seconds,
            )
        if lease.get("acquired"):
            return None, "singleflight_producer", lease

        cached, _cache_status = cache_lookup()
        if cached is not None:
            return cached, "singleflight_waiter", None
        time.sleep(0.2)


def _build_initial_query_state(
    *,
    principal: Principal,
    run: QueryRun,
    conversation,
    turn,
    history: ConversationHistorySnapshot,
    frozen: _FrozenExecution,
) -> dict[str, Any]:
    """构造唯一 initial state；accept/retry 在同事务内立即加密持久化。"""

    from app.process.query.agent.state import create_query_default_state

    snapshot = frozen.config_snapshot
    egress = dict(snapshot["egress"])
    llm_decision = dict(egress["llm"])
    data_classification = DataClassification(str(snapshot["data_classification"]))
    request_filter = _request_filter(frozen.scope)
    return create_query_default_state(
        session_id=conversation.conversation_id,
        conversation_id=conversation.conversation_id,
        turn_id=run.turn_id,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        original_query=frozen.execution_query,
        kb_id=conversation.kb_id,
        web_search_allowed=bool(egress.get("web_search_allowed", False)),
        external_llm_allowed=bool(snapshot.get("external_llm_allowed", False)),
        data_classification=data_classification.value,
        egress_policies=dict(snapshot["egress_policies"]),
        egress_decisions=[
            dict(item)
            for item in (egress.get("decisions") or {}).values()
            if isinstance(item, dict)
        ],
        content_security_policy=dict(snapshot["content_security_policy"]),
        config_version=str(snapshot["config_version"]),
        llm_model_id=str(snapshot.get("llm_model_id") or settings.llm_default_model or ""),
        retrieval_profile=_profile_config(frozen.profile),
        retrieval_profile_id=str(frozen.profile["profile_id"]),
        retrieval_profile_version=int(frozen.profile["version"]),
        retrieval_profile_hash=str(frozen.profile["config_hash"]),
        retrieval_filter=(
            request_filter.model_dump(mode="json") if request_filter is not None else None
        ),
        history=_sanitize_history(
            history.messages,
            llm_decision=llm_decision,
            local_generator=bool(egress.get("llm_local", False)),
        ),
        history_snapshot_loaded=True,
        history_snapshot_hash=history.content_hash,
        history_base_turn_seq=history.base_turn_seq,
        history_version=history.history_version,
        runtime_managed_history=True,
        authorized_scope=frozen.scope.model_dump(mode="json"),
        vector_read_scopes=list(frozen.index_snapshot["vector_read_scopes"]),
        vector_read_set_digest=str(frozen.index_snapshot["vector_read_set_digest"]),
        answer_id=f"ans_{sha256(run.query_run_id.encode()).hexdigest()[:32]}",
        query_run_id=run.query_run_id,
        trace_id=run.trace_id,
        deadline_at=(run.deadline_at.isoformat() if run.deadline_at else None),
        is_stream=False,
    )


def _query_boundary_guard(principal: Principal, query_run_id: str):
    def guard(_state: Any) -> None:
        with session_scope() as session:
            run = _runtime_repository(session, principal).get_run(query_run_id)
            if run.status in {
                QueryRunStatus.CANCEL_REQUESTED,
                QueryRunStatus.CANCELLED,
            }:
                raise QueryBoundaryCancellationRequested("query cancellation was persisted")

    return guard


def _revalidate_frozen_scope(
    *,
    principal: Principal,
    scope: AuthorizedRetrievalScope,
) -> None:
    documents = tuple(
        RequestedDocument(kb_id=resource.kb_id, document_id=document_id)
        for resource in scope.resources
        for document_id in (resource.document_ids or ())
    )
    requested = RequestedRetrievalScope(
        kb_ids=scope.allowed_kb_ids,
        documents=documents,
        business_filter=scope.business_filter,
    )
    try:
        current = _authorization_service(principal).authorize(
            principal=principal,
            requested=requested,
            workspace_id=scope.workspace_id,
        )
    except (ScopeAuthorizationError, EntitlementUnavailableError) as exc:
        raise QueryRuntimeApiError(
            status.HTTP_403_FORBIDDEN,
            "QUERY_RESUME_SCOPE_REVOKED",
            "查询执行前授权范围已失效。",
        ) from exc
    if current.scope_hash != scope.scope_hash:
        raise QueryRuntimeApiError(
            status.HTTP_403_FORBIDDEN,
            "QUERY_RESUME_SCOPE_CHANGED",
            "查询执行前授权或发布索引事实已变化。",
        )


def _finalize_worker_failure(
    *,
    principal: Principal,
    query_run_id: str,
    error_code: str,
    safe_message: str,
    retryable: bool,
    timed_out: bool = False,
    response_status_code: int | None = None,
) -> None:
    try:
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            run = repository.get_run(query_run_id)
            if run.status in RUN_TERMINAL_STATUSES:
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if run.status == QueryRunStatus.CANCEL_REQUESTED:
                repository.finalize_cancel(
                    query_run_id,
                    expected_revision=run.revision,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_409_CONFLICT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            repository.fail_run(
                query_run_id,
                expected_revision=run.revision,
                error_code=error_code,
                safe_message=safe_message,
                retryable=retryable,
                error_detail_ref=f"trace:{run.trace_id}",
                timed_out=timed_out,
            )
            _mark_idempotency_complete(
                session,
                principal=principal,
                query_run_id=query_run_id,
                response_status=(
                    response_status_code
                    if response_status_code is not None
                    else (
                        status.HTTP_504_GATEWAY_TIMEOUT
                        if timed_out
                        else status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                ),
            )
            _reconcile_terminal_execution(
                session,
                principal=principal,
                query_run_id=query_run_id,
            )
    except (
        StaleQueryRevision,
        InvalidStateTransition,
        ResourceConflictError,
        ResourceNotFoundError,
    ):
        # 取消/完成与失败并发时，数据库中的先胜终态是唯一事实；绝不覆盖。
        logger.info(
            "查询运行失败终结丢失 CAS 竞争，保留数据库终态：query_run_id={}",
            query_run_id,
        )


def _execute_query_run(
    query_run_id: str,
    principal: Principal,
    execution_query: str | None,
    *,
    recovery_controller: DurableQueryRecoveryController | None = None,
) -> None:
    """从持久 safe boundary 执行 Query；参数正文仅保留旧调用签名兼容。"""

    _ = execution_query
    principal_token = None
    fingerprint: QueryFingerprint | None = None
    owned_lease: dict[str, Any] | None = None
    lease_finalized = False
    try:
        cipher = _checkpoint_cipher()
        with session_scope() as session:
            execution = QueryCheckpointRepository(
                session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            ).get_execution(query_run_id)
        # 同进程 BackgroundTask 与跨进程恢复共用同一身份复核边界；不信接收时
        # 捕获的 roles/scopes，也不在 credential/membership 撤销后进入 cache/provider。
        principal = recover_query_principal(execution)
        principal_token = set_current_principal(principal)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            run = repository.get_run(query_run_id)
            if run.status in RUN_TERMINAL_STATUSES:
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if run.status == QueryRunStatus.CANCEL_REQUESTED:
                repository.finalize_cancel(
                    query_run_id,
                    expected_revision=run.revision,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_409_CONFLICT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if _deadline_expired(run):
                repository.fail_run(
                    query_run_id,
                    expected_revision=run.revision,
                    error_code="QUERY_TIMED_OUT",
                    safe_message="查询在开始执行前已超过截止时间。",
                    retryable=True,
                    error_detail_ref=f"trace:{run.trace_id}",
                    timed_out=True,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            conversation = repository.get_conversation(run.conversation_id)
            turn = repository.get_turn(run.turn_id)
            history = repository.get_history_snapshot(query_run_id)
            snapshot = dict(run.config_snapshot or {})

        try:
            scope = AuthorizedRetrievalScope.model_validate(snapshot["authorized_scope"])
            profile = dict(snapshot["retrieval_profile"])
            vector_read_scopes = list(run.index_snapshot["vector_read_scopes"])
            vector_read_set_digest = str(run.index_snapshot["vector_read_set_digest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "QUERY_SNAPSHOT_INVALID",
                "查询运行的冻结配置不可重放。",
            ) from exc
        if (
            scope.tenant_id != principal.tenant_id
            or scope.principal_id != principal.user_id
            or scope.allowed_kb_ids != (conversation.kb_id,)
            or snapshot.get("config_version") != settings.config_version
            or snapshot.get("vector_read_set_digest") != vector_read_set_digest
        ):
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "QUERY_SNAPSHOT_SCOPE_MISMATCH",
                "查询运行的冻结身份或配置版本不一致。",
            )
        _validate_frozen_vector_read_set(
            scope=scope,
            rows=vector_read_scopes,
            digest=vector_read_set_digest,
        )
        # 任何 cache、handler 或 provider 前都用当前控制库重新授权冻结 KB/ACL。
        _revalidate_frozen_scope(principal=principal, scope=scope)

        from app.process.query.agent.main_graph import query_node_handlers

        executor = build_durable_query_executor(
            principal=principal,
            cipher=cipher,
            node_handlers=query_node_handlers(),
            lease_owner=(recovery_controller.owner_id if recovery_controller is not None else None),
        )
        if recovery_controller is None:
            execution_lease = executor.claim(query_run_id)
        else:
            with recovery_controller.claim_gate():
                execution_lease = executor.claim(query_run_id)
        execution_preflight = executor.preflight(
            query_run_id=query_run_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            lease=execution_lease,
        )

        fingerprint = _query_fingerprint(
            principal=principal,
            run=run,
            history=history,
            scope=scope,
            profile=profile,
            original_query=turn.user_content,
        )
        cached, cache_role, owned_lease = _cache_or_singleflight(
            principal=principal,
            run=run,
            fingerprint=fingerprint,
        )
        if cached is not None:
            _cached_completion(
                principal=principal,
                query_run_id=query_run_id,
                cached=cached,
                fingerprint=fingerprint,
                cache_role=cache_role,
                execution_lease=execution_lease,
                execution_preflight=execution_preflight,
            )
            return

        hook = QueryRuntimeProgressHook(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            query_run_id=query_run_id,
        )
        with bind_query_runtime_hook(hook):
            durable_result = executor.execute(
                query_run_id=query_run_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                boundary_guard=_query_boundary_guard(principal, query_run_id),
                lease=execution_lease,
                preflight=execution_preflight,
            )
            hook.advance_validating()
        result_state = dict(durable_result.state)
        answer = str(result_state.get("answer") or "")
        public_result = _public_result(result_state)
        public_result["cache"] = {
            "status": "miss",
            "role": cache_role,
            "fingerprint": fingerprint.key,
        }

        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            checkpoint_repository = QueryCheckpointRepository(
                session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
            checkpoint_repository.assert_lease(durable_result.lease)
            current = repository.get_run(query_run_id)
            if current.status in RUN_TERMINAL_STATUSES:
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if current.status == QueryRunStatus.CANCEL_REQUESTED:
                repository.finalize_cancel(
                    query_run_id,
                    expected_revision=current.revision,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_409_CONFLICT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if _deadline_expired(current):
                repository.fail_run(
                    query_run_id,
                    expected_revision=current.revision,
                    error_code="QUERY_TIMED_OUT",
                    safe_message="查询超过了服务端截止时间。",
                    retryable=True,
                    error_detail_ref=f"trace:{current.trace_id}",
                    timed_out=True,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
                return
            if not bool(result_state.get("is_stream", False)):
                # 流式图已经由 QueryRuntimeProgressHook 在输出安全检查后逐段落库；
                # 当前非流执行只补一个可续传的终态答案序列。
                _emit_answer_events(repository, query_run_id, answer)
            # append_event 只推进 event_sequence，不改变 CAS revision。
            completed = repository.complete_run(
                query_run_id,
                expected_revision=current.revision,
                answer=answer,
                result=public_result,
            )
            _reconcile_terminal_execution(
                session,
                principal=principal,
                query_run_id=query_run_id,
            )
            _mark_idempotency_complete(
                session,
                principal=principal,
                query_run_id=query_run_id,
                response_status=status.HTTP_200_OK,
            )

        if _cacheable_query(completed):
            try:
                with session_scope() as session:
                    reliability = QueryReliabilityRepository(
                        session,
                        tenant_id=principal.tenant_id,
                    )
                    stored = reliability.put_cache(
                        cache_key=fingerprint.key,
                        authorized_scope_hash=fingerprint.authorized_scope_hash,
                        layer="answer",
                        value=_cache_value(public_result),
                        versions=fingerprint.model_dump(mode="json"),
                        classification=str(completed.config_snapshot.get("data_classification")),
                        ttl=timedelta(seconds=settings.query_cache_ttl_seconds),
                    )
                    if stored and owned_lease is not None:
                        try:
                            reliability.complete_singleflight(
                                fingerprint=fingerprint.key,
                                owner_id=query_run_id,
                                lease_token=str(owned_lease["lease_token"]),
                                fencing_token=int(owned_lease["fencing_token"]),
                                result_cache_key=fingerprint.key,
                            )
                            lease_finalized = True
                        except ResourceConflictError:
                            # 运行终态与 cache 已安全提交；过期 lease 只影响合并
                            # 收益，不能反向把正确答案改成失败。
                            logger.info(
                                "singleflight 完成时 lease 已失效：query_run_id={}",
                                query_run_id,
                            )
            except Exception:
                # Cache 是性能层；不可用时必须旁路，不能改变业务终态。
                logger.error(
                    "query durability auxiliary write failed: code={} stage={} query_run_id={}",
                    "QUERY_CACHE_WRITE_FAILED",
                    "completed",
                    query_run_id,
                )

        try:
            with session_scope() as session:
                _save_replay_manifest(
                    session=session,
                    principal=principal,
                    run=completed,
                    history=history,
                    fingerprint=fingerprint,
                    result=public_result,
                )
        except Exception:
            logger.error(
                "query durability auxiliary write failed: code={} stage={} query_run_id={}",
                "QUERY_REPLAY_MANIFEST_WRITE_FAILED",
                "completed",
                query_run_id,
            )
    except (QueryCancellationRequested, QueryExecutionCancelled):
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_CANCELLED",
            safe_message="查询已由用户取消。",
            retryable=True,
        )
    except QueryResumeIdentityError as exc:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code=exc.code,
            safe_message=exc.safe_message,
            retryable=False,
            response_status_code=status.HTTP_403_FORBIDDEN,
        )
    except QueryExecutionDeadlineExceeded:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_TIMED_OUT",
            safe_message="查询超过了服务端截止时间。",
            retryable=True,
            timed_out=True,
        )
    except QueryInterruptedDuringNode:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_INTERRUPTED_DURING_NODE",
            safe_message="查询在可能产生外部副作用的节点中断，未自动重放；请安全重试。",
            retryable=True,
        )
    except QueryCheckpointLoadError:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_CHECKPOINT_CORRUPT",
            safe_message="查询 checkpoint 无法验证，请创建安全重试。",
            retryable=True,
        )
    except QueryCheckpointCompatibilityError:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_CHECKPOINT_VERSION_OR_ROUTE_UNSUPPORTED",
            safe_message="查询 checkpoint 与当前工作流不兼容，请创建安全重试。",
            retryable=True,
        )
    except QueryExecutionNotInitialized:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_CHECKPOINT_NOT_INITIALIZED",
            safe_message="查询缺少可验证的初始 checkpoint，请安全重试。",
            retryable=True,
        )
    except (QueryExecutionLeaseBusy, QueryExecutionFenceLost, QueryRecoveryStopped):
        # 另一个恢复者或终态事务已取得更高 fence；迟到 worker 不得改写事实。
        logger.info(
            "可靠查询执行未取得或已失去 fence：query_run_id={}",
            query_run_id,
        )
    except RetrievalEngineError as exc:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code=exc.code,
            safe_message=exc.safe_message,
            retryable=exc.retryable,
        )
    except TimeoutError:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_TIMED_OUT",
            safe_message="查询依赖调用超时，请稍后重试。",
            retryable=True,
            timed_out=True,
        )
    except ScopeAuthorizationError:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="RETRIEVAL_SCOPE_DENIED",
            safe_message="冻结的检索授权范围已失效。",
            retryable=False,
        )
    except QueryContentSecurityViolation:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_CONTENT_SECURITY_BLOCKED",
            safe_message="查询内容触发安全策略，已阻断执行。",
            retryable=False,
        )
    except AnswerProofNotCommittableError as exc:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code=exc.code,
            safe_message=exc.safe_message,
            retryable=exc.retryable,
        )
    except QueryEgressDenied as exc:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_EGRESS_DENIED",
            safe_message=f"外部调用未通过冻结策略：{exc.reason_code}",
            retryable=False,
        )
    except QueryRuntimeApiError as exc:
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code=exc.code,
            safe_message=exc.message,
            retryable=False,
            response_status_code=exc.status_code,
        )
    except RequiredRouteError as exc:
        logger.warning(
            "查询必选路线失败：query_run_id={} route_id={} error_code={}",
            query_run_id,
            exc.route_id,
            exc.error_code,
        )
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="REQUIRED_RETRIEVAL_ROUTE_FAILED",
            safe_message="必选的本地知识库检索失败，请稍后重试。",
            retryable=True,
        )
    except Exception:
        logger.error(
            "query durability execution failed: code={} stage={} query_run_id={}",
            "QUERY_EXECUTION_UNHANDLED",
            "worker",
            query_run_id,
        )
        _finalize_worker_failure(
            principal=principal,
            query_run_id=query_run_id,
            error_code="QUERY_EXECUTION_FAILED",
            safe_message="查询执行失败，请稍后重试。",
            retryable=True,
        )
    finally:
        if owned_lease is not None and not lease_finalized:
            _abandon_singleflight(
                principal=principal,
                query_run_id=query_run_id,
                fingerprint=fingerprint,
                lease=owned_lease,
            )
        if principal_token is not None:
            reset_current_principal(principal_token)


def _retry_payload(
    *,
    run: QueryRun,
    principal: Principal,
    deadline_seconds: int | None,
) -> _FrozenExecution:
    try:
        frozen_scope = AuthorizedRetrievalScope.model_validate(
            run.config_snapshot["authorized_scope"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_SNAPSHOT_INVALID",
            "原查询运行的冻结授权范围不可重放。",
        ) from exc
    documents = tuple(
        RequestedDocument(kb_id=resource.kb_id, document_id=document_id)
        for resource in frozen_scope.resources
        for document_id in (resource.document_ids or ())
    )
    with session_scope() as session:
        repository = _runtime_repository(session, principal)
        turn = repository.get_turn(run.turn_id)
    request = QueryRunAcceptRequest(
        query=turn.user_content,
        kb_ids=frozen_scope.allowed_kb_ids,
        documents=documents,
        business_filter=frozen_scope.business_filter,
        conversation_id=run.conversation_id,
        client_turn_id=turn.client_turn_id,
        workspace_id=frozen_scope.workspace_id,
        web_search_requested=bool(
            (run.config_snapshot.get("egress") or {}).get(
                "web_search_allowed",
                False,
            )
        ),
        deadline_seconds=deadline_seconds or _default_deadline_seconds(),
    )
    current = _freeze_execution(principal=principal, payload=request)
    if current.scope.scope_hash != frozen_scope.scope_hash:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_RETRY_SCOPE_CHANGED",
            "权限、发布索引或策略已变化；请创建一个新的查询轮次。",
        )
    if current.profile_version != run.profile_version:
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_RETRY_PROFILE_CHANGED",
            "检索配置已变化；请创建一个新的查询轮次。",
        )
    if current.config_snapshot["config_version"] != run.config_snapshot.get("config_version"):
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_RETRY_CONFIG_CHANGED",
            "服务配置已变化；请创建一个新的查询轮次。",
        )
    if canonical_hash(current.config_snapshot.get("egress")) != canonical_hash(
        run.config_snapshot.get("egress")
    ):
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_RETRY_EGRESS_CHANGED",
            "外发策略已变化；请创建一个新的查询轮次。",
        )
    if current.index_snapshot.get("vector_read_set_digest") != run.index_snapshot.get(
        "vector_read_set_digest"
    ):
        raise QueryRuntimeApiError(
            status.HTTP_409_CONFLICT,
            "QUERY_RETRY_VECTOR_PLACEMENT_CHANGED",
            "物理索引读集已变化；请创建一个新的查询轮次。",
        )
    return current


def register_query_runtime_routes(app: FastAPI) -> None:
    """注册可靠查询运行 API；重复调用是幂等的。"""

    if getattr(app.state, "query_runtime_routes_registered", False):
        return
    app.state.query_runtime_routes_registered = True

    async def handle_runtime_error(
        request: Request,
        exc: QueryRuntimeApiError,
    ) -> JSONResponse:
        error: dict[str, Any] = {
            "code": exc.code,
            "message": exc.message,
        }
        if exc.details:
            error["details"] = exc.details
        return JSONResponse(
            status_code=exc.status_code,
            content={"request_id": _request_id(request), "error": error},
        )

    app.add_exception_handler(QueryRuntimeApiError, handle_runtime_error)

    @app.get("/workbench/chat", include_in_schema=False)
    @app.get("/workbench/retrieval", include_in_schema=False)
    def query_workbench() -> FileResponse:
        """两个可深链工作台视图共用同一安全前端资源。"""

        html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
        return FileResponse(path=html_path, media_type="text/html; charset=utf-8")

    @app.get(
        "/api/v1/evidence/{evidence_id}/source-inline",
        include_in_schema=False,
    )
    def evidence_source_inline(evidence_id: str, request: Request) -> FileResponse:
        """在浏览器内打开已授权固定版本原件，供 page/bbox 引用核验。"""

        path, media_type = _authorized_inline_source(
            principal=_principal(request),
            evidence_id=evidence_id,
        )
        encoded_name = quote(path.name, safe="")
        return FileResponse(
            path=path,
            media_type=media_type,
            headers={
                "Cache-Control": "private, no-store",
                "Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "SAMEORIGIN",
                "Content-Security-Policy": "default-src 'none'; frame-ancestors 'self'; sandbox",
            },
        )

    @app.post(
        "/api/v1/query-runs",
        response_model=QueryRunAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["可靠查询"],
    )
    def accept_query_run(
        payload: QueryRunAcceptRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            max_length=256,
        ),
    ) -> QueryRunAcceptedResponse:
        principal = _principal(request)
        # key 必须在任何 QueryRun/Turn/Conversation 写入前验证。
        cipher = _checkpoint_cipher()
        frozen = _freeze_execution(principal=principal, payload=payload)
        persisted_key = (
            _effective_idempotency(principal, idempotency_key) if idempotency_key else None
        )
        payload_hash = _idempotency_payload_hash(
            operation="accept",
            principal=principal,
            payload=payload.model_dump(mode="json"),
            frozen=frozen,
        )
        client_turn_id = payload.client_turn_id
        effective_deadline_seconds = payload.deadline_seconds or _default_deadline_seconds()
        if client_turn_id is None:
            client_turn_id = (
                f"idem_{persisted_key[7:39]}"
                if persisted_key is not None
                else f"client_{uuid4().hex}"
            )

        deduplicated_run_id: str | None = None
        with session_scope() as session:
            auth_binding = _bind_execution_identity(session, principal)
            if persisted_key is not None:
                existing = _existing_idempotent_run(
                    session,
                    principal=principal,
                    persisted_key=persisted_key,
                    payload_hash=payload_hash,
                )
                if existing is not None:
                    try:
                        QueryCheckpointRepository(
                            session,
                            tenant_id=principal.tenant_id,
                            user_id=principal.user_id,
                        ).get_execution(existing.query_run_id)
                    except QueryExecutionNotInitialized as exc:
                        raise QueryRuntimeApiError(
                            status.HTTP_409_CONFLICT,
                            "QUERY_CHECKPOINT_NOT_INITIALIZED",
                            "既有查询缺少可验证的初始 checkpoint，请使用新的请求标识。",
                        ) from exc
                    return _accepted_response(existing, created=False)

            repository = _runtime_repository(session, principal)
            service = QueryRuntimeService(repository)
            try:
                with session.begin_nested():
                    accepted = service.accept(
                        kb_id=frozen.scope.allowed_kb_ids[0],
                        client_turn_id=client_turn_id,
                        user_content=payload.query,
                        conversation_id=payload.conversation_id,
                        conversation_title=payload.conversation_title,
                        idempotency_key=persisted_key,
                        profile_version=frozen.profile_version,
                        index_snapshot=frozen.index_snapshot,
                        config_snapshot=frozen.config_snapshot,
                        deadline_at=datetime.now(UTC)
                        + timedelta(seconds=effective_deadline_seconds),
                    )
                    if accepted.created:
                        initial_state = _build_initial_query_state(
                            principal=principal,
                            run=accepted.run,
                            conversation=accepted.conversation,
                            turn=accepted.turn,
                            history=accepted.history,
                            frozen=frozen,
                        )
                        initialize_query_execution(
                            session,
                            principal=principal,
                            run=accepted.run,
                            state=initial_state,
                            cipher=cipher,
                            auth_binding=auth_binding,
                        )
                    else:
                        QueryCheckpointRepository(
                            session,
                            tenant_id=principal.tenant_id,
                            user_id=principal.user_id,
                        ).get_execution(accepted.run.query_run_id)
                    if persisted_key is not None:
                        claimed_run_id, claimed = QueryReliabilityRepository(
                            session,
                            tenant_id=principal.tenant_id,
                        ).claim_idempotency(
                            idempotency_key=persisted_key,
                            payload_hash=payload_hash,
                            query_run_id=accepted.run.query_run_id,
                        )
                        if not claimed and claimed_run_id != accepted.run.query_run_id:
                            raise _DeduplicatedAcceptance(claimed_run_id)
            except _DeduplicatedAcceptance as exc:
                deduplicated_run_id = exc.query_run_id
            if deduplicated_run_id is not None:
                accepted_run = repository.get_run(deduplicated_run_id)
                try:
                    QueryCheckpointRepository(
                        session,
                        tenant_id=principal.tenant_id,
                        user_id=principal.user_id,
                    ).get_execution(accepted_run.query_run_id)
                except QueryExecutionNotInitialized as exc:
                    raise QueryRuntimeApiError(
                        status.HTTP_409_CONFLICT,
                        "QUERY_CHECKPOINT_NOT_INITIALIZED",
                        "既有查询缺少可验证的初始 checkpoint，请使用新的请求标识。",
                    ) from exc
                return _accepted_response(accepted_run, created=False)
            response = _accepted_response(
                accepted.run,
                created=accepted.created,
            )

        if accepted.created:
            background_tasks.add_task(
                _execute_query_run,
                accepted.run.query_run_id,
                principal,
                frozen.execution_query,
            )
        return response

    @app.post(
        "/api/v1/query-runs/{query_run_id}:cancel",
        response_model=QueryRunCancelResponse,
        tags=["可靠查询"],
    )
    def cancel_query_run(
        query_run_id: str,
        payload: QueryRunCancelRequest,
        request: Request,
    ) -> QueryRunCancelResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            before = repository.get_run(query_run_id)
            expected_revision = (
                before.revision if payload.expected_revision is None else payload.expected_revision
            )
            run, accepted = repository.request_cancel(
                query_run_id,
                expected_revision=expected_revision,
                reason_code=payload.reason_code,
            )
            if accepted and before.status == QueryRunStatus.ACCEPTED:
                run = repository.finalize_cancel(
                    query_run_id,
                    expected_revision=run.revision,
                )
                _mark_idempotency_complete(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                    response_status=status.HTTP_409_CONFLICT,
                )
                _reconcile_terminal_execution(
                    session,
                    principal=principal,
                    query_run_id=query_run_id,
                )
            return QueryRunCancelResponse(
                query_run_id=query_run_id,
                accepted=accepted,
                status=run.status,
                revision=run.revision,
            )

    @app.post(
        "/api/v1/query-runs/{query_run_id}:retry",
        response_model=QueryRunAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["可靠查询"],
    )
    def retry_query_run(
        query_run_id: str,
        payload: QueryRunRetryRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        header_idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            max_length=256,
        ),
    ) -> QueryRunAcceptedResponse:
        principal = _principal(request)
        cipher = _checkpoint_cipher()
        if (
            payload.idempotency_key
            and header_idempotency_key
            and payload.idempotency_key != header_idempotency_key
        ):
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "IDEMPOTENCY_KEY_CONFLICT",
                "请求头与请求体中的 Idempotency-Key 不一致。",
            )
        raw_key = payload.idempotency_key or header_idempotency_key
        if not raw_key:
            raise QueryRuntimeApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "IDEMPOTENCY_KEY_REQUIRED",
                "重试必须提供 Idempotency-Key。",
            )
        persisted_key = _effective_idempotency(principal, raw_key)
        with session_scope() as session:
            original = _runtime_repository(session, principal).get_run(query_run_id)
        effective_deadline_seconds = payload.deadline_seconds or _default_deadline_seconds()
        frozen = _retry_payload(
            run=original,
            principal=principal,
            deadline_seconds=effective_deadline_seconds,
        )
        payload_hash = _idempotency_payload_hash(
            operation="retry",
            principal=principal,
            payload=payload.model_dump(
                mode="json",
                exclude={"idempotency_key"},
            ),
            frozen=frozen,
            source_query_run_id=query_run_id,
        )

        deduplicated_run_id: str | None = None
        with session_scope() as session:
            auth_binding = _bind_execution_identity(session, principal)
            existing = _existing_idempotent_run(
                session,
                principal=principal,
                persisted_key=persisted_key,
                payload_hash=payload_hash,
            )
            if existing is not None:
                try:
                    QueryCheckpointRepository(
                        session,
                        tenant_id=principal.tenant_id,
                        user_id=principal.user_id,
                    ).get_execution(existing.query_run_id)
                except QueryExecutionNotInitialized as exc:
                    raise QueryRuntimeApiError(
                        status.HTTP_409_CONFLICT,
                        "QUERY_CHECKPOINT_NOT_INITIALIZED",
                        "既有重试缺少可验证的初始 checkpoint，请使用新的请求标识。",
                    ) from exc
                return _accepted_response(existing, created=False)
            repository = _runtime_repository(session, principal)
            try:
                with session.begin_nested():
                    run, retry_history, created = repository.retry_run(
                        query_run_id,
                        idempotency_key=persisted_key,
                        deadline_at=datetime.now(UTC)
                        + timedelta(seconds=effective_deadline_seconds),
                    )
                    if created:
                        initial_state = _build_initial_query_state(
                            principal=principal,
                            run=run,
                            conversation=repository.get_conversation(run.conversation_id),
                            turn=repository.get_turn(run.turn_id),
                            history=retry_history,
                            frozen=frozen,
                        )
                        initialize_query_execution(
                            session,
                            principal=principal,
                            run=run,
                            state=initial_state,
                            cipher=cipher,
                            auth_binding=auth_binding,
                        )
                    else:
                        QueryCheckpointRepository(
                            session,
                            tenant_id=principal.tenant_id,
                            user_id=principal.user_id,
                        ).get_execution(run.query_run_id)
                    claimed_run_id, claimed = QueryReliabilityRepository(
                        session,
                        tenant_id=principal.tenant_id,
                    ).claim_idempotency(
                        idempotency_key=persisted_key,
                        payload_hash=payload_hash,
                        query_run_id=run.query_run_id,
                    )
                    if not claimed and claimed_run_id != run.query_run_id:
                        raise _DeduplicatedAcceptance(claimed_run_id)
            except _DeduplicatedAcceptance as exc:
                deduplicated_run_id = exc.query_run_id
            if deduplicated_run_id is not None:
                run = repository.get_run(deduplicated_run_id)
                try:
                    QueryCheckpointRepository(
                        session,
                        tenant_id=principal.tenant_id,
                        user_id=principal.user_id,
                    ).get_execution(run.query_run_id)
                except QueryExecutionNotInitialized as exc:
                    raise QueryRuntimeApiError(
                        status.HTTP_409_CONFLICT,
                        "QUERY_CHECKPOINT_NOT_INITIALIZED",
                        "既有重试缺少可验证的初始 checkpoint，请使用新的请求标识。",
                    ) from exc
                created = False
            response = _accepted_response(run, created=created)

        if created:
            background_tasks.add_task(
                _execute_query_run,
                run.query_run_id,
                principal,
                frozen.execution_query,
            )
        return response

    @app.post(
        "/api/v1/query-runs/{query_run_id}:replay",
        response_model=QueryRunReplayResponse,
        tags=["可靠查询"],
    )
    def replay_query_run(
        query_run_id: str,
        payload: QueryRunReplayRequest,
        request: Request,
    ) -> QueryRunReplayResponse:
        principal = _principal(request)
        with session_scope() as session:
            # Replay repository 以 tenant 为边界；先通过 user-scoped runtime
            # repository 验证所有权，避免同租户用户枚举其他人的 manifest。
            _runtime_repository(session, principal).get_run(query_run_id)
            manifest = QueryReliabilityRepository(
                session,
                tenant_id=principal.tenant_id,
            ).get_replay_manifest(query_run_id)
            replay_state = _restore_replay_proof_state_or_error(
                session=session,
                principal=principal,
                manifest=manifest,
            )
        result = ReplayService().replay(
            manifest,
            mode=payload.mode,
            # 在线固定快照重放只恢复已通过 proof 重验的权威 envelope，不调用
            # LLM/Web/工具，也不创建 Turn/Message/Usage。
            generator=lambda _item: dict(
                (_public_result(replay_state).get("answer_envelope") or {})
            ),
        )
        return QueryRunReplayResponse(result=result)

    @app.get(
        "/api/v1/query-runs/{query_run_id}/replay",
        response_model=QueryReplayManifestResponse,
        tags=["可靠查询"],
    )
    def get_query_replay_manifest(
        query_run_id: str,
        request: Request,
    ) -> QueryReplayManifestResponse:
        principal = _principal(request)
        with session_scope() as session:
            _runtime_repository(session, principal).get_run(query_run_id)
            manifest = QueryReliabilityRepository(
                session,
                tenant_id=principal.tenant_id,
            ).get_replay_manifest(query_run_id)
            _restore_replay_proof_state_or_error(
                session=session,
                principal=principal,
                manifest=manifest,
            )
        return QueryReplayManifestResponse(manifest=manifest)

    @app.get(
        "/api/v1/query-runs/{query_run_id}/events/stream",
        tags=["可靠查询"],
    )
    async def stream_query_events(
        query_run_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
            max_length=128,
        ),
    ) -> StreamingResponse:
        principal = _principal(request)
        # 在发送响应头前先验证所有权，避免把“不存在”和“其他用户资源”变成
        # 一个看似成功但随后 reset 的 SSE 连接。
        with session_scope() as session:
            _runtime_repository(session, principal).get_run(query_run_id)

        def read_page_sync(
            run_id: str,
            after_seq: int,
            resume_event_id: str | None,
            limit: int,
        ):
            with session_scope() as session:
                return _runtime_repository(
                    session,
                    principal,
                ).read_event_page(
                    run_id,
                    after_seq=after_seq,
                    last_event_id=resume_event_id,
                    limit=limit,
                )

        async def read_page(
            run_id: str,
            after_seq: int,
            resume_event_id: str | None,
            limit: int,
        ):
            return await asyncio.to_thread(
                read_page_sync,
                run_id,
                after_seq,
                resume_event_id,
                limit,
            )

        generator = durable_sse_generator(
            query_run_id=query_run_id,
            read_page=read_page,
            last_event_id=last_event_id,
            initial_after_seq=after_seq,
            policy=DurableStreamPolicy(
                batch_size=settings.query_event_batch_size,
                max_lag_events=settings.query_stream_max_lag_events,
                max_batch_bytes=settings.query_stream_max_batch_bytes,
            ),
            is_disconnected=request.is_disconnected,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get(
        "/api/v1/query-runs/{query_run_id}/events",
        response_model=QueryEventPageResponse,
        tags=["可靠查询"],
    )
    def list_query_events(
        query_run_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0),
        limit: int = Query(
            default=settings.query_event_batch_size,
            ge=1,
            le=1000,
        ),
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
            max_length=128,
        ),
    ) -> QueryEventPageResponse:
        principal = _principal(request)
        try:
            with session_scope() as session:
                page = _runtime_repository(
                    session,
                    principal,
                ).read_event_page(
                    query_run_id,
                    after_seq=after_seq,
                    last_event_id=last_event_id,
                    limit=limit,
                )
        except EventCursorExpired as exc:
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "EVENT_CURSOR_EXPIRED",
                "事件续传窗口已过期，请读取 QueryRun 快照后重新订阅。",
            ) from exc
        except EventCursorInvalid as exc:
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "EVENT_CURSOR_INVALID",
                "Last-Event-ID 不属于当前 QueryRun。",
            ) from exc
        return QueryEventPageResponse(page=page)

    @app.get(
        "/api/v1/query-runs/{query_run_id}/result",
        response_model=QueryRunResultResponse,
        tags=["可靠查询"],
    )
    def get_query_result(
        query_run_id: str,
        request: Request,
    ) -> QueryRunResultResponse:
        principal = _principal(request)
        try:
            with session_scope() as session:
                repository = _runtime_repository(session, principal)
                run = repository.get_run(query_run_id)
                result = repository.get_result(query_run_id)
                return QueryRunResultResponse(
                    query_run_id=query_run_id,
                    status=run.status,
                    result=result,
                )
        except ResourceConflictError as exc:
            raise QueryRuntimeApiError(
                status.HTTP_409_CONFLICT,
                "QUERY_RESULT_NOT_READY",
                "查询结果尚未可用，请先读取 QueryRun 状态或事件。",
            ) from exc

    @app.get(
        "/api/v1/query-runs/{query_run_id}",
        response_model=QueryRunResponse,
        tags=["可靠查询"],
    )
    def get_query_run(
        query_run_id: str,
        request: Request,
    ) -> QueryRunResponse:
        principal = _principal(request)
        with session_scope() as session:
            run = _runtime_repository(session, principal).get_run(query_run_id)
            return QueryRunResponse.from_domain(run)

    @app.get(
        "/api/v1/conversations",
        response_model=ConversationListResponse,
        tags=["可靠查询"],
    )
    def list_conversations(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        include_archived: bool = False,
        status_filter: Literal["active", "archived", "all"] | None = Query(
            default=None,
            alias="status",
        ),
        search: str = Query(default="", max_length=200),
    ) -> ConversationListResponse:
        principal = _principal(request)
        effective_status = status_filter or ("all" if include_archived else "active")
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            candidate_kb_ids = repository.list_conversation_kb_ids(
                include_archived=include_archived,
                status_filter=effective_status,
            )
            allowed_kb_ids = _authorized_conversation_kb_ids(
                principal=principal,
                candidate_kb_ids=candidate_kb_ids,
            )
            rows = repository.list_conversations(
                limit=limit,
                offset=offset,
                include_archived=include_archived,
                allowed_kb_ids=allowed_kb_ids,
                status_filter=effective_status,
                search=search,
            )
            return ConversationListResponse(
                items=tuple(rows),
                limit=limit,
                offset=offset,
                status_filter=effective_status,
                search=search.strip(),
            )

    @app.get(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=ConversationMessageListResponse,
        tags=["可靠查询"],
    )
    def list_conversation_messages(
        conversation_id: str,
        request: Request,
    ) -> ConversationMessageListResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            conversation = _require_authorized_conversation(
                repository=repository,
                principal=principal,
                conversation_id=conversation_id,
            )
            items = tuple(
                ConversationMessageResponse.from_domain(
                    message,
                    metadata=_public_message_metadata(message.metadata),
                )
                for message in repository.list_messages(conversation_id)
                if message.status == "committed"
                and message.role in {"user", "assistant"}
                and message.committed_at is not None
            )
            latest_runs = {item.turn_id: item for item in repository.list_runs(conversation_id)}
            turns = tuple(
                ConversationTurnResponse.from_domain(
                    turn,
                    run=latest_runs.get(turn.turn_id),
                )
                for turn in repository.list_turns(conversation_id)
                if turn.status != TurnStatus.COMMITTED
            )
            return ConversationMessageListResponse(
                conversation=conversation,
                items=items,
                turns=turns,
            )

    @app.post(
        "/api/v1/conversations/{conversation_id}:archive",
        response_model=ConversationMutationResponse,
        tags=["可靠查询"],
    )
    def archive_conversation(
        conversation_id: str,
        payload: ConversationMutationRequest,
        request: Request,
    ) -> ConversationMutationResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            _require_authorized_conversation(
                repository=repository,
                principal=principal,
                conversation_id=conversation_id,
            )
            try:
                conversation = repository.archive_conversation(
                    conversation_id,
                    expected_revision=payload.expected_revision,
                )
            except ResourceConflictError as exc:
                raise _conversation_conflict(exc) from exc
            return ConversationMutationResponse(conversation=conversation)

    @app.post(
        "/api/v1/conversations/{conversation_id}:restore",
        response_model=ConversationMutationResponse,
        tags=["可靠查询"],
    )
    def restore_conversation(
        conversation_id: str,
        payload: ConversationMutationRequest,
        request: Request,
    ) -> ConversationMutationResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            _require_authorized_conversation(
                repository=repository,
                principal=principal,
                conversation_id=conversation_id,
            )
            try:
                conversation = repository.restore_conversation(
                    conversation_id,
                    expected_revision=payload.expected_revision,
                )
            except ResourceConflictError as exc:
                raise _conversation_conflict(exc) from exc
            return ConversationMutationResponse(conversation=conversation)

    @app.delete(
        "/api/v1/conversations/{conversation_id}",
        response_model=ConversationDeleteResponse,
        tags=["可靠查询"],
    )
    def delete_conversation(
        conversation_id: str,
        payload: ConversationMutationRequest,
        request: Request,
    ) -> ConversationDeleteResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            _require_authorized_conversation(
                repository=repository,
                principal=principal,
                conversation_id=conversation_id,
            )
            try:
                revision = repository.delete_conversation(
                    conversation_id,
                    expected_revision=payload.expected_revision,
                )
            except ResourceConflictError as exc:
                raise _conversation_conflict(exc) from exc
            return ConversationDeleteResponse(
                conversation_id=conversation_id,
                revision=revision,
            )

    @app.get(
        "/api/v1/conversations/{conversation_id}/query-runs",
        response_model=ConversationRunListResponse,
        tags=["可靠查询"],
    )
    def list_conversation_runs(
        conversation_id: str,
        request: Request,
    ) -> ConversationRunListResponse:
        principal = _principal(request)
        with session_scope() as session:
            repository = _runtime_repository(session, principal)
            _require_authorized_conversation(
                repository=repository,
                principal=principal,
                conversation_id=conversation_id,
            )
            rows = repository.list_runs(conversation_id)
            return ConversationRunListResponse(
                conversation_id=conversation_id,
                items=tuple(QueryRunResponse.from_domain(item) for item in rows),
            )


__all__ = ["register_query_runtime_routes"]
