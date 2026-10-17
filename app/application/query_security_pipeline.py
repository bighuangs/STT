"""把 TODO-26/TODO-34 的安全契约接入现有查询图。

本模块只在状态中传播不可变授权快照和脱敏判定摘要。原始恶意正文不会进入
审计/日志；Evidence 的权威正文与 hash 保持不变，ContextPack 通过
``security_safe_content`` 使用经过检查的模型输入表示。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.application.rag_security_service import RagSecurityService
from app.domain.content_security import ContentSecurityPolicy, SecurityAction
from app.domain.query_security import (
    AuthorizedRetrievalScope,
    DataClassification,
    EgressPolicySnapshot,
    ExternalDataRequest,
    ScopeAuthorizationError,
    decide_external_data,
)
from app.security.principal import current_principal
from app.security.service import audit


class QueryContentSecurityViolation(ValueError):
    """携带脱敏判定字段的确定性输入安全阻断。"""

    code = "QUERY_CONTENT_SECURITY_BLOCKED"
    safe_message = "查询内容触发安全策略，已阻断执行。"
    retryable = False

    def __init__(
        self,
        *,
        stage: str,
        action: str,
        reason_codes: tuple[str, ...],
        policy_version: str,
        decision_id: str,
    ) -> None:
        super().__init__(self.safe_message)
        self.stage = stage
        self.action = action
        self.reason_codes = tuple(reason_codes)
        self.policy_version = policy_version
        self.decision_id = decision_id

    def audit_fields(self) -> dict[str, Any]:
        """返回可进入运行记录和评测观测的脱敏字段。"""

        return {
            "code": self.code,
            "safe_message": self.safe_message,
            "retryable": self.retryable,
            "stage": self.stage,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
            "decision_id": self.decision_id,
        }


class QueryEgressDenied(PermissionError):
    """一次具体外部调用未通过冻结外发策略。"""

    code = "QUERY_EGRESS_DENIED"

    def __init__(self, reason_code: str) -> None:
        super().__init__(f"external request denied: {reason_code}")
        self.reason_code = reason_code


class _SqlContentSecurityAudit:
    def record(self, *, fields: dict[str, Any]) -> None:
        principal = current_principal()
        if principal is None:
            return
        audit(
            principal,
            "query.content_security.decide",
            "recorded",
            principal.request_id,
            None,
            fields,
        )


def content_policy_from_state(state: dict[str, Any]) -> ContentSecurityPolicy:
    """从服务端注入的策略快照构造内容策略；未知字段被拒绝前先显式丢弃。"""

    raw = state.get("content_security_policy")
    raw = dict(raw) if isinstance(raw, dict) else {}
    allowed = set(ContentSecurityPolicy.model_fields)
    values = {key: value for key, value in raw.items() if key in allowed}
    values.setdefault("web_enabled", bool(state.get("web_search_allowed", False)))
    return ContentSecurityPolicy.model_validate(values)


def _service(state: dict[str, Any]) -> RagSecurityService:
    return RagSecurityService(
        policy=content_policy_from_state(state),
        audit=_SqlContentSecurityAudit(),
    )


def consume_authorized_scope(
    state: dict[str, Any],
    stage: str,
    *,
    required: bool | None = None,
) -> AuthorizedRetrievalScope | None:
    """校验敏感阶段使用的范围没有被替换、换租户或扩展。

    可靠 QueryRun 默认强制；旧接口在迁移期没有快照时保持兼容，但一旦携带快照
    就执行同样的 fail-closed 校验。
    """

    raw = state.get("authorized_scope")
    must_exist = bool(state.get("runtime_managed_history")) if required is None else required
    if not raw:
        if must_exist:
            raise ScopeAuthorizationError(f"{stage} requires authorized retrieval scope")
        return None
    scope = AuthorizedRetrievalScope.model_validate(raw)
    principal = current_principal()
    tenant_id = str(state.get("tenant_id") or (principal.tenant_id if principal else ""))
    user_id = str(state.get("user_id") or (principal.user_id if principal else ""))
    kb_id = str(state.get("kb_id") or "")
    if (
        not tenant_id
        or not user_id
        or scope.tenant_id != tenant_id
        or scope.principal_id != user_id
        or kb_id not in scope.allowed_kb_ids
    ):
        raise ScopeAuthorizationError(f"{stage} scope identity mismatch")
    consumed = dict(state.get("scope_consumed_stages") or {})
    previous = consumed.get(stage)
    if previous is not None and previous != scope.scope_hash:
        raise ScopeAuthorizationError(f"{stage} attempted to replace authorized scope")
    consumed[stage] = scope.scope_hash
    state["scope_consumed_stages"] = consumed
    return scope


def authorize_external_call(
    state: dict[str, Any],
    *,
    purpose: str,
    provider: str,
    target_domain: str | None = None,
) -> None:
    """在每一次 Web/LLM 外呼前消费冻结 scope 与策略，默认拒绝。"""

    scope = consume_authorized_scope(
        state,
        "web" if purpose == "web_search" else f"egress:{purpose}",
    )
    policies = state.get("egress_policies")
    policies = policies if isinstance(policies, dict) else {}
    raw_policy = policies.get(purpose)
    if isinstance(raw_policy, dict):
        policy = EgressPolicySnapshot.model_validate(raw_policy)
    elif scope is None:
        # 兼容旧入口：只接受 register_query_session 从控制库注入的布尔策略，
        # 不读取客户端布尔值。可靠入口没有策略快照时始终走下面的默认拒绝。
        enabled = bool(
            state.get(
                "web_search_allowed" if purpose == "web_search" else "external_llm_allowed",
                False,
            )
        )
        policy = EgressPolicySnapshot(
            policy_version="egress.legacy-control-plane.v1",
            enabled=enabled,
            allowed_providers=(provider,) if enabled else (),
            allowed_purposes=(purpose,) if enabled else (),
            allowed_domains=((target_domain,) if enabled and target_domain else ()),
            max_classification=DataClassification.INTERNAL,
            retention_policy="provider-zero-retention-required",
        )
    else:
        policy = EgressPolicySnapshot()

    raw_classification = str(state.get("data_classification") or "internal").lower()
    try:
        classification = DataClassification(raw_classification)
    except ValueError:
        classification = DataClassification.RESTRICTED
    scope_hash = scope.scope_hash if scope is not None else "scope_" + "0" * 64
    decision = decide_external_data(
        ExternalDataRequest(
            requested=True,
            provider=provider,
            purpose=purpose,
            scope_hash=scope_hash,
            data_classes=(classification,),
            target_domain=target_domain,
        ),
        policy,
    )
    principal = current_principal()
    if principal is not None:
        audit(
            principal,
            "query.egress.decide",
            "allowed" if decision.allowed else "denied",
            principal.request_id,
            None,
            decision.audit_fields(),
        )
    state["egress_decisions"] = [
        *(state.get("egress_decisions") or []),
        decision.audit_fields(),
    ]
    if not decision.allowed:
        raise QueryEgressDenied(decision.reason_code)


def inspect_user_query_state(state: dict[str, Any]) -> None:
    tenant_id = str(state.get("tenant_id") or "")
    principal = current_principal()
    tenant_id = tenant_id or (principal.tenant_id if principal else "")
    if not tenant_id:
        # 旧离线单元调用不伪造租户；可靠在线入口必须始终有 tenant_id。
        if state.get("runtime_managed_history"):
            raise ScopeAuthorizationError("query security requires tenant identity")
        tenant_id = "tenant_legacy"
    message_id = str(
        state.get("turn_id")
        or state.get("query_run_id")
        or state.get("session_id")
        or "legacy-query"
    )
    secured = _service(state).inspect_query(
        text=str(state.get("original_query") or ""),
        message_id=message_id,
        tenant_id=tenant_id,
    )
    state["content_security"] = {
        **dict(state.get("content_security") or {}),
        "policy_version": secured.decision.policy_version,
        "input": secured.decision.audit_fields(),
    }
    if secured.decision.action not in {SecurityAction.ALLOW, SecurityAction.REDACT}:
        # 异常、运行日志和后续终态都不得继续持有被阻断的原始输入。
        state["original_query"] = ""
        raise QueryContentSecurityViolation(
            stage=secured.decision.stage.value,
            action=secured.decision.action.value,
            reason_codes=secured.decision.reason_codes,
            policy_version=secured.decision.policy_version,
            decision_id=secured.decision.decision_id,
        )
    state["original_query"] = secured.safe_representation


def secure_evidence_documents(
    state: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """隔离恶意 Evidence，仅把安全表示交给 ContextPack。"""

    if not documents:
        return []
    tenant_id = str(state.get("tenant_id") or "")
    principal = current_principal()
    tenant_id = tenant_id or (principal.tenant_id if principal else "tenant_legacy")
    normalized: list[dict[str, Any]] = []
    originals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, document in enumerate(documents, start=1):
        item = dict(document)
        evidence = item.get("evidence")
        source = evidence.get("source") if isinstance(evidence, dict) else {}
        source_type = str((source or {}).get("source_type") or item.get("type") or "")
        item["type"] = "local" if source_type in {"local", "milvus", "knowledge_base"} else "web"
        candidate_id = str(
            item.get("chunk_id")
            or item.get("evidence_id")
            or item.get("url")
            or f"candidate-{position}"
        )
        originals[candidate_id].append(document)
        normalized.append(item)

    batch = _service(state).inspect_evidence(normalized, tenant_id=tenant_id)
    approved: list[dict[str, Any]] = []
    for candidate in batch.approved:
        candidates = originals.get(candidate.candidate_id) or []
        if not candidates:
            continue
        derivation_receipt = candidate.content.derivation_receipt
        if derivation_receipt is None:
            raise RuntimeError(
                "approved evidence is missing its safe representation derivation receipt"
            )
        document = dict(candidates.pop(0))
        document["security_safe_content"] = candidate.content.safe_representation
        document["safe_representation_derivation"] = derivation_receipt.to_dict()
        document["content_security"] = candidate.content.decision.audit_fields()
        approved.append(document)
    summary = {
        "approved_count": len(approved),
        "quarantined_count": len(batch.quarantined),
        "quarantined": [
            {
                "candidate_id": item.candidate_id,
                "content_hash": item.content_hash,
                "decision_id": item.decision_id,
                "reason_codes": list(item.reason_codes),
            }
            for item in batch.quarantined
        ],
    }
    state["content_security"] = {
        **dict(state.get("content_security") or {}),
        "evidence": summary,
    }
    return approved


def secure_answer_state(state: dict[str, Any]) -> None:
    """净化模型终态；泄漏/危险输出转换成稳定拒答，不提交原始结果。"""

    answer = str(state.get("answer") or "")
    if not answer:
        return
    allowed_ids = {
        str(item.get("evidence_id"))
        for item in (state.get("evidences") or [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    result = _service(state).inspect_answer(
        output=answer,
        answer_id=str(state.get("answer_id") or state.get("query_run_id") or "answer"),
        allowed_citation_ids=allowed_ids,
    )
    state["content_security"] = {
        **dict(state.get("content_security") or {}),
        "output": {
            **result.content.decision.audit_fields(),
            "removed_citation_ids": list(result.removed_citation_ids),
        },
    }
    if result.content.decision.action in {SecurityAction.BLOCK, SecurityAction.QUARANTINE}:
        state["answer"] = "回答内容触发安全策略，已阻断输出。请缩小问题范围后重试。"
        state["answer_decision"] = {
            "action": "REFUSE",
            "reason_code": "OUTPUT_SECURITY_BLOCKED",
            "policy_version": result.content.decision.policy_version,
        }
        state["citations"] = []
        state["sources"] = []
        return
    state["answer"] = result.safe_output


__all__ = [
    "QueryContentSecurityViolation",
    "QueryEgressDenied",
    "authorize_external_call",
    "consume_authorized_scope",
    "content_policy_from_state",
    "inspect_user_query_state",
    "secure_answer_state",
    "secure_evidence_documents",
]
