"""TODO-22 长期记忆应用核心。

默认不自动提取、不自动激活；所有公开操作要求 actor 与 subject 相同，并且 repository
查询始终携带 tenant/user/KB 三元 scope。知识库证据不属于允许的 provenance 类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from typing import Protocol, runtime_checkable
from uuid import uuid4

from app.domain.memory import (
    ForgetReceipt,
    LongTermMemory,
    MemoryAccessScope,
    MemoryAuditAction,
    MemoryAuditEvent,
    MemoryCandidate,
    MemoryConsent,
    MemoryContext,
    MemoryNamespace,
    MemoryProvenance,
    MemoryRetrievalHit,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    MemoryType,
    memory_candidate_id,
    memory_id_for_candidate,
    provenance_hash,
    utc_now,
)


_EXPLICIT_REMEMBER_RE = re.compile(
    r"^\s*(?:请(?:帮我)?记住|记住|以后请|remember(?:\s+that)?)[:：,\s]*",
    re.IGNORECASE,
)
_NEGATIVE_REMEMBER_RE = re.compile(r"(?:不要|别|无需)(?:永久)?记住|忘记(?:这|上面)")
_RESTRICTED_RE = re.compile(
    r"(?:管理员|admin).{0,8}(?:密码|密钥|secret|token)"
    r"|(?:密码|口令|私钥|api[_ -]?key|access[_ -]?token|身份证号|银行卡号)",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(r"(?:手机号|电话号码|住址|地址|生日|邮箱)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]")


class MemoryServiceError(RuntimeError):
    code = "MEMORY_ERROR"


class MemoryAccessDeniedError(MemoryServiceError):
    code = "MEMORY_ACCESS_DENIED"


class MemoryNotFoundError(MemoryServiceError):
    code = "MEMORY_NOT_FOUND"


class MemoryPolicyRejectedError(MemoryServiceError):
    code = "MEMORY_POLICY_REJECTED"


class MemoryStateConflictError(MemoryServiceError):
    code = "MEMORY_STATE_CONFLICT"


@dataclass(frozen=True)
class MemoryPolicy:
    auto_candidate_extraction_enabled: bool = False
    auto_activation_enabled: bool = False
    preference_ttl_days: int = 365
    profile_fact_ttl_days: int = 180
    episodic_ttl_days: int = 30
    max_context_tokens: int = 256
    max_hits: int = 8
    embedding_version: str = "memory-lexical/1.0.0"

    def ttl_for(self, memory_type: MemoryType) -> timedelta:
        days = {
            MemoryType.PREFERENCE: self.preference_ttl_days,
            MemoryType.PROFILE_FACT: self.profile_fact_ttl_days,
            MemoryType.EPISODIC_SUMMARY: self.episodic_ttl_days,
        }.get(memory_type)
        if days is None:
            raise MemoryPolicyRejectedError(
                "working memory cannot be persisted in long-term memory"
            )
        if days <= 0:
            raise MemoryPolicyRejectedError("memory TTL must be positive")
        return timedelta(days=days)


@dataclass(frozen=True)
class MemoryProposalResult:
    proposed: tuple[LongTermMemory, ...] = ()
    rejected_reason_codes: tuple[str, ...] = ()
    extraction_skipped_reason: str | None = None


@runtime_checkable
class MemoryRepository(Protocol):
    def create_proposed(
        self,
        memory: LongTermMemory,
        audit: MemoryAuditEvent,
    ) -> LongTermMemory: ...

    def get(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> LongTermMemory | None: ...

    def list(
        self,
        namespace: MemoryNamespace,
        *,
        statuses: tuple[MemoryStatus, ...] | None = None,
    ) -> tuple[LongTermMemory, ...]: ...

    def activate_with_supersede(
        self,
        *,
        proposed: LongTermMemory,
        active: LongTermMemory,
        audit: MemoryAuditEvent,
        supersede_audits: tuple[MemoryAuditEvent, ...],
    ) -> tuple[LongTermMemory, tuple[LongTermMemory, ...]]: ...

    def replace(
        self,
        *,
        current: LongTermMemory,
        replacement: LongTermMemory,
        audit: MemoryAuditEvent,
    ) -> LongTermMemory: ...

    def list_audits(
        self,
        namespace: MemoryNamespace,
    ) -> tuple[MemoryAuditEvent, ...]: ...

    def append_audit(self, audit: MemoryAuditEvent) -> None: ...


@runtime_checkable
class MemoryDerivativeStore(Protocol):
    """向量、缓存与派生摘要清理端口。"""

    def index(self, memory: LongTermMemory) -> None: ...

    def delete(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> tuple[str, ...]: ...


class MemoryCandidateExtractor:
    version = "memory-rules/1.0.0"

    def __init__(self, policy: MemoryPolicy) -> None:
        self._policy = policy

    def extract(
        self,
        *,
        scope: MemoryAccessScope,
        text: str,
        session_id: str,
        source_message_id: str,
        explicit_request: bool,
        now: datetime,
    ) -> tuple[MemoryCandidate, ...]:
        normalized = " ".join(text.split()).strip()
        if not normalized or _NEGATIVE_REMEMBER_RE.search(normalized):
            return ()
        if not explicit_request and not self._policy.auto_candidate_extraction_enabled:
            return ()
        content = _EXPLICIT_REMEMBER_RE.sub("", normalized).strip(" ,，。:：")
        if not content:
            return ()
        if _RESTRICTED_RE.search(content):
            raise MemoryPolicyRejectedError(
                "restricted credential or identity data cannot become memory"
            )

        memory_type, structured_value, conflict_key = self._classify(content)
        sensitivity = (
            MemorySensitivity.SENSITIVE
            if _SENSITIVE_RE.search(content)
            else MemorySensitivity.PERSONAL
        )
        provenance = MemoryProvenance(
            source_kind=MemorySourceKind.USER_MESSAGE,
            session_id=session_id,
            source_message_ids=(source_message_id,),
            extractor_version=self.version,
            source_excerpt_hash=provenance_hash(text),
        )
        candidate_id = memory_candidate_id(
            namespace=scope.namespace,
            memory_type=memory_type,
            content=content,
            source_message_ids=provenance.source_message_ids,
        )
        return (
            MemoryCandidate(
                candidate_id=candidate_id,
                namespace=scope.namespace,
                type=memory_type,
                content=content,
                structured_value=structured_value,
                conflict_key=conflict_key,
                confidence=0.98 if explicit_request else 0.72,
                sensitivity=sensitivity,
                consent=(
                    MemoryConsent.EXPLICIT_REQUEST if explicit_request else MemoryConsent.NONE
                ),
                provenance=provenance,
                expires_at=now + self._policy.ttl_for(memory_type),
                proposed_at=now,
            ),
        )

    @staticmethod
    def _classify(content: str) -> tuple[MemoryType, dict, str]:
        lowered = content.casefold()
        if any(word in content for word in ("中文", "英文")) and any(
            word in content for word in ("回答", "回复", "交流", "语言")
        ):
            value = "zh-CN" if "中文" in content else "en"
            return (
                MemoryType.PREFERENCE,
                {"key": "response_language", "value": value},
                "preference:response_language",
            )
        if any(word in content for word in ("表格", "列表", "要点", "简洁", "详细")):
            values = [word for word in ("表格", "列表", "要点", "简洁", "详细") if word in content]
            return (
                MemoryType.PREFERENCE,
                {"key": "response_format", "value": values},
                "preference:response_format",
            )
        if any(marker in content for marker in ("我喜欢", "我偏好", "我习惯", "我更喜欢", "请用")):
            return (
                MemoryType.PREFERENCE,
                {"key": "general_preference", "value": content},
                "preference:general",
            )
        if any(marker in content for marker in ("我是", "我的职位", "我所在")):
            digest = provenance_hash(lowered)[:16]
            return (
                MemoryType.PROFILE_FACT,
                {"key": f"profile:{digest}", "value": content},
                f"profile:{digest}",
            )
        digest = provenance_hash(lowered)[:16]
        return (
            MemoryType.EPISODIC_SUMMARY,
            {"key": f"episode:{digest}", "value": content},
            f"episode:{digest}",
        )


class LongTermMemoryService:
    def __init__(
        self,
        *,
        repository: MemoryRepository,
        derivatives: MemoryDerivativeStore,
        policy: MemoryPolicy | None = None,
        extractor: MemoryCandidateExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._derivatives = derivatives
        self._policy = policy or MemoryPolicy()
        self._extractor = extractor or MemoryCandidateExtractor(self._policy)

    def propose_from_message(
        self,
        *,
        scope: MemoryAccessScope,
        text: str,
        session_id: str,
        source_message_id: str,
        explicit_request: bool = False,
        now: datetime | None = None,
    ) -> MemoryProposalResult:
        self._ensure_owner(scope)
        observed_at = now or utc_now()
        if not explicit_request and not self._policy.auto_candidate_extraction_enabled:
            return MemoryProposalResult(extraction_skipped_reason="automatic_memory_disabled")
        try:
            candidates = self._extractor.extract(
                scope=scope,
                text=text,
                session_id=session_id,
                source_message_id=source_message_id,
                explicit_request=explicit_request,
                now=observed_at,
            )
        except MemoryPolicyRejectedError:
            return MemoryProposalResult(rejected_reason_codes=("restricted_or_sensitive_secret",))

        proposed: list[LongTermMemory] = []
        for candidate in candidates:
            memory = LongTermMemory(
                memory_id=memory_id_for_candidate(candidate.candidate_id),
                candidate_id=candidate.candidate_id,
                namespace=candidate.namespace,
                type=candidate.type,
                content=candidate.content,
                structured_value=candidate.structured_value,
                conflict_key=candidate.conflict_key,
                provenance=candidate.provenance,
                confidence=candidate.confidence,
                sensitivity=candidate.sensitivity,
                consent=candidate.consent,
                status=MemoryStatus.PROPOSED,
                valid_from=observed_at,
                expires_at=candidate.expires_at,
                created_by=scope.actor_user_id,
                created_at=observed_at,
                updated_at=observed_at,
            )
            audit = self._audit(
                scope,
                memory.memory_id,
                MemoryAuditAction.PROPOSED,
                "explicit_request" if explicit_request else "model_candidate",
            )
            proposed.append(self._repository.create_proposed(memory, audit))

        # 即使显式“记住”，也先 proposed；调用 confirm 才能进入 active。
        return MemoryProposalResult(proposed=tuple(proposed))

    def confirm(
        self,
        scope: MemoryAccessScope,
        memory_id: str,
        *,
        now: datetime | None = None,
    ) -> LongTermMemory:
        self._ensure_owner(scope)
        observed_at = now or utc_now()
        proposed = self._required(scope.namespace, memory_id)
        if proposed.status == MemoryStatus.ACTIVE:
            return proposed
        if proposed.status != MemoryStatus.PROPOSED:
            raise MemoryStateConflictError(
                f"only proposed memory can be confirmed; got {proposed.status.value}"
            )
        if proposed.expires_at <= observed_at:
            self._expire(scope, proposed, observed_at)
            raise MemoryStateConflictError("expired candidate cannot be confirmed")
        if proposed.sensitivity == MemorySensitivity.RESTRICTED:
            raise MemoryPolicyRejectedError("restricted memory cannot be confirmed")
        if (
            proposed.sensitivity == MemorySensitivity.SENSITIVE
            and proposed.consent != MemoryConsent.EXPLICIT_REQUEST
        ):
            raise MemoryPolicyRejectedError("sensitive memory requires an explicit user request")

        conflicts = tuple(
            memory
            for memory in self._repository.list(
                scope.namespace,
                statuses=(MemoryStatus.ACTIVE,),
            )
            if memory.type == proposed.type
            and memory.conflict_key == proposed.conflict_key
            and memory.memory_id != proposed.memory_id
        )
        active = proposed.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "consent": MemoryConsent.CONFIRMED,
                "embedding_version": self._policy.embedding_version,
                "supersedes_memory_id": (conflicts[0].memory_id if conflicts else None),
                "valid_from": observed_at,
                "updated_at": observed_at,
            }
        )
        audit = self._audit(
            scope,
            memory_id,
            MemoryAuditAction.CONFIRMED,
            "user_confirmation",
        )
        supersede_audits = tuple(
            self._audit(
                scope,
                conflict.memory_id,
                MemoryAuditAction.SUPERSEDED,
                "newer_confirmed_conflict",
                metadata={"superseded_by": memory_id},
            )
            for conflict in conflicts
        )
        activated, superseded = self._repository.activate_with_supersede(
            proposed=proposed,
            active=active,
            audit=audit,
            supersede_audits=supersede_audits,
        )
        for old in superseded:
            self._derivatives.delete(scope.namespace, old.memory_id)
        self._derivatives.index(activated)
        return activated

    def reject(
        self,
        scope: MemoryAccessScope,
        memory_id: str,
        *,
        reason_code: str = "user_rejected",
        now: datetime | None = None,
    ) -> LongTermMemory:
        self._ensure_owner(scope)
        current = self._required(scope.namespace, memory_id)
        if current.status != MemoryStatus.PROPOSED:
            raise MemoryStateConflictError("only proposed memory can be rejected")
        replacement = current.model_copy(
            update={
                "status": MemoryStatus.REJECTED,
                "consent": MemoryConsent.WITHDRAWN,
                "content": "",
                "structured_value": {},
                "updated_at": now or utc_now(),
            }
        )
        return self._repository.replace(
            current=current,
            replacement=replacement,
            audit=self._audit(
                scope,
                memory_id,
                MemoryAuditAction.REJECTED,
                reason_code,
            ),
        )

    def edit(
        self,
        scope: MemoryAccessScope,
        memory_id: str,
        *,
        new_content: str,
        session_id: str,
        source_message_id: str,
        now: datetime | None = None,
    ) -> LongTermMemory:
        """编辑会创建新 memory 并 supersede 旧值，不原地改写历史正文。"""

        self._ensure_owner(scope)
        observed_at = now or utc_now()
        current = self._required(scope.namespace, memory_id)
        if current.status != MemoryStatus.ACTIVE:
            raise MemoryStateConflictError("only active memory can be edited")
        normalized = " ".join(new_content.split()).strip()
        if not normalized:
            raise MemoryPolicyRejectedError("edited memory content must not be empty")
        if _RESTRICTED_RE.search(normalized):
            raise MemoryPolicyRejectedError(
                "restricted credential or identity data cannot become memory"
            )
        classified_type, structured_value, _ = self._extractor._classify(normalized)
        if classified_type != current.type:
            structured_value = {
                "key": current.structured_value.get("key", current.conflict_key),
                "value": normalized,
            }
        provenance = MemoryProvenance(
            source_kind=MemorySourceKind.MANUAL_ENTRY,
            session_id=session_id,
            source_message_ids=(source_message_id,),
            extractor_version=self._extractor.version,
            source_excerpt_hash=provenance_hash(new_content),
        )
        candidate_id = memory_candidate_id(
            namespace=scope.namespace,
            memory_type=current.type,
            content=normalized,
            source_message_ids=provenance.source_message_ids,
        )
        proposed = LongTermMemory(
            memory_id=memory_id_for_candidate(candidate_id),
            candidate_id=candidate_id,
            namespace=scope.namespace,
            type=current.type,
            content=normalized,
            structured_value=structured_value,
            conflict_key=current.conflict_key,
            provenance=provenance,
            confidence=1.0,
            sensitivity=(
                MemorySensitivity.SENSITIVE
                if _SENSITIVE_RE.search(normalized)
                else current.sensitivity
            ),
            consent=MemoryConsent.EXPLICIT_REQUEST,
            status=MemoryStatus.PROPOSED,
            valid_from=observed_at,
            expires_at=observed_at + self._policy.ttl_for(current.type),
            created_by=scope.actor_user_id,
            supersedes_memory_id=current.memory_id,
            created_at=observed_at,
            updated_at=observed_at,
        )
        persisted = self._repository.create_proposed(
            proposed,
            self._audit(
                scope,
                proposed.memory_id,
                MemoryAuditAction.PROPOSED,
                "user_edit",
                metadata={"replaces_memory_id": current.memory_id},
            ),
        )
        return self.confirm(scope, persisted.memory_id, now=observed_at)

    def forget(
        self,
        scope: MemoryAccessScope,
        memory_id: str,
        *,
        reason_code: str = "user_requested",
        now: datetime | None = None,
    ) -> ForgetReceipt:
        self._ensure_owner(scope)
        observed_at = now or utc_now()
        current = self._required(scope.namespace, memory_id)
        if current.status == MemoryStatus.FORGOTTEN:
            audits = self._repository.list_audits(scope.namespace)
            prior = next(
                audit
                for audit in reversed(audits)
                if audit.memory_id == memory_id and audit.action == MemoryAuditAction.FORGOTTEN
            )
            return ForgetReceipt(
                memory_id=memory_id,
                forgotten=False,
                derivative_deletions=(),
                audit_id=prior.audit_id,
                forgotten_at=current.forgotten_at or observed_at,
            )
        forgotten = current.model_copy(
            update={
                "status": MemoryStatus.FORGOTTEN,
                "consent": MemoryConsent.WITHDRAWN,
                "content": "",
                "structured_value": {},
                "valid_to": observed_at,
                "forgotten_at": observed_at,
                "embedding_version": None,
                "updated_at": observed_at,
            }
        )
        audit = self._audit(
            scope,
            memory_id,
            MemoryAuditAction.FORGOTTEN,
            reason_code,
        )
        self._repository.replace(
            current=current,
            replacement=forgotten,
            audit=audit,
        )
        deleted = self._derivatives.delete(scope.namespace, memory_id)
        return ForgetReceipt(
            memory_id=memory_id,
            forgotten=True,
            derivative_deletions=deleted,
            audit_id=audit.audit_id,
            forgotten_at=observed_at,
        )

    def withdraw_consent(
        self,
        scope: MemoryAccessScope,
        memory_id: str,
        *,
        now: datetime | None = None,
    ) -> ForgetReceipt:
        return self.forget(
            scope,
            memory_id,
            reason_code="consent_withdrawn",
            now=now,
        )

    def retrieve(
        self,
        scope: MemoryAccessScope,
        query: str,
        *,
        token_budget: int | None = None,
        max_hits: int | None = None,
        now: datetime | None = None,
    ) -> MemoryContext:
        self._ensure_owner(scope)
        observed_at = now or utc_now()
        budget = (
            self._policy.max_context_tokens
            if token_budget is None
            else max(0, min(token_budget, self._policy.max_context_tokens))
        )
        hit_limit = (
            self._policy.max_hits
            if max_hits is None
            else max(0, min(max_hits, self._policy.max_hits))
        )
        memories = self._repository.list(
            scope.namespace,
            statuses=(MemoryStatus.ACTIVE,),
        )
        active: list[LongTermMemory] = []
        for memory in memories:
            if memory.expires_at <= observed_at:
                self._expire(scope, memory, observed_at)
            else:
                active.append(memory)

        query_terms = set(_TOKEN_RE.findall(query.casefold()))
        ranked: list[tuple[float, LongTermMemory, float]] = []
        for memory in active:
            content_terms = set(_TOKEN_RE.findall(memory.content.casefold()))
            overlap = len(query_terms & content_terms)
            union = len(query_terms | content_terms)
            relevance = overlap / union if union else 0.0
            if memory.type == MemoryType.PREFERENCE:
                relevance = max(relevance, 0.2)
            if relevance <= 0:
                continue
            age_days = max(
                0.0,
                (observed_at - memory.valid_from).total_seconds() / 86_400,
            )
            recency = 1 / (1 + age_days / 30)
            score = min(
                1.0,
                0.55 * relevance + 0.30 * memory.confidence + 0.15 * recency,
            )
            ranked.append((score, memory, relevance))
        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1].valid_from.timestamp(),
                item[1].memory_id,
            )
        )

        hits: list[MemoryRetrievalHit] = []
        lines: list[str] = []
        used_tokens = 0
        omitted = 0
        for score, memory, relevance in ranked:
            if len(hits) >= hit_limit:
                omitted += 1
                continue
            line = (
                f"- [{memory.type.value}] {memory.content} "
                f"(memory_id={memory.memory_id}, "
                f"source_messages={','.join(memory.provenance.source_message_ids)})"
            )
            line_tokens = _estimated_tokens(line)
            if used_tokens + line_tokens > budget:
                omitted += 1
                continue
            used_tokens += line_tokens
            lines.append(line)
            hits.append(
                MemoryRetrievalHit(
                    memory_id=memory.memory_id,
                    type=memory.type,
                    content=memory.content,
                    structured_value=memory.structured_value,
                    provenance=memory.provenance,
                    relevance=relevance,
                    confidence=memory.confidence,
                    score=score,
                    expires_at=memory.expires_at,
                )
            )
        return MemoryContext(
            namespace=scope.namespace,
            query=query,
            rendered="\n".join(lines),
            hits=tuple(hits),
            used_memory_ids=tuple(hit.memory_id for hit in hits),
            estimated_tokens=used_tokens,
            token_budget=budget,
            omitted_count=omitted,
        )

    def list_memories(
        self,
        scope: MemoryAccessScope,
        *,
        include_inactive: bool = False,
    ) -> tuple[LongTermMemory, ...]:
        self._ensure_owner(scope)
        statuses = None if include_inactive else (MemoryStatus.PROPOSED, MemoryStatus.ACTIVE)
        return self._repository.list(scope.namespace, statuses=statuses)

    def export(self, scope: MemoryAccessScope) -> tuple[dict, ...]:
        self._ensure_owner(scope)
        memories = self._repository.list(scope.namespace)
        # Forgotten/rejected rows remain as content-free deletion proofs.
        exported = tuple(memory.model_dump(mode="json") for memory in memories)
        self._repository.append_audit(
            self._audit(
                scope,
                memory_id=f"export_{uuid4().hex[:24]}",
                action=MemoryAuditAction.EXPORTED,
                reason_code="user_export",
                metadata={"record_count": len(exported)},
            )
        )
        return exported

    def cleanup_expired(
        self,
        scope: MemoryAccessScope,
        *,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        self._ensure_owner(scope)
        observed_at = now or utc_now()
        expired: list[str] = []
        for memory in self._repository.list(
            scope.namespace,
            statuses=(MemoryStatus.ACTIVE, MemoryStatus.PROPOSED),
        ):
            if memory.expires_at <= observed_at:
                self._expire(scope, memory, observed_at)
                expired.append(memory.memory_id)
        return tuple(expired)

    def _expire(
        self,
        scope: MemoryAccessScope,
        memory: LongTermMemory,
        now: datetime,
    ) -> None:
        expired = memory.model_copy(
            update={
                "status": MemoryStatus.EXPIRED,
                "content": "",
                "structured_value": {},
                "valid_to": now,
                "embedding_version": None,
                "updated_at": now,
            }
        )
        self._repository.replace(
            current=memory,
            replacement=expired,
            audit=self._audit(
                scope,
                memory.memory_id,
                MemoryAuditAction.EXPIRED,
                "ttl_elapsed",
            ),
        )
        self._derivatives.delete(scope.namespace, memory.memory_id)

    @staticmethod
    def _ensure_owner(scope: MemoryAccessScope) -> None:
        if scope.actor_user_id != scope.subject_user_id:
            # 管理员也不能默认读取个人记忆正文。
            raise MemoryAccessDeniedError("actor cannot access another user's long-term memory")

    def _required(
        self,
        namespace: MemoryNamespace,
        memory_id: str,
    ) -> LongTermMemory:
        memory = self._repository.get(namespace, memory_id)
        if memory is None:
            # 作用域外资源也返回 not found，不泄露其存在性。
            raise MemoryNotFoundError("memory was not found in the requested scope")
        return memory

    @staticmethod
    def _audit(
        scope: MemoryAccessScope,
        memory_id: str,
        action: MemoryAuditAction,
        reason_code: str,
        *,
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> MemoryAuditEvent:
        return MemoryAuditEvent(
            audit_id=f"maud_{uuid4().hex}",
            namespace=scope.namespace,
            memory_id=memory_id,
            actor_user_id=scope.actor_user_id,
            action=action,
            reason_code=reason_code,
            metadata=metadata or {},
        )


def _estimated_tokens(text: str) -> int:
    # 保守估计中文约 1 字/token、Latin 约 4 字符/token，永不返回 0。
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin = len(text) - cjk
    return max(1, cjk + math.ceil(latin / 4))


__all__ = [
    "LongTermMemoryService",
    "MemoryAccessDeniedError",
    "MemoryCandidateExtractor",
    "MemoryDerivativeStore",
    "MemoryNotFoundError",
    "MemoryPolicy",
    "MemoryPolicyRejectedError",
    "MemoryProposalResult",
    "MemoryRepository",
    "MemoryServiceError",
    "MemoryStateConflictError",
]
