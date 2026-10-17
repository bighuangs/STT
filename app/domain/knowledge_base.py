"""知识库领域状态。"""

from enum import StrEnum

from app.domain._state_machine import ensure_transition


class KnowledgeBaseStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DELETED = "deleted"


_TRANSITIONS = {
    KnowledgeBaseStatus.ACTIVE: {
        KnowledgeBaseStatus.DISABLED,
        KnowledgeBaseStatus.DELETED,
    },
    KnowledgeBaseStatus.DISABLED: {
        KnowledgeBaseStatus.ACTIVE,
        KnowledgeBaseStatus.DELETED,
    },
    KnowledgeBaseStatus.DELETED: set(),
}


def ensure_knowledge_base_transition(current: str, target: str) -> None:
    ensure_transition(
        KnowledgeBaseStatus(current), KnowledgeBaseStatus(target), _TRANSITIONS
    )
