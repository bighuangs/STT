"""文档与文档版本领域状态。"""

from enum import StrEnum

from app.domain._state_machine import ensure_transition


class DocumentStatus(StrEnum):
    DRAFT = "draft"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"
    DELETED = "deleted"


class DocumentVersionStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STAGED = "staged"
    PUBLISHED = "published"
    FAILED = "failed"


_DOCUMENT_TRANSITIONS = {
    DocumentStatus.DRAFT: {DocumentStatus.PROCESSING, DocumentStatus.DELETED},
    DocumentStatus.PROCESSING: {
        DocumentStatus.READY,
        DocumentStatus.FAILED,
        DocumentStatus.DELETED,
    },
    DocumentStatus.READY: {
        DocumentStatus.PROCESSING,
        DocumentStatus.DISABLED,
        DocumentStatus.DELETED,
    },
    DocumentStatus.FAILED: {DocumentStatus.PROCESSING, DocumentStatus.DELETED},
    DocumentStatus.DISABLED: {DocumentStatus.READY, DocumentStatus.DELETED},
    DocumentStatus.DELETED: set(),
}

_VERSION_TRANSITIONS = {
    DocumentVersionStatus.UPLOADED: {
        DocumentVersionStatus.PARSING,
        DocumentVersionStatus.FAILED,
    },
    DocumentVersionStatus.PARSING: {
        DocumentVersionStatus.CHUNKING,
        DocumentVersionStatus.FAILED,
    },
    DocumentVersionStatus.CHUNKING: {
        DocumentVersionStatus.EMBEDDING,
        DocumentVersionStatus.FAILED,
    },
    DocumentVersionStatus.EMBEDDING: {
        DocumentVersionStatus.STAGED,
        DocumentVersionStatus.FAILED,
    },
    DocumentVersionStatus.STAGED: {
        DocumentVersionStatus.PUBLISHED,
        DocumentVersionStatus.FAILED,
    },
    DocumentVersionStatus.PUBLISHED: set(),
    DocumentVersionStatus.FAILED: {DocumentVersionStatus.PARSING},
}


def ensure_document_transition(current: str, target: str) -> None:
    ensure_transition(DocumentStatus(current), DocumentStatus(target), _DOCUMENT_TRANSITIONS)


def ensure_document_version_transition(current: str, target: str) -> None:
    ensure_transition(
        DocumentVersionStatus(current),
        DocumentVersionStatus(target),
        _VERSION_TRANSITIONS,
    )
