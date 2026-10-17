"""
历史记录仓储模块，统一封装聊天记录的读写操作。
"""
from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    save_chat_message,
)
from app.security.principal import current_principal


class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        principal = current_principal()
        return get_recent_messages(
            session_id, limit=limit,
            tenant_id=principal.tenant_id if principal else None,
            user_id=principal.user_id if principal else None,
        )

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        image_urls: list[str] | None = None,
        message_id: str | None = None,
        answer_id: str | None = None,
        citations: list[dict] | None = None,
        sources: list[dict] | None = None,
        citation_validation: dict | None = None,
        retrieval_profile: dict | None = None,
        retrieval_diagnostics: dict | None = None,
        answer_envelope: dict | None = None,
    ) -> str:
        principal = current_principal()
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            image_urls=image_urls,
            message_id=message_id,
            answer_id=answer_id,
            citations=citations,
            sources=sources,
            citation_validation=citation_validation,
            retrieval_profile=retrieval_profile,
            retrieval_diagnostics=retrieval_diagnostics,
            answer_envelope=answer_envelope,
            tenant_id=principal.tenant_id if principal else None,
            user_id=principal.user_id if principal else None,
        )

    def clear_session(self, session_id: str) -> int:
        principal = current_principal()
        return clear_history(
            session_id,
            tenant_id=principal.tenant_id if principal else None,
            user_id=principal.user_id if principal else None,
        )


history_repository = HistoryRepository()
