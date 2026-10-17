"""把 SQL active index/document binding 编译为 Milvus 强制可见性过滤器。"""

from __future__ import annotations

import json

from app.application.index_lifecycle_service import active_index_scopes


def _literal(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def load_active_scopes(kb_id: str | None = None) -> list[dict]:
    return active_index_scopes(kb_id)


def build_active_scope_expr(
    scopes: list[dict], doc_ids: list[str] | None = None
) -> str:
    allowed_docs = set(doc_ids or [])
    clauses: list[str] = []
    for scope in scopes:
        if allowed_docs and scope["document_id"] not in allowed_docs:
            continue
        clauses.append(
            "(" + " and ".join(
                [
                    f'tenant_id == {_literal(scope["tenant_id"])}',
                    f'kb_id == {_literal(scope["kb_id"])}',
                    f'doc_id == {_literal(scope["document_id"])}',
                    f'document_version_id == {_literal(scope["document_version_id"])}',
                    f'index_version_id == {_literal(scope["index_version_id"])}',
                ]
            ) + ")"
        )
    if not clauses:
        return 'available_int == 1 and doc_id == "__stt_no_active_document__"'
    return f"available_int == 1 and ({' or '.join(clauses)})"
