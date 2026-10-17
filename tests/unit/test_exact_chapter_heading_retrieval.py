from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.application.retrieval_service as retrieval_service
import app.rag.query.embedding_search_service as embedding_search_service
from app.application.query_understanding_service import extract_chapter_title_references
from app.infra.persistence.sql.models import Base, EvidenceRecordModel
from app.infra.persistence.sql.repositories import ControlRepository
from app.infra.retrieval.control_plane_resolvers import (
    active_scope_digest,
    serialize_active_scopes,
)
from app.infra.retrieval.milvus_adapter import ActiveIndexScope
from app.security.principal import Principal


def _active_scope(
    *,
    document_id: str = "doc-synthetic-a",
    document_version_id: str = "dv-synthetic-a-v2",
    index_version_id: str = "idx-synthetic-a-v2",
) -> ActiveIndexScope:
    return ActiveIndexScope(
        tenant_id="tenant-synthetic",
        kb_id="kb-synthetic",
        document_id=document_id,
        document_version_id=document_version_id,
        index_version_id=index_version_id,
        collection_name="synthetic_chunks",
    )


def _frozen_state(*scopes: ActiveIndexScope, doc_ids: list[str] | None = None) -> dict:
    selected = list(scopes or (_active_scope(),))
    return {
        "kb_id": "kb-synthetic",
        "doc_ids": list(doc_ids or []),
        "vector_read_scopes": serialize_active_scopes(selected),
        "vector_read_set_digest": active_scope_digest(selected),
    }


def _evidence_row(
    *,
    evidence_id: str,
    document_id: str,
    document_version_id: str,
    index_version_id: str,
    chunk_id: str,
    heading_path: list[str],
) -> EvidenceRecordModel:
    return EvidenceRecordModel(
        id=evidence_id,
        schema_version="evidence.v1",
        source_type="local",
        tenant_id="tenant-synthetic",
        kb_id="kb-synthetic",
        document_id=document_id,
        document_version_id=document_version_id,
        index_version_id=index_version_id,
        chunk_id=chunk_id,
        content=f"{' / '.join(heading_path)}\n合成章节正文。",
        content_hash="sha256:" + evidence_id[-1] * 64,
        source_block_ids_json=[f"block-{chunk_id}"],
        provenance_spans_json=[
            {
                "block_id": f"block-{chunk_id}",
                "availability": "unavailable",
                "coordinate_space": "none",
                "reason_code": "synthetic_fixture",
            }
        ],
        heading_path_json=heading_path,
        asset_refs_json=[],
        source_label="synthetic-manual.pdf",
        primary_block_id=f"block-{chunk_id}",
        primary_availability="unavailable",
        primary_coordinate_space="none",
        primary_reason_code="synthetic_fixture",
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("第７．３．２节的标题是什么？", ("7.3.2",)),
        ("What is the heading of section 9.4?", ("9.4",)),
    ],
)
def test_extract_chapter_title_references_requires_anchored_title_intent(
    query: str,
    expected: tuple[str, ...],
) -> None:
    assert extract_chapter_title_references(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "版本 7.3.2 的标题是什么？",
        "IP 地址 172.20.3.4 的标题是什么？",
        "7.3.2节包含哪些内容？",
        "7.3.2节包含哪些内容？文档标题是什么？",
        "What is the document title? See section 9.4.",
        "第7章第3节第2节的题名是什么？",
    ],
)
def test_extract_chapter_title_references_rejects_numeric_false_positives(query: str) -> None:
    assert extract_chapter_title_references(query) == ()


def test_repository_exact_heading_lookup_is_terminal_and_frozen_scope_bound() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, tables=[EvidenceRecordModel.__table__])
    rows = [
        _evidence_row(
            evidence_id="ev_" + "1" * 40,
            document_id="doc-synthetic-a",
            document_version_id="dv-synthetic-a-v2",
            index_version_id="idx-synthetic-a-v2",
            chunk_id="chunk-exact",
            heading_path=["7.3.2. Synthetic terminal heading"],
        ),
        _evidence_row(
            evidence_id="ev_" + "2" * 40,
            document_id="doc-synthetic-a",
            document_version_id="dv-synthetic-a-v2",
            index_version_id="idx-synthetic-a-v2",
            chunk_id="chunk-descendant",
            heading_path=[
                "7.3.2. Synthetic terminal heading",
                "7.3.2.1. Synthetic child heading",
            ],
        ),
        _evidence_row(
            evidence_id="ev_" + "3" * 40,
            document_id="doc-synthetic-a",
            document_version_id="dv-synthetic-a-v1",
            index_version_id="idx-synthetic-a-v1",
            chunk_id="chunk-stale-version",
            heading_path=["7.3.2. Stale heading"],
        ),
        _evidence_row(
            evidence_id="ev_" + "4" * 40,
            document_id="doc-synthetic-b",
            document_version_id="dv-synthetic-b-v2",
            index_version_id="idx-synthetic-b-v2",
            chunk_id="chunk-other-document",
            heading_path=["7.3.2. Other document heading"],
        ),
        _evidence_row(
            evidence_id="ev_" + "5" * 40,
            document_id="doc-synthetic-a",
            document_version_id="dv-synthetic-a-v2",
            index_version_id="idx-synthetic-a-v2",
            chunk_id="chunk-inherited-heading",
            heading_path=["7.3.2. Synthetic terminal heading"],
        ),
        _evidence_row(
            evidence_id="ev_" + "6" * 40,
            document_id="doc-synthetic-a",
            document_version_id="dv-synthetic-a-v2",
            index_version_id="idx-synthetic-a-v2",
            chunk_id="chunk-conflicting-heading",
            heading_path=["7.3.2. Conflicting terminal heading"],
        ),
    ]
    try:
        with Session(engine) as session:
            session.add_all(rows)
            session.commit()
            repository = ControlRepository(session, tenant_id="tenant-synthetic")

            matches, scanned_count, truncated = repository.find_exact_terminal_heading_evidence(
                chapter_paths=("7.3.2",),
                scope_identities=(
                    (
                        "tenant-synthetic",
                        "kb-synthetic",
                        "doc-synthetic-a",
                        "dv-synthetic-a-v2",
                        "idx-synthetic-a-v2",
                    ),
                ),
                document_ids=("doc-synthetic-a",),
            )
            limited_matches, limited_scanned_count, limited_truncated = (
                repository.find_exact_terminal_heading_evidence(
                    chapter_paths=("7.3.2",),
                    scope_identities=(
                        (
                            "tenant-synthetic",
                            "kb-synthetic",
                            "doc-synthetic-a",
                            "dv-synthetic-a-v2",
                            "idx-synthetic-a-v2",
                        ),
                    ),
                    scan_limit=1,
                )
            )
    finally:
        engine.dispose()

    assert [row.chunk_id for row in matches] == [
        "chunk-exact",
        "chunk-inherited-heading",
        "chunk-conflicting-heading",
    ]
    assert scanned_count == 4
    assert truncated is False
    assert limited_matches == []
    assert limited_scanned_count == 1
    assert limited_truncated is True


def _evidence_payload(row: SimpleNamespace) -> dict:
    suffix = str(row.document_id)[-1]
    return {
        "schema_version": "evidence.v1",
        "evidence_id": f"ev_{suffix * 40}",
        "source": {
            "source_type": "local",
            "tenant_id": "tenant-synthetic",
            "kb_id": "kb-synthetic",
            "document_id": row.document_id,
            "document_version_id": row.document_version_id,
            "index_version_id": row.index_version_id,
            "chunk_id": f"chunk-{suffix}",
            "source_block_ids": [f"block-{suffix}"],
        },
        "content": "9.7.4. Synthetic lookup heading\nSynthetic body.",
        "content_hash": "sha256:" + suffix * 64,
        "provenance_spans": [
            {
                "block_id": f"block-{suffix}",
                "availability": "unavailable",
                "coordinate_space": "none",
                "reason_code": "synthetic_fixture",
            }
        ],
        "heading_path": ["9.7.4. Synthetic lookup heading"],
        "asset_refs": [],
        "source_label": "synthetic-manual.pdf",
    }


def test_exact_heading_service_is_noop_for_non_title_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"sentinel": ["unchanged"]}
    monkeypatch.setattr(
        retrieval_service,
        "current_principal",
        lambda: pytest.fail("非标题查询不应读取认证主体或访问检索范围"),
    )

    candidates, diagnostics = retrieval_service.retrieve_exact_chapter_heading_candidates(
        state,
        query_text="7.3.2节包含哪些内容？",
    )

    assert candidates == []
    assert diagnostics["status"] == "not_applicable"
    assert state == {"sentinel": ["unchanged"]}


def test_exact_heading_service_uses_frozen_identities_and_emits_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    frozen = _active_scope()
    row = SimpleNamespace(
        document_id=frozen.document_id,
        document_version_id=frozen.document_version_id,
        index_version_id=frozen.index_version_id,
    )

    class FakeRepository:
        def __init__(self, _session, *, tenant_id: str) -> None:
            captured["tenant_id"] = tenant_id

        def find_exact_terminal_heading_evidence(self, **kwargs):
            captured.update(kwargs)
            return [row], 12, False

        @staticmethod
        def evidence_dict(value):
            return _evidence_payload(value)

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr(
        retrieval_service,
        "current_principal",
        lambda: Principal(
            user_id="user-synthetic",
            tenant_id="tenant-synthetic",
            roles=("viewer",),
            scopes=("kb:read",),
            auth_type="test",
            credential_id="credential-synthetic",
        ),
    )
    monkeypatch.setattr(retrieval_service, "session_scope", fake_session_scope)
    monkeypatch.setattr(retrieval_service, "ControlRepository", FakeRepository)

    candidates, diagnostics = retrieval_service.retrieve_exact_chapter_heading_candidates(
        _frozen_state(frozen, doc_ids=[frozen.document_id]),
        query_text="9.7.4章的标题是什么？",
        sub_query_id="sq-synthetic",
    )

    assert captured["scope_identities"] == (
        (
            frozen.tenant_id,
            frozen.kb_id,
            frozen.document_id,
            frozen.document_version_id,
            frozen.index_version_id,
        ),
    )
    assert captured["document_ids"] == (frozen.document_id,)
    assert candidates[0]["chunk_id"] == "chunk-a"
    assert candidates[0]["retrieval_observations"] == [
        {
            "route": "base.heading_exact",
            "raw_score": 1.0,
            "rank": 1,
            "metadata": {"match_kind": "exact_terminal_heading"},
            "sub_query_id": "sq-synthetic",
        }
    ]
    assert diagnostics == {
        "route": "base.heading_exact",
        "requested_paths": ["9.7.4"],
        "candidate_count": 1,
        "status": "succeeded",
        "scanned_count": 12,
        "matched_row_count": 1,
        "matched_document_count": 1,
    }


@pytest.mark.parametrize(
    "rows",
    [
        [
            SimpleNamespace(
                document_id="doc-synthetic-a",
                chunk_id="chunk-owner",
                heading_path_json=["9.7.4. Synthetic heading"],
            ),
            SimpleNamespace(
                document_id="doc-synthetic-a",
                chunk_id="chunk-inherited",
                heading_path_json=["9.7.4. Synthetic heading"],
            ),
        ],
        [
            SimpleNamespace(
                document_id="doc-synthetic-a",
                chunk_id="chunk-first-title",
                heading_path_json=["9.7.4. First synthetic heading"],
            ),
            SimpleNamespace(
                document_id="doc-synthetic-a",
                chunk_id="chunk-conflicting-title",
                heading_path_json=["9.7.4. Conflicting synthetic heading"],
            ),
        ],
    ],
    ids=["inherited-heading-multiple-chunks", "same-number-conflicting-titles"],
)
def test_exact_heading_service_falls_back_when_one_document_has_multiple_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[SimpleNamespace],
) -> None:
    class FakeRepository:
        def __init__(self, _session, *, tenant_id: str) -> None:
            assert tenant_id == "tenant-synthetic"

        def find_exact_terminal_heading_evidence(self, **_kwargs):
            return rows, 20, False

        @staticmethod
        def evidence_dict(_value):
            pytest.fail("歧义 Evidence 不得通过 rows[0] 生成候选")

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr(
        retrieval_service,
        "current_principal",
        lambda: Principal(
            user_id="user-synthetic",
            tenant_id="tenant-synthetic",
            roles=("viewer",),
            scopes=("kb:read",),
            auth_type="test",
            credential_id="credential-synthetic",
        ),
    )
    monkeypatch.setattr(retrieval_service, "session_scope", fake_session_scope)
    monkeypatch.setattr(retrieval_service, "ControlRepository", FakeRepository)

    candidates, diagnostics = retrieval_service.retrieve_exact_chapter_heading_candidates(
        _frozen_state(_active_scope()),
        query_text="9.7.4节的题名是什么？",
    )

    assert candidates == []
    assert diagnostics["status"] == "fallback"
    assert diagnostics["reason"] == "ambiguous_heading_evidence"
    assert diagnostics["matched_row_count"] == 2
    assert diagnostics["matched_document_count"] == 1


def test_exact_heading_service_falls_back_on_cross_document_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scopes = (
        _active_scope(),
        _active_scope(
            document_id="doc-synthetic-b",
            document_version_id="dv-synthetic-b-v2",
            index_version_id="idx-synthetic-b-v2",
        ),
    )

    class FakeRepository:
        def __init__(self, _session, *, tenant_id: str) -> None:
            assert tenant_id == "tenant-synthetic"

        def find_exact_terminal_heading_evidence(self, **_kwargs):
            return (
                [
                    SimpleNamespace(document_id="doc-synthetic-a"),
                    SimpleNamespace(document_id="doc-synthetic-b"),
                ],
                20,
                False,
            )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr(
        retrieval_service,
        "current_principal",
        lambda: Principal(
            user_id="user-synthetic",
            tenant_id="tenant-synthetic",
            roles=("viewer",),
            scopes=("kb:read",),
            auth_type="test",
            credential_id="credential-synthetic",
        ),
    )
    monkeypatch.setattr(retrieval_service, "session_scope", fake_session_scope)
    monkeypatch.setattr(retrieval_service, "ControlRepository", FakeRepository)

    candidates, diagnostics = retrieval_service.retrieve_exact_chapter_heading_candidates(
        _frozen_state(*scopes),
        query_text="9.7.4节的题名是什么？",
    )

    assert candidates == []
    assert diagnostics["status"] == "fallback"
    assert diagnostics["reason"] == "ambiguous_documents"
    assert diagnostics["matched_document_count"] == 2


def test_embedding_search_prepends_exact_heading_and_deduplicates_vector_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = {"chunk_id": "chunk-exact", "content": "exact"}
    vector_duplicate = {"chunk_id": "chunk-exact", "content": "vector duplicate"}
    vector_other = {"chunk_id": "chunk-other", "content": "other"}

    monkeypatch.setattr(
        embedding_search_service,
        "retrieve_candidates",
        lambda *_args, **_kwargs: (
            [vector_duplicate, vector_other],
            {"trace_id": "trace-synthetic"},
        ),
    )
    monkeypatch.setattr(
        embedding_search_service,
        "retrieve_exact_chapter_heading_candidates",
        lambda *_args, **_kwargs: (
            [exact],
            {"status": "succeeded", "route": "base.heading_exact"},
        ),
    )
    state = {"original_query": "9.7.4章的标题是什么？"}

    chunks = embedding_search_service.search_by_embedding(state)

    assert [item["chunk_id"] for item in chunks] == ["chunk-exact", "chunk-other"]
    assert chunks[0]["content"] == "exact"
    diagnostics = state["embedding_retrieval_diagnostics"]["subqueries"][0]
    assert diagnostics["heading_exact"] == {
        "status": "succeeded",
        "route": "base.heading_exact",
    }
