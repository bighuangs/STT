from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import inspect

from app.application import control_plane_service as service
from app.domain.evidence import (
    AnswerSpan,
    CitationFact,
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
    sha256_digest,
)
from app.infra.persistence.sql.database import configure_database, session_scope
from app.infra.persistence.sql.repositories import (
    ResourceConflictError,
    ResourceNotFoundError,
)


@pytest.fixture()
def control_db(tmp_path: Path):
    db_path = tmp_path / "todo08.db"
    configure_database(f"sqlite:///{db_path.as_posix()}")
    service.bootstrap_control_plane()
    return db_path


def _registered(tmp_path: Path) -> dict:
    source = tmp_path / "manual.md"
    payload = b"# Safety\nGround the machine before startup."
    source.write_bytes(payload)
    return service.register_uploaded_document(
        kb_id="kb_default",
        logical_name=source.name,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        mime_type="text/markdown",
        object_key=str(source),
    )


def _local_evidence(registered: dict, content: str, *, chunk_id: str) -> Evidence:
    return Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id="tenant_default",
            kb_id="kb_default",
            document_id=registered["document"]["id"],
            document_version_id=registered["document_version"]["id"],
            index_version_id=registered["index_version"]["id"],
            chunk_id=chunk_id,
            source_block_ids=["block-1"],
        ),
        content=content,
        provenance_spans=[
            EvidenceLocationSpan(
                block_id="block-1",
                availability="exact",
                coordinate_space="text_lines",
                line_start=2,
                line_end=2,
            )
        ],
        source_label="manual.md",
    )


def _publish(registered: dict) -> None:
    job_id = registered["ingestion_job"]["id"]
    service.start_ingestion_job(job_id)
    service.complete_ingestion_job(job_id, chunk_count=1)


def test_staging_evidence_is_idempotent_and_citation_quote_is_checked(
    control_db: Path, tmp_path: Path
) -> None:
    registered = _registered(tmp_path)
    content = "Ground the machine before startup."
    evidence = _local_evidence(registered, content, chunk_id="chk_grounding")
    first = service.record_evidence_batch([evidence], tenant_id="tenant_default")
    second = service.record_evidence_batch([evidence], tenant_id="tenant_default")
    assert first[0]["evidence_id"] == second[0]["evidence_id"]
    _publish(registered)

    quote = "Ground the machine"
    citation = CitationFact(
        answer_id="ans_persist",
        evidence_id=evidence.evidence_id,
        answer_span=AnswerSpan(start=0, end=12),
        source_quote=quote,
        quote_hash=sha256_digest(quote),
        source_label="manual.md",
        validation_status="valid",
    )
    saved = service.record_citation_fact(citation, tenant_id="tenant_default")
    assert saved["citation_id"] == citation.citation_id
    assert service.list_citation_facts("ans_persist", tenant_id="tenant_default")

    with pytest.raises(ResourceNotFoundError):
        service.get_evidence(evidence.evidence_id, tenant_id="tenant_other")


def test_citation_bundle_is_atomically_idempotent(
    control_db: Path, tmp_path: Path
) -> None:
    registered = _registered(tmp_path)
    _publish(registered)
    content = "Ground the machine before startup."
    evidence = _local_evidence(registered, content, chunk_id="chk_bundle_ok")
    quote = "Ground the machine"
    citation = CitationFact(
        answer_id="ans_bundle_ok",
        evidence_id=evidence.evidence_id,
        answer_span=AnswerSpan(start=0, end=18),
        source_quote=quote,
        quote_hash=sha256_digest(quote),
        source_label="manual.md",
        validation_status="valid",
    )

    first = service.record_citation_bundle(
        [evidence], [citation], tenant_id="tenant_default"
    )
    second = service.record_citation_bundle(
        [evidence], [citation], tenant_id="tenant_default"
    )

    assert first == second
    assert [item["evidence_id"] for item in first["evidences"]] == [
        evidence.evidence_id
    ]
    assert [item["citation_id"] for item in first["citations"]] == [
        citation.citation_id
    ]
    assert len(
        service.list_evidence(
            registered["document_version"]["id"],
            index_version_id=registered["index_version"]["id"],
            tenant_id="tenant_default",
        )
    ) == 1
    assert len(
        service.list_citation_facts(
            citation.answer_id, tenant_id="tenant_default"
        )
    ) == 1


def test_citation_bundle_rolls_back_everything_when_second_citation_fails(
    control_db: Path, tmp_path: Path
) -> None:
    registered = _registered(tmp_path)
    _publish(registered)
    content = "Ground the machine before startup."
    evidence = _local_evidence(registered, content, chunk_id="chk_bundle_rollback")
    valid_quote = "Ground the machine"
    valid = CitationFact(
        answer_id="ans_bundle_rollback",
        evidence_id=evidence.evidence_id,
        answer_span=AnswerSpan(start=0, end=18),
        source_quote=valid_quote,
        quote_hash=sha256_digest(valid_quote),
        source_label="manual.md",
        validation_status="valid",
        ordinal=1,
    )
    absent_quote = "Disconnect the imaginary battery"
    second_fails_repository_validation = CitationFact(
        answer_id="ans_bundle_rollback",
        evidence_id=evidence.evidence_id,
        answer_span=AnswerSpan(start=19, end=35),
        source_quote=absent_quote,
        quote_hash=sha256_digest(absent_quote),
        source_label="manual.md",
        validation_status="valid",
        ordinal=2,
    )

    with pytest.raises(
        ResourceConflictError,
        match="source_quote is not present",
    ):
        service.record_citation_bundle(
            [evidence],
            [valid, second_fails_repository_validation],
            tenant_id="tenant_default",
        )

    assert (
        service.list_evidence(
            registered["document_version"]["id"],
            index_version_id=registered["index_version"]["id"],
            tenant_id="tenant_default",
        )
        == []
    )
    assert (
        service.list_citation_facts(
            "ans_bundle_rollback", tenant_id="tenant_default"
        )
        == []
    )


def test_web_evidence_is_tenant_scoped(control_db: Path) -> None:
    content = "Vendor advisory snapshot."
    evidence = Evidence(
        source=EvidenceSourceIdentity(
            source_type="web",
            tenant_id="tenant_default",
            source_url="https://example.test/advisory",
            content_hash=sha256_digest(content),
            fetched_at=datetime(2026, 7, 23, tzinfo=UTC),
        ),
        content=content,
        source_label="Vendor advisory",
    )

    saved = service.record_evidence(evidence, tenant_id="tenant_default")
    assert (
        service.get_evidence(evidence.evidence_id, tenant_id="tenant_default")[
            "evidence_id"
        ]
        == saved["evidence_id"]
    )
    with pytest.raises(ResourceNotFoundError):
        service.get_evidence(evidence.evidence_id, tenant_id="tenant_other")


def test_migration_creates_evidence_and_citation_tables(control_db: Path) -> None:
    with session_scope() as session:
        tables = set(inspect(session.bind).get_table_names())
    assert {"evidence_records", "citation_facts"} <= tables
