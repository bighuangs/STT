from __future__ import annotations

from app.domain.evidence import (
    Evidence,
    EvidenceLocationSpan,
    EvidenceSourceIdentity,
)
from app.rag.query import evidence_service


KB_ID = "kb-read-model"
ACTIVE_SCOPE = (
    "tenant_default",
    KB_ID,
    "doc-read-model",
    "dv-read-model",
    "idx-read-model",
)


def _repository_shaped_evidence() -> dict:
    domain = Evidence(
        source=EvidenceSourceIdentity(
            source_type="local",
            tenant_id=ACTIVE_SCOPE[0],
            kb_id=ACTIVE_SCOPE[1],
            document_id=ACTIVE_SCOPE[2],
            document_version_id=ACTIVE_SCOPE[3],
            index_version_id=ACTIVE_SCOPE[4],
            chunk_id="chk-read-model",
            source_block_ids=["block-1"],
        ),
        content="使用 HAK180 前应检查设备接地和安全防护。",
        provenance_spans=[
            EvidenceLocationSpan(
                block_id="block-1",
                availability="exact",
                coordinate_space="normalized_page",
                page_no=3,
                bbox=(0.1, 0.2, 0.8, 0.4),
            )
        ],
        source_label="hak180产品安全手册.pdf",
    ).model_dump(mode="json", exclude_none=True)
    return {
        **domain,
        "created_at": "2026-08-07T03:11:14Z",
        "updated_at": "2026-08-07T03:11:15Z",
        "primary_location": {
            "block_id": "block-1",
            "availability": "exact",
            "coordinate_space": "normalized_page",
            "page_no": 3,
            "bbox": [0.1, 0.2, 0.8, 0.4],
        },
    }


def _install_active_repository_fakes(monkeypatch, evidence: dict) -> None:
    monkeypatch.setattr(evidence_service, "_active_scope_keys", lambda _kb_id: {ACTIVE_SCOPE})
    monkeypatch.setattr(
        evidence_service,
        "_load_evidence_rows",
        lambda evidence_ids: (
            {evidence["evidence_id"]: evidence}
            if evidence["evidence_id"] in evidence_ids
            else {}
        ),
    )


def test_repository_read_model_becomes_downstream_safe_domain_evidence(monkeypatch) -> None:
    repository_evidence = _repository_shaped_evidence()
    _install_active_repository_fakes(monkeypatch, repository_evidence)

    prepared = evidence_service.prepare_reranked_evidence(
        [
            {
                "evidence_id": repository_evidence["evidence_id"],
                "evidence": repository_evidence,
                "rerank_score": 0.91,
                "retrieval_route": "local_hybrid",
            }
        ],
        kb_id=KB_ID,
    )

    assert len(prepared) == 1
    restored = prepared[0]["evidence"]
    assert restored["evidence_id"] == repository_evidence["evidence_id"]
    assert restored["content_hash"] == repository_evidence["content_hash"]
    # read-model 展示字段不属于 Evidence 领域契约；页码和坐标事实由
    # provenance_spans 无损保留，且产物可继续通过 Context Pack/Citation 的
    # extra="forbid" 校验。
    assert "primary_location" not in restored
    assert "created_at" not in restored
    assert "updated_at" not in restored
    assert restored["provenance_spans"][0] == {
        "block_id": "block-1",
        "availability": "exact",
        "coordinate_space": "normalized_page",
        "page_no": 3,
        "bbox": [0.1, 0.2, 0.8, 0.4],
    }
    Evidence.model_validate(restored)
    assert restored["retrieval"]["rerank_score"] == 0.91


def test_read_model_projection_does_not_bypass_content_hash_validation(monkeypatch) -> None:
    repository_evidence = _repository_shaped_evidence()
    _install_active_repository_fakes(monkeypatch, repository_evidence)
    tampered = {**repository_evidence, "content": "被替换的正文"}

    prepared = evidence_service.prepare_reranked_evidence(
        [{"evidence_id": tampered["evidence_id"], "evidence": tampered}],
        kb_id=KB_ID,
    )

    assert prepared == []


def test_read_model_projection_does_not_bypass_active_scope_validation(monkeypatch) -> None:
    repository_evidence = _repository_shaped_evidence()
    monkeypatch.setattr(evidence_service, "_active_scope_keys", lambda _kb_id: set())

    prepared = evidence_service.prepare_reranked_evidence(
        [
            {
                "evidence_id": repository_evidence["evidence_id"],
                "evidence": repository_evidence,
            }
        ],
        kb_id=KB_ID,
    )

    assert prepared == []
