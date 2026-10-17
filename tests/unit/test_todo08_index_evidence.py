from __future__ import annotations

import pytest

from app.rag.import_ import index_service


def _state() -> dict:
    return {
        "tenant_id": "tenant_default",
        "kb_id": "kb_demo",
        "document_id": "doc_demo",
        "document_version_id": "dver_demo",
        "index_version_id": "idx_demo",
        "file_title": "HAK180 产品安全手册.pdf",
        "collection_name": "chunks_idx_demo",
        "vector_dim": 3,
        "chunks": [
            {
                "id": "chk_todo07_stable",
                "chunk_id": "chk_todo07_stable",
                "content": "额定工作压力为 16 MPa。",
                "type": "text",
                "dense_vector": [0.1, 0.2, 0.3],
                "sparse_vector": {1: 0.7},
                "images": ["assets/pressure.png"],
                "img_id": "imagetemps-doc_demo/pressure.png",
                "img_ids": [
                    "imagetemps-doc_demo/pressure.png",
                    "imagetemps-doc_demo/pressure-chart.png",
                ],
                "provenance": {
                    "source_block_ids": ["block-heading", "block-pressure"],
                    "locations": [
                        {
                            "block_id": "block-pressure",
                            "page_no": 12,
                            "bbox": [10, 20, 100, 80],
                            "coordinate_space": "pdf_points",
                        }
                    ],
                    "heading_path": ["技术参数", "压力"],
                },
            },
            {
                "id": "chk_not_searchable",
                "chunk_id": "chk_not_searchable",
                "content": "https://example.invalid",
                "type": "hyperlinks",
                "dense_vector": [],
                "sparse_vector": {},
                "provenance": {
                    "source_block_ids": ["block-link"],
                    "locations": [],
                    "heading_path": [],
                },
            },
        ],
    }


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted_filters: list[str] = []
        self.flushed: list[str] = []

    def delete(self, *, collection_name: str, filter: str) -> None:
        self.deleted_filters.append(filter)
        self.rows.clear()

    def insert(self, *, collection_name: str, data: list[dict]) -> dict:
        ids = list(range(901, 901 + len(data)))
        self.rows = [
            {**row, "chunk_id": vector_pk}
            for row, vector_pk in zip(data, ids, strict=True)
        ]
        return {"insert_count": len(data), "ids": ids}

    def flush(self, *, collection_name: str) -> None:
        self.flushed.append(collection_name)

    def query(self, **_: object) -> list[dict]:
        return [dict(row) for row in self.rows]


def test_build_local_evidence_preserves_every_source_block_and_location_truth() -> None:
    chunk = _state()["chunks"][0]
    evidence = index_service._build_local_evidence(
        chunk,
        doc_id="doc_demo",
        kb_id="kb_demo",
        tenant_id="tenant_default",
        document_version_id="dver_demo",
        index_version_id="idx_demo",
        file_title="HAK180 产品安全手册.pdf",
        position=0,
    )

    assert evidence.source.chunk_id == "chk_todo07_stable"
    assert evidence.source.source_block_ids == ["block-heading", "block-pressure"]
    assert [span.block_id for span in evidence.provenance_spans] == [
        "block-heading",
        "block-pressure",
    ]
    assert evidence.provenance_spans[0].availability == "unavailable"
    assert evidence.provenance_spans[0].reason_code == "parser_location_unavailable"
    assert evidence.provenance_spans[1].page_no == 12
    assert evidence.provenance_spans[1].bbox == (10.0, 20.0, 100.0, 80.0)
    assert evidence.provenance_spans[1].coordinate_space == "pdf_points"
    assert evidence.heading_path == ["技术参数", "压力"]
    assert evidence.asset_refs == [
        "assets/pressure.png",
        "imagetemps-doc_demo/pressure.png",
        "imagetemps-doc_demo/pressure-chart.png",
    ]
    assert evidence.content_hash.startswith("sha256:")
    assert evidence.evidence_id.startswith("ev_")


def test_legacy_bbox_without_coordinate_space_does_not_guess_units() -> None:
    span = index_service._evidence_location_span(
        "block-legacy",
        {"block_id": "block-legacy", "page_no": 3, "bbox": [1, 2, 3, 4]},
    )

    assert span.availability == "exact"
    assert span.page_no == 3
    assert span.bbox is None
    assert span.coordinate_space == "none"


def test_degenerate_bbox_degrades_to_page_locator_instead_of_failing_index() -> None:
    for bad_bbox in (
        [0.2, 0.3, 0.2, 0.8],
        ["0.2", "0.3", "0.7", "0.8"],
        [False, False, True, True],
    ):
        span = index_service._evidence_location_span(
            "block-degenerate",
            {
                "block_id": "block-degenerate",
                "page_no": 4,
                "bbox": bad_bbox,
                "coordinate_space": "normalized_page",
            },
        )

        assert span.availability == "exact"
        assert span.page_no == 4
        assert span.bbox is None
        assert span.coordinate_space == "none"


def test_fallback_source_chunk_id_is_deterministic_and_namespaced() -> None:
    chunk = {"id": "legacy-hash", "chunk_id": 17}
    first = index_service._stable_source_chunk_id(
        chunk,
        document_version_id="dver_demo",
        position=4,
        content="same",
    )
    second = index_service._stable_source_chunk_id(
        chunk,
        document_version_id="dver_demo",
        position=4,
        content="same",
    )

    assert first == second
    assert first.startswith("chk_")


def test_index_persists_evidence_after_vector_manifest_and_keeps_chunk_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMilvusClient()
    captured: list = []
    monkeypatch.setattr(index_service, "prepare_chunks_collection", lambda *_: None)
    monkeypatch.setattr(index_service.milvus_gateway, "client", lambda: fake)

    def _record_batch(values, *, tenant_id=None):
        evidence = list(values)
        captured.extend(evidence)
        assert tenant_id == "tenant_default"
        return [item.model_dump(mode="json") for item in evidence]

    monkeypatch.setattr(index_service, "record_evidence_batch", _record_batch)
    state = _state()
    result = index_service.index_chunks(state)

    assert len(captured) == 1
    evidence = captured[0]
    assert evidence.evidence_id == fake.rows[0]["evidence_id"]
    assert evidence.content_hash == fake.rows[0]["content_hash"]
    assert fake.rows[0]["source_chunk_id"] == "chk_todo07_stable"
    assert fake.rows[0]["location_status"] == "exact"
    assert fake.rows[0]["primary_page_no"] == 12

    searchable, skipped = result["chunks"]
    assert searchable["chunk_id"] == "chk_todo07_stable"
    assert searchable["source_chunk_id"] == "chk_todo07_stable"
    assert searchable["vector_pk"] == 901
    assert searchable["evidence_id"] == evidence.evidence_id
    assert skipped["chunk_id"] == "chk_not_searchable"
    assert skipped["vector_pk"] is None
    assert result["index_manifest"]["evidence_count"] == 1
    assert result["index_manifest"]["evidence_ids"] == [evidence.evidence_id]
    assert result["index_manifest"]["validation_status"] == "validated"


def test_evidence_batch_failure_fails_job_before_manifest_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMilvusClient()
    monkeypatch.setattr(index_service, "prepare_chunks_collection", lambda *_: None)
    monkeypatch.setattr(index_service.milvus_gateway, "client", lambda: fake)

    def _fail_batch(*_args, **_kwargs):
        raise RuntimeError("control database unavailable")

    monkeypatch.setattr(index_service, "record_evidence_batch", _fail_batch)
    state = _state()
    with pytest.raises(RuntimeError, match="control database unavailable"):
        index_service.index_chunks(state)

    assert fake.rows, "Milvus staging 写入可留待幂等重试清理"
    assert "index_manifest" not in state
    assert "vector_pk" not in state["chunks"][0]
    assert state["chunks"][0]["chunk_id"] == "chk_todo07_stable"


class _Schema:
    def __init__(self) -> None:
        self.fields: list[str] = []

    def add_field(self, *, field_name: str, **_: object) -> None:
        self.fields.append(field_name)


class _IndexParams:
    def add_index(self, **_: object) -> None:
        return None


class _CreateCollectionClient:
    def __init__(self) -> None:
        self.schema = _Schema()
        self.created = False

    def has_collection(self, _name: str) -> bool:
        return False

    def create_schema(self, **_: object) -> _Schema:
        return self.schema

    def prepare_index_params(self) -> _IndexParams:
        return _IndexParams()

    def create_collection(self, **_: object) -> None:
        self.created = True


def test_new_collection_has_static_evidence_compact_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _CreateCollectionClient()
    collection = "todo08_new_schema"
    index_service._verified_collections.discard(collection)
    monkeypatch.setattr(index_service.milvus_gateway, "client", lambda: fake)

    index_service.prepare_chunks_collection(collection, vector_dim=3)

    assert fake.created
    assert {
        "evidence_id",
        "content_hash",
        "source_chunk_id",
        "location_status",
        "primary_page_no",
    } <= set(fake.schema.fields)


class _LegacyCollectionClient:
    def has_collection(self, _name: str) -> bool:
        return True

    def describe_collection(self, _name: str) -> dict:
        core_fields = {
            "chunk_id", "doc_id", "kb_id", "docnm_kwd", "content_with_weight",
            "dense_vector", "sparse_vector", "type", "img_id", "id",
            "create_time", "create_timestamp_flt", "available_int", "position_int",
            "important_kwd", "important_tks", "question_kwd", "question_tks",
            "meta_fields", "tag_kwd",
        }
        return {"fields": [{"name": field} for field in core_fields]}

    def create_schema(self, **_: object):  # pragma: no cover - must not be called
        raise AssertionError("legacy collection must not be recreated")


def test_legacy_dynamic_collection_is_accepted_without_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = "todo08_legacy_schema"
    index_service._verified_collections.discard(collection)
    monkeypatch.setattr(
        index_service.milvus_gateway,
        "client",
        lambda: _LegacyCollectionClient(),
    )

    index_service.prepare_chunks_collection(collection, vector_dim=3)

    assert collection in index_service._verified_collections
