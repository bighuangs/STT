from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from app.application.multimodal_service import (
    MediaPolicyError,
    MediaValidationError,
    MultimodalService,
    rank_multimodal_candidates,
    sniff_image,
)
from app.domain.multimodal import (
    MediaAsset,
    MediaIngestPolicy,
    MediaLocation,
    MultimodalCandidate,
    MultimodalEvaluationReport,
    MultimodalRetrievalProfile,
)


SNAPSHOT_HASH = "sha256:" + "a" * 64


def _png(width: int = 20, height: int = 10) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), "white").save(stream, format="PNG")
    return stream.getvalue()


def _document_asset(service: MultimodalService | None = None):
    gateway = service or MultimodalService()
    asset = gateway.ingest_image(
        _png(),
        claimed_media_type="image/png",
        tenant_id="tenant-a",
        source_kind="document_asset",
        classification="internal",
        kb_id="kb-a",
        document_id="doc-a",
        document_version_id="dv-a",
        page_no=1,
        bbox=(0.1, 0.2, 0.8, 0.9),
    )
    return gateway, asset


def _derivation(service: MultimodalService, asset, **overrides):
    payload = {
        "kind": "vision_description",
        "text": "图示要求双手托住设备两侧搬运。",
        "provider_id": "local",
        "model_id": "vision-fixture-v1",
        "prompt_version": "vision-safety.v1",
        "model_snapshot_hash": SNAPSHOT_HASH,
    }
    payload.update(overrides)
    return service.record_derivation(asset, **payload)


def test_sniff_image_uses_magic_and_decoder() -> None:
    media_type, width, height = sniff_image(
        _png(),
        claimed_media_type="image/png",
        policy=MediaIngestPolicy(),
    )
    assert (media_type, width, height) == ("image/png", 20, 10)


def test_sniff_image_rejects_mime_mismatch_and_corrupt_bytes() -> None:
    with pytest.raises(MediaValidationError, match="claimed MIME"):
        sniff_image(
            _png(),
            claimed_media_type="image/jpeg",
            policy=MediaIngestPolicy(),
        )
    with pytest.raises(MediaValidationError, match="corrupt or unsafe"):
        sniff_image(
            b"\x89PNG\r\n\x1a\nnot-a-real-image",
            claimed_media_type="image/png",
            policy=MediaIngestPolicy(),
        )
    corrupt_crc = bytearray(_png())
    idat = corrupt_crc.index(b"IDAT")
    idat_length = int.from_bytes(corrupt_crc[idat - 4 : idat], "big")
    idat_crc = idat + 4 + idat_length
    corrupt_crc[idat_crc] ^= 0xFF
    with pytest.raises(MediaValidationError, match="corrupt or unsafe"):
        sniff_image(
            bytes(corrupt_crc),
            claimed_media_type="image/png",
            policy=MediaIngestPolicy(),
        )


def test_sniff_image_enforces_byte_and_pixel_limits() -> None:
    with pytest.raises(MediaValidationError, match="max_image_bytes"):
        sniff_image(
            _png(),
            claimed_media_type="image/png",
            policy=MediaIngestPolicy(max_image_bytes=8),
        )
    with pytest.raises(MediaValidationError, match="max_image_pixels"):
        sniff_image(
            _png(),
            claimed_media_type="image/png",
            policy=MediaIngestPolicy(max_image_pixels=100),
        )


def test_document_asset_has_stable_identity_and_location() -> None:
    service, first = _document_asset()
    _, second = _document_asset(service)
    assert first.asset_id == second.asset_id
    assert first.location.page_no == 1
    assert first.location.bbox == (0.1, 0.2, 0.8, 0.9)
    assert first.content_hash.startswith("sha256:")
    assert service.object_store.get(first.object_key) == _png()


def test_document_image_requires_version_scope_and_bbox() -> None:
    with pytest.raises(ValidationError, match="document asset requires"):
        MediaAsset(
            tenant_id="tenant-a",
            source_kind="document_asset",
            modality="image",
            media_type="image/png",
            content_hash="sha256:" + "b" * 64,
            size_bytes=12,
            width=1,
            height=1,
            object_key="safe/key.png",
            location=MediaLocation(page_no=1, bbox=(0.0, 0.0, 1.0, 1.0)),
        )
    with pytest.raises(ValidationError, match="exact normalized page region"):
        MediaAsset(
            tenant_id="tenant-a",
            kb_id="kb-a",
            document_id="doc-a",
            document_version_id="dv-a",
            source_kind="document_asset",
            modality="image",
            media_type="image/png",
            content_hash="sha256:" + "b" * 64,
            size_bytes=12,
            width=1,
            height=1,
            object_key="safe/key.png",
        )


def test_external_derivation_is_default_denied() -> None:
    service, asset = _document_asset()
    with pytest.raises(MediaPolicyError, match="forbidden"):
        _derivation(service, asset, external_processing=True)


def test_external_derivation_requires_explicit_public_policy() -> None:
    service = MultimodalService(
        policy=MediaIngestPolicy(
            external_processing_allowed=True,
            allowed_external_classifications=("public",),
        )
    )
    public_service, asset = _document_asset(service)
    public_asset = asset.model_copy(update={"classification": "public"})
    derivation = _derivation(
        public_service,
        public_asset,
        external_processing=True,
    )
    assert derivation.external_processing is True


def test_derivation_text_and_snapshot_are_content_addressed() -> None:
    service, asset = _document_asset()
    first = _derivation(service, asset)
    second = _derivation(service, asset)
    changed = _derivation(service, asset, prompt_version="vision-safety.v2")
    assert first.derivation_id == second.derivation_id
    assert first.derivation_id != changed.derivation_id
    assert first.content_hash.startswith("sha256:")


def test_document_media_becomes_traceable_evidence() -> None:
    service, asset = _document_asset()
    derivation = _derivation(service, asset)
    evidence = service.to_evidence(
        asset,
        derivation,
        index_version_id="idx-a",
        chunk_id="chunk-image-1",
        source_block_id="block-image-1",
        source_label="HAK180 第 1 页搬运图示",
    )
    assert evidence.modality == "image"
    assert evidence.asset is not None
    assert evidence.asset.asset_id == asset.asset_id
    assert evidence.derivation is not None
    assert evidence.derivation.output_hash == evidence.content_hash
    assert evidence.primary_location_span().bbox == (0.1, 0.2, 0.8, 0.9)


def test_query_attachment_is_scoped_and_not_kb_evidence() -> None:
    service = MultimodalService()
    asset = service.ingest_image(
        _png(),
        claimed_media_type="image/png",
        tenant_id="tenant-a",
        source_kind="query_attachment",
        owner_user_id="user-a",
        classification="internal",
    )
    derivation = _derivation(service, asset)
    context = service.query_attachment_context(
        asset,
        derivation,
        tenant_id="tenant-a",
        user_id="user-a",
    )
    assert context["asset_id"] == asset.asset_id
    with pytest.raises(MediaPolicyError, match="scope mismatch"):
        service.query_attachment_context(
            asset,
            derivation,
            tenant_id="tenant-a",
            user_id="user-b",
        )
    with pytest.raises(MediaValidationError, match="published document"):
        service.to_evidence(
            asset,
            derivation,
            index_version_id="idx-a",
            chunk_id="chunk-a",
            source_block_id="block-a",
            source_label="attachment",
        )


def test_multimodal_route_is_default_disabled() -> None:
    candidates = [
        MultimodalCandidate(
            evidence_id="ev_" + "1" * 40,
            asset_id="asset_" + "1" * 32,
            modality="image",
            text_score=0.7,
            modality_score=0.9,
        )
    ]
    selected, diagnostics = rank_multimodal_candidates(candidates, MultimodalRetrievalProfile())
    assert selected == []
    assert diagnostics["reason"] == "profile_disabled"


def test_multimodal_ranking_is_deterministic_and_applies_asset_quota() -> None:
    candidates = [
        MultimodalCandidate(
            evidence_id="ev_" + str(index) * 40,
            asset_id="asset_" + ("a" if index < 3 else "b") * 32,
            modality="image",
            text_score=float(index),
            modality_score=float(5 - index),
        )
        for index in range(1, 5)
    ]
    profile = MultimodalRetrievalProfile(
        profile_id="mm-hak180",
        version=2,
        enabled=True,
        status="published",
        route_weights={"text": 1.0, "image": 1.0},
        max_candidates=4,
        max_per_asset=1,
    )
    selected, diagnostics = rank_multimodal_candidates(candidates, profile)
    assert len(selected) == 2
    assert [item.rank for item in selected] == [1, 2]
    assert len({item.asset_id for item in selected}) == 2
    assert diagnostics["dropped_by_asset_quota"] == 2


def test_enabled_profile_must_be_published() -> None:
    with pytest.raises(ValidationError, match="must be published"):
        MultimodalRetrievalProfile(enabled=True, status="draft")


def test_evaluation_gate_requires_gain_and_location_quality() -> None:
    passed = MultimodalEvaluationReport(
        dataset_id="hak180-mm-v1",
        baseline_recall=0.5,
        candidate_recall=0.75,
        baseline_citation_location_accuracy=0.5,
        candidate_citation_location_accuracy=1.0,
        latency_delta_ms=42,
        cost_delta=0.01,
        minimum_gain=0.1,
    )
    failed = MultimodalEvaluationReport(
        dataset_id="hak180-mm-v1",
        baseline_recall=0.5,
        candidate_recall=0.55,
        baseline_citation_location_accuracy=1.0,
        candidate_citation_location_accuracy=0.9,
        latency_delta_ms=1,
        cost_delta=0,
        minimum_gain=0.1,
    )
    assert passed.passed is True
    assert failed.passed is False
