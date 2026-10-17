from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from app.domain.multimodal import MediaLocation


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "app" / "evaluation" / "datasets" / "hak180_multimodal_v1.json"
PDF_PATH = ROOT / "doc" / "hak180产品安全手册.pdf"


def _dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_hak180_multimodal_dataset_is_bound_to_the_real_manual() -> None:
    dataset = _dataset()

    assert dataset["schema_version"] == "stt.multimodal.eval.v1"
    assert dataset["source"]["relative_path"] == "doc/hak180产品安全手册.pdf"
    assert sha256(PDF_PATH.read_bytes()).hexdigest() == dataset["source"]["content_sha256"]
    assert dataset["source"]["page_count"] == 1
    assert dataset["source"]["reviewed_from_render"] is True


def test_hak180_multimodal_regions_follow_the_runtime_coordinate_contract() -> None:
    dataset = _dataset()
    seen_region_ids: set[str] = set()

    for case in dataset["cases"]:
        for region in case["relevant_regions"]:
            location = MediaLocation(page_no=region["page"], bbox=region["bbox"])
            assert location.page_no == 1
            assert location.bbox is not None
            assert region["region_id"] not in seen_region_ids
            seen_region_ids.add(region["region_id"])

    assert seen_region_ids == {
        "p1-safe-lifting-diagrams",
        "p1-hazardous-substances-table",
        "p1-hot-surface-warnings",
    }


def test_hak180_multimodal_dataset_has_isolation_and_quality_gates() -> None:
    dataset = _dataset()
    cases = {case["case_id"]: case for case in dataset["cases"]}

    assert dataset["gate_policy"]["external_vision_allowed"] is False
    assert dataset["gate_policy"]["query_attachment_must_not_become_kb_evidence"] is True
    assert cases["HAK180-MM-004"]["unanswerable"] is True
    assert cases["HAK180-MM-004"]["relevant_regions"] == []
    assert all(any(tag == "critical" for tag in case["tags"]) for case in dataset["cases"])
