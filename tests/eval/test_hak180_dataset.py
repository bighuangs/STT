"""HAK180 黄金集的可复现性与源文件完整性测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.domain.evaluation import content_digest
from app.evaluation.dataset_loader import BUILTIN_DATASET_PATH, load_dataset, verify_source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_COUNT = 28
EXPECTED_SOURCE_SHA256 = (
    "dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765"
)
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "7ea4edf3b981b8497191bcdd8e7e5ee0b871bad222ec29e27915a9ba662ca172"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "2294cc2a1f01a5bec991ca5bfdb16fba396b99ad2bffb0757139f91b9bc6fe9a"
)


def test_hak180_manifest_is_a_fixed_28_case_snapshot() -> None:
    manifest = load_dataset()
    cases = manifest["cases"]

    assert manifest["dataset_key"] == "hak180-safety"
    assert manifest["version"] == "1.0.0"
    assert manifest["schema_version"] == "stt.eval.v1"
    assert len(cases) == EXPECTED_CASE_COUNT
    assert [case["case_id"] for case in cases] == [
        f"HAK180-{ordinal:03d}" for ordinal in range(1, EXPECTED_CASE_COUNT + 1)
    ]
    assert len({case["case_id"] for case in cases}) == EXPECTED_CASE_COUNT
    assert sum(bool(case.get("unanswerable")) for case in cases) == 2
    assert all(case.get("query", "").strip() for case in cases)
    assert all(
        case.get("unanswerable") or case.get("reference_answer", "").strip()
        for case in cases
    )


def test_hak180_manifest_hashes_pin_semantics_and_file_bytes() -> None:
    manifest = load_dataset()

    # canonical JSON 锁定清单语义；文件字节哈希还能发现空白/编码等意外改写。
    assert content_digest(manifest) == EXPECTED_MANIFEST_CONTENT_SHA256
    assert hashlib.sha256(BUILTIN_DATASET_PATH.read_bytes()).hexdigest() == (
        EXPECTED_MANIFEST_FILE_SHA256
    )


def test_hak180_pdf_source_matches_the_reviewed_snapshot() -> None:
    manifest = load_dataset()
    source = manifest["source"]

    assert source["logical_name"] == "hak180产品安全手册.pdf"
    assert source["relative_path"] == "doc/hak180产品安全手册.pdf"
    assert source["content_sha256"] == EXPECTED_SOURCE_SHA256
    assert source["page_count"] == 1
    assert source["reviewed_from_render"] is True

    result = verify_source(manifest, PROJECT_ROOT)
    assert result == {
        "verified": True,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "actual_sha256": EXPECTED_SOURCE_SHA256,
        "path": str(PROJECT_ROOT / "doc" / "hak180产品安全手册.pdf"),
        "reason": None,
    }


def test_source_verification_fails_closed_for_a_changed_pdf(tmp_path: Path) -> None:
    manifest = load_dataset()
    changed = tmp_path / "doc" / "hak180产品安全手册.pdf"
    changed.parent.mkdir(parents=True)
    changed.write_bytes(b"not-the-reviewed-pdf")

    result = verify_source(manifest, tmp_path)

    assert result["verified"] is False
    assert result["expected_sha256"] == EXPECTED_SOURCE_SHA256
    assert result["actual_sha256"] != EXPECTED_SOURCE_SHA256
    assert result["reason"] == "source sha256 mismatch"


def test_source_verification_rejects_escape_and_project_files_outside_doc(
    tmp_path: Path,
) -> None:
    secret = tmp_path / ".env"
    secret.write_text("SECRET=must-not-be-hashed", encoding="utf-8")
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"outside")

    project_file = {
        "source": {
            "relative_path": ".env",
            "content_sha256": hashlib.sha256(secret.read_bytes()).hexdigest(),
        }
    }
    escaped = {
        "source": {
            "relative_path": "../outside.pdf",
            "content_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    }

    project_result = verify_source(project_file, tmp_path)
    escaped_result = verify_source(escaped, tmp_path)
    assert project_result == {
        "verified": False,
        "reason": "dataset source path must stay inside the project doc directory",
    }
    assert escaped_result == project_result


def test_source_verification_allows_doc_subdirectories(tmp_path: Path) -> None:
    source = tmp_path / "doc" / "manuals" / "safe.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes((b"streamed-source-block" * 100_000) + b"tail")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    result = verify_source(
        {
            "source": {
                "relative_path": "doc/manuals/safe.pdf",
                "content_sha256": digest,
            }
        },
        tmp_path,
    )

    assert result["verified"] is True
    assert result["actual_sha256"] == digest
