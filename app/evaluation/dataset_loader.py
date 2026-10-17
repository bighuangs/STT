"""Versioned dataset loading and source-integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.domain.evaluation import validate_dataset_manifest


BUILTIN_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "hak180_safety_v1.json"


def load_dataset(path: str | Path = BUILTIN_DATASET_PATH) -> dict:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    validate_dataset_manifest(manifest)
    return manifest


def verify_source(manifest: dict, project_root: str | Path) -> dict:
    source = dict(manifest.get("source") or {})
    relative = source.get("relative_path")
    expected = str(source.get("content_sha256") or "").lower()
    if not relative or not expected:
        return {"verified": False, "reason": "dataset source path or sha256 is missing"}
    root = Path(project_root).resolve()
    document_root = (root / "doc").resolve()
    path = (root / relative).resolve()
    if path != document_root and document_root not in path.parents:
        return {
            "verified": False,
            "reason": "dataset source path must stay inside the project doc directory",
        }
    if not path.is_file():
        return {"verified": False, "reason": f"source file not found: {relative}"}
    hasher = hashlib.sha256()
    with path.open("rb") as source_file:
        while block := source_file.read(1024 * 1024):
            hasher.update(block)
    digest = hasher.hexdigest()
    return {
        "verified": digest == expected,
        "expected_sha256": expected,
        "actual_sha256": digest,
        "path": str(path),
        "reason": None if digest == expected else "source sha256 mismatch",
    }
