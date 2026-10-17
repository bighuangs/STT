"""通过内容哈希解析冻结数据集，迁移期间不复制大体量资产。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("registry.json")


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """已完成路径与哈希验证的数据集资产。"""

    dataset_id: str
    path: Path
    sha256: str
    status: str
    manifest: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def load_dataset_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = _load_json_object(Path(path))
    if registry.get("schema_version") != "stt.eval.dataset-registry.v1":
        raise ValueError("unsupported evaluation dataset registry schema")
    if not isinstance(registry.get("datasets"), dict):
        raise ValueError("evaluation dataset registry must contain datasets")
    return registry


def load_dataset_asset(
    dataset_id: str,
    *,
    project_root: str | Path,
    registry_path: str | Path = REGISTRY_PATH,
) -> DatasetAsset:
    """解析并校验单个数据集；路径逃逸与字节漂移均失败关闭。"""

    registry = load_dataset_registry(registry_path)
    raw = registry["datasets"].get(dataset_id)
    if not isinstance(raw, dict):
        raise KeyError(f"unknown evaluation dataset: {dataset_id}")
    root = Path(project_root).resolve()
    relative_path = raw.get("path")
    expected = str(raw.get("sha256") or "").lower()
    if not isinstance(relative_path, str) or not relative_path or len(expected) != 64:
        raise ValueError(f"invalid evaluation dataset registry entry: {dataset_id}")
    source = (root / relative_path).resolve()
    if not _inside(root, source):
        raise ValueError("evaluation dataset path must stay inside project root")
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = _sha256(source)
    if actual != expected:
        raise ValueError(f"evaluation dataset sha256 mismatch for {dataset_id}: {actual}")
    return DatasetAsset(
        dataset_id=dataset_id,
        path=source,
        sha256=actual,
        status=str(raw.get("status") or "unknown"),
        manifest=_load_json_object(source),
    )
