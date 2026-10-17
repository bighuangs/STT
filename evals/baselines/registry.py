"""只允许读取注册且哈希匹配的评测基线。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("registry.json")


@dataclass(frozen=True, slots=True)
class BaselineAsset:
    """经过审批状态、路径与哈希校验的基线。"""

    baseline_id: str
    path: Path
    sha256: str
    payload: dict[str, Any]


def load_baseline_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("evaluation baseline registry must be an object")
    if registry.get("schema_version") != "stt.eval.baseline-registry.v1":
        raise ValueError("unsupported evaluation baseline registry schema")
    if not isinstance(registry.get("baselines"), dict):
        raise ValueError("evaluation baseline registry must contain baselines")
    return registry


def load_baseline_asset(
    baseline_id: str,
    *,
    project_root: str | Path,
    registry_path: str | Path = REGISTRY_PATH,
) -> BaselineAsset:
    """读取已审批基线；候选或未注册输出不能被当作门禁基线。"""

    registry = load_baseline_registry(registry_path)
    raw = registry["baselines"].get(baseline_id)
    if not isinstance(raw, dict):
        raise KeyError(f"unknown evaluation baseline: {baseline_id}")
    if raw.get("status") != "approved":
        raise ValueError(f"evaluation baseline is not approved: {baseline_id}")
    root = Path(project_root).resolve()
    relative_path = raw.get("path")
    expected = str(raw.get("sha256") or "").lower()
    if not isinstance(relative_path, str) or not relative_path or len(expected) != 64:
        raise ValueError(f"invalid evaluation baseline registry entry: {baseline_id}")
    source = (root / relative_path).resolve()
    if source != root and root not in source.parents:
        raise ValueError("evaluation baseline path must stay inside project root")
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"evaluation baseline sha256 mismatch for {baseline_id}: {actual}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation baseline payload must be an object")
    return BaselineAsset(baseline_id, source, actual, payload)
