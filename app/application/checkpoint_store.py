"""本地检查点制品存储。

状态由 Worker 生成，以 gzip+pickle 保存；加载前强制校验根目录与 SHA-256。
数据库只保存 URI、哈希及输入/配置指纹，避免大对象进入控制库。
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import re
from pathlib import Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class CheckpointIntegrityError(RuntimeError):
    pass


class LocalCheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, job_id: str, sequence: int, stage: str, state: dict) -> dict:
        if not _SAFE_ID.fullmatch(job_id) or not _SAFE_ID.fullmatch(stage):
            raise ValueError("unsafe checkpoint job id or stage")
        directory = (self.root / job_id).resolve()
        if self.root not in directory.parents:
            raise ValueError("checkpoint path escaped configured root")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{sequence:06d}-{stage}.pkl.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(temporary, "wb", compresslevel=5) as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        raw = target.read_bytes()
        return {
            "artifact_uri": str(target),
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_size": len(raw),
        }

    def load(self, checkpoint: dict) -> dict:
        target = Path(checkpoint["artifact_uri"]).expanduser().resolve()
        if target != self.root and self.root not in target.parents:
            raise CheckpointIntegrityError("checkpoint artifact is outside configured root")
        raw = target.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != checkpoint["artifact_sha256"]:
            raise CheckpointIntegrityError("checkpoint artifact SHA-256 mismatch")
        with gzip.open(target, "rb") as stream:
            state = pickle.load(stream)  # noqa: S301 - internal file + root/SHA validation
        if not isinstance(state, dict):
            raise CheckpointIntegrityError("checkpoint payload is not a graph state")
        return state
