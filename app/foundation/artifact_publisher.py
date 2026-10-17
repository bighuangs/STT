"""以纯标准库在同一文件系统内原子发布一组文本制品。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PUBLICATION_SCHEMA = "stt.eval.artifact-publication.v1"


class ArtifactPublicationError(RuntimeError):
    """制品集合无法安全发布。"""


def _content_bytes(content: str) -> bytes:
    return content.encode("utf-8")


def _publication_id(artifacts: Mapping[str, tuple[str, str]]) -> str:
    manifest = [
        {
            "name": name,
            "filename": filename,
            "sha256": hashlib.sha256(_content_bytes(content)).hexdigest(),
            "size_bytes": len(_content_bytes(content)),
        }
        for name, (filename, content) in sorted(artifacts.items())
    ]
    canonical = json.dumps(
        {"schema": _PUBLICATION_SCHEMA, "artifacts": manifest},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:32]


def _validate_inputs(run_id: str, artifacts: Mapping[str, tuple[str, str]]) -> None:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ArtifactPublicationError("evaluation run id is not a safe path segment")
    if not artifacts:
        raise ArtifactPublicationError("evaluation artifact set is empty")
    filenames: set[str] = set()
    for name, value in artifacts.items():
        if not _SAFE_ARTIFACT_NAME.fullmatch(name):
            raise ArtifactPublicationError("evaluation artifact name is invalid")
        if not isinstance(value, tuple) or len(value) != 2:
            raise ArtifactPublicationError("evaluation artifact entry is invalid")
        filename, content = value
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ArtifactPublicationError("evaluation artifact filename is invalid")
        if filename in filenames:
            raise ArtifactPublicationError("evaluation artifact filename is duplicated")
        if not isinstance(content, str):
            raise ArtifactPublicationError("evaluation artifact content must be text")
        filenames.add(filename)


def _write_staged_file(path: Path, payload: bytes) -> None:
    """以二进制模式耐久写入，避免宿主平台替换换行符。"""

    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_existing_generation(
    generation: Path,
    artifacts: Mapping[str, tuple[str, str]],
) -> None:
    expected_names = {filename for filename, _content in artifacts.values()}
    actual_names = {item.name for item in generation.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ArtifactPublicationError("existing artifact generation has an invalid file set")
    for filename, content in artifacts.values():
        path = generation / filename
        if not path.is_file() or path.read_bytes() != _content_bytes(content):
            raise ArtifactPublicationError(
                "existing artifact generation failed integrity validation"
            )


def _publish_staging_directory(staging: Path, generation: Path) -> None:
    """同盘目录 rename 是制品集合的唯一可见性切换点。"""

    try:
        os.replace(staging, generation)
    except OSError:
        # 并发的同内容发布可能已先完成；只接受逐字节相同的既有 generation。
        if not generation.is_dir():
            raise


def publish_text_artifacts(
    *,
    project_root: Path,
    report_root: Path,
    run_id: str,
    artifacts: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, object]]:
    """完整写入并原子发布一代制品，返回带哈希的相对路径。"""

    _validate_inputs(run_id, artifacts)
    project = project_root.resolve()
    root = report_root.resolve()
    if root != project and project not in root.parents:
        raise ArtifactPublicationError("evaluation report root escaped project root")
    root.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".tmp", dir=root)).resolve()
    try:
        for _name, (filename, content) in sorted(artifacts.items()):
            _write_staged_file(staging / filename, _content_bytes(content))

        generation = (root / run_id / f"generation-{_publication_id(artifacts)}").resolve()
        run_root = (root / run_id).resolve()
        if generation.parent != run_root or run_root.parent != root:
            raise ArtifactPublicationError("evaluation artifact generation escaped report root")
        run_root.mkdir(parents=True, exist_ok=True)
        if generation.exists():
            _validate_existing_generation(generation, artifacts)
        else:
            _publish_staging_directory(staging, generation)
            _validate_existing_generation(generation, artifacts)

        paths: dict[str, dict[str, object]] = {}
        for name, (filename, _content) in artifacts.items():
            path = generation / filename
            stored = path.read_bytes()
            paths[name] = {
                "path": path.relative_to(project).as_posix(),
                "sha256": hashlib.sha256(stored).hexdigest(),
                "size_bytes": len(stored),
                "publication_id": generation.name.removeprefix("generation-"),
            }
        return paths
    except Exception as exc:
        if isinstance(exc, ArtifactPublicationError):
            raise
        raise ArtifactPublicationError("evaluation artifact publication failed") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def discard_artifact_publication(
    *,
    project_root: Path,
    report_root: Path,
    run_id: str,
    paths: Mapping[str, Mapping[str, object]],
) -> None:
    """数据库终态提交失败时，仅删除本次已验证的完整 generation。"""

    if not _SAFE_RUN_ID.fullmatch(run_id) or not paths:
        raise ArtifactPublicationError("artifact rollback contract is invalid")
    publication_ids = {str(item.get("publication_id") or "") for item in paths.values()}
    if len(publication_ids) != 1:
        raise ArtifactPublicationError("artifact rollback publication id is ambiguous")
    publication_id = publication_ids.pop()
    if not re.fullmatch(r"[0-9a-f]{32}", publication_id):
        raise ArtifactPublicationError("artifact rollback publication id is invalid")

    project = project_root.resolve()
    root = report_root.resolve()
    run_root = (root / run_id).resolve()
    generation = (run_root / f"generation-{publication_id}").resolve()
    if root != project and project not in root.parents:
        raise ArtifactPublicationError("artifact rollback report root escaped project root")
    if run_root.parent != root or generation.parent != run_root:
        raise ArtifactPublicationError("artifact rollback path escaped report root")
    for item in paths.values():
        stored = (project / str(item.get("path") or "")).resolve()
        if stored.parent != generation:
            raise ArtifactPublicationError("artifact rollback record escaped publication")

    if generation.is_dir():
        shutil.rmtree(generation)
    if run_root.is_dir() and not any(run_root.iterdir()):
        run_root.rmdir()


__all__ = [
    "ArtifactPublicationError",
    "discard_artifact_publication",
    "publish_text_artifacts",
]
