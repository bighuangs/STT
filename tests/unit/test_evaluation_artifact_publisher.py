from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.foundation import artifact_publisher


def _artifacts() -> dict[str, tuple[str, str]]:
    return {
        "json": ("report.json", '{"status":"ok"}\n'),
        "markdown": ("report.md", "# 结果\n\n通过\n"),
        "results": ("results.jsonl", '{"case":"a"}\n'),
    }


def test_publication_is_complete_hashed_and_idempotent(tmp_path: Path) -> None:
    report_root = tmp_path / "var" / "evals"

    first = artifact_publisher.publish_text_artifacts(
        project_root=tmp_path,
        report_root=report_root,
        run_id="erun_atomic",
        artifacts=_artifacts(),
    )
    second = artifact_publisher.publish_text_artifacts(
        project_root=tmp_path,
        report_root=report_root,
        run_id="erun_atomic",
        artifacts=_artifacts(),
    )

    assert second == first
    generations = list((report_root / "erun_atomic").glob("generation-*"))
    assert len(generations) == 1
    assert not list(report_root.glob(".*.tmp"))
    assert {item.name for item in generations[0].iterdir()} == {
        "report.json",
        "report.md",
        "results.jsonl",
    }
    for record in first.values():
        stored = tmp_path / str(record["path"])
        payload = stored.read_bytes()
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["size_bytes"] == len(payload)
        assert record["publication_id"]


def test_failed_staging_write_leaves_no_partial_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_root = tmp_path / "var" / "evals"
    calls = 0
    original = artifact_publisher._write_staged_file

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        original(path, payload)

    monkeypatch.setattr(artifact_publisher, "_write_staged_file", fail_second)

    with pytest.raises(
        artifact_publisher.ArtifactPublicationError,
        match="publication failed",
    ):
        artifact_publisher.publish_text_artifacts(
            project_root=tmp_path,
            report_root=report_root,
            run_id="erun_failed",
            artifacts=_artifacts(),
        )

    assert not (report_root / "erun_failed").exists()
    assert not list(report_root.glob(".*.tmp"))


def test_database_failure_can_discard_only_the_published_generation(tmp_path: Path) -> None:
    report_root = tmp_path / "var" / "evals"
    paths = artifact_publisher.publish_text_artifacts(
        project_root=tmp_path,
        report_root=report_root,
        run_id="erun_rollback",
        artifacts=_artifacts(),
    )

    artifact_publisher.discard_artifact_publication(
        project_root=tmp_path,
        report_root=report_root,
        run_id="erun_rollback",
        paths=paths,
    )

    assert not (report_root / "erun_rollback").exists()


def test_rollback_rejects_a_path_from_another_publication(tmp_path: Path) -> None:
    report_root = tmp_path / "var" / "evals"
    paths = artifact_publisher.publish_text_artifacts(
        project_root=tmp_path,
        report_root=report_root,
        run_id="erun_bound",
        artifacts=_artifacts(),
    )
    tampered = {name: dict(value) for name, value in paths.items()}
    tampered["json"]["path"] = "var/evals/another/report.json"

    with pytest.raises(artifact_publisher.ArtifactPublicationError, match="escaped"):
        artifact_publisher.discard_artifact_publication(
            project_root=tmp_path,
            report_root=report_root,
            run_id="erun_bound",
            paths=tampered,
        )

    assert (tmp_path / str(paths["json"]["path"])).is_file()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", ""])
def test_run_id_must_be_one_safe_path_segment(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(artifact_publisher.ArtifactPublicationError, match="run id"):
        artifact_publisher.publish_text_artifacts(
            project_root=tmp_path,
            report_root=tmp_path / "var" / "evals",
            run_id=run_id,
            artifacts=_artifacts(),
        )
