from __future__ import annotations

from pathlib import Path

from app.application import evaluation_service


def _source_tree(root: Path, *, reverse_creation_order: bool = False) -> None:
    files = [
        ("app/query.py", b"QUERY = 'stable'\n"),
        ("app/page.html", b"<main>stable</main>\n"),
        ("app/policy.prompt", "可靠回答\n".encode()),
        ("alembic/versions/0001.py", b"revision = '0001'\n"),
        ("alembic/script.py.mako", b"${up_revision}\n"),
        ("pyproject.toml", b"[project]\nname='stt'\n"),
        ("alembic.ini", b"[alembic]\n"),
    ]
    for relative, content in reversed(files) if reverse_creation_order else files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_source_manifest_digest_changes_when_controlled_source_bytes_change(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    first = evaluation_service._source_manifest_digest(tmp_path)

    (tmp_path / "app/query.py").write_bytes(b"QUERY = 'changed'\n")
    second = evaluation_service._source_manifest_digest(tmp_path)

    assert first != second
    assert len(first) == len(second) == 64


def test_source_manifest_digest_is_independent_of_creation_order(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _source_tree(first_root)
    _source_tree(second_root, reverse_creation_order=True)

    assert evaluation_service._source_manifest_digest(
        first_root
    ) == evaluation_service._source_manifest_digest(second_root)


def test_code_revision_prefers_explicit_build_revision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(evaluation_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("STT_CODE_REVISION", "image-sha256:trusted-build")

    assert evaluation_service._code_revision() == "image-sha256:trusted-build"


def test_code_revision_contains_head_and_source_digest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    git_dir = tmp_path / ".git"
    (git_dir / "refs/heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    revision = "0123456789abcdef0123456789abcdef01234567"
    (git_dir / "refs/heads/main").write_text(revision + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("STT_CODE_REVISION", raising=False)

    value = evaluation_service._code_revision()

    assert value == (
        f"head:{revision};"
        f"source-sha256:{evaluation_service._source_manifest_digest(tmp_path)}"
    )

