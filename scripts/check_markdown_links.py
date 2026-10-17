"""Validate repository-local links in Markdown without network access."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(?P<target><[^>]+>|\S+)", re.MULTILINE)
FENCED_CODE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
SKIPPED_SCHEMES = {"http", "https", "mailto", "tel", "data"}


def markdown_files(paths: list[str] | None = None) -> list[Path]:
    """Return explicit Markdown files or every Git-visible Markdown file."""
    if paths:
        result: list[Path] = []
        for raw in paths:
            candidate = (
                (ROOT / raw).resolve()
                if not Path(raw).is_absolute()
                else Path(raw).resolve()
            )
            if candidate.is_dir():
                result.extend(candidate.rglob("*.md"))
            elif candidate.suffix.lower() == ".md" and candidate.is_file():
                result.append(candidate)
        return sorted(set(result))

    try:
        output = subprocess.check_output(
            [
                "git",
                "-c",
                "core.quotepath=false",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "*.md",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        )
        return sorted(
            ROOT / relative
            for relative in output.splitlines()
            if relative and "test/test_md_images/" not in relative.replace("\\", "/")
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in ROOT.rglob("*.md")
            if not any(
                part in {".git", ".venv", ".test-artifacts"} for part in path.parts
            )
        )


def _targets(text: str) -> list[tuple[int, str]]:
    def mask_code(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    text = FENCED_CODE.sub(mask_code, text)
    text = INLINE_CODE.sub(mask_code, text)
    found: list[tuple[int, str]] = []
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(text):
            target = match.group("target").strip().strip("<>")
            found.append((text.count("\n", 0, match.start()) + 1, target))
    return found


def _local_target(source: Path, raw_target: str) -> Path | None:
    if (
        not raw_target
        or raw_target.startswith("#")
        or "${" in raw_target
        or "{{" in raw_target
    ):
        return None
    parsed = urlsplit(raw_target)
    if parsed.scheme.lower() in SKIPPED_SCHEMES:
        return None
    if parsed.scheme or parsed.netloc:
        # A Windows drive-letter link is absolute and non-portable in repository docs.
        return Path(unquote(parsed.path or raw_target))
    path_text = unquote(parsed.path)
    return (source.parent / path_text).resolve() if path_text else None


def check(paths: list[str] | None = None) -> list[str]:
    failures: list[str] = []
    root = ROOT.resolve()
    for source in markdown_files(paths):
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(
                f"{source.relative_to(ROOT)}: cannot read UTF-8 Markdown ({exc})"
            )
            continue
        for line, raw_target in _targets(text):
            target = _local_target(source, raw_target)
            if target is None:
                continue
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(
                    f"{source.relative_to(ROOT)}:{line}: "
                    f"local link escapes repository: {raw_target}"
                )
                continue
            if not target.exists():
                failures.append(
                    f"{source.relative_to(ROOT)}:{line}: missing local target: {raw_target}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Optional Markdown files or directories")
    args = parser.parse_args(argv)
    failures = check(args.paths or None)
    if failures:
        print("Markdown link check failed:")
        print("\n".join(failures))
        return 1
    print(f"Markdown link check passed ({len(markdown_files(args.paths or None))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
