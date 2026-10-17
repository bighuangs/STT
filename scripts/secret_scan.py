"""轻量 secret scan；用于提交前/CI 检查，不读取被 .gitignore 排除的本地 .env。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "openai-like-key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "credential-assignment": re.compile(
        r"(?i)(?:^|[\s{,])(?:['\"](?:api[_-]?key|secret[_-]?key|password|token)['\"]|"
        r"(?:api[_-]?key|secret[_-]?key|password|token))\s*[:=]\s*['\"][^'\"]{16,}['\"]"
    ),
}
SAFE_MARKERS = (
    "replace-with-", "your-key", "your-mineru-token", "dev-only-change-this",
    "Stt-Admin-ChangeMe!", "secret-scan: allow-test",
)
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".html", ".js", ".env", ".example", ".ini"}


def candidate_files() -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    files = []
    for relative in output.splitlines():
        path = ROOT / relative
        if path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or path.name == ".env.example"):
            files.append(path)
    return files


def scan() -> list[str]:
    findings: list[str] = []
    for path in candidate_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            if any(marker in line for marker in SAFE_MARKERS):
                continue
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(ROOT)}:{number}: {name}")
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print("Secret scan failed（只输出位置，不输出命中值）:")
        print("\n".join(findings))
        return 1
    print("Secret scan passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
