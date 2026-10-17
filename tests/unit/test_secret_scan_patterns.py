"""保护发布扫描的真实赋值检测，避免把字符串操作误报为凭据。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('api_key="synthetic-long-value-for-scanner"', True),  # secret-scan: allow-test
        ('    "password": "synthetic-long-value-for-scanner",', True),  # secret-scan: allow-test
        ('{"token": "synthetic-long-value-for-scanner"}', True),  # secret-scan: allow-test
        ('key.removeprefix("token:") and key.startswith("token:")', False),
        ('token = "short"', False),
    ],
)
def test_credential_assignment_pattern(line: str, expected: bool) -> None:
    source = Path(__file__).resolve().parents[2] / "scripts" / "secret_scan.py"
    spec = importlib.util.spec_from_file_location("release_secret_scan", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert bool(module.PATTERNS["credential-assignment"].search(line)) is expected
