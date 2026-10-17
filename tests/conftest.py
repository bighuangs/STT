"""Shared deterministic and isolation fixtures for the offline test suite."""
from __future__ import annotations

import ipaddress
import os
import random
import shutil
import socket
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACT_ROOT = PROJECT_ROOT / ".test-artifacts"


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temporary writes inside the workspace unless explicitly set."""
    if config.option.basetemp is None:
        # pytest only creates the final basetemp component; its parent must exist.
        TEST_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(TEST_ARTIFACT_ROOT / "pytest")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag directory-based suites without changing whether offline CI executes them."""
    for item in items:
        parts = {part.lower() for part in Path(str(item.path)).parts}
        if "tests" in parts and "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "tests" in parts and "external" in parts:
            item.add_marker(pytest.mark.external)


@pytest.fixture(autouse=True)
def deterministic_random_seed() -> Iterator[int]:
    """Reset common PRNGs for every case and restore their visible state."""
    seed = 20260721
    python_state = random.getstate()
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - optional in lightweight installs
        np = None
    numpy_state = np.random.get_state() if np is not None else None
    if np is not None:
        np.random.seed(seed)
    old_hash_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        yield seed
    finally:
        random.setstate(python_state)
        if np is not None and numpy_state is not None:
            np.random.set_state(numpy_state)
        if old_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_hash_seed


@dataclass(frozen=True)
class IsolatedDatabase:
    path: Path
    url: str


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IsolatedDatabase:
    """Give a test a unique SQLite URL without mutating the process-wide engine."""
    path = tmp_path / "control.db"
    database = IsolatedDatabase(path=path, url=f"sqlite:///{path.as_posix()}")
    monkeypatch.setenv("CONTROL_DATABASE_URL", database.url)
    monkeypatch.setenv("STT_CONTROL_DATABASE_URL", database.url)
    return database


def _is_loopback(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in socket guard: loopback stays available for TestClient/local servers."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def guarded_connect(sock: socket.socket, address):
        if not isinstance(address, tuple) or _is_loopback(address[0]):
            return original_connect(sock, address)
        raise OSError(f"external network is disabled for this test: {address[0]}")

    def guarded_connect_ex(sock: socket.socket, address):
        if not isinstance(address, tuple) or _is_loopback(address[0]):
            return original_connect_ex(sock, address)
        raise OSError(f"external network is disabled for this test: {address[0]}")

    def guarded_create_connection(address, *args, **kwargs):
        if isinstance(address, tuple) and _is_loopback(address[0]):
            return original_create_connection(address, *args, **kwargs)
        host = address[0] if isinstance(address, tuple) else address
        raise OSError(f"external network is disabled for this test: {host}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


@pytest.fixture(scope="session", autouse=True)
def clean_test_artifacts_after_session() -> Iterator[None]:
    """Remove only the suite-owned cache directory; never touch user output."""
    yield
    cache = TEST_ARTIFACT_ROOT / "cache"
    if cache.exists():
        try:
            shutil.rmtree(cache)
        except PermissionError as exc:
            # Windows 上一次由不同 Codex/CI 沙箱 SID 创建的 pytest cache
            # 可能无法由当前进程枚举。缓存清理不是测试事实，不能把已经通过
            # 的业务断言改判为失败；保留路径并给出可见告警即可。
            warnings.warn(
                pytest.PytestWarning(
                    f"pytest cache cleanup skipped because access was denied: {exc}"
                ),
                stacklevel=1,
            )
