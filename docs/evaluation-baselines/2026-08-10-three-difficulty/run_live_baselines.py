#!/usr/bin/env python3
"""Run and archive the three online/live evaluation baselines.

This script is intentionally a thin client of the local query API.  It does
not import application services or touch the control database directly.  The
three immutable dataset manifests are imported in easy/medium/hard order,
preflighted in ``live`` mode, executed as first baselines
(``baseline_run_id=null``), and archived below this directory.

Credentials are read from the project ``.env``/process environment and are
never accepted as command-line arguments.  Every HTTP exchange is captured in
a redacted JSONL audit log.  Runtime artifacts remain under
``output/evaluation`` on the server; the copies in ``docs/runs`` include both
the server/source digest and the downloaded/target digest.

Typical invocation from the repository root::

    uv run python docs/evaluation-baselines/2026-08-10-three-difficulty/\
        run_live_baselines.py --kb-id kb_xxx

Use ``--resume`` after an interrupted poll/archive.  Use ``--overwrite`` only
to start a new set of baseline runs; prior indexes are copied to
``baselines/history`` and prior run directories are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from dotenv import dotenv_values


DIFFICULTIES = ("easy", "medium", "hard")
ARTIFACT_FILENAMES = {
    "json": "report.json",
    "markdown": "report.md",
    "html": "report.html",
    "summary": "summary.json",
    "manifest": "manifest.json",
    "results": "results.jsonl",
}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_RUN_STATUSES = {"queued", "running"}
RUN_INDEX_SCHEMA = "stt.eval.live-baseline-run-index.v1"
BASELINES_SCHEMA = "stt.eval.live-baselines.v1"
ARTIFACT_COPY_SCHEMA = "stt.eval.artifact-copy-manifest.v1"
SNAPSHOT_SCHEMA = "stt.eval.api-snapshots.v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_POLL_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0

SCRIPT_PATH = Path(__file__).resolve()
SUITE_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = next(
    (candidate for candidate in (SUITE_ROOT, *SUITE_ROOT.parents) if (candidate / "pyproject.toml").is_file()),
    None,
)
if PROJECT_ROOT is None:  # pragma: no cover - deployment/layout guard
    raise RuntimeError("cannot locate repository root containing pyproject.toml")

DATASET_DIR = SUITE_ROOT / "datasets"
RUNS_DIR = SUITE_ROOT / "runs"
BASELINES_DIR = SUITE_ROOT / "baselines"
BASELINES_PATH = BASELINES_DIR / "baselines.json"
QUALITY_SUMMARY_PATH = SUITE_ROOT / "quality" / "quality-summary.json"
RUN_INDEX_PATH = SUITE_ROOT / "run-index.json"
CHECKSUMS_PATH = SUITE_ROOT / "SHA256SUMS"
API_SNAPSHOT_ROOT = SUITE_ROOT / "api-snapshots"


class BaselineRunnerError(RuntimeError):
    """A safe, user-facing workflow failure."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    root = SUITE_ROOT.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BaselineRunnerError(f"path escapes suite root: {resolved}") from exc


def _safe_component(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
        raise BaselineRunnerError(f"unsafe {label} returned by API")
    if normalized in {".", ".."}:
        raise BaselineRunnerError(f"unsafe {label} returned by API")
    return normalized


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Tolerate short-lived Windows reader/antivirus locks on result files."""

    for attempt in range(1, 21):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(min(0.5, 0.05 * attempt))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not repeat the destination name in the temporary leaf.  Deep audited
    # run directories can otherwise cross Windows MAX_PATH even though the
    # final artifact path itself is valid.
    temporary = path.parent / f".{uuid.uuid4().hex[:12]}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{uuid.uuid4().hex[:12]}.part"
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineRunnerError(f"cannot read valid JSON from {_relative(path)}") from exc


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _normalize_local_base_url(raw: str) -> str:
    try:
        parsed = urlsplit(str(raw).strip())
        port = parsed.port
    except ValueError as exc:
        raise BaselineRunnerError("invalid --base-url") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BaselineRunnerError("--base-url must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BaselineRunnerError("--base-url cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise BaselineRunnerError("--base-url cannot contain an API path")
    hostname = parsed.hostname.lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise BaselineRunnerError(
            "refusing to send bootstrap credentials to a non-loopback --base-url"
        )
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/")


def _load_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for key, value in dotenv_values(env_path).items():
            if key and value is not None:
                values[str(key)] = str(value)
    for key, value in os.environ.items():
        values[str(key)] = str(value)
    return values


def _first_config(config: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = str(config.get(name) or "").strip()
        if value:
            return value
    return default


def _first_secret_config(config: dict[str, str], *names: str) -> str:
    """Return a secret verbatim; stripping could change a valid password."""

    for name in names:
        value = config.get(name)
        if value is not None and str(value):
            return str(value)
    return ""


class Redactor:
    """Redact credential fields and known secret values from persisted output."""

    _SENSITIVE_KEYS = {
        "password",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "set_cookie",
        "api_key",
        "client_secret",
        "token_secret",
    }

    def __init__(self, *values: str) -> None:
        self._values: set[str] = set()
        for value in values:
            self.add(value)

    def add(self, value: Any) -> None:
        if isinstance(value, str) and len(value) >= 4:
            self._values.add(value)

    @classmethod
    def _sensitive_key(cls, key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        return (
            normalized in cls._SENSITIVE_KEYS
            or "api_key" in normalized
            or "secret" in normalized
            or normalized.endswith("_password")
            or normalized.endswith("_access_token")
            or normalized.endswith("_refresh_token")
            or normalized.endswith("_api_key")
            or normalized.endswith("_secret")
        )

    def text(self, value: str) -> str:
        redacted = value
        for secret in sorted(self._values, key=len, reverse=True):
            redacted = redacted.replace(secret, "<redacted>")
        return redacted

    def clean(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    "<redacted>" if self._sensitive_key(str(key)) else self.clean(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.clean(item) for item in value]
        if isinstance(value, tuple):
            return [self.clean(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


class ApiSnapshotRecorder:
    """Append every redacted HTTP request/response snapshot to one JSONL log."""

    _HEADER_ALLOWLIST = {
        "content-type",
        "content-length",
        "content-disposition",
        "x-request-id",
        "x-content-type-options",
        "cache-control",
    }

    def __init__(self, invocation_id: str, base_url: str, redactor: Redactor) -> None:
        self.invocation_id = invocation_id
        self.redactor = redactor
        self.directory = API_SNAPSHOT_ROOT / invocation_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log_path = self.directory / "requests.jsonl"
        self.metadata_path = self.directory / "metadata.json"
        self._sequence = 0
        self._started_at = _utc_now()
        self._closed = False
        _write_json(
            self.metadata_path,
            {
                "schema_version": SNAPSHOT_SCHEMA,
                "invocation_id": invocation_id,
                "base_url": base_url,
                "started_at": self._started_at,
                "status": "running",
                "request_count": 0,
                "log_path": _relative(self.log_path),
            },
        )

    @staticmethod
    def _headers(response: requests.Response) -> dict[str, str]:
        return {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in ApiSnapshotRecorder._HEADER_ALLOWLIST
        }

    def record(
        self,
        *,
        method: str,
        path: str,
        request_json: Any,
        response: requests.Response,
        response_json: Any | None = None,
        response_bytes: bytes | None = None,
        attempt: int = 1,
    ) -> int:
        self._sequence += 1
        response_record: dict[str, Any] = {
            "status_code": response.status_code,
            "headers": self._headers(response),
        }
        if response_bytes is not None:
            response_record["body"] = {
                "kind": "binary",
                "sha256": _sha256_bytes(response_bytes),
                "size_bytes": len(response_bytes),
            }
        elif response_json is not None:
            response_record["body"] = {
                "kind": "json",
                "value": self.redactor.clean(response_json),
            }
        else:
            excerpt = self.redactor.text(response.text[:8000])
            response_record["body"] = {"kind": "text", "excerpt": excerpt}
        record = {
            "schema_version": SNAPSHOT_SCHEMA,
            "sequence": self._sequence,
            "captured_at": _utc_now(),
            "attempt": attempt,
            "request": {
                "method": method.upper(),
                "path": path,
                "json": self.redactor.clean(request_json),
            },
            "response": response_record,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
        return self._sequence

    def close(self, *, status: str, error_type: str | None = None) -> dict[str, Any]:
        if self._closed:
            return {
                "directory": _relative(self.directory),
                "log_path": _relative(self.log_path),
                "metadata_path": _relative(self.metadata_path),
            }
        self._closed = True
        metadata = {
            "schema_version": SNAPSHOT_SCHEMA,
            "invocation_id": self.invocation_id,
            "started_at": self._started_at,
            "ended_at": _utc_now(),
            "status": status,
            "error_type": error_type,
            "request_count": self._sequence,
            "log_path": _relative(self.log_path),
        }
        _write_json(self.metadata_path, metadata)
        return {
            "directory": _relative(self.directory),
            "log_path": _relative(self.log_path),
            "log_sha256": _sha256_file(self.log_path) if self.log_path.is_file() else None,
            "metadata_path": _relative(self.metadata_path),
            "metadata_sha256": _sha256_file(self.metadata_path),
            "request_count": self._sequence,
        }


class ApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        tenant_id: str | None,
        timeout_seconds: float,
        recorder: ApiSnapshotRecorder,
        redactor: Redactor,
    ) -> None:
        self.base_url = base_url
        self.email = email
        self.password = password
        self.tenant_id = tenant_id
        self.timeout_seconds = timeout_seconds
        self.recorder = recorder
        self.redactor = redactor
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "stt-live-baseline-archiver/1.0",
            }
        )
        self.logged_in = False

    @staticmethod
    def _json(response: requests.Response) -> Any | None:
        try:
            return response.json()
        except (ValueError, requests.JSONDecodeError):
            return None

    def _remember_auth_secrets(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.redactor.add(payload.get("access_token"))
        self.redactor.add(payload.get("refresh_token"))

    def _raise_http(self, method: str, path: str, response: requests.Response, sequence: int) -> None:
        raise BaselineRunnerError(
            f"{method.upper()} {path} failed with HTTP {response.status_code}; "
            f"see API snapshot sequence {sequence}"
        )

    def login(self) -> dict[str, Any]:
        path = "/api/v1/auth/login"
        body: dict[str, Any] = {"email": self.email, "password": self.password}
        if self.tenant_id:
            body["tenant_id"] = self.tenant_id
        response = self.session.post(
            f"{self.base_url}{path}", json=body, timeout=self.timeout_seconds
        )
        payload = self._json(response)
        self._remember_auth_secrets(payload)
        sequence = self.recorder.record(
            method="POST",
            path=path,
            request_json=body,
            response=response,
            response_json=payload,
        )
        if not response.ok or not isinstance(payload, dict):
            self._raise_http("POST", path, response, sequence)
        self.logged_in = True
        return payload

    def _refresh(self) -> None:
        path = "/api/v1/auth/refresh"
        body: dict[str, Any] = {}
        response = self.session.post(
            f"{self.base_url}{path}", json=body, timeout=self.timeout_seconds
        )
        payload = self._json(response)
        self._remember_auth_secrets(payload)
        sequence = self.recorder.record(
            method="POST",
            path=path,
            request_json=body,
            response=response,
            response_json=payload,
        )
        if not response.ok or not isinstance(payload, dict):
            self._raise_http("POST", path, response, sequence)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        retry_auth: bool = True,
        transport_attempt: int = 1,
    ) -> dict[str, Any]:
        try:
            response = self.session.request(
                method.upper(),
                f"{self.base_url}{path}",
                json=body,
                timeout=self.timeout_seconds,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Poll/read operations are safe to replay.  Do not blindly retry a
            # POST because the server may have committed it before disconnecting.
            if method.upper() == "GET" and transport_attempt < 5:
                time.sleep(float(transport_attempt))
                return self.request_json(
                    method,
                    path,
                    body=body,
                    retry_auth=retry_auth,
                    transport_attempt=transport_attempt + 1,
                )
            raise BaselineRunnerError(
                f"{method.upper()} {path} transport failed after {transport_attempt} attempt(s)"
            ) from exc
        payload = self._json(response)
        sequence = self.recorder.record(
            method=method,
            path=path,
            request_json=body,
            response=response,
            response_json=payload,
        )
        if response.status_code == 401 and retry_auth:
            self._refresh()
            return self.request_json(
                method,
                path,
                body=body,
                retry_auth=False,
                transport_attempt=transport_attempt,
            )
        if not response.ok:
            self._raise_http(method, path, response, sequence)
        if not isinstance(payload, dict):
            raise BaselineRunnerError(
                f"{method.upper()} {path} returned a non-object JSON response; "
                f"see API snapshot sequence {sequence}"
            )
        return payload

    def download(
        self,
        path: str,
        *,
        retry_auth: bool = True,
        transport_attempt: int = 1,
    ) -> bytes:
        try:
            response = self.session.get(
                f"{self.base_url}{path}", timeout=self.timeout_seconds
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if transport_attempt < 5:
                time.sleep(float(transport_attempt))
                return self.download(
                    path,
                    retry_auth=retry_auth,
                    transport_attempt=transport_attempt + 1,
                )
            raise BaselineRunnerError(
                f"GET {path} transport failed after {transport_attempt} attempt(s)"
            ) from exc
        content_type = str(response.headers.get("content-type") or "").lower()
        if response.ok:
            content = response.content
            sequence = self.recorder.record(
                method="GET",
                path=path,
                request_json=None,
                response=response,
                response_bytes=content,
            )
        else:
            payload = self._json(response) if "json" in content_type else None
            sequence = self.recorder.record(
                method="GET",
                path=path,
                request_json=None,
                response=response,
                response_json=payload,
            )
            content = b""
        if response.status_code == 401 and retry_auth:
            self._refresh()
            return self.download(
                path,
                retry_auth=False,
                transport_attempt=transport_attempt,
            )
        if not response.ok:
            self._raise_http("GET", path, response, sequence)
        return content

    def logout(self) -> None:
        if not self.logged_in:
            return
        path = "/api/v1/auth/logout"
        response = self.session.post(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        payload = self._json(response)
        sequence = self.recorder.record(
            method="POST",
            path=path,
            request_json=None,
            response=response,
            response_json=payload,
        )
        self.logged_in = False
        if not response.ok:
            self._raise_http("POST", path, response, sequence)


@dataclass(frozen=True)
class DatasetSpec:
    difficulty: str
    path: Path
    file_sha256: str
    request_body: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def case_count(self) -> int:
        return len(self.manifest.get("cases") or [])


def _load_dataset(difficulty: str) -> DatasetSpec:
    path = DATASET_DIR / f"{difficulty}.json"
    if not path.is_file():
        raise BaselineRunnerError(f"missing dataset file: {_relative(path)}")
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise BaselineRunnerError(f"dataset must be a JSON object: {_relative(path)}")
    if isinstance(raw.get("manifest"), dict):
        manifest = dict(raw["manifest"])
        request_body: dict[str, Any] = {"manifest": manifest}
        if raw.get("parent_dataset_id") is not None:
            request_body["parent_dataset_id"] = raw["parent_dataset_id"]
    else:
        manifest = dict(raw)
        request_body = {"manifest": manifest}
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) < 50:
        raise BaselineRunnerError(
            f"{difficulty} dataset must contain at least 50 cases"
        )
    declared = str(manifest.get("difficulty") or "").strip().lower()
    if declared and declared != difficulty:
        raise BaselineRunnerError(
            f"{difficulty} dataset declares mismatched difficulty {declared!r}"
        )
    default_mode = str(manifest.get("evaluation_mode_default") or "live")
    if default_mode != "live":
        raise BaselineRunnerError(
            f"{difficulty} dataset evaluation_mode_default must be live"
        )
    for required in ("dataset_key", "name", "version", "schema_version", "source"):
        if not manifest.get(required):
            raise BaselineRunnerError(
                f"{difficulty} dataset is missing required field {required}"
            )
    return DatasetSpec(
        difficulty=difficulty,
        path=path,
        file_sha256=_sha256_file(path),
        request_body=request_body,
        manifest=manifest,
    )


def _verify_quality_gate() -> dict[str, Any]:
    """Fail closed before any API mutation when dataset QA is stale or failed."""

    if not QUALITY_SUMMARY_PATH.is_file():
        raise BaselineRunnerError("quality/quality-summary.json is required before import")
    summary = _read_json(QUALITY_SUMMARY_PATH)
    if not isinstance(summary, dict) or summary.get("ready_for_import") is not True:
        raise BaselineRunnerError("dataset quality gate is not ready_for_import")
    if summary.get("suite_key") != "three-difficulty-live-2026-08-10":
        raise BaselineRunnerError("dataset quality gate suite_key does not match")
    expected = summary.get("dataset_checksums")
    if not isinstance(expected, dict):
        raise BaselineRunnerError("dataset quality gate has no file checksums")
    actual = {
        difficulty: _sha256_file(DATASET_DIR / f"{difficulty}.json")
        for difficulty in DIFFICULTIES
    }
    if expected != actual:
        raise BaselineRunnerError(
            "dataset files changed after quality validation; rerun validate_suite.py --online-audit"
        )
    return {
        "path": _relative(QUALITY_SUMMARY_PATH),
        "sha256": _sha256_file(QUALITY_SUMMARY_PATH),
        "deterministic_status": summary.get("deterministic_status"),
        "online_audit_status": summary.get("online_audit_status"),
        "dataset_checksums": actual,
    }


def _envelope_data(payload: dict[str, Any], *, operation: str) -> Any:
    if "data" not in payload:
        raise BaselineRunnerError(f"{operation} response has no ResourceEnvelope.data")
    return payload["data"]


def _archive_previous_state() -> str | None:
    candidates = [path for path in (RUN_INDEX_PATH, BASELINES_PATH) if path.is_file()]
    if not candidates:
        return None
    archive = BASELINES_DIR / "history" / f"{_timestamp_slug()}-{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, Any]] = []
    for source in candidates:
        target = archive / source.name
        shutil.copy2(source, target)
        copied.append(
            {
                "source": _relative(source),
                "archived_copy": _relative(target),
                "sha256": _sha256_file(target),
            }
        )
    _write_json(
        archive / "archive-manifest.json",
        {"archived_at": _utc_now(), "files": copied},
    )
    return _relative(archive)


def _new_run_index(*, base_url: str, kb_id: str, archive: str | None) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": RUN_INDEX_SCHEMA,
        "suite_root": _relative(SUITE_ROOT),
        "mode": "live",
        "base_url": base_url,
        "kb_id": kb_id,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "supersedes_archive": archive,
        "invocations": [],
        "difficulties": {},
    }


def _persist_run_index(index: dict[str, Any]) -> None:
    index["updated_at"] = _utc_now()
    _write_json(RUN_INDEX_PATH, index)


def _prepare_run_index(args: argparse.Namespace) -> dict[str, Any]:
    existing_index = RUN_INDEX_PATH.is_file()
    existing_baselines = BASELINES_PATH.is_file()
    if args.resume:
        if not existing_index:
            raise BaselineRunnerError("--resume requires an existing run-index.json")
        index = _read_json(RUN_INDEX_PATH)
        if not isinstance(index, dict) or index.get("schema_version") != RUN_INDEX_SCHEMA:
            raise BaselineRunnerError("run-index.json has an unsupported schema")
        if index.get("base_url") != args.base_url or index.get("kb_id") != args.kb_id:
            raise BaselineRunnerError(
                "--resume requires the same --base-url and --kb-id as run-index.json"
            )
        index["status"] = "in_progress"
        return index
    if not args.overwrite and (existing_index or existing_baselines):
        raise BaselineRunnerError(
            "baseline state already exists; use --resume or --overwrite"
        )
    archive = _archive_previous_state() if args.overwrite else None
    return _new_run_index(base_url=args.base_url, kb_id=args.kb_id, archive=archive)


def _baseline_status(index: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "schema_version": BASELINES_SCHEMA,
        "status": status,
        "suite_key": None,
        "mode": "live",
        "base_url": index["base_url"],
        "kb_id": index["kb_id"],
        "updated_at": _utc_now(),
        "baselines": {},
    }


def _poll_run(
    client: ApiClient,
    *,
    run_id: str,
    difficulty: str,
    timeout_seconds: float,
    interval_seconds: float,
    index: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = quote(run_id, safe="")
    path = f"/api/v1/evaluation/runs/{encoded}"
    deadline = time.monotonic() + timeout_seconds
    last_display: tuple[str, int, int] | None = None
    while True:
        envelope = client.request_json("GET", path)
        run = _envelope_data(envelope, operation=f"poll {difficulty}")
        if not isinstance(run, dict):
            raise BaselineRunnerError(f"poll {difficulty} returned non-object run data")
        if str(run.get("id") or "") != run_id:
            raise BaselineRunnerError(f"poll {difficulty} returned the wrong run id")
        status = str(run.get("status") or "")
        completed = int(run.get("completed_cases") or 0)
        total = int(run.get("total_cases") or 0)
        display = (status, completed, total)
        if display != last_display:
            print(f"[{difficulty}] run {run_id}: {status} ({completed}/{total})", flush=True)
            last_display = display
            entry = index["difficulties"][difficulty]
            entry.update(
                {
                    "status": status,
                    "completed_cases": completed,
                    "total_cases": total,
                    "last_polled_at": _utc_now(),
                }
            )
            _persist_run_index(index)
        if status in TERMINAL_RUN_STATUSES:
            return envelope, run
        if status not in ACTIVE_RUN_STATUSES:
            raise BaselineRunnerError(
                f"{difficulty} run entered unsupported status {status!r}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BaselineRunnerError(
                f"poll timeout reached for {difficulty} run {run_id}; rerun with --resume"
            )
        time.sleep(min(interval_seconds, remaining))


def _archive_artifacts(
    client: ApiClient,
    *,
    difficulty: str,
    run: dict[str, Any],
    allow_repair: bool,
) -> tuple[Path, dict[str, Any]]:
    run_id = _safe_component(str(run.get("id") or ""), label="run id")
    run_dir = RUNS_DIR / difficulty / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_paths = run.get("report_paths")
    if not isinstance(report_paths, dict):
        raise BaselineRunnerError(f"{difficulty} run has no report_paths")
    copies: dict[str, Any] = {}
    for artifact, filename in ARTIFACT_FILENAMES.items():
        source = report_paths.get(artifact)
        if not isinstance(source, dict):
            raise BaselineRunnerError(
                f"{difficulty} run is missing required {artifact} artifact"
            )
        expected_sha = str(source.get("sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise BaselineRunnerError(
                f"{difficulty} run has invalid source digest for {artifact}"
            )
        api_path = (
            f"/api/v1/evaluation/runs/{quote(run_id, safe='')}/artifacts/"
            f"{quote(artifact, safe='')}"
        )
        content = client.download(api_path)
        downloaded_sha = _sha256_bytes(content)
        if downloaded_sha != expected_sha:
            raise BaselineRunnerError(
                f"{difficulty} {artifact} download digest does not match server manifest"
            )
        destination = run_dir / filename
        previous_sha = _sha256_file(destination) if destination.is_file() else None
        if previous_sha is not None and previous_sha != downloaded_sha and not allow_repair:
            raise BaselineRunnerError(
                f"archive target differs for {_relative(destination)}; use --resume to repair"
            )
        action = "verified_existing"
        if previous_sha != downloaded_sha:
            _write_bytes(destination, content)
            action = "created" if previous_sha is None else "repaired"
        target_sha = _sha256_file(destination)
        if target_sha != expected_sha:
            raise BaselineRunnerError(
                f"target digest verification failed for {_relative(destination)}"
            )
        copies[artifact] = {
            "filename": filename,
            "action": action,
            "source": {
                "api_path": api_path,
                "server_relative_path": source.get("path"),
                "server_sha256": expected_sha,
                "server_size_bytes": source.get("size_bytes"),
                "download_sha256": downloaded_sha,
                "download_size_bytes": len(content),
            },
            "target": {
                "relative_path": _relative(destination),
                "previous_sha256": previous_sha,
                "sha256": target_sha,
                "size_bytes": destination.stat().st_size,
            },
            "source_target_sha256_match": downloaded_sha == target_sha == expected_sha,
        }
    manifest = {
        "schema_version": ARTIFACT_COPY_SCHEMA,
        "created_at": _utc_now(),
        "difficulty": difficulty,
        "run_id": run_id,
        "server_artifacts_remain_authoritative": True,
        "server_artifact_root": f"output/evaluation/{run_id}",
        "copies": copies,
    }
    manifest_path = run_dir / "artifact-copy-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _write_api_copy(run_dir: Path, filename: str, payload: Any, redactor: Redactor) -> Path:
    path = run_dir / "api" / filename
    _write_json(path, redactor.clean(payload))
    return path


def _process_difficulty(
    *,
    difficulty: str,
    args: argparse.Namespace,
    client: ApiClient,
    redactor: Redactor,
    index: dict[str, Any],
) -> dict[str, Any]:
    spec = _load_dataset(difficulty)
    existing = index["difficulties"].get(difficulty)
    if args.resume and isinstance(existing, dict):
        prior_hash = str(existing.get("dataset_file_sha256") or "")
        if prior_hash and prior_hash != spec.file_sha256:
            raise BaselineRunnerError(
                f"cannot resume {difficulty}: dataset file checksum changed"
            )
        entry = dict(existing)
    else:
        entry = {}
    entry.update(
        {
            "difficulty": difficulty,
            "dataset_path": _relative(spec.path),
            "dataset_file_sha256": spec.file_sha256,
            "local_case_count": spec.case_count,
            "mode": "live",
            "status": entry.get("status") or "dataset_import_pending",
        }
    )
    index["difficulties"][difficulty] = entry
    _persist_run_index(index)

    print(f"[{difficulty}] importing immutable dataset ({spec.case_count} cases)", flush=True)
    import_envelope = client.request_json(
        "POST", "/api/v1/evaluation/datasets", body=spec.request_body
    )
    dataset = _envelope_data(import_envelope, operation=f"import {difficulty}")
    if not isinstance(dataset, dict):
        raise BaselineRunnerError(f"import {difficulty} returned non-object dataset data")
    dataset_id = _safe_component(str(dataset.get("id") or ""), label="dataset id")
    if int(dataset.get("case_count") or 0) < 50:
        raise BaselineRunnerError(f"imported {difficulty} dataset contains fewer than 50 cases")
    if int(dataset.get("case_count") or 0) != spec.case_count:
        raise BaselineRunnerError(
            f"imported {difficulty} dataset case count differs from local manifest"
        )
    prior_dataset_id = str(entry.get("dataset_id") or "")
    if prior_dataset_id and prior_dataset_id != dataset_id:
        raise BaselineRunnerError(f"cannot resume {difficulty}: dataset id changed")
    entry.update(
        {
            "dataset_id": dataset_id,
            "dataset_key": dataset.get("dataset_key"),
            "dataset_version": dataset.get("version"),
            "dataset_checksum": dataset.get("checksum"),
            "case_count": dataset.get("case_count"),
            "status": "preflight_pending",
        }
    )
    _persist_run_index(index)

    preflight_body = {"dataset_id": dataset_id, "kb_id": args.kb_id, "mode": "live"}
    print(f"[{difficulty}] running live preflight", flush=True)
    preflight_envelope = client.request_json(
        "POST", "/api/v1/evaluation/preflight", body=preflight_body
    )
    preflight = _envelope_data(preflight_envelope, operation=f"preflight {difficulty}")
    if not isinstance(preflight, dict):
        raise BaselineRunnerError(f"preflight {difficulty} returned non-object data")
    entry["preflight_ready"] = bool(preflight.get("ready"))
    entry["preflight_at"] = _utc_now()
    if not preflight.get("ready"):
        checks = preflight.get("checks") if isinstance(preflight.get("checks"), dict) else {}
        entry["preflight_failed_checks"] = sorted(
            str(name) for name, passed in checks.items() if not passed
        )
        entry["status"] = "preflight_failed"
        _persist_run_index(index)
        failed = ", ".join(entry["preflight_failed_checks"]) or "unknown checks"
        raise BaselineRunnerError(f"{difficulty} live preflight failed: {failed}")

    run_id = str(entry.get("run_id") or "") if args.resume else ""
    create_envelope: dict[str, Any] | None = None
    if run_id:
        run_id = _safe_component(run_id, label="run id")
        resume_envelope = client.request_json(
            "GET", f"/api/v1/evaluation/runs/{quote(run_id, safe='')}"
        )
        run = _envelope_data(resume_envelope, operation=f"resume {difficulty}")
        if not isinstance(run, dict):
            raise BaselineRunnerError(f"resume {difficulty} returned non-object run data")
        if (
            run.get("dataset_id") != dataset_id
            or run.get("kb_id") != args.kb_id
            or run.get("mode") != "live"
            or run.get("baseline_run_id") is not None
        ):
            raise BaselineRunnerError(f"cannot resume {difficulty}: run identity changed")
        initial_envelope = resume_envelope
        print(f"[{difficulty}] resuming baseline run {run_id}", flush=True)
    else:
        create_body = {
            "dataset_id": dataset_id,
            "kb_id": args.kb_id,
            "mode": "live",
            "name": f"three-difficulty-{difficulty}-live-baseline",
            "baseline_run_id": None,
        }
        print(f"[{difficulty}] creating first live baseline run", flush=True)
        create_envelope = client.request_json(
            "POST", "/api/v1/evaluation/runs", body=create_body
        )
        run = _envelope_data(create_envelope, operation=f"create run {difficulty}")
        if not isinstance(run, dict):
            raise BaselineRunnerError(f"create run {difficulty} returned non-object data")
        run_id = _safe_component(str(run.get("id") or ""), label="run id")
        if run.get("baseline_run_id") is not None or run.get("mode") != "live":
            raise BaselineRunnerError(f"created {difficulty} run is not a first live baseline")
        entry.update(
            {
                "run_id": run_id,
                "status": str(run.get("status") or "queued"),
                "created_at": run.get("created_at"),
            }
        )
        _persist_run_index(index)
        initial_envelope = create_envelope

    run_dir = RUNS_DIR / difficulty / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_api_copy(run_dir, "dataset-import.json", import_envelope, redactor)
    _write_api_copy(run_dir, "preflight.json", preflight_envelope, redactor)
    _write_api_copy(
        run_dir,
        "run-create.json" if create_envelope is not None else "run-resume.json",
        initial_envelope,
        redactor,
    )

    final_envelope, final_run = _poll_run(
        client,
        run_id=run_id,
        difficulty=difficulty,
        timeout_seconds=args.poll_timeout,
        interval_seconds=args.poll_interval,
        index=index,
    )
    _write_api_copy(run_dir, "run-final.json", final_envelope, redactor)
    entry.update(
        {
            "status": final_run.get("status"),
            "completed_cases": final_run.get("completed_cases"),
            "total_cases": final_run.get("total_cases"),
            "completed_at": final_run.get("completed_at"),
            "gate_status": (final_run.get("gate_result") or {}).get("status"),
            "gate_passed": (final_run.get("gate_result") or {}).get("passed"),
        }
    )
    _persist_run_index(index)
    if final_run.get("status") != "succeeded":
        error_code = final_run.get("error_code") or "unknown_error"
        raise BaselineRunnerError(
            f"{difficulty} run {run_id} ended as {final_run.get('status')} ({error_code})"
        )

    encoded_run_id = quote(run_id, safe="")
    results_envelope = client.request_json(
        "GET", f"/api/v1/evaluation/runs/{encoded_run_id}/results"
    )
    comparison_envelope = client.request_json(
        "GET", f"/api/v1/evaluation/runs/{encoded_run_id}/comparison"
    )
    _write_api_copy(run_dir, "results.json", results_envelope, redactor)
    _write_api_copy(run_dir, "comparison.json", comparison_envelope, redactor)
    results = _envelope_data(results_envelope, operation=f"results {difficulty}")
    if not isinstance(results, list) or len(results) != int(final_run.get("total_cases") or 0):
        raise BaselineRunnerError(f"{difficulty} result count does not match run total")

    manifest_path, copy_manifest = _archive_artifacts(
        client,
        difficulty=difficulty,
        run=final_run,
        allow_repair=bool(args.resume or args.overwrite),
    )
    if not all(
        bool(item.get("source_target_sha256_match"))
        for item in copy_manifest["copies"].values()
    ):
        raise BaselineRunnerError(f"{difficulty} artifact copy verification failed")

    baseline_record = {
        "difficulty": difficulty,
        "mode": "live",
        "kb_id": args.kb_id,
        "dataset": {
            "id": dataset_id,
            "key": dataset.get("dataset_key"),
            "version": dataset.get("version"),
            "checksum": dataset.get("checksum"),
            "case_count": dataset.get("case_count"),
            "source_path": _relative(spec.path),
            "source_sha256": spec.file_sha256,
        },
        "run_id": run_id,
        "baseline_run_id": None,
        "status": final_run.get("status"),
        "gate_result": final_run.get("gate_result") or {},
        "metrics_summary": final_run.get("metrics_summary") or {},
        "comparison_summary": final_run.get("comparison_summary") or {},
        "config_snapshot": final_run.get("config_snapshot") or {},
        "started_at": final_run.get("started_at"),
        "completed_at": final_run.get("completed_at"),
        "result_count": len(results),
        "run_directory": _relative(run_dir),
        "artifact_copy_manifest": {
            "path": _relative(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
    }
    entry.update(
        {
            "status": "archived",
            "result_count": len(results),
            "run_directory": _relative(run_dir),
            "artifact_copy_manifest": baseline_record["artifact_copy_manifest"],
            "baseline": baseline_record,
        }
    )
    _persist_run_index(index)
    print(f"[{difficulty}] archived and verified: {_relative(run_dir)}", flush=True)
    return baseline_record


def _finalize_baselines(
    index: dict[str, Any], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if tuple(records) != DIFFICULTIES:
        raise BaselineRunnerError("all three difficulties must finish before baseline finalization")
    suite_keys = {
        str(_load_dataset(difficulty).manifest.get("suite_key") or "")
        for difficulty in DIFFICULTIES
    }
    suite_keys.discard("")
    if len(suite_keys) > 1:
        raise BaselineRunnerError("dataset suite_key values do not match")
    payload = {
        "schema_version": BASELINES_SCHEMA,
        "status": "succeeded",
        "suite_key": next(iter(suite_keys), "three-difficulty-live-2026-08-10"),
        "mode": "live",
        "base_url": index["base_url"],
        "kb_id": index["kb_id"],
        "generated_at": _utc_now(),
        "difficulty_order": list(DIFFICULTIES),
        "baselines": records,
    }
    _write_json(BASELINES_PATH, payload)
    index.update(
        {
            "status": "succeeded",
            "baselines_path": _relative(BASELINES_PATH),
            "baselines_sha256": _sha256_file(BASELINES_PATH),
        }
    )
    _persist_run_index(index)
    return payload


def _checksum_files() -> list[Path]:
    fixed = [
        SCRIPT_PATH,
        SUITE_ROOT / "suite.json",
        DATASET_DIR / "easy.json",
        DATASET_DIR / "medium.json",
        DATASET_DIR / "hard.json",
        BASELINES_PATH,
        RUN_INDEX_PATH,
    ]
    candidates = [path for path in fixed if path.is_file()]
    for directory in (RUNS_DIR, API_SNAPSHOT_ROOT, BASELINES_DIR / "history"):
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    unique = {path.resolve(): path for path in candidates if path.resolve() != CHECKSUMS_PATH.resolve()}
    return sorted(unique.values(), key=_relative)


def _write_checksums() -> None:
    lines = [f"{_sha256_file(path)}  {_relative(path)}" for path in _checksum_files()]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    _write_bytes(CHECKSUMS_PATH, content)


def _parse_args(config: dict[str, str]) -> argparse.Namespace:
    default_base_url = _first_config(
        config,
        "STT_EVAL_BASE_URL",
        "STT_QUERY_BASE_URL",
        default=DEFAULT_BASE_URL,
    )
    default_kb_id = _first_config(config, "STT_EVAL_KB_ID", "EVALUATION_KB_ID")
    default_poll_timeout = _first_config(
        config,
        "STT_EVAL_POLL_TIMEOUT",
        default=str(DEFAULT_POLL_TIMEOUT_SECONDS),
    )
    default_poll_interval = _first_config(
        config,
        "STT_EVAL_POLL_INTERVAL",
        default=str(DEFAULT_POLL_INTERVAL_SECONDS),
    )
    default_request_timeout = _first_config(
        config,
        "STT_EVAL_REQUEST_TIMEOUT",
        default=str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )
    parser = argparse.ArgumentParser(
        description="Run easy/medium/hard live baselines and archive verified artifacts."
    )
    parser.add_argument(
        "--base-url",
        default=default_base_url,
        help="local query API root (default: STT_EVAL_BASE_URL or http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--kb-id",
        default=default_kb_id,
        help="target knowledge-base id (or STT_EVAL_KB_ID)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=_positive_float,
        default=_positive_float(default_poll_timeout),
        metavar="SECONDS",
        help="maximum poll time for each run",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_float,
        default=_positive_float(default_poll_interval),
        metavar="SECONDS",
        help="poll interval (default: 10 seconds)",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_float,
        default=_positive_float(default_request_timeout),
        metavar="SECONDS",
        help="timeout for one HTTP request",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="resume run ids already recorded in run-index.json",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="start new baseline runs and archive prior index pointers",
    )
    args = parser.parse_args()
    args.base_url = _normalize_local_base_url(args.base_url)
    args.kb_id = _safe_component(str(args.kb_id or ""), label="knowledge-base id")
    return args


def main() -> int:
    config = _load_environment()
    args = _parse_args(config)
    quality_gate = _verify_quality_gate()
    email = _first_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_EMAIL",
        "AUTH_BOOTSTRAP_ADMIN_EMAIL",
        default="admin@stt.local",
    )
    password = _first_secret_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_PASSWORD",
        "AUTH_BOOTSTRAP_ADMIN_PASSWORD",
    )
    tenant_id = _first_config(
        config,
        "STT_EVAL_TENANT_ID",
        "AUTH_BOOTSTRAP_TENANT_ID",
    ) or None
    if not password:
        # Match the running application's validated development defaults when
        # the bootstrap credential was not repeated in the local .env file.
        # The resolved secret is used only in memory and is immediately added
        # to the snapshot redactor below.
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.shared.config.settings_config import settings

        email = settings.auth_bootstrap_admin_email
        password = settings.auth_bootstrap_admin_password
    if not password:
        raise BaselineRunnerError("bootstrap admin password could not be resolved")

    redactor = Redactor(password)
    index = _prepare_run_index(args)
    index["quality_gate"] = quality_gate
    prior_baselines = _read_json(BASELINES_PATH) if BASELINES_PATH.is_file() else None
    invocation_id = f"{_timestamp_slug()}-{uuid.uuid4().hex[:8]}"
    invocation = {
        "id": invocation_id,
        "started_at": _utc_now(),
        "status": "running",
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
    }
    index.setdefault("invocations", []).append(invocation)
    _persist_run_index(index)
    prior_baselines_succeeded = bool(
        args.resume
        and isinstance(prior_baselines, dict)
        and prior_baselines.get("status") == "succeeded"
    )
    if not prior_baselines_succeeded:
        _write_json(BASELINES_PATH, _baseline_status(index, status="in_progress"))

    recorder = ApiSnapshotRecorder(invocation_id, args.base_url, redactor)
    client = ApiClient(
        base_url=args.base_url,
        email=email,
        password=password,
        tenant_id=tenant_id,
        timeout_seconds=args.request_timeout,
        recorder=recorder,
        redactor=redactor,
    )
    workflow_error: Exception | None = None
    records: dict[str, dict[str, Any]] = {}
    try:
        login_payload = client.login()
        principal = login_payload.get("principal") if isinstance(login_payload, dict) else None
        if not isinstance(principal, dict):
            raise BaselineRunnerError("login response has no principal snapshot")
        print("Authenticated to the local query API; credentials are not logged.", flush=True)
        for difficulty in DIFFICULTIES:
            records[difficulty] = _process_difficulty(
                difficulty=difficulty,
                args=args,
                client=client,
                redactor=redactor,
                index=index,
            )
        _finalize_baselines(index, records)
    except Exception as exc:  # preserve partial state and redacted HTTP audit
        workflow_error = exc
        index["status"] = "failed"
        index["error_type"] = type(exc).__name__
        if not prior_baselines_succeeded:
            _write_json(BASELINES_PATH, _baseline_status(index, status="failed"))
    finally:
        try:
            client.logout()
        except Exception as logout_exc:
            print(
                f"Warning: logout failed ({type(logout_exc).__name__}); see API snapshots.",
                file=sys.stderr,
            )
        invocation["ended_at"] = _utc_now()
        invocation["status"] = "failed" if workflow_error else "succeeded"
        if workflow_error is not None:
            invocation["error_type"] = type(workflow_error).__name__
        snapshot_info = recorder.close(
            status=invocation["status"],
            error_type=invocation.get("error_type"),
        )
        invocation["api_snapshots"] = snapshot_info
        _persist_run_index(index)
        _write_checksums()

    if workflow_error is not None:
        safe_message = redactor.text(str(workflow_error))
        print(
            f"ERROR [{type(workflow_error).__name__}]: {safe_message}",
            file=sys.stderr,
        )
        return 1
    print(f"Baselines: {_relative(BASELINES_PATH)}", flush=True)
    print(f"Run index: {_relative(RUN_INDEX_PATH)}", flush=True)
    print(f"Checksums: {_relative(CHECKSUMS_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaselineRunnerError as exc:
        # Failures before the recorder exists are still deliberately concise;
        # no credential value is included in this exception type.
        print(f"ERROR [BaselineRunnerError]: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
