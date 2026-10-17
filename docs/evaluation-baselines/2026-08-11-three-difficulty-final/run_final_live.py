#!/usr/bin/env python3
"""Run one ordered, immutable live candidate root and archive its evidence.

The client deliberately executes exactly one difficulty per invocation.  A
new label must start with ``easy``; ``medium`` and ``hard`` require ``--resume``
and the preceding target.  Successful runs remain candidate roots.  Canonical
registration is a separate, local-only certification step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


SCRIPT_PATH = Path(__file__).absolute()
FINAL_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (FINAL_ROOT, *FINAL_ROOT.parents)
        if (candidate / "pyproject.toml").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:  # pragma: no cover - repository layout guard
    raise RuntimeError("cannot locate repository root")

SOURCE_RUNNER_PATH = FINAL_ROOT.parent / "2026-08-10-three-difficulty" / "run_live_baselines.py"


def _early_reparse_guard(path: Path, *, root: Path, label: str) -> None:
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    path_absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:  # pragma: no cover - repository layout guard
        raise RuntimeError(f"{label} escapes repository root") from exc
    for component in (Path(), *relative.parents[::-1], relative):
        candidate = root_absolute if component == Path() else root_absolute / component
        if candidate.exists() or candidate.is_symlink():
            attributes = int(getattr(os.lstat(candidate), "st_file_attributes", 0))
            if candidate.is_symlink() or bool(attributes & 0x400):
                raise RuntimeError(f"{label} contains a symlink or reparse point")
    try:
        path_absolute.resolve(strict=False).relative_to(root_absolute.resolve(strict=True))
    except ValueError as exc:  # pragma: no cover - repository layout guard
        raise RuntimeError(f"{label} resolves outside repository root") from exc


_early_reparse_guard(SCRIPT_PATH, root=PROJECT_ROOT, label="final live runner")
_early_reparse_guard(SOURCE_RUNNER_PATH, root=PROJECT_ROOT, label="shared live runner")


def _load_source_runner() -> Any:
    _early_reparse_guard(SOURCE_RUNNER_PATH, root=PROJECT_ROOT, label="shared live runner")
    if not SOURCE_RUNNER_PATH.is_file():
        raise RuntimeError(f"shared live client is missing: {SOURCE_RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "stt_shared_live_baseline_runner", SOURCE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError("cannot load shared live client")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _early_reparse_guard(SOURCE_RUNNER_PATH, root=PROJECT_ROOT, label="shared live runner")
    return module


br = _load_source_runner()

DIFFICULTIES = ("easy", "medium", "hard")
EXPECTED_SUITE_KEY = "three-difficulty-live-2026-08-11-reviewed-v2"
EXPECTED_DATASET_VERSION = "2.0.0-reviewed-v2.20260811"
EXPECTED_DATASET_KEYS = {
    difficulty: f"pantum-p3500-{difficulty}-live-reviewed-v2-2026-08-11"
    for difficulty in DIFFICULTIES
}
EXPECTED_CASE_COUNT = 50
TARGET_CORRECT_COUNTS = {"easy": 48, "medium": 43, "hard": None}
ACTIVATION_SCHEMA = "stt.eval.final-suite-activation.v1"
INDEX_SCHEMA = "stt.eval.final-candidate-run-index.v1"
CANDIDATE_REGISTRY_SCHEMA = "stt.eval.candidate-root-runs.v1"
CASE_CONTRACT = "stt.eval.case.v3"
AUTHORITATIVE_METRIC = "metrics.case_outcome.final_answer_correct"
QUALITY_TARGETS_SCHEMA = "stt.eval.quality-targets.v1"
EXPECTED_KB_ID = "kb_2fad25538aab42d19baf79fba782a06c"
EXPECTED_DOCUMENT_ID = "doc_9353bf25fcfa4991a8a0425713a3c9f1"
EXPECTED_DOCUMENT_VERSION_ID = "dv_8d7fe6750ff5445e8dfbfd80b913b216"
EXPECTED_INDEX_VERSION_ID = "idx_fb5eeac133f540f4af31004dbde23151"
EXPECTED_CHUNK_COUNT = 218
EXPECTED_CASE_PASS_POLICY: dict[str, Any] = {
    "policy_version": "stt.eval.case.v3",
    "quality_score_minimum": 0.75,
    "answer_correctness_minimum": 0.75,
    "faithfulness_minimum": 0.8,
    "citation_coverage_minimum": 0.8,
    "answer_correctness_weight": 0.7,
    "faithfulness_weight": 0.15,
    "citation_coverage_weight": 0.1,
    "retrieval_support_weight": 0.05,
    "required_branch_coverage_minimum": 1.0,
    "concept_group_coverage_minimum": 1.0,
    "unsupported_factual_claim_maximum": 0,
    "contradictory_factual_claim_maximum": 0,
    "abstention_factual_claim_maximum": 0,
}
STATIC_CHECKSUM_FILENAME = "SHA256SUMS"
STATIC_DYNAMIC_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    "runs",
    "baselines",
    "certification",
    "semantic-reviews",
    "service",
}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_RUN_STATUSES = {"queued", "running"}


class FinalLiveError(RuntimeError):
    """Fail-closed workflow error safe to show without credentials."""


class FinalRedactor(br.Redactor):
    """Extend the shared credential redactor for generic OAuth/session tokens."""

    _SENSITIVE_KEYS = br.Redactor._SENSITIVE_KEYS | {
        "token",
        "id_token",
        "session_token",
        "bearer_token",
    }


class LabelLock:
    """Hold one OS-released advisory lock for a candidate label."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.handle: Any | None = None
        self.path = FINAL_ROOT / "runs" / ".locks" / f"{label}.lock"

    def __enter__(self) -> LabelLock:
        runs_root = FINAL_ROOT / "runs"
        _assert_safe_path(
            runs_root,
            root=FINAL_ROOT,
            label="runs root",
            require_exists=False,
        )
        runs_root.mkdir(parents=True, exist_ok=True)
        _assert_safe_path(runs_root, root=FINAL_ROOT, label="runs root")
        lock_root = self.path.parent
        _assert_safe_path(
            lock_root,
            root=FINAL_ROOT,
            label="runner lock root",
            require_exists=False,
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        _assert_safe_path(lock_root, root=FINAL_ROOT, label="runner lock root")
        _assert_safe_path(
            self.path,
            root=FINAL_ROOT,
            label="runner lock file",
            require_exists=False,
        )
        handle = self.path.open("a+b")
        _assert_safe_path(self.path, root=FINAL_ROOT, label="runner lock file")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the target workspace
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise FinalLiveError(f"another runner owns label {self.label!r}") from exc
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - Windows is the target workspace
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass(frozen=True)
class DatasetSpec:
    difficulty: str
    path: Path
    file_sha256: str
    manifest: dict[str, Any]
    request_body: dict[str, Any]
    case_ids: tuple[str, ...]
    manifest_sha256: str

    @property
    def case_count(self) -> int:
        return len(self.case_ids)


@dataclass(frozen=True)
class SuiteSnapshot:
    activation: dict[str, Any]
    activation_path: Path
    activation_sha256: str
    suite_key: str
    suite_identity_sha256: str
    datasets: dict[str, DatasetSpec]
    quality_targets_path: Path
    quality_targets_sha256: str
    source_index_identity: dict[str, Any]
    static_checksums_sha256: str


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_safe_path(
    path: Path,
    *,
    root: Path,
    label: str,
    require_exists: bool = True,
) -> Path:
    """Reject lexical escapes and every existing symlink/junction component.

    Both lexical and resolved containment are checked.  Callers invoke this
    before and after I/O so a replaced parent cannot silently redirect an
    activation, manifest, receipt, archive, or checksum operation.
    """

    root_absolute = _path_absolute(root)
    path_absolute = _path_absolute(path)
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise FinalLiveError(f"{label} escapes its trusted root") from exc
    project_absolute = _path_absolute(PROJECT_ROOT)
    try:
        root_relative = root_absolute.relative_to(project_absolute)
    except ValueError:
        root_relative = None
    if root_relative is not None:
        for component in (Path(), *root_relative.parents[::-1], root_relative):
            candidate = project_absolute if component == Path() else project_absolute / component
            if candidate.exists() or candidate.is_symlink():
                if _is_reparse_point(candidate):
                    raise FinalLiveError(
                        f"{label} trusted-root chain contains a symlink or reparse point: "
                        f"{candidate}"
                    )
        try:
            root_absolute.resolve(strict=True).relative_to(project_absolute.resolve(strict=True))
        except ValueError as exc:
            raise FinalLiveError(f"{label} trusted root resolves outside repository") from exc
    current = root_absolute
    for component in (Path(), *relative.parents[::-1], relative):
        candidate = current if component == Path() else root_absolute / component
        if candidate.exists() or candidate.is_symlink():
            if _is_reparse_point(candidate):
                raise FinalLiveError(f"{label} contains a symlink or reparse point: {candidate}")
    resolved_root = root_absolute.resolve(strict=True)
    resolved_path = path_absolute.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise FinalLiveError(f"{label} resolves outside its trusted root") from exc
    if require_exists and not path_absolute.exists():
        raise FinalLiveError(f"{label} is missing: {path_absolute}")
    return path_absolute


def _sha256_file(path: Path, *, root: Path | None = None, label: str = "file") -> str:
    trusted_root = root or PROJECT_ROOT
    safe_path = _assert_safe_path(path, root=trusted_root, label=label)
    if not safe_path.is_file():
        raise FinalLiveError(f"{label} is not a regular file: {safe_path}")
    digest = hashlib.sha256()
    with safe_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    _assert_safe_path(safe_path, root=trusted_root, label=label)
    return digest.hexdigest()


def _read_json(path: Path, *, root: Path | None = None, label: str = "JSON file") -> Any:
    trusted_root = root or PROJECT_ROOT
    safe_path = _assert_safe_path(path, root=trusted_root, label=label)
    if not safe_path.is_file():
        raise FinalLiveError(f"{label} is not a regular file: {safe_path}")
    try:
        value = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalLiveError(f"cannot read valid JSON: {safe_path}") from exc
    _assert_safe_path(safe_path, root=trusted_root, label=label)
    return value


def _write_json(path: Path, value: Any) -> None:
    _write_bytes_safe(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        root=FINAL_ROOT,
        label="runner JSON output",
        overwrite=True,
    )


def _write_bytes_safe(
    path: Path,
    content: bytes,
    *,
    root: Path,
    label: str,
    overwrite: bool,
) -> None:
    safe_path = _assert_safe_path(
        path,
        root=root,
        label=label,
        require_exists=False,
    )
    parent = safe_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(parent, root=root, label=f"{label} parent")
    if safe_path.exists() and not overwrite:
        raise FinalLiveError(f"refusing to overwrite {label}: {safe_path}")
    temporary = safe_path.with_name(f".{safe_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    _assert_safe_path(temporary, root=root, label=f"{label} temporary", require_exists=False)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_safe_path(temporary, root=root, label=f"{label} temporary")
        _assert_safe_path(parent, root=root, label=f"{label} parent")
        os.replace(temporary, safe_path)
        _assert_safe_path(safe_path, root=root, label=label)
        if safe_path.read_bytes() != content:
            raise FinalLiveError(f"{label} post-write verification failed")
        _assert_safe_path(safe_path, root=root, label=label)
    finally:
        if temporary.exists() and not _is_reparse_point(temporary):
            temporary.unlink()


def _expected_quality_targets() -> dict[str, Any]:
    return {
        "schema_version": QUALITY_TARGETS_SCHEMA,
        "suite_key": EXPECTED_SUITE_KEY,
        "primary_metric": {
            "name": "final_visible_answer_binary_accuracy",
            "production_path": "metrics.case_outcome.final_answer_accuracy",
            "description": "最终呈现给用户的答案按题二元判定正确或错误，再按难度汇总正确率。",
        },
        "independent_blind_review": {
            "role": "audit_only",
            "counts_toward_target": False,
            "description": "独立盲审仅用于审计生产评测结论，不替代或混入生产正确率。",
        },
        "targets": {
            "easy": {
                "required": True,
                "operator": ">",
                "threshold": 0.95,
                "case_count": 50,
                "minimum_correct_cases": 48,
            },
            "medium": {
                "required": True,
                "operator": ">",
                "threshold": 0.85,
                "case_count": 50,
                "minimum_correct_cases": 43,
            },
            "hard": {
                "required": False,
                "operator": None,
                "threshold": None,
                "case_count": 50,
                "minimum_correct_cases": None,
            },
        },
    }


def _static_files(root: Path) -> list[Path]:
    root = _assert_safe_path(root, root=root, label="static suite root")
    files: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        _assert_safe_path(directory, root=root, label="static suite directory")
        for child in directory.iterdir():
            relative = child.relative_to(root)
            if any(part in STATIC_DYNAMIC_DIRECTORIES for part in relative.parts):
                continue
            if relative.as_posix() == STATIC_CHECKSUM_FILENAME:
                continue
            _assert_safe_path(child, root=root, label="static suite artifact")
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                files.append(child)
            else:
                raise FinalLiveError(f"unsupported static suite artifact: {child}")
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _tree_files(root: Path, *, trusted_root: Path, label: str) -> list[Path]:
    safe_root = _assert_safe_path(root, root=trusted_root, label=label)
    files: list[Path] = []
    stack = [safe_root]
    while stack:
        directory = stack.pop()
        _assert_safe_path(directory, root=trusted_root, label=label)
        for child in directory.iterdir():
            _assert_safe_path(child, root=trusted_root, label=label)
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                files.append(child)
            else:
                raise FinalLiveError(f"{label} contains an unsupported filesystem entry")
    return sorted(files, key=lambda item: item.relative_to(safe_root).as_posix())


def _write_static_checksums(root: Path = FINAL_ROOT) -> Path:
    files = _static_files(root)
    content = "".join(
        f"{_sha256_file(path, root=root, label='static suite artifact')}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in files
    ).encode("utf-8")
    destination = root / STATIC_CHECKSUM_FILENAME
    _write_bytes_safe(
        destination,
        content,
        root=root,
        label="static SHA256SUMS",
        overwrite=True,
    )
    return destination


def _validate_static_checksums(root: Path) -> str:
    checksum_path = root / STATIC_CHECKSUM_FILENAME
    safe_path = _assert_safe_path(
        checksum_path,
        root=root,
        label="static SHA256SUMS",
    )
    rows: dict[str, str] = {}
    for line in safe_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if match is None:
            raise FinalLiveError("static SHA256SUMS has an invalid row")
        digest, relative = match.groups()
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in rows:
            raise FinalLiveError("static SHA256SUMS has an unsafe or duplicate path")
        rows[relative] = digest
    expected_files = _static_files(root)
    expected_relatives = {path.relative_to(root).as_posix() for path in expected_files}
    if set(rows) != expected_relatives:
        raise FinalLiveError("static SHA256SUMS does not exactly cover the static suite")
    for path in expected_files:
        relative = path.relative_to(root).as_posix()
        if _sha256_file(path, root=root, label=f"static artifact {relative}") != rows[relative]:
            raise FinalLiveError(f"static SHA256SUMS mismatch: {relative}")
    _assert_safe_path(safe_path, root=root, label="static SHA256SUMS")
    return _sha256_file(safe_path, root=root, label="static SHA256SUMS")


def _relative_final(path: Path) -> str:
    try:
        return path.resolve().relative_to(FINAL_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalLiveError(f"path escapes final suite root: {path}") from exc


def _relative_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FinalLiveError(f"path escapes repository root: {path}") from exc


def _safe_label(value: str) -> str:
    normalized = str(value or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", normalized):
        raise argparse.ArgumentTypeError("label must match [a-z0-9][a-z0-9._-]{0,63}")
    return normalized


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    return path.is_symlink() or bool(attributes & 0x400)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _manifest_from_raw(
    raw: Mapping[str, Any], difficulty: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    embedded = raw.get("manifest")
    if isinstance(embedded, Mapping):
        manifest = dict(embedded)
        request: dict[str, Any] = {"manifest": manifest}
        if raw.get("parent_dataset_id") is not None:
            request["parent_dataset_id"] = raw["parent_dataset_id"]
        return manifest, request
    return dict(raw), {"manifest": dict(raw)}


def _resolve_frozen_input_path(root: Path, relative: Any, *, label: str) -> tuple[Path, Path]:
    value = str(relative or "")
    candidate_relative = Path(value)
    if (
        not value
        or candidate_relative.is_absolute()
        or ".." in candidate_relative.parts
        or "\\" in value
    ):
        raise FinalLiveError(f"{label} must be a safe relative path")
    matches: list[tuple[Path, Path]] = []
    for base in (root, PROJECT_ROOT):
        candidate = base / candidate_relative
        if candidate.exists() or candidate.is_symlink():
            matches.append(
                (
                    _assert_safe_path(candidate, root=base, label=label),
                    base,
                )
            )
    unique = {(path.resolve(), base.resolve()): (path, base) for path, base in matches}
    if not unique:
        raise FinalLiveError(f"{label} is missing")
    resolved_files = {item[0].resolve() for item in unique.values()}
    if len(resolved_files) != 1:
        raise FinalLiveError(f"{label} is ambiguous between trusted roots")
    return next(iter(unique.values()))


def _validate_derivation(
    root: Path,
    manifest: Mapping[str, Any],
    activation_entry: Mapping[str, Any],
    *,
    difficulty: str,
) -> None:
    derivation = manifest.get("activation_derivation")
    input_receipt = activation_entry.get("input")
    if not isinstance(derivation, Mapping) or not isinstance(input_receipt, Mapping):
        raise FinalLiveError(f"{difficulty} activation derivation receipt is required")
    receipt_without_hash = {
        key: value for key, value in derivation.items() if key != "receipt_sha256"
    }
    receipt_sha256 = _sha256_json(receipt_without_hash)
    expected_derivation = {
        "schema_version": "stt.eval.dataset-candidate-activation-derivation.v1",
        "output_state": "candidate_active",
        "rules_version": "stt.eval.reviewed-suite-activation.v1",
    }
    if any(derivation.get(key) != value for key, value in expected_derivation.items()):
        raise FinalLiveError(f"{difficulty} activation derivation state/version drifted")
    if (
        derivation.get("receipt_sha256") != receipt_sha256
        or activation_entry.get("derivation_receipt_sha256") != receipt_sha256
    ):
        raise FinalLiveError(f"{difficulty} activation derivation receipt SHA drifted")
    expected_input = {
        "path": derivation.get("input_path"),
        "file_sha256": derivation.get("input_file_sha256"),
        "manifest_sha256": derivation.get("input_manifest_sha256"),
    }
    if dict(input_receipt) != expected_input:
        raise FinalLiveError(f"{difficulty} activation input receipt drifted")
    input_path, input_root = _resolve_frozen_input_path(
        root,
        expected_input["path"],
        label=f"{difficulty} reviewed activation input",
    )
    if (
        _sha256_file(
            input_path,
            root=input_root,
            label=f"{difficulty} reviewed activation input",
        )
        != expected_input["file_sha256"]
    ):
        raise FinalLiveError(f"{difficulty} reviewed activation input file SHA drifted")
    input_manifest = _read_json(
        input_path,
        root=input_root,
        label=f"{difficulty} reviewed activation input",
    )
    if _sha256_json(input_manifest) != expected_input["manifest_sha256"]:
        raise FinalLiveError(f"{difficulty} reviewed activation input manifest SHA drifted")


def _load_dataset(
    root: Path,
    difficulty: str,
    activation_entry: Mapping[str, Any],
    source_index_identity: Mapping[str, Any],
) -> DatasetSpec:
    expected_relative = f"datasets/{difficulty}.json"
    if activation_entry.get("path") != expected_relative:
        raise FinalLiveError(f"activation dataset path must be {expected_relative}")
    path = _assert_safe_path(
        root / expected_relative,
        root=root,
        label=f"{difficulty} dataset",
    )
    if not path.is_file():
        raise FinalLiveError(f"missing dataset: {expected_relative}")
    file_sha256 = _sha256_file(path, root=root, label=f"{difficulty} dataset")
    if activation_entry.get("sha256") != file_sha256:
        raise FinalLiveError(f"activation SHA does not match {expected_relative}")
    raw = _read_json(path, root=root, label=f"{difficulty} dataset")
    if not isinstance(raw, Mapping):
        raise FinalLiveError(f"{expected_relative} must contain an object")
    manifest, request_body = _manifest_from_raw(raw, difficulty)
    for required in ("schema_version", "suite_key", "dataset_key", "name", "version", "source"):
        if not manifest.get(required):
            raise FinalLiveError(f"{difficulty} manifest is missing required field {required}")
    if manifest.get("suite_key") != EXPECTED_SUITE_KEY:
        raise FinalLiveError(f"{difficulty} suite_key does not match the frozen final suite")
    if manifest.get("dataset_key") != EXPECTED_DATASET_KEYS[difficulty]:
        raise FinalLiveError(f"{difficulty} dataset_key does not match the frozen identity")
    if manifest.get("version") != EXPECTED_DATASET_VERSION:
        raise FinalLiveError(f"{difficulty} version does not match the frozen identity")
    if str(manifest.get("difficulty") or "").lower() != difficulty:
        raise FinalLiveError(f"{difficulty} manifest has the wrong difficulty")
    if str(manifest.get("evaluation_mode_default") or "live") != "live":
        raise FinalLiveError(f"{difficulty} evaluation_mode_default must be live")
    expected_state = {
        "status": "candidate_active",
        "import_allowed": True,
        "candidate_run_allowed": True,
        "baseline_registration_allowed": False,
    }
    if any(manifest.get(key) != value for key, value in expected_state.items()):
        raise FinalLiveError(f"{difficulty} manifest is not internally candidate-active")
    generation = manifest.get("generation")
    gate_policy = manifest.get("gate_policy")
    if (
        not isinstance(generation, Mapping)
        or generation.get("independent_validation_status") != "passed"
        or not isinstance(gate_policy, Mapping)
        or gate_policy.get("case_pass_policy") != EXPECTED_CASE_PASS_POLICY
        or gate_policy.get("case_pass_policy_activation_status") != "active_candidate"
        or gate_policy.get("case_pass_policy_materialization_required") is not False
    ):
        raise FinalLiveError(f"{difficulty} manifest policy/validation activation state drifted")
    source = manifest.get("source")
    expected_source = {
        "relative_path": source_index_identity["source_path"],
        "content_sha256": source_index_identity["source_sha256"],
        "page_count": source_index_identity["source_page_count"],
        "kb_id": source_index_identity["kb_id"],
        "document_id": source_index_identity["document_id"],
        "document_version_id": source_index_identity["document_version_id"],
        "index_version_id": source_index_identity["index_version_id"],
    }
    if not isinstance(source, Mapping) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise FinalLiveError(f"{difficulty} manifest source/index identity drifted")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise FinalLiveError(
            f"{difficulty} must contain exactly {EXPECTED_CASE_COUNT} cases for the 150-case audit"
        )
    case_ids: list[str] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise FinalLiveError(f"{difficulty} case {position} is not an object")
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise FinalLiveError(f"{difficulty} case {position} has no case_id")
        groups = row.get("concept_groups")
        all_of = groups.get("all_of") if isinstance(groups, Mapping) else None
        if not isinstance(all_of, list) or not all_of:
            raise FinalLiveError(f"{difficulty} case {case_id} has no concept_groups.all_of")
        for group_index, group in enumerate(all_of, start=1):
            variants = group.get("any_of") if isinstance(group, Mapping) else None
            if (
                not isinstance(variants, list)
                or not variants
                or any(not isinstance(value, str) or not value.strip() for value in variants)
            ):
                raise FinalLiveError(
                    f"{difficulty} case {case_id} concept_groups.all_of[{group_index}] is invalid"
                )
        answer_contract = row.get("answer_contract")
        review = answer_contract.get("review") if isinstance(answer_contract, Mapping) else None
        if (
            not isinstance(answer_contract, Mapping)
            or answer_contract.get("contract_status") != "llm_reviewed_candidate_active"
            or not isinstance(review, Mapping)
            or review.get("independent_validation_status") != "passed"
        ):
            raise FinalLiveError(f"{difficulty} case {case_id} is not candidate-active")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise FinalLiveError(f"{difficulty} case_id values are not unique")
    if activation_entry.get("case_count") != EXPECTED_CASE_COUNT:
        raise FinalLiveError(f"activation case_count for {difficulty} must be 50")
    if activation_entry.get("dataset_key") != EXPECTED_DATASET_KEYS[difficulty]:
        raise FinalLiveError(f"activation dataset_key for {difficulty} does not match")
    if activation_entry.get("version") != EXPECTED_DATASET_VERSION:
        raise FinalLiveError(f"activation version for {difficulty} does not match")
    manifest_sha256 = _sha256_json(manifest)
    if activation_entry.get("manifest_sha256") != manifest_sha256:
        raise FinalLiveError(f"activation canonical manifest SHA differs for {difficulty}")
    _validate_derivation(root, manifest, activation_entry, difficulty=difficulty)
    return DatasetSpec(
        difficulty=difficulty,
        path=path,
        file_sha256=file_sha256,
        manifest=manifest,
        request_body=request_body,
        case_ids=tuple(case_ids),
        manifest_sha256=manifest_sha256,
    )


def _load_source_index_identity(
    root: Path,
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    integrity = activation.get("source_and_index_integrity")
    source = integrity.get("source") if isinstance(integrity, Mapping) else None
    active_index = integrity.get("active_index") if isinstance(integrity, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(active_index, Mapping):
        raise FinalLiveError("activation source/index integrity receipt is required")
    identity = {
        "source_path": str(source.get("path") or ""),
        "source_sha256": str(source.get("sha256") or ""),
        "source_page_count": source.get("page_count"),
        "kb_id": active_index.get("kb_id"),
        "document_id": active_index.get("document_id"),
        "document_version_id": active_index.get("document_version_id"),
        "index_version_id": active_index.get("index_version_id"),
        "frozen_chunk_count": active_index.get("frozen_chunk_count"),
    }
    expected = {
        "kb_id": EXPECTED_KB_ID,
        "document_id": EXPECTED_DOCUMENT_ID,
        "document_version_id": EXPECTED_DOCUMENT_VERSION_ID,
        "index_version_id": EXPECTED_INDEX_VERSION_ID,
        "frozen_chunk_count": EXPECTED_CHUNK_COUNT,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise FinalLiveError("activation source/index identity differs from the frozen suite")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", identity["source_sha256"])
        or identity["source_page_count"] != 95
    ):
        raise FinalLiveError("activation source receipt is incomplete")
    source_path, source_root = _resolve_frozen_input_path(
        root,
        identity["source_path"],
        label="frozen source PDF",
    )
    if (
        _sha256_file(source_path, root=source_root, label="frozen source PDF")
        != identity["source_sha256"]
    ):
        raise FinalLiveError("frozen source PDF SHA drifted")

    source_receipt_path, source_receipt_root = _resolve_frozen_input_path(
        root,
        active_index.get("source_qa_summary_path"),
        label="source QA receipt",
    )
    blocker_path, blocker_root = _resolve_frozen_input_path(
        root,
        active_index.get("blocker_validation_path"),
        label="blocker/index receipt",
    )
    if _sha256_file(
        source_receipt_path, root=source_receipt_root, label="source QA receipt"
    ) != active_index.get("source_qa_summary_sha256") or _sha256_file(
        blocker_path, root=blocker_root, label="blocker/index receipt"
    ) != active_index.get("blocker_validation_sha256"):
        raise FinalLiveError("source/index receipt SHA drifted")
    source_receipt = _read_json(
        source_receipt_path,
        root=source_receipt_root,
        label="source QA receipt",
    )
    blocker_receipt = _read_json(
        blocker_path,
        root=blocker_root,
        label="blocker/index receipt",
    )
    checks = source_receipt.get("input_checks") if isinstance(source_receipt, Mapping) else None
    blocker_index = (
        blocker_receipt.get("active_index") if isinstance(blocker_receipt, Mapping) else None
    )
    if (
        not isinstance(checks, Mapping)
        or checks.get("source_path") != identity["source_path"]
        or checks.get("source_sha256") != identity["source_sha256"]
        or checks.get("source_hash_matches") is not True
        or checks.get("source_page_count_matches") is not True
        or checks.get("source_pdf_page_count") != identity["source_page_count"]
        or checks.get("joined_case_count") != 150
        or checks.get("indexed_evidence_count") != EXPECTED_CHUNK_COUNT
        or checks.get("index_version_id") != identity["index_version_id"]
        or not isinstance(blocker_index, Mapping)
        or blocker_receipt.get("status") != "passed"
        or blocker_receipt.get("selected_chunks_resolve_to_frozen_active_index") is not True
        or blocker_index.get("index_version_id") != identity["index_version_id"]
        or blocker_index.get("frozen_chunk_count") != EXPECTED_CHUNK_COUNT
        or blocker_index.get("index_rebuild_required") is not False
        or blocker_index.get("evidence_refresh_required") is not False
    ):
        raise FinalLiveError("source/index receipt content is not frozen or complete")
    return identity


def load_suite_snapshot(root: Path = FINAL_ROOT) -> SuiteSnapshot:
    root = _assert_safe_path(root, root=root, label="final suite root")
    activation_path = _assert_safe_path(
        root / "activation.json",
        root=root,
        label="activation.json",
    )
    activation = _read_json(activation_path, root=root, label="activation.json")
    if not isinstance(activation, dict):
        raise FinalLiveError("activation.json must contain an object")
    if activation.get("schema_version") != ACTIVATION_SCHEMA:
        raise FinalLiveError("activation.json has an unsupported schema_version")
    if activation.get("suite_key") != EXPECTED_SUITE_KEY:
        raise FinalLiveError("activation suite_key does not match the frozen final suite")
    if activation.get("import_allowed") is not True:
        raise FinalLiveError("activation import_allowed must be true")
    if activation.get("candidate_run_allowed") is not True:
        raise FinalLiveError("activation candidate_run_allowed must be true")
    # Registration is deliberately authorized by a later immutable certification
    # file.  Mutating activation after a live run would invalidate its identity.
    if activation.get("baseline_registration_allowed") is not False:
        raise FinalLiveError("candidate activation baseline_registration_allowed must remain false")
    contract = activation.get("evaluation_contract")
    if not isinstance(contract, Mapping):
        raise FinalLiveError("activation evaluation_contract is required")
    if contract.get("case_schema_version") != CASE_CONTRACT:
        raise FinalLiveError(f"evaluation contract must be {CASE_CONTRACT}")
    if contract.get("authoritative_metric") != AUTHORITATIVE_METRIC:
        raise FinalLiveError("activation authoritative metric does not match")
    expected_targets = {
        "easy": {"correct_count": 48, "case_count": 50},
        "medium": {"correct_count": 43, "case_count": 50},
        "hard": {"correct_count": None, "case_count": 50},
    }
    if contract.get("target_counts") != expected_targets:
        raise FinalLiveError("activation target_counts do not match the frozen contract")
    if contract.get("case_pass_policy") != EXPECTED_CASE_PASS_POLICY:
        raise FinalLiveError(
            "activation case_pass_policy does not match the exact frozen v3 policy"
        )
    quality_receipt = activation.get("quality_targets")
    if not isinstance(quality_receipt, Mapping) or quality_receipt.get("path") != (
        "quality-targets.json"
    ):
        raise FinalLiveError("activation canonical quality-target receipt is required")
    quality_path = _assert_safe_path(
        root / "quality-targets.json",
        root=root,
        label="canonical quality-target contract",
    )
    quality_sha256 = _sha256_file(
        quality_path,
        root=root,
        label="canonical quality-target contract",
    )
    quality_targets = _read_json(
        quality_path,
        root=root,
        label="canonical quality-target contract",
    )
    if (
        quality_receipt.get("schema_version") != QUALITY_TARGETS_SCHEMA
        or quality_receipt.get("sha256") != quality_sha256
        or quality_targets != _expected_quality_targets()
    ):
        raise FinalLiveError("canonical quality-target schema, content, or SHA drifted")
    source_index_identity = _load_source_index_identity(root, activation)
    entries = activation.get("datasets")
    if not isinstance(entries, Mapping) or set(entries) != set(DIFFICULTIES):
        raise FinalLiveError("activation datasets must contain easy, medium, and hard")
    datasets: dict[str, DatasetSpec] = {}
    for difficulty in DIFFICULTIES:
        entry = entries.get(difficulty)
        if not isinstance(entry, Mapping):
            raise FinalLiveError(f"activation dataset entry is invalid: {difficulty}")
        datasets[difficulty] = _load_dataset(
            root,
            difficulty,
            entry,
            source_index_identity,
        )
    all_case_ids = [
        case_id for difficulty in DIFFICULTIES for case_id in datasets[difficulty].case_ids
    ]
    if len(all_case_ids) != 150 or len(set(all_case_ids)) != 150:
        raise FinalLiveError("case_id values must be globally unique across all 3x50 cases")
    static_checksums_sha256 = _validate_static_checksums(root)
    identity = {
        "activation_sha256": _sha256_file(
            activation_path,
            root=root,
            label="activation.json",
        ),
        "suite_key": EXPECTED_SUITE_KEY,
        "dataset_files": {
            difficulty: datasets[difficulty].file_sha256 for difficulty in DIFFICULTIES
        },
        "dataset_keys": EXPECTED_DATASET_KEYS,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "case_contract": CASE_CONTRACT,
        "authoritative_metric": AUTHORITATIVE_METRIC,
        "quality_targets_sha256": quality_sha256,
        "source_index_identity": source_index_identity,
        "static_checksums_sha256": static_checksums_sha256,
    }
    return SuiteSnapshot(
        activation=activation,
        activation_path=activation_path,
        activation_sha256=identity["activation_sha256"],
        suite_key=EXPECTED_SUITE_KEY,
        suite_identity_sha256=_sha256_json(identity),
        datasets=datasets,
        quality_targets_path=quality_path,
        quality_targets_sha256=quality_sha256,
        source_index_identity=source_index_identity,
        static_checksums_sha256=static_checksums_sha256,
    )


def _current_code_revision() -> str:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.application.evaluation_service import _code_revision

        value = str(_code_revision() or "").strip()
    finally:
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:  # pragma: no cover - path already present/removed
            pass
    if not value:
        raise FinalLiveError("cannot freeze the local evaluation code revision")
    return value


def _local_runtime_identity() -> dict[str, str]:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.shared.config.settings_config import settings

        value = {
            "config_version": str(settings.config_version),
            "llm_model": str(settings.llm_default_model),
            "embedding_model": str(settings.bge_m3_path or settings.bge_m3),
            "reranker_model": str(settings.bge_reranker_large),
        }
    finally:
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:  # pragma: no cover - path already present/removed
            pass
    if value["llm_model"] != "qwen3.7-max":
        raise FinalLiveError("final live candidate requires llm_model=qwen3.7-max")
    if any(not item for item in value.values()):
        raise FinalLiveError("cannot freeze the complete local runtime identity")
    return value


def _new_index(
    args: argparse.Namespace, suite: SuiteSnapshot, code_revision: str
) -> dict[str, Any]:
    now = br._utc_now()
    return {
        "schema_version": INDEX_SCHEMA,
        "label": args.label,
        "role": "candidate_root_run",
        "canonical": False,
        "suite_key": suite.suite_key,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "mode": "live",
        "baseline_run_id": None,
        "selection": "all",
        "planned_difficulties": list(DIFFICULTIES),
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "code_revision": code_revision,
        "runtime_identity": dict(args.runtime_identity),
        "runner": {
            "path": _relative_project(SCRIPT_PATH),
            "sha256": _sha256_file(SCRIPT_PATH, root=PROJECT_ROOT, label="final live runner"),
            "shared_client_path": _relative_project(SOURCE_RUNNER_PATH),
            "shared_client_sha256": _sha256_file(
                SOURCE_RUNNER_PATH,
                root=PROJECT_ROOT,
                label="shared live runner",
            ),
        },
        "activation": {
            "path": _relative_final(suite.activation_path),
            "sha256": suite.activation_sha256,
            "import_allowed": True,
            "candidate_run_allowed": True,
            "baseline_registration_allowed": False,
        },
        "quality_targets": {
            "path": _relative_final(suite.quality_targets_path),
            "sha256": suite.quality_targets_sha256,
            "schema_version": QUALITY_TARGETS_SCHEMA,
        },
        "source_index_identity": dict(suite.source_index_identity),
        "static_checksums_sha256": suite.static_checksums_sha256,
        "dataset_files": {
            difficulty: {
                "path": _relative_final(suite.datasets[difficulty].path),
                "sha256": suite.datasets[difficulty].file_sha256,
                "manifest_sha256": suite.datasets[difficulty].manifest_sha256,
                "case_count": suite.datasets[difficulty].case_count,
                "dataset_key": suite.datasets[difficulty].manifest["dataset_key"],
                "version": suite.datasets[difficulty].manifest["version"],
            }
            for difficulty in DIFFICULTIES
        },
        "target_contract": {
            "authoritative_metric": AUTHORITATIVE_METRIC,
            "strict_metrics_count_toward_target": False,
            "blind_audit_counts_toward_target": False,
            "correct_counts": TARGET_CORRECT_COUNTS,
            "case_count": EXPECTED_CASE_COUNT,
        },
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "invocations": [],
        "difficulties": {},
    }


def _persist_index(index_path: Path, index: dict[str, Any]) -> None:
    index["updated_at"] = br._utc_now()
    _write_json(index_path, index)


def _validate_frozen_index(
    index: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    code_revision: str,
) -> None:
    expected = {
        "schema_version": INDEX_SCHEMA,
        "label": args.label,
        "role": "candidate_root_run",
        "canonical": False,
        "suite_key": suite.suite_key,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "mode": "live",
        "baseline_run_id": None,
        "selection": "all",
        "planned_difficulties": list(DIFFICULTIES),
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
        "code_revision": code_revision,
        "runtime_identity": dict(args.runtime_identity),
    }
    changed = [key for key, value in expected.items() if index.get(key) != value]
    runner = index.get("runner")
    if not isinstance(runner, Mapping):
        changed.append("runner")
    else:
        if runner.get("sha256") != _sha256_file(
            SCRIPT_PATH,
            root=PROJECT_ROOT,
            label="final live runner",
        ):
            changed.append("runner.sha256")
        if runner.get("shared_client_sha256") != _sha256_file(
            SOURCE_RUNNER_PATH,
            root=PROJECT_ROOT,
            label="shared live runner",
        ):
            changed.append("runner.shared_client_sha256")
    activation = index.get("activation")
    if not isinstance(activation, Mapping) or activation.get("sha256") != suite.activation_sha256:
        changed.append("activation.sha256")
    quality = index.get("quality_targets")
    if not isinstance(quality, Mapping) or quality != {
        "path": _relative_final(suite.quality_targets_path),
        "sha256": suite.quality_targets_sha256,
        "schema_version": QUALITY_TARGETS_SCHEMA,
    }:
        changed.append("quality_targets")
    if index.get("source_index_identity") != suite.source_index_identity:
        changed.append("source_index_identity")
    if index.get("static_checksums_sha256") != suite.static_checksums_sha256:
        changed.append("static_checksums_sha256")
    files = index.get("dataset_files")
    if not isinstance(files, Mapping):
        changed.append("dataset_files")
    else:
        for difficulty in DIFFICULTIES:
            item = files.get(difficulty)
            if (
                not isinstance(item, Mapping)
                or item.get("sha256") != suite.datasets[difficulty].file_sha256
                or item.get("manifest_sha256") != suite.datasets[difficulty].manifest_sha256
            ):
                changed.append(f"dataset_files.{difficulty}.sha256")
    if changed:
        raise FinalLiveError(
            "refusing to continue because frozen label identity changed: "
            + ", ".join(sorted(set(changed)))
        )


def _prepare_label(
    args: argparse.Namespace, suite: SuiteSnapshot, code_revision: str
) -> tuple[Path, Path, dict[str, Any]]:
    runs_root = FINAL_ROOT / "runs"
    _assert_safe_path(
        runs_root,
        root=FINAL_ROOT,
        label="runs root",
        require_exists=False,
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(runs_root, root=FINAL_ROOT, label="runs root")
    label_root = runs_root / args.label
    _assert_safe_path(
        label_root,
        root=runs_root,
        label="candidate label root",
        require_exists=False,
    )
    index_path = label_root / "run-index.json"
    if args.resume:
        if not label_root.is_dir() or not index_path.is_file():
            raise FinalLiveError("--resume requires runs/<label>/run-index.json")
        _tree_files(label_root, trusted_root=FINAL_ROOT, label="candidate label")
        index = _read_json(index_path, root=FINAL_ROOT, label="candidate run index")
        if not isinstance(index, dict):
            raise FinalLiveError("run-index.json must contain an object")
        _validate_frozen_index(
            index,
            args=args,
            suite=suite,
            code_revision=code_revision,
        )
        index["status"] = "in_progress"
        return label_root, index_path, index
    if label_root.exists():
        raise FinalLiveError("label already exists; use --resume or choose a new label")
    if args.difficulty != "easy":
        raise FinalLiveError("a new label must start with easy")
    label_root.mkdir(parents=False, exist_ok=False)
    _assert_safe_path(label_root, root=FINAL_ROOT, label="candidate label root")
    index = _new_index(args, suite, code_revision)
    _persist_index(index_path, index)
    return label_root, index_path, index


def _target_count_from_rows(
    rows: Sequence[Any], *, expected_run_id: str, expected_case_ids: Sequence[str]
) -> dict[str, Any]:
    if len(rows) != len(expected_case_ids):
        raise FinalLiveError("result count does not match the frozen 50-case selection")
    actual_ids: list[str] = []
    values: list[bool] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise FinalLiveError(f"result row {position} is not an object")
        if row.get("run_id") != expected_run_id:
            raise FinalLiveError(f"result row {position} has the wrong run id")
        case_id = str(row.get("case_id") or "")
        actual_ids.append(case_id)
        report = row.get("metrics")
        metrics = report.get("metrics") if isinstance(report, Mapping) else None
        outcome = metrics.get("case_outcome") if isinstance(metrics, Mapping) else None
        value = outcome.get("final_answer_correct") if isinstance(outcome, Mapping) else None
        if type(value) is not bool:  # bool only; numeric/truthy substitutes fail closed
            raise FinalLiveError(f"{case_id or position} is missing boolean {AUTHORITATIVE_METRIC}")
        values.append(value)
    if actual_ids != list(expected_case_ids) or len(actual_ids) != len(set(actual_ids)):
        raise FinalLiveError("result case ids/order differ from the frozen manifest")
    correct = sum(values)
    return {
        "authoritative_metric": AUTHORITATIVE_METRIC,
        "correct_count": correct,
        "incorrect_count": len(values) - correct,
        "case_count": len(values),
        "accuracy": correct / len(values),
        "strict_metrics_counted": False,
        "blind_audit_counted": False,
    }


def _read_results_jsonl(path: Path, *, root: Path = FINAL_ROOT) -> list[Any]:
    rows: list[Any] = []
    safe_path = _assert_safe_path(path, root=root, label="archived results JSONL")
    if not safe_path.is_file():
        raise FinalLiveError(f"archived results is not a regular file: {safe_path}")
    try:
        handle = safe_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise FinalLiveError(f"cannot read archived results: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise FinalLiveError(f"invalid results JSONL row {line_number}") from exc
    _assert_safe_path(safe_path, root=root, label="archived results JSONL")
    return rows


def _entry_target_passed(entry: Mapping[str, Any], difficulty: str) -> bool:
    target = entry.get("target_result")
    if not isinstance(target, Mapping):
        return False
    if target.get("case_count") != EXPECTED_CASE_COUNT:
        return False
    required = TARGET_CORRECT_COUNTS[difficulty]
    return required is None or int(target.get("correct_count") or 0) >= required


def _contained_label_path(suite_root: Path, label_root: Path, relative: str, *, label: str) -> Path:
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in relative
    ):
        raise FinalLiveError(f"{label} must be suite-root-relative")
    safe_label_root = _assert_safe_path(label_root, root=suite_root, label="candidate label root")
    path = _assert_safe_path(
        suite_root / relative_path,
        root=suite_root,
        label=label,
        require_exists=False,
    )
    try:
        path.resolve().relative_to(safe_label_root.resolve())
    except ValueError as exc:
        raise FinalLiveError(f"{label} escapes the selected label") from exc
    return path


def _server_artifact_receipt(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    report_paths = run.get("report_paths")
    if not isinstance(report_paths, Mapping) or set(report_paths) != set(br.ARTIFACT_FILENAMES):
        raise FinalLiveError("server run does not identify all six report artifacts")
    receipt: dict[str, dict[str, Any]] = {}
    for artifact in br.ARTIFACT_FILENAMES:
        source = report_paths.get(artifact)
        if not isinstance(source, Mapping):
            raise FinalLiveError(f"server run has no {artifact} report artifact")
        path = source.get("path")
        sha256 = str(source.get("sha256") or "")
        size = source.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or type(size) is not int
            or size < 0
        ):
            raise FinalLiveError(f"server {artifact} artifact receipt is invalid")
        receipt[artifact] = {"path": path, "sha256": sha256, "size_bytes": size}
    return receipt


def _verify_artifact_chain(
    *,
    suite_root: Path,
    label_root: Path,
    entry: Mapping[str, Any],
    difficulty: str,
    run: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, str]:
    receipt = entry.get("artifact_copy_manifest")
    if not isinstance(receipt, Mapping):
        raise FinalLiveError(f"{difficulty} has no artifact-copy receipt")
    manifest_path = _contained_label_path(
        suite_root,
        label_root,
        str(receipt.get("path") or ""),
        label="artifact-copy manifest",
    )
    if not manifest_path.is_file() or receipt.get("sha256") != _sha256_file(
        manifest_path,
        root=suite_root,
        label="artifact-copy manifest",
    ):
        raise FinalLiveError(f"{difficulty} artifact-copy receipt changed")
    value = _read_json(manifest_path, root=suite_root, label="artifact-copy manifest")
    if not isinstance(value, dict):
        raise FinalLiveError("artifact-copy manifest is not an object")
    if value.get("run_id") != entry.get("run_id") or value.get("difficulty") != difficulty:
        raise FinalLiveError("artifact-copy manifest has the wrong identity")
    copies = value.get("copies")
    if not isinstance(copies, Mapping) or set(copies) != set(br.ARTIFACT_FILENAMES):
        raise FinalLiveError("artifact-copy manifest does not contain all six artifacts")
    stored_server_receipt = entry.get("server_artifact_receipt")
    live_server_receipt = _server_artifact_receipt(run) if run is not None else None
    if stored_server_receipt is None:
        if live_server_receipt is None:
            raise FinalLiveError(f"{difficulty} has no frozen server artifact receipt")
        server_receipt = live_server_receipt
    elif not isinstance(stored_server_receipt, Mapping) or set(stored_server_receipt) != set(
        br.ARTIFACT_FILENAMES
    ):
        raise FinalLiveError(f"{difficulty} frozen server artifact receipt is invalid")
    else:
        server_receipt = dict(stored_server_receipt)
        if live_server_receipt is not None and server_receipt != live_server_receipt:
            raise FinalLiveError(f"{difficulty} server artifact receipt changed")
    results_path: Path | None = None
    results_sha256 = ""
    for artifact, filename in br.ARTIFACT_FILENAMES.items():
        copy = copies.get(artifact)
        source = copy.get("source") if isinstance(copy, Mapping) else None
        target = copy.get("target") if isinstance(copy, Mapping) else None
        if (
            not isinstance(copy, Mapping)
            or not isinstance(source, Mapping)
            or not isinstance(target, Mapping)
        ):
            raise FinalLiveError(f"{difficulty} {artifact} artifact receipt is invalid")
        server_sha = str(source.get("server_sha256") or "")
        download_sha = str(source.get("download_sha256") or "")
        target_sha = str(target.get("sha256") or "")
        if (
            copy.get("filename") != filename
            or not re.fullmatch(r"[0-9a-f]{64}", server_sha)
            or download_sha != server_sha
            or target_sha != server_sha
            or copy.get("source_target_sha256_match") is not True
        ):
            raise FinalLiveError(f"{difficulty} {artifact} source/download/target SHA chain failed")
        target_path = _contained_label_path(
            suite_root,
            label_root,
            str(target.get("relative_path") or ""),
            label=f"{artifact} artifact",
        )
        if target_path.name != filename or not target_path.is_file():
            raise FinalLiveError(f"{difficulty} {artifact} target is missing")
        target_size = target_path.stat().st_size
        sizes = (
            source.get("server_size_bytes"),
            source.get("download_size_bytes"),
            target.get("size_bytes"),
        )
        if _sha256_file(
            target_path, root=suite_root, label=f"{artifact} artifact"
        ) != target_sha or any(size is not None and size != target_size for size in sizes):
            raise FinalLiveError(f"{difficulty} {artifact} target content/size changed")
        server = server_receipt.get(artifact)
        if (
            not isinstance(server, Mapping)
            or source.get("server_relative_path") != server.get("path")
            or server.get("sha256") != server_sha
            or server.get("size_bytes") != target_size
        ):
            raise FinalLiveError(f"{difficulty} {artifact} server receipt changed")
        if artifact == "results":
            results_path = target_path
            results_sha256 = target_sha
    if results_path is None:  # pragma: no cover - set equality above guarantees it
        raise FinalLiveError("artifact-copy manifest has no results artifact")
    return value, results_path, results_sha256


def _verify_archived_target(
    *,
    suite_root: Path,
    label_root: Path,
    entry: Mapping[str, Any],
    difficulty: str,
    spec: DatasetSpec,
) -> dict[str, Any]:
    _, results_path, results_sha256 = _verify_artifact_chain(
        suite_root=suite_root,
        label_root=label_root,
        entry=entry,
        difficulty=difficulty,
    )
    target = _target_count_from_rows(
        _read_results_jsonl(results_path, root=suite_root),
        expected_run_id=str(entry.get("run_id") or ""),
        expected_case_ids=spec.case_ids,
    )
    if target != entry.get("target_result") or results_sha256 != entry.get("results_sha256"):
        raise FinalLiveError(f"{difficulty} archived target evidence changed")
    return target


def _assert_progression(
    index: Mapping[str, Any],
    difficulty: str,
    *,
    suite: SuiteSnapshot,
    label_root: Path,
) -> None:
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping):
        raise FinalLiveError("run index has no difficulty map")
    position = DIFFICULTIES.index(difficulty)
    for preceding in DIFFICULTIES[:position]:
        entry = entries.get(preceding)
        if not isinstance(entry, Mapping) or entry.get("status") != "archived":
            raise FinalLiveError(f"{difficulty} requires archived {preceding} first")
        target = _verify_archived_target(
            suite_root=FINAL_ROOT,
            label_root=label_root,
            entry=entry,
            difficulty=preceding,
            spec=suite.datasets[preceding],
        )
        _verify_archived_config_identity(
            index,
            entry,
            difficulty=preceding,
            spec=suite.datasets[preceding],
        )
        target_passed = target["correct_count"] >= int(TARGET_CORRECT_COUNTS[preceding] or 0)
        if not target_passed:
            required = TARGET_CORRECT_COUNTS[preceding]
            raise FinalLiveError(
                f"{difficulty} is blocked: {preceding} did not reach {required}/50"
            )
    for later in DIFFICULTIES[position + 1 :]:
        if later in entries:
            raise FinalLiveError("run-index difficulty order is invalid")


def _profile_identity(preflight: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    profile = preflight.get("retrieval_profile")
    if not isinstance(profile, Mapping):
        raise FinalLiveError("live preflight has no retrieval profile snapshot")
    if profile.get("profile_id") != args.retrieval_profile_id:
        raise FinalLiveError("preflight retrieval profile id differs from the explicit pin")
    if profile.get("version") != args.retrieval_profile_version:
        raise FinalLiveError("preflight retrieval profile version differs from the explicit pin")
    config_hash = str(profile.get("config_hash") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise FinalLiveError("preflight retrieval profile has no valid config_hash")
    vector_digest = str(preflight.get("vector_read_set_digest") or "")
    if not vector_digest:
        raise FinalLiveError("preflight has no frozen vector_read_set_digest")
    return {
        "profile_id": args.retrieval_profile_id,
        "version": args.retrieval_profile_version,
        "config_hash": config_hash,
        "snapshot_sha256": _sha256_json(profile),
        "vector_read_set_digest": vector_digest,
    }


def _validate_preflight(
    preflight: Any,
    *,
    dataset_id: str,
    dataset_checksum: str,
    spec: DatasetSpec,
    args: argparse.Namespace,
    suite: SuiteSnapshot | None = None,
) -> dict[str, Any]:
    if not isinstance(preflight, Mapping) or preflight.get("ready") is not True:
        raise FinalLiveError(f"{spec.difficulty} pinned live preflight is not ready")
    dataset = preflight.get("dataset")
    knowledge_base = preflight.get("knowledge_base")
    checks = preflight.get("checks")
    selection = preflight.get("retrieval_profile_selection")
    if preflight.get("evaluation_mode") != "live":
        raise FinalLiveError("preflight evaluation_mode is not live")
    if not isinstance(dataset, Mapping) or {
        "id": dataset_id,
        "key": spec.manifest["dataset_key"],
        "version": spec.manifest["version"],
        "checksum": dataset_checksum,
    } != {key: dataset.get(key) for key in ("id", "key", "version", "checksum")}:
        raise FinalLiveError("preflight dataset identity differs from the frozen dataset")
    if (
        not isinstance(knowledge_base, Mapping)
        or knowledge_base.get("id") != args.kb_id
        or knowledge_base.get("status") != "active"
    ):
        raise FinalLiveError("preflight knowledge-base identity/status is invalid")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise FinalLiveError("preflight checks are incomplete or not all true")
    if not isinstance(selection, Mapping) or {
        "profile_id": args.retrieval_profile_id,
        "version": args.retrieval_profile_version,
        "source": "explicit",
    } != {key: selection.get(key) for key in ("profile_id", "version", "source")}:
        raise FinalLiveError("preflight retrieval profile was not selected explicitly")
    profile_identity = _profile_identity(preflight, args)
    if selection.get("config_hash") != profile_identity["config_hash"]:
        raise FinalLiveError("preflight profile selection config_hash differs")
    identity = (
        suite.source_index_identity
        if suite is not None
        else {
            "source_sha256": spec.manifest["source"]["content_sha256"],
            "kb_id": spec.manifest["source"]["kb_id"],
            "document_id": spec.manifest["source"]["document_id"],
            "document_version_id": spec.manifest["source"]["document_version_id"],
            "index_version_id": spec.manifest["source"]["index_version_id"],
            "frozen_chunk_count": EXPECTED_CHUNK_COUNT,
        }
    )
    source_integrity = preflight.get("source_integrity")
    active_indexes = preflight.get("active_indexes")
    matching_versions = preflight.get("matching_versions")
    vector_scopes = preflight.get("vector_read_scopes")
    expected_scope = {
        "kb_id": identity["kb_id"],
        "document_id": identity["document_id"],
        "document_version_id": identity["document_version_id"],
        "index_version_id": identity["index_version_id"],
    }
    scope_identities = {
        tuple(item.get(key) for key in expected_scope)
        for item in vector_scopes or []
        if isinstance(item, Mapping)
    }
    expected_scope_tuple = tuple(expected_scope.values())
    matching_identities = {
        (
            item.get("document_id"),
            item.get("active_version_id"),
            item.get("active_index_version_id"),
            item.get("chunk_count"),
        )
        for item in matching_versions or []
        if isinstance(item, Mapping)
    }
    active_identities = {
        (item.get("id"), item.get("kb_id"), item.get("status"))
        for item in active_indexes or []
        if isinstance(item, Mapping)
    }
    if (
        args.kb_id != identity["kb_id"]
        or not isinstance(source_integrity, Mapping)
        or source_integrity.get("verified") is not True
        or source_integrity.get("actual_sha256") != identity["source_sha256"]
        or source_integrity.get("expected_sha256") != identity["source_sha256"]
        or active_identities != {(identity["index_version_id"], identity["kb_id"], "active")}
        or matching_identities
        != {
            (
                identity["document_id"],
                identity["document_version_id"],
                identity["index_version_id"],
                identity["frozen_chunk_count"],
            )
        }
        or scope_identities != {expected_scope_tuple}
    ):
        raise FinalLiveError(
            "preflight source/index identity does not match manifest "
            "KB/document/version/index/chunk contract"
        )
    return profile_identity


def _preflight_invariant(preflight: Mapping[str, Any]) -> dict[str, Any]:
    # Dataset id/checksum differs by difficulty; every other preflight field is
    # execution infrastructure and must remain byte-for-byte stable.
    return {key: value for key, value in preflight.items() if key != "dataset"}


def _freeze_or_validate_preflight(index: dict[str, Any], preflight: Mapping[str, Any]) -> None:
    current = {
        "invariant_sha256": _sha256_json(_preflight_invariant(preflight)),
        "evaluation_mode": preflight.get("evaluation_mode"),
        "checks": dict(preflight.get("checks") or {}),
    }
    frozen = index.get("preflight_identity")
    if frozen is None:
        index["preflight_identity"] = current
    elif frozen != current:
        raise FinalLiveError("live preflight infrastructure changed within the label")


def _execution_config_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"dataset_version", "dataset_checksum", "selected_case_ids", "preflight"}
    invariant = {key: value for key, value in snapshot.items() if key not in excluded}
    return {
        "snapshot_invariant_sha256": _sha256_json(invariant),
        "config_version": snapshot.get("config_version"),
        "llm_model": snapshot.get("llm_model"),
        "embedding_model": snapshot.get("embedding_model"),
        "reranker_model": snapshot.get("reranker_model"),
        "gate_policy_sha256": _sha256_json(snapshot.get("gate_policy")),
        "evaluation_security_sha256": _sha256_json(snapshot.get("evaluation_security")),
        "vector_read_scopes_sha256": _sha256_json(snapshot.get("vector_read_scopes")),
        "telemetry_coverage_sha256": _sha256_json(snapshot.get("telemetry_coverage")),
    }


def _freeze_or_validate_execution_config(
    index: dict[str, Any], snapshot: Mapping[str, Any]
) -> None:
    current = _execution_config_identity(snapshot)
    frozen = index.get("execution_config_identity")
    if frozen is None:
        index["execution_config_identity"] = current
    elif frozen != current:
        raise FinalLiveError("execution config changed within the candidate label")


def _verify_archived_config_identity(
    index: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    difficulty: str,
    spec: DatasetSpec,
) -> None:
    snapshot = entry.get("config_snapshot")
    runtime = index.get("runtime_identity")
    profile_identity = index.get("retrieval_profile_identity")
    preflight_identity = index.get("preflight_identity")
    execution_identity = index.get("execution_config_identity")
    changed: list[str] = []
    if not isinstance(snapshot, Mapping):
        changed.append("config_snapshot")
    else:
        if snapshot.get("code_revision") != index.get("code_revision"):
            changed.append("code_revision")
        if snapshot.get("dataset_checksum") != entry.get("dataset_checksum"):
            changed.append("dataset_checksum")
        if snapshot.get("dataset_version") != spec.manifest.get("version"):
            changed.append("dataset_version")
        if snapshot.get("evaluation_mode") != "live":
            changed.append("evaluation_mode")
        if snapshot.get("selected_case_ids") != list(spec.case_ids):
            changed.append("selected_case_ids")
        if not isinstance(runtime, Mapping) or any(
            snapshot.get(field) != runtime.get(field)
            for field in ("config_version", "llm_model", "embedding_model", "reranker_model")
        ):
            changed.append("runtime_identity")
        profile = snapshot.get("retrieval_profile")
        if (
            not isinstance(profile, Mapping)
            or not isinstance(profile_identity, Mapping)
            or profile.get("profile_id") != profile_identity.get("profile_id")
            or profile.get("version") != profile_identity.get("version")
            or profile.get("config_hash") != profile_identity.get("config_hash")
            or _sha256_json(profile) != profile_identity.get("snapshot_sha256")
            or snapshot.get("vector_read_set_digest")
            != profile_identity.get("vector_read_set_digest")
        ):
            changed.append("retrieval_profile_identity")
        preflight = snapshot.get("preflight")
        if (
            not isinstance(preflight, Mapping)
            or not isinstance(preflight_identity, Mapping)
            or _sha256_json(_preflight_invariant(preflight))
            != preflight_identity.get("invariant_sha256")
        ):
            changed.append("preflight_identity")
        if (
            not isinstance(execution_identity, Mapping)
            or _execution_config_identity(snapshot) != execution_identity
        ):
            changed.append("execution_config_identity")
    if changed:
        raise FinalLiveError(
            f"{difficulty} archived execution identity changed: " + ", ".join(sorted(set(changed)))
        )


def _freeze_or_validate_profile(index: dict[str, Any], current: Mapping[str, Any]) -> None:
    frozen = index.get("retrieval_profile_identity")
    if frozen is None:
        index["retrieval_profile_identity"] = dict(current)
        return
    if frozen != current:
        raise FinalLiveError("retrieval profile or vector read set changed within the label")


def _assert_dataset_api(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
    prior_id: str = "",
    prior_checksum: str = "",
) -> str:
    dataset_id = br._safe_component(str(dataset.get("id") or ""), label="dataset id")
    if prior_id and dataset_id != prior_id:
        raise FinalLiveError("server dataset id changed on resume")
    expected = {
        "dataset_key": spec.manifest["dataset_key"],
        "version": spec.manifest["version"],
        "case_count": spec.case_count,
    }
    changed = [key for key, value in expected.items() if dataset.get(key) != value]
    if changed:
        raise FinalLiveError("server dataset identity differs: " + ", ".join(changed))
    checksum = str(dataset.get("checksum") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise FinalLiveError("server dataset has no valid checksum")
    if prior_checksum and checksum != prior_checksum:
        raise FinalLiveError("server dataset checksum changed on resume")
    if checksum != spec.manifest_sha256:
        raise FinalLiveError("server dataset canonical manifest checksum differs from local")
    server_manifest = dataset.get("manifest_json")
    if (
        not isinstance(server_manifest, Mapping)
        or dict(server_manifest) != spec.manifest
        or _sha256_json(server_manifest) != spec.manifest_sha256
    ):
        raise FinalLiveError("server dataset manifest_json differs from local canonical manifest")
    return dataset_id


def _assert_run_identity(
    run: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
    dataset_checksum: str,
    args: argparse.Namespace,
    index: Mapping[str, Any],
    case_ids: Sequence[str],
) -> None:
    changed: list[str] = []
    expected = {
        "id": run_id,
        "dataset_id": dataset_id,
        "kb_id": args.kb_id,
        "mode": "live",
        "baseline_run_id": None,
    }
    changed.extend(key for key, value in expected.items() if run.get(key) != value)
    snapshot = run.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        changed.append("config_snapshot")
    else:
        runtime = index.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            changed.append("runtime_identity")
        else:
            for field in ("config_version", "llm_model", "embedding_model", "reranker_model"):
                if snapshot.get(field) != runtime.get(field):
                    changed.append(f"config_snapshot.{field}")
        if snapshot.get("code_revision") != index.get("code_revision"):
            changed.append("config_snapshot.code_revision")
        if snapshot.get("dataset_checksum") != dataset_checksum:
            changed.append("config_snapshot.dataset_checksum")
        if snapshot.get("evaluation_mode") != "live":
            changed.append("config_snapshot.evaluation_mode")
        selected = snapshot.get("selected_case_ids")
        if not isinstance(selected, list) or selected != list(case_ids):
            changed.append("config_snapshot.selected_case_ids")
        profile = snapshot.get("retrieval_profile")
        frozen = index.get("retrieval_profile_identity")
        if not isinstance(profile, Mapping) or not isinstance(frozen, Mapping):
            changed.append("config_snapshot.retrieval_profile")
        else:
            profile_checks = {
                "profile_id": frozen.get("profile_id"),
                "version": frozen.get("version"),
                "config_hash": frozen.get("config_hash"),
            }
            if any(profile.get(key) != value for key, value in profile_checks.items()):
                changed.append("config_snapshot.retrieval_profile")
            if _sha256_json(profile) != frozen.get("snapshot_sha256"):
                changed.append("config_snapshot.retrieval_profile.snapshot")
        if snapshot.get("vector_read_set_digest") != (
            index.get("retrieval_profile_identity") or {}
        ).get("vector_read_set_digest"):
            changed.append("config_snapshot.vector_read_set_digest")
        preflight = snapshot.get("preflight")
        frozen_preflight = index.get("preflight_identity")
        if (
            not isinstance(preflight, Mapping)
            or not isinstance(frozen_preflight, Mapping)
            or _sha256_json(_preflight_invariant(preflight))
            != frozen_preflight.get("invariant_sha256")
        ):
            changed.append("config_snapshot.preflight")
        execution_identity = index.get("execution_config_identity")
        if (
            not isinstance(execution_identity, Mapping)
            or _execution_config_identity(snapshot) != execution_identity
        ):
            changed.append("config_snapshot.execution_config_identity")
    if changed:
        raise FinalLiveError("run identity changed: " + ", ".join(sorted(set(changed))))


def _poll(
    client: Any,
    *,
    run_id: str,
    difficulty: str,
    args: argparse.Namespace,
    index_path: Path,
    index: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = f"/api/v1/evaluation/runs/{quote(run_id, safe='')}"
    deadline = time.monotonic() + args.poll_timeout
    last: tuple[str, int, int] | None = None
    while True:
        envelope = client.request_json("GET", path)
        run = br._envelope_data(envelope, operation=f"poll {difficulty}")
        if not isinstance(run, dict) or run.get("id") != run_id:
            raise FinalLiveError("poll returned the wrong run identity")
        status = str(run.get("status") or "")
        completed = int(run.get("completed_cases") or 0)
        total = int(run.get("total_cases") or 0)
        display = (status, completed, total)
        if display != last:
            print(f"[{difficulty}] {status} ({completed}/{total})", flush=True)
            entry = index["difficulties"][difficulty]
            entry.update(
                {
                    "status": status,
                    "completed_cases": completed,
                    "total_cases": total,
                    "last_polled_at": br._utc_now(),
                }
            )
            _persist_index(index_path, index)
            last = display
        if status in TERMINAL_RUN_STATUSES:
            return envelope, run
        if status not in ACTIVE_RUN_STATUSES:
            raise FinalLiveError(f"unsupported run status: {status!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FinalLiveError("poll timeout; rerun the same difficulty with --resume")
        time.sleep(min(args.poll_interval, remaining))


def _load_existing_copy_manifest(
    manifest_path: Path,
    *,
    label_root: Path,
    difficulty: str,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    value, _, _ = _verify_artifact_chain(
        suite_root=FINAL_ROOT,
        label_root=label_root,
        entry={
            "run_id": run.get("id"),
            "artifact_copy_manifest": {
                "path": _relative_final(manifest_path),
                "sha256": _sha256_file(
                    manifest_path,
                    root=FINAL_ROOT,
                    label="artifact-copy manifest",
                ),
            },
        },
        difficulty=difficulty,
        run=run,
    )
    return value


def _write_api_copy(
    run_dir: Path, invocation_id: str, filename: str, value: Any, redactor: Any
) -> None:
    short_id = invocation_id.rsplit("-", maxsplit=1)[-1]
    _write_json(run_dir / "api" / short_id / filename, redactor.clean(value))


def _process_difficulty(
    *,
    difficulty: str,
    args: argparse.Namespace,
    suite: SuiteSnapshot,
    label_root: Path,
    index_path: Path,
    index: dict[str, Any],
    client: Any,
    redactor: Any,
    invocation_id: str,
) -> None:
    _assert_progression(index, difficulty, suite=suite, label_root=label_root)
    spec = suite.datasets[difficulty]
    existing = index["difficulties"].get(difficulty)
    if isinstance(existing, Mapping) and existing.get("status") == "archived":
        if not args.resume:
            raise FinalLiveError("archived difficulty requires --resume for local verification")
        _verify_archived_target(
            suite_root=FINAL_ROOT,
            label_root=label_root,
            entry=existing,
            difficulty=difficulty,
            spec=spec,
        )
        _verify_archived_config_identity(
            index,
            existing,
            difficulty=difficulty,
            spec=spec,
        )
        print(f"[{difficulty}] verified existing immutable candidate archive", flush=True)
        if not _entry_target_passed(existing, difficulty):
            required = TARGET_CORRECT_COUNTS[difficulty]
            raise FinalLiveError(
                f"{difficulty} remains below the required {required}/50 target; "
                "start a new label after a generic workflow change"
            )
        return

    entry = dict(existing) if isinstance(existing, Mapping) else {}
    entry.update(
        {
            "difficulty": difficulty,
            "role": "candidate_root_run",
            "mode": "live",
            "baseline_run_id": None,
            "dataset_source_path": _relative_project(spec.path),
            "dataset_source_sha256": spec.file_sha256,
            "selected_case_count": spec.case_count,
            "selected_case_ids": list(spec.case_ids),
            "status": entry.get("status") or "dataset_import_pending",
        }
    )
    index["difficulties"][difficulty] = entry
    _persist_index(index_path, index)

    prior_dataset_id = str(entry.get("dataset_id") or "")
    prior_dataset_checksum = str(entry.get("dataset_checksum") or "")
    if prior_dataset_id:
        dataset_envelope = client.request_json(
            "GET", f"/api/v1/evaluation/datasets/{quote(prior_dataset_id, safe='')}"
        )
    else:
        dataset_envelope = client.request_json(
            "POST", "/api/v1/evaluation/datasets", body=spec.request_body
        )
    dataset = br._envelope_data(dataset_envelope, operation=f"dataset {difficulty}")
    if not isinstance(dataset, Mapping):
        raise FinalLiveError("dataset API returned non-object data")
    dataset_id = _assert_dataset_api(
        dataset,
        spec,
        prior_dataset_id,
        prior_dataset_checksum,
    )
    dataset_checksum = str(dataset["checksum"])
    entry.update(
        {
            "dataset_id": dataset_id,
            "dataset_key": dataset.get("dataset_key"),
            "dataset_version": dataset.get("version"),
            "dataset_checksum": dataset_checksum,
            "status": "preflight_pending",
        }
    )
    _persist_index(index_path, index)

    preflight_body = {
        "dataset_id": dataset_id,
        "kb_id": args.kb_id,
        "mode": "live",
        "retrieval_profile_id": args.retrieval_profile_id,
        "retrieval_profile_version": args.retrieval_profile_version,
    }
    preflight_envelope = client.request_json(
        "POST", "/api/v1/evaluation/preflight", body=preflight_body
    )
    preflight = br._envelope_data(preflight_envelope, operation=f"preflight {difficulty}")
    profile_identity = _validate_preflight(
        preflight,
        dataset_id=dataset_id,
        dataset_checksum=dataset_checksum,
        spec=spec,
        args=args,
        suite=suite,
    )
    _freeze_or_validate_profile(index, profile_identity)
    _freeze_or_validate_preflight(index, preflight)
    _persist_index(index_path, index)

    run_id = str(entry.get("run_id") or "") if args.resume else ""
    create_envelope: dict[str, Any] | None = None
    if run_id:
        run_id = br._safe_component(run_id, label="run id")
        initial_envelope = client.request_json(
            "GET", f"/api/v1/evaluation/runs/{quote(run_id, safe='')}"
        )
    else:
        body = {
            "dataset_id": dataset_id,
            "kb_id": args.kb_id,
            "mode": "live",
            "name": f"final-{args.label}-{difficulty}-candidate-root",
            "baseline_run_id": None,
            "retrieval_profile_id": args.retrieval_profile_id,
            "retrieval_profile_version": args.retrieval_profile_version,
        }
        create_envelope = client.request_json("POST", "/api/v1/evaluation/runs", body=body)
        initial_envelope = create_envelope
    initial_run = br._envelope_data(initial_envelope, operation=f"run {difficulty}")
    if not isinstance(initial_run, Mapping):
        raise FinalLiveError("run API returned non-object data")
    if not run_id:
        run_id = br._safe_component(str(initial_run.get("id") or ""), label="run id")
        entry.update(
            {
                "run_id": run_id,
                "status": str(initial_run.get("status") or "queued"),
                "created_at": initial_run.get("created_at"),
            }
        )
        _persist_index(index_path, index)
    snapshot = initial_run.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        raise FinalLiveError("run has no config_snapshot")
    _freeze_or_validate_execution_config(index, snapshot)
    _persist_index(index_path, index)
    _assert_run_identity(
        initial_run,
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_checksum=dataset_checksum,
        args=args,
        index=index,
        case_ids=spec.case_ids,
    )

    run_dir = label_root / "artifacts" / difficulty / run_id
    _assert_safe_path(
        run_dir,
        root=FINAL_ROOT,
        label=f"{difficulty} run directory",
        require_exists=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(run_dir, root=FINAL_ROOT, label=f"{difficulty} run directory")
    _write_api_copy(run_dir, invocation_id, "dataset.json", dataset_envelope, redactor)
    _write_api_copy(run_dir, invocation_id, "preflight.json", preflight_envelope, redactor)
    _write_api_copy(
        run_dir,
        invocation_id,
        "run-create.json" if create_envelope is not None else "run-resume.json",
        initial_envelope,
        redactor,
    )

    final_envelope, final_run = _poll(
        client,
        run_id=run_id,
        difficulty=difficulty,
        args=args,
        index_path=index_path,
        index=index,
    )
    _write_api_copy(run_dir, invocation_id, "run-final.json", final_envelope, redactor)
    if final_run.get("status") != "succeeded":
        raise FinalLiveError(
            f"{difficulty} run ended as {final_run.get('status')} "
            f"({final_run.get('error_code') or 'unknown_error'})"
        )
    _assert_run_identity(
        final_run,
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_checksum=dataset_checksum,
        args=args,
        index=index,
        case_ids=spec.case_ids,
    )

    encoded = quote(run_id, safe="")
    results_envelope = client.request_json("GET", f"/api/v1/evaluation/runs/{encoded}/results")
    comparison_envelope = client.request_json(
        "GET", f"/api/v1/evaluation/runs/{encoded}/comparison"
    )
    _write_api_copy(run_dir, invocation_id, "results.json", results_envelope, redactor)
    _write_api_copy(run_dir, invocation_id, "comparison.json", comparison_envelope, redactor)
    api_rows = br._envelope_data(results_envelope, operation=f"results {difficulty}")
    if not isinstance(api_rows, list):
        raise FinalLiveError("results API returned non-list data")
    api_target = _target_count_from_rows(
        api_rows, expected_run_id=run_id, expected_case_ids=spec.case_ids
    )

    manifest_path = run_dir / "artifact-copy-manifest.json"
    if manifest_path.is_file():
        copy_manifest = _load_existing_copy_manifest(
            manifest_path,
            label_root=label_root,
            difficulty=difficulty,
            run=final_run,
        )
    else:
        manifest_path, copy_manifest = br._archive_artifacts(
            client,
            difficulty=difficulty,
            run=dict(final_run),
            allow_repair=bool(args.resume),
        )
    copy_manifest, results_path, results_sha256 = _verify_artifact_chain(
        suite_root=FINAL_ROOT,
        label_root=label_root,
        entry={
            "run_id": run_id,
            "artifact_copy_manifest": {
                "path": _relative_final(manifest_path),
                "sha256": _sha256_file(
                    manifest_path,
                    root=FINAL_ROOT,
                    label="artifact-copy manifest",
                ),
            },
        },
        difficulty=difficulty,
        run=final_run,
    )
    archived_target = _target_count_from_rows(
        _read_results_jsonl(results_path, root=FINAL_ROOT),
        expected_run_id=run_id,
        expected_case_ids=spec.case_ids,
    )
    if archived_target != api_target:
        raise FinalLiveError("archived target result differs from the results API")
    required = TARGET_CORRECT_COUNTS[difficulty]
    target_passed = required is None or archived_target["correct_count"] >= required
    entry.update(
        redactor.clean(
            {
                "status": "archived",
                "completed_at": final_run.get("completed_at"),
                "completed_cases": final_run.get("completed_cases"),
                "total_cases": final_run.get("total_cases"),
                "run_directory": _relative_final(run_dir),
                "artifact_copy_manifest": {
                    "path": _relative_final(manifest_path),
                    "sha256": _sha256_file(
                        manifest_path,
                        root=FINAL_ROOT,
                        label="artifact-copy manifest",
                    ),
                },
                "server_artifact_receipt": _server_artifact_receipt(final_run),
                "results_path": _relative_final(results_path),
                "results_sha256": results_sha256,
                "target_result": archived_target,
                "target_required_correct_count": required,
                "target_passed": target_passed,
                "gate_result_diagnostic_only": final_run.get("gate_result") or {},
                "metrics_summary_diagnostic_only": final_run.get("metrics_summary") or {},
                "comparison_summary_diagnostic_only": final_run.get("comparison_summary") or {},
                "config_snapshot": final_run.get("config_snapshot") or {},
            }
        )
    )
    _persist_index(index_path, index)
    print(
        f"[{difficulty}] candidate root archived: "
        f"{archived_target['correct_count']}/{archived_target['case_count']}",
        flush=True,
    )
    if not target_passed:
        raise FinalLiveError(f"{difficulty} did not reach the required {required}/50 target")


def _candidate_record(
    entry: Mapping[str, Any], difficulty: str, index: Mapping[str, Any]
) -> dict[str, Any]:
    source_path = (PROJECT_ROOT / str(entry["dataset_source_path"])).resolve()
    return {
        "role": "candidate_root_run",
        "canonical": False,
        "difficulty": difficulty,
        "mode": "live",
        "kb_id": index["kb_id"],
        "run_id": entry["run_id"],
        "baseline_run_id": None,
        "status": "succeeded",
        "dataset": {
            "id": entry["dataset_id"],
            "key": entry["dataset_key"],
            "version": entry["dataset_version"],
            "checksum": entry["dataset_checksum"],
            "case_count": entry["selected_case_count"],
            "source_path": _relative_final(source_path),
            "source_sha256": entry["dataset_source_sha256"],
        },
        "target_result": entry["target_result"],
        "config_snapshot": entry["config_snapshot"],
        "artifact_copy_manifest": entry["artifact_copy_manifest"],
        "server_artifact_receipt": entry["server_artifact_receipt"],
        "run_directory": entry["run_directory"],
    }


def _finalize_candidate_registry(label_root: Path, index_path: Path, index: dict[str, Any]) -> None:
    entries = index.get("difficulties")
    if not isinstance(entries, Mapping):
        return
    if any(
        not isinstance(entries.get(difficulty), Mapping)
        or entries[difficulty].get("status") != "archived"
        or not _entry_target_passed(entries[difficulty], difficulty)
        for difficulty in DIFFICULTIES
    ):
        return
    registry = {
        "schema_version": CANDIDATE_REGISTRY_SCHEMA,
        "status": "succeeded",
        "role": "candidate_root_run",
        "canonical": False,
        "label": index["label"],
        "suite_key": index["suite_key"],
        "mode": "live",
        "kb_id": index["kb_id"],
        "generated_at": br._utc_now(),
        "difficulty_order": list(DIFFICULTIES),
        "baselines": {
            difficulty: _candidate_record(entries[difficulty], difficulty, index)
            for difficulty in DIFFICULTIES
        },
    }
    path = label_root / "candidate-roots.json"
    _write_json(path, registry)
    receipt = {
        "path": _relative_final(path),
        "sha256": _sha256_file(path, root=FINAL_ROOT, label="candidate registry"),
    }
    # Compatibility alias for the existing semantic reviewer.  The referenced
    # file explicitly identifies itself as non-canonical candidate roots.
    index["candidate_roots"] = receipt
    index["source_baseline_registry"] = {**receipt, "compatibility_alias": True}
    index["status"] = "succeeded"
    _persist_index(index_path, index)


def _write_checksums(label_root: Path) -> None:
    destination = label_root / "SHA256SUMS"
    files = [
        path
        for path in _tree_files(
            label_root,
            trusted_root=FINAL_ROOT,
            label="candidate label",
        )
        if path != destination
    ]
    content = "".join(
        f"{_sha256_file(path, root=FINAL_ROOT, label='candidate artifact')}  "
        f"{path.relative_to(label_root).as_posix()}\n"
        for path in files
    )
    _write_bytes_safe(
        destination,
        content.encode("utf-8"),
        root=FINAL_ROOT,
        label="candidate SHA256SUMS",
        overwrite=True,
    )


def _parse_args(config: Mapping[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, type=_safe_label)
    parser.add_argument("--difficulty", required=True, choices=DIFFICULTIES)
    parser.add_argument(
        "--retrieval-profile-id",
        required=True,
        help="mandatory immutable profile pin; environment defaults are intentionally rejected",
    )
    parser.add_argument(
        "--retrieval-profile-version",
        required=True,
        type=_positive_int,
        help="mandatory immutable profile version pin",
    )
    parser.add_argument(
        "--base-url",
        default=br._first_config(
            dict(config),
            "STT_EVAL_BASE_URL",
            "STT_QUERY_BASE_URL",
            default=br.DEFAULT_BASE_URL,
        ),
    )
    parser.add_argument(
        "--kb-id",
        default=br._first_config(dict(config), "STT_EVAL_KB_ID", "EVALUATION_KB_ID"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--poll-timeout",
        type=br._positive_float,
        default=float(br.DEFAULT_POLL_TIMEOUT_SECONDS),
    )
    parser.add_argument(
        "--poll-interval",
        type=br._positive_float,
        default=float(br.DEFAULT_POLL_INTERVAL_SECONDS),
    )
    parser.add_argument(
        "--request-timeout",
        type=br._positive_float,
        default=float(br.DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )
    args = parser.parse_args()
    args.base_url = br._normalize_local_base_url(args.base_url)
    args.kb_id = br._safe_component(str(args.kb_id or ""), label="knowledge-base id")
    args.retrieval_profile_id = br._safe_component(
        str(args.retrieval_profile_id or ""), label="retrieval profile id"
    )
    return args


def _validate_cli_suite_identity(args: argparse.Namespace, suite: SuiteSnapshot) -> None:
    expected_kb_id = suite.source_index_identity["kb_id"]
    if args.kb_id != expected_kb_id:
        raise FinalLiveError(
            "--kb-id does not match the KB frozen in all three canonical manifests"
        )


def _run(
    args: argparse.Namespace,
    config: dict[str, str],
    suite: SuiteSnapshot,
    code_revision: str,
) -> int:
    label_root, index_path, index = _prepare_label(args, suite, code_revision)
    _assert_progression(
        index,
        args.difficulty,
        suite=suite,
        label_root=label_root,
    )

    email = br._first_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_EMAIL",
        "AUTH_BOOTSTRAP_ADMIN_EMAIL",
        default="admin@stt.local",
    )
    password = br._first_secret_config(
        config,
        "STT_AUTH_BOOTSTRAP_ADMIN_PASSWORD",
        "AUTH_BOOTSTRAP_ADMIN_PASSWORD",
    )
    tenant_id = br._first_config(config, "STT_EVAL_TENANT_ID", "AUTH_BOOTSTRAP_TENANT_ID") or None
    if not password:
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.shared.config.settings_config import settings

        email = settings.auth_bootstrap_admin_email
        password = settings.auth_bootstrap_admin_password
    if not password:
        raise FinalLiveError("bootstrap admin password could not be resolved")

    br.SUITE_ROOT = FINAL_ROOT
    br.RUNS_DIR = label_root / "artifacts"
    br.API_SNAPSHOT_ROOT = label_root / "api-snapshots"
    redactor = FinalRedactor(password)
    invocation_id = f"{br._timestamp_slug()}-{uuid.uuid4().hex[:8]}"
    invocation = {
        "id": invocation_id,
        "difficulty": args.difficulty,
        "started_at": br._utc_now(),
        "status": "running",
        "resume": bool(args.resume),
    }
    index["invocations"].append(invocation)
    _persist_index(index_path, index)
    recorder = br.ApiSnapshotRecorder(invocation_id, args.base_url, redactor)
    client = br.ApiClient(
        base_url=args.base_url,
        email=email,
        password=password,
        tenant_id=tenant_id,
        timeout_seconds=args.request_timeout,
        recorder=recorder,
        redactor=redactor,
    )
    error: Exception | None = None
    try:
        client.login()
        print("Authenticated to the loopback query API; credentials are redacted.", flush=True)
        _process_difficulty(
            difficulty=args.difficulty,
            args=args,
            suite=suite,
            label_root=label_root,
            index_path=index_path,
            index=index,
            client=client,
            redactor=redactor,
            invocation_id=invocation_id,
        )
        _finalize_candidate_registry(label_root, index_path, index)
        if index.get("status") != "succeeded":
            index["status"] = "in_progress"
    except Exception as exc:
        error = exc
        index["status"] = "failed"
        index["error_type"] = type(exc).__name__
    finally:
        try:
            client.logout()
        except Exception as logout_exc:
            invocation["logout_error_type"] = type(logout_exc).__name__
        invocation["ended_at"] = br._utc_now()
        invocation["status"] = "failed" if error else "succeeded"
        if error is not None:
            invocation["error_type"] = type(error).__name__
        finalization_error: Exception | None = None
        try:
            invocation["api_snapshots"] = recorder.close(
                status=invocation["status"],
                error_type=invocation.get("error_type"),
            )
            _persist_index(index_path, redactor.clean(index))
            _write_checksums(label_root)
        except Exception as exc:
            finalization_error = exc
    if error is not None:
        raise FinalLiveError(f"{type(error).__name__}: {redactor.text(str(error))}") from error
    if finalization_error is not None:
        raise FinalLiveError(
            f"finalization {type(finalization_error).__name__}: "
            f"{redactor.text(str(finalization_error))}"
        ) from finalization_error
    print(f"Candidate run index: {_relative_project(index_path)}", flush=True)
    return 0


def main() -> int:
    config = br._load_environment()
    args = _parse_args(config)
    suite = load_suite_snapshot()
    _validate_cli_suite_identity(args, suite)
    code_revision = _current_code_revision()
    args.runtime_identity = _local_runtime_identity()
    with LabelLock(args.label):
        return _run(args, config, suite, code_revision)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalLiveError, br.BaselineRunnerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
