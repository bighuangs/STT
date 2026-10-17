from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "run_live_remediation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_live_remediation_profile_preflight_test",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module()


def _profile_hash(marker: str) -> str:
    return hashlib.sha256(marker.encode("utf-8")).hexdigest()


def _preflight(
    *,
    kb_id: str = "kb_target",
    profile_id: str = "rpf_default",
    version: int = 4,
    config_hash: str | None = None,
) -> dict[str, Any]:
    resolved_hash = config_hash or _profile_hash("expected")
    return {
        "ready": True,
        "knowledge_base": {"id": kb_id},
        "retrieval_profile": {
            "tenant_id": "tenant_test",
            "kb_id": kb_id,
            "profile_id": profile_id,
            "version": version,
            "config_hash": resolved_hash,
            "status": "draft",
        },
        "retrieval_profile_selection": {
            "profile_id": profile_id,
            "version": version,
            "config_hash": resolved_hash,
            "source": "explicit",
        },
    }


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "label": "profile-preflight-test",
        "base_url": "http://127.0.0.1:8000",
        "kb_id": "kb_target",
        "difficulty": ("easy",),
        "selection": "all",
        "mode": "live",
        "retrieval_profile_id": "rpf_default",
        "retrieval_profile_version": 4,
        "retrieval_profile_config_hash": _profile_hash("expected"),
        "as_root": False,
        "resume": False,
        "poll_timeout": 1.0,
        "poll_interval": 0.01,
        "request_timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_new_explicit_profile_requires_expected_config_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--label",
            "profile-preflight-test",
            "--kb-id",
            "kb_target",
            "--retrieval-profile-id",
            "rpf_default",
            "--retrieval-profile-version",
            "4",
        ],
    )

    with pytest.raises(
        runner.br.BaselineRunnerError,
        match="require --retrieval-profile-config-hash",
    ):
        runner._parse_args({})


def test_profile_preflight_freezes_kb_scoped_identity() -> None:
    expected_hash = _profile_hash("expected")
    receipt = runner._assert_preflight_profile_identity(
        _preflight(config_hash=expected_hash),
        kb_id="kb_target",
        expected_pin={
            "profile_id": "rpf_default",
            "version": 4,
            "config_hash": expected_hash,
        },
    )

    assert receipt == {
        "schema_version": runner.PROFILE_PROVENANCE_SCHEMA,
        "tenant_id": "tenant_test",
        "kb_id": "kb_target",
        "profile_id": "rpf_default",
        "version": 4,
        "config_hash": expected_hash,
        "status": "draft",
        "selection_source": "explicit",
        "identity_sha256": receipt["identity_sha256"],
    }
    assert len(receipt["identity_sha256"]) == 64


def test_frozen_profile_identity_allows_selection_source_to_become_explicit() -> None:
    receipt = runner._assert_preflight_profile_identity(
        _preflight(),
        kb_id="kb_target",
        expected_pin=None,
    )
    first = {**receipt, "selection_source": "current"}
    pinned = {**receipt, "selection_source": "explicit"}
    index = {"retrieval_profile_provenance": first}
    entry: dict[str, Any] = {}

    runner._freeze_profile_provenance(index, entry, pinned)

    assert index["retrieval_profile_provenance"] == first
    assert entry["retrieval_profile_provenance"] == pinned


@pytest.mark.parametrize(
    ("preflight", "expected_hash", "reason"),
    [
        (
            _preflight(kb_id="kb_other"),
            _profile_hash("expected"),
            "knowledge_base.id",
        ),
        (
            _preflight(config_hash=_profile_hash("unexpected")),
            _profile_hash("expected"),
            "retrieval_profile.config_hash",
        ),
    ],
)
def test_profile_preflight_rejects_wrong_kb_or_hash_before_run_creation(
    preflight: dict[str, Any],
    expected_hash: str,
    reason: str,
) -> None:
    with pytest.raises(runner.br.BaselineRunnerError, match=reason):
        runner._assert_preflight_profile_identity(
            preflight,
            kb_id="kb_target",
            expected_pin={
                "profile_id": "rpf_default",
                "version": 4,
                "config_hash": expected_hash,
            },
        )


def test_process_difficulty_stops_after_bad_profile_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_root = tmp_path / "suite"
    dataset_path = suite_root / "datasets" / "easy.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("{}", encoding="utf-8")
    results_path = suite_root / "runs" / "easy" / "base_run" / "api" / "results.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "case_id": "case-safe-placeholder",
                        "run_id": "base_run",
                        "passed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "SUITE_ROOT", suite_root)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner.br, "RUNS_DIR", tmp_path / "artifacts")

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request_json(
            self,
            method: str,
            path: str,
            *,
            body: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del body
            self.calls.append((method, path))
            if path == "/api/v1/evaluation/preflight":
                return {"data": _preflight(config_hash=_profile_hash("wrong-profile-config"))}
            raise AssertionError("run creation must not be reached")

    class IdentityRedactor:
        @staticmethod
        def clean(value: Any) -> Any:
            return value

    args = _args()
    client = FakeClient()
    registry = {
        "baselines": {
            "easy": {
                "run_id": "base_run",
                "dataset": {
                    "id": "dataset_easy",
                    "key": "easy",
                    "version": "1",
                    "checksum": "dataset-checksum",
                    "source_path": "datasets/easy.json",
                    "source_sha256": runner.br._sha256_file(dataset_path),
                    "case_count": 1,
                },
            }
        }
    }
    index = {"difficulties": {}}

    with pytest.raises(
        runner.br.BaselineRunnerError,
        match="retrieval_profile.config_hash",
    ):
        runner._process_difficulty(
            difficulty="easy",
            args=args,
            client=client,
            redactor=IdentityRedactor(),
            registry=registry,
            index_path=tmp_path / "run-index.json",
            index=index,
            invocation_id="invocation-test",
        )

    assert client.calls == [("POST", "/api/v1/evaluation/preflight")]
    assert "run_id" not in index["difficulties"]["easy"]


def test_run_snapshot_must_match_frozen_profile_hash() -> None:
    expected_hash = _profile_hash("expected")
    run = {
        "id": "run_test",
        "dataset_id": "dataset_easy",
        "kb_id": "kb_target",
        "mode": "live",
        "baseline_run_id": None,
        "total_cases": 1,
        "config_snapshot": {
            "selected_case_ids": ["case-safe-placeholder"],
            "dataset_checksum": "dataset-checksum",
            "retrieval_profile": {
                "kb_id": "kb_target",
                "profile_id": "rpf_default",
                "version": 4,
                "config_hash": _profile_hash("wrong-profile-config"),
            },
        },
    }
    expected = {
        "dataset_id": "dataset_easy",
        "baseline_run_id": None,
        "selected_case_count": 1,
        "selected_case_ids": ["case-safe-placeholder"],
        "dataset_checksum": "dataset-checksum",
    }

    with pytest.raises(
        runner.br.BaselineRunnerError,
        match="config_snapshot.retrieval_profile.config_hash",
    ):
        runner._assert_run_identity(
            run,
            run_id="run_test",
            args=_args(),
            expected=expected,
            profile_provenance={
                "kb_id": "kb_target",
                "profile_id": "rpf_default",
                "version": 4,
                "config_hash": expected_hash,
            },
        )
