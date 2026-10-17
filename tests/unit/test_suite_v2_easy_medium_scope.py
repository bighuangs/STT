from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = (
    PROJECT_ROOT
    / "docs"
    / "evaluation-baselines"
    / "2026-08-10-three-difficulty"
    / "remediation-2026-08-10"
    / "suite-v2-concept-contract"
)
sys.path.insert(0, str(CONTRACT_ROOT))

import easy_medium_scope as scope_contract  # noqa: E402
import review_blocker_rereview as blocker_wrapper  # noqa: E402
import review_contract_rereview as contract_wrapper  # noqa: E402
import review_contracts as runner  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scope_authority(pack_name: str, requests: list[dict]) -> list[dict]:
    """从独立候选或原始150题账本构造scope权威行。"""

    if pack_name == "blocker-rereview":
        candidates = json.loads(
            (CONTRACT_ROOT / pack_name / "candidates.json").read_text(encoding="utf-8")
        )
        request_map = json.loads(
            (CONTRACT_ROOT / pack_name / "request-map.json").read_text(encoding="utf-8")
        )
        proposals: list[dict] = []
        for path in (
            CONTRACT_ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json",
            CONTRACT_ROOT / "reviewer-block-supplemental-proposals.json",
        ):
            proposals.extend(json.loads(path.read_text(encoding="utf-8"))["proposals"])
        return scope_contract.build_blocker_scope_authority(
            candidates,
            request_map=request_map,
            proposals=proposals,
        )
    return scope_contract.build_contract_scope_authority(
        requests,
        _read_jsonl(CONTRACT_ROOT / "review" / "requests.jsonl"),
        json.loads(
            (CONTRACT_ROOT / "review" / "request-map.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _synthetic_request(custom_id: str, marker: str) -> dict:
    """构造满足强payload身份契约的最小测试请求。"""

    payload = {"marker": marker}
    return {
        "custom_id": custom_id,
        "payload_sha256": scope_contract._sha256_json(payload),
        "request": {
            "messages": [
                {"role": "system", "content": "test"},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ]
        },
    }


def test_scope_profile_is_frozen_and_non_promotable() -> None:
    assert scope_contract.scope_profile() == {
        "schema_version": "stt.eval.easy-medium-evaluation-scope.v1",
        "scope_name": "easy-medium-evaluation",
        "scope_id": "easy-medium-evaluation-2026-08-13-v1",
        "difficulties": ["easy", "medium"],
        "quality_targets": {"easy": 48, "medium": 43},
        "hard_reference_only": True,
        "candidate_run_allowed": True,
        "canonical_promotion_allowed": False,
        "promotable": False,
        "baseline_registration_allowed": False,
    }


@pytest.mark.parametrize(
    ("pack_name", "expected_count", "expected_by_difficulty"),
    [
        ("blocker-rereview", 4, {"easy": 4, "medium": 0}),
        ("contract-rereview", 26, {"easy": 8, "medium": 18}),
    ],
)
def test_scope_selection_uses_full_pack_and_exact_request_map(
    pack_name: str,
    expected_count: int,
    expected_by_difficulty: dict[str, int],
) -> None:
    pack_root = CONTRACT_ROOT / pack_name
    requests = _read_jsonl(pack_root / "requests.jsonl")
    request_map = json.loads((pack_root / "request-map.json").read_text(encoding="utf-8"))

    selection = scope_contract.select_scope_requests(
        requests,
        request_map,
        scope_contract.SCOPE_NAME,
        authority_rows=_scope_authority(pack_name, requests),
    )

    assert selection.full_case_count == len(requests)
    assert selection.target_case_count == expected_count
    assert selection.target_difficulty_counts == expected_by_difficulty
    assert set(selection.target_custom_ids).issubset(
        {str(row["custom_id"]) for row in requests}
    )
    assert len(selection.full_request_pack_sha256) == 64
    assert len(selection.target_request_pack_sha256) == 64


def test_scope_selection_rejects_incomplete_request_map() -> None:
    requests = [
        _synthetic_request("easy-id", "a"),
        _synthetic_request("hard-id", "b"),
    ]
    request_map = [
        {
            "custom_id": "easy-id",
            "parent_case_id": "P3500-EASY-001",
            "case_id": "P3500-EASY-V2-001",
            "difficulty": "easy",
        }
    ]

    with pytest.raises(RuntimeError, match="request-map.*完整覆盖"):
        scope_contract.select_scope_requests(
            requests,
            request_map,
            scope_contract.SCOPE_NAME,
            authority_rows=[
                {
                    "custom_id": "easy-id",
                    "parent_case_id": "P3500-EASY-001",
                    "case_id": "P3500-EASY-V2-001",
                    "difficulty": "easy",
                    "payload_sha256": requests[0]["payload_sha256"],
                },
                {
                    "custom_id": "hard-id",
                    "parent_case_id": "P3500-HARD-001",
                    "case_id": "P3500-HARD-V2-001",
                    "difficulty": "hard",
                    "payload_sha256": requests[1]["payload_sha256"],
                },
            ],
        )


@pytest.mark.parametrize("pack_name", ["blocker-rereview", "contract-rereview"])
def test_scope_selection_rejects_easy_hard_difficulty_swap(pack_name: str) -> None:
    pack_root = CONTRACT_ROOT / pack_name
    requests = _read_jsonl(pack_root / "requests.jsonl")
    request_map = json.loads((pack_root / "request-map.json").read_text(encoding="utf-8"))
    authority = _scope_authority(pack_name, requests)
    tampered = deepcopy(request_map)
    easy = next(row for row in tampered if row["difficulty"] == "easy")
    hard = next(row for row in tampered if row["difficulty"] == "hard")
    easy["difficulty"], hard["difficulty"] = hard["difficulty"], easy["difficulty"]

    with pytest.raises(RuntimeError, match="request-map.*authority"):
        scope_contract.select_scope_requests(
            requests,
            tampered,
            scope_contract.SCOPE_NAME,
            authority_rows=authority,
        )


def test_contract_authority_rejects_coordinated_dataset_and_map_swap() -> None:
    """即使三处difficulty同步互换，也不能重定义冻结候选题身份。"""

    manifests = {
        difficulty: json.loads(
            (CONTRACT_ROOT / "datasets" / f"{difficulty}.v2.json").read_text(
                encoding="utf-8"
            )
        )
        for difficulty in ("easy", "medium", "hard")
    }
    suite = json.loads((CONTRACT_ROOT / "suite-v2.json").read_text(encoding="utf-8"))
    qa_summary = json.loads(
        (CONTRACT_ROOT / "qa" / "summary.json").read_text(encoding="utf-8")
    )
    tampered = deepcopy(manifests)
    easy_case = next(
        row for row in tampered["easy"]["cases"] if row["case_id"].endswith("-007")
    )
    hard_case = next(
        row for row in tampered["hard"]["cases"] if row["case_id"].endswith("-022")
    )
    easy_case["metadata"]["difficulty"], hard_case["metadata"]["difficulty"] = (
        hard_case["metadata"]["difficulty"],
        easy_case["metadata"]["difficulty"],
    )

    with pytest.raises(RuntimeError, match="candidate manifest.*frozen|namespace"):
        scope_contract.build_candidate_manifest_scope_authority(
            tampered,
            suite_receipt=suite,
            qa_summary=qa_summary,
        )


def test_blocker_authority_rejects_coordinated_candidate_and_map_swap() -> None:
    """候选题与map成套互换lineage/hash时仍必须由冻结receipt拒绝。"""

    pack_root = CONTRACT_ROOT / "blocker-rereview"
    candidates = json.loads((pack_root / "candidates.json").read_text(encoding="utf-8"))
    request_map = json.loads((pack_root / "request-map.json").read_text(encoding="utf-8"))
    proposal_rows: list[dict] = []
    for path in (
        CONTRACT_ROOT.parent / "suite-v2-blocker-remediation" / "proposals.json",
        CONTRACT_ROOT / "reviewer-block-supplemental-proposals.json",
    ):
        proposal_rows.extend(json.loads(path.read_text(encoding="utf-8"))["proposals"])
    tampered_candidates = deepcopy(candidates)
    tampered_map = deepcopy(request_map)
    easy_case = next(
        row
        for row in tampered_candidates
        if (row.get("metadata") or {}).get("parent_case_id") == "P3500-EASY-004"
    )
    hard_case = next(
        row
        for row in tampered_candidates
        if (row.get("metadata") or {}).get("parent_case_id") == "P3500-HARD-003"
    )
    fields = ("parent_case_id", "difficulty", "blocker_remediation_proposal_sha256")
    for field in fields:
        easy_case["metadata"][field], hard_case["metadata"][field] = (
            hard_case["metadata"][field],
            easy_case["metadata"][field],
        )
    easy_case["case_id"], hard_case["case_id"] = hard_case["case_id"], easy_case["case_id"]
    easy_map = next(row for row in tampered_map if row["parent_case_id"] == "P3500-EASY-004")
    hard_map = next(row for row in tampered_map if row["parent_case_id"] == "P3500-HARD-003")
    for field in ("parent_case_id", "case_id", "difficulty", "proposed_case_sha256"):
        easy_map[field], hard_map[field] = hard_map[field], easy_map[field]
    by_case_id = {row["case_id"]: row for row in tampered_candidates}
    for row in tampered_map:
        row["candidate_case_sha256"] = scope_contract._sha256_json(by_case_id[row["case_id"]])

    with pytest.raises(RuntimeError, match="blocker.*frozen|namespace"):
        scope_contract.build_blocker_scope_authority(
            tampered_candidates,
            request_map=tampered_map,
            proposals=proposal_rows,
        )


def test_scope_progress_counts_only_target_successes_but_discloses_full_state() -> None:
    requests = [
        _synthetic_request("easy-a", "a"),
        _synthetic_request("medium-b", "b"),
        _synthetic_request("hard-c", "c"),
    ]
    request_map = [
        {
            "custom_id": "easy-a",
            "parent_case_id": "P3500-EASY-001",
            "case_id": "P3500-EASY-V2-001",
            "difficulty": "easy",
        },
        {
            "custom_id": "medium-b",
            "parent_case_id": "P3500-MEDIUM-001",
            "case_id": "P3500-MEDIUM-V2-001",
            "difficulty": "medium",
        },
        {
            "custom_id": "hard-c",
            "parent_case_id": "P3500-HARD-001",
            "case_id": "P3500-HARD-V2-001",
            "difficulty": "hard",
        },
    ]
    selection = scope_contract.select_scope_requests(
        requests,
        request_map,
        scope_contract.SCOPE_NAME,
        authority_rows=[
            {**row, "payload_sha256": requests[index]["payload_sha256"]}
            for index, row in enumerate(request_map)
        ],
    )
    state = {
        "successful_by_id": {"easy-a": {}, "hard-c": {}},
        "retry_eligible_by_id": {},
        "invalidated_by_id": {},
        "retry_started_by_id": {},
        "terminal_by_id": {},
    }

    assert scope_contract.scope_progress(state, selection) == {
        "scope_case_expected_count": 2,
        "scope_case_success_count": 1,
        "scope_remaining_case_calls": 1,
        "scope_completed_custom_ids": ["easy-a"],
        "scope_remaining_custom_ids": ["medium-b"],
        "full_case_expected_count": 3,
        "full_case_success_count": 2,
        "full_remaining_case_calls": 1,
    }


@pytest.mark.parametrize(
    ("wrapper", "expected_scope"),
    [
        (blocker_wrapper, 4),
        (contract_wrapper, 26),
    ],
)
def test_wrapper_scope_dry_run_uses_full_ledger_and_scope_progress(
    wrapper,
    expected_scope: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests, _expected_cases = wrapper._validate_requests()
    request_map = json.loads(wrapper.REQUEST_MAP_PATH.read_text(encoding="utf-8"))
    authority = (
        wrapper._scope_authority(request_map)
        if wrapper is blocker_wrapper
        else wrapper._scope_authority(requests)
    )
    selection = scope_contract.select_scope_requests(
        requests,
        request_map,
        scope_contract.SCOPE_NAME,
        authority_rows=authority,
    )
    target_ids = set(selection.target_custom_ids)
    physical_rows = runner._read_jsonl(wrapper.RESPONSES_PATH)
    prepared = json.loads(
        (wrapper.REREVIEW_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    formal_state = wrapper._formal_state(
        physical_rows,
        requests,
        execution_contract_sha256=prepared["execution_contract_sha256"],
        label="dynamic scoped wrapper test",
    )
    expected_completed = len(target_ids & set(formal_state["successful_by_id"]))
    expected_physical = sum(
        str(row.get("custom_id") or "") in target_ids for row in physical_rows
    )
    expected_retry_consumed = len(formal_state["episode_by_id"])

    assert wrapper.main(["--scope", scope_contract.SCOPE_NAME]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "dry_run_only"
    assert payload["scope_name"] == scope_contract.SCOPE_NAME
    assert payload["scope_id"] == scope_contract.SCOPE_ID
    assert payload["scope_case_expected_count"] == expected_scope
    assert payload["scope_case_success_count"] == expected_completed
    assert payload["scope_remaining_case_calls"] == expected_scope - expected_completed
    assert payload["scope_physical_response_row_count"] == expected_physical
    assert payload["materialization_adoption"][
        "global_fresh_retry_consumed_count"
    ] == expected_retry_consumed
    assert payload["hard_reference_only"] is True
    assert payload["promotable"] is False
    assert payload["baseline_registration_allowed"] is False


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_wrapper_default_scope_remains_full(wrapper) -> None:
    args = wrapper._parse_args([])
    assert args.scope == "full"


@pytest.mark.parametrize("wrapper", [blocker_wrapper, contract_wrapper])
def test_wrapper_default_full_does_not_read_scope_authority(
    wrapper,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """默认full路径不得因scope权威文件发生行为漂移。"""

    monkeypatch.setattr(
        wrapper,
        "_scope_authority",
        lambda *_args: (_ for _ in ()).throw(AssertionError("scope authority was read")),
    )
    assert wrapper.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run_only"


@pytest.mark.parametrize(
    ("wrapper", "pack_name"),
    [
        (blocker_wrapper, "blocker-rereview"),
        (contract_wrapper, "contract-rereview"),
    ],
)
@pytest.mark.parametrize("malformed_kind", ["hard_unknown", "foreign_failed"])
def test_scope_wrapper_rejects_non_target_malformed_full_ledger_event(
    wrapper,
    pack_name: str,
    malformed_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scope仍须完整拒绝Hard或外来物理账本事件。"""

    response_rows = _read_jsonl(wrapper.RESPONSES_PATH)
    if malformed_kind == "hard_unknown":
        request_map = json.loads(
            (CONTRACT_ROOT / pack_name / "request-map.json").read_text(encoding="utf-8")
        )
        hard_id = next(row["custom_id"] for row in request_map if row["difficulty"] == "hard")
        malformed = {"custom_id": hard_id, "status": "totally_unknown_status"}
    else:
        malformed = {"custom_id": "foreign-id", "status": "failed"}
    original_read_jsonl = runner._read_jsonl

    def read_jsonl(path: Path) -> list[dict]:
        if path == wrapper.RESPONSES_PATH:
            return [*response_rows, malformed]
        return original_read_jsonl(path)

    monkeypatch.setattr(runner, "_read_jsonl", read_jsonl)
    with pytest.raises(RuntimeError, match="outside current pack|unknown.*status"):
        wrapper.main(["--scope", scope_contract.SCOPE_NAME])


def _synthetic_state(rows: list[dict]) -> dict[str, dict]:
    successful = {
        str(row["custom_id"]): row
        for row in rows
        if row.get("status") == "succeeded"
    }
    return {
        "successful_by_id": successful,
        "retry_eligible_by_id": {},
        "invalidated_by_id": {},
        "retry_started_by_id": {},
        "terminal_by_id": {
            str(row["custom_id"]): row
            for row in rows
            if row.get("status") in {
                "materialization_rejected",
                "materialization_call_failed",
            }
        },
        "episode_by_id": {},
        "adopted_by_id": {},
    }


@pytest.mark.parametrize(
    ("wrapper", "pack_name"),
    [
        (blocker_wrapper, "blocker-rereview"),
        (contract_wrapper, "contract-rereview"),
    ],
)
def test_scope_execute_calls_only_remaining_easy_medium_target_and_writes_separate_summary(
    wrapper,
    pack_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack_root = CONTRACT_ROOT / pack_name
    requests = _read_jsonl(pack_root / "requests.jsonl")
    request_map = json.loads((pack_root / "request-map.json").read_text(encoding="utf-8"))
    selection = scope_contract.select_scope_requests(
        requests,
        request_map,
        scope_contract.SCOPE_NAME,
        authority_rows=_scope_authority(pack_name, requests),
    )
    remaining_id = selection.target_custom_ids[-1]
    initial_rows = [
        {
            "custom_id": custom_id,
            "status": "succeeded",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        for custom_id in selection.target_custom_ids[:-1]
    ]
    hard_id = next(
        str(row["custom_id"])
        for row in requests
        if str(row["custom_id"]) not in set(selection.target_custom_ids)
    )
    initial_rows.append(
        {"custom_id": hard_id, "status": "materialization_rejected", "attempt": 0}
    )
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in initial_rows),
        encoding="utf-8",
    )
    full_summary_path = tmp_path / "execution-summary.json"
    scoped_root = tmp_path / pack_name
    calls: list[str] = []
    original_read_json = wrapper._read_json

    monkeypatch.setattr(wrapper, "RESPONSES_PATH", responses_path)
    monkeypatch.setattr(wrapper, "SUMMARY_PATH", full_summary_path)
    monkeypatch.setattr(wrapper, "REREVIEW_ROOT", scoped_root)
    monkeypatch.setattr(wrapper, "REQUEST_MAP_PATH", pack_root / "request-map.json")
    monkeypatch.setattr(wrapper, "_validate_requests", lambda: (requests, len(requests)))
    monkeypatch.setattr(wrapper, "_original_review_complete", lambda: True)
    monkeypatch.setattr(
        wrapper,
        "_read_json",
        lambda path: (
            {"execution_contract_sha256": "e" * 64}
            if path == scoped_root / "summary.json"
            else original_read_json(path)
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "validate_rereview_request_policy",
        lambda *_args, **_kwargs: "e" * 64,
    )
    monkeypatch.setattr(wrapper, "validate_rereview_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        wrapper,
        "_formal_success_ledger",
        lambda rows, *_args, **_kwargs: _synthetic_state(rows)["successful_by_id"],
    )
    monkeypatch.setattr(
        wrapper,
        "_formal_state",
        lambda rows, *_args, **_kwargs: _synthetic_state(rows),
    )
    monkeypatch.setattr(
        wrapper,
        "rereview_calibration_ledger",
        lambda _rows, *, requests, **_kwargs: {
            str(row["custom_id"]): {} for row in requests
        },
    )
    monkeypatch.setattr(runner, "_acquire_execution_lock", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_recover_latest_saved_same_concept_variant_subset",
        lambda **_kwargs: None,
    )

    def execute_one(*, row: dict, **_kwargs) -> dict:
        calls.append(str(row["custom_id"]))
        return {
            "custom_id": str(row["custom_id"]),
            "status": "succeeded",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "completed_at": datetime.now(UTC).isoformat(),
        }

    monkeypatch.setattr(runner, "_execute_one", execute_one)
    monkeypatch.setenv("POST150_SCOPE_TEST_KEY", "test-only-key")

    code = wrapper.main(
        [
            "--scope",
            scope_contract.SCOPE_NAME,
            "--execute",
            "--acknowledge-cost",
            "--confirm-full-review",
            "--confirm-no-concurrent-review",
            "--api-key-env",
            "POST150_SCOPE_TEST_KEY",
        ]
    )

    scoped_summary_path = scope_contract.scope_summary_path(
        scoped_root,
        scope_contract.SCOPE_NAME,
    )
    summary = json.loads(scoped_summary_path.read_text(encoding="utf-8"))
    assert code == 0
    assert calls == [remaining_id]
    assert summary["status"] == "succeeded"
    assert summary["case_success_count"] == selection.target_case_count
    assert summary["case_expected_count"] == selection.target_case_count
    assert summary["scope_remaining_case_calls"] == 0
    assert summary["full_remaining_case_calls"] == len(requests) - selection.target_case_count
    assert not full_summary_path.exists()
