#!/usr/bin/env python3
"""对 easy/medium scoped 契约执行确定性、零网络自检。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import run_easy_medium_live as live


def _row(case_id: str, correct: bool) -> dict:
    return {
        "run_id": "erun_scoped_self_check",
        "case_id": case_id,
        "passed": not correct,
        "metrics": {
            "metrics": {
                "case_outcome": {
                    "final_answer_correct": correct,
                    "strict_supported_case_pass": not correct,
                }
            }
        },
    }


def _contract_check() -> dict:
    observed: dict[str, str] = {}
    for difficulty in live.DIFFICULTIES:
        case_ids = [f"{difficulty.upper()}-{number:03d}" for number in range(1, 51)]
        minimum = live._minimum_correct(difficulty)
        rows = [
            _row(case_id, position <= minimum)
            for position, case_id in enumerate(case_ids, start=1)
        ]
        target = live._target_count_from_rows(
            rows,
            expected_run_id="erun_scoped_self_check",
            expected_case_ids=case_ids,
        )
        assert target["correct_count"] == minimum
        assert live._entry_target_passed({"target_result": target}, difficulty)
        target["correct_count"] = minimum - 1
        assert not live._entry_target_passed({"target_result": target}, difficulty)
        observed[difficulty] = f"{minimum}/50"
    profile = live.scope.scope_profile()
    assert tuple(profile["difficulties"]) == live.DIFFICULTIES
    assert profile["hard_reference_only"] is True
    assert profile["promotable"] is False
    assert profile["canonical_promotion_allowed"] is False
    assert profile["baseline_registration_allowed"] is False
    return {
        "contract_check": "passed",
        "scope": profile,
        "suite_key": live.EXPECTED_SUITE_KEY,
        "dataset_version": live.EXPECTED_DATASET_VERSION,
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "target_gates": observed,
        "promotion_allowed": False,
        "network_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="仅运行合成契约检查，不读取正式 activation",
    )
    args = parser.parse_args(argv)
    result = _contract_check()
    if args.contract_only:
        result.update(
            {
                "status": "contract_only_self_test_passed",
                "real_scope_validated": False,
            }
        )
    else:
        try:
            suite = live.load_suite_snapshot(live.FINAL_ROOT)
        except (live.ScopedLiveError, live.full.FinalLiveError) as exc:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reason": str(exc),
                        "real_scope_validated": False,
                        "network_calls": 0,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        result.update(
            {
                "status": "real_scope_self_check_passed",
                "real_scope_validated": True,
                "activation_sha256": suite.activation_sha256,
                "suite_identity_sha256": suite.suite_identity_sha256,
                "quality_targets_sha256": suite.quality_targets_sha256,
                "static_checksums_sha256": suite.static_checksums_sha256,
                "case_counts": {
                    difficulty: suite.datasets[difficulty].case_count
                    for difficulty in live.DIFFICULTIES
                },
                "source_index_identity": suite.source_index_identity,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
