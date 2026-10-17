#!/usr/bin/env python3
"""Deterministic no-network smoke checks for the final baseline contracts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import run_final_live as live


def _row(case_id: str, correct: bool) -> dict:
    return {
        "run_id": "erun_self_check",
        "case_id": case_id,
        "passed": not correct,  # strict/legacy fields must not affect the target
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
    case_ids = [f"CASE-{number:03d}" for number in range(1, 51)]
    rows = [_row(case_id, position <= 48) for position, case_id in enumerate(case_ids, 1)]
    target = live._target_count_from_rows(
        rows,
        expected_run_id="erun_self_check",
        expected_case_ids=case_ids,
    )
    assert target["correct_count"] == 48
    assert target["case_count"] == 50
    assert target["strict_metrics_counted"] is False
    index = {
        "difficulties": {
            "easy": {"status": "archived", "target_result": target},
        }
    }
    assert live._entry_target_passed(index["difficulties"]["easy"], "easy")
    index["difficulties"]["easy"]["target_result"]["correct_count"] = 47
    assert not live._entry_target_passed(index["difficulties"]["easy"], "easy")
    return {
        "contract_check": "passed",
        "suite_key": live.EXPECTED_SUITE_KEY,
        "dataset_version": live.EXPECTED_DATASET_VERSION,
        "authoritative_metric": live.AUTHORITATIVE_METRIC,
        "easy_gate": "48/50",
        "medium_gate": "43/50",
        "hard_gate": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="run only synthetic no-network contract checks before activation exists",
    )
    args = parser.parse_args(argv)
    result = _contract_check()
    if args.contract_only:
        result.update(
            {
                "status": "contract_only_self_test_passed",
                "real_suite_validated": False,
                "network_calls": 0,
            }
        )
    else:
        try:
            suite = live.load_suite_snapshot(live.FINAL_ROOT)
        except live.FinalLiveError as exc:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reason": str(exc),
                        "real_suite_validated": False,
                        "hint": "use --contract-only only for the explicit synthetic check",
                        "network_calls": 0,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        result.update(
            {
                "status": "real_suite_self_check_passed",
                "real_suite_validated": True,
                "activation_sha256": suite.activation_sha256,
                "suite_identity_sha256": suite.suite_identity_sha256,
                "quality_targets_sha256": suite.quality_targets_sha256,
                "static_checksums_sha256": suite.static_checksums_sha256,
                "case_counts": {
                    difficulty: suite.datasets[difficulty].case_count
                    for difficulty in live.DIFFICULTIES
                },
                "source_index_identity": suite.source_index_identity,
                "network_calls": 0,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
