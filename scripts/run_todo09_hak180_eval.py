"""运行 TODO-09 HAK180 retrieval-only 成对评测。

该脚本只创建一条 candidate evaluation run，不修改数据集、文档、索引或
RetrievalProfile。默认以仓库已记录的权威 retrieval-only baseline 做 paired diff。
"""

from __future__ import annotations

import argparse
import json

from app.application.evaluation_service import (
    create_run,
    execute_run,
    get_run,
    report_path,
)
from app.security.principal import Principal, reset_current_principal, set_current_principal


DEFAULT_DATASET_ID = "eds_cff7b6f0f1cfc994a5731543a1c00e33"
DEFAULT_KB_ID = "kb_1fd79001176c412d8699ee438a8db000"
DEFAULT_BASELINE_RUN_ID = "erun_872af04ef9a64570a6dfadcf3166153d"
DEFAULT_USER_ID = "usr_f88fe37c7414489891723478f0102337"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID)
    parser.add_argument("--baseline-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--tenant-id", default="tenant_default")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--name", default="TODO-09 RetrievalProfile candidate 2026-07-24")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    principal = Principal(
        user_id=args.user_id,
        tenant_id=args.tenant_id,
        roles=("TenantAdmin",),
        scopes=(),
        auth_type="local-evaluation",
        credential_id="todo09-eval-script",
    )
    token = set_current_principal(principal)
    try:
        run = create_run(
            dataset_id=args.dataset_id,
            kb_id=args.kb_id,
            mode="retrieval_only",
            name=args.name,
            baseline_run_id=args.baseline_run_id,
            case_ids=None,
        )
        execute_run(run["id"], principal)
        completed = get_run(run["id"])
        comparison_file = report_path(run["id"], "comparison")
        comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "run": completed,
                    "comparison": comparison,
                    "comparison_file": str(comparison_file),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if completed.get("status") == "succeeded" else 1
    finally:
        reset_current_principal(token)


if __name__ == "__main__":
    raise SystemExit(main())
