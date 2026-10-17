"""只读预检入口：验证配置、场景与数据集，不执行正式评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.configs import load_harness_config
from evals.datasets import load_dataset_asset
from evals.scenarios import load_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 STT 评测 harness 资产")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = load_harness_config(args.config) if args.config is not None else load_harness_config()
    dataset = load_dataset_asset(
        config.dataset_id,
        project_root=args.project_root,
    )
    scenario = load_scenario(config.scenario_id)
    print(
        json.dumps(
            {
                "config_id": config.config_id,
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.sha256,
                "scenario_id": scenario.scenario_id,
                "formal_evaluation": config.formal_evaluation,
                "status": "validated",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
