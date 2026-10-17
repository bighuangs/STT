"""构建 RAGAS 全系统评测所需的规范化数据资产。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.ragas_support.dataset import build_ragas_dataset  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把三难度冻结集与可选 HAK180 v1.4 规范化为 RAGAS 数据资产",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="包含 easy.json、medium.json、hard.json 的目录",
    )
    parser.add_argument(
        "--hak180",
        type=Path,
        default=None,
        help="可选的 HAK180 v1.4 数据集 JSON",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换输出目录中的同名评测资产",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_paths = [
        args.datasets_dir / "easy.json",
        args.datasets_dir / "medium.json",
        args.datasets_dir / "hard.json",
    ]
    if args.hak180 is not None:
        dataset_paths.append(args.hak180)
    stats = build_ragas_dataset(
        dataset_paths,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats["totals"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
