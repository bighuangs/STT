"""从正式评测结果与本地 Milvus 物化 Ragas Judge 的最小输入。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.shared.clients.milvus_utils import get_milvus_client  # noqa: E402
from evals.ragas_support.materializer import (  # noqa: E402
    ContextMaterializationError,
    ObservationSource,
    materialize_judge_observations,
    write_materialized_observations,
)
from evals.ragas_support.runner import load_jsonl  # noqa: E402
from scripts.run_ragas_evaluation import _load_judge_case_ids  # noqa: E402


def _parse_source(value: str) -> tuple[Path, str]:
    path_text, separator, collection_name = value.rpartition("=")
    if not separator or not path_text.strip() or not collection_name.strip():
        raise argparse.ArgumentTypeError(
            "observation source must use RESULTS_JSONL=COLLECTION_NAME"
        )
    return Path(path_text.strip()), collection_name.strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按内容哈希从本地 Milvus 回填显式 Judge 样本的上下文正文",
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--judge-sample-list", type=Path, required=True)
    parser.add_argument(
        "--observation-source",
        action="append",
        type=_parse_source,
        required=True,
        help="可重复；格式为 RESULTS_JSONL=COLLECTION_NAME",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        client = get_milvus_client()
        if client is None:
            raise ContextMaterializationError("Milvus client is unavailable")
        sources = [
            ObservationSource(collection_name=name, rows=load_jsonl(path))
            for path, name in args.observation_source
        ]
        rows = materialize_judge_observations(
            load_jsonl(args.samples),
            sources,
            case_ids=_load_judge_case_ids(args.judge_sample_list),
            client=client,
        )
        summary = write_materialized_observations(
            args.output,
            rows,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {"status": "completed", "output": str(args.output.resolve()), **summary},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, ContextMaterializationError) as exc:
        print(f"Ragas context materialization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
