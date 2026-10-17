"""运行离线 RAG 指标与可选的 Ragas 0.4.3 Judge。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.ragas_support import (  # noqa: E402
    DEFAULT_JUDGE_METRICS,
    JudgeConfig,
    PricingConfig,
    evaluate_records,
    load_jsonl,
    write_metric_records,
)
from evals.ragas_support.runner import RagasRunnerError  # noqa: E402


def _load_judge_case_ids(path: Path) -> list[str]:
    """读取 JSON 数组、带 case_ids 的 JSON 对象或逐行 case_id。"""

    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("case_ids")
    if not isinstance(payload, list) or not all(
        isinstance(case_id, str) and case_id.strip() for case_id in payload
    ):
        raise RagasRunnerError(
            "judge sample list must be a JSON string array, a case_ids object, or one ID per line"
        )
    return [case_id.strip() for case_id in payload]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 samples.jsonl 执行零调用离线指标，并可显式启用 Ragas Judge",
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        action="append",
        default=[],
        help="按 case_id 与 samples 左连接的系统 observation/result JSONL；可重复",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--allow-paid-model-calls", action="store_true")
    parser.add_argument(
        "--judge-case-id",
        action="append",
        default=None,
        help="显式 Judge 采样 case_id；可重复，默认不选择任何题",
    )
    parser.add_argument(
        "--judge-sample-list",
        type=Path,
        default=None,
        help="显式 Judge 采样清单；支持 JSON 数组、case_ids 对象或逐行 ID",
    )
    parser.add_argument("--judge-max-cases", type=int, default=24)
    parser.add_argument("--judge-max-cost-cny", type=float, default=10.0)
    parser.add_argument(
        "--judge-metric",
        action="append",
        default=None,
        help="可重复；仅支持经审计的 Ragas 0.4.3 Collections 指标",
    )
    parser.add_argument("--judge-model", default="qwen3.7-plus")
    parser.add_argument("--judge-embedding-model", default="text-embedding-v4")
    parser.add_argument("--judge-max-output-tokens", type=int, default=2000)
    parser.add_argument(
        "--judge-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--judge-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--judge-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--aliyun-input-price-per-million", type=float, default=2.0)
    parser.add_argument("--aliyun-output-price-per-million", type=float, default=8.0)
    parser.add_argument("--aliyun-embedding-price-per-million", type=float, default=0.5)
    parser.add_argument("--estimated-input-tokens-per-call", type=int, default=8000)
    parser.add_argument("--estimated-output-tokens-per-call", type=int, default=2000)
    parser.add_argument("--estimated-embedding-tokens-per-case", type=int, default=2000)
    parser.add_argument("--currency", default="CNY")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metrics = tuple(args.judge_metric or DEFAULT_JUDGE_METRICS)
        judge_case_ids = list(args.judge_case_id or [])
        if args.judge_sample_list is not None:
            judge_case_ids.extend(_load_judge_case_ids(args.judge_sample_list))
        pricing = PricingConfig(
            input_price_per_million=args.aliyun_input_price_per_million,
            output_price_per_million=args.aliyun_output_price_per_million,
            embedding_price_per_million=args.aliyun_embedding_price_per_million,
            estimated_input_tokens_per_call=args.estimated_input_tokens_per_call,
            estimated_output_tokens_per_call=args.estimated_output_tokens_per_call,
            estimated_embedding_tokens_per_case=args.estimated_embedding_tokens_per_case,
            currency=args.currency,
        )
        judge = JudgeConfig(
            enabled=args.judge,
            allow_paid_model_calls=args.allow_paid_model_calls,
            metrics=metrics,
            sample_case_ids=tuple(judge_case_ids),
            max_cases=args.judge_max_cases,
            max_cost_cny=args.judge_max_cost_cny,
            model=args.judge_model,
            embedding_model=args.judge_embedding_model,
            max_output_tokens=args.judge_max_output_tokens,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
            timeout_seconds=args.judge_timeout_seconds,
        )
        samples = load_jsonl(args.samples)
        observations = [
            row
            for observations_path in args.observations
            for row in load_jsonl(observations_path)
        ]
        records = evaluate_records(
            samples,
            observations=observations,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            judge=judge,
            pricing=pricing,
        )
        if args.output is not None:
            write_metric_records(args.output, records)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "record_count": len(records),
                        "output": str(args.output.resolve()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            for record in records:
                print(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
        return 0
    except (OSError, ValueError, RagasRunnerError) as exc:
        print(f"Ragas evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
