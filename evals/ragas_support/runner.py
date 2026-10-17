"""Ragas 0.4.3 的隐私收敛评测运行器。

离线路径不导入 Ragas，也不发起网络请求。只有显式开启 Judge 且再次确认付费
调用后，才延迟加载 Ragas Collections API。持久化结果采用严格白名单，原始问题、
回答、参考答案、上下文和 Judge reasoning 均只存在于进程内存中。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import types
from typing import Any
from urllib.parse import quote

from .dataset import SAMPLE_SCHEMA_VERSION

ABSOLUTE_JUDGE_CASE_CAP = 24
ABSOLUTE_JUDGE_BUDGET_CNY = 10.0
ALLOWED_JUDGE_METRICS = frozenset(
    {
        "answer_correctness",
        "answer_relevancy",
        "context_precision_with_reference",
        "context_recall",
        "factual_correctness",
        "faithfulness",
    }
)
DEFAULT_JUDGE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
)
_FORBIDDEN_JUDGE_NAMES = frozenset(
    {
        "multimodalfaithfulness",
        "multi_modal_faithfulness",
        "multi-modal-faithfulness",
    }
)
_OUTPUT_FIELDS = frozenset(
    {"metric", "value", "status", "error", "latency", "usage", "hash"}
)
_SAFE_ERROR_TOKEN = re.compile(r"[^a-zA-Z0-9_.:-]+")


class RagasRunnerError(RuntimeError):
    """评测输入、权限或可选依赖不满足运行契约。"""


@dataclass(frozen=True, slots=True)
class PricingConfig:
    """阿里云兼容模型的参数化费用估算。"""

    input_price_per_million: float | None = 2.0
    output_price_per_million: float | None = 8.0
    embedding_price_per_million: float | None = 0.5
    estimated_input_tokens_per_call: int = 8000
    estimated_output_tokens_per_call: int = 2000
    estimated_embedding_tokens_per_case: int = 2000
    currency: str = "CNY"

    def __post_init__(self) -> None:
        for name, value in (
            ("input_price_per_million", self.input_price_per_million),
            ("output_price_per_million", self.output_price_per_million),
            ("embedding_price_per_million", self.embedding_price_per_million),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.estimated_input_tokens_per_call < 0:
            raise ValueError("estimated_input_tokens_per_call must be non-negative")
        if self.estimated_output_tokens_per_call < 0:
            raise ValueError("estimated_output_tokens_per_call must be non-negative")
        if self.estimated_embedding_tokens_per_case < 0:
            raise ValueError("estimated_embedding_tokens_per_case must be non-negative")
        if not self.currency.strip():
            raise ValueError("currency must not be empty")

    def usage_for_calls(
        self,
        llm_calls: int,
        *,
        embedding_requests: int = 0,
        embedding_tokens: int = 0,
    ) -> dict[str, Any]:
        """返回逻辑调用量、token 与费用估算；不伪装成实际账单。"""

        input_tokens = llm_calls * self.estimated_input_tokens_per_call
        output_tokens = llm_calls * self.estimated_output_tokens_per_call
        cost: float | None = None
        if (
            self.input_price_per_million is not None
            and self.output_price_per_million is not None
        ):
            cost = (
                input_tokens * self.input_price_per_million
                + output_tokens * self.output_price_per_million
            ) / 1_000_000
            if embedding_tokens:
                if self.embedding_price_per_million is None:
                    cost = None
                else:
                    cost += (
                        embedding_tokens * self.embedding_price_per_million
                    ) / 1_000_000
        return {
            "kind": "estimate",
            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "embedding_requests": embedding_requests,
            "embedding_tokens": embedding_tokens,
            "external_requests": llm_calls + embedding_requests,
            "cost": cost,
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """显式付费 Judge 的最小配置，不包含或持久化 API key。"""

    enabled: bool = False
    allow_paid_model_calls: bool = False
    metrics: tuple[str, ...] = DEFAULT_JUDGE_METRICS
    sample_case_ids: tuple[str, ...] = ()
    max_cases: int = ABSOLUTE_JUDGE_CASE_CAP
    max_cost_cny: float = ABSOLUTE_JUDGE_BUDGET_CNY
    model: str = "qwen3.7-plus"
    embedding_model: str = "text-embedding-v4"
    max_output_tokens: int = 2000
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        validate_judge_metrics(self.metrics)
        if len(set(self.sample_case_ids)) != len(self.sample_case_ids):
            raise ValueError("judge sample_case_ids must be unique")
        if not 1 <= self.max_cases <= ABSOLUTE_JUDGE_CASE_CAP:
            raise ValueError(
                f"judge max_cases must be between 1 and {ABSOLUTE_JUDGE_CASE_CAP}"
            )
        if (
            not math.isfinite(self.max_cost_cny)
            or self.max_cost_cny <= 0
            or self.max_cost_cny > ABSOLUTE_JUDGE_BUDGET_CNY
        ):
            raise ValueError(
                f"judge max_cost_cny must be within (0, {ABSOLUTE_JUDGE_BUDGET_CNY}]"
            )
        if not self.model.strip():
            raise ValueError("judge model must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("judge embedding_model must not be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("judge max_output_tokens must be positive")
        if not self.base_url.strip():
            raise ValueError("judge base_url must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("judge api_key_env must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("judge timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class _ScorePoint:
    metric: str
    value: float | None
    status: str
    difficulty: str
    tags: tuple[str, ...]
    sample_hash: str


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_path_component(value: str) -> str:
    return quote(value, safe="-_.~")


def _empty_usage() -> dict[str, Any]:
    return {
        "kind": "measured",
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "embedding_requests": 0,
        "embedding_tokens": 0,
        "external_requests": 0,
        "cost": 0.0,
        "currency": None,
    }


def _record(
    metric: str,
    *,
    value: float | int | None,
    status: str,
    error: str | None,
    latency_ms: float = 0.0,
    usage: Mapping[str, Any] | None = None,
    digest: str,
) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value):
        value = None
        status = "undefined"
        error = error or "non_finite_metric_value"
    return {
        "metric": metric,
        "value": value,
        "status": status,
        "error": error,
        "latency": {"milliseconds": round(max(0.0, latency_ms), 3)},
        "usage": dict(usage or _empty_usage()),
        "hash": digest,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL，并对坏行和非对象给出确定位置。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagasRunnerError(
                    f"invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise RagasRunnerError(
                    f"JSONL row at {path}:{line_number} must be an object"
                )
            rows.append(payload)
    return rows


def _case_id(row: Mapping[str, Any], *, source: str) -> str:
    value = row.get("case_id")
    if not isinstance(value, str) or not value.strip():
        raise RagasRunnerError(f"{source} row is missing a non-empty case_id")
    return value.strip()


def _first_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _string_value(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    value = _first_value(row, names)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _string_sequence(value: Any) -> tuple[list[str] | None, str | None]:
    if value is None:
        return None, "missing"
    if not isinstance(value, (list, tuple)):
        return None, "invalid_type"
    normalized: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            return None, "invalid_item_type"
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized, None


def _text_sequence(value: Any) -> tuple[list[str] | None, str | None]:
    if value is None:
        return None, "missing"
    if not isinstance(value, (list, tuple)):
        return None, "invalid_type"
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None, "invalid_item_type"
        if item.strip():
            normalized.append(item)
    return normalized, None


def _nested_mapping(row: Mapping[str, Any], *names: str) -> dict[str, Any]:
    merged = dict(row)
    for name in names:
        nested = row.get(name)
        if isinstance(nested, Mapping):
            merged.update(nested)
    return merged


def _mapping_item_value(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    value = _first_value(item, names)
    if value is not None:
        return value
    evidence = item.get("evidence")
    if isinstance(evidence, Mapping):
        value = _first_value(evidence, names)
        if value is not None:
            return value
    source = item.get("source")
    if isinstance(source, Mapping):
        return _first_value(source, names)
    return None


def _extract_ids_from_items(items: Any) -> list[str] | None:
    if not isinstance(items, list):
        return None
    values: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            value = _mapping_item_value(
                item,
                ("chunk_id", "source_chunk_id", "context_id", "evidence_id", "id"),
            )
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                values.append(str(value))
    return values


def _extract_texts_from_items(items: Any) -> list[str] | None:
    if not isinstance(items, list):
        return None
    values: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            values.append(item)
        elif isinstance(item, Mapping):
            value = _mapping_item_value(item, ("text", "content", "source_quote"))
            if isinstance(value, str) and value.strip():
                values.append(value)
    return values


def _flatten_existing_metrics(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, float | None, str]]:
    if isinstance(value, bool):
        if prefix:
            yield prefix, float(value), "ok"
        return
    if isinstance(value, (int, float)):
        if prefix:
            number = float(value)
            yield prefix, number if math.isfinite(number) else None, (
                "ok" if math.isfinite(number) else "undefined"
            )
        return
    if value is None:
        if prefix:
            yield prefix, None, "missing"
        return
    if not isinstance(value, Mapping):
        return
    for key in sorted(value, key=str):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        yield from _flatten_existing_metrics(value[key], prefix=child_prefix)


def _observation_existing_metrics(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = row.get("existing_metrics")
    if isinstance(direct, Mapping):
        return direct
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get("metrics")
        return nested if isinstance(nested, Mapping) else metrics
    grades = row.get("grades")
    if isinstance(grades, list):
        extracted: dict[str, Any] = {}
        for grade in grades:
            if not isinstance(grade, Mapping):
                continue
            name = grade.get("grader") or grade.get("name")
            if isinstance(name, str) and name:
                extracted[name] = dict(grade)
        return extracted
    return None


def _normalize_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = _nested_mapping(row, "result", "output", "observation")
    normalized: dict[str, Any] = {}

    response = _string_value(merged, ("response", "answer", "final_answer"))
    if response is not None:
        normalized["response"] = response

    explicit_ids = _first_value(
        merged, ("retrieved_context_ids", "retrieved_chunk_ids")
    )
    ids, ids_error = _string_sequence(explicit_ids)
    if ids_error == "missing":
        for field in ("reranked", "retrieved", "context"):
            ids = _extract_ids_from_items(merged.get(field))
            if ids is not None:
                break
    elif ids_error is not None:
        normalized["retrieved_context_ids_error"] = ids_error
    if ids is not None:
        normalized["retrieved_context_ids"] = ids

    explicit_contexts = merged.get("retrieved_contexts")
    contexts, contexts_error = _text_sequence(explicit_contexts)
    if contexts_error == "missing":
        for field in ("context", "reranked", "retrieved"):
            contexts = _extract_texts_from_items(merged.get(field))
            if contexts is not None:
                break
    elif contexts_error is not None:
        normalized["retrieved_contexts_error"] = contexts_error
    if contexts is not None:
        normalized["retrieved_contexts"] = contexts

    existing = _observation_existing_metrics(row)
    if existing is not None:
        normalized["existing_metrics"] = dict(existing)
    usage = _first_value(merged, ("usage", "token_usage"))
    if isinstance(usage, Mapping):
        normalized["usage"] = dict(usage)

    error_value = row.get("error_type") or row.get("error") or merged.get("error")
    status = row.get("status") or merged.get("status")
    if error_value or status in {"error", "failed"}:
        normalized["source_error"] = _safe_source_error(error_value)
    return normalized


def _safe_source_error(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("code") or value.get("type") or "source_error"
    if not isinstance(value, str) or not value.strip():
        return "source_error"
    token = _SAFE_ERROR_TOKEN.sub("_", value.strip())[:80]
    return f"source_error:{token}" if token else "source_error"


def join_observations(
    samples: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """按 case_id 左连接系统结果；未知 observation 计数但不污染样本。"""

    by_case: dict[str, Mapping[str, Any]] = {}
    for observation in observations:
        case_id = _case_id(observation, source="observation")
        if case_id in by_case:
            raise RagasRunnerError(f"duplicate observation case_id: {case_id}")
        by_case[case_id] = observation

    joined: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for raw_sample in samples:
        case_id = _case_id(raw_sample, source="sample")
        if case_id in sample_ids:
            raise RagasRunnerError(f"duplicate sample case_id: {case_id}")
        sample_ids.add(case_id)
        schema_version = raw_sample.get("schema_version")
        if schema_version not in {None, SAMPLE_SCHEMA_VERSION}:
            raise RagasRunnerError(
                f"unsupported sample schema for {case_id}: {schema_version}"
            )
        sample = dict(raw_sample)
        observation = by_case.get(case_id)
        if observation is not None:
            sample.update(_normalize_observation(observation))
        joined.append(sample)
    unknown_count = len(set(by_case) - sample_ids)
    return joined, unknown_count


def _sample_hash(sample: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "case_id": sample.get("case_id"),
            "dataset_version": sample.get("dataset_version"),
            "user_input": _string_value(sample, ("user_input", "query")),
            "reference": _string_value(sample, ("reference", "reference_answer")),
            "response": _string_value(sample, ("response", "answer")),
            "reference_context_ids": _first_value(
                sample, ("reference_context_ids", "relevant_chunk_ids")
            ),
            "retrieved_context_ids": _first_value(
                sample, ("retrieved_context_ids", "retrieved_chunk_ids")
            ),
            "retrieved_contexts": sample.get("retrieved_contexts"),
        }
    )


def _sample_slices(sample: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    difficulty = sample.get("difficulty")
    if not isinstance(difficulty, str) or not difficulty.strip():
        metadata = sample.get("metadata")
        difficulty = (
            metadata.get("difficulty") if isinstance(metadata, Mapping) else "unknown"
        )
    difficulty = str(difficulty or "unknown").strip().lower()
    tags_value = sample.get("tags")
    tags = tuple(
        sorted(
            {
                str(tag).strip()
                for tag in tags_value
                if isinstance(tag, (str, int)) and str(tag).strip()
            }
        )
    ) if isinstance(tags_value, list) else ()
    return difficulty, tags


def _source_error(sample: Mapping[str, Any]) -> str | None:
    value = sample.get("source_error")
    return value if isinstance(value, str) and value else None


def _id_metric(
    sample: Mapping[str, Any],
    *,
    precision: bool,
) -> tuple[float | None, str, str | None]:
    source_error = _source_error(sample)
    if source_error:
        return None, "error", source_error
    retrieved, retrieved_error = _string_sequence(
        _first_value(sample, ("retrieved_context_ids", "retrieved_chunk_ids"))
    )
    reference, reference_error = _string_sequence(
        _first_value(sample, ("reference_context_ids", "relevant_chunk_ids"))
    )
    if retrieved_error is not None:
        return None, "missing", f"retrieved_context_ids:{retrieved_error}"
    if reference_error is not None:
        return None, "missing", f"reference_context_ids:{reference_error}"
    assert retrieved is not None and reference is not None
    retrieved_set = set(retrieved)
    reference_set = set(reference)
    denominator = len(retrieved_set if precision else reference_set)
    if denominator == 0:
        field = "retrieved_context_ids" if precision else "reference_context_ids"
        return None, "undefined", f"{field}:empty_denominator"
    hits = len(retrieved_set & reference_set)
    return hits / denominator, "ok", None


def _case_records_and_points(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[_ScorePoint]]:
    records: list[dict[str, Any]] = []
    points: list[_ScorePoint] = []
    for sample in samples:
        case_id = _case_id(sample, source="sample")
        case_path = _safe_path_component(case_id)
        digest = _sample_hash(sample)
        difficulty, tags = _sample_slices(sample)
        for name, precision in (
            ("id_based_context_precision", True),
            ("id_based_context_recall", False),
        ):
            value, status, error = _id_metric(sample, precision=precision)
            records.append(
                _record(
                    f"case/{case_path}/{name}",
                    value=value,
                    status=status,
                    error=error,
                    digest=digest,
                )
            )
            points.append(
                _ScorePoint(name, value, status, difficulty, tags, digest)
            )

        existing = sample.get("existing_metrics")
        if isinstance(existing, Mapping):
            for metric_name, value, status in _flatten_existing_metrics(existing):
                name = f"existing/{metric_name}"
                error = None if status == "ok" else "existing_metric_value_missing"
                records.append(
                    _record(
                        f"case/{case_path}/{_safe_path_component(name)}",
                        value=value,
                        status=status,
                        error=error,
                        digest=digest,
                    )
                )
                points.append(
                    _ScorePoint(name, value, status, difficulty, tags, digest)
                )
    return records, points


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 20260828,
) -> tuple[float, float, float]:
    """以固定随机种子的 percentile bootstrap 计算均值与 95% CI。"""

    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        raise ValueError("bootstrap requires at least one finite value")
    if resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, mean, mean
    generator = random.Random(seed)
    sample_size = len(clean)
    means = sorted(
        sum(generator.choice(clean) for _ in range(sample_size)) / sample_size
        for _ in range(resamples)
    )
    return mean, _percentile(means, 0.025), _percentile(means, 0.975)


def _aggregate_records(
    points: Sequence[_ScorePoint],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[_ScorePoint]] = {}
    for point in points:
        slices = ["overall", f"difficulty={point.difficulty}"]
        slices.extend(f"tag={tag}" for tag in point.tags)
        for slice_name in slices:
            groups.setdefault((slice_name, point.metric), []).append(point)

    records: list[dict[str, Any]] = []
    for index, ((slice_name, metric), group) in enumerate(sorted(groups.items())):
        valid = [
            point.value
            for point in group
            if point.status == "ok" and point.value is not None
        ]
        group_hash = _stable_hash(
            {
                "slice": slice_name,
                "metric": metric,
                "samples": [point.sample_hash for point in group],
            }
        )
        base = (
            f"aggregate/{_safe_path_component(slice_name)}/"
            f"{_safe_path_component(metric)}"
        )
        coverage = len(valid) / len(group) if group else 0.0
        for stat, value in (
            ("n_total", len(group)),
            ("n_valid", len(valid)),
            ("coverage", coverage),
        ):
            records.append(
                _record(
                    f"{base}/{stat}",
                    value=value,
                    status="ok",
                    error=None,
                    digest=group_hash,
                )
            )
        if not valid:
            for stat in ("mean", "ci95_low", "ci95_high"):
                records.append(
                    _record(
                        f"{base}/{stat}",
                        value=None,
                        status="missing",
                        error="no_valid_values",
                        digest=group_hash,
                    )
                )
            continue
        metric_seed = seed + index
        mean, low, high = bootstrap_mean_ci(
            valid,
            resamples=resamples,
            seed=metric_seed,
        )
        for stat, value in (
            ("mean", mean),
            ("ci95_low", low),
            ("ci95_high", high),
        ):
            records.append(
                _record(
                    f"{base}/{stat}",
                    value=value,
                    status="ok",
                    error=None,
                    digest=group_hash,
                )
            )
    return records


def _normalized_judge_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "response_relevancy":
        return "answer_relevancy"
    return normalized


def validate_judge_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """拒绝多模态 Faithfulness，并只允许经审计的 Collections 指标。"""

    normalized: list[str] = []
    for raw_name in metrics:
        name = _normalized_judge_name(str(raw_name))
        compact = name.replace("_", "")
        if name in _FORBIDDEN_JUDGE_NAMES or compact == "multimodalfaithfulness":
            raise RagasRunnerError("MultiModalFaithfulness is forbidden")
        if name not in ALLOWED_JUDGE_METRICS:
            raise RagasRunnerError(f"unsupported Judge metric: {raw_name}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise RagasRunnerError("at least one Judge metric is required")
    return tuple(normalized)


def _judge_inputs(
    sample: Mapping[str, Any],
    metric: str,
) -> tuple[dict[str, Any] | None, int, str | None]:
    sensitivity = str(sample.get("sensitivity") or "").strip().lower()
    if (
        sample.get("protected") is True
        or sensitivity in {"protected", "restricted"}
        or sample.get("judge_eligible") is not True
    ):
        return None, 0, "judge_not_eligible"
    if _source_error(sample):
        return None, 0, "source_error"
    user_input = _string_value(sample, ("user_input", "query"))
    response = _string_value(sample, ("response", "answer"))
    reference = _string_value(sample, ("reference", "reference_answer"))
    contexts, contexts_error = _text_sequence(sample.get("retrieved_contexts"))

    if metric == "faithfulness":
        if user_input is None or response is None or contexts_error or not contexts:
            return None, 0, "missing_judge_input"
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": contexts,
        }, 2, None
    if metric == "answer_correctness":
        if user_input is None or response is None or reference is None:
            return None, 0, "missing_judge_input"
        return {
            "user_input": user_input,
            "response": response,
            "reference": reference,
        }, 3, None
    if metric == "answer_relevancy":
        if user_input is None or response is None:
            return None, 0, "missing_judge_input"
        return {
            "user_input": user_input,
            "response": response,
        }, 3, None
    if metric == "factual_correctness":
        if response is None or reference is None:
            return None, 0, "missing_judge_input"
        return {
            "response": response,
            "reference": reference,
        }, 4, None
    if metric == "context_precision_with_reference":
        if user_input is None or reference is None or contexts_error or not contexts:
            return None, 0, "missing_judge_input"
        return {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": contexts,
        }, len(contexts), None
    if metric == "context_recall":
        if user_input is None or reference is None or contexts_error or not contexts:
            return None, 0, "missing_judge_input"
        return {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": contexts,
        }, 1, None
    raise RagasRunnerError(f"unsupported Judge metric: {metric}")


def estimate_judge_usage(
    samples: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_JUDGE_METRICS,
    pricing: PricingConfig | None = None,
    case_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """仅对显式采样清单估算 Ragas 逻辑调用量与阿里云费用。"""

    pricing = pricing or PricingConfig()
    selected = validate_judge_metrics(metrics)
    requested_ids = tuple(str(case_id).strip() for case_id in case_ids if str(case_id).strip())
    if len(set(requested_ids)) != len(requested_ids):
        raise RagasRunnerError("judge sampling list contains duplicate case_id")
    by_case = {_case_id(sample, source="sample"): sample for sample in samples}
    unknown = [case_id for case_id in requested_ids if case_id not in by_case]
    if unknown:
        raise RagasRunnerError(
            "judge sampling list contains unknown case_id: " + ", ".join(unknown)
        )
    selected_samples = [by_case[case_id] for case_id in requested_ids]
    calls_by_metric = {metric: 0 for metric in selected}
    embedding_requests_by_metric = {metric: 0 for metric in selected}
    embedding_tokens_by_metric = {metric: 0 for metric in selected}
    eligible_by_metric = {metric: 0 for metric in selected}
    for sample in selected_samples:
        for metric in selected:
            inputs, calls, _ = _judge_inputs(sample, metric)
            if inputs is not None:
                eligible_by_metric[metric] += 1
                calls_by_metric[metric] += calls
                if metric == "answer_relevancy":
                    embedding_requests_by_metric[metric] += 2
                    embedding_tokens_by_metric[metric] += (
                        pricing.estimated_embedding_tokens_per_case
                    )

    config_hash = _stable_hash(
        {
            "metrics": selected,
            "llm_calls": calls_by_metric,
            "pricing": {
                "input": pricing.input_price_per_million,
                "output": pricing.output_price_per_million,
                "embedding": pricing.embedding_price_per_million,
                "input_tokens": pricing.estimated_input_tokens_per_call,
                "output_tokens": pricing.estimated_output_tokens_per_call,
                "embedding_tokens": pricing.estimated_embedding_tokens_per_case,
                "currency": pricing.currency,
            },
            "case_ids": requested_ids,
        }
    )
    records: list[dict[str, Any]] = []
    records.append(
        _record(
            "estimate/aliyun/selection/cases",
            value=len(selected_samples),
            status="estimated" if selected_samples else "missing",
            error=None if selected_samples else "explicit_judge_sampling_list_required",
            digest=config_hash,
        )
    )
    for metric in selected:
        usage = pricing.usage_for_calls(
            calls_by_metric[metric],
            embedding_requests=embedding_requests_by_metric[metric],
            embedding_tokens=embedding_tokens_by_metric[metric],
        )
        for name, value in (
            ("llm_calls", calls_by_metric[metric]),
            ("eligible_cases", eligible_by_metric[metric]),
            ("input_tokens", usage["input_tokens"]),
            ("output_tokens", usage["output_tokens"]),
            ("embedding_requests", usage["embedding_requests"]),
            ("embedding_tokens", usage["embedding_tokens"]),
            ("external_requests", usage["external_requests"]),
        ):
            records.append(
                _record(
                    f"estimate/aliyun/{metric}/{name}",
                    value=value,
                    status="estimated",
                    error=None,
                    usage=usage,
                    digest=config_hash,
                )
            )
        records.append(
            _record(
                f"estimate/aliyun/{metric}/cost",
                value=usage["cost"],
                status="estimated" if usage["cost"] is not None else "missing",
                error=None if usage["cost"] is not None else "pricing_not_configured",
                usage=usage,
                digest=config_hash,
            )
        )

    total_calls = sum(calls_by_metric.values())
    total_embedding_requests = sum(embedding_requests_by_metric.values())
    total_embedding_tokens = sum(embedding_tokens_by_metric.values())
    total_usage = pricing.usage_for_calls(
        total_calls,
        embedding_requests=total_embedding_requests,
        embedding_tokens=total_embedding_tokens,
    )
    for name, value, status, error in (
        ("llm_calls", total_calls, "estimated", None),
        ("input_tokens", total_usage["input_tokens"], "estimated", None),
        ("output_tokens", total_usage["output_tokens"], "estimated", None),
        (
            "embedding_requests",
            total_usage["embedding_requests"],
            "estimated",
            None,
        ),
        ("embedding_tokens", total_usage["embedding_tokens"], "estimated", None),
        ("external_requests", total_usage["external_requests"], "estimated", None),
        (
            "cost",
            total_usage["cost"],
            "estimated" if total_usage["cost"] is not None else "missing",
            None if total_usage["cost"] is not None else "pricing_not_configured",
        ),
    ):
        records.append(
            _record(
                f"estimate/aliyun/total/{name}",
                value=value,
                status=status,
                error=error,
                usage=total_usage,
                digest=config_hash,
            )
        )
    return records


def _preflight_judge_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    config: JudgeConfig,
    pricing: PricingConfig,
) -> list[Mapping[str, Any]]:
    """在导入 Ragas 或创建客户端前完成采样、权限与预算硬门。"""

    if not config.allow_paid_model_calls:
        raise RagasRunnerError("Judge execution requires --allow-paid-model-calls")
    if not config.sample_case_ids:
        raise RagasRunnerError("Judge execution requires an explicit sampling list")
    if len(config.sample_case_ids) > config.max_cases:
        raise RagasRunnerError(
            f"Judge sampling list exceeds configured cap of {config.max_cases} cases"
        )
    if len(config.sample_case_ids) > ABSOLUTE_JUDGE_CASE_CAP:
        raise RagasRunnerError(
            f"Judge sampling list exceeds hard cap of {ABSOLUTE_JUDGE_CASE_CAP} cases"
        )
    if pricing.currency.upper() != "CNY":
        raise RagasRunnerError("Judge budget preflight requires CNY pricing")
    if pricing.estimated_output_tokens_per_call < config.max_output_tokens:
        raise RagasRunnerError(
            "Judge output-token estimate must cover the configured API output cap"
        )

    selected_metrics = validate_judge_metrics(config.metrics)
    by_case = {_case_id(sample, source="sample"): sample for sample in samples}
    unknown = [case_id for case_id in config.sample_case_ids if case_id not in by_case]
    if unknown:
        raise RagasRunnerError(
            "Judge sampling list contains unknown case_id: " + ", ".join(unknown)
        )
    selected_samples = [by_case[case_id] for case_id in config.sample_case_ids]
    total_calls = 0
    total_embedding_requests = 0
    total_embedding_tokens = 0
    for sample in selected_samples:
        case_id = _case_id(sample, source="sample")
        for metric in selected_metrics:
            inputs, calls, error = _judge_inputs(sample, metric)
            if inputs is None:
                raise RagasRunnerError(
                    f"Judge preflight rejected {case_id}/{metric}: {error}"
                )
            total_calls += calls
            if metric == "answer_relevancy":
                total_embedding_requests += 2
                total_embedding_tokens += pricing.estimated_embedding_tokens_per_case
    usage = pricing.usage_for_calls(
        total_calls,
        embedding_requests=total_embedding_requests,
        embedding_tokens=total_embedding_tokens,
    )
    estimated_cost = usage["cost"]
    if estimated_cost is None:
        raise RagasRunnerError(
            "Judge budget preflight requires input, output and applicable embedding prices"
        )
    if estimated_cost > config.max_cost_cny or estimated_cost > ABSOLUTE_JUDGE_BUDGET_CNY:
        raise RagasRunnerError(
            f"Judge estimated cost CNY {estimated_cost:.4f} exceeds hard budget"
        )
    return selected_samples


def _install_ragas_043_vertex_import_shim() -> None:
    """补齐 Ragas 0.4.3 对已移除 Vertex 旧模块的无行为类型引用。"""

    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
    # Ragas 仅把此类型放入 legacy isinstance 元组；本运行器只使用 OpenAI
    # compatible 的现代 Collections 路径，因此不提供任何 Vertex 行为。
    module = types.ModuleType(module_name)
    module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = module


def _load_ragas_components() -> tuple[Any, dict[str, Any], Any, Any]:
    """延迟加载固定版本 Collections API；明确关闭遥测和持久化缓存。"""

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    try:
        version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RagasRunnerError(
            "ragas 0.4.3 is unavailable; run with `uv run --with ragas==0.4.3`"
        ) from exc
    if version != "0.4.3":
        raise RagasRunnerError(f"ragas version must be 0.4.3, found {version}")
    _install_ragas_043_vertex_import_shim()
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RagasRunnerError(
            "ragas 0.4.3 Collections API could not be imported in this environment"
        ) from exc
    classes = {
        "answer_correctness": AnswerCorrectness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision_with_reference": ContextPrecisionWithReference,
        "context_recall": ContextRecall,
        "factual_correctness": FactualCorrectness,
        "faithfulness": Faithfulness,
    }
    return AsyncOpenAI, classes, llm_factory, embedding_factory


async def _judge_records_and_points(
    samples: Sequence[Mapping[str, Any]],
    *,
    config: JudgeConfig,
    pricing: PricingConfig,
) -> tuple[list[dict[str, Any]], list[_ScorePoint]]:
    if not config.allow_paid_model_calls:
        raise RagasRunnerError("Judge execution requires --allow-paid-model-calls")
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RagasRunnerError(f"Judge API key environment variable is missing: {config.api_key_env}")
    selected = validate_judge_metrics(config.metrics)
    AsyncOpenAI, classes, llm_factory, embedding_factory = _load_ragas_components()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=0,
    )
    llm = llm_factory(
        config.model,
        provider="openai",
        client=client,
        cache=None,
        max_tokens=config.max_output_tokens,
    )
    embeddings = embedding_factory(
        "openai",
        model=config.embedding_model,
        client=client,
        interface="modern",
        cache=None,
    )
    metric_instances = {
        "answer_correctness": classes["answer_correctness"](
            llm=llm,
            weights=[1.0, 0.0],
        ),
        "answer_relevancy": classes["answer_relevancy"](
            llm=llm,
            embeddings=embeddings,
            strictness=3,
        ),
        "context_precision_with_reference": classes[
            "context_precision_with_reference"
        ](llm=llm),
        "context_recall": classes["context_recall"](llm=llm),
        "factual_correctness": classes["factual_correctness"](
            llm=llm,
            mode="f1",
        ),
        "faithfulness": classes["faithfulness"](llm=llm),
    }
    records: list[dict[str, Any]] = []
    points: list[_ScorePoint] = []
    try:
        for sample in samples:
            case_id = _case_id(sample, source="sample")
            case_path = _safe_path_component(case_id)
            digest = _sample_hash(sample)
            difficulty, tags = _sample_slices(sample)
            for metric_name in selected:
                inputs, expected_calls, input_error = _judge_inputs(sample, metric_name)
                if inputs is None:
                    status = "skipped" if input_error == "judge_not_eligible" else "missing"
                    records.append(
                        _record(
                            f"case/{case_path}/ragas/{metric_name}",
                            value=None,
                            status=status,
                            error=input_error,
                            usage=pricing.usage_for_calls(0),
                            digest=digest,
                        )
                    )
                    points.append(
                        _ScorePoint(
                            f"ragas/{metric_name}",
                            None,
                            status,
                            difficulty,
                            tags,
                            digest,
                        )
                    )
                    continue
                started = time.perf_counter()
                try:
                    result = await metric_instances[metric_name].ascore(**inputs)
                    value = float(result.value)
                    status = "ok" if math.isfinite(value) else "undefined"
                    error = None if status == "ok" else "judge_returned_non_finite"
                except Exception as exc:  # noqa: BLE001 - 外部边界需收敛成安全记录
                    value = None
                    status = "error"
                    error = f"judge_failed:{type(exc).__name__}"
                elapsed_ms = (time.perf_counter() - started) * 1000
                embedding_requests = 2 if metric_name == "answer_relevancy" else 0
                embedding_tokens = (
                    pricing.estimated_embedding_tokens_per_case
                    if metric_name == "answer_relevancy"
                    else 0
                )
                usage = pricing.usage_for_calls(
                    expected_calls,
                    embedding_requests=embedding_requests,
                    embedding_tokens=embedding_tokens,
                )
                records.append(
                    _record(
                        f"case/{case_path}/ragas/{metric_name}",
                        value=value,
                        status=status,
                        error=error,
                        latency_ms=elapsed_ms,
                        usage=usage,
                        digest=digest,
                    )
                )
                points.append(
                    _ScorePoint(
                        f"ragas/{metric_name}",
                        value,
                        status,
                        difficulty,
                        tags,
                        digest,
                    )
                )
    finally:
        await client.close()
    return records, points


def evaluate_records(
    samples: Sequence[Mapping[str, Any]],
    *,
    observations: Sequence[Mapping[str, Any]] = (),
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260828,
    judge: JudgeConfig | None = None,
    pricing: PricingConfig | None = None,
) -> list[dict[str, Any]]:
    """执行离线评测、可选 Judge、切片聚合及费用估算。"""

    pricing = pricing or PricingConfig()
    judge = judge or JudgeConfig()
    joined, unknown_observations = join_observations(samples, observations)
    case_records, points = _case_records_and_points(joined)
    if judge.enabled:
        selected_samples = _preflight_judge_samples(
            joined,
            config=judge,
            pricing=pricing,
        )
        judge_records, judge_points = asyncio.run(
            _judge_records_and_points(selected_samples, config=judge, pricing=pricing)
        )
        case_records.extend(judge_records)
        points.extend(judge_points)
    aggregate_records = _aggregate_records(
        points,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    estimate_records = estimate_judge_usage(
        joined,
        metrics=judge.metrics,
        pricing=pricing,
        case_ids=judge.sample_case_ids,
    )
    join_hash = _stable_hash(
        {
            "sample_hashes": [_sample_hash(sample) for sample in joined],
            "unknown_observations": unknown_observations,
        }
    )
    join_record = _record(
        "join/unknown_observations",
        value=unknown_observations,
        status="ok" if unknown_observations == 0 else "warning",
        error=None if unknown_observations == 0 else "observations_without_matching_sample",
        digest=join_hash,
    )
    records = case_records + aggregate_records + estimate_records + [join_record]
    for record in records:
        if set(record) != _OUTPUT_FIELDS:
            raise AssertionError("metric record escaped the persistence whitelist")
    return records


def write_metric_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """只写白名单字段的 JSONL，拒绝 NaN 和附带原文的扩展字段。"""

    lines: list[str] = []
    for record in records:
        if set(record) != _OUTPUT_FIELDS:
            unexpected = sorted(set(record) - _OUTPUT_FIELDS)
            raise RagasRunnerError(f"metric record contains forbidden fields: {unexpected}")
        lines.append(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


__all__ = [
    "ABSOLUTE_JUDGE_BUDGET_CNY",
    "ABSOLUTE_JUDGE_CASE_CAP",
    "ALLOWED_JUDGE_METRICS",
    "DEFAULT_JUDGE_METRICS",
    "JudgeConfig",
    "PricingConfig",
    "RagasRunnerError",
    "bootstrap_mean_ci",
    "estimate_judge_usage",
    "evaluate_records",
    "join_observations",
    "load_jsonl",
    "validate_judge_metrics",
    "write_metric_records",
]
