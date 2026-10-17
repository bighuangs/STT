"""本地评测资产与 Phoenix Dataset/Experiment 的显式桥接。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from app.application.evaluation_target import (
    EvaluationRequest,
    EvaluationTarget,
    TargetInput,
)


class _DatasetResource(Protocol):
    def create_dataset(self, **kwargs: Any) -> Any: ...


class _PhoenixClient(Protocol):
    datasets: _DatasetResource


_CASE_METADATA_FIELDS = frozenset({"category", "difficulty", "metadata", "sensitivity", "tags"})


def _is_reference_field(name: str) -> bool:
    """识别不得泄漏给被测目标的参考答案与相关性标注。"""

    return (
        name == "unanswerable"
        or name == "concept_groups"
        or name.startswith(("expected_", "forbidden_", "reference_", "relevant_"))
    )


def build_phoenix_examples(
    cases: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str,
    dataset_checksum: str,
) -> list[dict[str, Any]]:
    """把本地 case 投影为带稳定 ID 的 Phoenix upsert examples。"""

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_case in cases:
        case = deepcopy(dict(raw_case))
        case_id = str(case.get("case_id") or "").strip()
        query = str(case.get("query") or "").strip()
        if not case_id or not query:
            raise ValueError("Phoenix dataset case requires non-empty case_id and query")
        if case_id in seen:
            raise ValueError(f"Phoenix dataset contains duplicate case_id: {case_id}")
        seen.add(case_id)
        target_case = {
            key: value
            for key, value in case.items()
            if key not in _CASE_METADATA_FIELDS and not _is_reference_field(key)
        }
        expected = {key: value for key, value in case.items() if _is_reference_field(key)}
        examples.append(
            {
                "id": case_id,
                # Phoenix 按 input/expected 动态绑定 task 与 evaluator；参考答案不得进入
                # 被测目标输入，否则正式 Experiment 会产生评测泄漏。
                "input": {"case": target_case},
                "output": expected,
                "metadata": {
                    "case_id": case_id,
                    "dataset_version": dataset_version,
                    "dataset_checksum": dataset_checksum,
                    "difficulty": case.get("difficulty"),
                    "category": case.get("category"),
                },
            }
        )
    if not examples:
        raise ValueError("Phoenix dataset must contain at least one case")
    return examples


def _new_client() -> _PhoenixClient:
    """延迟创建官方客户端；URL/API key 由 Phoenix 官方环境变量读取。"""

    from phoenix.client import Client

    return Client()


def publish_dataset(
    *,
    name: str,
    description: str,
    cases: Iterable[Mapping[str, Any]],
    dataset_version: str,
    dataset_checksum: str,
    client: _PhoenixClient | None = None,
) -> dict[str, Any]:
    """以稳定 case ID upsert Phoenix Dataset，并返回脱敏发布回执。"""

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Phoenix dataset name must not be empty")
    resolved = client or _new_client()
    dataset = resolved.datasets.create_dataset(
        name=normalized_name,
        dataset_description=description.strip(),
        examples=build_phoenix_examples(
            cases,
            dataset_version=dataset_version,
            dataset_checksum=dataset_checksum,
        ),
    )
    return {
        "name": str(getattr(dataset, "name", normalized_name)),
        "version_id": str(getattr(dataset, "version_id", "")),
        "example_count": int(getattr(dataset, "example_count", 0)),
    }


def build_experiment_task(
    target: EvaluationTarget,
    *,
    kb_id: str,
    run_id: str,
    config: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """把稳定 EvaluationTarget 适配为 Phoenix experiment task。"""

    def task(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        case = inputs.get("case")
        if not isinstance(case, Mapping):
            raise ValueError("Phoenix experiment input is missing evaluation case")
        return target.evaluate(
            EvaluationRequest(
                target_input=TargetInput.from_case(
                    case,
                    correlation_id=hashlib.sha256(
                        f"{run_id}|{case.get('query', '')}".encode("utf-8")
                    ).hexdigest()[:32],
                ),
                kb_id=kb_id,
                run_id=run_id,
                config=config,
            )
        )

    return task


def run_experiment(
    *,
    dataset: Any,
    target: EvaluationTarget,
    kb_id: str,
    run_id: str,
    config: Mapping[str, Any],
    experiment_name: str,
    evaluators: Sequence[Callable[..., Any]] = (),
    execution_authorized: bool = False,
) -> Any:
    """显式运行 Phoenix Experiment；未授权时不加载客户端也不执行目标。"""

    if not execution_authorized:
        raise PermissionError("Phoenix experiment execution requires explicit authorization")
    from phoenix.client.experiments import run_experiment as phoenix_run_experiment

    return phoenix_run_experiment(
        dataset=dataset,
        task=build_experiment_task(target, kb_id=kb_id, run_id=run_id, config=config),
        evaluators=list(evaluators),
        experiment_name=experiment_name,
        experiment_metadata={
            "run_id": run_id,
            "config_version": config.get("config_version"),
            "dataset_version": config.get("dataset_version"),
            "dataset_checksum": config.get("dataset_checksum"),
        },
    )


__all__ = [
    "build_experiment_task",
    "build_phoenix_examples",
    "publish_dataset",
    "run_experiment",
]
