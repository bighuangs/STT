"""用于 CI 与契约测试的确定性评测目标。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import EvaluationRequest, TargetInput


class StaticEvaluationTarget:
    """从冻结观测映射读取结果，不访问模型、网络或生产存储。"""

    def __init__(
        self,
        observations: dict[str, dict[str, Any]],
        *,
        fail_cases: set[str] | None = None,
    ) -> None:
        self.observations = deepcopy(observations)
        self.fail_cases = set(fail_cases or ())

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        correlation_id = request.target_input.correlation_id
        if correlation_id in self.fail_cases:
            raise TimeoutError(f"deterministic failure for {correlation_id}")
        value = deepcopy(self.observations.get(correlation_id, {}))
        value.setdefault("observation_schema", "stt.eval.observation.v2")
        value.setdefault("answer", "")
        value.setdefault("retrieved", [])
        value.setdefault("reranked", value["retrieved"])
        value.setdefault("context", value["reranked"])
        value.setdefault("latency_ms", 1.0)
        value.setdefault("token_usage", None)
        value.setdefault("cost", None)
        value.setdefault("versions", dict(request.config))
        value.setdefault("side_effects", {})
        value.setdefault("answer_id", f"ans-{request.run_id}-{correlation_id}")
        return value

    def observe(
        self,
        case: dict,
        *,
        kb_id: str,
        run_id: str,
        config: dict,
    ) -> dict[str, Any]:
        """兼容旧 application service；新代码应调用 evaluate。"""

        return self.evaluate(
            EvaluationRequest(
                target_input=TargetInput(
                    query=str(case["query"]),
                    correlation_id=str(case["correlation_id"]),
                    sensitivity=(
                        str(case["sensitivity"]) if case.get("sensitivity") is not None else None
                    ),
                    conversation=tuple(case.get("conversation") or ()),
                ),
                kb_id=kb_id,
                run_id=run_id,
                config=config,
            )
        )
