"""确定性 grader 与旧指标函数的注入式适配器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from evals.harness.contracts import ensure_observation_contract


@dataclass(frozen=True, slots=True)
class ObservationContractGrader:
    """把观测契约作为最前置、可独立运行的质量门。"""

    name: str = "observation_contract"

    def grade(
        self,
        case: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        del case
        ensure_observation_contract(observation)
        return {"grader": self.name, "passed": True, "score": 1.0}


@dataclass(frozen=True, slots=True)
class FunctionGrader:
    """通过依赖注入复用既有确定性指标，不反向导入生产模块。"""

    name: str
    evaluator: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

    def grade(
        self,
        case: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = self.evaluator(case, observation)
        if not isinstance(value, Mapping):
            raise TypeError(f"grader {self.name} must return a mapping")
        return {"grader": self.name, **dict(value)}
