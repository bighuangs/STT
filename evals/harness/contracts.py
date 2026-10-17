"""评测 harness 与被测系统之间的稳定公开契约。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from app.application.evaluation_target import (
    EvaluationRequest,
    EvaluationTarget,
    EvaluationTargetResult,
    LegacyObservationTarget,
    TargetInput,
)


_OBSERVATION_SCHEMA = re.compile(r"^stt\.eval\.observation\.v[1-9][0-9]*$")


class ObservationContractError(ValueError):
    """观测不满足稳定评测契约。"""


@runtime_checkable
class EvaluationGrader(Protocol):
    """对一个 case 与其观测打分的纯函数式接口。"""

    @property
    def name(self) -> str: ...

    def grade(
        self,
        case: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def ensure_observation_contract(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """校验并复制稳定观测；不接受私有 Graph state 作为隐式输入。"""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("evaluation observation must be a mapping")
    payload = dict(observation)
    schema = payload.get("observation_schema")
    if not isinstance(schema, str) or _OBSERVATION_SCHEMA.fullmatch(schema) is None:
        raise ObservationContractError("observation_schema must match stt.eval.observation.v<N>")
    if not isinstance(payload.get("answer"), str):
        raise ObservationContractError("observation answer must be a string")
    for field in ("retrieved", "reranked", "context"):
        if not isinstance(payload.get(field), list):
            raise ObservationContractError(f"observation {field} must be a list")
    latency = payload.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ObservationContractError("observation latency_ms must be a non-negative number")
    for field in ("versions", "side_effects"):
        if not isinstance(payload.get(field), Mapping):
            raise ObservationContractError(f"observation {field} must be a mapping")
    return payload


__all__ = [
    "EvaluationGrader",
    "EvaluationRequest",
    "EvaluationTarget",
    "EvaluationTargetResult",
    "LegacyObservationTarget",
    "TargetInput",
    "ObservationContractError",
    "ensure_observation_contract",
]
