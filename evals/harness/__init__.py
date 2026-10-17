"""不依赖生产实现细节的评测执行器。"""

from .contracts import (
    EvaluationGrader,
    EvaluationRequest,
    EvaluationTarget,
    EvaluationTargetResult,
    LegacyObservationTarget,
    ObservationContractError,
    TargetInput,
    ensure_observation_contract,
)
from .runner import EvaluationHarness
from .static_target import StaticEvaluationTarget

__all__ = [
    "EvaluationGrader",
    "EvaluationHarness",
    "EvaluationRequest",
    "EvaluationTarget",
    "EvaluationTargetResult",
    "LegacyObservationTarget",
    "ObservationContractError",
    "StaticEvaluationTarget",
    "TargetInput",
    "ensure_observation_contract",
]
