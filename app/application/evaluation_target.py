"""Application 与离线评测 harness 共用的稳定被测目标契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.domain.evaluation_proof import ProofEvaluationAuthority


@dataclass(frozen=True, slots=True)
class TargetInput:
    """允许进入被测系统的盲测输入；不包含 qrel、reference 或 case_id。"""

    query: str
    correlation_id: str
    sensitivity: str | None = None
    conversation: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("evaluation query must be a non-empty string")
        if not isinstance(self.correlation_id, str) or not self.correlation_id.strip():
            raise ValueError("evaluation correlation_id must be a non-empty string")
        if self.sensitivity is not None and not isinstance(self.sensitivity, str):
            raise ValueError("evaluation sensitivity must be a string or None")
        if not isinstance(self.conversation, tuple) or not all(
            isinstance(item, Mapping) for item in self.conversation
        ):
            raise ValueError("evaluation conversation must contain mappings")

    @classmethod
    def from_case(
        cls,
        case: Mapping[str, Any],
        *,
        correlation_id: str,
    ) -> TargetInput:
        """在 grader 边界将完整 case 投影为最小被测输入。"""

        conversation = case.get("conversation") or ()
        if not isinstance(conversation, Sequence) or isinstance(conversation, (str, bytes)):
            raise ValueError("evaluation conversation must be a sequence")
        return cls(
            query=str(case.get("query") or ""),
            correlation_id=correlation_id,
            sensitivity=(str(case["sensitivity"]) if case.get("sensitivity") is not None else None),
            conversation=tuple(
                deepcopy(dict(item)) for item in conversation if isinstance(item, Mapping)
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        """为迁移期 observe 目标生成同样受限的 allowlist 投影。"""

        return {
            "query": self.query,
            "correlation_id": self.correlation_id,
            "sensitivity": self.sensitivity,
            "conversation": [deepcopy(dict(item)) for item in self.conversation],
        }


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """一次被测目标调用的不可变盲测边界输入。"""

    target_input: TargetInput
    kb_id: str
    run_id: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.target_input, TargetInput):
            raise ValueError("evaluation target_input must be TargetInput")
        if not isinstance(self.config, Mapping):
            raise ValueError("evaluation config must be a mapping")
        if not isinstance(self.kb_id, str) or not self.kb_id.strip():
            raise ValueError("evaluation kb_id must be a non-empty string")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("evaluation run_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EvaluationTargetResult:
    """被测目标的安全观测与仅限当前调用的可信 proof sidecar。"""

    observation: Mapping[str, Any]
    proof_authority: ProofEvaluationAuthority | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise ValueError("evaluation target observation must be a mapping")
        if self.proof_authority is not None and not isinstance(
            self.proof_authority,
            ProofEvaluationAuthority,
        ):
            raise ValueError("evaluation proof authority has an invalid type")


@runtime_checkable
class EvaluationTarget(Protocol):
    """评测 harness 唯一允许调用的被测目标接口。"""

    def evaluate(
        self,
        request: EvaluationRequest,
    ) -> Mapping[str, Any] | EvaluationTargetResult: ...


@runtime_checkable
class LegacyObservationTarget(Protocol):
    """旧 evaluation service 的 observe 调用契约。"""

    def observe(
        self,
        case: dict,
        *,
        kb_id: str,
        run_id: str,
        config: dict,
    ) -> dict: ...


__all__ = [
    "EvaluationRequest",
    "EvaluationTarget",
    "EvaluationTargetResult",
    "LegacyObservationTarget",
    "TargetInput",
]
