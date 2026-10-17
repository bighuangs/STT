"""无 I/O、无模型调用的通用评测 harness。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    EvaluationGrader,
    EvaluationRequest,
    EvaluationTarget,
    EvaluationTargetResult,
    TargetInput,
    ensure_observation_contract,
)


@dataclass(frozen=True, slots=True)
class EvaluationHarness:
    """负责目标调用、观测契约校验和 grader 编排。"""

    target: EvaluationTarget
    graders: Sequence[EvaluationGrader]

    def run_case(
        self,
        request: EvaluationRequest,
        *,
        grader_case: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_result = self.target.evaluate(request)
        try:
            if isinstance(target_result, EvaluationTargetResult):
                observation = ensure_observation_contract(target_result.observation)
            else:
                observation = ensure_observation_contract(target_result)
        finally:
            # 通用 harness 不具备读取 raw proof authority 的职责；无论观测契约
            # 是否有效都立即释放 wrapper，使 grader、异常和返回结果不可达 sidecar。
            target_result = None
        grades = []
        for grader in self.graders:
            result = dict(grader.grade(grader_case, observation))
            result.setdefault("grader", grader.name)
            grades.append(result)
        return {
            "schema_version": "stt.eval.case-result.v1",
            "case_id": str(grader_case["case_id"]),
            "observation": observation,
            "grades": grades,
        }

    def run(
        self,
        cases: Iterable[Mapping[str, Any]],
        *,
        kb_id: str,
        run_id: str,
        config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """按输入顺序运行 case；持久化与并发由外层 application 决定。"""

        return [
            self.run_case(
                EvaluationRequest(
                    target_input=TargetInput.from_case(
                        case,
                        correlation_id=hashlib.sha256(
                            f"{run_id}|{index}".encode("utf-8")
                        ).hexdigest()[:32],
                    ),
                    kb_id=kb_id,
                    run_id=run_id,
                    config=config,
                ),
                grader_case=case,
            )
            for index, case in enumerate(cases)
        ]
