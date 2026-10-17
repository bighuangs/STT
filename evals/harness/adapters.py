"""旧目标与稳定 EvaluationTarget 契约之间的纯边界适配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import EvaluationRequest, LegacyObservationTarget


@dataclass(frozen=True, slots=True)
class LegacyTargetBridge:
    """在迁移期把 observe 目标包装为 evaluate 目标。"""

    target: LegacyObservationTarget

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        return self.target.observe(
            request.target_input.as_mapping(),
            kb_id=request.kb_id,
            run_id=request.run_id,
            config=dict(request.config),
        )
