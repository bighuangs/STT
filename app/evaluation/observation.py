"""旧评测观测入口的迁移兼容门面。

新 harness 应直接依赖 ``evals.harness`` 的 ``EvaluationTarget`` 契约；生产
实现由 ``app.infrastructure.evaluation`` 组装。此门面仅用于保持现有
``evaluation_service`` 与测试的 import 路径稳定，完成调用方迁移后应删除。
"""

from __future__ import annotations

from evals.harness.contracts import LegacyObservationTarget
from evals.harness.static_target import StaticEvaluationTarget

from app.infrastructure.evaluation.query_targets import (
    QueryGraphEvalAdapter,
    RetrievalOnlyEvalAdapter,
    _frozen_vector_scope,
    _packed_context,
)


# 迁移债务：application service 仍以 observe 方法命名该端口。
EvalObservationAdapter = LegacyObservationTarget


class StaticObservationAdapter(StaticEvaluationTarget):
    """保留旧类名的确定性目标；实现已归属根级 evals harness。"""


__all__ = [
    "EvalObservationAdapter",
    "QueryGraphEvalAdapter",
    "RetrievalOnlyEvalAdapter",
    "StaticObservationAdapter",
    "_frozen_vector_scope",
    "_packed_context",
]
