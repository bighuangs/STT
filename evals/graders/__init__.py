"""可组合、与生产代码解耦的 grader。"""

from .deterministic import FunctionGrader, ObservationContractGrader

__all__ = ["FunctionGrader", "ObservationContractGrader"]
