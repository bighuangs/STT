"""领域状态转换公共校验。"""

from collections.abc import Mapping, Set
from enum import StrEnum


class InvalidStateTransition(ValueError):
    """请求的领域状态转换不合法。"""


def ensure_transition(
    current: StrEnum,
    target: StrEnum,
    transitions: Mapping[StrEnum, Set[StrEnum]],
) -> None:
    """校验状态转换；幂等地写入当前状态被允许。"""
    if current == target:
        return
    allowed = transitions.get(current, set())
    if target not in allowed:
        raise InvalidStateTransition(
            f"invalid transition: {current.value} -> {target.value}"
        )
