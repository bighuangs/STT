"""索引版本领域状态。"""

from enum import StrEnum

from app.domain._state_machine import ensure_transition


class IndexVersionStatus(StrEnum):
    CREATING = "creating"
    STAGING = "staging"
    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"
    FAILED = "failed"


_TRANSITIONS = {
    IndexVersionStatus.CREATING: {
        IndexVersionStatus.STAGING,
        IndexVersionStatus.FAILED,
    },
    IndexVersionStatus.STAGING: {
        IndexVersionStatus.ACTIVE,
        IndexVersionStatus.FAILED,
    },
    IndexVersionStatus.ACTIVE: {
        IndexVersionStatus.RETIRING,
        IndexVersionStatus.RETIRED,
    },
    IndexVersionStatus.RETIRING: {IndexVersionStatus.RETIRED},
    # TODO-03 显式回滚会把仍在保留期内的上一版本重新激活。
    IndexVersionStatus.RETIRED: {IndexVersionStatus.ACTIVE},
    IndexVersionStatus.FAILED: set(),
}


def ensure_index_version_transition(current: str, target: str) -> None:
    ensure_transition(IndexVersionStatus(current), IndexVersionStatus(target), _TRANSITIONS)
