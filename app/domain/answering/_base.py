"""回答契约共享的不可变模型、规范化 JSON 与稳定身份工具。"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from hashlib import sha256
import json
from typing import Any, ClassVar, Mapping, Self

from pydantic import BaseModel, ConfigDict


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
SCOPE_HASH_PATTERN = r"^scope_[0-9a-f]{64}$"
FACET_ID_PATTERN = r"^qf_[0-9a-f]{32}$"
EVIDENCE_ID_PATTERN = r"^ev_[0-9a-f]{40}$"
REPRESENTATION_ID_PATTERN = r"^rep_[0-9a-f]{40}$"
WITNESS_ID_PATTERN = r"^wit_[0-9a-f]{40}$"
CLAIM_ID_PATTERN = r"^clm_[1-9][0-9]*$"


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", round_trip=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """以 UTF-8 语义生成稳定、无空白且键有序的 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def sha256_digest(value: str | bytes) -> str:
    """返回带算法前缀的稳定 SHA-256。"""

    payload = value.encode("utf-8") if isinstance(value, str) else value
    return f"sha256:{sha256(payload).hexdigest()}"


def stable_content_hash(value: Any) -> str:
    """对规范化 JSON 计算稳定内容哈希。"""

    return sha256_digest(canonical_json(value))


def stable_identifier(prefix: str, value: Any, *, length: int = 40) -> str:
    """从规范化内容生成固定前缀的截断 SHA-256 身份。"""

    if not prefix or not prefix.endswith("_"):
        raise ValueError("identifier prefix must be non-empty and end with '_'")
    if length < 16 or length > 64:
        raise ValueError("identifier digest length must be between 16 and 64")
    digest = stable_content_hash(value).removeprefix("sha256:")
    return f"{prefix}{digest[:length]}"


class AnsweringContractModel(BaseModel):
    """所有回答领域契约共享的严格序列化边界。"""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """输出只含 JSON 数据的稳定边界表示。"""

        return self.model_dump(mode="json", round_trip=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """从不可信映射重新执行完整 Pydantic 校验。"""

        return cls.model_validate(dict(value))


def require_unique(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    """校验标识符非空、无首尾空白且不重复。"""

    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{field_name} must contain trimmed non-empty identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


__all__ = [
    "CLAIM_ID_PATTERN",
    "EVIDENCE_ID_PATTERN",
    "FACET_ID_PATTERN",
    "REPRESENTATION_ID_PATTERN",
    "SHA256_PATTERN",
    "SCOPE_HASH_PATTERN",
    "WITNESS_ID_PATTERN",
    "AnsweringContractModel",
    "canonical_json",
    "require_unique",
    "sha256_digest",
    "stable_content_hash",
    "stable_identifier",
]
