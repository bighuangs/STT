"""P3 查询可靠性、安全与向量拓扑验收 API schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.query_security import DataClassification


class P3SelfCheckRequest(BaseModel):
    """运行九项纯内存、无外调、无持久写入的确定性检查。"""

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    question: str = Field(
        default="请验证当前授权知识库的查询可靠性与安全边界。",
        min_length=1,
        max_length=4096,
    )


class P3SecurityInspectRequest(BaseModel):
    """只返回安全判定和内容哈希，不回显待检查正文。"""

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=200_000)
    source: Literal["user", "knowledge_base", "web"] = "user"
    data_classification: DataClassification = DataClassification.INTERNAL


class P3TopologyInventoryRequest(BaseModel):
    """Milvus 只读拓扑检查参数。

    物理目标名称仍会在路由层与服务端配置比对；客户端不能借此枚举任意
    collection 或切换数据库。
    """

    model_config = ConfigDict(extra="forbid")

    physical_database: str = Field(
        default="default",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,62}$",
    )
    physical_collection: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,62}$",
    )
    sample_limit: int = Field(default=1000, ge=1, le=10_000)

    @model_validator(mode="after")
    def reject_blank_optional_collection(self) -> "P3TopologyInventoryRequest":
        if self.physical_collection is not None and not self.physical_collection.strip():
            raise ValueError("physical_collection cannot be blank")
        return self


__all__ = [
    "P3SecurityInspectRequest",
    "P3SelfCheckRequest",
    "P3TopologyInventoryRequest",
]
