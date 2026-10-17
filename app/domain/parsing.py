"""TODO-07 文档解析与切块的稳定领域契约。

该模块只描述“解析后是什么”，不依赖 FastAPI、LangGraph、MinerU 或具体 Office
实现。解析器可以替换，后续切块、预览、缓存与索引仍消费同一份结构。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BlockType = Literal["text", "heading", "table", "image", "list", "code"]
WarningSeverity = Literal["info", "warning", "error"]
CoordinateSpace = Literal["normalized_page", "pdf_points"]


class SourceLocation(BaseModel):
    """源文档位置；不同格式只填写自己能够可靠提供的坐标。

    空间 bbox 统一使用展示旋转后的左上角原点（x 向右、y 向下）；PDF provider
    若输出下左原点或旋转前坐标，必须先换算后才能声明为可靠位置。
    """

    model_config = ConfigDict(extra="forbid")

    page_no: int | None = Field(default=None, ge=1)
    bbox: list[float] | None = None
    coordinate_space: CoordinateSpace | None = None
    slide_no: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    cell_range: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must contain [x0, y0, x1, y1]")
        if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
            raise ValueError("bbox coordinates must be finite numbers")
        coordinates = [float(item) for item in value]
        if any(item != item or item in {float("inf"), float("-inf")} for item in coordinates):
            raise ValueError("bbox coordinates must be finite numbers")
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            raise ValueError("bbox must satisfy x1 > x0 and y1 > y0")
        return coordinates

    @model_validator(mode="after")
    def validate_line_range(self) -> "SourceLocation":
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        if self.bbox is None and self.coordinate_space is not None:
            raise ValueError("coordinate_space requires bbox")
        if self.bbox is not None:
            if self.page_no is None:
                raise ValueError("bbox requires page_no")
            if self.coordinate_space is None:
                raise ValueError("bbox requires coordinate_space")
            if self.coordinate_space == "normalized_page" and any(
                coordinate < 0.0 or coordinate > 1.0 for coordinate in self.bbox
            ):
                raise ValueError("normalized_page bbox coordinates must be between 0 and 1")
            if self.coordinate_space == "pdf_points" and any(
                coordinate < 0.0 for coordinate in self.bbox
            ):
                raise ValueError("pdf_points bbox coordinates must be non-negative")
        return self


class ParsedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=128)
    object_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=255)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    size_bytes: int = Field(default=0, ge=0)


class ParseWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=128)
    severity: WarningSeverity = "warning"
    message: str = Field(min_length=1, max_length=2000)
    location: SourceLocation | None = None


class ParsedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1, max_length=128)
    type: BlockType
    text: str = ""
    structured_data: dict[str, Any] | list[Any] | None = None
    location: SourceLocation = Field(default_factory=SourceLocation)
    parent_heading_path: list[str] = Field(default_factory=list)
    order: int = Field(ge=0)
    asset_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> "ParsedBlock":
        if not self.text.strip() and self.structured_data is None and not self.asset_refs:
            raise ValueError("parsed block must contain text, structured_data or asset_refs")
        return self


class ParsedDocument(BaseModel):
    """可持久化并可重复切块的解析制品。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["parsed-document.v1"] = "parsed-document.v1"
    parser_id: str = Field(min_length=1, max_length=128)
    parser_version: str = Field(min_length=1, max_length=64)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_format: str = Field(min_length=1, max_length=64)
    source_mime_type: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[ParsedBlock] = Field(default_factory=list)
    assets: list[ParsedAsset] = Field(default_factory=list)
    warnings: list[ParseWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references_and_order(self) -> "ParsedDocument":
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_id must be unique within one document")
        if [block.order for block in self.blocks] != list(range(len(self.blocks))):
            raise ValueError("block order must be continuous and start at zero")
        asset_ids = {asset.asset_id for asset in self.assets}
        unknown = {
            asset_id
            for block in self.blocks
            for asset_id in block.asset_refs
            if asset_id not in asset_ids
        }
        if unknown:
            raise ValueError(f"block references unknown assets: {sorted(unknown)}")
        return self


class ParserProfile(BaseModel):
    """文件落盘后得到的实际解析选择，写入 DocumentVersion 快照。"""

    model_config = ConfigDict(extra="forbid")

    detected_format: str
    detected_mime_type: str
    parser_id: str
    parser_version: str
    template_id: str
    template_version: str
    suggested_template_id: str
    warnings: list[ParseWarning] = Field(default_factory=list)


class ChunkingTemplateInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    version: str
    name: str
    description: str
    supported_block_types: list[BlockType]


class ParseArtifactInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_hit: bool = False
    object_key: str | None = None


class DocumentParseError(RuntimeError):
    """对 API、Worker 与测试都稳定的解析失败。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        self.status_code = status_code
