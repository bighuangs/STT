"""版本化的结构感知切块模板。

TODO-07 将“解析”和“切块”分为两个稳定边界：解析器只产出
:class:`~app.domain.parsing.ParsedDocument`，本模块只消费该领域对象。这样切换模板时
可以直接复用解析制品，不需要重新执行 OCR 或 Office 解析。

设计约束：

* 默认模板固定为 ``general@1.0.0``，模板建议只作为提示，不能悄悄改变结果；
* 输出仍是现有导入链能够消费的 ``list[dict]``，同时补齐可审计的 provenance；
* chunk id、顺序和来源块顺序均为确定性的；
* 幻灯片与工作表是硬边界，任何模板都不会跨边界合并；
* 所有面向调用方的失败都转换为带稳定错误码的 ``DocumentParseError``。
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any, Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.domain.parsing import (
    ChunkingTemplateInfo,
    DocumentParseError,
    ParsedBlock,
    ParsedDocument,
    SourceLocation,
)


DEFAULT_TEMPLATE_ID = "general"
DEFAULT_TEMPLATE_VERSION = "1.0.0"
MANUAL_TABLE_AWARE_BASE_VERSION = "1.1.0"
MANUAL_TABLE_AWARE_VERSION = "1.1.1"

ERROR_TEMPLATE_NOT_FOUND = "CHUNK_TEMPLATE_NOT_FOUND"
ERROR_TEMPLATE_VERSION_NOT_FOUND = "CHUNK_TEMPLATE_VERSION_NOT_FOUND"
ERROR_TEMPLATE_DUPLICATE = "CHUNK_TEMPLATE_DUPLICATE"
ERROR_CONFIG_INVALID = "CHUNK_CONFIG_INVALID"
ERROR_DOCUMENT_EMPTY = "CHUNK_DOCUMENT_EMPTY"
ERROR_OUTPUT_EMPTY = "CHUNK_OUTPUT_EMPTY"

TablePolicy = Literal["structured_text", "text_only", "skip"]
ImagePolicy = Literal["reference", "caption", "skip"]
TemplateStrategy = Literal["general", "manual", "paper", "qa", "table", "presentation"]

_encoder = tiktoken.get_encoding("cl100k_base")
_template_id_pattern = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_version_pattern = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _token_ids(text: str) -> list[int]:
    return _encoder.encode(text, disallowed_special=()) if text else []


def count_tokens(text: str) -> int:
    """使用导入链既有的 cl100k_base 口径计算 token 数。"""

    return len(_token_ids(text))


class ChunkingTemplateConfig(BaseModel):
    """会随文档版本一并快照的切块参数。

    ``llm_enhancement`` 仅记录是否允许后续增强节点运行；本模块本身不会调用模型，
    因而同一解析制品与同一配置始终得到相同结果。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tokens: int = Field(default=512, ge=8, le=8192)
    overlap_tokens: int = Field(default=64, ge=0, le=2048)
    inherit_titles: bool = True
    table_policy: TablePolicy = "structured_text"
    image_policy: ImagePolicy = "reference"
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", ". ", " ")
    llm_enhancement: bool = False

    @field_validator("separators")
    @classmethod
    def validate_separators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ValueError("separators must contain between 1 and 16 items")
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("separator must be a non-empty string")
        if len(set(value)) != len(value):
            raise ValueError("separators must be unique")
        return value

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingTemplateConfig":
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self

    @classmethod
    def from_parser_config(
        cls,
        parser_config: Mapping[str, Any] | None,
        *,
        defaults: "ChunkingTemplateConfig | None" = None,
    ) -> "ChunkingTemplateConfig":
        """从现有大而杂的 parser_config 中只提取切块字段。

        支持顶层字段，也支持 ``chunking`` / ``chunking_config`` 子对象；其余解析器、
        OCR、模型字段会被忽略，不会因为领域配置隔离而误报为非法参数。
        """

        return _config_from_parser_config(parser_config, defaults=defaults or cls())


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    block_type: str
    source_block_ids: tuple[str, ...]
    locations: tuple[tuple[str, SourceLocation], ...]
    heading_path: tuple[str, ...]
    heading_source_ids: tuple[str, ...]
    images: tuple[str, ...]
    boundary: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _HtmlTableCell:
    text: str
    colspan: int
    rowspan: int
    is_header: bool


@dataclass(frozen=True, slots=True)
class _HtmlTableRow:
    cells: tuple[_HtmlTableCell, ...]
    is_thead: bool


class _FirstHtmlTableParser(HTMLParser):
    """只读取首个 HTML table，并把标签安全地归一为行/单元格。

    MinerU 的 ``table_body`` 是 HTML fragment。这里不保留原始标签，因此后续 token
    切分不可能落在 ``<td>``、属性或实体中间；嵌套标签只贡献可见文本，嵌套表格不会
    被误当作外层表格的新行。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_HtmlTableRow] = []
        self.caption_parts: list[str] = []
        self._table_depth = 0
        self._finished = False
        self._in_thead = False
        self._in_caption = False
        self._row_cells: list[_HtmlTableCell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_tag = ""
        self._cell_colspan = 1
        self._cell_rowspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "table":
            if self._finished:
                return
            self._table_depth += 1
            return
        if self._table_depth != 1 or self._finished:
            return
        if normalized == "thead":
            self._in_thead = True
        elif normalized == "caption":
            self._in_caption = True
        elif normalized == "tr":
            self._finish_row()
            self._row_cells = []
        elif normalized in {"td", "th"} and self._row_cells is not None:
            self._finish_cell()
            values = {str(name).lower(): value for name, value in attrs}
            self._cell_parts = []
            self._cell_tag = normalized
            self._cell_colspan = _bounded_html_span(values.get("colspan"))
            self._cell_rowspan = _bounded_html_span(values.get("rowspan"))
        elif normalized == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "table":
            if self._table_depth == 1:
                self._finish_cell()
                self._finish_row()
                self._in_caption = False
                self._in_thead = False
                self._finished = True
            if self._table_depth:
                self._table_depth -= 1
            return
        if self._table_depth != 1 or self._finished:
            return
        if normalized in {"td", "th"}:
            self._finish_cell()
        elif normalized == "tr":
            self._finish_cell()
            self._finish_row()
        elif normalized == "caption":
            self._in_caption = False
        elif normalized == "thead":
            self._in_thead = False

    def handle_data(self, data: str) -> None:
        if self._table_depth < 1 or self._finished:
            return
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        elif self._table_depth == 1 and self._in_caption:
            self.caption_parts.append(data)

    def close(self) -> None:
        super().close()
        if not self._finished:
            self._finish_cell()
            self._finish_row()

    @property
    def caption(self) -> str:
        return _normalize_html_text("".join(self.caption_parts))

    def _finish_cell(self) -> None:
        if self._cell_parts is None or self._row_cells is None:
            return
        self._row_cells.append(
            _HtmlTableCell(
                text=_normalize_html_text("".join(self._cell_parts)),
                colspan=self._cell_colspan,
                rowspan=self._cell_rowspan,
                is_header=self._cell_tag == "th",
            )
        )
        self._cell_parts = None
        self._cell_tag = ""
        self._cell_colspan = 1
        self._cell_rowspan = 1

    def _finish_row(self) -> None:
        if self._row_cells is None:
            return
        if self._row_cells:
            self.rows.append(_HtmlTableRow(tuple(self._row_cells), self._in_thead))
        self._row_cells = None


class ChunkingTemplate(ABC):
    """可注册、可版本化的切块策略端口。"""

    template_id: str
    version: str
    name: str
    description: str
    strategy: TemplateStrategy
    default_config: ChunkingTemplateConfig

    @property
    def info(self) -> ChunkingTemplateInfo:
        return ChunkingTemplateInfo(
            template_id=self.template_id,
            version=self.version,
            name=self.name,
            description=self.description,
            supported_block_types=["text", "heading", "table", "image", "list", "code"],
        )

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        *,
        config: ChunkingTemplateConfig | Mapping[str, Any] | None = None,
        file_title: str | None = None,
    ) -> list[dict[str, Any]]:
        """将一个解析制品转换为现有导入链兼容的 chunks。"""


@dataclass(frozen=True, slots=True)
class StructuredChunkingTemplate(ChunkingTemplate):
    template_id: str
    version: str
    name: str
    description: str
    strategy: TemplateStrategy
    default_config: ChunkingTemplateConfig

    def chunk(
        self,
        document: ParsedDocument,
        *,
        config: ChunkingTemplateConfig | Mapping[str, Any] | None = None,
        file_title: str | None = None,
    ) -> list[dict[str, Any]]:
        config = resolve_chunking_config(self, config)
        if not document.blocks:
            raise DocumentParseError(
                ERROR_DOCUMENT_EMPTY,
                "解析制品中没有可切块的内容",
                retryable=False,
            )

        manual_structure_v2 = (
            self.template_id == "manual"
            and _version_sort_key(self.version)
            >= _version_sort_key(MANUAL_TABLE_AWARE_BASE_VERSION)
        )
        segments = _document_segments(
            document,
            config=config,
            strategy=self.strategy,
            table_aware_html=manual_structure_v2,
        )
        if manual_structure_v2:
            segments = _attach_headings_to_following_content(
                segments,
                inherit_titles=config.inherit_titles,
            )
        if not segments:
            raise DocumentParseError(
                ERROR_OUTPUT_EMPTY,
                "当前切块策略过滤了全部内容，未生成可索引块",
                retryable=False,
            )

        resolved_title = str(
            file_title
            or document.metadata.get("title")
            or document.metadata.get("file_name")
            or ""
        ).strip()
        return _merge_segments(
            document,
            segments,
            template=self,
            config=config,
            file_title=resolved_title,
        )


class ChunkingTemplateRegistry:
    """按 ``template_id + version`` 寻址的只增注册表。"""

    def __init__(self) -> None:
        self._templates: dict[str, dict[str, ChunkingTemplate]] = {}

    def register(self, template: ChunkingTemplate) -> None:
        if not _template_id_pattern.fullmatch(template.template_id):
            raise DocumentParseError(
                ERROR_CONFIG_INVALID,
                f"非法模板 ID：{template.template_id!r}",
                retryable=False,
            )
        if not _version_pattern.fullmatch(template.version):
            raise DocumentParseError(
                ERROR_CONFIG_INVALID,
                f"非法模板版本：{template.version!r}",
                retryable=False,
            )
        versions = self._templates.setdefault(template.template_id, {})
        if template.version in versions:
            raise DocumentParseError(
                ERROR_TEMPLATE_DUPLICATE,
                f"切块模板已注册：{template.template_id}@{template.version}",
                retryable=False,
                status_code=409,
            )
        versions[template.version] = template

    def get(self, template_id: str, version: str | None = None) -> ChunkingTemplate:
        versions = self._templates.get(template_id)
        if not versions:
            raise DocumentParseError(
                ERROR_TEMPLATE_NOT_FOUND,
                f"未知切块模板：{template_id}",
                retryable=False,
            )
        if version is None:
            return versions[max(versions, key=_version_sort_key)]
        normalized_version = _normalize_version_alias(version)
        template = versions.get(normalized_version)
        if template is None:
            raise DocumentParseError(
                ERROR_TEMPLATE_VERSION_NOT_FOUND,
                f"切块模板版本不存在：{template_id}@{version}",
                retryable=False,
            )
        return template

    def resolve(self, template_id: str, version: str | None = None) -> ChunkingTemplate:
        """``get`` 的领域化别名，供应用服务固定使用。"""

        return self.get(template_id, version)

    def list_infos(self) -> list[ChunkingTemplateInfo]:
        result: list[ChunkingTemplateInfo] = []
        for template_id in sorted(self._templates):
            versions = self._templates[template_id]
            for version in sorted(versions, key=_version_sort_key, reverse=True):
                result.append(versions[version].info)
        return result

    def capabilities(self) -> list[dict[str, Any]]:
        """返回可直接用于 API 响应的模板能力与默认配置快照。"""

        result: list[dict[str, Any]] = []
        for info in self.list_infos():
            template = self.get(info.template_id, info.version)
            result.append(
                {
                    **info.model_dump(mode="json"),
                    "default_config": template.default_config.model_dump(mode="json"),
                    "is_latest": self.get(info.template_id).version == info.version,
                }
            )
        return result

    def chunk(
        self,
        document: ParsedDocument,
        *,
        template_id: str = DEFAULT_TEMPLATE_ID,
        template_version: str | None = None,
        config: ChunkingTemplateConfig | Mapping[str, Any] | None = None,
        file_title: str | None = None,
    ) -> list[dict[str, Any]]:
        template = self.get(template_id, template_version)
        resolved_config = resolve_chunking_config(template, config)
        return template.chunk(document, config=resolved_config, file_title=file_title)


def _version_sort_key(version: str) -> tuple[int, int, int, str]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version)
    if match is None:
        return (0, 0, 0, version)
    return (int(match[1]), int(match[2]), int(match[3]), match[4])


def _normalize_version_alias(version: str) -> str:
    match = re.fullmatch(r"v(\d+)", version.strip(), flags=re.IGNORECASE)
    return f"{match[1]}.0.0" if match else version.strip()


def resolve_chunking_config(
    template: ChunkingTemplate,
    value: ChunkingTemplateConfig | Mapping[str, Any] | None,
) -> ChunkingTemplateConfig:
    """把调用方参数与模板默认值合并，并统一产生稳定错误码。"""

    return _resolve_config_with_defaults(template.default_config, value)


def _resolve_config_with_defaults(
    defaults: ChunkingTemplateConfig,
    value: ChunkingTemplateConfig | Mapping[str, Any] | None,
) -> ChunkingTemplateConfig:

    if value is None:
        return defaults
    if isinstance(value, ChunkingTemplateConfig):
        return value
    if not isinstance(value, Mapping):
        raise DocumentParseError(
            ERROR_CONFIG_INVALID,
            "切块配置必须是对象",
            retryable=False,
        )

    raw = dict(value)
    aliases = {
        "chunk_token_num": "target_tokens",
        "chunk_size": "target_tokens",
        "title_inheritance": "inherit_titles",
        "inherit_headings": "inherit_titles",
        "sub_separators": "separators",
    }
    for old_name, new_name in aliases.items():
        if old_name in raw:
            if new_name in raw:
                raise DocumentParseError(
                    ERROR_CONFIG_INVALID,
                    f"切块配置不能同时包含 {old_name} 与 {new_name}",
                    retryable=False,
                )
            raw[new_name] = raw.pop(old_name)

    percent = raw.pop("overlapped_percent", None)
    base = defaults.model_dump()
    if "target_tokens" in raw and "overlap_tokens" not in raw and percent is None:
        # 模板默认 overlap 不能使一个较小的显式 target 变为无效配置。
        target = raw["target_tokens"]
        if isinstance(target, int) and not isinstance(target, bool) and target > 0:
            base["overlap_tokens"] = min(base["overlap_tokens"], max(0, target // 8))
    base.update(raw)
    if percent is not None:
        if isinstance(percent, bool) or not isinstance(percent, int) or not 0 <= percent < 100:
            raise DocumentParseError(
                ERROR_CONFIG_INVALID,
                "overlapped_percent 必须是 [0, 100) 的整数",
                retryable=False,
            )
        target = base.get("target_tokens")
        if not isinstance(target, int) or isinstance(target, bool):
            raise DocumentParseError(
                ERROR_CONFIG_INVALID,
                "target_tokens 必须是整数",
                retryable=False,
            )
        base["overlap_tokens"] = int(target * percent / 100)

    try:
        return ChunkingTemplateConfig.model_validate(base)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        )
        raise DocumentParseError(
            ERROR_CONFIG_INVALID,
            f"非法切块配置：{detail}",
            retryable=False,
        ) from exc


def _config_from_parser_config(
    parser_config: Mapping[str, Any] | None,
    *,
    defaults: ChunkingTemplateConfig,
) -> ChunkingTemplateConfig:
    if parser_config is None:
        return defaults
    if not isinstance(parser_config, Mapping):
        raise DocumentParseError(
            ERROR_CONFIG_INVALID,
            "parser_config 必须是对象",
            retryable=False,
        )
    nested = parser_config.get("chunking_config", parser_config.get("chunking"))
    source = nested if nested is not None else parser_config
    if not isinstance(source, Mapping):
        raise DocumentParseError(
            ERROR_CONFIG_INVALID,
            "parser_config.chunking 必须是对象",
            retryable=False,
        )
    accepted = {
        "target_tokens",
        "overlap_tokens",
        "inherit_titles",
        "table_policy",
        "image_policy",
        "separators",
        "llm_enhancement",
        "chunk_token_num",
        "chunk_size",
        "title_inheritance",
        "inherit_headings",
        "sub_separators",
        "overlapped_percent",
    }
    selected = {key: value for key, value in source.items() if key in accepted}
    return _resolve_config_with_defaults(defaults, selected)


def _source_boundary(block: ParsedBlock) -> tuple[str, str]:
    if block.location.slide_no is not None:
        return ("slide", str(block.location.slide_no))
    if block.location.sheet:
        return ("sheet", block.location.sheet)
    return ("flow", "")


def _strategy_boundary(
    block: ParsedBlock,
    *,
    strategy: TemplateStrategy,
    heading_path: tuple[str, ...],
) -> tuple[str, str]:
    source_boundary = _source_boundary(block)
    # 表格在所有模板中都是独立检索单元，避免与前后叙述合并后丢失 table 类型。
    if block.type == "table":
        owner = f"{source_boundary[0]}:{source_boundary[1]}:{block.block_id}"
        return ("table", owner)
    if source_boundary[0] != "flow":
        return source_boundary
    if strategy in {"manual", "paper"} and heading_path:
        return ("section", " / ".join(heading_path))
    if strategy == "table":
        return ("block", block.block_id)
    if strategy == "presentation":
        return ("block", block.block_id)
    return source_boundary


def _document_segments(
    document: ParsedDocument,
    *,
    config: ChunkingTemplateConfig,
    strategy: TemplateStrategy,
    table_aware_html: bool = False,
) -> list[_Segment]:
    assets = {asset.asset_id: asset.object_key for asset in document.assets}
    result: list[_Segment] = []
    active_headings: tuple[str, ...] = ()
    active_heading_ids: tuple[str, ...] = ()
    active_qa_id = ""
    active_qa_source_boundary: tuple[str, str] | None = None

    for block in document.blocks:
        block_text = block.text.strip()
        declared_path = tuple(item.strip() for item in block.parent_heading_path if item.strip())
        if block.type == "heading":
            heading_text = block_text.strip().lstrip("#").strip()
            active_headings = declared_path + ((heading_text,) if heading_text else ())
            active_heading_ids = active_heading_ids[: len(declared_path)] + (block.block_id,)
            block_text = f"{'#' * min(6, max(1, len(active_headings)))} {heading_text}".strip()
        else:
            if declared_path:
                active_headings = declared_path
                active_heading_ids = active_heading_ids[: len(declared_path)]
            elif not active_headings:
                active_heading_ids = ()

        heading_path = active_headings
        image_refs = tuple(
            dict.fromkeys(assets[asset_id] for asset_id in block.asset_refs if asset_id in assets)
        )
        if block.type == "table":
            if config.table_policy == "skip":
                continue
            if config.table_policy == "structured_text":
                block_text = _render_table(block, table_aware_html=table_aware_html)
        elif block.type == "image":
            if config.image_policy == "skip":
                continue
            if config.image_policy == "reference" and image_refs:
                refs = "\n".join(f"![图片]({item})" for item in image_refs)
                block_text = f"{block_text}\n{refs}".strip()
            elif not block_text and config.image_policy == "caption":
                block_text = "[图片]"
        elif block.type == "code" and block_text and not block_text.startswith("```"):
            block_text = f"```\n{block_text}\n```"

        if not block_text:
            continue
        if strategy == "qa":
            # 一个“问题标题/问题标记”开启新单元，随后答案块沿用该边界；slide/sheet
            # 永远是更高优先级的硬边界，切换来源单元时必须重置 QA 单元。
            source_boundary = _source_boundary(block)
            if source_boundary != active_qa_source_boundary:
                active_qa_id = ""
                active_qa_source_boundary = source_boundary
            if _looks_like_question(block_text, block.type) or not active_qa_id:
                active_qa_id = block.block_id
            boundary = (
                f"qa:{source_boundary[0]}",
                f"{source_boundary[1]}:{active_qa_id}",
            )
        else:
            boundary = _strategy_boundary(
                block,
                strategy=strategy,
                heading_path=heading_path,
            )
        base = _Segment(
            text=block_text,
            block_type=block.type,
            source_block_ids=(block.block_id,),
            locations=((block.block_id, block.location),),
            heading_path=heading_path,
            heading_source_ids=active_heading_ids,
            images=image_refs,
            boundary=boundary,
        )
        if strategy == "qa":
            result.extend(_split_qa_segment(base))
        else:
            result.append(base)
    return result


def _same_hard_source_boundary(left: _Segment, right: _Segment) -> bool:
    """标题可以跨页继承，但绝不跨工作表或幻灯片。"""

    def key(segment: _Segment) -> tuple[str, str]:
        if not segment.locations:
            return ("flow", "")
        location = segment.locations[0][1]
        if location.slide_no is not None:
            return ("slide", str(location.slide_no))
        if location.sheet:
            return ("sheet", location.sheet)
        return ("flow", "")

    return key(left) == key(right)


def _attach_headings_to_following_content(
    segments: Sequence[_Segment],
    *,
    inherit_titles: bool,
) -> list[_Segment]:
    """消除 heading-only 检索块，并把标题溯源绑定到紧随内容。

    解析器可能把同级标题暂时挂到前一标题下，而正文的 ``parent_heading_path``
    已恢复正确路径。与其固化那个短暂且错误的边界，这里以紧随正文的边界为准。
    只有文档末尾标题，或 slide/sheet 硬边界前的标题，会保留为独立块。
    """

    result: list[_Segment] = []
    pending: list[_Segment] = []
    for segment in segments:
        if segment.block_type == "heading":
            if pending and not _same_hard_source_boundary(pending[-1], segment):
                result.extend(pending)
                pending = []
            pending.append(segment)
            continue
        if not pending:
            result.append(segment)
            continue
        if not _same_hard_source_boundary(pending[-1], segment):
            result.extend(pending)
            pending = []
            result.append(segment)
            continue

        source_ids = tuple(
            _ordered_unique(
                [
                    source_id
                    for heading in pending
                    for source_id in (*heading.heading_source_ids, *heading.source_block_ids)
                ]
                + list(segment.heading_source_ids)
            )
        )
        locations = tuple(
            [location for heading in pending for location in heading.locations]
            + list(segment.locations)
        )
        images = tuple(
            _ordered_unique(
                [image for heading in pending for image in heading.images]
                + list(segment.images)
            )
        )
        text = segment.text
        if not inherit_titles:
            heading_text = "\n\n".join(item.text for item in pending if item.text.strip())
            text = f"{heading_text}\n\n{text}".strip()
        result.append(
            replace(
                segment,
                text=text,
                heading_source_ids=source_ids,
                locations=locations,
                images=images,
            )
        )
        pending = []
    result.extend(pending)
    return result


def _render_table(block: ParsedBlock, *, table_aware_html: bool = False) -> str:
    """把常见二维结构渲染为可检索 Markdown；未知结构稳定退化为 JSON。"""

    data = block.structured_data
    prefix = block.text.strip()
    headers: list[str] = []
    rows: list[list[Any]] = []

    if table_aware_html:
        html_markdown = _render_html_table(block)
        if html_markdown is not None:
            return html_markdown

    if isinstance(data, dict):
        raw_headers = data.get("headers") or data.get("columns")
        raw_rows = data.get("rows") or data.get("data")
        if isinstance(raw_headers, Sequence) and not isinstance(raw_headers, (str, bytes)):
            headers = [str(value) for value in raw_headers]
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
            if not headers and raw_rows and all(isinstance(row, Mapping) for row in raw_rows):
                headers = list(
                    dict.fromkeys(str(key) for row in raw_rows for key in row)
                )
            rows = _normalize_rows(raw_rows, headers)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if data and all(isinstance(row, Mapping) for row in data):
            headers = list(dict.fromkeys(str(key) for row in data for key in row))
        rows = _normalize_rows(data, headers)

    if rows:
        if not headers:
            width = max(len(row) for row in rows)
            headers = [f"列{index + 1}" for index in range(width)]
        width = len(headers)
        normalized_rows = [(row + [""] * width)[:width] for row in rows]
        lines = [
            "| " + " | ".join(_escape_table_cell(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend(
            "| " + " | ".join(_escape_table_cell(value) for value in row) + " |"
            for row in normalized_rows
        )
        table_text = "\n".join(lines)
    elif data is not None:
        table_text = "```json\n" + json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n```"
    else:
        table_text = prefix
        prefix = ""
    return f"{prefix}\n{table_text}".strip()


def _bounded_html_span(value: str | None) -> int:
    try:
        span = int(value or "1")
    except (TypeError, ValueError):
        return 1
    return max(1, min(span, 64))


def _normalize_html_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _html_table_source(block: ParsedBlock) -> str:
    data = block.structured_data
    if isinstance(data, Mapping):
        table_body = data.get("table_body")
        if isinstance(table_body, str) and re.search(r"(?is)<\s*table\b", table_body):
            return table_body
    if re.search(r"(?is)<\s*table\b", block.text):
        return block.text
    return ""


def _table_notes(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = _normalize_html_text(value)
        return [normalized] if normalized else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _ordered_unique(
            [
                normalized
                for item in value
                if (normalized := _normalize_html_text(str(item)))
            ]
        )
    return []


def _expanded_html_rows(rows: Sequence[_HtmlTableRow]) -> list[list[str]]:
    """展开 colspan/rowspan，得到列数稳定的检索网格。"""

    expanded: list[list[str]] = []
    # column -> (remaining rows, visible text)
    spans: dict[int, tuple[int, str]] = {}
    for row in rows:
        values: dict[int, str] = {}
        column = 0

        def consume_spans() -> None:
            nonlocal column
            while column in spans:
                remaining, text = spans[column]
                values[column] = text
                if remaining <= 1:
                    del spans[column]
                else:
                    spans[column] = (remaining - 1, text)
                column += 1

        for cell in row.cells:
            consume_spans()
            for offset in range(cell.colspan):
                values[column + offset] = cell.text if offset == 0 else ""
                if cell.rowspan > 1:
                    spans[column + offset] = (
                        cell.rowspan - 1,
                        cell.text if offset == 0 else "",
                    )
            column += cell.colspan
        consume_spans()
        if spans:
            last_column = max(spans)
            while column <= last_column:
                consume_spans()
                if column <= last_column and column not in spans:
                    values[column] = ""
                    column += 1
        width = max(values, default=-1) + 1
        expanded.append([values.get(index, "") for index in range(width)])
    return expanded


_table_value_pattern = re.compile(
    r"(?ix)^(?:"
    r"[×○●◯√✓✕✗—–-]+|"
    r"(?:yes|no|true|false|n/?a)|"
    r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*[%°a-z]+)?"
    r")$"
)


def _looks_like_table_data_row(values: Sequence[str]) -> bool:
    candidates = [value.strip() for value in values[1:] if value.strip()]
    if not candidates:
        return False
    data_like = sum(bool(_table_value_pattern.fullmatch(value)) for value in candidates)
    return data_like * 2 >= len(candidates)


def _html_header_row_count(
    parsed_rows: Sequence[_HtmlTableRow],
    expanded_rows: Sequence[Sequence[str]],
) -> int:
    if not parsed_rows:
        return 0
    explicit = 0
    for row in parsed_rows:
        if row.is_thead or any(cell.is_header for cell in row.cells):
            explicit += 1
        else:
            break
    if explicit:
        return explicit
    if len(parsed_rows) == 1:
        return 1

    # 无 thead/th 的 OCR 表格仍常用 colspan/rowspan 表达 2~3 层复合表头。
    # 最多推断三行，并在遇到明显数值/符号数据行时停止，至少留一行正文。
    first_is_complex = any(
        cell.colspan > 1 or cell.rowspan > 1 for cell in parsed_rows[0].cells
    )
    if not first_is_complex:
        return 1
    header_count = 1
    upper = min(3, len(parsed_rows) - 1)
    for index in range(1, upper):
        if _looks_like_table_data_row(expanded_rows[index]):
            break
        header_count += 1
    return header_count


def _render_html_table(block: ParsedBlock) -> str | None:
    source = _html_table_source(block)
    if not source:
        return None
    parser = _FirstHtmlTableParser()
    try:
        parser.feed(source)
        parser.close()
    except (ValueError, AssertionError):
        # 不完整 HTML 稳定退化到旧的 JSON/text 路径；绝不猜测或丢弃原文。
        return None
    if not parser.rows:
        return None

    rows = _expanded_html_rows(parser.rows)
    width = max((len(row) for row in rows), default=0)
    if not width:
        return None
    normalized_rows = [(row + [""] * width)[:width] for row in rows]
    header_count = _html_header_row_count(parser.rows, normalized_rows)
    header_count = max(1, min(header_count, len(normalized_rows)))

    data = block.structured_data if isinstance(block.structured_data, Mapping) else {}
    captions = _ordered_unique(
        ([parser.caption] if parser.caption else [])
        + _table_notes(data.get("table_caption"))
    )
    footnotes = _table_notes(data.get("table_footnote"))
    lines = [f"表题：{caption}" for caption in captions]
    lines.extend(
        "表头：" + " | ".join(_escape_table_cell(value) for value in row)
        for row in normalized_rows[: max(0, header_count - 1)]
    )
    markdown_header = normalized_rows[header_count - 1]
    lines.extend(
        [
            "| " + " | ".join(_escape_table_cell(value) for value in markdown_header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
    )
    lines.extend(
        "| " + " | ".join(_escape_table_cell(value) for value in row) + " |"
        for row in normalized_rows[header_count:]
    )
    # 表注属于解释表格列值的固定上下文，放在表头前缀中可随子块重复。
    if footnotes:
        insert_at = len(captions)
        lines[insert_at:insert_at] = [f"表注：{footnote}" for footnote in footnotes]
    return "\n".join(lines).strip()


def _normalize_rows(raw_rows: Sequence[Any], headers: Sequence[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in raw_rows:
        if isinstance(row, Mapping):
            ordered_headers = list(headers) or [str(key) for key in row]
            rows.append([row.get(header, "") for header in ordered_headers])
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            rows.append(list(row))
        else:
            rows.append([row])
    return rows


def _escape_table_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>")


_qa_marker = re.compile(
    r"(?im)(?=^(?:q(?:uestion)?|问题|问)\s*[:：]|^#{1,6}\s+.*[?？]\s*$)"
)


def _looks_like_question(text: str, block_type: str) -> bool:
    normalized = text.strip()
    return bool(
        re.match(r"^(?:#{1,6}\s+)?(?:q(?:uestion)?|问题|问)\s*[:：]", normalized, re.I)
        or (block_type == "heading" and normalized.rstrip("# ").endswith(("?", "？")))
    )


def _split_qa_segment(segment: _Segment) -> list[_Segment]:
    starts = [match.start() for match in _qa_marker.finditer(segment.text)]
    if len(starts) <= 1:
        return [segment]
    starts.append(len(segment.text))
    result: list[_Segment] = []
    for index in range(len(starts) - 1):
        text = segment.text[starts[index] : starts[index + 1]].strip()
        if text:
            result.append(replace(segment, text=text, boundary=("qa", f"{segment.boundary[1]}:{index}")))
    return result or [segment]


def _heading_prefix(path: tuple[str, ...], target_tokens: int) -> str:
    if not path:
        return ""
    prefix = " > ".join(path)
    max_tokens = max(4, target_tokens // 4)
    if count_tokens(prefix) > max_tokens:
        prefix = _tail_text_by_tokens(prefix, max_tokens)
    return f"标题路径：{prefix}"


def _render_segment_group(segments: Sequence[_Segment], *, config: ChunkingTemplateConfig) -> str:
    if not segments:
        return ""
    body = "\n\n".join(segment.text.strip() for segment in segments if segment.text.strip())
    if not body:
        return ""
    if config.inherit_titles and segments[0].block_type != "heading":
        prefix = _heading_prefix(segments[0].heading_path, config.target_tokens)
        if prefix:
            return f"{prefix}\n\n{body}"
    return body


def _group_key(
    segment: _Segment,
    *,
    config: ChunkingTemplateConfig,
) -> tuple[tuple[str, str], tuple[str, ...]]:
    heading_key = segment.heading_path if config.inherit_titles else ()
    return segment.boundary, heading_key


def _split_oversize_segment(
    segment: _Segment,
    *,
    config: ChunkingTemplateConfig,
) -> list[_Segment]:
    if count_tokens(_render_segment_group([segment], config=config)) <= config.target_tokens:
        return [segment]

    prefix_tokens = 0
    if config.inherit_titles and segment.block_type != "heading":
        prefix_tokens = count_tokens(_heading_prefix(segment.heading_path, config.target_tokens)) + 1
    body_limit = max(1, config.target_tokens - prefix_tokens)
    if segment.block_type == "table":
        table_parts = _split_markdown_table(segment.text, body_limit)
        if len(table_parts) > 1:
            return [replace(segment, text=part) for part in table_parts]

    parts = _split_text_by_tokens(
        segment.text,
        limit=body_limit,
        overlap=min(config.overlap_tokens, body_limit - 1),
        separators=config.separators,
    )
    return [replace(segment, text=text) for text in parts]


def _split_markdown_table(text: str, body_limit: int) -> list[str]:
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index in range(len(lines) - 1)
            if lines[index].lstrip().startswith("|")
            and re.match(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+", lines[index + 1])
        ),
        None,
    )
    if header_index is None:
        return [text]
    prefix_lines = lines[:header_index]
    header = lines[header_index : header_index + 2]
    rows = lines[header_index + 2 :]
    if not rows:
        return [text]

    fixed = "\n".join(prefix_lines + header).strip()
    parts: list[str] = []
    current_rows: list[str] = []
    for row in rows:
        candidate = "\n".join([fixed, *current_rows, row]).strip()
        if current_rows and count_tokens(candidate) > body_limit:
            parts.append("\n".join([fixed, *current_rows]).strip())
            current_rows = []
        single = "\n".join([fixed, row]).strip()
        if count_tokens(single) > body_limit:
            available = max(1, body_limit - count_tokens(fixed) - 1)
            for row_part in _split_text_by_tokens(
                row,
                limit=available,
                overlap=0,
                separators=("<br>", " ", "|"),
            ):
                parts.append(f"{fixed}\n{row_part}".strip())
            continue
        current_rows.append(row)
    if current_rows:
        parts.append("\n".join([fixed, *current_rows]).strip())
    return parts or [text]


def _tail_segments(segments: Sequence[_Segment], overlap_tokens: int) -> list[_Segment]:
    if overlap_tokens <= 0:
        return []
    remaining = overlap_tokens
    reversed_tail: list[_Segment] = []
    for segment in reversed(segments):
        segment_tokens = count_tokens(segment.text)
        if not segment_tokens:
            continue
        take = min(remaining, segment_tokens)
        tail_text = _tail_text_by_tokens(segment.text, take).strip()
        if tail_text:
            reversed_tail.append(replace(segment, text=tail_text))
        remaining -= take
        if remaining <= 0:
            break
    return list(reversed(reversed_tail))


def _head_end_for_token_limit(text: str, start: int, limit: int) -> int:
    """返回不破坏 Unicode 字符且 token 数不超过 limit 的最大字符下标。"""

    low, high = start + 1, len(text)
    best = start
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[start:middle]) <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best or min(len(text), start + 1)


def _tail_text_by_tokens(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if count_tokens(text) <= limit:
        return text
    low, high = 0, len(text) - 1
    best = len(text)
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[middle:]) <= limit:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return text[best:]


def _preferred_split(text: str, start: int, hard_end: int, separators: Sequence[str]) -> int:
    """在后 40% 范围内优先选择自然分隔符，否则使用硬字符边界。"""

    minimum = start + max(1, int((hard_end - start) * 0.6))
    best = -1
    for separator in separators:
        found = text.rfind(separator, minimum, hard_end)
        if found >= minimum:
            best = max(best, found + len(separator))
    return best if best > start else hard_end


def _split_text_by_tokens(
    text: str,
    *,
    limit: int,
    overlap: int,
    separators: Sequence[str],
) -> list[str]:
    """按字符边界进行 token 预算切分，避免 token 字节片段导致中文出现 U+FFFD。"""

    if not text.strip():
        return []
    if count_tokens(text) <= limit:
        return [text.strip()]
    result: list[str] = []
    start = 0
    while start < len(text):
        hard_end = _head_end_for_token_limit(text, start, limit)
        if hard_end >= len(text):
            part = text[start:].strip()
            if part:
                result.append(part)
            break
        end = _preferred_split(text, start, hard_end, separators)
        part = text[start:end].strip()
        if not part:
            end = hard_end
            part = text[start:end].strip()
        if part:
            result.append(part)
        if overlap <= 0:
            next_start = end
        else:
            tail = _tail_text_by_tokens(text[start:end], overlap)
            next_start = max(start + 1, end - len(tail))
        start = next_start
        while start < len(text) and text[start].isspace():
            start += 1
    return result


def _merge_segments(
    document: ParsedDocument,
    segments: Sequence[_Segment],
    *,
    template: StructuredChunkingTemplate,
    config: ChunkingTemplateConfig,
    file_title: str,
) -> list[dict[str, Any]]:
    atomic = [
        part
        for segment in segments
        for part in _split_oversize_segment(segment, config=config)
    ]
    groups: list[list[_Segment]] = []
    current: list[_Segment] = []
    current_key: tuple[tuple[str, str], tuple[str, ...]] | None = None

    for segment in atomic:
        segment_key = _group_key(segment, config=config)
        if current and segment_key != current_key:
            groups.append(current)
            current = []
        current_key = segment_key

        candidate = [*current, segment]
        if current and count_tokens(_render_segment_group(candidate, config=config)) > config.target_tokens:
            groups.append(current)
            current = _tail_segments(current, config.overlap_tokens)
            if current and count_tokens(_render_segment_group([*current, segment], config=config)) > config.target_tokens:
                current = []
        current.append(segment)
    if current:
        groups.append(current)

    chunks: list[dict[str, Any]] = []
    for position, group in enumerate(groups):
        content = _render_segment_group(group, config=config).strip()
        if not content:
            continue
        source_ids = _ordered_unique(
            [
                source_id
                for segment in group
                for source_id in (*segment.heading_source_ids, *segment.source_block_ids)
            ]
        )
        images = _ordered_unique([image for segment in group for image in segment.images])
        locations = _ordered_locations(group)
        heading_path = list(group[0].heading_path) if group else []
        block_types = {segment.block_type for segment in group if segment.block_type != "heading"}
        chunk_type = next(iter(block_types)) if len(block_types) == 1 else "text"
        if not block_types and any(segment.block_type == "heading" for segment in group):
            chunk_type = "heading"

        stable_payload = {
            "source_sha256": document.source_sha256,
            "template_id": template.template_id,
            "template_version": template.version,
            "position": position,
            "source_block_ids": source_ids,
            "content": content,
        }
        chunk_id = "chk_" + hashlib.sha256(
            json.dumps(
                stable_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]
        chunks.append(
            {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "content": content,
                "images": images or None,
                "type": chunk_type,
                "file_title": file_title,
                "position_int": position,
                "provenance": {
                    "source_block_ids": source_ids,
                    "locations": locations,
                    "heading_path": heading_path,
                    "parser_id": document.parser_id,
                    "parser_version": document.parser_version,
                    "template_id": template.template_id,
                    "template_version": template.version,
                },
            }
        )
    if not chunks:
        raise DocumentParseError(
            ERROR_OUTPUT_EMPTY,
            "切块结果为空",
            retryable=False,
        )
    return chunks


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _ordered_locations(segments: Sequence[_Segment]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in segments:
        for block_id, location in segment.locations:
            value = {"block_id": block_id, **location.model_dump(mode="json", exclude_none=True)}
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                result.append(value)
    return result


def suggest_template(document: ParsedDocument) -> str:
    """返回模板建议；调用者仍须显式选择，默认模板不会随建议变化。"""

    source_format = document.source_format.lower().lstrip(".")
    if source_format in {"ppt", "pptx", "odp"} or any(
        block.location.slide_no is not None for block in document.blocks
    ):
        return "presentation"
    if source_format in {"xls", "xlsx", "csv", "ods"} or (
        document.blocks
        and sum(block.type == "table" for block in document.blocks) / len(document.blocks) >= 0.5
    ):
        return "table"

    searchable = "\n".join(block.text for block in document.blocks).lower()
    headings = [block.text.strip().lower() for block in document.blocks if block.type == "heading"]
    if any(name in searchable for name in ("abstract", "摘要")) and any(
        name in searchable for name in ("references", "参考文献")
    ):
        return "paper"
    if sum(bool(re.search(r"(?:^|\n)(?:q(?:uestion)?|问题|问)\s*[:：]", block.text, re.I)) for block in document.blocks) >= 2:
        return "qa"
    manual_markers = ("安全", "警告", "安装", "操作", "维护", "故障", "说明", "warning", "installation")
    if sum(any(marker in heading for marker in manual_markers) for heading in headings) >= 2:
        return "manual"
    return DEFAULT_TEMPLATE_ID


def create_default_chunking_template_registry() -> ChunkingTemplateRegistry:
    registry = ChunkingTemplateRegistry()
    definitions: tuple[
        tuple[str, str, str, TemplateStrategy, ChunkingTemplateConfig], ...
    ] = (
        (
            "general",
            "通用文档",
            "按标题路径与自然顺序合并文本，适合未知或混合结构文档。",
            "general",
            ChunkingTemplateConfig(target_tokens=512, overlap_tokens=64),
        ),
        (
            "manual",
            "产品手册",
            "按章节硬边界切块并继承标题路径，适合操作与安全手册。",
            "manual",
            ChunkingTemplateConfig(target_tokens=480, overlap_tokens=48),
        ),
        (
            "paper",
            "论文",
            "保留论文章节边界并使用较大的上下文窗口。",
            "paper",
            ChunkingTemplateConfig(target_tokens=768, overlap_tokens=96),
        ),
        (
            "qa",
            "问答对",
            "优先保持问题与答案成对，不跨问答单元合并。",
            "qa",
            ChunkingTemplateConfig(target_tokens=384, overlap_tokens=0),
        ),
        (
            "table",
            "表格",
            "将二维结构渲染为带表头的检索文本，不跨表格或工作表。",
            "table",
            ChunkingTemplateConfig(target_tokens=512, overlap_tokens=0),
        ),
        (
            "presentation",
            "演示文稿",
            "按幻灯片形成独立上下文，绝不跨页合并。",
            "presentation",
            ChunkingTemplateConfig(target_tokens=384, overlap_tokens=0),
        ),
    )
    for template_id, name, description, strategy, config in definitions:
        registry.register(
            StructuredChunkingTemplate(
                template_id=template_id,
                version=DEFAULT_TEMPLATE_VERSION,
                name=name,
                description=description,
                strategy=strategy,
                default_config=config,
            )
        )
    for version in (MANUAL_TABLE_AWARE_BASE_VERSION, MANUAL_TABLE_AWARE_VERSION):
        registry.register(
            StructuredChunkingTemplate(
                template_id="manual",
                version=version,
                name="产品手册",
                description=(
                    "按章节硬边界切块并继承标题路径；HTML 表格按行分组，"
                    "每个子块重复表题与复合表头。"
                ),
                strategy="manual",
                default_config=ChunkingTemplateConfig(
                    target_tokens=480,
                    overlap_tokens=48,
                ),
            )
        )
    return registry


default_chunking_template_registry = create_default_chunking_template_registry()


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    template_id: str | None = None,
    template_version: str | None = None,
    config: ChunkingTemplateConfig | Mapping[str, Any] | None = None,
    parser_config: Mapping[str, Any] | None = None,
    file_title: str | None = None,
    registry: ChunkingTemplateRegistry | None = None,
) -> list[dict[str, Any]]:
    """公共入口；未指定模板时始终使用 general，而不是自动采用 suggestion。"""

    chunks, _ = chunk_parsed_document_with_profile(
        document,
        template_id=template_id or DEFAULT_TEMPLATE_ID,
        template_version=template_version,
        config=config,
        parser_config=parser_config,
        file_title=file_title,
        registry=registry,
    )
    return chunks


def chunk_parsed_document_with_profile(
    document: ParsedDocument,
    *,
    template_id: str = DEFAULT_TEMPLATE_ID,
    template_version: str | None = None,
    file_title: str | None = None,
    parser_config: Mapping[str, Any] | None = None,
    config: ChunkingTemplateConfig | Mapping[str, Any] | None = None,
    registry: ChunkingTemplateRegistry | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """切块并返回实际模板/配置快照，供 DocumentVersion 与 API 回显。"""

    if config is not None and parser_config is not None:
        raise DocumentParseError(
            ERROR_CONFIG_INVALID,
            "config 与 parser_config 不能同时提供",
            retryable=False,
        )
    target_registry = registry or default_chunking_template_registry
    template = target_registry.resolve(template_id, template_version)
    resolved_config = (
        ChunkingTemplateConfig.from_parser_config(
            parser_config,
            defaults=template.default_config,
        )
        if parser_config is not None
        else resolve_chunking_config(template, config)
    )
    chunks = template.chunk(document, config=resolved_config, file_title=file_title)
    resolved = {
        "template_id": template.template_id,
        "template_version": template.version,
        "suggested_template_id": suggest_template(document),
        "config": resolved_config.model_dump(mode="json"),
    }
    return chunks, resolved


def list_chunking_templates(
    registry: ChunkingTemplateRegistry | None = None,
) -> list[ChunkingTemplateInfo]:
    return (registry or default_chunking_template_registry).list_infos()


__all__ = [
    "ChunkingTemplate",
    "ChunkingTemplateConfig",
    "ChunkingTemplateRegistry",
    "DEFAULT_TEMPLATE_ID",
    "DEFAULT_TEMPLATE_VERSION",
    "MANUAL_TABLE_AWARE_BASE_VERSION",
    "MANUAL_TABLE_AWARE_VERSION",
    "StructuredChunkingTemplate",
    "chunk_parsed_document",
    "chunk_parsed_document_with_profile",
    "count_tokens",
    "create_default_chunking_template_registry",
    "default_chunking_template_registry",
    "list_chunking_templates",
    "resolve_chunking_config",
    "suggest_template",
]
