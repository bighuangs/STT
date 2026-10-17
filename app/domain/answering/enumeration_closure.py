"""来源有界、可重放的开放枚举表格记录闭合回执。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self, TypeAlias
import unicodedata

from pydantic import Field, model_validator

from app.domain.answering._base import (
    EVIDENCE_ID_PATTERN,
    FACET_ID_PATTERN,
    REPRESENTATION_ID_PATTERN,
    SHA256_PATTERN,
    WITNESS_ID_PATTERN,
    AnsweringContractModel,
    sha256_digest,
    stable_content_hash,
    stable_identifier,
)
from app.domain.answering.item_boundary import (
    is_image_derivation_metadata_start,
    is_markdown_image_only_line,
    markdown_table_row_shape,
    top_level_list_item_body_span,
)
from app.domain.answering.fact_set_closure import (
    QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION,
    QueryAnchoredScopedFactSetClosureReceipt,
)
from app.domain.answering.markdown_table import (
    MarkdownTableBlock,
    scan_markdown_table_blocks,
)
from app.domain.answering.question import (
    QuestionContract,
    QuestionObligation,
    RequiredOutputForm,
)
from app.domain.answering.witness import (
    PACKED_TRUNCATION_MARKER,
    EvidenceWitness,
    WitnessAtomKind,
    WitnessCatalogReceipt,
    WitnessUniverseStatus,
)
from app.domain.query_planning import FacetClosure
from app.domain.safe_representation import (
    MARKDOWN_IMAGE_CONTROL_TEXT,
    SafeSpanRelation,
    derive_safe_representation,
)


QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION = "query-anchored-table-row-closure-receipt.v4"
QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-procedure-list-closure-receipt.v1"
)
QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-scoped-procedure-list-closure-receipt.v4"
)
QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-scoped-table-selection-closure-receipt.v2"
)
QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-scoped-bullet-list-closure-receipt.v1"
)
QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-inline-action-chain-closure-receipt.v1"
)
QUERY_ANCHOR_NORMALIZATION_VERSION = "nfkc-default-ignorable-markup-aware-token.v3"


_MARKDOWN_INLINE_MARKERS = frozenset("*_~`")
_CONFUSABLE_ASCII_SKELETON = {
    # 只用于保守的 catalog-wide 同一性扫描；不把跨脚本字符用于查询锚定。
    "α": "a",
    "β": "b",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "y",
    "χ": "x",
    "а": "a",
    "в": "b",
    "е": "e",
    "к": "k",
    "м": "m",
    "н": "h",
    "о": "o",
    "р": "p",
    "с": "c",
    "т": "t",
    "у": "y",
    "х": "x",
}


def _is_default_ignorable(character: str) -> bool:
    """覆盖常见 default-ignorable code point，避免视觉同一键被控制符拆开。"""

    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or 0x180B <= codepoint <= 0x180D
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _normalize_anchor_text(value: str) -> str:
    """统一标识和查询文本，移除空白、标点与不可见格式控制符。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
        and not _is_default_ignorable(character)
    )


def _identity_scan_text(value: str, *, map_confusables: bool = True) -> str:
    """生成只用于拒绝潜在同键补充的保守 plain-text 身份投影。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_tags = re.sub(r"<[^>\r\n]{1,256}>", "", normalized)
    return "".join(
        (_CONFUSABLE_ASCII_SKELETON.get(character, character) if map_confusables else character)
        for character in without_tags
        if not _is_default_ignorable(character) and character not in _MARKDOWN_INLINE_MARKERS
    )


def _normalize_table_cell(value: str) -> str:
    """保留结构标点，仅容忍 Unicode 兼容形式、大小写和空白差异。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


_GENERIC_IDENTIFIER_KEYS = frozenset(
    _normalize_anchor_text(value)
    for value in (
        "id",
        "identifier",
        "header",
        "column",
        "row",
        "no",
        "no.",
        "number",
        "index",
        "name",
        "title",
        "item",
        "entry",
        "product",
        "model",
        "type",
        "category",
        "feature",
        "function",
        "key",
        "value",
        "label",
        "field",
        "attribute",
        "property",
        "parameter",
        "option",
        "code",
        "description",
        "details",
        "content",
        "status",
        "remark",
        "note",
        "序号",
        "编号",
        "索引",
        "表头",
        "列",
        "列名",
        "行",
        "标识",
        "标识符",
        "名称",
        "标题",
        "项目",
        "条目",
        "产品",
        "型号",
        "类型",
        "类别",
        "功能",
        "键",
        "值",
        "标签",
        "字段",
        "属性",
        "参数",
        "选项",
        "代码",
        "描述",
        "详情",
        "内容",
        "状态",
        "备注",
        "说明",
        "同上",
        "同前",
        "同前项",
        "同左",
        "见上",
        "如上",
        "续",
        "same",
        "same as above",
        "ditto",
    )
)

_ENGLISH_IDENTIFIER_HEADER_TERMS = frozenset(
    {
        "id",
        "identifier",
        "name",
        "code",
        "object",
        "device",
        "product",
        "model",
        "item",
        "entry",
        "button",
        "key",
        "module",
        "component",
        "type",
        "category",
    }
)
_ENGLISH_IDENTIFIER_HEADER_COMPOUNDS = frozenset(
    f"{prefix}{suffix}"
    for prefix in _ENGLISH_IDENTIFIER_HEADER_TERMS
    for suffix in ("id", "identifier", "name", "code")
)
_CHINESE_IDENTIFIER_HEADER_SUFFIXES = tuple(
    _normalize_anchor_text(value)
    for value in (
        "标识",
        "标识符",
        "名称",
        "编号",
        "代码",
        "对象",
        "设备",
        "产品",
        "型号",
        "项目",
        "条目",
        "按键",
        "按钮",
        "模块",
        "部件",
        "组件",
        "类型",
        "类别",
    )
)


@dataclass(frozen=True)
class _PhysicalLine:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _MarkdownRow:
    line: _PhysicalLine
    cells: tuple[str, ...]
    is_separator: bool


@dataclass(frozen=True)
class _SimpleMarkdownTable:
    start: int
    end: int
    header: _MarkdownRow
    separator: _MarkdownRow
    data_rows: tuple[_MarkdownRow, ...]
    column_count: int
    witness_row_index: int


@dataclass(frozen=True)
class _OrderedListLine:
    line: _PhysicalLine
    content_start: int
    content_end: int
    ordinal: int
    marker_style: str
    body: str


@dataclass(frozen=True)
class _ProcedureListBlock:
    start: int
    end: int
    title: _PhysicalLine
    title_content_start: int
    title_content_end: int
    normalized_title: str
    items: tuple[_OrderedListLine, ...]
    marker_style: str
    gap_kinds: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ScopedListAuthorityRepresentation:
    """新 scoped receipt 构建时必须提供的完整 representation authority。"""

    evidence_id: str
    representation_id: str
    canonical_text: str
    packed_text: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"ev_[0-9a-f]{40}", self.evidence_id) is None:
            raise ValueError("scoped list authority Evidence ID is invalid")
        if re.fullmatch(r"rep_[0-9a-f]{40}", self.representation_id) is None:
            raise ValueError("scoped list authority representation ID is invalid")
        if not self.canonical_text or not self.packed_text:
            raise ValueError("scoped list authority texts must be non-empty")


ScopedTableAuthorityRepresentation: TypeAlias = ScopedListAuthorityRepresentation
ScopedBulletAuthorityRepresentation: TypeAlias = ScopedListAuthorityRepresentation


@dataclass(frozen=True)
class _ScopedProcedureCandidate:
    block: _ProcedureListBlock
    anchor_field: Literal["source_query", "information_need"]
    anchor_identifier: str
    anchor_match_keys: tuple[str, ...]
    selection_kind: Literal[
        "complete_sequence",
        "query_selected_continuation",
        "query_selected_branch",
        "query_selected_prefix",
    ]
    selection_priority: int
    exact_identifier_match: bool
    longest_ascii_entity_token_length: int
    discriminative_match_key_count: int
    body_match_key_count: int
    blocked_reason: str | None
    prefix_trigger_hash: str | None
    excluded_successor_ordinal: int | None
    excluded_successor_hash: str | None
    boundary_rule_version: str | None


@dataclass(frozen=True)
class _ScopedProcedureSequence:
    items: tuple[_OrderedListLine, ...]
    gap_kinds: tuple[tuple[str, ...], ...]
    last_position: int
    blocked_reason: str | None


@dataclass(frozen=True)
class _ScopedProcedurePrefixSelection:
    items: tuple[_OrderedListLine, ...]
    gap_kinds: tuple[tuple[str, ...], ...]
    anchor_field: Literal["source_query", "information_need"]
    anchor_identifier: str
    anchor: _PhysicalLine
    anchor_trimmed: tuple[int, int, str]
    anchor_match_keys: tuple[str, ...]
    prefix_trigger_hash: str
    excluded_successor_ordinal: int
    excluded_successor_hash: str
    boundary_rule_version: Literal["explicit-query-before-successor.v1"]


@dataclass(frozen=True)
class _ScopedTableSelectionCandidate:
    authority_index: int
    table_index: int
    canonical_table: MarkdownTableBlock
    packed_table: MarkdownTableBlock
    selector_column_index: int
    selector_values: tuple[str, ...]
    selected_row_indices: tuple[int, ...]
    anchor_field: Literal["source_query", "information_need"]


@dataclass(frozen=True)
class _BulletListLine:
    line: _PhysicalLine
    content_start: int
    content_end: int
    body_start: int
    body_end: int
    marker_style: str
    body: str


@dataclass(frozen=True)
class _BulletListBlock:
    start: int
    end: int
    anchor: _PhysicalLine
    anchor_content_start: int
    anchor_content_end: int
    normalized_anchor: str
    anchor_kind: Literal[
        "title_closed_intro",
        "ordered_closed_intro",
        "prose_closed_intro",
        "bounded_title",
    ]
    title: _PhysicalLine | None
    title_content_start: int | None
    title_content_end: int | None
    normalized_title: str
    items: tuple[_BulletListLine, ...]
    marker_style: str
    local_section_start: int
    local_section_end: int


@dataclass(frozen=True)
class _ScopedBulletListCandidate:
    authority_index: int
    block_index: int
    canonical_block: _BulletListBlock
    packed_block: _BulletListBlock
    anchor_field: Literal["source_query", "information_need"]
    anchor_match_keys: tuple[str, ...]


_ARABIC_ORDERED_LIST_RE = re.compile(
    r"^(?P<ordinal>\d{1,3})(?:(?P<dot>[.)．])(?=\s|[^\d\s])|(?P<dunhao>、)(?!\d))"
    r"\s*(?P<body>\S(?:.*\S)?)$"
)
_CHINESE_ORDERED_LIST_RE = re.compile(
    r"^(?P<ordinal>[一二三四五六七八九十]{1,3})(?P<marker>[、.)．])"
    r"\s*(?P<body>\S(?:.*\S)?)$"
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")
_LEADING_SECTION_NUMBER_RE = re.compile(
    r"^(?:(?:第\s*)?\d+(?:\s*[.．]\s*\d+){0,5}\s*(?:章|节|章节)?\s*)"
)
_LEADING_CHINESE_SECTION_NUMBER_RE = re.compile(
    r"^(?:第\s*)?[一二三四五六七八九十]{1,3}\s*(?:[、.．)]|章|节)\s*"
)
_GENERIC_PROCEDURE_TITLES = frozenset(
    _normalize_anchor_text(value)
    for value in (
        "步骤",
        "操作步骤",
        "操作流程",
        "流程",
        "操作方法",
        "procedure",
        "steps",
        "instructions",
    )
)


def _trimmed_physical_line(line: _PhysicalLine) -> tuple[int, int, str] | None:
    left = 0
    right = len(line.text)
    while left < right and line.text[left].isspace():
        left += 1
    while right > left and line.text[right - 1].isspace():
        right -= 1
    if left >= right:
        return None
    return line.start + left, line.start + right, line.text[left:right]


def _chinese_ordinal(value: str) -> int | None:
    digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" not in value:
        return digits.get(value)
    left, _, right = value.partition("十")
    tens = digits.get(left, 1) if left else 1
    ones = digits.get(right, 0) if right else 0
    result = tens * 10 + ones
    return result if 1 <= result <= 64 else None


def _ordered_list_line(line: _PhysicalLine) -> _OrderedListLine | None:
    trimmed = _trimmed_physical_line(line)
    if trimmed is None:
        return None
    content_start, content_end, content = trimmed
    matched = _ARABIC_ORDERED_LIST_RE.fullmatch(content)
    if matched is not None:
        ordinal = int(matched.group("ordinal"))
        marker = matched.group("dot") or "、"
        if marker == "．":
            marker = "."
        marker_style = f"arabic:{marker}"
        body = matched.group("body")
    else:
        matched = _CHINESE_ORDERED_LIST_RE.fullmatch(content)
        if matched is None:
            return None
        ordinal = _chinese_ordinal(matched.group("ordinal")) or 0
        marker = matched.group("marker")
        if marker == "．":
            marker = "."
        marker_style = f"chinese:{marker}"
        body = matched.group("body")
    if not 1 <= ordinal <= 64 or top_level_list_item_body_span(content) is None:
        return None
    return _OrderedListLine(
        line=line,
        content_start=content_start,
        content_end=content_end,
        ordinal=ordinal,
        marker_style=marker_style,
        body=body,
    )


def _normalized_procedure_title(value: str) -> str:
    title = _MARKDOWN_HEADING_RE.sub("", value.strip())
    if len(title) >= 4 and title[:2] == title[-2:] and title[:2] in {"**", "__"}:
        title = title[2:-2].strip()
    title = _LEADING_SECTION_NUMBER_RE.sub("", title).strip(" \t:：-—")
    title = _LEADING_CHINESE_SECTION_NUMBER_RE.sub("", title).strip(" \t:：-—")
    return _normalize_anchor_text(title)


def _locate_complete_procedure_list_block(text: str) -> _ProcedureListBlock:
    """只接受覆盖唯一 representation 的标题加连续编号列表。"""

    lines = _physical_lines(text)
    nonempty = tuple(
        (line, trimmed) for line in lines if (trimmed := _trimmed_physical_line(line)) is not None
    )
    if len(nonempty) < 3:
        raise ValueError("procedure list closure requires a title and at least two steps")
    title, title_trimmed = nonempty[0]
    if _ordered_list_line(title) is not None or _parse_simple_markdown_row(title) is not None:
        raise ValueError("procedure list closure requires an explicit non-list title")
    item_lines = tuple(_ordered_list_line(line) for line, _ in nonempty[1:])
    if any(item is None for item in item_lines):
        raise ValueError("procedure list closure permits only ordered list items after the title")
    items = tuple(item for item in item_lines if item is not None)
    if tuple(item.line.index for item in items) != tuple(
        range(title.index + 1, title.index + 1 + len(items))
    ):
        raise ValueError("procedure list title and items must be physically contiguous")
    if tuple(item.ordinal for item in items) != tuple(range(1, len(items) + 1)):
        raise ValueError("procedure list ordinals must start at one and increase without gaps")
    marker_styles = {item.marker_style for item in items}
    if len(marker_styles) != 1:
        raise ValueError("procedure list items must use one stable marker style")
    normalized_bodies = tuple(_normalize_table_cell(item.body) for item in items)
    if any(not item for item in normalized_bodies) or len(normalized_bodies) != len(
        set(normalized_bodies)
    ):
        raise ValueError("procedure list items must be non-empty and semantically distinct")
    normalized_title = _normalized_procedure_title(title_trimmed[2])
    if (
        not _is_specific_identifier(normalized_title)
        or normalized_title in _GENERIC_PROCEDURE_TITLES
    ):
        raise ValueError("procedure list title must be a specific query anchor")
    block_start = title_trimmed[0]
    block_end = items[-1].content_end
    if any(not character.isspace() for character in text[:block_start] + text[block_end:]):
        raise ValueError("procedure list block must cover the full representation")
    return _ProcedureListBlock(
        start=block_start,
        end=block_end,
        title=title,
        title_content_start=title_trimmed[0],
        title_content_end=title_trimmed[1],
        normalized_title=normalized_title,
        items=items,
        marker_style=next(iter(marker_styles)),
    )


def _procedure_list_structure_payload(block: _ProcedureListBlock) -> dict[str, Any]:
    return {
        "title": block.normalized_title,
        "title_line_index": block.title.index,
        "first_item_line_index": block.items[0].line.index,
        "marker_style": block.marker_style,
        "ordinals": [item.ordinal for item in block.items],
        "items": [_normalize_table_cell(item.body) for item in block.items],
    }


def _validate_procedure_list_pair(
    *,
    canonical_block: _ProcedureListBlock,
    packed_block: _ProcedureListBlock,
) -> str:
    canonical_structure = _procedure_list_structure_payload(canonical_block)
    packed_structure = _procedure_list_structure_payload(packed_block)
    if canonical_structure != packed_structure:
        raise ValueError("canonical and packed procedure list structures differ")
    return stable_content_hash(canonical_structure)


def _select_title_anchor(
    *,
    question: QuestionContract,
    obligation: QuestionObligation,
    normalized_title: str,
) -> tuple[Literal["source_query", "information_need"], str]:
    anchors: tuple[tuple[Literal["source_query", "information_need"], str], ...] = (
        ("source_query", question.source_query),
        ("information_need", obligation.information_need),
    )
    for anchor_field, anchor in anchors:
        if _anchor_contains_identifier(normalized_title, anchor):
            return anchor_field, normalized_title
    raise ValueError("procedure list title is not explicitly anchored in the query")


_SCOPED_TITLE_GENERIC_SUFFIXES = tuple(
    _normalize_anchor_text(value)
    for value in (
        "连接方法",
        "配置方法",
        "设置方法",
        "安装步骤",
        "操作步骤",
        "操作流程",
        "使用方法",
        "处理步骤",
        "方法",
        "步骤",
        "流程",
        "procedure",
        "instructions",
        "steps",
    )
)
_SCOPED_OPEN_ENUMERATION_RE = re.compile(
    r"(?:包括但不限于|但不限于|等等|诸如|\betc\.?\b|"
    r"等(?:打印介质|内容|操作|步骤|项目|类型|情况|选项|功能|信息)?(?=[。；;，,\s]|$))",
    re.IGNORECASE,
)
_SCOPED_OPEN_REFERENCE_RE = re.compile(
    r"(?:参阅|详见|另见|请?查阅|参考)(?:\s*第?\s*[0-9一二三四五六七八九十]+)?"
    r"|\b(?:see|refer\s+to)\b",
    re.IGNORECASE,
)
_SCOPED_PARENTHETICAL_REFERENCE_RE = re.compile(
    r"(?P<open>[(（])(?P<body>[^()（）\r\n]{1,320})(?P<close>[)）])"
)
_SCOPED_PROCEDURE_NOTE_RE = re.compile(
    r"^(?:注|注意|提示|说明|备注)\s*[:：]",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_CLOSED_INTRO_RE = re.compile(
    r"(?:以下|如下|下列).{0,24}(?:步骤|操作|流程|方法)"
    r"|(?:步骤|操作|流程|方法).{0,12}(?:如下|以下)"
    r"|(?:按照|按).{0,24}(?:步骤|顺序).{0,12}(?:操作|处理|进行)"
    r"|(?:设置|安装|处理|操作)\s*[:：]\s*$",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_BRANCH_INTRO_RE = re.compile(
    r"^(?:如果|若|当|如)(?=.{0,96}(?:下一步|以下|如下|步骤|操作|处理))",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_TEMPORAL_QUERY_RE = re.compile(
    r"(?:之后|以后|随后|后续|完成后|输入[^，。；;]{0,32}后|取出[^，。；;]{0,32}后|更换[^，。；;]{0,32}后)"
    r"|\b(?:after|following|subsequent)\b",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_CONDITIONAL_QUERY_RE = re.compile(
    r"(?:如果|若|当).{0,96}(?:未|没有|无法|不能|看不到|失败|卡纸)"
    r"|\b(?:if|when)\b",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_BEFORE_QUERY_RE = re.compile(
    r"(?:^|[，,。；;！？?\s])(?:在|于)?"
    r"(?P<trigger>[\u3400-\u9fffA-Za-z0-9._-]{2,24})前"
    r"(?=(?:需|要|应|必须|先|如何|[，,。；;！？?\s]|$))",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_PHASE_START_RE = re.compile(
    r"(?:系统)?(?:自动|开始|正式|进入|执行|进行|启动|弹出)"
    r"|(?:过程|阶段)(?:开始|启动)",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_FINITE_SUBLIST_RE = re.compile(
    r"(?:有|为|包括|支持)(?P<count>[2-9]|[二两三四五六七八九十])\s*"
    r"(?:种|类|模式|方式|选项)"
)
_SCOPED_PROCEDURE_GENERIC_BIGRAMS = frozenset(
    {
        "以下",
        "如下",
        "操作",
        "步骤",
        "流程",
        "方法",
        "设置",
        "安装",
        "处理",
        "进行",
        "完成",
        "需要",
        "说明",
        "具体",
        "打印",
        "印机",
        "打印机",
        "系统",
        "界面",
    }
)
_MAX_SCOPED_PROCEDURE_GAP_LINES = 80
_MAX_SCOPED_PROCEDURE_GAP_CHARACTERS = 12_000
_SCOPED_PROCEDURE_OPTIONAL_NOTE_RE = re.compile(
    r"(?:默认为|默认是|还可以|也可以|亦可|可通过|可选择|无需|不必|optional)",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_REQUIRED_NOTE_ACTION_RE = re.compile(
    r"(?:必须|务必|需要|应当|应先|请先|请再|不得|禁止)"
    r"|\b(?:must|required|shall)\b",
    re.IGNORECASE,
)
_SCOPED_PROCEDURE_GUI_ACTION_RE = re.compile(
    r"(?:按下?|点击|选择|打开|关闭|进入|退出|保存|安装|连接|取出|放入|检查|确认|"
    r"设置为|切换|返回|启动|重新启动)",
    re.IGNORECASE,
)
_SCOPED_TRUNCATION_RE = re.compile(r"(?:\.\.\.|…{1,}|\[\s*truncated\s*\])\s*$", re.IGNORECASE)
_SCOPED_TABLE_ANAPHOR_CELL_KEYS = frozenset(
    _normalize_anchor_text(value)
    for value in (
        "同上",
        "续",
        "续上",
        "同前",
        "同前项",
        "同左",
        "见上",
        "如上",
        "ditto",
        "same as above",
    )
)
_SCOPED_TABLE_ANAPHOR_TEXT_RE = re.compile(
    r"(?:同上|同前项?|同左|见上|如上|上述|前述)"
    r"|(?:该|此|以上)(?:型号|对象|设备|产品|项目|条目|记录|功能)"
    r"|其(?:还|也|另|同时|支持|功能|配置|状态|参数)"
    r"|(?:^|\n)\s*续(?:表|页|项|[:：\s]|$)",
    re.IGNORECASE,
)
_SCOPED_TABLE_LOCAL_SUPPLEMENT_RE = re.compile(
    r"(?:补充|另(?:有|外)?|还|也|同时|此外|续(?:表|页|项)?)",
    re.IGNORECASE,
)
_BULLET_LIST_LINE_RE = re.compile(r"^(?P<marker>[-*•+])(?P<gap>\s+)(?P<body>\S(?:.*?\S)?)\s*$")
_RAW_BULLET_MARKER_RE = re.compile(r"^\s*[-*•+](?:\s|$)")
_SCOPED_BULLET_CLOSED_INTRO_RE = re.compile(
    r"(?:以下|如下|下列)(?:的)?(?:情况|内容|项目|事项|方面|检查点|条件|原因|情形|要点|各项)?"
    r"|(?:the\s+following|as\s+follows)\b",
    re.IGNORECASE,
)
_SCOPED_BULLET_BOUNDED_TITLE_RE = re.compile(
    r"(?:完整|全部|所有)?(?:清单|列表|检查点|检查项目|项目表|事项表)\s*[:：]?\s*$"
    r"|\b(?:complete\s+)?(?:checklist|list)\s*[:：]?\s*$",
    re.IGNORECASE,
)
_SCOPED_BULLET_QUERY_ENUMERATION_RE = re.compile(
    r"(?:哪些|哪几|所有|全部|完整|列出|清单|列表|检查点|各项)"
    r"|\b(?:which|what|all|every|complete|list)\b",
    re.IGNORECASE,
)
_SCOPED_BULLET_ANAPHOR_RE = re.compile(
    r"(?:同上|续上?|上述|前述|如上|见上|其)",
    re.IGNORECASE,
)
_SCOPED_BULLET_SUPPLEMENT_RE = re.compile(
    r"(?:补充|此外|另外|除此之外|同时还|还应|也应|还需|也需|还包括)"
    r"[^。；;\n]{0,32}(?:检查|确认|包括|考虑|处理|项目|事项|情况)",
    re.IGNORECASE,
)
_SCOPED_BULLET_GENERIC_BIGRAMS = frozenset(
    {
        "以下",
        "如下",
        "下列",
        "检查",
        "项目",
        "内容",
        "情况",
        "事项",
        "方面",
        "哪些",
        "哪几",
        "所有",
        "全部",
        "列出",
        "相关",
        "关键",
        "请列",
        "请检",
        "应检",
        "需要",
        "进行",
        "首先",
        "出现",
        "问题",
    }
)


def _bullet_list_line(line: _PhysicalLine) -> _BulletListLine | None:
    """只解析顶格、含非空正文的单行项目符号项。"""

    matched = _BULLET_LIST_LINE_RE.fullmatch(line.text)
    if matched is None or line.text[:1].isspace():
        return None
    body = matched.group("body")
    return _BulletListLine(
        line=line,
        content_start=line.start,
        content_end=line.end,
        body_start=line.start + matched.start("body"),
        body_end=line.start + matched.end("body"),
        marker_style=f"bullet:{matched.group('marker')}",
        body=body,
    )


def _is_explicit_title_line(line: _PhysicalLine) -> bool:
    stripped = line.text.strip()
    return bool(_MARKDOWN_HEADING_RE.match(stripped) or re.match(r"^标题路径\s*[:：]", stripped))


def _bullet_anchor_kind(
    anchor: _PhysicalLine,
) -> (
    Literal[
        "title_closed_intro",
        "ordered_closed_intro",
        "prose_closed_intro",
        "bounded_title",
    ]
    | None
):
    text = anchor.text.strip()
    if _SCOPED_BULLET_CLOSED_INTRO_RE.search(text):
        if _is_explicit_title_line(anchor):
            return "title_closed_intro"
        if _ordered_list_line(anchor) is not None:
            return "ordered_closed_intro"
        return "prose_closed_intro"
    if _is_explicit_title_line(anchor) and _SCOPED_BULLET_BOUNDED_TITLE_RE.search(text):
        return "bounded_title"
    return None


def _bullet_anchor_match_keys(source_text: str, query_text: str) -> tuple[str, ...]:
    """用非通用 CJK bigram / ASCII token 证明 Query 与局部列表主题相交。"""

    normalized_source = _normalize_anchor_text(source_text)
    normalized_query = _normalize_anchor_text(query_text)
    source_keys: set[str] = set()
    query_keys: set[str] = set()
    for normalized, target in (
        (normalized_source, source_keys),
        (normalized_query, query_keys),
    ):
        for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
            target.update(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
                if sequence[index : index + 2] not in _SCOPED_BULLET_GENERIC_BIGRAMS
            )
        target.update(
            f"token:{token}"
            for token in re.findall(r"[a-z0-9]{3,}", normalized)
            if not token.isdecimal()
        )
    return tuple(sorted(source_keys & query_keys))


def _bullet_title_before(
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    anchor_index: int,
) -> tuple[_PhysicalLine, tuple[int, int, str]] | None:
    for cursor in range(anchor_index, -1, -1):
        line, trimmed = nonempty[cursor]
        if _is_explicit_title_line(line):
            return line, trimmed
    return None


def _bullet_local_section_end(
    *,
    text: str,
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    after_item_index: int,
) -> int:
    for line, _ in nonempty[after_item_index:]:
        if _is_explicit_title_line(line) or _ordered_list_line(line) is not None:
            return line.start
    return len(text)


def _locate_all_bullet_list_blocks(text: str) -> tuple[_BulletListBlock, ...]:
    """扫描所有“显式引导行 + 连续顶层 bullet”结构，不在此步做 Query 选择。"""

    if PACKED_TRUNCATION_MARKER in text:
        raise ValueError("scoped bullet-list authority is truncated")
    nonempty = tuple(
        (line, trimmed)
        for line in _physical_lines(text)
        if (trimmed := _trimmed_physical_line(line)) is not None
    )
    blocks: list[_BulletListBlock] = []
    cursor = 0
    while cursor < len(nonempty):
        first_item = _bullet_list_line(nonempty[cursor][0])
        if first_item is None:
            cursor += 1
            continue
        end = cursor + 1
        items = [first_item]
        while end < len(nonempty):
            item = _bullet_list_line(nonempty[end][0])
            if item is None:
                break
            items.append(item)
            end += 1
        if cursor == 0:
            cursor = end
            continue
        anchor, anchor_trimmed = nonempty[cursor - 1]
        anchor_kind = _bullet_anchor_kind(anchor)
        if anchor_kind is None or _parse_simple_markdown_row(anchor) is not None:
            cursor = end
            continue
        title_pair = _bullet_title_before(nonempty, cursor - 1)
        title = title_pair[0] if title_pair is not None else None
        title_trimmed = title_pair[1] if title_pair is not None else None
        local_section_start = title_trimmed[0] if title_trimmed is not None else anchor_trimmed[0]
        blocks.append(
            _BulletListBlock(
                start=anchor_trimmed[0],
                end=items[-1].content_end,
                anchor=anchor,
                anchor_content_start=anchor_trimmed[0],
                anchor_content_end=anchor_trimmed[1],
                normalized_anchor=_normalize_table_cell(anchor_trimmed[2]),
                anchor_kind=anchor_kind,
                title=title,
                title_content_start=(title_trimmed[0] if title_trimmed is not None else None),
                title_content_end=(title_trimmed[1] if title_trimmed is not None else None),
                normalized_title=(
                    _normalize_table_cell(title_trimmed[2]) if title_trimmed is not None else ""
                ),
                items=tuple(items),
                marker_style=first_item.marker_style,
                local_section_start=local_section_start,
                local_section_end=_bullet_local_section_end(
                    text=text,
                    nonempty=nonempty,
                    after_item_index=end,
                ),
            )
        )
        cursor = end
    return tuple(blocks)


def _bullet_list_structure_payload(block: _BulletListBlock) -> dict[str, Any]:
    return {
        "anchor_kind": block.anchor_kind,
        "anchor": block.normalized_anchor,
        "title": block.normalized_title,
        "marker_style": block.marker_style,
        "items": [_normalize_table_cell(item.body) for item in block.items],
    }


def _validate_scoped_bullet_block(*, text: str, block: _BulletListBlock) -> None:
    if len(block.items) < 2:
        raise ValueError("scoped bullet-list requires at least two items")
    marker_styles = {item.marker_style for item in block.items}
    if len(marker_styles) != 1:
        raise ValueError("scoped bullet-list items must use one stable marker style")
    normalized_bodies = tuple(_normalize_table_cell(item.body) for item in block.items)
    if any(not body for body in normalized_bodies) or len(normalized_bodies) != len(
        set(normalized_bodies)
    ):
        raise ValueError("scoped bullet-list items must be non-empty and distinct")
    scope_text = text[block.start : block.end]
    if _SCOPED_OPEN_ENUMERATION_RE.search(scope_text):
        raise ValueError("scoped bullet-list contains an open-ended enumeration cue")
    if _SCOPED_OPEN_REFERENCE_RE.search(scope_text):
        raise ValueError("scoped bullet-list contains an open cross-reference")
    if _SCOPED_TRUNCATION_RE.search(scope_text.rstrip()):
        raise ValueError("scoped bullet-list scope appears truncated")
    if _SCOPED_BULLET_ANAPHOR_RE.search(scope_text):
        raise ValueError("scoped bullet-list contains an ambiguous continuation")

    local_lines = tuple(
        line
        for line in _physical_lines(text)
        if line.start >= block.local_section_start and line.end <= block.local_section_end
    )
    member_line_indices = {item.line.index for item in block.items}
    raw_bullet_lines = tuple(line for line in local_lines if _RAW_BULLET_MARKER_RE.match(line.text))
    if any(_bullet_list_line(line) is None for line in raw_bullet_lines):
        raise ValueError("scoped bullet-list contains an empty or nested item")
    if {line.index for line in raw_bullet_lines} != member_line_indices:
        raise ValueError("scoped bullet-list has another item outside the selected scope")
    trailing_text = text[block.end : block.local_section_end]
    if any(not character.isspace() for character in trailing_text):
        raise ValueError("scoped bullet-list has an unowned continuation after the final item")


def _scoped_bullet_candidates_for_anchor(
    *,
    block_pairs: Sequence[tuple[int, int, _BulletListBlock, _BulletListBlock, str, str]],
    anchor: str,
    anchor_field: Literal["source_query", "information_need"],
) -> tuple[_ScopedBulletListCandidate, ...]:
    if not _SCOPED_BULLET_QUERY_ENUMERATION_RE.search(anchor):
        return ()
    candidates: list[_ScopedBulletListCandidate] = []
    for authority_index, block_index, canonical, packed, canonical_text, packed_text in block_pairs:
        source_anchor = "\n".join(
            value for value in (canonical.normalized_title, canonical.normalized_anchor) if value
        )
        match_keys = _bullet_anchor_match_keys(source_anchor, anchor)
        if len(match_keys) < 2:
            continue
        _validate_scoped_bullet_block(text=canonical_text, block=canonical)
        _validate_scoped_bullet_block(text=packed_text, block=packed)
        candidates.append(
            _ScopedBulletListCandidate(
                authority_index=authority_index,
                block_index=block_index,
                canonical_block=canonical,
                packed_block=packed,
                anchor_field=anchor_field,
                anchor_match_keys=match_keys,
            )
        )
    return tuple(candidates)


def _reject_catalog_bullet_supplements(
    *,
    authorities: Sequence[ScopedBulletAuthorityRepresentation],
    candidate: _ScopedBulletListCandidate,
    anchor: str,
) -> None:
    """拒绝其他位置中与所选主题相交的显式补充语句。"""

    for authority_index, authority in enumerate(authorities):
        for projection, text in (
            ("canonical", authority.canonical_text),
            ("packed", authority.packed_text),
        ):
            selected_block = (
                candidate.canonical_block if projection == "canonical" else candidate.packed_block
            )
            physical_lines = _physical_lines(text)
            for line_index, line in enumerate(physical_lines):
                if (
                    authority_index == candidate.authority_index
                    and line.start >= selected_block.start
                    and line.end <= selected_block.end
                ):
                    continue
                if not _SCOPED_BULLET_SUPPLEMENT_RE.search(line.text):
                    continue
                preceding = tuple(
                    item
                    for item in physical_lines[: line_index + 1]
                    if _trimmed_physical_line(item) is not None
                )
                title = next(
                    (item.text for item in reversed(preceding) if _is_explicit_title_line(item)),
                    "",
                )
                if len(_bullet_anchor_match_keys(f"{title}\n{line.text}", anchor)) >= 2:
                    raise ValueError(
                        "scoped bullet-list authority contains a same-topic supplement"
                    )


def _scoped_anchor_identifiers(value: str) -> tuple[str, ...]:
    """从标题路径保守提取可由 Query 明确绑定的完整标题和实体核心。"""

    title = _MARKDOWN_HEADING_RE.sub("", value.strip())
    if ">" in title or "＞" in title:
        title = re.split(r"[>＞]", title)[-1].strip()
    title = re.sub(r"^标题路径\s*[:：]?\s*", "", title).strip()
    normalized = _normalized_procedure_title(title)
    if not _is_specific_identifier(normalized):
        return ()
    candidates = [normalized]
    for suffix in _SCOPED_TITLE_GENERIC_SUFFIXES:
        if normalized.endswith(suffix):
            core = normalized[: -len(suffix)]
            if _is_specific_identifier(core):
                candidates.append(core)
            break
    return tuple(dict.fromkeys(candidates))


def _procedure_anchor_match_keys(source_text: str, query_text: str) -> tuple[str, ...]:
    """提取流程主题与 Query 共享的非通用 CJK bigram / ASCII token。"""

    normalized_source = _normalize_anchor_text(source_text)
    normalized_query = _normalize_anchor_text(query_text)
    source_keys: set[str] = set()
    query_keys: set[str] = set()
    for normalized, target in (
        (normalized_source, source_keys),
        (normalized_query, query_keys),
    ):
        for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
            target.update(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
                if sequence[index : index + 2] not in _SCOPED_PROCEDURE_GENERIC_BIGRAMS
            )
        target.update(
            f"token:{token}"
            for token in re.findall(r"[a-z0-9]{3,}", normalized)
            if not token.isdecimal()
        )
    return tuple(sorted(source_keys & query_keys))


def _closed_procedure_action_body(body: str) -> str:
    """机械删除纯括号参阅；正文仍须形成独立、非开放的动作。"""

    removed_reference = False

    def _remove_reference(match: re.Match[str]) -> str:
        nonlocal removed_reference
        if _SCOPED_OPEN_REFERENCE_RE.search(match.group("body")) is None:
            return match.group(0)
        removed_reference = True
        return ""

    closed_body = _SCOPED_PARENTHETICAL_REFERENCE_RE.sub(_remove_reference, body).strip()
    if _SCOPED_OPEN_REFERENCE_RE.search(closed_body):
        raise ValueError("scoped procedure list contains an open cross-reference")
    normalized = _normalize_table_cell(closed_body.strip(" \t。；;，,：:"))
    if len(normalized) < 4 or not any(character.isalnum() for character in normalized):
        if removed_reference:
            raise ValueError("scoped procedure parenthetical reference has no complete main action")
        raise ValueError("scoped procedure list item has no complete main action")
    return closed_body


def _paired_scoped_procedure_media_line_indices(
    *,
    canonical_text: str,
    packed_text: str,
) -> tuple[frozenset[int], frozenset[int]]:
    """仅把可由冻结安全派生逐一重放的独立图片与占位符视为同一媒体。"""

    packed_lines = tuple(
        (line, trimmed)
        for line in _physical_lines(packed_text)
        if (trimmed := _trimmed_physical_line(line)) is not None
        and trimmed[2] == MARKDOWN_IMAGE_CONTROL_TEXT
    )
    if not packed_lines:
        return frozenset(), frozenset()

    derivation = derive_safe_representation(canonical_text)
    if derivation.receipt.has_unproven_transform or derivation.safe_representation != packed_text:
        raise ValueError("scoped procedure image placeholders require a replayed safe derivation")
    control_spans = tuple(
        span
        for span in derivation.receipt.spans
        if span.relation == SafeSpanRelation.CONTROL_MARKDOWN_IMAGE
    )
    if len(control_spans) != len(packed_lines):
        raise ValueError("scoped procedure images and packed placeholders must pair one-to-one")

    canonical_lines_by_span = {
        (trimmed[0], trimmed[1]): line
        for line in _physical_lines(canonical_text)
        if (trimmed := _trimmed_physical_line(line)) is not None
    }
    canonical_indices: list[int] = []
    packed_indices: list[int] = []
    for span, (packed_line, packed_trimmed) in zip(
        control_spans,
        packed_lines,
        strict=True,
    ):
        canonical_line = canonical_lines_by_span.get((span.canonical_start, span.canonical_end))
        if (
            canonical_line is None
            or not is_markdown_image_only_line(canonical_line.text.strip())
            or (span.safe_start, span.safe_end) != (packed_trimmed[0], packed_trimmed[1])
        ):
            raise ValueError("scoped procedure media pair must preserve physical line boundaries")
        canonical_indices.append(canonical_line.index)
        packed_indices.append(packed_line.index)
    if canonical_indices != sorted(set(canonical_indices)) or packed_indices != sorted(
        set(packed_indices)
    ):
        raise ValueError("scoped procedure media pairs must preserve physical source order")
    return frozenset(canonical_indices), frozenset(packed_indices)


def _image_metadata_positions(
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    *,
    paired_media_line_indices: frozenset[int] = frozenset(),
) -> frozenset[int]:
    """标记显式图片说明区，避免其中的编号图例成为伪流程。"""

    positions: set[int] = set()
    image_pending = False
    metadata_active = False
    for position, (line, trimmed) in enumerate(nonempty):
        if _is_explicit_title_line(line):
            image_pending = False
            metadata_active = False
            continue
        if metadata_active:
            positions.add(position)
            continue
        if is_markdown_image_only_line(trimmed[2]) or line.index in paired_media_line_indices:
            image_pending = True
            continue
        if image_pending and is_image_derivation_metadata_start(trimmed[2]):
            metadata_active = True
            positions.add(position)
            continue
        if _ordered_list_line(line) is not None:
            image_pending = False
    return frozenset(positions)


def _declared_sublist_count(value: str) -> int | None:
    matched = _SCOPED_PROCEDURE_FINITE_SUBLIST_RE.search(value)
    if matched is None:
        return None
    raw = matched.group("count")
    if raw.isdecimal():
        return int(raw)
    if raw == "两":
        return 2
    return _chinese_ordinal(raw)


def _finite_subordinate_positions(
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    *,
    metadata_positions: frozenset[int],
) -> frozenset[int]:
    """只隔离由 bullet 精确声明数量、且 marker 独立的有限说明子表。"""

    result: set[int] = set()
    for position, (line, _) in enumerate(nonempty):
        if position == 0 or position in metadata_positions:
            continue
        first_item = _ordered_list_line(line)
        if first_item is None or first_item.ordinal != 1:
            continue
        intro = nonempty[position - 1][1][2]
        if _RAW_BULLET_MARKER_RE.match(intro) is None:
            continue
        declared_count = _declared_sublist_count(intro)
        if declared_count is None:
            continue
        members: list[int] = []
        valid = True
        for offset in range(declared_count):
            member_position = position + offset
            if member_position >= len(nonempty) or member_position in metadata_positions:
                valid = False
                break
            member = _ordered_list_line(nonempty[member_position][0])
            if (
                member is None
                or member.marker_style != first_item.marker_style
                or member.ordinal != offset + 1
            ):
                valid = False
                break
            try:
                _closed_procedure_action_body(member.body)
            except ValueError:
                valid = False
                break
            members.append(member_position)
        if not valid or len(members) != declared_count:
            continue
        result.update(members)
    return frozenset(result)


def _owned_optional_note_gap_kinds(
    gap: Sequence[tuple[int, tuple[_PhysicalLine, tuple[int, int, str]]]],
    *,
    preceding_items: Sequence[_OrderedListLine],
    next_item: _OrderedListLine,
) -> tuple[str, ...] | None:
    """识别不改变主步骤全集、且主题归属于相邻步骤的显式可选注释块。"""

    if len(gap) < 2 or len(gap) > 4:
        return None
    values = tuple(pair[1][2] for _, pair in gap)
    if _SCOPED_PROCEDURE_NOTE_RE.match(values[0]) is None or any(
        _RAW_BULLET_MARKER_RE.match(value) is None for value in values[1:]
    ):
        return None
    text = "\n".join(values)
    if (
        len(text) > 1_200
        or _SCOPED_TRUNCATION_RE.search(text.rstrip())
        or _SCOPED_OPEN_ENUMERATION_RE.search(text)
        or _SCOPED_PROCEDURE_REQUIRED_NOTE_ACTION_RE.search(text)
        or _SCOPED_PROCEDURE_OPTIONAL_NOTE_RE.search(text) is None
    ):
        return None
    cleaned: list[str] = []
    for index, value in enumerate(values):
        body = value
        if index == 0:
            body = _SCOPED_PROCEDURE_NOTE_RE.sub("", body, count=1)
        body = _RAW_BULLET_MARKER_RE.sub("", body, count=1).strip()
        if not body:
            return None
        try:
            cleaned.append(_closed_procedure_action_body(body))
        except ValueError:
            return None
    # 注释可能说明当前流程较早一步建立的可选状态（例如热点默认关闭），
    # 因而所有已连续证明的前序步骤都属于有限 owner scope；仍要求至少两个
    # 非通用主题键，避免仅凭“设置”“步骤”等泛词吞并游离备注。
    owner_text = "\n".join((*tuple(item.body for item in preceding_items), next_item.body))
    if len(_procedure_anchor_match_keys("\n".join(cleaned), owner_text)) < 2:
        return None
    return ("owned_optional_note",) + tuple("owned_optional_note_item" for _ in values[1:])


def _quoted_procedure_targets(body: str) -> tuple[str, ...]:
    targets = tuple(
        normalized
        for match in re.finditer(r"[\"“](?P<value>[^\"”\r\n]{1,80})[\"”]", body)
        if len(normalized := _normalize_anchor_text(match.group("value"))) >= 3
        and normalized not in {"下一步", "上一步", "完成", "返回"}
    )
    return tuple(dict.fromkeys(targets))


def _owned_gui_state_echo_gap_kinds(
    gap: Sequence[tuple[int, tuple[_PhysicalLine, tuple[int, int, str]]]],
    *,
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    previous_item: _OrderedListLine,
    paired_media_line_indices: frozenset[int],
    used_media_positions: set[int],
) -> tuple[str, ...] | None:
    """仅跨越可与相邻安全派生图片一对一绑定的前一步 GUI 状态回显。"""

    media_positions = {
        position
        for position, (line, _) in enumerate(nonempty)
        if line.index in paired_media_line_indices
    }
    if not media_positions:
        return None
    echo_entries = tuple(
        (position, pair) for position, pair in gap if position not in media_positions
    )
    if len(echo_entries) != 1:
        return None
    echo_position, echo_pair = echo_entries[0]
    echo = echo_pair[1][2]
    normalized_echo = _normalize_anchor_text(echo)
    targets = _quoted_procedure_targets(previous_item.body)
    if (
        len(normalized_echo) < 4
        or len(normalized_echo) > 80
        or _SCOPED_PROCEDURE_GUI_ACTION_RE.search(echo)
        or _SCOPED_OPEN_REFERENCE_RE.search(echo)
        or _SCOPED_OPEN_ENUMERATION_RE.search(echo)
        or _SCOPED_TRUNCATION_RE.search(echo.rstrip())
        or not any(target in normalized_echo for target in targets)
    ):
        return None
    candidate_media_positions: list[int] = []
    for media_position in media_positions - used_media_positions:
        if abs(media_position - echo_position) > 2:
            continue
        between = range(
            min(media_position, echo_position) + 1,
            max(media_position, echo_position),
        )
        if any(_ordered_list_line(nonempty[position][0]) is None for position in between):
            continue
        candidate_media_positions.append(media_position)
    if not candidate_media_positions:
        return None
    candidate_media_positions.sort(
        key=lambda position: (
            abs(position - echo_position),
            position < echo_position,
            position,
        )
    )
    chosen_media_position = candidate_media_positions[0]
    used_media_positions.add(chosen_media_position)
    return tuple(
        "media" if position in media_positions else "gui_state_echo" for position, _ in gap
    )


def _procedure_gap_kinds(
    gap: Sequence[tuple[int, tuple[_PhysicalLine, tuple[int, int, str]]]],
    *,
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    preceding_items: Sequence[_OrderedListLine],
    previous_item: _OrderedListLine,
    next_item: _OrderedListLine,
    subordinate_positions: frozenset[int],
    metadata_positions: frozenset[int],
    paired_media_line_indices: frozenset[int],
    used_gui_media_positions: set[int],
) -> tuple[str, ...] | None:
    if not gap:
        return ()
    if len(gap) > _MAX_SCOPED_PROCEDURE_GAP_LINES:
        return None
    text = "\n".join(pair[1][2] for _, pair in gap)
    if len(text) > _MAX_SCOPED_PROCEDURE_GAP_CHARACTERS:
        return None
    if (
        _SCOPED_OPEN_ENUMERATION_RE.search(text)
        or _SCOPED_OPEN_REFERENCE_RE.search(text)
        or _SCOPED_TRUNCATION_RE.search(text.rstrip())
    ):
        optional_note = _owned_optional_note_gap_kinds(
            gap,
            preceding_items=preceding_items,
            next_item=next_item,
        )
        if optional_note is None:
            return None
        return optional_note
    if any(
        position in metadata_positions
        or _is_explicit_title_line(pair[0])
        or _parse_simple_markdown_row(pair[0]) is not None
        for position, pair in gap
    ):
        return None
    ordered_positions = {
        position for position, pair in gap if _ordered_list_line(pair[0]) is not None
    }
    if ordered_positions - subordinate_positions:
        return None
    optional_note = _owned_optional_note_gap_kinds(
        gap,
        preceding_items=preceding_items,
        next_item=next_item,
    )
    if optional_note is not None:
        return optional_note
    gui_echo = _owned_gui_state_echo_gap_kinds(
        gap,
        nonempty=nonempty,
        previous_item=previous_item,
        paired_media_line_indices=paired_media_line_indices,
        used_media_positions=used_gui_media_positions,
    )
    if gui_echo is not None:
        return gui_echo
    has_subordinate = bool(ordered_positions)
    has_media = any(
        is_markdown_image_only_line(pair[1][2]) or pair[0].index in paired_media_line_indices
        for _, pair in gap
    )
    if not has_media and not has_subordinate:
        if len(gap) > 2 or any(
            _SCOPED_PROCEDURE_NOTE_RE.match(pair[1][2]) is None for _, pair in gap
        ):
            return None
    if not has_subordinate and any(
        _RAW_BULLET_MARKER_RE.match(pair[1][2]) is not None for _, pair in gap
    ):
        return None
    kinds: list[str] = []
    for position, pair in gap:
        value = pair[1][2]
        if position in subordinate_positions:
            kinds.append("finite_subordinate_item")
        elif is_markdown_image_only_line(value) or pair[0].index in paired_media_line_indices:
            kinds.append("media")
        elif _SCOPED_PROCEDURE_NOTE_RE.match(value):
            kinds.append("owned_note")
        elif _RAW_BULLET_MARKER_RE.match(value):
            kinds.append("finite_subordinate_context")
        else:
            kinds.append("media_prose")
    return tuple(kinds)


def _is_explicit_procedure_branch_boundary(
    gap: Sequence[tuple[int, tuple[_PhysicalLine, tuple[int, int, str]]]],
) -> bool:
    if any(_is_explicit_title_line(pair[0]) for _, pair in gap):
        return True
    return any(
        _SCOPED_PROCEDURE_BRANCH_INTRO_RE.search(pair[1][2])
        or _SCOPED_PROCEDURE_CLOSED_INTRO_RE.search(pair[1][2])
        for _, pair in gap[-3:]
    )


def _scoped_procedure_sequence_from(
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    *,
    start_position: int,
    subordinate_positions: frozenset[int],
    metadata_positions: frozenset[int],
    paired_media_line_indices: frozenset[int],
) -> _ScopedProcedureSequence:
    first = _ordered_list_line(nonempty[start_position][0])
    if first is None:
        raise ValueError("scoped procedure sequence must start at an ordered item")
    items = [first]
    gap_kinds: list[tuple[str, ...]] = []
    used_gui_media_positions: set[int] = set()
    current_position = start_position
    while True:
        next_position = current_position + 1
        while next_position < len(nonempty):
            if next_position in metadata_positions or next_position in subordinate_positions:
                next_position += 1
                continue
            if _ordered_list_line(nonempty[next_position][0]) is not None:
                break
            next_position += 1
        if next_position >= len(nonempty):
            return _ScopedProcedureSequence(
                items=tuple(items),
                gap_kinds=tuple(gap_kinds),
                last_position=current_position,
                blocked_reason=None,
            )
        next_item = _ordered_list_line(nonempty[next_position][0])
        if next_item is None:
            raise AssertionError("ordered-item search resolved a non-item")
        gap = tuple(
            (position, nonempty[position])
            for position in range(current_position + 1, next_position)
        )
        if (
            next_item.marker_style == first.marker_style
            and next_item.ordinal == items[-1].ordinal + 1
        ):
            validated_gap = _procedure_gap_kinds(
                gap,
                nonempty=nonempty,
                preceding_items=items,
                previous_item=items[-1],
                next_item=next_item,
                subordinate_positions=subordinate_positions,
                metadata_positions=metadata_positions,
                paired_media_line_indices=paired_media_line_indices,
                used_gui_media_positions=used_gui_media_positions,
            )
            if validated_gap is None:
                return _ScopedProcedureSequence(
                    items=tuple(items),
                    gap_kinds=tuple(gap_kinds),
                    last_position=current_position,
                    blocked_reason="scoped procedure list contains an unbound continuation",
                )
            gap_kinds.append(validated_gap)
            items.append(next_item)
            current_position = next_position
            continue
        if _is_explicit_procedure_branch_boundary(gap):
            return _ScopedProcedureSequence(
                items=tuple(items),
                gap_kinds=tuple(gap_kinds),
                last_position=current_position,
                blocked_reason=None,
            )
        return _ScopedProcedureSequence(
            items=tuple(items),
            gap_kinds=tuple(gap_kinds),
            last_position=current_position,
            blocked_reason="scoped procedure list contains an ambiguous ordinal reset or branch",
        )


def _nearest_explicit_procedure_title(
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    *,
    before_position: int,
) -> tuple[_PhysicalLine, tuple[int, int, str]] | None:
    for position in range(before_position - 1, -1, -1):
        line, trimmed = nonempty[position]
        if _is_explicit_title_line(line):
            return line, trimmed
    return None


def _query_selects_procedure_continuation(query: str, item_body: str) -> bool:
    keys = _procedure_anchor_match_keys(item_body, query)
    if _SCOPED_PROCEDURE_TEMPORAL_QUERY_RE.search(query):
        return len(keys) >= 2
    return bool(
        _SCOPED_PROCEDURE_CONDITIONAL_QUERY_RE.search(query)
        and _SCOPED_PROCEDURE_CONDITIONAL_QUERY_RE.search(item_body)
        and len(keys) >= 3
    )


def _query_before_triggers(query: str) -> tuple[str, ...]:
    """只提取带明确“前”边界且非通用动作名的查询触发语。"""

    generic = {
        _normalize_anchor_text(value)
        for value in (
            "操作",
            "使用",
            "处理",
            "开始",
            "完成",
            "打印",
            "安装",
            "连接",
            "设置",
            "配置",
        )
    }
    triggers = tuple(
        normalized
        for match in _SCOPED_PROCEDURE_BEFORE_QUERY_RE.finditer(query)
        if len(normalized := _normalize_anchor_text(match.group("trigger"))) >= 4
        and normalized not in generic
    )
    return tuple(dict.fromkeys(triggers))


def _select_query_bounded_procedure_prefix(
    *,
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    start_position: int,
    sequence: _ScopedProcedureSequence,
    question: QuestionContract,
    obligation: QuestionObligation,
) -> _ScopedProcedurePrefixSelection | None:
    """选择由 Query 的“X前”与首个显式 X 执行项共同闭合的严格前缀。"""

    if (
        sequence.blocked_reason is not None
        or len(sequence.items) < 3
        or sequence.items[0].ordinal != 1
    ):
        return None
    title_pair = _nearest_explicit_procedure_title(
        nonempty,
        before_position=start_position,
    )
    if title_pair is None:
        return None
    choices: list[tuple[int, bool, _ScopedProcedurePrefixSelection]] = []
    for anchor_field, query in (
        ("source_query", question.source_query),
        ("information_need", obligation.information_need),
    ):
        triggers = _query_before_triggers(query)
        if len(triggers) != 1:
            continue
        trigger = triggers[0]
        successor_positions = tuple(
            index
            for index, item in enumerate(sequence.items)
            if index >= 2
            and trigger in _normalize_anchor_text(_closed_procedure_action_body(item.body))
            and _SCOPED_PROCEDURE_PHASE_START_RE.search(item.body)
        )
        if len(successor_positions) != 1:
            continue
        successor_position = successor_positions[0]
        selected_items = sequence.items[:successor_position]
        if len(selected_items) < 2:
            continue
        topic_text = "\n".join((title_pair[1][2], *(item.body for item in selected_items)))
        match_keys = _procedure_anchor_match_keys(topic_text, query)
        direct_identifiers = tuple(
            identifier
            for identifier in _scoped_anchor_identifiers(title_pair[1][2])
            if _anchor_contains_identifier(identifier, query)
        )
        if len(match_keys) < 2 and not direct_identifiers:
            continue
        if direct_identifiers:
            match_keys = tuple(
                dict.fromkeys((*match_keys, *(f"identifier:{item}" for item in direct_identifiers)))
            )
        match_keys = tuple(dict.fromkeys((*match_keys, f"prefix:{trigger}")))
        identifier_candidates = _scoped_anchor_identifiers(title_pair[1][2])
        anchor_identifier = (
            direct_identifiers[0]
            if direct_identifiers
            else identifier_candidates[0]
            if identifier_candidates
            else _normalize_table_cell(title_pair[1][2])
        )
        if not _is_specific_identifier(anchor_identifier):
            continue
        successor = sequence.items[successor_position]
        selection = _ScopedProcedurePrefixSelection(
            items=selected_items,
            gap_kinds=sequence.gap_kinds[: max(0, len(selected_items) - 1)],
            anchor_field=anchor_field,
            anchor_identifier=anchor_identifier,
            anchor=title_pair[0],
            anchor_trimmed=title_pair[1],
            anchor_match_keys=match_keys,
            prefix_trigger_hash=sha256_digest(f"{trigger}前"),
            excluded_successor_ordinal=successor.ordinal,
            excluded_successor_hash=sha256_digest(
                _normalize_table_cell(_closed_procedure_action_body(successor.body))
            ),
            boundary_rule_version="explicit-query-before-successor.v1",
        )
        choices.append(
            (
                len(match_keys) + (100 if direct_identifiers else 0),
                anchor_field == "source_query",
                selection,
            )
        )
    if not choices:
        return None
    choices.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(choices) > 1 and choices[0][:2] == choices[1][:2]:
        return None
    return choices[0][2]


def _first_item_matches_obligation_local_trigger(
    *,
    question: QuestionContract,
    obligation: QuestionObligation,
    first_item: _OrderedListLine,
) -> bool:
    """多义务问题中，禁止用 source_query 的较早 facet 补足局部续问。"""

    if len(question.obligations) <= 1:
        return True
    need = obligation.information_need
    body = _normalize_anchor_text(first_item.body)
    temporal = re.search(
        r"(?P<trigger>(?:输入|取出|更换|完成|连接|安装|设置|配置|关闭|打开|保存|检查)"
        r"[^，,。；;！？?]{0,24}(?:后|之后|以后))",
        need,
    )
    if temporal is not None:
        return _normalize_anchor_text(temporal.group("trigger")) in body
    conditional = re.search(
        r"(?P<trigger>(?:如果|若|当)[^，,。；;！？?]{2,48})",
        need,
    )
    if conditional is not None:
        trigger = _normalize_anchor_text(conditional.group("trigger"))
        return trigger in body
    return True


def _select_scoped_procedure_anchor(
    *,
    nonempty: Sequence[tuple[_PhysicalLine, tuple[int, int, str]]],
    start_position: int,
    first_item: _OrderedListLine,
    question: QuestionContract,
    obligation: QuestionObligation,
    paired_media_line_indices: frozenset[int],
) -> (
    tuple[
        Literal["source_query", "information_need"],
        str,
        _PhysicalLine,
        tuple[int, int, str],
        tuple[str, ...],
        Literal[
            "complete_sequence",
            "query_selected_continuation",
            "query_selected_branch",
            "query_selected_prefix",
        ],
        int,
    ]
    | None
):
    title_pair = _nearest_explicit_procedure_title(
        nonempty,
        before_position=start_position,
    )
    local_pair = nonempty[start_position - 1] if start_position > 0 else None
    if local_pair is not None and (
        is_markdown_image_only_line(local_pair[1][2])
        or local_pair[0].index in paired_media_line_indices
        or is_image_derivation_metadata_start(local_pair[1][2])
        or _parse_simple_markdown_row(local_pair[0]) is not None
    ):
        local_pair = None
    previous_item = _ordered_list_line(local_pair[0]) if local_pair is not None else None
    local_is_branch = bool(
        local_pair is not None
        and (
            _SCOPED_PROCEDURE_BRANCH_INTRO_RE.search(local_pair[1][2])
            or previous_item is not None
            and previous_item.marker_style != first_item.marker_style
            and local_pair[1][2].rstrip().endswith((":", "："))
        )
    )
    anchors: tuple[tuple[Literal["source_query", "information_need"], str], ...] = (
        ("source_query", question.source_query),
        ("information_need", obligation.information_need),
    )
    choices: list[
        tuple[
            int,
            int,
            Literal["source_query", "information_need"],
            str,
            _PhysicalLine,
            tuple[int, int, str],
            tuple[str, ...],
            Literal[
                "complete_sequence",
                "query_selected_continuation",
                "query_selected_branch",
                "query_selected_prefix",
            ],
        ]
    ] = []
    for anchor_field, query in anchors:
        title_text = title_pair[1][2] if title_pair is not None else ""
        local_text = local_pair[1][2] if local_pair is not None else ""
        topic_text = "\n".join(value for value in (title_text, local_text) if value)
        topic_keys = _procedure_anchor_match_keys(topic_text, query)
        direct_identifiers = tuple(
            identifier
            for source in (title_text, local_text)
            for identifier in _scoped_anchor_identifiers(source)
            if _anchor_contains_identifier(identifier, query)
        )
        if direct_identifiers:
            topic_keys = tuple(
                dict.fromkeys((*topic_keys, *(f"identifier:{item}" for item in direct_identifiers)))
            )
        if len(topic_keys) < 2 and not direct_identifiers:
            continue
        physical_anchor = (
            local_pair if local_is_branch and local_pair is not None else title_pair or local_pair
        )
        if physical_anchor is None:
            continue
        identifier_candidates = _scoped_anchor_identifiers(physical_anchor[1][2])
        anchor_identifier = (
            direct_identifiers[0]
            if direct_identifiers
            else identifier_candidates[0]
            if identifier_candidates
            else _normalize_table_cell(physical_anchor[1][2])
        )
        if not _is_specific_identifier(anchor_identifier):
            continue
        selection_kind: Literal[
            "complete_sequence",
            "query_selected_continuation",
            "query_selected_branch",
            "query_selected_prefix",
        ] = "complete_sequence"
        selection_priority = 2
        match_keys = topic_keys
        if _query_selects_procedure_continuation(query, first_item.body):
            item_keys = _procedure_anchor_match_keys(first_item.body, query)
            match_keys = tuple(dict.fromkeys((*topic_keys, *item_keys)))
            selection_kind = "query_selected_continuation"
            selection_priority = 5
        elif local_is_branch:
            local_keys = _procedure_anchor_match_keys(local_text, query)
            local_direct = any(
                _anchor_contains_identifier(identifier, query)
                for identifier in _scoped_anchor_identifiers(local_text)
            )
            if len(local_keys) >= 3 or local_direct:
                match_keys = tuple(dict.fromkeys((*topic_keys, *local_keys)))
                selection_kind = "query_selected_branch"
                selection_priority = 4
        elif first_item.ordinal != 1:
            continue
        anchor_score = len(match_keys) + (100 if direct_identifiers else 0)
        choices.append(
            (
                selection_priority,
                anchor_score,
                anchor_field,
                anchor_identifier,
                physical_anchor[0],
                physical_anchor[1],
                match_keys,
                selection_kind,
            )
        )
    if not choices:
        return None
    choices.sort(key=lambda item: (item[0], item[1], item[2] == "source_query"), reverse=True)
    best = choices[0]
    return best[2], best[3], best[4], best[5], best[6], best[7], best[0]


def _validate_scoped_procedure_items(items: tuple[_OrderedListLine, ...]) -> str:
    if len(items) < 2:
        raise ValueError("scoped procedure list requires at least two ordered steps")
    expected_ordinals = tuple(range(items[0].ordinal, items[0].ordinal + len(items)))
    if tuple(item.ordinal for item in items) != expected_ordinals:
        raise ValueError("scoped procedure list ordinals must increase without gaps")
    marker_styles = {item.marker_style for item in items}
    if len(marker_styles) != 1:
        raise ValueError("scoped procedure list items must use one stable marker style")
    closed_bodies = tuple(_closed_procedure_action_body(item.body) for item in items)
    normalized_bodies = tuple(_normalize_table_cell(item) for item in closed_bodies)
    if any(not item for item in normalized_bodies) or len(normalized_bodies) != len(
        set(normalized_bodies)
    ):
        raise ValueError("scoped procedure list items must be non-empty and semantically distinct")
    joined = "\n".join(closed_bodies)
    if _SCOPED_OPEN_ENUMERATION_RE.search(joined):
        raise ValueError("scoped procedure list contains an open-ended enumeration cue")
    if _SCOPED_TRUNCATION_RE.search(closed_bodies[-1].rstrip()):
        raise ValueError("scoped procedure list appears truncated")
    return next(iter(marker_styles))


def _locate_query_anchored_scoped_procedure_blocks(
    *,
    text: str,
    question: QuestionContract,
    obligation: QuestionObligation,
    paired_media_line_indices: frozenset[int] = frozenset(),
) -> tuple[_ScopedProcedureCandidate, ...]:
    """扫描可重放的完整流程、查询后缀和显式条件分支候选。"""

    if PACKED_TRUNCATION_MARKER in text:
        raise ValueError("scoped procedure list authority is truncated")
    nonempty = tuple(
        (line, trimmed)
        for line in _physical_lines(text)
        if (trimmed := _trimmed_physical_line(line)) is not None
    )
    metadata_positions = _image_metadata_positions(
        nonempty,
        paired_media_line_indices=paired_media_line_indices,
    )
    subordinate_positions = _finite_subordinate_positions(
        nonempty,
        metadata_positions=metadata_positions,
    )
    candidates: list[_ScopedProcedureCandidate] = []
    for start_position, (line, _) in enumerate(nonempty):
        if start_position in metadata_positions or start_position in subordinate_positions:
            continue
        first_item = _ordered_list_line(line)
        if first_item is None:
            continue
        if not _first_item_matches_obligation_local_trigger(
            question=question,
            obligation=obligation,
            first_item=first_item,
        ):
            continue
        sequence = _scoped_procedure_sequence_from(
            nonempty,
            start_position=start_position,
            subordinate_positions=subordinate_positions,
            metadata_positions=metadata_positions,
            paired_media_line_indices=paired_media_line_indices,
        )
        if len(sequence.items) < 2:
            continue
        prefix = _select_query_bounded_procedure_prefix(
            nonempty=nonempty,
            start_position=start_position,
            sequence=sequence,
            question=question,
            obligation=obligation,
        )
        if prefix is not None:
            anchor_field = prefix.anchor_field
            anchor_identifier = prefix.anchor_identifier
            anchor = prefix.anchor
            anchor_trimmed = prefix.anchor_trimmed
            anchor_match_keys = prefix.anchor_match_keys
            selection_kind = "query_selected_prefix"
            selection_priority = 6
            items = prefix.items
            selected_gap_kinds = prefix.gap_kinds
            blocked_reason = None
            prefix_trigger_hash = prefix.prefix_trigger_hash
            excluded_successor_ordinal = prefix.excluded_successor_ordinal
            excluded_successor_hash = prefix.excluded_successor_hash
            boundary_rule_version = prefix.boundary_rule_version
        else:
            selected_anchor = _select_scoped_procedure_anchor(
                nonempty=nonempty,
                start_position=start_position,
                first_item=first_item,
                question=question,
                obligation=obligation,
                paired_media_line_indices=paired_media_line_indices,
            )
            if selected_anchor is None:
                continue
            (
                anchor_field,
                anchor_identifier,
                anchor,
                anchor_trimmed,
                anchor_match_keys,
                selection_kind,
                selection_priority,
            ) = selected_anchor
            items = sequence.items
            selected_gap_kinds = sequence.gap_kinds
            blocked_reason = sequence.blocked_reason
            prefix_trigger_hash = None
            excluded_successor_ordinal = None
            excluded_successor_hash = None
            boundary_rule_version = None
        marker_style = _validate_scoped_procedure_items(items)
        scope_text = text[anchor_trimmed[0] : items[-1].content_end]
        if _SCOPED_OPEN_ENUMERATION_RE.search(anchor_trimmed[2]):
            raise ValueError("scoped procedure list anchor contains an open-ended cue")
        if _SCOPED_OPEN_REFERENCE_RE.search(anchor_trimmed[2]):
            raise ValueError("scoped procedure list anchor contains an open cross-reference")
        if _SCOPED_TRUNCATION_RE.search(scope_text.rstrip()):
            raise ValueError("scoped procedure list scope appears truncated")
        body_match_keys = _procedure_anchor_match_keys(
            "\n".join(item.body for item in items),
            question.source_query
            if anchor_field == "source_query"
            else obligation.information_need,
        )
        ascii_entity_lengths = tuple(
            len(key.removeprefix("token:"))
            for key in anchor_match_keys
            if key.startswith("token:") and len(key.removeprefix("token:")) >= 8
        )
        candidates.append(
            _ScopedProcedureCandidate(
                block=_ProcedureListBlock(
                    start=anchor_trimmed[0],
                    end=items[-1].content_end,
                    title=anchor,
                    title_content_start=anchor_trimmed[0],
                    title_content_end=anchor_trimmed[1],
                    normalized_title=anchor_identifier,
                    items=items,
                    marker_style=marker_style,
                    gap_kinds=selected_gap_kinds,
                ),
                anchor_field=anchor_field,
                anchor_identifier=anchor_identifier,
                anchor_match_keys=anchor_match_keys,
                selection_kind=selection_kind,
                selection_priority=selection_priority,
                exact_identifier_match=any(
                    key.startswith("identifier:") for key in anchor_match_keys
                ),
                longest_ascii_entity_token_length=max(ascii_entity_lengths, default=0),
                discriminative_match_key_count=len(anchor_match_keys),
                body_match_key_count=len(body_match_keys),
                blocked_reason=blocked_reason,
                prefix_trigger_hash=prefix_trigger_hash,
                excluded_successor_ordinal=excluded_successor_ordinal,
                excluded_successor_hash=excluded_successor_hash,
                boundary_rule_version=boundary_rule_version,
            )
        )
    unique: dict[tuple[int, int, str, str], _ScopedProcedureCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.block.items[0].content_start,
            candidate.block.items[-1].content_end,
            candidate.selection_kind,
            candidate.blocked_reason or "",
        )
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _scoped_procedure_structure_payload(
    candidate: _ScopedProcedureCandidate,
) -> dict[str, Any]:
    block = candidate.block
    return {
        "normalized_title": block.normalized_title,
        "anchor_field": candidate.anchor_field,
        "anchor_identifier": candidate.anchor_identifier,
        "anchor_match_key_hashes": [sha256_digest(item) for item in candidate.anchor_match_keys],
        "selection_kind": candidate.selection_kind,
        "exact_identifier_match": candidate.exact_identifier_match,
        "longest_ascii_entity_token_length": (candidate.longest_ascii_entity_token_length),
        "discriminative_match_key_count": candidate.discriminative_match_key_count,
        "body_match_key_count": candidate.body_match_key_count,
        "blocked_reason": candidate.blocked_reason,
        "prefix_trigger_hash": candidate.prefix_trigger_hash,
        "excluded_successor_ordinal": candidate.excluded_successor_ordinal,
        "excluded_successor_hash": candidate.excluded_successor_hash,
        "boundary_rule_version": candidate.boundary_rule_version,
        "marker_style": block.marker_style,
        "ordinals": [item.ordinal for item in block.items],
        "items": [
            _normalize_table_cell(_closed_procedure_action_body(item.body)) for item in block.items
        ],
        "gap_kinds": [list(item) for item in block.gap_kinds],
    }


def _scoped_procedure_candidate_rank(
    candidate: _ScopedProcedureCandidate,
) -> tuple[int, int, int, int, int]:
    """保守排序：查询显式选择、精确标识、长实体 token、区分性覆盖依次优先。"""

    return (
        candidate.selection_priority,
        int(candidate.exact_identifier_match),
        candidate.longest_ascii_entity_token_length,
        candidate.discriminative_match_key_count,
        candidate.body_match_key_count,
    )


def _physical_lines(value: str) -> tuple[_PhysicalLine, ...]:
    lines: list[_PhysicalLine] = []
    start = 0
    for delimiter in re.finditer(r"\r\n|\r|\n", value):
        lines.append(
            _PhysicalLine(
                index=len(lines),
                start=start,
                end=delimiter.start(),
                text=value[start : delimiter.start()],
            )
        )
        start = delimiter.end()
    if start < len(value):
        lines.append(
            _PhysicalLine(
                index=len(lines),
                start=start,
                end=len(value),
                text=value[start:],
            )
        )
    return tuple(lines)


def _parse_simple_markdown_row(line: _PhysicalLine) -> _MarkdownRow | None:
    shape = markdown_table_row_shape(line.text)
    if shape is None:
        return None
    body = line.text.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = tuple(cell.strip() for cell in body.split("|"))
    if len(cells) != shape[0]:
        raise ValueError("Markdown table row parser produced an inconsistent shape")
    return _MarkdownRow(line=line, cells=cells, is_separator=shape[1])


def _locate_containing_table(
    *,
    text: str,
    witness_start: int,
    witness_end: int,
) -> _SimpleMarkdownTable:
    lines = _physical_lines(text)
    witness_line = next(
        (line for line in lines if line.start == witness_start and line.end == witness_end),
        None,
    )
    if witness_line is None:
        raise ValueError("table_data_row witness must cover exactly one complete physical line")
    witness_row = _parse_simple_markdown_row(witness_line)
    if witness_row is None or witness_row.is_separator:
        raise ValueError("witness is not a complete simple Markdown data row")
    column_count = len(witness_row.cells)

    separator_index = witness_line.index - 1
    while separator_index >= 0:
        candidate = _parse_simple_markdown_row(lines[separator_index])
        if candidate is None or candidate.is_separator:
            break
        if len(candidate.cells) != column_count:
            raise ValueError("Markdown table data rows have inconsistent column counts")
        separator_index -= 1
    if separator_index < 1:
        raise ValueError("containing Markdown table is missing an adjacent header and separator")

    separator = _parse_simple_markdown_row(lines[separator_index])
    if separator is None or not separator.is_separator:
        raise ValueError("containing Markdown table is missing an adjacent separator row")
    if len(separator.cells) != column_count:
        raise ValueError("Markdown table separator has a different column count")

    header = _parse_simple_markdown_row(lines[separator_index - 1])
    if header is None or header.is_separator:
        raise ValueError("containing Markdown table is missing a simple header row")
    if len(header.cells) != column_count:
        raise ValueError("Markdown table header has a different column count")
    if separator_index >= 2 and _parse_simple_markdown_row(lines[separator_index - 2]) is not None:
        raise ValueError("adjacent Markdown tables require a non-table boundary")

    data_start_index = separator_index + 1
    for line_index in range(data_start_index, witness_line.index + 1):
        candidate = _parse_simple_markdown_row(lines[line_index])
        if candidate is None or candidate.is_separator:
            raise ValueError("witness is separated from the table header by a non-data row")
        if len(candidate.cells) != column_count:
            raise ValueError("Markdown table data rows have inconsistent column counts")

    data_end_index = witness_line.index
    for line_index in range(witness_line.index + 1, len(lines)):
        candidate = _parse_simple_markdown_row(lines[line_index])
        if candidate is None:
            break
        if candidate.is_separator:
            raise ValueError("adjacent Markdown tables require a non-table boundary")
        if len(candidate.cells) != column_count:
            raise ValueError("Markdown table data rows have inconsistent column counts")
        data_end_index = line_index

    data_rows = tuple(
        candidate
        for line_index in range(data_start_index, data_end_index + 1)
        if (candidate := _parse_simple_markdown_row(lines[line_index])) is not None
    )
    return _SimpleMarkdownTable(
        start=header.line.start,
        end=data_rows[-1].line.end,
        header=header,
        separator=separator,
        data_rows=data_rows,
        column_count=column_count,
        witness_row_index=witness_line.index - data_start_index,
    )


def _table_structure_payload(table: _SimpleMarkdownTable) -> dict[str, Any]:
    return {
        "column_count": table.column_count,
        "header": [_normalize_table_cell(cell) for cell in table.header.cells],
        "separator": [_normalize_table_cell(cell) for cell in table.separator.cells],
        "data_rows": [
            [_normalize_table_cell(cell) for cell in row.cells] for row in table.data_rows
        ],
        "witness_row_index": table.witness_row_index,
    }


def _scoped_table_structure_payload(table: MarkdownTableBlock) -> dict[str, Any]:
    return {
        "parse_mode": table.parse_mode.value,
        "column_count": table.column_count,
        "header": [_normalize_table_cell(cell.text) for cell in table.header.cells],
        "separator": [_normalize_table_cell(cell.text) for cell in table.separator.cells],
        "data_rows": [
            [_normalize_table_cell(cell.text) for cell in row.cells] for row in table.data_rows
        ],
    }


def _is_specific_table_selector(value: str) -> bool:
    normalized = _normalize_table_cell(value)
    anchor_normalized = _normalize_anchor_text(value)
    return (
        len(normalized) >= 2
        and bool(anchor_normalized)
        and anchor_normalized not in _GENERIC_IDENTIFIER_KEYS
        and not normalized.isdecimal()
    )


def _anchor_explicitly_mentions_table_selector(value: str, anchor: str) -> bool:
    if not _is_specific_table_selector(value):
        return False
    normalized_value = _normalize_table_cell(value)
    normalized_anchor = _normalize_table_cell(anchor)
    if any(not character.isalnum() for character in normalized_value):
        if normalized_value in normalized_anchor:
            return True
    elif _anchor_contains_identifier(value, anchor):
        return True
    # OCR 菜单表常把展示序号并入 cell；查询通常只复述序号后的标签。
    without_ordinal = re.sub(r"^\d{1,3}(?:[.)、．])\s*", "", value).strip()
    return without_ordinal != value and _anchor_contains_identifier(without_ordinal, anchor)


def _scoped_table_candidates_for_anchor(
    *,
    table_pairs: Sequence[tuple[int, int, MarkdownTableBlock, MarkdownTableBlock]],
    anchor: str,
    anchor_field: Literal["source_query", "information_need"],
) -> tuple[_ScopedTableSelectionCandidate, ...]:
    candidates: list[_ScopedTableSelectionCandidate] = []
    for authority_index, table_index, canonical_table, packed_table in table_pairs:
        canonical_headers = tuple(
            _normalize_table_cell(item.text) for item in canonical_table.header.cells
        )
        packed_headers = tuple(
            _normalize_table_cell(item.text) for item in packed_table.header.cells
        )
        if canonical_headers != packed_headers:
            raise ValueError("canonical and packed scoped table headers differ")
        if len(canonical_headers) != len(set(canonical_headers)):
            raise ValueError("scoped table-selection columns are not uniquely identified")
        for column_index in range(canonical_table.column_count):
            canonical_values = tuple(
                _normalize_table_cell(row.cells[column_index].text)
                for row in canonical_table.data_rows
            )
            packed_values = tuple(
                _normalize_table_cell(row.cells[column_index].text)
                for row in packed_table.data_rows
            )
            if canonical_values != packed_values:
                raise ValueError("canonical and packed scoped table selector columns differ")
            distinct_values = tuple(dict.fromkeys(canonical_values))
            selector_values = tuple(
                value
                for value in distinct_values
                if _anchor_explicitly_mentions_table_selector(
                    canonical_table.data_rows[canonical_values.index(value)]
                    .cells[column_index]
                    .text,
                    anchor,
                )
            )
            if not selector_values:
                continue
            if any(
                not row.cells[column_index].normalized_text
                and any(
                    cell.normalized_text
                    for other_index, cell in enumerate(row.cells)
                    if other_index != column_index
                )
                for row in canonical_table.data_rows
            ):
                # 该列存在无法归属的空 selector 延续，只淘汰这一列候选；同一
                # authority 的其他显式完整 selector 仍可参加 catalog-wide 唯一性比较。
                continue
            selected_row_indices = tuple(
                index for index, value in enumerate(canonical_values) if value in selector_values
            )
            if not selected_row_indices:
                raise ValueError("scoped table-selection selector selected no rows")
            if any(
                not cell.normalized_text
                for row_index in selected_row_indices
                for cell in canonical_table.data_rows[row_index].cells
            ):
                # 可以保留同一 authority 中无关分组的不完整 OCR 行，但绝不把它们
                # 签成查询所选的完整记录。
                continue
            candidates.append(
                _ScopedTableSelectionCandidate(
                    authority_index=authority_index,
                    table_index=table_index,
                    canonical_table=canonical_table,
                    packed_table=packed_table,
                    selector_column_index=column_index,
                    selector_values=selector_values,
                    selected_row_indices=selected_row_indices,
                    anchor_field=anchor_field,
                )
            )
    return tuple(candidates)


def _unique_most_specific_table_candidate(
    candidates: Sequence[_ScopedTableSelectionCandidate],
) -> _ScopedTableSelectionCandidate | None:
    """仅在一个 selector 的每个值都比其余候选更具体时选中。"""

    if len(candidates) == 1:
        return candidates[0]
    ranked: list[tuple[int, _ScopedTableSelectionCandidate]] = []
    for candidate in candidates:
        lengths = tuple(
            len(_normalize_anchor_text(re.sub(r"^\d{1,3}(?:[.)、．])\s*", "", value).strip()))
            for value in candidate.selector_values
        )
        if lengths and all(lengths):
            ranked.append((min(lengths), candidate))
    if not ranked:
        return None
    best_score = max(score for score, _ in ranked)
    best = [candidate for score, candidate in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _adjacent_nonempty_table_prose(
    *,
    text: str,
    table: MarkdownTableBlock,
) -> tuple[str, ...]:
    lines = _physical_lines(text)
    before = next(
        (
            line.text.strip()
            for line in reversed(lines)
            if line.end <= table.start and line.text.strip()
        ),
        None,
    )
    after = next(
        (line.text.strip() for line in lines if line.start >= table.end and line.text.strip()),
        None,
    )
    return tuple(value for value in (before, after) if value is not None)


def _validate_selected_table_continuations(
    *,
    text: str,
    table: MarkdownTableBlock,
    selector_values: Sequence[str],
) -> None:
    for row in table.data_rows:
        for cell in row.cells:
            normalized = _normalize_anchor_text(cell.text)
            if normalized in _SCOPED_TABLE_ANAPHOR_CELL_KEYS or (
                _SCOPED_TABLE_ANAPHOR_TEXT_RE.search(_identity_scan_text(cell.text))
            ):
                raise ValueError(
                    "scoped table-selection selected table contains an ambiguous continuation"
                )
    for prose in _adjacent_nonempty_table_prose(text=text, table=table):
        identity = _identity_scan_text(prose)
        if _SCOPED_TABLE_ANAPHOR_TEXT_RE.search(identity):
            raise ValueError("scoped table-selection selected table has an ambiguous local anaphor")
        if _SCOPED_TABLE_LOCAL_SUPPLEMENT_RE.search(identity) and any(
            _identifier_mention_count(selector, prose) > 0 for selector in selector_values
        ):
            raise ValueError(
                "scoped table-selection selected table has a local selector supplement"
            )


def _validate_catalog_table_selector_supplements(
    *,
    texts: Sequence[str],
    selector_values: Sequence[str],
) -> None:
    """拒绝任一 representation 表外对已选 selector 的显式续写。"""

    for text in texts:
        tables = scan_markdown_table_blocks(text)
        for line in _physical_lines(text):
            if any(table.start <= line.start and line.end <= table.end for table in tables):
                continue
            prose = line.text.strip()
            if not prose:
                continue
            identity = _identity_scan_text(prose)
            is_supplement = _SCOPED_TABLE_LOCAL_SUPPLEMENT_RE.search(identity) is not None
            if not is_supplement:
                continue
            mentions_selector = any(
                _identifier_mention_count(selector, prose) > 0 for selector in selector_values
            )
            is_ambiguous_continuation = _SCOPED_TABLE_ANAPHOR_TEXT_RE.search(identity) is not None
            if mentions_selector or is_ambiguous_continuation:
                raise ValueError(
                    "scoped table-selection authority has an external selector supplement"
                )


def _catalog_selector_cell_counts(
    *,
    selector_values: Sequence[str],
    tables: Sequence[MarkdownTableBlock],
) -> tuple[int, ...]:
    return tuple(
        sum(
            _identifier_mention_count(selector, cell.text) > 0
            for table in tables
            for row in table.data_rows
            for cell in row.cells
        )
        for selector in selector_values
    )


def _validate_table_pair(
    *,
    canonical_table: _SimpleMarkdownTable,
    packed_table: _SimpleMarkdownTable,
) -> str:
    if canonical_table.header.line.index != packed_table.header.line.index:
        raise ValueError("canonical and packed witnesses resolve to different table positions")
    canonical_structure = _table_structure_payload(canonical_table)
    packed_structure = _table_structure_payload(packed_table)
    if canonical_structure != packed_structure:
        raise ValueError("canonical and packed Markdown table structures differ")
    return stable_content_hash(canonical_structure)


def _is_specific_identifier(value: str) -> bool:
    return len(value) >= 2 and not value.isnumeric() and value not in _GENERIC_IDENTIFIER_KEYS


def _is_identifier_header(value: str) -> bool:
    normalized_value = unicodedata.normalize("NFKC", value).casefold()
    ascii_tokens = tuple(re.findall(r"[a-z0-9]+", normalized_value))
    if ascii_tokens:
        return (
            ascii_tokens[-1] in _ENGLISH_IDENTIFIER_HEADER_TERMS
            or len(ascii_tokens) == 1
            and ascii_tokens[0] in _ENGLISH_IDENTIFIER_HEADER_COMPOUNDS
        )
    normalized = _normalize_anchor_text(value)
    return bool(normalized) and normalized.endswith(_CHINESE_IDENTIFIER_HEADER_SUFFIXES)


def _scan_mention_count(
    identifier: str,
    text: str,
    *,
    map_confusables: bool,
) -> int:
    """按 CJK substring / ASCII token 规则统计身份投影。"""

    normalized_identifier = _identity_scan_text(identifier, map_confusables=map_confusables)
    normalized_text = _identity_scan_text(text, map_confusables=map_confusables)
    if any(ord(character) > 127 and character.isalnum() for character in normalized_identifier):
        normalized_key = _normalize_anchor_text(normalized_identifier)
        return _normalize_anchor_text(normalized_text).count(normalized_key)
    identifier_tokens = tuple(re.findall(r"[a-z0-9]+", normalized_identifier))
    text_tokens = tuple(re.findall(r"[a-z0-9]+", normalized_text))
    if not identifier_tokens or len(identifier_tokens) > len(text_tokens):
        return 0
    width = len(identifier_tokens)
    return sum(
        text_tokens[index : index + width] == identifier_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _identifier_mention_count(identifier: str, text: str) -> int:
    """保守统计 catalog 中的显式键或视觉同键提及。"""

    return _scan_mention_count(identifier, text, map_confusables=True)


def _anchor_contains_identifier(identifier: str, anchor: str) -> bool:
    # 查询锚定不接受跨脚本 confusable；安全骨架仅用于拒绝 catalog 歧义。
    return _scan_mention_count(identifier, anchor, map_confusables=False) > 0


def _key_occurrences(table: _SimpleMarkdownTable, *, normalized_key: str) -> int:
    return sum(
        _normalize_anchor_text(row.cells[cell_index]) == normalized_key
        for row in table.data_rows
        for cell_index in (0, 1)
    )


def _validate_explicit_primary_key_column(
    table: _SimpleMarkdownTable,
    *,
    key_cell_index: Literal[0, 1],
) -> None:
    """严格闭包不推断 ditto/空白续行，并要求所选键列显式唯一。"""

    normalized_keys = tuple(
        _normalize_anchor_text(_identity_scan_text(row.cells[key_cell_index]))
        for row in table.data_rows
    )
    if any(not _is_specific_identifier(item) for item in normalized_keys):
        raise ValueError(
            "query-anchored closure requires an explicit primary key in every table row"
        )
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ValueError("query-anchored closure requires unique explicit table primary keys")


def _select_query_anchor(
    *,
    canonical_table: _SimpleMarkdownTable,
    packed_table: _SimpleMarkdownTable,
    question: QuestionContract,
    obligation: QuestionObligation,
) -> tuple[int, Literal["source_query", "information_need"], str, Literal[1]]:
    anchors: tuple[tuple[Literal["source_query", "information_need"], str], ...] = (
        ("source_query", question.source_query),
        ("information_need", obligation.information_need),
    )
    canonical_witness_row = canonical_table.data_rows[canonical_table.witness_row_index]
    packed_witness_row = packed_table.data_rows[packed_table.witness_row_index]
    anchored_candidates: dict[tuple[int, str], Literal["source_query", "information_need"]] = {}
    for cell_index in (0, 1):
        canonical_cell = canonical_witness_row.cells[cell_index]
        canonical_key = _normalize_anchor_text(canonical_cell)
        packed_key = _normalize_anchor_text(packed_witness_row.cells[cell_index])
        if canonical_key != packed_key or not _is_specific_identifier(canonical_key):
            continue
        for anchor_field, anchor in anchors:
            if not _anchor_contains_identifier(canonical_cell, anchor):
                continue
            anchored_candidates.setdefault((cell_index, canonical_key), anchor_field)
    if len(anchored_candidates) > 1:
        raise ValueError("exactly one of the first two table cells may anchor the query")
    if anchored_candidates:
        (cell_index, canonical_key), anchor_field = next(iter(anchored_candidates.items()))
        if not _is_identifier_header(canonical_table.header.cells[cell_index]):
            raise ValueError(
                "query-anchored closure requires an explicit identifier header for the key column"
            )
        _validate_explicit_primary_key_column(
            canonical_table,
            key_cell_index=cell_index,
        )
        _validate_explicit_primary_key_column(
            packed_table,
            key_cell_index=cell_index,
        )
        canonical_occurrences = _key_occurrences(canonical_table, normalized_key=canonical_key)
        packed_occurrences = _key_occurrences(packed_table, normalized_key=canonical_key)
        if canonical_occurrences == packed_occurrences == 1:
            return cell_index, anchor_field, canonical_key, 1
        raise ValueError("query-anchored table key must occur exactly once in the containing table")
    raise ValueError(
        "neither of the first two table cells is a specific identifier anchored in the query"
    )


def _receipt_payload(
    *,
    question_contract_id: str,
    question_contract_hash: str,
    catalog_id: str,
    catalog_hash: str,
    obligation_id: str,
    obligation_hash: str,
    witness_id: str,
    witness_hash: str,
    evidence_id: str,
    evidence_content_hash: str,
    representation_id: str,
    span_ref_hash: str,
    row_hash: str,
    packed_row_hash: str,
    table_hash: str,
    packed_table_hash: str,
    header_hash: str,
    packed_header_hash: str,
    separator_hash: str,
    packed_separator_hash: str,
    table_structure_hash: str,
    table_start: int,
    table_end: int,
    packed_table_start: int,
    packed_table_end: int,
    table_line_index: int,
    table_column_count: int,
    table_data_row_count: int,
    witness_table_row_index: int,
    key_hash: str,
    key_cell_index: Literal[0, 1],
    key_occurrence_count: Literal[1],
    catalog_representation_count: Literal[1],
    catalog_key_mention_count: Literal[1],
    canonical_non_table_character_count: Literal[0],
    packed_non_table_character_count: Literal[0],
    anchor_field: Literal["source_query", "information_need"],
) -> dict[str, Any]:
    return {
        "schema_version": QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION,
        "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
        "question_contract_id": question_contract_id,
        "question_contract_hash": question_contract_hash,
        "catalog_id": catalog_id,
        "catalog_hash": catalog_hash,
        "obligation_id": obligation_id,
        "obligation_hash": obligation_hash,
        "witness_id": witness_id,
        "witness_hash": witness_hash,
        "evidence_id": evidence_id,
        "evidence_content_hash": evidence_content_hash,
        "representation_id": representation_id,
        "span_ref_hash": span_ref_hash,
        "row_hash": row_hash,
        "packed_row_hash": packed_row_hash,
        "table_hash": table_hash,
        "packed_table_hash": packed_table_hash,
        "header_hash": header_hash,
        "packed_header_hash": packed_header_hash,
        "separator_hash": separator_hash,
        "packed_separator_hash": packed_separator_hash,
        "table_structure_hash": table_structure_hash,
        "table_start": table_start,
        "table_end": table_end,
        "packed_table_start": packed_table_start,
        "packed_table_end": packed_table_end,
        "table_line_index": table_line_index,
        "table_column_count": table_column_count,
        "table_data_row_count": table_data_row_count,
        "witness_table_row_index": witness_table_row_index,
        "key_hash": key_hash,
        "key_cell_index": key_cell_index,
        "key_occurrence_count": key_occurrence_count,
        "catalog_representation_count": catalog_representation_count,
        "catalog_key_mention_count": catalog_key_mention_count,
        "canonical_non_table_character_count": canonical_non_table_character_count,
        "packed_non_table_character_count": packed_non_table_character_count,
        "anchor_field": anchor_field,
    }


class QueryAnchoredTableRowClosureReceipt(AnsweringContractModel):
    """证明完整表格中唯一记录键由原问题显式锚定。"""

    schema_version: Literal["query-anchored-table-row-closure-receipt.v4"] = (
        QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-default-ignorable-markup-aware-token.v3"] = (
        QUERY_ANCHOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qtrc_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    witness_id: str = Field(pattern=WITNESS_ID_PATTERN)
    witness_hash: str = Field(pattern=SHA256_PATTERN)
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    span_ref_hash: str = Field(pattern=SHA256_PATTERN)
    row_hash: str = Field(pattern=SHA256_PATTERN)
    packed_row_hash: str = Field(pattern=SHA256_PATTERN)
    table_hash: str = Field(pattern=SHA256_PATTERN)
    packed_table_hash: str = Field(pattern=SHA256_PATTERN)
    header_hash: str = Field(pattern=SHA256_PATTERN)
    packed_header_hash: str = Field(pattern=SHA256_PATTERN)
    separator_hash: str = Field(pattern=SHA256_PATTERN)
    packed_separator_hash: str = Field(pattern=SHA256_PATTERN)
    table_structure_hash: str = Field(pattern=SHA256_PATTERN)
    table_start: int = Field(ge=0)
    table_end: int = Field(gt=0)
    packed_table_start: int = Field(ge=0)
    packed_table_end: int = Field(gt=0)
    table_line_index: int = Field(ge=0)
    table_column_count: int = Field(ge=2)
    table_data_row_count: int = Field(ge=1)
    witness_table_row_index: int = Field(ge=0)
    key_hash: str = Field(pattern=SHA256_PATTERN)
    key_cell_index: Literal[0, 1]
    key_occurrence_count: Literal[1] = 1
    catalog_representation_count: Literal[1] = 1
    catalog_key_mention_count: Literal[1] = 1
    canonical_non_table_character_count: Literal[0] = 0
    packed_non_table_character_count: Literal[0] = 0
    anchor_field: Literal["source_query", "information_need"]

    def _identity_payload(self) -> dict[str, Any]:
        return _receipt_payload(
            question_contract_id=self.question_contract_id,
            question_contract_hash=self.question_contract_hash,
            catalog_id=self.catalog_id,
            catalog_hash=self.catalog_hash,
            obligation_id=self.obligation_id,
            obligation_hash=self.obligation_hash,
            witness_id=self.witness_id,
            witness_hash=self.witness_hash,
            evidence_id=self.evidence_id,
            evidence_content_hash=self.evidence_content_hash,
            representation_id=self.representation_id,
            span_ref_hash=self.span_ref_hash,
            row_hash=self.row_hash,
            packed_row_hash=self.packed_row_hash,
            table_hash=self.table_hash,
            packed_table_hash=self.packed_table_hash,
            header_hash=self.header_hash,
            packed_header_hash=self.packed_header_hash,
            separator_hash=self.separator_hash,
            packed_separator_hash=self.packed_separator_hash,
            table_structure_hash=self.table_structure_hash,
            table_start=self.table_start,
            table_end=self.table_end,
            packed_table_start=self.packed_table_start,
            packed_table_end=self.packed_table_end,
            table_line_index=self.table_line_index,
            table_column_count=self.table_column_count,
            table_data_row_count=self.table_data_row_count,
            witness_table_row_index=self.witness_table_row_index,
            key_hash=self.key_hash,
            key_cell_index=self.key_cell_index,
            key_occurrence_count=self.key_occurrence_count,
            catalog_representation_count=self.catalog_representation_count,
            catalog_key_mention_count=self.catalog_key_mention_count,
            canonical_non_table_character_count=self.canonical_non_table_character_count,
            packed_non_table_character_count=self.packed_non_table_character_count,
            anchor_field=self.anchor_field,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.table_end <= self.table_start:
            raise ValueError("table span must satisfy table_start < table_end")
        if self.packed_table_end <= self.packed_table_start:
            raise ValueError("packed table span must satisfy packed_table_start < packed_table_end")
        if self.witness_table_row_index >= self.table_data_row_count:
            raise ValueError("witness_table_row_index lies outside the containing table")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match QueryAnchoredTableRowClosureReceipt")
        if self.receipt_id != stable_identifier("qtrc_", payload):
            raise ValueError("receipt_id does not match QueryAnchoredTableRowClosureReceipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witness: EvidenceWitness,
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
        catalog_texts: Sequence[str],
    ) -> Self:
        """由完整 artifact 图建立来源有界的表记录闭包回执。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        witness = EvidenceWitness.from_dict(witness.to_dict())

        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("query-anchored closure requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("witness catalog belongs to a different question contract")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("obligation does not belong to the question contract")
        if obligation.closure != FacetClosure.OPEN_ENUMERATION:
            raise ValueError("query-anchored closure requires an open_enumeration obligation")
        if witness.atom_kind != WitnessAtomKind.TABLE_DATA_ROW:
            raise ValueError("query-anchored closure requires a table_data_row witness")

        try:
            catalog_index = catalog.witness_ids.index(witness.witness_id)
        except ValueError as exc:
            raise ValueError("witness does not belong to the complete catalog") from exc
        if witness.ordinal != catalog_index:
            raise ValueError("witness ordinal does not match its catalog position")
        if (
            witness.context_pack_hash != catalog.context_pack_hash
            or witness.authorized_scope_hash != catalog.authorized_scope_hash
        ):
            raise ValueError("witness scope does not match the complete catalog")

        span_ref = witness.span_ref.validate_artifacts(
            evidence_id=evidence_id,
            representation_id=representation_id,
            canonical_text=canonical_text,
            packed_text=packed_text,
        )
        canonical_table = _locate_containing_table(
            text=canonical_text,
            witness_start=span_ref.canonical_start,
            witness_end=span_ref.canonical_end,
        )
        packed_table = _locate_containing_table(
            text=packed_text,
            witness_start=span_ref.packed_start,
            witness_end=span_ref.packed_end,
        )
        table_structure_hash = _validate_table_pair(
            canonical_table=canonical_table,
            packed_table=packed_table,
        )
        key_cell_index, anchor_field, normalized_key, key_occurrence_count = _select_query_anchor(
            canonical_table=canonical_table,
            packed_table=packed_table,
            question=question,
            obligation=obligation,
        )
        canonical_non_table_character_count = sum(
            not character.isspace()
            for character in (
                canonical_text[: canonical_table.start] + canonical_text[canonical_table.end :]
            )
        )
        packed_non_table_character_count = sum(
            not character.isspace()
            for character in packed_text[: packed_table.start] + packed_text[packed_table.end :]
        )
        if canonical_non_table_character_count or packed_non_table_character_count:
            raise ValueError(
                "query-anchored closure requires the table to cover the full representation"
            )
        if len(catalog_texts) != 1 or catalog_texts[0] != packed_text:
            raise ValueError(
                "query-anchored closure requires exactly one complete catalog representation"
            )
        key_identifier = canonical_table.data_rows[canonical_table.witness_row_index].cells[
            key_cell_index
        ]
        catalog_key_mention_count = sum(
            _identifier_mention_count(key_identifier, text) for text in catalog_texts
        )
        if catalog_key_mention_count != 1:
            raise ValueError(
                "query-anchored table key must occur exactly once across the complete catalog"
            )

        canonical_row = canonical_table.data_rows[canonical_table.witness_row_index].line.text
        packed_row = packed_table.data_rows[packed_table.witness_row_index].line.text
        payload = _receipt_payload(
            question_contract_id=question.contract_id,
            question_contract_hash=question.contract_hash,
            catalog_id=catalog.catalog_id,
            catalog_hash=catalog.catalog_hash,
            obligation_id=obligation.obligation_id,
            obligation_hash=obligation.obligation_hash,
            witness_id=witness.witness_id,
            witness_hash=witness.witness_hash,
            evidence_id=span_ref.evidence_id,
            evidence_content_hash=span_ref.evidence_content_hash,
            representation_id=span_ref.representation_id,
            span_ref_hash=span_ref.span_ref_hash,
            row_hash=sha256_digest(canonical_row),
            packed_row_hash=sha256_digest(packed_row),
            table_hash=sha256_digest(canonical_text[canonical_table.start : canonical_table.end]),
            packed_table_hash=sha256_digest(packed_text[packed_table.start : packed_table.end]),
            header_hash=sha256_digest(canonical_table.header.line.text),
            packed_header_hash=sha256_digest(packed_table.header.line.text),
            separator_hash=sha256_digest(canonical_table.separator.line.text),
            packed_separator_hash=sha256_digest(packed_table.separator.line.text),
            table_structure_hash=table_structure_hash,
            table_start=canonical_table.start,
            table_end=canonical_table.end,
            packed_table_start=packed_table.start,
            packed_table_end=packed_table.end,
            table_line_index=canonical_table.header.line.index,
            table_column_count=canonical_table.column_count,
            table_data_row_count=len(canonical_table.data_rows),
            witness_table_row_index=canonical_table.witness_row_index,
            key_hash=sha256_digest(normalized_key),
            key_cell_index=key_cell_index,
            key_occurrence_count=key_occurrence_count,
            catalog_representation_count=1,
            catalog_key_mention_count=1,
            canonical_non_table_character_count=0,
            packed_non_table_character_count=0,
            anchor_field=anchor_field,
        )
        return cls(
            **payload,
            receipt_id=stable_identifier("qtrc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witness: EvidenceWitness,
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
        catalog_texts: Sequence[str],
    ) -> Self:
        """从全部输入重建表记录回执，拒绝只重验自洽哈希。"""

        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witness=witness,
            evidence_id=evidence_id,
            representation_id=representation_id,
            canonical_text=canonical_text,
            packed_text=packed_text,
            catalog_texts=catalog_texts,
        )
        if rebuilt != self:
            raise ValueError(
                "QueryAnchoredTableRowClosureReceipt does not match supplied artifacts"
            )
        return self


def _procedure_list_receipt_payload(
    *,
    question_contract_id: str,
    question_contract_hash: str,
    catalog_id: str,
    catalog_hash: str,
    obligation_id: str,
    obligation_hash: str,
    witness_ids: tuple[str, ...],
    witness_hashes: tuple[str, ...],
    witness_ordinals: tuple[int, ...],
    span_ref_hashes: tuple[str, ...],
    evidence_id: str,
    evidence_content_hash: str,
    representation_id: str,
    canonical_item_hashes: tuple[str, ...],
    packed_item_hashes: tuple[str, ...],
    canonical_block_hash: str,
    packed_block_hash: str,
    canonical_title_hash: str,
    packed_title_hash: str,
    list_structure_hash: str,
    canonical_block_start: int,
    canonical_block_end: int,
    packed_block_start: int,
    packed_block_end: int,
    title_line_index: int,
    first_item_line_index: int,
    item_count: int,
    item_ordinals: tuple[int, ...],
    marker_style: str,
    title_anchor_hash: str,
    anchor_field: Literal["source_query", "information_need"],
    catalog_representation_count: Literal[1],
    canonical_non_block_character_count: Literal[0],
    packed_non_block_character_count: Literal[0],
) -> dict[str, Any]:
    return {
        "schema_version": QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION,
        "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
        "question_contract_id": question_contract_id,
        "question_contract_hash": question_contract_hash,
        "catalog_id": catalog_id,
        "catalog_hash": catalog_hash,
        "obligation_id": obligation_id,
        "obligation_hash": obligation_hash,
        "witness_ids": list(witness_ids),
        "witness_hashes": list(witness_hashes),
        "witness_ordinals": list(witness_ordinals),
        "span_ref_hashes": list(span_ref_hashes),
        "evidence_id": evidence_id,
        "evidence_content_hash": evidence_content_hash,
        "representation_id": representation_id,
        "canonical_item_hashes": list(canonical_item_hashes),
        "packed_item_hashes": list(packed_item_hashes),
        "canonical_block_hash": canonical_block_hash,
        "packed_block_hash": packed_block_hash,
        "canonical_title_hash": canonical_title_hash,
        "packed_title_hash": packed_title_hash,
        "list_structure_hash": list_structure_hash,
        "canonical_block_start": canonical_block_start,
        "canonical_block_end": canonical_block_end,
        "packed_block_start": packed_block_start,
        "packed_block_end": packed_block_end,
        "title_line_index": title_line_index,
        "first_item_line_index": first_item_line_index,
        "item_count": item_count,
        "item_ordinals": list(item_ordinals),
        "marker_style": marker_style,
        "title_anchor_hash": title_anchor_hash,
        "anchor_field": anchor_field,
        "catalog_representation_count": catalog_representation_count,
        "canonical_non_block_character_count": canonical_non_block_character_count,
        "packed_non_block_character_count": packed_non_block_character_count,
    }


class QueryAnchoredProcedureListClosureReceipt(AnsweringContractModel):
    """证明唯一完整 representation 中的有序步骤块已被全量选择。"""

    schema_version: Literal["query-anchored-procedure-list-closure-receipt.v1"] = (
        QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-default-ignorable-markup-aware-token.v3"] = (
        QUERY_ANCHOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qplc_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    witness_ids: tuple[str, ...] = Field(min_length=2, max_length=64)
    witness_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    witness_ordinals: tuple[int, ...] = Field(min_length=2, max_length=64)
    span_ref_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    canonical_item_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    packed_item_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    canonical_block_hash: str = Field(pattern=SHA256_PATTERN)
    packed_block_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_title_hash: str = Field(pattern=SHA256_PATTERN)
    packed_title_hash: str = Field(pattern=SHA256_PATTERN)
    list_structure_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_block_start: int = Field(ge=0)
    canonical_block_end: int = Field(gt=0)
    packed_block_start: int = Field(ge=0)
    packed_block_end: int = Field(gt=0)
    title_line_index: int = Field(ge=0)
    first_item_line_index: int = Field(ge=1)
    item_count: int = Field(ge=2, le=64)
    item_ordinals: tuple[int, ...] = Field(min_length=2, max_length=64)
    marker_style: str = Field(pattern=r"^(?:arabic|chinese):[.)．、]$")
    title_anchor_hash: str = Field(pattern=SHA256_PATTERN)
    anchor_field: Literal["source_query", "information_need"]
    catalog_representation_count: Literal[1] = 1
    canonical_non_block_character_count: Literal[0] = 0
    packed_non_block_character_count: Literal[0] = 0

    def _identity_payload(self) -> dict[str, Any]:
        return _procedure_list_receipt_payload(
            question_contract_id=self.question_contract_id,
            question_contract_hash=self.question_contract_hash,
            catalog_id=self.catalog_id,
            catalog_hash=self.catalog_hash,
            obligation_id=self.obligation_id,
            obligation_hash=self.obligation_hash,
            witness_ids=self.witness_ids,
            witness_hashes=self.witness_hashes,
            witness_ordinals=self.witness_ordinals,
            span_ref_hashes=self.span_ref_hashes,
            evidence_id=self.evidence_id,
            evidence_content_hash=self.evidence_content_hash,
            representation_id=self.representation_id,
            canonical_item_hashes=self.canonical_item_hashes,
            packed_item_hashes=self.packed_item_hashes,
            canonical_block_hash=self.canonical_block_hash,
            packed_block_hash=self.packed_block_hash,
            canonical_title_hash=self.canonical_title_hash,
            packed_title_hash=self.packed_title_hash,
            list_structure_hash=self.list_structure_hash,
            canonical_block_start=self.canonical_block_start,
            canonical_block_end=self.canonical_block_end,
            packed_block_start=self.packed_block_start,
            packed_block_end=self.packed_block_end,
            title_line_index=self.title_line_index,
            first_item_line_index=self.first_item_line_index,
            item_count=self.item_count,
            item_ordinals=self.item_ordinals,
            marker_style=self.marker_style,
            title_anchor_hash=self.title_anchor_hash,
            anchor_field=self.anchor_field,
            catalog_representation_count=self.catalog_representation_count,
            canonical_non_block_character_count=self.canonical_non_block_character_count,
            packed_non_block_character_count=self.packed_non_block_character_count,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        lengths = {
            len(self.witness_ids),
            len(self.witness_hashes),
            len(self.witness_ordinals),
            len(self.span_ref_hashes),
            len(self.canonical_item_hashes),
            len(self.packed_item_hashes),
            len(self.item_ordinals),
            self.item_count,
        }
        if len(lengths) != 1:
            raise ValueError("procedure list receipt arrays must equal item_count")
        if len(self.witness_ids) != len(set(self.witness_ids)):
            raise ValueError("procedure list receipt witness IDs must be unique")
        is_scoped = (
            self.schema_version == QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
        )
        if is_scoped:
            if self.witness_ordinals != tuple(sorted(set(self.witness_ordinals))):
                raise ValueError("scoped procedure witnesses must be unique and source ordered")
            expected_item_ordinals = tuple(
                range(self.item_ordinals[0], self.item_ordinals[0] + self.item_count)
            )
            if self.item_ordinals != expected_item_ordinals:
                raise ValueError("scoped procedure item ordinals must increase without gaps")
        else:
            if self.witness_ordinals != tuple(
                range(self.witness_ordinals[0], self.witness_ordinals[0] + self.item_count)
            ):
                raise ValueError(
                    "procedure list receipt witnesses must be contiguous in the catalog"
                )
            if self.item_ordinals != tuple(range(1, self.item_count + 1)):
                raise ValueError("procedure list receipt item ordinals must be contiguous from one")
        if self.canonical_block_end <= self.canonical_block_start:
            raise ValueError("canonical procedure block span is invalid")
        if self.packed_block_end <= self.packed_block_start:
            raise ValueError("packed procedure block span is invalid")
        if (
            self.schema_version == QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
            and self.first_item_line_index != self.title_line_index + 1
        ):
            raise ValueError("procedure title must be adjacent to its first item")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match procedure list closure receipt")
        if self.receipt_id != stable_identifier("qplc_", payload):
            raise ValueError("receipt_id does not match procedure list closure receipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
        catalog_texts: Sequence[str],
    ) -> Self:
        """从完整 artifact 图建立标题锚定的有序列表闭包回执。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("procedure list closure requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("witness catalog belongs to a different question contract")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("obligation does not belong to the question contract")
        if (
            obligation.closure != FacetClosure.OPEN_ENUMERATION
            or obligation.required_output_form != RequiredOutputForm.ORDERED_STEPS
        ):
            raise ValueError("procedure list closure requires an open ordered_steps obligation")
        if len(typed_witnesses) < 2 or any(
            item.atom_kind != WitnessAtomKind.LIST_ITEM for item in typed_witnesses
        ):
            raise ValueError("procedure list closure requires at least two list_item witnesses")
        witness_ordinals = tuple(item.ordinal for item in typed_witnesses)
        if witness_ordinals != tuple(
            range(witness_ordinals[0], witness_ordinals[0] + len(typed_witnesses))
        ):
            raise ValueError("procedure list witnesses must be contiguous in catalog order")
        for witness in typed_witnesses:
            try:
                catalog_index = catalog.witness_ids.index(witness.witness_id)
            except ValueError as exc:
                raise ValueError("procedure list witness is outside the catalog") from exc
            if witness.ordinal != catalog_index:
                raise ValueError("procedure list witness ordinal does not match the catalog")
            if (
                witness.context_pack_hash != catalog.context_pack_hash
                or witness.authorized_scope_hash != catalog.authorized_scope_hash
            ):
                raise ValueError("procedure list witness scope differs from the catalog")
            witness.span_ref.validate_artifacts(
                evidence_id=evidence_id,
                representation_id=representation_id,
                canonical_text=canonical_text,
                packed_text=packed_text,
            )
        if len(catalog_texts) != 1 or catalog_texts[0] != packed_text:
            raise ValueError(
                "procedure list closure requires exactly one complete catalog representation"
            )
        canonical_block = _locate_complete_procedure_list_block(canonical_text)
        packed_block = _locate_complete_procedure_list_block(packed_text)
        list_structure_hash = _validate_procedure_list_pair(
            canonical_block=canonical_block,
            packed_block=packed_block,
        )
        if len(canonical_block.items) != len(typed_witnesses):
            raise ValueError("procedure list receipt must bind every source item witness")
        for witness, canonical_item, packed_item in zip(
            typed_witnesses,
            canonical_block.items,
            packed_block.items,
            strict=True,
        ):
            if (
                witness.span_ref.canonical_start != canonical_item.content_start
                or witness.span_ref.canonical_end != canonical_item.content_end
                or witness.span_ref.packed_start != packed_item.content_start
                or witness.span_ref.packed_end != packed_item.content_end
            ):
                raise ValueError("procedure list witness does not cover its complete source line")
        anchor_field, normalized_title = _select_title_anchor(
            question=question,
            obligation=obligation,
            normalized_title=canonical_block.normalized_title,
        )
        payload = _procedure_list_receipt_payload(
            question_contract_id=question.contract_id,
            question_contract_hash=question.contract_hash,
            catalog_id=catalog.catalog_id,
            catalog_hash=catalog.catalog_hash,
            obligation_id=obligation.obligation_id,
            obligation_hash=obligation.obligation_hash,
            witness_ids=tuple(item.witness_id for item in typed_witnesses),
            witness_hashes=tuple(item.witness_hash for item in typed_witnesses),
            witness_ordinals=witness_ordinals,
            span_ref_hashes=tuple(item.span_ref.span_ref_hash for item in typed_witnesses),
            evidence_id=evidence_id,
            evidence_content_hash=typed_witnesses[0].span_ref.evidence_content_hash,
            representation_id=representation_id,
            canonical_item_hashes=tuple(
                sha256_digest(canonical_text[item.content_start : item.content_end])
                for item in canonical_block.items
            ),
            packed_item_hashes=tuple(
                sha256_digest(packed_text[item.content_start : item.content_end])
                for item in packed_block.items
            ),
            canonical_block_hash=sha256_digest(
                canonical_text[canonical_block.start : canonical_block.end]
            ),
            packed_block_hash=sha256_digest(packed_text[packed_block.start : packed_block.end]),
            canonical_title_hash=sha256_digest(
                canonical_text[
                    canonical_block.title_content_start : canonical_block.title_content_end
                ]
            ),
            packed_title_hash=sha256_digest(
                packed_text[packed_block.title_content_start : packed_block.title_content_end]
            ),
            list_structure_hash=list_structure_hash,
            canonical_block_start=canonical_block.start,
            canonical_block_end=canonical_block.end,
            packed_block_start=packed_block.start,
            packed_block_end=packed_block.end,
            title_line_index=canonical_block.title.index,
            first_item_line_index=canonical_block.items[0].line.index,
            item_count=len(typed_witnesses),
            item_ordinals=tuple(item.ordinal for item in canonical_block.items),
            marker_style=canonical_block.marker_style,
            title_anchor_hash=sha256_digest(normalized_title),
            anchor_field=anchor_field,
            catalog_representation_count=1,
            canonical_non_block_character_count=0,
            packed_non_block_character_count=0,
        )
        return cls(
            **payload,
            receipt_id=stable_identifier("qplc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        evidence_id: str,
        representation_id: str,
        canonical_text: str,
        packed_text: str,
        catalog_texts: Sequence[str],
    ) -> Self:
        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            evidence_id=evidence_id,
            representation_id=representation_id,
            canonical_text=canonical_text,
            packed_text=packed_text,
            catalog_texts=catalog_texts,
        )
        if rebuilt != self:
            raise ValueError("procedure list closure receipt does not match supplied artifacts")
        return self


class QueryAnchoredScopedProcedureListClosureReceipt(QueryAnchoredProcedureListClosureReceipt):
    """证明多 representation authority 中唯一查询锚定的局部编号流程已闭合。"""

    schema_version: Literal["query-anchored-scoped-procedure-list-closure-receipt.v4"] = (
        QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION
    )
    catalog_representation_count: int = Field(ge=1, le=256)
    canonical_non_block_character_count: int = Field(ge=0)
    packed_non_block_character_count: int = Field(ge=0)
    catalog_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_evidence_content_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_packed_text_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_authority_hash: str = Field(pattern=SHA256_PATTERN)
    selected_authority_index: int = Field(ge=0, le=255)
    authority_candidate_count: Literal[1] = 1
    canonical_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    packed_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    normalized_anchor_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_anchor_start: int = Field(ge=0)
    canonical_anchor_end: int = Field(gt=0)
    packed_anchor_start: int = Field(ge=0)
    packed_anchor_end: int = Field(gt=0)
    canonical_item_starts: tuple[int, ...] = Field(min_length=2, max_length=64)
    canonical_item_ends: tuple[int, ...] = Field(min_length=2, max_length=64)
    packed_item_starts: tuple[int, ...] = Field(min_length=2, max_length=64)
    packed_item_ends: tuple[int, ...] = Field(min_length=2, max_length=64)
    selection_kind: Literal[
        "complete_sequence",
        "query_selected_continuation",
        "query_selected_branch",
        "query_selected_prefix",
    ]
    anchor_match_key_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    prefix_trigger_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    excluded_successor_ordinal: int | None = Field(default=None, ge=2, le=999)
    excluded_successor_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    boundary_rule_version: Literal["explicit-query-before-successor.v1"] | None = None

    def _identity_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
            exclude={"receipt_id", "receipt_hash"},
        )

    @model_validator(mode="after")
    def validate_scoped_identity(self) -> Self:
        if len(self.anchor_match_key_hashes) != len(set(self.anchor_match_key_hashes)) or any(
            re.fullmatch(SHA256_PATTERN, item) is None for item in self.anchor_match_key_hashes
        ):
            raise ValueError(
                "scoped procedure anchor match-key hashes must be unique SHA-256 values"
            )
        if self.selection_kind == "complete_sequence" and self.item_ordinals[0] != 1:
            raise ValueError("complete scoped procedure sequence must begin at ordinal one")
        if self.selection_kind == "query_selected_continuation" and self.item_ordinals[0] == 1:
            raise ValueError("query-selected continuation must begin after ordinal one")
        prefix_values = (
            self.prefix_trigger_hash,
            self.excluded_successor_ordinal,
            self.excluded_successor_hash,
            self.boundary_rule_version,
        )
        if self.selection_kind == "query_selected_prefix":
            if any(value is None for value in prefix_values):
                raise ValueError("query-selected prefix requires complete boundary evidence")
            if self.item_ordinals[0] != 1:
                raise ValueError("query-selected prefix must begin at ordinal one")
            if self.excluded_successor_ordinal != self.item_ordinals[-1] + 1:
                raise ValueError(
                    "query-selected prefix successor must immediately follow the selected prefix"
                )
        elif any(value is not None for value in prefix_values):
            raise ValueError("non-prefix scoped procedure cannot carry prefix boundary evidence")
        authority_lengths = {
            self.catalog_representation_count,
            len(self.catalog_evidence_ids),
            len(self.catalog_representation_ids),
            len(self.catalog_evidence_content_hashes),
            len(self.catalog_packed_text_hashes),
        }
        if len(authority_lengths) != 1:
            raise ValueError("scoped procedure authority arrays must equal representation count")
        if len(self.catalog_evidence_ids) != len(set(self.catalog_evidence_ids)):
            raise ValueError("scoped procedure authority Evidence IDs must be unique")
        if len(self.catalog_representation_ids) != len(set(self.catalog_representation_ids)):
            raise ValueError("scoped procedure authority representation IDs must be unique")
        if self.selected_authority_index >= self.catalog_representation_count:
            raise ValueError("selected scoped procedure authority index is out of range")
        if (
            self.catalog_evidence_ids[self.selected_authority_index] != self.evidence_id
            or self.catalog_representation_ids[self.selected_authority_index]
            != self.representation_id
            or self.catalog_evidence_content_hashes[self.selected_authority_index]
            != self.evidence_content_hash
        ):
            raise ValueError("selected scoped procedure authority identity is inconsistent")
        item_lengths = {
            self.item_count,
            len(self.canonical_item_starts),
            len(self.canonical_item_ends),
            len(self.packed_item_starts),
            len(self.packed_item_ends),
        }
        if len(item_lengths) != 1:
            raise ValueError("scoped procedure item spans must equal item_count")
        if any(
            start < self.canonical_block_start or end > self.canonical_block_end or end <= start
            for start, end in zip(
                self.canonical_item_starts,
                self.canonical_item_ends,
                strict=True,
            )
        ) or any(
            start < self.packed_block_start or end > self.packed_block_end or end <= start
            for start, end in zip(
                self.packed_item_starts,
                self.packed_item_ends,
                strict=True,
            )
        ):
            raise ValueError("scoped procedure item span is outside its bound scope")
        if tuple(self.canonical_item_starts) != tuple(sorted(self.canonical_item_starts)) or tuple(
            self.packed_item_starts
        ) != tuple(sorted(self.packed_item_starts)):
            raise ValueError("scoped procedure item spans must preserve source order")
        if not (
            self.canonical_block_start
            <= self.canonical_anchor_start
            < self.canonical_anchor_end
            <= self.canonical_item_starts[0]
        ) or not (
            self.packed_block_start
            <= self.packed_anchor_start
            < self.packed_anchor_end
            <= self.packed_item_starts[0]
        ):
            raise ValueError("scoped procedure anchor must precede its members")
        expected_authority_hash = stable_content_hash(
            {
                "evidence_ids": list(self.catalog_evidence_ids),
                "representation_ids": list(self.catalog_representation_ids),
                "canonical_hashes": list(self.catalog_evidence_content_hashes),
                "packed_hashes": list(self.catalog_packed_text_hashes),
            }
        )
        if self.catalog_authority_hash != expected_authority_hash:
            raise ValueError("catalog_authority_hash does not match scoped procedure authority")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedListAuthorityRepresentation],
    ) -> Self:
        """从完整 ContextPack authority 重扫并签发唯一局部编号流程回执。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        authorities = tuple(
            ScopedListAuthorityRepresentation(
                evidence_id=item.evidence_id,
                representation_id=item.representation_id,
                canonical_text=item.canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority_representations
        )
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("scoped procedure list closure requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("witness catalog belongs to a different question contract")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("obligation does not belong to the question contract")
        if (
            obligation.closure != FacetClosure.OPEN_ENUMERATION
            or obligation.required_output_form != RequiredOutputForm.ORDERED_STEPS
        ):
            raise ValueError(
                "scoped procedure list closure requires an open ordered_steps obligation"
            )
        if not authorities:
            raise ValueError("scoped procedure list closure requires complete raw authority")
        evidence_ids = tuple(item.evidence_id for item in authorities)
        representation_ids = tuple(item.representation_id for item in authorities)
        if len(evidence_ids) != len(set(evidence_ids)) or len(representation_ids) != len(
            set(representation_ids)
        ):
            raise ValueError("scoped procedure list authority identities must be unique")
        canonical_hashes = tuple(sha256_digest(item.canonical_text) for item in authorities)
        packed_hashes = tuple(sha256_digest(item.packed_text) for item in authorities)
        matched_candidates: list[
            tuple[
                int,
                ScopedListAuthorityRepresentation,
                _ScopedProcedureCandidate,
                _ScopedProcedureCandidate,
            ]
        ] = []
        for authority_index, authority in enumerate(authorities):
            canonical_media_lines, packed_media_lines = _paired_scoped_procedure_media_line_indices(
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )
            canonical_candidates = _locate_query_anchored_scoped_procedure_blocks(
                text=authority.canonical_text,
                question=question,
                obligation=obligation,
                paired_media_line_indices=canonical_media_lines,
            )
            packed_candidates = _locate_query_anchored_scoped_procedure_blocks(
                text=authority.packed_text,
                question=question,
                obligation=obligation,
                paired_media_line_indices=packed_media_lines,
            )
            if len(canonical_candidates) != len(packed_candidates):
                raise ValueError("canonical and packed scoped procedure candidates differ")
            unmatched_packed = list(packed_candidates)
            for canonical_candidate in canonical_candidates:
                canonical_structure = _scoped_procedure_structure_payload(canonical_candidate)
                matches = [
                    item
                    for item in unmatched_packed
                    if _scoped_procedure_structure_payload(item) == canonical_structure
                ]
                if len(matches) != 1:
                    raise ValueError("canonical and packed scoped procedure structures differ")
                packed_candidate = matches[0]
                unmatched_packed.remove(packed_candidate)
                matched_candidates.append(
                    (
                        authority_index,
                        authority,
                        canonical_candidate,
                        packed_candidate,
                    )
                )
        valid_candidates = [item for item in matched_candidates if item[2].blocked_reason is None]
        if not valid_candidates:
            raise ValueError(
                "scoped procedure list closure requires exactly one catalog-wide candidate"
            )
        best_rank = max(_scoped_procedure_candidate_rank(item[2]) for item in valid_candidates)
        winners = [
            item
            for item in valid_candidates
            if _scoped_procedure_candidate_rank(item[2]) == best_rank
        ]
        if len(winners) != 1:
            raise ValueError(
                "scoped procedure list closure requires exactly one catalog-wide candidate"
            )
        winner_candidate = winners[0][2]
        if winner_candidate.exact_identifier_match and any(
            item[0] != winners[0][0]
            and item[2].exact_identifier_match
            and _normalize_anchor_text(item[2].anchor_identifier)
            == _normalize_anchor_text(winner_candidate.anchor_identifier)
            for item in valid_candidates
        ):
            raise ValueError(
                "scoped procedure list closure requires exactly one catalog-wide candidate"
            )
        blocked_candidates = [
            item
            for item in matched_candidates
            if item[2].blocked_reason is not None
            and _scoped_procedure_candidate_rank(item[2]) >= best_rank
        ]
        if blocked_candidates:
            raise ValueError(
                "scoped procedure list has an equally or more specific blocked candidate"
            )
        authority_index, authority, canonical_candidate, packed_candidate = winners[0]
        canonical_block = canonical_candidate.block
        packed_block = packed_candidate.block
        canonical_structure = _scoped_procedure_structure_payload(canonical_candidate)
        list_structure_hash = stable_content_hash(canonical_structure)

        member_witnesses: list[EvidenceWitness] = []
        for canonical_item, packed_item in zip(
            canonical_block.items,
            packed_block.items,
            strict=True,
        ):
            matches = [
                witness
                for witness in typed_witnesses
                if witness.atom_kind == WitnessAtomKind.LIST_ITEM
                and witness.span_ref.evidence_id == authority.evidence_id
                and witness.span_ref.representation_id == authority.representation_id
                and witness.span_ref.canonical_start == canonical_item.content_start
                and witness.span_ref.canonical_end == canonical_item.content_end
                and witness.span_ref.packed_start == packed_item.content_start
                and witness.span_ref.packed_end == packed_item.content_end
            ]
            if len(matches) != 1:
                raise ValueError("scoped procedure list must bind every source item witness once")
            member_witnesses.append(matches[0])
        typed_members = tuple(member_witnesses)
        witness_ordinals = tuple(item.ordinal for item in typed_members)
        if witness_ordinals != tuple(sorted(set(witness_ordinals))):
            raise ValueError("scoped procedure witnesses must be unique and source ordered")
        for witness in typed_members:
            try:
                catalog_index = catalog.witness_ids.index(witness.witness_id)
            except ValueError as exc:
                raise ValueError("scoped procedure witness is outside the catalog") from exc
            if witness.ordinal != catalog_index:
                raise ValueError("scoped procedure witness ordinal does not match the catalog")
            if (
                witness.context_pack_hash != catalog.context_pack_hash
                or witness.authorized_scope_hash != catalog.authorized_scope_hash
            ):
                raise ValueError("scoped procedure witness scope differs from the catalog")
            witness.span_ref.validate_artifacts(
                evidence_id=authority.evidence_id,
                representation_id=authority.representation_id,
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )

        canonical_outside = (
            authority.canonical_text[: canonical_block.start],
            authority.canonical_text[canonical_block.end :],
        )
        packed_outside = (
            authority.packed_text[: packed_block.start],
            authority.packed_text[packed_block.end :],
        )
        catalog_authority_payload = {
            "evidence_ids": list(evidence_ids),
            "representation_ids": list(representation_ids),
            "canonical_hashes": list(canonical_hashes),
            "packed_hashes": list(packed_hashes),
        }
        payload: dict[str, Any] = {
            "schema_version": QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION,
            "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
            "question_contract_id": question.contract_id,
            "question_contract_hash": question.contract_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "obligation_id": obligation.obligation_id,
            "obligation_hash": obligation.obligation_hash,
            "witness_ids": tuple(item.witness_id for item in typed_members),
            "witness_hashes": tuple(item.witness_hash for item in typed_members),
            "witness_ordinals": witness_ordinals,
            "span_ref_hashes": tuple(item.span_ref.span_ref_hash for item in typed_members),
            "evidence_id": authority.evidence_id,
            "evidence_content_hash": canonical_hashes[authority_index],
            "representation_id": authority.representation_id,
            "canonical_item_hashes": tuple(
                sha256_digest(authority.canonical_text[item.content_start : item.content_end])
                for item in canonical_block.items
            ),
            "packed_item_hashes": tuple(
                sha256_digest(authority.packed_text[item.content_start : item.content_end])
                for item in packed_block.items
            ),
            "canonical_block_hash": sha256_digest(
                authority.canonical_text[canonical_block.start : canonical_block.end]
            ),
            "packed_block_hash": sha256_digest(
                authority.packed_text[packed_block.start : packed_block.end]
            ),
            "canonical_title_hash": sha256_digest(
                authority.canonical_text[
                    canonical_block.title_content_start : canonical_block.title_content_end
                ]
            ),
            "packed_title_hash": sha256_digest(
                authority.packed_text[
                    packed_block.title_content_start : packed_block.title_content_end
                ]
            ),
            "list_structure_hash": list_structure_hash,
            "canonical_block_start": canonical_block.start,
            "canonical_block_end": canonical_block.end,
            "packed_block_start": packed_block.start,
            "packed_block_end": packed_block.end,
            "title_line_index": canonical_block.title.index,
            "first_item_line_index": canonical_block.items[0].line.index,
            "item_count": len(typed_members),
            "item_ordinals": tuple(item.ordinal for item in canonical_block.items),
            "marker_style": canonical_block.marker_style,
            "title_anchor_hash": sha256_digest(canonical_candidate.anchor_identifier),
            "anchor_field": canonical_candidate.anchor_field,
            "catalog_representation_count": len(authorities),
            "canonical_non_block_character_count": sum(
                not character.isspace() for part in canonical_outside for character in part
            ),
            "packed_non_block_character_count": sum(
                not character.isspace() for part in packed_outside for character in part
            ),
            "catalog_evidence_ids": evidence_ids,
            "catalog_representation_ids": representation_ids,
            "catalog_evidence_content_hashes": canonical_hashes,
            "catalog_packed_text_hashes": packed_hashes,
            "catalog_authority_hash": stable_content_hash(catalog_authority_payload),
            "selected_authority_index": authority_index,
            "authority_candidate_count": 1,
            "canonical_outside_scope_hash": stable_content_hash(canonical_outside),
            "packed_outside_scope_hash": stable_content_hash(packed_outside),
            "normalized_anchor_hash": sha256_digest(canonical_block.normalized_title),
            "canonical_anchor_start": canonical_block.title_content_start,
            "canonical_anchor_end": canonical_block.title_content_end,
            "packed_anchor_start": packed_block.title_content_start,
            "packed_anchor_end": packed_block.title_content_end,
            "canonical_item_starts": tuple(item.content_start for item in canonical_block.items),
            "canonical_item_ends": tuple(item.content_end for item in canonical_block.items),
            "packed_item_starts": tuple(item.content_start for item in packed_block.items),
            "packed_item_ends": tuple(item.content_end for item in packed_block.items),
            "selection_kind": canonical_candidate.selection_kind,
            "anchor_match_key_hashes": tuple(
                sha256_digest(item) for item in canonical_candidate.anchor_match_keys
            ),
            "prefix_trigger_hash": canonical_candidate.prefix_trigger_hash,
            "excluded_successor_ordinal": (canonical_candidate.excluded_successor_ordinal),
            "excluded_successor_hash": canonical_candidate.excluded_successor_hash,
            "boundary_rule_version": canonical_candidate.boundary_rule_version,
        }
        return cls(
            **payload,
            receipt_id=stable_identifier("qplc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedListAuthorityRepresentation],
    ) -> Self:
        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            authority_representations=authority_representations,
        )
        if rebuilt != self:
            raise ValueError(
                "scoped procedure list closure receipt does not match supplied artifacts"
            )
        return self


class _QueryAnchoredScopedTableSelectionClosureReceiptFields(AnsweringContractModel):
    """证明完整 authority 中由显式 cell selector 唯一选出的全部表格行。"""

    schema_version: Literal["query-anchored-scoped-table-selection-closure-receipt.v2"] = (
        QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-default-ignorable-markup-aware-token.v3"] = (
        QUERY_ANCHOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qtsc_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    witness_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    witness_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    witness_ordinals: tuple[int, ...] = Field(min_length=1, max_length=64)
    span_ref_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    catalog_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_evidence_content_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_packed_text_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_count: int = Field(ge=1, le=256)
    catalog_authority_hash: str = Field(pattern=SHA256_PATTERN)
    selected_authority_index: int = Field(ge=0, le=255)
    selected_table_index: int = Field(ge=0)
    authority_candidate_count: Literal[1] = 1
    anchor_field: Literal["source_query", "information_need"]
    selector_column_index: int = Field(ge=0)
    selector_header_hash: str = Field(pattern=SHA256_PATTERN)
    selector_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    selector_occurrence_counts: tuple[int, ...] = Field(min_length=1, max_length=64)
    catalog_canonical_selector_cell_counts: tuple[int, ...] = Field(min_length=1, max_length=64)
    catalog_packed_selector_cell_counts: tuple[int, ...] = Field(min_length=1, max_length=64)
    selector_count: int = Field(ge=1, le=64)
    selector_values_hash: str = Field(pattern=SHA256_PATTERN)
    selected_row_indices: tuple[int, ...] = Field(min_length=1, max_length=64)
    selected_row_count: int = Field(ge=1, le=64)
    canonical_row_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    packed_row_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    canonical_row_starts: tuple[int, ...] = Field(min_length=1, max_length=64)
    canonical_row_ends: tuple[int, ...] = Field(min_length=1, max_length=64)
    packed_row_starts: tuple[int, ...] = Field(min_length=1, max_length=64)
    packed_row_ends: tuple[int, ...] = Field(min_length=1, max_length=64)
    canonical_table_hash: str = Field(pattern=SHA256_PATTERN)
    packed_table_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_header_hash: str = Field(pattern=SHA256_PATTERN)
    packed_header_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_separator_hash: str = Field(pattern=SHA256_PATTERN)
    packed_separator_hash: str = Field(pattern=SHA256_PATTERN)
    table_structure_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_table_start: int = Field(ge=0)
    canonical_table_end: int = Field(gt=0)
    packed_table_start: int = Field(ge=0)
    packed_table_end: int = Field(gt=0)
    table_line_index: int = Field(ge=0)
    table_parse_mode: Literal[
        "standard",
        "ocr_header_data_before_separator",
    ]
    table_column_count: int = Field(ge=2)
    table_data_row_count: int = Field(ge=1)
    canonical_outside_table_hash: str = Field(pattern=SHA256_PATTERN)
    packed_outside_table_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_non_table_character_count: int = Field(ge=0)
    packed_non_table_character_count: int = Field(ge=0)

    def _identity_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
            exclude={"receipt_id", "receipt_hash"},
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority_lengths = {
            self.catalog_representation_count,
            len(self.catalog_evidence_ids),
            len(self.catalog_representation_ids),
            len(self.catalog_evidence_content_hashes),
            len(self.catalog_packed_text_hashes),
        }
        if len(authority_lengths) != 1:
            raise ValueError("scoped table authority arrays must equal representation count")
        if len(self.catalog_evidence_ids) != len(set(self.catalog_evidence_ids)):
            raise ValueError("scoped table authority Evidence IDs must be unique")
        if len(self.catalog_representation_ids) != len(set(self.catalog_representation_ids)):
            raise ValueError("scoped table authority representation IDs must be unique")
        if self.selected_authority_index >= self.catalog_representation_count:
            raise ValueError("selected scoped table authority index is out of range")
        if (
            self.catalog_evidence_ids[self.selected_authority_index] != self.evidence_id
            or self.catalog_representation_ids[self.selected_authority_index]
            != self.representation_id
            or self.catalog_evidence_content_hashes[self.selected_authority_index]
            != self.evidence_content_hash
        ):
            raise ValueError("selected scoped table authority identity is inconsistent")
        selector_lengths = {
            self.selector_count,
            len(self.selector_hashes),
            len(self.selector_occurrence_counts),
            len(self.catalog_canonical_selector_cell_counts),
            len(self.catalog_packed_selector_cell_counts),
        }
        if len(selector_lengths) != 1 or any(
            count < 1 for count in self.selector_occurrence_counts
        ):
            raise ValueError("scoped table selector arrays are inconsistent")
        if (
            self.catalog_canonical_selector_cell_counts != self.selector_occurrence_counts
            or self.catalog_packed_selector_cell_counts != self.selector_occurrence_counts
        ):
            raise ValueError(
                "scoped table selectors must occur only in selected accepted-table cells"
            )
        member_lengths = {
            self.selected_row_count,
            len(self.selected_row_indices),
            len(self.witness_ids),
            len(self.witness_hashes),
            len(self.witness_ordinals),
            len(self.span_ref_hashes),
            len(self.canonical_row_hashes),
            len(self.packed_row_hashes),
            len(self.canonical_row_starts),
            len(self.canonical_row_ends),
            len(self.packed_row_starts),
            len(self.packed_row_ends),
        }
        if len(member_lengths) != 1:
            raise ValueError("scoped table member arrays must equal selected_row_count")
        if self.selector_column_index >= self.table_column_count:
            raise ValueError("scoped table selector column lies outside the table")
        if self.selected_row_indices != tuple(sorted(set(self.selected_row_indices))) or any(
            index >= self.table_data_row_count for index in self.selected_row_indices
        ):
            raise ValueError("scoped table selected row indices are invalid")
        if self.witness_ordinals != tuple(sorted(set(self.witness_ordinals))):
            raise ValueError("scoped table witness ordinals must be unique and source ordered")
        if len(self.witness_ids) != len(set(self.witness_ids)):
            raise ValueError("scoped table witness IDs must be unique")
        if self.canonical_table_end <= self.canonical_table_start or (
            self.packed_table_end <= self.packed_table_start
        ):
            raise ValueError("scoped table spans are invalid")
        if any(
            start < self.canonical_table_start or end > self.canonical_table_end or end <= start
            for start, end in zip(
                self.canonical_row_starts,
                self.canonical_row_ends,
                strict=True,
            )
        ) or any(
            start < self.packed_table_start or end > self.packed_table_end or end <= start
            for start, end in zip(
                self.packed_row_starts,
                self.packed_row_ends,
                strict=True,
            )
        ):
            raise ValueError("scoped table member span lies outside the selected table")
        authority_payload = {
            "evidence_ids": list(self.catalog_evidence_ids),
            "representation_ids": list(self.catalog_representation_ids),
            "canonical_hashes": list(self.catalog_evidence_content_hashes),
            "packed_hashes": list(self.catalog_packed_text_hashes),
        }
        if self.catalog_authority_hash != stable_content_hash(authority_payload):
            raise ValueError("catalog_authority_hash does not match scoped table authority")
        if self.selector_values_hash != stable_content_hash(list(self.selector_hashes)):
            raise ValueError("selector_values_hash does not match scoped table selectors")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match scoped table-selection receipt")
        if self.receipt_id != stable_identifier("qtsc_", payload):
            raise ValueError("receipt_id does not match scoped table-selection receipt")
        return self


class QueryAnchoredScopedBulletListClosureReceipt(AnsweringContractModel):
    """证明完整多 representation authority 中唯一、显式有界的顶层 bullet 列表。"""

    schema_version: Literal["query-anchored-scoped-bullet-list-closure-receipt.v1"] = (
        QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-default-ignorable-markup-aware-token.v3"] = (
        QUERY_ANCHOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qblc_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    witness_ids: tuple[str, ...] = Field(min_length=2, max_length=64)
    witness_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    witness_ordinals: tuple[int, ...] = Field(min_length=2, max_length=64)
    span_ref_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    catalog_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_evidence_content_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_packed_text_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_count: int = Field(ge=1, le=256)
    catalog_authority_hash: str = Field(pattern=SHA256_PATTERN)
    selected_authority_index: int = Field(ge=0, le=255)
    selected_block_index: int = Field(ge=0)
    authority_candidate_count: Literal[1] = 1
    anchor_field: Literal["source_query", "information_need"]
    anchor_kind: Literal[
        "title_closed_intro",
        "ordered_closed_intro",
        "prose_closed_intro",
        "bounded_title",
    ]
    anchor_match_key_hashes: tuple[str, ...] = Field(min_length=2, max_length=256)
    anchor_match_key_count: int = Field(ge=2, le=256)
    anchor_match_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_anchor_hash: str = Field(pattern=SHA256_PATTERN)
    packed_anchor_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_anchor_start: int = Field(ge=0)
    canonical_anchor_end: int = Field(gt=0)
    packed_anchor_start: int = Field(ge=0)
    packed_anchor_end: int = Field(gt=0)
    canonical_anchor_line_index: int = Field(ge=0)
    packed_anchor_line_index: int = Field(ge=0)
    canonical_title_hash: str | None = None
    packed_title_hash: str | None = None
    canonical_title_start: int | None = Field(default=None, ge=0)
    canonical_title_end: int | None = Field(default=None, gt=0)
    packed_title_start: int | None = Field(default=None, ge=0)
    packed_title_end: int | None = Field(default=None, gt=0)
    marker_style: str = Field(pattern=r"^bullet:[-*•+]$")
    item_count: int = Field(ge=2, le=64)
    canonical_item_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    packed_item_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    canonical_item_body_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    packed_item_body_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    canonical_item_starts: tuple[int, ...] = Field(min_length=2, max_length=64)
    canonical_item_ends: tuple[int, ...] = Field(min_length=2, max_length=64)
    packed_item_starts: tuple[int, ...] = Field(min_length=2, max_length=64)
    packed_item_ends: tuple[int, ...] = Field(min_length=2, max_length=64)
    canonical_scope_hash: str = Field(pattern=SHA256_PATTERN)
    packed_scope_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_scope_start: int = Field(ge=0)
    canonical_scope_end: int = Field(gt=0)
    packed_scope_start: int = Field(ge=0)
    packed_scope_end: int = Field(gt=0)
    list_structure_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_local_section_hash: str = Field(pattern=SHA256_PATTERN)
    packed_local_section_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_local_section_start: int = Field(ge=0)
    canonical_local_section_end: int = Field(gt=0)
    packed_local_section_start: int = Field(ge=0)
    packed_local_section_end: int = Field(gt=0)
    canonical_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    packed_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_non_scope_character_count: int = Field(ge=0)
    packed_non_scope_character_count: int = Field(ge=0)

    def _identity_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
            exclude={"receipt_id", "receipt_hash"},
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        authority_lengths = {
            self.catalog_representation_count,
            len(self.catalog_evidence_ids),
            len(self.catalog_representation_ids),
            len(self.catalog_evidence_content_hashes),
            len(self.catalog_packed_text_hashes),
        }
        if len(authority_lengths) != 1:
            raise ValueError("scoped bullet authority arrays must equal representation count")
        if len(self.catalog_evidence_ids) != len(set(self.catalog_evidence_ids)):
            raise ValueError("scoped bullet authority Evidence IDs must be unique")
        if len(self.catalog_representation_ids) != len(set(self.catalog_representation_ids)):
            raise ValueError("scoped bullet authority representation IDs must be unique")
        if self.selected_authority_index >= self.catalog_representation_count:
            raise ValueError("selected scoped bullet authority index is out of range")
        if (
            self.catalog_evidence_ids[self.selected_authority_index] != self.evidence_id
            or self.catalog_representation_ids[self.selected_authority_index]
            != self.representation_id
            or self.catalog_evidence_content_hashes[self.selected_authority_index]
            != self.evidence_content_hash
        ):
            raise ValueError("selected scoped bullet authority identity is inconsistent")
        member_lengths = {
            self.item_count,
            len(self.witness_ids),
            len(self.witness_hashes),
            len(self.witness_ordinals),
            len(self.span_ref_hashes),
            len(self.canonical_item_hashes),
            len(self.packed_item_hashes),
            len(self.canonical_item_body_hashes),
            len(self.packed_item_body_hashes),
            len(self.canonical_item_starts),
            len(self.canonical_item_ends),
            len(self.packed_item_starts),
            len(self.packed_item_ends),
        }
        if len(member_lengths) != 1:
            raise ValueError("scoped bullet member arrays must equal item_count")
        if len(self.witness_ids) != len(set(self.witness_ids)):
            raise ValueError("scoped bullet witness IDs must be unique")
        if self.witness_ordinals != tuple(sorted(set(self.witness_ordinals))):
            raise ValueError("scoped bullet witnesses must be unique and source ordered")
        if self.anchor_match_key_count != len(self.anchor_match_key_hashes):
            raise ValueError("scoped bullet anchor match arrays are inconsistent")
        if self.anchor_match_hash != stable_content_hash(list(self.anchor_match_key_hashes)):
            raise ValueError("anchor_match_hash does not match scoped bullet anchor")
        title_values = (
            self.canonical_title_hash,
            self.packed_title_hash,
            self.canonical_title_start,
            self.canonical_title_end,
            self.packed_title_start,
            self.packed_title_end,
        )
        if any(value is None for value in title_values) and not all(
            value is None for value in title_values
        ):
            raise ValueError("scoped bullet title fields must be all present or all absent")
        for title_hash in (self.canonical_title_hash, self.packed_title_hash):
            if title_hash is not None and re.fullmatch(r"sha256:[0-9a-f]{64}", title_hash) is None:
                raise ValueError("scoped bullet title hash is invalid")
        if (
            self.canonical_scope_start != self.canonical_anchor_start
            or self.packed_scope_start != self.packed_anchor_start
            or self.canonical_scope_end <= self.canonical_scope_start
            or self.packed_scope_end <= self.packed_scope_start
        ):
            raise ValueError("scoped bullet scope and anchor spans are inconsistent")
        if self.canonical_anchor_end <= self.canonical_anchor_start or (
            self.packed_anchor_end <= self.packed_anchor_start
        ):
            raise ValueError("scoped bullet anchor spans are invalid")
        if any(
            start < self.canonical_anchor_end or end > self.canonical_scope_end or end <= start
            for start, end in zip(
                self.canonical_item_starts,
                self.canonical_item_ends,
                strict=True,
            )
        ) or any(
            start < self.packed_anchor_end or end > self.packed_scope_end or end <= start
            for start, end in zip(
                self.packed_item_starts,
                self.packed_item_ends,
                strict=True,
            )
        ):
            raise ValueError("scoped bullet item span lies outside the selected scope")
        if self.canonical_item_starts != tuple(sorted(self.canonical_item_starts)) or (
            self.packed_item_starts != tuple(sorted(self.packed_item_starts))
        ):
            raise ValueError("scoped bullet item spans must be source ordered")
        if not (
            self.canonical_local_section_start
            <= self.canonical_scope_start
            < self.canonical_scope_end
            <= self.canonical_local_section_end
        ) or not (
            self.packed_local_section_start
            <= self.packed_scope_start
            < self.packed_scope_end
            <= self.packed_local_section_end
        ):
            raise ValueError("scoped bullet scope lies outside its local section")
        authority_payload = {
            "evidence_ids": list(self.catalog_evidence_ids),
            "representation_ids": list(self.catalog_representation_ids),
            "canonical_hashes": list(self.catalog_evidence_content_hashes),
            "packed_hashes": list(self.catalog_packed_text_hashes),
        }
        if self.catalog_authority_hash != stable_content_hash(authority_payload):
            raise ValueError("catalog_authority_hash does not match scoped bullet authority")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match scoped bullet-list receipt")
        if self.receipt_id != stable_identifier("qblc_", payload):
            raise ValueError("receipt_id does not match scoped bullet-list receipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedBulletAuthorityRepresentation],
    ) -> Self:
        """从完整 raw authority 重扫并绑定唯一的 Query-anchored bullet 列表。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        authorities = tuple(
            ScopedBulletAuthorityRepresentation(
                evidence_id=item.evidence_id,
                representation_id=item.representation_id,
                canonical_text=item.canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority_representations
        )
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("scoped bullet-list requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("witness catalog belongs to a different question contract")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("obligation does not belong to the question contract")
        if (
            obligation.closure != FacetClosure.OPEN_ENUMERATION
            or obligation.required_output_form != RequiredOutputForm.LIST
        ):
            raise ValueError("scoped bullet-list requires an open list obligation")
        if not authorities:
            raise ValueError("scoped bullet-list requires complete raw authority")
        evidence_ids = tuple(item.evidence_id for item in authorities)
        representation_ids = tuple(item.representation_id for item in authorities)
        if len(evidence_ids) != len(set(evidence_ids)) or len(representation_ids) != len(
            set(representation_ids)
        ):
            raise ValueError("scoped bullet-list authority identities must be unique")
        canonical_hashes = tuple(sha256_digest(item.canonical_text) for item in authorities)
        packed_hashes = tuple(sha256_digest(item.packed_text) for item in authorities)

        block_pairs: list[tuple[int, int, _BulletListBlock, _BulletListBlock, str, str]] = []
        for authority_index, authority in enumerate(authorities):
            canonical_blocks = _locate_all_bullet_list_blocks(authority.canonical_text)
            packed_blocks = _locate_all_bullet_list_blocks(authority.packed_text)
            if len(canonical_blocks) != len(packed_blocks):
                raise ValueError("canonical and packed scoped bullet-list counts differ")
            for block_index, (canonical_block, packed_block) in enumerate(
                zip(canonical_blocks, packed_blocks, strict=True)
            ):
                if _bullet_list_structure_payload(canonical_block) != (
                    _bullet_list_structure_payload(packed_block)
                ):
                    raise ValueError("canonical and packed scoped bullet-list structures differ")
                block_pairs.append(
                    (
                        authority_index,
                        block_index,
                        canonical_block,
                        packed_block,
                        authority.canonical_text,
                        authority.packed_text,
                    )
                )

        information_candidates = _scoped_bullet_candidates_for_anchor(
            block_pairs=block_pairs,
            anchor=obligation.information_need,
            anchor_field="information_need",
        )
        if len(information_candidates) == 1:
            candidate = information_candidates[0]
            selected_anchor = obligation.information_need
        else:
            source_candidates = _scoped_bullet_candidates_for_anchor(
                block_pairs=block_pairs,
                anchor=question.source_query,
                anchor_field="source_query",
            )
            if len(source_candidates) != 1:
                raise ValueError("scoped bullet-list requires exactly one catalog-wide candidate")
            candidate = source_candidates[0]
            selected_anchor = question.source_query
        _reject_catalog_bullet_supplements(
            authorities=authorities,
            candidate=candidate,
            anchor=selected_anchor,
        )

        authority = authorities[candidate.authority_index]
        canonical_block = candidate.canonical_block
        packed_block = candidate.packed_block
        member_witnesses: list[EvidenceWitness] = []
        for canonical_item, packed_item in zip(
            canonical_block.items,
            packed_block.items,
            strict=True,
        ):
            matches = [
                witness
                for witness in typed_witnesses
                if witness.atom_kind == WitnessAtomKind.LIST_ITEM
                and witness.span_ref.evidence_id == authority.evidence_id
                and witness.span_ref.representation_id == authority.representation_id
                and witness.span_ref.canonical_start == canonical_item.content_start
                and witness.span_ref.canonical_end == canonical_item.content_end
                and witness.span_ref.packed_start == packed_item.content_start
                and witness.span_ref.packed_end == packed_item.content_end
            ]
            if len(matches) != 1:
                raise ValueError("scoped bullet-list must bind every source item witness once")
            member_witnesses.append(matches[0])
        typed_members = tuple(member_witnesses)
        for witness in typed_members:
            try:
                catalog_index = catalog.witness_ids.index(witness.witness_id)
            except ValueError as exc:
                raise ValueError("scoped bullet-list witness is outside the catalog") from exc
            if witness.ordinal != catalog_index:
                raise ValueError("scoped bullet-list witness ordinal does not match the catalog")
            if (
                witness.context_pack_hash != catalog.context_pack_hash
                or witness.authorized_scope_hash != catalog.authorized_scope_hash
            ):
                raise ValueError("scoped bullet-list witness scope differs from the catalog")
            witness.span_ref.validate_artifacts(
                evidence_id=authority.evidence_id,
                representation_id=authority.representation_id,
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )

        canonical_title = (
            authority.canonical_text[
                canonical_block.title_content_start : canonical_block.title_content_end
            ]
            if canonical_block.title_content_start is not None
            and canonical_block.title_content_end is not None
            else None
        )
        packed_title = (
            authority.packed_text[packed_block.title_content_start : packed_block.title_content_end]
            if packed_block.title_content_start is not None
            and packed_block.title_content_end is not None
            else None
        )
        anchor_match_key_hashes = tuple(
            sha256_digest(value) for value in candidate.anchor_match_keys
        )
        canonical_outside = (
            authority.canonical_text[: canonical_block.start],
            authority.canonical_text[canonical_block.end :],
        )
        packed_outside = (
            authority.packed_text[: packed_block.start],
            authority.packed_text[packed_block.end :],
        )
        authority_payload = {
            "evidence_ids": list(evidence_ids),
            "representation_ids": list(representation_ids),
            "canonical_hashes": list(canonical_hashes),
            "packed_hashes": list(packed_hashes),
        }
        payload: dict[str, Any] = {
            "schema_version": QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION,
            "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
            "question_contract_id": question.contract_id,
            "question_contract_hash": question.contract_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "obligation_id": obligation.obligation_id,
            "obligation_hash": obligation.obligation_hash,
            "witness_ids": tuple(item.witness_id for item in typed_members),
            "witness_hashes": tuple(item.witness_hash for item in typed_members),
            "witness_ordinals": tuple(item.ordinal for item in typed_members),
            "span_ref_hashes": tuple(item.span_ref.span_ref_hash for item in typed_members),
            "evidence_id": authority.evidence_id,
            "evidence_content_hash": canonical_hashes[candidate.authority_index],
            "representation_id": authority.representation_id,
            "catalog_evidence_ids": evidence_ids,
            "catalog_representation_ids": representation_ids,
            "catalog_evidence_content_hashes": canonical_hashes,
            "catalog_packed_text_hashes": packed_hashes,
            "catalog_representation_count": len(authorities),
            "catalog_authority_hash": stable_content_hash(authority_payload),
            "selected_authority_index": candidate.authority_index,
            "selected_block_index": candidate.block_index,
            "authority_candidate_count": 1,
            "anchor_field": candidate.anchor_field,
            "anchor_kind": canonical_block.anchor_kind,
            "anchor_match_key_hashes": anchor_match_key_hashes,
            "anchor_match_key_count": len(anchor_match_key_hashes),
            "anchor_match_hash": stable_content_hash(list(anchor_match_key_hashes)),
            "canonical_anchor_hash": sha256_digest(
                authority.canonical_text[
                    canonical_block.anchor_content_start : canonical_block.anchor_content_end
                ]
            ),
            "packed_anchor_hash": sha256_digest(
                authority.packed_text[
                    packed_block.anchor_content_start : packed_block.anchor_content_end
                ]
            ),
            "canonical_anchor_start": canonical_block.anchor_content_start,
            "canonical_anchor_end": canonical_block.anchor_content_end,
            "packed_anchor_start": packed_block.anchor_content_start,
            "packed_anchor_end": packed_block.anchor_content_end,
            "canonical_anchor_line_index": canonical_block.anchor.index,
            "packed_anchor_line_index": packed_block.anchor.index,
            "canonical_title_hash": (
                sha256_digest(canonical_title) if canonical_title is not None else None
            ),
            "packed_title_hash": sha256_digest(packed_title) if packed_title is not None else None,
            "canonical_title_start": canonical_block.title_content_start,
            "canonical_title_end": canonical_block.title_content_end,
            "packed_title_start": packed_block.title_content_start,
            "packed_title_end": packed_block.title_content_end,
            "marker_style": canonical_block.marker_style,
            "item_count": len(canonical_block.items),
            "canonical_item_hashes": tuple(
                sha256_digest(authority.canonical_text[item.content_start : item.content_end])
                for item in canonical_block.items
            ),
            "packed_item_hashes": tuple(
                sha256_digest(authority.packed_text[item.content_start : item.content_end])
                for item in packed_block.items
            ),
            "canonical_item_body_hashes": tuple(
                sha256_digest(_normalize_table_cell(item.body)) for item in canonical_block.items
            ),
            "packed_item_body_hashes": tuple(
                sha256_digest(_normalize_table_cell(item.body)) for item in packed_block.items
            ),
            "canonical_item_starts": tuple(item.content_start for item in canonical_block.items),
            "canonical_item_ends": tuple(item.content_end for item in canonical_block.items),
            "packed_item_starts": tuple(item.content_start for item in packed_block.items),
            "packed_item_ends": tuple(item.content_end for item in packed_block.items),
            "canonical_scope_hash": sha256_digest(
                authority.canonical_text[canonical_block.start : canonical_block.end]
            ),
            "packed_scope_hash": sha256_digest(
                authority.packed_text[packed_block.start : packed_block.end]
            ),
            "canonical_scope_start": canonical_block.start,
            "canonical_scope_end": canonical_block.end,
            "packed_scope_start": packed_block.start,
            "packed_scope_end": packed_block.end,
            "list_structure_hash": stable_content_hash(
                _bullet_list_structure_payload(canonical_block)
            ),
            "canonical_local_section_hash": sha256_digest(
                authority.canonical_text[
                    canonical_block.local_section_start : canonical_block.local_section_end
                ]
            ),
            "packed_local_section_hash": sha256_digest(
                authority.packed_text[
                    packed_block.local_section_start : packed_block.local_section_end
                ]
            ),
            "canonical_local_section_start": canonical_block.local_section_start,
            "canonical_local_section_end": canonical_block.local_section_end,
            "packed_local_section_start": packed_block.local_section_start,
            "packed_local_section_end": packed_block.local_section_end,
            "canonical_outside_scope_hash": stable_content_hash(canonical_outside),
            "packed_outside_scope_hash": stable_content_hash(packed_outside),
            "canonical_non_scope_character_count": sum(
                not character.isspace() for part in canonical_outside for character in part
            ),
            "packed_non_scope_character_count": sum(
                not character.isspace() for part in packed_outside for character in part
            ),
        }
        return cls(
            **payload,
            receipt_id=stable_identifier("qblc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedBulletAuthorityRepresentation],
    ) -> Self:
        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            authority_representations=authority_representations,
        )
        if rebuilt != self:
            raise ValueError("scoped bullet-list closure receipt does not match supplied artifacts")
        return self


class QueryAnchoredScopedTableSelectionClosureReceipt(
    _QueryAnchoredScopedTableSelectionClosureReceiptFields
):
    """证明标准或严格 OCR 表中唯一 selector 对应的全部 data rows。"""

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedTableAuthorityRepresentation],
    ) -> Self:
        """从完整多 representation authority 重扫显式 cell selector 与全部匹配行。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        authorities = tuple(
            ScopedTableAuthorityRepresentation(
                evidence_id=item.evidence_id,
                representation_id=item.representation_id,
                canonical_text=item.canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority_representations
        )
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("scoped table-selection requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("witness catalog belongs to a different question contract")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("obligation does not belong to the question contract")
        if (
            obligation.closure != FacetClosure.OPEN_ENUMERATION
            or obligation.required_output_form != RequiredOutputForm.LIST
        ):
            raise ValueError("scoped table-selection requires an open list obligation")
        if not authorities:
            raise ValueError("scoped table-selection requires complete raw authority")
        evidence_ids = tuple(item.evidence_id for item in authorities)
        representation_ids = tuple(item.representation_id for item in authorities)
        if len(evidence_ids) != len(set(evidence_ids)) or len(representation_ids) != len(
            set(representation_ids)
        ):
            raise ValueError("scoped table-selection authority identities must be unique")
        canonical_hashes = tuple(sha256_digest(item.canonical_text) for item in authorities)
        packed_hashes = tuple(sha256_digest(item.packed_text) for item in authorities)
        table_pairs: list[tuple[int, int, MarkdownTableBlock, MarkdownTableBlock]] = []
        for authority_index, authority in enumerate(authorities):
            canonical_tables = scan_markdown_table_blocks(authority.canonical_text)
            packed_tables = scan_markdown_table_blocks(authority.packed_text)
            if len(canonical_tables) != len(packed_tables):
                raise ValueError("canonical and packed scoped table counts differ")
            for table_index, (canonical_table, packed_table) in enumerate(
                zip(canonical_tables, packed_tables, strict=True)
            ):
                canonical_structure = _scoped_table_structure_payload(canonical_table)
                if canonical_structure != _scoped_table_structure_payload(packed_table):
                    raise ValueError("canonical and packed scoped table structures differ")
                table_pairs.append(
                    (
                        authority_index,
                        table_index,
                        canonical_table,
                        packed_table,
                    )
                )

        information_candidates = _scoped_table_candidates_for_anchor(
            table_pairs=table_pairs,
            anchor=obligation.information_need,
            anchor_field="information_need",
        )
        candidate = _unique_most_specific_table_candidate(information_candidates)
        if candidate is None:
            source_candidates = _scoped_table_candidates_for_anchor(
                table_pairs=table_pairs,
                anchor=question.source_query,
                anchor_field="source_query",
            )
            candidate = _unique_most_specific_table_candidate(source_candidates)
            if candidate is None:
                raise ValueError(
                    "scoped table-selection requires exactly one catalog-wide selector candidate"
                )

        authority = authorities[candidate.authority_index]
        canonical_table = candidate.canonical_table
        packed_table = candidate.packed_table
        _validate_catalog_table_selector_supplements(
            texts=tuple(item.canonical_text for item in authorities),
            selector_values=candidate.selector_values,
        )
        _validate_catalog_table_selector_supplements(
            texts=tuple(item.packed_text for item in authorities),
            selector_values=candidate.selector_values,
        )
        _validate_selected_table_continuations(
            text=authority.canonical_text,
            table=canonical_table,
            selector_values=candidate.selector_values,
        )
        _validate_selected_table_continuations(
            text=authority.packed_text,
            table=packed_table,
            selector_values=candidate.selector_values,
        )
        member_witnesses: list[EvidenceWitness] = []
        for row_index in candidate.selected_row_indices:
            canonical_row = canonical_table.data_rows[row_index]
            packed_row = packed_table.data_rows[row_index]
            matches = [
                witness
                for witness in typed_witnesses
                if witness.atom_kind == WitnessAtomKind.TABLE_DATA_ROW
                and witness.span_ref.evidence_id == authority.evidence_id
                and witness.span_ref.representation_id == authority.representation_id
                and witness.span_ref.canonical_start == canonical_row.line.start
                and witness.span_ref.canonical_end == canonical_row.line.end
                and witness.span_ref.packed_start == packed_row.line.start
                and witness.span_ref.packed_end == packed_row.line.end
            ]
            if len(matches) != 1:
                raise ValueError("scoped table-selection must bind every selected row witness once")
            member_witnesses.append(matches[0])
        typed_members = tuple(member_witnesses)
        for witness in typed_members:
            try:
                catalog_index = catalog.witness_ids.index(witness.witness_id)
            except ValueError as exc:
                raise ValueError("scoped table-selection witness is outside the catalog") from exc
            if witness.ordinal != catalog_index:
                raise ValueError(
                    "scoped table-selection witness ordinal does not match the catalog"
                )
            if (
                witness.context_pack_hash != catalog.context_pack_hash
                or witness.authorized_scope_hash != catalog.authorized_scope_hash
            ):
                raise ValueError("scoped table-selection witness scope differs from the catalog")
            witness.span_ref.validate_artifacts(
                evidence_id=authority.evidence_id,
                representation_id=authority.representation_id,
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )

        selector_hashes = tuple(sha256_digest(value) for value in candidate.selector_values)
        selector_occurrence_counts = tuple(
            sum(
                _normalize_table_cell(row.cells[candidate.selector_column_index].text) == value
                for row in canonical_table.data_rows
            )
            for value in candidate.selector_values
        )
        catalog_canonical_selector_cell_counts = _catalog_selector_cell_counts(
            selector_values=candidate.selector_values,
            tables=tuple(item[2] for item in table_pairs),
        )
        catalog_packed_selector_cell_counts = _catalog_selector_cell_counts(
            selector_values=candidate.selector_values,
            tables=tuple(item[3] for item in table_pairs),
        )
        if (
            catalog_canonical_selector_cell_counts != selector_occurrence_counts
            or catalog_packed_selector_cell_counts != selector_occurrence_counts
        ):
            raise ValueError("scoped table selector exists outside selected accepted-table cells")
        canonical_outside = (
            authority.canonical_text[: canonical_table.start],
            authority.canonical_text[canonical_table.end :],
        )
        packed_outside = (
            authority.packed_text[: packed_table.start],
            authority.packed_text[packed_table.end :],
        )
        authority_payload = {
            "evidence_ids": list(evidence_ids),
            "representation_ids": list(representation_ids),
            "canonical_hashes": list(canonical_hashes),
            "packed_hashes": list(packed_hashes),
        }
        payload: dict[str, Any] = {
            "schema_version": QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION,
            "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
            "question_contract_id": question.contract_id,
            "question_contract_hash": question.contract_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "obligation_id": obligation.obligation_id,
            "obligation_hash": obligation.obligation_hash,
            "witness_ids": tuple(item.witness_id for item in typed_members),
            "witness_hashes": tuple(item.witness_hash for item in typed_members),
            "witness_ordinals": tuple(item.ordinal for item in typed_members),
            "span_ref_hashes": tuple(item.span_ref.span_ref_hash for item in typed_members),
            "evidence_id": authority.evidence_id,
            "evidence_content_hash": canonical_hashes[candidate.authority_index],
            "representation_id": authority.representation_id,
            "catalog_evidence_ids": evidence_ids,
            "catalog_representation_ids": representation_ids,
            "catalog_evidence_content_hashes": canonical_hashes,
            "catalog_packed_text_hashes": packed_hashes,
            "catalog_representation_count": len(authorities),
            "catalog_authority_hash": stable_content_hash(authority_payload),
            "selected_authority_index": candidate.authority_index,
            "selected_table_index": candidate.table_index,
            "authority_candidate_count": 1,
            "anchor_field": candidate.anchor_field,
            "selector_column_index": candidate.selector_column_index,
            "selector_header_hash": sha256_digest(
                _normalize_table_cell(
                    canonical_table.header.cells[candidate.selector_column_index].text
                )
            ),
            "selector_hashes": selector_hashes,
            "selector_occurrence_counts": selector_occurrence_counts,
            "catalog_canonical_selector_cell_counts": (catalog_canonical_selector_cell_counts),
            "catalog_packed_selector_cell_counts": (catalog_packed_selector_cell_counts),
            "selector_count": len(selector_hashes),
            "selector_values_hash": stable_content_hash(list(selector_hashes)),
            "selected_row_indices": candidate.selected_row_indices,
            "selected_row_count": len(candidate.selected_row_indices),
            "canonical_row_hashes": tuple(
                sha256_digest(authority.canonical_text[row.line.start : row.line.end])
                for row in (
                    canonical_table.data_rows[index] for index in candidate.selected_row_indices
                )
            ),
            "packed_row_hashes": tuple(
                sha256_digest(authority.packed_text[row.line.start : row.line.end])
                for row in (
                    packed_table.data_rows[index] for index in candidate.selected_row_indices
                )
            ),
            "canonical_row_starts": tuple(
                canonical_table.data_rows[index].line.start
                for index in candidate.selected_row_indices
            ),
            "canonical_row_ends": tuple(
                canonical_table.data_rows[index].line.end
                for index in candidate.selected_row_indices
            ),
            "packed_row_starts": tuple(
                packed_table.data_rows[index].line.start for index in candidate.selected_row_indices
            ),
            "packed_row_ends": tuple(
                packed_table.data_rows[index].line.end for index in candidate.selected_row_indices
            ),
            "canonical_table_hash": sha256_digest(
                authority.canonical_text[canonical_table.start : canonical_table.end]
            ),
            "packed_table_hash": sha256_digest(
                authority.packed_text[packed_table.start : packed_table.end]
            ),
            "canonical_header_hash": sha256_digest(
                authority.canonical_text[
                    canonical_table.header.line.start : canonical_table.header.line.end
                ]
            ),
            "packed_header_hash": sha256_digest(
                authority.packed_text[packed_table.header.line.start : packed_table.header.line.end]
            ),
            "canonical_separator_hash": sha256_digest(
                authority.canonical_text[
                    canonical_table.separator.line.start : canonical_table.separator.line.end
                ]
            ),
            "packed_separator_hash": sha256_digest(
                authority.packed_text[
                    packed_table.separator.line.start : packed_table.separator.line.end
                ]
            ),
            "table_structure_hash": stable_content_hash(
                _scoped_table_structure_payload(canonical_table)
            ),
            "canonical_table_start": canonical_table.start,
            "canonical_table_end": canonical_table.end,
            "packed_table_start": packed_table.start,
            "packed_table_end": packed_table.end,
            "table_line_index": canonical_table.header.line.index,
            "table_parse_mode": canonical_table.parse_mode.value,
            "table_column_count": canonical_table.column_count,
            "table_data_row_count": len(canonical_table.data_rows),
            "canonical_outside_table_hash": stable_content_hash(canonical_outside),
            "packed_outside_table_hash": stable_content_hash(packed_outside),
            "canonical_non_table_character_count": sum(
                not character.isspace() for part in canonical_outside for character in part
            ),
            "packed_non_table_character_count": sum(
                not character.isspace() for part in packed_outside for character in part
            ),
        }
        return cls(
            **payload,
            receipt_id=stable_identifier("qtsc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedTableAuthorityRepresentation],
    ) -> Self:
        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            authority_representations=authority_representations,
        )
        if rebuilt != self:
            raise ValueError(
                "scoped table-selection closure receipt does not match supplied artifacts"
            )
        return self


@dataclass(frozen=True, slots=True)
class _InlineActionParse:
    """单个 Witness 内机械闭合的动作及可选终端结果坐标。"""

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    terminal_result_start: int | None
    terminal_result_end: int | None


@dataclass(frozen=True, slots=True)
class _InlineActionCandidate:
    authority_index: int
    witness: EvidenceWitness
    canonical_title_start: int
    canonical_title_end: int
    packed_title_start: int
    packed_title_end: int
    normalized_title: str
    canonical_parse: _InlineActionParse
    packed_parse: _InlineActionParse
    anchor_match_keys: tuple[str, ...]
    local_trigger: str | None
    score: tuple[int, int, int, int, int]


_INLINE_ACTION_SEPARATOR_RE = re.compile(r"(?:然后|随后|并且|并)|[，,、；;]")
_INLINE_ACTION_OPEN_REFERENCE_RE = re.compile(
    r"(?:参阅|详见|另见|请?查阅|参考|见第|见图|按(?:照)?上述|按(?:照)?前述)"
    r"|\b(?:see|refer\s+to)\b",
    re.IGNORECASE,
)
_INLINE_ACTION_OPEN_OR_BRANCH_RE = re.compile(
    r"(?:如果|若|否则|除非|或者|或是|二选一|视情况|根据情况)"
    r"|(?:^|[，,、；;\s])或(?=\S)",
    re.IGNORECASE,
)
_INLINE_ACTION_ANAPHOR_RE = re.compile(
    r"(?:同上|同前|续上|上述|前述|如上|见上|照此|按此|其余步骤|剩余步骤)"
    r"|(?:该|此)(?:操作|步骤|项|方式|方法)"
    r"|(?:^|[，,、；;\s])其(?:中|后|前|余|它|他|还|再|也)",
    re.IGNORECASE,
)
_INLINE_ACTION_ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…{1,}|省略)")
_INLINE_ACTION_TEMPORAL_PREFIX_RE = re.compile(
    r"^(?:在)?[^，,、；;]{1,48}(?:后|之后|以后|完成后)$",
    re.IGNORECASE,
)
_INLINE_ACTION_PURPOSE_RE = re.compile(
    r"^(?:为(?:了)?|以便|以免|从而|用于|用来|目的是|以确保)",
    re.IGNORECASE,
)
_INLINE_ACTION_TERMINAL_RESULT_RE = re.compile(
    r"^(?:(?:打印机|系统|设备|产品|机器|组件|部件|文档|窗口|页面|任务|作业)"
    r"[^，,、；;]{0,16}(?:将|会|可)?(?:自动)?"
    r"(?:恢复|打开|关闭|显示|进入|完成|开始|启动|停止|输出|返回|更新|取出|装入)"
    r"|(?:即可|从而|以便|以免|这样(?:就)?能|由此(?:可)?))",
    re.IGNORECASE,
)
_INLINE_ACTION_PASSIVE_RE = re.compile(
    r"(?:^|[，,、；;\s])(?:被|由)[^，,、；;]{0,20}"
    r"(?:打开|关闭|取出|装入|移除|恢复|完成|启动|停止)",
    re.IGNORECASE,
)
_INLINE_ACTION_PREDICATE_RE = re.compile(
    r"(?:打开|点击|选择|进入|取出|拿出|抽出|拉出|拔出|移除|装回|装入|放回|放入|"
    r"插入|合上|关闭|盖住|检查|确认|确保|按下|按住|松开|连接|断开|设置|输入|"
    r"保存|启动|停止|安装|卸下|清除|清理|等待|抬起|推入|旋转|更换|调整|勾选|"
    r"取消|执行|使用|启用|禁用|返回)",
    re.IGNORECASE,
)
_INLINE_ACTION_LOCAL_TEMPORAL_TRIGGER_RE = re.compile(
    r"(?P<trigger>(?:输入|取出|更换|完成|连接|安装|设置|配置|关闭|打开|保存|检查|清除)"
    r"[^，,。；;！？?]{0,24}(?:后|之后|以后))",
    re.IGNORECASE,
)
_INLINE_ACTION_LOCAL_CONDITIONAL_TRIGGER_RE = re.compile(
    r"(?P<trigger>(?:如果|若|当)[^，,。；;！？?]{2,48})",
    re.IGNORECASE,
)


def _inline_action_body_span(value: str) -> tuple[int, int]:
    """删除单个列表 marker 与外围空白，但保留正文中的精确坐标。"""

    if "\r" in value or "\n" in value:
        raise ValueError("inline action-chain witness must occupy one physical line")
    start = 0
    end = len(value)
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    if start >= end:
        raise ValueError("inline action-chain witness is blank")
    local = value[start:end]
    list_body = top_level_list_item_body_span(local)
    if list_body is not None:
        start += list_body[0]
        end = start + (list_body[1] - list_body[0])
    while end > start and value[end - 1] in "。！？!? ":
        end -= 1
    if start >= end:
        raise ValueError("inline action-chain witness has no body")
    return start, end


def _inline_action_clause_spans(value: str, *, start: int, end: int) -> tuple[tuple[int, int], ...]:
    """仅在顶层受限分隔符处分句，括号内容始终归属相邻动作。"""

    spans: list[tuple[int, int]] = []
    cursor = start
    position = start
    depth = 0
    while position < end:
        character = value[position]
        if character in "（(":
            depth += 1
            position += 1
            continue
        if character in "）)":
            depth -= 1
            if depth < 0:
                raise ValueError("inline action-chain witness has unbalanced parentheses")
            position += 1
            continue
        if depth:
            position += 1
            continue
        separator = _INLINE_ACTION_SEPARATOR_RE.match(value, position, end)
        if separator is None:
            position += 1
            continue
        clause_start = cursor
        clause_end = position
        while clause_start < clause_end and value[clause_start].isspace():
            clause_start += 1
        while clause_end > clause_start and value[clause_end - 1].isspace():
            clause_end -= 1
        if clause_start >= clause_end:
            raise ValueError("inline action-chain contains an empty action clause")
        spans.append((clause_start, clause_end))
        cursor = separator.end()
        position = separator.end()
    if depth:
        raise ValueError("inline action-chain witness has unbalanced parentheses")
    clause_start = cursor
    clause_end = end
    while clause_start < clause_end and value[clause_start].isspace():
        clause_start += 1
    while clause_end > clause_start and value[clause_end - 1].isspace():
        clause_end -= 1
    if clause_start >= clause_end:
        raise ValueError("inline action-chain contains an empty trailing clause")
    spans.append((clause_start, clause_end))
    return tuple(spans)


def _inline_clause_has_explicit_action(value: str) -> bool:
    """要求动作谓词出现在短前导之后，避免把对象名或结果误判为动作。"""

    if _INLINE_ACTION_PURPOSE_RE.match(value) or _INLINE_ACTION_PASSIVE_RE.search(value):
        return False
    matched = _INLINE_ACTION_PREDICATE_RE.search(value)
    if matched is None or matched.start() > 16:
        return False
    prefix = value[: matched.start()].strip()
    if prefix and re.search(r"(?:导致|使得|以致|因而|所以|结果)", prefix):
        return False
    return True


def _parse_inline_action_chain(value: str, *, absolute_start: int) -> _InlineActionParse:
    """将一个 Witness 机械拆为动作 span；结果仅作为终端残余保存。"""

    body_start, body_end = _inline_action_body_span(value)
    body = value[body_start:body_end]
    if (
        _SCOPED_OPEN_ENUMERATION_RE.search(body)
        or _INLINE_ACTION_OPEN_REFERENCE_RE.search(body)
        or _INLINE_ACTION_OPEN_OR_BRANCH_RE.search(body)
        or _INLINE_ACTION_ANAPHOR_RE.search(body)
        or _INLINE_ACTION_ELLIPSIS_RE.search(body)
    ):
        raise ValueError("inline action-chain contains an open, branched, or dependent clause")

    clauses = _inline_action_clause_spans(value, start=body_start, end=body_end)
    action_spans: list[tuple[int, int]] = []
    pending_temporal_start: int | None = None
    terminal_result: tuple[int, int] | None = None
    for index, (clause_start, clause_end) in enumerate(clauses):
        clause = value[clause_start:clause_end].strip()
        if _INLINE_ACTION_TERMINAL_RESULT_RE.match(clause):
            if index != len(clauses) - 1 or pending_temporal_start is not None:
                raise ValueError("inline action-chain result must be a terminal residual")
            terminal_result = (clause_start, clause_end)
            continue
        if _INLINE_ACTION_TEMPORAL_PREFIX_RE.fullmatch(clause):
            if pending_temporal_start is not None:
                raise ValueError("inline action-chain contains stacked temporal prefixes")
            pending_temporal_start = clause_start
            continue
        if not _inline_clause_has_explicit_action(clause):
            raise ValueError("inline action-chain clause lacks an explicit action predicate")
        resolved_start = (
            pending_temporal_start if pending_temporal_start is not None else clause_start
        )
        action_spans.append((resolved_start, clause_end))
        pending_temporal_start = None
    if pending_temporal_start is not None:
        raise ValueError("inline action-chain ends with an unowned temporal prefix")
    if len(action_spans) < 2:
        raise ValueError("inline action-chain requires at least two explicit actions")
    normalized_actions = tuple(
        _normalize_table_cell(value[start:end]) for start, end in action_spans
    )
    if any(len(item) < 3 for item in normalized_actions) or len(set(normalized_actions)) != len(
        normalized_actions
    ):
        raise ValueError("inline action-chain actions must be distinct and non-trivial")
    return _InlineActionParse(
        starts=tuple(absolute_start + start for start, _ in action_spans),
        ends=tuple(absolute_start + end for _, end in action_spans),
        terminal_result_start=(
            absolute_start + terminal_result[0] if terminal_result is not None else None
        ),
        terminal_result_end=(
            absolute_start + terminal_result[1] if terminal_result is not None else None
        ),
    )


def _inline_title_before_span(text: str, *, span_start: int) -> tuple[int, int, str]:
    """返回所选 Witness 之前最近的显式标题及其精确内容坐标。"""

    for line in reversed(_physical_lines(text)):
        if line.end > span_start or not _is_explicit_title_line(line):
            continue
        trimmed = _trimmed_physical_line(line)
        if trimmed is not None:
            return trimmed
    raise ValueError("inline action-chain candidate has no explicit owning title")


def _inline_obligation_local_trigger(
    *,
    question: QuestionContract,
    obligation: QuestionObligation,
) -> str | None:
    """多义务查询只从当前 information_need 提取局部时序/条件触发语。"""

    if len(question.obligations) <= 1:
        return None
    need = obligation.information_need
    matches = tuple(
        _normalize_anchor_text(match.group("trigger"))
        for pattern in (
            _INLINE_ACTION_LOCAL_TEMPORAL_TRIGGER_RE,
            _INLINE_ACTION_LOCAL_CONDITIONAL_TRIGGER_RE,
        )
        for match in pattern.finditer(need)
    )
    unique = tuple(dict.fromkeys(item for item in matches if item))
    has_local_marker = bool(re.search(r"(?:后|之后|以后|如果|若|当)", need, re.IGNORECASE))
    if has_local_marker and len(unique) != 1:
        raise ValueError("inline action-chain obligation has no unique local trigger")
    return unique[0] if unique else None


def _inline_candidate_score(
    *,
    title: str,
    witness_text: str,
    information_need: str,
    local_trigger: str | None,
) -> tuple[tuple[int, int, int, int, int], tuple[str, ...]] | None:
    """以 obligation-local 主题交集评分，禁止借完整 source_query 的其他 facet。"""

    normalized_witness = _normalize_anchor_text(witness_text)
    if local_trigger is not None and local_trigger not in normalized_witness:
        return None
    match_keys = _procedure_anchor_match_keys(
        f"{title}\n{witness_text}",
        information_need,
    )
    identifiers = tuple(
        item
        for item in _scoped_anchor_identifiers(title)
        if _anchor_contains_identifier(item, information_need)
    )
    if not identifiers and len(match_keys) < 3:
        return None
    longest_identifier = max((len(item) for item in identifiers), default=0)
    score = (
        int(local_trigger is not None),
        int(bool(identifiers)),
        longest_identifier,
        len(match_keys),
        sum(len(item) for item in match_keys),
    )
    keys = tuple(dict.fromkeys((*match_keys, *(f"identifier:{item}" for item in identifiers))))
    return score, keys


def _inline_action_hashes(
    text: str,
    *,
    starts: Sequence[int],
    ends: Sequence[int],
) -> tuple[str, ...]:
    return tuple(sha256_digest(text[start:end]) for start, end in zip(starts, ends, strict=True))


def _inline_authority_pair_is_proved(
    authority: ScopedListAuthorityRepresentation,
) -> bool:
    """允许可重放安全派生的标点规范化，同时拒绝任意 canonical/packed 漂移。"""

    if authority.canonical_text == authority.packed_text:
        return True
    derivation = derive_safe_representation(authority.canonical_text)
    return bool(
        not derivation.receipt.has_unproven_transform
        and not derivation.receipt.mapping_budget_exceeded
        and derivation.safe_representation == authority.packed_text
    )


def _inline_non_candidate_parse_failure(message: str) -> bool:
    """普通单动作/叙述句不是候选；结构开放或漂移仍会阻断同分候选。"""

    return any(
        marker in message
        for marker in (
            "requires at least two explicit actions",
            "clause lacks an explicit action predicate",
        )
    )


class QueryAnchoredInlineActionChainClosureReceipt(AnsweringContractModel):
    """证明唯一 Witness 内、由当前义务锚定的有序动作链已闭合。"""

    schema_version: Literal["query-anchored-inline-action-chain-closure-receipt.v1"] = (
        QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-default-ignorable-markup-aware-token.v3"] = (
        QUERY_ANCHOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qiac_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    source_query_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    information_need_hash: str = Field(pattern=SHA256_PATTERN)
    witness_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    witness_hashes: tuple[str, ...] = Field(min_length=1, max_length=1)
    witness_ordinals: tuple[int, ...] = Field(min_length=1, max_length=1)
    span_ref_hashes: tuple[str, ...] = Field(min_length=1, max_length=1)
    witness_atom_kind: Literal["sentence", "list_item"]
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    canonical_witness_start: int = Field(ge=0)
    canonical_witness_end: int = Field(gt=0)
    packed_witness_start: int = Field(ge=0)
    packed_witness_end: int = Field(gt=0)
    canonical_witness_hash: str = Field(pattern=SHA256_PATTERN)
    packed_witness_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_title_start: int = Field(ge=0)
    canonical_title_end: int = Field(gt=0)
    packed_title_start: int = Field(ge=0)
    packed_title_end: int = Field(gt=0)
    canonical_title_hash: str = Field(pattern=SHA256_PATTERN)
    packed_title_hash: str = Field(pattern=SHA256_PATTERN)
    normalized_title_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_action_starts: tuple[int, ...] = Field(min_length=2, max_length=16)
    canonical_action_ends: tuple[int, ...] = Field(min_length=2, max_length=16)
    packed_action_starts: tuple[int, ...] = Field(min_length=2, max_length=16)
    packed_action_ends: tuple[int, ...] = Field(min_length=2, max_length=16)
    canonical_action_hashes: tuple[str, ...] = Field(min_length=2, max_length=16)
    packed_action_hashes: tuple[str, ...] = Field(min_length=2, max_length=16)
    action_ordinals: tuple[int, ...] = Field(min_length=2, max_length=16)
    action_count: int = Field(ge=2, le=16)
    action_structure_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_terminal_result_start: int | None = Field(default=None, ge=0)
    canonical_terminal_result_end: int | None = Field(default=None, gt=0)
    packed_terminal_result_start: int | None = Field(default=None, ge=0)
    packed_terminal_result_end: int | None = Field(default=None, gt=0)
    canonical_terminal_result_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    packed_terminal_result_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    anchor_field: Literal["information_need"] = "information_need"
    anchor_match_key_hashes: tuple[str, ...] = Field(min_length=1, max_length=128)
    local_trigger_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    selection_score: tuple[int, int, int, int, int]
    eligible_candidate_count: int = Field(ge=1, le=4096)
    top_score_candidate_count: Literal[1] = 1
    catalog_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_evidence_content_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_packed_text_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_count: int = Field(ge=1, le=256)
    catalog_authority_hash: str = Field(pattern=SHA256_PATTERN)
    selected_authority_index: int = Field(ge=0, le=255)

    def _identity_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
            exclude={"receipt_id", "receipt_hash"},
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        witness_lengths = {
            len(self.witness_ids),
            len(self.witness_hashes),
            len(self.witness_ordinals),
            len(self.span_ref_hashes),
        }
        if witness_lengths != {1}:
            raise ValueError("inline action-chain receipt must bind exactly one witness")
        action_lengths = {
            self.action_count,
            len(self.canonical_action_starts),
            len(self.canonical_action_ends),
            len(self.packed_action_starts),
            len(self.packed_action_ends),
            len(self.canonical_action_hashes),
            len(self.packed_action_hashes),
            len(self.action_ordinals),
        }
        if len(action_lengths) != 1:
            raise ValueError("inline action-chain arrays must equal action_count")
        if self.action_ordinals != tuple(range(1, self.action_count + 1)):
            raise ValueError("inline action-chain ordinals must be contiguous from one")
        if any(
            re.fullmatch(SHA256_PATTERN, item) is None
            for item in (
                *self.witness_hashes,
                *self.span_ref_hashes,
                *self.canonical_action_hashes,
                *self.packed_action_hashes,
                *self.anchor_match_key_hashes,
            )
        ):
            raise ValueError("inline action-chain arrays contain an invalid hash")
        if len(self.anchor_match_key_hashes) != len(set(self.anchor_match_key_hashes)):
            raise ValueError("inline action-chain anchor hashes must be unique")
        if not (
            0
            <= self.canonical_title_start
            < self.canonical_title_end
            <= self.canonical_witness_start
            < self.canonical_witness_end
        ) or not (
            0
            <= self.packed_title_start
            < self.packed_title_end
            <= self.packed_witness_start
            < self.packed_witness_end
        ):
            raise ValueError("inline action-chain title/witness spans are invalid")
        for starts, ends, witness_start, witness_end in (
            (
                self.canonical_action_starts,
                self.canonical_action_ends,
                self.canonical_witness_start,
                self.canonical_witness_end,
            ),
            (
                self.packed_action_starts,
                self.packed_action_ends,
                self.packed_witness_start,
                self.packed_witness_end,
            ),
        ):
            if any(
                not witness_start <= start < end <= witness_end
                for start, end in zip(starts, ends, strict=True)
            ) or any(left_end > right_start for left_end, right_start in zip(ends, starts[1:])):
                raise ValueError("inline action-chain action spans are invalid or unordered")
        terminal_values = (
            self.canonical_terminal_result_start,
            self.canonical_terminal_result_end,
            self.packed_terminal_result_start,
            self.packed_terminal_result_end,
            self.canonical_terminal_result_hash,
            self.packed_terminal_result_hash,
        )
        if any(item is None for item in terminal_values) and not all(
            item is None for item in terminal_values
        ):
            raise ValueError("inline action-chain terminal result identity must be all-or-none")
        if self.canonical_terminal_result_start is not None:
            if not (
                self.canonical_action_ends[-1]
                <= self.canonical_terminal_result_start
                < self.canonical_terminal_result_end
                <= self.canonical_witness_end
                and self.packed_action_ends[-1]
                <= self.packed_terminal_result_start
                < self.packed_terminal_result_end
                <= self.packed_witness_end
            ):
                raise ValueError("inline action-chain terminal result must follow all actions")
        authority_lengths = {
            self.catalog_representation_count,
            len(self.catalog_evidence_ids),
            len(self.catalog_representation_ids),
            len(self.catalog_evidence_content_hashes),
            len(self.catalog_packed_text_hashes),
        }
        if len(authority_lengths) != 1:
            raise ValueError("inline action-chain authority arrays have inconsistent lengths")
        if len(self.catalog_evidence_ids) != len(set(self.catalog_evidence_ids)) or len(
            self.catalog_representation_ids
        ) != len(set(self.catalog_representation_ids)):
            raise ValueError("inline action-chain authority identities must be unique")
        if self.selected_authority_index >= self.catalog_representation_count:
            raise ValueError("inline action-chain selected authority index is out of range")
        if (
            self.catalog_evidence_ids[self.selected_authority_index] != self.evidence_id
            or self.catalog_representation_ids[self.selected_authority_index]
            != self.representation_id
            or self.catalog_evidence_content_hashes[self.selected_authority_index]
            != self.evidence_content_hash
        ):
            raise ValueError("inline action-chain selected authority identity is inconsistent")
        expected_authority_hash = stable_content_hash(
            {
                "evidence_ids": list(self.catalog_evidence_ids),
                "representation_ids": list(self.catalog_representation_ids),
                "canonical_hashes": list(self.catalog_evidence_content_hashes),
                "packed_hashes": list(self.catalog_packed_text_hashes),
            }
        )
        if self.catalog_authority_hash != expected_authority_hash:
            raise ValueError("inline action-chain catalog_authority_hash is invalid")
        expected_structure_hash = stable_content_hash(
            {
                "action_ordinals": list(self.action_ordinals),
                "canonical_action_hashes": list(self.canonical_action_hashes),
                "packed_action_hashes": list(self.packed_action_hashes),
                "canonical_terminal_result_hash": self.canonical_terminal_result_hash,
                "packed_terminal_result_hash": self.packed_terminal_result_hash,
            }
        )
        if self.action_structure_hash != expected_structure_hash:
            raise ValueError("inline action-chain structure hash is invalid")
        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match inline action-chain closure receipt")
        if self.receipt_id != stable_identifier("qiac_", payload):
            raise ValueError("receipt_id does not match inline action-chain closure receipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedListAuthorityRepresentation],
        catalog_witnesses: Sequence[EvidenceWitness],
    ) -> Self:
        """从完整 authority 与 Witness 图选择唯一义务内联动作链。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_catalog_witnesses = tuple(
            EvidenceWitness.from_dict(item.to_dict()) for item in catalog_witnesses
        )
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        authorities = tuple(
            ScopedListAuthorityRepresentation(
                evidence_id=item.evidence_id,
                representation_id=item.representation_id,
                canonical_text=item.canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority_representations
        )
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("inline action-chain closure requires a complete witness catalog")
        if (
            catalog.question_contract_id != question.contract_id
            or catalog.question_contract_hash != question.contract_hash
        ):
            raise ValueError("inline action-chain catalog belongs to another question")
        matched_obligation = next(
            (
                item
                for item in question.obligations
                if item.obligation_id == obligation.obligation_id
            ),
            None,
        )
        if matched_obligation != obligation:
            raise ValueError("inline action-chain obligation is outside the question contract")
        if (
            obligation.closure != FacetClosure.OPEN_ENUMERATION
            or obligation.required_output_form != RequiredOutputForm.ORDERED_STEPS
        ):
            raise ValueError("inline action-chain closure requires open ordered_steps")
        catalog.validate_witnesses(typed_catalog_witnesses)
        expected_witnesses = tuple(
            item
            for item in typed_catalog_witnesses
            if item.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
        )
        if typed_witnesses != expected_witnesses:
            raise ValueError("inline action-chain builder requires every sentence/list witness")
        if not typed_witnesses or not authorities:
            raise ValueError("inline action-chain builder has no candidate authority")
        evidence_ids = tuple(item.evidence_id for item in authorities)
        representation_ids = tuple(item.representation_id for item in authorities)
        if len(evidence_ids) != len(set(evidence_ids)) or len(representation_ids) != len(
            set(representation_ids)
        ):
            raise ValueError("inline action-chain authorities must have unique identities")
        authority_index_by_identity = {
            (item.evidence_id, item.representation_id): index
            for index, item in enumerate(authorities)
        }
        canonical_hashes = tuple(sha256_digest(item.canonical_text) for item in authorities)
        packed_hashes = tuple(sha256_digest(item.packed_text) for item in authorities)
        local_trigger = _inline_obligation_local_trigger(
            question=question,
            obligation=obligation,
        )

        candidates: list[_InlineActionCandidate] = []
        blocked: list[tuple[tuple[int, int, int, int, int], str]] = []
        for witness in typed_witnesses:
            if witness.context_pack_hash != catalog.context_pack_hash or (
                witness.authorized_scope_hash != catalog.authorized_scope_hash
            ):
                raise ValueError("inline action-chain witness scope differs from catalog")
            authority_index = authority_index_by_identity.get(
                (witness.span_ref.evidence_id, witness.span_ref.representation_id)
            )
            if authority_index is None:
                raise ValueError("inline action-chain witness has no authority representation")
            authority = authorities[authority_index]
            witness.span_ref.validate_artifacts(
                evidence_id=authority.evidence_id,
                representation_id=authority.representation_id,
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )
            canonical_witness = authority.canonical_text[
                witness.span_ref.canonical_start : witness.span_ref.canonical_end
            ]
            packed_witness = authority.packed_text[
                witness.span_ref.packed_start : witness.span_ref.packed_end
            ]
            try:
                canonical_title = _inline_title_before_span(
                    authority.canonical_text,
                    span_start=witness.span_ref.canonical_start,
                )
                packed_title = _inline_title_before_span(
                    authority.packed_text,
                    span_start=witness.span_ref.packed_start,
                )
            except ValueError:
                continue
            scored = _inline_candidate_score(
                title=canonical_title[2],
                witness_text=canonical_witness,
                information_need=obligation.information_need,
                local_trigger=local_trigger,
            )
            if scored is None:
                continue
            score, match_keys = scored
            if not _inline_authority_pair_is_proved(authority):
                blocked.append((score, "canonical and packed inline action-chain structures drift"))
                continue
            try:
                canonical_parse = _parse_inline_action_chain(
                    canonical_witness,
                    absolute_start=witness.span_ref.canonical_start,
                )
            except ValueError as exc:
                if _inline_non_candidate_parse_failure(str(exc)):
                    continue
                blocked.append((score, str(exc)))
                continue
            try:
                packed_parse = _parse_inline_action_chain(
                    packed_witness,
                    absolute_start=witness.span_ref.packed_start,
                )
            except ValueError as exc:
                blocked.append((score, str(exc)))
                continue
            try:
                canonical_actions = tuple(
                    authority.canonical_text[start:end]
                    for start, end in zip(
                        canonical_parse.starts,
                        canonical_parse.ends,
                        strict=True,
                    )
                )
                packed_actions = tuple(
                    authority.packed_text[start:end]
                    for start, end in zip(
                        packed_parse.starts,
                        packed_parse.ends,
                        strict=True,
                    )
                )
                canonical_terminal = (
                    authority.canonical_text[
                        canonical_parse.terminal_result_start : canonical_parse.terminal_result_end
                    ]
                    if canonical_parse.terminal_result_start is not None
                    and canonical_parse.terminal_result_end is not None
                    else None
                )
                packed_terminal = (
                    authority.packed_text[
                        packed_parse.terminal_result_start : packed_parse.terminal_result_end
                    ]
                    if packed_parse.terminal_result_start is not None
                    and packed_parse.terminal_result_end is not None
                    else None
                )
                if (
                    _normalize_table_cell(canonical_title[2])
                    != _normalize_table_cell(packed_title[2])
                    or _normalize_table_cell(canonical_witness)
                    != _normalize_table_cell(packed_witness)
                    or tuple(_normalize_table_cell(item) for item in canonical_actions)
                    != tuple(_normalize_table_cell(item) for item in packed_actions)
                    or (
                        _normalize_table_cell(canonical_terminal or "")
                        != _normalize_table_cell(packed_terminal or "")
                    )
                ):
                    raise ValueError("canonical and packed inline action-chain structures drift")
            except ValueError as exc:
                blocked.append((score, str(exc)))
                continue
            candidates.append(
                _InlineActionCandidate(
                    authority_index=authority_index,
                    witness=witness,
                    canonical_title_start=canonical_title[0],
                    canonical_title_end=canonical_title[1],
                    packed_title_start=packed_title[0],
                    packed_title_end=packed_title[1],
                    normalized_title=_normalize_anchor_text(canonical_title[2]),
                    canonical_parse=canonical_parse,
                    packed_parse=packed_parse,
                    anchor_match_keys=match_keys,
                    local_trigger=local_trigger,
                    score=score,
                )
            )
        if not candidates:
            raise ValueError("inline action-chain closure has no eligible candidate")
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidate = candidates[0]
        top_count = sum(item.score == candidate.score for item in candidates)
        if top_count != 1:
            raise ValueError("inline action-chain closure has a tied top candidate")
        if any(score >= candidate.score for score, _ in blocked):
            raise ValueError(
                "inline action-chain closure has an equally specific blocked candidate"
            )

        authority = authorities[candidate.authority_index]
        witness = candidate.witness
        canonical_action_hashes = _inline_action_hashes(
            authority.canonical_text,
            starts=candidate.canonical_parse.starts,
            ends=candidate.canonical_parse.ends,
        )
        packed_action_hashes = _inline_action_hashes(
            authority.packed_text,
            starts=candidate.packed_parse.starts,
            ends=candidate.packed_parse.ends,
        )
        canonical_terminal_hash = (
            sha256_digest(
                authority.canonical_text[
                    candidate.canonical_parse.terminal_result_start : candidate.canonical_parse.terminal_result_end
                ]
            )
            if candidate.canonical_parse.terminal_result_start is not None
            and candidate.canonical_parse.terminal_result_end is not None
            else None
        )
        packed_terminal_hash = (
            sha256_digest(
                authority.packed_text[
                    candidate.packed_parse.terminal_result_start : candidate.packed_parse.terminal_result_end
                ]
            )
            if candidate.packed_parse.terminal_result_start is not None
            and candidate.packed_parse.terminal_result_end is not None
            else None
        )
        action_ordinals = tuple(range(1, len(canonical_action_hashes) + 1))
        action_structure_payload = {
            "action_ordinals": list(action_ordinals),
            "canonical_action_hashes": list(canonical_action_hashes),
            "packed_action_hashes": list(packed_action_hashes),
            "canonical_terminal_result_hash": canonical_terminal_hash,
            "packed_terminal_result_hash": packed_terminal_hash,
        }
        authority_payload = {
            "evidence_ids": list(evidence_ids),
            "representation_ids": list(representation_ids),
            "canonical_hashes": list(canonical_hashes),
            "packed_hashes": list(packed_hashes),
        }
        payload: dict[str, Any] = {
            "schema_version": QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION,
            "normalization_version": QUERY_ANCHOR_NORMALIZATION_VERSION,
            "question_contract_id": question.contract_id,
            "question_contract_hash": question.contract_hash,
            "source_query_hash": question.source_query_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "obligation_id": obligation.obligation_id,
            "obligation_hash": obligation.obligation_hash,
            "information_need_hash": sha256_digest(obligation.information_need),
            "witness_ids": (witness.witness_id,),
            "witness_hashes": (witness.witness_hash,),
            "witness_ordinals": (witness.ordinal,),
            "span_ref_hashes": (witness.span_ref.span_ref_hash,),
            "witness_atom_kind": witness.atom_kind.value,
            "evidence_id": authority.evidence_id,
            "evidence_content_hash": canonical_hashes[candidate.authority_index],
            "representation_id": authority.representation_id,
            "canonical_witness_start": witness.span_ref.canonical_start,
            "canonical_witness_end": witness.span_ref.canonical_end,
            "packed_witness_start": witness.span_ref.packed_start,
            "packed_witness_end": witness.span_ref.packed_end,
            "canonical_witness_hash": sha256_digest(
                authority.canonical_text[
                    witness.span_ref.canonical_start : witness.span_ref.canonical_end
                ]
            ),
            "packed_witness_hash": sha256_digest(
                authority.packed_text[witness.span_ref.packed_start : witness.span_ref.packed_end]
            ),
            "canonical_title_start": candidate.canonical_title_start,
            "canonical_title_end": candidate.canonical_title_end,
            "packed_title_start": candidate.packed_title_start,
            "packed_title_end": candidate.packed_title_end,
            "canonical_title_hash": sha256_digest(
                authority.canonical_text[
                    candidate.canonical_title_start : candidate.canonical_title_end
                ]
            ),
            "packed_title_hash": sha256_digest(
                authority.packed_text[candidate.packed_title_start : candidate.packed_title_end]
            ),
            "normalized_title_hash": sha256_digest(candidate.normalized_title),
            "canonical_action_starts": candidate.canonical_parse.starts,
            "canonical_action_ends": candidate.canonical_parse.ends,
            "packed_action_starts": candidate.packed_parse.starts,
            "packed_action_ends": candidate.packed_parse.ends,
            "canonical_action_hashes": canonical_action_hashes,
            "packed_action_hashes": packed_action_hashes,
            "action_ordinals": action_ordinals,
            "action_count": len(action_ordinals),
            "action_structure_hash": stable_content_hash(action_structure_payload),
            "canonical_terminal_result_start": (candidate.canonical_parse.terminal_result_start),
            "canonical_terminal_result_end": candidate.canonical_parse.terminal_result_end,
            "packed_terminal_result_start": candidate.packed_parse.terminal_result_start,
            "packed_terminal_result_end": candidate.packed_parse.terminal_result_end,
            "canonical_terminal_result_hash": canonical_terminal_hash,
            "packed_terminal_result_hash": packed_terminal_hash,
            "anchor_field": "information_need",
            "anchor_match_key_hashes": tuple(
                sha256_digest(item) for item in candidate.anchor_match_keys
            ),
            "local_trigger_hash": (
                sha256_digest(candidate.local_trigger)
                if candidate.local_trigger is not None
                else None
            ),
            "selection_score": candidate.score,
            "eligible_candidate_count": len(candidates),
            "top_score_candidate_count": 1,
            "catalog_evidence_ids": evidence_ids,
            "catalog_representation_ids": representation_ids,
            "catalog_evidence_content_hashes": canonical_hashes,
            "catalog_packed_text_hashes": packed_hashes,
            "catalog_representation_count": len(authorities),
            "catalog_authority_hash": stable_content_hash(authority_payload),
            "selected_authority_index": candidate.authority_index,
        }
        return cls(
            **payload,
            receipt_id=stable_identifier("qiac_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[ScopedListAuthorityRepresentation],
        catalog_witnesses: Sequence[EvidenceWitness],
    ) -> Self:
        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            authority_representations=authority_representations,
            catalog_witnesses=catalog_witnesses,
        )
        if rebuilt != self:
            raise ValueError(
                "inline action-chain closure receipt does not match supplied artifacts"
            )
        return self


SourceEnumerationClosureReceipt: TypeAlias = (
    QueryAnchoredTableRowClosureReceipt
    | QueryAnchoredProcedureListClosureReceipt
    | QueryAnchoredScopedProcedureListClosureReceipt
    | QueryAnchoredScopedBulletListClosureReceipt
    | QueryAnchoredScopedTableSelectionClosureReceipt
    | QueryAnchoredInlineActionChainClosureReceipt
    | QueryAnchoredScopedFactSetClosureReceipt
)


def parse_source_enumeration_closure_receipt(
    value: Mapping[str, Any],
) -> SourceEnumerationClosureReceipt:
    schema_version = str(value.get("schema_version") or "")
    if schema_version == QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredTableRowClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredProcedureListClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredScopedProcedureListClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredScopedTableSelectionClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredScopedBulletListClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredInlineActionChainClosureReceipt.from_dict(value)
    if schema_version == QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION:
        return QueryAnchoredScopedFactSetClosureReceipt.from_dict(value)
    raise ValueError("unsupported source enumeration closure receipt schema")


def source_closure_witness_ids(
    receipt: SourceEnumerationClosureReceipt,
) -> tuple[str, ...]:
    if isinstance(receipt, QueryAnchoredTableRowClosureReceipt):
        return (receipt.witness_id,)
    return receipt.witness_ids


__all__ = [
    "QUERY_ANCHORED_INLINE_ACTION_CHAIN_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_SCOPED_BULLET_LIST_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_SCOPED_PROCEDURE_LIST_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_SCOPED_TABLE_SELECTION_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHORED_TABLE_ROW_CLOSURE_SCHEMA_VERSION",
    "QUERY_ANCHOR_NORMALIZATION_VERSION",
    "QueryAnchoredInlineActionChainClosureReceipt",
    "QueryAnchoredProcedureListClosureReceipt",
    "QueryAnchoredScopedBulletListClosureReceipt",
    "QueryAnchoredScopedFactSetClosureReceipt",
    "QueryAnchoredScopedProcedureListClosureReceipt",
    "QueryAnchoredScopedTableSelectionClosureReceipt",
    "QueryAnchoredTableRowClosureReceipt",
    "SourceEnumerationClosureReceipt",
    "ScopedListAuthorityRepresentation",
    "ScopedBulletAuthorityRepresentation",
    "ScopedTableAuthorityRepresentation",
    "parse_source_enumeration_closure_receipt",
    "source_closure_witness_ids",
]
