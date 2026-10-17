"""Query 锚定、来源有界且可重放的事实集合闭包回执。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import html
import re
from collections.abc import Sequence
from typing import Any, Literal, Self
import unicodedata

from pydantic import Field, model_validator

from app.domain.answering._base import (
    EVIDENCE_ID_PATTERN,
    FACET_ID_PATTERN,
    REPRESENTATION_ID_PATTERN,
    SHA256_PATTERN,
    AnsweringContractModel,
    sha256_digest,
    stable_content_hash,
    stable_identifier,
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


QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION = (
    "query-anchored-scoped-fact-set-closure-receipt.v1"
)
FACT_SET_SELECTOR_NORMALIZATION_VERSION = "nfkc-markup-aware-fact-selector.v1"
SENTENCE_FACT_SEGMENTATION_METHOD = "open_enumeration_witness_atom.v1"
LIST_ITEM_FACT_SEGMENTATION_METHOD = "top_level_list_entry.v1"


class FactSetMode(StrEnum):
    """来源事实集合的可证明闭合形态。"""

    CLOSED_INLINE_COORDINATION = "closed_inline_coordination"
    SCOPED_TOPIC_PREDICATE_SET = "scoped_topic_predicate_set"
    SCOPED_BULLET_LEAD_CLAUSE_SET = "scoped_bullet_lead_clause_set"


class FactSetRelation(StrEnum):
    """允许签发闭包的显式关系族；未列出的关系默认不推断。"""

    INCLUSION = "inclusion"
    REQUIRED_INFORMATION = "required_information"
    DIRECTIVE_COMPLIANCE = "directive_compliance"
    APPLICABILITY_RESTRICTION = "applicability_restriction"
    SAFETY_INSTRUCTION = "safety_instruction"
    OWNERSHIP_REGISTRATION = "ownership_registration"


@dataclass(frozen=True, slots=True)
class FactSetAuthorityRepresentation:
    """构建回执时必须完整提供的一组 raw canonical/packed authority。"""

    evidence_id: str
    representation_id: str
    canonical_text: str
    packed_text: str

    def __post_init__(self) -> None:
        if re.fullmatch(EVIDENCE_ID_PATTERN, self.evidence_id) is None:
            raise ValueError("fact-set authority Evidence ID is invalid")
        if re.fullmatch(REPRESENTATION_ID_PATTERN, self.representation_id) is None:
            raise ValueError("fact-set authority representation ID is invalid")
        if not self.canonical_text or not self.packed_text:
            raise ValueError("fact-set authority texts must be non-empty")


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    content_start: int
    content_end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Section:
    index: int
    start: int
    end: int
    title: str | None
    title_start: int | None
    title_end: int | None
    heading_start: int | None
    heading_end: int | None


@dataclass(frozen=True, slots=True)
class _Selector:
    anchor_field: Literal["source_query", "information_need"]
    relation: FactSetRelation
    title_anchor: str | None
    topic_groups: tuple[str, ...]
    anchor_match_keys: tuple[str, ...]
    inspection_bullet_set: bool = False
    prerequisite_information: bool = False


@dataclass(frozen=True, slots=True)
class _WitnessView:
    witness: EvidenceWitness
    authority_index: int
    canonical_text: str
    packed_text: str
    canonical_section: _Section
    packed_section: _Section


@dataclass(frozen=True, slots=True)
class _Candidate:
    mode: FactSetMode
    relation: FactSetRelation
    authority_index: int
    members: tuple[_WitnessView, ...]
    canonical_scope: _Section
    packed_scope: _Section
    topic_group_member_counts: tuple[int, ...]
    member_semantics: tuple[tuple[str, ...], ...]


_MARKUP_PATTERN = re.compile(r"<[^>\r\n]{1,512}>")
_DEFAULT_IGNORABLE_MARKDOWN = frozenset("*_~`")
_OPEN_MARKER_PATTERN = re.compile(
    r"(?:包括\s*但\s*不\s*限\s*于|不\s*限\s*于|诸\s*如|例\s*如|比\s*如|"
    r"等\s*等|及\s*其\s*他|以\s*及\s*其\s*他|"
    r"(?<!相)(?<!平)(?<!对)等(?=[\s、，,；;。.!！？?]|$)|"
    r"\betc\.?\b|\be\.g\.?\b|…|\.\.\.)",
    flags=re.IGNORECASE,
)
_REFERENCE_MARKER_PATTERN = re.compile(
    r"(?:参\s*阅|参\s*考|另\s*见|详\s*见|请\s*见|见\s*第\s*[0-9一二三四五六七八九十]+\s*[章节页]|"
    r"\bsee\s+also\b|\brefer(?:red)?\s+to\b|\bas\s+described\s+in\b)",
    flags=re.IGNORECASE,
)
_TRUNCATION_MARKER_PATTERN = re.compile(
    rf"(?:{re.escape(PACKED_TRUNCATION_MARKER)}|\[\s*truncated\s*\]|正文[^\n]{{0,16}}截断|内容[^\n]{{0,16}}截断)",
    flags=re.IGNORECASE,
)
_LEADING_ANAPHOR_PATTERN = re.compile(
    r"^\s*(?:(?:此外|另外|同时|并且|而且|补充)[，,:：\s]*)?"
    r"(?:其|该|此|它|它们|上述|前述|后者|the\s+same\b|it\b|they\b|this\b|these\b)",
    flags=re.IGNORECASE,
)
_SUPPLEMENT_CONNECTOR_PATTERN = re.compile(
    r"^\s*(?:此外|另外|同时|并且|而且|补充|还|又|also\b|additionally\b|furthermore\b)",
    flags=re.IGNORECASE,
)
_DANGLING_CONNECTOR_PATTERN = re.compile(
    r"(?:、|，|,|和|与|及|以及|或|\b(?:and|or)\b)\s*[。.!！？?]?\s*$",
    flags=re.IGNORECASE,
)
_LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*+•]|[0-9]+[.)、])\s+")
_LIST_MARKER_CAPTURE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?:[-*+•]|[0-9]+[.)、])[ \t]+(?P<body>.+)$",
    flags=re.DOTALL,
)
_EMBEDDED_LIST_MARKER_PATTERN = re.compile(r"(?:\r?\n)[ \t]*(?:[-*+•]|[0-9]+[.)、])[ \t]+")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_TITLE_PATH_PATTERN = re.compile(r"^\s*标题路径\s*[:：]\s*(?P<title>.+?)\s*$")
_MEDIA_PLACEHOLDER_PATTERN = re.compile(
    r"^\[\s*(?:image|figure|photo|illustration|diagram|chart)\s+"
    r"(?:omitted|removed|unavailable)\s*\]$",
    flags=re.IGNORECASE,
)
_STATEFUL_TOPIC_MODIFIERS = (
    "高温",
    "低温",
    "高压",
    "低压",
    "潮湿",
    "干燥",
    "易燃",
    "有毒",
    "锋利",
)
_INSPECTION_ENUMERATION_PATTERN = re.compile(
    r"(?:应|应当|需要|要|请)?\s*(?:检查|核对|确认)\s*"
    r"(?:哪几项|哪些|什么|所有|相关|以下|下列|各项|几项|检查点)",
    flags=re.IGNORECASE,
)
_PREREQUISITE_INFORMATION_QUERY_PATTERN = re.compile(
    r"(?:配置|安装|连接)[^。.!！？?\n]{0,120}?"
    r"(?:必须|需要|应当)?\s*满足\s*(?:什么|哪些|何种)?\s*前提条件",
    flags=re.IGNORECASE,
)
_PREREQUISITE_INFORMATION_SOURCE_PATTERN = re.compile(
    r"[^。.!！？?\r\n]{0,40}?(?:安装|配置|连接)前\s*[,，]?\s*"
    r"(?:必须|应当|需要)\s*(?:了解|掌握|准备|获取)\s*"
    r"(?P<items>.+?)[。.!]?$",
    flags=re.IGNORECASE,
)
_CHECKLIST_TITLE_PATTERN = re.compile(
    r"(?:请|应|应当|需要|要)?\s*(?:检查|核对|确认)\s*"
    r"(?:以下|下列|这些)?\s*(?:几)?\s*(?:方面|项(?:内容|目)?|内容|检查点)",
    flags=re.IGNORECASE,
)
_INDEPENDENT_CHECK_LEAD_PATTERN = re.compile(
    r"^(?:(?:请|应|应当|需要|必须|务必)\s*)?"
    r"(?:检查|核对|确认|查看|确保|关闭|打开|开启|重新启动|重启|连接|断开|"
    r"清除|设置|验证|测试|打印|更换|移除|安装|启动|停止)",
    flags=re.IGNORECASE,
)
_TAIL_SIBLING_OR_NORMATIVE_PATTERN = re.compile(
    r"(?:^|[。.!！？?；;]\s*)"
    r"(?:还应|还需|另外|此外|同时|并且|而且|另有|"
    r"以及\s*(?:应|需|要)?\s*(?:检查|核对|确认)|"
    r"其他|必须|应当|务必|不得|严禁|"
    r"需要\s*(?:检查|核对|确认|执行|完成)|"
    r"(?:检查|核对|确认)\s*(?:另|其|下|是|有))",
    flags=re.IGNORECASE,
)
_NARROW_XREF_PATTERN = re.compile(
    r"(?:请\s*)?(?:参阅|参考|另见|详见|请见)"
    r"[^。.!！？?\r\n]{1,120}?"
    r"(?:用户指南|操作指南|使用指南|说明书|手册|文档|第\s*"
    r"[0-9一二三四五六七八九十]+\s*[章节页])",
    flags=re.IGNORECASE,
)


def _is_default_ignorable(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or 0x180B <= codepoint <= 0x180D
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _plain_text(value: str) -> str:
    """移除展示标记但保留自然语言标点，供 fail-closed 结构分析。"""

    decoded = html.unescape(value)
    without_tags = _MARKUP_PATTERN.sub("", decoded)
    normalized = unicodedata.normalize("NFKC", without_tags).casefold()
    return "".join(
        character
        for character in normalized
        if not _is_default_ignorable(character) and character not in _DEFAULT_IGNORABLE_MARKDOWN
    )


def _normalize_semantic_text(value: str) -> str:
    plain = _plain_text(value)
    return "".join(character for character in plain if not character.isspace())


def _normalize_anchor_text(value: str) -> str:
    plain = _plain_text(value)
    return "".join(
        character
        for character in plain
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _normalize_regulatory_match_text(value: str) -> str:
    """将欧盟/欧共体同义标题规约到同一可审计匹配键。"""

    normalized = _normalize_anchor_text(value)
    normalized = re.sub(r"欧共体(?:ec)?|european(?:community|union)", "欧盟", normalized)
    return normalized


def _normalize_title_match_text(value: str) -> str:
    normalized = _normalize_regulatory_match_text(value)
    return normalized.replace("合规要求", "合规").replace("合规性", "合规").replace("的", "")


def _physical_lines(text: str) -> tuple[_Line, ...]:
    lines: list[_Line] = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        left = len(body) - len(body.lstrip())
        right = len(body.rstrip())
        lines.append(
            _Line(
                start=cursor,
                end=cursor + len(raw),
                content_start=cursor + left,
                content_end=cursor + right,
                text=body[left:right],
            )
        )
        cursor += len(raw)
    if cursor < len(text):
        body = text[cursor:]
        left = len(body) - len(body.lstrip())
        right = len(body.rstrip())
        lines.append(
            _Line(
                start=cursor,
                end=len(text),
                content_start=cursor + left,
                content_end=cursor + right,
                text=body[left:right],
            )
        )
    return tuple(lines)


def _heading_from_line(line: _Line) -> tuple[str, int, int] | None:
    path_match = _TITLE_PATH_PATTERN.fullmatch(line.text)
    if path_match is not None:
        raw_title = path_match.group("title")
        title = re.split(r"\s*>\s*", raw_title)[-1].strip()
        if not title:
            return None
        relative = line.text.rfind(title)
        return title, line.content_start + relative, line.content_start + relative + len(title)
    markdown_match = _MARKDOWN_HEADING_PATTERN.fullmatch(line.text)
    if markdown_match is not None:
        title = markdown_match.group("title").strip()
        relative = line.text.find(title)
        return title, line.content_start + relative, line.content_start + relative + len(title)
    plain = _plain_text(line.text).strip()
    if _MEDIA_PLACEHOLDER_PATTERN.fullmatch(plain) is not None:
        return None
    if (
        1 <= len(plain) <= 48
        and not re.search(r"[。.!！？?；;:]$", plain)
        and _LIST_MARKER_PATTERN.match(line.text) is None
        and not line.text.startswith(("![", "|", "<"))
        and not re.search(r"(?:请勿|不得|必须|包括|包含|是|为|are\b|is\b)", plain)
    ):
        return line.text, line.content_start, line.content_end
    return None


def _locate_sections(text: str) -> tuple[_Section, ...]:
    headings: list[tuple[_Line, str, int, int]] = []
    for line in _physical_lines(text):
        resolved = _heading_from_line(line)
        if resolved is not None:
            headings.append((line, *resolved))
    if not headings:
        return (_Section(0, 0, len(text), None, None, None, None, None),)
    sections: list[_Section] = []
    if headings[0][0].start > 0:
        sections.append(_Section(0, 0, headings[0][0].start, None, None, None, None, None))
    for offset, (line, title, title_start, title_end) in enumerate(headings):
        start = line.end
        end = headings[offset + 1][0].start if offset + 1 < len(headings) else len(text)
        sections.append(
            _Section(
                index=len(sections),
                start=start,
                end=end,
                title=title,
                title_start=title_start,
                title_end=title_end,
                heading_start=line.content_start,
                heading_end=line.content_end,
            )
        )
    return tuple(sections)


def _section_or_structural_title_for_span(
    sections: tuple[_Section, ...],
    *,
    start: int,
    end: int,
) -> _Section | None:
    """解析内容 section；覆盖其标题行的 atom 仅作为结构边界，不作为事实。"""

    matches = [section for section in sections if section.start <= start < end <= section.end]
    if len(matches) == 1:
        return matches[0]
    heading_matches = [
        section
        for section in sections
        if section.heading_start is not None
        and section.heading_end is not None
        and section.heading_start <= start < end <= section.heading_end
    ]
    if len(matches) == 0 and len(heading_matches) == 1:
        return None
    raise ValueError("fact-set witness does not belong to exactly one authority section")


def _relation_for_query(value: str) -> FactSetRelation | None:
    plain = _plain_text(value)
    rules: tuple[tuple[FactSetRelation, str], ...] = (
        (
            FactSetRelation.OWNERSHIP_REGISTRATION,
            r"(?:注册(?:的)?商标|注册商标|商标(?:属于|所有)|registered\s+(?:trade|service)\s*marks?)",
        ),
        (
            FactSetRelation.SAFETY_INSTRUCTION,
            r"(?:安全提示|安全要求|警告|危险|防止(?:触电|烫伤|伤害)|safety|warning)",
        ),
        (
            FactSetRelation.APPLICABILITY_RESTRICTION,
            r"(?:适用(?:于|范围)?|不适用|限制|仅限|(?<!前提)条件|applicab|restrict)",
        ),
        (
            FactSetRelation.REQUIRED_INFORMATION,
            r"(?:所需(?:信息|资料|字段)|需要提供|必填|required\s+(?:information|fields?))|"
            r"(?:配置|安装|连接)[^。.!！？?\n]{0,120}?"
            r"(?:必须|需要|应当)?\s*满足\s*(?:什么|哪些|何种)?\s*前提条件",
        ),
        (
            FactSetRelation.DIRECTIVE_COMPLIANCE,
            r"(?:必须|应当|需要遵守|合规要求|操作要求|must|shall|required\s+to)|"
            r"(?:应|需要|要|请)?\s*(?:检查|核对|确认)\s*"
            r"(?:哪几项|哪些|什么|所有|相关|以下|下列|各项|几项|检查点)",
        ),
        (
            FactSetRelation.INCLUSION,
            r"(?:包括|包含|含有|组成|有哪些|哪些|include|contain|consist)",
        ),
    )
    for relation, pattern in rules:
        if re.search(pattern, plain, flags=re.IGNORECASE):
            return relation
    return None


def _inspection_context(value: str) -> str | None:
    """提取“某条件下检查若干项”查询中的故障或场景锚点。"""

    plain = _plain_text(value)
    if _INSPECTION_ENUMERATION_PATTERN.search(plain) is None:
        return None
    conditional = re.search(
        r"(?:当|如果)\s*(?P<context>[^？?。；;\n]{2,120}?)\s*时\s*[,，]?\s*"
        r"(?:应|应当|需要|要|请)?\s*(?:检查|核对|确认)",
        plain,
        flags=re.IGNORECASE,
    )
    if conditional is None:
        return None
    context = _clean_topic(conditional.group("context"))
    return context if len(_normalize_anchor_text(context)) >= 4 else None


def _prerequisite_information_context(value: str) -> str | None:
    """仅解析网络安装/配置场景下显式询问的前提条件。"""

    plain = _plain_text(value)
    if _PREREQUISITE_INFORMATION_QUERY_PATTERN.search(plain) is None:
        return None
    match = re.search(
        r"(?:在|当)?\s*(?P<context>[^。.!！？?\n]{2,120}?)\s*时\s*[,，]?\s*"
        r"(?:必须|需要|应当)?\s*满足\s*(?:什么|哪些|何种)?\s*前提条件",
        plain,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    context = _clean_topic(match.group("context"))
    context_key = _normalize_anchor_text(context)
    if len(context_key) < 4 or re.search(r"(?:无线|网络|wi-?fi)", context_key) is None:
        return None
    return context


def _clean_topic(value: str) -> str:
    cleaned = _plain_text(value).strip(" \t\r\n，,。.!！？?；;:：")
    cleaned = re.sub(
        r"^(?:请(?:列出|说明|给出|回答|指出)?|查询|关于|针对|其中|以及|还有|the)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:分别|相关|方面|所对应|所涉及|的相关内容)\s*$",
        "",
        cleaned,
    )
    return cleaned.strip(" \t\r\n，,。.!！？?；;:：")


def _split_topics(value: str) -> tuple[str, ...]:
    parts = re.split(r"\s*(?:、|以及|和|与|及|\band\b)\s*", value, flags=re.IGNORECASE)
    cleaned = tuple(_clean_topic(part) for part in parts)
    return tuple(dict.fromkeys(part for part in cleaned if len(_normalize_anchor_text(part)) >= 2))


def _extract_title_anchor(value: str) -> str | None:
    plain = _plain_text(value)
    inspection_context = _inspection_context(value)
    if inspection_context is not None:
        return inspection_context
    match = re.search(
        r"(?:^|[，,。；;])\s*在\s*(?P<title>[^，,。；;!?？]{1,48}?)\s*(?:中|内)\s*[，,]", plain
    )
    if match is None:
        compliance_match = re.search(
            r"(?P<title>(?:欧盟|欧共体|european\s+(?:community|union))"
            r"[^，,。；;!?？\n]{0,32}?指令(?:的)?合规(?:要求|性)?)",
            plain,
            flags=re.IGNORECASE,
        )
        if compliance_match is None:
            return None
        title = _clean_topic(compliance_match.group("title"))
        return title or None
    title = _clean_topic(match.group("title"))
    return title or None


def _extract_topic_groups(
    value: str,
    *,
    relation: FactSetRelation,
) -> tuple[str, ...]:
    plain = _plain_text(value)
    if relation == FactSetRelation.DIRECTIVE_COMPLIANCE:
        inspection_context = _inspection_context(value)
        if inspection_context is not None:
            return (inspection_context,)
    if relation == FactSetRelation.REQUIRED_INFORMATION:
        prerequisite_context = _prerequisite_information_context(value)
        if prerequisite_context is not None:
            return (prerequisite_context,)
    about_match = re.search(
        r"关于\s*(?P<topics>.+?)\s*(?:分别|各自|各)\s*(?:提出|给出|说明|包含|包括|有|是)",
        plain,
    )
    if about_match is not None:
        topics = _split_topics(about_match.group("topics"))
        if len(topics) >= 2:
            return topics

    if relation == FactSetRelation.OWNERSHIP_REGISTRATION:
        owner_match = re.search(
            r"(?P<topic>[^？?。；;\n]{2,120}?)\s*(?:所)?注册(?:的)?\s*(?:商标|服务标志)",
            plain,
        )
        if owner_match is not None:
            topic = re.split(r"[？?。；;]", owner_match.group("topic"))[-1]
            cleaned = _clean_topic(topic)
            if len(_normalize_anchor_text(cleaned)) >= 2:
                return (cleaned,)
        english_owner = re.search(
            r"(?P<topic>[a-z][a-z0-9 .,&'\-]{1,80}?)\s+(?:registered\s+)?(?:trade|service)\s*marks?",
            plain,
            flags=re.IGNORECASE,
        )
        if english_owner is not None:
            cleaned = _clean_topic(english_owner.group("topic"))
            if len(_normalize_anchor_text(cleaned)) >= 2:
                return (cleaned,)

    if relation == FactSetRelation.DIRECTIVE_COMPLIANCE:
        directive_match = re.search(
            r"(?:符合|遵守)\s*(?:了)?(?:哪些|哪几项?)?\s*"
            r"(?P<topic>(?:欧盟|欧共体|european\s+(?:community|union))?"
            r"[^，,。；;!?？\n]{0,32}?指令)\s*(?:的)?合规(?:要求|性)",
            plain,
            flags=re.IGNORECASE,
        )
        if directive_match is not None:
            topic = _clean_topic(directive_match.group("topic"))
            if len(_normalize_anchor_text(topic)) >= 2:
                return (topic,)

    if relation == FactSetRelation.APPLICABILITY_RESTRICTION and re.search(
        r"(?:地理)?环境(?:限制|要求)?", plain
    ):
        # “地理环境限制”是值域名称，不会逐字出现在限制句中；仅在来源同时明确
        # 热带区域与海拔区域两个独立维度时，后续唯一 section 检查才允许签发。
        return ("热带地区", "海拔地区")

    relation_patterns = {
        FactSetRelation.INCLUSION: r"(?P<topic>[^？?。；;\n]{2,100}?)\s*(?:包括|包含|含有|由哪些)",
        FactSetRelation.REQUIRED_INFORMATION: r"(?P<topic>[^？?。；;\n]{2,100}?)\s*(?:所需|需要提供|必填)",
        FactSetRelation.DIRECTIVE_COMPLIANCE: r"(?P<topic>[^？?。；;\n]{2,100}?)\s*(?:必须|应当|需要遵守)",
        FactSetRelation.APPLICABILITY_RESTRICTION: r"(?P<topic>[^？?。；;\n]{2,100}?)\s*(?:适用|限制|仅限)",
        FactSetRelation.SAFETY_INSTRUCTION: r"关于\s*(?P<topic>[^？?。；;\n]{2,100}?)\s*(?:的)?安全",
    }
    pattern = relation_patterns.get(relation)
    if pattern is not None and (match := re.search(pattern, plain)) is not None:
        topic = re.split(r"[，,]", match.group("topic"))[-1]
        cleaned = _clean_topic(topic)
        if len(_normalize_anchor_text(cleaned)) >= 2:
            return (cleaned,)
    return ()


def _selector_for_anchor(
    value: str,
    *,
    anchor_field: Literal["source_query", "information_need"],
) -> _Selector | None:
    relation = _relation_for_query(value)
    if relation is None:
        return None
    topics = _extract_topic_groups(value, relation=relation)
    if not topics:
        return None
    normalized_topics = tuple(_normalize_anchor_text(topic) for topic in topics)
    if len(normalized_topics) != len(set(normalized_topics)):
        return None
    title_anchor = _extract_title_anchor(value)
    keys = tuple(
        dict.fromkeys(
            (
                relation.value,
                *(topic for topic in topics),
                *((title_anchor,) if title_anchor is not None else ()),
            )
        )
    )
    if len(keys) < 2:
        return None
    return _Selector(
        anchor_field=anchor_field,
        relation=relation,
        title_anchor=title_anchor,
        topic_groups=topics,
        anchor_match_keys=keys,
        inspection_bullet_set=(
            relation == FactSetRelation.DIRECTIVE_COMPLIANCE
            and _inspection_context(value) is not None
        ),
        prerequisite_information=(
            relation == FactSetRelation.REQUIRED_INFORMATION
            and _prerequisite_information_context(value) is not None
        ),
    )


def _title_matches(anchor: str, title: str | None) -> bool:
    if title is None:
        return False
    anchor_key = _normalize_title_match_text(anchor)
    title_key = _normalize_title_match_text(title)
    return bool(
        anchor_key
        and title_key
        and len(title_key) >= 2
        and (title_key in anchor_key or anchor_key in title_key)
    )


def _cjk_bigrams(value: str) -> frozenset[str]:
    result: set[str] = set()
    for sequence in re.findall(r"[\u3400-\u9fff]+", value):
        result.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return frozenset(result)


def _inspection_title_matches(anchor: str, title: str | None) -> bool:
    """要求标题既声明封闭检查清单，又与查询场景有充分词面交集。"""

    if title is None:
        return False
    title_plain = re.sub(
        r"^\s*[0-9一二三四五六七八九十]+\s*[.．、)]\s*",
        "",
        _plain_text(title),
    )
    if _CHECKLIST_TITLE_PATTERN.search(title_plain) is None:
        return False
    anchor_key = _normalize_anchor_text(anchor)
    title_key = _normalize_anchor_text(title_plain)
    if not anchor_key or not title_key:
        return False
    if min(len(anchor_key), len(title_key)) >= 4 and (
        anchor_key in title_key or title_key in anchor_key
    ):
        return True

    anchor_bigrams = _cjk_bigrams(anchor_key)
    title_bigrams = _cjk_bigrams(title_key)
    if anchor_bigrams:
        required_overlap = max(3, (len(anchor_bigrams) + 3) // 4)
        if len(anchor_bigrams & title_bigrams) >= required_overlap:
            return True

    ignored_latin_tokens = {"the", "and", "for", "with", "when"}
    anchor_tokens = {
        item for item in re.findall(r"[a-z0-9]{3,}", anchor_key) if item not in ignored_latin_tokens
    }
    title_tokens = {
        item for item in re.findall(r"[a-z0-9]{3,}", title_key) if item not in ignored_latin_tokens
    }
    return len(anchor_tokens & title_tokens) >= 2


def _topic_matches(topic: str, value: str) -> bool:
    topic_key = _normalize_regulatory_match_text(topic)
    value_key = _normalize_regulatory_match_text(value)
    if not topic_key or not value_key:
        return False
    if topic_key in value_key:
        return True
    latin_tokens = tuple(
        token for token in re.findall(r"[a-z0-9]{2,}", topic_key) if token not in {"the"}
    )
    if latin_tokens:
        return all(token in value_key for token in latin_tokens)
    stateful_modifier = next(
        (modifier for modifier in _STATEFUL_TOPIC_MODIFIERS if topic_key.startswith(modifier)),
        None,
    )
    if stateful_modifier is not None:
        nominal = topic_key.removeprefix(stateful_modifier)
        modifier_is_state = (
            re.search(
                rf"{re.escape(stateful_modifier)}(?:状态|环境|条件|运行|工作|区域)",
                value_key,
            )
            is not None
            or re.search(
                rf"(?:处于|达到|保持|呈现){re.escape(stateful_modifier)}",
                value_key,
            )
            is not None
        )
        if not modifier_is_state or not nominal:
            return False
        nominal_characters = tuple(
            dict.fromkeys(character for character in nominal if "\u3400" <= character <= "\u9fff")
        )
        if not nominal_characters:
            return False
        return (
            sum(character in value_key for character in nominal_characters)
            / len(nominal_characters)
            >= 0.8
        )
    characters = tuple(
        dict.fromkeys(character for character in topic_key if "\u3400" <= character <= "\u9fff")
    )
    if len(characters) < 4:
        return False
    covered = sum(character in value_key for character in characters)
    bigrams = tuple(topic_key[index : index + 2] for index in range(len(topic_key) - 1))
    return covered / len(characters) >= 0.8 and any(item in value_key for item in bigrams)


def _has_open_marker(value: str) -> bool:
    return _OPEN_MARKER_PATTERN.search(_plain_text(value)) is not None


def _has_reference_marker(value: str) -> bool:
    return _REFERENCE_MARKER_PATTERN.search(_plain_text(value)) is not None


def _has_truncation_marker(value: str) -> bool:
    return _TRUNCATION_MARKER_PATTERN.search(value) is not None


def _has_ambiguous_anaphor(value: str) -> bool:
    return _LEADING_ANAPHOR_PATTERN.search(_plain_text(value)) is not None


def _has_dangling_connector(value: str) -> bool:
    return _DANGLING_CONNECTOR_PATTERN.search(_plain_text(value)) is not None


def _split_closed_coordination(value: str) -> tuple[str, ...]:
    plain = _plain_text(value).strip(" \t\r\n。.!！？?；;")
    if _has_open_marker(plain) or _has_dangling_connector(plain):
        return ()
    parts = re.split(
        r"\s*(?:、|；|;|，|,(?!\s*(?:inc|ltd|llc|corp)\b)|以及|和|与|及|\band\b|\bor\b)\s*",
        plain,
        flags=re.IGNORECASE,
    )
    cleaned = tuple(
        re.sub(r"^(?:the|a|an)\s+", "", part.strip(), flags=re.IGNORECASE) for part in parts
    )
    normalized = tuple(_normalize_semantic_text(part) for part in cleaned if part)
    if len(normalized) < 2 or any(not part for part in normalized):
        return ()
    if len(normalized) != len(set(normalized)):
        return ()
    return normalized


def _inline_semantics(
    value: str,
    *,
    relation: FactSetRelation,
) -> tuple[str, ...]:
    plain = _plain_text(value).strip()
    if (
        _has_open_marker(plain)
        or _has_reference_marker(plain)
        or _has_truncation_marker(value)
        or _has_ambiguous_anaphor(plain)
        or _has_dangling_connector(plain)
    ):
        return ()

    items: str | None = None
    if relation == FactSetRelation.OWNERSHIP_REGISTRATION:
        company_first = re.search(
            r"(?:注册(?:的)?(?:商标|服务标志)|(?:商标|服务标志)(?:包括|包含))\s*(?:包括|包含|有|为|是)?\s*[:：]?\s*(?P<items>.+?)[。.!]?$",
            plain,
            flags=re.IGNORECASE,
        )
        subject_first = re.search(
            r"^(?P<items>.+?)\s*(?:是|为)\s*.+?(?:注册(?:的)?(?:商标|服务标志)|注册商标)[。.!]?$",
            plain,
            flags=re.IGNORECASE,
        )
        english = re.search(
            r"^(?P<items>.+?)\s+(?:are|is)\s+(?:registered\s+)?(?:trademarks?|service\s+marks?).+?\bof\b\s+.+?[.!]?$",
            plain,
            flags=re.IGNORECASE,
        )
        match = subject_first or english or company_first
        if match is not None:
            items = match.group("items")
    else:
        enumerator_patterns = {
            FactSetRelation.INCLUSION: r"(?:包括|包含|含有|由)\s*[:：]?\s*(?P<items>.+?)(?:组成)?[。.!]?$",
            FactSetRelation.REQUIRED_INFORMATION: (
                r"(?:所需(?:信息|资料|字段)?(?:包括|为|是)|需要提供|"
                r"必填(?:项|字段)?(?:包括|为|是))\s*[:：]?\s*"
                r"(?P<items>.+?)[。.!]?$|"
                + _PREREQUISITE_INFORMATION_SOURCE_PATTERN.pattern.replace(
                    "(?P<items>", "(?P<prerequisite_items>"
                )
            ),
            FactSetRelation.DIRECTIVE_COMPLIANCE: (
                r"(?:符合|遵守)\s*(?P<items>.+?)\s*指令"
                r"(?:的[^。.!！？?]{0,160})?(?:保护|安全|合规)要求[。.!！？?]?$|"
                r"(?:必须|应当|需要遵守)(?:满足|执行|提供|完成)?\s*[:：]?\s*"
                r"(?P<directive_items>.+?)[。.!]?$"
            ),
            FactSetRelation.APPLICABILITY_RESTRICTION: r"(?:适用(?:范围)?(?:包括|为|是)|限制(?:包括|为|是)|仅限于)\s*[:：]?\s*(?P<items>.+?)[。.!]?$",
            FactSetRelation.SAFETY_INSTRUCTION: r"(?:安全提示|警告事项)(?:包括|为|是)\s*[:：]?\s*(?P<items>.+?)[。.!]?$",
        }
        pattern = enumerator_patterns[relation]
        if (match := re.search(pattern, plain, flags=re.IGNORECASE)) is not None:
            items = (
                match.groupdict().get("items")
                or match.groupdict().get("directive_items")
                or match.groupdict().get("prerequisite_items")
            )
    if items is None:
        return ()
    return _split_closed_coordination(items)


def _text_matches_relation(value: str, relation: FactSetRelation) -> bool:
    plain = _plain_text(value)
    patterns = {
        FactSetRelation.INCLUSION: r"(?:包括|包含|含有|组成|include|contain|consist)",
        FactSetRelation.REQUIRED_INFORMATION: r"(?:所需|需要提供|必填|required\s+(?:information|field))",
        FactSetRelation.DIRECTIVE_COMPLIANCE: r"(?:符合|遵守|必须|应当|需要|请|不得|must|shall|should)",
        FactSetRelation.APPLICABILITY_RESTRICTION: r"(?:适用|不适用|限制|仅限|条件|applicab|restrict)",
        FactSetRelation.SAFETY_INSTRUCTION: r"(?:请勿|不得|严禁|必须|应当|需要|检查|保持|避免|以免|小心|警告|危险|触电|烫伤|高温|safety|warning|danger)",
        FactSetRelation.OWNERSHIP_REGISTRATION: r"(?:注册(?:的)?商标|注册商标|服务标志|registered\s+(?:trademark|service\s+mark))",
    }
    return re.search(patterns[relation], plain, flags=re.IGNORECASE) is not None


def _candidate_title_pair(
    view: _WitnessView,
) -> tuple[_Section, _Section] | None:
    canonical_title = view.canonical_section.title
    packed_title = view.packed_section.title
    if (canonical_title is None) != (packed_title is None):
        return None
    if canonical_title is not None and _normalize_semantic_text(canonical_title) != (
        _normalize_semantic_text(packed_title or "")
    ):
        return None
    return view.canonical_section, view.packed_section


def _prerequisite_information_source_matches(
    selector: _Selector,
    view: _WitnessView,
) -> bool:
    """把网络安装前提限定为显式的接入点凭据了解义务。"""

    if not selector.prerequisite_information or len(selector.topic_groups) != 1:
        return False
    context_key = _normalize_anchor_text(selector.topic_groups[0])
    title_key = _normalize_anchor_text(view.canonical_section.title or "")
    canonical = _plain_text(view.canonical_text)
    packed = _plain_text(view.packed_text)
    if (
        re.search(r"(?:无线|网络|wi-?fi)", context_key, flags=re.IGNORECASE) is None
        or "安装" not in title_key
        or re.search(r"(?:连接|网络|无线)", title_key, flags=re.IGNORECASE) is None
    ):
        return False
    for value in (canonical, packed):
        if (
            _PREREQUISITE_INFORMATION_SOURCE_PATTERN.fullmatch(value) is None
            or re.search(r"(?:接入点|无线路由器|access\s+point)", value, flags=re.IGNORECASE)
            is None
            or re.search(r"(?:网络名|ssid)", value, flags=re.IGNORECASE) is None
            or re.search(r"(?:密码|口令|凭据|password)", value, flags=re.IGNORECASE) is None
        ):
            return False
    return True


def _list_item_parts(value: str) -> tuple[int, str] | None:
    match = _LIST_MARKER_CAPTURE_PATTERN.fullmatch(value)
    if match is None:
        return None
    indent = len(match.group("indent").expandtabs(4))
    body = match.group("body").strip()
    return (indent, body) if body else None


def _bullet_lead_semantic(value: str) -> str | None:
    """返回一个完整 bullet 的独立首句语义；任何尾部新增义务均拒绝。"""

    parsed = _list_item_parts(value)
    if parsed is None or _EMBEDDED_LIST_MARKER_PATTERN.search(value) is not None:
        return None
    _, body = parsed
    if _has_open_marker(body) or _has_truncation_marker(body) or _has_dangling_connector(body):
        return None
    terminator = re.search(r"[。.!！？?]", body)
    if terminator is None:
        return None
    lead = body[: terminator.end()].strip()
    tail = body[terminator.end() :].strip()
    lead_plain = _plain_text(lead).strip()
    if (
        not lead_plain
        or _has_reference_marker(lead)
        or _has_ambiguous_anaphor(lead)
        or (_INDEPENDENT_CHECK_LEAD_PATTERN.search(lead_plain) is None and "是否" not in lead_plain)
    ):
        return None
    if not tail:
        return _normalize_semantic_text(lead)
    if (
        _has_ambiguous_anaphor(tail)
        or _SUPPLEMENT_CONNECTOR_PATTERN.search(_plain_text(tail)) is not None
        or _TAIL_SIBLING_OR_NORMATIVE_PATTERN.search(_plain_text(tail)) is not None
        or "其他" in _plain_text(tail)
    ):
        return None
    reference_count = len(_REFERENCE_MARKER_PATTERN.findall(_plain_text(tail)))
    if reference_count > 1 or (
        reference_count == 1 and _NARROW_XREF_PATTERN.search(_plain_text(tail)) is None
    ):
        return None
    return _normalize_semantic_text(lead)


def _bullet_views_are_one_contiguous_run(
    views: tuple[_WitnessView, ...],
    *,
    authority: FactSetAuthorityRepresentation,
) -> bool:
    for previous, following in zip(views, views[1:], strict=False):
        canonical_gap = authority.canonical_text[
            previous.witness.span_ref.canonical_end : following.witness.span_ref.canonical_start
        ]
        packed_gap = authority.packed_text[
            previous.witness.span_ref.packed_end : following.witness.span_ref.packed_start
        ]
        if canonical_gap.strip() or packed_gap.strip():
            return False
    return True


def _find_scoped_bullet_lead_candidates(
    *,
    selector: _Selector,
    views: tuple[_WitnessView, ...],
    authorities: tuple[FactSetAuthorityRepresentation, ...],
) -> tuple[_Candidate, ...]:
    if (
        not selector.inspection_bullet_set
        or selector.title_anchor is None
        or len(selector.topic_groups) != 1
    ):
        return ()
    grouped: dict[tuple[int, int], list[_WitnessView]] = {}
    for view in views:
        if view.witness.atom_kind == WitnessAtomKind.LIST_ITEM:
            grouped.setdefault((view.authority_index, view.canonical_section.index), []).append(
                view
            )

    candidates: list[_Candidate] = []
    for (authority_index, _), raw_section_views in grouped.items():
        section_views = tuple(sorted(raw_section_views, key=lambda item: item.witness.ordinal))
        if not 2 <= len(section_views) <= 64:
            continue
        first = section_views[0]
        title_pair = _candidate_title_pair(first)
        if title_pair is None or not _inspection_title_matches(
            selector.title_anchor, first.canonical_section.title
        ):
            continue
        if not _inspection_title_matches(selector.title_anchor, first.packed_section.title):
            continue
        canonical_parts = tuple(_list_item_parts(item.canonical_text) for item in section_views)
        packed_parts = tuple(_list_item_parts(item.packed_text) for item in section_views)
        if any(item is None for item in canonical_parts) or any(
            item is None for item in packed_parts
        ):
            continue
        canonical_indents = tuple(item[0] for item in canonical_parts if item is not None)
        packed_indents = tuple(item[0] for item in packed_parts if item is not None)
        if (
            len(set(canonical_indents)) != 1
            or len(set(packed_indents)) != 1
            or canonical_indents != packed_indents
        ):
            continue
        authority = authorities[authority_index]
        if not _bullet_views_are_one_contiguous_run(section_views, authority=authority):
            continue
        canonical_semantics = tuple(
            _bullet_lead_semantic(item.canonical_text) for item in section_views
        )
        packed_semantics = tuple(_bullet_lead_semantic(item.packed_text) for item in section_views)
        if (
            any(item is None for item in canonical_semantics)
            or canonical_semantics != packed_semantics
            or len(set(canonical_semantics)) != len(canonical_semantics)
        ):
            continue
        canonical_scope = _Section(
            index=first.canonical_section.index,
            start=section_views[0].witness.span_ref.canonical_start,
            end=section_views[-1].witness.span_ref.canonical_end,
            title=first.canonical_section.title,
            title_start=first.canonical_section.title_start,
            title_end=first.canonical_section.title_end,
            heading_start=first.canonical_section.heading_start,
            heading_end=first.canonical_section.heading_end,
        )
        packed_scope = _Section(
            index=first.packed_section.index,
            start=section_views[0].witness.span_ref.packed_start,
            end=section_views[-1].witness.span_ref.packed_end,
            title=first.packed_section.title,
            title_start=first.packed_section.title_start,
            title_end=first.packed_section.title_end,
            heading_start=first.packed_section.heading_start,
            heading_end=first.packed_section.heading_end,
        )
        candidates.append(
            _Candidate(
                mode=FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET,
                relation=selector.relation,
                authority_index=authority_index,
                members=section_views,
                canonical_scope=canonical_scope,
                packed_scope=packed_scope,
                topic_group_member_counts=(len(section_views),),
                member_semantics=tuple((item,) for item in canonical_semantics if item is not None),
            )
        )
    return tuple(candidates)


def _find_candidates(
    *,
    selector: _Selector,
    views: tuple[_WitnessView, ...],
    authorities: tuple[FactSetAuthorityRepresentation, ...],
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    supported_views = tuple(
        view
        for view in views
        if view.witness.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
    )

    for view in supported_views:
        if selector.prerequisite_information:
            topic_matches = _prerequisite_information_source_matches(selector, view)
        else:
            topic_matches = all(
                _topic_matches(topic, view.canonical_text) for topic in selector.topic_groups
            )
        if not topic_matches:
            continue
        semantics = _inline_semantics(view.canonical_text, relation=selector.relation)
        packed_semantics = _inline_semantics(view.packed_text, relation=selector.relation)
        if not semantics or semantics != packed_semantics:
            continue
        title_pair = _candidate_title_pair(view)
        if title_pair is None:
            continue
        member_scope = _Section(
            index=view.canonical_section.index,
            start=view.witness.span_ref.canonical_start,
            end=view.witness.span_ref.canonical_end,
            title=view.canonical_section.title,
            title_start=view.canonical_section.title_start,
            title_end=view.canonical_section.title_end,
            heading_start=view.canonical_section.heading_start,
            heading_end=view.canonical_section.heading_end,
        )
        packed_member_scope = _Section(
            index=view.packed_section.index,
            start=view.witness.span_ref.packed_start,
            end=view.witness.span_ref.packed_end,
            title=view.packed_section.title,
            title_start=view.packed_section.title_start,
            title_end=view.packed_section.title_end,
            heading_start=view.packed_section.heading_start,
            heading_end=view.packed_section.heading_end,
        )
        candidates.append(
            _Candidate(
                mode=FactSetMode.CLOSED_INLINE_COORDINATION,
                relation=selector.relation,
                authority_index=view.authority_index,
                members=(view,),
                canonical_scope=member_scope,
                packed_scope=packed_member_scope,
                topic_group_member_counts=(len(semantics),),
                member_semantics=(semantics,),
            )
        )

    if len(selector.topic_groups) >= 2 and selector.title_anchor is not None:
        grouped: dict[tuple[int, int], list[_WitnessView]] = {}
        for view in supported_views:
            grouped.setdefault((view.authority_index, view.canonical_section.index), []).append(
                view
            )
        for (_, _), section_views in grouped.items():
            first = section_views[0]
            title_pair = _candidate_title_pair(first)
            if title_pair is None or not _title_matches(
                selector.title_anchor, first.canonical_section.title
            ):
                continue
            chosen: list[_WitnessView] = []
            ambiguous = False
            for topic in selector.topic_groups:
                matches = [
                    view
                    for view in section_views
                    if _topic_matches(topic, view.canonical_text)
                    and _topic_matches(topic, view.packed_text)
                    and _text_matches_relation(view.canonical_text, selector.relation)
                    and _text_matches_relation(view.packed_text, selector.relation)
                ]
                if len(matches) != 1:
                    ambiguous = True
                    break
                chosen.append(matches[0])
            if ambiguous or len({item.witness.witness_id for item in chosen}) != len(chosen):
                continue
            ordered = tuple(sorted(chosen, key=lambda item: item.witness.ordinal))
            if any(_has_ambiguous_anaphor(item.canonical_text) for item in ordered):
                continue
            semantics = tuple((_normalize_semantic_text(item.canonical_text),) for item in ordered)
            packed_semantics = tuple(
                (_normalize_semantic_text(item.packed_text),) for item in ordered
            )
            if semantics != packed_semantics:
                continue
            candidates.append(
                _Candidate(
                    mode=FactSetMode.SCOPED_TOPIC_PREDICATE_SET,
                    relation=selector.relation,
                    authority_index=first.authority_index,
                    members=ordered,
                    canonical_scope=first.canonical_section,
                    packed_scope=first.packed_section,
                    topic_group_member_counts=tuple(1 for _ in selector.topic_groups),
                    member_semantics=semantics,
                )
            )
    candidates.extend(
        _find_scoped_bullet_lead_candidates(
            selector=selector,
            views=supported_views,
            authorities=authorities,
        )
    )
    return tuple(candidates)


def _scope_diagnostic_counts(value: str) -> tuple[int, int, int, int]:
    plain = _plain_text(value)
    return (
        len(_OPEN_MARKER_PATTERN.findall(plain)),
        len(_REFERENCE_MARKER_PATTERN.findall(plain)),
        len(_TRUNCATION_MARKER_PATTERN.findall(value)),
        len(_LEADING_ANAPHOR_PATTERN.findall(plain)),
    )


def _reject_unclosed_candidate(
    *,
    selector: _Selector,
    candidate: _Candidate,
    views: tuple[_WitnessView, ...],
    authorities: tuple[FactSetAuthorityRepresentation, ...],
) -> None:
    authority = authorities[candidate.authority_index]
    canonical_scope_text = authority.canonical_text[
        candidate.canonical_scope.start : candidate.canonical_scope.end
    ]
    packed_scope_text = authority.packed_text[
        candidate.packed_scope.start : candidate.packed_scope.end
    ]
    canonical_counts = _scope_diagnostic_counts(canonical_scope_text)
    packed_counts = _scope_diagnostic_counts(packed_scope_text)
    if candidate.mode == FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET:
        canonical_unclosed = (
            canonical_counts[0],
            canonical_counts[2],
            canonical_counts[3],
        )
        packed_unclosed = (
            packed_counts[0],
            packed_counts[2],
            packed_counts[3],
        )
    else:
        canonical_unclosed = canonical_counts
        packed_unclosed = packed_counts
    if canonical_unclosed != tuple(0 for _ in canonical_unclosed) or (
        packed_unclosed != tuple(0 for _ in packed_unclosed)
    ):
        raise ValueError("fact-set candidate scope contains an unclosed source marker")
    if any(
        _has_dangling_connector(item.canonical_text) or _has_dangling_connector(item.packed_text)
        for item in candidate.members
    ):
        raise ValueError("fact-set candidate contains a dangling connector")

    member_ids = {item.witness.witness_id for item in candidate.members}
    for view in views:
        if view.witness.witness_id in member_ids or view.witness.atom_kind not in {
            WitnessAtomKind.SENTENCE,
            WitnessAtomKind.LIST_ITEM,
        }:
            continue
        if candidate.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET and (
            view.authority_index != candidate.authority_index
            or view.canonical_section.index != candidate.canonical_scope.index
        ):
            continue
        if candidate.mode == FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET and (
            view.authority_index != candidate.authority_index
            or view.witness.span_ref.canonical_start < candidate.canonical_scope.start
            or view.witness.span_ref.canonical_end > candidate.canonical_scope.end
        ):
            continue
        topic_related = any(
            _topic_matches(topic, view.canonical_text) for topic in selector.topic_groups
        )
        predicate_related = _text_matches_relation(view.canonical_text, selector.relation)
        anaphoric_supplement = _SUPPLEMENT_CONNECTOR_PATTERN.search(
            _plain_text(view.canonical_text)
        ) is not None and _has_ambiguous_anaphor(view.canonical_text)
        if predicate_related and (topic_related or anaphoric_supplement):
            raise ValueError("fact-set authority contains an omitted or ambiguous supplement")


class QueryAnchoredScopedFactSetClosureReceipt(AnsweringContractModel):
    """证明完整 authority 中唯一、查询锚定且无遗漏的事实集合。"""

    schema_version: Literal["query-anchored-scoped-fact-set-closure-receipt.v1"] = (
        QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION
    )
    normalization_version: Literal["nfkc-markup-aware-fact-selector.v1"] = (
        FACT_SET_SELECTOR_NORMALIZATION_VERSION
    )
    receipt_id: str = Field(pattern=r"^qfsc_[0-9a-f]{40}$")
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    question_contract_id: str = Field(pattern=r"^qct_[0-9a-f]{40}$")
    question_contract_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_id: str = Field(pattern=r"^wcat_[0-9a-f]{40}$")
    catalog_hash: str = Field(pattern=SHA256_PATTERN)
    obligation_id: str = Field(pattern=FACET_ID_PATTERN)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    mode: FactSetMode
    relation: FactSetRelation
    anchor_field: Literal["source_query", "information_need"]
    anchor_match_key_hashes: tuple[str, ...] = Field(min_length=2, max_length=64)
    anchor_match_key_count: int = Field(ge=2, le=64)
    anchor_match_hash: str = Field(pattern=SHA256_PATTERN)
    topic_group_hashes: tuple[str, ...] = Field(min_length=1, max_length=32)
    topic_group_count: int = Field(ge=1, le=32)
    topic_group_member_counts: tuple[int, ...] = Field(min_length=1, max_length=32)
    predicate_signature_hash: str = Field(pattern=SHA256_PATTERN)
    selector_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_evidence_content_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_packed_text_hashes: tuple[str, ...] = Field(min_length=1, max_length=256)
    catalog_representation_count: int = Field(ge=1, le=256)
    catalog_authority_hash: str = Field(pattern=SHA256_PATTERN)
    selected_authority_index: int = Field(ge=0, le=255)
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    evidence_content_hash: str = Field(pattern=SHA256_PATTERN)
    representation_id: str = Field(pattern=REPRESENTATION_ID_PATTERN)
    authority_candidate_count: Literal[1] = 1
    canonical_title_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    packed_title_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    canonical_title_start: int | None = Field(default=None, ge=0)
    canonical_title_end: int | None = Field(default=None, gt=0)
    packed_title_start: int | None = Field(default=None, ge=0)
    packed_title_end: int | None = Field(default=None, gt=0)
    canonical_scope_hash: str = Field(pattern=SHA256_PATTERN)
    packed_scope_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_scope_start: int = Field(ge=0)
    canonical_scope_end: int = Field(gt=0)
    packed_scope_start: int = Field(ge=0)
    packed_scope_end: int = Field(gt=0)
    canonical_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    packed_outside_scope_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_non_scope_character_count: int = Field(ge=0)
    packed_non_scope_character_count: int = Field(ge=0)
    member_count: int = Field(ge=1, le=64)
    witness_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    witness_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    witness_ordinals: tuple[int, ...] = Field(min_length=1, max_length=64)
    span_ref_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    member_atom_kinds: tuple[WitnessAtomKind, ...] = Field(min_length=1, max_length=64)
    segmentation_methods: tuple[str, ...] = Field(min_length=1, max_length=64)
    canonical_member_starts: tuple[int, ...] = Field(min_length=1, max_length=64)
    canonical_member_ends: tuple[int, ...] = Field(min_length=1, max_length=64)
    packed_member_starts: tuple[int, ...] = Field(min_length=1, max_length=64)
    packed_member_ends: tuple[int, ...] = Field(min_length=1, max_length=64)
    canonical_member_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    packed_member_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    canonical_member_normalized_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    packed_member_normalized_hashes: tuple[str, ...] = Field(min_length=1, max_length=64)
    member_semantic_counts: tuple[int, ...] = Field(min_length=1, max_length=64)
    flattened_semantic_hashes: tuple[str, ...] = Field(min_length=2, max_length=128)
    semantic_count: int = Field(ge=2, le=128)
    omission_marker_count: Literal[0] = 0
    open_marker_count: Literal[0] = 0
    reference_marker_count: Literal[0] = 0
    ambiguous_anaphor_count: Literal[0] = 0
    truncation_marker_count: Literal[0] = 0
    supplement_candidate_count: Literal[0] = 0

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
            raise ValueError("fact-set authority arrays must equal representation count")
        if len(self.catalog_evidence_ids) != len(set(self.catalog_evidence_ids)) or len(
            self.catalog_representation_ids
        ) != len(set(self.catalog_representation_ids)):
            raise ValueError("fact-set authority identities must be unique")
        if self.selected_authority_index >= self.catalog_representation_count:
            raise ValueError("selected fact-set authority index is out of range")
        if (
            self.catalog_evidence_ids[self.selected_authority_index] != self.evidence_id
            or self.catalog_representation_ids[self.selected_authority_index]
            != self.representation_id
            or self.catalog_evidence_content_hashes[self.selected_authority_index]
            != self.evidence_content_hash
        ):
            raise ValueError("selected fact-set authority identity is inconsistent")
        authority_payload = {
            "evidence_ids": list(self.catalog_evidence_ids),
            "representation_ids": list(self.catalog_representation_ids),
            "canonical_hashes": list(self.catalog_evidence_content_hashes),
            "packed_hashes": list(self.catalog_packed_text_hashes),
        }
        if self.catalog_authority_hash != stable_content_hash(authority_payload):
            raise ValueError("catalog_authority_hash does not match fact-set authority")

        if self.anchor_match_key_count != len(self.anchor_match_key_hashes) or (
            self.anchor_match_hash != stable_content_hash(list(self.anchor_match_key_hashes))
        ):
            raise ValueError("fact-set anchor match arrays are inconsistent")
        if len(self.anchor_match_key_hashes) != len(set(self.anchor_match_key_hashes)):
            raise ValueError("fact-set anchor match keys must be unique")
        if self.topic_group_count != len(self.topic_group_hashes) or (
            self.topic_group_count != len(self.topic_group_member_counts)
        ):
            raise ValueError("fact-set topic group arrays are inconsistent")
        if len(self.topic_group_hashes) != len(set(self.topic_group_hashes)):
            raise ValueError("fact-set topic groups must be unique")
        selector_payload = {
            "mode": self.mode.value,
            "relation": self.relation.value,
            "anchor_field": self.anchor_field,
            "anchor_match_key_hashes": list(self.anchor_match_key_hashes),
            "topic_group_hashes": list(self.topic_group_hashes),
            "topic_group_member_counts": list(self.topic_group_member_counts),
            "predicate_signature_hash": self.predicate_signature_hash,
        }
        if self.selector_hash != stable_content_hash(selector_payload):
            raise ValueError("selector_hash does not match fact-set selector")

        member_lengths = {
            self.member_count,
            len(self.witness_ids),
            len(self.witness_hashes),
            len(self.witness_ordinals),
            len(self.span_ref_hashes),
            len(self.member_atom_kinds),
            len(self.segmentation_methods),
            len(self.canonical_member_starts),
            len(self.canonical_member_ends),
            len(self.packed_member_starts),
            len(self.packed_member_ends),
            len(self.canonical_member_hashes),
            len(self.packed_member_hashes),
            len(self.canonical_member_normalized_hashes),
            len(self.packed_member_normalized_hashes),
            len(self.member_semantic_counts),
        }
        if len(member_lengths) != 1:
            raise ValueError("fact-set member arrays must equal member_count")
        if len(self.witness_ids) != len(set(self.witness_ids)) or self.witness_ordinals != (
            tuple(sorted(set(self.witness_ordinals)))
        ):
            raise ValueError("fact-set witnesses must be unique and source ordered")
        if self.canonical_member_starts != tuple(sorted(self.canonical_member_starts)) or (
            self.packed_member_starts != tuple(sorted(self.packed_member_starts))
        ):
            raise ValueError("fact-set member spans must be source ordered")
        if any(
            start < self.canonical_scope_start or end > self.canonical_scope_end or end <= start
            for start, end in zip(
                self.canonical_member_starts,
                self.canonical_member_ends,
                strict=True,
            )
        ) or any(
            start < self.packed_scope_start or end > self.packed_scope_end or end <= start
            for start, end in zip(
                self.packed_member_starts,
                self.packed_member_ends,
                strict=True,
            )
        ):
            raise ValueError("fact-set member span lies outside selected scope")
        for atom_kind, method in zip(
            self.member_atom_kinds, self.segmentation_methods, strict=True
        ):
            expected = (
                SENTENCE_FACT_SEGMENTATION_METHOD
                if atom_kind == WitnessAtomKind.SENTENCE
                else LIST_ITEM_FACT_SEGMENTATION_METHOD
                if atom_kind == WitnessAtomKind.LIST_ITEM
                else None
            )
            if method != expected:
                raise ValueError("fact-set segmentation method disagrees with atom kind")
        if self.semantic_count != sum(self.member_semantic_counts) or (
            self.semantic_count != len(self.flattened_semantic_hashes)
        ):
            raise ValueError("fact-set semantic arrays are inconsistent")
        if self.mode == FactSetMode.CLOSED_INLINE_COORDINATION:
            if self.member_count != 1 or self.member_semantic_counts[0] < 2:
                raise ValueError("inline fact-set requires one multi-semantic sentence")
        elif self.mode == FactSetMode.SCOPED_TOPIC_PREDICATE_SET:
            if (
                self.topic_group_count < 2
                or any(count != 1 for count in self.member_semantic_counts)
                or self.topic_group_member_counts != tuple(1 for _ in self.topic_group_hashes)
            ):
                raise ValueError("scoped fact-set requires one semantic member per topic")
        elif self.mode == FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET:
            if (
                self.member_count < 2
                or self.topic_group_count != 1
                or self.topic_group_member_counts != (self.member_count,)
                or any(count != 1 for count in self.member_semantic_counts)
                or any(kind != WitnessAtomKind.LIST_ITEM for kind in self.member_atom_kinds)
            ):
                raise ValueError(
                    "scoped bullet-lead fact-set requires complete list-item lead semantics"
                )
        else:
            raise ValueError("fact-set mode is unsupported")

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
            raise ValueError("fact-set title fields must be all present or all absent")
        if self.mode in {
            FactSetMode.SCOPED_TOPIC_PREDICATE_SET,
            FactSetMode.SCOPED_BULLET_LEAD_CLAUSE_SET,
        } and all(value is None for value in title_values):
            raise ValueError("scoped fact-set requires a title boundary")
        if self.canonical_scope_end <= self.canonical_scope_start or (
            self.packed_scope_end <= self.packed_scope_start
        ):
            raise ValueError("fact-set scope spans are invalid")

        payload = self._identity_payload()
        if self.receipt_hash != stable_content_hash(payload):
            raise ValueError("receipt_hash does not match fact-set closure receipt")
        if self.receipt_id != stable_identifier("qfsc_", payload):
            raise ValueError("receipt_id does not match fact-set closure receipt")
        return self

    @classmethod
    def build(
        cls,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[FactSetAuthorityRepresentation],
        catalog_witnesses: Sequence[EvidenceWitness] | None = None,
    ) -> Self:
        """从完整 raw authority 和 COMPLETE catalog 重建唯一事实集合。"""

        question = QuestionContract.from_dict(question.to_dict())
        catalog = WitnessCatalogReceipt.from_dict(catalog.to_dict())
        obligation = QuestionObligation.from_dict(obligation.to_dict())
        typed_witnesses = tuple(EvidenceWitness.from_dict(item.to_dict()) for item in witnesses)
        typed_catalog_witnesses = tuple(
            EvidenceWitness.from_dict(item.to_dict())
            for item in (catalog_witnesses if catalog_witnesses is not None else witnesses)
        )
        authorities = tuple(
            FactSetAuthorityRepresentation(
                evidence_id=item.evidence_id,
                representation_id=item.representation_id,
                canonical_text=item.canonical_text,
                packed_text=item.packed_text,
            )
            for item in authority_representations
        )
        if catalog.universe_status != WitnessUniverseStatus.COMPLETE:
            raise ValueError("fact-set closure requires a complete witness catalog")
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
            raise ValueError("fact-set closure requires an open list obligation")
        if not authorities:
            raise ValueError("fact-set closure requires complete raw authority")
        evidence_ids = tuple(item.evidence_id for item in authorities)
        representation_ids = tuple(item.representation_id for item in authorities)
        if len(evidence_ids) != len(set(evidence_ids)) or len(representation_ids) != len(
            set(representation_ids)
        ):
            raise ValueError("fact-set authority identities must be unique")
        if any(
            _has_truncation_marker(item.canonical_text) or _has_truncation_marker(item.packed_text)
            for item in authorities
        ):
            raise ValueError("fact-set authority contains a truncation marker")
        catalog.validate_witnesses(typed_catalog_witnesses)
        if len({item.witness_id for item in typed_witnesses}) != len(typed_witnesses):
            raise ValueError("fact-set witnesses must be unique")
        expected_fact_witnesses = tuple(
            item
            for item in typed_catalog_witnesses
            if item.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
        )
        if typed_witnesses != expected_fact_witnesses:
            raise ValueError(
                "fact-set witnesses must exactly match all sentence/list-item catalog artifacts"
            )

        authority_by_identity = {
            (item.evidence_id, item.representation_id): (index, item)
            for index, item in enumerate(authorities)
        }
        section_pairs = tuple(
            (_locate_sections(item.canonical_text), _locate_sections(item.packed_text))
            for item in authorities
        )
        views: list[_WitnessView] = []
        for witness in typed_witnesses:
            identity = (witness.span_ref.evidence_id, witness.span_ref.representation_id)
            resolved = authority_by_identity.get(identity)
            if resolved is None:
                raise ValueError("fact-set witness authority is missing from raw authority")
            authority_index, authority = resolved
            witness.span_ref.validate_artifacts(
                evidence_id=authority.evidence_id,
                representation_id=authority.representation_id,
                canonical_text=authority.canonical_text,
                packed_text=authority.packed_text,
            )
            canonical_text = authority.canonical_text[
                witness.span_ref.canonical_start : witness.span_ref.canonical_end
            ]
            packed_text = authority.packed_text[
                witness.span_ref.packed_start : witness.span_ref.packed_end
            ]
            if _normalize_semantic_text(canonical_text) != _normalize_semantic_text(packed_text):
                raise ValueError("fact-set witness canonical and packed semantics differ")
            canonical_sections, packed_sections = section_pairs[authority_index]
            canonical_section = _section_or_structural_title_for_span(
                canonical_sections,
                start=witness.span_ref.canonical_start,
                end=witness.span_ref.canonical_end,
            )
            packed_section = _section_or_structural_title_for_span(
                packed_sections,
                start=witness.span_ref.packed_start,
                end=witness.span_ref.packed_end,
            )
            if (canonical_section is None) != (packed_section is None):
                raise ValueError("fact-set canonical and packed title boundaries differ")
            if canonical_section is None or packed_section is None:
                continue
            views.append(
                _WitnessView(
                    witness=witness,
                    authority_index=authority_index,
                    canonical_text=canonical_text,
                    packed_text=packed_text,
                    canonical_section=canonical_section,
                    packed_section=packed_section,
                )
            )
        typed_views = tuple(views)

        selector_candidates: tuple[tuple[_Selector, tuple[_Candidate, ...]], ...] = tuple(
            (
                selector,
                _find_candidates(
                    selector=selector,
                    views=typed_views,
                    authorities=authorities,
                ),
            )
            for selector in (
                _selector_for_anchor(
                    obligation.information_need,
                    anchor_field="information_need",
                ),
                _selector_for_anchor(question.source_query, anchor_field="source_query"),
            )
            if selector is not None
        )
        selected: tuple[_Selector, _Candidate] | None = None
        for selector, candidates in selector_candidates:
            if len(candidates) == 1:
                selected = selector, candidates[0]
                break
        if selected is None:
            raise ValueError("fact-set closure requires exactly one catalog-wide candidate")
        selector, candidate = selected
        _reject_unclosed_candidate(
            selector=selector,
            candidate=candidate,
            views=typed_views,
            authorities=authorities,
        )

        authority = authorities[candidate.authority_index]
        canonical_hashes = tuple(sha256_digest(item.canonical_text) for item in authorities)
        packed_hashes = tuple(sha256_digest(item.packed_text) for item in authorities)
        authority_payload = {
            "evidence_ids": list(evidence_ids),
            "representation_ids": list(representation_ids),
            "canonical_hashes": list(canonical_hashes),
            "packed_hashes": list(packed_hashes),
        }
        anchor_key_hashes = tuple(
            sha256_digest(_normalize_anchor_text(value)) for value in selector.anchor_match_keys
        )
        topic_group_hashes = tuple(
            sha256_digest(_normalize_anchor_text(value)) for value in selector.topic_groups
        )
        predicate_signature_hash = sha256_digest(selector.relation.value)
        selector_payload = {
            "mode": candidate.mode.value,
            "relation": selector.relation.value,
            "anchor_field": selector.anchor_field,
            "anchor_match_key_hashes": list(anchor_key_hashes),
            "topic_group_hashes": list(topic_group_hashes),
            "topic_group_member_counts": list(candidate.topic_group_member_counts),
            "predicate_signature_hash": predicate_signature_hash,
        }
        canonical_scope_text = authority.canonical_text[
            candidate.canonical_scope.start : candidate.canonical_scope.end
        ]
        packed_scope_text = authority.packed_text[
            candidate.packed_scope.start : candidate.packed_scope.end
        ]
        canonical_outside = (
            authority.canonical_text[: candidate.canonical_scope.start],
            authority.canonical_text[candidate.canonical_scope.end :],
        )
        packed_outside = (
            authority.packed_text[: candidate.packed_scope.start],
            authority.packed_text[candidate.packed_scope.end :],
        )
        canonical_title = (
            authority.canonical_text[
                candidate.canonical_scope.title_start : candidate.canonical_scope.title_end
            ]
            if candidate.canonical_scope.title_start is not None
            and candidate.canonical_scope.title_end is not None
            else None
        )
        packed_title = (
            authority.packed_text[
                candidate.packed_scope.title_start : candidate.packed_scope.title_end
            ]
            if candidate.packed_scope.title_start is not None
            and candidate.packed_scope.title_end is not None
            else None
        )
        members = candidate.members
        segmentation_methods = tuple(
            SENTENCE_FACT_SEGMENTATION_METHOD
            if item.witness.atom_kind == WitnessAtomKind.SENTENCE
            else LIST_ITEM_FACT_SEGMENTATION_METHOD
            for item in members
        )
        flattened_semantics = tuple(
            semantic
            for member_semantics in candidate.member_semantics
            for semantic in member_semantics
        )
        payload: dict[str, Any] = {
            "schema_version": QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION,
            "normalization_version": FACT_SET_SELECTOR_NORMALIZATION_VERSION,
            "question_contract_id": question.contract_id,
            "question_contract_hash": question.contract_hash,
            "catalog_id": catalog.catalog_id,
            "catalog_hash": catalog.catalog_hash,
            "obligation_id": obligation.obligation_id,
            "obligation_hash": obligation.obligation_hash,
            "mode": candidate.mode,
            "relation": selector.relation,
            "anchor_field": selector.anchor_field,
            "anchor_match_key_hashes": anchor_key_hashes,
            "anchor_match_key_count": len(anchor_key_hashes),
            "anchor_match_hash": stable_content_hash(list(anchor_key_hashes)),
            "topic_group_hashes": topic_group_hashes,
            "topic_group_count": len(topic_group_hashes),
            "topic_group_member_counts": candidate.topic_group_member_counts,
            "predicate_signature_hash": predicate_signature_hash,
            "selector_hash": stable_content_hash(selector_payload),
            "catalog_evidence_ids": evidence_ids,
            "catalog_representation_ids": representation_ids,
            "catalog_evidence_content_hashes": canonical_hashes,
            "catalog_packed_text_hashes": packed_hashes,
            "catalog_representation_count": len(authorities),
            "catalog_authority_hash": stable_content_hash(authority_payload),
            "selected_authority_index": candidate.authority_index,
            "evidence_id": authority.evidence_id,
            "evidence_content_hash": canonical_hashes[candidate.authority_index],
            "representation_id": authority.representation_id,
            "authority_candidate_count": 1,
            "canonical_title_hash": (
                sha256_digest(canonical_title) if canonical_title is not None else None
            ),
            "packed_title_hash": sha256_digest(packed_title) if packed_title is not None else None,
            "canonical_title_start": candidate.canonical_scope.title_start,
            "canonical_title_end": candidate.canonical_scope.title_end,
            "packed_title_start": candidate.packed_scope.title_start,
            "packed_title_end": candidate.packed_scope.title_end,
            "canonical_scope_hash": sha256_digest(canonical_scope_text),
            "packed_scope_hash": sha256_digest(packed_scope_text),
            "canonical_scope_start": candidate.canonical_scope.start,
            "canonical_scope_end": candidate.canonical_scope.end,
            "packed_scope_start": candidate.packed_scope.start,
            "packed_scope_end": candidate.packed_scope.end,
            "canonical_outside_scope_hash": stable_content_hash(canonical_outside),
            "packed_outside_scope_hash": stable_content_hash(packed_outside),
            "canonical_non_scope_character_count": sum(
                not character.isspace() for part in canonical_outside for character in part
            ),
            "packed_non_scope_character_count": sum(
                not character.isspace() for part in packed_outside for character in part
            ),
            "member_count": len(members),
            "witness_ids": tuple(item.witness.witness_id for item in members),
            "witness_hashes": tuple(item.witness.witness_hash for item in members),
            "witness_ordinals": tuple(item.witness.ordinal for item in members),
            "span_ref_hashes": tuple(item.witness.span_ref.span_ref_hash for item in members),
            "member_atom_kinds": tuple(item.witness.atom_kind for item in members),
            "segmentation_methods": segmentation_methods,
            "canonical_member_starts": tuple(
                item.witness.span_ref.canonical_start for item in members
            ),
            "canonical_member_ends": tuple(item.witness.span_ref.canonical_end for item in members),
            "packed_member_starts": tuple(item.witness.span_ref.packed_start for item in members),
            "packed_member_ends": tuple(item.witness.span_ref.packed_end for item in members),
            "canonical_member_hashes": tuple(
                sha256_digest(item.canonical_text) for item in members
            ),
            "packed_member_hashes": tuple(sha256_digest(item.packed_text) for item in members),
            "canonical_member_normalized_hashes": tuple(
                sha256_digest(_normalize_semantic_text(item.canonical_text)) for item in members
            ),
            "packed_member_normalized_hashes": tuple(
                sha256_digest(_normalize_semantic_text(item.packed_text)) for item in members
            ),
            "member_semantic_counts": tuple(len(item) for item in candidate.member_semantics),
            "flattened_semantic_hashes": tuple(sha256_digest(item) for item in flattened_semantics),
            "semantic_count": len(flattened_semantics),
            "omission_marker_count": 0,
            "open_marker_count": 0,
            "reference_marker_count": 0,
            "ambiguous_anaphor_count": 0,
            "truncation_marker_count": 0,
            "supplement_candidate_count": 0,
        }
        return cls(
            **payload,
            receipt_id=stable_identifier("qfsc_", payload),
            receipt_hash=stable_content_hash(payload),
        )

    def validate_artifacts(
        self,
        *,
        question: QuestionContract,
        catalog: WitnessCatalogReceipt,
        obligation: QuestionObligation,
        witnesses: Sequence[EvidenceWitness],
        authority_representations: Sequence[FactSetAuthorityRepresentation],
        catalog_witnesses: Sequence[EvidenceWitness] | None = None,
    ) -> Self:
        """从原始 artifact 重建并要求完整字段逐项一致。"""

        rebuilt = type(self).build(
            question=question,
            catalog=catalog,
            obligation=obligation,
            witnesses=witnesses,
            authority_representations=authority_representations,
            catalog_witnesses=catalog_witnesses,
        )
        if rebuilt != self:
            raise ValueError("fact-set closure receipt does not match supplied artifacts")
        return self


def build_query_anchored_scoped_fact_set_closure_receipt(
    *,
    question: QuestionContract,
    catalog: WitnessCatalogReceipt,
    obligation: QuestionObligation,
    witnesses: Sequence[EvidenceWitness],
    authority_representations: Sequence[FactSetAuthorityRepresentation],
    catalog_witnesses: Sequence[EvidenceWitness] | None = None,
) -> QueryAnchoredScopedFactSetClosureReceipt:
    """供 application builder 调用的唯一公开构建入口。"""

    return QueryAnchoredScopedFactSetClosureReceipt.build(
        question=question,
        catalog=catalog,
        obligation=obligation,
        witnesses=witnesses,
        authority_representations=authority_representations,
        catalog_witnesses=catalog_witnesses,
    )


__all__ = [
    "FACT_SET_SELECTOR_NORMALIZATION_VERSION",
    "LIST_ITEM_FACT_SEGMENTATION_METHOD",
    "QUERY_ANCHORED_SCOPED_FACT_SET_CLOSURE_SCHEMA_VERSION",
    "SENTENCE_FACT_SEGMENTATION_METHOD",
    "FactSetAuthorityRepresentation",
    "FactSetMode",
    "FactSetRelation",
    "QueryAnchoredScopedFactSetClosureReceipt",
    "build_query_anchored_scoped_fact_set_closure_receipt",
]
