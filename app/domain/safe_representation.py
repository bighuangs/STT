"""安全表示的版本化确定性派生与无正文坐标回执。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import html
import html.entities
import json
import re
import sys
from typing import Any, Literal, Self
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator
import regex


SAFE_REPRESENTATION_SCHEMA_VERSION = "safe-representation-derivation.v2"
SAFE_REPRESENTATION_SPAN_SCHEMA_VERSION = "safe-representation-span.v2"
SAFE_REPRESENTATION_TRANSFORM_VERSION = "rag-safe-representation.v2"
MARKDOWN_IMAGE_CONTROL_TEXT = "[image omitted]"
MAX_DERIVATION_SPANS = 32_768

INVISIBLE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
DANGEROUS_SCHEME_PATTERN = re.compile(r"(?i)(?:javascript|data|vbscript|file|gopher|ftp)\s*:")
_HTML_ENTITY_PATTERN = re.compile(r"&(?:#[0-9]{1,7};?|#[xX][0-9A-Fa-f]+;?|[^\t\n\f <&#;]{1,32};?)")
_GRAPHEME_PATTERN = regex.compile(r"\X")
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ACTIVE_HTML_OPEN_PATTERN = re.compile(
    r"<\s*(?:script|style|iframe|object|embed|form|svg|math|meta|link)\b",
    re.IGNORECASE,
)


def _find_token(content: str, token: str, start: int) -> int:
    """为线性 scanner 提供可计数的单调查找原语。"""

    return content.find(token, start)


def find_html_comment_spans(content: str) -> tuple[tuple[int, int], ...]:
    """线性定位 HTML 注释；未闭合注释保守覆盖到输入末尾。"""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = _find_token(content, "<!--", cursor)
        if start < 0:
            break
        closing = _find_token(content, "-->", start + 4)
        if closing < 0:
            spans.append((start, len(content)))
            break
        end = closing + 3
        spans.append((start, end))
        cursor = end
    return tuple(spans)


def find_active_html_spans(content: str) -> tuple[tuple[int, int], ...]:
    """线性定位活动 HTML opening tag；畸形未闭合标签覆盖到末尾。"""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(content):
        match = _ACTIVE_HTML_OPEN_PATTERN.search(content, cursor)
        if match is None:
            break
        start = match.start()
        closing = _find_token(content, ">", match.end())
        if closing < 0:
            spans.append((start, len(content)))
            break
        end = closing + 1
        spans.append((start, end))
        cursor = end
    return tuple(spans)


def find_html_tag_spans(content: str) -> tuple[tuple[int, int], ...]:
    """线性定位冻结的普通 HTML tag 子集；未闭合 ``<`` 保持原文。"""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = _find_token(content, "<", cursor)
        if start < 0:
            break
        closing = _find_token(content, ">", start + 1)
        if closing < 0:
            break
        selected_start = start
        if closing - start - 1 > 4_096:
            selected_start = _find_token(content, "<", closing - 4_097)
            if selected_start < 0 or selected_start >= closing:
                cursor = closing + 1
                continue
        if closing - selected_start - 1 >= 1:
            spans.append((selected_start, closing + 1))
        cursor = closing + 1
    return tuple(spans)


def find_markdown_image_spans(content: str) -> tuple[tuple[int, int], ...]:
    """按冻结 Markdown image 子集单调扫描，避免畸形 opener 回溯。"""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = _find_token(content, "![", cursor)
        if start < 0:
            break
        alt_end = _find_token(content, "]", start + 2)
        if alt_end < 0:
            break
        if alt_end + 1 >= len(content) or content[alt_end + 1] != "(":
            cursor = alt_end + 1
            continue
        target_start = alt_end + 2
        target_end = target_start
        while (
            target_end < len(content)
            and content[target_end] != ")"
            and not content[target_end].isspace()
        ):
            target_end += 1
        if target_end == target_start:
            cursor = target_start
            continue
        if target_end < len(content) and content[target_end] == ")":
            spans.append((start, target_end + 1))
            cursor = target_end + 1
            continue
        closing = _find_token(content, ")", target_end)
        if closing < 0:
            break
        title = content[target_end:closing].strip()
        if len(title) >= 2 and title[0] in {"'", '"'} and title[-1] in {"'", '"'}:
            spans.append((start, closing + 1))
            cursor = closing + 1
        else:
            cursor = closing + 1
    return tuple(spans)


def replace_scanned_spans(
    content: str,
    spans: tuple[tuple[int, int], ...],
    replacement: str | Any,
) -> str:
    """用已排序互斥扫描区间做一次线性替换。"""

    if not spans:
        return content
    result: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor or end <= start or end > len(content):
            raise ValueError("scanned spans must be ordered, disjoint and in bounds")
        result.append(content[cursor:start])
        source = content[start:end]
        result.append(str(replacement(source) if callable(replacement) else replacement))
        cursor = end
    result.append(content[cursor:])
    return "".join(result)


def _digest(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _digest(payload)


_HTML_ENTITY_TABLE_HASH = _stable_hash(
    sorted(html.entities.html5.items(), key=lambda item: item[0])
)


class SafeRepresentationRuntimeBinding(BaseModel):
    """冻结 sanitizer 实际依赖的 Unicode 与 entity 运行时表。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str = Field(min_length=1, max_length=32)
    unicode_database_version: str = Field(min_length=1, max_length=32)
    regex_version: str = Field(min_length=1, max_length=32)
    regex_unicode_database_version: str = Field(min_length=1, max_length=32)
    html_entity_table_hash: str = Field(pattern=_SHA256_PATTERN)


def current_safe_representation_runtime_binding() -> SafeRepresentationRuntimeBinding:
    """返回参与安全派生语义的当前运行时身份。"""

    regex_unicode = getattr(
        getattr(regex, "_regex_core", None),
        "unicodedata",
        None,
    )
    regex_unicode_version = str(getattr(regex_unicode, "unidata_version", "unknown"))
    return SafeRepresentationRuntimeBinding(
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        unicode_database_version=unicodedata.unidata_version,
        regex_version=str(regex.__version__),
        regex_unicode_database_version=regex_unicode_version,
        html_entity_table_hash=_HTML_ENTITY_TABLE_HASH,
    )


class SafeSpanRelation(StrEnum):
    """canonical 与 safe span 之间已获准的关系。"""

    EXACT_UTF8 = "exact_utf8"
    NFKC_EQUIVALENT = "nfkc_equivalent"
    SECURITY_SAFE_DERIVED = "security_safe_derived"
    CONTROL_MARKDOWN_IMAGE = "control_markdown_image"
    MAPPING_BUDGET_EXCEEDED = "mapping_budget_exceeded"
    UNPROVEN = "unproven"


class SafeSpanOffsetMapping(StrEnum):
    """safe span 内部坐标是否可按 codepoint 线性换算。"""

    LINEAR = "linear"
    UNIFORM_STEP = "uniform_step"
    ENDPOINTS_ONLY = "endpoints_only"


class SafeRepresentationSpan(BaseModel):
    """不含正文的 canonical↔safe 半开区间。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["safe-representation-span.v2"] = SAFE_REPRESENTATION_SPAN_SCHEMA_VERSION
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(gt=0)
    safe_start: int = Field(ge=0)
    safe_end: int = Field(gt=0)
    relation: SafeSpanRelation
    offset_mapping: SafeSpanOffsetMapping
    canonical_step: int | None = Field(default=None, gt=0)
    safe_step: int | None = Field(default=None, gt=0)
    control_kind: Literal["markdown_image_removed"] | None = None
    canonical_span_hash: str = Field(pattern=_SHA256_PATTERN)
    safe_span_hash: str = Field(pattern=_SHA256_PATTERN)
    span_hash: str = Field(pattern=_SHA256_PATTERN)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_start": self.canonical_start,
            "canonical_end": self.canonical_end,
            "safe_start": self.safe_start,
            "safe_end": self.safe_end,
            "relation": self.relation.value,
            "offset_mapping": self.offset_mapping.value,
            "canonical_step": self.canonical_step,
            "safe_step": self.safe_step,
            "control_kind": self.control_kind,
            "canonical_span_hash": self.canonical_span_hash,
            "safe_span_hash": self.safe_span_hash,
        }

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.canonical_end <= self.canonical_start:
            raise ValueError("safe derivation canonical span must be non-empty")
        if self.safe_end <= self.safe_start:
            raise ValueError("safe derivation output span must be non-empty")
        if (self.relation == SafeSpanRelation.CONTROL_MARKDOWN_IMAGE) != (
            self.control_kind == "markdown_image_removed"
        ):
            raise ValueError("safe derivation control relation and kind disagree")
        if self.offset_mapping == SafeSpanOffsetMapping.LINEAR and (
            self.canonical_end - self.canonical_start != self.safe_end - self.safe_start
        ):
            raise ValueError("linear safe derivation span must preserve codepoint length")
        if self.offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP:
            if self.relation != SafeSpanRelation.NFKC_EQUIVALENT:
                raise ValueError("uniform-step mapping is only valid for NFKC spans")
            if self.canonical_step is None or self.safe_step is None:
                raise ValueError("uniform-step mapping requires both step sizes")
            canonical_length = self.canonical_end - self.canonical_start
            safe_length = self.safe_end - self.safe_start
            if (
                canonical_length % self.canonical_step
                or safe_length % self.safe_step
                or canonical_length // self.canonical_step != safe_length // self.safe_step
            ):
                raise ValueError("uniform-step span lengths do not share a step count")
        elif self.canonical_step is not None or self.safe_step is not None:
            raise ValueError("non-uniform span cannot carry mapping step sizes")
        if self.relation == SafeSpanRelation.EXACT_UTF8:
            if self.offset_mapping != SafeSpanOffsetMapping.LINEAR:
                raise ValueError("exact safe derivation span must use linear offsets")
            if self.canonical_span_hash != self.safe_span_hash:
                raise ValueError("exact safe derivation span hashes must match")
        if (
            self.relation
            in {
                SafeSpanRelation.SECURITY_SAFE_DERIVED,
                SafeSpanRelation.CONTROL_MARKDOWN_IMAGE,
                SafeSpanRelation.MAPPING_BUDGET_EXCEEDED,
                SafeSpanRelation.UNPROVEN,
            }
            and self.offset_mapping != SafeSpanOffsetMapping.ENDPOINTS_ONLY
        ):
            raise ValueError("non-normalization safe span must use endpoint offsets")
        if self.span_hash != _stable_hash(self._identity_payload()):
            raise ValueError("safe derivation span_hash does not match span")
        return self

    @classmethod
    def build(
        cls,
        *,
        canonical_text: str,
        safe_text: str,
        canonical_start: int,
        canonical_end: int,
        safe_start: int,
        safe_end: int,
        relation: SafeSpanRelation,
        offset_mapping: SafeSpanOffsetMapping,
        canonical_step: int | None = None,
        safe_step: int | None = None,
        control_kind: Literal["markdown_image_removed"] | None = None,
    ) -> "SafeRepresentationSpan":
        payload = {
            "schema_version": SAFE_REPRESENTATION_SPAN_SCHEMA_VERSION,
            "canonical_start": canonical_start,
            "canonical_end": canonical_end,
            "safe_start": safe_start,
            "safe_end": safe_end,
            "relation": relation.value,
            "offset_mapping": offset_mapping.value,
            "canonical_step": canonical_step,
            "safe_step": safe_step,
            "control_kind": control_kind,
            "canonical_span_hash": _digest(canonical_text[canonical_start:canonical_end]),
            "safe_span_hash": _digest(safe_text[safe_start:safe_end]),
        }
        return cls(**payload, span_hash=_stable_hash(payload))


class SafeRepresentationDerivationReceipt(BaseModel):
    """把安全表示绑定到 canonical 内容和冻结变换版本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["safe-representation-derivation.v2"] = (
        SAFE_REPRESENTATION_SCHEMA_VERSION
    )
    transform_version: Literal["rag-safe-representation.v2"] = SAFE_REPRESENTATION_TRANSFORM_VERSION
    runtime_binding: SafeRepresentationRuntimeBinding
    canonical_content_hash: str = Field(pattern=_SHA256_PATTERN)
    safe_representation_hash: str = Field(pattern=_SHA256_PATTERN)
    spans: tuple[SafeRepresentationSpan, ...] = Field(default=(), max_length=MAX_DERIVATION_SPANS)
    unproven_transform_count: int = Field(default=0, ge=0)
    receipt_hash: str = Field(pattern=_SHA256_PATTERN)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
            "runtime_binding": self.runtime_binding.model_dump(mode="json"),
            "canonical_content_hash": self.canonical_content_hash,
            "safe_representation_hash": self.safe_representation_hash,
            "spans": [item.model_dump(mode="json") for item in self.spans],
            "unproven_transform_count": self.unproven_transform_count,
        }

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.runtime_binding != current_safe_representation_runtime_binding():
            raise ValueError("safe derivation runtime binding does not match this process")
        previous_safe_end = 0
        for span in self.spans:
            if span.safe_start != previous_safe_end:
                raise ValueError("safe derivation spans must be contiguous and ordered")
            previous_safe_end = span.safe_end
        if self.receipt_hash != _stable_hash(self._identity_payload()):
            raise ValueError("safe derivation receipt_hash does not match receipt")
        return self

    @property
    def has_unproven_transform(self) -> bool:
        return self.unproven_transform_count > 0 or any(
            item.relation == SafeSpanRelation.UNPROVEN for item in self.spans
        )

    @property
    def mapping_budget_exceeded(self) -> bool:
        return any(item.relation == SafeSpanRelation.MAPPING_BUDGET_EXCEEDED for item in self.spans)

    @classmethod
    def from_dict(cls, value: Any) -> "SafeRepresentationDerivationReceipt":
        return cls.model_validate(value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def validate_artifacts(
        self,
        *,
        canonical_text: str,
        safe_text: str,
    ) -> Self:
        if self.canonical_content_hash != _digest(canonical_text):
            raise ValueError("safe derivation canonical content hash mismatch")
        if self.safe_representation_hash != _digest(safe_text):
            raise ValueError("safe derivation safe representation hash mismatch")
        rebuilt = derive_safe_representation(canonical_text)
        if rebuilt.safe_representation != safe_text or rebuilt.receipt != self:
            raise ValueError("safe derivation receipt does not replay supplied artifacts")
        return self


@dataclass(frozen=True, slots=True)
class SafeRepresentationDerivation:
    safe_representation: str
    receipt: SafeRepresentationDerivationReceipt


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    canonical_start: int
    canonical_end: int
    relation: SafeSpanRelation
    offset_mapping: SafeSpanOffsetMapping = SafeSpanOffsetMapping.LINEAR
    canonical_step: int | None = None
    safe_step: int | None = None
    control_kind: Literal["markdown_image_removed"] | None = None


@dataclass(slots=True)
class _PieceCursor:
    pieces: tuple[_Piece, ...]
    piece_index: int = 0
    piece_offset: int = 0
    text_cursor: int = 0

    @classmethod
    def build(cls, pieces: Any) -> "_PieceCursor":
        return cls(tuple(item for item in pieces if item.text))

    def consume_until(self, end: int) -> list[_Piece]:
        """按单调文本游标消费 piece，并保留可证明的局部坐标。"""

        if end < self.text_cursor:
            raise ValueError("piece cursor cannot move backwards")
        consumed: list[_Piece] = []
        while self.text_cursor < end:
            if self.piece_index >= len(self.pieces):
                raise ValueError("piece cursor exceeds source text")
            piece = self.pieces[self.piece_index]
            available = len(piece.text) - self.piece_offset
            take = min(end - self.text_cursor, available)
            consumed.append(
                _piece_slice(
                    piece,
                    self.piece_offset,
                    self.piece_offset + take,
                )
            )
            self.piece_offset += take
            self.text_cursor += take
            if self.piece_offset == len(piece.text):
                self.piece_index += 1
                self.piece_offset = 0
        return consumed


def _initial_pieces(content: str) -> list[_Piece]:
    """按 ``html.unescape`` 的最长合法前缀规则建立初始坐标片段。"""

    pieces: list[_Piece] = []
    cursor = 0
    for match in _HTML_ENTITY_PATTERN.finditer(content):
        if cursor < match.start():
            pieces.append(
                _Piece(
                    content[cursor : match.start()],
                    cursor,
                    match.start(),
                    SafeSpanRelation.EXACT_UTF8,
                )
            )
        source = match.group(0)
        consumed_length = len(source)
        if source.startswith("&#"):
            decoded = html.unescape(source)
        else:
            candidate = source[1:]
            entity_key = next(
                (
                    candidate[:length]
                    for length in range(len(candidate), 1, -1)
                    if candidate[:length] in html.entities.html5
                ),
                None,
            )
            if entity_key is None:
                decoded = source
            else:
                decoded = html.entities.html5[entity_key]
                consumed_length = len(entity_key) + 1
        consumed_source = source[:consumed_length]
        if decoded != consumed_source:
            pieces.append(
                _Piece(
                    decoded,
                    match.start(),
                    match.start() + consumed_length,
                    SafeSpanRelation.SECURITY_SAFE_DERIVED,
                    SafeSpanOffsetMapping.ENDPOINTS_ONLY,
                )
            )
            if consumed_length < len(source):
                pieces.append(
                    _Piece(
                        source[consumed_length:],
                        match.start() + consumed_length,
                        match.end(),
                        SafeSpanRelation.EXACT_UTF8,
                    )
                )
        else:
            pieces.append(
                _Piece(
                    source,
                    match.start(),
                    match.end(),
                    SafeSpanRelation.EXACT_UTF8,
                )
            )
        cursor = match.end()
    if cursor < len(content):
        pieces.append(_Piece(content[cursor:], cursor, len(content), SafeSpanRelation.EXACT_UTF8))
    return pieces


def _nfkc_pieces(pieces: list[_Piece]) -> list[_Piece]:
    cursor = _PieceCursor.build(pieces)
    if not cursor.pieces:
        return []
    text = "".join(item.text for item in cursor.pieces)
    normalized_text = unicodedata.normalize("NFKC", text)
    if all(item.relation == SafeSpanRelation.EXACT_UTF8 for item in cursor.pieces):
        canonical_start = cursor.pieces[0].canonical_start
        canonical_end = cursor.pieces[-1].canonical_end
        if normalized_text == text:
            return [
                _Piece(
                    text,
                    canonical_start,
                    canonical_end,
                    SafeSpanRelation.EXACT_UTF8,
                )
            ]
        if len(normalized_text) == len(text):
            per_codepoint = tuple(unicodedata.normalize("NFKC", item) for item in text)
            if all(len(item) == 1 for item in per_codepoint) and (
                "".join(per_codepoint) == normalized_text
            ):
                return [
                    _Piece(
                        normalized_text,
                        canonical_start,
                        canonical_end,
                        SafeSpanRelation.NFKC_EQUIVALENT,
                        SafeSpanOffsetMapping.LINEAR,
                    )
                ]

    protected: list[tuple[int, int]] = []
    text_offset = 0
    for piece in cursor.pieces:
        piece_end = text_offset + len(piece.text)
        if piece.relation != SafeSpanRelation.EXACT_UTF8 and piece_end - text_offset > 1:
            protected.append((text_offset, piece_end))
        text_offset = piece_end
    protected_index = 0

    def protected_boundary(position: int) -> bool:
        nonlocal protected_index
        while protected_index < len(protected) and protected[protected_index][1] <= position:
            protected_index += 1
        if protected_index >= len(protected):
            return False
        start, end = protected[protected_index]
        return start < position < end

    ranges: list[tuple[int, int]] = []
    cluster_start = 0
    first = True
    for match in _GRAPHEME_PATTERN.finditer(text):
        if first:
            first = False
            continue
        boundary = match.start()
        if not protected_boundary(boundary):
            ranges.append((cluster_start, boundary))
            cluster_start = boundary
    ranges.append((cluster_start, len(text)))

    result: list[_Piece] = []
    for start, end in ranges:
        members = cursor.consume_until(end)
        source = text[start:end]
        normalized = unicodedata.normalize("NFKC", source)
        relations = {item.relation for item in members}
        if SafeSpanRelation.UNPROVEN in relations:
            relation = SafeSpanRelation.UNPROVEN
        elif relations.intersection(
            {
                SafeSpanRelation.SECURITY_SAFE_DERIVED,
                SafeSpanRelation.CONTROL_MARKDOWN_IMAGE,
            }
        ):
            relation = SafeSpanRelation.SECURITY_SAFE_DERIVED
        elif normalized != source or SafeSpanRelation.NFKC_EQUIVALENT in relations:
            relation = SafeSpanRelation.NFKC_EQUIVALENT
        else:
            relation = SafeSpanRelation.EXACT_UTF8
        linear = relation == SafeSpanRelation.EXACT_UTF8
        if relation == SafeSpanRelation.NFKC_EQUIVALENT and len(normalized) == len(source):
            per_codepoint = tuple(unicodedata.normalize("NFKC", item) for item in source)
            linear = all(len(item) == 1 for item in per_codepoint) and (
                "".join(per_codepoint) == normalized
            )
        result.append(
            _Piece(
                normalized,
                min(item.canonical_start for item in members),
                max(item.canonical_end for item in members),
                relation,
                (
                    SafeSpanOffsetMapping.LINEAR
                    if linear
                    else (
                        SafeSpanOffsetMapping.UNIFORM_STEP
                        if relation == SafeSpanRelation.NFKC_EQUIVALENT
                        else SafeSpanOffsetMapping.ENDPOINTS_ONLY
                    )
                ),
                (
                    max(item.canonical_end for item in members)
                    - min(item.canonical_start for item in members)
                    if relation == SafeSpanRelation.NFKC_EQUIVALENT and not linear
                    else None
                ),
                (
                    len(normalized)
                    if relation == SafeSpanRelation.NFKC_EQUIVALENT and not linear
                    else None
                ),
            )
        )
    return result


def _piece_slice(piece: _Piece, start: int, end: int) -> _Piece:
    if start == 0 and end == len(piece.text):
        return piece
    relation = piece.relation
    offset_mapping = piece.offset_mapping
    canonical_start = piece.canonical_start
    canonical_end = piece.canonical_end
    canonical_step = piece.canonical_step
    safe_step = piece.safe_step
    if offset_mapping == SafeSpanOffsetMapping.LINEAR:
        canonical_start += start
        canonical_end = piece.canonical_start + end
    elif (
        offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP
        and canonical_step is not None
        and safe_step is not None
        and start % safe_step == 0
        and end % safe_step == 0
    ):
        canonical_start += (start // safe_step) * canonical_step
        canonical_end = piece.canonical_start + (end // safe_step) * canonical_step
    else:
        relation = SafeSpanRelation.UNPROVEN
        offset_mapping = SafeSpanOffsetMapping.ENDPOINTS_ONLY
        canonical_step = None
        safe_step = None
    return _Piece(
        piece.text[start:end],
        canonical_start,
        canonical_end,
        relation,
        offset_mapping,
        canonical_step,
        safe_step,
        None,
    )


def _replace_pieces(
    pieces: list[_Piece],
    pattern: re.Pattern[str],
    replacement: str | Any,
    *,
    relation: SafeSpanRelation,
    control_kind: Literal["markdown_image_removed"] | None = None,
) -> tuple[list[_Piece], int]:
    cursor = _PieceCursor.build(pieces)
    text = "".join(item.text for item in cursor.pieces)
    result: list[_Piece] = []

    unproven = 0
    for match in pattern.finditer(text):
        result.extend(cursor.consume_until(match.start()))
        matched = cursor.consume_until(match.end())
        if not matched:
            continue
        value = replacement(match) if callable(replacement) else replacement
        if value:
            result.append(
                _Piece(
                    str(value),
                    min(item.canonical_start for item in matched),
                    max(item.canonical_end for item in matched),
                    relation,
                    SafeSpanOffsetMapping.ENDPOINTS_ONLY,
                    None,
                    None,
                    control_kind,
                )
            )
        if relation == SafeSpanRelation.UNPROVEN:
            unproven += 1
    result.extend(cursor.consume_until(len(text)))
    return result, unproven


def _replace_scanned_pieces(
    pieces: list[_Piece],
    spans: tuple[tuple[int, int], ...],
    replacement: str | Any,
    *,
    relation: SafeSpanRelation,
    control_kind: Literal["markdown_image_removed"] | None = None,
) -> tuple[list[_Piece], int]:
    """把线性扫描结果应用到 piece 坐标链。"""

    cursor = _PieceCursor.build(pieces)
    text = "".join(item.text for item in cursor.pieces)
    result: list[_Piece] = []
    unproven = 0
    for start, end in spans:
        if start < cursor.text_cursor or end <= start or end > len(text):
            raise ValueError("scanned replacement spans are invalid")
        result.extend(cursor.consume_until(start))
        matched = cursor.consume_until(end)
        if not matched:
            continue
        source = text[start:end]
        value = replacement(source) if callable(replacement) else replacement
        if value:
            result.append(
                _Piece(
                    str(value),
                    min(item.canonical_start for item in matched),
                    max(item.canonical_end for item in matched),
                    relation,
                    SafeSpanOffsetMapping.ENDPOINTS_ONLY,
                    None,
                    None,
                    control_kind,
                )
            )
        if relation == SafeSpanRelation.UNPROVEN:
            unproven += 1
    result.extend(cursor.consume_until(len(text)))
    return result, unproven


def _merge_proven_pieces(
    pieces: list[_Piece],
    *,
    canonical_text: str,
) -> list[_Piece]:
    """线性压缩 proven runs，并保留 NFKC cluster 的离散合法边界。"""

    source = [piece for piece in pieces if piece.text]
    result: list[_Piece] = []
    cursor = 0
    while cursor < len(source):
        first = source[cursor]
        end = cursor + 1
        if (
            first.offset_mapping == SafeSpanOffsetMapping.LINEAR
            and first.relation in {SafeSpanRelation.EXACT_UTF8, SafeSpanRelation.NFKC_EQUIVALENT}
            and first.control_kind is None
        ):
            while end < len(source):
                candidate = source[end]
                previous = source[end - 1]
                if not (
                    candidate.offset_mapping == SafeSpanOffsetMapping.LINEAR
                    and candidate.relation
                    in {
                        SafeSpanRelation.EXACT_UTF8,
                        SafeSpanRelation.NFKC_EQUIVALENT,
                    }
                    and candidate.control_kind is None
                    and previous.canonical_end == candidate.canonical_start
                ):
                    break
                end += 1
            run = source[cursor:end]
            combined_text = "".join(item.text for item in run)
            canonical_start = run[0].canonical_start
            canonical_end = run[-1].canonical_end
            if (
                unicodedata.normalize("NFKC", canonical_text[canonical_start:canonical_end])
                == combined_text
            ):
                result.append(
                    _Piece(
                        combined_text,
                        canonical_start,
                        canonical_end,
                        (
                            SafeSpanRelation.EXACT_UTF8
                            if all(item.relation == SafeSpanRelation.EXACT_UTF8 for item in run)
                            else SafeSpanRelation.NFKC_EQUIVALENT
                        ),
                        SafeSpanOffsetMapping.LINEAR,
                    )
                )
            else:
                result.extend(run)
        elif (
            first.relation == SafeSpanRelation.NFKC_EQUIVALENT
            and first.offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP
            and first.canonical_step is not None
            and first.safe_step is not None
        ):
            while end < len(source):
                candidate = source[end]
                previous = source[end - 1]
                if not (
                    candidate.relation == SafeSpanRelation.NFKC_EQUIVALENT
                    and candidate.offset_mapping == SafeSpanOffsetMapping.UNIFORM_STEP
                    and candidate.canonical_step == first.canonical_step
                    and candidate.safe_step == first.safe_step
                    and previous.canonical_end == candidate.canonical_start
                ):
                    break
                end += 1
            run = source[cursor:end]
            combined_text = "".join(item.text for item in run)
            canonical_start = run[0].canonical_start
            canonical_end = run[-1].canonical_end
            if (
                unicodedata.normalize("NFKC", canonical_text[canonical_start:canonical_end])
                == combined_text
            ):
                result.append(
                    _Piece(
                        combined_text,
                        canonical_start,
                        canonical_end,
                        SafeSpanRelation.NFKC_EQUIVALENT,
                        SafeSpanOffsetMapping.UNIFORM_STEP,
                        first.canonical_step,
                        first.safe_step,
                    )
                )
            else:
                result.extend(run)
        else:
            result.append(first)
        cursor = end
    return result


def _expected_safe_representation(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(content))
    normalized = INVISIBLE_PATTERN.sub("", normalized)
    normalized = replace_scanned_spans(
        normalized,
        find_html_comment_spans(normalized),
        "",
    )
    normalized = replace_scanned_spans(
        normalized,
        find_active_html_spans(normalized),
        "[active content removed]",
    )
    normalized = replace_scanned_spans(
        normalized,
        find_markdown_image_spans(normalized),
        MARKDOWN_IMAGE_CONTROL_TEXT,
    )
    normalized = replace_scanned_spans(
        normalized,
        find_active_html_spans(normalized),
        "[active content removed]",
    )
    normalized = replace_scanned_spans(
        normalized,
        find_html_tag_spans(normalized),
        html.escape,
    )
    return DANGEROUS_SCHEME_PATTERN.sub("blocked-scheme:", normalized)


def derive_safe_representation(content: str) -> SafeRepresentationDerivation:
    """执行冻结 sanitizer，并同时生成不含正文的可重放派生回执。"""

    pieces = _nfkc_pieces(_initial_pieces(content))
    unproven = 0
    pieces, count = _replace_pieces(
        pieces,
        INVISIBLE_PATTERN,
        "",
        relation=SafeSpanRelation.UNPROVEN,
    )
    unproven += count
    piece_text = "".join(item.text for item in pieces)
    pieces, count = _replace_scanned_pieces(
        pieces,
        find_html_comment_spans(piece_text),
        "",
        relation=SafeSpanRelation.UNPROVEN,
    )
    unproven += count
    piece_text = "".join(item.text for item in pieces)
    pieces, count = _replace_scanned_pieces(
        pieces,
        find_active_html_spans(piece_text),
        "[active content removed]",
        relation=SafeSpanRelation.UNPROVEN,
    )
    unproven += count
    piece_text = "".join(item.text for item in pieces)
    pieces, _ = _replace_scanned_pieces(
        pieces,
        find_markdown_image_spans(piece_text),
        MARKDOWN_IMAGE_CONTROL_TEXT,
        relation=SafeSpanRelation.CONTROL_MARKDOWN_IMAGE,
        control_kind="markdown_image_removed",
    )
    piece_text = "".join(item.text for item in pieces)
    pieces, count = _replace_scanned_pieces(
        pieces,
        find_active_html_spans(piece_text),
        "[active content removed]",
        relation=SafeSpanRelation.UNPROVEN,
    )
    unproven += count
    piece_text = "".join(item.text for item in pieces)
    pieces, _ = _replace_scanned_pieces(
        pieces,
        find_html_tag_spans(piece_text),
        html.escape,
        relation=SafeSpanRelation.SECURITY_SAFE_DERIVED,
    )
    pieces, count = _replace_pieces(
        pieces,
        DANGEROUS_SCHEME_PATTERN,
        "blocked-scheme:",
        relation=SafeSpanRelation.UNPROVEN,
    )
    unproven += count
    pieces = _merge_proven_pieces(pieces, canonical_text=content)
    safe = "".join(item.text for item in pieces)
    expected = _expected_safe_representation(content)
    over_mapping_budget = len(pieces) > MAX_DERIVATION_SPANS
    mapping_budget_exceeded = (
        safe == expected
        and over_mapping_budget
        and unproven == 0
        and all(item.relation != SafeSpanRelation.UNPROVEN for item in pieces)
    )
    if safe != expected or over_mapping_budget:
        safe = expected
        pieces = (
            [
                _Piece(
                    safe,
                    0,
                    len(content),
                    (
                        SafeSpanRelation.MAPPING_BUDGET_EXCEEDED
                        if mapping_budget_exceeded
                        else SafeSpanRelation.UNPROVEN
                    ),
                    SafeSpanOffsetMapping.ENDPOINTS_ONLY,
                )
            ]
            if content and safe
            else []
        )
        if not mapping_budget_exceeded:
            unproven = max(1, unproven)

    spans: list[SafeRepresentationSpan] = []
    safe_cursor = 0
    for piece in pieces:
        safe_end = safe_cursor + len(piece.text)
        spans.append(
            SafeRepresentationSpan.build(
                canonical_text=content,
                safe_text=safe,
                canonical_start=piece.canonical_start,
                canonical_end=piece.canonical_end,
                safe_start=safe_cursor,
                safe_end=safe_end,
                relation=piece.relation,
                offset_mapping=piece.offset_mapping,
                canonical_step=piece.canonical_step,
                safe_step=piece.safe_step,
                control_kind=piece.control_kind,
            )
        )
        safe_cursor = safe_end
    payload = {
        "schema_version": SAFE_REPRESENTATION_SCHEMA_VERSION,
        "transform_version": SAFE_REPRESENTATION_TRANSFORM_VERSION,
        "runtime_binding": current_safe_representation_runtime_binding().model_dump(mode="json"),
        "canonical_content_hash": _digest(content),
        "safe_representation_hash": _digest(safe),
        "spans": [item.model_dump(mode="json") for item in spans],
        "unproven_transform_count": unproven,
    }
    receipt = SafeRepresentationDerivationReceipt(
        **payload,
        receipt_hash=_stable_hash(payload),
    )
    return SafeRepresentationDerivation(safe_representation=safe, receipt=receipt)


__all__ = [
    "DANGEROUS_SCHEME_PATTERN",
    "INVISIBLE_PATTERN",
    "MARKDOWN_IMAGE_CONTROL_TEXT",
    "SAFE_REPRESENTATION_TRANSFORM_VERSION",
    "SafeRepresentationRuntimeBinding",
    "SafeRepresentationDerivation",
    "SafeRepresentationDerivationReceipt",
    "SafeRepresentationSpan",
    "SafeSpanOffsetMapping",
    "SafeSpanRelation",
    "current_safe_representation_runtime_binding",
    "derive_safe_representation",
    "find_active_html_spans",
    "find_html_comment_spans",
    "find_html_tag_spans",
    "find_markdown_image_spans",
    "replace_scanned_spans",
]
