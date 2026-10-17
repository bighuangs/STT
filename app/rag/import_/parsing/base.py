"""多格式文档解析器的最小端口与公共限制。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from app.domain.parsing import ParsedDocument, ParseWarning


@dataclass(slots=True)
class DetectionResult:
    """多证据文件探测结果；MIME 是实际探测值而非客户端声明值。"""

    format: str
    mime_type: str
    warnings: list[ParseWarning] = field(default_factory=list)
    parser_id: str = ""
    parser_version: str = ""


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """本地解析器资源上限；调用方只能通过任务配置显式调整。"""

    max_source_bytes: int = 256 * 1024 * 1024
    max_zip_entries: int = 10_000
    max_zip_uncompressed_bytes: int = 512 * 1024 * 1024
    max_zip_entry_bytes: int = 128 * 1024 * 1024
    max_compression_ratio: float = 200.0
    max_xml_bytes: int = 128 * 1024 * 1024
    max_rows: int = 200_000
    max_columns: int = 16_384
    max_blocks: int = 250_000
    max_cell_chars: int = 1_000_000

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ParserLimits":
        if not config:
            return cls()
        defaults = cls()
        values: dict[str, int | float] = {}
        for name in cls.__dataclass_fields__:
            raw = config.get(name, getattr(defaults, name))
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
                raise ValueError(f"invalid parser limit {name}: expected a positive number")
            default = getattr(defaults, name)
            values[name] = float(raw) if isinstance(default, float) else int(raw)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ParseRequest:
    """解析调用上下文；解析器不得写出 ``work_dir``。"""

    path: Path
    work_dir: Path
    detection: DetectionResult
    source_sha256: str
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def limits(self) -> ParserLimits:
        return ParserLimits.from_config(self.config)


@runtime_checkable
class ParserPort(Protocol):
    """新格式接入注册表所需实现的全部接口。"""

    parser_id: str
    parser_version: str
    formats: tuple[str, ...]
    mime_types: tuple[str, ...]
    extensions: tuple[str, ...]

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """解析本地文件；不得联网或执行文档中的任何内容。"""


__all__ = ["DetectionResult", "ParseRequest", "ParserLimits", "ParserPort"]
