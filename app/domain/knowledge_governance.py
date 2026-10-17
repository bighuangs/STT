"""TODO-15 元数据、标签与知识运营的稳定领域契约。

本模块不依赖 SQLAlchemy、Milvus、FastAPI 或 LLM。控制库实现保存这里定义的
不可变事实；向量库只能消费 :class:`MetadataProjectionBatch`，不能反向成为事实源。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


METADATA_SCHEMA_VERSION = "metadata-schema.v1"
METADATA_VALUE_VERSION = "metadata-value.v1"
TAG_TAXONOMY_VERSION = "tag-taxonomy.v1"
METADATA_RELEASE_VERSION = "metadata-release.v1"
METADATA_FILTER_PLAN_VERSION = "metadata-filter-plan.v1"

MetadataValueType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "date",
    "datetime",
    "string_list",
    "json",
]
MetadataScope = Literal["document", "document_version", "chunk"]
DefinitionStatus = Literal["draft", "published", "retired"]
ProposalStatus = Literal["proposed", "published", "rejected"]
ValueSource = Literal["auto", "manual", "import"]
CatalogMode = Literal["none", "distinct", "range"]
EnhancementRunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "partially_succeeded",
    "failed",
    "cancelled",
]
ProjectionStatus = Literal["pending", "syncing", "in_sync", "failed", "stale"]
AuditAction = Literal[
    "schema_created",
    "schema_published",
    "schema_retired",
    "value_proposed",
    "value_rejected",
    "release_published",
    "release_rolled_back",
    "taxonomy_created",
    "taxonomy_published",
    "tag_proposed",
    "projection_started",
    "projection_succeeded",
    "projection_failed",
    "enhancement_started",
    "enhancement_finished",
]
FilterOperator = Literal[
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts_with",
    "ends_with",
    "exists",
]
FilterBooleanOperator = Literal["and", "or", "not"]
JsonScalar: TypeAlias = str | int | float | bool | None
NormalizedMetadataValue: TypeAlias = (
    str | int | float | bool | tuple[str, ...] | dict[str, Any] | list[Any] | None
)

_FIELD_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,126}[A-Za-z0-9]$|^[A-Za-z]$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SCALAR_TYPES = {"string", "integer", "number", "boolean", "date", "datetime"}
_ORDERED_TYPES = {"integer", "number", "date", "datetime"}
_TEXT_TYPES = {"string"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_governance_id(prefix: str, payload: Any) -> str:
    """为可重放事实生成稳定 ID，不依赖进程随机数。"""

    digest = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:40]
    return f"{prefix}_{digest}"


def _normalize_identifier(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized or not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} is not a safe identifier")
    return normalized


def _normalize_string(value: str, *, name: str, max_length: int = 4096) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be blank")
    if len(normalized) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return normalized


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def _normalize_enum_scalar(
    value: JsonScalar,
    *,
    field_type: MetadataValueType,
) -> JsonScalar:
    """对 enum/default 使用与运行时值相同的严格类型语义。"""

    return normalize_metadata_value(field_type, value)


def normalize_metadata_value(
    value_type: MetadataValueType,
    value: Any,
) -> NormalizedMetadataValue:
    """严格校验并规范化一个元数据值。

    不执行易产生歧义的强制转换：``"1"`` 不是整数，``1`` 也不是布尔值。日期和
    时间输出 ISO-8601 字符串；datetime 必须带时区并归一化为 UTC。
    """

    if value is None:
        return None
    if value_type == "string":
        if not isinstance(value, str):
            raise ValueError("string metadata requires a string")
        return _normalize_string(value, name="metadata string", max_length=32_000)
    if value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer metadata requires an integer")
        return value
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("number metadata requires a number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("number metadata must be finite")
        return normalized
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("boolean metadata requires a boolean")
        return value
    if value_type == "date":
        if isinstance(value, datetime):
            raise ValueError("date metadata cannot be a datetime")
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            raise ValueError("date metadata requires an ISO date")
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError as exc:
            raise ValueError("date metadata requires YYYY-MM-DD") from exc
    if value_type == "datetime":
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("datetime metadata requires ISO-8601") from exc
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError("datetime metadata requires an ISO datetime")
        return _utc(parsed, name="metadata datetime").isoformat().replace("+00:00", "Z")
    if value_type == "string_list":
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
            raise ValueError("string_list metadata requires an array")
        normalized = tuple(
            _normalize_string(str(item), name="string_list item", max_length=4096)
            if isinstance(item, str)
            else ""
            for item in value
        )
        if any(not item for item in normalized):
            raise ValueError("string_list metadata only accepts strings")
        if len(normalized) > 1000:
            raise ValueError("string_list metadata cannot exceed 1000 items")
        if len(normalized) != len(set(normalized)):
            raise ValueError("string_list metadata cannot contain duplicates")
        return normalized
    if value_type == "json":
        try:
            serialized = _canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("json metadata must be finite JSON") from exc
        if len(serialized) > 256_000:
            raise ValueError("json metadata cannot exceed 256000 serialized characters")
        return json.loads(serialized)
    raise ValueError(f"unsupported metadata value type: {value_type}")


class MetadataFieldDefinition(BaseModel):
    """一个版本化字段定义；发布后不可原地修改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=128)
    value_type: MetadataValueType
    description: str = Field(min_length=1, max_length=2000)
    scope: MetadataScope
    required: bool = False
    filterable: bool = False
    enum_values: tuple[JsonScalar, ...] = ()
    default_value: Any | None = None
    catalog_mode: CatalogMode = "none"
    max_distinct_values: int | None = Field(default=None, ge=1, le=100_000)
    compliance_controlled: bool = False

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        normalized = value.strip()
        if not _FIELD_KEY.fullmatch(normalized) or normalized.startswith("_"):
            raise ValueError("metadata field key is unsafe")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return _normalize_string(value, name="description", max_length=2000)

    @model_validator(mode="after")
    def validate_indexability(self) -> "MetadataFieldDefinition":
        if self.filterable and self.value_type not in _SCALAR_TYPES:
            raise ValueError("filterable metadata must use a scalar type")
        if self.catalog_mode == "distinct" and self.value_type not in _SCALAR_TYPES:
            raise ValueError("distinct catalogs require a scalar type")
        if self.catalog_mode == "range" and self.value_type not in _ORDERED_TYPES:
            raise ValueError("range catalogs require an ordered type")
        if self.catalog_mode == "distinct":
            if self.max_distinct_values is None:
                raise ValueError("distinct catalogs require max_distinct_values")
            if self.max_distinct_values > 500:
                raise ValueError("UI distinct catalogs cannot exceed 500 values")
        elif self.max_distinct_values is not None:
            raise ValueError("max_distinct_values is only valid for distinct catalogs")

        normalized_enum = tuple(
            _normalize_enum_scalar(item, field_type=self.value_type) for item in self.enum_values
        )
        if len(normalized_enum) != len(set(_canonical_json(item) for item in normalized_enum)):
            raise ValueError("metadata enum values must be unique")
        if self.enum_values and self.value_type not in _SCALAR_TYPES:
            raise ValueError("enum metadata must use a scalar type")
        if self.default_value is not None:
            normalized_default = normalize_metadata_value(
                self.value_type,
                self.default_value,
            )
            if normalized_enum and normalized_default not in normalized_enum:
                raise ValueError("default metadata value must belong to enum")
        return self

    def normalize_value(self, value: Any) -> NormalizedMetadataValue:
        normalized = normalize_metadata_value(self.value_type, value)
        if self.enum_values and normalized not in tuple(
            _normalize_enum_scalar(item, field_type=self.value_type) for item in self.enum_values
        ):
            raise ValueError(f"value for {self.key} is outside the published enum")
        if self.required and normalized is None:
            raise ValueError(f"required field {self.key} cannot be null")
        return normalized

    def supports_operator(self, operator: FilterOperator) -> bool:
        if operator == "exists":
            return True
        if operator in {"eq", "ne", "in", "not_in"}:
            return self.value_type in _SCALAR_TYPES
        if operator in {"gt", "gte", "lt", "lte"}:
            return self.value_type in _ORDERED_TYPES
        if operator in {"contains", "starts_with", "ends_with"}:
            return self.value_type in _TEXT_TYPES
        return False


class MetadataSchemaVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["metadata-schema.v1"] = METADATA_SCHEMA_VERSION
    schema_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    revision: int = Field(ge=1)
    status: DefinitionStatus = "draft"
    fields: tuple[MetadataFieldDefinition, ...] = Field(min_length=1, max_length=256)
    created_by: str = Field(min_length=1, max_length=128)
    created_at: datetime
    published_at: datetime | None = None

    @field_validator("schema_id", "tenant_id", "kb_id", "created_by")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("created_at", "published_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_schema(self) -> "MetadataSchemaVersion":
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("metadata schema field keys must be unique")
        if self.status in {"published", "retired"} and self.published_at is None:
            raise ValueError("published metadata schema requires published_at")
        if self.status == "draft" and self.published_at is not None:
            raise ValueError("draft metadata schema cannot contain published_at")
        return self

    def field(self, key: str) -> MetadataFieldDefinition:
        for item in self.fields:
            if item.key == key:
                return item
        raise KeyError(key)


class MetadataTarget(BaseModel):
    """元数据/标签的真实作用对象，不允许用缺字段的自由 dict 表示。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    scope: MetadataScope
    document_id: str = Field(min_length=1, max_length=128)
    document_version_id: str | None = Field(default=None, max_length=128)
    chunk_id: str | None = Field(default=None, max_length=128)

    @field_validator(
        "tenant_id",
        "kb_id",
        "document_id",
        "document_version_id",
        "chunk_id",
    )
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _normalize_identifier(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_scope(self) -> "MetadataTarget":
        if self.scope == "document":
            if self.document_version_id is not None or self.chunk_id is not None:
                raise ValueError("document metadata cannot contain version/chunk IDs")
        elif self.scope == "document_version":
            if self.document_version_id is None or self.chunk_id is not None:
                raise ValueError("document_version metadata requires only document_version_id")
        elif self.document_version_id is None or self.chunk_id is None:
            raise ValueError("chunk metadata requires document_version_id and chunk_id")
        return self

    @property
    def target_key(self) -> str:
        values = [
            self.tenant_id,
            self.kb_id,
            self.scope,
            self.document_id,
            self.document_version_id or "-",
            self.chunk_id or "-",
        ]
        return "|".join(values)


class EnhancementReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)

    @field_validator("run_id", "prompt_id", "prompt_version")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("model_id")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        return _normalize_string(value, name="model_id", max_length=512)


class MetadataValueRecord(BaseModel):
    """不可变的建议/审核事实；生效集合由 MetadataRelease 固定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["metadata-value.v1"] = METADATA_VALUE_VERSION
    record_id: str = Field(pattern=r"^mv_[0-9a-f]{40}$")
    schema_id: str = Field(min_length=1, max_length=128)
    schema_revision: int = Field(ge=1)
    field_key: str = Field(min_length=1, max_length=128)
    target: MetadataTarget
    value: Any
    source: ValueSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ProposalStatus = "proposed"
    proposed_by: str = Field(min_length=1, max_length=128)
    proposed_at: datetime
    reviewed_by: str | None = Field(default=None, max_length=128)
    reviewed_at: datetime | None = None
    review_reason: str | None = Field(default=None, max_length=2000)
    enhancement: EnhancementReference | None = None

    @field_validator("schema_id", "proposed_by", "reviewed_by")
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _normalize_identifier(value, name=info.field_name)

    @field_validator("field_key")
    @classmethod
    def validate_field_key(cls, value: str) -> str:
        normalized = value.strip()
        if not _FIELD_KEY.fullmatch(normalized):
            raise ValueError("metadata field key is unsafe")
        return normalized

    @field_validator("proposed_at", "reviewed_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @field_validator("confidence")
    @classmethod
    def finite_confidence(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("metadata confidence must be finite")
        return value

    @model_validator(mode="after")
    def validate_source_and_review(self) -> "MetadataValueRecord":
        if self.source == "auto" and self.confidence is None:
            raise ValueError("automatic metadata requires confidence")
        if self.source != "auto" and self.enhancement is not None:
            raise ValueError("only automatic metadata may reference an enhancement run")
        if self.status == "proposed":
            if self.reviewed_by or self.reviewed_at or self.review_reason:
                raise ValueError("unreviewed proposal cannot contain review fields")
        elif self.reviewed_by is None or self.reviewed_at is None:
            raise ValueError("reviewed metadata requires reviewer and reviewed_at")
        return self


class TagDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tag_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    parent_tag_id: str | None = Field(default=None, max_length=128)
    aliases: tuple[str, ...] = Field(default_factory=tuple, max_length=128)
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    description: str = Field(default="", max_length=2000)
    status: Literal["active", "deprecated"] = "active"
    source: Literal["manual", "import"] = "manual"

    @field_validator("tag_id", "parent_tag_id")
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _normalize_identifier(value, name=info.field_name)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_string(value, name="tag name", max_length=255)

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _normalize_string(item, name="tag alias", max_length=255) for item in value
        )
        if len(normalized) != len(set(item.casefold() for item in normalized)):
            raise ValueError("tag aliases must be unique")
        return normalized

    @field_validator("color")
    @classmethod
    def normalize_color(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _HEX_COLOR.fullmatch(value):
            raise ValueError("tag color must use #RRGGBB")
        return value.upper()


class TagTaxonomyVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tag-taxonomy.v1"] = TAG_TAXONOMY_VERSION
    taxonomy_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    revision: int = Field(ge=1)
    status: DefinitionStatus = "draft"
    tags: tuple[TagDefinition, ...] = Field(min_length=1, max_length=10_000)
    created_by: str = Field(min_length=1, max_length=128)
    created_at: datetime
    published_at: datetime | None = None

    @field_validator("taxonomy_id", "tenant_id", "kb_id", "created_by")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("created_at", "published_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_tree(self) -> "TagTaxonomyVersion":
        ids = [item.tag_id for item in self.tags]
        if len(ids) != len(set(ids)):
            raise ValueError("tag IDs must be unique")
        by_id = {item.tag_id: item for item in self.tags}
        names: dict[str, str] = {}
        for item in self.tags:
            if item.parent_tag_id is not None and item.parent_tag_id not in by_id:
                raise ValueError(f"tag {item.tag_id} references an unknown parent")
            for label in (item.name, *item.aliases):
                folded = label.casefold()
                previous = names.get(folded)
                if previous is not None and previous != item.tag_id:
                    raise ValueError(f"tag name/alias collision: {label}")
                names[folded] = item.tag_id
        for start in ids:
            visited: set[str] = set()
            cursor: str | None = start
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("tag taxonomy cannot contain a cycle")
                visited.add(cursor)
                cursor = by_id[cursor].parent_tag_id
        if self.status in {"published", "retired"} and self.published_at is None:
            raise ValueError("published taxonomy requires published_at")
        if self.status == "draft" and self.published_at is not None:
            raise ValueError("draft taxonomy cannot contain published_at")
        return self

    def resolve(self, value: str) -> TagDefinition | None:
        normalized = value.strip().casefold()
        for tag in self.tags:
            if tag.name.casefold() == normalized:
                return tag
            if normalized in {alias.casefold() for alias in tag.aliases}:
                return tag
        return None


class TagAssignmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str = Field(pattern=r"^ta_[0-9a-f]{40}$")
    taxonomy_id: str = Field(min_length=1, max_length=128)
    taxonomy_revision: int = Field(ge=1)
    tag_id: str = Field(min_length=1, max_length=128)
    target: MetadataTarget
    source: ValueSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ProposalStatus = "proposed"
    proposed_by: str = Field(min_length=1, max_length=128)
    proposed_at: datetime
    reviewed_by: str | None = Field(default=None, max_length=128)
    reviewed_at: datetime | None = None
    review_reason: str | None = Field(default=None, max_length=2000)
    enhancement: EnhancementReference | None = None

    @field_validator(
        "taxonomy_id",
        "tag_id",
        "proposed_by",
        "reviewed_by",
    )
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _normalize_identifier(value, name=info.field_name)

    @field_validator("proposed_at", "reviewed_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_review(self) -> "TagAssignmentRecord":
        if self.source == "auto" and self.confidence is None:
            raise ValueError("automatic tag proposals require confidence")
        if self.source != "auto" and self.enhancement is not None:
            raise ValueError("only automatic tags may reference an enhancement run")
        if self.status == "proposed":
            if self.reviewed_by or self.reviewed_at or self.review_reason:
                raise ValueError("unreviewed tag proposal cannot contain review fields")
        elif self.reviewed_by is None or self.reviewed_at is None:
            raise ValueError("reviewed tag assignment requires reviewer and reviewed_at")
        return self


class UnknownTagCandidate(BaseModel):
    """未命中已发布 taxonomy 的标签建议；它不能直接进入检索投影。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=r"^tc_[0-9a-f]{40}$")
    taxonomy_id: str = Field(min_length=1, max_length=128)
    taxonomy_revision: int = Field(ge=1)
    target: MetadataTarget
    raw_label: str = Field(min_length=1, max_length=255)
    normalized_label: str = Field(min_length=1, max_length=255)
    source: Literal["auto", "import"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    proposed_at: datetime
    enhancement: EnhancementReference | None = None

    @field_validator("taxonomy_id")
    @classmethod
    def normalize_taxonomy_id(cls, value: str) -> str:
        return _normalize_identifier(value, name="taxonomy_id")

    @field_validator("raw_label", "normalized_label")
    @classmethod
    def normalize_labels(cls, value: str, info) -> str:
        return _normalize_string(value, name=info.field_name, max_length=255)

    @field_validator("proposed_at")
    @classmethod
    def validate_proposed_at(cls, value: datetime) -> datetime:
        return _utc(value, name="proposed_at")

    @model_validator(mode="after")
    def validate_source(self) -> "UnknownTagCandidate":
        if self.source == "auto" and self.confidence is None:
            raise ValueError("automatic tag candidates require confidence")
        if self.source == "import" and self.enhancement is not None:
            raise ValueError("imported tag candidates cannot reference an enhancement run")
        return self


class MetadataRelease(BaseModel):
    """一次可回滚的控制库发布快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["metadata-release.v1"] = METADATA_RELEASE_VERSION
    release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    metadata_schema_id: str = Field(min_length=1, max_length=128)
    metadata_schema_revision: int = Field(ge=1)
    taxonomy_id: str | None = Field(default=None, max_length=128)
    taxonomy_revision: int | None = Field(default=None, ge=1)
    value_record_ids: tuple[str, ...] = ()
    tag_assignment_ids: tuple[str, ...] = ()
    source_release_id: str | None = Field(
        default=None,
        pattern=r"^mr_[0-9a-f]{40}$",
    )
    action: Literal["publish", "rollback"] = "publish"
    published_by: str = Field(min_length=1, max_length=128)
    published_at: datetime

    @field_validator(
        "tenant_id",
        "kb_id",
        "metadata_schema_id",
        "taxonomy_id",
        "published_by",
    )
    @classmethod
    def normalize_ids(cls, value: str | None, info) -> str | None:
        return None if value is None else _normalize_identifier(value, name=info.field_name)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        return _utc(value, name="published_at")

    @field_validator("value_record_ids", "tag_assignment_ids")
    @classmethod
    def unique_fact_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_taxonomy_pair(self) -> "MetadataRelease":
        if (self.taxonomy_id is None) != (self.taxonomy_revision is None):
            raise ValueError("taxonomy_id and taxonomy_revision must be set together")
        if self.action == "rollback" and self.source_release_id is None:
            raise ValueError("rollback releases require source_release_id")
        return self


class EnhancementRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    target_scope: MetadataScope
    target_count: int = Field(ge=0)
    status: EnhancementRunStatus
    succeeded_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    started_at: datetime
    finished_at: datetime | None = None
    error_summary: str | None = Field(default=None, max_length=4000)

    @field_validator(
        "run_id",
        "tenant_id",
        "kb_id",
        "job_id",
        "prompt_id",
        "prompt_version",
    )
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @field_validator("cost")
    @classmethod
    def finite_cost(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("enhancement cost must be finite")
        return value

    @model_validator(mode="after")
    def validate_counts_and_terminal_state(self) -> "EnhancementRun":
        if self.succeeded_count + self.failed_count > self.target_count:
            raise ValueError("enhancement result counts exceed target_count")
        terminal = self.status in {"succeeded", "partially_succeeded", "failed", "cancelled"}
        if terminal != (self.finished_at is not None):
            raise ValueError("terminal enhancement state and finished_at must agree")
        return self


class ProjectionState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    projection_name: str = Field(min_length=1, max_length=128)
    control_revision: int = Field(ge=0)
    projected_revision: int = Field(ge=0)
    status: ProjectionStatus
    attempt: int = Field(default=0, ge=0)
    last_started_at: datetime | None = None
    last_succeeded_at: datetime | None = None
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=2000)

    @field_validator("tenant_id", "kb_id", "projection_name")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("last_started_at", "last_succeeded_at")
    @classmethod
    def validate_times(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, name=info.field_name)

    @model_validator(mode="after")
    def validate_projection(self) -> "ProjectionState":
        if self.projected_revision > self.control_revision:
            raise ValueError("projected revision cannot exceed control revision")
        if self.status == "in_sync" and self.projected_revision != self.control_revision:
            raise ValueError("in_sync projection revisions must match")
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed projection requires error_code")
        if self.status != "failed" and (self.error_code or self.error_message):
            raise ValueError("only failed projection may contain error details")
        return self


class MetadataAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(pattern=r"^ma_[0-9a-f]{40}$")
    tenant_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(min_length=1, max_length=128)
    action: AuditAction
    actor_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    object_type: str = Field(min_length=1, max_length=128)
    object_id: str = Field(min_length=1, max_length=256)
    revision: int | None = Field(default=None, ge=1)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id", "kb_id", "actor_id")
    @classmethod
    def normalize_ids(cls, value: str, info) -> str:
        return _normalize_identifier(value, name=info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, name="occurred_at")


class MetadataCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_key: str = Field(min_length=1, max_length=128)
    value_type: MetadataValueType
    scope: MetadataScope
    catalog_mode: CatalogMode
    distinct_values: tuple[JsonScalar, ...] = ()
    minimum: JsonScalar = None
    maximum: JsonScalar = None
    distinct_count: int = Field(ge=0)
    truncated: bool = False


class MetadataCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    kb_id: str
    release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    revision: int = Field(ge=1)
    entries: tuple[MetadataCatalogEntry, ...]
    built_from: Literal["control_store"] = "control_store"


class MetadataFieldQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_key: str
    scope: MetadataScope
    expected_target_count: int = Field(ge=0)
    published_target_count: int = Field(ge=0)
    proposed_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    manual_published_count: int = Field(ge=0)
    automatic_published_count: int = Field(ge=0)
    average_automatic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)


class MetadataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    kb_id: str
    release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    revision: int = Field(ge=1)
    fields: tuple[MetadataFieldQuality, ...]
    required_field_coverage: float = Field(ge=0.0, le=1.0)
    pending_review_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    projection_lag: int = Field(ge=0)


class MetadataMigrationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str | None = Field(default=None, pattern=r"^mv_[0-9a-f]{40}$")
    field_key: str
    code: Literal[
        "FIELD_REMOVED",
        "SCOPE_CHANGED",
        "VALUE_INCOMPATIBLE",
        "REQUIRED_FIELD_UNPOPULATED",
    ]
    message: str = Field(min_length=1, max_length=2000)


class MetadataMigrationPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    kb_id: str
    source_release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    source_schema_revision: int = Field(ge=1)
    target_schema_revision: int = Field(ge=1)
    compatible_record_ids: tuple[str, ...]
    issues: tuple[MetadataMigrationIssue, ...]
    publishable: bool


class MappedMetadataPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["predicate"] = "predicate"
    field_key: str
    projection_path: str
    operator: FilterOperator
    value: Any
    value_type: MetadataValueType | Literal["identifier"]


class MappedMetadataGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["group"] = "group"
    operator: FilterBooleanOperator
    conditions: tuple["MappedMetadataPredicate | MappedMetadataGroup", ...]


MappedMetadataFilter: TypeAlias = MappedMetadataPredicate | MappedMetadataGroup


class MetadataFilterPlan(BaseModel):
    """经过已发布 schema 映射的引擎无关过滤计划。

    ``mandatory_*`` 由服务端授权上下文写入，永远不从客户端 AST 读取。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["metadata-filter-plan.v1"] = METADATA_FILTER_PLAN_VERSION
    tenant_id: str
    kb_id: str
    release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    control_revision: int = Field(ge=1)
    mandatory_tenant_id: str
    mandatory_kb_id: str
    filter: MappedMetadataFilter | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "MetadataFilterPlan":
        if self.tenant_id != self.mandatory_tenant_id:
            raise ValueError("mandatory tenant scope cannot differ from plan tenant")
        if self.kb_id != self.mandatory_kb_id:
            raise ValueError("mandatory KB scope cannot differ from plan KB")
        return self


MappedMetadataGroup.model_rebuild()


class ProjectionValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(pattern=r"^mv_[0-9a-f]{40}$")
    target: MetadataTarget
    field_key: str
    value: Any
    value_type: MetadataValueType


class ProjectionTag(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_id: str = Field(pattern=r"^ta_[0-9a-f]{40}$")
    target: MetadataTarget
    tag_id: str
    tag_name: str
    tag_path: tuple[str, ...]


class MetadataProjectionBatch(BaseModel):
    """控制库 release 到检索投影的幂等载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(pattern=r"^mp_[0-9a-f]{40}$")
    tenant_id: str
    kb_id: str
    release_id: str = Field(pattern=r"^mr_[0-9a-f]{40}$")
    control_revision: int = Field(ge=1)
    values: tuple[ProjectionValue, ...]
    tags: tuple[ProjectionTag, ...]
