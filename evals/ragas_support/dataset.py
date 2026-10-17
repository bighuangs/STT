"""把冻结评测集规范化为 RAGAS 与确定性检索评测共用的数据资产。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO


PROTECTED_CASE_ID = "P3500-EASY-002"
DATASET_SCHEMA_VERSION = "stt.ragas.dataset.v1"
SAMPLE_SCHEMA_VERSION = "stt.ragas.sample.v1"
ANNOTATION_GUIDELINE_VERSION = "stt-ragas-normalization-v1"

OUTPUT_FILENAMES = (
    "cases.csv",
    "relevance-judgments.csv",
    "claims.csv",
    "samples.jsonl",
    "dataset-stats.json",
    "SHA256SUMS",
)

CASES_FIELDS = (
    "case_id",
    "parent_case_id",
    "dataset_version",
    "split",
    "slice_id",
    "difficulty",
    "answerability_condition",
    "domain_language_profile",
    "lexical_gap_level",
    "reasoning_structure",
    "noise_conflict_level",
    "evidence_position_bucket",
    "context_length_bucket",
    "citation_requirement",
    "freshness_class",
    "domain",
    "task",
    "language",
    "query",
    "conversation_context",
    "knowledge_base_version",
    "answerable",
    "gold_answer",
    "expected_behavior",
    "gold_provenance_summary",
    "source_type",
    "source_ref",
    "risk_level",
    "traffic_weight",
    "annotation_guideline_version",
    "reviewer_ids",
    "adjudication_status",
    "created_at",
    "metadata_json",
)

RELEVANCE_FIELDS = (
    "case_id",
    "knowledge_base_version",
    "document_id",
    "chunk_id",
    "relevance_grade",
    "evidence_start",
    "evidence_end",
    "evidence_text_hash",
    "source_uri",
    "source_version",
    "judge_type",
    "judge_id",
    "annotation_guideline_version",
    "adjudication_status",
    "reviewed_at",
    "notes",
)

CLAIMS_FIELDS = (
    "case_id",
    "claim_id",
    "claim_text",
    "required",
    "importance",
    "reference_document_id",
    "reference_chunk_id",
    "reference_evidence_start",
    "reference_evidence_end",
    "verification_method",
    "annotation_guideline_version",
    "reviewer_ids",
    "adjudication_status",
    "notes",
)

LEXICAL_REQUIREMENT_NOTE = (
    "expected_terms 仅表示 lexical_requirement，不构成完整 Gold claim"
)


@dataclass(frozen=True, slots=True)
class _CaseEnvelope:
    case_id: str
    payload: dict[str, Any] | None
    protected: bool


@dataclass(frozen=True, slots=True)
class _DatasetInput:
    path: Path
    metadata: dict[str, Any]
    cases: tuple[_CaseEnvelope, ...]
    sha256: str


class _CharReader:
    """提供单字符回退的流式读取器，保护用例正文不会被累计到内存。"""

    def __init__(self, source: TextIO) -> None:
        self._source = source
        self._pushed: str | None = None

    def read(self) -> str:
        if self._pushed is not None:
            value = self._pushed
            self._pushed = None
            return value
        return self._source.read(1)

    def push(self, value: str) -> None:
        if not value or self._pushed is not None:
            raise RuntimeError("字符回退状态非法")
        self._pushed = value

    def next_non_whitespace(self) -> str:
        while True:
            value = self.read()
            if not value or not value.isspace():
                return value


def _expect(reader: _CharReader, expected: str) -> None:
    actual = reader.next_non_whitespace()
    if actual != expected:
        raise ValueError(f"JSON 结构错误：期望 {expected!r}，实际 {actual!r}")


def _read_json_string(reader: _CharReader, *, first: str | None = None) -> tuple[str, str]:
    opening = first if first is not None else reader.next_non_whitespace()
    if opening != '"':
        raise ValueError("JSON 字段名或字符串值必须以双引号开始")
    raw = ['"']
    escaped = False
    while True:
        value = reader.read()
        if not value:
            raise ValueError("JSON 字符串未闭合")
        raw.append(value)
        if escaped:
            escaped = False
        elif value == "\\":
            escaped = True
        elif value == '"':
            break
    token = "".join(raw)
    decoded = json.loads(token)
    if not isinstance(decoded, str):
        raise ValueError("JSON 字符串解析结果非法")
    return token, decoded


def _read_json_value_raw(reader: _CharReader) -> str:
    """读取一个普通 JSON 值；调用方仅对非保护数据执行反序列化。"""

    first = reader.next_non_whitespace()
    if not first:
        raise ValueError("JSON 值缺失")
    if first == '"':
        raw, _ = _read_json_string(reader, first=first)
        return raw
    if first in "[{":
        expected_closers = ["]" if first == "[" else "}"]
        raw = [first]
        in_string = False
        escaped = False
        while expected_closers:
            value = reader.read()
            if not value:
                raise ValueError("JSON 复合值未闭合")
            raw.append(value)
            if in_string:
                if escaped:
                    escaped = False
                elif value == "\\":
                    escaped = True
                elif value == '"':
                    in_string = False
                continue
            if value == '"':
                in_string = True
            elif value == "[":
                expected_closers.append("]")
            elif value == "{":
                expected_closers.append("}")
            elif value in "]}":
                expected = expected_closers.pop()
                if value != expected:
                    raise ValueError("JSON 复合值括号不匹配")
        return "".join(raw)

    raw = [first]
    while True:
        value = reader.read()
        if not value:
            break
        if value in ",}]":
            reader.push(value)
            break
        raw.append(value)
    return "".join(raw).rstrip()


def _read_object_tail(reader: _CharReader, *, collect: bool) -> str:
    """读取 Case 首字段之后的对象尾部；保护路径只做括号扫描。"""

    expected_closers = ["}"]
    raw: list[str] = []
    in_string = False
    escaped = False
    while expected_closers:
        value = reader.read()
        if not value:
            raise ValueError("Case JSON 对象未闭合")
        if collect:
            raw.append(value)
        if in_string:
            if escaped:
                escaped = False
            elif value == "\\":
                escaped = True
            elif value == '"':
                in_string = False
            continue
        if value == '"':
            in_string = True
        elif value == "{":
            expected_closers.append("}")
        elif value == "[":
            expected_closers.append("]")
        elif value in "}]":
            expected = expected_closers.pop()
            if value != expected:
                raise ValueError("Case JSON 对象括号不匹配")
    return "".join(raw)


def _read_case(reader: _CharReader) -> _CaseEnvelope:
    """先且仅先解析首字段 case_id，再决定是否允许解析剩余对象。"""

    opening = reader.next_non_whitespace()
    if opening != "{":
        raise ValueError("cases 数组成员必须是 JSON 对象")
    key_raw, key = _read_json_string(reader)
    if key != "case_id":
        raise ValueError("每个 Case 的首字段必须是 case_id；已按保护策略失败关闭")
    _expect(reader, ":")
    case_id_raw = _read_json_value_raw(reader)
    try:
        case_id = json.loads(case_id_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Case 的 case_id 不是合法 JSON 字符串") from exc
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Case 的 case_id 必须是非空字符串")

    prefix = "{" + key_raw + ":" + case_id_raw
    if case_id == PROTECTED_CASE_ID:
        _read_object_tail(reader, collect=False)
        return _CaseEnvelope(case_id=case_id, payload=None, protected=True)

    tail = _read_object_tail(reader, collect=True)
    try:
        payload = json.loads(prefix + tail)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Case {case_id} 不是合法 JSON 对象") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Case {case_id} 必须是 JSON 对象")
    return _CaseEnvelope(case_id=case_id, payload=payload, protected=False)


def _read_cases_array(reader: _CharReader) -> tuple[_CaseEnvelope, ...]:
    _expect(reader, "[")
    cases: list[_CaseEnvelope] = []
    next_value = reader.next_non_whitespace()
    if next_value == "]":
        return ()
    reader.push(next_value)
    while True:
        cases.append(_read_case(reader))
        separator = reader.next_non_whitespace()
        if separator == "]":
            break
        if separator != ",":
            raise ValueError("cases 数组成员之间缺少逗号")
    return tuple(cases)


def _read_dataset(path: Path) -> _DatasetInput:
    metadata: dict[str, Any] = {}
    cases: tuple[_CaseEnvelope, ...] | None = None
    with path.open("r", encoding="utf-8") as source:
        reader = _CharReader(source)
        _expect(reader, "{")
        next_value = reader.next_non_whitespace()
        if next_value == "}":
            raise ValueError(f"数据集为空：{path}")
        reader.push(next_value)
        while True:
            _, key = _read_json_string(reader)
            _expect(reader, ":")
            if key == "cases":
                if cases is not None:
                    raise ValueError(f"数据集重复声明 cases：{path}")
                cases = _read_cases_array(reader)
            else:
                raw_value = _read_json_value_raw(reader)
                try:
                    metadata[key] = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"数据集字段 {key!r} 不是合法 JSON：{path}") from exc
            separator = reader.next_non_whitespace()
            if separator == "}":
                break
            if separator != ",":
                raise ValueError(f"数据集字段之间缺少逗号：{path}")
        if reader.next_non_whitespace():
            raise ValueError(f"数据集顶层对象之后存在额外内容：{path}")
    if cases is None:
        raise ValueError(f"数据集缺少 cases 数组：{path}")
    return _DatasetInput(
        path=path,
        metadata=metadata,
        cases=cases,
        sha256=_sha256_file(path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dataset_identity(dataset: _DatasetInput) -> tuple[str, str, str]:
    dataset_key = str(dataset.metadata.get("dataset_key") or dataset.path.stem)
    dataset_version = str(dataset.metadata.get("version") or "unknown")
    difficulty = str(dataset.metadata.get("difficulty") or "").strip().lower()
    if not difficulty:
        difficulty = "safety" if dataset_key.lower().startswith("hak180") else "unspecified"
    return dataset_key, dataset_version, difficulty


def _source_version(metadata: dict[str, Any]) -> str:
    source = metadata.get("source")
    if not isinstance(source, dict):
        return "unknown"
    return str(source.get("content_sha256") or source.get("sha256") or "unknown")


def _source_uri(metadata: dict[str, Any], dataset_key: str) -> str:
    source = metadata.get("source")
    if isinstance(source, dict):
        for key in ("relative_path", "uri", "logical_name", "original_name"):
            value = str(source.get(key) or "").strip()
            if value:
                return value.replace("\\", "/")
    return dataset_key


def _reasoning_structure(case: dict[str, Any]) -> str:
    metadata = case.get("metadata")
    if isinstance(metadata, dict):
        value = str(metadata.get("type") or "").strip()
        if value:
            return value
    tags = {tag.lower() for tag in _strings(case.get("tags"))}
    for candidate in ("multiturn", "cross-span", "table", "figure", "single-span"):
        if candidate in tags:
            return candidate.replace("-", "_")
    return "unspecified"


def _context_length_bucket(evidence: Sequence[dict[str, Any]]) -> str:
    length = sum(len(str(item.get("text") or "")) for item in evidence)
    if length == 0:
        return "none"
    if length <= 800:
        return "short"
    if length <= 2400:
        return "medium"
    return "long"


def _risk_level(case: dict[str, Any]) -> str:
    sensitivity = str(case.get("sensitivity") or "internal").strip().lower()
    metadata = case.get("metadata")
    critical = isinstance(metadata, dict) and bool(metadata.get("critical"))
    if sensitivity == "restricted":
        return "restricted"
    return "high" if critical else "standard"


def _evidence_records(case: dict[str, Any]) -> list[dict[str, str]]:
    evidence = case.get("relevant_evidence")
    evidence_rows = [item for item in evidence or [] if isinstance(item, dict)]
    listed_chunk_ids = _strings(case.get("relevant_chunk_ids"))
    listed_document_ids = _strings(case.get("relevant_document_ids"))
    listed_document_names = _strings(case.get("relevant_document_names"))
    count = max(len(evidence_rows), len(listed_chunk_ids))
    records: list[dict[str, str]] = []
    for index in range(count):
        item = evidence_rows[index] if index < len(evidence_rows) else {}
        chunk_id = str(item.get("chunk_id") or item.get("evidence_id") or "").strip()
        if not chunk_id and index < len(listed_chunk_ids):
            chunk_id = listed_chunk_ids[index]
        if not chunk_id:
            raise ValueError(f"Case {case['case_id']} 的相关证据缺少 chunk/evidence id")
        document_id = str(item.get("document_id") or "").strip()
        if not document_id and listed_document_ids:
            document_id = listed_document_ids[min(index, len(listed_document_ids) - 1)]
        if not document_id and listed_document_names:
            document_id = listed_document_names[min(index, len(listed_document_names) - 1)]
        text = str(item.get("text") or "")
        records.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "",
                "original_content_hash": str(item.get("content_hash") or ""),
            }
        )
    return records


def _protected_case_row(
    *, dataset_key: str, dataset_version: str, difficulty: str
) -> dict[str, Any]:
    metadata = {
        "content_parsed": False,
        "excluded_from_judge": True,
        "excluded_from_samples": True,
        "historical_denominator": True,
        "protected": True,
        "source_dataset": dataset_key,
    }
    row = {field: "" for field in CASES_FIELDS}
    row.update(
        {
            "case_id": PROTECTED_CASE_ID,
            "dataset_version": dataset_version,
            "split": "test",
            "slice_id": "protected_excluded",
            "difficulty": difficulty,
            "answerability_condition": "protected_unparsed",
            "domain_language_profile": "protected",
            "citation_requirement": "not_evaluated",
            "freshness_class": "frozen",
            "domain": "protected",
            "task": "historical_denominator_only",
            "expected_behavior": "exclude_from_content_processing_and_judge",
            "source_type": "protected_placeholder",
            "source_ref": dataset_key,
            "risk_level": "protected",
            "traffic_weight": "1",
            "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
            "adjudication_status": "protected_by_policy",
            "metadata_json": _json_text(metadata),
        }
    )
    return row


def _normal_case_rows(
    case: dict[str, Any], dataset: _DatasetInput
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_key, dataset_version, dataset_difficulty = _dataset_identity(dataset)
    case_id = str(case.get("case_id") or "")
    if not case_id:
        raise ValueError(f"数据集 {dataset.path} 存在空 case_id")
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    difficulty = str(metadata.get("difficulty") or dataset_difficulty).strip().lower()
    answerable = not bool(case.get("unanswerable"))
    evidence = _evidence_records(case)
    expected_terms = _strings(case.get("expected_terms"))
    tags = _strings(case.get("tags"))
    source_version = _source_version(dataset.metadata)
    source_uri = _source_uri(dataset.metadata, dataset_key)
    conversation = case.get("conversation") if isinstance(case.get("conversation"), list) else []
    sensitivity = str(case.get("sensitivity") or "internal").strip().lower()
    judge_eligible = sensitivity not in {"protected", "restricted"}
    slice_id = f"{dataset_difficulty}:{'answerable' if answerable else 'unanswerable'}"

    normalized_metadata = {
        "expected_terms_semantics": "lexical_requirement_not_complete_gold_claim",
        "judge_eligible": judge_eligible,
        "original_metadata": metadata,
        "source_dataset": dataset_key,
        "source_file_sha256": dataset.sha256,
        "tags": tags,
    }
    case_row = {field: "" for field in CASES_FIELDS}
    case_row.update(
        {
            "case_id": case_id,
            "dataset_version": dataset_version,
            "split": "test",
            "slice_id": slice_id,
            "difficulty": difficulty,
            "answerability_condition": "answerable" if answerable else "unanswerable",
            "domain_language_profile": "zh-CN",
            "lexical_gap_level": str(metadata.get("lexical_gap_level") or "unspecified"),
            "reasoning_structure": _reasoning_structure(case),
            "noise_conflict_level": str(metadata.get("noise_conflict_level") or "none"),
            "evidence_position_bucket": "reference_only",
            "context_length_bucket": _context_length_bucket(
                [item for item in case.get("relevant_evidence") or [] if isinstance(item, dict)]
            ),
            "citation_requirement": "required" if evidence else "not_applicable",
            "freshness_class": "frozen",
            "domain": "printer_manual",
            "task": _reasoning_structure(case),
            "language": "zh-CN",
            "query": str(case.get("query") or ""),
            "conversation_context": _json_text(conversation),
            "knowledge_base_version": source_version,
            "answerable": str(answerable).lower(),
            "gold_answer": str(case.get("reference_answer") or ""),
            "expected_behavior": "grounded_answer" if answerable else "grounded_abstention",
            "gold_provenance_summary": _json_text(
                {
                    "document_ids": list(dict.fromkeys(item["document_id"] for item in evidence)),
                    "reference_context_ids": [item["chunk_id"] for item in evidence],
                }
            ),
            "source_type": "frozen_evaluation_dataset",
            "source_ref": dataset_key,
            "risk_level": _risk_level(case),
            "traffic_weight": "1",
            "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
            "adjudication_status": "source_frozen",
            "metadata_json": _json_text(normalized_metadata),
        }
    )

    qrels = [
        {
            "case_id": case_id,
            "knowledge_base_version": source_version,
            "document_id": item["document_id"],
            "chunk_id": item["chunk_id"],
            "relevance_grade": "1",
            "evidence_start": "",
            "evidence_end": "",
            "evidence_text_hash": item["text_hash"],
            "source_uri": source_uri,
            "source_version": source_version,
            "judge_type": "authored_gold",
            "judge_id": dataset_key,
            "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
            "adjudication_status": "source_frozen",
            "reviewed_at": "",
            "notes": (
                _json_text({"original_content_hash": item["original_content_hash"]})
                if item["original_content_hash"]
                else ""
            ),
        }
        for item in evidence
    ]

    claims: list[dict[str, Any]] = []
    for index, term in enumerate(expected_terms, start=1):
        matching_evidence = next(
            (item for item in evidence if term in item["text"]),
            evidence[0] if evidence else None,
        )
        claims.append(
            {
                "case_id": case_id,
                "claim_id": f"{case_id}-LEX-{index:03d}",
                "claim_text": term,
                "required": "true",
                "importance": "1.0",
                "reference_document_id": (
                    matching_evidence["document_id"] if matching_evidence else ""
                ),
                "reference_chunk_id": (
                    matching_evidence["chunk_id"] if matching_evidence else ""
                ),
                "reference_evidence_start": "",
                "reference_evidence_end": "",
                "verification_method": "lexical_requirement",
                "annotation_guideline_version": ANNOTATION_GUIDELINE_VERSION,
                "reviewer_ids": "",
                "adjudication_status": "source_frozen",
                "notes": LEXICAL_REQUIREMENT_NOTE,
            }
        )

    sample = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "case_id": case_id,
        "source_dataset": dataset_key,
        "dataset_version": dataset_version,
        "difficulty": difficulty,
        "slice_id": slice_id,
        "user_input": str(case.get("query") or ""),
        "conversation": conversation,
        "reference": str(case.get("reference_answer") or ""),
        "answerable": answerable,
        "reference_context_ids": [item["chunk_id"] for item in evidence],
        "reference_contexts": [item["text"] for item in evidence],
        "expected_terms": expected_terms,
        "tags": tags,
        "sensitivity": sensitivity,
        "judge_eligible": judge_eligible,
    }
    return case_row, qrels, claims, sample


def _csv_content(fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_ragas_dataset(
    dataset_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """规范化数据集并写入不可含保护用例正文的本地评测资产。"""

    paths = [Path(path).resolve() for path in dataset_paths]
    if not paths:
        raise ValueError("至少需要一个输入数据集")
    if len(paths) != len(set(paths)):
        raise ValueError("输入数据集路径不得重复")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = {name: destination / name for name in OUTPUT_FILENAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "拒绝覆盖既有评测资产；如确需替换请显式传入 overwrite=True："
            + ", ".join(str(path) for path in existing)
        )

    datasets = [_read_dataset(path) for path in paths]
    case_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    by_dataset: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    protected_count = 0
    judge_eligible_count = 0
    answerable_count = 0
    unanswerable_count = 0

    for dataset in datasets:
        dataset_key, dataset_version, dataset_difficulty = _dataset_identity(dataset)
        dataset_stats = {
            "source_cases": 0,
            "historical_denominator": 0,
            "protected_placeholders": 0,
            "evaluation_eligible_samples": 0,
            "judge_eligible_samples": 0,
        }
        if dataset_key in by_dataset:
            raise ValueError(f"dataset_key 重复：{dataset_key}")
        by_dataset[dataset_key] = dataset_stats
        for envelope in dataset.cases:
            if envelope.case_id in seen_case_ids:
                raise ValueError(f"case_id 重复：{envelope.case_id}")
            seen_case_ids.add(envelope.case_id)
            dataset_stats["source_cases"] += 1
            dataset_stats["historical_denominator"] += 1
            if envelope.protected:
                protected_count += 1
                dataset_stats["protected_placeholders"] += 1
                case_rows.append(
                    _protected_case_row(
                        dataset_key=dataset_key,
                        dataset_version=dataset_version,
                        difficulty=dataset_difficulty,
                    )
                )
                difficulty_stats = by_difficulty.setdefault(
                    dataset_difficulty,
                    {
                        "historical_denominator": 0,
                        "evaluation_eligible_samples": 0,
                        "judge_eligible_samples": 0,
                    },
                )
                difficulty_stats["historical_denominator"] += 1
                continue

            if envelope.payload is None:
                raise AssertionError("非保护 Case 不得缺少 payload")
            case_row, qrels, claims, sample = _normal_case_rows(envelope.payload, dataset)
            case_rows.append(case_row)
            qrel_rows.extend(qrels)
            claim_rows.extend(claims)
            samples.append(sample)
            dataset_stats["evaluation_eligible_samples"] += 1
            if sample["judge_eligible"]:
                dataset_stats["judge_eligible_samples"] += 1
                judge_eligible_count += 1
            if sample["answerable"]:
                answerable_count += 1
            else:
                unanswerable_count += 1
            difficulty = str(sample["difficulty"])
            difficulty_stats = by_difficulty.setdefault(
                difficulty,
                {
                    "historical_denominator": 0,
                    "evaluation_eligible_samples": 0,
                    "judge_eligible_samples": 0,
                },
            )
            difficulty_stats["historical_denominator"] += 1
            difficulty_stats["evaluation_eligible_samples"] += 1
            if sample["judge_eligible"]:
                difficulty_stats["judge_eligible_samples"] += 1

    stats = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "inputs": [
            {
                "dataset_key": _dataset_identity(dataset)[0],
                "dataset_version": _dataset_identity(dataset)[1],
                "difficulty": _dataset_identity(dataset)[2],
                "filename": dataset.path.name,
                "sha256": dataset.sha256,
            }
            for dataset in datasets
        ],
        "totals": {
            "source_cases": len(case_rows),
            "historical_denominator": len(case_rows),
            "protected_placeholders": protected_count,
            "evaluation_eligible_samples": len(samples),
            "judge_eligible_samples": judge_eligible_count,
            "answerable_samples": answerable_count,
            "unanswerable_samples": unanswerable_count,
            "relevance_judgments": len(qrel_rows),
            "lexical_requirements": len(claim_rows),
        },
        "by_dataset": by_dataset,
        "by_difficulty": by_difficulty,
        "protected_policy": {
            "case_ids": [PROTECTED_CASE_ID] if protected_count else [],
            "historical_denominator_retained": True,
            "content_parsed": False,
            "samples_and_judge_excluded": True,
        },
        "claim_semantics": {
            "expected_terms": "lexical_requirement",
            "complete_gold_claims": False,
            "note": LEXICAL_REQUIREMENT_NOTE,
        },
    }

    contents = {
        "cases.csv": _csv_content(CASES_FIELDS, case_rows),
        "relevance-judgments.csv": _csv_content(RELEVANCE_FIELDS, qrel_rows),
        "claims.csv": _csv_content(CLAIMS_FIELDS, claim_rows),
        "samples.jsonl": "".join(_json_text(sample) + "\n" for sample in samples),
        "dataset-stats.json": json.dumps(
            stats, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    }
    checksum_lines = [
        f"{_sha256_text(contents[name])}  {name}"
        for name in sorted(contents)
    ]
    contents["SHA256SUMS"] = "\n".join(checksum_lines) + "\n"

    for name in OUTPUT_FILENAMES:
        target = targets[name]
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(contents[name], encoding="utf-8", newline="")
        temporary.replace(target)
    return stats
