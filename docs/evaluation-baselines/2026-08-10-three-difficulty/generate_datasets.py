"""Generate three immutable 50-case evaluation datasets from published evidence.

The generator reads the active Pantum P3500 evidence set from the control database,
uses the configured online LLM only to author grounded Chinese questions/answers, and
then applies deterministic provenance and quality checks before writing manifests.
No credential or source document content is written outside this result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = RESULT_ROOT / "datasets"
GENERATION_ROOT = RESULT_ROOT / "generation"
RAW_ROOT = GENERATION_ROOT / "raw"
CONTROL_DB = PROJECT_ROOT / "volumes" / "control" / "stt-control.db"

DEFAULT_KB_ID = "kb_2fad25538aab42d19baf79fba782a06c"
SOURCE_RELATIVE_PATH = "doc/PantumP3500用户手册zh_CNV1.2_1644316283788.pdf"
SOURCE_SHA256 = "c4c7b2a434f3f86c4988d3961ac7304970029e71b0201fb6c10ca722b63bce2b"
SOURCE_LOGICAL_NAME = "source.pdf"
SOURCE_ORIGINAL_NAME = "PantumP3500用户手册zh_CNV1.2_1644316283788.pdf"
SUITE_KEY = "three-difficulty-live-2026-08-10"
CASE_COUNT = 50
ANSWERABLE_HARD_CASES = 45

DIFFICULTIES = {
    "easy": {
        "label": "简单",
        "prefix": "P3500-EASY",
        "name": "Pantum P3500 简单难度全链路测试集",
        "description": "50 道单证据、直接事实型问题，验证基础召回、回答、引用与延迟。",
    },
    "medium": {
        "label": "中等",
        "prefix": "P3500-MEDIUM",
        "name": "Pantum P3500 中等难度全链路测试集",
        "description": "50 道多约束、表格或操作步骤型问题，验证完整信息提取与可信引用。",
    },
    "hard": {
        "label": "困难",
        "prefix": "P3500-HARD",
        "name": "Pantum P3500 困难难度全链路测试集",
        "description": "45 道跨证据综合题与 5 道知识边界拒答题，验证组合推理、完整性和拒答。",
    },
}

SYSTEM_PROMPT = """你是企业 RAG 测评集的资深标注员。你只能依据给定证据编写中文问题和参考答案。
要求：
1. 禁止引入证据中没有的事实、数值、型号、步骤或因果关系。
2. 问题必须能够仅凭绑定证据回答；不要询问证据没有明确给出的信息。
3. reference_answer 要简洁、完整且可直接用于自动评分，不要添加来源编号或页码引用。
4. expected_terms 必须是 reference_answer 和绑定证据中都逐字存在的高辨识度短语；不要只用“打印机”“使用”等泛词。
5. 输出严格 JSON 对象，不要 Markdown，不要解释。每个 plan_id 恰好输出一次。
"""

DIFFICULTY_PROMPTS = {
    "easy": """为每个计划各写 1 道简单题。题目只考一个直接事实或一个明确动作；答案包含 1 至 3 个核心事实；expected_terms 取 1 至 3 个。不要做跨段推理。""",
    "medium": """为每个计划各写 1 道中等题。题目应要求同一证据中的 2 至 4 个约束、步骤、条件或表格值，使用自然改写而非照抄标题；expected_terms 取 2 至 5 个。答案必须覆盖问题的每个分支。""",
    "hard": """为每个计划各写 1 道困难题。必须同时使用该计划给出的两条证据，形成比较、条件化处置、跨步骤综合或跨页规格归纳；答案必须清楚覆盖两条证据的事实；expected_terms 取 3 至 6 个。允许用 1 至 2 条 conversation 构造有明确指代的多轮问题，但最终 query 仍须可结合 conversation 理解。""",
}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    page: int
    heading_path: tuple[str, ...]
    content: str
    content_hash: str
    source_label: str

    @property
    def chapter(self) -> str:
        for heading in reversed(self.heading_path):
            match = re.match(r"\s*(\d+)(?:\.|\s)", heading)
            if match:
                return match.group(1)
        return "front"

    def prompt_dict(self, key: str) -> dict[str, Any]:
        return {
            "key": key,
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "heading_path": list(self.heading_path),
            "content": self.content[:2200],
        }


@dataclass(frozen=True)
class Plan:
    plan_id: str
    evidence: tuple[Evidence, ...]

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "evidence": [
                item.prompt_dict(f"{self.plan_id}-E{index}")
                for index, item in enumerate(self.evidence, start=1)
            ],
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).lower())


def _decode_headings(raw: Any) -> tuple[str, ...]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _source_digest() -> str:
    source = PROJECT_ROOT / SOURCE_RELATIVE_PATH
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect_read_only() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{CONTROL_DB.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_active_evidence(kb_id: str) -> tuple[dict[str, Any], list[Evidence]]:
    with _connect_read_only() as connection:
        scope = connection.execute(
            """
            SELECT iv.id AS index_version_id, iv.version_no, d.id AS document_id,
                   dv.id AS document_version_id, dv.content_sha256,
                   b.chunk_count, d.logical_name
            FROM index_versions iv
            JOIN document_index_bindings b ON b.index_version_id = iv.id
            JOIN document_versions dv ON dv.id = b.document_version_id
            JOIN documents d ON d.id = dv.document_id
            WHERE iv.kb_id = ? AND iv.status = 'active' AND iv.active_slot = 'active'
              AND b.status = 'published' AND d.active_version_id = dv.id
              AND lower(dv.content_sha256) = ?
            """,
            (kb_id, SOURCE_SHA256),
        ).fetchall()
        if len(scope) != 1:
            raise RuntimeError(
                f"expected exactly one active published source scope, found {len(scope)}"
            )
        scope_row = dict(scope[0])
        if scope_row["logical_name"] != SOURCE_LOGICAL_NAME:
            raise RuntimeError("active source logical name no longer matches the dataset anchor")
        rows = connection.execute(
            """
            SELECT id, chunk_id, primary_page_no, heading_path_json, content,
                   content_hash, source_label
            FROM evidence_records
            WHERE index_version_id = ? AND document_version_id = ?
            ORDER BY primary_page_no, id
            """,
            (scope_row["index_version_id"], scope_row["document_version_id"]),
        ).fetchall()
    evidence = [
        Evidence(
            evidence_id=str(row["id"]),
            chunk_id=str(row["chunk_id"]),
            page=int(row["primary_page_no"] or 0),
            heading_path=_decode_headings(row["heading_path_json"]),
            content=str(row["content"]),
            content_hash=str(row["content_hash"]),
            source_label=str(row["source_label"]),
        )
        for row in rows
    ]
    if len(evidence) != int(scope_row["chunk_count"]):
        raise RuntimeError(
            "published binding chunk count does not match persisted evidence count"
        )
    return scope_row, evidence


def _eligible(item: Evidence) -> bool:
    compact = _compact(item.content)
    if item.page <= 1 or len(compact) < 55:
        return False
    headings = " ".join(item.heading_path)
    if "目录" in headings:
        return False
    if item.content.count("![图片]") >= 3 and len(compact) < 180:
        return False
    image_terms = ("视觉类型", "图像左侧", "整体为", "无任何可识别")
    if sum(term in item.content for term in image_terms) >= 2 and len(compact) < 260:
        return False
    return True


def _distributed_take(items: list[Evidence], count: int, *, offset: int = 0) -> list[Evidence]:
    if len(items) < count:
        raise RuntimeError(f"not enough eligible evidence: need {count}, have {len(items)}")
    ordered = sorted(items, key=lambda item: (item.page, item.chunk_id))
    selected: list[Evidence] = []
    used: set[str] = set()
    for index in range(count):
        position = min(len(ordered) - 1, math.floor((index + 0.5) * len(ordered) / count))
        position = (position + offset) % len(ordered)
        for delta in range(len(ordered)):
            candidate = ordered[(position + delta) % len(ordered)]
            if candidate.chunk_id not in used:
                used.add(candidate.chunk_id)
                selected.append(candidate)
                break
    return selected


def build_plans(evidence: list[Evidence]) -> dict[str, list[Plan]]:
    eligible = [item for item in evidence if _eligible(item)]
    simple_pool = [
        item
        for item in eligible
        if 70 <= len(_compact(item.content)) <= 750
        and item.content.count("![图片]") <= 2
    ]
    simple_rows = _distributed_take(simple_pool, CASE_COUNT)

    simple_ids = {item.chunk_id for item in simple_rows}
    medium_pool = [
        item
        for item in eligible
        if item.chunk_id not in simple_ids
        and 140 <= len(_compact(item.content)) <= 1900
        and (
            any(token in item.content for token in ("步骤", "如果", "注：", "设置", "安装", "|"))
            or len(_compact(item.content)) >= 360
        )
    ]
    if len(medium_pool) < CASE_COUNT:
        medium_pool.extend(
            item
            for item in eligible
            if item.chunk_id not in simple_ids
            and item.chunk_id not in {row.chunk_id for row in medium_pool}
        )
    medium_rows = _distributed_take(medium_pool, CASE_COUNT, offset=3)

    rich = [item for item in eligible if len(_compact(item.content)) >= 150]
    by_chapter: dict[str, list[Evidence]] = defaultdict(list)
    for item in rich:
        by_chapter[item.chapter].append(item)
    pairs: list[tuple[Evidence, Evidence]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for chapter in sorted(by_chapter, key=lambda value: (value == "front", value)):
        rows = sorted(by_chapter[chapter], key=lambda item: (item.page, item.chunk_id))
        for index, left in enumerate(rows):
            candidates = rows[index + 1 : index + 7]
            for right in candidates:
                if left.chunk_id == right.chunk_id:
                    continue
                key = tuple(sorted((left.chunk_id, right.chunk_id)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append((left, right))
                break
    if len(pairs) < ANSWERABLE_HARD_CASES:
        raise RuntimeError(
            f"not enough thematic evidence pairs: need {ANSWERABLE_HARD_CASES}, have {len(pairs)}"
        )
    pair_indexes = _distributed_indexes(len(pairs), ANSWERABLE_HARD_CASES)
    hard_pairs = [pairs[index] for index in pair_indexes]

    return {
        "easy": [
            Plan(f"easy-{index:03d}", (item,))
            for index, item in enumerate(simple_rows, start=1)
        ],
        "medium": [
            Plan(f"medium-{index:03d}", (item,))
            for index, item in enumerate(medium_rows, start=1)
        ],
        "hard": [
            Plan(f"hard-{index:03d}", pair)
            for index, pair in enumerate(hard_pairs, start=1)
        ],
    }


def _distributed_indexes(total: int, count: int) -> list[int]:
    if total < count:
        raise ValueError("total must be at least count")
    return [min(total - 1, math.floor((index + 0.5) * total / count)) for index in range(count)]


def _extract_json(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def _usage_dict(completion: Any, *, difficulty: str, batch: int, attempt: int) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    return {
        "difficulty": difficulty,
        "batch": batch,
        "attempt": attempt,
        "model": str(getattr(completion, "model", "")),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _grounded_expected_terms(
    reference: str,
    evidence_text: str,
    preferred: list[str],
    *,
    minimum: int,
    maximum: int,
) -> list[str]:
    """Keep model labels when valid and derive exact grounded spans when needed."""

    compact_reference = _compact(reference)
    compact_evidence = _compact(evidence_text)
    accepted: list[str] = []
    accepted_compact: set[str] = set()

    def add(term: str) -> None:
        value = term.strip(" \t\r\n，。；：、（）()")
        compact_value = _compact(value)
        if (
            not value
            or len(compact_value) < 2
            or compact_value in accepted_compact
            or compact_value not in compact_reference
            or compact_value not in compact_evidence
        ):
            return
        if any(
            compact_value in existing or existing in compact_value
            for existing in accepted_compact
        ):
            return
        accepted.append(value)
        accepted_compact.add(compact_value)

    for term in preferred:
        add(term)

    high_information_patterns = (
        r"[A-Za-z]+(?:[A-Za-z0-9./_-]*[A-Za-z0-9])?",
        r"\d+(?:[.,]\d+)*(?:\s*(?:mm|cm|ppm|dpi|MB|GB|GHz|Hz|V|W|℃|°C|秒|分钟|页|张|年|%))?",
    )
    for pattern in high_information_patterns:
        for match in re.finditer(pattern, reference, flags=re.IGNORECASE):
            add(match.group(0))

    chinese_runs = re.findall(r"[\u4e00-\u9fff]{4,}", reference)
    for size in (12, 10, 8, 6, 5, 4):
        for run in chinese_runs:
            if len(run) < size:
                continue
            for offset in range(0, len(run) - size + 1):
                add(run[offset : offset + size])
                if len(accepted) >= maximum:
                    break
            if len(accepted) >= maximum:
                break
        if len(accepted) >= maximum:
            break

    if len(accepted) < minimum:
        raise ValueError(
            f"could only derive {len(accepted)} grounded expected terms; need {minimum}"
        )
    return accepted[:maximum]


def _validate_authored_batch(
    raw: dict[str, Any], plans: list[Plan], difficulty: str
) -> list[dict[str, Any]]:
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError("response has no cases array")
    expected_ids = [plan.plan_id for plan in plans]
    actual_ids = [str(item.get("plan_id") or "") for item in cases if isinstance(item, dict)]
    if sorted(actual_ids) != sorted(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError(f"plan ids mismatch: expected={expected_ids}, actual={actual_ids}")
    by_id = {str(item["plan_id"]): item for item in cases}
    validated: list[dict[str, Any]] = []
    for plan in plans:
        item = dict(by_id[plan.plan_id])
        query = str(item.get("query") or "").strip()
        reference = str(item.get("reference_answer") or "").strip()
        expected_terms = [
            str(term).strip()
            for term in item.get("expected_terms") or []
            if str(term).strip()
        ]
        if not query or len(query) < 8:
            raise ValueError(f"{plan.plan_id}: query is missing or too short")
        if not reference or len(_compact(reference)) < 2:
            raise ValueError(f"{plan.plan_id}: reference answer is missing or too short")
        minimum_terms = 1 if difficulty == "easy" else 2 if difficulty == "medium" else 3
        maximum_terms = 3 if difficulty == "easy" else 5 if difficulty == "medium" else 6
        evidence_text = "\n".join(source.content for source in plan.evidence)
        expected_terms = _grounded_expected_terms(
            reference,
            evidence_text,
            expected_terms,
            minimum=minimum_terms,
            maximum=maximum_terms,
        )
        reference_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", reference))
        evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", evidence_text))
        if not reference_numbers.issubset(evidence_numbers):
            raise ValueError(
                f"{plan.plan_id}: answer introduced numbers not present in evidence: "
                f"{sorted(reference_numbers - evidence_numbers)}"
            )
        conversation = item.get("conversation") or []
        if not isinstance(conversation, list) or len(conversation) > 2:
            raise ValueError(f"{plan.plan_id}: conversation must contain at most two messages")
        validated.append(
            {
                "query": query,
                "conversation": conversation,
                "reference_answer": reference,
                "expected_terms": expected_terms,
                "forbidden_terms": [
                    str(term).strip()
                    for term in item.get("forbidden_terms") or []
                    if str(term).strip()
                ],
                "model_tags": [
                    str(tag).strip()
                    for tag in item.get("tags") or []
                    if str(tag).strip()
                ],
                "model_metadata": (
                    dict(item.get("metadata") or {})
                    if isinstance(item.get("metadata") or {}, dict)
                    else {}
                ),
            }
        )
    return validated


def author_cases(
    client: OpenAI,
    *,
    model: str,
    difficulty: str,
    plans: list[Plan],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authored: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(plans), batch_size), start=1):
        batch = plans[start : start + batch_size]
        payload = {
            "difficulty": difficulty,
            "rules": DIFFICULTY_PROMPTS[difficulty],
            "output_schema": {
                "cases": [
                    {
                        "plan_id": "string",
                        "query": "string",
                        "conversation": [],
                        "reference_answer": "string",
                        "expected_terms": ["string"],
                        "forbidden_terms": [],
                        "tags": ["string"],
                        "metadata": {"type": "direct_fact|procedure|table|synthesis"},
                    }
                ]
            },
            "plans": [plan.prompt_dict() for plan in batch],
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        errors: list[str] = []
        for attempt in range(1, 5):
            completion = client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=7000,
                extra_body={"enable_thinking": False},
            )
            content = completion.choices[0].message.content or ""
            record_path = RAW_ROOT / f"{difficulty}-batch-{batch_index:02d}-attempt-{attempt}.json"
            record_path.write_text(
                json.dumps(
                    {
                        "difficulty": difficulty,
                        "batch": batch_index,
                        "attempt": attempt,
                        "plan_ids": [plan.plan_id for plan in batch],
                        "response": content,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            usage_rows.append(
                _usage_dict(
                    completion,
                    difficulty=difficulty,
                    batch=batch_index,
                    attempt=attempt,
                )
            )
            try:
                parsed = _extract_json(content)
                authored.extend(_validate_authored_batch(parsed, batch, difficulty))
                print(
                    f"[{difficulty}] batch {batch_index}: {len(batch)} cases accepted "
                    f"(attempt {attempt})",
                    flush=True,
                )
                break
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt}: {exc}")
                if attempt == 4:
                    raise RuntimeError(
                        f"{difficulty} batch {batch_index} failed validation: "
                        + " | ".join(errors)
                    ) from exc
                time.sleep(min(2**attempt, 8))
    if len(authored) != len(plans):
        raise RuntimeError(
            f"{difficulty} authored count mismatch: expected {len(plans)}, got {len(authored)}"
        )
    return authored, usage_rows


def _relevant_evidence(plan: Plan, expected_terms: list[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in plan.evidence:
        matching_terms = [
            term for term in expected_terms if _compact(term) in _compact(item.content)
        ]
        values.append(
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.chunk_id,
                "page": item.page,
                "text": item.content[:1800],
                "match_terms": matching_terms,
                "content_hash": item.content_hash,
            }
        )
    return values


def _materialize_answerable_cases(
    difficulty: str, plans: list[Plan], authored: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    config = DIFFICULTIES[difficulty]
    values: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for ordinal, (plan, item) in enumerate(zip(plans, authored, strict=True), start=1):
        query_key = _compact(item["query"])
        if query_key in seen_queries:
            raise RuntimeError(f"duplicate query in {difficulty} dataset: {item['query']}")
        seen_queries.add(query_key)
        case_id = f"{config['prefix']}-{ordinal:03d}"
        base_tags = [
            "manual",
            "printer",
            "zh-CN",
            "live-baseline",
            f"difficulty-{difficulty}",
        ]
        if difficulty == "easy":
            base_tags.append("single-evidence")
        elif difficulty == "medium":
            base_tags.append("multi-constraint")
        else:
            base_tags.extend(("multi-evidence", "synthesis"))
        tags = list(dict.fromkeys([*base_tags, *item["model_tags"]]))
        metadata = {
            **item["model_metadata"],
            "difficulty": difficulty,
            "suite_key": SUITE_KEY,
            "evidence_count": len(plan.evidence),
            "critical": ordinal <= 5 or ordinal % 10 == 0,
        }
        values.append(
            {
                "case_id": case_id,
                "query": item["query"],
                "conversation": item["conversation"],
                "reference_answer": item["reference_answer"],
                "unanswerable": False,
                "expected_terms": item["expected_terms"],
                "forbidden_terms": item["forbidden_terms"],
                "relevant_document_ids": [
                    "doc_9353bf25fcfa4991a8a0425713a3c9f1"
                ],
                "relevant_document_names": [SOURCE_LOGICAL_NAME],
                "relevant_chunk_ids": [source.chunk_id for source in plan.evidence],
                "relevant_evidence": _relevant_evidence(plan, item["expected_terms"]),
                "tags": tags,
                "sensitivity": "internal",
                "metadata": metadata,
            }
        )
    return values


def _hard_unanswerable_cases(start_ordinal: int) -> list[dict[str, Any]]:
    questions = [
        (
            "请给出 P3500 在 2026 年 8 月上海地区的实时零售价、现货库存和最低成交价。",
            "该 V1.2 用户手册没有提供 2026 年 8 月上海地区的实时价格、库存或成交价，因此无法仅依据手册回答。",
        ),
        (
            "P3500 的 Windows 12 驱动计划在哪一天发布，下载包 SHA-256 是什么？",
            "该 V1.2 用户手册没有提供 Windows 12 驱动发布日期或下载包 SHA-256，因此无法仅依据手册确定。",
        ),
        (
            "请列出 P3500 在 2026 年最新固件版本中修复的全部 CVE 编号和补丁日期。",
            "该 V1.2 用户手册没有提供 2026 年最新固件、CVE 编号或补丁日期，因此无法仅依据手册回答。",
        ),
        (
            "根据手册计算 P3500 使用五年后的年度故障概率和平均维修费用。",
            "手册没有给出五年期故障统计、概率模型或维修价格，无法据此计算年度故障概率和平均维修费用。",
        ),
        (
            "手册推荐购买哪个第三方碳粉盒品牌，并提供当前有效的官方折扣码。",
            "手册没有提供第三方碳粉盒品牌推荐或当前折扣码，因此无法仅依据手册回答。",
        ),
    ]
    values: list[dict[str, Any]] = []
    for offset, (query, answer) in enumerate(questions):
        ordinal = start_ordinal + offset
        values.append(
            {
                "case_id": f"{DIFFICULTIES['hard']['prefix']}-{ordinal:03d}",
                "query": query,
                "conversation": [],
                "reference_answer": answer,
                "unanswerable": True,
                "expected_terms": ["没有提供", "无法"],
                "forbidden_terms": [],
                "relevant_document_ids": [
                    "doc_9353bf25fcfa4991a8a0425713a3c9f1"
                ],
                "relevant_document_names": [SOURCE_LOGICAL_NAME],
                "relevant_chunk_ids": [],
                "relevant_evidence": [],
                "tags": [
                    "manual",
                    "printer",
                    "zh-CN",
                    "live-baseline",
                    "difficulty-hard",
                    "unanswerable",
                    "knowledge-boundary",
                ],
                "sensitivity": "internal",
                "metadata": {
                    "type": "unanswerable",
                    "difficulty": "hard",
                    "suite_key": SUITE_KEY,
                    "evidence_count": 0,
                    "critical": True,
                },
            }
        )
    return values


def _manifest(
    *,
    difficulty: str,
    cases: list[dict[str, Any]],
    scope: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    config = DIFFICULTIES[difficulty]
    return {
        "dataset_key": f"pantum-p3500-{difficulty}-live-baseline",
        "name": config["name"],
        "description": config["description"],
        "version": "1.0.0",
        "schema_version": "stt.eval.v1",
        "suite_key": SUITE_KEY,
        "difficulty": difficulty,
        "difficulty_label": config["label"],
        "minimum_case_count": CASE_COUNT,
        "evaluation_mode_default": "live",
        "tags": [
            "manual",
            "printer",
            "zh-CN",
            "pantum-p3500",
            "three-difficulty-baseline",
            f"difficulty-{difficulty}",
            "live",
        ],
        "source": {
            "logical_name": SOURCE_LOGICAL_NAME,
            "original_logical_name": SOURCE_ORIGINAL_NAME,
            "relative_path": SOURCE_RELATIVE_PATH,
            "content_sha256": SOURCE_SHA256,
            "page_count": 95,
            "layout": "A4 portrait, 12 chapters, tables and installation screenshots",
            "reviewed_from_published_evidence": True,
            "kb_id": DEFAULT_KB_ID,
            "document_id": scope["document_id"],
            "document_version_id": scope["document_version_id"],
            "index_version_id": scope["index_version_id"],
        },
        "generation": {
            "authoring_model": model,
            "method": "online-llm-grounded-plus-deterministic-qc",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_evidence_count": int(scope["chunk_count"]),
        },
        "gate_policy": {
            "policy_version": "stt.eval.gate.v2",
            "recall_at_1_max_drop": 0.02,
            "recall_at_3_max_drop": 0.02,
            "recall_at_10_max_drop": 0.02,
            "mrr_max_drop": 0.02,
            "ndcg_at_10_max_drop": 0.02,
            "grounding_coverage_max_drop": 0.0,
            "unanswerable_false_answer_max_increase": 0.0,
            "p95_latency_max_increase_ratio": 0.2,
            "critical_failure_blocks": True,
            "minimum_metric_coverage": 0.8,
        },
        "cases": cases,
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    from app.domain.evaluation import validate_dataset_manifest

    validate_dataset_manifest(manifest)
    difficulty = str(manifest.get("difficulty") or "")
    cases = manifest["cases"]
    if difficulty not in DIFFICULTIES:
        raise ValueError("dataset difficulty is invalid")
    if len(cases) < CASE_COUNT:
        raise ValueError(f"{difficulty} dataset must contain at least {CASE_COUNT} cases")
    if len(cases) != CASE_COUNT:
        raise ValueError(f"{difficulty} dataset must contain exactly {CASE_COUNT} cases")
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{difficulty} dataset has duplicate case ids")
    queries = [_compact(case["query"]) for case in cases]
    if len(queries) != len(set(queries)):
        raise ValueError(f"{difficulty} dataset has duplicate questions")
    if any(case.get("metadata", {}).get("difficulty") != difficulty for case in cases):
        raise ValueError(f"{difficulty} dataset contains a case with wrong difficulty")


def _write_generation_summary(
    *,
    scope: dict[str, Any],
    evidence: list[Evidence],
    plans: dict[str, list[Plan]],
    usage: list[dict[str, Any]],
    model: str,
) -> None:
    summary = {
        "schema_version": "stt.eval.dataset-generation.v1",
        "suite_key": SUITE_KEY,
        "generated_at": datetime.now(UTC).isoformat(),
        "authoring_model": model,
        "source_scope": scope,
        "source_evidence_count": len(evidence),
        "source_evidence_digest": _sha256_json(
            [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": item.chunk_id,
                    "content_hash": item.content_hash,
                    "page": item.page,
                }
                for item in evidence
            ]
        ),
        "plans": {
            difficulty: [
                {
                    "plan_id": plan.plan_id,
                    "evidence_ids": [item.evidence_id for item in plan.evidence],
                    "chunk_ids": [item.chunk_id for item in plan.evidence],
                    "pages": [item.page for item in plan.evidence],
                }
                for plan in values
            ]
            for difficulty, values in plans.items()
        },
        "usage": {
            "calls": len(usage),
            "prompt_tokens": sum(row["prompt_tokens"] for row in usage),
            "completion_tokens": sum(row["completion_tokens"] for row in usage),
            "total_tokens": sum(row["total_tokens"] for row in usage),
            "records": usage,
        },
    }
    (GENERATION_ROOT / "generation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.batch_size <= 10:
        parser.error("--batch-size must be in 1..10")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATION_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {difficulty: DATASET_ROOT / f"{difficulty}.json" for difficulty in DIFFICULTIES}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(
            "refusing to overwrite immutable dataset files; pass --overwrite only before publication: "
            + ", ".join(existing)
        )

    if _source_digest().lower() != SOURCE_SHA256:
        raise RuntimeError("source PDF sha256 no longer matches the frozen source")

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.shared.config.settings_config import settings

    model = settings.llm_default_model
    if not settings.openai_base_url or not model or not settings.llm.api_key:
        raise RuntimeError("online LLM configuration is incomplete")
    scope, evidence = load_active_evidence(args.kb_id)
    plans = build_plans(evidence)
    print(
        f"source ready: index={scope['index_version_id']} evidence={len(evidence)} model={model}",
        flush=True,
    )
    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
        max_retries=2,
    )
    usage: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        authored, difficulty_usage = author_cases(
            client,
            model=model,
            difficulty=difficulty,
            plans=plans[difficulty],
            batch_size=args.batch_size,
        )
        usage.extend(difficulty_usage)
        cases = _materialize_answerable_cases(
            difficulty, plans[difficulty], authored
        )
        if difficulty == "hard":
            cases.extend(_hard_unanswerable_cases(len(cases) + 1))
        manifest = _manifest(
            difficulty=difficulty,
            cases=cases,
            scope=scope,
            model=model,
        )
        _validate_manifest(manifest)
        manifests[difficulty] = manifest
        outputs[difficulty].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{difficulty}] wrote {len(cases)} cases -> {outputs[difficulty]}",
            flush=True,
        )

    _write_generation_summary(
        scope=scope,
        evidence=evidence,
        plans=plans,
        usage=usage,
        model=model,
    )
    suite = {
        "schema_version": "stt.eval.suite.v1",
        "suite_key": SUITE_KEY,
        "evaluation_mode": "live",
        "knowledge_base_id": args.kb_id,
        "source_sha256": SOURCE_SHA256,
        "index_version_id": scope["index_version_id"],
        "datasets": {
            difficulty: {
                "path": str(outputs[difficulty].relative_to(RESULT_ROOT)).replace("\\", "/"),
                "dataset_key": manifest["dataset_key"],
                "version": manifest["version"],
                "case_count": len(manifest["cases"]),
                "checksum": _sha256_json(manifest),
            }
            for difficulty, manifest in manifests.items()
        },
    }
    (RESULT_ROOT / "suite.json").write_text(
        json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "dataset generation complete: "
        + ", ".join(
            f"{difficulty}={len(manifest['cases'])}"
            for difficulty, manifest in manifests.items()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
