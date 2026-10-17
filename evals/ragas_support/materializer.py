"""为 Ragas Judge 从本地向量库物化最小必要的文本 observation。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, Sequence


class ContextMaterializationError(RuntimeError):
    """上下文无法按冻结标识安全、完整地物化。"""


class MilvusQueryClient(Protocol):
    """物化器需要的最小 Milvus 只读接口。"""

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ObservationSource:
    """一组正式评测 observation 及其冻结物理集合。"""

    collection_name: str
    rows: Sequence[Mapping[str, Any]]


def _non_empty_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _case_id(row: Mapping[str, Any], *, source: str) -> str:
    value = _non_empty_string(row.get("case_id"))
    if value is None:
        raise ContextMaterializationError(f"{source} row is missing case_id")
    return value


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _eligible_sample(sample: Mapping[str, Any]) -> bool:
    sensitivity = str(sample.get("sensitivity") or "").strip().lower()
    return (
        sample.get("protected") is not True
        and sensitivity not in {"protected", "restricted"}
        and sample.get("judge_eligible") is True
    )


def _observation_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("observation")
    return value if isinstance(value, Mapping) else row


def _context_refs(row: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    payload = _observation_payload(row)
    values = payload.get("context")
    if not isinstance(values, list) or not values:
        raise ContextMaterializationError(
            f"{_case_id(row, source='observation')} has no ranked context references"
        )
    refs: list[tuple[str, str, str]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ContextMaterializationError(
                f"{_case_id(row, source='observation')} context[{index}] is invalid"
            )
        chunk_id = _non_empty_string(item.get("chunk_id"))
        content_hash = _non_empty_string(item.get("content_hash"))
        document_version_id = _non_empty_string(item.get("document_version_id"))
        if chunk_id is None or content_hash is None or document_version_id is None:
            raise ContextMaterializationError(
                f"{_case_id(row, source='observation')} context[{index}] "
                "is missing chunk_id/content_hash/document_version_id"
            )
        refs.append((chunk_id, content_hash, document_version_id))
    return refs


def _query_collection(
    client: MilvusQueryClient,
    *,
    collection_name: str,
    chunk_ids: Sequence[str],
) -> dict[tuple[str, str, str], str]:
    unique_ids = sorted(set(chunk_ids))
    expression = "source_chunk_id in " + json.dumps(unique_ids, ensure_ascii=False)
    rows = client.query(
        collection_name=collection_name,
        filter=expression,
        output_fields=[
            "source_chunk_id",
            "content_with_weight",
            "content_hash",
            "document_version_id",
        ],
    )
    materialized: dict[tuple[str, str, str], str] = {}
    for row in rows:
        chunk_id = _non_empty_string(row.get("source_chunk_id"))
        text = _non_empty_string(row.get("content_with_weight"))
        stored_hash = _non_empty_string(row.get("content_hash"))
        document_version_id = _non_empty_string(row.get("document_version_id"))
        if None in {chunk_id, text, stored_hash, document_version_id}:
            continue
        assert chunk_id is not None
        assert text is not None
        assert stored_hash is not None
        assert document_version_id is not None
        if _content_hash(text) != stored_hash:
            raise ContextMaterializationError(
                f"Milvus content hash mismatch for {chunk_id}/{document_version_id}"
            )
        key = (chunk_id, stored_hash, document_version_id)
        existing = materialized.get(key)
        if existing is not None and existing != text:
            raise ContextMaterializationError(
                f"ambiguous Milvus content for {chunk_id}/{document_version_id}"
            )
        materialized[key] = text
    return materialized


def materialize_judge_observations(
    samples: Sequence[Mapping[str, Any]],
    sources: Sequence[ObservationSource],
    *,
    case_ids: Sequence[str],
    client: MilvusQueryClient,
) -> list[dict[str, Any]]:
    """仅物化显式允许的 Judge 样本，并用内容哈希锁定文本版本。"""

    requested = tuple(case_id.strip() for case_id in case_ids if case_id.strip())
    if not requested:
        raise ContextMaterializationError("an explicit Judge case list is required")
    if len(set(requested)) != len(requested):
        raise ContextMaterializationError("Judge case list contains duplicate case_id")

    sample_by_case = {_case_id(sample, source="sample"): sample for sample in samples}
    unknown = [case_id for case_id in requested if case_id not in sample_by_case]
    if unknown:
        raise ContextMaterializationError(
            "Judge case list contains unknown case_id: " + ", ".join(unknown)
        )
    forbidden = [
        case_id for case_id in requested if not _eligible_sample(sample_by_case[case_id])
    ]
    if forbidden:
        raise ContextMaterializationError(
            "Judge case list contains protected/restricted/ineligible case_id: "
            + ", ".join(forbidden)
        )

    located: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for source in sources:
        collection_name = source.collection_name.strip()
        if not collection_name:
            raise ContextMaterializationError("observation source has no collection_name")
        for row in source.rows:
            case_id = _case_id(row, source="observation")
            if case_id not in requested:
                continue
            if case_id in located:
                raise ContextMaterializationError(
                    f"duplicate selected observation case_id: {case_id}"
                )
            located[case_id] = (collection_name, row)
    missing = [case_id for case_id in requested if case_id not in located]
    if missing:
        raise ContextMaterializationError(
            "selected observation is missing: " + ", ".join(missing)
        )

    refs_by_case: dict[str, list[tuple[str, str, str]]] = {}
    cases_by_collection: dict[str, list[str]] = {}
    for case_id in requested:
        collection_name, row = located[case_id]
        refs_by_case[case_id] = _context_refs(row)
        cases_by_collection.setdefault(collection_name, []).append(case_id)

    text_by_collection: dict[str, dict[tuple[str, str, str], str]] = {}
    for collection_name, collection_cases in cases_by_collection.items():
        chunk_ids = [
            chunk_id
            for case_id in collection_cases
            for chunk_id, _, _ in refs_by_case[case_id]
        ]
        text_by_collection[collection_name] = _query_collection(
            client,
            collection_name=collection_name,
            chunk_ids=chunk_ids,
        )

    output: list[dict[str, Any]] = []
    for case_id in requested:
        collection_name, row = located[case_id]
        payload = _observation_payload(row)
        response = _non_empty_string(
            payload.get("answer") or payload.get("response") or payload.get("final_answer")
        )
        if response is None:
            raise ContextMaterializationError(f"{case_id} has no non-empty response")
        contexts: list[str] = []
        context_ids: list[str] = []
        for ref in refs_by_case[case_id]:
            text = text_by_collection[collection_name].get(ref)
            if text is None:
                chunk_id, content_hash, document_version_id = ref
                raise ContextMaterializationError(
                    "Milvus context is missing for "
                    f"{case_id}/{chunk_id}/{content_hash}/{document_version_id}"
                )
            context_ids.append(ref[0])
            contexts.append(text)
        output.append(
            {
                "case_id": case_id,
                "response": response,
                "retrieved_context_ids": context_ids,
                "retrieved_contexts": contexts,
            }
        )
    return output


def write_materialized_observations(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """原子写入本地临时 Judge 输入，并返回不含正文的校验摘要。"""

    if path.exists() and not overwrite:
        raise ContextMaterializationError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary_path.replace(path)
    return {
        "case_count": len(rows),
        "context_count": sum(len(row.get("retrieved_contexts") or []) for row in rows),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "ContextMaterializationError",
    "MilvusQueryClient",
    "ObservationSource",
    "materialize_judge_observations",
    "write_materialized_observations",
]
