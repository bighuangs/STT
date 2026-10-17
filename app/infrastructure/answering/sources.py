"""把已引用 Evidence 投影为公开来源 DTO。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.rag.query.evidence_asset_service import evidence_image_projection


def _location_label(evidence: Mapping[str, Any]) -> str:
    labels: list[str] = []
    for raw in evidence.get("provenance_spans") or []:
        if not isinstance(raw, Mapping) or raw.get("availability") == "unavailable":
            continue
        if raw.get("page_no"):
            labels.append(f"第{raw['page_no']}页")
        elif raw.get("slide_no"):
            labels.append(f"第{raw['slide_no']}张幻灯片")
        elif raw.get("sheet"):
            cell = f" {raw['cell_range']}" if raw.get("cell_range") else ""
            labels.append(f"工作表 {raw['sheet']}{cell}")
        elif raw.get("line_start"):
            end = raw.get("line_end") or raw["line_start"]
            labels.append(f"第{raw['line_start']}–{end}行")
    unique = list(dict.fromkeys(labels))
    return "、".join(unique) if unique else "位置不可用"


def build_answer_sources(
    aliases: list[dict[str, Any]],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    """只公开实际进入最终证明图的来源。"""

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in aliases:
        evidence = item.get("evidence") if isinstance(item, dict) else None
        if not isinstance(evidence, dict):
            continue
        evidence_id = str(evidence.get("evidence_id") or "")
        if evidence_id not in evidence_ids or evidence_id in seen:
            continue
        seen.add(evidence_id)
        source = evidence.get("source") if isinstance(evidence.get("source"), dict) else {}
        sources.append(
            {
                "label": str(item.get("label") or ""),
                "evidence_id": evidence_id,
                "source_type": str(source.get("source_type") or "local"),
                "source_label": str(evidence.get("source_label") or item.get("label") or ""),
                "knowledge_base_id": source.get("kb_id"),
                "document_id": source.get("document_id"),
                "document_version_id": source.get("document_version_id"),
                "index_version_id": source.get("index_version_id"),
                "chunk_id": source.get("chunk_id"),
                "url": source.get("source_url"),
                "content_hash": str(evidence.get("content_hash") or ""),
                "location_label": _location_label(evidence),
                "provenance_spans": list(evidence.get("provenance_spans") or []),
                "heading_path": list(evidence.get("heading_path") or []),
                "images": evidence_image_projection(evidence),
                "modality": str(evidence.get("modality") or "text"),
                "asset": evidence.get("asset"),
                "derivation": evidence.get("derivation"),
                "retrieval": evidence.get("retrieval") or {},
                "preview_url": f"/api/v1/evidence/{evidence_id}",
            }
        )
    return sources
