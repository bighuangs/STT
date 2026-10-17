"""将旧 Milvus doc_id/kb_id 建立为 TODO-01 控制面事实。

默认只生成报告；加 --apply 才写控制库。缺 doc_id/kb_id 的记录进入 quarantine，
不会擅自归入 default tenant。--rollback-batch 可回滚本脚本创建的控制记录，
不会删除 Milvus 原数据。
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.application.control_plane_service import (
    bootstrap_control_plane,
    import_legacy_document,
    rollback_legacy_batch,
)
from app.infra.vectorstore import milvus_gateway
from app.shared.runtime.logger import PROJECT_ROOT


def build_legacy_groups(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"chunk_count": 0, "logical_name": ""}
    )
    quarantine: list[dict] = []
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        kb_id = str(row.get("kb_id") or "").strip()
        if not doc_id or not kb_id:
            quarantine.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "reason": "missing_doc_id_or_kb_id",
                }
            )
            continue
        item = grouped[(kb_id, doc_id)]
        item["chunk_count"] += 1
        item["logical_name"] = item["logical_name"] or str(
            row.get("docnm_kwd") or doc_id
        )
    documents = [
        {
            "kb_id": kb_id,
            "document_id": doc_id,
            "logical_name": value["logical_name"],
            "chunk_count": value["chunk_count"],
        }
        for (kb_id, doc_id), value in sorted(grouped.items())
    ]
    return documents, quarantine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际写入控制库；默认 dry-run")
    parser.add_argument("--limit", type=int, default=16384)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rollback-batch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_control_plane()
    if args.rollback_batch:
        removed = rollback_legacy_batch(args.rollback_batch)
        print(json.dumps({"rollback_batch": args.rollback_batch, "removed": removed}, ensure_ascii=False))
        return

    collection = milvus_gateway.chunks_collection
    rows = milvus_gateway.client().query(
        collection_name=collection,
        filter="",
        output_fields=["chunk_id", "doc_id", "kb_id", "docnm_kwd"],
        limit=args.limit,
    )
    documents, quarantine = build_legacy_groups(rows)
    batch_id = f"legacy_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    results = []
    if args.apply:
        for document in documents:
            results.append(
                import_legacy_document(
                    **document,
                    collection_name=collection,
                    batch_id=batch_id,
                )
            )
    report = {
        "batch_id": batch_id,
        "dry_run": not args.apply,
        "collection": collection,
        "scanned_rows": len(rows),
        "documents": documents,
        "quarantine": quarantine,
        "results": results,
    }
    report_path = args.report or PROJECT_ROOT / "output" / f"legacy-migration-{batch_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **{k: report[k] for k in ("batch_id", "dry_run", "scanned_rows")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
