"""只读盘点 Milvus collection 的身份、partition、index 与兼容漂移。

示例：
  uv run python scripts/audit_milvus_topology.py --collection kb_chunks
  uv run python scripts/audit_milvus_topology.py \
    --offline-snapshot snapshot.json --expected-profile profile.json

该脚本没有任何 delete/drop/create/compact 代码路径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.vector_topology_service import VectorTopologyService  # noqa: E402
from app.domain.vector_topology import (  # noqa: E402
    CollectionInventorySnapshot,
    CompatibilityProfile,
    audit_inventory,
    compatibility_fingerprint,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway  # noqa: E402
from app.infra.vectorstore.milvus_topology_inspector import (  # noqa: E402
    MilvusReadOnlyTopologyInspector,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a read-only Milvus topology inventory report."
    )
    parser.add_argument(
        "--database",
        default="default",
        help="Milvus database already bound to the configured client (default: default).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Physical collection to inspect; defaults to configured chunks collection.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Identity-only sample size, 1..10000.",
    )
    parser.add_argument(
        "--offline-snapshot",
        type=Path,
        help="Audit a saved CollectionInventorySnapshot JSON without connecting to Milvus.",
    )
    parser.add_argument(
        "--expected-profile",
        type=Path,
        help="CompatibilityProfile JSON used to report drift.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Stdout is always emitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected = None
    if args.expected_profile:
        profile = CompatibilityProfile.model_validate(_load_json(args.expected_profile))
        expected = compatibility_fingerprint(profile)

    if args.offline_snapshot:
        snapshot = CollectionInventorySnapshot.model_validate(_load_json(args.offline_snapshot))
        report = audit_inventory(snapshot, expected_fingerprint=expected)
    else:
        collection = args.collection or milvus_gateway.chunks_collection
        inspector = MilvusReadOnlyTopologyInspector(
            milvus_gateway.client(),
            configured_database=args.database,
        )
        report = VectorTopologyService.audit_collection(
            inspector,
            physical_database=args.database,
            physical_collection=collection,
            expected_fingerprint=expected,
            sample_limit=args.sample_limit,
        )

    payload = {
        "command": "milvus_topology_inventory",
        "safety": {
            "dry_run": True,
            "mutating_operations_exposed": False,
            "sample_contains_document_content": False,
        },
        "report": report.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 2 if report.drift_paths or report.missing_identity_rows else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "command": "milvus_topology_inventory",
                    "dry_run": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
