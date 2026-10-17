"""Deterministically replay archived pre-grounding answers through current grounding code.

This script makes no LLM call and does not mutate the archived run.  It reconstructs
the frozen ContextPack aliases from each observation, revalidates the archived safe
pre-grounding answer, and writes a separate, hash-audited replay directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.query.citation_service import new_answer_id, validate_answer_citations
from app.rag.query.sufficiency_service import enforce_post_generation_grounding


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aliases(observation: dict[str, Any]) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for document in observation.get("context") or []:
        if not isinstance(document, dict):
            continue
        label = str(document.get("context_label") or "")
        evidence = document.get("evidence")
        if not label or not isinstance(evidence, dict):
            continue
        aliases.append(
            {
                "label": label,
                "evidence": evidence,
                "document": document,
                "packed_text": str(
                    document.get("security_safe_content")
                    or document.get("text")
                    or evidence.get("content")
                    or ""
                ),
            }
        )
    return aliases


def _kb_id(aliases: list[dict[str, Any]]) -> str | None:
    for item in aliases:
        source = item["evidence"].get("source") or {}
        value = source.get("kb_id")
        if value:
            return str(value)
    return None


def _replay_case(record: dict[str, Any]) -> dict[str, Any]:
    observation = copy.deepcopy(record.get("observation") or {})
    pre = str(observation.get("pre_grounding_answer") or "")
    old_final = str(observation.get("answer") or "")
    aliases = _aliases(observation)
    answer_id = str(observation.get("answer_id") or "") or new_answer_id()
    kb_id = _kb_id(aliases)
    bundle = validate_answer_citations(
        pre,
        aliases,
        answer_id=answer_id,
        kb_id=kb_id,
        persist=False,
    )
    initial_validation = copy.deepcopy(bundle.get("citation_validation") or {})
    initial_citations = copy.deepcopy(bundle.get("citations") or [])

    state = observation
    state.update(bundle)
    state.update(
        {
            "answer_id": answer_id,
            "context_aliases": aliases,
            "evaluation_mode": True,
            "kb_id": kb_id,
            "original_query": str(observation.get("query") or ""),
        }
    )
    enforce_post_generation_grounding(state)
    new_final = str(state.get("answer") or "")
    validation = state.get("citation_validation") or {}
    grounding = state.get("grounding") or {}
    decision = state.get("answer_decision") or {}
    return {
        "case_id": str(record.get("case_id") or ""),
        "pre_grounding_answer": pre,
        "old_final_answer": old_final,
        "replayed_final_answer": new_final,
        "hashes": {
            "pre_grounding_sha256": _sha256_text(pre),
            "old_final_sha256": _sha256_text(old_final),
            "replayed_final_sha256": _sha256_text(new_final),
        },
        "changed_from_old_final": new_final != old_final,
        "preserved_pre_grounding": new_final == pre,
        "citation_validation": validation,
        "grounding": grounding,
        "answer_decision": decision,
        "initial_citation_validation": initial_validation,
        "initial_citations": initial_citations,
        "trusted_citation_count": int(validation.get("trusted_count") or 0),
        "marker_normalization_count": int(
            initial_validation.get("marker_normalization_count") or 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing replay directory: {output}")
    result_paths = sorted(run_root.glob("artifacts/*/*/results.jsonl"))
    if not result_paths:
        raise SystemExit(f"no archived results found under {run_root}")
    output.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_hashes: dict[str, str] = {}
    for result_path in result_paths:
        difficulty = result_path.parts[-3]
        input_hashes[str(result_path.relative_to(run_root)).replace("\\", "/")] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            replayed = _replay_case(json.loads(line))
            replayed["difficulty"] = difficulty
            rows.append(replayed)
            by_difficulty[difficulty].append(replayed)

    results_path = output / "replay-results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summaries: dict[str, Any] = {}
    for difficulty, items in sorted(by_difficulty.items()):
        reason_codes = Counter(
            str((item.get("answer_decision") or {}).get("reason_code") or "unknown")
            for item in items
        )
        summaries[difficulty] = {
            "case_count": len(items),
            "changed_from_old_final_count": sum(item["changed_from_old_final"] for item in items),
            "preserved_pre_grounding_count": sum(item["preserved_pre_grounding"] for item in items),
            "trusted_citation_case_count": sum(item["trusted_citation_count"] > 0 for item in items),
            "marker_normalized_case_count": sum(
                item["marker_normalization_count"] > 0 for item in items
            ),
            "answer_decision_reason_counts": dict(sorted(reason_codes.items())),
        }
    summary = {
        "schema_version": "stt.grounding-replay.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "input_sha256": input_hashes,
        "current_code_files": {
            "citation_service_sha256": hashlib.sha256(
                (REPO_ROOT / "app/rag/query/citation_service.py").read_bytes()
            ).hexdigest(),
            "sufficiency_service_sha256": hashlib.sha256(
                (REPO_ROOT / "app/rag/query/sufficiency_service.py").read_bytes()
            ).hexdigest(),
            "claim_binding_sha256": hashlib.sha256(
                (REPO_ROOT / "app/shared/utils/claim_binding.py").read_bytes()
            ).hexdigest(),
        },
        "by_difficulty": summaries,
        "limitations": [
            "Deterministic replay reuses archived model output and ContextPack; it does not rerun retrieval or the LLM.",
            "Replay preservation is a pipeline diagnostic, not a semantic-correctness judgment.",
        ],
    }
    _write_json(output / "summary.json", summary)
    checksums = []
    for path in sorted(output.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_difficulty"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
