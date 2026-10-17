"""在冻结 Medium 检索上下文上重放当前查询与来源闭包契约。

本脚本不调用模型、检索服务或评分器。它只读取已归档 results.jsonl 中的
Evidence ID，从本地 SQLite 只读恢复相同 Evidence 正文，然后用当前确定性
QueryContract、ContextPack、WitnessCatalog 与 typed closure builder 做诊断回放。
结果只能证明结构候选覆盖，不能替代正式 final/strict 测评。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 允许按文件路径执行，并与从仓库根目录导入时保持相同包边界。
from app.application.query_answering.pipeline import (  # noqa: E402
    freeze_question_contract,
    freeze_witness_catalog,
)
from app.application.query_security_pipeline import secure_evidence_documents  # noqa: E402
from app.application.query_understanding_service import (  # noqa: E402
    DeterministicQueryPlanner,
    QueryPlanningInput,
)
from app.domain.answering.enumeration_closure import (  # noqa: E402
    QueryAnchoredScopedBulletListClosureReceipt,
    QueryAnchoredScopedProcedureListClosureReceipt,
    QueryAnchoredScopedTableSelectionClosureReceipt,
    ScopedListAuthorityRepresentation,
)
from app.domain.answering.fact_set_closure import (  # noqa: E402
    FactSetAuthorityRepresentation,
    QueryAnchoredScopedFactSetClosureReceipt,
)
from app.domain.answering.question import (  # noqa: E402
    QuestionContract,
    RequiredOutputForm,
)
from app.domain.answering.witness import (  # noqa: E402
    EvidenceWitness,
    WitnessAtomKind,
    WitnessCatalogReceipt,
)
from app.domain.evidence import Evidence  # noqa: E402
from app.domain.query_planning import (  # noqa: E402
    AuthorizedScopeRef,
    PlanningPolicy,
    SubQueryPurpose,
)
from app.process.query.agent.state import create_query_default_state  # noqa: E402
from app.rag.query.context_pack_service import (  # noqa: E402
    ContextBudget,
    ContextPack,
    build_context_pack,
)

DEFAULT_DATASET = (
    ROOT / "docs/evaluation-baselines/2026-08-10-three-difficulty/datasets/medium.json"
)
DEFAULT_RESULTS = (
    ROOT
    / "var/evals/erun_d1b179e0579e407cbe19cfe2aef11258"
    / "generation-fe83c80cb960443e18a8152b4353bec6/results.jsonl"
)
DEFAULT_DATABASE = ROOT / "volumes/control/stt-control.db"
DEFAULT_OUTPUT = Path(__file__).with_name("offline-structural-replay.json")


class _CharacterCounter:
    @property
    def tokenizer_id(self) -> str:
        return "offline-structural-character-counter"

    @property
    def tokenizer_version(self) -> str:
        return "v1"

    def count(self, text: str) -> int:
        return len(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: max(0, max_tokens)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: object, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    parsed = json.loads(str(value))
    return parsed


def _read_dataset(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Medium dataset must contain a cases array")
    result: dict[str, str] = {}
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("Medium dataset case must be an object")
        case_id = str(row.get("case_id") or "")
        query = str(row.get("query") or "")
        if not case_id or not query or case_id in result:
            raise ValueError("Medium dataset case identity/query is invalid")
        result[case_id] = query
    return result


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("results.jsonl row must be an object")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in result:
            raise ValueError("results.jsonl case identity is invalid")
        result[case_id] = row
    return result


def _evidence_ids(result: dict[str, Any]) -> tuple[str, ...]:
    observation = result.get("observation")
    if not isinstance(observation, dict):
        return ()
    context = observation.get("context")
    if not isinstance(context, list):
        return ()
    rows = sorted(
        (item for item in context if isinstance(item, dict)),
        key=lambda item: int(item.get("rank") or 10_000),
    )
    ids: list[str] = []
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)
    return tuple(ids)


def _load_evidence(
    connection: sqlite3.Connection,
    evidence_ids: tuple[str, ...],
) -> tuple[Evidence, ...]:
    if not evidence_ids:
        return ()
    placeholders = ",".join("?" for _ in evidence_ids)
    rows = connection.execute(
        f"""
        SELECT id, schema_version, source_type, tenant_id, kb_id, document_id,
               document_version_id, index_version_id, chunk_id, source_url,
               source_fetched_at, content, content_hash, source_block_ids_json,
               provenance_spans_json, heading_path_json, asset_refs_json, source_label
        FROM evidence_records
        WHERE id IN ({placeholders})
        """,
        evidence_ids,
    ).fetchall()
    by_id = {str(row["id"]): row for row in rows}
    if set(by_id) != set(evidence_ids):
        missing = sorted(set(evidence_ids) - set(by_id))
        raise ValueError(f"archived Evidence is missing from SQLite: {missing}")

    result: list[Evidence] = []
    for evidence_id in evidence_ids:
        row = by_id[evidence_id]
        source_type = str(row["source_type"])
        if source_type == "local":
            source = {
                "source_type": "local",
                "tenant_id": row["tenant_id"],
                "kb_id": row["kb_id"],
                "document_id": row["document_id"],
                "document_version_id": row["document_version_id"],
                "index_version_id": row["index_version_id"],
                "chunk_id": row["chunk_id"],
                "source_block_ids": _json_value(
                    row["source_block_ids_json"],
                    default=[],
                ),
            }
        elif source_type == "web":
            source = {
                "source_type": "web",
                "tenant_id": row["tenant_id"],
                "source_url": row["source_url"],
                "content_hash": row["content_hash"],
                "fetched_at": row["source_fetched_at"],
                "source_block_ids": _json_value(
                    row["source_block_ids_json"],
                    default=[],
                ),
            }
        else:
            raise ValueError(f"unsupported Evidence source_type: {source_type}")
        result.append(
            Evidence.model_validate(
                {
                    "schema_version": row["schema_version"],
                    "evidence_id": row["id"],
                    "source": source,
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "provenance_spans": _json_value(
                        row["provenance_spans_json"],
                        default=[],
                    ),
                    "heading_path": _json_value(row["heading_path_json"], default=[]),
                    "asset_refs": _json_value(row["asset_refs_json"], default=[]),
                    "source_label": row["source_label"],
                }
            )
        )
    return tuple(result)


def _plan(query: str, evidences: tuple[Evidence, ...]):
    local = next((item for item in evidences if item.source.source_type == "local"), None)
    if local is None:
        tenant_id = "offline-medium"
        kb_ids = ("offline-medium",)
    else:
        tenant_id = str(local.source.tenant_id)
        kb_ids = tuple(
            dict.fromkeys(
                str(item.source.kb_id) for item in evidences if item.source.source_type == "local"
            )
        )
    planning_input = QueryPlanningInput(
        original_query=query,
        authorized_scope_ref=AuthorizedScopeRef(
            scope_id="scope-medium-offline-structural-replay",
            tenant_id=tenant_id,
            kb_ids=kb_ids,
        ),
        profile_version=1,
        policy=PlanningPolicy(
            allow_web=False,
            total_timeout_ms=1_000,
            token_budget=20_000,
            cost_budget=1.0,
        ),
        deadline_at=datetime.now(UTC) + timedelta(seconds=30),
    )
    return DeterministicQueryPlanner().plan(planning_input)


def _state_for_case(case_id: str, query: str, evidences: tuple[Evidence, ...]):
    plan = _plan(query, evidences)
    retrieval_sub_query_ids = tuple(
        item.sub_query_id for item in plan.sub_queries if item.purpose == SubQueryPurpose.RETRIEVAL
    ) or ("sq_primary",)
    documents = [
        {
            "evidence_id": evidence.evidence_id,
            "evidence": evidence.model_dump(mode="json", exclude_none=True),
            "title": evidence.source_label,
            "text": evidence.content,
            "content": evidence.content,
            "type": "milvus",
            "rerank_score": 1.0 / (index + 1),
            # 归档 observation 不公开逐 sub-query lineage；把相同冻结 Evidence
            # 投影给所有当前检索分支，只诊断结构上限，不宣称检索重放等价。
            "sub_query_ids": list(retrieval_sub_query_ids),
        }
        for index, evidence in enumerate(evidences)
    ]
    tenant_id = str(
        next(
            (item.source.tenant_id for item in evidences if item.source.source_type == "local"),
            "offline-medium",
        )
    )
    security_state: dict[str, Any] = {"tenant_id": tenant_id}
    secured_documents = secure_evidence_documents(security_state, documents)
    packed = build_context_pack(
        secured_documents,
        query_run_id="qry_" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:32],
        budget=ContextBudget.from_window(
            model_context_limit=200_000,
            reserved_system=1_000,
            reserved_output=3_000,
            safety_margin=1_000,
            evidence_budget=190_000,
            evidence_character_budget=1_000_000,
        ),
        counter=_CharacterCounter(),
        required_sub_query_ids=retrieval_sub_query_ids,
        facet_ledger=plan.facet_ledger,
    )
    packed_by_label = {block.label: block.text for block in packed.pack.packed_blocks}
    aliases = [
        {**item, "packed_text": packed_by_label[item["label"]]} for item in packed.citation_aliases
    ]
    local = next((item for item in evidences if item.source.source_type == "local"), None)
    kb_id = str(local.source.kb_id) if local is not None else "offline-medium"
    state = create_query_default_state(
        session_id="session-medium-offline",
        conversation_id="conversation-medium-offline",
        turn_id="turn-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16],
        tenant_id=tenant_id,
        user_id="offline-structural-replay",
        kb_id=kb_id,
        original_query=plan.understanding.original_query,
        rewritten_query=plan.understanding.standalone_query,
        query_plan=plan.model_dump(mode="json"),
        query_facet_ledger=plan.facet_ledger.model_dump(mode="json"),
        query_run_id="qry_" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:32],
        answer_id="ans_" + hashlib.sha256((case_id + ":answer").encode("utf-8")).hexdigest()[:32],
        authorized_scope={"scope_hash": "scope_" + "0" * 64},
        context_pack=packed.pack.model_dump(mode="json"),
        context_aliases=aliases,
        evidences=[item["evidence"] for item in aliases],
        evaluation_mode=True,
        llm_model_id="offline-no-model-call",
        is_stream=False,
        content_security=security_state.get("content_security") or {},
    )
    return state


def _baseline_fields(result: dict[str, Any]) -> dict[str, Any]:
    observation = result.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    decision = observation.get("answer_decision")
    decision = decision if isinstance(decision, dict) else {}
    proof = observation.get("answer_proof")
    proof = proof if isinstance(proof, dict) else {}
    metrics_envelope = result.get("metrics")
    metrics_envelope = metrics_envelope if isinstance(metrics_envelope, dict) else {}
    metric_groups = metrics_envelope.get("metrics")
    metric_groups = metric_groups if isinstance(metric_groups, dict) else {}
    case_outcome = metric_groups.get("case_outcome")
    case_outcome = case_outcome if isinstance(case_outcome, dict) else {}
    return {
        # 顶层 passed 是 strict gate，不是 final-answer accuracy；两者必须分列。
        "baseline_final_passed": case_outcome.get("final_answer_correct") is True,
        "baseline_strict_passed": case_outcome.get("strict_supported_case_pass") is True,
        "baseline_action": decision.get("action"),
        "baseline_proof_strict_eligible": bool(proof.get("strict_eligible")),
        "baseline_error_type": result.get("error_type"),
    }


def _closure_diagnostics(state: dict[str, Any]) -> list[dict[str, Any]]:
    """重放各类 scoped builder，并保留 fail-closed 的首个具体原因。"""

    question = QuestionContract.from_dict(state.get("question_contract") or {})
    bundle = state.get("witness_catalog") or {}
    catalog = WitnessCatalogReceipt.from_dict(bundle.get("receipt") or {})
    witnesses = tuple(
        EvidenceWitness.from_dict(item)
        for item in bundle.get("witnesses") or []
        if isinstance(item, dict)
    )
    pack = ContextPack.model_validate(state.get("context_pack") or {})
    aliases = {
        str(item.get("label") or ""): item
        for item in state.get("context_aliases") or []
        if isinstance(item, dict)
    }
    authorities: list[ScopedListAuthorityRepresentation] = []
    fact_authorities: list[FactSetAuthorityRepresentation] = []
    for block in pack.packed_blocks:
        alias = aliases.get(block.label)
        if alias is None or not isinstance(alias.get("evidence"), dict):
            raise ValueError("offline closure diagnostics is missing an Evidence alias")
        evidence = Evidence.model_validate(alias["evidence"])
        authorities.append(
            ScopedListAuthorityRepresentation(
                evidence_id=evidence.evidence_id,
                representation_id=block.representation_id,
                canonical_text=evidence.content,
                packed_text=block.text,
            )
        )
        fact_authorities.append(
            FactSetAuthorityRepresentation(
                evidence_id=evidence.evidence_id,
                representation_id=block.representation_id,
                canonical_text=evidence.content,
                packed_text=block.text,
            )
        )

    list_witnesses = tuple(
        item for item in witnesses if item.atom_kind == WitnessAtomKind.LIST_ITEM
    )
    table_witnesses = tuple(
        item for item in witnesses if item.atom_kind == WitnessAtomKind.TABLE_DATA_ROW
    )
    rows: list[dict[str, Any]] = []
    for obligation in question.obligations:
        if obligation.closure.value != "open_enumeration":
            continue
        attempts: dict[str, dict[str, Any]] = {}

        def attempt(name: str, builder: Any, *, members: tuple[EvidenceWitness, ...]) -> None:
            if not members:
                attempts[name] = {"status": "not_applicable", "reason": "no_matching_atom_kind"}
                return
            try:
                issued = builder.build(
                    question=question,
                    catalog=catalog,
                    obligation=obligation,
                    witnesses=members,
                    authority_representations=tuple(authorities),
                )
            except ValueError as exc:
                attempts[name] = {"status": "rejected", "reason": str(exc)[:500]}
            else:
                attempts[name] = {
                    "status": "issued",
                    "schema_version": issued.schema_version,
                    "member_count": len(issued.witness_ids),
                }

        if obligation.required_output_form == RequiredOutputForm.LIST:
            attempt(
                "scoped_table_selection",
                QueryAnchoredScopedTableSelectionClosureReceipt,
                members=table_witnesses,
            )
            attempt(
                "scoped_bullet_list",
                QueryAnchoredScopedBulletListClosureReceipt,
                members=list_witnesses,
            )
            fact_witnesses = tuple(
                item
                for item in witnesses
                if item.atom_kind in {WitnessAtomKind.SENTENCE, WitnessAtomKind.LIST_ITEM}
            )
            if not fact_witnesses:
                attempts["scoped_fact_set"] = {
                    "status": "not_applicable",
                    "reason": "no_matching_atom_kind",
                }
            else:
                try:
                    issued = QueryAnchoredScopedFactSetClosureReceipt.build(
                        question=question,
                        catalog=catalog,
                        obligation=obligation,
                        witnesses=fact_witnesses,
                        authority_representations=tuple(fact_authorities),
                        catalog_witnesses=witnesses,
                    )
                except ValueError as exc:
                    attempts["scoped_fact_set"] = {
                        "status": "rejected",
                        "reason": str(exc)[:500],
                    }
                else:
                    attempts["scoped_fact_set"] = {
                        "status": "issued",
                        "schema_version": issued.schema_version,
                        "member_count": len(issued.witness_ids),
                    }
        elif obligation.required_output_form == RequiredOutputForm.ORDERED_STEPS:
            attempt(
                "scoped_procedure_list",
                QueryAnchoredScopedProcedureListClosureReceipt,
                members=list_witnesses,
            )
        rows.append(
            {
                "obligation_id": obligation.obligation_id,
                "required_output_form": obligation.required_output_form.value,
                "attempts": attempts,
            }
        )
    return rows


def run(*, dataset_path: Path, results_path: Path, database_path: Path) -> dict[str, Any]:
    dataset = _read_dataset(dataset_path)
    results = _read_results(results_path)
    if set(dataset) != set(results):
        raise ValueError("dataset/results case sets differ")
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    try:
        for case_id, query in dataset.items():
            baseline = results[case_id]
            evidence_ids = _evidence_ids(baseline)
            row: dict[str, Any] = {
                "case_id": case_id,
                **_baseline_fields(baseline),
                "archived_context_evidence_count": len(evidence_ids),
                "replay_status": "not_run",
                "obligations": [],
                "source_closure_receipts": [],
            }
            if not evidence_ids:
                row["replay_status"] = "missing_archived_context"
                rows.append(row)
                continue
            try:
                evidences = _load_evidence(connection, evidence_ids)
                state = _state_for_case(case_id, query, evidences)
                freeze_question_contract(state)
                freeze_witness_catalog(state)
                row["replay_status"] = "succeeded"
                row["witness_universe_status"] = state["witness_catalog"]["receipt"][
                    "universe_status"
                ]
                row["witness_count"] = len(state["witness_catalog"]["witnesses"])
                row["witness_atom_kind_counts"] = dict(
                    sorted(
                        Counter(
                            str(item.get("atom_kind") or "")
                            for item in state["witness_catalog"]["witnesses"]
                        ).items()
                    )
                )
                row["obligations"] = [
                    {
                        "obligation_id": item["obligation_id"],
                        "information_need": item["information_need"],
                        "kind": item["kind"],
                        "necessity": item["necessity"],
                        "closure": item["closure"],
                        "required_output_form": item["required_output_form"],
                    }
                    for item in state["question_contract"]["obligations"]
                ]
                row["source_closure_receipts"] = [
                    {
                        "schema_version": item["schema_version"],
                        "obligation_id": item["obligation_id"],
                        "member_count": len(item.get("witness_ids") or [item.get("witness_id")]),
                    }
                    for item in state["witness_catalog"]["source_enumeration_closure_receipts"]
                ]
                required_open_ids = {
                    str(item["obligation_id"])
                    for item in row["obligations"]
                    if item["necessity"] == "required" and item["closure"] == "open_enumeration"
                }
                receipted_ids = {
                    str(item["obligation_id"]) for item in row["source_closure_receipts"]
                }
                row["required_open_obligation_count"] = len(required_open_ids)
                row["receipted_required_open_obligation_count"] = len(
                    required_open_ids & receipted_ids
                )
                row["all_required_open_obligations_receipted"] = (
                    bool(required_open_ids) and required_open_ids <= receipted_ids
                )
                row["closure_diagnostics"] = _closure_diagnostics(state)
            except Exception as exc:  # noqa: BLE001 - 逐 case 诊断必须保留其余回放结果。
                row["replay_status"] = "failed"
                row["failure_type"] = type(exc).__name__
                row["failure_message"] = str(exc)[:500]
            rows.append(row)
    finally:
        connection.close()

    schema_counts = Counter(
        receipt["schema_version"] for row in rows for receipt in row["source_closure_receipts"]
    )
    receipt_case_ids = [row["case_id"] for row in rows if row["source_closure_receipts"]]
    receipt_complete_open_case_ids = [
        row["case_id"] for row in rows if row.get("all_required_open_obligations_receipted") is True
    ]
    return {
        "schema_version": "hak180-medium-offline-structural-replay.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "diagnostic_only_not_formal_evaluation",
        "no_model_calls": True,
        "no_retrieval_service_calls": True,
        "limitations": [
            "Uses archived top-5 Evidence identities and local raw Evidence, not a new retrieval run.",
            "Archived observation omits per-sub-query lineage; each Evidence is projected to every current retrieval branch.",
            "Receipt issuance proves only deterministic structural eligibility, not model plan quality or final/strict correctness.",
        ],
        "inputs": {
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": _sha256(dataset_path),
            "results": str(results_path.resolve()),
            "results_sha256": _sha256(results_path),
            "database": str(database_path.resolve()),
        },
        "summary": {
            "case_count": len(rows),
            "succeeded_count": sum(row["replay_status"] == "succeeded" for row in rows),
            "failed_count": sum(row["replay_status"] == "failed" for row in rows),
            "missing_context_count": sum(
                row["replay_status"] == "missing_archived_context" for row in rows
            ),
            "baseline_final_pass_count": sum(row["baseline_final_passed"] is True for row in rows),
            "baseline_strict_pass_count": sum(
                row["baseline_strict_passed"] is True for row in rows
            ),
            "receipt_case_count": len(receipt_case_ids),
            "receipt_case_ids": receipt_case_ids,
            "receipt_complete_required_open_case_count": len(receipt_complete_open_case_ids),
            "receipt_complete_required_open_case_ids": receipt_complete_open_case_ids,
            "receipt_schema_counts": dict(sorted(schema_counts.items())),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        dataset_path=args.dataset.resolve(),
        results_path=args.results.resolve(),
        database_path=args.database.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
