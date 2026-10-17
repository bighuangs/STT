"""Tenant-scoped persistence for TODO-18 evaluation and feedback facts."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.domain.evaluation import (
    AnnotationStatus,
    FeedbackStatus,
    FORMAL_EVALUATION_CONTRACT_VERSION,
    FORMAL_EVALUATION_INTENT_VERSION,
    FORMAL_EVALUATION_SNAPSHOT_KEY,
    content_digest,
    ensure_feedback_transition,
    ensure_run_transition,
    finalize_formal_evaluation_contract,
    freeze_formal_evaluation_contract,
    freeze_formal_evaluation_intent,
    validate_dataset_manifest,
)
from app.infra.persistence.sql.models import (
    AnnotationTaskModel,
    EvaluationCaseModel,
    EvaluationDatasetModel,
    EvaluationResultModel,
    EvaluationRunModel,
    FeedbackModel,
)
from app.infra.persistence.sql.repositories import (
    DEFAULT_TENANT_ID,
    ResourceConflictError,
    ResourceNotFoundError,
    new_id,
)
from app.security.principal import current_principal


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


_EXECUTION_KEY = "execution"
_CASE_STORAGE_KEY = "__stt_evaluation_case_storage__"
_CASE_STORAGE_SCHEMA = "stt.evaluation.case-storage.v1"


def _case_metadata_for_storage(source: dict[str, Any]) -> dict[str, Any]:
    """在既有 JSON 列中封装无法映射到关系列的顶层 case 字段。"""

    metadata = dict(source.get("metadata") or {})
    case_fields = {
        field: source[field] for field in ("concept_groups", "authored_claims") if field in source
    }
    if not case_fields:
        return metadata
    return {
        _CASE_STORAGE_KEY: {
            "schema": _CASE_STORAGE_SCHEMA,
            "metadata": metadata,
            "case_fields": case_fields,
        }
    }


def _case_metadata_from_storage(
    stored: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """解开仓储保留信封；没有信封的旧数据保持原有 metadata 语义。"""

    raw = dict(stored or {})
    envelope = raw.get(_CASE_STORAGE_KEY)
    if len(raw) != 1 or not isinstance(envelope, dict):
        return raw, {}
    if envelope.get("schema") != _CASE_STORAGE_SCHEMA:
        return raw, {}
    metadata = envelope.get("metadata")
    case_fields = envelope.get("case_fields")
    if not isinstance(metadata, dict) or not isinstance(case_fields, dict):
        return raw, {}
    return dict(metadata), dict(case_fields)


class EvaluationRunOwnershipLost(ResourceConflictError):
    """运行已由恢复器终结或不再属于当前执行进程。"""


def _execution_state(row: EvaluationRunModel) -> dict:
    gate = row.gate_result if isinstance(row.gate_result, dict) else {}
    execution = gate.get(_EXECUTION_KEY)
    return dict(execution) if isinstance(execution, dict) else {}


def _execution_gate(
    *,
    status: str,
    owner_id: str | None,
    heartbeat_at: datetime,
) -> dict:
    return {
        "status": status,
        "passed": None,
        _EXECUTION_KEY: {
            "owner_id": owner_id,
            "heartbeat_at": _iso(heartbeat_at),
        },
    }


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{sha256(payload).hexdigest()[:32]}"


def _suite_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Expose optional suite facts without adding mutable database columns."""

    return {
        "suite_key": manifest.get("suite_key"),
        "difficulty": manifest.get("difficulty"),
        "minimum_case_count": manifest.get("minimum_case_count"),
        "evaluation_mode_default": manifest.get("evaluation_mode_default"),
    }


def _freeze_run_config_snapshot(
    config_snapshot: dict[str, Any],
    *,
    dataset: EvaluationDatasetModel,
    mode: str,
    run_id: str,
    tenant_id: str,
    kb_id: str,
    baseline_run_id: str | None,
    baseline_formal_contract_sha256: str | None,
    selected_case_ids: list[str],
    total_cases: int,
) -> dict[str, Any]:
    """校验正式运行权威；没有该字段的历史运行保持兼容。"""

    frozen = dict(config_snapshot)
    if FORMAL_EVALUATION_SNAPSHOT_KEY not in frozen:
        return frozen
    try:
        frozen[FORMAL_EVALUATION_SNAPSHOT_KEY] = freeze_formal_evaluation_contract(
            frozen[FORMAL_EVALUATION_SNAPSHOT_KEY],
            expected_mode=mode,
            expected_dataset_record_id=dataset.id,
            expected_dataset_checksum=dataset.checksum,
            expected_dataset_key=dataset.dataset_key,
            expected_dataset_version=dataset.version,
            expected_run_id=run_id,
            expected_tenant_id=tenant_id,
            expected_kb_id=kb_id,
            expected_baseline_run_id=baseline_run_id,
            expected_baseline_formal_contract_sha256=baseline_formal_contract_sha256,
            expected_selected_case_ids=selected_case_ids,
            expected_total_cases=total_cases,
            expected_runtime_snapshot=frozen,
        )
        contract = frozen[FORMAL_EVALUATION_SNAPSHOT_KEY]
        if contract.get("contract_version") == FORMAL_EVALUATION_CONTRACT_VERSION:
            authority = contract["run_authority"]
            expected_baseline = {
                "baseline_run_id": baseline_run_id,
                "baseline_formal_contract_sha256": baseline_formal_contract_sha256,
            }
            if authority["baseline"] != expected_baseline:
                raise ValueError("formal evaluation baseline authority does not match the row")
    except (TypeError, ValueError) as exc:
        raise ResourceConflictError("formal evaluation contract is invalid") from exc
    return frozen


class EvaluationRepository:
    def __init__(
        self,
        session: Session,
        *,
        tenant_id: str | None = None,
        actor_id: str | None = None,
    ) -> None:
        principal = current_principal()
        self.session = session
        self.tenant_id = tenant_id or (principal.tenant_id if principal else DEFAULT_TENANT_ID)
        self.actor_id = actor_id or (principal.user_id if principal else "system")

    # ------------------------------------------------------------------
    # Immutable datasets and cases
    # ------------------------------------------------------------------
    def seed_manifest(self, manifest: dict, *, parent_dataset_id: str | None = None) -> dict:
        validate_dataset_manifest(manifest)
        checksum = content_digest(manifest)
        dataset_key = str(manifest["dataset_key"])
        version = str(manifest["version"])
        existing = self.session.scalar(
            select(EvaluationDatasetModel).where(
                EvaluationDatasetModel.tenant_id == self.tenant_id,
                EvaluationDatasetModel.dataset_key == dataset_key,
                EvaluationDatasetModel.version == version,
            )
        )
        if existing is not None:
            if existing.checksum != checksum:
                raise ResourceConflictError(
                    f"published evaluation dataset is immutable: {dataset_key}@{version}"
                )
            snapshot = dict(existing.source_snapshot or {})
            if not isinstance(snapshot.get("manifest_json"), dict):
                # 旧版记录只保存 source。内容校验和相同，故可安全补齐同一份
                # 不可变 manifest，而不改变数据集语义或版本。
                existing.source_snapshot = {"manifest_json": manifest}
                self.session.flush()
            return self.dataset_dict(existing, include_cases=True)

        if parent_dataset_id:
            parent = self._dataset_model(parent_dataset_id)
            if parent.dataset_key != dataset_key:
                raise ResourceConflictError(
                    "evaluation dataset parent must use the same dataset_key"
                )

        dataset_id = _stable_id("eds", self.tenant_id, dataset_key, version)
        dataset = EvaluationDatasetModel(
            id=dataset_id,
            tenant_id=self.tenant_id,
            dataset_key=dataset_key,
            name=str(manifest["name"]),
            description=str(manifest.get("description", "")),
            version=version,
            schema_version=str(manifest["schema_version"]),
            checksum=checksum,
            # 在现有 JSON 列中保存完整不可变 manifest，避免为 gate_policy
            # 再引入迁移；对外 source 仍由 dataset_dict 单独投影。
            source_snapshot={"manifest_json": manifest},
            tags=list(manifest.get("tags") or []),
            status="active",
            parent_dataset_id=parent_dataset_id,
            created_by=self.actor_id,
        )
        self.session.add(dataset)
        # SQLite 开启外键校验时必须先落父记录；否则一次 flush 中 cases 可能
        # 在 dataset 之前发出 INSERT，导致 FOREIGN KEY constraint failed。
        self.session.flush()
        for ordinal, source in enumerate(manifest["cases"], start=1):
            case_id = str(source["case_id"])
            row = EvaluationCaseModel(
                id=_stable_id("eca", dataset_id, case_id),
                dataset_id=dataset_id,
                case_id=case_id,
                ordinal=ordinal,
                query=str(source["query"]),
                conversation=list(source.get("conversation") or []),
                reference_answer=str(source.get("reference_answer", "")),
                unanswerable=bool(source.get("unanswerable", False)),
                relevant_document_ids=list(source.get("relevant_document_ids") or []),
                relevant_document_names=list(source.get("relevant_document_names") or []),
                relevant_chunk_ids=list(source.get("relevant_chunk_ids") or []),
                relevant_evidence=list(source.get("relevant_evidence") or []),
                expected_terms=list(source.get("expected_terms") or []),
                forbidden_terms=list(source.get("forbidden_terms") or []),
                tags=list(source.get("tags") or []),
                sensitivity=str(source.get("sensitivity", "internal")),
                metadata_json=_case_metadata_for_storage(source),
            )
            self.session.add(row)
        self.session.flush()
        return self.dataset_dict(dataset, include_cases=True)

    def _dataset_model(self, dataset_id: str) -> EvaluationDatasetModel:
        dataset = self.session.scalar(
            select(EvaluationDatasetModel).where(
                EvaluationDatasetModel.id == dataset_id,
                EvaluationDatasetModel.tenant_id == self.tenant_id,
            )
        )
        if dataset is None:
            raise ResourceNotFoundError(f"evaluation dataset not found: {dataset_id}")
        return dataset

    def list_datasets(self) -> list[dict]:
        rows = self.session.scalars(
            select(EvaluationDatasetModel)
            .where(
                EvaluationDatasetModel.tenant_id == self.tenant_id,
                EvaluationDatasetModel.status == "active",
            )
            .order_by(EvaluationDatasetModel.dataset_key, EvaluationDatasetModel.created_at)
        ).all()
        return [self.dataset_dict(row, include_cases=False) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict:
        return self.dataset_dict(self._dataset_model(dataset_id), include_cases=True)

    def dataset_dict(self, row: EvaluationDatasetModel, *, include_cases: bool) -> dict:
        case_count = (
            self.session.scalar(
                select(func.count(EvaluationCaseModel.id)).where(
                    EvaluationCaseModel.dataset_id == row.id
                )
            )
            or 0
        )
        snapshot = dict(row.source_snapshot or {})
        manifest_json: dict[str, Any] = (
            dict(snapshot["manifest_json"])
            if isinstance(snapshot.get("manifest_json"), dict)
            else {}
        )
        source = dict(manifest_json.get("source") or {}) if manifest_json else snapshot
        value = {
            "id": row.id,
            "dataset_key": row.dataset_key,
            "name": row.name,
            "description": row.description,
            "version": row.version,
            "schema_version": row.schema_version,
            "checksum": row.checksum,
            "source": source,
            "manifest_json": manifest_json,
            "tags": row.tags or [],
            "status": row.status,
            "parent_dataset_id": row.parent_dataset_id,
            "case_count": int(case_count),
            "created_by": row.created_by,
            "created_at": _iso(row.created_at),
            **_suite_projection(manifest_json),
        }
        if include_cases:
            cases = self.session.scalars(
                select(EvaluationCaseModel)
                .where(EvaluationCaseModel.dataset_id == row.id)
                .order_by(EvaluationCaseModel.ordinal)
            ).all()
            value["cases"] = [self.case_dict(case) for case in cases]
        return value

    @staticmethod
    def case_dict(row: EvaluationCaseModel) -> dict:
        metadata, case_fields = _case_metadata_from_storage(row.metadata_json)
        value = {
            "id": row.id,
            "case_id": row.case_id,
            "ordinal": row.ordinal,
            "query": row.query,
            "conversation": row.conversation or [],
            "reference_answer": row.reference_answer,
            "unanswerable": row.unanswerable,
            "relevant_document_ids": row.relevant_document_ids or [],
            "relevant_document_names": row.relevant_document_names or [],
            "relevant_chunk_ids": row.relevant_chunk_ids or [],
            "relevant_evidence": row.relevant_evidence or [],
            "expected_terms": row.expected_terms or [],
            "forbidden_terms": row.forbidden_terms or [],
            "tags": row.tags or [],
            "sensitivity": row.sensitivity,
            "metadata": metadata,
        }
        for field in ("concept_groups", "authored_claims"):
            if field in case_fields:
                value[field] = case_fields[field]
        return value

    # ------------------------------------------------------------------
    # Durable evaluation runs and per-case observations
    # ------------------------------------------------------------------
    def create_run(
        self,
        *,
        dataset_id: str,
        kb_id: str,
        name: str,
        mode: str,
        baseline_run_id: str | None,
        config_snapshot: dict,
        case_ids: list[str] | None = None,
        execution_owner: str | None = None,
    ) -> dict:
        dataset = self._dataset_model(dataset_id)
        dataset_case_ids = list(
            self.session.scalars(
                select(EvaluationCaseModel.case_id)
                .where(EvaluationCaseModel.dataset_id == dataset.id)
                .order_by(EvaluationCaseModel.ordinal)
            ).all()
        )
        requested_case_ids = list(dict.fromkeys(case_ids or []))
        if requested_case_ids:
            unknown = sorted(set(requested_case_ids) - set(dataset_case_ids))
            if unknown:
                raise ResourceConflictError(
                    "evaluation run contains unknown case ids: " + ", ".join(unknown)
                )
            selected_case_ids = [
                case_id for case_id in dataset_case_ids if case_id in requested_case_ids
            ]
        else:
            selected_case_ids = dataset_case_ids
        total = len(selected_case_ids)
        if not selected_case_ids:
            raise ResourceConflictError("evaluation run contains no matching cases")

        baseline: EvaluationRunModel | None = None
        baseline_formal: dict[str, Any] | None = None
        baseline_formal_sha256: str | None = None
        if baseline_run_id:
            baseline = self._run_model(baseline_run_id)
            if baseline.dataset_id != dataset_id or baseline.kb_id != kb_id:
                raise ResourceConflictError(
                    "baseline must use the same dataset version and knowledge base"
                )
            if baseline.mode != mode:
                raise ResourceConflictError(
                    "baseline and candidate must use the same evaluation mode"
                )
            raw_baseline_formal = (baseline.config_snapshot or {}).get(
                FORMAL_EVALUATION_SNAPSHOT_KEY
            )
            if isinstance(raw_baseline_formal, dict):
                baseline_formal = raw_baseline_formal
                baseline_formal_sha256 = str(baseline_formal.get("contract_sha256") or "")
            baseline_selected = list(
                (baseline.config_snapshot or {}).get("selected_case_ids") or []
            )
            # 历史 run 以空列表表达“数据集全集”；数据集版本不可变，因此可以从
            # 同一 dataset 安全恢复精确集合，不修改历史 snapshot。
            if not baseline_selected:
                baseline_selected = dataset_case_ids
            if baseline_selected != selected_case_ids:
                raise ResourceConflictError(
                    "baseline and candidate must use the same complete case set in dataset order"
                )

        raw_formal = config_snapshot.get(FORMAL_EVALUATION_SNAPSHOT_KEY)
        formal_intent: dict[str, Any] | None = None
        if raw_formal is not None:
            if (
                not isinstance(raw_formal, dict)
                or raw_formal.get("intent_version") != FORMAL_EVALUATION_INTENT_VERSION
            ):
                raise ResourceConflictError(
                    "new formal evaluation runs require a server-finalized intent"
                )
            try:
                formal_intent = freeze_formal_evaluation_intent(raw_formal)
            except (TypeError, ValueError) as exc:
                raise ResourceConflictError("formal evaluation intent is invalid") from exc
            intent_dataset = formal_intent["dataset"]
            intent_expectations = {
                "execution mode": (mode, formal_intent["execution_mode"]),
                "dataset record id": (dataset.id, intent_dataset["record_id"]),
                "dataset checksum": (dataset.checksum, intent_dataset["record_checksum"]),
                "dataset key": (dataset.dataset_key, intent_dataset["dataset_key"]),
                "dataset version": (dataset.version, intent_dataset["dataset_version"]),
            }
            drift = [
                label
                for label, (expected, actual) in intent_expectations.items()
                if expected != actual
            ]
            if drift:
                raise ResourceConflictError(
                    "formal evaluation intent does not match the run: " + ", ".join(drift)
                )
            if baseline_run_id:
                if (
                    baseline_formal is None
                    or baseline_formal.get("contract_version") != FORMAL_EVALUATION_CONTRACT_VERSION
                ):
                    raise ResourceConflictError(
                        "formal candidate requires a current formal baseline authority"
                    )
                if baseline_formal.get("intent") != formal_intent:
                    raise ResourceConflictError(
                        "baseline and candidate must use the same formal evaluation intent"
                    )
        elif baseline_formal is not None:
            raise ResourceConflictError(
                "baseline and candidate must use the same formal evaluation authority"
            )

        run_id = new_id("erun")
        snapshot_to_freeze = {
            **dict(config_snapshot),
            "selected_case_ids": selected_case_ids,
        }
        if formal_intent is not None:
            snapshot_to_freeze[FORMAL_EVALUATION_SNAPSHOT_KEY] = (
                finalize_formal_evaluation_contract(
                    formal_intent,
                    config_snapshot=snapshot_to_freeze,
                    run_id=run_id,
                    tenant_id=self.tenant_id,
                    kb_id=kb_id,
                    baseline_run_id=baseline_run_id,
                    baseline_formal_contract_sha256=baseline_formal_sha256,
                    selected_case_ids=selected_case_ids,
                )
            )
        frozen_config_snapshot = _freeze_run_config_snapshot(
            snapshot_to_freeze,
            dataset=dataset,
            mode=mode,
            run_id=run_id,
            tenant_id=self.tenant_id,
            kb_id=kb_id,
            baseline_run_id=baseline_run_id,
            baseline_formal_contract_sha256=baseline_formal_sha256,
            selected_case_ids=selected_case_ids,
            total_cases=total,
        )
        if formal_intent is not None and baseline_formal is not None:
            candidate_authority = frozen_config_snapshot[FORMAL_EVALUATION_SNAPSHOT_KEY][
                "run_authority"
            ]
            baseline_authority = baseline_formal["run_authority"]
            allowed_interventions = set(formal_intent["allowed_intervention_dimensions"])
            dimension_fields = {
                "code_revision": ("code_revision",),
                "config_version": ("config_version",),
                "retrieval_profile": ("retrieval_profile",),
                "vector_read_set": ("vector_read_set",),
                "evaluation_security": (
                    "evaluation_security",
                    "evaluation_security_sha256",
                ),
                "model_ids": ("model_ids",),
                "runtime_flags": ("runtime_flags", "runtime_flags_sha256"),
            }
            undeclared_drift = any(
                candidate_authority[field] != baseline_authority[field]
                for dimension, fields in dimension_fields.items()
                if dimension not in allowed_interventions
                for field in fields
            )
            unclassified_drift = (
                candidate_authority["comparison_snapshot_sha256"]
                != baseline_authority["comparison_snapshot_sha256"]
            )
            if undeclared_drift or unclassified_drift:
                raise ResourceConflictError(
                    "baseline and candidate contain an undeclared runtime intervention"
                )
        now = _now()
        run = EvaluationRunModel(
            id=run_id,
            tenant_id=self.tenant_id,
            dataset_id=dataset_id,
            kb_id=kb_id,
            baseline_run_id=baseline_run_id,
            name=name,
            mode=mode,
            status="queued",
            progress=0.0,
            total_cases=total,
            completed_cases=0,
            config_snapshot=frozen_config_snapshot,
            metrics_summary={},
            comparison_summary={},
            gate_result=_execution_gate(
                status="queued",
                owner_id=execution_owner,
                heartbeat_at=now,
            ),
            report_paths={},
            created_by=self.actor_id,
        )
        self.session.add(run)
        self.session.flush()
        return self.run_dict(run)

    def _validated_run_config_snapshot(
        self,
        run: EvaluationRunModel,
        *,
        _visited: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        dataset = self.session.get(EvaluationDatasetModel, run.dataset_id)
        if dataset is None:
            raise ResourceConflictError("evaluation run dataset is unavailable")
        selected_case_ids = list((run.config_snapshot or {}).get("selected_case_ids") or [])
        if not selected_case_ids:
            # 历史 run 以空集合表达整个不可变数据集；只用于兼容旧 contract。
            selected_case_ids = list(
                self.session.scalars(
                    select(EvaluationCaseModel.case_id)
                    .where(EvaluationCaseModel.dataset_id == dataset.id)
                    .order_by(EvaluationCaseModel.ordinal)
                ).all()
            )
        baseline_formal_sha256: str | None = None
        if run.baseline_run_id:
            baseline = self._run_model(run.baseline_run_id, _visited=_visited)
            baseline_formal = (baseline.config_snapshot or {}).get(FORMAL_EVALUATION_SNAPSHOT_KEY)
            if isinstance(baseline_formal, dict):
                baseline_formal_sha256 = str(baseline_formal.get("contract_sha256") or "")
        return _freeze_run_config_snapshot(
            dict(run.config_snapshot or {}),
            dataset=dataset,
            mode=run.mode,
            run_id=run.id,
            tenant_id=run.tenant_id,
            kb_id=run.kb_id,
            baseline_run_id=run.baseline_run_id,
            baseline_formal_contract_sha256=baseline_formal_sha256,
            selected_case_ids=selected_case_ids,
            total_cases=run.total_cases,
        )

    def _run_model(
        self,
        run_id: str,
        *,
        _visited: frozenset[str] = frozenset(),
    ) -> EvaluationRunModel:
        if run_id in _visited:
            raise ResourceConflictError("evaluation baseline chain contains a cycle")
        run = self.session.scalar(
            select(EvaluationRunModel).where(
                EvaluationRunModel.id == run_id,
                EvaluationRunModel.tenant_id == self.tenant_id,
            )
        )
        if run is None:
            raise ResourceNotFoundError(f"evaluation run not found: {run_id}")
        self._validated_run_config_snapshot(
            run,
            _visited=_visited | {run_id},
        )
        return run

    def list_runs(self, *, limit: int = 50) -> list[dict]:
        rows = self.session.scalars(
            select(EvaluationRunModel)
            .where(EvaluationRunModel.tenant_id == self.tenant_id)
            .order_by(EvaluationRunModel.created_at.desc())
            .limit(max(1, min(limit, 200)))
        ).all()
        return [self.run_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict:
        return self.run_dict(self._run_model(run_id))

    def selected_cases_for_run(self, run_id: str) -> list[dict]:
        run = self._run_model(run_id)
        selected = list((run.config_snapshot or {}).get("selected_case_ids") or [])
        query = select(EvaluationCaseModel).where(EvaluationCaseModel.dataset_id == run.dataset_id)
        if selected:
            query = query.where(EvaluationCaseModel.case_id.in_(selected))
        rows = self.session.scalars(query.order_by(EvaluationCaseModel.ordinal)).all()
        return [self.case_dict(row) for row in rows]

    def start_run(self, run_id: str, *, owner_id: str | None = None) -> dict:
        run = self._run_model(run_id)
        ensure_run_transition(run.status, "running")
        execution = _execution_state(run)
        assigned_owner = str(execution.get("owner_id") or "")
        if owner_id and assigned_owner and assigned_owner != owner_id:
            raise EvaluationRunOwnershipLost("evaluation run belongs to another executor")
        now = _now()
        run.status = "running"
        run.started_at = now
        run.error_detail = None
        run.gate_result = _execution_gate(
            status="running",
            owner_id=owner_id or assigned_owner or None,
            heartbeat_at=now,
        )
        self.session.flush()
        return self.run_dict(run)

    def heartbeat_run(self, run_id: str, *, owner_id: str) -> bool:
        """更新执行心跳；状态或 owner 已变化时拒绝迟到执行器。"""

        run = self._run_model(run_id)
        if run.status != "running":
            return False
        execution = _execution_state(run)
        if str(execution.get("owner_id") or "") != owner_id:
            return False
        now = _now()
        expected_updated_at = run.updated_at
        result = self.session.execute(
            update(EvaluationRunModel)
            .where(
                EvaluationRunModel.id == run.id,
                EvaluationRunModel.tenant_id == self.tenant_id,
                EvaluationRunModel.status == "running",
                EvaluationRunModel.updated_at <= expected_updated_at,
            )
            .values(
                gate_result=_execution_gate(
                    status="running",
                    owner_id=owner_id,
                    heartbeat_at=now,
                ),
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            self.session.expire(run)
            refreshed = self._run_model(run_id)
            execution = _execution_state(refreshed)
            return (
                refreshed.status == "running" and str(execution.get("owner_id") or "") == owner_id
            )
        self.session.expire(run)
        return True

    @staticmethod
    def _assert_owned_running(run: EvaluationRunModel, owner_id: str) -> None:
        execution = _execution_state(run)
        if run.status != "running" or str(execution.get("owner_id") or "") != owner_id:
            raise EvaluationRunOwnershipLost("evaluation run was interrupted or ownership changed")

    def record_result(
        self,
        run_id: str,
        case_id: str,
        *,
        observation: dict,
        metrics: dict,
        passed: bool,
        error_type: str | None = None,
        error_detail: str | None = None,
        owner_id: str | None = None,
    ) -> dict:
        run = self._run_model(run_id)
        if owner_id:
            self._assert_owned_running(run, owner_id)
        existing = self.session.scalar(
            select(EvaluationResultModel).where(
                EvaluationResultModel.run_id == run_id,
                EvaluationResultModel.case_id == case_id,
            )
        )
        if existing is None:
            existing = EvaluationResultModel(
                id=_stable_id("eres", run_id, case_id),
                run_id=run_id,
                case_id=case_id,
                observation=observation,
                metrics=metrics,
                passed=passed,
                error_type=error_type,
                error_detail=error_detail,
                latency_ms=observation.get("latency_ms"),
            )
            self.session.add(existing)
            run.completed_cases = min(run.total_cases, run.completed_cases + 1)
        else:
            existing.observation = observation
            existing.metrics = metrics
            existing.passed = passed
            existing.error_type = error_type
            existing.error_detail = error_detail
            existing.latency_ms = observation.get("latency_ms")
        run.progress = run.completed_cases / run.total_cases if run.total_cases else 0.0
        self.session.flush()
        return self.result_dict(existing)

    def complete_run(
        self,
        run_id: str,
        *,
        metrics_summary: dict,
        comparison_summary: dict,
        gate_result: dict,
        report_paths: dict,
        owner_id: str | None = None,
        completed_at: datetime | None = None,
    ) -> dict:
        run = self._run_model(run_id)
        if owner_id:
            self._assert_owned_running(run, owner_id)
        ensure_run_transition(run.status, "succeeded")
        run.status = "succeeded"
        run.progress = 1.0
        run.completed_cases = run.total_cases
        run.metrics_summary = metrics_summary
        run.comparison_summary = comparison_summary
        run.gate_result = gate_result
        run.report_paths = report_paths
        finished_at = completed_at or _now()
        if finished_at.tzinfo is None:
            raise ValueError("evaluation completion timestamp must include timezone")
        run.completed_at = finished_at
        self.session.flush()
        return self.run_dict(run)

    def fail_run(
        self,
        run_id: str,
        detail: str,
        *,
        error_code: str = "EVALUATION_RUN_FAILED",
        owner_id: str | None = None,
    ) -> dict:
        run = self._run_model(run_id)
        if run.status in {"succeeded", "failed", "cancelled"}:
            return self.run_dict(run)
        if owner_id:
            self._assert_owned_running(run, owner_id)
        ensure_run_transition(run.status, "failed")
        run.status = "failed"
        run.error_detail = detail[:8000]
        now = _now()
        run.completed_at = now
        run.gate_result = {
            "status": "failed",
            "passed": False,
            "error_code": error_code,
            "reason": detail[:2000],
            "progress": {
                "completed_cases": run.completed_cases,
                "total_cases": run.total_cases,
                "value": run.progress,
            },
            _EXECUTION_KEY: {
                **_execution_state(run),
                "finished_at": _iso(now),
            },
        }
        self.session.flush()
        return self.run_dict(run)

    def interrupt_run(
        self,
        run_id: str,
        *,
        expected_updated_at: datetime,
        error_code: str,
        reason: str,
        expected_owner: str | None = None,
        now: datetime | None = None,
    ) -> dict | None:
        """以 updated_at 作为 CAS 栅栏终结失去执行器的非终态运行。"""

        run = self._run_model(run_id)
        if run.status not in {"queued", "running"}:
            return None
        execution = _execution_state(run)
        owner_id = str(execution.get("owner_id") or "") or None
        if expected_owner is not None and owner_id != expected_owner:
            return None
        expected = _aware(expected_updated_at)
        actual = _aware(run.updated_at)
        if actual != expected:
            return None
        interrupted_at = now or _now()
        if interrupted_at.tzinfo is None:
            raise ValueError("evaluation interruption timestamp must include timezone")
        interrupted_gate = {
            "status": "interrupted",
            "passed": False,
            "error_code": error_code,
            "reason": reason,
            "progress": {
                "completed_cases": run.completed_cases,
                "total_cases": run.total_cases,
                "value": run.progress,
            },
            _EXECUTION_KEY: {
                **execution,
                "owner_id": owner_id,
                "interrupted_at": _iso(interrupted_at),
            },
        }
        result = self.session.execute(
            update(EvaluationRunModel)
            .where(
                EvaluationRunModel.id == run.id,
                EvaluationRunModel.tenant_id == self.tenant_id,
                EvaluationRunModel.status.in_(("queued", "running")),
                EvaluationRunModel.updated_at <= run.updated_at,
            )
            .values(
                status="failed",
                error_detail=reason[:8000],
                gate_result=interrupted_gate,
                completed_at=interrupted_at,
                updated_at=interrupted_at,
            )
        )
        if result.rowcount != 1:
            return None
        self.session.expire(run)
        return self.run_dict(run)

    def list_results(self, run_id: str, *, only_failed: bool = False) -> list[dict]:
        self._run_model(run_id)
        query = select(EvaluationResultModel).where(EvaluationResultModel.run_id == run_id)
        if only_failed:
            query = query.where(EvaluationResultModel.passed.is_(False))
        rows = self.session.scalars(query.order_by(EvaluationResultModel.case_id)).all()
        return [self.result_dict(row) for row in rows]

    @staticmethod
    def result_dict(row: EvaluationResultModel) -> dict:
        return {
            "id": row.id,
            "run_id": row.run_id,
            "case_id": row.case_id,
            "observation": row.observation or {},
            "metrics": row.metrics or {},
            "passed": row.passed,
            "error_type": row.error_type,
            "error_detail": row.error_detail,
            "latency_ms": row.latency_ms,
            "created_at": _iso(row.created_at),
        }

    def run_dict(self, row: EvaluationRunModel) -> dict:
        dataset = self.session.get(EvaluationDatasetModel, row.dataset_id)
        if dataset is None:
            raise ResourceConflictError("evaluation run dataset is unavailable")
        config_snapshot = self._validated_run_config_snapshot(row, _visited=frozenset({row.id}))
        dataset_snapshot = dict(dataset.source_snapshot or {}) if dataset is not None else {}
        manifest_json = (
            dict(dataset_snapshot["manifest_json"])
            if isinstance(dataset_snapshot.get("manifest_json"), dict)
            else {}
        )
        return {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "dataset_key": dataset.dataset_key if dataset else None,
            "dataset_version": dataset.version if dataset else None,
            "dataset_checksum": dataset.checksum if dataset else None,
            "kb_id": row.kb_id,
            "baseline_run_id": row.baseline_run_id,
            "name": row.name,
            "mode": row.mode,
            "status": row.status,
            "progress": row.progress,
            "total_cases": row.total_cases,
            "completed_cases": row.completed_cases,
            "config_snapshot": config_snapshot,
            "metrics_summary": row.metrics_summary or {},
            "comparison_summary": row.comparison_summary or {},
            "gate_result": row.gate_result or {},
            "report_paths": row.report_paths or {},
            "error_code": (row.gate_result or {}).get("error_code"),
            "error_detail": row.error_detail,
            "created_by": row.created_by,
            "started_at": _iso(row.started_at),
            "completed_at": _iso(row.completed_at),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            **_suite_projection(manifest_json),
        }

    # ------------------------------------------------------------------
    # Feedback -> two-person annotation -> arbitration
    # ------------------------------------------------------------------
    def create_feedback(self, payload: dict) -> dict:
        rating = int(payload["rating"])
        if rating not in {-1, 0, 1}:
            raise ValueError("feedback rating must be -1, 0 or 1")
        evaluation_result_id = payload.get("evaluation_result_id")
        if evaluation_result_id:
            scoped = self.session.execute(
                select(EvaluationResultModel, EvaluationRunModel)
                .join(
                    EvaluationRunModel,
                    EvaluationRunModel.id == EvaluationResultModel.run_id,
                )
                .where(
                    EvaluationResultModel.id == evaluation_result_id,
                    EvaluationRunModel.tenant_id == self.tenant_id,
                )
            ).first()
            if scoped is None:
                raise ResourceNotFoundError(f"evaluation result not found: {evaluation_result_id}")
            scoped_result, scoped_run = scoped
            if scoped_run.kb_id != str(payload["kb_id"]):
                raise ResourceConflictError(
                    "feedback knowledge base does not match evaluation result"
                )
            observed_answer_id = str((scoped_result.observation or {}).get("answer_id") or "")
            if observed_answer_id and observed_answer_id != str(payload["answer_id"]):
                raise ResourceConflictError("feedback answer_id does not match evaluation result")
        row = FeedbackModel(
            id=new_id("fb"),
            tenant_id=self.tenant_id,
            kb_id=str(payload["kb_id"]),
            actor_id=self.actor_id,
            answer_id=str(payload["answer_id"]),
            evaluation_result_id=evaluation_result_id,
            session_id=payload.get("session_id"),
            rating=rating,
            reason_codes=list(payload.get("reason_codes") or []),
            comment=str(payload.get("comment", "")),
            selected_citation=dict(payload.get("selected_citation") or {}),
            corrected_answer=str(payload.get("corrected_answer", "")),
            status=FeedbackStatus.PENDING.value,
        )
        self.session.add(row)
        self.session.flush()
        return self.feedback_dict(row)

    def _feedback_model(self, feedback_id: str) -> FeedbackModel:
        row = self.session.scalar(
            select(FeedbackModel).where(
                FeedbackModel.id == feedback_id,
                FeedbackModel.tenant_id == self.tenant_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError(f"feedback not found: {feedback_id}")
        return row

    def list_feedback(self, *, status: str | None = None, mine: bool = False) -> list[dict]:
        query = select(FeedbackModel).where(FeedbackModel.tenant_id == self.tenant_id)
        if status:
            query = query.where(FeedbackModel.status == status)
        if mine:
            query = query.where(FeedbackModel.actor_id == self.actor_id)
        rows = self.session.scalars(query.order_by(FeedbackModel.created_at.desc())).all()
        return [self.feedback_dict(row) for row in rows]

    def triage_feedback(self, feedback_id: str, *, status: str, review_note: str) -> dict:
        row = self._feedback_model(feedback_id)
        ensure_feedback_transition(row.status, status)
        row.status = status
        row.reviewed_by = self.actor_id
        row.review_note = review_note
        self.session.flush()
        return self.feedback_dict(row)

    @staticmethod
    def feedback_dict(row: FeedbackModel) -> dict:
        return {
            "id": row.id,
            "kb_id": row.kb_id,
            "actor_id": row.actor_id,
            "answer_id": row.answer_id,
            "evaluation_result_id": row.evaluation_result_id,
            "session_id": row.session_id,
            "rating": row.rating,
            "reason_codes": row.reason_codes or [],
            "comment": row.comment,
            "selected_citation": row.selected_citation or {},
            "corrected_answer": row.corrected_answer,
            "status": row.status,
            "reviewed_by": row.reviewed_by,
            "review_note": row.review_note,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    def create_annotation_task(
        self, feedback_id: str, *, target_dataset_key: str, proposed_case: dict
    ) -> dict:
        feedback = self._feedback_model(feedback_id)
        if feedback.status == FeedbackStatus.PENDING.value:
            ensure_feedback_transition(feedback.status, FeedbackStatus.TRIAGED.value)
            feedback.status = FeedbackStatus.TRIAGED.value
        if feedback.status == FeedbackStatus.TRIAGED.value:
            ensure_feedback_transition(feedback.status, FeedbackStatus.IN_REVIEW.value)
            feedback.status = FeedbackStatus.IN_REVIEW.value
        if feedback.status != FeedbackStatus.IN_REVIEW.value:
            raise ResourceConflictError("feedback is not eligible for annotation")
        existing = self.session.scalar(
            select(AnnotationTaskModel).where(
                AnnotationTaskModel.feedback_id == feedback_id,
                AnnotationTaskModel.tenant_id == self.tenant_id,
            )
        )
        if existing is not None:
            return self.annotation_dict(existing)
        row = AnnotationTaskModel(
            id=new_id("ann"),
            tenant_id=self.tenant_id,
            feedback_id=feedback_id,
            target_dataset_key=target_dataset_key,
            proposed_case=proposed_case,
            reviews=[],
            status=AnnotationStatus.IN_REVIEW.value,
            resolution={},
        )
        self.session.add(row)
        self.session.flush()
        return self.annotation_dict(row)

    def _annotation_model(self, task_id: str) -> AnnotationTaskModel:
        row = self.session.scalar(
            select(AnnotationTaskModel).where(
                AnnotationTaskModel.id == task_id,
                AnnotationTaskModel.tenant_id == self.tenant_id,
            )
        )
        if row is None:
            raise ResourceNotFoundError(f"annotation task not found: {task_id}")
        return row

    def list_annotation_tasks(self) -> list[dict]:
        rows = self.session.scalars(
            select(AnnotationTaskModel)
            .where(AnnotationTaskModel.tenant_id == self.tenant_id)
            .order_by(AnnotationTaskModel.created_at.desc())
        ).all()
        return [self.annotation_dict(row) for row in rows]

    def review_annotation(self, task_id: str, *, decision: str, note: str) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("annotation decision must be approve or reject")
        row = self._annotation_model(task_id)
        if row.status not in {AnnotationStatus.IN_REVIEW.value, AnnotationStatus.PENDING.value}:
            raise ResourceConflictError("annotation task is no longer reviewable")
        reviews = list(row.reviews or [])
        if any(review.get("reviewer_id") == self.actor_id for review in reviews):
            raise ResourceConflictError("the same reviewer cannot provide both reviews")
        reviews.append(
            {
                "reviewer_id": self.actor_id,
                "decision": decision,
                "note": note,
                "reviewed_at": _iso(_now()),
            }
        )
        row.reviews = reviews
        if len(reviews) >= 2:
            first, second = reviews[0], reviews[1]
            if first["decision"] == second["decision"]:
                row.status = (
                    AnnotationStatus.APPROVED.value
                    if decision == "approve"
                    else AnnotationStatus.REJECTED.value
                )
                row.resolution = {"kind": "review_consensus", "decision": decision}
                feedback = self._feedback_model(row.feedback_id)
                feedback.status = (
                    FeedbackStatus.ACCEPTED.value
                    if decision == "approve"
                    else FeedbackStatus.REJECTED.value
                )
            else:
                row.status = AnnotationStatus.CONFLICT.value
        else:
            row.status = AnnotationStatus.IN_REVIEW.value
        self.session.flush()
        return self.annotation_dict(row)

    def arbitrate_annotation(
        self, task_id: str, *, decision: str, note: str, resolved_case: dict | None
    ) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("arbitration decision must be approve or reject")
        row = self._annotation_model(task_id)
        if row.status != AnnotationStatus.CONFLICT.value:
            raise ResourceConflictError("only conflicting annotations can be arbitrated")
        if any(review.get("reviewer_id") == self.actor_id for review in (row.reviews or [])):
            raise ResourceConflictError(
                "annotation arbiter must be independent from both reviewers"
            )
        row.arbiter_id = self.actor_id
        row.status = (
            AnnotationStatus.APPROVED.value
            if decision == "approve"
            else AnnotationStatus.REJECTED.value
        )
        row.resolution = {
            "kind": "arbitration",
            "decision": decision,
            "note": note,
            "resolved_case": resolved_case or row.proposed_case,
            "resolved_at": _iso(_now()),
        }
        feedback = self._feedback_model(row.feedback_id)
        feedback.status = (
            FeedbackStatus.ACCEPTED.value
            if decision == "approve"
            else FeedbackStatus.REJECTED.value
        )
        self.session.flush()
        return self.annotation_dict(row)

    @staticmethod
    def annotation_dict(row: AnnotationTaskModel) -> dict:
        return {
            "id": row.id,
            "feedback_id": row.feedback_id,
            "target_dataset_key": row.target_dataset_key,
            "proposed_case": row.proposed_case or {},
            "reviews": row.reviews or [],
            "status": row.status,
            "arbiter_id": row.arbiter_id,
            "resolution": row.resolution or {},
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }
