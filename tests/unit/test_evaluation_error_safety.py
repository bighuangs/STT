from __future__ import annotations

import json
import inspect
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.application import evaluation_runtime_maintenance, evaluation_service
from app.application.evaluation_target import EvaluationTargetResult
from app.domain.evaluation import freeze_evaluation_gate_policy
from app.domain.evaluation_proof import (
    ProofEvaluationAuthority,
    ProofEvidenceAuthority,
    ProofRepresentationAuthority,
)
from app.domain.retrieval import RetrievalEngineError
from app.evaluation.dataset_loader import load_dataset
from app.infra.persistence.sql.database import (
    configure_database,
    dispose_database,
    session_scope,
)
from app.infra.persistence.sql.evaluation_repository import EvaluationRepository
from app.infra.persistence.sql.models import (
    Base,
    EvaluationCaseModel,
    EvaluationDatasetModel,
    EvaluationResultModel,
    EvaluationRunModel,
)
from app.infrastructure.model_calls.query_retry import (
    QueryModelRetryPolicy,
    invoke_query_model_call,
)
from app.security.principal import Principal
from app.shared.runtime.logger import node_log


class _LoggerProbe:
    """截获日志调用，避免失败测试把合成敏感值写到测试日志。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple, dict]] = []

    def error(self, *args, **kwargs) -> None:
        self.events.append(("error", args, kwargs))

    def exception(self, *args, **kwargs) -> None:
        self.events.append(("exception", args, kwargs))


class _SyntheticValidationModel(BaseModel):
    safe_field: int


def test_evaluation_maintenance_never_logs_raw_exception_tracebacks() -> None:
    """后台恢复失败也只能记录稳定定位字段。"""

    source = inspect.getsource(evaluation_runtime_maintenance)
    assert "logger.exception" not in source
    assert "exc_info=" not in source


class _FailingAdapter:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def evaluate(self, _request) -> dict:
        payload = " | ".join(f"{name}={value}" for name, value in self._values.items())
        raise RuntimeError(payload)


class _SuccessfulAdapter:
    def evaluate(self, _request) -> dict:
        return {
            "answer": "",
            "retrieved": [],
            "reranked": [],
            "context": [],
            "latency_ms": 1.0,
            "tokens": None,
            "cost": None,
            "versions": {},
            "scope": "retrieval_only",
        }


class _AuthoritySidecarAdapter(_SuccessfulAdapter):
    def __init__(self, raw_text: str) -> None:
        self._raw_text = raw_text
        self.last_authority: ProofEvaluationAuthority | None = None

    def evaluate(self, request) -> EvaluationTargetResult:
        authority = ProofEvaluationAuthority(
            question_contract_json="{}",
            witness_catalog_json="{}",
            witness_rows_json=(),
            evidences=(
                ProofEvidenceAuthority(
                    evidence_id="evd_authority_sidecar",
                    canonical_text=self._raw_text,
                ),
            ),
            representations=(
                ProofRepresentationAuthority(
                    representation_id="rep_authority_sidecar",
                    packed_text=self._raw_text,
                ),
            ),
            source_ledger_complete=False,
            witness_ledger_complete=False,
        )
        self.last_authority = authority
        return EvaluationTargetResult(
            observation=super().evaluate(request),
            proof_authority=authority,
        )


def _raise_evaluation_node_error(_state: dict, error: Exception) -> None:
    raise error


def _sensitive_values() -> dict[str, str]:
    """使用拆分常量，确保测试失败输出不会意外展示完整合成值。"""

    return {
        "exception": "".join(("SYNTH", "-EXCEPTION-", "NO-PERSIST")),
        "query": "".join(("SYNTH", "-QUERY-", "NO-PERSIST")),
        "evidence": "".join(("SYNTH", "-EVIDENCE-", "NO-PERSIST")),
        "credential": "".join(("sk-test-", "SYNTH-", "NO-PERSIST")),
    }


def _assert_no_sensitive_value(payload: bytes, values: dict[str, str]) -> None:
    if any(value.encode("utf-8") in payload for value in values.values()):
        pytest.fail("合成敏感值进入了持久化或日志投影", pytrace=False)


def _decode_safe_error(value: str) -> dict:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        pytest.fail("错误详情不是类型化安全契约", pytrace=False)
    if not isinstance(decoded, dict):
        pytest.fail("错误详情不是对象契约", pytrace=False)
    return decoded


@pytest.fixture
def evaluation_environment(tmp_path: Path, monkeypatch):
    dispose_database()
    database_path = tmp_path / "evaluation-safety.db"
    engine = configure_database(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(
        engine,
        tables=[
            EvaluationDatasetModel.__table__,
            EvaluationCaseModel.__table__,
            EvaluationRunModel.__table__,
            EvaluationResultModel.__table__,
        ],
    )
    report_root = tmp_path / "var" / "evals"
    monkeypatch.setattr(evaluation_service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_service, "REPORT_ROOT", report_root)
    logger_probe = _LoggerProbe()
    monkeypatch.setattr(evaluation_service, "logger", logger_probe)
    try:
        yield database_path, report_root, logger_probe
    finally:
        dispose_database()


def _create_run(*, tenant_id: str, owner_id: str) -> tuple[str, Principal]:
    manifest = deepcopy(load_dataset())
    manifest["dataset_key"] = "evaluation-safe-error-fixture"
    manifest["version"] = "1.0.0-test"
    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id=tenant_id,
            actor_id="evaluation-test",
        )
        dataset = repository.seed_manifest(manifest)
        case_id = str(dataset["cases"][0]["case_id"])
        run = repository.create_run(
            dataset_id=dataset["id"],
            kb_id="kb-evaluation-safety",
            name="safe error fixture",
            mode="retrieval_only",
            baseline_run_id=None,
            config_snapshot={
                "code_revision": "test",
                "gate_policy": freeze_evaluation_gate_policy({}),
            },
            case_ids=[case_id],
            execution_owner=owner_id,
        )
    principal = Principal(
        user_id="evaluation-test",
        tenant_id=tenant_id,
        roles=("admin",),
        scopes=("*",),
        auth_type="test",
        credential_id="test-credential",
    )
    return str(run["id"]), principal


def _scan_persistent_outputs(
    *,
    database_path: Path,
    report_root: Path,
    values: dict[str, str],
) -> None:
    for path in database_path.parent.glob(f"{database_path.name}*"):
        if path.is_file():
            _assert_no_sensitive_value(path.read_bytes(), values)
    for path in report_root.rglob("*"):
        if path.is_file():
            _assert_no_sensitive_value(path.read_bytes(), values)


def test_unknown_exception_text_cannot_change_public_type_or_fingerprint() -> None:
    values = _sensitive_values()
    first = evaluation_service._safe_evaluation_error(
        RuntimeError(values["exception"]),
        stage="case_observation",
    )
    second = evaluation_service._safe_evaluation_error(
        RuntimeError(values["query"]),
        stage="case_observation",
    )

    assert first == second
    assert first["schema"] == "stt.eval.safe-error.v2"
    assert first["type"] == "EvaluationCaseExecutionError"
    assert first["code"] == "EVALUATION_CASE_FAILED"
    assert first["diagnostics"] == {}
    _assert_no_sensitive_value(evaluation_service._safe_error_json(first).encode(), values)


def test_node_origin_metadata_is_projected_without_exception_text() -> None:
    values = _sensitive_values()
    raw_error = RuntimeError(values["exception"])
    wrapped = node_log("node_answer_render")(_raise_evaluation_node_error)

    with pytest.raises(RuntimeError) as caught:
        wrapped({"trace_id": "trace-evaluation-safe-001"}, raw_error)

    safe_error = evaluation_service._safe_evaluation_error(
        caught.value,
        stage="case_observation",
    )
    diagnostics = safe_error["diagnostics"]
    assert diagnostics == {
        "origin_node": "node_answer_render",
        "origin_function": "_raise_evaluation_node_error",
        "origin_exception_type": "RuntimeError",
        "origin_fingerprint": getattr(raw_error, "origin_fingerprint"),
    }
    assert len(diagnostics["origin_fingerprint"]) == 64
    _assert_no_sensitive_value(
        evaluation_service._safe_error_json(safe_error).encode(),
        values,
    )


def test_pydantic_diagnostics_keep_only_model_location_and_error_type() -> None:
    values = _sensitive_values()
    with pytest.raises(ValidationError) as caught:
        _SyntheticValidationModel.model_validate({"safe_field": values["evidence"]})

    safe_error = evaluation_service._safe_evaluation_error(
        caught.value,
        stage="case_observation",
    )

    assert safe_error["type"] == "EvaluationContractError"
    assert safe_error["diagnostics"] == {
        "validation_model": "_SyntheticValidationModel",
        "validation_error_count": 1,
        "validation_locations": ("safe_field",),
        "validation_error_types": ("int_parsing",),
    }
    _assert_no_sensitive_value(
        evaluation_service._safe_error_json(safe_error).encode(),
        values,
    )


def test_structured_model_diagnostics_use_fixed_v2_contract() -> None:
    values = _sensitive_values()
    logical_call_id = "qmc_" + ("a" * 32)

    def build_error(
        message: str,
        *,
        scene: str = "answer_plan_semantic_review",
    ) -> ValueError:
        error = ValueError(message)
        error.safe_component = "structured_answer_model"
        error.safe_scene = scene
        error.safe_failure_stage = "wire_contract"
        error.safe_failure_reason_code = "DUPLICATE_JSON_KEY"
        error.safe_provider_attempt_count = 3
        error.safe_logical_call_id = logical_call_id
        return error

    first = evaluation_service._safe_evaluation_error(
        build_error(values["exception"]),
        stage="case_observation",
    )
    second = evaluation_service._safe_evaluation_error(
        build_error(values["query"]),
        stage="case_observation",
    )
    different_scene = evaluation_service._safe_evaluation_error(
        build_error(values["query"], scene="answer_plan_selection"),
        stage="case_observation",
    )

    assert first == second
    assert first["schema"] == "stt.eval.safe-error.v2"
    assert first["type"] == "StructuredAnswerModelError"
    assert first["code"] == "STRUCTURED_ANSWER_MODEL_INVALID"
    assert first["retryable"] is False
    assert different_scene["type"] == first["type"]
    assert different_scene["code"] == first["code"]
    assert different_scene["fingerprint"] != first["fingerprint"]
    assert first["diagnostics"] == {
        "component": "structured_answer_model",
        "scene": "answer_plan_semantic_review",
        "failure_stage": "wire_contract",
        "failure_reason_code": "DUPLICATE_JSON_KEY",
        "provider_attempt_count": 3,
        "logical_call_id": logical_call_id,
    }
    _assert_no_sensitive_value(evaluation_service._safe_error_json(first).encode(), values)


def test_unknown_diagnostic_values_are_dropped_without_hashing_them() -> None:
    values = _sensitive_values()

    def build_error(value: str) -> RuntimeError:
        error = RuntimeError(value)
        error.origin_node = value
        error.origin_function = value
        error.origin_exception_type = "RuntimeError"
        error.origin_fingerprint = "a" * 64
        error.safe_component = value
        error.safe_scene = value
        error.safe_failure_stage = value
        error.safe_failure_reason_code = value
        error.safe_provider_attempt_count = True
        error.safe_logical_call_id = value
        return error

    first = evaluation_service._safe_evaluation_error(
        build_error(values["exception"]),
        stage="case_observation",
    )
    second = evaluation_service._safe_evaluation_error(
        build_error(values["query"]),
        stage="case_observation",
    )

    assert first == second
    assert first["diagnostics"] == {}
    assert first["type"] == "EvaluationCaseExecutionError"
    assert first["code"] == "EVALUATION_CASE_FAILED"
    _assert_no_sensitive_value(evaluation_service._safe_error_json(first).encode(), values)


def test_retryable_retrieval_error_preserves_only_whitelisted_diagnostics() -> None:
    values = _sensitive_values()
    error = evaluation_service._safe_evaluation_error(
        RetrievalEngineError(
            "RETRIEVAL_TIMEOUT",
            values["evidence"],
            retryable=True,
            details={"credential": values["credential"]},
        ),
        stage="case_observation",
    )

    assert error["type"] == "RetrievalEngineError"
    assert error["code"] == "RETRIEVAL_TIMEOUT"
    assert error["retryable"] is True
    _assert_no_sensitive_value(evaluation_service._safe_error_json(error).encode(), values)


def test_transport_exhaustion_maps_to_retryable_no_leak_evaluation_error() -> None:
    """不同网络正文必须收敛到同一安全评测指纹。"""

    safe_errors: list[dict] = []
    raw_errors: list[Exception] = []
    for marker in ("synthetic-network-a", "synthetic-network-b"):
        transport = httpx.ConnectError(
            marker,
            request=httpx.Request(
                "POST",
                "https://provider.invalid/v1/chat/completions",
            ),
        )

        def invoke(_remaining_seconds: float, *, current=transport) -> None:
            raise current

        try:
            invoke_query_model_call(
                {},
                scene="answer_generation",
                model_id="test-model",
                invoke=invoke,
                policy=QueryModelRetryPolicy(
                    max_attempts=1,
                    base_delay_seconds=0.0,
                    max_delay_seconds=0.0,
                    total_timeout_seconds=10.0,
                ),
            )
        except Exception as exc:
            raw_errors.append(exc)
            safe_errors.append(
                evaluation_service._safe_evaluation_error(
                    exc,
                    stage="case_observation",
                )
            )
        else:
            pytest.fail("网络异常未传播到评测安全边界", pytrace=False)

    assert [type(item).__name__ for item in raw_errors] == [
        "QueryModelCallTransportUnavailable",
        "QueryModelCallTransportUnavailable",
    ]
    assert safe_errors[0] == safe_errors[1]
    assert safe_errors[0]["type"] == "QueryModelCallTransportUnavailable"
    assert safe_errors[0]["code"] == "EVALUATION_DEPENDENCY_UNAVAILABLE"
    assert safe_errors[0]["retryable"] is True
    serialized = evaluation_service._safe_error_json(safe_errors[0])
    assert "synthetic-network" not in serialized


def test_failed_case_uses_typed_safe_error_in_db_and_every_report_artifact(
    evaluation_environment,
) -> None:
    database_path, report_root, logger_probe = evaluation_environment
    values = _sensitive_values()
    owner_id = "evaluation-test:case-owner"
    run_id, principal = _create_run(tenant_id="tenant-case-error", owner_id=owner_id)

    evaluation_service.execute_run(
        run_id,
        principal,
        adapter=_FailingAdapter(values),
        owner_id=owner_id,
        heartbeat_seconds=60,
    )

    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
        )
        run = repository.get_run(run_id)
        result = repository.list_results(run_id)[0]

    assert run["status"] == "succeeded"
    assert set(run["report_paths"]) == {
        "html",
        "json",
        "manifest",
        "markdown",
        "results",
        "summary",
    }
    contract = _decode_safe_error(result["error_detail"])
    assert contract == result["observation"]["error"]
    assert result["error_type"] == contract["type"]
    assert contract["schema"] == "stt.eval.safe-error.v2"
    assert contract["diagnostics"] == {}
    assert contract["code"] == "EVALUATION_CASE_FAILED"
    assert contract["retryable"] is False
    assert len(contract["fingerprint"]) == 64
    if any(level == "exception" for level, _args, _kwargs in logger_probe.events):
        pytest.fail("评测错误路径仍在记录原始异常堆栈", pytrace=False)
    _assert_no_sensitive_value(repr(logger_probe.events).encode("utf-8"), values)
    _scan_persistent_outputs(
        database_path=database_path,
        report_root=report_root,
        values=values,
    )


def test_raw_proof_authority_sidecar_is_released_before_every_persistence_path(
    evaluation_environment,
) -> None:
    database_path, report_root, logger_probe = evaluation_environment
    values = _sensitive_values()
    owner_id = "evaluation-test:authority-owner"
    run_id, principal = _create_run(tenant_id="tenant-authority-sidecar", owner_id=owner_id)
    adapter = _AuthoritySidecarAdapter(values["evidence"])

    target_result = adapter.evaluate(object())
    assert values["evidence"] not in repr(target_result)
    evaluation_service.execute_run(
        run_id,
        principal,
        adapter=adapter,
        owner_id=owner_id,
        heartbeat_seconds=60,
    )

    with session_scope() as session:
        repository = EvaluationRepository(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
        )
        result = repository.list_results(run_id)[0]
    _assert_no_sensitive_value(json.dumps(result, ensure_ascii=False).encode(), values)
    _assert_no_sensitive_value(repr(logger_probe.events).encode(), values)
    _scan_persistent_outputs(
        database_path=database_path,
        report_root=report_root,
        values=values,
    )


def test_raw_proof_authority_is_released_before_run_failure_persistence(
    evaluation_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, report_root, logger_probe = evaluation_environment
    values = _sensitive_values()
    owner_id = "evaluation-test:authority-failure-owner"
    run_id, principal = _create_run(
        tenant_id="tenant-authority-sidecar-failure",
        owner_id=owner_id,
    )
    adapter = _AuthoritySidecarAdapter(values["evidence"])
    original_fail_run = EvaluationRepository.fail_run
    lifecycle_checked = False

    def fail_evaluation(*_args, proof_authority=None, **_kwargs):
        assert proof_authority is adapter.last_authority
        raise RuntimeError(values["exception"])

    def checked_fail_run(repository, *args, **kwargs):
        nonlocal lifecycle_checked
        authority = adapter.last_authority
        assert authority is not None
        execute_frames = [
            frame.frame for frame in inspect.stack() if frame.function == "execute_run"
        ]
        assert len(execute_frames) == 1
        execute_locals = execute_frames[0].f_locals
        assert execute_locals.get("proof_authority") is None
        assert execute_locals.get("target_result") is None
        failure = execute_locals.get("exc")
        traceback = getattr(failure, "__traceback__", None)
        while traceback is not None:
            assert all(value is not authority for value in traceback.tb_frame.f_locals.values())
            traceback = traceback.tb_next
        lifecycle_checked = True
        return original_fail_run(repository, *args, **kwargs)

    monkeypatch.setattr(evaluation_service, "evaluate_case", fail_evaluation)
    monkeypatch.setattr(EvaluationRepository, "fail_run", checked_fail_run)

    evaluation_service.execute_run(
        run_id,
        principal,
        adapter=adapter,
        owner_id=owner_id,
        heartbeat_seconds=60,
    )

    assert lifecycle_checked is True
    _assert_no_sensitive_value(repr(logger_probe.events).encode("utf-8"), values)
    _scan_persistent_outputs(
        database_path=database_path,
        report_root=report_root,
        values=values,
    )


def test_run_failure_persists_only_safe_contract_and_never_raw_traceback(
    evaluation_environment,
    monkeypatch,
) -> None:
    database_path, report_root, logger_probe = evaluation_environment
    values = _sensitive_values()
    owner_id = "evaluation-test:run-owner"
    run_id, principal = _create_run(tenant_id="tenant-run-error", owner_id=owner_id)

    def fail_artifact_publish(_payload: dict) -> dict:
        payload = " | ".join(f"{name}={value}" for name, value in values.items())
        raise RuntimeError(payload)

    monkeypatch.setattr(evaluation_service, "_write_artifacts", fail_artifact_publish)
    evaluation_service.execute_run(
        run_id,
        principal,
        adapter=_SuccessfulAdapter(),
        owner_id=owner_id,
        heartbeat_seconds=60,
    )

    with session_scope() as session:
        run = EvaluationRepository(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
        ).get_run(run_id)

    assert run["status"] == "failed"
    contract = _decode_safe_error(run["error_detail"])
    assert contract["schema"] == "stt.eval.safe-error.v2"
    assert contract["diagnostics"] == {}
    assert contract["code"] == "EVALUATION_RUN_FAILED"
    assert contract["retryable"] is False
    assert len(contract["fingerprint"]) == 64
    assert run["gate_result"]["error_code"] == contract["code"]
    if any(level == "exception" for level, _args, _kwargs in logger_probe.events):
        pytest.fail("评测运行错误仍在记录原始异常堆栈", pytrace=False)
    _assert_no_sensitive_value(repr(logger_probe.events).encode("utf-8"), values)
    _scan_persistent_outputs(
        database_path=database_path,
        report_root=report_root,
        values=values,
    )
