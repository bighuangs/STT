"""TODO-18 application service: immutable datasets, durable runs and reviewed feedback."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

from pydantic import ValidationError as PydanticValidationError

from app.evaluation import (
    OBSERVATION_SCHEMA_V3,
    aggregate_results,
    evaluate_case,
    is_v3_observation,
    paired_diff,
    project_proof_observation,
    render_html_report,
    render_json_report,
    render_markdown_report,
)
from app.evaluation.dataset_loader import load_dataset, verify_source
from app.evaluation.observation import (
    EvalObservationAdapter,
    QueryGraphEvalAdapter,
    RetrievalOnlyEvalAdapter,
)
from app.application.evaluation_target import (
    EvaluationRequest,
    EvaluationTargetResult,
    TargetInput,
)
from app.infra.config import settings
from app.infra.persistence.sql.database import initialize_control_database, session_scope
from app.infra.persistence.sql.evaluation_repository import EvaluationRepository
from app.infra.persistence.sql.evaluation_repository import EvaluationRunOwnershipLost
from app.infra.persistence.sql.query_runtime_repository import QueryRuntimeRepository
from app.infra.persistence.sql.repositories import ControlRepository, ResourceConflictError
from app.foundation.artifact_publisher import (
    discard_artifact_publication,
    publish_text_artifacts,
)
from app.application.model_ports import QueryModelCallTransportUnavailable
from app.application.query_authorization_service import (
    EntitlementUnavailableError,
    QueryAuthorizationService,
)
from app.application.query_security_pipeline import QueryContentSecurityViolation
from app.application.query_answering.proposals import MAX_PROPOSAL_BYTES
from app.domain.query_security import (
    DataClassification,
    ExternalDataRequest,
    RequestedRetrievalScope,
    ScopeAuthorizationError,
)
from app.domain.evaluation import (
    AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
    CASE_PASS_POLICY_VERSION,
    EVALUATION_GATE_POLICY_VERSION,
    LEGACY_CASE_PASS_POLICY_VERSION,
    LEGACY_EVALUATION_GATE_POLICY_VERSION,
    PREVIOUS_CASE_PASS_POLICY_VERSION,
    SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
    TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    freeze_case_pass_policy,
    freeze_evaluation_gate_policy,
)
from app.domain.retrieval import RetrievalEngineError, RetrievalScope
from app.infra.retrieval.control_plane_resolvers import freeze_vector_read_set
from app.infra.security.sql_query_authorization import (
    SqlActiveIndexBindingResolver,
    SqlEntitlementResolver,
    SqlScopeDecisionAudit,
    resolve_egress_policy,
)
from app.security.principal import (
    Principal,
    current_principal,
    reset_current_principal,
    set_current_principal,
)
from app.security.service import assert_query_session_owner
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.application.evaluation_runtime_maintenance import (
    DEFAULT_HEARTBEAT_SECONDS,
    evaluation_execution_owner,
)


REPORT_ROOT = PROJECT_ROOT / "var" / "evals"
LEGACY_REPORT_ROOT = PROJECT_ROOT / "output" / "evaluation"

_COMPLETENESS_PRESERVED_REASON = "POST_GENERATION_COMPLETENESS_PRESERVED"
_STRICT_GROUNDING_INELIGIBLE_REASON = "visible_answer_not_strict_grounding_eligible"
_COMPLETENESS_PRESERVED_BADGE_CODE = "ANSWER_COMPLETENESS_PRESERVED_CITATION_DEGRADED"
_COMPLETENESS_PRESERVED_BADGE_LABEL = "答案完整性优先，部分内容未通过引用校验"

_SAFE_ERROR_SCHEMA = "stt.eval.safe-error.v2"
_SAFE_ERROR_STAGES = frozenset(
    {
        "case_observation",
        "failure_persistence",
        "preflight_vector_read_set",
        "run_execution",
        "run_heartbeat",
    }
)
_SAFE_SECURITY_STAGES = frozenset({"input", "evidence", "context", "output", "fetch"})
_SAFE_SECURITY_ACTIONS = frozenset({"block", "quarantine", "review"})
_SAFE_SECURITY_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_SAFE_RETRIEVAL_ERROR_CODES = frozenset(
    {
        "RETRIEVAL_UNAVAILABLE",
        "RETRIEVAL_TIMEOUT",
        "RETRIEVAL_DIMENSION_MISMATCH",
        "RETRIEVAL_STALE_SCOPE",
        "RETRIEVAL_FILTER_UNSUPPORTED",
        "RETRIEVAL_INVALID_RESPONSE",
        "RERANKER_UNAVAILABLE",
        "RERANKER_INVALID_RESPONSE",
    }
)
_SAFE_ORIGIN_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/-]{1,160}$")
_SAFE_ORIGIN_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DIAGNOSTIC_COMPONENTS = frozenset({"structured_answer_model"})
_SAFE_DIAGNOSTIC_SCENES = frozenset({"answer_plan_selection", "answer_plan_semantic_review"})
_SAFE_DIAGNOSTIC_FAILURE_STAGES = frozenset(
    {
        "endpoint_validation",
        "authorization",
        "client_init",
        "bind",
        "transport",
        "wire_contract",
        "parser_invariant",
        "budget_reconciliation",
        "retry_control",
        "server_projection",
    }
)
_SAFE_DIAGNOSTIC_FAILURE_REASONS = frozenset(
    {
        "WIRE_CONTRACT_INVALID",
        "INVALID_JSON",
        "DUPLICATE_JSON_KEY",
        "NON_FINITE_JSON_NUMBER",
        "INVALID_UNICODE",
        "PAYLOAD_TOO_LARGE",
        "ROOT_NOT_OBJECT",
        "SCHEMA_INVALID",
        "ALLOWLIST_VIOLATION",
        "REPAIR_STATE_INVALID",
    }
)
_SAFE_LOGICAL_CALL_ID = re.compile(r"^qmc_[0-9a-f]{32}$")
_SAFE_VALIDATION_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SAFE_VALIDATION_ERROR_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class _SafeEvaluationDiagnostics(TypedDict, total=False):
    """评测错误允许持久化的低基数来源诊断。"""

    origin_node: str
    origin_function: str
    origin_exception_type: str
    origin_fingerprint: str
    component: str
    scene: str
    failure_stage: str
    failure_reason_code: str
    provider_attempt_count: int
    logical_call_id: str
    validation_model: str
    validation_error_count: int
    validation_locations: tuple[str, ...]
    validation_error_types: tuple[str, ...]


class _SafeEvaluationError(TypedDict):
    """评测持久化边界允许保存的最小错误契约。"""

    schema: str
    stage: str
    code: str
    type: str
    safe_message: str
    retryable: bool
    diagnostics: _SafeEvaluationDiagnostics
    fingerprint: str


def _exception_attributes(exc: Exception) -> dict[str, object]:
    """仅读取异常实例字典，避免触发异常子类的动态属性逻辑。"""

    try:
        attributes = object.__getattribute__(exc, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return attributes if isinstance(attributes, dict) else {}


def _origin_failure_fingerprint(
    *,
    node_name: str,
    function_name: str,
    exception_type: str,
) -> str:
    payload = json.dumps(
        {
            "exception_type": exception_type,
            "function": function_name,
            "operation": node_name,
            "stage": "node",
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_evaluation_diagnostics(exc: Exception) -> _SafeEvaluationDiagnostics:
    """按字段名、枚举与格式白名单投影内部异常诊断。"""

    attributes = _exception_attributes(exc)
    diagnostics: _SafeEvaluationDiagnostics = {}

    if isinstance(exc, PydanticValidationError):
        title = str(exc.title)
        if _SAFE_VALIDATION_IDENTIFIER.fullmatch(title):
            diagnostics["validation_model"] = title
        raw_errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        diagnostics["validation_error_count"] = min(len(raw_errors), 1_000)
        locations: list[str] = []
        error_types: list[str] = []
        for item in raw_errors[:8]:
            raw_type = item.get("type")
            if isinstance(raw_type, str) and _SAFE_VALIDATION_ERROR_TYPE.fullmatch(raw_type):
                error_types.append(raw_type)
            raw_location = item.get("loc")
            if not isinstance(raw_location, tuple):
                continue
            segments: list[str] = []
            valid_location = True
            for segment in raw_location:
                if isinstance(segment, int) and not isinstance(segment, bool) and 0 <= segment <= 1_024:
                    segments.append(str(segment))
                elif isinstance(segment, str) and _SAFE_VALIDATION_IDENTIFIER.fullmatch(segment):
                    segments.append(segment)
                else:
                    valid_location = False
                    break
            if valid_location and segments:
                locations.append(".".join(segments))
        diagnostics["validation_locations"] = tuple(locations)
        diagnostics["validation_error_types"] = tuple(error_types)

    origin_node = attributes.get("origin_node")
    origin_function = attributes.get("origin_function")
    origin_exception_type = attributes.get("origin_exception_type")
    origin_fingerprint = attributes.get("origin_fingerprint")
    expected_exception_type = type(exc).__name__
    if not (
        isinstance(expected_exception_type, str)
        and len(expected_exception_type) <= 160
        and _SAFE_ORIGIN_IDENTIFIER.fullmatch(expected_exception_type)
    ):
        expected_exception_type = "Exception"
    if (
        isinstance(origin_node, str)
        and len(origin_node) <= 64
        and _SAFE_ORIGIN_IDENTIFIER.fullmatch(origin_node)
        and isinstance(origin_function, str)
        and _SAFE_ORIGIN_IDENTIFIER.fullmatch(origin_function)
        and origin_exception_type == expected_exception_type
        and isinstance(origin_fingerprint, str)
        and _SAFE_ORIGIN_FINGERPRINT.fullmatch(origin_fingerprint)
        and origin_fingerprint
        == _origin_failure_fingerprint(
            node_name=origin_node,
            function_name=origin_function,
            exception_type=origin_exception_type,
        )
    ):
        diagnostics.update(
            {
                "origin_node": origin_node,
                "origin_function": origin_function,
                "origin_exception_type": origin_exception_type,
                "origin_fingerprint": origin_fingerprint,
            }
        )

    component = attributes.get("safe_component")
    if isinstance(component, str) and component in _SAFE_DIAGNOSTIC_COMPONENTS:
        diagnostics["component"] = component
    scene = attributes.get("safe_scene")
    if isinstance(scene, str) and scene in _SAFE_DIAGNOSTIC_SCENES:
        diagnostics["scene"] = scene
    failure_stage = attributes.get("safe_failure_stage")
    if isinstance(failure_stage, str) and failure_stage in _SAFE_DIAGNOSTIC_FAILURE_STAGES:
        diagnostics["failure_stage"] = failure_stage
    failure_reason = attributes.get("safe_failure_reason_code")
    if isinstance(failure_reason, str) and failure_reason in _SAFE_DIAGNOSTIC_FAILURE_REASONS:
        diagnostics["failure_reason_code"] = failure_reason
    provider_attempt_count = attributes.get("safe_provider_attempt_count")
    if (
        isinstance(provider_attempt_count, int)
        and not isinstance(provider_attempt_count, bool)
        and 0 <= provider_attempt_count <= 3
    ):
        diagnostics["provider_attempt_count"] = provider_attempt_count
    logical_call_id = attributes.get("safe_logical_call_id")
    if isinstance(logical_call_id, str) and _SAFE_LOGICAL_CALL_ID.fullmatch(logical_call_id):
        diagnostics["logical_call_id"] = logical_call_id
    return diagnostics


def _make_safe_error(
    *,
    stage: str,
    code: str,
    error_type: str,
    safe_message: str,
    retryable: bool,
    diagnostics: Mapping[str, object] | None = None,
) -> _SafeEvaluationError:
    """仅用白名单字段生成可关联指纹，绝不散列异常正文。"""

    normalized_stage = stage if stage in _SAFE_ERROR_STAGES else "run_execution"
    safe_diagnostics = dict(diagnostics or {})
    fingerprint_source = {
        "schema": _SAFE_ERROR_SCHEMA,
        "stage": normalized_stage,
        "code": code,
        "type": error_type,
        "retryable": bool(retryable),
        "diagnostics": safe_diagnostics,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_source,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": _SAFE_ERROR_SCHEMA,
        "stage": normalized_stage,
        "code": code,
        "type": error_type,
        "safe_message": safe_message,
        "retryable": bool(retryable),
        "diagnostics": safe_diagnostics,
        "fingerprint": fingerprint,
    }


def _stage_default_error(
    stage: str,
    *,
    diagnostics: Mapping[str, object] | None = None,
) -> _SafeEvaluationError:
    if stage == "case_observation":
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_CASE_FAILED",
            error_type="EvaluationCaseExecutionError",
            safe_message="评测样本执行失败，已记录安全诊断标识。",
            retryable=False,
            diagnostics=diagnostics,
        )
    if stage == "run_heartbeat":
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_HEARTBEAT_FAILED",
            error_type="EvaluationHeartbeatError",
            safe_message="评测运行心跳暂时不可用。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if stage == "failure_persistence":
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_FAILURE_PERSISTENCE_FAILED",
            error_type="EvaluationFailurePersistenceError",
            safe_message="评测失败终态未能持久化。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if stage == "preflight_vector_read_set":
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_PREFLIGHT_FAILED",
            error_type="EvaluationPreflightError",
            safe_message="评测预检未能冻结向量读集。",
            retryable=False,
            diagnostics=diagnostics,
        )
    return _make_safe_error(
        stage="run_execution",
        code="EVALUATION_RUN_FAILED",
        error_type="EvaluationRunExecutionError",
        safe_message="评测运行失败，已记录安全诊断标识。",
        retryable=False,
        diagnostics=diagnostics,
    )


def _safe_evaluation_error(exc: Exception, *, stage: str) -> _SafeEvaluationError:
    """把异常归入固定白名单；未知异常按所属阶段 fail-closed。"""

    diagnostics = _safe_evaluation_diagnostics(exc)

    structured_diagnostic_fields = {
        "component",
        "scene",
        "failure_stage",
        "failure_reason_code",
        "provider_attempt_count",
        "logical_call_id",
    }
    if structured_diagnostic_fields.issubset(diagnostics):
        return _make_safe_error(
            stage=stage,
            code="STRUCTURED_ANSWER_MODEL_INVALID",
            error_type="StructuredAnswerModelError",
            safe_message="结构化回答模型未能返回有效契约。",
            retryable=False,
            diagnostics=diagnostics,
        )

    if isinstance(exc, QueryContentSecurityViolation):
        return _make_safe_error(
            stage=stage,
            code="QUERY_CONTENT_SECURITY_BLOCKED",
            error_type="QueryContentSecurityViolation",
            safe_message="查询内容触发安全策略，已阻断执行。",
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(exc, EvaluationRunOwnershipLost):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_RUN_INTERRUPTED",
            error_type="EvaluationRunOwnershipLost",
            safe_message="评测运行执行权已丢失，运行已安全中止。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(exc, RetrievalEngineError):
        code = str(exc.code)
        if code not in _SAFE_RETRIEVAL_ERROR_CODES:
            return _stage_default_error(stage, diagnostics=diagnostics)
        return _make_safe_error(
            stage=stage,
            code=code,
            error_type="RetrievalEngineError",
            safe_message="评测检索依赖未能返回可用结果。",
            retryable=bool(exc.retryable),
            diagnostics=diagnostics,
        )
    if isinstance(exc, EntitlementUnavailableError):
        return _make_safe_error(
            stage=stage,
            code="ENTITLEMENT_UNAVAILABLE",
            error_type="EntitlementUnavailableError",
            safe_message="评测权限事实源暂时不可用。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(exc, ScopeAuthorizationError):
        return _make_safe_error(
            stage=stage,
            code="RETRIEVAL_SCOPE_DENIED",
            error_type="ScopeAuthorizationError",
            safe_message="评测检索范围未获授权。",
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(exc, ResourceConflictError):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_STATE_CONFLICT",
            error_type="ResourceConflictError",
            safe_message="评测运行状态与当前操作冲突。",
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(exc, QueryModelCallTransportUnavailable):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_DEPENDENCY_UNAVAILABLE",
            error_type="QueryModelCallTransportUnavailable",
            safe_message="评测依赖暂时不可用。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(exc, TimeoutError):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_DEPENDENCY_TIMEOUT",
            error_type="TimeoutError",
            safe_message="评测依赖调用超时。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(exc, ConnectionError):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_DEPENDENCY_UNAVAILABLE",
            error_type="ConnectionError",
            safe_message="评测依赖暂时不可用。",
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(exc, (TypeError, ValueError)):
        return _make_safe_error(
            stage=stage,
            code="EVALUATION_CONTRACT_INVALID",
            error_type="EvaluationContractError",
            safe_message="评测运行数据未满足内部契约。",
            retryable=False,
            diagnostics=diagnostics,
        )
    return _stage_default_error(stage, diagnostics=diagnostics)


def _safe_error_json(error: Mapping[str, object]) -> str:
    """以稳定 JSON 保存安全契约，避免自由文本再次扩张。"""

    return json.dumps(
        dict(error),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_security_projection(exc: Exception) -> dict | None:
    """只投影安全异常中受控枚举字段，不复制异常消息。"""

    if not isinstance(exc, QueryContentSecurityViolation):
        return None
    stage = str(exc.stage).lower()
    action = str(exc.action).lower()
    if stage not in _SAFE_SECURITY_STAGES:
        stage = "input"
    if action not in _SAFE_SECURITY_ACTIONS:
        action = "block"
    reason_codes = [
        code
        for code in (str(value) for value in exc.reason_codes)
        if _SAFE_SECURITY_REASON_CODE.fullmatch(code)
    ][:20]
    return {
        "stage": stage,
        "action": action,
        "reason_codes": reason_codes,
        "retrieval_called": False,
    }


def _log_safe_error(
    event: str,
    error: Mapping[str, object],
    *,
    run_id: str,
    case_id: str | None = None,
) -> None:
    """日志仅包含低基数错误类别与关联指纹。"""

    logger.error(
        "{}: run={}, case={}, code={}, retryable={}, fingerprint={}",
        event,
        run_id,
        case_id or "not_applicable",
        error["code"],
        error["retryable"],
        error["fingerprint"],
    )


_SOURCE_MANIFEST_SUFFIXES = frozenset(
    {
        ".css",
        ".html",
        ".js",
        ".json",
        ".prompt",
        ".py",
        ".sql",
        ".ts",
        ".yaml",
        ".yml",
    }
)
_ALEMBIC_SOURCE_SUFFIXES = _SOURCE_MANIFEST_SUFFIXES | {".ini", ".mako"}
_SOURCE_MANIFEST_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
)


def _configured_llm_target_domain() -> str:
    """从服务端 LLM URL 提取可冻结的单一安全域名。"""

    parsed = urlsplit(str(settings.openai_base_url or "").strip())
    host = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return host


def _evaluation_security_snapshot(*, kb: dict, mode: str) -> dict:
    """仅从当前服务端身份和控制面事实生成评测安全快照。"""

    principal = current_principal()
    if principal is None:
        return {
            "scope_authorized": False,
            "reason_code": "AUTHENTICATED_PRINCIPAL_REQUIRED",
            "authorized_scope": {},
            "data_classification": str(kb.get("data_classification") or "restricted"),
            "external_llm_allowed": False,
            "answer_generator_configured": False,
            "egress_policies": {},
        }

    authorization = QueryAuthorizationService(
        entitlements=SqlEntitlementResolver(),
        active_indexes=SqlActiveIndexBindingResolver(),
        audit=SqlScopeDecisionAudit(principal),
    )
    try:
        scope = authorization.authorize(
            principal=principal,
            requested=RequestedRetrievalScope(kb_ids=(str(kb["id"]),)),
        )
    except ScopeAuthorizationError:
        return {
            "scope_authorized": False,
            "reason_code": "RETRIEVAL_SCOPE_DENIED",
            "authorized_scope": {},
            "data_classification": str(kb.get("data_classification") or "restricted"),
            "external_llm_allowed": False,
            "answer_generator_configured": False,
            "egress_policies": {},
        }
    except EntitlementUnavailableError:
        return {
            "scope_authorized": False,
            "reason_code": "ENTITLEMENT_UNAVAILABLE",
            "authorized_scope": {},
            "data_classification": str(kb.get("data_classification") or "restricted"),
            "external_llm_allowed": False,
            "answer_generator_configured": False,
            "egress_policies": {},
        }

    raw_classification = str(kb.get("data_classification") or "internal").lower()
    try:
        classification = DataClassification(raw_classification)
    except ValueError:
        classification = DataClassification.RESTRICTED
    llm_target_domain = _configured_llm_target_domain()
    llm_policy = resolve_egress_policy(
        tenant_id=principal.tenant_id,
        kb_ids=(str(kb["id"]),),
        purpose="llm_generation",
        allowed_providers=("configured-llm",),
        allowed_domains=((llm_target_domain,) if llm_target_domain else ()),
    )
    web_policy = resolve_egress_policy(
        tenant_id=principal.tenant_id,
        kb_ids=(str(kb["id"]),),
        purpose="web_search",
        allowed_providers=("dashscope-web",),
        allowed_domains=settings.query_web_allowed_domains,
    )
    generator_configured = bool(llm_target_domain and settings.llm_default_model)
    external_llm_allowed = False
    reason_code = "NOT_REQUIRED_FOR_RETRIEVAL_ONLY"
    if mode == "live":
        decision = authorization.decide_egress(
            request=ExternalDataRequest(
                requested=True,
                provider="configured-llm",
                purpose="llm_generation",
                scope_hash=scope.scope_hash,
                data_classes=(classification,),
                target_domain=llm_target_domain or None,
            ),
            policy=llm_policy,
        )
        external_llm_allowed = bool(decision.allowed and generator_configured)
        reason_code = (
            "EGRESS_ALLOWED"
            if external_llm_allowed
            else (
                "QUERY_GENERATOR_UNAVAILABLE"
                if decision.allowed and not generator_configured
                else decision.reason_code
            )
        )
    return {
        "scope_authorized": True,
        "reason_code": reason_code,
        "authorized_scope": scope.model_dump(mode="json"),
        "data_classification": classification.value,
        "external_llm_allowed": external_llm_allowed,
        "answer_generator_configured": generator_configured,
        "egress_policies": {
            "llm_generation": llm_policy.model_dump(mode="json"),
            "web_search": web_policy.model_dump(mode="json"),
        },
        "content_security_policy": {
            "policy_version": "rag-security.evaluation.v1",
            "web_enabled": False,
            "high_risk_sources_enabled": False,
            "allowed_fetch_domains": [],
            "allowed_output_link_domains": list(settings.query_output_link_allowed_domains),
        },
    }


def _controlled_source_paths(root: Path) -> tuple[Path, ...]:
    """Return the reproducibility-critical source set in stable relative order."""

    candidates: set[Path] = set()
    for name in ("pyproject.toml", "alembic.ini"):
        path = root / name
        if path.is_file():
            candidates.add(path)

    for tree_name, suffixes in (
        ("app", _SOURCE_MANIFEST_SUFFIXES),
        ("alembic", _ALEMBIC_SOURCE_SUFFIXES),
    ):
        tree = root / tree_name
        if not tree.is_dir():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(root)
            if any(part in _SOURCE_MANIFEST_EXCLUDED_PARTS for part in relative.parts):
                continue
            candidates.add(path)
    return tuple(sorted(candidates, key=lambda path: path.relative_to(root).as_posix()))


def _source_manifest_digest(root: Path) -> str:
    """Hash stable relative paths and exact bytes without scanning runtime artifacts."""

    digest = hashlib.sha256()
    digest.update(b"stt-evaluation-source-manifest.v1\0")
    for path in _controlled_source_paths(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        # Length framing prevents path/content boundary ambiguity while retaining the
        # exact bytes that the interpreter or browser consumes.
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_directory(root: Path) -> Path | None:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    try:
        value = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value.startswith("gitdir: "):
        return None
    target = Path(value[8:].strip())
    return target if target.is_absolute() else (root / target).resolve()


def _git_head_revision(root: Path) -> str:
    git_dir = _git_directory(root)
    if git_dir is None:
        return "worktree-unknown"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "worktree-unknown"
    if not head.startswith("ref: "):
        return head[:40] or "worktree-unknown"

    ref_name = head[5:].strip()
    try:
        return (git_dir / ref_name).read_text(encoding="utf-8").strip()[:40]
    except OSError:
        try:
            packed_refs = (git_dir / "packed-refs").read_text(encoding="utf-8")
        except OSError:
            return "worktree-unknown"
        for line in packed_refs.splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            revision, separator, name = line.partition(" ")
            if separator and name.strip() == ref_name:
                return revision[:40]
        return "worktree-unknown"


def _code_revision() -> str:
    explicit = os.getenv("STT_CODE_REVISION", "").strip()
    if explicit:
        return explicit
    return (
        f"head:{_git_head_revision(PROJECT_ROOT)};"
        f"source-sha256:{_source_manifest_digest(PROJECT_ROOT)}"
    )


def _snapshot(
    dataset: dict,
    preflight: dict,
    *,
    mode: str,
    formal_evaluation_contract: Mapping[str, object] | None = None,
) -> dict:
    manifest = dataset.get("manifest_json") or {}
    if not isinstance(manifest, dict):
        manifest = {}
    snapshot = {
        "code_revision": _code_revision(),
        # 该值属于回答实现契约，而不是 retrieval profile 的旧预留默认值。
        # 正式 run authority 必须记录实际传给 provider 的上限。
        "answering_contract": {
            "schema_version": "answering-output-contract.v1",
            "generation_output_token_cap": MAX_PROPOSAL_BYTES,
        },
        "config_version": settings.config_version,
        "dataset_version": dataset["version"],
        "dataset_checksum": dataset["checksum"],
        "source_sha256": (dataset.get("source") or {}).get("content_sha256"),
        "evaluation_mode": mode,
        "embedding_model": settings.bge_m3_path or settings.bge_m3,
        "reranker_model": settings.bge_reranker_large,
        "retrieval_profile": preflight.get("retrieval_profile") or {},
        # 评测运行必须复用接受时冻结的物理读集；否则默认空列表会让每个 case
        # 在检索阶段失败，运行期间重新解析又会破坏可重放性。
        "vector_read_scopes": list(preflight.get("vector_read_scopes") or []),
        "vector_read_set_digest": str(preflight.get("vector_read_set_digest") or ""),
        # 只能来自 evaluation_preflight 的服务端授权结果；创建 API 不接受这些字段。
        "evaluation_security": dict(preflight.get("evaluation_security") or {}),
        "llm_model": settings.llm_default_model,
        # 正式运行权威会散列整个 runtime snapshot；这里仅冻结会改变实际执行语义、
        # 且允许耐久保存的非敏感开关，避免仅凭代码版本猜测运行时配置。
        "runtime_flags": {
            "query_default_deadline_ms": settings.query_default_deadline_ms,
            "llm_request_timeout_seconds": settings.llm_request_timeout,
            "llm_max_attempts": settings.llm_max_attempts,
            "llm_retry_base_delay_seconds": settings.llm_retry_base_delay,
            "llm_retry_max_delay_seconds": settings.llm_retry_max_delay,
            "reranker_failure_mode": settings.bge_reranker_failure_mode,
            "vector_placement_mode": settings.vector_placement_mode,
            "vector_placement_policy_version": settings.vector_placement_policy_version,
        },
        # A manifest may omit newly introduced fields, but every new run freezes
        # a complete, versioned server-side policy. Historical snapshots are not
        # backfilled or reinterpreted.
        "gate_policy": freeze_evaluation_gate_policy(manifest.get("gate_policy")),
        "preflight": preflight,
        "telemetry_coverage": {
            "latency": "available",
            "tokens": "unavailable until TODO-14 installs a graph-wide collector",
            "cost": "unavailable until TODO-14 installs provider pricing snapshots",
            "structured_citations": "available via answer.v1",
            "generation": "available" if mode == "live" else "not_applicable",
            "external_llm": "enabled" if mode == "live" else "disabled",
        },
    }
    if formal_evaluation_contract is not None:
        snapshot["formal_evaluation"] = dict(formal_evaluation_contract)
    return snapshot


def ensure_builtin_dataset(*, tenant_id: str | None = None, actor_id: str | None = None) -> dict:
    manifest = load_dataset()
    with session_scope() as session:
        return EvaluationRepository(session, tenant_id=tenant_id, actor_id=actor_id).seed_manifest(
            manifest
        )


def bootstrap_evaluation() -> None:
    initialize_control_database()
    ensure_builtin_dataset()


def list_datasets() -> list[dict]:
    ensure_builtin_dataset()
    with session_scope() as session:
        return EvaluationRepository(session).list_datasets()


def get_dataset(dataset_id: str) -> dict:
    ensure_builtin_dataset()
    with session_scope() as session:
        return EvaluationRepository(session).get_dataset(dataset_id)


def import_dataset(manifest: dict, *, parent_dataset_id: str | None = None) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).seed_manifest(
            manifest, parent_dataset_id=parent_dataset_id
        )


def evaluation_preflight(
    dataset_id: str,
    kb_id: str,
    *,
    mode: str = "retrieval_only",
    retrieval_profile_id: str | None = None,
    retrieval_profile_version: int | None = None,
) -> dict:
    from app.infra.persistence.sql.retrieval_repository import RetrievalProfileRepository

    if mode not in {"live", "retrieval_only"}:
        raise ValueError("evaluation mode must be live or retrieval_only")

    with session_scope() as session:
        dataset = EvaluationRepository(session).get_dataset(dataset_id)
        control = ControlRepository(session)
        kb_model = control.get_knowledge_base_model(kb_id)
        kb = control.knowledge_base_dict(kb_model)
        kb_tenant_id = kb_model.tenant_id
        documents = control.list_documents(kb_id)
        indexes = control.list_index_versions(kb_id)
        retrieval_profile = {}
        if any(item.get("status") == "active" for item in indexes):
            profile_repository = RetrievalProfileRepository(session)
            retrieval_profile = profile_repository.profile_dict(
                profile_repository.resolve_for_evaluation(
                    kb_id,
                    profile_id=retrieval_profile_id,
                    version=retrieval_profile_version,
                )
            )

    evaluation_security = _evaluation_security_snapshot(kb=kb, mode=mode)

    source = dataset.get("source") or {}
    expected_name = source.get("logical_name")
    expected_hash = str(source.get("content_sha256") or "").lower()
    matching_versions: list[dict] = []
    for document in documents:
        if expected_name and document.get("logical_name") != expected_name:
            continue
        for version in document.get("versions") or []:
            if str(version.get("content_sha256") or "").lower() == expected_hash:
                matching_versions.append(
                    {
                        "document_id": document["id"],
                        "document_status": document["status"],
                        "active_version_id": document.get("active_version_id"),
                        "active_index_version_id": document.get("active_index_version_id"),
                        "chunk_count": document.get("chunk_count", 0),
                        "document_version": version,
                    }
                )
    active_indexes = [item for item in indexes if item.get("status") == "active"]
    vector_read_scopes: list[dict] = []
    vector_read_set_digest = ""
    vector_read_set_error: dict | None = None
    principal = current_principal()
    try:
        _resolved, vector_read_scopes, vector_read_set_digest = freeze_vector_read_set(
            RetrievalScope(
                tenant_id=kb_tenant_id,
                user_id=(principal.user_id if principal is not None else "evaluation-preflight"),
                kb_ids=(kb_id,),
            )
        )
    except RetrievalEngineError as exc:
        safe_error = _safe_evaluation_error(exc, stage="preflight_vector_read_set")
        vector_read_set_error = {
            **safe_error,
            # 保留既有 API 的 message 字段，但内容只来自安全契约。
            "message": safe_error["safe_message"],
        }
    source_integrity = verify_source({"source": source}, PROJECT_ROOT)
    ready_matches = [
        item
        for item in matching_versions
        if item["document_status"] == "ready"
        and item["active_version_id"] == item["document_version"]["id"]
        and item["active_index_version_id"]
        and int(item.get("chunk_count") or 0) > 0
    ]
    checks = {
        "query_scope_authorized": bool(evaluation_security.get("scope_authorized")),
        "knowledge_base_active": kb.get("status") == "active",
        "web_search_disabled": not bool(kb.get("web_search_allowed")),
        "source_file_integrity": bool(source_integrity.get("verified")),
        "matching_document_version": bool(matching_versions),
        "matching_document_published": bool(ready_matches),
        "active_index": bool(active_indexes),
        "retrieval_profile_resolved": bool(retrieval_profile),
        "vector_read_set": bool(vector_read_scopes and vector_read_set_digest),
    }
    authorized_scope = evaluation_security.get("authorized_scope") or {}
    authorized_indexes = {
        (
            str(item.get("kb_id") or ""),
            str(item.get("document_id") or ""),
            str(item.get("document_version_id") or ""),
            str(item.get("index_version_id") or ""),
        )
        for item in (authorized_scope.get("active_indexes") or [])
    }
    frozen_indexes = {
        (
            str(item.get("kb_id") or ""),
            str(item.get("document_id") or ""),
            str(item.get("document_version_id") or ""),
            str(item.get("index_version_id") or ""),
        )
        for item in vector_read_scopes
    }
    checks["authorized_vector_read_set"] = bool(
        authorized_indexes and authorized_indexes == frozen_indexes
    )
    if mode == "live":
        checks["answer_generator_configured"] = bool(
            evaluation_security.get("answer_generator_configured")
        )
        checks["external_llm_authorized"] = bool(evaluation_security.get("external_llm_allowed"))
    return {
        "evaluation_mode": mode,
        "ready": all(checks.values()),
        "checks": checks,
        "dataset": {
            "id": dataset["id"],
            "key": dataset["dataset_key"],
            "version": dataset["version"],
            "checksum": dataset["checksum"],
        },
        "knowledge_base": {
            "id": kb["id"],
            "name": kb["name"],
            "status": kb["status"],
            "web_search_allowed": kb.get("web_search_allowed"),
            "external_llm_allowed": kb.get("external_llm_allowed"),
            "data_classification": kb.get("data_classification"),
        },
        "evaluation_security": evaluation_security,
        "source_integrity": source_integrity,
        "matching_versions": matching_versions,
        "active_indexes": active_indexes,
        "vector_read_scopes": vector_read_scopes,
        "vector_read_set_digest": vector_read_set_digest,
        "vector_read_set_error": vector_read_set_error,
        "retrieval_profile": retrieval_profile,
        "retrieval_profile_selection": {
            "source": "explicit" if retrieval_profile_id is not None else "active",
            "profile_id": retrieval_profile.get("profile_id"),
            "version": retrieval_profile.get("version"),
            "config_hash": retrieval_profile.get("config_hash"),
        },
    }


def create_run(
    *,
    dataset_id: str,
    kb_id: str,
    mode: str = "live",
    name: str | None,
    baseline_run_id: str | None,
    case_ids: list[str] | None,
    retrieval_profile_id: str | None = None,
    retrieval_profile_version: int | None = None,
    formal_evaluation_contract: Mapping[str, object] | None = None,
) -> dict:
    if mode not in {"live", "retrieval_only"}:
        raise ValueError("evaluation mode must be live or retrieval_only")
    if (retrieval_profile_id is None) != (retrieval_profile_version is None):
        raise ValueError("retrieval profile id and version must be provided together")
    profile_pin = (
        {
            "retrieval_profile_id": retrieval_profile_id,
            "retrieval_profile_version": retrieval_profile_version,
        }
        if retrieval_profile_id is not None
        else {}
    )
    preflight = evaluation_preflight(
        dataset_id,
        kb_id,
        mode=mode,
        **profile_pin,
    )
    if not preflight["ready"]:
        failed = [name for name, ok in preflight["checks"].items() if not ok]
        raise ResourceConflictError("evaluation preflight failed: " + ", ".join(failed))
    with session_scope() as session:
        repo = EvaluationRepository(session)
        dataset = repo.get_dataset(dataset_id)
        return repo.create_run(
            dataset_id=dataset_id,
            kb_id=kb_id,
            name=name or f"{dataset['dataset_key']}@{dataset['version']}",
            mode=mode,
            baseline_run_id=baseline_run_id,
            config_snapshot=_snapshot(
                dataset,
                preflight,
                mode=mode,
                formal_evaluation_contract=formal_evaluation_contract,
            ),
            case_ids=case_ids,
            execution_owner=evaluation_execution_owner(),
        )


def list_runs(limit: int = 50) -> list[dict]:
    with session_scope() as session:
        return EvaluationRepository(session).list_runs(limit=limit)


def get_run(run_id: str) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).get_run(run_id)


def list_results(run_id: str, *, only_failed: bool = False) -> list[dict]:
    with session_scope() as session:
        return EvaluationRepository(session).list_results(run_id, only_failed=only_failed)


def _last_retrieval_recall(retrieval: dict) -> float | None:
    recall_at_k = retrieval.get("recall_at_k") or {}
    recall = recall_at_k.get("10")
    if recall is None and recall_at_k:
        recall = recall_at_k.get(max(recall_at_k, key=lambda item: int(item)))
    if isinstance(recall, bool) or not isinstance(recall, (int, float)):
        return None
    value = float(recall)
    return value if math.isfinite(value) else None


def _metric_component(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _visible_answer_policy_snapshot(
    evaluated: Mapping[str, object],
    observation: Mapping[str, object] | None,
) -> tuple[dict, str]:
    """Return the archived display policy and its authoritative reason.

    ``observation`` is authoritative for newly executed runs.  The evaluated
    projection is also accepted so deterministic/unit callers can evaluate an
    observation first and decide it later.  Missing fields are intentionally
    treated as the legacy contract, with no inferred degradation.
    """

    raw_policy: object = None
    raw_decision: object = None
    if isinstance(observation, Mapping):
        raw_policy = observation.get("visible_answer_policy")
        raw_decision = observation.get("answer_decision")
    if not isinstance(raw_policy, Mapping) or not raw_policy:
        raw_policy = evaluated.get("visible_answer_policy")
    if not isinstance(raw_decision, Mapping) or not raw_decision:
        raw_decision = evaluated.get("answer_decision")

    policy = deepcopy(dict(raw_policy)) if isinstance(raw_policy, Mapping) else {}
    policy_reason = str(policy.get("reason_code") or policy.get("reason") or "")
    decision_reason = (
        str(raw_decision.get("reason_code") or raw_decision.get("reason") or "")
        if isinstance(raw_decision, Mapping)
        else ""
    )
    return policy, policy_reason or decision_reason


def _enforce_visible_answer_strictness(
    outcome: dict,
    *,
    evaluated: Mapping[str, object],
    observation: Mapping[str, object] | None,
) -> dict:
    """按 observation 版本执行 proof-first 或历史显示策略 strict guard。

    v3 只信任最终 AnswerProof；v1/v2 保持冻结的 visible-answer fallback。
    两条路径都只改变 strict supported/pass，不改最终答案正确性与准确率。
    """

    proof_source: Mapping[str, object] | None = None
    if is_v3_observation(observation):
        proof_source = observation
    elif is_v3_observation(evaluated):
        proof_source = evaluated
    if proof_source is not None:
        metrics = evaluated.get("metrics") if isinstance(evaluated, Mapping) else None
        # 安全阻断由独立 safety 契约裁决，不要求生成 AnswerProof。
        if isinstance(metrics, Mapping) and isinstance(metrics.get("safety"), Mapping):
            return outcome
        raw_proof_audit = metrics.get("answer_proof") if isinstance(metrics, Mapping) else None
        proof_audit = (
            dict(raw_proof_audit)
            if isinstance(raw_proof_audit, Mapping)
            else {
                "available": False,
                "applicable": True,
                "contract_valid": False,
                "strict_eligible": False,
                "strict_graph_valid": False,
                "reason_codes": ["answer_proof_trusted_audit_missing"],
            }
        )
        if proof_audit.get("applicable") is False:
            projected = dict(outcome)
            projected["answer_proof_audit"] = proof_audit
            return projected
        proof_passed = (
            proof_audit.get("contract_valid") is True
            and proof_audit.get("strict_eligible") is True
            and proof_audit.get("strict_graph_valid") is True
        )
        if proof_passed:
            projected = dict(outcome)
            projected["answer_proof_audit"] = proof_audit
            return projected

        guarded = dict(outcome)
        guarded["passed"] = False
        guarded["pass_rate"] = 0.0
        guarded["strict_supported_case_pass"] = False
        guarded["strict_supported_case_pass_rate"] = 0.0
        reasons = list(guarded.get("reason_codes") or [])
        reasons.extend(str(item) for item in proof_audit.get("reason_codes") or [])
        if proof_audit.get("strict_eligible") is not True:
            reasons.append("answer_proof_not_strict_eligible")
        elif proof_audit.get("strict_graph_valid") is not True:
            reasons.append("answer_proof_strict_graph_invalid")
        guarded["reason_codes"] = list(dict.fromkeys(reasons))
        guarded["answer_proof_audit"] = proof_audit
        return guarded

    policy, reason_code = _visible_answer_policy_snapshot(evaluated, observation)
    metrics = evaluated.get("metrics") if isinstance(evaluated, Mapping) else None
    generation = metrics.get("generation") if isinstance(metrics, Mapping) else None
    correctness_components = (
        generation.get("correctness_components") if isinstance(generation, Mapping) else None
    )
    abstention_factual_claim_count = (
        correctness_components.get("abstention_factual_claim_count")
        if isinstance(correctness_components, Mapping)
        else None
    )
    # 合同正确且没有事实主张的拒答没有可做 grounding/citation 的对象；显示层
    # 的 answerable-only 降级标记不应把它改判为严格失败。v2 的回移必须显式
    # 校验事实主张计数，避免错误拒答或夹带事实陈述的拒答借此放行。
    if (
        outcome.get("policy_version")
        in {
            PREVIOUS_CASE_PASS_POLICY_VERSION,
            AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
            TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
        }
        and outcome.get("final_answer_correct") is True
        and "support_not_applicable_to_abstention" in set(outcome.get("diagnostic_codes") or [])
        and isinstance(abstention_factual_claim_count, int)
        and not isinstance(abstention_factual_claim_count, bool)
        and abstention_factual_claim_count == 0
    ):
        return outcome
    grounding_ineligible = policy.get("strict_grounding_eligible") is False
    completeness_preserved = reason_code == _COMPLETENESS_PRESERVED_REASON
    if not grounding_ineligible and not completeness_preserved:
        return outcome

    guarded = dict(outcome)
    guarded["passed"] = False
    guarded["pass_rate"] = 0.0
    guarded["strict_supported_case_pass"] = False
    guarded["strict_supported_case_pass_rate"] = 0.0

    reasons = list(guarded.get("reason_codes") or [])
    if _STRICT_GROUNDING_INELIGIBLE_REASON not in reasons:
        reasons.append(_STRICT_GROUNDING_INELIGIBLE_REASON)
    guarded["reason_codes"] = reasons
    guarded["visible_answer_policy"] = policy
    guarded["answer_display_audit"] = {
        "status": "degraded",
        "badge_code": _COMPLETENESS_PRESERVED_BADGE_CODE,
        "label": _COMPLETENESS_PRESERVED_BADGE_LABEL,
        "reason_code": reason_code or _STRICT_GROUNDING_INELIGIBLE_REASON,
        "strict_grounding_eligible": False,
    }
    return guarded


def _legacy_case_decision(
    case: dict,
    evaluated: dict,
    *,
    had_error: bool,
    mode: str = "live",
) -> dict:
    """Reproduce the pre-v2 decision for already-frozen historical runs."""

    if had_error:
        passed = False
        reasons = ["execution_error"]
    else:
        passed = None
        reasons = []
    metrics = evaluated.get("metrics") or {}
    if passed is None and mode == "retrieval_only":
        retrieval = metrics.get("retrieval") or {}
        if not retrieval.get("applicable", False):
            # 布尔列暂不能表达 N/A；metrics.applicable=false 才是权威语义。
            passed = True
        else:
            recall_10 = _last_retrieval_recall(retrieval)
            passed = recall_10 is not None and recall_10 > 0
            if not passed:
                reasons.append("retrieval_recall_missing")
    generation = metrics.get("generation") or {}
    abstention = metrics.get("abstention") or {}
    if passed is None and case.get("unanswerable"):
        passed = bool(abstention.get("correct"))
        if not passed:
            reasons.append("abstention_incorrect")
    if passed is None:
        expected_recall = generation.get("expected_term_recall")
        forbidden_rate = generation.get("forbidden_term_hit_rate")
        retrieval = metrics.get("retrieval") or {}
        recall_10 = _last_retrieval_recall(retrieval)
        passed = (
            expected_recall is not None
            and expected_recall >= 0.75
            and (forbidden_rate is None or forbidden_rate == 0)
            and (recall_10 is None or recall_10 > 0)
        )
        if expected_recall is None or expected_recall < 0.75:
            reasons.append("expected_term_recall_below_minimum")
        if forbidden_rate is not None and forbidden_rate > 0:
            reasons.append("forbidden_term_hit")
        if recall_10 is not None and recall_10 <= 0:
            reasons.append("retrieval_recall_missing")
    correctness = generation.get("correctness")
    forbidden_rate = generation.get("forbidden_term_hit_rate")
    final_answer_correct = (
        not had_error
        and _metric_component(correctness) >= 0.75
        and (forbidden_rate is None or _metric_component(forbidden_rate) == 0)
    )
    return {
        "policy_version": LEGACY_CASE_PASS_POLICY_VERSION,
        "passed": bool(passed),
        "pass_rate": 1.0 if passed else 0.0,
        "strict_supported_case_pass": bool(passed),
        "strict_supported_case_pass_rate": 1.0 if passed else 0.0,
        "quality_score": _metric_component(correctness),
        "answer_correctness": correctness,
        "final_answer_correctness": correctness,
        "final_answer_correct": final_answer_correct,
        "final_answer_accuracy": 1.0 if final_answer_correct else 0.0,
        "faithfulness": generation.get("faithfulness"),
        "citation_coverage": (metrics.get("citation") or {}).get("coverage"),
        "retrieval_support": None,
        "reason_codes": reasons,
        "diagnostic_codes": [],
    }


def _strict_correctness_failures(
    case: Mapping[str, object],
    generation: Mapping[str, object],
    policy: Mapping[str, object],
) -> list[str]:
    """返回独立于严格引用通过条件的最终正确性失败。"""

    raw_components = generation.get("correctness_components")
    if not isinstance(raw_components, Mapping):
        return ["correctness_contract_missing"]
    components = raw_components
    if bool(case.get("unanswerable")):
        residual_count = components.get("abstention_factual_claim_count")
        if not isinstance(residual_count, int) or isinstance(residual_count, bool):
            return ["correctness_contract_missing"]
        return (
            ["abstention_contains_factual_claim"]
            if residual_count > int(policy["abstention_factual_claim_maximum"])
            else []
        )

    failures: list[str] = []
    policy_version = policy.get("policy_version")
    if (
        policy_version
        in {
            SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
            AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
            TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
        }
        and components.get("concept_contract_valid") is not True
    ):
        failures.append("correctness_contract_missing")
    if components.get("critical_numbers_preserved") is not True:
        failures.append("critical_number_mismatch")
    if components.get("critical_identifiers_preserved") is not True:
        failures.append("critical_identifier_mismatch")
    if components.get("polarity_constraints_preserved") is not True:
        failures.append("polarity_mismatch")

    concept_count = components.get("concept_group_count")
    concept_coverage = components.get("concept_group_coverage")
    if isinstance(concept_count, int) and not isinstance(concept_count, bool):
        if (
            policy_version
            in {
                SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
                AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
                TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
            }
            and isinstance(case.get("concept_groups"), Mapping)
            and concept_count == 0
        ):
            failures.append("correctness_contract_missing")
        if concept_count and (
            not isinstance(concept_coverage, (int, float))
            or isinstance(concept_coverage, bool)
            or not math.isfinite(float(concept_coverage))
            or float(concept_coverage) < float(policy["concept_group_coverage_minimum"])
        ):
            failures.append("concept_group_incomplete")
    else:
        failures.append("correctness_contract_missing")

    if policy_version in {
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    }:
        concept_groups = case.get("concept_groups")
        raw_groups = concept_groups.get("all_of") if isinstance(concept_groups, Mapping) else None
        has_query_predicate_ellipsis = bool(
            isinstance(raw_groups, list)
            and any(
                isinstance(group, Mapping)
                and isinstance(group.get("satisfaction"), Mapping)
                and group["satisfaction"].get("mode") == "query_predicate_ellipsis"
                for group in raw_groups
            )
        )
        has_authored_claim_contract = isinstance(case.get("authored_claims"), Mapping)
        if has_query_predicate_ellipsis or has_authored_claim_contract:
            authored_claim_count = components.get("authored_claim_count")
            authored_claim_coverage = components.get("authored_claim_coverage")
            if (
                not isinstance(authored_claim_count, int)
                or isinstance(authored_claim_count, bool)
                or authored_claim_count <= 0
                or not isinstance(authored_claim_coverage, (int, float))
                or isinstance(authored_claim_coverage, bool)
                or not math.isfinite(float(authored_claim_coverage))
            ):
                failures.append("authored_claim_contract_missing")
            elif float(authored_claim_coverage) < 1.0:
                failures.append("authored_claim_incomplete")

    branch_count = components.get("required_branch_count")
    branch_coverage = components.get("required_branch_coverage")
    if isinstance(branch_count, int) and not isinstance(branch_count, bool):
        if (
            concept_count
            and branch_count
            and (
                not isinstance(branch_coverage, (int, float))
                or isinstance(branch_coverage, bool)
                or not math.isfinite(float(branch_coverage))
                or float(branch_coverage) < float(policy["required_branch_coverage_minimum"])
            )
        ):
            failures.append("required_branch_incomplete")
    else:
        failures.append("correctness_contract_missing")

    count_rules = (
        (
            "unsupported_factual_claim_count",
            "unsupported_factual_claim_maximum",
            "unsupported_factual_claim",
        ),
        (
            "contradictory_factual_claim_count",
            "contradictory_factual_claim_maximum",
            "contradictory_factual_claim",
        ),
    )
    for component, maximum, reason in count_rules:
        value = components.get(component)
        if not isinstance(value, int) or isinstance(value, bool):
            failures.append("correctness_contract_missing")
        elif value > int(policy[maximum]):
            failures.append(reason)
    return list(dict.fromkeys(failures))


def _case_decision(
    case: dict,
    evaluated: dict,
    *,
    had_error: bool,
    mode: str = "live",
    case_policy: dict | None = None,
    observation: Mapping[str, object] | None = None,
) -> dict:
    """Return a durable, explainable case decision without case-id rules."""

    def finalize(outcome: dict) -> dict:
        return _enforce_visible_answer_strictness(
            outcome,
            evaluated=evaluated,
            observation=observation,
        )

    requested_policy = case_policy
    if (
        requested_policy is not None
        and requested_policy.get("policy_version") == LEGACY_CASE_PASS_POLICY_VERSION
    ):
        return finalize(
            _legacy_case_decision(
                case,
                evaluated,
                had_error=had_error,
                mode=mode,
            )
        )
    policy = freeze_case_pass_policy(requested_policy)
    policy_version = str(policy["policy_version"])
    uses_strict_contract = policy_version in {
        CASE_PASS_POLICY_VERSION,
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    }
    metrics = evaluated.get("metrics") or {}
    retrieval = metrics.get("retrieval") or {}
    generation = metrics.get("generation") or {}
    citation = metrics.get("citation") or {}
    abstention = metrics.get("abstention") or {}

    metadata = case.get("metadata")
    security_expectation = (
        metadata.get("security_expectation") if isinstance(metadata, Mapping) else None
    )
    if isinstance(security_expectation, Mapping):
        raw_safety = metrics.get("safety")
        safety = dict(raw_safety) if isinstance(raw_safety, Mapping) else {}
        reasons = [str(code) for code in (safety.get("reason_codes") or []) if code]
        passed = safety.get("passed") is True and not reasons
        if not safety:
            reasons = ["safety_contract_missing"]
            passed = False
        elif not passed and not reasons:
            reasons = ["safety_guardrail_failed"]
        return finalize(
            {
                "policy_version": policy_version,
                "passed": passed,
                "pass_rate": 1.0 if passed else 0.0,
                "strict_supported_case_pass": passed,
                "strict_supported_case_pass_rate": 1.0 if passed else 0.0,
                "quality_score": 1.0 if passed else 0.0,
                "answer_correctness": None,
                "final_answer_correctness": None,
                "final_answer_correct": passed,
                "final_answer_accuracy": None,
                "faithfulness": None,
                "citation_coverage": None,
                "retrieval_support": None,
                "reason_codes": reasons,
                "diagnostic_codes": ["security_guardrail"],
                "safety_guardrail": safety,
            }
        )

    if mode == "retrieval_only":
        applicable = bool(retrieval.get("applicable", False))
        recall = _last_retrieval_recall(retrieval)
        passed = not applicable or (recall is not None and recall > 0)
        return finalize(
            {
                "policy_version": policy_version,
                "passed": passed and not had_error,
                "pass_rate": 1.0 if passed and not had_error else 0.0,
                "strict_supported_case_pass": passed and not had_error,
                "strict_supported_case_pass_rate": (1.0 if passed and not had_error else 0.0),
                "quality_score": None if not applicable else _metric_component(recall),
                "answer_correctness": None,
                "final_answer_correctness": None,
                "final_answer_correct": None,
                "final_answer_accuracy": None,
                "faithfulness": None,
                "citation_coverage": None,
                "retrieval_support": None
                if not applicable
                else (1.0 if recall and recall > 0 else 0.0),
                "reason_codes": (
                    ["execution_error"]
                    if had_error
                    else ([] if passed else ["retrieval_recall_missing"])
                ),
                "diagnostic_codes": ["retrieval_not_applicable"] if not applicable else [],
            }
        )

    coverage_correctness = _metric_component(generation.get("correctness"))
    if case.get("unanswerable"):
        correctness_failures = (
            _strict_correctness_failures(case, generation, policy) if uses_strict_contract else []
        )
        abstention_correct = (
            abstention.get("contract_correct")
            if uses_strict_contract
            else abstention.get("correct")
        )
        final_answer_correct = (
            not had_error and bool(abstention_correct) and not correctness_failures
        )
        correctness = (
            _metric_component(generation.get("contract_correctness"))
            if uses_strict_contract
            else coverage_correctness
        )
        passed = final_answer_correct
        reasons = (
            ["execution_error"]
            if had_error
            else (([] if abstention_correct else ["abstention_incorrect"]) + correctness_failures)
        )
        return finalize(
            {
                "policy_version": policy_version,
                "passed": passed,
                "pass_rate": 1.0 if passed else 0.0,
                "strict_supported_case_pass": passed,
                "strict_supported_case_pass_rate": 1.0 if passed else 0.0,
                "quality_score": correctness,
                "answer_correctness": correctness,
                "coverage_correctness": coverage_correctness,
                "final_answer_correctness": correctness,
                "final_answer_correct": final_answer_correct,
                "final_answer_accuracy": 1.0 if final_answer_correct else 0.0,
                "faithfulness": generation.get("faithfulness"),
                "citation_coverage": citation.get("coverage"),
                "retrieval_support": None,
                "reason_codes": reasons,
                "diagnostic_codes": ["support_not_applicable_to_abstention"],
                "correctness_components": generation.get("correctness_components"),
            }
        )

    faithfulness = _metric_component(generation.get("faithfulness"))
    citation_coverage = _metric_component(citation.get("coverage"))
    recall = _last_retrieval_recall(retrieval)
    retrieval_applicable = bool(retrieval.get("applicable", False))
    retrieval_support = (
        1.0 if not retrieval_applicable else (1.0 if recall is not None and recall > 0 else 0.0)
    )
    correctness_failures = (
        _strict_correctness_failures(case, generation, policy) if uses_strict_contract else []
    )
    correctness = (
        _metric_component(generation.get("contract_correctness"))
        if uses_strict_contract
        else coverage_correctness
    )
    quality_score = (
        correctness * float(policy["answer_correctness_weight"])
        + faithfulness * float(policy["faithfulness_weight"])
        + citation_coverage * float(policy["citation_coverage_weight"])
        + retrieval_support * float(policy["retrieval_support_weight"])
    )
    forbidden_rate = generation.get("forbidden_term_hit_rate")
    final_answer_correct = (
        not had_error
        and correctness >= float(policy["answer_correctness_minimum"])
        and (forbidden_rate is None or _metric_component(forbidden_rate) == 0)
        and not correctness_failures
    )
    reasons: list[str] = []
    if had_error:
        reasons.append("execution_error")
    if correctness < float(policy["answer_correctness_minimum"]):
        reasons.append("answer_correctness_below_minimum")
    if forbidden_rate is not None and _metric_component(forbidden_rate) > 0:
        reasons.append("forbidden_term_hit")
    reasons.extend(correctness_failures)
    # These two independent grounding guards prevent a memorized or lucky
    # answer from passing without evidence, regardless of its lexical score.
    if faithfulness < float(policy["faithfulness_minimum"]):
        reasons.append("faithfulness_below_minimum")
    if citation_coverage < float(policy["citation_coverage_minimum"]):
        reasons.append("citation_coverage_below_minimum")
    if quality_score < float(policy["quality_score_minimum"]):
        reasons.append("quality_score_below_minimum")

    diagnostics: list[str] = []
    if not retrieval_applicable:
        diagnostics.append("retrieval_not_applicable")
    elif retrieval_support == 0:
        diagnostics.append("retrieval_ground_truth_miss")
    return finalize(
        {
            "policy_version": policy_version,
            "passed": not reasons,
            "pass_rate": 1.0 if not reasons else 0.0,
            "strict_supported_case_pass": not reasons,
            "strict_supported_case_pass_rate": 1.0 if not reasons else 0.0,
            "quality_score": quality_score,
            "answer_correctness": correctness,
            "coverage_correctness": coverage_correctness,
            "final_answer_correctness": correctness,
            "final_answer_correct": final_answer_correct,
            "final_answer_accuracy": 1.0 if final_answer_correct else 0.0,
            "faithfulness": faithfulness,
            "citation_coverage": citation_coverage,
            "retrieval_support": retrieval_support,
            "reason_codes": reasons,
            "diagnostic_codes": diagnostics,
            "correctness_components": generation.get("correctness_components"),
            "thresholds": {
                "quality_score": policy["quality_score_minimum"],
                "answer_correctness": policy["answer_correctness_minimum"],
                "faithfulness": policy["faithfulness_minimum"],
                "citation_coverage": policy["citation_coverage_minimum"],
            },
            "weights": {
                "answer_correctness": policy["answer_correctness_weight"],
                "faithfulness": policy["faithfulness_weight"],
                "citation_coverage": policy["citation_coverage_weight"],
                "retrieval_support": policy["retrieval_support_weight"],
            },
        }
    )


def _case_passed(
    case: dict,
    evaluated: dict,
    *,
    had_error: bool,
    mode: str = "live",
    case_policy: dict | None = None,
) -> bool:
    return bool(
        _case_decision(
            case,
            evaluated,
            had_error=had_error,
            mode=mode,
            case_policy=case_policy,
        )["passed"]
    )


def _failed_case_observation(
    *,
    mode: str,
    config_snapshot: dict,
    exception: Exception | None = None,
    error_type: str | None = None,
    error_detail: str | None = None,
) -> dict:
    """保留运行的评测范围，避免错误样本伪装成生成质量观测。"""

    security_projection: dict | None = None
    if exception is not None:
        safe_error = _safe_evaluation_error(exception, stage="case_observation")
        security_projection = _safe_security_projection(exception)
    elif error_type == "QueryContentSecurityViolation":
        safe_error = _make_safe_error(
            stage="case_observation",
            code="QUERY_CONTENT_SECURITY_BLOCKED",
            error_type="QueryContentSecurityViolation",
            safe_message="查询内容触发安全策略，已阻断执行。",
            retryable=False,
        )
        legacy_detail = error_detail or ""
        legacy_stage = "output" if "输出" in legacy_detail else "input"
        match = re.search(
            r"[：:]\s*([A-Z][A-Z0-9_,， -]*)\s*$",
            legacy_detail,
        )
        reason_codes = (
            [
                code
                for code in (
                    item.strip() for item in re.split(r"[,，]", match.group(1)) if item.strip()
                )
                if _SAFE_SECURITY_REASON_CODE.fullmatch(code)
            ][:20]
            if match
            else []
        )
        security_projection = {
            "stage": legacy_stage,
            "action": "block",
            "reason_codes": reason_codes,
            "retrieval_called": False,
        }
    else:
        # 历史单元调用可能仍传入自由文本；该文本只用于当前栈判断，绝不进入输出。
        safe_error = _stage_default_error("case_observation")

    observation = {
        "observation_schema": (
            "stt.eval.observation.v1" if mode == "retrieval_only" else OBSERVATION_SCHEMA_V3
        ),
        "answer": "",
        "retrieved": [],
        "reranked": [],
        "context": [],
        "latency_ms": None,
        "tokens": None,
        "cost": None,
        "versions": config_snapshot,
        "scope": "retrieval_only" if mode == "retrieval_only" else "full",
        "error": safe_error,
    }
    if mode == "retrieval_only":
        observation["visible_answer_policy"] = {}
    else:
        observation.update(project_proof_observation({}))
    if security_projection is not None:
        # 阻断投影只保留受控枚举；原始输入、Evidence 与异常消息均不进入观测。
        observation["security"] = security_projection
        observation["side_effects"] = {"external_llm": False}
    return observation


def _gate(
    run: dict,
    cases: list[dict],
    results: list[dict],
    summary: dict,
    comparison: dict,
) -> dict:
    retrieval_only = run.get("mode") == "retrieval_only"
    security_case_ids = {
        str(case.get("case_id") or "")
        for case in cases
        if isinstance((case.get("metadata") or {}).get("security_expectation"), Mapping)
    }
    safety_failed = sorted(
        str(result.get("case_id") or "")
        for result in results
        if str(result.get("case_id") or "") in security_case_ids
        and (
            ((result.get("metrics") or {}).get("metrics") or {}).get("safety", {}).get("passed")
            is not True
        )
    )
    critical_failed = sorted(
        result["case_id"]
        for result in results
        if not result.get("passed")
        and (
            not retrieval_only
            or bool(
                ((result.get("metrics") or {}).get("metrics") or {})
                .get("retrieval", {})
                .get("applicable", False)
            )
        )
        and any(
            case["case_id"] == result["case_id"]
            and bool((case.get("metadata") or {}).get("critical"))
            for case in cases
        )
    )
    policy = (run.get("config_snapshot") or {}).get("gate_policy") or {}
    policy_version = str(policy.get("policy_version") or LEGACY_EVALUATION_GATE_POLICY_VERSION)
    case_policy = policy.get("case_pass_policy") or {}
    case_policy_version = str(case_policy.get("policy_version") or LEGACY_CASE_PASS_POLICY_VERSION)
    if not run.get("baseline_run_id"):
        passed = not critical_failed and not safety_failed
        return {
            "evaluation_scope": "retrieval_only" if retrieval_only else "full",
            "status": ("baseline_recorded" if passed else "baseline_recorded_with_failures"),
            "passed": passed,
            "reason": (
                "首次运行已建立事实基线"
                if passed
                else "首次运行已建立事实基线，但关键 case 失败，不能视为质量门禁通过"
            ),
            "critical_failed_cases": critical_failed,
            "safety_failed_cases": safety_failed,
            "policy_version": policy_version,
            "case_pass_policy_version": case_policy_version,
            "policy": policy,
        }

    failures: list[dict] = []
    deltas = comparison.get("metric_deltas") or {}
    expected_case_ids = {str(case.get("case_id") or "") for case in cases if case.get("case_id")}
    matched_case_ids = {
        str(case_id) for case_id in (comparison.get("matched_case_ids") or []) if case_id
    }
    baseline_only_case_ids = sorted(
        str(case_id) for case_id in (comparison.get("baseline_only_case_ids") or []) if case_id
    )
    candidate_only_case_ids = sorted(
        str(case_id) for case_id in (comparison.get("candidate_only_case_ids") or []) if case_id
    )
    if (
        not matched_case_ids
        or matched_case_ids != expected_case_ids
        or baseline_only_case_ids
        or candidate_only_case_ids
        or int(comparison.get("matched_case_count") or 0) != len(expected_case_ids)
    ):
        failures.append(
            {
                "rule": "paired_case_set",
                "reason": "zero_overlap" if not matched_case_ids else "incomplete_pairing",
                "expected_case_count": len(expected_case_ids),
                "matched_case_count": len(matched_case_ids),
                "missing_expected_case_ids": sorted(expected_case_ids - matched_case_ids),
                "unexpected_matched_case_ids": sorted(matched_case_ids - expected_case_ids),
                "baseline_only_case_ids": baseline_only_case_ids,
                "candidate_only_case_ids": candidate_only_case_ids,
            }
        )

    def apply_max_drop(*, metric: str, threshold: str, rule: str) -> None:
        delta = deltas.get(metric)
        if not delta:
            return
        max_drop = float(policy.get(threshold, 0.02))
        if delta["delta"] < -max_drop:
            failures.append(
                {
                    "rule": rule,
                    "metric": metric,
                    "delta": delta["delta"],
                    "max_drop": max_drop,
                    "paired_count": int(delta.get("paired_count") or 0),
                    "unknown_pair_count": int(delta.get("unknown_pair_count") or 0),
                }
            )

    # Historical snapshots have no explicit version and retain their original
    # v1 contract. New v2 snapshots additionally guard early-rank quality.
    apply_max_drop(
        metric="metrics.retrieval.recall_at_k.10",
        threshold="recall_at_10_max_drop",
        rule="recall_at_10",
    )
    if policy_version == EVALUATION_GATE_POLICY_VERSION:
        for metric, threshold, rule in (
            (
                "metrics.retrieval.recall_at_k.1",
                "recall_at_1_max_drop",
                "recall_at_1",
            ),
            (
                "metrics.retrieval.recall_at_k.3",
                "recall_at_3_max_drop",
                "recall_at_3",
            ),
            ("metrics.retrieval.mrr", "mrr_max_drop", "mrr"),
            (
                "metrics.retrieval.ndcg_at_k.10",
                "ndcg_at_10_max_drop",
                "ndcg_at_10",
            ),
        ):
            apply_max_drop(metric=metric, threshold=threshold, rule=rule)
    elif policy_version != LEGACY_EVALUATION_GATE_POLICY_VERSION:
        failures.append(
            {
                "rule": "unsupported_gate_policy",
                "policy_version": policy_version,
            }
        )
    abstention_metric = (
        "metrics.abstention.contract_correct"
        if case_policy_version
        in {
            CASE_PASS_POLICY_VERSION,
            SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
            AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
            TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
        }
        else "metrics.abstention.correct"
    )
    abstention = deltas.get(abstention_metric)
    if abstention and abstention["delta"] < -float(
        policy.get("unanswerable_false_answer_max_increase", 0.0)
    ):
        failures.append({"rule": "unanswerable_accuracy", "delta": abstention["delta"]})
    baseline_latency = (
        (comparison.get("baseline") or {}).get("runtime", {}).get("latency_ms", {}).get("p95")
    )
    candidate_latency = (
        (comparison.get("candidate") or {}).get("runtime", {}).get("latency_ms", {}).get("p95")
    )
    if baseline_latency and candidate_latency is not None:
        increase = (candidate_latency - baseline_latency) / baseline_latency
        if increase > float(policy.get("p95_latency_max_increase_ratio", 0.2)):
            failures.append({"rule": "latency_increase", "ratio": increase})
    grounding = deltas.get("metrics.citation.coverage")
    if grounding and grounding["delta"] < -float(policy.get("grounding_coverage_max_drop", 0.0)):
        failures.append({"rule": "grounding_coverage", "delta": grounding["delta"]})
    if not retrieval_only and case_policy_version in {
        PREVIOUS_CASE_PASS_POLICY_VERSION,
        CASE_PASS_POLICY_VERSION,
        SOURCE_ALIGNED_CASE_PASS_POLICY_VERSION,
        AUTHORED_CLAIM_CASE_PASS_POLICY_VERSION,
        TEMPORAL_DERIVED_CASE_PASS_POLICY_VERSION,
    }:
        for metric, threshold, rule in (
            (
                "metrics.case_outcome.final_answer_accuracy",
                "answer_correctness_max_drop",
                "answer_correctness",
            ),
            (
                "metrics.case_outcome.quality_score",
                "case_quality_max_drop",
                "case_quality",
            ),
            (
                "metrics.generation.faithfulness",
                "faithfulness_max_drop",
                "faithfulness",
            ),
        ):
            apply_max_drop(metric=metric, threshold=threshold, rule=rule)
    if policy.get("critical_failure_blocks", True) and critical_failed:
        failures.append({"rule": "critical_cases", "case_ids": critical_failed})
    if safety_failed:
        failures.append({"rule": "safety_contracts", "case_ids": safety_failed})
    return {
        "evaluation_scope": "retrieval_only" if retrieval_only else "full",
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "failures": failures,
        "critical_failed_cases": critical_failed,
        "safety_failed_cases": safety_failed,
        "policy_version": policy_version,
        "case_pass_policy_version": case_policy_version,
        "policy": policy,
        "metric_coverage": {
            "tokens": (summary.get("runtime") or {}).get("total_tokens", {}).get("count", 0),
            "cost": (summary.get("runtime") or {}).get("cost", {}).get("count", 0),
        },
    }


def _write_artifacts(payload: dict) -> dict:
    run_id = payload["run_id"]
    artifacts = {
        "json": ("report.json", render_json_report(payload)),
        "markdown": ("report.md", render_markdown_report(payload)),
        "html": ("report.html", render_html_report(payload)),
        "summary": ("summary.json", render_json_report(payload["summary"])),
        "manifest": ("manifest.json", render_json_report(payload["run"])),
    }
    results_jsonl = (
        "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payload["results"]
        )
        + "\n"
    )
    artifacts["results"] = ("results.jsonl", results_jsonl)
    if payload.get("comparison"):
        artifacts["comparison"] = (
            "compare.json",
            render_json_report(payload["comparison"]),
        )
    return publish_text_artifacts(
        project_root=PROJECT_ROOT,
        report_root=REPORT_ROOT,
        run_id=run_id,
        artifacts=artifacts,
    )


def _final_report_run_snapshot(
    run: Mapping,
    *,
    summary: dict,
    comparison: dict,
    gate: dict,
    completed_at: datetime,
) -> dict:
    """构造与数据库终态一致、且不依赖自引用产物哈希的运行快照。"""

    if completed_at.tzinfo is None:
        raise ValueError("evaluation completion timestamp must include timezone")
    completed_at_iso = completed_at.isoformat()
    return {
        **run,
        "status": "succeeded",
        "progress": 1.0,
        "completed_cases": run["total_cases"],
        "metrics_summary": summary,
        "comparison_summary": comparison,
        "gate_result": gate,
        "completed_at": completed_at_iso,
        "updated_at": completed_at_iso,
    }


def _heartbeat_evaluation_run(
    run_id: str,
    owner_id: str,
    stop: threading.Event,
    ownership_lost: threading.Event,
    *,
    interval_seconds: float,
) -> None:
    """独立心跳避免单个慢 case 被恢复器误判为已崩溃。"""

    while not stop.wait(interval_seconds):
        try:
            with session_scope() as session:
                alive = EvaluationRepository(session).heartbeat_run(
                    run_id,
                    owner_id=owner_id,
                )
            if not alive:
                ownership_lost.set()
                return
        except Exception as exc:
            # 短暂 SQLite/控制库争用不能立即判定 owner 丢失；超时恢复器仍以最后一次
            # 成功 heartbeat/updated_at 为准，下一周期继续尝试。
            safe_error = _safe_evaluation_error(exc, stage="run_heartbeat")
            _log_safe_error("evaluation heartbeat failed", safe_error, run_id=run_id)


def execute_run(
    run_id: str,
    principal: Principal,
    *,
    adapter: EvalObservationAdapter | None = None,
    owner_id: str | None = None,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> None:
    """Execute a durable run. One failed case is recorded and does not abort the set."""

    if heartbeat_seconds <= 0:
        raise ValueError("evaluation heartbeat interval must be positive")
    execution_owner = owner_id or evaluation_execution_owner()
    heartbeat_stop = threading.Event()
    ownership_lost = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    published_paths: dict[str, dict[str, object]] = {}
    proof_authority = None
    target_result = None
    token = set_current_principal(principal)
    try:
        with session_scope() as session:
            repo = EvaluationRepository(session)
            run = repo.start_run(run_id, owner_id=execution_owner)
            cases = repo.selected_cases_for_run(run_id)
        heartbeat_thread = threading.Thread(
            target=_heartbeat_evaluation_run,
            args=(run_id, execution_owner, heartbeat_stop, ownership_lost),
            kwargs={"interval_seconds": heartbeat_seconds},
            name=f"evaluation-heartbeat-{run_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        observer = _adapter_for_mode(run["mode"], adapter)
        for case_index, case in enumerate(cases):
            if ownership_lost.is_set():
                raise EvaluationRunOwnershipLost(
                    "evaluation run ownership was lost before the next case"
                )
            error_type = None
            error_detail = None
            proof_authority = None
            target_result = None
            try:
                correlation_id = hashlib.sha256(
                    f"{run_id}|{case_index}".encode("utf-8")
                ).hexdigest()[:32]
                target_input = TargetInput.from_case(
                    case,
                    correlation_id=correlation_id,
                )
                evaluate = getattr(observer, "evaluate", None)
                if callable(evaluate):
                    target_result = evaluate(
                        EvaluationRequest(
                            target_input=target_input,
                            kb_id=run["kb_id"],
                            run_id=run_id,
                            config=run["config_snapshot"],
                        )
                    )
                    if isinstance(target_result, EvaluationTargetResult):
                        observation = dict(target_result.observation)
                        proof_authority = target_result.proof_authority
                    else:
                        observation = dict(target_result)
                    # 避免结果 wrapper 在后续 repository 调用期间继续持有 sidecar；
                    # raw authority 的唯一剩余引用由下方显式生命周期变量管理。
                    target_result = None
                else:
                    # Legacy observe 也只能在 service 边界取得 allowlist 投影；
                    # 完整 case 始终留在下方 grader 路径。
                    observation = observer.observe(
                        target_input.as_mapping(),
                        kb_id=run["kb_id"],
                        run_id=run_id,
                        config=run["config_snapshot"],
                    )
            except Exception as exc:  # one bad case must not erase the run
                # 观测投影失败时 wrapper 仍可能持有 raw sidecar；日志或后续
                # 失败观测生成前先断开全部引用。
                target_result = None
                proof_authority = None
                safe_error = _safe_evaluation_error(exc, stage="case_observation")
                _log_safe_error(
                    "evaluation case failed",
                    safe_error,
                    run_id=run_id,
                    case_id=str(case["case_id"]),
                )
                error_type = safe_error["type"]
                error_detail = _safe_error_json(safe_error)
                observation = _failed_case_observation(
                    mode=run["mode"],
                    config_snapshot=run["config_snapshot"],
                    exception=exc,
                )
            if ownership_lost.is_set():
                raise EvaluationRunOwnershipLost(
                    "evaluation run was interrupted while a case was executing"
                )
            gate_policy = (run.get("config_snapshot") or {}).get("gate_policy") or {}
            frozen_case_policy = gate_policy.get("case_pass_policy")
            if not isinstance(frozen_case_policy, dict):
                # A pre-v2 case snapshot must keep the exact legacy decision
                # contract when an interrupted historical run is resumed.
                frozen_case_policy = {"policy_version": LEGACY_CASE_PASS_POLICY_VERSION}
            try:
                evaluated = evaluate_case(
                    case,
                    observation,
                    case_policy=frozen_case_policy,
                    proof_authority=proof_authority,
                )
                case_outcome = _case_decision(
                    case,
                    evaluated,
                    had_error=error_type is not None,
                    mode=run["mode"],
                    case_policy=frozen_case_policy,
                    observation=observation,
                )
            except Exception as exc:
                # 丢弃下游 traceback 中持有 authority 参数的 frame；随后 finally
                # 先释放 sidecar，再把同一异常交给 run 级安全错误处理。
                raise exc.with_traceback(None) from None
            finally:
                # raw canonical/packed sidecar 在任何 repository/report/log 调用前
                # 都必须不可达，包括 evaluator 与 decision 的异常路径。
                proof_authority = None
                target_result = None
            evaluated.setdefault("metrics", {})["case_outcome"] = case_outcome
            passed = bool(case_outcome["passed"])
            with session_scope() as session:
                EvaluationRepository(session).record_result(
                    run_id,
                    case["case_id"],
                    observation=observation,
                    metrics=evaluated,
                    passed=passed,
                    error_type=error_type,
                    error_detail=error_detail,
                    owner_id=execution_owner,
                )

        with session_scope() as session:
            repo = EvaluationRepository(session)
            run = repo.get_run(run_id)
            persisted = repo.list_results(run_id)
        evaluated_results = [item["metrics"] for item in persisted]
        summary = aggregate_results(evaluated_results)
        comparison: dict = {}
        if run.get("baseline_run_id"):
            with session_scope() as session:
                baseline = EvaluationRepository(session).list_results(run["baseline_run_id"])
            comparison = paired_diff([item["metrics"] for item in baseline], evaluated_results)
        gate = _gate(run, cases, persisted, summary, comparison)
        completed_at = datetime.now(UTC)
        report_run = _final_report_run_snapshot(
            run,
            summary=summary,
            comparison=comparison,
            gate=gate,
            completed_at=completed_at,
        )
        payload = {
            "report_schema": "stt.eval.report.v1",
            "evaluation_scope": ("retrieval_only" if run["mode"] == "retrieval_only" else "full"),
            "run_id": run_id,
            "dataset_id": run["dataset_id"],
            "dataset_version": run["dataset_version"],
            "baseline_run_id": run.get("baseline_run_id"),
            "run": report_run,
            "summary": summary,
            "comparison": comparison,
            "gate": gate,
            "results": persisted,
        }
        with session_scope() as session:
            if not EvaluationRepository(session).heartbeat_run(
                run_id,
                owner_id=execution_owner,
            ):
                raise EvaluationRunOwnershipLost(
                    "evaluation run ownership was lost before report publication"
                )
        paths = _write_artifacts(payload)
        published_paths = paths
        with session_scope() as session:
            EvaluationRepository(session).complete_run(
                run_id,
                metrics_summary=summary,
                comparison_summary=comparison,
                gate_result=gate,
                report_paths=paths,
                owner_id=execution_owner,
                completed_at=completed_at,
            )
    except Exception as exc:
        # 覆盖 ownership-loss 等跳过 case-local finally 的路径；失败日志、文件
        # 回滚和 fail_run 持久化均只能在 sidecar 释放后发生。
        proof_authority = None
        target_result = None
        if published_paths:
            try:
                discard_artifact_publication(
                    project_root=PROJECT_ROOT,
                    report_root=REPORT_ROOT,
                    run_id=run_id,
                    paths=published_paths,
                )
            except Exception:
                # 回滚失败也不得把文件系统异常正文写入日志或耐久错误字段。
                logger.warning("evaluation artifact rollback failed run_id=%s", run_id)
        safe_error = _safe_evaluation_error(exc, stage="run_execution")
        _log_safe_error("evaluation run failed", safe_error, run_id=run_id)
        try:
            with session_scope() as session:
                EvaluationRepository(session).fail_run(
                    run_id,
                    _safe_error_json(safe_error),
                    error_code=safe_error["code"],
                    owner_id=execution_owner,
                )
        except Exception as persist_exc:
            persistence_error = _safe_evaluation_error(
                persist_exc,
                stage="failure_persistence",
            )
            _log_safe_error(
                "failed to persist evaluation run failure",
                persistence_error,
                run_id=run_id,
            )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=max(1.0, heartbeat_seconds + 1.0))
        reset_current_principal(token)


def report_path(run_id: str, artifact: str) -> Path:
    run = get_run(run_id)
    record = (run.get("report_paths") or {}).get(artifact)
    if not record:
        raise ResourceConflictError(f"evaluation artifact is not available: {artifact}")
    path = (PROJECT_ROOT / record["path"]).resolve()
    allowed_roots = {
        (REPORT_ROOT / run_id).resolve(),
        (LEGACY_REPORT_ROOT / run_id).resolve(),
    }
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ResourceConflictError("invalid evaluation artifact path")
    if not path.is_file():
        raise ResourceConflictError("evaluation artifact file is missing")
    expected_digest = str(record.get("sha256") or "")
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_digest or actual_digest != expected_digest:
        raise ResourceConflictError("evaluation artifact integrity check failed")
    return path


def _adapter_for_mode(
    mode: str,
    explicit: EvalObservationAdapter | None = None,
) -> EvalObservationAdapter:
    """Select the only adapter allowed for a persisted evaluation scope."""

    if explicit is not None:
        return explicit
    if mode == "retrieval_only":
        return RetrievalOnlyEvalAdapter()
    if mode == "live":
        return QueryGraphEvalAdapter()
    raise ResourceConflictError(f"unsupported persisted evaluation mode: {mode}")


# Feedback and annotation application facades ---------------------------------
def create_feedback(payload: dict) -> dict:
    principal = current_principal()
    session_id = payload.get("session_id")
    if principal is not None and session_id:
        normalized_session_id = str(session_id)
        if normalized_session_id.startswith("conv_"):
            # 可靠查询入口以 SQL Conversation 为权威身份；旧 /query 仍保留
            # QuerySession。反馈必须同时支持二者，且都复核 owner + KB。
            with session_scope() as session:
                conversation = QueryRuntimeRepository(
                    session,
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                ).get_conversation(normalized_session_id)
            session_kb_id = conversation.kb_id
        else:
            query_session = assert_query_session_owner(principal, normalized_session_id)
            session_kb_id = query_session.kb_id
        if session_kb_id != str(payload["kb_id"]):
            raise ResourceConflictError("feedback session and knowledge base do not match")
    with session_scope() as session:
        # Resolve KB in the current principal scope before persisting the fact.
        ControlRepository(session).get_knowledge_base_model(str(payload["kb_id"]))
        return EvaluationRepository(session).create_feedback(payload)


def list_feedback(*, status: str | None, mine: bool) -> list[dict]:
    with session_scope() as session:
        return EvaluationRepository(session).list_feedback(status=status, mine=mine)


def triage_feedback(feedback_id: str, *, status: str, review_note: str) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).triage_feedback(
            feedback_id, status=status, review_note=review_note
        )


def create_annotation(feedback_id: str, *, target_dataset_key: str, proposed_case: dict) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).create_annotation_task(
            feedback_id,
            target_dataset_key=target_dataset_key,
            proposed_case=proposed_case,
        )


def list_annotations() -> list[dict]:
    with session_scope() as session:
        return EvaluationRepository(session).list_annotation_tasks()


def review_annotation(task_id: str, *, decision: str, note: str) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).review_annotation(
            task_id, decision=decision, note=note
        )


def arbitrate_annotation(
    task_id: str, *, decision: str, note: str, resolved_case: dict | None
) -> dict:
    with session_scope() as session:
        return EvaluationRepository(session).arbitrate_annotation(
            task_id,
            decision=decision,
            note=note,
            resolved_case=resolved_case,
        )
