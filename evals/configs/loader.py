"""加载版本化 harness 配置，不负责组装生产依赖。"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("offline_contract.json")


@dataclass(frozen=True, slots=True)
class HarnessBudgets:
    """旧版 v2 查询规划字段；只用于读取历史配置。"""

    scope: str
    token_budget: int
    tool_budget: int
    cost_budget: float
    cost_currency: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "token_budget": self.token_budget,
            "tool_budget": self.tool_budget,
            "cost_budget": self.cost_budget,
            "cost_currency": self.cost_currency,
        }


@dataclass(frozen=True, slots=True)
class PlanningAdmissionBudgets:
    """逐用例 QueryPlan 准入上限，不代表实际供应商用量。"""

    scope: str
    max_plan_tokens: int
    max_routes: int
    max_planned_cost_units: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "max_plan_tokens": self.max_plan_tokens,
            "max_routes": self.max_routes,
            "max_planned_cost_units": self.max_planned_cost_units,
        }


@dataclass(frozen=True, slots=True)
class UsageAccounting:
    """正式运行当前可提供的实际用量记账与硬限制能力。"""

    provider_tokens: str
    provider_cost_usd: str
    retrieval_calls: str
    enforcement: str

    def as_dict(self) -> dict[str, str]:
        return {
            "provider_tokens": self.provider_tokens,
            "provider_cost_usd": self.provider_cost_usd,
            "retrieval_calls": self.retrieval_calls,
            "enforcement": self.enforcement,
        }


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """可审计、可复现的评测执行选择。"""

    schema_version: str
    config_id: str
    target_id: str
    dataset_id: str
    scenario_id: str
    graders: tuple[str, ...]
    failure_policy: str
    formal_evaluation: bool
    budgets: HarnessBudgets | None
    planning_admission_budgets: PlanningAdmissionBudgets | None
    usage_accounting: UsageAccounting | None
    allowed_intervention_dimensions: tuple[str, ...]
    canonical_sha256: str


_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "config_id",
        "target_id",
        "dataset_id",
        "scenario_id",
        "graders",
        "failure_policy",
        "formal_evaluation",
    }
)
_SUPPORTED_SCHEMAS = frozenset(
    {
        "stt.eval.harness-config.v1",
        "stt.eval.harness-config.v2",
        "stt.eval.harness-config.v3",
    }
)
_INTERVENTION_DIMENSIONS = frozenset(
    {
        "code_revision",
        "config_version",
        "retrieval_profile",
        "vector_read_set",
        "evaluation_security",
        "model_ids",
        "runtime_flags",
    }
)


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"evaluation harness config has duplicate field: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"evaluation harness config contains non-finite number: {value}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identifier(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("evaluation harness config has missing identifiers")
    return value


def _load_budgets(value: Any) -> HarnessBudgets:
    if not isinstance(value, dict):
        raise ValueError("evaluation harness config v2 must define budgets")
    expected = {
        "scope",
        "token_budget",
        "tool_budget",
        "cost_budget",
        "cost_currency",
    }
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ValueError("invalid evaluation harness budgets: " + "; ".join(detail))
    if value.get("scope") != "per_case_query_plan":
        raise ValueError("evaluation budget scope must be per_case_query_plan")
    token_budget = value.get("token_budget")
    tool_budget = value.get("tool_budget")
    cost_budget = value.get("cost_budget")
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or not 128 <= token_budget <= 131_072
    ):
        raise ValueError("evaluation token_budget must be an integer within 128..131072")
    if (
        isinstance(tool_budget, bool)
        or not isinstance(tool_budget, int)
        or not 1 <= tool_budget <= 8
    ):
        raise ValueError("evaluation tool_budget must be an integer within 1..8")
    if isinstance(cost_budget, bool) or not isinstance(cost_budget, (int, float)):
        raise ValueError("evaluation cost_budget must be a finite number")
    normalized_cost = float(cost_budget)
    if not math.isfinite(normalized_cost) or not 0.05 <= normalized_cost <= 1_000.0:
        raise ValueError("evaluation cost_budget must be within 0.05..1000")
    if value.get("cost_currency") != "USD":
        raise ValueError("evaluation cost_currency must be USD")
    return HarnessBudgets(
        scope="per_case_query_plan",
        token_budget=token_budget,
        tool_budget=tool_budget,
        cost_budget=normalized_cost,
        cost_currency="USD",
    )


def _load_planning_admission_budgets(value: Any) -> PlanningAdmissionBudgets:
    if not isinstance(value, dict):
        raise ValueError("evaluation harness config v3 must define planning_admission_budgets")
    expected = {
        "scope",
        "max_plan_tokens",
        "max_routes",
        "max_planned_cost_units",
    }
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ValueError(
            "invalid evaluation harness planning admission budgets: " + "; ".join(detail)
        )
    if value.get("scope") != "per_case_query_plan_admission":
        raise ValueError("evaluation planning budget scope must be per_case_query_plan_admission")
    max_plan_tokens = value.get("max_plan_tokens")
    max_routes = value.get("max_routes")
    max_planned_cost_units = value.get("max_planned_cost_units")
    if (
        isinstance(max_plan_tokens, bool)
        or not isinstance(max_plan_tokens, int)
        or not 128 <= max_plan_tokens <= 131_072
    ):
        raise ValueError("evaluation max_plan_tokens must be an integer within 128..131072")
    if isinstance(max_routes, bool) or not isinstance(max_routes, int) or not 1 <= max_routes <= 8:
        raise ValueError("evaluation max_routes must be an integer within 1..8")
    if isinstance(max_planned_cost_units, bool) or not isinstance(
        max_planned_cost_units, (int, float)
    ):
        raise ValueError("evaluation max_planned_cost_units must be a finite number")
    normalized_cost = float(max_planned_cost_units)
    if not math.isfinite(normalized_cost) or not 0.05 <= normalized_cost <= 1_000.0:
        raise ValueError("evaluation max_planned_cost_units must be within 0.05..1000")
    return PlanningAdmissionBudgets(
        scope="per_case_query_plan_admission",
        max_plan_tokens=max_plan_tokens,
        max_routes=max_routes,
        max_planned_cost_units=normalized_cost,
    )


def _load_usage_accounting(value: Any) -> UsageAccounting:
    if not isinstance(value, dict):
        raise ValueError("evaluation harness config v3 must define usage_accounting")
    expected = {
        "provider_tokens",
        "provider_cost_usd",
        "retrieval_calls",
        "enforcement",
    }
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ValueError("invalid evaluation usage accounting: " + "; ".join(detail))
    normalized = UsageAccounting(
        provider_tokens=str(value.get("provider_tokens") or ""),
        provider_cost_usd=str(value.get("provider_cost_usd") or ""),
        retrieval_calls=str(value.get("retrieval_calls") or ""),
        enforcement=str(value.get("enforcement") or ""),
    )
    if normalized.as_dict() != {
        "provider_tokens": "unavailable",
        "provider_cost_usd": "unavailable",
        "retrieval_calls": "unavailable",
        "enforcement": "not_enforced",
    }:
        raise ValueError(
            "evaluation usage accounting must explicitly declare unavailable/not_enforced"
        )
    return normalized


def load_harness_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> HarnessConfig:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_fields,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("evaluation harness config must be an object")
    schema_version = payload.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMAS:
        raise ValueError("unsupported evaluation harness config schema")
    expected_fields = set(_BASE_FIELDS)
    if schema_version == "stt.eval.harness-config.v2":
        expected_fields.add("budgets")
    elif schema_version == "stt.eval.harness-config.v3":
        expected_fields.update(
            {
                "planning_admission_budgets",
                "usage_accounting",
                "allowed_intervention_dimensions",
            }
        )
    unknown_fields = sorted(set(payload) - expected_fields)
    missing_fields = sorted(expected_fields - set(payload))
    if unknown_fields:
        raise ValueError(
            "evaluation harness config has unknown fields: " + ", ".join(unknown_fields)
        )
    if missing_fields:
        raise ValueError(
            "evaluation harness config has missing fields: " + ", ".join(missing_fields)
        )
    identifiers = {
        field: _identifier(payload, field)
        for field in ("config_id", "target_id", "dataset_id", "scenario_id")
    }
    graders = payload.get("graders")
    if (
        not isinstance(graders, list)
        or not graders
        or not all(isinstance(item, str) and item for item in graders)
    ):
        raise ValueError("evaluation harness config must select graders")
    if len(graders) != len(set(graders)):
        raise ValueError("evaluation harness config graders must be unique")
    failure_policy = str(payload.get("failure_policy") or "")
    if failure_policy not in {"fail_fast", "continue"}:
        raise ValueError("unsupported evaluation failure policy")
    formal = payload.get("formal_evaluation")
    if not isinstance(formal, bool):
        raise ValueError("formal_evaluation must be a boolean")
    budgets = (
        _load_budgets(payload.get("budgets"))
        if schema_version == "stt.eval.harness-config.v2"
        else None
    )
    planning_admission_budgets = (
        _load_planning_admission_budgets(payload.get("planning_admission_budgets"))
        if schema_version == "stt.eval.harness-config.v3"
        else None
    )
    usage_accounting = (
        _load_usage_accounting(payload.get("usage_accounting"))
        if schema_version == "stt.eval.harness-config.v3"
        else None
    )
    raw_interventions = (
        payload.get("allowed_intervention_dimensions")
        if schema_version == "stt.eval.harness-config.v3"
        else []
    )
    if (
        not isinstance(raw_interventions, list)
        or not all(isinstance(item, str) and item for item in raw_interventions)
        or len(raw_interventions) != len(set(raw_interventions))
    ):
        raise ValueError("allowed_intervention_dimensions must be unique strings")
    unknown_interventions = sorted(set(raw_interventions) - _INTERVENTION_DIMENSIONS)
    if unknown_interventions:
        raise ValueError("unsupported intervention dimensions: " + ", ".join(unknown_interventions))
    allowed_interventions = tuple(sorted(raw_interventions))
    normalized_payload: dict[str, Any] = {
        "schema_version": schema_version,
        **identifiers,
        "graders": list(graders),
        "failure_policy": failure_policy,
        "formal_evaluation": formal,
    }
    if budgets is not None:
        normalized_payload["budgets"] = budgets.as_dict()
    if planning_admission_budgets is not None:
        normalized_payload["planning_admission_budgets"] = planning_admission_budgets.as_dict()
    if usage_accounting is not None:
        normalized_payload["usage_accounting"] = usage_accounting.as_dict()
        normalized_payload["allowed_intervention_dimensions"] = list(allowed_interventions)
    canonical_sha256 = hashlib.sha256(
        _canonical_json(normalized_payload).encode("utf-8")
    ).hexdigest()
    return HarnessConfig(
        schema_version=schema_version,
        config_id=identifiers["config_id"],
        target_id=identifiers["target_id"],
        dataset_id=identifiers["dataset_id"],
        scenario_id=identifiers["scenario_id"],
        graders=tuple(graders),
        failure_policy=failure_policy,
        formal_evaluation=formal,
        budgets=budgets,
        planning_admission_budgets=planning_admission_budgets,
        usage_accounting=usage_accounting,
        allowed_intervention_dimensions=allowed_interventions,
        canonical_sha256=canonical_sha256,
    )
