"""从冻结 harness 配置创建并执行耐久评测运行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

# 直接以文件路径执行时，Python 只把 scripts/ 放入 sys.path。显式加入仓库根目录，
# 保证与 ``python -m scripts.run_formal_evaluation`` 使用相同的包解析边界。
_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PROJECT_ROOT))

from evals.configs import load_harness_config  # noqa: E402
from evals.datasets import DatasetAsset, load_dataset_asset  # noqa: E402
from evals.scenarios import load_scenario  # noqa: E402
from app.domain.evaluation import (  # noqa: E402
    FORMAL_EVALUATION_SNAPSHOT_KEY,
    build_formal_evaluation_intent,
    content_digest,
    freeze_formal_evaluation_contract,
)


SUPPORTED_TARGET = "query-graph-evaluation-target"
SUPPORTED_GRADERS = {"retrieval", "generation", "abstention", "safety"}
_GRADERS_BY_MODE = {
    "live": SUPPORTED_GRADERS,
    "retrieval_only": {"retrieval"},
}
_MODE_BY_OBSERVATION_SCOPE = {
    "full": "live",
    "retrieval_only": "retrieval_only",
}
_SIDE_EFFECTS_BY_MODE = {
    "live": {"vector_read", "external_llm"},
    "retrieval_only": {"vector_read"},
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预检或执行由 registry 与内容哈希冻结的正式评测",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--tenant-id", default="tenant_default")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--mode", choices=("live", "retrieval_only"), default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--profile-version", type=int, default=None)
    parser.add_argument("--baseline-run-id", default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="通过后创建 SQL run 并执行；省略时保持只读",
    )
    parser.add_argument(
        "--allow-paid-model-calls",
        action="store_true",
        help="live 模式的独立费用确认；retrieval_only 不需要",
    )
    return parser.parse_args(argv)


def _scenario_snapshot(scenario: Any) -> dict[str, Any]:
    snapshot = {
        "scenario_id": str(scenario.scenario_id),
        "observation_scope": str(scenario.observation_scope),
        "allowed_side_effects": list(scenario.allowed_side_effects),
        "external_model_allowed": bool(scenario.external_model_allowed),
        "formal_evaluation": bool(scenario.formal_evaluation),
    }
    return {
        **snapshot,
        "scenario_sha256": content_digest(snapshot),
    }


def _validate_selection(args: argparse.Namespace, config: Any, scenario: Any) -> str:
    if not config.formal_evaluation:
        raise ValueError("harness config is not approved for formal evaluation")
    if (
        config.schema_version != "stt.eval.harness-config.v3"
        or config.planning_admission_budgets is None
        or config.usage_accounting is None
    ):
        raise ValueError("formal execution requires harness config v3 authority fields")
    if config.target_id != SUPPORTED_TARGET:
        raise ValueError(f"unsupported formal evaluation target: {config.target_id}")
    unknown_graders = set(config.graders) - SUPPORTED_GRADERS
    if unknown_graders:
        raise ValueError(
            "unsupported formal evaluation graders: " + ", ".join(sorted(unknown_graders))
        )
    if not scenario.formal_evaluation:
        raise ValueError("scenario is not approved for formal evaluation")
    mode = _MODE_BY_OBSERVATION_SCOPE.get(scenario.observation_scope)
    if mode is None:
        raise ValueError(
            f"unsupported formal evaluation observation scope: {scenario.observation_scope}"
        )
    if args.mode is not None and args.mode != mode:
        raise ValueError(f"CLI mode {args.mode} conflicts with scenario-derived mode {mode}")
    if set(scenario.allowed_side_effects) != _SIDE_EFFECTS_BY_MODE[mode]:
        raise ValueError("scenario side effects do not match the supported evaluation target")
    if bool(scenario.external_model_allowed) is not (mode == "live"):
        raise ValueError("scenario model permission does not match its observation scope")
    expected_graders = _GRADERS_BY_MODE[mode]
    if set(config.graders) != expected_graders or len(config.graders) != len(expected_graders):
        raise ValueError("formal graders do not match the scenario-derived execution mode")
    # 当前 durable executor 对每个失败 case 留痕后继续；拒绝接受无法实际执行的
    # fail_fast 声明，避免配置与运行事实分叉。
    if config.failure_policy != "continue":
        raise ValueError("query graph formal evaluation only supports continue failure policy")
    if (args.profile_id is None) != (args.profile_version is None):
        raise ValueError("profile id and version must be provided together")
    if args.allow_paid_model_calls and not scenario.external_model_allowed:
        raise ValueError("scenario does not allow paid external model calls")
    if args.execute and scenario.external_model_allowed and not args.allow_paid_model_calls:
        raise ValueError("live execution requires --allow-paid-model-calls")
    return mode


def _load_evaluation_service() -> ModuleType:
    from app.application import evaluation_service

    return evaluation_service


def _preflight_summary(
    args: argparse.Namespace,
    config: Any,
    asset: DatasetAsset,
    scenario: Any,
    *,
    mode: str,
) -> dict[str, Any]:
    scenario_snapshot = _scenario_snapshot(scenario)
    return {
        "status": "validated" if not args.execute else "execution_requested",
        "config_id": config.config_id,
        "config_sha256": config.canonical_sha256,
        "target_id": config.target_id,
        "scenario": scenario_snapshot,
        "dataset_id": asset.dataset_id,
        "dataset_sha256": asset.sha256,
        "mode": mode,
        "graders": list(config.graders),
        "failure_policy": config.failure_policy,
        "planning_admission_budgets": config.planning_admission_budgets.as_dict(),
        "usage_accounting": config.usage_accounting.as_dict(),
        "allowed_intervention_dimensions": list(config.allowed_intervention_dimensions),
        "profile": (
            {"profile_id": args.profile_id, "version": args.profile_version}
            if args.profile_id is not None
            else {"source": "active"}
        ),
        "requested_case_count": len(dict.fromkeys(args.case_id or [])) or None,
        "formal_evaluation": config.formal_evaluation,
        "will_write": bool(args.execute),
        "validation_scope": "config_scenario_and_dataset_asset_only",
        "paid_model_calls_allowed": bool(
            scenario.external_model_allowed and args.allow_paid_model_calls
        ),
    }


def _execute(
    args: argparse.Namespace,
    config: Any,
    asset: DatasetAsset,
    scenario: Any,
    *,
    mode: str,
) -> dict[str, Any]:
    from app.security.principal import Principal, reset_current_principal, set_current_principal

    service = _load_evaluation_service()
    principal = Principal(
        user_id=args.user_id,
        tenant_id=args.tenant_id,
        roles=("TenantAdmin",),
        scopes=(),
        auth_type="local-evaluation",
        credential_id="formal-evaluation-cli",
    )
    token = set_current_principal(principal)
    try:
        dataset = service.import_dataset(asset.manifest)
        formal_intent = build_formal_evaluation_intent(
            config_schema_version=config.schema_version,
            config_id=config.config_id,
            config_sha256=config.canonical_sha256,
            target_id=config.target_id,
            dataset={
                "registry_id": asset.dataset_id,
                "asset_sha256": asset.sha256,
                "record_id": dataset.get("id"),
                "record_checksum": dataset.get("checksum"),
                "dataset_key": dataset.get("dataset_key"),
                "dataset_version": dataset.get("version"),
            },
            scenario=_scenario_snapshot(scenario),
            graders=list(config.graders),
            failure_policy=config.failure_policy,
            execution_mode=mode,
            planning_admission_budgets=config.planning_admission_budgets.as_dict(),
            usage_accounting=config.usage_accounting.as_dict(),
            allowed_intervention_dimensions=config.allowed_intervention_dimensions,
        )
        profile_pin = (
            {
                "retrieval_profile_id": args.profile_id,
                "retrieval_profile_version": args.profile_version,
            }
            if args.profile_id is not None
            else {}
        )
        run = service.create_run(
            dataset_id=dataset["id"],
            kb_id=args.kb_id,
            mode=mode,
            name=args.name or f"{config.config_id} formal evaluation",
            baseline_run_id=args.baseline_run_id,
            case_ids=args.case_id,
            formal_evaluation_contract=formal_intent,
            **profile_pin,
        )
        persisted_contract = freeze_formal_evaluation_contract(
            (run.get("config_snapshot") or {}).get(FORMAL_EVALUATION_SNAPSHOT_KEY),
            expected_mode=mode,
            expected_dataset_record_id=str(dataset.get("id") or ""),
            expected_dataset_checksum=str(dataset.get("checksum") or ""),
            expected_dataset_key=str(dataset.get("dataset_key") or ""),
            expected_dataset_version=str(dataset.get("version") or ""),
            expected_run_id=str(run.get("id") or ""),
            expected_tenant_id=args.tenant_id,
            expected_kb_id=args.kb_id,
            expected_total_cases=int(run.get("total_cases") or 0),
            expected_runtime_snapshot=run.get("config_snapshot") or {},
        )
        if persisted_contract.get("intent") != formal_intent:
            raise RuntimeError("formal evaluation contract was not durably bound to the run")
        selected_case_ids = persisted_contract["run_authority"]["selected_cases"][
            "selected_case_ids"
        ]
        if args.case_id and set(selected_case_ids) != set(dict.fromkeys(args.case_id)):
            raise RuntimeError("durable formal case selection does not match the request")
        service.execute_run(run["id"], principal)
        completed = service.get_run(run["id"])
        completed_contract = freeze_formal_evaluation_contract(
            (completed.get("config_snapshot") or {}).get(FORMAL_EVALUATION_SNAPSHOT_KEY),
            expected_mode=str(completed.get("mode") or mode),
            expected_dataset_record_id=str(completed.get("dataset_id") or ""),
            expected_dataset_checksum=str(completed.get("dataset_checksum") or ""),
            expected_dataset_key=str(completed.get("dataset_key") or ""),
            expected_dataset_version=str(completed.get("dataset_version") or ""),
            expected_run_id=str(completed.get("id") or ""),
            expected_tenant_id=args.tenant_id,
            expected_kb_id=str(completed.get("kb_id") or ""),
            expected_total_cases=int(completed.get("total_cases") or 0),
            expected_runtime_snapshot=completed.get("config_snapshot") or {},
        )
        if completed_contract != persisted_contract:
            raise RuntimeError("formal evaluation contract changed during execution")
        durable_intent = completed_contract["intent"]
        authority = completed_contract["run_authority"]
        selected = authority["selected_cases"]
        durable_preflight = dict((completed.get("config_snapshot") or {}).get("preflight") or {})
        metrics = dict(completed.get("metrics_summary") or {})
        gate = dict(completed.get("gate_result") or {})
        return {
            "status": completed.get("status"),
            "run_id": completed.get("id"),
            "formal_contract_sha256": completed_contract["contract_sha256"],
            "formal_intent_sha256": durable_intent["intent_sha256"],
            "config_id": durable_intent["config_id"],
            "config_sha256": durable_intent["config_sha256"],
            "dataset_id": completed.get("dataset_id"),
            "dataset_key": completed.get("dataset_key"),
            "dataset_version": completed.get("dataset_version"),
            "dataset_asset_sha256": durable_intent["dataset"]["asset_sha256"],
            "mode": durable_intent["execution_mode"],
            "kb_id": authority["kb_id"],
            "baseline": authority["baseline"],
            "selected_cases": selected,
            "profile": authority["retrieval_profile"],
            "vector_read_set": {
                "digest": authority["vector_read_set"]["digest"],
                "scope_count": len(authority["vector_read_set"]["scopes"]),
                "scopes_sha256": authority["vector_read_set"]["scopes_sha256"],
            },
            "runtime_snapshot_sha256": authority["runtime_snapshot_sha256"],
            "runtime_flags": authority["runtime_flags"],
            "enforced_execution_controls": authority["enforced_execution_controls"],
            "preflight": {
                "ready": durable_preflight.get("ready"),
                "checks": durable_preflight.get("checks") or {},
                "sha256": authority["preflight_sha256"],
            },
            "planning_admission_budgets": durable_intent["planning_admission_budgets"],
            "usage_accounting": durable_intent["usage_accounting"],
            "allowed_intervention_dimensions": durable_intent["allowed_intervention_dimensions"],
            "completed_cases": metrics.get("case_count"),
            "gate_status": gate.get("status"),
            "gate_passed": gate.get("passed"),
            "artifact_directory": f"var/evals/{completed.get('id')}",
        }
    finally:
        reset_current_principal(token)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_harness_config(args.config)
    scenario = load_scenario(config.scenario_id)
    mode = _validate_selection(args, config, scenario)
    asset = load_dataset_asset(
        config.dataset_id,
        project_root=args.project_root,
    )
    result = (
        _execute(args, config, asset, scenario, mode=mode)
        if args.execute
        else _preflight_summary(args, config, asset, scenario, mode=mode)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"validated", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
