"""读取版本化评测场景，避免执行模式由调用方自由解释。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("catalog.json")


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """一个场景的能力与副作用约束。"""

    scenario_id: str
    observation_scope: str
    allowed_side_effects: tuple[str, ...]
    external_model_allowed: bool
    formal_evaluation: bool


def load_scenario_catalog(path: str | Path = CATALOG_PATH) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation scenario catalog must be an object")
    if payload.get("schema_version") != "stt.eval.scenario-catalog.v1":
        raise ValueError("unsupported evaluation scenario catalog schema")
    if not isinstance(payload.get("scenarios"), dict):
        raise ValueError("evaluation scenario catalog must contain scenarios")
    return payload


def load_scenario(
    scenario_id: str,
    *,
    catalog_path: str | Path = CATALOG_PATH,
) -> EvaluationScenario:
    payload = load_scenario_catalog(catalog_path)
    raw = payload["scenarios"].get(scenario_id)
    if not isinstance(raw, dict):
        raise KeyError(f"unknown evaluation scenario: {scenario_id}")
    side_effects = raw.get("allowed_side_effects")
    if not isinstance(side_effects, list) or not all(
        isinstance(item, str) and item for item in side_effects
    ):
        raise ValueError(f"invalid allowed_side_effects for {scenario_id}")
    for flag in ("external_model_allowed", "formal_evaluation"):
        if not isinstance(raw.get(flag), bool):
            raise ValueError(f"{flag} must be a boolean for {scenario_id}")
    scope = raw.get("observation_scope")
    if scope not in {"contract_only", "retrieval_only", "full"}:
        raise ValueError(f"invalid observation_scope for {scenario_id}")
    return EvaluationScenario(
        scenario_id=scenario_id,
        observation_scope=scope,
        allowed_side_effects=tuple(side_effects),
        external_model_allowed=raw["external_model_allowed"],
        formal_evaluation=raw["formal_evaluation"],
    )
