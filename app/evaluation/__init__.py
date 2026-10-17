"""Deterministic, dependency-free evaluation primitives for STT.

The package deliberately accepts and returns plain dictionaries.  This keeps the
offline quality gate usable from tests, command-line runners, HTTP adapters and
future persistence implementations without coupling the metric definitions to
one transport or database model.
"""

from app.domain.evaluation_proof import (
    OBSERVATION_SCHEMA_V3,
    answer_proof_evaluation,
    is_v3_observation,
    project_proof_observation,
    proof_citation_metrics,
    proof_generation_metrics,
)

from .metrics import aggregate_results, evaluate_case, paired_diff
from .reports import render_html_report, render_json_report, render_markdown_report

__all__ = [
    "OBSERVATION_SCHEMA_V3",
    "aggregate_results",
    "answer_proof_evaluation",
    "evaluate_case",
    "is_v3_observation",
    "paired_diff",
    "project_proof_observation",
    "proof_citation_metrics",
    "proof_generation_metrics",
    "render_html_report",
    "render_json_report",
    "render_markdown_report",
]
