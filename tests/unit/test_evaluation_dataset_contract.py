from __future__ import annotations

import copy

import pytest

from app.domain.evaluation import validate_dataset_manifest


def _manifest(case_count: int = 3) -> dict:
    return {
        "dataset_key": "suite-contract-test",
        "name": "Suite contract test",
        "version": "1.0.0",
        "schema_version": "stt.eval.v1",
        "source": {"logical_name": "test.pdf"},
        "cases": [
            {
                "case_id": f"CASE-{ordinal:03d}",
                "query": f"Question {ordinal}?",
                "reference_answer": f"Answer {ordinal}",
            }
            for ordinal in range(1, case_count + 1)
        ],
    }


def test_legacy_manifest_without_suite_fields_remains_valid() -> None:
    validate_dataset_manifest(_manifest())


def test_complete_suite_contract_is_valid() -> None:
    manifest = {
        **_manifest(),
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": "easy",
        "minimum_case_count": 3,
    }

    validate_dataset_manifest(manifest)


@pytest.mark.parametrize("declared_field", ["suite_key", "difficulty", "minimum_case_count"])
def test_any_suite_field_requires_the_complete_contract(declared_field: str) -> None:
    values = {
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": "medium",
        "minimum_case_count": 3,
    }
    manifest = _manifest()
    manifest[declared_field] = values[declared_field]

    with pytest.raises(ValueError, match="suite manifest missing fields"):
        validate_dataset_manifest(manifest)


@pytest.mark.parametrize("difficulty", ["simple", "EASY", "", None])
def test_suite_difficulty_uses_the_canonical_three_levels(difficulty: object) -> None:
    manifest = {
        **_manifest(),
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": difficulty,
        "minimum_case_count": 3,
    }

    with pytest.raises(ValueError, match="easy, medium, hard"):
        validate_dataset_manifest(manifest)


@pytest.mark.parametrize("minimum_case_count", [True, 0, -1, 1.5, "3"])
def test_suite_minimum_case_count_is_a_positive_integer(
    minimum_case_count: object,
) -> None:
    manifest = {
        **_manifest(),
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": "hard",
        "minimum_case_count": minimum_case_count,
    }

    with pytest.raises(ValueError, match="positive integer"):
        validate_dataset_manifest(manifest)


def test_suite_rejects_fewer_cases_than_its_declared_minimum() -> None:
    manifest = {
        **_manifest(case_count=2),
        "suite_key": "three-difficulty-live-2026-08-10",
        "difficulty": "hard",
        "minimum_case_count": 3,
    }

    with pytest.raises(ValueError, match="requires at least 3 cases, got 2"):
        validate_dataset_manifest(manifest)


def test_suite_key_must_be_non_empty() -> None:
    manifest = {
        **copy.deepcopy(_manifest()),
        "suite_key": " ",
        "difficulty": "easy",
        "minimum_case_count": 3,
    }

    with pytest.raises(ValueError, match="suite_key must be a non-empty string"):
        validate_dataset_manifest(manifest)


def test_explicit_security_expectation_contract_is_valid() -> None:
    manifest = _manifest()
    manifest["cases"][0]["metadata"] = {
        "security_expectation": {
            "contract_version": "stt.eval.security.v1",
            "stage": "input",
            "action": "block",
            "reason_codes": ["DIRECT_PROMPT_INJECTION"],
            "forbidden_canaries": ["AUTH_TOKEN_SECRET"],
        }
    }

    validate_dataset_manifest(manifest)


@pytest.mark.parametrize(
    "expectation",
    [
        {},
        {"contract_version": "stt.eval.security.v0", "stage": "input", "action": "block", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "prompt", "action": "block", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "evidence", "action": "block", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "output", "action": "block", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "allow", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "redact", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "refuse", "reason_codes": ["DIRECT_PROMPT_INJECTION"]},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "block", "reason_codes": []},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "block", "reason_codes": ["raw attack text"]},
        {"contract_version": "stt.eval.security.v1", "stage": "input", "action": "block", "reason_codes": ["DIRECT_PROMPT_INJECTION"], "forbidden_canaries": [""]},
    ],
)
def test_security_expectation_rejects_incomplete_or_unsafe_contracts(
    expectation: object,
) -> None:
    manifest = _manifest()
    manifest["cases"][0]["metadata"] = {"security_expectation": expectation}

    with pytest.raises(ValueError, match="security_expectation"):
        validate_dataset_manifest(manifest)
