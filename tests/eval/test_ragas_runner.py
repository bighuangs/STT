from __future__ import annotations

import json
from pathlib import Path

import pytest

import evals.ragas_support.runner as runner_module
from evals.ragas_support.runner import (
    JudgeConfig,
    PricingConfig,
    RagasRunnerError,
    bootstrap_mean_ci,
    estimate_judge_usage,
    evaluate_records,
    join_observations,
    validate_judge_metrics,
    write_metric_records,
)
from scripts.run_ragas_evaluation import main


OUTPUT_FIELDS = {"metric", "value", "status", "error", "latency", "usage", "hash"}


def _sample(case_id: str = "case-1") -> dict:
    return {
        "schema_version": "stt.ragas.sample.v1",
        "case_id": case_id,
        "dataset_version": "v1",
        "difficulty": "easy",
        "user_input": "不会持久化的问题",
        "reference": "不会持久化的参考答案",
        "reference_context_ids": ["a", "b"],
        "tags": ["printer", "zh-CN"],
        "sensitivity": "internal",
        "judge_eligible": True,
    }


def _find(records: list[dict], suffix: str) -> dict:
    return next(record for record in records if record["metric"].endswith(suffix))


def test_offline_id_metrics_match_ragas_set_semantics_and_expose_missing() -> None:
    first = _sample()
    first["retrieved_context_ids"] = ["a", "x", "a"]
    missing = _sample("case-2")

    records = evaluate_records([first, missing], bootstrap_resamples=100)

    precision = _find(records, "case-1/id_based_context_precision")
    recall = _find(records, "case-1/id_based_context_recall")
    assert precision["value"] == 0.5
    assert recall["value"] == 0.5
    assert _find(records, "case-2/id_based_context_precision")["status"] == "missing"
    assert all(set(record) == OUTPUT_FIELDS for record in records)


def test_empty_denominators_are_undefined_instead_of_serialized_nan() -> None:
    sample = _sample()
    sample["retrieved_context_ids"] = []
    sample["reference_context_ids"] = []

    records = evaluate_records([sample], bootstrap_resamples=100)

    assert _find(records, "case-1/id_based_context_precision")["status"] == "undefined"
    assert _find(records, "case-1/id_based_context_recall")["status"] == "undefined"
    assert "NaN" not in "\n".join(json.dumps(record) for record in records)


def test_observation_join_extracts_answer_ranked_ids_context_and_existing_metrics() -> None:
    sample = _sample()
    observation = {
        "case_id": "case-1",
        "metrics": {
            "metrics": {
                "generation": {"faithfulness": 0.75},
                "retrieval": {"mrr": 1.0},
            }
        },
        "observation": {
            "answer": "不会持久化的系统答案",
            "reranked": [
                {"chunk_id": "a", "text": "候选证据 A"},
                {"chunk_id": "x", "text": "候选证据 X"},
            ],
            "context": [{"chunk_id": "a", "content": "最终证据 A"}],
        },
    }

    joined, unknown = join_observations([sample], [observation])
    assert unknown == 0
    assert joined[0]["response"] == "不会持久化的系统答案"
    assert joined[0]["retrieved_context_ids"] == ["a", "x"]
    assert joined[0]["retrieved_contexts"] == ["最终证据 A"]

    records = evaluate_records([sample], observations=[observation], bootstrap_resamples=100)
    assert _find(records, "case-1/id_based_context_precision")["value"] == 0.5
    assert _find(records, "existing%2Fgeneration.faithfulness")["value"] == 0.75


def test_bootstrap_ci_is_seeded_and_slices_include_difficulty_and_tags() -> None:
    assert bootstrap_mean_ci([0.0, 1.0], resamples=200, seed=7) == (
        bootstrap_mean_ci([0.0, 1.0], resamples=200, seed=7)
    )
    sample = _sample()
    sample["retrieved_context_ids"] = ["a"]
    records = evaluate_records([sample], bootstrap_resamples=100)
    metric_names = {record["metric"] for record in records}
    assert any("difficulty%3Deasy" in name and name.endswith("/ci95_low") for name in metric_names)
    assert any("tag%3Dprinter" in name and name.endswith("/mean") for name in metric_names)


def test_judge_call_and_aliyun_cost_estimate_is_parameterized() -> None:
    sample = _sample()
    sample.update(
        {
            "response": "回答",
            "retrieved_contexts": ["上下文一", "上下文二"],
        }
    )
    pricing = PricingConfig(
        input_price_per_million=2.0,
        output_price_per_million=8.0,
        embedding_price_per_million=1.0,
        estimated_input_tokens_per_call=100,
        estimated_output_tokens_per_call=20,
        estimated_embedding_tokens_per_case=600,
    )

    records = estimate_judge_usage([sample], pricing=pricing, case_ids=["case-1"])

    calls = _find(records, "total/llm_calls")
    cost = _find(records, "total/cost")
    # 三项指标的逻辑调用量依次为 2、3、4。
    assert calls["value"] == 9
    assert cost["value"] == pytest.approx(0.00384)
    assert cost["usage"]["embedding_requests"] == 2
    assert cost["usage"]["external_requests"] == 11
    assert cost["usage"]["kind"] == "estimate"


def test_default_twenty_four_case_plan_separates_llm_embedding_and_external_requests() -> None:
    samples = []
    for index in range(24):
        sample = _sample(f"case-{index}")
        sample.update({"response": "回答", "retrieved_contexts": ["上下文"]})
        samples.append(sample)

    records = estimate_judge_usage(
        samples,
        case_ids=[sample["case_id"] for sample in samples],
    )

    assert _find(records, "total/llm_calls")["value"] == 216
    assert _find(records, "total/embedding_requests")["value"] == 48
    assert _find(records, "total/external_requests")["value"] == 264
    assert _find(records, "total/cost")["value"] == pytest.approx(6.936)


@pytest.mark.parametrize(
    "name",
    ["MultiModalFaithfulness", "multi_modal_faithfulness", "multi-modal-faithfulness"],
)
def test_multimodal_faithfulness_is_strictly_forbidden(name: str) -> None:
    with pytest.raises(RagasRunnerError, match="forbidden"):
        validate_judge_metrics([name])


def test_judge_requires_separate_paid_call_authority_before_optional_import() -> None:
    sample = _sample()
    sample.update({"response": "回答", "retrieved_contexts": ["上下文"]})
    config = JudgeConfig(enabled=True, allow_paid_model_calls=False)

    with pytest.raises(RagasRunnerError, match="allow-paid-model-calls"):
        evaluate_records([sample], bootstrap_resamples=100, judge=config)


def test_judge_hard_cap_and_budget_block_before_optional_import() -> None:
    samples = []
    for index in range(25):
        sample = _sample(f"case-{index}")
        sample.update({"response": "回答", "retrieved_contexts": ["上下文"]})
        samples.append(sample)
    over_cap = JudgeConfig(
        enabled=True,
        allow_paid_model_calls=True,
        sample_case_ids=tuple(sample["case_id"] for sample in samples),
    )
    with pytest.raises(RagasRunnerError, match="cap of 24"):
        evaluate_records(
            samples,
            bootstrap_resamples=100,
            judge=over_cap,
            pricing=PricingConfig(
                input_price_per_million=1.0,
                output_price_per_million=1.0,
                embedding_price_per_million=1.0,
            ),
        )

    one = samples[0]
    over_budget = JudgeConfig(
        enabled=True,
        allow_paid_model_calls=True,
        sample_case_ids=(one["case_id"],),
    )
    with pytest.raises(RagasRunnerError, match="exceeds hard budget"):
        evaluate_records(
            [one],
            bootstrap_resamples=100,
            judge=over_budget,
            pricing=PricingConfig(
                input_price_per_million=1_000_000.0,
                output_price_per_million=1_000_000.0,
                embedding_price_per_million=1_000_000.0,
            ),
        )


def test_restricted_sample_is_rejected_by_judge_preflight() -> None:
    sample = _sample()
    sample.update(
        {
            "response": "回答",
            "retrieved_contexts": ["上下文"],
            "sensitivity": "restricted",
        }
    )
    judge = JudgeConfig(
        enabled=True,
        allow_paid_model_calls=True,
        sample_case_ids=(sample["case_id"],),
    )
    with pytest.raises(RagasRunnerError, match="judge_not_eligible"):
        evaluate_records(
            [sample],
            bootstrap_resamples=100,
            judge=judge,
            pricing=PricingConfig(
                input_price_per_million=1.0,
                output_price_per_million=1.0,
                embedding_price_per_million=1.0,
            ),
        )


def test_judge_client_disables_sdk_retries_and_caps_output_before_fake_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        async def close(self) -> None:
            seen["closed"] = True

    class FakeResult:
        value = 0.75

    class FakeMetric:
        def __init__(self, **kwargs):
            del kwargs

        async def ascore(self, **kwargs):
            del kwargs
            return FakeResult()

    def fake_llm_factory(model: str, **kwargs):
        seen["llm"] = {"model": model, **kwargs}
        return object()

    def fake_embedding_factory(provider: str, **kwargs):
        seen["embedding"] = {"provider": provider, **kwargs}
        return object()

    classes = {name: FakeMetric for name in runner_module.ALLOWED_JUDGE_METRICS}
    monkeypatch.setattr(
        runner_module,
        "_load_ragas_components",
        lambda: (FakeClient, classes, fake_llm_factory, fake_embedding_factory),
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    sample = _sample()
    sample.update({"response": "回答", "retrieved_contexts": ["上下文"]})
    judge = JudgeConfig(
        enabled=True,
        allow_paid_model_calls=True,
        sample_case_ids=(sample["case_id"],),
    )

    records = evaluate_records([sample], bootstrap_resamples=100, judge=judge)

    assert seen["client"]["max_retries"] == 0
    assert seen["llm"]["model"] == "qwen3.7-plus"
    assert seen["llm"]["max_tokens"] == 2000
    assert seen["closed"] is True
    assert any(record["metric"].endswith("ragas/faithfulness") for record in records)


def test_cli_output_whitelist_does_not_persist_raw_judge_inputs(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    sample = _sample()
    samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    observation = {
        "case_id": "case-1",
        "response": "绝不能写入结果的回答",
        "retrieved_context_ids": ["a"],
        "retrieved_contexts": ["绝不能写入结果的上下文"],
    }
    observations_path.write_text(
        json.dumps(observation, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--samples",
            str(samples_path),
            "--observations",
            str(observations_path),
            "--output",
            str(output_path),
            "--bootstrap-resamples",
            "100",
        ]
    )

    assert exit_code == 0
    persisted = output_path.read_text(encoding="utf-8")
    for secret_text in (
        sample["user_input"],
        sample["reference"],
        observation["response"],
        observation["retrieved_contexts"][0],
    ):
        assert secret_text not in persisted
    assert all(set(json.loads(line)) == OUTPUT_FIELDS for line in persisted.splitlines())


def test_cli_accepts_multiple_observation_files(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    first_observations_path = tmp_path / "observations-1.jsonl"
    second_observations_path = tmp_path / "observations-2.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    first_sample = _sample("case-1")
    second_sample = _sample("case-2")
    samples_path.write_text(
        "\n".join(
            json.dumps(sample, ensure_ascii=False)
            for sample in (first_sample, second_sample)
        )
        + "\n",
        encoding="utf-8",
    )
    first_observations_path.write_text(
        json.dumps(
            {"case_id": "case-1", "retrieved_context_ids": ["a"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    second_observations_path.write_text(
        json.dumps(
            {"case_id": "case-2", "retrieved_context_ids": ["a"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--samples",
            str(samples_path),
            "--observations",
            str(first_observations_path),
            "--observations",
            str(second_observations_path),
            "--output",
            str(output_path),
            "--bootstrap-resamples",
            "100",
        ]
    )

    assert exit_code == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    records_by_metric = {record["metric"]: record for record in records}
    assert records_by_metric["case/case-1/id_based_context_precision"]["status"] == "ok"
    assert records_by_metric["case/case-2/id_based_context_precision"]["status"] == "ok"
    assert records_by_metric["join/unknown_observations"]["value"] == 0


def test_writer_rejects_fields_outside_the_persistence_whitelist(tmp_path: Path) -> None:
    with pytest.raises(RagasRunnerError, match="forbidden fields"):
        write_metric_records(
            tmp_path / "bad.jsonl",
            [
                {
                    "metric": "x",
                    "value": 1.0,
                    "status": "ok",
                    "error": None,
                    "latency": {"milliseconds": 0.0},
                    "usage": {},
                    "hash": "sha256:x",
                    "context": "不得写入",
                }
            ],
        )
