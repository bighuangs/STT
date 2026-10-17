from __future__ import annotations

from app.application.observability_service import (
    MetricRegistry,
    TraceRecorder,
    UsageLedger,
    UsageRecord,
    redact,
)


def test_redaction_keeps_diagnostics_but_never_records_body_prompt_or_secret() -> None:
    value = {
        "tenant_id": "tenant-a",
        "stage": "generation",
        "prompt": "CANARY_PROMPT",
        "document_content": "CANARY_BODY",
        "Authorization": "Bearer CANARY_SECRET",
        "nested": {"query": "在使用 HAK180 之前需要做什么？"},
    }

    safe = redact(value)
    rendered = str(safe)

    assert safe["tenant_id"] == "tenant-a"
    assert safe["stage"] == "generation"
    assert "CANARY_PROMPT" not in rendered
    assert "CANARY_BODY" not in rendered
    assert "CANARY_SECRET" not in rendered
    assert "在使用 HAK180" not in rendered
    assert safe["prompt"]["hash"].startswith("sha256:")


def test_trace_parentage_and_error_code_are_preserved_without_exception_text() -> None:
    recorder = TraceRecorder()

    try:
        with recorder.span("query", attributes={"query": "CANARY"}):
            with recorder.span("milvus.search", attributes={"stage": "retrieval"}):
                raise TimeoutError("secret backend message")
    except TimeoutError:
        pass

    records = recorder.records()
    # span 按完成时间写入，因此内部 span 先于父 span。
    inner, outer = records
    assert outer.trace_id == inner.trace_id
    assert inner.parent_span_id == outer.span_id
    assert inner.status == "error"
    assert inner.error_code == "TIMEOUTERROR"
    assert "CANARY" not in str(records)
    assert "secret backend message" not in str(records)


def test_metrics_reject_high_cardinality_labels() -> None:
    registry = MetricRegistry()
    registry.increment("rag_query_total", labels={"route": "local_hybrid", "result": "ok"})
    registry.observe("rag_query_latency_ms", 25.0, labels={"route": "local_hybrid"})

    snapshot = registry.snapshot()
    assert snapshot["counters"][0]["value"] == 1.0
    assert snapshot["histograms"][0]["count"] == 1

    try:
        registry.increment("bad_metric", labels={"query_run_id": "run-1"})
    except ValueError as exc:
        assert "high-cardinality" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("high-cardinality label must be rejected")


def test_metric_histogram_is_constant_memory_and_series_are_bounded() -> None:
    registry = MetricRegistry(max_series=2)
    for value in range(10_000):
        registry.observe(
            "rag_query_latency_ms",
            float(value),
            labels={"route": "local_hybrid"},
        )
    registry.increment("rag_query_total", labels={"result": "ok"})

    histogram = registry.snapshot()["histograms"][0]
    assert histogram == {
        "name": "rag_query_latency_ms",
        "labels": {"route": "local_hybrid"},
        "count": 10_000,
        "min": 0.0,
        "max": 9_999.0,
        "avg": 4_999.5,
    }
    try:
        registry.increment("third_series", labels={"result": "failed"})
    except ValueError as exc:
        assert "series limit" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("metric series cardinality must be bounded")


def test_usage_ledger_counts_every_fallback_call_and_marks_estimates() -> None:
    ledger = UsageLedger()
    for sequence, model, estimated in (
        (0, "primary", False),
        (1, "fallback", True),
    ):
        ledger.append(
            UsageRecord(
                usage_id=f"use_{sequence + 1:032x}",
                tenant_id="tenant-a",
                kb_id="kb-hak180",
                query_run_id="qrun-1",
                provider="fake",
                model_id=model,
                scene="generation",
                input_tokens=100,
                output_tokens=20,
                unit_price_version="price-v1",
                cost=0.1,
                estimated=estimated,
                fallback_sequence=sequence,
            )
        )

    summary = ledger.aggregate("tenant-a", query_run_id="qrun-1")
    assert summary == {
        "tenant_id": "tenant-a",
        "query_run_id": "qrun-1",
        "calls": 2,
        "input_tokens": 200,
        "output_tokens": 40,
        "cache_read_tokens": 0,
        "cost": 0.2,
        "estimated": True,
        "record_count": 2,
    }


def test_usage_ledger_has_a_hard_memory_bound_and_evicts_oldest() -> None:
    ledger = UsageLedger(max_records=2)
    for sequence in range(3):
        ledger.append(
            UsageRecord(
                usage_id=f"use_{sequence + 1:032x}",
                tenant_id="tenant-a",
                query_run_id=f"qry_{sequence}",
                provider="fake",
                model_id="fake",
                scene="generation",
                unit_price_version="price-v1",
                estimated=True,
            )
        )

    assert ledger.aggregate("tenant-a")["record_count"] == 2
    assert ledger.aggregate("tenant-a", query_run_id="qry_0")["record_count"] == 0
    assert ledger.aggregate("tenant-a", query_run_id="qry_2")["record_count"] == 1
