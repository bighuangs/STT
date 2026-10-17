import math

import pytest

from app.rag.query.rrf_service import rrf, validate_rrf_inputs


def _doc(chunk_id: str | None, content: str = "text") -> dict:
    return {"chunk_id": chunk_id, "content": content}


def test_rrf_empty_sources_are_an_empty_result():
    assert rrf([]) == []
    assert rrf([([], 1.0)]) == []


def test_rrf_accumulates_weights_across_sources_and_keeps_source_data():
    result = rrf(
        [
            ([_doc("shared", "first copy"), _doc("only-a")], 2.0),
            ([_doc("shared", "second copy"), _doc("only-b")], 1.0),
        ],
        k=10,
        top=3,
    )
    assert [row["chunk_id"] for row in result] == ["shared", "only-a", "only-b"]
    assert result[0]["content"] == "first copy"
    assert result[0]["score"] == pytest.approx(3.0 / 11.0)


def test_rrf_deduplicates_a_chunk_within_one_ranked_source():
    duplicate_source = [_doc("dup", "first rank"), _doc("dup", "duplicate")]
    result = rrf([(duplicate_source, 1.0)], k=10, top=5)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "dup"
    assert result[0]["content"] == "first rank"
    assert result[0]["score"] == pytest.approx(1.0 / 11.0)
    assert result[0]["rrf_score"] == pytest.approx(1.0 / 11.0)


def test_rrf_ignores_missing_ids_and_does_not_mutate_input():
    source = [_doc(None), _doc(""), _doc("kept")]
    snapshot = [row.copy() for row in source]
    assert [row["chunk_id"] for row in rrf([(source, 1.0)])] == ["kept"]
    assert source == snapshot


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"k": 0}, "k"), ({"k": -1}, "k"), ({"top": -1}, "top")],
)
def test_rrf_rejects_invalid_rank_boundaries(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rrf([([_doc("a")], 1.0)], **kwargs)


@pytest.mark.parametrize("weight", [-1.0, math.inf, -math.inf, math.nan])
def test_rrf_rejects_negative_or_nonfinite_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        rrf([([_doc("a")], weight)])


def test_rrf_zero_weight_does_not_emit_a_candidate():
    assert rrf([([_doc("a")], 0.0)]) == []


def test_rrf_top_zero_is_an_explicit_empty_result():
    assert rrf([([_doc("a")], 1.0)], top=0) == []


def test_validate_rrf_inputs_allows_one_route_but_rejects_no_routes():
    assert validate_rrf_inputs({"embedding_chunks": [_doc("a")]}) == ([_doc("a")], [])
    with pytest.raises(ValueError, match="均为空"):
        validate_rrf_inputs({"embedding_chunks": [], "hyde_embedding_chunks": []})
