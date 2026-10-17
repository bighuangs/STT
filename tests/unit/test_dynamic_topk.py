from copy import deepcopy

from app.rag.query.rerank_service import dynamic_topk


def _scores(*values: float) -> list[dict]:
    return [
        {"id": str(index), "score": value} for index, value in enumerate(values)
    ]


def test_dynamic_topk_handles_empty_and_singleton_inputs():
    assert dynamic_topk([]) == []
    singleton = _scores(0.5)
    assert dynamic_topk(singleton) == singleton


def test_dynamic_topk_caps_smooth_tail_at_ten_without_mutating_input():
    candidates = _scores(*(1.0 - index * 0.01 for index in range(12)))
    snapshot = deepcopy(candidates)
    assert len(dynamic_topk(candidates)) == 10
    assert candidates == snapshot


def test_dynamic_topk_cuts_at_a_clear_ratio_cliff():
    candidates = _scores(0.92, 0.88, 0.20, 0.19)
    assert dynamic_topk(candidates) == candidates[:2]


def test_dynamic_topk_keeps_equal_zero_scores_without_division_errors():
    equal = _scores(0.0, 0.0, 0.0)
    assert dynamic_topk(equal) == equal


def test_dynamic_topk_uses_absolute_gap_for_unbounded_model_scores():
    candidates = _scores(8.0, 7.5, 4.0, 3.9)
    assert dynamic_topk(candidates) == candidates[:2]


def test_dynamic_topk_missing_scores_are_zero_at_the_tail():
    candidates = [
        {"id": "good", "score": 0.9},
        {"id": "missing"},
        {"id": "zero", "score": 0.0},
    ]
    assert dynamic_topk(candidates) == candidates[:1]
