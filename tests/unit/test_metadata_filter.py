import pytest

from app.rag.query.query_rewrite_service import _meta_filter


@pytest.fixture
def metas() -> dict:
    return {
        "category": {
            "Pressure": ["doc-b", "doc-a", "doc-a"],
            "Temperature": ["doc-c"],
        },
        "range": {
            "10": ["doc-a"],
            "20": ["doc-b"],
            "not-a-number": ["doc-c"],
        },
        "note": {"": ["doc-c"], "Safety Manual": ["doc-a", "doc-b"]},
    }


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"key": "category", "op": "=", "value": "pressure"}, ["doc-a", "doc-b"]),
        ({"key": "category", "op": "≠", "value": "pressure"}, ["doc-c"]),
        (
            {"key": "category", "op": "contains", "value": "press"},
            ["doc-a", "doc-b"],
        ),
        (
            {"key": "category", "op": "not contains", "value": "press"},
            ["doc-c"],
        ),
        ({"key": "category", "op": "start with", "value": "temp"}, ["doc-c"]),
        (
            {"key": "category", "op": "end with", "value": "sure"},
            ["doc-a", "doc-b"],
        ),
        ({"key": "note", "op": "empty", "value": "ignored"}, ["doc-c"]),
        (
            {"key": "note", "op": "not empty", "value": "ignored"},
            ["doc-a", "doc-b"],
        ),
    ],
)
def test_string_operators_are_case_insensitive_unique_and_sorted(
    metas, condition, expected
):
    assert _meta_filter(metas, [condition]) == expected


def test_in_is_exact_not_substring_and_ignores_blank_tokens(metas):
    assert _meta_filter(
        metas,
        [{"key": "category", "op": "in", "value": " Temperature，,Pressure "}],
    ) == ["doc-a", "doc-b", "doc-c"]
    assert _meta_filter(
        metas, [{"key": "category", "op": "in", "value": "Press"}]
    ) == []
    assert _meta_filter(
        metas, [{"key": "category", "op": "in", "value": " , ， "}]
    ) == []


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (">", "10", ["doc-b"]),
        ("≥", "20", ["doc-b"]),
        ("<", "20", ["doc-a"]),
        ("≤", "10", ["doc-a"]),
    ],
)
def test_numeric_comparisons_ignore_non_numeric_metadata(
    metas, operator, value, expected
):
    assert _meta_filter(
        metas, [{"key": "range", "op": operator, "value": value}]
    ) == expected


def test_and_or_logic_and_stable_order(metas):
    category = {"key": "category", "op": "=", "value": "Pressure"}
    exact_range = {"key": "range", "op": "=", "value": "10"}
    assert _meta_filter(metas, [category, exact_range], "and") == ["doc-a"]
    assert _meta_filter(
        metas,
        [
            category,
            {"key": "category", "op": "=", "value": "Temperature"},
        ],
        "or",
    ) == ["doc-a", "doc-b", "doc-c"]


def test_empty_filters_and_missing_keys_do_not_broaden_scope(metas):
    assert _meta_filter(metas, []) == []
    assert _meta_filter(
        metas, [{"key": "unknown", "op": "=", "value": "anything"}]
    ) == []


@pytest.mark.parametrize("logic", ["xor", "OR 1=1", "", None])
def test_unknown_logic_is_rejected_instead_of_silently_using_or(metas, logic):
    with pytest.raises(ValueError, match="logic"):
        _meta_filter(
            metas,
            [{"key": "category", "op": "=", "value": "Pressure"}],
            logic,
        )


def test_blank_broadening_filters_match_nothing(metas):
    assert _meta_filter(
        metas, [{"key": "category", "op": "contains", "value": ""}]
    ) == []
    assert _meta_filter(
        metas, [{"key": "category", "op": "not contains", "value": ""}]
    ) == []


def test_malicious_strings_are_inert_data(metas):
    payload = "Pressure' OR 1=1 --"
    assert _meta_filter(
        metas, [{"key": "category", "op": "=", "value": payload}]
    ) == []
    assert _meta_filter(
        metas,
        [
            {
                "key": "category",
                "op": "__import__('os').system",
                "value": payload,
            }
        ],
    ) == []
