import pytest

from quantlab_env import questions
from quantlab_env.tools import Episode, compute, execute, get_fundamentals


def revenue_for(period_end: str, as_of: str) -> int | None:
    result = get_fundamentals("JNJ", "revenue", as_of)
    return next((r["value"] for r in result.get("rows", []) if r["period_end"] == period_end), None)


def test_point_in_time_sees_restatement_only_after_it_was_filed():
    assert revenue_for("2022-01-02", "2022-02-16") is None  # 10-K filed 2022-02-17
    assert revenue_for("2022-01-02", "2022-02-17") == 93_775_000_000
    assert revenue_for("2022-01-02", "2024-02-15") == 93_775_000_000
    assert revenue_for("2022-01-02", "2024-02-16") == 78_740_000_000  # restated


def test_nothing_before_first_filing_is_no_data():
    assert get_fundamentals("KO", "revenue", "2019-01-01")["error"]["code"] == "no_data"


def test_rows_expose_only_values_and_metadata():
    row = get_fundamentals("AAPL", "net_income", "2024-12-01")["rows"][-1]
    assert set(row) == {"period_end", "value", "filed", "form"}


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"ticker": "TSLA", "concept": "revenue", "as_of": "2024-01-01"}, "unknown_ticker"),
        ({"ticker": "AAPL", "concept": "ebitda", "as_of": "2024-01-01"}, "unknown_concept"),
        ({"ticker": "AAPL", "concept": "revenue", "as_of": "2024-02-30"}, "bad_argument"),
    ],
)
def test_lookup_errors(args, code):
    assert get_fundamentals(**args)["error"]["code"] == code


@pytest.mark.parametrize("expr", ["2 ** 10", "abs(-3)", "x + 1", "1; 2", "True + 1", "'1' * 3"])
def test_compute_accepts_only_literal_arithmetic(expr):
    assert compute(expr)["error"]["code"] == "bad_argument"


def test_compute_arithmetic_and_limits():
    assert compute("(20878 - 0) / 78740 * 100")["value"] == pytest.approx(26.515, abs=1e-3)
    assert compute("1 / 0")["error"]["code"] == "bad_argument"
    assert compute("1+" * 200 + "1")["error"]["code"] == "bad_argument"


@pytest.mark.parametrize("expr", ["1e308 * 10", "1e999", "1e999 - 1e999"])
def test_compute_rejects_non_finite_results(expr):
    # inf/nan would reach the agent over MCP as `value: null` with no error
    assert compute(expr)["error"]["code"] == "bad_argument"


def test_execute_logs_every_call_including_failures():
    ep = Episode()
    execute("nope", {}, ep)
    execute("compute", "1 + 1", ep)
    execute("compute", {"expr": "1 + 1"}, ep)
    assert [c.ok for c in ep.calls] == [False, False, True]
    assert ep.calls[0].result["error"]["code"] == "unknown_tool"
    ep.reset()
    assert ep.calls == []


def test_every_default_question_is_answerable_from_the_tools():
    qs = questions.default_questions()
    assert len(qs) == 18
    for q in qs:
        rows = {r["period_end"]: r["value"] for r in get_fundamentals(q.ticker, "revenue", q.as_of)["rows"]}
        assert rows[q.period_end] == q.revenue
