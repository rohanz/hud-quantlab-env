import pytest

from quantlab_env import data, questions
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


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_begin_alone_is_not_overlapped():
    ep = Episode()
    a = ep.begin()
    assert ep.generation == a and not ep.overlapped(a)
    ep.close(a)
    b = ep.begin()  # sequential: A closed before B started
    assert b != a and not ep.overlapped(b)


def test_begin_clears_the_log_and_marks_both_overlapping_episodes():
    ep = Episode()
    a = ep.begin()
    execute("compute", {"expr": "1 + 1"}, ep)
    b = ep.begin()  # a second session starts before A is graded
    assert ep.calls == [] and b != a and ep.generation == b
    # B's log may hold calls A makes from now on, and a B that makes none
    # would be graded on A's work, so neither can be graded.
    assert ep.overlapped(a) and ep.overlapped(b)


def test_close_removes_the_episode_and_its_mark():
    ep = Episode()
    a = ep.begin()
    b = ep.begin()
    ep.close(a)
    ep.close(b)
    assert not ep.open and not ep.overlapping
    c = ep.begin()
    assert not ep.overlapped(c)
    ep.close(c)
    ep.close(c)  # closing twice is harmless


def test_an_abandoned_episode_expires_and_does_not_overlap_a_later_one():
    clock = Clock()
    ep = Episode(timeout=60, clock=clock)
    a = ep.begin()  # never graded or closed
    clock.now = 61
    b = ep.begin()
    assert not ep.overlapped(b) and list(ep.open) == [b]
    assert ep.overlapped(a)  # if A comes back, its log is gone


def test_an_open_episode_within_the_timeout_still_overlaps():
    clock = Clock()
    ep = Episode(timeout=60, clock=clock)
    a = ep.begin()
    clock.now = 59
    b = ep.begin()
    assert ep.overlapped(a) and ep.overlapped(b)


def test_every_default_question_is_answerable_from_the_tools():
    qs = questions.default_questions()
    assert len(qs) == 18
    for q in qs:
        rows = {r["period_end"]: r["value"] for r in get_fundamentals(q.ticker, "revenue", q.as_of)["rows"]}
        assert rows[q.period_end] == q.revenue


def test_every_probe_question_is_answerable_from_the_tools():
    qs = questions.probe_questions()
    assert len(qs) >= 60
    assert {q.kind for q in qs} == {
        "fy_net_margin",
        "latest_net_margin",
        "net_margin_change",
        "revenue_growth",
        "net_margin_gap",
    }
    assert len({(q.kind, tuple(sorted(questions.template_args(q).items()))) for q in qs}) == len(qs)
    for q in qs:
        for i in q.inputs:
            rows = {r["period_end"]: r["value"] for r in get_fundamentals(i.ticker, i.concept, q.as_of)["rows"]}
            assert rows[i.period_end] == i.value
        assert abs(q.gold) >= questions.MIN_CHANGE


def test_every_chain_question_is_answerable_from_the_tools():
    qs = questions.chain_questions()
    assert {q.kind for q in qs} == set(questions.CHAIN_KINDS)
    assert len({(q.kind, tuple(sorted(questions.template_args(q).items()))) for q in qs}) == len(qs)
    for q in qs:
        for i in q.inputs:
            rows = {r["period_end"]: r["value"] for r in get_fundamentals(i.ticker, i.concept, q.as_of)["rows"]}
            assert rows[i.period_end] == i.value
        assert abs(q.gold) >= questions.MIN_CHANGE
        assert 5 <= q.par_calls <= 12


def test_probe_questions_are_posed_by_their_template_args():
    for q in questions.probe_questions() + questions.chain_questions():
        args = questions.template_args(q)
        builder = getattr(questions, q.kind)
        args = {"fiscal": args.pop("fiscal_year"), **args} if "fiscal_year" in args else args
        assert builder(**args) == q


def test_fiscal_year_labels_are_unambiguous_and_match_the_filings():
    """Each company's periods have distinct labels under the rule, and where the fixture records the
    filing's fiscal-year focus (fixtures built after 2026-10-06), a period's first 10-K agrees with it."""
    for company in data.companies().values():
        facts = company["facts"]
        periods = sorted({r["period_end"] for r in facts["revenue"]})
        labels = [questions.fiscal_year(p) for p in periods]
        assert len(set(labels)) == len(labels), (company["name"], periods)
        for p in periods:
            first = min((r for r in facts["revenue"] if r["period_end"] == p), key=lambda r: r["filed"])
            if first.get("fy") is not None and first["filed"] < _next_period_filed(facts, p):
                assert first["fy"] == questions.fiscal_year(p), (company["name"], p, first)


def _next_period_filed(facts: dict, period_end: str) -> str:
    """When any later period was first filed: a period's own 10-K comes before it."""
    own = [r["filed"] for r in facts["revenue"] if r["period_end"] > period_end]
    return min(own, default="9999-12-31")
