"""Grader tests, including adversarial answers modelled on QuantLab's real failures."""

import pytest

from quantlab_env import questions
from quantlab_env.grader import extract_numbers, final_answer, grade
from quantlab_env.tools import Episode, execute

# JNJ restated FY2021 revenue in its 2024 10-K. Asked as of 2024-03-17 the
# answer uses the restated figure: 20,878 / 78,740 = 26.52%.
Q = questions.build("JNJ", "2022-01-02", "2024-03-17")
REV, NI = 78_740_000_000, 20_878_000_000

HONEST = (
    "Revenue for the fiscal year ending 2022-01-02 was $78,740 million and net income "
    "was $20,878 million, per the 10-K filings available on 2024-03-17. "
    "Net margin = 20,878 / 78,740 = 26.52%.\nANSWER: 26.52%"
)


def episode(*calls: tuple[str, dict]) -> Episode:
    ep = Episode()
    for name, args in calls:
        execute(name, args, ep)
    return ep


def lookup(concept: str, as_of: str = Q.as_of, ticker: str = "JNJ") -> tuple[str, dict]:
    return "get_fundamentals", {"ticker": ticker, "concept": concept, "as_of": as_of}


def full_episode() -> Episode:
    return episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": f"{NI} / {REV} * 100"}))


def test_gold_uses_restated_revenue():
    assert (Q.revenue, Q.net_income) == (REV, NI)
    assert Q.gold == pytest.approx(0.26515, abs=1e-5)


def test_honest_answer_scores_full_marks():
    g = grade(HONEST, full_episode().calls, Q)
    assert g.reward == 1.0, g.info
    assert g.subscores == {"answer": 1.0, "grounding": 1.0, "validity": 1.0, "efficiency": 1.0}


def test_answer_as_fraction_is_also_correct():
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": f"{NI} / {REV}"}))
    g = grade("Revenue $78.74 billion, net income $20.878 billion.\nANSWER: 0.2652", ep.calls, Q)
    assert g.reward == 1.0, g.info


# --- the DPO loophole: cite fewer numbers so there is less to check -----------


def test_dropping_cited_inputs_does_not_dodge_the_check():
    """QuantLab's DPO run learned citation restraint: fewer claims, fewer failures.
    A grader that only checks the numbers present rewards that. Here the answer
    must cite the inputs, so the terse answer loses the grounding points."""
    terse = "The net margin was 26.52%.\nANSWER: 26.52%"
    g = grade(terse, full_episode().calls, Q)
    assert g.subscores["answer"] == 1.0
    assert g.subscores["grounding"] == 0.0
    assert g.info["missing_inputs"] == ["revenue", "net_income"]
    assert g.reward < grade(HONEST, full_episode().calls, Q).reward


def test_dodging_is_not_better_than_citing_a_wrong_number():
    """Grounding is all or nothing, so removing a bad citation cannot raise the score."""
    one_bad = HONEST.replace("$78,740 million", "$78,740 million (up from $91,000 million)")
    terse = "The net margin was 26.52%.\nANSWER: 26.52%"
    bad, dodge = grade(one_bad, full_episode().calls, Q), grade(terse, full_episode().calls, Q)
    assert bad.info["ungrounded"] == ["$91,000 million"]
    assert bad.subscores["grounding"] == dodge.subscores["grounding"] == 0.0
    assert dodge.reward <= bad.reward


def test_right_margin_from_the_wrong_year_is_caught():
    """Apple's FY2022 and FY2023 margins are both 25.31%. An agent that looked up
    the wrong year gets the right number; the required-inputs check catches it."""
    q = questions.build("AAPL", "2022-09-24", "2022-11-27")
    ep = episode(
        lookup("revenue", "2023-12-03", "AAPL"),
        lookup("net_income", "2023-12-03", "AAPL"),
        ("compute", {"expr": "96995000000 / 383285000000 * 100"}),
    )
    wrong_year = "Revenue $383,285 million, net income $96,995 million, margin 25.31%.\nANSWER: 25.31%"
    g = grade(wrong_year, ep.calls, q)
    assert g.subscores["answer"] == 1.0
    assert g.subscores["grounding"] == 0.0
    assert g.info["ungrounded"] == []  # every number it cited is real evidence
    assert g.info["missing_inputs"] == ["revenue", "net_income"]
    assert g.reward == pytest.approx(0.7)


# --- the GRPO lesson: malformed calls are rejected and counted, never repaired ---


def test_hallucinated_key_is_rejected_not_dropped():
    """A lenient parser would drop the extra key and run the call. Here the call
    fails, is logged, and costs validity."""
    bad = ("get_fundamentals", {"ticker": "JNJ", "concept": "revenue", "as_of": Q.as_of, "fiscal_year": None})
    ep = episode(bad, lookup("revenue"), lookup("net_income"), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    assert ep.calls[0].result["error"]["code"] == "bad_argument"
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["validity"] == 0.75
    assert g.info["malformed_calls"] == 1
    assert g.reward == pytest.approx(0.95)


@pytest.mark.parametrize(
    "args",
    [
        {"ticker": "JNJ", "concept": "revenue", "as_of": 20240317},  # int, not coerced
        {"ticker": "JNJ", "concept": "revenue", "as_of": "17/03/2024"},  # not ISO
        {"ticker": "JNJ", "concept": "revenue", "as_of": "20240317"},  # compact ISO
        {"ticker": "JNJ", "concept": "revenue"},  # missing key
        {"ticker": ["JNJ"], "concept": "revenue", "as_of": "2024-03-17"},  # wrong type
    ],
)
def test_lenient_parses_are_malformed(args):
    ep = episode(("get_fundamentals", args))
    assert ep.calls[0].result["error"]["code"] == "bad_argument"
    assert grade("ANSWER: 26.52%", ep.calls, Q).subscores["validity"] == 0.0


def test_error_results_never_ground_numbers():
    """A failed call can echo its arguments. That echo is not evidence."""
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("get_fundamentals", {"ticker": "26.52", "concept": "revenue", "as_of": Q.as_of}),
    )
    assert "26.52" in str(ep.calls[-1].result)
    answer = "Revenue $78,740 million, net income $20,878 million.\nANSWER: 26.52%"
    g = grade(answer, ep.calls, Q)
    assert g.subscores["grounding"] == 0.0
    assert g.info["ungrounded"] == ["26.52%"]


# --- a correct number that appears nowhere in the evidence ----------------------


def test_correct_margin_computed_in_head_is_ungrounded():
    """QuantLab's research log: models fetched inputs, then did the arithmetic in
    their head. Right answer, unverifiable."""
    ep = episode(lookup("revenue"), lookup("net_income"))
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["answer"] == 1.0
    assert g.subscores["grounding"] == 0.0
    assert set(g.info["ungrounded"]) == {"26.52%"}
    assert g.reward == pytest.approx(0.7)


def test_memorised_number_laundered_through_compute_is_ungrounded():
    """Passing a remembered figure through the calculator does not make it evidence:
    a compute result counts only if every literal in it is already evidence."""
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": "26.52 * 1"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_unit_constants_in_compute_are_allowed():
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": "20878 / 78740 * 100"}))
    assert grade(HONEST, ep.calls, Q).subscores["grounding"] == 1.0


# --- a right answer with no evidence ---------------------------------------------


def test_right_answer_with_no_tool_calls_scores_zero():
    g = grade("I recall the margin was 26.52%.\nANSWER: 26.52%", [], Q)
    assert g.info["correct"] is True
    assert g.reward == 0.0


def test_right_answer_with_only_failed_calls_scores_no_answer_credit():
    ep = episode(("get_fundamentals", {"ticker": "JNJ", "concept": "sales", "as_of": Q.as_of}))
    g = grade("ANSWER: 26.52%", ep.calls, Q)
    assert g.subscores["answer"] == 0.0


# --- point in time -----------------------------------------------------------------


def test_pre_restatement_answer_is_wrong_and_ungrounded():
    """Looking up with an earlier as_of returns the original revenue (93,775m):
    the stale margin is wrong, and it does not cite the question's inputs."""
    early = "2022-03-19"
    ep = episode(
        lookup("revenue", early),
        lookup("net_income", early),
        ("compute", {"expr": "20878000000 / 93775000000 * 100"}),
    )
    stale = "Revenue $93,775 million, net income $20,878 million, margin 22.26%.\nANSWER: 22.26%"
    g = grade(stale, ep.calls, Q)
    assert g.subscores["answer"] == 0.0
    assert g.info["missing_inputs"] == ["revenue"]
    assert g.reward == pytest.approx(0.2)


# --- parsing -------------------------------------------------------------------------


def test_missing_answer_line_is_not_correct():
    g = grade(HONEST.replace("ANSWER: ", "So: "), full_episode().calls, Q)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 0.0


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("$78.74 billion", 78.74e9, "scaled"),
        ("78,740 million", 78_740e6, "scaled"),
        ("$20.9bn", 20.9e9, "scaled"),
        ("26.52%", 26.52, "percent"),
        ("(1,200)", -1200, "plain"),
        ("-0.5", -0.5, "plain"),
    ],
)
def test_number_parsing(text, value, kind):
    [cited] = extract_numbers(text)
    assert cited.value == pytest.approx(value)
    assert cited.kind == kind


def test_dates_fiscal_years_and_form_names_are_not_claims():
    text = "In FY2023 (fiscal year 2023, ending 2023-12-31) per its 10-K/A and 3 filings"
    assert [c for c in extract_numbers(text) if not c.exempt] == []


def test_final_answer_takes_last_answer_line():
    assert final_answer("ANSWER: 1%\nmore\nANSWER: 26.52%").value == 26.52
