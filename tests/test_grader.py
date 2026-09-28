"""Grader tests, including adversarial answers modelled on QuantLab's real failures."""

import pytest

from quantlab_env import questions
from quantlab_env.grader import admissible_lookup, build_evidence, extract_numbers, final_answer, grade
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
    q = questions.build("AAPL", "2023-09-30", "2023-12-03")
    ep = episode(
        lookup("revenue", q.as_of, "AAPL"),
        lookup("net_income", q.as_of, "AAPL"),
        ("compute", {"expr": "99803000000 / 394328000000 * 100"}),  # FY2022's rows, also returned
    )
    wrong_year = "Revenue $394,328 million, net income $99,803 million, margin 25.31%.\nANSWER: 25.31%"
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


ONE, FOUR, TWO = "20878 / 20878", "(20878 + 20878 + 20878 + 20878) / 20878", "(20878 + 20878) / 20878"
BUILT_FROM_ONES = f"100 / ({FOUR}) + {ONE} + (100 / ({TWO}) + {ONE} + {ONE}) / 100"


@pytest.mark.parametrize(
    "expr",
    [
        "100/(1+1+1+1)+1+(100/(1+1)+1+1)/100",  # 26.52 from unit constants alone
        "(100/(1+1+1+1)+1+(100/(1+1)+1+1)/100) * 20878 / 20878",  # evidence that cancels out
        "100/(1+1+1+1)+1+(100/(1+1)+1+1)/100 + 20878 / 78740 / 1e9",  # evidence that barely matters
        f"{BUILT_FROM_ONES}",  # 26.52 from x / x, so it never depends on x
        f"{BUILT_FROM_ONES} + 20878 / 78740 / 1e9",  # depends on evidence, but is not the margin
    ],
)
def test_margin_rebuilt_from_unit_constants_is_ungrounded(expr):
    """Codex review: unit constants are allowed literals, so `1+1+...` could spell
    any number. Constants may only scale evidence, the result must depend on a
    looked-up value, and the final answer must be net income over revenue."""
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": expr}))
    assert ep.calls[-1].ok
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["grounding"] == 0.0, g.info
    assert g.info["answer_derived"] is False
    assert g.reward == pytest.approx(0.7)


def test_margin_computed_in_steps_is_grounded():
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("compute", {"expr": "20878 / 78740"}),
        ("compute", {"expr": "0.26515 * 100"}),
    )
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["grounding"] == 1.0, g.info


def test_unit_constants_in_compute_are_allowed():
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": "20878 / 78740 * 100"}))
    assert grade(HONEST, ep.calls, Q).subscores["grounding"] == 1.0


def test_literal_binds_to_the_closest_evidence():
    """NVIDIA FY2022 revenue (26,914m) is within 0.5% of FY2023's (26,974m). The
    calculator literal must bind to the year it actually is."""
    q = questions.build("NVDA", "2023-01-29", "2023-03-26")
    ep = episode(
        lookup("revenue", q.as_of, "NVDA"),
        lookup("net_income", q.as_of, "NVDA"),
        ("compute", {"expr": "4368000000 / 26974000000 * 100"}),
    )
    answer = "Revenue $26,974 million, net income $4,368 million, margin 16.19%.\nANSWER: 16.19%"
    assert grade(answer, ep.calls, q).reward == 1.0


@pytest.mark.parametrize(
    "exprs",
    [
        ["20878 * 1.3", "27141.4 / 78740 * 100"],  # invented factor, then used downstream
        [f"{NI} / {REV} * 100 + 0.004"],  # evidence plus an invented offset
        [f"{NI} / {REV}", "0.26515 * 100 * 1.02"],  # a clean step, then an invented factor
        ["20878 / 78740 * 100", "26.515 - 26.515 + 30"],  # evidence cancelled, an invented value added
        ["20878 / 20878", "26.52 * 1"],  # a constant from x / x does not ground a memorised literal
    ],
)
def test_tainted_calculation_chains_are_ungrounded(exprs):
    """Mixing evidence with an invented number anywhere in a chain taints every result after it."""
    ep = episode(lookup("revenue"), lookup("net_income"), *[("compute", {"expr": e}) for e in exprs])
    assert all(c.ok for c in ep.calls)
    last = ep.calls[-1].result["value"]
    assert last not in build_evidence(ep.calls, Q.as_of).values
    g = grade(HONEST.replace("26.52%", f"{last:.2f}%"), ep.calls, Q)
    assert g.subscores["grounding"] == 0.0, g.info


def test_invented_input_within_tolerance_is_not_evidence():
    """Second review: 20,885 is within 0.5% of the real 20,878, so the invented
    figure passed as evidence, both as a calculator literal and as a citation,
    and the tainted margin was rebound to the real inputs (0.975). A number now
    has to equal evidence at the precision it is written with."""
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("compute", {"expr": "20878+7"}),
        ("compute", {"expr": "20885/78740*100"}),
    )
    assert all(c.ok for c in ep.calls)
    # neither compute result is evidence
    assert build_evidence(ep.calls, Q.as_of).values == build_evidence(ep.calls[:2], Q.as_of).values
    g = grade("Revenue $78.74 billion; net income $20.885 billion.\nANSWER: 26.52%", ep.calls, Q)
    assert g.info["answer_derived"] is False
    assert g.info["ungrounded"] == ["$20.885 billion", "26.52%"]
    assert g.info["missing_inputs"] == ["net_income"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.675)


@pytest.mark.parametrize(
    "expr",
    [
        "20885 / 78740 * 100",  # within 0.5% of 20,878
        "20880 / 78740 * 100",  # an integer literal is exact
        "20878 / 78745 * 100",
        "0x518E / 78740 * 100",  # 20878 in hex: not a decimal amount
    ],
)
def test_calculator_literals_must_equal_evidence(expr):
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": expr}))
    last = ep.calls[-1].result["value"]
    assert last not in build_evidence(ep.calls, Q.as_of).values
    assert grade(HONEST, ep.calls, Q).info["answer_derived"] is False


@pytest.mark.parametrize(
    ("net_income", "grounded"),
    [
        ("$20.878 billion", True),
        ("$20.88 billion", True),  # a rounding of 20.878
        ("$20.9 billion", True),
        ("$21 billion", True),
        ("20,878 million", True),
        ("$20.885 billion", False),  # within 0.5%, but not a rounding of 20.878
        ("$20.87 billion", False),
        ("20,880 million", False),
        ("$20,900,000,000", False),
    ],
)
def test_cited_numbers_match_at_their_displayed_precision(net_income, grounded):
    answer = f"Revenue $78.74 billion, net income {net_income}. Net margin 26.52%.\nANSWER: 26.52%"
    g = grade(answer, full_episode().calls, Q)
    assert (g.info["ungrounded"] == []) is grounded, g.info
    assert g.subscores["grounding"] == float(grounded)


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


# --- false claims: sign and scale are part of the claim -----------------------


@pytest.mark.parametrize(
    ("text", "ungrounded"),
    [
        ("Revenue $78.74 billion, net income -$20.878 billion.", ["-$20.878 billion"]),
        ("Revenue $78.74 billion, net income was -20,878 million.", ["-20,878 million"]),
        ("Revenue $78.74, net income $20.878.", ["$78.74", "$20.878"]),
    ],
)
def test_wrong_sign_or_missing_scale_is_ungrounded(text, ungrounded):
    """Codex review: magnitudes were compared without sign, and a bare "$78.74"
    could stand for 78.74 billion. A loss is not a profit, and a dollar amount
    means what it says."""
    g = grade(f"{text} Net margin 20,878 / 78,740 = 26.52%.\nANSWER: 26.52%", full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("net income - $20.878 billion", [-20.878e9]),
        ("net income −  $20.878 billion", [-20.878e9]),
        ("net income: -  20,878 million", [-20_878e6]),
        ("net income (- 20,878 million)", [-20_878e6]),
        ("net income = - 20,878 million", [-20_878e6]),
        ("net income $ - 20.878 billion", [-20.878e9]),
        ("fiscal 2021 - 2022", [2021, 2022]),  # a range
        ("20,878 - 1,000 million", [20_878, 1_000e6]),  # subtraction
        ("the 10-K - 2 filings", [2]),  # a hyphen after a form name
        ("- $78.74 billion revenue", [78.74e9]),  # a list bullet
        ("  - 20,878 million net income", [20_878e6]),  # an indented bullet
        ("pre-2022 filings, re- 20 amended", [2022, 20]),  # word hyphens
    ],
)
def test_spaced_minus_parsing(text, values):
    assert [c.value for c in extract_numbers(text)] == pytest.approx(values)


@pytest.mark.parametrize(
    "inputs",
    [
        "Revenue $78.74 billion, net income - $20.878 billion.",
        "Revenue $78.74 billion, net income −  $20.878 billion.",
        "Revenue $78.74 billion, net income: -  20,878 million.",
    ],
)
def test_spaced_negative_input_is_not_grounded(inputs):
    """Second review: a minus separated from the amount by spaces was dropped, so
    a claimed loss of 20,878 million got full grounding."""
    g = grade(f"{inputs} Net margin 26.52%.\nANSWER: 26.52%", full_episode().calls, Q)
    assert len(g.info["ungrounded"]) == 1 and g.info["ungrounded"][0].startswith(("-", "−"))
    assert g.info["missing_inputs"] == ["net_income"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize(
    "inputs",
    [
        "Revenue $78.74 billion, net income $20.878 billion.",
        "Revenue 78,740 million, net income 20,878 million.",
        "Revenue $78.74B, net income $20.878B.",
        "Revenue $78,740,000,000, net income $20,878,000,000.",
        "Revenue USD 78.74 billion, net income USD 20.878 billion.",
        "Revenue 78,740, net income 20,878 (USD millions).",
        "- Revenue: $78.74 billion\n- $20.878 billion net income\n",
    ],
)
def test_legitimate_amount_forms_are_grounded(inputs):
    g = grade(f"{inputs} Net margin 26.52%.\nANSWER: 26.52%", full_episode().calls, Q)
    assert g.reward == 1.0, g.info


@pytest.mark.parametrize(
    ("inputs", "missing"),
    [
        ("Revenue 78,740,000,000%, net income 20,878,000,000%.", ["revenue", "net_income"]),
        ("Revenue 78740000000 percent, net income $20.878 billion.", ["revenue"]),
        ("Revenue $78.74 billion, net income 20,878,000,000%.", ["net_income"]),
        ("Revenue 100%, net income 26.52%.", ["revenue", "net_income"]),
    ],
)
def test_a_percent_never_cites_an_input(inputs, missing):
    """Second review: a percent reads as either a fraction or the number itself, so
    "net income 20,878,000,000%" matched the net income and satisfied the input
    requirement at 1.0. Inputs must be cited as amounts."""
    g = grade(f"{inputs} Net margin 26.52%.\nANSWER: 26.52%", full_episode().calls, Q)
    assert g.info["missing_inputs"] == missing
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_lookups_past_the_question_date_are_not_evidence():
    """Codex review: a lookup as of 2099-01-01 sees filings the question rules out.
    It is well formed, so it is not malformed, but it is neither evidence nor a
    successful lookup, and it still costs an efficiency call."""
    future = "2099-01-01"
    ep = episode(lookup("revenue", future), lookup("net_income", future), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.info["future_lookups"] == 2
    assert g.subscores == {"answer": 0.0, "grounding": 0.0, "validity": 1.0, "efficiency": 0.0}
    assert g.reward == pytest.approx(0.1)


def test_one_future_input_breaks_grounding():
    ep = episode(lookup("revenue"), lookup("net_income", "2099-01-01"), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["answer"] == 1.0
    assert g.subscores["grounding"] == 0.0
    assert "$20,878 million" in g.info["ungrounded"]
    assert g.reward == pytest.approx(0.7)


def test_lookup_on_the_question_date_is_evidence():
    assert admissible_lookup(full_episode().calls[0], Q.as_of)
    assert not admissible_lookup(episode(lookup("revenue", "2024-03-18")).calls[0], Q.as_of)


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
        ("$78.74B", 78.74e9, "scaled"),
        ("USD 78.74 billion", 78.74e9, "scaled"),
        ("$78.74", 78.74, "currency"),
        ("-$20.878 billion", -20.878e9, "scaled"),
        ("$-20.878 billion", -20.878e9, "scaled"),
        ("(1,200)", 1200, "plain"),  # parentheses are prose, not an accounting negative
        ("-0.5", -0.5, "plain"),
    ],
)
def test_number_parsing(text, value, kind):
    [cited] = extract_numbers(text)
    assert cited.value == pytest.approx(value)
    assert cited.kind == kind


@pytest.mark.parametrize(
    "text",
    [
        "In FY2023 (fiscal year 2023, ending 2023-12-31) per its 10-K/A and 3 filings",
        "Filed on March 17, 2024, 17 Mar 2024 or 03/17/2024, for Q4 of fiscal 2021.",
        "1. Look up revenue.\n2) Look up net income.\n(3) Compute.\n- 4. Answer.",
    ],
)
def test_dates_years_counts_and_list_markers_are_not_claims(text):
    assert [c.text for c in extract_numbers(text) if not c.exempt] == []


@pytest.mark.parametrize(
    ("extra", "ungrounded"),
    [
        (" Debt $7, assets $2023.", ["$7", "$2023"]),
        (" Debt 7, assets 2,023.", ["7", "2,023"]),
        (" Debt was 7.", ["7"]),
        (" Cash 2023 dollars, 3%.", ["2023 dollars", "3%"]),
    ],
)
def test_small_or_year_like_amounts_are_claims(extra, ungrounded):
    """Codex review: small integers and year-like values were exempt by value, so
    "debt $7, assets $2023" rode along with an honest answer at 1.0. Only
    context exempts a number: a year or count with no currency or unit, a date,
    a list marker."""
    g = grade(HONEST.replace("\nANSWER", f"{extra}\nANSWER"), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_final_answer_takes_last_answer_line():
    assert final_answer("ANSWER: 1%\nmore\nANSWER: 26.52%").value == 26.52


@pytest.mark.parametrize(("line", "value"), [("ANSWER: -3.1%", -3.1), ("ANSWER: - 3.1%", -3.1), ("ANSWER:  3.1%", 3.1)])
def test_final_answer_keeps_its_sign(line, value):
    assert final_answer(line).value == value
