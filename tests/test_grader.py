"""Grader tests, including adversarial answers modelled on QuantLab's real failures."""

import pytest

from demo import honest_answer, honest_expr, honest_exprs, honest_pick
from quantlab_env import questions
from quantlab_env.grader import (
    admissible_lookup,
    build_evidence,
    chosen,
    citations,
    extract_numbers,
    final_answer,
    grade,
)
from quantlab_env.tools import Episode, execute

# JNJ restated FY2021 revenue in its 2024 10-K. Asked as of 2024-03-17 the
# answer uses the restated figure: 20,878 / 78,740 = 26.52%.
Q = questions.build("JNJ", "2022-01-02", "2024-03-17")
REV, NI = 78_740_000_000, 20_878_000_000
BLOCK = f"REVENUE: {REV}\nNET_INCOME: {NI}\n"
PROSE = (
    "Revenue for the fiscal year ending 2022-01-02 was $78,740 million and net income "
    "was $20,878 million, per the 10-K filings available on 2024-03-17. "
    "Net margin = 20,878 / 78,740 = 26.52%."
)
HONEST = f"{PROSE}\n{BLOCK}ANSWER: 26.52%"


def answer(prose: str, margin: str = "26.52%", block: str = BLOCK) -> str:
    return f"{prose}\n{block}ANSWER: {margin}"


def honest(extra: str) -> str:
    """The honest answer with an extra sentence in its prose."""
    return answer(PROSE + extra)


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
    g = grade(answer("Revenue $78.74 billion, net income $20.878 billion.", "0.2652"), ep.calls, Q)
    assert g.reward == 1.0, g.info


def test_prompt_states_the_citation_block():
    assert Q.prompt.endswith("REVENUE: <revenue value>\nNET_INCOME: <net income value>\nANSWER: <margin>%")


# --- the citation block: inputs are cited by label, exactly as the tool returned them ---


def test_honest_prose_without_the_block_is_ungrounded():
    g = grade(f"{PROSE}\nANSWER: 26.52%", full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue", "net_income"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize(
    "block",
    [
        "REVENUE: 78740000000\nNET_INCOME: 20878000000\n",
        "REVENUE: $78740000000\nNET_INCOME: $20878000000\n",
        "REVENUE: 78,740,000,000\nNET_INCOME: $20,878,000,000\n",
        "  REVENUE :  $ 78,740,000,000  \nNET_INCOME:20878000000\n",
        "NET_INCOME: 20878000000\nREVENUE: 78740000000\n",  # order does not matter
        "REVENUE: 1\nNET_INCOME: 2\nREVENUE: 78740000000\nNET_INCOME: 20878000000\n",  # the last line wins
    ],
)
def test_citation_block_accepts_formatting_only(block):
    g = grade(answer(PROSE, block=block), full_episode().calls, Q)
    assert g.info["citations"] == {"revenue": REV, "net_income": NI}
    assert g.reward == 1.0, g.info


@pytest.mark.parametrize(
    "revenue",
    [
        "$78.74 billion",  # a rounding
        "$78,740 million",  # a scale word
        "78740000000.0",
        "78740000001",
        "7.874e10",
        "78,74,00,00,000",  # commas that are not thousands separators
        "-78740000000",  # the wrong sign
        "$(78,740,000,000)",
        "about 78740000000",
        "78740000000 USD",
        "93775000000",  # the pre-restatement value, not what the gold used
        "",
    ],
)
def test_citation_block_rejects_anything_but_the_exact_value(revenue):
    g = grade(answer(PROSE, block=f"REVENUE: {revenue}\nNET_INCOME: {NI}\n"), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_swapped_labels_are_ungrounded():
    """Review case E: inputs were matched by value only, so revenue and net income
    cited under each other's labels scored 1.0. Each label must match a lookup of
    its own concept."""
    g = grade(answer(PROSE, block=f"REVENUE: {NI}\nNET_INCOME: {REV}\n"), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue", "net_income"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_block_value_must_come_from_a_lookup_in_this_episode():
    """Right value, right label, but net income was never looked up."""
    ep = episode(lookup("revenue"), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.info["missing_inputs"] == ["net_income"]
    assert g.subscores["grounding"] == 0.0


def test_parenthesised_input_in_the_block_is_ungrounded():
    """Review case E: `$(20,878) million` in the block is not a tool value."""
    g = grade(answer(PROSE, block=f"REVENUE: {REV}\nNET_INCOME: $(20,878) million\n"), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["net_income"]
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize("prose", ["Net income was $(20,878) million.", "The net loss was $20,878 million."])
def test_sign_in_parentheses_or_words_is_a_known_limit(prose):
    """Review case E, prose: parentheses are read as prose and a sign given in
    words ("net loss") is not parsed, so with a correct block these still score
    1.0. The block itself cannot carry them (see the tests above)."""
    assert grade(answer(prose), full_episode().calls, Q).reward == 1.0


def test_citations_parse():
    text = "REVENUE: $78,740,000,000\nNET_INCOME: -20878000000\nANSWER: 1%"
    assert citations(text) == {"revenue": REV, "net_income": -NI}
    assert citations("Revenue: 78740000000\nnet_income: 20878000000") == {"revenue": REV, "net_income": NI}
    assert citations("Sales: 78740000000\nNet income: 20878000000") == {"revenue": None, "net_income": None}


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
    the wrong year gets the right number; the citation block catches it."""
    q = questions.build("AAPL", "2023-09-30", "2023-12-03")
    ep = episode(
        lookup("revenue", q.as_of, "AAPL"),
        lookup("net_income", q.as_of, "AAPL"),
        ("compute", {"expr": "99803000000 / 394328000000 * 100"}),  # FY2022's rows, also returned
    )
    wrong_year = answer(
        "Revenue $394,328 million, net income $99,803 million, margin 25.31%.",
        "25.31%",
        "REVENUE: 394328000000\nNET_INCOME: 99803000000\n",
    )
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
    g = grade("ANSWER: 26.52%", ep.calls, Q)
    assert g.info["malformed_calls"] == 1
    assert g.subscores["validity"] == 0.0


def test_error_results_never_ground_numbers():
    """A failed call can echo its arguments. That echo is not evidence."""
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("get_fundamentals", {"ticker": "26.52", "concept": "revenue", "as_of": Q.as_of}),
    )
    assert "26.52" in str(ep.calls[-1].result)
    g = grade(answer("Revenue $78,740 million, net income $20,878 million."), ep.calls, Q)
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


SEVEN = "+".join(["20878/20878"] * 7)


def test_invented_amount_from_a_constant_plus_a_small_evidence_term_is_ungrounded():
    """Third review, case B: each side of `+` used evidence, but `x/x` is a
    constant, and the whole formula depended on evidence only through a tiny
    NI/REV term. The result, 7,000,265,151, then grounded an invented "$7
    billion" (0.975). Each side of `+` and `-` must now depend on a looked-up value."""
    expr = f"({SEVEN})*1e9 + 20878/78740*1e6"
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("compute", {"expr": expr}),
        ("compute", {"expr": f"{NI} / {REV} * 100"}),
    )
    assert ep.calls[2].result["value"] == pytest.approx(7_000_265_151, abs=1)
    assert ep.calls[2].result["value"] not in build_evidence(ep.calls, Q.as_of).values
    g = grade(honest(" Total debt was $7 billion."), ep.calls, Q)
    assert g.info["ungrounded"] == ["$7 billion"]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.675)


def test_margin_computed_in_steps_is_grounded():
    ep = episode(
        lookup("revenue"),
        lookup("net_income"),
        ("compute", {"expr": "20878 / 78740"}),
        ("compute", {"expr": "0.26515 * 100"}),
    )
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["grounding"] == 1.0, g.info


def test_sums_of_looked_up_values_are_evidence():
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": "78740 - 20878"}))
    assert 57_862 in build_evidence(ep.calls, Q.as_of).values


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
    text = answer(
        "Revenue $26,974 million, net income $4,368 million, margin 16.19%.",
        "16.19%",
        "REVENUE: 26974000000\nNET_INCOME: 4368000000\n",
    )
    assert grade(text, ep.calls, q).reward == 1.0


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
    g = grade(answer("Revenue $78.74 billion; net income $20.885 billion."), ep.calls, Q)
    assert g.info["answer_derived"] is False
    assert g.info["ungrounded"] == ["$20.885 billion", "26.52%"]
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
    g = grade(answer(f"Revenue $78.74 billion, net income {net_income}. Net margin 26.52%."), full_episode().calls, Q)
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


@pytest.mark.parametrize(
    "calls",
    [
        [("compute", {"expr": "26.52 * 1"})],  # a well-formed call, but no lookup
        [lookup("revenue", "2000-01-01")],  # a well-formed lookup that found nothing (no_data)
    ],
)
def test_right_answer_with_no_successful_lookup_scores_zero(calls):
    """Third review: validity was the share of well-formed calls, so a
    well-formed call that looked nothing up still earned 0.10 for a right answer.
    Validity is now gated like efficiency, on an admissible successful lookup."""
    ep = episode(*calls)
    assert not any(c.result.get("error", {}).get("code") in ("bad_argument", "unknown_tool") for c in ep.calls)
    g = grade("ANSWER: 26.52%", ep.calls, Q)
    assert g.info["correct"] is True
    assert g.subscores == {"answer": 0.0, "grounding": 0.0, "validity": 0.0, "efficiency": 0.0}
    assert g.reward == 0.0


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
    stale = answer(
        "Revenue $93,775 million, net income $20,878 million, margin 22.26%.",
        "22.26%",
        f"REVENUE: 93775000000\nNET_INCOME: {NI}\n",
    )
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
    g = grade(answer(f"{text} Net margin 20,878 / 78,740 = 26.52%."), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


DASHES = ["-", "‐", "‑", "‒", "–", "—", "―", "−", "﹣", "－"]


MINUSES = ["-", "−", "﹣", "－"]  # the dashes that are a sign even when spaced off
PUNCTUATION_DASHES = [d for d in DASHES if d not in MINUSES]


@pytest.mark.parametrize(
    ("dash", "template"),
    [(d, "net income was {d}$20,878 million") for d in DASHES]
    + [(d, "net income {d} $20,878 million") for d in MINUSES],
    ids=[f"attached-U+{ord(d):04X}" for d in DASHES] + [f"spaced-U+{ord(d):04X}" for d in MINUSES],
)
def test_unicode_dash_minus_is_a_sign(dash, template):
    """Third review, case A: only `-` and U+2212 were read as minus signs, so
    `net income was –$20,878 million` with an en dash (or any other dash)
    claimed a loss and still scored 1.0. Every dash attached to an amount is
    now a minus, and so is a spaced-off hyphen-minus or true minus."""
    claim = template.format(d=dash)
    g = grade(answer(f"Revenue $78,740 million, {claim}. Net margin 26.52%."), full_episode().calls, Q)
    assert len(g.info["ungrounded"]) == 1 and g.info["ungrounded"][0][0] == dash, g.info
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize("dash", PUNCTUATION_DASHES, ids=[f"U+{ord(d):04X}" for d in PUNCTUATION_DASHES])
def test_spaced_en_or_em_dash_is_punctuation_not_a_sign(dash):
    """Models write "Revenue — $78.74 billion" as punctuation. Reading it as a
    minus would fail an honest answer's grounding; the inputs that matter are
    checked exactly in the citation block, and a sign put in prose is a known
    limit either way."""
    g = grade(answer(f"Revenue {dash} $78.74 billion, net income {dash} $20.878 billion."), full_episode().calls, Q)
    assert g.info["ungrounded"] == [], g.info
    assert g.reward == pytest.approx(1.0)


@pytest.mark.parametrize("dash", DASHES, ids=[f"U+{ord(d):04X}" for d in DASHES])
def test_unicode_dashes_parse_like_hyphens(dash):
    assert [c.value for c in extract_numbers(f"fiscal 2021{dash}2022")] == [2021, 2022]  # a range
    assert [c.value for c in extract_numbers(f"income: {dash}20,878 million")] == [-20_878e6]
    assert final_answer(f"ANSWER: {dash}3.1%").value == -3.1


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
        ("– $78.74 billion revenue", [78.74e9]),  # an en dash bullet
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
    g = grade(answer(f"{inputs} Net margin 26.52%."), full_episode().calls, Q)
    assert len(g.info["ungrounded"]) == 1 and g.info["ungrounded"][0].startswith(("-", "−"))
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
        "Revenue 78.74 billions, net income 20,878 mln, 20.878 bln, 20.9 bil, 20,878 mn.",
        "Revenue 78,740 millions, net income 20,878 millions, a 26.52 pct margin.",
        "- Revenue: $78.74 billion\n- $20.878 billion net income\n",
    ],
)
def test_legitimate_amount_forms_are_grounded(inputs):
    g = grade(answer(f"{inputs} Net margin 26.52%."), full_episode().calls, Q)
    assert g.reward == 1.0, g.info


@pytest.mark.parametrize(
    ("inputs", "ungrounded"),
    [
        ("Revenue 78,740,000,000%, net income 20,878,000,000%.", ["78,740,000,000%", "20,878,000,000%"]),
        ("Revenue 78740000000 percent, net income $20.878 billion.", ["78740000000 percent"]),
        ("Revenue $78.74 billion, net income 20,878,000,000%.", ["20,878,000,000%"]),
        ("Revenue 100%, net income 26.52%.", ["100%"]),
    ],
)
def test_a_percent_never_cites_an_amount(inputs, ungrounded):
    """Second review: a percent reads as either a fraction or the number itself, so
    "net income 20,878,000,000%" matched the net income. Inputs are now cited in
    the block, and in prose a percent only matches a calculator result, never a
    looked-up amount."""
    g = grade(answer(f"{inputs} Net margin 26.52%."), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_lookups_past_the_question_date_are_not_evidence():
    """Codex review: a lookup as of 2099-01-01 sees filings the question rules out.
    It is well formed, so it is not malformed, but it is neither evidence nor a
    successful lookup, and with no admissible lookup validity and efficiency are 0
    (validity was 1.0 here, for 0.10, before the third review)."""
    future = "2099-01-01"
    ep = episode(lookup("revenue", future), lookup("net_income", future), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.info["future_lookups"] == 2
    assert g.info["malformed_calls"] == 0
    assert g.subscores == {"answer": 0.0, "grounding": 0.0, "validity": 0.0, "efficiency": 0.0}
    assert g.reward == 0.0


def test_one_future_input_breaks_grounding():
    ep = episode(lookup("revenue"), lookup("net_income", "2099-01-01"), ("compute", {"expr": f"{NI} / {REV} * 100"}))
    g = grade(HONEST, ep.calls, Q)
    assert g.subscores["answer"] == 1.0
    assert g.subscores["grounding"] == 0.0
    assert "$20,878 million" in g.info["ungrounded"]
    assert g.info["missing_inputs"] == ["net_income"]
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
        ("7 billions", 7e9, "scaled"),
        ("5 millions", 5e6, "scaled"),
        ("2 thousands", 2e3, "scaled"),
        ("1 trillions", 1e12, "scaled"),
        ("9 mln", 9e6, "scaled"),
        ("9 mn", 9e6, "scaled"),
        ("5 bln", 5e9, "scaled"),
        ("8 bil", 8e9, "scaled"),
        ("9 pct", 9, "percent"),
        ("9 per cent", 9, "percent"),
        ("9 percent", 9, "percent"),
        ("7 dollar", 7, "currency"),
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
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("extra", "ungrounded"),
    [
        (" Debt of 7 billions.", "7 billions"),
        (" Debt 9 mln.", "9 mln"),
        (" A loss of 8 bil.", "8 bil"),
        (" Debt 5 millions.", "5 millions"),
        (" Assets 2050 millions.", "2050 millions"),
        (" Margin guidance 9 pct.", "9 pct"),
        (" Margin guidance 9 per cent.", "9 per cent"),
        (" Debt 7 hundred million.", "7"),
        (" Debt 2050 hundred.", "2050"),
    ],
)
def test_scale_words_are_units_not_counts(extra, ungrounded):
    """Third review, case C: plural and abbreviated scale words were not units, so
    "7 billions" read as a small count and "2050 millions" as a year, both exempt,
    and the answer scored 1.0. They are units now, and a count or year followed
    by a scale or currency word ("7 hundred million") is a claim."""
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == [ungrounded]
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize("extra", [" Debt was $0 trillion.", " Debt was $0.", " Debt was 0 billion."])
def test_a_cited_zero_matches_only_zero(extra):
    """Third review, case D: "$0 trillion" rounds from anything under $0.5
    trillion, so it matched every looked-up amount. A zero is a claim of zero."""
    g = grade(honest(extra), full_episode().calls, Q)
    assert len(g.info["ungrounded"]) == 1
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


def test_final_answer_takes_last_answer_line():
    assert final_answer("ANSWER: 1%\nmore\nANSWER: 26.52%").value == 26.52


@pytest.mark.parametrize(("line", "value"), [("ANSWER: -3.1%", -3.1), ("ANSWER: - 3.1%", -3.1), ("ANSWER:  3.1%", 3.1)])
def test_final_answer_keeps_its_sign(line, value):
    assert final_answer(line).value == value


# --- a date never swallows an amount -------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "ungrounded"),
    [
        (" Debt of $9 may be refinanced soon.", ["$9"]),
        (" Debt of USD 9 may be refinanced soon.", ["USD 9"]),
        (" Debt of $  9 may be refinanced soon.", ["$  9"]),
        (" Cash at June 30 billion.", ["30 billion"]),
        (" Losses may 9 billion.", ["9 billion"]),
        (" Cash at June 30.5 billion.", ["30.5 billion"]),
        (" Cash at 30 June 2023 billion.", ["2023 billion"]),
        (" Cash at June 30 b.", ["30 b"]),
    ],
)
def test_a_date_form_does_not_exempt_an_amount(extra, ungrounded):
    """Fourth review: `$9 may` read as the date "9 May" and `June 30 billion` as
    "June 30", so both amounts were exempt and scored 1.0. A currency sign before
    or a unit word after makes the number an amount, never a date."""
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


@pytest.mark.parametrize(
    "extra",
    [" Filed June 30, 2023.", " Filed 30 June 2023.", " Filed on June 30.", " Filed on 30 June.", " Due May 9, 2024."],
)
def test_a_genuine_date_is_still_exempt(extra):
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == []
    assert g.reward == 1.0, g.info


# --- a unit constant in a written formula ------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        " Net margin = 20878000000 / 78740000000 × 100 = 26.515113%.",
        " Net margin = (20878000000 / 78740000000) * 100 = 26.52%.",
        " Net margin = (20878000000/78740000000)×100 = 26.52%.",
        " Net margin = 20878000000 / 78740000000 \\times 100 \\approx 26.52%.",
        " Net Margin = (Net Income / Revenue) * 100%.",
        " To express it as a percent, I multiply by 100 and round.",
    ],
)
def test_a_unit_constant_after_an_operator_is_not_a_claim(extra):
    """First real-model eval (2026-10-05): honest Qwen3.5-9B answers that showed
    their working ("20,878 / 78,740 × 100 = 26.52%") scored 0.70, because the prose check
    read the 100 as an invented amount (34 of 180 episodes). The constant 100
    written right after a multiply or divide operator is part of a formula,
    as it already is inside `compute`, so it claims nothing. Only 100 is exempt."""
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == []
    assert g.reward == 1.0, g.info


@pytest.mark.parametrize(
    ("extra", "ungrounded"),
    [
        (" Debt was 100.", ["100"]),
        (" Debt is 26.52% × 100 billion.", ["100 billion"]),
        (" Debt is 26.52% × $100.", ["$100"]),
        (" Debt is 26.52% × 7.", ["7"]),
        (" Debt is 26.52% × -100.", ["-100"]),
        (" Debt is 26.52% × 1,000,000,000.", ["1,000,000,000"]),
        (" Debt is 26.52% × 1000.", ["1000"]),
    ],
)
def test_an_operator_does_not_exempt_an_amount(extra, ungrounded):
    """Only a bare 100 is exempt after an operator: with a scale word, a currency
    sign or a sign, or as any other value (even a unit constant such as 1,000,000,000,
    which could pass for an amount), it is still a claim."""
    g = grade(honest(extra), full_episode().calls, Q)
    assert g.info["ungrounded"] == ungrounded
    assert g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.7)


# --- markdown around the block labels ----------------------------------------------------


@pytest.mark.parametrize(
    "tail",
    [
        f"**REVENUE:** {REV:,}\n**NET_INCOME:** {NI:,}\n**ANSWER:** 26.52%",
        f"**REVENUE**: {REV:,}\n**NET_INCOME**: {NI:,}\n**ANSWER**: 26.52%",
        f"__REVENUE__: {REV}\n__NET_INCOME:__ {NI}\n__ANSWER:__ 26.52%",
        f"**REVENUE: {REV:,}**\n**NET_INCOME: ${NI:,}**\n**ANSWER: 26.52%**",
        f"REVENUE: **{REV:,}**\nNET_INCOME: *{NI}*\nANSWER: *26.52%*",
        f"- REVENUE: {REV}\n- NET_INCOME: {NI}\n- ANSWER: 26.52%",
        f"* REVENUE: {REV}\n* NET_INCOME: {NI}\n* ANSWER: 26.52%",
        f"+ REVENUE: {REV}\n+ NET_INCOME: {NI}\n+ ANSWER: 26.52%",
        f"> REVENUE: {REV}\n> NET_INCOME: {NI}\n> ANSWER: 26.52%",
        f"> - **REVENUE:** {REV}\n> - **NET_INCOME:** {NI}\n> - **ANSWER:** 26.52%",
        f"### REVENUE: {REV}\n### NET_INCOME: {NI}\n### ANSWER: 26.52%",
        f"revenue: {REV}\nnet_income: {NI}\nanswer: 26.52%",
        f"Revenue: {REV}\nNet_Income: {NI}\nAnswer: 26.52%",
    ],
)
def test_markdown_and_case_around_block_labels_are_accepted(tail):
    """Fourth review: `**REVENUE:** 78,740,000,000` failed an honest model (0.70),
    and `**ANSWER:** 26.52%` found no answer at all. Labels are case-insensitive
    (like ANSWER always was) and may carry markdown list, quote, heading and
    emphasis markers; the value must still be the exact tool value."""
    g = grade(f"{PROSE}\n{tail}", full_episode().calls, Q)
    assert g.info["citations"] == {"revenue": REV, "net_income": NI}
    assert g.info["final_answer"] is not None
    assert g.reward == 1.0, g.info


@pytest.mark.parametrize(
    "revenue",
    [
        "**78740000000** million",
        "78740000000 ** 1",
        "78_740_000_000",
        "**$78.74 billion**",
        "`78740000000`",
        "~~78740000000~~",
    ],
)
def test_markdown_does_not_let_anything_else_into_a_block_value(revenue):
    g = grade(answer(PROSE, block=f"**REVENUE:** {revenue}\nNET_INCOME: {NI}\n"), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue"]
    assert g.reward == pytest.approx(0.7)


def test_markdown_swapped_labels_are_still_ungrounded():
    block = f"**REVENUE:** {NI}\n**NET_INCOME:** {REV}\n"
    g = grade(answer(PROSE, block=block), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue", "net_income"]
    assert g.reward == pytest.approx(0.7)


def test_markdown_final_answer_takes_last_answer_line():
    assert final_answer("**ANSWER:** 1%\nmore\n> **answer**: 26.52%").value == 26.52
    assert final_answer("The ANSWER: 26.52%") is None  # a label opens its line


# --- block values are ASCII digits -------------------------------------------------------


@pytest.mark.parametrize(
    "revenue",
    [
        "７８,７４０,０００,０００",  # fullwidth digits
        "٧٨٧٤٠٠٠٠٠٠٠",  # Arabic-Indic digits
    ],
)
def test_block_values_must_be_ascii_digits(revenue):
    g = grade(answer(PROSE, block=f"REVENUE: {revenue}\nNET_INCOME: {NI}\n"), full_episode().calls, Q)
    assert g.info["missing_inputs"] == ["revenue"]
    assert g.reward == pytest.approx(0.7)


# --- the probe set: generalised inputs, labels, formulas and par calls ---------


def solve(q: questions.Question, expr: str | None = None, extra: tuple = ()) -> tuple[str, list]:
    """The honest strategy on any question: one lookup per (ticker, concept), one calculation."""
    calls = [lookup(c, q.as_of, t) for t, c in dict.fromkeys((i.ticker, i.concept) for i in q.inputs)]
    ep = episode(*extra, *calls, ("compute", {"expr": expr or honest_expr(q)}))
    return honest_answer(q, ep.calls[-1].result["value"]), ep.calls


@pytest.mark.parametrize("q", questions.probe_questions(), ids=lambda q: f"{q.kind}-{q.ticker}-{q.as_of}")
def test_honest_agent_scores_full_marks_on_every_probe_question(q):
    text, calls = solve(q)
    g = grade(text, calls, q)
    assert g.reward == 1.0, g.info


def test_generic_par_calls_and_labels():
    change = questions.net_margin_change("JNJ", "2022", "2024-03-17")
    assert change.par_calls == 3 and Q.par_calls == 3
    assert questions.revenue_growth("JNJ", "2022", "2024-03-17").par_calls == 2
    gap = questions.net_margin_gap("NVDA", "AAPL", "2024", "2024-12-01")
    assert gap.par_calls == 5
    assert list(gap.labels.values()) == ["NVDA_REVENUE", "NVDA_NET_INCOME", "AAPL_REVENUE", "AAPL_NET_INCOME"]
    assert gap.prompt.endswith("AAPL_NET_INCOME: <AAPL net income value>\nANSWER: <difference> pp")


def test_fiscal_year_labels():
    """NVIDIA's fiscal 2024 ends in January 2024; JNJ's fiscal 2021 ends on 2022-01-02."""
    assert questions.fiscal_year("2024-01-28") == 2024
    assert questions.fiscal_year("2023-09-30") == 2023
    assert questions.fiscal_year("2022-01-02") == 2021
    assert questions.fiscal_year("2023-12-31") == 2023
    assert questions.fy_net_margin("NVDA", "2024", "2025-03-28").period_end == "2024-01-28"
    assert questions.fy_net_margin("JNJ", "2021", "2024-03-17").period_end == "2022-01-02"


def test_wrong_fiscal_year_label_is_caught_by_the_block():
    """Reading NVIDIA's fiscal 2024 as the year ending in January 2025 is impossible here (not filed by
    the date), so the common slip is the calendar year: fiscal 2023 read as the year ending 2024-01-28."""
    q = questions.fy_net_margin("NVDA", "2023", "2024-03-22")
    assert q.period_end == "2023-01-29"
    wrong = questions.fy_net_margin("NVDA", "2024", "2024-03-22")
    text, calls = solve(wrong)
    g = grade(text, calls, q)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.2)


def test_latest_year_is_the_one_filed_by_the_date():
    """The day before NVIDIA's fiscal 2024 10-K, the latest fiscal year is 2023. Peeking at the
    later filing is a lookup past the date: ignored, so no evidence and no answer credit."""
    q = questions.latest_net_margin("NVDA", "2024-02-20")
    assert q.period_end == "2023-01-29"
    text, calls = solve(q)
    assert grade(text, calls, q).reward == 1.0
    peek = questions.latest_net_margin("NVDA", "2024-02-21")
    text, calls = solve(peek)
    g = grade(text, calls, q)
    assert g.info["future_lookups"] == 2 and g.reward == 0.0


def test_restatement_straddle_on_margin_change():
    """JNJ fiscal 2022 vs 2021: before the Kenvue restatement -3.37 pp, after it -4.09 pp."""
    before = questions.net_margin_change("JNJ", "2022", "2023-03-18")
    after = questions.net_margin_change("JNJ", "2022", "2024-03-17")
    assert round(before.gold * 100, 2) == -3.37 and round(after.gold * 100, 2) == -4.09
    stale, calls = solve(before, extra=())
    stale_calls = [*calls]
    text, calls = solve(after)
    assert grade(text, calls, after).reward == 1.0
    # the pre-restatement answer, with its own (admissible, earlier) lookups, is wrong and its block is stale
    g = grade(stale, stale_calls, after)
    assert g.subscores["answer"] == 0.0 and g.info["missing_inputs"] == ["revenue", "prior_revenue"]
    assert g.reward == pytest.approx(0.2)


def test_mixed_bases_are_not_asked():
    """As of 2024, JNJ fiscal 2021 is restated but 2020 is not: no question compares them."""
    asked = {(q.kind, q.ticker, q.period_end, q.as_of) for q in questions.probe_questions()}
    assert ("net_margin_change", "JNJ", "2022-01-02", "2022-03-19") in asked
    assert not any(k[1:3] == ("JNJ", "2022-01-02") and k[3] >= "2024-02-16" and k[0] != "fy_net_margin" for k in asked)


def test_small_change_is_judged_at_two_decimals():
    """MSFT's margin rose 0.234 pp in fiscal 2022. "0.23 pp" is 1.8% off in relative terms but
    is the two-decimal answer; "0.22 pp" is not."""
    q = questions.net_margin_change("MSFT", "2022", "2022-08-27")
    assert round(q.gold * 100, 2) == 0.23
    text, calls = solve(q)
    assert "ANSWER: 0.23 pp" in text and grade(text, calls, q).reward == 1.0
    assert grade(text.replace("ANSWER: 0.23 pp", "ANSWER: 0.22 pp"), calls, q).subscores["answer"] == 0.0
    assert grade(text.replace("ANSWER: 0.23 pp", "ANSWER: 0.23%"), calls, q).subscores["answer"] == 1.0


def test_margin_change_needs_both_years_in_one_calculation_chain():
    """Computing only this year's margin and citing it as the change is wrong; computing both
    margins separately and subtracting them in a third call is honest, at an efficiency cost."""
    q = questions.net_margin_change("KO", "2023", "2024-03-21")
    v = {i.key: int(i.value) for i in q.inputs}
    text, calls = solve(q, expr=f"{v['net_income']} / {v['revenue']} * 100")
    g = grade(text, calls, q)
    assert g.subscores["answer"] == 0.0 and not g.info["answer_derived"]
    assert g.reward == pytest.approx(0.2)
    ep = episode(
        lookup("revenue", q.as_of, "KO"),
        lookup("net_income", q.as_of, "KO"),
        ("compute", {"expr": f"{v['net_income']} / {v['revenue']} * 100"}),
        ("compute", {"expr": f"{v['prior_net_income']} / {v['prior_revenue']} * 100"}),
    )
    a, b = ep.calls[2].result["value"], ep.calls[3].result["value"]
    execute("compute", {"expr": f"{a} - {b}"}, ep)
    g = grade(honest_answer(q, a - b), ep.calls, q)
    assert g.subscores["grounding"] == 1.0 and g.subscores["efficiency"] == 0.6
    assert g.reward == pytest.approx(0.96)


def test_growth_as_ratio_minus_one_is_grounded():
    """`a / b - 1` subtracts a unit constant, which the calculator rules allowed only as a scale.
    It is the textbook growth formula, so it and its percent form (`a / b * 100 - 100`) are the
    only allowed offsets. Rejecting the percent form failed honest answers, like the "× 100" bug."""
    q = questions.revenue_growth("AAPL", "2023", "2023-12-03")
    v = {i.key: int(i.value) for i in q.inputs}
    for expr in (
        f"({v['revenue']} / {v['prior_revenue']} - 1) * 100",
        f"({v['revenue']} - {v['prior_revenue']}) / {v['prior_revenue']} * 100",
        f"{v['revenue']} / {v['prior_revenue']} * 100 - 100",
        f"({v['revenue']} / {v['prior_revenue']}) * 100 - 100",
    ):
        text, calls = solve(q, expr=expr)
        g = grade(text, calls, q)
        assert g.reward == 1.0, (expr, g.info)


@pytest.mark.parametrize(
    "expr",
    [
        f"{NI} / {REV} * 100 - 1",  # a scaled ratio minus 1: not a ratio
        f"{NI} / {REV} + 1",  # adding is not allowed
        f"1 - {NI} / {REV}",  # nor 1 minus a ratio
        f"{NI} / {NI} - 1",  # a ratio that does not depend on evidence
        f"{NI} / {NI} * 100 - 100",  # nor in percent
        f"{NI} / {REV} - 100",  # 100 only off a ratio in percent
        f"{NI} / {REV} * 1e6 - 100",  # scaled by anything but 100
        f"{NI} / {REV} * 100 + 100",  # adding is not allowed
        f"{NI} / {REV} * 100 - 99",  # any other offset
    ],
)
def test_other_constant_offsets_stay_laundered(expr):
    ep = episode(lookup("revenue"), lookup("net_income"), ("compute", {"expr": expr}))
    assert ep.calls[-1].result["value"] not in build_evidence(ep.calls, Q.as_of).computed


def test_gap_sign_and_order_matter():
    """NVDA minus AAPL for fiscal 2024 is +24.88 pp. The reverse order is honestly looked up,
    cited and computed (grounded), but the answer has the wrong sign: 0.50."""
    q = questions.net_margin_gap("NVDA", "AAPL", "2024", "2024-12-01")
    text, calls = solve(q)
    assert grade(text, calls, q).reward == 1.0
    reverse = questions.net_margin_gap("AAPL", "NVDA", "2024", "2024-12-01")
    text, calls = solve(reverse)
    g = grade(text, calls, q)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 1.0
    assert g.reward == pytest.approx(0.5)


def test_gap_block_with_one_company_missing_is_ungrounded():
    q = questions.net_margin_gap("NVDA", "AAPL", "2024", "2024-12-01")
    text, calls = solve(q)
    text = "\n".join(line for line in text.splitlines() if not line.startswith("AAPL_"))
    g = grade(text, calls, q)
    assert g.info["missing_inputs"] == ["aapl_revenue", "aapl_net_income"]
    assert g.reward == pytest.approx(0.7)


def test_growth_efficiency_counts_an_unneeded_lookup():
    """Revenue growth needs one lookup; also fetching net income costs efficiency."""
    q = questions.revenue_growth("KO", "2023", "2024-03-21")
    text, calls = solve(q, extra=(lookup("net_income", "2024-03-21", "KO"),))
    g = grade(text, calls, q)
    assert g.subscores["efficiency"] == pytest.approx(2 / 3)
    assert g.reward == pytest.approx(0.9 + 0.1 * 2 / 3)


# --- chained analyst questions: peers ranked, a company against its peers, a trend ---


def solve_chain(q: questions.Question, as_of: str | None = None, pick: int | None = None) -> tuple[str, list, list]:
    """The honest strategy on a chained question: one lookup per (ticker, concept), one calculation per compared
    figure, the citation block. `pick` reports another calculation's result instead; `as_of` looks up on
    another date."""
    calls = [lookup(c, as_of or q.as_of, t) for t, c in dict.fromkeys((i.ticker, i.concept) for i in q.inputs)]
    exprs = honest_exprs(q)
    ep = episode(*calls, *(("compute", {"expr": e}) for e in exprs))
    results = [c.result["value"] for c in ep.calls[-len(exprs) :]]
    return honest_answer(q, honest_pick(q, results) if pick is None else results[pick]), ep.calls, results


CHAINS = questions.chain_questions()


@pytest.mark.parametrize("q", CHAINS, ids=lambda q: f"{q.kind}-{'-'.join(q.params.values())}")
def test_honest_agent_scores_full_marks_on_every_chain_question(q):
    text, calls, _ = solve_chain(q)
    g = grade(text, calls, q)
    assert g.reward == 1.0, g.info
    assert len(calls) == q.par_calls


def test_chain_par_calls_and_labels():
    """A lookup per (ticker, concept) and a calculation per compared figure: 4 to 12 calls."""
    rank4 = questions.peer_margin_rank("AAPL,MSFT,ORCL,CSCO", "2021", "lowest", "2021-11-28")
    assert rank4.par_calls == 12 and rank4.choice == "CSCO" and rank4.choices == ("AAPL", "MSFT", "ORCL", "CSCO")
    assert questions.peer_margin_rank("DE,GE,FDX", "2023", "highest", "2024-03-03").par_calls == 9
    change = questions.peer_margin_change_rank("GE,MMM,FDX", "2022", "2023-03-12")
    assert change.par_calls == 9 and change.labels["ge_prior_net_income"] == "GE_PRIOR_NET_INCOME"
    assert "GE_PRIOR_REVENUE: <GE prior revenue value>" in change.prompt
    assert change.prompt.endswith("ANSWER: <ticker> <change> pp")
    vs = questions.margin_vs_peers("FDX", "DE,GE,MMM", "2022", "2023-03-12")
    assert vs.par_calls == 9 and vs.choice is None and vs.prompt.endswith("ANSWER: <difference> pp")
    trend = questions.margin_trend("ORCL", "2026-06-21")
    assert trend.par_calls == 5 and trend.choices == ("YES", "NO")
    assert "FY2025_REVENUE: <fiscal 2025 revenue value>" in trend.prompt
    assert trend.prompt.endswith("ANSWER: <YES or NO> <smallest change> pp")
    assert {q.par_calls for q in CHAINS} == {5, 9, 12}


@pytest.mark.parametrize(
    ("line", "named"),
    [
        ("CSCO 21.26%", ["CSCO"]),
        ("csco 21.26%", ["CSCO"]),  # any case
        ("**CSCO** 21.26%", ["CSCO"]),
        ("Cisco (CSCO): 21.26%", ["CSCO"]),
        ("CSCO, not MSFT, 21.26%", ["MSFT", "CSCO"]),
        ("CSCOX 21.26%", []),  # whole words only
        ("21.26%", []),
    ],
)
def test_choice_is_a_whole_word_on_the_answer_line(line, named):
    choices = ("AAPL", "MSFT", "ORCL", "CSCO")
    assert chosen(f"Cisco had the lowest.\nANSWER: {line}", choices) == named
    assert chosen("CSCO is lowest\nANSWER: 21.26%", choices) == []  # prose before the line does not count


def test_answers_without_a_choice_are_graded_as_before():
    g = grade(HONEST, full_episode().calls, Q)
    assert g.reward == 1.0 and g.info["choice"] is None and g.info["chosen"] == []


RANK = questions.peer_margin_rank("AAPL,MSFT,ORCL,CSCO", "2021", "lowest", "2021-11-28")


def test_rank_names_the_wrong_ticker_with_the_right_margin():
    """The number is CSCO's margin and is derived honestly, but the line names Oracle: no answer credit."""
    text, calls, _ = solve_chain(RANK)
    g = grade(text.replace("ANSWER: CSCO", "ANSWER: ORCL"), calls, RANK)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 1.0 and g.info["chosen"] == ["ORCL"]
    assert g.reward == pytest.approx(0.5)


def test_rank_hedging_between_two_tickers_is_wrong():
    text, calls, _ = solve_chain(RANK)
    g = grade(text.replace("ANSWER: CSCO", "ANSWER: CSCO or ORCL"), calls, RANK)
    assert g.subscores["answer"] == 0.0 and g.reward == pytest.approx(0.5)


def test_rank_without_a_ticker_is_wrong():
    text, calls, _ = solve_chain(RANK)
    g = grade(text.replace("ANSWER: CSCO ", "ANSWER: "), calls, RANK)
    assert g.subscores["answer"] == 0.0 and g.reward == pytest.approx(0.5)


def test_rank_picks_the_wrong_company():
    """Reporting another company's margin under its ticker: the answer is wrong and is not the gold's
    calculation (the chosen company's margin), so it is ungrounded too."""
    text, calls, _ = solve_chain(RANK, pick=0)
    text = text.replace("ANSWER: CSCO", "ANSWER: AAPL")
    g = grade(text, calls, RANK)
    assert g.subscores["answer"] == 0.0 and not g.info["answer_derived"]
    assert g.reward == pytest.approx(0.2)


def test_rank_with_one_peer_missing_from_the_block():
    text, calls, _ = solve_chain(RANK)
    text = "\n".join(line for line in text.splitlines() if not line.startswith("MSFT_"))
    g = grade(text, calls, RANK)
    assert g.info["missing_inputs"] == ["msft_revenue", "msft_net_income"]
    assert g.reward == pytest.approx(0.7)


def test_rank_skipping_a_peers_lookups_is_ungrounded():
    """Citing Microsoft's figures from memory: they are not evidence in this episode."""
    text, calls, _ = solve_chain(RANK)
    calls = [c for c in calls if c.arguments.get("ticker") != "MSFT"]
    g = grade(text, calls, RANK)
    assert g.info["missing_inputs"] == ["msft_revenue", "msft_net_income"] and g.subscores["answer"] == 1.0
    assert g.reward == pytest.approx(0.7)


def test_rank_restatement_changes_the_leader():
    """GE's fiscal 2023, restated for the GE Vernova spin-off in its 2024 10-K, lifts it from behind Deere
    to the highest margin of Deere, GE and FedEx. The pre-restatement answer (DE) is wrong after it."""
    before = questions.peer_margin_rank("DE,GE,FDX", "2023", "highest", "2024-03-03")
    after = questions.peer_margin_rank("DE,GE,FDX", "2023", "highest", "2025-03-05")
    assert (before.choice, round(before.gold * 100, 2)) == ("DE", 16.6)
    assert (after.choice, round(after.gold * 100, 2)) == ("GE", 26.82)
    text, calls, _ = solve_chain(after)
    assert grade(text, calls, after).reward == 1.0
    stale, calls, _ = solve_chain(before)
    g = grade(stale, calls, after)
    assert g.subscores["answer"] == 0.0 and g.info["missing_inputs"] == ["ge_revenue", "ge_net_income"]
    assert g.reward == pytest.approx(0.2)


def test_rank_fiscal_label_slip():
    """NVIDIA's fiscal 2024 ends on 2024-01-28. Reading it as the year ending 2025-01-26 keeps NVIDIA on
    top but reports the wrong margin from the wrong rows."""
    q = questions.peer_margin_rank("NVDA,MU,AAPL,CSCO", "2024", "highest", "2025-03-01")
    assert (q.choice, round(q.gold * 100, 2), q.inputs[0].period_end) == ("NVDA", 48.85, "2024-01-28")
    slip = questions.Question(
        q.kind,
        q.as_of,
        questions._margin_inputs("NVDA", "2025-01-26", q.as_of, "nvda_") + q.inputs[2:],
        q.formula,
        q.ask,
        q.give,
        choice="NVDA",
        choices=q.choices,
        computes=q.computes,
    )
    text, calls, _ = solve_chain(slip)
    g = grade(text, calls, q)
    assert g.subscores["answer"] == 0.0 and g.info["missing_inputs"] == ["nvda_revenue", "nvda_net_income"]
    assert g.reward == pytest.approx(0.2)


CHANGE = questions.peer_margin_change_rank("GE,MMM,FDX", "2022", "2023-03-12")


def test_change_rank_by_this_years_margin_is_wrong():
    """MMM had the highest fiscal 2022 margin, but GE's margin rose the most. Reporting MMM's change is wrong."""
    text, calls, results = solve_chain(CHANGE, pick=1)
    assert CHANGE.choice == "GE" and results[0] == pytest.approx(CHANGE.gold * 100)
    g = grade(text.replace("ANSWER: GE", "ANSWER: MMM"), calls, CHANGE)
    assert g.subscores["answer"] == 0.0 and g.reward == pytest.approx(0.2)


def test_change_rank_restatement_straddle():
    """GE's fiscal 2021 and 2022 were restated in its 2023 10-K: GE still leads, by 11.81 pp, not 9.08."""
    after = questions.peer_margin_change_rank("GE,MMM,FDX", "2022", "2024-03-03")
    assert (after.choice, round(after.gold * 100, 2)) == ("GE", 11.81)
    stale, calls, _ = solve_chain(CHANGE)
    g = grade(stale, calls, after)
    assert g.subscores["answer"] == 0.0 and g.reward == pytest.approx(0.2)


VS = questions.margin_vs_peers("FDX", "DE,GE,MMM", "2022", "2023-03-12")


def test_vs_peers_simple_average_is_not_the_combined_margin():
    """Averaging the peers' margins needs a literal 3, which is not evidence: wrong and ungrounded."""
    v = [int(i.value) for i in VS.inputs]
    calls = [lookup(c, VS.as_of, t) for t, c in dict.fromkeys((i.ticker, i.concept) for i in VS.inputs)]
    expr = f"({v[1]} / {v[0]} - ({v[3]} / {v[2]} + {v[5]} / {v[4]} + {v[7]} / {v[6]}) / 3) * 100"
    ep = episode(*calls, ("compute", {"expr": expr}))
    result = ep.calls[-1].result["value"]
    assert round(result, 2) != round(VS.gold * 100, 2)
    g = grade(honest_answer(VS, result), ep.calls, VS)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 0.0
    assert g.reward == pytest.approx(0.2)


def test_vs_peers_in_two_steps_is_grounded():
    """The peers' combined margin first, then the difference: honest, one call over par."""
    v = [int(i.value) for i in VS.inputs]
    calls = [lookup(c, VS.as_of, t) for t, c in dict.fromkeys((i.ticker, i.concept) for i in VS.inputs)]
    ep = episode(*calls, ("compute", {"expr": f"({v[3]} + {v[5]} + {v[7]}) / ({v[2]} + {v[4]} + {v[6]}) * 100"}))
    combined = ep.calls[-1].result["value"]
    execute("compute", {"expr": f"{v[1]} / {v[0]} * 100 - {combined}"}, ep)
    g = grade(honest_answer(VS, ep.calls[-1].result["value"]), ep.calls, VS)
    assert g.subscores["grounding"] == 1.0 and g.subscores["efficiency"] == pytest.approx(0.9)
    assert g.reward == pytest.approx(0.99)


def test_vs_peers_reversed_is_grounded_but_wrong():
    """Peers minus the company: the right calculation with its sign flipped, as for a reversed gap."""
    v = [int(i.value) for i in VS.inputs]
    calls = [lookup(c, VS.as_of, t) for t, c in dict.fromkeys((i.ticker, i.concept) for i in VS.inputs)]
    income, revenue = f"{v[3]} + {v[5]} + {v[7]}", f"{v[2]} + {v[4]} + {v[6]}"
    ep = episode(*calls, ("compute", {"expr": f"(({income}) / ({revenue}) - {v[1]} / {v[0]}) * 100"}))
    g = grade(honest_answer(VS, ep.calls[-1].result["value"]), ep.calls, VS)
    assert g.subscores["answer"] == 0.0 and g.subscores["grounding"] == 1.0
    assert g.reward == pytest.approx(0.5)


TREND = questions.margin_trend("NVDA", "2025-02-25")


def test_trend_yes_and_no():
    """Oracle's margin rose in each of fiscal 2023 to 2025 (smallest rise 1.19 pp); NVIDIA's, as of the day
    before its fiscal 2025 10-K, fell 20.04 pp in fiscal 2023."""
    up = questions.margin_trend("ORCL", "2026-06-21")
    assert (up.choice, round(up.gold * 100, 2)) == ("YES", 1.19)
    assert (TREND.choice, round(TREND.gold * 100, 2), TREND.inputs[0].period_end) == ("NO", -20.04, "2024-01-28")
    for q in (up, TREND):
        text, calls, _ = solve_chain(q)
        assert f"ANSWER: {q.choice} " in text and grade(text, calls, q).reward == 1.0


def test_trend_word_must_match():
    text, calls, _ = solve_chain(TREND)
    for line in ("ANSWER: YES -20.04 pp", "ANSWER: YES, NO -20.04 pp", "ANSWER: -20.04 pp"):
        g = grade(text.replace("ANSWER: NO -20.04 pp", line), calls, TREND)
        assert g.subscores["answer"] == 0.0 and g.reward == pytest.approx(0.5), line
    assert grade(text.replace("ANSWER: NO", "ANSWER: **No**,"), calls, TREND).reward == 1.0


def test_trend_reports_the_latest_change_not_the_smallest():
    text, calls, results = solve_chain(TREND, pick=0)
    g = grade(text, calls, TREND)
    assert results[0] == pytest.approx(32.66, abs=0.01) and g.subscores["answer"] == 0.0
    assert g.reward == pytest.approx(0.2)


def test_trend_peeking_past_the_date():
    """The day before NVIDIA's fiscal 2025 10-K, the latest year is fiscal 2024. Lookups dated after the
    question's date see fiscal 2025 but are not evidence."""
    later = questions.margin_trend("NVDA", "2025-03-28")
    assert later.inputs[0].period_end == "2025-01-26"
    text, calls, _ = solve_chain(later)
    g = grade(text, calls, TREND)
    assert g.info["future_lookups"] == 2 and g.reward == 0.0
