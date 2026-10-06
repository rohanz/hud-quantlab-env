"""Question types: point-in-time ratios over annual revenue and net income.

Every question names the figures it needs (its `inputs`: a ticker, a concept and
a fiscal period, each cited in the answer under its own label), a gold formula
over those figures, and the prompt. The gold uses exactly what the tools serve
on the question's date (`data.facts_as_of`), so restatements filed later are
invisible to it.

Three task sets:

- the legacy set (`default_questions`): 18 net-margin questions that name the
  fiscal period by its end date. QuantLab's base model lost most of its points
  on fiscal-year labels, so these avoided them.
- the probe set (`probe_questions`): harder questions derived from the
  fixture. They name fiscal years by label ("fiscal 2023") as the company
  does, ask for the latest year filed by a date, straddle restatements, and
  combine two years or two companies.
- the chained set (`chain_questions`): what an analyst asks about a peer group
  or several years at once (the highest margin, the largest margin expansion,
  a company against its peers' combined margin, a three-year trend), with a
  calculation per compared figure. Frozen subsets of the probe sets:
  `probe_sample` (v1) and `probe_sample_v2`.

Fiscal-year labels (`fiscal_year`): a fiscal year is labelled by the calendar
year in which it ends, except that a 52/53-week year ending in the first week
of January belongs to the year before. So NVIDIA's fiscal 2024 ends on
2024-01-28, Apple's fiscal 2023 on 2023-09-30, Microsoft's on 2023-06-30, and
Johnson & Johnson's fiscal 2021 on 2022-01-02. This is how every company in
the fixture labels its own years; retailers that label a year by the calendar
year in which it starts (fiscal 2023 ending in early 2024) do not fit it and
are kept out of the fixture.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import pairwise

from . import data

Formula = Callable[[Sequence[float]], float]
_CONCEPT_NAMES = {"revenue": "revenue", "net_income": "net income"}
MIN_CHANGE = 0.001  # a change, growth or gap under 0.1 percentage points rounds to nothing at two decimals


@dataclass(frozen=True)
class Input:
    """One figure a question needs, cited in the answer as `LABEL: value`."""

    key: str  # "revenue", "prior_net_income", "aapl_revenue"; the citation label is key.upper()
    ticker: str
    concept: str
    period_end: str
    value: float

    @property
    def label(self) -> str:
        return self.key.upper()


@dataclass(frozen=True)
class Question:
    kind: str  # "net_margin", "fy_net_margin", "latest_net_margin", "net_margin_change", "revenue_growth", ...
    as_of: str
    inputs: tuple[Input, ...]
    formula: Formula = field(compare=False, repr=False)  # gold, as a fraction, over the input values in order
    ask: str = field(compare=False, repr=False)  # the question sentence
    give: str = field(compare=False, repr=False)  # "the margin as a percentage"
    placeholder: str = "margin"  # the ANSWER line's placeholder
    unit: str = "%"  # how the ANSWER line ends: "%" or " pp"
    choice: str | None = None  # a word the ANSWER line must name ("MSFT", "YES"), and no other of `choices`
    choices: tuple[str, ...] = ()
    computes: int = 1  # calculations at par: one per figure a ranking or trend has to compare
    params: dict[str, str] = field(default_factory=dict, compare=False, repr=False)  # template args, if not derived

    @property
    def ticker(self) -> str:
        return self.inputs[0].ticker

    @property
    def period_end(self) -> str:
        return self.inputs[0].period_end

    def value(self, key: str) -> float:
        return next(i.value for i in self.inputs if i.key == key)

    @property
    def revenue(self) -> float:
        return self.value("revenue")

    @property
    def net_income(self) -> float:
        return self.value("net_income")

    @property
    def gold(self) -> float:
        return self.formula([i.value for i in self.inputs])

    @property
    def labels(self) -> dict[str, str]:
        return {i.key: i.label for i in self.inputs}

    @property
    def par_calls(self) -> int:
        """One lookup per (ticker, concept) needed, and one calculation (one per compared figure for a
        ranking or a trend)."""
        return len({(i.ticker, i.concept) for i in self.inputs}) + self.computes

    @property
    def prompt(self) -> str:
        n = len(self.inputs)
        count = _COUNTS.get(n + 1, str(n + 1))
        if self.kind == "net_margin":
            what = "the revenue and net income you used for that fiscal year"
        else:
            what = "each figure you used"
        block = "".join(f"{i.label}: <{_describe(i)}>\n" for i in self.inputs)
        return (
            f"{self.ask}\n\n"
            "Use the tools to look up every figure and the compute tool for arithmetic. "
            "Any other number you mention must be exactly as a tool returned it or rounded "
            f"from it. Give {self.give} to two decimal places. End with "
            f"exactly these {count} lines, citing {what} exactly as get_fundamentals returned them: the whole "
            "value in USD with its sign, digits only (a leading $ and thousands commas are allowed; no rounding, no "
            f"units):\n{block}ANSWER: {self.answer_prefix}<{self.placeholder}>{self.unit}"
        )

    @property
    def answer_prefix(self) -> str:
        """What the ANSWER line names before its number: "<ticker> ", "<YES or NO> " or nothing."""
        if self.choice is None:
            return ""
        return "<YES or NO> " if set(self.choices) == {"YES", "NO"} else "<ticker> "


_COUNTS = dict(enumerate(["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"], 2))
_COUNTS.update({12: "twelve", 13: "thirteen"})


def _describe(i: Input) -> str:
    """The placeholder in the prompt's citation block: "revenue value", "prior net income value",
    "AAPL prior revenue value", "fiscal 2024 net income value"."""
    words, parts = i.key.split("_"), []
    if words[0] == i.ticker.lower():
        parts.append(i.ticker)
        words = words[1:]
    elif re.fullmatch(r"fy\d{4}", words[0]):
        parts.append(f"fiscal {words[0][2:]}")
        words = words[1:]
    if words[0] == "prior":
        parts.append("prior")
    return " ".join([*parts, _CONCEPT_NAMES[i.concept], "value"])


def _name(ticker: str) -> str:
    return f"{data.companies()[ticker]['name']} ({ticker})"


def _margin(v: Sequence[float]) -> float:
    return v[1] / v[0]


def _margin_change(v: Sequence[float]) -> float:
    return v[1] / v[0] - v[3] / v[2]


def _growth(v: Sequence[float]) -> float:
    return v[0] / v[1] - 1


def fiscal_year(period_end: str) -> int:
    """The fiscal-year label of a period: the year it ends in, a first-week-of-January end counting as the year before."""
    return (date.fromisoformat(period_end) - timedelta(days=7)).year


def _visible(ticker: str, concept: str, as_of: str) -> dict[str, dict]:
    return {r["period_end"]: r for r in data.facts_as_of(ticker, concept, as_of)}


def _input(key: str, ticker: str, concept: str, period_end: str, as_of: str) -> Input:
    rows = _visible(ticker, concept, as_of)
    if period_end not in rows:
        raise ValueError(f"{ticker} {concept} {period_end} not filed by {as_of}")
    return Input(key, ticker, concept, period_end, float(rows[period_end]["value"]))


def _period(ticker: str, fiscal: int | str, as_of: str) -> str:
    """The period end of a company's fiscal year, among periods filed by `as_of`."""
    ends = [p for p in _visible(ticker, "revenue", as_of) if fiscal_year(p) == int(fiscal)]
    if len(ends) != 1:
        raise ValueError(f"{ticker} fiscal {fiscal}: {len(ends)} periods filed by {as_of}")
    return ends[0]


def _margin_inputs(ticker: str, period_end: str, as_of: str, prefix: str = "") -> tuple[Input, Input]:
    return (
        _input(f"{prefix}revenue", ticker, "revenue", period_end, as_of),
        _input(f"{prefix}net_income", ticker, "net_income", period_end, as_of),
    )


_MARGIN = "net margin (net income divided by revenue)"


def build(ticker: str, period_end: str, as_of: str) -> Question:
    """The legacy question: net margin for the fiscal year ending on a given date."""
    return Question(
        "net_margin",
        as_of,
        _margin_inputs(ticker, period_end, as_of),
        _margin,
        f"What was {_name(ticker)}'s {_MARGIN} for the fiscal year ending {period_end}, "
        f"using only filings available on {as_of}?",
        "the margin as a percentage",
    )


def fy_net_margin(ticker: str, fiscal: str, as_of: str) -> Question:
    """Net margin for a fiscal year named by its label, not its end date."""
    return Question(
        "fy_net_margin",
        as_of,
        _margin_inputs(ticker, _period(ticker, fiscal, as_of), as_of),
        _margin,
        f"What was {_name(ticker)}'s {_MARGIN} for its fiscal {fiscal}, using only filings available on "
        f"{as_of}? Fiscal years are named as the company names them.",
        "the margin as a percentage",
    )


def latest_net_margin(ticker: str, as_of: str) -> Question:
    """Net margin for the most recent fiscal year whose 10-K was filed by `as_of`."""
    return Question(
        "latest_net_margin",
        as_of,
        _margin_inputs(ticker, _latest_period(ticker, as_of), as_of),
        _margin,
        f"What was {_name(ticker)}'s {_MARGIN} for its most recent fiscal year reported in a 10-K filed "
        f"on or before {as_of}, using only filings available on {as_of}?",
        "the margin as a percentage",
    )


def net_margin_change(ticker: str, fiscal: str, as_of: str) -> Question:
    """Change in net margin from the prior fiscal year, in percentage points."""
    period, prior = _period(ticker, fiscal, as_of), _period(ticker, int(fiscal) - 1, as_of)
    return Question(
        "net_margin_change",
        as_of,
        _margin_inputs(ticker, period, as_of) + _margin_inputs(ticker, prior, as_of, "prior_"),
        _margin_change,
        f"By how many percentage points did {_name(ticker)}'s {_MARGIN} change from fiscal {int(fiscal) - 1} "
        f"to fiscal {fiscal}, using only filings available on {as_of}? Fiscal years are named as the company "
        "names them.",
        "the change in percentage points (negative if the margin fell)",
        "change",
        " pp",
    )


def revenue_growth(ticker: str, fiscal: str, as_of: str) -> Question:
    """Revenue growth over the prior fiscal year, in percent."""
    period, prior = _period(ticker, fiscal, as_of), _period(ticker, int(fiscal) - 1, as_of)
    return Question(
        "revenue_growth",
        as_of,
        (
            _input("revenue", ticker, "revenue", period, as_of),
            _input("prior_revenue", ticker, "revenue", prior, as_of),
        ),
        _growth,
        f"By what percentage did {_name(ticker)}'s revenue grow from fiscal {int(fiscal) - 1} to fiscal "
        f"{fiscal}, using only filings available on {as_of}? Fiscal years are named as the company names them.",
        "the growth as a percentage (negative if revenue fell)",
        "growth",
    )


def net_margin_gap(ticker: str, other: str, fiscal: str, as_of: str) -> Question:
    """How far one company's net margin was above another's for the same fiscal-year label."""
    return Question(
        "net_margin_gap",
        as_of,
        _margin_inputs(ticker, _period(ticker, fiscal, as_of), as_of, f"{ticker.lower()}_")
        + _margin_inputs(other, _period(other, fiscal, as_of), as_of, f"{other.lower()}_"),
        _margin_change,
        f"By how many percentage points was {_name(ticker)}'s {_MARGIN} for its fiscal {fiscal} above "
        f"{_name(other)}'s for its fiscal {fiscal}, using only filings available on {as_of}? Fiscal years "
        "are named as each company names them.",
        "the difference in percentage points (negative if it was below)",
        "difference",
        " pp",
    )


# --- chained analyst questions: several companies or several years, compared --
#
# Each needs the figures of several companies or years, a calculation per figure compared, and a
# comparison. The gold formula is the answer's own calculation (the chosen company's margin, the
# smallest change), so the grader's derivation check is unchanged; the choice the comparison makes
# (a ticker, YES or NO) must also be named on the ANSWER line.

PICK_TOL = 0.005  # the grader's relative answer tolerance: a picked figure must stand clear of the others by twice it


def _names(tickers: Sequence[str]) -> str:
    names = [_name(t) for t in tickers]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _pick(values: Sequence[float], highest: bool) -> int:
    """The index of the highest (or lowest) value; ValueError unless it stands clear of every other, so the
    answer's number alone tells the pick apart (by 0.1 percentage points and twice the answer tolerance)."""
    k = max(range(len(values)), key=lambda j: values[j] if highest else -values[j])
    gap = max(MIN_CHANGE, 2 * PICK_TOL * abs(values[k]))
    if any(abs(v - values[k]) < gap for j, v in enumerate(values) if j != k):
        raise ValueError(f"no clear pick among {values}")
    return k


def _margin_of(k: int) -> Formula:
    """The margin of the k-th (revenue, net income) pair."""
    return lambda v: v[2 * k + 1] / v[2 * k]


def _change_of(k: int, stride: int) -> Formula:
    """A margin minus the next pair's: at inputs k * stride (this year) and k * stride + 2 (the year before)."""
    i = k * stride
    return lambda v: v[i + 1] / v[i] - v[i + 3] / v[i + 2]


def peer_margin_rank(tickers: str, fiscal: str, rank: str, as_of: str) -> Question:
    """Which of several peers had the highest (or lowest) net margin for a fiscal-year label, and that margin."""
    if rank not in ("highest", "lowest"):
        raise ValueError(f"rank must be highest or lowest, not {rank!r}")
    ts = tickers.split(",")
    inputs = tuple(x for t in ts for x in _margin_inputs(t, _period(t, fiscal, as_of), as_of, f"{t.lower()}_"))
    values = [i.value for i in inputs]
    k = _pick([_margin_of(j)(values) for j in range(len(ts))], rank == "highest")
    return Question(
        "peer_margin_rank",
        as_of,
        inputs,
        _margin_of(k),
        f"Which of {_names(ts)} had the {rank} {_MARGIN} for its fiscal {fiscal}, using only filings "
        f"available on {as_of}? Fiscal years are named as each company names them.",
        "that company's ticker and its margin as a percentage",
        choice=ts[k],
        choices=tuple(ts),
        computes=len(ts),
        params={"tickers": tickers, "fiscal_year": fiscal, "rank": rank, "as_of": as_of},
    )


def peer_margin_change_rank(tickers: str, fiscal: str, as_of: str) -> Question:
    """Which of several peers expanded its net margin the most over the prior fiscal year, and by how much."""
    ts, year = tickers.split(","), int(fiscal)
    inputs = tuple(
        x
        for t in ts
        for x in _margin_inputs(t, _period(t, year, as_of), as_of, f"{t.lower()}_")
        + _margin_inputs(t, _period(t, year - 1, as_of), as_of, f"{t.lower()}_prior_")
    )
    values = [i.value for i in inputs]
    k = _pick([_change_of(j, 4)(values) for j in range(len(ts))], highest=True)
    return Question(
        "peer_margin_change_rank",
        as_of,
        inputs,
        _change_of(k, 4),
        f"Which of {_names(ts)} expanded its {_MARGIN} the most from fiscal {year - 1} to fiscal {year} "
        f"(the largest change in percentage points, or the smallest fall if none rose), using only filings "
        f"available on {as_of}? Fiscal years are named as each company names them.",
        "that company's ticker and its change in percentage points (negative if its margin fell)",
        "change",
        " pp",
        choice=ts[k],
        choices=tuple(ts),
        computes=len(ts),
        params={"tickers": tickers, "fiscal_year": fiscal, "as_of": as_of},
    )


def _vs_combined(v: Sequence[float]) -> float:
    """The first company's margin minus the rest's combined margin (total net income over total revenue)."""
    return v[1] / v[0] - sum(v[3::2]) / sum(v[2::2])


def margin_vs_peers(ticker: str, peers: str, fiscal: str, as_of: str) -> Question:
    """One company's net margin against its peers' combined net margin, for a fiscal-year label, in pp."""
    ts = [ticker, *peers.split(",")]
    inputs = tuple(x for t in ts for x in _margin_inputs(t, _period(t, fiscal, as_of), as_of, f"{t.lower()}_"))
    return Question(
        "margin_vs_peers",
        as_of,
        inputs,
        _vs_combined,
        f"By how many percentage points was {_name(ticker)}'s {_MARGIN} for its fiscal {fiscal} above the "
        f"combined net margin of {_names(ts[1:])} (their total net income divided by their total revenue, each "
        f"for its own fiscal {fiscal}), using only filings available on {as_of}? Fiscal years are named as "
        "each company names them.",
        "the difference in percentage points (negative if it was below)",
        "difference",
        " pp",
        params={"ticker": ticker, "peers": peers, "fiscal_year": fiscal, "as_of": as_of},
    )


TREND_YEARS = 3  # year-over-year changes asked about; one more year of figures is needed


def margin_trend(ticker: str, as_of: str) -> Question:
    """Whether net margin rose in each of the last three fiscal years filed by `as_of`, and the smallest change."""
    latest = fiscal_year(_latest_period(ticker, as_of))
    years = [latest - j for j in range(TREND_YEARS + 1)]
    inputs = tuple(x for y in years for x in _margin_inputs(ticker, _period(ticker, y, as_of), as_of, f"fy{y}_"))
    values = [i.value for i in inputs]
    changes = [_change_of(j, 2)(values) for j in range(TREND_YEARS)]
    k = _pick(changes, highest=False)
    return Question(
        "margin_trend",
        as_of,
        inputs,
        _change_of(k, 2),
        f"Did {_name(ticker)}'s {_MARGIN} rise in each of its last three fiscal years reported in a 10-K filed "
        f"on or before {as_of}, each year against the year before, using only filings available on {as_of}? "
        "Fiscal years are named as the company names them.",
        "YES or NO, then the smallest of the three year-over-year changes in percentage points (negative if "
        "the margin fell)",
        "smallest change",
        " pp",
        choice="YES" if changes[k] > 0 else "NO",
        choices=("YES", "NO"),
        computes=TREND_YEARS,
        params={"ticker": ticker, "as_of": as_of},
    )


# --- the legacy set ------------------------------------------------------------

# Pinned so it stays these 18 after the fixture grows: one question per company-year,
# 30 days after the first 10-K for that year, plus JNJ's restatement traps.
LEGACY = (
    ("AAPL", "2022-09-24", "2022-11-27"),
    ("AAPL", "2023-09-30", "2023-12-03"),
    ("AAPL", "2024-09-28", "2024-12-01"),
    ("MSFT", "2022-06-30", "2022-08-27"),
    ("MSFT", "2023-06-30", "2023-08-26"),
    ("MSFT", "2024-06-30", "2024-08-29"),
    ("NVDA", "2022-01-30", "2022-04-17"),
    ("NVDA", "2023-01-29", "2023-03-26"),
    ("NVDA", "2024-01-28", "2024-03-22"),
    ("KO", "2022-12-31", "2023-03-23"),
    ("KO", "2023-12-31", "2024-03-21"),
    ("KO", "2024-12-31", "2025-03-22"),
    ("JNJ", "2022-01-02", "2022-03-19"),
    ("JNJ", "2022-01-02", "2024-03-17"),
    ("JNJ", "2023-01-01", "2023-03-18"),
    ("JNJ", "2023-01-01", "2025-03-15"),
    ("JNJ", "2023-12-31", "2024-03-17"),
    ("JNJ", "2024-12-29", "2025-03-15"),
)


def default_questions() -> list[Question]:
    """The legacy set: net margin for a fiscal year named by its end date."""
    return [build(*args) for args in LEGACY]


# --- the probe set, derived from the fixture ----------------------------------


def _days(day: str, n: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()


def _rows(ticker: str, concept: str, period_end: str) -> list[dict]:
    return sorted(
        (r for r in data.companies()[ticker]["facts"][concept] if r["period_end"] == period_end),
        key=lambda r: (r["filed"], r["accn"]),
    )


def _first_filed(ticker: str, period_end: str) -> str:
    return max(_rows(ticker, c, period_end)[0]["filed"] for c in ("revenue", "net_income"))


def _restated(ticker: str, periods: Sequence[str], concepts: Sequence[str]) -> list[str]:
    """Filing dates on which any of these figures changed value."""
    return sorted(
        {
            b["filed"]
            for p in periods
            for c in concepts
            for a, b in pairwise(_rows(ticker, c, p))
            if a["value"] != b["value"]
        }
    )


def _fiscal_periods(ticker: str) -> dict[int, str]:
    """Fiscal-year label to period end, for periods with both concepts and a label used by one period only."""
    facts = data.companies()[ticker]["facts"]
    periods = sorted({r["period_end"] for r in facts["revenue"]} & {r["period_end"] for r in facts["net_income"]})
    labels = [fiscal_year(p) for p in periods]
    return {fiscal_year(p): p for p in periods if labels.count(fiscal_year(p)) == 1}


def _filings(ticker: str) -> list[str]:
    facts = data.companies()[ticker]["facts"]
    return sorted({r["filed"] for c in ("revenue", "net_income") for r in facts[c]})


def _latest_period(ticker: str, as_of: str) -> str:
    periods = set(_visible(ticker, "revenue", as_of)) & set(_visible(ticker, "net_income", as_of))
    if not periods:
        raise ValueError(f"{ticker}: nothing filed by {as_of}")
    return max(periods)


def _dates(ticker: str, periods: Sequence[str], concepts: Sequence[str], base: str) -> list[str]:
    """`base`, plus 30 days after every restatement of these figures, plus the view before the first one."""
    return _group_dates({ticker: periods}, concepts, base)


def _group_dates(periods: dict[str, Sequence[str]], concepts: Sequence[str], base: str) -> list[str]:
    """`_dates` over several companies' periods: a restatement of any of them is asked before and after."""
    changes = [_days(d, 30) for t, ps in periods.items() for d in _restated(t, ps, concepts)]
    first = _days(max(_first_filed(t, p) for t, ps in periods.items() for p in ps), 30)
    dates = {base, *changes}
    if changes and base >= min(changes):
        dates.add(first)
    return sorted(dates)


def _same_filing(ticker: str, concepts: Sequence[str], period: str, prior: str, as_of: str) -> bool:
    """Whether both years are on one basis: a restated current figure needs the prior year from the same filing.

    JNJ's 2024 10-K restated fiscal 2021 for the Kenvue separation but did not
    report 2020, so as of 2024 the two years are not comparable and no question
    mixes them.
    """
    for c in concepts:
        current, before = _visible(ticker, c, as_of)[period], _visible(ticker, c, as_of)[prior]
        restated = any(r["value"] != current["value"] for r in _rows(ticker, c, period) if r["filed"] <= as_of)
        same = any(r["accn"] == current["accn"] and r["value"] == before["value"] for r in _rows(ticker, c, prior))
        if restated and not same:
            return False
    return True


def _probe_fy_margins() -> list[Question]:
    """Every company-year by label: 30 days after the next year's 10-K (so later rows are there to confuse),
    or after its own if there is none, and around every restatement."""
    out = []
    for ticker in data.companies():
        filings = _filings(ticker)
        for fiscal, period in _fiscal_periods(ticker).items():
            own = _first_filed(ticker, period)
            later = [f for f in filings if f > own]
            base = _days(later[0] if later else own, 30)
            for as_of in _dates(ticker, [period], ("revenue", "net_income"), base):
                out.append(fy_net_margin(ticker, str(fiscal), as_of))
    return out


def _probe_latest() -> list[Question]:
    """The day before each 10-K: the latest year is the previous one, though the calendar suggests otherwise."""
    out = []
    for ticker in data.companies():
        labelled = set(_fiscal_periods(ticker).values())
        for filed in _filings(ticker)[1:]:
            as_of = _days(filed, -1)
            period = _latest_period(ticker, as_of)
            # skip dates where the newest 10-K reports a year outside the fixture's window
            if period in labelled and _first_filed(ticker, period) == _filings_before(ticker, as_of)[-1]:
                out.append(latest_net_margin(ticker, as_of))
    return out


def _filings_before(ticker: str, as_of: str) -> list[str]:
    return [f for f in _filings(ticker) if f <= as_of]


def _probe_year_over_year(
    build_question: Callable[[str, str, str], Question], concepts: Sequence[str]
) -> list[Question]:
    """Each fiscal year against the one before, 30 days after its 10-K and around restatements."""
    out = []
    for ticker in data.companies():
        periods = _fiscal_periods(ticker)
        for fiscal, period in periods.items():
            prior = periods.get(fiscal - 1)
            if prior is None:
                continue
            base = _days(_first_filed(ticker, period), 30)
            for as_of in _dates(ticker, [period, prior], concepts, base):
                if not _same_filing(ticker, concepts, period, prior, as_of):
                    continue
                q = build_question(ticker, str(fiscal), as_of)
                if abs(q.gold) >= MIN_CHANGE:
                    out.append(q)
    return out


def _probe_gaps() -> list[Question]:
    """For each fiscal-year label, each company against the next (alphabetically, wrapping round)."""
    by_year: dict[int, list[str]] = {}
    for ticker in sorted(data.companies()):
        for fiscal in _fiscal_periods(ticker):
            by_year.setdefault(fiscal, []).append(ticker)
    out = []
    for fiscal, tickers in sorted(by_year.items()):
        if len(tickers) < 2:
            continue
        pairs = list(zip(tickers, tickers[1:] + tickers[:1], strict=True))
        for a, b in pairs if len(tickers) > 2 else pairs[:1]:
            periods = {t: _fiscal_periods(t)[fiscal] for t in (a, b)}
            as_of = _days(max(_first_filed(t, p) for t, p in periods.items()), 30)
            q = net_margin_gap(a, b, str(fiscal), as_of)
            if abs(q.gold) >= MIN_CHANGE:
                out.append(q)
    return out


def probe_questions() -> list[Question]:
    """The probe set, derived from whatever companies and years the fixture holds."""
    return (
        _probe_fy_margins()
        + _probe_latest()
        + _probe_year_over_year(net_margin_change, ("revenue", "net_income"))
        + _probe_year_over_year(revenue_growth, ("revenue",))
        + _probe_gaps()
    )


# Peer groups an analyst would compare: each a sector's names in the fixture, mostly with fiscal years
# that do not end in December (and NVIDIA's and Walmart's end in January of the year they are named for).
PEER_GROUPS = (
    ("AAPL", "MSFT", "ORCL", "CSCO"),  # large-cap technology
    ("NVDA", "MU", "AAPL", "CSCO"),  # semiconductors and hardware
    ("KO", "NKE", "COST", "WMT"),  # consumer
    ("DE", "GE", "MMM", "FDX"),  # industrials
)


def _peer_sets() -> list[tuple[tuple[str, ...], int, int]]:
    """Each peer group and each three of it (numbered 0 to 4), with each fiscal-year label all of them have."""
    out = []
    for group in PEER_GROUPS:
        years = set.intersection(*(set(_fiscal_periods(t)) for t in group))
        for n, members in enumerate([group, *(tuple(t for t in group if t != drop) for drop in group)]):
            out += [(members, n, fiscal) for fiscal in sorted(years)]
    return out


def _peer_dates(members: Sequence[str], fiscal_years: Sequence[int]) -> list[str]:
    """30 days after the last of these 10-Ks, and around every restatement of any member's figures."""
    periods = {t: [_fiscal_periods(t)[y] for y in fiscal_years] for t in members}
    base = _days(max(_first_filed(t, p) for t, ps in periods.items() for p in ps), 30)
    return _group_dates(periods, ("revenue", "net_income"), base)


def _clear(build_question: Callable[..., Question], *args: str) -> Question | None:
    """The question, unless its pick is not clear (`_pick`) or its answer rounds to nothing."""
    try:
        q = build_question(*args)
    except ValueError:
        return None
    return q if abs(q.gold) >= MIN_CHANGE else None


def _probe_peer_ranks() -> list[Question]:
    """Each peer group and each three of it, asking for the highest or the lowest margin in turn."""
    out = []
    for members, n, fiscal in _peer_sets():
        rank = "highest" if (n + fiscal) % 2 == 0 else "lowest"
        for as_of in _peer_dates(members, [fiscal]):
            out.append(_clear(peer_margin_rank, ",".join(members), str(fiscal), rank, as_of))
    return [q for q in out if q]


def _probe_peer_change_ranks() -> list[Question]:
    """The largest margin expansion in each three of a peer group (four companies would be 16 figures), on one
    basis per company."""
    out = []
    for members, n, fiscal in _peer_sets():
        if not n or not all(fiscal - 1 in _fiscal_periods(t) for t in members):
            continue
        for as_of in _peer_dates(members, [fiscal, fiscal - 1]):
            pairs = {t: (_period(t, fiscal, as_of), _period(t, fiscal - 1, as_of)) for t in members}
            if all(_same_filing(t, ("revenue", "net_income"), *ps, as_of) for t, ps in pairs.items()):
                out.append(_clear(peer_margin_change_rank, ",".join(members), str(fiscal), as_of))
    return [q for q in out if q]


def _probe_vs_peers() -> list[Question]:
    """Each member of each peer group against the other three combined."""
    out = []
    for members, n, fiscal in _peer_sets():
        if n:
            continue
        for as_of in _peer_dates(members, [fiscal]):
            for t in members:
                peers = ",".join(p for p in members if p != t)
                out.append(_clear(margin_vs_peers, t, peers, str(fiscal), as_of))
    return [q for q in out if q]


def _probe_trends() -> list[Question]:
    """The last three years' trend, the day before each 10-K (the latest year is the previous one) and 30
    days after it, on one basis throughout; one question per distinct set of figures, at its latest date."""
    out: dict[tuple, Question] = {}
    concepts = ("revenue", "net_income")
    for ticker in data.companies():
        labelled = _fiscal_periods(ticker)
        for as_of in sorted({_days(f, n) for f in _filings(ticker)[1:] for n in (-1, 30)}):
            latest = fiscal_year(_latest_period(ticker, as_of))
            years = [latest - j for j in range(TREND_YEARS + 1)]
            # skip dates where the newest 10-K reports a year outside the fixture's window
            if not all(y in labelled for y in years) or _first_filed(ticker, labelled[latest]) != max(
                _filings_before(ticker, as_of)
            ):
                continue
            periods = [_period(ticker, y, as_of) for y in years]
            if not all(_same_filing(ticker, concepts, p, prior, as_of) for p, prior in pairwise(periods)):
                continue
            if q := _clear(margin_trend, ticker, as_of):
                out[(ticker, q.inputs)] = q
    return list(out.values())


def chain_questions() -> list[Question]:
    """The chained analyst questions (probe v2): peers ranked, a company against its peers, a trend. Where two
    dates see the same figures (restatements of different members filed days apart), only the first is asked."""
    out: dict[tuple, Question] = {}
    for q in _probe_peer_ranks() + _probe_peer_change_ranks() + _probe_vs_peers() + _probe_trends():
        args = tuple((k, v) for k, v in q.params.items() if k != "as_of")
        out.setdefault((q.kind, args, q.inputs), q)
    return list(out.values())


PROBE_SAMPLE_SEED = 20261006
PROBE_SAMPLE_PER_KIND = 20
CHAIN_KINDS = ("peer_margin_rank", "peer_margin_change_rank", "margin_vs_peers", "margin_trend")
ONE_HOP_KINDS = ("fy_net_margin", "latest_net_margin", "net_margin_change", "revenue_growth", "net_margin_gap")
# v2: weighted to the chained types, keeping four of each v1 type for continuity (100 tasks)
PROBE_V2_COUNTS = {**dict.fromkeys(CHAIN_KINDS, 20), **dict.fromkeys(ONE_HOP_KINDS, 4)}


def stratified_sample(qs: Sequence[Question], counts: dict[str, int], seed: int) -> list[Question]:
    """A seeded subset with `counts[kind]` questions of each kind, drawn by design and never from model results.

    Kinds are visited in sorted order and each kind's questions are ordered by prompt before
    sampling, so the subset depends only on the questions, `counts` and `seed`.
    """
    rng = random.Random(seed)
    picked: list[Question] = []
    for kind in sorted(counts):
        pool = sorted((q for q in qs if q.kind == kind), key=lambda q: q.prompt)
        picked += rng.sample(pool, min(counts[kind], len(pool)))
    return picked


def probe_sample(
    qs: Sequence[Question], per_kind: int = PROBE_SAMPLE_PER_KIND, seed: int = PROBE_SAMPLE_SEED
) -> list[Question]:
    """The v1 subset: `per_kind` questions of each kind present (frozen as tasks_probe100.py)."""
    return stratified_sample(qs, dict.fromkeys({q.kind for q in qs}, per_kind), seed)


def probe_sample_v2(qs: Sequence[Question], seed: int = PROBE_SAMPLE_SEED) -> list[Question]:
    """The v2 subset (`PROBE_V2_COUNTS`), drawn from the probe and chained questions together."""
    return stratified_sample(qs, PROBE_V2_COUNTS, seed)


def template_args(q: Question) -> dict[str, str]:
    """The arguments of the env template that poses `q`."""
    if q.params:
        return dict(q.params)
    if q.kind == "net_margin":
        return {"ticker": q.ticker, "period_end": q.period_end, "as_of": q.as_of}
    if q.kind == "latest_net_margin":
        return {"ticker": q.ticker, "as_of": q.as_of}
    fiscal = str(fiscal_year(q.period_end))
    if q.kind == "net_margin_gap":
        return {"ticker": q.ticker, "other": q.inputs[2].ticker, "fiscal_year": fiscal, "as_of": q.as_of}
    return {"ticker": q.ticker, "fiscal_year": fiscal, "as_of": q.as_of}
