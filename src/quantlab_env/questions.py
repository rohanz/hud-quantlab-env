"""Question types: point-in-time ratios over annual revenue and net income.

Every question names the figures it needs (its `inputs`: a ticker, a concept and
a fiscal period, each cited in the answer under its own label), a gold formula
over those figures, and the prompt. The gold uses exactly what the tools serve
on the question's date (`data.facts_as_of`), so restatements filed later are
invisible to it.

Two task sets:

- the legacy set (`default_questions`): 18 net-margin questions that name the
  fiscal period by its end date. QuantLab's base model lost most of its points
  on fiscal-year labels, so these avoided them.
- the probe set (`probe_questions`): harder questions derived from the
  fixture. They name fiscal years by label ("fiscal 2023") as the company
  does, ask for the latest year filed by a date, straddle restatements, and
  combine two years or two companies.

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
        """One lookup per (ticker, concept) needed, and one calculation."""
        return len({(i.ticker, i.concept) for i in self.inputs}) + 1

    @property
    def prompt(self) -> str:
        n = len(self.inputs)
        count = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n + 1, str(n + 1))
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
            f"units):\n{block}ANSWER: <{self.placeholder}>{self.unit}"
        )


def _describe(i: Input) -> str:
    """The placeholder in the prompt's citation block: "revenue value", "prior net income value"."""
    words = i.key.split("_")
    if words[0] == i.ticker.lower():
        return f"{i.ticker} {_CONCEPT_NAMES[i.concept]} value"
    prefix = "prior " if words[0] == "prior" else ""
    return f"{prefix}{_CONCEPT_NAMES[i.concept]} value"


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
    changes = [_days(d, 30) for d in _restated(ticker, periods, concepts)]
    first = _days(max(_first_filed(ticker, p) for p in periods), 30)
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


PROBE_SAMPLE_SEED = 20261006
PROBE_SAMPLE_PER_KIND = 20


def probe_sample(
    qs: Sequence[Question], per_kind: int = PROBE_SAMPLE_PER_KIND, seed: int = PROBE_SAMPLE_SEED
) -> list[Question]:
    """A seeded subset with `per_kind` questions of each kind, drawn by design and never from model results.

    Kinds are visited in sorted order and each kind's questions are ordered by prompt before
    sampling, so the subset depends only on the fixture, `per_kind` and `seed`.
    """
    rng = random.Random(seed)
    picked: list[Question] = []
    for kind in sorted({q.kind for q in qs}):
        pool = sorted((q for q in qs if q.kind == kind), key=lambda q: q.prompt)
        picked += rng.sample(pool, min(per_kind, len(pool)))
    return picked


def template_args(q: Question) -> dict[str, str]:
    """The arguments of the env template that poses `q`."""
    if q.kind == "net_margin":
        return {"ticker": q.ticker, "period_end": q.period_end, "as_of": q.as_of}
    if q.kind == "latest_net_margin":
        return {"ticker": q.ticker, "as_of": q.as_of}
    fiscal = str(fiscal_year(q.period_end))
    if q.kind == "net_margin_gap":
        return {"ticker": q.ticker, "other": q.inputs[2].ticker, "fiscal_year": fiscal, "as_of": q.as_of}
    return {"ticker": q.ticker, "fiscal_year": fiscal, "as_of": q.as_of}
