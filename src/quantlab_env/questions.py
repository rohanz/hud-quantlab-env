"""The one question type: point-in-time net margin for a fiscal year.

"What was {company}'s net margin for the fiscal year ending {period_end},
using only filings available on {as_of}?"

It needs two lookups and one calculation, and the gold is computed from the
same point-in-time data the tools serve. Periods are named by their end date
rather than a fiscal-year label, because labels differ by company (NVIDIA's
fiscal 2024 ends in January 2024) and QuantLab's base model lost most of its
points on exactly that confusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from . import data


@dataclass(frozen=True)
class Question:
    ticker: str
    period_end: str
    as_of: str
    revenue: float
    net_income: float

    @property
    def gold(self) -> float:
        return self.net_income / self.revenue

    @property
    def prompt(self) -> str:
        name = data.companies()[self.ticker]["name"]
        return (
            f"What was {name} ({self.ticker})'s net margin (net income divided by revenue) "
            f"for the fiscal year ending {self.period_end}, using only filings available "
            f"on {self.as_of}?\n\n"
            "Use the tools to look up every figure and the compute tool for arithmetic. "
            "Any other number you mention must be exactly as a tool returned it or rounded "
            "from it. Give the margin as a percentage to two decimal places. End with "
            "exactly these three lines, citing the revenue and net income you used for that "
            "fiscal year exactly as get_fundamentals returned them: the whole value in USD "
            "with its sign, digits only (a leading $ and thousands commas are allowed; no rounding, no "
            "units):\n"
            "REVENUE: <revenue value>\n"
            "NET_INCOME: <net income value>\n"
            "ANSWER: <margin>%"
        )


def build(ticker: str, period_end: str, as_of: str) -> Question:
    """A question whose gold uses exactly what the tools would return on `as_of`."""
    values = {}
    for concept in ("revenue", "net_income"):
        rows = {r["period_end"]: r for r in data.facts_as_of(ticker, concept, as_of)}
        if period_end not in rows:
            raise ValueError(f"{ticker} {concept} {period_end} not filed by {as_of}")
        values[concept] = float(rows[period_end]["value"])
    return Question(ticker, period_end, as_of, values["revenue"], values["net_income"])


def default_questions() -> list[Question]:
    """One question per company-year, 30 days after the first 10-K for that year,
    plus the restatement traps: periods whose revenue changed in a later filing,
    asked again after the restatement."""
    questions = []
    for ticker, company in data.companies().items():
        facts = company["facts"]
        periods = sorted({r["period_end"] for r in facts["revenue"]} & {r["period_end"] for r in facts["net_income"]})
        for period_end in periods:
            if period_end < "2022-01-01":
                continue
            first_filed = max(
                min(r["filed"] for r in facts[c] if r["period_end"] == period_end) for c in ("revenue", "net_income")
            )
            as_of = (date.fromisoformat(first_filed) + timedelta(days=30)).isoformat()
            questions.append(build(ticker, period_end, as_of))
            values = {r["filed"]: r["value"] for r in facts["revenue"] if r["period_end"] == period_end}
            if len(set(values.values())) > 1:
                restated_on = max(values)
                later = (date.fromisoformat(restated_on) + timedelta(days=30)).isoformat()
                questions.append(build(ticker, period_end, later))
    return questions
