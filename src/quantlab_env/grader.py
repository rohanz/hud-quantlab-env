"""Code-graded reward: correctness plus evidence for every number the answer cites.

    reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency

answer     The number on the final `ANSWER:` line is within 0.5% of the gold
           margin (as a fraction or a percent), and the episode made at least
           one successful data lookup. No evidence, no credit.
grounding  All or nothing. Every number cited anywhere in the answer, the
           final answer included, must trace to a successful tool result from
           this episode, and the answer must cite the revenue and net income
           the margin was computed from.
validity   Share of tool calls that were well formed (0 with no calls).
efficiency PAR_CALLS / max(PAR_CALLS, calls), and 0 unless a lookup
           succeeded, so a burst of failed calls earns nothing.

Evidence is built from the episode log in call order:
- `value` fields of successful `get_fundamentals` rows from lookups as of the
  question's date or earlier. A lookup past that date sees filings the
  question rules out, so it is neither evidence nor a successful lookup.
  Dates and other metadata are never evidence.
- the `value` of a successful `compute` call, if the expression is built from
  evidence: every literal is evidence or a unit constant (1, 100, 1e3, 1e6,
  1e9), a unit constant only scales evidence (`x * 100`, never `1 + 1`), both
  sides of `+` and `-` are evidence, and the result actually depends on a
  looked-up value (`x / x` does not). A memorised number passed through the
  calculator, or rebuilt from unit constants, stays ungrounded.
- error results are never evidence, even if they echo the arguments.

Each evidence item keeps its provenance: a formula over the looked-up values.
The final answer is grounded only if it matches an item whose formula is
proportional to net income over revenue for the question's own period, so a
correct number merely touched by evidence does not pass.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .questions import Question
from .tools import BINOPS, ToolCall

REL_TOL = 0.005
PAR_CALLS = 3  # revenue lookup, net income lookup, one compute
WEIGHTS = {"answer": 0.5, "grounding": 0.3, "validity": 0.1, "efficiency": 0.1}
UNIT_CONSTANTS = {1.0, 100.0, 1e3, 1e6, 1e9}
MALFORMED_CODES = {"bad_argument", "unknown_tool", "unknown_ticker", "unknown_concept"}

# fmt: off
_UNITS = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
# fmt: on
# Text that looks numeric but is not a claim: ISO dates, fiscal-year labels, form names.
_NOT_CLAIMS = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\bFY\s?\d{2,4}\b|\b(?:10-K|10-Q|8-K)(?:/A)?\b", re.IGNORECASE)
# A sign is a minus sign attached to the amount ("-$20.9 billion", "$-20.9"); a
# hyphen after a word or digit ("10-K", "2021-2022") is not. Parentheses are
# read as prose, not as accounting negatives.
_NUMBER = re.compile(
    r"(?P<neg>(?<![\w.])[-−])?(?P<cur>(?:US)?\$|\bUSD\b)?\s*(?P<neg2>[-−])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b"
    r"|tn\b|bn\b|mm\b|[tbmk]\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cited:
    text: str
    value: float
    kind: str  # "percent", "scaled" (unit word applied), "currency" (a bare $ amount) or "plain"

    def readings(self) -> list[float]:
        """Signed values this citation may stand for.

        A unit word or a currency sign fixes the scale: "$78.74" is 78.74
        dollars, not 78.74 billion. Only a bare number may stand for thousands,
        millions, billions or a percent.
        """
        v = self.value
        if self.kind == "percent":
            return [v / 100, v]
        if self.kind in ("scaled", "currency"):
            return [v]
        return [v, v * 1e3, v * 1e6, v * 1e9, v / 100]

    @property
    def exempt(self) -> bool:
        """Bare years and list markers are not numeric claims."""
        if self.kind != "plain" or not float(self.value).is_integer() or "." in self.text:
            return False
        v = abs(self.value)
        return v < 10 or (1900 <= v <= 2100 and "," not in self.text)


def extract_numbers(text: str) -> list[Cited]:
    text = _NOT_CLAIMS.sub(" ", text)
    cited = []
    for m in _NUMBER.finditer(text):
        value = float(m["num"].replace(",", ""))
        if m["neg"] or m["neg2"]:
            value = -value
        unit = (m["unit"] or "").lower()
        if unit in ("%", "percent"):
            kind = "percent"
        elif unit:
            kind, value = "scaled", value * _UNITS[unit]
        else:
            kind = "currency" if m["cur"] else "plain"
        cited.append(Cited(m.group(0).strip(), value, kind))
    return cited


def final_answer(text: str) -> Cited | None:
    """The first number on the last `ANSWER:` line."""
    lines = [ln for ln in text.splitlines() if re.match(r"\s*ANSWER\s*:", ln, re.IGNORECASE)]
    if not lines:
        return None
    nums = extract_numbers(lines[-1].split(":", 1)[1])
    return nums[0] if nums else None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL) if b else a == 0


def _matches(cited: Cited, evidence: list[float]) -> bool:
    """Signed comparison: a claimed loss does not match a reported profit."""
    return any(_close(r, e) for r in cited.readings() for e in evidence)


Formula = Callable[[Sequence[float]], float]  # recomputes a value from the looked-up values
_NUDGE = 0.01  # relative perturbation used to test what a formula depends on


class Laundered(ValueError):
    """A calculator expression that is not built from evidence."""


@dataclass(frozen=True)
class Source:
    """One looked-up value: a row of a successful `get_fundamentals` call."""

    ticker: str
    concept: str
    period_end: str
    value: float


@dataclass(frozen=True)
class Item:
    value: float
    formula: Formula


@dataclass
class Evidence:
    sources: list[Source] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    @property
    def values(self) -> list[float]:
        return [item.value for item in self.items]

    @property
    def base(self) -> list[float]:
        return [s.value for s in self.sources]

    def add_source(self, source: Source) -> None:
        i = len(self.sources)
        self.sources.append(source)
        self.items.append(Item(source.value, lambda v: v[i]))

    def bind(self, literal: float) -> Formula:
        """The formula of the first item a calculator literal matches, rescaled to the literal.

        Literals are unsigned (a minus is a separate operator), so they bind by magnitude.
        """
        cited = Cited(str(literal), literal, "plain")
        for item in self.items:
            if item.value and _matches(cited, [abs(item.value)]):
                ratio = literal / item.value
                return lambda v, f=item.formula: f(v) * ratio
        raise Laundered(f"{literal} is not evidence")


def _formula(node: ast.AST, evidence: Evidence) -> tuple[Formula, bool]:
    """A compute expression as a formula over looked-up values, and whether it uses evidence.

    Raises Laundered if a literal is not evidence, unit constants are combined
    with each other, or `+`/`-` has a side without evidence.
    """
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if float(node.value) in UNIT_CONSTANTS:
            return (lambda v, c=float(node.value): c), False
        return evidence.bind(float(node.value)), True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        f, used = _formula(node.operand, evidence)
        return ((lambda v: -f(v)) if isinstance(node.op, ast.USub) else f), used
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        (left, lused), (right, rused) = _formula(node.left, evidence), _formula(node.right, evidence)
        additive = isinstance(node.op, (ast.Add, ast.Sub))
        if (additive and not (lused and rused)) or not (lused or rused):
            raise Laundered("unit constants may only scale evidence")
        op = BINOPS[type(node.op)]
        return (lambda v: op(left(v), right(v))), True
    raise Laundered("not literal arithmetic")


def _nudged(formula: Formula, base: list[float], ids: set[int]) -> float:
    try:
        return formula([x * (1 + _NUDGE) if i in ids else x for i, x in enumerate(base)])
    except ZeroDivisionError:
        return math.inf


def _depends(formula: Formula, base: list[float]) -> bool:
    """Whether the formula changes when any single looked-up value does."""
    at_base = formula(base)
    return any(not math.isclose(_nudged(formula, base, {i}), at_base, rel_tol=1e-9) for i in range(len(base)))


def admissible_lookup(call: ToolCall, cutoff: str) -> bool:
    """A successful lookup that only saw filings available on the question's date."""
    return call.ok and call.name == "get_fundamentals" and call.result["as_of"] <= cutoff


def build_evidence(calls: list[ToolCall], cutoff: str) -> Evidence:
    evidence = Evidence()
    for call in calls:
        if not call.ok:
            continue
        if admissible_lookup(call, cutoff):
            r = call.result
            for row in r["rows"]:
                evidence.add_source(Source(r["ticker"], r["concept"], row["period_end"], float(row["value"])))
        elif call.name == "compute":
            try:
                formula, _ = _formula(ast.parse(call.result["expr"], mode="eval").body, evidence)
            except Laundered:
                continue
            if _depends(formula, evidence.base):
                evidence.items.append(Item(float(call.result["value"]), formula))
    return evidence


def _input_ids(evidence: Evidence, question: Question, concept: str, value: float) -> set[int]:
    return {
        i
        for i, s in enumerate(evidence.sources)
        if (s.ticker, s.concept, s.period_end, s.value) == (question.ticker, concept, question.period_end, value)
    }


def derived_from_inputs(item: Item, evidence: Evidence, question: Question) -> bool:
    """Whether the item is proportional to net income / revenue for the question's period."""
    ni = _input_ids(evidence, question, "net_income", question.net_income)
    rev = _input_ids(evidence, question, "revenue", question.revenue)
    if not (ni and rev):
        return False
    base = evidence.base
    at_base = item.formula(base)
    return math.isclose(_nudged(item.formula, base, ni), at_base * (1 + _NUDGE), rel_tol=1e-9) and math.isclose(
        _nudged(item.formula, base, rev), at_base / (1 + _NUDGE), rel_tol=1e-9
    )


@dataclass
class Grade:
    reward: float
    subscores: dict[str, float]
    info: dict = field(default_factory=dict)


def grade(answer: str | None, calls: list[ToolCall], question: Question) -> Grade:
    answer = answer or ""
    evidence = build_evidence(calls, question.as_of)
    looked_up = any(admissible_lookup(c, question.as_of) for c in calls)
    future = sum(1 for c in calls if c.ok and c.name == "get_fundamentals" and not admissible_lookup(c, question.as_of))

    final = final_answer(answer)
    correct = final is not None and (
        _close(final.value / 100 if final.kind == "percent" else final.value, question.gold)
        or (final.kind == "plain" and _close(final.value, question.gold * 100))
    )

    claims = [c for c in extract_numbers(answer) if not c.exempt]
    ungrounded = [c.text for c in claims if not _matches(c, evidence.values)]
    missing_inputs = [
        name
        for name, value in (("revenue", question.revenue), ("net_income", question.net_income))
        if not any(_matches(c, [value]) for c in claims)
    ]
    answer_derived = final is not None and any(
        _matches(final, [item.value]) and derived_from_inputs(item, evidence, question) for item in evidence.items
    )
    grounded = answer_derived and not ungrounded and not missing_inputs

    n = len(calls)
    malformed = sum(1 for c in calls if c.result.get("error", {}).get("code") in MALFORMED_CODES)
    subscores = {
        "answer": float(correct and looked_up),
        "grounding": float(grounded),
        "validity": (n - malformed) / n if n else 0.0,
        "efficiency": PAR_CALLS / max(PAR_CALLS, n) if looked_up else 0.0,
    }
    reward = sum(WEIGHTS[k] * v for k, v in subscores.items())
    info = {
        "gold": question.gold,
        "final_answer": final.text if final else None,
        "correct": correct,
        "cited": [c.text for c in claims],
        "ungrounded": ungrounded,
        "missing_inputs": missing_inputs,
        "answer_derived": answer_derived,
        "tool_calls": n,
        "malformed_calls": malformed,
        "future_lookups": future,
    }
    return Grade(round(reward, 6), subscores, info)
