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
  evidence: every literal equals evidence at its written precision or is a
  unit constant (1, 100, 1e3, 1e6, 1e9), a unit constant only scales evidence (`x * 100`, never `1 + 1`), both
  sides of `+` and `-` are evidence, and the result actually depends on a
  looked-up value (`x / x` does not). A memorised number passed through the
  calculator, or rebuilt from unit constants, stays ungrounded.
- error results are never evidence, even if they echo the arguments.

Cited numbers and calculator literals match evidence at the precision they are
written with (`$78.74 billion` is a rounding of 78,740 million; `$20.885 billion`
is not 20,878 million). Only the final answer's correctness uses the 0.5%
tolerance, against gold.

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
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
# Text that looks numeric but is not a claim: dates, fiscal-year and quarter labels,
# form names, and numbered-list markers at the start of a line.
_NOT_CLAIMS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"
    rf"|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b|\b\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?\b"
    r"|\bFY\s?\d{2,4}\b|\bQ[1-4]\b|\b(?:10-K|10-Q|8-K)(?:/A)?\b"
    r"|^\s*(?:[-*]\s*)?\(?\d{1,2}[.)](?=\s)",
    re.IGNORECASE | re.MULTILINE,
)
# A sign is a minus attached to the amount ("-$20.9 billion", "$-20.9", "$ - 20.9");
# a hyphen after a word or digit ("10-K", "2021-2022") is not. Parentheses are
# read as prose, not as accounting negatives.
_NUMBER = re.compile(
    r"(?P<neg>(?<![\w.])[-−](?=(?:US)?\$|USD\b|\d))?(?:(?P<cur>(?:US)?\$|\bUSD\b)\s*(?P<neg2>[-−]\s*)?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b"
    r"|tn\b|bn\b|mm\b|[tbmk]\b)?(?P<cur2>\s*(?:dollars|USD)\b)?",
    re.IGNORECASE,
)
# A minus spaced off from the amount that plainly prefixes it: after a word
# ("net income - $20.9 billion") or after : = ( , ; ("income: -  20,878"). After a
# digit it is a range or a subtraction ("2021 - 2022"), after a word with no space
# a word hyphen ("re- 20"), and a hyphen opening a line is a bullet. A true minus
# sign (U+2212) opening a line is still a sign.
_SPACED_MINUS = re.compile(
    r"(?:(?<=[^\W\d_])[^\S\n]+|[:=(,;][^\S\n]*|^[^\S\n]*(?=−))(?P<minus>[-−])[^\S\n]+\Z",
    re.MULTILINE,
)
_EXEMPTED = "#"  # stands in for exempt text, so "10-K - 2" is not read as a word before a minus
_COUNTED = re.compile(r"\s*[a-z]", re.IGNORECASE)  # "3 filings": a small count of something named


def _ulp(digits: str) -> float:
    """One unit in the last written place: "78.74" is 0.01, "20,878" is 1, "2.0878e10" is 1e6."""
    mantissa, _, exponent = digits.lower().replace(",", "").replace("_", "").partition("e")
    return 10.0 ** (int(exponent or 0) - len(mantissa.partition(".")[2]))


@dataclass(frozen=True)
class Cited:
    text: str
    value: float
    kind: str  # "percent", "scaled" (unit word applied), "currency" (a bare $ amount) or "plain"
    exempt: bool = False  # a year or a small count, by context; never an amount with a unit or currency
    ulp: float = 0.0  # one unit in the last written place, in the same units as value

    def readings(self) -> list[tuple[float, float]]:
        """Signed values this citation may stand for, each with its rounding half-width.

        A unit word or a currency sign fixes the scale: "$78.74" is 78.74
        dollars, not 78.74 billion. Only a bare number may stand for thousands,
        millions, billions or a percent.
        """
        v, h = self.value, self.ulp / 2
        if self.kind == "percent":
            return [(v / 100, h / 100), (v, h)]
        if self.kind in ("scaled", "currency"):
            return [(v, h)]
        return [(v * s, h * s) for s in (1, 1e3, 1e6, 1e9)] + [(v / 100, h / 100)]


def _exempt(m: re.Match, kind: str, negative: bool, text: str) -> bool:
    """A bare year ("in 2023") or a small count ("3 filings"). Context, not value alone,
    decides: "$7", "7%", "debt 7," and "$2023" are claims."""
    num = m["num"]
    if kind != "plain" or negative or not num.isdigit():
        return False
    if len(num) == 4 and 1900 <= int(num) <= 2100:
        return True
    return int(num) < 10 and bool(_COUNTED.match(text, m.end("num")))


def extract_numbers(text: str) -> list[Cited]:
    text = _NOT_CLAIMS.sub(_EXEMPTED, text)
    cited = []
    for m in _NUMBER.finditer(text):
        start, negative = m.start(), bool(m["neg"] or m["neg2"])
        if not negative and (spaced := _SPACED_MINUS.search(text, max(0, start - 64), start)):
            start, negative = spaced.start("minus"), True
        value = float(m["num"].replace(",", ""))
        if negative:
            value = -value
        ulp = _ulp(m["num"])
        unit = (m["unit"] or "").lower()
        if unit in ("%", "percent"):
            kind = "percent"
        elif unit:
            kind, value, ulp = "scaled", value * _UNITS[unit], ulp * _UNITS[unit]
        else:
            kind = "currency" if m["cur"] or m["cur2"] else "plain"
        cited.append(Cited(text[start : m.end()].strip(), value, kind, _exempt(m, kind, negative, text), ulp))
    return cited


def final_answer(text: str) -> Cited | None:
    """The first number on the last `ANSWER:` line."""
    lines = [ln for ln in text.splitlines() if re.match(r"\s*ANSWER\s*:", ln, re.IGNORECASE)]
    if not lines:
        return None
    nums = extract_numbers(lines[-1])  # the label has no digits; its colon can precede a spaced minus
    return nums[0] if nums else None


def _close(a: float, b: float) -> bool:
    """Within the answer tolerance. Used only to judge the final answer against gold."""
    return math.isclose(a, b, rel_tol=REL_TOL) if b else a == 0


def _rounds_to(reading: float, half: float, e: float) -> bool:
    """Whether `e` rounds to `reading` at the written precision (half a unit in the last place)."""
    return abs(reading - e) <= half + 1e-9 * max(abs(reading), abs(e))


def _matches(cited: Cited, evidence: list[float]) -> bool:
    """Signed comparison at the written precision: "$78.74 billion" matches
    78,740 million, but "$20.885 billion" does not match 20,878 million, and a
    claimed loss does not match a reported profit."""
    return any(_rounds_to(r, h, e) for r, h in cited.readings() for e in evidence)


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

    def bind(self, text: str) -> Formula:
        """The formula of the item a calculator literal matches most closely, rescaled to the literal.

        The literal must equal the item at the precision it is written with, so
        `20878` binds to 20,878,000,000 but `20885` binds to nothing. Closest,
        not first, when a coarse literal rounds from several items. Literals are
        unsigned (a minus is a separate operator), so they bind by magnitude.
        """
        try:
            literal = float(text)
        except ValueError:  # 0x51, 0o17: not written as a decimal amount
            raise Laundered(f"{text} is not a decimal literal") from None
        readings = Cited(text, literal, "plain", ulp=_ulp(text)).readings()
        candidates = [
            (abs(r - abs(item.value)) / abs(item.value), item)
            for item in self.items
            for r, h in readings
            if item.value and _rounds_to(r, h, abs(item.value))
        ]
        if not candidates:
            raise Laundered(f"{literal} is not evidence")
        item = min(candidates, key=lambda c: c[0])[1]
        ratio = literal / item.value
        return lambda v: item.formula(v) * ratio


def _formula(node: ast.AST, expr: str, evidence: Evidence) -> tuple[Formula, bool]:
    """A compute expression as a formula over looked-up values, and whether it uses evidence.

    Raises Laundered if a literal is not evidence, unit constants are combined
    with each other, or `+`/`-` has a side without evidence.
    """
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if float(node.value) in UNIT_CONSTANTS:
            return (lambda v, c=float(node.value): c), False
        return evidence.bind(ast.get_source_segment(expr, node)), True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        f, used = _formula(node.operand, expr, evidence)
        return ((lambda v: -f(v)) if isinstance(node.op, ast.USub) else f), used
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        (left, lused), (right, rused) = _formula(node.left, expr, evidence), _formula(node.right, expr, evidence)
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
            expr = call.result["expr"]
            try:
                formula, _ = _formula(ast.parse(expr, mode="eval").body, expr, evidence)
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
