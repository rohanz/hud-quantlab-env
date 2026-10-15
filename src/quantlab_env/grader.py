"""Code-graded reward: correctness plus evidence for every number the answer cites.

    reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency

answer     The number on the final `ANSWER:` line is within 0.5% of the gold
           (as a fraction or a percent), or within 0.005 percentage points of
           it (the two decimals the prompt asks for), and the episode made at
           least one successful data lookup. No evidence, no credit. A
           ranking or yes/no question also needs its choice (a ticker, YES
           or NO) named on that line as a whole word, in any case, and no
           other of its choices.
grounding  All or nothing. The answer's citation block gives each of the
           question's inputs exactly as a lookup returned them, each under its
           own label, the final answer is a calculator result derived from
           them by the question's formula, and every other number cited in the
           answer traces to a successful tool result from this episode.
validity   Share of tool calls that were well formed, and 0 unless a lookup
           succeeded (so also 0 with no calls).
efficiency par / max(par, calls), and 0 unless a lookup succeeded, so a
           burst of failed calls earns nothing. Par is one lookup per
           (ticker, concept) the question needs plus one calculation, or one
           per figure a ranking or trend compares (PAR_CALLS = 3 for net
           margin).

The citation block is one labelled line per input before the `ANSWER:` line;
for net margin:

    REVENUE: 78740000000
    NET_INCOME: 20878000000

Each value must equal the input's row (its ticker, concept and period) of an
admissible lookup, and the value the gold used. Only formatting is allowed: a
leading `$` and thousands commas. No rounding, no scale words.

Evidence is built from the episode log in call order:
- `value` fields of successful `get_fundamentals` rows from lookups as of the
  question's date or earlier. A lookup past that date sees filings the
  question rules out, so it is neither evidence nor a successful lookup.
  Dates and other metadata are never evidence.
- the `value` of a successful `compute` call, if the expression is built from
  evidence: every literal equals evidence at its written precision or is a
  unit constant (1, 100, 1e3, 1e6, 1e9), a unit constant only scales evidence (`x * 100`, never `1 + 1`), both
  sides of `+` and `-` depend on a looked-up value (`x / x` does not), except
  growth: 1 may be subtracted from a ratio (`a / b - 1`), or 100 from it in
  percent (`a / b * 100 - 100`), and so does the result. A memorised number passed through the
  calculator, or rebuilt from unit constants, stays ungrounded.
- error results are never evidence, even if they echo the arguments.

Cited numbers and calculator literals match evidence at the precision they are
written with (`$78.74 billion` is a rounding of 78,740 million; `$20.885 billion`
is not 20,878 million). A cited zero matches only a zero, and a percent only a
calculator result, never a looked-up amount. Only the final answer's
correctness uses the 0.5% tolerance, against gold.

Each evidence item keeps its provenance: a formula over the looked-up values.
The final answer is grounded only if it matches an item whose formula is
proportional to the question's gold formula over its own inputs (net income
over revenue for the question's period, for net margin), so a correct number
merely touched by evidence does not pass.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache

from .data import companies
from .questions import Input, Question
from .tools import BINOPS, ToolCall

REL_TOL = 0.005
ABS_TOL = 0.00005  # 0.005 percentage points: the rounding of an answer given to two decimals
PAR_CALLS = 3  # net margin: revenue lookup, net income lookup, one compute
WEIGHTS = {"answer": 0.5, "grounding": 0.3, "validity": 0.1, "efficiency": 0.1}
UNIT_CONSTANTS = {1.0, 100.0, 1e3, 1e6, 1e9}
MALFORMED_CODES = {"bad_argument", "unknown_tool", "unknown_ticker", "unknown_concept"}

# fmt: off
_UNITS = {
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "bln": 1e9, "bil": 1e9, "b": 1e9,
    "million": 1e6, "millions": 1e6, "mm": 1e6, "mln": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
}
# fmt: on
_PERCENT = re.compile(r"%|pct|per\s*cent", re.IGNORECASE)
# Every dash an answer may write as a minus: hyphen-minus, U+2010 to U+2015
# (hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar),
# and the true minus signs U+2212, U+FE63 and U+FF0D.
_TRUE_MINUS = "\u2212\ufe63\uff0d"
_DASH = f"\\-\u2010-\u2015{_TRUE_MINUS}"
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
# A unit or currency word after a number, as _NUMBER reads it, plus "hundred": "June 30 billion" is an amount.
_AMOUNT_WORD = (
    r"\s*(?:%|percent\b|per\s?cent\b|pct\b|hundreds?\b|thousands?\b|millions?\b|billions?\b|trillions?\b"
    r"|tn\b|bn\b|bln\b|bil\b|mln\b|mn\b|mm\b|[tbmk]\b|dollars?\b|usd\b)"
)
_DATE = (
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?|(?<![$.,\d])\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?"
)
# A company the fixture names by a digit ("3M Co.", MMM): the exact name token, as the
# prompt writes it, is a name, not 3 million. "$3M", "3 M" and "3m" are still amounts.
_DIGIT_NAMES = sorted({n.split()[0] for n in (c["name"] for c in companies().values()) if n[:1].isdigit()})
# Text that looks numeric but is not a claim: dates, fiscal-year and quarter labels,
# form names (singular or plural), a company named by a digit, and numbered-list markers
# at the start of a line. A date followed by a unit word, or a day after a currency sign
# ("$9 may", "USD 9 may"), is an amount: the `amount` branch consumes a currency sign and
# its number first, and is kept.
_NOT_CLAIMS = re.compile(
    rf"(?P<amount>(?:\$|\busd\b)\s*[{_DASH}]?\s*\d+)"
    rf"|\b(?:{_DATE})\b(?!\.\d|{_AMOUNT_WORD})"
    r"|\bFY\s?\d{2,4}\b|\bQ[1-4]\b|\b(?:10-K|10-Q|8-K)(?:/A)?s?\b"
    rf"|(?-i:(?<![\w$.,])(?:{'|'.join(map(re.escape, _DIGIT_NAMES))})(?!\w))"
    r"|^\s*(?:[-*]\s*)?\(?\d{1,2}[.)](?=\s)",
    re.IGNORECASE | re.MULTILINE,
)
# A sign is a minus attached to the amount ("-$20.9 billion", "$-20.9", "$ - 20.9");
# a hyphen after a word or digit ("10-K", "2021-2022") is not. Parentheses are
# read as prose, not as accounting negatives.
_NUMBER = re.compile(
    rf"(?P<neg>(?<![\w.])[{_DASH}](?=(?:US)?\$|USD\b|\d))?(?:(?P<cur>(?:US)?\$|\bUSD\b)\s*(?P<neg2>[{_DASH}]\s*)?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|per[^\S\n]?cent\b|pct\b|trillions?\b|billions?\b|millions?\b|thousands?\b"
    r"|tn\b|bn\b|bln\b|bil\b|mln\b|mn\b|mm\b|[tbmk]\b)?(?P<cur2>\s*(?:dollars?|USD)\b)?",
    re.IGNORECASE,
)
# A minus spaced off from the amount that plainly prefixes it: after a word
# ("net income - $20.9 billion") or after : = ( , ; ("income: -  20,878"). After a
# digit it is a range or a subtraction ("2021 - 2022"), after a word with no space
# a word hyphen ("re- 20"), and a dash opening a line is a bullet. A true minus
# sign opening a line is still a sign. Spaced off, only a hyphen-minus or a true
# minus is a sign: a spaced en or em dash ("Revenue — $78.74 billion") is prose
# punctuation, which models write all the time.
_SPACED_MINUS = re.compile(
    rf"(?:(?<=[^\W\d_])[^\S\n]+|[:=(,;][^\S\n]*|^[^\S\n]*(?=[{_TRUE_MINUS}]))(?P<minus>[\-{_TRUE_MINUS}])[^\S\n]+\Z",
    re.MULTILINE,
)
# A multiply or divide operator right before a number: "* 100", "×100", "\times 100", "multiply by 100".
_OPERATOR = re.compile(
    r"(?:[*×·⋅/÷]|\\times|\\cdot|(?<![^\W\d_])x|\b(?:multipl(?:y|ied)|divided?)\s+by|\btimes)[^\S\n]*\Z",
    re.IGNORECASE,
)
_EXEMPTED = "#"  # stands in for exempt text, so "10-K - 2" is not read as a word before a minus
_COUNTED = re.compile(r"\s*[a-z]", re.IGNORECASE)  # "3 filings": a small count of something named
# "7 hundred million", "2023 dollar": a scale or currency word makes a count or a year an amount
_SCALE_WORD = re.compile(
    r"\s*(?:hundreds?|thousands?|millions?|billions?|trillions?|bn|bln|bil|mn|mln|mm|tn|dollars?|usd|cents?|bucks|grand)\b",
    re.IGNORECASE,
)
# The citation block: "REVENUE: <value>" and "NET_INCOME: <value>" lines, each a tool
# value written out whole in ASCII digits, optionally with a leading $ and thousands
# commas. Labels (these and ANSWER) are case-insensitive and open their line, after
# any markdown list, quote, heading or emphasis markers ("> - **REVENUE:** x"); the
# value may only be wrapped in * or _ emphasis ("**78,740,000,000**").
_LABELS = {"revenue": "REVENUE", "net_income": "NET_INCOME"}  # net margin's; other questions bring their own
_LABEL_LINE = r"^(?:[^\S\n]|[*_+>#\-])*({labels})[*_]*[^\S\n]*:[^\S\n]*[*_]*([^\n]*)$"
_ANSWER = re.compile(_LABEL_LINE.format(labels="ANSWER"), re.IGNORECASE | re.MULTILINE)
_TOOL_VALUE = re.compile(
    rf"[\s*_]*(?P<sign>[{_DASH}]?)\s*\$?\s*(?P<sign2>[{_DASH}]?)\s*(?P<num>[0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]+)[\s*_]*"
)


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
        millions, billions or a percent. A zero is exactly zero: "$0 trillion"
        is not a rounding of every amount under half a trillion.
        """
        v, h = self.value, self.ulp / 2
        if v == 0:
            return [(0.0, 0.0)]
        if self.kind == "percent":
            return [(v / 100, h / 100), (v, h)]
        if self.kind in ("scaled", "currency"):
            return [(v, h)]
        return [(v * s, h * s) for s in (1, 1e3, 1e6, 1e9)] + [(v / 100, h / 100)]


def _exempt(m: re.Match, kind: str, negative: bool, text: str) -> bool:
    """A bare year ("in 2023"), a small count ("3 filings") or a unit constant in a written
    formula ("NI / REV * 100", "* 100%"; only 100). Context, not value alone, decides: "$7", "7%",
    "debt 7,", "$2023", "7 hundred million" and "× 100 billion" are claims."""
    num = m["num"]
    if negative or _SCALE_WORD.match(text, m.end("num")):
        return False
    # Only 100: every larger unit constant could pass for an amount ("debt × 1,000,000,000").
    if (
        kind in ("plain", "percent")
        and float(num.replace(",", "")) == 100
        and _OPERATOR.search(text, max(0, m.start() - 24), m.start())
    ):
        return True
    if kind != "plain" or not num.isdigit():
        return False
    if len(num) == 4 and 1900 <= int(num) <= 2100:
        return True
    return int(num) < 10 and bool(_COUNTED.match(text, m.end("num")))


def extract_numbers(text: str) -> list[Cited]:
    text = _NOT_CLAIMS.sub(lambda m: m[0] if m["amount"] else _EXEMPTED, text)
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
        if _PERCENT.fullmatch(unit):
            kind = "percent"
        elif unit:
            kind, value, ulp = "scaled", value * _UNITS[unit], ulp * _UNITS[unit]
        else:
            kind = "currency" if m["cur"] or m["cur2"] else "plain"
        cited.append(Cited(text[start : m.end()].strip(), value, kind, _exempt(m, kind, negative, text), ulp))
    return cited


def answer_line(text: str) -> str | None:
    """What follows the label on the last `ANSWER:` line."""
    lines = _ANSWER.findall(text)
    return lines[-1][1] if lines else None


def final_answer(text: str) -> Cited | None:
    """The first number on the last `ANSWER:` line."""
    line = answer_line(text)
    if line is None:
        return None
    # rebuilt without markdown: the label has no digits, and its colon can precede a spaced minus
    nums = extract_numbers(f"ANSWER: {line}")
    return nums[0] if nums else None


def chosen(text: str, choices: Sequence[str]) -> list[str]:
    """The choices (tickers, YES or NO) the last `ANSWER:` line names, as whole words, in any case."""
    line = answer_line(text) or ""
    return [c for c in choices if re.search(rf"(?<![^\W_]){re.escape(c)}(?![^\W_])", line, re.IGNORECASE)]


@cache
def _citation(labels: tuple[str, ...]) -> re.Pattern:
    alternatives = "|".join(sorted(labels, key=len, reverse=True))
    return re.compile(_LABEL_LINE.format(labels=alternatives), re.IGNORECASE | re.MULTILINE)


def citations(text: str, labels: dict[str, str] | None = None) -> dict[str, int | None]:
    """The citation block: the last line of each label (by default `REVENUE:` and
    `NET_INCOME:`), each read as a whole tool value, or None if the line is missing or
    holds anything else. Keyed by input key."""
    labels = labels or _LABELS
    # the last line of each label wins
    lines = {label.upper(): value for label, value in _citation(tuple(labels.values())).findall(text)}
    cited = {}
    for concept, label in labels.items():
        m = _TOOL_VALUE.fullmatch(lines.get(label, ""))
        if not m or (m["sign"] and m["sign2"]):
            cited[concept] = None
            continue
        value = int(m["num"].replace(",", ""))
        cited[concept] = -value if m["sign"] or m["sign2"] else value
    return cited


def _close(a: float, b: float) -> bool:
    """Within the answer tolerance. Used only to judge the final answer against gold."""
    return math.isclose(a, b, rel_tol=REL_TOL) if b else a == 0


def _correct(final: Cited, gold: float) -> bool:
    """The answer as a percent, or a bare fraction or percent, within 0.5% of gold or
    within 0.005 percentage points (an answer rounded to two decimals; this only
    matters for golds under 1%, such as a small change in margin)."""
    readings = [final.value / 100] if final.kind == "percent" else [final.value]
    if final.kind == "plain":
        readings.append(final.value / 100)
    return any(_close(r, gold) or abs(r - gold) <= ABS_TOL + 1e-12 for r in readings)


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
    computed: bool = False  # a calculator result, not a looked-up value


@dataclass
class Evidence:
    sources: list[Source] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    @property
    def values(self) -> list[float]:
        return [item.value for item in self.items]

    @property
    def computed(self) -> list[float]:
        return [item.value for item in self.items if item.computed]

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


def _is_constant(node: ast.AST, value: float) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value == value


def _is_ratio(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)


def _growth_offset(node: ast.BinOp) -> bool:
    """Growth, the only offset by a constant: a ratio minus 1 ("a / b - 1"), or the ratio
    in percent minus 100 ("a / b * 100 - 100"). The caller checks the left side uses evidence."""
    if not isinstance(node.op, ast.Sub):
        return False
    left = node.left
    if _is_constant(node.right, 1):
        return _is_ratio(left)
    return (
        _is_constant(node.right, 100)
        and isinstance(left, ast.BinOp)
        and isinstance(left.op, ast.Mult)
        and _is_ratio(left.left)
        and _is_constant(left.right, 100)
    )


def _formula(node: ast.AST, expr: str, evidence: Evidence) -> tuple[Formula, bool]:
    """A compute expression as a formula over looked-up values, and whether it uses evidence.

    Raises Laundered if a literal is not evidence, unit constants are combined
    with each other, or `+`/`-` has a side that does not depend on a looked-up
    value (`x / x * 7e9 + x / y` is an invented 7e9 riding on a small evidence term).
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
        lused, rused = lused and _depends(left, evidence.base), rused and _depends(right, evidence.base)
        additive = isinstance(node.op, (ast.Add, ast.Sub))
        if (additive and not (lused and rused) and not (lused and _growth_offset(node))) or not (lused or rused):
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
                evidence.items.append(Item(float(call.result["value"]), formula, computed=True))
    return evidence


def _input_ids(evidence: Evidence, inp: Input, value: float) -> set[int]:
    return {
        i
        for i, s in enumerate(evidence.sources)
        if (s.ticker, s.concept, s.period_end, s.value) == (inp.ticker, inp.concept, inp.period_end, value)
    }


def derived_from_inputs(item: Item, evidence: Evidence, question: Question) -> bool:
    """Whether the item is proportional to the question's gold formula over its inputs.

    Each input is nudged in turn (every looked-up copy of it at once), and the
    item must respond as the gold formula does: for net margin, up 1% with net
    income and down by 1/1.01 with revenue.
    """
    ids = [_input_ids(evidence, inp, inp.value) for inp in question.inputs]
    if not all(ids):
        return False
    base, values = evidence.base, [inp.value for inp in question.inputs]
    at_base, gold = item.formula(base), question.formula(values)
    for k, inp_ids in enumerate(ids):
        nudged = [v * (1 + _NUDGE) if j == k else v for j, v in enumerate(values)]
        expected = at_base * question.formula(nudged) / gold if gold else 0.0
        if not math.isclose(_nudged(item.formula, base, inp_ids), expected, rel_tol=1e-9):
            return False
    return True


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
    # a ranking or a yes/no question: the ANSWER line must name its choice and no other
    named = chosen(answer, question.choices)
    chose = question.choice is None or named == [question.choice]
    correct = final is not None and _correct(final, question.gold) and chose

    claims = [c for c in extract_numbers(answer) if not c.exempt]
    # a percent is a ratio the calculator produced, never a looked-up amount
    ungrounded = [
        c.text for c in claims if not _matches(c, evidence.computed if c.kind == "percent" else evidence.values)
    ]
    cited = citations(answer, question.labels)
    missing_inputs = [
        inp.key
        for inp in question.inputs
        if (value := cited[inp.key]) is None or value != inp.value or not _input_ids(evidence, inp, float(value))
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
        "validity": (n - malformed) / n if looked_up else 0.0,
        "efficiency": question.par_calls / max(question.par_calls, n) if looked_up else 0.0,
    }
    reward = sum(WEIGHTS[k] * v for k, v in subscores.items())
    info = {
        "gold": question.gold,
        "final_answer": final.text if final else None,
        "correct": correct,
        "choice": question.choice,
        "chosen": named,
        "citations": cited,
        "cited": [c.text for c in claims],
        "ungrounded": ungrounded,
        "missing_inputs": missing_inputs,
        "answer_derived": answer_derived,
        "tool_calls": n,
        "malformed_calls": malformed,
        "future_lookups": future,
    }
    return Grade(round(reward, 6), subscores, info)
