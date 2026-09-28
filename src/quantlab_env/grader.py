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
- `value` fields of successful `get_fundamentals` rows. Dates and other
  metadata are never evidence.
- the `value` of a successful `compute` call, but only if every literal in
  its expression is already evidence (or a unit constant like 100). A
  memorised number passed through the calculator stays ungrounded.
- error results are never evidence, even if they echo the arguments.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

from .questions import Question
from .tools import ToolCall

REL_TOL = 0.005
PAR_CALLS = 3  # revenue lookup, net income lookup, one compute
WEIGHTS = {"answer": 0.5, "grounding": 0.3, "validity": 0.1, "efficiency": 0.1}
UNIT_CONSTANTS = {1.0, 100.0, 1e3, 1e6, 1e9}
MALFORMED_CODES = {"bad_argument", "unknown_tool", "unknown_ticker", "unknown_concept"}

_UNITS = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
# Text that looks numeric but is not a claim: ISO dates, fiscal-year labels, form names.
_NOT_CLAIMS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\bFY\s?\d{2,4}\b|\b(?:10-K|10-Q|8-K)(?:/A)?\b", re.IGNORECASE
)
_NUMBER = re.compile(
    r"(?P<neg>[-−(])?\s*\$?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\)?\s*(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b"
    r"|tn\b|bn\b|mm\b|[tbmk]\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cited:
    text: str
    value: float
    kind: str  # "percent", "scaled" (unit word applied) or "plain"

    def readings(self) -> list[float]:
        """Values this citation may stand for, as magnitudes."""
        v = abs(self.value)
        if self.kind == "percent":
            return [v / 100, v]
        if self.kind == "scaled":
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
        if m["neg"] in ("-", "−", "("):
            value = -value
        unit = (m["unit"] or "").lower()
        if unit in ("%", "percent"):
            kind = "percent"
        elif unit:
            kind, value = "scaled", value * _UNITS[unit]
        else:
            kind = "plain"
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
    return any(_close(r, abs(e)) for r in cited.readings() for e in evidence)


def _literals(expr: str) -> list[float]:
    return [
        abs(float(node.value))
        for node in ast.walk(ast.parse(expr, mode="eval"))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float)
    ]


def build_evidence(calls: list[ToolCall]) -> list[float]:
    evidence: list[float] = []
    for call in calls:
        if not call.ok:
            continue
        if call.name == "get_fundamentals":
            evidence += [float(row["value"]) for row in call.result["rows"]]
        elif call.name == "compute":
            literals = [Cited(str(x), x, "plain") for x in _literals(call.result["expr"])]
            if all(x.value in UNIT_CONSTANTS or _matches(x, evidence) for x in literals):
                evidence.append(float(call.result["value"]))
    return evidence


@dataclass
class Grade:
    reward: float
    subscores: dict[str, float]
    info: dict = field(default_factory=dict)


def grade(answer: str | None, calls: list[ToolCall], question: Question) -> Grade:
    answer = answer or ""
    evidence = build_evidence(calls)
    looked_up = any(c.ok and c.name == "get_fundamentals" for c in calls)

    final = final_answer(answer)
    correct = final is not None and (
        _close(final.value / 100 if final.kind == "percent" else final.value, question.gold)
        or (final.kind == "plain" and _close(final.value, question.gold * 100))
    )

    claims = [c for c in extract_numbers(answer) if not c.exempt]
    ungrounded = [c.text for c in claims if not _matches(c, evidence)]
    missing_inputs = [
        name
        for name, value in (("revenue", question.revenue), ("net_income", question.net_income))
        if not any(_matches(c, [value]) for c in claims)
    ]
    grounded = (
        final is not None
        and _matches(final, evidence)
        and not ungrounded
        and not missing_inputs
    )

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
        "tool_calls": n,
        "malformed_calls": malformed,
    }
    return Grade(round(reward, 6), subscores, info)
