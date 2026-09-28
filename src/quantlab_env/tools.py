"""The agent's tools, with strict argument checking and a per-episode call log.

Every call goes through `execute`, which records the raw arguments and the
result. The grader reads that log as the episode's evidence, so a call that
never reached the log could never count for or against the agent.

Arguments are checked strictly and never repaired: an unknown key, a missing
key or a value of the wrong type is a `bad_argument` error the agent sees and
the grader counts. QuantLab's GRPO run trained under a parser that silently
dropped hallucinated keys while evaluation rejected them; the model learned
the lenient contract and lost points where the two disagreed. One strict
contract, used everywhere, avoids that.
"""

from __future__ import annotations

import ast
import inspect
import operator
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import data

MAX_EXPR_CHARS = 200


def get_fundamentals(ticker: str, concept: str, as_of: str) -> dict:
    """Annual values of one concept for a company, as known on `as_of`.

    ticker: e.g. "AAPL". concept: "revenue" or "net_income".
    as_of: ISO date YYYY-MM-DD; only filings made on or before it are visible,
    and for each fiscal period the most recent such filing wins.
    Returns rows of {period_end, value (USD), filed, form}.
    """
    if ticker not in data.companies():
        return _error("unknown_ticker", f"no company {ticker!r}; known: {sorted(data.companies())}")
    if concept not in data.concepts():
        return _error("unknown_concept", f"no concept {concept!r}; known: {data.concepts()}")
    if not _is_iso_date(as_of):
        return _error("bad_argument", "as_of must be an ISO date YYYY-MM-DD")
    rows = data.facts_as_of(ticker, concept, as_of)
    if not rows:
        return _error("no_data", f"nothing filed for {ticker} {concept} on or before {as_of}")
    return {
        "ticker": ticker,
        "concept": concept,
        "as_of": as_of,
        "rows": [
            {k: r[k] for k in ("period_end", "value", "filed", "form")} for r in rows
        ],
    }


def compute(expr: str) -> dict:
    """Evaluate arithmetic on literal numbers only: + - * / and parentheses.

    Names, function calls and powers are rejected, so a remembered figure can
    only enter as a literal, where the grader can check it against evidence.
    """
    if len(expr) > MAX_EXPR_CHARS:
        return _error("bad_argument", f"expression longer than {MAX_EXPR_CHARS} characters")
    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return _error("bad_argument", "division by zero")
    except (SyntaxError, ValueError):
        return _error("bad_argument", "only numbers, + - * / and parentheses are allowed")
    return {"expr": expr, "value": value}


TOOLS = {f.__name__: f for f in (get_fundamentals, compute)}

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("disallowed expression")


def _is_iso_date(s: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False
    try:
        date.fromisoformat(s)
    except ValueError:
        return False
    return True


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


@dataclass
class ToolCall:
    name: str
    arguments: Any
    result: dict

    @property
    def ok(self) -> bool:
        return "error" not in self.result


@dataclass
class Episode:
    """Every tool call made since the task started, in order."""

    calls: list[ToolCall] = field(default_factory=list)

    def reset(self) -> None:
        self.calls.clear()


def validate(name: str, arguments: Any) -> dict | None:
    """Strict check of raw arguments against the tool's signature; None if valid."""
    if name not in TOOLS:
        return _error("unknown_tool", f"no tool {name!r}; available: {sorted(TOOLS)}")
    if not isinstance(arguments, dict):
        return _error("bad_argument", "arguments must be a JSON object")
    params = inspect.signature(TOOLS[name]).parameters
    unknown = sorted(set(arguments) - set(params))
    if unknown:
        return _error("bad_argument", f"unknown argument(s) {unknown}; expected {list(params)}")
    missing = [p for p in params if p not in arguments]
    if missing:
        return _error("bad_argument", f"missing argument(s) {missing}")
    for key, value in arguments.items():
        if type(value) is not str:  # every parameter is a string; no coercion
            return _error("bad_argument", f"{key} must be a string")
    return None


def execute(name: str, arguments: Any, episode: Episode) -> dict:
    """Validate, run and log one tool call. Never raises for bad input."""
    result = validate(name, arguments) or TOOLS[name](**arguments)
    episode.calls.append(ToolCall(name, arguments, result))
    return result
