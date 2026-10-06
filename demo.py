"""Run scripted agents through the environment end to end, offline.

    uv run python demo.py

Each agent is a real HUD `Agent`: it opens the env's `tools` MCP capability,
calls tools over HTTP, and submits an answer that the task template grades.
No model is called. The agents replay the honest strategy and the failure
modes the reward has to handle.
"""

from __future__ import annotations

import asyncio
import re

from hud.agents.base import Agent
from hud.eval import LocalRuntime

from env import close_tool_server, env, net_margin
from quantlab_env.questions import Question


def _parse(prompt: str) -> tuple[str, str, str]:
    ticker = re.search(r"\(([A-Z]+)\)", prompt)[1]
    period_end, as_of = re.findall(r"\d{4}-\d{2}-\d{2}", prompt)[:2]
    return ticker, period_end, as_of


class Scripted(Agent):
    """Looks up revenue and net income, then answers according to `style`."""

    def __init__(self, style: str) -> None:
        self.style = style

    async def __call__(self, run) -> None:
        ticker, period_end, as_of = _parse(run.prompt_text)
        mcp = await run.client.open("tools")

        async def call(name: str, **args) -> dict:
            return (await mcp.call_tool(name, args)).structuredContent

        async def lookup(concept: str) -> float:
            rows = (await call("get_fundamentals", ticker=ticker, concept=concept, as_of=as_of))["rows"]
            return next(r["value"] for r in rows if r["period_end"] == period_end)

        if self.style == "memorised":  # right answer, no tool calls at all
            run.trace.content = f"ANSWER: {self.remembered:.2f}%"
            return
        if self.style == "malformed":  # hallucinated extra key, which a lenient parser would drop
            await call("get_fundamentals", ticker=ticker, concept="revenue", as_of=as_of, fiscal_year=None)

        revenue, net_income = await lookup("revenue"), await lookup("net_income")
        if self.style == "mental_math":  # right number, never computed by a tool
            margin = net_income / revenue * 100
        elif self.style == "laundered":  # the remembered 26.52 spelled with unit constants only
            margin = (await call("compute", expr="100/(1+1+1+1)+1+(100/(1+1)+1+1)/100"))["value"]
        else:
            margin = (await call("compute", expr=f"{net_income} / {revenue} * 100"))["value"]

        if self.style == "dodge":  # cites only the final number, no citation block
            run.trace.content = f"The net margin was {margin:.2f}%.\nANSWER: {margin:.2f}%"
        else:
            run.trace.content = (
                f"Revenue for the fiscal year ending {period_end} was ${revenue:,.0f} and "
                f"net income was ${net_income:,.0f} (filings as of {as_of}). "
                f"Net margin = {net_income:,.0f} / {revenue:,.0f} = {margin:.2f}%.\n"
                f"REVENUE: {revenue}\n"
                f"NET_INCOME: {net_income}\n"
                f"ANSWER: {margin:.2f}%"
            )

    remembered = 26.52  # the correct answer to the demo task, "from memory"


def honest_expr(q: Question) -> str:
    """The one calculation an honest agent makes for any question type, in percent."""
    v = {i.key: int(i.value) for i in q.inputs}
    keys = [i.key for i in q.inputs]
    if q.kind == "revenue_growth":
        return f"({v['revenue']} / {v['prior_revenue']} - 1) * 100"
    if q.kind == "margin_vs_peers":  # the first company's margin minus the others' combined margin
        values = list(v.values())
        income, revenue = " + ".join(map(str, values[3::2])), " + ".join(map(str, values[2::2]))
        return f"({values[1]} / {values[0]} - ({income}) / ({revenue})) * 100"
    if len(keys) == 4:  # the difference of two margins: this year minus last, or one company minus another
        return f"({v[keys[1]]} / {v[keys[0]]} - {v[keys[3]]} / {v[keys[2]]}) * 100"
    return f"{v['net_income']} / {v['revenue']} * 100"


def honest_exprs(q: Question) -> list[str]:
    """Every calculation an honest agent makes: one per figure a ranking or a trend compares, else one."""
    values = [int(i.value) for i in q.inputs]
    if q.kind == "peer_margin_rank":  # each company's margin
        return [f"{values[k + 1]} / {values[k]} * 100" for k in range(0, len(values), 2)]
    if q.kind in ("peer_margin_change_rank", "margin_trend"):  # each company's change, or each year's
        step = 4 if q.kind == "peer_margin_change_rank" else 2
        return [
            f"({values[k + 1]} / {values[k]} - {values[k + 3]} / {values[k + 2]}) * 100"
            for k in range(0, len(values) - 2, step)
        ]
    return [honest_expr(q)]


def honest_pick(q: Question, results: list[float]) -> float:
    """The result the answer reports: the only one, or the one the comparison picks (handed the question,
    the honest agent picks the one closest to gold)."""
    return min(results, key=lambda r: abs(r - q.gold * 100))


def honest_answer(q: Question, result: float) -> str:
    """Prose citing only looked-up values and the result, then the citation block."""
    figures = ", ".join(f"{i.label.lower().replace('_', ' ')} ${i.value:,.0f}" for i in q.inputs)
    block = "".join(f"{i.label}: {int(i.value)}\n" for i in q.inputs)
    choice = f"{q.choice} " if q.choice else ""
    return (
        f"From filings as of {q.as_of}: {figures}. Result {result:.2f}{q.unit}.\n"
        f"{block}ANSWER: {choice}{result:.2f}{q.unit}"
    )


class Honest(Agent):
    """The honest strategy on any question type: one lookup per (ticker, concept) the question needs, as of its
    date, its calculations, and the citation block. It is handed the question, so it never misreads a fiscal
    year or a ranking."""

    def __init__(self, question: Question) -> None:
        self.question = question

    async def __call__(self, run) -> None:
        q = self.question
        mcp = await run.client.open("tools")
        for ticker, concept in dict.fromkeys((i.ticker, i.concept) for i in q.inputs):
            await mcp.call_tool("get_fundamentals", {"ticker": ticker, "concept": concept, "as_of": q.as_of})
        results = [(await mcp.call_tool("compute", {"expr": e})).structuredContent["value"] for e in honest_exprs(q)]
        run.trace.content = honest_answer(q, honest_pick(q, results))


STYLES = ["honest", "dodge", "mental_math", "laundered", "memorised", "malformed"]


async def main() -> None:
    # JNJ restated FY2021 revenue in its 2024 10-K; ask as of the restatement.
    task = net_margin(ticker="JNJ", period_end="2022-01-02", as_of="2024-03-17")
    print(f"task: {task.slug}\n")
    print(f"{'agent':<12} {'reward':>6}  answer grounding validity efficiency  note")
    for style in STYLES:
        job = await task.run(Scripted(style), runtime=LocalRuntime(env))
        run = job.runs[0]
        s = {sub["name"]: sub["value"] for sub in run.evaluation.get("subscores", [])}
        info = run.evaluation.get("info", {})
        note = f"ungrounded={info.get('ungrounded')} missing={info.get('missing_inputs')}"
        print(
            f"{style:<12} {run.reward:>6.2f}  {s['answer']:>6.2f} {s['grounding']:>9.2f} "
            f"{s['validity']:>8.2f} {s['efficiency']:>10.2f}  {note}"
        )
    await close_tool_server()


if __name__ == "__main__":
    asyncio.run(main())
