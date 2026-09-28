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
