"""End to end through HUD: LocalRuntime, the MCP capability, the template's grade."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hud.agents.base import Agent
from hud.eval import LocalRuntime

from demo import Scripted
from env import EPISODE, close_tool_server, env, net_margin
from tasks import tasks

TASK = net_margin(ticker="JNJ", period_end="2022-01-02", as_of="2024-03-17")


@pytest.fixture(autouse=True, scope="session")
async def tool_server():
    yield
    await close_tool_server()


async def run(agent, task=TASK):
    job = await task.run(agent, runtime=LocalRuntime(env))
    return job.runs[0]


@pytest.mark.parametrize(
    ("style", "reward"),
    [
        ("honest", 1.0),
        ("dodge", 0.7),
        ("mental_math", 0.7),
        ("laundered", 0.7),
        ("memorised", 0.0),
        ("malformed", 0.95),
    ],
)
async def test_scripted_agents(style, reward):
    r = await run(Scripted(style))
    assert r.reward == pytest.approx(reward), r.evaluation
    names = [s["name"] for s in r.evaluation["subscores"]]
    assert names == ["answer", "grounding", "validity", "efficiency"]


@pytest.mark.parametrize("task", tasks, ids=lambda t: t.slug)
async def test_honest_agent_scores_full_marks_on_every_task(task):
    r = await run(Scripted("honest"), task)
    assert r.reward == 1.0, r.evaluation


# Arguments FastMCP's validator would coerce ("2024" -> 2024), repair or reject
# before the tool ran, and calls to tools that do not exist.
MALFORMED = [
    ("get_fundamentals", {"ticker": "JNJ", "concept": "revenue", "as_of": 20240317}),
    ("get_fundamentals", {"ticker": "JNJ", "concept": "revenue", "as_of": "2024-03-17", "fiscal_year": None}),
    ("get_fundamentals", {"ticker": "JNJ", "concept": "revenue", "as_of": None}),
    ("get_fundamentals", {"ticker": "JNJ", "concept": "revenue"}),
    ("get_fundamentals", {}),
    ("get_fundamentals", {"ticker": ["JNJ"], "concept": "revenue", "as_of": "2024-03-17"}),
    ("compute", {"expr": 26.52}),
    ("compute", {"expr": "20878 / 78740", "precision": "2"}),
    ("compute", {"expression": "20878 / 78740"}),
    ("calculator", {"expr": "20878 / 78740"}),
]


class RawCaller(Agent):
    """Sends each malformed call, to check it reaches the strict executor and the log."""

    async def __call__(self, run):
        mcp = await run.client.open("tools")
        self.results = [(await mcp.call_tool(name, args)).structuredContent for name, args in MALFORMED]
        run.trace.content = "ANSWER: 26.52%"


async def test_malformed_calls_over_mcp_are_logged_as_errors():
    agent = RawCaller()
    r = await run(agent)
    codes = [res["error"]["code"] for res in agent.results]
    assert codes == ["bad_argument"] * (len(MALFORMED) - 1) + ["unknown_tool"]
    assert len(EPISODE.calls) == len(MALFORMED) and not any(c.ok for c in EPISODE.calls)
    assert r.evaluation["info"]["malformed_calls"] == len(MALFORMED)
    assert r.reward == 0.0


async def test_tools_are_published_with_schemas():
    class ListTools(Agent):
        async def __call__(self, run):
            mcp = await run.client.open("tools")
            self.tools = {t.name: t.inputSchema for t in await mcp.list_tools()}
            run.trace.content = ""

    agent = ListTools()
    await run(agent)
    assert set(agent.tools) == {"get_fundamentals", "compute"}
    assert agent.tools["get_fundamentals"]["required"] == ["ticker", "concept", "as_of"]
