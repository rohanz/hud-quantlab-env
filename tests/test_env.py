"""End to end through HUD: LocalRuntime, the MCP capability, the template's grade."""

import asyncio

import pytest
from hud.agents.base import Agent
from hud.environment.server import bind
from hud.eval import LocalRuntime, Runtime

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


class Gated(Agent):
    """The honest scripted agent, with hooks to interleave it with another session."""

    def __init__(self, before=None, after=None):
        self.before, self.after, self.called = before, after, asyncio.Event()

    async def __call__(self, run):
        if self.before:
            self.before.set()
        await Scripted("honest")(run)
        self.called.set()
        if self.after:
            await self.after.wait()


async def test_an_episode_whose_log_a_newer_session_cleared_is_refused_not_misgraded():
    # One served process, two control sessions, as with `hud serve` and a
    # concurrent `hud eval --runtime tcp://...`: A makes its calls, B starts
    # (and clears the shared log) before A is graded.
    await env.start()
    server = await bind(env)
    serving = asyncio.create_task(server.serve_forever())
    runtime = Runtime(f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    try:
        b_started = asyncio.Event()
        a, b = Gated(after=b_started), Gated(before=b_started)
        job_a = asyncio.create_task(TASK.run(a, runtime=runtime))
        await a.called.wait()
        job_b = asyncio.create_task(TASK.run(b, runtime=runtime))
        run_a, run_b = [(await j).runs[0] for j in (job_a, job_b)]
    finally:
        serving.cancel()
        server.close()
    # Before the guard, A was graded 0.0 on the log B cleared. Now A fails with
    # the reason, and B, the newest session, is graded on its own calls.
    assert run_a.trace.status == "error" and not run_a.evaluation, run_a.evaluation
    assert "another task session" in run_a.trace.steps[-1].error
    assert run_b.reward == 1.0, run_b.evaluation
