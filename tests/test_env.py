"""End to end through HUD: LocalRuntime, the MCP capability, the template's grade."""

import asyncio
import contextlib

import pytest
from hud.agents.base import Agent
from hud.environment.server import TaskRunner, bind
from hud.eval import LocalRuntime, Runtime

from demo import Honest, Scripted
from env import EPISODE, close_tool_server, env, net_margin
from quantlab_env.questions import probe_questions
from tasks import as_tasks, legacy, probe

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


@pytest.mark.parametrize("task", list(legacy), ids=lambda t: t.slug)
async def test_honest_agent_scores_full_marks_on_every_legacy_task(task):
    r = await run(Scripted("honest"), task)
    assert r.reward == 1.0, r.evaluation


PROBE = probe_questions()


def test_probe_taskset_binds_every_probe_question():
    assert [t.slug for t in probe] == [t.slug for t in as_tasks(PROBE)]
    assert not {t.slug for t in probe} & {t.slug for t in legacy}


@pytest.mark.parametrize(
    ("question", "task"), list(zip(PROBE, probe, strict=True)), ids=lambda x: getattr(x, "slug", "")
)
async def test_honest_agent_scores_full_marks_on_every_probe_task(question, task):
    r = await run(Honest(question), task)
    assert r.reward == 1.0, r.evaluation
    assert r.evaluation["info"]["tool_calls"] == question.par_calls


async def test_a_probe_task_grades_a_wrong_fiscal_year():
    """End to end: NVIDIA fiscal 2023 answered with the year ending 2024-01-28 (the calendar-year slip)."""
    from env import fy_net_margin
    from quantlab_env.questions import fy_net_margin as build_fy

    task = fy_net_margin(ticker="NVDA", fiscal_year="2023", as_of="2024-03-22")
    r = await run(Honest(build_fy("NVDA", "2024", "2024-03-22")), task)
    assert r.reward == pytest.approx(0.2), r.evaluation


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
    """The honest scripted agent, with hooks to interleave it with other sessions."""

    def __init__(self, hold=()):
        self.hold, self.started, self.called = hold, asyncio.Event(), asyncio.Event()

    async def __call__(self, run):
        self.started.set()
        await Scripted("honest")(run)
        self.content = run.trace.content
        self.called.set()
        for event in self.hold:  # stay open (ungraded) until these fire
            await event.wait()


class Freeloader(Agent):
    """Makes no tool calls and submits another session's honest answer from memory."""

    def __init__(self, source):
        self.source, self.started = source, asyncio.Event()

    async def __call__(self, run):
        self.started.set()
        run.trace.content = self.source.content


@contextlib.asynccontextmanager
async def served():
    """One served env process, as with `hud serve`, reached over tcp like `--runtime tcp://...`."""
    await env.start()
    server = await bind(env)
    serving = asyncio.create_task(server.serve_forever())
    try:
        yield Runtime(f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    finally:
        serving.cancel()
        server.close()


def refused(r):
    return r.trace.status == "error" and not r.evaluation and "another task session" in r.trace.steps[-1].error


async def test_a_freeloader_is_not_graded_on_an_older_sessions_calls():
    # A makes its calls; B starts, makes none and submits A's block. Before,
    # B scored 1.00 on A's calls in the shared log and A failed.
    async with served() as runtime:
        a = Gated()
        b = Freeloader(source=a)
        a.hold = (b.started,)
        job_a = asyncio.create_task(TASK.run(a, runtime=runtime))
        await a.called.wait()
        run_b = (await TASK.run(b, runtime=runtime)).runs[0]
        run_a = (await job_a).runs[0]
    assert refused(run_b), run_b.evaluation
    assert refused(run_a), run_a.evaluation
    assert not EPISODE.open


async def test_overlapping_honest_sessions_all_fail_loudly_never_misgraded():
    # Four concurrent rollouts: before, the last was graded on all 12 calls.
    async with served() as runtime:
        agents = [Gated() for _ in range(4)]
        for agent in agents:
            agent.hold = tuple(other.called for other in agents)
        jobs = await asyncio.gather(*(TASK.run(agent, runtime=runtime) for agent in agents))
    runs = [job.runs[0] for job in jobs]
    assert all(refused(r) for r in runs), [r.evaluation for r in runs]
    assert not EPISODE.open


async def test_sequential_sessions_in_one_process_are_each_graded():
    async with served() as runtime:
        for _ in range(3):
            r = (await TASK.run(Scripted("honest"), runtime=runtime)).runs[0]
            assert r.reward == 1.0, r.evaluation
    assert not EPISODE.open


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize(("elapsed", "reward"), [(61, 1.0), (59, None)])
async def test_an_abandoned_session_stops_blocking_after_the_timeout(monkeypatch, elapsed, reward):
    clock = Clock()
    monkeypatch.setattr(EPISODE, "clock", clock, raising=False)
    monkeypatch.setattr(EPISODE, "timeout", 60, raising=False)
    abandoned = TaskRunner(net_margin, {"ticker": "JNJ", "period_end": "2022-01-02", "as_of": "2024-03-17"})
    await abandoned.start()  # started, never graded or closed
    try:
        clock.now = elapsed
        r = await run(Scripted("honest"))
    finally:
        await abandoned.cancel()
    if reward is None:  # still within the timeout: the abandoned episode counts as open
        assert refused(r), r.evaluation
    else:
        assert r.reward == reward, r.evaluation
    assert not EPISODE.open
