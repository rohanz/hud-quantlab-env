"""HUD environment: point-in-time net margin from SEC filings, graded on evidence.

The tools are served over MCP by a FastMCP server that an `@env.initialize`
hook starts and publishes as the `tools` capability. A middleware routes every
tool call through `quantlab_env.tools.execute`, so each call is checked
strictly and logged before FastMCP's own (coercing) argument validation could
repair or drop it. The task template resets that log, yields the prompt, and
grades the answer against it.

One episode at a time per env process. The call log is process state: HUD
publishes the `tools` URL once per process, and an MCP tool call carries
nothing that ties it to a task session, so calls cannot be split per session.
`LocalRuntime` serialises acquisitions of an env instance, and `hud eval`
and `DockerRuntime` give each rollout its own process. One `hud serve` process
can still host concurrent sessions (for example `--runtime tcp://...` with
several rollouts). Each episode therefore takes a generation of the log when
it starts, and the newest session wins: an older session whose log was cleared
by a newer one fails at grade time with `OverlappingEpisodes` instead of being
graded on the wrong log. What this cannot catch: if the older session keeps
calling tools after the newer one started, those calls land in the newer
session's log.
"""

from __future__ import annotations

import asyncio
import socket

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from hud.capabilities import Capability
from hud.environment import Environment
from hud.graders import EvaluationResult, SubScore

from quantlab_env import grader, questions, tools

EPISODE = tools.Episode()


class OverlappingEpisodes(RuntimeError):
    """Raised at grade time when a newer session cleared this episode's tool log."""


class StrictToolLog(Middleware):
    """Answer every tool call from the strict executor and record it."""

    async def on_call_tool(self, context, call_next):
        msg = context.message
        return ToolResult(structured_content=tools.execute(msg.name, msg.arguments or {}, EPISODE))


server = FastMCP(name="quantlab-sec", middleware=[StrictToolLog()])
for fn in tools.TOOLS.values():
    server.tool(fn)  # publishes name, docstring and JSON schema

env = Environment(name="quantlab-margin")
_http: tuple[str, uvicorn.Server, asyncio.Task] | None = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@env.initialize
async def _start_tools() -> None:
    """Serve the tools once per process and (re)publish the address.

    The server outlives env stop/start cycles on purpose: restarting FastMCP's
    HTTP app in the same process after a forced stop left the next MCP
    handshake hanging, and there is no per-episode state in the server itself
    (the call log is reset by the task template).
    """
    global _http
    if _http is None or _http[2].done():
        port = _free_port()
        config = uvicorn.Config(server.http_app(), host="127.0.0.1", port=port, log_level="warning")
        http = uvicorn.Server(config)
        _http = (f"http://127.0.0.1:{port}/mcp", http, asyncio.create_task(http.serve()))
        while not http.started:
            if _http[2].done():
                raise RuntimeError("tool server failed to start")
            await asyncio.sleep(0.02)
    env.add_capability(Capability.mcp(name="tools", url=_http[0]))


async def close_tool_server() -> None:
    """Stop the tool server; for scripts that host the env in-process."""
    global _http
    if _http is not None:
        _, http, task = _http
        http.should_exit = True
        if not (await asyncio.wait({task}, timeout=2))[0]:
            task.cancel()
        _http = None


@env.template(id="net_margin", description="Point-in-time net margin, graded on correctness and evidence")
async def net_margin(ticker: str, period_end: str, as_of: str):
    question = questions.build(ticker, period_end, as_of)
    generation = EPISODE.begin()
    answer = yield question.prompt
    if EPISODE.generation != generation:
        raise OverlappingEpisodes(
            "another task session started in this env process during this episode and cleared "
            "its tool-call log; run one episode at a time per process"
        )
    result = grader.grade(answer, EPISODE.calls, question)
    yield EvaluationResult(
        reward=result.reward,
        content=f"reward {result.reward:.2f}: {result.subscores}",
        info=result.info,
        subscores=[SubScore(name=k, weight=grader.WEIGHTS[k], value=v) for k, v in result.subscores.items()],
    )
