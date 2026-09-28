# Decisions

Short, dated entries. Newest last. Add one whenever a design choice would not be
obvious from the code.

## 2026-09-28: Depend on `hud` 0.6.18, not `hud-python`

The HUD SDK is published on PyPI as `hud` (the GitHub repo is still
`hud-evals/hud-python`). `pyproject.toml` pins `hud>=0.6.18`, the version the
environment, tests and demo were built and checked against. The API used
(`Environment`, `@env.initialize`, `@env.template`, `Capability.mcp`,
`LocalRuntime`, `EvaluationResult`/`SubScore`) is from that release.

## 2026-09-28: One tool server per process

The FastMCP server is started once per process and reused across env
stop/start cycles. Restarting FastMCP's HTTP app in the same process after a
forced stop left the next MCP handshake hanging, and the server holds no
per-episode state (the task template resets the call log). The consequence is
one episode at a time per process, like the board in HUD's 2048 cookbook.
`LocalRuntime` serialises acquisitions and `hud eval` runs each rollout in its
own subprocess, so this holds in practice.

## 2026-09-28: Bypass FastMCP argument validation with a middleware

FastMCP validates arguments before the tool runs: it coerces `"2024"`-style
values and rejects bad calls outright, so those calls would never reach the
episode log and could not be graded. `StrictToolLog` answers every call from
`tools.execute`, which checks raw arguments strictly, logs the call, and returns
bad calls as `bad_argument` errors. One strict contract for the agent and the
grader is the point of the malformed-call case.

## 2026-09-28: Grounding is stricter than QuantLab's

QuantLab's grounding checked the final number against tool results and carried
0.1. Here grounding covers every cited number, requires the answer to cite the
gold's revenue and net income, and carries 0.3; answer credit also needs a
successful lookup. This closes the citation dodge and the wrong-year case, and
makes a correct answer without evidence score 0.

## 2026-09-28: Efficiency needs a successful lookup

The end-to-end test showed that an episode of only failed calls still earned
the 0.1 efficiency share. Efficiency is now 0 unless a `get_fundamentals` call
succeeded (commit 84a5f67).

## 2026-09-28: compute rejects non-finite results

`compute("1e308 * 10")` returned `inf`, which reached the agent over MCP as
`value: null` with no error. Non-finite results are now `bad_argument`, like
division by zero.

## 2026-09-28: Docker image not built yet

`Dockerfile.hud` exists for `hud deploy` but has not been built or run. Local
work uses `LocalRuntime` (tests, `demo.py`) and the HUD CLI against
`tasks.py`. Build and smoke-test the image before deploying.
