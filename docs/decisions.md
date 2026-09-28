# Decisions

Short, dated entries. Newest last. Add one whenever a design choice would not be
obvious from the code.

## 2026-09-28: Depend on `hud` 0.6.18, not `hud-python`

(The bound is narrowed below: `hud>=0.6.18,<0.7`.)

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

(Superseded below: the image now installs at build time and has been built locally.)

`Dockerfile.hud` exists for `hud deploy` but has not been built or run. Local
work uses `LocalRuntime` (tests, `demo.py`) and the HUD CLI against
`tasks.py`. Build and smoke-test the image before deploying.

## 2026-09-28: An adversarial review found reward hacks; fixed with tests

After the first version was done, a Codex review attacked the grader directly
and found answers that scored 1.0 without being grounded. Every exploit below
is now a regression test (the docstrings cite the review), and the entries
that follow record each fix. The adversarial table in
[reward-design.md](reward-design.md) has the current scores. The original
laundering case (`compute("26.52 * 1")`) was already caught; the review showed
that closing one spelling of an exploit is not the same as closing the family.

## 2026-09-28: Track calculator provenance; the answer must be net income over revenue

Unit constants (1, 100, 1e3, 1e6, 1e9) were allowed calculator literals, so
`100/(1+1+1+1)+1+(100/(1+1)+1+1)/100` spelled 26.52 with no evidence and a
memorised margin scored 1.0. Each evidence item now carries a formula over the
looked-up values. Constants may only scale evidence, both sides of `+` and `-`
must use evidence, a result must depend on a looked-up value (so `x / x` is
not a way to make 1), and the final answer must match an item proportional to
the question's net income over its revenue. Checking provenance, rather than
adding more rules about which literals are allowed, is what closes the whole
family: variants that cancel evidence out or add a tiny evidence term fail the
same check.

## 2026-09-28: Signs and currency scale are part of a claim

Claims were compared as magnitudes, so "net income -$20.878 billion" matched a
reported profit, and a bare `$78.74` could stand for 78.74 billion. Comparison
is now signed, a minus attached to an amount is its sign (a hyphen after a word
or digit is not), and a currency amount without a unit word means exactly that
many dollars. Only a bare number may stand for thousands, millions, billions or
a percent, because agents legitimately write "78,740 (USD millions)".
Parentheses are read as prose, not as accounting negatives, since they appear
far more often as asides.

## 2026-09-28: Lookups past the question's date are not evidence

A lookup as of 2099-01-01 saw filings the question rules out (including
restatements filed later) and still counted as evidence, scoring 1.0. Now only
lookups with `as_of` on or before the question's date are evidence or count
as a successful lookup. Such a call is well formed, so it is not malformed; it
is ignored and still costs efficiency. The cutoff lives in the grader, not the
tools, so the tools stay question agnostic; `future_lookups` in the grade info
reports these calls.

## 2026-09-28: Exempt numbers by context, not value

Small integers and year-like values were exempt by value alone, so "debt 7"
rode along with an honest answer at 1.0 (as did "$7" and "$2023" until the
currency fix above). Now a number is exempt
only by context: dates (ISO, numeric, month names), FY and quarter labels, form
names, numbered-list markers at the start of a line, a bare four-digit year
with no currency or unit, or a single-digit count followed by a word
("3 filings"). Anything with a currency sign, "dollars" or "USD", a unit word
or a percent is always a claim. A bare "2023" is still read as a year; the
remaining exemptions are listed as known limits.

## 2026-09-28: Bind a calculator literal to the closest evidence

Found by running the honest agent on every task, not by the review. NVIDIA's
FY2022 revenue (26,914m) is within the 0.5% tolerance of FY2023's (26,974m),
so the provenance check bound the FY2023 literal to the FY2022 row and the
honest answer scored 0.7. A literal now binds to the evidence item it matches
most closely. The end-to-end tests now require 1.0 for the honest agent on all
18 tasks, and demo.py gained a `laundered` agent (0.70).

## 2026-09-28: Narrow the claims in the docs

The README said answering from memory was "always" the worse strategy and that
dropping a bad citation could never raise the score. Neither was what the code
guarantees: removing a wrong number from an answer that cites its inputs does
raise the score (as it should), and the grader has blind spots. The docs now
state only what the code enforces and the tests check, and README.md and
reward-design.md list the known limits.

## 2026-09-28: Apply XBRL tag precedence per filing, not per period

`build_fixture.py` let the preferred revenue tag win a whole fiscal period, so
when a company switched tags, earlier filings that reported the period under a
fallback tag were dropped and point-in-time history was lost. Precedence now
applies per `(period_end, accn)`. Tests cover a tag transition, restatements
and duplicate facts on synthetic companyfacts, and check the checked-in
fixture is a fixed point of the builder. The current fixture still lacks three
NVIDIA revenue rows from the old rule; no default question can see them, and
the test pins them until the fixture is regenerated (which needs the network).
Hand-checked values in `tests/test_fixture.py` now pin the data independently
of the gold code.

## 2026-09-28: Ship the fixture as package data

`data/fixture.json` sat outside the package, so `data.py` assumed an editable
checkout and an installed wheel had no data. It moved to
`src/quantlab_env/fixture.json` and loads with `importlib.resources`.

## 2026-09-28: Bound hud below 0.7 and declare direct imports

`hud>=0.6.18` allowed a future 0.7, which may change the `Environment` and eval
API. The pin is now `hud>=0.6.18,<0.7`. `env.py` imports `fastmcp` and
`uvicorn`, and `build_fixture.py` imports `httpx`; they were only present
transitively through hud, so they are now declared with bounds.

## 2026-09-28: Docker image installs at build time, telemetry off

`Dockerfile.hud` synced an editable install at build time but started with
`uv run`, which rebuilt the project on every container start. It now installs
the locked dependencies in their own layer, then the project non-editable into
`.venv`, and runs that venv's `hud serve` directly, so starting the container
builds nothing. `HUD_TELEMETRY_ENABLED=false` is set in the image.
`.dockerignore` matches `.git` as a file too, because in a git worktree it is
one. The image builds locally as `hud-quantlab-env:local` and serves on port
8765; it has not been deployed to the HUD platform.
