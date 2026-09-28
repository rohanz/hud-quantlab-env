# Environment

How the environment is put together: the tools, the MCP surface, the task
template and the data behind them.

## Tools

Defined in `src/quantlab_env/tools.py` and listed in `TOOLS`. Every parameter
is a string.

### `get_fundamentals(ticker, concept, as_of)`

Annual values of one concept for one company, as known on `as_of`.

- `ticker`: one of `AAPL`, `MSFT`, `NVDA`, `KO`, `JNJ`.
- `concept`: `revenue` or `net_income` (the keys of the fixture's `concepts`).
- `as_of`: ISO date `YYYY-MM-DD`. Only filings made on or before it are visible,
  and for each fiscal period the newest such filing wins (ties broken by
  accession number). This is `data.facts_as_of`.

Success:

```json
{"ticker": "JNJ", "concept": "revenue", "as_of": "2024-03-17",
 "rows": [{"period_end": "2022-01-02", "value": 78740000000, "filed": "2024-02-16", "form": "10-K"}]}
```

Errors: `unknown_ticker`, `unknown_concept`, `bad_argument` (not a valid ISO
date), `no_data` (nothing filed by `as_of`).

### `compute(expr)`

Arithmetic on literal numbers: `+ - * /`, unary minus and plus, parentheses. At
most 200 characters. Names, calls, powers, booleans and strings are rejected.
Division by zero and non-finite results are `bad_argument`. Success returns
`{"expr": ..., "value": ...}`.

### Errors

Every error has the shape `{"error": {"code": "...", "message": "..."}}`. The
message is written for the agent to read and recover from.

## MCP surface

`env.py` builds a FastMCP server named `quantlab-sec` and registers each
function in `TOOLS`, so `list_tools` shows the name, docstring and a JSON schema
with all three `get_fundamentals` parameters required.

An `@env.initialize` hook starts the server with uvicorn on a free local port,
once per process, and publishes it as an MCP capability named `tools`
(`mcp/2025-11-25`, `http://127.0.0.1:<port>/mcp`). An agent opens it with
`run.client.open("tools")`. `close_tool_server()` stops it, for scripts and
tests that host the env in-process.

The `StrictToolLog` middleware intercepts `on_call_tool` and never calls the
next handler. It returns `tools.execute(...)` as structured content, so every
call is validated strictly and logged, and FastMCP's own argument validation
(which coerces values and rejects bad calls before the tool runs) is bypassed.

## Task template

```python
@env.template(id="net_margin", ...)
async def net_margin(ticker: str, period_end: str, as_of: str): ...
```

1. `questions.build(ticker, period_end, as_of)` computes the gold from the
   point-in-time data and fails if the period was not filed by `as_of`.
2. `EPISODE.begin()` clears the call log and returns a new generation.
3. It yields the prompt (which asks for the `REVENUE:`, `NET_INCOME:` and
   `ANSWER:` lines) and receives the agent's final text.
4. If `EPISODE.generation` is no longer this episode's, a newer session
   cleared the log, and it raises `OverlappingEpisodes` instead of grading
   (see [concurrency](#concurrency)).
5. It yields an `EvaluationResult` with the reward, one `SubScore` per
   component (weights from `grader.WEIGHTS`) and an `info` dict: gold, final
   answer, correct, the parsed citation block, cited numbers, ungrounded
   numbers, missing inputs, whether the answer is derived, call counts
   (total, malformed, future lookups).

`tasks.py` binds the template to `questions.default_questions()`: 18 tasks, one
per company and fiscal year ending 2022 to 2024, asked 30 days after the later
of the first revenue and net income filings for that period. Where a period's
revenue changed in a later filing, the period is asked again 30 days after the
restatement. Task ids look like `net_margin-83d5f782`.

## Concurrency

One episode at a time per env process. The call log (`EPISODE`) is process
state: HUD publishes the `tools` URL once per process, and an MCP tool call
carries no session id, so calls cannot be split per session.

- Safe: `LocalRuntime` (serialises acquisitions of an env instance),
  and `hud eval` and `DockerRuntime`, which give each rollout its own process.
- Unsafe to rely on: several concurrent rollouts against one `hud serve`
  process, for example `hud eval --runtime tcp://...:8765` with the default
  concurrency.

For that case each episode takes a generation of the log when it starts, and
the newest session wins: an older session whose log a newer one cleared fails
at grade time with `OverlappingEpisodes` instead of being graded on the wrong
log. What the guard cannot catch: if the older session keeps calling tools
after the newer one started, those calls land in the newer session's log.

An open-episode lock (refuse a new episode while one is open) was rejected:
an abandoned or parked session, such as `hud task start` that is never
graded, would block every later episode in the process.

## Fixture

The fixture is `src/quantlab_env/fixture.json`. It ships as package data inside
the `quantlab_env` wheel, and `data.load_fixture()` reads it with
`importlib.resources`, so the environment works from an installed wheel (as in
the Docker image) as well as from a checkout. Reading it needs no network.

Schema:

```
{
  "source": "SEC EDGAR XBRL companyfacts API (data.sec.gov), public domain",
  "concepts": {"revenue": [xbrl tags...], "net_income": ["NetIncomeLoss"]},
  "companies": {
    "<TICKER>": {
      "name": "...",
      "cik": 123456,
      "facts": {
        "<concept>": [
          {"period_start", "period_end", "value", "form", "filed", "accn", "tag"}
        ]
      }
    }
  }
}
```

Rows are sorted by `(period_end, filed, accn)`. A period appears once per
10-K that reported it, so a restatement adds a second row for the same period
with a later `filed` date. The tools expose only `period_end`, `value`,
`filed` and `form`.

## Regenerating the fixture

`scripts/build_fixture.py` downloads `companyfacts` for each company in
`COMPANIES` from `data.sec.gov` and keeps:

- facts under the tags listed in `CONCEPTS`, with tag precedence applied per
  filing: for each `(period_end, accn)` the first tag in the list that the
  filing used for that period wins (revenue falls back from
  `RevenueFromContractWithCustomerExcludingAssessedTax` to `Revenues` to
  `SalesRevenueNet`). Precedence used to be per period, so when a company
  switched tags, older filings that reported the period under a fallback tag
  were dropped and point-in-time history was lost,
- USD values from `10-K` and `10-K/A` forms only,
- durations of 350 to 380 days (annual periods),
- periods ending in 2021 to 2024,
- one row per `(period_end, accn)`; companyfacts repeats a fact once per
  frame, and the duplicates collapse here.

It writes `src/quantlab_env/fixture.json`.

SEC's fair-access policy requires a User-Agent with contact details, so the
script refuses to run without `SEC_USER_AGENT` in the environment:

```bash
SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py
```

After regenerating, run `uv run pytest`. These tests catch data changes:

- `tests/test_fixture.py` pins hand-checked values read straight from the JSON
  (JNJ FY2021 revenue as first reported and as restated, AAPL and KO FY2023
  revenue and net income), independent of the gold code.
- `tests/test_build_fixture.py` runs the builder on synthetic companyfacts (a
  tag transition, restatements, duplicate facts) and checks that the checked-in
  fixture is a fixed point of the builder. It also pins three NVIDIA revenue
  rows the current fixture lacks, from filings built under the old per-period
  precedence (NVIDIA moved from `Revenues` to
  `RevenueFromContractWithCustomerExcludingAssessedTax`). No default question
  can see those filings. Regenerating should add them; then empty
  `KNOWN_MISSING_REVENUE`.
- `test_every_default_question_is_answerable_from_the_tools` in
  `tests/test_tools.py`.

The environment itself only reads the JSON and never touches the network.

## Dependencies

`pyproject.toml` pins `hud>=0.6.18,<0.7`. The environment was built and tested
against hud 0.6.18, and 0.7 may change the `Environment` and eval API. `env.py`
imports `fastmcp` (`>=3.2,<4`) and `uvicorn` (`>=0.34,<1`), and
`scripts/build_fixture.py` imports `httpx` (`>=0.28,<1`); these are declared
directly rather than relied on through hud. `uv.lock` fixes exact versions.

## Docker

`Dockerfile.hud` starts from `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`,
sets `HUD_TELEMETRY_ENABLED=false`, installs the locked dependencies in one
layer and the project (non-editable, fixture included) in the next, and runs
the venv's `hud serve env.py` on port 8765. Nothing is built or synced when
the container starts.

- It runs as the unprivileged user `app` (uid 1000). The code and `.venv`
  stay root-owned and read-only to it; nothing is written under `/hud` at
  runtime.
- A `HEALTHCHECK` (every 30s) connects to port 8765, which is hud's
  newline-delimited JSON-RPC control channel, not HTTP, and sends a `hello`.
  That opens a session without starting a task, so the probe has no side
  effects; it passes if the reply carries a `session_id`.
- `.dockerignore` keeps `.git`, `.room/`, `.claude/`, `.venv/`, caches,
  `tests/`, `docs/`, `scripts/`, `*.md`, `demo.py` and the Docker files
  themselves out of the build context. The image needs only `pyproject.toml`,
  `uv.lock`, `.python-version`, `env.py`, `tasks.py` and `src/`.

```bash
docker build -f Dockerfile.hud -t hud-quantlab-env:local .
docker run --rm -p 8765:8765 hud-quantlab-env:local
```

Run one episode at a time against a served container (see
[concurrency](#concurrency)).
