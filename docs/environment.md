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

## Task templates

```python
@env.template(id="net_margin", ...)
async def net_margin(ticker: str, period_end: str, as_of: str): ...
# the probe set
async def fy_net_margin(ticker: str, fiscal_year: str, as_of: str): ...
async def latest_net_margin(ticker: str, as_of: str): ...
async def net_margin_change(ticker: str, fiscal_year: str, as_of: str): ...
async def revenue_growth(ticker: str, fiscal_year: str, as_of: str): ...
async def net_margin_gap(ticker: str, other: str, fiscal_year: str, as_of: str): ...
# chained analyst questions (tickers and peers are comma-separated, "AAPL,MSFT,ORCL")
async def peer_margin_rank(tickers: str, fiscal_year: str, rank: str, as_of: str): ...  # rank: highest or lowest
async def peer_margin_change_rank(tickers: str, fiscal_year: str, as_of: str): ...
async def margin_vs_peers(ticker: str, peers: str, fiscal_year: str, as_of: str): ...
async def margin_trend(ticker: str, as_of: str): ...
```

Each template builds its question and relays it through `_episode`:

1. The question builder (`questions.build` for `net_margin`, the function of
   the same name for the others) computes the gold from the point-in-time
   data and fails if a period was not filed by `as_of` (or, for a fiscal-year
   label, if no single filed period carries it, or, for a ranking or a
   trend, if the pick does not stand clear of the other candidates).
2. `EPISODE.begin()` clears the call log, opens the episode and returns its
   generation.
3. It yields the prompt (which asks for one labelled line per input, for net
   margin `REVENUE:` and `NET_INCOME:`, and the `ANSWER:` line) and receives
   the agent's final text.
4. If `EPISODE.overlapped(generation)` (another episode was open at the same
   time), it raises `OverlappingEpisodes` instead of grading (see
   [concurrency](#concurrency)). A `finally` calls `EPISODE.close(generation)`
   on grade, on error and when HUD closes an abandoned generator.
5. It yields an `EvaluationResult` with the reward, one `SubScore` per
   component (weights from `grader.WEIGHTS`) and an `info` dict: gold, final
   answer, correct, the choice and the choices the ANSWER line named (for
   a ranking or a trend), the parsed citation block, cited numbers, ungrounded
   numbers, missing inputs, whether the answer is derived, call counts
   (total, malformed, future lookups).

`tasks.py` holds three named sets (HUD `Taskset`s); `tasks_legacy.py` and
`tasks_probe.py` expose one each for `--source`, and `tasks_probe100.py` (v1)
and `tasks_probe_v2.py` expose the frozen subsets. `tasks_grpo_<split>.py`
(`grpo_splits.py`) poses the GRPO splits written by the training repo's
`scripts/hud_split.py`, read from the directory in `QUANTLAB_GRPO_SPLITS`
(without it those modules raise `ImportError`, so a directory scan skips them).

- `legacy`: `net_margin` bound to `questions.default_questions()`, 18 tasks
  pinned in `questions.LEGACY`: one per company and fiscal year ending 2022 to
  2024, asked 30 days after the later of the first revenue and net income
  filings for that period, plus JNJ's restatement re-asks. Pinned so the set
  does not grow with the fixture. Task ids look like `net_margin-83d5f782`.
- `probe`: each probe template bound to `questions.probe_questions()`,
  derived from the fixture (350 tasks with the 15-company fixture):
  - `fy_net_margin`: every company and fiscal-year label, 30 days after the
    next year's 10-K (so later rows are there to confuse), or after its own
    if there is none; plus 30 days after every restatement of its inputs and
    the view before the first one.
  - `latest_net_margin`: the day before each 10-K after a company's first,
    when the fixture holds the year that 10-K's predecessor reported.
  - `net_margin_change` and `revenue_growth`: each fiscal year against the
    one before, 30 days after its 10-K and around restatements, skipped when
    a restated year would be compared with a prior year from another filing,
    or when the result is under 0.1 percentage points.
  - `net_margin_gap`: for each fiscal-year label, each company (alphabetical)
    against the next, wrapping round, 30 days after the later 10-K; skipped
    under 0.1 percentage points.
- `chains`: each chained template bound to `questions.chain_questions()`,
  derived from the fixture over the peer groups in `questions.PEER_GROUPS`
  (294 tasks: 113, 62, 96 and 23 of the four types). Each peer question is
  asked 30 days after the last member's 10-K for the year, plus 30 days after
  every restatement of any member's figures and the view before the first
  one; where two dates see the same figures, only the first is kept.
  - `peer_margin_rank`: each group of four and each three of it, every
    fiscal-year label they all have, asking for the highest or the lowest in
    turn; asked only if the pick stands clear of every other margin by 0.1
    percentage points and twice the 0.5% answer tolerance.
  - `peer_margin_change_rank`: each three of a group (four would be 16
    figures), each year against the one before, on one basis per company (as
    for `net_margin_change`), with the same clearance for the largest change.
  - `margin_vs_peers`: each member of a group of four against the other three
    combined (total net income over total revenue); skipped under 0.1
    percentage points.
  - `margin_trend`: the day before each 10-K and 30 days after it, when the
    fixture holds the latest year filed and the three before it, on one basis
    throughout; asked only if the smallest of the three changes stands clear
    of the other two and is at least 0.1 percentage points.

The frozen subsets are drawn by `questions.stratified_sample` (seed
20261006, each kind's questions sorted by prompt): v1
(`questions.probe_sample`, 20 per v1 type) and v2
(`questions.probe_sample_v2`, `PROBE_V2_COUNTS`: 20 per chained type and 4
per v1 type, from the probe and chained sets together). Their prompts are
pinned by sha256 in `tests/test_probe_sample.py`.

Fiscal-year labels (`questions.fiscal_year`): the calendar year a fiscal year
ends in, except that a 52/53-week year ending in the first week of January
belongs to the year before. NVIDIA's fiscal 2024 ends 2024-01-28 and JNJ's
fiscal 2021 ends 2022-01-02.

## Concurrency

One episode at a time per env process. The call log (`EPISODE`) is process
state: HUD publishes the `tools` URL once per process, and an MCP tool call
carries no session id, so calls cannot be split per session.

- Safe: `LocalRuntime` (serialises acquisitions of an env instance),
  and `hud eval` and `DockerRuntime`, which give each rollout its own process.
- Not graded: several concurrent rollouts against one `hud serve` process,
  for example `hud eval --runtime tcp://...:8765` with the default
  concurrency. Pass `--max-concurrent 1` with a `tcp://` runtime, or use one
  process per rollout.

For that case `Episode` (in `tools.py`) keeps a symmetric guard. `open` maps
each episode not yet closed to its start time. When an episode begins while
another is open, every open episode, the new one included, is marked as
overlapping, and an overlapped episode raises `OverlappingEpisodes` at grade
time instead of being graded. The older episodes lost their log, and the
newer one's log can hold the older ones' calls. Episodes close on grade, on
error and on HUD's `aclose` of the template generator.

Before this guard the newest session won: an older session failed, but a
newer one was graded on the shared log, so a freeloader session that made no
tool calls and submitted an older session's answer scored 1.00 on the older
session's calls. Now both fail loudly, and `tests/test_env.py` checks this
over a real `tcp://` runtime, along with four overlapping honest sessions
(all refused) and sequential sessions in one process (each graded 1.00).

An episode that is never closed (an abandoned session, or `hud task start`
that is never graded) stops counting as open `QUANTLAB_EPISODE_TIMEOUT`
seconds after it began (default 3600, `tools.DEFAULT_EPISODE_TIMEOUT`, above
a typical rollout timeout). Expired entries are dropped when the next episode
begins.

What the guard cannot do:

- Attribute calls to sessions. It refuses overlapping episodes; it cannot
  grade them.
- Ignore an abandoned session before it expires: until then, every episode
  that begins in the process overlaps it and is refused.
- Protect a newer episode from an expired one. If an abandoned session comes
  back after a newer episode began, it is refused (its log is gone), but any
  tool calls it makes meanwhile land in the newer episode's log.

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
          {"period_start", "period_end", "value", "form", "filed", "accn", "tag", "fy"}
        ]
      }
    }
  }
}
```

`fy` is the filing's fiscal-year focus as companyfacts reports it (in a
period's first 10-K, that period's own label); fixtures built before
2026-10-06, including the checked-in one, have no `fy`. Rows are sorted by
`(period_end, filed, accn)`. A period appears once per
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
- periods ending in 2021 to 2025,
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
  can see those filings. The 2026-10-06 rebuild restored them, so
  `KNOWN_MISSING_REVENUE` is empty. Walmart's revenue uses its own tag order
  (`TAG_OVERRIDES`): total revenues (`Revenues`), not net sales.
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

Run one episode at a time against a served container; overlapping episodes
are refused with `OverlappingEpisodes` (see [concurrency](#concurrency)).
`QUANTLAB_EPISODE_TIMEOUT` can be set on the container to change the expiry.
