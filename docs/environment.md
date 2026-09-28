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
2. `EPISODE.reset()` clears the call log.
3. It yields the prompt and receives the agent's final text.
4. It yields an `EvaluationResult` with the reward, one `SubScore` per
   component (weights from `grader.WEIGHTS`) and an `info` dict: gold, final
   answer, cited numbers, ungrounded numbers, missing inputs, call counts.

`tasks.py` binds the template to `questions.default_questions()`: 18 tasks, one
per company and fiscal year ending 2022 to 2024, asked 30 days after the later
of the first revenue and net income filings for that period. Where a period's
revenue changed in a later filing, the period is asked again 30 days after the
restatement. Task ids look like `net_margin-83d5f782`.

## Fixture schema

`data/fixture.json`:

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

- facts under the tags listed in `CONCEPTS`; for each period the first tag in
  the list that reports it wins (revenue falls back from
  `RevenueFromContractWithCustomerExcludingAssessedTax` to `Revenues` to
  `SalesRevenueNet`),
- USD values from `10-K` and `10-K/A` forms only,
- durations of 350 to 380 days (annual periods),
- periods ending in 2021 to 2024,
- one row per `(period_end, accn)`.

SEC's fair-access policy requires a User-Agent with contact details, so the
script refuses to run without `SEC_USER_AGENT` in the environment:

```bash
SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py
```

After regenerating, run `uv run pytest`. `test_every_default_question_is_answerable_from_the_tools`
and the pinned JNJ values will catch data changes. The environment itself only
reads the JSON and never touches the network.
