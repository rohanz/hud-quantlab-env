# AGENTS.md

Working guide for agents and humans in this repo. The README is the public
overview; this file is how to work on it.

## What this is

A [HUD](https://github.com/hud-evals/hud-python) RL environment that ports one
task type from the QuantLab research agent: **point-in-time net margin**. The
agent is asked for a company's net margin for a fiscal year "using only filings
available on" a given date. It looks figures up through MCP tools backed by an
offline SEC fixture, does the arithmetic with a calculator tool, and is graded
on the answer and on whether every number it cites came from a tool result in
that episode.

Public background:
[quantlab-agentic](https://www.rohanjk.xyz/projects/quantlab-agentic/) and
[quantlab-analyst](https://www.rohanjk.xyz/projects/quantlab-analyst/).

## Layout

```
env.py                         HUD Environment, FastMCP tool server, strict middleware, net_margin template,
                               symmetric session guard (OverlappingEpisodes, QUANTLAB_EPISODE_TIMEOUT)
tasks.py                       the 18 concrete tasks (one per default question)
demo.py                        six scripted HUD Agents run end to end over MCP, offline
Dockerfile.hud                 image for hud serve / hud deploy (installs at build, telemetry off, port 8765,
                               non-root user app, HEALTHCHECK); .dockerignore keeps dev files out
src/quantlab_env/data.py       fixture loading (importlib.resources) and point-in-time lookup (facts_as_of)
src/quantlab_env/tools.py      get_fundamentals, compute, strict validate/execute, Episode (call log, open
                               episodes, overlap marks, DEFAULT_EPISODE_TIMEOUT)
src/quantlab_env/grader.py     reward: answer, grounding (citation block, calculator provenance, prose check),
                               validity, efficiency
src/quantlab_env/questions.py  Question, gold answers, prompt text (with the citation block), default_questions()
src/quantlab_env/fixture.json  package data: public SEC XBRL revenue and net income, 5 companies, 2021 to 2024
scripts/build_fixture.py       regenerate the fixture from SEC EDGAR (network, needs SEC_USER_AGENT)
tests/                         test_grader.py, test_tools.py, test_fixture.py, test_build_fixture.py (offline
                               units), test_env.py (end to end, every task, overlapping and freeloading
                               sessions over tcp, session expiry); 292 tests (223 in test_grader.py, 31 in
                               test_env.py, 23 in test_tools.py); pytest's pythonpath puts the root on sys.path
docs/                          reward-design.md, environment.md, decisions.md
```

## Commands

Always use `uv`. Keep HUD telemetry off for anything that imports or runs HUD:
`export HUD_TELEMETRY_ENABLED=false`. Data access needs no network; the fixture
ships inside the package.

```bash
export HUD_TELEMETRY_ENABLED=false
uv sync                                  # install, including dev tools (pytest, ruff)
uv run pytest                            # all tests, offline, a couple of seconds
uv run ruff format && uv run ruff check  # format and lint (line length 120)
uv run python demo.py                    # scripted agents through the env, offline
```

HUD CLI task lifecycle (offline, no model; prefer the tests and demo):

```bash
export HUD_TELEMETRY_ENABLED=false
uv run hud task list --source tasks.py
uv run hud task start net_margin-83d5f782 --source tasks.py
uv run hud task grade net_margin-83d5f782 --source tasks.py --answer "ANSWER: 26.52%" --json
```

Real-model eval (needs a provider key, costs money, ask before running):

```bash
export HUD_TELEMETRY_ENABLED=false
export ANTHROPIC_API_KEY=...             # or HUD_API_KEY with --gateway
uv run hud eval tasks.py claude --task-ids net_margin-83d5f782 --max-steps 10
uv run hud eval tasks.py claude --all --max-steps 10
```

Docker (the image sets `HUD_TELEMETRY_ENABLED=false` itself):

```bash
docker build -f Dockerfile.hud -t hud-quantlab-env:local .
docker run --rm -p 8765:8765 hud-quantlab-env:local   # hud serve on port 8765
```

The image runs as the non-root user `app` with a `HEALTHCHECK` (a JSON-RPC
`hello` on port 8765). One episode at a time per process: tool calls carry no
session id, so the call log is per process. `LocalRuntime`, `hud eval` (one
process per rollout) and `DockerRuntime` are safe. Several concurrent rollouts
against one `hud serve` (for example `--runtime tcp://...:8765` with the default
concurrency) are not graded at all: `EPISODE` tracks open episodes, and when
one begins while another is open, all of them, the new one included, fail at
grade with `OverlappingEpisodes` (before, the newest won, and a freeloader
that made no calls could score 1.00 on an older session's calls). Use
`--max-concurrent 1` with a `tcp://` runtime. Episodes close on grade, on
error and on HUD's `aclose`; an unclosed one expires after
`QUANTLAB_EPISODE_TIMEOUT` seconds (default 3600, `tools.DEFAULT_EPISODE_TIMEOUT`).
Gaps: calls still cannot be attributed to sessions; an abandoned session
blocks overlapping episodes until it expires; after expiry, an abandoned
session that returns after a newer one began is refused, and calls it makes
meanwhile land in the newer log.

Regenerating the fixture (network, only when changing data):

```bash
SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py
```

`pyproject.toml` pins `hud>=0.6.18,<0.7` and declares `fastmcp`, `uvicorn` and
`httpx` directly, because the code imports them.

## How the reward works

```
reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency
```

- **answer**: the first number on the last `ANSWER:` line is within 0.5% of the
  gold (as a percent or a fraction), and at least one admissible
  `get_fundamentals` call succeeded.
- **grounding** (all or nothing): the citation block is correct, the final
  answer matches a calculator result derived from the question's net income
  over its revenue, and every other cited number matches evidence from this
  episode (the prose check). Matching is at the precision a number is written
  with; the 0.5% tolerance is only for the answer against gold.
- **validity**: share of calls without a malformed error code (`bad_argument`,
  `unknown_tool`, `unknown_ticker`, `unknown_concept`), and 0 unless an
  admissible lookup succeeded (so also 0 with no calls).
- **efficiency**: `3 / max(3, calls)`, and 0 unless an admissible lookup
  succeeded.

Strict tool calls: a FastMCP middleware (`StrictToolLog` in `env.py`) answers
every call from `tools.execute`, which validates the raw arguments (no unknown
keys, no missing keys, strings only, no coercion), runs the tool and logs the
call. Bad calls come back as `{"error": {"code", "message"}}` and are counted,
never repaired or dropped.

Citation block: the prompt asks the answer to end with

```
REVENUE: <value>
NET_INCOME: <value>
ANSWER: <margin>%
```

with each value exactly as `get_fundamentals` returned it: whole USD with its
sign, an optional leading `$` and thousands commas, no rounding and no units.
`citations()` in `grader.py` reads the last line of each label (`_LABELS`), and
each value must equal a row of its own concept, for the question's ticker and
period, from an admissible lookup in this episode, and the value the gold
used. A missing value, any other formatting, or a swapped value lands in
`missing_inputs`. Labels (`REVENUE`, `NET_INCOME`, `ANSWER`) are
case-insensitive and open their line after any markdown list, quote, heading
or emphasis markers (`**REVENUE:** x`, `- REVENUE:`, `> REVENUE:`,
`**ANSWER:** 26.52%`); a block value may be wrapped only in `*` or `_`
emphasis and must be ASCII digits (fullwidth or Arabic-Indic digits land in
`missing_inputs`). Since labels are case-insensitive, a later prose line like
`Revenue: ...` counts as a block line (the last line of each label wins), as
`Answer:` already did.

Evidence rules (details: [docs/reward-design.md](docs/reward-design.md)):

- Only `value` fields of successful lookups are evidence, and only from lookups
  whose `as_of` is on or before the question's date (the filing-date cutoff).
  A later lookup is not malformed, but it is ignored.
- Calculator provenance: each evidence item carries a formula over the
  looked-up values. A `compute` result is evidence only if every literal equals
  evidence at its written precision (bound to the closest match) or is a unit
  constant (1, 100, 1e3, 1e6, 1e9), constants only scale evidence, each side of
  `+` and `-` itself depends on a looked-up value (`x / x` does not), and the
  result depends on a looked-up value.
- Error results never count.
- Cited numbers match at their written precision (`$78.74 billion` matches
  78,740 million; `$20.885 billion` does not match 20,878 million). A cited
  zero matches only zero. A percent matches only a calculator result, never a
  looked-up amount.
- Cited numbers keep their sign, and a currency amount without a unit word
  (`$78.74`) means that many dollars. Only a bare number may be read as
  thousands, millions, billions or a percent. Scale words: `trillion(s)`,
  `tn`, `t`, `billion(s)`, `bn`, `bln`, `bil`, `b`, `million(s)`, `mn`, `mln`,
  `mm`, `m`, `thousand(s)`, `k`; percent: `%`, `percent`, `pct`, `per cent`.
- Signs: a dash attached to an amount, or after a currency sign, is a minus
  for hyphen-minus, U+2010 to U+2015, U+2212, U+FE63 and U+FF0D. Spaced off
  after a word or `:`, `=`, `(`, `,`, `;` (`net income - $20.9 billion`), only
  a hyphen-minus or a true minus (U+2212, U+FE63, U+FF0D) is a sign; a spaced
  en or em dash is punctuation. After a digit, inside a word or as a bullet
  opening a line, a dash is not a sign.
- Dates, FY and quarter labels, form names, list markers, a bare year and a
  single-digit count followed by a word are exempt by context, never by value.
  A count or year followed by a scale or currency word (`7 hundred million`,
  `2023 dollars`) is an amount, not exempt. So is a date form followed by a
  unit or currency word, or a number right after a currency sign (`$9 may`,
  `USD 9 may`, `June 30 billion`, `Losses may 9 billion`,
  `30 June 2023 billion`). Genuine dates (`June 30, 2023`, `30 June 2023`,
  `May 9, 2024`) stay exempt.

Known limits (what the grader cannot catch) are listed in the README and in
reward-design.md. Do not describe the grader as stronger than that.

## Adversarial cases and their scores

All on the JNJ FY2021 restatement task unless noted; each has a test.

| case | reward |
|---|---|
| honest: looks up, computes, cites the block | 1.00 |
| honest, markdown-bold block (`**REVENUE:** 78,740,000,000`, `**ANSWER:** 26.52%`) | 1.00 |
| dodge: cites only the final number, no block | 0.70 |
| missing citation block (honest prose, no block) | 0.70 |
| labels swapped in the block | 0.70 |
| block value in fullwidth digits | 0.70 |
| one malformed call (extra key), then recovers | 0.95 |
| mental math: right number, not computed by a tool | 0.70 |
| memorised number laundered through `compute("26.52 * 1")` | 0.70 |
| margin spelled from unit constants or `x / x` | 0.70 |
| right answer, no tool calls | 0.00 |
| right margin from the wrong year (AAPL FY2022 vs FY2023) | 0.70 |
| stale, pre-restatement answer | 0.20 |
| wrong sign in prose (any attached dash, or a spaced hyphen-minus or true minus) | 0.70 |
| dropped scale on an input in prose (`$78.74`) | 0.70 |
| invented input within 0.5%, `compute("20885/78740*100")` (one extra call) | 0.675 |
| `x/x*7e9 + NI/REV*1e6` computed, then `$7 billion` cited (one extra call) | 0.675 |
| input cited as a percent (`net income 20,878,000,000%`) | 0.70 |
| extra ungrounded amount (`debt $7`, `assets $2023`) | 0.70 |
| scale-word variants (`7 billions`, `9 mln`, `7 hundred million`, `9 pct`) | 0.70 |
| a coarse zero (`$0 trillion`) | 0.70 |
| amount dressed as a date (`$9 may`, `June 30 billion`, `Losses may 9 billion`) | 0.70 |
| both lookups past the question's date | 0.00 |
| only malformed calls, then the right answer | 0.00 |
| freeloader: zero calls, newer session overlapping an older one on one `hud serve` | refused, `OverlappingEpisodes` (both) |

`demo.py` runs honest, dodge, mental_math, laundered, memorised and malformed
(1.00, 0.70, 0.70, 0.70, 0.00, 0.95). The end-to-end tests also require 1.00
for the honest agent on all 18 tasks.

## Adding a new question type

1. **Data.** If it needs a new concept, add its XBRL tags to `CONCEPTS` in
   `scripts/build_fixture.py` and regenerate `src/quantlab_env/fixture.json`. `data.concepts()`
   and `get_fundamentals` pick it up from the fixture; update the
   `get_fundamentals` docstring, which is the schema the agent sees.
2. **Tools.** Only if the task needs a new capability: add a function in
   `src/quantlab_env/tools.py` with string parameters and a docstring, and add
   it to `TOOLS`. `validate` checks calls against its signature, and `env.py`
   publishes everything in `TOOLS`. Teach `grader.build_evidence` which result
   fields count as evidence.
3. **Question.** In `src/quantlab_env/questions.py`, add a builder that computes
   the gold from `data.facts_as_of` (the same view the tools serve), the prompt
   ending in the citation block and the `ANSWER:` line, and a generator for the
   default set.
4. **Grader.** In `src/quantlab_env/grader.py`, the citation block (`_LABELS`,
   `_CITATION`, `citations()` and the `missing_inputs` check in `grade`, which
   reads the gold inputs off `Question` by concept name), `derived_from_inputs`
   (net income over revenue) and `PAR_CALLS` are specific to net margin.
   Generalise them (for example, the question exposes its required inputs,
   their labels and its par call count) rather than branching on type.
5. **Template.** Add an `@env.template` in `env.py` that builds the question,
   takes a generation with `EPISODE.begin()`, yields the prompt, raises
   `OverlappingEpisodes` if `EPISODE.overlapped(generation)`, grades, and
   calls `EPISODE.close(generation)` in a `finally`, like `net_margin`.
6. **Tasks.** Bind the defaults in `tasks.py`.
7. **Tests.** Add grader tests for the honest path and each relevant
   adversarial case in `tests/test_grader.py`, an answerability check in
   `tests/test_tools.py`, and an end-to-end case in `tests/test_env.py`.
8. **Docs.** Update this file, the README and `docs/`, and add a dated entry to
   `docs/decisions.md`.

## Rules

- No secrets and no email addresses in the repo. `SEC_USER_AGENT` comes from the
  environment only; examples use a placeholder without an `@`.
- No private QuantLab code. This is a port of ideas, written fresh. Describe
  QuantLab failures no more deeply than the public write-ups linked above.
- Fixtures must be public-safe: SEC public data only.
- No em or en dashes in docs.
- Do not change grading behaviour without a test that shows why.
- Run `uv run ruff format`, `uv run ruff check` and `uv run pytest` before
  committing.
- Keep HUD telemetry off (`HUD_TELEMETRY_ENABLED=false`). Prefer the tests and
  `demo.py` over HUD CLI commands.
- The environment itself never touches the network.
