# AGENTS.md

Working guide for agents and humans in this repo. The README is the public
overview; this file is how to work on it.

## What this is

A [HUD](https://github.com/hud-evals/hud-python) RL environment that ports one
task type from the QuantLab research agent: **point-in-time net margin**. The
agent is asked for a company's net margin for a fiscal year "using only filings
available on" a given date. That is the 18-task legacy set; a harder probe set
(fiscal-year labels, latest year by a date, restatements, margin change,
revenue growth, two-company gaps) is derived from the fixture for use as a
held-out probe, and chained analyst questions (peers ranked by margin or margin
change, a company against its peers' combined margin, a three-year margin
trend) make the frozen probe v2 subset. It looks figures up through MCP tools backed by an
offline SEC fixture, does the arithmetic with a calculator tool, and is graded
on the answer and on whether every number it cites came from a tool result in
that episode.

Public background:
[quantlab-agentic](https://www.rohanjk.xyz/projects/quantlab-agentic/) and
[quantlab-analyst](https://www.rohanjk.xyz/projects/quantlab-analyst/).

## Layout

```
env.py                         HUD Environment, FastMCP tool server, strict middleware, one template per question
                               type (_episode does the work; TEMPLATES by kind), symmetric session guard
                               (OverlappingEpisodes, QUANTLAB_EPISODE_TIMEOUT)
tasks.py                       three named Tasksets: legacy (18, pinned), probe and chains (derived from the fixture)
tasks_legacy.py, tasks_probe.py  one set each, for hud --source (a module may not expose a task twice)
tasks_probe100.py              the frozen 100-task v1 probe subset (questions.probe_sample: 20 per type, seeded)
tasks_probe_v2.py              the frozen 100-task v2 subset (questions.probe_sample_v2: 20 per chained type, 4 per
                               v1 type, seeded); both subset modules keep their Taskset private for hud --source
demo.py                        six scripted HUD Agents run end to end over MCP, offline; Honest, honest_expr(s),
                               honest_pick and honest_answer: the honest strategy on any question type (tests)
Dockerfile.hud                 image for hud serve / hud deploy (installs at build, telemetry off, port 8765,
                               non-root user app, HEALTHCHECK); .dockerignore keeps dev files out
src/quantlab_env/data.py       fixture loading (importlib.resources) and point-in-time lookup (facts_as_of)
src/quantlab_env/tools.py      get_fundamentals, compute, strict validate/execute, Episode (call log, open
                               episodes, overlap marks, DEFAULT_EPISODE_TIMEOUT)
src/quantlab_env/grader.py     reward: answer (and a ranking's or trend's choice), grounding (citation block,
                               calculator provenance, prose check), validity, efficiency; reads inputs, labels,
                               formula, choice and par calls off the Question
src/quantlab_env/questions.py  Input, Question (inputs, gold formula, prompt with the citation block, choice and
                               choices, computes, par_calls), fiscal_year, one builder per type, PEER_GROUPS,
                               default_questions() (LEGACY, pinned), probe_questions() and chain_questions()
                               (derived from the fixture), stratified_sample, probe_sample(_v2), template_args
src/quantlab_env/fixture.json  package data: public SEC XBRL revenue and net income, 15 companies, 2021 to 2025
scripts/build_fixture.py       regenerate the fixture from SEC EDGAR (network, needs SEC_USER_AGENT); lists 15
                               companies and periods ending 2021 to 2025, records each filing's fy
tests/                         test_grader.py, test_tools.py, test_fixture.py, test_build_fixture.py (offline
                               units), test_env.py (end to end, every task, overlapping and freeloading
                               sessions over tcp, session expiry, every probe and chained task),
                               test_probe_sample.py (v1 and v2 hashes); 1654 tests (927 in test_grader.py, 678 in
                               test_env.py, 27 in test_tools.py); pytest's pythonpath
                               puts the root on sys.path
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
uv run hud task list --source tasks.py          # all three sets; tasks_legacy.py / tasks_probe.py for one
uv run hud task start net_margin-83d5f782 --source tasks.py
uv run hud task grade net_margin-83d5f782 --source tasks.py --answer "ANSWER: 26.52%" --json
```

Real-model eval (needs a provider key, costs money, ask before running):

```bash
export HUD_TELEMETRY_ENABLED=false
export ANTHROPIC_API_KEY=...             # or HUD_API_KEY with --gateway
uv run hud eval tasks.py claude --task-ids net_margin-83d5f782 --max-steps 10
uv run hud eval tasks_legacy.py claude --all --max-steps 10
uv run hud eval tasks_probe.py claude --all --max-steps 10
uv run hud eval tasks_probe_v2.py claude --all --max-steps 16   # chained tasks take up to 12 calls
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

The fixture was rebuilt on 2026-10-06 with all 15 companies (350 probe
tasks). After any rebuild, check each new company's revenue against its 10-K:
the first matching tag in `CONCEPTS` may be a partial revenue line. Walmart
was (net sales under the default first tag), so `TAG_OVERRIDES` gives it its
own order, recorded in the fixture as that company's `concepts`.

`pyproject.toml` pins `hud>=0.6.18,<0.7` and declares `fastmcp`, `uvicorn` and
`httpx` directly, because the code imports them.

## How the reward works

```
reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency
```

- **answer**: the first number on the last `ANSWER:` line is within 0.5% of the
  gold (as a percent or a fraction) or within 0.005 percentage points of it
  (two decimals), and at least one admissible `get_fundamentals` call
  succeeded. A ranking or trend (`Question.choice`) also needs the line to
  name its choice (the ticker, or `YES` or `NO`) as a whole word, in any
  case, and no other of `Question.choices` (`grader.chosen`). The gold is the
  picked figure itself, and a pick is asked only if it stands clear of every
  other candidate by 0.1 pp and twice the answer tolerance.
- **grounding** (all or nothing): the citation block is correct, the final
  answer matches a calculator result derived from the question's inputs by
  its gold formula (net income over revenue, for net margin), and every other
  cited number matches evidence from this episode (the prose check). Matching
  is at the precision a number is written with; the answer tolerances are
  only for the answer against gold.
- **validity**: share of calls without a malformed error code (`bad_argument`,
  `unknown_tool`, `unknown_ticker`, `unknown_concept`), and 0 unless an
  admissible lookup succeeded (so also 0 with no calls).
- **efficiency**: `par / max(par, calls)`, and 0 unless an admissible lookup
  succeeded. Par is one lookup per (ticker, concept) the question needs plus
  one compute (`Question.computes`, one per compared figure for a ranking or
  trend): 3 for net margin and margin change, 2 for revenue growth, 5 for a
  two-company gap, 9 for a three-peer ranking or for a company against three
  peers, 12 for a four-peer ranking, 5 for a trend.

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
`citations()` in `grader.py` reads the last line of each label (`_LABELS` for
net margin, `Question.labels` in general), and each value must equal a row of
its own input's concept, ticker and period, from an admissible lookup in this
episode, and the value the gold used. Probe labels: `PRIOR_REVENUE` and
`PRIOR_NET_INCOME` (the prior fiscal year), `<TICKER>_REVENUE` and
`<TICKER>_NET_INCOME` (each company of a gap). Chained labels:
`<TICKER>_REVENUE`, `<TICKER>_NET_INCOME`, `<TICKER>_PRIOR_REVENUE`,
`<TICKER>_PRIOR_NET_INCOME` (each peer) and `FY<YEAR>_REVENUE`,
`FY<YEAR>_NET_INCOME` (each year of a trend). A missing value, any other formatting, or a swapped value lands in
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
  result depends on a looked-up value. One offset is allowed, for growth: 1
  subtracted from a ratio of evidence (`a / b - 1`), or 100 from that ratio
  times 100 (`a / b * 100 - 100`).
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
- A bare `100` or `100%` right after a multiply or divide operator (`* 100`,
  `×100`, `\times 100`, `multiply by 100`) is part of a written formula and
  exempt. With a scale word, a currency sign or a sign (`× 100 billion`,
  `× $100`, `× -100`), or as any other value (`× 7`, `× 1,000,000,000`), it is
  a claim.

Known limits (what the grader cannot catch) are listed in the README and in
reward-design.md. Do not describe the grader as stronger than that.

## Adversarial cases and their scores

All on the JNJ FY2021 restatement task unless noted; each has a test.

| case | reward |
|---|---|
| honest: looks up, computes, cites the block | 1.00 |
| honest, markdown-bold block (`**REVENUE:** 78,740,000,000`, `**ANSWER:** 26.52%`) | 1.00 |
| honest, shows its working in prose (`20878000000 / 78740000000 × 100 = 26.52%`) | 1.00 |
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

Probe set (each has a test in `tests/test_grader.py`):

| case | reward |
|---|---|
| honest, every probe question | 1.00 |
| NVIDIA fiscal 2023 answered with the year ending 2024-01-28 | 0.20 |
| latest year: lookups past the date | 0.00 |
| JNJ margin change with pre-restatement figures | 0.20 |
| margin change cited as this year's margin | 0.20 |
| margin change from two margin computes and a subtraction | 0.96 |
| `0.23 pp` for a 0.234 pp change | 1.00 |
| growth as `(a / b - 1) * 100` | 1.00 |
| growth as `a / b * 100 - 100` | 1.00 |
| growth as `a / b * 100 - 1`, `a / b * 1e6 - 100`, `x / x * 100 - 100` (laundered) | 0.70 |
| revenue growth with an unneeded net income lookup | 0.967 |
| gap in the reverse order (B minus A) | 0.50 |
| gap with one company's block lines missing | 0.70 |

Chained questions (each has a test in `tests/test_grader.py`):

| case | reward |
|---|---|
| honest, every chained question | 1.00 |
| rank: right margin, wrong ticker, two tickers or none on the ANSWER line | 0.50 |
| rank: another company's margin under its ticker | 0.20 |
| rank: one peer's block lines missing, or its figures never looked up | 0.70 |
| rank across GE's restatement with the pre-restatement leader (DE, not GE) | 0.20 |
| rank with NVIDIA's fiscal 2024 read as the year ending 2025-01-26 | 0.20 |
| change rank by this year's margin, or with pre-restatement figures | 0.20 |
| vs peers: simple average of the peers' margins (`/ 3`) | 0.20 |
| vs peers: combined margin, then the difference (one call over par) | 0.99 |
| vs peers: peers minus the company | 0.50 |
| trend: `YES` for a fall, both words, or none | 0.50 |
| trend: the latest change instead of the smallest | 0.20 |
| trend: lookups past the date | 0.00 |

`demo.py` runs honest, dodge, mental_math, laundered, memorised and malformed
(1.00, 0.70, 0.70, 0.70, 0.00, 0.95). The end-to-end tests also require 1.00
for the honest agent on all 18 legacy tasks and for `demo.Honest` on every
probe and chained task.

## The probe set

`questions.probe_questions()` derives it from the fixture, so it grows when
the fixture does (350 tasks with the 15-company fixture). Difficulty comes
from design, never from keeping tasks a model failed: fiscal years named by
label (`questions.fiscal_year`: the calendar year a fiscal year ends in, a
first-week-of-January end counting as the year before), the latest year filed
by a date (asked the day before each 10-K), every restatement asked before
and after, and four-input arithmetic. Year-over-year questions never mix a
restated year with a prior year from another filing, and results under 0.1
percentage points are not asked. Do not prune or hand-pick probe tasks by
model results; the set and grader version are to be hashed and frozen.

v1 (`tasks_probe100.py`) was measured on 2026-10-06 and was too easy (base
0.901, SFT 0.864; one-hop single-company types near 0.98), so its numbers are
recorded and it stays frozen. `questions.chain_questions()` adds chained
analyst questions over the sector peer groups in `PEER_GROUPS` (294 tasks):
`peer_margin_rank`, `peer_margin_change_rank`, `margin_vs_peers` and
`margin_trend`, each with its "an analyst asks this when" rationale in
docs/decisions.md. The v2 subset (`tasks_probe_v2.py`) is 20 of each chained
type plus 4 of each v1 type, seeded (20261006) and pinned by sha256 in
`tests/test_probe_sample.py`. A new question type must be realistic (say
when an analyst asks it); reject artificial ones.

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
   default set. A ranking or yes/no question sets `choice`, `choices` and
   `computes`, makes the gold the picked figure, and asks only clear picks
   (`_pick`); set `params` if the template args cannot be read off the
   inputs.
4. **Grader.** Usually nothing: the grader reads the question's `inputs`
   (labels, tickers, concepts, periods, values), its gold `formula` and its
   `par_calls`, so a new type over the existing concepts needs no grader
   change. Never branch on `kind` in the grader. If the answer is not a ratio
   in percent or percentage points, revisit `_correct` and the prompt text.
5. **Template.** Add an `@env.template` in `env.py` that builds the question
   and relays it through `_episode` (which opens the episode, yields the
   prompt, refuses overlaps, grades and closes in a `finally`), like the
   others, and add it to `TEMPLATES`; teach `questions.template_args` its
   arguments.
6. **Tasks.** Add its generator to `chain_questions` (or a new named set in
   `tasks.py`); keep the legacy set and both frozen subsets pinned
   (`probe_questions` feeds v1, so adding to it changes v1's hash).
7. **Tests.** Teach `demo.honest_expr` (or `honest_exprs`) the calculation, add grader tests for
   the honest path and each relevant adversarial case in
   `tests/test_grader.py`; the answerability check in `tests/test_tools.py`
   and the end-to-end run in `tests/test_env.py` cover every probe task.
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
