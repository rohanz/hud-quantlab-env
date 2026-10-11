# hud-quantlab-env

A small [HUD](https://github.com/hud-evals/hud-python) RL environment ported from
the research agent in my QuantLab project
([write-up](https://www.rohanjk.xyz/projects/quantlab-agentic/)). The agent answers a
financial question by looking facts up in SEC filings through MCP tools, and the
reward checks both the answer and whether every number it cites came from those
tools during the episode.

## The task

Point-in-time ratios over annual revenue and net income, in three task sets
(legacy, probe and chained analyst questions). The legacy set asks one question
type, point-in-time net margin:

> What was Johnson & Johnson (JNJ)'s net margin for the fiscal year ending
> 2022-01-02, using only filings available on 2024-03-17?

To answer it, the agent makes two lookups and one calculation. The gold answer is
computed from the same data the tools serve. There are 18 legacy tasks (`tasks.py`): five
companies, fiscal years 2022 to 2024, each asked 30 days after the 10-K came out.
Two extra JNJ tasks re-ask a year after JNJ restated its revenue (its 2024 10-K
recast earlier years after the Kenvue separation). Asked on 2022-03-19, the FY2021
margin is 22.26%. Asked on 2024-03-17, it is 26.52%. Both answers are right for
their date, and an agent that ignores the date gets one of them wrong.

The fiscal year is named by its end date because fiscal-year labels vary by company
(NVIDIA's fiscal 2024 ended in January 2024). In QuantLab the base model scored
0.54 on exactly this kind of question.

The prompt asks the agent to end with a citation block and the answer, citing
the inputs exactly as `get_fundamentals` returned them:

```
REVENUE: 78740000000
NET_INCOME: 20878000000
ANSWER: 26.52%
```

### The probe set

The legacy set is too easy to separate models: base Qwen3.5-9B scored 0.982
on it (see [docs/decisions.md](docs/decisions.md)). The probe set is a
harder, larger, held-out set built from the same tools and grader, with its
difficulty coming from the questions themselves:

| type | asks for | inputs | par calls |
|---|---|---|---|
| `fy_net_margin` | net margin for "fiscal 2023", named by label, not end date | 2 | 3 |
| `latest_net_margin` | net margin for the latest fiscal year with a 10-K filed by a date | 2 | 3 |
| `net_margin_change` | change in net margin over the prior fiscal year, in percentage points | 4 | 3 |
| `revenue_growth` | revenue growth over the prior fiscal year, in percent | 2 | 2 |
| `net_margin_gap` | one company's net margin minus another's for the same fiscal-year label, in pp | 4 | 5 |

- **Fiscal-year labels.** A fiscal year is labelled by the calendar year it
  ends in, except that a 52/53-week year ending in the first week of January
  belongs to the year before: NVIDIA's fiscal 2024 ends 2024-01-28, Apple's
  fiscal 2023 ends 2023-09-30, JNJ's fiscal 2021 ends 2022-01-02. That is how
  every company in the fixture names its own years, and the prompt says only
  that fiscal years are named as the company names them; the tools return
  period end dates, not labels.
- **Latest year by a date.** Asked the day before each 10-K, so the latest
  fiscal year filed is the previous one even when the calendar suggests a
  newer one has ended (NVIDIA on 2024-02-20: fiscal 2023, not 2024).
- **Restatements.** Every question whose inputs a later filing changed is
  asked both before and after the change (JNJ's Kenvue recast). Year-over-year
  questions are never asked across bases: a restated year is compared only
  with a prior year from the same filing.
- **Two-year and two-company arithmetic.** Four inputs, one calculation, and a
  sign that matters. Changes, growth rates and gaps under 0.1 percentage
  points are not asked, since they round to nothing at two decimals.

The set is derived from the fixture (`questions.probe_questions`), so it grows
when the fixture does: 350 tasks with the 15-company fixture (84
`fy_net_margin`, 68 `latest_net_margin`, 58 `net_margin_change`,
65 `revenue_growth`, 75 `net_margin_gap`), mostly companies with
non-December fiscal years, periods ending 2021 to 2025. Each type has its own template; the grader reads the inputs,
citation labels, gold formula and par call count off the question, so the
reward is the same formula for every type.

### Chained analyst questions (probe v2)

Measured on 2026-10-06 on the frozen 100-task v1 subset (`tasks_probe100.py`),
Qwen3.5-9B base scored 0.901 (target 0.55 to 0.80) and its SFT model 0.864.
Single-company, one-hop types were nearly solved (`fy_net_margin` 0.977,
`latest_net_margin` 0.979); the only multi-company type, `net_margin_gap`, was
the hardest (0.782). So the chained types ask what an analyst or PM asks about
several companies or several years at once, with 5 to 12 tool calls where
later steps depend on earlier results (which year is the latest, which margin
is the highest):

| type | an analyst asks this when | example | par calls |
|---|---|---|---|
| `peer_margin_rank` | building a comp table and naming the best or worst operator in a peer group | Which of Apple, Microsoft, Oracle and Cisco had the lowest net margin for its fiscal 2021, using only filings available on 2021-11-28? (`ANSWER: CSCO 21.26%`) | 9 (three companies) or 12 (four) |
| `peer_margin_change_rank` | asking who expanded margins most this year, the usual earnings-season question | Which of GE, 3M and FedEx expanded its net margin the most from fiscal 2021 to fiscal 2022, as of 2023-03-12? (`ANSWER: GE 9.08 pp`) | 9 |
| `margin_vs_peers` | judging whether a company earns a premium or discount margin to its sector | By how many pp was FedEx's fiscal 2022 net margin above the combined net margin of Deere, GE and 3M (their total net income over total revenue)? (`ANSWER: -3.95 pp`) | 9 |
| `margin_trend` | checking a "margins expand every year" thesis before a 10-K lands | Did NVIDIA's net margin rise in each of its last three fiscal years reported in a 10-K filed on or before 2025-02-25? Give YES or NO and the smallest annual change. (`ANSWER: NO -20.04 pp`) | 5 |

- **Peer groups** are sectors in the fixture (`questions.PEER_GROUPS`): large-cap
  technology (AAPL, MSFT, ORCL, CSCO), semiconductors and hardware (NVDA, MU,
  AAPL, CSCO), consumer (KO, NKE, COST, WMT) and industrials (DE, GE, MMM,
  FDX). Most fiscal years do not end in December, and each company's year is
  named by its own label, so one question mixes years ending in January, May,
  July, September and December.
- **Restatements across companies.** Every restatement of any member's figures
  is asked before and after. GE's restated fiscal 2023 moves the highest margin
  of Deere, GE and FedEx from Deere (16.60%, as of 2024-03-03) to GE (26.82%, as
  of 2025-03-05).
- **Rankings and yes/no answers are scored by a tested rule.** The gold is the
  picked company's margin (or change), or the smallest annual change for a
  trend, and the ANSWER line must also name the choice (the ticker, or YES or
  NO) as a whole word, in any case, and no other candidate. A pick is asked
  only if it stands clear of every other candidate by 0.1 pp and twice the
  answer tolerance, so the number alone identifies it.
- **The combined margin** is defined in the prompt as total net income over
  total revenue, how sector margins are aggregated. A simple average of the
  peers' margins needs a literal count, which the calculator rules reject.

`questions.chain_questions()` derives 294 chained tasks from the fixture (113
`peer_margin_rank`, 62 `peer_margin_change_rank`, 96 `margin_vs_peers`, 23
`margin_trend`). The frozen v2 subset `tasks_probe_v2.py` has 100 tasks: 20 of
each chained type and 4 of each v1 type for continuity, seeded and pinned by a
hash in `tests/test_probe_sample.py`. `tasks_probe100.py` (v1) is unchanged.

## Tools (MCP)

`env.py` starts a FastMCP server in an `@env.initialize` hook and publishes it as
the `tools` capability (`mcp/2025-11-25`). An agent that speaks MCP sees:

- `get_fundamentals(ticker, concept, as_of)` returns annual `revenue` or
  `net_income` rows, as known on `as_of`. For each period, the newest filing made
  on or before `as_of` wins, so nothing filed later is visible.
- `compute(expr)` does arithmetic on literal numbers only (`+ - * /`, parentheses).
  It accepts no names, function calls or powers, so a remembered figure can only
  enter as a literal, where the grader can check it.

A FastMCP middleware sends every call to one strict executor (`src/quantlab_env/tools.py`),
which also logs it for the episode. This matters because FastMCP's own validation
turns `"2024"` into `2024`, and it rejects bad calls before the tool function runs,
so those calls would never be logged. Here an unknown key, a missing key or a
non-string value comes back as a `bad_argument` error that the agent can read and
the grader counts.

## Reward

```
reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency
```

| component | rule |
|---|---|
| answer | The number on the final `ANSWER:` line is within 0.5% of the gold margin (as a percent or a fraction), and at least one admissible lookup succeeded. With no evidence, a correct answer earns nothing. A ranking or trend question also needs its choice (a ticker, or YES or NO) on that line, and no other candidate. |
| grounding | All or nothing. The citation block must give the revenue and net income the gold was computed from, each under its own label, exactly as an admissible lookup returned them; the final answer must match a calculator result derived from the question's net income over its revenue; and every other number the answer cites must match evidence from this episode. |
| validity | The share of tool calls that were well formed. It pays nothing unless an admissible lookup succeeded, so it is also 0 with no calls. |
| efficiency | `par / max(par, calls)`, par 3 for net margin (one lookup per ticker and concept plus one calculation, or one per figure a ranking or trend compares). It pays nothing unless an admissible lookup succeeded. |

The citation block (`citations` in `src/quantlab_env/grader.py`):

- Each value must be the whole USD value with its sign, exactly as the tool
  returned it. Only formatting is allowed: an optional leading `$` and
  thousands commas. No rounding, no scale words, no decimals, no parentheses.
- Each label is checked against its own concept: `REVENUE:` must equal a
  revenue row, and `NET_INCOME:` a net income row, for the question's ticker
  and period, from an admissible lookup in this episode, and equal to the value
  the gold used. Swapping the labels fails both.
- Labels (`REVENUE`, `NET_INCOME` and `ANSWER`) are case-insensitive and must
  open their line, after any markdown list, quote, heading or emphasis markers
  (`**REVENUE:** 78,740,000,000`, `- REVENUE:`, `> REVENUE:`,
  `**ANSWER:** 26.52%`). A value may be wrapped only in `*` or `_` emphasis,
  and must be written in ASCII digits: fullwidth or Arabic-Indic digits count
  as missing.
- If a label appears on several lines, the last one wins. Because labels are
  case-insensitive, a later prose line such as `Revenue: ...` counts as a
  block line too, as a later `Answer:` line already did.

How evidence is built from the episode's call log (`src/quantlab_env/grader.py`):

- Only `value` fields of successful lookups count. Dates, fiscal years and
  other metadata never count.
- Filing-date cutoff: a lookup with an `as_of` later than the question's date
  saw filings the question rules out. It is not malformed, but it is not
  evidence and does not count as a successful lookup.
- Calculator provenance: a `compute` result counts only if it is derived from
  evidence. Every literal must equal evidence at the precision it is written
  with, or be a unit constant (1, 100, 1e3, 1e6, 1e9), constants may only scale
  evidence, each side of `+` and `-` must itself depend on a looked-up value
  (so `x / x * 7e9 + x / y` is not evidence), and the result must depend on a
  looked-up value. The grader tracks each result as a formula over the
  looked-up values, and the final answer must be proportional to the
  question's net income over its revenue.
- Error results never count, even when they echo the agent's arguments.

How the other numbers in the answer (the prose) are checked:

- Cited numbers are read with units (`$78.74 billion`, `78,740 million`,
  `26.52%`) and compared with their sign, at the precision they are written
  with: `$78.74 billion` is a rounding of 78,740 million, but `$20.885 billion`
  is not 20,878 million. The 0.5% tolerance applies only to the final answer
  against gold. A currency amount with no unit means what it says: `$78.74` is
  78.74 dollars, not 78.74 billion. A cited zero matches only zero, so
  `$0 trillion` is not a rounding of every amount.
- Scale words are `trillion(s)`, `tn`, `t`, `billion(s)`, `bn`, `bln`, `bil`,
  `b`, `million(s)`, `mn`, `mln`, `mm`, `m`, `thousand(s)`, `k`; percent is
  `%`, `percent`, `pct` or `per cent`. A percent matches only a calculator
  result, never a looked-up amount.
- A dash attached to an amount, or after a currency sign (spaced or not), is a
  minus (`-$20.9 billion`, `$ - 20.9 billion`), whichever dash it is:
  hyphen-minus, U+2010 to U+2015 (hyphen, non-breaking hyphen, figure dash, en
  dash, em dash, horizontal bar), U+2212, U+FE63 or U+FF0D. Spaced off after a
  word or after `:`, `=`, `(`, `,` or `;` (`net income - $20.9 billion`,
  `income: - 20,878 million`), only a hyphen-minus or a true minus (U+2212,
  U+FE63, U+FF0D) is a sign. A spaced en or em dash there is punctuation,
  because models write `Revenue <em dash> $78.74 billion` that way; the inputs
  themselves are checked exactly in the block. A dash after a digit
  (`2021 - 2022`), a word hyphen and a bullet at the start of a line are not
  signs.
- Dates, `FY2023`, `Q4`, form names like `10-K`, list markers at the start of a
  line, a bare year and a single-digit count followed by a word (`3 filings`)
  are not claims. Context decides, not value: `$7`, `7%` and `$2023` are
  claims, and so is a count or year followed by a scale or currency word
  (`7 hundred million`, `2023 dollars`). The same holds for dates: a date form
  followed by a unit or currency word, or a number right after a currency sign,
  is an amount (`$9 may`, `USD 9 may`, `June 30 billion`,
  `Losses may 9 billion`, `30 June 2023 billion`), while `June 30, 2023`,
  `30 June 2023` and `May 9, 2024` are still dates.

The weights differ from QuantLab's (0.7 / 0.1 / 0.1 / 0.1). There, a correct
numeric answer that could not be traced lost only the 0.1 grounding share. Here
the code guarantees that a correct answer with no successful lookup scores 0,
and a correct answer that fails grounding scores at most 0.7, against 1.0 for a
grounded one. That makes an ungrounded answer strictly worse than a grounded
one for the cases the grader can see; see [Known limits](#known-limits) for the
ones it cannot.

## Adversarial cases

The cases in bold come from failures in the real project. Each case has a test in
`tests/test_grader.py`, and the scripted ones also run end to end in
`tests/test_env.py` and `demo.py`. Rewards are for the JNJ FY2021 restatement
task unless noted.

| case | where it comes from | reward here |
|---|---|---|
| Honest answer with the citation block | baseline | 1.00 |
| Honest answer with a markdown block | `**REVENUE:** 78,740,000,000` and `**ANSWER:** 26.52%`, as models often write it. Labels tolerate markdown markers and case. | 1.00 |
| **Cites fewer numbers to dodge the check** | In the earlier memo writer, DPO on pass/fail pairs learned to cite fewer numbers instead of correct ones ([write-up](https://www.rohanjk.xyz/projects/quantlab-analyst/)). A grader that only checks the numbers present rewards that. Here the answer must cite its inputs in the citation block, and a test checks that leaving them out scores no better than citing them next to a wrong number. | 0.70 |
| Missing citation block | Honest prose that cites the inputs, but no `REVENUE:` and `NET_INCOME:` lines. | 0.70 |
| Labels swapped in the block | `REVENUE: 20878000000` and `NET_INCOME: 78740000000`. Each label is checked against its own concept. | 0.70 |
| Block value in non-ASCII digits | `REVENUE: ７８,７４０,０００,０００` (fullwidth). Block values must be ASCII digits. | 0.70 |
| **Malformed tool call** | The GRPO training loop parsed tool calls slightly too leniently, and the RL model lost points exactly where it had been lenient. Here one strict contract applies everywhere, and bad calls are logged and counted. | 0.95 (one bad call, then recovery) |
| **Correct number not in the evidence** | Models fetched the inputs and then did the arithmetic in their heads: right answers that nobody could verify. | 0.70 |
| **Memorised number laundered through the calculator** | `compute("26.52 * 1")`, or 26.52 spelled from unit constants (`100/(1+1+1+1)+1+...`) or with `x / x`. Caught by calculator provenance. | 0.70 |
| **Right answer, no evidence** | Answering from memory. | 0.00 |
| Right margin from the wrong year | Apple's FY2022 and FY2023 margins are both 25.31%. The citation block and provenance checks catch an agent that used the wrong year's rows. | 0.70 |
| Stale, pre-restatement answer | The point-in-time trap. | 0.20 |
| Wrong sign on an input in prose | `-$20.878 billion` or `net income -$20,878 million` with any attached dash (en dash, em dash, U+2212 and the rest), or `net income - $20.878 billion` with a spaced hyphen-minus or true minus, for a profit. | 0.70 |
| Dropped scale on an input in prose | `$78.74` for 78.74 billion. | 0.70 |
| Invented input within 0.5% of the real one | `compute("20885/78740*100")` and `net income $20.885 billion` when the filing says 20,878 million. Matching is at written precision, so neither is evidence. | 0.675 (one extra compute call) |
| Invented amount riding on a small evidence term | An extra `compute("(20878/20878+...)*1e9 + 20878/78740*1e6")` returns 7,000,265,151, then the answer adds `Total debt was $7 billion.` Each side of `+` must depend on a looked-up value, and `x / x` does not. | 0.675 (one extra compute call) |
| Input cited as a percent | `net income 20,878,000,000%` or `revenue 100%` in prose. A percent matches only a calculator result. | 0.70 |
| Extra ungrounded amount | `debt $7`, `assets $2023`, or a scale-word variant: `7 billions`, `9 mln`, `7 hundred million`, `9 pct`. | 0.70 |
| A coarse zero | `Debt was $0 trillion.` A zero matches only zero. | 0.70 |
| An amount dressed as a date | `Debt of $9 may be refinanced`, `Cash at June 30 billion`, `Losses may 9 billion`. A currency sign before or a unit word after makes it an amount. | 0.70 |
| Lookups past the question's date | Both inputs looked up as of 2099-01-01. No admissible lookup, so validity and efficiency are 0 too. | 0.00 |
| Burst of malformed calls, then the right answer | Not from QuantLab: an efficiency leak (0.1 for zero evidence) caught by the end-to-end test during development, now closed. | 0.00 |
| Freeloader session | Not from QuantLab: against one `hud serve`, a newer session makes no calls and submits an older session's answer, which used to score 1.00 on the older session's calls. Tested end to end in `tests/test_env.py`. | refused (`OverlappingEpisodes`, both sessions) |

Chained questions (each also in `tests/test_grader.py`):

| case | reward |
|---|---|
| honest, every chained task | 1.00 |
| rank: right margin, wrong or no ticker, or two tickers named | 0.50 |
| rank: another company's margin under its ticker | 0.20 |
| rank: one peer's figures missing from the block, or not looked up | 0.70 |
| rank across GE's restatement, with the pre-restatement leader | 0.20 |
| rank with NVIDIA's fiscal 2024 read as the year ending 2025-01-26 | 0.20 |
| change rank by this year's margin instead of the change | 0.20 |
| combined margin as a simple average of the peers' margins | 0.20 |
| combined margin in two calculations (one call over par) | 0.99 |
| peers minus the company (sign reversed) | 0.50 |
| trend: wrong word (`YES` for a fall), both words, or none | 0.50 |
| trend: the latest change instead of the smallest | 0.20 |
| trend: lookups past the date (a newer year) | 0.00 |

[docs/reward-design.md](docs/reward-design.md) has the full table with
subscores.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Data access needs no network: the
tools read an SEC fixture that ships inside the package. Keep HUD telemetry off.

```bash
export HUD_TELEMETRY_ENABLED=false
uv sync
uv run pytest                 # 1657 grader, tools, fixture and end-to-end tests, offline
uv run python demo.py         # scripted agents through the env over MCP, offline
```

`demo.py` runs six scripted HUD `Agent`s (honest, dodge, mental_math, laundered,
memorised, malformed) on the JNJ restatement task, using `LocalRuntime`. Each one
opens the `tools` capability, calls tools over HTTP, and is graded by the
template. No model is called. Current output:

```
task: net_margin-83d5f782

agent        reward  answer grounding validity efficiency  note
honest         1.00    1.00      1.00     1.00       1.00  ungrounded=[] missing=[]
dodge          0.70    1.00      0.00     1.00       1.00  ungrounded=[] missing=['revenue', 'net_income']
mental_math    0.70    1.00      0.00     1.00       1.00  ungrounded=['26.52%', '26.52%'] missing=[]
laundered      0.70    1.00      0.00     1.00       1.00  ungrounded=['26.52%', '26.52%'] missing=[]
memorised      0.00    0.00      0.00     0.00       0.00  ungrounded=['26.52%'] missing=['revenue', 'net_income']
malformed      0.95    1.00      1.00     0.75       0.75  ungrounded=[] missing=[]
```

`dodge` gives the margin without the citation block, so both inputs are
missing. `mental_math` and `laundered` cite a margin that no calculator result
derived from evidence supports.

To try the task lifecycle by hand with the HUD CLI:

```bash
export HUD_TELEMETRY_ENABLED=false
uv run hud task list --source tasks.py        # all three sets; tasks_legacy.py or tasks_probe.py for one
uv run hud task start net_margin-83d5f782 --source tasks.py
uv run hud task grade net_margin-83d5f782 --source tasks.py --answer "ANSWER: 26.52%" --json
```

To run a real model (this needs a provider key and costs money):

```bash
export HUD_TELEMETRY_ENABLED=false
export ANTHROPIC_API_KEY=...        # or HUD_API_KEY with --gateway
uv run hud eval tasks.py claude --task-ids net_margin-83d5f782 --max-steps 10
uv run hud eval tasks_legacy.py claude --all --max-steps 10   # the 18 legacy tasks
uv run hud eval tasks_probe.py claude --all --max-steps 10    # the probe set
uv run hud eval tasks_probe_v2.py claude --all --max-steps 16 # the frozen v2 subset (up to 12 calls)
```

### Docker

`Dockerfile.hud` installs the project and its locked dependencies at build time
(the fixture is included as package data), sets `HUD_TELEMETRY_ENABLED=false`,
and serves the environment with `hud serve` on port 8765 as the unprivileged
user `app`. A `HEALTHCHECK` sends a JSON-RPC `hello` to port 8765 (the control
channel, not HTTP), which opens a session without starting a task.
`.dockerignore` keeps tests, docs, scripts, markdown, `demo.py` and `.room`
out of the image.

```bash
docker build -f Dockerfile.hud -t hud-quantlab-env:local .
docker run --rm -p 8765:8765 hud-quantlab-env:local
```

Run one episode at a time against a served container. The tool-call log is
per process, so concurrent rollouts against one `hud serve` (for example
`hud eval --runtime tcp://localhost:8765` with the default concurrency) are
not graded at all: when two episodes are open at once, every one of them
fails at grade time with `OverlappingEpisodes`. That is loud on purpose.
Before, the newest session won, so a session that made no tool calls could
score 1.00 on an older session's calls. For a `tcp://` runtime, pass
`--max-concurrent 1`, or run one process per rollout (`hud eval` with its
local runtime, or `DockerRuntime`). See [Limits](#limits) for what the guard
still cannot do.

`hud deploy` builds the same file for the HUD platform. It has not been
deployed there yet.

## Data

`src/quantlab_env/fixture.json` (104 KB) holds annual revenue and net income for
15 companies (AAPL, MSFT, NVDA, KO, JNJ, ORCL, CSCO, NKE, FDX, COST, MU, WMT,
DE, GE and MMM), periods ending 2021 to 2025, and records each filing's
fiscal-year focus (`fy`), which a test checks against the fiscal-year
labelling rule. Walmart's revenue is total revenues, not net sales (its own
tag order, `TAG_OVERRIDES` in the build script). It ships
as package data and is loaded with `importlib.resources`, so an installed wheel works
without the source checkout. It keeps every 10-K filing that reported each
period, which is what makes point-in-time queries work. The data comes from the
SEC's public XBRL `companyfacts` API. To regenerate it (this is the only step
that uses the network):

```bash
SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py
```

The environment never touches the network.

## Layout

```
env.py                         HUD Environment, MCP tool server, one template per question type, session guard
tasks.py                       the three named task sets: legacy (18), probe and chains (derived from the fixture)
tasks_legacy.py, tasks_probe.py  one set each, for hud --source
tasks_probe100.py              the frozen v1 subset (100: 20 per v1 type)
tasks_probe_v2.py              the frozen v2 subset (100: 20 per chained type, 4 per v1 type)
tasks_grpo_<split>.py          GRPO training splits, rows read from $QUANTLAB_GRPO_SPLITS (grpo_splits.py)
demo.py                        scripted agents, end to end
Dockerfile.hud                 container image for hud serve / hud deploy
src/quantlab_env/data.py       point-in-time fixture access
src/quantlab_env/tools.py      tools, strict argument checks, episode call log and open episodes
src/quantlab_env/grader.py     reward
src/quantlab_env/questions.py  question types, fiscal-year labels, peer groups, gold answers, the task sets and subsets
src/quantlab_env/fixture.json  the SEC data the tools serve (package data)
scripts/build_fixture.py       regenerate the fixture from EDGAR
tests/                         grader, tools, fixture, fixture builder and end-to-end tests
docs/                          design notes (see below)
```

## Docs

- [AGENTS.md](AGENTS.md): working guide for this repo (commands, rules, how to
  add a question type).
- [docs/reward-design.md](docs/reward-design.md): the grader in detail and why
  each adversarial case exists.
- [docs/environment.md](docs/environment.md): tools, MCP surface, task template,
  fixture schema.
- [docs/decisions.md](docs/decisions.md): dated design decisions.

## Limits

- One episode at a time per env process. The call log is process state, like the
  board in HUD's 2048 cookbook, and an MCP tool call carries no session id, so
  calls cannot be split per session. `LocalRuntime` runs one acquisition of an
  env instance at a time, and `hud eval` and `DockerRuntime` run each rollout
  in its own process, so these are safe. One `hud serve` process can still
  host concurrent sessions. Each episode is opened when it starts and closed
  when it is graded, fails or is closed by HUD. If another episode is open
  when one begins, all open episodes, the new one included, fail at grade
  time with `OverlappingEpisodes` instead of being graded on a log that is not
  their own. What remains: calls still cannot be attributed to sessions, so
  the guard refuses overlapping episodes rather than grading them. A session
  that is abandoned without being closed blocks every overlapping episode
  until it expires after `QUANTLAB_EPISODE_TIMEOUT` seconds (default 3600).
  After it expires, a newer episode is graded normally, and if the abandoned
  session comes back after that newer one began, it is refused; any tool
  calls it makes in the meantime land in the newer episode's log.
- Compared with QuantLab this has ten question types over two concepts and
  two tools, not many question types and six tools. Grounding here checks every cited number, not
  only the final answer. That idea comes from the earlier memo writer's checker.
- The checked-in fixture is missing three NVIDIA revenue rows (FY2021 and
  FY2022 as re-reported in its 2023 and 2024 10-Ks), dropped by the old
  per-period tag precedence. They repeat values already in the fixture, so no
  gold changes. The fixed builder should add them when the fixture is regenerated;
  `tests/test_build_fixture.py` pins the gap until then.

## Known limits

What the grader still cannot catch (details in
[docs/reward-design.md](docs/reward-design.md#known-limits)):

The two inputs are checked exactly, by label, in the citation block. These
limits apply to the other numbers the answer cites in prose:

- The probe set's difficulty is by design. The v1 subset was too easy (base
  0.901 against a 0.55 to 0.80 target); the chained v2 subset has not been run
  on a model yet, so whether it lands in the band is untested.
- Choices are read as whole words in any case, so a ticker that is also a word
  (`COST`, `DE`, `GE`, `KO`) counts as named wherever it appears on the ANSWER
  line (`ANSWER: WMT, the lowest cost 2.39%` names COST too, and is wrong).
- A ranking's gold is the chosen company's own calculation. An agent that
  looks up and cites every peer but computes only the one it guessed is not
  penalised for skipping the comparison. 22 of the 23 trend tasks are NO; the
  word alone is easy to guess, but the smallest change must be right too.
- The fiscal-year labelling rule is one fixed rule. A company that labels a
  year by the calendar year it starts in would get wrong golds, so none is in
  the build list; after regenerating, `tests/test_tools.py` checks the rule
  against each filing's recorded fiscal-year focus. Revenue is the first
  matching XBRL tag in `CONCEPTS`; for a newly added company that tag may
  cover only part of total revenue, so check the new companies' values
  against their 10-Ks after regenerating.
- The answer must be proportional to the question's formula, so a negated
  result (a gap in the reverse order, B minus A) is grounded; the answer
  subscore is what catches its sign (0.50).
- Subtracting 1 from a ratio of evidence (`a / b - 1`), or 100 from it in
  percent (`a / b * 100 - 100`), is allowed for growth, so a prose number
  equal to some ratio of looked-up values minus 1 (or in percent, minus 100)
  counts as grounded.
- Claims written in words ("twenty billion", "seven billion") are not
  extracted or checked, and superscript digits are not read (`7³ billion`
  reads as a bare 7).
- Prose matches values, not labels. A cited number only has to match some
  value the episode looked up (admissibly) or computed, so calling the revenue
  "net income", another year's figure, or another ticker's figures (any ticker
  may be looked up) cited as extra claims pass.
- A sign given in words or parentheses is not read: `net loss of $20,878
  million` and `$(20,878) million` read as positive. A spaced en or em dash
  before an amount (`net income <en dash> $20,878 million`) is read as
  punctuation, not a minus, and a hyphen opening a line is read as a bullet,
  so `- 20,878 million` at the start of a line reads as positive.
- A bare number may be read at any scale (itself, thousands, millions,
  billions, percent), so a wrong bare number can match by coincidence. Matching
  is at written precision, so a coarse rounding (`$21 billion` for 20,878
  million, or a literal `2.1e4` in `compute`) is accepted, and the fewer digits
  a number has, the more values it rounds from.
- Calculator results are evidence, and products and quotients of many
  looked-up values (other tickers included) can reach almost any coarse claim
  such as `$7 billion`.
- A bare year-like number (1900 to 2100) and a single-digit count followed by
  a word are exempt, even if meant as amounts (so `assets 2023` is not
  checked, while `assets $2023` and `assets 2023 dollars` are). A foreign
  currency or any other word after a single digit (`9 euros`, `3x`) leaves it
  exempt as a count. A bare `100` right after a multiply or divide operator
  (`× 100`) is exempt as part of a written formula.
- LaTeX math delimiters are read as dollar signs: `$$ 35.96\% $$` reads as a
  dollar amount of 35.96, so an honest answer written in LaTeX can lose
  grounding.
