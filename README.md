# hud-quantlab-env

A small [HUD](https://github.com/hud-evals/hud-python) RL environment ported from
the research agent in my QuantLab project
([write-up](https://www.rohanjk.xyz/projects/quantlab-agentic/)). The agent answers a
financial question by looking facts up in SEC filings through MCP tools, and the
reward checks both the answer and whether every number it cites came from those
tools during the episode.

## The task

One question type, point-in-time net margin:

> What was Johnson & Johnson (JNJ)'s net margin for the fiscal year ending
> 2022-01-02, using only filings available on 2024-03-17?

To answer it, the agent makes two lookups and one calculation. The gold answer is
computed from the same data the tools serve. There are 18 tasks (`tasks.py`): five
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
| answer | The number on the final `ANSWER:` line is within 0.5% of the gold margin (as a percent or a fraction), and at least one admissible lookup succeeded. With no evidence, a correct answer earns nothing. |
| grounding | All or nothing. The citation block must give the revenue and net income the gold was computed from, each under its own label, exactly as an admissible lookup returned them; the final answer must match a calculator result derived from the question's net income over its revenue; and every other number the answer cites must match evidence from this episode. |
| validity | The share of tool calls that were well formed. It pays nothing unless an admissible lookup succeeded, so it is also 0 with no calls. |
| efficiency | `3 / max(3, calls)`. It pays nothing unless an admissible lookup succeeded. |

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

[docs/reward-design.md](docs/reward-design.md) has the full table with
subscores.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Data access needs no network: the
tools read an SEC fixture that ships inside the package. Keep HUD telemetry off.

```bash
export HUD_TELEMETRY_ENABLED=false
uv sync
uv run pytest                 # 292 grader, tools, fixture and end-to-end tests, offline
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
uv run hud task list --source tasks.py
uv run hud task start net_margin-83d5f782 --source tasks.py
uv run hud task grade net_margin-83d5f782 --source tasks.py --answer "ANSWER: 26.52%" --json
```

To run a real model (this needs a provider key and costs money):

```bash
export HUD_TELEMETRY_ENABLED=false
export ANTHROPIC_API_KEY=...        # or HUD_API_KEY with --gateway
uv run hud eval tasks.py claude --task-ids net_margin-83d5f782 --max-steps 10
uv run hud eval tasks.py claude --all --max-steps 10   # all 18 tasks
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

`src/quantlab_env/fixture.json` (29 KB) holds annual revenue and net income for
AAPL, MSFT, NVDA, KO and JNJ, periods ending 2021 to 2024. It ships as package
data and is loaded with `importlib.resources`, so an installed wheel works
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
env.py                         HUD Environment, MCP tool server, net_margin template, session guard
tasks.py                       the 18 concrete tasks
demo.py                        scripted agents, end to end
Dockerfile.hud                 container image for hud serve / hud deploy
src/quantlab_env/data.py       point-in-time fixture access
src/quantlab_env/tools.py      tools, strict argument checks, episode call log and open episodes
src/quantlab_env/grader.py     reward
src/quantlab_env/questions.py  question template and gold answers
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
- Compared with QuantLab this has one question type and two tools, not many
  question types and six tools. Grounding here checks every cited number, not
  only the final answer. That idea comes from the earlier memo writer's checker.
- The checked-in fixture is missing three NVIDIA revenue rows (FY2021 and
  FY2022 as re-reported in its 2023 and 2024 10-Ks), dropped by the old
  per-period tag precedence. No default question can see them, so no gold
  changes. The fixed builder should add them when the fixture is regenerated;
  `tests/test_build_fixture.py` pins the gap until then.

## Known limits

What the grader still cannot catch (details in
[docs/reward-design.md](docs/reward-design.md#known-limits)):

The two inputs are checked exactly, by label, in the citation block. These
limits apply to the other numbers the answer cites in prose:

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
  exempt as a count.
