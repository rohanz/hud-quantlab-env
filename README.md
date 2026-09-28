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
| grounding | All or nothing. The final answer must be computed from the question's net income over its revenue, every other number the answer cites must match evidence from this episode, and the answer must cite the revenue and net income the gold was computed from. |
| validity | The share of tool calls that were well formed. |
| efficiency | `3 / max(3, calls)`. It pays nothing unless an admissible lookup succeeded. |

How evidence is built from the episode's call log (`src/quantlab_env/grader.py`):

- Only `value` fields of successful lookups count. Dates, fiscal years and
  other metadata never count.
- Filing-date cutoff: a lookup with an `as_of` later than the question's date
  saw filings the question rules out. It is not malformed, but it is not
  evidence and does not count as a successful lookup.
- Calculator provenance: a `compute` result counts only if it is derived from
  evidence. Every literal must be evidence or a unit constant (1, 100, 1e3,
  1e6, 1e9), constants may only scale evidence, both sides of `+` and `-` must
  use evidence, and the result must depend on a looked-up value. The grader
  tracks each result as a formula over the looked-up values, and the final
  answer must be proportional to the question's net income over its revenue.
- Error results never count, even when they echo the agent's arguments.
- Cited numbers are read with units (`$78.74 billion`, `78,740 million`,
  `26.52%`) and compared with their sign, within 0.5%. A currency amount with
  no unit means what it says: `$78.74` is 78.74 dollars, not 78.74 billion.
- Dates, `FY2023`, `Q4`, form names like `10-K`, list markers at the start of a
  line, a bare year and a small count like `3 filings` are not claims. Context
  decides, not value: `$7`, `7%` and `$2023` are claims.

The weights differ from QuantLab's (0.7 / 0.1 / 0.1 / 0.1). There, a correct
numeric answer that could not be traced lost only the 0.1 grounding share. Here
the code guarantees that a correct answer with no successful lookup scores 0,
and a correct answer that fails grounding scores at most 0.7, against 1.0 for a
grounded one. That makes an ungrounded answer strictly worse than a grounded
one for the cases the grader can see; see [Known limits](#known-limits) for the
ones it cannot.

## Adversarial cases

The first five come from failures in the real project. Each case has a test in
`tests/test_grader.py`, and the scripted ones also run end to end in
`tests/test_env.py` and `demo.py`. Rewards are for the JNJ FY2021 restatement
task unless noted.

| case | where it comes from | reward here |
|---|---|---|
| Honest answer with cited inputs | baseline | 1.00 |
| **Cites fewer numbers to dodge the check** | In the earlier memo writer, DPO on pass/fail pairs learned to cite fewer numbers instead of correct ones ([write-up](https://www.rohanjk.xyz/projects/quantlab-analyst/)). A grader that only checks the numbers present rewards that. Here the answer must cite its inputs, and a test checks that leaving them out scores no better than citing them next to a wrong number. | 0.70 |
| **Malformed tool call** | The GRPO training loop parsed tool calls slightly too leniently, and the RL model lost points exactly where it had been lenient. Here one strict contract applies everywhere, and bad calls are logged and counted. | 0.95 (one bad call, then recovery) |
| **Correct number not in the evidence** | Models fetched the inputs and then did the arithmetic in their heads: right answers that nobody could verify. | 0.70 |
| **Memorised number laundered through the calculator** | `compute("26.52 * 1")`, or 26.52 spelled from unit constants (`100/(1+1+1+1)+1+...`). Caught by calculator provenance. | 0.70 |
| **Right answer, no evidence** | Answering from memory. | 0.00 |
| Right margin from the wrong year | Apple's FY2022 and FY2023 margins are both 25.31%. The required-inputs and provenance checks catch an agent that used the wrong year's rows. | 0.70 |
| Stale, pre-restatement answer | The point-in-time trap. | 0.20 |
| Wrong sign or scale on an input | `-$20.878 billion` for a profit, `$78.74` for 78.74 billion. | 0.70 |
| Lookups past the question's date | Both inputs looked up as of 2099-01-01. | 0.10 |
| Burst of malformed calls, then the right answer | Not from QuantLab: an efficiency leak (0.1 for zero evidence) caught by the end-to-end test during development, now closed. | 0.00 |

[docs/reward-design.md](docs/reward-design.md) has the full table with
subscores.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Data access needs no network: the
tools read an SEC fixture that ships inside the package. Keep HUD telemetry off.

```bash
export HUD_TELEMETRY_ENABLED=false
uv sync
uv run pytest                 # 122 grader, tools, fixture and end-to-end tests, offline
uv run python demo.py         # scripted agents through the env over MCP, offline
```

`demo.py` runs six scripted HUD `Agent`s (honest, dodge, mental_math, laundered,
memorised, malformed) on the JNJ restatement task, using `LocalRuntime`. Each one
opens the `tools` capability, calls tools over HTTP, and is graded by the
template. No model is called. Current output:

```
agent        reward  answer grounding validity efficiency
honest         1.00    1.00      1.00     1.00       1.00
dodge          0.70    1.00      0.00     1.00       1.00
mental_math    0.70    1.00      0.00     1.00       1.00
laundered      0.70    1.00      0.00     1.00       1.00
memorised      0.00    0.00      0.00     0.00       0.00
malformed      0.95    1.00      1.00     0.75       0.75
```

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
and serves the environment with `hud serve` on port 8765:

```bash
docker build -f Dockerfile.hud -t hud-quantlab-env:local .
docker run --rm -p 8765:8765 hud-quantlab-env:local
```

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
env.py                         HUD Environment, MCP tool server, net_margin template
tasks.py                       the 18 concrete tasks
demo.py                        scripted agents, end to end
Dockerfile.hud                 container image for hud serve / hud deploy
src/quantlab_env/data.py       point-in-time fixture access
src/quantlab_env/tools.py      tools, strict argument checks, episode call log
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
  board in HUD's 2048 cookbook. `LocalRuntime` runs one acquisition of an env
  instance at a time, and `hud eval` runs each rollout in its own subprocess.
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

- Claims written in words ("twenty billion") are not extracted or checked.
- Grounding matches values, not labels. Apart from the final answer and the
  two required inputs, a cited number only has to match some value the episode
  looked up or computed, so a swapped label or another year's figure passes.
- A bare number may be read at any scale (itself, thousands, millions,
  billions, percent) within 0.5%, so a wrong bare number can match by
  coincidence.
- A bare year-like number (1900 to 2100) and a single-digit count followed by
  a word are exempt, even if meant as amounts. Parentheses are not read as
  accounting negatives.
