# hud-quantlab-env

A small [HUD](https://github.com/hud-evals/hud-python) RL environment ported from
the research agent in my QuantLab project
([write-up](https://rohanjk.xyz/projects/quantlab-agentic)). The agent answers a
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
| answer | The number on the final `ANSWER:` line is within 0.5% of the gold margin (as a percent or a fraction), and at least one lookup succeeded. With no evidence, a correct answer earns nothing. |
| grounding | All or nothing. Every number the answer cites, the final one included, must match a value from a successful tool result in this episode. The answer must also cite the revenue and net income the gold was computed from. |
| validity | The share of tool calls that were well formed. |
| efficiency | `3 / max(3, calls)`. It pays nothing unless a lookup succeeded. |

How evidence is built from the episode's call log (`src/quantlab_env/grader.py`):

- Only `value` fields count. Dates, fiscal years and other metadata never count.
- A `compute` result counts only if every literal in its expression is already
  evidence, or is a unit constant (1, 100, 1e3, 1e6, 1e9). This is QuantLab's
  "transitive grounding" rule.
- Error results never count, even when they echo the agent's arguments.
- Cited numbers are read with units (`$78.74 billion`, `78,740 million`, `26.52%`)
  and compared as magnitudes, within 0.5%. ISO dates, `FY2023`, form names like
  `10-K`, bare years and single-digit list markers are not treated as claims.

The weights differ from QuantLab's (0.7 / 0.1 / 0.1 / 0.1). There, a correct
numeric answer that could not be traced lost only the 0.1 grounding share. Here a
correct answer with no evidence scores 0, and a correct answer with ungrounded
numbers scores at most 0.7, against 1.0 for the grounded version. So answering
from memory is always the worse strategy.

## Adversarial cases

These come from failures in the real project. Each has a unit test in
`tests/test_grader.py`, and most run end to end in `demo.py`.

| case | what happened in QuantLab | reward here |
|---|---|---|
| Honest answer with cited inputs | baseline | 1.00 |
| **Cites fewer numbers to dodge the check** | DPO learned "citation restraint": the passing and failing answers differed mostly in how many numbers they cited, so the model learned to cite less. A grader that only checks the numbers present rewards that. Here the answer must cite its inputs, and grounding is all or nothing, so dropping a bad number cannot raise the score. | 0.70 |
| **Malformed tool call** | In training, GRPO's parser silently dropped hallucinated `None`-valued keys that the evaluation parser rejected. The model learned the lenient contract and lost points where the two parsers disagreed. Here one strict contract applies everywhere, and bad calls are logged and counted. | 0.95 (one bad call, then recovery) |
| **Correct number not in the evidence** | Models fetched the inputs and then did the arithmetic in their heads: right answers that nobody could verify. Passing a remembered number through the calculator (`compute("26.52 * 1")`) is caught by the transitive rule. | 0.70 |
| **Right answer, no evidence** | Answering from memory. On restatement questions, memory is also often out of date. | 0.00 |
| Right margin from the wrong year | Apple's FY2022 and FY2023 margins are both 25.31%. The required-inputs check catches an agent that looked up the wrong year. | 0.70 |
| Stale, pre-restatement answer | The point-in-time trap. | 0.20 |
| Burst of malformed calls, no answer evidence | Not from QuantLab: an efficiency leak (0.1 for zero evidence) caught by the end-to-end test during development, now closed. | 0.00 |

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                 # grader, tools and end-to-end tests, offline
uv run python demo.py         # scripted agents through the env over MCP, offline
```

`demo.py` runs five scripted HUD `Agent`s (honest, dodge, mental_math, memorised,
malformed) on the JNJ restatement task, using `LocalRuntime`. Each one opens the
`tools` capability, calls tools over HTTP, and is graded by the template. No model
is called.

To try the task lifecycle by hand with the HUD CLI:

```bash
uv run hud task list --source tasks.py
uv run hud task start net_margin-83d5f782 --source tasks.py
uv run hud task grade net_margin-83d5f782 --source tasks.py --answer "ANSWER: 26.52%" --json
```

To run a real model (this needs a provider key and costs money):

```bash
export ANTHROPIC_API_KEY=...        # or HUD_API_KEY with --gateway
uv run hud eval tasks.py claude --task-ids net_margin-83d5f782 --max-steps 10
uv run hud eval tasks.py claude --all --max-steps 10   # all 18 tasks
```

To deploy to the HUD platform, run `hud deploy`, which builds `Dockerfile.hud`.

## Data

`data/fixture.json` (29 KB) holds annual revenue and net income for AAPL, MSFT,
NVDA, KO and JNJ, periods ending 2021 to 2024. It keeps every 10-K filing that
reported each period, which is what makes point-in-time queries work. The data
comes from the SEC's public XBRL `companyfacts` API. To regenerate it:

```bash
SEC_USER_AGENT="Your Name you@example.com" uv run python scripts/build_fixture.py
```

The environment never touches the network.

## Layout

```
env.py                     HUD Environment, MCP tool server, net_margin template
tasks.py                   the 18 concrete tasks
demo.py                    scripted agents, end to end
src/quantlab_env/data.py   point-in-time fixture access
src/quantlab_env/tools.py  tools, strict argument checks, episode call log
src/quantlab_env/grader.py reward
src/quantlab_env/questions.py  question template and gold answers
scripts/build_fixture.py   regenerate the fixture from EDGAR
```

## Limits

- One episode at a time per env process. The call log is process state, like the
  board in HUD's 2048 cookbook. `LocalRuntime` runs one acquisition of an env
  instance at a time, and `hud eval` runs each rollout in its own subprocess.
- Compared with QuantLab this has one question type and two tools, not thirteen
  templates and six tools. Grounding here checks every cited number, not only the
  final answer. That idea comes from QuantLab's earlier memo checker.
- Number extraction is regex based. A claim written in words ("twenty billion")
  is not checked.
