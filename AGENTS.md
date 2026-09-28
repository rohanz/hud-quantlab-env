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
env.py                         HUD Environment, FastMCP tool server, strict middleware, net_margin template
tasks.py                       the 18 concrete tasks (one per default question)
demo.py                        five scripted HUD Agents run end to end over MCP, offline
src/quantlab_env/data.py       fixture loading and point-in-time lookup (facts_as_of)
src/quantlab_env/tools.py      get_fundamentals, compute, strict validate/execute, Episode call log
src/quantlab_env/grader.py     reward: answer, grounding, validity, efficiency
src/quantlab_env/questions.py  Question, gold answers, prompt text, default_questions()
scripts/build_fixture.py       regenerate data/fixture.json from SEC EDGAR (network, needs SEC_USER_AGENT)
data/fixture.json              public SEC XBRL data: revenue and net income, 5 companies, 2021 to 2024
tests/                         test_grader.py, test_tools.py (offline units), test_env.py (end to end)
docs/                          reward-design.md, environment.md, decisions.md
```

## Commands

Always use `uv`. Keep HUD telemetry off for anything that imports or runs HUD:
`export HUD_TELEMETRY_ENABLED=false`.

```bash
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

Regenerating the fixture (network, only when changing data):

```bash
SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py
```

## How the reward works

```
reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency
```

- **answer**: the first number on the last `ANSWER:` line is within 0.5% of the
  gold (as a percent or a fraction), and at least one `get_fundamentals` call
  succeeded.
- **grounding** (all or nothing): every cited number, the final answer
  included, matches evidence from this episode, and the answer cites the exact
  revenue and net income the gold used.
- **validity**: share of calls without a malformed error code (`bad_argument`,
  `unknown_tool`, `unknown_ticker`, `unknown_concept`). 0 with no calls.
- **efficiency**: `3 / max(3, calls)`, and 0 unless a lookup succeeded.

Strict tool calls: a FastMCP middleware (`StrictToolLog` in `env.py`) answers
every call from `tools.execute`, which validates the raw arguments (no unknown
keys, no missing keys, strings only, no coercion), runs the tool and logs the
call. Bad calls come back as `{"error": {"code", "message"}}` and are counted,
never repaired or dropped.

Calculator evidence rules: only `value` fields of successful lookups are
evidence. A `compute` result becomes evidence only if every literal in its
expression is already evidence or a unit constant (1, 100, 1e3, 1e6, 1e9).
Error results never count. Details: [docs/reward-design.md](docs/reward-design.md).

## Adversarial cases and their scores

All on the JNJ FY2021 restatement task unless noted; each has a test.

| case | reward |
|---|---|
| honest: looks up, computes, cites inputs | 1.00 |
| dodge: cites only the final number | 0.70 |
| one malformed call (extra key), then recovers | 0.95 |
| mental math: right number, not computed by a tool | 0.70 |
| memorised number laundered through `compute("26.52 * 1")` | 0.70 |
| right answer, no tool calls | 0.00 |
| right margin from the wrong year (AAPL FY2022 vs FY2023) | 0.70 |
| stale, pre-restatement answer | 0.20 |
| only malformed calls, then the right answer | 0.00 |

## Adding a new question type

1. **Data.** If it needs a new concept, add its XBRL tags to `CONCEPTS` in
   `scripts/build_fixture.py` and regenerate `data/fixture.json`. `data.concepts()`
   and `get_fundamentals` pick it up from the fixture; update the
   `get_fundamentals` docstring, which is the schema the agent sees.
2. **Tools.** Only if the task needs a new capability: add a function in
   `src/quantlab_env/tools.py` with string parameters and a docstring, and add
   it to `TOOLS`. `validate` checks calls against its signature, and `env.py`
   publishes everything in `TOOLS`. Teach `grader.build_evidence` which result
   fields count as evidence.
3. **Question.** In `src/quantlab_env/questions.py`, add a builder that computes
   the gold from `data.facts_as_of` (the same view the tools serve), the prompt
   ending in the `ANSWER:` line, and a generator for the default set.
4. **Grader.** In `src/quantlab_env/grader.py`, the required-inputs check and
   `PAR_CALLS` are specific to net margin. Generalise them (for example, the
   question exposes its required inputs and par call count) rather than
   branching on type.
5. **Template.** Add an `@env.template` in `env.py` that builds the question,
   resets `EPISODE`, yields the prompt and grades, like `net_margin`.
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
