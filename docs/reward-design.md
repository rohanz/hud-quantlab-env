# Reward design

The grader is plain code in `src/quantlab_env/grader.py`. No LLM judge. It
reads two things: the agent's final text and the episode's tool call log
(`EPISODE.calls`, one `ToolCall(name, arguments, result)` per call, in order).

```
reward = 0.5 * answer + 0.3 * grounding + 0.1 * validity + 0.1 * efficiency
```

## Components

### answer (0.5)

`final_answer` takes the last line that starts with `ANSWER:` and reads the
first number on it. It is correct if it is within `REL_TOL` (0.5%) of the gold
margin, read either way: `26.52%` or `26.52` as a percent, `0.2652` as a
fraction. The gold is `net_income / revenue` from `data.facts_as_of` on the
question's `as_of` date, which is exactly what the tools return on that date.

The answer only scores if at least one `get_fundamentals` call succeeded. A
correct number with no evidence earns nothing here.

### grounding (0.3, all or nothing)

Three conditions, all required:

1. The final answer matches evidence.
2. Every other number cited anywhere in the text matches evidence.
3. The text cites the revenue and the net income the gold was computed from
   (`missing_inputs` in the grade info).

Number extraction (`extract_numbers`) reads units: `$78.74 billion`,
`78,740 million`, `26.52%`. A plain number may stand for itself, for thousands,
millions or billions, or for a percent, so `78,740` matches `78,740,000,000`.
Comparison is on magnitudes, within 0.5%. These are not treated as claims: ISO
dates, `FY2023`, form names like `10-K`, bare years (1900 to 2100 without a
comma) and single-digit integers such as list markers.

### validity (0.1)

The share of calls whose result is not a malformed error. Malformed codes are
`bad_argument`, `unknown_tool`, `unknown_ticker` and `unknown_concept`.
`no_data` (a well-formed query with nothing filed yet) is not malformed. With no
calls, validity is 0.

### efficiency (0.1)

`PAR_CALLS / max(PAR_CALLS, calls)` with `PAR_CALLS = 3` (revenue lookup, net
income lookup, one compute). It is 0 unless a lookup succeeded, so failed calls
alone never earn it.

## Evidence

`build_evidence` walks the log in call order:

- A successful `get_fundamentals` adds the `value` of every returned row. Dates,
  filing dates, forms and other metadata never count.
- A successful `compute` adds its result only if every literal in the
  expression is already evidence, or a unit constant (1, 100, 1e3, 1e6, 1e9).
  Call this the transitive rule: arithmetic on evidence is evidence, arithmetic
  on anything else is not.
- Error results never count, even though some echo the agent's arguments.

## Strict tool calls

Every MCP call goes through `StrictToolLog` in `env.py`, which returns
`tools.execute(name, arguments, EPISODE)` directly. `execute` validates raw
arguments against the tool's signature: unknown keys, missing keys and
non-string values are `bad_argument`. Nothing is coerced, dropped or repaired,
and every call, good or bad, is logged. Without the middleware FastMCP would
coerce `"2024"`-style values and reject bad calls before the tool ran, so they
would never reach the log and the grader could not count them.

## Adversarial cases

Each case has a unit test in `tests/test_grader.py`; the scripted ones also run
end to end in `tests/test_env.py` and `demo.py`. Rewards are for the JNJ FY2021
task asked on 2024-03-17 unless noted.

### Honest baseline: 1.00

Looks up revenue and net income, computes the margin with `compute`, cites both
inputs and the result. Every component is 1.

### Citation dodge: 0.70

Cites only the final margin. This comes from the earlier memo writer
([quantlab-analyst](https://www.rohanjk.xyz/projects/quantlab-analyst/)): DPO on
pass/fail pairs learned to cite fewer numbers instead of correct ones, because
fewer claims meant fewer chances to fail. A grader that only checks the numbers
that are present rewards that. Here the required-inputs check fails, and
grounding is all or nothing, so citing less never scores higher than citing
correctly. A test also checks that dodging is no better than citing a wrong
number.

### Malformed tool call: 0.95

One `get_fundamentals` call with an extra `fiscal_year` key, then the honest
path. The call fails with `bad_argument`, is logged, and costs validity (0.75)
and efficiency (3/4). This comes from the agent's GRPO training loop
([quantlab-agentic](https://www.rohanjk.xyz/projects/quantlab-agentic/)), whose
tool-call parsing was slightly too lenient; the RL model lost points exactly
where it had been lenient. Here there is one strict contract, used for every
call, and bad calls are visible to both the agent and the grader. Tests also
cover numbers sent as JSON numbers instead of strings, over real MCP.

### Correct number not in the evidence: 0.70

Two variants. Mental math: the agent looks up both inputs but computes the
margin in its head, so the final number is not evidence. Laundering: the agent
passes a remembered figure through the calculator, `compute("26.52 * 1")`. The
literal 26.52 is not evidence, so by the transitive rule neither is the result.
The public write-up describes the second as dressing up a memorised answer as a
computed one, found in adversarial review of the original reward.

### Right answer, no evidence: 0.00

Answers from memory with no tool calls. Answer credit needs a successful lookup,
and validity and efficiency are 0 with no calls. On restatement questions
memory is also often out of date.

### Right margin from the wrong year: 0.70

Apple's FY2022 and FY2023 margins are both 25.31%. An agent that looks up and
cites the wrong year's figures gets a correct-looking answer. The
required-inputs check catches it because the cited revenue and net income are
not the ones the gold used.

### Stale, pre-restatement answer: 0.20

JNJ restated FY2021 revenue in its 2024 10-K after the Kenvue separation. Asked
on 2022-03-19 the margin is 22.26%; asked on 2024-03-17 it is 26.52%. An agent
that ignores `as_of` and uses the original figures is wrong and ungrounded, and
keeps only validity and efficiency.

### Only malformed calls, then the right answer: 0.00

Not from QuantLab. The end-to-end test found that an episode of only failed
calls still earned the 0.1 efficiency share. Efficiency now requires a
successful lookup, so this scores 0.

## Differences from QuantLab

QuantLab weighted 0.7 answer, 0.1 validity, 0.1 efficiency, 0.1 grounding, and
grounding checked that the final number appeared in the tool results. Here
grounding checks every cited number plus the required inputs and carries 0.3,
and a correct answer with no evidence scores 0. Answering from memory is
always the worse strategy.

## Known limits

- Number extraction is regex based; a claim written in words is not checked.
- One episode at a time per process, because the call log is process state.
