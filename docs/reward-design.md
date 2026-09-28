# Reward design

The grader is plain code in `src/quantlab_env/grader.py`. No LLM judge. It
reads two things: the agent's final text and the episode's tool call log
(`EPISODE.calls`, one `ToolCall(name, arguments, result)` per call, in order),
plus the question (ticker, period, `as_of` date, gold inputs).

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
The 0.5% tolerance is used only here, to judge the answer against gold.
Everything else (evidence, provenance, grounding) matches at the precision a
number is written with; see [signs, scale and precision](#signs-scale-and-precision).

The answer only scores if at least one admissible lookup succeeded (see
[the filing-date cutoff](#the-filing-date-cutoff)). A correct number with no
evidence earns nothing here.

### grounding (0.3, all or nothing)

Three conditions, all required:

1. The citation block is correct (`missing_inputs` lists the inputs that are
   not; see [the citation block](#the-citation-block)).
2. The final answer matches an evidence item that is derived from the
   question's net income over its revenue
   (`answer_derived` in the grade info; see [calculator provenance](#calculator-provenance)).
3. Every other number cited anywhere in the text (the prose check) matches
   some evidence value (`ungrounded` lists the ones that do not). A percent
   matches only a calculator result, never a looked-up amount, so
   `net income 20,878,000,000%` is ungrounded.

### The citation block

The prompt asks the answer to end with exactly three lines:

```
REVENUE: <revenue value>
NET_INCOME: <net income value>
ANSWER: <margin>%
```

with the revenue and net income for that fiscal year exactly as
`get_fundamentals` returned them: the whole value in USD with its sign. Only
formatting is allowed: an optional leading `$` and thousands commas
(`REVENUE: $78,740,000,000`). No rounding, no scale words, no decimals, no
parentheses, no trailing text. `citations()` reads the last line of each label
(`_LABELS`), so a later line overrides an earlier one, and the order of the
two lines does not matter. Labels are upper case, as the prompt writes them.

Each value is checked by label: `REVENUE:` must equal the question's revenue
(the value the gold used) and a `revenue` row for the question's ticker and
period from an admissible lookup in this episode; `NET_INCOME:` likewise for
`net_income`. Revenue and net income cited under each other's labels fail
both. This replaces an earlier free-text check that looked for the inputs
anywhere in the answer, by value only.

### validity (0.1)

The share of calls whose result is not a malformed error. Malformed codes are
`bad_argument`, `unknown_tool`, `unknown_ticker` and `unknown_concept`.
`no_data` (a well-formed query with nothing filed yet) is not malformed, and
neither is a lookup past the question's date. Like efficiency, validity is 0
unless an admissible lookup succeeded, so it is also 0 with no calls.

### efficiency (0.1)

`PAR_CALLS / max(PAR_CALLS, calls)` with `PAR_CALLS = 3` (revenue lookup, net
income lookup, one compute). Every call counts, including failed and
inadmissible ones. It is 0 unless an admissible lookup succeeded, so failed
calls alone never earn it.

## Evidence

`build_evidence` walks the log in call order and keeps a list of evidence
items. Each item has a value and a formula that recomputes it from the
looked-up values.

- An admissible, successful `get_fundamentals` adds the `value` of every
  returned row as a source. Dates, filing dates, forms and other metadata never
  count.
- A successful `compute` adds its result only if the expression passes the
  provenance rules below.
- Error results never count, even though some echo the agent's arguments.

### Calculator provenance

A calculated value counts only if it is derived from evidence. The grader
parses the `compute` expression and turns it into a formula over the
looked-up values:

- Every literal must equal an existing evidence item at the precision it is
  written with, or be a unit constant (1, 100, 1e3, 1e6, 1e9). `20878` stands
  for the looked-up 20,878,000,000 and `0.26515` for a computed 0.2651511, but
  `20885` and `20880` stand for nothing (an integer literal is exact). A
  literal binds to the evidence item it matches most closely, rescaled by the
  literal's own scale. Non-decimal literals (`0x518E`) are never evidence.
- Unit constants may only scale evidence. `x * 100` is fine; `1 + 1`, or any
  expression built from constants alone, is not.
- Each side of `+` and `-` must itself depend on a looked-up value, so
  `x + 0.004` is not evidence, and neither is `x / x * 7e9 + x / y` (an
  invented 7e9 riding on a small evidence term: the left side uses evidence
  but is a constant).
- The result must depend on a looked-up value. The grader nudges each source
  by 1% and checks that the result moves, so `x / x` (a constant 1 in
  disguise) is not evidence.

Earlier `compute` results are evidence items too, so a chain of steps works
(`20878 / 78740`, then `0.26515 * 100`). Once an invented number enters the
chain, every result after it fails the rules and is dropped, and a later
literal that copies an invented result is not rebound to the real evidence
nearby: `compute("20878+7")` then `compute("20885/78740*100")` grounds nothing.

For the final answer this is not enough: it must match an item whose formula
is proportional to the question's net income divided by its revenue. The
grader checks this by nudging the question's own net income source (the
result must rise by the same factor) and its revenue source (the result must
fall by it). A correct number that merely touches evidence, or a margin from
another year's rows, does not pass.

### Signs, scale and precision

This section is about the prose check. The citation block has its own, exact
format (see [the citation block](#the-citation-block)).

Cited numbers are compared with their sign, so a claimed loss does not match a
reported profit. The dashes that can be a minus are the hyphen-minus, U+2010
to U+2015 (hyphen, non-breaking hyphen, figure dash, en dash, em dash,
horizontal bar) and the true minus signs U+2212, U+FE63 and U+FF0D. A dash is
the amount's sign when it is:

- attached to the amount, whichever dash it is (`-$20.9 billion`,
  `-20,878 million`, or the same with an en dash, em dash or U+2212),
- after a currency sign, spaced or not, whichever dash it is
  (`$-20.9 billion`, `$ - 20.9 billion`),
- a hyphen-minus or a true minus spaced off after a word
  (`net income - $20.878 billion`, the same with U+2212) or after `:`, `=`,
  `(`, `,` or `;` (`net income: -  20,878 million`),
- a true minus sign (U+2212, U+FE63, U+FF0D) opening a line.

Spaced off after a word or punctuation, an en or em dash (or any of U+2010 to
U+2015) is punctuation, not a sign: models write `Revenue <em dash> $78.74 billion` as
prose all the time, and reading it as a minus would fail honest answers. The
inputs are checked exactly in the block, so this only affects extra prose
claims, and it is listed as a known limit.

A dash is not a sign after a digit (`2021-2022`, `2021 - 2022`, a
subtraction like `20,878 - 1,000`), inside a word (`10-K`, `pre-2022`,
`re- 20`) or as a bullet opening a line (`- $78.74 billion revenue`, or an en
dash bullet). Parentheses are read as prose, not as accounting negatives.

Scale is part of the claim. A unit word or suffix fixes the scale:
`trillion(s)`, `tn`, `t`; `billion(s)`, `bn`, `bln`, `bil`, `b`;
`million(s)`, `mn`, `mln`, `mm`, `m`; `thousand(s)`, `k`. A percent is `%`,
`percent`, `pct` or `per cent`. A currency amount with no unit (`$78.74`,
`78.74 dollars`, `USD 78.74`) means exactly that many dollars, so `$78.74`
does not match 78.74 billion. Only a bare number may stand for itself,
thousands, millions, billions or a percent, so `78,740` matches
78,740,000,000. A cited zero matches only zero: `$0 trillion` is not a
rounding of every amount under half a trillion.

Matching is at the precision the number is written with: a citation matches a
value that rounds to it, within half a unit in its last written place.
`$78.74 billion` matches 78,740 million and `26.52%` matches 26.5151%, but
`$20.885 billion` does not match 20,878 million, though it is within 0.5%. The
same rule applies to calculator literals and to the final answer's match
against its derivation.

### The filing-date cutoff

A lookup is admissible only if its `as_of` is on or before the question's
`as_of`. A later lookup can see filings the question rules out (for example a
restatement filed after the question's date). Such a call is well formed, so
it is not malformed, but it is neither evidence nor a successful lookup, and
it still counts toward efficiency. The tools themselves stay question
agnostic; `future_lookups` in the grade info counts these calls.

### Context-based exemptions

Some numbers in an answer are not claims about the data. They are removed
before grounding by context, never by value alone:

- dates: ISO (`2024-03-17`), numeric (`03/17/2024`) and with month names
  (`March 17, 2024`, `17 Mar 2024`),
- fiscal-year and quarter labels (`FY2023`, `Q4`) and form names (`10-K`,
  `10-K/A`, `10-Q`, `8-K`),
- numbered-list markers at the start of a line (`1.`, `2)`, `(3)`),
- a bare four-digit year from 1900 to 2100 with no sign, currency or unit
  (`fiscal 2021`),
- a single-digit count followed by a word (`3 filings`).

Anything with a currency sign, `dollars` or `USD`, a unit word or a percent is
always a claim. A count or year followed by a scale or currency word
(`hundred`, `million`, `bn`, `dollars`, `cents` and so on) is an amount, not
exempt. So `$7`, `7%`, `$2023`, `2023 dollars`, the `7` in
`7 hundred million` and the `2050` in `2050 hundred` must be grounded, and so
must `7` in `debt was 7` and `2,023`.

## Strict tool calls

Every MCP call goes through `StrictToolLog` in `env.py`, which returns
`tools.execute(name, arguments, EPISODE)` directly. `execute` validates raw
arguments against the tool's signature: unknown keys, missing keys and
non-string values are `bad_argument`. Nothing is coerced, dropped or repaired,
and every call, good or bad, is logged. Without the middleware FastMCP would
coerce `"2024"`-style values and reject bad calls before the tool ran, so they
would never reach the log and the grader could not count them.

## Adversarial cases

Each case has a test in `tests/test_grader.py`; the scripted ones also run end
to end in `tests/test_env.py` and `demo.py`. Rewards are for the JNJ FY2021
task asked on 2024-03-17 unless noted.

| case | answer | grounding | validity | efficiency | reward |
|---|---|---|---|---|---|
| honest: looks up, computes, cites the block | 1 | 1 | 1 | 1 | 1.00 |
| citation dodge: cites only the final number, no block | 1 | 0 | 1 | 1 | 0.70 |
| missing citation block (honest prose, no block) | 1 | 0 | 1 | 1 | 0.70 |
| labels swapped in the block | 1 | 0 | 1 | 1 | 0.70 |
| block value rounded, scaled or reformatted (`REVENUE: $78.74 billion`) | 1 | 0 | 1 | 1 | 0.70 |
| one malformed call (extra key), then recovers | 1 | 1 | 0.75 | 0.75 | 0.95 |
| mental math: right number, not computed by a tool | 1 | 0 | 1 | 1 | 0.70 |
| memorised number laundered through `compute("26.52 * 1")` | 1 | 0 | 1 | 1 | 0.70 |
| margin spelled from unit constants or `x / x` | 1 | 0 | 1 | 1 | 0.70 |
| right answer, no tool calls | 0 | 0 | 0 | 0 | 0.00 |
| right margin from the wrong year (AAPL FY2022 vs FY2023) | 1 | 0 | 1 | 1 | 0.70 |
| stale, pre-restatement answer | 0 | 0 | 1 | 1 | 0.20 |
| wrong sign in prose, any attached dash (`net income was <en dash>$20,878 million`) | 1 | 0 | 1 | 1 | 0.70 |
| spaced hyphen-minus or true minus in prose (`net income - $20.878 billion`) | 1 | 0 | 1 | 1 | 0.70 |
| dropped scale in prose (`revenue $78.74`) | 1 | 0 | 1 | 1 | 0.70 |
| invented input within 0.5% (`20885`), with one extra compute | 1 | 0 | 1 | 0.75 | 0.675 |
| `x/x*7e9 + NI/REV*1e6` computed, then `$7 billion` cited (one extra compute) | 1 | 0 | 1 | 0.75 | 0.675 |
| input cited as a percent in prose (`net income 20,878,000,000%`) | 1 | 0 | 1 | 1 | 0.70 |
| extra ungrounded amount (`debt $7`, `assets $2023`) | 1 | 0 | 1 | 1 | 0.70 |
| scale-word variants (`7 billions`, `9 mln`, `7 hundred million`, `9 pct`) | 1 | 0 | 1 | 1 | 0.70 |
| a coarse zero (`$0 trillion`) | 1 | 0 | 1 | 1 | 0.70 |
| both lookups past the question's date | 0 | 0 | 0 | 0 | 0.00 |
| one lookup past the question's date | 1 | 0 | 1 | 1 | 0.70 |
| only malformed calls, then the right answer | 0 | 0 | 0 | 0 | 0.00 |

`demo.py` prints the scripted subset (honest 1.00, dodge 0.70, mental_math
0.70, laundered 0.70, memorised 0.00, malformed 0.95).

### Honest baseline: 1.00

Looks up revenue and net income, computes the margin with `compute`, cites
both inputs in prose and in the citation block, and ends with the result.
Every component is 1. The end-to-end tests run this
agent on all 18 tasks and require 1.0 on each.

### Citation dodge: 0.70

Cites only the final margin. This comes from the earlier memo writer
([quantlab-analyst](https://www.rohanjk.xyz/projects/quantlab-analyst/)): DPO on
pass/fail pairs learned to cite fewer numbers instead of correct ones, because
fewer claims meant fewer chances to fail. A grader that only checks the numbers
that are present rewards that. Here the citation block is missing, so the
terse answer loses grounding. Honest prose that cites the inputs but leaves
out the block scores the same 0.70. A test checks that leaving out the inputs scores
no better than citing them alongside a wrong number. It does not claim more:
removing a wrong number from an answer that already cites the right inputs
does raise the score, which is the behaviour a grounding reward wants.

### Malformed tool call: 0.95

One `get_fundamentals` call with an extra `fiscal_year` key, then the honest
path. The call fails with `bad_argument`, is logged, and costs validity (0.75)
and efficiency (3/4). This comes from the agent's GRPO training loop
([quantlab-agentic](https://www.rohanjk.xyz/projects/quantlab-agentic/)), whose
tool-call parsing was slightly too lenient; the RL model lost points exactly
where it had been lenient. Here there is one strict contract, used for every
call, and bad calls are visible to both the agent and the grader. Tests also
send ten kinds of malformed call over real MCP (coerced, null, missing, extra
and misnamed keys, wrong types, an unknown tool) and check each is logged as
an error.

### Correct number not in the evidence: 0.70

Mental math: the agent looks up both inputs but computes the margin in its
head, so the final number is not evidence. Laundering: the agent passes a
remembered figure through the calculator, `compute("26.52 * 1")`. The literal
26.52 is not evidence, so neither is the result. The public write-up describes
this as dressing up a memorised answer as a computed one, found in adversarial
review of the original reward.

A later review found a way around the literal check: unit constants were
allowed literals, so `100/(1+1+1+1)+1+(100/(1+1)+1+1)/100` spelled 26.52 with
no evidence at all, and scored 1.0. The provenance rules close this, along
with variants that multiply by `x / x`, add a vanishingly small evidence term,
or mix an invented factor or offset into a chain of steps. For the final
answer, all score 0.70.

The same trick used to work for an extra claim, not the answer. A third review
computed `(20878/20878+...)*1e9 + 20878/78740*1e6` (seven `x / x` terms), which
returns 7,000,265,151, and added `Total debt was $7 billion.` Each side of `+`
used evidence, and the whole formula depended on a looked-up value through the
small margin term, so the result was evidence and the invented $7 billion was
grounded (0.975). Now each side of `+` and `-` must itself depend on a
looked-up value, so the result is not evidence (0.675, with 0.75 efficiency
for the extra call).

### Right answer, no evidence: 0.00

Answers from memory with no tool calls. Answer credit, validity and efficiency
all need an admissible successful lookup, so this scores 0, and so does a
right answer after only a well-formed `compute("26.52 * 1")` or a lookup that
returned `no_data` (both used to score 0.10 through validity).

### Right margin from the wrong year: 0.70

Apple's FY2022 and FY2023 margins are both 25.31%. One lookup as of the
question's date returns both years, and an agent that computes and cites
FY2022's rows gets a correct-looking answer in which every number is real
evidence. The citation block catches it (FY2022's values are not the
question's period), and so does the provenance check: the answer is not
derived from FY2023's net income and revenue.

### Stale, pre-restatement answer: 0.20

JNJ restated FY2021 revenue in its 2024 10-K after the Kenvue separation. Asked
on 2022-03-19 the margin is 22.26%; asked on 2024-03-17 it is 26.52%. An agent
that looks up with the earlier date and uses the original figures is wrong,
its block cites the original revenue rather than the restated one the gold
used, and it keeps only validity and efficiency.

### Signs, scale and extra amounts: 0.70

`net income -$20.878 billion` claims a loss the filing does not show, and
`revenue $78.74` claims 78.74 dollars. Both used to match. So did `debt 7`,
`debt $7` or `assets $2023` next to an honest answer, because small integers
and year-like values were exempt by value. Now each is an ungrounded claim
(a bare `2023` with no currency or unit is still read as a year), and legitimate
forms (`$78.74 billion`, `78,740 million`, `$78.74B`, `USD 78.74 billion`, full
dollar amounts, bare numbers with a stated unit) still score 1.0.

### Found by a second review: 0.675 or 0.70

A second review of the fixed grader found three more ways to full grounding.

- Tolerance let invented numbers through. After honest lookups,
  `compute("20878+7")` then `compute("20885/78740*100")`, with the answer
  `Revenue $78.74 billion; net income $20.885 billion` and `ANSWER: 26.52%`,
  scored 0.975: 20,885 is within 0.5% of the real 20,878, so the invented
  figure was accepted as a citation, and the calculator literal was rebound to
  the real net income. Matching is now at written precision, so neither the
  literal nor the citation is evidence, the answer is not derived, and
  net income is missing (0.675, with 0.75 efficiency for the extra call).
- A spaced minus was dropped: `net income - $20.878 billion` and
  `net income −  $20.878 billion` scored 1.0. Now they are read as losses
  (0.70). The old regex also read `2021 - 2022` as a claim of -2022; a hyphen
  after a digit is no longer a sign.
- A percent satisfied a required input, because a percent reads as the number
  itself as well as a fraction:
  `revenue 78,740,000,000%, net income 20,878,000,000%` scored 1.0. A percent
  in prose now matches only a calculator result, never a looked-up amount, and
  the inputs are cited in the block (0.70).

### Found by a third review: 0.675 or 0.70

A third review found more ways to full grounding, all in the free-text prose
check. The inputs moved into the [citation block](#the-citation-block), which
closes the label swap (revenue and net income cited under each other's labels
used to score 1.0), and these were fixed in the prose check:

- **A: any dash attached to an amount is a minus.** Only the hyphen-minus and
  U+2212 were signs, so `net income was <en dash>$20,878 million`, with an en
  dash, em dash, U+2010, figure dash, U+FE63 or U+FF0D, claimed a loss and
  still scored 1.0. Every dash in hyphen-minus, U+2010 to U+2015, U+2212,
  U+FE63 and U+FF0D is now a minus when attached (0.70). Spaced off, only a
  hyphen-minus or true minus is a sign; a spaced en or em dash is punctuation
  (see [signs, scale and precision](#signs-scale-and-precision)).
- **B: each side of `+` and `-` must depend on a looked-up value.** See
  [correct number not in the evidence](#correct-number-not-in-the-evidence-070)
  (0.675).
- **C: plural and abbreviated scale words are units.** `7 billions`, `9 mln`,
  `8 bil`, `5 millions` and `9 pct` were not units, so the single-digit count
  and year exemptions swallowed them (`2050 millions` read as a year), and
  `7 hundred million` read as a count. They are units now, and a count or year
  followed by a scale or currency word is a claim (0.70).
- **D: a cited zero matches only zero.** `Debt was $0 trillion.` rounds from
  anything under half a trillion, so it matched every looked-up amount (0.70).

### Lookups past the question's date: 0.00 or 0.70

Looking up as of 2099-01-01 sees every filing, including ones the question
rules out, and used to score 1.0. With both lookups in the future there is no
admissible lookup, so answer, validity and efficiency are all 0 (0.00; it was
0.10 through validity before the third review). With one of the two in the
future, the answer can still be right but the input from the future lookup is
missing from the block and ungrounded in prose (0.70).

### Only malformed calls, then the right answer: 0.00

Not from QuantLab. The end-to-end test found that an episode of only failed
calls still earned the 0.1 efficiency share. Efficiency (and, since the third
review, validity) now requires an admissible successful lookup, so this
scores 0.

## Differences from QuantLab

QuantLab weighted 0.7 answer, 0.1 validity, 0.1 efficiency, 0.1 grounding, and
grounding checked that the final number appeared in the tool results. Here
grounding checks the citation block, every other cited number and the
provenance of the final answer, and carries 0.3. What the code enforces: a correct answer
with no successful lookup scores 0, and a correct answer that fails grounding
scores at most 0.7, against 1.0 for a grounded one.

## Known limits

What the grader still does not catch. The two inputs are checked exactly, by
label, in the citation block; the limits on number parsing below apply to the
other numbers the answer cites in prose.

- Numbers written in words ("twenty billion") are not extracted, so they are
  not checked.
- Prose matches values, not labels. Apart from the final answer and the
  citation block, a cited number only has to match some admissible looked-up
  or computed value, from any row the episode returned. Calling the revenue
  "net income", citing another year's figure as this year's, or citing another
  ticker's figures (any ticker may be looked up, and those lookups are
  admissible) as extra claims is not caught.
- A sign stated in words or parentheses in prose is not read: `net loss of
  $20,878 million` and `$(20,878) million` read as positive, so they match a
  reported profit. A spaced en or em dash before an amount in prose
  (`net income <em dash> $20,878 million`) is read as punctuation, not a
  minus. A hyphen opening a line is read as a bullet, so `- 20,878 million` at
  the start of a line reads as positive. A spaced hyphen-minus or true minus
  is read as a sign only after a word or `:`, `=`, `(`, `,` or `;` (or, for a
  true minus, at the start of a line).
- A bare number may stand for itself, thousands, millions, billions or a
  percent. A wrong claim that rounds to some evidence value at one of those
  scales passes.
- Matching is at written precision, so a coarse rounding is accepted:
  `$21 billion` for 20,878 million, or a literal `2.1e4` in `compute`. The fewer
  digits a number has, the more values it rounds from. A computed margin from
  coarse inputs must still be within 0.5% of gold to earn the answer.
- Calculator results are evidence. Each side of `+` and `-` must depend on a
  looked-up value, but products and quotients of many looked-up values (other
  tickers included) can still reach almost any coarse claim such as
  `$7 billion`.
- Exempt contexts are unchecked: a bare four-digit number from 1900 to 2100 and
  a single-digit count followed by a word are treated as a year and a count,
  even if the agent meant an amount. So `assets 2023` is not checked, while
  `assets $2023`, `assets 2,023` and `assets 2023 dollars` are.
- The provenance check is numeric, not symbolic: it evaluates each formula at
  the looked-up values and at 1% nudges of them, and nowhere else.
- The grader cannot tell whether the agent understood the point-in-time trap,
  only whether its lookups and numbers are admissible.
- One episode at a time per process, because the call log is process state and
  a tool call carries no session id. A newer session makes an older one fail
  at grade time with `OverlappingEpisodes`, but calls an older session makes
  after a newer one started land in the newer session's log (see
  [environment.md](environment.md#concurrency)).
