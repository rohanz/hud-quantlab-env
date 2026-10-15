# Decisions

Short, dated entries. Newest last. Add one whenever a design choice would not be
obvious from the code.

## 2026-09-28: Depend on `hud` 0.6.18, not `hud-python`

(The bound is narrowed below: `hud>=0.6.18,<0.7`.)

The HUD SDK is published on PyPI as `hud` (the GitHub repo is still
`hud-evals/hud-python`). `pyproject.toml` pins `hud>=0.6.18`, the version the
environment, tests and demo were built and checked against. The API used
(`Environment`, `@env.initialize`, `@env.template`, `Capability.mcp`,
`LocalRuntime`, `EvaluationResult`/`SubScore`) is from that release.

## 2026-09-28: One tool server per process

The FastMCP server is started once per process and reused across env
stop/start cycles. Restarting FastMCP's HTTP app in the same process after a
forced stop left the next MCP handshake hanging, and the server holds no
per-episode state (the task template resets the call log). The consequence is
one episode at a time per process, like the board in HUD's 2048 cookbook.
`LocalRuntime` serialises acquisitions and `hud eval` runs each rollout in its
own subprocess, so this holds in practice.

## 2026-09-28: Bypass FastMCP argument validation with a middleware

FastMCP validates arguments before the tool runs: it coerces `"2024"`-style
values and rejects bad calls outright, so those calls would never reach the
episode log and could not be graded. `StrictToolLog` answers every call from
`tools.execute`, which checks raw arguments strictly, logs the call, and returns
bad calls as `bad_argument` errors. One strict contract for the agent and the
grader is the point of the malformed-call case.

## 2026-09-28: Grounding is stricter than QuantLab's

QuantLab's grounding checked the final number against tool results and carried
0.1. Here grounding covers every cited number, requires the answer to cite the
gold's revenue and net income, and carries 0.3; answer credit also needs a
successful lookup. This closes the citation dodge and the wrong-year case, and
makes a correct answer without evidence score 0.

## 2026-09-28: Efficiency needs a successful lookup

The end-to-end test showed that an episode of only failed calls still earned
the 0.1 efficiency share. Efficiency is now 0 unless a `get_fundamentals` call
succeeded (commit 84a5f67).

## 2026-09-28: compute rejects non-finite results

`compute("1e308 * 10")` returned `inf`, which reached the agent over MCP as
`value: null` with no error. Non-finite results are now `bad_argument`, like
division by zero.

## 2026-09-28: Docker image not built yet

(Superseded below: the image now installs at build time and has been built locally.)

`Dockerfile.hud` exists for `hud deploy` but has not been built or run. Local
work uses `LocalRuntime` (tests, `demo.py`) and the HUD CLI against
`tasks.py`. Build and smoke-test the image before deploying.

## 2026-09-28: An adversarial review found reward hacks; fixed with tests

After the first version was done, a Codex review attacked the grader directly
and found answers that scored 1.0 without being grounded. Every exploit below
is now a regression test (the docstrings cite the review), and the entries
that follow record each fix. The adversarial table in
[reward-design.md](reward-design.md) has the current scores. The original
laundering case (`compute("26.52 * 1")`) was already caught; the review showed
that closing one spelling of an exploit is not the same as closing the family.

## 2026-09-28: Track calculator provenance; the answer must be net income over revenue

Unit constants (1, 100, 1e3, 1e6, 1e9) were allowed calculator literals, so
`100/(1+1+1+1)+1+(100/(1+1)+1+1)/100` spelled 26.52 with no evidence and a
memorised margin scored 1.0. Each evidence item now carries a formula over the
looked-up values. Constants may only scale evidence, both sides of `+` and `-`
must use evidence, a result must depend on a looked-up value (so `x / x` is
not a way to make 1), and the final answer must match an item proportional to
the question's net income over its revenue. Checking provenance, rather than
adding more rules about which literals are allowed, is what closes the whole
family: variants that cancel evidence out or add a tiny evidence term fail the
same check.

## 2026-09-28: Signs and currency scale are part of a claim

Claims were compared as magnitudes, so "net income -$20.878 billion" matched a
reported profit, and a bare `$78.74` could stand for 78.74 billion. Comparison
is now signed, a minus attached to an amount is its sign (a hyphen after a word
or digit is not), and a currency amount without a unit word means exactly that
many dollars. Only a bare number may stand for thousands, millions, billions or
a percent, because agents legitimately write "78,740 (USD millions)".
Parentheses are read as prose, not as accounting negatives, since they appear
far more often as asides.

## 2026-09-28: Lookups past the question's date are not evidence

A lookup as of 2099-01-01 saw filings the question rules out (including
restatements filed later) and still counted as evidence, scoring 1.0. Now only
lookups with `as_of` on or before the question's date are evidence or count
as a successful lookup. Such a call is well formed, so it is not malformed; it
is ignored and still costs efficiency. The cutoff lives in the grader, not the
tools, so the tools stay question agnostic; `future_lookups` in the grade info
reports these calls.

## 2026-09-28: Exempt numbers by context, not value

Small integers and year-like values were exempt by value alone, so "debt 7"
rode along with an honest answer at 1.0 (as did "$7" and "$2023" until the
currency fix above). Now a number is exempt
only by context: dates (ISO, numeric, month names), FY and quarter labels, form
names, numbered-list markers at the start of a line, a bare four-digit year
with no currency or unit, or a single-digit count followed by a word
("3 filings"). Anything with a currency sign, "dollars" or "USD", a unit word
or a percent is always a claim. A bare "2023" is still read as a year; the
remaining exemptions are listed as known limits.

## 2026-09-28: Bind a calculator literal to the closest evidence

Found by running the honest agent on every task, not by the review. NVIDIA's
FY2022 revenue (26,914m) is within the 0.5% tolerance of FY2023's (26,974m),
so the provenance check bound the FY2023 literal to the FY2022 row and the
honest answer scored 0.7. A literal now binds to the evidence item it matches
most closely. The end-to-end tests now require 1.0 for the honest agent on all
18 tasks, and demo.py gained a `laundered` agent (0.70).

## 2026-09-28: Narrow the claims in the docs

The README said answering from memory was "always" the worse strategy and that
dropping a bad citation could never raise the score. Neither was what the code
guarantees: removing a wrong number from an answer that cites its inputs does
raise the score (as it should), and the grader has blind spots. The docs now
state only what the code enforces and the tests check, and README.md and
reward-design.md list the known limits.

## 2026-09-28: Apply XBRL tag precedence per filing, not per period

`build_fixture.py` let the preferred revenue tag win a whole fiscal period, so
when a company switched tags, earlier filings that reported the period under a
fallback tag were dropped and point-in-time history was lost. Precedence now
applies per `(period_end, accn)`. Tests cover a tag transition, restatements
and duplicate facts on synthetic companyfacts, and check the checked-in
fixture is a fixed point of the builder. The current fixture still lacks three
NVIDIA revenue rows from the old rule; no default question can see them, and
the test pins them until the fixture is regenerated (which needs the network).
Hand-checked values in `tests/test_fixture.py` now pin the data independently
of the gold code.

## 2026-09-28: Ship the fixture as package data

`data/fixture.json` sat outside the package, so `data.py` assumed an editable
checkout and an installed wheel had no data. It moved to
`src/quantlab_env/fixture.json` and loads with `importlib.resources`.

## 2026-09-28: Bound hud below 0.7 and declare direct imports

`hud>=0.6.18` allowed a future 0.7, which may change the `Environment` and eval
API. The pin is now `hud>=0.6.18,<0.7`. `env.py` imports `fastmcp` and
`uvicorn`, and `build_fixture.py` imports `httpx`; they were only present
transitively through hud, so they are now declared with bounds.

## 2026-09-28: Docker image installs at build time, telemetry off

`Dockerfile.hud` synced an editable install at build time but started with
`uv run`, which rebuilt the project on every container start. It now installs
the locked dependencies in their own layer, then the project non-editable into
`.venv`, and runs that venv's `hud serve` directly, so starting the container
builds nothing. `HUD_TELEMETRY_ENABLED=false` is set in the image.
`.dockerignore` matches `.git` as a file too, because in a git worktree it is
one. The image builds locally as `hud-quantlab-env:local` and serves on port
8765; it has not been deployed to the HUD platform.

## 2026-09-28: A second review found three more holes; fixed with tests

A second, confirmation review of the fixed grader found three answers that
still scored high, each now a regression test in `tests/test_grader.py`:

- **Match at written precision, not within 0.5%.** An invented net income of
  20,885 is within 0.5% of the real 20,878, so `compute("20878+7")` then
  `compute("20885/78740*100")`, citing `net income $20.885 billion`, scored
  0.975: the literal was rebound to the real net income through the tolerance,
  and the citation matched it too. Cited numbers and calculator literals must
  now equal evidence at the precision they are written with (half a unit in
  the last written place), so a rounding such as `$78.74 billion` for 78,740
  million still matches but `20.885` never matches `20.878`, and an integer
  literal is exact. The 0.5% tolerance now only judges the final answer
  against gold. The exploit scores 0.675. Accepting any rounding means a coarse
  one (`$21 billion`) passes; it is listed as a known limit. Binding a literal
  to the closest item stays, for coarse literals that round from several.
- **A spaced minus is a sign.** `net income - $20.878 billion` and
  `net income −  $20.878 billion` kept full grounding. A minus spaced off after
  a word or after `:`, `=`, `(`, `,` or `;`, or after a currency sign, now signs
  the amount (0.70). The old regex also read any hyphen followed by spaces as a
  sign, so `2021 - 2022` became a claim of -2022; a hyphen after a digit, a
  word hyphen and a bullet opening a line are now not signs. The `ANSWER:`
  line is parsed whole, so `ANSWER: - 3.1%` keeps its sign.
- **A percent never cites an input.** A percent reads as the number itself as
  well as a fraction, so `net income 20,878,000,000%` satisfied the required
  net income at 1.0. Inputs must be cited as amounts (0.70). Bare numbers
  still count, because honest answers write `78,740, net income 20,878 (USD
  millions)` and `20,878 / 78,740`; that bare numbers can stand in for inputs
  at any scale is a known limit.

A correction to the history: the commit message of 40ca234 ("exempt years,
dates, counts and list markers by context, not value") says "assets 2023"
rode along at 1.0 and implies it was fixed. It was not. A bare year from 1900
to 2100 is still exempt by design, so `assets 2023` is not checked; only
`assets $2023`, `assets 2,023` and other forms with a currency sign, unit or
separator are claims. The known limits say so.

## 2026-09-28: A third review (Opus) found more holes; inputs move to a citation block

A third review, by Opus, of the grader at b715a0b found that the free-text
number parser kept leaking, and that a shared `hud serve` could silently
mis-grade. Each fix below has a regression test that cites the review.

- **Citation block.** The prompt now asks the answer to end with
  `REVENUE: <value>`, `NET_INCOME: <value>` and `ANSWER: <margin>%`, the
  inputs written exactly as `get_fundamentals` returned them (whole USD with
  sign; only a leading `$` and thousands commas allowed). Each label is checked
  against an admissible lookup of its own concept, ticker and period and
  against the value the gold used, and the last line of each label wins. This
  replaces the free-text check that the inputs appeared somewhere in the
  answer as amounts, which matched by value only, so revenue and net income
  cited under each other's labels scored 1.0 (case E). Parsing two exact lines
  removes most of the attack surface for the inputs; the prose check stays for
  every other number.
- **A: any attached dash is a minus.** Only `-` and U+2212 were signs, so an
  en dash, em dash, U+2010, figure dash, U+FE63 or U+FF0D before an amount
  claimed a loss and scored 1.0. Attached, or after a currency sign, every
  dash in hyphen-minus, U+2010 to U+2015, U+2212, U+FE63 and U+FF0D is a minus.
  Spaced off after a word or punctuation, only a hyphen-minus or a true minus
  (U+2212, U+FE63, U+FF0D) is. We decided a spaced en or em dash is
  punctuation: models write `Revenue <em dash> $78.74 billion` as prose all
  the time, reading it as a minus would fail honest answers, and the inputs
  are now checked exactly in the block, so the cost is limited to extra prose
  claims. It is listed as a known limit.
- **B: each side of `+` and `-` must depend on a looked-up value.** The rule
  only checked that each side used evidence, and `x / x` uses evidence but is
  a constant. `(x/x+...)*1e9 + NI/REV*1e6` returned 7,000,265,151 and depended
  on evidence through the small margin term, so an invented `$7 billion`
  scored 0.975. Now it scores 0.675. Products and quotients of many looked-up
  values can still reach a coarse claim; that is a known limit.
- **C: scale words.** `billions`, `millions`, `thousands`, `trillions`, `bln`,
  `bil`, `mln`, `mn`, `pct` and `per cent` are units now, and a count or year
  followed by a scale or currency word (`7 hundred million`, `2023 dollars`) is
  an amount, not exempt. Before, the count and year exemptions swallowed them
  (1.0; now 0.70).
- **D: a cited zero matches only zero.** `$0 trillion` rounded from anything
  under half a trillion.
- **Validity is gated like efficiency.** A right answer after only a
  well-formed `compute("26.52*1")`, or a lookup that returned `no_data`,
  scored 0.10 through validity, so the docs' "a correct answer with no
  successful lookup scores 0" was false. Validity is now 0 unless an
  admissible lookup succeeded. Both lookups past the question's date now
  score 0.00 (was 0.10).
- **A generation guard for the episode log.** One `hud serve` process hosts
  many sessions, and `EPISODE` is one per process because a tool call carries
  no session id. Starting session B before A was graded made A's honest
  episode score 0.0. Each episode now takes a generation with
  `EPISODE.begin()`, and an episode whose generation changed fails at grade
  time with `OverlappingEpisodes` (newest wins). We chose this over an
  open-episode lock that refuses new episodes while one is open, because an
  abandoned or parked session (for example `hud task start` never graded)
  would block every later episode in the process. The guard cannot catch an
  older session that keeps calling tools after a newer one started; those
  calls land in the newer log. `LocalRuntime`, `hud eval` and `DockerRuntime`
  are safe; several concurrent rollouts against one `hud serve` are not.
- **Docker hardening.** The image runs as the non-root user `app`, has a
  `HEALTHCHECK` that sends a JSON-RPC `hello` to the control port 8765, and
  `.dockerignore` keeps tests, docs, scripts, markdown, `demo.py` and `.room`
  out of the build context.
- **Tests.** `tests/test_env.py` no longer edits `sys.path`; pytest's
  `pythonpath` setting puts the repo root on it. 248 tests pass.

## 2026-09-28: A fourth (final Opus) review; symmetric session guard, dates and markdown labels

Found by the final Opus review, fixed with tests.

- **Symmetric session guard with a timeout.** The generation guard let the
  newest session win: a freeloader session that started after an honest one,
  made no tool calls and submitted the honest answer scored 1.00 on the older
  session's calls, while the honest session failed. `Episode` now tracks open
  episodes; when one begins while another is open, all open episodes, the new
  one included, are marked as overlapping and fail at grade time with
  `OverlappingEpisodes`. Episodes close on grade, on error and on HUD's
  `aclose`. So concurrent rollouts against one `hud serve` now fail loudly
  instead of mis-grading, and a `tcp://` runtime needs `--max-concurrent 1`
  (or one process per rollout) to get grades at all. This is the open-episode
  lock we rejected earlier, made safe by an expiry: an unclosed episode stops
  counting as open after `QUANTLAB_EPISODE_TIMEOUT` seconds (default 3600,
  `tools.DEFAULT_EPISODE_TIMEOUT`). Remaining gaps: calls still cannot be
  attributed to sessions; an abandoned session blocks overlapping episodes
  until it expires; after expiry, an abandoned session that returns after a
  newer one began is refused, and calls it makes meanwhile land in the newer
  log.
- **Dates no longer exempt amounts.** `$9 may` read as the date "9 May" and
  `June 30 billion` as "June 30", so both invented amounts were exempt (1.0).
  A number right after a currency sign (`$9 may`, `USD 9 may`) and a date form
  followed by a unit or currency word (`June 30 billion`,
  `Losses may 9 billion`, `30 June 2023 billion`) are now amounts (0.70).
  Genuine dates (`June 30, 2023`, `30 June 2023`, `May 9, 2024`) stay exempt.
- **Markdown and case tolerant labels.** `**REVENUE:** 78,740,000,000` failed
  an honest model (0.70) and `**ANSWER:** 26.52%` found no answer. `REVENUE`,
  `NET_INCOME` and `ANSWER` are now case-insensitive and may follow markdown
  list, quote, heading and emphasis markers; a block value may be wrapped only
  in `*` or `_` emphasis. The trade-off: a later lowercase `Revenue: ...` line
  in prose now counts as a block line (the last line of each label wins), as
  `ANSWER` already did.
- **ASCII digits in the block.** Block values must be ASCII digits; fullwidth
  or Arabic-Indic digits land in `missing_inputs` (0.70).
- **Known limits added.** A foreign currency or other word after a single
  digit (`9 euros`, `3x`) is exempt as a count, superscript digits are not
  read, and number words ("seven billion") are not read.
- **Tests.** 292 tests pass (223 in `tests/test_grader.py`).

## 2026-10-05: First real-model eval; a bare 100 in a written formula is not a claim

Until now every score came from scripted agents. This was the first run with
real models: Qwen3.5-9B base and the QuantLab agentic-analyst SFT adapter
(trained on QuantLab's own tools and answer format, not this environment's),
served on a local RTX 4090 through QuantLab's transformers + PEFT server
(the path QuantLab trusts over vLLM for its adapters). Same server, same
sampling (temperature 0.9), all 18 tasks, 5 runs each, 180 episodes. The server's
parser was switched to keep unknown argument keys, so the strict validator
judges them, and tool properties were rendered alphabetically for both models,
as the adapter saw in training. A greedy differential check confirmed the
adapter was applied.

- **Grader false negative found.** Honest answers that computed the margin
  with the tool and then showed the working in prose
  (`0.2652 × 100 = 26.52%`, `(NI / REV) * 100%`, `multiply by 100`) scored
  0.70: the prose check read the `100` as an invented amount. This hit 34 of
  180 episodes (7 base, 27 SFT), so it penalised showing work. A bare `100`
  or `100%` right after a multiply or divide operator is now exempt as part
  of a formula, matching how `compute` already treats unit constants. Only
  100: larger unit constants after an operator (`× 1,000,000,000`) could pass
  for amounts and stay claims, as do `× 100 billion`, `× $100` and `× -100`.
  Replaying all 180 recorded episodes offline, the old grader reproduced
  every recorded reward exactly, and the new one changed only these 34.
- **Results after the fix** (mean reward, 95% bootstrap CI): base 0.982
  (0.968 to 0.993), SFT 0.968 (0.950 to 0.983); SFT minus base -0.014
  (-0.034 to +0.007, bootstrap over tasks), so no measurable difference.
  Before the fix the same episodes read base 0.959, SFT 0.877. Neither model
  gave a wrong answer in 180 episodes. The remaining losses are conventions
  carried over from training: two `compute` calls (fraction, then `× 100`;
  27 SFT and 13 base episodes, efficiency 0.75) and decimals on whole-dollar
  block values (`78740000000.0`, 7 SFT episodes). Both are scored as designed.
- **What it shows.** On these 18 questions base is near the ceiling, so the
  set cannot separate the models or give a training signal: GRPO needs
  questions some rollouts fail. Training on this environment needs more and
  harder questions with a held-out split first.
- **Known limit added.** LaTeX math delimiters read as dollar signs
  (`$$ 35.96\% $$`), once in 180 episodes.
- **Tests.** 305 tests pass (236 in `tests/test_grader.py`).

## 2026-10-06: A probe set, and a grader generalised over question types

The 2026-10-05 eval showed the 18 legacy tasks cannot separate models (base
0.982, SFT 0.968). The target for a held-out "unseen environment" probe is a
base mean reward of roughly 0.55 to 0.80, reward variance across rollouts on
at least 30% of tasks, at least 60 tasks over at least 10 companies, with the
task set and grader version hashed and frozen later. Difficulty must come
from the design of the questions, not from keeping tasks a model happened to
fail.

- **Two named sets.** `tasks.py` now holds two HUD `Taskset`s: `legacy`, the
  18 net-margin tasks, pinned in `questions.LEGACY` so they do not grow with
  the fixture (prompts, golds and slugs unchanged), and `probe`, derived from
  the fixture by `questions.probe_questions()`. HUD refuses a source module
  that exposes a task twice, so `tasks_legacy.py` and `tasks_probe.py` expose
  one set each for `--source`.
- **Probe question types**, each its own template: `fy_net_margin` (fiscal
  year named by label), `latest_net_margin` (the latest fiscal year with a
  10-K filed by the date, asked the day before each 10-K),
  `net_margin_change` (percentage points over the prior year),
  `revenue_growth` and `net_margin_gap` (one company minus another for the
  same fiscal-year label). Every restatement of a question's inputs is asked
  both before and after. 96 tasks with the current five companies: 23, 21,
  16, 17 and 19 of each type.
- **Fiscal-year labelling rule.** A fiscal year is labelled by the calendar
  year it ends in, except that a 52/53-week year ending in the first week of
  January belongs to the year before (`questions.fiscal_year`). This matches
  how NVIDIA, Apple, Microsoft, Coca-Cola and J&J label their years. The
  prompt says only that fiscal years are named as the company names them, as
  an analyst would know; the tools still return end dates, not labels.
  Retailers that label by the start year (fiscal 2023 ending in early 2024)
  break the rule and are kept out of the build list.
- **No mixed bases, no rounding-to-nothing.** A year-over-year question with
  a restated current year is asked only if the prior year comes from the same
  filing (JNJ's 2024 10-K restated fiscal 2021 but not 2020, so that pair is
  not asked as of 2024). Changes, growth and gaps under 0.1 percentage points
  are not asked (Apple's fiscal 2023 margin change was -0.004 pp).
- **Grader generalisation, not branching.** A `Question` now exposes its
  `inputs` (key, ticker, concept, period, value; the citation label is the key
  in upper case), a gold `formula` and `par_calls` (one lookup per ticker and
  concept, plus one compute). `citations`, `missing_inputs`,
  `derived_from_inputs` and efficiency read these. Derivation nudges each
  input in turn and requires the answer's formula to move as the gold formula
  does; for net margin that is the old check exactly. All existing tests and
  scores are unchanged, and the six demo scores are the same.
- **Two grading changes, each with tests.** (1) The answer is also correct
  within 0.005 percentage points of gold, the rounding of a two-decimal
  answer. Without it, MSFT's 0.234 pp margin change answered `0.23 pp` was
  wrong (1.8% off). For golds above 1%, including every legacy task, the 0.5%
  relative tolerance is already wider, so nothing changes there. (2) `compute`
  may subtract 1 from a ratio of evidence (`a / b - 1`), or 100 from that
  ratio times 100 (`a / b * 100 - 100`), the textbook growth formula in its
  two common forms, which the "constants only scale evidence" rule rejected.
  Rejecting the percent form would fail honest answers, a false negative of
  the same kind as the 2026-10-05 `× 100` one. Only those shapes:
  `x / y * 100 - 1`, `x / y - 100`, `x / y * 1e6 - 100`, `x / y * 100 + 100`,
  `x / y * 100 - 99`, `x / y + 1`, `1 - x / y`, `x / x - 1` and
  `x / x * 100 - 100` stay laundered (tested). Known limit: a prose number
  equal to a ratio of looked-up values minus 1 (or, in percent, minus 100) is
  now grounded.
- **More companies.** `scripts/build_fixture.py` lists 15 companies (adding
  ORCL, CSCO, NKE, FDX, COST, MU, WMT, DE, GE and MMM; mostly non-December
  fiscal years, and GE and MMM for spin-off restatements), periods ending
  2021 to 2025, and records each filing's fiscal-year focus (`fy`). A test
  checks every period's first 10-K `fy` against the labelling rule once the
  fixture has it. The fixture was not regenerated here (it needs the network
  and a contact User-Agent); the probe set grows when it is.
- **Not calibrated.** No model has been run on the probe set, so whether it
  lands in the target band is unknown. The honest scripted strategy
  (`demo.Honest`) scores 1.00 on every probe task end to end.
- **Tests.** 524 tests pass (353 in `tests/test_grader.py`, 129 in
  `tests/test_env.py`, 26 in `tests/test_tools.py`).

## 2026-10-06: Fixture rebuilt with 15 companies; Walmart revenue is total revenues

The fixture was regenerated from SEC EDGAR with the 15 companies the build
script lists, periods ending 2021 to 2025. The probe set grew from 96 to 350
tasks (84 `fy_net_margin`, 68 `latest_net_margin`, 58 `net_margin_change`,
65 `revenue_growth`, 75 `net_margin_gap`); the 18 legacy tasks and their golds
are unchanged.

- **Revenue checked company by company.** The latest year's revenue for each
  company was compared with its reported total. Fourteen matched. Walmart did
  not: the default first tag
  (`RevenueFromContractWithCustomerExcludingAssessedTax`) is its net sales
  ($674.5 billion for the year ending 2025-01-31), while its income statement
  total is total revenues ($681.0 billion, `Revenues`), which includes
  membership and other income. `TAG_OVERRIDES` gives Walmart its own tag
  order, recorded in the fixture as that company's `concepts` so the fixture
  stays a fixed point of the builder; a test pins it.
- **NVIDIA's missing revenue rows restored.** Per-filing precedence added the
  three filings the old build dropped, so `KNOWN_MISSING_REVENUE` is empty.
- **Probe size.** 350 tasks is more than the 60 to 100 the QuantLab plan asks
  for at 5 rollouts each. `questions.probe_sample` draws a seeded subset of 20
  per question type (seed 20261006), exposed as `tasks_probe100.py`. It was
  frozen before any model ran on the probe, so it cannot depend on model
  results; `tests/test_probe_sample.py` pins its prompts (sha256
  `f41b9cdbae8ff3c0246f2b1c426376cbf5463e3291dfe31ec6fb5590a07c69c8`).
- **Tests.** 1036 tests pass (607 in `tests/test_grader.py`, 383 in
  `tests/test_env.py`, 26 in `tests/test_tools.py`).

## 2026-10-06: Chained analyst questions and a frozen probe v2 subset

Measured on the frozen 100-task v1 subset (`tasks_probe100.py`), Qwen3.5-9B
base via vLLM, temperature 0.9, 5 rollouts: base 0.901 (target 0.55 to 0.80),
SFT 0.864. By type (base): `fy_net_margin` 0.977, `latest_net_margin` 0.979,
`revenue_growth` 0.904, `net_margin_change` 0.864, `net_margin_gap` 0.782.
Single-company one-hop questions are nearly solved, so they cannot separate
models; the only multi-company type was the hardest. The new types ask what a
working analyst or PM asks about several companies or several years at once,
needing 5 to 12 tool calls in which later steps depend on earlier results.

- **Four types kept**, each with its reason to exist:
  - `peer_margin_rank`: an analyst asks this when building a comp table and
    naming the best or worst operator in a peer group ("Which of Apple,
    Microsoft, Oracle and Cisco had the lowest net margin for its fiscal
    2021, using only filings available on 2021-11-28?", `CSCO 21.26%`). Three
    or four peers, highest or lowest in turn; par 9 or 12.
  - `peer_margin_change_rank`: an analyst asks this in earnings season, when
    the question is who expanded margins most ("Which of GE, 3M and FedEx
    expanded its net margin the most from fiscal 2021 to fiscal 2022?",
    `GE 9.08 pp`). Three peers (four would be 16 cited figures); par 9.
  - `margin_vs_peers`: an analyst asks this when judging whether a company
    earns a premium or discount margin to its sector ("By how many
    percentage points was FedEx's fiscal 2022 net margin above the combined
    net margin of Deere, GE and 3M?", `-3.95 pp`). Par 9.
  - `margin_trend`: an analyst asks this when checking a "margins expand
    every year" thesis, often just before a 10-K lands ("Did NVIDIA's net
    margin rise in each of its last three fiscal years reported in a 10-K
    filed on or before 2025-02-25?", `NO -20.04 pp`, the smallest change).
    Asked the day before each 10-K and 30 days after it; par 5.
- **Rejected.** Revenue CAGR over three or more years: the most common growth
  metric, but it needs a power, and `compute` has none on purpose. A power
  with a free exponent would let any target be manufactured from evidence
  (`(a / b) ** c` for a chosen `c`), breaking calculator provenance, and an
  arithmetic mean of annual growth rates is not what analysts mean by CAGR,
  so no tool was added. A simple average of peer margins (needs a literal
  count, which the calculator rules reject as a constant that is not
  evidence): the combined margin, total net income over total revenue, is
  how sector margins are aggregated, and the prompt defines it. "In how many
  of the last three years did the margin rise" (a count is guessable and has
  no formula to derive). Each of these is either artificial or would weaken
  the grader.
- **Difficulty by design**: peer groups are the fixture's sectors
  (`PEER_GROUPS`: large-cap technology, semiconductors and hardware,
  consumer, industrials), mostly with non-December fiscal years named by
  label, so one question mixes years ending in January to December (NVIDIA's
  and Walmart's fiscal years end in January of the year they are named for).
  Every restatement of any member is asked before and after (GE's spin-off
  restatements change the leader of Deere, GE and FedEx for fiscal 2023, from
  DE at 16.60% to GE at 26.82%). A trend needs the latest year filed by the
  date, then the three before it, on one basis. No task is kept or dropped by
  model results.
- **Scoring a ranking or a yes/no answer** (grader change, tested): a
  question may carry a `choice` and its `choices`; the ANSWER line must name
  the choice as a whole word, in any case, and no other candidate
  (`grader.chosen`). The gold is the picked figure itself (the chosen
  company's margin or change, or the smallest annual change), so the
  existing derivation check applies unchanged, and a calculation of another
  company's margin is not grounded. A pick is asked only if it stands clear
  of every other candidate by 0.1 percentage points and twice the 0.5%
  answer tolerance (`questions._pick`), so the number alone identifies it.
  Questions without a choice grade exactly as before (tested).
- **Efficiency par** gains `Question.computes`: one calculation per figure a
  ranking or trend has to compare (default 1, so every existing par is
  unchanged).
- **Sets.** `questions.chain_questions()` derives 294 tasks from the fixture
  (113 `peer_margin_rank`, 62 `peer_margin_change_rank`, 96
  `margin_vs_peers`, 23 `margin_trend`), bound as the `chains` set in
  `tasks.py`. `probe_questions()` is unchanged, so the v1 subset and its hash
  are too. The v2 subset (`tasks_probe_v2.py`, `questions.probe_sample_v2`,
  seed 20261006) has 100 tasks: 20 of each chained type and 4 of each v1 type
  for continuity, mean par 7.1 calls (8.1 over the chained types). Frozen
  before any model ran on it; `tests/test_probe_sample.py` pins its prompts
  (sha256 `9cde6035723745581a635ab06e572421de55f8de6054b73a208a15d37ad0503e`).
  `probe_sample` (v1) now calls the shared `stratified_sample` and draws the
  same subset (its hash test is unchanged).
- **`hud --source` fix.** `tasks_probe100.py` exposed its subset twice (as a
  `Taskset` and as `tasks`), so `hud task list --source tasks_probe100.py`
  failed with duplicate slugs. The `Taskset` is now private (`_probe100`), as
  in `tasks_probe.py`; the tasks are unchanged. `tasks_probe_v2.py` does the
  same.
- **Known limits.** A ranking agent that guesses the winner and computes only
  its margin (while citing every peer) is not penalised for skipping the
  comparison. A ticker that is also a word (`COST`, `DE`, `GE`, `KO`) on the
  ANSWER line counts as named. 22 of the 23 trend tasks are NO. Not
  calibrated: no model has run on v2 yet. At up to 12 calls, `hud eval` needs
  `--max-steps` above 10 for these tasks.
- **Tests.** 1654 tests pass (927 in `tests/test_grader.py`, 678 in
  `tests/test_env.py`, 27 in `tests/test_tools.py`, 5 in
  `tests/test_probe_sample.py`); `demo.py` is unchanged (1.00, 0.70, 0.70,
  0.70, 0.00, 0.95).

## 2026-10-11: GRPO split modules (tooling, grading unchanged)

- **What.** `grpo_splits.py` and `tasks_grpo_<split>.py` (train, val,
  test_in, test_heldout) pose the GRPO-from-base splits through `TEMPLATES`,
  as `tasks.py` does for the named sets, so training rollouts, sweeps and the
  final HUD tests use `Taskset.run` and `hud eval --source
  tasks_grpo_<split>.py` directly.
- **Where the rows live.** The split is defined, written and sha256-pinned in
  the training repo (quantlab-generality `scripts/hud_split.py`, seed
  20261009; 339 / 80 / 100 / 143 of the 662 tasks; the margin-change family
  held out). The modules read `<split>.jsonl` from the directory in
  `QUANTLAB_GRPO_SPLITS`, so this repo carries no copy that could drift. A row
  whose slug a template no longer reproduces fails loudly (the env changed
  since the split). Without the variable the modules raise `ImportError`, so
  a directory scan skips them.
- **Unchanged.** Templates, grader, fixture, the legacy set and both frozen
  probe subsets.
- **Tests.** 1657 pass (3 new in `tests/test_grpo_splits.py`).

## 2026-10-15: First hosted-model run through `hud eval` (claude-sonnet-4-6)

- **What.** The HUD CLI's own `claude` agent, model picked with
  `--config model=claude-sonnet-4-6` (naming the model directly needs the HUD
  model catalog and a `HUD_API_KEY`), one subprocess per rollout, telemetry
  off. Legacy set 0.983 (18 tasks, one at 0.70); probe v2 0.863 as run and
  0.871 with the one episode lost to an Anthropic concurrency 429 re-run
  (0.96). Files and the per-type table are in
  `results/2026-10-15_claude_eval/`.
- **What the grader caught.** A memorised prior-year margin in prose; peer
  totals added by hand before `compute` (17 of 20 `margin_vs_peers`, the
  documented mental-math case); the latest change reported where the smallest
  was asked (2 trend tasks, 0.16); a sign dropped in prose (`a decline of
  2.72 pp` for -2.72) and a cited `−0.00 pp`; an uncomputed `~6.4%` growth
  aside. Every deduction was checked against the trace's grader `info`; none
  is a grader bug, and the 0.98 episodes show the honest path scores on every
  type.
- **Calibration.** Only `margin_vs_peers` (0.711) sits in the 0.55 to 0.80
  band; single-hop types are solved (1.00), as they were for the 9B base on
  v1. The set stays frozen; the README's limits now say so.
- **Unchanged.** Templates, grader, fixture, tasksets, tests.
