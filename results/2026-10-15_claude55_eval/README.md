# 2026-10-15: claude-sonnet-5-5 and claude-opus-5-5 through `hud eval`

Same settings as the claude-sonnet-4-6 run in `../2026-10-15_claude_eval`:
the CLI's `claude` agent with `--config model=<id>`, `tasks_probe_v2.py`,
`--all --max-steps 16`, one subprocess per rollout, telemetry off. The
legacy set was skipped (saturated at 0.983 on 4.6). No 429s this time; the
two models ran one after the other.

Files: `probe_v2_<model>.json` and `.log` (the run as graded on the day),
`regrade_after_3M_10Ks_fix.json` (every episode of all three models and the
4.6 legacy set regraded from its stored trace with the grader fixed below;
the only scores that moved are the ones listed there).

## What the traces showed first: two grader false positives

- The prompt names MMM as "3M Co.". The grader read `3M` in prose as 3
  million, an amount nothing looked up, and zeroed grounding. Ten episodes
  across the three runs lost 0.3 to this and nothing else.
- `10-K` was exempt as a form name but the plural `10-Ks` was not.

Both fixed in `grader.py` with tests (`test_a_company_named_by_a_digit_is_
not_an_amount` and a new case in the dates and form names test): the exact
name token of a company the fixture names by a digit is exempt; `$3M`,
`3 M` and `3m` stay amounts; form names take a plural. Every stored episode
was regraded from its trace; the regrade reproduces every original score
except the twelve the fix frees.

## Scores (probe v2, 100 tasks, after the fix)

| model | as run | regraded |
|---|---|---|
| claude-sonnet-4-6 (2026-10-15 folder, 429 episode re-run spliced) | 0.871 | 0.883 |
| claude-sonnet-5-5 | 0.807 | 0.816 |
| claude-opus-5-5 | 0.830 | 0.845 |

By type, regraded:

| type | n | sonnet-4-6 | sonnet-5-5 | opus-5-5 |
|---|---|---|---|---|
| fy_net_margin | 4 | 1.000 | 1.000 | 1.000 |
| latest_net_margin | 4 | 1.000 | 1.000 | 1.000 |
| revenue_growth | 4 | 1.000 | 1.000 | 0.925 |
| net_margin_gap | 4 | 0.896 | 0.700 | 0.843 |
| net_margin_change | 4 | 0.810 | 0.775 | 0.840 |
| peer_margin_rank | 20 | 0.970 | 0.970 | 1.000 |
| peer_margin_change_rank | 20 | 0.877 (incl. the 429 at 0.20) | 0.820 | 0.745 |
| margin_trend | 20 | 0.845 | 0.679 | 0.835 |
| margin_vs_peers | 20 | 0.741 | 0.715 | 0.725 |

## Why the newer models score lower

Not because they get questions wrong. Sonnet 5.5 and Opus 5.5 answered
every one of the 100 questions correctly (the 4.6 run got 2 wrong and lost
one to a 429). They lose on grounding, in 60 and 51 episodes respectively
against 26 for 4.6,
because they write more working in prose, and the grader only credits
numbers that a tool produced. Every deduction was classified from the
trace's grader `info` against the episode's evidence:

| cause (an episode may have several) | sonnet-5-5 | opus-5-5 | sonnet-4-6 |
|---|---|---|---|
| a per-year or per-company margin written in prose from mental arithmetic (`about 22.13%`), never passed through `compute` | 34 | 28 | 3 |
| sign dropped on a computed change (`fell by 2.72 points` for a computed -2.72; the grader keeps signs) | 22 | 38 | 8 |
| peer totals added by hand before `compute` (`$11,296,000,000`) | 18 | 21 | 14 |
| another number never computed (a pp change the model worked out itself, `0.0034 points`, `53-week`) | 20 | 3 | 4 |
| wrong answer (latest change reported where the smallest was asked) | 0 | 0 | 3 |
| efficiency only | 0 | 1 | 41 |

So the pattern the 4.6 run showed on `margin_vs_peers` (hand-added totals)
persists in both newer models, and two more join it: intermediate margins
stated in prose without a calculator call, and the sign of a fall written
as a positive "fell by" amount. Sonnet 5.5 called `compute` once per
question on the chained types (par is 5 to 12 calls, and it used 5 to 9),
doing the intermediate arithmetic in its head; Opus 5.5 computed more but
narrated the falls as unsigned numbers. One Sonnet 5.5 answer was written
in Chinese (graded normally: the block and ANSWER line were in the required
form).

## The fall-word rule (2026-10-16)

The sign rule is what stops "a loss of 20.9 billion" matching a positive
20.9 billion, but "fell by 2.72 points" is ordinary analyst English for a
computed -2.72, and sixteen Opus 5.5 episodes lost 0.3 to that alone. So a
narrow relaxation was made, with tests: a bare number or percent right
after a fall word may match a negative computed value; `fell to 11%` and
`down $5 billion` do not flip. Every stored episode was regraded again
(`regrade_after_fall_words.json`; the file keeps both regrades' deltas
against the as-run scores):

| model | as run | after `3M` and `10-Ks` | after the fall-word rule |
|---|---|---|---|
| claude-sonnet-4-6 (429 re-run spliced) | 0.871 | 0.883 | 0.883 |
| claude-sonnet-5-5 | 0.807 | 0.816 | 0.828 |
| claude-opus-5-5 | 0.830 | 0.845 | 0.890 |

By type after the rule, Opus 5.5: `margin_trend` 0.940 (was 0.835),
`peer_margin_change_rank` 0.865 (was 0.745); Sonnet 5.5:
`net_margin_change` 0.850, `peer_margin_change_rank` 0.865. The deductions
that remain are margins and totals worked out in prose without the
calculator, which is the behaviour the grader exists to catch.
