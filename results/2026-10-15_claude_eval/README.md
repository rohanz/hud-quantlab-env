# 2026-10-15: claude-sonnet-4-6 through `hud eval`

The first run of a hosted model through the HUD CLI's own agent (`claude`,
model chosen by `--config model=claude-sonnet-4-6`), each rollout in a fresh
subprocess, telemetry off. Files: `legacy.json` and `legacy.log` (18 tasks,
`--max-steps 10`), `probe_v2.json` and `probe_v2.log` (100 tasks,
`--max-steps 16`), `probe_v2_rerun_429.json` (one task re-run after an
Anthropic concurrency 429 on the first pass; the env was not involved).
Per-step traces for every episode are in the local `~/.hud/spans` directory,
not in the repo.

Commands:

```bash
export HUD_TELEMETRY_ENABLED=false HUD_CLI_ANALYTICS_ENABLED=false ANTHROPIC_API_KEY=...
uv run hud eval tasks_legacy.py claude --config model=claude-sonnet-4-6 --all --max-steps 10 --yes --json
uv run hud eval tasks_probe_v2.py claude --config model=claude-sonnet-4-6 --all --max-steps 16 --yes --json
```

## Scores

| set | n | mean reward | notes |
|---|---|---|---|
| legacy (`tasks_legacy.py`) | 18 | 0.983 | 17 at 1.00, one at 0.70 |
| probe v2 (`tasks_probe_v2.py`) | 100 | 0.863 | as run; one episode lost to a 429 |
| probe v2, 429 task re-run spliced in | 100 | 0.871 | the re-run scored 0.96 |

Probe v2 by type (as run; the 429 task is in `peer_margin_change_rank`):

| type | n | mean | distribution |
|---|---|---|---|
| fy_net_margin | 4 | 1.000 | all 1.00 |
| latest_net_margin | 4 | 1.000 | all 1.00 |
| revenue_growth | 4 | 1.000 | all 1.00 |
| net_margin_gap | 4 | 0.896 | 3 at 0.97, 1 at 0.67 |
| net_margin_change | 4 | 0.810 | 2 at 0.96, 2 at 0.66 |
| peer_margin_rank | 20 | 0.955 | 17 at 1.00, 3 at 0.70 |
| peer_margin_change_rank | 20 | 0.877 | 16 at 0.96, 3 at 0.66, 1 at 0.20 (the 429) |
| margin_trend | 20 | 0.830 | 14 at 0.96, 1 at 0.95, 3 at 0.66, 2 at 0.16 |
| margin_vs_peers | 20 | 0.711 | 17 at 0.68, 1 at 0.67, 2 at 0.98 |

## What the deductions were

Every deduction below was checked against the grader's `info` in the trace.

- **Memorised number in prose (legacy, 0.70).** NVIDIA fiscal 2023: the
  answer was right and grounded, but the prose added "FY2022, when the margin
  was ~36%", a figure never looked up or computed. Grounding 0.
- **Mental math on the sums (margin_vs_peers, 17 of 20 at 0.68).** The model
  added the three peers' net incomes and revenues in prose
  (`... = $11,296,000,000`) and only then called `compute` on the totals. The
  totals never came from a tool, so grounding is 0 and one extra lookup costs
  efficiency. This is the documented mental-math case; the two 0.98 episodes
  did the sums inside `compute`.
- **Latest change instead of the smallest (margin_trend, 2 at 0.16).** The
  question asks for the smallest year-over-year change; the model reported the
  most recent one. Answer 0, grounding 0. The documented adversarial case.
- **Sign dropped in prose (net_margin_change, margin_trend, 0.66).** "a decline
  of about 2.72 percentage points" cites +2.72 while the computed change is
  -2.72; "−0.00 pp" cites a zero that matches nothing. Cited numbers keep
  their sign and a zero only matches zero.
- **Uncomputed growth aside (peer_margin_change_rank, 0.66).** "revenue
  growth of ~6.4%" was never computed.
- **Efficiency only (the 0.95 to 0.97 scores).** One lookup over par.

Nothing here is a grader bug: the model's mistakes are the ones the grader
was built to catch, and the 0.98 episodes show the honest path scores
cleanly on every type. The scores land in the intended 0.55 to 0.80 band only
for `margin_vs_peers`; the single-hop types are solved, as they were for the
9B base on v1.

## Regrade note (2026-10-16)

The claude-sonnet-5-5 and claude-opus-5-5 runs in `../2026-10-15_claude55_eval`
exposed two grader false positives (`3M` read as 3 million, `10-Ks` not
exempt). After the fix, this run regrades to 0.875 as run and 0.883 with
the 429 task re-run spliced in; four episodes move (one trend, two
peers-comparison, one peer rank), listed in
`../2026-10-15_claude55_eval/regrade_after_3M_10Ks_fix.json`. The legacy
set is unchanged at 0.983.
