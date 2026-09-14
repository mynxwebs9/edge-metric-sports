# Phase 5 — Market Benchmark and Edge Analysis

## Objective

Does the frozen independent football model built through Phase 4 contain useful
information relative to historical NFL betting markets? The independent models
(`elo_v2`, `ridge_margin_E_v1`, `ridge_margin_CORE_v1`, `logistic_win_E_v1`,
`lightgbm_F_v1`) remain exactly as frozen in Phase 4 - nothing here retrains them, and no
market data was used in their training, selection, or Phase 4 evaluation. This phase reads
the already-persisted Phase 4 prediction ledger and joins it against historical sportsbook
lines for the first time in this project.

## 1-2, 20 — The market data layer and its provenance

`src/nfl_predict/market/snapshot_store.py` builds a `market_snapshot` table entirely
separate from the Phase 2 football feature tables, from market fields
(`spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `home_spread_odds`,
`away_spread_odds`, `over_odds`, `under_odds`) that have sat in the raw `schedules`
snapshots, denylisted from the independent model, since Phase 1.

**Coverage**: all 16 seasons (2010-2025), 4,363 games, **zero nulls** in every market
column - verified directly against the raw Parquet files before any code was written.

**What this source actually is** (verified against nflverse's own published field
dictionary, `nflreadr::dictionary_schedules`, fetched and quoted directly rather than
guessed): `spread_line` — "a positive number means the home team was favored by that many
points, a negative number means the away team was favored." This is the OPPOSITE sign from
this project's `docs/MODEL_SPEC.md` cover-probability convention (negative = home favored);
`home_spread_traditional = -spread_line` is the one conversion point
(`odds_math.nflverse_spread_to_traditional_home_spread`).

**Snapshot timing**: exactly ONE row per game, no sportsbook attribution, no documented
open/mid/pregame/closing timing anywhere in nflverse's own schema. Every historical row is
labeled `snapshot_type="UNKNOWN"` - never `"CLOSING"` - per the brief's explicit instruction
not to guess. This is why Step 11 (CLV) cannot be computed on this data (see below).

Per-season provenance (source, retrieval id, retrieved-at timestamp, raw content hash,
missingness, sign-convention documentation) is written alongside each season's Parquet file
at `data/market/snapshots/source=nflverse_schedules/season=<year>/provenance.json`.

## 3 — No-vig implied probability

`src/nfl_predict/market/odds_math.py` implements American-odds <-> probability conversion
and two-sided no-vig normalization (`no_vig_two_way`). Worked example from the brief (Team A
-120, Team B +105): raw implied probabilities 54.55%/48.78% sum to 103.3% (the vig);
de-vigged, 52.80%/47.20%. `-110` break-even is 52.38% exactly, as expected. Every
probability-based analysis below (moneyline edge, benchmark) uses the de-vigged probability,
never the raw one.

## 4 — Model fair line

`src/nfl_predict/market/fair_line.py` derives a fair win probability and cover probability
from each candidate's own frozen walk-forward margin prediction (never recomputed here)
using a zero-mean Normal approximation with **`margin_residual_std = 13.059063938633658`**
- the Phase 3-frozen, Phase-4-holdout-validated (ratio 1.02, `docs/PHASE4_BACKTEST_REPORT.md`
Step 8) margin residual standard deviation. **No Elo-specific (or any other
candidate-specific) uncertainty estimate was ever established in Phase 3/4** - this is
stated explicitly rather than fabricating per-candidate precision. Nothing here was tuned
against 2024-2025 outcomes or market data.

## 5 — Primary market benchmark

The market, scored with the identical `nfl_predict.models.metrics` functions Phase 4 used,
on the identical 570 holdout games:

| Predictor | Target | n | MAE | RMSE | Bias | Corr | Brier | AUC |
|---|---|---|---|---|---|---|---|---|
| **Market (spread)** | margin | 570 | **9.687** | 12.485 | −0.643 | 0.496 | — | — |
| Elo (best independent) | margin | 570 | 10.100 | 13.010 | −0.075 | 0.416 | — | — |
| **Market (moneyline, no-vig)** | win | 569 | — | — | — | — | **0.206** | **0.739** |
| Elo (best independent) | win | 569 | — | — | — | — | 0.216 | 0.716 |
| **Market (total)** | total | 570 | **10.096** | 12.916 | −1.306 | **0.308** | — | — |
| Naive (best independent) | total | 570 | 10.561 | 13.516 | −0.458 | −0.027 | — | — |

**The market beats every independent candidate on every target** - margin, win probability,
and (notably) total points, where the market shows genuine correlation (0.31) that every
independent total model (naive, Ridge, LightGBM - see `docs/PHASE4_BACKTEST_REPORT.md`)
completely lacked (−0.03 to 0.07). Average bookmaker hold: **4.28%**, a plausible value for a
mostly-(-110)/(-110) market. This confirms the Phase 5 core principle directly: the market
is a strong predictor and our system should not be assumed to beat it.

## 6 — Model-market disagreement

`edge_points = predicted_home_margin - market_implied_home_margin` (both in home-margin
space, so the sign is unambiguous regardless of which team is favored - see
`src/nfl_predict/market/disagreement.py`). Worked example: model fair line BUF −5.5 (home
margin +5.5), market BUF −3.0 (home margin +3.0) → `edge_points = +2.5`, "the model favors
BUF [home] by an additional 2.5 points" - exactly the brief's own example. Buckets
(0-0.99/1-1.99/2-2.99/3-3.99/4+) were fixed before any betting result was computed.

## 7-8 — Against-the-spread evaluation (real price, then assumed -110, kept separate)

Every candidate bets whichever side its fair line disagrees with the market toward. Real
recorded prices exist for **100%** of bets (verified zero nulls) - the assumed-`-110`
column below is reported purely for reference, never blended into the real-price numbers:

| Model | n | W-L-P | Cover rate (95% CI) | Real-price ROI/bet (95% CI) | Avg. real price |
|---|---|---|---|---|---|
| elo_v2 | 570 | 284-281-5 | 50.27% [46.15%, 54.37%] | −3.78% [−12.03%, +3.76%] | −95.0 |
| ridge_margin_E_v1 | 570 | 287-278-5 | 50.80% [46.68%, 54.90%] | −2.95% [−10.30%, +5.00%] | −98.6 |
| ridge_margin_CORE_v1 | 570 | 293-272-5 | 51.86% [47.74%, 55.95%] | −0.89% [−8.44%, +6.85%] | −98.5 |
| lightgbm_F_v1 | 570 | 288-277-5 | 50.97% [46.86%, 55.07%] | −2.66% [−10.02%, +5.46%] | −98.7 |

Every candidate's cover rate is at or below the standard -110 break-even (52.38%); every
95% CI straddles zero ROI. **None of these results is statistically distinguishable from a
break-even (or losing) betting strategy** - consistent with the market benchmark above.

## 9 — Moneyline edge analysis

Predefined probability-edge buckets (<1%/1-2%/2-3%/3-5%/5%+), betting whichever side each
model rates above the market's no-vig probability:

| Model | n bets | Actual win rate | Mean model expected prob. | Calibration gap | ROI/bet (95% CI) |
|---|---|---|---|---|---|
| elo_v2 | 570 | 42.36% | 50.53% | **−8.17 pts** | −7.29% [−16.98%, +3.05%] |
| logistic_win_E_v1 | 569 | 44.11% | 54.30% | **−10.19 pts** | −8.55% [−18.27%, +1.56%] |
| lightgbm_F_v1 | 569 | 43.76% | 53.41% | **−9.65 pts** | −9.35% [−18.93%, +0.38%] |

Every model's actual win rate on its OWN selected "edge" bets is **8-10 percentage points
below what the model itself predicted** for those exact bets, and every ROI is negative with
a CI that does not clearly exclude zero (LightGBM's upper bound is +0.38%, essentially at the
line). Betting the side a model disagrees with the market about does not perform anywhere
near what that model's own confidence would suggest.

## 10 — Spread edge by bucket, split by season (Elo, the best margin candidate)

| Edge bucket | n (2024/2025) | Cover 2024 | Cover 2025 | ROI 2024 | ROI 2025 |
|---|---|---|---|---|---|
| 0.00-0.99 | 68/60 | 57.4% | 51.7% | +10.4% | −1.8% |
| 1.00-1.99 | 54/86 | 40.4% | 47.1% | −21.9% | −10.0% |
| 2.00-2.99 | 52/46 | 50.0% | 50.0% | −4.0% | −4.7% |
| 3.00-3.99 | 44/33 | 40.9% | 57.6% | −21.7% | +9.9% |
| 4.00+ | 67/60 | 59.7% | 46.7% | +14.9% | −11.0% |

**No bucket is positive in both 2024 AND 2025** on ROI - the brief's own bar for a
convincing edge. The smallest-disagreement bucket (0.00-0.99) comes closest: cover rate is
on the right side of 50% both years, though 2025's ROI is barely negative once real prices
are applied. The largest-disagreement bucket (4.00+) looks the best in the *combined*
number but is driven almost entirely by 2024 (+14.9%) while 2025 lost badly (−11.0%) -
exactly the "don't hide a losing season inside a profitable aggregate" trap the brief warns
about. **No bucket shows a stable, reproducible edge.**

## 11 — Closing line value

**Not computable from the historical data.** The `market_snapshot` layer has exactly one
`UNKNOWN`-timing row per game - there is no closing line to compare an entry against. CLV
math (`src/nfl_predict/market/clv.py`) is implemented and unit-tested against synthetic
multi-snapshot data (matching the brief's own worked example: entry −2.5, close −3.5 → +1.0
point of favorable CLV) and is ready for the moment Steps 21-22's live storage accumulates
real multiple snapshots per game.

## 12-13 — Incremental information beyond the market

Simple OLS regressions (`actual_margin ~ market_spread` vs.
`actual_margin ~ market_spread + model_margin`), fit on one holdout season and evaluated
out-of-sample on the other (both directions - the only genuine OOS split available without
touching development data or re-fitting against the same season being "validated"):

| Model | Fit 2024→Eval 2025 (Δ MAE) | Fit 2025→Eval 2024 (Δ MAE) |
|---|---|---|
| elo_v2 | −0.052 (hurts) | −0.068 (hurts) |
| ridge_margin_E_v1 | +0.006 | −0.024 |
| ridge_margin_CORE_v1 | −0.003 | −0.041 |
| lightgbm_F_v1 | **+0.022** | **+0.009** |

(Positive = adding the model improves on market-only OOS MAE.) **None of these improvements
is practically meaningful** - all are a small fraction of a point on an ~9.7-10.1-point-MAE
base. LightGBM is the only candidate positive in both directions, consistent with it having
the smallest (though still not significant, see Step 6/`docs/PHASE4_BACKTEST_REPORT.md`)
gap to Elo among the non-Elo candidates. Elo actually makes market-only prediction slightly
*worse* in both directions once combined linearly - not evidence Elo is bad (it's the best
standalone predictor, per Step 5), just evidence that its information mostly overlaps with
what the market already prices in.

## 14 — Totals

Per the brief: Phase 4 found the independent total-points model does not beat naive, so
**no totals betting strategy was built.** The market benchmark (Step 5) shows the market
itself has real predictive skill on totals (MAE 10.10, corr 0.31) that the independent
model lacks entirely - documented for context, not acted on.

**Totals conclusion: DEFERRED / INSUFFICIENT INDEPENDENT SIGNAL.**

## 15 — Temporal discipline

The frozen Phase 4 2024-2025 predictions were generated before any sportsbook information
was introduced (Phase 4 never touched market data - see
`docs/PHASE4_BACKTEST_REPORT.md`/Step 11's denylist test), so evaluating them against
market data now is legitimate. Nothing in this phase re-optimizes a decision threshold on
2024-2025 and reports the same seasons as "validation" of it: the edge buckets (Step 6) and
probability buckets (Step 9) are the brief's own predetermined boundaries, not fit to this
data; the incremental-information test (Steps 12-13) explicitly fits on one season and
evaluates on the other, never on the same rows.

## 16 — No threshold overfitting

Every bucket boundary used in this report (spread-edge buckets, probability-edge buckets)
is the Phase 5 brief's own predetermined set - none were adjusted after seeing results.

## 17 — Statistical uncertainty

Every cover rate is reported with a Wilson score interval; every ROI with a 2,000-resample
game-level bootstrap CI (`src/nfl_predict/market/ats.py`, mirroring
`nfl_predict.backtesting.bootstrap`'s approach). Every CI in this report straddles (or comes
very close to straddling) zero. Language used throughout: **"not statistically
distinguishable from break-even," "unstable across seasons," "insufficient sample"** - never
"profitable."

## 18 — By-season stability

Reported explicitly for every major finding (Step 10's table above; full combined/2024/2025
breakdowns for every model in `data/reports/phase5_results.json`). No aggregate number in
this report hides a losing season - see Step 10's explicit callout of the 4.00+ bucket.

## 19 — Representative case review

Selected via `src/nfl_predict/market/error_review.py` from Elo's largest-disagreement bets
(category A: model disagreed and was right; B: model disagreed and market was right; C:
model and market agreed and both missed badly). Full game lists are in
`data/reports/phase5_results.json`'s `error_review` key. These are qualitative,
football-level observations for the future LLM research layer (Phase 6) - nothing here
altered any frozen prediction.

## 20 — Provenance

Documented in Section 1-2 above and in each season's `provenance.json`. No historical
snapshot was ever overwritten - each season's file is a single immutable write per this
phase's ingestion run.

## 21 — Live odds provider interface

`src/nfl_predict/market/odds_provider.py` defines `OddsProvider` (an ABC:
`get_events()`/`get_markets()`/`get_snapshot()`), `TheOddsAPIProvider` (the chosen live
adapter - constructor gates on `NFL_ODDS_API_KEY`, raises `OddsProviderNotConfiguredError`
if unset, and its HTTP methods deliberately raise `NotImplementedError` rather than
fabricate a response since no live prediction pipeline exists yet to consume them), and
`FixtureOddsProvider` (an offline adapter reading local JSON fixtures, used by tests to
exercise the interface without network access or credentials).

## 22 — Live market storage

`src/nfl_predict/market/live_snapshot_store.py` implements append-only timestamped storage,
keyed by `(provider_event_id, bookmaker, fetched_at)` - a duplicate key raises
`DuplicateSnapshotError` rather than silently overwriting. Nothing in production calls this
yet (no live pipeline exists); it is exercised only by tests against synthetic snapshots,
per the brief's explicit "do not fabricate live data" instruction.

## Known limitations

- The historical market source has no sportsbook attribution and no documented snapshot
  timing - every historical row is `UNKNOWN`, not `CLOSING`, and CLV cannot be computed on
  it (Step 11).
- `fair_line.py`'s uncertainty estimate is not candidate-specific (Step 4) - the only
  Phase 3/4-validated margin residual_std applies to a Ridge/F_full spec, not Elo
  specifically; documented as an approximation, not fabricated precision.
- The incremental-information test's out-of-sample split (Steps 12-13) uses only the two
  holdout seasons (fit on one, eval on the other) - a 2-fold split at n≈285/season is a
  real but modest-power OOS check, not a large-sample guarantee.
- Sample sizes in every spread/moneyline edge bucket are small (33-343 games) - explicitly
  reflected in wide confidence intervals throughout, never overstated.

## Implications for a future decision engine

No edge found here clears the bar for building a betting decision engine yet: cover rates
hover at or below break-even, no probability-edge bucket shows a credible positive
calibration gap, and the one incremental-information result that's consistently positive
(LightGBM, both directions) is a fraction of a point of MAE - not remotely close to
"exploitable." If Phase 7's decision engine is built before a genuine edge is found, it
should default to `NO BET` far more often than not, consistent with
`docs/DECISION_ENGINE.md`'s "zero qualifying bets in a week is valid" principle.
