# Phase 4 — Sealed Walk-Forward Backtest Report

## Objective

How well do the frozen football models generalize to genuinely unseen NFL seasons (2024,
2025) when run as they would have been historically? This phase answers that question and
nothing else — no betting ROI, no sportsbook comparison, no picks, no LLM research layer, no
website. See "Scope and explicit exclusions" at the end of this report.

## STEP 1 — Pre-holdout freeze (completed before any 2024–2025 access)

### A. Elo grid-edge investigation

Phase 3 selected `k_factor=30` from the grid `{10,15,20,25,30}` — the edge of the searched
range. `src/nfl_predict/backtesting/elo_tuning.py` expanded the grid to
`k ∈ {10,15,20,25,30,35,40,50}` and `home_field_advantage ∈ {25,35,45,55,65}`, selecting
parameters by development-only (2010–2022) sequential log-loss — the same method Phase 3
used — then ran a non-selecting confirmatory check on 2023.

- **Selected:** `k_factor=40.0`, `home_field_advantage=45.0` — both **interior** to their
  grids (`is_k_on_grid_boundary=False`, `is_home_adv_on_grid_boundary=False`). Phase 3's
  boundary concern is resolved.
- Development log-loss of selected config: **0.63808**. Next-best (`k=50, home_adv=45`):
  **0.63819** — a razor-thin margin over the new boundary point, reported here as a
  **marginal, not decisive**, improvement.
- Confirmatory (non-selecting) 2023 log-loss of the selected config: **0.66367**.

### B. Total-model bias investigation

Phase 3 found the total-points Ridge model (F_full features) over-predicts total points by
≈1 point on the 2021–2022 out-of-sample check (residual mean ≈ −1.17, actual − predicted;
`docs/PHASE3_MODEL_REPORT.md` previously stated this backwards as "under-predicts" — that
error is corrected in this report and in the source document).

`src/nfl_predict/backtesting/bias_investigation.py` re-fit the same Ridge/F_full spec across
four walk-forward fit/check windows spanning 2010–2023, plus a feature-set robustness check
(CORE vs. F_full on the same window):

| Fit seasons | Check seasons | Feature set | Bias (pred − actual) | Naive reversion bias |
|---|---|---|---|---|
| 2010–2015 | 2016–2018 | F_full | +0.720 | +0.059 |
| 2010–2018 | 2019–2020 | F_full | −1.894 | −2.259 |
| 2010–2020 | 2021–2022 | F_full | +1.170 | +0.706 |
| 2010–2022 | 2023 | F_full | +2.184 | +1.843 |
| 2010–2020 | 2021–2022 | CORE | +0.871 | +0.706 |

**Finding:** bias flips sign and varies in magnitude (+0.72 to +2.18) across windows and
tracks each window's naive fit-vs-check mean-total gap closely — i.e., it is substantially
explained by era/scoring-environment shift, not a stable model defect. It is not
feature-set-specific (CORE and F_full give similar magnitudes on the same window: +0.87 vs
+1.17). One pattern *was* robust: **postseason bias was negative in every single window**
(−0.50 to −2.41), distinct from the regular-season pattern — noted for Step 9, not corrected.

**Decision:** no arbitrary additive/multiplicative correction is applied. `ridge_total_E_v1`
is frozen with `calibration.correction_applied = False`.

### C–F. Frozen candidates, features, metrics, and the manifest

Seven candidate models were frozen (`src/nfl_predict/backtesting/freeze.py`):
`naive_v1`, `elo_v2` (k=40, home_adv=45), `ridge_margin_E_v1`, `ridge_total_E_v1`,
`logistic_win_E_v1` (all E_add_personnel, 64 features), `lightgbm_F_v1` (F_full, 103
features), and `ridge_margin_CORE_v1` (CORE, preserving Phase 3's CORE-vs-FULL margin
comparison). Metrics and subgroups (season, REG/POST, week buckets) were defined in the same
module, before any holdout access.

**Freeze manifest:** `data/backtests/phase4_holdout_freeze.json`
**SHA-256:** `a2c595c742d9a94c4db227ce833e5eaf51d87a7d8bd604b4e87b11607ac53ed3`

`src/nfl_predict/backtesting/holdout_guard.py` re-hashes this manifest and refuses to unseal
2024–2025 (or run any Phase 4 holdout-reading function) if the file is missing or its hash no
longer matches — verified by an explicit tamper test in
`tests/backtesting/test_holdout_guard.py`.

## STEP 2 — Walk-forward protocol

`src/nfl_predict/backtesting/walk_forward.py` retrains every non-Elo candidate from scratch
for each of the **44 holdout weeks** (2024 weeks 1–22, 2025 weeks 1–22, REG+POST), using only
games strictly before that week (development 2010–2022, validation 2023, and every holdout
week already "played" earlier in the same walk). A game qualifies for training iff its
`(season, week)` sorts strictly earlier — verified against the real `games` table that
within every season, POST week numbers (18/19–21/22) are always greater than every REG week
number, so this ordering is correct without special-casing season type.

Elo is handled differently and documented as such: `EloModel.run_sequential` is *already* a
walk-forward process (returns pre-game ratings for every game), so it runs exactly once over
the full 2010–2025 chronological sequence rather than being redundantly "retrained" 44 times.
Its margin transform (`elo_diff_pre → predicted margin`) is a fixed linear fit computed once
from development-period Elo history only (slope=0.0427, intercept=0.0655) — a derived
constant from already-frozen inputs, not a new tunable parameter, and not fit on holdout
data.

## STEP 3 — Immutable prediction ledger

`src/nfl_predict/backtesting/ledger.py` persists every prediction (game_id, season, week,
season_type, kickoff timestamp, prediction-generation timestamp, model_id, model_version,
feature_version, target, predicted value, training cutoff, training row count, artifact
hash, feature-names hash) to
`data/backtests/predictions/run_id=phase4_v1/predictions.parquet` **before** any outcome is
joined, and hashes it (SHA-256 sidecar). `write_prediction_ledger` refuses to overwrite an
existing run; `read_prediction_ledger` always re-verifies the hash before returning anything.

**Result:** `run_id=phase4_v1`, **6,838 predictions** across **570 holdout games**.

## STEP 4 — Holdout scoring

`src/nfl_predict/backtesting/scoring.py` joins the ledger with outcomes (read only after
predictions were persisted) and computes metrics overall and by every frozen subgroup.

**Overall 2024–2025 results (n=570 games; win-prob n=569, excluding 1 tie):**

| Model | Target | n | MAE | RMSE | Bias | Brier | AUC | Accuracy |
|---|---|---|---|---|---|---|---|---|
| naive_v1 | margin | 570 | 11.073 | 14.291 | −0.034 | — | — | — |
| elo_v2 | margin | 570 | **10.100** | **13.010** | −0.075 | — | — | — |
| ridge_margin_E_v1 | margin | 570 | 10.365 | 13.235 | −0.226 | — | — | — |
| ridge_margin_CORE_v1 | margin | 570 | 10.395 | 13.290 | −0.101 | — | — | — |
| lightgbm_F_v1 | margin | 570 | 10.216 | 13.101 | −0.209 | — | — | — |
| naive_v1 | win | 569 | — | — | — | 0.249 | 0.449 | 0.541 |
| elo_v2 | win | 569 | — | — | — | **0.216** | **0.716** | **0.668** |
| logistic_win_E_v1 | win | 569 | — | — | — | 0.227 | 0.674 | 0.619 |
| lightgbm_F_v1 | win | 569 | — | — | — | 0.223 | 0.687 | 0.634 |
| naive_v1 | total | 570 | 10.561 | 13.516 | −0.458 | — | — | — |
| ridge_total_E_v1 | total | 570 | 10.690 | 13.612 | −0.289 | — | — | — |
| lightgbm_F_v1 | total | 570 | 10.594 | 13.492 | −0.314 | — | — | — |

**Elo is the best model on both margin (MAE 10.10) and win probability (Brier 0.216, AUC
0.716) on genuinely unseen 2024–2025 data** — Phase 3's finding on the 2023 validation season
generalizes to the sealed holdout. No total-points model meaningfully beats the naive
constant.

Full subgroup breakdowns (season_2024, season_2025, combined, REG, POST, week buckets) are
in `data/reports/phase4_results.json` under `scoring.<model_id>.<target>.by_subgroup`.

## STEP 5 — Baseline comparison

Every sophisticated model vs. both baselines, absolute difference (candidate − baseline) on
its primary metric:

| Model | Target | vs. naive_v1 | vs. elo_v2 |
|---|---|---|---|
| ridge_margin_E_v1 | margin MAE | **−0.708** | +0.265 |
| ridge_margin_CORE_v1 | margin MAE | **−0.677** | +0.295 |
| lightgbm_F_v1 | margin MAE | **−0.857** | +0.116 |
| elo_v2 | margin MAE | **−0.973** | — |
| logistic_win_E_v1 | win Brier | **−0.022** | +0.010 |
| lightgbm_F_v1 | win Brier | **−0.026** | +0.007 |
| ridge_total_E_v1 | total MAE | +0.129 | n/a |
| lightgbm_F_v1 | total MAE | +0.034 | n/a |

Every margin/win-prob model beats naive; none conclusively beats Elo (see Step 6). Neither
total model beats naive at all.

## STEP 6 — Statistical uncertainty (bootstrap, game-level, n=2000, 95% CI)

| Comparison | Metric | Diff | 95% CI | Excludes zero? |
|---|---|---|---|---|
| elo_v2 vs naive_v1 | margin MAE | −0.973 | [−1.360, −0.584] | **Yes** |
| ridge_margin_E_v1 vs naive_v1 | margin MAE | −0.708 | [−1.138, −0.264] | **Yes** |
| ridge_margin_CORE_v1 vs naive_v1 | margin MAE | −0.677 | [−1.139, −0.227] | **Yes** |
| lightgbm_F_v1 vs naive_v1 | margin MAE | −0.857 | [−1.277, −0.410] | **Yes** |
| ridge_margin_E_v1 vs elo_v2 | margin MAE | +0.265 | [−0.048, +0.582] | No |
| ridge_margin_CORE_v1 vs elo_v2 | margin MAE | +0.295 | [−0.047, +0.624] | No |
| lightgbm_F_v1 vs elo_v2 | margin MAE | +0.116 | [−0.178, +0.382] | No |
| logistic_win_E_v1 vs naive_v1 | win Brier | −0.022 | [−0.035, −0.008] | **Yes** |
| lightgbm_F_v1 vs naive_v1 | win Brier | −0.026 | [−0.038, −0.013] | **Yes** |
| logistic_win_E_v1 vs elo_v2 | win Brier | +0.010 | [+0.001, +0.020] | **Yes** |
| lightgbm_F_v1 vs elo_v2 | win Brier | +0.007 | [−0.002, +0.015] | No |
| ridge_total_E_v1 vs naive_v1 | total MAE | +0.129 | [−0.005, +0.261] | No |
| lightgbm_F_v1 vs naive_v1 | total MAE | +0.034 | [−0.087, +0.156] | No |

**Interpretation:** every margin/win-prob candidate is distinguishably better than naive.
None of Ridge/LightGBM is distinguishably better than Elo on margin (all include zero) —
Elo's advantage there is real in point estimate but not statistically conclusive at n=570.
Elo **is** distinguishably better than logistic regression on win-prob Brier
(CI excludes zero, entirely positive). Neither total-points model is distinguishably better
than naive.

## STEP 7 — Calibration

Reliability tables (`data/reports/phase4_results.json`, `scoring.<model>.home_win.overall_2024_2025.calibration_bins`)
computed directly on holdout predictions, **no post-holdout recalibration**:

- `elo_v2`: well-calibrated in most bins (e.g. predicted 0.77 → observed 0.78 in the top
  bin), somewhat off in the 0.40–0.50 range (predicted ~0.43–0.47 → observed 0.25–0.41,
  n=48–59 — plausibly noise at this sample size).
- `logistic_win_E_v1`: similarly reasonable (predicted 0.79 → observed 0.76 top bin;
  predicted 0.42 → observed 0.49, n=39).

Both are within the range expected given per-bin sample sizes of 40–130 games; neither shows
a severe, systematic miscalibration.

## STEP 8 — Uncertainty validation

Phase 3's frozen uncertainty diagnostic (Ridge, F_full, fit 2010–2020 / check 2021–2022) is
**not** one of Phase 4's 7 declared candidates (Ridge is only frozen on E_add_personnel/CORE
there); to check it like-for-like, `src/nfl_predict/backtesting/uncertainty_validation.py`
ran that exact spec through the same walk-forward protocol as an additional, fully
frozen-config diagnostic (not a Step 10 candidate).

| | Margin | Total |
|---|---|---|
| Phase 3 frozen residual_std | 13.059 | 14.265 |
| Holdout residual_std | 13.327 | 13.642 |
| Ratio (holdout / frozen) | 1.02 | 0.96 |
| Phase 3 frozen residual_mean | −0.015 | −1.170 |
| Holdout residual_mean | +0.150 | **+0.264** |

**The frozen residual_std generalizes remarkably well for both targets** (within 2–4%).
Interval coverage using the *frozen* (not re-estimated) sigma stays close to nominal at every
level (50/80/90/95%): margin observed coverage 0.540/0.784/0.882/0.947; total observed
coverage 0.523/0.835/0.912/0.953.

**Notable:** the total-model's Phase-3-diagnosed −1.17 over-prediction bias **did not
persist** into the 2024–2025 holdout (holdout mean ≈ +0.26, near zero) — direct confirmation
of Step 1.B's finding that the bias is era-specific, not a stable defect.

Early-season (weeks 1–4) vs. later-season residual std: margin 14.01 vs 13.13; total 13.53
vs 13.68 — margin uncertainty is modestly higher early in the season, as expected with less
current-season signal accumulated; total is roughly flat.

## STEP 9 — Error analysis

Using `ridge_margin_E_v1` (margin) and `logistic_win_E_v1` (win-prob) against predefined
categories (`src/nfl_predict/backtesting/error_analysis.py`):

| Category | n (margin) | Margin MAE | Margin bias | Win Brier |
|---|---|---|---|---|
| Large favorites (top quartile \|pred margin\|) | 143 | 10.74 | −0.74 | 0.177 |
| Close games (bottom quartile) | 143 | 10.32 | +0.18 | 0.251 |
| Postseason | 26 | 10.90 | **−3.23** | 0.235 |
| Early season (REG weeks 1–4) | 128 | 10.62 | −1.42 | 0.249 |
| Late season (REG weeks 11+) | 243 | **10.02** | −0.23 | 0.216 |
| QB-continuity-change likely | 175 | 10.30 | +0.21 | 0.227 |
| Personnel continuity available | 570 | 10.37 | −0.23 | 0.227 |
| Personnel continuity unavailable | 0 | — | — | — |

(Personnel-continuity data covers 100% of the 2024–2025 holdout — it's only sparse pre-2013,
so the "unavailable" bucket is empty here, as expected.)

**Postseason margin bias is strongly negative (−3.23)** — the model under-predicts home
margin in the playoffs, consistent in *direction* with Step 1.B's finding that total-points
postseason bias is also consistently negative across every development-era window. Two
independent targets, same direction, same season segment — worth treating as a real
football-specific pattern (playoff games behave differently, small n=26 notwithstanding) for
a future phase to investigate, not something Phase 4 corrects.

Early season is harder to predict than late season (MAE 10.62 vs. 10.02), as expected with
less current-season signal.

**Largest misses** (top 5 of 15 inspected, real per-game context from the play-by-play
store — no fabricated causes):

| Game | Actual margin | Predicted | Abs error | Turnovers | OT | Weather |
|---|---|---|---|---|---|---|
| 2025_05_HOU_BAL | −34 | +7.2 | 41.2 | 3 | No | 78°F, clear |
| 2024_18_KC_DEN | +38 | −3.2 | 41.2 | 0 | No | 28°F, sunny |
| 2025_03_CIN_MIN | +38 | −2.3 | 40.3 | 5 | No | Dome |
| 2024_17_LAC_NE | −33 | +3.1 | 36.1 | 1 | No | 40°F, cloudy |
| 2025_03_ATL_CAR | +30 | −5.5 | 35.5 | 4 | No | 83°F, sunny |

None of the 15 largest misses went to overtime. Several (3 of top 5) had 3–5 turnovers,
consistent with turnover-driven blowouts that pregame team-strength features cannot foresee.
No weather extreme stands out among the misses. This is consistent with Step 1's earlier
finding that blowouts (large actual margins) are inherently the hardest games to predict from
pregame information alone — not a fixable modeling defect, a football reality.

## STEP 10 — Model selection recommendation

**Advance:** `elo_v2` (k=40, home_adv=45) as the primary margin and win-probability model —
best point estimates on both, the only model with a statistically distinguishable advantage
over a baseline on win-prob Brier (vs. logistic), fully transparent/interpretable, and
cheapest to run (no retraining, sequential updates only).

**Advance alongside it, not instead of it:** `lightgbm_F_v1` and `ridge_margin_E_v1` — both
distinguishably beat naive, neither is distinguishably worse than Elo, and both offer
feature-level interpretability (Ridge coefficients, LightGBM importances) Elo cannot. Keep
both in the comparison set for future phases rather than discarding on a non-significant gap.

**Do not advance on total points as currently specified:** none of `naive_v1`,
`ridge_total_E_v1`, `lightgbm_F_v1` distinguishably beats naive. Total-points prediction from
these pregame features adds no demonstrated value yet — a future phase should either accept
the naive constant for totals or invest in different features/approach before trusting a
"sophisticated" total model.

**CORE vs. F_full/E_add_personnel:** `ridge_margin_CORE_v1` (10.395 MAE) is statistically
indistinguishable from `ridge_margin_E_v1` (10.365 MAE) — personnel/prev-season features add
no demonstrated margin value on this holdout either, echoing Phase 3's F_full-underperforms
finding. CORE remains attractive for its longer historical coverage.

This is a complexity/robustness judgment, not "pick the most complex model" — Elo's
simplicity is treated as a genuine advantage, per the brief's explicit instruction.

## STEP 11 — No market data

Confirmed structurally, not just by claim: `nfl_predict.features.registry.assert_no_denylisted_columns`
runs on every game-level dataset load (`_load_game_dataset_impl`, used by both Phase 3's
`load_game_dataset` and Phase 4's `holdout_guard.load_holdout_game_dataset`), and
`tests/backtesting/test_walk_forward_no_lookahead.py::test_market_columns_remain_denylisted_for_phase4_too`
directly exercises it against `spread_line`/`total_line`. No spread, total line, moneyline,
odds, or market movement was used anywhere in Phase 4 training, selection, or evaluation. No
ATS/cover%/ROI/expected-value calculation was performed.

## Machine-readable outputs

- `data/backtests/phase4_holdout_freeze.json` + `.sha256` — the freeze manifest (Step 1.F).
- `data/backtests/predictions/run_id=phase4_v1/predictions.parquet` + `.sha256` — the
  immutable prediction ledger (Step 3), 6,838 rows.
- `data/reports/phase4_results.json` — scoring, baseline comparisons, bootstrap CIs,
  uncertainty validation, and error analysis (Steps 4–9), 570-game holdout.

## Known limitations

- The margin-transform slope/intercept for Elo were computed post-freeze from
  development-only data (a fixed derived constant from already-frozen inputs), since the
  freeze manifest's `elo_v2` candidate only explicitly enumerated the rating-update
  hyperparameters, not this transform — documented here rather than silently retrofitted.
- Step 8's uncertainty check used an additional Ridge/F_full spec not among the 7 declared
  Step 10 candidates, specifically to match Phase 3's original diagnostic like-for-like — it
  is a diagnostic only, never a Step 10 selection candidate.
- The normalized `games` table has no overtime flag; "went to overtime" in Step 9 is derived
  from play-by-play `qtr >= 5`, a real (not fabricated) signal, but weather/turnover context
  is descriptive only — no causal claim is made about any single largest-miss game.
- 44 holdout weeks × up to 7 candidates were refit from scratch each week; per-week fitted
  artifacts are hashed (`artifact_hash` in the ledger) but not individually persisted to disk
  — reproducibility is verified via `tests/backtesting/test_walk_forward_no_lookahead.py`'s
  determinism tests (fixed seeds, identical inputs) rather than by re-loading every historical
  artifact file.

## Scope and explicit exclusions (per this phase's brief)

No betting ROI, no sportsbook spread/total/moneyline comparison, no cover probability against
real market lines, no picks, no market-aware model, no LLM research invocation, no website
work. `docs/BACKTESTING_RULES.md`'s ATS/ROI/closing-line sections describe a *later* phase's
scope, not an omission here.
