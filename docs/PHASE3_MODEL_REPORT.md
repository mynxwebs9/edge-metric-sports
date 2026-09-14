# Phase 3 Model Report

## Purpose

Phase 3's job was to determine whether the leakage-safe pregame features built in Phase 2
contain real predictive signal, and to establish a reproducible family of baseline models —
**not** to build a betting system. No ROI, ATS result, or sportsbook-line comparison
appears anywhere in this document, and none was calculated. That is explicitly Phase 4+
scope.

**2024–2025 remain sealed.** No model in this report was evaluated, tuned, or even loaded
against those seasons — enforced by an active runtime guard
(`src/nfl_predict/models/split.py`), not just a promise; see "Sealed holdout guard," below.

## Data split

| Segment | Seasons | Games (approx.) | Role |
|---|---|---|---|
| Development | 2010–2022 | 3,508 team-games' worth of rows (game-level) | Fit every model |
| Validation | 2023 | 285 | Phase 3's own held-out check — every metric below |
| **Sealed holdout** | **2024–2025** | **1,140** | **Untouched. Reserved for Phase 4.** |

No random shuffling anywhere — NFL games are temporal, and every split is by season.

### Sealed holdout guard

`src/nfl_predict/models/split.py` defines `SEALED_HOLDOUT_SEASONS = [2024, 2025]` and
`assert_seasons_not_sealed()`, which `load_game_dataset()` — the single choke point every
Phase 3 model goes through to read game-level data — calls on every invocation. Attempting
to load 2024 or 2025 raises `SealedHoldoutError` immediately. Tested directly
(`tests/models/test_split_guard.py`, 9 tests): rejects 2024 alone, 2025 alone, and a mixed
list containing either. The Phase 2 feature-correctness audits that touched 2024 data (see
`docs/PHASE2_FEATURE_REPORT.md`) never evaluated a predictive outcome, so they do not
compromise this seal, per the Phase 3 brief's own clarification.

## Target dataset

Versioned separately from feature storage: `data/targets/target_version=v1/season=<year>/
data.parquet`, built by `src/nfl_predict/models/targets.py` from the normalized `games`
table. Written for 2010–2023 only (development + validation) — **not** for 2024–2025,
consistent with the seal.

Targets: `home_score`, `away_score`, `home_margin` (= home − away), `total_points` (= home +
away), `home_win` (1/0), plus `is_tie` and metadata (`game_id`, `season`, `week`,
`season_type`, `kickoff_time_naive`, `game_date`, `home_team_id`, `away_team_id`).

**Ties handled explicitly, not coerced**: 13 real ties occurred in 2010–2025 (all regular
season — playoff rules prevent ties). For a tied game, `home_margin = 0` (correct),
`total_points` is well-defined, but `home_win = null` — neither a win nor a loss for either
team. Tied games are excluded from every win-probability model's training and evaluation
set (`without_ties()`), never coerced into 0 or 1. Verified: `tests/models/test_targets.py`.

**Never written back into Phase 2 feature Parquet** — verified directly:
`test_target_columns_are_never_present_in_the_feature_parquet` reads the actual stored
`data/features/game/feature_version=v1/season=2022/data.parquet` and asserts none of
`home_score`/`away_score`/`home_margin`/`total_points`/`home_win`/`is_tie` are present.

## Season type

`season_type` (`REG`/`POST`) is carried as evaluation metadata only — **not** added as a
predictive feature (no documented reason to override that default this phase). Every
validation metric below is reported for `REG` and `POST` separately, never pooled as if
they were the same population. Postseason games remain valid prior evidence inside Phase 2's
rolling/season-to-date windows (unchanged from Phase 2 — see
`docs/PHASE2_FEATURE_REPORT.md`'s note that windows aren't filtered by `season_type`).

## Model input source and market-field exclusion

Every model reads `data/features/game/feature_version=v1/` exclusively, joined to Phase 3
targets by `game_id` (`src/nfl_predict/models/feature_matrix.py::load_game_dataset`). Every
feature column used is `diff_<name>` (home − away) or, for the two symmetric boolean
context flags (`is_neutral_site`, `is_divisional_game`), `home_<name>` — and every `<name>`
is a real `config/features.yaml` entry, checked at import time
(`validate_ablation_features_are_registered()`, runs automatically on module import).

**No market field can enter an independent model.** `assert_no_denylisted_columns` (Phase
2's guard, reused here) runs on every load. Tested directly against the exact fields named
in the Phase 3 brief — `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`,
`spread_odds`, `total_odds`, `home_spread_odds`, `away_spread_odds`, `under_odds`,
`over_odds`, `closing_line`, `opening_line` — in
`tests/models/test_no_market_leakage.py` (3 tests), including a defensive proof that the
guard actually fires when handed a market field.

## Team identifiers are never numeric signals

A new guard, `assert_no_identifier_columns` (`feature_matrix.py`), rejects
`game_id`/`team_id`/`opponent_id`/`home_team_id`/`away_team_id` from ever being selected as
a model feature, and `select_features()` additionally rejects any non-numeric (string)
column outright — so an identifier could not silently become a feature even without the
name-based check. Verified: `tests/models/test_identifiers_not_numeric.py` (5 tests),
including confirming `team_id`/`game_id` really are stored as strings (e.g. `"0200"`), not
integers, in the source data. Feature-importance/coefficient review (below) additionally
confirms no identity-shaped field appears among the influential features of any fitted
model — the check is empirical as well as structural.

## Baseline 0: naive historical constant

Fit on development data only (2010–2022, 3,508 games):

| Constant | Value |
|---|---|
| Home margin mean | **+2.05** points |
| Total points mean | **45.64** |
| Home win rate (ties excluded) | **55.9%** |

Both figures are consistent with widely-known long-run NFL baselines (historical home-field
advantage of roughly 2–3 points; historical home win rate in the mid-to-upper 50s), which is
a reassuring sanity check on the underlying data, not a claim of precision.

## Baseline 1: Elo

Sequential, one game at a time, strictly chronological (`sort_games_chronologically`).
Requirements satisfied:

- **Ratings only update after a completed game**; a game's expected-outcome prediction is
  recorded using only its own *pre-game* ratings.
- **Future games never affect earlier ratings** — proven directly
  (`test_appending_a_future_unrelated_game_does_not_change_earlier_ratings`): appending a
  brand-new game after the fact leaves every earlier game's recorded pre-game rating and
  prediction byte-for-byte identical.
- **Home advantage is explicit**: a single additive rating-points constant, tested to
  actually move the predicted probability.
- **Postseason handling**: postseason games update ratings with the same K-factor as
  regular season — a documented simplification, not stakes-weighted.
- **Offseason carryover**: ratings regress 1/3 of the way back to the mean (1500) at every
  season boundary — a fixed, documented constant (not grid-searched), verified directly
  against the exact formula in `test_season_regression_reverts_ratings_toward_the_mean_at_season_boundary`.
- **Team identity**: uses Phase 1's stable `team_id` directly — no relocation/expansion
  mapping needed, since that was already solved upstream.

**Parameter selection** (K-factor, home-field advantage): a small, predetermined 5×5 grid
(`K ∈ {10,15,20,25,30}`, `home_adv ∈ {0,25,45,65,85}` rating points), scored by
sequential log-loss on **development data only** (Elo's own walk-forward predict-then-update
process makes this an honest in-sample-of-development search — never touches validation or
the sealed holdout). Selected: **K=30, home_field_advantage=45** (dev log-loss 0.6401; full
grid in `data/reports/phase3_elo_grid_search.csv`).

**Caveat, disclosed rather than hidden**: K=30 is the edge of the searched grid — a wider
grid might find an even better value. Not chased further this phase, consistent with "do
not perform aggressive hyperparameter optimization."

**Margin transform**: a one-feature OLS fit on development Elo history only —
`margin ≈ 0.0500 × elo_diff − 0.268` — a documented linear transform, not a second model and
not an invented points-per-Elo-point constant.

## Baseline 2/3/4: Ridge margin, Ridge total, logistic win probability

`src/nfl_predict/models/linear_models.py`. Each is a proper scikit-learn `Pipeline`
(`SimpleImputer(strategy="median", add_indicator=True)` → `StandardScaler` → `Ridge`/
`LogisticRegression`), fit **only** on the training partition passed to `.fit()` — the
Pipeline abstraction is what guarantees imputation statistics and scaling parameters never
see validation or holdout rows. Missingness indicators are preserved (not discarded), so
e.g. a personnel-continuity feature that's null pre-2013 is distinguishable from a
personnel-continuity feature that's genuinely near zero. `Ridge(alpha=5.0)`,
`LogisticRegression(C=1.0)` — moderate, fixed regularization, not tuned per feature set.

## Model 5: LightGBM (the one nonlinear comparison model)

**Chosen over XGBoost/HistGradientBoosting** because it's an existing project dependency,
handles missing values natively (no imputation step needed, which matters given Phase 2's
deliberate era-based nulls), and trains fast enough for a clean comparison sweep. Fixed,
predetermined hyperparameters (`n_estimators=150, learning_rate=0.03, max_depth=4,
num_leaves=15, min_child_samples=30`, L1/L2 regularization) — deliberately conservative to
resist overfitting ~3,500 training rows, **not grid-searched, not tuned against
validation**. Only the `F_full` feature set was used for this comparison model (margin,
total, and a `LGBMClassifier` for win probability), per the brief's "one restrained
nonlinear model."

## Feature-family ablations (A → F) and CORE vs. FULL

`src/nfl_predict/models/feature_matrix.py::ABLATIONS`, each a strict superset of the last:

| Tier | Adds | # features |
|---|---|---|
| A. Basic context | rest days, short week, post-bye, neutral site, divisional | 7 |
| B. + Offense/defense efficiency | season-to-date EPA/success rate, overall/pass/rush, both sides | 19 |
| C. + QB features | QB-specific EPA/dropback, success/sack/INT rate, CPOE, scramble EPA, consecutive starts | 26 |
| D. + Recent windows | 3g/5g/8g trailing variants of the 12 core efficiency metrics | 62 |
| E. + Personnel continuity | offense/defense snap continuity | 64 |
| F. Full Phase 2 set | explosive plays, pressure/sacks, turnovers, early-down, 3rd down, red zone, competitive-play variants, opponent quality, special teams, previous season | 103 |
| **CORE** (broad historical availability only) | F minus personnel (2013+) and previous-season (2011+) | 97 |

Every model (Ridge margin, Ridge total, logistic win) was fit and evaluated at every tier —
**7 fits × 3 targets = 21 linear model fits**, not a combinatorial feature-subset search.

### Ablation results (validation season 2023, n=285)

| Feature set | # feat | Margin MAE | Margin RMSE | Total MAE | Win Brier | Win LogLoss | Win AUC | Win Acc |
|---|---|---|---|---|---|---|---|---|
| A_basic_context | 7 | 11.08 | 14.41 | 10.97 | 0.2452 | 0.6835 | 0.558 | 0.554 |
| B_off_def_efficiency | 19 | 10.69 | 13.74 | 11.00 | 0.2395 | 0.6720 | 0.616 | 0.582 |
| C_add_qb | 26 | 10.81 | 13.85 | 10.99 | 0.2435 | 0.6810 | 0.607 | 0.589 |
| D_add_recent_windows | 62 | 10.73 | 13.88 | 11.00 | 0.2380 | 0.6687 | **0.622** | 0.618 |
| E_add_personnel | 64 | 10.72 | 13.88 | 11.03 | **0.2379** | **0.6684** | **0.624** | **0.618** |
| F_full | 103 | 10.78 | 13.90 | 11.24 | 0.2483 | 0.6955 | 0.599 | 0.596 |
| CORE | 97 | 10.81 | 13.93 | 11.15 | 0.2490 | 0.6960 | 0.588 | 0.572 |
| naive (floor) | 0 | 11.08 | 14.41 | 10.94 | 0.2458 | 0.6848 | 0.500 | 0.565 |
| Elo | — | **10.53** | **13.75** | — | 0.2333 | 0.6600 | 0.641 | 0.614 |
| LightGBM (F_full) | 103 | 10.86 | 13.92 | 10.97 | 0.2454 | 0.6847 | 0.598 | 0.579 |

Full precision in `data/reports/phase3_model_comparison.csv`.

### Ablation findings — reported honestly, not cherry-picked

- **A beats nothing meaningfully**: basic schedule context alone (rest/bye/divisional) is
  statistically indistinguishable from naive on margin (MAE 11.08 vs. 11.08) and barely
  above chance on win probability (AUC 0.558). This is the expected, correct result for a
  feature set containing no team-performance information — a useful sanity floor, not a
  disappointing model.
- **B (offense/defense efficiency) is where the real jump happens**: margin MAE improves
  from 11.08 → 10.69, win AUC from 0.558 → 0.616. Football performance data is where the
  signal actually is, not schedule context.
- **QB features (C) did not clearly help over B on this validation season** — margin MAE
  and win AUC both moved slightly *worse* than B (10.81 vs. 10.69; AUC 0.607 vs. 0.616).
  This does not prove QB context is worthless; it's one validation season's evidence, and
  QB features may matter more in games with a genuine QB change (which are a minority of
  285 games) than in aggregate. Documented as an open question for Phase 4's larger sample,
  not resolved here.
- **Recent 3g/5g/8g windows (D) DID add value beyond season-to-date alone** — the clearest
  positive ablation result: win AUC improved from 0.607 (C) to 0.622 (D), and win Brier/
  log-loss also improved. Recent form carries real information beyond a full-season
  average.
- **Personnel continuity (E) added a further small improvement** on top of D (AUC 0.622 →
  0.624, Brier 0.2380 → 0.2379) — a genuine but marginal effect, consistent with personnel
  continuity being a second-order signal.
- **The full feature set (F) performed WORSE than D and E**, on both margin and win metrics,
  and CORE (F minus personnel/prev-season) was worse still. With only ~3,500 training rows
  and 103 features, this is a plausible, honest case of the ~40 additional situational/
  small-sample features (third down, red zone, explosive rate, special teams, turnovers)
  adding noise faster than signal on top of an already-solid E-tier model. This is
  precisely why the ablation ladder was required — "more features" was not free, and F is
  not simply the best model by virtue of using everything.
- **Competitive-play (`_comp`) variants**: included within D/E/F (not isolated as their own
  ablation tier, to keep the ladder to six tiers as specified) — their Ridge coefficients
  are of comparable magnitude to the all-play equivalents (see coefficient review, below),
  neither dominating nor negligible. A dedicated all-play-vs-competitive-play ablation is a
  reasonable Phase 4 follow-up if this distinction becomes decision-relevant.
- **Elo beat every linear/boosted model on win probability** (Brier 0.2333, AUC 0.641) and
  matched or beat them on margin (MAE 10.53, best of any model) — on this single validation
  season. This is a genuinely useful, if slightly humbling, result: the simplest, most
  transparent baseline was the best performer here. It is reported as such, not
  downplayed — "complexity must earn its place," and on 2023 alone, LightGBM's 103-feature
  nonlinear model did not clearly earn it over Elo or over the leaner E-tier linear model.
  This is one validation season; Phase 4's larger walk-forward sample will show whether
  this holds.

## CORE vs. FULL (historical-availability comparison)

CORE (97 features, broadly available since ~1999) slightly underperformed FULL/F (103
features, personnel from 2013+ and previous-season from 2011+) on total-points MAE (11.15
vs. 11.24 — CORE actually *better* here) but was worse on margin and win metrics. Given F
itself underperformed the leaner D/E tiers (above), this comparison mainly reinforces the
same finding: raw feature count is not the deciding factor: **which** features are added
matters more than **how many**.

## Coefficient / importance review

Ridge (F_full) top-|coefficient| features: `diff_def_pass_epa_allowed_5g`,
`diff_def_epa_pp_allowed_5g`, `diff_def_success_rate_allowed_3g`,
`diff_off_pass_success_rate_season`, `diff_off_success_rate_comp_season` — all genuine
football efficiency differentials. LightGBM (F_full) top-importance features:
`diff_prev_season_def_epa_pp_allowed`, `diff_prev_season_off_success_rate`,
`diff_off_pass_epa_dropback_season`, `diff_opp_off_epa_faced_season`,
`diff_qb_consecutive_starts` — again, all football-plausible. **No identity-like field
(team ID, game ID, timestamp) appears among either model's influential features** — checked
directly, not assumed; full tables in `data/reports/phase3_ridge_margin_coefficients_F_full.csv`
and `data/reports/phase3_lgbm_margin_importance_F_full.csv`.

**One genuinely surprising, worth-flagging pattern**: `missingindicator_diff_punt_yards_avg_season`
carries one of the largest Ridge coefficients (+2.71) in the F_full margin model. This is
not identity leakage or a bug — it's the imputer's *missingness flag* for that feature
carrying real signal: a team punting rarely enough this season to have too small a punt
sample is plausibly a team with a strong offense that doesn't need to punt. Flagged here as
an interesting, plausible finding, not dismissed — but also not leaned on further this
phase (correlation is not causation, and this is a single validation season).

Coefficients are not naively over-interpreted given known feature correlation (the 3g/5g/8g
window variants are correlated with each other and with season-to-date by construction) —
see multicollinearity, next.

## Multicollinearity / redundancy

Correlated variants (3g/5g/8g/season-to-date) were kept, per instruction — not removed.
Ridge's L2 regularization is exactly the tool for this: it shrinks and shares weight across
correlated features rather than requiring the redundancy be resolved beforehand. No
numerical instability was observed (no NaN/inf coefficients, no fit failures) across any of
the 21 linear-model fits. A full pairwise correlation matrix across the 103 F_full features
was not separately generated as a standalone diagnostic file this phase (a reasonable,
disclosed scope limitation) — the practical evidence that regularization is handling
collinearity adequately is the absence of any numerical pathology across every fit.

## Missing data / historical coverage

Handled by the Pipeline's `SimpleImputer(strategy="median", add_indicator=True)` — fit on
training data only, and missingness indicators preserved as genuine model inputs. LightGBM
uses its native missing-value handling (no imputation at all). Nothing was filled with zero.
See CORE vs. FULL, above, for the explicit broad-vs-full-availability comparison.

## QB features: unchanged from Phase 2, explicitly re-confirmed

**No new hindsight-derived QB information was added during modeling.** Phase 3 consumes
`qb_*` columns exactly as Phase 2 produced them — the primary-QB-of-the-previous-game
methodology (`docs/PHASE2_FEATURE_REPORT.md#qb-features-and-the-leakage-rule`) is untouched.
No model in this phase infers or uses the target game's own starting QB.

## Predictive uncertainty (Phase 3 beginning, not finishing, the architecture)

Genuinely out-of-sample residuals: Ridge (F_full) fit on 2010–2020 only, residuals computed
on 2021–2022 (still development, never validation/holdout) — 569 games.

**Margin**: residual mean ≈ **−0.015** (essentially unbiased), residual std ≈ **13.06**.
A Normal(0, 13.06) approximation calibrates reasonably well on this out-of-sample check:

| Nominal coverage | Normal-approx observed | Empirical-quantile observed |
|---|---|---|
| 50% | 54.3% | 50.1% |
| 80% | 80.7% | 80.0% |
| 90% | 89.3% | 89.8% |
| 95% | 93.7% | 94.7% |

Mild heteroskedasticity detected: residual std rises from ~12.6 (smallest-magnitude
predictions) to ~14.4 (largest-magnitude predictions) — real, small, not dramatic.

**Total**: residual mean ≈ **−1.17** (residual = actual − predicted, so a negative mean
means predicted > actual — a real, non-trivial bias where the model OVER-predicts total
points by about a point on this 2021–2022 check), residual std ≈ **14.26**. Interval
coverage is close to nominal but slightly conservative at the 50%/80% levels. The bias is
flagged as a genuine finding for Phase 4 to investigate (candidate explanation: the 2021
rule change to a 17-game season and general scoring-environment drift across
2010–2020-fit vs. 2021–2022-observed data — not confirmed, just plausible).

**No fixed/arbitrary standard deviation was assumed** — both the Normal σ and the empirical
quantiles above are measured, not invented. **No cover/over probability against a
sportsbook spread was computed** — that is explicitly deferred to Phase 4, once these
diagnostics are validated at production scale on true out-of-sample (walk-forward) data.

## Calibration (win probability, F_full)

| Bucket | n | Mean predicted (logistic) | Observed (logistic) | Mean predicted (LightGBM) | Observed (LightGBM) |
|---|---|---|---|---|---|
| 0.00–0.40 | 61 | 0.306 | 0.508 | 0.330 | 0.459 |
| 0.40–0.45 | 18/23 | 0.424 | 0.389 | 0.425 | 0.522 |
| 0.45–0.50 | 30/24 | 0.479 | 0.400 | 0.474 | 0.500 |
| 0.50–0.55 | 29/30 | 0.527 | 0.586 | 0.525 | 0.600 |
| 0.55–0.60 | 24/30 | 0.574 | 0.625 | 0.573 | 0.567 |
| 0.60–0.65 | 27/25 | 0.624 | 0.481 | 0.626 | 0.520 |
| 0.65–0.70 | 24/22 | 0.672 | 0.667 | 0.671 | 0.636 |
| 0.70–1.00 | 72/70 | 0.785 | 0.694 | 0.789 | 0.671 |

Full tables in `data/reports/phase3_calibration_{logistic,lgbm}.csv`. Bins are small (18–72
games) and **not over-interpreted** — the 0.60–0.65 bucket underperforms its predicted rate
for both models on this one validation season, which given n≈25–27 is plausibly noise, not
necessarily a systematic miscalibration; it would need a larger sample (Phase 4) to confirm.
The 0.00–0.40 bucket for the logistic model shows observed 50.8% against a mean prediction
of 30.6% — a notable gap, also worth watching in Phase 4's larger sample.

## Segment breakdown (Ridge, F_full, margin)

| Segment | n | MAE | RMSE | Bias | Corr |
|---|---|---|---|---|---|
| REG | 272 | 10.73 | 13.83 | −0.73 | 0.303 |
| POST | 13 | 11.80 | 15.32 | −4.37 | −0.174 |
| Weeks 1–4 | 64 | 11.94 | 15.43 | +0.29 | 0.360 |
| Weeks 5–10 | 86 | 10.01 | 13.09 | −1.28 | 0.238 |
| Weeks 11+ | 122 | 10.60 | 13.43 | −0.87 | 0.317 |
| POST (week bucket) | 13 | — | — | — | — |

**Early-season degradation confirmed**: Weeks 1–4 has the worst MAE (11.94) of any REG
segment, consistent with Phase 2's documentation that early-season rolling/season-to-date
samples are small. **Postseason (n=13) shows a negative correlation (−0.17) and the largest
bias (−4.37)** — with only 13 games, this is far too small a sample to draw a real
conclusion from (a single upset-heavy playoff bracket can flip a correlation sign by
chance), and it is reported with that explicit caveat rather than as evidence the model
fails in the postseason.

## Known weaknesses

- QB-feature ablation (C) did not show a clear win-metric improvement over B on this single
  validation season — inconclusive, not "QB features don't matter."
- The full feature set (F) underperformed the leaner D/E tiers — more features did not mean
  a better model at this training-data scale.
- Postseason sample size (13 games) is too small for a reliable read this phase.
- Total-points model over-predicts total points by ~1 point on the 2021–2022 out-of-sample
  uncertainty check (residual mean ≈ −1.17, actual − predicted) — flagged, not resolved.
- Elo's selected K-factor (30) sits at the edge of the searched grid.
- No standalone pairwise-correlation-matrix file was produced (see "Multicollinearity,"
  above) — regularization's absence of numerical pathology was used as the practical check
  instead.
- Calibration bins are small; the 0.60–0.65 and 0.00–0.40 buckets show the largest gaps and
  should be watched, not treated as confirmed miscalibration, until Phase 4's larger sample.

## Anything surprising discovered

- A missingness indicator (`punt_yards_avg_season` being null, i.e. very few punts that
  season) carries real predictive signal in the linear margin model — plausible (dominant
  offenses punt less) and a reminder that "missing" itself can be informative, exactly why
  Phase 2/3 preserve missingness indicators instead of just imputing silently.
- Elo — the cheapest, most transparent model in this report — was the single best performer
  on win probability and tied-or-best on margin, on this validation season. A useful,
  humbling result to carry into Phase 4's larger evaluation.

## Recommendation for Phase 4

1. Carry forward **Elo**, **Ridge margin/total (E_add_personnel tier)**, **logistic win
   probability (E_add_personnel tier)**, and **LightGBM (F_full)** as the model roster to
   walk-forward backtest at scale — the single-season 2023 read here is suggestive, not
   definitive, and Phase 4's larger sample (plus, eventually, the sealed 2024–2025) is what
   should decide which family is actually best.
2. Re-run the ablation ladder inside Phase 4's walk-forward framework before concluding
   anything definitive about QB features or the full feature set's underperformance — one
   validation season is a real result but a small one.
3. Investigate the total-points model's out-of-sample bias before relying on its predictive
   distribution for anything downstream.
4. Do not unseal 2024–2025 until Phase 4's methodology (walk-forward cadence, retraining
   policy) is fully specified — per `docs/BACKTESTING_RULES.md`.
5. The uncertainty diagnostics here (residual std, empirical quantiles, interval
   calibration) are a Phase 3 starting point, not a finished calibration — Phase 4 must
   validate them properly on genuine walk-forward out-of-sample data before they're used to
   compute any cover/over probability.

## Reproducing this report

```bash
python -m nfl_predict.models.train
python -m pytest tests/models/
```
